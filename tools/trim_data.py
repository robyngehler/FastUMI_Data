#!/usr/bin/env python3
"""
Trim FastUMI batches when only these files are available per task directory:
- camera/temp_video_0.mp4, temp_video_1.mp4, temp_video_2.mp4
- episode_0.hdf5, episode_1.hdf5, episode_2.hdf5
- csv/frame_timestamps.csv
- states.csv
- optionally csv/temp_video_timestamps.csv and csv/temp_trajectory.csv (usually only useful for the last episode)

Design goals:
- Robustly trim each episode using its own local MP4 and the known fixed local clip duration (~20 s).
- Do NOT require shared temp_video_timestamps.csv / temp_trajectory.csv to contain all episodes.
- Create usable trimmed outputs for training/debugging even when exact original 60 Hz / 200 Hz absolute streams
  are unavailable for early episodes.

Important limitation:
- If only one shared temp_trajectory.csv exists and it only covers the last recorded episode, then exact 200 Hz
  trajectory reconstruction for earlier episodes is impossible. In that case this script writes a fallback
  trajectory.csv derived from states.csv (20 Hz) and records that fact in the manifest.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import h5py
import numpy as np
import pandas as pd

RANGE_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*-\s*([0-9]+(?:\.[0-9]+)?)\s*$")

VIDEO_CANDIDATES = [
    "camera/temp_video_{episode}.mp4",
    "camera/video_{episode}.mp4",
    "camera/episode_{episode}.mp4",
]
HDF5_CANDIDATES = [
    "episode_{episode}.hdf5",
    "episode_{episode:06d}.hdf5",
]

@dataclass
class TrimWindow:
    label: str
    relative_start_sec: float
    relative_end_sec: float
    duration_sec: float

@dataclass
class TrimResult:
    label: str
    out_dir: str
    video_frames: int
    traj_rows: int
    downsampled_frames: int
    has_hdf5: bool
    has_states: bool
    relative_start_sec: float
    relative_end_sec: float

@dataclass
class BatchTaskPlan:
    task_name: str
    episode_specs: list[str]

@dataclass
class BatchEpisodeRequest:
    task_name: str
    task_dir: str
    episode_index: int
    trim_spec: str
    skip: bool
    start_sec: Optional[float]
    end_sec: Optional[float]

@dataclass
class BatchEpisodeOutcome:
    task_name: str
    task_dir: str
    episode_index: int
    trim_spec: str
    skip: bool
    status: str
    message: Optional[str]
    summary_path: Optional[str]
    result: Optional[dict[str, Any]]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trim FastUMI batches from per-episode MP4 + HDF5 + frame_timestamps.csv.")
    p.add_argument("--task-dir", type=Path, default=None)
    p.add_argument("--episode-index", type=int, default=None)
    p.add_argument("--start-sec", type=float, default=None)
    p.add_argument("--end-sec", type=float, default=None)

    p.add_argument("--plan-file", type=Path, default=None)
    p.add_argument("--tasks-root", type=Path, default=None)
    p.add_argument("--jobs", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    p.add_argument("--fail-fast", action="store_true")

    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--codec", type=str, default="mp4v")
    p.add_argument("--video-fps", type=float, default=None,
                  help="Override FPS used for synthetic local timestamps. If omitted, infer from video.")
    p.add_argument("--video-duration-sec", type=float, default=20.0,
                  help="Nominal per-episode local video length in seconds. Used for sanity checks only.")
    p.add_argument("--copy-original-manifest", action="store_true")
    return p.parse_args()

def validate_mode(args: argparse.Namespace) -> str:
    single_fields = [args.task_dir, args.episode_index, args.start_sec, args.end_sec]
    single_mode = all(v is not None for v in single_fields)
    batch_mode = args.plan_file is not None
    if single_mode and batch_mode:
        raise ValueError("Choose either single mode or batch mode, not both.")
    if not single_mode and not batch_mode:
        raise ValueError("Provide either single trim args or --plan-file/--tasks-root.")
    return "batch" if batch_mode else "single"

def parse_trim_spec(spec: str) -> tuple[bool, Optional[float], Optional[float]]:
    raw = spec.strip()
    if not raw:
        raise ValueError("Empty trim specification")
    if raw.lower() == "x":
        return True, None, None
    m = RANGE_RE.match(raw)
    if not m:
        raise ValueError(f"Invalid trim spec '{spec}'. Expected 'start-end' or 'x'.")
    start_sec = float(m.group(1))
    end_sec = float(m.group(2))
    if end_sec <= start_sec:
        raise ValueError(f"Invalid trim spec '{spec}': end must be larger than start.")
    return False, start_sec, end_sec

def parse_plan_file(plan_file: Path) -> list[BatchTaskPlan]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for raw_line in plan_file.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                blocks.append(current)
                current = []
            continue
        if line.startswith("#"):
            continue
        current.append(line)
    if current:
        blocks.append(current)

    plans: list[BatchTaskPlan] = []
    for idx, block in enumerate(blocks):
        if len(block) < 2:
            raise ValueError(f"Invalid block #{idx+1}: need task name plus one trim line.")
        task_name = block[0]
        specs = block[1:]
        for spec in specs:
            parse_trim_spec(spec)
        plans.append(BatchTaskPlan(task_name=task_name, episode_specs=specs))
    return plans

def build_batch_requests(tasks_root: Path, plans: list[BatchTaskPlan]) -> list[BatchEpisodeRequest]:
    reqs: list[BatchEpisodeRequest] = []
    for plan in plans:
        task_dir = (tasks_root / plan.task_name).resolve()
        for ep_idx, spec in enumerate(plan.episode_specs):
            skip, start_sec, end_sec = parse_trim_spec(spec)
            reqs.append(BatchEpisodeRequest(
                task_name=plan.task_name,
                task_dir=str(task_dir),
                episode_index=ep_idx,
                trim_spec=spec,
                skip=skip,
                start_sec=start_sec,
                end_sec=end_sec,
            ))
    return reqs

def resolve_video(task_dir: Path, episode_index: int) -> Path:
    for pat in VIDEO_CANDIDATES:
        p = task_dir / pat.format(episode=episode_index)
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find video for {task_dir}, episode {episode_index}")

def resolve_hdf5(task_dir: Path, episode_index: int) -> Optional[Path]:
    for pat in HDF5_CANDIDATES:
        p = task_dir / pat.format(episode=episode_index)
        if p.exists():
            return p
    return None

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def detect_video_meta(video_path: Path, fps_override: Optional[float]) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if fps_override is not None and fps_override > 0:
        fps = float(fps_override)
    if fps <= 1e-3:
        fps = 60.0
    return fps, frame_count, width, height

def get_episode_start_timestamp(frame_ts_df: pd.DataFrame, episode_index: int) -> float:
    if "Episode Index" in frame_ts_df.columns:
        matches = frame_ts_df[frame_ts_df["Episode Index"].astype(int) == int(episode_index)]
        if matches.empty:
            raise IndexError(f"Episode {episode_index} not found in frame_timestamps.csv")
        return float(matches.iloc[0]["Timestamp"])
    if episode_index >= len(frame_ts_df):
        raise IndexError(f"Episode {episode_index} out of range for frame_timestamps.csv")
    ts_col = "Timestamp" if "Timestamp" in frame_ts_df.columns else frame_ts_df.columns[-1]
    return float(frame_ts_df.iloc[episode_index][ts_col])

def build_local_video_timestamps(frame_count: int, fps: float, start_ts: float) -> pd.DataFrame:
    t = start_ts + np.arange(frame_count, dtype=np.float64) / float(fps)
    return pd.DataFrame({"Frame Index": np.arange(frame_count, dtype=np.int64), "Timestamp": t})

def get_hdf5_length(hdf5_path: Path) -> int:
    with h5py.File(hdf5_path, "r") as f:
        lengths = []
        if "observations" in f and "qpos" in f["observations"]:
            lengths.append(int(f["observations"]["qpos"].shape[0]))
        if "observations" in f and "images" in f["observations"]:
            for cam in f["observations"]["images"].keys():
                lengths.append(int(f["observations"]["images"][cam].shape[0]))
                break
        if "action" in f:
            lengths.append(int(f["action"].shape[0]))
    if not lengths:
        raise ValueError(f"Could not infer length from HDF5 {hdf5_path}")
    return min(lengths)

def trim_video_frames(video_path: Path, out_path: Path, kept_frame_indices: np.ndarray, fps: float, codec: str) -> int:
    if kept_frame_indices.size == 0:
        raise ValueError("No frames selected for trimmed video")
    start_frame = int(kept_frame_indices[0])
    end_frame = int(kept_frame_indices[-1])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {out_path}")

    keep = set(int(x) for x in kept_frame_indices.tolist())
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_no = start_frame
    written = 0
    while frame_no <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_no in keep:
            writer.write(frame)
            written += 1
        frame_no += 1
    cap.release()
    writer.release()
    if written != kept_frame_indices.size:
        raise RuntimeError(f"Expected {kept_frame_indices.size} video frames but wrote {written}")
    return written

def trim_hdf5(in_hdf5: Path, out_hdf5: Path, indices: np.ndarray) -> None:
    with h5py.File(in_hdf5, "r") as src, h5py.File(out_hdf5, "w") as dst:
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        obs_src = src["observations"]
        obs_dst = dst.create_group("observations")
        img_dst = obs_dst.create_group("images")
        for cam in obs_src["images"].keys():
            img_dst.create_dataset(cam, data=obs_src["images"][cam][indices], compression="gzip", compression_opts=4)
        obs_dst.create_dataset("qpos", data=obs_src["qpos"][indices])
        if "action" in src:
            dst.create_dataset("action", data=src["action"][indices])

def maybe_extract_states_episode(states_df: pd.DataFrame, frame_ts_df: pd.DataFrame, episode_index: int) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    if "Start Time" in states_df.columns:
        uniq = []
        for v in states_df["Start Time"].astype(float).tolist():
            if v not in uniq:
                uniq.append(v)
        if len(uniq) >= episode_index + 1:
            target = uniq[episode_index]
            grp = states_df[np.isclose(states_df["Start Time"].astype(float).to_numpy(), target)].copy().reset_index(drop=True)
            if not grp.empty:
                notes.append("Episode states selected by unique Start Time group.")
                return grp, notes

    start_ts = get_episode_start_timestamp(frame_ts_df, episode_index)
    if "Episode Index" in frame_ts_df.columns:
        matches = frame_ts_df[frame_ts_df["Episode Index"].astype(int) == int(episode_index)]
        row = int(matches.index[0])
    else:
        row = int(episode_index)
    end_ts = None
    if row + 1 < len(frame_ts_df):
        end_ts = float(frame_ts_df.iloc[row + 1]["Timestamp"])
    if end_ts is None:
        end_ts = start_ts + 25.0
    grp = states_df[(states_df["Frame Timestamp"].astype(float) >= start_ts) &
                    (states_df["Frame Timestamp"].astype(float) < end_ts)].copy().reset_index(drop=True)
    if grp.empty:
        raise ValueError(f"Could not isolate states for episode {episode_index}")
    notes.append("Episode states selected by Frame Timestamp range from frame_timestamps.csv.")
    return grp, notes

def build_fallback_trajectory_from_states(states_ep: pd.DataFrame) -> pd.DataFrame:
    cols = ["Trajectory Timestamp", "Pos X", "Pos Y", "Pos Z", "Q_X", "Q_Y", "Q_Z", "Q_W"]
    missing = [c for c in cols if c not in states_ep.columns]
    if missing:
        raise ValueError(f"Cannot build fallback trajectory from states; missing columns {missing}")
    out = states_ep[cols].copy().reset_index(drop=True)
    out = out.rename(columns={"Trajectory Timestamp": "Timestamp"})
    return out

def maybe_use_shared_last_episode_csv(task_dir: Path, episode_index: int) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], list[str]]:
    notes: list[str] = []
    video_ts_path = task_dir / "csv" / "temp_video_timestamps.csv"
    traj_path = task_dir / "csv" / "temp_trajectory.csv"
    if episode_index != 2:
        return None, None, notes
    vdf = None
    tdf = None
    if video_ts_path.exists():
        try:
            tmp = load_csv(video_ts_path)
            if {"Frame Index", "Timestamp"}.issubset(tmp.columns):
                vdf = tmp
                notes.append("Used shared temp_video_timestamps.csv for episode 2.")
        except Exception:
            pass
    if traj_path.exists():
        try:
            tmp = load_csv(traj_path)
            if "Timestamp" in tmp.columns:
                tdf = tmp
                notes.append("Used shared temp_trajectory.csv for episode 2.")
        except Exception:
            pass
    return vdf, tdf, notes

def trim_episode(
    task_dir: Path,
    episode_index: int,
    start_sec: float,
    end_sec: float,
    output_root: Optional[Path],
    codec: str,
    fps_override: Optional[float],
    nominal_duration_sec: float,
    copy_original_manifest: bool,
) -> dict[str, Any]:
    if end_sec <= start_sec:
        raise ValueError("end-sec must be larger than start-sec")

    task_dir = task_dir.resolve()
    video_path = resolve_video(task_dir, episode_index)
    hdf5_path = resolve_hdf5(task_dir, episode_index)
    frame_ts_path = task_dir / "csv" / "frame_timestamps.csv"
    states_path = task_dir / "states.csv"
    if not frame_ts_path.exists():
        raise FileNotFoundError(f"Missing {frame_ts_path}")
    if not states_path.exists():
        raise FileNotFoundError(f"Missing {states_path}")

    frame_ts_df = load_csv(frame_ts_path)
    states_df = load_csv(states_path)

    fps, frame_count, _, _ = detect_video_meta(video_path, fps_override)
    start_ts = get_episode_start_timestamp(frame_ts_df, episode_index)
    video_ts_df = build_local_video_timestamps(frame_count, fps, start_ts)

    notes = []
    approx_duration = frame_count / fps if fps > 0 else 0.0
    notes.append(f"Synthesized local video timestamps from per-episode MP4 at {fps:.6f} FPS.")
    if abs(approx_duration - nominal_duration_sec) > 1.0:
        notes.append(f"Warning: local video duration is {approx_duration:.3f}s, not close to nominal {nominal_duration_sec:.3f}s.")

    states_ep, state_notes = maybe_extract_states_episode(states_df, frame_ts_df, episode_index)
    notes.extend(state_notes)

    shared_vdf, shared_tdf, shared_notes = maybe_use_shared_last_episode_csv(task_dir, episode_index)
    notes.extend(shared_notes)

    if shared_vdf is not None:
        video_ts_df = shared_vdf.copy().reset_index(drop=True)
        notes.append("For episode 2, raw shared video timestamps were used instead of synthetic local timestamps.")
    if shared_tdf is not None:
        traj_df = shared_tdf.copy().reset_index(drop=True)
    else:
        traj_df = build_fallback_trajectory_from_states(states_ep)
        notes.append("trajectory.csv is a 20 Hz fallback reconstructed from states.csv because no episode-specific raw trajectory was available.")

    base_window = TrimWindow(
        label="base_trim",
        relative_start_sec=float(start_sec),
        relative_end_sec=float(end_sec),
        duration_sec=float(end_sec - start_sec),
    )

    out_root = (output_root or (task_dir / "trimmed")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_dir = out_root / f"episode_{episode_index}_{base_window.label}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "camera").mkdir(parents=True, exist_ok=True)
    (out_dir / "csv").mkdir(parents=True, exist_ok=True)

    abs_start = float(video_ts_df["Timestamp"].iloc[0]) + float(start_sec)
    abs_end = float(video_ts_df["Timestamp"].iloc[0]) + float(end_sec)

    vmask = (video_ts_df["Timestamp"].astype(float).to_numpy() >= abs_start) & (video_ts_df["Timestamp"].astype(float).to_numpy() <= abs_end)
    kept_video_ts = video_ts_df.loc[vmask].copy().reset_index(drop=True)
    if kept_video_ts.empty:
        raise ValueError("Requested window selects no video frames")
    original_frame_indices = kept_video_ts["Frame Index"].astype(int).to_numpy()
    kept_video_ts.insert(1, "Original Frame Index", original_frame_indices)
    kept_video_ts["Frame Index"] = np.arange(len(kept_video_ts), dtype=np.int64)
    kept_video_ts["Timestamp"] = kept_video_ts["Timestamp"].astype(float) - abs_start
    written_frames = trim_video_frames(video_path, out_dir / "camera" / f"trimmed_episode_{episode_index}.mp4",
                                       original_frame_indices, fps, codec)
    kept_video_ts.to_csv(out_dir / "csv" / "video_timestamps.csv", index=False)

    traj_mask = (traj_df["Timestamp"].astype(float).to_numpy() >= abs_start) & (traj_df["Timestamp"].astype(float).to_numpy() <= abs_end)
    trimmed_traj = traj_df.loc[traj_mask].copy().reset_index(drop=True)
    if trimmed_traj.empty:
        raise ValueError("Requested window selects no trajectory rows")
    trimmed_traj["Timestamp"] = trimmed_traj["Timestamp"].astype(float) - abs_start
    trimmed_traj.to_csv(out_dir / "csv" / "trajectory.csv", index=False)

    trimmed_states = states_ep[(states_ep["Frame Timestamp"].astype(float) >= abs_start) &
                               (states_ep["Frame Timestamp"].astype(float) <= abs_end)].copy().reset_index(drop=True)
    has_states = False
    if not trimmed_states.empty:
        trimmed_states["Frame Timestamp"] = trimmed_states["Frame Timestamp"].astype(float) - abs_start
        if "Trajectory Timestamp" in trimmed_states.columns:
            trimmed_states["Trajectory Timestamp"] = trimmed_states["Trajectory Timestamp"].astype(float) - abs_start
        trimmed_states.to_csv(out_dir / "states.csv", index=False)
        has_states = True

    has_hdf5 = False
    ds_count = 0
    if hdf5_path is not None and hdf5_path.exists():
        hdf5_len = get_hdf5_length(hdf5_path)
        ds_times_abs = start_ts + np.arange(hdf5_len, dtype=np.float64) / 20.0
        ds_idx = np.where((ds_times_abs >= abs_start) & (ds_times_abs <= abs_end))[0]
        if ds_idx.size == 0:
            raise ValueError("Requested window selects no 20 Hz HDF5 frames")
        ds_df = pd.DataFrame({
            "Frame Index": np.arange(len(ds_idx), dtype=np.int64),
            "Original Downsampled Index": ds_idx.astype(np.int64),
            "Timestamp": ds_times_abs[ds_idx] - abs_start,
        })
        ds_df.to_csv(out_dir / "csv" / "downsampled_video_timestamps_20hz.csv", index=False)
        trim_hdf5(hdf5_path, out_dir / f"episode_{episode_index}.hdf5", ds_idx.astype(np.int64))
        has_hdf5 = True
        ds_count = int(ds_idx.size)
        notes.append("20 Hz timestamps were synthesized from HDF5 length at 20 Hz, anchored to frame_timestamps.csv start.")
    else:
        notes.append("No HDF5 found for this episode.")

    manifest = {
        "task_dir": str(task_dir),
        "episode_index": episode_index,
        "window": asdict(base_window),
        "source_video_fps": fps,
        "codec": codec,
        "notes": notes,
    }
    if copy_original_manifest:
        manifest["sources"] = {
            "video": str(video_path),
            "hdf5": str(hdf5_path) if hdf5_path else None,
            "frame_timestamps_csv": str(frame_ts_path),
            "states_csv": str(states_path),
            "shared_temp_video_timestamps_csv": str(task_dir / "csv" / "temp_video_timestamps.csv") if (task_dir / "csv" / "temp_video_timestamps.csv").exists() else None,
            "shared_temp_trajectory_csv": str(task_dir / "csv" / "temp_trajectory.csv") if (task_dir / "csv" / "temp_trajectory.csv").exists() else None,
        }
    (out_dir / "trim_manifest.json").write_text(json.dumps(manifest, indent=2))

    summary = {
        "task_dir": str(task_dir),
        "episode_index": episode_index,
        "input_files": manifest.get("sources", {
            "video": str(video_path),
            "hdf5": str(hdf5_path) if hdf5_path else None,
            "frame_timestamps_csv": str(frame_ts_path),
            "states_csv": str(states_path),
        }),
        "notes": notes,
        "results": [asdict(TrimResult(
            label=base_window.label,
            out_dir=str(out_dir),
            video_frames=written_frames,
            traj_rows=len(trimmed_traj),
            downsampled_frames=ds_count,
            has_hdf5=has_hdf5,
            has_states=has_states,
            relative_start_sec=base_window.relative_start_sec,
            relative_end_sec=base_window.relative_end_sec,
        ))],
        "recommendations": [
            "Use the base trim for your essential episode window.",
            "For episode 0/1, trajectory.csv may be only a 20 Hz fallback if no per-episode raw trajectory exists.",
            "Check trim_manifest.json before downstream conversion if exact raw timestamp provenance matters.",
        ],
    }
    (out_root / f"episode_{episode_index}_trim_summary.json").write_text(json.dumps(summary, indent=2))
    return summary

def process_batch_episode(payload: dict[str, Any]) -> dict[str, Any]:
    req = BatchEpisodeRequest(**payload["request"])
    output_root_base = Path(payload["output_root_base"]).resolve() if payload["output_root_base"] is not None else None
    if req.skip:
        return asdict(BatchEpisodeOutcome(
            task_name=req.task_name,
            task_dir=req.task_dir,
            episode_index=req.episode_index,
            trim_spec=req.trim_spec,
            skip=True,
            status="skipped",
            message="Skipped because trim spec is 'x'.",
            summary_path=None,
            result=None,
        ))
    task_dir = Path(req.task_dir).resolve()
    output_root = (output_root_base / req.task_name) if output_root_base is not None else (task_dir / "trimmed")
    try:
        summary = trim_episode(
            task_dir=task_dir,
            episode_index=req.episode_index,
            start_sec=float(req.start_sec),
            end_sec=float(req.end_sec),
            output_root=output_root,
            codec=str(payload["codec"]),
            fps_override=payload["video_fps"],
            nominal_duration_sec=float(payload["video_duration_sec"]),
            copy_original_manifest=bool(payload["copy_original_manifest"]),
        )
        summary_path = str(output_root / f"episode_{req.episode_index}_trim_summary.json")
        return asdict(BatchEpisodeOutcome(
            task_name=req.task_name,
            task_dir=req.task_dir,
            episode_index=req.episode_index,
            trim_spec=req.trim_spec,
            skip=False,
            status="ok",
            message=None,
            summary_path=summary_path,
            result=summary,
        ))
    except Exception as exc:
        return asdict(BatchEpisodeOutcome(
            task_name=req.task_name,
            task_dir=req.task_dir,
            episode_index=req.episode_index,
            trim_spec=req.trim_spec,
            skip=False,
            status="error",
            message=str(exc),
            summary_path=None,
            result=None,
        ))

def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    if args.tasks_root is None:
        raise ValueError("Batch mode requires --tasks-root")
    tasks_root = args.tasks_root.resolve()
    plans = parse_plan_file(args.plan_file.resolve())
    requests = build_batch_requests(tasks_root, plans)
    output_root_base = args.output_root.resolve() if args.output_root is not None else None
    if output_root_base is not None:
        output_root_base.mkdir(parents=True, exist_ok=True)

    payloads = [{
        "request": asdict(req),
        "output_root_base": str(output_root_base) if output_root_base is not None else None,
        "codec": args.codec,
        "video_fps": args.video_fps,
        "video_duration_sec": args.video_duration_sec,
        "copy_original_manifest": args.copy_original_manifest,
    } for req in requests]

    outcomes: list[dict[str, Any]] = []
    jobs = max(1, int(args.jobs))
    if jobs == 1:
        for payload in payloads:
            outcome = process_batch_episode(payload)
            outcomes.append(outcome)
            if args.fail_fast and outcome["status"] == "error":
                break
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futures = [ex.submit(process_batch_episode, payload) for payload in payloads]
            for fut in as_completed(futures):
                outcome = fut.result()
                outcomes.append(outcome)
                if args.fail_fast and outcome["status"] == "error":
                    for f in futures:
                        f.cancel()
                    break

    outcomes.sort(key=lambda item: (item["task_name"], item["episode_index"]))
    num_ok = sum(1 for x in outcomes if x["status"] == "ok")
    num_skipped = sum(1 for x in outcomes if x["status"] == "skipped")
    num_error = sum(1 for x in outcomes if x["status"] == "error")

    summary = {
        "plan_file": str(args.plan_file.resolve()),
        "tasks_root": str(tasks_root),
        "output_root_base": str(output_root_base) if output_root_base is not None else None,
        "submitted": len(requests),
        "jobs": jobs,
        "num_ok": num_ok,
        "num_skipped": num_skipped,
        "num_error": num_error,
        "results": outcomes,
        "notes": [
            "Each task block maps trim lines to episode indices starting at 0.",
            "This script synthesizes local timestamps from per-episode videos and HDF5 lengths.",
            "For early episodes, trajectory.csv may be a 20 Hz fallback derived from states.csv.",
        ],
    }
    summary_path = (output_root_base / "trim_batch_summary.json") if output_root_base is not None else (tasks_root / "trim_batch_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    summary["summary_path"] = str(summary_path)
    return summary

def main() -> None:
    args = parse_args()
    mode = validate_mode(args)
    if mode == "batch":
        print(json.dumps(run_batch(args), indent=2))
    else:
        print(json.dumps(trim_episode(
            task_dir=args.task_dir,
            episode_index=int(args.episode_index),
            start_sec=float(args.start_sec),
            end_sec=float(args.end_sec),
            output_root=args.output_root,
            codec=args.codec,
            fps_override=args.video_fps,
            nominal_duration_sec=float(args.video_duration_sec),
            copy_original_manifest=args.copy_original_manifest,
        ), indent=2))

if __name__ == "__main__":
    main()
