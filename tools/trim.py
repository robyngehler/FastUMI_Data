#!/usr/bin/env python3
"""
Trim FastUMI-style raw episodes consistently across video, timestamps, trajectory,
optional HDF5, and optional states.csv.

Adapted for both old and new data layouts:
- prefers episode_events.csv + per-episode files when available
- falls back to old shared temp_* streams only when necessary
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import h5py
import numpy as np
import pandas as pd

VIDEO_TS_CANDIDATES = [
    "csv/temp_video_timestamps_{episode}.csv",
    "csv/video_timestamps_{episode}.csv",
    "csv/video_timestamps_episode_{episode}.csv",
    "csv/temp_video_timestamps.csv",
]
TRAJ_CANDIDATES = [
    "csv/temp_trajectory_{episode}.csv",
    "csv/trajectory_{episode}.csv",
    "csv/trajectory_episode_{episode}.csv",
    "csv/temp_trajectory.csv",
]
EVENT_CANDIDATES = ["csv/episode_events.csv", "episode_events.csv"]
VIDEO_CANDIDATES = [
    "camera/temp_video_{episode}.mp4",
    "camera/video_{episode}.mp4",
    "camera/episode_{episode}.mp4",
]
HDF5_CANDIDATES = ["episode_{episode}.hdf5", "episode_{episode:06d}.hdf5"]
RANGE_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*-\s*([0-9]+(?:\.[0-9]+)?)\s*$")

SHARED_VIDEO_TS_FILENAMES = {"temp_video_timestamps.csv"}
SHARED_TRAJ_FILENAMES = {"temp_trajectory.csv"}
FRAME_TS_INDEX_COLUMNS = ["Episode Index", "episode_index", "Episode", "episode"]
FRAME_TS_TIMESTAMP_COLUMNS = ["Timestamp", "Frame Timestamp", "Start Time", "timestamp"]


@dataclass
class TrimWindow:
    label: str
    absolute_start_ts: float
    absolute_end_ts: float
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
    p = argparse.ArgumentParser(description="Trim FastUMI-style episodes and batch plans.")
    p.add_argument("--task-dir", type=Path, default=None)
    p.add_argument("--episode-index", type=int, default=None)
    p.add_argument("--start-sec", type=float, default=None)
    p.add_argument("--end-sec", type=float, default=None)
    p.add_argument("--plan-file", type=Path, default=None)
    p.add_argument("--tasks-root", type=Path, default=None)
    p.add_argument("--jobs", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--video", type=Path, default=None)
    p.add_argument("--video-ts", type=Path, default=None)
    p.add_argument("--trajectory", type=Path, default=None)
    p.add_argument("--hdf5", type=Path, default=None)
    p.add_argument("--states-csv", type=Path, default=None)
    p.add_argument("--frame-timestamps-csv", type=Path, default=None)
    p.add_argument("--episode-events-csv", type=Path, default=None)
    p.add_argument("--keep-absolute-timestamps", action="store_true")
    p.add_argument("--video-fps", type=float, default=None)
    p.add_argument("--codec", type=str, default="mp4v")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-random-start-crops", type=int, default=0)
    p.add_argument("--max-random-start-crop-sec", type=float, default=2.0)
    p.add_argument("--min-remaining-sec", type=float, default=2.0)
    p.add_argument("--copy-original-manifest", action="store_true")
    return p.parse_args()


def resolve_candidate(task_dir: Path, episode: int, override: Optional[Path], candidates: Sequence[str], kind: str) -> Optional[Path]:
    if override is not None:
        if not override.exists():
            raise FileNotFoundError(f"{kind} override does not exist: {override}")
        return override
    for pattern in candidates:
        path = task_dir / pattern.format(episode=episode)
        if path.exists():
            return path
    return None


def load_csv_any(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def ensure_columns(df: pd.DataFrame, columns: Sequence[str], path: Path) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")


def maybe_ffprobe(path: Path) -> dict:
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-print_format", "json", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(proc.stdout)
    except Exception:
        return {}


def detect_video_fps(video_path: Path, video_ts: pd.DataFrame, override_fps: Optional[float]) -> float:
    if override_fps is not None and override_fps > 0:
        return float(override_fps)
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if fps > 1e-3:
        return fps
    if len(video_ts) >= 2:
        dt = np.median(np.diff(video_ts["Timestamp"].astype(float).to_numpy()))
        if dt > 0:
            return 1.0 / dt
    return 60.0


def is_shared_fallback_csv(path: Optional[Path], shared_names: set[str]) -> bool:
    return path is not None and path.name in shared_names


def get_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for frame-count detection: {video_path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return count


def find_first_existing_column(df: pd.DataFrame, candidates: Sequence[str], csv_path: Path) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(f"Could not find any of columns {list(candidates)} in {csv_path}")


def find_episode_event(events_df: pd.DataFrame, episode_index: int) -> Optional[pd.Series]:
    ensure_columns(events_df, ["Episode Index", "Recording Start Timestamp", "Recording Stop Timestamp"], Path("episode_events.csv dataframe"))
    matches = events_df[events_df["Episode Index"].astype(int) == int(episode_index)]
    if matches.empty:
        return None
    return matches.iloc[-1]


def resolve_path_from_event(task_dir: Path, path_str: Any) -> Optional[Path]:
    if pd.isna(path_str):
        return None
    raw = str(path_str).strip()
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p
    parts = p.parts
    if "dataset" in parts:
        idx = parts.index("dataset")
        local_rel = Path(*parts[idx + 2:]) if len(parts) > idx + 2 else None
        if local_rel is not None:
            candidate = task_dir / local_rel
            if candidate.exists():
                return candidate
    if raw.startswith("./"):
        raw = raw[2:]
    candidate = task_dir / raw
    if candidate.exists():
        return candidate
    candidate = task_dir / p.name
    if candidate.exists():
        return candidate
    return task_dir / raw


def resolve_inputs_from_episode_events(task_dir: Path, episode_index: int, episode_events_path: Optional[Path]) -> tuple[dict[str, Optional[Path]], Optional[pd.DataFrame], list[str]]:
    notes: list[str] = []
    if episode_events_path is None or not episode_events_path.exists():
        return {}, None, notes
    events_df = load_csv_any(episode_events_path)
    event_row = find_episode_event(events_df, episode_index)
    if event_row is None:
        return {}, events_df, notes
    resolved = {
        "video_path": resolve_path_from_event(task_dir, event_row.get("Video Path")),
        "video_ts_path": resolve_path_from_event(task_dir, event_row.get("Video Timestamp CSV Path")),
        "traj_path": resolve_path_from_event(task_dir, event_row.get("Trajectory CSV Path")),
        "hdf5_path": resolve_path_from_event(task_dir, event_row.get("HDF5 Path")),
    }
    notes.append("Resolved episode-specific inputs from episode_events.csv.")
    return resolved, events_df, notes


def get_frame_timestamp_bounds(*, frame_ts_path: Optional[Path], episode_events_path: Optional[Path], episode_index: int, video_path: Path, fps: float) -> tuple[Optional[float], Optional[float], Optional[pd.DataFrame], list[str]]:
    notes: list[str] = []
    episode_events_df: Optional[pd.DataFrame] = None
    if episode_events_path is not None and episode_events_path.exists():
        episode_events_df = load_csv_any(episode_events_path)
        event_row = find_episode_event(episode_events_df, episode_index)
        if event_row is not None:
            start_ts = float(event_row["Recording Start Timestamp"])
            stop_ts = float(event_row["Recording Stop Timestamp"])
            if stop_ts > start_ts:
                notes.append("Episode bounds inferred from episode_events.csv.")
                return start_ts, stop_ts, episode_events_df, notes
    if frame_ts_path is None or not frame_ts_path.exists():
        return None, None, episode_events_df, notes
    frame_ts_df = load_csv_any(frame_ts_path)
    idx_col = next((name for name in FRAME_TS_INDEX_COLUMNS if name in frame_ts_df.columns), None)
    ts_col = find_first_existing_column(frame_ts_df, FRAME_TS_TIMESTAMP_COLUMNS, frame_ts_path)
    if idx_col is not None:
        frame_ts_df = frame_ts_df.sort_values(idx_col).reset_index(drop=True)
        matches = frame_ts_df[frame_ts_df[idx_col].astype(int) == int(episode_index)]
        if matches.empty:
            raise IndexError(f"Episode index {episode_index} not found in {frame_ts_path}")
        row_pos = int(matches.index[0])
    else:
        row_pos = int(episode_index)
        if row_pos >= len(frame_ts_df):
            raise IndexError(f"Episode index {episode_index} exceeds number of rows in {frame_ts_path} ({len(frame_ts_df)})")
    start_ts = float(frame_ts_df.iloc[row_pos][ts_col])
    next_start_ts = float(frame_ts_df.iloc[row_pos + 1][ts_col]) if row_pos + 1 < len(frame_ts_df) else None
    if next_start_ts is None:
        frame_count = get_frame_count(video_path)
        approx_duration = frame_count / float(fps) if fps > 0 else 0.0
        next_start_ts = start_ts + approx_duration + max(1.0 / max(fps, 1.0), 0.02)
        notes.append("Episode end inferred from local video duration because no next frame timestamp row existed.")
    else:
        notes.append("Episode bounds inferred from frame_timestamps.csv.")
    return start_ts, next_start_ts, episode_events_df, notes


def segment_shared_episode_stream(*, df: pd.DataFrame, timestamp_col: str, start_ts: float, end_ts: Optional[float], kind: str) -> pd.DataFrame:
    ts = df[timestamp_col].astype(float).to_numpy()
    mask = (ts >= start_ts) if end_ts is None else ((ts >= start_ts) & (ts < end_ts))
    out = df.loc[mask].copy().reset_index(drop=True)
    if out.empty:
        raise ValueError(f"Shared {kind} CSV could not be segmented for episode window [{start_ts}, {end_ts if end_ts is not None else 'inf'}).")
    return out


def prepare_episode_streams(*, task_dir: Path, episode_index: int, video_path: Path, video_ts_path: Path, traj_path: Path, frame_ts_path: Optional[Path], episode_events_path: Optional[Path], video_fps_override: Optional[float]) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], float, list[str]]:
    raw_video_ts_df = load_csv_any(video_ts_path)
    ensure_columns(raw_video_ts_df, ["Frame Index", "Timestamp"], video_ts_path)
    fps = detect_video_fps(video_path, raw_video_ts_df, video_fps_override)
    raw_traj_df = load_csv_any(traj_path)
    ensure_columns(raw_traj_df, ["Timestamp"], traj_path)
    episode_notes: list[str] = []
    shared_video_ts = is_shared_fallback_csv(video_ts_path, SHARED_VIDEO_TS_FILENAMES)
    shared_traj = is_shared_fallback_csv(traj_path, SHARED_TRAJ_FILENAMES)
    if not shared_video_ts and not shared_traj:
        episode_events_df = load_csv_any(episode_events_path) if episode_events_path is not None and episode_events_path.exists() else None
        episode_notes.append("Using episode-specific timestamp and trajectory files directly.")
        return raw_video_ts_df, raw_traj_df, episode_events_df, fps, episode_notes
    start_ts, end_ts, episode_events_df, bound_notes = get_frame_timestamp_bounds(frame_ts_path=frame_ts_path, episode_events_path=episode_events_path, episode_index=episode_index, video_path=video_path, fps=fps)
    episode_notes.extend(bound_notes)
    if start_ts is None:
        raise ValueError("Shared temp CSV fallback was selected, but no frame_timestamps.csv or usable episode_events.csv was available to segment the shared stream per episode.")
    if shared_video_ts:
        episode_notes.append(f"Segmented shared video timestamp CSV {video_ts_path.name} for this episode.")
        raw_video_ts_df = segment_shared_episode_stream(df=raw_video_ts_df, timestamp_col="Timestamp", start_ts=start_ts, end_ts=end_ts, kind="video timestamp")
        raw_video_ts_df["Frame Index"] = np.arange(len(raw_video_ts_df), dtype=np.int64)
    if shared_traj:
        episode_notes.append(f"Segmented shared trajectory CSV {traj_path.name} for this episode.")
        raw_traj_df = segment_shared_episode_stream(df=raw_traj_df, timestamp_col="Timestamp", start_ts=start_ts, end_ts=end_ts, kind="trajectory")
    return raw_video_ts_df, raw_traj_df, episode_events_df, fps, episode_notes


def get_hdf5_num_steps(in_hdf5: Path) -> int:
    with h5py.File(in_hdf5, "r") as src:
        candidates: list[int] = []
        if "observations" in src and "qpos" in src["observations"]:
            candidates.append(int(src["observations"]["qpos"].shape[0]))
        if "observations" in src and "images" in src["observations"]:
            for cam_name in src["observations"]["images"].keys():
                candidates.append(int(src["observations"]["images"][cam_name].shape[0]))
                break
        if "action" in src:
            candidates.append(int(src["action"].shape[0]))
    if not candidates:
        raise ValueError(f"Could not infer sequence length from HDF5 {in_hdf5}")
    return min(candidates)


def build_window(video_ts_df: pd.DataFrame, start_sec: float, end_sec: float) -> TrimWindow:
    if end_sec <= start_sec:
        raise ValueError("end-sec must be larger than start-sec")
    ensure_columns(video_ts_df, ["Frame Index", "Timestamp"], Path("video timestamps dataframe"))
    first_ts = float(video_ts_df["Timestamp"].iloc[0])
    return TrimWindow("base_trim", first_ts + float(start_sec), first_ts + float(end_sec), float(start_sec), float(end_sec), float(end_sec - start_sec))


def trim_video_frames(video_path: Path, out_path: Path, kept_frame_indices: np.ndarray, fps: float, codec: str) -> int:
    if kept_frame_indices.size == 0:
        raise ValueError("No frames selected for the trimmed clip")
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
        raise RuntimeError(f"Could not open output video writer for {out_path}")
    keep_set = set(int(x) for x in kept_frame_indices.tolist())
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_no = start_frame
    written = 0
    while frame_no <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_no in keep_set:
            writer.write(frame)
            written += 1
        frame_no += 1
    cap.release()
    writer.release()
    if written != kept_frame_indices.size:
        raise RuntimeError(f"Expected to write {kept_frame_indices.size} frames, but wrote {written}.")
    return written


def write_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def infer_states_group(states_df: pd.DataFrame, episode_index: int) -> pd.DataFrame:
    ensure_columns(states_df, ["Start Time", "Frame Timestamp"], Path("states.csv dataframe"))
    ordered_start_times: list[float] = []
    seen = set()
    for value in states_df["Start Time"].tolist():
        key = float(value)
        if key not in seen:
            ordered_start_times.append(key)
            seen.add(key)
    if episode_index >= len(ordered_start_times):
        raise IndexError(f"Episode index {episode_index} exceeds the number of unique Start Time groups in states.csv ({len(ordered_start_times)})")
    target = ordered_start_times[episode_index]
    return states_df[np.isclose(states_df["Start Time"].astype(float).to_numpy(), target)]


def infer_states_group_with_events(states_df: pd.DataFrame, episode_index: int, events_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if events_df is not None:
        event_row = find_episode_event(events_df, episode_index)
        if event_row is not None:
            start_ts = float(event_row["Recording Start Timestamp"])
            stop_ts = float(event_row["Recording Stop Timestamp"])
            if "Start Time" in states_df.columns:
                grouped = states_df[np.isclose(states_df["Start Time"].astype(float).to_numpy(), start_ts)]
                if not grouped.empty:
                    return grouped
            if "Frame Timestamp" in states_df.columns:
                grouped = states_df[(states_df["Frame Timestamp"].astype(float) >= start_ts) & (states_df["Frame Timestamp"].astype(float) <= stop_ts)]
                if not grouped.empty:
                    return grouped
    return infer_states_group(states_df, episode_index)


def trim_hdf5(in_hdf5: Path, out_hdf5: Path, downsampled_indices: np.ndarray) -> int:
    if downsampled_indices.size == 0:
        raise ValueError("No downsampled indices selected for HDF5 trimming")
    with h5py.File(in_hdf5, "r") as src, h5py.File(out_hdf5, "w") as dst:
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        obs_src = src["observations"]
        obs_dst = dst.create_group("observations")
        img_dst_grp = obs_dst.create_group("images")
        for cam_name in obs_src["images"].keys():
            img_dst_grp.create_dataset(cam_name, data=obs_src["images"][cam_name][downsampled_indices], compression="gzip", compression_opts=4)
        obs_dst.create_dataset("qpos", data=obs_src["qpos"][downsampled_indices])
        if "action" in src:
            dst.create_dataset("action", data=src["action"][downsampled_indices])
    return int(downsampled_indices.size)


def trim_once(*, task_dir: Path, episode_index: int, output_root: Path, video_path: Path, video_ts_path: Path, traj_path: Path, hdf5_path: Optional[Path], states_path: Optional[Path], frame_ts_path: Optional[Path], episode_events_path: Optional[Path], video_ts_df: pd.DataFrame, traj_df: pd.DataFrame, episode_events_df: Optional[pd.DataFrame], source_notes: Sequence[str], window: TrimWindow, keep_absolute_timestamps: bool, fps: float, codec: str, include_original_manifest: bool) -> TrimResult:
    out_dir = output_root / f"episode_{episode_index}_{window.label}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "camera").mkdir(parents=True, exist_ok=True)
    (out_dir / "csv").mkdir(parents=True, exist_ok=True)
    ensure_columns(video_ts_df, ["Frame Index", "Timestamp"], video_ts_path)
    ensure_columns(traj_df, ["Timestamp"], traj_path)
    ts = video_ts_df["Timestamp"].astype(float).to_numpy()
    keep_mask = (ts >= window.absolute_start_ts) & (ts <= window.absolute_end_ts)
    kept_video_ts = video_ts_df.loc[keep_mask].copy().reset_index(drop=True)
    if kept_video_ts.empty:
        raise ValueError(f"Trim window {window.relative_start_sec:.3f}-{window.relative_end_sec:.3f}s selects no raw video frames")
    original_frame_indices = kept_video_ts["Frame Index"].astype(int).to_numpy()
    kept_video_ts.insert(1, "Original Frame Index", original_frame_indices)
    kept_video_ts["Frame Index"] = np.arange(len(kept_video_ts), dtype=np.int64)
    if not keep_absolute_timestamps:
        kept_video_ts["Timestamp"] = kept_video_ts["Timestamp"].astype(float) - window.absolute_start_ts
    write_csv(kept_video_ts, out_dir / "csv" / "video_timestamps.csv")
    trimmed_video_path = out_dir / "camera" / f"trimmed_episode_{episode_index}.mp4"
    written_frames = trim_video_frames(video_path, trimmed_video_path, original_frame_indices, fps, codec)
    traj_ts = traj_df["Timestamp"].astype(float).to_numpy()
    traj_keep = (traj_ts >= window.absolute_start_ts) & (traj_ts <= window.absolute_end_ts)
    trimmed_traj = traj_df.loc[traj_keep].copy().reset_index(drop=True)
    if trimmed_traj.empty:
        raise ValueError(f"Trim window {window.relative_start_sec:.3f}-{window.relative_end_sec:.3f}s selects no trajectory rows")
    if not keep_absolute_timestamps:
        trimmed_traj["Timestamp"] = trimmed_traj["Timestamp"].astype(float) - window.absolute_start_ts
    write_csv(trimmed_traj, out_dir / "csv" / "trajectory.csv")
    downsampled_source = video_ts_df.iloc[::3].copy().reset_index(drop=True)
    ds_ts = downsampled_source["Timestamp"].astype(float).to_numpy()
    ds_keep = np.where((ds_ts >= window.absolute_start_ts) & (ds_ts <= window.absolute_end_ts))[0]
    if ds_keep.size == 0:
        raise ValueError(f"Trim window {window.relative_start_sec:.3f}-{window.relative_end_sec:.3f}s selects no 20 Hz frames.")
    hdf5_notes: list[str] = []
    if hdf5_path is not None and hdf5_path.exists():
        hdf5_len = get_hdf5_num_steps(hdf5_path)
        if len(downsampled_source) != hdf5_len:
            hdf5_notes.append(f"Downsampled source length ({len(downsampled_source)}) differs from HDF5 length ({hdf5_len}); indices were clipped when necessary.")
        if ds_keep.max() >= hdf5_len:
            clipped = ds_keep[ds_keep < hdf5_len]
            if clipped.size == 0:
                raise ValueError(f"Trim requested 20 Hz indices up to {int(ds_keep.max())}, but HDF5 only has indices 0-{hdf5_len-1}.")
            hdf5_notes.append(f"Clipped {ds_keep.size - clipped.size} invalid 20 Hz indices because HDF5 only supports 0-{hdf5_len-1}.")
            ds_keep = clipped
    trimmed_downsampled = downsampled_source.iloc[ds_keep].copy().reset_index(drop=True)
    trimmed_downsampled.insert(1, "Original Downsampled Index", ds_keep)
    trimmed_downsampled["Frame Index"] = np.arange(len(trimmed_downsampled), dtype=np.int64)
    if not keep_absolute_timestamps:
        trimmed_downsampled["Timestamp"] = trimmed_downsampled["Timestamp"].astype(float) - window.absolute_start_ts
    write_csv(trimmed_downsampled, out_dir / "csv" / "downsampled_video_timestamps_20hz.csv")
    has_hdf5 = False
    if hdf5_path is not None and hdf5_path.exists():
        trim_hdf5(hdf5_path, out_dir / f"episode_{episode_index}.hdf5", ds_keep)
        has_hdf5 = True
    has_states = False
    states_notes: list[str] = []
    if states_path is not None and states_path.exists():
        states_df = load_csv_any(states_path)
        try:
            ep_states = infer_states_group_with_events(states_df, episode_index, episode_events_df)
            trimmed_states = ep_states[(ep_states["Frame Timestamp"].astype(float) >= window.absolute_start_ts) & (ep_states["Frame Timestamp"].astype(float) <= window.absolute_end_ts)].copy().reset_index(drop=True)
            if not trimmed_states.empty:
                if not keep_absolute_timestamps:
                    trimmed_states["Frame Timestamp"] = trimmed_states["Frame Timestamp"].astype(float) - window.absolute_start_ts
                    if "Trajectory Timestamp" in trimmed_states.columns:
                        trimmed_states["Trajectory Timestamp"] = trimmed_states["Trajectory Timestamp"].astype(float) - window.absolute_start_ts
                write_csv(trimmed_states, out_dir / "states.csv")
                has_states = True
        except Exception as exc:
            states_notes.append(f"Could not trim states.csv reliably: {exc}")
            (out_dir / "states_trim_warning.txt").write_text("Could not trim states.csv reliably.\n" + f"Reason: {exc}\n" + "Grouping attempted via episode_events.csv first, then via Start Time fallback.\n")
    if episode_events_df is not None:
        event_row = find_episode_event(episode_events_df, episode_index)
        if event_row is not None:
            write_csv(pd.DataFrame([event_row]), out_dir / "csv" / "episode_events.csv")
    manifest_notes = [
        "Relative trim times are interpreted relative to the first frame timestamp of the selected episode.",
        "If per-episode temp_video_timestamps_<episode>.csv exists, the HDF5 trim uses every third local raw frame timestamp.",
    ] + list(source_notes) + hdf5_notes + states_notes
    manifest = {
        "task_dir": str(task_dir),
        "episode_index": episode_index,
        "window": asdict(window),
        "source_video_fps": fps,
        "codec": codec,
        "output_video_ffprobe": maybe_ffprobe(trimmed_video_path),
        "notes": manifest_notes,
    }
    if include_original_manifest:
        manifest["sources"] = {
            "video": str(video_path),
            "video_timestamps": str(video_ts_path),
            "trajectory_csv": str(traj_path),
            "hdf5": str(hdf5_path) if hdf5_path else None,
            "states_csv": str(states_path) if states_path else None,
            "frame_timestamps_csv": str(frame_ts_path) if frame_ts_path else None,
            "episode_events_csv": str(episode_events_path) if episode_events_path else None,
        }
    (out_dir / "trim_manifest.json").write_text(json.dumps(manifest, indent=2))
    return TrimResult(window.label, str(out_dir), written_frames, len(trimmed_traj), len(trimmed_downsampled), has_hdf5, has_states, window.relative_start_sec, window.relative_end_sec)


def build_augmented_windows(base: TrimWindow, num: int, max_crop_sec: float, min_remaining_sec: float, seed: int) -> list[TrimWindow]:
    rng = random.Random(seed)
    windows: list[TrimWindow] = []
    for i in range(num):
        max_allowed = min(max_crop_sec, max(0.0, base.duration_sec - min_remaining_sec))
        if max_allowed <= 0:
            break
        delta = rng.uniform(0.0, max_allowed)
        windows.append(TrimWindow(f"aug_startcrop_{i+1:02d}", base.absolute_start_ts + delta, base.absolute_end_ts, base.relative_start_sec + delta, base.relative_end_sec, base.relative_end_sec - (base.relative_start_sec + delta)))
    return windows


def parse_trim_spec(spec: str) -> tuple[bool, Optional[float], Optional[float]]:
    raw = spec.strip()
    if not raw:
        raise ValueError("Empty trim specification")
    if raw.lower() == "x":
        return True, None, None
    match = RANGE_RE.match(raw)
    if not match:
        raise ValueError(f"Invalid trim spec '{spec}'. Expected 'start-end' or 'x'.")
    start_sec = float(match.group(1))
    end_sec = float(match.group(2))
    if end_sec <= start_sec:
        raise ValueError(f"Invalid trim spec '{spec}': end must be larger than start.")
    return False, start_sec, end_sec


def parse_plan_file(plan_file: Path) -> list[BatchTaskPlan]:
    blocks, current = [], []
    for raw_line in plan_file.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                blocks.append(current)
                current = []
            continue
        if not line.startswith("#"):
            current.append(line)
    if current:
        blocks.append(current)
    plans: list[BatchTaskPlan] = []
    for idx, block in enumerate(blocks):
        if len(block) < 2:
            raise ValueError(f"Invalid block #{idx + 1} in {plan_file}: each block needs a task name plus at least one trim spec line")
        task_name, episode_specs = block[0], block[1:]
        for spec in episode_specs:
            parse_trim_spec(spec)
        plans.append(BatchTaskPlan(task_name, episode_specs))
    if not plans:
        raise ValueError(f"No valid task blocks found in {plan_file}")
    return plans


def build_batch_requests(tasks_root: Path, plans: list[BatchTaskPlan]) -> list[BatchEpisodeRequest]:
    requests: list[BatchEpisodeRequest] = []
    for plan in plans:
        task_dir = (tasks_root / plan.task_name).resolve()
        for episode_index, spec in enumerate(plan.episode_specs):
            skip, start_sec, end_sec = parse_trim_spec(spec)
            requests.append(BatchEpisodeRequest(plan.task_name, str(task_dir), episode_index, spec, skip, start_sec, end_sec))
    return requests


def auto_detect_inputs(task_dir: Path, episode_index: int, episode_events_override: Optional[Path] = None) -> dict[str, Optional[Path]]:
    episode_events_path = resolve_candidate(task_dir, episode_index, episode_events_override, EVENT_CANDIDATES, "episode events CSV")
    from_events, _, event_notes = resolve_inputs_from_episode_events(task_dir, episode_index, episode_events_path)
    video_path = from_events.get("video_path") if from_events else None
    video_ts_path = from_events.get("video_ts_path") if from_events else None
    traj_path = from_events.get("traj_path") if from_events else None
    hdf5_path = from_events.get("hdf5_path") if from_events else None
    if video_path is None:
        video_path = resolve_candidate(task_dir, episode_index, None, VIDEO_CANDIDATES, "video")
    if video_ts_path is None:
        video_ts_path = resolve_candidate(task_dir, episode_index, None, VIDEO_TS_CANDIDATES, "video timestamps")
    if traj_path is None:
        traj_path = resolve_candidate(task_dir, episode_index, None, TRAJ_CANDIDATES, "trajectory CSV")
    if hdf5_path is None:
        hdf5_path = resolve_candidate(task_dir, episode_index, None, HDF5_CANDIDATES, "HDF5")
    states_path = task_dir / "states.csv" if (task_dir / "states.csv").exists() else None
    frame_ts_path = task_dir / "csv" / "frame_timestamps.csv" if (task_dir / "csv" / "frame_timestamps.csv").exists() else None
    if video_path is None or video_ts_path is None or traj_path is None:
        raise FileNotFoundError(f"Could not auto-detect required inputs for {task_dir}, episode {episode_index}")
    return {
        "video_path": video_path,
        "video_ts_path": video_ts_path,
        "traj_path": traj_path,
        "hdf5_path": hdf5_path,
        "states_path": states_path,
        "frame_ts_path": frame_ts_path,
        "episode_events_path": episode_events_path,
        "event_resolution_notes": event_notes,
    }


def run_single_trim(*, task_dir: Path, episode_index: int, start_sec: float, end_sec: float, output_root: Optional[Path], video: Optional[Path], video_ts: Optional[Path], trajectory: Optional[Path], hdf5: Optional[Path], states_csv: Optional[Path], frame_timestamps_csv: Optional[Path], episode_events_csv: Optional[Path], keep_absolute_timestamps: bool, video_fps: Optional[float], codec: str, seed: int, num_random_start_crops: int, max_random_start_crop_sec: float, min_remaining_sec: float, copy_original_manifest: bool) -> dict[str, Any]:
    task_dir = task_dir.resolve()
    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory does not exist: {task_dir}")
    auto = auto_detect_inputs(task_dir, episode_index, episode_events_csv if episode_events_csv else None)
    video_path = video if video is not None else auto["video_path"]
    video_ts_path = video_ts if video_ts is not None else auto["video_ts_path"]
    traj_path = trajectory if trajectory is not None else auto["traj_path"]
    hdf5_path = hdf5 if hdf5 is not None else auto["hdf5_path"]
    states_path = states_csv if states_csv else auto["states_path"]
    frame_ts_path = frame_timestamps_csv if frame_timestamps_csv else auto["frame_ts_path"]
    episode_events_path = episode_events_csv if episode_events_csv else auto["episode_events_path"]
    prepared_video_ts_df, prepared_traj_df, episode_events_df, fps, source_notes = prepare_episode_streams(task_dir=task_dir, episode_index=episode_index, video_path=video_path, video_ts_path=video_ts_path, traj_path=traj_path, frame_ts_path=frame_ts_path, episode_events_path=episode_events_path, video_fps_override=video_fps)
    source_notes = list(auto.get("event_resolution_notes", [])) + list(source_notes)
    base_window = build_window(prepared_video_ts_df, start_sec, end_sec)
    final_output_root = (output_root or (task_dir / "trimmed")).resolve()
    final_output_root.mkdir(parents=True, exist_ok=True)
    windows = [base_window] + build_augmented_windows(base_window, num_random_start_crops, max_random_start_crop_sec, min_remaining_sec, seed)
    results = [
        trim_once(task_dir=task_dir, episode_index=episode_index, output_root=final_output_root, video_path=video_path, video_ts_path=video_ts_path, traj_path=traj_path, hdf5_path=hdf5_path, states_path=states_path, frame_ts_path=frame_ts_path, episode_events_path=episode_events_path, video_ts_df=prepared_video_ts_df, traj_df=prepared_traj_df, episode_events_df=episode_events_df, source_notes=source_notes, window=window, keep_absolute_timestamps=keep_absolute_timestamps, fps=fps, codec=codec, include_original_manifest=copy_original_manifest)
        for window in windows
    ]
    summary = {
        "task_dir": str(task_dir),
        "episode_index": episode_index,
        "input_files": {
            "video": str(video_path),
            "video_timestamps": str(video_ts_path),
            "trajectory": str(traj_path),
            "hdf5": str(hdf5_path) if hdf5_path else None,
            "states_csv": str(states_path) if states_path else None,
            "frame_timestamps_csv": str(frame_ts_path) if frame_ts_path else None,
            "episode_events_csv": str(episode_events_path) if episode_events_path else None,
        },
        "notes": list(source_notes),
        "results": [asdict(r) for r in results],
        "recommendations": [
            "Use the base trim for your essential episode window.",
            "When episode_events.csv exists, its file paths and recording bounds are preferred.",
            "Only use random start-crop augmentations when the first cropped-away seconds are not themselves the core of the task.",
        ],
    }
    (final_output_root / f"episode_{episode_index}_trim_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def process_batch_episode(payload: dict[str, Any]) -> dict[str, Any]:
    request = BatchEpisodeRequest(**payload["request"])
    output_root_base = Path(payload["output_root_base"]).resolve() if payload["output_root_base"] is not None else None
    if request.skip:
        return asdict(BatchEpisodeOutcome(request.task_name, request.task_dir, request.episode_index, request.trim_spec, True, "skipped", "Skipped because trim spec is 'x'.", None, None))
    task_dir = Path(request.task_dir).resolve()
    output_root = task_dir / "trimmed" if output_root_base is None else output_root_base / request.task_name
    try:
        summary = run_single_trim(task_dir=task_dir, episode_index=request.episode_index, start_sec=float(request.start_sec), end_sec=float(request.end_sec), output_root=output_root, video=None, video_ts=None, trajectory=None, hdf5=None, states_csv=None, frame_timestamps_csv=None, episode_events_csv=None, keep_absolute_timestamps=bool(payload["keep_absolute_timestamps"]), video_fps=payload["video_fps"], codec=str(payload["codec"]), seed=int(payload["seed"]), num_random_start_crops=int(payload["num_random_start_crops"]), max_random_start_crop_sec=float(payload["max_random_start_crop_sec"]), min_remaining_sec=float(payload["min_remaining_sec"]), copy_original_manifest=bool(payload["copy_original_manifest"]))
        return asdict(BatchEpisodeOutcome(request.task_name, request.task_dir, request.episode_index, request.trim_spec, False, "ok", None, str(output_root / f"episode_{request.episode_index}_trim_summary.json"), summary))
    except Exception as exc:
        return asdict(BatchEpisodeOutcome(request.task_name, request.task_dir, request.episode_index, request.trim_spec, False, "error", str(exc), None, None))


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
        "keep_absolute_timestamps": args.keep_absolute_timestamps,
        "video_fps": args.video_fps,
        "codec": args.codec,
        "seed": args.seed,
        "num_random_start_crops": args.num_random_start_crops,
        "max_random_start_crop_sec": args.max_random_start_crop_sec,
        "min_remaining_sec": args.min_remaining_sec,
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
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = [executor.submit(process_batch_episode, payload) for payload in payloads]
            for future in as_completed(futures):
                outcome = future.result()
                outcomes.append(outcome)
                if args.fail_fast and outcome["status"] == "error":
                    for f in futures:
                        f.cancel()
                    break
    outcomes.sort(key=lambda item: (item["task_name"], item["episode_index"]))
    summary = {
        "plan_file": str(args.plan_file.resolve()),
        "tasks_root": str(tasks_root),
        "output_root_base": str(output_root_base) if output_root_base is not None else None,
        "submitted": len(requests),
        "jobs": jobs,
        "num_ok": sum(1 for item in outcomes if item["status"] == "ok"),
        "num_skipped": sum(1 for item in outcomes if item["status"] == "skipped"),
        "num_error": sum(1 for item in outcomes if item["status"] == "error"),
        "results": outcomes,
        "notes": [
            "Each task block maps trim lines to episode indices starting at 0.",
            "A trim spec 'x' means that episode is skipped.",
            "When episode_events.csv exists, its file paths and recording bounds are preferred.",
        ],
    }
    summary_path = (output_root_base / "trim_batch_summary.json") if output_root_base is not None else (tasks_root / "trim_batch_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    summary["summary_path"] = str(summary_path)
    return summary


def validate_mode(args: argparse.Namespace) -> str:
    single_fields = [args.task_dir, args.episode_index, args.start_sec, args.end_sec]
    single_mode = all(value is not None for value in single_fields)
    batch_mode = args.plan_file is not None
    if single_mode and batch_mode:
        raise ValueError("Choose either single-episode mode or batch mode, not both.")
    if not single_mode and not batch_mode:
        raise ValueError("Provide either --task-dir/--episode-index/--start-sec/--end-sec for single mode, or --plan-file/--tasks-root for batch mode.")
    return "batch" if batch_mode else "single"


def main() -> None:
    args = parse_args()
    mode = validate_mode(args)
    if mode == "batch":
        print(json.dumps(run_batch(args), indent=2))
        return
    summary = run_single_trim(task_dir=args.task_dir, episode_index=int(args.episode_index), start_sec=float(args.start_sec), end_sec=float(args.end_sec), output_root=args.output_root, video=args.video, video_ts=args.video_ts, trajectory=args.trajectory, hdf5=args.hdf5, states_csv=args.states_csv, frame_timestamps_csv=args.frame_timestamps_csv, episode_events_csv=args.episode_events_csv, keep_absolute_timestamps=args.keep_absolute_timestamps, video_fps=args.video_fps, codec=args.codec, seed=args.seed, num_random_start_crops=args.num_random_start_crops, max_random_start_crop_sec=args.max_random_start_crop_sec, min_remaining_sec=args.min_remaining_sec, copy_original_manifest=args.copy_original_manifest)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
