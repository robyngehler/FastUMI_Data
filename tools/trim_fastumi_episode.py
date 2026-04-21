#!/usr/bin/env python3
"""
Trim a FastUMI-style raw episode consistently across video, timestamps, trajectory,
optional HDF5, and optional states.csv. Can also create temporal augmentations by
randomly cropping the first part of an already-trimmed essential episode.

Assumptions / intended use:
- Raw video is stored per episode, e.g. camera/temp_video_<idx>.mp4.
- Per-episode raw video timestamps and raw trajectory CSVs are available, ideally as
    csv/temp_video_timestamps_<idx>.csv and csv/temp_trajectory_<idx>.csv.
- episode_events.csv is optional but recommended because it records the Enter-triggered
    tracking/recording boundaries per episode and improves states.csv correlation.
- Relative trim times are given in seconds relative to the first video frame of the
  selected episode, matching what a normal video player shows.

Outputs are written into dedicated subfolders so the original recording remains untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence

import cv2
import h5py
import numpy as np
import pandas as pd


VIDEO_TS_CANDIDATES = [
    "csv/temp_video_timestamps_{episode}.csv",
    "csv/video_timestamps_{episode}.csv",
    "csv/video_timestamps_episode_{episode}.csv",
    "csv/temp_video_timestamps.csv",  # only safe for a single-episode recording session
]
TRAJ_CANDIDATES = [
    "csv/temp_trajectory_{episode}.csv",
    "csv/trajectory_{episode}.csv",
    "csv/trajectory_episode_{episode}.csv",
    "csv/temp_trajectory.csv",  # only safe for a single-episode recording session
]
EVENT_CANDIDATES = [
    "csv/episode_events.csv",
]
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trim FastUMI-style episodes and optionally generate random start-crop augmentations.")
    p.add_argument("--task-dir", type=Path, required=True, help="FastUMI task directory, e.g. dataset/pour_water")
    p.add_argument("--episode-index", type=int, required=True, help="Episode index to trim")
    p.add_argument("--start-sec", type=float, required=True, help="Trim start in seconds relative to the first frame of the selected episode")
    p.add_argument("--end-sec", type=float, required=True, help="Trim end in seconds relative to the first frame of the selected episode")
    p.add_argument("--output-root", type=Path, default=None, help="Where trimmed outputs are written. Defaults to <task-dir>/trimmed")

    p.add_argument("--video", type=Path, default=None, help="Override input raw video path")
    p.add_argument("--video-ts", type=Path, default=None, help="Override input raw video timestamp CSV path")
    p.add_argument("--trajectory", type=Path, default=None, help="Override input raw trajectory CSV path")
    p.add_argument("--hdf5", type=Path, default=None, help="Optional override input HDF5 path")
    p.add_argument("--states-csv", type=Path, default=None, help="Optional override states.csv path")
    p.add_argument("--frame-timestamps-csv", type=Path, default=None, help="Optional override frame_timestamps.csv path")
    p.add_argument("--episode-events-csv", type=Path, default=None, help="Optional override episode_events.csv path")

    p.add_argument("--keep-absolute-timestamps", action="store_true", help="Keep original timestamps instead of rebasing to zero / first sample")
    p.add_argument("--video-fps", type=float, default=None, help="Override output video FPS. If omitted, infer from source or timestamps")
    p.add_argument("--codec", type=str, default="mp4v", help="OpenCV fourcc codec for trimmed mp4 output")
    p.add_argument("--seed", type=int, default=0, help="Random seed for augmentation generation")

    p.add_argument("--num-random-start-crops", type=int, default=0, help="Generate N extra variants by cropping a random amount from the beginning of the essential trimmed clip")
    p.add_argument("--max-random-start-crop-sec", type=float, default=2.0, help="Maximum additional random crop from the beginning, in seconds")
    p.add_argument("--min-remaining-sec", type=float, default=2.0, help="Minimum remaining clip duration after random start crop")

    p.add_argument("--copy-original-manifest", action="store_true", help="Copy source file paths into the output manifest for traceability")
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
    # Trim accidental whitespace in column names.
    df.columns = [str(c).strip() for c in df.columns]
    return df


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


def ensure_columns(df: pd.DataFrame, columns: Sequence[str], path: Path) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")


def build_window(video_ts_df: pd.DataFrame, start_sec: float, end_sec: float) -> TrimWindow:
    if end_sec <= start_sec:
        raise ValueError("end-sec must be larger than start-sec")

    ensure_columns(video_ts_df, ["Frame Index", "Timestamp"], Path("video timestamps dataframe"))
    first_ts = float(video_ts_df["Timestamp"].iloc[0])
    abs_start = first_ts + float(start_sec)
    abs_end = first_ts + float(end_sec)
    return TrimWindow(
        label="base_trim",
        absolute_start_ts=abs_start,
        absolute_end_ts=abs_end,
        relative_start_sec=float(start_sec),
        relative_end_sec=float(end_sec),
        duration_sec=float(end_sec - start_sec),
    )


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
        raise RuntimeError(
            f"Expected to write {kept_frame_indices.size} frames, but wrote {written}. "
            f"This usually indicates a corrupted source video or a sparse frame index selection."
        )
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
        raise IndexError(
            f"Episode index {episode_index} exceeds the number of unique Start Time groups in states.csv ({len(ordered_start_times)})"
        )
    target = ordered_start_times[episode_index]
    return states_df[np.isclose(states_df["Start Time"].astype(float).to_numpy(), target)]


def find_episode_event(events_df: pd.DataFrame, episode_index: int) -> Optional[pd.Series]:
    ensure_columns(
        events_df,
        ["Episode Index", "Recording Start Timestamp", "Recording Stop Timestamp"],
        Path("episode_events.csv dataframe"),
    )
    matches = events_df[events_df["Episode Index"].astype(int) == int(episode_index)]
    if matches.empty:
        return None
    return matches.iloc[-1]


def infer_states_group_with_events(
    states_df: pd.DataFrame,
    episode_index: int,
    events_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if events_df is not None:
        event_row = find_episode_event(events_df, episode_index)
        if event_row is not None:
            target = float(event_row["Recording Start Timestamp"])
            grouped = states_df[np.isclose(states_df["Start Time"].astype(float).to_numpy(), target)]
            if not grouped.empty:
                return grouped
    return infer_states_group(states_df, episode_index)


def trim_hdf5(in_hdf5: Path, out_hdf5: Path, downsampled_indices: np.ndarray) -> int:
    if downsampled_indices.size == 0:
        raise ValueError("No downsampled indices selected for HDF5 trimming")

    with h5py.File(in_hdf5, "r") as src, h5py.File(out_hdf5, "w") as dst:
        for key, value in src.attrs.items():
            dst.attrs[key] = value

        # observations/images/*
        obs_src = src["observations"]
        obs_dst = dst.create_group("observations")
        img_src_grp = obs_src["images"]
        img_dst_grp = obs_dst.create_group("images")
        for cam_name in img_src_grp.keys():
            img_arr = img_src_grp[cam_name][downsampled_indices]
            img_dst_grp.create_dataset(cam_name, data=img_arr, compression="gzip", compression_opts=4)

        # qpos and action
        obs_dst.create_dataset("qpos", data=obs_src["qpos"][downsampled_indices])
        if "action" in src:
            dst.create_dataset("action", data=src["action"][downsampled_indices])
    return int(downsampled_indices.size)


def maybe_ffprobe(path: Path) -> dict:
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-print_format",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(proc.stdout)
    except Exception:
        return {}


def trim_once(
    *,
    task_dir: Path,
    episode_index: int,
    output_root: Path,
    video_path: Path,
    video_ts_path: Path,
    traj_path: Path,
    hdf5_path: Optional[Path],
    states_path: Optional[Path],
    frame_ts_path: Optional[Path],
    episode_events_path: Optional[Path],
    window: TrimWindow,
    keep_absolute_timestamps: bool,
    fps: float,
    codec: str,
    include_original_manifest: bool,
) -> TrimResult:
    out_dir = output_root / f"episode_{episode_index}_{window.label}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "camera").mkdir(parents=True, exist_ok=True)
    (out_dir / "csv").mkdir(parents=True, exist_ok=True)

    video_ts = load_csv_any(video_ts_path)
    traj_df = load_csv_any(traj_path)
    episode_events_df = load_csv_any(episode_events_path) if episode_events_path is not None and episode_events_path.exists() else None
    ensure_columns(video_ts, ["Frame Index", "Timestamp"], video_ts_path)
    ensure_columns(traj_df, ["Timestamp"], traj_path)

    # Select raw video frames by timestamp window.
    ts = video_ts["Timestamp"].astype(float).to_numpy()
    keep_mask = (ts >= window.absolute_start_ts) & (ts <= window.absolute_end_ts)
    kept_video_ts = video_ts.loc[keep_mask].copy().reset_index(drop=True)
    if kept_video_ts.empty:
        raise ValueError(f"Trim window {window.relative_start_sec:.3f}-{window.relative_end_sec:.3f}s selects no raw video frames")

    original_frame_indices = kept_video_ts["Frame Index"].astype(int).to_numpy()
    new_frame_indices = np.arange(len(kept_video_ts), dtype=np.int64)
    kept_video_ts.insert(1, "Original Frame Index", original_frame_indices)
    kept_video_ts["Frame Index"] = new_frame_indices

    if keep_absolute_timestamps:
        rebased_video_ts = kept_video_ts
    else:
        kept_video_ts["Timestamp"] = kept_video_ts["Timestamp"].astype(float) - window.absolute_start_ts
        rebased_video_ts = kept_video_ts

    trimmed_video_path = out_dir / "camera" / f"trimmed_episode_{episode_index}.mp4"
    written_frames = trim_video_frames(video_path, trimmed_video_path, original_frame_indices, fps, codec)
    write_csv(rebased_video_ts, out_dir / "csv" / "video_timestamps.csv")

    # Select raw trajectory rows.
    traj_ts = traj_df["Timestamp"].astype(float).to_numpy()
    traj_keep = (traj_ts >= window.absolute_start_ts) & (traj_ts <= window.absolute_end_ts)
    trimmed_traj = traj_df.loc[traj_keep].copy().reset_index(drop=True)
    if trimmed_traj.empty:
        raise ValueError(f"Trim window {window.relative_start_sec:.3f}-{window.relative_end_sec:.3f}s selects no trajectory rows")
    if not keep_absolute_timestamps:
        trimmed_traj["Timestamp"] = trimmed_traj["Timestamp"].astype(float) - window.absolute_start_ts
    write_csv(trimmed_traj, out_dir / "csv" / "trajectory.csv")

    # Downsampled stream / HDF5 alignment: original HDF5 uses every third video timestamp.
    downsampled_source = video_ts.iloc[::3].copy().reset_index(drop=True)
    ds_ts = downsampled_source["Timestamp"].astype(float).to_numpy()
    ds_keep = np.where((ds_ts >= window.absolute_start_ts) & (ds_ts <= window.absolute_end_ts))[0]
    if ds_keep.size == 0:
        raise ValueError(
            f"Trim window {window.relative_start_sec:.3f}-{window.relative_end_sec:.3f}s selects no 20 Hz frames. "
            "Choose a wider window or inspect the timestamps."
        )

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
    if states_path is not None and states_path.exists():
        states_df = load_csv_any(states_path)
        try:
            ep_states = infer_states_group_with_events(states_df, episode_index, episode_events_df)
            trimmed_states = ep_states[
                (ep_states["Frame Timestamp"].astype(float) >= window.absolute_start_ts)
                & (ep_states["Frame Timestamp"].astype(float) <= window.absolute_end_ts)
            ].copy().reset_index(drop=True)
            if not trimmed_states.empty:
                if not keep_absolute_timestamps:
                    trimmed_states["Frame Timestamp"] = trimmed_states["Frame Timestamp"].astype(float) - window.absolute_start_ts
                    if "Trajectory Timestamp" in trimmed_states.columns:
                        trimmed_states["Trajectory Timestamp"] = trimmed_states["Trajectory Timestamp"].astype(float) - window.absolute_start_ts
                write_csv(trimmed_states, out_dir / "states.csv")
                has_states = True
        except Exception as exc:
            warning_path = out_dir / "states_trim_warning.txt"
            episode_events_hint = (
                "episode_events.csv was not available or did not contain a matching episode row.\n"
                if episode_events_df is None or find_episode_event(episode_events_df, episode_index) is None
                else ""
            )
            warning_path.write_text(
                "Could not trim states.csv reliably.\n"
                f"Reason: {exc}\n"
                + episode_events_hint +
                "states.csv does not store an explicit episode id,\n"
                "so grouping falls back to Start Time matching when needed.\n"
            )

    if episode_events_df is not None:
        event_row = find_episode_event(episode_events_df, episode_index)
        if event_row is not None:
            write_csv(pd.DataFrame([event_row]), out_dir / "csv" / "episode_events.csv")

    # Persist manifest.
    manifest = {
        "task_dir": str(task_dir),
        "episode_index": episode_index,
        "window": asdict(window),
        "source_video_fps": fps,
        "codec": codec,
        "output_video_ffprobe": maybe_ffprobe(trimmed_video_path),
        "notes": [
            "Relative trim times are interpreted relative to the first frame timestamp of the selected episode.",
            "The HDF5 trim uses the FastUMI baseline rule of selecting every third raw video frame to form the 20 Hz stream.",
        ],
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

    return TrimResult(
        label=window.label,
        out_dir=str(out_dir),
        video_frames=written_frames,
        traj_rows=len(trimmed_traj),
        downsampled_frames=len(trimmed_downsampled),
        has_hdf5=has_hdf5,
        has_states=has_states,
        relative_start_sec=window.relative_start_sec,
        relative_end_sec=window.relative_end_sec,
    )


def build_augmented_windows(base: TrimWindow, num: int, max_crop_sec: float, min_remaining_sec: float, seed: int) -> list[TrimWindow]:
    rng = random.Random(seed)
    windows: list[TrimWindow] = []
    for i in range(num):
        max_allowed = min(max_crop_sec, max(0.0, base.duration_sec - min_remaining_sec))
        if max_allowed <= 0:
            break
        delta = rng.uniform(0.0, max_allowed)
        start_rel = base.relative_start_sec + delta
        end_rel = base.relative_end_sec
        windows.append(
            TrimWindow(
                label=f"aug_startcrop_{i+1:02d}",
                absolute_start_ts=base.absolute_start_ts + delta,
                absolute_end_ts=base.absolute_end_ts,
                relative_start_sec=start_rel,
                relative_end_sec=end_rel,
                duration_sec=end_rel - start_rel,
            )
        )
    return windows


def main() -> None:
    args = parse_args()

    task_dir = args.task_dir.resolve()
    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory does not exist: {task_dir}")

    video_path = resolve_candidate(task_dir, args.episode_index, args.video, VIDEO_CANDIDATES, "video")
    video_ts_path = resolve_candidate(task_dir, args.episode_index, args.video_ts, VIDEO_TS_CANDIDATES, "video timestamps")
    traj_path = resolve_candidate(task_dir, args.episode_index, args.trajectory, TRAJ_CANDIDATES, "trajectory CSV")
    hdf5_path = resolve_candidate(task_dir, args.episode_index, args.hdf5, HDF5_CANDIDATES, "HDF5")
    states_path = args.states_csv if args.states_csv else (task_dir / "states.csv" if (task_dir / "states.csv").exists() else None)
    frame_ts_path = args.frame_timestamps_csv if args.frame_timestamps_csv else (task_dir / "csv" / "frame_timestamps.csv" if (task_dir / "csv" / "frame_timestamps.csv").exists() else None)
    episode_events_path = resolve_candidate(task_dir, args.episode_index, args.episode_events_csv, EVENT_CANDIDATES, "episode events CSV")

    if video_path is None:
        raise FileNotFoundError("Could not auto-detect raw video path. Provide --video explicitly.")
    if video_ts_path is None:
        raise FileNotFoundError(
            "Could not auto-detect raw video timestamp CSV. Provide --video-ts explicitly. "
            "Expected a per-episode file such as csv/temp_video_timestamps_<episode>.csv."
        )
    if traj_path is None:
        raise FileNotFoundError(
            "Could not auto-detect raw trajectory CSV. Provide --trajectory explicitly. "
            "Expected a per-episode file such as csv/temp_trajectory_<episode>.csv."
        )

    video_ts_df = load_csv_any(video_ts_path)
    ensure_columns(video_ts_df, ["Frame Index", "Timestamp"], video_ts_path)
    base_window = build_window(video_ts_df, args.start_sec, args.end_sec)
    fps = detect_video_fps(video_path, video_ts_df, args.video_fps)
    output_root = (args.output_root or (task_dir / "trimmed")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    windows = [base_window] + build_augmented_windows(
        base_window,
        args.num_random_start_crops,
        args.max_random_start_crop_sec,
        args.min_remaining_sec,
        args.seed,
    )

    results: list[TrimResult] = []
    for window in windows:
        result = trim_once(
            task_dir=task_dir,
            episode_index=args.episode_index,
            output_root=output_root,
            video_path=video_path,
            video_ts_path=video_ts_path,
            traj_path=traj_path,
            hdf5_path=hdf5_path,
            states_path=states_path,
            frame_ts_path=frame_ts_path,
            episode_events_path=episode_events_path,
            window=window,
            keep_absolute_timestamps=args.keep_absolute_timestamps,
            fps=fps,
            codec=args.codec,
            include_original_manifest=args.copy_original_manifest,
        )
        results.append(result)

    summary = {
        "task_dir": str(task_dir),
        "episode_index": args.episode_index,
        "input_files": {
            "video": str(video_path),
            "video_timestamps": str(video_ts_path),
            "trajectory": str(traj_path),
            "hdf5": str(hdf5_path) if hdf5_path else None,
            "states_csv": str(states_path) if states_path else None,
            "frame_timestamps_csv": str(frame_ts_path) if frame_ts_path else None,
            "episode_events_csv": str(episode_events_path) if episode_events_path else None,
        },
        "results": [asdict(r) for r in results],
        "recommendations": [
            "Use the base trim for your essential episode window.",
            "Only use random start-crop augmentations when the first cropped-away seconds are not themselves the core of the task.",
            "Prefer episode_events.csv when correlating trims back to Enter-triggered tracking and recording boundaries.",
        ],
    }
    (output_root / f"episode_{args.episode_index}_trim_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
