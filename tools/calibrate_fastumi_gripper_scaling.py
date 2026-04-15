#!/usr/bin/env python3
"""
Interactive helper to calibrate FastUMI-style gripper scaling values from a ROS image topic.

What it does:
- subscribes to a ROS image topic,
- detects two configured ArUco markers,
- lets the user capture OPEN and CLOSED marker distances with keypresses,
- computes robust medians,
- writes a JSON snippet for FastUMI config values.

This is intentionally stricter than the public FastUMI processing code:
for calibration we only accept frames where BOTH markers are visible.
"""

from __future__ import annotations

import argparse
import json
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np

try:
    import rospy
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "This script requires ROS (rospy), sensor_msgs and cv_bridge in the active environment."
    ) from exc


@dataclass
class CaptureSummary:
    count: int
    median_px: Optional[float]
    mean_px: Optional[float]
    std_px: Optional[float]
    min_px: Optional[float]
    max_px: Optional[float]


class LatestImageSubscriber:
    def __init__(self, topic: str):
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._stamp: Optional[float] = None
        self._sub = rospy.Subscriber(topic, Image, self._callback, queue_size=1)

    def _callback(self, msg: Image) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logwarn_throttle(2.0, f"cv_bridge conversion failed: {exc}")
            return
        with self._lock:
            self._frame = frame.copy()
            self._stamp = msg.header.stamp.to_sec() if msg.header.stamp else rospy.Time.now().to_sec()

    def get_latest(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        with self._lock:
            if self._frame is None:
                return None, None
            return self._frame.copy(), self._stamp


def get_aruco_dict(name: str):
    if not hasattr(cv2.aruco, name):
        valid = [k for k in dir(cv2.aruco) if k.startswith("DICT_")]
        raise ValueError(f"Unknown ArUco dictionary '{name}'. Valid examples: {valid[:10]} ...")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def detect_marker_distance(
    frame_bgr: np.ndarray,
    aruco_dict,
    marker_id_0: int,
    marker_id_1: int,
) -> Tuple[Optional[float], np.ndarray, dict]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    parameters = cv2.aruco.DetectorParameters()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    annotated = frame_bgr.copy()
    found = {}

    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
        for idx, marker_id in enumerate(ids.flatten()):
            if marker_id in (marker_id_0, marker_id_1):
                center = np.mean(corners[idx][0], axis=0)
                found[int(marker_id)] = center
                cx, cy = int(round(center[0])), int(round(center[1]))
                cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(
                    annotated,
                    f"ID {int(marker_id)}",
                    (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

    distance_px = None
    if marker_id_0 in found and marker_id_1 in found:
        p0 = found[marker_id_0]
        p1 = found[marker_id_1]
        distance_px = float(np.linalg.norm(p0 - p1))
        cv2.line(
            annotated,
            (int(round(p0[0])), int(round(p0[1]))),
            (int(round(p1[0])), int(round(p1[1]))),
            (0, 255, 255),
            2,
        )

    return distance_px, annotated, found


def summarize(values: List[float]) -> CaptureSummary:
    if not values:
        return CaptureSummary(0, None, None, None, None, None)
    arr = np.asarray(values, dtype=np.float64)
    return CaptureSummary(
        count=int(arr.size),
        median_px=float(np.median(arr)),
        mean_px=float(np.mean(arr)),
        std_px=float(np.std(arr)),
        min_px=float(np.min(arr)),
        max_px=float(np.max(arr)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate FastUMI marker_min / marker_max from a ROS image topic.")
    parser.add_argument("--image-topic", default="/usb_cam/image_raw", help="ROS image topic to subscribe to.")
    parser.add_argument("--marker-id-0", type=int, default=0, help="First ArUco marker ID.")
    parser.add_argument("--marker-id-1", type=int, default=1, help="Second ArUco marker ID.")
    parser.add_argument("--aruco-dict", default="DICT_4X4_50", help="OpenCV ArUco dictionary name.")
    parser.add_argument(
        "--gripper-max",
        type=float,
        default=None,
        help="Optional physical max gripper opening for the config snippet, e.g. 0.08 for 80 mm if you use meters.",
    )
    parser.add_argument(
        "--output",
        default="fastumi_gripper_scaling_calibration.json",
        help="Where to write the JSON result.",
    )
    args = parser.parse_args()

    rospy.init_node("fastumi_gripper_scaling_calibration", anonymous=True)
    aruco_dict = get_aruco_dict(args.aruco_dict)
    image_sub = LatestImageSubscriber(args.image_topic)

    open_samples: List[float] = []
    closed_samples: List[float] = []

    print("\nInteractive controls")
    print("  o : capture current marker distance as OPEN sample")
    print("  c : capture current marker distance as CLOSED sample")
    print("  p : print current statistics")
    print("  r : reset all samples")
    print("  s : save result JSON")
    print("  q : quit")
    print("\nTip: keep BOTH markers fully visible during calibration. One-marker fallback is intentionally disabled here.\n")

    while not rospy.is_shutdown():
        frame, stamp = image_sub.get_latest()
        if frame is None:
            rospy.sleep(0.03)
            continue

        distance_px, annotated, found = detect_marker_distance(
            frame,
            aruco_dict,
            args.marker_id_0,
            args.marker_id_1,
        )

        status = (
            f"stamp={stamp:.3f}  markers={sorted(found.keys())}  "
            + (f"distance={distance_px:.2f}px" if distance_px is not None else "distance=NA")
        )
        cv2.putText(
            annotated,
            status,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"OPEN samples={len(open_samples)}  CLOSED samples={len(closed_samples)}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("FastUMI gripper scaling calibration", annotated)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("r"):
            open_samples.clear()
            closed_samples.clear()
            print("Samples reset.")
            continue
        if key == ord("p"):
            print("OPEN  :", asdict(summarize(open_samples)))
            print("CLOSED:", asdict(summarize(closed_samples)))
            continue

        if key in (ord("o"), ord("c")):
            if distance_px is None:
                print("Skipped capture: both markers must be visible in the current frame.")
                continue
            if key == ord("o"):
                open_samples.append(distance_px)
                print(f"Added OPEN sample: {distance_px:.3f} px")
            else:
                closed_samples.append(distance_px)
                print(f"Added CLOSED sample: {distance_px:.3f} px")
            continue

        if key == ord("s"):
            open_summary = summarize(open_samples)
            closed_summary = summarize(closed_samples)
            if open_summary.count == 0 or closed_summary.count == 0:
                print("Need at least one OPEN and one CLOSED sample before saving.")
                continue
            if open_summary.median_px is None or closed_summary.median_px is None:
                print("Unexpected empty summaries; not saving.")
                continue
            if open_summary.median_px <= closed_summary.median_px:
                print(
                    "Warning: OPEN median is not larger than CLOSED median. "
                    "That usually means labels were swapped or detection is wrong."
                )

            result = {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "image_topic": args.image_topic,
                "marker_ids": [args.marker_id_0, args.marker_id_1],
                "aruco_dict": args.aruco_dict,
                "open_summary": asdict(open_summary),
                "closed_summary": asdict(closed_summary),
                "fastumi_config_snippet": {
                    "distances": {
                        "marker_max": open_summary.median_px,
                        "marker_min": closed_summary.median_px,
                        "gripper_max": args.gripper_max,
                    }
                },
                "notes": [
                    "FastUMI maps marker distance linearly from marker_min..marker_max to gripper_max.",
                    "Use the medians as a robust starting point, then validate on real episodes.",
                    "For calibration this script only accepts frames with BOTH markers visible.",
                ],
            }
            output_path = Path(args.output)
            output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"Saved calibration result to: {output_path.resolve()}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
