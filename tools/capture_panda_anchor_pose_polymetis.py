#!/usr/bin/env python3
"""
Capture the current Panda joint configuration and TCP pose from a running Polymetis server.

Typical workflow:
1. put the Panda into the desired reference pose manually / user-stop mode,
2. run this script,
3. press Enter to record,
4. save a JSON file containing start_qpos and TCP pose.

This is deliberately conservative: it only reads the current state and does not move the robot.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R


def to_list(x) -> list:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, dtype=np.float64).tolist()


def maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture Panda anchor pose from Polymetis.")
    parser.add_argument("--robot-ip", default="localhost", help="Polymetis controller manager host.")
    parser.add_argument("--robot-port", type=int, default=50051, help="Polymetis robot port.")
    parser.add_argument("--gripper-port", type=int, default=50052, help="Polymetis gripper port.")
    parser.add_argument("--label", default="panda_anchor_pose", help="Label stored in the JSON output.")
    parser.add_argument("--output", default="panda_anchor_pose.json", help="Where to write the JSON result.")
    parser.add_argument(
        "--with-gripper",
        action="store_true",
        help="Also try to read the gripper state from the Polymetis gripper server.",
    )
    parser.add_argument(
        "--enforce-version",
        action="store_true",
        help="Enable Polymetis client/server version enforcement. Disabled by default for convenience.",
    )
    args = parser.parse_args()

    from polymetis import RobotInterface

    robot = RobotInterface(
        ip_address=args.robot_ip,
        port=args.robot_port,
        enforce_version=args.enforce_version,
    )

    gripper = None
    if args.with_gripper:
        try:
            from polymetis import GripperInterface

            gripper = GripperInterface(ip_address=args.robot_ip, port=args.gripper_port)
        except Exception as exc:
            print(f"Warning: could not connect to gripper server: {exc}")
            gripper = None

    print("\nBring the Panda to the desired reference pose manually.")
    print("When the robot is stationary, press Enter to record the current state.")
    print("Type 'q' and press Enter to abort.\n")

    while True:
        user = input("Press Enter to record current pose, or 'q' to quit: ").strip().lower()
        if user == "q":
            print("Aborted without recording.")
            return
        if user == "":
            break

    joint_positions = np.asarray(to_list(robot.get_joint_positions()), dtype=np.float64)
    ee_position, ee_quaternion = robot.get_ee_pose()
    ee_position = np.asarray(to_list(ee_position), dtype=np.float64)
    ee_quaternion = np.asarray(to_list(ee_quaternion), dtype=np.float64)
    ee_rpy_deg = R.from_quat(ee_quaternion).as_euler("xyz", degrees=True)

    result: Dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "robot_connection": {
            "ip": args.robot_ip,
            "robot_port": args.robot_port,
            "gripper_port": args.gripper_port,
        },
        "start_qpos": joint_positions.tolist(),
        "tcp_pose": {
            "position_m": ee_position.tolist(),
            "quaternion_xyzw": ee_quaternion.tolist(),
            "euler_xyz_deg": ee_rpy_deg.tolist(),
        },
        "fastumi_relevant_values": {
            "start_qpos": joint_positions.tolist(),
            "robot_anchor_tcp_position_m": ee_position.tolist(),
            "robot_anchor_tcp_euler_xyz_deg": ee_rpy_deg.tolist(),
        },
        "notes": [
            "This records the Panda-side reference pose only.",
            "It does not measure the teaching-device dock pose or T265-to-tool offset.",
            "Use this as the anchor pose you want the teaching-device TCP/gripper-center to correspond to at initialization.",
        ],
    }

    if gripper is not None:
        try:
            gstate = gripper.get_state()
            result["gripper_state"] = {
                "width": maybe_float(getattr(gstate, "width", None)),
                "max_width": maybe_float(getattr(gstate, "max_width", None)),
                "is_grasped": getattr(gstate, "is_grasped", None),
            }
        except Exception as exc:
            result["gripper_state_error"] = str(exc)

    output_path = Path(args.output)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\nRecorded Panda anchor pose:")
    print(json.dumps(result, indent=2))
    print(f"\nSaved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
