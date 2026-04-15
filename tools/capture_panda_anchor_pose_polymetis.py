#!/usr/bin/env python3
"""
Capture the current Panda joint configuration and TCP pose via the Franka ZeroRPC bridge.

Typical workflow:
1. put the Panda into the desired reference pose manually / user-stop mode,
2. run this script,
3. press Enter to record,
4. save a JSON file containing start_qpos and TCP pose.

This is deliberately conservative: it only reads the current state and does not move the robot.

Connection path matches the working fastumi_dp deployment:
inference machine -> ZeroRPC bridge on control machine -> local Polymetis services.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R
import zerorpc


tx_flange_flangerot45 = np.identity(4)
tx_flange_flangerot45[:3, :3] = R.from_euler("z", [-3 * np.pi / 4]).as_matrix()

tx_flangerot45_tip = np.identity(4)
tx_flangerot45_tip[:3, 3] = np.array([0, 0, 0.18832])

tx_flange_tip = tx_flange_flangerot45 @ tx_flangerot45_tip


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


def pose_to_mat(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    transform = np.identity(4)
    transform[:3, :3] = R.from_rotvec(pose[3:]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def mat_to_pose(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    return np.concatenate([
        transform[:3, 3],
        R.from_matrix(transform[:3, :3]).as_rotvec(),
    ])


def flange_pose_to_tip_pose(flange_pose: np.ndarray) -> np.ndarray:
    return mat_to_pose(pose_to_mat(flange_pose) @ tx_flange_tip)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture Panda anchor pose via the Franka ZeroRPC bridge.")
    parser.add_argument("--server-ip", default="localhost", help="Control-side ZeroRPC bridge host.")
    parser.add_argument("--robot-ip", dest="server_ip", help=argparse.SUPPRESS)
    parser.add_argument("--server-port", type=int, default=4242, help="Control-side ZeroRPC bridge port.")
    parser.add_argument("--label", default="panda_anchor_pose", help="Label stored in the JSON output.")
    parser.add_argument("--output", default="panda_anchor_pose.json", help="Where to write the JSON result.")
    parser.add_argument(
        "--with-gripper",
        action="store_true",
        help="Also try to read the gripper state via the ZeroRPC bridge.",
    )
    args = parser.parse_args()

    client = zerorpc.Client(heartbeat=20)
    try:
        client.connect(f"tcp://{args.server_ip}:{args.server_port}")

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

        joint_positions = np.asarray(to_list(client.get_joint_positions()), dtype=np.float64)
        flange_pose = np.asarray(to_list(client.get_ee_pose()), dtype=np.float64)
        tip_pose = flange_pose_to_tip_pose(flange_pose)
        ee_position = tip_pose[:3]
        ee_quaternion = R.from_rotvec(tip_pose[3:]).as_quat()
        ee_rpy_deg = R.from_quat(ee_quaternion).as_euler("xyz", degrees=True)

        result: Dict[str, Any] = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "label": args.label,
            "rpc_connection": {
                "ip": args.server_ip,
                "port": args.server_port,
                "path": "inference -> zerorpc bridge -> local polymetis",
            },
            "start_qpos": joint_positions.tolist(),
            "tcp_pose": {
                "position_m": ee_position.tolist(),
                "quaternion_xyzw": ee_quaternion.tolist(),
                "euler_xyz_deg": ee_rpy_deg.tolist(),
            },
            "franka_bridge_frames": {
                "bridge_pose_frame": "flange",
                "recorded_pose_frame": "tip",
            },
            "fastumi_relevant_values": {
                "start_qpos": joint_positions.tolist(),
                "robot_anchor_tcp_position_m": ee_position.tolist(),
                "robot_anchor_tcp_euler_xyz_deg": ee_rpy_deg.tolist(),
            },
            "notes": [
                "This records the Panda-side reference pose only.",
                "It does not measure the teaching-device dock pose or T265-to-tool offset.",
                "The ZeroRPC bridge returns a flange pose; this script converts it to the tool-tip pose used by fastumi_dp.",
                "Use this as the anchor pose you want the teaching-device TCP/gripper-center to correspond to at initialization.",
            ],
        }

        if args.with_gripper:
            try:
                gstate = client.get_gripper_state()
                result["gripper_state"] = {
                    "position": maybe_float(gstate.get("position")),
                    "width": maybe_float(gstate.get("position")),
                    "is_moving": gstate.get("is_moving"),
                    "prev_command_successful": gstate.get("prev_command_successful"),
                    "error_code": gstate.get("error_code"),
                }
            except Exception as exc:
                result["gripper_state_error"] = str(exc)

        output_path = Path(args.output)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        print("\nRecorded Panda anchor pose:")
        print(json.dumps(result, indent=2))
        print(f"\nSaved to: {output_path.resolve()}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
