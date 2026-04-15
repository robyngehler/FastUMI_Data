#!/usr/bin/env python3
"""
Capture Panda anchor state using the FastUMI-specific client-side frame convention.

Connection path matches the working Franka deployment:
inference machine -> ZeroRPC bridge on control machine -> local Polymetis services.

The raw server pose is treated as panda_link8. This script then derives:
- the operational panda_hand_tcp-aligned pose using the fixed URDF transform,
- FastUMI TCP pose using a translation-only soft-tip offset along local hand Z.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import zerorpc

from panda_fastumi_frames import (
    DEFAULT_SOFT_TIP_OFFSET_M,
    LINK8_TO_HAND_ROTATION_DEG_Z,
    LINK8_TO_HAND_TRANSLATION_M,
    link8_pose_to_fastumi_tcp_pose,
    link8_pose_to_hand_pose,
    pose_to_dict,
)


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
    parser = argparse.ArgumentParser(description="Capture Panda anchor pose using the FastUMI frame convention.")
    parser.add_argument("--server-ip", default="localhost", help="Control-side ZeroRPC bridge host.")
    parser.add_argument("--server-port", type=int, default=4242, help="Control-side ZeroRPC bridge port.")
    parser.add_argument("--label", default="panda_anchor_pose_fastumi", help="Label stored in the JSON output.")
    parser.add_argument("--output", default="panda_anchor_pose_fastumi.json", help="Where to write the JSON result.")
    parser.add_argument(
        "--soft-tip-offset-m",
        type=float,
        default=DEFAULT_SOFT_TIP_OFFSET_M,
        help="Translation-only offset from panda_hand_tcp to the chosen FastUMI TCP along local hand Z.",
    )
    parser.add_argument(
        "--with-gripper",
        action="store_true",
        help="Also try to read the gripper state via the ZeroRPC bridge.",
    )
    args = parser.parse_args()

    client = zerorpc.Client(heartbeat=20)
    try:
        client.connect(f"tcp://{args.server_ip}:{args.server_port}")

        print("\nBring the Panda to the desired FastUMI reference pose manually.")
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
        raw_link8_pose = np.asarray(to_list(client.get_ee_pose()), dtype=np.float64)
        hand_pose = link8_pose_to_hand_pose(raw_link8_pose)
        fastumi_tcp_pose = link8_pose_to_fastumi_tcp_pose(raw_link8_pose, offset_m=args.soft_tip_offset_m)

        result: Dict[str, Any] = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "label": args.label,
            "rpc_connection": {
                "ip": args.server_ip,
                "port": args.server_port,
                "path": "inference -> zerorpc bridge -> local polymetis",
            },
            "frame_metadata": {
                "raw_runtime_frame": "panda_link8",
                "ik_target_frame": "panda_hand_tcp",
                "fastumi_tcp_frame": "panda_softtip",
                "link8_to_hand_rotation_deg_z": LINK8_TO_HAND_ROTATION_DEG_Z,
                "link8_to_hand_translation_m": LINK8_TO_HAND_TRANSLATION_M,
                "soft_tip_offset_m": args.soft_tip_offset_m,
            },
            "start_qpos": joint_positions.tolist(),
            "raw_link8_pose": pose_to_dict(raw_link8_pose),
            "hand_pose": pose_to_dict(hand_pose),
            "fastumi_tcp_pose": pose_to_dict(fastumi_tcp_pose),
            "fastumi_relevant_values": {
                "start_qpos": joint_positions.tolist(),
                "robot_anchor_tcp_position_m": fastumi_tcp_pose[:3].tolist(),
                "robot_anchor_tcp_rotvec_xyz": fastumi_tcp_pose[3:].tolist(),
                "robot_anchor_tcp_quaternion_xyzw": pose_to_dict(fastumi_tcp_pose)["quaternion_xyzw"],
                "robot_anchor_tcp_euler_xyz_deg": pose_to_dict(fastumi_tcp_pose)["euler_xyz_deg"],
            },
            "notes": [
                "Raw bridge pose is treated as panda_link8.",
                "The local operational hand frame includes the fixed 0.1034 m hand_tcp extension from the URDF.",
                "FastUMI IK target frame is the operational panda_hand_tcp-aligned frame.",
                "FastUMI TCP keeps panda_hand_tcp orientation and applies only a forward local-Z translation.",
                "Use the FastUMI TCP pose for base_position/base_orientation in the Panda recording config.",
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

        print("\nRecorded Panda FastUMI anchor pose:")
        print(json.dumps(result, indent=2))
        print(f"\nSaved to: {output_path.resolve()}")
    finally:
        client.close()


if __name__ == "__main__":
    main()