from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


LINK8_TO_HAND_ROTATION_DEG_Z = -45.0
LINK8_TO_HAND_TRANSLATION_M = 0.1034
DEFAULT_SOFT_TIP_OFFSET_M = 0.09


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


def link8_to_hand_transform() -> np.ndarray:
    transform = np.identity(4)
    transform[:3, :3] = R.from_euler("z", LINK8_TO_HAND_ROTATION_DEG_Z, degrees=True).as_matrix()
    transform[:3, 3] = np.array([0.0, 0.0, LINK8_TO_HAND_TRANSLATION_M], dtype=np.float64)
    return transform


def hand_to_softtip_transform(offset_m: float = DEFAULT_SOFT_TIP_OFFSET_M) -> np.ndarray:
    transform = np.identity(4)
    transform[:3, 3] = np.array([0.0, 0.0, float(offset_m)], dtype=np.float64)
    return transform


def link8_pose_to_hand_pose(link8_pose: np.ndarray) -> np.ndarray:
    return mat_to_pose(pose_to_mat(link8_pose) @ link8_to_hand_transform())


def hand_pose_to_fastumi_tcp_pose(hand_pose: np.ndarray, offset_m: float = DEFAULT_SOFT_TIP_OFFSET_M) -> np.ndarray:
    return mat_to_pose(pose_to_mat(hand_pose) @ hand_to_softtip_transform(offset_m))


def fastumi_tcp_pose_to_hand_pose(tcp_pose: np.ndarray, offset_m: float = DEFAULT_SOFT_TIP_OFFSET_M) -> np.ndarray:
    return mat_to_pose(pose_to_mat(tcp_pose) @ np.linalg.inv(hand_to_softtip_transform(offset_m)))


def link8_pose_to_fastumi_tcp_pose(link8_pose: np.ndarray, offset_m: float = DEFAULT_SOFT_TIP_OFFSET_M) -> np.ndarray:
    return mat_to_pose(
        pose_to_mat(link8_pose)
        @ link8_to_hand_transform()
        @ hand_to_softtip_transform(offset_m)
    )


def fastumi_tcp_pose_to_link8_pose(tcp_pose: np.ndarray, offset_m: float = DEFAULT_SOFT_TIP_OFFSET_M) -> np.ndarray:
    return mat_to_pose(
        pose_to_mat(tcp_pose)
        @ np.linalg.inv(hand_to_softtip_transform(offset_m))
        @ np.linalg.inv(link8_to_hand_transform())
    )


def pose_to_dict(pose: np.ndarray) -> dict:
    pose = np.asarray(pose, dtype=np.float64)
    quat = R.from_rotvec(pose[3:]).as_quat()
    euler = R.from_quat(quat).as_euler("xyz", degrees=True)
    return {
        "position_m": pose[:3].tolist(),
        "rotvec_xyz": pose[3:].tolist(),
        "quaternion_xyzw": quat.tolist(),
        "euler_xyz_deg": euler.tolist(),
    }