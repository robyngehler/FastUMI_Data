from __future__ import annotations

import argparse
import json
import os
import re
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Sequence

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / 'config' / 'config.json'


def natural_key(value: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', value)]


def load_processing_config(config_path: str | os.PathLike[str]):
    with open(config_path, 'r', encoding='utf-8') as config_file:
        config = json.load(config_file)
    return config['data_process_config']


def get_aruco_components(config):
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config['aruco_dict']))
    parameters = cv2.aruco.DetectorParameters()
    return aruco_dict, parameters


def get_gripper_width(image_source: Sequence, config):
    """
    Calculate gripper width from detected ArUco markers in the images.

    `image_source` can be either a numpy array or an h5py dataset. Frames are
    accessed one-by-one to avoid loading entire episodes into memory.
    """
    aruco_dict, parameters = get_aruco_components(config)
    distances = []
    distances_index = []
    current_frame = 0
    frame_count = len(image_source)
    for i in range(frame_count):
        current_frame += 1
        frame = image_source[i]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            
            marker_centers = []
            for idx, marker_id in enumerate(ids.flatten()):
                if marker_id in [config["marker_id_0"], config["marker_id_1"]]:
                    marker_corners = corners[idx][0]
                    center = np.mean(marker_corners, axis=0).astype(int)
                    marker_centers.append(center)
            
            if len(marker_centers) >= 2:
                distance = np.linalg.norm(marker_centers[0] - marker_centers[1])
                distances.append(distance)
                distances_index.append(current_frame)
            elif len(marker_centers) == 1:
                distance = abs(gray.shape[1] / 2 - marker_centers[0][0]) * 2
                distances.append(distance)
                distances_index.append(current_frame)

    if len(distances) == 0:
        raise ValueError(
            'No ArUco marker detections were found in this episode. '
            'Check marker visibility, marker IDs, and gripper scaling calibration.'
        )

    distances = np.array(distances)
    distances_index = np.array(distances_index)
    distances = ((distances - config["distances"]["marker_min"]) / (config["distances"]["marker_max"] - config["distances"]["marker_min"]) * config["distances"]["gripper_max"]).astype(np.int16).clip(0, config["distances"]["gripper_max"])
    
    new_distances = []
    for i in range(len(distances) - 1):
        if i == 0:
            if distances_index[i] == 1:
                new_distances.append(distances[0])
                continue
            else:
                for _ in range(distances_index[0]):
                    new_distances.append(distances[0])
        else:
            if distances_index[i + 1] - distances_index[i]==1:
                new_distances.append(distances[i])
            else:
                for k in range(distances_index[i + 1] - distances_index[i]):
                    interpolated_distance = int(
                        k * (distances[i + 1] - distances[i]) /
                        (distances_index[i + 1] - distances_index[i]) +
                        distances[i])
                    new_distances.append(interpolated_distance)
    new_distances.append(distances[-1])
    if len(new_distances) < frame_count:
        for _ in range(frame_count - len(new_distances)):
            new_distances.append(distances[-1])
    
    return np.array(new_distances)

def transform_to_base_quat(x, y, z, qx, qy, qz, qw, T_base_to_local):
    rotation_local = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T_local = np.eye(4)
    T_local[:3, :3] = rotation_local
    T_local[:3, 3] = [x, y, z]
    T_base_r = np.dot(T_local[:3, :3] , T_base_to_local[:3, :3] )
    x_base, y_base, z_base = T_base_to_local[:3, 3] + T_local[:3, 3]
    rotation_base = R.from_matrix(T_base_r)
    roll_base, pitch_base, yaw_base = rotation_base.as_euler('xyz', degrees=False)
    qx_base, qy_base, qz_base, qw_base = rotation_base.as_quat()
    
    return x_base, y_base, z_base, qx_base, qy_base, qz_base, qw_base, roll_base, pitch_base, yaw_base


def normalize_and_save_base_tcp_hdf5(args):
    input_file, output_file, config = args
    base_x, base_y, base_z = config["base_position"]["x"], config["base_position"]["y"], config["base_position"]["z"] # FastUMI TCP anchor position in the robot base frame (in meters)
    base_roll, base_pitch, base_yaw = np.deg2rad([config["base_orientation"]["roll"], config["base_orientation"]["pitch"], config["base_orientation"]["yaw"]]) # FastUMI TCP anchor orientation in the robot base frame (roll, pitch, yaw in degrees)
    rotation_base_to_local = R.from_euler('xyz', [base_roll, base_pitch, base_yaw]).as_matrix()
    
    T_base_to_local = np.eye(4)
    T_base_to_local[:3, :3] = rotation_base_to_local
    T_base_to_local[:3, 3] = [base_x, base_y, base_z]
    
    try:
        with h5py.File(input_file, 'r') as f_in:
            action_data = f_in['action'][:]
            qpos_data = f_in['observations/qpos'][:]
            image_dataset = f_in['observations/images/front']
            normalized_qpos = np.copy(qpos_data)

            for i in range(normalized_qpos.shape[0]):
                x, y, z, qx, qy, qz, qw = normalized_qpos[i, 0:7]
                x -= config["offset"]["x"]
                z += config["offset"]["z"]

                x_base, y_base, z_base, qx_base, qy_base, qz_base, qw_base, _, _, _ = transform_to_base_quat(x, y, z, qx, qy, qz, qw, T_base_to_local)
                ori = R.from_quat([qx_base, qy_base, qz_base, qw_base]).as_matrix()
                pos = np.array([x_base, y_base, z_base])
                pos += config["offset"]["x"] * ori[:, 2] 
                pos -= config["offset"]["z"] * ori[:, 0]
                x_base, y_base, z_base = pos
                normalized_qpos[i, :] = [x_base, y_base, z_base, qx_base, qy_base, qz_base, qw_base]

            gripper_open_width = get_gripper_width(image_dataset, config)
            gripper_open_width = gripper_open_width / config["distances"]["gripper_max"]

            gripper_width = gripper_open_width.reshape(-1, 1)
            normalized_qpos_with_gripper = np.concatenate((normalized_qpos, gripper_width), axis=1)
            normalized_action_with_gripper = np.copy(normalized_qpos_with_gripper)

            with h5py.File(output_file, 'w') as f_out:
                f_out.create_dataset('action', data=normalized_action_with_gripper)
                observations_group = f_out.create_group('observations')
                images_group = observations_group.create_group('images')
                
                max_timesteps = f_in['observations/images/front'].shape[0]
                cam_hight, cam_width = f_in['observations/images/front'].shape[1], f_in['observations/images/front'].shape[2]

                images_group.create_dataset(
                    'front',
                    (max_timesteps, cam_hight, cam_width, 3),
                    dtype='uint8',
                    chunks=(1, cam_hight, cam_width, 3),
                    compression='gzip',
                    compression_opts=4
                )
                for frame_idx in range(max_timesteps):
                    images_group['front'][frame_idx] = image_dataset[frame_idx]
                observations_group.create_dataset('qpos', data=normalized_qpos_with_gripper)
                                
                print(f"Normalized data saved to: {output_file}")
    except Exception as e:
        print(f"Error processing {input_file}: {e}")


def sorted_hdf5_files(input_dir: Path):
    files = [path for path in input_dir.iterdir() if path.suffix == '.hdf5']
    return sorted(files, key=lambda path: natural_key(path.name))


def process_single_directory(input_dir: Path, output_dir: Path, config, num_processes: int | None = None):
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    file_list = sorted_hdf5_files(input_dir)
    if not file_list:
        raise FileNotFoundError(f'No .hdf5 files found in {input_dir}')

    args_list = []
    for input_file in file_list:
        output_file = output_dir / input_file.name
        args_list.append((str(input_file), str(output_file), config))

    if num_processes is None:
        # Default to serial processing. Episodes are image-heavy, and the old
        # cpu_count() default could easily exhaust RAM by loading many full HD
        # episodes at once across worker processes.
        num_processes = 1

    print(f'Processing {len(args_list)} episodes: {input_dir} -> {output_dir}')
    if num_processes <= 1:
        for task in tqdm(args_list, total=len(args_list), desc=f'Processing {input_dir.name}'):
            normalize_and_save_base_tcp_hdf5(task)
        return

    with Pool(num_processes, maxtasksperchild=1) as pool:
        list(
            tqdm(
                pool.imap_unordered(normalize_and_save_base_tcp_hdf5, args_list, chunksize=1),
                total=len(args_list),
                desc=f'Processing {input_dir.name}'
            )
        )


def resolve_group_jobs(args, config):
    input_dir = Path(args.input_dir).expanduser().resolve() if args.input_dir else None
    input_root = Path(args.input_root).expanduser().resolve() if args.input_root else None

    if input_dir is not None and input_root is not None:
        raise ValueError('Use either --input-dir or --input-root, not both.')

    if input_root is not None:
        output_root = Path(args.output_root).expanduser().resolve() if args.output_root else input_root.with_name(f'{input_root.name}_tcp')
        group_dirs = sorted(
            [path for path in input_root.glob(args.group_glob) if path.is_dir()],
            key=lambda path: natural_key(path.name)
        )
        if not group_dirs:
            raise FileNotFoundError(f'No group directories matching {args.group_glob!r} found in {input_root}')
        return [(group_dir, output_root / group_dir.name) for group_dir in group_dirs]

    if input_dir is None:
        input_dir = Path(config['input_dir']).expanduser().resolve()

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else Path(config['output_tcp_dir']).expanduser().resolve()
    return [(input_dir, output_dir)]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert FastUMI raw HDF5 episodes to TCP-space HDF5 files compatible with the downstream UMI-style Zarr conversion.'
    )
    parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH), help='Path to FastUMI config.json')
    parser.add_argument('--input-dir', help='Single input directory containing episode_*.hdf5 files.')
    parser.add_argument('--output-dir', help='Output directory for a single converted TCP group.')
    parser.add_argument('--input-root', help='Root directory that contains multiple group subdirectories, e.g. pour_coke_v0, pour_coke_v1, ...')
    parser.add_argument('--output-root', help='Output root for grouped conversion. Each input group is written to output-root/<group-name>.')
    parser.add_argument('--group-glob', default='*', help='Glob used with --input-root to select group directories.')
    parser.add_argument('--num-processes', type=int, default=None, help='Worker process count. Defaults to 1 for RAM safety. Increase only if you have headroom.')
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    config = load_processing_config(cli_args.config)
    jobs = resolve_group_jobs(cli_args, config)

    print(f'Starting TCP conversion for {len(jobs)} group(s).')
    for input_dir, output_dir in jobs:
        process_single_directory(input_dir, output_dir, config, num_processes=cli_args.num_processes)
    print('Processing completed.')