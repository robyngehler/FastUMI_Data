from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
import zarr
from scipy.spatial.transform import Rotation

from imagecodecs_numcodecs import JpegXl, register_codecs
from replay_buffer import ReplayBuffer

register_codecs()

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / 'config' / 'config.json'
TCP_POSE_DIM = 7


def natural_key(value: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', value)]


def load_processing_config(config_path: str | os.PathLike[str]):
    with open(config_path, 'r', encoding='utf-8') as config_file:
        config = json.load(config_file)
    return config['data_process_config']


def split_tcp_action(action_data):
    if action_data.shape[1] < TCP_POSE_DIM + 1:
        raise ValueError(
            f"Expected TCP HDF5 action data to contain at least {TCP_POSE_DIM + 1} columns "
            f"([x, y, z, qx, qy, qz, qw, gripper]), but got shape {action_data.shape}."
        )

    pos = np.asarray(action_data[:, :3])
    rot_quat_xyzw = np.asarray(action_data[:, 3:TCP_POSE_DIM])
    gripper_width = np.asarray(action_data[:, -1])
    return pos, rot_quat_xyzw, gripper_width


def mat_to_pos_rot(mat):
    pos = (mat[..., :3, 3].T / mat[..., 3, 3].T).T
    rot = Rotation.from_matrix(mat[..., :3, :3])
    return pos, rot


def pos_rot_to_pose(pos, rot):
    shape = pos.shape[:-1]
    pose = np.zeros(shape + (6,), dtype=pos.dtype)
    pose[..., :3] = pos
    pose[..., 3:] = rot.as_rotvec()
    return pose


def mat_to_pose(mat):
    return pos_rot_to_pose(*mat_to_pos_rot(mat))


def get_image_transform(in_res, out_res, crop_ratio: float = 1.0, bgr_to_rgb: bool = False):
    iw, ih = in_res
    ow, oh = out_res
    ch = round(ih * crop_ratio)
    cw = round(ih * crop_ratio / oh * ow)
    interp_method = cv2.INTER_AREA

    w_slice_start = (iw - cw) // 2
    w_slice = slice(w_slice_start, w_slice_start + cw)
    h_slice_start = (ih - ch) // 2
    h_slice = slice(h_slice_start, h_slice_start + ch)
    c_slice = slice(None)
    if bgr_to_rgb:
        c_slice = slice(None, None, -1)

    def transform(img: np.ndarray):
        assert img.shape == (ih, iw, 3)
        img = img[h_slice, w_slice, c_slice]
        img = cv2.resize(img, out_res, interpolation=interp_method)
        return img

    return transform


def sorted_hdf5_files(input_dir: Path):
    files = [path for path in input_dir.iterdir() if path.suffix == '.hdf5']
    return sorted(files, key=lambda path: natural_key(path.name))


def resolve_group_dirs(config, input_dir=None, input_root=None, group_glob='*'):
    if input_dir and input_root:
        raise ValueError('Use either --input-dir or --input-root, not both.')

    if input_root:
        root = Path(input_root).expanduser().resolve()
        group_dirs = sorted([path for path in root.glob(group_glob) if path.is_dir()], key=lambda path: natural_key(path.name))
        if not group_dirs:
            raise FileNotFoundError(f'No group directories matching {group_glob!r} found in {root}')
        return group_dirs

    if input_dir:
        return [Path(input_dir).expanduser().resolve()]

    return [Path(config['output_tcp_dir']).expanduser().resolve()]


def validate_group_episode_counts(group_specs, allow_incomplete_groups=False):
    counts = [group_spec['episode_count'] for group_spec in group_specs]
    if not counts:
        return

    max_count = max(counts)
    if all(count == max_count for count in counts):
        return

    short_groups = [group_spec for group_spec in group_specs if group_spec['episode_count'] != max_count]
    if len(short_groups) == 1 and short_groups[0]['name'] == group_specs[-1]['name'] and short_groups[0]['episode_count'] < max_count:
        return

    if not allow_incomplete_groups:
        bad = [(group_spec['name'], group_spec['episode_count']) for group_spec in short_groups]
        raise ValueError(
            'Inconsistent TCP group sizes detected. '
            f'Most groups have {max_count} episodes, but these differ: {bad}. '
            'This usually means tcp conversion did not finish cleanly. '
            'Re-run data_processing_to_tcp.py or pass --allow-incomplete-groups if this is intentional.'
        )


def prepare_episode_from_file(hdf5_path: Path):
    with h5py.File(hdf5_path, 'r') as handle:
        action_data = np.asarray(handle['action'])
        pos, rot_quat_xyzw, gripper_width = split_tcp_action(action_data)

    rot = Rotation.from_quat(rot_quat_xyzw)
    pose = np.zeros((pos.shape[0], 4, 4), dtype=np.float32)
    pose[:, 3, 3] = 1
    pose[:, :3, 3] = pos
    pose[:, :3, :3] = rot.as_matrix()
    pose = mat_to_pose(pose).astype(np.float32)

    demo_start_pose = pose[0].astype(np.float32)
    demo_end_pose = pose[-1].astype(np.float32)

    grippers_dict = {
        'tcp_pose': pose,
        'gripper_width': np.asarray(gripper_width, dtype=np.float32),
        'demo_start_pose': demo_start_pose,
        'demo_end_pose': demo_end_pose,
    }
    return {'grippers': [grippers_dict]}


def infer_input_resolution(group_dirs):
    for group_dir in group_dirs:
        files = sorted_hdf5_files(group_dir)
        for hdf5_path in files:
            with h5py.File(hdf5_path, 'r') as handle:
                img = handle['observations/images/front']
                if img.ndim != 4 or img.shape[-1] != 3:
                    raise ValueError(f'Expected observations/images/front to have shape (T,H,W,3), got {img.shape} in {hdf5_path}')
                return (img.shape[2], img.shape[1])
    raise FileNotFoundError('Could not infer image resolution because no input HDF5 files were found.')


def convert_groups_to_zarr(
    group_dirs,
    output_path,
    out_res,
    compression_level,
    overwrite=False,
    count_path=None,
    manifest_path=None,
    cleanup_input_groups=False,
    allow_incomplete_groups=False,
):
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_store_path = output_path.parent / f'.{output_path.stem}_tmp.zarr'

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f'Output file already exists: {output_path}. Pass --overwrite to replace it.')
        output_path.unlink()
    if temp_store_path.exists():
        shutil.rmtree(temp_store_path)

    count_path = Path(count_path).expanduser().resolve() if count_path else output_path.parent / 'count.txt'
    manifest_path = Path(manifest_path).expanduser().resolve() if manifest_path else output_path.parent / 'conversion_manifest.json'
    count_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    group_specs = []
    for group_dir in group_dirs:
        hdf5_files = sorted_hdf5_files(group_dir)
        if not hdf5_files:
            raise FileNotFoundError(f'No .hdf5 files found in {group_dir}')
        group_specs.append({'name': group_dir.name, 'dir': group_dir, 'files': hdf5_files, 'episode_count': len(hdf5_files)})

    validate_group_episode_counts(group_specs, allow_incomplete_groups=allow_incomplete_groups)

    print(f'Converting {len(group_specs)} TCP group(s) into {output_path}')
    print(f'Using temporary on-disk store: {temp_store_path}')
    print(f'Preparing low-dim data for {sum(item["episode_count"] for item in group_specs)} episode(s) total')

    cv2.setNumThreads(1)
    out_replay_buffer = ReplayBuffer.create_empty_zarr(storage=zarr.DirectoryStore(str(temp_store_path)))

    n_grippers = None
    total_episodes = 0
    for group_spec in group_specs:
        print(f'Loading low-dim group {group_spec["name"]}: {group_spec["episode_count"]} episode(s)')
        for hdf5_path in group_spec['files']:
            plan_episode = prepare_episode_from_file(hdf5_path)
            grippers = plan_episode['grippers']
            if n_grippers is None:
                n_grippers = len(grippers)
            else:
                assert n_grippers == len(grippers)

            episode_data = {}
            for gripper_id, gripper in enumerate(grippers):
                eef_pose = gripper['tcp_pose']
                eef_pos = eef_pose[..., :3]
                eef_rot = eef_pose[..., 3:]
                gripper_widths = gripper['gripper_width']
                demo_start_pose = np.empty_like(eef_pose)
                demo_start_pose[:] = gripper['demo_start_pose']
                demo_end_pose = np.empty_like(eef_pose)
                demo_end_pose[:] = gripper['demo_end_pose']

                robot_name = f'robot{gripper_id}'
                episode_data[robot_name + '_eef_pos'] = eef_pos.astype(np.float32)
                episode_data[robot_name + '_eef_rot_axis_angle'] = eef_rot.astype(np.float32)
                episode_data[robot_name + '_gripper_width'] = np.expand_dims(gripper_widths, axis=-1).astype(np.float32)
                episode_data[robot_name + '_demo_start_pose'] = demo_start_pose.astype(np.float32)
                episode_data[robot_name + '_demo_end_pose'] = demo_end_pose.astype(np.float32)

            out_replay_buffer.add_episode(data=episode_data, compressors=None)
            total_episodes += 1

    input_res = infer_input_resolution(group_dirs)
    resize_tf = get_image_transform(in_res=input_res, out_res=out_res)

    img_compressor = JpegXl(level=compression_level, numthreads=1)
    name = 'camera0_rgb'
    _ = out_replay_buffer.data.require_dataset(
        name=name,
        shape=(out_replay_buffer['robot0_eef_pos'].shape[0],) + out_res + (3,),
        chunks=(1,) + out_res + (3,),
        compressor=img_compressor,
        dtype=np.uint8,
    )

    img_array = out_replay_buffer.data[name]
    buffer_index = 0
    for group_spec in group_specs:
        print(f'Loading images for group {group_spec["name"]}: {group_spec["episode_count"]} episode(s)')
        for hdf5_path in group_spec['files']:
            print(f'Processing images from {hdf5_path}')
            with h5py.File(hdf5_path, 'r') as handle:
                images = handle['observations/images/front']
                for frame_idx in range(images.shape[0]):
                    img_array[buffer_index] = resize_tf(images[frame_idx])
                    buffer_index += 1
        if cleanup_input_groups:
            print(f'Removing merged input group {group_spec["dir"]}')
            shutil.rmtree(group_spec['dir'])

    print(f'Saving ReplayBuffer to {output_path}')
    try:
        with zarr.ZipStore(str(output_path), mode='w') as zip_store:
            out_replay_buffer.save_to_store(store=zip_store)
    finally:
        if temp_store_path.exists():
            shutil.rmtree(temp_store_path)

    counts = [group_spec['episode_count'] for group_spec in group_specs]
    count_path.write_text('\n'.join(str(count) for count in counts) + '\n', encoding='utf-8')
    manifest = {
        'output_dataset': str(output_path),
        'count_txt': str(count_path),
        'total_episodes': total_episodes,
        'groups': [
            {
                'name': group_spec['name'],
                'source_dir': str(group_spec['dir']),
                'episode_count': group_spec['episode_count'],
                'files': [path.name for path in group_spec['files']],
            }
            for group_spec in group_specs
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    print('-' * 60)
    print(f'Wrote dataset: {output_path}')
    print(f'Wrote count.txt: {count_path}')
    print(f'Wrote manifest: {manifest_path}')
    print(f'Total groups: {len(group_specs)}')
    print(f'Total episodes: {total_episodes}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert one or more FastUMI TCP directories into a merged UMI-style dataset.zarr.zip plus count.txt.'
    )
    parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH), help='Path to FastUMI config.json')
    parser.add_argument('--input-dir', help='Single TCP directory containing episode_*.hdf5 files.')
    parser.add_argument('--input-root', help='Root directory that contains multiple TCP group subdirectories.')
    parser.add_argument('--group-glob', default='*', help='Glob used with --input-root to select group directories.')
    parser.add_argument('--output', help='Output dataset.zarr.zip path. Defaults to config[dp_train_data_dir].')
    parser.add_argument('--out-res', help='Output image resolution as WIDTH,HEIGHT. Defaults to config[dp_data_res].')
    parser.add_argument('--compression-level', type=int, help='JPEG XL compression level. Defaults to config[compression_level].')
    parser.add_argument('--count-output', help='Optional explicit count.txt output path.')
    parser.add_argument('--manifest-output', help='Optional explicit conversion manifest output path.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite an existing output dataset.')
    parser.add_argument('--cleanup-input-groups', action='store_true', help='Delete each TCP input group after it has been merged into the output dataset.')
    parser.add_argument('--allow-incomplete-groups', action='store_true', help='Allow merging groups with inconsistent episode counts. Usually not needed for the final remainder-group case.')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_processing_config(args.config)
    output_path = args.output or config['dp_train_data_dir']
    out_res = tuple(int(value.strip()) for value in (args.out_res or config['dp_data_res']).split(','))
    compression_level = args.compression_level if args.compression_level is not None else config['compression_level']
    group_dirs = resolve_group_dirs(config, input_dir=args.input_dir, input_root=args.input_root, group_glob=args.group_glob)

    convert_groups_to_zarr(
        group_dirs=group_dirs,
        output_path=output_path,
        out_res=out_res,
        compression_level=compression_level,
        overwrite=args.overwrite,
        count_path=args.count_output,
        manifest_path=args.manifest_output,
        cleanup_input_groups=args.cleanup_input_groups,
        allow_incomplete_groups=args.allow_incomplete_groups,
    )


if __name__ == '__main__':
    main()
