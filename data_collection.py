import os
import json
import shutil
import torch
import cv2
import h5py
import argparse
from tqdm import tqdm
from time import sleep
import numpy as np
import pyrealsense2 as rs
import apriltag
import rospy

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
import csv
import threading
from collections import deque
from datetime import datetime
import pandas as pd
import gc

# Load configuration from config.json
with open('config/config.json', 'r') as f:
    config = json.load(f)

# Set environment variables
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"

# Set device
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

ROBOT_TYPE = config['device_settings']["robot_type"]
TASK_CONFIG = config['task_config']


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default="test3")  # open_lid, open_fridge, open_drawer, pick_place_pot
parser.add_argument('--num_episodes', type=int, default=2)
args = parser.parse_args()
task = args.task
num_episodes = args.num_episodes

cfg = TASK_CONFIG
robot = ROBOT_TYPE

data_path = os.path.join(config['device_settings']["data_dir"], "dataset" ,str(task))
os.makedirs(data_path, exist_ok=True)

IMAGE_PATH = os.path.join(data_path, 'camera', 'images')
os.makedirs(IMAGE_PATH, exist_ok=True)

CSV_PATH = os.path.join(data_path, 'csv/')
os.makedirs(CSV_PATH, exist_ok=True)

STATE_PATH = os.path.join(data_path, 'states.csv')
if not os.path.exists(STATE_PATH):
    with open(STATE_PATH, 'w') as csv_file2:
        csv_writer2 = csv.writer(csv_file2)
        csv_writer2.writerow(['Index', 'Start Time', 'Trajectory Timestamp', 'Frame Timestamp', 'Pos X', 'Pos Y', 'Pos Z', 'Q_X', 'Q_Y', 'Q_Z', 'Q_W'])

VIDEO_PATH_TEMP = os.path.join(data_path, 'camera', 'temp_video_n.mp4')
TRAJECTORY_PATH_TEMP = os.path.join(data_path, 'csv', 'temp_trajectory.csv')
TIMESTAMP_PATH_TEMP = os.path.join(data_path, 'csv', 'temp_video_timestamps.csv')
FRAME_TIMESTAMP_PATH_TEMP = os.path.join(data_path, 'csv', 'frame_timestamps.csv')
EPISODE_EVENTS_PATH = os.path.join(data_path, 'csv', 'episode_events.csv')

video_subscriber = None
trajectory_subscriber = None

# Initialize ROS node
rospy.init_node('video_trajectory_recorder', anonymous=True)

# Video writer parameters for 60 Hz recording
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_width, frame_height = cfg['cam_width'], cfg['cam_height']

# Buffers for storing incoming data
video_buffer = deque()
trajectory_buffer = deque()

# Lock for thread synchronization
buffer_lock = threading.Lock()

# Initialize CvBridge for image conversion
cv_bridge = CvBridge()

# Recorder state shared between callbacks, writer threads, and the main thread.
first_frame_timestamp = None
first_time_judger = False
recording_phase = 'idle'
recording_start_time = None
recording_stop_time = None
recording_stop_event = threading.Event()
tracking_started_at = None
latest_tracking_pose = None
video_writer = None
video_thread = None
trajectory_thread = None
trajectory_writer = None
timestamp_writer = None
trajectory_file_handle = None
timestamp_file_handle = None
active_trajectory_path = None
active_timestamp_path = None


def get_episode_raw_paths(episode):
    return {
        'video_path': VIDEO_PATH_TEMP.replace("_n", f"_{episode}"),
        'trajectory_path': os.path.join(CSV_PATH, f'temp_trajectory_{episode}.csv'),
        'timestamp_path': os.path.join(CSV_PATH, f'temp_video_timestamps_{episode}.csv'),
    }


def sync_latest_raw_csv_aliases(trajectory_path, timestamp_path):
    shutil.copyfile(trajectory_path, TRAJECTORY_PATH_TEMP)
    shutil.copyfile(timestamp_path, TIMESTAMP_PATH_TEMP)


def is_recording_timestamp(timestamp):
    return (
        recording_phase in {'recording', 'stopping'}
        and recording_start_time is not None
        and timestamp >= recording_start_time
        and (recording_stop_time is None or timestamp <= recording_stop_time)
    )

# Callback for video frames (60 Hz expected)
def video_callback(msg):
    global first_frame_timestamp, first_time_judger
    frame = cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    timestamp = msg.header.stamp.to_sec()

    with buffer_lock:
        if is_recording_timestamp(timestamp):
            video_buffer.append((frame, timestamp))
            if first_time_judger:
                first_frame_timestamp = timestamp
                first_time_judger = False

# Callback for trajectory data (e.g., T265 at 200 Hz)
def trajectory_callback(msg):
    global latest_tracking_pose
    timestamp = msg.header.stamp.to_sec()  # Ensure timestamp is in Unix format (float)
    with buffer_lock:
        pose = msg.pose.pose
        if recording_phase == 'tracking':
            latest_tracking_pose = (
                timestamp,
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            )
        elif is_recording_timestamp(timestamp):
            trajectory_buffer.append((timestamp, pose.position.x, pose.position.y, pose.position.z,
                                      pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w))


def write_video():
    global timestamp_file_handle
    frame_index = 0
    with tqdm(desc='Recording Frames', unit='frame') as pbar:
        while not rospy.is_shutdown():
            frame = None
            timestamp = None
            with buffer_lock:
                if video_buffer:
                    frame, timestamp = video_buffer.popleft()
                should_stop = recording_stop_event.is_set() and not video_buffer

            if frame is not None:
                video_writer.write(frame)
                timestamp_writer.writerow([frame_index, timestamp])
                frame_index += 1
                if frame_index % 60 == 0 and timestamp_file_handle is not None:
                    timestamp_file_handle.flush()
                pbar.update(1)
                continue

            if should_stop:
                break

            sleep(0.001)  # Small sleep to avoid CPU overload

    if timestamp_file_handle is not None:
        timestamp_file_handle.flush()
    print(f"Video Done! Wrote {frame_index} frames.")


def write_trajectory():
    global trajectory_file_handle
    counter = 0
    while not rospy.is_shutdown():
        sample = None
        with buffer_lock:
            if trajectory_buffer:
                sample = trajectory_buffer.popleft()
            should_stop = recording_stop_event.is_set() and not trajectory_buffer

        if sample is not None:
            Timestamp, PosX, PosY, PosZ, Q_X, Q_Y, Q_Z, Q_W = sample
            trajectory_writer.writerow([Timestamp, PosX, PosY, PosZ, Q_X, Q_Y, Q_Z, Q_W])
            counter += 1
            if counter % 200 == 0 and trajectory_file_handle is not None:
                trajectory_file_handle.flush()
            continue

        if should_stop:
            break

        sleep(0.001)  # Small sleep to avoid CPU overload

    if trajectory_file_handle is not None:
        trajectory_file_handle.flush()
    print(f"Trajectory Done! Wrote {counter} samples.")


def get_next_episode_path(base_dir):
    episode_indices = []
    for name in os.listdir(base_dir):
        if not name.startswith('episode_') or not name.endswith('.hdf5'):
            continue
        try:
            episode_indices.append(int(name[len('episode_'):-len('.hdf5')]))
        except ValueError:
            continue

    next_index = max(episode_indices, default=-1) + 1
    return os.path.join(base_dir, f'episode_{next_index}.hdf5')


def get_nearest_trajectory_indices(trajectory_timestamps, frame_timestamps):
    insert_positions = np.searchsorted(trajectory_timestamps, frame_timestamps)
    insert_positions = np.clip(insert_positions, 0, len(trajectory_timestamps) - 1)
    previous_positions = np.clip(insert_positions - 1, 0, len(trajectory_timestamps) - 1)

    use_previous = np.abs(trajectory_timestamps[previous_positions] - frame_timestamps) <= np.abs(
        trajectory_timestamps[insert_positions] - frame_timestamps
    )
    return np.where(use_previous, previous_positions, insert_positions)


def get_tracking_anchor_pose():
    with buffer_lock:
        return latest_tracking_pose


def begin_tracking_phase():
    global recording_phase, tracking_started_at, latest_tracking_pose
    with buffer_lock:
        video_buffer.clear()
        trajectory_buffer.clear()
        latest_tracking_pose = None
        tracking_started_at = rospy.Time.now().to_sec()
        recording_phase = 'tracking'


def open_recording_outputs(episode):
    global video_writer, trajectory_writer, timestamp_writer, trajectory_file_handle, timestamp_file_handle
    global active_trajectory_path, active_timestamp_path
    paths = get_episode_raw_paths(episode)
    temp_video_path = paths['video_path']
    active_trajectory_path = paths['trajectory_path']
    active_timestamp_path = paths['timestamp_path']
    video_writer = cv2.VideoWriter(temp_video_path, fourcc, 60, (frame_width, frame_height))
    if not video_writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {temp_video_path}")
    trajectory_file_handle = open(active_trajectory_path, 'w', newline='')
    timestamp_file_handle = open(active_timestamp_path, 'w', newline='')
    trajectory_writer = csv.writer(trajectory_file_handle)
    timestamp_writer = csv.writer(timestamp_file_handle)
    trajectory_writer.writerow(['Timestamp', 'Pos X', 'Pos Y', 'Pos Z', 'Q_X', 'Q_Y', 'Q_Z', 'Q_W'])
    timestamp_writer.writerow(['Frame Index', 'Timestamp'])
    return temp_video_path, active_trajectory_path, active_timestamp_path


def close_recording_outputs():
    global video_writer, trajectory_writer, timestamp_writer, trajectory_file_handle, timestamp_file_handle
    global active_trajectory_path, active_timestamp_path
    if video_writer is not None:
        video_writer.release()
        video_writer = None
    if trajectory_file_handle is not None:
        trajectory_file_handle.close()
        trajectory_file_handle = None
    if timestamp_file_handle is not None:
        timestamp_file_handle.close()
        timestamp_file_handle = None
    trajectory_writer = None
    timestamp_writer = None
    active_trajectory_path = None
    active_timestamp_path = None


def begin_recording_phase():
    global first_frame_timestamp, first_time_judger, recording_phase
    global recording_start_time, recording_stop_time, recording_stop_event
    global video_thread, trajectory_thread

    with buffer_lock:
        video_buffer.clear()
        trajectory_buffer.clear()
        first_frame_timestamp = None
        first_time_judger = True
        recording_start_time = rospy.Time.now().to_sec()
        recording_stop_time = None
        recording_phase = 'recording'

    recording_stop_event = threading.Event()
    video_thread = threading.Thread(target=write_video, daemon=True)
    trajectory_thread = threading.Thread(target=write_trajectory, daemon=True)
    video_thread.start()
    trajectory_thread.start()


def stop_recording_phase():
    global recording_phase, recording_stop_time, video_thread, trajectory_thread

    with buffer_lock:
        recording_stop_time = rospy.Time.now().to_sec()
        recording_phase = 'stopping'

    recording_stop_event.set()

    if video_thread is not None:
        video_thread.join()
        video_thread = None
    if trajectory_thread is not None:
        trajectory_thread.join()
        trajectory_thread = None

    with buffer_lock:
        recording_phase = 'idle'
        video_buffer.clear()
        trajectory_buffer.clear()


def append_episode_event(
    episode,
    record_start_time,
    record_stop_time_value,
    tracking_anchor_pose_snapshot,
    trajectory_path,
    timestamp_path,
    dataset_path,
):
    event_exists = os.path.exists(EPISODE_EVENTS_PATH)
    with open(EPISODE_EVENTS_PATH, 'a', newline='') as events_file:
        writer = csv.writer(events_file)
        if not event_exists or os.path.getsize(EPISODE_EVENTS_PATH) == 0:
            writer.writerow([
                'Episode Index',
                'Tracking Start Timestamp',
                'Tracking End Timestamp',
                'Recording Start Timestamp',
                'Recording Stop Timestamp',
                'First Frame Timestamp',
                'Tracking Anchor Timestamp',
                'Tracking Anchor Pos X',
                'Tracking Anchor Pos Y',
                'Tracking Anchor Pos Z',
                'Tracking Anchor Q_X',
                'Tracking Anchor Q_Y',
                'Tracking Anchor Q_Z',
                'Tracking Anchor Q_W',
                'Video Path',
                'Trajectory CSV Path',
                'Video Timestamp CSV Path',
                'HDF5 Path',
            ])
        writer.writerow([
            episode,
            tracking_started_at,
            record_start_time,
            record_start_time,
            record_stop_time_value,
            first_frame_timestamp,
            tracking_anchor_pose_snapshot[0],
        ] + list(tracking_anchor_pose_snapshot[1:]) + [
            temp_video_path,
            trajectory_path,
            timestamp_path,
            dataset_path,
        ])


def process_recorded_episode(episode, temp_video_path, trajectory_path, timestamp_path, record_start_time, tracking_anchor_pose_snapshot):
    if tracking_anchor_pose_snapshot is None:
        raise RuntimeError("No T265 tracking anchor pose was captured before recording started.")

    timestamps = pd.read_csv(timestamp_path)
    if timestamps.empty:
        raise RuntimeError("No video timestamps were written during recording.")

    downsampled_timestamps = timestamps.iloc[::3].reset_index(drop=True)
    cap = cv2.VideoCapture(temp_video_path)

    trajectory = pd.read_csv(trajectory_path)
    if trajectory.empty:
        raise RuntimeError("No trajectory samples were written during recording.")

    trajectory['Timestamp'] = trajectory['Timestamp'].astype(float)
    trajectory_timestamps = trajectory['Timestamp'].to_numpy()
    frame_timestamps = downsampled_timestamps['Timestamp'].to_numpy(dtype=float)
    target_frame_indices = downsampled_timestamps['Frame Index'].to_numpy(dtype=int)
    nearest_indices = get_nearest_trajectory_indices(trajectory_timestamps, frame_timestamps)

    max_timesteps = len(downsampled_timestamps)
    dataset_path = get_next_episode_path(data_path)
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    episode_image_path = os.path.join(IMAGE_PATH, f'episode_{episode:06d}')
    os.makedirs(episode_image_path, exist_ok=True)

    try:
        with h5py.File(dataset_path, 'w', rdcc_nbytes=2 * 1024 ** 2) as root:
            root.attrs['sim'] = False
            root.attrs['recording_start_timestamp'] = float(record_start_time)
            root.attrs['recording_stop_timestamp'] = float(recording_stop_time)
            root.attrs['first_frame_timestamp'] = float(first_frame_timestamp)
            root.attrs['tracking_anchor_timestamp'] = float(tracking_anchor_pose_snapshot[0])
            root.attrs['tracking_anchor_pose_xyzw'] = np.asarray(tracking_anchor_pose_snapshot[1:], dtype=np.float64)
            obs = root.create_group('observations')
            image_grp = obs.create_group('images')
            image_datasets = dict()
            for cam_name in cfg['camera_names']:
                image_datasets[cam_name] = image_grp.create_dataset(
                    cam_name,
                    shape=(max_timesteps, frame_height, frame_width, 3),
                    dtype=np.uint8,
                    chunks=(1, frame_height, frame_width, 3),
                    compression='gzip',
                    compression_opts=4
                )

            qpos_dataset = root.create_dataset('observations/qpos', shape=(max_timesteps, 7), dtype=np.float64)
            action_dataset = root.create_dataset('action', shape=(max_timesteps, 7), dtype=np.float64)

            with open(STATE_PATH, 'a', newline='') as csv_file2:
                csv_writer2 = csv.writer(csv_file2)
                next_target_idx = 0
                current_frame_idx = 0

                with tqdm(total=max_timesteps, desc='Extracting Images + States') as pbar:
                    while next_target_idx < max_timesteps:
                        ret, frame = cap.read()
                        if not ret:
                            raise RuntimeError(
                                f"Failed to decode frame {current_frame_idx} from temporary recording before all sampled frames were extracted."
                            )

                        if current_frame_idx == target_frame_indices[next_target_idx]:
                            row = downsampled_timestamps.iloc[next_target_idx]
                            filename = f"{next_target_idx:06d}.jpg"
                            cv2.imwrite(os.path.join(episode_image_path, filename), frame)
                            for cam_name in cfg['camera_names']:
                                image_datasets[cam_name][next_target_idx] = frame

                            closest_row = trajectory.iloc[int(nearest_indices[next_target_idx])]
                            pos_quat = [
                                closest_row['Pos X'], closest_row['Pos Y'], closest_row['Pos Z'],
                                closest_row['Q_X'], closest_row['Q_Y'], closest_row['Q_Z'], closest_row['Q_W']
                            ]
                            qpos_dataset[next_target_idx] = pos_quat
                            action_dataset[next_target_idx] = pos_quat
                            csv_writer2.writerow([
                                next_target_idx,
                                record_start_time,
                                closest_row['Timestamp'],
                                row['Timestamp']
                            ] + pos_quat)

                            next_target_idx += 1
                            pbar.update(1)

                        current_frame_idx += 1
    finally:
        cap.release()

    del downsampled_timestamps
    del trajectory
    del timestamps
    gc.collect()

    append_episode_event(
        episode=episode,
        record_start_time=record_start_time,
        record_stop_time_value=recording_stop_time,
        tracking_anchor_pose_snapshot=tracking_anchor_pose_snapshot,
        trajectory_path=trajectory_path,
        timestamp_path=timestamp_path,
        dataset_path=dataset_path,
    )

if __name__ == "__main__":
    # Initialize subscribers
    cv_bridge = CvBridge()
    video_subscriber = rospy.Subscriber(config['task_config']['ros']['video_topic'], Image, video_callback, queue_size=config['task_config']['ros']['queue_size'])
    trajectory_subscriber = rospy.Subscriber(config['task_config']['ros']['trajectory_topic'], Odometry, trajectory_callback, queue_size=config['task_config']['ros']['queue_size'])

    # Initialize frame timestamp file
    frame_timestamp_exists = os.path.exists(FRAME_TIMESTAMP_PATH_TEMP)
    with open(FRAME_TIMESTAMP_PATH_TEMP, "a", newline='') as frame_timestamp_file:
        frame_timestamp_writer = csv.writer(frame_timestamp_file)
        if not frame_timestamp_exists or os.path.getsize(FRAME_TIMESTAMP_PATH_TEMP) == 0:
            frame_timestamp_writer.writerow(['Episode Index', 'Timestamp'])

        for episode in range(num_episodes):
            temp_video_path = None
            record_started_at = None
            tracking_anchor_pose_snapshot = None
            trajectory_path = None
            timestamp_path = None

            try:
                input(f"Episode {episode + 1}/{num_episodes}: Press Enter to start T265 tracking-only mode...")
                begin_tracking_phase()
                print("Tracking active. Move to the record-init pose, then press Enter again to start recording.")
                input(f"Episode {episode + 1}/{num_episodes}: Press Enter to start recording...")

                tracking_anchor_pose_snapshot = get_tracking_anchor_pose()
                if tracking_anchor_pose_snapshot is None:
                    raise RuntimeError("No T265 pose received yet. Wait for tracking to stabilize before starting recording.")

                temp_video_path, trajectory_path, timestamp_path = open_recording_outputs(episode)
                begin_recording_phase()
                record_started_at = recording_start_time
                print(f"Episode {episode + 1}/{num_episodes} recording. Press Enter to stop, save, and process the episode.")
                input()

                stop_recording_phase()
                close_recording_outputs()
                sync_latest_raw_csv_aliases(trajectory_path, timestamp_path)

                frame_timestamp_writer.writerow([episode, first_frame_timestamp])
                frame_timestamp_file.flush()
                process_recorded_episode(
                    episode,
                    temp_video_path,
                    trajectory_path,
                    timestamp_path,
                    record_started_at,
                    tracking_anchor_pose_snapshot,
                )
                print(f"Episode {episode + 1}/{num_episodes} saved successfully.")
            except KeyboardInterrupt:
                print("\nKeyboard interrupt received. Stopping recording cleanly.")
                if recording_phase in {'recording', 'stopping'}:
                    stop_recording_phase()
                close_recording_outputs()
                raise
            except Exception as e:
                print(f"An error occurred during episode {episode + 1}: {e}")
                if recording_phase in {'recording', 'stopping'}:
                    stop_recording_phase()
                close_recording_outputs()
                raise
            finally:
                with buffer_lock:
                    recording_phase = 'idle'
                    video_buffer.clear()
                    trajectory_buffer.clear()
                    latest_tracking_pose = None
                    tracking_started_at = None
                    recording_start_time = None
                    recording_stop_time = None
                    first_frame_timestamp = None

    print("All episodes completed successfully!")