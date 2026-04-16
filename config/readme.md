# Configuration Parameters Documentation

This document provides a brief explanation of the JSON configuration parameters used in the system, divided into three sections:

1. **Device Settings**
2. **Task Configuration**
3. **Data Processing Configuration**

---

## Table of Contents

1. [Device Settings](#device-settings)
2. [Task Configuration](#task-configuration)
3. [Data Processing Configuration](#data-processing-configuration)

---

## Device Settings

Defines the robot type, data directory, and computational device.

```json
"device_settings": {
    "robot_type": "PANDA",
    "data_dir": "./",
    "device": "cpu"
}
```

- **robot_type** (`string`): Type of the robot.
- **data_dir** (`string`): Path for data storage.
- **device** (`string`): Computational device.

---

## Task Configuration

Sets the task duration, state and action dimensions, camera settings, and ROS topics.

```json
"task_config": {
    "episode_len": 180,
    "state_dim": 7,
    "action_dim": 7,
    "cam_width": 1920,
    "cam_height": 1080,
    "camera_names": ["front"],
    "camera_port": 0,
    "ros": {
        "video_topic": "/usb_cam/image_raw",
        "trajectory_topic": "/camera/odom/sample",
        "queue_size": 1000
    }
}
```

- **episode_len** (`integer`): Length of each task episode, please adjust according to the task duration.
- **state_dim** (`integer`): Dimension of the state space.
- **action_dim** (`integer`): Dimension of the action space.
- **cam_width** (`integer`): Camera width in pixels.
- **cam_height** (`integer`): Camera height in pixels.
- **camera_names** (`array`): Names of the cameras.
- **camera_port** (`integer`): Camera port number.

### ROS Configuration

- **video_topic** (`string`): ROS topic for video data.
- **trajectory_topic** (`string`): ROS topic for trajectory data.
- **queue_size** (`integer`): Size of the ROS message queue.

---

## Data Processing Configuration

Manages data processing paths, markers, robot position, and initial joint states. 
We have already performed coordinate transformation in the code. The corresponding relationship diagram of coordinate transformation is shown below:


<figure align="center">
  <img src="../docs/2.png" width="400" />
  <img src="../docs/3.jpg" width="400" />
  <figcaption>Coordinate system of T265 in rviz</figcaption>
</figure>
<figure align="center">
  <img src="../docs/1.png" width="400" />
  <figcaption>Coordinate system of Xarm6</figcaption>
</figure>

The figure above is the legacy xArm6 example from the original FastUMI release. For Panda FastUMI, use the active Panda config and Panda recording guide as the source of truth for frame semantics and IK settings.


The following configuration is based on the coordinate system conversion shown in the figure.


```json
"data_process_config": {
    "marker_id_0":0,
    "marker_id_1":1,
    "input_dir": "./dataset/test",
    "output_joint_dir": "./dataset/test_joint_with_gripper",
    "output_tcp_dir": "./dataset/test_tcp_with_gripper",
    "dp_train_data_dir": "./dataset/dp_train_data.zarr.zip",
    "dp_data_res": "224, 224",
    "compression_level": 99,
    "urdf_path": "./assets/fer_franka_hand.urdf",
    "ik_base_elements": ["base"],
    "aruco_dict": "DICT_4X4_50",
    "base_position": {
        "x": 0.64071,
        "y": -0.34948,
        "z": 0.37141
    },
    "base_orientation": {
        "roll": -135.9873,
        "pitch": 79.4097,
        "yaw": -135.8613
    },
    "offset": {
        "x": 0.1489,
        "z": 0.1483
    },
    "distances": {
        "marker_max": 513.18,
        "marker_min": 113.57,
        "gripper_max": 850.0,
        "gripper_min": 0,
        "flange_to_tcp": 0.093
    },
    "frame_conventions": {
        "runtime_raw_frame": "panda_link8",
        "ik_target_frame": "panda_hand_tcp",
        "fastumi_tcp_frame": "panda_softtip",
        "link8_to_hand_rotation_deg_z": -45.0,
        "link8_to_hand_translation_m": 0.1034,
        "soft_tip_offset_m": 0.093
    },
    "start_qpos": [1.1461714506149292, -1.5383373498916626, -1.5858408212661743, -2.0414700508117676, -2.0066282749176025, 3.025059461593628, -2.0232622623443604]
}
```

- **marker_id_0 / marker_id_1** (`integer`): IDs for ArUco markers.
- **input_dir** (`string`): Path for input data.
- **output_joint_dir / output_tcp_dir** (`string`): Paths for output data.
- **dp_train_data_dir** (`string`): Path to training data archive.
- **dp_data_res** (`string`): Image resolution for data processing.
- **compression_level** (`integer`): Compression level for data storage.
- **urdf_path** (`string`): Path to the URDF file.
- **ik_base_elements** (`array[string]`): URDF base-link hint for `ikpy`. This is required when the robot base link is not `world`.
- **aruco_dict** (`string`): Type of ArUco dictionary.

### Position & Orientation

- **base_position**: Initial Tool Center Point position of Robot arm. For Panda FastUMI, this should be the FastUMI TCP anchor pose, not `panda_link8`, not bare `panda_hand`, and not the intermediate `panda_hand_tcp` frame.
- **base_orientation**: Initial Tool Center Point orientation of Robot arm. For Panda FastUMI, this should be the FastUMI TCP anchor orientation, not the intermediate hand-aligned frame.
- **offset**: Positional offsets of the RealSense T265 relative to the Tool Center Point on Handheld Device.

### Distance Parameters

- **marker_max / marker_min** (`float`): Marker detection distance range.
- **gripper_max / gripper_min** (`float`): Gripper open/close range (Refers to the value transmitted in the gripper control command).
- **flange_to_tcp** (`float`): Distance from the IK target frame to the stored TCP. In the Panda FastUMI path this is used as the `panda_hand_tcp -> panda_softtip` offset.

### Initial Joint Positions

- **start_qpos** (`array`): Initial positions for the robot arm's actuated joints. For Panda this is a 7-joint seed; for xArm6 it is a 6-joint seed.

---

## Summary

This configuration file is essential for setting up the robot's hardware, task parameters, and data processing workflows. Adjust these parameters according to your specific needs to ensure paths and values correctly match your system setup.

For further assistance or questions, please refer to the system documentation or contact the support team.
