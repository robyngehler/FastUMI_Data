# FastUMI-style Recording Guide for Franka Emika Panda
## Systematic Recording, Calibration, and Validation Guide

## Purpose

This guide defines a **clear Panda-specific FastUMI recording workflow** for collecting demonstrations with:

- a **Franka Emika Panda** as the target rollout robot,
- a **handheld FastUMI-style teaching device** with **T265** tracking,
- **one RGB recording stream** in the public FastUMI baseline,
- **Polymetis gRPC** already running on the robot side,
- `fastUMI` / `FastUMI_Data` and `librealsense` already installed.

It is written to be **operationally unambiguous** and to remove the common confusions around:

- the meaning of the **reference / init pose**,
- what must be calibrated **once** versus what is done **per episode**,
- when trajectories should be **trimmed**,
- which FastUMI config values are actually load-bearing,
- and where the custom helper scripts fit into the workflow.

---

## 1. Core concept: what the reference pose actually means

The most important conceptual point is this:

> The FastUMI reference pose is a **calibration anchor** for mapping the teaching-device trajectory into the Panda base frame. It is **not** primarily a requirement that the Panda must physically revisit that pose during every later rollout.

### 1.1 The correct frame relationship

You first choose a fixed Panda reference joint pose:

- `q_start`

From that you determine the corresponding Panda TCP pose in the Panda base frame:

- `T_base_tcp_ref`

Then you define one **tool / gripper-center frame** on the teaching device.

The teaching device must have a **start dock** such that, when docked, **its defined tool-center frame corresponds to `T_base_tcp_ref`**.

That does **not** mean:

- the entire teaching device body must geometrically overlap with the Panda,
- the T265 itself must sit where the Panda TCP sits,
- or the Panda must physically remain in that pose during recording.

It means only this:

- the **teaching-device tool-center / gripper-center** at initialization represents the same 6D pose as the Panda TCP at `q_start`,
- while the T265 may sit elsewhere because it is connected to that tool-center by a fixed rigid offset.

### 1.2 Why this matters

This reference pose is used for:

- `start_qpos` as the initial IK seed,
- `base_position` / `base_orientation` as the global anchor,
- validating the T265-to-tool offset,
- checking whether the first reconstructed sample is geometrically correct.

For deployment later, starting the Panda near this pose is usually still a good idea because it matches the training distribution, but **that is a data-distribution issue, not a recording-geometry requirement**.

---

## 2. What the public FastUMI code path actually does

Before adapting anything for Panda, freeze the public baseline behavior.

### 2.1 Recording stage

In the public FastUMI baseline:

- the RGB stream is written at **60 Hz**,
- the T265 odometry stream is written at **200 Hz**,
- the final paired dataset is produced at **20 Hz**,
- `episode_len = 180` corresponds to **180 samples at 20 Hz**, i.e. about **9 seconds**,
- recording starts **manually** when the user presses Enter,
- the script records a **fixed-length** window.

### 2.2 Raw HDF5 semantics

At raw recording time:

- `observations/qpos` stores pose + quaternion from the T265-derived trajectory,
- `action` initially mirrors the same pose + quaternion values,
- robot-specific TCP/joint semantics are added only later by the processing scripts.

### 2.3 Practical consequence for Panda

Treat the full pipeline as two separate phases:

### Phase A — raw recording
Keep this simple and reliable:

- image stream,
- T265 odometry,
- timestamps,
- fixed episode window.

### Phase B — Panda-specific processing
Only later convert into:

- Panda-compatible TCP trajectories,
- or Panda-compatible joint trajectories,
- with Panda URDF, Panda `start_qpos`, Panda `flange_to_tcp`, Panda gripper scaling, and Panda-validated offsets.

Do not mix these two stages mentally. That is where a lot of elegant nonsense begins.

---

## 3. Recording system architecture

## 3.1 Components

A clean Panda FastUMI-style setup has four layers:

### A. Target robot anchor
Used only for calibration and later rollout conventions:

- Franka Emika Panda
- Polymetis control stack
- known Panda base frame
- known Panda TCP convention

### B. Teaching device
Used for demonstration recording:

- T265 tracking module
- handheld gripper / proxy gripper
- two ArUco markers on the gripper jaws
- fixed RGB camera stream
- rigidly fixed assembly

### C. Start dock / reference fixture
Used to make initialization repeatable:

- fixes the teaching-device start pose,
- allows T265 reinitialization in a known pose,
- defines the episode start anchor.

### D. Recording environment
Used to keep the T265 sane:

- stable lighting,
- textured scene,
- distinct reference region for loop closure,
- no large featureless or reflective surfaces dominating the field of view.

---

## 4. What must be fixed once before any dataset collection

These items must be frozen before you collect a serious batch.

## 4.1 Mechanical assembly

Freeze the final assembly of:

- T265 mount,
- RGB camera mount,
- teaching-device gripper geometry,
- ArUco marker placement,
- tool-center definition.

If you change any of these later, your offset calibration is no longer trustworthy.

## 4.2 Tool-center definition

Define one single tool-center / gripper-center convention and never casually rename it later.

Recommended choice:

- midpoint between the gripper fingertips, at a defined nominal opening.

This definition must be used consistently in:

- Panda FK / rollout,
- teaching-device geometry,
- offset measurements,
- TCP reconstruction,
- and documentation.

## 4.3 Panda reference pose

Choose one canonical Panda anchor pose:

- collision-free,
- repeatable,
- visible,
- comfortable for the intended task family,
- inside the relevant workspace.

This pose is your:

- `q_start`,
- IK initialization anchor,
- corresponding robot-side TCP reference.

---

## 5. One-time calibration package

This section is the actual calibration workflow. Do it once carefully instead of fifty times badly.

## 5.1 Capture the Panda anchor pose

Move the Panda manually to the chosen reference pose and record:

- joint configuration,
- TCP position,
- TCP orientation.

Use the provided helper script:

- `capture_panda_anchor_pose_polymetis.py`

This script connects to the control-side Franka ZeroRPC bridge, which forwards to local Polymetis services, waits for a keypress, then stores:

- `start_qpos`,
- TCP position,
- TCP quaternion,
- TCP Euler angles,
- optional gripper state.

### Typical use

```bash
python capture_panda_anchor_pose_polymetis.py \
  --server-ip 192.168.1.10 \
  --with-gripper \
  --output panda_anchor_pose.json
```

### Important limitation

This script records the **robot-side anchor only**. It does **not** measure:

- the teaching-device dock pose,
- the T265-to-tool offset,
- or the marker scaling.

That is correct behavior, not laziness.

## 5.2 Build the teaching-device start dock

Build a mechanical fixture that lets the operator place the teaching device into the **same start pose every time**.

Good enough examples:

- V-block,
- 3D-printed cradle,
- hard-stop corner,
- groove plus stop,
- any rigid repeatable arrêtierung.

This dock has two purposes:

1. it defines the **initialization pose** of the teaching device,
2. it gives you a stable pose for **T265 reinitialization**.

## 5.3 Align the dock with the Panda anchor pose

This is the step that caused the most confusion, so here is the clean version:

1. Take the recorded Panda anchor pose from `panda_anchor_pose.json`.
2. Define the teaching-device tool-center physically.
3. Adjust the start dock so that, when the teaching device is docked, its **tool-center** corresponds to the Panda TCP anchor pose.
4. The T265 itself is allowed to be somewhere else in space, because the rigid T265-to-tool offset accounts for that.

The dock therefore matches the **tool-center pose**, not the shape of the full device.

## 5.4 Measure the T265-to-tool offset

The public FastUMI config only exposes:

```json
"offset": {"x": ..., "z": ...}
```

So the public pipeline assumes a reduced offset model rather than an arbitrary 6D transform.

### Recommended procedure

1. Freeze the final teaching-device assembly.
2. Identify the T265 camera center.
3. Identify the defined tool-center.
4. Measure the rigid vector from T265 center to tool-center.
5. Express the values in the axis/sign convention expected by the FastUMI scripts.
6. Store the result as initial candidates for:
   - `offset.x`
   - `offset.z`

### Best practice

- derive the nominal values from CAD if possible,
- verify with calipers on the real assembly,
- then validate against the reconstructed start pose.

Treat the first measured values as a **hypothesis** until validated.

## 5.5 Record `base_position` and `base_orientation`

Once the dock is aligned:

1. place the teaching device in the dock,
2. reinitialize the T265 while it is stationary,
3. use the Panda anchor pose as the source of truth for the corresponding base-frame pose,
4. write the pose into:
   - `base_position`
   - `base_orientation`

Interpretation:

- `base_position` / `base_orientation` describe the pose in the Panda base frame that the teaching-device tool-center corresponds to at initialization.

## 5.6 Determine `flange_to_tcp`

If your Panda URDF terminates at the flange rather than the chosen TCP/tool-center, measure the distance from:

- Panda flange frame
to
- chosen TCP / gripper-center frame

and store it as:

- `distances.flange_to_tcp`

Use the exact same TCP definition as in the rest of the guide. Mixing tool conventions is a very efficient way to become unhappy.

## 5.7 Calibrate gripper marker scaling

FastUMI derives gripper width from the pixel distance between two ArUco markers.

The public repo contains the mapping logic inside the processing scripts, but not a dedicated interactive calibration helper. For real use, use the provided helper script:

- `calibrate_fastumi_gripper_scaling.py`

This script:

- subscribes to a ROS image topic,
- detects two configured ArUco markers,
- lets you capture OPEN and CLOSED samples interactively,
- uses robust medians,
- writes a JSON snippet for:
  - `marker_max`
  - `marker_min`
  - `gripper_max`

### Typical use

```bash
python calibrate_fastumi_gripper_scaling.py \
  --image-topic /usb_cam/image_raw \
  --marker-id-0 0 \
  --marker-id-1 1 \
  --aruco-dict DICT_4X4_50 \
  --gripper-max 0.08 \
  --output gripper_scaling_panda.json
```

### What to do physically

- fully open the teaching-device gripper and capture several OPEN samples,
- fully close it to the minimum useful width and capture several CLOSED samples,
- use the medians as:
  - `marker_max` for OPEN,
  - `marker_min` for CLOSED.

### Important note

For calibration, this script deliberately accepts samples only when **both markers are visible**. That is stricter than the public processing path and exactly why it is useful.

---

## 6. What each config value means in practice

The Panda-relevant FastUMI config fields are the following.

## 6.1 `device_settings`

```json
"device_settings": {
  "robot_type": "PANDA",
  "data_dir": "./",
  "device": "cpu"
}
```

Meaning:

- `robot_type`: Panda label for bookkeeping,
- `data_dir`: dataset root,
- `device`: execution device for the scripts.

## 6.2 `task_config`

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

Meaning:

- `episode_len`: final number of **20 Hz samples** per episode,
- `state_dim`, `action_dim`: raw recording representation, not final Panda joint semantics,
- `camera_names`: keep `["front"]` for the public single-camera baseline,
- `video_topic`: RGB topic,
- `trajectory_topic`: T265 odometry topic.

## 6.3 `data_process_config`

```json
"data_process_config": {
  "marker_id_0": 0,
  "marker_id_1": 1,
  "urdf_path": "./assets/franka_panda.urdf",
  "base_position": {"x": 0.0, "y": 0.0, "z": 0.0},
  "base_orientation": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
  "offset": {"x": 0.0, "z": 0.0},
  "distances": {
    "marker_max": 0.0,
    "marker_min": 0.0,
    "gripper_max": 0.0,
    "gripper_min": 0.0,
    "flange_to_tcp": 0.0
  },
  "start_qpos": [ ... ]
}
```

Meaning:

- `marker_id_0`, `marker_id_1`: marker IDs on the teaching-device gripper,
- `urdf_path`: Panda URDF used in processing,
- `base_position`, `base_orientation`: Panda-base anchor pose of the initialized teaching-device tool-center,
- `offset`: T265 center to tool-center offset in the script’s convention,
- `marker_max`, `marker_min`: OPEN/CLOSED marker pixel distances,
- `gripper_max`: physical or convention-level maximum opening used downstream,
- `flange_to_tcp`: Panda flange to chosen TCP distance,
- `start_qpos`: Panda anchor joint pose used for initial IK seeding.

---

## 7. Standard operating procedure for a recording batch

This is the practical recording routine.

## 7.1 Before the batch

1. power the full setup,
2. start the required ROS graph,
3. confirm that the image topic is publishing,
4. confirm that the T265 odometry topic is publishing,
5. inspect tracking quality in RViz,
6. inspect marker visibility at OPEN and CLOSED gripper states,
7. place the teaching device into the start dock,
8. reinitialize the T265 while stationary,
9. confirm that the scene and objects are already arranged before recording starts.

### Minimal launch sequence

```bash
roscore
roslaunch realsense2_camera rs_t265.launch
roslaunch usb_cam usb_cam-test.launch
rviz
```

## 7.2 Per episode

For each episode:

1. place the teaching device in the dock,
2. hold it still briefly,
3. make sure the task scene is already prepared,
4. start the recording script,
5. press Enter only when the operator is ready,
6. execute the demonstration immediately,
7. stay within the fixed episode window,
8. if practical, return near the reference region at the end.

### Public baseline command

```bash
python data_collection.py --task <task_name> --num_episodes <N>
```

### Behavioral rule

Do not waste the first second after pressing Enter by hesitating. The public script records a fixed window. Dead time becomes dead data unless you trim it later.

---

## 8. Trimming policy: what relative trajectories do and do not solve

FastUMI uses relative trajectories to reduce dependence on absolute coordinates and exact base alignment. That is useful, but it does **not** remove the need for clean episode boundaries.

### What relative trajectories help with

They help reduce sensitivity to:

- absolute base-frame placement,
- exact global start location,
- embodiment transfer issues between setups.

### What they do not solve automatically

They do not automatically remove:

- idle time at the start,
- operator hesitation,
- irrelevant pre-motion,
- return-to-dock motion,
- inconsistent task onset.

### Practical implication

The public `data_collection.py` does **not** perform semantic trimming. It records a fixed-length window starting at the Enter press.

So for Panda data collection, the sensible policy is:

### Recommended policy
Use **raw fixed-window recording plus post-trim** during the first data-collection phase.

Trim if:

- the operator hesitated after start,
- the first part is only “move out of dock” behavior that should not belong to the skill,
- the episode ends early and then idles,
- the return-to-reference motion should not be learned.

Keep the pre-motion only if it is genuinely part of the deployed skill.

---

## 9. Validation before large-scale collection

Do not start a big dataset campaign before these checks pass.

## 9.1 Anchor pose validation

Perform a short no-motion test:

1. place the teaching device in the start dock,
2. reinitialize the T265,
3. record 1–2 seconds while holding perfectly still,
4. run the processing step.

Expected result:

- reconstructed TCP stays almost constant,
- the first reconstructed TCP matches the Panda anchor pose,
- the first reconstructed joint state is close to `start_qpos`.

If not, something is wrong in:

- dock alignment,
- offset sign convention,
- base pose definition,
- TCP definition,
- or processing assumptions.

## 9.2 Marker validation

Reject the setup if:

- one marker disappears for large parts of motion,
- the width signal is flat while the gripper visibly moves,
- glare, blur, or curvature destroys detection.

## 9.3 T265 quality validation

Reject or re-record if:

- there is obvious drift,
- large jumps appear,
- the end pose does not make sense when returning near the reference region,
- the environment is too dark or texture-poor.

## 9.4 Synchronization validation

Check that:

- timestamps are monotonic,
- camera frames align with the nearest T265 pose correctly,
- the extracted images and paired poses correspond visually.

---

## 10. Panda-specific processing caveats

This part matters more than people would like.

## 10.1 The public pipeline is not automatically Panda-ready

Changing only:

- `urdf_path`
- `start_qpos`

is almost certainly **not sufficient**.

The public processing path is visibly tailored to the example robot. For Panda you must verify:

- IK chain length,
- joint ordering,
- saved joint dimensionality,
- TCP convention,
- gripper appending logic.

## 10.2 Treat processed output as suspect until validated

Before trusting any processed Panda dataset, check that:

- the first sample maps near `q_start`,
- the trajectory direction matches the human demonstration,
- the motion stays inside the Panda workspace,
- the gripper signal behaves sensibly.

---

## 11. Minimal Panda config template

This is a template, not a magical truth dispenser.

```json
{
  "device_settings": {
    "robot_type": "PANDA",
    "data_dir": "./",
    "device": "cpu"
  },
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
  },
  "data_process_config": {
    "marker_id_0": 0,
    "marker_id_1": 1,
    "input_dir": "./dataset/<task>",
    "output_joint_dir": "./dataset/<task>_joint_with_gripper",
    "output_tcp_dir": "./dataset/<task>_tcp_with_gripper",
    "dp_train_data_dir": "./dataset/<task>.zarr.zip",
    "dp_data_res": "224, 224",
    "compression_level": 99,
    "urdf_path": "./assets/franka_panda.urdf",
    "aruco_dict": "DICT_4X4_50",
    "base_position": {
      "x": <from_anchor_definition>,
      "y": <from_anchor_definition>,
      "z": <from_anchor_definition>
    },
    "base_orientation": {
      "roll": <from_anchor_definition_deg>,
      "pitch": <from_anchor_definition_deg>,
      "yaw": <from_anchor_definition_deg>
    },
    "offset": {
      "x": <measure_and_validate>,
      "z": <measure_and_validate>
    },
    "distances": {
      "marker_max": <from_calibration_script>,
      "marker_min": <from_calibration_script>,
      "gripper_max": <set_in_target_convention>,
      "gripper_min": 0,
      "flange_to_tcp": <measure_or_compute>
    },
    "start_qpos": [
      <panda_joint_1>,
      <panda_joint_2>,
      <panda_joint_3>,
      <panda_joint_4>,
      <panda_joint_5>,
      <panda_joint_6>,
      <panda_joint_7>
    ]
  }
}
```

---

## 12. Recommended workflow summary

If the goal is a clean Panda-compatible FastUMI recording path, the correct order is:

1. **Freeze the mechanical teaching-device assembly.**
2. **Define the teaching-device tool-center clearly.**
3. **Choose one Panda anchor pose `q_start`.**
4. **Capture the Panda anchor with `capture_panda_anchor_pose_polymetis.py`.**
5. **Build a repeatable teaching-device start dock.**
6. **Align the dock so the teaching-device tool-center corresponds to the Panda TCP anchor pose.**
7. **Measure and validate the T265-to-tool offset.**
8. **Measure and store `flange_to_tcp`.**
9. **Calibrate marker scaling with `calibrate_fastumi_gripper_scaling.py`.**
10. **Fill the Panda FastUMI config.**
11. **Run no-motion validation and a tiny pilot batch.**
12. **Only then collect the real dataset.**
13. **Trim episodes afterwards when the fixed-window recording contains irrelevant pre/post motion.**

That is the disciplined version of the workflow. The alternative is debugging geometry by emotional interpretation, which remains a surprisingly popular but weak method.

---

## 13. Script reference

## 13.1 `capture_panda_anchor_pose_polymetis.py`

Use this for:

- `start_qpos`
- Panda anchor TCP pose
- optional gripper state snapshot

Output:

- JSON file with `start_qpos`
- TCP position
- TCP quaternion
- TCP Euler angles

Example:

```bash
python capture_panda_anchor_pose_polymetis.py \
  --server-ip 192.168.1.10 \
  --with-gripper \
  --output panda_anchor_pose.json
```

## 13.2 `calibrate_fastumi_gripper_scaling.py`

Use this for:

- `marker_max`
- `marker_min`
- helper output for `gripper_max`

Output:

- JSON file with robust OPEN/CLOSED statistics
- FastUMI config snippet for `distances`

Example:

```bash
python calibrate_fastumi_gripper_scaling.py \
  --image-topic /usb_cam/image_raw \
  --marker-id-0 0 \
  --marker-id-1 1 \
  --aruco-dict DICT_4X4_50 \
  --gripper-max 0.08 \
  --output gripper_scaling_panda.json
```

---

## 14. References

Primary references used to ground this guide:

- FastUMI paper
- public FastUMI config
- public `data_collection.py`
- public `data_processing_to_tcp.py`
- public `data_processing_to_joint.py`
- Polymetis user documentation
- helper scripts:
  - `capture_panda_anchor_pose_polymetis.py`
  - `calibrate_fastumi_gripper_scaling.py`
