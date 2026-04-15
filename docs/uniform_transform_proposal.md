# Proposal: FastUMI-specific Panda frame path and calibration updates

## Status
Draft for direct implementation in the local codebase.

## Goal
Implement a **FastUMI-specific client-side frame path** for the Franka Panda that:

1. **does not modify the server/control-side stack**,
2. **does not break the existing UMI / Data Scaling Laws runtime**,
3. gives FastUMI a **clean, explicit TCP convention** suitable for recording and joint reconstruction, and
4. removes the current ambiguity around "flange / link8 / hand / soft-tip" semantics.

This proposal is intended to be actionable for both:
- a human implementing the changes, and
- an agent modifying the repository.

---

# 1. Constraints and established facts

## 1.1 Hard constraint
The server/control side is treated as **read-only**.

Reason:
- the same server-side stack is already used successfully by the existing UMI / DSL eval path,
- touching it would create unnecessary integration risk.

## 1.2 What the current Polymetis metadata says
The currently provided Polymetis configs use:
- `ee_link_name: panda_link8`
- not `panda_hand`

This means the standard Python `RobotInterface.get_ee_pose()` computes FK to **`panda_link8`**, unless the local ZeroRPC bridge adds its own transform.

## 1.3 What the current inference-side scripts assume
The current inference-side UMI/DSL scripts apply a **fixed hard-coded transform** on top of the raw EE pose:
- rotation about local Z by `-135°`
- translation by `0.18832 m`

This is internally consistent because the runtime uses the forward transform when reading poses and the inverse transform when sending poses back.

## 1.4 What this does and does not prove
The fact that UMI / DSL eval already moves the robot correctly proves:
- the **current runtime is internally self-consistent**.

It does **not** prove that the current transform is semantically equal to:
- `flange -> panda_hand`, or
- `franka_hand -> soft_finger_tip`.

It is much more likely a **legacy UMI operational tool frame**.

## 1.5 What the Panda URDF says
For the local Panda arm-with-hand URDF used here:
- `panda_hand` is a fixed child of `panda_link8`
- with **rotation about Z = -45°**
- and **no translation**
- plus a fixed `panda_hand_tcp` child of `panda_hand`
- with **translation `[0, 0, 0.1034]`** and identity rotation

Therefore:
- if the raw runtime frame is `panda_link8`,
- and FastUMI should use a TCP aligned with the hand / finger direction,
- then the local client-side rigid hand transform should include the **-45° rotation** and the fixed **0.1034 m forward extension** to the operational `panda_hand_tcp` frame.

This explains the missing semantic piece:
- the user intuition "just take hand and extend in Z" is correct,
- but in the local URDF/tooling path the rigid operational intermediate frame is effectively `panda_hand_tcp`, reached after rotating into `panda_hand` and applying the fixed `0.1034 m` extension.

---

# 2. Target convention for the FastUMI path

## 2.1 Do not reuse the legacy UMI tool convention for FastUMI
The current UMI transform should remain untouched for UMI / DSL.

For FastUMI, define a **new explicit TCP convention**:

### Raw control/runtime frame
`raw_ee_frame := panda_link8`

### FastUMI IK frame
`ik_frame := panda_hand_tcp`

### FastUMI dataset / recording TCP
`fastumi_tcp := panda_hand_tcp + z-forward offset to the chosen soft-finger-tip point`

## 2.2 Required semantics
The FastUMI TCP must satisfy:
- same orientation as `panda_hand_tcp`
- only a forward translation along the local hand Z-axis
- no extra arbitrary rotation

This is important because the current `data_processing_to_joint.py` can only move from TCP back to the IK target by:
- subtracting a scalar distance along local Z
- while keeping the quaternion unchanged

So the FastUMI TCP must be **translation-only** relative to the IK target frame.

## 2.3 Chosen transform chain
Use the following chain for FastUMI:

```text
raw runtime pose (panda_link8)
    -> fixed URDF-consistent transform link8_to_hand
    -> fixed measured transform hand_to_softtip
    -> FastUMI TCP pose
```

Where:

### `link8_to_hand`
From the local Panda URDF/tooling path:
- translation: `[0, 0, 0.1034]`
- rotation: `Rz(-45°)`

### `hand_to_softtip`
From measurement / CAD / physical definition:
- translation: `[0, 0, soft_tip_offset_m]`
- rotation: identity

Initial engineering estimate:
- `soft_tip_offset_m ≈ 0.09 m`

But this value must be **measured and validated**, not treated as holy scripture.

---

# 3. Proposed code strategy

## 3.1 Do not modify the shared UMI / DSL runtime files in place
To avoid collateral damage, do **not** overwrite the current shared runtime behavior.

Instead create a **FastUMI-specific client-side path** in one of these two styles:

### Preferred option
Create separate files:
- `capture_panda_anchor_pose_fastumi.py`
- `franka_interpolation_controller_fastumi.py`
- optionally `panda_fastumi_frames.py`

### Acceptable alternative
Parameterize the existing files with a mode switch, but keep the current default behavior unchanged:
- `tool_convention = "umi_legacy" | "panda_fastumi"`

If this parameterized option is chosen, the default must remain the current UMI behavior.

---

# 4. Required new shared transform utility

Create a small shared utility module, for example:

```text
umi/common/panda_fastumi_frames.py
```

or, if preferred in the FastUMI repo:

```text
tools/panda_fastumi_frames.py
```

## 4.1 Functions to provide

### `link8_to_hand_transform()`
Returns the fixed transform:
- `Rz(-45°)`
- translation `[0, 0, 0.1034]`

### `hand_to_softtip_transform(offset_m)`
Returns:
- identity rotation
- translation `[0, 0, offset_m]`

### `link8_pose_to_fastumi_tcp_pose(link8_pose, offset_m)`
Applies:

```text
T_link8_tcp = T_link8_hand * T_hand_softtip
```

### `fastumi_tcp_pose_to_link8_pose(tcp_pose, offset_m)`
Applies the inverse transform.

### Optional explicit helpers
- `link8_pose_to_hand_pose(...)`
- `hand_pose_to_fastumi_tcp_pose(...)`
- `fastumi_tcp_pose_to_hand_pose(...)`

## 4.2 Implementation note
Do not scatter the constants
- `-45°`
- `soft_tip_offset_m`

through multiple files.

Put them into one transform utility or one config block.

---

# 5. Capture script changes

## 5.1 File
Create:

```text
tools/capture_panda_anchor_pose_fastumi.py
```

starting from the current `capture_panda_anchor_pose_polymetis.py`.

## 5.2 Remove
Remove the current legacy UMI transform logic:
- `tx_flange_flangerot45`
- `tx_flangerot45_tip`
- `tx_flange_tip`
- `flange_pose_to_tip_pose(...)`

## 5.3 Replace with
Interpret the raw `client.get_ee_pose()` as **`panda_link8` pose**, not as hand pose.

Then compute:

```text
raw_link8_pose -> hand_pose -> fastumi_tcp_pose
```

using:
- `link8_to_hand_transform()`
- `hand_to_softtip_transform(soft_tip_offset_m)`

## 5.4 Script outputs
The JSON output should contain **both**:

### Raw runtime frame
- `raw_link8_pose`

### FastUMI-relevant frames
- `hand_pose`
- `fastumi_tcp_pose`

### Metadata block
- `raw_runtime_frame: panda_link8`
- `ik_target_frame: panda_hand_tcp`
- `fastumi_tcp_frame: panda_softtip`
- `link8_to_hand_rotation_deg_z: -45`
- `soft_tip_offset_m: ...`

## 5.5 Why this is needed
This prevents future ambiguity about what exactly was recorded.

The current script only stores the transformed output and a note saying the bridge returns flange pose. That is not sufficient for a clean FastUMI geometry path.

---

# 6. Runtime controller changes for FastUMI

## 6.1 File
Create:

```text
umi/real_world/franka_interpolation_controller_fastumi.py
```

starting from the current `franka_interpolation_controller.py`.

## 6.2 Reading robot state
Change `get_ee_pose()` and `get_robot_state()` so that:
- raw server pose is treated as `panda_link8`
- exposed external pose is the **FastUMI TCP**

That means:

```text
raw link8 pose
    -> link8_to_hand
    -> hand_to_softtip
    -> exposed pose to FastUMI side
```

## 6.3 Writing desired pose back
Before calling `update_desired_ee_pose()` on the server, invert the same transform:

```text
fastumi tcp pose
    -> inverse(hand_to_softtip)
    -> inverse(link8_to_hand)
    -> raw link8 pose
```

## 6.4 Important acceptance criterion
The FastUMI runtime must remain internally self-consistent exactly like the UMI runtime is today:
- read path and write path must be strict inverses.

## 6.5 Important non-goal
Do **not** alter the existing UMI / DSL controller file unless a mode switch is introduced and the default remains unchanged.

---

# 7. Data processing changes

## 7.1 `data_processing_to_tcp.py`
No conceptual geometry rewrite is needed.

This script already assumes that the stored pose after base normalization is the desired TCP pose.

### Requirement
`base_position` and `base_orientation` must correspond to the **FastUMI TCP anchor pose**, not to `panda_link8`, not to bare `panda_hand`, and not to the intermediate `panda_hand_tcp` frame.

That means the anchor capture script must write the FastUMI TCP pose used for initialization.

## 7.2 `data_processing_to_joint.py`
This script can already support the proposed convention, but only if the semantics are set correctly.

Current behavior:
- it interprets stored pose as TCP
- then moves back along local Z by `config["distances"]["flange_to_tcp"]`
- then performs IK

### For the proposed convention
Use:

```text
stored pose := FastUMI TCP = panda_hand_tcp + local z offset
IK target frame := panda_hand_tcp
config["distances"]["flange_to_tcp"] := hand_to_softtip_offset_m
```

The key name is misleading, but functionally usable.

## 7.3 Recommended improvement
Add a code comment and guide note that in the Panda FastUMI path:

```text
config["distances"]["flange_to_tcp"]
```

actually means:

```text
ik_target_to_fastumi_tcp_offset_m
```

or more concretely:

```text
hand_to_softtip_offset_m
```

## 7.4 Important caveat
This proposal does **not** fix the other known Panda issues in `data_processing_to_joint.py`, such as:
- `full_joint_angles[2:]`
- assumptions around the IK chain root and dimensions

Those must still be validated separately.

---

# 8. Calibration script proposal

## 8.1 Gripper scaling script
The current `calibrate_fastumi_gripper_scaling.py` is conceptually fine.

### Required usage change
Use it as a **multi-sample calibration tool**, not as a one-shot probe.

### Minimum recommended procedure
- 10–20 OPEN samples
- 10–20 CLOSED samples
- use medians, not single frames
- keep both markers fully visible
- run with the real recording camera setup

### Optional code improvement
Add to the JSON output:
- full list of captured distances
- coefficient of variation
- warning if `count < 5`

This is helpful but not strictly required.

## 8.2 Anchor pose capture script
The current anchor pose script must be replaced for FastUMI use.

It currently bakes in the legacy UMI transform and therefore hides which frame was actually captured.

The new FastUMI capture script must:
- preserve raw runtime pose
- expose hand pose
n- expose final FastUMI TCP pose
- record the exact transform parameters used

---

# 9. Config changes

## 9.1 Existing `config.json`
Keep the current public FastUMI structure, but add an explicit documentation block for human readability.

### Proposed additional block
```json
"frame_conventions": {
  "runtime_raw_frame": "panda_link8",
  "ik_target_frame": "panda_hand_tcp",
  "fastumi_tcp_frame": "panda_softtip",
  "link8_to_hand_rotation_deg_z": -45.0,
  "link8_to_hand_translation_m": 0.1034,
  "soft_tip_offset_m": 0.09,
  "notes": "FastUMI TCP keeps the operational panda_hand_tcp orientation and only extends forward in local z."
}
```

This block is optional for code, but strongly recommended for traceability.

## 9.2 Existing `data_process_config`
Use:
- `base_position` / `base_orientation` from the FastUMI TCP anchor pose
- `distances.flange_to_tcp = soft_tip_offset_m`
- `start_qpos` from the Panda reference joint pose

## 9.3 Do not change control-side metadata
Do not change:
- `ee_link_name`
- server-side metadata
- Polymetis robot model

for the first FastUMI integration pass.

The entire point of this proposal is to avoid breaking UMI / DSL.

---

# 10. Recommended validation plan

## 10.1 Static frame sanity check
For one recorded anchor pose:
- record raw `panda_link8` pose
- convert to `panda_hand_tcp`
- convert to FastUMI TCP
- numerically inspect:
  - same translation as expected?
  - same orientation as hand?
  - only one known offset along local hand Z?

## 10.2 Runtime inverse consistency check
For the new FastUMI controller:
- convert raw link8 -> FastUMI TCP
- then convert FastUMI TCP -> link8
- verify numerical reconstruction error is near zero

## 10.3 Robot-side motion sanity check
Using a safe test trajectory:
- send a small straight-line command in FastUMI TCP space
- verify real robot motion is physically sensible and matches the expected hand / finger direction

## 10.4 Data processing sanity check
On a tiny dataset:
- run `data_processing_to_tcp.py`
- run `data_processing_to_joint.py`
- verify the first frame joint reconstruction is close to `start_qpos`
- verify no orientation mismatch is visible between intended hand frame and reconstructed IK frame

---

# 11. Implementation sequence

## Phase 1 — Introduce new frame utility
- add `panda_fastumi_frames.py`
- define and unit test transforms

## Phase 2 — FastUMI capture path
- add `capture_panda_anchor_pose_fastumi.py`
- output raw link8 pose, hand pose, and FastUMI TCP pose

## Phase 3 — FastUMI runtime path
- add `franka_interpolation_controller_fastumi.py`
- keep existing UMI controller untouched

## Phase 4 — Config update
- write anchor pose into `base_position` / `base_orientation`
- set `distances.flange_to_tcp = soft_tip_offset_m`
- document frame conventions in config

## Phase 5 — Small-scale validation
- 1–3 recorded episodes
- TCP conversion sanity check
- joint reconstruction sanity check

Only after this should larger-scale data collection begin.

---

# 12. Explicit non-goals
This proposal does **not** attempt to:
- change the server-side ZeroRPC bridge
- change Polymetis metadata
- replace the existing UMI / DSL operational frame
- solve all Panda IK issues in `data_processing_to_joint.py`
- standardize the entire codebase on a single TCP convention

The goal is only:
- create a **FastUMI-specific, geometrically clean, client-side path** that coexists with the current UMI / DSL path.

---

# 13. Summary of required code changes

## New files
- `tools/capture_panda_anchor_pose_fastumi.py`
- `umi/real_world/franka_interpolation_controller_fastumi.py`
- `tools/panda_fastumi_frames.py` or equivalent shared module

## Existing files to leave unchanged initially
- current UMI / DSL capture script
- current UMI / DSL interpolation controller
- server-side bridge
- Polymetis metadata / hardware config

## Existing files to update only with comments / config semantics
- `config/config.json`
- `data_processing_to_joint.py` comments / docs
- recording guide / proposal references

---

# 14. Final recommendation

The safest and cleanest path is:

1. **Treat the current UMI / DSL runtime as a legacy but working operational frame path.**
2. **Do not touch server-side control or Polymetis metadata in the first FastUMI pass.**
3. **Build a FastUMI-specific client-side frame layer from `panda_link8` to a new TCP aligned with the operational `panda_hand_tcp` frame.**
4. **Define FastUMI TCP as `panda_hand_tcp` orientation plus a measured forward Z-offset to the chosen soft-finger-tip point.**
5. **Use that TCP consistently for anchor calibration, recording, TCP processing, and joint reconstruction.**

That gives FastUMI a clean convention without destabilizing the already working UMI / DSL stack.
