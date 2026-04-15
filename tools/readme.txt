## calibrate gripper state:
~~~bash
python calibrate_fastumi_gripper_scaling.py \
  --image-topic /usb_cam/image_raw \
  --marker-id-0 0 \
  --marker-id-1 1 \
  --aruco-dict DICT_4X4_50 \
  --gripper-max 0.08 \
  --output gripper_scaling_panda.json
~~~

## capture panda init pose:
~~~bash
python capture_panda_anchor_pose_polymetis.py \
  --server-ip 192.168.1.10 \
  --with-gripper \
  --output panda_anchor_pose.json
~~~

## trim episodes:
~~~bash
python trim_fastumi_episode.py \
  --task-dir ./dataset/pour_water \
  --episode-index 0 \
  --start-sec 1.35 \
  --end-sec 7.90
~~~

### with random crops:
~~~bash
python trim_fastumi_episode.py \
  --task-dir ./dataset/pour_water \
  --episode-index 0 \
  --start-sec 1.35 \
  --end-sec 7.90 \
  --num-random-start-crops 3 \
  --max-random-start-crop-sec 1.5 \
  --min-remaining-sec 3.0 \
  --seed 42
~~~
