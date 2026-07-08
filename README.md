# Yahboom MicroROS-Pi5 × Coral Edge TPU

**English** · [简体中文](README.zh-CN.md)

Second-development project for the **Yahboom MicroROS-Pi5** robot car: the stock vision perception (classic CV / CPU CNN) is replaced with **pre-compiled `*_edgetpu.tflite` models running on a Coral USB Edge TPU** for higher performance — the control layer (PID / servos / `cmd_vel`) is reused unchanged.

`ROS 2 Humble` · `Raspberry Pi 5 (aarch64)` · `Coral USB Edge TPU` · `Python 3.10` · `Docker`

---

## Why & how

The stock car follows a clean **`perception → message → controller`** split. Swapping in a neural net means **replacing only the perception layer**; the PID/servo/chassis controller stays byte-for-byte identical. Landing form: a `*_tpu.py` node next to the original, no new ROS package. Models are trained/compiled elsewhere — this repo only does *load inference + wire to ROS*.

## Hardware & environment

| Item | Requirement |
|---|---|
| Board | Raspberry Pi 5, **aarch64** |
| Accelerator | Coral USB Accelerator (Edge TPU) — plug into a **USB3** port |
| Runtime | Docker; ROS 2 Humble container `yahboomtechnology/ros-humble:4.1.2` (Python **3.10**) |
| Inference | `tflite_runtime` 2.16.2 (feranick, aarch64/cp310) + `libedgetpu` (feranick, `16.0TF2.16.1`, arm64/ubuntu22.04) + `numpy<2` |

> ⚠️ Use the **aarch64/arm64** wheels & debs — not the x86_64 ones.

## Repository layout

```
.
├── root/                          # car-side /root (baked into the image)
│   ├── yahboomcar_ws/src/         # main workspace (TPU-replaceable vision pkgs)
│   │   ├── yahboomcar_astra/      # perception (face/color/line) + TPU nodes
│   │   ├── yahboomcar_mediapipe/  # MediaPipe hand/pose/face on CPU
│   │   ├── yahboomcar_visual/     # vision basics / detection demo
│   │   └── yahboomcar_msgs/       # custom messages (dependency)
│   ├── config_robot.py            # base-board serial config tool
│   ├── ros2_humble.sh             # container launch script (reference)
│   └── start_agent_rpi5.sh        # micro-ROS agent launch (reference)
├── docker/Dockerfile.tpu          # child-image recipe → :4.1.2-tpu
└── deploy/pack_and_push.sh        # one-click: download deps → bundle → scp to car
```

## Quick start

The Edge TPU deps are added as a **child image** `:4.1.2-tpu` layered on the stock `:4.1.2` — the original image is untouched.

**1. On the dev machine — download deps, assemble the build bundle, push to the car** ([deploy/pack_and_push.sh](deploy/pack_and_push.sh)):

```bash
./deploy/pack_and_push.sh                # download + pack + scp (default car pi@10.42.0.1)
# if internet and the car's hotspot are on different networks, split it:
./deploy/pack_and_push.sh --pack-only    # while online: download + assemble
./deploy/pack_and_push.sh --push-only    # after joining the car hotspot: scp only
```

**2. On the car — build the image** (offline; the bundle is self-contained, see [docker/Dockerfile.tpu](docker/Dockerfile.tpu)):

```bash
cd ~/tpu-build
docker build -t yahboomtechnology/ros-humble:4.1.2-tpu -f Dockerfile.tpu .
```

**3. Run a node** (inside the container):

```bash
source /root/yahboomcar_ws/install/setup.bash
```

**Face following** ([face_fllow_tpu.py](root/yahboomcar_ws/src/yahboomcar_astra/yahboomcar_astra/face_fllow_tpu.py)):

```bash
ros2 run yahboomcar_astra face_fllow_tpu                          # windowed (VNC/desktop)
ros2 run yahboomcar_astra face_fllow_tpu --ros-args -p show_image:=false   # headless (SSH)
```

Parameters: `model_path` (edgetpu tflite path), `conf_threshold` (default `0.3`), `show_image` (default `true`).

**Person / object following** — detector + one of three follow stacks:

```bash
# gimbal-only: reuse the stock controller unchanged
ros2 run yahboomcar_astra colorTracker &
ros2 run yahboomcar_astra objTracker_tpu

# gimbal + LiDAR-ranged base following + nav2 collision safety (stop/slow, no detour)
ros2 launch yahboomcar_astra follow_collision.launch.py &          # static TF + nav2_collision_monitor
ros2 run yahboomcar_astra objControl --ros-args -r /cmd_vel:=/cmd_vel_raw &
ros2 run yahboomcar_astra objTracker_tpu

# Nav2 follow: camera-bearing + LiDAR-range → odom person estimate → Nav2 real obstacle-avoidance detour
# ONE command brings up everything (bringup + Nav2 + person-goal bridge + detector); gimbal keeps you centred, searches when lost
ros2 launch yahboomcar_astra follow_nav2.launch.py                 # default controller:=dwb; controller:=rpp to A/B
```

Detector params: `target_label` (default `person`), `conf_threshold` (default `0.5`), `min_hits` (debounce, default `3`). Controller `objControl` follow/search params (`target_dist`, `front_angle`, `angular_kp`, …) are tunable at runtime — see the node's `declare_param`.

Point the autostart `ros2_humble.sh` at `:4.1.2-tpu` and add `-v /dev/bus/usb:/dev/bus/usb` to give the container USB access to the Edge TPU.

## Features

- **Face following** — [face_fllow_tpu.py](root/yahboomcar_ws/src/yahboomcar_astra/yahboomcar_astra/face_fllow_tpu.py): Haar cascade → Edge TPU SSD MobileNet v2 face detector (≈23 ms/inference, measured), box center fed to the original PID/servo loop.
- **Person / object following** — detector [objTracker_tpu.py](root/yahboomcar_ws/src/yahboomcar_astra/yahboomcar_astra/objTracker_tpu.py): Edge TPU SSD MobileNet v2 **COCO** detector, class-filtered (`target_label`, default `person`) + N-frame debounce, publishing the same `/Current_point` so the stock `colorTracker.py` (gimbal-only) works unchanged. Controller [objControl.py](root/yahboomcar_ws/src/yahboomcar_astra/yahboomcar_astra/objControl.py) adds **LiDAR-ranged base following** (turn from the gimbal angle; drive forward/back to hold a set distance from `/scan`) with a lost-target gimbal search. Obstacle safety via **`nav2_collision_monitor`** ([follow_collision.launch.py](root/yahboomcar_ws/src/yahboomcar_astra/launch/follow_collision.launch.py), footprint-aware stop/slow zones).
- **Nav2 follow** — [person_goal_bridge.py](root/yahboomcar_ws/src/yahboomcar_astra/yahboomcar_astra/person_goal_bridge.py): fuses camera bearing + LiDAR range into a **person estimate in the odom frame** (EMA + occlusion gating so a chair between you and the robot doesn't stop it), feeds it as a moving goal to a map-less **Nav2** stack for **real obstacle-avoidance detour**. Decoupled gimbal points at the estimate (keeps you centred, searches when lost). One-command launch, `controller:=rpp|dwb`.

## Restore to factory

All changes land in **one script** (`/home/pi/ros2_humble.sh` — back it up first) and **one child image**; the stock image and source are untouched. Rollback = restore the script and switch back to `:4.1.2`.

## Acknowledgements

- Yahboom (亚博智能) — original [MicroROS-Pi5](https://www.yahboom.net/study/MicroROS-Pi5) car source.
- [feranick](https://github.com/feranick) — `tflite_runtime` & `libedgetpu` builds for modern kernels/arches.
- [Google Coral](https://github.com/google-coral) — pre-trained Edge TPU face & SSD MobileNet v2 COCO detection models + labels.
