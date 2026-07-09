# Yahboom MicroROS-Pi5 × Coral Edge TPU

[English](README.md) · **简体中文**

**Yahboom MicroROS-Pi5** 小车的二次开发项目：把原厂视觉感知（传统 CV / CPU 上的 CNN）替换成**跑在 Coral USB Edge TPU 上的已编译 `*_edgetpu.tflite` 模型**以提升性能——控制层（PID / 舵机 / `cmd_vel`）原样复用。

`ROS 2 Humble` · `树莓派 5（aarch64）` · `Coral USB Edge TPU` · `Python 3.10` · `Docker`

---

## 为什么 & 怎么做

原厂小车是清晰的 **`感知 → 消息 → 控制器`** 分层。换神经网络＝**只换感知层**，PID/舵机/底盘控制器一行不动。落地形式：在原节点旁新增 `*_tpu.py`，不新建 ROS 包。模型的训练/编译在别处完成，本仓库只负责*加载推理 + 接 ROS*。

## 硬件与环境

| 项 | 要求 |
|---|---|
| 主控 | 树莓派 5，**aarch64** |
| 加速器 | Coral USB Accelerator（Edge TPU）——插 **USB3** 口 |
| 运行时 | Docker；ROS 2 Humble 容器 `yahboomtechnology/ros-humble:4.1.2`（Python **3.10**） |
| 推理 | `tflite_runtime` 2.16.2（feranick，aarch64/cp310）+ `libedgetpu`（feranick，`16.0TF2.16.1`，arm64/ubuntu22.04）+ `numpy<2` |

> ⚠️ 必须用 **aarch64/arm64** 的轮子和 deb，别用 x86_64 版。

## 仓库结构

```
.
├── root/                          # 车端 /root（烤进镜像）
│   ├── yahboomcar_ws/src/         # 主工作空间（可 TPU 替换的视觉包）
│   │   ├── yahboomcar_astra/      # 感知（人脸/颜色/巡线）+ TPU 节点
│   │   ├── yahboomcar_mediapipe/  # MediaPipe 手势/姿态/人脸（CPU）
│   │   ├── yahboomcar_visual/     # 视觉基础 / 检测 demo
│   │   └── yahboomcar_msgs/       # 自定义消息（依赖）
│   ├── config_robot.py            # 底板串口配置工具
│   ├── ros2_humble.sh             # 容器启动脚本（参考）
│   └── start_agent_rpi5.sh        # micro-ROS agent 启动（参考）
├── docker/Dockerfile.tpu          # 子镜像构建配方 → :4.1.2-tpu
└── deploy/pack_and_push.sh        # 一键：下载依赖 → 组装 → scp 到车
```

## 快速开始

TPU 依赖以**子镜像** `:4.1.2-tpu` 的形式叠在原厂 `:4.1.2` 上，原镜像不动。

**1. 开发机——下载依赖、组装构建包、推送到车**（[deploy/pack_and_push.sh](deploy/pack_and_push.sh)）：

```bash
./deploy/pack_and_push.sh                # 下载+打包+scp（默认小车 pi@10.42.0.1）
# 若"下载"和"车热点"不在同一网络，分两步：
./deploy/pack_and_push.sh --pack-only    # 有公网时：下载+组装
./deploy/pack_and_push.sh --push-only    # 连上车热点后：只 scp
```

**2. 车上——构建镜像**（离线，构建包自包含，见 [docker/Dockerfile.tpu](docker/Dockerfile.tpu)）：

```bash
cd ~/tpu-build
docker build -t yahboomtechnology/ros-humble:4.1.2-tpu -f Dockerfile.tpu .
```

**3. 运行节点**（容器内）：

```bash
source /root/yahboomcar_ws/install/setup.bash
```

**人脸跟随**（[face_fllow_tpu.py](root/yahboomcar_ws/src/yahboomcar_astra/yahboomcar_astra/face_fllow_tpu.py)）：

```bash
ros2 run yahboomcar_astra face_fllow_tpu                          # 有头（VNC/桌面）
ros2 run yahboomcar_astra face_fllow_tpu --ros-args -p show_image:=false   # 无头（SSH）
```

参数：`model_path`（edgetpu tflite 路径）、`conf_threshold`（默认 `0.3`）、`show_image`（默认 `true`）。

**目标 / 人跟随**——检测器 + 三种跟随栈三选一：

```bash
# 仅云台：原样复用原厂控制器
ros2 run yahboomcar_astra colorTracker &
ros2 run yahboomcar_astra objTracker_tpu

# 云台 + 雷达定距底盘跟随 + nav2 防撞（只停/减速，不绕行）
ros2 launch yahboomcar_astra follow_collision.launch.py &          # 静态 TF + nav2_collision_monitor
ros2 run yahboomcar_astra objControl --ros-args -r /cmd_vel:=/cmd_vel_raw &
ros2 run yahboomcar_astra objTracker_tpu

# Nav2 跟随：相机方位 + 雷达距离 → odom 系人位姿估计 → 喂 Nav2 真绕行避障
# 一条命令起全套（bringup + Nav2 + 跟人目标桥 + 检测器）；云台把你保持在画面中央、丢失时慢扫搜索
ros2 launch yahboomcar_astra follow_nav2.launch.py                 # 默认 controller:=dwb；controller:=rpp 可 A/B 对比

# Nav2 跟随 + Re-ID 身份锁定（第二个 TPU 网络，拒误检/多人不跳/遮挡重认）
# enable_gimbal_pan:=false：锁定前云台会一直慢扫搜索，跟"站稳3秒才锁定"手势冲突，故关闭
ros2 launch yahboomcar_astra follow_nav2.launch.py detector:=objTracker_reid_tpu enable_gimbal_pan:=false
```

检测器参数：`target_label`（默认 `person`）、`conf_threshold`（默认 `0.5`）、`min_hits`（去抖，默认 `3`）。控制器 `objControl` 的跟随/搜索参数（`target_dist`、`front_angle`、`angular_kp` …）可运行时调，见节点 `declare_param`。

把自启脚本 `ros2_humble.sh` 指向 `:4.1.2-tpu` 并加 `-v /dev/bus/usb:/dev/bus/usb`，让容器能访问 Edge TPU 的 USB 设备。

## 功能

- **人脸跟随** — [face_fllow_tpu.py](root/yahboomcar_ws/src/yahboomcar_astra/yahboomcar_astra/face_fllow_tpu.py)：Haar 级联 → Edge TPU 上的 SSD MobileNet v2 人脸检测（实测单帧 invoke ≈23 ms），取框中心喂给原 PID/舵机回路。
- **目标 / 人跟随** — 检测器 [objTracker_tpu.py](root/yahboomcar_ws/src/yahboomcar_astra/yahboomcar_astra/objTracker_tpu.py)：Edge TPU SSD MobileNet v2 **COCO** 检测，按类别过滤（`target_label`，默认 `person`）+ 连续 N 帧去抖，发布同样的 `/Current_point`，故原厂 `colorTracker.py`（仅云台）可零改动复用。控制器 [objControl.py](root/yahboomcar_ws/src/yahboomcar_astra/yahboomcar_astra/objControl.py) 增加**雷达定距底盘跟随**（转向追云台偏角；按 `/scan` 前后保持设定距离）+ 丢目标云台扫描。避障用 **`nav2_collision_monitor`**（[follow_collision.launch.py](root/yahboomcar_ws/src/yahboomcar_astra/launch/follow_collision.launch.py)，footprint 感知的停止/减速区）。
- **Nav2 跟随** — [person_goal_bridge.py](root/yahboomcar_ws/src/yahboomcar_astra/yahboomcar_astra/person_goal_bridge.py)：把相机方位 + 雷达距离融合成 **odom 系的人位姿估计**（EMA + 遮挡门控，人和车之间被椅子挡住也不停车），当移动目标喂给 map-less **Nav2** 做**真绕行避障**。云台解耦、指向估计（把你保持在画面中央、丢失时慢扫搜索）。一条命令起全套，`controller:=rpp|dwb`。
- **多人单目标锁定 / Re-ID** — 检测器 [objTracker_reid_tpu.py](root/yahboomcar_ws/src/yahboomcar_astra/yahboomcar_astra/objTracker_reid_tpu.py)：第二个 TPU 网络（Tencent Youtu Lab 行人 Re-ID，ResNet50 backbone，本项目自己重新量化+编译）逐候选算外观指纹，锁定一个人、拒误检、多人不跳、遮挡后重认；站稳 3 秒（够近+不动）触发初始锁定，或发 `/Reid_Lock` 话题立即换人。上游接在 `/Current_point`，桥节点零改动。

## 恢复到原厂

所有改动只落在**一个脚本**（`/home/pi/ros2_humble.sh`，先自行备份）和**一个子镜像**上，原镜像与源码零改动。回滚＝恢复脚本 + 切回 `:4.1.2`。

## 致谢 / 来源

- Yahboom（亚博智能）——原厂 [MicroROS-Pi5](https://www.yahboom.net/study/MicroROS-Pi5) 小车源码。
- [feranick](https://github.com/feranick)——面向新内核/新架构的 `tflite_runtime` 与 `libedgetpu` 构建。
- [Google Coral](https://github.com/google-coral)——预训练 Edge TPU 人脸 & SSD MobileNet v2 COCO 检测模型 + 标签。
