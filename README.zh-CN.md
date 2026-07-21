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
├── deploy/fetch_models.sh         # 一键：从 HuggingFace 下载 Re-ID 模型（默认全下）
└── deploy/pack_and_push.sh        # 一键：下载依赖 → 组装 → scp 到车
```

## 快速开始

TPU 依赖以**子镜像** `:4.1.2-tpu` 的形式叠在原厂 `:4.1.2` 上，原镜像不动。

**0. 克隆后先取模型**（[deploy/fetch_models.sh](deploy/fetch_models.sh)）：Re-ID 模型是 `edgetpu_compiler`
的 co-compile 产物、无官方下载地址且体积较大（~85MB），不入 git，托管在
[HuggingFace](https://huggingface.co/jiaheguo521/microros-pi5-coral-tpu-models)：

```bash
./deploy/fetch_models.sh              # 默认全下（已存在且校验通过会跳过，可重复运行）
./deploy/fetch_models.sh --list       # 看有哪些模型、各自的速度/精度权衡
./deploy/fetch_models.sh reid_youtu   # 只下某一个
```

> ⚠️ `det` 和 `emb` 必须来自**同一个目录**——每对是一次 co-compile 的产物，两个网络共享同一块
> 8MB 片上 SRAM、缓存划分在编译期定死，跨目录混用会导致延迟劣化和行为异常。
> （SSD 人脸/COCO 检测器有 Coral 官方地址，`pack_and_push.sh` 会自动下载并缓存。）

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
# 云台主动跟人构图 + 跟丢后高阈值重锁，均已默认开启
ros2 launch yahboomcar_astra follow_nav2.launch.py detector:=objTracker_reid_tpu
```

检测器参数：`target_label`（默认 `person`）、`conf_threshold`（默认 `0.5`）、`min_hits`（去抖，默认 `3`）。控制器 `objControl` 的跟随/搜索参数（`target_dist`、`front_angle`、`angular_kp` …）可运行时调，见节点 `declare_param`。

### 现场调参指南（都是 launch 参数，改完即生效、无需重建镜像）

**下列默认值只是本车实测的起点**，和相机高度、场地、穿着强相关，建议现场调。

**1) Re-ID 身份阈值（最重要，换 `reid_model` 必须重标）**——先开 `debug_sim:=true` 看每帧各候选的实际相似度（`*`=被选中），让**目标本人**和**旁人**分别走动，记下两组范围再定：

| 参数 | 默认 | 怎么定 |
|---|---|---|
| `sim_floor` | `0.6` | 维持已锁目标的门槛。**必须高于"旁人的最高分"**，否则目标一走开、旁人勉强过线就被当成目标接上（且系统会一直停在"正在跟"状态，下面那道重捕闸永远不触发）；但要低于目标侧身/走远时的分数，否则老丢自己。 |
| `relock_sim_floor` | `0.75` | **跟丢后重捕**的高门槛，要比 `sim_floor` 明显高（建议 +0.1 以上）。 |
| `relock_min_hits` | `5` | 重捕还需连续这么多帧都过高门槛才认。 |
| `ema_min_sim` | `0.65` | 只有分数≥此值才把外观写回模板，防边界误匹配带偏模板。 |

> 本车实测（未剪枝 Youtu）：目标 0.87~0.98、旁人 0.09~0.65 → `sim_floor` 取 0.65~0.70、`relock_sim_floor` 取 0.75~0.80 较稳妥。**换模型后全部作废，必须重标。**

**2) 云台构图**：`tilt_head_frac`(0.3，头老出画就调大)、`tilt_kp`/`tilt_smooth`(0.05/0.6，上下抖就减 kp 增 smooth)、`tilt_sign`/`servo_sign`(舵机极性，转反了取反号)、`pan_max_step_deg`(6.0，横走跟不上调大)、`enable_gimbal_tilt`/`enable_gimbal_pan`(可单独关一路来定位问题)。

**3) 跟随距离/近距停车**：`standoff`(1.5m)；`close_stop_px`(320)——检测框(EMA)超过此值就锁死不前进，**近距时相机方位和雷达距离都会失效，只有"框够大"可靠**；贴太近就调小、太早停就调大，按日志 `follow: dist=.. size=.. ema=..` 走到想停的距离取当时的 `ema` 值。`enable_lost_search`(false，跟丢后云台默认不动、保持最后指向；要开可配 `search_step_deg`)。

**4) 帧率 vs 精度**：`reid_model:=reid_youtu`(默认最准 ~4.4Hz)｜`reid_youtu_p70`(~18Hz，云台明显跟手，身份区分略弱)。**云台延迟主要由检测帧率决定、不是 CPU**；换模型记得重标阈值。

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
