#!/usr/bin/env bash
# 一键：下载 TPU 离线依赖(带缓存) → 组装构建包(依赖 + 需要传给小车的环境文件) → scp 到小车
#
# 用法：
#   ./deploy/pack_and_push.sh                # 下载+打包+推送（默认小车 pi@10.42.0.1）
#   ./deploy/pack_and_push.sh --pack-only    # 只下载+打包，不推送（本机联网、还没连车热点时）
#   ./deploy/pack_and_push.sh --push-only    # 只推送已打好的包（已连车热点时）
#   ./deploy/pack_and_push.sh --host pi@1.2.3.4
#
# 之后到小车上构建：
#   cd ~/tpu-build && docker build -t yahboomtechnology/ros-humble:4.1.2-tpu -f Dockerfile.tpu .
set -euo pipefail

# ---------------- 配置 ----------------
PI_HOST="${PI_HOST:-pi@10.42.0.1}"        # 小车（热点网关）
PI_DEST="${PI_DEST:-tpu-build}"            # 小车上的构建目录（相对远端 home；scp 走 SFTP 不展开 $HOME）
IMAGE_TAG="yahboomtechnology/ros-humble:4.1.2-tpu"

TFLITE_WHL="tflite_runtime-2.16.2-cp310-cp310-linux_aarch64.whl"
EDGETPU_DEB="libedgetpu1-std_16.0tf2.16.1-1.ubuntu22.04_arm64.deb"
LIBUSB_DEB="libusb-1.0-0_arm64.deb"
MODEL="ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite"
COCO_MODEL="ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
COCO_LABELS="coco_labels.txt"
REID_DET="det_reid_edgetpu.tflite"
REID_EMB="emb_reid_edgetpu.tflite"

TFLITE_WHL_URL="https://github.com/feranick/TFlite-builds/releases/download/v2.16.2/${TFLITE_WHL}"
EDGETPU_DEB_URL="https://github.com/feranick/libedgetpu/releases/download/16.0TF2.16.1-1/${EDGETPU_DEB}"
LIBUSB_DEB_URL="http://ports.ubuntu.com/ubuntu-ports/pool/main/libu/libusb-1.0/libusb-1.0-0_1.0.25-1ubuntu2_arm64.deb"
MODEL_URL="https://github.com/google-coral/edgetpu/raw/master/test_data/${MODEL}"
COCO_MODEL_URL="https://github.com/google-coral/edgetpu/raw/master/test_data/${COCO_MODEL}"
COCO_LABELS_URL="https://github.com/google-coral/edgetpu/raw/master/test_data/${COCO_LABELS}"

# ---------------- 路径 ----------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ASTRA="$REPO_ROOT/root/yahboomcar_ws/src/yahboomcar_astra"
DOCKERFILE="$REPO_ROOT/docker/Dockerfile.tpu"
CACHE="$SCRIPT_DIR/dist/cache"            # 下载缓存（跨次复用，gitignore）
BUNDLE="$SCRIPT_DIR/dist/tpu-build"       # 组装好的构建包（scp 到小车，gitignore）

# ---------------- 参数 ----------------
DO_PACK=1; DO_PUSH=1
while [ $# -gt 0 ]; do
  case "$1" in
    --pack-only) DO_PUSH=0 ;;
    --push-only) DO_PACK=0 ;;
    --host) PI_HOST="$2"; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
  shift
done

dl() {  # dl <url> <dest>；已存在则跳过
  if [ -f "$2" ]; then echo "  缓存命中 $(basename "$2")"; return; fi
  echo "  下载 $(basename "$2")"; wget -q --show-progress -O "$2" "$1"
}

if [ "$DO_PACK" = 1 ]; then
  mkdir -p "$CACHE" "$BUNDLE/models" "$BUNDLE/models/reid" "$BUNDLE/models/reid_youtu" \
           "$BUNDLE/models/reid_osnet05" "$BUNDLE/models/reid_osnet075" "$BUNDLE/models/reid_mnv2" \
           "$BUNDLE/models/reid_youtu_p70"

  echo "[1/3] 下载依赖(带缓存) → $CACHE"
  dl "$TFLITE_WHL_URL"  "$CACHE/$TFLITE_WHL"
  dl "$EDGETPU_DEB_URL" "$CACHE/$EDGETPU_DEB"
  dl "$LIBUSB_DEB_URL"  "$CACHE/$LIBUSB_DEB"

  echo "[2/3] 组装构建包 → $BUNDLE"
  cp "$CACHE/$TFLITE_WHL"  "$BUNDLE/"
  cp "$CACHE/$EDGETPU_DEB" "$BUNDLE/"
  cp "$CACHE/$LIBUSB_DEB"  "$BUNDLE/"
  cp "$DOCKERFILE"                               "$BUNDLE/Dockerfile.tpu"          # 构建配方
  cp "$ASTRA/setup.py"                           "$BUNDLE/setup.py"                # 入口(环境文件)
  cp "$ASTRA/yahboomcar_astra/face_fllow_tpu.py" "$BUNDLE/face_fllow_tpu.py"       # 节点真源
  cp "$ASTRA/yahboomcar_astra/objTracker_tpu.py" "$BUNDLE/objTracker_tpu.py"       # 目标跟随检测器真源
  cp "$ASTRA/yahboomcar_astra/objTracker_reid_tpu.py" "$BUNDLE/objTracker_reid_tpu.py"  # Re-ID 锁定检测器真源
  cp "$ASTRA/yahboomcar_astra/objControl.py"     "$BUNDLE/objControl.py"           # 目标跟随控制器(云台+底盘)真源
  cp "$ASTRA/yahboomcar_astra/person_goal_bridge.py" "$BUNDLE/person_goal_bridge.py"  # 路B 跟人目标桥真源
  cp "$ASTRA/launch/follow_collision.launch.py"  "$BUNDLE/follow_collision.launch.py"  # 路C collision_monitor launch
  cp "$ASTRA/launch/follow_nav2.launch.py"       "$BUNDLE/follow_nav2.launch.py"    # 路B map-less Nav2 跟人 launch
  cp "$ASTRA/config/nav2_follow.yaml"            "$BUNDLE/nav2_follow.yaml"         # 路B Nav2 配置(map-less RPP)
  cp "$ASTRA/config/nav2_follow_dwb.yaml"        "$BUNDLE/nav2_follow_dwb.yaml"     # 路B Nav2 配置(DWB 对比，仅控制器块不同)
  if [ -f "$ASTRA/yahboomcar_astra/models/$MODEL" ]; then                          # 模型：优先仓库真源
    cp "$ASTRA/yahboomcar_astra/models/$MODEL" "$BUNDLE/models/"
  else
    dl "$MODEL_URL" "$CACHE/$MODEL"; cp "$CACHE/$MODEL" "$BUNDLE/models/"
  fi
  if [ -f "$ASTRA/yahboomcar_astra/models/$COCO_MODEL" ]; then                     # COCO 模型：优先仓库真源
    cp "$ASTRA/yahboomcar_astra/models/$COCO_MODEL" "$BUNDLE/models/"
  else
    dl "$COCO_MODEL_URL" "$CACHE/$COCO_MODEL"; cp "$CACHE/$COCO_MODEL" "$BUNDLE/models/"
  fi
  if [ -f "$ASTRA/yahboomcar_astra/models/$COCO_LABELS" ]; then                    # COCO labels：优先仓库真源
    cp "$ASTRA/yahboomcar_astra/models/$COCO_LABELS" "$BUNDLE/models/"
  else
    dl "$COCO_LABELS_URL" "$CACHE/$COCO_LABELS"; cp "$CACHE/$COCO_LABELS" "$BUNDLE/models/"
  fi
  # Re-ID co-compile 产物：edgetpu_compiler 编出来的，官方没有发布地址，体积大(~85MB)不入 git，
  # 托管在 HuggingFace，缺失就提示跑 ./deploy/fetch_models.sh 取回。
  # reid_youtu/=默认(未剪 base,最准) reid_youtu_p70/=剪70%(~18Hz,帧率优先) reid/=MobileNetV1(弱)
  # reid_osnet05|osnet075|mnv2/=单源 Market(贴地不可用,仅平视备选)，每对独立 co-compile 不能混用
  for reid_dir in reid reid_youtu reid_osnet05 reid_osnet075 reid_mnv2 reid_youtu_p70; do
    for f in "$REID_DET" "$REID_EMB"; do
      if [ -f "$ASTRA/yahboomcar_astra/models/$reid_dir/$f" ]; then
        cp "$ASTRA/yahboomcar_astra/models/$reid_dir/$f" "$BUNDLE/models/$reid_dir/"
      else
        echo "缺少模型 $reid_dir/$f" >&2
        echo "  → 先跑：./deploy/fetch_models.sh        （默认全下，已有的会跳过）" >&2
        echo "  → 或只下这一个：./deploy/fetch_models.sh $reid_dir" >&2
        exit 1
      fi
    done
  done
  echo "  构建包内容："; ls -la "$BUNDLE"
fi

if [ "$DO_PUSH" = 1 ]; then
  [ -d "$BUNDLE" ] || { echo "没有构建包，先跑 --pack-only"; exit 1; }
  echo "[3/3] 推送到小车 $PI_HOST:$PI_DEST"
  ssh "$PI_HOST" "mkdir -p $PI_DEST"
  scp -r "$BUNDLE/." "$PI_HOST:$PI_DEST/"
  echo
  echo "完成。到小车上构建镜像："
  echo "  ssh $PI_HOST"
  echo "  cd ~/tpu-build && docker build -t $IMAGE_TAG -f Dockerfile.tpu ."
fi
