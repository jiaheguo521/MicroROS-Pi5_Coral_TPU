#!/usr/bin/env bash
# =============================================================================
# fetch_models.sh — 下载 Edge TPU Re-ID 模型（托管在 HuggingFace）
# =============================================================================
# 这些是 edgetpu_compiler 的 co-compile 产物，没有官方下载地址、体积大(~85MB)，
# 所以不入 git，克隆仓库后跑本脚本取回。
#
# 用法：
#   ./deploy/fetch_models.sh                    # 默认：全部下载
#   ./deploy/fetch_models.sh reid_youtu_p70     # 只下指定的一个/多个
#   ./deploy/fetch_models.sh --list             # 列出可选模型
#   ./deploy/fetch_models.sh --force            # 已存在也重新下载
#
# 已存在且 sha256 正确的文件会跳过，可重复运行。只依赖 curl/wget + sha256sum。
#
# ⚠️ det 和 emb 必须来自**同一个目录**：每对是一次 co-compile 的产物，两个网络共享
#    同一块 8MB 片上 SRAM、缓存划分是编译时定死的。跨目录混用会导致延迟劣化/行为异常。
# =============================================================================
set -o pipefail

REPO="jiaheguo521/microros-pi5-coral-tpu-models"
BASE_URL="https://huggingface.co/${REPO}/resolve/main"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$SCRIPT_DIR/../root/yahboomcar_ws/src/yahboomcar_astra/yahboomcar_astra/models"

ALL_VARIANTS=(reid_youtu reid_youtu_p70 reid reid_mnv2 reid_osnet05 reid_osnet075)

# 每个模型一行说明（--list 用）
describe() {
  case "$1" in
    reid_youtu)     echo "Youtu ReID ResNet50 未剪枝 — 最准(顶替率4%)，60ms/~4.4Hz  [默认]" ;;
    reid_youtu_p70) echo "Youtu 剪枝70% — 15.6ms/~18Hz(云台更跟手)，顶替率13%" ;;
    reid)           echo "MobileNetV1 通用特征 — 3.7ms，非专训 Re-ID，弱，旧备选" ;;
    reid_mnv2)      echo "MobileNetV2 (Market-1501) — 3.8ms  ⚠ 贴地视角实测不可用" ;;
    reid_osnet05)   echo "OSNet x0.5  (Market-1501) — 8.6ms  ⚠ 贴地视角实测不可用" ;;
    reid_osnet075)  echo "OSNet x0.75 (Market-1501) — 12.6ms ⚠ 贴地视角实测不可用" ;;
  esac
}

# sha256（对应 HF 仓库 main 分支的文件；防止半截下载导致车上诡异崩溃）
sha_of() {
  case "$1" in
    reid/det_reid_edgetpu.tflite)            echo 6f7c8c48f9fac2efe862f69d0630825b6f50865d7ce0a9732ae308ec7a43c1b2 ;;
    reid/emb_reid_edgetpu.tflite)            echo e6a83375deed26255c21e309d291d2c069a5901063a33489630da1efe74745f8 ;;
    reid_youtu/det_reid_edgetpu.tflite)      echo ad1a042308cdc7ec353a26dfcd769760decb3b2c980730df494fa53bd7328b88 ;;
    reid_youtu/emb_reid_edgetpu.tflite)      echo 0fedfdadf5c2b8531af74aa181cb6702e7f63609ac094b3eac6c53050c3d2709 ;;
    reid_youtu_p70/det_reid_edgetpu.tflite)  echo 10d99e70024487e373c6c79f5d19b0a7cd0ea42a8aed00019f55e28c48d323a6 ;;
    reid_youtu_p70/emb_reid_edgetpu.tflite)  echo 0c2edbcc39c185437ddf02c7a322974b71e3d7f4253b863e02168173e6c1a49f ;;
    reid_mnv2/det_reid_edgetpu.tflite)       echo 3c88d73c980d8e54287b7bfbba1b46ab136d8bf371d08add1bef97c188736a59 ;;
    reid_mnv2/emb_reid_edgetpu.tflite)       echo aed85ea222906b7be4742356a50593255529cff4b5650bbc4bd33f6bb98eb746 ;;
    reid_osnet05/det_reid_edgetpu.tflite)    echo 3aabb94dd2e9ba1e401bf20cebfdceff3f5ea1377dbdec39471504af7d6b85ea ;;
    reid_osnet05/emb_reid_edgetpu.tflite)    echo e6f366ff8b9f1e7b3c23fabc06315eff94b260e7b0c9c72dcc45d2be0082f42c ;;
    reid_osnet075/det_reid_edgetpu.tflite)   echo c97618b5487535fb931426b3f7ecf8369b818de841829d78ecd7c6d3b21a5be7 ;;
    reid_osnet075/emb_reid_edgetpu.tflite)   echo 8e334a6149f2e90801ac17893a16dd3a9549043cba675812ad18aee4165c9b53 ;;
  esac
}

FORCE=0
SELECTED=()
while [ $# -gt 0 ]; do
  case "$1" in
    --list)
      echo "可选模型（默认全下）："
      for v in "${ALL_VARIANTS[@]}"; do printf "  %-16s %s\n" "$v" "$(describe "$v")"; done
      echo
      echo "托管于 https://huggingface.co/${REPO}"
      exit 0 ;;
    --force) FORCE=1; shift ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*) echo "未知参数: $1（用 --help 看用法）" >&2; exit 1 ;;
    *)  SELECTED+=("$1"); shift ;;
  esac
done
[ ${#SELECTED[@]} -eq 0 ] && SELECTED=("${ALL_VARIANTS[@]}")

# 校验选择的名字合法
for v in "${SELECTED[@]}"; do
  ok=0; for a in "${ALL_VARIANTS[@]}"; do [ "$v" = "$a" ] && ok=1; done
  if [ $ok -eq 0 ]; then
    echo "未知模型: $v" >&2; echo "可用：${ALL_VARIANTS[*]}" >&2; exit 1
  fi
done

# 下载器：优先 curl，退回 wget
if command -v curl >/dev/null 2>&1;   then DL="curl -fL --retry 3 --retry-delay 2 -# -o"
elif command -v wget >/dev/null 2>&1; then DL="wget -q --tries=3 -O"
else echo "需要 curl 或 wget" >&2; exit 1; fi

verify() {  # verify <文件路径> <期望sha256>；一致返回 0
  [ -f "$1" ] || return 1
  [ -n "$2" ] || return 0                       # 没记录 sha 就只看存在
  command -v sha256sum >/dev/null 2>&1 || return 0
  [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ]
}

echo ">>> 目标目录: $DEST"
echo ">>> 来源: https://huggingface.co/${REPO}"
mkdir -p "$DEST"
failed=0; got=0; skipped=0

for v in "${SELECTED[@]}"; do
  mkdir -p "$DEST/$v"
  for f in det_reid_edgetpu.tflite emb_reid_edgetpu.tflite; do
    rel="$v/$f"; out="$DEST/$rel"; want="$(sha_of "$rel")"

    if [ $FORCE -eq 0 ] && verify "$out" "$want"; then
      echo "  ✓ 已存在且校验通过，跳过  $rel"; skipped=$((skipped+1)); continue
    fi

    echo "  ↓ 下载 $rel"
    if ! $DL "$out.part" "$BASE_URL/$rel"; then
      echo "    ✗ 下载失败: $rel" >&2; rm -f "$out.part"; failed=$((failed+1)); continue
    fi
    if ! verify "$out.part" "$want"; then
      echo "    ✗ sha256 校验失败(文件可能损坏/被截断): $rel" >&2
      rm -f "$out.part"; failed=$((failed+1)); continue
    fi
    mv "$out.part" "$out"; got=$((got+1))
  done
done

echo
echo "完成：新下载 $got 个，跳过 $skipped 个，失败 $failed 个。"
if [ $failed -gt 0 ]; then
  echo "有文件未取到，请检查网络后重跑（已下好的会自动跳过）。" >&2
  exit 1
fi
echo "接下来可以打包上车： ./deploy/pack_and_push.sh"
