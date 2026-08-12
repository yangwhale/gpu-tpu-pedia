#!/usr/bin/env bash
# 在 torch-tpu base 镜像的 pod 里把 vllm-torchtpu 装到可推理状态。
# 2026-08-12 在 bodaborg-tpu7x-nap / tpu7x 2x2x1（4 chip）上实测通过，
# 端到端到 Qwen3-0.6B 出正确输出。
#
# 用法（在 pod 内）：
#   export AR_TOKEN="$(在集群外执行 gcloud auth print-access-token 得到)"
#   bash setup-in-pod.sh /work/vllm-torchtpu
#
# 前提：/work/vllm-torchtpu 已有源码（kubectl cp 或 git clone 均可）。
set -euo pipefail

SRC="${1:-/work/vllm-torchtpu}"
WORK="$(dirname "$SRC")"
: "${AR_TOKEN:?需要 AR_TOKEN（gcloud auth print-access-token）}"

export DEBIAN_FRONTEND=noninteractive
export PIP_INDEX_URL="https://oauth2accesstoken:${AR_TOKEN}@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/"
export PIP_EXTRA_INDEX_URL="https://pypi.org/simple"
export VLLM_TARGET_DEVICE=tpu

echo "=== [1/5] 工具链（base 镜像缺 git / cmake / 编译器 / curl）==="
apt-get update -qq
# curl 看着可有可无，实际是致命的：run_benchmarks.sh 用
#   curl -s -o /dev/null --connect-timeout 1 http://localhost:$PORT/health
# 探活。镜像里没有 curl → 命令永远失败 → runner 永远认为 server 没起来，
# 一路刷 "Waiting for server..." 直到 pod 被杀。而 server 其实早就 200 了。
# 这一个缺失的包，今晚废掉了三个 60 分钟窗口。
apt-get install -y -qq git cmake build-essential ninja-build curl
git --version; cmake --version | head -1; curl --version | head -1

echo "=== [2/5] 版本矩阵 —— 必须先装，且必须用 pip 不用 uv ==="
# uv 不读 PIP_INDEX_URL，会报 "torch-tpu was not found in the package registry"。
# 版本来源：vllm-torchtpu/pyproject.toml 的 dependencies，那里是唯一权威。
python3 -m pip install --no-cache-dir \
  "jax==0.10.2" "jaxlib==0.10.2" "libtpu==0.0.44.1" \
  "torch==2.13.0" "torchvision==0.28.0" \
  "torch-tpu==0.1.1.dev20260804130134" \
  "tpu-raiden-torch==0.0.1.dev20260808010148" \
  "numba==0.65.0" "tpu-info" "portpicker" "pathwaysutils"

echo "=== [3/5] vLLM v0.26.1rc0 ==="
# 优先用随源码包一起传进来的 tarball。pod 内直连 GitHub 实测不稳定：
# 踩过 HTTP 503 和 RemoteDisconnected 各一次，每次废掉一个 60 分钟窗口。
if [ ! -d "$WORK/vllm" ]; then
  if [ -s "$WORK/vllm.tgz" ]; then
    echo "使用随包传入的 vllm.tgz（$(du -h "$WORK/vllm.tgz" | cut -f1)）"
  else
    echo "本地无 vllm.tgz，回退到 GitHub 下载（最多 5 次）"
    VLLM_URL="https://github.com/vllm-project/vllm/archive/refs/tags/v0.26.1rc0.tar.gz"
    ok=0
    for attempt in 1 2 3 4 5; do
      if curl -fsSL --connect-timeout 20 --max-time 300 -o "$WORK/vllm.tgz" "$VLLM_URL"; then ok=1; break; fi
      echo "第 $attempt 次下载失败，等 $((attempt * 10))s 后重试"
      sleep $((attempt * 10))
    done
    [ "$ok" = 1 ] || { echo "SETUP_FAILED: vLLM 源码下载 5 次均失败"; exit 1; }
  fi
  tar xzf "$WORK/vllm.tgz" -C "$WORK" && mv "$WORK/vllm-0.26.1rc0" "$WORK/vllm"
fi
[ -d "$WORK/vllm" ] || { echo "SETUP_FAILED: vllm 源码目录不存在"; exit 1; }
# 必须删掉，否则会把上游 tpu-inference plugin 跟 vllm-torchtpu 装到一起
sed -i '/tpu-inference/d' "$WORK/vllm/requirements/tpu.txt"
SETUPTOOLS_SCM_PRETEND_VERSION=0.26.1rc0 VLLM_TARGET_DEVICE=tpu MAX_JOBS=32 \
  python3 -m pip install --no-cache-dir -e "$WORK/vllm"

echo "=== [4/5] vllm_torchtpu —— 必须 --no-deps ==="
# 不加 --no-deps 时 pip 会因为 "vllm @ git+..." 这条依赖去从 git 重建 vLLM，
# 而构建隔离环境里 VLLM_TARGET_DEVICE 落回 cpu → 编译失败 → 整步回滚。
# 用 tarball 传源码时 .git 缺失，还需要下面这个 PRETEND_VERSION。
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU=0.1.0
python3 -m pip install --no-cache-dir --pre --no-deps -e "$SRC"

echo "=== [5/5] 验证（必须 cd 出源码目录，否则 import vllm 会串到本地目录）==="
cd /tmp
python3 - <<'PY'
import importlib.metadata as m
for p in ["torch","jax","jaxlib","libtpu","torch-tpu","torchvision","vllm"]:
    try: print(f"  {p:14s} {m.version(p)}")
    except Exception: print(f"  {p:14s} MISSING")
from jax.experimental.pallas import tpu as pltpu
assert hasattr(pltpu, "BufferType"), "jax 版本不对：pallas.tpu 缺 BufferType"
import vllm, vllm_torchtpu, torch, torch_tpu
from vllm.platforms import current_platform
print("  platform     :", type(current_platform).__name__)
print("  tpu devices  :", torch.tpu.device_count())
print("SETUP_OK")
PY
