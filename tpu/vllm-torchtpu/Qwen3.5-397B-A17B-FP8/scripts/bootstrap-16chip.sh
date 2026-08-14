#!/usr/bin/env bash
# 每个 worker 上跑一次：装环境 + 拉权重。两件事并行 —— 一个打 pypi，一个打 GCS，
# 互不抢带宽瓶颈。上周在 bodaborg 是串行的，白白多花 7 分钟。
set -uo pipefail

SRC=/work/vllm-torchtpu
MODELS=/work/models/qwen3.5-397b
: "${AR_TOKEN:?需要 AR_TOKEN}"

export DEBIAN_FRONTEND=noninteractive
export PIP_INDEX_URL="https://oauth2accesstoken:${AR_TOKEN}@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/"
export PIP_EXTRA_INDEX_URL="https://pypi.org/simple"
export VLLM_TARGET_DEVICE=tpu
export PYTHONUNBUFFERED=1

ts(){ echo "@@ $1 $(date -u +%H:%M:%S)"; }
ts BOOT_START

echo "=== [1/6] 工具链 ==="
# curl 不是可选项：run_benchmarks.sh 用 curl 探 /health，缺了它 runner 会永远
# 刷 "Waiting for server" 而 server 其实早就 200 了。这一个包上周废掉三个 60 分钟窗口。
apt-get update -qq 2>&1 | tail -2
apt-get install -y -qq git cmake build-essential ninja-build curl 2>&1 | tail -3
command -v curl >/dev/null || { echo "BOOTSTRAP_FAILED: curl 没装上"; exit 1; }
ts TOOLCHAIN_DONE

echo "=== [2/6] GCS client（权重下载要用）==="
python3 -m pip install --no-cache-dir -q google-cloud-storage 2>&1 | tail -2
ts GCSLIB_DONE

echo "=== [3/6] 后台起权重下载（与 pip 安装并行）==="
nohup python3 /work/fetch-weights.py "$MODELS" > /work/fetch.log 2>&1 &
FETCH_PID=$!
echo "fetch pid=$FETCH_PID"

echo "=== [4/6] 版本矩阵 ==="
# 镜像自带 torch 2.11 / jax 0.9.2 / libtpu 0.0.41，比 pyproject.toml 要求的旧一档，必须升。
# 必须用 pip 不用 uv —— uv 不读 PIP_INDEX_URL，会报 torch-tpu not found。
python3 -m pip install --no-cache-dir \
  "jax==0.10.2" "jaxlib==0.10.2" "libtpu==0.0.44.1" \
  "torch==2.13.0" "torchvision==0.28.0" \
  "torch-tpu==0.1.1.dev20260804130134" \
  "tpu-raiden-torch==0.0.1.dev20260808010148" \
  "numba==0.65.0" "tpu-info" "portpicker" "pathwaysutils" 2>&1 | tail -5
ts MATRIX_DONE

echo "=== [5/6] vLLM + vllm_torchtpu ==="
[ -d /work/vllm ] || { tar xzf /work/vllm-src.tgz -C /work && mv /work/vllm-0.26.1rc0 /work/vllm 2>/dev/null; }
[ -d /work/vllm ] || { echo "BOOTSTRAP_FAILED: vllm 源码不存在"; exit 1; }
# 不删这行会把上游 tpu-inference plugin 和 vllm-torchtpu 装到一起，两个 platform 插件打架
sed -i '/tpu-inference/d' /work/vllm/requirements/tpu.txt
SETUPTOOLS_SCM_PRETEND_VERSION=0.26.1rc0 VLLM_TARGET_DEVICE=tpu MAX_JOBS=64 \
  python3 -m pip install --no-cache-dir -e /work/vllm 2>&1 | tail -4
# --no-deps 必须加：否则 pip 会顺着 "vllm @ git+..." 去从 git 重建 vLLM，
# 而构建隔离环境里 VLLM_TARGET_DEVICE 落回 cpu，编译失败并回滚整步。
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU=0.1.0
python3 -m pip install --no-cache-dir --pre --no-deps -e "$SRC" 2>&1 | tail -3
ts PIP_DONE

echo "=== [6/6] 等权重 + 验证 ==="
wait $FETCH_PID; FETCH_RC=$?
tail -3 /work/fetch.log
[ $FETCH_RC -eq 0 ] || { echo "BOOTSTRAP_FAILED: 权重下载失败"; exit 1; }
ts WEIGHTS_DONE

cd /tmp   # 必须离开源码目录，否则 import vllm 会串到本地同名目录
# GKE 注入的是多机 mesh env（TPU_PROCESS_ADDRESSES 列了全部 4 台）。若直接初始化 TPU，
# 进程会阻塞等另外 3 台会合 —— 4 个 pod 并行跑 bootstrap 时很容易互相卡死。
# 这里显式降回单机视图，只验本机 8 个 device，也正好是路线 A 要用的 env。
export TPU_WORKER_ID=0
export TPU_PROCESS_ADDRESSES=localhost:8471
export TPU_WORKER_HOSTNAMES=localhost
export TPU_HOST_BOUNDS=1,1,1
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1
export TPU_ACCELERATOR_TYPE=tpu7x-8
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
PY
[ $? -eq 0 ] || { echo "BOOTSTRAP_FAILED: 验证不过"; exit 1; }
ts BOOT_DONE
echo "BOOTSTRAP_OK"
