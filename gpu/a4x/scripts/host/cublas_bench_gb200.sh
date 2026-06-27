#!/bin/bash
# cublas_bench_gb200.sh
#
# 在 forrest GB200 worker host 上跑 NVIDIA cublasMatmulBench (6 dtype FP4/FP8/FP16/BF16/TF32/FP32) — 单 GPU GEMM peak。
#
# Host 只有 driver R580 (CUDA 13.0 runtime),没装 CUDA toolkit (没 libcublas)。
# 此脚本自动从已 pull 的 megatron-ngc image 内 ctr mount 提取 libcublas + libcublasLt + libcudart 到 /tmp/cublas_bench/,
# 然后 LD_LIBRARY_PATH 跑 binary。**不起 k8s pod, host 上直接跑**。
#
# 用法 (在控制平面/本机执行,自动 ssh 进 worker):
#   gx k8n "ssh -i ~/.ssh/google_compute_engine maxwellx@forrest-gb200-XX 'bash -s'" < scripts/host/cublas_bench_gb200.sh
# 或直接在 worker 上跑 (已 ssh 进 worker):
#   bash cublas_bench_gb200.sh
#
# 结果 echo 到 stdout — 推荐 tee 到 docs/k8s134/cublas-bench-results/<hostname>.log 保存。
#
# 输出例 (每 dtype):
#   FP4：GB200=6507              # NVIDIA 参考目标
#   ^^^^ gpu time statistics: runs 1000, mean 0.381498 ms, ...
#   ^^^^ CUDA : elapsed = 0.381498 sec,  Gflops = 6844957.711    # 实测 (1 Gflops = 1e-3 TFLOPs)
#
# 上游 binary + script:
#   https://github.com/compute-dev/ai_infra_perf_prepare/tree/main/cublas_bench

set -euo pipefail

WORK=/tmp/cublas_bench
IMAGE=us-east1-docker.pkg.dev/gpu-launchpad-playground/forrest-repo-us-east1/megatron-ngc:tev2.15-mgcore_r0.16.0-pt26.05-py3-v2
UP_REPO=https://raw.githubusercontent.com/compute-dev/ai_infra_perf_prepare/main/cublas_bench
MOUNT=/mnt/cublas-img

mkdir -p "$WORK"
cd "$WORK"

# 1. download binary + driver script (idempotent)
if [ ! -x cublasMatmulBench_gb2_3 ]; then
  echo "[$(date +%H:%M:%S)] download cublasMatmulBench_gb2_3 + cublas_bench_gb2_3.sh"
  curl -sLfo cublasMatmulBench_gb2_3 "$UP_REPO/cublasMatmulBench_gb2_3"
  curl -sLfo cublas_bench_gb2_3.sh   "$UP_REPO/cublas_bench_gb2_3.sh"
  chmod +x cublasMatmulBench_gb2_3 cublas_bench_gb2_3.sh
fi

# 2. extract libcublas + libcublasLt + libcudart from already-pulled container image
if [ ! -f libcublas.so.13 ]; then
  echo "[$(date +%H:%M:%S)] mount $IMAGE → $MOUNT, copy CUDA libs"
  sudo mkdir -p "$MOUNT"
  sudo ctr -n k8s.io images mount "$IMAGE" "$MOUNT" >/dev/null
  sudo cp -a "$MOUNT"/usr/local/cuda-13.2/targets/sbsa-linux/lib/libcublas* "$WORK/"
  sudo cp -a "$MOUNT"/usr/local/cuda-13.2/targets/sbsa-linux/lib/libcudart* "$WORK/"
  sudo chown -R "$(id -u):$(id -g)" "$WORK"/libcublas* "$WORK"/libcudart*
  sudo ctr -n k8s.io images unmount "$MOUNT" >/dev/null
  sudo rmdir "$MOUNT" 2>/dev/null || true
fi

# 3. run bench
echo "=== cublas_bench on $(hostname) at $(date) ==="
echo "GPU: $(nvidia-smi -L 2>&1 | head -1)"
echo "Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
echo
LD_LIBRARY_PATH="$WORK" bash cublas_bench_gb2_3.sh
echo
echo "=== done at $(date) ==="
