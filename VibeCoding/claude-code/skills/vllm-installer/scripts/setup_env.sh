#!/bin/bash
# =============================================================================
# vLLM Environment Setup Script
#
# This script sets up the environment variables required for vLLM to run
# correctly, including LD_LIBRARY_PATH for NVIDIA libraries.
#
# Usage:
#   source setup_env.sh
# =============================================================================

# CUDA configuration
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$HOME/.local/bin:$PATH

# Collect NVIDIA pip package lib paths
NVIDIA_LIB_PATHS=""

# System-level pip packages
for d in /usr/local/lib/python3.*/dist-packages/nvidia/*/lib; do
    [ -d "$d" ] && NVIDIA_LIB_PATHS="${d}:${NVIDIA_LIB_PATHS}"
done

# User-level pip packages
for d in $HOME/.local/lib/python3.*/site-packages/nvidia/*/lib; do
    [ -d "$d" ] && NVIDIA_LIB_PATHS="${d}:${NVIDIA_LIB_PATHS}"
done

# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${NVIDIA_LIB_PATHS}${LD_LIBRARY_PATH}

# HuggingFace cache on LSSD (if available)
if [ -d /lssd/huggingface ]; then
    export HF_HOME=/lssd/huggingface
fi

# DeepEP paths (if installed)
if [ -d /opt/deepep/gdrcopy ]; then
    export GDRCOPY_HOME=/opt/deepep/gdrcopy
fi
if [ -d /opt/deepep/nvshmem ]; then
    export NVSHMEM_DIR=/opt/deepep/nvshmem
fi

# vLLM configuration
export VLLM_USAGE_SOURCE=production-script

# Print summary
echo "=============================================="
echo "vLLM Environment Loaded"
echo "=============================================="
echo "CUDA_HOME: $CUDA_HOME"
echo "HF_HOME: ${HF_HOME:-~/.cache/huggingface}"
echo "LD_LIBRARY_PATH configured with $(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -c nvidia) NVIDIA lib paths"

if [ -n "$GDRCOPY_HOME" ]; then
    echo "GDRCOPY_HOME: $GDRCOPY_HOME"
fi
if [ -n "$NVSHMEM_DIR" ]; then
    echo "NVSHMEM_DIR: $NVSHMEM_DIR"
fi

echo ""
echo "Start vLLM server:"
echo "  vllm serve Qwen/Qwen2.5-7B-Instruct \\"
echo "    --tensor-parallel-size 4 \\"
echo "    --port 8000"
echo ""
echo "Test server:"
echo "  curl http://localhost:8000/v1/models"
echo "=============================================="
