#!/bin/bash
# SGLang Environment Setup Script
#
# Usage: source scripts/setup_env.sh
#
# This script sets up the necessary environment variables for running SGLang,
# including CUDA paths and LD_LIBRARY_PATH for NVIDIA pip packages.

# Set CUDA_HOME
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

# Collect all nvidia pip package lib paths
# These are installed by pip packages like nvidia-cudnn-cu12, nvidia-nccl-cu12, etc.
NVIDIA_LIB_PATHS=""

# System-wide python packages
for d in /usr/local/lib/python3.12/dist-packages/nvidia/*/lib; do
    [ -d "$d" ] && NVIDIA_LIB_PATHS="${d}:${NVIDIA_LIB_PATHS}"
done

# User python packages
for d in $HOME/.local/lib/python3.12/site-packages/nvidia/*/lib; do
    [ -d "$d" ] && NVIDIA_LIB_PATHS="${d}:${NVIDIA_LIB_PATHS}"
done

# Also check python 3.10 and 3.11 paths for compatibility
for pyver in 3.10 3.11; do
    for d in /usr/local/lib/python${pyver}/dist-packages/nvidia/*/lib; do
        [ -d "$d" ] && NVIDIA_LIB_PATHS="${d}:${NVIDIA_LIB_PATHS}"
    done
    for d in $HOME/.local/lib/python${pyver}/site-packages/nvidia/*/lib; do
        [ -d "$d" ] && NVIDIA_LIB_PATHS="${d}:${NVIDIA_LIB_PATHS}"
    done
done

# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${NVIDIA_LIB_PATHS}${LD_LIBRARY_PATH}

# Optional: Set HuggingFace cache directory for faster model loading
if [ -d "/lssd/huggingface" ]; then
    export HF_HOME=/lssd/huggingface
    export HUGGINGFACE_HUB_CACHE=/lssd/huggingface/hub
fi

# Verify setup
echo "=========================================="
echo "SGLang Environment Loaded"
echo "=========================================="
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: configured with $(echo $NVIDIA_LIB_PATHS | tr ':' '\n' | wc -l) nvidia lib paths"

# Quick verification
if command -v python3 &> /dev/null; then
    python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "Warning: PyTorch verification failed"
fi

echo ""
echo "Ready to run SGLang. Example:"
echo "  python3 -m sglang.launch_server --model-path MODEL --tp TP_SIZE"
