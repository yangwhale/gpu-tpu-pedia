#!/bin/bash
# DeepEP 运行时环境配置
# 使用方法: source /path/to/unified-env.sh

export NVSHMEM_HOME=/opt/deepep/nvshmem
export CUDA_HOME=/usr/local/cuda

export LD_LIBRARY_PATH=${NVSHMEM_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
export LD_PRELOAD="${NVSHMEM_HOME}/lib/libnvshmem_host.so.3"

# NVSHMEM IBGDA 配置
export NVSHMEM_REMOTE_TRANSPORT=ibgda
export NVSHMEM_IB_ENABLE_IBGDA=1
export NVSHMEM_IBGDA_NIC_HANDLER=gpu      # GPU Handler 模式，无需 GDRCopy 内核模块
export NVSHMEM_HCA_PREFIX=mlx5
export NVSHMEM_IB_GID_INDEX=3
export NVSHMEM_DISABLE_CUDA_VMM=1
export NVSHMEM_ENABLE_NIC_PE_MAPPING=1

# PR #466: GPU 到 NIC 显式映射 (8 GPU : 8 NIC)
export DEEP_EP_DEVICE_TO_HCA_MAPPING=0:mlx5_0:1,1:mlx5_1:1,2:mlx5_2:1,3:mlx5_3:1,4:mlx5_4:1,5:mlx5_5:1,6:mlx5_6:1,7:mlx5_7:1

export PYTHONPATH=/opt/deepep/DeepEP:${PYTHONPATH:-}

echo "DeepEP environment loaded"
echo "  NVSHMEM_HOME: $NVSHMEM_HOME"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  NVSHMEM_IBGDA_NIC_HANDLER: $NVSHMEM_IBGDA_NIC_HANDLER"
