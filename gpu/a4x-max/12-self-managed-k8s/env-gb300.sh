#!/bin/bash
# env-gb300.sh — GB300 自建 K8s 部署变量 (实测验证 2026-07-12)

# ========== GCP ==========
export PROJECT="tencent-gcp-taiji-poc"
export ZONE="us-central1-b"
export REGION="us-central1"

# ========== 机器 ==========
export MACHINE_TYPE="a4x-maxgpu-4g-metal"
export CP_MACHINE_TYPE="n2-standard-4"
export WORKER_IMAGE="rocky-linux-9-optimized-gcp-nvidia-580-arm64-v20260615"
export WORKER_IMAGE_PROJECT="rocky-linux-accelerator-cloud"
export CP_IMAGE="rocky-linux-9-v20260615"
export CP_IMAGE_PROJECT="rocky-linux-cloud"

# ========== 网络 ==========
export MGMT_VPC="chrisya-gb300-mgmt-v2"
export MGMT_SUB_0="chrisya-gb300-mgmt-sub-0"    # 10.150.0.0/24 (Master + Worker nic0)
export MGMT_SUB_1="chrisya-gb300-mgmt-sub-1"    # 10.150.1.0/24 (Worker nic1)
export RDMA_VPC="chrisya-gb300-rdma-v2"
export RDMA_SUB="default-subnet-1-chrisya-gb300-rdma-v2"

# ========== Reservation ==========
export RESERVATION_PROJECT="tencent-gcp-taiji"
export RESERVATION_NAME="nvidia-gb300-dxkhoz4ypk4mh"
export BLOCK="nvidia-gb300-dxkhoz4ypk4mh-block-0001"
# 用法: --reservation=.../$BLOCK/reservationSubBlocks/$BLOCK-subblock-NNNN
# 对应: --resource-policies=gb300-central-nvl72-policy-NNNN

# ========== K8s ==========
export K8S_VERSION="1.34"
export CALICO_VERSION="v3.29.3"
export CP_IP="10.150.0.2"

# ========== 组件版本 ==========
export GIB_VERSION="v1.1.2"
export DRA_GPU_DRIVER_VERSION="25.8.0"
export DRANET_VERSION="v1.3.0"

# ========== GB300 特有 ==========
export RDMA_NIC_COUNT=8
export GPU_COUNT=4
export HUGEPAGE_2M_COUNT=4096

# ========== NCCL (nccl.a4xmax.conf) ==========
export NCCL_NET="gIB"
export NCCL_IB_ADDR_FAMILY="AF_INET6"
export NCCL_IB_GID_INDEX=7  # RoCE v2 ipvlan GID (c0de suffix). Index 7 = RoCE v2, Index 6 = RoCE v1
export NCCL_IB_TC=52
export NCCL_IB_FIFO_TC=84
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_ADAPTIVE_ROUTING=1
export NCCL_IB_MERGE_VFS=1
export NCCL_CUMEM_ENABLE=1

echo "[env-gb300.sh] loaded: PROJECT=$PROJECT MACHINE=$MACHINE_TYPE" >&2
