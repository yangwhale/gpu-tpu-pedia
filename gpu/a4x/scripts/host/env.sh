#!/bin/bash
# =============================================================================
# env.sh — forrest GB200 k8s 1.34 部署 ENV (集中变量)
# 用法: source env.sh (其他脚本统一 source 这个)
# ⚠️ TS_AUTHKEY 不放这里 (从 VM metadata 注入)
# =============================================================================

# ========== GCP 项目与区域 ==========
export PROJECT="gpu-launchpad-playground"
export ZONE="us-east1-d"
export REGION="us-east1"
export RESERVATION="nvidia-gb200-z4pzosg110ik8"

# ========== 机器与镜像 ==========
export MACHINE_TYPE="a4x-highgpu-4g"
export IMAGE="tlinux-server-4-gb200-v2"
export IMAGE_PROJECT="gpu-launchpad-playground"
export BOOT_DISK_SIZE="200GB"
export BOOT_DISK_TYPE="hyperdisk-balanced"

# ========== 网络 ==========
export GVNIC_NET="forrest-gvnic-net-0"          # 主 GVNIC (bond0) — VPC / kubelet --node-ip / kubeadm join
export GVNIC_SUB="forrest-gvnic-sub-0"
export VPC_CIDR="10.10.0.0/16"                  # forrest-gvnic-net-0 CIDR (Calico/cluster 通信走此)
export GVNIC_NET_1="forrest-gvnic-net-1"        # sec GVNIC (bond1)
export GVNIC_SUB_1="forrest-gvnic-sub-1"
export RDMA_NET="forrest-rdma-net-us-east1-d"   # RDMA (bond2-5)
export RDMA_SUB_0="forrest-rdma-sub-us-east1-d-0"
export RDMA_SUB_1="forrest-rdma-sub-us-east1-d-1"
export RDMA_SUB_2="forrest-rdma-sub-us-east1-d-2"
export RDMA_SUB_3="forrest-rdma-sub-us-east1-d-3"

# 网卡 OS 命名 (bond rename via udev rule, IMEX reboot 一并生效)
#   bond0 = 主 GVNIC (VPC / kubelet --node-ip / kubeadm join / 节点 SSH)
#   bond1 = sec GVNIC (no-address, 备用)
#   bond2-bond5 = 4 × MRDMA (NCCL inter-node, DRANET 注入 pod ns)

# ========== 集群命名 ==========
export CLUSTER_NAME="kubernetes"                # kubeadm 默认 clusterName (不改)
export CP_NAME="forrest-k8s-cp-tlinux"          # master VM hostname (us-east1-d, Rocky 9, 同 VPC)
export CP_VPC_IP="10.10.0.18"                   # master VPC IP (apiserver 绑此, cluster 通信走 VPC)
export API_SERVER="${CP_VPC_IP}:6443"
export DNS_DOMAIN="cluster.local"
# 注: master tailnet IP (100.76.123.19) 只在 forrest POC SSH 跳板用, 不用于 cluster 数据面

# Worker 列表 (同域 A vs 跨域 B)
export SD_WORKERS=("forrest-gb200-01" "forrest-gb200-02")    # 同域 (placement policy A)
export CD_WORKERS=("forrest-gb200-03" "forrest-gb200-04")    # 跨域 (placement policy B)
export PLACEMENT_POLICY_SD="forrest-a4x-1x72-policy"         # 域 A
export PLACEMENT_POLICY_CD="a4x-nvl72-policy"                # 域 B

# ========== K8s ==========
export K8S_VERSION="1.34.1"
export POD_CIDR="10.244.0.0/16"
export SERVICE_CIDR="10.96.0.0/12"

# ========== 组件版本 ==========
export GIB_VERSION="v1.1.2"
export DEVICE_PLUGIN_VERSION="0.19.2"
export GFD_VERSION="0.19.2"                     # GPU Feature Discovery (打 nvidia.com/gpu.clique 等官方 label)
export DRANET_VERSION="v1.3.0"
export DRA_GPU_DRIVER_VERSION="0.4.0"           # NVIDIA DRA GPU Driver (ComputeDomain)
export CALICO_VERSION="v3.32.0"                 # master 实际装的 (via tigera-operator)
export PYTORCH_TAG="26.05-py3"
export TRANSFORMER_ENGINE_VERSION="v2.15"
export MEGATRON_LM_TAG="core_r0.16.0"

# 镜像
export GIB_IMAGE="us-docker.pkg.dev/gce-ai-infra/gpudirect-gib/nccl-plugin-gib-diagnostic-arm64:${GIB_VERSION}"
export PYTORCH_IMAGE="nvcr.io/nvidia/pytorch:${PYTORCH_TAG}"

# Megatron 自 build 镜像 (默认 NGC base, Dockerfile.ngc-megatron-k8s134)
# 选 NGC pytorch:26.05-py3 因已预装 TE 2.15 + CUDA 13.2 + flash-attn + modelopt, 省 TE 编译 ~40min
# 如客户合规要求 TencentOS base, 切回 Dockerfile.tencentos4-megatron-k8s134 + tag megatron-tencentos4:...
export MEGATRON_DOCKERFILE="Dockerfile.ngc-megatron-k8s134"
export MEGATRON_BASE_IMAGE="nvcr.io/nvidia/pytorch:${PYTORCH_TAG}"
export MEGATRON_AR_REPO="us-east1-docker.pkg.dev/${PROJECT}/forrest-repo-us-east1"
export MEGATRON_IMAGE_TAG="megatron-ngc:te${TRANSFORMER_ENGINE_VERSION}-mg${MEGATRON_LM_TAG}-pt${PYTORCH_TAG}-v2"
export MEGATRON_IMAGE="${MEGATRON_AR_REPO}/${MEGATRON_IMAGE_TAG}"

# ========== Lustre ==========
export LUSTRE_INSTANCE="forrest-lustre"
export LUSTRE_LOCATION="${ZONE}"
export LUSTRE_PROJECT="${PROJECT}"
export LUSTRE_IP="10.158.224.3"
export LUSTRE_FS="data"                         # mountPoint 10.158.224.3@tcp:/data 里的 fs name
export LUSTRE_MOUNT="/data"
export LUSTRE_HANDLE="${LUSTRE_PROJECT}/${LUSTRE_LOCATION}/${LUSTRE_INSTANCE}"
export LUSTRE_CAPACITY_GI="72000"               # PV storage 字段 (TODO: 按 instance 实际容量调)

# ========== Tailscale (仅 forrest POC SSH 跳板, 客户环境不用) ==========
# 用途: master 在节点 init 阶段 ssh 进新节点跑 kubeadm join (节点未 join 前没别的路径)
# 客户环境用 ansible 推送, 不传此 metadata, startup script Phase A 自动跳过
# forrest 部署时:
#   TS_AUTHKEY=$(ssh maxwellx@${CP_NAME} "grep -oP 'authkey=\"\\K[^\"]+' /home/maxwellx/tailscale_up.sh")
#   gcloud compute instances create ... --metadata=...,tailscale-authkey=${TS_AUTHKEY}

# ========== Helm repo 元信息 ==========
export DRA_GPU_DRIVER_HELM="oci://registry.k8s.io/dra-driver-nvidia/charts/dra-driver-nvidia-gpu"
export DRANET_HELM="oci://registry.k8s.io/networking/charts/dranet"
export NVDP_HELM_REPO="https://nvidia.github.io/k8s-device-plugin"

# ========== Stamp 文件 ==========
export STAMP_DIR="/etc"
export STAMP_DONE="${STAMP_DIR}/.startup-done"
export STAMP_IMEX="${STAMP_DIR}/.imex-configured"
export STAMP_NIC_RENAME="${STAMP_DIR}/.nic-renamed"

# ========== Log 路径 ==========
export STARTUP_LOG="/var/log/forrest-worker-init.log"

# ========== 验证场景 ==========
# 1. 单节点 4 GPU NVLink baseline
# 2. 同域 2 节点 MNNVL (~836 GB/s)
# 3. 跨域 2 节点 RDMA (~328 GB/s)
# 4. 跨域 4 节点 mixed (~625 GB/s)
# (不验 72/144 卡)

echo "[env.sh] loaded: PROJECT=${PROJECT} K8S=${K8S_VERSION} IMAGE=${IMAGE}" >&2
