# A4X GB200 NVL72 on GKE (Google Kubernetes Engine)

GKE 原生方式部署 A4X GPU 集群。复测确认可复现。

**实测结果**：
- Qwen3 30B (8 GPU): **925 TFLOP/s/GPU**（官方 DGX-GB200: 936, 差 1.2%）
- Qwen3 235B (64 GPU): **686 TFLOP/s/GPU**（PP=2 EP=32 + MNNVL=2，比默认 V1 +89%）

**参考文档**：[Create a custom AI-optimized GKE cluster which uses A4X](https://docs.cloud.google.com/ai-hypercomputer/docs/create/gke-ai-hypercompute-custom-a4x)

## 前提条件

- GCP 项目，已启用 GKE API + Compute API
- A4X Reservation（`specificReservationRequired: true`，`shareType: LOCAL`）
- 足够的 quota（GPU、CPU、IP）

> **Reservation 注意**：`shareType: LOCAL` 可被同项目 GKE 直接消费。`shareType: SPECIFIC_PROJECTS` 可能需要额外权限配置（实测 us-central1-b 7 轮全部 GCE_STOCKOUT）。

## 部署步骤

### Step 1: VPC 网络

A4X 每节点 6 个网卡，需要 3 种 VPC。RDMA VPC 必须绑 `vpc-roce` network profile（zone 级别，不能跨 region 共用）。

```bash
PROJECT=supercomputer-testing
REGION=europe-west4
ZONE=europe-west4-b

# 主管理 VPC（可跨 region 复用）
gcloud compute networks create chrisya-gke-mgmt \
  --subnet-mode=custom --mtu=8244 --project=$PROJECT

gcloud compute networks subnets create chrisya-gke-sub-${REGION} \
  --network=chrisya-gke-mgmt --region=$REGION \
  --range=10.51.0.0/16 \
  --secondary-range=pods=10.64.0.0/14,services=10.68.0.0/20 \
  --project=$PROJECT

# 额外 GVNIC（可跨 region 复用）
gcloud compute networks create chrisya-gke-net-1 \
  --subnet-mode=custom --mtu=8244 --project=$PROJECT

gcloud compute networks subnets create chrisya-gke-sub-1-${REGION} \
  --network=chrisya-gke-net-1 --region=$REGION \
  --range=10.61.0.0/18 --project=$PROJECT

# RDMA VPC（每 region 独立，绑 vpc-roce profile）
gcloud compute networks create chrisya-gke-rdma-${REGION} \
  --subnet-mode=custom --mtu=8896 \
  --network-profile=projects/$PROJECT/global/networkProfiles/${ZONE}-vpc-roce \
  --project=$PROJECT

for i in 0 1 2 3; do
  BASE=$((192 + i * 16))
  gcloud compute networks subnets create chrisya-gke-rdma-${REGION}-sub-${i} \
    --network=chrisya-gke-rdma-${REGION} --region=$REGION \
    --range=192.168.${BASE}.0/20 --project=$PROJECT
done

# 防火墙
gcloud compute firewall-rules create chrisya-gke-allow-internal \
  --network=chrisya-gke-mgmt --allow=tcp,udp,icmp \
  --source-ranges=10.50.0.0/8 --project=$PROJECT

gcloud compute firewall-rules create chrisya-gke-allow-ssh \
  --network=chrisya-gke-mgmt --allow=tcp:22 \
  --source-ranges=35.235.240.0/20 --project=$PROJECT

for NET in chrisya-gke-net-1 chrisya-gke-rdma-${REGION}; do
  gcloud compute firewall-rules create ${NET}-allow-internal \
    --network=$NET --allow=tcp,udp,icmp \
    --source-ranges=0.0.0.0/0 --project=$PROJECT
done
```

### Step 2: Placement Policy

```bash
gcloud compute resource-policies create workload-policy chrisya-a4x-placement-${REGION} \
  --type=HIGH_THROUGHPUT \
  --accelerator-topology=1x72 \
  --region=$REGION --project=$PROJECT
```

> 必须用 `workload-policy` 类型。`group-placement` 即使加 `--gpu-topology=1x72` 也会报 GCE_STOCKOUT。

### Step 3: 创建 GKE 集群

```bash
gcloud container clusters create chrisya-a4x-gke-${REGION} \
  --region=$REGION \
  --network=chrisya-gke-mgmt \
  --subnetwork=chrisya-gke-sub-${REGION} \
  --cluster-secondary-range-name=pods \
  --services-secondary-range-name=services \
  --release-channel=rapid \
  --enable-ip-alias \
  --enable-multi-networking \
  --enable-dataplane-v2 \
  --enable-private-nodes \
  --no-enable-private-endpoint \
  --master-authorized-networks=0.0.0.0/0 \
  --workload-pool=${PROJECT}.svc.id.goog \
  --addons=GcsFuseCsiDriver \
  --num-nodes=1 \
  --machine-type=e2-standard-16 \
  --project=$PROJECT
```

> `--enable-multi-networking` 和 `--enable-dataplane-v2` 创建后不可更改。`--workload-pool` 是 GCSFuse CSI 前提。

### Step 4: Cloud NAT（private nodes 访问外网拉镜像）

```bash
gcloud compute routers create chrisya-gke-router \
  --network=chrisya-gke-mgmt --region=$REGION --project=$PROJECT

gcloud compute routers nats create chrisya-gke-nat \
  --router=chrisya-gke-router --region=$REGION \
  --auto-allocate-nat-external-ips \
  --nat-all-subnet-ip-ranges --project=$PROJECT
```

### Step 5: 创建 A4X Node Pool

```bash
RESERVATION=nvidia-gb200-jwmrpsfbs8szi  # 替换为实际 reservation 名

gcloud container node-pools create a4x-pool \
  --cluster=chrisya-a4x-gke-${REGION} \
  --region=$REGION \
  --node-locations=$ZONE \
  --machine-type=a4x-highgpu-4g \
  --accelerator=type=nvidia-gb200,count=4,gpu-driver-version=LATEST \
  --num-nodes=2 \
  --reservation-affinity=specific \
  --reservation=projects/$PROJECT/reservations/$RESERVATION \
  --placement-policy=chrisya-a4x-placement-${REGION} \
  --additional-node-network=network=chrisya-gke-net-1,subnetwork=chrisya-gke-sub-1-${REGION} \
  --additional-node-network=network=chrisya-gke-rdma-${REGION},subnetwork=chrisya-gke-rdma-${REGION}-sub-0 \
  --additional-node-network=network=chrisya-gke-rdma-${REGION},subnetwork=chrisya-gke-rdma-${REGION}-sub-1 \
  --additional-node-network=network=chrisya-gke-rdma-${REGION},subnetwork=chrisya-gke-rdma-${REGION}-sub-2 \
  --additional-node-network=network=chrisya-gke-rdma-${REGION},subnetwork=chrisya-gke-rdma-${REGION}-sub-3 \
  --ephemeral-storage-local-ssd=count=4 \
  --enable-gvnic \
  --project=$PROJECT
```

### Step 6: 安装 GPU Stack 组件

Node pool 创建后，安装 3 个组件启用 MNNVL 和 RDMA：

```bash
# 6.1 NCCL RDMA DaemonSet
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/gpudirect-rdma/nccl-rdma-installer-a4x.yaml

# 6.2 GKE Network 对象（GVNIC + RDMA 模式）
cat <<EOF | kubectl apply -f -
apiVersion: networking.gke.io/v1
kind: GKENetworkParamSet
metadata:
  name: gvnic-1
spec:
  vpc: chrisya-gke-net-1
  vpcSubnet: chrisya-gke-sub-1-${REGION}
  deviceMode: NetDevice
---
apiVersion: networking.gke.io/v1
kind: Network
metadata:
  name: gvnic-1
spec:
  type: Device
  parametersRef:
    group: networking.gke.io
    kind: GKENetworkParamSet
    name: gvnic-1
EOF

for i in 0 1 2 3; do
cat <<EOF | kubectl apply -f -
apiVersion: networking.gke.io/v1
kind: GKENetworkParamSet
metadata:
  name: rdma-${i}
spec:
  vpc: chrisya-gke-rdma-${REGION}
  vpcSubnet: chrisya-gke-rdma-${REGION}-sub-${i}
  deviceMode: RDMA
---
apiVersion: networking.gke.io/v1
kind: Network
metadata:
  name: rdma-${i}
spec:
  type: Device
  parametersRef:
    group: networking.gke.io
    kind: GKENetworkParamSet
    name: rdma-${i}
EOF
done

# 6.3 NVIDIA DRA Driver（ComputeDomain + IMEX）
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia && helm repo update
kubectl create ns nvidia-dra-driver-gpu

kubectl apply -n nvidia-dra-driver-gpu -f - <<EOF
apiVersion: v1
kind: ResourceQuota
metadata:
  name: nvidia-dra-driver-gpu-quota
spec:
  hard:
    pods: "37"
  scopeSelector:
    matchExpressions:
    - operator: In
      scopeName: PriorityClass
      values: [system-node-critical, system-cluster-critical]
EOF

helm install nvidia-dra-driver-gpu nvidia/nvidia-dra-driver-gpu \
  --version="25.12.0" \
  --namespace nvidia-dra-driver-gpu \
  -f - <<EOF
nvidiaDriverRoot: /home/kubernetes/bin/nvidia
resources:
  gpus:
    enabled: false
controller:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: nvidia.com/gpu
            operator: DoesNotExist
kubeletPlugin:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
          - matchExpressions:
              - key: cloud.google.com/gke-accelerator
                operator: In
                values: [nvidia-gb200]
              - key: kubernetes.io/arch
                operator: In
                values: [arm64]
  tolerations:
    - key: nvidia.com/gpu
      operator: Equal
      value: present
      effect: NoSchedule
    - key: kubernetes.io/arch
      operator: Equal
      value: arm64
      effect: NoSchedule
EOF
```

> **DRA Driver 版本**：K8s 1.36 DRA API 是 `resource.k8s.io/v1` (GA)。25.3.x 用 v1beta1 会报错，需要 **25.12.0+**。

### Step 7: 验证

```bash
# ComputeDomain CRD
kubectl api-resources | grep computedomain

# DRA pods
kubectl get pods -n nvidia-dra-driver-gpu

# NCCL RDMA DaemonSet
kubectl get ds -n kube-system nccl-rdma-installer

# GPU
kubectl exec <任意A4X-pod> -- nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

## 训练部署

训练 Pod 需要 ComputeDomain + 多网卡 annotation + hostPath 挂载 GIB/NVIDIA。完整 YAML 见 [nemo-gke-v3.yaml](yamls/nemo-gke-v3.yaml)。

### Qwen3 30B（8 GPU, 2 节点）

```bash
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_IP --master_port=29600 \
  run_script.py \
    -m qwen -mr qwen3_30b_a3b --task pretrain \
    -g gb200 -c fp8_mx -ng 8 --data mock \
    --max_steps 20 --log_dir /tmp/nemo-results \
    -wde bench -wdj qwen3_30b
```

环境变量：
```bash
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/gib/lib64:$LD_LIBRARY_PATH
source /usr/local/gib/scripts/set_nccl_env.sh
export NCCL_SOCKET_IFNAME=eth0,eth1
export NCCL_MNNVL_ENABLE=2
export NCCL_CUMEM_ENABLE=1
```

### Qwen3 235B（64 GPU, 16 节点, PP=2 EP=32 优化版）

```bash
torchrun --nproc_per_node=4 --nnodes=16 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_IP --master_port=29600 \
  run_script.py \
    -m qwen -mr qwen3_235b_a22b --task pretrain \
    -g gb200 -c fp8_mx -ng 64 --data mock \
    --max_steps 20 --log_dir /tmp/nemo-results \
    -wde bench -wdj qwen3_235b \
    -cv v1 \
    --pipeline_model_parallel_size 2 \
    --expert_model_parallel_size 32 \
    --global_batch_size 512 \
    --micro_batch_size 1
```

额外环境变量：`NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=32`

## 实测结果（复测确认）

### Qwen3 30B (8 GPU)

| 指标 | GKE | 自建 K8s | DGX-GB200 (官方) |
|---|---|---|---|
| TFLOP/s/GPU | **925** | 914 | 936 |
| Step Time | 6.52s | 6.60s | — |
| 差距 vs 官方 | -1.2% | -2.3% | baseline |

### Qwen3 235B (64 GPU)

| 指标 | V1 默认 (PP=8 EP=8) | PP=2 EP=32 (MNNVL=0) | PP=2 EP=32 (MNNVL=2) |
|---|---|---|---|
| TFLOP/s/GPU | 360 / 376 peak | 587 / 595 peak | **680 / 686 peak** |
| Step Time | ~27s | 8.2s | **7.1s** |
| NCCL transport | RDMA | RDMA | **NVLink** |
| 提升 | baseline | +63% | **+89%** |

> MNNVL=2 让 NCCL allreduce/PP p2p 走 NVLink (900 GB/s) 而非 RDMA (200 GB/s)，单域内不 hang。官方 V2 (256 GPU) 1092 TFLOP/s 需要 VPP=3 + full_iteration CG，64 GPU 上两者不可用。

## 235B 优化迭代全记录（64 GPU, Qwen3 235B-A22B MoE）

### 完整参数对比表

| 轮次 | PP | EP | TP | DP | MBS | GBS | CUDA Graph | MNNVL | recompute | TFLOP/s | step time | 状态 | 改了什么 → 为什么 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 8 | 8 | 1 | 8 | 1 | 1024 | TE scoped | 2 | 无 | **360** | 27s | ✅ | V1 默认 recipe |
| R1c | 8 | 8 | 1 | 8 | 1 | 1024 | **full_iteration** | 2 | 无 | crash | — | ❌ | 想开 full CG 加速 → HybridEP fabric memory 不兼容 PP>1 的 stream capture |
| R3 | 8 | 8 | 1 | 8 | 1 | 1024 | **full_iteration** | 2 | **48层** | crash | — | ❌ | 加 recompute 省内存给 CG → HBM 78→省出空间但 HybridEP 仍 capture 失败（不是内存问题） |
| R4 | 8 | 8 | 1 | 8 | 1 | 1024 | TE scoped | 2 | **24层** | crash | — | ❌ | 换回 TE CG + recompute → assert: recompute 只支持 full_iteration CG |
| R5 | 8 | 8 | 1 | 8 | 1 | 1024 | TE scoped | 2 | 无 | crash | — | ❌ | 加 NCCL_GRAPH_REGISTER=1 → 与 expandable_segments 冲突 assert |
| R7 | **1** | **64** | 1 | 64 | 4 | 512 | **full_iteration** | 2 | 无 | crash | — | ❌ | PP=1 让 full CG 兼容 HybridEP → DOCA QP 创建失败（EP=64 跨 16 节点） |
| R8 | **2** | **32** | 1 | 32 | 1 | 512 | **full_iteration** | 2 | 无 | crash (458 raw) | — | ❌ | PP=2 折衷：减 bubble + 试 full CG → PP>1 HybridEP 仍不兼容，但 raw 性能 458（比 PP=8 的 360 高 27%） |
| R9 | **2** | **32** | 1 | 32 | 1 | 512 | TE scoped | **0** | 无 | **595** | 8.2s | ✅ | 保持 PP=2 EP=32 + 回到 TE CG → bubble 从 30%→1.5%, EP group 32 GPU 通信效率高, **+63%** |
| **R10** | **2** | **32** | 1 | 32 | 1 | 512 | TE scoped | **2** | 无 | **686** | **7.1s** | ✅ | MNNVL=0→2: NCCL allreduce 从 RDMA 切到 NVLink → 单域 64 GPU 不 hang, **+15%** |

### 每轮的核心洞察

**R1c-R3（full_iteration CG 之路）**：HybridEP 的 CUDA fabric memory 操作在跨 pipeline stage 的 stream capture 时 invalidate。这不是内存问题（R3 recompute 把 HBM 从 130 降到 78 GiB 仍 crash），而是 HybridEP 架构与 multi-stream CUDA Graph 的根本不兼容。30B PP=1 能跑是因为单 stream。

**R4-R5（小优化碰壁）**：recompute 只支持 full_iteration CG（assert 硬限制）。NCCL_GRAPH_REGISTER=1 与 expandable_segments=True 冲突（assert 硬限制）。两条路都堵死。

**R7-R8（改并行策略）**：从"优化现有 PP=8 配置"转向"改变并行策略本身"。PP=1 EP=64 的 DOCA QP 失败暴露了 EP 跨多节点的 RDMA 限制。PP=2 EP=32 的 raw 性能 458 证明了方向正确。

**R9（突破）**：PP=2 EP=32 + TE CG = 595。两个改动叠加：pipeline stages 8→2 消除了 30% bubble，EP 8→32 让每卡只持 4 expert（vs 16），通信效率提高。

**R10（NVLink 加持）**：MNNVL=0 是照搬奚老师跨域 workaround 的错误。单域内 64 GPU 设 MNNVL=2 完全安全。NCCL allreduce 从 RDMA (200 GB/s) 切到 NVLink (900 GB/s)，+15%。

### 最终最优配置

```
PP=2  EP=32  TP=1  DP=32  MBS=1  GBS=512
CUDA Graph: TE scoped (attn + moe_router + moe_preprocess)
NCCL_MNNVL_ENABLE=2  USE_MNNVL=1  NCCL_NVLS_ENABLE=1
→ 686 TFLOP/s/GPU, step time 7.1s
→ 比 V1 默认 (360) 提升 89%
```

### 关键发现

1. **并行策略比 kernel 优化影响更大**：PP 8→2 一个改动就涨 63%，CUDA Graph 和 NCCL 调参加起来才涨 15%
2. **full_iteration CG 与 HybridEP 在 PP>1 时不兼容**：根因是 fabric memory 操作 invalidate multi-stream capture，不是内存问题
3. **NCCL_MNNVL_ENABLE 必须按实际拓扑设置**：单域设 2（NVLink 900GB/s），跨域才需要设 0（RDMA fallback）
4. **recompute 只支持 full_iteration CG**：TE scoped 下不可用，这是 Megatron Bridge 的硬限制
5. **NCCL_GRAPH_REGISTER=1 与 expandable_segments 互斥**：必须设 0

## GKE 搭建踩坑总结

| 问题 | 原因 | 修复 |
|---|---|---|
| GPU placement policy must be provided | --placement-type=COMPACT 不够 | 用 workload-policy 类型 |
| GCE_STOCKOUT (group-placement) | groupPlacementPolicy 无法匹配 reservation | 改用 workload-policy |
| Network can't host RDMA NIC | RDMA VPC 缺 vpc-roce profile | --network-profile=...vpc-roce |
| GCE_STOCKOUT (SPECIFIC_PROJECTS) | reservation shareType 限制 | 用 shareType: LOCAL 的 reservation |
| GCSFuse CSI requires Workload Identity | 缺 --workload-pool | 创建集群时加 |
| nvcr.io 镜像拉不下来 | private nodes 无外网 | 加 Cloud NAT |
| DRA Driver CRD not found | 25.3.x 用 v1beta1 vs K8s 1.36 v1 | 升级到 25.12.0 |
| GIB NCCL 2.28 vs NeMo 26.06 NCCL 2.30 | ncclWaitSignal symbol missing | 用容器自带 NCCL，GIB 只提供 transport |

## 清理

```bash
# 删除 node pool（保留集群和 VPC 供下次使用）
gcloud container node-pools delete a4x-pool \
  --cluster=chrisya-a4x-gke-${REGION} --region=$REGION \
  --project=$PROJECT --quiet

# 完全清理
gcloud container clusters delete chrisya-a4x-gke-${REGION} \
  --region=$REGION --project=$PROJECT --quiet
```
