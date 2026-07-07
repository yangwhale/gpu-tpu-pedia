# gpu-launchpad-playground GKE A4X 集群搭建指南

> 对标 baker (supercomputer-testing) 的 GKE 集群，在 gpu-launchpad-playground 项目的 us-east1-d 建立。
>
> 集群名: `chrisya-a4x-gke-v2`，16 台 GB200 (64 GPU)，单域 NVL72。

## 前提条件

- 项目: `gpu-launchpad-playground`
- Zone: `us-east1-d`
- 预留: `nvidia-gb200-z4pzosg110ik8` (36 台 a4x-highgpu-4g, 2 subblock × 18)
- Placement Policy: `a4x-nvl72-policy` (domain 1) + `forrest-a4x-1x72-policy` (domain 2)
- VPC 已存在:
  - 主管理网络: `chrisya-gvnic-net-0` / `chrisya-gvnic-sub-0` (10.14.0.0/16) — 与 CC-TW 同 VPC 内网互联
  - 二级 gVNIC: `chrisya-gvnic-net-1` / `chrisya-gvnic-sub-1-ue1` (10.15.0.0/16)
  - RDMA: `chrisya-a4x-rdma-net` / 4 subnets (10.10.16-28.0/22)

## Step 1: 添加 GKE Secondary Ranges

GKE 需要 Pod 和 Service 的 secondary IP range:

```bash
gcloud compute networks subnets update chrisya-gvnic-sub-0 \
    --project=gpu-launchpad-playground --region=us-east1 \
    --add-secondary-ranges=gke-pods=10.28.0.0/14,gke-services=10.32.0.0/20
```

## Step 2: 创建 Cloud NAT

私有集群需要 NAT 让 pod 拉公共镜像:

```bash
gcloud compute routers create chrisya-gke-nat-router \
    --project=gpu-launchpad-playground --region=us-east1 \
    --network=chrisya-gvnic-net-0

gcloud compute routers nats create chrisya-gke-nat \
    --project=gpu-launchpad-playground --region=us-east1 \
    --router=chrisya-gke-nat-router \
    --auto-allocate-nat-external-ips \
    --nat-all-subnet-ip-ranges
```

## Step 3: 创建 GKE 集群

```bash
gcloud container clusters create chrisya-a4x-gke-v2 \
    --project=gpu-launchpad-playground \
    --zone=us-east1-d \
    --release-channel=rapid \
    --enable-ip-alias \
    --network=chrisya-gvnic-net-0 \
    --subnetwork=chrisya-gvnic-sub-0 \
    --cluster-secondary-range-name=gke-pods \
    --services-secondary-range-name=gke-services \
    --enable-dataplane-v2 \
    --enable-multi-networking \
    --enable-private-nodes \
    --master-ipv4-cidr=172.16.2.0/28 \
    --workload-pool=gpu-launchpad-playground.svc.id.goog \
    --num-nodes=1 \
    --machine-type=e2-standard-16 \
    --disk-type=pd-ssd --disk-size=200 \
    --addons=GcsFuseCsiDriver,LustreCsiDriver \
    --scopes=cloud-platform
```

关键参数说明:
- `--enable-multi-networking`: RDMA 多网卡必需
- `--enable-dataplane-v2`: 对标 baker 的 ADVANCED_DATAPATH
- `--enable-private-nodes`: 安全隔离
- `--addons=GcsFuseCsiDriver,LustreCsiDriver`: GCS 和 Lustre 存储

## Step 4: 配置 Master Authorized Networks

```bash
gcloud container clusters update chrisya-a4x-gke-v2 \
    --zone=us-east1-d --project=gpu-launchpad-playground \
    --enable-master-authorized-networks \
    --master-authorized-networks=<CC-TW-IP>/32,<GLINUX-IP>/32,10.14.0.0/16
```

## Step 5: 创建 A4X GPU Node Pool

每个 domain 一个 node pool，通过 placement policy 控制物理域:

```bash
# Domain 1 (a4x-nvl72-policy → subblock-0001)
gcloud container node-pools create a4x-domain-1 \
    --project=gpu-launchpad-playground \
    --cluster=chrisya-a4x-gke-v2 --zone=us-east1-d \
    --machine-type=a4x-highgpu-4g \
    --accelerator=type=nvidia-gb200,count=4,gpu-driver-version=latest \
    --num-nodes=8 \
    --disk-type=hyperdisk-balanced --disk-size=100 \
    --ephemeral-storage-local-ssd=count=4 \
    --reservation-affinity=specific --reservation=nvidia-gb200-z4pzosg110ik8 \
    --placement-type=COMPACT --placement-policy=a4x-nvl72-policy \
    --enable-gvnic \
    --additional-node-network=network=chrisya-gvnic-net-1,subnetwork=chrisya-gvnic-sub-1-ue1 \
    --additional-node-network=network=chrisya-a4x-rdma-net,subnetwork=chrisya-a4x-rdma-net-sub-0 \
    --additional-node-network=network=chrisya-a4x-rdma-net,subnetwork=chrisya-a4x-rdma-net-sub-1 \
    --additional-node-network=network=chrisya-a4x-rdma-net,subnetwork=chrisya-a4x-rdma-net-sub-2 \
    --additional-node-network=network=chrisya-a4x-rdma-net,subnetwork=chrisya-a4x-rdma-net-sub-3 \
    --scopes=cloud-platform

# Domain 2 (forrest-a4x-1x72-policy → subblock-0002)
# 同上，改 --placement-policy=forrest-a4x-1x72-policy
```

### Placement Policy 关键说明

**必须用预留已绑定的 placement policy**。每个 `1x72` COLLOCATED policy 会锁定到一个物理 NVL72 subblock。新建的 policy 如果两个 subblock 都被现有 policy 占了，会报 `ZONE_RESOURCE_POOL_EXHAUSTED`。

当前预留 subblock 绑定关系:

| Placement Policy | Subblock | 备注 |
|---|---|---|
| `a4x-nvl72-policy` | subblock-0001 | ivy 17 台 + tlinux 1 台 |
| `forrest-a4x-1x72-policy` | subblock-0002 | 我们 16 台 + tlinux 1 台 |

### `--ephemeral-storage-local-ssd=count=4`

**必须指定**。预留的 instance spec 要求 4 个 3TB local SSD。不加这个参数会报 `ZONE_RESOURCE_POOL_EXHAUSTED`（预留配置不匹配）。

## Step 6: 安装 GPU Stack 组件

GKE 集群创建后，以下组件自动安装:
- GPU device plugin (nvidia-gpu-device-plugin-large-cos)
- Lustre CSI
- GCS Fuse CSI
- Networking DRA driver (gke-managed-networking-dra-driver)

以下需要手动安装:

### 6.1 LeaderWorkerSet Controller

```bash
kubectl apply --server-side -f https://github.com/kubernetes-sigs/lws/releases/latest/download/manifests.yaml
```

### 6.2 NCCL RDMA Installer (GIB)

```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/gpudirect-rdma/nccl-rdma-installer-a4x.yaml
```

安装后每节点 `/home/kubernetes/bin/gib/` 目录包含 GIB NCCL plugin (`libnccl-net.so`)。

### 6.3 NVIDIA DRA GPU Driver (ComputeDomain)

> **关键**: 必须用 NVIDIA NGC Helm repo 的 **v25.12.0+**，不要用开源 registry.k8s.io 的 v0.4.0。v25.3.x 的 ComputeDomain daemon 无法正确初始化 IMEX（0/1 not ready），v25.12.0 修复了此问题。

```bash
# 1. ResourceQuota (DRA daemon 用 system-critical priority)
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
      values:
        - system-node-critical
        - system-cluster-critical
EOF

# 2. Install via NGC Helm repo
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia && helm repo update

cat > /tmp/dra-values.yaml <<EOF
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
          - key: "nvidia.com/gpu"
            operator: "DoesNotExist"
kubeletPlugin:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
          - matchExpressions:
              - key: cloud.google.com/gke-accelerator
                operator: In
                values:
                  - nvidia-gb200
              - key: kubernetes.io/arch
                operator: In
                values:
                  - arm64
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

helm install nvidia-dra-driver-gpu nvidia/nvidia-dra-driver-gpu \
    --version="25.12.0" \
    --namespace nvidia-dra-driver-gpu \
    -f /tmp/dra-values.yaml
```

安装后验证：`kubectl get pods -n nvidia-dra-driver-gpu` 应有 1 controller + N kubelet-plugin 全部 Running。

**GKE COS 要点**:
- `nvidiaDriverRoot` 必须为 `/home/kubernetes/bin/nvidia`（COS 的 driver 路径）
- 不需要手动打 NFD label（NGC chart 用 `cloud.google.com/gke-accelerator` node affinity）
- 不需要手动安装 ComputeDomain CRD（NGC chart 内含）

### 6.4 Network Objects (RDMA)

```yaml
# gvnic-1 + rdma-0~3 的 Network + GKENetworkParamSet
# RDMA 的 deviceMode 必须设为 RDMA
apiVersion: networking.gke.io/v1
kind: GKENetworkParamSet
metadata:
  name: rdma-0
spec:
  vpc: chrisya-a4x-rdma-net
  vpcSubnet: chrisya-a4x-rdma-net-sub-0
  deviceMode: RDMA
```

## Step 7: 创建 ComputeDomain 并标记节点

```bash
# 创建 ComputeDomain
kubectl apply -f - <<EOF
apiVersion: resource.nvidia.com/v1beta1
kind: ComputeDomain
metadata:
  name: my-compute-domain
spec:
  numNodes: 0
  channel:
    resourceClaimTemplate:
      name: my-compute-domain-channel
EOF

# 获取 UID 并标记 GPU 节点（触发 ComputeDomain daemon 部署）
CD_UID=$(kubectl get computedomain my-compute-domain -o jsonpath='{.metadata.uid}')
kubectl get nodes -l cloud.google.com/gke-accelerator=nvidia-gb200 --no-headers | \
  awk '{print $1}' | xargs -I{} kubectl label node {} "resource.nvidia.com/computeDomain=$CD_UID" --overwrite

# 验证 daemon 全部 1/1 Ready
kubectl get pods -n nvidia-dra-driver-gpu | grep computedomain
```

## Step 8: 验证

```bash
# GPU nodes
kubectl get nodes -l cloud.google.com/gke-accelerator=nvidia-gb200

# DRA driver (1 controller + N kubelet-plugin)
kubectl get pods -n nvidia-dra-driver-gpu

# ComputeDomain daemon (应全部 1/1 Ready)
kubectl get pods -n nvidia-dra-driver-gpu | grep computedomain

# NCCL RDMA installer (N/N Running)
kubectl get daemonsets -n kube-system | grep nccl-rdma

# DeviceClasses
kubectl get deviceclasses  # compute-domain-*.nvidia.com + mrdma.google.com

# LWS
kubectl get pods -n lws-system
```

## Step 9: 部署训练 Workload

Pod 需要以下配置（参考 GKE 官方文档）:

```yaml
metadata:
  annotations:
    networking.gke.io/default-interface: 'eth0'
    networking.gke.io/interfaces: |
      [
        {"interfaceName":"eth0","network":"default"},
        {"interfaceName":"eth2","network":"rdma-0"},
        {"interfaceName":"eth3","network":"rdma-1"},
        {"interfaceName":"eth4","network":"rdma-2"},
        {"interfaceName":"eth5","network":"rdma-3"}
      ]
spec:
  resourceClaims:
  - name: compute-domain-channel
    resourceClaimTemplateName: my-compute-domain-channel
  volumes:
  - {name: nvidia, hostPath: {path: /home/kubernetes/bin/nvidia}}
  - {name: gib, hostPath: {path: /home/kubernetes/bin/gib}}
  containers:
  - resources:
      claims: [{name: compute-domain-channel}]
      limits: {nvidia.com/gpu: "4"}
    volumeMounts:
    - {name: nvidia, mountPath: /usr/local/nvidia}
    - {name: gib, mountPath: /usr/local/gib}
    env:
    - {name: LD_LIBRARY_PATH, value: "/usr/local/nvidia/lib64"}
```

### DSv3 16L 训练实测 (2026-07-08)

单域 16 节点 64 GPU，NeMo Bridge `run_script.py -m deepseek -mr deepseek_v3 -c fp8_mx`：

| iter | step time | 备注 |
|---|---|---|
| 6 | 3207ms | warmup |
| 7-10 | ~2540ms | **稳态** |
| 11 | 3159ms | VPP spike (正常) |
| 12-15 | ~2540ms | **稳态** |
| 16 | 3165ms | VPP spike |
| 17-20 | ~2545ms | **稳态** |

稳态 ~2.54s/step，估算 **~1030 TFLOPs/GPU**。20 步全跑完，零错误。

**注意**: 单域 `NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN` 必须等于 EP 度（EP=32 → 设 32，不是 64）。

## 踩坑记录

| 问题 | 原因 | 修复 |
|---|---|---|
| `ZONE_RESOURCE_POOL_EXHAUSTED` 建 VM | 新 placement policy 无法分配到已被占的 subblock | 用预留已绑定的 placement policy |
| `ZONE_RESOURCE_POOL_EXHAUSTED` 不带 policy | 预留要求 4 local SSD，创建命令没指定 | 加 `--ephemeral-storage-local-ssd=count=4` |
| LWS image pull 失败 | 私有集群无 Cloud NAT | 创建 Cloud Router + NAT |
| kubectl 连不上私有集群 | Master authorized networks 未配置 | 加 CC-TW 和 gLinux 的 IP |
| Lustre CSI 未启用 | 集群创建时忘了加 addon | `--addons=LustreCsiDriver` 或后续 `cluster update` |
| DRA v25.3.x CD daemon 0/1 not ready | IMEX 初始化失败 + 409 Conflict race | **升级到 v25.12.0** |
| CUDA 801 `operation not supported` | ComputeDomain daemon 未就绪时训练启动 | 确保 CD daemon 全部 1/1 Ready 后再启动训练 |
| `ranks 32 not divisible by ranks_per_node 64` | 单域 EP=32 但 `EP_RANKS_PER_DOMAIN=64` | 改为 `NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=32` |
| DRA PreBind `nil request mappings` (v0.4.0) | 开源 DRA driver 与 GKE DRA scheduler 不兼容 | 换 NGC v25.12.0 |
| NeMo 镜像拉不到 | 跨项目 AR 无权限 | `crane copy` 到本项目 AR |
