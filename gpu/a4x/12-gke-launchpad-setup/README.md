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

### 6.2 NVIDIA DRA GPU Driver (ComputeDomain)

```bash
# 1. Label GPU nodes (GKE COS 没有 NFD labels)
kubectl get nodes -l cloud.google.com/gke-accelerator=nvidia-gb200 --no-headers | \
  awk '{print $1}' | xargs -I{} kubectl label node {} feature.node.kubernetes.io/pci-10de.present=true

# 2. Install via Helm
helm upgrade --install nvidia-dra-driver-gpu \
  oci://registry.k8s.io/dra-driver-nvidia/charts/dra-driver-nvidia-gpu \
  --version 0.4.0 \
  --namespace nvidia-dra-driver-gpu --create-namespace \
  --set nameOverride=nvidia-dra-driver-gpu \
  --set nvidiaDriverRoot=/home/kubernetes/bin/nvidia \
  --set controller.affinity=null \
  --set controller.priorityClassName='' \
  --set kubeletPlugin.priorityClassName='' \
  --set gpuResourcesEnabledOverride=true \
  --set 'kubeletPlugin.tolerations[0].key=nvidia.com/gpu' \
  --set 'kubeletPlugin.tolerations[0].operator=Exists' \
  --set 'kubeletPlugin.tolerations[0].effect=NoSchedule' \
  --set 'kubeletPlugin.tolerations[1].key=kubernetes.io/arch' \
  --set 'kubeletPlugin.tolerations[1].operator=Equal' \
  --set 'kubeletPlugin.tolerations[1].value=arm64' \
  --set 'kubeletPlugin.tolerations[1].effect=NoSchedule'

# 3. Install ComputeDomain CRD
kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/dra-driver-nvidia-gpu/v0.4.0/deployments/helm/dra-driver-nvidia-gpu/crds/resource.nvidia.com_computedomains.yaml
```

**GKE COS 踩坑**: `nvidiaDriverRoot` 必须设为 `/home/kubernetes/bin/nvidia`（不是 `/`）。GKE COS 的 GPU driver 由 device-plugin 安装到这个路径。

### 6.3 Network Objects (RDMA)

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

## Step 7: 验证

```bash
# GPU nodes
kubectl get nodes -l cloud.google.com/gke-accelerator=nvidia-gb200

# DRA driver
kubectl get pods -n nvidia-dra-driver-gpu  # 1 controller + N kubelet-plugin

# ComputeDomain
kubectl api-resources | grep computedomain

# DeviceClasses
kubectl get deviceclasses  # 应有 compute-domain-*.nvidia.com + mrdma.google.com

# LWS
kubectl get pods -n lws-system
```

## 踩坑记录

| 问题 | 原因 | 修复 |
|---|---|---|
| `ZONE_RESOURCE_POOL_EXHAUSTED` 建 VM | 新 placement policy 无法分配到已被占的 subblock | 用预留已绑定的 placement policy |
| `ZONE_RESOURCE_POOL_EXHAUSTED` 不带 policy | 预留要求 4 local SSD，创建命令没指定 | 加 `--ephemeral-storage-local-ssd=count=4` |
| LWS image pull 失败 | 私有集群无 Cloud NAT | 创建 Cloud Router + NAT |
| DRA kubelet-plugin 0/0 desired | GPU 节点缺 NFD label | 手动 label `feature.node.kubernetes.io/pci-10de.present=true` |
| DRA init "nvidia-smi not found" | `nvidiaDriverRoot=/` 在 GKE COS 上不对 | 改为 `/home/kubernetes/bin/nvidia` |
| kubectl 连不上私有集群 | Master authorized networks 未配置 | 加 CC-TW 和 gLinux 的 IP |
| Lustre CSI 未启用 | 集群创建时忘了加 addon | `--addons=LustreCsiDriver` 或后续 `cluster update` |
