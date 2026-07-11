# A4X Max GB300 NVL72 on GKE (Google Kubernetes Engine)

GKE 原生方式部署 A4X Max (GB300) GPU 集群。基于 GB200 (A4X) 实战经验 + GCP 官方文档整理。

**参考文档**：[Create a custom AI-optimized GKE cluster which uses A4X Max](https://docs.cloud.google.com/ai-hypercomputer/docs/create/gke-ai-hypercompute-custom-a4x-max)

## GB300 vs GB200 GKE 部署差异总结

| 维度 | GB200 (A4X) | GB300 (A4X Max) | 影响 |
|------|-------------|-----------------|------|
| 机器类型 | `a4x-highgpu-4g` | `a4x-maxgpu-4g-metal` (裸金属) | node pool 参数 |
| GPU 类型 | `nvidia-gb200` | `nvidia-gb300` | accelerator 参数 |
| GKE 最低版本 | 1.32.8+ | **1.34.3-gke.1318000+** (推荐 1.35.0-gke.2745000+) | cluster version |
| 网络模式 | `--enable-multi-networking` + 手动 VPC | **DRANET** (`--accelerator-network-profile=auto`) | **重大简化** |
| RDMA VPC | 手动创建 4 个子网 + `vpc-roce` profile | **自动创建** (auto profile, `vpc-roce-metal` Bimetal) | 无需手动配置 |
| 管理网卡 | GVNIC | IDPF (裸金属原生) | 驱动不同 |
| RDMA 接口数 | 4 (CX-7 VF) | **8** (CX-8 SuperNIC PF, GPUDirect) | ResourceClaim count |
| RDMA 配置 | `nccl-rdma-installer` DaemonSet | **`asapd-lite`** DaemonSet | 安装步骤不同 |
| DRA Driver 版本 | 25.12.0 | **25.8.0+** (推荐) | helm 参数 |
| Hugepages | 未配置 | **必须** `hugepage_size2m: 4096` | node pool 参数 |
| 网络栈 | IPv4 | **IPv6-only** | 子网/防火墙规则 |
| Shielded Nodes | 默认启用 | **必须关闭** (`--no-enable-shielded-nodes`) | cluster 参数 |
| Node 镜像 | COS 或 Ubuntu | **仅 COS** | 限制 |
| Reservation 格式 | `projects/P/reservations/R` | `projects/P/reservations/R/reservationBlocks/B/reservationSubBlocks/S` | 更精细 |

## 前提条件

- GCP 项目 `tencent-gcp-taiji-poc`，已启用 GKE API + Compute API
- Reservation `nvidia-gb300-dxkhoz4ypk4mh` 在 `tencent-gcp-taiji` 项目（跨项目消费）
- 12 个 Placement Policy `gb300-central-nvl72-policy-0001~0012` 在 POC 项目
- Zone: `us-central1-b`

## 环境变量

```bash
export PROJECT=tencent-gcp-taiji-poc
export RESERVATION_PROJECT=tencent-gcp-taiji
export REGION=us-central1
export ZONE=us-central1-b
export RESERVATION_NAME=nvidia-gb300-dxkhoz4ypk4mh
export BLOCK_NAME=nvidia-gb300-dxkhoz4ypk4mh-block-0001
export SUBBLOCK_NAME=nvidia-gb300-dxkhoz4ypk4mh-block-0001-subblock-0001  # 18/18 healthy, 0 inUse
export CLUSTER_NAME=gb300-gke-test
```

## 部署步骤

### Step 0: 查询 Reservation Block/Sub-Block 名称

GB300 reservation 使用分层结构。先查询实际的 block 和 sub-block 名称：

```bash
# 查看 reservation 详情（在 reservation 所属项目查询）
gcloud compute reservations describe $RESERVATION_NAME \
  --zone=$ZONE --project=$RESERVATION_PROJECT

# 查看所有 placement policy
gcloud compute resource-policies list \
  --filter="name~gb300" --project=$PROJECT
```

> **Reservation 路径格式**：`projects/$RESERVATION_PROJECT/reservations/$RESERVATION_NAME/reservationBlocks/$BLOCK/reservationSubBlocks/$SUBBLOCK`
>
> 每个 sub-block = 18 节点 = 72 GPU = 1 个 NVL72 domain。我们有 12 个 sub-block 对应 12 个 placement policy。

### Step 1: 管理 VPC 网络

GB300 的 RDMA 网络由 `--accelerator-network-profile=auto` 自动创建（使用 `vpc-roce-metal` profile, MTU 自动 8896），但 GKE 集群本身仍需要一个管理网络。

> **MTU 重要**：A4X Max 官方推荐管理 VPC MTU = **8896**。`default` VPC 的 MTU 是 1460，虽然 RDMA 不受影响（auto profile 单独创建），但管理网络流量（k8s 控制面、存储、Pod 间通信）性能不是最优。生产环境建议用自定义 VPC。

**方式 A：使用 default VPC（快速测试）**

如果只是快速验证，可以直接用 `default` VPC，跳到 Step 2。RDMA 网络由 auto profile 自动处理。缺点是管理 MTU=1460 不是最优。

**方式 B：创建自定义 VPC（推荐生产使用）**

```bash
# 创建管理 VPC（MTU 8896）
gcloud compute networks create gb300-gke-mgmt \
  --subnet-mode=custom --mtu=8896 --project=$PROJECT

# 创建管理子网（含 Pod/Service secondary ranges）
gcloud compute networks subnets create gb300-gke-sub-${REGION} \
  --network=gb300-gke-mgmt --region=$REGION \
  --range=10.51.0.0/16 \
  --secondary-range=pods=10.64.0.0/14,services=10.68.0.0/20 \
  --project=$PROJECT

# 防火墙：内部通信
gcloud compute firewall-rules create gb300-gke-allow-internal \
  --network=gb300-gke-mgmt --allow=tcp,udp,icmp \
  --source-ranges=10.0.0.0/8 --project=$PROJECT

# 防火墙：IAP SSH
gcloud compute firewall-rules create gb300-gke-allow-ssh \
  --network=gb300-gke-mgmt --allow=tcp:22 \
  --source-ranges=35.235.240.0/20 --project=$PROJECT
```

> **RDMA VPC 不需要手动创建**。GB200 需要 3 个 VPC（管理 + GVNIC + RDMA）+ 4 个 RDMA 子网。GB300 只需 1 个管理 VPC，RDMA 由 `--accelerator-network-profile=auto` 自动创建（`vpc-roce-metal` profile + 自动子网 `default-subnet-1-*`）。
>
> **已有 VPC 参考**：POC 项目里已有 `gb300-central-idpf-net`（MTU 8896，手建 VM 用的管理网络）和 `gb300-central-rdma-net`（MTU 8896，RoCE profile，手建 VM 用的 RDMA 网络）。GKE 的 auto profile 会自己创建 RDMA 网络，不需要复用这些。

### Step 2: 创建 GKE 集群

```bash
gcloud container clusters create $CLUSTER_NAME \
  --region=$REGION \
  --network=gb300-gke-mgmt \
  --subnetwork=gb300-gke-sub-${REGION} \
  --cluster-secondary-range-name=pods \
  --services-secondary-range-name=services \
  --cluster-version=1.35.0-gke.2745000 \
  --enable-dataplane-v2 \
  --enable-ip-alias \
  --enable-private-nodes \
  --no-enable-private-endpoint \
  --no-enable-shielded-nodes \
  --master-authorized-networks=0.0.0.0/0 \
  --workload-pool=${PROJECT}.svc.id.goog \
  --addons=GcsFuseCsiDriver \
  --num-nodes=1 \
  --machine-type=e2-standard-16 \
  --project=$PROJECT
```

> **关键差异 vs GB200**：
> - `--no-enable-shielded-nodes`（GB300 裸金属要求，官方必须）
> - **不用** `--enable-multi-networking`（GB300 不支持，用 DRANET 替代，官方明确）
> - GKE 版本 >= 1.34.3-gke.1318000（推荐 1.35.0+）
>
> **可选参数**（非官方必须，但推荐生产环境使用）：
> - `--enable-private-nodes` + `--no-enable-private-endpoint`：节点无公网 IP，需配 Cloud NAT（Step 3）
> - `--master-authorized-networks`：限制 API server 访问来源
> - `--workload-pool`：启用 Workload Identity
> - `--addons=GcsFuseCsiDriver`：GCS 挂载支持
> - `--network` / `--subnetwork`：如果 POC 项目没有 default VPC 则必须指定

```bash
# 获取集群凭证
gcloud container clusters get-credentials $CLUSTER_NAME \
  --region=$REGION --project=$PROJECT
```

### Step 3: Cloud NAT（private nodes 访问外网拉镜像）

```bash
gcloud compute routers create gb300-gke-router \
  --network=gb300-gke-mgmt --region=$REGION --project=$PROJECT

gcloud compute routers nats create gb300-gke-nat \
  --router=gb300-gke-router --region=$REGION \
  --auto-allocate-nat-external-ips \
  --nat-all-subnet-ip-ranges --project=$PROJECT
```

### Step 4: Workload Policy

```bash
gcloud beta compute resource-policies create workload-policy gb300-gke-workload-policy \
  --type=HIGH_THROUGHPUT \
  --accelerator-topology=1x72 \
  --region=$REGION --project=$PROJECT
```

> **重要**：我们已有 12 个 `gb300-central-nvl72-policy-XXXX`，这些是 `group-placement` 类型（用于手动创建 VM + `--resource-policies` 绑定）。GKE node pool 的 `--placement-policy` 需要 `workload-policy` 类型。两种类型**不兼容**，GKE 必须用 workload-policy。所以这里新建了 `gb300-gke-workload-policy`。
>
> 如果将来需要在同一个 sub-block 上既有自建 VM 又有 GKE node pool，两种 policy 可以共存（指向同一个物理 sub-block），但需要注意容量竞争。

### Step 5: 创建 Hugepages 配置文件

GB300 裸金属节点需要预分配 hugepages：

```bash
cat > /tmp/node_custom.yaml <<EOF
linuxConfig:
  hugepageConfig:
    hugepage_size2m: 4096
EOF
```

### Step 6: 创建 A4X Max Node Pool

```bash
gcloud container node-pools create gb300-pool \
  --cluster=$CLUSTER_NAME \
  --region=$REGION \
  --node-locations=$ZONE \
  --num-nodes=18 \
  --placement-policy=gb300-gke-workload-policy \
  --machine-type=a4x-maxgpu-4g-metal \
  --accelerator=type=nvidia-gb300,count=4,gpu-driver-version=latest \
  --system-config-from-file=/tmp/node_custom.yaml \
  --accelerator-network-profile=auto \
  --node-labels=cloud.google.com/gke-networking-dra-driver=true,cloud.google.com/gke-dpv2-unified-cni=cni-migration \
  --reservation-affinity=specific \
  --reservation=projects/$RESERVATION_PROJECT/reservations/$RESERVATION_NAME/reservationBlocks/$BLOCK_NAME/reservationSubBlocks/$SUBBLOCK_NAME \
  --project=$PROJECT
```

> **`--accelerator-network-profile=auto`**：这是 GB300 最大的简化。GKE 自动为 A4X Max 节点创建 RDMA VPC 网络和子网（使用 `vpc-roce-metal` Bimetal profile）。不需要手动创建 RDMA VPC、子网、或 `--additional-node-network` 参数。
>
> **Reservation 跨项目**：reservation 在 `tencent-gcp-taiji`，node pool 在 `tencent-gcp-taiji-poc`。路径必须用完整的 `projects/$RESERVATION_PROJECT/reservations/...` 格式。
>
> **节点数**：18 节点 = 1 个完整 NVL72 sub-block。如果需要更少的节点做测试，可以先设 `--num-nodes=2`。

### Step 7: 安装 asapd-lite（配置 MRDMA 网卡）

GB300 使用 `asapd-lite` 替代 GB200 的 `nccl-rdma-installer`：

```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/refs/heads/master/asapd-lite-installer/asapd-lite-installer-a4x-max-bm-cos.yaml
```

验证：

```bash
kubectl get daemonset -n kube-system asapd-lite
# READY 数应等于 A4X Max 节点数
```

### Step 8: 安装 NVIDIA DRA Driver（ComputeDomain + IMEX）

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia && helm repo update

kubectl create ns nvidia-dra-driver-gpu

# ResourceQuota: 至少 2 * 节点数 + 1
NODE_COUNT=18
POD_QUOTA=$((NODE_COUNT * 2 + 1))

kubectl apply -n nvidia-dra-driver-gpu -f - <<EOF
apiVersion: v1
kind: ResourceQuota
metadata:
  name: nvidia-dra-driver-gpu-quota
spec:
  hard:
    pods: "${POD_QUOTA}"
  scopeSelector:
    matchExpressions:
    - operator: In
      scopeName: PriorityClass
      values: [system-node-critical, system-cluster-critical]
EOF

helm install nvidia-dra-driver-gpu nvidia/nvidia-dra-driver-gpu \
  --set controller.args.v=4 --set kubeletPlugin.args.v=4 \
  --version="25.8.0" \
  --create-namespace \
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
                values: [nvidia-gb300]
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

> **vs GB200**：accelerator selector 从 `nvidia-gb200` 改为 `nvidia-gb300`。DRA 版本从 25.12.0 改为 25.8.0（官方推荐）。`--set controller.args.v=4 --set kubeletPlugin.args.v=4` 开启 debug 日志方便排查。

### Step 9: 验证

```bash
# asapd-lite DaemonSet
kubectl get ds -n kube-system asapd-lite

# DRA pods
kubectl get pods -n nvidia-dra-driver-gpu

# ComputeDomain CRD
kubectl api-resources | grep computedomain

# 节点标签
kubectl get nodes -l cloud.google.com/gke-accelerator=nvidia-gb300

# GPU 检查
kubectl exec <任意A4X-Max-pod> -- nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# 预期: NVIDIA B300, 288 GB
```

## Workload 部署模板

GB300 workload 需要 ComputeDomain + DRANET ResourceClaimTemplate + RDMA 设备请求。与 GB200 不同，不再使用 GKE Network annotation，而是用 DRA 标准。

### ComputeDomain + ResourceClaimTemplate

```yaml
apiVersion: resource.nvidia.com/v1beta1
kind: ComputeDomain
metadata:
  name: gb300-compute-domain
spec:
  numNodes: 18  # 1 个 NVL72 sub-block
  channel:
    resourceClaimTemplate:
      name: gb300-compute-domain-channel
---
apiVersion: resource.k8s.io/v1
kind: ResourceClaimTemplate
metadata:
  name: all-mrdma
spec:
  spec:
    devices:
      requests:
      - name: req-mrdma
        exactly:
          deviceClassName: mrdma.google.com
          allocationMode: ExactCount
          count: 8  # GB300: 8 个 MRDMA 接口 (vs GB200 的 4 个)
```

### Pod 模板

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gb300-training-pod
spec:
  nodeSelector:
    gke.networks.io/accelerator-network-profile: auto
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: kubernetes.io/arch
            operator: In
            values: [arm64]
  hostNetwork: true
  volumes:
    - name: library-dir-host
      hostPath:
        path: /home/kubernetes/bin/nvidia
  containers:
    - name: training
      image: <NeMo 镜像>
      volumeMounts:
        - name: library-dir-host
          mountPath: /usr/local/nvidia
      env:
        - name: LD_LIBRARY_PATH
          value: /usr/local/nvidia/lib64
      resources:
        limits:
          nvidia.com/gpu: 4
        claims:
          - name: compute-domain-channel
          - name: rdma
  resourceClaims:
    - name: compute-domain-channel
      resourceClaimTemplateName: gb300-compute-domain-channel
    - name: rdma
      resourceClaimTemplateName: all-mrdma
```

> **vs GB200 Pod 差异**：
> - `nodeSelector: gke.networks.io/accelerator-network-profile: auto`（auto 网络创建的节点标签）
> - `hostNetwork: true`（裸金属需要）
> - `resourceClaims` 替代 `k8s.v1.cni.cncf.io/networks` annotation
> - MRDMA count 8（vs GB200 的 4）

### 容器镜像要求：DOCA OFED + NCCL 升级

GB300 workload 容器镜像**必须**安装 DOCA OFED 用户空间库并升级 NCCL（官方要求）：

```bash
# 在 Dockerfile 或容器启动时执行
apt update -y && apt install -y curl
export DOCA_URL="https://linux.mellanox.com/public/repo/doca/3.1.0/ubuntu22.04/arm64-sbsa/"
curl https://linux.mellanox.com/public/repo/doca/GPG-KEY-Mellanox.pub | \
  gpg --dearmor > /etc/apt/trusted.gpg.d/GPG-KEY-Mellanox.pub
echo "deb [signed-by=/etc/apt/trusted.gpg.d/GPG-KEY-Mellanox.pub] $DOCA_URL ./" \
  > /etc/apt/sources.list.d/doca.list
apt update
apt -y install doca-ofed-userspace
# 升级 NCCL 到 2.28.9+（默认安装的是 2.27.7）
apt install --only-upgrade --allow-change-held-packages -y libnccl2 libnccl-dev
```

> **重要**：不装 `doca-ofed-userspace` 的话 CX-8 RDMA 无法正常工作。NeMo 26.06 容器已内置这些依赖，自定义镜像需要手动安装。

## 实测记录 (2026-07-11)

### 已验证

- 小爱同学建了 GKE 集群 `gb300-gke-test` (GKE 1.36.0, default VPC, dataplane v2, shielded-nodes 已关，创建时间 2026-07-11 21:55 HKT)
- 直接在该集群上加了 node pool（跳过 Step 1-3）
- **注意**：该集群用 `default` VPC (MTU 1460)，非最优。RDMA 不受影响（auto profile 单独创建），但管理网络 MTU 偏小
- workload-policy 创建成功：`gb300-gke-workload-policy` (HIGH_THROUGHPUT, 1x72)
- node pool `gb300-pool-1` 创建成功，1 节点 RUNNING
- 跨项目 reservation 消费成功，完整路径：
  ```
  projects/tencent-gcp-taiji/reservations/nvidia-gb300-dxkhoz4ypk4mh/reservationBlocks/nvidia-gb300-dxkhoz4ypk4mh-block-0001/reservationSubBlocks/nvidia-gb300-dxkhoz4ypk4mh-block-0001-subblock-0001
  ```
- `--accelerator-network-profile=auto` 正常工作

### 踩坑

1. **kubectl 认证失败**：gLinux 上 `gke-gcloud-auth-plugin` 版本 v0.1.0 太旧，跟 GKE 1.36 不兼容。报错 `the server has asked for the client to provide credentials`。解法：更新 auth plugin 或通过项目内 master 节点执行 kubectl。
2. **复用已有集群**：如果项目里已有 GKE 集群且满足版本要求（1.34.3+ / 1.35.0+），可以跳过 Step 1-3 直接从 Step 4 开始。确认集群 `--no-enable-shielded-nodes` 和 `ADVANCED_DATAPATH` 即可。
3. **CC-TW 无法直连**：CC-TW 的 SA `604327164091-compute@developer.gserviceaccount.com` 没有 `container.clusters.get` 权限，需要通过 gLinux 操作 GKE。

### 待验证（需要 kubectl 访问）

- [ ] asapd-lite DaemonSet 安装和 READY 状态
- [ ] DRA driver helm install
- [ ] ComputeDomain CRD
- [ ] GPU nvidia-smi 输出（确认 B300 + 288GB）
- [ ] RDMA rdma link show（确认 8 个 MRDMA 接口）

## GKE 搭建预期踩坑

| 问题 | 可能原因 | 解法 |
|------|---------|------|
| node pool 创建 GCE_STOCKOUT | reservation 路径错 / block name 不对 | 检查 Step 0 查到的 block/subblock 名 |
| node pool 创建权限不够 | 跨项目 reservation 未共享 | 检查 reservation shareType + 项目 IAM |
| asapd-lite DaemonSet 不 Ready | CX-8 驱动问题 | 检查节点日志 `kubectl logs -n kube-system <asapd-pod>` |
| DRA Driver pod CrashLoop | K8s DRA API 版本不匹配 | 确认 GKE 版本 >= 1.34.3，DRA 25.8.0 |
| nvcr.io 镜像拉不下来 | private nodes 无外网 | 确认 Cloud NAT 正常 |
| RDMA 不通 | asapd-lite 未配置好 | 检查 `rdma link show` + asapd logs |
| 子网 CIDR 冲突 | 管理子网 vs auto 创建的 RDMA 子网 | 管理子网用 10.x，RDMA auto 用 192.168.x |

## 清理

```bash
# 删除 node pool（保留集群和 VPC 供下次使用）
gcloud container node-pools delete gb300-pool \
  --cluster=$CLUSTER_NAME --region=$REGION \
  --project=$PROJECT --quiet

# 完全清理
gcloud container clusters delete $CLUSTER_NAME \
  --region=$REGION --project=$PROJECT --quiet

gcloud compute routers nats delete gb300-gke-nat \
  --router=gb300-gke-router --region=$REGION --project=$PROJECT --quiet
gcloud compute routers delete gb300-gke-router \
  --region=$REGION --project=$PROJECT --quiet

gcloud compute firewall-rules delete gb300-gke-allow-internal --project=$PROJECT --quiet
gcloud compute firewall-rules delete gb300-gke-allow-ssh --project=$PROJECT --quiet
gcloud compute networks subnets delete gb300-gke-sub-${REGION} \
  --region=$REGION --project=$PROJECT --quiet
gcloud compute networks delete gb300-gke-mgmt --project=$PROJECT --quiet
```
