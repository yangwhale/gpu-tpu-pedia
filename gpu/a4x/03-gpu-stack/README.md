# 3. 安装 GPU Stack + ComputeDomain + 4. 共享存储

**k8s 1.34 DRA 状态**：DRA 在 k8s 1.34 中为 **GA（正式发布）**，默认启用，无需 feature gate。ResourceClaimTemplate/DeviceClass API 版本为 `resource.k8s.io/v1`。

## 3.1 安装 nvidia-device-plugin

使用 nvidia-device-plugin DaemonSet 将 GPU 作为 `nvidia.com/gpu` 资源暴露给 k8s。

```bash
# 安装 Helm（如尚未安装）
curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# 安装 nvidia-device-plugin
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm repo update
helm install nvidia-device-plugin nvdp/nvidia-device-plugin \
  --namespace kube-system \
  --wait --timeout 300s
```

> **Node label 要求**：nvidia-device-plugin DaemonSet 默认使用 nodeAffinity 匹配 `feature.node.kubernetes.io/pci-10de.present=true`（NVIDIA PCI vendor ID）。如果没有安装 Node Feature Discovery (NFD)，需要手动给 GPU Worker 打这个 label：
> ```bash
> kubectl label node <WORKER_NAME> feature.node.kubernetes.io/pci-10de.present=true
> ```
> 不打此 label，DaemonSet 的 DESIRED=0，GPU 不会被发现。

```bash
# 验证 GPU 可见
kubectl get nodes -o custom-columns="NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu"
```

**预期输出**：每个 Worker 节点显示 `4` 个 GPU。如果显示 `<none>`，检查 device-plugin pod 是否调度到了 Worker（`kubectl get pods -n kube-system -l app.kubernetes.io/name=nvidia-device-plugin`）。

## 3.2 DRA GPU Driver（含 ComputeDomain 控制器）

**DRA GPU Driver**（v25.12.0+）提供 ComputeDomain CRD 和控制器，用于管理 IMEX daemon 生命周期。安装后会部署 kubelet plugin（DaemonSet）和 controller（Deployment）。

```bash
# 安装 DRA GPU Driver via Helm (OCI registry, 非传统 repo)
# 注意：版本号为 0.4.0（不是 v25.12.0）
helm upgrade --install nvidia-dra-driver-gpu \
  oci://registry.k8s.io/dra-driver-nvidia/charts/dra-driver-nvidia-gpu \
  --version 0.4.0 \
  --namespace nvidia-dra-driver-gpu --create-namespace \
  --set nameOverride=nvidia-dra-driver-gpu \
  --set nvidiaDriverRoot=/ \
  --set gpuResourcesEnabledOverride=true \
  --set controller.affinity=null \
  --set controller.priorityClassName='' \
  --set kubeletPlugin.priorityClassName=''

# v0.4.0 CRD 需手动 apply（helm chart 默认不含 ComputeDomain CRD）
kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/dra-driver-nvidia-gpu/v0.4.0/deployments/helm/dra-driver-nvidia-gpu/crds/resource.nvidia.com_computedomains.yaml

# 验证 DRA GPU Driver 组件
kubectl -n nvidia-dra-driver-gpu get pods -o wide
# 应看到: controller Pod (1/1) + 每个 GPU Worker 一个 kubelet-plugin Pod (2/2)

# 验证 ComputeDomain CRD 已注册
kubectl get crd | grep computedomain
# 应有: computedomains.resource.nvidia.com + computedomaincliques.resource.nvidia.com
```

> **实测验证**（2026-06-27）：DRA GPU Driver 0.4.0 从 OCI registry `registry.k8s.io/dra-driver-nvidia/charts/dra-driver-nvidia-gpu` 安装成功。CRD 需单独 apply。

**前提**：Worker 节点的 IMEX channels 必须已创建（`/dev/nvidia-caps-imex-channels/channel0` 存在）。这在 Worker 启动脚本的 Phase 1 中已配置（`NVreg_CreateImexChannel0=1` + `dracut --force` + reboot）。

**互斥**：ComputeDomain 与 nvidia-imex systemd 服务互斥。如 Worker 上有 `nvidia-imex.service` 在运行，必须先停止：`systemctl stop nvidia-imex && systemctl disable nvidia-imex`

## 3.3 安装 DRANET

DRANET v1.3.0 是 Kubernetes SIG 的 DRA 网络驱动，用于将 RDMA NIC 作为 DRA 设备分配给 Pod。

```bash
# 安装 DRANET v1.3.0 via Helm
helm install dranet oci://registry.k8s.io/networking/charts/dranet \
  --version $DRANET_VERSION \
  --namespace kube-system \
  --wait --timeout 300s

# 验证 DRANET DaemonSet
kubectl -n kube-system get ds dranet

# 验证 ResourceSlice（每个节点应发现 RDMA 设备）
kubectl get resourceslice -o wide | head -20
```

## 3.4 创建 RDMA DeviceClass

```bash
kubectl apply -f yamls/k8s1341-rdma-deviceclass.yaml
```

DeviceClass 内容（注意 API 为 `resource.k8s.io/v1`）：

```yaml
apiVersion: resource.k8s.io/v1
kind: DeviceClass
metadata:
  name: rdma-devices
spec:
  selectors:
  - cel:
      expression: |
        device.driver == "dra.net" &&
        has(device.attributes["dra.net"].rdma) && device.attributes["dra.net"].rdma == true
```

#### 验证安装

```bash
kubectl get deviceclass
# 应有: rdma-devices

kubectl get resourceslice -o wide | head -20
```

## 3.5 Scheduler RBAC 补全（kubeadm 1.34 必需）

**关键步骤**：kubeadm 1.34.x 的默认 `system:kube-scheduler` ClusterRole 严重不完整——不仅缺少 DRA 权限，连 pods、nodes、services 等基础资源的 list/watch 权限都缺失。如果跳过此步骤：
- ComputeDomain daemon pods 永久停留在 ContainerCreating（无日志无报错）
- DRA ResourceClaims 永久 pending（scheduler 无法 list ResourceSlices/DeviceClasses）
- 使用 podAffinity/podAntiAffinity 的 Pod 无法调度

```bash
# Scheduler RBAC 补全 — 必须在部署任何 GPU 工作负载之前执行
cat <<'EOF' | kubectl apply -f -
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: system:kube-scheduler:full
rules:
- apiGroups: [""]
  resources: ["pods", "pods/status", "pods/binding"]
  verbs: ["get", "list", "watch", "update", "patch", "create", "delete"]
- apiGroups: [""]
  resources: ["nodes", "nodes/status"]
  verbs: ["get", "list", "watch", "update", "patch"]
- apiGroups: [""]
  resources: ["services", "endpoints", "namespaces", "configmaps"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["events"]
  verbs: ["create", "patch", "update", "get", "list", "watch"]
- apiGroups: [""]
  resources: ["persistentvolumeclaims", "persistentvolumeclaims/status", "persistentvolumes"]
  verbs: ["get", "list", "watch", "update", "patch"]
- apiGroups: [""]
  resources: ["replicationcontrollers"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["replicasets", "statefulsets", "daemonsets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["policy"]
  resources: ["poddisruptionbudgets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["storage.k8s.io"]
  resources: ["storageclasses", "csinodes", "csidrivers", "csistoragecapacities"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["resource.k8s.io"]
  resources: ["resourceclaims", "resourceclaims/status"]
  verbs: ["get", "list", "watch", "update", "patch"]
- apiGroups: ["resource.k8s.io"]
  resources: ["resourceslices", "deviceclasses"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["coordination.k8s.io"]
  resources: ["leases"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["events.k8s.io"]
  resources: ["events"]
  verbs: ["create", "patch", "update"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: system:kube-scheduler:full
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: system:kube-scheduler:full
subjects:
- apiGroup: rbac.authorization.k8s.io
  kind: User
  name: system:kube-scheduler
EOF

# 重启 Scheduler 使 informer 重新同步
kubectl delete pod -n kube-system -l component=kube-scheduler
kubectl get pods -n kube-system -l component=kube-scheduler -w
# 等待 Running 1/1
```

## 3.6 创建 ComputeDomain

**ComputeDomain** 是 DRA GPU Driver 提供的 CRD，声明式管理 IMEX daemon 生命周期。创建后，控制器自动：

1. 创建 daemon DaemonSet → 在域内节点上启动 IMEX daemon pods
2. 创建 ResourceClaimTemplate（Pod 通过此模板获取 IMEX channel）
3. IMEX daemon pods 通过 pod CIDR 互联（port 50000），建立 IMEX session

```bash
# 前提：nvidia-imex.service 必须已禁用（3.2 步骤已处理）

# 为每个 Domain 创建 ComputeDomain
for d in $(seq 1 $NUM_DOMAINS); do
  CD_NAME="domain-${d}-compute-domain"
  echo "=== Creating ComputeDomain: ${CD_NAME} ==="

  cat <<EOF | kubectl apply -f -
apiVersion: resource.nvidia.com/v1beta1
kind: ComputeDomain
metadata:
  name: ${CD_NAME}
spec:
  numNodes: 0
  channel:
    allocationMode: Single
    resourceClaimTemplate:
      name: ${CD_NAME}-channel
EOF

  # numNodes: 0 — 推荐值（IMEXDaemonsWithDNSNames=true 时），不门控 daemon 启动
  # allocationMode: Single — 每个 Pod 独占一个 IMEX channel

  # 获取 ComputeDomain UID 并标记域内节点
  CD_UID=$(kubectl get computedomain ${CD_NAME} -o jsonpath='{.metadata.uid}')
  echo "ComputeDomain UID: $CD_UID"

  for i in $(seq 0 $((NODES_PER_DOMAIN - 1))); do
    kubectl label node ${WORKER_PREFIX}-d${d}-w${i} \
      resource.nvidia.com/computeDomain=$CD_UID --overwrite
  done
done

# 等待 daemon pods 启动（通常 30-60 秒）
kubectl get pods -n gpu-operator -l app=computedomain-daemon -w

# 验证所有 ComputeDomain 状态
kubectl get computedomain -o wide
# 每个 ComputeDomain 状态应为 Ready
```

**预期**：ComputeDomain 状态 Ready，同域节点各有一个 daemon pod 在运行，IMEX session 建立成功。

**注**：daemon 日志中会不断尝试连接不存在的 peer（如 domain 共 18 节点但只使用 2 节点），属正常行为可忽略。

**生产环境**：每个物理 NVL72 域需要一个 ComputeDomain。1800 张卡 = 25 个域 × 18 节点/域 = 25 个 ComputeDomain + 25 个 Placement Policy。

**域内多任务**：ComputeDomain **支持**域内多任务并存。每个节点最多属于一个 ComputeDomain（每节点只能运行一个 IMEX daemon），但同一 ComputeDomain 的 ResourceClaimTemplate 可被多个 Pod 共享。

---

## 4. GCSFuse + Lustre 共享存储

### 4.1 GCSFuse（已在 startup script 中配置）

每个 Worker 的 startup script (`tlinux4-k8s134-worker.sh`) 已包含 GCSFuse 安装和挂载：

```bash
# startup script 自动执行：
dnf install -y gcsfuse

# 挂载 GCS bucket（使用 Local SSD 作为文件缓存）
mkdir -p ${LOCAL_SSD_MOUNT}/shared
gcsfuse --implicit-dirs \
  --file-cache-dir=${LOCAL_SSD_MOUNT}/gcsfuse-cache \
  $GCSFUSE_BUCKET ${LOCAL_SSD_MOUNT}/shared
```

Pod 中通过 hostPath 访问：

```yaml
volumes:
- name: shared-data
  hostPath:
    path: /mnt/stateful_partition/shared    # GCSFuse 挂载点
- name: scratch-data
  hostPath:
    path: /mnt/stateful_partition/scratch-data  # Local SSD 高速缓存
```

### 4.2 GCSFuse v2 高性能配置（可选）

GCSFuse v2 支持并行下载和内核缓存，可显著提升大模型 checkpoint 加载速度。

```bash
# 在 startup script 中启用 v2 高性能挂载
gcsfuse \
  --implicit-dirs \
  --file-cache-dir=${LOCAL_SSD_MOUNT}/gcsfuse-cache \
  --file-cache-capacity-per-file-mb=-1 \
  --file-cache-cache-file-for-range-read \
  --file-cache-enable-parallel-downloads \
  --file-cache-parallel-downloads-per-file=16 \
  --file-cache-download-chunk-size-mb=50 \
  --file-cache-max-parallel-downloads=64 \
  --kernel-list-cache-ttl-secs=600 \
  --metadata-cache-ttl-secs=600 \
  --stat-cache-capacity=20000 \
  --type-cache-max-size-mb=32 \
  --rename-dir-limit=200000 \
  $GCSFUSE_BUCKET ${LOCAL_SSD_MOUNT}/shared
```

**关键参数说明**：

- `--file-cache-enable-parallel-downloads`：启用文件级并行下载（v2 新增）
- `--file-cache-parallel-downloads-per-file=16`：每个文件最多 16 个并行块
- `--file-cache-capacity-per-file-mb=-1`：不限制单文件缓存大小
- `--kernel-list-cache-ttl-secs=600`：目录列表缓存 10 分钟，减少 metadata API 调用

**注**：文件缓存位于 Local SSD RAID0（`/mnt/stateful_partition/gcsfuse-cache`），A4X 自带 ~12TB NVMe，提供极高的缓存 IOPS。

### 4.3 Lustre（可选）

如果客户提供 Lustre 文件系统，使用 Lustre CSI Driver + PV/PVC 方式挂载。

#### Kernel 6.6 与 Lustre 版本兼容性

GB200 Worker 运行 kernel 6.6（TLinux 4 / Rocky Linux 9.x），**Lustre 2.14 client 内核模块无法在 kernel 6.6 上编译**。2.14 仅支持 kernel 4.18-5.x（RHEL 8 时代），6.x 内核的 VFS/网络栈 API 变更导致源码不兼容。

| Lustre 版本 | 支持的最高 kernel | 备注 |
|---|---|---|
| 2.14.x | ~5.x (RHEL 8) | 无法在 kernel 6.6 上编译 |
| 2.15.7+ | 5.14 (RHEL 9.6) | 部分 6.x 支持 |
| 2.16+ | 6.x (RHEL 9.4+) | 推荐 |
| 2.17+ | 6.6+ | 完整支持 |

如果客户 Lustre **服务端**是 2.14，**客户端必须用 2.16+ 或 2.17**（向后兼容，新 client 可连老 server）。

#### Grant 死锁：2.17 Client + 2.14 Server 踩坑（实测）

**现象**：buffered I/O 写入（`cp -a`、`tar`、应用写入）后，sync 卡死，后续所有 I/O 包括读都被阻塞。`dd oflag=direct` 测试**无法复现**（direct I/O 绕过页缓存和 grant 机制）。

**根因**：client/server 版本混搭导致 grant 参数不匹配。

- Lustre 2.14 server：`initial_grant=8MB`，动态最大 ~60MB
- Lustre 2.17 client 默认：`max_dirty_mb=64MB`（每 OST 脏页上限）
- 64MB > 60MB → 客户端积压的脏页超过 server 授权量 → grant 耗尽（`cur_grant_bytes=0`）→ 刷盘 RPC 无法发送 → 永久死锁

**内核报错**：
```
LustreError: osc_announce_cached: data-OST0004-osc: dirty 16675 > dirty_max 16384
LustreError: osc_extent_wait: data-OST0007-osc: wait ext to 0 timedout, recovery in progress?
```

**修复**：降低 client 参数，确保脏页上限 < server grant 上限。

```bash
# 临时生效
lctl set_param osc.*.max_dirty_mb=32       # 默认 64 → 32（< server max grant ~60MB）
lctl set_param osc.*.max_rpcs_in_flight=4   # 默认 8 → 4（给 2.14 server 更多处理时间）
```

**持久化**：`/etc/lctl.conf` 只在模块加载时生效一次，无法覆盖 CSI 挂载创建的新 OSC 实例。必须用 systemd service：

```bash
cat <<'EOF' | sudo tee /etc/systemd/system/lustre-tuning.service
[Unit]
Description=Lustre client tuning for 2.14 server compatibility
After=kubelet.service

[Service]
Type=oneshot
ExecStartPre=/bin/sleep 30
ExecStart=/usr/sbin/lctl set_param osc.*.max_dirty_mb=32
ExecStart=/usr/sbin/lctl set_param osc.*.max_rpcs_in_flight=4
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload && sudo systemctl enable lustre-tuning
```

#### GCS → Lustre 数据传输

直接从 GCS 写入 Lustre（`gcloud storage rsync`）会因 composite download 产生大量并发 buffered 写入，快速耗尽 grant。推荐两段式：

```bash
# 1. GCS → 本地 SSD（~3.8 GB/s）
gcloud storage rsync -r gs://bucket/models /lssd/models

# 2. 本地 SSD → Lustre（多目录并行 cp，~6 GB/s）
for dir in /lssd/models/*/; do
  cp -a "$dir" /mnt/lustre/models/ &
done
wait

# 3. 确认脏页清零（非零说明 grant 死锁，需 umount -l + 重新 mount）
lctl get_param osc.*.cur_dirty_bytes | grep -v "=0$"
```

#### vLLM 从 Lustre 加载模型

vLLM 默认使用 mmap 加载 safetensors。`tensor-parallel-size > 1` 时，多 GPU worker 同时触发 mmap page fault，Lustre 的 `ll_filemap_fault` 会串行化这些 fault，读取性能随 shard 数指数级下降。解法：先 `cp` 模型到 Local SSD 或 `/dev/shm`，再从本地加载。

---

## k8s 1.34 DRA + ComputeDomain 注意事项

| 项目 | 说明 |
|------|------|
| DRA API 版本 | ResourceClaimTemplate/DeviceClass 使用 `resource.k8s.io/v1`（GA API，非 v1beta2） |
| DRA 状态 | k8s 1.34 中 DRA 为 **GA**，默认启用，无需 feature gate |
| kubeadm Scheduler RBAC | kubeadm 1.34.x 默认 `system:kube-scheduler` ClusterRole 严重不完整——不仅缺 DRA 权限，连 pods/nodes/services 等基础权限都缺。**必须手动补全**（见 3.5） |
| DRANET 版本 | v1.3.0，安装: `helm install dranet oci://registry.k8s.io/networking/charts/dranet --version v1.3.0 -n kube-system` |
| ComputeDomain 互斥 | 与 `nvidia-imex.service`（systemd）和 IMEX Manager DaemonSet **互斥** |
| ComputeDomain 节点标签 | 创建后需手动 `kubectl label node <NODE> resource.nvidia.com/computeDomain=<UID>` |
| IMEX daemon 日志噪音 | <18 节点环境下 daemon 会持续尝试连接不存在的节点 — **无害可忽略** |
| NCCL 测试二进制 | pytorch 镜像自带的 `all_reduce_perf`**未链接 MPI**，必须从源码编译 MPI 版 |
| mpirun 路径 | `/usr/local/mpi/bin/mpirun`（非 `/usr/local/gib/bin/mpirun`） |
| Calico 多网卡 | A4X Worker 有 6 个 NIC（2 GVNIC + 4 MRDMA），Calico Installation 必须设置 `nodeAddressAutodetectionV4.cidrs` 为管理子网。默认 `firstFound` 会选中 RDMA 网卡导致 BGP 失败、Pod DNS 瘫痪（见 02-k8s-cluster Step 7 注释） |
| SSH 密钥 | 容器内必须用 `ed25519`（`ssh-keygen -t ed25519`），RSA 因无 `/dev/tty` 会失败 |
