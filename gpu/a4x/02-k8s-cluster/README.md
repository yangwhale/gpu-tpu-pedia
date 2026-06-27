# 2. 创建 k8s 1.34.1 集群

**k8s 1.34 关键变化**：DRA (Dynamic Resource Allocation) 在 k8s 1.34 中为 **GA**，默认启用，无需 feature gate。ResourceClaimTemplate/DeviceClass API 版本为 `resource.k8s.io/v1`（非 v1beta2）。

## 2.1 Control Plane 节点

CP 节点使用 x86_64 VM（无需 GPU），仅连接主 GVNIC 管理网络。

> **机型建议**：`n4-standard-8`（8 核 32GB）或更高。`e2-standard-4` 可以跑但偏弱——etcd 和 API server 在大集群下对 CPU 和 IOPS 有要求。磁盘建议 200GB+。

> **网络复用**：如果已有管理网络（如 `chrisya-gvnic-net-0`），可以直接复用，无需新建 VPC。只需确保目标 region 有子网。复用已有网络的好处是 SSH 直通——同 VPC 内任意 VM 可通过内网 IP 直接 SSH。

```bash
# 方式 A：使用已有网络（推荐，SSH 直通）
gcloud compute instances create $CP_NAME \
  --project=$PROJECT --zone=$ZONE \
  --machine-type=n4-standard-8 \
  --image-family=rocky-linux-9 --image-project=rocky-linux-cloud \
  --boot-disk-size=200GB \
  --network-interface=network=$GVNIC_NET,subnet=$GVNIC_SUB \
  --scopes=cloud-platform

# 方式 B：新建独立网络（需要 IAP tunnel SSH）
gcloud compute instances create $CP_NAME \
  --project=$PROJECT --zone=$ZONE \
  --machine-type=n4-standard-8 \
  --image-family=rocky-linux-9 --image-project=rocky-linux-cloud \
  --boot-disk-size=200GB \
  --network-interface=network=$GVNIC_NET,subnet=$GVNIC_SUB \
  --metadata-from-file=startup-script=scripts/kubeadm-control-plane-k8s134.sh \
  --scopes=cloud-platform
```

### SSH 到 CP 节点

```bash
# 方式 A：同 VPC 内网直连（需先注入 SSH key）
# 在创建时通过 --metadata 注入，或创建后：
gcloud compute instances add-metadata $CP_NAME \
  --zone=$ZONE --project=$PROJECT \
  --metadata=ssh-keys="$USER:$(cat ~/.ssh/id_ed25519.pub)"
ssh $USER@<CP_INTERNAL_IP>

# 方式 B：IAP tunnel（不同网络时使用）
gcloud compute ssh $CP_NAME --zone=$ZONE --project=$PROJECT --tunnel-through-iap
```

### CP 手动安装步骤

如果未使用 startup script，SSH 到 CP 后手动执行以下 7 步：

```bash
# Step 1: 禁用 swap 和 SELinux
sudo swapoff -a
sudo sed -i '/swap/d' /etc/fstab
sudo setenforce 0
sudo sed -i 's/^SELINUX=enforcing$/SELINUX=permissive/' /etc/selinux/config

# Step 2: 加载内核模块 + sysctl
sudo tee /etc/modules-load.d/k8s.conf <<EOF
overlay
br_netfilter
EOF
sudo modprobe overlay && sudo modprobe br_netfilter

sudo tee /etc/sysctl.d/k8s.conf <<EOF
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
EOF
sudo sysctl --system

# Step 3: 安装 containerd (Docker CE repo)
sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo dnf install -y containerd.io
sudo mkdir -p /etc/containerd
sudo sh -c 'containerd config default > /etc/containerd/config.toml'
sudo sed -i 's/SystemdCgroup = false/SystemdCgroup = true/' /etc/containerd/config.toml
sudo systemctl enable --now containerd

# Step 4: 安装 kubeadm/kubelet/kubectl
#   注意：必须从 pkgs.k8s.io 安装 kubectl，不要用 google-cloud-sdk 自带的
#   （google-cloud-sdk 的 kubectl 版本号不匹配 k8s 1.34）
sudo tee /etc/yum.repos.d/kubernetes.repo <<EOF
[kubernetes]
name=Kubernetes
baseurl=https://pkgs.k8s.io/core:/stable:/v1.34/rpm/
enabled=1
gpgcheck=1
gpgkey=https://pkgs.k8s.io/core:/stable:/v1.34/rpm/repodata/repomd.xml.key
exclude=kubectl
EOF
sudo dnf install -y kubelet kubeadm --disableexcludes=kubernetes
# 单独从 k8s repo 安装 kubectl（避免 google-cloud-sdk 版本）
sudo dnf install -y --repo=kubernetes kubectl
sudo systemctl enable --now kubelet

# Step 5: kubeadm init
#   --node-name 指定干净的主机名，避免使用 FQDN
sudo kubeadm init \
  --pod-network-cidr=10.244.0.0/16 \
  --node-name=$CP_NAME

# Step 6: 配置 kubeconfig
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

# Step 7: 安装 Calico v3.29.3 CNI
kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.29.3/manifests/tigera-operator.yaml
# 等待 tigera-operator 就绪
sleep 10
cat <<EOF | kubectl create -f -
apiVersion: operator.tigera.io/v1
kind: Installation
metadata:
  name: default
spec:
  calicoNetwork:
    nodeAddressAutodetectionV4:
      cidrs:
      - "$MGMT_SUBNET"    # 管理子网 CIDR（如 10.14.0.0/24）
    ipPools:
    - blockSize: 26
      cidr: 10.244.0.0/16
      encapsulation: VXLANCrossSubnet
      natOutgoing: Enabled
      nodeSelector: all()
EOF

# 验证
kubectl get nodes  # 应显示 CP 节点 Ready
kubectl get pods -n calico-system  # Calico pods 应逐步 Running
# 全部 calico-node 必须 1/1 Ready — 0/1 说明 BGP peering 失败
```

> **Calico 多网卡陷阱（关键）**：A4X Worker 有 6 个 NIC（2 GVNIC + 4 MRDMA）。Calico 默认 `firstFound: true` IP 自动检测会选中 RDMA 网卡（如 10.10.28.x）而非管理 GVNIC（如 10.14.0.x），导致 BIRD BGP peering 失败 → CP 上的 calico-node 永远 0/1 Not Ready → Pod DNS 完全瘫痪（CoreDNS 在 CP 上，VXLAN 隧道不通）。
>
> 修复：必须在 Installation CRD 中设置 `nodeAddressAutodetectionV4.cidrs` 为管理子网 CIDR。如果忘了设置或设错了，后续 patch 方法：
> ```bash
> kubectl patch installation default --type=json -p '[
>   {"op": "replace", "path": "/spec/calicoNetwork/nodeAddressAutodetectionV4",
>    "value": {"cidrs": ["10.14.0.0/24"]}}
> ]'
> kubectl delete pods -n calico-system -l k8s-app=calico-node  # 重启 calico-node
> ```

> **kubectl 版本陷阱**：Rocky Linux 如果配置了 google-cloud-sdk repo，`dnf install kubectl` 会优先安装 google-cloud-sdk 版本（如 574.0.0），而非 k8s 1.34.x 版本。建议在 kubernetes.repo 中 `exclude=kubectl` 避免冲突，然后 `--repo=kubernetes` 单独安装。

### 获取 Join 信息

```bash
# kubeadm init 输出的最后几行包含 join 命令，格式如：
# kubeadm join <CP_IP>:6443 --token <TOKEN> --discovery-token-ca-cert-hash sha256:<HASH>

# 也可以后续提取：
CP_IP=$(hostname -I | awk '{print $1}')
JOIN_TOKEN=$(kubeadm token list -o jsonpath='{.token}' | head -1)
JOIN_HASH=$(openssl x509 -pubkey -in /etc/kubernetes/pki/ca.crt | \
  openssl rsa -pubin -outform der 2>/dev/null | sha256sum | awk '{print $1}')
echo "CP_IP=$CP_IP JOIN_TOKEN=$JOIN_TOKEN JOIN_HASH=$JOIN_HASH"

# Token 有效期 24h，过期后重新生成：
kubeadm token create --print-join-command
```

## 2.2 Placement Policy 与 Worker 节点 VM 创建

**Domain 与 Placement Policy 的关系**：

- 每个 NVL72 Domain = 18 节点 × 4 GPU = 72 GPU，是物理 NVSwitch 拓扑决定的
- 每个 Domain 需要独立的 `Placement Policy`（[01-environment-setup](../01-environment-setup/) 1.4 节创建），`--resource-policies` 参数决定 VM 分配到哪个 Domain
- 生产环境（如 1800 GPU = 25 Domain）：为每个 Domain 批量创建 Worker，每批使用对应 Domain 的 Placement Policy
- 使用 GA `gcloud compute`（不是 alpha/beta），不加 `--local-ssd`（A4X 自动挂载 12TB NVMe）
- `no-address` on MRDMA — 网络 profile 不允许 MRDMA 接口有 AccessConfig
- **不需要一次创建 18 台**——只建 2~4 台测试完全可以，空位以后再加

### 查看和复用现有 Placement Policy

在创建新 Policy 之前，先查看项目里已有的 Policy 和使用情况：

```bash
# 列出所有 Placement Policy
gcloud beta compute resource-policies list \
  --project=$PROJECT --filter="region~$REGION" \
  --format="table(name,region.basename(),status)"

# 查看每台 A4X VM 使用的 Policy（找空位多的 Policy 复用）
gcloud compute instances list \
  --project=$PROJECT --zones=$ZONE \
  --filter="machineType~a4x" \
  --format="table(name,status,resourcePolicies.basename())"
```

如果某个 Policy 只有 2~4 台 VM（空位 14~16 个），可以直接复用——你的 VM 会加入同一个物理域。注意同域只能有一个 ComputeDomain（详见 [01-environment-setup](../01-environment-setup/) 0.6 节）。

### Worker 镜像选择

| 镜像 | 项目 | 预装内容 | 适用场景 |
|------|------|----------|----------|
| `rocky-linux-9-optimized-gcp-nvidia-580-arm64` | `rocky-linux-accelerator-cloud` | NVIDIA 580 驱动 + RDMA NIC 驱动 | **推荐**：GCP 官方公共镜像，从此开始安装 containerd/kubelet/IMEX |
| `rocky-linux-9-optimized-gcp-nvidia-latest-arm64` | `rocky-linux-accelerator-cloud` | 最新 NVIDIA 驱动 | 需要最新驱动特性时使用 |
| `tlinux-server-4-gb200-v4` | 项目内私有 | NVIDIA 驱动 + TLinux 4 定制 OS | 客户定制环境（配合 `tlinux4-k8s134-worker.sh` startup script） |
| `rocky-linux-9-arm64` | `rocky-linux-cloud` | 纯净 OS，无 NVIDIA 驱动 | 需要完全自主安装驱动时使用（不推荐，GB200 驱动安装复杂） |

> **实测验证**（2026-06-27）：`rocky-linux-9-optimized-gcp-nvidia-580-arm64` 镜像创建 A4X VM 后，4 块 GB200 GPU (189GB HBM each) 和 4 块 RDMA NIC 自动就绪，`nvidia-smi` 正常工作。

### NIC 配置要求

> **重要**：A4X 的 NIC 配置有严格顺序要求——**前 2 个必须是 GVNIC，后 4 个必须是 MRDMA**。不满足此要求 `gcloud` 会直接报错。
>
> 错误示例：只给 1 个 GVNIC + 4 个 MRDMA → `On a4x-highgpu-4g, the first NIC (if present) and the second NIC (if present) must be of type GVNIC. These must be followed by 0 or 4 MRDMA NICs.`
>
> 因此必须准备 **2 个 GVNIC 网络**（主管理网络 + 辅助网络）和 **1 个 RDMA 网络**（4 个子网）。

### 批量创建 Worker（每个 Domain 使用对应的 Placement Policy）

**生产环境示例**：25 个 Domain × 18 节点/Domain = 450 Worker VM。

```bash
# RDMA 子网名（由 01-environment-setup 创建时的命名决定）
RDMA_SUB_0="${RDMA_NET}-sub-0"
RDMA_SUB_1="${RDMA_NET}-sub-1"
RDMA_SUB_2="${RDMA_NET}-sub-2"
RDMA_SUB_3="${RDMA_NET}-sub-3"

# 循环创建所有 Domain 的 Worker VM
for d in $(seq 1 $NUM_DOMAINS); do
  echo "=== Creating workers for Domain ${d} ==="
  for i in $(seq 0 $((NODES_PER_DOMAIN - 1))); do
    gcloud compute instances create ${WORKER_PREFIX}-d${d}-w${i} \
      --project=$PROJECT --zone=$ZONE \
      --machine-type=$MACHINE_TYPE \
      --image-family=rocky-linux-9-optimized-gcp-nvidia-580-arm64 \
      --image-project=rocky-linux-accelerator-cloud \
      --boot-disk-size=500GB --boot-disk-type=hyperdisk-balanced \
      --scopes=cloud-platform \
      --reservation-affinity=specific --reservation=$RESERVATION \
      --maintenance-policy=TERMINATE \
      --restart-on-failure \
      --resource-policies=${PLACEMENT_PREFIX}-${d} \
      --network-interface=nic-type=GVNIC,network=$GVNIC_NET,subnet=$GVNIC_SUB \
      --network-interface=nic-type=GVNIC,network=$GVNIC_NET_1,subnet=$GVNIC_SUB_1,no-address \
      --network-interface=nic-type=MRDMA,network=$RDMA_NET,subnet=$RDMA_SUB_0,no-address \
      --network-interface=nic-type=MRDMA,network=$RDMA_NET,subnet=$RDMA_SUB_1,no-address \
      --network-interface=nic-type=MRDMA,network=$RDMA_NET,subnet=$RDMA_SUB_2,no-address \
      --network-interface=nic-type=MRDMA,network=$RDMA_NET,subnet=$RDMA_SUB_3,no-address \
      --metadata=ssh-keys="$USER:$(cat ~/.ssh/id_ed25519.pub)" &
  done
  wait  # 等待该 Domain 的 Worker 全部创建完成
done
```

> **TLinux 镜像用户**：如果使用 `tlinux-server-4-gb200-v*` 镜像，可以通过 `--metadata-from-file=startup-script=scripts/tlinux4-k8s134-worker.sh` 和 `--metadata=cp-ip=$CP_IP,join-token=$JOIN_TOKEN,join-hash=$JOIN_HASH` 实现自动安装和 join。

**命名规则**：Worker 名称格式为 `${WORKER_PREFIX}-d${DOMAIN}-w${INDEX}`，例如 `gb200-d1-w0` 到 `gb200-d1-w17` 为 Domain 1 的 18 个节点。

## 2.3 Worker 硬件调优 + k8s 加入

Worker 使用 GCP 官方 `rocky-linux-9-optimized-gcp-nvidia-580-arm64` 镜像时，GPU 驱动已预装，但需要手动完成硬件调优和 k8s 软件安装。分两阶段（中间需重启一次）。

### Phase 1：硬件调优（需重启）

```bash
# SSH 到 Worker
ssh $USER@<WORKER_IP>

# === IMEX initramfs 配置 ===
# 启用 IMEX channel 设备文件（跨节点 NVLink 的前提）
echo "options nvidia NVreg_CreateImexChannel0=1" | sudo tee /etc/modprobe.d/nvidia.conf
sudo dracut --force --add-drivers "gve"

# === 内核模块 (k8s 前置) ===
sudo tee /etc/modules-load.d/k8s.conf <<EOF
overlay
br_netfilter
tcp_bbr
EOF
sudo modprobe overlay && sudo modprobe br_netfilter && sudo modprobe tcp_bbr

# === sysctl (k8s + Grace CPU 调优) ===
sudo tee /etc/sysctl.d/k8s.conf <<EOF
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
EOF

sudo tee /etc/sysctl.d/90-grace-gb200.conf <<EOF
kernel.numa_balancing = 0
net.ipv4.tcp_congestion_control = bbr
net.core.default_qdisc = fq
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216
vm.swappiness = 0
vm.zone_reclaim_mode = 0
EOF
sudo sysctl --system

# === GPU persist mode + 关闭不需要的服务 ===
sudo nvidia-smi -pm 1
sudo systemctl disable --now irqbalance 2>/dev/null || true

# === 重启（IMEX initramfs 生效需要重启） ===
sudo reboot
```

> **GB200 重启时间**：GB200 A4X 重启比普通 VM 慢很多——GPU 卸载 + NVSwitch 释放 + UEFI 重新初始化，总共约 **5-8 分钟**。NVIDIA Persistence Daemon 的 stop job 可能卡 3 分钟（正常现象，不要强杀）。

> **BBR 内核模块**：sysctl 设置 `tcp_congestion_control = bbr` 需要先加载 `tcp_bbr` 内核模块，否则重启后会回退到 `cubic`。必须在 `/etc/modules-load.d/` 中添加 `tcp_bbr`。

### Phase 1 验证（重启后）

```bash
ssh $USER@<WORKER_IP>

# IMEX channel 应该存在
ls /dev/nvidia-caps-imex-channels/
# 期望输出: channel0

# sysctl 应该持久化
echo "numa=$(cat /proc/sys/kernel/numa_balancing) tcp=$(cat /proc/sys/net/ipv4/tcp_congestion_control)"
# 期望输出: numa=0 tcp=bbr

# GPU
nvidia-smi --query-gpu=name,persistence_mode --format=csv,noheader | head -1
# 期望输出: NVIDIA GB200, Enabled
```

### Phase 2：安装 containerd + kubelet + join 集群

```bash
# === containerd ===
sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo dnf install -y containerd.io
sudo mkdir -p /etc/containerd
sudo sh -c 'containerd config default > /etc/containerd/config.toml'
sudo sed -i 's/SystemdCgroup = false/SystemdCgroup = true/' /etc/containerd/config.toml
sudo systemctl enable --now containerd

# === NVIDIA Container Toolkit ===
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo dnf install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=containerd --set-as-default
sudo systemctl restart containerd

# === kubelet / kubeadm ===
sudo tee /etc/yum.repos.d/kubernetes.repo <<EOF
[kubernetes]
name=Kubernetes
baseurl=https://pkgs.k8s.io/core:/stable:/v1.34/rpm/
enabled=1
gpgcheck=1
gpgkey=https://pkgs.k8s.io/core:/stable:/v1.34/rpm/repodata/repomd.xml.key
EOF
sudo dnf install -y kubelet kubeadm --disableexcludes=kubernetes
# kubectl 单独装：先临时去掉 exclude，指定 repo 安装，再加回
sudo sed -i '/^exclude/d' /etc/yum.repos.d/kubernetes.repo
sudo dnf install -y --repo=kubernetes kubectl
echo "exclude=kubectl" | sudo tee -a /etc/yum.repos.d/kubernetes.repo
sudo systemctl enable kubelet

# === kubeadm join ===
# 使用 CP 节点的 join 命令（从 CP 上获取）
sudo kubeadm join <CP_IP>:6443 \
  --token <JOIN_TOKEN> \
  --discovery-token-ca-cert-hash sha256:<JOIN_HASH> \
  --node-name $(hostname)
```

### Phase 2 验证（在 CP 节点上）

```bash
kubectl get nodes -o wide
# 应看到 Worker 节点 Ready
# 注意：Worker 首次加入后 kubelet 需要 30-60 秒拉取 CNI 插件才会变为 Ready
```

> **kubectl 版本陷阱**：Rocky Linux 如果配了 google-cloud-sdk repo，`dnf install kubectl` 会装到 gcloud 版本（如 574.0.0）。必须从 kubernetes repo 单独安装——在 kubernetes.repo 中 `exclude=kubectl`，然后 `dnf install -y --repo=kubernetes kubectl`。

## 2.4 节点标签

### 方法 A：手动标记（小规模 / 验证环境）

```bash
# 按 Domain 标记 Worker 节点
for d in $(seq 1 $NUM_DOMAINS); do
  for i in $(seq 0 $((NODES_PER_DOMAIN - 1))); do
    kubectl label node ${WORKER_PREFIX}-d${d}-w${i} \
      nvl72-domain=domain-${d} --overwrite
  done
done

# 验证
kubectl get nodes -L nvl72-domain
```

### 方法 B：Startup Script 自动注册（生产 / 大规模环境，推荐）

GCP VM 可通过 Metadata Server 获取物理拓扑信息，无需 IAM 权限。同一 NVL72 domain 内的 VM 返回相同的拓扑哈希值。

```bash
# 在 Worker startup script 中（kubeadm join 之前）添加：

# 1. 从 Metadata Server 获取物理拓扑哈希（同 domain 的 VM 返回相同值）
TOPO_HASH=$(curl -sf -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/physical_host_topology" \
  | md5sum | cut -c1-8)  # 截取前 8 位作为域标识符

# 2. 配置 kubelet 启动参数，自动附带拓扑标签
mkdir -p /etc/systemd/system/kubelet.service.d
cat > /etc/default/kubelet <<KUBELET_EOF
KUBELET_EXTRA_ARGS=--node-labels=nvl72-domain=domain-${TOPO_HASH}
KUBELET_EOF
systemctl daemon-reload

# 3. kubeadm join 时节点自动携带标签
kubeadm join ${CP_IP}:6443 \
  --token "${JOIN_TOKEN}" \
  --discovery-token-ca-cert-hash "sha256:${JOIN_HASH}" \
  --node-name "${NODE_NAME}" \
  --ignore-preflight-errors=Hostname
```

**拓扑发现的两种 API**：

| 方式 | 端点 | 权限要求 | 返回值 |
|------|------|----------|--------|
| Metadata Server（推荐） | `http://metadata.google.internal/.../physical_host_topology` | 无（VM 内部直接访问） | 拓扑哈希字符串 |
| Compute API | `gcloud compute instances describe --format='value(resourceStatus.physicalHostTopology.subblock)'` | 需要 `compute.instances.get` IAM 权限 | 结构化的 subblock 标识 |

**注意**：Metadata Server 返回的是哈希值（不可读但稳定），同 domain 的 VM 保证返回相同值。Compute API 返回可读的 subblock 名称，但需要额外 IAM 配置。
