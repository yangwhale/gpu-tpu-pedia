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
```

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

## 2.3 Worker 节点 k8s 加入

TLinux 4 Worker 使用 2 阶段设置脚本 `tlinux4-k8s134-worker.sh`：

- **Phase 1**：配置 IMEX initramfs (`NVreg_CreateImexChannel0=1`) + `dracut --force` + 自动重启
- **Phase 2**：安装 containerd → NVIDIA Container Toolkit → kubeadm/kubelet → `kubeadm join`

```bash
# SSH 到每个 Worker，运行设置脚本（可批量并行）
for d in $(seq 1 $NUM_DOMAINS); do
  for i in $(seq 0 $((NODES_PER_DOMAIN - 1))); do
    WORKER="${WORKER_PREFIX}-d${d}-w${i}"
    gcloud compute ssh $WORKER --zone=$ZONE --project=$PROJECT --tunnel-through-iap \
      --command="sudo bash /root/scripts/tlinux4-k8s134-worker.sh" &
  done
done
wait

# Phase 1 完成后节点自动重启，重启后 Phase 2 自动运行并 join 集群
# 等待 ~5 分钟让所有 Worker 完成 Phase 2

# 在 CP 上验证
kubectl get nodes -o wide
# 应看到 1 CP + (NUM_DOMAINS × NODES_PER_DOMAIN) Worker 节点 (全部 Ready)
```

**TLinux 4 注意事项**：

- Docker CE repo 需要硬编码 `baseurl=https://download.docker.com/linux/rhel/9/$(uname -m)/stable`（TLinux 4 基于 RHEL 9）
- NVIDIA Container Toolkit 使用 `nvidia.github.io/libnvidia-container/stable/rpm` repo
- 启动盘设备名不固定，脚本使用 `findmnt` 动态查找

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
