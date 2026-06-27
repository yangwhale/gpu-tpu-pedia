# 2. 创建 k8s 1.34.1 集群

**k8s 1.34 关键变化**：DRA (Dynamic Resource Allocation) 在 k8s 1.34 中为 **GA**，默认启用，无需 feature gate。ResourceClaimTemplate/DeviceClass API 版本为 `resource.k8s.io/v1`（非 v1beta2）。

## 2.1 Control Plane 节点

CP 节点使用 x86_64 轻量 VM（无需 GPU），仅连接主 GVNIC 网络。

```bash
gcloud compute instances create $CP_NAME \
  --project=$PROJECT --zone=$ZONE \
  --machine-type=e2-standard-4 \
  --image-family=rocky-linux-9 --image-project=rocky-linux-cloud \
  --boot-disk-size=100GB \
  --network-interface=network=$GVNIC_NET,subnet=$GVNIC_SUB \
  --metadata-from-file=startup-script=scripts/kubeadm-control-plane-k8s134.sh \
  --scopes=cloud-platform
```

### CP 启动脚本要点 (kubeadm-control-plane-k8s134.sh)

1. 禁用 swap 和 SELinux
2. 加载内核模块 (overlay, br_netfilter) + sysctl
3. 安装 containerd (Docker CE repo, SystemdCgroup=true)
4. 安装 kubeadm/kubelet/kubectl (pkgs.k8s.io v1.34 rpm repo)
5. `kubeadm init --pod-network-cidr=10.244.0.0/16`
6. 安装 Calico v3.29.3 CNI
7. 生成 join token 和 SSH key pair

### 获取 Join 信息

```bash
# SSH 到 CP 节点
gcloud compute ssh $CP_NAME --zone=$ZONE --project=$PROJECT --tunnel-through-iap

# 查看 join 命令
cat /root/kubeadm-join-command.txt

# 提取 token 和 hash
CP_IP=$(hostname -I | awk '{print $1}')
JOIN_TOKEN=$(kubeadm token list -o jsonpath='{.token}' | head -1)
JOIN_HASH=$(openssl x509 -pubkey -in /etc/kubernetes/pki/ca.crt | \
  openssl rsa -pubin -outform der 2>/dev/null | sha256sum | awk '{print $1}')
echo "CP_IP=$CP_IP JOIN_TOKEN=$JOIN_TOKEN JOIN_HASH=$JOIN_HASH"
```

## 2.2 Placement Policy 与 Worker 节点 VM 创建

**Domain 与 Placement Policy 的关系**：

- 每个 NVL72 Domain = 18 节点 × 4 GPU = 72 GPU，是物理 NVSwitch 拓扑决定的
- 每个 Domain 需要独立的 `Placement Policy`（[01-environment-setup](../01-environment-setup/) 1.4 节创建），`--resource-policies` 参数决定 VM 分配到哪个 Domain
- 生产环境（如 1800 GPU = 25 Domain）：为每个 Domain 批量创建 Worker，每批使用对应 Domain 的 Placement Policy
- 使用 GA `gcloud compute`（不是 alpha/beta），不加 `--local-ssd`（A4X 自动挂载 12TB NVMe）
- `no-address` on MRDMA — 网络 profile 不允许 MRDMA 接口有 AccessConfig

### 批量创建 Worker（每个 Domain 使用对应的 Placement Policy）

**生产环境示例**：25 个 Domain × 18 节点/Domain = 450 Worker VM。

```bash
# 循环创建所有 Domain 的 Worker VM
for d in $(seq 1 $NUM_DOMAINS); do
  echo "=== Creating workers for Domain ${d} ==="
  for i in $(seq 0 $((NODES_PER_DOMAIN - 1))); do
    gcloud compute instances create ${WORKER_PREFIX}-d${d}-w${i} \
      --project=$PROJECT --zone=$ZONE \
      --machine-type=$MACHINE_TYPE \
      --image=$IMAGE --image-project=$IMAGE_PROJECT \
      --boot-disk-size=1000GB --boot-disk-type=hyperdisk-balanced \
      --scopes=cloud-platform \
      --reservation-affinity=specific --reservation=$RESERVATION \
      --provisioning-model=RESERVATION_BOUND \
      --instance-termination-action=STOP \
      --maintenance-policy=TERMINATE \
      --restart-on-failure \
      --resource-policies=${PLACEMENT_PREFIX}-${d} \
      --network-interface=nic-type=GVNIC,network=$GVNIC_NET,subnet=$GVNIC_SUB \
      --network-interface=nic-type=GVNIC,network=$GVNIC_NET_1,subnet=$GVNIC_SUB_1,no-address \
      --network-interface=nic-type=MRDMA,network=$RDMA_NET,subnet=$RDMA_SUB_0,no-address \
      --network-interface=nic-type=MRDMA,network=$RDMA_NET,subnet=$RDMA_SUB_1,no-address \
      --network-interface=nic-type=MRDMA,network=$RDMA_NET,subnet=$RDMA_SUB_2,no-address \
      --network-interface=nic-type=MRDMA,network=$RDMA_NET,subnet=$RDMA_SUB_3,no-address \
      --metadata=cp-ip=$CP_IP,join-token=$JOIN_TOKEN,join-hash=$JOIN_HASH \
      --metadata-from-file=startup-script=scripts/tlinux4-k8s134-worker.sh &
  done
  wait  # 等待该 Domain 的 Worker 全部创建完成
done
```

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
