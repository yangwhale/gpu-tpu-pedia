# GB300 (A4X Max) 自建 Kubernetes 端到端部署与 NCCL Benchmark

> **实测验证**: 2026-07-13, `tencent-gcp-taiji-poc` 项目, subblock-0005 + subblock-0006, 16 节点 64 GPU
> **教学文档**: [GB300 RDMA + NVLink 全栈组件详解](https://cc.higcp.com/pages/gb300-rdma-stack-guide-20260713.html)

---

## NCCL Benchmark 结果

### 跨域 MNNVL=2 (NVLink + RDMA 混合)

| Collective | @16G busbw (GB/s) | GB200 参考 | vs GB200 |
|-----------|-------------------|-----------|---------|
| all_reduce | **330** | 330 | 持平 |
| all_gather | **224** | 189 | **+19%** |
| reduce_scatter | **225** | 189 | **+19%** |
| alltoall (优化后) | **98** | 83 | **+18%** |

### 同域 NVLink (MNNVL=2)

| 测试 | GPU | busbw (GB/s) | GB200 |
|------|-----|-------------|-------|
| 单节点 | 4 | 682 | 684 |
| 2n MNNVL | 8 | **841** | 835 |
| 4n MNNVL | 16 | **915** | - |

---

## 前提条件

| 项目 | 说明 |
|------|------|
| GCP 项目 | 需要 GB300 reservation (subblock 级别) |
| OS 镜像 | `rocky-linux-9-optimized-gcp-nvidia-580-arm64-v20260615` from `rocky-linux-accelerator-cloud` |
| K8s | 1.34 (kubeadm), CP 用 x86 `n2-standard-4` |
| 网络 | 管理 VPC (IPv4) + RDMA VPC (IPv6-only, RoCE Metal profile) |
| env-gb300.sh | 集中变量文件 (PROJECT, ZONE, MACHINE_TYPE, RESERVATION 等) |

---

## 1. 环境准备 (一次性)

### 1.1 VPC + 防火墙

```bash
source env-gb300.sh

# 管理 VPC
gcloud compute networks create $MGMT_VPC --subnet-mode=custom --mtu=8896 --project=$PROJECT
gcloud compute networks subnets create $MGMT_SUB_0 --network=$MGMT_VPC --region=$REGION --range=10.150.0.0/24 --project=$PROJECT
gcloud compute networks subnets create $MGMT_SUB_1 --network=$MGMT_VPC --region=$REGION --range=10.150.1.0/24 --project=$PROJECT

# RDMA VPC (子网自动创建, 不需要也不能手动加防火墙)
gcloud compute networks create $RDMA_VPC --network-profile=${ZONE}-vpc-roce-metal --subnet-mode=custom --mtu=8896 --project=$PROJECT

# 管理 VPC 防火墙
gcloud compute firewall-rules create ${MGMT_VPC}-internal --network=$MGMT_VPC --action=ALLOW \
  --rules=tcp:0-65535,udp:0-65535,icmp --source-ranges=10.150.0.0/16,10.100.0.0/24 --project=$PROJECT
```

### 1.2 Control Plane

```bash
gcloud compute instances create chrisya-gb300-cp \
  --machine-type=n2-standard-4 --zone=$ZONE --project=$PROJECT \
  --image=$CP_IMAGE --image-project=$CP_IMAGE_PROJECT \
  --boot-disk-size=200GB \
  --network-interface=network=$MGMT_VPC,subnet=$MGMT_SUB_0 \
  --scopes=cloud-platform
```

CP 初始化:
```bash
sudo kubeadm init --pod-network-cidr=10.244.0.0/16 --node-name=$(hostname)

# Calico (必须指定管理子网 CIDR, 否则选中 RDMA 网卡导致 BGP 失败)
kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.29.3/manifests/tigera-operator.yaml
cat <<EOF | kubectl create -f -
apiVersion: operator.tigera.io/v1
kind: Installation
metadata: { name: default }
spec:
  calicoNetwork:
    nodeAddressAutodetectionV4:
      cidrs: ["10.150.0.0/16"]
    ipPools:
    - { blockSize: 26, cidr: 10.244.0.0/16, encapsulation: VXLANCrossSubnet, natOutgoing: Enabled, nodeSelector: all() }
EOF
```

---

## 2. Worker 节点部署

### 2.1 批量创建 (每域 8 台)

```bash
source env-gb300.sh
JOIN_CMD=$(ssh cp "sudo kubeadm token create --print-join-command")
RDMA_SUB="default-subnet-1-$RDMA_VPC"
DOMAIN=5  # subblock 编号, 按需修改

NAMES=""
for i in $(seq 1 8); do NAMES="$NAMES chrisya-gb300-d${DOMAIN}-w${i}"; done

gcloud compute instances create $NAMES \
  --machine-type=$MACHINE_TYPE --zone=$ZONE --project=$PROJECT \
  --image=$WORKER_IMAGE --image-project=$WORKER_IMAGE_PROJECT \
  --boot-disk-size=50GB \
  --reservation-affinity=specific \
  --reservation="projects/$RESERVATION_PROJECT/reservations/$RESERVATION_NAME/reservationBlocks/$BLOCK/reservationSubBlocks/${BLOCK}-subblock-$(printf '%04d' $DOMAIN)" \
  --provisioning-model=RESERVATION_BOUND \
  --resource-policies=gb300-central-nvl72-policy-$(printf '%04d' $DOMAIN) \
  --maintenance-policy=TERMINATE --restart-on-failure \
  --network-interface="network=$MGMT_VPC,subnet=$MGMT_SUB_0" \
  --network-interface="network=$MGMT_VPC,subnet=$MGMT_SUB_1,no-address" \
  --network-interface="network=$RDMA_VPC,subnet=$RDMA_SUB,nic-type=MRDMA,no-address,stack-type=IPV6_ONLY" \
  --network-interface="network=$RDMA_VPC,subnet=$RDMA_SUB,nic-type=MRDMA,no-address,stack-type=IPV6_ONLY" \
  --network-interface="network=$RDMA_VPC,subnet=$RDMA_SUB,nic-type=MRDMA,no-address,stack-type=IPV6_ONLY" \
  --network-interface="network=$RDMA_VPC,subnet=$RDMA_SUB,nic-type=MRDMA,no-address,stack-type=IPV6_ONLY" \
  --network-interface="network=$RDMA_VPC,subnet=$RDMA_SUB,nic-type=MRDMA,no-address,stack-type=IPV6_ONLY" \
  --network-interface="network=$RDMA_VPC,subnet=$RDMA_SUB,nic-type=MRDMA,no-address,stack-type=IPV6_ONLY" \
  --network-interface="network=$RDMA_VPC,subnet=$RDMA_SUB,nic-type=MRDMA,no-address,stack-type=IPV6_ONLY" \
  --network-interface="network=$RDMA_VPC,subnet=$RDMA_SUB,nic-type=MRDMA,no-address,stack-type=IPV6_ONLY" \
  --metadata="cp-join-cmd=$JOIN_CMD"
```

> 裸金属约 10 分钟启动。创建后需要添加 SSH key: `gcloud compute instances add-metadata <name> --metadata="ssh-keys=chrisya:$(cat ~/.ssh/google_compute_engine.pub)"`

### 2.2 Worker Setup (每台执行)

```bash
# [1] 禁用慢速 repo + 安装基础包
sudo dnf config-manager --set-disabled ciq-sigcloud-next ciq-sigcloud-next-nonfree doca
sudo dnf install -y cloud-utils-growpart

# [2] containerd + nvidia runtime
sudo dnf config-manager --add-repo https://download.docker.com/linux/rhel/docker-ce.repo
sudo dnf config-manager --add-repo https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo
sudo dnf install -y containerd.io nvidia-container-toolkit-1.19.0
sudo nvidia-ctk runtime configure --runtime=containerd

# ⚠️ 必须同时改主 config 和 drop-in
sudo sed -i 's/default_runtime_name = "runc"/default_runtime_name = "nvidia"/' \
  /etc/containerd/config.toml /etc/containerd/conf.d/99-nvidia.toml
sudo systemctl enable --now containerd

# [3] kubeadm
cat | sudo tee /etc/yum.repos.d/kubernetes.repo <<EOF
[kubernetes]
name=Kubernetes
baseurl=https://pkgs.k8s.io/core:/stable:/v1.34/rpm/
enabled=1
gpgcheck=1
gpgkey=https://pkgs.k8s.io/core:/stable:/v1.34/rpm/repodata/repomd.xml.key
EOF
sudo dnf install -y kubelet kubeadm kubectl
sudo systemctl enable kubelet

# [4] GPU 必需模块
sudo modprobe nvidia-peermem
echo nvidia-peermem | sudo tee /etc/modules-load.d/nvidia-peermem.conf
echo 4096 | sudo tee /proc/sys/vm/nr_hugepages
echo vm.nr_hugepages=4096 | sudo tee /etc/sysctl.d/hugepages.conf

# [5] 扩容磁盘
sudo growpart /dev/nvme2n1 2 && sudo xfs_growfs /

# [6] Join K8s
sudo $JOIN_CMD --node-name=$(hostname)
```

### 2.3 ⚠️ NVLink P2P 验证 (创建后必须检查)

```bash
# 每台 Worker 必须验证 — NS 的节点不可用于 NVLink 测试
nvidia-smi topo -p2p r | head -6    # 期望: OK (不是 NS)
nvidia-smi nvlink -s | head -4       # 期望: 53.125 GB/s (不是 Sleep)
nvidia-smi topo -m | head -6         # 期望: NV18 (不是 NODE)
```

**如果看到 NS/Sleep/NODE**: 该物理机的 NVSwitch 未被 GCP 基础设施正确初始化。**删除 VM 重建**（换物理机）或换 subblock。GPU reset 和 reboot 无法修复。

> NVLink 初始化由 NMX-C (NVLink Management Controller) 在 NVSwitch tray 上完成，不是宿主机上的 Fabric Manager。GB300 NVL72 的 NVSwitch 在独立 switch tray 上，计算节点无法直接管理。

### 2.4 GPU Stack (CP 上执行)

```bash
# device-plugin
helm install nvidia-device-plugin nvdp/nvidia-device-plugin -n kube-system

# Label 所有 worker
kubectl label node <worker-name> feature.node.kubernetes.io/pci-10de.present=true

# asapd-lite: 需要手动拉镜像 (私有 registry)
# 在每台 Worker 上:
TOKEN=$(curl -sf -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" | jq -r .access_token)
sudo ctr -n k8s.io image pull --user "_token:$TOKEN" us-docker.pkg.dev/gce-ai-infra/asapd-lite/asapd-lite:v0.0.8
sudo ctr -n k8s.io image pull --user "_token:$TOKEN" us-docker.pkg.dev/gce-ai-infra/netslo-ebpf-manager/ebpf_manager:release
sudo ctr -n k8s.io image pull --user "_token:$TOKEN" us-docker.pkg.dev/gce-ai-infra/gpudirect-gib/nccl-plugin-gib-diagnostic-arm64:v1.1.2
```

### 2.5 IMEX Daemon (同域 NVLink/MNNVL 必需)

```bash
# 每台同域 Worker 执行 (只列同域节点 IP):
sudo mkdir -p /etc/nvidia-imex
cat | sudo tee /etc/nvidia-imex/nodes_config.cfg <<EOF
<同域节点1 IP>
<同域节点2 IP>
...
EOF

sudo nvidia-imex &
nvidia-imex-ctl -N  # 验证: All Connected

# 创建 IMEX channel
MAJOR=$(grep nvidia-caps-imex-channels /proc/devices | awk '{print $1}')
sudo mkdir -p /dev/nvidia-caps-imex-channels
sudo mknod /dev/nvidia-caps-imex-channels/channel0 c $MAJOR 0
sudo chmod 666 /dev/nvidia-caps-imex-channels/channel0
```

---

## 3. NCCL 测试

### 3.1 GIB Pod

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nccl-<worker-short-name>
spec:
  hostNetwork: true
  dnsPolicy: ClusterFirstWithHostNet
  nodeName: <worker-full-name>
  containers:
  - name: gib
    image: us-docker.pkg.dev/gce-ai-infra/gpudirect-gib/nccl-plugin-gib-diagnostic-arm64:v1.1.2
    imagePullPolicy: IfNotPresent
    command: ["bash", "-c", "service ssh restart; echo 'MaxStartups 100:30:200' >> /etc/ssh/sshd_config; service ssh reload; sleep infinity"]
    securityContext: { privileged: true }
    volumeMounts:
    - { name: dev, mountPath: /dev }
    - { name: shm, mountPath: /dev/shm }
    resources: { limits: { nvidia.com/gpu: "4" } }
  volumes:
  - { name: dev, hostPath: { path: /dev } }
  - { name: shm, emptyDir: { medium: Memory, sizeLimit: 128Gi } }
  restartPolicy: Never
```

### 3.2 NCCL 运行命令

```bash
mpirun --allow-run-as-root \
  --mca pml ob1 --mca orte_keep_fqdn_hostnames t \
  --mca btl tcp,self --bind-to none \
  --mca btl_tcp_if_include eth0 --mca oob_tcp_if_include eth0 \
  --mca routed direct --mca plm_rsh_no_tree_spawn 1 \
  -np $NP --hostfile hostfile \
  -x PATH -x LD_LIBRARY_PATH="/usr/local/gib/lib64:$LD_LIBRARY_PATH" \
  -x NCCL_DEBUG=WARN \
  -x NCCL_MNNVL_ENABLE=2 -x NCCL_CUMEM_ENABLE=1 \
  -x NCCL_IB_DATA_DIRECT=0 \
  -x NCCL_TESTS_SPLIT_MASK=0x0 \
  bash -c "source /usr/local/gib/scripts/set_nccl_env.sh; \
           /third_party/nccl-tests/build/all_reduce_perf -b 1M -e 16G -f 2 -w 5 -n 20"
```

**关键环境变量**:

| 变量 | 值 | 为什么 |
|------|-----|------|
| `NCCL_IB_DATA_DIRECT` | **0** | NVIDIA 580 不支持 CX-8 Data Direct DMA |
| `NCCL_IB_GID_INDEX` | **7** | RoCE v2 ipvlan GID (偶数=RoCE v1, 奇数=RoCE v2, c0de 后缀) |
| `NCCL_MNNVL_ENABLE` | **2** | 同域 NVLink + 跨域 RDMA 自动切换 |
| `NCCL_P2P_NET_CHUNKSIZE` | **4194304** | AllToAll 优化 (默认 256K 太小) |
| `NCCL_NCHANNELS_PER_NET_PEER` | **4** | AllToAll 多 channel 并行 (默认 1) |

> P2P 优化参数只影响 AllToAll，对 AllReduce/AllGather/ReduceScatter 无影响。

---

## 4. 踩坑速查

| 问题 | 原因 | 修复 |
|------|------|------|
| **NVLink P2P=NS, Sleep** | NVSwitch 未初始化 (NMX-C 层) | **删 VM 重建** (换物理机) 或换 subblock |
| device-plugin NVML not found | containerd runtime 是 runc | 改 config.toml **和** conf.d/99-nvidia.toml 的 `default_runtime_name` |
| Cuda failure 800 (MNNVL) | IMEX channel 不存在 | `mknod /dev/nvidia-caps-imex-channels/channel0` |
| MPI 8+ 节点路由失败 | 树形路由不稳定 | `--mca routed direct --mca plm_rsh_no_tree_spawn 1` |
| AllToAll 只有 43 GB/s | NCHANNELS_PER_NET_PEER=1 | 设 4 + P2P_NET_CHUNKSIZE=4M |
| mlx5dv_reg_dmabuf_mr error 524 | NVIDIA 580 不支持 Data Direct DMA | `NCCL_IB_DATA_DIRECT=0` |
| Pod Evicted | 磁盘太小 | `--boot-disk-size=50GB` |
| dnf 超时 | 无外网或慢 repo | 加外网 IP + 禁用 ciq/doca repo |
| subblock 资源不足 | ZONE_RESOURCE_POOL_EXHAUSTED | 换其他 subblock |
| Fabric Manager NV_WARN_NOTHING_TO_DO | GB300 NVSwitch 在 switch tray 上 | FM 不适用于 NVL72 计算节点 |
| **RDMA 跨节点 RETRY_EXC_ERR** | mlx5_7 firmware internal error | `NCCL_IB_HCA=mlx5_0,...,mlx5_6` 排除坏 NIC |
| RDMA 用 PF GID 失败 | RoCE v1 GID 不可路由 | `NCCL_IB_GID_INDEX=7` (RoCE v2 ipvlan) |
| 同节点跨 CX-8 port RDMA 不通 | RoCE Metal 只路由跨节点流量 | 正常，MNNVL=2 同节点走 NVLink |

---

## 5. 参考

- [GCP: 在 VM 上跑 NCCL (非 GKE)](https://docs.google.com/ai-hypercomputer/docs/nccl/test-vms)
- [GCP: Slurm 集群部署 A4X Max](https://docs.google.com/cluster-toolkit/docs/deploy/slurm/create-a4x-max-cluster)
- [GCP: Cluster Health Scanner](https://docs.google.com/ai-hypercomputer/docs/troubleshooting/test-clusters)
- [NVIDIA: NVLink Management (NMX)](https://docs.nvidia.com/mission-control/docs/systems-administration-guide/2.0.0/high-speed-fabric-management.html)
- [NCCL Issue #890: GID 自动检测](https://github.com/NVIDIA/nccl/issues/890)
- [教学文档: GB300 全栈组件详解](https://cc.higcp.com/pages/gb300-rdma-stack-guide-20260713.html)

---

*基于 2026-07-12~13 三轮端到端实测验证 · GCP GPU Infrastructure Team*
