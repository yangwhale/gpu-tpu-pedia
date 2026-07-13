# GB300 (A4X Max) 自建 Kubernetes 端到端部署与 NCCL Benchmark

> **实测验证**: 2026-07-13, `tencent-gcp-taiji-poc` 项目, subblock-0001 + subblock-0004, 16 节点 64 GPU

---

## NCCL Benchmark 结果

### 跨域 2n 8GPU (subblock-0001 ↔ subblock-0004, MNNVL=2)

| Collective | @16G busbw (GB/s) | GB200 参考 | vs GB200 |
|-----------|-------------------|-----------|---------|
| all_reduce | **330** | 330 | 持平 |
| all_gather | **224** | 189 | **+19%** |
| reduce_scatter | **225** | 189 | **+19%** |
| alltoall (优化后) | **98** | 83 | **+18%** |

### 同域 NVLink (subblock-0001, MNNVL=2)

| 测试 | GPU | busbw (GB/s) | GB200 |
|------|-----|-------------|-------|
| 单节点 | 4 | 682 | 684 |
| 2n MNNVL | 8 | **838** | 835 |
| 4n MNNVL | 16 | **915** | - |

> GB200 参考值来自 Maxwell Xi 内部 benchmark (GKE, GIB v1.1.2, 2026-06-14)

---

## 硬件规格

| 维度 | GB200 (A4X) | GB300 (A4X Max) |
|------|-------------|-----------------|
| 机型 | `a4x-highgpu-4g` (VM) | `a4x-maxgpu-4g-metal` (Bare Metal) |
| GPU | 4× B200 (186 GB HBM3e) | 4× B300 Ultra (**278 GB** HBM3e) |
| 管理网 | GVNIC IPv4 | IDPF IPv4 |
| RDMA | 4× CX-7 VF, 1600 Gbps | **8× CX-8 PF, 3200 Gbps** |
| RDMA VPC | 4 独立子网, IPv4 | 1 共享子网, **IPv6-only** (RoCE Metal) |
| NVLink 域 | 18 节点 72 GPU | 18 节点 72 GPU |
| 启动时间 | ~2 min | ~10 min (裸金属 PCIe 枚举) |

---

## 1. 环境准备

### 1.1 VPC 网络

```bash
source env-gb300.sh  # PROJECT, REGION 等变量

# 管理 VPC (IPv4, 两个子网)
gcloud compute networks create $MGMT_VPC --subnet-mode=custom --mtu=8896 --project=$PROJECT
gcloud compute networks subnets create $MGMT_SUB_0 --network=$MGMT_VPC --region=$REGION --range=10.150.0.0/24 --project=$PROJECT
gcloud compute networks subnets create $MGMT_SUB_1 --network=$MGMT_VPC --region=$REGION --range=10.150.1.0/24 --project=$PROJECT

# RDMA VPC (IPv6-only, RoCE Metal profile, 子网自动创建)
gcloud compute networks create $RDMA_VPC --network-profile=us-central1-b-vpc-roce-metal --subnet-mode=custom --mtu=8896 --project=$PROJECT

# 防火墙 (管理 VPC 内部全通)
gcloud compute firewall-rules create ${MGMT_VPC}-internal --network=$MGMT_VPC --action=ALLOW \
  --rules=tcp:0-65535,udp:0-65535,icmp --source-ranges=10.150.0.0/16,10.100.0.0/24 --project=$PROJECT
```

### 1.2 Control Plane

```bash
gcloud compute instances create chrisya-gb300-cp \
  --machine-type=n2-standard-4 --zone=$ZONE --project=$PROJECT \
  --image=rocky-linux-9-v20260615 --image-project=rocky-linux-cloud \
  --boot-disk-size=200GB \
  --network-interface=network=$MGMT_VPC,subnet=$MGMT_SUB_0 \
  --scopes=cloud-platform
```

CP 上执行:
```bash
# 安装 containerd + kubeadm (详见 env-gb300.sh)
sudo kubeadm init --pod-network-cidr=10.244.0.0/16 --node-name=$(hostname)

# Calico CNI (必须指定管理子网 CIDR, 避免选中 RDMA 网卡)
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

### 1.3 Worker 节点 (批量创建)

```bash
JOIN_CMD=$(ssh cp "sudo kubeadm token create --print-join-command")
RDMA_SUB="default-subnet-1-$RDMA_VPC"

# 同域 8 节点 (subblock-0001)
for i in $(seq 1 8); do
  NAMES="$NAMES chrisya-gb300-d1-w${i}"
done

gcloud compute instances create $NAMES \
  --machine-type=$MACHINE_TYPE --zone=$ZONE --project=$PROJECT \
  --image=$WORKER_IMAGE --image-project=$WORKER_IMAGE_PROJECT \
  --boot-disk-size=50GB \
  --reservation-affinity=specific \
  --reservation="projects/$RESERVATION_PROJECT/reservations/$RESERVATION_NAME/reservationBlocks/$BLOCK/reservationSubBlocks/${BLOCK}-subblock-0001" \
  --provisioning-model=RESERVATION_BOUND \
  --resource-policies=gb300-central-nvl72-policy-0001 \
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

> 裸金属约 10 分钟启动。需要为每个实例添加 SSH key (`gcloud compute instances add-metadata`)。

### 1.4 Worker Setup (每台执行)

```bash
# 禁用慢速 repo
sudo dnf config-manager --set-disabled ciq-sigcloud-next ciq-sigcloud-next-nonfree doca

# containerd + nvidia-container-toolkit (必须 1.19.0 版本)
sudo dnf install -y containerd.io nvidia-container-toolkit-1.19.0
sudo nvidia-ctk runtime configure --runtime=containerd
sudo sed -i 's/default_runtime_name = "runc"/default_runtime_name = "nvidia"/' /etc/containerd/config.toml
sudo systemctl enable --now containerd

# kubeadm
sudo dnf install -y kubelet kubeadm kubectl
sudo systemctl enable kubelet

# GPU 必需模块
sudo modprobe nvidia-peermem
echo nvidia-peermem | sudo tee /etc/modules-load.d/nvidia-peermem.conf
echo 4096 | sudo tee /proc/sys/vm/nr_hugepages
echo vm.nr_hugepages=4096 | sudo tee /etc/sysctl.d/hugepages.conf

# 扩容磁盘
sudo growpart /dev/nvme2n1 2 && sudo xfs_growfs /

# Join K8s
sudo $JOIN_CMD --node-name=$(hostname)

# GPU 节点 label (device-plugin 需要)
# 在 CP 上: kubectl label node <name> feature.node.kubernetes.io/pci-10de.present=true
```

**关键**: `default_runtime_name` 必须改成 `nvidia`。`nvidia-ctk` 只写 drop-in 文件，主 config 需要 `sed` 手改。否则 device-plugin 报 `NVML ERROR_LIBRARY_NOT_FOUND`。

### 1.5 GPU Stack

```bash
# nvidia-device-plugin
helm install nvidia-device-plugin nvdp/nvidia-device-plugin -n kube-system

# DRA GPU Driver (ComputeDomain + IMEX)
helm install nvidia-dra-driver-gpu nvidia/nvidia-dra-driver-gpu --version=25.8.0 \
  -n nvidia-dra-driver-gpu --create-namespace \
  --set nvidiaDriverRoot=/ --set resources.gpus.enabled=false

# DRANET (RDMA NIC 分配)
helm install dranet oci://registry.k8s.io/networking/charts/dranet --version v1.3.0 -n kube-system

# asapd-lite (RDMA NIC 配置: MTU, ECN, PFC, DCQCN)
# 由 DaemonSet 自动调度, 需手动拉镜像:
sudo ctr -n k8s.io image pull --user "_token:$(gcloud auth print-access-token)" \
  us-docker.pkg.dev/gce-ai-infra/asapd-lite/asapd-lite:v0.0.8
```

### 1.6 IMEX Daemon (MNNVL NVLink 必需)

ComputeDomain DRA 和手动 IMEX **不能共存**。hostNetwork 模式用手动方式:

```bash
# 删除 DRA 管理的 ComputeDomain
kubectl delete computedomain gb300-cd

# 每台 Worker 执行:
# 1. 创建 nodes_config.cfg (同域所有节点 IP)
cat > /etc/nvidia-imex/nodes_config.cfg << EOF
10.150.0.12
10.150.0.13
... (每行一个 IP)
EOF

# 2. 启动 IMEX daemon
nvidia-imex &
nvidia-imex-ctl -N  # 验证 All Connected

# 3. 创建 IMEX channel 设备
MAJOR=$(grep nvidia-caps-imex-channels /proc/devices | awk '{print $1}')
mkdir -p /dev/nvidia-caps-imex-channels
mknod /dev/nvidia-caps-imex-channels/channel0 c $MAJOR 0
chmod 666 /dev/nvidia-caps-imex-channels/channel0
```

---

## 2. NCCL 测试

### 2.1 GIB 诊断 Pod (hostNetwork 模式)

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nccl-test
spec:
  hostNetwork: true
  dnsPolicy: ClusterFirstWithHostNet
  nodeName: <worker-name>
  containers:
  - name: gib
    image: us-docker.pkg.dev/gce-ai-infra/gpudirect-gib/nccl-plugin-gib-diagnostic-arm64:v1.1.2
    imagePullPolicy: IfNotPresent
    command: ["bash", "-c", "service ssh restart; sleep infinity"]
    securityContext:
      privileged: true
    volumeMounts:
    - { name: dev, mountPath: /dev }
    - { name: shm, mountPath: /dev/shm }
    resources:
      limits:
        nvidia.com/gpu: "4"
  volumes:
  - { name: dev, hostPath: { path: /dev } }
  - { name: shm, emptyDir: { medium: Memory, sizeLimit: 128Gi } }
  restartPolicy: Never
```

### 2.2 NCCL 环境变量

GIB env plugin (`NCCL_ENV_PLUGIN=gcp`) 从 `nccl.a4xmax.conf` 自动加载:

| 变量 | 值 | 说明 |
|------|-----|------|
| NCCL_NET | gIB | GIB GPUDirect RDMA 插件 |
| NCCL_IB_ADDR_FAMILY | AF_INET6 | GB300 IPv6 RDMA |
| NCCL_IB_DATA_DIRECT | **0** | **必须关闭** — NVIDIA 580 不支持 CX-8 Data Direct DMA |
| NCCL_IB_GID_INDEX | **不设** | **让 NCCL >= 2.21 自动检测** (不同设备 GID index 不同) |
| NCCL_MNNVL_ENABLE | 2 | 同域 NVLink + 跨域 RDMA 自动混合 |
| NCCL_CUMEM_ENABLE | 1 | MNNVL 必需 |

**AllToAll 优化** (P2P 通信模式专用):

| 变量 | 默认值 | 优化值 | 影响 |
|------|-------|-------|------|
| NCCL_P2P_NET_CHUNKSIZE | 262144 | **4194304** | AllToAll +128% |
| NCCL_NCHANNELS_PER_NET_PEER | 1 | **4** | 多 channel 并行 |

> `NCHANNELS_PER_NET_PEER=1` 是 `nccl.a4xmax.conf` 写死的。对 AllReduce/AllGather/ReduceScatter 无影响，但严重限制 AllToAll。

### 2.3 MPI 运行参数

```bash
mpirun --allow-run-as-root \
  --mca pml ob1 --mca orte_keep_fqdn_hostnames t \
  --mca btl tcp,self --bind-to none \
  --mca btl_tcp_if_include eth0 --mca oob_tcp_if_include eth0 \
  --mca routed direct --mca plm_rsh_no_tree_spawn 1 \  # 8+ 节点必需
  -np $NP --hostfile hostfile \
  -x PATH -x LD_LIBRARY_PATH="/usr/local/gib/lib64:$LD_LIBRARY_PATH" \
  -x NCCL_DEBUG=WARN -x NCCL_MNNVL_ENABLE=2 -x NCCL_CUMEM_ENABLE=1 \
  -x NCCL_IB_DATA_DIRECT=0 -x NCCL_TESTS_SPLIT_MASK=0x0 \
  bash -c "source /usr/local/gib/scripts/set_nccl_env.sh; \
           /third_party/nccl-tests/build/all_reduce_perf -b 1M -e 16G -f 2 -w 5 -n 20"
```

**MPI 关键参数**:
- `--mca routed direct`: 8+ 节点必须，否则 `ORTE does not know how to route`
- `--mca plm_rsh_no_tree_spawn 1`: 线性 SSH spawn，避免树形路由故障
- SSH `MaxStartups 100:30:200`: 容器和宿主机都需要增大

---

## 3. 关键发现

### 3.1 RDMA 必须走 ipvlan GID

GB300 CX-8 NIC 有两类 GID:
- PF GID (fd36::...0:0): RDMA **不通** — `ibv_rc_pingpong` retry exceeded
- **ipvlan GID (fd36::...c0de:0): RDMA 通** — asapd-lite 创建的 ipvlan 接口

不同 mlx5 设备的 ipvlan GID index 不同 (mlx5_0=9, mlx5_1~7=7)。**不能硬编码 `NCCL_IB_GID_INDEX`**，必须让 NCCL >= 2.21 自动检测 (`NCCL_IB_GID_INDEX=-1`, 默认值)。

参考: [NCCL Issue #890](https://github.com/NVIDIA/nccl/issues/890)

### 3.2 Data Direct DMA 不可用

GIB v1.1.2 + CX-8 触发 `mlx5dv_reg_dmabuf_mr` error 524 (ENOTSUPP)。NVIDIA 580 驱动不支持 CX-8 Data Direct DMA。必须 `NCCL_IB_DATA_DIRECT=0` 退回 nvidia-peermem 路径。595 驱动可能解锁此特性。

### 3.3 AllToAll 需要 P2P 参数调优

`nccl.a4xmax.conf` 设 `NCHANNELS_PER_NET_PEER=1`，适合 AllReduce (ring/tree) 但限制 AllToAll (P2P sendrecv)。设 `NCCL_P2P_NET_CHUNKSIZE=4M` + `NCCL_NCHANNELS_PER_NET_PEER=4` 后 AllToAll 从 43→98 GB/s (+128%)。

### 3.4 IMEX 手动配置

MNNVL NVLink 需要 IMEX daemon + channel 设备。hostNetwork pod 不能用 DRA ComputeDomain (两者冲突: `NV_ERR_IN_USE`)。必须:
1. 删除 ComputeDomain CRD
2. 手动启动 `nvidia-imex` + `mknod channel0`
3. 跨域节点需要各自独立的 IMEX domain (nodes_config.cfg 只列同域节点)

---

## 4. 踩坑速查

| 问题 | 原因 | 修复 |
|------|------|------|
| device-plugin NVML not found | containerd 默认 runtime 是 runc | `sed` 改 config.toml `default_runtime_name="nvidia"` |
| ibv_rc_pingpong retry exceeded | 用了 PF GID，不是 ipvlan GID | 不设 NCCL_IB_GID_INDEX，让自动检测 |
| Cuda failure 800 (MNNVL) | IMEX channel 不存在 | `mknod /dev/nvidia-caps-imex-channels/channel0` |
| MPI 8+ 节点路由失败 | 默认树形路由不稳定 | `--mca routed direct --mca plm_rsh_no_tree_spawn 1` |
| AllToAll 只有 43 GB/s | NCHANNELS_PER_NET_PEER=1 | 设为 4 + P2P_NET_CHUNKSIZE=4M |
| mlx5dv_reg_dmabuf_mr error 524 | NVIDIA 580 不支持 Data Direct DMA | `NCCL_IB_DATA_DIRECT=0` |
| Rocky 10 RDMA 不通 | CX-8 没分配 fd36 IPv6 地址 | 用 Rocky 9 |
| Pod Evicted | boot-disk-size 20GB 太小 | 创建时用 `--boot-disk-size=50GB` |
| dnf 超时 | 无外网 IP 或慢速 repo | 加外网 IP + 禁用 ciq/doca repo |

---

*基于 2026-07-12~13 端到端实测验证 · GCP GPU Infrastructure Team*
