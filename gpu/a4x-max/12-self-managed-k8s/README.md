# GB300 (A4X Max) 自建 Kubernetes 部署与验证指南

**平台**：NVIDIA GB300 (A4X Max) · ARM64 Grace · NVL72 · Bare Metal

**Kubernetes**：1.34.9 (kubeadm) + Calico v3.29.3

**OS**：Rocky Linux 9.8（CP: x86_64 `rocky-linux-cloud`, Worker: ARM64 `rocky-linux-accelerator-cloud` + NVIDIA 580）

**GPU 管理**：nvidia-device-plugin + DRA GPU Driver v25.8.0 (ComputeDomain + IMEX)

**RDMA**：DRANET v1.3.0 + 8×CX-8 SuperNIC PF · RoCEv2 over IPv6 · GIB v1.1.2

> **实测验证**：2026-07-12，`tencent-gcp-taiji-poc` 项目，subblock-0003，2 节点 8 GPU。

---

## 测试结果

| 测试 | GPU | 互联 | busbw @8G (GB/s) | vs GB200 |
|------|-----|------|------------------|----------|
| 单节点 4GPU | 4 | NVLink | **681.97** | 684 (-0.3%) |
| 同域 2节点 MNNVL | 8 | NVLink+GIB | **838.08** | 835 (+0.4%) |

---

## 0. 架构概述

### 网络架构

```
管理 VPC (IPv4, MTU 8896)          RDMA VPC (IPv6-only, RoCE Metal)
┌──────────────────────┐           ┌──────────────────────────┐
│ sub-0: 10.150.0.0/24 │           │ fd36::/48 (自动子网)      │
│   CP:  10.150.0.2    │           │   8×MRDMA per Worker     │
│   W1:  10.150.0.3    │           │   GIB GPUDirect RDMA     │
│   W2:  10.150.0.4    │           │   RoCEv2 over IPv6       │
│ sub-1: 10.150.1.0/24 │           └──────────────────────────┘
│   Worker nic1 备用    │
└──────────────────────┘
         │ VPC Peering
    cc-tw (10.100.0.0/24)
```

### 组件版本

| 组件 | 版本 | 来源 |
|------|------|------|
| K8s | 1.34.9 | pkgs.k8s.io |
| Calico | v3.29.3 | tigera-operator |
| nvidia-device-plugin | latest | `nvidia.github.io/k8s-device-plugin` |
| DRA GPU Driver | 25.8.0 | NGC Helm (`nvidia/nvidia-dra-driver-gpu`) |
| DRANET | v1.3.0 | `oci://registry.k8s.io/networking/charts/dranet` |
| GIB | v1.1.2 | `us-docker.pkg.dev/gce-ai-infra/gpudirect-gib/nccl-plugin-gib-diagnostic-arm64` |
| containerd | 2.2.6 | Docker CE repo |
| nvidia-container-toolkit | latest | `nvidia.github.io/libnvidia-container` |
| GPU Driver | 580.159.03 | 镜像预装 |
| CUDA | 13.0 | 镜像预装 |
| NCCL | 2.30.4 | GIB 内置 |

### GB300 vs GB200 关键差异

| 维度 | GB200 (A4X) | GB300 (A4X Max) |
|------|-------------|-----------------|
| 机型 | `a4x-highgpu-4g` (VM) | `a4x-maxgpu-4g-metal` (Bare Metal) |
| GPU | 4× B200 (186 GB) | 4× B300 Ultra (**278 GB** HBM3e) |
| 管理网 | GVNIC, IPv4 | **IDPF**, IPv4 |
| RDMA | 4× CX-7 VF, IPv4 | **8× CX-8 PF**, **IPv6-only** |
| RDMA VPC | 4 独立子网 | **1 共享子网** (RoCE Metal profile) |
| RDMA 带宽 | 2,000 Gbps | **3,200 Gbps** |
| DRA GPU Driver | v0.4.0 (OCI) | **v25.8.0** (NGC Helm) |
| NCCL IB_ADDR_FAMILY | 默认 (IPv4) | **AF_INET6** |
| 启动时间 | ~2 min | **~10 min** (裸金属 PCIe 枚举) |

---

## 1. 环境准备

### 1.1 管理 VPC (IPv4)

```bash
PROJECT=tencent-gcp-taiji-poc
REGION=us-central1

gcloud compute networks create chrisya-gb300-mgmt-v2 \
  --subnet-mode=custom --mtu=8896 --project=$PROJECT

gcloud compute networks subnets create chrisya-gb300-mgmt-sub-0 \
  --network=chrisya-gb300-mgmt-v2 --region=$REGION \
  --range=10.150.0.0/24 --project=$PROJECT

gcloud compute networks subnets create chrisya-gb300-mgmt-sub-1 \
  --network=chrisya-gb300-mgmt-v2 --region=$REGION \
  --range=10.150.1.0/24 --project=$PROJECT
```

### 1.2 RDMA VPC (IPv6-only, RoCE Metal)

```bash
gcloud compute networks create chrisya-gb300-rdma-v2 \
  --network-profile=us-central1-b-vpc-roce-metal \
  --subnet-mode=custom --mtu=8896 --project=$PROJECT
# 自动创建 fd36::/48 子网，不需要手动建
# RoCE Metal VPC 不允许手动加防火墙规则（内置安全）
```

### 1.3 防火墙

```bash
# 管理 VPC 内部全通 + cc-tw 内网
gcloud compute firewall-rules create chrisya-gb300-mgmt-v2-internal \
  --network=chrisya-gb300-mgmt-v2 --action=ALLOW \
  --rules=tcp:0-65535,udp:0-65535,icmp \
  --source-ranges=10.150.0.0/16,10.100.0.0/24 \
  --project=$PROJECT

# SSH from cc-tw 外网 IP
gcloud compute firewall-rules create chrisya-gb300-mgmt-v2-ssh-cctw \
  --network=chrisya-gb300-mgmt-v2 --action=ALLOW \
  --rules=tcp:22 --source-ranges=34.80.76.71/32 \
  --project=$PROJECT

# IAP SSH
gcloud compute firewall-rules create chrisya-gb300-mgmt-v2-iap \
  --network=chrisya-gb300-mgmt-v2 --action=ALLOW \
  --rules=tcp:22 --source-ranges=35.235.240.0/20 \
  --project=$PROJECT
```

> **网段规划**: 10.150.0.0/16 避开 cc-tw 的 10.2-14/10.20/10.40/10.60/10.100-101 和 gb300-central 的 10.200。

### 1.4 VPC Peering (cc-tw 内网直连)

```bash
# cc-tw 侧 (gpu-launchpad-playground)
gcloud compute networks peerings create cctw-to-gb300-mgmt \
  --network=chrisya-gvnic-net-0 \
  --peer-project=tencent-gcp-taiji-poc \
  --peer-network=chrisya-gb300-mgmt-v2 \
  --project=gpu-launchpad-playground

# GB300 侧 (tencent-gcp-taiji-poc)
gcloud compute networks peerings create gb300-mgmt-to-cctw \
  --network=chrisya-gb300-mgmt-v2 \
  --peer-project=gpu-launchpad-playground \
  --peer-network=chrisya-gvnic-net-0 \
  --project=tencent-gcp-taiji-poc
```

> Peering 建好后 `ssh cp` 直接通过内网 10.150.0.2 连接 Master。延迟 ~150ms（跨太平洋）。

### 1.5 SSH config

```
Host cp
    HostName 10.150.0.2
    User chrisya
    IdentityFile ~/.ssh/google_compute_engine
    StrictHostKeyChecking accept-new
    IdentitiesOnly=yes
    CheckHostIP=no

Host gb1
    HostName 10.150.0.3
    ...

Host gb2
    HostName 10.150.0.4
    ...
```

### 1.6 项目级 SSH Key

```bash
gcloud compute project-info add-metadata \
  --project=tencent-gcp-taiji-poc \
  --metadata=ssh-keys="chrisya:$(cat ~/.ssh/google_compute_engine.pub)"
```

> 项目级 SSH key 让所有新建 VM 自动注入公钥，无需逐个配置。

---

## 2. Control Plane 节点

### 镜像

`rocky-linux-9-v20260615` from `rocky-linux-cloud` (x86_64)

### 创建

```bash
gcloud compute instances create chrisya-gb300-cp \
  --project=$PROJECT --zone=us-central1-b \
  --machine-type=n2-standard-4 \
  --image=rocky-linux-9-v20260615 --image-project=rocky-linux-cloud \
  --boot-disk-size=200GB --boot-disk-type=pd-balanced \
  --network-interface=network=chrisya-gb300-mgmt-v2,subnet=chrisya-gb300-mgmt-sub-0 \
  --scopes=cloud-platform \
  --metadata-from-file=startup-script=cp-startup-rocky.sh
```

### CP Startup Script 内容

自动安装（不做 `kubeadm init`）：

1. swap off + SELinux permissive
2. 内核模块 (`overlay`, `br_netfilter`) + sysctl (`ip_forward`)
3. containerd (Docker CE RHEL repo, SystemdCgroup=true)
4. kubeadm/kubelet/kubectl 1.34 (pkgs.k8s.io)
5. Helm

### kubeadm init (手动)

```bash
ssh cp
sudo kubeadm init --pod-network-cidr=10.244.0.0/16 --node-name=$(hostname)

mkdir -p $HOME/.kube
sudo cp /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config
```

### Calico CNI

```bash
kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.29.3/manifests/tigera-operator.yaml
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
      - "10.150.0.0/16"
    ipPools:
    - blockSize: 26
      cidr: 10.244.0.0/16
      encapsulation: VXLANCrossSubnet
      natOutgoing: Enabled
      nodeSelector: all()
EOF
```

> **Calico 多网卡陷阱**: GB300 Worker 有 10 个 NIC (2 IDPF + 8 MRDMA)。`nodeAddressAutodetectionV4.cidrs` 必须精确指定管理子网 CIDR，否则选中 RDMA 网卡导致 BGP 失败。

### 本地 kubectl

```bash
scp cp:/etc/kubernetes/admin.conf ~/.kube/gb300-config
sed -i 's|https://.*:6443|https://10.150.0.2:6443|' ~/.kube/gb300-config
# .bashrc
alias kb='KUBECONFIG=~/.kube/gb300-config kubectl'
```

---

## 3. Worker 节点

### 镜像

```
rocky-linux-9-optimized-gcp-nvidia-580-arm64-v20260615
project: rocky-linux-accelerator-cloud
```

> **关键**: 必须用 `rocky-linux-accelerator-cloud` 项目，不是 `rocky-linux-cloud`。预装 NVIDIA driver 580.159.03 + CUDA 13.0 + google-guest-agent。

### 创建

```bash
RDMA_SUB="default-subnet-1-chrisya-gb300-rdma-v2"
SUBBLOCK="nvidia-gb300-dxkhoz4ypk4mh-block-0001-subblock-0003"
POLICY="gb300-central-nvl72-policy-0003"
JOIN_CMD="kubeadm join 10.150.0.2:6443 --token ... --discovery-token-ca-cert-hash sha256:..."

gcloud compute instances create chrisya-gb300-d3-w1 \
  --machine-type=a4x-maxgpu-4g-metal \
  --zone=us-central1-b --project=$PROJECT \
  --image=rocky-linux-9-optimized-gcp-nvidia-580-arm64-v20260615 \
  --image-project=rocky-linux-accelerator-cloud \
  --boot-disk-size=1000GB --boot-disk-type=hyperdisk-balanced \
  --network-interface=network=chrisya-gb300-mgmt-v2,subnet=chrisya-gb300-mgmt-sub-0 \
  --network-interface=network=chrisya-gb300-mgmt-v2,subnet=chrisya-gb300-mgmt-sub-1,no-address \
  --network-interface=subnet=$RDMA_SUB,stack-type=IPV6_ONLY,nic-type=MRDMA \
  --network-interface=subnet=$RDMA_SUB,stack-type=IPV6_ONLY,nic-type=MRDMA \
  --network-interface=subnet=$RDMA_SUB,stack-type=IPV6_ONLY,nic-type=MRDMA \
  --network-interface=subnet=$RDMA_SUB,stack-type=IPV6_ONLY,nic-type=MRDMA \
  --network-interface=subnet=$RDMA_SUB,stack-type=IPV6_ONLY,nic-type=MRDMA \
  --network-interface=subnet=$RDMA_SUB,stack-type=IPV6_ONLY,nic-type=MRDMA \
  --network-interface=subnet=$RDMA_SUB,stack-type=IPV6_ONLY,nic-type=MRDMA \
  --network-interface=subnet=$RDMA_SUB,stack-type=IPV6_ONLY,nic-type=MRDMA \
  --reservation-affinity=specific \
  --reservation=projects/tencent-gcp-taiji/reservations/nvidia-gb300-dxkhoz4ypk4mh/reservationBlocks/nvidia-gb300-dxkhoz4ypk4mh-block-0001/reservationSubBlocks/$SUBBLOCK \
  --provisioning-model=RESERVATION_BOUND \
  --resource-policies=$POLICY \
  --maintenance-policy=TERMINATE --restart-on-failure \
  --scopes=cloud-platform \
  --metadata-from-file=startup-script=worker-startup-rocky.sh \
  --metadata=cp-join-cmd="$JOIN_CMD"
```

> **Reservation 绑定**: `--reservation` 指定到 subblock 级别，`--resource-policies` 对应编号的 placement policy。  
> **启动时间**: 裸金属约 10 分钟（PCIe 枚举 4 GPU + 8 CX-8 + NVSwitch）。

### Worker Startup Script 内容

| Phase | 内容 | 备注 |
|-------|------|------|
| [1] Base | swap off, SELinux off, sysctl, memlock unlimited | |
| [2] containerd | Docker CE repo, SystemdCgroup=true | |
| [3] nvidia-ctk | nvidia-container-toolkit, `--set-as-default` | **必须设为默认 runtime** |
| [4] kubeadm | pkgs.k8s.io, kubelet/kubeadm/kubectl 1.34 | |
| [5] nvidia-peermem | `modprobe nvidia-peermem` | **GPUDirect RDMA 必需** |
| [6] Hugepages | `echo 4096 > /proc/sys/vm/nr_hugepages` | RDMA 注册用 |
| [7-8] GPU/RDMA verify | nvidia-smi, ls /sys/class/infiniband/ | |
| [9] kubeadm join | 从 metadata `cp-join-cmd` 读取 | 自动 join |

**nvidia-container-toolkit 关键配置**:

```bash
nvidia-ctk runtime configure --runtime=containerd --set-as-default
# 还需要手动改主 config:
sed -i 's/default_runtime_name = "runc"/default_runtime_name = "nvidia"/' /etc/containerd/config.toml
systemctl restart containerd
```

> 必须把 nvidia 设为默认 containerd runtime。否则 nvidia-device-plugin 找不到 NVML (ERROR_LIBRARY_NOT_FOUND)。`nvidia-ctk` 写到 drop-in 文件 (`/etc/containerd/conf.d/99-nvidia.toml`)，但主 config 的 `default_runtime_name` 必须也改。

### Worker join 后配置

```bash
# GPU 节点 label
kubectl label node $WORKER feature.node.kubernetes.io/pci-10de.present=true

# 手动创建 IMEX channels (ComputeDomain daemon 需要)
ssh $WORKER "
  MAJOR=\$(grep nvidia-caps-imex-channels /proc/devices | awk '{print \$1}')
  sudo mkdir -p /dev/nvidia-caps-imex-channels
  for i in \$(seq 0 255); do
    sudo mknod /dev/nvidia-caps-imex-channels/channel\$i c \$MAJOR \$i 2>/dev/null
  done
  sudo chmod 666 /dev/nvidia-caps-imex-channels/channel*
"
```

### Worker 验证清单

```bash
ssh gb1

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
# 预期: NVIDIA GB300, 580.159.03, 284208 MiB  (×4)

lsmod | grep nvidia_peermem
# 预期: nvidia_peermem  16384  0

ls /sys/class/infiniband/ | wc -l
# 预期: 8

grep HugePages_Total /proc/meminfo
# 预期: HugePages_Total:    4096

ls /dev/nvidia-caps-imex-channels/ | wc -l
# 预期: 256
```

### 实测结果

| 检查项 | Worker 1 | Worker 2 |
|--------|----------|----------|
| GPU | 4× GB300, 284208 MiB, 580.159.03 | 同 |
| nvidia-peermem | ✅ loaded | ✅ loaded |
| RDMA | 8× mlx5 | 8× mlx5 |
| Hugepages | 4096 × 2MB | 4096 × 2MB |
| IMEX channels | 256 | 256 |
| K8s join | Ready | Ready |

---

## 4. GPU Stack 安装

在 CP 上执行 (`ssh cp`):

### 4.1 nvidia-device-plugin

```bash
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin && helm repo update
helm install nvidia-device-plugin nvdp/nvidia-device-plugin \
  --namespace kube-system --wait --timeout 300s

# 验证
kubectl get nodes -o custom-columns='NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu'
# 每个 Worker 应显示 4
```

### 4.2 DRA GPU Driver v25.8.0 (ComputeDomain)

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia && helm repo update

kubectl create ns nvidia-dra-driver-gpu

helm install nvidia-dra-driver-gpu nvidia/nvidia-dra-driver-gpu \
  --version="25.8.0" \
  --namespace nvidia-dra-driver-gpu \
  --set nvidiaDriverRoot=/ \
  --set resources.gpus.enabled=false \
  --set controller.affinity=null \
  --set controller.priorityClassName='' \
  --set kubeletPlugin.priorityClassName='' \
  --set 'kubeletPlugin.tolerations[0].operator=Exists' \
  --wait --timeout 300s
```

> **`nvidiaDriverRoot=/`**: Rocky Linux 驱动在系统标准路径 `/usr/lib64/`。GKE COS 用 `/home/kubernetes/bin/nvidia`。

### 4.3 Scheduler RBAC 补全

kubeadm 1.34 的 scheduler ClusterRole 缺少 DRA 权限，必须补全。

```bash
kubectl apply -f scheduler-rbac.yaml
kubectl delete pod -n kube-system -l component=kube-scheduler
```

RBAC 内容：需要给 `system:kube-scheduler` 添加 `resource.k8s.io` 的 resourceclaims/resourceslices/deviceclasses 权限。完整 YAML 见 [a4x/03-gpu-stack](../../a4x/03-gpu-stack/) 3.5 节。

### 4.4 DRANET v1.3.0 + RDMA DeviceClass

```bash
helm install dranet oci://registry.k8s.io/networking/charts/dranet \
  --version v1.3.0 --namespace kube-system --wait

# RDMA DeviceClass (必须加 rdma == true 过滤)
cat <<EOF | kubectl apply -f -
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
EOF
```

> **`rdma == true` 必须加**: 不加会分配到 Calico vxlan 接口导致 `network is unreachable`。

### 4.5 ComputeDomain

```bash
# 创建 ComputeDomain
cat <<EOF | kubectl apply -f -
apiVersion: resource.nvidia.com/v1beta1
kind: ComputeDomain
metadata:
  name: gb300-cd
spec:
  numNodes: 0
  channel:
    resourceClaimTemplate:
      name: gb300-cd-channel
EOF

# 给 Worker 打 ComputeDomain label
CD_UID=$(kubectl get computedomain gb300-cd -o jsonpath='{.metadata.uid}')
kubectl label node chrisya-gb300-d3-w1 resource.nvidia.com/computeDomain=$CD_UID --overwrite
kubectl label node chrisya-gb300-d3-w2 resource.nvidia.com/computeDomain=$CD_UID --overwrite

# 验证 IMEX daemon pods 启动
kubectl get pods -n nvidia-dra-driver-gpu | grep gb300-cd
# 每个 Worker 应有一个 daemon pod (0/1 Running 正常,在找其他 16 个不存在的 peer)
```

### 4.6 RDMA ResourceClaimTemplate

```bash
cat <<EOF | kubectl apply -f -
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
          deviceClassName: rdma-devices
          allocationMode: ExactCount
          count: 8
EOF
```

---

## 5. NCCL 测试

### 5.1 NCCL Pod 模板

**关键要点**:
- 使用 **GIB 诊断镜像** (`nccl-plugin-gib-diagnostic-arm64:v1.1.2`)
- **不用 hostNetwork** — DRANET 通过 DRA claims 注入 RDMA 设备
- DRA claims 同时要 **rdma-nics** 和 **compute-domain-channel**
- `LD_LIBRARY_PATH` 包含 `/usr/local/gib/lib64` **和** `/usr/local/nvidia/lib64`
- `privileged: true` — GIB GPUDirect RDMA 需要

```yaml
containers:
- image: us-docker.pkg.dev/gce-ai-infra/gpudirect-gib/nccl-plugin-gib-diagnostic-arm64:v1.1.2
  securityContext:
    privileged: true
  env:
  - name: LD_LIBRARY_PATH
    value: "/usr/local/gib/lib64:/usr/local/nvidia/lib64"
  - name: NCCL_MNNVL_ENABLE
    value: "2"
  - name: NCCL_CUMEM_ENABLE
    value: "1"
  - name: NCCL_IB_GID_INDEX
    value: "3"
  resources:
    limits:
      nvidia.com/gpu: 4
    claims:
    - name: rdma-nics
    - name: compute-domain-channel
  command: ["/bin/bash", "-c"]
  args:
  - |
    source /usr/local/gib/scripts/set_nccl_env.sh 2>/dev/null || true
    mkdir -p /run/sshd
    sed -i 's/^#\?Port .*/Port 222/' /etc/ssh/sshd_config
    /usr/sbin/sshd
    sleep 7200
resourceClaims:
- name: rdma-nics
  resourceClaimTemplateName: all-mrdma
- name: compute-domain-channel
  resourceClaimTemplateName: gb300-cd-channel
```

### 5.2 单节点测试 (NVLink baseline)

```bash
kubectl exec nccl-w0 -- bash -c \
  "all_reduce_perf -b 1M -e 8G -f 2 -g 4 -n 20"
```

**结果**: busbw **681.97 GB/s** @8G (NVLink, 单节点)

### 5.3 同域 2 节点 MNNVL 测试

```bash
kubectl exec nccl-w0 -- bash -c '
source /usr/local/gib/scripts/set_nccl_env.sh 2>/dev/null || true
W0_IP=<pod0_ip>
W1_IP=<pod1_ip>

/usr/local/gib/scripts/init_ssh.sh -p 222 $W0_IP $W1_IP
/usr/local/gib/scripts/gen_hostfiles.sh -p 222 $W0_IP $W1_IP

/usr/local/gib/scripts/run_nccl_tests.sh \
  -t allreduce -b 1M -e 8G -f 2 -p 222 -g 4 \
  $W0_IP $W1_IP
'
```

> **注意**: 使用 Pod IP 而非 hostname（非 hostNetwork 的 Pod DNS 解析可能不及时）。

**结果**:

```
     1048576   float sum    30.29   60.58 GB/s
     8388608   float sum    75.61  194.16 GB/s
    67108864   float sum   279.97  419.47 GB/s
   268435456   float sum   722.93  649.80 GB/s
  1073741824   float sum  2617.08  718.00 GB/s
  4294967296   float sum  9039.72  831.46 GB/s
  8589934592   float sum 17936.80  838.08 GB/s
```

峰值 busbw **838.08 GB/s** (MNNVL, 2 节点 8 GPU)

### 5.4 NCCL 环境变量 (nccl.a4xmax.conf)

GIB `set_nccl_env.sh` + GCP env plugin 自动设置：

| 变量 | 值 | 说明 |
|------|-----|------|
| NCCL_NET | gIB | GIB GPUDirect RDMA 插件 |
| NCCL_IB_ADDR_FAMILY | AF_INET6 | **GB300 关键** — RoCEv2 走 IPv6 |
| NCCL_IB_GID_INDEX | 3 | RoCEv2 GID (fd36 ULA) |
| NCCL_IB_TC | 52 | DSCP 流量标记 |
| NCCL_IB_FIFO_TC | 84 | FIFO 流量标记 |
| NCCL_IB_QPS_PER_CONNECTION | 4 | 每连接 QP 数 |
| NCCL_IB_ADAPTIVE_ROUTING | 1 | 自适应路由 |
| NCCL_IB_MERGE_VFS | 1 | 合并 VF |
| NCCL_PXN_C2C | 1 | PCIe 跨节点 relay |
| NCCL_CUMEM_ENABLE | 1 | CUDA Memory Manager |
| NCCL_MNNVL_ENABLE | 2 | NVLink 自动检测 |

---

## 6. 踩坑记录

### 镜像选择

| 镜像 | 结果 | 原因 |
|------|------|------|
| TLinux v5dot4 | ❌ 30min 不启动 | v5dot4 镜像可能有 bug |
| TLinux v5dot3 | ❌ sshd 不监听 VPC | sshd 绑定 Tailscale 接口 |
| Ubuntu Accelerator ARM64 | ❌ sshd 不启动 | 默认不装 openssh-server |
| **Rocky Linux 9 ARM64 NVIDIA 580** | ✅ | 预装 driver + guest-agent + sshd |

### 网络

| 问题 | 原因 | 修复 |
|------|------|------|
| gLinux SSH 不通 | 跨组织 IAP 权限 | VPC peering + cc-tw 内网直连 |
| VPC peering 后 ping 不通 | 防火墙不放行 cc-tw 内网 IP | 加 10.100.0.0/24 到 source-ranges |
| hostNetwork + DRA claims 冲突 | NRI 拒绝 host 设备 claim | 不用 hostNetwork，走 DRA 注入 |
| Pod DNS 解析失败 | 非 hostNetwork Pod 的 DNS 延迟 | 用 Pod IP 代替 hostname |

### GPU Stack

| 问题 | 原因 | 修复 |
|------|------|------|
| device-plugin NVML not found | containerd 默认 runtime 是 runc | 改 `default_runtime_name = "nvidia"` + 手动 sed 主 config |
| GKE 上 nvidia-peermem 加载失败 | COS 内核不支持 ib_peer_memory | 自建 K8s + Rocky Linux 直接 modprobe |
| GIB Cuda failure 4 | 用 PyTorch 镜像但 GIB 不在 | 必须用 GIB 诊断镜像 |
| GIB 路径为空 | 未正确使用 GIB 诊断镜像 | 确认 image 是 `nccl-plugin-gib-diagnostic-arm64:v1.1.2` |
| IMEX daemon 0/1 Ready | 只有 2 节点但 daemon 找 18 个 peer | 正常行为，不影响功能 |

### 裸金属

| 问题 | 原因 | 修复 |
|------|------|------|
| 启动 ~10min | PCIe 枚举 4 GPU + 8 NIC + NVSwitch | 正常，等就行 |
| Serial console 无后续输出 | ARM64 console 重定向 | 用 SSH 验证，不依赖 serial |
| subblock-0001 Internal Error | 硬件/配额问题 | 换 subblock-0003 |

---

## 7. 对比汇总

### GB300 vs GB200 NCCL Benchmark

| 测试 | 传输 | GPU | GB300 (busbw) | GB200 (busbw) | 差异 |
|------|------|-----|---------------|---------------|------|
| 单节点 @8G | NVLink | 4 | 681.97 | 683.75 | -0.3% |
| 同域 2n MNNVL @8G | NVLink | 8 | **838.08** | **834.95** | **+0.4%** |
| 跨域 2n RDMA all_reduce @1G | RDMA (8 NIC) | 8 | **316.20** | ~330* | **-4%** |
| 跨域 2n RDMA all_gather @1G | RDMA (8 NIC) | 8 | **220.29** | ~189* | **+17%** |
| 跨域 2n RDMA reduce_scatter @1G | RDMA (8 NIC) | 8 | **219.06** | ~189* | **+16%** |
| 跨域 2n RDMA alltoall @1G | RDMA (8 NIC) | 8 | 42.59 | ~83* | -49% |
| 跨域 2n TCP Socket @256M | TCP | 8 | 3.67 | - | - |

**结论**:
- NVLink 5 代同速，GB300 和 GB200 的 NVSwitch 带宽一致 (<1% 差异)
- **跨域 RDMA all_reduce 316 GB/s busbw（全 8 NIC，subblock-0001 ↔ subblock-0004，接近 GKE 330 GB/s 参考值）**
- all_gather/reduce_scatter 220/219 GB/s，超过 GKE 参考值 189 GB/s
- 参考值来自 GKE 内部 benchmark (Maxwell Xi 2026-06-14)
- 跨域 alltoall 无层级优化，仅 42 GB/s — MoE EP 组必须控制在同域内
- TCP Socket 仅 3.67 GB/s，证明 RDMA 带来 67x 加速

---

## 8. 跨域 RDMA 调试关键发现

### 根因：RDMA 必须走 ipvlan GID

GB300 的 CX-8 RDMA NIC 每个物理端口有两类 GID：

| GID Index | 类型 | 地址 | 网络设备 | RDMA |
|-----------|------|------|----------|------|
| 0 | IB/RoCE v1 | fe80::... (link-local) | gpu0rdma0 (PF) | ❌ |
| 1 | RoCE v2 | fe80::... (link-local) | gpu0rdma0 (PF) | ❌ |
| 2 | IB/RoCE v1 | fd36::...0:0 (ULA) | gpu0rdma0 (PF) | ❌ |
| 3 | RoCE v2 | fd36::...0:0 (ULA) | gpu0rdma0 (PF) | ❌ |
| 4 | IB/RoCE v1 | fe80::... | gpu0ipvlan0 | ❌ |
| 5 | RoCE v2 | fe80::... | gpu0ipvlan0 | ❌ |
| 6 | IB/RoCE v1 | fd36::...c0de:0 (ipvlan) | gpu0ipvlan0 | ❌ |
| **7** | **RoCE v2** | **fd36::...c0de:0 (ipvlan)** | **gpu0ipvlan0** | **✅** |

**只有 GID index 7**（RoCE v2 + ipvlan + fd36 ULA `:c0de:` 地址）能通 RDMA。
PF 的 GID (index 0-3) 全部 retry exceeded 失败。

### 关键环境变量

```bash
# 跨域 RDMA NCCL 必需
export NCCL_IB_GID_INDEX=7      # 使用 ipvlan 的 RoCE v2 GID
export NCCL_IB_DATA_DIRECT=0    # 禁用 Data Direct DMA (NVIDIA 580 不支持)
export NCCL_MNNVL_ENABLE=0      # 强制 RDMA（跨域自动路由用 =2）
```

### Data Direct DMA (error 524)

GIB v1.1.2 检测到 CX-8 后会尝试 Data Direct DMA 路径。NVIDIA 580 驱动 + Rocky 9 kernel 5.14 下 `mlx5dv_reg_dmabuf_mr` 返回 error 524 (ENOTSUPP)。必须设 `NCCL_IB_DATA_DIRECT=0` 退回 nvidia-peermem 路径。

### Subblock-0003 gpu1 Rail 异常

ibv_rc_pingpong 跨域测试中 mlx5_2/mlx5_3 (gpu1rdma0/gpu1rdma1, BDF 0002:03:00.x) 在 subblock-0003 的两台不同物理机上（d3-w1 和 d3-w3）均 retry exceeded。d4-w1 (subblock-0004) 同一设备正常。疑似 subblock-0003 的 gpu1 rail 交换机/线缆问题。

用 `NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_6,mlx5_7` 排除后正常运行（6/8 NIC = 2400 Gbps，全 8 NIC 应为 3200 Gbps）。

### GID Index 因机器而异

不同机器的 ipvlan fd36 GID index 不同（d4-w1 = 7, d3-w3 = 9），因此不能硬编码 `NCCL_IB_GID_INDEX`。需要自动检测：

```bash
# 自动检测 ipvlan RoCE v2 fd36 GID index
for i in $(seq 0 15); do
  type=$(cat /sys/class/infiniband/mlx5_0/ports/1/gid_attrs/types/$i 2>/dev/null)
  ndev=$(cat /sys/class/infiniband/mlx5_0/ports/1/gid_attrs/ndevs/$i 2>/dev/null)
  gid=$(cat /sys/class/infiniband/mlx5_0/ports/1/gids/$i 2>/dev/null)
  if [ "$type" = "RoCE v2" ] && echo "$ndev" | grep -q ipvlan && echo "$gid" | grep -q fd36; then
    export NCCL_IB_GID_INDEX=$i; break
  fi
done
```

### 595 驱动升级路径

NVIDIA SBSA 官方 repo 有 595.71.05 for Rocky 9 ARM64：

```bash
dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/sbsa/cuda-rhel9.repo
# 先卸载 CIQ nvidia-dc 包，再装 NVIDIA 官方包
dnf remove 'nvidia-dc-*' 'kmod-nvidia-dc-*'
dnf install nvidia-open --allowerasing
```

GCP 已测试验证 R595 for A4X Max。580→595 可能解锁 Data Direct DMA 性能。

---

*文档基于 2026-07-12 端到端实测验证 · GCP GPU Infrastructure Team*

---

## 9. 大规模测试结果 (2026-07-13)

### 集群配置
- 16 节点: d1×8 (subblock-0001) + d4×8 (subblock-0004)
- 64 GPU (16×4 B300 Ultra)
- Rocky Linux 9 + NVIDIA 580.159.03

### 9.1 同域 8 节点 32 GPU RDMA (MNNVL=0)

| Collective | @8G busbw (GB/s) | vs GB200 NVLink |
|-----------|-----------------|-----------------|
| all_reduce | **170** | 839 (NVLink) |

> 注: 这是纯 RDMA 结果，不含 NVLink。MNNVL=2 需要 IMEX channel (ComputeDomain DRA)，尚未在 hostNetwork 模式下配通。

### 关键发现: 多节点 GID Index 不一致

不同 mlx5 设备的 ipvlan GID index 不同（mlx5_0=9, mlx5_1~7=7），单一 `NCCL_IB_GID_INDEX` 无法覆盖所有设备。

**临时方案**: `NCCL_IB_HCA=^mlx5_0` 排除 GID 不一致的设备，用 7/8 NIC。
**正确方案**: 使用 DRA/DRANET 分配 RDMA NIC（pod 网络命名空间只看到 ipvlan，无 PF GID 干扰）。
**另一方案**: 删除 PF fd36 地址（未验证是否影响 GID 缓存）。

### 9.2 跨域 8 节点 32 GPU RDMA (4 d1 + 4 d4, MNNVL=0)

| Collective | @8G busbw (GB/s) | vs 2-node 跨域 |
|-----------|-----------------|----------------|
| all_reduce | **173** | 170 (同域 8n) |

### 9.3 跨域 4 节点 16 GPU RDMA (2 d1 + 2 d4, MNNVL=0)

| Collective | @8G busbw (GB/s) | vs 2-node |
|-----------|-----------------|-----------|
| all_reduce | **167** | 316 (2n) |

### 9.4 MPI 16 节点限制

hostNetwork 模式下 MPI 16 节点启动失败（orted 路由错误），可能与 GIB 容器内 MPI 的 OOB 通信限制有关。8 节点（单域或跨域）均正常。正式大规模测试应使用 DRA/DRANET Pod 模式。

### GB300 RDMA vs GB200 NVLink 完整对比

| 测试 | GPU | 传输 | GB300 busbw | GB200 busbw | 差异 |
|------|-----|------|------------|------------|------|
| 单节点 | 4 | NVLink | 682 | 684 | -0.3% |
| 2n 同域 MNNVL | 8 | NVLink | **838** | **835** | +0.4% |
| 2n 跨域 RDMA | 8 | RDMA (8 NIC) | **316** | ~330* | -4% |
| 8n 同域 RDMA | 32 | RDMA (7 NIC) | **170** | - | - |
| 8n 跨域 RDMA | 32 | RDMA (7 NIC) | **173** | - | - |
| 4n 跨域 RDMA | 16 | RDMA (7 NIC) | 167 | - | - |

> *GB200 参考值来自 Maxwell Xi 内部 benchmark (GKE, 2026-06-14)
> 8n+ 测试使用 7/8 NIC (排除 mlx5_0 因 GID index 不一致)

### 9.5 IMEX Channel 手动配置 (MNNVL NVLink 前提)

MNNVL 需要 IMEX daemon + channel 设备。hostNetwork pod 不能用 DRA ComputeDomain，必须手动配置。

**步骤 1: 创建 nodes_config.cfg (列出同域所有节点 IP)**
```bash
cat > /etc/nvidia-imex/nodes_config.cfg << EOF
10.150.0.12
10.150.0.13
... (每行一个节点 IP)
EOF
```

**步骤 2: 启动 IMEX daemon**
```bash
nvidia-imex &
nvidia-imex-ctl -N  # 验证所有节点 Connected
```

**步骤 3: 创建 IMEX channel0 设备**
```bash
MAJOR=$(grep nvidia-caps-imex-channels /proc/devices | awk '{print $1}')
mkdir -p /dev/nvidia-caps-imex-channels
mknod /dev/nvidia-caps-imex-channels/channel0 c $MAJOR 0
chmod 666 /dev/nvidia-caps-imex-channels/channel0
```

**步骤 4: MPI 8 节点路由修复**
```bash
mpirun ... --mca routed direct --mca plm_rsh_no_tree_spawn 1 ...
```
SSH MaxStartups 需增大到 `100:30:200`。

### 9.6 同域 4 节点 16 GPU NVLink MNNVL=2

| Collective | @16G busbw (GB/s) | vs 2n MNNVL | vs GB200 |
|-----------|-------------------|-------------|----------|
| all_reduce | **915** | 838 (2n) | - |

> 4 节点 NVLink 比 2 节点更高！NVSwitch 全互联在更多节点下有更多并行传输路径。

### 9.7 跨域 8 节点 32 GPU 全 Collective (MNNVL=0 RDMA)

| Collective | @8G busbw (GB/s) | vs 2n 跨域 |
|-----------|-----------------|------------|
| all_reduce | **173** | 316 (2n 8NIC) |
| all_gather | **192** | 220 (2n) |
| reduce_scatter | **193** | 219 (2n) |
| alltoall | **52** | 43 (2n) |

> 8 节点跨域 RDMA 的 alltoall 52 GB/s 比 2 节点 43 GB/s 更高，因为更多路径并行。
> all_reduce/all_gather/reduce_scatter 因节点数增加导致通信跳数增多，带宽稍有下降。

### 9.8 8 节点 32 GPU MNNVL=2 (NVLink) 限制

8 节点 MNNVL=2 在 hostNetwork 模式下不稳定：
- kubectl exec TLS 超时（长时间 NCCL 初始化）
- Pod 容器状态异常（IMEX 操作可能导致 GPU 状态异常触发 containerd 重启）
- 正确方案：使用 DRA ComputeDomain 管理 IMEX 生命周期 + Pod 级别资源隔离

**已验证**: 4 节点 16 GPU MNNVL 915 GB/s all_reduce@16G，NVLink 5 正常工作。
**待验证**: 8+ 节点需要完整 DRA 集成。

### 总结对比表（完整）

| 测试 | GPU | 传输 | 节点配置 | busbw@max (GB/s) | GB200 参考 |
|------|-----|------|---------|-----------------|-----------|
| 单节点 | 4 | NVLink | 1n | 682 | 684 |
| 2n 同域 MNNVL | 8 | NVLink | d1×2 | **838** | 835 |
| 4n 同域 MNNVL | 16 | NVLink | d1×4 | **915** | - |
| 2n 跨域 RDMA | 8 | RDMA 8NIC | d1+d4 | **316** | ~330 |
| 8n 同域 RDMA | 32 | RDMA 7NIC | d1×8 | 170 | - |
| 8n 跨域 RDMA | 32 | RDMA 7NIC | d1×4+d4×4 | 173 | - |
| 8n 跨域 all_gather | 32 | RDMA 7NIC | d1×4+d4×4 | 192 | - |
| 8n 跨域 reduce_scatter | 32 | RDMA 7NIC | d1×4+d4×4 | 193 | - |
| 8n 跨域 alltoall | 32 | RDMA 7NIC | d1×4+d4×4 | 52 | - |
| 2n TCP Socket | 8 | TCP | d1+d4 | 3.67 | - |

### 9.9 跨域 2 节点 8 GPU MNNVL=2 (NVLink + RDMA 混合)

| Collective | @16G busbw (GB/s) | GKE 参考 | 差异 |
|-----------|-------------------|---------|------|
| all_reduce | **331** | ~330 | **+0.3%** |

> **关键突破**: MNNVL=2 跨域测试完美匹配 GKE 参考值！
> 同域流量走 NVLink (915+ GB/s)，跨域走 RDMA (7/8 NIC = 2800 Gbps)
> NCCL 层级优化自动在两种传输间切换

### 9.10 跨域 2n MNNVL=2 全 Collective

| Collective | @16G busbw (GB/s) | GKE 参考 | MNNVL=0 纯RDMA |
|-----------|-------------------|---------|---------------|
| all_reduce | **331** | ~330 | 316 |
| all_gather | **193** | ~189 | 220 |
| reduce_scatter | **193** | ~189 | 219 |
| alltoall | **43** | ~83 | 43 |

> MNNVL=2 在 all_reduce 上比纯 RDMA 高 5%（331 vs 316），因为 NVLink 辅助了 hierarchical reduction
> all_gather/reduce_scatter 与 GKE 参考值一致
> alltoall 无 hierarchical 优化，与纯 RDMA 相同

### 9.11 最终优化结果：全 8 NIC + GID 自动检测

**关键修复**: 不设 `NCCL_IB_GID_INDEX` 和 `NCCL_IB_HCA`，让 GIB env plugin + NCCL 2.30.4 自动检测每个设备的正确 GID（NCCL >= 2.21 支持）。之前手动设 `NCCL_IB_GID_INDEX=7` 导致 mlx5_0（GID 在 index 9）被排除。

| Collective | @16G busbw | MNNVL=0 (RDMA) | MNNVL=2 (混合) | GB200 参考 | vs GB200 |
|-----------|-----------|----------------|---------------|-----------|---------|
| all_reduce | | 329 | **329** | 330 | -0.3% |
| all_gather | | **224** | **224** | 189 | **+19%** |
| reduce_scatter | | **225** | **224** | 189 | **+19%** |
| alltoall | | 43 | 43 | 83 | -48% |

**分析**:
- **all_gather/reduce_scatter +19%**: CX-8 8×400G (3200 Gbps) vs CX-7 4×400G (1600 Gbps) 的带宽优势在这两个 bandwidth-sensitive collective 上体现
- **all_reduce 持平**: hierarchical algorithm 瓶颈不在 RDMA 带宽，同域 NVLink local reduce 后跨域只传 1 份数据
- **alltoall -48%**: NCCL alltoall 实现未做多 rail 优化，GB300 的 8 NIC 没有被充分利用
- 全 8 NIC auto-detect 比之前 7 NIC 手动设置 all_gather 提升 16% (193→224)

### GID 自动检测说明

NCCL >= 2.21 引入 `NCCL_IB_GID_INDEX=-1`（默认值），自动扫描每个设备的 GID 表选择正确的 RoCE v2 GID。配合 `NCCL_IB_ADDR_FAMILY=AF_INET6`（GIB env plugin 自动设），无需手动指定 GID index。

参考: [NCCL Issue #890](https://github.com/NVIDIA/nccl/issues/890), [NCCL Issue #1333](https://github.com/NVIDIA/nccl/issues/1333)

### 9.12 AllToAll 优化: P2P_NET_CHUNKSIZE + NCHANNELS_PER_NET_PEER

默认 `P2P_NET_CHUNKSIZE=512K` 导致 AllToAll 只有 43 GB/s。调大后显著提升：

| 配置 | AllToAll @16G busbw | 提升 |
|------|-------------------|------|
| 默认 (512K chunk, 1 channel) | 43 GB/s | baseline |
| **4M chunk + 4 channels** | **98 GB/s** | **+128%** |
| 8M chunk + 8 channels | 98 GB/s | 同上 |

> `NCCL_P2P_NET_CHUNKSIZE=4194304` (4MB) 减少 chunk 开销
> `NCCL_NCHANNELS_PER_NET_PEER=4` 每对 GPU 用 4 channel 并行传输
> GB300 优化后 98 GB/s vs GB200 83 GB/s → **+18%**
