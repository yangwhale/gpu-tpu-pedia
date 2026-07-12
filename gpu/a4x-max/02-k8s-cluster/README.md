# GB300 (A4X Max) 自建 K8s 集群

自建 Kubernetes 集群（非 GKE 托管），用于 GB300 NVL72 Bare Metal 节点。

> GKE 托管集群方案见 [11-gke-setup/](../11-gke-setup/)。

## 与 GB200 的核心差异

| 维度 | GB200 (A4X) | GB300 (A4X Max) |
|------|-------------|-----------------|
| 节点类型 | VM | Bare Metal |
| kubeadm join | IPv4 `CP_IP:6443` | **IPv6** `[fd20::x]:6443` |
| Calico CNI | IPv4 Pod CIDR | **IPv6 或双栈** |
| 网卡 | GVNIC (enp0s3) | **IDPF** (设备名待确认) |
| K8s 版本 | 1.34+ | **1.34.3+** (官方推荐 1.35+) |
| 容器运行时 | containerd | containerd (不变) |

## 自建 K8s 步骤

### Step 1: Control Plane 节点

使用普通 VM（非 GPU 节点）作为 control plane。

```bash
# 创建 CP 节点 (n2-standard-4, 同 VPC)
gcloud compute instances create gb300-k8s-master \
  --machine-type=n2-standard-4 --zone=$ZONE --project=$PROJECT \
  --network-interface=nic-type=GVNIC,network=$PREFIX-idpf-net,subnet=$PREFIX-idpf-sub-mgmt \
  --boot-disk-size=200GB --boot-disk-type=hyperdisk-balanced
```

### Step 2: kubeadm init

```bash
# IPv6 环境下 kubeadm init
kubeadm init \
  --apiserver-advertise-address=${CP_IPV6} \
  --pod-network-cidr=fd00:10:244::/48 \
  --service-cidr=fd00:10:96::/112 \
  --node-name=$(hostname)
```

> **IPv6 注意**: GB300 全栈 IPv6，kubeadm 的地址参数需要用 IPv6 格式。如果 CP 节点在 IPv4 子网上，需要配置双栈或用 IPv4 管理子网。

### Step 3: Calico CNI

```bash
# IPv6 模式 Calico
kubectl apply -f https://raw.githubusercontent.com/projectcalico/calico/v3.28/manifests/calico.yaml

# 配置 IPv6 IP 池
kubectl apply -f - <<EOF
apiVersion: crd.projectcalico.org/v1
kind: IPPool
metadata:
  name: default-ipv6-ippool
spec:
  cidr: fd00:10:244::/48
  ipipMode: Never
  vxlanMode: CrossSubnet
  natOutgoing: true
  nodeSelector: all()
EOF
```

### Step 4: Worker 节点加入

GB300 Bare Metal 节点通过 startup script 自动 kubeadm join（见 [01-environment-setup](../01-environment-setup/)）。

```bash
# 在 GB300 节点上执行
kubeadm join [${CP_IPV6}]:6443 \
  --token ${JOIN_TOKEN} \
  --discovery-token-ca-cert-hash sha256:${JOIN_HASH} \
  --node-name ${NODE_NAME}
```

### Step 5: 节点标签

```bash
# GPU 节点标签
kubectl label node $NODE nvidia.com/gpu=present
kubectl label node $NODE cloud.google.com/gke-accelerator=nvidia-gb300
```

## GB200 参考

GB200 自建 K8s 集群文档: [a4x/02-k8s-cluster/](../../a4x/02-k8s-cluster/)

## GB300 实测记录

| 步骤 | 状态 | 备注 |
|------|------|------|
| CP 节点创建 | — | |
| kubeadm init | — | |
| Calico 部署 | — | |
| Worker join | — | |
| GPU 标签 | — | |
