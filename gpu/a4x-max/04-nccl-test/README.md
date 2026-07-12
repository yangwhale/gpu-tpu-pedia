# 4. NCCL 通信测试（GB300 / A4X Max）

本章覆盖 6 个层级的 NCCL 通信测试：单节点 NVLink、同域 2 节点 MNNVL、跨域 2 节点 RDMA、混合 4 节点、全域 18 节点 72 GPU、跨域 36 节点 144 GPU。每项测试包含 GB200 基线数据、GB300 预期参考值和我方验证预留栏。

> **官方文档**: [Run NCCL on custom GKE clusters that use A4X Max](https://docs.cloud.google.com/ai-hypercomputer/docs/nccl/test-gke-custom-a4x-max)

## GB300 vs GB200 通信硬件对比

| 维度 | GB200 (A4X) | GB300 (A4X Max) | 预期影响 |
|------|-------------|-----------------|----------|
| 机器类型 | a4x-highgpu-4g | a4x-maxgpu-4g-metal (裸金属) | 消除 hypervisor 开销 |
| GPU | 4x NVIDIA B200 | 4x NVIDIA GB300 (B300 Ultra) | GPU 型号变更 |
| GPU 显存 | 744 GB (186 GB/GPU) | 1,112 GB (278 GB/GPU, HBM3e) | +50% |
| RDMA 网卡 | CX-7 VF (SR-IOV, 挂 CPU) | CX-8 SuperNIC PF (直连 GPU, GPUDirect) | 延迟更低 |
| RDMA 接口数 | 4 (单端口) | 8 (CX-8 双端口, 8-way rail) | 聚合 RDMA 带宽翻倍 |
| 网络带宽 | 2,000 Gbps | 3,200 Gbps | +60% |
| 网络栈 | IPv4 | IPv6-only | 全栈变更 |
| NVLink 域 | 18 节点 = 72 GPU | 18 节点 = 72 GPU | 不变 |
| NCCL 版本 | 2.30.4 | 待确认 (>= 2.28.9) | — |
| GIB 插件 | GIB (GPUDirect-TCPX 演进) | GIB (ARM64 版) | 同 |
| RDMA 配置 | RDMA installer DaemonSet | asapd-lite DaemonSet | 配置方式变更 |
| DRA Driver | v0.4.0 | v25.8.0 | Helm chart 安装 |
| ResourceClaim RDMA count | 4 | 8 | Pod YAML 变更 |

## 环境准备

### GKE 集群信息

```bash
# 连接到 GB300 GKE 集群
gcloud container clusters get-credentials chrisya-gb300-gke \
  --location=us-central1 \
  --project=tencent-gcp-taiji-poc

# 验证节点就绪
kubectl get nodes -o wide
```

### 前置检查

```bash
# 1. 确认 asapd-lite DaemonSet 就绪（MRDMA NIC 配置）
kubectl get daemonset -n kube-system asapd-lite
# 预期: READY 数 = 节点数

# 2. 确认 DRA driver 就绪
kubectl get pods -n nvidia-dra-driver-gpu
# 预期: controller + kubelet-plugin 全部 Running

# 3. 确认 ComputeDomain CRD 可用
kubectl get crd computedomains.resource.nvidia.com
```

### NCCL 环境变量

GIB 诊断镜像内置 `set_nccl_env.sh` 脚本自动设置最优 NCCL 参数。关键变量：

| 变量 | 推荐值 | GB200 对比 | 说明 |
|---|---|---|---|
| NCCL_NET | gIB | 相同 | 启用 GPUDirect RDMA (GIB) |
| NCCL_MNNVL_ENABLE | 2（同域）/ 0（跨域） | 相同 | 2=自动检测 MNNVL，0=强制禁用 |
| NCCL_CUMEM_ENABLE | 1 | 相同 | 启用 CUDA Memory Manager |
| NCCL_IB_GID_INDEX | 3 | 相同 | RoCEv2 GID index |
| NCCL_IB_ADAPTIVE_ROUTING | 1 | 相同 | 启用自适应路由 |
| NCCL_IB_QPS_PER_CONNECTION | 4 | 相同 | 每连接 QP 数 |
| NCCL_IB_TC | 52 | 相同 | Traffic Class（DSCP 标记） |
| NCCL_PXN_C2C | 1 | 相同 | 启用 PCIe 跨节点 NVLink relay |
| NCCL_NVLS_ENABLE | 0 | 相同 | 关闭 NVLink SHARP |
| NCCL_IB_MERGE_NICS | 0 | 相同 | 不合并 NIC（每 GPU 独立 NIC） |

> **GB300 差异点**: RDMA 接口从 4 增加到 8，但 NCCL 环境变量本身不需要改——GIB 会自动发现所有可用的 MRDMA 接口。变更体现在 Pod YAML 的 ResourceClaimTemplate 中 (`count: 8`)。

### ResourceClaimTemplate（GB300 专用）

GB300 使用 DRANET 分配 RDMA 设备，与 GB200 的 RDMA installer DaemonSet 不同：

```yaml
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
          count: 8    # GB200 是 4，GB300 是 8
```

### Pod 声明模式（GB300 专用）

GB300 Pod 必须声明以下资源（与 GB200 的关键差异已标注）：

```yaml
# Pod 通过 DRA 声明两类资源：
#   1. compute-domain-channel → ComputeDomain 自动生成的 ResourceClaimTemplate（提供 IMEX channel）
#   2. rdma                   → DRANET ResourceClaimTemplate（提供 8 张 MRDMA 网卡，GB200 是 4 张）
#
# Pod spec 关键字段：
spec:
  affinity:
    nodeAffinity:                          # GB300 ARM64 节点亲和
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: kubernetes.io/arch
            operator: In
            values:
            - arm64
  volumes:
    - name: library-dir-host
      hostPath:
        path: /home/kubernetes/bin/nvidia  # GB300 COS 驱动路径
  containers:
    - name: nccl-test
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
      resourceClaimTemplateName: a4x-max-compute-domain-channel
    - name: rdma
      resourceClaimTemplateName: all-mrdma
```

---

## GB200 Baseline（基线数据）

以下为 GB200 (A4X) 全量 NCCL 实测结果，作为 GB300 测试的对比基线。数据来自 [a4x/04-nccl-test](../../a4x/04-nccl-test/)。

### 单域 MNNVL 基线

| 测试配置 | GPU | all_reduce | all_gather | reduce_scatter | alltoall | 来源 |
|---|---|---|---|---|---|---|
| 单节点 4GPU @8G | 4 | 683 | — | — | — | 我方验证 |
| 同域 2n @16G | 8 | **842** | 683 | 693 | 683 | 我方验证 |
| 同域 8n @16G | 32 | **909** | 691 | 708 | 676 | 我方验证 |
| 同域 16n @16G | 64 | **910** | 693 | 707 | 667 | 我方验证 |
| 全域 18n @16G | 72 | **877** (v1) | 674 (v1) | 697 (v1) | 627 (v1) | v1 镜像实测 |

### 跨域 RDMA 基线

| 测试配置 | GPU | all_reduce | all_gather | reduce_scatter | alltoall | 来源 |
|---|---|---|---|---|---|---|
| 跨域 2n @8G | 8 | **326** | — | — | — | 我方验证 |
| 跨域 4n @16G | 16 | **691** | 378 | 379 | 88 | 我方验证 |
| 跨域 16n @16G | 64 | **798** | 688 | 702 | 83 | 我方验证 |
| 跨域 36n @16G | 144 | **748** (v1) | 674 (v1) | 691 (v1) | 65 (v1) | v1 镜像实测 |

单位：GB/s busbw

### GB300 预期性能

基于硬件差异的理论预估：

| 场景 | GB200 基线 | GB300 预期 | 理由 |
|---|---|---|---|
| 同域 NVLink | ~910 | ~910 或更高 | NVSwitch fabric 架构相同，B300 NVLink 带宽待确认 |
| 跨域 RDMA | ~326 (2n) | ~500+ (2n) | RDMA 带宽 3200 vs 2000 Gbps (+60%), CX-8 GPUDirect 延迟更低 |
| 跨域大规模 | ~798 (16n) | ~900+ (16n) | 8 NIC vs 4 NIC, 聚合带宽翻倍 |

> **注意**: 以上为理论估算，实际性能取决于 GIB 对 CX-8 的优化程度和 IPv6 RoCE 实现。需实测验证。

---

## 4.1 单节点 4 GPU (NVLink 基线)

### 官方快速测试（推荐）

使用官方 NCCL 测试 YAML 快速验证：

```bash
# 部署官方 2 节点测试 YAML（单节点内也可用于验证 NVLink）
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/gpudirect-rdma/nccl-test-a4x-max.yaml

# 检查 Pod 状态
kubectl get pods nccl-test-host-1 nccl-test-host-2

# 运行测试
kubectl exec nccl-test-host-1 -it -- bash -c \
  "/usr/local/gib/scripts/run_nccl_tests.sh -t allreduce -b 512M -e 8G nccl-host-1"
```

### 手动测试（灵活控制参数）

```bash
# 部署单节点测试 Pod
kubectl apply -f yamls/nccl-single-node.yaml

# 等待完成
kubectl logs nccl-single-node -f
```

### GB200 基线

| 指标 | GB200 实测结果 |
|------|---------------|
| all_reduce 4 GPU @8G | **683 GB/s busbw** |

### GB300 实测结果

| 指标 | GB300 实测结果 | vs GB200 |
|------|---------------|----------|
| all_reduce 4 GPU @8G | — | — |

---

## 4.2 同域 2 节点 MNNVL（ComputeDomain + DRANET）

ComputeDomain 管理 IMEX daemon 生命周期，Pod 通过 `spec.resourceClaims` 引用 ResourceClaimTemplate 获取 IMEX channel，启用 MNNVL。

> **GB300 差异**: RDMA 使用 DRANET + asapd-lite（非 RDMA installer DaemonSet），ResourceClaimTemplate 中 MRDMA count = 8（非 4）。ComputeDomain 和 IMEX 逻辑与 GB200 相同。

### 官方快速测试

```bash
# 部署官方 2 节点测试
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/gpudirect-rdma/nccl-test-a4x-max.yaml

# 检查 Pod 状态
kubectl get pods nccl-test-host-1 nccl-test-host-2

# 触发 alltoall 测试
HOSTS="nccl-host-1 nccl-host-2"
kubectl exec nccl-test-host-1 -it -- bash -c \
  "/usr/local/gib/scripts/run_nccl_tests.sh -t alltoall -b 1M -e 16G ${HOSTS}"

# 触发 all_reduce 测试
kubectl exec nccl-test-host-1 -it -- bash -c \
  "/usr/local/gib/scripts/run_nccl_tests.sh -t allreduce -b 512M -e 16G ${HOSTS}"
```

### 手动测试

```bash
# 部署 NCCL 同域测试 Pod（使用 ComputeDomain + DRANET）
kubectl apply -f yamls/nccl-same-domain-dranet.yaml

# 等待 Pod 就绪
kubectl get pods -l name -w

# 验证 IMEX channel 已挂载
kubectl exec nccl-sd-h1 -- ls /dev/nvidia-caps-imex-channels/
# 预期: channel0

# 交换 SSH 密钥（必须使用 ed25519）
HOST1_KEY=$(kubectl exec nccl-sd-h1 -- cat /root/.ssh/id_ed25519.pub)
HOST2_KEY=$(kubectl exec nccl-sd-h2 -- cat /root/.ssh/id_ed25519.pub)
kubectl exec nccl-sd-h1 -- bash -c "echo '$HOST2_KEY' >> /root/.ssh/authorized_keys"
kubectl exec nccl-sd-h2 -- bash -c "echo '$HOST1_KEY' >> /root/.ssh/authorized_keys"
```

**MPI 编译注意**（与 GB200 相同）：

- pytorch 镜像自带的 `all_reduce_perf`**未链接 MPI**，多节点测试会退化为独立单 GPU 基准，必须从源码编译 MPI 版
- mpirun 路径为 `/usr/local/mpi/bin/mpirun`（非 `/usr/local/gib/bin/mpirun`）
- 必须使用 `-o BatchMode=yes` 避免交互式 SSH 提示

```bash
# 在 nccl-sd-h1 内编译 MPI 版 nccl-tests
kubectl exec nccl-sd-h1 -- bash -c "
  cd /tmp && git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git
  cd nccl-tests && make -j8 MPI=1 \
    MPI_HOME=/usr/local/mpi \
    NCCL_HOME=/usr/local/gib \
    CUDA_HOME=/usr/local/cuda
  ldd build/all_reduce_perf | grep libmpi  # 应输出 libmpi.so.40
  cp build/all_reduce_perf /tmp/all_reduce_perf
"

# scp 到 h2
HOST2_IP=$(kubectl get pod nccl-sd-h2 -o jsonpath='{.status.podIP}')
kubectl exec nccl-sd-h1 -- scp -P 2222 -o StrictHostKeyChecking=no \
  /tmp/all_reduce_perf ${HOST2_IP}:/tmp/all_reduce_perf
```

```bash
# 运行 MNNVL 测试
HOST1_IP=$(kubectl get pod nccl-sd-h1 -o jsonpath='{.status.podIP}')
HOST2_IP=$(kubectl get pod nccl-sd-h2 -o jsonpath='{.status.podIP}')

kubectl exec nccl-sd-h1 -- bash -c "
  source /usr/local/gib/scripts/set_nccl_env.sh 2>/dev/null
  export LD_LIBRARY_PATH=/usr/local/gib/lib64:/usr/local/nvidia/lib64:\$LD_LIBRARY_PATH
  /usr/local/mpi/bin/mpirun --allow-run-as-root \
    -np 8 -npernode 4 \
    --host ${HOST1_IP}:4,${HOST2_IP}:4 \
    -x LD_LIBRARY_PATH -x NCCL_MNNVL_ENABLE=2 -x NCCL_CUMEM_ENABLE=1 \
    --mca plm_rsh_args '-p 2222 -o BatchMode=yes -o StrictHostKeyChecking=no' \
    /tmp/all_reduce_perf -b 512M -e 16G -f 2 -g 1
"
```

### GB200 基线

| Collective | @16G busbw (GB/s) |
|------|----------|
| all_reduce | **842** |
| all_gather | **683** |
| reduce_scatter | **693** |
| alltoall | **683** |

### GB300 实测结果

| Collective | @16G busbw (GB/s) | vs GB200 | 备注 |
|------|----------|----------|------|
| all_reduce | — | — | |
| all_gather | — | — | |
| reduce_scatter | — | — | |
| alltoall | — | — | |

---

## 4.3 跨域 2 节点 RDMA（DRANET）

跨域节点无 MNNVL，使用纯 RDMA (GPUDirect-TCPX/GIB) 通信。不需要 ComputeDomain channel（无 IMEX）。

> **GB300 预期提升**: 跨域 RDMA 是 GB300 最大的性能提升点。CX-8 直连 GPU (GPUDirect RDMA) + 8 NIC (vs 4) + 3200 Gbps (vs 2000 Gbps)，理论跨域带宽提升 60%+。

### 官方快速测试

```bash
# 使用官方 YAML（确保 Pod 调度到不同 NVLink domain）
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/gpudirect-rdma/nccl-test-a4x-max.yaml

# 修改 Pod affinity 确保跨域调度
# 在 YAML 中添加:
#   podAntiAffinity:
#     preferredDuringSchedulingIgnoredDuringExecution:
#     - weight: 100
#       podAffinityTerm:
#         labelSelector:
#           matchLabels:
#             app: nccl-test
#         topologyKey: cloud.google.com/gke-nodepool

# 运行跨域 all_reduce 测试
HOSTS="nccl-host-1 nccl-host-2"
kubectl exec nccl-test-host-1 -it -- bash -c \
  "/usr/local/gib/scripts/run_nccl_tests.sh -t allreduce -b 512M -e 8G ${HOSTS}"
```

### 手动测试

```bash
# 部署跨域 NCCL 测试 Pod
kubectl apply -f yamls/nccl-cross-domain-dranet.yaml

# 等待 Pod 就绪 + 交换 SSH 密钥（同 4.2，使用 ed25519）
# 编译 MPI 版 nccl-tests（同 4.2）

HOST1_IP=$(kubectl get pod nccl-cd-h1 -o jsonpath='{.status.podIP}')
HOST2_IP=$(kubectl get pod nccl-cd-h2 -o jsonpath='{.status.podIP}')

kubectl exec nccl-cd-h1 -- bash -c "
  source /usr/local/gib/scripts/set_nccl_env.sh 2>/dev/null
  export LD_LIBRARY_PATH=/usr/local/gib/lib64:/usr/local/nvidia/lib64:\$LD_LIBRARY_PATH
  /usr/local/mpi/bin/mpirun --allow-run-as-root \
    -np 8 -npernode 4 \
    --host ${HOST1_IP}:4,${HOST2_IP}:4 \
    -x LD_LIBRARY_PATH -x NCCL_MNNVL_ENABLE=0 -x NCCL_CUMEM_ENABLE=1 \
    --mca plm_rsh_args '-p 2222 -o BatchMode=yes -o StrictHostKeyChecking=no' \
    /tmp/all_reduce_perf -b 512M -e 8G -f 2 -g 1
"
```

### GB200 基线

| 指标 | GB200 实测结果 |
|------|---------------|
| all_reduce 8 GPU @8G (RDMA) | **325.88 GB/s busbw** |

### GB300 实测结果

| 指标 | GB300 实测结果 | vs GB200 | 备注 |
|------|---------------|----------|------|
| all_reduce 8 GPU @8G (RDMA) | — | — | |

---

## 4.4 混合 4 节点（2 同域 + 2 跨域）

```bash
# 部署 4 节点混合测试
kubectl apply -f yamls/nccl-4node-mixed-dranet.yaml

# 等待所有 4 个 Pod 就绪 + 交换 SSH 密钥（4 个 Pod 间两两交换 ed25519 公钥）
# 编译 MPI 版 nccl-tests 并 scp 到所有节点（同 4.2）

HOST1_IP=$(kubectl get pod nccl-mix-h1 -o jsonpath='{.status.podIP}')
HOST2_IP=$(kubectl get pod nccl-mix-h2 -o jsonpath='{.status.podIP}')
HOST3_IP=$(kubectl get pod nccl-mix-h3 -o jsonpath='{.status.podIP}')
HOST4_IP=$(kubectl get pod nccl-mix-h4 -o jsonpath='{.status.podIP}')

kubectl exec nccl-mix-h1 -- bash -c "
  source /usr/local/gib/scripts/set_nccl_env.sh 2>/dev/null
  export LD_LIBRARY_PATH=/usr/local/gib/lib64:/usr/local/nvidia/lib64:\$LD_LIBRARY_PATH
  /usr/local/mpi/bin/mpirun --allow-run-as-root \
    -np 16 -npernode 4 \
    --host ${HOST1_IP}:4,${HOST2_IP}:4,${HOST3_IP}:4,${HOST4_IP}:4 \
    -x LD_LIBRARY_PATH -x NCCL_MNNVL_ENABLE=0 -x NCCL_CUMEM_ENABLE=1 \
    --mca plm_rsh_args '-p 2222 -o BatchMode=yes -o StrictHostKeyChecking=no' \
    /tmp/all_reduce_perf -b 1M -e 8G -f 2 -g 1
"
```

**NCCL_MNNVL_ENABLE 注意**（与 GB200 相同）：混合域测试中必须设置 `NCCL_MNNVL_ENABLE=0`。设置为 `2`（自动检测）时，NCCL 会在跨域节点（无 IMEX channel）上尝试探测 MNNVL 能力导致 CUDA error。如需同时利用同域 MNNVL 和跨域 RDMA，应将同域与跨域分离为独立的通信组。

### GB200 基线

| 指标 | GB200 实测结果 |
|------|---------------|
| all_reduce 16 GPU @8G (纯 RDMA) | **162.45 GB/s busbw** |

### GB300 实测结果

| 指标 | GB300 实测结果 | vs GB200 | 备注 |
|------|---------------|----------|------|
| all_reduce 16 GPU @8G (纯 RDMA) | — | — | |

---

## 4.5 全域 18 节点 72 GPU（MNNVL 满配）

全域测试覆盖单个 NVL72 domain 的全部 18 节点 72 GPU，验证 NVSwitch fabric 的满负荷性能。

### 部署

**方式一：JobSet（推荐，官方方式）**

```bash
# 安装 JobSet（如果尚未安装）
VERSION=v0.10.1
kubectl apply --server-side -f https://github.com/kubernetes-sigs/jobset/releases/download/$VERSION/manifests.yaml

# 下载并部署 18 节点 NCCL 测试
wget https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/refs/heads/master/gpudirect-rdma/nccl-test-a4x-max-jobset.yaml

NUM_NODES=18
sed "s|__NUM_NODES__|${NUM_NODES}|" nccl-test-a4x-max-jobset.yaml | kubectl apply -f -

# 等待全部 Pod 完成
kubectl get pods | grep allgather-worker

# 查看测试结果（从 head Pod）
kubectl logs $(kubectl get pods -o go-template='{{range .items}}{{.metadata.name}}{{"\\n"}}{{end}}' | grep allgather-worker-0-0)
```

**方式二：StatefulSet（手动编排）**

```bash
# 使用 StatefulSet 方式部署
kubectl apply -f yamls/nccl-18node-1domain-sts.yaml

# 等待全部 18 Pod 就绪（镜像已缓存时 ~2 分钟）
kubectl get pods -l app=nccl-18n-g1 -o wide -w

# master (ordinal 0) 自动编排：等待所有 peer sshd 就绪 -> 依次运行 4 项 collective
kubectl logs nccl-18n-g1-0 -f
```

StatefulSet YAML 自动处理：ComputeDomain 创建、RDMA NIC 分配（GB300 为 8 NIC）、SSH 密钥交换、MPI 启动。全部 18 Pod 通过 `podAffinity (nvidia.com/gpu.clique)` 约束到同一 NVL72 domain，`podAntiAffinity (kubernetes.io/hostname)` 确保每节点一个 Pod。

### GB200 基线

#### GB200 v1 镜像实测（tlinux-server-4-gb200-v1, 2026-06-28）

| Collective | @16G in-place busbw (GB/s) | avg busbw (GB/s) |
|---|---|---|
| all_reduce | **876.55** | 887.66 |
| all_gather | **673.66** | 675.04 |
| reduce_scatter | **697.43** | 696.94 |
| alltoall | **627.18** | 627.22 |

#### GB200 标称参考值

| Collective | @16G in-place busbw (GB/s) |
|---|---|
| all_reduce | **905.05** |
| all_gather | **681.38** |
| reduce_scatter | **702.67** |
| alltoall | **650.96** |

### GB300 实测结果

| Collective | @16G in-place busbw (GB/s) | avg busbw (GB/s) | vs GB200 v1 | 备注 |
|---|---|---|---|---|
| all_reduce | — | — | — | |
| all_gather | — | — | — | |
| reduce_scatter | — | — | — | |
| alltoall | — | — | — | |

---

## 4.6 跨域 36 节点 144 GPU（2 x NVL72 Domain）

跨域测试使用 2 个 NVL72 domain 共 36 节点 144 GPU。域内通信走 MNNVL (NVLink)，跨域通信走 RDMA (GIB)。

### 部署

**方式一：JobSet（推荐）**

```bash
# 使用 JobSet 部署 36 节点测试
wget https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/refs/heads/master/gpudirect-rdma/nccl-test-a4x-max-jobset.yaml

NUM_NODES=36
sed "s|__NUM_NODES__|${NUM_NODES}|" nccl-test-a4x-max-jobset.yaml | kubectl apply -f -

# 等待完成并查看结果
kubectl get pods | grep allgather-worker
kubectl logs $(kubectl get pods -o go-template='{{range .items}}{{.metadata.name}}{{"\\n"}}{{end}}' | grep allgather-worker-0-0)
```

**方式二：StatefulSet**

```bash
# 使用 StatefulSet 方式部署（2 domain 版本）
kubectl apply -f yamls/nccl-36node-2domain-sts.yaml

# 等待全部 36 Pod 就绪
kubectl get pods -l app-wide=nccl-36n -o wide -w

# 查看测试进度
kubectl logs nccl-36n-g1-0 -f
```

**关键配置差异**（与 GB200 相同）：
- 2 个 ComputeDomain（每域 18 节点），各自管理独立的 IMEX session
- `NCCL_MNNVL_ENABLE=0`：跨域测试必须关闭 MNNVL 自动检测，否则 NCCL 在跨域节点探测 IMEX channel 会导致 CUDA error
- alltoall 跨域性能会大幅下降（已知 NCCL chain pollution 问题）

### GB200 基线

#### GB200 v1 镜像实测（tlinux-server-4-gb200-v1, 2026-06-28）

| Collective | @16G in-place busbw (GB/s) | avg busbw (GB/s) |
|---|---|---|
| all_reduce | **748.24** | 741.10 |
| all_gather | **674.32** | 670.95 |
| reduce_scatter | **690.80** | 691.69 |
| alltoall | **65.26** | 66.31 |

#### GB200 标称参考值

| Collective | @16G in-place busbw (GB/s) |
|---|---|
| all_reduce | **688.14** |
| all_gather | **704.13** |
| reduce_scatter | **699.75** |
| alltoall | **40.59** * |

\* alltoall 跨 2 ComputeDomain 受 NCCL chain pollution 影响（已知 issue），单独跑 vanilla 模式约 40 GB/s。

### GB300 实测结果

| Collective | @16G in-place busbw (GB/s) | avg busbw (GB/s) | vs GB200 v1 | 备注 |
|---|---|---|---|---|
| all_reduce | — | — | — | |
| all_gather | — | — | — | |
| reduce_scatter | — | — | — | |
| alltoall | — | — | — | |

---

## 性能调优

### 常见性能问题排查

| 现象 | 可能原因 | 排查方法 |
|---|---|---|
| busbw 远低于预期 | GIB 未启用 | 检查 `NCCL_NET=gIB`，确认 `NCCL_DEBUG=INFO` 日志中有 `gIB` 字样 |
| 同域 busbw < 800 | MNNVL 未启用 | 检查 `NCCL_MNNVL_ENABLE=2`，确认 IMEX channel 存在 (`/dev/nvidia-caps-imex-channels/channel0`) |
| 跨域 CUDA error | MNNVL 在跨域节点探测失败 | 设置 `NCCL_MNNVL_ENABLE=0` |
| alltoall 跨域极慢 | NCCL chain pollution | 已知 issue，单独跑 alltoall 可用 dedicated YAML |
| 测试卡在 barrier | SSH 不通 | 检查 sshd 端口 (2222)，DNS 解析，网络 |
| Pod 卡在 ContainerCreating | DRANET DeviceClass 未过滤非 RDMA 接口 | 确认 DeviceClass 包含 `rdma == true` 过滤 |
| RDMA 连接失败 | asapd-lite 未就绪 | `kubectl get daemonset -n kube-system asapd-lite`，READY 数须等于节点数 |
| 编译后 all_reduce_perf 单 GPU 退化 | 未链接 MPI | `ldd build/all_reduce_perf \| grep libmpi` 确认 |

### GB300 特有注意事项

1. **ARM64 架构**: GB300 使用 Grace ARM64 CPU，容器镜像必须为 ARM64 或 multi-arch。使用 `nodeSelector: kubernetes.io/arch: arm64` 确保调度正确
2. **IPv6-only 网络**: 所有 NIC 使用 IPv6，NCCL/GIB 的 RoCEv2 走 IPv6。如遇连接问题优先检查 IPv6 路由和防火墙规则
3. **MRDMA 数量 8 (非 4)**: ResourceClaimTemplate 中 `count: 8`，Pod 必须请求全部 8 个 MRDMA 接口
4. **DOCA/OFED 用户空间库**: 容器镜像需安装 ARM64 版 DOCA OFED 用户空间库（`doca-ofed-userspace`），参考[官方文档](https://docs.cloud.google.com/ai-hypercomputer/docs/create/gke-ai-hypercompute-custom-a4x-max)
5. **Hugepages**: 节点池需配置 `hugepage_size2m: 4096`（GKE node-pool 创建时 `--system-config-from-file`）
6. **NCCL 版本**: 官方推荐 libnccl2 >= 2.28.9，容器镜像中可能需要手动升级（`apt install --only-upgrade libnccl2 libnccl-dev`）

---

## 测试结果汇总

### GB200 全量 Benchmark 汇总（基线）

| 测试配置 | GPU | 互联 | all_reduce | all_gather | reduce_scatter | alltoall | 来源 |
|---|---|---|---|---|---|---|---|
| 单节点 4GPU | 4 | NVLink | 684 | — | — | — | GB200 我方验证 |
| 同域 2n @16G | 8 | MNNVL | **842** | 683 | 693 | 683 | GB200 我方验证 |
| 同域 8n @16G | 32 | MNNVL | **909** | 691 | 708 | 676 | GB200 我方验证 |
| 同域 16n @16G | 64 | MNNVL | **910** | 693 | 707 | 667 | GB200 我方验证 |
| 全域 18n @16G | 72 | MNNVL | 877 (v1) | 674 (v1) | 697 (v1) | 627 (v1) | GB200 v1 实测 |
| 跨域 4n @16G | 16 | MNNVL+RDMA | **691** | 378 | 379 | 88 | GB200 我方验证 |
| 跨域 16n @16G | 64 | MNNVL+RDMA | **798** | 688 | 702 | 83 | GB200 我方验证 |
| 跨域 36n @16G | 144 | MNNVL+RDMA | 748 (v1) | 674 (v1) | 691 (v1) | 65 (v1) | GB200 v1 实测 |

单位：GB/s busbw

### GB300 全量 Benchmark 汇总（待填写）

| 测试配置 | GPU | 互联 | all_reduce | all_gather | reduce_scatter | alltoall | vs GB200 all_reduce | 备注 |
|---|---|---|---|---|---|---|---|---|
| 单节点 4GPU | 4 | NVLink | — | — | — | — | — | |
| 同域 2n @16G | 8 | MNNVL | — | — | — | — | — | |
| 同域 8n @16G | 32 | MNNVL | — | — | — | — | — | |
| 同域 16n @16G | 64 | MNNVL | — | — | — | — | — | |
| 全域 18n @16G | 72 | MNNVL | — | — | — | — | — | |
| 跨域 4n @16G | 16 | MNNVL+RDMA | — | — | — | — | — | |
| 跨域 16n @16G | 64 | MNNVL+RDMA | — | — | — | — | — | |
| 跨域 36n @16G | 144 | MNNVL+RDMA | — | — | — | — | — | |

单位：GB/s busbw

---

## 跨域 NCCL 排查经验（继承自 GB200）

GB200 上跨域 NCCL 从完全不通到跑通经历了 3 轮调试（详见 [a4x/04-nccl-test](../../a4x/04-nccl-test/)），核心教训在 GB300 上同样适用：

1. **每个 Pod 必须有 ComputeDomain channel**：即使跨域 Pod 不走 NVLink，GIB 插件初始化时会探测所有通信路径，缺少 IMEX channel 会导致 GIB 挂住
2. **跨域需要双 ComputeDomain + 双 StatefulSet**：每个 domain 独立的 ComputeDomain + StatefulSet，通过 `nodeSelector: nvidia.com/gpu.clique` 约束
3. **`NCCL_MNNVL_ENABLE=2`（跨域时）**：让 NCCL 自动检测域内/域间路径（与 GB200 相同）
4. **DRANET DeviceClass 必须含 `rdma == true` 过滤**：否则可能分配到 Calico vxlan 接口导致 `network is unreachable`

### GB300 额外注意

- **asapd-lite 健康检查**：GB300 使用 asapd-lite DaemonSet 配置 MRDMA NIC，DaemonSet 不健康 = 无 RDMA 连接。部署后第一步检查 `kubectl get daemonset -n kube-system asapd-lite`
- **IPv6 路由**：跨域 RDMA 走 IPv6，确保跨域子网 IPv6 路由可达
- **8 NIC 全部就绪**：`rdma link show` 应显示 8 个 MRDMA 设备全部 active

## 官方文档参考

| 主题 | URL |
|------|-----|
| NCCL 测试（A4X Max GKE） | https://docs.cloud.google.com/ai-hypercomputer/docs/nccl/test-gke-custom-a4x-max |
| A4X Max GKE 部署 | https://docs.cloud.google.com/ai-hypercomputer/docs/create/gke-ai-hypercompute-custom-a4x-max |
| NCCL/gIB 概览 | https://docs.cloud.google.com/ai-hypercomputer/docs/nccl/overview |
| GPU 网络带宽 | https://cloud.google.com/compute/docs/gpus/gpu-network-bandwidth |
| GB200 NCCL 测试（对比） | [a4x/04-nccl-test](../../a4x/04-nccl-test/) |

## GB300 实测结果 (2026-07-12)

### 4.1 单节点 4 GPU NVLink (Test 1)

集群: chrisya-gb300-gke, subblock-0005, 1 节点 4 GPU

```
# nccl-tests version 2.19.6
#       size         count      type   redop     time   algbw   busbw
     1048576        262144     float     sum     25.35   41.37   62.06
     2097152        524288     float     sum     25.65   81.76  122.63
     4194304       1048576     float     sum     42.88   97.82  146.73
     8388608       2097152     float     sum     46.03  182.26  273.39
    16777216       4194304     float     sum    100.52  166.91  250.36
    33554432       8388608     float     sum    121.21  276.82  415.23
    67108864      16777216     float     sum    182.47  367.78  551.67
   134217728      33554432     float     sum    339.62  395.20  592.81
   268435456      67108864     float     sum    657.68  408.15  612.23
   536870912     134217728     float     sum   1269.07  423.04  634.56
  1073741824     268435456     float     sum   2489.71  431.27  646.91
# Avg bus bandwidth: 390.79 GB/s
```

峰值 busbw **646.91 GB/s** (NV18 NVLink, 单节点)

### 4.2 同域 2 节点 MNNVL (Test 2) — 2026-07-12

集群: chrisya-gb300-gke, subblock-0005, 2 节点 8 GPU, MNNVL=2

**手动 IMEX 配置步骤**:
1. 两节点启动 `nvidia-imex` daemon (互写 nodes_config.cfg)
2. `mknod` 创建 `/dev/nvidia-caps-imex-channels/channel{0..255}` (major=240)
3. `chmod 666` 所有 channel 设备

```
Size:    1 MB | Time:     0.06 ms | BusBW:    31.99 GB/s
Size:   16 MB | Time:     0.10 ms | BusBW:   303.50 GB/s
Size:   64 MB | Time:     0.28 ms | BusBW:   418.84 GB/s
Size:  256 MB | Time:     0.72 ms | BusBW:   652.38 GB/s
Size:  512 MB | Time:     1.35 ms | BusBW:   697.37 GB/s
Size: 1024 MB | Time:     2.61 ms | BusBW:   721.22 GB/s
```

峰值 busbw **721.22 GB/s** (NVLink MNNVL, 2 节点 8 GPU)
vs 单节点 646.91 GB/s — MNNVL 跨节点性能 **+11.5%** (NVSwitch 全带宽)

### 4.2 续：大数据量公平对比 (vs GB200)

同样用 PyTorch NCCL all-reduce，MNNVL=2，2 节点 8 GPU:

```
Size:   512 MB | BusBW:   698.88 GB/s
Size:  1024 MB | BusBW:   722.90 GB/s
Size:  2048 MB | BusBW:   806.53 GB/s
Size:  4096 MB | BusBW:   820.65 GB/s
Size:  8192 MB | BusBW:   831.82 GB/s
```

**GB300 vs GB200 公平对比**:

| 数据量 | GB300 | GB200 | 差距 |
|--------|-------|-------|------|
| 8 GB | **831.82 GB/s** | **839.54 GB/s** (nccl-tests @16G) | **-0.9%** |

结论: **NVLink 5 同代同速**。GB300 和 GB200 的 NVLink 带宽基本一致，差距在测试误差范围内。
