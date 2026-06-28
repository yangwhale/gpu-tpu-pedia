# 5. NCCL 通信测试

本章覆盖 6 个层级的 NCCL 通信测试：单节点 NVLink、同域 2 节点 MNNVL、跨域 2 节点 RDMA、混合 4 节点、全域 18 节点 72 GPU、跨域 36 节点 144 GPU。每项测试包含标称参考值、v1 镜像实测数据和我方验证预留栏。

## 5.1 单节点 4 GPU (NVLink 基线)

```bash
# 部署单节点测试 Pod
kubectl apply -f yamls/k8s1341-nccl-single-node.yaml

# 等待完成
kubectl logs nccl-single-node -f
```

| 指标 | 实测结果 |
|------|----------|
| all_reduce 4 GPU @8G | **683 GB/s busbw** |

**单节点 NCCL 测试通过**：all_reduce 683 GB/s busbw（实测 2026-06-27 验证一致）。

> **为什么 4 GPU 只有 683 而 8 GPU 跨节点有 835？** NVSwitch 是为 72 GPU 全互联设计的。4 GPU 的 ring all_reduce 只有 3 hop，无法充分利用 NVSwitch 的全部并行通道。8 GPU 跨 2 节点时 NCCL 有更多 NVLink 并行传输路径，busbw 反而更高。18 节点 72 GPU 全域 all_reduce 才能逼近 900 GB/s 的理论上限。

## 5.2 同域 2 节点 MNNVL（ComputeDomain + DRANET）

ComputeDomain（[03-gpu-stack](../03-gpu-stack/) 3.6 步骤已创建）管理 IMEX daemon 生命周期，Pod 通过 `spec.resourceClaims` 引用 ResourceClaimTemplate 获取 IMEX channel，启用 MNNVL。核心声明模式（与 [07-megatron-training](../07-megatron-training/) 相同）：

```yaml
# Pod 通过 DRA 声明两类资源：
#   1. compute-domain-channel → ComputeDomain 自动生成的 ResourceClaimTemplate（提供 IMEX channel）
#   2. rdma-nics             → 手动创建的 ResourceClaimTemplate（提供 4 张 RDMA 网卡）
#
# Pod spec 关键字段：
#   spec.resourceClaims:
#   - name: compute-domain-channel
#     resourceClaimTemplateName: sd-compute-domain-channel   # ← ComputeDomain 3.6 步自动生成
#   - name: rdma-nics
#     resourceClaimTemplateName: rdma-nics-sd-h1             # ← YAML 中手动定义
#   containers[].resources.claims:
#   - name: compute-domain-channel                           # ← 容器级别引用
#   - name: rdma-nics
```

```bash
# 部署 NCCL 同域测试 Pod（使用 ComputeDomain + DRANET）
kubectl apply -f yamls/k8s1341-nccl-same-domain-dranet.yaml

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

**MPI 编译注意**：

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
    /tmp/all_reduce_perf -b 512M -e 8G -f 2 -g 1
"
```

| Collective | @16G busbw (GB/s) |
|------|----------|
| all_reduce | **839.54** |
| all_gather | **683.83** |
| reduce_scatter | **693.07** |
| alltoall | **682.73** |

**同域 MNNVL 测试通过**：4 项 collective 均正常（实测 2026-06-27），与生产环境标称值（[10-production-ops](../10-production-ops/) 8.2 节）完全一致（±0.5% 以内）：

| Collective | 实测 busbw | 标称参考值 | 差异 |
|---|---|---|---|
| all_reduce | 839.54 | 840.12 | -0.07% |
| all_gather | 683.83 | 683.37 | +0.07% |
| reduce_scatter | 693.07 | 693.12 | -0.01% |
| alltoall | 682.73 | 679.93 | +0.41% |

> **StatefulSet 方式（推荐）**：使用 GIB 诊断镜像自带的 `run_nccl_tests.sh` 脚本，StatefulSet 自动编排 SSH 密钥交换和 MPI 启动，无需手动操作。参考 `yamls/benchmark/k8s134-nccl-2node-1domain-sts.yaml`。

## 5.3 跨域 2 节点 RDMA（DRANET）

跨域节点无 MNNVL，使用纯 RDMA (GPUDirect-TCPX/GIB) 通信。不需要 ComputeDomain channel（无 IMEX）。

```bash
# 部署跨域 NCCL 测试 Pod
kubectl apply -f yamls/k8s1341-nccl-cross-domain-dranet.yaml

# 等待 Pod 就绪 + 交换 SSH 密钥（同 5.2，使用 ed25519）
# 编译 MPI 版 nccl-tests（同 5.2）

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

| 指标 | 实测结果 |
|------|----------|
| all_reduce 8 GPU @8G (RDMA) | **325.88 GB/s busbw** |

**跨域 RDMA 测试通过**：all_reduce 8GPU @8G 达到 325.88 GB/s busbw（平均 318.7 GB/s）。纯 GPUDirect RDMA 通信正常，无需 ComputeDomain/IMEX。

## 5.4 混合 4 节点（2 同域 + 2 跨域）

```bash
# 部署 4 节点混合测试
kubectl apply -f yamls/k8s1341-nccl-4node-mixed-dranet.yaml

# 等待所有 4 个 Pod 就绪 + 交换 SSH 密钥（4 个 Pod 间两两交换 ed25519 公钥）
# 编译 MPI 版 nccl-tests 并 scp 到所有节点（同 5.2）

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

**NCCL_MNNVL_ENABLE 注意**：混合域测试中必须设置 `NCCL_MNNVL_ENABLE=0`。设置为 `2`（自动检测）时，NCCL 会在跨域节点（无 IMEX channel）上尝试探测 MNNVL 能力导致 CUDA error。如需同时利用同域 MNNVL 和跨域 RDMA，应将同域与跨域分离为独立的通信组。

| 指标 | 实测结果 |
|------|----------|
| all_reduce 16 GPU @8G (纯 RDMA) | **162.45 GB/s busbw** |

**混合 4 节点测试通过**：all_reduce 16GPU @8G 达到 162.45 GB/s busbw（平均 156.7 GB/s）。所有 4 节点纯 RDMA 通信正常。

## 5.5 全域 18 节点 72 GPU（MNNVL 满配）

全域测试覆盖单个 NVL72 domain 的全部 18 节点 72 GPU，验证 NVSwitch fabric 的满负荷性能。

### 部署

```bash
# 使用 StatefulSet 方式部署（推荐）
kubectl apply -f yamls/benchmark/k8s134-nccl-18node-1domain-sts.yaml

# 等待全部 18 Pod 就绪（镜像已缓存时 ~2 分钟）
kubectl get pods -l app=nccl-18n-g1 -o wide -w

# master (ordinal 0) 自动编排：等待所有 peer sshd 就绪 → 依次运行 4 项 collective
kubectl logs nccl-18n-g1-0 -f
```

StatefulSet YAML 自动处理：ComputeDomain 创建、RDMA NIC 分配、SSH 密钥交换、MPI 启动。全部 18 Pod 通过 `podAffinity (nvidia.com/gpu.clique)` 约束到同一 NVL72 domain，`podAntiAffinity (kubernetes.io/hostname)` 确保每节点一个 Pod。

### 测试结果

#### v1 镜像实测（tlinux-server-4-gb200-v1, 2026-06-28）

| Collective | @16G in-place busbw (GB/s) | avg busbw (GB/s) |
|---|---|---|
| all_reduce | **876.55** | 887.66 |
| all_gather | **673.66** | 675.04 |
| reduce_scatter | **697.43** | 696.94 |
| alltoall | **627.18** | 627.22 |

#### 标称参考值（[10-production-ops](../10-production-ops/) 8.2 节）

| Collective | @16G in-place busbw (GB/s) |
|---|---|
| all_reduce | **905.05** |
| all_gather | **681.38** |
| reduce_scatter | **702.67** |
| alltoall | **650.96** |

#### 对比分析

| Collective | v1 镜像实测 | 标称参考 | 差异 |
|---|---|---|---|
| all_reduce | 876.55 | 905.05 | -3.1% |
| all_gather | 673.66 | 681.38 | -1.1% |
| reduce_scatter | 697.43 | 702.67 | -0.7% |
| alltoall | 627.18 | 650.96 | -3.6% |

#### 我方验证（待测）

| Collective | @16G in-place busbw (GB/s) | avg busbw (GB/s) | 备注 |
|---|---|---|---|
| all_reduce | — | — | |
| all_gather | — | — | |
| reduce_scatter | — | — | |
| alltoall | — | — | |

## 5.6 跨域 36 节点 144 GPU（2 × NVL72 Domain）

跨域测试使用 2 个 NVL72 domain 共 36 节点 144 GPU。域内通信走 MNNVL (NVLink)，跨域通信走 RDMA (GIB)。

### 部署

```bash
# 使用 StatefulSet 方式部署（2 domain 版本）
kubectl apply -f yamls/benchmark/k8s134-nccl-36node-2domain-sts.yaml

# 等待全部 36 Pod 就绪
kubectl get pods -l app-wide=nccl-36n -o wide -w

# 查看测试进度
kubectl logs nccl-36n-g1-0 -f
```

**关键配置差异**：
- 2 个 ComputeDomain（每域 18 节点），各自管理独立的 IMEX session
- `NCCL_MNNVL_ENABLE=0`：跨域测试必须关闭 MNNVL 自动检测，否则 NCCL 在跨域节点探测 IMEX channel 会导致 CUDA error
- alltoall 跨域性能会大幅下降（已知 NCCL chain pollution 问题）

### 测试结果

#### v1 镜像实测（tlinux-server-4-gb200-v1, 2026-06-28）

| Collective | @16G in-place busbw (GB/s) | avg busbw (GB/s) |
|---|---|---|
| all_reduce | **748.24** | 741.10 |
| all_gather | **674.32** | 670.95 |
| reduce_scatter | **690.80** | 691.69 |
| alltoall | **65.26** | 66.31 |

#### 标称参考值

| Collective | @16G in-place busbw (GB/s) |
|---|---|
| all_reduce | **688.14** |
| all_gather | **704.13** |
| reduce_scatter | **699.75** |
| alltoall | **40.59** * |

\* alltoall 跨 2 ComputeDomain 受 NCCL chain pollution 影响（已知 issue），单独跑 vanilla 模式约 40 GB/s。

#### 对比分析

| Collective | v1 镜像实测 | 标称参考 | 差异 | 备注 |
|---|---|---|---|---|
| all_reduce | 748.24 | 688.14 | +8.7% | v1 实测优于标称 |
| all_gather | 674.32 | 704.13 | -4.2% | |
| reduce_scatter | 690.80 | 699.75 | -1.3% | |
| alltoall | 65.26 | 40.59 | +60.8% | 两者都受 chain pollution 影响，绝对值低 |

#### 我方验证（待测）

| Collective | @16G in-place busbw (GB/s) | avg busbw (GB/s) | 备注 |
|---|---|---|---|
| all_reduce | — | — | |
| all_gather | — | — | |
| reduce_scatter | — | — | |
| alltoall | — | — | |

---

## 性能调优

### NCCL 环境变量

GIB 诊断镜像内置 `set_nccl_env.sh` 脚本自动设置最优 NCCL 参数。关键变量：

| 变量 | 推荐值 | 说明 |
|---|---|---|
| NCCL_NET | gIB | 启用 GPUDirect RDMA (GIB) |
| NCCL_MNNVL_ENABLE | 2（同域）/ 0（跨域） | 2=自动检测 MNNVL，0=强制禁用 |
| NCCL_CUMEM_ENABLE | 1 | 启用 CUDA Memory Manager |
| NCCL_IB_GID_INDEX | 3 | RoCEv2 GID index |
| NCCL_IB_ADAPTIVE_ROUTING | 1 | 启用自适应路由 |
| NCCL_IB_QPS_PER_CONNECTION | 4 | 每连接 QP 数 |
| NCCL_IB_TC | 52 | Traffic Class（DSCP 标记） |
| NCCL_PXN_C2C | 1 | 启用 PCIe 跨节点 NVLink relay |
| NCCL_NVLS_ENABLE | 0 | 关闭 NVLink SHARP（GB200 不支持） |
| NCCL_IB_MERGE_NICS | 0 | 不合并 NIC（每 GPU 独立 NIC） |

### 常见性能问题排查

| 现象 | 可能原因 | 排查方法 |
|---|---|---|
| busbw 远低于预期 | GIB 未启用 | 检查 `NCCL_NET=gIB`，确认 `NCCL_DEBUG=INFO` 日志中有 `gIB` 字样 |
| 同域 busbw < 800 | MNNVL 未启用 | 检查 `NCCL_MNNVL_ENABLE=2`，确认 IMEX channel 存在 (`/dev/nvidia-caps-imex-channels/channel0`) |
| 跨域 CUDA error | MNNVL 在跨域节点探测失败 | 设置 `NCCL_MNNVL_ENABLE=0` |
| alltoall 跨域极慢 | NCCL chain pollution | 已知 issue，单独跑 alltoall 可用 `k8s134-nccl-36node-2domain-alltoall-sts.yaml` |
| 测试卡在 barrier | SSH 不通 | 检查 sshd 端口 (222)，DNS 解析，Calico 网络 |

---

## 测试结果汇总

| 测试 | GPU 数 | 互联方式 | all_reduce busbw (GB/s) | 数据来源 |
|------|--------|----------|-------------|---|
| 单节点 4 GPU @8G | 4 | NVLink | **683.75** | 我方验证 |
| 同域 2 节点 @16G | 8 | MNNVL | **839.54** | 我方验证 |
| 跨域 2 节点 @8G | 8 | RDMA (GIB) | **325.88** | 标称 |
| 混合 4 节点 @8G | 16 | RDMA (GIB) | **162.45** | 标称 |
| 全域 18 节点 @16G | 72 | MNNVL | **876.55** (v1) / 905.05 (标称) | v1 镜像实测 |
| 跨域 36 节点 @16G | 144 | MNNVL + RDMA | **748.24** (v1) / 688.14 (标称) | v1 镜像实测 |
