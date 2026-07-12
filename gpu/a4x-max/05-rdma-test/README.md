# 6. RDMA 带宽测试 (ib_write_bw)

以 GB200 (A4X) 实测数据为 baseline，验证 GB300 (A4X MAX) CX-8 SuperNIC GPUDirect RDMA 带宽。

## GB300 vs GB200 RDMA 架构对比

| 维度 | GB200 (A4X) | GB300 (A4X MAX) | 影响 |
|------|-------------|-----------------|------|
| 网卡型号 | CX-7 | CX-8 SuperNIC | 新一代 |
| 暴露方式 | SR-IOV VF (虚拟功能) | PF (物理功能) | 无虚拟化开销 |
| 连接方式 | 挂在 CPU 上 | **直连 GPU (GPUDirect RDMA)** | 延迟显著降低 |
| 每卡端口数 | 1 | 2 (800 Gbps = 2x400 Gbps) | 带宽翻倍 |
| MRDMA 接口数 | 4 | **8 (8-way rail)** | 子网翻倍 |
| 总网络带宽 | 2,000 Gbps | **3,200 Gbps** | +60% |
| IB 设备 | mlx5_0~3 | 待确认 (预期 8 个 PF) | 设备数翻倍 |

### Data Direct Interface (DDI)

GB300 的核心架构升级: NIC 通过 PCIe 直接连接到 GPU，绕过 CPU。

```
GB200 路径:  GPU ←PCIe→ CPU ←PCIe→ CX-7 NIC ←RDMA→ 远端
GB300 路径:  GPU ←PCIe→ CX-8 NIC ←RDMA→ 远端     (CPU 不在数据路径上)
```

**预期收益**:
- **延迟更低**: 数据不经过 CPU，减少 PCIe hop + CPU 内存拷贝
- **带宽更高**: 8-way rail 聚合 3,200 Gbps (vs 4-way 2,000 Gbps)
- **CPU 开销更低**: CPU 不参与数据搬运，释放 Grace ARM64 算力

## GB200 Baseline 结果

实测 2026-06-27，使用 GIB 诊断镜像自带 perftest:

| NIC | BW avg (Gbps) |
|------|----------|
| mlx5_0 | **382.10** |
| mlx5_1 | **382.12** |
| mlx5_2 | **382.19** |
| mlx5_3 | **382.15** |
| 4xNIC aggregate | **~1528 Gbps** |

实测 382.1-382.2 Gbps/NIC，与标称参考值 ~381 Gbps 一致。CX-7 NIC 均达到 400GbE 线速（理论 ~400 Gbps，实际 ~382 Gbps 考虑协议开销正常）。

## 测试步骤

### 步骤 1: 确认 CX-8 IB 设备

```bash
# 拿到 GB300 VM 后首先确认 IB 设备名和数量
rdma link show
ibstat

# 预期: 8 个 PF 设备 (mlx5_0~7 或类似命名)
# GB200 只有 4 个: mlx5_0~3
```

### 步骤 2: 标准 RDMA 带宽测试

可复用 NCCL 同域测试的 Pod（GIB 诊断镜像自带 perftest，无需额外安装）。

```bash
# 设置 Pod 名称（复用 NCCL 测试 Pod，或任意同域 2 Pod）
POD0=nccl-2n-g1-0   # server
POD1=nccl-2n-g1-1   # client
POD0_IP=$(kubectl get pod $POD0 -o jsonpath='{.status.podIP}')

# 逐个测试 8 块 CX-8 RDMA NIC
for i in 0 1 2 3 4 5 6 7; do
  NIC=mlx5_$i         # 设备名待拿到 VM 后确认
  PORT=$((18515 + i))
  echo "=== Testing $NIC ==="

  # Server（后台）
  kubectl exec $POD1 -- bash -c \
    "LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu ib_write_bw -p $PORT -d $NIC -s 65536 --report_gbits -F -D 5" &
  sleep 2

  # Client
  kubectl exec $POD0 -- bash -c \
    "LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu ib_write_bw -p $PORT -d $NIC -s 65536 --report_gbits -F -D 5 $POD0_IP"

  wait
done
```

### 步骤 3: GPUDirect RDMA 带宽测试

GB300 的 CX-8 NIC 直连 GPU，使用 `--use_cuda` 让数据直接从 GPU 显存发送，验证 GPUDirect 路径:

```bash
# GPUDirect 模式: 数据在 GPU 显存中，NIC 直接 DMA 读写 GPU 内存
for i in 0 1 2 3 4 5 6 7; do
  NIC=mlx5_$i
  PORT=$((18615 + i))
  echo "=== GPUDirect Testing $NIC ==="

  # Server（后台）
  kubectl exec $POD1 -- bash -c \
    "LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu ib_write_bw -p $PORT -d $NIC -s 65536 --report_gbits -F -D 5 --use_cuda=$i" &
  sleep 2

  # Client
  kubectl exec $POD0 -- bash -c \
    "LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu ib_write_bw -p $PORT -d $NIC -s 65536 --report_gbits -F -D 5 --use_cuda=$i $POD0_IP"

  wait
done
```

> **注意**: `--use_cuda=$i` 指定使用第 $i 号 GPU 的显存作为数据源/目标。GB300 CX-8 直连 GPU，GPUDirect 路径应比 GB200 CX-7 (经过 CPU) 有显著延迟优势。GPU 到 NIC 的映射关系需拿到 VM 后通过 `nvidia-smi topo -m` 确认。

### 步骤 4: 延迟测试

```bash
# RDMA 写延迟 (验证 GPUDirect 延迟优势)
for i in 0 1 2 3 4 5 6 7; do
  NIC=mlx5_$i
  PORT=$((18715 + i))
  echo "=== Latency Testing $NIC ==="

  kubectl exec $POD1 -- bash -c \
    "LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu ib_write_lat -p $PORT -d $NIC -s 65536 -n 1000" &
  sleep 2

  kubectl exec $POD0 -- bash -c \
    "LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu ib_write_lat -p $PORT -d $NIC -s 65536 -n 1000 $POD0_IP"

  wait
done
```

## GB300 实测结果

### 标准 RDMA 带宽

<!-- 拿到 GB300 VM 后填入实测数据 -->

| NIC | BW avg (Gbps) | vs GB200 baseline |
|------|----------|----------|
| mlx5_0 | — | — |
| mlx5_1 | — | — |
| mlx5_2 | — | — |
| mlx5_3 | — | — |
| mlx5_4 | — | — |
| mlx5_5 | — | — |
| mlx5_6 | — | — |
| mlx5_7 | — | — |
| 8xNIC aggregate | — | — |

**预期**: 每块 CX-8 NIC ~382 Gbps (与 GB200 CX-7 单卡带宽一致，均为 400GbE 线速)。聚合带宽 ~3,056 Gbps (8x382)，对比 GB200 ~1,528 Gbps (4x382) 提升 **+100%**。

### GPUDirect RDMA 带宽

<!-- 拿到 GB300 VM 后填入实测数据 -->

| NIC | GPU | BW avg (Gbps) | vs 标准 RDMA |
|------|-----|----------|----------|
| mlx5_0 | GPU 0 | — | — |
| mlx5_1 | GPU 0 | — | — |
| mlx5_2 | GPU 1 | — | — |
| mlx5_3 | GPU 1 | — | — |
| mlx5_4 | GPU 2 | — | — |
| mlx5_5 | GPU 2 | — | — |
| mlx5_6 | GPU 3 | — | — |
| mlx5_7 | GPU 3 | — | — |

> GPU-NIC 映射为预期值 (每 GPU 2 块 CX-8 NIC)，需通过 `nvidia-smi topo -m` 和 `rdma link show` 确认实际映射。

### RDMA 写延迟

<!-- 拿到 GB300 VM 后填入实测数据 -->

| NIC | 标准延迟 (us) | GPUDirect 延迟 (us) | GB200 标准 (参考) |
|------|------------|-----------------|--------------|
| mlx5_0 | — | — | — |
| ... | — | — | — |

**预期**: GPUDirect 模式延迟应低于标准模式 (数据不经 CPU)，且 GB300 GPUDirect 延迟应低于 GB200 标准 RDMA (CX-7 经 CPU 路径)。

## 注意事项

### IBVERBS 路径冲突 (同 GB200)

GIB 的 libibverbs 库 (`/usr/local/gib/lib64/`) 可能覆盖系统版本。运行 `ib_write_bw` 前需切换:

```bash
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu
```

### GB300 网卡映射

GB300 每节点 4 GPU + 8 CX-8 NIC (每 GPU 2 NIC)，预期映射:

| GPU | RDMA NIC | 说明 |
|-----|----------|------|
| GPU 0 | mlx5_0, mlx5_1 | CX-8 双端口 (800 Gbps) |
| GPU 1 | mlx5_2, mlx5_3 | CX-8 双端口 (800 Gbps) |
| GPU 2 | mlx5_4, mlx5_5 | CX-8 双端口 (800 Gbps) |
| GPU 3 | mlx5_6, mlx5_7 | CX-8 双端口 (800 Gbps) |

> 以上为推测映射，需拿到 VM 后通过 `nvidia-smi topo -m` 确认。GB200 是每 GPU 1 NIC (1:1)，GB300 是每 GPU 2 NIC (1:2)。

### RDMA 子网差异

| 维度 | GB200 | GB300 |
|------|-------|-------|
| 子网数 | 4 个独立子网 | 1 个共享子网 |
| 配置方式 | 手动创建 4 个子网 | RoCE VPC network profile 自动创建 |
| 接口命名 | MRDMA 0~3 | MRDMA 0~7 |

## GKE 测试环境

- 集群: `chrisya-gb300-gke`
- 项目: `tencent-gcp-taiji-poc`
- Zone: `us-central1-b`
- Placement policies: `gb300-central-nvl72-policy-0001~0012`
