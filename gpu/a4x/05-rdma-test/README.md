> 🌐 **中文** | [English](README.en.md)

# 6. RDMA 带宽测试 (ib_write_bw)

在同域 NCCL 测试 Pod 中运行 RDMA 带宽测试，验证每块 CX-7 NIC 的原始 RDMA 带宽。

## 测试步骤

可复用 NCCL 同域测试的 Pod（GIB 诊断镜像自带 perftest，无需额外安装）。

```bash
# 设置 Pod 名称（复用 NCCL 测试 Pod，或任意同域 2 Pod）
POD0=nccl-2n-g1-0   # server
POD1=nccl-2n-g1-1   # client
POD0_IP=$(kubectl get pod $POD0 -o jsonpath='{.status.podIP}')

# 逐个测试 4 块 CX-7 RDMA NIC
for i in 0 1 2 3; do
  NIC=mlx5_$i
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

## 测试结果

| NIC | BW avg (Gbps) |
|------|----------|
| mlx5_0 | **382.10** |
| mlx5_1 | **382.12** |
| mlx5_2 | **382.19** |
| mlx5_3 | **382.15** |
| 4×NIC aggregate | **~1528 Gbps** |

实测 382.1-382.2 Gbps/NIC，与标称参考值 ~381 Gbps 一致（实测 2026-06-27）。CX-7 NIC 均达到 400GbE 线速（理论 ~400 Gbps，实际 ~382 Gbps 考虑协议开销正常）。

## 注意事项

### IBVERBS 路径冲突

GIB 的 libibverbs 库 (`/usr/local/gib/lib64/`) 可能覆盖系统版本。运行 `ib_write_bw` 前需切换：

```bash
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu
```

### A4X 网卡映射

A4X 每节点 4 GPU + 4 CX-7 NIC，1:1 映射：

| GPU | RDMA NIC |
|-----|----------|
| GPU 0 | mlx5_0 |
| GPU 1 | mlx5_1 |
| GPU 2 | mlx5_2 |
| GPU 3 | mlx5_3 |
