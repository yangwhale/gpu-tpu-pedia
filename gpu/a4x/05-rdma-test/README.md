# 6. RDMA 带宽测试 (ib_write_bw)

在同域 NCCL 测试 Pod 中运行 RDMA 带宽测试，验证每块 CX-7 NIC 的原始 RDMA 带宽。

## 测试步骤

```bash
# 安装 perftest
kubectl exec nccl-sd-h1 -- bash -c "apt-get update -qq && apt-get install -y -qq perftest"
kubectl exec nccl-sd-h2 -- bash -c "apt-get update -qq && apt-get install -y -qq perftest"

# 测试时需切换 LD_LIBRARY_PATH 避免 GIB libibverbs 覆盖系统版本
# 在 h2 上启动 server（每个 RDMA NIC 一个端口）
for port in 18515 18516 18517 18518; do
  kubectl exec nccl-sd-h2 -- bash -c \
    "LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu ib_write_bw -p $port -d mlx5_0 -s 65536 --report_gbits -F" &
done

# 在 h1 上运行 client
HOST2_IP=$(kubectl get pod nccl-sd-h2 -o jsonpath='{.status.podIP}')
for port in 18515 18516 18517 18518; do
  kubectl exec nccl-sd-h1 -- bash -c \
    "LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu ib_write_bw -p $port -d mlx5_0 -s 65536 --report_gbits -F $HOST2_IP"
done
```

## 测试结果

| 指标 | 实测结果 |
|------|----------|
| ib_write_bw per NIC | **~381 Gbps** |
| 4×NIC aggregate | **~1524 Gbps (4 × ~381 Gbps)** |

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
