> 🌐 [中文](README.md) | **English**

# 6. RDMA Bandwidth Test (ib_write_bw)

Run an RDMA bandwidth test inside the same-domain NCCL test Pods to verify the raw RDMA bandwidth of each CX-7 NIC.

## Test Steps

You can reuse the same-domain NCCL test Pods (the GIB diagnostic image ships with `perftest`, so no extra installation is needed).

```bash
# Set Pod names (reuse the NCCL test Pods, or any 2 Pods in the same domain)
POD0=nccl-2n-g1-0   # server
POD1=nccl-2n-g1-1   # client
POD0_IP=$(kubectl get pod $POD0 -o jsonpath='{.status.podIP}')

# Test the 4 CX-7 RDMA NICs one by one
for i in 0 1 2 3; do
  NIC=mlx5_$i
  PORT=$((18515 + i))
  echo "=== Testing $NIC ==="

  # Server (background)
  kubectl exec $POD1 -- bash -c \
    "LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu ib_write_bw -p $PORT -d $NIC -s 65536 --report_gbits -F -D 5" &
  sleep 2

  # Client
  kubectl exec $POD0 -- bash -c \
    "LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu ib_write_bw -p $PORT -d $NIC -s 65536 --report_gbits -F -D 5 $POD0_IP"

  wait
done
```

## Test Results

| NIC | BW avg (Gbps) |
|------|----------|
| mlx5_0 | **382.10** |
| mlx5_1 | **382.12** |
| mlx5_2 | **382.19** |
| mlx5_3 | **382.15** |
| 4×NIC aggregate | **~1528 Gbps** |

Measured 382.1–382.2 Gbps per NIC, consistent with the nominal reference of ~381 Gbps (measured 2026-06-27). Every CX-7 NIC reaches 400GbE line rate (theoretical ~400 Gbps; the actual ~382 Gbps is normal once protocol overhead is accounted for).

## Notes

### IBVERBS Path Conflict

GIB's `libibverbs` library (`/usr/local/gib/lib64/`) may override the system version. Switch the path before running `ib_write_bw`:

```bash
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu
```

### A4X NIC Mapping

Each A4X node has 4 GPUs + 4 CX-7 NICs, mapped 1:1:

| GPU | RDMA NIC |
|-----|----------|
| GPU 0 | mlx5_0 |
| GPU 1 | mlx5_1 |
| GPU 2 | mlx5_2 |
| GPU 3 | mlx5_3 |
