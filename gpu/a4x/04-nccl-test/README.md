# 5. NCCL 通信测试

本章覆盖 4 个层级的 NCCL 通信测试：单节点 NVLink、同域 MNNVL、跨域 RDMA、混合多节点。

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

| 指标 | 实测结果 |
|------|----------|
| all_reduce 8 GPU @8G (MNNVL) | **834.95 GB/s busbw** |

**同域 MNNVL 测试通过**：all_reduce 8GPU @8G 达到 834.95 GB/s busbw。ComputeDomain 管理的 IMEX daemon 正常工作，MNNVL 通信正常。

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

---

## 测试结果汇总

| 测试 | GPU 数 | 互联方式 | busbw (GB/s) |
|------|--------|----------|-------------|
| 单节点 4 GPU @8G | 4 | NVLink | **683.75** |
| 同域 2 节点 @8G | 8 | MNNVL (NVLink) | **834.95** |
| 跨域 2 节点 @8G | 8 | RDMA (GIB) | **325.88** |
| 混合 4 节点 @8G | 16 | RDMA (GIB) | **162.45** |
