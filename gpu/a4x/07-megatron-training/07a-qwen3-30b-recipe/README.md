# Qwen3 30B-A3B MoE Training on GB200 NVL72 (A4X)

Megatron Bridge + NeMo 26.06 容器，Qwen3 30B-A3B MoE 预训练 benchmark。

**结果**：8 GPU (2 节点) 达到 **914 TFLOP/s/GPU**，官方 DGX-GB200 为 936（差 2.3%）。100% 复刻官方 recipe，无 recompute。

## 前提条件

- 2+ 台 A4X worker，同一 NVL72 域（同 Placement Policy）
- k8s 1.34+ 集群 + GPU Stack（device-plugin + DRA + DRANET + ComputeDomain）
- Worker 镜像：`chrisya-a4x-worker-v3`（kernel 锁定 + NVIDIA 580 + Lustre + IMEX）

## Step 1: 部署 NeMo 26.06 训练 Pod

每个 worker 一个 Pod，使用 `nvcr.io/nvidia/nemo:26.06` 容器 + GIB v1.1.2 NCCL 插件。

```yaml
containers:
- image: nvcr.io/nvidia/nemo:26.06
  resources:
    limits: {nvidia.com/gpu: 4}
    claims: [{name: cd}]   # ComputeDomain channel
  env:
  - {name: LD_PRELOAD, value: "/usr/local/gib/lib64/libnccl.so.2"}
  - {name: LD_LIBRARY_PATH, value: "/usr/local/gib/lib64:/usr/local/nvidia/lib64"}
  - {name: NCCL_MNNVL_ENABLE, value: "2"}
  - {name: NCCL_CUMEM_ENABLE, value: "1"}
initContainers:
- name: gib-installer
  image: us-docker.pkg.dev/gce-ai-infra/gpudirect-gib/nccl-plugin-gib-diagnostic-arm64:v1.1.2
  args:
  - |
    /scripts/container_entry.sh install --install-nccl
    cp -a /usr/local/gib/. /target/gib/
    cp -a /usr/lib/aarch64-linux-gnu/libibverbs.so* /target/gib/lib64/
    cp -a /usr/lib/aarch64-linux-gnu/libmlx5.so* /target/gib/lib64/
    cp -a /usr/lib/aarch64-linux-gnu/librdmacm.so* /target/gib/lib64/
    mkdir -p /target/gib/lib64/libibverbs
    cp -a /usr/lib/aarch64-linux-gnu/libibverbs/libmlx5-rdmav34.so /target/gib/lib64/libibverbs/ 2>/dev/null || true
resourceClaims:
- {name: cd, resourceClaimTemplateName: cd-chrisya-channel}
```

## Step 2: 环境变量

Megatron Bridge 的 Slurm launcher (`perf_plugins.py`) 自动设置以下变量。用 torchrun 直接跑必须手动设：

```bash
source /usr/local/gib/scripts/set_nccl_env.sh

export CUDA_DEVICE_MAX_CONNECTIONS=1

# CuTeDSL fused grouped MLP
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1
export CUDNNFE_CLUSTER_OVERLAP_MARGIN=8

# NVL72 域配置（hybridep 必需）
export NVLINK_DOMAIN_SIZE=72
export USE_MNNVL=1
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=8
export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128

# GB200 特定
export NCCL_CTA_POLICY=1

# CUDA Graph 内存管理
export TORCH_NCCL_AVOID_RECORD_STREAMS=0
export NCCL_GRAPH_REGISTER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,graph_capture_record_stream_reuse:True

# LayerNorm SM margin
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
```

> **`TORCH_NCCL_AVOID_RECORD_STREAMS` 必须为 0**。CUDA Graph 模式下设 1 会导致 Graph 不生效，性能从 914 跌到 284。配合 `graph_capture_record_stream_reuse:True` 使用。

## Step 3: 启动训练

使用 `run_script.py`（不是 `run_recipe.py`）。

```bash
cd /opt/Megatron-Bridge/scripts/performance

torchrun --nproc_per_node=4 --nnodes=2 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_IP --master_port=29600 \
  run_script.py \
    -m qwen \
    -mr qwen3_30b_a3b \
    --task pretrain \
    -g gb200 \
    -c fp8_mx \
    -ng 8 \
    --data mock \
    --max_steps 20 \
    --log_dir /tmp/nemo-results \
    -wde bench \
    -wdj qwen3_30b
```

### Recipe 自动加载的配置

| 配置 | 值 |
|---|---|
| EP | 8 |
| TP / PP | 1 / 1 |
| MBS / GBS | 4 / 512 |
| seq_length | 4096 |
| num_layers | 48（完整模型） |
| cuda_graph_impl | full_iteration |
| moe_flex_dispatcher_backend | hybridep |
| moe_a2a_overlap | True |
| cutedsl_fused_grouped_mlp | True |
| fp8_dot_product_attention | True |

## 性能结果

| 指标 | A4X (GCP) | DGX-GB200 (官方) |
|---|---|---|
| **Model TFLOP/s/GPU** | **914** | **936** |
| 差距 | -2.3% | baseline |
| Step Time | 6.60s | — |
| HBM Peak | 184.7 GiB | — |
| Alloc Retries | 0 | — |

## 优化迭代总结

从 89 到 914 的关键步骤：

| 阶段 | TFLOP/s | 关键操作 |
|---|---|---|
| 1. 正确入口 | 89 | 用 `run_script.py` 不是 `run_recipe.py` |
| 2. + cutedsl | 284 | `NVTE_CUTEDSL_FUSED_GROUPED_MLP=1` |
| 3. + full CUDA Graph | **914** | `AVOID_RECORD_STREAMS=0` + `graph_capture_record_stream_reuse:True` + NVL72 env vars |

### 核心教训

1. **`run_recipe.py` vs `run_script.py`**：前者不加载 GPU 特定优化配置（CUDA Graph、hybridep、cutedsl 等），只有后者调 `get_perf_optimized_recipe()` 完整加载
2. **Slurm 环境变量**：`perf_plugins.py` 为 Slurm executor 自动设 ~15 个环境变量。torchrun 跑不经过 Slurm，全部漏掉。最关键的是 `NVTE_CUTEDSL_FUSED_GROUPED_MLP=1` 和 NVL72 domain 变量
3. **CUDA Graph + AVOID_RECORD_STREAMS**：这个变量在非 CG 模式设 1 省内存，CG 模式必须设 0。设反了 CUDA Graph 静默失效（不报错，只是慢）
4. **CUDA Graph 对 MoE 的作用**：Qwen3 30B 有 128 expert × 48 层，每步上万个 CUDA kernel launch。CUDA Graph 把 host overhead 从毫秒级降到微秒级，step time 从 21s 降到 6.6s（3.2×）
