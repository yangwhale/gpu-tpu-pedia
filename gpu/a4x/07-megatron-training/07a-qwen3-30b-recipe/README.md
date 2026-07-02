# Qwen3 30B-A3B MoE Training on GB200 NVL72 (A4X)

Megatron Bridge + NeMo 26.06 容器，Qwen3 30B-A3B MoE 预训练 benchmark。

**结果**：8 GPU (2 节点) 达到 **866 TFLOP/s/GPU**，官方 DGX-GB200 为 936（差 7.5%，因 A4X 需 selective recompute 换空间给 CUDA Graph）。

## 前提条件

- 2+ 台 A4X worker，同一 NVL72 域（同 Placement Policy）
- k8s 1.34+ 集群 + GPU Stack（device-plugin + DRA + DRANET + ComputeDomain）
- Worker 镜像：`chrisya-a4x-worker-v3`（kernel 锁定 + NVIDIA 580 + Lustre + IMEX）

## Step 1: 部署 NeMo 26.06 训练 Pod

每个 worker 一个 Pod，使用 `nvcr.io/nvidia/nemo:26.06` 容器 + GIB v1.1.2 NCCL 插件。

```yaml
# Pod spec 关键字段（完整 YAML 见 yamls/nemo-train-2node.yaml）
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

> **GIB 安装方式**：用 `LD_PRELOAD` 加载 GIB 的 libnccl，并将 libibverbs/libmlx5/librdmacm 拷贝到 GIB 目录（保留 RDMA 库，不禁用）。

## Step 2: 环境变量（关键）

以下环境变量在 Slurm launcher 的 `perf_plugins.py` 中自动设置，torchrun 直接跑必须手动设：

```bash
# === GIB NCCL ===
source /usr/local/gib/scripts/set_nccl_env.sh

# === 通用 ===
export CUDA_DEVICE_MAX_CONNECTIONS=1

# === CuTeDSL fused grouped MLP（最大单项优化） ===
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1
export CUDNNFE_CLUSTER_OVERLAP_MARGIN=8

# === NVL72 域配置（hybridep 必需） ===
export NVLINK_DOMAIN_SIZE=72
export USE_MNNVL=1
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=8
export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128

# === GB200 特定 ===
export NCCL_CTA_POLICY=1

# === CUDA Graph 内存管理 ===
export TORCH_NCCL_AVOID_RECORD_STREAMS=0       # 注意：CUDA Graph 模式下必须=0
export NCCL_GRAPH_REGISTER=0                     # 解决 expandable_segments + CG 冲突
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,graph_capture_record_stream_reuse:True

# === LayerNorm SM margin（DP comm overlap） ===
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
```

> **`TORCH_NCCL_AVOID_RECORD_STREAMS` 陷阱**：不开 CUDA Graph 时设 `=1` 省内存。开 CUDA Graph 时必须设 `=0` + `graph_capture_record_stream_reuse:True`。设反了 CUDA Graph 不生效，性能从 866 跌到 284。

## Step 3: 启动训练

使用 `run_script.py`（不是 `run_recipe.py`，后者不加载 GPU 特定优化配置）。

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
    --recompute_modules core_attn,layernorm,moe_act \
    --log_dir /tmp/nemo-results \
    -wde bench \
    -wdj qwen3_30b_866
```

### 参数说明

| 参数 | 值 | 说明 |
|---|---|---|
| `-m qwen -mr qwen3_30b_a3b` | — | 模型 family + recipe 名 |
| `-g gb200` | — | 目标 GPU 类型（加载 GB200 特定配置） |
| `-c fp8_mx` | — | MXFP8 精度（NVIDIA 推荐，优于 BF16 和传统 FP8） |
| `-ng 8` | — | 总 GPU 数 |
| `--recompute_modules core_attn,layernorm,moe_act` | — | Selective recompute，省 ~10 GiB 给 CUDA Graph |
| `--data mock` | — | Mock data（benchmark 用） |

### Recipe 自动加载的配置

`run_script.py` 从 `QWEN3_30B_A3B_PRETRAIN_CONFIG_GB200_FP8_MX_V1` 加载：

| 配置 | 值 | 说明 |
|---|---|---|
| EP | 8 | Expert Parallelism |
| TP/PP | 1/1 | 无 Tensor/Pipeline 并行 |
| MBS | 4 | Micro Batch Size |
| GBS | 512 | Global Batch Size |
| seq_length | 4096 | 序列长度 |
| num_layers | 48 | 完整模型 |
| cuda_graph_impl | full_iteration | 整个 iteration CUDA Graph |
| moe_flex_dispatcher_backend | hybridep | NVL72 优化的 MoE dispatcher |
| moe_a2a_overlap | True | 计算通信重叠 |
| cutedsl_fused_grouped_mlp | True | CuTe DSL 融合 grouped MLP |
| fp8_dot_product_attention | True | FP8 attention |

## 性能结果

| 指标 | 值 |
|---|---|
| **Model TFLOP/s/GPU** | **866** |
| Step Time | 6.97s |
| HBM Peak | 184 GiB (0 alloc retry) |
| MFU (BF16 基准 2250 TFLOP/s) | 38.5% |

### 与官方 DGX-GB200 对比

| 维度 | A4X (本文) | DGX-GB200 (官方) |
|---|---|---|
| TFLOP/s/GPU | 866 | 936 |
| 差距 | -7.5% | baseline |
| Recompute | core_attn,layernorm,moe_act | 无 |
| CUDA Graph | full_iteration | full_iteration |
| HBM 可用 | 184 GiB | 184 GiB |

> 7.5% 差距完全来自 selective recompute 的额外计算开销。A4X 必须开 recompute 才能让 CUDA Graph fit 进 184 GiB。

## 优化迭代记录

从 89 到 866 的完整迭代路径：

| 优化 | TFLOP/s | 增幅 | 关键发现 |
|---|---|---|---|
| Baseline (run_script.py) | 89 | — | 正确入口但缺环境变量 |
| + TE CUDA Graph | 208 | +134% | `NCCL_GRAPH_REGISTER=0` 解决 CG + expandable_segments 冲突 |
| + cutedsl fused MLP | 284 | +37% | `NVTE_CUTEDSL_FUSED_GROUPED_MLP=1` 单项最大优化 |
| + NVL72 domain env vars | 294 | +3% | NVLINK_DOMAIN_SIZE + USE_MNNVL |
| + full CG + recompute + 正确 env | **866** | +194% | `AVOID_RECORD_STREAMS=0` + `graph_capture_record_stream_reuse:True` |

### 走过的弯路

| 尝试 | 结果 | 教训 |
|---|---|---|
| `run_recipe.py` | 40 TFLOP/s | 不加载 GPU 特定配置，必须用 `run_script.py` |
| VBoost (`nvidia-smi boost-slider --vboost 1`) | 190（-9%） | 反而变慢，可能触发热限制 |
| `TORCH_NCCL_AVOID_RECORD_STREAMS=1` + CUDA Graph | 208 | **设反了！** CG 模式下必须=0 |
| full_iteration CG 不开 recompute | OOM | 184 GiB 放不下 CG replay buffer |
| cutedsl + TE CG 组合 | crash | Triton kernel CPU tensor 兼容性问题 |
| activation offload to CPU | crash | `moe_paged_stash` 与 `cpu_offloading` 互斥 |
| full recompute | assertion | `full` recompute 与 `overlap_moe_expert_parallel_comm` 互斥 |

## 常见问题

**Q: 为什么用 `run_script.py` 不用 `run_recipe.py`？**
A: `run_recipe.py` 不调用 `get_perf_optimized_recipe()`，不加载 GPU 特定的并行/CG/dispatcher 配置。`run_script.py` 调用完整的 recipe 加载链。

**Q: 不开 recompute 能跑吗？**
A: A4X 上不行，CUDA Graph replay buffer 需要 ~10 GiB，不开 recompute 超出 184 GiB。DGX-GB200 据报也是 184 GiB 可用，但官方 benchmark 不开 recompute。可能存在我们未知的内存优化或容器差异。

**Q: 能扩到 8 节点 32 GPU 吗？**
A: 需要验证。之前在旧集群上 32 GPU EP=32 遇到 CUDA error 801（不同域的 IMEX 冲突）。同域 8 节点应该可行，需确保所有节点在同一 Placement Policy 且无 IMEX 冲突。
