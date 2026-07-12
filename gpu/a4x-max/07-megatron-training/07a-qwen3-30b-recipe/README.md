# Qwen3 30B-A3B MoE Training on GB300 NVL72 (A4X MAX)

Megatron Bridge + NeMo 26.06 容器，Qwen3 30B-A3B MoE 预训练 benchmark。

**GB200 baseline**: A4X (GCP) 实测 **914 TFLOP/s/GPU**，DGX-GB200 官方 **936-940 TFLOP/s/GPU**（NeMo Bridge 26.06）。
**GB300 NVIDIA 参考**: **1041 TFLOP/s/GPU** (+11% vs GB200)。

**参考链接**：
- GB200 完整 recipe: [a4x/07-megatron-training/07a-qwen3-30b-recipe/](../../../a4x/07-megatron-training/07a-qwen3-30b-recipe/)
- [Megatron Bridge Performance Summary](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html) — 官方 benchmark 数据
- [Qwen3 Workload Base Configs](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/scripts/performance/configs/qwen/qwen3_workload_base_configs.py) — Recipe 并行度配置

## 模型规格

| 参数 | 值 |
|---|---|
| 模型 | Qwen3-30B-A3B |
| 总参数 | 30B |
| 每 token 激活参数 | 3B |
| Expert 数量 | 128 routed + shared expert |
| TopK | 8 |
| 层数 | 48 |

## GB200 Baseline

| 指标 | A4X 实测 | DGX-GB200 官方 | 来源 |
|------|---------|---------------|------|
| Model TFLOP/s/GPU | **914** | **936-940** | NeMo Bridge 26.06 |
| Step Time | 6.60s | — | 2 节点 8 GPU |
| HBM Peak | 184.7 GiB | — | 接近 192 GB 上限 |
| CUDA Graph | full_iteration | full_iteration | 关键性能因素 |
| 差距 | -2.3% | baseline | — |

### GB200 优化迭代路径

```
run_script.py 无优化 (89) → + cutedsl (284) → + full CUDA Graph + env vars (914)
```

## GB300 vs GB200 参数对比

| 参数 | GB200 (baseline) | GB300 (本次) | 变化原因 |
|------|----------------|------------|---------|
| GPU 数 | 8 (2 节点) | 8 (2 节点) | 不变 |
| PP | 1 | 1 | 30B 模型小，无需 PP |
| EP | 8 | 8 | 不变（128 expert / 8 GPU = 16 expert/卡） |
| TP | 1 | 1 | 不变 |
| MBS | 4 | **8** | 288 GB HBM 允许翻倍 |
| GBS | 512 | **1024** | MBS 翻倍 + 保持 GA 不变 → GBS 翻倍 |
| seq_length | 4096 | 4096 | 不变 |
| 精度 | MXFP8 | MXFP8 | 不变（可选 NVFP4 进阶测试） |
| CUDA Graph | full_iteration | full_iteration | 不变 |
| cutedsl | Yes | Yes | 不变 |
| moe_flex_dispatcher | hybridep | hybridep | 不变 |
| EP_RANKS_PER_DOMAIN | 8 | 8 | 不变 |

### 为什么 MBS 可以翻倍

GB200 (192 GB): HBM Peak 184.7 GiB，MBS=4 几乎满载，MBS=8 会 OOM。
GB300 (288 GB): 多出 96 GB，MBS=8 的 activation 增量约 ~30 GB，288 GB 轻松容纳。

MBS 翻倍的收益:
- 更大矩阵让 Tensor Core 计算效率更高
- Grouped GEMM 的 batch 维度翻倍，throughput 提升

### 并行度计算

- 总 GPU: 8
- TP=1 × PP=1 × EP=8 × DP=1 = 8 GPU ✓
- GA: GBS / (MBS × EP) = 1024 / (8 × 8) = 16

## 前提条件

- 2 台 A4X MAX worker，同一 NVL72 域（同 Placement Policy）
- k8s 集群 + GPU Stack（device-plugin + DRA + DRANET + ComputeDomain）
- 容器: `nvcr.io/nvidia/nemo:26.06`

## Step 1: 部署训练 Pod

每个 worker 一个 Pod，使用 NeMo 26.06 容器 + GIB NCCL 插件。YAML 结构参考 GB200 recipe（见 [a4x/07a](../../../a4x/07-megatron-training/07a-qwen3-30b-recipe/)），注意以下 GB300 差异：

- ResourceClaimTemplate RDMA count: 4 → **8**（CX-8 双端口）
- nodeSelector: `nvidia-gb300`

## Step 2: 环境变量

```bash
source /usr/local/gib/scripts/set_nccl_env.sh

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1
export CUDNNFE_CLUSTER_OVERLAP_MARGIN=8
export NVLINK_DOMAIN_SIZE=72
export USE_MNNVL=1
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=8
export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128
export NCCL_CTA_POLICY=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=0
export NCCL_GRAPH_REGISTER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,graph_capture_record_stream_reuse:True
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
```

## Step 3: 启动训练

```bash
cd /opt/Megatron-Bridge/scripts/performance

torchrun --nproc_per_node=4 --nnodes=2 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_IP --master_port=29600 \
  run_script.py \
    -m qwen \
    -mr qwen3_30b_a3b \
    --task pretrain \
    -g gb300 \
    -c fp8_mx \
    -ng 8 \
    --data mock \
    --max_steps 20 \
    --log_dir /tmp/nemo-results \
    -wde bench \
    -wdj qwen3_30b \
    --micro_batch_size 8 \
    --global_batch_size 1024
```

> 注意: `-g gb300` 会加载 GB300 专属 recipe config。如果 Bridge 还没有 gb300 config，用 `-g gb200` 并手动覆盖 MBS 和 GBS。

### Recipe 自动加载的配置

| 配置 | GB200 值 | GB300 预期值 |
|---|---|---|
| EP | 8 | 8 |
| TP / PP | 1 / 1 | 1 / 1 |
| MBS / GBS | 4 / 512 | **8 / 1024** |
| seq_length | 4096 | 4096 |
| num_layers | 48 | 48 |
| cuda_graph_impl | full_iteration | full_iteration |
| moe_flex_dispatcher_backend | hybridep | hybridep |
| cutedsl_fused_grouped_mlp | True | True |

## 预期性能

| 指标 | GB200 实测 | DGX-GB200 官方 | GB300 NVIDIA 参考 | GB300 预期 |
|------|----------|---------------|-----------------|----------|
| TFLOP/s/GPU | 914 | 936-940 | **1041** | > 1000 |
| Step Time | 6.60s | — | — | < 6.0s |
| 提升幅度 | — | — | +11% vs GB200 | — |

### MBS 翻倍对性能的影响

基于 GB200 实测经验，MBS 翻倍通常带来 5-15% 计算效率提升（更大矩阵 Tensor Core 利用率更高）。结合 B300 Ultra 20% 更高算力，GB300 预期 TFLOP/s 应在 1000-1100 范围。

## GB300 实测结果

| Config | GPU 数 | MBS | GBS | TFLOP/s/GPU | Step Time | HBM Peak | 备注 |
|--------|-------|-----|-----|-------------|-----------|----------|------|
| — | — | — | — | — | — | — | 待测 |

## 原始日志

```
(待实测填入 20 步 per-step TFLOPs 日志)
```

## 进阶测试：NVFP4 精度

B300 Ultra 强化了 FP4 硬件加速。可选测试 NVFP4 精度对 MoE 模型的效果：

```bash
# NVFP4 测试（如果 Bridge 支持）
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_IP --master_port=29600 \
  run_script.py \
    -m qwen -mr qwen3_30b_a3b --task pretrain \
    -g gb300 -c fp4_nv -ng 8 --data mock \
    --max_steps 20 --log_dir /tmp/nemo-results \
    -wde bench -wdj qwen3_30b \
    --micro_batch_size 8 --global_batch_size 1024
```

### NVFP4 实测结果

| Config | 精度 | TFLOP/s/GPU | vs MXFP8 | 备注 |
|--------|------|-------------|----------|------|
| — | NVFP4 | — | — | 待测 |

## 注意事项

1. **`-g gb300` flag**: 必须使用 GB300 GPU type，让 recipe 加载 GB300 专属优化
2. **MBS=8 验证**: 先确认 MBS=8 不 OOM，否则退回 MBS=4（与 GB200 一致）
3. **RDMA count**: GB300 CX-8 双端口，ResourceClaimTemplate RDMA 数从 4 改为 8
4. **IPv6**: GB300 全栈 IPv6，确认 torchrun 和 Gloo 兼容
