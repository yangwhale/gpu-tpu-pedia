# 7. Megatron-LM 训练 — GB300 (A4X MAX)

以 GB200 (A4X) 实测数据为 baseline，在 GB300 NVL72 上对标训练 benchmark。

## GB300 vs GB200 核心硬件差异

| 维度 | GB200 (B200 Ultra) | GB300 (B300 Ultra) | 影响 |
|------|-------------------|-------------------|------|
| HBM 容量 | 192 GB/GPU | **288 GB/GPU** (+50%) | PP 减半、MBS 翻倍、EP 可降低 |
| FP8 峰值算力 | 4,500 TFLOP/s | **~5,400 TFLOP/s** (+20%) | 理论性能上限提升 |
| BF16 峰值算力 | 2,250 TFLOP/s | ~2,700 TFLOP/s | — |
| NVFP4 算力 | 有限支持 | **强化 FP4 加速** | Llama 405B NVFP4 3575 vs FP8 2646 (+35%) |
| NVLink 域 | NVL72 (18 节点/域) | NVL72 (18 节点/域) | 不变 |
| 每节点 GPU | 4 | 4 | 不变 |
| NIC | GVNIC + MRDMA (4 口) | **IDPF + MRDMA (CX-8)** | startup script v2、RDMA count=8 |
| 全栈 IPv6 | 否 | **是** | torchrun master_addr 可能需要 IPv6 格式 |

## GB300 训练参数调优核心原理

### 为什么 PP 可以减半

GB200: 192 GB/GPU → 94 层 Qwen3 235B 需要 PP=8（每 stage ~12 层, ~24 GB/stage）
GB300: 288 GB/GPU → PP=4 就够（每 stage ~24 层, ~48 GB/stage, 288 GB 容得下）

PP 减半的收益:
- Pipeline bubble 理论最大 overhead 从 (PP-1)/PP ≈ 87.5% 降到 75%
- 配合 VP interleaving 后实际 bubble 更小: PP=8 实测 ~30-50%, PP=4 降到 ~15-25%
- 这是 20-30% 性能提升的第一大来源

### 为什么 EP 可以降低

GB200: 192 GB → DSv3 256 experts, EP=64 → 每卡 4 experts（每 expert ~3 GB activation+weight）
GB300: 288 GB → EP=32 → 每卡 8 experts（每 expert ~3 GB, 总计 ~24 GB, 288 GB 轻松）

EP 降低的收益:
- All-to-all 通信量与 EP 成正比, EP 减半 → 通信减半
- HybridEP 在 NVLink 域内更高效

### 为什么 MBS 可以翻倍

GB200: 激活内存紧张, MBS=1 才能放下（大模型场景）
GB300: 多 96 GB, 可以 MBS=2, GPU 计算利用率更高（更大矩阵, Tensor Core 效率更高）

### NVFP4 新精度模式（可选进阶测试）

B300 Ultra 强化了 FP4 硬件加速。NVIDIA 官方 Llama 405B NVFP4 跑到 3575 TFLOP/s/GPU，比 FP8 的 2646 高 35%。MoE 模型的 NVFP4 效果待验证（expert 计算可能受益, 但 routing 精度敏感）。

## 测试环境

- 集群: tencent-gcp-taiji-poc (us-central1-b)
- 预留: nvidia-gb300-dxkhoz4ypk4mh (214 台, 856 GPU)
- 容器: NeMo 26.06 + Megatron Bridge
- 入口: `run_script.py`（Bridge 优化, 不要用 `pretrain_gpt.py`）
- 拓扑: NVL72 (18 节点/域), B300 Ultra (288 GB HBM3e/GPU)

## 参考链接

- [Megatron Bridge Performance Summary](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html) — 官方 benchmark 数据
- [Megatron Bridge Performance Tuning Guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-guide.html) — 性能调优指南
- GB200 Megatron 训练: [a4x/07-megatron-training/](../../a4x/07-megatron-training/)

## Recipe 列表

| Recipe | 模型 | 规模 | 文档 |
|--------|------|------|------|
| 07a | Qwen3 30B-A3B | 8 GPU | [07a-qwen3-30b-recipe/](07a-qwen3-30b-recipe/) |
| 07b | Qwen3 235B-A22B | 64-128 GPU | [07b-qwen3-235b-recipe/](07b-qwen3-235b-recipe/) |
| 07c | DeepSeek V3 671B | 64-128 GPU | [07c-deepseekv3-671b-recipe/](07c-deepseekv3-671b-recipe/) |

## 环境变量

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
# NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN 按模型 EP 设置
export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128

# GB300 特定
export NCCL_CTA_POLICY=1

# CUDA Graph 内存管理
export TORCH_NCCL_AVOID_RECORD_STREAMS=0
export NCCL_GRAPH_REGISTER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,graph_capture_record_stream_reuse:True

# LayerNorm SM margin
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
```

> **`TORCH_NCCL_AVOID_RECORD_STREAMS` 必须为 0**。CUDA Graph 模式下设 1 会导致 Graph 不生效，性能暴跌。配合 `graph_capture_record_stream_reuse:True` 使用。

## GB300 部署注意事项

1. **`-g gb300` flag**: Bridge recipe 需要 GB300 config。如果报错，用 `-g gb200` 并手动覆盖参数
2. **NIC 改名**: GB300 用 IDPF 不是 GVNIC，startup script 需用 v2 版本（自动检测）
3. **IPv6**: GB300 全栈 IPv6，torchrun master_addr 可能需要 IPv6 格式
4. **RDMA count**: ResourceClaimTemplate 从 4 改成 8（CX-8 双端口）
5. **DRA driver**: 推荐 v25.8.0+（GB300 CX-8 兼容）

## 全局对比表

详见各 recipe 文档和 [02-moe-training-tests](../02-moe-training-tests/) 中的全局对比表。
