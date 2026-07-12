# DeepSeek V3 671B MoE Training on GB300 NVL72 (A4X MAX)

Megatron Bridge + NeMo 26.06 容器，DeepSeek V3 671B MoE 预训练 benchmark。

**GB200 baseline**:
- raw Megatron-LM (pretrain_gpt.py): **992 TFLOP/s/GPU** (32L 128 GPU, wgrad-defer)
- NeMo Bridge V2 (run_script.py): **1114-1124 TFLOP/s/GPU** (16L 64 GPU, full_iteration graph)
- NVIDIA 官方 (256 GPU): **1292 TFLOP/s/GPU** (GB200)

**GB300 NVIDIA 参考**: **1648 TFLOP/s/GPU** (+27.6% vs GB200, 256 GPU)

**参考链接**：
- GB200 完整 recipe: [a4x/07-megatron-training/07c-deepseekv3-671b-recipe/](../../../a4x/07-megatron-training/07c-deepseekv3-671b-recipe/)
- GB300 MoE 测试方案: [02-moe-training-tests/](../../02-moe-training-tests/)
- [Megatron Bridge Performance Summary](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html) — 官方 benchmark 数据
- [奚老师 DSv3 优化报告 v2](https://doc.maxwell-x.dev/dsv3-hybridep-128g-optimization-v2?t=A5st8MCjDgjk0pj8z_8bPw) — 40+ 组实验全记录

## 模型规格

| 参数 | 值 |
|---|---|
| 模型 | DeepSeek V3 |
| 总参数 | 671B |
| 每 token 激活参数 | ~37B |
| Expert 数量 | 256 routed, top-8 |
| 层数 | 61 (全量), 32 (缩减), 16 (最小) |
| hidden_size | 7168 |
| 架构特殊 | MLA (Multi-Latent Attention) + MTP (Multi-Token Prediction) |

## GB200 Baseline 汇总

### 实测结果

| Config | Framework | 层数 | GPU 数 | PP | EP | Graph | TFLOPs | 来源 |
|--------|-----------|------|-------|----|----|-------|--------|------|
| raw Megatron v3.1 + wgrad-defer | pretrain_gpt.py | 32L | 128 | 2 | 64 | TE scoped (attn) | **992** | 奚老师实测 |
| raw Megatron v3.1 | pretrain_gpt.py | 32L | 128 | 2 | 64 | TE scoped (attn) | 981 | 奚老师实测 |
| **NeMo Bridge V2** | **run_script.py** | **16L** | **64** | **2** | **32** | **full_iteration** | **1114-1176** | **实测 (MNNVL 0/2)** |
| NeMo Bridge V2 | run_script.py | 32L | 128 | — | — | full_iteration | **失败** | 9 轮全 crash/hang |

### NVIDIA 官方参考

| 平台 | GPU 数 | PP | EP | VP | MBS | TFLOPs |
|------|-------|----|----|----|----|--------|
| GB200 | 256 | 4 | 64 | 4 | 1 | **1292** |
| **GB300** | **256** | **2** | **32** | **8** | **1** | **1648** |

### GB200 核心优化路径

```
alltoall (300) → HybridEP (474) → CUDA graph partial (928) → mxfp8+fp32 optimizer (975)
→ MCore 0.18.0 (981) → wgrad-defer (992) → Bridge full_iteration graph (1114-1176, 64 GPU only)
```

## GB300 vs GB200 参数对比

| 参数 | GB200 (实测/官方) | GB300 (NVIDIA 官方) | 变化原因 |
|------|-----------------|-------------------|---------|
| PP | 4 (官方) / 2 (实测) | **2** | 288 GB → PP 可进一步减少 |
| VP | 4 (官方) | **8** | PP=2 配合更多 virtual stage |
| EP | 64 (实测) / 64 (官方) | **32** | 288 GB 每卡放 8 experts (vs 4) |
| TP | 1 | 1 | 不变 |
| MBS | 1 | 1 | DSv3 MoE expert buffer 是瓶颈，MBS 难提升 |
| GBS | 2048 | 2048 | 不变 |
| seq_length | 8192 | 8192 | 不变 |
| 精度 | MXFP8 | MXFP8 | 不变 |
| CUDA Graph | TE scoped (raw) / full_iteration (Bridge) | full_iteration | Bridge V2 |
| EP_RANKS_PER_DOMAIN | 64 | **32** | 匹配 EP=32 |

### EP 从 64 降到 32 的原理

GB200 (192 GB): 256 experts, EP=64 → 每卡 4 experts (每 expert ~3 GB activation+weight = ~12 GB)
GB300 (288 GB): EP=32 → 每卡 8 experts (~24 GB), 288 GB 轻松

EP 减半的收益:
- All-to-all 通信量与 EP 成正比, EP 减半 → 通信减半
- HybridEP 在更少的 rank 间通信效率更高
- 这是 GB300 DSv3 **27.6% 性能提升**的最大来源

### PP 从 4 降到 2

NVIDIA 官方 GB200 用 PP=4（61 层切 4 段），GB300 降到 PP=2（每 stage ~31 层, 288 GB 放得下）:
- Pipeline bubble 从 3/4=75% 降到 1/2=50%（理论最大）
- 配合 VP=8，实际 bubble 从 ~20% 降到 ~10%

### 为什么 MBS 不变

DSv3 的 256 expert + top-8 routing 在 HybridEP dispatch 时需要大量 buffer。MBS=1 的 expert buffer 已占用 ~50 GB，MBS=2 会 OOM。这是 DSv3 与 Qwen3 235B 的关键差异（Qwen3 128 expert, DSv3 256 expert）。

## 前提条件

- 16+ 台 A4X MAX worker（64+ GPU），同一 NVL72 域
- k8s 集群 + GPU Stack（device-plugin + DRA + DRANET + ComputeDomain）
- 容器: `nvcr.io/nvidia/nemo:26.06`

## 环境变量

```bash
source /usr/local/gib/scripts/set_nccl_env.sh

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1
export CUDNNFE_CLUSTER_OVERLAP_MARGIN=8
export NVLINK_DOMAIN_SIZE=72
export USE_MNNVL=1
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=32   # ← 匹配 EP=32
export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128
export NCCL_CTA_POLICY=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=0
export NCCL_GRAPH_REGISTER=0
export NCCL_NVLS_ENABLE=0                             # ← DSv3 必须关闭 NVLS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,graph_capture_record_stream_reuse:True
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
```

> **`NCCL_NVLS_ENABLE=0` 必须设置**。NVLS 在 DSv3 上有 iter 20-40 后性能渐降 30-50% 的已知 bug（GB200 实测确认，7 组对照实验）。

---

## Test 1: DSv3 16L — 64 GPU (16 节点, 单域)

缩减版 DSv3 (16 层, ~110B)，对标 GB200 同配置。

### 参数配置

| 参数 | GB200 (baseline) | GB300 (本次) | 变化原因 |
|------|----------------|------------|---------|
| 模型层数 | 16 | 16 | 不变 |
| PP | 2 | 2 | 16L PP=2 已最优 |
| VP | 2 | 2 | 不变 |
| EP | 32 (Bridge) / 64 (raw) | **32** | 288 GB 每卡更多 expert |
| TP | 1 | 1 | 不变 |
| MBS | 1 | 1 | DSv3 expert buffer 限制 |
| GBS | 512 | 512 | 不变 |
| seq_len | 8192 | 8192 | 不变 (DSv3 默认) |
| 精度 | MXFP8 | MXFP8 | 不变 |
| CUDA Graph | full_iteration (Bridge V2) | full_iteration | 不变 |
| EP_RANKS_PER_DOMAIN | 32 | 32 | 不变 |

### 启动命令

```bash
cd /opt/Megatron-Bridge/scripts/performance
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export GLOO_SOCKET_IFNAME=eth0 NCCL_SOCKET_IFNAME=eth0
export NCCL_MNNVL_ENABLE=2 NCCL_CUMEM_ENABLE=1
export USE_MNNVL=1
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=32

torchrun --nproc_per_node=4 --nnodes=16 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_IP --master_port=29600 \
  run_script.py \
    -m deepseek -mr deepseek_v3 --task pretrain \
    -g gb300 -c fp8_mx -ng 64 --data mock \
    --max_steps 20 --log_dir /tmp/nemo-results \
    -wde bench -wdj deepseek_v3 \
    --pipeline_model_parallel_size 2 \
    --expert_model_parallel_size 32 \
    --num_layers 16 \
    --virtual_pipeline_model_parallel_size 2 \
    --pipeline_model_parallel_layout "Etttt|tttt|tttt|ttttmL" \
    --global_batch_size 512 \
    --micro_batch_size 1
```

> **DSv3 改层数必须三件套同改**: `--num_layers` + `-vp` + `--pipeline_model_parallel_layout`（含 MTP `m` 层）。漏任何一个都会 assert 失败。

### 预期与实测

| 指标 | GB200 baseline | GB300 预期 | GB300 实测 |
|------|---------------|----------|----------|
| MNNVL=2 TFLOPs | 1176 | > 1400 (+20%) | — |
| MNNVL=0 TFLOPs | 1100 | > 1350 (+23%) | — |
| Step Time (MNNVL=2) | 2.22s | < 1.9s | — |

### 原始日志

```
(待实测填入 20 步 per-step TFLOPs 日志)
```

---

## Test 2: DSv3 32L — 128 GPU (32 节点, 跨域)

缩减版 DSv3 (32 层, ~221B)，对标 GB200 同配置 (992 raw Megatron / 1114 Bridge)。

### 参数配置

| 参数 | GB200 (baseline) | GB300 (本次) | 变化原因 |
|------|----------------|------------|---------|
| 模型层数 | 32 | 32 | 不变 |
| PP | 2 | 2 | 不变 |
| VP | — (raw) / 2 (Bridge) | **4** | GB300 配合 full_iteration graph |
| EP | 64 (raw) / 32 (Bridge) | **32** | 288 GB 每卡更多 expert |
| TP | 1 | 1 | 不变 |
| MBS | 1 | 1 | 不变 |
| GBS | 2048 | 2048 | 不变 |
| seq_len | 8192 | 8192 | 不变 |
| 精度 | MXFP8 | MXFP8 | 不变 |
| CUDA Graph | TE scoped (raw) / full_iteration (Bridge, 失败) | full_iteration | Bridge V2 |
| EP_RANKS_PER_DOMAIN | 64 (raw) / 32 (Bridge) | 32 | 匹配 EP=32 |

> **GB200 上 Bridge 128 GPU 失败**：9 轮实验全 crash/hang（PP interleaving 的 `torch.cuda.synchronize()` 不兼容 graph capture）。GB300 更大 HBM 可能缓解内存瓶颈，但 fundamental 不兼容需 NVIDIA 修复。如果 Bridge 仍失败，退回 raw Megatron-LM + wgrad-defer。

### 启动命令 (Bridge)

```bash
cd /opt/Megatron-Bridge/scripts/performance
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export GLOO_SOCKET_IFNAME=eth0 NCCL_SOCKET_IFNAME=eth0
export NCCL_MNNVL_ENABLE=2 NCCL_CUMEM_ENABLE=1
export USE_MNNVL=1
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=32

torchrun --nproc_per_node=4 --nnodes=32 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_IP --master_port=29600 \
  run_script.py \
    -m deepseek -mr deepseek_v3 --task pretrain \
    -g gb300 -c fp8_mx -ng 128 --data mock \
    --max_steps 20 --log_dir /tmp/nemo-results \
    -wde bench -wdj deepseek_v3 \
    -cv v2 \
    --pipeline_model_parallel_size 2 \
    --expert_model_parallel_size 32 \
    --num_layers 32 \
    --virtual_pipeline_model_parallel_size 4 \
    --global_batch_size 2048 \
    --micro_batch_size 1
```

### 启动命令 (raw Megatron-LM 备选)

如果 Bridge 128 GPU 仍 crash，退回 raw Megatron-LM + wgrad-defer:

```bash
cd /opt/Megatron-LM

torchrun --nproc_per_node=4 --nnodes=32 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_IP --master_port=29600 \
  pretrain_gpt.py \
    --use-mcore-models \
    --transformer-impl transformer_engine \
    --num-layers 32 --hidden-size 7168 \
    --num-attention-heads 128 --group-query-attention --num-query-groups 1 \
    --qk-head-dim 128 --qk-pos-emb-head-dim 64 --q-lora-rank 1536 --kv-lora-rank 512 \
    --v-head-dim 128 --qk-layernorm \
    --num-experts 256 --moe-ffn-hidden-size 2048 --moe-router-topk 8 \
    --moe-token-dispatcher-type alltoall --moe-grouped-gemm \
    --multi-head-latent-attention \
    --seq-length 8192 --max-position-embeddings 8192 \
    --micro-batch-size 1 --global-batch-size 2048 \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 2 \
    --expert-model-parallel-size 32 \
    --moe-router-dtype fp32 --moe-router-force-load-balancing \
    --position-embedding-type rope --rotary-base 10000 --rotary-percent 1.0 \
    --swiglu --normalization RMSNorm --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --attention-backend fused \
    --bf16 --grad-reduce-in-bf16 \
    --fp8-format hybrid --fp8-recipe delayed \
    --fp8-amax-history-len 1024 --fp8-amax-compute-algo max --fp8-param-gather \
    --mxfp8-precision e4m3 \
    --cross-entropy-loss-fusion --calculate-per-token-loss \
    --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
    --sequence-parallel \
    --ddp-average-in-collective --wgrad-deferral-limit -1 \
    --cuda-graph-impl transformer_engine --cuda-graph-modules attn \
    --recompute-granularity selective --recompute-modules moe_act mlp \
    --manual-gc --empty-unused-memory-level 1 \
    --train-iters 20 --lr 0.00015 --min-lr 0.00001 \
    --lr-decay-style cosine --lr-warmup-iters 5 \
    --clip-grad 1.0 --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.95 \
    --mock-data --tokenizer-type NullTokenizer --vocab-size 128256 \
    --no-create-attention-mask-in-dataloader --no-mmap-bin-files \
    --log-interval 1 --log-throughput --eval-iters 0 \
    --distributed-timeout-minutes 60
```

### 预期与实测

| 指标 | GB200 baseline | GB300 预期 | GB300 实测 |
|------|---------------|----------|----------|
| Bridge V2 TFLOPs | 1114 (64 GPU, Bridge) | > 1400 (+26%) | — |
| raw Megatron TFLOPs | 992 (128 GPU, wgrad-defer) | > 1200 (+21%) | — |
| NVIDIA GB300 256 GPU 参考 | — | 1648 (对标) | — |
| Step Time | — | — | — |

### 原始日志

```
(待实测填入 20 步 per-step TFLOPs 日志)
```

---

## Test 3: DSv3 Full 61L — 256 GPU (64 节点, 跨域, 对标 NVIDIA 官方)

全量 DSv3 671B（61 层），256 GPU 直接对标 NVIDIA 官方 benchmark 1648 TFLOP/s/GPU。

### 参数 (直接使用 NVIDIA 官方 GB300 recipe)

| 参数 | 值 | 备注 |
|------|------|------|
| 模型层数 | 61 | 全量模型 |
| PP | 2 | NVIDIA 官方 GB300 值 (GB200 需 PP=4) |
| VP | 8 | NVIDIA 官方 GB300 值 |
| EP | 32 | NVIDIA 官方 GB300 值 (GB200 需 EP=64) |
| TP | 1 | 不变 |
| MBS | 1 | DSv3 expert buffer 是瓶颈 |
| GBS | 4096 | 标准值 (可选 15360) |
| 精度 | MXFP8 | 对标官方 |
| CUDA Graph | full_iteration (V2) | 必须 |

**节点分配**: 64 节点 = 4 个 subblock (避开 rollback 中的 0007/0009)

> **61L PP=2 在 GB200 上 OOM**：GB200 192 GB 放不下 61L/PP=2（每 stage 30 层 × 256 expert），必须 PP=4。GB300 288 GB 可以，这是 GB300 最大的结构性优势。

### 启动命令

```bash
torchrun --nproc_per_node=4 --nnodes=64 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_IP --master_port=29600 \
  run_script.py -m deepseek -mr deepseek_v3 --task pretrain \
  -g gb300 -c fp8_mx -ng 256 --data mock \
  --max_steps 20 --log_dir /tmp/nemo-results \
  -wde bench -wdj deepseek_v3 \
  -cv v2 \
  --pipeline_model_parallel_size 2 \
  --expert_model_parallel_size 32 \
  --global_batch_size 4096 \
  --micro_batch_size 1
```

### 预期与实测

| 指标 | NVIDIA GB300 参考 | GB300 实测 |
|------|-----------------|----------|
| TFLOPs (GBS=4096) | **1648** | — |
| TFLOPs (GBS=15360) | **1670** | — |
| Tokens/s/GPU | 6338-6422 | — |

### 原始日志
```
(待实测填入 20 步 per-step TFLOPs 日志)
```

---

## 全局对比表

| # | 模型 | 层数 | 规模 | 平台 | PP | EP | VP | MBS | 精度 | TFLOPs | 来源 |
|---|------|------|------|------|----|----|----|-----|------|--------|------|
| 1 | DSv3 | 16L | 64 GPU | GB200 | 2 | 32 | 2 | 1 | MXFP8 | 1176 | 实测 (MNNVL=2, Bridge) |
| 2 | DSv3 | 16L | 64 GPU | GB200 | 2 | 32 | 2 | 1 | MXFP8 | 1100 | 实测 (MNNVL=0, Bridge) |
| 3 | DSv3 | 32L | 128 GPU | GB200 | 2 | 64 | — | 1 | MXFP8 | 992 | 实测 (raw Megatron, wgrad-defer) |
| 4 | DSv3 | 61L | 256 GPU | GB200 | 4 | 64 | 4 | 1 | MXFP8 | 1292 | NVIDIA 参考 |
| 5 | **DSv3** | **61L** | **256 GPU** | **GB300** | **2** | **32** | **8** | **1** | MXFP8 | **1648** | **NVIDIA 参考 (+27.6%)** |
| 6 | **DSv3** | **16L** | **64 GPU** | **GB300** | **2** | **32** | **2** | **1** | MXFP8 | **—** | **待测 (Test 1)** |
| 7 | **DSv3** | **32L** | **128 GPU** | **GB300** | **2** | **32** | **4** | **1** | MXFP8 | **—** | **待测 (Test 2)** |
| 8 | **DSv3** | **61L** | **256 GPU** | **GB300** | **2** | **32** | **8** | **1** | MXFP8 | **—** | **待测 (Test 3, 对标 NVIDIA)** |

## GB300 DSv3 性能提升来源分析

NVIDIA GB300 相比 GB200 提升 27.6%（1292→1648），主要来自三个因素：

### 1. EP 减半 (64→32, 最大贡献 ~15%)

256 expert + EP=64 时每卡 4 expert，all-to-all 通信量大（64 路 dispatch/combine）。EP=32 每卡 8 expert，通信量减半，HybridEP 域内效率更高。DSv3 是 expert-heavy 模型（256 expert, H=7168），EP 通信是主要瓶颈。

### 2. PP 减半 (4→2, ~8%)

PP=4 → PP=2 减少 pipeline bubble。配合 VP=8（vs VP=4），bubble 进一步降低。

### 3. 算力提升 (~5%)

B300 Ultra FP8 峰值算力比 B200 Ultra 高约 20%，但受限于 memory bandwidth 和通信，实际提升约 5%。

## DSv3 特有踩坑记录

### PP layout 三件套

DSv3 有 MTP（Multi-Token Prediction）层，PP layout 必须包含 `m` 标记。改层数必须同时改三个参数：

```
--num_layers 16 → --virtual_pipeline_model_parallel_size 2 → --pipeline_model_parallel_layout "Etttt|tttt|tttt|ttttmL"
--num_layers 32 → --virtual_pipeline_model_parallel_size 4 → (layout 由 Bridge 计算)
```

### NVLS 退化 bug

`NCCL_NVLS_ENABLE=1` 在 DSv3 上 iter 20-40 后性能渐降 30-50%，已通过 7 组 GB200 对照实验确认。**必须设 `NCCL_NVLS_ENABLE=0`**。

### 3 个致命参数

| 参数 | 必须值 | 错误值后果 |
|------|-------|----------|
| `--cuda-graph-impl transformer_engine` | 必须显式设 | 漏掉 → graph 静默禁用 (981→836) |
| `NCCL_GRAPH_REGISTER` | 0 | 1 → AssertionError crash |
| `NCCL_NVLS_ENABLE` | 0 | 1 → iter 20+ 后性能渐降 30-50% |

## 注意事项

1. **`-g gb300` flag**: Bridge recipe 需要 GB300 GPU type。如果报错用 `-g gb200` 手动覆盖参数
2. **Bridge 128 GPU 可能失败**: GB200 上 9 轮全 crash/hang。GB300 HBM 更大可能缓解部分问题，但 fundamental 不兼容仍在。准备好 raw Megatron-LM 备选方案
3. **DSv3 PP layout**: 改层数必须三件套同改 (`--num_layers` + `-vp` + `--pipeline_model_parallel_layout`)
4. **NVLS**: 必须 `NCCL_NVLS_ENABLE=0`
5. **Recompute**: raw Megatron-LM 方案需要 `--recompute-granularity selective --recompute-modules moe_act mlp`
6. **wgrad-defer**: raw Megatron-LM 最佳配置需要 `--ddp-average-in-collective --wgrad-deferral-limit -1`（+4.4%）
