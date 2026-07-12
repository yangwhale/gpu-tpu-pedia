# Qwen3 235B-A22B MoE Training on GB300 NVL72 (A4X MAX)

Megatron Bridge + NeMo 26.06 容器，Qwen3 235B-A22B MoE 预训练 benchmark。

**GB200 baseline**:
- V1 (PP=8 EP=8, 64 GPU 单域): **930 TFLOP/s/GPU**
- V2 (PP=2 EP=32, 64 GPU 跨域): **1124 TFLOP/s/GPU** — full_iteration graph + paged stash
- NVIDIA 官方 (256 GPU): **1092 TFLOP/s/GPU** (GB200), **1335 TFLOP/s/GPU** (GB300, +22.3%)

**参考链接**：
- GB200 完整 recipe: [a4x/07-megatron-training/07b-qwen3-235b-recipe/](../../../a4x/07-megatron-training/07b-qwen3-235b-recipe/)
- GB300 MoE 测试方案: [02-moe-training-tests/](../../02-moe-training-tests/)
- [Megatron Bridge Performance Summary](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html) — 官方 benchmark 数据
- [Qwen3 Workload Base Configs](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/scripts/performance/configs/qwen/qwen3_workload_base_configs.py) — Recipe 并行度配置

## 模型规格

| 参数 | 值 |
|---|---|
| 模型 | Qwen3-235B-A22B |
| 总参数 | 235B |
| 每 token 激活参数 | 22B |
| Expert 数量 | 128 routed |
| TopK | 8 |
| 层数 | 94 |

## GB200 Baseline 汇总

### 实测结果

| Config | GPU 数 | 拓扑 | PP | EP | VP | Graph | TFLOPs | Step Time |
|--------|-------|------|----|----|-------|-------|--------|-----------|
| V1 baseline | 64 | 单域 | 8 | 8 | — | TE scoped | **930** | 6.47s |
| PP=2 EP=32 跨域 V1 | 64 | 跨域 | 2 | 32 | — | TE scoped | 685 | 7.1s |
| **PP=2 EP=32 跨域 V2** | **64** | **跨域** | **2** | **32** | **—** | **full_iteration** | **1124** | **4.31s** |

### NVIDIA 官方参考

| 平台 | GPU 数 | PP | EP | VP | MBS | tok/s/GPU | TFLOP/s/GPU |
|------|-------|----|----|----|----|-----------|-------------|
| GB200 (V2) | 256 | 8 | 32 | 3 | 1 | 7376 | **1092** |
| **GB300 (V2)** | **256** | **4** | **32** | **12** | **2** | **9015** | **1335** |

## GB300 vs GB200 参数对比

| 参数 | GB200 (V1, 实测) | GB200 (V2 NVIDIA, 256 GPU) | GB300 (NVIDIA, 256 GPU) | 变化原因 |
|------|-----------------|--------------------------|------------------------|---------|
| PP | 8 | 8 | **4** | 288 GB → 每 stage 24 层 (vs 12 层) |
| VP | — | 3 | **12** | PP=4 + VP=12 → 更细粒度 interleaving |
| EP | 8 | 32 | 32 | 不变 |
| TP | 1 | 1 | 1 | 不变 |
| MBS | 1 | 1 | **2** | 288 GB 可放更大 micro batch |
| GBS | 1024 | 8192 | 8192 | 不变 (256 GPU) |
| seq_length | 4096 | 4096 | 4096 | 不变 |
| 精度 | MXFP8 | MXFP8 | MXFP8 | 不变 |
| CUDA Graph | TE scoped (V1) | full_iteration (V2) | full_iteration (V2) | 不变 |
| moe_paged_stash | False (V1) | True (V2) | True (V2) | 不变 |

### PP 减半的原理

GB200 (192 GB): 94 层 Qwen3 235B 需要 PP=8（每 stage ~12 层, 每 stage 权重+optimizer ~24 GB）
GB300 (288 GB): PP=4 就够（每 stage ~24 层, 每 stage ~48 GB, 288 GB 容得下）

PP 从 8 降到 4 的收益:
- Pipeline bubble 理论 overhead: PP=8 是 7/8=87.5%, PP=4 是 3/4=75%
- 配合 VP=12 interleaving: PP=8 VP=3 实测 bubble ~30-50%, PP=4 VP=12 预计 ~10-15%
- 结合 full_iteration graph, 预期 **20-30% 性能提升**

### VP 为什么从 3 增到 12

VP（Virtual Pipeline）把每个 PP stage 进一步切分为多个 virtual stage，减少 pipeline bubble:
- GB200: PP=8, VP=3, 94/8/3 ≈ 4 层/virtual stage — 但 94 不整除 8×3=24
- GB300: PP=4, VP=12, 94 不整除 4×12=48 — Bridge 支持不均匀切分

VP 越大 bubble 越小，但通信次数更多。PP=4 VP=12 是 GB300 的最优平衡点。

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
export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128
export NCCL_CTA_POLICY=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=0
export NCCL_GRAPH_REGISTER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,graph_capture_record_stream_reuse:True
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
```

## Test 1: 64 GPU (16 节点, 单域)

全量 94 层 Qwen3 235B MoE 模型。GB300 288 GB 显存允许 PP 从 8 降到 4。

### 参数配置

| 参数 | GB200 V1 (baseline) | GB300 (本次) | 变化原因 |
|------|-------------------|------------|---------|
| PP | 8 | **4** | 288 GB → 每 stage 24 层 |
| VP | — | **12** | 更细粒度 interleaving |
| EP | 8 | 8 | 不变 |
| TP | 1 | 1 | 不变 |
| MBS | 1 | **2** | 288 GB 可放更大 micro batch |
| GBS | 1024 | 1024 | 不变 |
| seq_length | 4096 | 4096 | 不变 |
| CUDA Graph | TE scoped (V1) | **full_iteration (V2)** | PP=4 更适合 full graph |
| Config variant | V1 | **V2** | 启用 full_iteration + paged stash |
| EP_RANKS_PER_DOMAIN | 8 | 8 | 不变 |

**并行度验证**: TP=1 × PP=4 × EP=8 × DP=2 = 64 GPU ✓

### 启动命令

```bash
cd /opt/Megatron-Bridge/scripts/performance
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export GLOO_SOCKET_IFNAME=eth0 NCCL_SOCKET_IFNAME=eth0
export NCCL_MNNVL_ENABLE=2 NCCL_CUMEM_ENABLE=1
export USE_MNNVL=1
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=8

torchrun --nproc_per_node=4 --nnodes=16 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_IP --master_port=29600 \
  run_script.py \
    -m qwen -mr qwen3_235b_a22b --task pretrain \
    -g gb300 -c fp8_mx -ng 64 --data mock \
    --max_steps 20 --log_dir /tmp/nemo-results \
    -wde bench -wdj qwen3_235b \
    -cv v2 \
    --pipeline_model_parallel_size 4 \
    --expert_model_parallel_size 8 \
    --virtual_pipeline_model_parallel_size 12 \
    --global_batch_size 1024 \
    --micro_batch_size 2
```

### 预期与实测

| 指标 | GB200 V1 (930) | GB200 V2 (1124) | NVIDIA GB300 256 GPU | GB300 64 GPU 预期 | GB300 实测 |
|------|---------------|-----------------|---------------------|-----------------|----------|
| TFLOPs | 930 | 1124 | 1335 | > 1200 | — |
| Step Time | 6.47s | 4.31s | — | < 4.0s | — |
| 提升 vs GB200 V1 | baseline | +21% | +43% | +29% | — |

### 原始日志

```
(待实测填入 20 步 per-step TFLOPs 日志)
```

---

## Test 2: 128 GPU (32 节点, 跨域)

全量 94 层 Qwen3 235B，扩展到 128 GPU 验证 scaling。

### 参数配置

| 参数 | GB200 参考 (256 GPU) | GB300 (本次, 128 GPU) | 变化原因 |
|------|-------------------|---------------------|---------|
| PP | 8 | **4** | 288 GB + PP=4 足够 |
| VP | 3 | **12** | 更多 interleaving |
| EP | 32 | 32 | 128/(TP×PP)=32 |
| TP | 1 | 1 | 不变 |
| MBS | 1 | **2** | 288 GB 允许 |
| GBS | 8192 | **4096** | 按 128 GPU 缩放 |
| 精度 | MXFP8 | MXFP8 | 不变 |
| Config variant | V2 | V2 | 不变 |
| EP_RANKS_PER_DOMAIN | 32 | 32 | 匹配 EP=32 |

**并行度验证**: TP=1 × PP=4 × EP=32 × DP=1 = 128 GPU ✓

### 启动命令

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
    -m qwen -mr qwen3_235b_a22b --task pretrain \
    -g gb300 -c fp8_mx -ng 128 --data mock \
    --max_steps 20 --log_dir /tmp/nemo-results \
    -wde bench -wdj qwen3_235b \
    -cv v2 \
    --pipeline_model_parallel_size 4 \
    --expert_model_parallel_size 32 \
    --virtual_pipeline_model_parallel_size 12 \
    --global_batch_size 4096 \
    --micro_batch_size 2
```

### 预期与实测

| 指标 | NVIDIA GB300 256 GPU 参考 | GB300 128 GPU 预期 | GB300 实测 |
|------|------------------------|------------------|----------|
| TFLOPs | 1335 | > 1200 (scaling ~90%) | — |
| Tokens/s/GPU | 9015 | > 8000 | — |
| Step Time | — | — | — |

### 原始日志

```
(待实测填入 20 步 per-step TFLOPs 日志)
```

---

## 全局对比表

| # | 模型 | 规模 | 平台 | PP | EP | VP | MBS | 精度 | TFLOPs | 来源 |
|---|------|------|------|----|----|----|----|------|--------|------|
| 1 | Qwen3 235B | 64 GPU | GB200 | 8 | 8 | — | 1 | MXFP8 | 930 | 实测 (V1) |
| 2 | Qwen3 235B | 64 GPU | GB200 | 2 | 32 | — | 1 | MXFP8 | 1124 | 实测 (V2, 跨域) |
| 3 | Qwen3 235B | 256 GPU | GB200 | 8 | 32 | 3 | 1 | MXFP8 | 1092 | NVIDIA 参考 |
| 4 | Qwen3 235B | 256 GPU | **GB300** | **4** | **32** | **12** | **2** | MXFP8 | **1335** | NVIDIA 参考 (+22.3%) |
| 5 | Qwen3 235B | 64 GPU | **GB300** | **4** | **8** | **12** | **2** | MXFP8 | **—** | **待测 (Test 1)** |
| 6 | Qwen3 235B | 128 GPU | **GB300** | **4** | **32** | **12** | **2** | MXFP8 | **—** | **待测 (Test 2)** |

## GB300 参数调优核心原理

### PP 从 8 降到 4

GB200 (192 GB): 94 层需要 PP=8 (每 stage ~12 层, ~24 GB/stage)
GB300 (288 GB): PP=4 就够 (每 stage ~24 层, ~48 GB/stage, 288 GB 容得下)

PP 减半的收益:
- Pipeline bubble 配合 VP interleaving 后: PP=8 实测 ~30-50%, PP=4 降到 ~15-25%
- 更少的 PP 通信: 跨域 RDMA p2p 次数减半

### MBS 从 1 翻到 2

GB200: 激活内存紧张, MBS=1 才能放下
GB300: 多 96 GB, MBS=2 的 activation 增量约 ~20 GB, 288 GB 轻松

### VP 从 3 增到 12

PP=4 VP=12: 每 virtual stage ~2 层, pipeline bubble 极小 (~10-15%)
这是 NVIDIA GB300 官方 recipe 的配置

## GKE 部署方式

参考 GB200 235B recipe 的 LeaderWorkerSet 部署方式。GB300 差异：

| 维度 | GB200 LWS | GB300 LWS |
|------|----------|----------|
| nodeSelector | nvidia-gb200 | **nvidia-gb300** |
| RDMA count | 4 | **8** (CX-8) |
| size (Test 1) | 16 | 16 |
| size (Test 2) | — | 32 |
| NUM_OF_HYBRID_EP_RANKS_PER_DOMAIN | 8 或 32 | 8 或 32 |

## 注意事项

1. **`-g gb300` flag**: Bridge recipe 需要 GB300 GPU type。如果报错用 `-g gb200` 手动覆盖参数
2. **V2 recipe 必须**: 必须用 `-cv v2` 启用 full_iteration graph + paged stash，否则性能差 64%
3. **PP layout**: PP=4 + VP=12 的 layout 由 Bridge 自动计算，不需要手动指定（DSv3 需要，Qwen3 不需要）
4. **跨域 MNNVL**: 128 GPU 跨域时用 `NCCL_MNNVL_ENABLE=0` + `USE_MNNVL=1`（HybridEP 域内 NVLink）
5. **NVLS**: 235B 模型 HBM 占用高，`NCCL_NVLS_ENABLE=0` 避免 multicast buffer OOM
