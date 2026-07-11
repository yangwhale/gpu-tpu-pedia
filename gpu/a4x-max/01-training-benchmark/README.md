# GB300 (A4X MAX) 训练 Benchmark 对标分析

以 GB200 (A4X) 为 baseline，对标 GB300 (A4X MAX) 的训练性能优势、参数差异和测试方案。
数据来源: NeMo Megatron Bridge 26.06 官方 Performance Summary + MLPerf Training v5.1。

## 1. 硬件规格对比 (训练相关)

| 维度 | B200 (GB200 Superchip) | B300 Ultra (GB300 Superchip) | 提升 |
|------|----------------------|---------------------------|------|
| HBM 容量 | 192 GB (4x48 GB) | 288 GB (4x72 GB) HBM3e 12-layer | +50% |
| HBM 带宽 | 8 TB/s | 8 TB/s | 持平 |
| 每域 HBM | 13.4 TB (18x744 GB) | 20 TB (18x1116 GB) | +49% |
| FP8 算力 | 4,500 TFLOPS/GPU (dense) | 5,000 TFLOPS/GPU (dense) | +11% |
| FP4 算力 | 9,000 TFLOPS/GPU (dense) | 10,000 TFLOPS/GPU (dense) | +11% |
| RDMA 网卡 | CX-7 VF (挂在 CPU) | CX-8 PF (GPUDirect 直连) | 延迟更低 |
| RDMA 带宽 | 2,000 Gbps (4x400G) | 3,200 Gbps (8x400G) | +60% |
| NVLink 域 | 1x72 (18 节点) | 1x72 (18 节点) | 不变 |

> GB300 的核心优势: (1) 显存 +50% → MBS 翻倍、减少 PP/TP → 更少 bubble/通信开销; (2) RDMA GPUDirect → 跨域通信延迟更低; (3) 新增 NVFP4 高效训练模式

## 2. NeMo Bridge 官方 Benchmark (26.06 Container)

### 2.1 Pre-Training 性能对比

以下所有数据来自 [Megatron Bridge Performance Summary](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html)，26.06 NeMo Container。

#### Llama 3.1 405B (Dense, 256 GPU)

| 系统 | 精度 | TP | PP | CP | VP | MBS | Tokens/s/GPU | Model TFLOP/s/GPU | vs GB200 |
|------|------|----|----|----|----|-----|-------------|-------------------|---------|
| DGX-GB300 | FP8 | 4 | 8 | 1 | 4 | 1 | 1048 | 2646 | **+24.3%** |
| DGX-GB300 | MXFP8 | 2 | 8 | 2 | 4 | 1 | 952 | 2403 | +21.6% |
| DGX-GB300 | NVFP4 | 4 | 8 | 1 | 4 | 1 | 1413 | 3575 | +21.4% |
| DGX-GB200 | FP8 | 4 | 16 | 1 | 4 | 1 | 843 | 2129 | baseline |
| DGX-GB200 | MXFP8 | 4 | 16 | 1 | 8 | 1 | 783 | 1976 | baseline |
| DGX-GB200 | NVFP4 | 4 | 16 | 1 | 8 | 1 | 1166 | 2944 | baseline |
| DGX-H100 | FP8 | 8 | 8 | 2 | 8 | 1 | 326 | 822 | — |

**关键参数差异**:
- PP: GB200 需要 PP=16，GB300 只要 PP=8 (显存大，层数分配更灵活)
- TP: GB300 MXFP8 模式可降到 TP=2 (GB200 需 TP=4)
- CP: GB300 MXFP8 启用 CP=2 (GB200 用 CP=1)
- 性能提升 20-24%，主要来自更少 PP stage → 更小 bubble

#### DeepSeek V3 671B (MoE, 256 GPU)

| 系统 | 精度 | TP | PP | VP | EP | MBS | GBS | Tokens/s/GPU | Model TFLOP/s/GPU | vs GB200 |
|------|------|----|----|----|----|-----|-----|-------------|-------------------|---------|
| DGX-GB300 | MXFP8 | 1 | 2 | 8 | 32 | 1 | 4096 | 6338 | 1648 | **+27.6%** |
| DGX-GB300 | MXFP8 | 1 | 2 | 8 | 32 | 1 | 15360 | 6422 | 1670 | +29.3% |
| DGX-GB200 | MXFP8 | 1 | 4 | 4 | 64 | 1 | 4096 | 4969 | 1292 | baseline |
| DGX-B300 | MXFP8 | 1 | 8 | — | 8 | 2 | 4096 | 3541 | 920 | — |

**关键参数差异**:
- PP: GB300 PP=2 vs GB200 PP=4 (显存大 → 减半 PP → bubble 大幅降低)
- VP: GB300 VP=8 vs GB200 VP=4
- EP: GB300 EP=32 vs GB200 EP=64 (NVLink 域内通信更高效)
- MBS: 均为 1 (MoE 模型显存瓶颈在 expert buffer)
- GBS: GB300 支持更大 GBS=15360 (+0.5% 额外提升)
- 性能提升 27-29%

#### Qwen3 235B-A22B (MoE, 256 GPU)

| 系统 | 精度 | TP | PP | VP | EP | MBS | Tokens/s/GPU | Model TFLOP/s/GPU | vs GB200 |
|------|------|----|----|----|----|-----|-------------|-------------------|---------|
| DGX-GB300 | MXFP8 | 1 | 4 | 12 | 32 | 2 | 9015 | 1335 | **+22.3%** |
| DGX-GB200 | MXFP8 | 1 | 8 | 3 | 32 | 1 | 7376 | 1092 | baseline |

**关键参数差异**:
- PP: GB300 PP=4 vs GB200 PP=8 (减半!)
- VP: GB300 VP=12 vs GB200 VP=3 (更细粒度 pipeline interleaving)
- MBS: GB300 MBS=2 vs GB200 MBS=1 (显存大 → micro batch 翻倍)
- 性能提升 22%

#### Qwen3 30B-A3B (MoE, 8 GPU, 单节点)

| 系统 | 精度 | TP | PP | EP | MBS | Tokens/s/GPU | Model TFLOP/s/GPU | vs GB200 |
|------|------|----|----|-----|-----|-------------|-------------------|---------|
| DGX-GB300 | MXFP8 | 1 | 1 | 8 | 8 | 45275 | 1041 | **+11.2%** |
| DGX-GB200 | MXFP8 | 1 | 1 | 8 | 4 | 40706 | 936 | baseline |
| DGX-B300 | MXFP8 | 1 | 1 | 8 | 8 | 40769 | 938 | — |

**关键参数差异**:
- PP/TP 不变 (单节点小模型，不受显存限制)
- MBS: GB300 MBS=8 vs GB200 MBS=4 (翻倍!)
- 单节点场景提升 11% (没有 PP 优化空间，纯算力+显存优势)

#### GPT OSS 120B (MoE, 64 GPU)

| 系统 | 精度 | TP | PP | EP | MBS | Tokens/s/GPU | Model TFLOP/s/GPU | vs GB200 |
|------|------|----|----|-----|-----|-------------|-------------------|---------|
| DGX-GB300 | MXFP8 | 1 | 1 | 16 | 4 | 33166 | 1081 | **+14.6%** |
| DGX-GB200 | MXFP8 | 1 | 1 | 64 | 4 | 28947 | 943 | baseline |
| DGX-B300 | MXFP8 | 1 | 1 | 8 | 4 | 18534 | 604 | — |

**关键参数差异**:
- EP: GB300 EP=16 vs GB200 EP=64 (显存大 → 每 GPU 放更多 expert → EP 降低 → 通信开销减少)

#### Nemotron 3 Super (MoE, 64 GPU)

| 系统 | 精度 | TP | EP | MBS | Tokens/s/GPU | Model TFLOP/s/GPU | vs GB200 |
|------|------|----|----|-----|-------------|-------------------|---------|
| DGX-GB300 | MXFP8 | 1 | 64 | 1 | 9652 | 817 | **+43.1%** |
| DGX-GB300 | NVFP4 | 1 | 64 | 1 | 9900 | 839 | +42.9% |
| DGX-GB200 | MXFP8 | 2 | 64 | 1 | 6742 | 571 | baseline |
| DGX-GB200 | NVFP4 | 2 | 64 | 1 | 6928 | 587 | baseline |

**关键参数差异**:
- TP: GB300 TP=1 vs GB200 TP=2 (显存大 → 不需要 TP 拆分 → 省掉 all-reduce 通信)
- 性能提升高达 43%! (TP=2→1 是最大功臣)

### 2.2 SFT 性能对比 (Llama3 70B, 32 GPU)

| 系统 | 精度 | TP | PP | VP | Tokens/s/GPU | Model TFLOP/s/GPU | vs GB200 |
|------|------|----|----|------|-------------|-------------------|---------|
| DGX-GB300 | FP8 | 1 | 2 | 20 | 4819 | 2083 | **+24.7%** |
| DGX-GB300 | MXFP8 | 1 | 2 | 20 | 4312 | 1877 | +20.8% |
| DGX-GB200 | FP8 | 1 | 8 | 10 | 3864 | 1671 | baseline |
| DGX-GB200 | MXFP8 | 1 | 8 | 10 | 3593 | 1553 | baseline |
| DGX-H100 | FP8 | 4 | 4 | 5 | 1638 | 710 | — |

**关键参数差异**:
- PP: GB300 PP=2 vs GB200 PP=8 (4 倍减少! SFT 场景最受益)
- VP: GB300 VP=20 vs GB200 VP=10

### 2.3 LoRA 性能对比 (Llama3 70B, 8 GPU)

| 系统 | 精度 | TP | PP | VP | Tokens/s/GPU | Model TFLOP/s/GPU | vs GB200 |
|------|------|----|----|------|-------------|-------------------|---------|
| DGX-GB300 | FP8 | 1 | 2 | 20 | 7481 | 2086 | **+20.5%** |
| DGX-GB300 | MXFP8 | 1 | 2 | 20 | 7447 | 2072 | +24.6% |
| DGX-GB200 | FP8 | 1 | 2 | 20 | 6206 | 1731 | baseline |
| DGX-GB200 | MXFP8 | 1 | 4 | 20 | 5958 | 1663 | baseline |

## 3. MLPerf Training 第三方验证

### MLPerf Training v5.1 (2025-11)

来源: [Lambda MLPerf v5.1](https://lambda.ai/blog/lambda-mlperf-training-benchmarks-v5.1)

| Benchmark | GB300 NVL72 (72 GPU) | GB200 NVL72 (72 GPU) | 提升 |
|-----------|---------------------|---------------------|------|
| Llama 2-70B LoRA | 1.26 min | 1.598 min | **+27%** |

性能提升来源分解 (Lambda 报告):
- 硬件 (B300 Ultra GPU + 288 GB HBM3e): **1.12x**
- 软件栈 (驱动/CUDA/NCCL/cuBLAS/cuDNN): **1.13x**
- 综合: 1.12 x 1.13 = **1.27x**

### MLPerf Training v5.1 - NVIDIA 官方结果

来源: [NVIDIA Developer Blog](https://developer.nvidia.com/blog/nvidia-blackwell-enables-3x-faster-training-and-nearly-2x-training-performance-per-dollar-than-previous-gen-architecture)

| Benchmark | GB300 NVL72 (512 GPU) vs GB200 NVL72 (512 GPU) | vs H100 (512 GPU) |
|-----------|----------------------------------------------|------------------|
| Llama 3.1 405B | **1.9x** faster (NVFP4) | **4.2x** faster |

### MLPerf Training v6.0 (2026-06)

来源: [Nebius MLPerf v6.0](https://nebius.com/blog/posts/mlperf-training-v6-0-results)

| Benchmark | GB300 NVL72 (32 GPU) | HGX B300 (32 GPU) | 提升 |
|-----------|---------------------|-------------------|------|
| FLUX.1 训练 | 65.85 min | 77.48 min | **+18%** |

GB300 NVL72 比 HGX B300 快 18% (NVLink 域带来的额外优势)。

## 4. GB300 vs GB200 训练参数调优总结

GB300 的 288 GB 显存 (vs GB200 的 192 GB) 使得训练参数可以系统性优化:

| 优化维度 | GB200 典型值 | GB300 典型值 | 原理 | 性能收益 |
|---------|-----------|-----------|------|---------|
| PP (Pipeline Parallel) | PP=8~16 | PP=2~8 | 显存大 → 每 stage 放更多层 → 需要更少 stage | 减少 pipeline bubble (最大收益源) |
| VP (Virtual Pipeline) | VP=3~8 | VP=4~12 | PP 减少后可用更多 VP interleaving | 进一步减少 bubble |
| MBS (Micro Batch Size) | MBS=1 | MBS=1~2 | 显存大 → 可放更大 micro batch | 提高计算效率 |
| TP (Tensor Parallel) | TP=2~4 | TP=1~2 | 显存大 → 不需要拆分 → 省 all-reduce | 减少通信开销 |
| EP (Expert Parallel) | EP=32~64 | EP=16~32 | 每 GPU 可放更多 expert → EP 降低 | 减少 all-to-all 通信 |
| 精度 | MXFP8 | MXFP8 / NVFP4 | B300 Ultra FP4 硬件加速更强 | NVFP4 额外 +13~21% |

**核心规律**: GB300 的 50% 显存增加主要转化为 PP 减少 (pipeline bubble 是最大性能杀手)。对于大模型 (405B, 671B)，PP 从 16→8 或 8→4 带来的 bubble 减少是 20-30% 性能提升的主要来源。

## 5. 我们的测试计划

### 5.1 测试环境

- 集群: tencent-gcp-taiji-poc (us-central1-b)
- 预留: nvidia-gb300-dxkhoz4ypk4mh (214 台, 856 GPU)
- 容器: NeMo 26.06 + Megatron Bridge
- 拓扑: NVL72 (18 节点/域)

### 5.2 测试矩阵

以 GB200 实测数据为 baseline (来自 a4x/ 目录)，GB300 逐项对标:

| # | 模型 | 规模 | GB200 baseline | GB300 目标 | 预期提升 | 状态 |
|---|------|------|---------------|----------|---------|------|
| 1 | DSv3 16L | 64 GPU (单域) | 1176 TFLOPs (MNNVL=2) | > 1400 | +20-28% | 待测 |
| 2 | DSv3 16L | 64 GPU (MNNVL=0) | 1100 TFLOPs | > 1350 | +23% | 待测 |
| 3 | Qwen3 235B | 64 GPU (PP=8 EP=8) | 931 TFLOPs | > 1100 | +18% | 待测 |
| 4 | Qwen3 235B | 64 GPU (PP=4 EP=32) | — | > 1335 | 对标 NVIDIA 参考值 | 待测 |
| 5 | Qwen3 235B | 256 GPU (PP=4 EP=32) | — | 1335 (NVIDIA 参考) | — | 待测 |
| 6 | Llama 3.1 405B | 256 GPU (FP8) | — | 2646 (NVIDIA 参考) | — | 待测 |
| 7 | Llama 3.1 405B | 256 GPU (NVFP4) | — | 3575 (NVIDIA 参考) | — | 待测 |
| 8 | DSv3 Full | 256 GPU | 1292 (NVIDIA GB200 参考) | 1648 (NVIDIA GB300 参考) | +27.6% | 待测 |

### 5.3 测试重点

1. **PP 减半验证**: 对比 GB200 PP=8 vs GB300 PP=4 的实际 bubble 差异
2. **MBS 翻倍验证**: 确认 288 GB 是否支持 MBS=2 (GB200 MBS=1)
3. **NVFP4 效果**: B300 Ultra 新增 NVFP4 精度的实测训练加速比
4. **CX-8 GPUDirect 通信**: 跨域 RDMA 延迟对比 (CX-7 VF vs CX-8 PF)
5. **NCCL all-reduce**: 单域/跨域 bandwidth 对比 (3200 Gbps vs 2000 Gbps)

### 5.4 关键参数调整清单 (GB200 → GB300)

基于 NVIDIA 官方 recipe 差异，我们的测试需修改:

```bash
# DSv3 671B: GB200 → GB300
--pipeline_model_parallel_size 4 → 2     # PP 减半
--virtual_pipeline_model_parallel_size 4 → 8  # VP 翻倍
--expert_model_parallel_size 64 → 32     # EP 减半
# EP_RANKS_PER_NVLINK_DOMAIN 64 → 32

# Qwen3 235B: GB200 → GB300
--pipeline_model_parallel_size 8 → 4     # PP 减半
--virtual_pipeline_model_parallel_size 3 → 12  # VP 4x
--micro_batch_size 1 → 2                 # MBS 翻倍

# Llama 405B: GB200 → GB300
--pipeline_model_parallel_size 16 → 8    # PP 减半
# TP, CP, VP 根据精度模式调整
```

## 6. 参考链接

| 来源 | URL |
|------|-----|
| NeMo Bridge Performance Summary | https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html |
| NeMo Bridge Recipe Usage | https://docs.nvidia.com/nemo/megatron-bridge/latest/recipe-usage.html |
| Megatron Bridge GitHub | https://github.com/nvidia-nemo/megatron-Bridge |
| Lambda MLPerf v5.1 (GB300 +27%) | https://lambda.ai/blog/lambda-mlperf-training-benchmarks-v5.1 |
| NVIDIA MLPerf Blog (GB300 1.9x) | https://developer.nvidia.com/blog/nvidia-blackwell-enables-3x-faster-training-and-nearly-2x-training-performance-per-dollar-than-previous-gen-architecture |
| Nebius MLPerf v6.0 | https://nebius.com/blog/posts/mlperf-training-v6-0-results |
| NVIDIA B300 Specs | https://www.spheron.network/blog/nvidia-b300-blackwell-ultra-guide |
| GB300 vs GB200 Deep Dive | https://www.naddod.com/blog/nvidia-gb300-deep-dive-performance-breakthroughs-vs-gb200 |
