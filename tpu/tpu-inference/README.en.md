# Large Model Inference on TPU v7x — Quick Start Guide

**English** | [中文](./README.md)

> **Positioning Statement**
>
> This repository is a **POC (Proof of Concept) quick start guide**, designed to help users run large model inference on TPU v7x with minimum friction.
>
> - **Executable first**: Each model provides end-to-end steps from scratch to successful inference
> - **Not a benchmark**: Performance numbers appearing in the docs are byproducts of functional verification — they **do not represent optimized production performance** and should not be used as evaluation metrics
> - **Not production-ready**: No production-level tuning (KV cache strategies, batch scheduling, PD disaggregation, etc.) — only functional correctness is guaranteed
>
> For performance comparisons or production deployment plans, please contact the TPU Inference team.

## Model Overview

| Model | Params | Architecture | Quantization | TPU Topology | Cold Start | Docs |
|-------|--------|--------------|-------------|--------------|------------|------|
| DeepSeek R1 | 671B | MoE 256E top-8, MLA | FP4 MoE + FP8 Attn | v7x-8 | ~4-6 min | [Details](./DeepSeek-R1-671B-FP4/) |
| DeepSeek V3.2 | 671B | MoE 256E top-8, MLA | FP4 MoE + FP8 Attn | v7x-8 | ~4-6 min | [Details](./DeepSeek-V3.2-671B-FP4/) |
| GLM-5.1 | 754B | MoE 180E top-8 | FP4 MoE + FP8 Attn | v7x-8 | ~3-4 min | [Details](./GLM-5.1-754B-FP4/) |
| Kimi K2.6 | 1T / 32B active | MoE, native INT4 | INT4 | v7x-16 | ~6 min | [Details](./Kimi-K2.6-1T-A32B-INT4/) |
| Qwen3.5 | 397B / 17B active | Hybrid GDN+Attn, 512E | FP8 | v7x-8 | ~7 min | [Details](./Qwen3.5-397B-A17B-FP8/) |
| Qwen3-Coder | 480B / 35B active | MoE, FP8 native | FP8 | v7x-8 | ~7 min | [Details](./Qwen3-Coder-480B/) |

**Hardware Baseline**: TPU v7x-8 = 4 chips / 8 devices / 768 GB HBM / ~944 GB host memory.

## Verification Status

| Model | Inference | Quality Eval | Known Limitations |
|-------|-----------|-------------|-------------------|
| DeepSeek R1 | ✅ Passed | GSM8K 94.92% | — |
| DeepSeek V3.2 | ✅ Passed | Smoke test | Requires hot-patch to register V32 architecture |
| GLM-5.1 | ✅ Passed | GSM8K 89.46% | First JIT compilation ~13 min |
| Kimi K2.6 | ✅ Passed | Smoke test | Full 61 layers require v7x-16; v7x-8 only runs 40 layers |
| Qwen3.5 | ✅ Passed | GSM8K 93.93% | Chat path unstable, only completion mode reliable |
| Qwen3-Coder | ✅ Passed | Smoke test | — |

## Model Architecture & Feature Matrix

> ✅ Supported and implemented　🔇 Supported but bypassed　— Not in model　⏳ Pending verification

| Model | Attention | MoE | Layers | Hidden | Pos Enc | MTP | DSA | Vision | Hybrid KV |
|-------|-----------|-----|--------|--------|---------|:---:|:---:|:------:|:---------:|
| DeepSeek R1 | MLA | 256E top-8 | 61 (3D+58M) | 7168 | RoPE+YaRN | 🔇 | — | — | — |
| DeepSeek V3.2 | MLA | 256E top-8 | 61 (3D+58M) | 7168 | RoPE+YaRN | 🔇 | 🔇 | — | — |
| GLM-5.1 | MLA | 256E top-8 | 78+1 (3D+75M+MTP) | 6144 | RoPE (θ=1M) | 🔇 | 🔇 | — | — |
| Kimi K2.6 | MLA | 384E+1S top-8 | 61 (1D+60M) | 7168 | RoPE+YaRN | — | — | 🔇 | — |
| Qwen3.5 | GQA (32Q/2KV) | 512E+1S top-10 | 60 (45 GDN+15 GQA) | 4096 | YaRN+mrope | — | — | 🔇 | ✅ |
| Qwen3-Coder | GQA (40Q/8KV) | 128E top-8 | 94 | 5120 | RoPE | — | — | — | — |
| MiMo-V2-Flash | MHA | Dense | ⏳ | ⏳ | RoPE | ⏳ | — | — | — |

> **Layer abbreviations**: D = Dense, M = MoE, MTP = Multi-Token Prediction, GDN = Gated Delta Network (linear attention), S = Shared Expert

### Bypassed Feature Details

| Feature | Affected Models | Bypass Method | Potential Impact |
|---------|----------------|---------------|-----------------|
| **MTP** | R1, V3.2, GLM-5.1 | MTP heads not enabled in TPU inference; GLM layer 78 explicitly skipped | 30-50% decode throughput improvement not utilized |
| **DSA** | V3.2, GLM-5.1 | V3.2: indexer weights skipped (`skip_substrs=['indexer']`); GLM: SparseAttnIndexer is GPU-only path, inactive on TPU | Long-context sparse attention optimization not enabled |
| **Vision** | K2.6, Qwen3.5 | Disabled via `--limit-mm-per-prompt='{"image":0,"video":0}'` | K2.6's MoonViT 400M and Qwen3.5's multimodal capabilities unused |

## Deployment Capabilities

| Model | Single-Node | PD Disagg | Multi-Node | Notes |
|-------|:-----------:|:---------:|:----------:|-------|
| DeepSeek R1 | ✅ | ⏳ | ⏳ | v7x-8, EP=8 |
| DeepSeek V3.2 | ✅ | ⏳ | ⏳ | v7x-8, EP=8 |
| GLM-5.1 | ✅ | ✅ | ⏳ | v7x-8, EP=8; PD disagg requires vLLM v1 scheduler (V0 DPScheduler removed) |
| Kimi K2.6 | ❌ | ⏳ | ✅ | v7x-8 full 61 layers OOM (weights + KV cache exceed HBM); inference only on v7x-16 |
| Qwen3.5 | ✅ | ✅ | ✅ | All three modes verified |
| Qwen3-Coder | ✅ | ✅ | ✅ | Multi-node TP=16 throughput 15-63% worse, not recommended |

> ✅ Verified working　⚠️ Unstable / known issues　⏳ Pending verification　❌ Not feasible

## Performance Overview (POC Reference Data)

> ⚠️ **These numbers are byproducts of functional verification, not optimized production performance.** All tests ran with `--enforce-eager` (no XLA compilation optimization) and no batch scheduling tuning.

### 1K Input / 1K Output

| Model | TTFT (c=1) | TPOT (c=1) | Per-User tok/s | Peak Throughput | @ Concurrency |
|-------|-----------|-----------|---------------|----------------|---------------|
| DeepSeek R1 | 480 ms | 26 ms | 37.6 | 7,309 tok/s | c=2048 |
| DeepSeek V3.2 | ~480 ms ¹ | ~26 ms ¹ | ~37.6 ¹ | ~7,309 ¹ | c=2048 |
| GLM-5.1 | 534 ms | 35 ms | 28.4 | 6,504 tok/s | c=1024 |
| Kimi K2.6 ² | 1,142 ms | 49 ms | 20.0 | 592 tok/s | c=32 |
| Qwen3.5 | — | ~20 ms | 49.6 | 2,097 tok/s | c=128 |
| Qwen3-Coder | 95 ms | 20.6 ms | 48.0 | 1,478 tok/s | c=64 |

¹ V3.2 shares identical architecture with R1; numbers reference R1 benchmarks. Independent V3.2 benchmarks pending.
² Kimi K2.6 data measured on v7x-16 (full 61 layers); v7x-8 can only run 40 layers.

### 8K Scenarios

| Model | 8K In / 1K Out (tok/s) | 1K In / 8K Out (tok/s) | Concurrency |
|-------|:----------------------:|:----------------------:|:-----------:|
| DeepSeek R1 | — | — | Not tested |
| DeepSeek V3.2 | — | — | Not tested |
| GLM-5.1 | — | — | Not tested |
| Kimi K2.6 | — | 581 ³ | c=32 |
| Qwen3.5 | 850 | 1,702 | c=64 |
| Qwen3-Coder | 943 | 1,623 | c=64 |

³ Kimi K2.6 measured at 1K In / 7K Out

## Current Status Summary

All 6 models have completed **inference functional verification** on TPU v7x, with quality evaluations meeting expectations (GSM8K 89-95% for tested models). Current performance is at the **"functionally usable but not optimized"** stage:

- **Per-user latency** (TPOT 20-50 ms) is adequate for interactive chat scenarios
- **System throughput** has preliminary data, but all tests ran in `enforce_eager` mode without XLA graph compilation optimization
- **Long context** (8K+) only tested for Qwen series; other models pending
- **PD disaggregation / multi-node** fully verified only for Qwen3.5 and Qwen3-Coder; Kimi K2.6 multi-node works; others not yet implemented

For production-grade performance data or optimization plans, please contact the TPU Inference team.

## Quick Start

```
Download Weights     Prepare Cache (FP4 models)       Launch vLLM
      │                      │                            │
      ▼                      ▼                            ▼
HuggingFace or       gen_fp4_cache_cpu_parallel.py     vllm serve \
 GCS copy              + extract_non_moe_weights.py      --tensor-parallel-size 8 \
                     Copy to /dev/shm                     --quantization fp8 ...
```

**Three Quantization Paths**:

| Path | Models | Cache Generation | /dev/shm Required |
|------|--------|-----------------|-------------------|
| **FP4 MoE** | R1, V3.2, GLM-5.1 | Required (CPU parallel script, ~28 min) | ~610-735 GB |
| **INT4 MoE** | Kimi K2.6 | Required (model-native conversion) | ~532 GB |
| **FP8 Native** | Qwen3.5, Qwen3-Coder | **Not required** (reads weights directly) | Not required |

## Environment Variables Reference

### FP4 MoE Models (DeepSeek R1 / V3.2 / GLM-5.1)

```bash
export MOE_REQUANTIZE_WEIGHT_DTYPE=float4_e2m1fn
export NEW_MODEL_DESIGN=1
export MOE_WEIGHT_CACHE_DIR=/dev/shm
```

### INT4 MoE Model (Kimi K2.6)

```bash
export K26_USE_V16=1
export MOE_WEIGHT_CACHE_DIR=/dev/shm/k26_cache_v2
```

### FP8 Native Models (Qwen3.5 / Qwen3-Coder)

```bash
export MODEL_IMPL_TYPE=vllm
export SKIP_JAX_PRECOMPILE=1
export VLLM_XLA_CHECK_RECOMPILATION=0
```

## Model Storage (GCS)

Model weights and precomputed caches are stored in a GCS bucket, organized as follows:

```
gs://<YOUR_BUCKET>/models/
├── deepseek-r1-671b/
│   ├── weights/                          # 163 safetensors shards (~642 GB)
│   └── cache/fp4/ep8_tp1_.../            # 58-layer FP4 npy_v1 + non_moe_weights.safetensors
├── deepseek-v3.2-671b/
│   ├── weights/                          # 163 safetensors shards (~643 GB)
│   └── cache/fp4/ep8_tp1_.../            # 58-layer FP4 npy_v1 + non_moe_weights.safetensors
├── glm-5.1-754b/
│   ├── weights/                          # 143 safetensors shards (~705 GB)
│   └── cache/fp4/ep8_tp1_.../            # 76-layer FP4 npy_v1 + non_moe_weights.safetensors
├── kimi-k2.6/
│   ├── weights/                          # 64 safetensors shards (~555 GB)
│   └── cache/fp4/                        # 60-layer INT4 cache (~532 GB)
├── qwen3-coder-480b-fp8/
│   └── weights/                          # 49 safetensors shards (~449 GB)
├── qwen3.5-397b-a17b-fp8/
│   └── weights/                          # 94 safetensors shards (~378 GB)
└── MiMo-V2-Flash/
    └── weights/                          # 145 safetensors shards (~292 GB)
```

> FP8 native models (Qwen series) do not need a cache directory — inference reads weights directly.

## Tool Scripts

| Script | Purpose | Dependencies | Applicable Models |
|--------|---------|-------------|-------------------|
| `gen_fp4_cache_cpu_parallel.py` | Generate FP4 MoE cache from safetensors | CPU only (numpy) | R1, V3.2, GLM-5.1 |
| `extract_non_moe_weights.py` | Extract non-MoE weights into a single file | CPU only (torch) | R1, V3.2, GLM-5.1 |
| `validate_weights.py` | Validate weight integrity | torch | GLM-5.1 |

These scripts are located in each model's subdirectory and run on CPU only — no TPU/GPU required.

## Infrastructure

For TPU VM creation and storage configuration, see the [TPU-VM Guide](./TPU-VM/), which covers:

- Hyperdisk ML data disk creation and mounting
- TPU v7x-8 VM instance creation
- fio disk performance benchmarking
- GCS bucket configuration

## Directory Structure

```
tpu-inference/
├── README.md                          # Overview (Chinese)
├── README.en.md                       # Overview (English, this file)
├── DeepSeek-R1-671B-FP4/              # DeepSeek R1 inference guide
├── DeepSeek-V3.2-671B-FP4/            # DeepSeek V3.2 inference guide
├── GLM-5.1-754B-FP4/                  # GLM-5.1 inference guide
├── Kimi-K2.6-1T-A32B-INT4/            # Kimi K2.6 inference guide
├── Qwen3.5-397B-A17B-FP8/             # Qwen3.5 inference guide
├── Qwen3-Coder-480B/                  # Qwen3-Coder inference guide
└── TPU-VM/                            # TPU VM infrastructure guide
```
