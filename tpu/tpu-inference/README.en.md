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
