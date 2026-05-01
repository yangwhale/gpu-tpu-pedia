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
| **FP8 Native** | Qwen3.5, Qwen3-Coder | **Not required** (reads weights directly) | Recommended to pre-load (see below) |

### Startup Acceleration: Pre-loading Weights to /dev/shm

FP8 Native models (Qwen3.5, Qwen3-Coder) load 50-94 safetensors shards serially when starting from network storage (Lustre/GCS). When host RAM is insufficient to prefetch the entire checkpoint (e.g., /dev/shm is occupied by FP4 cache), **each shard takes up to 90 seconds, resulting in 15+ minute total startup time**.

**Best practice: Copy weights to /dev/shm first, then point vLLM to the local path.**

```bash
# Option 1: Copy from Lustre (recommended, fast intra-network)
cp -r /lustre/models/Qwen3.5-397B-A17B-FP8 /dev/shm/qwen35-weights/

# Option 2: Copy from GCS directly (gcloud 4.5 GB/s)
gcloud storage cp -r gs://<BUCKET>/models/qwen3.5-397b-a17b-fp8/weights/ /dev/shm/qwen35-weights/

# Launch vLLM pointing to /dev/shm
vllm serve /dev/shm/qwen35-weights/ \
  --tensor-parallel-size 8 --enable-expert-parallel ...
```

| Model | Weight Size | Recommended Strategy | Cold Start |
|-------|------------|---------------------|------------|
| Qwen3.5 | 379 GB | /dev/shm pre-load (copy ~4.5 min) | ~4 min (vs 15+ min direct read) |
| Qwen3-Coder | 450 GB | Lustre direct read (empty /dev/shm) | ~10 min |

> ⚠️ **Qwen3-Coder is too large for /dev/shm pre-loading**: 450 GB weights occupy 57% of /dev/shm, and combined with vLLM's MoE requantization temporary memory, the total exceeds the 920 GiB container memory limit, triggering OOM Killed. The correct approach is to keep /dev/shm empty (800 GB available) and read from Lustre directly — the OS then has sufficient RAM for auto-prefetch, achieving ~5-9 seconds per shard (vs 90 seconds when RAM is insufficient).
>
> ⚠️ **Test one model at a time.** FP4 model caches (R1/V3.2/GLM, ~610-735 GB) also reside in /dev/shm — they cannot coexist.

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

## Upstream Compatibility Issues Found (Verified 2026-05-01)

> The following issues were discovered during smoke testing after the upstream merge (`507cfa16..0b9f5583`, 63 commits) and have been fixed/worked around. They affect the **JAX flax_nnx inference path** (R1, V3.2, GLM-5.1, K2.6); the PyTorch+TorchAX path (Qwen3.5, Qwen3-Coder) is only affected by Issue 3.

### Issue 1: USE_MOE_EP_KERNEL Semantic Change → FUSED_MOE Error

- **Symptom**: `NotImplementedError: Unsupported moe backend: MoEBackend.FUSED_MOE`
- **Cause**: Upstream now maps `USE_MOE_EP_KERNEL=1` + `use_ep=True` to the new `FUSED_MOE` backend, which the JAX path does not support
- **Affected**: All FP4 MoE models (R1, V3.2, GLM-5.1) and Kimi K2.6
- **Fix**: **Do not set `USE_MOE_EP_KERNEL=1`**. With `use_ep=True`, the system auto-falls back to `GMM_EP` (correct path)
- **Code location**: `tpu_inference/layers/jax/moe/utils.py:select_moe_backend()`
- **Details**: See [R1 README Pitfall #23](./DeepSeek-R1-671B-FP4/)

### Issue 2: Incomplete additional-config → GMM_TP Instead of GMM_EP

- **Symptom**: Garbled inference output / precision collapse (MoE routed via Tensor Parallel instead of Expert Parallel)
- **Cause**: `--additional-config` missing `expert_parallelism:8` and `tensor_parallelism:1` in `sharding_strategy`, causing `use_ep=False`
- **Affected**: All MLA + EP models (R1, V3.2, GLM-5.1, K2.6)
- **Fix**: `additional-config` must include all three parameters:
  ```json
  {
    "enable_dp_attention": true,
    "sharding_strategy": {
      "expert_parallelism": 8,
      "tensor_parallelism": 1
    }
  }
  ```
- **Details**: See [R1 README Pitfall #24](./DeepSeek-R1-671B-FP4/)

### Issue 3: dp_scheduler.py hash_block_size Incompatibility

- **Symptom**: `TypeError: Scheduler.__init__() got unexpected keyword argument 'hash_block_size'`
- **Cause**: PR [#2412](https://github.com/vllm-project/tpu-inference/pull/2412) passes `hash_block_size` through `DPScheduler`, but the base vLLM `Scheduler` does not accept this parameter
- **Affected**: All scenarios using `DPScheduler` (**all 6 models**)
- **Fix**: Use `inspect.signature` to dynamically check if the parameter exists before passing it
- **Patch location**: `tpu_inference/core/sched/dp_scheduler.py` (committed to fork)
- **Details**: See [R1 README Pitfall #25](./DeepSeek-R1-671B-FP4/)

### Impact Matrix

| Issue | R1 | V3.2 | GLM-5.1 | K2.6 | Qwen3.5 | Qwen3-Coder |
|-------|:--:|:----:|:-------:|:----:|:-------:|:-----------:|
| USE_MOE_EP_KERNEL | ✅ | ✅ | ✅ | ✅ | — | — |
| additional-config | ✅ | ✅ | ✅ | ✅ | — | — |
| hash_block_size | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

> ✅ = Affected　— = Not affected (uses PyTorch+TorchAX path)

---

## Recent Upstream Updates & Pending Verification

> **Sync date**: 2026-05-01 | **Upstream commit range**: `507cfa16..0b9f5583` (63 commits)
>
> The following updates from [vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference) have been merged into the fork `chrisya/main` branch but have not yet been verified on each model.

### 1. KV Cache Offload to Host Memory

| PR | Description | Status |
|----|-------------|:------:|
| [#2390](https://github.com/vllm-project/tpu-inference/pull/2390) | Offload KV cache from HBM to host DRAM with async transfer + staging buffer | ⏳ |
| [#2454](https://github.com/vllm-project/tpu-inference/pull/2454) | Fix KV offloading performance test in nightly tests | ⏳ |

**Potential benefits**:
- **Kimi K2.6**: Currently OOMs on v7x-8 (weights 84.58 GB/chip + KV cache exceeds HBM). Offloading may enable single-node inference, reducing deployment requirement from v7x-16 to v7x-8
- **DeepSeek R1/V3.2**: KV cache is the bottleneck at high concurrency (c=2048); offloading enables larger batch sizes
- **GLM-5.1**: Only 36 GB/device remaining for KV cache; offloading expands capacity
- **Qwen3.5**: Benefits long-context (262K) scenarios

Environment variables: `TPU_OFFLOAD_SKIP_JAX_PRECOMPILE`, `TPU_OFFLOAD_DECODE_SAVE`, `TPU_OFFLOAD_NUM_CPU_CHUNKS`, `TPU_OFFLOAD_NUM_STAGING_BLOCKS`

### 2. Qwen3.5 GDN Fixes (4 PRs)

| PR | Description | Impact | Status |
|----|-------------|--------|:------:|
| [#2408](https://github.com/vllm-project/tpu-inference/pull/2408) | GDN l2norm + sigmoid promoted to fp32 to match GPU FLA precision | CoT accuracy fix (verified on GPQA-Diamond) | ⏳ |
| [#2431](https://github.com/vllm-project/tpu-inference/pull/2431) | Fused GDN kernel correctly handles has_initial_state | Chunked-prefill continuation correctness | ⏳ |
| [#2416](https://github.com/vllm-project/tpu-inference/pull/2416) | Compact mamba KV cache — GDN layers only allocate active request slots | Significant HBM reduction for 45 GDN layers | ⏳ |
| [#2469](https://github.com/vllm-project/tpu-inference/pull/2469) | Clear vacated slot IDs to null in InputBatch | Prevents GDN state leakage across requests | ⏳ |

**Verification plan**: Re-run GSM8K eval to compare accuracy (previous: 93.93%), and measure HBM usage at high concurrency with compact KV.

### 3. MLA / DeepSeek Fixes

| PR | Description | Affected Models | Status |
|----|-------------|----------------|:------:|
| [#2462](https://github.com/vllm-project/tpu-inference/pull/2462) | Fix upstream break for mla_attention | R1, V3.2, GLM-5.1, K2.6 | ⏳ |
| [#2407](https://github.com/vllm-project/tpu-inference/pull/2407) | Re-enable AG-FP8 (All-Gather FP8), previously disabled for ablation | R1, V3.2 | ⏳ |
| [#2343](https://github.com/vllm-project/tpu-inference/pull/2343) | Return routed expert IDs from MoE | All MoE models | ⏳ |
| [#2412](https://github.com/vllm-project/tpu-inference/pull/2412) | Add input_ids + hash_block_size for incoming DeepSeek V4 | Forward-looking | — |

**Note**: #2462 is a compatibility fix — without it, MLA models may fail on newer versions.

### 4. Multi-host / PD Disaggregation Improvements

| PR | Description | Affected Models | Status |
|----|-------------|----------------|:------:|
| [#2414](https://github.com/vllm-project/tpu-inference/pull/2414) | Fix wrong device_put usage for multi-host | K2.6 multi-host, all multi-node setups | ⏳ |
| [#2435](https://github.com/vllm-project/tpu-inference/pull/2435) | Stagger prefill/decode startup to relieve host memory pressure | GLM PD, Qwen3.5 PD, Qwen3-Coder PD | ⏳ |
| [#2392](https://github.com/vllm-project/tpu-inference/pull/2392) | Extend attn_dp_expert to emulate attn_dp | R1, V3.2 | ⏳ |

### 5. Quantization + MoE Infrastructure

| PR | Description | Affected Models | Status |
|----|-------------|----------------|:------:|
| [#2236](https://github.com/vllm-project/tpu-inference/pull/2236) | W4A8 FP8 linear layers (compressed tensors) | Potential new quantization path | ⏳ |
| [#2270](https://github.com/vllm-project/tpu-inference/pull/2270) | Jax native UnquantizedFusedMoE sharding fix | All Jax native MoE models | ⏳ |
| [#2398](https://github.com/vllm-project/tpu-inference/pull/2398) | DCP sharding axis + KV cache support | Infrastructure | ⏳ |

### 6. Stability & Maintainability

| PR | Description | Status |
|----|-------------|:------:|
| [#2399](https://github.com/vllm-project/tpu-inference/pull/2399) | deepcopy model_config to prevent config mutation | ⏳ |
| [#2441](https://github.com/vllm-project/tpu-inference/pull/2441) | Pin transformers==5.5.3 | ✅ Merged |
| [#2417](https://github.com/vllm-project/tpu-inference/pull/2417) | MLA KV cache text_config for multi-modal models | ✅ Merged (replaces our K2.6 fallback) |
| [#2418](https://github.com/vllm-project/tpu-inference/pull/2418) | Add Kimi K2.6 to nightly CI | ✅ Merged |
| [#2396](https://github.com/vllm-project/tpu-inference/pull/2396) | Qwen3.5 jittable vision tower | ⏳ |
| [#2346](https://github.com/vllm-project/tpu-inference/pull/2346) | Open-source kernel tuning infrastructure | ⏳ |

### Per-Model Impact Matrix

| Model | KV Offload | GDN Fixes | MLA Fix | AG-FP8 | Multi-host Fix | PD Opt | Compact KV |
|-------|:----------:|:---------:|:-------:|:------:|:--------------:|:------:|:----------:|
| DeepSeek R1 | Med | — | **High** | **High** | — | — | — |
| DeepSeek V3.2 | Med | — | **High** | **High** | — | — | — |
| GLM-5.1 | Med | — | **High** | — | — | Med | — |
| Kimi K2.6 | **High** | — | **High** | — | **High** | — | — |
| Qwen3.5 | Low | **High** | — | — | — | Med | **High** |
| Qwen3-Coder | Low | — | — | — | — | Med | — |

> **High** = Directly affects correctness or removes hardware limitations　**Med** = Performance/capacity improvement　**Low** = Marginal benefit　**—** = N/A

### Verification Priority

1. **P0 (Must verify)**: #2462 MLA fix — compatibility fix for all MLA models, may break without it
2. **P0 (High value)**: #2390 KV Offload + K2.6 single-node — could halve K2.6 deployment requirements
3. **P1 (Quality)**: #2408 + #2431 Qwen3.5 GDN precision — re-run GSM8K to compare
4. **P1 (Performance)**: #2407 AG-FP8 restore + #2416 Compact KV — R1/V3.2 throughput and Qwen3.5 HBM efficiency
5. **P2 (PD improvement)**: #2435 PD startup optimization + #2414 multi-host fix — GLM/Qwen PD and K2.6 multi-node
