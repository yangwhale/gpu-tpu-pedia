# GLM-5.1 754B FP4 Inference on TPU v7x-8

> 🌐 **Languages** | **语言**: [中文](README.md) · **English**

> End-to-end guide: Run GLM-5.1 754B (FP4 quantized) inference on a single TPU v7x-8 node.
> Beginners can follow these steps to complete the full workflow.
>
> **Code repository**: https://github.com/yangwhale/tpu-inference (branch: `feature/glm51-inference`)
>
> **Model**: [zai-org/GLM-5.1-FP8](https://huggingface.co/zai-org/GLM-5.1-FP8) (142 safetensors, 705 GB)

## 🎯 Key Performance (Measured 2026-04-24, TPU v7x-8 4 chips, FP4)

| Operating Point | Concurrency | Throughput | tok/s/chip | tok/s/user | TPOT |
|--------|-----|-----------|------------|-----------|------|
| 🚀 Max Throughput | 1,024 | **6,504 tok/s** | **1,626** | 6.4 | 125 ms |
| ⚖️ Balanced | 64 | 1,570 tok/s | 393 | 24.6 | 38 ms |
| 💨 Low Latency | 4 | 130 tok/s | 33 | 32.6 | 30 ms |

> **Throughput**: GLM-5.1 outperforms DeepSeek R1 across p16-p1024 (1.26×~3.46×), p1024 not yet saturated
> **Quality**: GSM8K 89.46% (5-shot), AIME 2026 95.3%, GPQA 86.2%, SWE-Bench Pro 58.4% (open-source SOTA)

---

## ⚠️ Three Required Environment Variables

> **Must set every time before launching vLLM, none can be omitted!**

| Environment Variable | Value | Consequence if Omitted |
|---------|-----|---------|
| `MOE_REQUANTIZE_WEIGHT_DTYPE=float4_e2m1fn` | **Required** | Defaults to FP8 cache lookup → cache miss → **HBM OOM** (651 GB vs 94.75 GB/device) |
| `NEW_MODEL_DESIGN=1` | **Required** | MLA models mandate this; without it, immediate exit on error |
| `MOE_WEIGHT_CACHE_DIR=/dev/shm` | **Required** | Cannot find FP4 cache, triggers online requantization |

**`MOE_REQUANTIZE_WEIGHT_DTYPE` is the easiest to miss and most fatal**: it controls the cache subdirectory name.
When unset, defaults to `fp8`, subdirectory becomes `ep8_tp1_gmm_ep_fp8e4m3_bsNone`,
while FP4 cache lives at `ep8_tp1_gmm_ep_fp4e2m1_bsNone`, causing all-layer cache miss → OOM.
The error message (`CompileTimeHbmOom`) **does not point to the env var as root cause**, making debugging extremely difficult.

---

## Hardware and Model Overview

| Item | Requirement |
|------|------|
| TPU | v7x-8 (2x2x1 topology, 4 chips = 8 devices) |
| HBM | 94.75 GB/device, 758 GB total |
| Host Memory | ≥940 GB (model loading + /dev/shm cache) |
| Storage | ≥2.0 TB (model ~705 GB + FP4 cache ~735 GB) |

| Model Parameter | Value |
|---------|-----|
| Architecture | MoE (256 experts, top-8) + MLA + DSA + MTP |
| Total Parameters | ~754B |
| Total Layers | 78 (layer 0-77 standard + layer 78 MTP) |
| MoE Layers | 75 (layer 3-77, MTP layer 78 skipped at inference) |
| Dense Layers (first few) | 3 (layer 0-2) |
| Quantization Scheme | FP4 MoE experts + FP8 attention + BF16 non-MoE |
| FP4 MoE HBM | ~27.5 GB/device (EP=8 sharded, 220 GB total ÷ 8) |
| Non-MoE HBM | ~21 GB/device (replicated across 8 devices) |
| Total HBM Usage | **58.43 GB/device (61.6%)**, 36 GB remaining for KV cache |

### Key Differences vs DeepSeek R1

| Parameter | DeepSeek R1 | GLM-5.1 |
|------|-------------|---------|
| hidden_size | 7168 | **6144** |
| num_hidden_layers | 61 | **78** |
| MoE Layers | 58 | **76** |
| num_attention_heads | 128 | **64** |
| q_lora_rank | 1536 | **2048** |
| qk_nope_head_dim | 128 | **192** |
| v_head_dim | 128 | **256** |
| rope_theta | 10000 (YaRN) | **1000000** (no YaRN) |
| rope_interleave | false | **true** |
| Total Parameters | ~671B | **~754B** |

> Quantization strategy, cache generation flow, and vLLM launch process are **identical** to DeepSeek R1.

---

## Step 1: Create GKE TPU Pod

In an existing GKE cluster (with TPU v7x node pool), create the Pod:

```bash
kubectl apply -f - <<'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: vllm-glm51
spec:
  containers:
  - name: main
    image: <YOUR_DOCKER_REGISTRY>/vllm-tpu:latest
    resources:
      limits:
        google.com/tpu: 8
    volumeMounts:
    - name: data
      mountPath: /data
    - name: dshm
      mountPath: /dev/shm
  volumes:
  - name: data
    persistentVolumeClaim:
      claimName: data-pvc      # Lustre PVC or Hyperdisk PVC
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: 800Gi         # /dev/shm needs ≥757 GB
  restartPolicy: Never
  nodeSelector:
    cloud.google.com/gke-tpu-topology: 2x2x1
    cloud.google.com/gke-tpu-accelerator: tpu-v7x-lite-podslice
EOF
```

> **Storage requirement**: Model ~705 GB + FP4 cache ~735 GB ≈ 1.4 TB. Recommended: Lustre PVC or Hyperdisk Extreme 4 TB.

After Pod becomes Running, enter it:

```bash
kubectl exec -it vllm-glm51 -- bash
```

---

## Step 2: Prepare Code

The Docker image's `tpu_inference` is an editable install; just switch to the GLM branch:

```bash
cd /workspace/tpu_inference
git fetch origin
git checkout feature/glm51-inference

# Verify
python3 -c "import tpu_inference; print('OK')"
```

---

## Step 3: Download Model Weights

```bash
# Install download tool (if not installed)
pip install huggingface_hub

# Download GLM-5.1 FP8 quantized model (~705 GB)
huggingface-cli download zai-org/GLM-5.1-FP8 \
  --local-dir /data/models/GLM-5.1-FP8

# Verify
ls /data/models/GLM-5.1-FP8/*.safetensors | wc -l
# Expected: 142
```

Set the model path (used in subsequent steps):

```bash
export MODEL=/data/models/GLM-5.1-FP8
```

---

## Step 4: Generate FP4 MoE Cache

> Recommended: **CPU parallel direct conversion**: pure numpy, no TPU/JAX needed, 12 workers parallel, 75 layers in just **~28 min**.

### 4a: Ensure /dev/shm is empty

```bash
# Check /dev/shm usage
df -h /dev/shm
ls /dev/shm/

# ⚠️ If old cache exists, must clean! Otherwise 12 workers will OOM kill the entire Pod
rm -rf /dev/shm/*
```

### 4b: Run FP4 conversion

```bash
# gen_fp4_cache_cpu_parallel.py is in the same directory as this README
# To download:
# curl -LO https://raw.githubusercontent.com/yangwhale/gpu-tpu-pedia/main/tpu/tpu-inference/GLM-5.1-754B-FP4/gen_fp4_cache_cpu_parallel.py

python3 -u gen_fp4_cache_cpu_parallel.py \
  --model-dir $MODEL \
  --cache-dir /data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone \
  --workers 12

# Resume after interrupt (auto-skips completed layers)
```

> **Worker count**: Adjust based on available RAM (each worker peaks at ~70 GB).
> v7x-8 machine with 944 GB RAM → max 12 workers.

### 4c: Extract Non-MoE weights

Consolidate non-MoE weights scattered across 142 safetensors into a single file, **loading from 4:26 → 21s**:

```bash
python3 extract_non_moe_weights.py \
  --model-dir $MODEL \
  --output /data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone/non_moe_weights.safetensors
```

### 4d: Verify

```bash
# Check layer count
ls /data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone/ | grep model_layers | wc -l
# Expected: 75 (layer 3-77, MTP layer 78 not needed)

# Check non-MoE file
ls -lh /data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone/non_moe_weights.safetensors
# Expected: ~21 GB

# Check FP4 shape
python3 -c "
import numpy as np
d = '/data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone/model_layers_3_mlp_experts'
for name in ['w13_weight', 'w13_weight_scale', 'w2_weight', 'w2_weight_scale']:
    a = np.load(f'{d}/{name}.npy')
    print(f'{name}: {a.shape} {a.dtype}')
"
# Expected output:
#   w13_weight:       (256, 6144, 4096) |V1    (float4_e2m1fn)
#   w13_weight_scale: (256, 1, 1, 4096) float32
#   w2_weight:        (256, 2048, 6144) |V1
#   w2_weight_scale:  (256, 1, 1, 6144) float32
```

---

## Step 5: Copy Cache to /dev/shm

Preload FP4 cache + Non-MoE weights to `/dev/shm` (tmpfs), **dramatically accelerating startup + avoiding MoE prefetch deadlock**.

```bash
SRC=/data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone
DST=/dev/shm/ep8_tp1_gmm_ep_fp4e2m1_bsNone

mkdir -p $DST

# Copy non-MoE weights
cp $SRC/non_moe_weights.safetensors $DST/

# Parallel copy of 75 layers MoE cache (8 workers, ~4 min)
ls -d $SRC/model_layers_* | xargs -P 8 -I {} cp -r {} $DST/

# Verify
ls $DST/ | grep model_layers | wc -l   # Expected: 75 (layer 3-77, MTP layer 78 not needed)
ls -lh $DST/non_moe_weights.safetensors  # Expected: ~21 GB
df -h /dev/shm                           # Expected ~757 GB used
```

> **Don't use single-thread `cp -r`**! Single-thread takes ~8 min, `xargs -P 8` parallel takes ~4 min.
>
> **Total usage**: FP4 cache ~735 GB + non-MoE 21 GB ≈ **757 GB**, /dev/shm 800 GB is sufficient.

---

## Step 6: Launch vLLM Inference Service

> ⚠️ **Reminder again: All three environment variables are required!**

```bash
# ⚠️ Three required environment variables
export MOE_REQUANTIZE_WEIGHT_DTYPE=float4_e2m1fn   # Controls FP4 cache lookup, missing = OOM
export NEW_MODEL_DESIGN=1                           # Required for MLA models
export MOE_WEIGHT_CACHE_DIR=/dev/shm                # Points to cache root directory

# ⚠️ Must cd to non-tpu-inference directory to avoid Python namespace conflict
cd /tmp

python3 -m vllm.entrypoints.openai.api_server \
  --model $MODEL \
  --tensor-parallel-size 8 \
  --quantization fp8 \
  --enforce-eager \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-model-len 4096 \
  --trust-remote-code \
  --additional-config '{
    "sharding": {
      "sharding_strategy": {
        "enable_dp_attention": true,
        "expert_parallelism": 8,
        "tensor_parallelism": 1
      }
    },
    "replicate_attn_weights": "True",
    "sparse_matmul": "True"
  }'
```

Wait for log to show `Application startup complete`.

> **Startup time**:
> - Unoptimized: **~10 min** (non-MoE loading 6m29s, `jax.clear_caches()` causes recompilation per tensor)
> - Optimized: **~3-4 min** (comment out `jax.clear_caches()` in `weight_utils.py`, see [Performance Optimization](#performance-optimization-optional) below)

### Parameter Explanation

| Parameter | Actual Meaning | Common Misconceptions |
|------|----------|-------------|
| `--tensor-parallel-size 8` | Total device count | **Not TP=8**. Actually TP=1, EP=8 (controlled by additional-config) |
| `--quantization fp8` | vLLM quantization schema name | **Not FP8 inference**. MoE FP4 controlled by env var |
| `expert_parallelism: 8` | EP=8 | 256 experts ÷ 8 = 32 experts per device |
| `tensor_parallelism: 1` | TP=1 | Attention weights replicated instead of sharded |

---

## Step 7: Verify Inference

In **another terminal** (`kubectl exec -it vllm-glm51 -- bash`), send requests:

```bash
# Test 1: Math calculation
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/data/models/GLM-5.1-FP8",
    "messages": [{"role": "user", "content": "What is 2+3? Answer with just the number."}],
    "max_tokens": 256
  }' | python3 -m json.tool

# Test 2: Chinese conversation
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/data/models/GLM-5.1-FP8",
    "messages": [{"role": "user", "content": "你是谁？用一句话介绍自己。"}],
    "max_tokens": 128
  }' | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])"

# Health check
curl -s http://localhost:8000/health
# Expected: {"status":"ok"}
```

### Verification Results (2026-04-24, GKE E2E pod, v7x-8)

| Test | Result |
|------|------|
| 2+3 math calculation | ✅ Correctly answers 5 (with chain-of-thought reasoning) |
| Chinese self-introduction | ✅ Identifies as "Z.ai's large language model", fluent output |
| English logical reasoning (60km/h × 2.5h) | ✅ Correctly reasons 150 km |
| Health check /health | ✅ Normal |
| HBM Usage | 58.43/94.75 GiB per device (61.6%) |
| KV cache | 2,309,120 tokens |
| MoE cache | 75/75 layers all hit (FP4) |

### Quality Evaluation: GSM8K Math Reasoning (2026-04-24, full 1319 questions)

| Method | Questions | strict-match | flexible-extract | Notes |
|---------|------|--------------|------------------|------|
| 0-shot CoT | 1,319 | 11.68% | **87.49% ± 0.91%** | Model freely generates "The answer is N" format |
| **5-shot ⭐** | **1,319** | **89.46% ± 0.85%** | **89.39% ± 0.85%** | **Format alignment + ICL boost** |

**Key Finding: 5-shot raises both strict and flexible scores**

The 5-shot prompt embeds 5 "#### N" formatted training examples. The model mimics this format via in-context learning → strict-match jumps from 11.68% to 89.46% (+77.78pp). This isn't the model getting smarter, it's **eval setup alignment**.

| Metric | Value |
|------|-----|
| **Test set** | GSM8K test (1,319 questions) |
| **Highest accuracy** | **89.46% (5-shot, strict-match)** |
| Eval tool | lm-evaluation-harness 0.4.9.2 |
| Inference backend | local-chat-completions (HTTP to local vLLM) |
| Generation params | greedy (temp=0), max_gen_toks=2048 |
| Quantization | FP4 (E2M1) MoE + FP8 attention |
| Eval duration | ~30 min (concurrency=64, 5-shot) |
| Reasoning parser | `--reasoning-parser glm45` correctly identifies `<think>` block |

**Output format note**: GLM-5.1 is a thinking model, output is `<think>...reasoning...</think> The answer is N`.
- 0-shot: model uses natural language closing → strict fails (11.68%) / flexible takes last number (87.49%)
- 5-shot: model mimics example's `#### N` format → strict also passes (89.46%)

**Caveat: max_gen_toks truncation**:
- 11.4% of questions had output length close to max_gen_toks limit
- max-model-len=4096 limited reasoning expansion
- Assuming truncated questions failed, theoretical ceiling ~92%; full fix requires restarting vLLM with max-model-len 16384+

**Comparison**:

| Model | GSM8K | Framework | Quantization | Notes |
|------|-------|------|------|------|
| DeepSeek R1 671B | 94.92% | vLLM + tpu-inference | FP4 | Same TPU v7x-8 hardware |
| **GLM-5.1 754B** | **89.46%** | vLLM + tpu-inference | FP4 | 5-shot ceiling, unoptimized |

The 5.46 pp gap mainly comes from (in order of contribution):
1. **No self-consistency** (multiple sampling + voting) — reasoning models with single-greedy are sensitive to single-chain errors, papers commonly use maj@8/16, can recover 5-8pp
2. **max-model-len=4096 limits thinking expansion** — some hard questions truncated
3. **vLLM HTTP backend** vs R1 testing possibly used direct vllm import, slightly different chat template handling
4. **Few-shot example selection sensitivity** — lm_eval default seed=1234 randomly picks 5 questions, 1-3pp variance across seeds

GLM-5.1 has very strong math capabilities on official benchmarks (AIME 2026: 95.3%, HMMT Nov 2025: 94.0%, GPQA-Diamond: 86.2%), positioned as agentic engineering model, achieving open-source SOTA on SWE-Bench Pro (58.4%) / CyberGym (68.7%) / BrowseComp (68.0%).

---

## Common Issue Troubleshooting

### OOM: `CompileTimeHbmOom: Used 651.83G of 94.75G hbm`

**Cause**: `MOE_REQUANTIZE_WEIGHT_DTYPE` not set, defaults to FP8 cache lookup → cache miss → OOM.

**Fix**: Confirm all three environment variables are set:
```bash
echo $MOE_REQUANTIZE_WEIGHT_DTYPE  # Should be float4_e2m1fn
echo $NEW_MODEL_DESIGN             # Should be 1
echo $MOE_WEIGHT_CACHE_DIR         # Should be /dev/shm
```

### Error: `MLA models require NEW_MODEL_DESIGN=1`

**Fix**: `export NEW_MODEL_DESIGN=1`

### vLLM hangs (0% CPU, all threads in futex_wait)

**Cause**: MoE prefetch deadlock. Semaphore deadlock when loading cache from disk.

**Fix**: Ensure cache is in `/dev/shm` (tmpfs), don't load from disk. See Step 5.

### TPU device busy

**Cause**: Previous vLLM process's EngineCore subprocess still alive.

**Fix**:
```bash
# Find and kill all related processes
ps aux | grep python | grep -v grep
kill -9 <PIDs>

# Confirm TPU device released
fuser /dev/vfio/*

# Clean up lockfile
rm -f /tmp/libtpu_lockfile /tmp/.vllm_ipc_*
```

### Multiple cache directories in `/dev/shm`

If both `ep8_tp1_gmm_ep_fp4e2m1_bsNone` and `ep8_tp1_gmm_ep_fp8e4m3_bsNone` exist:

```bash
# Remove unneeded FP8 cache leftover
rm -rf /dev/shm/ep8_tp1_gmm_ep_fp8e4m3_bsNone

# Keep only FP4 cache
ls /dev/shm/
# Should only have ep8_tp1_gmm_ep_fp4e2m1_bsNone/
```

### FP4 cache generation Pod gets OOM Killed (exit 137)

**Cause**: /dev/shm has old data, squeezing worker memory.

**Fix**: Clear /dev/shm before generation: `rm -rf /dev/shm/*`

---

## Performance Data (Measured 2026-04-24)

### Startup Time (cache + non-MoE both in /dev/shm)

| Phase | Unoptimized | Optimized (estimated) | Notes |
|------|--------|--------------|------|
| JAX init + Abstract model | ~1m44s | ~1m44s | mesh creation, 77-layer model config |
| Non-MoE weight loading | **6m29s** | **~25-108s** | When unoptimized, `jax.clear_caches()` recompiles per tensor |
| MoE Cache from /dev/shm | ~1m51s | ~1m51s | 75 layers mmap + device_put |
| Server init + warmup | ~25s | ~25s | KV cache + DPScheduler |
| **Total startup time** | **~10m44s** | **~4-6 min** | Optimization items below |

### HBM Usage (measured GKE E2E pod)

| Component | Per device | Notes |
|------|----------:|------|
| MoE Expert (FP4, EP=8 sharded) | ~27.5 GB | 220 GB total ÷ 8 devices |
| Non-MoE weights (replicated) | ~21 GB | attention + embedding + dense |
| System overhead (KV cache metadata, etc.) | ~10 GB | XLA scratch, runtime |
| **Used** | **58.43 GB (61.6%)** | Measured |
| **Available KV cache** | **~36 GB** | 94.75 − 58.43 |
| **Supported KV tokens** | **2,309,120** | Measured, max-model-len=4096 |
| HBM Total / device | 94.75 GB | 8 devices × 94.75 = 758 GB usable |

> **Note**: 4 chips × 192 GB raw HBM = 768 GB, but each device exposes 94.75 GB (runtime overhead difference).
> Non-MoE weights keep a full copy on each of 8 devices (168 GB across 8 devices), not sharded.

### FP4 Cache Generation

| Method | Duration | Notes |
|------|------|------|
| **CPU parallel direct conversion** ⭐ | **~28 min** | Pure numpy, 12 workers |
| Non-MoE extraction | ~2 min | 2292 keys → 21 GB |
| Copy to /dev/shm | ~4 min | xargs -P 8 parallel |

---

## Inference Performance Benchmark (Measured 2026-04-24, TPU v7x-8)

> **Test tool**: EvalScope perf v1.6.0 &nbsp;|&nbsp; **Method**: Each concurrency runs 1 warmup round (discarded) + 1 recording round

### 1K input / 1K output (Short Conversation Scenario)

| Concurrency | Output tok/s | tok/s/chip | TTFT (s) | TPOT (ms) | tok/s/user | Latency (s) |
|------------:|-------------:|-----------:|---------:|----------:|-----------:|------------:|
|           1 |         28.4 |        7.1 |    0.534 |        35 |       28.4 |       36.07 |
|           2 |         56.6 |       14.1 |    0.531 |        35 |       28.3 |       36.19 |
|           4 |        130.5 |       32.6 |    0.510 |        30 |       32.6 |       31.39 |
|           8 |        254.4 |       63.6 |    0.528 |        31 |       31.8 |       32.20 |
|          16 |        444.8 |      111.2 |    1.016 |        35 |       27.8 |       36.83 |
|          32 |        788.2 |      197.1 |    1.974 |        39 |       24.7 |       41.53 |
|          64 |        1,570 |      392.5 |    3.174 |        38 |       24.6 |       41.67 |
|         128 |      2,569.1 |      642.3 |    4.870 |        45 |       20.1 |       50.89 |
|         256 |      3,873.2 |      968.3 |    8.869 |        57 |       15.2 |       67.38 |
|         512 |      5,201.0 |    1,300.3 |   16.74  |        83 |       10.2 |      100.19 |
|   **1,024** | **6,504.5**  |  **1,626** |   31.38  |       125 |        6.4 |      159.27 |

**Key Pareto Operating Points:**

| Operating Point | Concurrency | Throughput | tok/s/user | TPOT | Use Case |
|--------|-----|-----------|-----------|------|---------|
| 🚀 **Max Throughput** | 1,024 | 6,504 tok/s (1,626/chip) | 6.4 | 125 ms | Offline batch processing |
| ⚖️ **Balanced** | 64 | 1,570 tok/s (393/chip) | 24.6 | 38 ms | Medium-load online service |
| 💨 **Low Latency** | 4 | 130 tok/s (33/chip) | 32.6 | 30 ms | Interactive conversation |

**Key Findings:**

- ✅ **Throughput scales continuously** — p1 → p1024 throughput grows 229×, concurrency grows 1024×, scaling efficiency 22%
- ⚠️ **p1024 not yet saturated** — vs p512 still has 25% improvement (1,300 → 1,626 tok/s/chip), need to test p2048+ to find saturation point
- ✅ **TTFT predictable** — even p1024 only 31s, no R1-style abnormal TTFT degradation at medium concurrency
- ✅ **100% success rate** — all 11 concurrency levels (p1-p1024) achieve 100% success rate
- 📈 **scaling decay pattern** — p64→128 (+1.64×), p128→256 (+1.51×), p256→512 (+1.34×), p512→1024 (+1.25×) — classic Amdahl curve

### Comparison vs DeepSeek R1 671B FP4 (same TPU v7x-8 hardware)

| Concurrency | GLM-5.1 (tok/s/chip) | DeepSeek R1 (tok/s/chip) | GLM/R1 |
|---------:|--------------------:|-----------------------:|:------:|
|        1 |                 7.1 |                    9.4 |  0.76× |
|        4 |                32.6 |                   36.7 |  0.89× |
|       16 |               111.2 |                   41.2 | **2.70×** |
|       64 |               392.5 |                  113.3 | **3.46×** |
|      128 |               642.3 |                  230.7 | **2.78×** |
|      256 |               968.3 |                  547.3 | **1.77×** |
|      512 |             1,300.3 |                  917.9 | **1.42×** |
|    1,024 |           **1,626** |              1,289.8   | **1.26×** |
|    2,048 |        ⏳ Not tested |              1,827.3   |  —    |

> **Observation**: GLM-5.1 outperforms R1 across p16-p1024 (1.26×~3.46×), consistent with GLM architecture being "thinner and longer" (hidden 6144 vs 7168, 78 vs 61 layers) — smaller per-token compute, higher batch scheduling efficiency. R1 reaches max throughput of 1,827 tok/s/chip at p2048; GLM-5.1 p2048 not tested.

### Throughput-Quality Comprehensive (GLM-5.1 vs DeepSeek R1)

| Dimension | GLM-5.1 754B | DeepSeek R1 671B |
|------|-------------|------------------|
| 1K/1K max throughput | 1,626 tok/s/chip @ p1024 | 1,827 tok/s/chip @ p2048 |
| 1K/1K balanced | **3.46× R1 @ p64** | Degrades at p16-p64 range |
| GSM8K math accuracy | 89.46% (5-shot, unoptimized) | 94.92% (official testing) |
| Model positioning | Agentic engineering | Reasoning-first |
| Cold start | ~14 min (incl. FP4 cache loading) | ~3.5 min |

### Long Context Scenarios (To Be Tested)

| Scenario | Status |
|------|------|
| 8K input / 1K output | ⏳ Pending (requires restarting vLLM with max-model-len 16384+) |
| 1K input / 8K output | ⏳ Pending |

---

## Performance Optimization (Optional)

### Comment out `jax.clear_caches()` — 10x faster Non-MoE loading

The `jax.clear_caches()` in `weight_utils.py` causes each tensor's `jax.device_put()` to recompile the transfer program.
2292 non-MoE tensors only have ~25 unique shapes, but they recompile every time, wasting 6 minutes.

```bash
# Execute inside Pod
cd /workspace/tpu_inference
grep -n 'jax.clear_caches()' tpu_inference/models/jax/utils/weight_utils.py
# Comment out all occurrences of jax.clear_caches()
sed -i 's/^        jax.clear_caches()/#        jax.clear_caches()/' \
  tpu_inference/models/jax/utils/weight_utils.py
```

> **Effect**: Non-MoE loading from 6m29s → ~25-108s (DeepSeek R1 measured 10x speedup).
> **Risk**: None. Cache stores compiled transfer programs, ~25 shapes × few KB ≈ less than 1 MB.

See [DeepSeek R1 Pitfall #18](../DeepSeek-R1-671B-FP4/README.en.md#18-jaxclear_caches-performance-bug--commenting-one-line-10x-faster) for details.

---

## End-to-End Workflow Summary

```
Step 1: Create Pod (kubectl apply)
    ↓
Step 2: Switch code branch (git checkout feature/glm51-inference)
    ↓
Step 3: Download model (huggingface-cli download, ~705 GB)
    ↓
Step 4: Generate FP4 cache (gen_fp4_cache_cpu_parallel.py, ~28 min)
       + Extract non-MoE weights (extract_non_moe_weights.py, ~2 min)
    ↓
Step 5: Copy cache to /dev/shm (xargs -P 8, ~4 min)
    ↓
Step 6: Launch vLLM (⚠️ Set three env vars!)
    ↓
Step 7: curl to verify inference
```

> **First deployment total time** (excluding model download): FP4 generation 28 min + extraction 2 min + copy 4 min + startup ~11 min ≈ **~45 min**
>
> **Subsequent restarts**: Only Step 6-7 (if /dev/shm cache still exists), **~11 min** (optimized ~4-6 min).

---

## Environment Variable Reference

| Variable | Description | Value | Required |
|------|------|-----|------|
| `MOE_REQUANTIZE_WEIGHT_DTYPE` | MoE target quantization type, controls cache subdir name | `float4_e2m1fn` | ⚠️ **Required** |
| `NEW_MODEL_DESIGN` | Enables MLA model design | `1` | ⚠️ **Required** |
| `MOE_WEIGHT_CACHE_DIR` | MoE weight cache root directory | `/dev/shm` | ⚠️ **Required** |
| `MOE_REQUANTIZE_BLOCK_SIZE` | Quantization block size | (optional) | Optional |

---

## References

| Resource | Link |
|------|------|
| DeepSeek R1 FP4 inference guide (full version) | [../DeepSeek-R1-671B-FP4/README.en.md](../DeepSeek-R1-671B-FP4/README.en.md) |
| GLM-5.1 HuggingFace model | [zai-org/GLM-5.1-FP8](https://huggingface.co/zai-org/GLM-5.1-FP8) |
| tpu-inference code | [github.com/yangwhale/tpu-inference](https://github.com/yangwhale/tpu-inference) branch: `feature/glm51-inference` |
