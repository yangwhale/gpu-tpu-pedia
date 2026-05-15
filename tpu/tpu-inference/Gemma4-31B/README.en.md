# Gemma4-31B Inference on TPU v7xe

> 🌐 **Languages** | **语言**: [中文](README.md) · **English**

> End-to-end reproduction guide: Running Gemma4-31B inference with vLLM on TPU v7xe. **Full 256K context window verified.**
>
> **Architecture**: 30.7B Dense / 60 layers / hybrid sliding-window + global attention / **256K** context / 262K vocab / multimodal (text + image)
>
> **Backend**: [vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference) (JAX backend, `gemma4.py`)
>
> **Model**: [google/gemma-4-31b-it](https://huggingface.co/google/gemma-4-31b-it) (BF16, ~61 GiB)

---

## 🎯 Headline Results

| Metric | Value | Test Condition |
|--------|-------|----------------|
| **Peak Throughput** | **6,144 tok/s** | 1K/1K, P=256, TP=4 |
| **Single-User TPOT** | **35 ms** | Stable across all context lengths (1K → 256K) |
| **Full 256K Context TTFT** | **695 ms** | Single user, 256K input |
| **128K Context TTFT** | **378 ms** | Single user, 128K input |
| **Cold Start** | **~3-5 min** | TP=4, weights on Lustre |

> 💡 **Production-ready**: All 16 benchmark tests in this guide pass, covering the full 1K → 256K context range.
> Decode latency (TPOT) stays within 34-49ms across all scenarios — **independent of context length**.

---

## 📋 Prerequisites

| Item | Requirement |
|------|-------------|
| GKE cluster | TPU v7 (Ironwood) enabled |
| Shared storage | Lustre / GCS / Filestore PVC (for weights and patches) |
| HuggingFace token | License for [Gemma 4](https://huggingface.co/google/gemma-4-31b-it) accepted |
| `gcloud` / `kubectl` | GKE context configured |

---

## ⚡ Quick Start (5 Steps for Experienced Users)

Assumes the GKE cluster, v7xe Pod, model weights, and kubectl context are all in place.

```bash
CTX=<your-gke-context>; POD=gemma4-31b; MODEL=/lustre/models/gemma-4-31b-it

# 1. Verify model weights
kubectl --context=$CTX exec $POD -- bash -c "ls $MODEL/*.safetensors | wc -l"

# 2. Apply required kernel patch (prefill_batch_size 2→1, fixes long-context VMEM overflow)
kubectl --context=$CTX exec $POD -- bash -c '
WRAPPER=/workspace/tpu_inference/tpu_inference/kernels/experimental/batched_rpa/wrapper.py
sed -i "s/    prefill_batch_size = 2$/    prefill_batch_size = 1/" $WRAPPER
grep "prefill_batch_size = 1" $WRAPPER && echo "Patch applied"'

# 3. Write launcher (defaults to 256K full context, TP=4)
cat > /tmp/launch_gemma4.sh <<'L'
#!/bin/bash
pgrep -f 'EngineCore|vllm' | xargs -r kill -9 2>/dev/null; sleep 2
rm -f /tmp/libtpu_lockfile /tmp/vllm_gemma4.log; touch /tmp/vllm_gemma4.log
setsid nohup env \
  USE_BATCHED_RPA_KERNEL=1 \
  VLLM_WORKER_MULTIPROC_METHOD=fork \
  SKIP_JAX_PRECOMPILE=1 \
  VLLM_XLA_CHECK_RECOMPILATION=0 \
  vllm serve /lustre/models/gemma-4-31b-it \
    --tensor-parallel-size 4 \
    --max-model-len 262144 \
    --max-num-batched-tokens 16384 \
    --enable-chunked-prefill \
    --async-scheduling \
    --gpu-memory-utilization 0.95 \
    --kv-cache-dtype fp8 \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    >> /tmp/vllm_gemma4.log 2>&1 < /dev/null & disown
exit 0
L

# 4. Copy + start + wait until ready (~3-5 min)
kubectl --context=$CTX cp /tmp/launch_gemma4.sh $POD:/tmp/launch_gemma4.sh
kubectl --context=$CTX exec $POD -- bash /tmp/launch_gemma4.sh
for i in $(seq 1 20); do
  C=$(kubectl --context=$CTX exec $POD -- curl -sf -o /dev/null -w "%{http_code}" http://localhost:8000/health)
  echo "T+$((i*30))s HTTP $C"; [ "$C" = "200" ] && break; sleep 30
done

# 5. Smoke test
kubectl --context=$CTX exec $POD -- curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/lustre/models/gemma-4-31b-it","messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"max_tokens":20,"temperature":0}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
# Expected: Paris
```

---

# End-to-End Deployment Steps

## Step 0: Environment Variables

```bash
export PROJECT=<your-gcp-project>
export CLUSTER=<your-gke-cluster>
export REGION=<your-region>          # e.g., us-central1
export ZONE=<your-zone>              # e.g., us-central1-c
export CTX=<your-gke-context>
export POD=gemma4-31b
export MODEL=/lustre/models/gemma-4-31b-it

kubectl --context=$CTX cluster-info | head -1
```

## Step 1: Create TPU v7xe Spot Node Pool

The minimum v7xe slice is 4 chips (2x2x1 torus). Spot is ~30% the cost of on-demand, suitable for benchmarking and dev; for production use on-demand or reserved capacity.

```bash
gcloud container node-pools create np-tpu7xe-spot-gemma4 \
  --cluster=$CLUSTER --region=$REGION --project=$PROJECT \
  --node-locations=$ZONE \
  --machine-type=tpu7x-standard-4t \
  --num-nodes=1 \
  --spot \
  --disk-type=hyperdisk-balanced --disk-size=200 \
  --node-taints=google.com/tpu=present:NoSchedule \
  --workload-metadata=GKE_METADATA \
  --enable-autorepair --enable-autoupgrade \
  --async

# Wait for node ready (~2-5 min)
watch -n 10 "kubectl --context=$CTX get nodes -l cloud.google.com/gke-tpu-topology=2x2x1"
```

> 💡 **Machine type**: `tpu7x-standard-4t` = TPU v7xe, 4 chips × 192 GB HBM = 768 GB total.
> 31B Dense weights occupy 61 GB; TP=4 distributes the KV Cache across all 4 chips with comfortable headroom.

## Step 2: Deploy TPU Pod

The repo includes [`gemma4-31b-pod.yaml`](gemma4-31b-pod.yaml). Adjust the PVC name as needed, then deploy:

```bash
# Check PVC name (default assumes lustre-pvc)
kubectl --context=$CTX get pvc

# Deploy
kubectl --context=$CTX apply -f gemma4-31b-pod.yaml
kubectl --context=$CTX wait --for=condition=Ready pod/$POD --timeout=600s
```

> 💡 **Key configuration**:
> - `image: vllm/vllm-tpu:nightly` — includes the latest tpu-inference backend
> - `cloud.google.com/gke-tpu-accelerator: tpu7x` — correct v7xe accelerator label
> - `command: ["sleep", "infinity"]` — keeps Pod alive; vLLM is started via the launcher script
> - `sizeLimit: 128Gi` — Dense models have small SHM needs (compare: 671B MoE needs 300Gi+)

## Step 3: Download Model Weights

> ⚠️ **Prerequisite**: Gemma4 is a gated model — you must first [accept the license](https://huggingface.co/google/gemma-4-31b-it)
> on HuggingFace, then log in with a token. The first command below handles the login; skipping it returns 401.

```bash
export HF_TOKEN=<your-hf-token>   # create at HuggingFace settings/tokens

# 1. Log in to HF inside the Pod (one-time; token cached at ~/.cache/huggingface/token)
kubectl --context=$CTX exec $POD -- bash -c "
  pip install -U 'huggingface_hub[hf_transfer]'
  huggingface-cli login --token $HF_TOKEN
"

# 2. Download Gemma4-31B-IT (BF16, ~61 GiB) — about 5 min on Lustre
kubectl --context=$CTX exec $POD -- bash -c "
  mkdir -p $MODEL
  HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
    google/gemma-4-31b-it --local-dir $MODEL
"

# 3. Verify shard count
kubectl --context=$CTX exec $POD -- bash -c "ls $MODEL/*.safetensors | wc -l"
# Expected: 14 (or whatever shard count model.safetensors.index.json defines)

# 4. Clean SHM residuals (avoid stale worker lock files)
kubectl --context=$CTX exec $POD -- bash -c "rm -rf /dev/shm/sem.* /dev/shm/wrk_* 2>/dev/null"
```

## Step 4: Apply Required Kernel Patch

> ⚠️ **Required step**: The current nightly image's Batched RPA kernel has a VMEM accounting bug in MIXED mode — scratch arrays are not counted. Long contexts (>80K tokens) trigger non-deterministic `E0200 RuntimeUnexpectedCoreHalt`.
>
> Fix: Reduce `prefill_batch_size` from 2 to 1, halving scratch memory usage (VMEM occupancy drops from ~93% to ~75%).
>
> Full root cause analysis: [Kernel Fix Deep-Dive](#-kernel-fix-deep-dive).

```bash
kubectl --context=$CTX exec $POD -- bash -c '
WRAPPER=/workspace/tpu_inference/tpu_inference/kernels/experimental/batched_rpa/wrapper.py

# Single-line sed fix
sed -i "s/    prefill_batch_size = 2$/    prefill_batch_size = 1/" $WRAPPER

# Verify patch applied
if grep -q "prefill_batch_size = 1" $WRAPPER; then
    echo "✅ Patch applied successfully"
    grep -n "prefill_batch_size" $WRAPPER
else
    echo "❌ Patch failed - check file content"
    exit 1
fi

# Clear Python cache (important)
find /workspace/tpu_inference -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
'
```

> 💡 **When to skip this step**: Once the [upstream fix](https://github.com/vllm-project/tpu-inference) is merged and the nightly image rebuilt, `prefill_batch_size` defaults to 1 and the sed becomes a no-op (still safe).

## Step 5: Start the vLLM Inference Server

### Launch Parameter Reference

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--tensor-parallel-size` | `4` | Use all 4 v7xe chips; TP<4 wastes compute and prevents KV Cache from spanning chips |
| `--max-model-len` | `262144` | Gemma4 native 256K context window |
| `--max-num-batched-tokens` | `16384` | Chunked prefill block size — balances throughput and latency |
| `--enable-chunked-prefill` | — | Required for long context: splits 256K input into 16 × 16K chunks |
| `--async-scheduling` | — | Decouples scheduling from execution, boosts concurrency |
| `--gpu-memory-utilization` | `0.95` | Per-chip 192GB × 0.95 = ~182GB available |
| `--kv-cache-dtype` | `fp8` | Halves KV Cache, doubles max batch |
| `--limit-mm-per-prompt` | `{"image":0,"video":0}` | Disables multimodal warmup, faster startup |

### Required Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `USE_BATCHED_RPA_KERNEL` | `1` | **Required**, enables Batched RPA kernel (mandatory for Gemma4's heterogeneous head_dim 256/512) |
| `VLLM_WORKER_MULTIPROC_METHOD` | `fork` | TP=4 multiprocess transport |
| `SKIP_JAX_PRECOMPILE` | `1` | Skip JAX pre-compile, ~30s faster cold start |
| `VLLM_XLA_CHECK_RECOMPILATION` | `0` | Disable XLA recompile check |

### Start (TP=4, 256K context)

```bash
cat > /tmp/launch_gemma4.sh <<'LAUNCHER'
#!/bin/bash
cd /tmp
pgrep -f 'EngineCore|vllm' | xargs -r kill -9 2>/dev/null
sleep 2
rm -f /tmp/libtpu_lockfile /tmp/vllm_gemma4.log
touch /tmp/vllm_gemma4.log

setsid nohup env \
  USE_BATCHED_RPA_KERNEL=1 \
  VLLM_WORKER_MULTIPROC_METHOD=fork \
  SKIP_JAX_PRECOMPILE=1 \
  VLLM_XLA_CHECK_RECOMPILATION=0 \
  vllm serve /lustre/models/gemma-4-31b-it \
    --tensor-parallel-size 4 \
    --max-model-len 262144 \
    --max-num-batched-tokens 16384 \
    --enable-chunked-prefill \
    --async-scheduling \
    --gpu-memory-utilization 0.95 \
    --kv-cache-dtype fp8 \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    >> /tmp/vllm_gemma4.log 2>&1 < /dev/null &
disown
echo "launched pid=$!"
exit 0
LAUNCHER

kubectl --context=$CTX cp /tmp/launch_gemma4.sh $POD:/tmp/launch_gemma4.sh
kubectl --context=$CTX exec $POD -- bash /tmp/launch_gemma4.sh

# Monitor startup (~3-5 min)
kubectl --context=$CTX exec $POD -- tail -f /tmp/vllm_gemma4.log
```

**Startup success signal**: log ends with
```
INFO:     Application startup complete.
```

## Step 6: Verify Inference

### Health Check

```bash
kubectl --context=$CTX exec $POD -- curl -sf -o /dev/null -w "%{http_code}\n" http://localhost:8000/health
# Expected: 200
```

### Smoke Test — Basic Q&A

```bash
kubectl --context=$CTX exec $POD -- curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/lustre/models/gemma-4-31b-it",
    "messages": [
      {"role": "user", "content": "What is the capital of France? Answer in one word."}
    ],
    "max_tokens": 20,
    "temperature": 0
  }' | python3 -c 'import sys,json; r=json.load(sys.stdin); m=r["choices"][0]["message"]; print("content:", repr(m["content"])); print("finish:", r["choices"][0]["finish_reason"])'
# Expected: content: 'Paris'  finish: stop
```

### Multi-Turn Chat

```bash
kubectl --context=$CTX exec $POD -- curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/lustre/models/gemma-4-31b-it",
    "messages": [
      {"role": "user", "content": "Tell me a fun fact about Tokyo."},
      {"role": "assistant", "content": "Tokyo has over 160,000 restaurants, more than any other city in the world."},
      {"role": "user", "content": "What about Paris?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }' | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
```

## Step 7: Performance Benchmarks

Uses the built-in `vllm bench serve` tool. One warmup request per test triggers XLA compilation before measurement.

### 7.1 Short / Medium Context Benchmarks (Tests 1-6, Random Tokens)

No vLLM restart needed.

```bash
# Test 1: Single user latency (1K/1K, P=1)
kubectl --context=$CTX exec $POD -- vllm bench serve \
  --model /lustre/models/gemma-4-31b-it \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 1024 \
  --num-prompts 1 --max-concurrency 1 --num-warmups 1 --ignore-eos 2>&1 | tail -30

# Test 2: Peak throughput (1K/1K, P=256) — headline number
kubectl --context=$CTX exec $POD -- vllm bench serve \
  --model /lustre/models/gemma-4-31b-it \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 1024 \
  --num-prompts 256 --max-concurrency 256 --num-warmups 1 --ignore-eos 2>&1 | tail -30

# Test 3: Long input (16K/1K, P=16)
kubectl --context=$CTX exec $POD -- vllm bench serve \
  --model /lustre/models/gemma-4-31b-it \
  --dataset-name random \
  --random-input-len 16384 --random-output-len 1024 \
  --num-prompts 16 --max-concurrency 16 --num-warmups 1 --ignore-eos 2>&1 | tail -30

# Test 4: Long output (1K/16K, P=4)
kubectl --context=$CTX exec $POD -- vllm bench serve \
  --model /lustre/models/gemma-4-31b-it \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 16384 \
  --num-prompts 4 --max-concurrency 4 --num-warmups 1 --ignore-eos 2>&1 | tail -30

# Test 5: 64K context (64K/1K, P=1) — 63488 leaves 1K headroom, matches authoritative benchmark
kubectl --context=$CTX exec $POD -- vllm bench serve \
  --model /lustre/models/gemma-4-31b-it \
  --dataset-name random \
  --random-input-len 63488 --random-output-len 1024 \
  --num-prompts 1 --max-concurrency 1 --num-warmups 1 --ignore-eos 2>&1 | tail -30

# Test 6: 128K context (128K/1K, P=1)
kubectl --context=$CTX exec $POD -- vllm bench serve \
  --model /lustre/models/gemma-4-31b-it \
  --dataset-name random \
  --random-input-len 130048 --random-output-len 1024 \
  --num-prompts 1 --max-concurrency 1 --num-warmups 1 --ignore-eos 2>&1 | tail -30
```

### 7.2 Real-Text Long Context Benchmarks (Tests 7-12, Sonnet)

The vLLM image ships Shakespeare sonnets (`/workspace/vllm/benchmarks/sonnet.txt`) for verifying real-text vs random-token performance parity.

```bash
SONNET=/workspace/vllm/benchmarks/sonnet.txt

# Tests 7-10: Single user 64K / 96K / 120K / 128K (input + 1K output ≤ max_position_embeddings)
for LEN in 63488 98304 122880 130048; do
  echo "=== Sonnet input=$LEN P=1 ==="
  kubectl --context=$CTX exec $POD -- vllm bench serve \
    --model /lustre/models/gemma-4-31b-it \
    --dataset-name sonnet --dataset-path $SONNET \
    --sonnet-input-len $LEN --sonnet-output-len 1024 \
    --num-prompts 1 --max-concurrency 1 --num-warmups 1 --ignore-eos 2>&1 | tail -15
done

# Test 11: 128K dual concurrent
kubectl --context=$CTX exec $POD -- vllm bench serve \
  --model /lustre/models/gemma-4-31b-it \
  --dataset-name sonnet --dataset-path $SONNET \
  --sonnet-input-len 130048 --sonnet-output-len 1024 \
  --num-prompts 2 --max-concurrency 2 --num-warmups 1 --ignore-eos 2>&1 | tail -30

# Test 12: 64K quad concurrent
kubectl --context=$CTX exec $POD -- vllm bench serve \
  --model /lustre/models/gemma-4-31b-it \
  --dataset-name sonnet --dataset-path $SONNET \
  --sonnet-input-len 63488 --sonnet-output-len 1024 \
  --num-prompts 4 --max-concurrency 4 --num-warmups 1 --ignore-eos 2>&1 | tail -30
```

### 7.3 Full 256K Context Benchmarks (Tests 13-16)

The server is already configured with `--max-model-len 262144` — no restart needed.

```bash
SONNET=/workspace/vllm/benchmarks/sonnet.txt

# Tests 13-15: 192K / 224K / 256K single user
for LEN in 196608 229376 261120; do
  echo "=== Sonnet input=$LEN P=1 ==="
  kubectl --context=$CTX exec $POD -- vllm bench serve \
    --model /lustre/models/gemma-4-31b-it \
    --dataset-name sonnet --dataset-path $SONNET \
    --sonnet-input-len $LEN --sonnet-output-len 1024 \
    --num-prompts 1 --max-concurrency 1 --num-warmups 1 --ignore-eos 2>&1 | tail -15
done

# Test 16: 256K dual concurrent
kubectl --context=$CTX exec $POD -- vllm bench serve \
  --model /lustre/models/gemma-4-31b-it \
  --dataset-name sonnet --dataset-path $SONNET \
  --sonnet-input-len 261120 --sonnet-output-len 1024 \
  --num-prompts 2 --max-concurrency 2 --num-warmups 1 --ignore-eos 2>&1 | tail -30
```

## Step 8: Cleanup

```bash
# Delete Pod
kubectl --context=$CTX delete pod $POD

# Delete node pool (optional — keep it to avoid the ~5 min spin-up next time)
gcloud container node-pools delete np-tpu7xe-spot-gemma4 \
  --cluster=$CLUSTER --region=$REGION --project=$PROJECT \
  --quiet --async
```

---

## 📊 Benchmark Results

> **Environment**: vllm/vllm-tpu:nightly (vLLM 0.20.2rc1.dev223+), TP=4, FP8 KV Cache, BF16 weights, with prefill_batch_size=1 patch
>
> **Methodology**: One warmup request per test (triggers XLA compile); main measurement reports median TTFT/TPOT, peak tok/s captures the highest instantaneous output rate. `--ignore-eos` forces full output length.

### Short / Medium Context (Tests 1-6, Random Tokens)

| # | Scenario | Input | Output | Concurrency | Output tok/s | Peak | TTFT | TPOT |
|---|----------|-------|--------|-------------|-------------|------|------|------|
| 1 | Single user latency | 1K | 1K | 1 | 28 | 29 | **86 ms** | **35 ms** |
| 2 | **Peak throughput** | 1K | 1K | 256 | 4,495 | **6,144** ⭐ | 7,756 ms¹ | 49 ms |
| 3 | Long input | 16K | 1K | 16 | 317 | 432 | 7,014 ms¹ | 43 ms |
| 4 | Long output | 1K | 16K | 4 | 110 | 116 | 142 ms | 36 ms |
| 5 | 64K context | 64K | 1K | 1 | 27 | 29 | 196 ms | 37 ms |
| 6 | 128K context | 128K | 1K | 1 | 27 | 28 | 421 ms | 37 ms |

### Real-Text Long Context (Tests 7-12, Sonnet Dataset)

| # | Input | Concurrency | Output tok/s | Peak | TTFT | TPOT |
|---|-------|-------------|-------------|------|------|------|
| 7  | 64K  | 1 | 27.49 | 29  | 231 ms     | 36 ms |
| 8  | 96K  | 1 | 26.74 | 28  | 302 ms     | 37 ms |
| 9  | 120K | 1 | 27.39 | 29  | 373 ms     | 36 ms |
| 10 | 128K | 1 | 27.98 | 29  | 378 ms     | 35 ms |
| 11 | 128K | 2 | 44.98 | 58  | 4,714 ms¹  | 40 ms |
| 12 | 64K  | 4 | 82.99 | 112 | 6,052 ms¹  | 42 ms |

### Full 256K Context (Tests 13-16, Sonnet Dataset)

| # | Input | Concurrency | Output tok/s | Peak | TTFT | TPOT |
|---|-------|-------------|-------------|------|------|------|
| 13 | 192K | 1 | 28.34 | 29 | 526 ms    | 35 ms |
| 14 | 224K | 1 | 28.21 | 29 | 644 ms    | 35 ms |
| 15 | **256K** | 1 | 28.60 | 30 | **695 ms** ⭐ | **34 ms** |
| 16 | 256K | 2 | 54.12 | 58 | 980 ms¹   | 36 ms |

¹ TTFT includes **queueing time** (concurrent prefills are scheduled serially). Single-user (P=1) TTFT reflects pure prefill compute time.

### Key Observations

| Dimension | Conclusion |
|-----------|-----------|
| **Decode latency stability** | TPOT 34-49ms across all scenarios (1K → 256K, 1 → 256 concurrency), independent of context length |
| **Linear prefill scaling** | Single-user TTFT: 1K (86ms) → 64K (196ms) → 128K (378ms) → 256K (695ms) |
| **Linear throughput scaling** | 64K concurrency: 1→4 users = 27 → 83 tok/s (3.1x); 256K: 1→2 users = 29 → 54 tok/s (1.9x) |
| **Real text vs random** | Sonnet real English vs random tokens — near-identical TPOT (35-37ms) |
| **256K dual concurrent** | Two simultaneous 256K requests succeed with near-2x throughput scaling |

---

## 🛠️ Kernel Fix Deep-Dive

### Symptom

Without the patch, the nightly image randomly crashes Gemma4-31B at context > 80K tokens with:
```
E0200 RuntimeUnexpectedCoreHalt
RPAm-p256-b2-q256-k256/pallas_call
```
The crash is at the TPU driver layer and **non-deterministic** — same input may pass or crash.

### Root Cause

In `tpu_inference/kernels/experimental/batched_rpa/wrapper.py`, the `calculate_vmem_usage()` function:
- ✅ Accounts for pipeline buffers (Q/KV/O arrays)
- ❌ **Omits scratch arrays** (`m`, `l`, `acc` from `lm_scratch_shape` and `acc_scratch_shape`)

With `prefill_batch_size=2` on TPU v7x's 64MB VMEM:
- Pipeline buffers (auto-tuned to 80% cap): ~36 MB
- Scratch arrays (untracked): ~24 MB
- **Total: ~60 MB ≈ 93% VMEM occupancy**

The marginal overflow causes MIXED mode (chunked prefill + decode) to crash randomly on long contexts.

### Fix (one line)

`tpu_inference/kernels/experimental/batched_rpa/wrapper.py`:

```diff
-    prefill_batch_size = 2
+    prefill_batch_size = 1
```

Halves MIXED mode batch size, dropping scratch memory from ~24MB to ~12MB and **total VMEM from ~93% to ~75%**, eliminating the overflow.

### Validation Matrix

| Context Length | Before Fix | After Fix |
|---------------|-----------|-----------|
| ≤ 80K (≤ 40K prompt) | ✅ occasional pass | ✅ |
| 95K | ❌ E0200 crash | ✅ |
| 128K (full context) | ❌ crash | ✅ |
| 256K (extended) | ❌ crash | ✅ |

**Performance impact**: Single-user TPOT remains 35-37ms (only chunked-prefill phase affected; no regression for single-sequence inference).

---

## 📋 Troubleshooting

### Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| `RESOURCE_EXHAUSTED` creating node pool | v7xe spot capacity exhausted | Try a different zone or region |
| Weight download 403 | HF token lacks Gemma4 access | Accept the license on HuggingFace |
| Startup hangs / no logs | libtpu lockfile residual | `rm -f /tmp/libtpu_lockfile` and restart |
| Long context (>80K) E0200 crash | Missing prefill_batch_size patch | Re-run [Step 4](#step-4-apply-required-kernel-patch) |
| TP=4 XLA Reshape layout error | nightly image missing Gemma4 layout fix | Pull latest nightly image |
| `ImportError: gemma4` | nightly image too old | Pull latest nightly image |
| Pod OOM Killed | SHM too small | yaml's `sizeLimit: 128Gi` is sufficient — check for parallel workloads |

### Logs and Diagnostics

```bash
# Full startup log
kubectl --context=$CTX exec $POD -- cat /tmp/vllm_gemma4.log

# HBM usage (per chip)
kubectl --context=$CTX exec $POD -- python3 -c "
import jax
for d in jax.devices():
    s = d.memory_stats()
    print(f'{d}: {s[\"bytes_in_use\"]/1e9:.1f} GB / {s[\"bytes_limit\"]/1e9:.1f} GB')
"

# vLLM process status
kubectl --context=$CTX exec $POD -- ps aux | grep -E 'vllm|EngineCore'
```

### Verified Image Compatibility

| Verified nightly version | Notes |
|--------------------------|-------|
| `vllm/vllm-tpu:nightly` (vLLM 0.20.2rc1.dev223+) | Includes Batched RPA Gemma4 layout fix (PR #2506) and K/V_proj sharding fix (PR #2585) |

> 💡 If long contexts still crash after applying the `prefill_batch_size` patch, confirm your nightly image is ≥ dev223 — earlier images lack the foundational Gemma4 patches.

---

## 📎 References

- [Gemma 4 Model Card](https://ai.google.dev/gemma/docs/core/model_card_4) — Official model specs
- [vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference) — TPU inference backend source
- [HuggingFace: google/gemma-4-31b-it](https://huggingface.co/google/gemma-4-31b-it) — Model weights
- [PR #2506](https://github.com/vllm-project/tpu-inference/pull/2506) — Batched RPA Gemma4 layout fix
- [PR #2585](https://github.com/vllm-project/tpu-inference/pull/2585) — K/V_proj sharding fix

---

> **Document version**: v2.0
>
> **Last updated**: 2026-05-15
>
> **Reproduction status**: ✅ 16/16 benchmark tests PASS (full 256K context window verified)
