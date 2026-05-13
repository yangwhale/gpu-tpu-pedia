# Gemma4-31B Inference on TPU v7xe

> 🌐 **Languages** | **语言**: [中文](README.md) · **English**

> End-to-end guide: Running Gemma4-31B BF16 inference on TPU v7xe (single chip, TP=1).
>
> **Architecture**: 30.7B Dense / 60 layers / hybrid sliding-window + global attention / 256K context / 262K vocab / multimodal (text + image)
>
> **Code Repository**: [vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference) (main branch, JAX backend `gemma4.py` / `gemma4_mm.py`)
>
> **Model**: [google/gemma-4-31b-it](https://huggingface.co/google/gemma-4-31b-it) (BF16, ~61 GiB)

---

## 🔍 Maturity Assessment

> ⚠️ **Alpha stage** — tpu-inference has Gemma4 JAX implementations (`gemma4.py` + `gemma4_mm.py`), but there are active bugs:
>
> | Issue | Problem | Impact |
> |-------|---------|--------|
> | [#2453](https://github.com/vllm-project/tpu-inference/issues/2453) | MoE variant weight loading OOM | Only affects 26B MoE, not 31B Dense |
> | [#2126](https://github.com/vllm-project/tpu-inference/issues/2126) | torchax backend fails on TPU | May affect PyTorch path, JAX path TBD |
> | [vllm#39827](https://github.com/vllm-project/vllm/issues/39827) | Output repeated tokens | Quality issue, pending investigation |
>
> **Test objective**: Verify 31B Dense end-to-end inference feasibility on TPU v7.

---

## 🧮 HBM Estimation

| Item | BF16 | FP8 (if supported) |
|------|------|---------------------|
| Model weights | 30.7B × 2B = **~61.4 GB** | 30.7B × 1B = **~30.7 GB** |
| KV Cache (4K ctx) | ~1-2 GB | ~0.5-1 GB |
| KV Cache (256K ctx) | ~30-60 GB (estimate) | ~15-30 GB |
| **Total (4K ctx)** | **~63 GB** | **~32 GB** |
| TPU v7xe single chip HBM | **192 GB** | **192 GB** |
| **Utilization** | **33%** | **17%** |

**Conclusion**: **Single chip is sufficient**. BF16 full precision uses only 33% HBM. v7xe minimum config is 4 chips (2x2x1), using TP=1 with only 1 chip active.

### KV Cache Detailed Estimation

Gemma4 31B uses **hybrid attention** (50 sliding + 10 full), with a unique KV Cache structure:

| Layer Type | Count | KV Heads | head_dim | Window | Per-token KV (BF16) | Per-token KV (FP8) |
|-----------|-------|----------|----------|--------|--------------------|--------------------|
| sliding_attention | 50 | 16 | 256 | 1024 tokens | 16,384 B (fixed cap) | 8,192 B |
| full_attention | 10 | 4 | 512 | unlimited | 8,192 B | 4,096 B |

**KV Cache per batch slot (max_model_len=4096)**:

| Component | BF16 | FP8 |
|-----------|------|-----|
| Sliding layers (50 × 1024 token window, fixed) | 838 MB | 419 MB |
| Full layers (10 × 4096 tokens) | 336 MB | 168 MB |
| **Total per slot** | **1.15 GB** | **0.57 GB** |

**Max batch size recommendation**:

```
Available HBM = 192 GB × 0.9 - 61.4 GB (weights) - 2 GB (buffer) ≈ 109 GB
```

| KV dtype | Per slot | **Max batch** | **Recommended max-num-seqs** |
|----------|---------|--------------|------------------------------|
| BF16 | 1.15 GB | ~94 | **80** |
| **FP8** ⭐ | **0.57 GB** | **~190** | **160** |

> 💡 **Recommended: FP8 KV Cache** (`--kv-cache-dtype fp8`): doubles batch capacity, significant throughput boost, minimal accuracy loss.

---

## 🧭 Deployment Options

| Mode | TPU Config | Notes |
|------|-----------|-------|
| **Single chip TP=1** ⭐ | 1 × v7xe (4 chips, only 1 used) | **Recommended**, 31B Dense fits easily on 1 chip |
| 4 chips TP=4 | 1 × v7xe (4 chips) | Optional, distributes KV Cache for longer context |

---

## ⚡ Quick Start (5 Commands for Experienced Users)

```bash
CTX=<your-gke-context>; POD=<your-tpu-pod>; MODEL=/lustre/models/gemma-4-31b-it

# 1. Verify model weights
kubectl --context=$CTX exec $POD -- bash -c "ls $MODEL/*.safetensors | wc -l"

# 2. Write launcher
cat > /tmp/launch_gemma4.sh <<'L'
#!/bin/bash
pgrep -f 'EngineCore|vllm' | xargs -r kill -9; sleep 2
rm -f /tmp/libtpu_lockfile /tmp/vllm_gemma4.log; touch /tmp/vllm_gemma4.log
setsid nohup env SKIP_JAX_PRECOMPILE=1 VLLM_XLA_CHECK_RECOMPILATION=0 \
  vllm serve /lustre/models/gemma-4-31b-it \
    --tensor-parallel-size 1 --max-model-len 4096 \
    --max-num-batched-tokens 4096 --max-num-seqs 160 \
    --no-enable-prefix-caching --gpu-memory-utilization 0.9 \
    --kv-cache-dtype fp8 --block-size 256 --trust-remote-code \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --async-scheduling \
    >> /tmp/vllm_gemma4.log 2>&1 < /dev/null & disown
exit 0
L

# 3. cp + run
kubectl --context=$CTX cp /tmp/launch_gemma4.sh $POD:/tmp/launch_gemma4.sh
kubectl --context=$CTX exec $POD -- bash /tmp/launch_gemma4.sh

# 4. Wait cold start (~3-5 min, much faster than 397B+ models)
for i in $(seq 1 20); do C=$(kubectl --context=$CTX exec $POD -- curl -sf -o /dev/null -w "%{http_code}" http://localhost:8000/health); echo "T+$((i*30))s HTTP $C"; [ "$C" = "200" ] && break; sleep 30; done

# 5. Smoke test
kubectl --context=$CTX exec $POD -- curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/lustre/models/gemma-4-31b-it","messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"max_tokens":20,"temperature":0}' \
  | python3 -c 'import sys,json;r=json.load(sys.stdin);print(r["choices"][0]["message"]["content"])'
# Expected: Paris
```

---

# End-to-End Deployment Steps

## Step 0: Prerequisites

### GKE Cluster Requirements

- GKE cluster with TPU v7 (Ironwood) support
- Lustre or GCS shared storage configured
- kubectl context configured

### Verify Cluster and Context

```bash
# Set variables
export PROJECT=<your-gcp-project>
export CLUSTER=<your-gke-cluster>
export REGION=<your-region>          # e.g., us-central1
export ZONE=<your-zone>              # e.g., us-central1-c
export CTX=<your-gke-context>

# Verify cluster reachable
kubectl --context=$CTX get nodes | grep tpu
```

## Step 1: Create TPU v7xe Spot Node Pool

```bash
# Create v7xe spot node pool (4 chips, 2x2x1 torus)
gcloud container node-pools create np-tpu7xe-spot-gemma4 \
  --cluster=$CLUSTER --region=$REGION --project=$PROJECT \
  --node-locations=$ZONE --machine-type=ct7xe-standard-4t --num-nodes=1 --spot \
  --disk-type=hyperdisk-balanced --disk-size=200 \
  --node-taints=google.com/tpu=present:NoSchedule \
  --workload-metadata=GKE_METADATA --enable-autorepair --enable-autoupgrade --async

# Wait for node ready (~2-5 min)
watch "kubectl --context=$CTX get nodes -l cloud.google.com/gke-tpu-topology=2x2x1"
```

> 💡 **Machine type**: `ct7xe-standard-4t` = TPU v7xe, 4 chips, 192 GB HBM per chip, 768 GB total.
> 31B Dense needs only ~63 GB on a single chip, TP=1. GKE minimum TPU pod slice is 4 chips.

## Step 2: Deploy TPU Pod

### Pod YAML

```yaml
# gemma4-31b-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: gemma4-31b
  labels:
    app: gemma4-31b
spec:
  nodeSelector:
    cloud.google.com/gke-tpu-topology: "2x2x1"
    cloud.google.com/gke-tpu-accelerator: tpu-v7xe-slice
  tolerations:
  - key: google.com/tpu
    operator: Exists
    effect: NoSchedule
  containers:
  - name: inference
    image: us-docker.pkg.dev/cloud-tpu-images/inference/vllm-tpu:latest
    ports:
    - containerPort: 8000
    resources:
      limits:
        google.com/tpu: 4
      requests:
        google.com/tpu: 4
    volumeMounts:
    - name: lustre-vol
      mountPath: /lustre
    - name: dshm
      mountPath: /dev/shm
    securityContext:
      privileged: true
  volumes:
  - name: lustre-vol
    persistentVolumeClaim:
      claimName: lustre-pvc        # Replace with your Lustre PVC name
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: 128Gi             # 31B Dense doesn't need large SHM
  restartPolicy: Never
```

> 💡 **SHM size**: MoE models need large SHM for expert re-quantization. 31B Dense needs 128Gi at most.
> Compare: DeepSeek-R1 671B needs 300Gi+, Qwen3.5 397B needs 200Gi+.

```bash
# Deploy
kubectl --context=$CTX apply -f gemma4-31b-pod.yaml

# Wait for ready
kubectl --context=$CTX wait --for=condition=Ready pod/gemma4-31b --timeout=600s
```

## Step 3: Download Model Weights

```bash
POD=gemma4-31b
MODEL=/lustre/models/gemma-4-31b-it

# Download Gemma4 31B IT (BF16) to Lustre (~61 GiB, Lustre ~5 min)
kubectl --context=$CTX exec $POD -- bash -c "
  mkdir -p $MODEL
  pip install -U 'huggingface_hub[hf_transfer]'
  HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
    google/gemma-4-31b-it \
    --local-dir $MODEL
"

# Verify weight files are complete
kubectl --context=$CTX exec $POD -- bash -c "ls $MODEL/*.safetensors | wc -l"

# Clean /dev/shm residuals
kubectl --context=$CTX exec $POD -- bash -c "rm -rf /dev/shm/sem.* /dev/shm/wrk_* 2>/dev/null"
```

> ⚠️ **HuggingFace access**: Gemma4 requires accepting the license agreement. Ensure your HF token has access to `google/gemma-4-31b-it`.
> Set token: `kubectl exec $POD -- bash -c "huggingface-cli login --token <your-hf-token>"`

## Step 4: Start vLLM Inference Server

### Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--tensor-parallel-size` | `1` | Single chip sufficient, no TP needed |
| `--max-model-len` | `4096` / `131072` / `262144` | Adjust per test scenario |
| `--max-num-seqs` | `160` (FP8 KV) / `80` (BF16 KV) | See KV Cache estimation above |
| `--kv-cache-dtype` | `fp8` | **Recommended**, halves KV Cache, doubles batch |
| `--no-enable-prefix-caching` | Required | Avoid potential prefix caching bugs |
| `--gpu-memory-utilization` | `0.9` | Single chip 192GB, 0.9 = 172GB available |
| `--block-size` | `256` | CI default |
| `--trust-remote-code` | Required | Gemma4 custom model code |

### Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `SKIP_JAX_PRECOMPILE` | `1` | Skip JAX pre-compilation, faster startup |
| `VLLM_XLA_CHECK_RECOMPILATION` | `0` | Disable XLA recompilation checks |

### Start Server

```bash
# 1. Write launcher
cat > /tmp/launch_gemma4.sh <<'LAUNCHER'
#!/bin/bash
cd /tmp
pgrep -f 'EngineCore|vllm' | xargs -r kill -9 2>/dev/null
sleep 2
rm -f /tmp/libtpu_lockfile /tmp/vllm_gemma4.log
touch /tmp/vllm_gemma4.log
setsid nohup env \
  SKIP_JAX_PRECOMPILE=1 VLLM_XLA_CHECK_RECOMPILATION=0 \
  vllm serve /lustre/models/gemma-4-31b-it \
    --tensor-parallel-size 1 \
    --max-num-batched-tokens 4096 --max-num-seqs 160 --max-model-len 4096 \
    --no-enable-prefix-caching --gpu-memory-utilization 0.9 \
    --kv-cache-dtype fp8 --block-size 256 --trust-remote-code \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    --async-scheduling \
    >> /tmp/vllm_gemma4.log 2>&1 < /dev/null &
disown
echo "launched pid=$!"
exit 0
LAUNCHER

# 2. cp + run
kubectl --context=$CTX cp /tmp/launch_gemma4.sh $POD:/tmp/launch_gemma4.sh
kubectl --context=$CTX exec $POD -- bash /tmp/launch_gemma4.sh

# 3. Monitor startup (~3-5 min cold start)
kubectl --context=$CTX exec $POD -- tail -f /tmp/vllm_gemma4.log
```

**Wait for key log message**:
```
INFO: Application startup complete.                    ← Startup successful
```

## Step 5: Verification

### Health Check

```bash
kubectl --context=$CTX exec $POD -- curl -sf -o /dev/null -w "%{http_code}\n" http://localhost:8000/health
# Expected: 200
```

### Smoke Test — Simple Q&A

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

### Thinking Mode Test

```bash
kubectl --context=$CTX exec $POD -- curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/lustre/models/gemma-4-31b-it",
    "messages": [
      {"role": "system", "content": "<|think|>You are a helpful assistant."},
      {"role": "user", "content": "What is 25 * 37?"}
    ],
    "max_tokens": 500,
    "temperature": 0
  }' | python3 -c 'import sys,json; r=json.load(sys.stdin); m=r["choices"][0]["message"]; print("content:", repr(m["content"][:200])); print("finish:", r["choices"][0]["finish_reason"])'
# Expected: model thinks step-by-step then outputs 925
```

## Step 6: Performance Benchmark

### Test Matrix

| ID | Scenario | Input Len | Output Len | Concurrency | max-model-len | Target Metrics |
|----|----------|-----------|------------|-------------|---------------|---------------|
| B1 | **Single user latency** | 1K | 1K | P1 | 4096 | TTFT, ITL, TPOT |
| B2 | **Standard throughput** | 1K | 1K | P64 | 4096 | tok/s, ITL |
| B3 | **Peak throughput** | 1K | 1K | P160 | 4096 | tok/s (find max batch) |
| B4 | **Long input, short output** | 16K | 1K | P4 | 32768 | TTFT (prefill perf) |
| B5 | **Short input, long output** | 1K | 16K | P4 | 32768 | ITL stability |
| B6 | **128K long context** | 128K | 256 | P1 | 131072 | TTFT, OOM check |
| B7 | **256K max context** | 256K | 256 | P1 | 262144 | TTFT, OOM check |

> ⚠️ B6/B7 long context tests require **restarting vLLM with adjusted `--max-model-len`** (131072 / 262144).
> 256K context full attention KV = 10 × 4 × 512 × 2 × 2 × 262144 = **20.5 GB** (BF16) / **10.2 GB** (FP8).
> Sliding layers remain at fixed 1024 window (~0.8 GB). Single chip 192GB can hold a single batch.

### 6.1 Short Context Tests (B1-B3, max-model-len=4096)

Use default launcher params (Step 4), no restart needed.

```bash
# B1: Single user latency (1K/1K, P1)
kubectl --context=$CTX exec $POD -- python3 -m vllm.entrypoints.openai.run_batch \
  --model /lustre/models/gemma-4-31b-it \
  --input-len 1024 --output-len 1024 \
  --num-prompts 1 2>&1 | tail -20

# B2: Standard throughput (1K/1K, P64)
kubectl --context=$CTX exec $POD -- python3 -m vllm.entrypoints.openai.run_batch \
  --model /lustre/models/gemma-4-31b-it \
  --input-len 1024 --output-len 1024 \
  --num-prompts 64 2>&1 | tail -20

# B3: Peak throughput (1K/1K, P160)
kubectl --context=$CTX exec $POD -- python3 -m vllm.entrypoints.openai.run_batch \
  --model /lustre/models/gemma-4-31b-it \
  --input-len 1024 --output-len 1024 \
  --num-prompts 160 2>&1 | tail -20
```

### 6.2 Medium-Long Context Tests (B4-B5, max-model-len=32768)

Requires vLLM restart with `--max-model-len 32768` and reduced `--max-num-seqs`.

```bash
# Restart vLLM with max-model-len=32768
cat > /tmp/launch_gemma4_32k.sh <<'LAUNCHER'
#!/bin/bash
cd /tmp
pgrep -f 'EngineCore|vllm' | xargs -r kill -9 2>/dev/null
sleep 2
rm -f /tmp/libtpu_lockfile /tmp/vllm_gemma4.log
touch /tmp/vllm_gemma4.log
setsid nohup env \
  SKIP_JAX_PRECOMPILE=1 VLLM_XLA_CHECK_RECOMPILATION=0 \
  vllm serve /lustre/models/gemma-4-31b-it \
    --tensor-parallel-size 1 \
    --max-num-batched-tokens 32768 --max-num-seqs 32 --max-model-len 32768 \
    --no-enable-prefix-caching --gpu-memory-utilization 0.9 \
    --kv-cache-dtype fp8 --block-size 256 --trust-remote-code \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    --async-scheduling \
    >> /tmp/vllm_gemma4.log 2>&1 < /dev/null &
disown
echo "launched pid=$!"
exit 0
LAUNCHER
kubectl --context=$CTX cp /tmp/launch_gemma4_32k.sh $POD:/tmp/launch_gemma4_32k.sh
kubectl --context=$CTX exec $POD -- bash /tmp/launch_gemma4_32k.sh

# Wait for cold start
for i in $(seq 1 20); do C=$(kubectl --context=$CTX exec $POD -- curl -sf -o /dev/null -w "%{http_code}" http://localhost:8000/health); echo "T+$((i*30))s HTTP $C"; [ "$C" = "200" ] && break; sleep 30; done

# B4: Long input, short output (16K/1K, P4)
kubectl --context=$CTX exec $POD -- python3 -m vllm.entrypoints.openai.run_batch \
  --model /lustre/models/gemma-4-31b-it \
  --input-len 16384 --output-len 1024 \
  --num-prompts 4 2>&1 | tail -20

# B5: Short input, long output (1K/16K, P4)
kubectl --context=$CTX exec $POD -- python3 -m vllm.entrypoints.openai.run_batch \
  --model /lustre/models/gemma-4-31b-it \
  --input-len 1024 --output-len 16384 \
  --num-prompts 4 2>&1 | tail -20
```

### 6.3 Ultra-Long Context Tests (B6-B7, 128K / 256K)

> ⚠️ **Extreme tests** to verify Gemma4's 256K context ceiling on TPU v7xe.

```bash
# Restart vLLM with max-model-len=262144 (256K)
cat > /tmp/launch_gemma4_256k.sh <<'LAUNCHER'
#!/bin/bash
cd /tmp
pgrep -f 'EngineCore|vllm' | xargs -r kill -9 2>/dev/null
sleep 2
rm -f /tmp/libtpu_lockfile /tmp/vllm_gemma4.log
touch /tmp/vllm_gemma4.log
setsid nohup env \
  SKIP_JAX_PRECOMPILE=1 VLLM_XLA_CHECK_RECOMPILATION=0 \
  vllm serve /lustre/models/gemma-4-31b-it \
    --tensor-parallel-size 1 \
    --max-num-batched-tokens 262144 --max-num-seqs 1 --max-model-len 262144 \
    --no-enable-prefix-caching --gpu-memory-utilization 0.95 \
    --kv-cache-dtype fp8 --block-size 256 --trust-remote-code \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    --async-scheduling \
    >> /tmp/vllm_gemma4.log 2>&1 < /dev/null &
disown
echo "launched pid=$!"
exit 0
LAUNCHER
kubectl --context=$CTX cp /tmp/launch_gemma4_256k.sh $POD:/tmp/launch_gemma4_256k.sh
kubectl --context=$CTX exec $POD -- bash /tmp/launch_gemma4_256k.sh

# Wait for cold start (may take longer, XLA compiling 256K shapes)
for i in $(seq 1 40); do C=$(kubectl --context=$CTX exec $POD -- curl -sf -o /dev/null -w "%{http_code}" http://localhost:8000/health); echo "T+$((i*30))s HTTP $C"; [ "$C" = "200" ] && break; sleep 30; done

# B6: 128K context (128K/256, P1)
kubectl --context=$CTX exec $POD -- python3 -m vllm.entrypoints.openai.run_batch \
  --model /lustre/models/gemma-4-31b-it \
  --input-len 131072 --output-len 256 \
  --num-prompts 1 2>&1 | tail -20

# B7: 256K context (256K/256, P1) — Gemma4 max context
kubectl --context=$CTX exec $POD -- python3 -m vllm.entrypoints.openai.run_batch \
  --model /lustre/models/gemma-4-31b-it \
  --input-len 262144 --output-len 256 \
  --num-prompts 1 2>&1 | tail -20
```

### 6.4 Results Template

| ID | Scenario | TTFT (ms) | ITL (ms) | Throughput (tok/s) | tok/s/user | Status |
|----|----------|-----------|----------|-------------------|------------|--------|
| B1 | 1K/1K P1 | — | — | — | — | ⏳ Pending |
| B2 | 1K/1K P64 | — | — | — | — | ⏳ Pending |
| B3 | 1K/1K P160 | — | — | — | — | ⏳ Pending |
| B4 | 16K/1K P4 | — | — | — | — | ⏳ Pending |
| B5 | 1K/16K P4 | — | — | — | — | ⏳ Pending |
| B6 | 128K/256 P1 | — | — | — | — | ⏳ Pending |
| B7 | 256K/256 P1 | — | — | — | — | ⏳ Pending |

> 📊 **Expected baseline** (31B Dense + single chip + BF16 weights + FP8 KV):
> - Cold start: ~2-3 min (small weights, no MoE re-quant)
> - B1 single user: expected > 50 tok/s/user (31B Dense much faster than 397B MoE)
> - B3 peak: depends on KV Cache capacity and XLA scheduling efficiency
> - B6/B7: primarily feasibility validation, TTFT may be very long (128K+ prefill)

## Step 7: Cleanup

```bash
# Delete pod
kubectl --context=$CTX delete pod gemma4-31b

# Delete node pool (optional)
gcloud container node-pools delete np-tpu7xe-spot-gemma4 \
  --cluster=$CLUSTER --region=$REGION --project=$PROJECT --quiet --async
```

---

## 📋 Troubleshooting

| Problem | Possible Cause | Solution |
|---------|---------------|----------|
| `RESOURCE_EXHAUSTED` creating node pool | v7xe spot capacity exhausted | Try different zone or wait |
| Weight download 403 | HF token lacks Gemma4 access | Accept license on HuggingFace |
| Startup hangs / no logs | libtpu lockfile residual | `rm -f /tmp/libtpu_lockfile` |
| Repeated token output | Known bug vllm#39827 | Lower temperature or wait for upstream fix |
| `ImportError: gemma4` | tpu-inference version too old | Ensure latest main branch image |
| OOM on model load | Should not happen (31B << 192GB) | Check for other processes using HBM |

---

## 📎 References

- [Gemma 4 Model Card](https://ai.google.dev/gemma/docs/core/model_card_4) — Official model specs
- [vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference) — TPU inference backend
- [Gemma4 Issues](https://github.com/vllm-project/tpu-inference/issues?q=gemma4) — Known issue tracking
- [HuggingFace: google/gemma-4-31b-it](https://huggingface.co/google/gemma-4-31b-it) — Model weights

---

---

## 🧪 Actual Test Results (2026-05-13)

### Environment

| Item | Value |
|------|-------|
| Cluster | chrisya-v7x-v134 (cloud-tpu-multipod-dev, us-central1) |
| Node Pool | np-tpu7x-spot-gemma4 (2x2x1, 4 chips, Spot) |
| Image | vllm/vllm-tpu:nightly (vLLM 0.20.2rc1.dev223) |
| Model | gemma-4-31b-it BF16 (59 GB, 2 safetensors) |
| Weight Download | Lustre, 40 seconds |

### Test Results

| Config | Cold Start | Smoke Test (25 tok) | 1K Token Prompt | Status |
|--------|-----------|---------------------|-----------------|--------|
| **TP=1** | 3 min | ✅ "Paris" | ❌ VMEM OOM (RPA Pallas kernel scratch) | **Short prompt OK, long prompt crashes** |
| **TP=4** | 3.5 min | ❌ XLA layout error | — | **Cannot run inference** |

### Bug 1: TP=1 VMEM OOM (Long Prompt)

Smoke test (25 tokens) outputs "Paris" correctly, but 1024 token prompt triggers VMEM OOM:

```
RPAm-p_256-bq_512_512-bkv_2048_512-sw_1024/pallas_call
Largest program allocations in vmem:
  1. Size: 36.00M  Shape: f8e4m3fn[2,2048,9,4,256]  Tag: scratch operand
```

**Root cause**: Gemma4's hybrid attention (sliding_window=1024 + global, head_dim=256/512) causes the RPA kernel to need scratch buffers that exceed single-chip VMEM capacity. Short prompts have smaller scratch and pass; long prompts trigger larger allocations.

### Bug 2: TP=4 XLA Reshape Layout

```
jax.errors.JaxRuntimeError: FAILED_PRECONDITION: 
  Reshape should have supported layout before reaching the emitter.
```

**Root cause**: Gemma4's attention head configuration (sliding: 16 KV heads × 256 dim, global: 4 KV heads × 512 dim) produces tensor shapes that are not supported by the XLA emitter after TP=4 sharding.

### Pitfall Log

| # | Problem | Solution | Time |
|---|---------|----------|------|
| 1 | GKE accelerator label `tpu-v7-lite-podslice` rejected | Correct label: `tpu7x` | 11:30 |
| 2 | Pod exits immediately without entrypoint | Add `command: ["sleep", "infinity"]` | 11:35 |
| 3 | Spot node preempted | Redeploy after node recreated | 11:40 |
| 4 | `huggingface-cli` deprecated | Use `hf` command instead | 11:45 |
| 5 | evalscope random dataset needs tokenizer-path | Add `--tokenizer-path` | 12:50 |
| 6 | TP=1 long prompt VMEM OOM | **Unresolved** — tpu-inference RPA kernel bug | 12:53 |
| 7 | TP=4 XLA Reshape layout incompatible | **Unresolved** — tpu-inference XLA bug | 12:58 |

### Conclusion

> ⚠️ **Gemma4 31B cannot be benchmarked on current tpu-inference nightly (0.20.2rc1).**
>
> - TP=1 can only handle very short prompts (<100 tokens); long prompts trigger VMEM OOM
> - TP=4 cannot run inference at all (XLA layout incompatibility)
> - Waiting for upstream tpu-inference fixes for Gemma4 RPA kernel and XLA compatibility
> - Related issues: [#2126](https://github.com/vllm-project/tpu-inference/issues/2126), [vllm#39827](https://github.com/vllm-project/vllm/issues/39827)

---

> **Document version**: v0.2 (added actual test results — Gemma4 not benchmark-ready)
>
> **Last updated**: 2026-05-13
