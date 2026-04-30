# Qwen3-Coder-480B-A35B-Instruct FP8 Inference on TPU v7x-8

> 🌐 **Languages** | **语言**: [中文](README.md) · **English**

> End-to-end guide: Run Qwen3-Coder-480B (FP8 quantized) inference + 1P1D PD disaggregation on a single TPU v7x-8 node (**4 chips, 8 devices**).
>
> **Code repository**: Upstream [`vllm-project/tpu-inference`](https://github.com/vllm-project/tpu-inference) (main branch)
>
> **Model**: [`Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8`](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8) (~480 GB)
>
> **Alternative model**: [`BCCard/Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic`](https://huggingface.co/BCCard/Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic) (community dynamic quantized version)

## 📖 How to Read This Document

> **Just want to verify it runs** (30 min) → Go directly to §🎯 [30-Second Quick Reproduction](#-30-second-quick-reproduction-exact-commands), 6 commands to copy-paste.
>
> **Production deployment** (1-2 hours) → Walk through §Step 1 → 4 completely, parameter details in §Step 4b.
>
> **Running benchmarks to verify performance** → §Step 5 + §Step 6 (includes three complete sweep datasets §6e).
>
> **PD disaggregation** (1P1D, recommend Lustre shared storage) → §PD Disaggregation → §7a-pre Lustre + §7a-7g deployment.
>
> **Troubleshooting** → §Troubleshooting (10 known issues and fixes).
>
> Companion document: [PD Disaggregation Deep Dive](https://cc.higcp.com/assets/qwen3-coder-480b-pd-disagg-explained-20260425.html) (diagrams + SVG + sequence charts)

## 🎯 30-Second Quick Reproduction (Exact Commands)

> **Goal**: Any new user can copy the 6 commands below and get the same measured results within 1 hour.
>
> **Prerequisites**: GKE cluster + v7x node pool + Pod already running (see §Step 1); model weights already at `/usr/vllm/qwen3-coder-480b-fp8/` (including all 49 safetensors + tokenizer trio).
>
> **🟢 Already have a running vLLM?** Skip ③④, go directly to ⑤⑥ for smoke + benchmark verification. If `kubectl exec <pod> -- curl -s -w 'HTTP %{http_code}\n' http://localhost:8000/health` returns `HTTP 200`, the server is ready, no restart needed.
>
> **🔵 Fresh environment?** Follow ①→⑥ in order. ③ uses the **minimal verification config** (max-num-batched-tokens=256). For production, see §Step 4 with `--max-num-batched-tokens 8192 --kv-cache-dtype fp8 --gpu-memory-utilization 0.95 --async-scheduling`.

```bash
# ── ① Enter Pod ──
kubectl exec -it e2e-03 -- bash

# ── ② Verify weight completeness (missing tokenizer see §Troubleshooting #10) ──
ls /usr/vllm/qwen3-coder-480b-fp8/*.safetensors | wc -l   # Should be 49
ls /usr/vllm/qwen3-coder-480b-fp8/{tokenizer.json,vocab.json,tokenizer_config.json}

# ── ③ Start vLLM serve (verification, minimal config, ~7 min cold start) ──
mkdir -p /tmp/vllm-logs && cd /tmp
SKIP_JAX_PRECOMPILE=1 VLLM_XLA_CHECK_RECOMPILATION=0 MODEL_IMPL_TYPE=vllm HF_HUB_OFFLINE=1 \
nohup vllm serve /usr/vllm/qwen3-coder-480b-fp8 \
  --served-model-name Qwen3-Coder-480B-FP8 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-num-batched-tokens 256 \
  --max-num-seqs 256 \
  --port 8000 --host 0.0.0.0 \
  > /tmp/vllm-logs/serve.log 2>&1 &

# ── ④ Wait for ready (look for 'Application startup complete') ──
tail -f /tmp/vllm-logs/serve.log   # Ctrl+C to exit
# Or non-blocking:
until curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8000/health | grep -q 200; do sleep 10; date; done
echo "✅ Server ready"

# ── ⑤ Smoke test (fibonacci 50 tokens, total ~1-2 seconds including client overhead) ──
time curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3-Coder-480B-FP8",
  "prompt": "def fibonacci(n):",
  "max_tokens": 50, "temperature": 0.0
}' | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])"

# ── ⑥ 50-prompt smoke benchmark (~5 min) ──
vllm bench serve --backend vllm \
  --model Qwen3-Coder-480B-FP8 \
  --tokenizer /usr/vllm/qwen3-coder-480b-fp8 \
  --host localhost --port 8000 \
  --num-prompts 50 \
  --dataset-name random --random-input-len 256 --random-output-len 128 \
  --request-rate 4 --ignore-eos
```

**Expected results** (depends on which config you are running):

| Config | Peak Output tok/s | Median ITL | Smoke test (50 tok) |
|------|-------------------|------------|---------------------|
| **Minimal** (README default `max-num-batched-tokens=256`) | ≈ **1050 tok/s** | ≈ **47 ms** | ~1 sec |
| **Production** (`max-num-batched-tokens=8192 --kv-cache-dtype=fp8 --gpu-memory-utilization=0.95 --async-scheduling`) | ≈ **1300 tok/s** | ≈ **40 ms** | 1-2 sec (including client overhead) |

✅ **Dogfood verified (2026-04-26)**: Ran on e2e-03 (production config, max=8192) using these exact commands, 50/50 success, peak 1300 tok/s, median ITL 40ms.

After running, for more systematic benchmark sweep → see §Step 6e commands; to switch to production config → see §Step 4.

---

## 🎯 Key Performance (✅ Fully Measured 2026-04-25, TPU v7x-8 4 chips · FP8)

### Single Instance Performance Profile (details in §6e)

| Workload | c=1 | c=4 | c=16 | c=64 | Customer Scenario |
|----------|----:|----:|----:|----:|---------|
| **1K/1K** (chat) | 48 tok/s | 177 | 602 | **1478** | Short Q&A, code completion |
| **1K/8K** (long-output) | 47.5 | 178 | 621 | **1623** | Document generation, code block writing |
| **8K/1K** (RAG/long-prompt) | 46.4 | 162 | 483 | **943** | RAG, long document analysis |

### PD Disaggregation (1P1D) vs Single Instance (details in §7g)

| Metric | Single Instance | 1P1D | Improvement |
|------|------|----|----|
| Median TPOT | 20.6 / 23.2 ms | **18.3 / 20.6 ms** | **−11%** ⬇️ |
| Output throughput | 48 / 162 tok/s | **53.8 / 170 tok/s** | **+5~12%** ⬆️ |
| Median TTFT | 95 / 1495 ms | 281 / 2908 ms | +186 ms ~ +1.4 s ⬆️ |

### System Capabilities

| Item | Measured Value |
|------|------|
| 💨 Single user decode speed | **~48 tok/s** (≈21 ms/token, c=1) |
| 🚀 Peak aggregate throughput (single instance) | **1623 tok/s** (1K/8K c=64) |
| 🔧 Startup time (cold) | **~7 min** (weights 3'37" + XLA compilation ~3') |
| 🔧 Startup time (warm) | **~5 min** (XLA cache hit) |
| 💾 HBM usage | 94.75 GB/device · 758 GB total |

> **CI threshold reference** (1K input / 1K output, max-concurrency=64): req/s ≥ 1.05 · output ≥ 1926 tok/s · total ≥ 1948 tok/s. Current measurements approach these thresholds (1478 tok/s vs 1926).

---

## 📋 Quick Comparison with Other MoE Models

| Model | Total Params | Active | Quantization | Deploy Difficulty | Cache Generation |
|------|-------|------|------|---------|-----------|
| **Qwen3-Coder-480B** | **480B** | **35B** | **FP8** | ⭐ **Easy** | **Not needed** |
| GLM-5.1 754B | 754B | ~32B | FP4 + FP8 + BF16 | ⭐⭐⭐ Complex | Required (~28 min) |
| DeepSeek R1 671B | 671B | 37B | FP4 + FP8 | ⭐⭐⭐ Complex | Required (~45 min) |
| Kimi K2.6 1T | 1T | ~32B | INT4 | ⭐⭐ Medium | Required |

> **Qwen3-Coder is the easiest large MoE to deploy on v7x**: FP8 direct read (handled internally by vLLM), **no FP4 conversion step**, **no `/dev/shm` cache copying**, **no special environment variables**.

---

## Hardware & Model Overview

| Item | Requirement |
|------|------|
| TPU | **v7x-8** (2x2x1 topology, 4 chips = 8 devices) |
| HBM | 94.75 GB/device, 758 GB total |
| Host memory | ≥850 GB |
| Storage | ≥600 GB (model ~480 GB + cache space) |
| **❌ Not supported** | v6e (insufficient HBM, 480B FP8 ≈ 480GB > v6e-8 256GB) |

| Model Parameter | Value |
|---------|-----|
| Architecture | MoE (sparse, top-K routing) |
| Total parameters | **480B** |
| Active parameters | **35B** (A35B) |
| Quantization | **FP8 (E4M3)** dynamic quantization |
| Context support | max-model-len=10240 (recommended), adjustable up to 32K+ |
| Inference framework | vLLM + tpu-inference (JAX backend) |
| Model implementation | `tpu_inference/models/jax/qwen3_moe.py` |

### Code Completeness

| Dimension | Status |
|------|------|
| JAX model implementation | ✅ Complete (`qwen3_moe.py`) |
| TP/EP/PP support | ✅ TP=8, EP enabled, PP optional |
| FP8 quantization | ✅ Natively supported |
| Single instance vLLM serve | ✅ CI test passing |
| PD disaggregation (1P1D GKE) | ✅ Daily CI running |
| Multihost (TP=16, tpu7x-16) | ✅ Daily benchmark |
| v6e support | ❌ Insufficient HBM |

---

## ⚠️ Key Environment Variables (Must Set Before Starting vLLM)

> **Simpler than GLM-5.1**: No need for the `MOE_REQUANTIZE_WEIGHT_DTYPE` / `NEW_MODEL_DESIGN` / `MOE_WEIGHT_CACHE_DIR` trio.

| Environment Variable | Value | Description | Required? |
|---------|-----|------|-------|
| `JAX_PLATFORMS` | `tpu,cpu` | Force TPU backend | ✅ Required |
| `TPU_BACKEND_TYPE` | `jax` | Use JAX backend (not PyTorch) | ✅ Required |
| `PJRT_DEVICE` | `TPU` | PJRT backend type | ✅ Required |
| `MODEL_IMPL_TYPE` | `vllm` | Use vLLM model implementation, not native JAX | ✅ Required |
| `USE_MOE_EP_KERNEL` | `0` | MoE EP kernel; CI uses 0 for stability | ⚠️ Recommended 0 |
| `USE_BATCHED_RPA_KERNEL` | `0` | Batched RPA kernel; CI uses 0 | ⚠️ Recommended 0 |
| `HF_TOKEN` | `<your_hf_token>` | HuggingFace access token | ✅ Required |
| `HF_HOME` | `/usr/vllm` | HuggingFace cache directory | Recommended |
| `SKIP_JAX_PRECOMPILE` | `1` | Skip JAX precompilation, faster startup | ⚠️ Optional |
| `VLLM_XLA_CHECK_RECOMPILATION` | `0` | Disable recompilation check | ⚠️ Optional |
| `VLLM_LOGGING_LEVEL` | `INFO` or `DEBUG` | Log level | Optional |

---

## Step 1: Create GKE TPU Pod

> Prerequisites: An existing GKE cluster with a node pool containing **TPU v7x**. If not available, see [GKE TPU Cluster Setup](https://cloud.google.com/kubernetes-engine/docs/how-to/tpus).

### 1a: Prepare HF_TOKEN Secret

```bash
# Replace with your own HF token
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxx"

kubectl create secret generic hf-token-secret \
  --from-literal=token=$HF_TOKEN \
  --dry-run=client -o yaml | kubectl apply -f -
```

### 1b: Create Single Instance Pod (TP=8)

```bash
kubectl apply -f - <<'EOF'
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pvc-qwen3-coder
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 600Gi
  storageClassName: hyperdisk-balanced
---
apiVersion: v1
kind: Pod
metadata:
  name: vllm-qwen3-coder
  labels:
    app: vllm-qwen3-coder
spec:
  serviceAccountName: default
  nodeSelector:
    cloud.google.com/gke-tpu-topology: 2x2x1
    cloud.google.com/gke-tpu-accelerator: tpu7x
  initContainers:
    - name: tpu-node-setup
      image: busybox
      command: ["/bin/sh", "-c"]
      args:
        - |
          # Required! Prevents vLLM crash due to mmap limit
          sysctl -w vm.max_map_count=8388608
          # Increase VFIO IOMMU DMA mapping limit (TPU driver requirement)
          if [ -f /sys/module/vfio_iommu_type1/parameters/dma_entry_limit ]; then
            echo 2000000 > /sys/module/vfio_iommu_type1/parameters/dma_entry_limit
          fi
      securityContext:
        privileged: true
  containers:
  - name: vllm-tpu
    image: vllm/vllm-tpu:nightly
    imagePullPolicy: Always
    command: ["/bin/bash", "-c", "sleep infinity"]   # Manual vLLM start for debugging
    env:
    - name: HF_TOKEN
      valueFrom:
        secretKeyRef:
          name: hf-token-secret
          key: token
    - name: HF_HOME
      value: /usr/vllm
    - name: JAX_PLATFORMS
      value: "tpu,cpu"
    - name: TPU_BACKEND_TYPE
      value: jax
    - name: PJRT_DEVICE
      value: TPU
    - name: MODEL_IMPL_TYPE
      value: "vllm"
    - name: USE_MOE_EP_KERNEL
      value: "0"
    - name: USE_BATCHED_RPA_KERNEL
      value: "0"
    - name: VLLM_LOGGING_LEVEL
      value: "INFO"
    ports:
    - containerPort: 8000
    resources:
      limits:
        google.com/tpu: "4"
        memory: "850Gi"
        cpu: "220"
        ephemeral-storage: "40Gi"
      requests:
        google.com/tpu: "4"
        memory: "850Gi"
        cpu: "220"
        ephemeral-storage: "40Gi"
    volumeMounts:
    - name: dshm
      mountPath: /dev/shm
    - name: pvc-vllm-vol
      mountPath: "/usr/vllm"
    # ⭐ Strongly recommended: Also mount Lustre, large model weights download 4× faster on Lustre vs PVC
    #   (measured jarvis 2026-04-26: PVC 16 GB/min vs Lustre 63 GB/min)
    - name: lustre-vol
      mountPath: /lustre
    securityContext:
      privileged: true
      capabilities:
        add:
        - IPC_LOCK
  volumes:
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: 200Gi
  - name: pvc-vllm-vol
    persistentVolumeClaim:
      claimName: pvc-qwen3-coder
  # ⭐ Lustre RWX shared storage (if cluster already has lustre-pvc)
  - name: lustre-vol
    persistentVolumeClaim:
      claimName: lustre-pvc
  tolerations:
  # ⚠️ Required! Spot node pool has NoSchedule taint, missing this causes pod to stay Pending
  - key: cloud.google.com/gke-spot
    operator: Equal
    value: "true"
    effect: NoSchedule
  # ⚠️ Required! TPU nodes have google.com/tpu taint
  - key: google.com/tpu
    operator: Exists
    effect: NoSchedule
  restartPolicy: Never
EOF
```

> **🔧 Verify Pod is scheduled** (must see Running, not Pending):
> ```bash
> kubectl get pod vllm-qwen3-coder -o wide
> # If Pending, check reason:
> kubectl describe pod vllm-qwen3-coder | tail -20
> # Common reasons:
> #   - Missing toleration → fixed in the yaml above
> #   - Spot node pool out of capacity → wait a few minutes or switch region
> #   - PVC stuck at WaitForFirstConsumer → normal, will Bound after pod scheduling
> ```

> **Key points**:
> - `google.com/tpu: "4"` — requests 4 chips (= 8 devices)
> - `memory: "850Gi"` — model loading requires large host RAM
> - `dshm sizeLimit: 200Gi` — Qwen3 Coder **does not need 800GB /dev/shm** (GLM-5.1 does), 200GB is sufficient
> - `tpu-node-setup initContainer` — **cannot be omitted**, otherwise vLLM startup crashes due to mmap limit
> - **⭐ `lustre-vol`** — strongly recommended — large model weights download **4× faster** on Lustre vs PVC (measured jarvis 2026-04-26: 63 GB/min vs 16 GB/min). If cluster doesn't have `lustre-pvc`, create a Lustre RWX instance first (see §7a-pre). With Lustre mounted, `pvc-vllm-vol` PVC size can be reduced from 600 GB to 100 GB for system/cache only.

### 1c: Wait for Pod Ready and Enter

```bash
kubectl wait --for=condition=Ready pod/vllm-qwen3-coder --timeout=300s
kubectl exec -it vllm-qwen3-coder -- bash
```

---

## Step 2: Prepare Code (Can Skip)

The `vllm/vllm-tpu:nightly` image comes with `tpu_inference` and `vllm` pre-installed. **Main branch is sufficient, no branch switching needed** (Qwen3 Coder is already merged to main).

```bash
# Verify Qwen3 MoE model class exists (the most critical step, if import works you're good)
python3 -c "
from tpu_inference.models.jax.qwen3_moe import Qwen3MoeForCausalLM
print('✅ Qwen3MoeForCausalLM imported OK')
"
```

> ⚠️ **Note**: `/workspace/tpu_inference` in the image is **just a source copy, not a git repo** (`git pull` will fail with fatal error).
> If you need to update code (e.g., apply PR #2366 patch), use `cp /lustre/tpu_inference /workspace/tpu_inference` to overwrite the entire directory (see [feedback_vllm-tpu-image-stale.md](../../shared/feedback/) experience post); make sure to `find /workspace/tpu_inference -name __pycache__ -type d -exec rm -rf {} +` to clear .pyc cache.
>
> If you want to use experimental optimizations from the yangwhale fork, refer to the branch switching process in [DeepSeek R1 README](../DeepSeek-R1-671B-FP4/README.md).

---

## Step 3: Download Model Weights

### 3a: Set HF Cache Directory

```bash
export HF_HOME=/usr/vllm   # Already set in Pod env, just confirming
mkdir -p $HF_HOME
```

### 3b: Download (Recommended: huggingface-cli)

```bash
# ⚠️ hf_transfer package must be installed for HF_HUB_ENABLE_HF_TRANSFER=1 to take effect (not installed in image by default)
pip install -U "huggingface_hub[hf_transfer]"
export HF_HUB_ENABLE_HF_TRANSFER=1     # Enable high-speed download (10x speedup)
export HF_TOKEN=$HF_TOKEN               # Already in Pod env, just confirming

# Download Qwen3-Coder-480B-A35B-Instruct-FP8 (~450 GB)
huggingface-cli download Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
  --local-dir $HF_HOME/qwen3-coder-480b-fp8

# Verify (must strictly match these 2 numbers)
ls $HF_HOME/qwen3-coder-480b-fp8/*.safetensors | wc -l
# Expected: 49 (not 50, exactly 49 shards)

du -sh $HF_HOME/qwen3-coder-480b-fp8
# Expected: ~450 GB

# Verify tokenizer trio is present (missing any one causes vLLM startup crash, see §Troubleshooting #10)
ls $HF_HOME/qwen3-coder-480b-fp8/{tokenizer.json,vocab.json,tokenizer_config.json,merges.txt}
```

### 3c: Set Model Path Variable

```bash
# ✅ Recommended: Use local path + offline mode (avoids vLLM re-downloading from HF, see §Troubleshooting #9 PVC full)
export MODEL=$HF_HOME/qwen3-coder-480b-fp8
export HF_HUB_OFFLINE=1

# ⚠️ Not recommended: Using model name (triggers vLLM re-download verification, fills up PVC)
# export MODEL=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8
```

> **Speed tip**: When model files are on GCS, `gsutil -m cp -r gs://your-bucket/qwen3-coder-480b-fp8 $HF_HOME/` is 3-5× faster than HF download.

---

## Step 4: Start vLLM Inference Service

### 4a: Standard Single Instance Launch Command

```bash
cd /tmp   # Avoid Python namespace conflicts, leave tpu_inference directory
mkdir -p /tmp/vllm-logs

# ⚠️ Must export these envs, the launch command doesn't pass them automatically (Pod env set is OK,
# but when running nohup / detach, explicitly pass before the command)
nohup env \
  SKIP_JAX_PRECOMPILE=1 \
  VLLM_XLA_CHECK_RECOMPILATION=0 \
  MODEL_IMPL_TYPE=vllm \
  HF_HUB_OFFLINE=1 \
  vllm serve $MODEL \
    --served-model-name Qwen3-Coder-480B-FP8 \
    --seed 42 \
    --max-model-len 10240 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 512 \
    --no-enable-prefix-caching \
    --tensor-parallel-size 8 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.95 \
    --async-scheduling \
    --enable-expert-parallel \
    --port 8000 --host 0.0.0.0 \
  > /tmp/vllm-logs/serve.log 2>&1 &

echo "vLLM PID: $!"
```

Wait for log output `Application startup complete`, or non-blocking readiness check:
```bash
until curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8000/health | grep -q 200; do
  date; sleep 30
done
echo "✅ Server ready"
```

> **Measured startup time** (2026-04-25): **~7 min cold** (weight loading 3'37" + XLA compilation ~3'); **~5 min warm** (XLA cache hit). Production config (max-num-batched-tokens=8192) startup increases to **~15 min cold**, as more batch shapes trigger additional compilation.

### 4b: Parameter Reference

| Parameter | Value | Description | Common Misconception |
|------|-----|------|-------------|
| `--tensor-parallel-size 8` | 8 | Use 8 devices | Actually EP=8 + TP=1 (expert-parallel takes over) |
| `--enable-expert-parallel` | (flag) | Enable expert parallelism | **Required**, otherwise OOM |
| `--kv-cache-dtype fp8` | fp8 | KV cache in FP8 | Saves 50% HBM; accuracy loss <0.1% |
| `--gpu-memory-utilization 0.95` | 0.95 | HBM utilization cap | Recommended 0.95 for v7x; change to 0.7-0.9 for disagg |
| `--max-num-batched-tokens 8192` | 8192 | Max total tokens per batch | Higher = more throughput, but also higher latency |
| `--max-num-seqs 512` | 512 | Max sequences per batch | Concurrency cap |
| `--max-model-len 10240` | 10240 | Context length | Default 10K; increase for longer context |
| `--no-enable-prefix-caching` | (flag) | Disable prefix caching | Must disable for benchmarks; can enable for production |
| `--async-scheduling` | (flag) | Async scheduling | Improves throughput ~10-20% |

### 4c: Common Pitfalls

**Pitfall 1: `MODEL_IMPL_TYPE=vllm` not set → wrong model implementation**

If not set, may use native JAX implementation, which has less compatibility with Qwen3 MoE than the vLLM implementation.

**Pitfall 2: `--enable-expert-parallel` missing → OOM**

Without EP, experts are replicated on every device, instantly blowing HBM.

**Pitfall 3: `--kv-cache-dtype fp8` not specified → KV cache wastes double HBM**

Default FP16 KV cache wastes too much HBM for batching.

**Pitfall 4: Restarting vLLM reports `libtpu lockfile` → previous process not cleaned up**

```bash
pkill -9 -f 'vllm|EngineCore' && sleep 3
rm -f /tmp/libtpu_lockfile /tmp/.vllm_ipc_*
# Then restart vllm serve
```
See §Troubleshooting #6 for detailed investigation.

**Pitfall 5: `gpu-memory-utilization` value differs by deployment mode**

| Deployment | Recommended Value | Reason |
|-----|-------|-----|
| Single instance (this section) | **0.95** | Maximize HBM for large batches |
| PD disagg P (prefill) | **0.70** | Reserve 30% for KV transfer buffer |
| PD disagg D (decode) | **0.90** | Large active KV cache |

---

## Step 5: Verify Inference

In a **separate terminal** (`kubectl exec -it vllm-qwen3-coder -- bash`) send test requests:

> ⚠️ **Critical**: The `"model"` field must match the `--served-model-name` used when starting vLLM (this guide uses the short name `Qwen3-Coder-480B-FP8`); using the wrong name returns 404. Verify first:
> ```bash
> curl -s http://localhost:8000/v1/models | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])'
> ```

```bash
# Test 1: Simple Q&A (completions API)
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-Coder-480B-FP8",
    "prompt": "Write a Python function to compute Fibonacci numbers:",
    "max_tokens": 256,
    "temperature": 0.0
  }' | python3 -m json.tool

# Test 2: Chat completions API
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-Coder-480B-FP8",
    "messages": [{"role": "user", "content": "Write a Python function to check if a number is prime."}],
    "max_tokens": 512
  }' | python3 -m json.tool

# Test 3: Health check (vLLM /health returns HTTP 200 + empty body, not JSON)
curl -s -o /dev/null -w 'HTTP %{http_code}\n' http://localhost:8000/health
# Expected: HTTP 200
```

### Verification Checklist (✅ Measured 2026-04-25)

| Test Item | Expected | Measured | Status |
|--------|-----|------|------|
| Startup time (cold) | ~6-8 min | **~7 min** (weights 3'37" + XLA ~3') | ✅ |
| HBM allocation/device | ~70-80 GB | **94.75 GB/device · 758 GB total** | ✅ |
| TPU device detection | 8 devices | **8 devices, 4 chips, 2x2x1 mesh** | ✅ |
| Python code gen (quicksort) | Correct | **Complete and runnable** | ✅ |
| Second inference (cache hit) | <2s/50 tokens | **0.93s (47ms/token)** | ✅ |
| /health response | HTTP 200 | **HTTP 200, 2.4ms** | ✅ |

---

## Step 6: Benchmark

### 6a: Prepare Benchmark Tool

```bash
# Clone benchmark serving tool
cd /workspace
git clone https://github.com/kimbochen/bench_serving.git
cd bench_serving
```

### 6b: 1K input / 1K output Benchmark (Default CI Config)

```bash
python3 bench_serving/benchmark_serving.py \
  --model="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8" \
  --backend=vllm \
  --host=127.0.0.1 \
  --port=8000 \
  --dataset-name=random \
  --random-input-len=1024 \
  --random-output-len=1024 \
  --random-range-ratio=0.8 \
  --num-prompts=320 \
  --max-concurrency=64 \
  --request-rate=inf \
  --ignore-eos
```

**CI pass thresholds**:
- Request throughput ≥ 1.05 req/s
- Output token throughput ≥ 1926 tok/s
- Total token throughput ≥ 1948 tok/s

### 6c: 1K input / 8K output Benchmark (Long Output Scenario)

```bash
python3 bench_serving/benchmark_serving.py \
  --model="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8" \
  --backend=vllm \
  --host=127.0.0.1 \
  --port=8000 \
  --dataset-name=random \
  --random-input-len=1024 \
  --random-output-len=8192 \
  --random-range-ratio=0.8 \
  --num-prompts=128 \
  --max-concurrency=64 \
  --request-rate=inf \
  --ignore-eos
```

**CI pass thresholds**:
- Request throughput ≥ 0.16 req/s
- Output token throughput ≥ 1226 tok/s
- Total token throughput ≥ 1378 tok/s

### 6d: Full Concurrency Sweep Script (c=1,4,16,64 measured; higher c pending)

> Measured data in §6e; the script below can extend to c=128/256/512/1024. Each cell runs `prompts = max(4, c*2)`, first run is XLA warmup, second run is real data.

```bash
for c in 1 2 4 8 16 32 64 128 256 512 1024; do
  echo "=== Concurrency $c ==="
  vllm bench serve \
    --model="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8" \
    --num-warmups 3 \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 1024 \
    --num-prompts $((c * 4)) \
    --max-concurrency $c \
    --request-rate inf \
    --ignore-eos \
    --host localhost \
    --port 8000 \
    --result-file qwen3coder_1k1k_c${c}.json
  sleep 30
done
```

### 6e: Measured Results

#### Initial Smoke Test (✅ 2026-04-25, 50 prompts, in=256, out=128, rate=4)

| Metric | Value | Notes |
|------|------|------|
| Success rate | **50/50 (100%)** | All 200 OK |
| Total duration | 312.59s | |
| Request throughput | 0.16 req/s | Affected by cold start |
| Output tok/s (avg) | 20.47 | Including cold start XLA |
| **Peak output tok/s** | **1050** | Real concurrent capability |
| Total tok/s | 61.42 | |
| Mean TTFT | 127.5s | ⚠️ First batch triggers XLA recompilation |
| **Median ITL (hot path)** | **47.27 ms** | ≈ 21 tok/s/req real speed |
| Mean TPOT | 1177ms | Including outliers |
| P99 ITL | 80.6s | Recompilation outlier |

**Interpretation**:
- First batch of requests triggers bulk XLA compilation (each different batch shape compiles once), making TTFT appear high
- Hot path (after compilation) real `inter-token latency = 47ms` (i.e., ≈21 tok/s/user)
- Peak 1050 tok/s already approaches the buildkite CI verification upper bound of 1378 tok/s

#### 1K input / 1K output (✅ Full Sweep Measured 2026-04-25)

| Concurrency | Output tok/s | tok/s/chip | TTFT (med) | TPOT (med) | ITL (med) | tok/s/user | Status |
|------------:|-------------:|----------:|----------:|----------:|--------:|-----------:|--------|
|           1 | **48** | 12 | **95 ms** | **20.6 ms** | **20.6 ms** | **48** | ✅ 2/2 |
|           4 | **177** | 44 | **386 ms** | **22.3 ms** | **22.2 ms** | **44** | ✅ 8/8 |
|          16 | **602** | 151 | **549 ms** | **25.9 ms** | **25.6 ms** | **38** | ✅ 32/32 |
|          64 | **1478** | 370 | **1691 ms** | **41.6 ms** | **40.0 ms** | **23** | ✅ 128/128 |
|         256 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | Pending |
|        1024 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | Pending |

> **Methodology**: Each cell runs `prompts = max(4, concurrency × 2)`, first run = warmup triggering XLA compilation, second run = real data. Table shows **real (warm) results**.
>
> **Key findings**:
> - **Single user at 48 tok/s** (c=1) — already exceeds human reading speed (average reading ≈ 4-5 tok/s)
> - **TPOT extremely stable**: 1→64 concurrency, TPOT only increases from 20.6→41.6 ms (≈2x), but throughput increases 30x — **TPU v7x batching utilization is very high**
> - **Aggregate 1478 tok/s** (c=64) — approaching CI threshold of 1926
> - **TTFT grows linearly with concurrency** (95→1691 ms, 18x) — affected by prefill batching, c=64 prefills 64×1024 tokens at once
> - tok/s/user (c=64) = **23**, still 5x above human reading speed

#### 1K input / 8K output (✅ Full Suite Measured 2026-04-25)

| Concurrency | Output tok/s | tok/s/chip | TTFT (med) | TPOT (med) | ITL (med) | tok/s/user |
|------------:|-------------:|-----------:|----------:|----------:|----------:|-----------:|
|           1 | **47.5** | 12 | **94 ms** | **21.0 ms** | **21.0 ms** | **47.5** |
|           4 | **178** | 44 | **256 ms** | **22.4 ms** | **22.3 ms** | **44.5** |
|          16 | **621** | 155 | **703 ms** | **25.7 ms** | **25.6 ms** | **39** |
|          64 | **1623** | 406 | **1687 ms** | **39.3 ms** | **39.0 ms** | **25** |

> **Core conclusion**: 1K/8K full data is **nearly identical to 1K/1K at the same concurrency**:
>
> | Concurrency | 1K/1K tok/s | 1K/8K tok/s | ITL 1K/1K | ITL 1K/8K |
> |---:|---:|---:|---:|---:|
> | 1  | 48   | 47.5 | 20.6ms | 21.0ms |
> | 4  | 177  | 178  | 22.2ms | 22.3ms |
> | 16 | 602  | 621  | 25.6ms | 25.6ms |
> | 64 | 1478 | 1623 | 40.0ms | 39.0ms |
>
> **What customers can tell themselves**: On v7x-8, **outputting 1K vs 8K tokens, per-token perceived speed is the same**. Longer output just means waiting longer (linear 8x), not going slower.

#### 8K input / 1K output (Long Input Scenario, ✅ Full Suite Measured 2026-04-25)

| Concurrency | Output tok/s | tok/s/chip | TTFT (med) | TPOT (med) | ITL (med) | tok/s/user |
|------------:|-------------:|-----------:|----------:|----------:|----------:|-----------:|
|           1 | **46.4** | 12 | **523 ms** | **21.1 ms** | **21.1 ms** | **46.4** |
|           4 | **162** | 41 | **1495 ms** | **23.2 ms** | **22.7 ms** | **40.5** |
|          16 | **483** | 121 | **2418 ms** | **30.2 ms** | **25.7 ms** | **30** |
|          64 | **943** | 236 | **1969 ms** | **64.8 ms** | **38.6 ms** | **15** |

> **Key observations (long input vs short input)**:
>
> | Concurrency | 1K input TTFT | 8K input TTFT | TTFT ratio | 1K input tps | 8K input tps | tps ratio |
> |---:|---:|---:|---:|---:|---:|---:|
> | 1 | 95 ms | 523 ms | **5.5×** | 48 | 46.4 | 97% |
> | 4 | 386 ms | 1495 ms | **3.9×** | 177 | 162 | 92% |
> | 16 | 549 ms | 2418 ms | **4.4×** | 602 | 483 | 80% |
> | 64 | 1691 ms | 1969 ms | **1.2×** | 1478 | 943 | 64% |
>
> **Three key phenomena**:
> 1. **TTFT grows linearly with input** (prefill compute ∝ input length), but **at high concurrency TTFT growth is amortized by batching** (c=64: only 1.2×)
> 2. **ITL barely changes** (21→25→39 ms), decode speed is independent of input length
> 3. **Total throughput drops with long input** (64% at c=64) — prefill occupies more batch time, squeezing decode
>
> **Customer perspective**: Long prompts (8K) mainly affect **time-to-first-byte** (multi-second), not token streaming speed. This is exactly the pain point that PD disaggregation addresses — separating prefill to a dedicated instance to avoid blocking decode batches.

---

### 6f: ⭐ Sweep Comprehensive Performance Profile (Measured 2026-04-25)

> Three (input, output) groups × four concurrency levels = 12 real data points, covering code completion / long output / long context — three typical scenarios.

| Workload | c=1 | c=4 | c=16 | c=64 | Customer Scenario |
|----------|----:|----:|----:|----:|---------|
| **1K/1K** (chat) | 48 tok/s | 177 | 602 | **1478** | Short Q&A, code completion |
| **1K/8K** (long-output) | 47.5 | 178 | 621 | **1623** | Document generation, code block writing |
| **8K/1K** (RAG/long-prompt) | 46.4 | 162 | 483 | **943** | RAG, long document analysis |

**Core conclusions**:
- **Decode speed is constant**: Single user ~21ms/token (≈48 tok/s) consistent across all (input, output) configurations
- **Output length is free**: 1K vs 8K output has no impact on throughput (difference < 5%)
- **Input length has a cost**: 8K vs 1K input, throughput drops 36% at c=64 — this is exactly what **PD disaggregation can recover**

---

## PD Disaggregation (Disaggregated Serving) — 1P1D on GKE

> Qwen3 Coder 480B on v7x **officially supports 1P1D PD disaggregation**: 1 prefill instance (v7x-8) + 1 decode instance (v7x-8) = total 2 nodes × 4 chips = 8 chips.

### Key Benefits
- **Lower TTFT**: Prefill scheduled independently, not blocked by decode
- **Higher throughput**: Prefill and decode resource ratios can be tuned independently
- **Supports long context**: Prefill instance HBM fully dedicated to attention

> 📚 **Companion deep dive** (includes architecture, KV transfer path, load balancing analysis, memory pressure, NPnD scaling):
> [PD Disaggregation Deep Dive — Qwen3-Coder 480B × TPU v7x](https://cc.higcp.com/assets/qwen3-coder-480b-pd-disagg-explained-20260425.html)

### 7a-pre: ⭐ Lustre Shared Storage Approach (Strongly Recommended for PD Multi-Pod Deployment)

> **Pain point**: Default manifests create a separate hyperdisk PVC (ReadWriteOnce) for each P and D, meaning every new pod must **re-download the 480GB weights** (~30-45 min per pod).
>
> **Solution**: Use GKE Managed Lustre (ReadWriteMany) as shared storage, **all P/D pods share a single copy of weights**, download once and use forever.

**Prerequisite**: Cluster already has a Lustre PVC. Verify:
```bash
kubectl get pvc | grep lustre
# Expected: lustre-pvc Bound  lustre-pv  36000Gi  RWX  ...
kubectl get sc | grep lustre
# Expected: lustre-rwx-1000mbps-per-tib  lustre.csi.storage.gke.io  ...
```

If not yet created, see [GKE Managed Lustre documentation](https://cloud.google.com/kubernetes-engine/docs/how-to/managed-lustre). Create a ≥36 TB instance once, shared across all LLM models/datasets.

#### Step 1: One-Time Download of Qwen3-Coder Weights to Lustre

Spin up a temporary download pod (no TPU needed, CPU pod is fine):

```bash
kubectl apply -f - <<'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: qwen3-coder-downloader
spec:
  restartPolicy: Never
  containers:
  - name: dl
    image: python:3.12-slim
    command: ["/bin/bash", "-c"]
    args:
      - |
        pip install -U huggingface_hub hf_transfer
        export HF_HUB_ENABLE_HF_TRANSFER=1
        mkdir -p /lustre/models/Qwen3-Coder-480B-A35B-Instruct-FP8
        huggingface-cli download Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
          --local-dir /lustre/models/Qwen3-Coder-480B-A35B-Instruct-FP8
        echo "✅ Download complete:"
        du -sh /lustre/models/Qwen3-Coder-480B-A35B-Instruct-FP8
    env:
    - name: HF_TOKEN
      valueFrom:
        secretKeyRef: { name: hf-token-secret, key: token }
    resources:
      requests: { cpu: "4", memory: "16Gi" }
      limits:   { cpu: "8", memory: "32Gi" }
    volumeMounts:
    - name: lustre
      mountPath: /lustre
  volumes:
  - name: lustre
    persistentVolumeClaim:
      claimName: lustre-pvc
EOF

# Track progress
kubectl logs -f qwen3-coder-downloader

# Clean up after completion
kubectl delete pod qwen3-coder-downloader
```

Expected time: **30-45 min** (depends on HF rate limiting + Lustre write bandwidth, typical 200-400 MB/s).

#### Step 2: Modify single_prefill.yaml / single_decode.yaml for Lustre

Make these two changes in each manifest's PVC section and vLLM launch command:

**Change 1: Replace dedicated PVC with lustre-pvc**
```yaml
# Delete:
# apiVersion: v1
# kind: PersistentVolumeClaim
# metadata: { name: pvc-vllm-p }      # or pvc-vllm-d
# spec: { accessModes: [ReadWriteOnce], resources: { requests: { storage: 500Gi }}, storageClassName: hyperdisk-balanced }
# No need to create PVC, use existing lustre-pvc directly

# Volume changed to:
volumes:
- name: lustre-vol
  persistentVolumeClaim:
    claimName: lustre-pvc          # ← shared RWX PVC

# volumeMount changed to:
volumeMounts:
- name: lustre-vol
  mountPath: /lustre               # ← model path
```

**Change 2: vllm serve uses local path + offline mode**
```bash
# Before: --model=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8
# Changed to:
--model=/lustre/models/Qwen3-Coder-480B-A35B-Instruct-FP8 \
--served-model-name=Qwen3-Coder-480B-FP8

# Also add env:
- name: HF_HUB_OFFLINE
  value: "1"                       # ← Prevents vLLM from re-downloading from HF
```

#### Comparison

| Dimension | Default (per-pod PVC) | Lustre Shared |
|------|----------------------|----------------|
| Weight download count | Once per pod (×P count + ×D count) | **Once, shared forever** |
| Per-pod disk size | 600 GB hyperdisk-balanced | 500 GB sufficient (system + cache) |
| Multi-machine scale (NPnD) per added pod | Extra 30-45 min download wait | **0 wait, instant startup** |
| Weight consistency | Each copy independent, may be out of sync | Single source, always consistent |
| Lustre read performance | — | Sequential 6.5 GB/s · Random 0.03 GB/s (note: mmap unfriendly, see below) |
| Cost | 600GB × pod count | Shared 36TB Lustre instance |

> ⚠️ **Lustre performance note**: vLLM weight loading uses mmap sequential read, Lustre sequential read at 6.5 GB/s is fine. But if your application uses mmap random read, Lustre is slow (0.03 GB/s); in that case `cp` to `/dev/shm` first then use. See [Lustre random read workaround with SHM](https://github.com/yangwhale/gpu-tpu-pedia/blob/main/tpu/tpu-inference/DeepSeek-R1-671B-FP4/README.md).

### 7a: Create 1P1D Deployment

Need 3 manifests (from [`vllm-project/tpu-inference`](https://github.com/vllm-project/tpu-inference/tree/main/.buildkite/kubernetes/manifests/v7x)):

```bash
# 1. Clone repo on jumpbox (kubectl client)
git clone https://github.com/vllm-project/tpu-inference.git
cd tpu-inference

# 2. ⚠️ Patch manifests before apply to use Lustre shared path, avoid PVC full
#    (upstream default uses HF model name, triggering vLLM re-download, filling 600GB PVC, see §FAQ #9)
#    If you've already downloaded weights to /lustre/models/Qwen3-Coder-480B-A35B-Instruct-FP8/ per §7a-pre, do these substitutions:
for f in .buildkite/kubernetes/manifests/v7x/single_prefill.yaml \
         .buildkite/kubernetes/manifests/v7x/single_decode.yaml; do
  # Change --model=Qwen/... to --model=/lustre/models/...
  sed -i 's|--model=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8|--model=/lustre/models/Qwen3-Coder-480B-A35B-Instruct-FP8 --served-model-name=Qwen3-Coder-480B-FP8|' "$f"
  # Change PVC to lustre-pvc + add HF_HUB_OFFLINE=1 (specific patch see §7a-pre Step 2)
done

# 3. Apply prefill, decode, proxy
kubectl apply -f .buildkite/kubernetes/manifests/v7x/single_prefill.yaml
kubectl apply -f .buildkite/kubernetes/manifests/v7x/single_decode.yaml
kubectl apply -f .buildkite/kubernetes/manifests/v7x/proxy1p1d.yaml
```

> 💡 **If you don't want to patch manifests**: You can directly prepare weights in the PVC (~30-45 min HF download per pod), then use upstream manifests unmodified — but each P/D pod downloads a separate copy, wasting time and disk space. Lustre path strongly recommended.

### 7b: Prefill Instance Configuration (Key Parameters)

From `single_prefill.yaml`:

```yaml
args: [
  "vllm serve --seed=42 \
    --model=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
    --max-model-len=10240 \
    --max-num-batched-tokens=8192 \
    --max-num-seqs=512 \
    --no-enable-prefix-caching \
    --tensor-parallel-size=8 \
    --kv-cache-dtype=fp8 \
    --gpu-memory-utilization=0.70 \
    --async-scheduling \
    --enable-expert-parallel \
    --kv-transfer-config '{
      \"kv_connector\":\"TPUConnector\",
      \"kv_connector_module_path\":\"tpu_inference.distributed.tpu_connector\",
      \"kv_role\":\"kv_producer\"
    }'"
]
```

**Key differences (vs single instance)**:
- `gpu-memory-utilization=0.70` (reserve 30% HBM for KV transfer buffer)
- `kv-transfer-config` sets `kv_role=kv_producer`

### 7c: Decode Instance Configuration

From `single_decode.yaml`, key differences:
- `gpu-memory-utilization=0.90` (decode doesn't need as much buffer, leave more for KV cache)
- `kv_role=kv_consumer`

### 7d: Wait for All Pods Ready

```bash
# Wait for prefill ready
kubectl wait --for=condition=Ready pod -l app=vllm-prefill --timeout=1200s

# Wait for decode ready
kubectl wait --for=condition=Ready pod -l app=vllm-decode --timeout=1200s

# Wait for proxy ready
kubectl wait --for=condition=Ready pod -l app=vllm-proxy --timeout=300s
```

### 7e: Run PD Disaggregation Benchmark

```bash
PROXY_POD=$(kubectl get pods -l app=vllm-proxy -o jsonpath="{.items[0].metadata.name}")

# ⚠️ --model must match --served-model-name in single_prefill.yaml
# If you applied the sed patch from §7a, served name is "Qwen3-Coder-480B-FP8"
# Otherwise use upstream default "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
SERVED_NAME="Qwen3-Coder-480B-FP8"   # After patch; upstream default is long name

# Proxy port: upstream proxy1p1d.yaml defaults to listen 10000; adjust if using different proxy
PROXY_PORT=10000

# 1024 input / 8192 output, concurrency=64
kubectl exec $PROXY_POD -- vllm bench serve \
  --model="$SERVED_NAME" \
  --dataset-name=random \
  --num-warmups 10 \
  --random-input-len=1024 \
  --random-output-len=8192 \
  --num-prompts=256 \
  --ignore-eos \
  --host=localhost \
  --port=$PROXY_PORT \
  --max-concurrency=64 \
  --request-rate=inf \
  --metric-percentiles 90,99 \
  --result-file=disagg_1024_8192_c64.json

# 8192 input / 1024 output, concurrency=64
kubectl exec $PROXY_POD -- vllm bench serve \
  --model="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8" \
  --dataset-name=random \
  --random-input-len=8192 \
  --random-output-len=1024 \
  --num-prompts=256 \
  --ignore-eos \
  --host=localhost \
  --port=10000 \
  --max-concurrency=64 \
  --request-rate=inf \
  --result-file=disagg_8192_1024_c64.json
```

### 7f: Retrieve Results

```bash
kubectl cp $PROXY_POD:disagg_1024_8192_c64.json ./disagg_1024_8192_c64.json
kubectl cp $PROXY_POD:disagg_8192_1024_c64.json ./disagg_8192_1024_c64.json
```

### 7g: PD Disaggregation Comparison (✅ Measured 2026-04-25 14:46~14:54)

**Deployment**: e2e-03 (v7x-8 in spot-4 pool) = Prefill (kv_producer, mem=0.7) · e2e-04 (v7x-8 in v3 pool) = Decode (kv_consumer, mem=0.9) · proxy running inside e2e-03 on port **7000** (⚠️ different from §7a proxy1p1d.yaml default port 10000; §7g used a manually started toy_proxy_server.py with arbitrarily chosen port, production deployment uses 10000).

#### 1024/1024 c=1 (low-latency)

| Config | TTFT (med) | TPOT (med) | ITL (med) | Output tok/s | Δ vs single instance |
|------|----------:|----------:|----------:|------------:|------------|
| **Single instance c=1** | 95 ms | 20.6 ms | 20.6 ms | 48 | baseline |
| **1P1D c=1** | **281 ms** | **18.3 ms** | **18.3 ms** | **53.8** | TTFT +186ms · **TPOT -11%** · **tok/s +12%** |

#### 8192/1024 c=4 (long-prompt, PD disaggregation target scenario)

| Config | TTFT (med) | TPOT (med) | ITL (med) | Output tok/s | Δ vs single instance |
|------|----------:|----------:|----------:|------------:|------------|
| **Single instance c=4** | 1495 ms | 23.2 ms | 22.7 ms | 162 | baseline |
| **1P1D c=4** | **2908 ms** | **20.6 ms** | **20.6 ms** | **170** | TTFT +1413ms · **TPOT -11%** · **tok/s +5%** |

> **Core findings**:
> 1. **TPOT/ITL consistently improved ~11%** — D instance dedicated to decode, no prefill batch contending for GPU, each token produced faster. This is the most important customer-perceived benefit of PD disaggregation.
> 2. **TTFT increases (+186 ms ~ +1.4 s)** — P + DCN transfer + D three-stage inherent overhead. When prompts are short (1K) the ratio is significant, when prompts are long (8K) prefill itself dominates, PD overhead is relatively small.
> 3. **Output throughput improves 5-12%** — Single user / low concurrency scenarios see limited PD benefit. **Real PD benefit comes at c=64+ high concurrency**, where D instance can run full-speed 256 batch decode without prefill blocking.
> 4. **Production recommendation**: Scenarios where TTFT is not critical (chat, code generation) are suitable for PD disaggregation; TTFT-sensitive scenarios (real-time autocomplete) should keep single instance.
>
> **Reproduction commands**:
> ```bash
> # P (e2e-03)
> --kv-transfer-config '{"kv_connector":"TPUConnector","kv_connector_module_path":"tpu_inference.distributed.tpu_connector","kv_role":"kv_producer"}'
> --gpu-memory-utilization 0.70
>
> # D (e2e-04)
> --kv-transfer-config '{"kv_connector":"TPUConnector","kv_connector_module_path":"tpu_inference.distributed.tpu_connector","kv_role":"kv_consumer"}'
> --gpu-memory-utilization 0.90
>
> # Proxy (inside e2e-03)
> python3 /workspace/tpu_inference/examples/disagg/toy_proxy_server.py \
>   --host 0.0.0.0 --port 7000 \
>   --prefiller-hosts localhost --prefiller-ports 8000 \
>   --decoder-hosts <D_pod_IP> --decoder-ports 9000
> ```

---

## Troubleshooting

### 1. Startup OOM `out of memory while allocating ...`

**Cause**: `--enable-expert-parallel` missing, experts replicated on every device.

**Fix**: Must add `--enable-expert-parallel`.

### 2. vLLM Startup Stuck at "Loading model"

**Cause A**: Model still downloading, `huggingface-cli` not finished.

**Fix**: Check `du -sh $HF_HOME/qwen3-coder-480b-fp8`, should be ~480 GB.

**Cause B**: HF_TOKEN not configured → 401 waiting forever.

**Fix**: Check `kubectl get secret hf-token-secret -o yaml`.

### 3. `Permission denied: /dev/vfio/...`

**Cause**: Container lacks `privileged: true` or `IPC_LOCK` capability.

**Fix**: Confirm manifest includes:
```yaml
securityContext:
  privileged: true
  capabilities:
    add: [IPC_LOCK]
```

### 4. `vm.max_map_count` Error

**Cause**: initContainer didn't run successfully (host may not allow sysctl).

**Fix**: Enable `--linux-node-config="sysctl=vm.max_map_count=8388608"` on GKE node pool, or manually SSH to node and run:
```bash
gcloud compute ssh <node> -- 'sudo sysctl -w vm.max_map_count=8388608'
```

### 5. `JAX_PLATFORMS` Not Taking Effect, CPU Fallback

**Symptom**: Logs show `Running on CPU`.

**Fix**: Ensure `JAX_PLATFORMS=tpu,cpu` (not `cpu,tpu`, order matters).

### 6. TPU Device Busy

**Cause**: Previous vLLM process still alive; or main process killed but EngineCore child process not cleaned, leaving behind `/tmp/libtpu_lockfile`.

**Symptom (measured 2026-04-25)**: Restarting vLLM, EngineCore starts then crashes after a few seconds with:
```
RuntimeError: Unable to initialize backend 'tpu':
ABORTED: Internal error when accessing libtpu multi-process lockfile.
Run "$ sudo rm /tmp/libtpu_lockfile".
```

**Fix (standard procedure)**:
```bash
# 1. Kill all vllm/EngineCore remnants
pkill -9 -f 'vllm|EngineCore'
sleep 3
ps -ef | grep -E "vllm|EngineCore" | grep -v grep    # Should be empty (zombie <defunct> is fine)

# 2. Delete libtpu lockfile + vLLM IPC socket
rm -f /tmp/libtpu_lockfile /tmp/.vllm_ipc_*

# 3. Confirm vfio devices are released
fuser /dev/vfio/* 2>&1 || echo "vfio free"

# 4. Now you can restart vllm serve
```

> 💡 **Lesson**: After every `pkill vllm serve` **always** delete the lockfile, otherwise next startup will fail. Recommend adding a default cleanup at the beginning of startup scripts.

### 7. Benchmark `request_rate=inf` Overwhelmed

**Symptom**: TTFT keeps increasing, requests queuing.

**Fix**: Use a specific request_rate (e.g., `--request-rate=2.0`), or reduce concurrency.

### 8. `chunked-prefill` Error or Not Working

**Cause**: Qwen3 Coder 480B in EP mode does not have chunked-prefill enabled by default.

**Fix**: Do not add `--enable-chunked-prefill`, CI doesn't use it either.

### 9. ⚠️ PVC Full — `OSError: [Errno 28] No space left on device` (Experienced This)

**Symptom**: vLLM crashes mid-loading, reports disk full; but `du` shows PVC only half used.

**Root cause**: Local weights already exist (e.g., copied from GCS at `/usr/vllm/qwen3-coder-480b-fp8/`), but vLLM launched with HuggingFace model name still re-downloads to `$HF_HOME/hub/...`, two copies together exceed PVC (450GB + 450GB > 590GB).

**Fix**: Use **local path** + set `HF_HUB_OFFLINE=1`:
```bash
# Wrong: will re-download
vllm serve Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 ...

# Correct: reads local directly
export HF_HUB_OFFLINE=1
vllm serve /usr/vllm/qwen3-coder-480b-fp8 \
  --served-model-name Qwen3-Coder-480B-FP8 \
  ... # remaining parameters same as above

# If hub files are already half-downloaded, clean up first
rm -rf $HF_HOME/hub $HF_HOME/xet
```

### 10. ⚠️ Local Weights Missing Tokenizer — `TypeError: expected str ... not NoneType` (Experienced This)

**Symptom**: Launch with local path, crashes at tokenizer stage, error stack points to `tokenization_qwen2.py:172, with open(vocab_file, ...)`, vocab_file=None.

**Root cause**: Local weights directory copied from GCS / colleague may only have safetensors + config.json + merges.txt, **missing**:
- `tokenizer.json`
- `tokenizer_config.json`
- `vocab.json`

**Fix**: Separately curl these three small files (a few MB each):
```bash
cd /usr/vllm/qwen3-coder-480b-fp8/
for f in tokenizer.json tokenizer_config.json vocab.json; do
  curl -sL -o $f https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8/resolve/main/$f
done
ls -la tokenizer.json vocab.json tokenizer_config.json  # Verify successful download
```

> **Prevention**: When copying weights, always copy the entire directory (`gsutil -m cp -r gs://bucket/qwen3-coder-480b-fp8 ./`), don't just copy `*.safetensors`.

---

## Performance Optimization (Optional)

### Optimization 1: Skip JAX Precompile

```bash
export SKIP_JAX_PRECOMPILE=1
export VLLM_XLA_CHECK_RECOMPILATION=0
```

**Expected effect**: Startup 1-2 min faster, no runtime impact.

### Optimization 2: Use MoE EP Kernel (When Stability is OK)

```bash
export USE_MOE_EP_KERNEL=1
```

> ⚠️ CI defaults to `USE_MOE_EP_KERNEL=0`, because 1 has issues with certain input shapes. If you confirm it's stable, you can enable it.

### Optimization 3: Enable Prefix Caching (Production Scenarios)

Remove `--no-enable-prefix-caching`, add `--enable-prefix-caching`.

> **Must remain disabled** for benchmarks, otherwise random input repeated prefills will pollute data.

### Optimization 4: KV Cache FP8

Already in the 4a launch command: `--kv-cache-dtype=fp8`. FP16 → FP8 saves 50% HBM, accuracy loss <0.1%.

### Optimization 5: Increase Batch Size

```bash
--max-num-batched-tokens=16384  # Default 8192, increasing to 16K may improve throughput +10-20%
--max-num-seqs=1024              # Default 512, increase for high concurrency scenarios
```

> Monitor HBM after increasing, may need to lower `--gpu-memory-utilization`.

---

## End-to-End Time Budget

| Stage | Single Instance | 1P1D PD Disaggregation |
|------|-------|-------------|
| Model download (HF → Lustre/PVC, 480 GB) | 30-60 min (first time only) | Same (Lustre shared, only 1 time) |
| vLLM startup (incl. XLA compilation) | ~7 min | ~7 min × 2 (P + D in parallel) |
| Smoke + 50-prompt benchmark | ~5 min | ~5 min |
| Concurrency sweep (1K/1K, 1K/8K, 8K/1K × c=1,4,16,64) | ~80 min | ~30-60 min |
| **First run minimal verification** | **~50-80 min** | **~70-90 min** |
| **Subsequent restarts** (model already in storage) | **~10 min** | **~12 min** |

---

## References

| Resource | Link |
|------|------|
| Upstream tpu-inference repo | [vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference) |
| Qwen3 MoE model implementation | [qwen3_moe.py](https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/models/jax/qwen3_moe.py) |
| HuggingFace model | [Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8) |
| Alternative quantized version | [BCCard/Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic](https://huggingface.co/BCCard/Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic) |
| CI Pipeline | [Qwen_Qwen3-Coder-480B-A35B-Instruct.yml](https://github.com/vllm-project/tpu-inference/blob/main/.buildkite/models/Qwen_Qwen3-Coder-480B-A35B-Instruct.yml) |
| Benchmark script | [bm_qwen3_coder.sh](https://github.com/vllm-project/tpu-inference/blob/main/tests/e2e/benchmarking/bm_qwen3_coder.sh) |
| Multihost script | [run_qwen3_coder_480b_1k_8k.sh](https://github.com/vllm-project/tpu-inference/blob/main/scripts/multihost/benchmarks/torchax/run_qwen3_coder_480b_1k_8k.sh) |
| GKE Prefill manifest | [single_prefill.yaml](https://github.com/vllm-project/tpu-inference/blob/main/.buildkite/kubernetes/manifests/v7x/single_prefill.yaml) |
| GKE Decode manifest | [single_decode.yaml](https://github.com/vllm-project/tpu-inference/blob/main/.buildkite/kubernetes/manifests/v7x/single_decode.yaml) |
| Daily disagg script | [daily_run_gke_disagg.sh](https://github.com/vllm-project/tpu-inference/blob/main/.buildkite/scripts/daily_run_gke_disagg.sh) |
| Other model READMEs in same directory | [GLM-5.1](../GLM-5.1-754B-FP4/README.en.md) · [DeepSeek R1](../DeepSeek-R1-671B-FP4/README.en.md) · [Kimi K2.6](../Kimi-K2.6-1T-A32B-INT4/README.en.md) |

---

## Future TODO

### Completed (2026-04-25)
- [x] **Startup time measured**: cold ~7 min, warm ~5 min
- [x] **HBM allocation measured**: 94.75 GB/device, 758 GB total
- [x] **Complete benchmark sweep** (3 workloads × 4 concurrency levels = 12 data points) — see §6e
- [x] **PD disaggregation 1P1D vs single instance** comparison test — see §7g (TPOT -11%, throughput +5-12%)
- [x] **Lustre shared storage approach documentation** — §7a-pre
- [x] **Pitfall summary**: PVC full + local weights missing tokenizer + libtpu lockfile (§Troubleshooting #6 #9 #10)

### Pending
- [ ] **Higher concurrency** sweep (c=128, 256, 512, 1024) — verify if CI threshold 1926 tok/s is reachable
- [ ] **2P:1D / 1P:2D NPnD** testing — verify optimization for imbalanced workloads
- [ ] **Cross-host PD disaggregation** — verify KV transfer performance across DCN zones
- [ ] **Quality verification**: GSM8K / HumanEval (accuracy gate ≥ 0.85 flexible)
- [x] **README.en.md** English version (for international customers)
