# MiMo-V2.5-Pro (BF16) Inference on 2× TPU v7x-8

> 🌐 **Languages** | **语言**: **English** | [中文](./README.md)

> End-to-end guide: running MiMo-V2.5-Pro BF16 inference on 2× TPU v7x-8 (16 devices, multi-host).
>
> **Architecture**: ~1T total params / 42B activated / **Hybrid SWA** (60 SWA + 10 Full Attention) + 384 routed experts + BF16
>
> **Inference framework**: [sglang-jax](https://github.com/sgl-project/sglang) (**not vLLM**) — multi-host TP=8, EP=2
>
> **Model**: MiMo-V2.5-Pro (~1T total params, BF16)

---

## 🚨 Known Critical Limitations

> The current deployment is in the **POC validation stage**, and the 5 SWA cache patches are mandatory.

- **Multi-host only**: ~1T total params in BF16 requires 2× v7x-8 (2 hosts × 8 devices); a single machine OOMs
- **5 SWA cache patches mandatory**: sglang-jax's SWA radix cache has multiple accounting bugs; without the patches the KV cache gets corrupted → subsequent requests output garbage
- **Low concurrency**: the SWA pool (7908 tokens) + context_length=4096 only holds ~1.9 complete sequences
- **Cold start ~50 min**: 25 min weight loading + 5 min FP8 QKV dequant + 15 min XLA compilation + 7 min DECODE PRECOMPILE
- **Benchmark not finished**: smoke test passes; throughput / latency / accuracy evaluation still to be done

---

## ⚠️ Mandatory Reading: Constraints

### A. The 5 SWA Cache Patches (all required)

sglang-jax's Hybrid SWA (Sliding Window Attention) radix cache has multiple accounting bugs that corrupt the KV cache. **All 5 patches are indispensable**:

| # | Patch file | Target file | What it fixes |
|---|-----------|---------|---------|
| 1 | `patch_swa_cache_leak.py` | `swa_radix_cache.py` | `inc_lock_ref` / `dec_lock_ref` see different `_swa_eff_len()` values at lock vs. unlock time (caused by node split); snapshotting eff_len fixes `swa_protected_size_` drift |
| 2 | `patch_swa_cache_leak.py` | `swa_radix_cache.py` | `_delete_tombstone_leaf` uses `len(node.key)` instead of `len(node.value)`, causing `full_evictable_size_` to be miscounted |
| 3 | `patch_sanity_check_tolerant.py` | `swa_radix_cache.py` | `sanity_check()` turns the `assert` on evictable-size mismatch into a `warning` (tolerating residual drift not fully fixed by the patches) |
| 4 | `patch_check_memory_tolerant.py` | `scheduler.py` | `check_memory()` turns the leak-detection `ValueError` into a `warning` (same as above) |
| 5 | `patch_disable_evict_swa_v2.py` | `schedule_batch.py` | **Core fix**: `maybe_evict_swa()` fully disabled — `_evict_swa()` frees SWA slots in both the decode and extend paths without notifying the tree cache, so on a prefix cache hit it reads stale KV data |

**Root cause analysis (Patch #5)**:
- `_evict_swa()` calls `free_swa()` to release SWA slots outside the sliding_window
- but it **never notifies the tree cache** (`notify_swa_mapping_freed()` is a no-op)
- the freed slots get allocated to new requests
- old requests' tree nodes still reference the freed SWA positions → **attention reads garbage data**
- both the decode and extend paths trigger it (the v1 patch only disabled decode, which is insufficient)

### B. Key Launch Parameters

| Parameter | Value | Purpose |
|------|---|------|
| `--tp-size` | `8` | Tensor Parallel = 8 devices/host |
| `--ep-size` | `2` | Expert Parallel = 2 (across 2 hosts) |
| `--nnodes` | `2` | Multi-host, 2 nodes |
| `--context-length` | `4096` | Limited by SWA pool capacity |
| `--max-total-tokens` | `8192` | Total token budget |
| `--mem-fraction-static` | `0.95` | Fraction of HBM used by KV cache |
| `--enable-cache-report` | — | Outputs cache usage information |

---

## 🧭 Deployment Steps

### Prerequisites

- GKE cluster + 2× TPU v7x-8 nodes (multi-host)
- Model weights path mounted via Lustre or GCS at `/lustre/models/MiMo-V2.5-Pro`
- sglang-jax container image (containing `/opt/sglang-jax/`)
- StatefulSet + Headless Service (the 2 pods discover each other)

### Step 1: Prepare Pods + Verify Model

```bash
CTX=<your-gke-context>
POD_0=mimo-benchmark-0
POD_1=mimo-benchmark-1
MODEL=/lustre/models/MiMo-V2.5-Pro

# Verify pods running
kubectl --context=$CTX get pods | grep mimo-benchmark
# Expected: mimo-benchmark-0  1/1  Running
#            mimo-benchmark-1  1/1  Running

# Verify model weights
kubectl --context=$CTX exec $POD_0 -- bash -c "ls $MODEL/*.safetensors | wc -l"
```

### Step 2: Apply the 5 Patches (on both pods)

cp the 4 patch scripts to both pods and run them:

```bash
PATCHES=(
  patch_swa_cache_leak.py
  patch_sanity_check_tolerant.py
  patch_check_memory_tolerant.py
  patch_disable_evict_swa_v2.py
)

for POD in $POD_0 $POD_1; do
  for P in "${PATCHES[@]}"; do
    kubectl --context=$CTX cp /tmp/$P $POD:/tmp/$P
    kubectl --context=$CTX exec $POD -- python3 /tmp/$P
  done
  echo "=== $POD patched ==="
done
```

**Verify the patches took effect** (on any pod):

```bash
kubectl --context=$CTX exec $POD_0 -- bash -c "
  echo 'Patch 1 (eff_len_at_lock):' \$(grep -c '_swa_eff_len_at_lock' /opt/sglang-jax/python/sgl_jax/srt/mem_cache/swa_radix_cache.py)
  echo 'Patch 2 (tombstone fix):' \$(grep -c 'len(node.value)' /opt/sglang-jax/python/sgl_jax/srt/mem_cache/swa_radix_cache.py)
  echo 'Patch 3 (sanity tolerant):' \$(grep -c 'evictable mismatch (tolerant)' /opt/sglang-jax/python/sgl_jax/srt/mem_cache/swa_radix_cache.py)
  echo 'Patch 4 (check_memory tolerant):' \$(grep -c '_leak_count' /opt/sglang-jax/python/sgl_jax/srt/managers/scheduler.py)
  echo 'Patch 5 (evict_swa disabled):' \$(grep -c 'DISABLED: _evict_swa frees SWA slots' /opt/sglang-jax/python/sgl_jax/srt/managers/schedule_batch.py)
"
# Expected: all outputs ≥1
```

### Step 3: Launch sglang-jax (File-based launcher, both pods)

⚠️ **You must use the file-based launcher**: the nohup from `kubectl exec` gets SIGKILLed.

```bash
cat > /tmp/launch_sglang_multihost.sh <<'LAUNCHER'
#!/bin/bash
set -e

NNODES=2
NPROC_PER_NODE=8
EP_SIZE=2
MODEL_PATH=/lustre/models/MiMo-V2.5-Pro
DIST_ADDR="mimo-benchmark-0.mimo-bench-headless-svc:5000"

# Extract node rank from hostname (mimo-benchmark-0 → 0, mimo-benchmark-1 → 1)
NODE_RANK=$(hostname | grep -oP '\d+$')

echo "Starting sglang-jax: node_rank=$NODE_RANK, nnodes=$NNODES, ep=$EP_SIZE"

pgrep -f 'sglang\|srt' | xargs -r kill -9 2>/dev/null || true
sleep 2
rm -f /tmp/libtpu_lockfile

setsid nohup python3 -m sgl_jax.launch_server \
  --model-path "$MODEL_PATH" \
  --tp-size $NPROC_PER_NODE --ep-size $EP_SIZE \
  --nnodes $NNODES --node-rank "$NODE_RANK" \
  --dist-init-addr "$DIST_ADDR" \
  --context-length 4096 --max-total-tokens 8192 \
  --mem-fraction-static 0.95 \
  --port 30271 --host 0.0.0.0 \
  --enable-cache-report --trust-remote-code \
  >> /tmp/sglang_mimo.log 2>&1 < /dev/null &
disown
echo "launched pid=$!"
exit 0
LAUNCHER

# cp + launch on both pods
for POD in $POD_0 $POD_1; do
  kubectl --context=$CTX cp /tmp/launch_sglang_multihost.sh $POD:/tmp/launch_sglang_multihost.sh
  kubectl --context=$CTX exec $POD -- bash /tmp/launch_sglang_multihost.sh
done
```

### Step 4: Wait for Cold Start (~50 min) + Health Check

```bash
# Poll health (60s interval, up to 60 min)
for i in $(seq 1 60); do
  C=$(kubectl --context=$CTX exec $POD_0 -- curl -sf -o /dev/null -w "%{http_code}" http://localhost:30271/health 2>/dev/null)
  C=${C:-000}
  echo "T+$((i*60))s HTTP $C"
  [ "$C" = "200" ] && break
  sleep 60
done
```

**Cold start phase timing reference**:
| Phase | Duration | Log marker |
|------|------|---------|
| Weight loading | ~25 min | `Loading model weights...` |
| FP8 QKV dequant | ~5 min | `dequantize`-related log |
| XLA compilation | ~15 min | `XLA compilation` |
| DECODE PRECOMPILE | ~7 min | `DECODE PRECOMPILE` |
| **Total** | **~50 min** | `Application startup complete` / health 200 |

### Step 5: Smoke Test

```bash
# Simple completion test
kubectl --context=$CTX exec $POD_0 -- curl -s http://localhost:30271/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"MiMo-V2.5-Pro","messages":[{"role":"user","content":"What is 2+3? Answer with just the number."}],"max_tokens":50,"temperature":0}' \
  | python3 -c 'import sys,json; r=json.load(sys.stdin); print("content:", repr(r["choices"][0]["message"]["content"]), "| finish:", r["choices"][0]["finish_reason"])'
# Expected: content: '5' | finish: stop

# Multiple-request validation (key: verify subsequent requests are not garbage)
for i in 1 2 3; do
  echo "--- Request $i ---"
  kubectl --context=$CTX exec $POD_0 -- curl -s http://localhost:30271/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"MiMo-V2.5-Pro","messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"max_tokens":20,"temperature":0}' \
    | python3 -c 'import sys,json; r=json.load(sys.stdin); print(repr(r["choices"][0]["message"]["content"]))'
done
# Expected: all 3 output 'Paris' or a coherent answer containing Paris
```

**Validation criteria**:
- ✅ 3/3 requests output coherent text (not garbage)
- ✅ 0 LEAK warnings (`grep LEAK /tmp/sglang_mimo.log`)
- ✅ 0 sanity check warnings (`grep "sanity check" /tmp/sglang_mimo.log`)
- ✅ SWA token count always positive (`grep "swa token" /tmp/sglang_mimo.log | head -5`)

---

## Troubleshooting

| Symptom | Root cause | Fix |
|------|------|------|
| **Subsequent requests output garbage** | `maybe_evict_swa` not disabled; SWA slots get reclaimed but the tree cache doesn't know | Confirm patch #5 is applied: `grep -c 'DISABLED: _evict_swa' schedule_batch.py` should output 1 |
| **Negative SWA token count** (`#swa token: -224`) | `_evict_swa` in the decode or extend path freed slots it shouldn't have | Same as above; confirm full disabling (v2 patch, not v1) |
| **AssertionError: evictable_size != lru_list_evictable_size** | SWA cache accounting drift | patch #3 (sanity_check tolerant) downgrades it to a warning |
| **ValueError: token_to_kv_pool_allocator memory leak** | SWA protected_size drift | patch #1 (eff_len snapshot) fixes it + patch #4 (check_memory tolerant) as a fallback |
| **Cold start >60 min** | `/dev/shm` leftovers occupying RAM | Clean up `/dev/shm`: `rm -rf /dev/shm/sem.* /dev/shm/wrk_*` |
| **The two pods cannot connect** | Headless service DNS not ready | Confirm the `mimo-bench-headless-svc` service exists and `nslookup mimo-benchmark-0.mimo-bench-headless-svc` resolves correctly |
| **Process disappears after kubectl exec launch** | Not a file-based launcher; got SIGKILLed | You must use the file-based launcher (Step 3) |

---

## Model Core Parameters

| Field | Value |
|------|---|
| Architecture | MoE + **Hybrid SWA** (60 SWA + 10 Full Attention) |
| Parameters | ~1T total / 42B activated / 384 routed experts |
| Dimensions | Hidden 6144, 8 KV heads, 70 layers |
| Quantization | BF16 (no quantization) |
| Sliding Window | 128 tokens (SWA layers) |
| Max position | 1,048,576 |

## Hardware Requirements

| Item | Requirement |
|---|---|
| TPU | **2× v7x-8** (16 devices, multi-host required) |
| HBM | 96 GB/device × 16 = 1,536 GB |
| Inference framework | sglang-jax (**not vLLM**) |
| Parallelism strategy | TP=8, EP=2 |

## To Do

- [ ] P1-3: R3 FusedMoE + R5 Prefetch combined test
- [ ] P1-4: BSZ Sweep (64/128/256/512)
- [ ] P1-5: `exp/skip-padding-tokens` branch test
- [ ] GSM8K / MMLU accuracy evaluation
- [ ] Throughput / latency benchmark data

## References

- READMEs in the same series: [DeepSeek R1 FP4](../DeepSeek-R1-671B-FP4/) · [Qwen3.5 FP8](../Qwen3.5-397B-A17B-FP8/) · [GLM-5.1 FP4](../GLM-5.1-754B-FP4/)
