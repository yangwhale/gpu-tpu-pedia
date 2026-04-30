# Qwen3.5-397B-A17B-FP8 Inference on TPU v7x-8

> 🌐 **Languages** | **语言**: [中文](README.md) · **English**

> End-to-end guide: Running Qwen3.5-397B-A17B-FP8 inference on TPU v7x-8 (8 chips, single-host).
>
> **Architecture**: 397B total params / 17B active / **hybrid GDN+Attention** (45 GDN + 15 Standard Attn) + 512 routed experts + FP8 native
>
> **Code Repository**: [vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference) (main branch ≥ 2026-04-23, including PR #2366)
>
> **Model**: [Qwen/Qwen3.5-397B-A17B-FP8](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8) (94 safetensors, ~378 GiB)

---

## 🚨 Known Critical Limitations

> Current deployment is **not suitable for conversational chatbot**.
> - **Chat path broken**: thinking OFF causes garbled output / infinite loops; thinking ON causes empty content / `Thinking\n` infinite loop for explanatory / casual questions
> - **Only stable path**: 5-shot Q/A completion pattern + `enable_thinking:false` (GSM8K 93.93% was achieved with this)
> - **Suitable use cases**: batch eval, structured generation, few-shot completion, code gen
> - ⚠️ High GSM8K accuracy ≠ chat ready — **don't be misled**

See [Required Constraint D](#d-thinking-behavior) and [Verification Steps](#step-4-verification-5-shot-hello-world) for details.

---

## 🎯 Key Performance (Measured)

| Operating Point | Measured | Notes |
|---|---|---|
| Cold start | **~7 min** | weight load + MoE re-quant + KV cache init |
| Single user latency (P1, 1K/1K) | **20.6 s, 49.7 tok/s/user** | 💨 Low Latency |
| Balanced (P64, 1K/1K) | **1508 tok/s, 23.5 tok/s/user** | ⚖️ Balanced |
| **🚀 Peak throughput (P128, 1K/1K)** | **2097 tok/s** ⭐ | Peak (not P256, see appendix) |
| **GSM8K full 1319 (5-shot, thinking OFF)** | **93.93% (1239/1319)** ✅ | length truncation only 1.06% |
| Long prompt 8K/1K P4 | **178.6 tok/s** | hybrid GDN long context advantage |

Full benchmark data in [Appendix: Throughput sweep + GSM8K](#appendix-full-benchmark-data).

---

## ⚠️ Required Constraints (4 Items)

### A. PR #2366 Patch (**Required**)

vLLM hybrid allocator shares 1 `KVCacheTensor` across 4 layers (GPU byte-level). But TPU `jax.Array` is strongly typed and must duplicate per-layer → vLLM scheduler's block_id pool is ~3.5× larger than actual TPU capacity → block_id out-of-bounds → JAX `dynamic_update_slice_in_dim` silently clips → multi-request state collapse → **gibberish output / OOM / EngineCore crash**.

**Fix**: Copy the patched `kv_cache_manager.py` from main branch to the pod. Verification:
```bash
kubectl exec $POD -- grep -c '_hybrid_uniform_page_size_bytes' \
  /workspace/tpu_inference/tpu_inference/runner/kv_cache_manager.py
# Output 7 = patched; Output 0 = needs patch (see Step 2)
```

### B. Three Required Environment Variables

| Variable | Value | Purpose |
|---|---|---|
| `MODEL_IMPL_TYPE` | `vllm` | Qwen3.5 uses vLLM PyTorch + TorchAX path |
| `SKIP_JAX_PRECOMPILE` | `1` | Skip JAX pre-compilation, saves 1-2 min at startup |
| `VLLM_XLA_CHECK_RECOMPILATION` | `0` | Disable XLA recompilation check |

### C. Key Startup Parameters

| Parameter | Value | Purpose |
|---|---|---|
| `--enable-expert-parallel` | Required | EP=8 is the correct parallelism strategy after PR #2366 fix |
| `--no-enable-prefix-caching` | Required | Otherwise triggers `chunked_mm_input` AssertionError |
| `--reasoning-parser` | `qwen3` | Correctly parse `<think>` tags |
| `--block-size` | `256` | CI default |
| `--kv-cache-dtype` | `fp8` | Halves KV cache size |
| `--gpu-memory-utilization` | `0.9` (single) / `0.7` (PD prefill) / `0.9` (PD decode) | |
| `--tensor-parallel-size` | `8` | TP=8 (8 chips) |
| `--max-model-len` | `4096` (single) / `16384` (PD long prompt) | |
| `--max-num-batched-tokens` | `4096` | CI accuracy test default |
| `--max-num-seqs` | `256` | CI default |
| `--limit-mm-per-prompt` | `'{"image":0,"video":0}'` | Skip vision encoder |
| `--async-scheduling` | Recommended | Async scheduling |

### D. Thinking Behavior

Qwen3.5 defaults to thinking ON (outputs `<think>...</think>` reasoning + answer). **All server-side methods to turn off thinking are unstable**:

| Attempt | Result |
|---|---|
| `--chat-template-kwargs='{"enable_thinking":false}'` (startup flag) | Silently ignored |
| Request body `chat_template_kwargs={"enable_thinking":false}` + regular chat prompt | Model enters `</think>` infinite loop → gibberish |
| **Request body same as above + single user message with 5-shot Q/A pattern** | ✅ **Works**, reasoning_len=0 |
| User prompt with `/no_think` tag | Ineffective |

**Production workarounds** (by reliability):
1. ⭐ Chat + 5-shot Q/A pattern + `enable_thinking:false` (**only stable method**, GSM8K measured 93.93%)
2. `/v1/completions` raw prompt — unstable, not recommended
3. Accept thinking ON + `max_tokens` ≥ 3500 — for long answer scenarios

---

## 🧭 Deployment Mode Selection

| Mode | TPU | Suitable Scenarios | Startup Time | Patches | Complexity | Dogfood Status |
|---|---|---|---|---|---|---|
| **1. Single-machine** | 1 × v7x-8 (8 chips, TP=8) | low-latency / batch eval / GSM8K / **most production use cases** ⭐ | ~7 min | 1 (PR #2366) | Low | 🟢 Fully verified (perf + GSM8K) |
| **2. PD Disaggregation (1P1D)** | 2 × v7x-8 (TP=8 P + TP=8 D) | long prompt high throughput / **TPOT optimization** / high concurrency | ~10 min | 2 (PR #2366 + HMA) | Medium | 🟡 Deploy+smoke ✅, perf pending |
| **3. Multi-host TP=16** | 2 × v7x-8 (16 chips, TP=16, LWS) | **very large models (>500B) / verify multi-host capability** | ~11-30 min | 3 (PR #2366 + 2 mrope bypass) | High | 🟡 Deploy+smoke ✅, perf pending |

**Selection guide**:
- ✅ Not sure? Start with **Mode 1** — covers 90% of use cases, fastest cold start, fewest patches
- 🚀 Need higher P95 throughput / long prompt optimization? Upgrade to **Mode 2 PD**
- 🔬 Verifying multi-host inference engine capability / want to run >500B models? Use **Mode 3** (397B fits on single machine, this is mainly a capability proof)

---

## ⚡ Quick Start (5 Commands for Experienced Users)

> Already deployed / familiar with GKE TPU + vLLM? Reuse the 5 commands below, skip the 200+ lines that follow. New users should follow [Deployment Mode 1 Full Steps](#deployment-mode-1-single-machine-vllm-serve).

```bash
CTX=<your-gke-context>; POD=<your-tpu-pod>; MODEL=/lustre/models/Qwen3.5-397B-A17B-FP8

# 1. Verify model + patch (PR #2366)
kubectl --context=$CTX exec $POD -- bash -c "ls $MODEL/*.safetensors | wc -l && grep -c '_hybrid_uniform_page_size_bytes' /workspace/tpu_inference/tpu_inference/runner/kv_cache_manager.py"
# Expected: 94  <newline>  7    (94 shards + 7 PR #2366 markers; 7 = patched)

# 2. Write file-based launcher to host (kubectl exec multi-line nohup gets SIGKILL)
cat > /tmp/launch_vllm.sh <<'L'
#!/bin/bash
pgrep -f 'EngineCore|vllm' | xargs -r kill -9; sleep 2
rm -f /tmp/libtpu_lockfile /tmp/vllm_qwen35.log; touch /tmp/vllm_qwen35.log
setsid nohup env SKIP_JAX_PRECOMPILE=1 VLLM_XLA_CHECK_RECOMPILATION=0 MODEL_IMPL_TYPE=vllm \
  vllm serve /lustre/models/Qwen3.5-397B-A17B-FP8 \
    --tensor-parallel-size 8 --enable-expert-parallel --max-model-len 4096 \
    --max-num-batched-tokens 4096 --max-num-seqs 256 --no-enable-prefix-caching \
    --gpu-memory-utilization 0.9 --kv-cache-dtype fp8 --block-size 256 --trust-remote-code \
    --limit-mm-per-prompt '{"image":0,"video":0}' --reasoning-parser qwen3 --async-scheduling \
    >> /tmp/vllm_qwen35.log 2>&1 < /dev/null & disown
exit 0
L

# 3. cp + run launcher
kubectl --context=$CTX cp /tmp/launch_vllm.sh $POD:/tmp/launch_vllm.sh && kubectl --context=$CTX exec $POD -- bash /tmp/launch_vllm.sh

# 4. Wait ~7 min cold start (poll health)
for i in $(seq 1 30); do C=$(kubectl --context=$CTX exec $POD -- curl -sf -o /dev/null -w "%{http_code}" http://localhost:8000/health); echo "T+$((i*30))s HTTP $C"; [ "$C" = "200" ] && break; sleep 30; done

# 5. Smoke test (5-shot, thinking OFF)
SHOTS="Question: Capital of Japan?\nAnswer: Tokyo.\n\nQuestion: Capital of Germany?\nAnswer: Berlin.\n\nQuestion: Capital of Italy?\nAnswer: Rome.\n\nQuestion: Capital of Spain?\nAnswer: Madrid.\n\nQuestion: Capital of Brazil?\nAnswer: Brasilia.\n\n"
kubectl --context=$CTX exec $POD -- curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"${SHOTS}Question: Capital of France?\nAnswer:\"}],\"max_tokens\":50,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
  | python3 -c 'import sys,json;r=json.load(sys.stdin);m=r["choices"][0]["message"];print(repr(m["content"]),"|",r["choices"][0]["finish_reason"])'
# Expected: ' Paris.' | stop
```

> 💡 **PD / Multi-host quick start** see [Deployment Mode 2](#deployment-mode-2-pd-disaggregation-1p1d) / [Deployment Mode 3](#deployment-mode-3-multi-host-tp16-2--v7x-8--measured-2026-04-26--lm-inference-55-pass) Steps 0-7. These modes require more patches + Ray + LWS compared to single-machine, not suitable for one-liner quick reference.

---

# Deployment Mode 1: Single-machine vLLM Serve

> Single v7x-8 pod running Qwen3.5-397B vLLM serve. Suitable for low-latency / small-to-medium concurrency / batch eval / GSM8K-style tasks.

### Step 1: Prepare Pod + Model

Requires GKE TPU pod (`tpu-v7x-lite-podslice`, 2x2x1, 8 chips) + shared storage (**Lustre RWX recommended**) + model weights.

> 💡 **Why Lustre instead of PVC**: Lustre download ≈ 63 GB/min vs PVC ≈ 16 GB/min (measured with Qwen3 ≈ 4× gap). 378 GiB weights take ~6 min on Lustre vs ~24 min on PVC. Multi-pod concurrent training/inference also requires Lustre RWX.

```bash
# Set GKE context (your cluster)
CTX=<your-gke-context>
POD=<your-tpu-pod-name>
MODEL=/lustre/models/Qwen3.5-397B-A17B-FP8

# Verify pod ready
kubectl --context="$CTX" get pods | grep $POD       # Should be Running 2/2

# Download weights to Lustre (if not done, ~6 min, 16 workers + hf_transfer acceleration)
kubectl exec $POD -- bash -c "
  mkdir -p /lustre/models/Qwen3.5-397B-A17B-FP8
  pip install -U 'huggingface_hub[hf_transfer]'
  HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
    Qwen/Qwen3.5-397B-A17B-FP8 \
    --local-dir /lustre/models/Qwen3.5-397B-A17B-FP8
"

# Verify all 94 shards are complete
kubectl exec $POD -- bash -c "ls $MODEL/*.safetensors | wc -l"   # Should output 94

# Clean /dev/shm residuals (avoids 50× weight load degradation)
kubectl exec $POD -- bash -c "
  ls -la /dev/shm/                # Check usage
  # Before deleting others' staging, fuser -v /dev/shm/<dir> to confirm nobody is using it
  rm -rf /dev/shm/sem.* /dev/shm/wrk_* 2>/dev/null
"
```

### Step 2: Apply PR #2366 Patch (If Not Already Patched)

```bash
# Check
kubectl exec $POD -- grep -c '_hybrid_uniform_page_size_bytes' \
  /workspace/tpu_inference/tpu_inference/runner/kv_cache_manager.py
# Output 7 = patched, skip; Output 0 = apply patch below
```

```bash
# Host side
TMP=$(mktemp /tmp/kv_cache_manager.XXXXXX.py)
curl -sf https://raw.githubusercontent.com/vllm-project/tpu-inference/main/tpu_inference/runner/kv_cache_manager.py -o $TMP
grep -c '_hybrid_uniform_page_size_bytes' $TMP   # Should output 7

KCM=/workspace/tpu_inference/tpu_inference/runner/kv_cache_manager.py
kubectl --context="$CTX" exec $POD -- cp $KCM ${KCM}.bak
kubectl --context="$CTX" cp $TMP $POD:$KCM
kubectl --context="$CTX" exec $POD -- bash -c "
  grep -c '_hybrid_uniform_page_size_bytes' $KCM   # verify 7
  find /workspace/tpu_inference -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
"
rm -f $TMP
```

### Step 3: Start vLLM (File-based Launcher)

⚠️ **Must use file-based launcher**: `kubectl exec $POD -- bash -c "<multi-line nohup>"` will be SIGKILL=137 (kubectl exec kills the process group when the stdin channel closes, nohup/setsid/disown cannot save it). File-based launcher lets bash cleanly fork+exit after reading the file.

```bash
# 1. Write launcher locally
cat > /tmp/launch_vllm.sh <<'LAUNCHER'
#!/bin/bash
cd /tmp
pgrep -f 'EngineCore|vllm' | xargs -r kill -9 2>/dev/null
sleep 2
rm -f /tmp/libtpu_lockfile /tmp/vllm_qwen35.log
touch /tmp/vllm_qwen35.log
setsid nohup env \
  SKIP_JAX_PRECOMPILE=1 VLLM_XLA_CHECK_RECOMPILATION=0 MODEL_IMPL_TYPE=vllm \
  vllm serve /lustre/models/Qwen3.5-397B-A17B-FP8 \
    --tensor-parallel-size 8 --enable-expert-parallel \
    --max-num-batched-tokens 4096 --max-num-seqs 256 --max-model-len 4096 \
    --no-enable-prefix-caching --gpu-memory-utilization 0.9 \
    --kv-cache-dtype fp8 --block-size 256 --trust-remote-code \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    --reasoning-parser qwen3 --async-scheduling \
    >> /tmp/vllm_qwen35.log 2>&1 < /dev/null &
disown
echo "launched pid=$!"
exit 0
LAUNCHER

# 2. cp to pod and execute
kubectl --context="$CTX" cp /tmp/launch_vllm.sh $POD:/tmp/launch_vllm.sh
kubectl --context="$CTX" exec $POD -- bash /tmp/launch_vllm.sh

# 3. Wait for cold start (~7 min) + monitor
kubectl --context="$CTX" exec $POD -- tail -f /tmp/vllm_qwen35.log
```

**Wait for key log messages (PR #2366 + startup complete indicators)**:
```
Hybrid KV cache: padding every layer spec to 23289856 bytes ...   ← PR #2366 padding
regular_attn_shape=(num_blocks, (1280, 8, 4, 256))                ← block_size 1280 (wrong 4352 before patch)
num_gpu_blocks_override=945
INFO: Application startup complete.
```

### Step 4: Verification (5-shot Hello World)

```bash
# Health check
kubectl exec $POD -- curl -sf -o /dev/null -w "%{http_code}\n" http://localhost:8000/health
# Should output 200

# Hello world — chat endpoint 5-shot single user message (only stable chat method, see Constraint D)
SHOTS="Question: Capital of Japan?\nAnswer: Tokyo.\n\nQuestion: Capital of Germany?\nAnswer: Berlin.\n\nQuestion: Capital of Italy?\nAnswer: Rome.\n\nQuestion: Capital of Spain?\nAnswer: Madrid.\n\nQuestion: Capital of Brazil?\nAnswer: Brasilia.\n\n"
P="${SHOTS}Question: Capital of France?\nAnswer:"
kubectl exec $POD -- curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$P\"}],\"max_tokens\":50,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
  | python3 -c 'import sys,json; r=json.load(sys.stdin); m=r["choices"][0]["message"]; print("content:", repr(m["content"])); print("reasoning_len:", len(m.get("reasoning") or "")); print("finish:", r["choices"][0]["finish_reason"])'
# Expected: content: ' Paris.'  reasoning_len: 0  finish: stop
```

**Expected**: `content: 'Paris.'` / `reasoning_len: 0` / `finish: stop`

---

# Deployment Mode 2: PD Disaggregation (1P1D)

> 1 prefill (kv_producer) + 1 decode (kv_consumer) + proxy three-pod deployment. Suitable for high-concurrency batch / TPOT-sensitive scenarios / long prompt RAG.

### 🚨 Qwen3.5 PD Required Differences (vs Qwen3-Coder PD)

| Item | Qwen3-Coder (pure attention) | **Qwen3.5 (hybrid GDN+Attn)** |
|---|---|---|
| `kv_connector` | `TPUConnector` | **`TPUConnectorHMA`** ⭐ |
| `kv_connector_module_path` | `tpu_inference.distributed.tpu_connector` | **`tpu_inference.distributed.tpu_connector_hma`** ⭐ |
| Hybrid KV cache manager flag | Not needed | **Required** `--no-disable-hybrid-kv-cache-manager` ⭐ |
| Image contains HMA | nightly has TPUConnector | **No HMA**, need to cp from main |

**Why `--no-disable-hybrid-kv-cache-manager` is required**: vLLM core **disables hybrid KV cache manager by default** when it sees `kv_transfer_config`, forcing `unify_hybrid_kv_cache_specs`. But Qwen3.5's 60 layers (45 GDN + 15 Attn) cannot be unified → `ValueError: Hybrid KV cache manager is disabled but failed to convert the KV cache specs to one unified type` → EngineCore crash. This flag explicitly tells vLLM "my connector is SupportsHMA, handle hybrid properly".

### Step 1: Create 2 v7x-8 Spot Node Pools

```bash
PROJECT=<your-gcp-project>
CLUSTER=<your-gke-cluster>
REGION=<your-region>      # e.g., us-central1
ZONE=<your-zone>          # e.g., us-central1-c

for role in p d; do
  gcloud container node-pools create np-tpu7x-spot-pd-$role \
    --cluster=$CLUSTER --region=$REGION --project=$PROJECT \
    --node-locations=$ZONE --machine-type=tpu7x-standard-4t --num-nodes=1 --spot \
    --disk-type=hyperdisk-balanced --disk-size=500 \
    --node-taints=google.com/tpu=present:NoSchedule \
    --workload-metadata=GKE_METADATA --enable-autorepair --enable-autoupgrade --async
done
# Regions with ample spot capacity usually ready in 2-5 min
```

### Step 2: Stage HMA + PR #2366 Patch to Lustre

```bash
# Copy scripts/tpu_connector_hma.py (already cp'd from main in this repo) to Lustre
kubectl cp scripts/tpu_connector_hma.py $POD:/tmp/tpu_connector_hma.py
kubectl exec $POD -- bash -c "
  mkdir -p /lustre/patches/qwen35-pd
  cp /tmp/tpu_connector_hma.py /lustre/patches/qwen35-pd/
  cp /workspace/tpu_inference/tpu_inference/runner/kv_cache_manager.py /lustre/patches/qwen35-pd/
  # verify
  echo 'HMA refs:' \$(grep -c TPUConnectorHMA /lustre/patches/qwen35-pd/tpu_connector_hma.py) '(expect ≥18)'
  echo 'PR #2366 refs:' \$(grep -c _hybrid_uniform_page_size_bytes /lustre/patches/qwen35-pd/kv_cache_manager.py) '(expect 7)'
"
```

### Step 3: Deploy P + D + Proxy (3 Manifests Committed to `manifests/`)

```bash
cd manifests/
kubectl --context="$CTX" apply -f qwen35_prefill.yaml -f qwen35_decode.yaml -f qwen35_proxy.yaml
```

The 3 manifests include init containers: each pod startup automatically copies HMA + PR #2366 files from Lustre `/lustre/patches/qwen35-pd/` to `/workspace/tpu_inference/`, deletes `__pycache__`, and verifies via grep.

### Step 4: Wait for Ready + Verify HMA Logs

```bash
# ~10 min cold start
kubectl --context="$CTX" wait --for=condition=Ready pod -l app=vllm-prefill --timeout=1200s
kubectl --context="$CTX" wait --for=condition=Ready pod -l app=vllm-decode --timeout=1200s

# Verify HMA + PR #2366 + Hybrid KV manager all enabled
PREFILL_POD=$(kubectl --context="$CTX" get pods -l app=vllm-prefill -o jsonpath='{.items[0].metadata.name}')
kubectl --context="$CTX" logs $PREFILL_POD | grep -E "TPUConnectorHMA|Hybrid KV cache: padding|Application startup"
```

**Expected output**:
```
Hybrid KV cache: padding every layer spec to 23289856 bytes               ← PR #2366 ✓
Hybrid KV cache layout: num_kv_cache_groups=4, ... duplicate_shared_layers=True
TPUConnectorHMA Worker 0 Prefill --> init | num_layers=60 | num_kv_groups=4 | group_is_mamba=[True, True, True, False]
Creating v1 connector with name: TPUConnectorHMA                          ← HMA registered ✓
Application startup complete
```

### Step 5: Smoke Test (Verify P→D KV Transfer Works)

```bash
PROXY_POD=$(kubectl --context="$CTX" get pods -l app=vllm-proxy -o jsonpath='{.items[0].metadata.name}')

# 5-shot examples (single user message, same pattern as single-machine/multi-host Step 4/5)
SHOTS="Question: Capital of Japan?\nAnswer: Tokyo.\n\nQuestion: Capital of Germany?\nAnswer: Berlin.\n\nQuestion: Capital of Italy?\nAnswer: Rome.\n\nQuestion: Capital of Spain?\nAnswer: Madrid.\n\nQuestion: Capital of Brazil?\nAnswer: Brasilia.\n\n"

for country in France Italy Australia Canada Brazil; do
  P="${SHOTS}Question: Capital of $country?\nAnswer:"
  result=$(kubectl --context="$CTX" exec $PROXY_POD -- curl -s http://localhost:10000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"Qwen3.5-397B-FP8\",\"messages\":[{\"role\":\"user\",\"content\":\"$P\"}],\"max_tokens\":50,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
    | python3 -c 'import sys,json; r=json.load(sys.stdin); m=r["choices"][0]["message"]; c=(m["content"] or "").replace(chr(10), " ")[:60]; print(repr(c) + " | " + r["choices"][0]["finish_reason"])')
  echo "$country: $result"
done
```

**Expected**: 5/5 countries all hit, all finish=stop, reasoning_len=0
```
France:    'Answer: Paris.' | stop
Italy:     ' Rome.'         | stop
Australia: ' Canberra.'     | stop
Canada:    ' Ottawa.'       | stop
Brazil:    ' Brasilia.'     | stop
```

---

# Deployment Mode 3: Multi-host TP=16 (2 × v7x-8) ⭐ Measured 2026-04-26 ⭐ LM Inference 5/5 PASS

> 1 LWS across 2 v7x-8 nodes with 16 chips total, TP=16 + Ray distributed executor. **Proves vLLM TPU inference engine supports hybrid model multi-host end-to-end LM inference**.
>
> ✅ **Current status**: Startup ✅ + HTTP 200 ✅ + **5/5 country capital chat completions all hit** (Paris/Rome/Canberra/Ottawa/Brasilia, finish=stop, reasoning_len=0).

### 🚨 Multi-host vs Single-host Key Differences (6-layer Root Cause / Fix)

| # | Multi-host Specific Fix | Consequence If Not Applied |
|---|---|---|
| 1 | **`--max-num-batched-tokens=16384`** (≥ Qwen3.5 `max_tokens_per_mm_item`) | Silent hang in init_device, 14 min no new logs, worker SIGSEGV |
| 2 | **PR #2366 patch (kv_cache_manager.py)** + **multi-group block_tables_cpu rebuild** | KV init `AssertionError: page_size_padded >= page_size` or `IndexError: list index out of range` |
| 3 | **`TPU_MULTIHOST_BACKEND=ray`** + `TPU_TOPOLOGY=2x2x2` + `TPU_HOST_BOUNDS=1,1,2` env override | Starts in single-host mode, TP=16 cannot span hosts |
| 4 | **`--distributed-executor-backend=ray`** + LWS pattern (1 leader + 1 worker) | UniProcExecutor cannot span pods |
| 5 | **`tpu_runner.py` patch (5 lines)**: set `self.uses_mrope=False` + `self.get_mrope_input_positions_fn=None` when `disable_mm_from_limits=True` | First chat request crashes `TypeError: Qwen3VL.get_mrope_input_positions() got unexpected hf_config` (vllm core vs Qwen3VL model class API mismatch) |
| 6 | **`persistent_batch_manager.py` patch (4 lines)**: defensive None check `if get_mrope_input_positions_fn is None: continue` | PersistentBatchManager captured uses_mrope=True at init, later setting False doesn't affect → calls None → TypeError |

> ⭐ **3 patches committed to repo** ([scripts/multihost-patches/](scripts/multihost-patches/)), anyone can deploy after checkout:
> - `kv_cache_manager.py` — PR #2366 hybrid padding + multi-group block_tables_cpu rebuild
> - `tpu_runner.py` — disable mrope when disable_mm_from_limits (5 lines)
> - `persistent_batch_manager.py` — defensive None check (4 lines)

### Step 0: Set Environment Variables (One-time Setup, Used in Steps 1-7)

```bash
# GKE
CTX=<your-gke-context>             # e.g., gke_PROJECT_REGION_CLUSTER (kubectl context)
PROJECT=<your-gcp-project>
CLUSTER=<your-gke-cluster>
REGION=<your-region>               # e.g., us-central1
ZONE=<your-zone>                   # e.g., us-central1-c

# Lustre patch staging
UTIL_POD=<your-pod-with-lustre-mount>   # Any pod with lustre-pvc mounted (used in Step 2)

# Repo (this repo root)
REPO_ROOT=<your-checkout-of-gpu-tpu-pedia>   # e.g., ~/gpu-tpu-pedia
cd $REPO_ROOT/tpu/tpu-inference/Qwen3.5-397B-A17B-FP8/
```

### Step 1: Prepare Multi-host Node Pool

```bash
# Create 2-node multi-host TPU pool (note --num-nodes=2 + --tpu-topology=2x2x2)
gcloud container node-pools create np-tpu7x-spot-mh-qwen35 \
  --cluster=$CLUSTER --region=$REGION --project=$PROJECT \
  --node-locations=$ZONE --machine-type=tpu7x-standard-4t --num-nodes=2 \
  --tpu-topology=2x2x2 --spot \
  --disk-type=hyperdisk-balanced --disk-size=500 \
  --node-taints=google.com/tpu=present:NoSchedule \
  --workload-metadata=GKE_METADATA --enable-autorepair --enable-autoupgrade
# Regions with ample spot capacity usually ready in 2-5 min
```

> **Spot capacity insufficient fallback** (when reporting `RESOURCE_EXHAUSTED` or quota error):
> - Switch zone (one of us-central1-a/b/c), or switch region (us-east5 / us-west4 / asia-northeast1)
> - Change `--spot` → `--reservation-affinity=specific --reservation=<your-reservation>` (on-demand, more expensive but stable)

### Step 2: Stage **3 Patches** to Lustre (Required for Multi-host)

```bash
# One-time cp of 3 patches to Lustre (multi-host needs 2 more patches than PD)
# ⚠️ Assumes you've cd'd to the model dir from Step 0 (relative path scripts/multihost-patches/ works)
for f in kv_cache_manager.py tpu_runner.py persistent_batch_manager.py; do
  kubectl --context="$CTX" cp scripts/multihost-patches/$f $UTIL_POD:/tmp/$f
done

kubectl --context="$CTX" exec $UTIL_POD -- bash -c "
  mkdir -p /lustre/patches/qwen35-pd
  cp /tmp/kv_cache_manager.py /tmp/tpu_runner.py /tmp/persistent_batch_manager.py /lustre/patches/qwen35-pd/
  echo 'Verify patches:'
  echo '  PR #2366 refs:' \$(grep -c '_hybrid_uniform_page_size_bytes' /lustre/patches/qwen35-pd/kv_cache_manager.py) '(expect 7)'
  echo '  block_tables_cpu rebuild patch:' \$(grep -c 'PATCH: rebuild block_tables_cpu' /lustre/patches/qwen35-pd/kv_cache_manager.py) '(expect 1)'
  echo '  mrope tpu_runner patch:' \$(grep -c 'PATCH: disable mrope' /lustre/patches/qwen35-pd/tpu_runner.py) '(expect 1)'
  echo '  mrope PBM patch:' \$(grep -c 'PATCH: skip mrope fn call' /lustre/patches/qwen35-pd/persistent_batch_manager.py) '(expect 1)'
"
```

> **The 3 patches are automatically copied to `/workspace/tpu_inference/` by the yaml's init container** on every pod startup, no need to manually cp into the pod each time.

### Step 3: Deploy LWS

```bash
cd manifests/
# If previously deployed, delete first (LWS doesn't support in-place yaml updates)
kubectl --context="$CTX" delete lws qwen35-mh --ignore-not-found --wait=false
sleep 30   # Wait for pod termination

kubectl --context="$CTX" apply -f qwen35_multihost.yaml
# LWS 1 group × 2 pods (1 leader on node A + 1 worker on node B)
```

> 💡 **Ignorable warning**: If Service `qwen35-mh` was previously deployed, you'll see `Warning: resource services/qwen35-mh is missing the kubectl.kubernetes.io/last-applied-configuration annotation`. This is a standard kubectl warning, does not affect deployment.
>
> 💡 **Redeploy after updating patches**: After modifying patches, you must (1) re-run [Step 2 cp to Lustre](#step-2-stage-3-patches-to-lustre-required-for-multi-host), (2) re-apply LWS (init container copies new patches from Lustre on pod startup). LWS cannot hot-reload patches in-place.

### Step 4: Wait for Ready + Verify Multi-host Startup Logs (**Measured 11-30 min**)

⚠️ Multi-host cold start is **2-3× slower than single-machine**: weight load shares Lustre bandwidth across machines + cross-host JAX init synchronously waits for all hosts to complete.
- **Smooth case**: ~11 min, RESTARTS=0 (measured multiple times)
- **Sentinel race triggered**: ~25-30 min, includes 1-2 K8s auto restarts (race is a known intermittent, doesn't affect final success)

**Don't use `kubectl wait condition=Ready`** — it gives false success during pod restart. Use health endpoint polling instead:

```bash
# Poll health until 200 (max 30 min)
for i in $(seq 1 60); do
  CODE=$(kubectl --context="$CTX" exec qwen35-mh-0 -- curl -sf -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null)
  CODE=${CODE:-000}   # Default 000 (avoids empty output when curl fails)
  echo "T+$((i*30))s: HTTP $CODE"
  [ "$CODE" = "200" ] && break
  sleep 30
done

# Verify multi-host key init logs (awk truncates each line to 150 chars to avoid device list flooding)
kubectl --context="$CTX" logs qwen35-mh-0 | grep -E "Init worker|Hybrid KV cache:|Application startup" | awk '{print substr($0,1,150)}' | head -10
```

**Expected key lines (multi-host characteristics)**:
- `Init worker | rank=0 | hbm=[(0.0, 94.75), ...] × 16` ← **16 chips** (single-machine is 8)
- `Hybrid KV cache: padding every layer spec to 13328384 bytes` ← multi-host PR #2366 padding (single-machine is 23289856, different because TP=16 shards are finer)
- `Hybrid KV cache layout: num_kv_cache_groups=4, ... num_blocks=5299` ← TP=16 num_blocks is 5.6× larger than single-machine 945
- `regular_attn_sharding=Mesh('data':1, 'model':16)` ← **TP=16 mesh** ✅
- `Application startup complete` ← ✅ ready signal

### Step 5: Smoke Test (Verify Multi-host LM Inference Works)

```bash
# 5-shot examples (single user message, hits thinking-OFF path; see Required Constraint D)
SHOTS="Question: Capital of Japan?\nAnswer: Tokyo.\n\nQuestion: Capital of Germany?\nAnswer: Berlin.\n\nQuestion: Capital of Italy?\nAnswer: Rome.\n\nQuestion: Capital of Spain?\nAnswer: Madrid.\n\nQuestion: Capital of Brazil?\nAnswer: Brasilia.\n\n"

for country in France Italy Australia Canada Brazil; do
  P="${SHOTS}Question: Capital of $country?\nAnswer:"
  result=$(kubectl --context="$CTX" exec qwen35-mh-0 -- curl -s http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"Qwen3.5-397B-FP8\",\"messages\":[{\"role\":\"user\",\"content\":\"$P\"}],\"max_tokens\":50,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
    | python3 -c 'import sys,json; r=json.load(sys.stdin); m=r["choices"][0]["message"]; print(repr(m["content"]) + " | " + r["choices"][0]["finish_reason"])')
  echo "$country: $result"
done
```

**Expected** (measured 2026-04-26 all pass, dogfood retested 2 times):
```
France:    'Paris.'    | stop
Italy:     'Rome.'     | stop
Australia: 'Canberra.' | stop
Canada:    'Ottawa.'   | stop
Brazil:    'Brasilia.' | stop
```

### Step 6: Troubleshooting (Multi-host Specific)

**Symptom 1: Step 4 health still not 200 after 30 min**
- Check restart count: `kubectl --context="$CTX" get pods -l leaderworkerset.sigs.k8s.io/name=qwen35-mh`
  - RESTARTS≥2 → `kubectl logs qwen35-mh-0 --previous | grep -E 'ERROR|Traceback' | tail -30` to find root cause (most likely one of the [6-layer root causes](#-multi-host-vs-single-host-key-differences-6-layer-root-cause--fix), verify patches are staged + verify counters ≥ expected values)
- Worker actor SIGSEGV: `kubectl logs qwen35-mh-0-1 --previous | tail -50`
- Spot preemption: `kubectl describe pod qwen35-mh-0 | grep Warning`

**Symptom 2: Step 5 some countries fail / garbled output**
- Output stuck in `</think>` loop → 5-shot pattern not matched, check if prompt is a single user message (not multiple messages)
- `KeyError: 'choices'` → vllm internal crash, container restarted, wait for health 200 and retry
- `TypeError: ... mrope ...` → mrope patch not applied, verify: `kubectl logs qwen35-mh-0 | grep 'mrope patch'` should output `(expect 1)`

**Symptom 3: LWS pod stuck Pending, not scheduled**
- TPU nodes all occupied. `kubectl get nodes -l cloud.google.com/gke-tpu-accelerator=tpu7x` to check usage; if others are using them, wait or create new node pool

**Symptom 4: `kubectl exec` reports `setns process: exit status 1`**
- Container is restarting, retry in a few seconds. Common during the window after vllm crash

### Step 7: Production Access / Teardown

**After smoke test passes, connect production traffic**:

```bash
# Service is already defined in the yaml (clusterIP, port 8000)
kubectl --context="$CTX" get svc qwen35-mh   # Check ClusterIP

# In-cluster access: any pod can call inference via service DNS
# http://qwen35-mh.default.svc.cluster.local:8000/v1/chat/completions

# External access (production scenario, via GKE Ingress or Gateway):
kubectl --context="$CTX" expose deployment vllm-disagg-proxy ... # omitted, standard GKE workflow
# Or port-forward for debugging: kubectl port-forward svc/qwen35-mh 8000:8000
```

**Teardown (stop to save cost)**:

```bash
# 1. Delete LWS (releases pods on TPU nodes, but spot node pool still runs)
kubectl --context="$CTX" delete lws qwen35-mh

# 2. (Optional) Delete multi-host node pool (truly zero cost; spot nodes incur charges even when idle)
gcloud container node-pools delete np-tpu7x-spot-mh-qwen35 \
  --cluster=$CLUSTER --region=$REGION --project=$PROJECT --quiet
```

> 💡 **Lustre patches don't need cleanup**: `/lustre/patches/qwen35-pd/*` takes < 200 KB, can be reused across redeployments.

### Multi-host Deployment Full Dogfood Record

Complete 8 test iterations + 6-layer root cause chain + 3 patches deep dive + lessons learned in **internal dogfood HTML** ([Appendix at bottom](#-internal-documents-dogfood-history--deep-analysis)).

---

# Benchmark

### Throughput (Single Instance, evalscope)

```bash
# Inside the pod
pip install -q evalscope[perf]
bash scripts/run_bench_qwen35.sh    # Default P1/P4/P16/P64/P256 5 tiers × 2 rounds (~24 min)
# Output /tmp/bench_qwen35/summary.txt
```

Full 17-tier sweep data in [Appendix](#appendix-full-benchmark-data).

### GSM8K Accuracy (5-shot, Thinking OFF via In-context, ~15 min)

```bash
# cp script to pod
kubectl cp scripts/run_gsm8k_qwen35.py $POD:/tmp/

# Run
kubectl exec $POD -- bash -lc '
python3 /tmp/run_gsm8k_qwen35.py \
  --model /lustre/models/Qwen3.5-397B-A17B-FP8 \
  --url http://localhost:8000/v1/chat/completions \
  --limit 1319 --parallel 8 --max-question-tokens 500 \
  --output /tmp/gsm8k_full.jsonl
'
```

**Measured**: 1319 questions ~15 min, **93.93% accuracy** (1239/1319), length truncation only 14 (1.06%). CI threshold 63%.

> Monitoring tip: Script stdout is buffered; check real-time progress with `wc -l /tmp/gsm8k_full.jsonl` (each completed question is written immediately).

### PD Disaggregation Benchmark 🟡 Measurement Pending

> **Status**: Deploy ✅ + 5/5 smoke test ✅ + 6-layer root cause solidified, **perf data pending** (reference Qwen3-Coder §7g: TPOT -11%, output tok/s +5-12% is the baseline for same framework + hybrid architecture).

```bash
PROXY_POD=$(kubectl get pods -l app=vllm-proxy -o jsonpath='{.items[0].metadata.name}')
kubectl exec $PROXY_POD -- vllm bench serve \
  --model=Qwen3.5-397B-FP8 --dataset-name=random \
  --num-warmups 10 --random-input-len=1024 --random-output-len=1024 \
  --num-prompts=256 --ignore-eos \
  --host=localhost --port=10000 --max-concurrency=1 \
  --result-file=disagg_qwen35_1024_1024_c1.json
# Also run 8192/1024 c=4 long prompt scenario (PD sweet spot)
```

### Multi-host TP=16 Benchmark 🟡 Measurement Pending

> **Status**: Deploy ✅ + 5/5 smoke test ✅ + 3 patches solidified, **perf vs single-machine TP=8 comparison pending** (hypothesis: TP=16 should not regress per-token throughput, may improve large-batch via 2× total HBM capacity = larger KV cache).

```bash
# Run same evalscope sweep against multi-host LWS leader pod
kubectl --context="$CTX" exec qwen35-mh-0 -- bash -c "
  pip install -q evalscope[perf]
  bash scripts/run_bench_qwen35.sh
"
```

---

## Troubleshooting

| Symptom | Root Cause | Fix |
|---|---|---|
| **Multi-concurrency garbled output / OOM `vmem 86MB > 64MB` / `HBM 95G > 94.75G` / EngineCore silent crash** | Missing PR #2366 (KV cache state corruption) | Follow [Step 2](#step-2-apply-pr-2366-patch-if-not-already-patched), grep should output 7 |
| **Weight load 80s/shard (vs normal 2s), startup from 7min to 2hr** | `/dev/shm` residuals → insufficient RAM → vLLM skips auto-prefetch | Clean `/dev/shm`, add `--safetensors-load-strategy=prefetch` at startup |
| **`ABORTED: libtpu lockfile` / `TPU device busy`** | Previous vLLM abnormal exit, orphan process holding `/dev/vfio/0` | `pgrep -f 'EngineCore\|vllm' \| xargs -r kill -9` + `rm -f /tmp/libtpu_lockfile` |
| **`kubectl exec ... bash -c "<multi-line nohup>"` returns exit 137** | kubectl exec kills process group when stdin closes, nohup can't save it | Use file-based launcher ([Step 3](#step-3-start-vllm-file-based-launcher)) |
| **PD mode: `ValueError: Hybrid KV cache manager is disabled but failed to convert KV cache specs to one unified type`** | vLLM defaults to disable HMA when `kv_transfer_config` is set | Add `--no-disable-hybrid-kv-cache-manager` flag ([PD mode required differences](#-qwen35-pd-required-differences-vs-qwen3-coder-pd)) |

---

## Appendix

### Full Benchmark Data

#### Throughput Sweep (1K/1K, evalscope perf, warmup + record)

| Batch | Latency | Throughput | Per-user | Pareto |
|---:|---:|---:|---:|---|
| P1 | 20.6 s | 49.6 tok/s | 49.6 | 💨 Low Latency |
| P2 | 21.2 s | 96.7 | 48.4 | |
| P4 | 21.9 s | 186.8 | 46.7 | Interactive |
| P8 | 22.8 s | 358.7 | 44.8 | |
| P16 | 25.6 s | 640 | 40.0 | Online serving |
| P32 | 28.9 s | 1129 | 35.3 | |
| P64 | 43.2 s | 1510 | 23.6 | ⚖️ Balanced |
| **P128** | 61.8 s | **2103 tok/s** ⭐ | 16.4 | 🚀 Peak |
| P256 | 108.4 s | 1877 ↓ | 7.3 | (Past sweet spot) |

**Pareto operating points**:
- Single user / TPOT < 25 ms → **P1** (49.6 tok/s/user)
- Medium concurrency → **P32-P64** (1100-1500 tok/s total, 23-35 tok/s/user)
- Offline batch / max throughput → **P128** (**2097 tok/s** peak, per-chip 262 tok/s)

#### Thinking ON vs OFF (1K/1K) — Raw Throughput Nearly Identical

| Batch | OFF (tok/s) | ON (tok/s) |
|---:|---:|---:|
| P1 | 49.6 | 49.6 |
| P16 | 640 | 638 |
| P64 | 1510 | 1518 |

⚠️ Business-effective tokens differ by 10×: thinking ON generates ~90% reasoning + 10% answer in the same time; OFF generates mostly answer. When thinking can be turned off, business efficiency improves 10× (see [Constraint D](#d-thinking-behavior) for how).

#### Long Context (vLLM `--max-model-len 16384`)

**8K input / 1K output (prefill heavy)**:

| Batch | Throughput (tok/s) | vs 1K input |
|---:|---:|---|
| P1 | 51.7 | **+4%** (low concurrency not impacted) |
| P4 | 178.6 | -4% |
| P16 | 499.1 | -22% |
| P64 | 849.9 | -44% |

**1K input / 8K output (decode heavy)**:

| Batch | Throughput (tok/s) | vs 1K out |
|---:|---:|---|
| P1 | 54.0 | **+9%** |
| P4 | 203.3 | **+9%** |
| P16 | 711.0 | **+11%** |
| **P64** | **1702** | **+13%** ⭐ |

🎯 **Long generation is 9-13% faster than short generation at all batch sizes** (pure decode keeps TPU MXU at sustained high utilization). **For long output scenarios (article/code generation), P64 1702 tok/s is the v7x-8 sweet spot**.

### Hardware Requirements

| Item | Requirement |
|---|---|
| TPU | v7x-8 (8 chips, v7x-16 not needed) |
| HBM | 94.75 GB/device × 8 = 758 GB (per-device uses ~85 GB / 90% util) |
| Host memory | ≥ 800 GB (page cache for 378 GB checkpoint) |
| Storage | ≥ 600 GB (model 378 GB + workspace) |

### Model Core Parameters

| Field | Value |
|---|---|
| Architecture | MoE + **Hybrid GDN/Attention** (60 layers = 45 GDN + 15 Standard Attn) |
| Parameters | 397B total / 17B active / 512 routed + 1 shared expert / Top-K=10 |
| Dimensions | Hidden 4096, Attn 32 Q + 2 KV (GQA), head_dim 256, Expert intermediate 1024 |
| Context | Native 262K, YaRN extensible to 1M |
| Quantization | FP8 native, vocab 248,320 |

### Upstream PR Watchlist (When Patches Can Be Removed)

> Once the vllm-tpu nightly image is upgraded to ≥ the PR merge date, the corresponding patch can be removed. Current image uses `nightly-20260330` which is 3-4 weeks earlier than all 4 PRs, so patches are still required.

| PR | Author | Merged | Status | Which Patch / Mode Depends On It |
|---|---|---|---|---|
| [#2322](https://github.com/vllm-project/tpu-inference/pull/2322) | wyzhang | 2026-04-20 | ✅ merged | PD: transfer stats tracker (no cp needed, already in main) |
| [#2336](https://github.com/vllm-project/tpu-inference/pull/2336) | wyzhang | 2026-04-22 | ✅ merged | PD: `tpu_connector_hma.py` (cp into pod) |
| [#2366](https://github.com/vllm-project/tpu-inference/pull/2366) | qizzzh | 2026-04-23 | ✅ merged | **All modes: `kv_cache_manager.py`** (cp into pod) |
| 🔬 mrope bypass (internal) | — | Pending upstream | ⚪ Internal patch | Multi-host: `tpu_runner.py` + `persistent_batch_manager.py` (5+4 lines defensive None check) |

**Cleanup procedure after image upgrade**:
1. `kubectl exec $POD -- ls /workspace/tpu_inference/.git/HEAD` to check main commit hash
2. Compare PR merge dates: if image commit ≥ PR commit → that patch can be removed
3. Remove init container's `cp ... patches/` + verify counter lines
4. **Do not remove mrope bypass** — no upstream PR yet, still required for multi-host

> 💡 **Future direction**: mrope bypass should be filed as a PR to vllm-project/tpu-inference. Fix point: when `disable_mm_from_limits=True`, `PersistentBatchManager.__init__` should lazy-read `runner.uses_mrope` instead of capturing it at init.

### References

- [PR #2366 — Hybrid KV cache OOB fix (required)](https://github.com/vllm-project/tpu-inference/pull/2366)
- [PR #2322 / #2327 / #2331 / #2336 — PD disagg 4 PR series (including HMA implementation)](https://github.com/vllm-project/tpu-inference/pull/2322)
- [tpu_connector_hma.py — TPUConnectorHMA source code](https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/distributed/tpu_connector_hma.py)
- [Qwen3.5-397B-A17B-FP8 HuggingFace](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8)
- Related READMEs: [DeepSeek R1 FP4](../DeepSeek-R1-671B-FP4/README.en.md) · [GLM-5.1 FP4](../GLM-5.1-754B-FP4/README.en.md) · [Kimi K2.6](../Kimi-K2.6-1T-A32B-INT4/README.en.md) · [Qwen3-Coder 480B](../Qwen3-Coder-480B/README.en.md)

### 📚 Internal Documents (Dogfood History + Deep Analysis)

4 complementary HTML documents in chronological order recording the complete dogfood from initial deployment to PD disaggregation:

| Date | Topic | Details |
|---|---|---|
| 2026-04-24 | [Deployment & Optimization Guide v1.5](https://cc.higcp.com/pages/qwen35-397b-tpu-inference-plan-20260424.html) | Complete deployment + optimization decisions (108 KB) |
| 2026-04-25 | [⭐ Single-machine Inference Debug Story — 4-hour Detour vs 14-minute Correct Path](https://cc.higcp.com/pages/qwen35-397b-debug-story-20260425.html) | Complete single-machine deployment debug record (45 KB) |
| 2026-04-26 | [README Reproducibility Verification Report](https://cc.higcp.com/pages/qwen35-readme-verification-20260426.html) | Blind run following README steps + verification (23 KB) |
| 2026-04-26 | [⭐ PD Disaggregation Deployment Dogfood Record](https://cc.higcp.com/pages/qwen35-pd-disagg-dogfood-20260426.html) | Full PD workflow + HMA root cause + 6 lessons (24 KB) |
| 2026-04-26 | [⭐ Multi-host TP=16 Dogfood Record](https://cc.higcp.com/pages/qwen35-multihost-dogfood-20260426.html) | 5 tests revealing 4-layer root cause chain + single/multi-host comparison (27 KB) |

> 💡 **Link behavior**: cc.higcp.com uses GCP IAP, accessible in browser with google account login; for external access use `https://storage.googleapis.com/chris-pgp-host-asia/cc-pages/pages/<file>.html` direct links.
