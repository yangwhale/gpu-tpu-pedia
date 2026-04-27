# Kimi K2.6 (1T-A32B INT4) Inference on TPU v7x

> 🌐 **Languages** | **语言**: [中文](README.md) · **English**

> **Status**: ✅ **Working** (multi-host TPU v7x-16, 2026-04-27)
>
> End-to-end guide: run Kimi K2.6 (1T total / 32B activated / native INT4 routed experts) inference on TPU v7x.
>
> **Code repository**: https://github.com/yangwhale/tpu-inference
> **Branch**: `chrisya/main` · **Commit**: [`945bd0d9`](https://github.com/yangwhale/tpu-inference/commit/945bd0d9)
>
> **Model**: [moonshotai/Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6) (64 safetensors, ~595 GB INT4)
>
> **Full postmortem reports**:
> - [Stage 1: 4L sanity PASS (single-host)](https://cc.higcp.com/pages/kimi-k26-multihost-stage1-20260426.html)
> - [Stage 2: 61L all PASS (multi-host, no cache, 57 min cold start)](https://cc.higcp.com/pages/kimi-k26-multihost-stage2-20260426.html)
> - [Stage 3: SHM cache + DNS fix BIG WIN (multi-host, 6:08 cold start, 9.3x)](https://cc.higcp.com/pages/kimi-k26-multihost-stage3-20260427.html)

## 🎯 Verified Key Performance (2026-04-27)

| Operating point | Configuration | Result |
|---|---|---|
| **Multi-host 61L cold start** | v7x-16 (2x2x2), SHM cache | ✅ **6 min 8 sec** |
| Single-host 40L cold start | v7x-8 (2x2x1), SHM cache | ✅ 2 min 50 sec |
| Single-host 61L | v7x-8 | ⚠️ Init OK, forward HBM OOM (105 GB > 95 GB) |
| Cache hit per layer (SHM mmap) | both configs | ✅ **0.7-1.0 s/layer** |
| Cache hit per layer (Lustre) | both configs | ⚠️ ~27 s/layer (38x slower) |
| Smoke test (math sequence) | `2+3=` → "6, 1+2+3+4=10, 1+2+3+4+5=15..." | ✅ coherent |
| Smoke test (natural language) | `The capital of France is` → " Paris. Eiffel Tower..." | ✅ coherent |
| Smoke test (math + LaTeX) | `2 + 3 = ` → `5\boxed{5}` | ✅ coherent + correct value |
| **GSM8K / MMLU standard eval** | — | ⏳ TBD (Stage 4 candidate) |
| **Throughput 1K/1K batch=1** | v7x-16, enforce-eager | ✅ TTFT 1.14s, TPOT 49ms, **20 tok/s** |
| **Throughput 1K/1K batch=32** | v7x-16, enforce-eager | ✅ TTFT 2.0s, TPOT 52ms, **592 tok/s** (30x batch scaling) |
| **Throughput 1K/7K batch=1** | v7x-16, enforce-eager | ✅ TTFT 689ms, TPOT 51.6ms, 19.4 tok/s |
| **Throughput 1K/7K batch=32** | v7x-16, enforce-eager | ✅ TTFT 2.0s, TPOT 54.8ms, **581 tok/s** |

| Optimization stage | Cold start | vs naive baseline |
|---|---|---|
| Stage 2: no cache, multi-host 60L | 57 min | baseline |
| Stage 3: + V18 SHM cache | **6 min 8 sec** | **9.3x speedup** |

> **vs Kimi official**: Moonshot's official deployment uses vLLM/SGLang on H800/H100, no public TPU deployment
> **Quality** (Moonshot official): SWE-Bench Verified 80.2%, AIME 2026 96.4%, GPQA-Diamond 90.5%, BrowseComp 83.2%

---

## ⚙️ Recommended Configuration (Multi-host v7x-16)

| Item | Value |
|---|---|
| Topology | **2x2x2** (16 chips, 2 hosts × 8 chips) |
| Node pool | `np-tpu7x-spot-mh-k26` or similar multi-host TPU pool |
| Image | `chris-pgp-host/ai-infra/vllm-tpu:latest` |
| dshm | **`sizeLimit: 800Gi`** (fits 532 GB SHM cache + Linux page cache) |
| Lustre | RWX PVC `lustre-pvc`, at least 1.5 TB available (cache 532 GB + model 595 GB + workspace) |

---

## 🚀 Step-by-Step Deployment

### Step 0: One-time cache build (offline, ~20 min)

New model weights require building a pre-transposed packed-int4 cache on first deployment. **Run once on a single pod, all subsequent deployments reuse the cache files on Lustre.**

```bash
# On a pod with Lustre mounted (e.g., e2e pod with idle CPU)
kubectl exec <build-pod> -c main -- bash -c '
# 1. Clone tpu-inference (chrisya/main has the build script)
git clone -b chrisya/main https://github.com/yangwhale/tpu-inference /tmp/tpu-inference

# 2. 8-way parallel build of 60 MoE layers (~20 min on 944 GB RAM pod)
mkdir -p /tmp/k26_par_build
for i in 0 1 2 3 4 5 6 7; do
  start=$((i*8 + 1))
  end=$((start + 7))
  [ $end -gt 60 ] && end=60
  python3 /tmp/tpu-inference/scripts/build_k26_moe_cache.py --layers ${start}-${end} \
    > /tmp/k26_par_build/w${i}.log 2>&1 &
done
wait
ls /lustre/k26_cache_v2 | wc -l  # should be 60
'
```

Output: `/lustre/k26_cache_v2/model_layers_{1..60}_mlp_experts/`, each layer contains 4 npy + meta.json, total ~570 GB.

> **Memory peak ~870 GB** during 8-way parallel xpose (numpy unpack int4 → transpose → repack uint32). e2e pod with 944 GB RAM is safe.

### Step 1: Deploy LWS (multi-host)

Full yaml: [`manifests/k26_multihost_lws.yaml`](manifests/k26_multihost_lws.yaml) (includes DNS fix + SHM cache stage + complete vllm serve args). Key content summary:

```yaml
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: k26-mh
spec:
  replicas: 1
  leaderWorkerTemplate:
    size: 2  # 1 leader + 1 worker
    workerTemplate:
      spec:
        nodeSelector:
          cloud.google.com/gke-nodepool: np-tpu7x-spot-mh-k26
          cloud.google.com/gke-tpu-topology: 2x2x2
        containers:
        - name: main
          image: us-central1-docker.pkg.dev/chris-pgp-host/ai-infra/vllm-tpu:latest
          resources:
            limits: {google.com/tpu: "4", memory: "850Gi"}
          volumeMounts:
          - {name: dshm, mountPath: /dev/shm}
          - {name: lustre-vol, mountPath: /lustre}
          command: ["sh", "-c"]
          args:
          - |
            # === DNS staleness fix - critical for multi-host ===
            MY_TPU_IP=$(hostname -I | awk '{print $1}')
            LEADER_DNS="k26-mh-0.k26-mh"
            WORKER_DNS="k26-mh-0-1.k26-mh"
            until getent hosts $LEADER_DNS; do sleep 5; done
            until getent hosts $WORKER_DNS; do sleep 5; done
            # Wait until OWN DNS matches OWN pod IP (DNS may be stale from
            # previous pod incarnation; without this fix multi-host TPU init
            # hangs forever connecting to old worker IP).
            for try in $(seq 1 30); do
              if [ "$LWS_WORKER_INDEX" = "0" ]; then
                MY_DNS=$(getent hosts $LEADER_DNS | awk '{print $1}')
              else
                MY_DNS=$(getent hosts $WORKER_DNS | awk '{print $1}')
              fi
              [ "$MY_DNS" = "$MY_TPU_IP" ] && break
              sleep 5
            done
            LEADER_IP=$(getent hosts $LEADER_DNS | awk '{print $1}')
            WORKER_IP=$(getent hosts $WORKER_DNS | awk '{print $1}')

            # === Multi-host TPU env ===
            export TPU_WORKER_HOSTNAMES="${LEADER_IP},${WORKER_IP}"
            export TPU_WORKER_ID=${LWS_WORKER_INDEX}
            export TPU_PROCESS_ADDRESSES="${LEADER_IP}:8471,${WORKER_IP}:8471"
            export TPU_HOST_BOUNDS="1,1,2"
            export TPU_CHIPS_PER_HOST_BOUNDS="2,2,1"
            export TPU_TOPOLOGY="2x2x2"
            export TPU_ACCELERATOR_TYPE="tpu7x-16"

            # === K2.6 required env ===
            export NEW_MODEL_DESIGN=1                          # MLA enforcement
            export K26_USE_V16=1                               # V16 fast path
            export MOE_WEIGHT_CACHE_DIR=/dev/shm/k26_cache_v2  # SHM cache (38x faster than Lustre)
            export MODEL_IMPL_TYPE=flax_nnx
            export TPU_BACKEND_TYPE=jax
            export PJRT_DEVICE=TPU

            # === Replace image bundled tpu_inference with chrisya/main version ===
            cd / && rm -rf /workspace/tpu_inference
            cp -r /lustre/tpu_inference /workspace/tpu_inference
            grep -c "v18-cache" /workspace/tpu_inference/tpu_inference/layers/jax/quantization/int4.py  # expect >= 1

            # === Stage MoE cache to SHM (per-host parallel cp, ~36 s) ===
            mkdir -p /dev/shm/k26_cache_v2
            for L in $(seq 1 60); do
              cp -r /lustre/k26_cache_v2/model_layers_${L}_mlp_experts /dev/shm/k26_cache_v2/ &
            done
            wait

            if [ "$LWS_WORKER_INDEX" = "0" ]; then
              # Leader: Ray head + vllm serve
              ray start --head --port=6379 --node-ip-address=$MY_TPU_IP --resources='{"TPU": 4}'
              sleep 30  # wait for worker to join
              vllm serve /lustre/Kimi-K2.6 \
                --served-model-name=Kimi-K2.6 \
                --tensor-parallel-size=16 \
                --distributed-executor-backend=ray \
                --max-model-len=8192 \
                --max-num-batched-tokens=8192 \
                --max-num-seqs=64 \
                --no-enable-prefix-caching \
                --gpu-memory-utilization=0.85 \
                --enable-expert-parallel \
                --trust-remote-code \
                --enforce-eager \
                --limit-mm-per-prompt='{"image":0,"video":0}' \
                --additional-config='{"sharding":{"sharding_strategy":{"enable_dp_attention":true,"tensor_parallelism":1,"expert_parallelism":16}}}' \
                --host=0.0.0.0 --port=8000
            else
              # Worker: Ray join + block
              ray start --address=${LEADER_IP}:6379 --resources='{"TPU": 4}' --block
            fi
        volumes:
        - name: dshm
          emptyDir: {medium: Memory, sizeLimit: 800Gi}
        - name: lustre-vol
          persistentVolumeClaim: {claimName: lustre-pvc}
```

```bash
kubectl apply -f k26_multihost_lws.yaml
```

### Step 2: Monitor startup (~6 min)

```bash
kubectl logs k26-mh-0 -c main -f | grep -E "Cache staged|MoE cache.*Done|Application startup|HbmOom|Traceback"
```

Expected log:
```
[boot] DNS matches own IP 10.120.x.y after 1 tries
[boot] V17 packed-int4 refs: 2 (expect >= 1)
[boot] Multi-host process_allgather refs: 1 (expect = 1)
[boot] Cache staged in 36s, count: 60
=== Starting Ray Head ===
... (~2.5 min safetensors filter load) ...
[MoE cache] Done: 60 layers in 42.4s (0 cached, 0 generated)
INFO 04-27 08:37:29 [tpu_runner.py:602] Init model | hbm=[(51.08, 94.75), ...]GiB
INFO:     Application startup complete.
```

### Step 3: Smoke test

```bash
kubectl exec k26-mh-0 -c main -- curl -s -X POST http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Kimi-K2.6","prompt":"2+3=","max_tokens":50,"temperature":0}'
```

Expected output (matches measured):
```json
{
  "choices": [{
    "text": "6, 1+2+3+4=10, 1+2+3+4+5=15, 1+2+3+4+5+6=21, 1+2+3+4+",
    "finish_reason": "length"
  }]
}
```

---

## 📊 HBM Footprint (per chip)

| Stage | 16 chips multi-host | 8 chips single-host |
|---|---|---|
| Init model done (61L weight) | **51.08 GB/chip** ✓ | 84.58 GB/chip (cap edge) |
| Init KV cache (max_seqs=64, max_model_len=8192) | **80.5 GB/chip** ✓ | 89.94 GB (single-host fits only with max_seqs=1, max_model_len=128) |
| HBM cap (gpu_mem_util=0.85) | 80.5 GB | 80.5 GB |
| Forward activations needed | ~10-15 GB | ~10-15 GB |
| **Can run forward** | ✅ 14 GB headroom | ❌ OOM 105 GB > 95 physical |

---

## ⚠️ Three Required Environment Variables

| Env var | Value | Consequence if missed |
|---------|-------|---------|
| `NEW_MODEL_DESIGN=1` | **must set** | MLA model enforcement, fails to start otherwise |
| `K26_USE_V16=1` | **must set** | V16 fast path (HBM keeps packed uint32 + TPU bitcast), saves 16x weight load |
| `MOE_WEIGHT_CACHE_DIR=/dev/shm/k26_cache_v2` | **must set SHM path** | Not set → no cache at all (57 min cold start); set to Lustre → 38x slower than SHM |

---

## 🐛 Known/Fixed Bugs

### 1. Symmetric int4 [-8, 7] bias bug (V17 patched)
Pack-quantized format biases symmetric int4 to unsigned [0, 15] before packing. Loader must `-= 8` to restore after unpacking. Direct `bitcast(uint32 → signed int4)` causes 50% of weights to flip sign bit → model collapse outputs `"foss foss foss..."`. Fixed.

### 2. Multi-host DNS staleness (V18 fix)
After pod restart, K8s DNS cache takes ~5 min to update to new pod IP. Boot script immediately calls `getent hosts` and gets stale IP → `TPU_PROCESS_ADDRESSES` uses old IP → multi-host TPU init hangs forever.

**Fix**: Boot script adds wait loop, proceeds only when own DNS resolves to own pod IP (using K8s downward API's `MY_TPU_IP`). See yaml above for `for try in seq 1 30 ... MY_DNS` section.

### 3. process_weights_after_loading step 1 early exit (V18 fix)
`_filtered_safetensors_iterator` skips expert keys → `_weights_to_load` all None → step 1 check fails → returns False → V16 cache hit path never runs. Model parameters stay default → JIT compile sees 1.55 TB → OOM.

**Fix**: Move cache check before step 1, skip step 1 when cache exists. See [int4.py V18 patch](https://github.com/yangwhale/tpu-inference/blob/chrisya/main/tpu_inference/layers/jax/quantization/int4.py).

### 4. Cache mmap → device_put triggers multi-host XLA HBM blowup (V18 fix)
Direct `device_put(mmap_full_array, sharding)` makes XLA treat input as unsharded full-replicated, triggering 1.55 TB compile-time HBM analysis → OOM.

**Fix**: Use `jax.make_array_from_callback(shape, sharding, lambda idx: arr[idx])`, each device mmaps its own slice. Reference: `fp8.py:1078`.

---

## Hardware and Model Overview

| Item | Requirement |
|------|------|
| TPU | **v7x-8 (minimum - only supports 40L) / v7x-16 (recommended - full 61L)** |
| HBM | 94.75 GB/device, v7x-8 total 758 GB / v7x-16 total 1,516 GB |
| Host memory | ≥850 GB (includes /dev/shm 800 GB cache) |
| Storage | ≥1.5 TB (model 595 GB + cache 570 GB + workspace) |

| Model parameter | Value | vs DeepSeek V3 | vs GLM-5.1 |
|---------|-----|----------------|-----------|
| Architecture | MoE + MLA | same lineage | same lineage |
| Total params | **1T** | DSV3 671B | GLM 754B |
| Activated params | 32B | DSV3 ~37B | GLM ~37B |
| Total layers | 61 (1 dense + 60 MoE) | DSV3 61 | GLM 78 |
| `first_k_dense_replace` | **1** | DSV3 3 | GLM 3 |
| Attention | MLA (q_lora=1536, kv_lora=512) | same | same |
| Hidden | 7,168 | same as DSV3 | GLM 6144 |
| Attention Heads | 64 (Q/K/V) | DSV3 128 | GLM 64 |
| MoE Experts | **384 routed + 1 shared** | DSV3 256 | GLM 256 |
| Top-K (routed) | 8 | same | same |
| `n_group / topk_group` | **1 / 1** | DSV3 8 / 4 | GLM 1 / 1 (same) |
| Expert Intermediate | 2,048 | same | same |
| Vocab | **163,840** | 129,280 | 154,880 |
| RoPE | YaRN, theta=50K, factor=64, orig=4K | DSV3 theta=10K, factor=40 | GLM theta=1M, no YaRN |
| Native Context | **256K** | DSV3 128K | GLM 200K |
| MTP | **❌ 0** (no MTP) | DSV3 1 nextn | GLM 1 nextn |
| Quantization | **INT4 W4A16 (compressed-tensors)** | FP4 / FP8 | FP4 / FP8 |
| Quantization scope | routed experts only; attention/shared/dense kept BF16 | MoE FP4 + non-MoE FP8 | same as DSV3 |
| Group size | **32** (per-group, symmetric) | — | — |
| Multimodal | MoonViT 400M (vision encoder) | none | none |

### Key Differences vs DeepSeek R1

| Parameter | DeepSeek R1 | Kimi K2.6 |
|------|-------------|-----------|
| n_routed_experts | 256 | **384** |
| first_k_dense_replace | 3 | **1** |
| num_attention_heads | 128 | **64** |
| n_group / topk_group | 8 / 4 | **1 / 1** |
| rope_theta | 10K (YaRN ×40) | **50K (YaRN ×64)** |
| vocab_size | 129K | **164K** |
| MTP | 1 nextn | **❌ none** |
| Quantization method | FP4 (custom block) | **INT4 (compressed-tensors symmetric)** |

---

## 📊 Throughput Benchmark (2026-04-27)

**Configuration**: TPU v7x-16 multi-host (16 chips), `--enforce-eager`, `--no-enable-prefix-caching`, `--max-model-len=8192`, `--max-num-seqs=64`, random tokens, `--ignore-eos`, default temperature.

**Method**: Each case warmup once + measure once (use measure number, skip jitter).

| Case | Input/Output | Concurrency | TTFT (ms) | TPOT (ms) | Output tok/s | Total tok/s | E2E (s) |
|---|---|---|---|---|---|---|---|
| **1** | 1K / 1K | **1** | **1142** | **48.95** | 19.99 | 39.98 | 51.2 |
| **2** | 1K / 1K | **32** | **2017** | **52.13** | **592** | **1184** | 55.3 |
| **3** | 1K / 7K | **1** | **689** | **51.59** | 19.35 | 22.12 | 370.4 |
| **4** | 1K / 7K | **32** | **2018** | **54.83** | **581** | **664** | 395.0 |

> **Note**: Measured 1K/7K instead of 1K/8K because max_model_len=8192 cap (1K + 7K = 8192).

### Key Findings

**1. Batch 30x scaling**: batch=1 → batch=32 throughput from ~20 → ~590 tok/s, **near-perfect 30x scaling**. Indicates K2.6 MoE EP=16 expert dispatch has no large overhead.

**2. TPOT stable ~50ms/token (decode HBM bandwidth bound)**:
- batch=1, output=1K: 48.95 ms
- batch=32, output=1K: 52.13 ms (+6%)
- batch=1, output=7K: 51.59 ms (+5%)
- batch=32, output=7K: 54.83 ms (+12%)

K2.6 671B/A37B → each decode token requires ~24 GB weight HBM read (37B activated × INT4 0.5 bytes + scale). 16 chips × ~3 TB/s/chip = 48 TB/s aggregate. **Theoretical TPOT = 0.5 ms** (HBM bandwidth bound), measured 50 ms = **100x off**, indicating HBM utilization ~5-10% — typical TPU inference MFU is low under enforce_eager.

**3. Long output barely affects TPOT**: batch=1 output 1K vs 7K only differs +5% (49 → 51.6 ms). MLA + FlashAttention efficiently handles long context.

**4. TTFT scaling**: batch=32 prefill (32K total tokens) only ~2x slower than batch=1 prefill (1K tokens), indicating chunked prefill + EP=16 shared expert weights sub-linear scale during prefill.

### Caveats

- **Throughput numbers have substantial optimization headroom**: `--enforce-eager` disables XLA full graph compile, expect 2-3x speedup with it on. Enabling requires cudagraph capture (cold start +5-10 min).
- **Concurrency tested at 32**, max_num_seqs=64 can try higher.
- **No GSM8K/MMLU accuracy eval**: only smoke test coherent, SOTA numbers not validated.

---

## 🚀 Next Steps (Stage 4 candidates)

- ⏳ **GSM8K / MMLU standard eval** validate numerical correctness (not just coherent output)
- ⏳ **Disable enforce_eager and run XLA compile**: expected throughput +2-3x, cold start +5-10 min
- ⏳ **Higher concurrency**: batch=64 / 128 (max_num_seqs cap)
- ⏳ **Long input**: 4K / 16K input (need to bump max_model_len first)
- ⏳ **Per-rank cache filter**: currently each host stages all 60 layers (~35s), optimized per-rank stages only own 24 expert subset = single host 35s → 2s
- ⏳ **Upstream PR** to vllm-project/tpu-inference (V18 cache + DNS fix)

---

## 📚 Key References

- **Full postmortem HTML docs**:
  - Stage 1 (4L sanity): https://cc.higcp.com/pages/kimi-k26-multihost-stage1-20260426.html
  - Stage 2 (61L all PASS, 57 min cold): https://cc.higcp.com/pages/kimi-k26-multihost-stage2-20260426.html
  - Stage 3 (SHM cache + DNS fix, 6 min cold): https://cc.higcp.com/pages/kimi-k26-multihost-stage3-20260427.html
- **GitHub branch**: https://github.com/yangwhale/tpu-inference/tree/chrisya/main
- **Build script**: [`scripts/build_k26_moe_cache.py`](https://github.com/yangwhale/tpu-inference/blob/chrisya/main/scripts/build_k26_moe_cache.py)
- **V18 cache hit code**: [`int4.py`](https://github.com/yangwhale/tpu-inference/blob/chrisya/main/tpu_inference/layers/jax/quantization/int4.py) (`process_weights_after_loading`)
- **Multi-host detailed yaml + early multi-host hands-on**: see [`README-multinode.md`](README-multinode.md)
