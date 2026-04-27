# Kimi K2.6 (1T-A32B INT4) Inference on TPU v7x

> 🌐 **Languages** | **语言**: **中文** · [English (TBD)](README.en.md)

> **状态**: ✅ **已调通** (multi-host TPU v7x-16, 2026-04-27)
>
> 端到端指南：在 TPU v7x 上运行 Kimi K2.6（1T 总参 / 32B 激活 / native INT4 routed experts）推理。
>
> **代码仓库**: https://github.com/yangwhale/tpu-inference
> **Branch**: `chrisya/main` · **Commit**: [`945bd0d9`](https://github.com/yangwhale/tpu-inference/commit/945bd0d9)
>
> **模型**: [moonshotai/Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6)（64 safetensors, ~595 GB INT4）
>
> **完整复盘报告**:
> - [Stage 1: 4L sanity PASS (single-host)](https://cc.higcp.com/pages/kimi-k26-multihost-stage1-20260426.html)
> - [Stage 2: 61L 全 PASS (multi-host, no cache, 57 min cold start)](https://cc.higcp.com/pages/kimi-k26-multihost-stage2-20260426.html)
> - [Stage 3: SHM cache + DNS fix BIG WIN (multi-host, 6:08 cold start, 9.3x)](https://cc.higcp.com/pages/kimi-k26-multihost-stage3-20260427.html)

## 🎯 已验证的关键性能 (2026-04-27)

| 操作点 | 配置 | 结果 |
|---|---|---|
| **Multi-host 61L cold start** | v7x-16 (2x2x2), SHM cache | ✅ **6 min 8 sec** |
| Single-host 40L cold start | v7x-8 (2x2x1), SHM cache | ✅ 2 min 50 sec |
| Single-host 61L | v7x-8 | ⚠️ Init OK, forward HBM OOM (105 GB > 95 GB) |
| Cache hit per layer (SHM mmap) | both configs | ✅ **0.7-1.0 s/layer** |
| Cache hit per layer (Lustre) | both configs | ⚠️ ~27 s/layer (38x slower) |
| Smoke test (math sequence) | `2+3=` → "6, 1+2+3+4=10, 1+2+3+4+5=15..." | ✅ coherent |
| Smoke test (natural language) | `The capital of France is` → " Paris. Eiffel Tower..." | ✅ coherent |
| Smoke test (math + LaTeX) | `2 + 3 = ` → `5\boxed{5}` | ✅ coherent + 数值正确 |
| **GSM8K / MMLU 标准评估** | — | ⏳ 待跑 (Stage 4 候选) |
| **Throughput 1K/1K batch=1** | v7x-16, enforce-eager | ✅ TTFT 1.14s, TPOT 49ms, **20 tok/s** |
| **Throughput 1K/1K batch=32** | v7x-16, enforce-eager | ✅ TTFT 2.0s, TPOT 52ms, **592 tok/s** (30x batch scaling) |
| **Throughput 1K/7K batch=1** | v7x-16, enforce-eager | ✅ TTFT 689ms, TPOT 51.6ms, 19.4 tok/s |
| **Throughput 1K/7K batch=32** | v7x-16, enforce-eager | ✅ TTFT 2.0s, TPOT 54.8ms, **581 tok/s** |

| 优化层级 | Cold start | vs naive baseline |
|---|---|---|
| Stage 2: no cache, multi-host 60L | 57 min | baseline |
| Stage 3: + V18 SHM cache | **6 min 8 sec** | **9.3x speedup** |

> **vs Kimi 官方**: Moonshot 官方部署用 vLLM/SGLang on H800/H100，未公开 TPU 部署
> **质量** (Moonshot 官方): SWE-Bench Verified 80.2%, AIME 2026 96.4%, GPQA-Diamond 90.5%, BrowseComp 83.2%

---

## ⚙️ 推荐配置 (Multi-host v7x-16)

| 项 | 值 |
|---|---|
| Topology | **2x2x2** (16 chips, 2 host × 8 chip) |
| Node pool | `np-tpu7x-spot-mh-k26` 或类似 multi-host TPU pool |
| Image | `chris-pgp-host/ai-infra/vllm-tpu:latest` |
| dshm | **`sizeLimit: 800Gi`** (装下 532 GB SHM cache + Linux page cache) |
| Lustre | RWX PVC `lustre-pvc`, 至少 1.5 TB 可用 (cache 532 GB + 模型 595 GB + 工作空间) |

---

## 🚀 Step-by-Step 部署步骤

### Step 0: 一次性 build cache (offline, ~20 min)

新模型权重首次部署需要 build pre-transposed packed-int4 cache。**只在一台 pod 上跑一次，之后所有部署复用 Lustre 上的 cache 文件**。

```bash
# 在挂 Lustre 的 pod 上 (e.g., e2e pod with idle CPU)
kubectl exec <build-pod> -c main -- bash -c '
# 1. clone tpu-inference (chrisya/main 含 build script)
git clone -b chrisya/main https://github.com/yangwhale/tpu-inference /tmp/tpu-inference

# 2. 8-way parallel build 60 MoE layers (~20 min on 944 GB RAM pod)
mkdir -p /tmp/k26_par_build
for i in 0 1 2 3 4 5 6 7; do
  start=$((i*8 + 1))
  end=$((start + 7))
  [ $end -gt 60 ] && end=60
  python3 /tmp/tpu-inference/scripts/build_k26_moe_cache.py --layers ${start}-${end} \
    > /tmp/k26_par_build/w${i}.log 2>&1 &
done
wait
ls /lustre/k26_cache_v2 | wc -l  # 应该 60
'
```

输出：`/lustre/k26_cache_v2/model_layers_{1..60}_mlp_experts/`，每 layer 含 4 个 npy + meta.json，总 ~570 GB。

> **Memory peak ~870 GB** during 8-way parallel xpose (numpy unpack int4 → transpose → repack uint32). e2e pod 带 944 GB RAM 安全。

### Step 1: 部署 LWS (multi-host)

完整 yaml: [`manifests/k26_multihost_lws.yaml`](manifests/k26_multihost_lws.yaml) (含 DNS fix + SHM cache stage + 完整 vllm serve args). 关键内容摘要：

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

            # === K2.6 必设 env ===
            export NEW_MODEL_DESIGN=1                          # MLA 强制
            export K26_USE_V16=1                               # V16 fast path
            export MOE_WEIGHT_CACHE_DIR=/dev/shm/k26_cache_v2  # SHM cache (38x faster than Lustre)
            export MODEL_IMPL_TYPE=flax_nnx
            export TPU_BACKEND_TYPE=jax
            export PJRT_DEVICE=TPU

            # === 替换 image bundled tpu_inference 为 chrisya/main 版 ===
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
              sleep 30  # 等 worker join
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

### Step 2: 监控启动 (~6 min)

```bash
kubectl logs k26-mh-0 -c main -f | grep -E "Cache staged|MoE cache.*Done|Application startup|HbmOom|Traceback"
```

预期 log:
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

### Step 3: smoke test

```bash
kubectl exec k26-mh-0 -c main -- curl -s -X POST http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Kimi-K2.6","prompt":"2+3=","max_tokens":50,"temperature":0}'
```

期待输出 (跟实测一致):
```json
{
  "choices": [{
    "text": "6, 1+2+3+4=10, 1+2+3+4+5=15, 1+2+3+4+5+6=21, 1+2+3+4+",
    "finish_reason": "length"
  }]
}
```

---

## 📊 HBM 占用 (per chip)

| 阶段 | 16 chips multi-host | 8 chips single-host |
|---|---|---|
| Init model done (61L weight) | **51.08 GB/chip** ✓ | 84.58 GB/chip (cap 边缘) |
| Init KV cache (max_seqs=64, max_model_len=8192) | **80.5 GB/chip** ✓ | 89.94 GB (single-host 用 max_seqs=1, max_model_len=128 才装下) |
| HBM cap (gpu_mem_util=0.85) | 80.5 GB | 80.5 GB |
| Forward activations needed | ~10-15 GB | ~10-15 GB |
| **是否能跑 forward** | ✅ 留 14 GB headroom | ❌ OOM 105 GB > 95 physical |

---

## ⚠️ 三个必设环境变量

| 环境变量 | 值 | 漏设后果 |
|---------|-----|---------|
| `NEW_MODEL_DESIGN=1` | **必须设** | MLA 模型强制要求，不设直接报错退出 |
| `K26_USE_V16=1` | **必须设** | V16 fast path (HBM 保留 packed uint32 + TPU bitcast)，省 16x weight load |
| `MOE_WEIGHT_CACHE_DIR=/dev/shm/k26_cache_v2` | **必须设 SHM 路径** | 不设 → 完全没 cache (57 min cold start)；设 Lustre → 38x 慢于 SHM |

---

## 🐛 已知/已修 Bug

### 1. Symmetric int4 [-8, 7] 偏置 bug (V17 patched)
Pack-quantized format 把 symmetric int4 偏置成 unsigned [0, 15] 再 pack。Loader 解包后必须 `-= 8` 还原。直接 `bitcast(uint32 → signed int4)` 让 50% weight 错位 sign bit → model collapse 输出 `"foss foss foss..."`。已修。

### 2. Multi-host DNS staleness (V18 fix)
Pod restart 后 K8s DNS cache 用 ~5 min 才更新到新 pod IP。Boot script 立即 `getent hosts` 拿到 stale IP → `TPU_PROCESS_ADDRESSES` 用旧 IP → multi-host TPU init 永远 hang。

**修复**: boot script 加 wait loop, own DNS resolves to own pod IP 才 proceed (用 K8s downward API 的 `MY_TPU_IP`)。详见上面 yaml 的 `for try in seq 1 30 ... MY_DNS` 段。

### 3. process_weights_after_loading step 1 早退 (V18 fix)
`_filtered_safetensors_iterator` 跳过 expert keys → `_weights_to_load` 全 None → step 1 检查 fail → return False → V16 cache hit path 永远不跑。模型参数留 default → JIT compile 见 1.55 TB → OOM.

**修复**: cache check 移 step 1 之前，detect cache exists 时 skip step 1。详见 [int4.py V18 patch](https://github.com/yangwhale/tpu-inference/blob/chrisya/main/tpu_inference/layers/jax/quantization/int4.py)。

### 4. Cache mmap → device_put 触发 multi-host XLA HBM blowup (V18 fix)
直接 `device_put(mmap_full_array, sharding)` 让 XLA 把 input 当 unsharded full-replicated 输入分析，触发 1.55 TB compile-time HBM analysis → OOM.

**修复**: 改用 `jax.make_array_from_callback(shape, sharding, lambda idx: arr[idx])`，每 device 自己 mmap own slice。Reference: `fp8.py:1078`.

---

## 硬件与模型概览

| 项目 | 要求 |
|------|------|
| TPU | **v7x-8（最低 - 仅支持 40L）/ v7x-16（推荐 - 完整 61L）** |
| HBM | 94.75 GB/device，v7x-8 共 758 GB / v7x-16 共 1,516 GB |
| 主机内存 | ≥850 GB（含 /dev/shm 800 GB cache） |
| 存储 | ≥1.5 TB（模型 595 GB + cache 570 GB + 工作空间） |

| 模型参数 | 值 | vs DeepSeek V3 | vs GLM-5.1 |
|---------|-----|----------------|-----------|
| 架构 | MoE + MLA | 同源 | 同源 |
| 总参数 | **1T** | DSV3 671B | GLM 754B |
| 激活参数 | 32B | DSV3 ~37B | GLM ~37B |
| 总层数 | 61（1 dense + 60 MoE） | DSV3 61 | GLM 78 |
| `first_k_dense_replace` | **1** | DSV3 3 | GLM 3 |
| Attention | MLA (q_lora=1536, kv_lora=512) | 同 | 同 |
| Hidden | 7,168 | 同 DSV3 | GLM 6144 |
| Attention Heads | 64 (Q/K/V) | DSV3 128 | GLM 64 |
| MoE Experts | **384 routed + 1 shared** | DSV3 256 | GLM 256 |
| Top-K (routed) | 8 | 同 | 同 |
| `n_group / topk_group` | **1 / 1** | DSV3 8 / 4 | GLM 1 / 1（同）|
| Expert Intermediate | 2,048 | 同 | 同 |
| Vocab | **163,840** | 129,280 | 154,880 |
| RoPE | YaRN, theta=50K, factor=64, orig=4K | DSV3 theta=10K, factor=40 | GLM theta=1M, no YaRN |
| Native Context | **256K** | DSV3 128K | GLM 200K |
| MTP | **❌ 0**（无 MTP） | DSV3 1 nextn | GLM 1 nextn |
| 量化 | **INT4 W4A16 (compressed-tensors)** | FP4 / FP8 | FP4 / FP8 |
| 量化范围 | 仅 routed experts；attention/shared/dense 保 BF16 | MoE FP4 + 非 MoE FP8 | 同 DSV3 |
| Group size | **32** (per-group, symmetric) | — | — |
| 多模态 | MoonViT 400M（vision encoder） | 无 | 无 |

### 与 DeepSeek R1 的关键差异

| 参数 | DeepSeek R1 | Kimi K2.6 |
|------|-------------|-----------|
| n_routed_experts | 256 | **384** |
| first_k_dense_replace | 3 | **1** |
| num_attention_heads | 128 | **64** |
| n_group / topk_group | 8 / 4 | **1 / 1** |
| rope_theta | 10K (YaRN ×40) | **50K (YaRN ×64)** |
| vocab_size | 129K | **164K** |
| MTP | 1 nextn | **❌ 无** |
| 量化方法 | FP4 (custom block) | **INT4 (compressed-tensors symmetric)** |

---

## 📊 Throughput Benchmark (2026-04-27)

**配置**: TPU v7x-16 multi-host (16 chips), `--enforce-eager`, `--no-enable-prefix-caching`, `--max-model-len=8192`, `--max-num-seqs=64`，random tokens, `--ignore-eos`, default temperature.

**方法**: 每 case warmup 1 次 + measure 1 次 (取 measure 数字，跳变异)。

| Case | Input/Output | Concurrency | TTFT (ms) | TPOT (ms) | Output tok/s | Total tok/s | E2E (s) |
|---|---|---|---|---|---|---|---|
| **1** | 1K / 1K | **1** | **1142** | **48.95** | 19.99 | 39.98 | 51.2 |
| **2** | 1K / 1K | **32** | **2017** | **52.13** | **592** | **1184** | 55.3 |
| **3** | 1K / 7K | **1** | **689** | **51.59** | 19.35 | 22.12 | 370.4 |
| **4** | 1K / 7K | **32** | **2018** | **54.83** | **581** | **664** | 395.0 |

> **注**: 测 1K/8K 而非 1K/8K 因为 max_model_len=8192 上限 (1K + 7K = 8192)。

### 关键发现

**1. Batch 30x scaling**: batch=1 → batch=32 throughput 从 ~20 → ~590 tok/s，**接近 30x 完美 scaling**。说明 K2.6 MoE EP=16 expert dispatch 没有 large overhead。

**2. TPOT 稳定 ~50ms/token (decode 受 HBM bandwidth 限)**:
- batch=1, output=1K: 48.95 ms
- batch=32, output=1K: 52.13 ms (+6%)
- batch=1, output=7K: 51.59 ms (+5%)
- batch=32, output=7K: 54.83 ms (+12%)

K2.6 671B/A37B → 每 decode token 需 ~24 GB weight HBM read (37B 激活 × INT4 0.5 bytes + scale)。16 chips × ~3 TB/s/chip = 48 TB/s aggregate。**理论 TPOT = 0.5 ms** (HBM bandwidth bound)，实测 50 ms = **100x off**，说明 HBM 利用率 ~5-10% — 典型 TPU inference MFU 在 enforce_eager 下偏低。

**3. Long output 几乎不影响 TPOT**: batch=1 output 1K vs 7K 只差 +5% (49 → 51.6 ms)。MLA + FlashAttention 处理长 context 高效。

**4. TTFT scaling**: batch=32 prefill (32K total tokens) 仅比 batch=1 prefill (1K tokens) 慢 ~2x，说明 chunked prefill + EP=16 共享 expert weights 在 prefill 阶段 sub-linear scale。

### Caveats

- **Throughput numbers 还有大幅优化空间**: `--enforce-eager` 关了 XLA full graph compile，预期开了能再快 2-3x。开启需 cudagraph capture (cold start +5-10 min)。
- **Concurrency 测了 32**, max_num_seqs=64 还可以试更高。
- **没跑 GSM8K/MMLU 准确度评估**: 只有 smoke test coherent，未验证 SOTA 数字。

---

## 🚀 下一步 (Stage 4 候选)

- ⏳ **GSM8K / MMLU 标准评估** validate 数值正确性 (不只是 coherent 输出)
- ⏳ **关 enforce_eager 跑 XLA compile**: 预期 throughput 提升 2-3x，cold start +5-10 min
- ⏳ **Higher concurrency**: batch=64 / 128 (max_num_seqs cap)
- ⏳ **Long input**: 4K / 16K input (需先调 max_model_len)
- ⏳ **Per-rank cache filter**: 当前每 host stage 全 60 layer (~35s), 优化后 per-rank 只 stage own 24 expert subset = single host 35s → 2s
- ⏳ **Upstream PR** to vllm-project/tpu-inference (V18 cache + DNS fix)

---

## 📚 关键参考

- **完整复盘 HTML 文档**:
  - Stage 1 (4L sanity): https://cc.higcp.com/pages/kimi-k26-multihost-stage1-20260426.html
  - Stage 2 (61L 全 PASS, 57 min cold): https://cc.higcp.com/pages/kimi-k26-multihost-stage2-20260426.html
  - Stage 3 (SHM cache + DNS fix, 6 min cold): https://cc.higcp.com/pages/kimi-k26-multihost-stage3-20260427.html
- **GitHub branch**: https://github.com/yangwhale/tpu-inference/tree/chrisya/main
- **Build script**: [`scripts/build_k26_moe_cache.py`](https://github.com/yangwhale/tpu-inference/blob/chrisya/main/scripts/build_k26_moe_cache.py)
- **V18 cache hit code**: [`int4.py`](https://github.com/yangwhale/tpu-inference/blob/chrisya/main/tpu_inference/layers/jax/quantization/int4.py) (`process_weights_after_loading`)
- **Multi-host 详细 yaml + 早期 multi-host 实战**: 见 [`README-multinode.md`](README-multinode.md)
