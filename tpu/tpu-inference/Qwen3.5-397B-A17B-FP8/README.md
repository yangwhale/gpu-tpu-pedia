# Qwen3.5-397B-A17B-FP8 Inference on TPU v7x-8

> 端到端指南：在 TPU v7x-8（8 chips, single-host）上运行 Qwen3.5-397B-A17B-FP8 推理。
>
> **架构**：397B 总参 / 17B 激活 / **hybrid GDN+Attention**（45 GDN + 15 Standard Attn） + 512 routed experts + FP8 native
>
> **代码仓库**: [vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference)（main branch ≥ 2026-04-23, 含 PR #2366）
>
> **模型**: [Qwen/Qwen3.5-397B-A17B-FP8](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8)（94 safetensors, ~378 GiB）

---

## 🚨 已知关键限制

> 当前部署 **不适合 conversational chatbot**。
> - **Chat 路径 broken**: thinking OFF 输出语言错乱/死循环；thinking ON 解释/闲聊类问题 content 输出空 / `Thinking\n` 死循环
> - **唯一稳定路径**: 5-shot Q/A completion pattern + `enable_thinking:false`（GSM8K 93.93% 就是用这个）
> - **适合用例**: batch eval, structured generation, few-shot completion, code gen
> - ⚠️ 高 GSM8K accuracy ≠ chat ready — **不要被误导**

详见 [必读约束 D](#d-thinking-行为) 和 [验证步骤](#step-4-验证)。

---

## 🎯 关键性能（实测）

| 操作点 | 实测值 | 备注 |
|---|---|---|
| Cold start | **~7 min** | weight load + MoE re-quant + KV cache init |
| Single user latency (P1, 1K/1K) | **20.6 s, 49.7 tok/s/user** | 💨 Low Latency |
| Balanced (P64, 1K/1K) | **1508 tok/s, 23.5 tok/s/user** | ⚖️ Balanced |
| **🚀 Peak throughput (P128, 1K/1K)** | **2097 tok/s** ⭐ | Peak (不是 P256, 见附录) |
| **GSM8K full 1319 (5-shot, thinking OFF)** | **93.93% (1239/1319)** ✅ | length 截断仅 1.06% |
| 长 prompt 8K/1K P4 | **178.6 tok/s** | hybrid GDN 长 context 优势 |

完整 benchmark 数据见 [附录: Throughput sweep + GSM8K](#附录-完整-benchmark-数据)。

---

## ⚠️ 必读约束（4 项）

### A. PR #2366 patch（**必须**）

vLLM hybrid allocator 把 4 layers 共享 1 个 `KVCacheTensor`（GPU byte-level）。但 TPU `jax.Array` strongly typed 必须 duplicate per-layer → vLLM scheduler 的 block_id pool 比 TPU 实际容量大 ~3.5× → block_id 越界 → JAX `dynamic_update_slice_in_dim` silently clip → 多 request 状态塌陷 → **gibberish output / OOM / EngineCore crash**。

**修复**：从 main branch cp 修复版的 `kv_cache_manager.py` 到 pod。检查方式：
```bash
kubectl exec $POD -- grep -c '_hybrid_uniform_page_size_bytes' \
  /workspace/tpu_inference/tpu_inference/runner/kv_cache_manager.py
# 输出 7 = 已 patch；输出 0 = 需要 patch（见 Step 2）
```

### B. 三个必设环境变量

| 环境变量 | 值 | 作用 |
|---|---|---|
| `MODEL_IMPL_TYPE` | `vllm` | Qwen3.5 走 vLLM PyTorch + TorchAX path |
| `SKIP_JAX_PRECOMPILE` | `1` | 跳过 JAX 预编译，启动快 1-2 min |
| `VLLM_XLA_CHECK_RECOMPILATION` | `0` | 关闭 XLA 重编译检查 |

### C. 关键启动参数

| 参数 | 取值 | 作用 |
|---|---|---|
| `--enable-expert-parallel` | 必须 | EP=8 是 PR #2366 fix 后的正确并行策略 |
| `--no-enable-prefix-caching` | 必须 | 否则触发 `chunked_mm_input` AssertionError |
| `--reasoning-parser` | `qwen3` | 正确解析 `<think>` tag |
| `--block-size` | `256` | CI 默认值 |
| `--kv-cache-dtype` | `fp8` | KV 减半 |
| `--gpu-memory-utilization` | `0.9` (单机) / `0.7` (PD prefill) / `0.9` (PD decode) | |
| `--tensor-parallel-size` | `8` | TP=8 (8 chips) |
| `--max-model-len` | `4096` (单机) / `16384` (PD long prompt) | |
| `--max-num-batched-tokens` | `4096` | CI accuracy test 默认 |
| `--max-num-seqs` | `256` | CI 默认 |
| `--limit-mm-per-prompt` | `'{"image":0,"video":0}'` | 跳过 vision encoder |
| `--async-scheduling` | 推荐 | 异步调度 |

### D. Thinking 行为

Qwen3.5 默认 thinking ON（输出 `<think>...</think>` reasoning + 答案）。**所有 server-side 关 thinking 的方法都不稳定**：

| 尝试 | 实测 |
|---|---|
| `--chat-template-kwargs='{"enable_thinking":false}'` (启动 flag) | silently 忽略 |
| Request body `chat_template_kwargs={"enable_thinking":false}` + 普通 chat prompt | chat 端模型陷入 `</think>` 死循环 → gibberish |
| **Request body 同上 + 单 user message 含 5-shot Q/A pattern** | ✅ **work**, reasoning_len=0 |
| User prompt 加 `/no_think` 标记 | 失效 |

**生产 workaround**（按可靠性）：
1. ⭐ Chat + 5-shot Q/A pattern + `enable_thinking:false`（**唯一稳定**, GSM8K 实测 93.93%）
2. `/v1/completions` raw prompt — 不稳定，不推荐
3. 接受 thinking ON + `max_tokens` ≥ 3500 — 长答案场景

---

## 🧭 部署模式选择

| 模式 | TPU | 适合场景 | 启动时间 | Patches | 复杂度 |
|---|---|---|---|---|---|
| **1. 单机** | 1 × v7x-8 (8 chips, TP=8) | low-latency / batch eval / GSM8K / **大多数生产用例** ⭐ | ~7 min | 1 (PR #2366) | 低 |
| **2. PD 分离 (1P1D)** | 2 × v7x-8 (TP=8 P + TP=8 D) | 长 prompt 高吞吐 / **TPOT 优化** / 大并发 | ~10 min | 2 (PR #2366 + HMA) | 中 |
| **3. Multi-host TP=16** | 2 × v7x-8 (16 chips, TP=16, LWS) | **超大模型 (>500B) / 验证 multi-host capability** | ~11-30 min | 3 (PR #2366 + 2 mrope bypass) | 高 |

**选择路径**：
- ✅ 不确定？先用**模式 1** — 覆盖 90% 用例，cold start 最快，patches 最少
- 🚀 需要更高 P95 throughput / 长 prompt 优化？升 **模式 2 PD**
- 🔬 验证 multi-host inference engine 能力 / 想跑 >500B 模型？用 **模式 3**（397B 在单机够用，主要是 capability proof）

---

# 部署模式 1：单机 vLLM Serve

> 单 v7x-8 pod 跑 Qwen3.5-397B vLLM serve。适合低延迟 / 中小并发 / batch eval / GSM8K-style 任务。

### Step 1: 准备 Pod + 模型

需要 GKE TPU pod (`tpu-v7x-lite-podslice`, 2x2x1, 8 chips) + 共享存储 (**Lustre RWX 推荐**) + 模型权重。

> 💡 **为什么 Lustre 不是 PVC**: Lustre 下载 ≈ 63 GB/min vs PVC ≈ 16 GB/min（实测 Qwen3 ≈ 4× 差距）。378 GiB 权重 Lustre 用 ~6 min，PVC 要 ~24 min。多 pod 并发训推也只有 Lustre RWX 支持。

```bash
# 设置 GKE context (你的 cluster)
CTX=<your-gke-context>
POD=<your-tpu-pod-name>
MODEL=/lustre/models/Qwen3.5-397B-A17B-FP8

# 验证 pod ready
kubectl --context="$CTX" get pods | grep $POD       # 应 Running 2/2

# 下载权重到 Lustre（如未下，~6 min, 16 worker + hf_transfer 加速）
kubectl exec $POD -- bash -c "
  mkdir -p /lustre/models/Qwen3.5-397B-A17B-FP8
  pip install -U 'huggingface_hub[hf_transfer]'
  HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
    Qwen/Qwen3.5-397B-A17B-FP8 \
    --local-dir /lustre/models/Qwen3.5-397B-A17B-FP8
"

# 验证 94 shards 完整
kubectl exec $POD -- bash -c "ls $MODEL/*.safetensors | wc -l"   # 应输出 94

# 清理 /dev/shm 残留（避免 weight load 退化 50×）
kubectl exec $POD -- bash -c "
  ls -la /dev/shm/                # 看占用
  # 删别人的 staging 前 fuser -v /dev/shm/<dir> 确认没人占
  rm -rf /dev/shm/sem.* /dev/shm/wrk_* 2>/dev/null
"
```

### Step 2: 应用 PR #2366 patch（如未 patch）

```bash
# 检查
kubectl exec $POD -- grep -c '_hybrid_uniform_page_size_bytes' \
  /workspace/tpu_inference/tpu_inference/runner/kv_cache_manager.py
# 输出 7 = 已 patch, 跳过；输出 0 = 走下面 patch
```

```bash
# Host 端
TMP=$(mktemp /tmp/kv_cache_manager.XXXXXX.py)
curl -sf https://raw.githubusercontent.com/vllm-project/tpu-inference/main/tpu_inference/runner/kv_cache_manager.py -o $TMP
grep -c '_hybrid_uniform_page_size_bytes' $TMP   # 应输出 7

KCM=/workspace/tpu_inference/tpu_inference/runner/kv_cache_manager.py
kubectl --context="$CTX" exec $POD -- cp $KCM ${KCM}.bak
kubectl --context="$CTX" cp $TMP $POD:$KCM
kubectl --context="$CTX" exec $POD -- bash -c "
  grep -c '_hybrid_uniform_page_size_bytes' $KCM   # verify 7
  find /workspace/tpu_inference -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
"
rm -f $TMP
```

### Step 3: 启动 vLLM（file-based launcher）

⚠️ **必须用 file-based launcher**：`kubectl exec $POD -- bash -c "<multi-line nohup>"` 会被 SIGKILL=137（kubectl exec 的 stdin channel 关闭时杀进程组，nohup/setsid/disown 都救不回来）。File-based launcher 让 bash 读完文件后干净 fork+exit。

```bash
# 1. 写 launcher 到本地
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

# 2. cp 到 pod 并执行
kubectl --context="$CTX" cp /tmp/launch_vllm.sh $POD:/tmp/launch_vllm.sh
kubectl --context="$CTX" exec $POD -- bash /tmp/launch_vllm.sh

# 3. 等 cold start (~7 min) + 监视
kubectl --context="$CTX" exec $POD -- tail -f /tmp/vllm_qwen35.log
```

**等待看到关键 log（PR #2366 + 启动完成的标志）**：
```
Hybrid KV cache: padding every layer spec to 23289856 bytes ...   ← PR #2366 padding
regular_attn_shape=(num_blocks, (1280, 8, 4, 256))                ← block_size 1280 (patch 前是错的 4352)
num_gpu_blocks_override=945
INFO: Application startup complete.
```

### Step 4: 验证（5-shot hello world）

```bash
# 健康检查
kubectl exec $POD -- curl -sf -o /dev/null -w "%{http_code}\n" http://localhost:8000/health
# 应输出 200

# Hello world — chat 端 5-shot 单 user message（chat 端唯一稳定方式, 见 Constraint D）
SHOTS="Question: Capital of Japan?\nAnswer: Tokyo.\n\nQuestion: Capital of Germany?\nAnswer: Berlin.\n\nQuestion: Capital of Italy?\nAnswer: Rome.\n\nQuestion: Capital of Spain?\nAnswer: Madrid.\n\nQuestion: Capital of Brazil?\nAnswer: Brasilia.\n\n"
P="${SHOTS}Question: Capital of France?\nAnswer:"
kubectl exec $POD -- curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$P\"}],\"max_tokens\":50,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
  | python3 -c 'import sys,json; r=json.load(sys.stdin); m=r["choices"][0]["message"]; print("content:", repr(m["content"])); print("reasoning_len:", len(m.get("reasoning") or "")); print("finish:", r["choices"][0]["finish_reason"])'
# 期望: content: ' Paris.'  reasoning_len: 0  finish: stop
```

**预期**：`content: 'Paris.'` / `reasoning_len: 0` / `finish: stop`

---

# 部署模式 2：PD 分离 (1P1D)

> 1 prefill (kv_producer) + 1 decode (kv_consumer) + proxy 三 pod 部署。适合高并发 batch / TPOT 敏感场景 / 长 prompt RAG。

### 🚨 Qwen3.5 PD 必读差异（vs Qwen3-Coder PD）

| 项 | Qwen3-Coder (pure attention) | **Qwen3.5 (hybrid GDN+Attn)** |
|---|---|---|
| `kv_connector` | `TPUConnector` | **`TPUConnectorHMA`** ⭐ |
| `kv_connector_module_path` | `tpu_inference.distributed.tpu_connector` | **`tpu_inference.distributed.tpu_connector_hma`** ⭐ |
| Hybrid KV cache manager flag | 不需要 | **必须** `--no-disable-hybrid-kv-cache-manager` ⭐ |
| Image 内是否含 HMA | nightly 有 TPUConnector | **没有 HMA**，需 cp from main |

**为什么必须 `--no-disable-hybrid-kv-cache-manager`**：vLLM core 看到 `kv_transfer_config` 时**默认 disable hybrid KV cache manager**，强制 `unify_hybrid_kv_cache_specs`。但 Qwen3.5 60 layer (45 GDN + 15 Attn) 无法 unify → `ValueError: Hybrid KV cache manager is disabled but failed to convert the KV cache specs to one unified type` → EngineCore 崩。这个 flag = 显式告诉 vllm "我的 connector 是 SupportsHMA, handle hybrid 妥当"。

### Step 1: 创建 2 个 v7x-8 spot node pool

```bash
PROJECT=<your-gcp-project>
CLUSTER=<your-gke-cluster>
REGION=<your-region>      # 例: us-central1
ZONE=<your-zone>          # 例: us-central1-c

for role in p d; do
  gcloud container node-pools create np-tpu7x-spot-pd-$role \
    --cluster=$CLUSTER --region=$REGION --project=$PROJECT \
    --node-locations=$ZONE --machine-type=tpu7x-standard-4t --num-nodes=1 --spot \
    --disk-type=hyperdisk-balanced --disk-size=500 \
    --node-taints=google.com/tpu=present:NoSchedule \
    --workload-metadata=GKE_METADATA --enable-autorepair --enable-autoupgrade --async
done
# spot 容量充裕的 region 通常 2-5 min ready
```

### Step 2: Stage HMA + PR #2366 patch 到 Lustre

```bash
# 把 scripts/tpu_connector_hma.py（本仓库已 cp from main）传到 Lustre
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

### Step 3: 部署 P + D + Proxy（3 个 manifest 已 commit 到 `manifests/`）

```bash
cd manifests/
kubectl --context="$CTX" apply -f qwen35_prefill.yaml -f qwen35_decode.yaml -f qwen35_proxy.yaml
```

3 个 manifest 自带 init container：每次 pod 启动从 Lustre `/lustre/patches/qwen35-pd/` cp HMA + PR #2366 文件到 `/workspace/tpu_inference/`，删 `__pycache__`，verify grep。

### Step 4: 等就绪 + 验证 HMA log

```bash
# ~10 min cold start
kubectl --context="$CTX" wait --for=condition=Ready pod -l app=vllm-prefill --timeout=1200s
kubectl --context="$CTX" wait --for=condition=Ready pod -l app=vllm-decode --timeout=1200s

# 验证 HMA + PR #2366 + Hybrid KV manager 全 enable
PREFILL_POD=$(kubectl --context="$CTX" get pods -l app=vllm-prefill -o jsonpath='{.items[0].metadata.name}')
kubectl --context="$CTX" logs $PREFILL_POD | grep -E "TPUConnectorHMA|Hybrid KV cache: padding|Application startup"
```

**期望看到**：
```
Hybrid KV cache: padding every layer spec to 23289856 bytes               ← PR #2366 ✓
Hybrid KV cache layout: num_kv_cache_groups=4, ... duplicate_shared_layers=True
TPUConnectorHMA Worker 0 Prefill --> init | num_layers=60 | num_kv_groups=4 | group_is_mamba=[True, True, True, False]
Creating v1 connector with name: TPUConnectorHMA                          ← HMA registered ✓
Application startup complete
```

### Step 5: Smoke Test（验证 P→D KV 传输工作）

```bash
PROXY_POD=$(kubectl --context="$CTX" get pods -l app=vllm-proxy -o jsonpath='{.items[0].metadata.name}')

# 5-shot examples (单 user message, 跟单机/multi-host Step 4/5 同 pattern)
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

**预期**：5/5 国家全 hit, 全 finish=stop, reasoning_len=0
```
France:    'Answer: Paris.' | stop
Italy:     ' Rome.'         | stop
Australia: ' Canberra.'     | stop
Canada:    ' Ottawa.'       | stop
Brazil:    ' Brasilia.'     | stop
```

---

# 部署模式 3：Multi-host TP=16 (2 × v7x-8) ⭐ 实测 2026-04-26 ⭐ LM 推理 5/5 PASS

> 1 LWS 跨 2 个 v7x-8 节点共 16 chips, TP=16 + Ray distributed executor。**证明 vLLM TPU 推理引擎支持 hybrid 模型 multi-host 端到端 LM 推理**。
>
> ✅ **当前状态**: 启动 ✅ + HTTP 200 ✅ + **5/5 国家首都 chat completions 全 hit** (Paris/Rome/Canberra/Ottawa/Brasilia, finish=stop, reasoning_len=0)。

### 🚨 Multi-host vs Single-host 关键差异（6 层 root cause / fix）

| # | Multi-host 特定 fix | 不修后果 |
|---|---|---|
| 1 | **`--max-num-batched-tokens=16384`** (≥ Qwen3.5 `max_tokens_per_mm_item`) | silent hang in init_device, 14 min 没新 log, worker SIGSEGV |
| 2 | **PR #2366 patch (kv_cache_manager.py)** + **多 group block_tables_cpu rebuild** | KV init `AssertionError: page_size_padded >= page_size` 或 `IndexError: list index out of range` |
| 3 | **`TPU_MULTIHOST_BACKEND=ray`** + `TPU_TOPOLOGY=2x2x2` + `TPU_HOST_BOUNDS=1,1,2` env override | 单机模式启动, TP=16 无法跨 host |
| 4 | **`--distributed-executor-backend=ray`** + LWS pattern (1 leader + 1 worker) | UniProcExecutor 无法跨 pod |
| 5 | **`tpu_runner.py` patch (5 行)**: `disable_mm_from_limits=True` 时 set `self.uses_mrope=False` + `self.get_mrope_input_positions_fn=None` | chat 端 first request 崩 `TypeError: Qwen3VL.get_mrope_input_positions() got unexpected hf_config` (vllm core vs Qwen3VL model class API 错位) |
| 6 | **`persistent_batch_manager.py` patch (4 行)**: defensive None check `if get_mrope_input_positions_fn is None: continue` | PersistentBatchManager 在 init 时已捕获 uses_mrope=True, 后期 set False 不影响 → call None → TypeError |

> ⭐ **3 个 patches commit 到 repo** ([scripts/multihost-patches/](scripts/multihost-patches/))，任何人 checkout 后可直接 deploy:
> - `kv_cache_manager.py` — PR #2366 hybrid padding + 多 group block_tables_cpu rebuild
> - `tpu_runner.py` — disable_mm_from_limits 时禁 mrope (5 行)
> - `persistent_batch_manager.py` — defensive None check (4 行)

### Step 0: 集中设置环境变量（一次设置，Step 1-7 全用）

```bash
# GKE
CTX=<your-gke-context>             # 例: gke_PROJECT_REGION_CLUSTER (kubectl context)
PROJECT=<your-gcp-project>
CLUSTER=<your-gke-cluster>
REGION=<your-region>               # 例: us-central1
ZONE=<your-zone>                   # 例: us-central1-c

# Lustre patch staging
UTIL_POD=<your-pod-with-lustre-mount>   # 任何挂 lustre-pvc 的 pod (Step 2 用)

# Repo (本仓库 root)
REPO_ROOT=<your-checkout-of-gpu-tpu-pedia>   # 例: ~/gpu-tpu-pedia
cd $REPO_ROOT/tpu/tpu-inference/Qwen3.5-397B-A17B-FP8/
```

### Step 1: 准备 multi-host node pool

```bash
# 创建 2 节点 multi-host TPU pool (注意 --num-nodes=2 + --tpu-topology=2x2x2)
gcloud container node-pools create np-tpu7x-spot-mh-qwen35 \
  --cluster=$CLUSTER --region=$REGION --project=$PROJECT \
  --node-locations=$ZONE --machine-type=tpu7x-standard-4t --num-nodes=2 \
  --tpu-topology=2x2x2 --spot \
  --disk-type=hyperdisk-balanced --disk-size=500 \
  --node-taints=google.com/tpu=present:NoSchedule \
  --workload-metadata=GKE_METADATA --enable-autorepair --enable-autoupgrade
# Spot 容量充裕的 region 通常 2-5 min ready
```

> **Spot 容量不够 fallback** (报 `RESOURCE_EXHAUSTED` 或 quota error 时):
> - 换 zone (us-central1-a/b/c 之一), 或换 region (us-east5 / us-west4 / asia-northeast1)
> - 改 `--spot` → `--reservation-affinity=specific --reservation=<your-reservation>`(on-demand, 贵但稳定)
> - 见 [feedback_capacity-check](capacity-check skill) 实时查容量

### Step 2: Stage **3 个 patches** 到 Lustre（multi-host 必须）

```bash
# 一次性 cp 3 个 patches 到 Lustre (multi-host 比 PD 多 2 个 patches)
# ⚠️ 假设你已 cd 到 Step 0 的 model dir (relative path scripts/multihost-patches/ 才工作)
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

> **3 个 patches 在 yaml 的 init container 自动 cp 到 `/workspace/tpu_inference/`**, 不需要每次手动 cp 进 pod。

### Step 3: 部署 LWS

```bash
cd manifests/
# 如果之前部署过, 先 delete (LWS 不能 in-place update yaml 修改)
kubectl --context="$CTX" delete lws qwen35-mh --ignore-not-found --wait=false
sleep 30   # 等 pod terminate

kubectl --context="$CTX" apply -f qwen35_multihost.yaml
# LWS 1 group × 2 pods (1 leader on node A + 1 worker on node B)
```

> 💡 **可忽略 warning**: 如果 Service `qwen35-mh` 之前部署过, 你会看到 `Warning: resource services/qwen35-mh is missing the kubectl.kubernetes.io/last-applied-configuration annotation`。这是 kubectl 的常规 warning, 不影响部署。
>
> 💡 **更新 patches 后 redeploy**: 改 patches 后必须 (1) 重跑 [Step 2 cp 到 Lustre](#step-2-stage-3-个-patches-到-lustremulti-host-必须), (2) 重 apply LWS (init container 启动时从 Lustre cp 新 patches 进 pod)。LWS 不能 in-place hot-reload patches。

### Step 4: 等就绪 + 验证 multi-host 启动 log（**实测 11-30 min**）

⚠️ Multi-host cold start **比单机慢 2-3×**: weight load 多机分摊 Lustre 带宽 + cross-host JAX init 是同步等所有 host 完成。
- **顺利情况**：~11 min, RESTARTS=0 (实测多次)
- **触发 sentinel race**: ~25-30 min, 含 K8s auto restart 1-2 次（race 是已知偶发, 不影响最终成功）

**不要用 `kubectl wait condition=Ready`**——pod restart 期间会 false success。改用 health endpoint 轮询:

```bash
# 轮询 health 直到 200 (最长 30 min)
for i in $(seq 1 60); do
  CODE=$(kubectl --context="$CTX" exec qwen35-mh-0 -- curl -sf -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null)
  CODE=${CODE:-000}   # 默认 000 (避免 curl fail 时 -w 不输出)
  echo "T+$((i*30))s: HTTP $CODE"
  [ "$CODE" = "200" ] && break
  sleep 30
done

# 验证 multi-host 关键 init log (用 awk 每行截到 150 字符避免 device 列表淹没)
kubectl --context="$CTX" logs qwen35-mh-0 | grep -E "Init worker|Hybrid KV cache:|Application startup" | awk '{print substr($0,1,150)}' | head -10
```

**期望关键 line（multi-host 特征）**：
- `Init worker | rank=0 | hbm=[(0.0, 94.75), ...] × 16` ← **16 chips** (单机是 8)
- `Hybrid KV cache: padding every layer spec to 13328384 bytes` ← multi-host PR #2366 padding (单机是 23289856, 不同因 TP=16 切分更细)
- `Hybrid KV cache layout: num_kv_cache_groups=4, ... num_blocks=5299` ← TP=16 num_blocks 比单机 945 大 5.6×
- `regular_attn_sharding=Mesh('data':1, 'model':16)` ← **TP=16 mesh** ✅
- `Application startup complete` ← ✅ ready 信号

### Step 5: Smoke Test（验证 multi-host LM 推理 work）

```bash
# 5-shot examples (单 user message, hit thinking-OFF 路径; 见 必读约束 D)
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

**预期** (实测 2026-04-26 全 pass, dogfood 复测 2 次):
```
France:    'Paris.'    | stop
Italy:     'Rome.'     | stop
Australia: 'Canberra.' | stop
Canada:    'Ottawa.'   | stop
Brazil:    'Brasilia.' | stop
```

### Step 6: Troubleshooting (multi-host 特定)

**Symptom 1: Step 4 health 30 min 还没 200**
- 看 restart 次数: `kubectl --context="$CTX" get pods -l leaderworkerset.sigs.k8s.io/name=qwen35-mh`
  - RESTARTS≥2 → `kubectl logs qwen35-mh-0 --previous | grep -E 'ERROR|Traceback' | tail -30` 找 root cause（大概率是 [6 层 root cause](#-multi-host-vs-single-host-关键差异6-层-root-cause--fix) 之一，确认 patches 已 stage + verify counter ≥ expect 值）
- Worker actor SIGSEGV: `kubectl logs qwen35-mh-0-1 --previous | tail -50`
- Spot preemption: `kubectl describe pod qwen35-mh-0 | grep Warning`

**Symptom 2: Step 5 部分国家 fail / 输出乱码**
- 输出 `</think>` 死循环 → 5-shot pattern 没 hit, 检查 prompt 是否单 user message (不是多个 messages)
- `KeyError: 'choices'` → vllm 内部崩, container 重启了, 等 health 200 再试
- `TypeError: ... mrope ...` → mrope patch 没生效, verify: `kubectl logs qwen35-mh-0 | grep 'mrope patch'` 应输出 `(expect 1)`

**Symptom 3: LWS pod 一直 Pending 不调度**
- TPU 节点全占。`kubectl get nodes -l cloud.google.com/gke-tpu-accelerator=tpu7x` 看占用; 别人在用就等或新建 node pool

**Symptom 4: `kubectl exec` 报 `setns process: exit status 1`**
- Container 正在 restart 中，几秒后再试。常见于 vllm crash 后窗口期

### Step 7: Production 接入 / Teardown

**Smoke test pass 后, 接 production traffic**:

```bash
# Service 已经在 yaml 定义好 (clusterIP, port 8000)
kubectl --context="$CTX" get svc qwen35-mh   # 看 ClusterIP

# 集群内访问: 任何 pod 用 service DNS 调 inference
# http://qwen35-mh.default.svc.cluster.local:8000/v1/chat/completions

# 集群外访问 (生产场景, 走 GKE Ingress 或 Gateway):
kubectl --context="$CTX" expose deployment vllm-disagg-proxy ... # 略, 标准 GKE 流程
# 或者 port-forward 调试: kubectl port-forward svc/qwen35-mh 8000:8000
```

**Teardown (停止节省 cost)**:

```bash
# 1. 删 LWS (释放 TPU 节点上的 pod, 但 spot node pool 仍在跑)
kubectl --context="$CTX" delete lws qwen35-mh

# 2. (可选) 删 multi-host node pool (彻底 0 cost; spot 即使空闲也按时长计费)
gcloud container node-pools delete np-tpu7x-spot-mh-qwen35 \
  --cluster=$CLUSTER --region=$REGION --project=$PROJECT --quiet
```

> 💡 **Lustre patches 不需要清理**: `/lustre/patches/qwen35-pd/*` 占 < 200 KB, 多次 redeploy 复用即可。

### Multi-host 部署完整 dogfood 详记

完整 8 次 test iteration + 6 层 root cause 链 + 3 patches deep dive + 经验教训见**内部 dogfood HTML**（[最下方附录](#-内部文档dogfood-历程--深度分析)）。

---

# Benchmark

### Throughput (单实例, evalscope)

```bash
# 在 pod 内
pip install -q evalscope[perf]
bash scripts/run_bench_qwen35.sh    # 默认 P1/P4/P16/P64/P256 5 档 × 2 round (~24 min)
# 输出 /tmp/bench_qwen35/summary.txt
```

完整 17 档 sweep 数据见[附录](#附录-完整-benchmark-数据)。

### GSM8K Accuracy（5-shot, thinking OFF via in-context, ~15 min）

```bash
# cp script 到 pod
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

**实测**：1319 题 ~15 min, **93.93% accuracy** (1239/1319), length 截断仅 14 (1.06%)。CI 阈值 63%。

> 监控提示：脚本 stdout 有 buffer，看实时进度用 `wc -l /tmp/gsm8k_full.jsonl`（每完成一题立即写入）。

### PD 分离 Benchmark（待补，参考 Qwen3-Coder §7g）

```bash
PROXY_POD=$(kubectl get pods -l app=vllm-proxy -o jsonpath='{.items[0].metadata.name}')
kubectl exec $PROXY_POD -- vllm bench serve \
  --model=Qwen3.5-397B-FP8 --dataset-name=random \
  --num-warmups 10 --random-input-len=1024 --random-output-len=1024 \
  --num-prompts=256 --ignore-eos \
  --host=localhost --port=10000 --max-concurrency=1 \
  --result-file=disagg_qwen35_1024_1024_c1.json
# 同样跑 8192/1024 c=4 长 prompt 场景
```

---

## Troubleshooting

| 症状 | 根因 | 修复 |
|---|---|---|
| **多并发输出乱码 / OOM `vmem 86MB > 64MB` / `HBM 95G > 94.75G` / EngineCore silent crash** | 缺 PR #2366 (KV cache 状态损坏) | 走 [Step 2](#step-2-应用-pr-2366-patch如未-patch)，grep 应输出 7 |
| **weight load 80s/shard (vs 正常 2s)，启动从 7min 变 2hr** | `/dev/shm` 残留 RAM 不足 → vLLM 跳过 auto-prefetch | 清 `/dev/shm`，启动加 `--safetensors-load-strategy=prefetch` |
| **`ABORTED: libtpu lockfile` / `TPU device busy`** | 上次 vLLM 异常退出，孤儿进程占 `/dev/vfio/0` | `pgrep -f 'EngineCore\|vllm' \| xargs -r kill -9` + `rm -f /tmp/libtpu_lockfile` |
| **`kubectl exec ... bash -c "<multi-line nohup>"` 返回 exit 137** | kubectl exec stdin 关闭杀进程组，nohup 救不回 | 用 file-based launcher（[Step 3](#step-3-启动-vllmfile-based-launcher)） |
| **PD 模式: `ValueError: Hybrid KV cache manager is disabled but failed to convert KV cache specs to one unified type`** | vLLM 默认 disable HMA when `kv_transfer_config` set | 加 `--no-disable-hybrid-kv-cache-manager` flag（[PD 模式必读差异](#-qwen35-pd-必读差异vs-qwen3-coder-pd)） |

---

## 附录

### 完整 Benchmark 数据

#### Throughput sweep (1K/1K, evalscope perf, warmup + record)

| Batch | Latency | Throughput | Per-user | Pareto |
|---:|---:|---:|---:|---|
| P1 | 20.6 s | 49.6 tok/s | 49.6 | 💨 Low Latency |
| P2 | 21.2 s | 96.7 | 48.4 | |
| P4 | 21.9 s | 186.8 | 46.7 | 交互对话 |
| P8 | 22.8 s | 358.7 | 44.8 | |
| P16 | 25.6 s | 640 | 40.0 | 在线服务 |
| P32 | 28.9 s | 1129 | 35.3 | |
| P64 | 43.2 s | 1510 | 23.6 | ⚖️ Balanced |
| **P128** | 61.8 s | **2103 tok/s** ⭐ | 16.4 | 🚀 Peak |
| P256 | 108.4 s | 1877 ↓ | 7.3 | (超 sweet spot) |

**Pareto 操作点**：
- 单用户 / TPOT < 25 ms → **P1** (49.6 tok/s/user)
- 中等并发 → **P32-P64** (1100-1500 tok/s 总, 23-35 tok/s/user)
- 离线批 / 最大吞吐 → **P128** (**2097 tok/s** peak, per-chip 262 tok/s)

#### Thinking ON vs OFF (1K/1K) — raw throughput 几乎一样

| Batch | OFF | ON |
|---:|---:|---:|
| P1 | 49.6 | 49.6 |
| P16 | 640 | 638 |
| P64 | 1510 | 1518 |

⚠️ 业务有效 token 差 10×：thinking ON 同时间 ~90% reasoning + 10% 答案；OFF 大部分是答案。能关 thinking 时业务效率提升 10×（关法见 [Constraint D](#d-thinking-行为)）。

#### 长 context (vLLM `--max-model-len 16384`)

**8K input / 1K output (prefill heavy)**：

| Batch | Throughput | vs 1K input |
|---:|---:|---|
| P1 | 51.7 tok/s | **+4%** (低并发不拖累) |
| P4 | 178.6 | -4% |
| P16 | 499.1 | -22% |
| P64 | 849.9 | -44% |

**1K input / 8K output (decode heavy)**：

| Batch | Throughput | vs 1K out |
|---:|---:|---|
| P1 | 54.0 | **+9%** |
| P4 | 203.3 | **+9%** |
| P16 | 711.0 | **+11%** |
| **P64** | **1702 tok/s** | **+13%** ⭐ |

🎯 **长 generation 在所有 batch size 都比短 generation 快 9-13%**（pure decode 让 TPU MXU 持续高利用率）。**长输出场景 (文章/代码生成) 用 P64 1702 tok/s 是 v7x-8 甜蜜点**。

### 硬件门槛

| 项 | 要求 |
|---|---|
| TPU | v7x-8（8 chips, 不需要 v7x-16）|
| HBM | 94.75 GB/device × 8 = 758 GB（per-device 用 ~85 GB / 90% util）|
| 主机内存 | ≥ 800 GB（page cache 装 378 GB checkpoint）|
| 存储 | ≥ 600 GB（模型 378 GB + 工作空间）|

### 模型核心参数

| 字段 | 值 |
|---|---|
| 架构 | MoE + **Hybrid GDN/Attention** (60 层 = 45 GDN + 15 Standard Attn) |
| 参数 | 397B 总 / 17B 激活 / 512 routed + 1 shared expert / Top-K=10 |
| 维度 | Hidden 4096, Attn 32 Q + 2 KV (GQA), head_dim 256, Expert intermediate 1024 |
| 上下文 | Native 262K, YaRN 可扩 1M |
| 量化 | FP8 native, vocab 248,320 |

### 参考

- [PR #2366 — Hybrid KV cache OOB fix（必备）](https://github.com/vllm-project/tpu-inference/pull/2366)
- [PR #2322 / #2327 / #2331 / #2336 — PD disagg 4 PR 系列（含 HMA 实现）](https://github.com/vllm-project/tpu-inference/pull/2322)
- [tpu_connector_hma.py — TPUConnectorHMA 源码](https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/distributed/tpu_connector_hma.py)
- [Qwen3.5-397B-A17B-FP8 HuggingFace](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8)
- 同体系 README: [DeepSeek R1 FP4](../DeepSeek-R1-671B-FP4/README.md) · [GLM-5.1 FP4](../GLM-5.1-754B-FP4/README.md) · [Kimi K2.6](../Kimi-K2.6/README.md) · [Qwen3-Coder 480B](../Qwen3-Coder-480B/README.md)

### 📚 内部文档（dogfood 历程 + 深度分析）

按时间顺序，4 个互补的 HTML 文档记录了从初次部署到 PD 分离的完整 dogfood：

| 日期 | 主题 | 详情 |
|---|---|---|
| 2026-04-24 | [部署与优化指南 v1.5](https://cc.higcp.com/pages/qwen35-397b-tpu-inference-plan-20260424.html) | 完整部署 + 优化决策（108 KB）|
| 2026-04-25 | [⭐ 单机推理踩坑记 — 4 小时弯路 vs 14 分钟正确路](https://cc.higcp.com/pages/qwen35-397b-debug-story-20260425.html) | 单机部署踩坑全记录（45 KB）|
| 2026-04-26 | [README 可复现性验证报告](https://cc.higcp.com/pages/qwen35-readme-verification-20260426.html) | 按 README 步骤盲跑 + 验证（23 KB）|
| 2026-04-26 | [⭐ PD 分离部署 Dogfood 详记](https://cc.higcp.com/pages/qwen35-pd-disagg-dogfood-20260426.html) | PD 全流程 + HMA root cause + 6 lessons（24 KB）|
| 2026-04-26 | [⭐ Multi-host TP=16 Dogfood 详记](https://cc.higcp.com/pages/qwen35-multihost-dogfood-20260426.html) | 5 次 test 揭露 4 层 root cause 链 + 单机/多机对比（27 KB）|

> 💡 **链接行为**：cc.higcp.com 走 GCP IAP，浏览器有 google account 登录就能直接看；外部访问可用 `https://storage.googleapis.com/chris-pgp-host-asia/cc-pages/pages/<file>.html` 直链。
