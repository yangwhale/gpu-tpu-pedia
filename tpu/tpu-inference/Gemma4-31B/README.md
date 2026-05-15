# Gemma4-31B Inference on TPU v7xe

> 🌐 **Languages** | **语言**: **中文** · [English](README.en.md)

> 端到端复现指南：在 TPU v7xe 上用 vLLM 跑 Gemma4-31B 推理，**全 256K context 已验证通过**。
>
> **架构**: 30.7B Dense / 60 layers / hybrid sliding-window + global attention / **256K** context / 262K vocab / 多模态（text + image）
>
> **后端**: [vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference) (JAX backend `gemma4.py`)
>
> **模型**: [google/gemma-4-31b-it](https://huggingface.co/google/gemma-4-31b-it) (BF16, ~61 GiB)

---

## 🎯 核心结论

| 指标 | 数值 | 测试条件 |
|------|------|---------|
| **Peak Throughput** | **6,144 tok/s** | 1K/1K, P=256, TP=4 |
| **Single-User TPOT** | **35 ms** | 全 context 长度稳定（1K → 256K） |
| **Full 256K Context TTFT** | **695 ms** | 单用户 256K 输入 |
| **128K Context TTFT** | **378 ms** | 单用户 128K 输入 |
| **Cold Start** | **~3-5 min** | TP=4，权重已在 Lustre |

> 💡 **生产就绪**：本指南的所有 16 项 benchmark 全部通过，含 1K → 256K 的完整 context 范围。
> Decode 延迟（TPOT）在所有场景下保持 34-49ms，**与 context 长度无关**。

---

## 📋 前置要求

| 项目 | 要求 |
|------|------|
| GKE 集群 | 支持 TPU v7（Ironwood） |
| 共享存储 | Lustre / GCS / Filestore PVC（用于权重和补丁） |
| HuggingFace Token | 已接受 [Gemma 4 license](https://huggingface.co/google/gemma-4-31b-it) |
| `gcloud` / `kubectl` | 已配置 GKE context |

---

## ⚡ Quick Start（老手 5 步复现）

假设 GKE 集群和 v7xe Pod 已就绪、模型权重已下载、kubectl context 已设置。

```bash
CTX=<your-gke-context>; POD=gemma4-31b; MODEL=/lustre/models/gemma-4-31b-it

# 1. 验证模型权重
kubectl --context=$CTX exec $POD -- bash -c "ls $MODEL/*.safetensors | wc -l"

# 2. 应用必需 kernel patch (prefill_batch_size 2→1, 解决长 context VMEM 溢出)
kubectl --context=$CTX exec $POD -- bash -c '
WRAPPER=/workspace/tpu_inference/tpu_inference/kernels/experimental/batched_rpa/wrapper.py
sed -i "s/    prefill_batch_size = 2$/    prefill_batch_size = 1/" $WRAPPER
grep "prefill_batch_size = 1" $WRAPPER && echo "Patch applied"'

# 3. 写 launcher（默认 256K full context, TP=4）
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

# 4. cp + 启动 + 等就绪 (~3-5 min)
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
# 期望: Paris
```

---

# 端到端部署步骤

## Step 0: 环境变量

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

## Step 1: 创建 TPU v7xe Spot Node Pool

v7xe 最小 slice 是 4 chips（2x2x1 torus）。Spot 价格约为 on-demand 的 30%，适合 benchmark 和开发；生产建议用 on-demand 或 reserved capacity。

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

# 等节点就绪 (~2-5 min)
watch -n 10 "kubectl --context=$CTX get nodes -l cloud.google.com/gke-tpu-topology=2x2x1"
```

> 💡 **机型说明**: `tpu7x-standard-4t` = TPU v7xe，4 chips × 192 GB HBM = 768 GB total。
> 31B Dense 权重只占 61 GB，TP=4 让 KV Cache 分布到 4 chips，余量充足。

## Step 2: 部署 TPU Pod

仓库已提供 [`gemma4-31b-pod.yaml`](gemma4-31b-pod.yaml)。按需修改 PVC 名称，然后部署：

```bash
# 检查 PVC 名（默认假设是 lustre-pvc）
kubectl --context=$CTX get pvc

# 部署
kubectl --context=$CTX apply -f gemma4-31b-pod.yaml
kubectl --context=$CTX wait --for=condition=Ready pod/$POD --timeout=600s
```

> 💡 **关键配置**:
> - `image: vllm/vllm-tpu:nightly` — 含最新 tpu-inference 后端
> - `cloud.google.com/gke-tpu-accelerator: tpu7x` — 正确的 v7xe accelerator label
> - `command: ["sleep", "infinity"]` — Pod 保持运行，vLLM 通过 launcher 脚本启动
> - `sizeLimit: 128Gi` — Dense 模型 SHM 需求小（对比 671B MoE 需要 300Gi+）

## Step 3: 下载模型权重

> ⚠️ **前置**: Gemma4 是 gated model，必须先在 HuggingFace [接受 license](https://huggingface.co/google/gemma-4-31b-it)，
> 然后用 token 登录。下面命令第一步即处理 token；如果跳过会报 401。

```bash
export HF_TOKEN=<your-hf-token>   # 在 HuggingFace settings/tokens 创建

# 1. Pod 内登录 HF（一次性，token 写入 ~/.cache/huggingface/token）
kubectl --context=$CTX exec $POD -- bash -c "
  pip install -U 'huggingface_hub[hf_transfer]'
  huggingface-cli login --token $HF_TOKEN
"

# 2. 下载 Gemma4-31B-IT (BF16, ~61 GiB)，Lustre 上约 5 分钟
kubectl --context=$CTX exec $POD -- bash -c "
  mkdir -p $MODEL
  HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
    google/gemma-4-31b-it --local-dir $MODEL
"

# 3. 验证 shard 数
kubectl --context=$CTX exec $POD -- bash -c "ls $MODEL/*.safetensors | wc -l"
# 期望: 14 (或 model.safetensors.index.json 决定的分片数)

# 4. 清理 SHM 残留（避免历史 worker 锁文件）
kubectl --context=$CTX exec $POD -- bash -c "rm -rf /dev/shm/sem.* /dev/shm/wrk_* 2>/dev/null"
```

## Step 4: 应用必需 Kernel Patch

> ⚠️ **必须步骤**：当前 nightly 镜像的 Batched RPA kernel 在 MIXED mode 下 VMEM 预算计算遗漏了 scratch arrays，长 context (>80K tokens) 会触发 `E0200 RuntimeUnexpectedCoreHalt`。
>
> 修复：将 `prefill_batch_size` 从 2 降为 1，让 scratch memory 减半，VMEM 占用从 ~93% 降到 ~75%。
>
> 详细根因分析见 [Kernel Fix 详解](#-kernel-fix-详解)。

```bash
kubectl --context=$CTX exec $POD -- bash -c '
WRAPPER=/workspace/tpu_inference/tpu_inference/kernels/experimental/batched_rpa/wrapper.py

# 单行 sed 修复
sed -i "s/    prefill_batch_size = 2$/    prefill_batch_size = 1/" $WRAPPER

# 验证 patch 已应用
if grep -q "prefill_batch_size = 1" $WRAPPER; then
    echo "✅ Patch applied successfully"
    grep -n "prefill_batch_size" $WRAPPER
else
    echo "❌ Patch failed - check file content"
    exit 1
fi

# 清除 Python cache（重要）
find /workspace/tpu_inference -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
'
```

> 💡 **何时可跳过此步骤**: 当 [PR 修复](https://github.com/vllm-project/tpu-inference) 合并到 upstream 且 nightly 镜像更新后，`prefill_batch_size` 默认就是 1，sed 变 no-op 不影响功能。

## Step 5: 启动 vLLM 推理服务

### 启动参数解读

| 参数 | 值 | 说明 |
|------|-----|------|
| `--tensor-parallel-size` | `4` | 用满 v7xe 全部 4 chips；TP<4 会浪费算力且 KV Cache 无法横跨 chips |
| `--max-model-len` | `262144` | Gemma4 原生 256K context window |
| `--max-num-batched-tokens` | `16384` | Chunked prefill 块大小，平衡吞吐与延迟 |
| `--enable-chunked-prefill` | — | 长 context 必需，将 256K 拆为 16 个 16K chunk 处理 |
| `--async-scheduling` | — | 解耦调度与执行，提升并发吞吐 |
| `--gpu-memory-utilization` | `0.95` | 单 chip 192GB × 0.95 = ~182GB 可用 |
| `--kv-cache-dtype` | `fp8` | KV Cache 减半，最大 batch 翻倍 |
| `--limit-mm-per-prompt` | `{"image":0,"video":0}` | 关闭多模态预热，节省启动时间 |

### 必需环境变量

| 变量 | 值 | 用途 |
|------|----|------|
| `USE_BATCHED_RPA_KERNEL` | `1` | **必需**，启用 Batched RPA kernel（Gemma4 异构 head_dim 256/512 必需） |
| `VLLM_WORKER_MULTIPROC_METHOD` | `fork` | TP=4 多进程通信方式 |
| `SKIP_JAX_PRECOMPILE` | `1` | 跳过 JAX 预编译，cold start 快 30s |
| `VLLM_XLA_CHECK_RECOMPILATION` | `0` | 关闭 XLA 重编译检查 |

### 启动 (TP=4, 256K context)

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

# 监控启动 (~3-5 min)
kubectl --context=$CTX exec $POD -- tail -f /tmp/vllm_gemma4.log
```

**启动成功标志**：日志末尾出现
```
INFO:     Application startup complete.
```

## Step 6: 验证推理

### 健康检查

```bash
kubectl --context=$CTX exec $POD -- curl -sf -o /dev/null -w "%{http_code}\n" http://localhost:8000/health
# 期望: 200
```

### Smoke Test — 基础问答

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
# 期望: content: 'Paris'  finish: stop
```

### 多轮对话

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

## Step 7: 性能 Benchmark

使用 vLLM 内置的 `vllm bench serve` 工具。每组测试前会 warmup 一次触发 XLA 编译。

### 7.1 短 / 中 Context Benchmark (Tests 1-6, Random Tokens)

无需重启 vLLM。

```bash
# Test 1: 单用户延迟 (1K/1K, P=1)
kubectl --context=$CTX exec $POD -- vllm bench serve \
  --model /lustre/models/gemma-4-31b-it \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 1024 \
  --num-prompts 1 --max-concurrency 1 --num-warmups 1 --ignore-eos 2>&1 | tail -30

# Test 2: 峰值吞吐 (1K/1K, P=256) — 重点测试
kubectl --context=$CTX exec $POD -- vllm bench serve \
  --model /lustre/models/gemma-4-31b-it \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 1024 \
  --num-prompts 256 --max-concurrency 256 --num-warmups 1 --ignore-eos 2>&1 | tail -30

# Test 3: 长输入 (16K/1K, P=16)
kubectl --context=$CTX exec $POD -- vllm bench serve \
  --model /lustre/models/gemma-4-31b-it \
  --dataset-name random \
  --random-input-len 16384 --random-output-len 1024 \
  --num-prompts 16 --max-concurrency 16 --num-warmups 1 --ignore-eos 2>&1 | tail -30

# Test 4: 长输出 (1K/16K, P=4)
kubectl --context=$CTX exec $POD -- vllm bench serve \
  --model /lustre/models/gemma-4-31b-it \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 16384 \
  --num-prompts 4 --max-concurrency 4 --num-warmups 1 --ignore-eos 2>&1 | tail -30

# Test 5: 64K context (64K/1K, P=1) — 用 63488 留 1K buffer，与权威 benchmark 对齐
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

### 7.2 真实文本长 Context Benchmark (Tests 7-12, Sonnet)

vLLM 镜像自带 Shakespeare sonnets 数据集（`/workspace/vllm/benchmarks/sonnet.txt`），用于验证真实文本和 random tokens 性能差异。

```bash
SONNET=/workspace/vllm/benchmarks/sonnet.txt

# Tests 7-10: 单用户 64K / 96K / 120K / 128K（均为 input + 1K output ≤ max_position_embeddings 边界）
for LEN in 63488 98304 122880 130048; do
  echo "=== Sonnet input=$LEN P=1 ==="
  kubectl --context=$CTX exec $POD -- vllm bench serve \
    --model /lustre/models/gemma-4-31b-it \
    --dataset-name sonnet --dataset-path $SONNET \
    --sonnet-input-len $LEN --sonnet-output-len 1024 \
    --num-prompts 1 --max-concurrency 1 --num-warmups 1 --ignore-eos 2>&1 | tail -15
done

# Test 11: 128K 双并发
kubectl --context=$CTX exec $POD -- vllm bench serve \
  --model /lustre/models/gemma-4-31b-it \
  --dataset-name sonnet --dataset-path $SONNET \
  --sonnet-input-len 130048 --sonnet-output-len 1024 \
  --num-prompts 2 --max-concurrency 2 --num-warmups 1 --ignore-eos 2>&1 | tail -30

# Test 12: 64K 四并发
kubectl --context=$CTX exec $POD -- vllm bench serve \
  --model /lustre/models/gemma-4-31b-it \
  --dataset-name sonnet --dataset-path $SONNET \
  --sonnet-input-len 63488 --sonnet-output-len 1024 \
  --num-prompts 4 --max-concurrency 4 --num-warmups 1 --ignore-eos 2>&1 | tail -30
```

### 7.3 完整 256K Context Benchmark (Tests 13-16)

服务器已经按 `--max-model-len 262144` 启动，无需重启。

```bash
SONNET=/workspace/vllm/benchmarks/sonnet.txt

# Tests 13-15: 192K / 224K / 256K 单用户
for LEN in 196608 229376 261120; do
  echo "=== Sonnet input=$LEN P=1 ==="
  kubectl --context=$CTX exec $POD -- vllm bench serve \
    --model /lustre/models/gemma-4-31b-it \
    --dataset-name sonnet --dataset-path $SONNET \
    --sonnet-input-len $LEN --sonnet-output-len 1024 \
    --num-prompts 1 --max-concurrency 1 --num-warmups 1 --ignore-eos 2>&1 | tail -15
done

# Test 16: 256K 双并发
kubectl --context=$CTX exec $POD -- vllm bench serve \
  --model /lustre/models/gemma-4-31b-it \
  --dataset-name sonnet --dataset-path $SONNET \
  --sonnet-input-len 261120 --sonnet-output-len 1024 \
  --num-prompts 2 --max-concurrency 2 --num-warmups 1 --ignore-eos 2>&1 | tail -30
```

## Step 8: 清理

```bash
# 删除 Pod
kubectl --context=$CTX delete pod $POD

# 删除 node pool（可选，保留可重复使用节省 ~5 min spin-up 时间）
gcloud container node-pools delete np-tpu7xe-spot-gemma4 \
  --cluster=$CLUSTER --region=$REGION --project=$PROJECT \
  --quiet --async
```

---

## 📊 Benchmark 结果

> **测试环境**：vllm/vllm-tpu:nightly (vLLM 0.20.2rc1.dev223+)，TP=4，FP8 KV Cache，BF16 weights，含 prefill_batch_size=1 patch
>
> **方法论**：每项测试前 1 个 warmup 请求（触发 XLA 编译），主测量取 median TTFT/TPOT，peak tok/s 反映瞬时最高输出速率。`--ignore-eos` 强制跑满输出长度。

### 短 / 中 Context (Tests 1-6, Random Tokens)

| # | 场景 | Input | Output | 并发 | Output tok/s | Peak | TTFT | TPOT |
|---|------|-------|--------|------|-------------|------|------|------|
| 1 | 单用户延迟 | 1K | 1K | 1 | 28 | 29 | **86 ms** | **35 ms** |
| 2 | **峰值吞吐** | 1K | 1K | 256 | 4,495 | **6,144** ⭐ | 7,756 ms¹ | 49 ms |
| 3 | 长输入 | 16K | 1K | 16 | 317 | 432 | 7,014 ms¹ | 43 ms |
| 4 | 长输出 | 1K | 16K | 4 | 110 | 116 | 142 ms | 36 ms |
| 5 | 64K context | 64K | 1K | 1 | 27 | 29 | 196 ms | 37 ms |
| 6 | 128K context | 128K | 1K | 1 | 27 | 28 | 421 ms | 37 ms |

### 真实文本长 Context (Tests 7-12, Sonnet 数据集)

| # | Input | 并发 | Output tok/s | Peak | TTFT | TPOT |
|---|-------|------|-------------|------|------|------|
| 7  | 64K  | 1 | 27.49 | 29  | 231 ms     | 36 ms |
| 8  | 96K  | 1 | 26.74 | 28  | 302 ms     | 37 ms |
| 9  | 120K | 1 | 27.39 | 29  | 373 ms     | 36 ms |
| 10 | 128K | 1 | 27.98 | 29  | 378 ms     | 35 ms |
| 11 | 128K | 2 | 44.98 | 58  | 4,714 ms¹  | 40 ms |
| 12 | 64K  | 4 | 82.99 | 112 | 6,052 ms¹  | 42 ms |

### 完整 256K Context (Tests 13-16, Sonnet 数据集)

| # | Input | 并发 | Output tok/s | Peak | TTFT | TPOT |
|---|-------|------|-------------|------|------|------|
| 13 | 192K | 1 | 28.34 | 29 | 526 ms    | 35 ms |
| 14 | 224K | 1 | 28.21 | 29 | 644 ms    | 35 ms |
| 15 | **256K** | 1 | 28.60 | 30 | **695 ms** ⭐ | **34 ms** |
| 16 | 256K | 2 | 54.12 | 58 | 980 ms¹   | 36 ms |

¹ TTFT 包含 **排队时延**（concurrent prefill 串行调度）。单用户测试（P=1）的 TTFT 反映纯 prefill 计算时间。

### 关键观察

| 维度 | 结论 |
|------|------|
| **Decode 延迟稳定性** | TPOT 全程 34-49ms（1K → 256K，1 → 256 并发），与 context 长度无关 |
| **Prefill 线性扩展** | TTFT 单用户：1K (86ms) → 64K (196ms) → 128K (378ms) → 256K (695ms) |
| **吞吐线性扩展** | 64K 并发：1→4 用户 = 27 → 83 tok/s (3.1x)；256K：1→2 用户 = 29 → 54 tok/s (1.9x) |
| **Real text vs random** | Sonnet 真实英文 vs random tokens 性能几乎一致（TPOT 35-37ms） |
| **256K 双并发** | 两个 256K 请求同时跑成功，吞吐近 2x scaling |

---

## 🛠️ Kernel Fix 详解

### 问题现象

未打补丁的 nightly 镜像：Gemma4-31B 在 context > 80K tokens 时随机触发：
```
E0200 RuntimeUnexpectedCoreHalt
RPAm-p256-b2-q256-k256/pallas_call
```
现象是 TPU driver 层崩溃，**非确定性**（同样输入有时通过有时崩）。

### 根因

`tpu_inference/kernels/experimental/batched_rpa/wrapper.py` 的 `calculate_vmem_usage()` 函数：
- ✅ 计算了 pipeline buffers（Q/KV/O arrays）
- ❌ **遗漏了 scratch arrays**（`m`, `l`, `acc` 来自 `lm_scratch_shape` 和 `acc_scratch_shape`）

在 `prefill_batch_size=2`、TPU v7x 64MB VMEM 上：
- Pipeline buffers (auto-tuned to 80% cap): ~36 MB
- Scratch arrays (未计入): ~24 MB
- **Total: ~60 MB ≈ 93% VMEM 占用**

边际溢出导致 MIXED mode（chunked prefill + decode 混合）在长 context 时随机崩溃。

### Fix（单行）

`tpu_inference/kernels/experimental/batched_rpa/wrapper.py`:

```diff
-    prefill_batch_size = 2
+    prefill_batch_size = 1
```

将 MIXED mode 的 batch size 减半，scratch memory 从 ~24MB 降到 ~12MB，**总 VMEM 占用从 ~93% 降到 ~75%**，彻底消除溢出。

### 验证矩阵

| Context Length | 修复前 | 修复后 |
|---------------|-------|-------|
| ≤ 80K (≤ 40K prompt) | ✅ 偶尔通过 | ✅ |
| 95K | ❌ E0200 crash | ✅ |
| 128K (full context) | ❌ crash | ✅ |
| 256K (extended) | ❌ crash | ✅ |

**性能影响**：单用户 TPOT 35-37ms 维持不变（仅影响 chunked prefill 阶段，单序列推理无回归）。

---

## 📋 Troubleshooting

### 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `RESOURCE_EXHAUSTED` 创建 node pool | v7xe spot 容量不足 | 换 zone 或换 region |
| 权重下载 403 | HF token 无 Gemma4 权限 | 在 HuggingFace 接受 license |
| 启动 hang / 无 log | libtpu lockfile 残留 | `rm -f /tmp/libtpu_lockfile` 后重启 |
| 长 context (>80K) E0200 crash | 未应用 prefill_batch_size patch | 重做 [Step 4](#step-4-应用必需-kernel-patch) |
| TP=4 XLA Reshape layout 错误 | nightly 镜像缺 Gemma4 layout fix | 拉最新 nightly 镜像 |
| `ImportError: gemma4` | nightly 镜像太旧 | 拉最新 nightly 镜像 |
| Pod OOM Killed | SHM 不够 | yaml 中 `sizeLimit: 128Gi` 已足够，检查并行进程 |

### 日志和诊断

```bash
# 完整启动日志
kubectl --context=$CTX exec $POD -- cat /tmp/vllm_gemma4.log

# HBM 使用情况（每 chip）
kubectl --context=$CTX exec $POD -- python3 -c "
import jax
for d in jax.devices():
    s = d.memory_stats()
    print(f'{d}: {s[\"bytes_in_use\"]/1e9:.1f} GB / {s[\"bytes_limit\"]/1e9:.1f} GB')
"

# TPU 进程状态
kubectl --context=$CTX exec $POD -- ps aux | grep -E 'vllm|EngineCore'
```

### 已知镜像兼容性

| 测试通过的 nightly 版本 | 备注 |
|------------------------|------|
| `vllm/vllm-tpu:nightly` (vLLM 0.20.2rc1.dev223+) | 含 Batched RPA Gemma4 layout fix (PR #2506) 和 K/V_proj sharding fix (PR #2585) |

> 💡 如果 `prefill_batch_size` patch 后仍长 context crash，先确认你的 nightly 镜像 ≥ dev223。早期镜像缺基础 Gemma4 patches。

---

## 📎 参考

- [Gemma 4 Model Card](https://ai.google.dev/gemma/docs/core/model_card_4) — 官方模型规格
- [vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference) — TPU 推理后端源码
- [HuggingFace: google/gemma-4-31b-it](https://huggingface.co/google/gemma-4-31b-it) — 模型权重
- [PR #2506](https://github.com/vllm-project/tpu-inference/pull/2506) — Batched RPA Gemma4 layout fix
- [PR #2585](https://github.com/vllm-project/tpu-inference/pull/2585) — K/V_proj sharding fix

---

> **文档版本**: v2.0
>
> **最后更新**: 2026-05-15
>
> **复现状态**: ✅ 16/16 benchmark tests PASS（含完整 256K context window）
