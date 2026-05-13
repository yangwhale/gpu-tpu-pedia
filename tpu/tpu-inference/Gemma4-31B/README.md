# Gemma4-31B Inference on TPU v7xe

> 🌐 **Languages** | **语言**: **中文** · [English](README.en.md)

> 端到端指南：在 TPU v7xe（单 chip, TP=1）上运行 Gemma4-31B BF16 推理。
>
> **架构**：30.7B Dense / 60 layers / hybrid sliding-window + global attention / 256K context / 262K vocab / 多模态（text + image）
>
> **代码仓库**: [vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference)（main branch, JAX backend `gemma4.py` / `gemma4_mm.py`）
>
> **模型**: [google/gemma-4-31b-it](https://huggingface.co/google/gemma-4-31b-it)（BF16, ~61 GiB）

---

## 🔍 成熟度评估

> ⚠️ **Alpha 阶段** — tpu-inference 已有 Gemma4 的 JAX 实现（`gemma4.py` + `gemma4_mm.py`），但存在活跃 bug：
>
> | Issue | 问题 | 影响 |
> |-------|------|------|
> | [#2453](https://github.com/vllm-project/tpu-inference/issues/2453) | MoE 变体权重加载 OOM | 仅影响 26B MoE，不影响 31B Dense |
> | [#2126](https://github.com/vllm-project/tpu-inference/issues/2126) | torchax 后端无法运行 | 可能影响 PyTorch path，JAX path 待验证 |
> | [vllm#39827](https://github.com/vllm-project/vllm/issues/39827) | 输出重复 token | 质量问题，待排查 |
>
> **本测试目标**：验证 31B Dense 在 TPU v7 上的端到端推理可行性。

---

## 🧮 HBM 估算

| 项目 | BF16 | FP8（如支持） |
|------|------|--------------|
| 模型权重 | 30.7B × 2B = **~61.4 GB** | 30.7B × 1B = **~30.7 GB** |
| KV Cache (4K ctx) | ~1-2 GB | ~0.5-1 GB |
| KV Cache (256K ctx) | ~30-60 GB（估算） | ~15-30 GB |
| **总计 (4K ctx)** | **~63 GB** | **~32 GB** |
| TPU v7xe 单 chip HBM | **192 GB** | **192 GB** |
| **利用率** | **33%** | **17%** |

**结论**：**单 chip 即可运行**，BF16 全精度只用 33% HBM。v7xe 最小配置 4 chips（2x2x1），TP=1 仅使用其中 1 个 chip。

### KV Cache 详细估算

Gemma4 31B 使用 **hybrid attention**（50 sliding + 10 full），KV Cache 结构特殊：

| 层类型 | 数量 | KV heads | head_dim | 窗口 | 每 token KV (BF16) | 每 token KV (FP8) |
|--------|------|----------|----------|------|-------------------|-------------------|
| sliding_attention | 50 | 16 | 256 | 1024 tokens | 16,384 B（固定上限） | 8,192 B |
| full_attention | 10 | 4 | 512 | 无限制 | 8,192 B | 4,096 B |

**每 batch slot 的 KV Cache（max_model_len=4096）**：

| 组件 | BF16 | FP8 |
|------|------|-----|
| Sliding 层 (50层 × 1024 tokens 窗口固定) | 838 MB | 419 MB |
| Full 层 (10层 × 4096 tokens) | 336 MB | 168 MB |
| **每 slot 总计** | **1.15 GB** | **0.57 GB** |

**最大 batch size 推荐**：

```
可用 HBM = 192 GB × 0.9 - 61.4 GB (weights) - 2 GB (buffer) ≈ 109 GB
```

| KV dtype | 每 slot | **max batch** | **推荐 max-num-seqs** |
|----------|---------|--------------|----------------------|
| BF16 | 1.15 GB | ~94 | **80** |
| **FP8** ⭐ | **0.57 GB** | **~190** | **160** |

> 💡 **推荐 FP8 KV Cache**（`--kv-cache-dtype fp8`）：batch 翻倍，吞吐量提升显著，精度损失极小。

---

## 🧭 部署方案

| 模式 | TPU 配置 | 说明 |
|------|---------|------|
| **单 chip TP=1** ⭐ | 1 × v7xe (4 chips, 只用 1 chip) | **推荐**，31B Dense 单 chip 绰绰有余 |
| 4 chips TP=4 | 1 × v7xe (4 chips) | 可选，分摊 KV Cache 支持更长 context |

---

## ⚡ Quick Start (老手 5 命令复现)

```bash
CTX=<your-gke-context>; POD=<your-tpu-pod>; MODEL=/lustre/models/gemma-4-31b-it

# 1. 验证模型权重
kubectl --context=$CTX exec $POD -- bash -c "ls $MODEL/*.safetensors | wc -l"

# 2. 写 launcher
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

# 4. 等 cold start (~3-5 min，31B 比 397B 快很多)
for i in $(seq 1 20); do C=$(kubectl --context=$CTX exec $POD -- curl -sf -o /dev/null -w "%{http_code}" http://localhost:8000/health); echo "T+$((i*30))s HTTP $C"; [ "$C" = "200" ] && break; sleep 30; done

# 5. Smoke test
kubectl --context=$CTX exec $POD -- curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/lustre/models/gemma-4-31b-it","messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"max_tokens":20,"temperature":0}' \
  | python3 -c 'import sys,json;r=json.load(sys.stdin);print(r["choices"][0]["message"]["content"])'
# 期望: Paris
```

---

# 端到端部署步骤

## Step 0: 环境准备

### GKE 集群要求

- GKE 集群需支持 TPU v7（Ironwood）
- 已配置 Lustre 或 GCS 共享存储
- kubectl 已配置 context

### 确认集群和 context

```bash
# 设置变量
export PROJECT=<your-gcp-project>
export CLUSTER=<your-gke-cluster>
export REGION=<your-region>          # e.g., us-central1
export ZONE=<your-zone>              # e.g., us-central1-c
export CTX=<your-gke-context>

# 验证集群可达
kubectl --context=$CTX get nodes | grep tpu
```

## Step 1: 创建 TPU v7xe Spot Node Pool

```bash
# 创建 v7xe spot node pool (4 chips, 2x2x1 torus)
gcloud container node-pools create np-tpu7xe-spot-gemma4 \
  --cluster=$CLUSTER --region=$REGION --project=$PROJECT \
  --node-locations=$ZONE --machine-type=ct7xe-standard-4t --num-nodes=1 --spot \
  --disk-type=hyperdisk-balanced --disk-size=200 \
  --node-taints=google.com/tpu=present:NoSchedule \
  --workload-metadata=GKE_METADATA --enable-autorepair --enable-autoupgrade --async

# 等待节点就绪 (~2-5 min)
watch "kubectl --context=$CTX get nodes -l cloud.google.com/gke-tpu-topology=2x2x1"
```

> 💡 **机型说明**: `ct7xe-standard-4t` = TPU v7xe, 4 chips, 每 chip 192 GB HBM, 总 768 GB。
> 31B Dense 单 chip 只需 ~63 GB，TP=1 即可。但 GKE 最小 TPU pod slice 是 4 chips。

## Step 2: 部署 TPU Pod

### 编写 Pod YAML

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
      claimName: lustre-pvc        # 替换为你的 Lustre PVC 名
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: 128Gi             # 31B Dense 不需要太大 SHM
  restartPolicy: Never
```

> 💡 **SHM 大小**: MoE 模型需要大 SHM 做 expert re-quant，31B Dense 模型 128Gi 足够。
> 对比：DeepSeek-R1 671B 需要 300Gi+，Qwen3.5 397B 需要 200Gi+。

```bash
# 部署
kubectl --context=$CTX apply -f gemma4-31b-pod.yaml

# 等待就绪
kubectl --context=$CTX wait --for=condition=Ready pod/gemma4-31b --timeout=600s
```

## Step 3: 下载模型权重

```bash
POD=gemma4-31b
MODEL=/lustre/models/gemma-4-31b-it

# 下载 Gemma4 31B IT (BF16) 到 Lustre (~61 GiB, Lustre ~5 min)
kubectl --context=$CTX exec $POD -- bash -c "
  mkdir -p $MODEL
  pip install -U 'huggingface_hub[hf_transfer]'
  HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
    google/gemma-4-31b-it \
    --local-dir $MODEL
"

# 验证权重文件完整
kubectl --context=$CTX exec $POD -- bash -c "ls $MODEL/*.safetensors | wc -l"
# 记录输出的 shard 数量

# 清理 /dev/shm 残留
kubectl --context=$CTX exec $POD -- bash -c "rm -rf /dev/shm/sem.* /dev/shm/wrk_* 2>/dev/null"
```

> ⚠️ **HuggingFace 访问**: Gemma4 需要接受 license agreement。确保你的 HF token 有权限访问 `google/gemma-4-31b-it`。
> 设置 token: `kubectl exec $POD -- bash -c "huggingface-cli login --token <your-hf-token>"`

## Step 4: 启动 vLLM 推理服务

### 关键参数说明

| 参数 | 取值 | 说明 |
|------|------|------|
| `--tensor-parallel-size` | `1` | 单 chip 即可，不需要 TP |
| `--max-model-len` | `4096` / `131072` / `262144` | 按测试场景调整 |
| `--max-num-seqs` | `160` (FP8 KV) / `80` (BF16 KV) | 见下方 KV Cache 估算 |
| `--kv-cache-dtype` | `fp8` | **推荐**，KV Cache 减半，batch 翻倍 |
| `--no-enable-prefix-caching` | 必须 | 避免潜在的 prefix caching bug |
| `--gpu-memory-utilization` | `0.9` | 单 chip 192GB, 0.9 = 172GB 可用 |
| `--block-size` | `256` | CI 默认值 |
| `--trust-remote-code` | 必须 | Gemma4 自定义模型代码 |

### 环境变量

| 变量 | 值 | 说明 |
|------|------|------|
| `SKIP_JAX_PRECOMPILE` | `1` | 跳过 JAX 预编译，加速启动 |
| `VLLM_XLA_CHECK_RECOMPILATION` | `0` | 关闭 XLA 重编译检查 |

> ⚠️ **注意**: Gemma4 在 tpu-inference 中使用 **JAX native** 实现（不是 PyTorch/TorchAX path），
> 所以**不需要** `MODEL_IMPL_TYPE=vllm`（那是 MoE / PyTorch 模型用的）。
> 如果启动时报 JAX/Flax 错误，可以尝试加 `MODEL_IMPL_TYPE=vllm` 切换到 TorchAX path。

### 启动服务

```bash
# 1. 写 launcher
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

# 3. 监控启动 (~3-5 min cold start)
kubectl --context=$CTX exec $POD -- tail -f /tmp/vllm_gemma4.log
```

**等待关键日志**：
```
INFO: Application startup complete.                    ← 启动成功
```

## Step 5: 验证推理

### 健康检查

```bash
kubectl --context=$CTX exec $POD -- curl -sf -o /dev/null -w "%{http_code}\n" http://localhost:8000/health
# 期望: 200
```

### Smoke Test — 简单问答

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

### Thinking Mode 测试

```bash
# Thinking ON (默认, 使用 <|think|> token)
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
# 期望: 模型先 thinking 再给出 925
```

### 多轮对话测试

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
  }' | python3 -c 'import sys,json; r=json.load(sys.stdin); print(r["choices"][0]["message"]["content"][:200])'
```

## Step 6: 性能 Benchmark

### 测试矩阵

| 编号 | 场景 | Input Len | Output Len | 并发 | max-model-len | 目标指标 |
|------|------|-----------|------------|------|---------------|---------|
| B1 | **单用户延迟** | 1K | 1K | P1 | 4096 | TTFT, ITL, TPOT |
| B2 | **标准吞吐** | 1K | 1K | P64 | 4096 | tok/s, ITL |
| B3 | **峰值吞吐** | 1K | 1K | P160 | 4096 | tok/s (找 max batch) |
| B4 | **长输入短输出** | 16K | 1K | P4 | 32768 | TTFT (prefill 性能) |
| B5 | **短输入长输出** | 1K | 16K | P4 | 32768 | ITL stability |
| B6 | **128K 长 context** | 128K | 256 | P1 | 131072 | TTFT, 是否 OOM |
| B7 | **256K 最大 context** | 256K | 256 | P1 | 262144 | TTFT, 是否 OOM |

> ⚠️ B6/B7 长 context 测试需要**重启 vLLM 并调整 `--max-model-len`**（131072 / 262144）。
> 256K context 下 full attention 层的 KV Cache = 10 × 4 × 512 × 2 × 2 × 262144 = **20.5 GB** (BF16) / **10.2 GB** (FP8)。
> sliding 层固定 1024 窗口不变（~0.8 GB）。单 chip 192GB 单 batch 可以放下。

### 6.1 短 context 测试 (B1-B3, max-model-len=4096)

使用默认启动参数（Step 4 的 launcher），无需重启。

```bash
# B1: 单用户延迟 (1K/1K, P1)
kubectl --context=$CTX exec $POD -- python3 -m vllm.entrypoints.openai.run_batch \
  --model /lustre/models/gemma-4-31b-it \
  --input-len 1024 --output-len 1024 \
  --num-prompts 1 2>&1 | tail -20

# B2: 标准吞吐 (1K/1K, P64)
kubectl --context=$CTX exec $POD -- python3 -m vllm.entrypoints.openai.run_batch \
  --model /lustre/models/gemma-4-31b-it \
  --input-len 1024 --output-len 1024 \
  --num-prompts 64 2>&1 | tail -20

# B3: 峰值吞吐 (1K/1K, P160)
kubectl --context=$CTX exec $POD -- python3 -m vllm.entrypoints.openai.run_batch \
  --model /lustre/models/gemma-4-31b-it \
  --input-len 1024 --output-len 1024 \
  --num-prompts 160 2>&1 | tail -20
```

### 6.2 中长 context 测试 (B4-B5, max-model-len=32768)

需要重启 vLLM（修改 `--max-model-len 32768`，`--max-num-seqs` 相应下调）。

```bash
# 重启 vLLM with max-model-len=32768
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

# 等 cold start
for i in $(seq 1 20); do C=$(kubectl --context=$CTX exec $POD -- curl -sf -o /dev/null -w "%{http_code}" http://localhost:8000/health); echo "T+$((i*30))s HTTP $C"; [ "$C" = "200" ] && break; sleep 30; done

# B4: 长输入短输出 (16K/1K, P4)
kubectl --context=$CTX exec $POD -- python3 -m vllm.entrypoints.openai.run_batch \
  --model /lustre/models/gemma-4-31b-it \
  --input-len 16384 --output-len 1024 \
  --num-prompts 4 2>&1 | tail -20

# B5: 短输入长输出 (1K/16K, P4)
kubectl --context=$CTX exec $POD -- python3 -m vllm.entrypoints.openai.run_batch \
  --model /lustre/models/gemma-4-31b-it \
  --input-len 1024 --output-len 16384 \
  --num-prompts 4 2>&1 | tail -20
```

### 6.3 超长 context 测试 (B6-B7, 128K / 256K)

> ⚠️ 这是**极端测试**，用于验证 Gemma4 的 256K context 上限在 TPU v7xe 上是否可行。

```bash
# 重启 vLLM with max-model-len=262144 (256K)
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

# 等 cold start (可能更久，XLA 编译 256K shape)
for i in $(seq 1 40); do C=$(kubectl --context=$CTX exec $POD -- curl -sf -o /dev/null -w "%{http_code}" http://localhost:8000/health); echo "T+$((i*30))s HTTP $C"; [ "$C" = "200" ] && break; sleep 30; done

# B6: 128K context (128K/256, P1)
kubectl --context=$CTX exec $POD -- python3 -m vllm.entrypoints.openai.run_batch \
  --model /lustre/models/gemma-4-31b-it \
  --input-len 131072 --output-len 256 \
  --num-prompts 1 2>&1 | tail -20

# B7: 256K context (256K/256, P1) — Gemma4 最大 context
kubectl --context=$CTX exec $POD -- python3 -m vllm.entrypoints.openai.run_batch \
  --model /lustre/models/gemma-4-31b-it \
  --input-len 262144 --output-len 256 \
  --num-prompts 1 2>&1 | tail -20
```

### 6.4 结果记录模板

| 编号 | 场景 | TTFT (ms) | ITL (ms) | 吞吐量 (tok/s) | tok/s/user | 状态 |
|------|------|-----------|----------|---------------|------------|------|
| B1 | 1K/1K P1 | — | — | — | — | ⏳ 待测 |
| B2 | 1K/1K P64 | — | — | — | — | ⏳ 待测 |
| B3 | 1K/1K P160 | — | — | — | — | ⏳ 待测 |
| B4 | 16K/1K P4 | — | — | — | — | ⏳ 待测 |
| B5 | 1K/16K P4 | — | — | — | — | ⏳ 待测 |
| B6 | 128K/256 P1 | — | — | — | — | ⏳ 待测 |
| B7 | 256K/256 P1 | — | — | — | — | ⏳ 待测 |

> 📊 **预期基线**（31B Dense + 单 chip + BF16 weights + FP8 KV）：
> - Cold start: ~2-3 min（权重小，无 MoE re-quant）
> - B1 单用户: 预期 > 50 tok/s/user（31B Dense 比 397B MoE 快很多）
> - B3 峰值: 取决于 KV Cache 容量和 XLA 调度效率
> - B6/B7: 主要验证可行性，TTFT 可能很长（128K+ prefill）

## Step 7: 清理

```bash
# 删除 pod
kubectl --context=$CTX delete pod gemma4-31b

# 删除 node pool (可选)
gcloud container node-pools delete np-tpu7xe-spot-gemma4 \
  --cluster=$CLUSTER --region=$REGION --project=$PROJECT --quiet --async
```

---

## 📋 Troubleshooting

### 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| `RESOURCE_EXHAUSTED` 创建 node pool | v7xe spot 容量不足 | 换 zone 或等待 |
| 权重下载 403 | HF token 无 Gemma4 访问权限 | 去 HuggingFace 接受 license |
| 启动后 hang / 无 log | libtpu lockfile 残留 | `rm -f /tmp/libtpu_lockfile` |
| 输出重复 token | 已知 bug vllm#39827 | 降低 temperature 或等待上游修复 |
| `ImportError: gemma4` | tpu-inference 版本太旧 | 确保用最新 main branch image |
| OOM on model load | 不应发生（31B << 192GB） | 检查是否有其他进程占用 HBM |

### 日志排查

```bash
# 查看完整启动日志
kubectl --context=$CTX exec $POD -- cat /tmp/vllm_gemma4.log

# 查看 HBM 使用
kubectl --context=$CTX exec $POD -- python3 -c "
import jax
for d in jax.devices():
    stats = d.memory_stats()
    print(f'{d}: {stats[\"bytes_in_use\"]/1e9:.1f} GB / {stats[\"bytes_limit\"]/1e9:.1f} GB')
"
```

---

## 📎 参考

- [Gemma 4 Model Card](https://ai.google.dev/gemma/docs/core/model_card_4) — 官方模型规格
- [vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference) — TPU 推理后端
- [Gemma4 Issues](https://github.com/vllm-project/tpu-inference/issues?q=gemma4) — 已知问题追踪
- [HuggingFace: google/gemma-4-31b-it](https://huggingface.co/google/gemma-4-31b-it) — 模型权重

---

---

## 🧪 实测记录 (2026-05-13)

### 环境

| 项目 | 值 |
|------|------|
| 集群 | chrisya-v7x-v134 (cloud-tpu-multipod-dev, us-central1) |
| Node Pool | np-tpu7x-spot-gemma4 (2x2x1, 4 chips, Spot) |
| Image | vllm/vllm-tpu:nightly (vLLM 0.20.2rc1.dev223) |
| 模型 | gemma-4-31b-it BF16 (59 GB, 2 safetensors) |
| 权重下载 | Lustre, 40 秒 |

### 实测结果

| 配置 | Cold Start | Smoke Test (25 tok) | 1K Token Prompt | 状态 |
|------|-----------|---------------------|-----------------|------|
| **TP=1** | 3 min | ✅ "Paris" | ❌ VMEM OOM (RPA Pallas kernel scratch) | **短 prompt 可用，长 prompt 崩** |
| **TP=4** | 3.5 min | ❌ XLA layout error | — | **无法推理** |

### Bug 1: TP=1 VMEM OOM (长 prompt)

Smoke test（25 tokens）正常输出 "Paris"，但 1024 token prompt 触发 VMEM OOM：

```
RPAm-p_256-bq_512_512-bkv_2048_512-sw_1024/pallas_call
Largest program allocations in vmem:
  1. Size: 36.00M  Shape: f8e4m3fn[2,2048,9,4,256]  Tag: scratch operand
```

**根因**: Gemma4 的 hybrid attention（sliding_window=1024 + global, head_dim=256/512）导致 RPA kernel 需要的 scratch buffer 超过单 chip VMEM 容量。短 prompt 的 scratch 较小能通过，长 prompt 触发更大分配。

### Bug 2: TP=4 XLA Reshape Layout

```
jax.errors.JaxRuntimeError: FAILED_PRECONDITION: 
  Reshape should have supported layout before reaching the emitter.
```

**根因**: Gemma4 的 attention head 配置（sliding: 16 KV heads × 256 dim, global: 4 KV heads × 512 dim）在 TP=4 分片后，tensor shape 不被 XLA emitter 支持。

### 踩坑记录

| # | 问题 | 解决 | 发现时间 |
|---|------|------|---------|
| 1 | GKE accelerator 标签 `tpu-v7-lite-podslice` 被拒 | 正确标签: `tpu7x` | 11:30 |
| 2 | Pod 无 entrypoint 自动退出 | 加 `command: ["sleep", "infinity"]` | 11:35 |
| 3 | Spot 节点被抢占 | 重建后重新部署 | 11:40 |
| 4 | `huggingface-cli` 废弃 | 改用 `hf` 命令 | 11:45 |
| 5 | evalscope random dataset 需要 tokenizer-path | 加 `--tokenizer-path` | 12:50 |
| 6 | TP=1 长 prompt VMEM OOM | **未解决** — tpu-inference RPA kernel bug | 12:53 |
| 7 | TP=4 XLA Reshape layout 不兼容 | **未解决** — tpu-inference XLA bug | 12:58 |

### 结论（Round 1）

> ⚠️ **默认 RPA v3 kernel 无法支持 Gemma4 的大 head_dim (256/512)。**

---

## 🧪 Round 2: Batched RPA + Patches (2026-05-13 14:00+)

### 修复方案

| 修复项 | 来源 | 说明 |
|--------|------|------|
| Batched RPA Gemma4 layout fix | PR [#2506](https://github.com/vllm-project/tpu-inference/pull/2506) (May 6) | 修复 XLA Reshape layout 错误 |
| K/V_proj sharding fix | PR [#2585](https://github.com/vllm-project/tpu-inference/pull/2585) (May 12) | 修复 K/V 投影分片 |
| n_buffer 3→2 | 本地 patch | Batched RPA 默认 n_buffer=3 对大 head_dim OOM，改为 2 |
| `USE_BATCHED_RPA_KERNEL=1` | 环境变量 | 启用 VMEM-aware 的 Batched RPA kernel |

### 部署步骤（从 Lustre 应用 patches）

```bash
# 从 main branch 下载的 patches 已存在 /lustre/patches/gemma4/
PATCH=/lustre/patches/gemma4
TPI=/workspace/tpu_inference/tpu_inference

cp $PATCH/gemma4.py $TPI/models/jax/gemma4.py
cp $PATCH/gemma4_mm.py $TPI/models/jax/gemma4_mm.py
cp $PATCH/attention_interface.py $TPI/layers/common/attention_interface.py
mkdir -p $TPI/kernels/experimental/batched_rpa
cp $PATCH/batched_rpa/__init__.py $TPI/kernels/experimental/batched_rpa/
cp $PATCH/batched_rpa/wrapper.py $TPI/kernels/experimental/batched_rpa/
cp $PATCH/rpa_v3/kernel.py $TPI/kernels/ragged_paged_attention/v3/kernel.py

# 关键：n_buffer 3→2
sed -i 's/n_buffer = 3/n_buffer = 2/' $TPI/kernels/experimental/batched_rpa/wrapper.py

find $TPI -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
```

### TP=1 Benchmark 结果 (evalscope, 1K/1K)

| 并发 | 请求数 | 成功率 | Avg Lat (s) | TTFT (ms) | TPOT (ms) | 吞吐 (tok/s) | Decode (tok/s/user) |
|------|--------|--------|------------|-----------|-----------|-------------|-------------------|
| P1 | 3 | 100% | 20.4 | 40.3 | — | 7.23 | 20.6 |
| P8 | 16 | 100% | 19.8 | 12,137 | 48.5 | 59.7 | 20.6 |
| P32 | 64 | 100% | 22.5 | 3,261 | 114.8 | 173.2 | 8.7 |
| **P64** | **128** | **100%** | **20.5** | **5,723** | **84.7** | **447.3** ⭐ | **11.8** |
| P128 | 256 | 100% | 38.0 | 21,865 | 90.8 | 407.1 | 11.0 |

**峰值吞吐：447 tok/s @ P64**（TP=1 单 chip）

---

> **文档版本**: v0.3 (Batched RPA 修复成功, TP=1 benchmark 数据)
>
> **最后更新**: 2026-05-13
