# Gemma4-31B Inference on TPU v7xe

> 🌐 **Languages** | **语言**: **中文** · [English](README.en.md)

> 端到端指南：在 TPU v7xe 上运行 Gemma4-31B 推理（TP=1 单 chip 或 TP=4 全 pod）。
>
> **架构**：30.7B Dense / 60 layers / hybrid sliding-window + global attention / 256K context / 262K vocab / 多模态（text + image）
>
> **代码仓库**: [vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference)（main branch, JAX backend `gemma4.py` / `gemma4_mm.py`）
>
> **模型**: [google/gemma-4-31b-it](https://huggingface.co/google/gemma-4-31b-it)（BF16, ~61 GiB）

---

## 🔍 成熟度评估

> **Beta 阶段** — tpu-inference 的 Gemma4 JAX 实现需要 Batched RPA patches 才能正常工作：
>
> | 项目 | 状态 | 说明 |
> |------|------|------|
> | TP=1 / TP=4 推理 | ✅ 可用 | 需要 Batched RPA + n_buffer=2 patch |
> | 短 context (≤32K) | ✅ 稳定 | 1K~32K 测试全部通过 |
> | 长 context (64K+) | ❌ 不可用 | Pallas kernel crash / TPU driver SIGABRT |
> | 多模态 (image) | ⏳ 未测 | `gemma4_mm.py` 存在但未验证 |
>
> **已知上游 Issues**:
> | Issue | 问题 | 影响 |
> |-------|------|------|
> | [#2453](https://github.com/vllm-project/tpu-inference/issues/2453) | MoE 变体权重加载 OOM | 仅影响 26B MoE，不影响 31B Dense |
> | [vllm#39827](https://github.com/vllm-project/vllm/issues/39827) | 输出重复 token | 质量问题，降低 temperature 可缓解 |

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

**结论**：**单 chip 即可运行**，BF16 全精度只用 33% HBM。v7xe 最小配置 4 chips（2x2x1），TP=1 仅使用其中 1 chip。

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

| KV dtype | 每 slot | max batch | 推荐 max-num-seqs |
|----------|---------|----------|-------------------|
| BF16 | 1.15 GB | ~94 | **80** |
| **FP8** ⭐ | **0.57 GB** | **~190** | **160** |

> 💡 **推荐 FP8 KV Cache**（`--kv-cache-dtype fp8`）：batch 翻倍，吞吐量提升显著，精度损失极小。

---

## 🧭 部署方案

| 模式 | TPU 配置 | 说明 |
|------|---------|------|
| 单 chip TP=1 | 1 × v7xe (4 chips, 只用 1 chip) | 开发验证、低流量场景 |
| **4 chips TP=4** ⭐ | 1 × v7xe (4 chips) | **推荐**，吞吐 ~10x，不浪费资源 |

> 💡 GKE 最小 TPU pod slice 是 4 chips。**TP=1 只用 1 个 chip，浪费 3 个**。
> TP=4 峰值吞吐 4,394 tok/s 是 TP=1 的 9.8x，接近线性扩展。建议生产部署用 TP=4。

---

## ⚡ Quick Start (老手 6 命令复现)

```bash
CTX=<your-gke-context>; POD=gemma4-31b; MODEL=/lustre/models/gemma-4-31b-it

# 1. 验证模型权重
kubectl --context=$CTX exec $POD -- bash -c "ls $MODEL/*.safetensors | wc -l"

# 2. 应用 Batched RPA patches（必须！默认 RPA 不支持 Gemma4 的大 head_dim）
kubectl --context=$CTX exec $POD -- bash -c '
TPI=/workspace/tpu_inference/tpu_inference
PATCH=/lustre/patches/gemma4
cp $PATCH/gemma4.py $PATCH/gemma4_mm.py $TPI/models/jax/
cp $PATCH/attention_interface.py $TPI/layers/common/
mkdir -p $TPI/kernels/experimental/batched_rpa
cp $PATCH/batched_rpa/__init__.py $PATCH/batched_rpa/wrapper.py $TPI/kernels/experimental/batched_rpa/
cp $PATCH/rpa_v3/kernel.py $TPI/kernels/ragged_paged_attention/v3/
sed -i "s/n_buffer = 3/n_buffer = 2/" $TPI/kernels/experimental/batched_rpa/wrapper.py
find $TPI -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null; echo "Patches applied"'

# 3. 写 launcher (TP=4 推荐)
cat > /tmp/launch_gemma4.sh <<'L'
#!/bin/bash
pgrep -f 'EngineCore|vllm' | xargs -r kill -9; sleep 2
rm -f /tmp/libtpu_lockfile /tmp/vllm_gemma4.log; touch /tmp/vllm_gemma4.log
setsid nohup env SKIP_JAX_PRECOMPILE=1 VLLM_XLA_CHECK_RECOMPILATION=0 USE_BATCHED_RPA_KERNEL=1 \
  vllm serve /lustre/models/gemma-4-31b-it \
    --tensor-parallel-size 4 --max-model-len 4096 \
    --max-num-batched-tokens 4096 --max-num-seqs 256 \
    --no-enable-prefix-caching --gpu-memory-utilization 0.9 \
    --kv-cache-dtype fp8 --block-size 256 --trust-remote-code \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --async-scheduling \
    >> /tmp/vllm_gemma4.log 2>&1 < /dev/null & disown
exit 0
L

# 4. cp + run
kubectl --context=$CTX cp /tmp/launch_gemma4.sh $POD:/tmp/launch_gemma4.sh
kubectl --context=$CTX exec $POD -- bash /tmp/launch_gemma4.sh

# 5. 等 cold start (~3-5 min)
for i in $(seq 1 20); do C=$(kubectl --context=$CTX exec $POD -- curl -sf -o /dev/null -w "%{http_code}" http://localhost:8000/health); echo "T+$((i*30))s HTTP $C"; [ "$C" = "200" ] && break; sleep 30; done

# 6. Smoke test
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
export PROJECT=<your-gcp-project>
export CLUSTER=<your-gke-cluster>
export REGION=<your-region>          # e.g., us-central1
export ZONE=<your-zone>              # e.g., us-central1-c
export CTX=<your-gke-context>

kubectl --context=$CTX get nodes | grep tpu
```

## Step 1: 创建 TPU v7xe Spot Node Pool

```bash
gcloud container node-pools create np-tpu7xe-spot-gemma4 \
  --cluster=$CLUSTER --region=$REGION --project=$PROJECT \
  --node-locations=$ZONE --machine-type=tpu7x-standard-4t --num-nodes=1 --spot \
  --disk-type=hyperdisk-balanced --disk-size=200 \
  --node-taints=google.com/tpu=present:NoSchedule \
  --workload-metadata=GKE_METADATA --enable-autorepair --enable-autoupgrade --async

# 等待节点就绪 (~2-5 min)
watch "kubectl --context=$CTX get nodes -l cloud.google.com/gke-tpu-topology=2x2x1"
```

> 💡 **机型说明**: `tpu7x-standard-4t` = TPU v7xe, 4 chips, 每 chip 192 GB HBM, 总 768 GB。
> 31B Dense 单 chip 只需 ~63 GB，TP=1 即可。但 GKE 最小 TPU pod slice 是 4 chips。

## Step 2: 部署 TPU Pod

使用仓库中的 [`gemma4-31b-pod.yaml`](gemma4-31b-pod.yaml)：

```bash
# 按需修改 PVC 名，然后部署
kubectl --context=$CTX apply -f gemma4-31b-pod.yaml

# 等待就绪
kubectl --context=$CTX wait --for=condition=Ready pod/gemma4-31b --timeout=600s
```

> 💡 **关键配置**：Pod 使用 `vllm/vllm-tpu:nightly` 镜像，accelerator label `tpu7x`。
> SHM 128Gi 对 31B Dense 足够（对比：DeepSeek-R1 671B 需要 300Gi+）。
> `command: ["sleep", "infinity"]` 让 Pod 保持运行，vLLM 通过 launcher 脚本手动启动。

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

# 清理 /dev/shm 残留
kubectl --context=$CTX exec $POD -- bash -c "rm -rf /dev/shm/sem.* /dev/shm/wrk_* 2>/dev/null"
```

> ⚠️ **HuggingFace 访问**: Gemma4 需要接受 license agreement。确保你的 HF token 有权限访问 `google/gemma-4-31b-it`。
> 设置 token: `kubectl exec $POD -- bash -c "huggingface-cli login --token <your-hf-token>"`

## Step 4: 应用 Batched RPA Patches

> ⚠️ **必须步骤**：默认的 RPA v3 kernel 不支持 Gemma4 的大 head_dim（256/512），
> 会导致 TP=1 VMEM OOM 和 TP=4 XLA layout 错误。必须应用 Batched RPA patches。

### 所需 Patches

| 修复项 | 来源 | 说明 |
|--------|------|------|
| Batched RPA Gemma4 layout fix | PR [#2506](https://github.com/vllm-project/tpu-inference/pull/2506) | 修复 XLA Reshape layout 错误 |
| K/V_proj sharding fix | PR [#2585](https://github.com/vllm-project/tpu-inference/pull/2585) | 修复 K/V 投影分片 |
| n_buffer 3→2 | 手动 patch | Batched RPA 默认 n_buffer=3 对大 head_dim OOM，改为 2 |
| `USE_BATCHED_RPA_KERNEL=1` | 环境变量 | 启用 VMEM-aware 的 Batched RPA kernel |

### 应用方法

```bash
# 从 Lustre 复制已准备好的 patches
PATCH=/lustre/patches/gemma4
TPI=/workspace/tpu_inference/tpu_inference

kubectl --context=$CTX exec $POD -- bash -c "
  cp $PATCH/gemma4.py $PATCH/gemma4_mm.py $TPI/models/jax/
  cp $PATCH/attention_interface.py $TPI/layers/common/
  mkdir -p $TPI/kernels/experimental/batched_rpa
  cp $PATCH/batched_rpa/__init__.py $PATCH/batched_rpa/wrapper.py $TPI/kernels/experimental/batched_rpa/
  cp $PATCH/rpa_v3/kernel.py $TPI/kernels/ragged_paged_attention/v3/

  # 关键：n_buffer 3→2（防止大 head_dim VMEM OOM）
  sed -i 's/n_buffer = 3/n_buffer = 2/' $TPI/kernels/experimental/batched_rpa/wrapper.py

  # 清除 Python cache
  find $TPI -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null
  echo 'Patches applied successfully'
"
```

> 💡 **Patches 来源**: 从 tpu-inference main branch 下载对应 PR 的文件，存到 `/lustre/patches/gemma4/`。
> 当 nightly 镜像更新到包含这些 PR 后，此步骤可跳过。

## Step 5: 启动 vLLM 推理服务

### 关键参数说明

| 参数 | TP=1 | TP=4 | 说明 |
|------|------|------|------|
| `--tensor-parallel-size` | `1` | `4` | TP=4 推荐，不浪费 chip |
| `--max-model-len` | `4096` | `4096` | 按场景调整（最大稳定 32768） |
| `--max-num-seqs` | `160` | `256` | FP8 KV Cache 下的推荐值 |
| `--kv-cache-dtype` | `fp8` | `fp8` | **推荐**，KV Cache 减半 |
| `--no-enable-prefix-caching` | 必须 | 必须 | 避免 prefix caching bug |
| `--gpu-memory-utilization` | `0.9` | `0.9` | 单 chip 192GB × 0.9 = 172GB |
| `--block-size` | `256` | `256` | CI 默认值 |
| `--trust-remote-code` | 必须 | 必须 | Gemma4 自定义模型代码 |

### 环境变量

| 变量 | 值 | 说明 |
|------|------|------|
| `SKIP_JAX_PRECOMPILE` | `1` | 跳过 JAX 预编译，加速启动 |
| `VLLM_XLA_CHECK_RECOMPILATION` | `0` | 关闭 XLA 重编译检查 |
| `USE_BATCHED_RPA_KERNEL` | `1` | **必须**，启用 Batched RPA kernel |

> ⚠️ Gemma4 使用 **JAX native** 实现（不是 PyTorch/TorchAX path），
> 所以**不需要** `MODEL_IMPL_TYPE=vllm`（那是 MoE / PyTorch 模型用的）。

### 启动服务（TP=4 推荐配置）

```bash
cat > /tmp/launch_gemma4.sh <<'LAUNCHER'
#!/bin/bash
cd /tmp
pgrep -f 'EngineCore|vllm' | xargs -r kill -9 2>/dev/null
sleep 2
rm -f /tmp/libtpu_lockfile /tmp/vllm_gemma4.log
touch /tmp/vllm_gemma4.log
setsid nohup env \
  SKIP_JAX_PRECOMPILE=1 VLLM_XLA_CHECK_RECOMPILATION=0 USE_BATCHED_RPA_KERNEL=1 \
  vllm serve /lustre/models/gemma-4-31b-it \
    --tensor-parallel-size 4 \
    --max-num-batched-tokens 4096 --max-num-seqs 256 --max-model-len 4096 \
    --no-enable-prefix-caching --gpu-memory-utilization 0.9 \
    --kv-cache-dtype fp8 --block-size 256 --trust-remote-code \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    --async-scheduling \
    >> /tmp/vllm_gemma4.log 2>&1 < /dev/null &
disown
echo "launched pid=$!"
exit 0
LAUNCHER

kubectl --context=$CTX cp /tmp/launch_gemma4.sh $POD:/tmp/launch_gemma4.sh
kubectl --context=$CTX exec $POD -- bash /tmp/launch_gemma4.sh

# 监控启动 (~3-5 min cold start)
kubectl --context=$CTX exec $POD -- tail -f /tmp/vllm_gemma4.log
```

**等待关键日志**：
```
INFO: Application startup complete.                    ← 启动成功
```

## Step 6: 验证推理

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

## Step 7: 性能 Benchmark

### 测试矩阵

| 编号 | 场景 | Input Len | Output Len | 并发 | max-model-len | 目标指标 |
|------|------|-----------|------------|------|---------------|---------|
| B1 | **单用户延迟** | 1K | 1K | P1 | 4096 | TTFT, TPOT |
| B2 | **标准吞吐** | 1K | 1K | P64-P128 | 4096 | tok/s |
| B3 | **峰值吞吐** | 1K | 1K | P256 | 4096 | tok/s (max batch) |
| B4 | **长输入短输出** | 16K | 1K | P1-P16 | 32768 | TTFT, total tok/s |
| B5 | **短输入长输出** | 1K | 16K | P4 | 32768 | TPOT stability |
| B6 | **128K 长 context** | 128K | 256 | P1 | 131072 | TTFT, 是否 OOM |
| B7 | **256K 最大 context** | 256K | 256 | P1 | 262144 | TTFT, 是否 OOM |

> ⚠️ B6/B7 长 context 测试需要**重启 vLLM 并调整 `--max-model-len`**。
> 当前已知 **>32K context 不稳定**（Pallas kernel crash），详见 Benchmark 结果。

### 7.1 短 Context 测试 (B1-B3, max-model-len=4096)

使用 Step 5 的默认启动参数，无需重启。

```bash
# B1: 单用户延迟 (1K/1K, P1)
kubectl --context=$CTX exec $POD -- timeout 600 vllm bench serve \
  --model /lustre/models/gemma-4-31b-it \
  --dataset-name random --random-input-len 1024 --random-output-len 1024 \
  --num-prompts 3 --max-concurrency 1 2>&1 | tail -30

# B2: 标准吞吐 (1K/1K, P64)
kubectl --context=$CTX exec $POD -- timeout 600 vllm bench serve \
  --model /lustre/models/gemma-4-31b-it \
  --dataset-name random --random-input-len 1024 --random-output-len 1024 \
  --num-prompts 128 --max-concurrency 64 2>&1 | tail -30

# B2: 标准吞吐 (1K/1K, P128)
kubectl --context=$CTX exec $POD -- timeout 600 vllm bench serve \
  --model /lustre/models/gemma-4-31b-it \
  --dataset-name random --random-input-len 1024 --random-output-len 1024 \
  --num-prompts 256 --max-concurrency 128 2>&1 | tail -30

# B3: 峰值吞吐 (1K/1K, P256)
kubectl --context=$CTX exec $POD -- timeout 600 vllm bench serve \
  --model /lustre/models/gemma-4-31b-it \
  --dataset-name random --random-input-len 1024 --random-output-len 1024 \
  --num-prompts 512 --max-concurrency 256 2>&1 | tail -30
```

### 7.2 中长 Context 测试 (B4-B5, max-model-len=32768)

需要重启 vLLM（修改 `--max-model-len 32768`，`--max-num-seqs` 下调）。

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
  SKIP_JAX_PRECOMPILE=1 VLLM_XLA_CHECK_RECOMPILATION=0 USE_BATCHED_RPA_KERNEL=1 \
  vllm serve /lustre/models/gemma-4-31b-it \
    --tensor-parallel-size 4 \
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
kubectl --context=$CTX exec $POD -- timeout 600 vllm bench serve \
  --model /lustre/models/gemma-4-31b-it \
  --dataset-name random --random-input-len 16384 --random-output-len 1024 \
  --num-prompts 12 --max-concurrency 4 2>&1 | tail -30

# B4: 长输入短输出 (16K/1K, P16)
kubectl --context=$CTX exec $POD -- timeout 600 vllm bench serve \
  --model /lustre/models/gemma-4-31b-it \
  --dataset-name random --random-input-len 16384 --random-output-len 1024 \
  --num-prompts 48 --max-concurrency 16 2>&1 | tail -30

# B5: 短输入长输出 (1K/16K, P4)
kubectl --context=$CTX exec $POD -- timeout 900 vllm bench serve \
  --model /lustre/models/gemma-4-31b-it \
  --dataset-name random --random-input-len 1024 --random-output-len 16384 \
  --num-prompts 4 --max-concurrency 4 2>&1 | tail -30
```

### 7.3 超长 Context 测试 (B6-B7, 128K / 256K)

> ⚠️ **已知限制**：64K+ context 当前触发 Pallas kernel crash / TPU driver SIGABRT。
> 此测试用于验证上游修复进展，预期会失败。

```bash
# 重启 vLLM with max-model-len=131072
cat > /tmp/launch_gemma4_128k.sh <<'LAUNCHER'
#!/bin/bash
cd /tmp
pgrep -f 'EngineCore|vllm' | xargs -r kill -9 2>/dev/null
sleep 2
rm -f /tmp/libtpu_lockfile /tmp/vllm_gemma4.log
touch /tmp/vllm_gemma4.log
setsid nohup env \
  SKIP_JAX_PRECOMPILE=1 VLLM_XLA_CHECK_RECOMPILATION=0 USE_BATCHED_RPA_KERNEL=1 \
  vllm serve /lustre/models/gemma-4-31b-it \
    --tensor-parallel-size 4 \
    --max-num-batched-tokens 65536 --max-num-seqs 1 --max-model-len 131072 \
    --no-enable-prefix-caching --gpu-memory-utilization 0.95 \
    --kv-cache-dtype fp8 --block-size 256 --trust-remote-code \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    --async-scheduling \
    >> /tmp/vllm_gemma4.log 2>&1 < /dev/null &
disown
echo "launched pid=$!"
exit 0
LAUNCHER
kubectl --context=$CTX cp /tmp/launch_gemma4_128k.sh $POD:/tmp/launch_gemma4_128k.sh
kubectl --context=$CTX exec $POD -- bash /tmp/launch_gemma4_128k.sh

# 等 cold start (XLA 编译 128K shape 可能更久)
for i in $(seq 1 40); do C=$(kubectl --context=$CTX exec $POD -- curl -sf -o /dev/null -w "%{http_code}" http://localhost:8000/health); echo "T+$((i*30))s HTTP $C"; [ "$C" = "200" ] && break; sleep 30; done

# B6: 64K context (64K/256, P1)
kubectl --context=$CTX exec $POD -- timeout 600 vllm bench serve \
  --model /lustre/models/gemma-4-31b-it \
  --dataset-name random --random-input-len 65536 --random-output-len 256 \
  --num-prompts 1 --max-concurrency 1 2>&1 | tail -30

# B6: 128K context (128K/256, P1)
kubectl --context=$CTX exec $POD -- timeout 600 vllm bench serve \
  --model /lustre/models/gemma-4-31b-it \
  --dataset-name random --random-input-len 131072 --random-output-len 256 \
  --num-prompts 1 --max-concurrency 1 2>&1 | tail -30
```

## Step 8: 清理

```bash
# 删除 pod
kubectl --context=$CTX delete pod gemma4-31b

# 删除 node pool (可选)
gcloud container node-pools delete np-tpu7xe-spot-gemma4 \
  --cluster=$CLUSTER --region=$REGION --project=$PROJECT --quiet --async
```

---

## 📊 Benchmark 结果

> **测试环境**: vllm/vllm-tpu:nightly (v0.20.2rc1.dev223), USE_BATCHED_RPA_KERNEL=1, FP8 KV Cache, BF16 weights, n_buffer=2
>
> **测试工具**: Round 2 使用 evalscope, Round 3 使用 `vllm bench serve`（vLLM 内置）

### TP=1 单 Chip 性能 (1K/1K, max-model-len=4096)

| 并发 | 请求数 | 成功率 | TTFT (ms) | TPOT (ms) | 吞吐 (tok/s) |
|------|--------|--------|-----------|-----------|-------------|
| P1 | 3 | 100% | 40 | — | 7.2 |
| P8 | 16 | 100% | 12,137 | 48.5 | 59.7 |
| P32 | 64 | 100% | 3,261 | 114.8 | 173.2 |
| **P64** | **128** | **100%** | **5,723** | **84.7** | **447** ⭐ |
| P128 | 256 | 100% | 21,865 | 90.8 | 407 |

**TP=1 峰值吞吐：447 tok/s @ P64**

### TP=4 四 Chip 性能

#### 短 Context (1K/1K, max-model-len=4096)

| 编号 | 并发 | TTFT (ms) | TPOT (ms) | Output tok/s | 状态 |
|------|------|-----------|-----------|-------------|------|
| B1 | P1 | 7,667 | 41.0 | 10.5 | ✅ |
| B2 | P64 | 15,101 | 125.3 | 1,873 | ✅ |
| B2 | P128 | 13,582 | 148.3 | 2,865 | ✅ |
| B3 | **P256** | **5,551** | **253.5** | **4,394** ⭐ | ✅ |

#### 长 Context (16K, max-model-len=32768)

| 编号 | 场景 | 并发 | TTFT (ms) | TPOT (ms) | Output tok/s | Total tok/s | 状态 |
|------|------|------|-----------|-----------|-------------|-------------|------|
| B4 | 16K/1K | P1 | 16,215 | — | 17.5 | 297 | ✅ |
| B4 | 16K/1K | P4 | 7,830 | 45.5 | 77.2 | 1,314 | ✅ |
| B4 | 16K/1K | P8 | 7,226 | — | 142.5 | 2,424 | ✅ |
| B4 | 16K/1K | **P16** | **8,087** | — | **246.4** | **4,192** ⭐ | ✅ |
| B5 | 1K/16K | P4 | 1,051 | 44.7 | 89.3 | — | ✅ |

> 💡 **B5 长 decode 特性**: TTFT 很快（1K prefill 仅 1s），TPOT 稳定 44.7ms。ITL P99 达 519ms（周期性 KV cache 管理开销）。

#### 超长 Context (64K-256K)

| 编号 | 场景 | 配置 | 状态 | 错误 |
|------|------|------|------|------|
| B6 | 64K/256 P1 | TP=4, 131K ctx | ❌ | FAILED_PRECONDITION: program continuator halt |
| B6 | 128K/256 P1 | TP=4, 131K ctx | ❌ | E0200: RuntimeUnexpectedCoreHalt (SIGABRT) |
| B6 | 128K/256 P1 | TP=1, 131K ctx | ❌ | KV cache 不足 (需 55GB, 只有 31.75GB) |
| B7 | 256K/256 P1 | — | ⏭️ 跳过 | 128K 已崩，256K 无法测试 |

> ⚠️ **最大稳定 context: 32K (max-model-len=32768)**。
> 64K+ context 触发 Batched RPA Pallas kernel 在 TPU driver 层的断言失败（非 HBM OOM，是计算执行错误）。
> 需要等待 LIBTPU/Pallas kernel 对 hybrid attention 大 head_dim 的优化。

### TP=1 vs TP=4 对比

| 指标 | TP=1 (单 chip) | TP=4 (4 chips) | 提升 |
|------|---------------|---------------|------|
| 峰值吞吐 | 447 tok/s @ P64 | 4,394 tok/s @ P256 | **9.8x** |
| 单用户 TPOT | ~46 ms | ~41 ms | 12% |
| 最大成功并发 | P128 | P256+ | 2x+ |

> 💡 **TP=4 吞吐量是 TP=1 的 ~10x**，接近线性扩展。
> **建议生产部署用 TP=4**，GKE 最小 4 chips，用 TP=1 浪费 3 个 chip。

---

## 📋 Troubleshooting

### 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| `RESOURCE_EXHAUSTED` 创建 node pool | v7xe spot 容量不足 | 换 zone 或等待 |
| 权重下载 403 | HF token 无 Gemma4 访问权限 | 去 HuggingFace 接受 license |
| 启动后 hang / 无 log | libtpu lockfile 残留 | `rm -f /tmp/libtpu_lockfile` |
| TP=1 VMEM OOM | 未应用 Batched RPA patches | 执行 Step 4 |
| TP=4 XLA Reshape layout | 未应用 Batched RPA patches | 执行 Step 4 |
| 输出重复 token | 已知 bug vllm#39827 | 降低 temperature 或等待上游修复 |
| `ImportError: gemma4` | tpu-inference 版本太旧 | 确保用最新 nightly image |
| 64K+ context crash | Pallas kernel bug | 当前无解，限制 max-model-len ≤ 32768 |

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

### 踩坑记录

| # | 问题 | 解决 |
|---|------|------|
| 1 | GKE machine type 名称 | 正确名称: `tpu7x-standard-4t`（非 `ct7xe-standard-4t`） |
| 2 | GKE accelerator 标签 | 正确标签: `tpu7x`（非 `tpu-v7xe-slice` / `tpu-v7-lite-podslice`） |
| 3 | Pod 无 entrypoint 自动退出 | 加 `command: ["sleep", "infinity"]` |
| 4 | nightly image 不在 cloud-tpu-images | 用 Docker Hub `vllm/vllm-tpu:nightly` |
| 5 | evalscope random dataset 参数变更 | 改用 `vllm bench serve --dataset-name random` |
| 6 | TP=4 128K HBM 差 537MB | 降低 `max-num-batched-tokens` 到 65536 可启动（但推理仍 crash） |
| 7 | 64K context Round 2 "通过" | 误报 — server 请求后实际也崩了，Round 3 确认 |

---

## 📎 参考

- [Gemma 4 Model Card](https://ai.google.dev/gemma/docs/core/model_card_4) — 官方模型规格
- [vllm-project/tpu-inference](https://github.com/vllm-project/tpu-inference) — TPU 推理后端
- [Gemma4 Issues](https://github.com/vllm-project/tpu-inference/issues?q=gemma4) — 已知问题追踪
- [HuggingFace: google/gemma-4-31b-it](https://huggingface.co/google/gemma-4-31b-it) — 模型权重
- PR [#2506](https://github.com/vllm-project/tpu-inference/pull/2506) — Batched RPA Gemma4 layout fix
- PR [#2585](https://github.com/vllm-project/tpu-inference/pull/2585) — K/V_proj sharding fix

---

> **文档版本**: v1.0
>
> **最后更新**: 2026-05-14
