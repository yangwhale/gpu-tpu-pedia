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
    --max-num-batched-tokens 4096 --max-num-seqs 64 \
    --no-enable-prefix-caching --gpu-memory-utilization 0.9 \
    --block-size 256 --trust-remote-code \
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
| `--max-model-len` | `4096` | 初始测试用短 context |
| `--max-num-seqs` | `64` | 初始测试用小并发 |
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
    --max-num-batched-tokens 4096 --max-num-seqs 64 --max-model-len 4096 \
    --no-enable-prefix-caching --gpu-memory-utilization 0.9 \
    --block-size 256 --trust-remote-code \
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

## Step 6: 性能 Benchmark (可选)

### 单用户延迟

```bash
kubectl --context=$CTX exec $POD -- curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/lustre/models/gemma-4-31b-it",
    "messages": [{"role": "user", "content": "Write a 200-word essay about artificial intelligence."}],
    "max_tokens": 300,
    "temperature": 0.7
  }' | python3 -c '
import sys, json
r = json.load(sys.stdin)
u = r.get("usage", {})
print(f"Prompt tokens: {u.get(\"prompt_tokens\", \"?\")}")
print(f"Completion tokens: {u.get(\"completion_tokens\", \"?\")}")
print(f"Content: {r["choices"][0]["message"]["content"][:100]}...")
'
```

### 吞吐量测试 (使用 vLLM benchmark 工具)

```bash
# 如果 pod 内有 benchmark_serving.py
kubectl --context=$CTX exec $POD -- python3 -m vllm.entrypoints.openai.run_batch \
  --model /lustre/models/gemma-4-31b-it \
  --input-len 128 --output-len 128 \
  --num-prompts 64
```

> 📊 **性能数据待实测填写**。31B Dense + 单 chip + BF16 预期：
> - Cold start: ~2-3 min（权重小，无 MoE re-quant）
> - 单用户延迟: 待测
> - 吞吐量: 待测

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

> **文档版本**: v0.1 (初始版本, 待实测验证)
>
> **最后更新**: 2026-05-13
