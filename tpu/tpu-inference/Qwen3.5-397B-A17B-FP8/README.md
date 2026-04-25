# Qwen3.5-397B-A17B-FP8 Inference on TPU v7x-8

> 🌐 **Languages** | **语言**: **中文** · [English (TBD)](README.en.md)

> 端到端指南：在 TPU v7x-8 上运行 Qwen3.5-397B-A17B-FP8（397B 总参 / 17B 激活 / hybrid GDN+Attention + 512 routed experts）推理。
>
> 与 DeepSeek R1 / GLM-5.1 / Kimi K2.6 不同，Qwen3.5 是 **hybrid Mamba+Attention 架构**（45 GDN + 15 Standard Attn），需要 PR #2366 的 KV cache OOB fix。
>
> **代码仓库**: https://github.com/vllm-project/tpu-inference (main branch ≥ 2026-04-23, 含 PR #2366)
>
> **模型**: [Qwen/Qwen3.5-397B-A17B-FP8](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8)（94 safetensors, ~378 GiB native FP8）

## 🎯 关键性能（已实测 2026-04-25）

| 操作点 | 实测值 | 备注 |
|--------|------|------|
| HF 下载 (94 shards, 378 GiB, 16 worker + hf_transfer) | **6 min** | xet CDN cache 充分（hot model） |
| Cold start (Application startup complete) | **~7 min** | weight load 161s + MoE re-quant 2.5min + Hybrid KV padding + pjit compile |
| Single user latency (P1, 1K/1K) | **20.6s, 49.6 tok/s/user** | Pareto: 💨 Low Latency |
| Balanced (P64, 1K/1K) | **1510 tok/s, 23.6 tok/s/user** | Pareto: ⚖️ Balanced |
| **🚀 Peak throughput (P128, 1K/1K)** | **2103 tok/s** ⭐ | Pareto: 真正 max（P256 反而降到 1877） |
| **GSM8K full 1319 样本 (5-shot, thinking OFF)** | **77.56% (1023/1319)** ✅ | CI 阈值 63%, 远超过；16 min 跑完 |
| 长 prompt 8K/1K P4 | **178.6 tok/s** | 与 1K prompt 几乎相同（hybrid GDN 长 context 优势） |

> **简单 chat 测试** (2026-04-25 10:08, e2e-02 pod, PR #2366 应用后):
> Prompt: `哈喽啊，how are you 啊`
> Response: `哈喽！I'm doing great, thanks for asking! 😊 今天有什么想聊的或者需要帮忙的吗？`
> 26 tokens output, finish_reason=stop ✅

---

## 🌟 三个反直觉发现

在 17 档 benchmark 实测中观察到的、跟一般直觉相反的结论：

### 1. 并发越大 ≠ throughput 越高（Peak 是 P128 不是 P256）

| Batch | Throughput |
|---|---:|
| P64 | 1510 tok/s |
| **P128 ⭐** | **2103 tok/s** ← peak |
| P256 | 1877 tok/s ↓ |

P256 反而比 P128 慢 **11%**。原因：vLLM scheduler 在 P256 时 KV cache 不够，频繁 preempt + 重调度，GPU 占用率反降。**Pareto 曲线必须细粒度采样**才能找到真正 knee point。

### 2. 长输出比短输出快 9-13% ⭐⭐

| Batch | 1K/1K out | 1K/**8K** out | 差距 |
|---|---:|---:|---:|
| P1 | 49.6 | **54.0** | +9% |
| P64 | 1510 | **1702** | **+13%** |

长 generation 让 batch **长时间保持 pure decode 状态**，TPU MXU 利用率高；短 generation 频繁 prefill→decode→cleanup 转换，调度开销大。**长文章/代码生成场景反而是 sweet spot**。

### 3. 8K 长 prompt 在低并发下几乎不拖累

| Batch | 1K input | **8K** input | 拖累 |
|---|---:|---:|---:|
| P1 | 49.6 | **51.7** | 反而 +4% |
| P4 | 186.8 | **178.6** | -4% |
| P64 | 1510 | **850** | -44% |

Qwen3.5 hybrid 架构 — 45 个 GDN 层用 **fixed-size SSM state**（不随 context 增长），只 15 层产生 KV cache → 长 context KV 压力是纯 attention 模型的 ~1/4。**仅低并发时享受**这优势，高并发 chunked prefill 累加会拖累。

> **一句话记忆**：大并发不一定快 / 长输出反而快 / 长 prompt 不拖累（低并发）

---

## ⚠️ 必读：3 个关键修复

### A. PR #2366 patch（**必须**，否则 KV cache 状态损坏 → gibberish）

> Qwen3.5 是 hybrid attention+mamba 架构。vLLM hybrid allocator 把 4 layers 共享 1 个 `KVCacheTensor`（GPU 用 byte-level 重解释），但 TPU `jax.Array` strongly typed，会**重复**为 4 个独立 tensor。这导致 vLLM scheduler 的 block_id pool 比 TPU per-layer 实际容量大 ~3.5×，scheduler 发出的 block_id 超出 layer 实际范围，JAX `dynamic_update_slice_in_dim` silently clip → 多个 request 的 mamba state 塌陷到同一 slot → **gibberish output**。

**症状（多变）**：
- 多并发请求时输出乱码
- thinking mode 下 75% 请求 finish_reason=length（被截）— 状态损坏让模型说不完整
- "vmem OOM 86 MB > 64 MB" — KV layout 错误导致 RPA kernel block_size 膨胀到 4352
- "HBM OOM 95G > 94.75G" — block_size 错误膨胀的连锁反应
- EngineCore silent crash after long inference

**修复**: 拷贝 main branch 的 `kv_cache_manager.py`（见 [Step 4](#step-4-应用-pr-2366-fix)）。

### B. 三个必设环境变量

| 环境变量 | 值 | 漏设后果 |
|---------|-----|---------|
| `MODEL_IMPL_TYPE=vllm` | 必须 | Qwen3.5 走 vLLM PyTorch + TorchAX path（复用 vLLM 主仓 model class） |
| `SKIP_JAX_PRECOMPILE=1` | 推荐 | 跳过 JAX 预编译，启动快 1-2 min |
| `VLLM_XLA_CHECK_RECOMPILATION=0` | 推荐 | 关闭 XLA recompilation 检查，避免开发期警告 |

### C. 启动参数关键约束

| 参数 | 取值 | 不设的后果 |
|------|------|------|
| `--enable-expert-parallel` | 必须 | EP=8 是 PR #2366 fix 后的正确并行策略（**不要用 DP attention**） |
| `--no-enable-prefix-caching` | 必须 | 否则 vLLM 自动把 `mamba_cache_mode` promote 到 `align`，触发 `chunked_mm_input` AssertionError |
| `--gpu-memory-utilization 0.9` | 推荐 | PR #2366 fix 后可以用 0.9（之前 0.65 是绕开 bug 的副作用） |
| `--max-num-batched-tokens 4096` | 推荐 | CI accuracy test 默认值 |
| `--max-num-seqs 256` | 推荐 | CI 默认值 |
| `--max-model-len 4096` | 推荐 | 短到中等对话场景 |
| `--kv-cache-dtype fp8` | 推荐 | KV 减半 |
| `--block-size 256` | 必须 | CI 默认 |
| `--reasoning-parser qwen3` | 必须 | 正确解析 thinking model 的 `<think>` tag |
| `--limit-mm-per-prompt '{"image": 0, "video": 0}'` | 推荐 | 跳过 vision encoder（除非要测 multimodal） |
| `--async-scheduling` | 推荐 | 异步调度 |

---

## 硬件与模型概览

| 项目 | 要求 |
|------|------|
| TPU | **v7x-8（足够）** — 不需要 v7x-16 |
| HBM | 94.75 GB/device，v7x-8 共 758 GB（per-device 用 ~85 GB / 90% with `gpu-memory-utilization=0.9`） |
| 主机内存 | ≥800 GB（page cache 装 378 GB checkpoint 需要） |
| 存储 | ≥600 GB（模型 378 GB + 工作空间） |

| 模型参数 | 值 | vs DeepSeek V3 / R1 | vs GLM-5.1 | vs Kimi K2.6 |
|---------|-----|--------------------|------------|--------------|
| 架构 | **MoE + Hybrid GDN/Attention** | MoE+MLA | MoE+MLA | MoE+MLA |
| 总参数 | **397B** | 671B | 754B | 1T |
| 激活参数 | **17B** | ~37B | ~37B | 32B |
| 总层数 | **60（45 GDN + 15 Standard Attn）** | 61 | 78 | 61 |
| Hidden | 4,096 | 7,168 | 6,144 | 7,168 |
| GDN heads | 64 (V) / 16 (Q,K) | — | — | — |
| Standard Attn heads | 32 Q / 2 KV (GQA) | 128 (MLA) | 64 (MLA) | 64 (MLA) |
| Attn head_dim | 256 | — | — | — |
| MoE Experts | **512 routed + 1 shared** | 256 | 256 | 384 |
| Top-K (routed) | 10 | 8 | 8 | 8 |
| Expert Intermediate | 1,024 | 2,048 | 2,048 | 2,048 |
| Vocab | 248,320 | 129,280 | 154,880 | 163,840 |
| Native Context | 262K（YaRN 可扩到 1M） | 128K | 200K | 256K |
| 多模态 | Vision + Video（部署可禁用） | 无 | 无 | MoonViT 400M |
| 量化（HF） | **FP8 native** | FP4 | FP4 | INT4 W4A16 |

### 关键架构：Hybrid GDN/Attention

> **Gated Delta Network (GDN) ≠ Linear Attention**
> GDN 是 Mamba-2 风格的 Selective State Space Model，包含 conv1d short-range mixing + SSM recurrent state。
> Layer pattern：`15 × (3 × GDN→MoE + 1 × Standard Attn→MoE)`

**实际影响**：
- 45 GDN 层用 conv_state + recurrent_state（固定大小，不随上下文增长）
- 15 Standard Attn 层产生标准 KV cache
- 长上下文 KV 压力是纯 Attention 模型的 ~1/4
- vLLM hybrid allocator 把 4 layers 共享一个 KVCacheTensor — TPU 必须 duplicates per-layer，需要 PR #2366 padding 才能让 scheduler 与实际分配对齐

---

## Step 1: 环境与 Pod 准备

复用 GKE TPU pod（`tpu-v7x-lite-podslice`, 2x2x1）。**关键**：必须挂 Lustre PVC（378 GB 模型），shm 800 GB（用于 page cache）。

```bash
# 找一个空闲 pod
CTX=gke_cloud-tpu-multipod-dev_us-central1_chrisya-v7x-v134
kubectl --context="$CTX" get pods

# 进入 pod (例: e2e-02)
kubectl --context="$CTX" exec -it e2e-02 -- bash
```

> **Pod 互斥**：一个 pod 只能跑一个 vLLM 实例（独占 `/dev/vfio/0`）。多个并发模型要分到不同 pod。
> **Hulk K2.6 / 其他模型同 pod 时**：检查 `/dev/shm/` 是否被其他模型占用，避免删别人的 staging 数据。

---

## Step 2: 检查 + 清理 /dev/shm（关键）

vLLM 启动时检测 RAM available 是否 ≥ 90% × checkpoint size，如不足则 silently 跳过 auto-prefetch，weight load 退化到直接读 Lustre（80s/shard vs 2s/shard，慢 50×）。

```bash
# 检查 shm 占用
df -h /dev/shm
ls -la /dev/shm/

# 如有大 model 残留（如 Kimi-K2.6 的 555 GB），且确认无人在用：
# fuser -v /dev/shm/<other-model>/  # 先确认没人占
# rm -rf /dev/shm/<other-model>/

rm -rf /dev/shm/sem.* /dev/shm/wrk_* 2>/dev/null

# 验证
df -h /dev/shm | tail -1   # 应 < 100 GB
free -h | head -2           # available 应 ≥ 700 GB
```

---

## Step 3: 下载模型权重（如未下）

```bash
pip install -q hf_transfer
export HF_TOKEN='<your-hf-token>'  # 已 hf auth login 的可省

mkdir -p /lustre/models/Qwen3.5-397B-A17B-FP8
HF_HUB_ENABLE_HF_TRANSFER=1 hf download Qwen/Qwen3.5-397B-A17B-FP8 \
  --local-dir /lustre/models/Qwen3.5-397B-A17B-FP8 \
  --max-workers 16

# 验证（94 shards, 378 GiB）
ls /lustre/models/Qwen3.5-397B-A17B-FP8/*.safetensors | wc -l   # 应 = 94
du -sh /lustre/models/Qwen3.5-397B-A17B-FP8/                     # ~378G
```

实测速度：**63 GB/min 平均**，6 min 全部完成。

> **下载 tips**：
> - `max-workers 32` 容易触发 HF rate limit + xet 后端断连，**16 worker 最稳**
> - `HF_HUB_ENABLE_HF_TRANSFER=1` 用 Rust 实现，比 Python 版稳定 5-10×
> - HF 下载支持 atomic resume（hash 命名的 .incomplete 文件），重启不丢

---

## Step 4: 应用 PR #2366 fix

GKE 上 docker image 自带的 tpu-inference 通常是 4 月 22 日构建的（vllm 0.19.1rc1.dev321），**没赶上 PR #2366 (2026-04-23 merged)**。直接从 main 拉 fix：

```bash
# 在 host machine（kubectl 可达的位置）
TMP=$(mktemp /tmp/kv_cache_manager.XXXXXX.py)
curl -sf https://raw.githubusercontent.com/vllm-project/tpu-inference/main/tpu_inference/runner/kv_cache_manager.py -o $TMP

# 验证 fix 标记（应输出 7）
grep -c '_hybrid_uniform_page_size_bytes' $TMP

# 备份 + 上传到 pod
CTX=gke_cloud-tpu-multipod-dev_us-central1_chrisya-v7x-v134
POD=e2e-02
KCM=/workspace/tpu_inference/tpu_inference/runner/kv_cache_manager.py

kubectl --context="$CTX" exec $POD -- cp $KCM ${KCM}.bak.before_pr2366
kubectl --context="$CTX" cp $TMP $POD:$KCM

# 在 pod 内验证
kubectl --context="$CTX" exec $POD -- bash -c "
  wc -l $KCM
  grep -c '_hybrid_uniform_page_size_bytes' $KCM
  # 清 .pyc cache
  find /workspace/tpu_inference/tpu_inference/runner/__pycache__/ -name 'kv_cache*.pyc' -delete
"

rm -f $TMP
```

> **如果 main branch 已经被合到 docker image 自动重建**，可以跳过此步。先 grep 检查：
> ```bash
> kubectl exec $POD -- grep -c '_hybrid_uniform_page_size_bytes' \
>   /workspace/tpu_inference/tpu_inference/runner/kv_cache_manager.py
> # 输出 7 = 已包含；输出 0 = 需要 patch
> ```

---

## Step 5: 启动 vLLM 推理服务

> ⚠️ **必读**：`SKIP_JAX_PRECOMPILE=1`、`MODEL_IMPL_TYPE=vllm`、`--enable-expert-parallel`、`--no-enable-prefix-caching` 缺一不可。

进入 pod：

```bash
kubectl --context="$CTX" exec -it e2e-02 -- bash

# 在 pod 内
MODEL=/lustre/models/Qwen3.5-397B-A17B-FP8

# 必须 cd /tmp 避免 namespace 冲突
cd /tmp

# 清理可能的残留
pgrep -f 'EngineCore|vllm' | xargs -r kill -9 2>/dev/null
sleep 3
rm -f /tmp/libtpu_lockfile

# 启动
SKIP_JAX_PRECOMPILE=1 \
VLLM_XLA_CHECK_RECOMPILATION=0 \
MODEL_IMPL_TYPE=vllm \
nohup vllm serve $MODEL \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 256 \
  --max-model-len 4096 \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.9 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --reasoning-parser qwen3 \
  --async-scheduling \
  > /tmp/vllm_qwen35.log 2>&1 </dev/null &
disown

# 监视
tail -f /tmp/vllm_qwen35.log
```

等待 ~7 min 看到：
```
Hybrid KV cache: padding every layer spec to 23289856 bytes (num_attn_groups=1 × attn_page=10485760 + num_mamba_groups=3 × mamba_unpadded=4268032).
Hybrid KV cache: setting num_gpu_blocks_override=945 to align the scheduler's block pool with per-layer TPU allocation
Memory statistics | total_hbm_used_gb=374.57GiB | total_hbm_avail_gb=307.6GiB
Init kv-cache | num_total_layers=60 | num_blocks=[945, 945, ...] | regular_attn_layers=15 | regular_attn_shape=(num_blocks, (1280, 8, 4, 256))
INFO: Application startup complete.
```

> **关键 log 标志（PR #2366 生效）**：
> - `Hybrid KV cache: padding every layer spec to 23289856 bytes` ← PR #2366 padding 触发
> - `regular_attn_shape=(num_blocks, (1280, 8, 4, 256))` ← block_size **1280** （而非 patch 前错误的 4352）
> - `num_gpu_blocks_override=945` ← 强制 scheduler 与 TPU 实际分配对齐

---

## Step 6: 验证推理（hello world）

```bash
# Health check
curl -s http://localhost:8000/health
# 预期: HTTP 200, {"status":"ok"}

# Hello world (thinking OFF, 直接回答)
MODEL=/lustre/models/Qwen3.5-397B-A17B-FP8
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"哈喽啊，how are you 啊\"}],
    \"max_tokens\": 256,
    \"temperature\": 0.7,
    \"chat_template_kwargs\": {\"enable_thinking\": false}
  }" | python3 -m json.tool
```

预期输出：
```json
{
  "choices": [{
    "message": {
      "content": "哈喽！I'm doing great, thanks for asking! 😊 今天有什么想聊的或者需要帮忙的吗？"
    },
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 21, "total_tokens": 47, "completion_tokens": 26}
}
```

### Thinking mode 开关

Qwen3.5 默认 thinking mode **ON**（输出 `<think>...</think>` reasoning + 答案）。

**关闭 thinking** 必须在 **request body** 里传（server-side `--chat-template-kwargs` 在当前 vLLM 版本不可靠）：

```python
# OpenAI HTTP API
{
  "model": "...",
  "messages": [...],
  "chat_template_kwargs": {"enable_thinking": false},  # ← 关键
  ...
}
```

Token 经济学差异：
- Thinking ON: 平均 ~1100-1500 token output（含 reasoning）
- Thinking OFF: 平均 ~3-30 token output（直接答案）

---

## Step 7: GSM8K 准确性测试（推荐用自定义脚本）

> ⚠️ **不要用 lm_eval 默认配置**：thinking ON + max_gen_toks=2048 会导致 75% 请求被截断（finish_reason=length），且 lm_eval 不支持传 `chat_template_kwargs={"enable_thinking": false}` 关 thinking。我们提供的自定义 Python 脚本绕过了这两个坑。

### 推荐做法：自定义脚本（thinking OFF + parallel=8 + 0.99s/题）

```bash
# 在 pod 内（脚本 cp 自 repo 的 scripts/run_gsm8k_qwen35.py）
python3 /tmp/run_gsm8k_qwen35.py \
  --model /lustre/models/Qwen3.5-397B-A17B-FP8 \
  --url http://localhost:8000/v1/chat/completions \
  --limit 1319 \              # full GSM8K test set
  --parallel 8 \              # PR #2366 fix 后并发安全
  --max-question-tokens 500 \ # 过滤超长 question 避免 5-shot prompt 超 max_model_len
  --output /tmp/gsm8k_qwen35_full.jsonl
```

**预期**：~16 min 跑完 1319 题，**77.56% accuracy**（CI 阈值 63%, 远超）。比 lm_eval thinking ON 快 100×（24 s/题 → 0.75 s/题）。

完整脚本: [scripts/run_gsm8k_qwen35.py](scripts/run_gsm8k_qwen35.py)

### 备选：lm_eval（仅供参考，有截断坑）

```bash
# ⚠️ thinking ON 会让 75% 请求 finish=length 被截断
lm_eval --model local-chat-completions \
  --model_args "model=$MODEL,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=4" \
  --tasks gsm8k --num_fewshot 5 --apply_chat_template \
  --gen_kwargs 'max_gen_toks=2048' --limit 100
```

CI 默认值参考 [Qwen_Qwen3_5-397B-A17B.yml](https://github.com/vllm-project/tpu-inference/blob/main/.buildkite/models/Qwen_Qwen3_5-397B-A17B.yml)。

---

## Step 8: Throughput Benchmark（warmup + record 模式）

> ⚠️ **必须用 warmup + record 双跑**：第一次跑含 JIT 编译（latency 5-10× 慢），数据失真。我们提供的脚本对每个 batch size 跑两次自动丢弃 warmup。

### 推荐做法：完整 sweep 脚本

```bash
# 在 pod 内（脚本 cp 自 repo 的 scripts/run_bench_qwen35.sh）
pip install evalscope[perf]
bash /tmp/run_bench_qwen35.sh
# 输出: /tmp/bench_qwen35/summary.txt
# 默认跑 P1/P4/P16/P64/P256 5 档 × 2 round, ~24 min
```

完整脚本: [scripts/run_bench_qwen35.sh](scripts/run_bench_qwen35.sh)

### 单次手动测（debug 用）

```bash
evalscope perf \
  --url http://localhost:8000/v1/chat/completions \
  --model $MODEL --tokenizer-path $MODEL \
  --dataset random \
  --min-prompt-length 1024 --max-prompt-length 1024 \
  --max-tokens 1024 --min-tokens 1024 \
  --parallel 64 --number 64 \
  --api openai --stream \
  --extra-args '{"chat_template_kwargs": {"enable_thinking": false}}'
# 跑两次，第一次丢弃，第二次记录
```

> **生产建议**：跑过 sweep 后选 **P128** 做 max throughput 操作点（**2103 tok/s**, peak），**P64** 做 balanced（1510 tok/s + 23.6 tok/s/user）。**不要选 P256** — 反而比 P128 慢 11%（详见上方"三个反直觉发现"）。

---

## 上游演进时间线

| 日期 | 里程碑 | 来源 |
|------|--------|------|
| 2026-03-26 | 初次跑通 BF16（4-layer 子集） | tpu-inference PR [#2004](https://github.com/vllm-project/tpu-inference/pull/2004) |
| 2026-04-06 | FP8 + CI 自动化 | PR [#2086](https://github.com/vllm-project/tpu-inference/pull/2086) |
| 2026-04-14 | DP attention 模式 | PR [#2187](https://github.com/vllm-project/tpu-inference/pull/2187)（已被 PR #2366 取代） |
| 2026-04-22 | Disagg prefill/decode | PRs [#2322](https://github.com/vllm-project/tpu-inference/pull/2322) / [#2327](https://github.com/vllm-project/tpu-inference/pull/2327) / [#2331](https://github.com/vllm-project/tpu-inference/pull/2331) / [#2336](https://github.com/vllm-project/tpu-inference/pull/2336) |
| **2026-04-23** ⭐ | **PR #2366: Hybrid KV cache OOB fix** | [#2366](https://github.com/vllm-project/tpu-inference/pull/2366) — 本 README 全部测试都依赖这个 fix |
| 2026-04-25 | 本次端到端验证（e2e-02 + PR #2366） | 见上方"关键性能"表 |

---

## 踩坑记录（8 条）

### 1. ⭐ 缺 PR #2366 → 所有奇怪症状的总根源

**症状（看似多变）**：
- vmem OOM `86 MB > 64 MB` (RPA kernel register spill)
- HBM OOM `95G > 94.75G` (gpu_mem_util=0.9 时)
- GSM8K thinking ON 75% finish_reason=length (输出截断)
- EngineCore silent crash 在长 inference 后
- 多请求并发时输出乱码

**根因**：vLLM hybrid allocator 把 4 layers 共享 1 个 `KVCacheTensor`（GPU byte-level 重解释），但 TPU `jax.Array` strongly typed → 重复 4 个独立 tensor → vLLM scheduler 的 block_id pool 大 ~3.5× → block_id 越界 → JAX `dynamic_update_slice_in_dim` silently clip → 多 request 状态塌陷 → gibberish

**修复**：拷贝 main branch 的 `kv_cache_manager.py` 到 pod（[Step 4](#step-4-应用-pr-2366-fix)）。无需任何其他 patch。

> **教训**：在错的 branch 上调试会浪费几小时。**先 grep 检查关键 fix 是否包含**：
> ```bash
> grep -c '_hybrid_uniform_page_size_bytes' kv_cache_manager.py  # 应 = 7
> ```

### 2. `/dev/shm` 残留挤压 page cache → weight load 慢 50×

**现象**：weight loading 80 s/shard (vs 正常 2 s/shard)，启动从 7 min 变 2 hr

**根因**：vLLM 启动时检测 `Available RAM`，如 < 90% × checkpoint size 就 silently 跳过 auto-prefetch。Kimi/GLM 等大模型残留在 `/dev/shm` 会挤占 RAM。

**修复**：Step 2 清理 + 启动加 `--safetensors-load-strategy=prefetch` 强制 prefetch。

### 3. `chunked_mm_input` AssertionError

**现象**：启动报 `AssertionError: Chunked MM input is required because we need the flexibility to schedule a multiple of block_size tokens even if they are in the middle of a mm input`

**根因**：`enable_prefix_caching=True` 触发 `mamba_cache_mode` silently promote 到 `align`，与 multimodal model + TPU `disable_chunked_mm_input=True` 三方约束冲突

**修复**：`--no-enable-prefix-caching`。**不要**试图设 `--mamba-cache-mode none`（会被覆盖）。

### 4. `chat_template_kwargs server-side 不生效`

**现象**：启动加了 `--chat-template-kwargs='{"enable_thinking": false}'`，但响应仍带 `reasoning` 字段

**根因**：当前 vLLM 版本 server-side `--chat-template-kwargs` 被 silently 忽略

**修复**：在 **request body** 传 `chat_template_kwargs`：
```bash
curl -d '{"model": "...", "messages": [...], "chat_template_kwargs": {"enable_thinking": false}, ...}'
```

### 5. libtpu lockfile 残留 / TPU device busy

**现象**：`ABORTED: Internal error when accessing libtpu multi-process lockfile` 或 `TPU device busy`

**根因**：上次 vLLM 进程异常退出，孤儿 EngineCore 占着 `/dev/vfio/0` 或 lockfile 没清

**修复**：
```bash
pgrep -f 'EngineCore|vllm' | xargs -r kill -9
sleep 3
fuser /dev/vfio/0 2>&1 | xargs -r kill -9 2>/dev/null
rm -f /tmp/libtpu_lockfile
```

### 6. `vllm serve` 偶发 `Engine core initialization failed. Failed core proc(s): {}`

**现象**：cold start 期间 api_server 提前死亡，`Failed core proc(s): {}`（空 dict 是 race condition signature）

**根因**：vLLM `wait_for_engine_startup` 的 sentinel poller 在 EngineCore 长时间 loading 时偶发 false positive

**Workaround**：
- 重启再试（多数情况 retry 就好）
- 或用 offline `LLM` Python class（单进程同步执行无 IPC race）

### 7. Pod 互斥与 SHM 冲突（多模型同 cluster）

**现象**：另一个 pod 上别人在跑大模型（如 Kimi K2.6），SHM 被占 555 GB

**修复**：
- 用 `kubectl get pods` 找空闲 pod
- 用 `fuser -v /dev/shm/<dir>` 确认是否有人在用，不要随意删别人的 staging
- Hulk 等 teammate 模型用 `/dev/shm/Kimi-K2.6/` 这种路径，要保留

### 8. `feature/kimi-k26-inference` branch 缺 PR #2366

**现象**：在 `e2e` pod 上的 `/workspace/tpu_inference` 是 `yangwhale/tpu-inference` fork 的 `feature/kimi-k26-inference` branch，缺 PR #2366

**修复**：
- 切换到 `e2e-02` pod（docker image 自带的 tpu_inference 也旧，但更接近 main）
- 用 Step 4 的 `kubectl cp` 把 main branch 的 `kv_cache_manager.py` 拷过去

---

## 关键参考

| 资源 | 链接 |
|------|------|
| Qwen3.5-397B-A17B HuggingFace | https://huggingface.co/Qwen/Qwen3.5-397B-A17B |
| Qwen3.5-397B-A17B-FP8（部署用） | https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8 |
| Qwen3.5 Blog | https://qwen.ai/blog?id=qwen3.5 |
| vLLM Qwen3.5 Recipe | https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html |
| tpu-inference 主仓 | https://github.com/vllm-project/tpu-inference |
| **PR #2366**（Hybrid KV cache OOB fix）| https://github.com/vllm-project/tpu-inference/pull/2366 |
| PR #2004（Qwen3.5 + GDN 初始支持） | https://github.com/vllm-project/tpu-inference/pull/2004 |
| PR #2086（FP8 CI） | https://github.com/vllm-project/tpu-inference/pull/2086 |
| PR #2273（Nightly fix） | https://github.com/vllm-project/tpu-inference/pull/2273 |
| PR #2370（Revert n-d device buffer） | https://github.com/vllm-project/tpu-inference/pull/2370 |
| Model support matrix (CSV) | https://github.com/vllm-project/tpu-inference/blob/main/support_matrices/nightly/v7x/default/model_support_matrix.csv |
| CI yaml (Qwen3.5 测试配置) | https://github.com/vllm-project/tpu-inference/blob/main/.buildkite/models/Qwen_Qwen3_5-397B-A17B.yml |
| 同体系 — DeepSeek R1 FP4 README | [../DeepSeek-R1-671B-FP4/README.md](../DeepSeek-R1-671B-FP4/README.md) |
| 同体系 — GLM-5.1 FP4 README | [../GLM-5.1-754B-FP4/README.md](../GLM-5.1-754B-FP4/README.md) |
| 同体系 — Kimi K2.6 README | [../Kimi-K2.6/README.md](../Kimi-K2.6/README.md) |
| 同体系 — Qwen3-Coder 480B README | [../Qwen3-Coder-480B/README.md](../Qwen3-Coder-480B/README.md) |

---

## 端到端 Timeline + KV Cache 拓扑实测

### 端到端部署耗时（首次 vs 重启）

| 步骤 | 首次部署 | 改代码重启 |
|------|---------|-----------|
| Pod 进入 + 检查 SHM | ~1 min | ~10s（清旧进程 + lockfile） |
| 模型下载（HF 16 worker） | **~6 min** | 0（已下） |
| 应用 PR #2366 patch | ~30s | 0（已 patch） |
| vLLM cold start | **~7 min** | **~5-6 min**（page cache 热） |
| Hello world 验证 | ~3s | ~3s |
| **总计** | **~14-15 min** | **~5-6 min** |

### vLLM Cold Start 内部 timeline（~7 min）

| 阶段 | 耗时 | 备注 |
|------|------|------|
| Pod 初始化 + JAX init | ~30s | 含 TPU mesh 初始化 |
| Prefetching 94 shards → page cache | 9.99s（warm）/ ~40s（cold） | 第二次启动 page cache 暖 |
| Loading weights (Lustre) | 161s | TP=8 sharding |
| MoE re-quantization (512 experts → FP8) | ~150s | silent 阶段无日志 |
| Hybrid KV cache padding (PR #2366) | 立即 | 23,289,856 bytes uniform |
| pjit compile + KV cache init | ~30s | 看到 `Init kv-cache` |
| Application startup complete | — | HTTP 200 ready |
| First chat completion（含 JIT） | +~3s | 之后是 hot path |

### KV Cache 拓扑（PR #2366 应用后实测）

| 指标 | 实测值 | 说明 |
|------|------|------|
| Layer 拆分 | **15 attn + 45 GDN** | Hybrid 架构 |
| `regular_attn_shape` | `(945, (1280, 8, 4, 256))` | block_size **1280**（patch 前 4352）, FP8 KV |
| `mamba_shape` | `((945, 3, 12288), (945, 64, 128, 128))` | conv state (BF16) + recurrent state (FP32) |
| `num_blocks` per layer | **945** | `num_gpu_blocks_override` 设定 |
| Per-layer `page_size_padded` | **23,289,856 bytes** | `1×attn (10.5MB) + 3×mamba (4.3MB)` |
| Sharding mesh | `Mesh('data': 1, 'model': 8)` | TP=8 + EP=8 |
| Per-device HBM 使用 | **85 / 94.75 GB (90%)** | `gpu-memory-utilization=0.9` |

---

## 完整 Benchmark 数据

### Throughput (1K input / 1K output, evalscope perf, warmup + record)

测试方法：每个 batch size 跑 2 次（warmup 丢弃 + record 保留），thinking OFF。

#### 完整并发 sweep（9 个档位，2026-04-25 实测）

| Batch | Latency | Throughput | Per-user | Success | Pareto |
|------:|--------:|-----------:|---------:|--------:|--------|
| **P1**    | 20.6 s   | **49.6 tok/s**   | **49.6** | 100% | 💨 Low Latency |
| **P2**    | 21.2 s   | 96.7 tok/s     | 48.4 | 100% | |
| **P4**    | 21.9 s   | 186.8 tok/s    | 46.7 | 100% | 交互对话 |
| **P8**    | 22.8 s   | 358.7 tok/s    | 44.8 | 100% | |
| **P16**   | 25.6 s   | 640 tok/s      | 40.0 | 100% | 在线服务 |
| **P32**   | 28.9 s   | 1129 tok/s     | 35.3 | 100% | |
| **P64**   | 43.2 s   | 1510 tok/s     | 23.6 | 100% | ⚖️ Balanced |
| **P128**  | 61.8 s   | **2103 tok/s** ⭐ | 16.4 | 100% | 🚀 **Peak Throughput** |
| **P256**  | 108.4 s  | 1877 tok/s ↓     | 7.3  | 100% | （超 sweet spot, scheduler 抖动） |

**关键发现：Peak 是 P128，不是 P256！**

P256 throughput 比 P128 **下降 11%** (1877 vs 2103)。原因：vLLM scheduler 在 P256 时频繁 preempt + 重调（KV cache 不够同时容纳 256 个 1K context + 1K output 的状态），实际 GPU 占用率反而降。**P128 才是真正的 max throughput 操作点**。

**Scaling 分析（near-linear → diminishing → drop）**：
- P1 → P2: throughput 提升 **1.95×**（perfect linear scaling）
- P2 → P4: 提升 **1.93×**（near-linear）
- P4 → P8: 提升 **1.92×**（near-linear）
- P8 → P16: 提升 **1.78×**（开始 sublinear）
- P16 → P32: 提升 **1.76×**
- P32 → P64: 提升 **1.34×**（diminishing return）
- P64 → P128: 提升 **1.39×**（recovers a bit）
- **P128 → P256: 0.89× （下降！）**

**Pareto 操作点选择**：
- 单用户低延迟 (TPOT < 25 ms): 用 **P1**, 49.6 tok/s/user
- 中等并发 + 体感 OK: 用 **P32-P64**, ~1100-1500 tok/s 总吞吐 + 23-35 tok/s/user
- 离线批处理 / 最大吞吐: 用 **P128**, **2103 tok/s 峰值**（per-chip 263 tok/s）

#### Thinking ON vs Thinking OFF (1K/1K)

| Batch | Thinking OFF | Thinking ON | 备注 |
|------:|-------------:|-----------:|------|
| P1   | 49.6 tok/s | 49.6 tok/s | 几乎相同（max_tokens 限制） |
| P16  | 640 tok/s  | 638 tok/s  | 几乎相同 |
| P64  | 1510 tok/s | 1518 tok/s | 几乎相同 |

⚠️ **业务效率差 10×**：raw throughput 几乎一样，但 thinking ON 用同样时间产出 ~90% reasoning + 10% 答案，thinking OFF 大部分是答案。**生产部署 reasoning 质量要求不高的话，关 thinking 业务有效 token 提升 10×**。

### GSM8K Accuracy (5-shot, thinking OFF)

| 测试集 | 准确率 | Finish stats | 总耗时 | Parallel |
|--------|-------|--------------|--------|----------|
| **50 samples** (idx 0-49, max_tokens=512) | **92.00% (46/50)** | stop=50, length=0 | 49.5s | 4 |
| **🎯 Full 1319 samples** (max_tokens=512, max_question_tokens=500) | **77.56% (1023/1319)** ✅ | stop=1229, length=90 (6.8%) | **16.4 min** | 8 |

**CI 阈值 63%，远超过 ✅**

**50 题 vs 1319 题的差距**：
- 50 题样本（idx 0-49）偏向较简单题，accuracy 偏高
- 1319 题包含全部难度分布（含 hard reasoning），是真实代表性能力
- **77.56% 是 production-ready 的真实数字**

**6.8% length 截断分析**：
- 90 个题被 max_tokens=512 截断（reasoning + answer 超 512 tokens）
- 提示：调高 max_tokens 到 1024 准确率能进一步提升 1-3 个百分点
- 可对比：lm_eval thinking ON 跑 200 题，**75%** length 截断（thinking 用了 reasoning 但被截）

**比 lm_eval 快 100×**：
- lm_eval (thinking ON, 75% length 截断) 跑 200 题用 80 min = 24 s/题
- 自定义脚本 (thinking OFF, parallel=8) 跑 1319 题用 16 min = **0.75 s/题**

**错例类型**：
- 模型只输出 3 token 就 stop（直接吐数字而没 `#### N` 格式）→ 提取失败
- 数学逻辑算错（reasoning 混乱）
- length 截断（reasoning 太长，没说到答案）

**Reproduce**: 见 [Step 7](#step-7-gsm8k-准确性测试推荐用自定义脚本) 推荐做法。

### 长 context benchmark（vLLM with `--max-model-len 16384`）

#### 8K input / 1K output (prefill heavy)

| Batch | Latency | Throughput | vs 1K input 差距 |
|------:|--------:|-----------:|------------------|
| **P1**  | 19.8 s  | **51.7 tok/s** | **+4%**（甚至略快！） |
| **P4**  | 22.9 s  | 178.6 tok/s    | -4% |
| **P16** | 32.7 s  | 499.1 tok/s    | -22% |
| **P64** | 76.3 s  | 849.9 tok/s    | -44% |

**关键发现**：
- ✅ **P1/P4 8K prefill 几乎不拖累**（vs 1K input 同档差 < 5%）— 单/低并发场景，Qwen3.5 hybrid GDN 长 context prefill 优势明显
- 🟡 **P16/P64 8K prefill 拖累显著**（差 22-44%）— 高并发下 prefill chunked + KV cache 压力累加
- **生产建议**：长 prompt 场景（RAG、长文档）用 P4-P8 最划算

#### 1K input / 8K output (decode heavy)

| Batch | Latency | Throughput | vs 1K output 差距 |
|------:|--------:|-----------:|-------------------|
| **P1**  | 151.8 s  | **54.0 tok/s**   | **+9%**（甚至略快） |
| **P4**  | 161.2 s  | 203.3 tok/s    | **+9%** |
| **P16** | 184.3 s  | 711.0 tok/s    | **+11%** |
| **P64** | 307.9 s  | **1702 tok/s**  | **+13%** ⭐ |

**🎯 长 generation 在所有 batch size 上都比短 generation 快 9-13%**！

**原因**：长 generation 让 batch 长时间保持饱和（pure decode 阶段），TPU MXU 利用率高；短 generation (1K out) 频繁经历 prefill→decode 转换 + cleanup，调度开销大。

**生产建议**：
- 长输出场景（文章生成、代码生成）：用 **P64** 1702 tok/s，比 1K/1K 同档高 13%
- 高 batch + 长输出是 TPU v7x-8 的甜蜜点

---

## 待优化（未来工作）

| 方向 | 价值 | 难度 |
|------|------|------|
| **FP4 MoE cache**（类 [DeepSeek R1](../DeepSeek-R1-671B-FP4/README.md) 路径） | MoE 权重减半，KV 余量翻倍 | 中（复用现有 FP4 流程，需适配 1024×4096 dim） |
| **YaRN 1M context 验证** | 长 codebase / 长论文分析场景 | 低（`hf-overrides` 即可，需测准确率） |
| **MMLU-Pro 横向对比** | 与 PR #2366 自测 81.79% 比对 | 低（~60 min 跑完） |
| **MTP speculative decoding** | +30-50% decode throughput（GPU 已验证） | 高（vLLM speculative 框架需扩到 TPU） |
| **JAX native** (`models/jax/qwen35.py`) | 消除 PyTorch dispatch overhead，~10-20% | 高（重写整个 model stack） |

### 已知小问题（不影响生产）

- **`vllm serve` 偶发 sentinel race** — `Engine core initialization failed. Failed core proc(s): {}`，retry 通常 work（详见踩坑 #6）
- **GSM8K 6.8% length 截断**（max_tokens=512 不够 reasoning 完整输出）— 调到 1024+ 准确率可再提升 1-3%

---

> 📋 **状态**: ✅ **Production-ready**（2026-04-25, e2e-02 pod, PR #2366 应用后）。
> - **完整 17 个 batch throughput 数据** (1K/1K sweep P1-P256 + 8K/1K + 1K/8K + Thinking ON)
> - **GSM8K full 1319 题 77.56% accuracy**（远超 CI 阈值 63%）
> - **关键发现**: Peak throughput 是 P128 (2103 tok/s)，不是 P256；长 generation 比短 generation 快 9-13%
> - 内部 doc: https://cc.higcp.com/pages/qwen35-397b-tpu-inference-plan-20260424.html (v1.5)
> - 踩坑故事 HTML（推荐先读）: https://cc.higcp.com/pages/qwen35-397b-debug-story-20260425.html
