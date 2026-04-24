# Kimi K2.6 (1T-A32B) FP8/INT4 Inference on TPU v7x

> 🌐 **Languages** | **语言**: **中文** · [English (TBD)](README.en.md)

> 端到端指南：在 TPU v7x 上运行 Kimi K2.6（1T 总参 / 32B 激活 / native INT4 routed experts）推理。
> 架构与 DeepSeek V3 高度同源，最大化复用 DeepSeek R1 / GLM-5.1 的代码基础。
>
> **代码仓库**: https://github.com/yangwhale/tpu-inference (branch: `feature/kimi-k26-inference`)
>
> **模型**: [moonshotai/Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6)（64 safetensors, ~595 GB INT4）

## 🎯 关键性能（待实测）

| 操作点 | 状态 |
|--------|------|
| 1 层 cold start | ⏳ 待测 |
| 4 层 cold start | ⏳ 待测 |
| 全模型（v7x-8 单机）| ⏳ 待测（HBM 紧，可能需要 v7x-16） |
| 1K/1K throughput | ⏳ 待测 |

> **vs Kimi 官方**: Moonshot 官方部署用 vLLM/SGLang on H800/H100，未公开 TPU 部署
> **质量**: SWE-Bench Verified 80.2% (开源 SOTA), AIME 2026 96.4%, GPQA-Diamond 90.5%, BrowseComp 83.2%

---

## ⚠️ 三个必设环境变量

> **每次启动 vLLM 前必须设置（与 DeepSeek R1 / GLM-5.1 同模式）。**

| 环境变量 | 值 | 漏设后果 |
|---------|-----|---------|
| `NEW_MODEL_DESIGN=1` | **必须设** | MLA 模型强制要求，不设直接报错退出 |
| `MODEL_IMPL_TYPE=vllm` | 推荐 | 走 vLLM PyTorch + TorchAX path（K2.6 直接复用 DeepSeek modeling） |
| `MOE_WEIGHT_CACHE_DIR=/dev/shm` | 可选 | 仅 JAX native path 需要（K2.6 默认走 vLLM path） |

---

## 硬件与模型概览

| 项目 | 要求 |
|------|------|
| TPU | **v7x-8（最低）/ v7x-16（推荐）** |
| HBM | 94.75 GB/device，v7x-8 共 758 GB / v7x-16 共 1,516 GB |
| 主机内存 | ≥800 GB（含 /dev/shm 部分缓存） |
| 存储 | ≥1 TB（模型 595 GB + 工作空间） |

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

> 量化路径不同（FP4 vs INT4）— K2.6 走 `compressed-tensors` 标准格式，需要 INT4 W4A16 量化方法（PR #2306 进行中）。

---

## Step 1: 创建 GKE TPU Pod

复用现有 e2e pod（`tpu-v7x-lite-podslice`, 2x2x1, Lustre PVC）。如果新建：

```bash
kubectl apply -f - <<'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: vllm-kimi-k26
spec:
  containers:
  - name: main
    image: <YOUR_DOCKER_REGISTRY>/vllm-tpu:latest
    resources:
      limits:
        google.com/tpu: 8
    volumeMounts:
    - name: data
      mountPath: /data
    - name: lustre
      mountPath: /lustre
    - name: dshm
      mountPath: /dev/shm
  volumes:
  - name: data
    persistentVolumeClaim:
      claimName: data-pvc
  - name: lustre
    persistentVolumeClaim:
      claimName: lustre-pvc
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: 800Gi
  restartPolicy: Never
  nodeSelector:
    cloud.google.com/gke-tpu-topology: 2x2x1
    cloud.google.com/gke-tpu-accelerator: tpu-v7x-lite-podslice
EOF
```

进入 Pod：

```bash
kubectl exec -it vllm-kimi-k26 -- bash
```

---

## Step 2: 准备代码

切到 Kimi K2.6 分支：

```bash
cd /workspace/tpu_inference
git fetch origin
git checkout feature/kimi-k26-inference

# 验证
python3 -c "import tpu_inference; print('OK')"
```

---

## Step 3: 下载模型权重

```bash
# 安装下载工具（如果未装）
pip install huggingface_hub

# 用 token 加速下载（HuggingFace 登录用户带宽更高）
export HF_TOKEN='<your-hf-token>'

# 下载到 Lustre（595 GB ~ 25-40 min）
hf download moonshotai/Kimi-K2.6 \
  --local-dir /lustre/models/Kimi-K2.6 \
  --max-workers 16

# 验证
ls /lustre/models/Kimi-K2.6/*.safetensors | wc -l
# 预期：64

du -sh /lustre/models/Kimi-K2.6/
# 预期：~595 GB
```

设置模型路径：

```bash
export MODEL=/lustre/models/Kimi-K2.6
```

---

## Step 4: 启动 vLLM 推理服务

> **K2.6 走 vLLM PyTorch + TorchAX path**（与 Qwen3.5 同模式），不需要预生成 FP4 cache（INT4 weights 直接加载）。

```bash
export NEW_MODEL_DESIGN=1
export MODEL_IMPL_TYPE=vllm

# 必须 cd 到非 tpu-inference 目录避免 namespace 冲突
cd /tmp

python3 -m vllm.entrypoints.openai.api_server \
  --model $MODEL \
  --tensor-parallel-size 8 \
  --quantization compressed-tensors \
  --enforce-eager \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-model-len 4096 \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --reasoning-parser deepseek_v3 \
  --additional-config '{"sharding":{"sharding_strategy":{"enable_dp_attention":true,"expert_parallelism":8,"tensor_parallelism":1}}}'
```

**关键参数说明：**

| 参数 | 取值 | 原因 |
|------|------|------|
| `--quantization compressed-tensors` | INT4 W4A16 | K2.6 native quantization format |
| `--limit-mm-per-prompt` | image=0, video=0 | 跳过 MoonViT vision encoder |
| `--enforce-eager` | — | 避免 cudagraph 在 TPU 上的兼容问题 |
| `--reasoning-parser` | deepseek_v3 | K2.6 用 DeepSeek 风格的 `<think>` token |
| `--max-model-len` | 4096（初测）/ 256K（生产） | 测试用 4K，生产可拉到 256K |

---

## Step 5: 验证推理

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/lustre/models/Kimi-K2.6",
    "messages": [{"role": "user", "content": "What is 2+3? Show your reasoning step by step."}],
    "max_tokens": 256,
    "temperature": 0.6,
    "extra_body": {"chat_template_kwargs": {"thinking": true}}
  }'
```

预期：返回 `<think>...</think>` 段 + 最终答案 5。

---

## 渐进测试策略（开发期）

### 1 层测试（最快验证）

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model $MODEL \
  --tensor-parallel-size 8 \
  --quantization compressed-tensors \
  --enforce-eager \
  --max-model-len 2048 \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' \
  --hf-overrides '{"text_config": {"num_hidden_layers": 1}}'
```

### 4 层测试（验证 routed experts dispatch）

```bash
# 同上但 num_hidden_layers: 4
--hf-overrides '{"text_config": {"num_hidden_layers": 4}}'
```

### 全模型（生产配置）

去掉 `--hf-overrides`，按 Step 4 完整启动。

---

## 已知问题与待优化

| 问题 | 状态 | 备注 |
|------|------|------|
| INT4 W4A16 量化 path | 🟡 PR #2306 进行中 | tpu-inference 上 INT4 实现仍在 review |
| Multi-host (1T 模型) | 🟡 部分支持 | PR #2324 提供 streaming loader |
| MTP speculative decoding | ⚪ K2.6 无 MTP | 不需要 |
| MoonViT vision | 🟡 跳过（text-only） | Phase 3 可选 |
| INT8 matmul 加速 | ⚪ 未实现 | 当前走 BF16 matmul（同 DeepSeek FP4 路径）|

---

## 参考资料

| 资源 | 链接 |
|------|------|
| Kimi K2.6 HuggingFace | [moonshotai/Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6) |
| Kimi 官方 deploy guide | [deploy_guidance.md](https://huggingface.co/moonshotai/Kimi-K2.6/blob/main/docs/deploy_guidance.md) |
| Kimi K2.6 Tech Blog | [kimi.com/blog/kimi-k2-6](https://www.kimi.com/blog/kimi-k2-6.html) |
| DeepSeek R1 FP4 推理指南 | [../DeepSeek-R1-671B-FP4/README.md](../DeepSeek-R1-671B-FP4/README.md) |
| GLM-5.1 FP4 推理指南 | [../GLM-5.1-754B-FP4/README.md](../GLM-5.1-754B-FP4/README.md) |
| tpu-inference 代码 | [github.com/yangwhale/tpu-inference](https://github.com/yangwhale/tpu-inference) `feature/kimi-k26-inference` |
| upstream Kimi K2.5 进度 | [PR #2306 (INT4 MoE)](https://github.com/vllm-project/tpu-inference/pull/2306), [jacobplatin/kimi-k2.5-support](https://github.com/vllm-project/tpu-inference/tree/jacobplatin/kimi-k2.5-support) |

---

> 📋 **状态**: 开发中 (2026-04-24) — 模型权重下载完毕后进入代码适配 + 测试阶段。
> 完成后会更新性能数据 + 测试结果到本 README。
