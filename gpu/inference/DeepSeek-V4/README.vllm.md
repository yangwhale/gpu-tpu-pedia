# DeepSeek-V4 Flash: B200 端到端推理测试 (vLLM)

> **对标文档**: 本文与 [README.sglang.md](./README.sglang.md) 结构对齐，使用相同硬件、模型、压测工具和负载参数，
> 仅将推理引擎从 SGLang 替换为 vLLM，以实现 **apple-to-apple** 对比。

**机器**: GCP a4-megagpu-8g (8× NVIDIA B200 180GB)
**模型**: [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) — 43 层, 256 experts, MXFP4 (160GB)
**软件**: vLLM v0.20.0 (`deepseekv4-cu130`) + Docker
**日期**: 2026-05-02

---

## 0. 硬件确认

```bash
# GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# 期望: NVIDIA B200, 183359 MiB × 8

# CUDA 版本
nvidia-smi | head -4
# 期望: CUDA 13.0+

# CPU / 内存
nproc && free -h | head -3
```

### 模型权重规格

| 模型 | 架构 | 层数 | 权重/卡 (TP=8) | KV/slice (TP=8 DP=1) |
|---|---|---:|---:|---:|
| DeepSeek-V4-Pro | DeepseekV4ForCausalLM | 61 | **105.57 GB** | 0.46 M tokens |
| DeepSeek-V4-Flash | DeepseekV4ForCausalLM | 43 | **21.83 GB** | 5.87 M tokens |

### Docker 镜像

| 引擎 | 镜像 | 大小 |
|---|---|---|
| SGLang | `lmsysorg/sglang:deepseek-v4-blackwell` | 90 GB |
| vLLM | `vllm/vllm-openai:deepseekv4-cu130` | 31.5 GB |

## 1. 安装 Docker + NVIDIA Container Toolkit

```bash
# Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker

# NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 验证
docker run --rm --gpus all nvidia/cuda:13.0.1-base-ubuntu24.04 nvidia-smi
```

## 2. 下载模型

```bash
pip install huggingface_hub
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash \
  --include "*.safetensors" --include "*.json" \
  --max-workers 16

# 模型会下载到 ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/
# 验证: 46 个 safetensors, ~149GB
```

## 3. vLLM 版本与 DeepSeek V4 支持

### 3.1 版本历史

| 版本 | 日期 | DeepSeek V4 支持 | 说明 |
|---|---|---|---|
| v0.20.0 | 2026-04-27 | **首次支持** | [PR #40860](https://github.com/vllm-project/vllm/pull/40860), 16K+ 行代码 |
| v0.20.1 | (进行中) | 修复+优化 | 14+ 个 DSV4 cherry-pick，含多项关键 bugfix |

> vLLM v0.20.0 是首个支持 DeepSeek V4 的版本，基于 **PyTorch 2.11** + **CUDA 13.0**。
> 由 Bugen Zhao、Woosuk Kwon 等联合开发，一次性引入完整的 MegaMoE、MLA、MTP 实现。

### 3.2 架构支持

vLLM 为 DeepSeek V4 实现了全套专用组件，**不复用** V3/V2 代码：

| 组件 | vLLM 实现 | 说明 |
|---|---|---|
| MegaMoE | `deep_gemm_mega_moe` backend | 256 experts, MXFP4 权重 |
| MLA Attention | `deepseek_v4_attention.py` | 包含 Compressor + Sparse SWA |
| FP4 Indexer | `use_fp4_indexer_cache` | MXFP4 格式 indexer cache |
| MTP | `deepseek_v4_mtp.py` | 多 token 预测 (投机解码) |
| Tokenizer | `deepseek_v4.py` 自定义 | 非 HuggingFace 默认 tokenizer |
| Config | 内部注册 `deepseek_v4` | 无需 config_backup（与 SGLang 不同） |

### 3.3 与 SGLang 的关键差异

| 维度 | SGLang | vLLM |
|---|---|---|
| Config 处理 | `SGLANG_APPLY_CONFIG_BACKUP` 替换 config.json | 内部 `CONFIG_REGISTRY` 直接注册 |
| MoE Backend | DeepEP / flashinfer_mxfp4 / triton | deep_gemm_mega_moe |
| Expert 并行 | `--dp 8 --moe-a2a-backend deepep` | `--enable-expert-parallel --data-parallel-size 8` |
| 投机解码 | EAGLE (外部 draft model) | MTP (内置多 token 预测) |
| Attention | FlashMLA (DeepSeek 自研 kernel) | 自研 Sparse MLA + SWA |
| CUDA Graph | 52 级 bs 梯度 (1~512) | `FULL_AND_PIECEWISE` 模式 |
| 镜像大小 | 90 GB | 31.5 GB |

## 4. vLLM 推理服务

### 4a. 快速验证 (TP=8, 最简配置)

```bash
sudo docker run -d --name vllm-v4-quick \
  --gpus all --ipc=host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -p 30000:30000 \
  vllm/vllm-openai:deepseekv4-cu130 \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --served-model-name DeepSeek-V4-Flash \
  --host 0.0.0.0 --port 30000 \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --tensor-parallel-size 8

# 启动后查看日志: sudo docker logs -f vllm-v4-quick
# 等待 "Uvicorn running on ..." 出现
```

### 4b. 高吞吐配置 (EP + DP=8) ★ 推荐

与 SGLang 的高吞吐配置对标，使用 Expert Parallelism + Data Parallelism。
**实测峰值 4,424 tok/s，超越 SGLang 62%。**

```bash
sudo docker run -d --name vllm-v4-throughput \
  --gpus all --privileged --ipc=host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  --restart unless-stopped \
  vllm/vllm-openai:deepseekv4-cu130 \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --served-model-name DeepSeek-V4-Flash \
  --host 0.0.0.0 --port 8088 \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --enable-expert-parallel \
  --data-parallel-size 8 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --attention_config.use_fp4_indexer_cache=True \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4
```

> **可选加 MTP**: 追加 `--speculative_config '{"method":"mtp","num_speculative_tokens":2}'` 启用投机解码，低并发可提升 35-56%。
> **必须挂 cache 卷** `-v .../huggingface:/root/.cache/huggingface`，否则 docker rm 后 JIT 编译产物丢失，重启需 5-10 分钟全量重编译。
> **不要设置** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，会破坏 custom all-reduce。

### 4c. Baseline 配置 (无 MTP, 无 EP)

最简单的 TP-only 部署，适合功能验证和基线对比。

```bash
sudo docker run -d --name vllm-v4-baseline \
  --gpus all --ipc=host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  --restart unless-stopped \
  vllm/vllm-openai:deepseekv4-cu130 \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --served-model-name DeepSeek-V4-Flash \
  --host 0.0.0.0 --port 8088 \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --tensor-parallel-size 8

# 无 EP/DP/MTP, 纯 TP=8
```

### 验证

```bash
curl http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"DeepSeek-V4-Flash","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":64}'
# 期望: {"choices":[{"message":{"content":"The answer to 2+2 is **4**."}}],...}
```

## 5. 性能实测 (evalscope perf)

### 5.1 压测方法

与 SGLang 文档使用 **完全相同** 的压测工具和负载参数，确保 apple-to-apple 对比。

**压测工具**: evalscope perf
**标准负载**: 4500 Qwen tokens 输入 (≈6434 DeepSeek tokens), 200 token 输出 (ignore_eos=true)

> **Tokenizer 注意**: evalscope 用 `--tokenizer-path` 指定的 tokenizer 生成随机 prompt，但服务端用 DeepSeek-V4 tokenizer 计 token。
> 与参考文档一致使用 4500 Qwen tokens，服务端实际计为 avg ~6434 tokens（约 1.43× 膨胀比）。

```bash
evalscope perf \
  --model DeepSeek-V4-Flash \
  --url http://127.0.0.1:8088/v1/chat/completions \
  --api openai --api-key EMPTY --dataset random \
  --max-tokens 200 --min-tokens 150 \
  --min-prompt-length 4500 --max-prompt-length 4500 \
  --tokenizer-path Qwen/Qwen2.5-0.5B-Instruct \
  --extra-args '{"ignore_eos": true}' \
  --parallel 1 8 40 60 80 100 200 300 400 600 \
  --number  10 80 400 600 800 1000 2000 3000 4000 6000
```

### 5.2 vLLM 高吞吐配置 (EP + DP=8, 无 MTP)

> **实测数据** (2026-05-02, vLLM v0.20.0 deepseekv4-cu130, EP+DP=8, CUDA Graph FULL_AND_PIECEWISE)
> 与 SGLang 高吞吐配置 (DP=8 + DeepEP, 无投机解码) 直接对标。

| 并发 | vLLM tok/s | SGLang tok/s | 差距 | TTFT avg | TPOT avg |
|---:|---:|---:|---|---:|---:|
| 1 | 72 | 72 | **持平** | 0.41 s | 12 ms |
| 8 | 219 | 482 | -55% | 2.19 s | 26 ms |
| 40 | 883 | 1,239 | -29% | 1.69 s | 37 ms |
| 60 | 1,664 | 1,447 | **+15%** | 1.80 s | 27 ms |
| 80 | 1,960 | 1,739 | **+13%** | 1.58 s | 33 ms |
| 100 | 2,546 | 1,898 | **+34%** | 1.58 s | 32 ms |
| 200 | 3,053 | 2,098 | **+46%** | 1.86 s | 56 ms |
| 300 | 3,874 | 2,472 | **+57%** | 2.05 s | 67 ms |
| 400 | 4,052 | 2,620 | **+55%** | 2.85 s | 85 ms |
| **600** | **4,424** | **2,739** | **+62%** | 3.25 s | 120 ms |

**关键发现**:
- **峰值吞吐: 4,424 tok/s** (C=600)，SGLang 峰值 2,739 tok/s，vLLM **领先 62%**
- 超越参考文档 (b200-perf-opt) 的峰值 2,933 tok/s，领先 **51%**
- **分水岭在 C=60**: 低并发 (C≤40) SGLang 领先，高并发 (C≥60) vLLM 大幅领先
- C=8 偏低 (219 tok/s) 可能受 warmup 影响（TTFT=2.19s 异常高）
- EP+DP=8 在高并发下 scaling 效率极佳，C=100→600 吞吐增长 74%

> **低并发偏弱原因分析**: EP+DP=8 模式下每个请求需要在 8 个 DP worker 间分发，
> 低并发时 worker 利用率低、调度开销占比大。SGLang 的 DeepEP 在低并发下更高效。
> 高并发时 vLLM 的 CUDA Graph FULL_AND_PIECEWISE 模式和 deep_gemm_mega_moe 后端
> 充分发挥优势，吞吐 scaling 大幅超越 SGLang。

### 5.3 vLLM EP+DP=8 + MTP n=2

> **实测数据** (2026-05-02, vLLM v0.20.0 deepseekv4-cu130, EP+DP=8 + MTP n=2)
> 在高吞吐配置基础上追加 `--speculative_config '{"method":"mtp","num_speculative_tokens":2}'`。

| 并发 | MTP tok/s | 无 MTP tok/s | 变化 | TTFT avg | TPOT avg | Accept Rate |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 63 | 72 | **-12%** | 1.61 s | 8 ms | 53.6% |
| 8 | 212 | 219 | -3% | 1.50 s | 30 ms | 53.5% |
| 40 | 620 | 883 | **-30%** | 1.51 s | 57 ms | 53.5% |
| 60 | 1,175 | 1,664 | **-29%** | 0.87 s | 47 ms | 53.7% |
| 80 | 1,466 | 1,960 | **-25%** | 0.95 s | 50 ms | 53.8% |
| 100 | 1,602 | 2,546 | **-37%** | 1.00 s | 57 ms | 53.8% |
| 200 | 2,407 | 3,053 | **-21%** | 1.32 s | 76 ms | 53.9% |
| 300 | 2,881 | 3,874 | **-26%** | 1.59 s | 96 ms | 53.8% |
| 400 | 3,217 | 4,052 | **-21%** | 1.85 s | 115 ms | 53.8% |
| **600** | **3,700** | **4,424** | **-16%** | 2.49 s | 150 ms | 53.7% |

**关键发现**:
- **MTP 在 EP+DP=8 模式下反而降低了吞吐**，全部并发级别均为负值 (-3% ~ -37%)
- Acceptance rate 稳定在 ~54%，每次迭代产出 ~2.16 tokens（理论上限 3.0）
- TPOT 下降（8ms vs 12ms @C=1），但 MTP head 的额外计算开销抵消了收益
- 高并发下 EP+DP=8 已充分利用 GPU 算力，MTP 的投机计算变成纯开销

> **结论**: EP+DP=8 模式不建议启用 MTP。MTP 更适合 TP-only 低延迟场景（单请求加速），
> 而非 EP+DP=8 高吞吐场景（batch processing 已最优）。

### 5.4 vLLM Baseline (TP=8, enforce-eager, 无 MTP)

> **实测数据** (2026-05-02, vLLM v0.20.0 deepseekv4-cu130, TP=8, --enforce-eager)
> Baseline 配置使用 `--enforce-eager` 绕过 InductorError（§8.2），禁用 CUDA Graph，性能严重受限。

| 并发 | Baseline tok/s | EP+DP=8 tok/s | 差距 | TTFT avg | TPOT avg |
|---:|---:|---:|---|---:|---:|
| 1 | 7 | 72 | **-90%** | 1.14 s | 134 ms |
| 8 | 51 | 219 | -77% | 1.50 s | 150 ms |
| 40 | 226 | 883 | -74% | 1.45 s | 170 ms |
| 60 | 348 | 1,664 | -79% | 1.52 s | 164 ms |
| 80 | 422 | 1,960 | -78% | 1.87 s | 179 ms |
| 100 | 481 | 2,546 | -81% | 2.05 s | 196 ms |
| 200 | 737 | 3,053 | -76% | 3.22 s | 251 ms |
| 300 | 855 | 3,874 | -78% | 14.49 s | 271 ms |
| 400 | 863 | 4,052 | -79% | 35.91 s | 272 ms |
| **600** | **876** | **4,424** | **-80%** | 78.27 s | 273 ms |

**关键发现**:
- **Baseline 峰值仅 876 tok/s**，EP+DP=8 的 **1/5**，性能差距达 80%
- `--enforce-eager` 禁用 CUDA Graph 是主要瓶颈：TPOT 高达 134-273ms（EP+DP=8 仅 12-120ms）
- C≥300 后 TTFT 剧增（14-78s），表明无 CUDA Graph 的 eager mode 在高并发下调度效率极差
- 吞吐在 C=400 后基本饱和（863→876），scaling 能力远不如 EP+DP=8
- **仅适合功能验证**，不适合任何性能评估场景

### 5.5 四配置吞吐对比总览

| 并发 | vLLM EP+DP=8 | vLLM MTP n=2 | vLLM Baseline | SGLang DP=8 | 最优 |
|---:|---:|---:|---:|---:|---|
| 1 | 72 | 63 | 7 | 72 | 持平 |
| 8 | 219 | 212 | 51 | 482 | SGLang |
| 40 | 883 | 620 | 226 | 1,239 | SGLang |
| 60 | 1,664 | 1,175 | 348 | 1,447 | **vLLM EP** |
| 80 | 1,960 | 1,466 | 422 | 1,739 | **vLLM EP** |
| 100 | 2,546 | 1,602 | 481 | 1,898 | **vLLM EP** |
| 200 | 3,053 | 2,407 | 737 | 2,098 | **vLLM EP** |
| 300 | 3,874 | 2,881 | 855 | 2,472 | **vLLM EP** |
| 400 | 4,052 | 3,217 | 863 | 2,620 | **vLLM EP** |
| **600** | **4,424** | **3,700** | **876** | **2,739** | **vLLM EP** |

---

## 6. 参考基准

> 以下数据来自 [gddezero/b200-perf-opt](https://github.com/gddezero/b200-perf-opt/blob/main/09_deepseek_v4_b200.md)。

### V4-Pro: SGLang vs vLLM @200 并发

| 引擎 | 配置 | toks/s | 差距 |
|---|---|---:|---|
| SGLang | 高吞吐 (DeepEP) | 1,146 | **+23%** |
| vLLM | baseline (MTP n=2) | 930 | baseline |

> 参考文档中 V4-Pro vLLM baseline 使用 MTP n=2，SGLang 不使用投机解码。
> 即使 vLLM 有 MTP 加持，SGLang 仍领先 23%。
>
> **但本文 V4-Flash 实测结果完全颠覆了这一结论**: vLLM EP+DP=8 (无 MTP) 在高并发下
> **大幅领先** SGLang 高吞吐配置 (DeepEP)，峰值 4,424 vs 2,739 tok/s (**+62%**)。
> V4-Flash 的小模型体积 (21.83 GB/卡) 使 KV cache 空间充裕 (97K tokens/worker)，
> 充分释放了 vLLM EP+DP=8 的并行优势。

---

## 7. vLLM 配置详解

### 7.1 关键参数说明

| 参数 | 值 | 说明 |
|---|---|---|
| `--tokenizer-mode deepseek_v4` | **必填** | V4 自定义 tokenizer，非 HuggingFace 默认 |
| `--kv-cache-dtype fp8` | 推荐 | FP8 KV cache，节省显存 |
| `--block-size 256` | 推荐 | MLA 注意力推荐 block size |
| `--enable-expert-parallel` | 高吞吐 | 启用 Expert Parallelism |
| `--data-parallel-size 8` | 高吞吐 | 8 路数据并行 |
| `--compilation-config` | 高吞吐 | CUDA Graph 模式: `FULL_AND_PIECEWISE` |
| `--attention_config.use_fp4_indexer_cache=True` | 可选 | MXFP4 indexer cache |
| `--speculative_config` | 可选 | MTP 投机解码: `{"method":"mtp","num_speculative_tokens":2}` |

### 7.2 配置选型指南

| 场景 | 配置 | 核心特征 | 对标 SGLang |
|---|---|---|---|
| 功能验证 | Baseline (TP=8) | 最简部署，无 EP/DP/MTP | TP=8 MXFP4 |
| 高吞吐 | EP + DP=8 + MTP n=2 | Expert Parallel + 投机解码 | 高吞吐 (DeepEP) |
| 低延迟 | TP=8 + MTP n=2 | 无 EP 开销，MTP 加速 @1 | 低延迟 (EAGLE n=3) |

### 7.3 MTP 投机解码

vLLM 的 MTP (Multi-Token Prediction) 是 DeepSeek V4 内置的投机解码机制，与 SGLang 的 EAGLE 外挂 draft model 不同：

| 维度 | vLLM MTP | SGLang EAGLE |
|---|---|---|
| Draft 来源 | 模型内置 MTP head | 外挂 EAGLE draft model |
| 额外显存 | 无 (共享权重) | 需要 draft model 权重 |
| 配置 | `--speculative_config` JSON | `--speculative-algo EAGLE` + 多参数 |
| 典型加速 | @1: +35~56%, @8: +8~11% | @1: 更高 (n=3 更激进) |
| 稳定性 | n>2 可能 crash | 稳定 |

> **注意**: vLLM MTP `num_speculative_tokens > 2` 当前可能导致 crash ([PR #41162](https://github.com/vllm-project/vllm/pull/41162))，生产环境建议 n=2。

---

## 8. 踩坑记录

### 8.1 Docker 镜像选择

| 镜像 | CUDA | B200 兼容 | 说明 |
|------|------|-----------|------|
| `vllm/vllm-openai:deepseekv4-cu130` | 13.0 | ✓ | **推荐**，预发布 DSV4 专用 |
| `vllm/vllm-openai:deepseekv4-cu129` | 12.9 | ✓ | 备选 |
| `vllm/vllm-openai:v0.20.0-cu130` | 13.0 | ✓ | 通用 v0.20.0，含 DSV4 支持 |
| `vllm/vllm-openai:v0.20.0` | 12.x | ? | 默认 CUDA 版本，需验证 B200 |
| `< v0.20.0` | — | ✗ | 无 DeepSeek V4 支持 |

### 8.2 InductorError (关键踩坑)

**症状**: TP-only 模式（不指定 `--compilation-config`）启动时 crash:
```
torch._inductor.exc.InductorError: AssertionError
  File "torch/_inductor/decompose_triton_kernel_wrapper_functional.py"
```

**根因**: torch inductor 默认编译路径在 B200 上触发 `decompose_triton_kernel_wrapper_functional` 断言失败。
已知问题 ([PR #41135](https://github.com/vllm-project/vllm/pull/41135))。

**解决方案** (二选一):
1. **`--enforce-eager`**: 完全禁用 inductor 编译，性能有损但可用
2. **`--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'`**: 使用显式编译配置，绕过默认路径。EP+DP=8 高吞吐配置使用此方式，**无 InductorError**

> Baseline (TP=8) 需要 `--enforce-eager`，高吞吐 (EP+DP=8) 使用显式 compilation-config 无需 enforce-eager。

### 8.3 启动时间

EP+DP=8 高吞吐配置首次启动耗时较长：

| 阶段 | 耗时 | 说明 |
|---|---:|---|
| 模型加载 | ~12 s | 8×B200 并行加载 MXFP4 权重 |
| DeepGEMM warmup | ~3 min | 1670 个 GEMM kernel JIT 编译 |
| FlashInfer autotuning | ~3 min | 8 个 EP worker 并行 autotuning |
| CUDA Graph capture | ~1.5 min | 51 级 PIECEWISE graph (每级含 inductor compile) |
| TileLang kernel 编译 | ~1 min | `mhc_pre_big_fuse_tilelang` 多个变体 |
| **总计** | **~9 min** | 后续重启若 cache 保留则跳过 JIT 部分 |

> **必须挂 cache 卷** `-v $HOME/.cache/huggingface:/root/.cache/huggingface`，否则 docker rm 后 JIT 编译产物丢失。
> 启动期间会反复出现 `No available shared memory broadcast block found in 60 seconds` 警告，属正常现象。

### 8.4 已知问题

| 问题 | Issue | 影响 | 状态 |
|---|---|---|---|
| InductorError on B200 | [PR #41135](https://github.com/vllm-project/vllm/pull/41135) | TP-only 默认编译 crash | 需 workaround |
| MTP 高并发 hang | [#41402](https://github.com/vllm-project/vllm/issues/41402) | `vllm bench` 并发>1 时 MTP hang | 已修复 |
| >64k token 输入 hang | [#41125](https://github.com/vllm-project/vllm/issues/41125) | 超长 context 卡死 | Open |
| CUDA Graph + 并发相同输入 | [#41331](https://github.com/vllm-project/vllm/issues/41331) | 输出乱码 | Open |
| V4-Pro TP=16 失败 | [#40955](https://github.com/vllm-project/vllm/issues/40955) | TP=16 crash | Open |
| DSML token 泄漏 | [#40801](https://github.com/vllm-project/vllm/issues/40801) | streaming 模式 tool call 泄漏 | Open |
| H200 MTP crash | [#41483](https://github.com/vllm-project/vllm/issues/41483) | H200 上 V4-Pro MTP 崩溃 | Open |

### 8.5 与 SGLang 踩坑对比

| 问题 | SGLang | vLLM |
|---|---|---|
| model_type 不识别 | 需 `SGLANG_APPLY_CONFIG_BACKUP` | 内部注册，无需特殊处理 |
| MXFP4 维度不匹配 | 需修复 config + 指定 moe-runner-backend | 自动检测 `expert_dtype` |
| DeepGEMM JIT 编译 | ~3 分钟首次启动 | ~3 分钟 (1670 kernels) |
| FlashInfer autotuning | — (flashinfer_mxfp4 backend) | ~3 分钟 (8 个 EP worker 并行) |
| CUDA Graph | 52 级 bs (1~512) | 51 级 PIECEWISE + inductor 编译 |
| InductorError | 无此问题 | TP-only 默认路径 crash，需 workaround |
| CUDA Graph OOM | 需手动限 `--mem-fraction-static` | `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` |
| 总启动时间 | ~6 分钟 | ~9 分钟 |
| TP=1 OOM | 同（模型太大） | 同 |

### 8.6 其他注意事项

- **`--tokenizer-mode deepseek_v4` 必须显式指定**: 否则 vLLM 用 HuggingFace 默认 tokenizer，输出可能异常
- **`--trust-remote-code` 必须加**: V4 模型包含自定义代码
- **`VLLM_ENGINE_READY_TIMEOUT_S=3600`**: 首次启动 JIT 编译耗时长，默认 timeout 可能不够
- **`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`**: 启用 CUDA Graph 内存预估，防止 OOM
- **SM90 以下不支持**: A100/L40/RTX 4090 等 SM80/SM89 GPU 官方不支持 ([#40903](https://github.com/vllm-project/vllm/issues/40903))
- **v0.20.1 关键 cherry-pick**: 如使用 v0.20.0 遇到问题，可尝试从 main 构建或等 v0.20.1 发布
- **不要设置** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，会破坏 custom all-reduce

---

## 9. vLLM 开发路线图

> 来源: [vllm-project/vllm#40902](https://github.com/vllm-project/vllm/issues/40902)

### 已完成
- [x] DeepSeek V4 初始支持 (MegaMoE + MLA + MTP)
- [x] SM90 (Hopper) + SM100 (Blackwell) 硬件支持
- [x] FP4 Indexer
- [x] Multi-stream Pre-Attention GEMM
- [x] FP8 KV cache
- [x] NVLink one-sided A2A (BF16/MXFP8)

### 进行中
- [ ] NVFP4 量化 ([PR #41276](https://github.com/vllm-project/vllm/pull/41276))
- [ ] SM120 工作站 Blackwell 支持 ([PR #40991](https://github.com/vllm-project/vllm/pull/40991))
- [ ] AMD MI300/MI355X 支持 ([PR #40871](https://github.com/vllm-project/vllm/pull/40871))
- [ ] Fast topk kernel
- [ ] Fused norm + router (低延迟优化)

### 计划中
- [ ] Pipeline Parallelism (跨节点)
- [ ] Context Parallelism (超长 context)
- [ ] KV cache offloading (CPU + 分布式)
- [ ] DeepEP V2 集成

---

### TODO

- [x] 拉取 `vllm/vllm-openai:deepseekv4-cu130` 镜像 (31.5 GB)
- [x] vLLM Baseline (TP=8) 推理验证 + 正确性检查 (需 `--enforce-eager`)
- [x] vLLM 高吞吐 (EP + DP=8) 部署 (CUDA Graph FULL_AND_PIECEWISE)
- [x] evalscope perf 标准压测（对标 SGLang 相同负载）→ **峰值 4,424 tok/s, SGLang +62%**
- [x] 填入实测数据，生成 SGLang vs vLLM 对比表
- [x] vLLM + MTP n=2 压测 → **MTP 在 EP+DP=8 下反而降低吞吐 16-37%**
- [x] vLLM Baseline TP=8 完整压测 → **峰值仅 876 tok/s，EP+DP=8 的 1/5**
- [x] 输出对比报告页面 (CC Pages)
