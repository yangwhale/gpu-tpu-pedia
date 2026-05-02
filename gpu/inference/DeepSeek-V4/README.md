# DeepSeek-V4 Flash: B200 端到端推理测试

> **参考文档**: 本文是对 [gddezero/b200-perf-opt — 09_deepseek_v4_b200.md](https://github.com/gddezero/b200-perf-opt/blob/main/09_deepseek_v4_b200.md) 的 **独立复测 (reproduce)**。
> 部署步骤、配置参数和测试方法均参考该文档，在 GCP a4-megagpu-8g 实例上从零完成端到端验证。
> 后续计划在此基础上扩展 vLLM 相关测试。

**机器**: GCP a4-megagpu-8g (8× NVIDIA B200 180GB)
**模型**: [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) — 43 层, 256 experts, MXFP4 (160GB)
**软件**: SGLang `deepseek-v4-blackwell` + FlashMLA + Docker
**日期**: 2026-05-01 ~ 2026-05-02

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

> Flash 权重为 Pro 的 1/5 (21.83 / 105.57 = 21%)，但 KV 空间 **~12.8×** (5.87M / 0.46M)，这是 Flash 高吞吐反超的根本原因。

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

## 3. FlashMLA Kernel 测试

FlashMLA 是 DeepSeek 的 MLA (Multi-head Latent Attention) 优化 kernel，B200 上支持 sparse/dense prefill 和 decoding。

```bash
# 在 Docker 内构建测试（需要 CUDA 12.9+）
docker run --rm -it --gpus all \
  nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04 bash

# 容器内：
apt-get update && apt-get install -y git python3-pip python3-venv
python3 -m venv /opt/venv && source /opt/venv/bin/activate
pip install torch

# CUDA 13.0 CCCL 头文件修复（必须）
ln -sf /usr/local/cuda/targets/x86_64-linux/include/cccl/cuda /usr/local/cuda/include/cuda

git clone https://github.com/deepseek-ai/FlashMLA.git /opt/flash-mla
cd /opt/flash-mla
git submodule update --init --recursive
pip install "setuptools<82"
pip install --no-build-isolation -v .

# 跑 benchmark（必须 cd 出源码目录）
cd /tmp
python -c "import flash_mla; print('OK')"
python /opt/flash-mla/tests/test_flash_mla_sparse_decoding.py
python /opt/flash-mla/tests/test_fmha_sm100.py
python /opt/flash-mla/tests/test_flash_mla_sparse_prefill.py
```

**B200 实测性能** (2026-05-01, CUDA 13.0, Driver 580.126.09):

### Test 1: Dense MLA Decoding
- **跳过**: 仅支持 SM90 (Hopper)，B200 为 SM100 (Blackwell)

### Test 2: Sparse MLA Decoding (FP8 KV)
- 4748/4748 正确性测试全部通过
- TFlops 几何平均值: **311.6**
- 峰值性能 (topk=16384, B=148, h=128, d_qk=512): **842 TFlops, 1277 GB/s**
- 典型性能 (topk=2048, B=74, h=128, d_qk=576): **553 TFlops, 1466 GB/s**
- 小 batch (B=2): 受 kernel launch 开销限制，~52 TFlops

### Test 3: MHA Prefill (SM100/B200, forward + backward)
- Forward 峰值 (GQA h=32/h_k=4, d=192, sq=8K): **1418 TFLOP/s**
- Forward MHA (h=128, d=192, sq=4K, non-causal): **1356 TFLOP/s**
- Forward MHA (h=128, d=128, sq=4K, non-causal): **1206 TFLOP/s**
- Backward MHA (h=128, d=128, sq=8K): **921-936 TFLOP/s**

### Test 4: Sparse MLA Prefill
- 617/617 正确性测试全部通过
- 峰值 (sq=4K, topk=2048, h=128, d_qk=576, sk=8K): **1379.5 TFlops, 6.38 TB/s**
- 长 context (sq=4K, topk=2048, sk=131K): **1100.9 TFlops, 5.09 TB/s**
- 小 head (h=64, topk=512): **725 TFlops, 7.08 TB/s**

## 4. SGLang 推理服务

### 关键修复：MXFP4 专家权重支持

DeepSeek V4 Flash 的 expert 权重是 **MXFP4** 格式（int8 打包，每字节 2 个 FP4 值）。
SGLang `deepseek-v4-blackwell` 镜像的 `config_backup_small.json` 缺少 `expert_dtype: fp4` 字段，
导致默认走 triton fused_moe 路径时出现 hidden size 2x 不匹配。

**修复方法**: 添加 `expert_dtype` 到 backup config + 指定 `--moe-runner-backend flashinfer_mxfp4`。

```bash
# Step 1: 修复 config backup（加入 expert_dtype）
docker create --name tmp lmsysorg/sglang:deepseek-v4-blackwell
docker cp tmp:/workspace/sglang/python/sglang/srt/configs/config_backup_small.json /tmp/config_backup_small.json
docker rm tmp

python3 -c "
import json
with open('/tmp/config_backup_small.json') as f: c = json.load(f)
c['expert_dtype'] = 'fp4'
with open('/tmp/config_backup_small_fixed.json', 'w') as f: json.dump(c, f, indent=2)
"

# Step 2: 启动服务
docker run -d --name sglang-v4 \
  --gpus all --shm-size 64g \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v /tmp/config_backup_small_fixed.json:/workspace/sglang/python/sglang/srt/configs/config_backup_small.json:ro \
  -p 30000:30000 \
  lmsysorg/sglang:deepseek-v4-blackwell \
  python3 -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V4-Flash \
    --tp 8 \
    --trust-remote-code \
    --moe-runner-backend flashinfer_mxfp4 \
    --max-running-requests 12 \
    --host 0.0.0.0 --port 30000

# 启动耗时约 6 分钟（含 DeepGEMM JIT 编译 + CUDA Graph 捕获）
# 查看日志: docker logs -f sglang-v4
```

### 验证

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":64}'
# 期望: {"choices":[{"message":{"content":"The answer to 2+2 is **4**."}}],...}
```

## 5. 性能实测 (evalscope perf)

### 5.1 高吞吐配置 (TP=8 DP=8 + DeepEP) ★

**配置**: 8× B200, TP=8 DP=8, DeepEP all-to-all, CUDA Graph (bs≤64), max_running=1024

```bash
sudo docker run -d \
  --name sglang-v4-ht \
  --gpus all --ipc=host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /path/to/config_backup_small_fixed.json:/workspace/sglang/python/sglang/srt/configs/config_backup_small.json:ro \
  -e SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024 \
  -e SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
  -e NCCL_IB_DISABLE=1 \
  --restart unless-stopped \
  lmsysorg/sglang:deepseek-v4-blackwell \
  python3 -m sglang.launch_server \
    --trust-remote-code \
    --model deepseek-ai/DeepSeek-V4-Flash \
    --served-model-name DeepSeek-V4-Flash \
    --tp 8 --dp 8 \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --deepep-config '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}' \
    --max-running-requests 1024 \
    --cuda-graph-max-bs 64 \
    --mem-fraction-static 0.82 \
    --enable-metrics \
    --host 0.0.0.0 --port 30000
```

> **注意**: `--moe-runner-backend flashinfer_mxfp4` 与 `--moe-a2a-backend deepep` **不兼容**（`DeepEPLLDispatchOutput` 无 `topk_output` 属性），高吞吐/平衡配置不要加此参数。

**压测工具**: evalscope perf (与参考文档相同)
**标准负载**: 4500 token 输入, 150-200 token 输出 (ignore_eos=true)

```bash
evalscope perf \
  --model DeepSeek-V4-Flash \
  --url http://127.0.0.1:30000/v1/chat/completions \
  --api openai --api-key EMPTY --dataset random \
  --max-tokens 200 --min-tokens 150 \
  --min-prompt-length 4500 --max-prompt-length 4500 \
  --tokenizer-path Qwen/Qwen2.5-0.5B-Instruct \
  --extra-args '{"ignore_eos": true}' \
  --parallel 1 2 4 8 20 40 60 80 100 200 300 400 600 \
  --number  10 20 40 80 200 400 600 800 1000 2000 3000 4000 6000
```

| 并发 | 请求数 | toks/s | Avg Lat (s) | P99 Lat (s) | TTFT avg (s) | TTFT P99 (s) | TPOT avg (ms) | TPOT P99 (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 71 | 2.81 | 3.67 | 0.66 | 1.51 | 11 | 11 |
| 2 | 20 | 133 | 3.01 | 3.83 | 0.80 | 1.65 | 11 | 12 |
| 4 | 40 | 224 | 3.58 | 6.99 | 1.12 | 4.72 | 12 | 27 |
| 8 | 80 | 410 | 3.90 | 8.11 | 1.53 | 5.77 | 12 | 13 |
| 20 | 200 | 835 | 4.76 | 6.10 | 1.07 | 2.17 | 19 | 25 |
| 40 | 400 | 1,422 | 5.61 | 7.69 | 1.75 | 2.93 | 19 | 32 |
| 60 | 600 | 1,577 | 7.57 | 11.80 | 1.79 | 4.76 | 29 | 48 |
| 80 | 800 | 1,765 | 9.02 | 13.72 | 2.57 | 5.57 | 32 | 56 |
| 100 | 1000 | 1,762 | 11.30 | 16.95 | 2.53 | 7.17 | 44 | 72 |
| 200 | 2000 | 2,308 | 17.19 | 27.41 | 5.72 | 13.32 | 58 | 105 |
| **300** | **3000** | **2,608** | 22.94 | 27.07 | 8.96 | 18.31 | 70 | 112 |
| 400 | 4000 | 2,603 | 30.54 | 37.70 | 12.24 | 25.62 | 92 | 151 |
| 600 | 6000 | 1,857 | 64.49 | 89.97 | 6.95 | 35.94 | 289 | 411 |

**关键结果**:
- **峰值吞吐: 2,608 tok/s** (C=300)，参考文档同配置: **2,933 tok/s** (C=600)
- @1 吞吐: 71 tok/s，参考: 76 tok/s — 基本一致
- C=300-400 区间吞吐平台，C=600 明显过载 (TPOT 从 92ms 飙到 289ms)
- 100% 成功率，全部 13 个并发梯度零失败

> **与参考差距分析**: 峰值低 11% (2,608 vs 2,933)，峰值并发更低 (300 vs 600)。
> 可能原因：evalscope tokenizer 采样的实际输入 token 数 ~6400 (高于目标 4500)，增加了 prefill 负载。

---

## 6. 参考基准：b200-perf-opt 性能数据

> 以下数据来自 [gddezero/b200-perf-opt](https://github.com/gddezero/b200-perf-opt/blob/main/09_deepseek_v4_b200.md) (2026-04-24~27)，使用 evalscope perf 压测，标准负载 4500 token 输入 + 150-200 token 输出。

### 6.1 总览

| 模型 | 引擎 | 配置 | 编号 | 峰值并发 | 峰值 toks/s | @1 toks/s | @1 TTFT |
|---|---|---|:---:|:---:|---:|---:|---:|
| V4-Flash | SGLang | **高吞吐** (无spec, max_req=1024) | #208 | 600 | **2,933** | 76 | 0.50 s |
| V4-Flash | SGLang | 平衡 (EAGLE n=1, max_req=512) | #207 | 400 | 2,393 | 96 | 0.55 s |
| V4-Pro | SGLang | 高吞吐 (无spec, max_req=256) | #201 | 200 | 1,146 | 47 | 0.78 s |
| V4-Pro | vLLM | baseline (MTP n=2) | #202 | 200 | 930 | 66 | 0.49 s |
| V4-Flash | SGLang | 低延迟 (TP=8, MXFP4, EAGLE n=3) | #206 | 180 | 829 | 160 | 0.30 s |
| V4-Pro | SGLang | 平衡 (EAGLE n=1, max_req=128) | #200 | 100 | 782 | 60 | 0.85 s |
| V4-Pro | SGLang | 低延迟 (TP=8, MXFP4, EAGLE n=3) | #199 | 20 | 312 | 110 | 0.49 s |

### 6.2 V4-Flash SGLang 高吞吐 (#208) 详细

| 并发 | toks/s | TTFT avg | TPOT avg |
|---:|---:|---:|---:|
| 1 | 76 | 0.50 s | 8 ms |
| 40 | 1,105 | 0.80 s | 26 ms |
| 100 | 1,754 | 1.47 s | 39 ms |
| 200 | 2,236 | 2.59 s | 60 ms |
| 400 | 2,703 | 5.18 s | 96 ms |
| **600** | **2,933** | 9.17 s | 131 ms |

### 6.3 V4-Pro SGLang vs vLLM @200 并发

| 引擎 | 配置 | toks/s | 差距 |
|---|---|---:|---|
| SGLang | 高吞吐 (DeepEP) | 1,146 | **+23%** |
| vLLM | baseline (MTP n=2) | 930 | baseline |

---

## 7. SGLang 三套生产配置

> 来自 [b200-perf-opt](https://github.com/gddezero/b200-perf-opt)，经过充分 warmup + 高并发压测验证。

### 通用前置

```bash
sudo docker pull lmsysorg/sglang:deepseek-v4-blackwell
sudo mkdir -p /lssd/cache && sudo chmod -R 777 /lssd/cache
sudo docker stop deepseek-v4-pro deepseek-v4-flash 2>/dev/null
sudo docker rm   deepseek-v4-pro deepseek-v4-flash 2>/dev/null
```

### 配置一：低延迟 (TP=8 + MXFP4 + EAGLE n=3)

适用：单/小并发优先 (≤40 for Pro, ≤180 for Flash)

```bash
sudo docker run -d \
  --name deepseek-v4-pro \
  --gpus all --ipc=host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  -v /lssd/models:/lssd/models:ro \
  -v /lssd/cache:/root/.cache \
  -e SGLANG_ENABLE_SPEC_V2=1 \
  -e SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
  -e NCCL_IB_DISABLE=1 \
  --restart unless-stopped \
  lmsysorg/sglang:deepseek-v4-blackwell \
  python3 -m sglang.launch_server \
    --trust-remote-code \
    --model-path /lssd/models/DeepSeek-V4-Pro \
    --served-model-name DeepSeek-V4-Pro \
    --tp 8 \
    --moe-runner-backend flashinfer_mxfp4 \
    --speculative-algo EAGLE \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --chunked-prefill-size 4096 \
    --disable-flashinfer-autotune \
    --mem-fraction-static 0.82 \
    --enable-metrics \
    --tool-call-parser deepseekv4 --reasoning-parser deepseek-v4 \
    --host 0.0.0.0 --port 8088
```

> `--mem-fraction-static 0.82` 必须显式设，否则 EAGLE n=3 cudagraph capture OOM。
> Flash 同配置：模型名换成 `DeepSeek-V4-Flash`，其余完全一致。

### 配置二：平衡 (TP=8 DP=8 + DeepEP + EAGLE n=1)

适用：中等并发 + 低延迟 (80 ≤ p ≤ 400)，生产首选。

```bash
sudo docker run -d \
  --name deepseek-v4-pro \
  --gpus all --ipc=host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  -v /lssd/models:/lssd/models:ro \
  -v /lssd/cache:/root/.cache \
  -e SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 \
  -e SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
  -e NCCL_IB_DISABLE=1 \
  --restart unless-stopped \
  lmsysorg/sglang:deepseek-v4-blackwell \
  python3 -m sglang.launch_server \
    --trust-remote-code \
    --model-path /lssd/models/DeepSeek-V4-Pro \
    --served-model-name DeepSeek-V4-Pro \
    --tp 8 --dp 8 \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --deepep-config '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}' \
    --speculative-algo EAGLE \
    --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2 \
    --max-running-requests 128 \
    --cuda-graph-max-bs 64 \
    --mem-fraction-static 0.82 \
    --enable-metrics \
    --enable-metrics-for-all-schedulers \
    --tool-call-parser deepseekv4 --reasoning-parser deepseek-v4 \
    --host 0.0.0.0 --port 8088
```

> **公式约束**: V4-Pro `128 × 2 = 256 ≤ 256` ✓
> **Flash 同配置**: `--max-running-requests 512` + `DISPATCH=1024`

### 配置三：高吞吐 (TP=8 DP=8 + DeepEP + 无 spec)

适用：高并发、批吞吐优先 (≥200)，单机最高吞吐配方。

```bash
sudo docker run -d \
  --name deepseek-v4-pro \
  --gpus all --ipc=host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  -v /lssd/models:/lssd/models:ro \
  -v /lssd/cache:/root/.cache \
  -e SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 \
  -e SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
  -e NCCL_IB_DISABLE=1 \
  --restart unless-stopped \
  lmsysorg/sglang:deepseek-v4-blackwell \
  python3 -m sglang.launch_server \
    --trust-remote-code \
    --model-path /lssd/models/DeepSeek-V4-Pro \
    --served-model-name DeepSeek-V4-Pro \
    --tp 8 --dp 8 \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --deepep-config '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}' \
    --max-running-requests 256 \
    --cuda-graph-max-bs 64 \
    --mem-fraction-static 0.82 \
    --enable-metrics \
    --enable-metrics-for-all-schedulers \
    --tool-call-parser deepseekv4 --reasoning-parser deepseek-v4 \
    --host 0.0.0.0 --port 8088
```

> **公式约束**: V4-Pro `256 × 1 = 256 ≤ 256` ✓
> **Flash 同配置**: `--max-running-requests 1024` + `DISPATCH=1024`

### 配置选型指南

| 场景 | 推荐配置 | 核心特征 | 适用并发 |
|---|---|---|---|
| 实时对话 | 低延迟 | MXFP4 + EAGLE n=3, @1 TTFT 最低 | ≤20 (Pro) / ≤180 (Flash) |
| 生产服务 | 平衡 | DeepEP + EAGLE n=1, 兼顾延迟和吞吐 | 80-400 |
| 批量推理 | 高吞吐 | DeepEP + 无spec, 最大化吞吐 | 200-1000 |

---

## 8. vLLM Baseline 部署

```bash
sudo docker run -d \
  --name deepseek-v4-pro \
  --gpus all --privileged --ipc=host \
  --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  -v /lssd/models:/lssd/models:ro \
  -v /lssd/cache:/root/.cache \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  --restart unless-stopped \
  vllm/vllm-openai:deepseekv4-cu130 \
  /lssd/models/DeepSeek-V4-Pro \
  --served-model-name DeepSeek-V4-Pro \
  --host 0.0.0.0 --port 8088 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --enable-expert-parallel \
  --data-parallel-size 8 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --attention_config.use_fp4_indexer_cache=True \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --speculative_config '{"method":"mtp","num_speculative_tokens":2}'
```

> **必须挂 cache 卷** `-v /lssd/cache:/root/.cache`，否则 docker rm 后所有 JIT 编译产物丢失，重启需 5-10 分钟全量重编译。

---

## 9. DeepEP Dispatch Token 约束

```
max_running_requests × draft_tokens ≤ SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK
```

| 场景 | max_req | draft | DISPATCH | 验证 |
|---|---:|---:|---:|---|
| V4-Pro 平衡 | 128 | 2 | 256 | 128×2=256 ≤ 256 ✓ |
| V4-Pro 高吞吐 | 256 | 1 | 256 | 256×1=256 ≤ 256 ✓ |
| V4-Flash 平衡 | 512 | 2 | 1024 | 512×2=1024 ≤ 1024 ✓ |
| V4-Flash 高吞吐 | 1024 | 1 | 1024 | 1024×1=1024 ≤ 1024 ✓ |

Flash 的 DISPATCH 可以设为 Pro 的 4× (256→1024)，因为 KV 空间 12.8× 大 → max_req 上限 4×。

---

## 10. 踩坑记录

### 10.1 FlashMLA 构建 (CUDA 13.0)

1. **CCCL 头文件路径变更**: CUDA 13.0 把 `cuda/std/` 移到了 `cccl/cuda/std/`
   - 修复: `ln -sf /usr/local/cuda/targets/x86_64-linux/include/cccl/cuda /usr/local/cuda/include/cuda`
2. **torch 依赖**: 必须用 `pip install --no-build-isolation -v .` (setup.py 依赖 torch)
3. **setuptools 版本**: 需要 `setuptools<82` (torch 2.11 兼容性)
4. **导入路径冲突**: 安装后在源码目录内 import 会找到本地 `flash_mla/`，需 `cd /tmp` 再 import
5. **Dense MLA Decoding**: 仅支持 SM90 (Hopper), B200 (SM100) 跳过

### 10.2 SGLang Docker 镜像选择

| 镜像 | 架构 | B200 兼容 | 说明 |
|------|------|-----------|------|
| `lmsysorg/sglang:latest` | 通用 | ✗ | `deepseek_v4` model_type 不被 transformers 识别 |
| `deepseek-v4-blackwell` | SM100 x86_64 | ✓* | 需修复 config backup + 指定 MXFP4 backend |
| `deepseek-v4-b300` | SM100 x86_64 | ✓* | 同上 |
| `deepseek-v4-hopper` | SM90 | ✗ | `sgl_kernel` 编译目标 SM90 |
| `deepseek-v4-grace-blackwell` | ARM | ✗ | `no matching manifest for linux/amd64` |
| `dev-cu13` | 通用 CUDA 13 | ✗ | 同 `latest` |

### 10.3 MXFP4 Expert 权重维度不匹配 (核心 Bug)

**症状**: `AssertionError: Hidden size mismatch` in `fused_moe.py:341`

**根因**: DeepSeek V4 Flash 的 expert 权重存储为 MXFP4 (int8 打包, 2 个 FP4/byte):
- 磁盘上 `w1.weight: [2048, 2048] int8` → 实际逻辑 `[2048, 4096]`
- `hidden_states.shape[1]=4096` vs `w1.shape[2]=2048` → 2x 不匹配

`config_backup_small.json` 缺少 `expert_dtype: fp4` 字段, 导致默认的 triton MoE runner
不知道需要 MXFP4 解包, 直接用 packed 维度做 shape check。

**修复**: 添加 `expert_dtype: fp4` + `--moe-runner-backend flashinfer_mxfp4`

### 10.4 其他注意事项

- **TP=1 不可行**: BF16 权重 173GB > 单卡 180GB 可用（KV cache 无空间）
- **Config Backup**: SGLang 用 `SGLANG_APPLY_CONFIG_BACKUP` 替换 model config, 将 `deepseek_v4` 映射为内部注册的 `deepseek_ref` 类型
- **DeepGEMM JIT**: 首次启动需编译 32768 个 GEMM kernel，约 40 秒/组，共 ~3 分钟
- **高并发 OOM (TP-only)**: B200 attention 的 einsum 在 TP-only 高并发下 CUBLAS 资源耗尽，限制 max_running_requests ≤ 12。DP=8+DeepEP 配置可支持 max_running_requests=1024 (128/DP)
- **DeepEP + flashinfer_mxfp4 不兼容**: `--moe-runner-backend flashinfer_mxfp4` 与 `--moe-a2a-backend deepep` 同时使用会报 `AttributeError: 'DeepEPLLDispatchOutput' object has no attribute 'topk_output'`。高吞吐/平衡配置不需要显式指定 moe-runner-backend
- **Warmup 注意**: @1 (并发=1) 指标必须充分 warmup (n≥100)，否则 CUDA graph/inductor 编译时间污染 TTFT

---

### TODO

- [x] Docker 安装 + NVIDIA Container Toolkit (Docker 29.4.2, NCTK 1.19.0)
- [x] FlashMLA kernel benchmark on B200 (4748+617 cases passed, peak 1418 TFLOP/s)
- [x] SGLang TP=8 MXFP4 推理 (修复 config backup + flashinfer_mxfp4 backend)
- [x] evalscope perf 标准压测 (峰值 **2,608 tok/s** @C=300, 参考 2,933 @C=600, 差 11%)
- [ ] 排查吞吐差距原因 (tokenizer 采样 ~6400 tok vs 目标 4500)
- [ ] 尝试 FP8 量化版 `sgl-project/DeepSeek-V4-Flash-FP8` 对比性能
- [ ] EAGLE 投机解码低延迟模式实测
- [ ] vLLM baseline 对比测试
