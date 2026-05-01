# DeepSeek-V4 Flash: B200 端到端推理测试

**机器**: GCP a4-megagpu-8g (8× NVIDIA B200 180GB)
**模型**: [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) — 43 层, 256 experts, MXFP4 (160GB)
**软件**: SGLang `deepseek-v4-blackwell` + FlashMLA + Docker
**日期**: 2026-05-01

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

## 5. 性能实测

**配置**: 8× B200, TP=8, MXFP4 Experts, CUDA Graph ON, max_running=12

每卡 VRAM 占用: 21.83 GB 权重 + KV cache，~154 GB 可用

| 测试 | 成功率 | 吞吐 (tok/s) | 单请求 TPS | P50 延迟 (s) | P99 延迟 (s) |
|------|--------|-------------|-----------|-------------|-------------|
| 短文本 (128 tok), C=1 | 10/10 | 115.8 | 109.0 | 1.03 | 1.04 |
| 短文本 (128 tok), C=2 | 20/20 | 187.1 | 90.2 | 1.19 | 2.00 |
| 短文本 (128 tok), C=4 | 20/20 | 265.4 | 70.9 | 1.51 | 2.40 |
| 短文本 (128 tok), C=8 | 40/40 | 426.5 | 57.5 | 1.90 | 3.34 |
| 长文本 (512 tok), C=1 | 3/3 | 137.5 | 137.5 | 3.72 | 3.73 |
| 长文本 (512 tok), C=2 | 6/6 | 254.4 | 127.2 | 3.98 | 4.12 |
| 长文本 (512 tok), C=4 | 6/6 | 371.0 | 122.4 | 4.29 | 4.29 |

**关键指标**:
- 单请求输出速度: **109-137 tok/s**
- 最大稳定吞吐: **~430 tok/s** (C=8, 短文本)
- TTFT (首 token 延迟): ~1s
- 高并发 (C>12) 会触发 CUBLAS OOM，建议 max_running_requests ≤ 12

## 6. 踩坑记录

### 6.1 FlashMLA 构建 (CUDA 13.0)

1. **CCCL 头文件路径变更**: CUDA 13.0 把 `cuda/std/` 移到了 `cccl/cuda/std/`
   - 修复: `ln -sf /usr/local/cuda/targets/x86_64-linux/include/cccl/cuda /usr/local/cuda/include/cuda`
2. **torch 依赖**: 必须用 `pip install --no-build-isolation -v .` (setup.py 依赖 torch)
3. **setuptools 版本**: 需要 `setuptools<82` (torch 2.11 兼容性)
4. **导入路径冲突**: 安装后在源码目录内 import 会找到本地 `flash_mla/`，需 `cd /tmp` 再 import
5. **Dense MLA Decoding**: 仅支持 SM90 (Hopper), B200 (SM100) 跳过

### 6.2 SGLang Docker 镜像选择

| 镜像 | 架构 | B200 兼容 | 说明 |
|------|------|-----------|------|
| `lmsysorg/sglang:latest` | 通用 | ✗ | `deepseek_v4` model_type 不被 transformers 识别 |
| `deepseek-v4-blackwell` | SM100 x86_64 | ✓* | 需修复 config backup + 指定 MXFP4 backend |
| `deepseek-v4-b300` | SM100 x86_64 | ✓* | 同上 |
| `deepseek-v4-hopper` | SM90 | ✗ | `sgl_kernel` 编译目标 SM90 |
| `deepseek-v4-grace-blackwell` | ARM | ✗ | `no matching manifest for linux/amd64` |
| `dev-cu13` | 通用 CUDA 13 | ✗ | 同 `latest` |

### 6.3 MXFP4 Expert 权重维度不匹配 (核心 Bug)

**症状**: `AssertionError: Hidden size mismatch` in `fused_moe.py:341`

**根因**: DeepSeek V4 Flash 的 expert 权重存储为 MXFP4 (int8 打包, 2 个 FP4/byte):
- 磁盘上 `w1.weight: [2048, 2048] int8` → 实际逻辑 `[2048, 4096]`
- `hidden_states.shape[1]=4096` vs `w1.shape[2]=2048` → 2x 不匹配

`config_backup_small.json` 缺少 `expert_dtype: fp4` 字段, 导致默认的 triton MoE runner
不知道需要 MXFP4 解包, 直接用 packed 维度做 shape check。

**修复**: 添加 `expert_dtype: fp4` + `--moe-runner-backend flashinfer_mxfp4`

### 6.4 其他注意事项

- **TP=1 不可行**: BF16 权重 173GB > 单卡 180GB 可用（KV cache 无空间）
- **Config Backup**: SGLang 用 `SGLANG_APPLY_CONFIG_BACKUP` 替换 model config, 将 `deepseek_v4` 映射为内部注册的 `deepseek_ref` 类型
- **DeepGEMM JIT**: 首次启动需编译 32768 个 GEMM kernel，约 40 秒/组，共 ~3 分钟
- **高并发 OOM**: B200 attention 的 einsum 在高并发下 CUBLAS 资源耗尽，限制 max_running_requests ≤ 12

---

### TODO

- [x] Docker 安装 + NVIDIA Container Toolkit (Docker 29.4.2, NCTK 1.19.0)
- [x] FlashMLA kernel benchmark on B200 (4748+617 cases passed, peak 1418 TFLOP/s)
- [x] SGLang TP=8 MXFP4 推理 (修复 config backup + flashinfer_mxfp4 backend)
- [x] 性能压测 (峰值 430 tok/s, 单请求 109-137 tok/s)
- [ ] 尝试 FP8 量化版 `sgl-project/DeepSeek-V4-Flash-FP8` 对比性能
- [ ] EP (Expert Parallelism) 模式测试
