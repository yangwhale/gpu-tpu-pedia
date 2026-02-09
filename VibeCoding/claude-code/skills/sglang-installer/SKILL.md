---
name: sglang-installer
description: This skill should be used when users need to install, configure, debug, or run SGLang inference server on NVIDIA GPUs (especially B200/H100/A100). It covers installation from source, dependency management, environment setup, common error diagnosis and fixes, tensor parallelism configuration, and server startup/testing.
license: MIT
---

# SGLang Installer

This skill provides comprehensive guidance for installing, configuring, and debugging SGLang on NVIDIA GPUs with CUDA 12.x.

## When to Use This Skill

- Installing SGLang from source on NVIDIA GPUs
- Debugging SGLang installation errors (missing libraries, version conflicts)
- Configuring tensor parallelism for different model architectures
- Setting up environment variables for CUDA and NVIDIA libraries
- Starting and testing SGLang inference server
- Fixing common runtime errors (cuDNN, cusparseLt, NCCL issues)

## Version Information (as of v0.5.8)

| Component | Version | Notes |
|-----------|---------|-------|
| SGLang | 0.5.8 | Latest stable (2026-01-29) |
| sgl-kernel | 0.3.21 | PyPI install for CUDA 12.9 |
| mooncake-transfer-engine | 0.3.8.post1 | KV cache transfer (requires nvidia_peermem) |
| nixl | 0.9.0 | KV cache transfer (DMA-BUF, recommended) |
| nvidia-nccl-cu12 | 2.28.3 | Force reinstall |
| nvidia-cudnn-cu12 | 9.16.0.29 | Required for PyTorch 2.9+ |
| flashinfer | 0.6.1 | SGLang 0.5.8 requires 0.6.1 (vLLM uses 0.5.3) |
| sglang-router | 0.5.8 | PD disaggregation 路由 (pip install sglang-router) |

### What's New in v0.5.8

- **1.5x faster diffusion models** across the board
- **Chunked Pipeline Parallelism** for million-token context (near-linear scaling)
- **EPD Disaggregation** for Vision-Language Models (elastic encoder scaling)
- **GLM4-MoE optimization**: 65% faster TTFT
- **New models**: GLM 4.7 Flash, LFM2, Qwen3-VL-Embedding/Reranker, DeepSeek V3.2 NVFP4, FLUX.2-klein-9B

### What's New in v0.5.7

- **Model Gateway v0.3.0** release
- **Scalable Pipeline Parallelism** with dynamic chunking for ultra-long contexts
- **Encoder Disaggregation** for multi-modal models
- **Diffusion improvements**: `--dit-layerwise-offload true` reduces peak VRAM by 30GB
- **New models**: Mimo-V2-Flash, Nemotron-Nano-v3, LLaDA 2.0, EAGLE 3 speculative decoding
- **Hardware support**: AMD/4090/5090 for diffusion

## Mooncake Transfer Engine

Mooncake is required for **prefill-decode disaggregation** mode, which separates prefill and decode phases across different nodes for production deployments.

### Installing Mooncake

```bash
pip install --break-system-packages mooncake-transfer-engine==0.3.8.post1
```

### Verifying Mooncake

```bash
python3 -c "from mooncake.engine import TransferEngine; print('Mooncake OK')"
```

### When is Mooncake Needed?

Mooncake is required when using:
- `--disaggregation-mode prefill` or `--disaggregation-mode decode`
- Multi-node deployments with KV cache transfer
- Production DeepSeek-V3/R1 deployments with prefill-decode separation

**Note:** For single-node testing without disaggregation, Mooncake is not required.

## NIXL Transfer Engine (Recommended)

NIXL (NVIDIA Inference Xfer Library) is an alternative to Mooncake that uses **DMA-BUF** instead of nvidia_peermem. It's the recommended choice when:

- Using NVIDIA Open Kernel Module (nvidia_peermem won't load)
- nvidia_peermem fails with "Invalid argument" error
- You want a more portable solution that doesn't depend on kernel modules

### Installing NIXL

```bash
pip install --break-system-packages nixl==0.9.0

# IMPORTANT: NIXL may downgrade NVIDIA libraries, reinstall correct versions:
pip install nvidia-nccl-cu12==2.28.3 --force-reinstall --no-deps
pip install nvidia-cudnn-cu12==9.16.0.29 --force-reinstall --no-deps
```

**注意**: NIXL 还会安装 `nvidia-nvshmem-cu12==3.4.5`，这个 pip 包**不会被使用**。
DeepEP 使用的是自编译的 NVSHMEM 3.5.19（带 IBGDA 支持），通过 `unified-env.sh` 中的 `LD_PRELOAD` 加载。

### Verifying NIXL

```bash
python3 -c "import nixl; print('NIXL OK')"
```

### Using NIXL for Disaggregation

Add `--disaggregation-transfer-backend nixl` to your launch command:

```bash
python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend nixl \  # Use NIXL instead of Mooncake
    --tp-size 8 \
    ...
```

### NIXL vs Mooncake

| Feature | NIXL | Mooncake |
|---------|------|----------|
| Memory registration | DMA-BUF (kernel native) | nvidia_peermem (kernel module) |
| Transport | UCX (TCP/RDMA/SHM) | RDMA or TCP |
| Kernel module required | No | nvidia_peermem (may fail) |
| Open Kernel Module compatible | Yes | No (fails to load) |
| Recommended for | NVIDIA Open driver, B200 | Legacy systems with nvidia_peermem |

**Recommendation:** Use NIXL for new deployments, especially on systems with NVIDIA Open Kernel Module.

## Installation Workflow

### Pre-requisites (Ubuntu 24.04)

Ubuntu 24.04 doesn't include pip by default. Install it first:

```bash
sudo apt-get update
sudo apt-get install -y python3-pip
```

### Step 1: Environment Setup

To set up the environment, ensure CUDA is properly configured:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export BUILD_TYPE=blackwell  # or "all" for general, "hopper" for H100
```

### Step 2: Clone and Install

To install SGLang from source:

```bash
mkdir -p /sgl-workspace && cd /sgl-workspace

# Clone specific version
git clone -b v0.5.8 --depth 1 https://github.com/sgl-project/sglang.git
cd sglang

# Install sgl-kernel first (for CUDA 12.9)
pip install sgl-kernel==0.3.21

# Install SGLang with blackwell support
pip install -e "python[blackwell]" --extra-index-url https://download.pytorch.org/whl/cu129
```

### Step 3: Install Additional Dependencies

To install required NVIDIA libraries and NIXL:

```bash
# NVIDIA libraries (required)
pip install nvidia-nccl-cu12==2.28.3 --force-reinstall --no-deps
pip install nvidia-cudnn-cu12==9.16.0.29 --force-reinstall --no-deps

# NIXL for KV cache transfer (RECOMMENDED for disaggregation mode)
pip install --break-system-packages nixl==0.9.0
# Re-install NVIDIA libs after NIXL (NIXL may downgrade them)
pip install nvidia-nccl-cu12==2.28.3 --force-reinstall --no-deps
pip install nvidia-cudnn-cu12==9.16.0.29 --force-reinstall --no-deps
```

> ⚠️ **Important**: NIXL is required for prefill-decode disaggregation mode. If you skip NIXL, you'll need nvidia_peermem kernel module (often fails on NVIDIA Open driver).

### Step 4: Configure LD_LIBRARY_PATH

To fix library loading issues, run `scripts/setup_env.sh` or manually set:

```bash
# Collect all nvidia pip package lib paths
NVIDIA_LIB_PATHS=""
for d in /usr/local/lib/python3.12/dist-packages/nvidia/*/lib; do
    [ -d "$d" ] && NVIDIA_LIB_PATHS="${d}:${NVIDIA_LIB_PATHS}"
done
for d in $HOME/.local/lib/python3.12/site-packages/nvidia/*/lib; do
    [ -d "$d" ] && NVIDIA_LIB_PATHS="${d}:${NVIDIA_LIB_PATHS}"
done
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${NVIDIA_LIB_PATHS}${LD_LIBRARY_PATH}
```

## Common Errors and Fixes

### Error: DeepSeek-V3 FP8 outputs garbage on Blackwell (B200)

**Symptom:**
```
Model outputs garbage characters like "################" or random symbols
```

**Log Warning:**
```
WARNING model_config.py:872: DeepGemm is enabled but the scale_fmt of checkpoint is not ue8m0. This might cause accuracy degradation on Blackwell.
```

**Diagnosis:** The DeepSeek-V3 official FP8 checkpoint uses `mscale` format which is incompatible with DeepGEMM on Blackwell (B200) GPUs. DeepGEMM expects `ue8m0` scale format.

**Workaround:** Use **vLLM** instead of SGLang for DeepSeek-V3 on Blackwell:
```bash
# vLLM handles the FP8 scale format correctly
vllm serve deepseek-ai/DeepSeek-V3 --tensor-parallel-size 8 --port 8000 --trust-remote-code
```

**Status:** Known issue as of SGLang 0.5.8 on Blackwell GPUs. Works correctly on H100/A100.

| Framework | DeepSeek-V3 FP8 on Blackwell |
|-----------|------------------------------|
| SGLang 0.5.8 | ❌ Garbage output |
| vLLM 0.14.1 | ✅ Works correctly |

---

### Error: Cannot uninstall typing_extensions (Ubuntu 24.04)

**Symptom:**
```
ERROR: Cannot uninstall typing_extensions 4.10.0, RECORD file not found.
Hint: The package was installed by debian.
```

**Diagnosis:** Ubuntu 24.04 installs `typing_extensions` as a system package managed by apt.

**Fix:**
```bash
pip install -e "python[blackwell]" --ignore-installed typing_extensions --break-system-packages
```

---

### Error: sgl-kernel ABI mismatch (PyTorch user/system conflict)

**Symptom:**
```
ImportError: .../sgl_kernel/sm100/common_ops.abi3.so: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib
```

**Diagnosis:** PyTorch installed in both user (`~/.local/lib/python3.12/site-packages/`) and system (`/usr/local/lib/python3.12/dist-packages/`) directories with different versions.

**Check:**
```bash
pip3 show torch | grep -E "Version|Location"
ls /usr/local/lib/python3.12/dist-packages/ | grep torch
ls ~/.local/lib/python3.12/site-packages/ | grep torch
```

**Fix:** Remove user-installed torch to use system version:
```bash
pip3 uninstall torch torchvision torchaudio -y --break-system-packages
python3 -c "import torch; print(torch.__version__)"  # Should show system version
```

---

### Error: libcudnn.so.9 not found

**Symptom:**
```
ImportError: libcudnn.so.9: cannot open shared object file
```

**Fix:**
```bash
pip install nvidia-cudnn-cu12==9.16.0.29 --force-reinstall --no-deps
# Then set LD_LIBRARY_PATH as described above
```

### Error: libcusparseLt.so.0 not found

**Symptom:**
```
ImportError: libcusparseLt.so.0: cannot open shared object file
```

**Fix:**
```bash
pip install nvidia-cusparselt-cu12
# Then set LD_LIBRARY_PATH as described above
```

### Error: assert self.total_num_heads % tp_size == 0

**Symptom:**
```
AssertionError: assert self.total_num_heads % tp_size == 0
```

**Diagnosis:** The model's attention head count is not divisible by the tensor parallelism size.

**Fix:** Choose a `--tp` value that divides the model's attention head count:

| Model | Attention Heads | Valid TP Values |
|-------|-----------------|-----------------|
| Qwen2.5-7B | 28 | 1, 2, 4, 7, 14 |
| Qwen2.5-72B | 64 | 1, 2, 4, 8, 16, 32 |
| Llama-3-8B | 32 | 1, 2, 4, 8, 16, 32 |
| Llama-3-70B | 64 | 1, 2, 4, 8, 16, 32 |
| DeepSeek-R1 | 128 | 1, 2, 4, 8, 16, 32, 64 |

To find the attention head count for any model:
```bash
python3 -c "from transformers import AutoConfig; c = AutoConfig.from_pretrained('MODEL_NAME'); print(f'Attention heads: {c.num_attention_heads}')"
```

### Error: NCCL errors or timeouts

**Fix:**
```bash
pip install nvidia-nccl-cu12==2.28.3 --force-reinstall --no-deps
```

### Error: sgl-kernel version mismatch

**Symptom:** SGLang installs an older sgl-kernel version than expected.

**Note:** As of v0.5.6.post2, SGLang's dependencies pin sgl-kernel to 0.3.19, so even if you pre-install 0.3.21, it will be downgraded during SGLang installation. This is expected behavior and 0.3.19 works correctly.

**If you need a specific version:** Install sgl-kernel AFTER SGLang:
```bash
pip install -e "python[blackwell]" ...
pip install sgl-kernel==0.3.21 --force-reinstall --no-deps  # if needed
```

### Error: sgl-kernel ABI incompatibility (undefined symbol)

**Symptom:**
```
ImportError: .../sgl_kernel/sm100/common_ops.abi3.so: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib
```

**Diagnosis:** sgl-kernel was compiled against a different PyTorch version than currently installed. This commonly happens when:
1. FlashInfer installation upgrades PyTorch to 2.10
2. vLLM installation changes PyTorch to 2.9.1
3. Manual PyTorch version changes

**Fix:**
1. First, check your current PyTorch version:
   ```bash
   python3 -c "import torch; print(torch.__version__)"
   ```

2. For vLLM compatibility (PyTorch 2.9.1), reinstall sgl-kernel:
   ```bash
   pip install torch==2.9.1+cu129 --index-url https://download.pytorch.org/whl/cu129 --force-reinstall
   pip install sgl-kernel==0.3.21 --force-reinstall --no-deps
   pip install nvidia-nccl-cu12==2.28.3 nvidia-cudnn-cu12==9.16.0.29 --force-reinstall --no-deps
   ```

3. For standalone SGLang (no vLLM), use the PyTorch version from SGLang installation.

**Root Cause:** The sgl-kernel binary is compiled against specific PyTorch CUDA APIs. When PyTorch version changes, the ABI symbols may not match.

### Error: FlashInfer changes PyTorch version

**Symptom:** After installing FlashInfer, other packages fail with version conflicts.

**Diagnosis:** `flashinfer-python` and `flashinfer-cubin` have their own PyTorch dependencies that may override your installed version.

**Fix:** After FlashInfer installation, always reinstall the correct PyTorch and NVIDIA libraries:
```bash
# For vLLM compatibility
pip install torch==2.9.1+cu129 --index-url https://download.pytorch.org/whl/cu129 --force-reinstall
pip install nvidia-nccl-cu12==2.28.3 nvidia-cudnn-cu12==9.16.0.29 --force-reinstall --no-deps
pip install sgl-kernel==0.3.21 --force-reinstall --no-deps
```

### Error: num_max_dispatch_tokens_per_rank assertion

**Symptom:**
```
assert self.num_max_dispatch_tokens_per_rank <= 1024
AssertionError
```

**Diagnosis:** `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` is set to a value > 1024.

**Fix:** Set the value to 1024 or less:
```bash
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024
```

### Error: Mooncake not installed

**Symptom:**
```
ModuleNotFoundError: No module named 'mooncake'
ImportError: Please install mooncake by following the instructions...
```

**Diagnosis:** Using `--disaggregation-mode prefill/decode` without Mooncake installed.

**Fix:**
```bash
pip install --break-system-packages mooncake-transfer-engine==0.3.8.post1
```

### Error: NIXL library version mismatch

**Symptom:**
Installing NIXL downgrades NVIDIA libraries, causing import errors.

**NIXL 会降级以下库:**
- `nvidia-nccl-cu12`: 2.28.3 → 2.27.5
- `nvidia-cudnn-cu12`: 9.16.0.29 → 9.10.2.21
- `nvidia-nvshmem-cu12`: 安装 3.4.5 (pip 包版本)

**Fix:**
After installing NIXL, reinstall NVIDIA libraries:
```bash
pip install nvidia-nccl-cu12==2.28.3 --force-reinstall --no-deps
pip install nvidia-cudnn-cu12==9.16.0.29 --force-reinstall --no-deps
```

**关于 nvidia-nvshmem-cu12:**
NIXL 安装的 `nvidia-nvshmem-cu12` (3.4.5) 是 pip 包版本，**不会影响 DeepEP**。
DeepEP 使用自编译的 NVSHMEM (3.5.19，带 IBGDA 支持)，通过 `unified-env.sh` 中的 `LD_PRELOAD` 强制加载。

### Error: DeepGEMM compile_deep_gemm OOM (CUDA out of memory)

**Symptom:**
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 7.00 GiB. GPU 0 has a total capacity of 178.35 GiB of which 2.31 GiB is free.
```

**Diagnosis:** `compile_deep_gemm` 默认在单卡上加载整个模型。DeepSeek-V3 (671B) 远超单卡 183GB 显存。

**Fix:** 必须指定 `--tp` 参数，与后续 `launch_server` 的 `--tp-size` 一致：
```bash
# 错误: 不加 --tp，单卡加载 671B 模型 → OOM
python3 -m sglang.compile_deep_gemm --model-path deepseek-ai/DeepSeek-V3

# 正确: 用 TP=8 分布到 8 卡
python3 -m sglang.compile_deep_gemm --model-path deepseek-ai/DeepSeek-V3 --tp 8
```

**注意:** cubin 缓存路径是 `~/.cache/deep_gemm/cache/`（不是 `~/.deep_gemm/cache/`），预期产出 ~692 个文件。

---

### Error: DeepGEMM JIT compilation causes DeepEP dispatch timeout

**Symptom:**
```
RuntimeError: DeepEP error: timeout (dispatch CPU)
# 或者 server warmup 阶段长时间卡住
```

**Diagnosis:** MoE 模型首次启动时，DeepGEMM JIT 编译 ~692 个 cubin 文件耗时 10-20 分钟。在多 DP rank 或 PD disaggregation 场景下，不同 rank 编译速度不同步，导致 DeepEP all-to-all 通信超时。

**Fix:** 在启动 server 前预编译 DeepGEMM：
```bash
source /opt/deepep/unified-env.sh
export HF_HOME=/lssd/huggingface
# IMPORTANT: --tp 必须与 server 启动时的 --tp-size 一致，避免单卡 OOM
python3 -m sglang.compile_deep_gemm --model-path deepseek-ai/DeepSeek-V3 --tp 8
# 编译完成后再启动 server
```

### Error: Direct request to Prefill node (bootstrap_room assertion)

**Symptom:**
```
AssertionError: bootstrap_room should not be None
```

**Diagnosis:** 直接向 Prefill 节点发送请求，而非通过 sglang-router。

**Fix:** 请求必须通过 sglang-router 路由：
```bash
# 错误: curl http://prefill-ip:30000/v1/chat/completions ...
# 正确: curl http://router-ip:30080/v1/chat/completions ...
```

### Error: Direct request to Decode node (bootstrap room id)

**Symptom:**
```
400 Bad Request: Disaggregated request received without bootstrap room id
```

**Diagnosis:** 直接向 Decode 节点发送请求，而非通过 sglang-router。

**Fix:** 同上，请求必须通过 sglang-router。PD disaggregation 模式下，Prefill 和 Decode 不接受直接客户端请求。

### Error: dist-init-addr bind failure on Decode node

**Symptom:**
```
zmq.error.ZMQError: No such device (addr: tcp://10.8.0.25:5757)
```

**Diagnosis:** `--dist-init-addr` 使用了 Prefill 节点的特定 IP，Decode 节点无法 bind 该地址。

**Fix:** 使用 `0.0.0.0` 代替特定 IP：
```bash
# 错误: --dist-init-addr 10.8.0.25:5757
# 正确: --dist-init-addr 0.0.0.0:5757
```

### Error: Deprecated environment variable warning

**Symptom:**
```
UserWarning: Environment variable SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK is deprecated
```

**Fix:** Use the new variable name:
```bash
# Old (deprecated)
export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=1

# New
export SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1
```

## DeepGEMM Precompilation (MoE Models)

MoE 模型（DeepSeek-V3/R1 等）使用 DeepGEMM 进行 FP8 矩阵乘法。DeepGEMM 采用 JIT 编译，首次启动时会编译 ~692 个 cubin 文件。

**问题:** 在 PD disaggregation 或多 DP rank 场景下，JIT 编译耗时 10-20 分钟，导致 DP rank 间不同步，触发 DeepEP dispatch 超时错误。

**解决方案:** 在启动 server 前预编译 DeepGEMM：

```bash
source /opt/deepep/unified-env.sh
export HF_HOME=/lssd/huggingface

# 预编译 DeepGEMM cubin（针对特定模型的 GEMM 维度）
# IMPORTANT: --tp 必须与 server 启动时的 --tp-size 一致
# DeepSeek-V3 (671B) 单卡放不下，不加 --tp 会 OOM
python3 -m sglang.compile_deep_gemm --model-path deepseek-ai/DeepSeek-V3 --tp 8

# 验证 cubin 缓存
ls ~/.cache/deep_gemm/cache/ | wc -l
# 预期: ~692 个文件（DeepSeek-V3 架构）
```

**注意:**
- **--tp 参数**: 必须与后续 `launch_server` 的 `--tp-size` 一致。不加 `--tp` 默认单卡加载，DeepSeek-V3 (671B) 会触发 CUDA OOM
- DeepGEMM 编译结果是半模型特定的：基于模型的 GEMM 维度（M/N/K），相同架构（如 DeepSeek-V3 和 DeepSeek-R1）可共享缓存
- 不同架构的模型（如 Qwen-MoE vs DeepSeek）需要重新编译
- cubin 缓存目录: `~/.cache/deep_gemm/cache/`（注意不是 `~/.deep_gemm/cache/`）
- 预编译后缓存持久化在磁盘上，重启后无需重新编译

## Starting the Server

To start the SGLang server:

```bash
# IMPORTANT: 必须先加载 DeepEP 环境（设置 LD_PRELOAD 等）
source /opt/deepep/unified-env.sh

# 设置 HuggingFace 缓存目录（可选，使用 LSSD 加速）
export HF_HOME=/lssd/huggingface

# MoE 模型: 先预编译 DeepGEMM（避免 warmup 超时）
# --tp 必须与 launch_server 的 --tp-size 一致，避免单卡 OOM
python3 -m sglang.compile_deep_gemm --model-path deepseek-ai/DeepSeek-V3 --tp 8

# Start server (adjust tp based on model architecture)
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --port 30000 \
    --host 0.0.0.0 \
    --tp 4 \
    --trust-remote-code
```

**注意**: 如果不 source `unified-env.sh`，DeepEP 会因为找不到正确的 NVSHMEM 库而报错。

## Disaggregation Mode (Prefill-Decode Separation)

For production DeepSeek-V3/R1 deployments, SGLang supports prefill-decode disaggregation where prefill and decode phases run on separate nodes.

### Prerequisites

1. **Transfer backend** - one of:
   - **NIXL** (recommended): `pip install nixl==0.9.0`
   - **Mooncake**: `pip install mooncake-transfer-engine==0.3.8.post1` (requires nvidia_peermem)
2. **DeepEP** for MoE all-to-all communication
3. **DeepEP config files** for expert placement
4. **sglang-router** for request routing (必须):
   ```bash
   pip install --break-system-packages sglang-router
   ```

### Architecture

PD disaggregation 的请求流程：

```
Client → sglang-router → Prefill Node (port 30000)
                       → Decode Node  (port 30001)
```

**IMPORTANT:** 客户端请求必须发送到 sglang-router，不能直接发送到 Prefill 或 Decode 节点。
- 直接请求 Prefill: `AssertionError: bootstrap_room should not be None`
- 直接请求 Decode: `400 Disaggregated request received without bootstrap room id`

### DeepSeek-V3 Prefill Node Example

```bash
source /opt/deepep/unified-env.sh

export HF_TOKEN=your_token_here

# IMPORTANT: Must be <= 1024, otherwise assertion error
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024 \
MC_TE_METRIC=true \
SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 \
SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
SGLANG_MOONCAKE_CUSTOM_MEM_POOL=false \
SGLANG_LOCAL_IP_NIC=enp0s19 \
GLOO_SOCKET_IFNAME=enp0s19 \
NCCL_SOCKET_IFNAME=enp0s19 \
NCCL_MNNVL_ENABLE=1 \
NCCL_CUMEM_ENABLE=1 \
SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --download-dir /lssd/huggingface/hub \
    --trust-remote-code \
    --disaggregation-mode prefill \
    --dist-init-addr 0.0.0.0:5757 \
    --nnodes 1 \
    --node-rank 0 \
    --tp-size 8 \
    --dp-size 8 \
    --enable-dp-attention \
    --host 0.0.0.0 \
    --context-length 2176 \
    --disable-radix-cache \
    --moe-dense-tp-size 1 \
    --enable-dp-lm-head \
    --disable-shared-experts-fusion \
    --ep-num-redundant-experts 32 \
    --eplb-algorithm deepseek \
    --deepep-config /path/to/deepep_config.json \
    --attention-backend cutlass_mla \
    --watchdog-timeout 1000000 \
    --init-expert-location /path/to/prefill_in4096.json \
    --disable-cuda-graph \
    --chunked-prefill-size 16384 \
    --max-total-tokens 32768 \
    --moe-a2a-backend deepep \
    --deepep-mode normal \
    --ep-dispatch-algorithm dynamic
```

### Key Configuration Files

1. **deepep_config.json** - DeepEP SM configuration:
```json
{
    "n_sms": 128,
    "normal_dispatch": {"num_sms": 128},
    "normal_combine": {"num_sms": 128}
}
```

2. **prefill_in4096.json** - Expert placement statistics for EPLB

### Environment Variables

| Variable | Value | Description |
|----------|-------|-------------|
| SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK | 1024 | Max dispatch tokens per rank (MUST be <= 1024) |
| SGLANG_LOCAL_IP_NIC | enp0s19 | Network interface for local IP |
| GLOO_SOCKET_IFNAME | enp0s19 | Gloo communication interface |
| NCCL_SOCKET_IFNAME | enp0s19 | NCCL communication interface |
| NCCL_MNNVL_ENABLE | 1 | Enable Multi-Node NVLink |
| MC_TE_METRIC | true | Enable Mooncake metrics |

### RDMA Memory Registration Errors (Mooncake)

When running with Mooncake, you may see RDMA memory registration errors:
```
RdmaTransport: Failed to register memory: addr 0x... length 37896192
```

**Root Cause:** nvidia_peermem module is not loaded or incompatible with your driver.

**Diagnosis:**
```bash
# Check if nvidia_peermem loads
sudo modprobe nvidia_peermem
# If you see: "could not insert 'nvidia_peermem': Invalid argument"
# This means you're using NVIDIA Open Kernel Module, which is incompatible
```

**Solutions (in order of preference):**

1. **Switch to NIXL backend** (recommended):
   ```bash
   pip install --break-system-packages nixl==0.9.0
   # Add to launch command:
   --disaggregation-transfer-backend nixl
   ```

2. **Use TCP fallback** (slower):
   Mooncake will automatically fall back to TCP, but this significantly impacts multi-node performance.

3. **Load nvidia_peermem** (only works with proprietary driver):
   ```bash
   sudo modprobe nvidia_peermem
   ```

**Note:** NIXL uses DMA-BUF which is built into the Linux kernel and doesn't require nvidia_peermem.

### 1P1D Quick Start (Minimal DeepSeek-V3 Example)

验证过的最小化 1P1D 部署步骤（2 节点 B200）：

**Step 1: 两节点预编译 DeepGEMM**

```bash
# 在 Prefill 和 Decode 节点上都执行
source /opt/deepep/unified-env.sh
export HF_HOME=/lssd/huggingface
# --tp 8 必须加，否则单卡加载 671B 模型会 OOM
python3 -m sglang.compile_deep_gemm --model-path deepseek-ai/DeepSeek-V3 --tp 8
# 验证: ls ~/.cache/deep_gemm/cache/ | wc -l  → 预期 ~692
```

**Step 2: Prefill Node (e.g. 10.8.0.25)**

```bash
source /opt/deepep/unified-env.sh
export HF_HOME=/lssd/huggingface

SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024 \
NCCL_SOCKET_IFNAME=enp0s19 \
GLOO_SOCKET_IFNAME=enp0s19 \
python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --port 30000 \
    --host 0.0.0.0 \
    --tp-size 8 \
    --trust-remote-code \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend nixl \
    --dist-init-addr 0.0.0.0:5757 \
    --moe-dense-tp-size 1 \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --deepep-mode normal \
    --disable-cuda-graph \
    --watchdog-timeout 1000000
```

**Step 3: Decode Node (e.g. 10.8.0.71)**

```bash
source /opt/deepep/unified-env.sh
export HF_HOME=/lssd/huggingface

SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024 \
NCCL_SOCKET_IFNAME=enp0s19 \
GLOO_SOCKET_IFNAME=enp0s19 \
python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --port 30001 \
    --host 0.0.0.0 \
    --tp-size 8 \
    --trust-remote-code \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend nixl \
    --dist-init-addr 0.0.0.0:5757 \
    --moe-dense-tp-size 1 \
    --enable-dp-attention \
    --moe-a2a-backend deepep \
    --deepep-mode normal \
    --disable-cuda-graph \
    --watchdog-timeout 1000000
```

**Step 4: sglang-router (在 Prefill 节点或独立节点上)**

```bash
pip install --break-system-packages sglang-router

python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --mini-lb \
    --prefill http://10.8.0.25:30000 \
    --decode http://10.8.0.71:30001 \
    --host 0.0.0.0 \
    --port 30080
```

**Step 5: 测试**

```bash
# 请求必须发送到 router (port 30080)，不能直接发到 Prefill/Decode
curl http://10.8.0.25:30080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V3",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

### dist-init-addr 注意事项

**CRITICAL:** `--dist-init-addr` 必须使用 `0.0.0.0:5757`，不能使用特定 IP（如 `10.8.0.25:5757`）。

原因：SGLang 内部使用 `dist-init-addr` 进行 ZMQ TCP bind。如果使用特定 IP，Decode 节点会尝试 bind 到 Prefill 节点的 IP，导致 bind 失败。

## Testing the Server

To verify the server is working:

```bash
# Health check
curl http://localhost:30000/health

# Chat completion test
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

## Diagnostic Script

To diagnose installation issues, run `scripts/diagnose.py`:

```bash
python3 scripts/diagnose.py
```

This script checks:
- CUDA installation and version
- PyTorch CUDA compatibility
- SGLang and sgl-kernel versions
- Required library availability
- GPU detection and memory
- LD_LIBRARY_PATH configuration
- **DeepEP installation** (required for MoE models)

## DeepEP Dependency

DeepEP (DeepSeek Expert Parallelism) is required for running MoE (Mixture of Experts) models:
- DeepSeek-V3
- DeepSeek-R1
- Mixtral (with Expert Parallelism)

### Detecting DeepEP

The diagnostic script automatically checks for DeepEP. If not installed, it will prompt:

```
⚠ DeepEP: Not installed
  DeepEP is required for MoE models (DeepSeek-V3, DeepSeek-R1)
  To install DeepEP, use the deepep-installer skill:
    /deepep-installer
```

### Installing DeepEP

If DeepEP is not installed, use the `deepep-installer` skill:

```bash
# Option 1: Use the deepep-installer skill (recommended)
/deepep-installer

# Option 2: Run the installation script directly
bash /path/to/gpu-tpu-pedia/gpu/deepep/install.sh
```

The `deepep-installer` skill handles:
- CUDA and gdrcopy setup
- NVSHMEM with IBGDA support
- DeepEP compilation for your GPU architecture
- Environment variable configuration

### Workflow Integration

When diagnosing SGLang installation, if DeepEP is missing and the user wants to run MoE models:

1. Detect DeepEP is not installed via `diagnose.py`
2. Prompt user to run `/deepep-installer` skill
3. After DeepEP installation, re-run SGLang diagnostic
4. Proceed with SGLang server startup

## Recommended Installation Order

For MoE models (DeepSeek, Qwen-MoE), the recommended installation order is:

1. **DeepEP first** (if needed for MoE models)
   - gdrcopy → NVSHMEM → DeepEP
   - Use the `deepep-installer` skill

2. **Then SGLang**
   - SGLang installation script will detect and use DeepEP if available

This ensures DeepEP is properly configured before SGLang tries to use it.

## SGLang/vLLM Coexistence

SGLang and vLLM can be installed on the same system, but they have some dependency version conflicts (grpcio, timm, xgrammar, etc.). For production use:

- **Recommended**: Use separate Python virtual environments
- **Alternative**: Accept the version mismatches (usually works for basic inference)

Common conflicts when both are installed:
- `grpcio`: SGLang wants 1.75.1, vLLM may install 1.76.0
- `timm`: SGLang wants 1.0.16, vLLM may install 1.0.24
- `xgrammar`: SGLang wants 0.1.27, vLLM may install 0.1.29

## Pre-downloading DeepSeek Weights (Optional)

For faster DeepSeek-V3/R1 model loading, you can pre-download weights from GCS instead of HuggingFace:

```bash
# Check if already downloaded
DEEPSEEK_PATH="/lssd/huggingface/hub/models--deepseek-ai--DeepSeek-V3"

if [ -d "$DEEPSEEK_PATH" ]; then
    echo "✓ DeepSeek-V3 weights already exist: $DEEPSEEK_PATH"
    du -sh "$DEEPSEEK_PATH"
else
    echo "Downloading DeepSeek-V3 weights from GCS..."
    gcloud storage cp -r gs://chrisya-gpu-pg-ase1/huggingface /lssd/
    echo "✓ DeepSeek-V3 weights downloaded"
fi
```

**Notes:**
- GCS bucket `gs://chrisya-gpu-pg-ase1/huggingface` contains pre-cached DeepSeek-V3 FP8 weights
- Downloading from GCS is much faster than HuggingFace (same-region high bandwidth)
- Weights are ~600GB, including complete safetensors files
- Requires LSSD to be mounted first (use `/lssd-mounter` skill)

## Resources

- `scripts/diagnose.py` - Diagnostic script for installation issues
- `scripts/setup_env.sh` - Environment variable setup script
- `references/version_matrix.md` - Version compatibility matrix
- `references/troubleshooting.md` - Extended troubleshooting guide

## Unified Environment Script

After installing DeepEP + SGLang + vLLM, use the unified environment script:

```bash
source /opt/deepep/unified-env.sh
```

This script sets up all necessary environment variables for DeepEP, NVSHMEM, gdrcopy, and NVIDIA libraries.

## Recommended Multi-Framework Installation Order

When installing SGLang alongside DeepEP and vLLM:

```
1. /lssd-mounter     → Mount high-speed local SSD
2. /deepep-installer → Install DeepEP (已使用 PyTorch 2.9.1)
3. /sglang-installer → Install SGLang (this skill)
4. /vllm-installer   → Install vLLM (可选)
```

**注意**: DeepEP 现在默认使用 PyTorch 2.9.1 编译，与 SGLang 0.5.8 保持一致，无需重新编译。

**Post-Installation Verification:**
```bash
source /opt/deepep/unified-env.sh
python3 -c "
import torch; print(f'PyTorch: {torch.__version__}')
import deep_ep; print('DeepEP: OK')
import sglang; print(f'SGLang: {sglang.__version__}')
import sgl_kernel; print(f'sgl-kernel: {sgl_kernel.__version__}')
import vllm; print(f'vLLM: {vllm.__version__}')
"
```

## Version History

- **2026-02-09**: DeepGEMM 预编译实战修复 (b3 安装验证)
  - **CRITICAL**: `compile_deep_gemm` 必须加 `--tp 8`，否则单卡加载 DeepSeek-V3 (671B) 会 OOM
  - **FIX**: DeepGEMM cubin 缓存路径为 `~/.cache/deep_gemm/cache/`（非 `~/.deep_gemm/cache/`）
  - **NEW**: 新增 "DeepGEMM compile_deep_gemm OOM" 错误条目
  - 更新所有 `compile_deep_gemm` 命令示例添加 `--tp 8` 和 `export HF_HOME`

- **2026-02-08**: PD disaggregation 实战经验 (b2+b3 1P1D 验证)
  - **NEW**: DeepGEMM 预编译节 (`python3 -m sglang.compile_deep_gemm`)
  - **NEW**: sglang-router 安装和使用（PD disaggregation 必须组件）
  - **NEW**: 1P1D Quick Start 完整部署示例（Prefill + Decode + Router）
  - **CRITICAL**: `--dist-init-addr` 必须使用 `0.0.0.0:5757`，不能使用特定 IP
  - **CRITICAL**: 客户端请求必须发到 router，直接发到 Prefill/Decode 会报错
  - **NEW**: 4 个新错误条目（DeepGEMM JIT 超时、bootstrap_room、bootstrap room id、dist-init-addr bind）
  - **FIX**: 生产 PD disaggregation 示例中的 dist-init-addr 从 `<MASTER_IP>` 改为 `0.0.0.0`
  - 添加 sglang-router 到版本表

- **2026-01-29**: DeepSeek-V3 FP8 Blackwell compatibility issue
  - **CRITICAL**: Documented DeepSeek-V3 FP8 outputs garbage on Blackwell (B200) GPUs
  - **ROOT CAUSE**: FP8 scale format mismatch (`mscale` vs `ue8m0` expected by DeepGEMM)
  - **WORKAROUND**: Use vLLM instead of SGLang for DeepSeek-V3 on Blackwell
  - Works correctly on H100/A100

- **2026-01-29**: Installation experience updates
  - **VERSION**: Corrected flashinfer version to 0.6.1 (SGLang 0.5.8 requires flashinfer_python==0.6.1)
  - **FIX**: Added `--ignore-installed typing_extensions` for Ubuntu 24.04 (system package conflict)
  - **FIX**: Documented PyTorch user/system directory conflict when mixing pip and sudo pip
  - **NOTE**: When vLLM is installed after SGLang, flashinfer is downgraded to 0.5.3 (acceptable for inference)

- **2026-01-29**: Added PyTorch/sgl-kernel ABI compatibility fixes
  - **CRITICAL**: Added sgl-kernel ABI incompatibility error and fix
  - **CRITICAL**: Documented FlashInfer changing PyTorch version issue
  - **NEW**: Added unified environment script reference
  - **NEW**: Added recommended multi-framework installation order
  - **NEW**: Added post-installation verification command

- **2026-01-29**: Added GCS DeepSeek weights pre-download
  - **NEW**: Added "Pre-downloading DeepSeek Weights" section
  - GCS source: `gs://chrisya-gpu-pg-ase1/huggingface`
  - Faster than HuggingFace download (same-region bandwidth)

- **2026-01-29**: Updated to SGLang v0.5.8
  - **VERSION BUMP**: SGLang 0.5.6.post2 → 0.5.8
  - **NEW**: Added v0.5.8 highlights (1.5x faster diffusion, chunked pipeline parallelism, EPD)
  - **NEW**: Added v0.5.7 highlights (Model Gateway v0.3.0, encoder disaggregation)
  - **NEW**: Added NIXL to Step 3 as recommended dependency
  - Updated git clone command to v0.5.8

- **2026-01-29**: Added NIXL transfer backend support
  - **NEW**: Added NIXL as recommended transfer backend (uses DMA-BUF, no nvidia_peermem needed)
  - **NEW**: NIXL installation and configuration instructions
  - **NEW**: NIXL vs Mooncake comparison table
  - **NEW**: RDMA memory registration error diagnosis and solutions
  - Updated prerequisites to include NIXL as preferred option
  - Added warning about NIXL downgrading NVIDIA libraries

- **2026-01-29**: Major update for DeepSeek-V3 disaggregation mode
  - **NEW**: Added Mooncake Transfer Engine installation instructions
  - **NEW**: Added prefill-decode disaggregation mode documentation
  - **NEW**: Added DeepSeek-V3 prefill node deployment example
  - **NEW**: Added SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK error fix (must be <= 1024)
  - **NEW**: Added deprecated environment variable warning fix
  - Clarified sgl-kernel version behavior (0.3.19 is pinned by SGLang dependencies)
  - Added note about NVIDIA library reinstallation after install

- **2026-01-28**: Updated based on installation experience
  - Added pip installation for Ubuntu 24.04
  - Added recommended installation order (DeepEP first for MoE)
  - Documented SGLang/vLLM dependency conflicts
