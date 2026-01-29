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

## Version Information (as of v0.5.6.post2)

| Component | Version | Notes |
|-----------|---------|-------|
| SGLang | 0.5.6.post2 | Latest stable |
| sgl-kernel | 0.3.21 | PyPI install for CUDA 12.9 |
| mooncake-transfer-engine | 0.3.8.post1 | KV cache transfer for disaggregation |
| nvidia-nccl-cu12 | 2.28.3 | Force reinstall |
| nvidia-cudnn-cu12 | 9.16.0.29 | Required for PyTorch 2.9+ |
| flashinfer | 0.5.3 | Attention backend |

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
git clone -b v0.5.6.post2 --depth 1 https://github.com/sgl-project/sglang.git
cd sglang

# Install sgl-kernel first (for CUDA 12.9)
pip install sgl-kernel==0.3.21

# Install SGLang with blackwell support
pip install -e "python[blackwell]" --extra-index-url https://download.pytorch.org/whl/cu129
```

### Step 3: Install Additional Dependencies

To install required NVIDIA libraries:

```bash
pip install nvidia-nccl-cu12==2.28.3 --force-reinstall --no-deps
pip install nvidia-cudnn-cu12==9.16.0.29 --force-reinstall --no-deps
```

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

## Starting the Server

To start the SGLang server:

```bash
# Load environment
source /sgl-workspace/sglang-env.sh

# Start server (adjust tp based on model architecture)
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --port 30000 \
    --host 0.0.0.0 \
    --tp 4 \
    --trust-remote-code
```

## Disaggregation Mode (Prefill-Decode Separation)

For production DeepSeek-V3/R1 deployments, SGLang supports prefill-decode disaggregation where prefill and decode phases run on separate nodes.

### Prerequisites

1. **Mooncake Transfer Engine** installed (see above)
2. **DeepEP** for MoE all-to-all communication
3. **DeepEP config files** for expert placement

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
    --dist-init-addr <MASTER_IP>:5757 \
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

### RDMA Memory Registration Warnings

When running with Mooncake, you may see RDMA memory registration warnings:
```
RdmaTransport: Failed to register memory: addr 0x... length 37896192
```

These warnings are **expected** if RDMA is not properly configured. Mooncake will fall back to TCP transport. For production deployments with RDMA, ensure:
- RDMA devices are properly configured
- Sufficient locked memory limits (`ulimit -l unlimited`)
- nvidia_peermem module is loaded (optional)

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

## Resources

- `scripts/diagnose.py` - Diagnostic script for installation issues
- `scripts/setup_env.sh` - Environment variable setup script
- `references/version_matrix.md` - Version compatibility matrix
- `references/troubleshooting.md` - Extended troubleshooting guide

## Version History

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
