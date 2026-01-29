---
name: vllm-installer
description: This skill should be used when users need to install, configure, debug, or run vLLM inference server on NVIDIA GPUs (especially B200/H100/A100). It covers installation from PyPI or source, dependency management, environment setup, common error diagnosis and fixes, tensor parallelism configuration, and server startup/testing. The skill automatically checks for LSSD mount status and DeepEP installation for MoE models.
license: MIT
---

# vLLM Installer

This skill provides comprehensive guidance for installing, configuring, and debugging vLLM on NVIDIA GPUs with CUDA 12.x.

## When to Use This Skill

- Installing vLLM on NVIDIA GPUs (B200/H100/A100)
- Debugging vLLM installation errors (missing libraries, version conflicts)
- Configuring tensor parallelism for different model architectures
- Setting up environment variables for CUDA and NVIDIA libraries
- Starting and testing vLLM OpenAI-compatible API server
- Fixing common runtime errors (cuDNN, cusparseLt, FlashInfer issues)

## Version Information (as of v0.14.1)

| Component | Version | Notes |
|-----------|---------|-------|
| vLLM | 0.14.1 | Latest stable (v0.15.0rc2 not yet on PyPI) |
| flashinfer-python | 0.5.3 | Attention backend |
| flashinfer-cubin | 0.5.3 | Must match flashinfer-python version |
| nixl | 0.9.0 | KV cache transfer (DMA-BUF, recommended for PD disaggregation) |
| nvidia-nccl-cu12 | 2.28.3 | Force reinstall |
| nvidia-cudnn-cu12 | 9.16.0.29 | Required for PyTorch 2.9+ |
| bitsandbytes | 0.46.1 | Quantization support |

## Pre-Installation Checks

### Step 0: Check Prerequisites

Before installing vLLM, the skill automatically checks:

1. **LSSD Mount Status** - High-speed local SSD for model caching
2. **DeepEP Installation** - Required for MoE models (DeepSeek-V3, DeepSeek-R1)

#### LSSD Check

```bash
# Check if /lssd is mounted
if mountpoint -q /lssd 2>/dev/null; then
    echo "✓ LSSD is mounted: $(df -h /lssd | tail -1 | awk '{print $2}')"
else
    echo "✗ LSSD is not mounted"
    echo "  Run: /lssd-mounter"
fi
```

If LSSD is not mounted, use the `lssd-mounter` skill:
```bash
/lssd-mounter
```

#### DeepEP Check (for MoE models)

```bash
# Check if DeepEP is installed
python3 -c "import deep_ep; print('✓ DeepEP installed')" 2>/dev/null || \
python3 -c "import deepep; print('✓ DeepEP installed')" 2>/dev/null || \
echo "✗ DeepEP not installed (required for MoE models)"
```

If DeepEP is not installed and you need to run MoE models, use the `deepep-installer` skill:
```bash
/deepep-installer
```

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

# Set HuggingFace cache to LSSD (if available)
if [ -d /lssd/huggingface ]; then
    export HF_HOME=/lssd/huggingface
fi
```

### Step 2: Install PyTorch

```bash
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu129
```

### Step 3: Install vLLM

```bash
# Install from PyPI (recommended)
pip install vllm==0.14.1 \
    --extra-index-url https://download.pytorch.org/whl/cu129
```

### Step 4: Install NVIDIA Libraries

These libraries must be installed with `--force-reinstall --no-deps` to avoid version conflicts:

```bash
pip install nvidia-nccl-cu12==2.28.3 --force-reinstall --no-deps
pip install nvidia-cudnn-cu12==9.16.0.29 --force-reinstall --no-deps
pip install nvidia-cusparselt-cu12 --force-reinstall --no-deps
```

### Step 5: Install FlashInfer

FlashInfer is the recommended attention backend for vLLM:

```bash
pip install flashinfer-python==0.5.3 flashinfer-cubin==0.5.3
```

**Important:** `flashinfer-python` and `flashinfer-cubin` versions MUST match exactly.

### Step 6: Configure LD_LIBRARY_PATH

To fix library loading issues, run `scripts/setup_env.sh` or manually set:

```bash
# Collect all nvidia pip package lib paths
NVIDIA_LIB_PATHS=""
for d in /usr/local/lib/python3.*/dist-packages/nvidia/*/lib; do
    [ -d "$d" ] && NVIDIA_LIB_PATHS="${d}:${NVIDIA_LIB_PATHS}"
done
for d in $HOME/.local/lib/python3.*/site-packages/nvidia/*/lib; do
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
pip install nvidia-cusparselt-cu12 --force-reinstall --no-deps
# Then set LD_LIBRARY_PATH as described above
```

### Error: FlashInfer version mismatch

**Symptom:**
```
ModuleNotFoundError: No module named 'flashinfer.jit.cubin_loader'
```
or
```
FLASHINFER_CUBIN_DIR not found
```

**Diagnosis:** `flashinfer-python` and `flashinfer-cubin` versions don't match.

**Fix:**
```bash
pip install flashinfer-python==0.5.3 flashinfer-cubin==0.5.3 --force-reinstall
```

### Error: WorkerProc failed to start

**Symptom:**
```
ERROR: WorkerProc failed to start.
File "vllm/v1/attention/selector.py" ...
```

**Diagnosis:** Usually caused by FlashInfer import failure.

**Fix:** Check FlashInfer versions match and LD_LIBRARY_PATH is set correctly.

### Error: assert self.total_num_heads % tp_size == 0

**Symptom:**
```
AssertionError: assert self.total_num_heads % tp_size == 0
```

**Diagnosis:** The model's attention head count is not divisible by the tensor parallelism size.

**Fix:** Choose a `--tensor-parallel-size` value that divides the model's attention head count:

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

## Starting the Server

To start the vLLM OpenAI-compatible API server:

```bash
# Load environment
source /vllm-workspace/vllm-env.sh

# Start server (adjust tp based on model architecture)
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 4 \
    --port 8000 \
    --host 0.0.0.0
```

Or using the Python module:
```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 4 \
    --port 8000 \
    --host 0.0.0.0
```

## Disaggregated Prefill (PD Separation)

vLLM supports prefill-decode disaggregation where prefill and decode phases run on separate instances. This allows independent tuning of TTFT (time-to-first-token) and ITL (inter-token-latency).

### KV Transfer Connectors

vLLM supports multiple KV transfer backends:

| Connector | Dependency | Use Case |
|-----------|------------|----------|
| **NixlConnector** | nixl | Recommended, uses DMA-BUF (no nvidia_peermem needed) |
| MooncakeConnector | mooncake | Requires nvidia_peermem kernel module |
| P2pNcclConnector | NCCL | Same-node P2P transfer |
| LMCacheConnector | lmcache | External KV cache |

### Installing NIXL for Disaggregation

```bash
pip install --break-system-packages nixl==0.9.0

# IMPORTANT: NIXL may downgrade NVIDIA libraries, reinstall correct versions:
pip install nvidia-nccl-cu12==2.28.3 --force-reinstall --no-deps
pip install nvidia-cudnn-cu12==9.16.0.29 --force-reinstall --no-deps
```

### Verifying NIXL

```bash
python3 -c "
from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import NixlConnector
print('NixlConnector OK')
"
```

### Disaggregated Prefill Configuration

**Prefill Node:**
```bash
vllm serve deepseek-ai/DeepSeek-V3 \
    --tensor-parallel-size 8 \
    --port 8100 \
    --download-dir /lssd/huggingface/hub \
    --kv-transfer-config '{
        "kv_connector": "NixlConnector",
        "kv_role": "kv_both",
        "kv_buffer_device": "cuda"
    }'
```

**Decode Node:**
```bash
vllm serve deepseek-ai/DeepSeek-V3 \
    --tensor-parallel-size 8 \
    --port 8200 \
    --download-dir /lssd/huggingface/hub \
    --kv-transfer-config '{
        "kv_connector": "NixlConnector",
        "kv_role": "kv_both",
        "kv_buffer_device": "cuda"
    }'
```

### KV Transfer Config Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| kv_connector | NixlConnector, MooncakeConnector, P2pNcclConnector | KV transfer backend |
| kv_role | kv_producer, kv_consumer, kv_both | Role in KV transfer (kv_both for most cases) |
| kv_buffer_device | cuda, cpu | Buffer device (cuda recommended, cpu for TPU) |
| kv_ip | IP address | Connector IP for distributed connection |
| kv_port | Port number | Connector port (default: 14579) |

### NIXL vs Mooncake

| Feature | NIXL | Mooncake |
|---------|------|----------|
| Memory registration | DMA-BUF (kernel native) | nvidia_peermem (kernel module) |
| Transport | UCX (TCP/RDMA/SHM) | RDMA or TCP |
| Kernel module required | No | nvidia_peermem (may fail on Open driver) |
| NVIDIA Open driver compatible | Yes | No |
| Recommended for | B200, new deployments | Legacy systems |

**Recommendation:** Use NixlConnector for new deployments, especially on systems with NVIDIA Open Kernel Module.

## Testing the Server

To verify the server is working:

```bash
# List models
curl http://localhost:8000/v1/models

# Chat completion test
curl http://localhost:8000/v1/chat/completions \
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
- vLLM version and import status
- FlashInfer versions (python and cubin must match)
- Required library availability
- GPU detection and memory
- LD_LIBRARY_PATH configuration
- **LSSD mount status** (prompts to run /lssd-mounter if not mounted)
- **DeepEP installation** (prompts to run /deepep-installer if not installed, for MoE models)

## Dependency Skills

### LSSD Mounter

If LSSD is not mounted, the diagnostic script will prompt:

```
⚠ LSSD: Not mounted
  High-speed local SSD recommended for model caching
  To mount LSSD, use the lssd-mounter skill:
    /lssd-mounter
```

### DeepEP Installer

If DeepEP is not installed and you plan to run MoE models:

```
⚠ DeepEP: Not installed
  DeepEP is required for MoE models (DeepSeek-V3, DeepSeek-R1)
  To install DeepEP, use the deepep-installer skill:
    /deepep-installer
```

### Workflow Integration

When diagnosing vLLM installation:

1. Run `diagnose.py` to check all prerequisites
2. If LSSD not mounted → prompt to run `/lssd-mounter`
3. If DeepEP not installed and MoE models needed → prompt to run `/deepep-installer`
4. After dependencies installed, re-run diagnostic
5. Proceed with vLLM server startup

### Pre-downloading DeepSeek Weights

For faster DeepSeek-V3/R1 model loading, pre-download weights from GCS instead of HuggingFace:

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
- Use `--download-dir /lssd/huggingface/hub` when starting vLLM server

## vLLM vs SGLang

| Feature | vLLM | SGLang |
|---------|------|--------|
| Attention | PagedAttention | RadixAttention |
| Strength | High throughput batch | Multi-turn, structured output |
| API | OpenAI compatible | OpenAI compatible |
| Default Port | 8000 | 30000 |

### Coexistence and Dependency Conflicts

Both can coexist on the same system but have dependency version conflicts:

| Package | SGLang wants | vLLM installs |
|---------|-------------|---------------|
| grpcio | 1.75.1 | 1.76.0 |
| timm | 1.0.16 | 1.0.24 |
| xgrammar | 0.1.27 | 0.1.29 |
| llguidance | <0.8.0,>=0.7.11 | 1.3.0 |

**Recommendations:**
1. **Production**: Use separate Python virtual environments for each
2. **Development**: Accept mismatches (usually works for basic inference)
3. **MoE models**: Install DeepEP first, then either framework

### DeepEP Recompilation After vLLM Install

**IMPORTANT**: If DeepEP was installed before vLLM, you may need to recompile DeepEP after installing vLLM due to PyTorch ABI changes.

**Symptom:**
```
ImportError: .../deep_ep_cpp.cpython-312-x86_64-linux-gnu.so: undefined symbol: _ZNK3c1010TensorImpl15incref_pyobjectEv
```

**Fix:**
```bash
cd /tmp/deepep_build  # or wherever DeepEP was cloned
rm -rf build/ dist/ *.egg-info
rm -rf ~/.local/lib/python3.12/site-packages/deep_ep-*.egg  # Remove old egg

export CUDA_HOME=/usr/local/cuda-12.9
export LD_LIBRARY_PATH=/opt/deepep/nvshmem/lib:/opt/deepep/gdrcopy/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$CUDA_HOME/lib64/stubs:$LIBRARY_PATH

TORCH_CUDA_ARCH_LIST="10.0" NVSHMEM_DIR=/opt/deepep/nvshmem python3 setup.py install --user
```

## Resources

- `scripts/diagnose.py` - Diagnostic script for installation issues
- `scripts/setup_env.sh` - Environment variable setup script
- `references/version_matrix.md` - Version compatibility matrix
- `references/troubleshooting.md` - Extended troubleshooting guide

## Version History

- **2026-01-29**: Added GCS DeepSeek weights pre-download
  - **NEW**: Added "Pre-downloading DeepSeek Weights" section in Dependency Skills
  - GCS source: `gs://chrisya-gpu-pg-ase1/huggingface`
  - Faster than HuggingFace download (same-region bandwidth)

- **2026-01-29**: Added Disaggregated Prefill (PD Separation) with NIXL support
  - **NEW**: Added NixlConnector as recommended KV transfer backend
  - **NEW**: Added disaggregated prefill configuration examples
  - **NEW**: Added KV transfer config parameters documentation
  - **NEW**: Added NIXL vs Mooncake comparison table
  - Added nixl 0.9.0 to version table

- **2026-01-29**: Updated based on full installation workflow with DeepEP + SGLang + vLLM
  - **CRITICAL**: Added DeepEP recompilation requirement after vLLM install
  - Documented PyTorch ABI incompatibility issue and fix
  - Added detailed dependency conflict table (numpy, llguidance versions)

- **2026-01-28**: Updated based on installation experience
  - Added pip installation for Ubuntu 24.04
  - Added FlashInfer 0.5.3 unified version with SGLang
  - Documented SGLang/vLLM dependency conflicts with specific versions
