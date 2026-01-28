# SGLang Version Compatibility Matrix

This document tracks version compatibility between SGLang, sgl-kernel, PyTorch, CUDA, and other dependencies.

## Recommended Versions (as of January 2026)

| Component | Version | Install Command |
|-----------|---------|-----------------|
| SGLang | 0.5.6.post2 | `pip install -e "python[blackwell]"` |
| sgl-kernel | 0.3.21 | `pip install sgl-kernel==0.3.21` |
| PyTorch | 2.9.1+cu129 | (installed with SGLang) |
| CUDA | 12.9 | System install |
| nvidia-nccl-cu12 | 2.28.3 | `pip install nvidia-nccl-cu12==2.28.3 --force-reinstall --no-deps` |
| nvidia-cudnn-cu12 | 9.16.0.29 | `pip install nvidia-cudnn-cu12==9.16.0.29 --force-reinstall --no-deps` |
| mooncake-transfer-engine | 0.3.8.post1 | `pip install mooncake-transfer-engine==0.3.8.post1` |
| flashinfer | 0.5.3 | (installed with SGLang) |

## CUDA Version Mapping

| CUDA Version | PyTorch Index | sgl-kernel Install |
|--------------|---------------|-------------------|
| 12.6.x | cu126 | whl from GitHub |
| 12.8.x | cu128 | `pip install sgl-kernel==VERSION` |
| 12.9.x | cu129 | `pip install sgl-kernel==VERSION` |
| 13.0.x | cu130 | whl from GitHub |

## Build Type Options

| Build Type | Target Hardware | Environment Variable |
|------------|-----------------|---------------------|
| `all` | General (all supported) | `BUILD_TYPE=all` |
| `blackwell` | NVIDIA B200/B100 | `BUILD_TYPE=blackwell` |
| `hopper` | NVIDIA H100/H200 | `BUILD_TYPE=hopper` |
| `ampere` | NVIDIA A100/A10 | `BUILD_TYPE=ampere` |

## GPU Architecture Support

| GPU | Architecture | Compute Capability | Recommended TP |
|-----|--------------|-------------------|----------------|
| B200 | Blackwell | 10.0 | 1-8 |
| B100 | Blackwell | 10.0 | 1-8 |
| H100 | Hopper | 9.0 | 1-8 |
| H200 | Hopper | 9.0 | 1-8 |
| A100 | Ampere | 8.0 | 1-8 |
| A10 | Ampere | 8.6 | 1-4 |

## Model Tensor Parallelism Compatibility

To use tensor parallelism (TP), the model's attention head count must be divisible by TP size.

| Model | Attention Heads | Valid TP Values |
|-------|-----------------|-----------------|
| Qwen2.5-0.5B | 14 | 1, 2, 7, 14 |
| Qwen2.5-1.5B | 12 | 1, 2, 3, 4, 6, 12 |
| Qwen2.5-3B | 16 | 1, 2, 4, 8, 16 |
| Qwen2.5-7B | 28 | 1, 2, 4, 7, 14, 28 |
| Qwen2.5-14B | 40 | 1, 2, 4, 5, 8, 10, 20, 40 |
| Qwen2.5-32B | 40 | 1, 2, 4, 5, 8, 10, 20, 40 |
| Qwen2.5-72B | 64 | 1, 2, 4, 8, 16, 32, 64 |
| Llama-3-8B | 32 | 1, 2, 4, 8, 16, 32 |
| Llama-3-70B | 64 | 1, 2, 4, 8, 16, 32, 64 |
| Llama-3.1-8B | 32 | 1, 2, 4, 8, 16, 32 |
| Llama-3.1-70B | 64 | 1, 2, 4, 8, 16, 32, 64 |
| Llama-3.1-405B | 128 | 1, 2, 4, 8, 16, 32, 64, 128 |
| DeepSeek-V3 | 128 | 1, 2, 4, 8, 16, 32, 64, 128 |
| DeepSeek-R1 | 128 | 1, 2, 4, 8, 16, 32, 64, 128 |
| Mistral-7B | 32 | 1, 2, 4, 8, 16, 32 |
| Mixtral-8x7B | 32 | 1, 2, 4, 8, 16, 32 |

## Version History

### v0.5.6.post2 (Latest)
- sgl-kernel 0.3.21
- PyTorch 2.9.1
- Flashinfer 0.5.3
- Improved Blackwell support

### v0.5.0rc2 (Previous)
- sgl-kernel 0.3.5
- PyTorch 2.10.0
- Flashinfer 0.3.x

## Dependency Update Commands

When updating from older versions:

```bash
# Update sgl-kernel first
pip install sgl-kernel==0.3.21

# Update SGLang
cd /sgl-workspace/sglang
git fetch --tags
git checkout v0.5.6.post2
pip install -e "python[blackwell]" --extra-index-url https://download.pytorch.org/whl/cu129

# Update NVIDIA libraries
pip install nvidia-nccl-cu12==2.28.3 --force-reinstall --no-deps
pip install nvidia-cudnn-cu12==9.16.0.29 --force-reinstall --no-deps

# Download flashinfer cubin cache
python3 -m flashinfer --download-cubin
```
