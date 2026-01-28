# vLLM Version Compatibility Matrix

## Current Recommended Versions (January 2026)

| Component | Version | Notes |
|-----------|---------|-------|
| vLLM | 0.14.1 | Latest stable on PyPI |
| PyTorch | 2.9.x | CUDA 12.9 builds |
| flashinfer-python | 0.5.3 | Must match cubin version |
| flashinfer-cubin | 0.5.3 | Must match python version |
| nvidia-nccl-cu12 | 2.28.3 | Required for multi-GPU |
| nvidia-cudnn-cu12 | 9.16.0.29 | Required for PyTorch 2.9+ |
| nvidia-cusparselt-cu12 | latest | Sparse operations |
| bitsandbytes | 0.46.1 | Quantization support |

## FlashInfer Version Notes

**Critical:** `flashinfer-python` and `flashinfer-cubin` versions MUST match exactly.

| FlashInfer | Compatibility |
|------------|---------------|
| 0.5.3 | ✓ Works with vLLM 0.14.1 and SGLang 0.5.6.post2 |
| 0.6.2 | ✗ vLLM 0.14.1 not compatible (installs 0.5.3) |

If both vLLM and SGLang are installed in the same environment, use FlashInfer 0.5.3.

## GPU Architecture Support

| GPU | Compute Capability | TORCH_CUDA_ARCH_LIST |
|-----|-------------------|---------------------|
| B200 | 10.0 | 10.0 |
| H100 | 9.0 | 9.0 |
| A100 | 8.0 | 8.0 |
| A10G | 8.6 | 8.6 |
| L4 | 8.9 | 8.9 |
| T4 | 7.5 | 7.5 |

## Model Tensor Parallelism Requirements

| Model | Attention Heads | Valid TP Values | Recommended TP |
|-------|-----------------|-----------------|----------------|
| Qwen2.5-7B | 28 | 1, 2, 4, 7, 14 | 4 |
| Qwen2.5-14B | 40 | 1, 2, 4, 5, 8, 10, 20, 40 | 4 or 8 |
| Qwen2.5-72B | 64 | 1, 2, 4, 8, 16, 32, 64 | 8 |
| Llama-3-8B | 32 | 1, 2, 4, 8, 16, 32 | 4 |
| Llama-3-70B | 64 | 1, 2, 4, 8, 16, 32, 64 | 8 |
| DeepSeek-V3 | 128 | 1, 2, 4, 8, 16, 32, 64, 128 | 8 (requires DeepEP) |
| DeepSeek-R1 | 128 | 1, 2, 4, 8, 16, 32, 64, 128 | 8 (requires DeepEP) |

## vLLM Version History

| Version | Release Date | Key Changes |
|---------|--------------|-------------|
| 0.14.1 | 2026-01 | Latest stable |
| 0.14.0 | 2026-01 | V1 engine improvements |
| 0.13.0 | 2025-12 | Async scheduling |
| 0.12.0 | 2025-11 | PagedAttention v2 |

## Dependency Conflicts with SGLang

When installing vLLM and SGLang in the same environment:

| Package | vLLM 0.14.1 | SGLang 0.5.6.post2 | Resolution |
|---------|-------------|-------------------|------------|
| flashinfer-python | 0.5.3 | 0.5.3 (declared) | Use 0.5.3 |
| grpcio | 1.76.0 | 1.75.1 | Warning only |
| xgrammar | 0.1.29 | 0.1.27 | Warning only |
| llguidance | 1.3.0 | <0.8.0 | Warning only |

These warnings are generally safe to ignore for inference workloads.
