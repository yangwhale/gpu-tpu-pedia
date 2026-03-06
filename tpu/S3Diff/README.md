# S3Diff on TPU v6e (torchax)

One-step diffusion-based 4x image super-resolution using [S3Diff](https://github.com/ArcticHare105/S3Diff), ported to TPU v6e via [torchax](https://github.com/pytorch/xla/tree/master/torchax).

## Architecture

S3Diff combines SD-Turbo (Stable Diffusion 2.1-Turbo, ~3.3B params) with degradation-guided LoRA modulation for single-step super-resolution:

```
Input LR Image (128x128)
       │
       ├──► DEResNet (CPU) ──► Degradation Score [blur, noise]
       │                              │
       │                    Fourier Embedding + MLP
       │                              │
       │                     de_mod matrices (per-layer)
       │                              │
       ├──► Bilinear 4x ──► VAE Encode ──► UNet (1 step, LoRA+de_mod) ──► VAE Decode
       │                                                                       │
       │                                                                 Output HR (512x512)
       │                                                                       │
       └──────────── Wavelet Color Fix ◄───────────────────────────────────────┘
```

Key design decisions for TPU:
- **No `torchax.compile()` on VAE encoder/UNet**: `de_mod` attributes change per input image, making them incompatible with JAX's static compilation. We use `torchax.enable_globally()` for traced execution instead.
- **VAE decoder IS compiled**: No LoRA/de_mod, so static compilation works and provides speedup.
- **Single-chip optimal**: SD-Turbo is small enough for 1 TPU chip. Tensor parallelism across 8 chips adds more communication overhead than it saves in compute.
- **bfloat16 throughout**: All model weights and activations use bfloat16 on TPU.

## Prerequisites

```bash
# On a TPU v6e VM (tested on v6e-8)
pip3 install peft==0.17.0 "transformers>=4.46.0,<5.0.0" basicsr==1.4.2

# torchax and jax should already be installed on TPU VMs
# Tested versions: torchax==0.0.11, jax==0.9.1
```

## Model Weights

Weights are auto-downloaded from HuggingFace on first run:
- **SD-Turbo**: `stabilityai/sd-turbo` (~3.3GB)
- **S3Diff LoRA + MLP**: `zhangap/S3Diff` → `s3diff.pkl` (~600MB)
- **DEResNet**: `assets/mm-realsr/de_net.pth` (~5MB, included in repo)

## Usage

### Single Image Super-Resolution

```bash
# Single TPU chip (recommended for latency)
TPU_VISIBLE_DEVICES=0 python generate_torchax.py \
    --input test_images/test_lr.png \
    --output output_sr.png \
    --warmup
```

### Benchmark with Multiple Iterations

```bash
TPU_VISIBLE_DEVICES=0 python generate_torchax.py \
    --input test_images/test_lr.png \
    --output output_sr.png \
    --warmup \
    --benchmark_iters 5
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | (required) | Input LR image path |
| `--output` | `{input_stem}_sr_4x.png` | Output SR image path |
| `--sd_path` | auto-download | SD-Turbo model directory |
| `--pretrained_path` | auto-download | S3Diff weights (.pkl) |
| `--de_net_path` | `assets/mm-realsr/de_net.pth` | DEResNet weights |
| `--pos_prompt` | "A high-resolution, 8K..." | Positive prompt |
| `--neg_prompt` | "oil painting, cartoon..." | Negative prompt |
| `--align_method` | `wavelet` | Color correction: `wavelet`, `adain`, `nofix` |
| `--warmup` | false | Run warmup pass for JIT compilation |
| `--benchmark_iters` | 1 | Number of benchmark iterations |

## Benchmark Results

**Hardware**: TPU v6e-8 (single chip)
**Input**: 128x128 → 512x512 (4x upscale), bfloat16

### Warmup (JIT Compilation)

| Stage | Time |
|-------|------|
| VAE Encode | 1.48s |
| UNet (1 step) | 8.32s |
| VAE Decode | 0.63s |
| **Total** | **10.52s** |

### Inference (Average over 5 iterations)

| Stage | Average | Min |
|-------|---------|-----|
| VAE Encode | 0.538s | 0.523s |
| UNet (1 step) | 5.019s | 4.588s |
| VAE Decode | 0.028s | 0.014s |
| **Total** | **5.595s** | **5.139s** |

### Multi-Chip Analysis

SD-Turbo (3.3B params) fits entirely on a single TPU v6e chip. Multi-chip tensor parallelism is **not recommended** for this model:
- Communication overhead between chips exceeds compute savings
- VAE encoder compilation alone takes 105s with 8-chip TP (vs 1.5s single-chip)
- For throughput scaling, use process-level data parallelism (one model per chip)

## File Structure

```
S3Diff/
├── README.md                    # This file
├── generate_torchax.py          # Main inference script (TPU)
├── splash_attention_utils.py    # SDPA reference implementation
├── assets/
│   └── mm-realsr/de_net.pth     # DEResNet weights
├── test_images/                 # Test images and outputs
└── src/                         # Original S3Diff source (reference)
```

## Technical Notes

### Why Not `torchax.compile()`?

S3Diff's LoRA layers have a `de_mod` attribute (degradation modulation matrix) that is dynamically computed per input image via:

```python
deg_score → Fourier Embedding → MLP → de_mod (rank x rank matrix)
```

This `de_mod` is set as a plain tensor attribute on each LoRA module and used in the forward pass via `torch.einsum()`. Since `torchax.compile()` expects static module state, it fails on these dynamic attributes.

**Solution**: Use `torchax.enable_globally()` which routes all PyTorch ops through JAX/XLA without static compilation. The VAE decoder (no LoRA) is still compiled for optimal performance.

### Conv2d Dtype Fix

The DDPM scheduler outputs float32 tensors, but model weights are bfloat16. The custom conv2d override auto-casts input dtype to match weight dtype:

```python
if jinput.dtype != jweight.dtype:
    jinput = jinput.astype(jweight.dtype)
```

## References

- [S3Diff Paper](https://arxiv.org/abs/2408.13252) - Degradation Guided One-step Image Super-resolution
- [S3Diff GitHub](https://github.com/ArcticHare105/S3Diff)
- [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo)
- [torchax](https://github.com/pytorch/xla/tree/master/torchax)
