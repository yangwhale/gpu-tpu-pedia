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
- **Single-chip only**: SD-Turbo (3.3B) fits on 1 TPU chip. Multi-chip TP tested and confirmed slower (see below).
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

| Stage | Warmup | Average (5 iter) | Min |
|-------|--------|------------------|-----|
| VAE Encode | 1.53s | 0.534s | 0.514s |
| UNet (1 step) | 8.20s | 4.716s | 4.479s |
| VAE Decode | 0.66s | 0.017s | 0.013s |
| **Total** | **10.47s** | **5.278s** | **5.039s** |

### HBM Usage (128×128 → 512×512, 1 chip)

| Stage | Current HBM | Peak HBM |
|-------|-------------|----------|
| After loading weights to XLA | 0.00 GB | 0.00 GB |
| After VAE Encode | 0.07 GB | 0.53 GB |
| After UNet (1 step) | 0.18 GB | 0.90 GB |
| After VAE Decode | 0.36 GB | 0.90 GB |
| **Steady-state (cached)** | **0.27 GB** | **1.00 GB** |

**Note**: Measured peak (1.0 GB) << theoretical weight total (2.33 GB) because `torchax.enable_globally()` uses JAX traced execution — weights are streamed to TPU HBM per-op and released after use. HBM only holds the currently executing op's weights and activations, not the full model.

**Model parameter breakdown (bf16)**:
- UNet: 865.9M params (1,652 MB)
- VAE Encoder: 34.2M params (65 MB)
- VAE Decoder: 49.5M params (94 MB)
- LoRA + MLP: ~300M params (~572 MB)
- Text Encoder: 340.4M (CPU only, not on HBM)
- DEResNet: ~2.5M (CPU only, not on HBM)
- **Total on XLA: ~1.25B params (2.33 GB bf16)**

Peak HBM utilization: **3.2%** of 31.2 GB available per chip.

### Why Not Multi-Chip?

8-chip tensor parallelism was tested and provides **no speedup** — inference is slightly slower (5.46s avg vs 5.28s single-chip) while warmup is 15x longer (155s vs 10.5s). SD-Turbo (3.3B params) is too small for the inter-chip communication cost to be worthwhile.

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
