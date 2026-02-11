# SDXL on TPU with Torchax

使用 Torchax 在 Google Cloud TPU 上运行 Stable Diffusion XL 图像生成模型。

## 概述

SDXL 是 Stability AI 推出的高质量文生图模型，具有 3.5B 参数的 UNet 和双 CLIP Text Encoder。本项目提供了在 TPU 上高效运行 SDXL 的实现：

- **单文件版本**：`generate_torchax.py` - 一键运行
- **分阶段版本**：`generate_diffusers_torchax_staged/` - 灵活的三阶段流水线

### 技术特点

- **双 Text Encoder**：使用 CLIP-ViT-L/14 + OpenCLIP-ViT-bigG-14
- **传统 CFG**：每步需要 2 次 UNet 前向传播
- **1D Mesh**：仅使用张量并行 (tp)，无需数据并行
- **UNet 分片**：Attention 和 FFN 层按 Megatron-style 张量并行分片

## 性能数据

测试环境：TPU v6e-8 (8 chips)，1024×1024 分辨率

### 单文件版本 (`generate_torchax.py`)

| 组件 | 设备 | 时间 |
|------|------|------|
| 编译 + 预热 (2步) | TPU | ~30s |
| UNet (25步) | TPU | ~2.2s (~0.09s/step) |
| VAE Decode | TPU | ~0.5s |
| **端到端总时间** | - | **~33s** (首次) / **~3s** (后续) |

### 分阶段版本

| 阶段 | 组件 | 设备 | 时间 |
|------|------|------|------|
| Stage 1 | Text Encoder (CLIP x2) | CPU | ~5s |
| Stage 2 | UNet (25步) | TPU | ~2.2s |
| Stage 3 | VAE Decoder | TPU | ~0.5s |

## 安装

### 1. 创建 TPU 实例

```bash
gcloud compute tpus tpu-vm create sdxl-tpu \
    --zone=us-east5-b \
    --accelerator-type=v6e-8 \
    --version=tpu-vm-v6e-base
```

### 2. SSH 连接

```bash
gcloud compute tpus tpu-vm ssh sdxl-tpu --zone=us-east5-b
```

### 3. 安装依赖

```bash
# 基础依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install torchax

# 安装 diffusers-tpu
git clone https://github.com/yangwhale/diffusers-tpu.git
cd diffusers-tpu
pip install -e .
cd ..

# 其他依赖
pip install transformers accelerate safetensors pillow tqdm
```

### 4. 获取代码

```bash
git clone https://github.com/yangwhale/gpu-tpu-pedia.git
cd gpu-tpu-pedia/tpu/SDXL
```

## 使用方法

### 单文件版本（推荐）

最简单的使用方式，一键生成图像：

```bash
python generate_torchax.py \
    --prompt "A beautiful sunset over the ocean" \
    --output output.png
```

#### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--prompt` | (示例) | 生成图像的描述 |
| `--negative_prompt` | "blurry, low quality..." | 负面提示词 |
| `--output` | `{timestamp}.png` | 输出图像路径 |
| `--height` | 1024 | 图像高度 |
| `--width` | 1024 | 图像宽度 |
| `--num_inference_steps` | 25 | 推理步数 |
| `--guidance_scale` | 7.5 | CFG 引导强度 |
| `--seed` | 42 | 随机种子 |
| `--warmup_steps` | 2 | 预热步数 |

#### 示例

```bash
# 基础用法
python generate_torchax.py --prompt "A cute cat sitting on a sofa"

# 自定义分辨率
python generate_torchax.py \
    --prompt "Mountain landscape" \
    --height 768 --width 1280

# 更多推理步数（更高质量）
python generate_torchax.py \
    --prompt "Professional portrait" \
    --num_inference_steps 50
```

### 分阶段版本

适合需要调试、分析或自定义流程的场景。

#### 阶段 1：Text Encoder (CPU)

```bash
cd generate_diffusers_torchax_staged
python stage1_text_encoder.py --prompt "Your prompt here"
```

输出：`./stage_outputs/stage1_embeddings.safetensors`

#### 阶段 2：UNet Denoising (TPU)

```bash
python stage2_unet.py
```

输入：`stage1_embeddings.safetensors`
输出：`./stage_outputs/stage2_latents.safetensors`

#### 阶段 3：VAE Decoder (TPU)

```bash
python stage3_vae_decoder.py
```

输入：`stage2_latents.safetensors`
输出：`./stage_outputs/output_image.png`

## 文件结构

```
SDXL/
├── README.md                    # 本文档
├── generate_torchax.py          # 单文件 TPU 版本 (推荐)
├── splash_attention_utils.py    # TPU Splash Attention
│
└── generate_diffusers_torchax_staged/  # 分阶段 TPU 版本
    ├── utils.py                 # 共享工具
    ├── stage1_text_encoder.py   # CPU 文本编码
    ├── stage2_unet.py           # TPU 去噪
    └── stage3_vae_decoder.py    # TPU VAE 解码
```

## 架构说明

### Text Encoder

SDXL 使用两个 CLIP Text Encoder：
- **text_encoder**: CLIPTextModel (`openai/clip-vit-large-patch14`)
- **text_encoder_2**: CLIPTextModelWithProjection (`laion/CLIP-ViT-bigG-14`)

输出：
- `prompt_embeds`: (batch, 77, 2048) - 两个 encoder 隐藏状态拼接
- `pooled_prompt_embeds`: (batch, 1280) - text_encoder_2 的 pooled output

### UNet

SDXL UNet 具有 3.5B 参数：
- 4 个 down blocks, 1 个 mid block, 4 个 up blocks
- 每个 block 包含 ResNet + Transformer attention layers
- 使用 cross-attention 将 text embeddings 注入到 UNet
- `add_time_ids` 包含原始尺寸、裁剪坐标和目标尺寸信息

### VAE

使用标准 AutoencoderKL，scaling factor 为 0.13025。

## 与 Flux.2 的区别

| 特性 | SDXL | Flux.2 |
|------|------|--------|
| 架构 | UNet (3.5B) | MMDiT (12B) |
| Text Encoder | 2x CLIP | Pixtral 7B (Mistral3) |
| CFG | 传统 (2x forward) | Embedded (1x forward) |
| VAE | 标准 AutoencoderKL | 专用 Flux VAE |
| 序列长度 | 77 tokens | 512 tokens |

## 故障排除

### 内存不足

```bash
# 减少分辨率
python generate_torchax.py --height 512 --width 512

# 或使用分阶段版本，每阶段单独运行
```

### JAX 编译缓存

编译结果会缓存在 `~/.cache/jax_cache`，首次运行较慢，后续运行会复用缓存。

```bash
# 清除缓存（如遇问题）
rm -rf ~/.cache/jax_cache
```

## 参考链接

- [SDXL HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [SDXL Paper](https://arxiv.org/abs/2307.01952)
- [diffusers-tpu](https://github.com/yangwhale/diffusers-tpu)
- [Torchax](https://github.com/pytorch/xla)

## License

本项目遵循 Apache 2.0 协议。SDXL 模型权重遵循 Stability AI 的使用条款。
