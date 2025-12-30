# Flux.2 on TPU with Torchax

使用 Torchax 在 Google Cloud TPU 上运行 Flux.2 图像生成模型。

## 概述

Flux.2 是 Black Forest Labs 推出的先进文生图模型，支持高分辨率图像生成。本项目提供了在 TPU 上高效运行 Flux.2 的实现：

- **单文件版本**：`generate_torchax.py` - 一键运行
- **分阶段版本**：`generate_diffusers_torchax_staged/` - 灵活的三阶段流水线

### 技术特点

- **Embedded CFG**：Flux.2 将 guidance_scale 嵌入到 timestep 中，只需单次前向传播
- **1D Mesh**：仅使用张量并行 (tp)，无需数据并行
- **Splash Attention**：TPU 优化的注意力机制，使用 exp2 代替 exp
- **K-Smooth 技术**：减少 attention 数值溢出

## 性能数据

测试环境：TPU v4-8 (8 chips)，1024×1024 分辨率

### 单文件版本 (`generate_torchax.py`)

| 组件 | 设备 | 时间 |
|------|------|------|
| 编译 + 预热 | TPU | ~15s |
| Transformer (50步) | TPU | ~13.5s |
| VAE Decode | TPU | ~1.3s |
| **端到端总时间** | - | **~30s** |

### 分阶段版本

| 阶段 | 组件 | 设备 | 时间 |
|------|------|------|------|
| Stage 1 | Text Encoder (Mistral3) | CPU | ~30s |
| Stage 2 | Transformer (50步) | TPU | ~13.5s |
| Stage 3 | VAE Decoder | TPU | ~1.5s |
| **总计** | - | - | **~45s** |

> 注：分阶段版本适合调试和优化，单文件版本适合生产使用。

## 安装

### 1. 创建 TPU 实例

```bash
gcloud compute tpus tpu-vm create flux2-tpu \
    --zone=us-central2-b \
    --accelerator-type=v4-8 \
    --version=tpu-ubuntu2204-base
```

### 2. SSH 连接

```bash
gcloud compute tpus tpu-vm ssh flux2-tpu --zone=us-central2-b
```

### 3. 安装依赖

```bash
# 基础依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install torchax

# 安装 diffusers-tpu (包含 Flux.2 TPU 优化)
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
cd gpu-tpu-pedia/tpu/Flux.2
```

### 5. 模型权限

Flux.2 需要接受使用条款：

1. 访问 https://huggingface.co/black-forest-labs/FLUX.2-dev
2. 登录并接受条款
3. 配置 HuggingFace token：

```bash
huggingface-cli login
# 输入你的 token
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
| `--output` | `output.png` | 输出图像路径 |
| `--height` | 1024 | 图像高度 |
| `--width` | 1024 | 图像宽度 |
| `--num_inference_steps` | 50 | 推理步数 |
| `--guidance_scale` | 2.5 | CFG 引导强度 |
| `--seed` | 42 | 随机种子 |
| `--remote_encoder` | (无) | 使用远程 vLLM 服务器编码 |
| `--remote_url` | `http://10.128.0.45:8888` | 远程编码器 URL |
| `--warmup_steps` | 2 | 预热步数 |

#### 示例

```bash
# 基础用法
python generate_torchax.py --prompt "A cute cat sitting on a sofa"

# 自定义分辨率
python generate_torchax.py \
    --prompt "Mountain landscape" \
    --height 768 --width 1280

# 使用远程编码器（可选，需要 vLLM 服务器）
python generate_torchax.py \
    --prompt "Abstract art" \
    --remote_encoder \
    --remote_url "http://your-vllm-server:8888"
```

### 分阶段版本

适合需要调试、分析或自定义流程的场景。

#### 阶段 1：Text Encoder (CPU)

```bash
cd generate_diffusers_torchax_staged
python stage1_text_encoder.py --prompt "Your prompt here"
```

输出：`./stage_outputs/stage1_embeddings.safetensors`

#### 阶段 2：Transformer (TPU)

```bash
python stage2_transformer.py
```

输入：`stage1_embeddings.safetensors`
输出：`./stage_outputs/stage2_latents.safetensors`

#### 阶段 3：VAE Decoder (TPU)

```bash
python stage3_vae_decoder.py
```

输入：`stage2_latents.safetensors`
输出：`./stage_outputs/output_image.png`

#### 分阶段参数

**Stage 1 参数：**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--prompt` | (示例) | 生成描述 |
| `--output_dir` | `./stage_outputs` | 输出目录 |

**Stage 2 参数：**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input_dir` | `./stage_outputs` | 输入目录 |
| `--output_dir` | (同 input) | 输出目录 |
| `--num_inference_steps` | 50 | 推理步数 |
| `--guidance_scale` | 2.5 | CFG 强度 |
| `--warmup_steps` | 2 | 预热步数 |

**Stage 3 参数：**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input_dir` | `./stage_outputs` | 输入目录 |
| `--warmup` | (无) | 启用预热 |

## 文件结构

```
Flux.2/
├── README.md                    # 本文档
├── generate_torchax.py          # 单文件 TPU 版本 (推荐)
├── generate_gpu.py              # GPU 参考实现
├── splash_attention_utils.py    # TPU Splash Attention
│
├── generate_diffusers_torchax_staged/  # 分阶段 TPU 版本
│   ├── utils.py                 # 共享工具
│   ├── stage1_text_encoder.py   # CPU 文本编码
│   ├── stage2_transformer.py    # TPU 去噪
│   └── stage3_vae_decoder.py    # TPU VAE 解码
│
└── generate_diffusers_gpu_staged/      # GPU 分阶段版本
    ├── stage1_text_encoder.py
    ├── stage2_transformer.py
    └── stage3_vae_decoder.py
```

## 架构说明

### Text Encoder

Flux.2 使用 Pixtral 7B 变体 (Mistral3) 作为文本编码器。由于动态控制流，目前在 CPU 上运行。

### Transformer

24 层 MMDiT 架构：
- Attention heads: 24
- Hidden size: 3072
- 使用 Splash Attention 优化

### VAE

Flux.2 使用专门设计的 VAE，与 SDXL VAE 不兼容。

## 故障排除

### 内存不足

```bash
# 减少分辨率
python generate_torchax.py --height 512 --width 512

# 或使用分阶段版本，每阶段单独运行
```

### HuggingFace 访问问题

```bash
# 确保已登录
huggingface-cli login

# 检查 token 权限
huggingface-cli whoami
```

### JAX 编译缓存

编译结果会缓存在 `~/.cache/jax_cache`，首次运行较慢，后续运行会复用缓存。

```bash
# 清除缓存（如遇问题）
rm -rf ~/.cache/jax_cache
```

## 参考链接

- [Flux.2 HuggingFace](https://huggingface.co/black-forest-labs/FLUX.2-dev)
- [diffusers-tpu](https://github.com/yangwhale/diffusers-tpu)
- [Torchax](https://github.com/pytorch/xla)
- [JAX Splash Attention](https://github.com/jax-ml/jax-triton)

## License

本项目遵循 Apache 2.0 协议。Flux.2 模型权重遵循 Black Forest Labs 的使用条款。
