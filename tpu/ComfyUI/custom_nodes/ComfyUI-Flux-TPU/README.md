# ComfyUI-Flux-TPU

在 Google Cloud TPU 上运行 Flux.2 图像生成的 ComfyUI 自定义节点。

## 功能特性

- 🚀 **TPU 加速**: 使用 torchax 在 TPU 上运行 Flux.2 Transformer 和 VAE
- 🔧 **模块化设计**: 分离的 Text Encoder、Sampler 和 VAE Decoder 节点
- ⚡ **Splash Attention**: 针对长序列的 TPU 优化 attention 实现（使用 exp2 优化）
- 🔄 **自动分片**: 自动将模型权重分布到 8 个 TPU 核心
- 🎨 **ComfyUI 集成**: 完整的可视化工作流支持

## 节点说明

| 节点 | 运行位置 | 功能 |
|------|----------|------|
| **Flux.2 Text Encoder (CPU)** | CPU | 使用 Mistral3 编码文本 prompt |
| **Flux.2 TPU Sampler** | TPU | 运行 Transformer 去噪，生成 latents |
| **Flux.2 TPU VAE Decoder** | TPU | 解码 latents 为最终图像 |
| **Flux.2 TPU Full Pipeline** | TPU | 端到端图像生成（组合以上三个） |

## 性能数据

测试环境：TPU v6e-8 (8 chips)

| 分辨率 | Steps | Transformer | VAE | 总时间 |
|--------|-------|-------------|-----|--------|
| 512x512 | 50 | ~20s | ~2s | ~30s |
| 1024x1024 | 50 | ~45s | ~3s | ~60s |

> 注：首次运行需要编译（约 15-30s），后续运行会使用 JAX 编译缓存。

## 安装

### 1. 创建 TPU 实例

```bash
# 创建 TPU v6e-8 实例
gcloud compute tpus tpu-vm create comfyui-tpu \
    --zone=us-central1-a \
    --accelerator-type=v6e-8 \
    --version=tpu-ubuntu2204-base
```

### 2. SSH 连接到 TPU

```bash
gcloud compute tpus tpu-vm ssh comfyui-tpu --zone=us-central1-a
```

### 3. 安装基础依赖

```bash
# PyTorch (CPU 版本，ComfyUI 框架用)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# JAX for TPU
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Torchax (PyTorch-to-JAX bridge)
pip install torchax

# 其他依赖
pip install transformers accelerate safetensors pillow tqdm
```

### 4. 安装 diffusers-tpu

```bash
# diffusers-tpu 包含 Flux.2 TPU 优化模型
git clone https://github.com/yangwhale/diffusers-tpu.git
cd diffusers-tpu
pip install -e .
cd ..
```

### 5. 克隆 ComfyUI-TPU

```bash
git clone https://github.com/yangwhale/ComfyUI-TPU.git
cd ComfyUI-TPU
```

### 6. 配置 HuggingFace 访问

Flux.2 需要接受模型使用条款：

1. 访问 https://huggingface.co/black-forest-labs/FLUX.2-dev
2. 登录 HuggingFace 并接受条款
3. 配置 token：

```bash
huggingface-cli login
# 输入你的 HuggingFace token
```

## 启动 ComfyUI

**重要**: 必须使用 `--cpu` 参数启动 ComfyUI：

```bash
cd ComfyUI-TPU
python main.py --cpu
```

> 为什么使用 `--cpu`？ComfyUI 的框架运行在 CPU 上，而我们的自定义节点会将 Flux.2 的 Transformer 和 VAE 部分调度到 TPU 上运行。

启动后访问: http://127.0.0.1:8188

## 使用方法

### 方法 1: 加载示例 Workflow（推荐）

1. 启动 ComfyUI: `python main.py --cpu`
2. 访问 http://127.0.0.1:8188
3. 点击界面左侧的 **Load** 按钮
4. 选择 `custom_nodes/ComfyUI-Flux-TPU/examples/flux2_tpu_basic.json`
5. 修改 prompt，点击 **Run** 生成图像

### 方法 2: 手动创建 Workflow

1. 右键画布 → Add Node → TPU/Flux.2 → **Flux.2 Text Encoder (CPU)**
   - 输入 prompt
   - 输出连接到 Sampler

2. 添加 **Flux.2 TPU Sampler** 节点
   - 设置 height/width (如 1024x1024)
   - 设置 steps (推荐 50)
   - 设置 guidance_scale (推荐 4.0)
   - 输出连接到 VAE Decoder

3. 添加 **Flux.2 TPU VAE Decoder** 节点
   - 确保 height/width 与 Sampler 一致
   - 输出连接到 Preview Image

4. 添加 **Preview Image** 节点查看结果

## Workflow 示意图

```
┌─────────────────────┐     ┌──────────────────┐     ┌─────────────────────┐     ┌───────────────┐
│ Flux.2 Text Encoder │────▶│ Flux.2 TPU      │────▶│ Flux.2 TPU VAE      │────▶│ Preview Image │
│ (CPU)               │     │ Sampler         │     │ Decoder             │     │               │
│                     │     │                 │     │                     │     │               │
│ prompt: "..."       │     │ height: 1024    │     │ height: 1024        │     │               │
│ model_id: ...       │     │ width: 1024     │     │ width: 1024         │     │               │
│                     │     │ steps: 50       │     │ model_id: ...       │     │               │
│                     │     │ guidance: 4.0   │     │                     │     │               │
│                     │     │ seed: ...       │     │                     │     │               │
└─────────────────────┘     └──────────────────┘     └─────────────────────┘     └───────────────┘
       prompt_embeds ────────────▶ LATENT ─────────────────▶ IMAGE ─────────────────▶
```

## 参数说明

### Text Encoder
| 参数 | 默认值 | 说明 |
|------|--------|------|
| prompt | - | 图像描述文本 |
| model_id | `black-forest-labs/FLUX.2-dev` | HuggingFace 模型 ID |

### TPU Sampler
| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| height | 1024 | 256-2048 | 输出图像高度 |
| width | 1024 | 256-2048 | 输出图像宽度 |
| num_inference_steps | 50 | 1-100 | 去噪步数 |
| guidance_scale | 4.0 | 0-20 | Embedded CFG 引导强度 |
| seed | 42 | - | 随机种子 |

### VAE Decoder
| 参数 | 默认值 | 说明 |
|------|--------|------|
| height | 1024 | 必须与 Sampler 一致 |
| width | 1024 | 必须与 Sampler 一致 |
| model_id | `black-forest-labs/FLUX.2-dev` | HuggingFace 模型 ID |

## 示例 Workflow

示例 workflow 文件位于 `examples/` 目录：

- [`flux2_tpu_basic.json`](examples/flux2_tpu_basic.json) - 基础三节点 workflow

## 架构说明

### Text Encoder (Mistral3)

Flux.2 使用 Pixtral 7B 变体 (Mistral3) 作为文本编码器。由于包含动态控制流，目前在 CPU 上运行。

### Transformer (TPU)

24 层 MMDiT 架构：
- Attention heads: 24
- Hidden size: 3072
- 使用 Splash Attention 优化（exp2 替代 exp）
- 权重自动分片到 8 个 TPU 核心

### VAE Decoder (TPU)

Flux.2 专用 VAE，与 SDXL VAE 不兼容。在 TPU 上运行以加速解码。

## 故障排除

### "torchax Tensors can only do math within the torchax environment"

这个错误已在最新版本中修复。确保使用最新代码：

```bash
cd ComfyUI-TPU
git pull
```

### 模型加载失败 / 401 Unauthorized

确保已登录 HuggingFace 并接受 Flux.2 使用条款：

```bash
huggingface-cli login
huggingface-cli whoami  # 检查登录状态
```

### JAX 编译缓存

编译结果缓存在 `~/.cache/jax_cache`，首次运行较慢。如遇编译问题：

```bash
# 清除缓存重新编译
rm -rf ~/.cache/jax_cache
```

### 内存不足

减少图像分辨率：

```bash
# 使用较小分辨率
height: 512, width: 512
```

## 相关项目

- [diffusers-tpu](https://github.com/yangwhale/diffusers-tpu) - Flux.2 TPU 优化模型
- [gpu-tpu-pedia/Flux.2](https://github.com/yangwhale/gpu-tpu-pedia/tree/main/tpu/Flux.2) - 命令行版本
- [Torchax](https://github.com/pytorch/xla) - PyTorch-to-JAX bridge
- [Flux.2 HuggingFace](https://huggingface.co/black-forest-labs/FLUX.2-dev) - 官方模型

## License

MIT License

Flux.2 模型权重遵循 Black Forest Labs 的使用条款。
