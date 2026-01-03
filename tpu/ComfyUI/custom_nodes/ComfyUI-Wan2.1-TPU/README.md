# ComfyUI-Wan-TPU

在 Google Cloud TPU 上运行 Wan 2.1 Text-to-Video 生成的 ComfyUI 自定义节点。

## 功能特性

- 🚀 **TPU 加速**: 使用 torchax 在 TPU 上运行 Wan 2.1 全部组件
- 🎬 **视频生成**: 支持 720P (1280x720) 和 480P (848x480) 分辨率
- 🔧 **模块化设计**: 分离的 Text Encoder、Sampler 和 VAE Decoder 节点
- ⚡ **Splash Attention**: 针对长序列的 TPU 优化 attention 实现（exp2 优化）
- 🔄 **2D Mesh 分片**: 自动将模型权重分布到 8 个 TPU 核心 (dp=2, tp=4)

## 节点说明

| 节点 | 运行位置 | 功能 |
|------|----------|------|
| **Wan 2.1 Text Encoder (TPU)** | TPU | 使用 T5-XXL 编码 prompt |
| **Wan 2.1 TPU Sampler** | TPU | 运行 Transformer 去噪，生成 latents |
| **Wan 2.1 TPU VAE Decoder** | TPU | 解码 latents 为视频帧 |
| **Wan 2.1 TPU Full Pipeline** | TPU | 端到端视频生成（组合以上三个） |

## 性能数据

测试环境：TPU v6e-8 (8 chips)

### 480P (848×480, 81 帧, 50 步)

| 阶段 | 预热 (JIT) | 正式运行 | 每步时间 |
|------|-----------|---------|---------|
| Text Encoder | - | ~3s | - |
| Transformer | ~93s | ~68s | ~1.37s |
| VAE Decoder | ~4s | ~0.5s | - |
| **总计** | ~97s | ~72s | - |

### 720P (1280×720, 81 帧, 50 步)

| 阶段 | 预热 (JIT) | 正式运行 | 每步时间 |
|------|-----------|---------|---------|
| Text Encoder | - | ~3s | - |
| Transformer | ~110s | ~230s | ~4.60s |
| VAE Decoder | ~80s | ~1s | - |
| **总计** | ~190s | ~234s | - |

> 注：首次运行需要 JIT 编译，后续运行使用 JAX 编译缓存。

## 安装

### 1. 创建 TPU 实例

```bash
# 创建 TPU v6e-8 实例
gcloud compute tpus tpu-vm create wan21-tpu \
    --zone=us-central1-a \
    --accelerator-type=v6e-8 \
    --version=tpu-ubuntu2204-base
```

### 2. SSH 连接到 TPU

```bash
gcloud compute tpus tpu-vm ssh wan21-tpu --zone=us-central1-a
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
pip install opencv-python imageio imageio-ffmpeg
```

### 4. 安装 diffusers-tpu

```bash
# diffusers-tpu 包含 Wan 2.1 TPU 优化模型
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

### 6. 安装 VideoHelperSuite（可选，用于视频输出）

```bash
cd custom_nodes
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
cd ..
pip install imageio_ffmpeg
```

> VideoHelperSuite 提供 VHS Video Combine 节点，用于将生成的帧合成为 MP4 视频。

## 启动 ComfyUI

**重要**: 必须使用 `--cpu` 参数启动 ComfyUI：

```bash
cd ComfyUI-TPU
python main.py --cpu
```

> 为什么使用 `--cpu`？ComfyUI 的框架运行在 CPU 上，而我们的自定义节点会将 Wan 2.1 的组件调度到 TPU 上运行。

启动后访问: http://127.0.0.1:8188

## 使用方法

### 方法 1: 加载示例 Workflow（推荐）

1. 启动 ComfyUI: `python main.py --cpu`
2. 访问 http://127.0.0.1:8188
3. 点击界面左侧的 **Load** 按钮
4. 选择 `custom_nodes/ComfyUI-Wan-TPU/examples/wan21_tpu_basic.json`
5. 修改 prompt，点击 **Run** 生成视频

### 方法 2: 手动创建 Workflow

1. 右键画布 → Add Node → TPU/Wan2.1 → **Wan 2.1 Text Encoder (TPU)**
   - 输入 prompt 和 negative_prompt
   - 输出连接到 Sampler

2. 添加 **Wan 2.1 TPU Sampler** 节点
   - 设置 height/width (720P: 1280x720, 480P: 848x480)
   - 设置 num_frames (推荐 81，约 5 秒视频)
   - 设置 steps (推荐 50)
   - 输出连接到 VAE Decoder

3. 添加 **Wan 2.1 TPU VAE Decoder** 节点
   - 输出连接到 VHS Video Combine 或 Save Image Sequence

4. 添加视频保存节点查看结果

## Workflow 示意图

```
┌───────────────────────┐     ┌──────────────────┐     ┌───────────────────────┐
│ Wan 2.1 Text Encoder  │────▶│ Wan 2.1 TPU     │────▶│ Wan 2.1 TPU VAE       │
│ (TPU)                 │     │ Sampler         │     │ Decoder               │
│                       │     │                 │     │                       │
│ prompt: "..."         │     │ height: 720     │     │                       │
│ negative_prompt: "..."│     │ width: 1280     │     │                       │
│                       │     │ num_frames: 81  │     │                       │
│                       │     │ steps: 50       │     │                       │
│                       │     │ guidance: 5.0   │     │                       │
└───────────────────────┘     └──────────────────┘     └───────────────────────┘
  prompt_embeds ──────────────────▶ latents ─────────────────▶ frames
  negative_prompt_embeds ─────────┘
```

## 参数说明

### Text Encoder
| 参数 | 默认值 | 说明 |
|------|--------|------|
| prompt | - | 视频描述文本 |
| negative_prompt | - | 负面提示词 |
| model_id | `Wan-AI/Wan2.1-T2V-14B-Diffusers` | HuggingFace 模型 ID |

### TPU Sampler
| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| height | 720 | 256-1280 | 视频高度 |
| width | 1280 | 256-1280 | 视频宽度 |
| num_frames | 81 | 17-121 | 视频帧数（需为 4n+1） |
| num_inference_steps | 50 | 1-100 | 去噪步数 |
| guidance_scale | 5.0 | 0-20 | CFG 引导强度 |
| seed | 2025 | - | 随机种子 |
| flow_shift | 5.0 | 1-10 | Flow Matching 位移（720P=5.0，480P=3.0） |

### VAE Decoder
| 参数 | 默认值 | 说明 |
|------|--------|------|
| fps | 16 | 输出视频帧率 |
| model_id | `Wan-AI/Wan2.1-T2V-14B-Diffusers` | HuggingFace 模型 ID |

## 分辨率推荐

| 分辨率 | height | width | flow_shift | 说明 |
|--------|--------|-------|------------|------|
| 720P | 720 | 1280 | 5.0 | 高质量，推荐 |
| 480P | 480 | 848 | 3.0 | 快速测试 |

## 示例 Workflow

示例 workflow 文件位于 `examples/` 目录：

- [`wan21_tpu_basic.json`](examples/wan21_tpu_basic.json) - 基础三节点 workflow

## 技术细节

### 2D Mesh 配置

使用 `(dp=2, tp=4)` 配置 8 个 TPU chips：
- dp: Data Parallel (batch sharding)
- tp: Tensor Parallel (weight sharding)

### Splash Attention

- 使用 exp2 代替 exp，利用 TPU VPU 硬件指令
- K-Smooth 技术减少数值溢出
- 长序列 (>20000) 使用 Splash Attention，短序列使用标准实现

### 权重分片策略

Transformer 使用 `('tp',)` 和 `(None, 'tp')` 分片模式，
VAE 使用 replicate（不分片）。

## 故障排除

### "torchax Tensors can only do math within the torchax environment"

这个错误已在最新版本中修复。确保使用最新代码：

```bash
cd ComfyUI-TPU
git pull
```

### 模型加载失败 / OOM

1. 确保 TPU 内存充足（720P 需要约 64GB）
2. 减少分辨率或帧数

### JAX 编译缓存

编译结果缓存在 `~/.cache/jax_cache`，首次运行较慢。如遇编译问题：

```bash
rm -rf ~/.cache/jax_cache
```

## 相关项目

- [diffusers-tpu](https://github.com/yangwhale/diffusers-tpu) - Wan 2.1 TPU 优化模型
- [gpu-tpu-pedia/Wan2.1](https://github.com/yangwhale/gpu-tpu-pedia/tree/main/tpu/Wan2.1) - 命令行版本
- [Torchax](https://github.com/pytorch/xla) - PyTorch-to-JAX bridge
- [Wan-AI](https://github.com/Wan-AI) - Wan 2.1 官方仓库

## License

MIT License
