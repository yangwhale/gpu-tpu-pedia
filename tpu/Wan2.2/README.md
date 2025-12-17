# Wan 2.2 Image-to-Video (I2V) on TPU

本项目实现了 Wan 2.2 I2V（图像到视频）模型在 Google Cloud TPU v5p 上的推理，采用三阶段分步执行架构，支持 81 帧 720×1280 分辨率视频生成。

## 目录

- [特性](#特性)
- [架构概览](#架构概览)
- [环境配置](#环境配置)
- [安装步骤](#安装步骤)
- [使用方法](#使用方法)
- [目录结构](#目录结构)
- [技术细节](#技术细节)
- [经验教训](#经验教训)
- [常见问题](#常见问题)

## 特性

- ✅ **三阶段分步执行**：编码器 → Transformer → 解码器
- ✅ **双 Transformer 架构**：高噪声阶段使用 `transformer`，低噪声阶段使用 `transformer_2`（boundary_ratio=0.9）
- ✅ **A14B 模式**：支持 `expand_timesteps=False` 的高效推理
- ✅ **Splash Attention**：TPU 优化的 attention 实现（exp2 优化）
- ✅ **中间结果缓存**：使用 SafeTensors 格式保存，支持分步调试
- ✅ **动态 VAE 参数**：从 VAE config 加载 `latents_mean` 和 `latents_std`

## 架构概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Wan 2.2 I2V 三阶段 Pipeline                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Stage 1: Encoder (TPU)                                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Text Encoder (UMT5-XXL)  ──→  prompt_embeds               │    │
│  │  VAE Encoder              ──→  latent_condition + mask     │    │
│  │                                                              │    │
│  │  输出: stage1_embeddings.safetensors                        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ▼                                        │
│  Stage 2: Transformer (TPU + Splash Attention)                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Dual Transformer Architecture:                              │    │
│  │  - t ≥ 0.9 × T_train: transformer (高噪声阶段)              │    │
│  │  - t < 0.9 × T_train: transformer_2 (低噪声阶段)            │    │
│  │                                                              │    │
│  │  40 步 denoising，CFG guidance_scale=3.5                    │    │
│  │  输出: stage2_latents.safetensors                           │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              ▼                                        │
│  Stage 3: VAE Decoder (TPU)                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  反归一化 latents  ──→  VAE Decode  ──→  视频帧             │    │
│  │                                                              │    │
│  │  输出: output_video.mp4                                      │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## 环境配置

### 硬件要求

- Google Cloud TPU v5p（推荐 8 chips）
- 内存：≥ 100GB

### 软件要求

- Python 3.10+
- JAX 0.4.35+（支持 TPU）
- PyTorch 2.4+
- torchax（最新版本）
- Diffusers（需要修改版本支持 Flax）

## 安装步骤

### 1. 创建 TPU 实例

```bash
# 创建 TPU v5p-8 实例
gcloud compute tpus tpu-vm create wan-tpu \
    --zone=us-east5-a \
    --accelerator-type=v5p-8 \
    --version=v2-alpha-tpuv5
```

### 2. 安装依赖

```bash
# SSH 连接到 TPU
gcloud compute tpus tpu-vm ssh wan-tpu --zone=us-east5-a

# 创建虚拟环境
python -m venv ~/venv
source ~/venv/bin/activate

# 安装 JAX for TPU
pip install -U jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# 安装 PyTorch
pip install torch torchvision

# 安装 torchax（从源码）
git clone https://github.com/pytorch/xla.git
cd xla/experimental/torch_xla2
pip install -e .
cd ~

# 安装 diffusers 和其他依赖
pip install diffusers transformers accelerate safetensors pillow imageio[ffmpeg]
```

### 3. 克隆代码

```bash
git clone https://github.com/your-repo/gpu-tpu-pedia.git
cd gpu-tpu-pedia/tpu/Wan2.2
```

### 4. 准备输入图像

```bash
# 将输入图像放到指定路径
cp /path/to/your/image.jpg ./wan_i2v_input.JPG
```

## 使用方法

### 方法一：三阶段分步执行（推荐用于调试）

```bash
cd generate_diffusers_i2v_flax_staged

# 阶段1：编码器
python stage1_encoder.py \
    --image ../wan_i2v_input.JPG \
    --prompt "A white cat wearing sunglasses sits on a surfboard..." \
    --size "720*1280" \
    --frames 81 \
    --output_dir ./stage_outputs

# 阶段2：Transformer
python stage2_transformer.py \
    --input_dir ./stage_outputs \
    --num_steps 40 \
    --guidance_scale 3.5 \
    --seed 42

# 阶段3：VAE 解码器
python stage3_vae_decoder.py \
    --input_dir ./stage_outputs \
    --fps 16
```

### 方法二：一键执行（完整 pipeline）

使用 `generate_i2v_flax.py` 在单个进程中完成全部推理流程（推荐用于生产环境）：

```bash
# 基本用法（使用默认参数）
python generate_i2v_flax.py

# 完整参数示例
python generate_i2v_flax.py \
    --model_id "Wan-AI/Wan2.2-I2V-A14B-Diffusers" \
    --image "wan_i2v_input.JPG" \
    --prompt "A white cat wearing sunglasses sits on a surfboard..." \
    --negative_prompt "低质量，模糊..." \
    --size "720*1280" \
    --frames 81 \
    --fps 16 \
    --num_steps 40 \
    --guidance_scale 3.5 \
    --seed 42 \
    --dp 2

# 启用 profiler
python generate_i2v_flax.py --profile --profile_output_path /tmp/wan_prof
```

### 命令行参数

#### generate_i2v_flax.py（一键执行）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_id` | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | 模型 ID |
| `--image` | (Hugging Face 示例图) | 输入图像路径或 URL |
| `--prompt` | (内置默认) | 正面提示词 |
| `--negative_prompt` | (内置默认) | 负面提示词 |
| `--size` | `720*1280` | 输出分辨率（可选：`720*1280`, `1280*720`, `480*832`, `832*480`） |
| `--frames` | `81` | 视频帧数 |
| `--fps` | `16` | 视频帧率 |
| `--num_steps` | `40` | 推理步数 |
| `--guidance_scale` | `3.5` | CFG 引导尺度 |
| `--seed` | `0` | 随机种子 |
| `--dp` | `2` | 数据并行维度 |
| `--profile` | `False` | 启用 JAX profiler |
| `--profile_output_path` | `/tmp/wan_prof` | Profiler 输出路径 |

#### stage1_encoder.py（三阶段）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--image` | `wan_i2v_input.JPG` | 输入图像路径 |
| `--prompt` | (内置默认) | 正面提示词 |
| `--negative_prompt` | (内置默认) | 负面提示词 |
| `--size` | `720*1280` | 输出分辨率 |
| `--frames` | `81` | 视频帧数 |
| `--output_dir` | `./stage_outputs` | 输出目录 |
| `--dp` | `2` | 数据并行维度 |

#### stage2_transformer.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input_dir` | `./stage_outputs` | 输入目录 |
| `--num_steps` | `40` | 推理步数 |
| `--guidance_scale` | `3.5` | CFG 引导尺度 |
| `--seed` | `0` | 随机种子 |
| `--warmup_steps` | `2` | 预热步数（JIT 编译） |
| `--profile` | `False` | 启用 JAX profiler |

#### stage3_vae_decoder.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input_dir` | `./stage_outputs` | 输入目录 |
| `--output_video` | (自动生成) | 输出视频路径 |
| `--fps` | `16` | 视频帧率 |

## 目录结构

```
Wan2.2/
├── README.md                           # 本文档
├── wan_i2v_input.JPG                   # 示例输入图像
├── custom_splash_attention.py          # TPU Splash Attention 实现
├── generate_i2v_flax.py                # 一键执行脚本
└── generate_diffusers_i2v_flax_staged/ # 三阶段分步实现
    ├── utils.py                        # 共享工具模块
    ├── stage1_encoder.py               # 阶段1：Text & Image Encoder
    ├── stage2_transformer.py           # 阶段2：Transformer
    └── stage3_vae_decoder.py           # 阶段3：VAE Decoder
```

## 技术细节

### A14B 模式（expand_timesteps=False）

在 A14B 模式下，图像条件的处理方式为：

```python
# 像素空间构建完整 video_condition
video_condition = concat([image, zeros, zeros, ...], dim=2)  # [B, 3, T, H, W]

# VAE 编码
latent_condition = vae.encode(video_condition).latent_dist.mode()

# 归一化
latent_condition = (latent_condition - mean) / std

# 构建 mask（第一帧=1，其他帧=0）
mask = ones([B, 4, T', H', W'])  # 4 是因为时间维度扩展

# 与 latent_condition 拼接
condition = concat([mask, latent_condition], dim=1)  # [B, 20, T', H', W']

# Transformer 输入
latent_model_input = concat([latents, condition], dim=1)  # [B, 36, T', H', W']
```

### 双 Transformer 切换

```python
boundary_timestep = 0.9 * scheduler.config.num_train_timesteps  # 900

for t in timesteps:
    if t >= boundary_timestep:
        model = transformer      # 高噪声阶段
    else:
        model = transformer_2    # 低噪声阶段
```

### VAE Latent 归一化

**重要**：必须从 VAE config 动态加载参数，不能硬编码！

```python
# 正确方式：从 VAE config 加载
latents_mean = vae.config.latents_mean  # 16 个值
latents_std = vae.config.latents_std    # 16 个值

# 归一化（编码后）
normalized = (latents - mean) / std

# 反归一化（解码前）
denormalized = latents * std + mean
```

### Splash Attention Block Size Padding

**关键 Bug 修复**：Splash Attention 对不完整的 block 会产生 NaN。

```python
# 问题分析
seq_len = 75348
BQSIZE = 3328
remainder = 75348 % 3328  # = 2132 tokens (2.83%)
# 这些 token 在不完整 block 中计算，导致 NaN

# 解决方案：pad 到 block size 的整数倍
def pad_to_block_multiple(x, block_size, axis):
    seq_len = x.shape[axis]
    pad_len = (block_size - seq_len % block_size) % block_size
    if pad_len == 0:
        return x, seq_len
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, pad_len)
    return jnp.pad(x, pad_width), seq_len

# 在 kernel_3d 中应用
q_padded, q_orig_len = pad_to_block_multiple(q, BQSIZE, axis=1)
k_padded, _ = pad_to_block_multiple(k, BKVSIZE, axis=1)
v_padded, _ = pad_to_block_multiple(v, BKVSIZE, axis=1)

# 计算 attention
out = splash_kernel(q_padded, k_padded, v_padded)

# 移除 padding
out = out[:, :q_orig_len, :]
```

## 经验教训

### 1. VAE 参数硬编码导致视频全黑

**问题**：最初硬编码了错误的 `latents_std` 值：
```python
# 错误值
LATENTS_STD = [5.4421, ...]  # CogVideoX 的值

# 正确值（从 Wan 2.2 VAE config）
LATENTS_STD = [2.8184, 1.4541, 2.3275, ...]
```

**解决**：动态从 VAE config 加载参数：
```python
latents_mean = vae.config.latents_mean
latents_std = vae.config.latents_std
```

### 2. Splash Attention NaN 问题

**问题**：2.83% 的 latents 在最后一帧出现 NaN。

**分析**：
- 序列长度 75,348 不是 BQSIZE=3,328 的整数倍
- 余数 2,132 tokens 在不完整 block 中计算
- Splash Attention kernel 对不完整 block 产生 NaN

**解决**：在 attention 计算前 pad 到 block size 的整数倍，计算后移除 padding。

### 3. Stage 3 dtype 不匹配

**问题**：Stage 2 保存的 latents 是 float32，但 VAE 期望 bfloat16。

**解决**：在 VAE 解码前转换 dtype：
```python
latents = latents.to(vae.dtype)  # 转换为 bfloat16
```

### 4. torchax 必须在模型加载后启用

**问题**：如果在加载模型前启用 torchax，safetensors 加载会失败。

**解决**：严格按顺序执行：
```python
# 1. 先加载模型
model = Model.from_pretrained(...)

# 2. 再启用 torchax
torchax.enable_globally()
env = torchax.default_env()

# 3. 最后移动到 XLA
move_module_to_xla(env, model)
```

### 5. PyTree 注册必须在 JAX 操作前完成

**问题**：没有注册 PyTree 节点会导致 JAX 无法处理 Hugging Face 的输出类型。

**解决**：在程序开始时注册所有需要的 PyTree 节点：
```python
jax.tree_util.register_pytree_node(
    DecoderOutput,
    flatten_fn,
    unflatten_fn
)
```

### 6. Scheduler shift 参数缺失导致视频动作快进

**问题**：三阶段脚本生成的视频动作跳跃、快进感强，与单步脚本质量差异明显。

**分析**：
- 单步脚本 `generate_i2v_flax.py` 使用 `WanImageToVideoPipeline`，Pipeline 内部自动读取模型配置中的 `shift=5.0`
- 三阶段脚本手动实现推理循环，调用 `scheduler.set_timesteps(num_steps)` 时**未设置 shift 参数**
- Diffusers 的 `FlowMatchEulerDiscreteScheduler` 默认 `shift=1.0`
- `shift` 参数控制采样时间步长分布：
  - `shift=5.0`：更多步数分配给低噪声阶段（生成细节、优化动作）
  - `shift=1.0`：时间步长均匀分布，减少细节阶段投入

**症状**：
- 视频动作变化过快、跳跃感强
- 细节不够精细
- 看起来像"快进"了

**解决**：
```python
# 错误方式（set_timesteps 不接受 shift 参数）
scheduler.set_timesteps(num_steps, shift=5.0)  # TypeError!

# 正确方式：使用 set_shift() 方法
scheduler.set_shift(5.0)  # 先设置 shift
scheduler.set_timesteps(num_steps)  # 再设置 timesteps
```

**关键代码**（`stage2_transformer.py`）：
```python
from utils import SHIFT  # SHIFT = 5.0

# 设置 shift（Wan 2.2 I2V 模型默认 shift=5.0）
shift_value = config.get('shift', SHIFT)
scheduler.set_shift(shift_value)

# 设置 timesteps
scheduler.set_timesteps(num_steps)
timesteps = scheduler.timesteps
```

**教训**：手动实现推理循环时，必须仔细对照原始 Pipeline 的每一个细节，尤其是 scheduler 配置参数。

## 常见问题

### Q1: 视频生成全黑或纯噪声

检查：
1. VAE 参数是否正确（从 config 加载）
2. latents 是否有 NaN 值
3. 归一化/反归一化是否正确应用

### Q2: OOM（内存不足）

尝试：
1. 减少分辨率（使用 `480*832`）
2. 减少帧数
3. 增加 data parallelism (`--dp 4`)

### Q3: 推理速度慢

检查：
1. 确认 JAX 编译缓存已启用
2. 使用 warmup 步骤触发 JIT 编译
3. 检查是否在 CPU 上运行

### Q4: Stage 2 程序不退出

这是 torchax 后台线程的已知问题。Stage 2 使用 `os._exit(0)` 强制退出。

## 性能数据

在 TPU v5p-8 上的参考性能：

| 配置 | Stage 1 | Stage 2 | Stage 3 | 总计 |
|------|---------|---------|---------|------|
| 720×1280, 81帧, 40步 | ~30s | ~180s | ~20s | ~230s |

---

## License

MIT License

## 致谢

- [Wan-AI](https://github.com/Wan-AI) - Wan 2.2 模型
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) - Diffusers 框架
- [Google JAX](https://github.com/google/jax) - JAX 框架
- [PyTorch XLA](https://github.com/pytorch/xla) - torchax