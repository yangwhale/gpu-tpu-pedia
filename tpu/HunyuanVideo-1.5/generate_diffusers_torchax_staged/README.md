# HunyuanVideo-1.5 三阶段分离生成（Diffusers 版本）

将 HunyuanVideo-1.5 文本到视频生成 Pipeline 拆分为三个独立阶段执行，便于调试、分析和优化。

> ⚠️ **依赖要求**：本目录必须使用 [diffusers-tpu](https://github.com/yangwhale/diffusers-tpu)，不能使用官方 diffusers。

## 环境安装

```bash
# 安装 diffusers-tpu（必需）
git clone https://github.com/yangwhale/diffusers-tpu.git ~/diffusers-tpu
cd ~/diffusers-tpu && pip install -e . && cd -

# 其他依赖
pip install transformers accelerate safetensors
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install torch torchvision torchax
```

## 概述

| 阶段 | 脚本 | 功能 | 设备 |
|------|------|------|------|
| Stage 1 | `stage1_text_encoder.py` | Text Encoder (Qwen2.5-VL + T5) | CPU |
| Stage 2 | `stage2_transformer.py` | DiT/Transformer + Splash Attention | TPU |
| Stage 3 | `stage3_vae_decoder.py` | Flax VAE Decoder | TPU |

## 快速开始

```bash
cd ~/gpu-tpu-pedia/tpu/HunyuanVideo-1.5/generate_diffusers_flax_staged

# 默认配置: 49帧, 720p, 50步
python stage1_text_encoder.py
python stage2_transformer.py
python stage3_vae_decoder.py

# 查看生成的视频
ls ./stage_outputs/output_video.mp4
```

## 详细说明

### Stage 1: Text Encoder

编码文本 prompt 为 embeddings。

```bash
# 默认 prompt
python stage1_text_encoder.py

# 自定义 prompt
python stage1_text_encoder.py --prompt "A cat playing piano"

# 使用负面提示词
python stage1_text_encoder.py --prompt "A beautiful sunset" --negative_prompt "blurry, low quality"
```

**输出文件：**
- `stage_outputs/stage1_embeddings.safetensors` - Prompt embeddings
- `stage_outputs/generation_config.json` - 配置文件

### Stage 2: Transformer (DiT)

使用 TPU + Splash Attention 运行 denoising loop 生成 latents。

```bash
# 默认配置 (49帧, 720p, 50步)
python stage2_transformer.py

# 自定义参数
python stage2_transformer.py --num_frames 49 --num_inference_steps 30 --seed 123

# 使用 sliding window attention（长视频）
python stage2_transformer.py --num_frames 97 --window_size 8192
```

**参数说明：**
- `--num_frames`: 视频帧数（默认 49，约2秒）
- `--num_inference_steps`: 推理步数（默认 50）
- `--height`, `--width`: 分辨率（默认 720×1280）
- `--guidance_scale`: CFG 引导尺度（默认 6.0）
- `--seed`: 随机种子（默认 42）
- `--window_size`: 滑动窗口大小，用于长视频（默认 None=全注意力）

**输出文件：**
- `stage_outputs/stage2_latents.safetensors` - 生成的 latents

### Stage 3: VAE Decoder

使用 Flax VAE 解码 latents 为视频帧。

```bash
# 默认配置（禁用 tiling，最佳性能）
python stage3_vae_decoder.py

# 启用 tiling（内存受限时）
python stage3_vae_decoder.py --enable_tiling

# 指定输出路径
python stage3_vae_decoder.py --output_video my_video.mp4
```

**参数说明：**
- `--enable_tiling`: 启用 VAE tiling（默认禁用以获得最佳性能）
- `--output_video`: 输出视频路径（默认 `stage_outputs/output_video.mp4`）
- `--fps`: 视频帧率（默认 24）

**输出文件：**
- `stage_outputs/stage3_frames.safetensors` - 原始帧 tensor（调试用）
- `stage_outputs/output_video.mp4` - 最终视频

## 中间文件格式

所有中间数据使用 SafeTensors 格式存储，支持 bfloat16。

| 文件 | 内容 | Shape | dtype |
|------|------|-------|-------|
| `stage1_embeddings.safetensors` | 8个 prompt embedding tensors | 多种 | bfloat16 |
| `stage2_latents.safetensors` | latents | [1, 32, T/4, H/16, W/16] | bfloat16 |
| `stage3_frames.safetensors` | 解码后的帧 | [1, T, H, W, 3] | bfloat16 |

## 性能基准

在 TPU v4-8 上的典型性能（49帧，720p）：

| 阶段 | 首次运行（含编译） | 后续运行 |
|------|-------------------|---------|
| Stage 1 | ~2 秒 | ~2 秒 |
| Stage 2 | ~233 秒 | ~100 秒 |
| Stage 3（禁用 tiling） | ~36 秒 | ~36 秒 |
| Stage 3（启用 tiling） | ~793 秒 | ~793 秒 |

**建议：** 在 TPU 上禁用 tiling 可获得最佳性能（22倍加速）。

## 内存限制

720p 视频帧数限制（无 sliding window）：

| 帧数 | 状态 |
|------|------|
| 49 | ✅ 正常 |
| 61 | ❌ OOM |
| 73+ | ❌ OOM |

对于更长的视频，请使用 `--window_size` 参数启用 sliding window attention。

## 依赖

- JAX (TPU)
- torchax
- **diffusers-tpu**（必需，[github.com/yangwhale/diffusers-tpu](https://github.com/yangwhale/diffusers-tpu)）
- safetensors
- transformers

> ⚠️ 不能使用官方 `pip install diffusers`，必须从源码安装 diffusers-tpu。

## 工具模块

`utils.py` 提供共享的工具函数：

- `save_embeddings_to_safetensors()` / `load_embeddings_from_safetensors()` - 保存/加载 embeddings
- `save_latents_to_safetensors()` / `load_latents_from_safetensors()` - 保存/加载 latents
- `save_generation_config()` / `load_generation_config()` - 保存/加载配置
- `setup_pytree_registrations()` - 注册 PyTree 节点

## 与完整版对比

三阶段分离版本与 `../generate_diffusers_flax.py` 完整版本功能等价，主要区别：

1. **可调试性**：可以单独运行每个阶段，保存中间结果
2. **灵活性**：可以替换单个阶段（如使用不同的 VAE）
3. **内存效率**：每个阶段独立运行，不需要同时加载所有模型

## 问题排查

### OOM 错误

如果遇到内存不足错误：
1. 减少 `--num_frames`（建议 49）
2. 使用 `--window_size 8192` 启用 sliding window attention
3. 在 Stage 3 使用 `--enable_tiling`

### 编译缓存

JAX 编译缓存存储在 `/dev/shm/jax_cache`。首次运行会较慢，后续运行会使用缓存加速。

```bash
# 清除编译缓存
rm -rf /dev/shm/jax_cache
```

## License

与 HunyuanVideo 项目保持一致。