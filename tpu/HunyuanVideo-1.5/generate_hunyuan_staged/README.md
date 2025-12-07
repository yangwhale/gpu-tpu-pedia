# HunyuanVideo-1.5 三阶段分离生成 (GPU 版本)

将 HunyuanVideo-1.5 文本到视频生成 Pipeline 拆分为三个独立阶段执行，适用于 GPU H100 8卡环境。

## 概述

| 阶段 | 脚本 | 功能 | 主要组件 |
|------|------|------|----------|
| Stage 1 | `stage1_text_encoder.py` | 文本编码 | LLM + ByT5 Text Encoder |
| Stage 2 | `stage2_transformer.py` | 视频生成 | DiT/Transformer |
| Stage 3 | `stage3_vae_decoder.py` | 视频解码 | VAE Decoder |

## 前提条件

1. **HunyuanVideo-1.5-TPU 代码库**：确保 `~/HunyuanVideo-1.5-TPU` 目录存在
2. **模型权重**：下载 HunyuanVideo-1.5 模型权重
3. **GPU 环境**：NVIDIA H100 或类似 GPU（推荐 8卡）
4. **依赖包**：
   ```bash
   pip install torch torchvision safetensors einops imageio loguru
   ```

## 快速开始

### 使用 Shell 脚本运行（推荐）

```bash
cd ~/gpu-tpu-pedia/tpu/HunyuanVideo-1.5/generate_hunyuan_staged

# 运行完整 pipeline（三个阶段顺序执行）
bash run_staged.sh

# 或分别运行各阶段
bash run_stage1.sh  # Text Encoder（单 GPU）
bash run_stage2.sh  # Transformer（8 GPU）
bash run_stage3.sh  # VAE Decoder（8 GPU）
```

### 直接使用 Python

```bash
cd ~/gpu-tpu-pedia/tpu/HunyuanVideo-1.5/generate_hunyuan_staged

# 阶段1：Text Encoder（单 GPU）
python stage1_text_encoder.py \
    --model_path /path/to/HunyuanVideo-1.5 \
    --prompt "A beautiful sunset over the ocean" \
    --output_dir ./stage_outputs

# 阶段2：Transformer（8 GPU，使用 torchrun）
torchrun --nproc_per_node=8 stage2_transformer.py \
    --input_dir ./stage_outputs

# 阶段3：VAE Decoder（8 GPU，使用 torchrun）
torchrun --nproc_per_node=8 stage3_vae_decoder.py \
    --input_dir ./stage_outputs

# 查看生成的视频
ls ./stage_outputs/output_video.mp4
```

## 详细说明

### Stage 1: Text Encoder

编码文本 prompt 为 embeddings，包括 LLM 和 ByT5 编码。

```bash
# 必需参数
python stage1_text_encoder.py \
    --model_path /path/to/HunyuanVideo-1.5 \
    --prompt "Your prompt here"

# 完整参数示例
python stage1_text_encoder.py \
    --model_path /path/to/HunyuanVideo-1.5 \
    --prompt "A cat playing piano in a jazz club" \
    --negative_prompt "blurry, low quality, distorted" \
    --resolution 720p \
    --output_dir ./my_outputs
```

**参数说明：**
- `--model_path`: 模型权重路径（必需）
- `--prompt`: 正面提示词（必需）
- `--negative_prompt`: 负面提示词
- `--resolution`: 视频分辨率 (`480p` 或 `720p`)，用于选择 transformer 版本
- `--image_path`: 参考图片路径（用于 i2v 模式）
- `--output_dir`: 输出目录
- `--dtype`: 数据类型（`bf16` 或 `fp32`）

**输出文件：**
- `stage_outputs/stage1_embeddings.safetensors` - Prompt embeddings
- `stage_outputs/generation_config.json` - 配置文件

### Stage 2: Transformer (DiT)

运行 denoising loop 生成 latents。

```bash
# 基本用法（使用默认参数）
python stage2_transformer.py --input_dir ./stage_outputs

# 完整参数示例
python stage2_transformer.py \
    --input_dir ./stage_outputs \
    --aspect_ratio 16:9 \
    --video_length 121 \
    --num_inference_steps 50 \
    --guidance_scale 6.0 \
    --seed 42
```

**参数说明（Stage 2 专属）：**
- `--input_dir`: 输入目录（包含 stage1 输出）
- `--output_dir`: 输出目录（默认与 input_dir 相同）
- `--aspect_ratio`: 视频宽高比（如 `16:9`, `9:16`，默认 `16:9`）
- `--video_length`: 视频帧数（默认 121，约5秒 @24fps）
- `--num_inference_steps`: 推理步数（默认 50）
- `--guidance_scale`: CFG 引导尺度（默认 6.0）
- `--seed`: 随机种子（默认 42）
- `--enable_offloading`: 启用模型 offloading（默认启用）
- `--disable_offloading`: 禁用模型 offloading

**输出文件：**
- `stage_outputs/stage2_latents.safetensors` - 生成的 latents

### Stage 3: VAE Decoder

解码 latents 为最终视频。

```bash
# 基本用法
python stage3_vae_decoder.py --input_dir ./stage_outputs

# 完整参数
python stage3_vae_decoder.py \
    --input_dir ./stage_outputs \
    --output_video my_video.mp4 \
    --fps 24 \
    --save_frames
```

**参数说明：**
- `--input_dir`: 输入目录（包含 stage2 输出）
- `--output_video`: 输出视频路径
- `--fps`: 视频帧率（默认 24）
- `--disable_tiling`: 禁用 VAE tiling（需要更多内存）
- `--save_frames`: 保存原始帧 tensor（调试用）

**输出文件：**
- `stage_outputs/output_video.mp4` - 最终视频
- `stage_outputs/stage3_frames.safetensors` - 原始帧 tensor（可选）

## 中间文件格式

所有中间数据使用 SafeTensors 格式存储，与 TPU 版本格式完全一致。

### Stage 1 Embeddings 格式

Stage 1 输出 8 个 tensor，分开存储 positive/negative embeddings：

| Tensor Name | Shape | 说明 |
|-------------|-------|------|
| `prompt_embeds` | [1, 1000, 3584] | LLM 正向 embedding |
| `negative_prompt_embeds` | [1, 1000, 3584] | LLM 负向 embedding |
| `prompt_embeds_mask` | [1, 1000] | LLM 正向 mask |
| `negative_prompt_embeds_mask` | [1, 1000] | LLM 负向 mask |
| `prompt_embeds_2` | [1, 256, 1472] | ByT5 正向 embedding |
| `negative_prompt_embeds_2` | [1, 256, 1472] | ByT5 负向 embedding |
| `prompt_embeds_mask_2` | [1, 256] | ByT5 正向 mask |
| `negative_prompt_embeds_mask_2` | [1, 256] | ByT5 负向 mask |

### 文件汇总

| 文件 | 内容 | 典型 Shape |
|------|------|------------|
| `stage1_embeddings.safetensors` | Prompt embeddings（8 个 tensor） | 见上表 |
| `stage2_latents.safetensors` | 生成的 latents | [1, 32, T/4, H/16, W/16] |
| `stage3_frames.safetensors` | 解码后的帧 | [1, 3, T, H, W] |
| `generation_config.json` | 配置参数 | JSON |

## 分布式运行

### Stage 2: Transformer（Sequence Parallelism）

Stage 2 使用 Sequence Parallelism (SP) 在多卡上并行处理 tokens：

```bash
# 8 GPU 运行（推荐）
torchrun --nproc_per_node=8 stage2_transformer.py --input_dir ./stage_outputs

# 4 GPU 运行
torchrun --nproc_per_node=4 stage2_transformer.py --input_dir ./stage_outputs
```

### Stage 3: VAE Decoder（Tile Parallelism）

Stage 3 使用 Tile Parallelism 在多卡上并行解码 spatial tiles：

```bash
# 8 GPU 运行（推荐）
torchrun --nproc_per_node=8 stage3_vae_decoder.py --input_dir ./stage_outputs
```

### 混合配置

```bash
# 在 GPU 0 上运行 Stage 1（单卡足够）
CUDA_VISIBLE_DEVICES=0 python stage1_text_encoder.py ...

# 在 GPU 0-7 上运行 Stage 2（多卡 SP）
torchrun --nproc_per_node=8 stage2_transformer.py ...

# 在 GPU 0-7 上运行 Stage 3（多卡 Tile Parallelism）
torchrun --nproc_per_node=8 stage3_vae_decoder.py ...
```

## 内存优化

### Transformer (Stage 2)
- **直接加载 Transformer**：避免加载整个 pipeline（节省 ~57GB/GPU）
- **Group Offloading**：默认启用，将 transformer blocks 分组 offload 到 CPU
- **Stream Offloading**：使用 CUDA streams 实现异步 offloading
- **Sequence Parallelism**：在多卡间分割 tokens 处理

### VAE (Stage 3)
- **Tiling**：默认启用，将大图分块解码（tile_size=256）
- **Tile Parallelism**：在多卡间分配 spatial tiles
- **FP16**：VAE 使用 FP16 节省内存
- **torch.no_grad()**：禁用梯度避免内存累积

### 内存使用（H100 8卡）

| 阶段 | 模型内存 | 峰值内存 | 备注 |
|------|----------|----------|------|
| Stage 1 | ~14GB | ~20GB | 单卡，LLM + ByT5 |
| Stage 2 | ~15.6GB | ~35GB | 每卡，Transformer only |
| Stage 3 | ~1GB | ~30GB | 每卡，VAE + tiles |

## 性能基准

在 H100 8卡上的实测性能（49帧，720p，50步）：

| 阶段 | 耗时 | GPU 数 | 备注 |
|------|------|--------|------|
| Stage 1 | ~5 秒 | 1 | Text encoding |
| Stage 2 | ~80 秒 | 8 | Transformer with SP |
| Stage 3 | ~4 秒 | 8 | VAE with Tile Parallelism |
| **总计** | **~89 秒** | | |

## Attention 模式性能对比

Stage 2 支持多种 Attention 实现模式，可通过 `--attn_mode` 参数切换：

### 可用模式

| 模式 | 说明 | 硬件要求 |
|------|------|----------|
| `flash` / `flash2` | Flash Attention 2（默认） | Ampere+ GPU |
| `flash3` | Flash Attention 3 | Hopper GPU (H100) |
| `sageattn` | SageAttention (INT8 量化) | Ampere+ GPU |
| `flex-block-attn` | Sparse Attention (SSTA) | Hopper GPU (H100) |
| `torch` | PyTorch 原生 SDPA | 所有 GPU |

### SageAttention vs Flash Attention 2 实测对比

**测试条件**：
- 硬件：NVIDIA H100 × 8
- 分辨率：720p (1280×720)
- 帧数：121 帧（约 5 秒 @24fps）
- 推理步数：50 步
- CFG Scale：6.0
- Prompt："A young woman with beautiful clear eyes and blonde hair in a suit is sitting in a high-end restaurant, looking at a book"

**性能结果**：

| 指标 | Flash Attention 2 | SageAttention | 变化 |
|------|-------------------|---------------|------|
| 每步耗时 | ~5.2 秒 | ~3.25 秒 | -37.5% |
| 总耗时 (50步) | ~260 秒 | ~162 秒 | -37.7% |
| 加速比 | 1.0x | **1.6x** | |
| 显存占用 | ~35GB/GPU | ~33GB/GPU | -5.7% |

**质量对比**：

| 维度 | Flash Attention 2 | SageAttention |
|------|-------------------|---------------|
| 整体画质 | ✅ 优秀 | ⚠️ 有损 |
| 细节保留 | ✅ 丰富细节 | ❌ 细节丢失 |
| 背景复杂度 | ✅ 复杂彩色书架 | ❌ 简单木栅栏 |
| 人物一致性 | ✅ 一致 | ✅ 一致 |
| 运动流畅度 | ✅ 流畅 | ✅ 流畅 |

**质量下降原因**：
- SageAttention 使用 INT8 量化 Q/K 矩阵以加速计算
- HunyuanVideo 1.5 有 53 层 Transformer，每层都会引入量化误差
- 长序列（111,600 video tokens + 1,256 text tokens = 112,856 tokens）放大误差累积
- 最终表现为：复杂背景细节丢失、色彩单一化、纹理简化

**使用建议**：

| 场景 | 推荐模式 | 原因 |
|------|----------|------|
| 生产环境/最终输出 | `flash` / `flash2` | 质量最优 |
| 快速迭代/预览 | `sageattn` | 1.6x 加速，可接受质量损失 |
| H100 + 超长视频 | `flex-block-attn` | Sparse Attention 适合长序列 |
| 调试/兼容模式 | `torch` | 最大兼容性 |

### 使用方法

```bash
# 使用 Flash Attention 2（默认，推荐生产环境）
bash run_stage2.sh

# 使用 SageAttention（快速预览）
bash run_stage2.sh --use_sageattn

# 使用 Flash Attention 3（H100 专属）
bash run_stage2.sh --attn_mode flash3

# 使用 Sparse Attention（H100 + 超长视频）
bash run_stage2.sh --sparse_attn

# 指定任意 attention 模式
bash run_stage2.sh --attn_mode sageattn
```

## 与完整版对比

三阶段分离版本与 `generate.py` 完整版本功能等价，主要区别：

1. **可调试性**：可以单独运行每个阶段，保存中间结果
2. **灵活性**：可以在不同 GPU 上运行不同阶段
3. **内存效率**：每个阶段独立运行，峰值内存更低
4. **可复用性**：相同的 embeddings 可用于多次生成

## 问题排查

### Import 错误
确保 `~/HunyuanVideo-1.5-TPU` 目录存在且包含正确的代码：
```bash
ls ~/HunyuanVideo-1.5-TPU/hyvideo
```

### OOM 错误
1. 启用 offloading：`--enable_offloading`（默认）
2. 减少 `--video_length`
3. 使用较低分辨率 `--resolution 480p`
4. Stage 3 启用 tiling（默认）

### CUDA 版本问题
确保 PyTorch 和 CUDA 版本兼容：
```bash
python -c "import torch; print(torch.version.cuda)"
```

## 文件结构

```
generate_hunyuan_staged/
├── README.md              # 本文档
├── utils.py               # 共享工具函数
├── stage1_text_encoder.py # 阶段1：文本编码
├── stage2_transformer.py  # 阶段2：Transformer 推理
├── stage3_vae_decoder.py  # 阶段3：VAE 解码
├── run_staged.sh          # 完整 pipeline 运行脚本
├── run_stage1.sh          # Stage 1 运行脚本
├── run_stage2.sh          # Stage 2 运行脚本
└── run_stage3.sh          # Stage 3 运行脚本
```

## 关键实现细节与 OOM 问题分析

### Stage 2 OOM：不要用 create_pipeline

**问题**：直接调用 `create_pipeline()` 会加载所有组件（Text Encoder 14GB + Transformer 26GB + VAE + ByT5），
导致每个 GPU 占用 ~73GB。

**解决方案**：Stage 2 只需要 Transformer，直接加载：

```python
# ❌ 错误：加载所有组件，~73GB/GPU
pipeline = create_pipeline(model_path, ...)

# ✓ 正确：只加载 Transformer，~15.6GB/GPU
transformer = HunyuanVideo_1_5_DiffusionTransformer.from_pretrained(
    transformer_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
```

### Stage 3 OOM：必须使用 torch.no_grad()

**问题**：VAE 解码时没有使用 `torch.no_grad()`，导致 PyTorch 保存中间激活用于反向传播，
每个 GPU 占用 ~74GB（几乎全部是梯度相关内存）。

**解决方案**：推理时必须禁用梯度：

```python
# ❌ 错误：保存梯度，~74GB/GPU
video_frames = vae.decode(latents, ...)

# ✓ 正确：禁用梯度，~30GB/GPU
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
    video_frames = vae.decode(latents, ...)
```

### 两个 OOM 问题对比

| 阶段 | OOM 根因 | 修复前内存 | 修复后内存 | 解决方案 |
|------|----------|------------|------------|----------|
| Stage 2 | `create_pipeline` 加载所有组件 | ~73GB/GPU | ~15.6GB/GPU | 直接加载 Transformer |
| Stage 3 | 未使用 `torch.no_grad()` | ~74GB/GPU | ~30GB/GPU | 添加 `torch.no_grad()` |

## License

与 HunyuanVideo 项目保持一致。