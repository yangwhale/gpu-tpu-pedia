# CogVideoX 三阶段生成流程

本目录包含将 `generate_flax.py` 拆分后的三阶段生成流程，用于在 TPU 上运行 CogVideoX 视频生成。

## 目录结构

```
generate_diffusers_flax_staged/
├── utils.py                    # 共享工具模块
├── stage1_text_encoder.py      # 阶段1：文本编码
├── stage2_transformer.py       # 阶段2：Transformer 推理
├── stage3_vae_decoder.py       # 阶段3：VAE 解码
├── stage_outputs/              # 中间文件存储目录
│   ├── generation_config.json  # 生成配置
│   ├── stage1_embeddings.safetensors  # 文本 embeddings
│   ├── stage2_latents.safetensors     # 生成的 latents
│   └── output_video.mp4               # 最终视频
└── README.md                   # 本文档
```

## 使用方法

### 阶段1：文本编码

```bash
python stage1_text_encoder.py \
  --prompt "A panda playing guitar in a bamboo forest" \
  --output_dir ./stage_outputs
```

**参数：**
- `--prompt`: 正面提示词
- `--negative_prompt`: 负面提示词（可选）
- `--model_id`: 模型 ID
- `--output_dir`: 输出目录

**输出文件：**
- `stage1_embeddings.safetensors`: prompt embeddings
- `generation_config.json`: 生成配置（仅包含文本相关参数）

### 阶段2：Transformer 推理 (TPU)

```bash
python stage2_transformer.py \
  --input_dir ./stage_outputs \
  --num_inference_steps 10 \
  --height 768 \
  --width 1360 \
  --frames 81 \
  --warmup_steps 2
```

**参数：**
- `--num_inference_steps`: 推理步数（默认10）
- `--height`: 视频高度（默认768）
- `--width`: 视频宽度（默认1360）
- `--frames`: 视频帧数（默认81）
- `--guidance_scale`: CFG 引导尺度（默认6.0）
- `--seed`: 随机种子（默认42）
- `--fps`: 视频帧率（默认8，保存到配置供阶段3使用）
- `--warmup_steps`: 预热步数（默认2）

**输出文件：**
- `stage2_latents.safetensors`: 生成的 latents

### 阶段3：VAE 解码 (TPU)

```bash
python stage3_vae_decoder.py \
  --input_dir ./stage_outputs \
  --output_video ./stage_outputs/output_video.mp4
```

**输出文件：**
- `output_video.mp4`: 最终生成的视频

## 配置参数

### 默认模型配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| MODEL_NAME | `zai-org/CogVideoX1.5-5B` | 模型 ID |
| WIDTH | 1360 | 视频宽度 |
| HEIGHT | 768 | 视频高度 |
| FRAMES | 81 | 视频帧数 |
| FPS | 8 | 输出视频帧率 |
| NUM_STEPS | 10 | 推理步数 |
| GUIDANCE_SCALE | 6.0 | CFG 引导尺度 |

### Splash Attention 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| BQSIZE | 2048 | Query 块大小 |
| BKVSIZE | 1024 | Key/Value 块大小 |
| BKVCOMPUTESIZE | 512 | KV 计算块大小 |
| BKVCOMPUTEINSIZE | 256 | KV 内层计算块大小 |
| USE_K_SMOOTH | True | 是否使用 K 平滑 |
| USE_CUSTOM_ATTENTION | True | 是否使用 exp2 优化版 |

### Sharding 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| USE_DP | False | 是否使用数据并行 |
| SP_NUM | 1 | 序列并行数量 |
| USE_FSDP | True | 是否使用 FSDP 模式 |

## 数据流

```
prompt (文本)
    ↓
[Stage 1: Text Encoder]
    ↓
embeddings.safetensors
    ↓
[Stage 2: Transformer + Splash Attention]
    ↓
latents.safetensors
    ↓
[Stage 3: Flax VAE Decoder]
    ↓
output_video.mp4
```

## 依赖项

- JAX/Flax (TPU 支持)
- PyTorch (模型加载)
- torchax (PyTorch-JAX 互操作)
- diffusers (CogVideoX Pipeline)
- transformers (T5 Text Encoder)
- safetensors (数据存储)
- imageio (视频导出)

## 与原版 generate_flax.py 的对比

| 特性 | generate_flax.py | 三阶段版本 |
|------|-----------------|------------|
| 内存使用 | 全部加载 | 分阶段加载 |
| 调试便利性 | 困难 | 可单独调试各阶段 |
| 中间结果 | 不保存 | 保存为 SafeTensors |
| 重试灵活性 | 需从头开始 | 可从任意阶段重试 |

## 注意事项

1. **阶段2 依赖 custom_splash_attention.py**：确保父目录中存在此文件
2. **阶段3 依赖 FlaxAutoencoderKLCogVideoX**：需要安装修改版 diffusers
3. **TPU 内存**：各阶段分开运行可降低峰值内存使用