# ComfyUI Wan 2.2 I2V TPU Nodes

在 TPU 上运行 Wan 2.2 Image-to-Video (I2V) 生成。

## 概述

本插件提供 4 个 ComfyUI 节点，实现 Wan 2.2 I2V A14B 模型在 TPU 上的推理：

| 节点 | 功能 | 输入 | 输出 |
|------|------|------|------|
| **Wan22I2VImageEncoder** | 图像条件编码 | IMAGE | CONDITION, LATENT_INFO |
| **Wan22I2VTextEncoder** | 文本编码 (UMT5-XXL) | prompt, negative_prompt | prompt_embeds, negative_prompt_embeds |
| **Wan22I2VTPUSampler** | 双 Transformer 去噪 | embeds, condition, latent_info | LATENT |
| **Wan22I2VTPUVAEDecoder** | VAE 解码 | LATENT | IMAGE |

## 工作流

```
┌─────────────┐     ┌─────────────────────┐
│ Load Image  │────▶│ Wan22I2VImageEncoder │──▶ condition
└─────────────┘     └─────────────────────┘     │
                                                 │
┌─────────────┐     ┌────────────────────┐      │
│ Prompt Text │────▶│ Wan22I2VTextEncoder │──┬──│
└─────────────┘     └────────────────────┘  │  │
                                             │  │
                    ┌────────────────────┐◀─┴──┘
                    │ Wan22I2VTPUSampler │
                    └────────────────────┘
                             │
                             ▼
                    ┌──────────────────────┐
                    │ Wan22I2VTPUVAEDecoder │──▶ VIDEO FRAMES
                    └──────────────────────┘
```

## 节点详情

### Wan22I2VImageEncoder

将输入图像编码为 latent condition（A14B 模式）。

**输入:**
- `image`: ComfyUI IMAGE 格式
- `height`: 目标高度 (默认 720)
- `width`: 目标宽度 (默认 1280)
- `num_frames`: 视频帧数 (默认 81)
- `model_id`: 模型路径 (可选)

**输出:**
- `condition`: 图像条件 tensor [B, 20, T_latent, H_latent, W_latent]
- `latent_info`: 尺寸信息字典

### Wan22I2VTextEncoder

使用 UMT5-XXL 编码文本提示词。

**输入:**
- `prompt`: 正面提示词
- `negative_prompt`: 负面提示词
- `model_id`: 模型路径 (可选)

**输出:**
- `prompt_embeds`: 正面 embeddings
- `negative_prompt_embeds`: 负面 embeddings

### Wan22I2VTPUSampler

使用双 Transformer 运行去噪循环。

**核心特点:**
- 使用 `boundary_ratio = 0.9` 切换双模型
- t >= 900: 使用 transformer (高噪声阶段)
- t < 900: 使用 transformer_2 (低噪声阶段)
- 使用 FlowMatchEulerDiscreteScheduler

**输入:**
- `prompt_embeds`: 文本 embeddings
- `negative_prompt_embeds`: 负面 embeddings
- `condition`: 图像条件
- `latent_info`: 尺寸信息
- `num_inference_steps`: 推理步数 (默认 40)
- `guidance_scale`: CFG 引导尺度 (默认 3.5)
- `seed`: 随机种子

**输出:**
- `latents`: 生成的 latents
- `num_frames`: 帧数

### Wan22I2VTPUVAEDecoder

使用 VAE 解码 latents 为视频帧。

**输入:**
- `latents`: LATENT dict
- `model_id`: 模型路径 (可选)
- `fps`: 帧率 (默认 16)

**输出:**
- `frames`: ComfyUI IMAGE 格式视频帧
- `fps`: 帧率

## 技术细节

### 双 Transformer 架构

Wan 2.2 I2V 使用双 Transformer 架构：
- **Transformer 1**: 处理高噪声阶段 (t >= 900)
- **Transformer 2**: 处理低噪声阶段 (t < 900)

切换阈值由 `BOUNDARY_RATIO = 0.9` 控制。

### A14B 模式

图像条件编码采用 A14B 模式：
1. 输入图像 resize 到目标分辨率
2. 构建 video_condition: [image, zeros, zeros, ...]
3. VAE 编码得到 latent_condition
4. 归一化: (x - mean) / std
5. 构建 mask (第一帧=1, 其他帧=0)
6. 拼接 condition = concat(mask, latent_condition)

### 分片策略

使用 2D Mesh (dp=2, tp=N/2) 进行模型并行：
- Text Encoder: 词嵌入和 FFN 分片
- Transformer: Attention 和 FFN 分片
- VAE: 复制 (不分片)

## 依赖

- JAX/XLA with TPU support
- torchax
- diffusers (with torchax patches)
- transformers
- ComfyUI

## 参考

基于 `gpu-tpu-pedia/tpu/Wan2.2/generate_diffusers_i2v_torchax_staged` 实现。
