# Wan 2.1 TPU 视频生成

本项目提供 Wan 2.1 Text-to-Video 模型在 TPU 上的高效推理实现，支持 JAX/Flax 和 Splash Attention 优化。

## 目录

- [项目概述](#项目概述)
- [环境要求](#环境要求)
- [安装步骤](#安装步骤)
- [使用方法](#使用方法)
  - [单步推理](#单步推理-generate_flaxpy)
  - [三阶段推理](#三阶段推理)
- [中间文件说明](#中间文件说明)
- [参数详解](#参数详解)
- [已知问题与解决方案](#已知问题与解决方案)
- [性能优化](#性能优化)
- [技术细节](#技术细节)

---

## 项目概述

### 目录结构

```
Wan2.1/
├── generate_flax.py                 # 单步推理脚本（一次性执行全流程）
├── custom_splash_attention.py       # 自定义 Splash Attention 实现
├── generate_diffusers_flax_staged/  # 三阶段推理目录
│   ├── utils.py                     # 共享工具函数和配置
│   ├── stage1_text_encoder.py       # 阶段1：Text Encoding
│   ├── stage2_transformer.py        # 阶段2：Transformer Denoising
│   ├── stage3_vae_decoder.py        # 阶段3：VAE Decoding
│   ├── stage_outputs/               # 中间输出目录
│   │   ├── stage1_embeddings.safetensors
│   │   ├── stage2_latents.safetensors
│   │   ├── generation_config.json
│   │   └── output_video.mp4
│   └── README.md                    # 本文档
└── diffusers-tpu/                   # 修改后的 Diffusers 库
    └── src/diffusers/pipelines/wan/
        └── pipeline_wan_flax.py     # 自定义 Pipeline 实现
```

### 两种推理模式

| 特性 | 单步推理 (`generate_flax.py`) | 三阶段推理 |
|------|------------------------------|-----------|
| 执行方式 | 一次性完成全部流程 | 可分步执行，支持暂停/恢复 |
| 调试便利性 | 难以在中途检查状态 | 可在任意阶段检查中间结果 |
| 内存占用 | 峰值较高（所有组件同时加载） | 各阶段独立加载/释放 |
| 结果复用 | 每次重新计算 | 可复用 embeddings 和 latents |
| 适用场景 | 快速生成、基准测试 | 开发调试、批量生成、实验对比 |

---

## 环境要求

### 硬件

- **TPU**: v4-8 或 v6e-8（8 chips 最小配置）
- **内存**: 建议 64GB+ 系统内存
- **存储**: 约 50GB 用于模型权重缓存

### 软件

- Python 3.10+
- JAX 0.4.35+（TPU 版本）
- PyTorch 2.5+（CPU 版本即可）
- torchax 0.0.11+

---

## 安装步骤

### 快速安装（推荐）

在 TPU VM 上运行以下命令：

```bash
# 1. 安装 PyTorch（CPU 版本，TPU 上不需要 CUDA）
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. 安装 JAX TPU 版本
pip install -U jax[tpu]

# 3. 安装 torchax（PyTorch-JAX 桥接）
pip install torchax

# 4. 安装其他依赖
pip install transformers accelerate safetensors
pip install opencv-python imageio imageio-ffmpeg
pip install flax optax

# 5. 克隆并安装修改版 diffusers
git clone https://github.com/yangwhale/diffusers-tpu.git
cd diffusers-tpu && pip install -e . && cd ..

# 6. 克隆并安装 MaxDiffusion（用于 VAE）
git clone https://github.com/AI-Hypercomputer/maxdiffusion.git
cd maxdiffusion && pip install -e . && cd ..

# 7. 克隆项目代码
git clone https://github.com/yangwhale/gpu-tpu-pedia.git
cd gpu-tpu-pedia/tpu/Wan2.1
```

### 验证安装

```bash
# 检查 JAX 设备
python -c "import jax; print(f'JAX devices: {jax.devices()}')"
# 预期输出: [TpuDevice(id=0, ...), TpuDevice(id=1, ...), ...]

# 检查 torchax 版本
python -c "import torchax; print(f'torchax version: {torchax.__version__}')"
# 预期输出: torchax version: 0.0.11

# 检查 Pipeline 导入
python -c "from diffusers.pipelines.wan.pipeline_wan_flax import WanPipeline; print('Pipeline OK')"
```

### 分步安装说明

如果快速安装遇到问题，可以按以下步骤逐一安装：

#### 1. 创建虚拟环境（可选）

```bash
python -m venv wan-venv
source wan-venv/bin/activate
```

#### 2. 安装 PyTorch（CPU 版本）

```bash
# Linux
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Mac (M1/M2)
pip install torch
```

**注意**: TPU 计算不需要 CUDA，安装 CPU 版本的 PyTorch 即可。torchax 会将 PyTorch 操作转换为 JAX 执行。

#### 3. 安装 JAX（TPU 版本）

```bash
pip install -U jax[tpu]
```

#### 4. 安装 torchax

```bash
pip install torchax
```

torchax 是 PyTorch 到 JAX 的桥接库，允许在 TPU 上运行 PyTorch 模型。

#### 5. 安装修改版 Diffusers

```bash
git clone https://github.com/yangwhale/diffusers-tpu.git
cd diffusers-tpu
pip install -e .
cd ..
```

这个修改版包含：
- `pipeline_wan_flax.py`: 支持 TPU 分片的 Wan Pipeline
- `scheduling_unipc_multistep.py`: 修复 bfloat16 精度问题

#### 6. 安装 MaxDiffusion（用于 VAE）

```bash
git clone https://github.com/AI-Hypercomputer/maxdiffusion.git
cd maxdiffusion
pip install -e .
cd ..
```

MaxDiffusion 提供 Flax 实现的 Wan VAE。

#### 7. 安装其他依赖

```bash
pip install transformers accelerate safetensors
pip install opencv-python imageio imageio-ffmpeg
pip install flax optax
```

---

## 使用方法

### 单步推理 (`generate_flax.py`)

一次性执行完整的 Text-to-Video 生成流程。

```bash
cd gpu-tpu-pedia/tpu/Wan2.1

# 基本用法
python generate_flax.py

# 自定义参数
python generate_flax.py \
    --width 1280 \
    --height 720 \
    --frames 81 \
    --num_inference_steps 50 \
    --use_dp \
    --use_custom_attention
```

**输出**: 当前目录下的 `YYYYMMDD_HHMMSS.mp4` 文件

### 三阶段推理

#### 完整执行流程

```bash
cd gpu-tpu-pedia/tpu/Wan2.1/generate_diffusers_flax_staged

# 阶段1：Text Encoding（可在 CPU 运行）
python stage1_text_encoder.py

# 阶段2：Transformer Denoising（需要 TPU）
python stage2_transformer.py --num_inference_steps 50

# 阶段3：VAE Decoding（需要 TPU）
python stage3_vae_decoder.py
```

#### 阶段1：Text Encoder

将 prompt 编码为 embeddings。

```bash
python stage1_text_encoder.py \
    --prompt "A cat playing piano in a jazz club" \
    --negative_prompt "blurry, low quality, static" \
    --width 1280 \
    --height 720 \
    --frames 81 \
    --output_dir ./stage_outputs
```

**输出**:
- `stage_outputs/stage1_embeddings.safetensors` - Prompt embeddings
- `stage_outputs/generation_config.json` - 生成配置

#### 阶段2：Transformer

执行去噪循环，生成 latents。

```bash
python stage2_transformer.py \
    --input_dir ./stage_outputs \
    --num_inference_steps 50
```

**输入**: 阶段1 的输出文件  
**输出**: `stage_outputs/stage2_latents.safetensors`

#### 阶段3：VAE Decoder

将 latents 解码为视频。

```bash
python stage3_vae_decoder.py \
    --input_dir ./stage_outputs \
    --output_video ./stage_outputs/output_video.mp4
```

**输入**: 阶段2 的 latents  
**输出**: `stage_outputs/output_video.mp4`

---

## 中间文件说明

### `stage1_embeddings.safetensors`

| 字段 | 形状 | 说明 |
|------|------|------|
| `prompt_embeds` | `[1, 226, 4096]` | 正面 prompt 的 T5 编码 |
| `negative_prompt_embeds` | `[1, 226, 4096]` | 负面 prompt 的 T5 编码 |

### `stage2_latents.safetensors`

| 字段 | 形状 | 说明 |
|------|------|------|
| `latents` | `[1, 16, 21, 90, 160]` | 去噪后的潜空间表示 |

- Batch: 1
- Channels: 16（Wan VAE 的 z_dim）
- Temporal: 21（81 帧 ÷ 4 temporal downsampling）
- Height: 90（720 ÷ 8 spatial downsampling）
- Width: 160（1280 ÷ 8 spatial downsampling）

### `generation_config.json`

```json
{
  "prompt": "...",
  "negative_prompt": "...",
  "width": 1280,
  "height": 720,
  "frames": 81,
  "fps": 16,
  "num_inference_steps": 50,
  "guidance_scale": 5.0,
  "seed": 42,
  "model_id": "Wan-AI/Wan2.1-T2V-14B-Diffusers"
}
```

---

## 参数详解

### 视频生成参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--width` | 1280 | 视频宽度（像素） |
| `--height` | 720 | 视频高度（像素） |
| `--frames` | 81 | 视频帧数（需为 4n+1） |
| `--fps` | 16 | 视频帧率 |
| `--num_inference_steps` | 50 | 去噪步数（更多=更高质量） |
| `--guidance_scale` | 5.0 | CFG 强度 |
| `--seed` | 42 | 随机种子 |
| `--flow_shift` | 5.0 | Flow Matching 位移（720P=5.0，480P=3.0） |

### Splash Attention 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--bqsize` | 3328 | Query 块大小 |
| `--bkvsize` | 2816 | Key-Value 块大小 |
| `--bkvcomputesize` | 256 | KV 计算块大小 |
| `--use_custom_attention` | True | 使用 exp2 优化的自定义 attention |
| `--window_size` | None | 局部注意力窗口（None=全局） |

### 分片参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_dp` | True | 启用数据并行（推荐） |
| `--use_fsdp` | True | 启用 FSDP 模式 |
| `--sp_num` | 1 | Sequence Parallel 分片数 |

---

## 已知问题与解决方案

### 1. MaxDiffusion VAE 颜色反转

**问题**: MaxDiffusion 的 Flax VAE 输出图像颜色反转（负片效果）。

**解决方案**: `stage3_vae_decoder.py` 中已实现 `invert_colors=True` 修复：
```python
video = 255 - video  # 颜色反转修复
```

**根本原因**: MaxDiffusion VAE 与 Diffusers VAE 的输出归一化方式不同。这是一个已知的兼容性问题。

### 2. Text Encoder None 检查

**问题**: 三阶段模式下，Pipeline 中 `text_encoder=None` 导致 `.device` 访问失败。

**解决方案**: 已在 `pipeline_wan_flax.py` 中添加 None 检查：
```python
if self.text_encoder is not None:
    device = self.text_encoder.device
```

### 3. bfloat16 保存/加载

**问题**: SafeTensors 不直接支持 bfloat16，导致加载后类型不匹配。

**解决方案**: `utils.py` 中自动处理类型转换：
```python
# 保存时记录原始类型
meta = {'dtype': 'bfloat16'}

# 加载时恢复
if meta.get('dtype') == 'bfloat16':
    tensor = tensor.to(torch.bfloat16)
```

### 4. PyTree 注册

**问题**: `BaseModelOutputWithPastAndCrossAttentions` 未注册为 PyTree，导致 JAX 转换失败。

**解决方案**: 在脚本开头注册：
```python
from jax.tree_util import register_pytree_node
from transformers import modeling_outputs

register_pytree_node(
    modeling_outputs.BaseModelOutputWithPastAndCrossAttentions,
    lambda obj: (obj.to_tuple(), type(obj)),
    lambda aux, children: aux(*children)
)
```

### 5. DecoderOutput.sample 提取

**问题**: VAE 返回 `DecoderOutput` 对象而非直接的 tensor。

**解决方案**: 在 `to_torch_recursive` 中处理：
```python
if hasattr(x, 'sample'):
    sample = to_torch_recursive(x.sample)
    return sample
```

### 6. torchax 全局状态管理

**问题**: `torchax.disable_globally()` 和 `torchax.enable_globally()` 交替调用可能导致状态不一致。

**解决方案**: 
- 在加载 PyTorch 模型前禁用，加载后立即重新启用
- 使用 `try/finally` 确保状态恢复

---

## 性能优化

### 1. Splash Attention exp2 优化

使用 TPU 原生 `exp2` 指令替代 `exp`，提升 ~10% 性能：

```python
# 标准实现
attention = softmax(Q @ K.T / sqrt(d))

# exp2 优化
_LOG2_E = 1.44269504
Q_scaled = Q * scale * _LOG2_E
attention = exp2(Q_scaled @ K.T)  # 使用 exp2 替代 exp
```

### 2. K Smoothing

减去 Key 的均值提高数值稳定性，避免溢出：

```python
key_mean = jnp.mean(key, axis=2, keepdims=True)
key = key - key_mean
```

### 3. 块大小调优

根据序列长度选择最优块大小：

| 场景 | BQSIZE | BKVSIZE | BKVCOMPUTESIZE |
|------|--------|---------|----------------|
| 720P (self-attn) | 3328 | 2816 | 256 |
| 480P (self-attn) | 2048 | 2048 | 256 |
| Cross-attn | 1024 | 512 | 256 |

### 4. 内存优化

- **阶段2**: 删除 VAE 和 Text Encoder 释放内存
- **阶段3**: 使用 VAE Cache 优化解码
- **权重**: 使用 bfloat16 减少 50% 内存

### 5. 编译缓存

启用 JAX 持久化编译缓存加速后续运行：

```python
jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
```

---

## 技术细节

### Mesh 配置

默认使用 `(dp=2, sp=1, tp=4)` 配置 8 个 TPU chips：

```python
mesh = Mesh(devices, ('dp', 'sp', 'tp'))
# dp: Data Parallel (batch sharding)
# sp: Sequence Parallel (未使用)
# tp: Tensor Parallel (weight sharding)
```

### 权重分片策略

**Transformer (FSDP 模式)**:
```python
{
    'attn1.to_q.weight': (None, ('tp', 'sp')),  # 列并行
    'attn1.to_out.0.weight': (('tp', 'sp'), None),  # 行并行
    'ffn.net.0.proj.weight': (None, ('tp', 'sp')),
    'ffn.net.2.weight': (('tp', 'sp'), None),
}
```

**VAE**:
```python
{
    'conv_out': ('tp', 'dp', 'sp'),
    'conv_in': ('tp', 'dp', 'sp'),
}
```

### SafeTensors 格式

- 安全的张量存储格式，避免 pickle 安全问题
- 支持内存映射，加载速度快
- 自动处理设备放置

---

## 典型用例

### 1. 快速测试（少步数）

```bash
python stage1_text_encoder.py --prompt "A robot dancing"
python stage2_transformer.py --num_inference_steps 3
python stage3_vae_decoder.py
```

### 2. 高质量生成

```bash
python stage1_text_encoder.py \
    --prompt "A detailed cinematic scene..." \
    --negative_prompt "blur, noise, artifacts"
python stage2_transformer.py --num_inference_steps 100
python stage3_vae_decoder.py
```

### 3. 批量生成（复用 embeddings）

```bash
# 生成一次 embeddings
python stage1_text_encoder.py --prompt "A cat" --output_dir ./cat

# 用不同 seed 多次生成
for seed in 1 2 3 4 5; do
    python stage2_transformer.py --input_dir ./cat --seed $seed
    python stage3_vae_decoder.py --input_dir ./cat --output_video ./cat/video_$seed.mp4
done
```

### 4. 对比不同步数效果

```bash
python stage1_text_encoder.py

for steps in 10 25 50 100; do
    python stage2_transformer.py --num_inference_steps $steps
    python stage3_vae_decoder.py --output_video output_${steps}steps.mp4
done
```

---

## 性能基准

| 配置 | 阶段1 | 阶段2 (50步) | 阶段3 | 总计 |
|------|-------|--------------|-------|------|
| TPU v4-8, 720P | ~2s | ~285s | ~11s | ~5min |
| TPU v4-8, 480P | ~2s | ~120s | ~5s | ~2min |

*注: 首次运行包含 JIT 编译时间，后续运行更快*

---

## 贡献指南

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

---

## 许可证

本项目遵循 Apache 2.0 许可证。

---

## 致谢

- [Wan-AI](https://github.com/Wan-AI) - Wan 2.1 模型
- [Google JAX Team](https://github.com/google/jax) - JAX 框架
- [Hugging Face](https://huggingface.co) - Diffusers 库
- [MaxDiffusion](https://github.com/google/maxdiffusion) - Flax VAE 实现