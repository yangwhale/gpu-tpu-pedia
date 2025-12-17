# Wan 2.1 TPU 视频生成

本项目提供 Wan 2.1 Text-to-Video 模型在 TPU 上的高效推理实现，支持 JAX/Flax 和 Splash Attention 优化。

## 目录

- [项目概述](#项目概述)
- [环境要求](#环境要求)
- [快速安装](#快速安装)
- [使用指南](#使用指南)
- [参数详解](#参数详解)
- [技术细节](#技术细节)
- [性能优化](#性能优化)
- [已知问题与解决方案](#已知问题与解决方案)
- [中间文件说明](#中间文件说明)
- [性能基准](#性能基准)
- [致谢](#致谢)

---

## 项目概述

### 目录结构

```
Wan2.1/
├── README.md                        # 本文档
├── generate_flax.py                 # 单步推理脚本（一次性执行全流程）
├── custom_splash_attention.py       # 自定义 Splash Attention 实现
└── generate_diffusers_flax_staged/  # 三阶段推理目录
    ├── utils.py                     # 共享工具函数和配置
    ├── stage1_text_encoder.py       # 阶段1：Text Encoding
    ├── stage2_transformer.py        # 阶段2：Transformer Denoising
    ├── stage3_vae_decoder.py        # 阶段3：VAE Decoding
    └── stage_outputs/               # 中间输出目录
        ├── stage1_embeddings.safetensors
        ├── stage2_latents.safetensors
        ├── generation_config.json
        └── output_video.mp4
```

### 两种推理模式

| 特性 | 单步推理 (`generate_flax.py`) | 三阶段推理 |
|------|------------------------------|-----------|
| 执行方式 | 一次性完成全部流程 | 可分步执行，支持暂停/恢复 |
| 调试便利性 | 难以在中途检查状态 | 可在任意阶段检查中间结果 |
| 内存占用 | 峰值较高（所有组件同时加载） | 各阶段独立加载/释放 |
| 结果复用 | 每次重新计算 | 可复用 embeddings 和 latents |
| 适用场景 | 快速生成、基准测试 | 开发调试、批量生成、实验对比 |

### 依赖仓库

| 仓库 | 用途 | 地址 |
|------|------|------|
| **diffusers-tpu** | Pipeline、Transformer、Scheduler、VAE (Flax 版本) | [github.com/yangwhale/diffusers-tpu](https://github.com/yangwhale/diffusers-tpu) |
| **torchax** | PyTorch-JAX 桥接 | [PyPI: torchax](https://pypi.org/project/torchax/) |

---

## 环境要求

### 硬件

- **TPU**: v4-8 或 v6e-8（8 chips 最小配置）
- **内存**: 建议 64GB+ 系统内存
- **存储**: 约 50GB 用于模型权重缓存

### 软件

- Python 3.10+
- JAX 0.4.35+ (TPU 版本)
- PyTorch 2.5+ (CPU 版本)
- torchax 0.0.11+

---

## 快速安装

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

# 5. 克隆并安装修改版 diffusers（包含 Flax VAE）
git clone https://github.com/yangwhale/diffusers-tpu.git
cd diffusers-tpu && pip install -e . && cd ..

# 6. 克隆项目代码
git clone https://github.com/yangwhale/gpu-tpu-pedia.git
cd gpu-tpu-pedia/tpu/Wan2.1
```

### 验证安装

```bash
# 检查 JAX 设备
python -c "import jax; print(f'JAX devices: {jax.devices()}')"
# 预期: [TpuDevice(id=0, ...), TpuDevice(id=1, ...), ...]

# 检查 torchax 版本
python -c "import torchax; print(f'torchax version: {torchax.__version__}')"
# 预期: torchax version: 0.0.11

# 检查 Pipeline 导入
python -c "from diffusers.pipelines.wan.pipeline_wan_flax import WanPipeline; print('Pipeline OK')"
```

---

## 使用指南

### 单步推理 (`generate_flax.py`)

一次性执行完整的 Text-to-Video 生成流程：

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

```bash
cd gpu-tpu-pedia/tpu/Wan2.1/generate_diffusers_flax_staged

# 阶段1：Text Encoding（仅编码 prompt）
python stage1_text_encoder.py \
    --prompt "A cat playing piano in a jazz club" \
    --negative_prompt "blurry, low quality, static"

# 阶段2：Transformer Denoising（设置视频参数）
python stage2_transformer.py \
    --height 480 --width 848 --frames 81 \
    --num_inference_steps 50

# 阶段3：VAE Decoding
python stage3_vae_decoder.py
```

**输出**: `stage_outputs/output_video.mp4`

**注意**: Stage1 仅负责 prompt 编码，视频参数（height、width、frames）在 Stage2 中设置。

---

## 参数详解

### 视频生成参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--width` | 1280 | 视频宽度（像素） |
| `--height` | 720 | 视频高度（像素） |
| `--frames` | 81 | 视频帧数（需为 4n+1） |
| `--fps` | 16 | 视频帧率 |
| `--num_inference_steps` | 50 | 去噪步数 |
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
| `--use_dp` | True | 启用数据并行 |
| `--use_fsdp` | True | 启用 FSDP 模式 |
| `--sp_num` | 1 | Sequence Parallel 分片数 |

---

## 技术细节

### Mesh 配置

使用 2D Mesh `(dp=2, tp=4)` 配置 8 个 TPU chips：

```python
mesh = Mesh(devices, ('dp', 'tp'))
# dp: Data Parallel (batch sharding)
# tp: Tensor Parallel (weight sharding)
```

### Splash Attention 分片策略

根据 attention 类型和 batch 维度选择不同的分片：

```python
# Self Attention (长 KV 序列 > 20000)
if batch_size > 1:
    dp_mesh_key = "dp"
    remain_mesh_key = ("tp",)
else:
    dp_mesh_key = None
    remain_mesh_key = ("dp", "tp")

# Context Parallel (长序列)
q_partition_spec = P(dp_mesh_key, remain_mesh_key, None, None)
kv_partition_spec = P(dp_mesh_key, remain_mesh_key, None, None)

# Sequence Parallel (短序列)
q_partition_spec = P(dp_mesh_key, None, remain_mesh_key, None)
kv_partition_spec = P(dp_mesh_key, None, None, None)
```

### 权重分片策略

**Transformer**:
```python
{
    'attn1.to_q.weight': ('tp',),        # 列并行
    'attn1.to_out.*.weight': (None, 'tp'),  # 行并行
    'ffn.net.*.proj.weight': ('tp',),
    'ffn.net.*.weight': (None, 'tp'),
}
```

---

## 性能优化

### Custom Attention (exp2 优化)

TPU 的 VPU 有专门的 `exp2` 硬件指令。Custom Attention 使用 `exp2` 替代 `exp`，提升 ~10% 性能：

```python
# 标准实现
attention = softmax(Q @ K.T / sqrt(d))

# exp2 优化
_LOG2_E = 1.44269504
Q_scaled = Q * scale * _LOG2_E
attention = exp2(Q_scaled @ K.T)  # 直接使用 TPU 原生指令
```

### torchax 0.0.11 手动分片

torchax 0.0.11 移除了自动 sharding 机制，需要手动对输入张量应用 sharding：

```python
from jax.sharding import NamedSharding, PartitionSpec as P

def apply_input_sharding(tensor, env, use_dp=False):
    mesh = getattr(env, '_mesh', None) or getattr(env.param, 'mesh', None)
    if mesh is None:
        return tensor
    
    ndim = tensor.ndim if not hasattr(tensor, '_elem') else tensor._elem.ndim
    
    if ndim == 5:  # latents
        pspec = P('dp', None, None, None, None) if use_dp else P()
    elif ndim == 3:  # prompt_embeds
        pspec = P('dp', None, None) if use_dp else P()
    else:
        pspec = P()
    
    sharding = NamedSharding(mesh, pspec)
    if hasattr(tensor, 'apply_jax_'):
        tensor.apply_jax_(jax.device_put, sharding)
    return tensor
```

**性能对比**:

| torchax 版本 | Sharding 方式 | 每步时间 |
|-------------|--------------|---------|
| 0.0.4 (hanq_wan_changes) | 自动 | ~5.3s |
| 0.0.11 (无优化) | 无 | ~54s |
| **0.0.11 (手动 sharding)** | **apply_input_sharding()** | **~5.9s** |

### K Smoothing

减去 Key 的均值提高数值稳定性：

```python
key_mean = jnp.mean(key, axis=2, keepdims=True)
key = key - key_mean
```

### 编译缓存

启用 JAX 持久化编译缓存加速后续运行：

```python
jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
```

---

## 已知问题与解决方案

### 1. Text Encoder None 检查

**问题**: 三阶段模式下 `text_encoder=None` 导致 `.device` 访问失败。

**解决方案**: 已在 `pipeline_wan_flax.py` 中添加 None 检查。

### 2. bfloat16 保存/加载

**问题**: SafeTensors 不直接支持 bfloat16。

**解决方案**: `utils.py` 中自动处理类型转换，保存时记录原始类型。

### 3. PyTree 注册

**问题**: `BaseModelOutputWithPastAndCrossAttentions` 未注册为 PyTree。

**解决方案**: 脚本开头注册：
```python
from jax.tree_util import register_pytree_node
from transformers import modeling_outputs

register_pytree_node(
    modeling_outputs.BaseModelOutputWithPastAndCrossAttentions,
    lambda obj: (obj.to_tuple(), type(obj)),
    lambda aux, children: aux(*children)
)
```

### 4. DecoderOutput.sample 提取

**问题**: VAE 返回 `DecoderOutput` 对象而非直接的 tensor。

**解决方案**: 在 `to_torch_recursive` 中处理 `.sample` 属性。

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

维度解释:
- Batch: 1
- Channels: 16（Wan VAE 的 z_dim）
- Temporal: 21（81 帧 ÷ 4）
- Height: 90（720 ÷ 8）
- Width: 160（1280 ÷ 8）

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

## 性能基准

### 测试环境

- **硬件**: TPU v6e-8 (8 chips)
- **模型**: Wan 2.1 14B
- **分辨率**: 832×480 或 1280×720, 81 帧

### 优化效果

**720P (1280×720, 81 帧, 50 步):**

| 版本 | Warmup | Benchmark | 每步时间 |
|------|--------|-----------|---------|
| 优化前 (maxdiffusion VAE) | OOM | OOM | N/A |
| **优化后 (diffusers Flax VAE)** | **515.53s** | **229.29s** | **4.59s/step** |

**480P (832×480, 81 帧, 50 步):**

| 版本 | Warmup | Benchmark | 每步时间 | 加速比 |
|------|--------|-----------|---------|-------|
| 优化前 (maxdiffusion VAE) | 196.43s | 90.40s | 1.81s/step | - |
| **优化后 (diffusers Flax VAE)** | **347.87s** | **63.02s** | **1.26s/step** | **30.3%** |

*注:
- 优化后 Warmup 时间较长是因为 diffusers VAE 需要更多 JIT 编译，但实际推理速度大幅提升
- 720P 之前会 OOM，优化后可以正常运行*

### 优化措施

1. **使用 diffusers Flax VAE**: 替换 maxdiffusion VAE，减少依赖
2. **PyTree 注册**: 添加 `DecoderOutput`、`AutoencoderKLOutput`、`DiagonalGaussianDistribution`
3. **Conv2d Op 覆盖**: 使用 `torch_conv2d_jax` 确保正确的 JAX 执行
4. **Pipeline 加载顺序**: 在 `torchax.enable_globally()` 之前加载避免 safetensors 问题
5. **2D Mesh**: 简化 mesh 配置从 3D (dp, sp, tp) 到 2D (dp, tp)

### 三阶段推理性能（TPU v6e-8）

**480P (848×480, 81 帧, 50 步)**:

| 阶段 | 预热 (JIT) | 正式运行 | 说明 |
|------|-----------|---------|------|
| Stage1 (Text Encoder) | - | ~3s | TPU 运行，使用 torchax |
| Stage2 (Transformer) | ~93s | ~68.55s | ~1.37s/step |
| Stage3 (VAE Decoder) | ~4.45s | ~0.50s | 预热后速度提升 9x |
| **总计** | **~97s** | **~72s** | 预热 + 正式运行 ≈ 169s |

*注:
- Stage2 和 Stage3 都支持预热运行，预热会触发 JIT 编译
- 后续运行使用编译缓存，无需重复预热*

**720P (1280×720, 81 帧, 50 步)**:

| 阶段 | 预热 (JIT) | 正式运行 | 说明 |
|------|-----------|---------|------|
| Stage1 (Text Encoder) | - | ~3s | TPU 运行，使用 torchax |
| Stage2 (Transformer) | ~110s | ~230s | ~4.60s/step |
| Stage3 (VAE Decoder) | ~80s | ~1s | 预热后速度提升 80x |
| **总计** | **~190s** | **~234s** | 预热 + 正式运行 ≈ 424s (7分钟) |

*注:
- 720P 是默认分辨率
- 后续运行使用编译缓存，无需重复预热*

---

## 典型用例

### 快速测试

```bash
python stage1_text_encoder.py --prompt "A robot dancing"
python stage2_transformer.py --num_inference_steps 3
python stage3_vae_decoder.py
```

### 高质量生成

```bash
python stage1_text_encoder.py \
    --prompt "A detailed cinematic scene..." \
    --negative_prompt "blur, noise, artifacts"
python stage2_transformer.py --num_inference_steps 100
python stage3_vae_decoder.py
```

### 批量生成（复用 embeddings）

```bash
# 生成一次 embeddings
python stage1_text_encoder.py --prompt "A cat" --output_dir ./cat

# 用不同 seed 多次生成
for seed in 1 2 3 4 5; do
    python stage2_transformer.py --input_dir ./cat --seed $seed
    python stage3_vae_decoder.py --output_video ./cat/video_$seed.mp4
done
```

---

## 许可证

本项目遵循 Apache 2.0 许可证。

---

## 致谢

- [Wan-AI](https://github.com/Wan-AI) - Wan 2.1 模型
- [Google JAX Team](https://github.com/google/jax) - JAX 框架
- [Hugging Face](https://huggingface.co) - Diffusers 库