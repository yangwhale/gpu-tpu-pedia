# HunyuanVideo-1.5 TPU 移植指南

本文档详细讲解如何将 HunyuanVideo-1.5 从 GPU 版本 (`generate_diffusers.py`) 改造成 TPU 版本 (`generate_diffusers_flax.py`)，使用 torchax + Splash Attention 在 Google TPU 上运行。

## 目录

1. [环境准备](#环境准备)
2. [架构概述](#架构概述)
3. [改造步骤详解](#改造步骤详解)
4. [关键问题与解决方案](#关键问题与解决方案)
5. [测试与验证](#测试与验证)
6. [性能数据](#性能数据)

---

## 环境准备

### 1. 安装 diffusers-tpu 库

TPU 版本需要使用修改过的 diffusers 库，包含对 Transformer 模型的 TPU 兼容修改：

```bash
# 克隆修改版 diffusers
git clone https://github.com/yangwhale/diffusers-tpu.git
cd diffusers-tpu

# 安装为可编辑模式
pip install -e .
```

### 2. 安装 torchax

torchax 是 PyTorch 到 JAX 的桥接库，使 PyTorch 代码可以在 TPU 上运行：

```bash
pip install torchax
```

### 3. 安装 JAX（TPU 版本）

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### 4. 其他依赖

```bash
pip install transformers accelerate
```

---

## 架构概述

### GPU 版本 (`generate_diffusers.py`)

```
GPU Pipeline:
┌──────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌───────┐
│ Text Encoder │ -> │   Transformer   │ -> │       VAE       │ -> │ Video │
│    (GPU)     │    │ (Flash Attn)    │    │    (Decoder)    │    │       │
└──────────────┘    └─────────────────┘    └─────────────────┘    └───────┘
```

### TPU 版本 (`generate_diffusers_flax.py`)

```
TPU Pipeline:
┌──────────────┐    ┌─────────────────┐    ┌───────────────────┐    ┌───────┐
│ Text Encoder │ -> │   Transformer   │ -> │      VAE          │ -> │ Video │
│    (CPU)     │    │ (Splash Attn)   │    │  (SDPA + Tiling)  │    │       │
│  预计算      │    │   torchax JIT   │    │      torchax      │    │       │
└──────────────┘    └─────────────────┘    └───────────────────┘    └───────┘
```

**关键差异：**
- Text Encoder 保持在 CPU 上运行（预计算 embeddings）
- Transformer 使用 Splash Attention 替代 Flash Attention
- VAE 使用 SDPA 参考实现 + Tiling
- 所有权重需要分片到多个 TPU 核心

---

## 改造步骤详解

### 步骤 1: 导入必要的库

**GPU 版本：**
```python
import torch
from diffusers import HunyuanVideo15Pipeline
```

**TPU 版本：**
```python
import jax
import jax.numpy as jnp
import torch
import torchax
from torchax.ops import ops_registry
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
from jax.experimental.pallas.ops.tpu import splash_attention
from jax.experimental.shard_map import shard_map
from jax.experimental import mesh_utils
```

### 步骤 2: 实现 Splash Attention

TPU 不支持 Flash Attention，需要实现 Splash Attention：

```python
def _tpu_splash_attention(query, key, value, mesh, scale=None, is_causal=False, window_size=None):
    """TPU Splash Attention 实现"""
    num_heads = query.shape[1]

    def _attention_on_slices(q, k, v):
        scale_factor = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
        q = q * scale_factor

        def kernel_3d(q_3d, k_3d, v_3d):
            # 填充到块大小的倍数
            q_3d_padded, q_orig_len = pad_to_multiple(q_3d, BQSIZE, axis=1)
            k_3d_padded, k_orig_len = pad_to_multiple(k_3d, BKVSIZE, axis=1)
            v_3d_padded, v_orig_len = pad_to_multiple(v_3d, BKVSIZE, axis=1)

            # 创建 Splash Attention kernel
            mask = splash_attention.MultiHeadMask(
                [splash_attention.FullMask((padded_q_seq_len, padded_kv_seq_len)) 
                 for _ in range(num_heads_on_device)]
            )
            block_sizes = splash_attention.BlockSizes(
                block_q=min(BQSIZE, padded_q_seq_len),
                block_kv=min(BKVSIZE, padded_kv_seq_len),
                block_kv_compute=min(BKVCOMPUTESIZE, padded_kv_seq_len),
            )
            splash_kernel = splash_attention.make_splash_mha(
                mask=mask, block_sizes=block_sizes, head_shards=1, q_seq_shards=1
            )
            out = splash_kernel(q_3d_padded, k_3d_padded, v_3d_padded)
            return out[:, :q_orig_len, :]

        vmapped_kernel = jax.vmap(kernel_3d, in_axes=(0, 0, 0), out_axes=0)
        return vmapped_kernel(q, k, v)

    # 使用 shard_map 在设备间分片执行
    sharded_fn = shard_map(
        _attention_on_slices,
        mesh=mesh,
        in_specs=(q_partition_spec, kv_partition_spec, kv_partition_spec),
        out_specs=q_partition_spec,
        check_rep=False,
    )
    return sharded_fn(query, key, value)
```

### 步骤 3: 注册自定义 SDPA

使用 torchax 的 ops_registry 替换 PyTorch 的 SDPA：

```python
def setup_pipeline_for_jax(pipe, model_id):
    # 创建 JAX mesh
    mesh_devices = mesh_utils.create_device_mesh((tp_dim, dp_dim, sp_dim))
    mesh = Mesh(mesh_devices, ('tp', 'dp', 'sp'))
    
    # 创建 torchax 环境
    env = torchax.default_env()
    env._mesh = mesh
    env.config.use_tpu_splash_attention = True
    
    # 保存原始 SDPA 实现
    global _ORIGINAL_SDPA
    _ORIGINAL_SDPA = torch.nn.functional.scaled_dot_product_attention
    
    # 注册自定义 SDPA
    custom_attention = functools.partial(
        scaled_dot_product_attention,
        env=env,
        window_size=WINDOW_SIZE
    )
    
    op_to_override = torch.nn.functional.scaled_dot_product_attention
    env._ops[op_to_override] = ops_registry.Operator(
        op_to_override,
        custom_attention,
        is_jax_function=False,
        is_user_defined=True,
        needs_env=False,
        is_view_op=False,
    )
```

### 步骤 4: 权重分片

为 Tensor Parallel 配置权重分片规则：

```python
# Megatron Column-Row 风格分片
transformer_shardings_tp = {
    # Column Parallel: Q/K/V 在 out_features 分片
    r'.*\.img_attn_q\.weight$': (('tp', 'sp'), None),
    r'.*\.img_attn_k\.weight$': (('tp', 'sp'), None),
    r'.*\.img_attn_v\.weight$': (('tp', 'sp'), None),
    # Row Parallel: Proj 在 in_features 分片
    r'.*\.img_attn_proj\.weight$': (None, ('tp', 'sp')),
    # MLP Column Parallel
    r'.*\.img_mlp\.fc1\.weight$': (('tp', 'sp'), None),
    # MLP Row Parallel
    r'.*\.img_mlp\.fc2\.weight$': (None, ('tp', 'sp')),
}
```

### 步骤 5: 预计算 Prompt Embeddings

Text Encoder 必须在 SDPA override 之前运行（在 CPU 上）：

```python
def precompute_all_prompt_embeds(pipe, prompt, negative_prompt="", device='cpu'):
    """在 JAX 配置之前预计算所有 prompt embeddings"""
    # 计算正面 prompt embeddings
    prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2 = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_videos_per_prompt=1,
    )
    
    # 计算负面 prompt embeddings
    neg_prompt_embeds, neg_prompt_embeds_mask, neg_prompt_embeds_2, neg_prompt_embeds_mask_2 = pipe.encode_prompt(
        prompt=negative_prompt if negative_prompt else "",
        device=device,
        num_videos_per_prompt=1,
    )
    
    return {
        'prompt_embeds': prompt_embeds,
        'prompt_embeds_mask': prompt_embeds_mask,
        'prompt_embeds_2': prompt_embeds_2,
        'prompt_embeds_mask_2': prompt_embeds_mask_2,
        'negative_prompt_embeds': neg_prompt_embeds,
        'negative_prompt_embeds_mask': neg_prompt_embeds_mask,
        'negative_prompt_embeds_2': neg_prompt_embeds_2,
        'negative_prompt_embeds_mask_2': neg_prompt_embeds_mask_2,
    }
```

### 步骤 6: 编译 Transformer

使用 torchax.compile 对 Transformer 进行 JIT 编译：

```python
# 编译 Transformer（is_t2v 必须是静态参数）
pipe.transformer = torchax.compile(
    pipe.transformer,
    torchax.CompileOptions(
        jax_jit_kwargs={'static_argnames': ('return_dict', 'is_t2v')}
    )
)
```

---

## 关键问题与解决方案

### 问题 1: Flash Attention 不可用

**症状：** `ImportError: No module named 'flash_attn'`

**原因：** TPU 不支持 Flash Attention CUDA 库

**解决方案：** 修改 `transformer_hunyuan_video15.py`，将 attention backend 改为 NATIVE：

```python
# diffusers-tpu/src/diffusers/models/transformers/transformer_hunyuan_video15.py
class HunyuanVideo15AttnProcessor2_0:
    # 原始：_attention_backend = AttentionBackendName.FLASH
    _attention_backend = AttentionBackendName.NATIVE  # TPU 兼容
```

### 问题 2: Text Encoder 与 XLA Tensor 混合

**症状：** `TypeError: expected Tensor as element 0 in argument 0, but got XLA tensor`

**原因：** Text Encoder 内部创建普通 torch.Tensor（position embeddings），与 XLA tensor 混合

**解决方案：** 实现 XLA tensor 检测和自动回退：

```python
def _is_xla_tensor(tensor):
    """检测 tensor 是否是 XLA/torchax tensor"""
    if tensor is None:
        return False
    if hasattr(tensor, '_elem'):
        return True
    if hasattr(tensor, 'device'):
        device_str = str(tensor.device)
        if 'jax' in device_str or 'xla' in device_str:
            return True
    return False

def scaled_dot_product_attention(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa, env, window_size):
    # 非 XLA tensor 使用原始 SDPA
    if not _is_xla_tensor(query):
        return _ORIGINAL_SDPA(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)
    # XLA tensor 使用 Splash Attention
    pass
```

### 问题 3: NaN 值从 Padding Tokens 产生

**症状：** 输出包含大量 NaN 值

**原因：** Attention mask 中 padding tokens 导致 softmax 输入全为 -inf，产生 NaN

**解决方案：** 在 softmax 后使用 masked_fill 处理 NaN：

```python
def _sdpa_reference(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa):
    # 计算 attention weights
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    
    # 应用 mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_weight = attn_weight.masked_fill(attn_mask.logical_not(), float("-inf"))
    
    # 关键修复：NaN 处理
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = attn_weight.masked_fill(torch.isnan(attn_weight), 0.0)
    
    return attn_weight @ value
```

### 问题 4: 动态布尔索引不兼容 JAX JIT

**症状：** `ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected`

**原因：** 原始代码使用 `is_t2v = torch.all(image_embeds == 0)` 动态计算，JAX JIT 需要静态值

**解决方案：** 将 `is_t2v` 改为静态参数：

```python
# transformer_hunyuan_video15.py
def forward(
    self,
    hidden_states,
    timestep,
    encoder_hidden_states,
    encoder_attention_mask,
    encoder_hidden_states_2=None,
    encoder_attention_mask_2=None,
    image_embeds=None,
    attention_kwargs=None,
    return_dict=True,
    is_t2v=True,  # 静态参数
):
    # 使用静态参数而非动态计算
    if is_t2v:
        encoder_hidden_states_3 = encoder_hidden_states_3 * 0.0
```

### 问题 5: VAE 内存溢出

**症状：** `RESOURCE_EXHAUSTED: Out of memory`

**原因：** VAE decoder 一次处理整个视频帧，内存占用过大

**解决方案：** 默认启用 VAE Tiling：

```python
# 默认启用 VAE Tiling
if not args.disable_vae_tiling:
    print("启用 VAE Tiling（默认开启以节省 VMEM）...")
    pipe.vae.enable_tiling()
```

---

## 测试与验证

### 快速测试（5 帧 5 步）

```bash
python generate_diffusers_flax.py \
    --prompt "A cat walking on the grass" \
    --num_frames 5 \
    --num_inference_steps 5 \
    --num_iterations 1 \
    --output_path test.mp4
```

预期输出：
```
✓ 视频已保存!
=== 性能统计 ===
总迭代次数: 1
第一次运行（含编译）: 约 500-600 秒
```

### 完整测试（720P 50 步）

```bash
python generate_diffusers_flax.py \
    --prompt 'A girl holding a paper with words "Hello, world!"' \
    --num_frames 121 \
    --num_inference_steps 50 \
    --height 720 \
    --width 1280 \
    --num_iterations 2 \
    --output_path output_720p.mp4
```

预期输出：
```
迭代 0:
100%|██████████| 50/50 [16:49<00:00, 20.18s/it]
  完成时间: 1009.70 秒 (包含 JIT 编译)

迭代 1:
100%|██████████| 50/50 [XX:XX<00:00, XX.XXs/it]
  完成时间: 约 XXX 秒

=== 性能统计 ===
第一次运行（含编译）: 1009.70 秒
后续运行平均时间: 约 XXX 秒
```

---

## 性能数据

| 配置 | 帧数 | 步数 | 分辨率 | 第一次运行 | 后续运行 |
|------|------|------|--------|------------|----------|
| v6e-8 | 5 | 5 | 默认 | 约 578 秒 | - |
| v6e-8 | 121 | 50 | 720P | 约 1010 秒 | TBD |

---

## 文件结构

```
HunyuanVideo-1.5/
├── generate_diffusers.py      # GPU 版本（原始）
├── generate_diffusers_flax.py # TPU 版本（改造后）
├── dit_flax.py                # DiT 性能测试（TPU）
├── vae_decode_flax.py         # VAE Flax 版本测试
└── README.md                  # 本文档
```

---

## VAE Flax 改造详解

本节详细讲解如何将 HunyuanVideo-1.5 的 VAE 从 PyTorch 版本改造成纯 Flax/JAX 版本。

### 改造背景

HunyuanVideo-1.5 使用的 VAE 与 CogVideoX 的 VAE 有显著差异：

| 特性 | CogVideoX VAE | HunyuanVideo-1.5 VAE |
|------|---------------|----------------------|
| 时间上采样方式 | `jax.image.resize` 插值 | DCAE 通道重排 |
| 时间压缩比 | 4x | 4x |
| 空间压缩比 | 8x | 16x |
| 帧数公式 | `4T+1` | `4T-3` |
| Latent 通道数 | 16 | 32 |

### DCAE 时间上采样机制

DCAE (Denoising Channel-wise Autoencoder) 使用通道重排而非插值进行时间上采样：

```
时间上采样公式: L latent 帧 → 4L-3 video 帧

示例：16 latent 帧 → 4×16-3 = 61 video 帧
```

**关键原理：**
- 第一帧：只进行空间上采样（1→1 帧）
- 后续帧：通道重排为时间+空间（1→2 帧）

```python
def _dcae_upsample_rearrange(self, x, r1, r2, r3):
    """DCAE 风格的通道重排上采样
    
    Args:
        x: 输入张量 [B, C*r1*r2*r3, T, H, W]
        r1: 时间上采样因子
        r2: 高度上采样因子
        r3: 宽度上采样因子
    
    Returns:
        输出张量 [B, C, T*r1, H*r2, W*r3]
    """
    b, c, t, h, w = x.shape
    new_c = c // (r1 * r2 * r3)
    # 重排通道维度到时间和空间维度
    x = jnp.reshape(x, (b, new_c, r1, r2, r3, t, h, w))
    x = jnp.transpose(x, (0, 1, 5, 2, 6, 3, 7, 4))  # [B, C, T, r1, H, r2, W, r3]
    x = jnp.reshape(x, (b, new_c, t * r1, h * r2, w * r3))
    return x
```

### is_first_frame 参数

为支持逐帧解码，需要区分首帧和后续帧的不同上采样行为：

```python
class FlaxHunyuanVideo15Upsample(nnx.Module):
    def __call__(self, x, feat_cache=None, feat_idx=None, is_first_frame=None):
        if is_first_frame is not None:
            if is_first_frame:
                # 首帧：只空间上采样，时间维度不变
                h = self._dcae_upsample_rearrange(h, r1=1, r2=2, r3=2)
                # 截取前一半（因为通道数是 4x，首帧只用空间部分）
                h = h[:, :, :, :, :h.shape[-1] // 2]
            else:
                # 后续帧：时间+空间上采样，1帧→2帧
                h = self._dcae_upsample_rearrange(h, r1=r1, r2=2, r3=2)
        else:
            # 批量解码模式
            h = self._dcae_upsample_rearrange(h, r1=r1, r2=2, r3=2)
```

### feat_cache 因果卷积缓存

CausalConv3d 需要缓存前一帧的特征以保证因果性：

```python
def _call_with_feat_cache(self, hidden_states, feat_cache, feat_idx, conv_fn):
    """执行带缓存的因果卷积"""
    kernel_t = self.kernel_size[0]
    
    if kernel_t == 1:
        # 无时间卷积，直接计算
        return conv_fn(hidden_states), feat_cache, feat_idx
    
    # 从缓存获取前一帧
    cache_key = feat_idx[0]
    if cache_key in feat_cache:
        cached = feat_cache[cache_key]
        # 拼接缓存和当前帧
        hidden_states = jnp.concatenate([cached, hidden_states], axis=2)
    else:
        # 首帧：复制自身作为填充
        hidden_states = jnp.concatenate(
            [hidden_states] * kernel_t, axis=2
        )
    
    # 更新缓存（存储最后 kernel_t-1 帧）
    feat_cache[cache_key] = hidden_states[:, :, -(kernel_t - 1):, :, :]
    feat_idx[0] += 1
    
    return conv_fn(hidden_states), feat_cache, feat_idx
```

**重要：** 缓存必须存储空间填充后的张量，否则会导致形状不匹配：

```python
# 正确：先空间填充，再存入缓存
hidden_states = jnp.pad(hidden_states, spatial_padding)  # 空间填充
feat_cache[cache_key] = hidden_states[:, :, -(kernel_t - 1):, :, :]  # 存储填充后的

# 错误：存储填充前的张量
feat_cache[cache_key] = hidden_states[:, :, -(kernel_t - 1):, :, :]  # ❌
hidden_states = jnp.pad(hidden_states, spatial_padding)  # 太晚了
```

### Flax NNX 0.12.0 兼容性

Flax NNX 0.12.0 要求使用 `nnx.List` 而非 Python list 包装模块：

```python
class FlaxHunyuanVideo15MidBlock(nnx.Module):
    def __init__(self, ...):
        # 错误：Python list
        # self.attentions = [attention1, attention2]  # ❌
        
        # 正确：nnx.List
        self.attentions = nnx.List([attention1, attention2])  # ✓
        
        # 或者单个 attention 时直接赋值
        if num_attentions == 1:
            self.attentions = attention1
        else:
            self.attentions = nnx.List([attention1, attention2])
```

### 完整改造文件

改造后的文件位置：
```
diffusers-tpu/src/diffusers/models/autoencoders/autoencoder_kl_hunyuanvideo15_flax.py
```

主要类结构：
```python
# 基础层
class FlaxHunyuanVideo15CausalConv3d(nnx.Module)     # 因果 3D 卷积
class FlaxHunyuanVideo15Upsample(nnx.Module)         # DCAE 上采样
class FlaxHunyuanVideo15Downsample(nnx.Module)       # DCAE 下采样

# 中间层
class FlaxHunyuanVideo15ResnetBlock3D(nnx.Module)    # 3D 残差块
class FlaxHunyuanVideo15Attention(nnx.Module)        # 3D 注意力
class FlaxHunyuanVideo15MidBlock(nnx.Module)         # 中间块

# 编解码器
class FlaxHunyuanVideo15Encoder3D(nnx.Module)        # 3D 编码器
class FlaxHunyuanVideo15Decoder3D(nnx.Module)        # 3D 解码器

# 主模型
class FlaxAutoencoderKLHunyuanVideo15(nnx.Module)    # VAE 主类
```

### 测试验证

测试文件：
```bash
cd gpu-tpu-pedia/tpu/HunyuanVideo-1.5
python vae_decode_flax.py
```

测试结果：
```
输入 latents: (1, 16, 34, 60, 32)
输出 video:   (1, 61, 544, 960, 3)

时间上采样验证:
  16 latent 帧 → 61 video 帧
  公式验证: 4×16-3 = 61 ✓

空间上采样验证:
  34 → 544 (×16) ✓
  60 → 960 (×16) ✓

性能: ~37 秒/batch (8 TPU v6e)
```

### 与 CogVideoX VAE 对比

| 改造步骤 | CogVideoX | HunyuanVideo-1.5 |
|----------|-----------|------------------|
| 时间上采样 | `jax.image.resize` | DCAE 通道重排 |
| 逐帧解码 | 直接插值到 2 帧 | 需要 `is_first_frame` 参数 |
| 缓存机制 | 相同 | 相同 |
| Attention | 无 | 有（需要 nnx.List） |
| 空间压缩 | 8x | 16x |

---

## diffusers-tpu 库修改说明

需要修改 `diffusers-tpu/src/diffusers/models/transformers/transformer_hunyuan_video15.py`：

1. 第 48 行：`_attention_backend = AttentionBackendName.NATIVE`
2. 第 623 行：`is_t2v: bool = True` 作为静态参数
3. 第 697-708 行：简化 tensor concatenation 逻辑，避免动态布尔索引

---

## 参考资料

- [torchax 文档](https://github.com/pytorch/xla)
- [JAX Splash Attention](https://github.com/google/jax/tree/main/jax/experimental/pallas/ops/tpu)
- [diffusers-tpu](https://github.com/yangwhale/diffusers-tpu)
- [HunyuanVideo-1.5](https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v)