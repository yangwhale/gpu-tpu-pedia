# CogVideoX VAE PyTorch 到 Flax 迁移详解

> **文档目标**：详细讲解 CogVideoX VAE 从 PyTorch 实现到 JAX/Flax 实现的完整迁移过程，包括每个函数的对照讲解、设计思路和实现细节。

---

## 📚 相关文档

本文档是 CogVideoX VAE 迁移系列文档之一，建议配合以下文档阅读：

### 🎯 迁移指南系列
- **[PyTorch 到 JAX 迁移圣经](./PYTORCH_TO_JAX_MIGRATION_BIBLE_ZH.md)** - 完整的 PyTorch → JAX 迁移指南
- **[PyTorch 到 JAX 范式转换](./PYTORCH_TO_JAX_PARADIGM_SHIFT_ZH.md)** - 理解两个框架的思维方式差异
- **[PyTorch-JAX 实现对比](./PYTORCH_JAX_IMPLEMENTATION_COMPARISON.md)** - 常见模式的对比实现

### 🔧 技术专题
- **[JAX 内存调试指南](./JAX_MEMORY_DEBUGGING_GUIDE.md)** - TPU 内存问题的诊断和解决
- **[JAX 并行编程详解](./JAX_PARALLEL_PROGRAMMING_GUIDE.md)** - JAX 的并行化策略和最佳实践

### 📖 案例研究
- **[CogVideoX VAE OOM 修复案例](./CASE_STUDY_COGVIDEOX_VAE_OOM_FIX.md)** - 逐帧解码方案的详细设计
- **[BF16 优化尝试记录](./BF16_OPTIMIZATION_ATTEMPTS.md)** - 混合精度训练的实践经验
- **[CogVideoX VAE Flax 说明](./COGVIDEOX_VAE_FLAX_README.md)** - Flax 实现的快速上手指南

---

## 📑 目录

1. [整体架构概述](#1-整体架构概述)
2. [调用流程详解](#2-调用流程详解)
3. [基础组件迁移](#3-基础组件迁移)
4. [核心模块迁移](#4-核心模块迁移)
5. [主 VAE 类迁移](#5-主-vae-类迁移)
6. [高级功能迁移](#6-高级功能迁移)
7. [关键差异总结](#7-关键差异总结)

---

## 1. 整体架构概述

### 1.1 CogVideoX VAE 简介

CogVideoX VAE 是一个专为视频数据设计的变分自编码器（Variational Autoencoder），主要特点：

- **3D 卷积架构**：处理时空数据（视频）
- **因果卷积**：时间维度上的因果性（causal）填充
- **空间归一化**：解码器使用 Spatial Normalization
- **分块处理**：支持 tiling 和 frame batching 以节省内存
- **时序压缩**：将视频压缩到潜在空间，默认时序压缩比为 4

### 1.2 文件对应关系

```
PyTorch 版本:  autoencoder_kl_cogvideox.py (1423 行)
Flax 版本:     autoencoder_kl_cogvideox_flax.py (2433 行)
```

### 1.3 主要组件层次结构

```
FlaxAutoencoderKLCogVideoX (主 VAE 类)
├── FlaxCogVideoXEncoder3D (编码器)
│   ├── FlaxCogVideoXCausalConv3d (输入卷积)
│   ├── FlaxCogVideoXDownBlock3D × 4 (下采样块)
│   │   ├── FlaxCogVideoXResnetBlock3D × 3 (ResNet 块)
│   │   └── FlaxConv2d (空间下采样)
│   ├── FlaxCogVideoXMidBlock3D (中间块)
│   │   └── FlaxCogVideoXResnetBlock3D × 2
│   └── FlaxCogVideoXCausalConv3d (输出卷积)
│
└── FlaxCogVideoXDecoder3D (解码器)
    ├── FlaxCogVideoXCausalConv3d (输入卷积)
    ├── FlaxCogVideoXMidBlock3D (中间块)
    │   └── FlaxCogVideoXResnetBlock3D × 2 (带 SpatialNorm)
    ├── FlaxCogVideoXUpBlock3D × 4 (上采样块)
    │   ├── FlaxCogVideoXResnetBlock3D × 4 (带 SpatialNorm)
    │   └── FlaxConv2d (空间上采样)
    └── FlaxCogVideoXCausalConv3d (输出卷积)
```

### 1.4 数据格式约定

| 框架 | 数据格式 | 说明 |
|------|----------|------|
| PyTorch | `(B, C, T, H, W)` | 批次、通道、时间、高度、宽度（channel-first）|
| JAX/Flax | `(B, T, H, W, C)` | 批次、时间、高度、宽度、通道（channel-last）|

**关键点**：所有 PyTorch 的 channel-first 数据都需要转换为 JAX 的 channel-last 格式。

---

## 2. 调用流程详解

### 2.1 编码流程 (Encode)

#### PyTorch 版本调用栈

```python
# 入口：autoencoder_kl_cogvideox.py 第 1154-1179 行
AutoencoderKLCogVideoX.encode(x: Tensor) -> AutoencoderKLOutput
  └─> _encode(x: Tensor) -> Tensor
       ├─> 分帧处理循环 (frame_batch_size = 8)
       │    └─> CogVideoXEncoder3D.forward(x_intermediate, conv_cache)
       │         ├─> conv_in: CausalConv3d (3 -> 128 通道)
       │         ├─> down_blocks[0-3]: DownBlock3D
       │         │    ├─> resnets × 3: ResnetBlock3D
       │         │    └─> downsampler: Downsample3D (空间 2x 下采样)
       │         ├─> mid_block: MidBlock3D
       │         │    └─> resnets × 2: ResnetBlock3D
       │         └─> conv_out: CausalConv3d (512 -> 32 通道，输出 mean+logvar)
       │
       └─> quant_conv (可选): Conv3d (32 -> 32)
  
  └─> DiagonalGaussianDistribution(h)  # 分离 mean 和 logvar
```

**输入**：视频张量 `x: (B, C, T, H, W)` 例如 `(1, 3, 49, 480, 720)`
**输出**：潜在分布 `posterior: DiagonalGaussianDistribution`，包含 mean 和 logvar

#### Flax 版本调用栈

```python
# 入口：autoencoder_kl_cogvideox_flax.py 第 1936-1953 行
FlaxAutoencoderKLCogVideoX.encode(x: Array) -> Tuple[Array, Array]
  └─> _encode(x: Array) -> Array
       ├─> 分帧处理循环 (frame_batch_size = 8)
       │    └─> FlaxCogVideoXEncoder3D.__call__(x_intermediate, conv_cache)
       │         ├─> conv_in: FlaxCogVideoXCausalConv3d (3 -> 128 通道)
       │         ├─> down_blocks[0-3]: FlaxCogVideoXDownBlock3D
       │         │    ├─> resnets × 3: FlaxCogVideoXResnetBlock3D
       │         │    └─> downsampler: FlaxConv2d (空间 2x 下采样)
       │         ├─> mid_block: FlaxCogVideoXMidBlock3D
       │         │    └─> resnets × 2: FlaxCogVideoXResnetBlock3D
       │         └─> conv_out: FlaxCogVideoXCausalConv3d (512 -> 32 通道)
       │
       └─> quant_conv (可选): FlaxConv3d (32 -> 32)
  
  └─> jnp.split(h, 2, axis=-1)  # 分离 mean 和 logvar
```

**输入**：视频张量 `x: (B, T, H, W, C)` 例如 `(1, 49, 480, 720, 3)`
**输出**：`(mean, logvar)` 元组，shape 均为 `(B, T//4, H//8, W//8, 16)`

**关键差异**：
1. ✅ 数据格式：PyTorch `BCTHW` → Flax `BTHWC`
2. ✅ 输出格式：PyTorch 返回 `DiagonalGaussianDistribution` 对象 → Flax 直接返回 `(mean, logvar)` 元组
3. ✅ Frame batching：两者都支持，默认 8 帧一批

### 2.2 解码流程 (Decode)

#### PyTorch 版本调用栈

```python
# 入口：autoencoder_kl_cogvideox.py 第 1210-1232 行
AutoencoderKLCogVideoX.decode(z: Tensor) -> DecoderOutput
  └─> _decode(z: Tensor) -> Union[DecoderOutput, Tensor]
       ├─> 分帧处理循环 (frame_batch_size = 2)
       │    ├─> post_quant_conv (可选): Conv3d
       │    └─> CogVideoXDecoder3D.forward(z_intermediate, conv_cache)
       │         ├─> conv_in: CausalConv3d (16 -> 512 通道)
       │         ├─> mid_block: MidBlock3D (带 SpatialNorm)
       │         │    └─> resnets × 2: ResnetBlock3D (spatial_norm_dim=16)
       │         ├─> up_blocks[0-3]: UpBlock3D
       │         │    ├─> resnets × 4: ResnetBlock3D (带 SpatialNorm)
       │         │    └─> upsampler: Upsample3D (空间/时序上采样)
       │         └─> conv_out: CausalConv3d (128 -> 3 通道)
       │
       └─> torch.cat(dec, dim=2)  # 拼接时间维度
  
  └─> DecoderOutput(sample=dec)
```

**输入**：潜在张量 `z: (B, C, T, H, W)` 例如 `(1, 16, 13, 60, 90)`
**输出**：重建视频 `(B, 3, 49, 480, 720)` (时序上采样 4 倍)

#### Flax 版本调用栈

```python
# 入口：autoencoder_kl_cogvideox_flax.py 第 2070-2086 行
FlaxAutoencoderKLCogVideoX.decode(z: Array, zq: Array) -> Array
  └─> _decode(z: Array, zq: Array) -> Array
       ├─> FlaxCogVideoXCache(decoder) 创建缓存管理器
       ├─> post_quant_conv (可选): FlaxConv3d
       ├─> 逐帧解码循环 (每次 1 帧)
       │    ├─> 重置索引：feat_cache_manager._conv_idx = [0]
       │    ├─> 提取当前帧：z_frame = z[:, i:i+1, ...]
       │    └─> FlaxCogVideoXDecoder3D.__call__(z_frame, zq_frame, feat_cache, feat_idx)
       │         ├─> conv_in: FlaxCogVideoXCausalConv3d
       │         ├─> mid_block: FlaxCogVideoXMidBlock3D (带 SpatialNorm)
       │         ├─> up_blocks[0-3]: FlaxCogVideoXUpBlock3D
       │         │    ├─> resnets × 4: FlaxCogVideoXResnetBlock3D
       │         │    └─> upsampler: FlaxConv2d + jax.image.resize
       │         └─> conv_out: FlaxCogVideoXCausalConv3d
       │
       └─> jnp.concatenate(decoded_frames_list, axis=1)  # 拼接时间维度
```

**输入**：潜在张量 `z: (B, T, H, W, C)` 例如 `(1, 13, 60, 90, 16)`
**输出**：重建视频 `(B, 49, 480, 720, 3)` (时序上采样 4 倍)

**关键差异**：
1. ✅ 数据格式：PyTorch `BCTHW` → Flax `BTHWC`
2. ⚠️ **批处理大小**：PyTorch 每批 2 帧 → Flax 每批 **1 帧**（避免 OOM）
3. ✅ **缓存机制**：
   - PyTorch: 使用 `conv_cache` 字典存储每层的缓存
   - Flax: 使用 `FlaxCogVideoXCache` 类管理 `feat_cache` 列表和 `feat_idx` 索引
4. ✅ 输出格式：PyTorch 返回 `DecoderOutput` 对象 → Flax 直接返回数组

### 2.3 完整的前向传播流程

#### PyTorch 版本

```python
# autoencoder_kl_cogvideox.py 第 1407-1423 行
def forward(sample, sample_posterior=False, generator=None):
    # 1. 编码
    posterior = self.encode(sample).latent_dist
    
    # 2. 采样或取模式
    if sample_posterior:
        z = posterior.sample(generator=generator)
    else:
        z = posterior.mode()  # 等于 mean
    
    # 3. 解码
    dec = self.decode(z).sample
    
    return DecoderOutput(sample=dec)
```

#### Flax 版本

```python
# autoencoder_kl_cogvideox_flax.py 第 2232-2263 行
def __call__(x, sample_posterior=False, rng=None):
    # 1. 编码
    mean, logvar = self.encode(x, deterministic=True)
    
    # 2. 采样或取模式
    if sample_posterior:
        std = jnp.exp(0.5 * logvar)
        z = mean + std * jax.random.normal(rng, mean.shape)
    else:
        z = mean  # mode
    
    # 3. 解码 (z 同时作为潜在表示和空间条件)
    dec = self.decode(z, zq=z, deterministic=True)
    
    return dec
```

**关键点**：
- PyTorch 的 `DiagonalGaussianDistribution.mode()` 等价于 Flax 直接使用 `mean`
- PyTorch 的 `DiagonalGaussianDistribution.sample()` 等价于 Flax 的重参数化技巧
- Flax 解码时需要同时传入 `z` 和 `zq`（空间条件），PyTorch 在内部处理

#### 🔍 什么是 `zq` (Spatial Conditioning)？

`zq` 是 **Spatial Conditioning**（空间条件）的缩写，在 CogVideoX VAE 解码器中扮演重要角色：

**1. 基本概念**
- `zq` 代表潜在表示 (latent representation)，用于 **Spatially Adaptive Normalization**
- 在 CogVideoX 中，`zq` 实际上就是 `z` 本身（即编码得到的潜在向量）
- 它通过 SpatialNorm3D 层为解码器提供空间自适应的归一化

**2. 工作原理**（参考 [SpatialNorm3D](#41-spatialnorm3d-空间归一化)）
```python
# 在 SpatialNorm3D 中的使用
def forward(f, zq):
    # 1. 将 zq 上采样到特征图 f 的空间尺寸
    zq_upsampled = resize(zq, size=f.shape)
    
    # 2. 通过两个 1x1x1 卷积生成归一化参数
    gamma = conv_y(zq_upsampled)  # 缩放因子
    beta = conv_b(zq_upsampled)   # 偏移
    
    # 3. 应用空间自适应归一化
    normalized_f = GroupNorm(f)
    output = normalized_f * gamma + beta  # 每个空间位置有不同的 gamma 和 beta
```

**3. 为什么需要 `zq`？**
- **传统 GroupNorm**：所有空间位置共享相同的归一化参数（gamma 和 beta）
- **SpatialNorm3D**：每个空间位置有不同的参数，这些参数由 `zq` 生成
- **优势**：解码器可以根据潜在表示的内容，为不同区域生成不同的归一化策略

**4. PyTorch vs Flax 的差异**

| 方面 | PyTorch | Flax |
|------|---------|------|
| 参数传递 | `decode(z)` - 内部自动使用 `z` 作为 `zq` | `decode(z, zq=z)` - 显式传入 |
| 实现位置 | `decoder.forward(sample, ...)` 内部处理 | 调用者需要显式传入 |
| 原因 | API 设计隐藏实现细节 | Flax 更倾向于显式参数传递 |

**5. 实际代码示例**

PyTorch:
```python
# autoencoder_kl_cogvideox.py
def decode(self, z):
    return self._decode(z)

def _decode(self, z):
    # 内部自动使用 z 作为 spatial conditioning
    z_intermediate, conv_cache = self.decoder(z_intermediate, conv_cache=conv_cache)
    # decoder 内部会将 z_intermediate 作为 zq 使用
```

Flax:
```python
# autoencoder_kl_cogvideox_flax.py
def decode(self, z, zq=None):
    if zq is None:
        zq = z  # 默认使用 z 作为 zq
    return self._decode(z, zq)

def _decode(self, z, zq):
    # 显式传递 z 和 zq 给 decoder
    decoded_frame, _ = self.decoder(z_frame, zq_frame, ...)
```

**6. 使用场景**
```python
# 标准用法：zq = z（最常见）
decoded = vae.decode(z, zq=z)

# 理论上可以用不同的条件（实验性）
# decoded = vae.decode(z, zq=other_conditioning)
# 但在 CogVideoX 中，始终使用 z 本身作为条件
```

**总结**：`zq` 是解码器中 SpatialNorm3D 层的空间条件信号，它使解码器能够为不同空间位置生成自适应的归一化参数，从而提高重建质量。在实践中，`zq` 总是等于 `z`。

---

## 3. 基础组件迁移

### 3.1 Conv3d 基础卷积

#### PyTorch: `CogVideoXSafeConv3d`

```python
# autoencoder_kl_cogvideox.py 第 38-66 行
class CogVideoXSafeConv3d(nn.Conv3d):
    """
    避免 OOM 的 3D 卷积，通过分块处理大张量
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        memory_count = (input.shape[0] * ... * input.shape[4]) * 2 / 1024**3
        
        if memory_count > 2:  # > 2GB
            # 分块处理
            part_num = int(memory_count / 2) + 1
            input_chunks = torch.chunk(input, part_num, dim=2)
            
            # 处理 kernel overlap
            if kernel_size > 1:
                input_chunks = [input_chunks[0]] + [
                    torch.cat((input_chunks[i-1][:, :, -kernel_size+1:], 
                              input_chunks[i]), dim=2)
                    for i in range(1, len(input_chunks))
                ]
            
            # 分块卷积
            output_chunks = [super().forward(chunk) for chunk in input_chunks]
            return torch.cat(output_chunks, dim=2)
        else:
            return super().forward(input)
```

**功能**：
- 输入：`(B, C, T, H, W)` 格式
- 自动检测内存使用，超过 2GB 时分块处理
- 处理时间维度的卷积核重叠

#### Flax: `FlaxConv3d`

```python
# autoencoder_kl_cogvideox_flax.py 第 182-219 行
class FlaxConv3d(nnx.Module):
    """基础 3D 卷积封装"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, rngs):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3
        
        # 处理 padding
        if isinstance(padding, int):
            if padding == 0:
                padding_mode = ((0, 0), (0, 0), (0, 0))
            else:
                padding_mode = ((padding, padding), (padding, padding), (padding, padding))
        
        self.conv = nnx.Conv(
            in_channels, out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding=padding_mode,
            rngs=rngs
        )
    
    def __call__(self, x):
        return self.conv(x)
```

**关键差异**：
1. ✅ **输入格式**：Flax 的 `nnx.Conv` 期望 `(B, T, H, W, C)` channel-last 格式
2. ✅ **Padding 格式**：
   - PyTorch: `padding=1` 表示每个维度填充 1
   - Flax: `padding=((1,1), (1,1), (1,1))` 显式指定前后填充
3. ❌ **内存优化**：Flax 版本**未实现**分块处理，依赖 JAX 的自动内存管理
   - 原因：JAX 的 XLA 编译器会自动优化内存使用
   - TPU 上通常不会遇到单个卷积超过 2GB 的情况

### 3.2 CausalConv3d 因果卷积

这是 CogVideoX 的核心组件，确保时间维度的因果性。

#### PyTorch: `CogVideoXCausalConv3d`

```python
# autoencoder_kl_cogvideox.py 第 69-147 行
class CogVideoXCausalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 dilation=1, pad_mode="constant"):
        super().__init__()
        
        # 计算填充
        time_pad = time_kernel_size - 1
        height_pad = (height_kernel_size - 1) // 2
        width_pad = (width_kernel_size - 1) // 2
        
        self.pad_mode = pad_mode
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)
        
        # 创建底层卷积
        self.conv = CogVideoXSafeConv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=(stride, 1, 1),
            padding=0 if pad_mode == "replicate" else (0, height_pad, width_pad),
        )
    
    def forward(self, inputs, conv_cache=None):
        # 因果填充
        if self.pad_mode == "replicate":
            inputs = F.pad(inputs, self.time_causal_padding, mode="replicate")
        else:
            # 使用 conv_cache
            if self.time_kernel_size > 1:
                if conv_cache is not None:
                    cached_inputs = conv_cache
                else:
                    cached_inputs = inputs[:, :, :1].repeat(1, 1, self.time_kernel_size-1, 1, 1)
                inputs = torch.cat([cached_inputs, inputs], dim=2)
        
        # 卷积
        output = self.conv(inputs)
        
        # 更新缓存
        if self.pad_mode != "replicate":
            conv_cache = inputs[:, :, -(self.time_kernel_size-1):].clone()
        
        return output, conv_cache
```

**核心逻辑**：
1. **时间因果性**：只在时间维度前面填充（`time_pad, 0`），不在后面
2. **空间对称**：高度和宽度对称填充
3. **两种模式**：
   - `replicate`：直接复制边缘值填充
   - `constant`：使用 `conv_cache` 缓存前几帧

#### Flax: `FlaxCogVideoXCausalConv3d`

```python
# autoencoder_kl_cogvideox_flax.py 第 260-494 行
class FlaxCogVideoXCausalConv3d(nnx.Module):
    CACHE_T = 2  # 缓存帧数
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 dilation=1, pad_mode="constant", rngs=None):
        # 计算填充
        self.time_pad = time_kernel_size - 1
        self.height_pad = (height_kernel_size - 1) // 2
        self.width_pad = (width_kernel_size - 1) // 2
        
        self.pad_mode = pad_mode
        self.time_kernel_size = time_kernel_size
        self.temporal_dim = 1  # JAX 中时间维度是 1
        
        # 创建卷积
        self.conv = FlaxConv3d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=(stride, 1, 1),
            padding=0 if pad_mode == "replicate" else (0, height_pad, width_pad),
            rngs=rngs
        )
    
    def __call__(self, inputs, conv_cache=None, feat_cache=None, feat_idx=None):
        # 支持两种缓存模式
        if feat_cache is not None and feat_idx is not None:
            return self._call_with_feat_cache(inputs, feat_cache, feat_idx)
        return self._call_with_conv_cache(inputs, conv_cache)
```

**新增功能**：双缓存模式

1. **旧模式** `_call_with_conv_cache`（兼容性）：

```python
def _call_with_conv_cache(self, inputs, conv_cache):
    # 类似 PyTorch 的实现
    if self.pad_mode == "replicate":
        pad_width = [(0,0), (self.time_pad, 0), (self.height_pad, self.height_pad), 
                     (self.width_pad, self.width_pad), (0,0)]
        inputs = jnp.pad(inputs, pad_width, mode='edge')
        conv_cache = None
    else:
        if self.time_kernel_size > 1:
            if conv_cache is not None:
                cached_inputs = conv_cache
            else:
                cached_inputs = jnp.tile(inputs[:, :1, :, :, :], 
                                        (1, self.time_kernel_size-1, 1, 1, 1))
            inputs = jnp.concatenate([cached_inputs, inputs], axis=1)
    
    output = self.conv(inputs)
    
    if self.pad_mode != "replicate":
        new_cache = inputs[:, -(self.time_kernel_size-1):, :, :, :]
    else:
        new_cache = None
    
    return output, new_cache
```

2. **新模式** `_call_with_feat_cache`（逐帧解码）：

```python
def _call_with_feat_cache(self, inputs, feat_cache, feat_idx):
    """
    参考 WAN VAE 的设计，支持逐帧处理
    """
    idx = feat_idx[0]
    
    if self.pad_mode == "replicate":
        # Replicate 模式
        pad_width = [(0,0), (self.time_pad, 0), ...]
        x = jnp.pad(inputs, pad_width, mode='edge')
    else:
        # Constant 模式：使用 feat_cache
        if self.time_kernel_size > 1:
            padding_needed = self.time_kernel_size - 1
            
            if feat_cache[idx] is not None:
                # 拼接缓存和当前输入
                x = jnp.concatenate([feat_cache[idx], inputs], axis=1)
                
                # 调整 padding
                cache_len = feat_cache[idx].shape[1]
                padding_needed -= cache_len
                if padding_needed > 0:
                    extra_padding = jnp.tile(x[:, :1, ...], (1, padding_needed, 1, 1, 1))
                    x = jnp.concatenate([extra_padding, x], axis=1)
            else:
                # 第一次：重复第一帧
                padding_frames = jnp.tile(inputs[:, :1, ...], (1, padding_needed, 1, 1, 1))
                x = jnp.concatenate([padding_frames, inputs], axis=1)
            
            # 执行卷积
            x2 = self.conv(x)
            
            # ⚠️ 关键：更新缓存（使用 inputs 而非 x2）
            if inputs.shape[1] < self.CACHE_T and feat_cache[idx] is not None:
                feat_cache[idx] = jnp.concatenate([
                    jnp.expand_dims(feat_cache[idx][:, -1, ...], axis=1),
                    inputs[:, -self.CACHE_T:, ...]
                ], axis=1)
            else:
                feat_cache[idx] = inputs[:, -self.CACHE_T:, ...]
            
            feat_idx[0] += 1
            return x2, None
        else:
            x = inputs
    
    output = self.conv(x)
    feat_idx[0] += 1
    return output, None
```

**关键差异**：
1. ✅ **维度调整**：`temporal_dim = 2` (PyTorch) → `temporal_dim = 1` (Flax)
2. ✅ **双缓存支持**：
   - `conv_cache`：向后兼容 PyTorch 的方式
   - `feat_cache + feat_idx`：新的逐帧解码方式（参考 WAN VAE）
3. ✅ **缓存更新逻辑**：
   - PyTorch: `conv_cache = inputs[:, :, -k+1:].clone()`
   - Flax: 使用列表 `feat_cache[idx]` 并支持动态更新

### 3.3 GroupNorm 组归一化

GroupNorm 是 VAE 中的关键归一化层。

#### PyTorch: `nn.GroupNorm`

```python
# PyTorch 内置，使用方式：
self.norm1 = nn.GroupNorm(num_channels=in_channels, num_groups=groups, eps=eps)

# 前向传播
hidden_states = self.norm1(hidden_states)  # (B, C, T, H, W)
```

**计算逻辑**：
1. 将通道分组：`C` → `num_groups × (C // num_groups)`
2. 计算每组的均值和方差（在 T, H, W 维度上）
3. 归一化：`(x - mean) / sqrt(var + eps)`
4. 仿射变换：`x * gamma + beta`

#### Flax: `FlaxGroupNorm`

```python
# autoencoder_kl_cogvideox_flax.py 第 497-589 行
class FlaxGroupNorm(nnx.Module):
    def __init__(self, num_groups, num_channels, epsilon=1e-6, rngs=None):
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.epsilon = epsilon
        
        # 可学习参数
        self.scale = nnx.Param(jnp.ones((num_channels,)))
        self.bias = nnx.Param(jnp.zeros((num_channels,)))
    
    def __call__(self, x):
        if len(x.shape) == 5:  # 5D: (B, T, H, W, C)
            B, T, H, W, C = x.shape
            channels_per_group = C // self.num_groups
            
            # Reshape 暴露组: (B, T, H, W, num_groups, C//num_groups)
            x_grouped = x.reshape(B, T, H, W, self.num_groups, channels_per_group)
            
            # 计算统计量（在 T, H, W, C//num_groups 维度上）
            mean = jnp.mean(x_grouped, axis=(1, 2, 3, 5), keepdims=True)
            var = jnp.var(x_grouped, axis=(1, 2, 3, 5), keepdims=True)
            
            # 归一化
            x_norm = (x_grouped - mean) / jnp.sqrt(var + self.epsilon)
            x_norm = x_norm.reshape(B, T, H, W, C)
            
            # 仿射变换
            x_out = x_norm * self.scale.value.reshape(1, 1, 1, 1, C) + \
                    self.bias.value.reshape(1, 1, 1, 1, C)
        else:  # 4D: (B, H, W, C)
            # 类似逻辑...
        
        return x_out
```

**关键差异**：
1. ✅ **格式转换**：直接在 channel-last 格式计算，避免转置开销
2. ✅ **数学等价**：
   - PyTorch 在 channel-first 格式计算统计量
   - Flax 直接在 channel-last 格式计算，数学上完全等价
3. ✅ **参数命名**：
   - PyTorch: `weight`, `bias`
   - Flax: `scale`, `bias` (更清晰的语义)

---

*（第一部分完成，接下来继续...）*


## 4. 核心模块迁移

### 4.1 SpatialNorm3D 空间归一化

SpatialNorm3D 是解码器专用的条件归一化层，使用潜在表示作为条件信号。

#### PyTorch: `CogVideoXSpatialNorm3D`

```python
# autoencoder_kl_cogvideox.py 第 149-197 行
class CogVideoXSpatialNorm3D(nn.Module):
    def __init__(self, f_channels, zq_channels, groups=32):
        super().__init__()
        self.norm_layer = nn.GroupNorm(num_channels=f_channels, num_groups=groups, eps=1e-6)
        self.conv_y = CogVideoXCausalConv3d(zq_channels, f_channels, kernel_size=1)
        self.conv_b = CogVideoXCausalConv3d(zq_channels, f_channels, kernel_size=1)
    
    def forward(self, f, zq, conv_cache=None):
        new_conv_cache = {}
        conv_cache = conv_cache or {}
        
        # 处理奇数帧特殊情况
        if f.shape[2] > 1 and f.shape[2] % 2 == 1:
            f_first, f_rest = f[:, :, :1], f[:, :, 1:]
            z_first, z_rest = zq[:, :, :1], zq[:, :, 1:]
            z_first = F.interpolate(z_first, size=f_first.shape[-3:])
            z_rest = F.interpolate(z_rest, size=f_rest.shape[-3:])
            zq = torch.cat([z_first, z_rest], dim=2)
        else:
            zq = F.interpolate(zq, size=f.shape[-3:])
        
        # 应用条件卷积
        conv_y, new_conv_cache["conv_y"] = self.conv_y(zq, conv_cache=conv_cache.get("conv_y"))
        conv_b, new_conv_cache["conv_b"] = self.conv_b(zq, conv_cache=conv_cache.get("conv_b"))
        
        # 归一化 + 条件
        norm_f = self.norm_layer(f)
        new_f = norm_f * conv_y + conv_b
        
        return new_f, new_conv_cache
```

**功能**：
1. 对特征 `f` 进行 GroupNorm 归一化
2. 将潜在表示 `zq` 上采样到 `f` 的空间尺寸
3. 使用两个 1x1x1 卷积生成缩放因子 `conv_y` 和偏移 `conv_b`
4. 应用仿射变换：`norm_f * conv_y + conv_b`

**输入**：
- `f`: 特征图 `(B, C, T, H, W)`
- `zq`: 潜在条件 `(B, C', T', H', W')`

**输出**：
- 条件归一化后的特征

#### Flax: `FlaxCogVideoXSpatialNorm3D`

```python
# autoencoder_kl_cogvideox_flax.py 第 592-727 行
class FlaxCogVideoXSpatialNorm3D(nnx.Module):
    def __init__(self, f_channels, zq_channels, groups=32, rngs=None):
        self.norm_layer = FlaxGroupNorm(
            num_groups=groups, num_channels=f_channels, epsilon=1e-6, rngs=rngs
        )
        self.conv_y = FlaxCogVideoXCausalConv3d(
            zq_channels, f_channels, kernel_size=1, stride=1, pad_mode="constant", rngs=rngs
        )
        self.conv_b = FlaxCogVideoXCausalConv3d(
            zq_channels, f_channels, kernel_size=1, stride=1, pad_mode="constant", rngs=rngs
        )
    
    def __call__(self, f, zq, conv_cache=None, feat_cache=None, feat_idx=None):
        # 支持两种缓存模式
        if feat_cache is not None and feat_idx is not None:
            return self._call_with_feat_cache(f, zq, feat_cache, feat_idx)
        return self._call_with_conv_cache(f, zq, conv_cache)
    
    def _call_with_conv_cache(self, f, zq, conv_cache):
        new_conv_cache = {}
        conv_cache = conv_cache or {}
        
        # 处理奇数帧（与 PyTorch 完全一致）
        B, T, H, W, C = f.shape
        if T > 1 and T % 2 == 1:
            f_first = f[:, :1, :, :, :]
            f_rest = f[:, 1:, :, :, :]
            z_first = zq[:, :1, :, :, :]
            z_rest = zq[:, 1:, :, :, :]
            
            # 分别 resize
            z_first = jax.image.resize(z_first, (B, 1, H, W, zq.shape[-1]), method='nearest')
            z_rest = jax.image.resize(z_rest, (B, T-1, H, W, zq.shape[-1]), method='nearest')
            
            zq = jnp.concatenate([z_first, z_rest], axis=1)
        else:
            zq = jax.image.resize(zq, (B, T, H, W, zq.shape[-1]), method='nearest')
        
        # 应用条件卷积
        conv_y, new_conv_cache["conv_y"] = self.conv_y(zq, conv_cache=conv_cache.get("conv_y"))
        conv_b, new_conv_cache["conv_b"] = self.conv_b(zq, conv_cache=conv_cache.get("conv_b"))
        
        # 归一化 + 条件
        norm_f = self.norm_layer(f)
        new_f = norm_f * conv_y + conv_b
        
        return new_f, new_conv_cache
```

**关键差异**：
1. ✅ **插值方法**：
   - PyTorch: `F.interpolate(...)` 默认双线性插值
   - Flax: `jax.image.resize(..., method='nearest')` 最近邻插值
   - 使用最近邻是为了与 PyTorch 的默认行为匹配
2. ✅ **奇数帧处理**：完全保留 PyTorch 的逻辑
   - 原因：避免第一帧和其余帧的不一致
3. ✅ **双缓存模式**：同样支持 `conv_cache` 和 `feat_cache` 两种模式

### 4.2 ResnetBlock3D 残差块

ResNet 块是 VAE 的基础构建单元。

#### PyTorch: `CogVideoXResnetBlock3D`

```python
# autoencoder_kl_cogvideox.py 第 200-328 行
class CogVideoXResnetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0, temb_channels=512,
                 groups=32, eps=1e-6, non_linearity="swish", conv_shortcut=False,
                 spatial_norm_dim=None, pad_mode="first"):
        super().__init__()
        out_channels = out_channels or in_channels
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nonlinearity = get_activation(non_linearity)
        self.use_conv_shortcut = conv_shortcut
        
        # 归一化层
        if spatial_norm_dim is None:
            # 编码器：使用 GroupNorm
            self.norm1 = nn.GroupNorm(num_channels=in_channels, num_groups=groups, eps=eps)
            self.norm2 = nn.GroupNorm(num_channels=out_channels, num_groups=groups, eps=eps)
        else:
            # 解码器：使用 SpatialNorm3D
            self.norm1 = CogVideoXSpatialNorm3D(
                f_channels=in_channels, zq_channels=spatial_norm_dim, groups=groups
            )
            self.norm2 = CogVideoXSpatialNorm3D(
                f_channels=out_channels, zq_channels=spatial_norm_dim, groups=groups
            )
        
        # 卷积层
        self.conv1 = CogVideoXCausalConv3d(in_channels, out_channels, kernel_size=3, pad_mode=pad_mode)
        
        # 时间嵌入投影
        if temb_channels > 0:
            self.temb_proj = nn.Linear(in_features=temb_channels, out_features=out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.conv2 = CogVideoXCausalConv3d(out_channels, out_channels, kernel_size=3, pad_mode=pad_mode)
        
        # 快捷连接
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CogVideoXCausalConv3d(in_channels, out_channels, kernel_size=3, pad_mode=pad_mode)
            else:
                self.conv_shortcut = CogVideoXSafeConv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, inputs, temb=None, zq=None, conv_cache=None):
        new_conv_cache = {}
        conv_cache = conv_cache or {}
        
        hidden_states = inputs
        
        # 第一个归一化 + 卷积
        if zq is not None:
            hidden_states, new_conv_cache["norm1"] = self.norm1(
                hidden_states, zq, conv_cache=conv_cache.get("norm1")
            )
        else:
            hidden_states = self.norm1(hidden_states)
        
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states, new_conv_cache["conv1"] = self.conv1(
            hidden_states, conv_cache=conv_cache.get("conv1")
        )
        
        # 时间嵌入
        if temb is not None:
            hidden_states = hidden_states + self.temb_proj(self.nonlinearity(temb))[:, :, None, None, None]
        
        # 第二个归一化 + 卷积
        if zq is not None:
            hidden_states, new_conv_cache["norm2"] = self.norm2(
                hidden_states, zq, conv_cache=conv_cache.get("norm2")
            )
        else:
            hidden_states = self.norm2(hidden_states)
        
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states, new_conv_cache["conv2"] = self.conv2(
            hidden_states, conv_cache=conv_cache.get("conv2")
        )
        
        # 快捷连接
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                inputs, new_conv_cache["conv_shortcut"] = self.conv_shortcut(
                    inputs, conv_cache=conv_cache.get("conv_shortcut")
                )
            else:
                inputs = self.conv_shortcut(inputs)
        
        # 残差连接
        hidden_states = hidden_states + inputs
        
        return hidden_states, new_conv_cache
```

**结构**：
```
输入 ──┬──> Norm1 -> Act -> Conv1 -> (+temb) -> Norm2 -> Act -> Dropout -> Conv2 ──┬──> 输出
       │                                                                             │
       └──────────────────────> Conv_shortcut (如果通道数改变) ──────────────────────┘
```

**输入**：
- `inputs`: `(B, C, T, H, W)`
- `temb`: 时间嵌入 `(B, temb_channels)` (可选)
- `zq`: 空间条件 `(B, C', T', H', W')` (解码器)

**输出**：
- 残差连接后的特征

#### Flax: `FlaxCogVideoXResnetBlock3D`

```python
# autoencoder_kl_cogvideox_flax.py 第 733-980 行
class FlaxCogVideoXResnetBlock3D(nnx.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0, temb_channels=512,
                 groups=32, eps=1e-6, non_linearity="swish", conv_shortcut=False,
                 spatial_norm_dim=None, pad_mode="first", rngs=None):
        out_channels = out_channels or in_channels
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.spatial_norm_dim = spatial_norm_dim
        
        # 归一化层
        if spatial_norm_dim is None:
            self.norm1 = FlaxGroupNorm(num_groups=groups, num_channels=in_channels, epsilon=eps, rngs=rngs)
            self.norm2 = FlaxGroupNorm(num_groups=groups, num_channels=out_channels, epsilon=eps, rngs=rngs)
        else:
            self.norm1 = FlaxCogVideoXSpatialNorm3D(
                f_channels=in_channels, zq_channels=spatial_norm_dim, groups=groups, rngs=rngs
            )
            self.norm2 = FlaxCogVideoXSpatialNorm3D(
                f_channels=out_channels, zq_channels=spatial_norm_dim, groups=groups, rngs=rngs
            )
        
        # 卷积层
        self.conv1 = FlaxCogVideoXCausalConv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, pad_mode=pad_mode, rngs=rngs
        )
        
        # 时间嵌入投影
        if temb_channels > 0:
            self.temb_proj = nnx.Linear(temb_channels, out_channels, rngs=rngs)
        else:
            self.temb_proj = None
        
        self.dropout_rate = dropout
        self.conv2 = FlaxCogVideoXCausalConv3d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, pad_mode=pad_mode, rngs=rngs
        )
        
        # 快捷连接
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = FlaxCogVideoXCausalConv3d(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=3, pad_mode=pad_mode, rngs=rngs
                )
            else:
                self.conv_shortcut = FlaxConv3d(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, rngs=rngs
                )
        else:
            self.conv_shortcut = None
    
    def __call__(self, inputs, temb=None, zq=None, conv_cache=None, 
                 feat_cache=None, feat_idx=None, deterministic=True):
        # 支持两种缓存模式
        if feat_cache is not None and feat_idx is not None:
            return self._call_with_feat_cache(inputs, temb, zq, feat_cache, feat_idx, deterministic)
        return self._call_with_conv_cache(inputs, temb, zq, conv_cache, deterministic)
    
    def _call_with_conv_cache(self, inputs, temb, zq, conv_cache, deterministic):
        new_conv_cache = {}
        conv_cache = conv_cache or {}
        
        hidden_states = inputs
        
        # 第一个归一化 + 卷积
        if zq is not None:
            hidden_states, new_conv_cache["norm1"] = self.norm1(
                hidden_states, zq, conv_cache=conv_cache.get("norm1")
            )
        else:
            hidden_states = self.norm1(hidden_states)
        
        hidden_states = jax.nn.silu(hidden_states)  # swish 激活
        hidden_states, new_conv_cache["conv1"] = self.conv1(
            hidden_states, conv_cache=conv_cache.get("conv1")
        )
        
        # 时间嵌入
        if temb is not None and self.temb_proj is not None:
            temb_proj = self.temb_proj(jax.nn.silu(temb))
            hidden_states = hidden_states + temb_proj[:, None, None, None, :]
        
        # 第二个归一化 + 卷积
        if zq is not None:
            hidden_states, new_conv_cache["norm2"] = self.norm2(
                hidden_states, zq, conv_cache=conv_cache.get("norm2")
            )
        else:
            hidden_states = self.norm2(hidden_states)
        
        hidden_states = jax.nn.silu(hidden_states)
        
        # Dropout
        if self.dropout_rate > 0 and not deterministic:
            hidden_states = nnx.Dropout(rate=self.dropout_rate)(hidden_states)
        
        hidden_states, new_conv_cache["conv2"] = self.conv2(
            hidden_states, conv_cache=conv_cache.get("conv2")
        )
        
        # 快捷连接
        if self.conv_shortcut is not None:
            if self.use_conv_shortcut:
                inputs, new_conv_cache["conv_shortcut"] = self.conv_shortcut(
                    inputs, conv_cache=conv_cache.get("conv_shortcut")
                )
            else:
                inputs = self.conv_shortcut(inputs)
        
        # 残差连接
        hidden_states = hidden_states + inputs
        
        return hidden_states, new_conv_cache
```

**关键差异**：
1. ✅ **激活函数**：
   - PyTorch: `self.nonlinearity = get_activation(non_linearity)` → 使用字符串配置
   - Flax: 直接使用 `jax.nn.silu`（因为 CogVideoX 只用 swish/silu）
2. ✅ **Dropout**：
   - PyTorch: `self.dropout = nn.Dropout(dropout)`
   - Flax: 使用 `nnx.Dropout` + `deterministic` 参数控制
3. ✅ **时间嵌入维度**：
   - PyTorch: `[:, :, None, None, None]` → `(B, C, 1, 1, 1)`
   - Flax: `[:, None, None, None, :]` → `(B, 1, 1, 1, C)`
   - 原因：数据格式不同（BCTHW vs BTHWC）

---

文档持续更新中，由于长度限制，剩余部分将在下一次继续完成。已完成的内容包括：

✅ 整体架构概述
✅ 调用流程详解（Encode/Decode）
✅ 基础组件迁移（Conv3d, GroupNorm, CausalConv3d）
✅ 核心模块迁移（SpatialNorm3D, ResnetBlock3D）

待完成部分：
- DownBlock、MidBlock、UpBlock 的迁移
- Encoder3D 和 Decoder3D 的迁移
- 主 VAE 类的迁移
- Tiling 功能的迁移
- Frame batching 和 conv_cache 机制
- 关键差异总结


### 4.3 DownBlock3D、MidBlock3D、UpBlock3D

这三个模块分别负责编码器的下采样、中间处理和解码器的上采样。

#### 4.3.1 DownBlock3D 下采样块

**PyTorch 版本**（第 331-441 行）：

```python
class CogVideoXDownBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels, dropout=0.0,
                 num_layers=1, resnet_eps=1e-6, resnet_act_fn="swish",
                 resnet_groups=32, add_downsample=True, downsample_padding=0,
                 compress_time=False, pad_mode="first"):
        super().__init__()
        
        # 创建多个 ResNet 块
        resnets = []
        for i in range(num_layers):
            in_channel = in_channels if i == 0 else out_channels
            resnets.append(
                CogVideoXResnetBlock3D(
                    in_channels=in_channel, out_channels=out_channels,
                    dropout=dropout, temb_channels=temb_channels,
                    groups=resnet_groups, eps=resnet_eps,
                    non_linearity=resnet_act_fn, pad_mode=pad_mode
                )
            )
        self.resnets = nn.ModuleList(resnets)
        
        # 下采样器
        if add_downsample:
            self.downsamplers = nn.ModuleList([
                CogVideoXDownsample3D(
                    out_channels, out_channels,
                    padding=downsample_padding, compress_time=compress_time
                )
            ])
        else:
            self.downsamplers = None
    
    def forward(self, hidden_states, temb=None, zq=None, conv_cache=None):
        new_conv_cache = {}
        conv_cache = conv_cache or {}
        
        # 依次通过每个 ResNet 块
        for i, resnet in enumerate(self.resnets):
            conv_cache_key = f"resnet_{i}"
            hidden_states, new_conv_cache[conv_cache_key] = resnet(
                hidden_states, temb, zq, conv_cache=conv_cache.get(conv_cache_key)
            )
        
        # 下采样
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        
        return hidden_states, new_conv_cache
```

**Flax 版本**（第 983-1113 行）：

```python
class FlaxCogVideoXDownBlock3D(nnx.Module):
    def __init__(self, in_channels, out_channels, temb_channels, dropout=0.0,
                 num_layers=1, resnet_eps=1e-6, resnet_act_fn="swish",
                 resnet_groups=32, add_downsample=True, downsample_padding=0,
                 compress_time=False, pad_mode="first", rngs=None):
        # 创建 ResNet 块列表
        resnets = []
        for i in range(num_layers):
            in_channel = in_channels if i == 0 else out_channels
            resnet = FlaxCogVideoXResnetBlock3D(
                in_channels=in_channel, out_channels=out_channels,
                dropout=dropout, temb_channels=temb_channels,
                groups=resnet_groups, eps=resnet_eps,
                non_linearity=resnet_act_fn, pad_mode=pad_mode, rngs=rngs
            )
            resnets.append(resnet)
        self.resnets = nnx.List(resnets)
        
        # 下采样器（使用 2D Conv）
        if add_downsample:
            downsampler = FlaxConv2d(
                out_channels, out_channels,
                kernel_size=3, stride=2, padding=0, rngs=rngs
            )
            self.downsamplers = nnx.List([downsampler])
            self.compress_time = compress_time
            self.downsample_padding = downsample_padding
        else:
            self.downsamplers = None
    
    def __call__(self, hidden_states, temb=None, zq=None, conv_cache=None, deterministic=True):
        new_conv_cache = {}
        conv_cache = conv_cache or {}
        
        # ResNet 块
        for i, resnet in enumerate(self.resnets):
            conv_cache_key = f"resnet_{i}"
            hidden_states, new_conv_cache[conv_cache_key] = resnet(
                hidden_states, temb, zq,
                conv_cache=conv_cache.get(conv_cache_key),
                deterministic=deterministic
            )
        
        # 下采样（包含时间压缩和空间下采样）
        if self.downsamplers is not None:
            # 时间压缩（如果需要）
            if self.compress_time:
                B, T, H, W, C = hidden_states.shape
                # 使用平均池化压缩时间维度
                hidden_states = hidden_states.reshape(B * H * W, T, C)
                hidden_states = hidden_states.transpose(0, 2, 1)  # (B*H*W, C, T)
                
                if T % 2 == 1:
                    # 奇数帧：保留第一帧，压缩其余帧
                    first_frame = hidden_states[:, :, 0:1]
                    rest_frames = hidden_states[:, :, 1:]
                    if rest_frames.shape[2] > 0:
                        rest_frames = jnp.mean(
                            rest_frames.reshape(B*H*W, C, rest_frames.shape[2]//2, 2),
                            axis=-1
                        )
                    hidden_states = jnp.concatenate([first_frame, rest_frames], axis=2)
                else:
                    # 偶数帧：直接压缩
                    hidden_states = jnp.mean(
                        hidden_states.reshape(B*H*W, C, T//2, 2), axis=-1
                    )
                
                # 重塑回 5D
                T_new = hidden_states.shape[2]
                hidden_states = hidden_states.transpose(0, 2, 1)
                hidden_states = hidden_states.reshape(B, H, W, T_new, C)
                hidden_states = hidden_states.transpose(0, 3, 1, 2, 4)  # (B, T_new, H, W, C)
            
            # 空间下采样（2D）
            for downsampler in self.downsamplers:
                B, T, H, W, C = hidden_states.shape
                
                # 添加手动填充 (0, 1, 0, 1) - PyTorch 的默认行为
                pad_width = [(0,0), (0,0), (0,1), (0,1), (0,0)]
                hidden_states = jnp.pad(hidden_states, pad_width, mode='constant')
                
                # 重塑为 4D 应用 2D 卷积
                _, _, H_padded, W_padded, _ = hidden_states.shape
                hidden_states = hidden_states.reshape(B * T, H_padded, W_padded, C)
                hidden_states = downsampler(hidden_states)
                
                # 重塑回 5D
                _, H_new, W_new, _ = hidden_states.shape
                hidden_states = hidden_states.reshape(B, T, H_new, W_new, C)
        
        return hidden_states, new_conv_cache
```

**关键差异**：
1. ✅ **时间压缩**：PyTorch 使用 `CogVideoXDownsample3D`，Flax 直接实现平均池化逻辑
2. ✅ **空间下采样**：
   - PyTorch: 使用 `Conv3d` 对空间维度下采样
   - Flax: 使用 `FlaxConv2d` + reshape 实现（更高效）
3. ✅ **手动填充**：Flax 需要手动添加 `(0, 1, 0, 1)` 填充以匹配 PyTorch

#### 4.3.2 MidBlock3D 中间块

**PyTorch 版本**（第 444-528 行）：

```python
class CogVideoXMidBlock3D(nn.Module):
    def __init__(self, in_channels, temb_channels, dropout=0.0, num_layers=1,
                 resnet_eps=1e-6, resnet_act_fn="swish", resnet_groups=32,
                 spatial_norm_dim=None, pad_mode="first"):
        super().__init__()
        
        # 创建多个 ResNet 块
        resnets = []
        for _ in range(num_layers):
            resnets.append(
                CogVideoXResnetBlock3D(
                    in_channels=in_channels, out_channels=in_channels,
                    dropout=dropout, temb_channels=temb_channels,
                    groups=resnet_groups, eps=resnet_eps,
                    spatial_norm_dim=spatial_norm_dim,
                    non_linearity=resnet_act_fn, pad_mode=pad_mode
                )
            )
        self.resnets = nn.ModuleList(resnets)
    
    def forward(self, hidden_states, temb=None, zq=None, conv_cache=None):
        new_conv_cache = {}
        conv_cache = conv_cache or {}
        
        for i, resnet in enumerate(self.resnets):
            conv_cache_key = f"resnet_{i}"
            hidden_states, new_conv_cache[conv_cache_key] = resnet(
                hidden_states, temb, zq, conv_cache=conv_cache.get(conv_cache_key)
            )
        
        return hidden_states, new_conv_cache
```

**Flax 版本**（第 1116-1186 行）：

```python
class FlaxCogVideoXMidBlock3D(nnx.Module):
    def __init__(self, in_channels, temb_channels, dropout=0.0, num_layers=1,
                 resnet_eps=1e-6, resnet_act_fn="swish", resnet_groups=32,
                 spatial_norm_dim=None, pad_mode="first", rngs=None):
        resnets = []
        for i in range(num_layers):
            resnet = FlaxCogVideoXResnetBlock3D(
                in_channels=in_channels, out_channels=in_channels,
                dropout=dropout, temb_channels=temb_channels,
                groups=resnet_groups, eps=resnet_eps,
                spatial_norm_dim=spatial_norm_dim,
                non_linearity=resnet_act_fn, pad_mode=pad_mode, rngs=rngs
            )
            resnets.append(resnet)
        self.resnets = nnx.List(resnets)
    
    def __call__(self, hidden_states, temb=None, zq=None, conv_cache=None,
                 feat_cache=None, feat_idx=None, deterministic=True):
        # 支持双缓存模式
        if feat_cache is not None and feat_idx is not None:
            for resnet in self.resnets:
                hidden_states, _ = resnet(
                    hidden_states, temb, zq,
                    feat_cache=feat_cache, feat_idx=feat_idx,
                    deterministic=deterministic
                )
            return hidden_states, None
        
        # 旧模式
        new_conv_cache = {}
        conv_cache = conv_cache or {}
        
        for i, resnet in enumerate(self.resnets):
            conv_cache_key = f"resnet_{i}"
            hidden_states, new_conv_cache[conv_cache_key] = resnet(
                hidden_states, temb, zq,
                conv_cache=conv_cache.get(conv_cache_key),
                deterministic=deterministic
            )
        
        return hidden_states, new_conv_cache
```

**关键差异**：
- ✅ 结构完全一致，只是 Flax 版本支持双缓存模式

#### 4.3.3 UpBlock3D 上采样块

**PyTorch 版本**（第 531-643 行）：

```python
class CogVideoXUpBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels, dropout=0.0,
                 num_layers=1, resnet_eps=1e-6, resnet_act_fn="swish",
                 resnet_groups=32, spatial_norm_dim=16, add_upsample=True,
                 upsample_padding=1, compress_time=False, pad_mode="first"):
        super().__init__()
        
        # ResNet 块
        resnets = []
        for i in range(num_layers):
            in_channel = in_channels if i == 0 else out_channels
            resnets.append(
                CogVideoXResnetBlock3D(
                    in_channels=in_channel, out_channels=out_channels,
                    dropout=dropout, temb_channels=temb_channels,
                    groups=resnet_groups, eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    spatial_norm_dim=spatial_norm_dim, pad_mode=pad_mode
                )
            )
        self.resnets = nn.ModuleList(resnets)
        
        # 上采样器
        if add_upsample:
            self.upsamplers = nn.ModuleList([
                CogVideoXUpsample3D(
                    out_channels, out_channels,
                    padding=upsample_padding, compress_time=compress_time
                )
            ])
        else:
            self.upsamplers = None
    
    def forward(self, hidden_states, temb=None, zq=None, conv_cache=None):
        new_conv_cache = {}
        conv_cache = conv_cache or {}
        
        # ResNet 块
        for i, resnet in enumerate(self.resnets):
            conv_cache_key = f"resnet_{i}"
            hidden_states, new_conv_cache[conv_cache_key] = resnet(
                hidden_states, temb, zq, conv_cache=conv_cache.get(conv_cache_key)
            )
        
        # 上采样
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        
        return hidden_states, new_conv_cache
```

**Flax 版本**（第 1189-1387 行）：

```python
class FlaxCogVideoXUpBlock3D(nnx.Module):
    def __init__(self, in_channels, out_channels, temb_channels, dropout=0.0,
                 num_layers=1, resnet_eps=1e-6, resnet_act_fn="swish",
                 resnet_groups=32, spatial_norm_dim=16, add_upsample=True,
                 upsample_padding=1, compress_time=False, pad_mode="first", rngs=None):
        # ResNet 块
        resnets = []
        for i in range(num_layers):
            in_channel = in_channels if i == 0 else out_channels
            resnet = FlaxCogVideoXResnetBlock3D(
                in_channels=in_channel, out_channels=out_channels,
                dropout=dropout, temb_channels=temb_channels,
                groups=resnet_groups, eps=resnet_eps,
                non_linearity=resnet_act_fn,
                spatial_norm_dim=spatial_norm_dim, pad_mode=pad_mode, rngs=rngs
            )
            resnets.append(resnet)
        self.resnets = nnx.List(resnets)
        
        # 上采样器（2D Conv）
        if add_upsample:
            upsampler = FlaxConv2d(
                out_channels, out_channels,
                kernel_size=3, stride=1, padding=upsample_padding, rngs=rngs
            )
            self.upsamplers = nnx.List([upsampler])
            self.compress_time = compress_time
        else:
            self.upsamplers = None
    
    def __call__(self, hidden_states, temb=None, zq=None, conv_cache=None,
                 feat_cache=None, feat_idx=None, deterministic=True):
        # 新模式：逐帧解码
        if feat_cache is not None and feat_idx is not None:
            for resnet in self.resnets:
                hidden_states, _ = resnet(
                    hidden_states, temb, zq,
                    feat_cache=feat_cache, feat_idx=feat_idx,
                    deterministic=deterministic
                )
            
            if self.upsamplers is not None:
                for upsampler in self.upsamplers:
                    B, T, H, W, C = hidden_states.shape
                    
                    # compress_time：时间 + 空间上采样
                    if self.compress_time:
                        if T == 1:
                            # 单帧 -> 2 帧 + 2x 空间
                            hidden_states = jax.image.resize(
                                hidden_states, (B, 2, H * 2, W * 2, C), method='nearest'
                            )
                        elif T > 1 and T % 2 == 1:
                            # 奇数帧：特殊处理
                            first_frame = hidden_states[:, 0, :, :, :]
                            rest_frames = hidden_states[:, 1:, :, :, :]
                            first_frame = jax.image.resize(
                                first_frame, (B, H * 2, W * 2, C), method='nearest'
                            )
                            first_frame = first_frame[:, None, :, :, :]
                            rest_frames = jax.image.resize(
                                rest_frames, (B, 2 * (T-1), H * 2, W * 2, C), method='nearest'
                            )
                            hidden_states = jnp.concatenate([first_frame, rest_frames], axis=1)
                        else:
                            # 偶数帧：常规上采样
                            hidden_states = jax.image.resize(
                                hidden_states, (B, T * 2, H * 2, W * 2, C), method='nearest'
                            )
                    else:
                        # 仅空间上采样
                        hidden_states = hidden_states.reshape(B * T, H, W, C)
                        hidden_states = jax.image.resize(
                            hidden_states, (B * T, H * 2, W * 2, C), method='nearest'
                        )
                        hidden_states = hidden_states.reshape(B, T, H * 2, W * 2, C)
                    
                    # 应用 2D 卷积
                    B, T_new, H_new, W_new, C = hidden_states.shape
                    hidden_states = hidden_states.reshape(B * T_new, H_new, W_new, C)
                    hidden_states = upsampler(hidden_states)
                    _, H_final, W_final, _ = hidden_states.shape
                    hidden_states = hidden_states.reshape(B, T_new, H_final, W_final, C)
            
            return hidden_states, None
        
        # 旧模式：（类似上面的逻辑，但使用 conv_cache）
        # ... (省略，与上面类似)
```

**关键差异**：
1. ✅ **上采样方法**：
   - PyTorch: `F.interpolate(scale_factor=2.0)` 
   - Flax: `jax.image.resize(..., method='nearest')` + 手动计算尺寸
2. ✅ **时间上采样**：
   - `compress_time=True`: 时间维度 × 2，空间维度 × 2
   - `compress_time=False`: 仅空间维度 × 2
3. ✅ **奇数帧处理**：完全复制 PyTorch 的逻辑

---

## 5. 主 VAE 类迁移

### 5.1 配置类

**PyTorch 版本**：使用 `@register_to_config` 装饰器

```python
@register_to_config
def __init__(
    self,
    in_channels: int = 3,
    out_channels: int = 3,
    # ... 更多参数
):
    super().__init__()
    # 配置自动存储在 self.config
```

**Flax 版本**：使用 `@dataclass`

```python
@dataclass
class FlaxAutoencoderKLCogVideoXConfig:
    in_channels: int = 3
    out_channels: int = 3
    # ... 更多参数
    
    @classmethod
    def from_dict(cls, config_dict: Dict):
        """从字典创建配置"""
        field_names = {f.name for f in dataclasses.fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
        return cls(**filtered_dict)
```

**关键差异**：
- PyTorch: 配置和模型合并在一个类
- Flax: 配置和模型分离（更清晰）

### 5.2 权重加载：from_pretrained

这是最复杂的部分，需要将 PyTorch 权重转换为 JAX 格式。

**Flax 实现**（第 2265-2433 行）：

```python
@classmethod
def from_pretrained(cls, pretrained_model_name_or_path, subfolder="vae", dtype=jnp.float32):
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    
    # 1. 下载配置
    config_path = hf_hub_download(pretrained_model_name_or_path, subfolder=subfolder, filename="config.json")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    config = cls.config_class.from_dict(config_dict)
    
    # 2. 下载 PyTorch 权重
    ckpt_path = hf_hub_download(
        pretrained_model_name_or_path, subfolder=subfolder,
        filename="diffusion_pytorch_model.safetensors"
    )
    
    # 3. 加载 PyTorch 权重
    pytorch_weights = {}
    with safe_open(ckpt_path, framework="np") as f:
        for key in f.keys():
            pytorch_weights[key] = f.get_tensor(key)
    
    # 4. 转换权重格式
    jax_weights = {}
    for pt_key, pt_tensor in pytorch_weights.items():
        jax_key = pt_key
        jax_tensor = pt_tensor
        
        # 移除 _orig_mod 前缀
        if jax_key.startswith("_orig_mod."):
            jax_key = jax_key[len("_orig_mod."):]
        
        # 转换卷积权重：PyTorch (O,I,T,H,W) -> JAX (T,H,W,I,O)
        if "conv" in jax_key and "weight" in jax_key:
            jax_key = jax_key.replace(".weight", ".kernel")
            
            if len(jax_tensor.shape) == 5:  # 3D conv
                jax_tensor = jax_tensor.transpose(2, 3, 4, 1, 0)
            elif len(jax_tensor.shape) == 4:  # 2D conv
                jax_tensor = jax_tensor.transpose(2, 3, 1, 0)
        
        # 转换归一化权重
        if ".weight" in jax_key and "norm" in jax_key:
            jax_key = jax_key.replace(".weight", ".scale")
        
        # 添加 .conv 路径（FlaxConv3d 包装）
        if needs_conv_wrapper(jax_key):
            parts = jax_key.rsplit('.', 1)
            jax_key = f"{parts[0]}.conv.{parts[1]}"
        
        jax_weights[jax_key] = jnp.array(jax_tensor, dtype=dtype)
    
    # 5. 创建模型并加载权重
    model = cls(config=config, rngs=nnx.Rngs(jax.random.key(0)), dtype=dtype)
    
    # 使用 NNX 的权重加载机制
    from flax.traverse_util import unflatten_dict
    nested_weights = unflatten_dict(jax_weights, sep=".")
    graphdef, _ = nnx.split(model)
    model = nnx.merge(graphdef, nested_weights)
    
    return model
```

**权重转换规则**：

| PyTorch | JAX | 说明 |
|---------|-----|------|
| `Conv3d.weight` (O,I,T,H,W) | `Conv.kernel` (T,H,W,I,O) | 5D 卷积 |
| `Conv2d.weight` (O,I,H,W) | `Conv.kernel` (H,W,I,O) | 2D 卷积 |
| `GroupNorm.weight` | `GroupNorm.scale` | 归一化缩放 |
| `Linear.weight` (O,I) | `Linear.kernel` (I,O) | 全连接层 |

---

## 6. 高级功能迁移

### 6.1 Tiling（分块处理）

Tiling 用于处理大分辨率视频，避免 OOM。

**核心思想**：
1. 将视频分割成重叠的空间块（tiles）
2. 独立处理每个块
3. 融合（blend）重叠区域

**PyTorch 实现**（第 1250-1322 行）：

```python
def tiled_encode(self, x):
    batch_size, num_channels, num_frames, height, width = x.shape
    
    # 计算 tile 参数
    overlap_height = int(self.tile_sample_min_height * (1 - self.tile_overlap_factor_height))
    overlap_width = int(self.tile_sample_min_width * (1 - self.tile_overlap_factor_width))
    blend_extent_height = int(self.tile_latent_min_height * self.tile_overlap_factor_height)
    blend_extent_width = int(self.tile_latent_min_width * self.tile_overlap_factor_width)
    
    # 分块处理
    rows = []
    for i in range(0, height, overlap_height):
        row = []
        for j in range(0, width, overlap_width):
            # 提取 tile
            tile = x[:, :, :, i:i+self.tile_sample_min_height, j:j+self.tile_sample_min_width]
            
            # 编码 tile
            tile_encoded = self.encoder(tile)
            row.append(tile_encoded)
        rows.append(row)
    
    # 融合 tiles
    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = self.blend_v(rows[i-1][j], tile, blend_extent_height)
            if j > 0:
                tile = self.blend_h(row[j-1], tile, blend_extent_width)
            result_row.append(tile[:, :, :, :row_limit_height, :row_limit_width])
        result_rows.append(torch.cat(result_row, dim=4))
    
    return torch.cat(result_rows, dim=3)
```

**Flax 实现**（第 2088-2154 行）：

```python
def tiled_encode(self, x, deterministic=True):
    # 完全相同的逻辑，只是数据格式不同
    batch_size, num_frames, height, width, num_channels = x.shape
    
    # ... (计算 tile 参数，同 PyTorch)
    
    # 分块处理
    rows = []
    for i in range(0, height, overlap_height):
        row = []
        for j in range(0, width, overlap_width):
            tile = x[:, :, i:i+self.tile_sample_min_height, j:j+self.tile_sample_min_width, :]
            tile_encoded = self.encoder(tile, deterministic=deterministic)
            row.append(tile_encoded)
        rows.append(row)
    
    # 融合（blend_v 和 blend_h 完全相同）
    # ...
    
    return jnp.concatenate(result_rows, axis=2)
```

**关键差异**：
- ✅ 算法完全相同
- ✅ 仅数据格式不同（BCTHW vs BTHWC）

### 6.2 Frame Batching（帧批处理）

用于编码/解码长视频，避免一次性处理所有帧。

**PyTorch 实现**（encode 时）：

```python
def _encode(self, x):
    frame_batch_size = self.num_sample_frames_batch_size  # 8
    num_batches = max(num_frames // frame_batch_size, 1)
    conv_cache = None
    enc = []
    
    for i in range(num_batches):
        # 计算当前批次的帧范围
        remaining_frames = num_frames % frame_batch_size
        start_frame = frame_batch_size * i + (0 if i == 0 else remaining_frames)
        end_frame = frame_batch_size * (i + 1) + remaining_frames
        
        # 提取帧批次
        x_intermediate = x[:, :, start_frame:end_frame]
        
        # 编码（携带 conv_cache）
        x_intermediate, conv_cache = self.encoder(x_intermediate, conv_cache=conv_cache)
        enc.append(x_intermediate)
    
    return torch.cat(enc, dim=2)
```

**Flax 实现**（完全相同）：

```python
def _encode(self, x, deterministic=True):
    frame_batch_size = self.num_sample_frames_batch_size  # 8
    num_batches = max(num_frames // frame_batch_size, 1)
    conv_cache = None
    enc = []
    
    for i in range(num_batches):
        # ... (同 PyTorch)
        x_intermediate = x[:, start_frame:end_frame, :, :, :]
        x_intermediate, conv_cache = self.encoder(
            x_intermediate, conv_cache=conv_cache, deterministic=deterministic
        )
        enc.append(x_intermediate)
    
    return jnp.concatenate(enc, axis=1)
```

### 6.3 逐帧解码（Frame-by-Frame Decoding）

这是 Flax 版本的**重要创新**，用于解决解码时的 OOM 问题。

**问题**：
- PyTorch 版本：每批处理 2 帧潜在表示 → 8 帧视频
- 内存需求：~40GB（超过 TPU v6e 的 32GB 限制）

**解决方案**：
- Flax 版本：每批处理 **1 帧**潜在表示 → 4 帧视频
- 使用 `FlaxCogVideoXCache` 管理所有 CausalConv3d 层的缓存

**实现**（第 1955-2069 行）：

```python
def _decode(self, z, zq, deterministic=True):
    batch_size, num_frames, height, width, num_channels = z.shape
    
    # 创建缓存管理器
    feat_cache_manager = FlaxCogVideoXCache(self.decoder)
    
    # 应用 post_quant_conv（整体）
    if self.post_quant_conv is not None:
        z = self.post_quant_conv(z)
    
    # 逐帧解码
    decoded_frames_list = []
    
    for i in range(num_frames):
        # 每帧重置索引（不清空缓存）
        feat_cache_manager._conv_idx = [0]
        
        # 提取当前帧
        z_frame = z[:, i:i+1, :, :, :]
        zq_frame = zq[:, i:i+1, :, :, :]
        
        # 解码（使用共享缓存）
        decoded_frame, _ = self.decoder(
            z_frame, zq_frame,
            feat_cache=feat_cache_manager._feat_map,
            feat_idx=feat_cache_manager._conv_idx,
            deterministic=deterministic
        )
        
        decoded_frames_list.append(decoded_frame)
    
    # 拼接所有帧
    decoded = jnp.concatenate(decoded_frames_list, axis=1)
    
    return decoded
```

**缓存管理**（第 1704-1746 行）：

```python
class FlaxCogVideoXCache:
    def __init__(self, decoder_module):
        self.decoder_module = decoder_module
        self.clear_cache()
    
    def clear_cache(self):
        # 计算 decoder 中的 CausalConv3d 层数量
        self._conv_num = self._count_causal_conv3d(self.decoder_module)
        self._conv_idx = [0]  # 当前索引
        self._feat_map = [None] * self._conv_num  # 缓存列表
    
    @staticmethod
    def _count_causal_conv3d(module):
        count = 0
        node_types = nnx.graph.iter_graph([module])
        for _, value in node_types:
            if isinstance(value, FlaxCogVideoXCausalConv3d):
                count += 1
        return count
```

**优势**：
- ✅ 内存占用减半（1 帧 vs 2 帧）
- ✅ 支持任意长度的视频
- ✅ 缓存在帧间共享，保持时序连续性

---

## 7. 关键差异总结

### 7.1 数据格式

| 维度 | PyTorch | JAX/Flax | 转换 |
|------|---------|----------|------|
| 视频 | (B, C, T, H, W) | (B, T, H, W, C) | `x.transpose(0, 2, 3, 4, 1)` |
| 图像 | (B, C, H, W) | (B, H, W, C) | `x.transpose(0, 2, 3, 1)` |
| 时间嵌入 | (B, C, 1, 1, 1) | (B, 1, 1, 1, C) | 广播维度不同 |

### 7.2 卷积权重

| 类型 | PyTorch Shape | JAX Shape | 转换代码 |
|------|---------------|-----------|----------|
| Conv3d | (O, I, T, H, W) | (T, H, W, I, O) | `w.transpose(2,3,4,1,0)` |
| Conv2d | (O, I, H, W) | (H, W, I, O) | `w.transpose(2,3,1,0)` |
| Linear | (O, I) | (I, O) | `w.transpose(1,0)` |

### 7.3 API 差异

| 功能 | PyTorch | JAX/Flax |
|------|---------|----------|
| 激活函数 | `F.silu(x)` | `jax.nn.silu(x)` |
| 插值 | `F.interpolate(x, scale_factor=2)` | `jax.image.resize(x, new_shape, method='nearest')` |
| Padding | `F.pad(x, (w,w,h,h,t,0), mode='replicate')` | `jnp.pad(x, [(0,0),(t,0),(h,h),(w,w),(0,0)], mode='edge')` |
| Concatenate | `torch.cat([a, b], dim=2)` | `jnp.concatenate([a, b], axis=1)` |
| Dropout | `nn.Dropout(rate)` | `nnx.Dropout(rate=rate)` + `deterministic` 参数 |

### 7.4 内存优化策略

| 策略 | PyTorch | Flax |
|------|---------|------|
| 编码批大小 | 8 帧/批 | 8 帧/批 ✅ |
| 解码批大小 | 2 帧/批 | **1 帧/批** ⚠️ |
| 缓存机制 | `conv_cache` 字典 | `feat_cache` 列表 + `feat_idx` 索引 |
| 分块处理 | `CogVideoXSafeConv3d` | 依赖 XLA 自动优化 |
| Tiling | ✅ 支持 | ✅ 支持 |

### 7.5 模型结构对比

| 组件 | PyTorch 类 | Flax 类 | 主要差异 |
|------|------------|---------|----------|
| 基础卷积 | `nn.Conv3d` | `nnx.Conv` | 数据格式、padding 格式 |
| 因果卷积 | `CogVideoXCausalConv3d` | `FlaxCogVideoXCausalConv3d` | 双缓存模式 |
| 组归一化 | `nn.GroupNorm` | `FlaxGroupNorm` | Channel-last 计算 |
| 空间归一化 | `CogVideoXSpatialNorm3D` | `FlaxCogVideoXSpatialNorm3D` | 插值方法 |
| ResNet 块 | `CogVideoXResnetBlock3D` | `FlaxCogVideoXResnetBlock3D` | Dropout 控制 |
| 编码器 | `CogVideoXEncoder3D` | `FlaxCogVideoXEncoder3D` | 完全一致 |
| 解码器 | `CogVideoXDecoder3D` | `FlaxCogVideoXDecoder3D` | 逐帧解码 |
| 主 VAE | `AutoencoderKLCogVideoX` | `FlaxAutoencoderKLCogVideoX` | 配置分离、权重转换 |

---

## 8. 总结

### 8.1 迁移要点

1. **数据格式转换**：所有输入/输出从 `BCTHW` 转为 `BTHWC`
2. **权重转换**：卷积核从 `(O,I,...)` 转为 `(...,I,O)`
3. **API 适配**：PyTorch → JAX/NNX 的函数映射
4. **缓存机制**：支持双模式（向后兼容 + 新的逐帧解码）
5. **内存优化**：解码时每批 1 帧（而非 2 帧）

### 8.2 性能优化

- ✅ 使用 `jax.image.resize` 代替 `F.interpolate`
- ✅ GroupNorm 直接在 channel-last 格式计算（避免转置）
- ✅ 逐帧解码避免 OOM
- ✅ 依赖 XLA 编译器优化内存分配

### 8.3 功能完整性

| 功能 | PyTorch | Flax | 状态 |
|------|---------|------|------|
| Encode | ✅ | ✅ | 完全一致 |
| Decode | ✅ | ✅ | 更优（逐帧） |
| Tiling | ✅ | ✅ | 完全一致 |
| Frame Batching | ✅ | ✅ | 完全一致 |
| from_pretrained | ✅ | ✅ | 自动转换权重 |
| Gradient Checkpointing | ✅ | ❌ | 未实现 |

---

**文档版本**：v1.0  
**最后更新**：2025-11-06  
**作者**：Based on CogVideoX PyTorch and Flax implementations
