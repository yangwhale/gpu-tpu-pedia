# PyTorch vs JAX 实现细节对比

## 总体架构对比

### 数据格式
| 维度 | PyTorch | JAX/Flax |
|------|---------|----------|
| 视频 | (B, C, T, H, W) | (B, T, H, W, C) |
| 2D | (B, C, H, W) | (B, H, W, C) |
| 1D | (B, C, L) | (B, L, C) |

---

## 1. CausalConv3d 对比

### PyTorch: `CogVideoXCausalConv3d`
```python
# diffusers/src/diffusers/models/autoencoders/autoencoder_kl_cogvideox.py:69-147
class CogVideoXCausalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, pad_mode="constant"):
        # 使用 CogVideoXSafeConv3d (继承自 nn.Conv3d)
        self.conv = CogVideoXSafeConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,  # (T, H, W)
            stride=stride if isinstance(stride, tuple) else (stride, 1, 1),
            dilation=(dilation, 1, 1),
            padding=0 if self.pad_mode == "replicate" else self.const_padding_conv3d,
            padding_mode="zeros",
        )
```

**关键点**：
- ✅ 使用 3D 卷积
- ✅ Padding: spatial (0, width_pad, height_pad)
- ✅ Stride: (temporal_stride, 1, 1)
- ✅ 返回 conv_cache

### JAX: `FlaxCogVideoXCausalConv3d`
```python
# diffusers-tpu-chris/src/diffusers/models/autoencoders/autoencoder_kl_cogvideox_flax.py:260-371
class FlaxCogVideoXCausalConv3d(nnx.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, pad_mode="constant", rngs=None):
        self.conv = FlaxConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride_tuple,  # (stride, 1, 1)
            padding=0 if self.pad_mode == "replicate" else const_padding_conv3d,
            rngs=rngs,
        )
```

**对比结果**: ✅ **完全一致**

---

## 2. Downsampler 对比

### PyTorch: `CogVideoXDownsample3D`
```python
# diffusers/src/diffusers/models/downsampling.py:288-353
class CogVideoXDownsample3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=0, compress_time=False):
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.compress_time = compress_time
    
    def forward(self, x):
        if self.compress_time:
            # 时间压缩: F.avg_pool1d
            ...
        
        # 手动 padding
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        
        # 应用 Conv2d (not Conv3d!)
        x = self.conv(x)  # 在 (B*frames, C, H, W) 上操作
```

**关键点**：
- ⚠️ 使用 **Conv2d**（不是 Conv3d！）
- ⚠️ 手动 padding: (0, 1, 0, 1)
- ✅ 默认 padding=0
- ✅ 时间压缩: F.avg_pool1d

### JAX: `FlaxCogVideoXDownBlock3D.downsamplers`
```python
# diffusers-tpu-chris/src/diffusers/models/autoencoders/autoencoder_kl_cogvideox_flax.py:745-834
# 使用 FlaxConv2d (✅ 正确)
downsampler = FlaxConv2d(
    out_channels, out_channels,
    kernel_size=3,
    stride=2,
    padding=0,  # ✅ No padding in conv
    rngs=rngs
)

# 在 forward 中手动添加 padding
pad_width = [
    (0, 0),  # batch
    (0, 0),  # time
    (0, 1),  # height: pad bottom
    (0, 1),  # width: pad right
    (0, 0),  # channels
]
hidden_states = jnp.pad(hidden_states, pad_width, mode='constant', constant_values=0)
```

**对比结果**: ✅ **完全一致**

---

## 3. Upsampler 对比

### PyTorch: `CogVideoXUpsample3D`
```python
# diffusers/src/diffusers/models/upsampling.py:359-420
class CogVideoXUpsample3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, compress_time=False):
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.compress_time = compress_time
    
    def forward(self, inputs):
        if self.compress_time:
            if inputs.shape[2] > 1 and inputs.shape[2] % 2 == 1:
                # 奇数帧: 分离第一帧
                x_first, x_rest = inputs[:, :, 0], inputs[:, :, 1:]
                x_first = F.interpolate(x_first, scale_factor=2.0)  # 2D 插值
                x_rest = F.interpolate(x_rest, scale_factor=2.0)    # 3D 插值
                x_first = x_first[:, :, None, :, :]
                inputs = torch.cat([x_first, x_rest], dim=2)
            elif inputs.shape[2] > 1:
                # 偶数帧: 3D 插值
                inputs = F.interpolate(inputs, scale_factor=2.0)
            else:
                # 单帧
                inputs = inputs.squeeze(2)
                inputs = F.interpolate(inputs, scale_factor=2.0)
                inputs = inputs[:, :, None, :, :]
        else:
            # 仅 2D 插值
            b, c, t, h, w = inputs.shape
            inputs = inputs.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            inputs = F.interpolate(inputs, scale_factor=2.0)
            inputs = inputs.reshape(b, t, c, *inputs.shape[2:]).permute(0, 2, 1, 3, 4)
        
        # 应用 Conv2d
        b, c, t, h, w = inputs.shape
        inputs = inputs.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        inputs = self.conv(inputs)
        inputs = inputs.reshape(b, t, *inputs.shape[1:]).permute(0, 2, 1, 3, 4)
```

**关键点**：
- ⚠️ 使用 **Conv2d**（不是 Conv3d！）
- ⚠️ padding=1（不是0！）
- ✅ compress_time: 3D interpolate (时间+空间)
- ✅ 非 compress_time: 2D interpolate (仅空间)

### JAX: `FlaxCogVideoXUpBlock3D.upsamplers`
```python
# diffusers-tpu-chris/src/diffusers/models/autoencoders/autoencoder_kl_cogvideox_flax.py:918-1024
upsampler = FlaxConv2d(
    out_channels, out_channels,
    kernel_size=3,
    stride=1,
    padding=upsample_padding,  # default is 1 ✅
    rngs=rngs
)

# compress_time 逻辑
if self.compress_time:
    if T > 1 and T % 2 == 1:
        # 奇数帧
        first_frame = hidden_states[:, 0, :, :, :]
        rest_frames = hidden_states[:, 1:, :, :, :]
        
        first_frame = jax.image.resize(first_frame, (B, H * 2, W * 2, C), method='nearest')
        first_frame = first_frame[:, None, :, :, :]
        
        rest_frames = jax.image.resize(rest_frames, (B, 2 * (T-1), H * 2, W * 2, C), method='nearest')
        
        hidden_states = jnp.concatenate([first_frame, rest_frames], axis=1)
    elif T > 1:
        # 偶数帧
        hidden_states = jax.image.resize(hidden_states, (B, T * 2, H * 2, W * 2, C), method='nearest')
    else:
        # 单帧
        hidden_states = hidden_states.reshape(B, H, W, C)
        hidden_states = jax.image.resize(hidden_states, (B, H * 2, W * 2, C), method='nearest')
        hidden_states = hidden_states[:, None, :, :, :]
else:
    # 仅 2D
    hidden_states = hidden_states.reshape(B * T, H, W, C)
    hidden_states = jax.image.resize(hidden_states, (B * T, H * 2, W * 2, C), method='nearest')
    hidden_states = hidden_states.reshape(B, T, H * 2, W * 2, C)
```

**对比结果**: ✅ **完全一致**

---

## 4. GroupNorm 对比

### PyTorch: `nn.GroupNorm`
```python
nn.GroupNorm(num_channels=channels, num_groups=groups, eps=1e-6, affine=True)
```

### JAX: `FlaxGroupNorm`
```python
# diffusers-tpu-chris/src/diffusers/models/autoencoders/autoencoder_kl_cogvideox_flax.py:374-449
class FlaxGroupNorm(nnx.Module):
    def __init__(self, num_groups, num_channels, epsilon=1e-6, rngs=None):
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.epsilon = epsilon
        self.scale = nnx.Param(jnp.ones((num_channels,)))
        self.bias = nnx.Param(jnp.zeros((num_channels,)))
```

**潜在问题**:
- ⚠️ epsilon 值相同吗？PyTorch 默认 1e-5，CogVideoX 使用 1e-6
- ⚠️ 归一化轴是否完全一致？

---

## 5. SpatialNorm3D 对比

### PyTorch: `CogVideoXSpatialNorm3D`
```python
# diffusers/src/diffusers/models/autoencoders/autoencoder_kl_cogvideox.py:149-197
class CogVideoXSpatialNorm3D(nn.Module):
    def __init__(self, f_channels, zq_channels, groups=32):
        self.norm_layer = nn.GroupNorm(num_channels=f_channels, num_groups=groups, eps=1e-6, affine=True)
        self.conv_y = CogVideoXCausalConv3d(zq_channels, f_channels, kernel_size=1, stride=1)
        self.conv_b = CogVideoXCausalConv3d(zq_channels, f_channels, kernel_size=1, stride=1)
    
    def forward(self, f, zq, conv_cache=None):
        # 处理奇数帧
        if f.shape[2] > 1 and f.shape[2] % 2 == 1:
            f_first, f_rest = f[:, :, :1], f[:, :, 1:]
            f_first_size, f_rest_size = f_first.shape[-3:], f_rest.shape[-3:]
            z_first, z_rest = zq[:, :, :1], zq[:, :, 1:]
            z_first = F.interpolate(z_first, size=f_first_size)
            z_rest = F.interpolate(z_rest, size=f_rest_size)
            zq = torch.cat([z_first, z_rest], dim=2)
        else:
            zq = F.interpolate(zq, size=f.shape[-3:])
        
        conv_y, new_conv_cache["conv_y"] = self.conv_y(zq, conv_cache=conv_cache.get("conv_y"))
        conv_b, new_conv_cache["conv_b"] = self.conv_b(zq, conv_cache=conv_cache.get("conv_b"))
        
        norm_f = self.norm_layer(f)
        new_f = norm_f * conv_y + conv_b
```

### JAX: `FlaxCogVideoXSpatialNorm3D`
```python
# diffusers-tpu-chris/src/diffusers/models/autoencoders/autoencoder_kl_cogvideox_flax.py:452-528
# 实现逻辑完全相同
```

**对比结果**: ✅ **逻辑一致**

---

## 6. 插值方法对比

### PyTorch: `F.interpolate`
```python
F.interpolate(x, scale_factor=2.0)  # 默认 mode='nearest'
F.interpolate(x, size=target_size)
```

### JAX: `jax.image.resize`
```python
jax.image.resize(x, shape, method='nearest')
```

**潜在问题**:
- ⚠️ 'nearest' 插值的边界处理可能不同
- ⚠️ PyTorch 的 F.interpolate 默认 align_corners=False

---

## 7. 激活函数对比

### PyTorch
```python
nn.SiLU()  # 或 F.silu()
```

### JAX
```python
jax.nn.silu()
```

**对比结果**: ✅ **数学定义相同** (silu(x) = x * sigmoid(x))

---

## 总结：已确认的差异点

### ✅ 已正确实现
1. Downsampler 使用 Conv2d + 手动 padding
2. Upsampler 使用 Conv2d + padding=1
3. CausalConv3d 使用 Conv3d
4. compress_time 逻辑完全一致
5. 形状转换全部正确

### ⚠️ 需要进一步检查的细节
1. **GroupNorm epsilon**: PyTorch 默认 1e-5，CogVideoX 使用 1e-6
2. **插值方法**: jax.image.resize vs F.interpolate 的细微差异
3. **浮点精度**: 计算顺序可能导致的累积误差
4. **Padding 模式**: 'constant' vs 'zeros' 的一致性

### 🔍 数值差异来源分析
当前误差水平：
- 编码 MAE: ~0.46-0.57
- 解码 MAE: ~0.31
- 最大误差: ~1.8-2.9

这些误差可能来自：
1. 深度网络的累积误差
2. GroupNorm 的数值稳定性
3. 插值方法的实现差异
4. 浮点运算顺序

**建议的优化方向**：
1. 确保所有 epsilon 值完全一致
2. 验证 GroupNorm 的计算轴
3. 测试不同的插值方法
4. 添加更详细的逐层数值对比