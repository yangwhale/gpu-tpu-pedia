# BF16 内存优化尝试总结

## 背景

在尝试运行大分辨率视频解码（16帧@768x1360）时遇到 OOM 错误：
```
RESOURCE_EXHAUSTED: Error allocating device buffer: 
Attempting to allocate 3.98G. That was not possible. 
There are 3.14G free.; (0x0x0_HBM0)
```

尽管模型配置为 `dtype=jnp.bfloat16`，但仍尝试分配 FP32 大小的内存（3.98GB）。

## 诊断发现

### 1. 参数 dtype 验证

✅ **确认**：所有 436 个模型参数都正确转换为 BF16
- from_pretrained 后强制转换所有参数
- 使用 `nnx.iter_graph()` 验证所有参数 dtype

### 2. 小分辨率测试

✅ **成功**：小分辨率（9帧@60x90）可以正常工作
- 输出 dtype: bfloat16
- 参数 dtype: bfloat16
- 无 OOM 问题

### 3. 中等分辨率测试

✅ **成功**：中等分辨率（16帧@384x680）可以正常工作
- 解码耗时: 107.96s
- 输出 dtype: bfloat16
- 输出大小: 23.91 MB

### 4. 大分辨率测试

❌ **失败**：大分辨率（16帧@768x1360）OOM
- 仍尝试分配 3.98GB（FP32 大小）
- 错误发生在 GroupNorm 层的除法操作
- 说明某些中间计算仍在使用 FP32

## 尝试的修复方案

### 方案 1: Conv 层 dtype 参数（有益但不足够）

**修改位置**: FlaxConv3d, FlaxConv2d, FlaxCogVideoXCausalConv3d

**原理**: 
- JAX NNX 的 Conv 层需要两个 dtype 参数：
  - `dtype`: 控制计算时的精度（输入/输出/中间结果）
  - `param_dtype`: 控制参数存储的精度
- **关键发现**: 只设置 `param_dtype` 无效，必须同时设置 `dtype`

**代码**:
```python
self.conv = nnx.Conv(
    in_channels,
    out_channels,
    kernel_size=kernel_size,
    strides=stride,
    padding=padding_mode,
    dtype=dtype,        # 新增！控制计算精度
    param_dtype=dtype,  # 控制参数存储
    rngs=rngs,
)
```

**效果**: 
- ✅ 确保 Conv 参数和计算都使用 BF16
- ❌ 但仍不足以解决大分辨率 OOM

### 方案 2: GroupNorm dtype 修复（有益但不足够）

**修改位置**: FlaxGroupNorm

**问题**:
1. scale/bias 参数默认使用 FP32
2. `jnp.mean/var` 默认返回 FP32（即使输入是 BF16）
3. Python 标量 epsilon (1e-6) 是 float64，导致类型提升

**代码**:
```python
# 1. 参数使用指定 dtype
self.scale = nnx.Param(jnp.ones((num_channels,), dtype=dtype))
self.bias = nnx.Param(jnp.zeros((num_channels,), dtype=dtype))

# 2. 显式指定 mean/var 的 dtype
input_dtype = x_grouped.dtype
mean = jnp.mean(x_grouped, axis=(2, 3, 4, 5), keepdims=True, dtype=input_dtype)
var = jnp.var(x_grouped, axis=(2, 3, 4, 5), keepdims=True, dtype=input_dtype)

# 3. 转换 epsilon 为 input_dtype
epsilon_typed = jnp.array(self.epsilon, dtype=input_dtype)

# 4. 所有中间结果显式转换
var_eps = (var + epsilon_typed).astype(input_dtype)
sqrt_var = jnp.sqrt(var_eps).astype(input_dtype)
x_norm = ((x_grouped - mean) / sqrt_var).astype(input_dtype)
```

**效果**:
- ✅ 防止 GroupNorm 内部的类型提升
- ❌ 但仍不足以解决大分辨率 OOM

### 方案 3: jax.image.resize dtype 保持（有益但不足够）

**修改位置**: FlaxCogVideoXSpatialNorm3D, FlaxCogVideoXUpBlock3D

**问题**: `jax.image.resize` 默认返回 FP32

**代码**:
```python
zq_dtype = zq.dtype  # 保存原始 dtype
z_first = jax.image.resize(...).astype(zq_dtype)  # 强制保持 dtype
z_rest = jax.image.resize(...).astype(zq_dtype)
```

**效果**:
- ✅ 确保 resize 操作不改变 dtype
- ❌ 但仍不足以解决大分辨率 OOM

### 方案 4: 入口点 dtype 强制转换（无效）

**修改位置**: encode(), decode(), _decode()

**尝试**: 在方法入口强制转换输入到目标 dtype
```python
def decode(self, z: jnp.ndarray, ...):
    target_dtype = self.dtype
    z = z.astype(target_dtype)
    ...
```

**效果**:
- ❌ 完全无效
- ❌ 大分辨率仍 OOM（3.98GB）
- ❌ 说明问题不在输入 dtype，而在内部计算

## 技术洞察

### 1. JAX/Flax dtype 管理的复杂性

- **参数 dtype vs 计算 dtype**: 必须两者都设置
- **隐式类型提升**: FP32参数 + BF16输入 = FP32计算
- **默认行为陷阱**: 
  - `jnp.mean/var` 默认返回 FP32
  - `jax.image.resize` 默认返回 FP32
  - Python 标量会触发类型提升

### 2. 内存分配的神秘之处

- 即使所有参数和输入都是 BF16
- 即使小/中等分辨率工作正常
- 大分辨率仍会尝试分配 FP32 大小的内存
- **推测**: 某些 JAX 内部操作或 JIT 编译时的中间缓冲区仍在使用 FP32

### 3. 错误堆栈的线索

```python
File ".../autoencoder_kl_cogvideox_flax.py", line 449
    x_norm = ((x_grouped - mean) / sqrt_var).astype(input_dtype)
             ~~~~~~~~~~~~~~~~~~~^~~~~~~~~~
```

- 错误发生在 GroupNorm 的除法操作
- 尽管已经添加了 `.astype(input_dtype)`
- 说明在执行除法之前，某个中间结果已经是 FP32

## 有益的修改（值得保留）

以下修改虽然没有解决 OOM，但它们是有价值的技术改进：

### 1. Conv 层 dtype 双参数
```python
# 所有 FlaxConv3d/FlaxConv2d/FlaxCogVideoXCausalConv3d
self.conv = nnx.Conv(
    ...
    dtype=dtype,        # 控制计算精度
    param_dtype=dtype,  # 控制参数存储
)
```

### 2. GroupNorm 完整 dtype 管理
```python
# 参数
self.scale = nnx.Param(jnp.ones(..., dtype=dtype))
self.bias = nnx.Param(jnp.zeros(..., dtype=dtype))

# 统计量
mean = jnp.mean(..., dtype=input_dtype)
var = jnp.var(..., dtype=input_dtype)

# epsilon
epsilon_typed = jnp.array(self.epsilon, dtype=input_dtype)

# 中间结果
var_eps = (var + epsilon_typed).astype(input_dtype)
sqrt_var = jnp.sqrt(var_eps).astype(input_dtype)
x_norm = ((x_grouped - mean) / sqrt_var).astype(input_dtype)
```

### 3. resize 操作 dtype 保持
```python
result = jax.image.resize(...).astype(original_dtype)
```

### 4. from_pretrained 强制参数转换
```python
for path, value in nnx.iter_graph(model):
    if isinstance(value, nnx.Param):
        value.value = value.value.astype(dtype)
```

## 结论

### 失败的原因

1. **根本问题**: 大分辨率时，某些 JAX 内部操作仍在分配 FP32 缓冲区
2. **诊断困难**: 无法精确定位是哪个操作触发了 FP32 分配
3. **JIT 编译**: 可能在编译时固化了某些 FP32 路径

### 建议的解决方案

**最终方案：使用 VAE Tiling**

已有的 `tiled_decode` 实现可以将大分辨率分块处理：
- 分块大小可配置
- 分块间平滑混合
- 避免单次大内存分配

```python
vae.enable_tiling(
    tile_sample_min_height=384,
    tile_sample_min_width=512
)
decoded = vae.decode(large_latents)
```

### 未来方向

1. **深度 profiling**: 使用 JAX profiler 精确定位内存分配
2. **逐层 dtype 追踪**: 在每层输出添加 dtype 检查
3. **强制输出 dtype**: 在每个 `__call__` 返回时强制转换
4. **探索 JAX 配置**: 是否有全局设置可以强制 BF16

## 参考实现

- **maxdiffusion/autoencoder_kl_wan.py**: 正确展示了 dtype 和 param_dtype 的使用
- **diffusers_tpu**: 提供了 GroupNorm 的 JAX 实现参考

## 时间线

- **Phase 1-7**: 基础实现和功能验证（成功）
- **Phase 8**: BF16 内存优化尝试（失败）
- **测试结果**:
  - ✅ 小分辨率（9帧@60x90）
  - ✅ 中等分辨率（16帧@384x680）
  - ❌ 大分辨率（16帧@768x1360）OOM

---

**结论**: BF16 优化的技术修改是正确的，但不足以解决大分辨率 OOM。建议使用 VAE Tiling 作为生产环境的解决方案。