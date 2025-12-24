# TorchAx Core ATen 操作符目录

> 本文档列出了 TorchAx 中实现的所有 PyTorch Core ATen 操作符及其 JAX 实现。
>
> 源文件: [`torchax/torchax/ops/jaten.py`](../../../../torchax/torchax/ops/jaten.py) (5801 行)

## 概述

TorchAx 实现了 **382 个 Core ATen 操作符**，这些是 PyTorch 最核心的张量运算。通过将这些操作符映射到 JAX 实现，TorchAx 可以让 PyTorch 代码无缝运行在 TPU 上。

---

## 操作符分类目录

### 1. 张量形状操作 (Shape Operations)

| 操作符 | JAX 实现 | 说明 | 源码位置 |
|--------|----------|------|----------|
| `view_copy` | `jnp.reshape` | 改变张量视图 | [L62](../../../../torchax/torchax/ops/jaten.py#L62) |
| `view` | `jnp.reshape` | 改变张量视图 | [L62](../../../../torchax/torchax/ops/jaten.py#L62) |
| `_unsafe_view` | `jnp.reshape` | 不安全视图操作 | [L62](../../../../torchax/torchax/ops/jaten.py#L62) |
| `reshape` | `jnp.reshape` | 重塑张量形状 | [L62](../../../../torchax/torchax/ops/jaten.py#L62) |
| `expand` | `jnp.broadcast_to` | 扩展张量维度 | [L704](../../../../torchax/torchax/ops/jaten.py#L704) |
| `expand_copy` | `jnp.broadcast_to` | 扩展张量维度副本 | [L704](../../../../torchax/torchax/ops/jaten.py#L704) |
| `squeeze` | `jnp.squeeze` | 压缩单一维度 | [L1020](../../../../torchax/torchax/ops/jaten.py#L1020) |
| `squeeze_copy` | `jnp.squeeze` | 压缩单一维度副本 | [L1020](../../../../torchax/torchax/ops/jaten.py#L1020) |
| `unsqueeze` | `jnp.expand_dims` | 扩展一个维度 | [L840](../../../../torchax/torchax/ops/jaten.py#L840) |
| `unsqueeze_copy` | `jnp.expand_dims` | 扩展一个维度副本 | [L840](../../../../torchax/torchax/ops/jaten.py#L840) |
| `permute` | `jnp.transpose` | 维度重排序 | [L833](../../../../torchax/torchax/ops/jaten.py#L833) |
| `permute_copy` | `jnp.transpose` | 维度重排序副本 | [L833](../../../../torchax/torchax/ops/jaten.py#L833) |
| `transpose` | `jnp.swapaxes` | 交换两个维度 | [L348](../../../../torchax/torchax/ops/jaten.py#L348) |
| `transpose_copy` | `jnp.swapaxes` | 交换两个维度副本 | [L348](../../../../torchax/torchax/ops/jaten.py#L348) |
| `t` | `jnp.transpose` | 转置 | [L343](../../../../torchax/torchax/ops/jaten.py#L343) |
| `numpy_T` | `jnp.transpose` | NumPy 风格转置 | [L312](../../../../torchax/torchax/ops/jaten.py#L312) |
| `flatten` | `jnp.reshape` | 展平张量 | [L4975](../../../../torchax/torchax/ops/jaten.py#L4975) |
| `narrow` | `jax.lax.dynamic_slice_in_dim` | 窄化张量 | [L4969](../../../../torchax/torchax/ops/jaten.py#L4969) |
| `narrow_copy` | `jax.lax.dynamic_slice_in_dim` | 窄化张量副本 | [L4969](../../../../torchax/torchax/ops/jaten.py#L4969) |
| `slice` | Python 切片 | 切片操作 | [L363](../../../../torchax/torchax/ops/jaten.py#L363) |
| `slice_copy` | Python 切片 | 切片操作副本 | [L363](../../../../torchax/torchax/ops/jaten.py#L363) |
| `split` | 自定义切片 | 分割张量 | [L802](../../../../torchax/torchax/ops/jaten.py#L802) |
| `split_copy` | 自定义切片 | 分割张量副本 | [L802](../../../../torchax/torchax/ops/jaten.py#L802) |
| `split_with_sizes` | 自定义切片 | 按大小分割 | [L802](../../../../torchax/torchax/ops/jaten.py#L802) |
| `stack` | `jnp.stack` | 堆叠张量 | [L435](../../../../torchax/torchax/ops/jaten.py#L435) |
| `cat` | `jnp.concatenate` | 连接张量 | [L1233](../../../../torchax/torchax/ops/jaten.py#L1233) |
| `movedim` | `jnp.moveaxis` | 移动维度 | [L2633](../../../../torchax/torchax/ops/jaten.py#L2633) |
| `unfold` | 自定义实现 | 展开维度 | [L5698](../../../../torchax/torchax/ops/jaten.py#L5698) |

### 2. 基础算术运算 (Arithmetic Operations)

| 操作符 | JAX 实现 | 说明 | 源码位置 |
|--------|----------|------|----------|
| `add.Tensor` | `x + y` | 加法（张量） | [L82](../../../../torchax/torchax/ops/jaten.py#L82) |
| `add.Scalar` | `x + y` | 加法（标量） | [L82](../../../../torchax/torchax/ops/jaten.py#L82) |
| `sub.Tensor` | `x - y` | 减法（张量） | [L306](../../../../torchax/torchax/ops/jaten.py#L306) |
| `sub.Scalar` | `x - y` | 减法（标量） | [L306](../../../../torchax/torchax/ops/jaten.py#L306) |
| `mul.Tensor` | `x * y` | 乘法（张量） | [L332](../../../../torchax/torchax/ops/jaten.py#L332) |
| `mul.Scalar` | `x * y` | 乘法（标量） | [L332](../../../../torchax/torchax/ops/jaten.py#L332) |
| `div` | `x / y` | 除法 | [L507](../../../../torchax/torchax/ops/jaten.py#L507) |
| `true_divide` | `x / y` | 真除法 | [L533](../../../../torchax/torchax/ops/jaten.py#L533) |
| `floor_divide` | `jnp.floor_divide` | 整除 | [L5686](../../../../torchax/torchax/ops/jaten.py#L5686) |
| `pow` | `jnp.power` | 幂运算 | [L482](../../../../torchax/torchax/ops/jaten.py#L482) |
| `neg` | `-1 * x` | 取负 | [L3099](../../../../torchax/torchax/ops/jaten.py#L3099) |
| `abs` | `jnp.abs` | 绝对值 | [L2304](../../../../torchax/torchax/ops/jaten.py#L2304) |
| `sqrt` | `jnp.sqrt` | 平方根 | [L1725](../../../../torchax/torchax/ops/jaten.py#L1725) |
| `rsqrt` | `jax.lax.rsqrt` | 平方根倒数 | [L698](../../../../torchax/torchax/ops/jaten.py#L698) |
| `reciprocal` | `1 / a` | 倒数 | [L2237](../../../../torchax/torchax/ops/jaten.py#L2237) |
| `sign` | `jnp.sign` | 符号函数 | [L1846](../../../../torchax/torchax/ops/jaten.py#L1846) |
| `signbit` | `jnp.signbit` | 符号位 | [L1852](../../../../torchax/torchax/ops/jaten.py#L1852) |
| `fmod` | 自定义实现 | 浮点取模 | [L2733](../../../../torchax/torchax/ops/jaten.py#L2733) |
| `remainder` | `inputs % other` | 取余 | [L3212](../../../../torchax/torchax/ops/jaten.py#L3212) |
| `ceil` | `jnp.ceil` | 向上取整 | [L1747](../../../../torchax/torchax/ops/jaten.py#L1747) |
| `floor` | `jnp.floor` | 向下取整 | [L2715](../../../../torchax/torchax/ops/jaten.py#L2715) |
| `trunc` | `jnp.trunc` | 截断 | [L117](../../../../torchax/torchax/ops/jaten.py#L117) |
| `round` | `jnp.round` | 四舍五入 | [L2281](../../../../torchax/torchax/ops/jaten.py#L2281) |
| `clamp` | `jnp.clip` | 截断到范围 | [L2501](../../../../torchax/torchax/ops/jaten.py#L2501) |
| `clamp_min` | `jnp.clip` | 下限截断 | [L2507](../../../../torchax/torchax/ops/jaten.py#L2507) |
| `minimum` | `jnp.minimum` | 元素级最小值 | [L1761](../../../../torchax/torchax/ops/jaten.py#L1761) |
| `maximum` | `jnp.maximum` | 元素级最大值 | [L2298](../../../../torchax/torchax/ops/jaten.py#L2298) |
| `fmax` | `jnp.fmax` | 忽略 NaN 的最大值 | [L2721](../../../../torchax/torchax/ops/jaten.py#L2721) |
| `fmin` | `jnp.fmin` | 忽略 NaN 的最小值 | [L2727](../../../../torchax/torchax/ops/jaten.py#L2727) |

### 3. 三角函数和双曲函数 (Trigonometric & Hyperbolic)

| 操作符 | JAX 实现 | 说明 | 源码位置 |
|--------|----------|------|----------|
| `sin` | `jnp.sin` | 正弦 | [L1486](../../../../torchax/torchax/ops/jaten.py#L1486) |
| `cos` | `jnp.cos` | 余弦 | [L2565](../../../../torchax/torchax/ops/jaten.py#L2565) |
| `tan` | `jnp.tan` | 正切 | [L1731](../../../../torchax/torchax/ops/jaten.py#L1731) |
| `sinh` | `jnp.sinh` | 双曲正弦 | [L1646](../../../../torchax/torchax/ops/jaten.py#L1646) |
| `cosh` | `jnp.cosh` | 双曲余弦 | [L2572](../../../../torchax/torchax/ops/jaten.py#L2572) |
| `tanh` | `jnp.tanh` | 双曲正切 | [L1739](../../../../torchax/torchax/ops/jaten.py#L1739) |
| `asin` | `jnp.arcsin` | 反正弦 | [L1753](../../../../torchax/torchax/ops/jaten.py#L1753) |
| `acos` | `jnp.arccos` | 反余弦 | [L1922](../../../../torchax/torchax/ops/jaten.py#L1922) |
| `atan` | `jnp.arctan` | 反正切 | [L1873](../../../../torchax/torchax/ops/jaten.py#L1873) |
| `atan2` | `jnp.arctan2` | 二参数反正切 | [L2397](../../../../torchax/torchax/ops/jaten.py#L2397) |
| `asinh` | `jnp.arcsinh` | 反双曲正弦 | [L1865](../../../../torchax/torchax/ops/jaten.py#L1865) |
| `acosh` | `jnp.arccosh` | 反双曲余弦 | [L2271](../../../../torchax/torchax/ops/jaten.py#L2271) |
| `atanh` | `jnp.arctanh` | 反双曲正切 | [L1680](../../../../torchax/torchax/ops/jaten.py#L1680) |
| `hypot` | `jnp.hypot` | 直角三角形斜边 | [L2801](../../../../torchax/torchax/ops/jaten.py#L2801) |

### 4. 指数和对数函数 (Exponential & Logarithmic)

| 操作符 | JAX 实现 | 说明 | 源码位置 |
|--------|----------|------|----------|
| `exp` | `jnp.exp` | 指数函数 | [L2665](../../../../torchax/torchax/ops/jaten.py#L2665) |
| `exp2` | `jnp.exp2` | 2 的幂 | [L2685](../../../../torchax/torchax/ops/jaten.py#L2685) |
| `expm1` | `jnp.expm1` | exp(x) - 1 | [L2675](../../../../torchax/torchax/ops/jaten.py#L2675) |
| `log` | `jnp.log` | 自然对数 | [L3013](../../../../torchax/torchax/ops/jaten.py#L3013) |
| `log10` | `jnp.log10` | 以 10 为底的对数 | [L3020](../../../../torchax/torchax/ops/jaten.py#L3020) |
| `log2` | `jnp.log2` | 以 2 为底的对数 | [L3033](../../../../torchax/torchax/ops/jaten.py#L3033) |
| `log1p` | `jnp.log1p` | log(1 + x) | [L3027](../../../../torchax/torchax/ops/jaten.py#L3027) |
| `logaddexp` | `jnp.logaddexp` | log(exp(a) + exp(b)) | [L3068](../../../../torchax/torchax/ops/jaten.py#L3068) |
| `logaddexp2` | `jnp.logaddexp2` | log2(2^a + 2^b) | [L3074](../../../../torchax/torchax/ops/jaten.py#L3074) |
| `logcumsumexp` | `jax.lax.cumlogsumexp` | 累积 log-sum-exp | [L3080](../../../../torchax/torchax/ops/jaten.py#L3080) |
| `logit` | `jnp.log(p/(1-p))` | logit 函数 | [L5664](../../../../torchax/torchax/ops/jaten.py#L5664) |

### 5. 矩阵运算 (Matrix Operations)

| 操作符 | JAX 实现 | 说明 | 源码位置 |
|--------|----------|------|----------|
| `mm` | `x @ y` | 矩阵乘法 | [L326](../../../../torchax/torchax/ops/jaten.py#L326) |
| `bmm` | `x @ y` | 批量矩阵乘法 | [L544](../../../../torchax/torchax/ops/jaten.py#L544) |
| `matmul` | `x @ y` | 通用矩阵乘法 | [L981](../../../../torchax/torchax/ops/jaten.py#L981) |
| `dot` | `jnp.dot` | 点积 | [L724](../../../../torchax/torchax/ops/jaten.py#L724) |
| `addmm` | `self + alpha * mat1 @ mat2` | 矩阵乘加 | [L987](../../../../torchax/torchax/ops/jaten.py#L987) |
| `addmv` | `self + alpha * mat1 @ mat2` | 矩阵向量乘加 | [L987](../../../../torchax/torchax/ops/jaten.py#L987) |
| `addbmm` | `einsum + add` | 批量矩阵乘加 | [L1006](../../../../torchax/torchax/ops/jaten.py#L1006) |
| `outer` | `jnp.outer` | 外积 | [L3709](../../../../torchax/torchax/ops/jaten.py#L3709) |
| `linear` | `input @ weight.T + bias` | 线性层 | [L5574](../../../../torchax/torchax/ops/jaten.py#L5574) |

### 6. 归约操作 (Reduction Operations)

| 操作符 | JAX 实现 | 说明 | 源码位置 |
|--------|----------|------|----------|
| `sum` | `jnp.sum` | 求和 | [L1717](../../../../torchax/torchax/ops/jaten.py#L1717) |
| `mean` | `jnp.mean` | 均值 | [L276](../../../../torchax/torchax/ops/jaten.py#L276) |
| `prod` | `jnp.prod` | 乘积 | [L3142](../../../../torchax/torchax/ops/jaten.py#L3142) |
| `min` | `jnp.min, jnp.argmin` | 最小值 | [L1450](../../../../torchax/torchax/ops/jaten.py#L1450) |
| `max` | `jnp.max, jnp.argmax` | 最大值 | [L2287](../../../../torchax/torchax/ops/jaten.py#L2287) |
| `amin` | `jnp.amin` | 沿维度最小值 | [L1476](../../../../torchax/torchax/ops/jaten.py#L1476) |
| `amax` | `jnp.amax` | 沿维度最大值 | [L2310](../../../../torchax/torchax/ops/jaten.py#L2310) |
| `argmin` | `jnp.argmin` | 最小值索引 | [L1481](../../../../torchax/torchax/ops/jaten.py#L1481) |
| `argmax` | `jnp.argmax` | 最大值索引 | [L2361](../../../../torchax/torchax/ops/jaten.py#L2361) |
| `any` | `jnp.any` | 任意为真 | [L2331](../../../../torchax/torchax/ops/jaten.py#L2331) |
| `var.correction` | `jnp.var` | 方差 | [L1497](../../../../torchax/torchax/ops/jaten.py#L1497) |
| `var_mean.correction` | `jnp.var, jnp.mean` | 方差和均值 | [L3539](../../../../torchax/torchax/ops/jaten.py#L3539) |
| `cumsum` | `jnp.cumsum` | 累积求和 | [L922](../../../../torchax/torchax/ops/jaten.py#L922) |
| `cumprod` | `jnp.cumprod` | 累积乘积 | [L932](../../../../torchax/torchax/ops/jaten.py#L932) |
| `cummax` | `jax.lax.associative_scan` | 累积最大值 | [L874](../../../../torchax/torchax/ops/jaten.py#L874) |
| `cummin` | `jax.lax.associative_scan` | 累积最小值 | [L898](../../../../torchax/torchax/ops/jaten.py#L898) |
| `mode` | `jax.scipy.stats.mode` | 众数 | [L1460](../../../../torchax/torchax/ops/jaten.py#L1460) |
| `median` | `jnp.quantile` | 中位数 | [L5122](../../../../torchax/torchax/ops/jaten.py#L5122) |
| `nanmedian` | `jnp.nanquantile` | 忽略 NaN 的中位数 | [L5139](../../../../torchax/torchax/ops/jaten.py#L5139) |

### 7. 激活函数 (Activation Functions)

| 操作符 | JAX 实现 | 说明 | 源码位置 |
|--------|----------|------|----------|
| `relu` | `jax.nn.relu` | ReLU | [L1228](../../../../torchax/torchax/ops/jaten.py#L1228) |
| `silu` | `jax.nn.silu` | SiLU/Swish | [L337](../../../../torchax/torchax/ops/jaten.py#L337) |
| `gelu` | `jax.nn.gelu` | GELU | [L1014](../../../../torchax/torchax/ops/jaten.py#L1014) |
| `sigmoid` | `jax.nn.sigmoid` | Sigmoid | [L1858](../../../../torchax/torchax/ops/jaten.py#L1858) |
| `tanh` | `jnp.tanh` | Tanh | [L1739](../../../../torchax/torchax/ops/jaten.py#L1739) |
| `softmax` | `jax.nn.softmax` | Softmax | [L440](../../../../torchax/torchax/ops/jaten.py#L440) |
| `_softmax` | `jax.nn.softmax` | 内部 Softmax | [L440](../../../../torchax/torchax/ops/jaten.py#L440) |
| `_log_softmax` | `jax.nn.log_softmax` | Log Softmax | [L3060](../../../../torchax/torchax/ops/jaten.py#L3060) |
| `log_sigmoid` | `jax.nn.log_sigmoid` | Log Sigmoid | [L5032](../../../../torchax/torchax/ops/jaten.py#L5032) |
| `leaky_relu` | `jax.nn.leaky_relu` | Leaky ReLU | [L3006](../../../../torchax/torchax/ops/jaten.py#L3006) |
| `hardtanh` | `jnp.clip` | Hard Tanh | [L2770](../../../../torchax/torchax/ops/jaten.py#L2770) |
| `glu` | `jax.nn.glu` | GLU | [L2764](../../../../torchax/torchax/ops/jaten.py#L2764) |
| `erf` | `jax.lax.erf` | 误差函数 | [L2652](../../../../torchax/torchax/ops/jaten.py#L2652) |
| `erfc` | `jax.lax.erfc` | 互补误差函数 | [L4871](../../../../torchax/torchax/ops/jaten.py#L4871) |
| `erfinv` | `jax.lax.erf_inv` | 反误差函数 | [L2658](../../../../torchax/torchax/ops/jaten.py#L2658) |

### 8. 归一化操作 (Normalization Operations)

| 操作符 | JAX 实现 | 说明 | 源码位置 |
|--------|----------|------|----------|
| `native_layer_norm` | 自定义实现 | Layer Norm | [L945](../../../../torchax/torchax/ops/jaten.py#L945) |
| `native_group_norm` | 自定义实现 | Group Norm | [L1512](../../../../torchax/torchax/ops/jaten.py#L1512) |
| `_native_batch_norm_legit` | 自定义实现 | Batch Norm | [L1160](../../../../torchax/torchax/ops/jaten.py#L1160) |
| `_native_batch_norm_legit_no_training` | 自定义实现 | 推理时 Batch Norm | [L1219](../../../../torchax/torchax/ops/jaten.py#L1219) |
| `native_batch_norm` | 自定义实现 | 原生 Batch Norm | [L3719](../../../../torchax/torchax/ops/jaten.py#L3719) |
| `linalg_vector_norm` | 自定义实现 | 向量范数 | [L1571](../../../../torchax/torchax/ops/jaten.py#L1571) |

### 9. 卷积和池化 (Convolution & Pooling)

| 操作符 | JAX 实现 | 说明 | 源码位置 |
|--------|----------|------|----------|
| `conv2d` | `_aten_convolution` | 2D 卷积 | [L1045](../../../../torchax/torchax/ops/jaten.py#L1045) |
| `convolution` | `jax.lax.conv_general_dilated` | 通用卷积 | [L1068](../../../../torchax/torchax/ops/jaten.py#L1068) |
| `avg_pool1d` | 自定义实现 | 1D 平均池化 | [L2141](../../../../torchax/torchax/ops/jaten.py#L2141) |
| `avg_pool2d` | 自定义实现 | 2D 平均池化 | [L2141](../../../../torchax/torchax/ops/jaten.py#L2141) |
| `avg_pool3d` | 自定义实现 | 3D 平均池化 | [L2141](../../../../torchax/torchax/ops/jaten.py#L2141) |
| `max_pool2d_with_indices` | 自定义 `max_pool` | 带索引的 2D 最大池化 | [L1397](../../../../torchax/torchax/ops/jaten.py#L1397) |
| `max_pool3d_with_indices` | 自定义 `max_pool` | 带索引的 3D 最大池化 | [L1397](../../../../torchax/torchax/ops/jaten.py#L1397) |
| `_adaptive_avg_pool2d` | 自定义实现 | 自适应 2D 平均池化 | [L2001](../../../../torchax/torchax/ops/jaten.py#L2001) |
| `_adaptive_avg_pool3d` | 自定义实现 | 自适应 3D 平均池化 | [L2001](../../../../torchax/torchax/ops/jaten.py#L2001) |
| `max_unpool2d` | 自定义实现 | 2D 最大反池化 | [L5245](../../../../torchax/torchax/ops/jaten.py#L5245) |
| `max_unpool3d` | 自定义实现 | 3D 最大反池化 | [L5245](../../../../torchax/torchax/ops/jaten.py#L5245) |
| `reflection_pad1d` | `jnp.pad(..., 'reflect')` | 1D 反射填充 | [L1631](../../../../torchax/torchax/ops/jaten.py#L1631) |
| `constant_pad_nd` | `jax.lax.pad` | 常量填充 | [L2513](../../../../torchax/torchax/ops/jaten.py#L2513) |
| `pad` | `jnp.pad` | 通用填充 | [L5607](../../../../torchax/torchax/ops/jaten.py#L5607) |

### 10. 索引和选择操作 (Indexing & Selection)

| 操作符 | JAX 实现 | 说明 | 源码位置 |
|--------|----------|------|----------|
| `select` | `jax.lax.index_in_dim` | 选择维度 | [L210](../../../../torchax/torchax/ops/jaten.py#L210) |
| `index_select` | `jnp.take` | 索引选择 | [L215](../../../../torchax/torchax/ops/jaten.py#L215) |
| `select_copy` | `jnp.take` | 索引选择副本 | [L215](../../../../torchax/torchax/ops/jaten.py#L215) |
| `index` | Python 索引 | 张量索引 | [L793](../../../../torchax/torchax/ops/jaten.py#L793) |
| `_unsafe_index` | Python 索引 | 不安全索引 | [L793](../../../../torchax/torchax/ops/jaten.py#L793) |
| `index_put` | `at[].set/add` | 索引赋值 | [L783](../../../../torchax/torchax/ops/jaten.py#L783) |
| `_unsafe_index_put` | `_aten_index_put` | 不安全索引赋值 | [L5017](../../../../torchax/torchax/ops/jaten.py#L5017) |
| `index_copy` | `at[].set` | 索引复制 | [L123](../../../../torchax/torchax/ops/jaten.py#L123) |
| `gather` | 自定义实现 | 收集操作 | [L2745](../../../../torchax/torchax/ops/jaten.py#L2745) |
| `scatter` | `at[].set` | 散射操作 | [L1881](../../../../torchax/torchax/ops/jaten.py#L1881) |
| `scatter_add` | `at[].add` | 散射加法 | [L1797](../../../../torchax/torchax/ops/jaten.py#L1797) |
| `scatter_reduce` | `at[].add/mul/max/min` | 散射归约 | [L1880](../../../../torchax/torchax/ops/jaten.py#L1880) |
| `select_scatter` | `at[].set` | 选择散射 | [L2244](../../../../torchax/torchax/ops/jaten.py#L2244) |
| `slice_scatter` | `at[].set` | 切片散射 | [L3233](../../../../torchax/torchax/ops/jaten.py#L3233) |
| `diagonal_scatter` | `at[].set` | 对角线散射 | [L2607](../../../../torchax/torchax/ops/jaten.py#L2607) |
| `masked_scatter` | `at[].set` | 掩码散射 | [L1806](../../../../torchax/torchax/ops/jaten.py#L1806) |
| `masked_select` | 条件索引 | 掩码选择 | [L1826](../../../../torchax/torchax/ops/jaten.py#L1826) |
| `take` | `self.flatten()[index]` | 取元素 | [L5601](../../../../torchax/torchax/ops/jaten.py#L5601) |
| `put` | `jnp.put` | 放置元素 | [L3149](../../../../torchax/torchax/ops/jaten.py#L3149) |
| `topk` | `jax.lax.top_k` | 前 k 个元素 | [L3259](../../../../torchax/torchax/ops/jaten.py#L3259) |
| `kthvalue` | `jnp.partition` | 第 k 个值 | [L5581](../../../../torchax/torchax/ops/jaten.py#L5581) |
| `where` | `jnp.where` | 条件选择 | [L3496](../../../../torchax/torchax/ops/jaten.py#L3496) |
| `nonzero` | `jnp.nonzero` | 非零元素索引 | [L3127](../../../../torchax/torchax/ops/jaten.py#L3127) |
| `nonzero_static` | `jnp.argwhere` | 静态非零索引 | [L3109](../../../../torchax/torchax/ops/jaten.py#L3109) |

### 11. 比较和逻辑操作 (Comparison & Logical)

| 操作符 | JAX 实现 | 说明 | 源码位置 |
|--------|----------|------|----------|
| `eq` | `==` | 等于 | [L2639](../../../../torchax/torchax/ops/jaten.py#L2639) |
| `ne` | `jnp.not_equal` | 不等于 | [L848](../../../../torchax/torchax/ops/jaten.py#L848) |
| `lt` | `<` | 小于 | [L1941](../../../../torchax/torchax/ops/jaten.py#L1941) |
| `le` | `<=` | 小于等于 | [L3000](../../../../torchax/torchax/ops/jaten.py#L3000) |
| `gt` | `>` | 大于 | [L1934](../../../../torchax/torchax/ops/jaten.py#L1934) |
| `ge` | `>=` | 大于等于 | [L2759](../../../../torchax/torchax/ops/jaten.py#L2759) |
| `equal` | `jnp.array_equal` | 数组完全相等 | [L2645](../../../../torchax/torchax/ops/jaten.py#L2645) |
| `allclose` | `jnp.allclose` | 近似相等 | [L3714](../../../../torchax/torchax/ops/jaten.py#L3714) |
| `logical_and` | `jnp.logical_and` | 逻辑与 | [L3040](../../../../torchax/torchax/ops/jaten.py#L3040) |
| `logical_or` | `jnp.logical_or` | 逻辑或 | [L3047](../../../../torchax/torchax/ops/jaten.py#L3047) |
| `logical_not` | `jnp.logical_not` | 逻辑非 | [L3054](../../../../torchax/torchax/ops/jaten.py#L3054) |
| `logical_xor` | `jnp.logical_xor` | 逻辑异或 | [L3089](../../../../torchax/torchax/ops/jaten.py#L3089) |
| `isinf` | `jnp.isinf` | 是否无穷 | [L2989](../../../../torchax/torchax/ops/jaten.py#L2989) |
| `isnan` | `jnp.isnan` | 是否 NaN | [L2995](../../../../torchax/torchax/ops/jaten.py#L2995) |
| `isfinite` | `jnp.isfinite` | 是否有限 | [L391](../../../../torchax/torchax/ops/jaten.py#L391) |

### 12. 位运算 (Bitwise Operations)

| 操作符 | JAX 实现 | 说明 | 源码位置 |
|--------|----------|------|----------|
| `bitwise_not` | `~self` | 按位取反 | [L1694](../../../../torchax/torchax/ops/jaten.py#L1694) |
| `bitwise_and` | `self & other` | 按位与 | [L2404](../../../../torchax/torchax/ops/jaten.py#L2404) |
| `bitwise_or` | `self \| other` | 按位或 | [L2411](../../../../torchax/torchax/ops/jaten.py#L2411) |
| `bitwise_xor` | `self ^ other` | 按位异或 | [L2417](../../../../torchax/torchax/ops/jaten.py#L2417) |
| `bitwise_left_shift` | `jnp.left_shift` | 左移 | [L1700](../../../../torchax/torchax/ops/jaten.py#L1700) |
| `bitwise_right_shift` | `jnp.right_shift` | 右移 | [L1707](../../../../torchax/torchax/ops/jaten.py#L1707) |
| `__lshift__` | `jnp.left_shift` | 左移运算符 | [L1700](../../../../torchax/torchax/ops/jaten.py#L1700) |
| `__rshift__` | `jnp.right_shift` | 右移运算符 | [L1707](../../../../torchax/torchax/ops/jaten.py#L1707) |
| `__and__` | `self & other` | 与运算符 | [L2404](../../../../torchax/torchax/ops/jaten.py#L2404) |
| `__or__` | `jnp.logical_or` | 或运算符 | [L3047](../../../../torchax/torchax/ops/jaten.py#L3047) |
| `__xor__` | `jnp.logical_xor` | 异或运算符 | [L3089](../../../../torchax/torchax/ops/jaten.py#L3089) |

### 13. 线性代数 (Linear Algebra)

| 操作符 | JAX 实现 | 说明 | 源码位置 |
|--------|----------|------|----------|
| `cholesky` | `jax.scipy.linalg.cholesky` | Cholesky 分解 | [L223](../../../../torchax/torchax/ops/jaten.py#L223) |
| `linalg_cholesky_ex` | `jax.scipy.linalg.cholesky` | 扩展 Cholesky | [L228](../../../../torchax/torchax/ops/jaten.py#L228) |
| `cholesky_solve` | `jax.scipy.linalg.cho_solve` | Cholesky 求解 | [L245](../../../../torchax/torchax/ops/jaten.py#L245) |
| `cholesky_inverse` | `jnp.linalg.inv` | Cholesky 逆 | [L2556](../../../../torchax/torchax/ops/jaten.py#L2556) |
| `qr` | `jnp.linalg.qr` | QR 分解 | [L5038](../../../../torchax/torchax/ops/jaten.py#L5038) |
| `linalg_qr` | `jnp.linalg.qr` | 线性代数 QR | [L5049](../../../../torchax/torchax/ops/jaten.py#L5049) |
| `linalg_eig` | `jnp.linalg.eig` | 特征值分解 | [L2828](../../../../torchax/torchax/ops/jaten.py#L2828) |
| `_linalg_eigh` | `jnp.linalg.eigh` | 厄米特特征值 | [L2833](../../../../torchax/torchax/ops/jaten.py#L2833) |
| `_linalg_svd` | `jnp.linalg.svd` | SVD 分解 | [L5069](../../../../torchax/torchax/ops/jaten.py#L5069) |
| `_linalg_slogdet` | `jnp.linalg.slogdet` | 符号 log 行列式 | [L5062](../../../../torchax/torchax/ops/jaten.py#L5062) |
| `linalg_inv_ex` | `jnp.linalg.inv` | 矩阵求逆 | [L5110](../../../../torchax/torchax/ops/jaten.py#L5110) |
| `linalg_lu` | `jax.lax.linalg.lu` | LU 分解 | [L2923](../../../../torchax/torchax/ops/jaten.py#L2923) |
| `linalg_lu_factor_ex` | `jax.lax.linalg.lu` | LU 因式分解 | [L2953](../../../../torchax/torchax/ops/jaten.py#L2953) |
| `linalg_lu_solve` | `jax.scipy.linalg.lu_solve` | LU 求解 | [L2962](../../../../torchax/torchax/ops/jaten.py#L2962) |
| `lu_unpack` | 自定义实现 | LU 解包 | [L5471](../../../../torchax/torchax/ops/jaten.py#L5471) |
| `linalg_lstsq` | `jnp.linalg.lstsq` | 最小二乘解 | [L2838](../../../../torchax/torchax/ops/jaten.py#L2838) |
| `linalg_pinv` | `jnp.linalg.pinv` | 伪逆 | [L5075](../../../../torchax/torchax/ops/jaten.py#L5075) |
| `_linalg_solve_ex` | `jnp.linalg.solve` | 线性求解 | [L5081](../../../../torchax/torchax/ops/jaten.py#L5081) |
| `linalg_solve_triangular` | `jax.scipy.linalg.solve_triangular` | 三角求解 | [L5096](../../../../torchax/torchax/ops/jaten.py#L5096) |
| `triangular_solve` | `jax.lax.linalg.triangular_solve` | 三角系统求解 | [L5173](../../../../torchax/torchax/ops/jaten.py#L5173) |
| `linalg_matrix_exp` | `jax.scipy.linalg.expm` | 矩阵指数 | [L5056](../../../../torchax/torchax/ops/jaten.py#L5056) |
| `linalg_householder_product` | `jax.lax.linalg.householder_product` | Householder 乘积 | [L205](../../../../torchax/torchax/ops/jaten.py#L205) |
| `linalg_ldl_factor_ex` | 自定义实现 | LDL 分解 | [L2899](../../../../torchax/torchax/ops/jaten.py#L2899) |
| `triu` | `jnp.triu` | 上三角 | [L358](../../../../torchax/torchax/ops/jaten.py#L358) |
| `tril_indices` | `jnp.tril_indices` | 下三角索引 | [L3317](../../../../torchax/torchax/ops/jaten.py#L3317) |
| `triu_indices` | `jnp.triu_indices` | 上三角索引 | [L3335](../../../../torchax/torchax/ops/jaten.py#L3335) |
| `diag` | `jnp.diag` | 对角线 | [L2578](../../../../torchax/torchax/ops/jaten.py#L2578) |
| `diagonal` | `jnp.diagonal` | 对角元素 | [L2584](../../../../torchax/torchax/ops/jaten.py#L2584) |
| `diagflat` | `jnp.diagflat` | 对角展平 | [L2628](../../../../torchax/torchax/ops/jaten.py#L2628) |

### 14. 随机数生成 (Random Number Generation)

| 操作符 | JAX 实现 | 说明 | 源码位置 |
|--------|----------|------|----------|
| `rand` | `jax.random.uniform` | 均匀分布 [0,1) | [L3686](../../../../torchax/torchax/ops/jaten.py#L3686) |
| `randn` | `jax.random.normal` | 标准正态分布 | [L3627](../../../../torchax/torchax/ops/jaten.py#L3627) |
| `randn_like` | `jax.random.normal` | 正态分布 (同形状) | [L3670](../../../../torchax/torchax/ops/jaten.py#L3670) |
| `randint` | `jax.random.randint` | 随机整数 | [L3783](../../../../torchax/torchax/ops/jaten.py#L3783) |
| `randint_like` | `jax.random.randint` | 随机整数 (同形状) | [L3808](../../../../torchax/torchax/ops/jaten.py#L3808) |
| `randperm` | `jax.random.permutation` | 随机排列 | [L3173](../../../../torchax/torchax/ops/jaten.py#L3173) |
| `uniform` | 自定义实现 | 均匀分布 | [L3770](../../../../torchax/torchax/ops/jaten.py#L3770) |
| `normal` | 自定义实现 | 正态分布 | [L3756](../../../../torchax/torchax/ops/jaten.py#L3756) |
| `bernoulli.p` | `jax.random.uniform` | 伯努利分布 | [L3650](../../../../torchax/torchax/ops/jaten.py#L3650) |
| `geometric` | `jax.random.geometric` | 几何分布 | [L3664](../../../../torchax/torchax/ops/jaten.py#L3664) |
| `multinomial` | `jax.random.choice` | 多项式采样 | [L4942](../../../../torchax/torchax/ops/jaten.py#L4942) |
| `cauchy_` | `jax.random.cauchy` | 柯西分布 | [L142](../../../../torchax/torchax/ops/jaten.py#L142) |
| `exponential_` | `jax.random.exponential` | 指数分布 | [L187](../../../../torchax/torchax/ops/jaten.py#L187) |

### 15. 嵌入操作 (Embedding Operations)

| 操作符 | JAX 实现 | 说明 | 源码位置 |
|--------|----------|------|----------|
| `embedding` | `jnp.take` | 嵌入查找 | [L579](../../../../torchax/torchax/ops/jaten.py#L579) |
| `embedding_renorm_` | 自定义实现 | 嵌入重归一化 | [L589](../../../../torchax/torchax/ops/jaten.py#L589) |
| `_embedding_bag` | 自定义实现 | 嵌入包 | [L613](../../../../torchax/torchax/ops/jaten.py#L613) |
| `_embedding_bag_forward_only` | 自定义实现 | 前向嵌入包 | [L613](../../../../torchax/torchax/ops/jaten.py#L613) |

### 16. FFT 操作 (FFT Operations)

| 操作符 | JAX 实现 | 说明 | 源码位置 |
|--------|----------|------|----------|
| `_fft_c2c` | `jnp.fft.fftn/ifftn` | 复数到复数 FFT | [L5188](../../../../torchax/torchax/ops/jaten.py#L5188) |
| `_fft_r2c` | `jnp.fft.rfftn/fftn` | 实数到复数 FFT | [L5206](../../../../torchax/torchax/ops/jaten.py#L5206) |
| `_fft_c2r` | `jnp.fft.irfftn` | 复数到实数 FFT | [L5219](../../../../torchax/torchax/ops/jaten.py#L5219) |

### 17. 张量创建 (Tensor Creation)

| 操作符 | JAX 实现 | 说明 | 源码位置 |
|--------|----------|------|----------|
| `empty` | `jnp.empty` | 空张量 | [L737](../../../../torchax/torchax/ops/jaten.py#L737) |
| `empty_like` | `jnp.empty_like` | 空张量 (同形状) | [L743](../../../../torchax/torchax/ops/jaten.py#L743) |
| `ones` | `jnp.ones` | 全 1 张量 | [L749](../../../../torchax/torchax/ops/jaten.py#L749) |
| `zeros` | `jnp.zeros` | 全 0 张量 | [L755](../../../../torchax/torchax/ops/jaten.py#L755) |
| `full` | `jnp.full` | 填充张量 | [L761](../../../../torchax/torchax/ops/jaten.py#L761) |
| `full_like` | `jnp.full` | 填充张量 (同形状) | [L2695](../../../../torchax/torchax/ops/jaten.py#L2695) |
| `arange` | `jnp.arange` | 范围张量 | [L2337](../../../../torchax/torchax/ops/jaten.py#L2337) |
| `empty_permuted` | `jnp.empty` | 置换空张量 | [L768](../../../../torchax/torchax/ops/jaten.py#L768) |
| `empty_strided` | `jnp.empty` | 步进空张量 | [L776](../../../../torchax/torchax/ops/jaten.py#L776) |
| `scalar_tensor` | `jnp.array` | 标量张量 | [L3553](../../../../torchax/torchax/ops/jaten.py#L3553) |
| `new_empty` | `jnp.empty` | 新空张量 | [L4997](../../../../torchax/torchax/ops/jaten.py#L4997) |
| `new_empty_strided` | `jnp.empty` | 新步进空张量 | [L5007](../../../../torchax/torchax/ops/jaten.py#L5007) |

### 18. 张量操作 (Tensor Manipulation)

| 操作符 | JAX 实现 | 说明 | 源码位置 |
|--------|----------|------|----------|
| `clone` | 返回自身 | 克隆 | [L111](../../../../torchax/torchax/ops/jaten.py#L111) |
| `copy` | `jnp.broadcast_to` | 复制 | [L2534](../../../../torchax/torchax/ops/jaten.py#L2534) |
| `copy_` | 赋值操作 | 原地复制 | [L92](../../../../torchax/torchax/ops/jaten.py#L92) |
| `_to_copy` | `astype/copy` | 类型转换复制 | [L729](../../../../torchax/torchax/ops/jaten.py#L729) |
| `to.dtype` | `astype` | 类型转换 | [L3506](../../../../torchax/torchax/ops/jaten.py#L3506) |
| `to.device` | 返回自身 | 设备转换 | [L3559](../../../../torchax/torchax/ops/jaten.py#L3559) |
| `detach` | 返回自身 | 分离梯度 | [L380](../../../../torchax/torchax/ops/jaten.py#L380) |
| `positive` | 返回自身 | 正值 | [L380](../../../../torchax/torchax/ops/jaten.py#L380) |
| `alias` | 返回自身 | 别名 | [L1640](../../../../torchax/torchax/ops/jaten.py#L1640) |
| `lift_fresh` | 返回自身 | 提升 | [L3765](../../../../torchax/torchax/ops/jaten.py#L3765) |
| `lift_fresh_copy` | `jnp.copy` | 提升复制 | [L2529](../../../../torchax/torchax/ops/jaten.py#L2529) |
| `flip` | `jnp.flip` | 翻转 | [L2706](../../../../torchax/torchax/ops/jaten.py#L2706) |
| `roll` | `jnp.roll` | 滚动 | [L3227](../../../../torchax/torchax/ops/jaten.py#L3227) |
| `repeat` | `jnp.tile` | 重复 | [L3217](../../../../torchax/torchax/ops/jaten.py#L3217) |
| `repeat_interleave` | `jnp.repeat` | 交错重复 | [L412](../../../../torchax/torchax/ops/jaten.py#L412) |
| `sort` | `jnp.sort/argsort` | 排序 | [L3244](../../../../torchax/torchax/ops/jaten.py#L3244) |
| `fill` | `jnp.full` | 填充 | [L2695](../../../../torchax/torchax/ops/jaten.py#L2695) |
| `resize_` | `jax.numpy.resize` | 调整大小 | [L401](../../../../torchax/torchax/ops/jaten.py#L401) |
| `resize_as_` | `jax.numpy.resize` | 调整为相同大小 | [L407](../../../../torchax/torchax/ops/jaten.py#L407) |

### 19. 复数操作 (Complex Number Operations)

| 操作符 | JAX 实现 | 说明 | 源码位置 |
|--------|----------|------|----------|
| `complex` | `real + 1j * imag` | 创建复数 | [L171](../../../../torchax/torchax/ops/jaten.py#L171) |
| `real` | `jnp.real` | 实部 | [L396](../../../../torchax/torchax/ops/jaten.py#L396) |
| `imag` | `jnp.imag` | 虚部 | [L386](../../../../torchax/torchax/ops/jaten.py#L386) |
| `view_as_real` | `jnp.stack([real, imag])` | 视为实数 | [L427](../../../../torchax/torchax/ops/jaten.py#L427) |
| `view_as_complex` | `jax.lax.complex` | 视为复数 | [L499](../../../../torchax/torchax/ops/jaten.py#L499) |
| `conj_physical` | `jnp.conjugate` | 共轭 | [L5022](../../../../torchax/torchax/ops/jaten.py#L5022) |
| `polar` | `jax.lax.complex` | 极坐标 | [L5377](../../../../torchax/torchax/ops/jaten.py#L5377) |

### 20. 特殊数学函数 (Special Math Functions)

| 操作符 | JAX 实现 | 说明 | 源码位置 |
|--------|----------|------|----------|
| `special_zeta` | `jax.scipy.special.zeta` | Riemann zeta | [L254](../../../../torchax/torchax/ops/jaten.py#L254) |
| `igammac` | `jax.scipy.special.gammaincc` | 不完全 gamma 函数 | [L264](../../../../torchax/torchax/ops/jaten.py#L264) |
| `igamma` | `jax.scipy.special.gammainc` | gamma 函数 | [L2813](../../../../torchax/torchax/ops/jaten.py#L2813) |
| `digamma` | `jax.scipy.special.digamma` | digamma 函数 | [L2805](../../../../torchax/torchax/ops/jaten.py#L2805) |
| `lgamma` | `jax.scipy.special.gammaln` | log gamma | [L2818](../../../../torchax/torchax/ops/jaten.py#L2818) |
| `mvlgamma` | `jax.scipy.special.multigammaln` | 多元 log gamma | [L2823](../../../../torchax/torchax/ops/jaten.py#L2823) |
| `polygamma` | `jax.lax.polygamma` | polygamma 函数 | [L4289](../../../../torchax/torchax/ops/jaten.py#L4289) |
| `special_ndtri` | `jax.scipy.special.ndtri` | 正态分位数 | [L4296](../../../../torchax/torchax/ops/jaten.py#L4296) |
| `special_erfcx` | `exp(x²) * erfc(x)` | 缩放互补误差函数 | [L4864](../../../../torchax/torchax/ops/jaten.py#L4864) |
| `i0` | `jax.scipy.special.i0` | 第一类修正贝塞尔 | [L3859](../../../../torchax/torchax/ops/jaten.py#L3859) |
| `special_i0e` | `jax.scipy.special.i0e` | 缩放 i0 | [L3865](../../../../torchax/torchax/ops/jaten.py#L3865) |
| `special_i1` | `jax.scipy.special.i1` | 第一类修正贝塞尔 i1 | [L3871](../../../../torchax/torchax/ops/jaten.py#L3871) |
| `special_i1e` | `jax.scipy.special.i1e` | 缩放 i1 | [L3877](../../../../torchax/torchax/ops/jaten.py#L3877) |
| `special_bessel_j0` | 自定义实现 | 第一类贝塞尔 j0 | [L4302](../../../../torchax/torchax/ops/jaten.py#L4302) |
| `special_bessel_j1` | 自定义实现 | 第一类贝塞尔 j1 | [L4419](../../../../torchax/torchax/ops/jaten.py#L4419) |
| `special_bessel_y0` | 自定义实现 | 第二类贝塞尔 y0 | [L4532](../../../../torchax/torchax/ops/jaten.py#L4532) |
| `special_bessel_y1` | 自定义实现 | 第二类贝塞尔 y1 | [L4650](../../../../torchax/torchax/ops/jaten.py#L4650) |
| `special_modified_bessel_i0` | 自定义实现 | 修正贝塞尔 i0 | [L3924](../../../../torchax/torchax/ops/jaten.py#L3924) |
| `special_modified_bessel_i1` | 自定义实现 | 修正贝塞尔 i1 | [L4018](../../../../torchax/torchax/ops/jaten.py#L4018) |
| `special_modified_bessel_k0` | 自定义实现 | 修正贝塞尔 k0 | [L4120](../../../../torchax/torchax/ops/jaten.py#L4120) |
| `special_modified_bessel_k1` | 自定义实现 | 修正贝塞尔 k1 | [L4203](../../../../torchax/torchax/ops/jaten.py#L4203) |
| `special_chebyshev_polynomial_t` | 自定义实现 | 切比雪夫 T | [L4768](../../../../torchax/torchax/ops/jaten.py#L4768) |
| `special_chebyshev_polynomial_u` | 自定义实现 | 切比雪夫 U | [L4814](../../../../torchax/torchax/ops/jaten.py#L4814) |
| `special_hermite_polynomial_h` | 自定义实现 | 厄米特 H | [L4876](../../../../torchax/torchax/ops/jaten.py#L4876) |
| `special_hermite_polynomial_he` | 自定义实现 | 厄米特 He | [L4909](../../../../torchax/torchax/ops/jaten.py#L4909) |
| `special_laguerre_polynomial_l` | 自定义实现 | 拉盖尔多项式 | [L3883](../../../../torchax/torchax/ops/jaten.py#L3883) |

### 21. 距离计算 (Distance Computation)

| 操作符 | JAX 实现 | 说明 | 源码位置 |
|--------|----------|------|----------|
| `dist` | 范数差 | p-范数距离 | [L538](../../../../torchax/torchax/ops/jaten.py#L538) |
| `_cdist_forward` | `jnp.linalg.norm` | 成对距离 | [L2539](../../../../torchax/torchax/ops/jaten.py#L2539) |
| `_pdist_forward` | 自定义实现 | 压缩距离 | [L2549](../../../../torchax/torchax/ops/jaten.py#L2549) |
| `cdist` | 自定义实现 | 完整距离计算 | [L5382](../../../../torchax/torchax/ops/jaten.py#L5382) |

### 22. 上采样 (Upsampling)

| 操作符 | JAX 实现 | 说明 | 源码位置 |
|--------|----------|------|----------|
| `_upsample_bilinear2d_aa` | 自定义实现 | 双线性上采样 (抗锯齿) | [L5335](../../../../torchax/torchax/ops/jaten.py#L5335) |
| `_upsample_bicubic2d_aa` | 自定义实现 | 双三次上采样 (抗锯齿) | [L5356](../../../../torchax/torchax/ops/jaten.py#L5356) |

### 23. 其他操作 (Miscellaneous)

| 操作符 | JAX 实现 | 说明 | 源码位置 |
|--------|----------|------|----------|
| `gcd` | `jnp.gcd` | 最大公约数 | [L2977](../../../../torchax/torchax/ops/jaten.py#L2977) |
| `lcm` | `jnp.lcm` | 最小公倍数 | [L2983](../../../../torchax/torchax/ops/jaten.py#L2983) |
| `bincount` | `jnp.bincount` | 计数 | [L1688](../../../../torchax/torchax/ops/jaten.py#L1688) |
| `bucketize` | `jnp.digitize` | 分桶 | [L1039](../../../../torchax/torchax/ops/jaten.py#L1039) |
| `searchsorted` | `jnp.searchsorted` | 排序搜索 | [L294](../../../../torchax/torchax/ops/jaten.py#L294) |
| `histc` | `jnp.histogram` | 直方图 | [L2783](../../../../torchax/torchax/ops/jaten.py#L2783) |
| `frexp` | `jnp.frexp` | 尾数和指数 | [L2739](../../../../torchax/torchax/ops/jaten.py#L2739) |
| `copysign` | `jnp.copysign` | 复制符号 | [L3846](../../../../torchax/torchax/ops/jaten.py#L3846) |
| `nextafter` | `jnp.nextafter` | 下一个浮点数 | [L3104](../../../../torchax/torchax/ops/jaten.py#L3104) |
| `atleast_1d` | `jnp.atleast_1d` | 至少 1D | [L165](../../../../torchax/torchax/ops/jaten.py#L165) |
| `atleast_2d` | `jnp.atleast_2d` | 至少 2D | [L160](../../../../torchax/torchax/ops/jaten.py#L160) |
| `broadcast_tensors` | `jax.lax.broadcast_in_dim` | 广播张量 | [L2423](../../../../torchax/torchax/ops/jaten.py#L2423) |
| `broadcast_to` | `jnp.broadcast_to` | 广播到形状 | [L2495](../../../../torchax/torchax/ops/jaten.py#L2495) |
| `as_strided` | 自定义实现 | 步进视图 | [L2380](../../../../torchax/torchax/ops/jaten.py#L2380) |
| `as_strided_copy` | 自定义实现 | 步进视图副本 | [L2380](../../../../torchax/torchax/ops/jaten.py#L2380) |
| `as_strided_scatter` | 自定义实现 | 步进散射 | [L2388](../../../../torchax/torchax/ops/jaten.py#L2388) |
| `unique_dim` | `jnp.unique` | 唯一值 (维度) | [L3359](../../../../torchax/torchax/ops/jaten.py#L3359) |
| `_unique` | `jnp.unique` | 唯一值 | [L3396](../../../../torchax/torchax/ops/jaten.py#L3396) |
| `_unique2` | `jnp.unique` | 唯一值 v2 | [L3415](../../../../torchax/torchax/ops/jaten.py#L3415) |
| `unique_consecutive` | 自定义实现 | 连续唯一值 | [L3429](../../../../torchax/torchax/ops/jaten.py#L3429) |
| `tensor_split` | `jnp.array_split` | 张量分割 | [L3621](../../../../torchax/torchax/ops/jaten.py#L3621) |
| `unbind_copy` | `jax.lax.index_in_dim` | 解绑定 | [L3351](../../../../torchax/torchax/ops/jaten.py#L3351) |
| `_trilinear` | `jnp.expand_dims * sum` | 三线性操作 | [L5233](../../../../torchax/torchax/ops/jaten.py#L5233) |
| `_local_scalar_dense` | `.item()` | 获取标量 | [L3617](../../../../torchax/torchax/ops/jaten.py#L3617) |
| `dim` | `len(shape)` | 维度数 | [L3841](../../../../torchax/torchax/ops/jaten.py#L3841) |
| `sym_size` | `shape[dim]` | 符号大小 | [L1492](../../../../torchax/torchax/ops/jaten.py#L1492) |
| `is_nonzero` | 自定义实现 | 是否非零 | [L5654](../../../../torchax/torchax/ops/jaten.py#L5654) |

---

### 24. 就地操作 (In-place Operations)

TorchAx 实现了 **38 个就地操作符**，这些操作符修改输入张量而不创建新张量。源码位置：Line 5745-5783

| 操作符 | 对应的函数式操作符 | 说明 |
|--------|-------------------|------|
| `add_` | `add` | 就地加法 |
| `sub_` | `sub` | 就地减法 |
| `mul_` | `mul` | 就地乘法 |
| `div_` | `div` | 就地除法 |
| `pow_` | `pow` | 就地幂运算 |
| `lt_` | `lt` | 就地小于比较 |
| `le_` | `le` | 就地小于等于比较 |
| `gt_` | `gt` | 就地大于比较 |
| `ge_` | `ge` | 就地大于等于比较 |
| `eq_` | `eq` | 就地等于比较 |
| `ne_` | `ne` | 就地不等于比较 |
| `bernoulli_` | `bernoulli.p` | 就地伯努利采样 |
| `bernoulli_.float` | `_aten_bernoulli` | 就地浮点伯努利采样 |
| `geometric_` | `geometric` | 就地几何分布采样 |
| `normal_` | `normal` | 就地正态分布采样 |
| `random_` | `uniform` | 就地随机采样 |
| `uniform_` | `uniform` | 就地均匀分布采样 |
| `relu_` | `relu` | 就地 ReLU |
| `squeeze_` | `squeeze` | 就地压缩维度 |
| `sqrt_` | `sqrt` | 就地平方根 |
| `clamp_` | `clamp` | 就地截断 |
| `clamp_min_` | `clamp_min` | 就地下限截断 |
| `sigmoid_` | `sigmoid` | 就地 Sigmoid |
| `tanh_` | `tanh` | 就地双曲正切 |
| `ceil_` | `ceil` | 就地向上取整 |
| `logical_not_` | `logical_not` | 就地逻辑非 |
| `unsqueeze_` | `unsqueeze` | 就地扩展维度 |
| `transpose_` | `transpose` | 就地转置 |
| `log_normal_` | `log_normal` | 就地对数正态采样 |
| `scatter_add_` | `scatter_add` | 就地散射加法 |
| `scatter_reduce_.two` | `scatter_reduce` | 就地散射归约 |
| `scatter_` | `scatter` | 就地散射 |
| `bitwise_or_` | `bitwise_or` | 就地按位或 |
| `floor_divide_` | `floor_divide` | 就地整除 |
| `remainder_` | `remainder` | 就地取余 |
| `index_put_` | `index_put` | 就地索引赋值 |
| `masked_scatter_` | `masked_scatter` | 就地掩码散射 |
| `cauchy_` | `jax.random.cauchy` | 就地柯西分布采样 |
| `exponential_` | `jax.random.exponential` | 就地指数分布采样 |

---

## 统计信息

| 类别 | 操作符数量 |
|------|-----------|
| 张量形状操作 | 27 |
| 基础算术运算 | 30 |
| 三角函数 | 14 |
| 指数对数 | 11 |
| 矩阵运算 | 9 |
| 归约操作 | 19 |
| 激活函数 | 15 |
| 归一化 | 6 |
| 卷积池化 | 13 |
| 索引选择 | 24 |
| 比较逻辑 | 15 |
| 位运算 | 11 |
| 线性代数 | 30 |
| 随机数 | 13 |
| 嵌入 | 4 |
| FFT | 3 |
| 张量创建 | 12 |
| 张量操作 | 18 |
| 复数操作 | 7 |
| 特殊函数 | 28 |
| 距离计算 | 4 |
| 上采样 | 2 |
| 其他 | 27 |
| 就地操作 | 38 |
| **总计** | **382** |

---

## 结论

TorchAx 通过将 **382 个 Core ATen 操作符** 映射到 JAX 实现，实现了 PyTorch 代码在 TPU 上的无缝运行。这些操作符覆盖了：

1. **所有基础张量运算** - 算术、三角函数、指数对数等
2. **高级线性代数** - 矩阵分解、求解线性系统
3. **神经网络层** - 卷积、池化、归一化、激活函数
4. **完整的索引系统** - scatter、gather、masked 操作
5. **随机数生成** - 多种分布
6. **特殊数学函数** - 贝塞尔函数、多项式等

这种设计的优势在于：
- **复用 PyTorch 成熟的算子分解机制**
- **保持与 PyTorch 的语义一致性**
- **利用 JAX 的高效编译和 TPU 优化**

