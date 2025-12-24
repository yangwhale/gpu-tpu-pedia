# TorchAx Core ATen 操作符目录

> 本文档列出了 TorchAx 中实现的所有 PyTorch Core ATen 操作符及其 JAX 实现。
>
> 源文件: [`torchax/torchax/ops/jaten.py`](../../../../torchax/torchax/ops/jaten.py) (5801 行)

## 概述

TorchAx 实现了 **382 个 Core ATen 操作符**，这些是 PyTorch 最核心的张量运算。通过将这些操作符映射到 JAX 实现，TorchAx 可以让 PyTorch 代码无缝运行在 TPU 上。

**PyTorch Core ATen 操作符总数约 2000+**，TorchAx 实现了最常用的 382 个（约 19%）。未实现的操作符列表见文档末尾。

---

## 操作符分类目录

### 1. 张量形状操作 (Shape Operations) - 27 个

这类操作符用于改变张量的形状和维度布局，不改变数据内容。在神经网络中用于数据格式转换（如 NCHW ↔ NHWC）、批处理维度调整等。

| 操作符 | JAX 实现 | 详细说明 | 源码位置 |
|--------|----------|----------|----------|
| `view_copy` | `jnp.reshape` | 返回具有指定形状的张量视图副本。形状的一个维度可以是 -1，表示自动推断。数据在内存中必须是连续的。 | [L62](../../../../torchax/torchax/ops/jaten.py#L62) |
| `view` | `jnp.reshape` | 返回张量的新视图，共享底层数据。这是零拷贝操作，要求输入张量内存连续。在 JAX 中实现为 reshape。 | [L62](../../../../torchax/torchax/ops/jaten.py#L62) |
| `_unsafe_view` | `jnp.reshape` | 不安全的视图操作，跳过连续性检查。仅在确定内存布局正确时使用。用于性能关键路径。 | [L62](../../../../torchax/torchax/ops/jaten.py#L62) |
| `reshape` | `jnp.reshape` | 通用的张量形状重塑。如果可能返回视图，否则返回副本。比 view 更灵活，不要求内存连续。 | [L62](../../../../torchax/torchax/ops/jaten.py#L62) |
| `expand` | `jnp.broadcast_to` | 沿大小为 1 的维度扩展张量，返回原始张量的视图。扩展不分配新内存，只改变步长。例如 (3,1) → (3,4)。 | [L704](../../../../torchax/torchax/ops/jaten.py#L704) |
| `expand_copy` | `jnp.broadcast_to` | expand 的副本版本，返回新分配的张量而非视图。用于需要修改扩展结果的场景。 | [L704](../../../../torchax/torchax/ops/jaten.py#L704) |
| `squeeze` | `jnp.squeeze` | 移除张量中大小为 1 的维度。可指定具体维度，或移除所有单一维度。例如 (1,3,1,4) → (3,4)。 | [L1020](../../../../torchax/torchax/ops/jaten.py#L1020) |
| `squeeze_copy` | `jnp.squeeze` | squeeze 的副本版本。在需要保留原始张量不变时使用。 | [L1020](../../../../torchax/torchax/ops/jaten.py#L1020) |
| `unsqueeze` | `jnp.expand_dims` | 在指定位置插入大小为 1 的新维度。例如 shape(3,4) 在 dim=0 → (1,3,4)。常用于添加批处理维度。 | [L840](../../../../torchax/torchax/ops/jaten.py#L840) |
| `unsqueeze_copy` | `jnp.expand_dims` | unsqueeze 的副本版本，返回新张量而非视图。 | [L840](../../../../torchax/torchax/ops/jaten.py#L840) |
| `permute` | `jnp.transpose` | 按指定顺序重新排列张量的维度。例如 (2,0,1) 将 dim2 移到最前。常用于 NCHW ↔ NHWC 格式转换。 | [L833](../../../../torchax/torchax/ops/jaten.py#L833) |
| `permute_copy` | `jnp.transpose` | permute 的副本版本，返回新内存布局的张量。 | [L833](../../../../torchax/torchax/ops/jaten.py#L833) |
| `transpose` | `jnp.swapaxes` | 交换张量的两个指定维度。是 permute 的简化版，只交换两个维度。例如 transpose(0,1) 交换前两维。 | [L348](../../../../torchax/torchax/ops/jaten.py#L348) |
| `transpose_copy` | `jnp.swapaxes` | transpose 的副本版本。 | [L348](../../../../torchax/torchax/ops/jaten.py#L348) |
| `t` | `jnp.transpose` | 2D 张量专用转置，交换第 0 和第 1 维。等价于 transpose(0,1)。对于 1D 张量返回原张量。 | [L343](../../../../torchax/torchax/ops/jaten.py#L343) |
| `numpy_T` | `jnp.transpose` | NumPy 风格的转置属性。对 N 维张量，反转所有维度顺序。等价于 permute(N-1, N-2, ..., 0)。 | [L312](../../../../torchax/torchax/ops/jaten.py#L312) |
| `flatten` | `jnp.reshape` | 将张量的连续维度范围展平为单一维度。默认展平所有维度。例如 (2,3,4) flatten(0,1) → (6,4)。 | [L4975](../../../../torchax/torchax/ops/jaten.py#L4975) |
| `narrow` | `jax.lax.dynamic_slice_in_dim` | 沿指定维度返回张量的窄化切片。参数：维度、起始索引、长度。等价于 tensor[..., start:start+length, ...]。 | [L4969](../../../../torchax/torchax/ops/jaten.py#L4969) |
| `narrow_copy` | `jax.lax.dynamic_slice_in_dim` | narrow 的副本版本，返回新分配的张量。 | [L4969](../../../../torchax/torchax/ops/jaten.py#L4969) |
| `slice` | Python 切片 | 基础切片操作，支持 start:stop:step 语法。这是 Python `__getitem__` 的底层实现。 | [L363](../../../../torchax/torchax/ops/jaten.py#L363) |
| `slice_copy` | Python 切片 | slice 的副本版本，保证返回新张量。 | [L363](../../../../torchax/torchax/ops/jaten.py#L363) |
| `split` | 自定义切片 | 将张量沿指定维度分割为相等大小的块。如果不能整除，最后一块较小。返回元组。 | [L802](../../../../torchax/torchax/ops/jaten.py#L802) |
| `split_copy` | 自定义切片 | split 的副本版本，每个分割块都是新分配的。 | [L802](../../../../torchax/torchax/ops/jaten.py#L802) |
| `split_with_sizes` | 自定义切片 | 按指定大小列表分割张量。例如 sizes=[2,3,1] 将 dim=0 分为大小 2、3、1 的三块。 | [L802](../../../../torchax/torchax/ops/jaten.py#L802) |
| `stack` | `jnp.stack` | 沿新维度堆叠张量序列。所有张量必须形状相同。结果维度数 +1。例如 3 个 (2,3) 张量 → (3,2,3)。 | [L435](../../../../torchax/torchax/ops/jaten.py#L435) |
| `cat` | `jnp.concatenate` | 沿现有维度连接张量序列。该维度大小相加，其他维度必须匹配。例如 (2,3) + (2,4) 沿 dim=1 → (2,7)。 | [L1233](../../../../torchax/torchax/ops/jaten.py#L1233) |
| `movedim` | `jnp.moveaxis` | 将指定维度移动到新位置，其他维度相对顺序不变。支持单个或多个维度移动。 | [L2633](../../../../torchax/torchax/ops/jaten.py#L2633) |
| `unfold` | 自定义实现 | 沿指定维度提取滑动窗口。返回维度增加的张量，新维度包含窗口元素。用于实现滑动窗口操作和卷积。 | [L5698](../../../../torchax/torchax/ops/jaten.py#L5698) |

### 2. 基础算术运算 (Arithmetic Operations) - 30 个

这类操作符是张量计算的基础，支持广播语义。所有操作都是元素级的，可以在 GPU/TPU 上高效并行执行。

| 操作符 | JAX 实现 | 详细说明 | 源码位置 |
|--------|----------|----------|----------|
| `add.Tensor` | `x + y` | 张量加法，支持广播。`out = self + alpha * other`，其中 alpha 默认为 1。结果类型遵循 PyTorch 类型提升规则。 | [L82](../../../../torchax/torchax/ops/jaten.py#L82) |
| `add.Scalar` | `x + y` | 张量与标量加法。标量被广播到张量的所有元素。比张量加法更高效。 | [L82](../../../../torchax/torchax/ops/jaten.py#L82) |
| `sub.Tensor` | `x - y` | 张量减法，`out = self - alpha * other`。支持 alpha 缩放参数。与 add 类似支持广播。 | [L306](../../../../torchax/torchax/ops/jaten.py#L306) |
| `sub.Scalar` | `x - y` | 张量减标量。等价于 `tensor + (-scalar)`。 | [L306](../../../../torchax/torchax/ops/jaten.py#L306) |
| `mul.Tensor` | `x * y` | 元素级张量乘法 (Hadamard 积)。`out[i] = self[i] * other[i]`。不是矩阵乘法。 | [L332](../../../../torchax/torchax/ops/jaten.py#L332) |
| `mul.Scalar` | `x * y` | 张量与标量相乘。等价于缩放操作，常用于梯度更新。 | [L332](../../../../torchax/torchax/ops/jaten.py#L332) |
| `div` | `x / y` | 张量除法，支持 rounding_mode 参数：None(真除)、'trunc'(截断)、'floor'(向下取整)。 | [L507](../../../../torchax/torchax/ops/jaten.py#L507) |
| `true_divide` | `x / y` | 真除法，总是返回浮点结果。即使两个整数张量相除也返回浮点数。Python 3 的 `/` 语义。 | [L533](../../../../torchax/torchax/ops/jaten.py#L533) |
| `floor_divide` | `jnp.floor_divide` | 向下取整除法，`out = floor(self / other)`。对于负数，结果向负无穷方向取整。Python 的 `//` 语义。 | [L5686](../../../../torchax/torchax/ops/jaten.py#L5686) |
| `pow` | `jnp.power` | 幂运算，`out = self ^ exponent`。支持张量或标量指数。对于负底数和非整数指数返回 NaN。 | [L482](../../../../torchax/torchax/ops/jaten.py#L482) |
| `neg` | `-1 * x` | 取负/符号翻转，`out = -self`。等价于 `mul(-1)`。对于复数同时取反实部和虚部。 | [L3099](../../../../torchax/torchax/ops/jaten.py#L3099) |
| `abs` | `jnp.abs` | 绝对值。对于复数返回模 `sqrt(real² + imag²)`。对于整数和浮点数返回非负值。 | [L2304](../../../../torchax/torchax/ops/jaten.py#L2304) |
| `sqrt` | `jnp.sqrt` | 平方根，`out = √self`。对于负数返回 NaN（不是复数）。对于复数使用主平方根。 | [L1725](../../../../torchax/torchax/ops/jaten.py#L1725) |
| `rsqrt` | `jax.lax.rsqrt` | 平方根的倒数，`out = 1/√self`。比 `1/sqrt(x)` 更高效，在归一化层中广泛使用。 | [L698](../../../../torchax/torchax/ops/jaten.py#L698) |
| `reciprocal` | `1 / a` | 倒数，`out = 1/self`。对于 0 返回 inf。在优化器和注意力机制中常用。 | [L2237](../../../../torchax/torchax/ops/jaten.py#L2237) |
| `sign` | `jnp.sign` | 符号函数，返回 -1（负）、0（零）或 1（正）。对于复数返回 `self/abs(self)`。 | [L1846](../../../../torchax/torchax/ops/jaten.py#L1846) |
| `signbit` | `jnp.signbit` | 检查符号位，返回布尔张量。True 表示负数（包括 -0.0）。用于区分 +0 和 -0。 | [L1852](../../../../torchax/torchax/ops/jaten.py#L1852) |
| `fmod` | 自定义实现 | 浮点取模（C 风格），结果符号与被除数相同。`fmod(-3, 2) = -1`。与 remainder 不同。 | [L2733](../../../../torchax/torchax/ops/jaten.py#L2733) |
| `remainder` | `inputs % other` | Python 风格取余，结果符号与除数相同。`remainder(-3, 2) = 1`。满足 `a = (a // b) * b + remainder`。 | [L3212](../../../../torchax/torchax/ops/jaten.py#L3212) |
| `ceil` | `jnp.ceil` | 向上取整（天花板函数），返回 ≥ 输入的最小整数。例如 `ceil(1.2) = 2, ceil(-1.2) = -1`。 | [L1747](../../../../torchax/torchax/ops/jaten.py#L1747) |
| `floor` | `jnp.floor` | 向下取整（地板函数），返回 ≤ 输入的最大整数。例如 `floor(1.8) = 1, floor(-1.2) = -2`。 | [L2715](../../../../torchax/torchax/ops/jaten.py#L2715) |
| `trunc` | `jnp.trunc` | 向零取整/截断，移除小数部分。`trunc(1.8) = 1, trunc(-1.8) = -1`。与 floor 对负数行为不同。 | [L117](../../../../torchax/torchax/ops/jaten.py#L117) |
| `round` | `jnp.round` | 四舍五入到最接近的整数。银行家舍入：0.5 舍入到最近的偶数。`round(0.5) = 0, round(1.5) = 2`。 | [L2281](../../../../torchax/torchax/ops/jaten.py#L2281) |
| `clamp` | `jnp.clip` | 将值限制在 [min, max] 范围内。`out = min(max(self, min_val), max_val)`。用于梯度裁剪、数值稳定性。 | [L2501](../../../../torchax/torchax/ops/jaten.py#L2501) |
| `clamp_min` | `jnp.clip` | 下限截断，`out = max(self, min_val)`。常用于保证非负值，如 ReLU 的简化版。 | [L2507](../../../../torchax/torchax/ops/jaten.py#L2507) |
| `minimum` | `jnp.minimum` | 元素级最小值，`out[i] = min(self[i], other[i])`。遇到 NaN 返回 NaN（与 fmin 不同）。 | [L1761](../../../../torchax/torchax/ops/jaten.py#L1761) |
| `maximum` | `jnp.maximum` | 元素级最大值，`out[i] = max(self[i], other[i])`。遇到 NaN 返回 NaN（与 fmax 不同）。 | [L2298](../../../../torchax/torchax/ops/jaten.py#L2298) |
| `fmax` | `jnp.fmax` | 忽略 NaN 的最大值。如果一个值是 NaN，返回另一个值。两个都是 NaN 时返回 NaN。 | [L2721](../../../../torchax/torchax/ops/jaten.py#L2721) |
| `fmin` | `jnp.fmin` | 忽略 NaN 的最小值。如果一个值是 NaN，返回另一个值。用于有缺失值的数据处理。 | [L2727](../../../../torchax/torchax/ops/jaten.py#L2727) |

### 3. 三角函数和双曲函数 (Trigonometric & Hyperbolic) - 14 个

这类操作符实现数学三角函数和双曲函数。输入为弧度制。在位置编码（Positional Encoding）、旋转位置嵌入（RoPE）等场景广泛使用。

| 操作符 | JAX 实现 | 详细说明 | 源码位置 |
|--------|----------|----------|----------|
| `sin` | `jnp.sin` | 正弦函数，输入为弧度。周期 2π，值域 [-1, 1]。在 Transformer 位置编码中使用：PE(pos, 2i) = sin(pos/10000^(2i/d))。 | [L1486](../../../../torchax/torchax/ops/jaten.py#L1486) |
| `cos` | `jnp.cos` | 余弦函数，输入为弧度。sin 的相移版本，cos(x) = sin(x + π/2)。在位置编码中与 sin 配对使用。 | [L2565](../../../../torchax/torchax/ops/jaten.py#L2565) |
| `tan` | `jnp.tan` | 正切函数，tan(x) = sin(x)/cos(x)。在 x = π/2 + nπ 处未定义。值域 (-∞, +∞)。 | [L1731](../../../../torchax/torchax/ops/jaten.py#L1731) |
| `sinh` | `jnp.sinh` | 双曲正弦，sinh(x) = (e^x - e^(-x))/2。无周期性，在实数域单调递增。用于某些激活函数变体。 | [L1646](../../../../torchax/torchax/ops/jaten.py#L1646) |
| `cosh` | `jnp.cosh` | 双曲余弦，cosh(x) = (e^x + e^(-x))/2。总是 ≥ 1，在 x=0 处最小值为 1。悬链线形状。 | [L2572](../../../../torchax/torchax/ops/jaten.py#L2572) |
| `tanh` | `jnp.tanh` | 双曲正切，tanh(x) = sinh(x)/cosh(x) = (e^x - e^(-x))/(e^x + e^(-x))。值域 (-1, 1)。经典激活函数，输出以零为中心。 | [L1739](../../../../torchax/torchax/ops/jaten.py#L1739) |
| `asin` | `jnp.arcsin` | 反正弦/arcsin，sin 的反函数。定义域 [-1, 1]，值域 [-π/2, π/2]。输入超出范围返回 NaN。 | [L1753](../../../../torchax/torchax/ops/jaten.py#L1753) |
| `acos` | `jnp.arccos` | 反余弦/arccos，cos 的反函数。定义域 [-1, 1]，值域 [0, π]。在计算角度时常用。 | [L1922](../../../../torchax/torchax/ops/jaten.py#L1922) |
| `atan` | `jnp.arctan` | 反正切/arctan，tan 的反函数。定义域 (-∞, +∞)，值域 (-π/2, π/2)。单调递增。 | [L1873](../../../../torchax/torchax/ops/jaten.py#L1873) |
| `atan2` | `jnp.arctan2` | 二参数反正切，atan2(y, x) 返回 (x,y) 的极角。值域 (-π, π]，正确处理所有象限。比 atan(y/x) 更稳定。 | [L2397](../../../../torchax/torchax/ops/jaten.py#L2397) |
| `asinh` | `jnp.arcsinh` | 反双曲正弦，sinh 的反函数。定义域 (-∞, +∞)。公式：asinh(x) = ln(x + √(x² + 1))。 | [L1865](../../../../torchax/torchax/ops/jaten.py#L1865) |
| `acosh` | `jnp.arccosh` | 反双曲余弦，cosh 的反函数。定义域 [1, +∞)，x < 1 返回 NaN。用于某些距离计算。 | [L2271](../../../../torchax/torchax/ops/jaten.py#L2271) |
| `atanh` | `jnp.arctanh` | 反双曲正切，tanh 的反函数。定义域 (-1, 1)，边界返回 ±∞，超出返回 NaN。 | [L1680](../../../../torchax/torchax/ops/jaten.py#L1680) |
| `hypot` | `jnp.hypot` | 直角三角形斜边，hypot(x, y) = √(x² + y²)。避免中间结果溢出，比直接计算更稳定。用于向量模长计算。 | [L2801](../../../../torchax/torchax/ops/jaten.py#L2801) |

### 4. 指数和对数函数 (Exponential & Logarithmic) - 11 个

这类操作符实现指数和对数运算。在 Softmax、交叉熵损失、概率模型中大量使用。部分函数提供数值稳定的特殊实现。

| 操作符 | JAX 实现 | 详细说明 | 源码位置 |
|--------|----------|----------|----------|
| `exp` | `jnp.exp` | 自然指数函数，e^x (e ≈ 2.71828)。增长极快，大输入容易溢出。Softmax 的核心操作。 | [L2665](../../../../torchax/torchax/ops/jaten.py#L2665) |
| `exp2` | `jnp.exp2` | 以 2 为底的指数，2^x。在二进制相关计算中使用，比 pow(2, x) 更高效。 | [L2685](../../../../torchax/torchax/ops/jaten.py#L2685) |
| `expm1` | `jnp.expm1` | e^x - 1，数值稳定版本。当 x 接近 0 时，直接计算 exp(x)-1 会损失精度，expm1 避免此问题。 | [L2675](../../../../torchax/torchax/ops/jaten.py#L2675) |
| `log` | `jnp.log` | 自然对数 ln(x)，exp 的反函数。定义域 (0, +∞)，x ≤ 0 返回 NaN 或 -inf。信息论和交叉熵核心操作。 | [L3013](../../../../torchax/torchax/ops/jaten.py#L3013) |
| `log10` | `jnp.log10` | 以 10 为底对数。log10(x) = ln(x) / ln(10)。用于分贝计算和科学计数法。 | [L3020](../../../../torchax/torchax/ops/jaten.py#L3020) |
| `log2` | `jnp.log2` | 以 2 为底对数。log2(x) = ln(x) / ln(2)。在信息论中表示比特数，计算熵时常用。 | [L3033](../../../../torchax/torchax/ops/jaten.py#L3033) |
| `log1p` | `jnp.log1p` | ln(1 + x)，数值稳定版本。当 x 接近 0 时精度更高。log1p(expm1(x)) ≈ x。用于对数似然计算。 | [L3027](../../../../torchax/torchax/ops/jaten.py#L3027) |
| `logaddexp` | `jnp.logaddexp` | log(exp(a) + exp(b))，数值稳定的 log-sum-exp。避免 exp 溢出。实现：max(a,b) + log1p(exp(-abs(a-b)))。 | [L3068](../../../../torchax/torchax/ops/jaten.py#L3068) |
| `logaddexp2` | `jnp.logaddexp2` | log2(2^a + 2^b)，以 2 为底的 logaddexp。用于信息论中的概率加法。 | [L3074](../../../../torchax/torchax/ops/jaten.py#L3074) |
| `logcumsumexp` | `jax.lax.cumlogsumexp` | 累积 log-sum-exp。对序列计算 log(∑exp(x[0:i]))。用于 CTC 损失等序列模型。 | [L3080](../../../../torchax/torchax/ops/jaten.py#L3080) |
| `logit` | `jnp.log(p/(1-p))` | Logit 函数，sigmoid 的反函数。将概率 p ∈ (0,1) 映射到 (-∞, +∞)。logit(sigmoid(x)) = x。 | [L5664](../../../../torchax/torchax/ops/jaten.py#L5664) |

### 5. 矩阵运算 (Matrix Operations) - 9 个

这类操作符实现矩阵乘法和相关运算。是神经网络计算的核心，全连接层、注意力机制都依赖矩阵乘法。TPU 专为这类运算优化。

| 操作符 | JAX 实现 | 详细说明 | 源码位置 |
|--------|----------|----------|----------|
| `mm` | `x @ y` | 2D 矩阵乘法，(M,K) @ (K,N) → (M,N)。不支持批处理维度。是全连接层的核心操作。 | [L326](../../../../torchax/torchax/ops/jaten.py#L326) |
| `bmm` | `x @ y` | 批量矩阵乘法，(B,M,K) @ (B,K,N) → (B,M,N)。批次内独立计算，批次间并行。注意力计算的核心。 | [L544](../../../../torchax/torchax/ops/jaten.py#L544) |
| `matmul` | `x @ y` | 通用矩阵乘法，支持任意批处理维度和广播。是 @ 运算符的实现。最灵活的矩阵乘法接口。 | [L981](../../../../torchax/torchax/ops/jaten.py#L981) |
| `dot` | `jnp.dot` | 点积/内积。对 1D 向量返回标量 ∑(a[i]*b[i])。对高维数组等价于最后/倒数第二维收缩。 | [L724](../../../../torchax/torchax/ops/jaten.py#L724) |
| `addmm` | `self + alpha * mat1 @ mat2` | 矩阵乘加融合，β*self + α*(mat1 @ mat2)。BLAS GEMM 操作。融合比分步计算更高效。线性层的底层实现。 | [L987](../../../../torchax/torchax/ops/jaten.py#L987) |
| `addmv` | `self + alpha * mat @ vec` | 矩阵-向量乘加，β*self + α*(mat @ vec)。BLAS GEMV 操作。用于偏置加法的融合。 | [L987](../../../../torchax/torchax/ops/jaten.py#L987) |
| `addbmm` | `einsum + add` | 批量矩阵乘加再求和。先批量矩阵乘，再沿批次维度求和加到 self。特殊优化场景使用。 | [L1006](../../../../torchax/torchax/ops/jaten.py#L1006) |
| `outer` | `jnp.outer` | 外积，(M,) ⊗ (N,) → (M,N)。结果矩阵 out[i,j] = a[i] * b[j]。用于构造秩-1 矩阵更新。 | [L3709](../../../../torchax/torchax/ops/jaten.py#L3709) |
| `linear` | `input @ weight.T + bias` | 线性层前向传播。F.linear(input, weight, bias) = input @ weight.T + bias。是 nn.Linear 的底层实现。 | [L5574](../../../../torchax/torchax/ops/jaten.py#L5574) |

### 6. 归约操作 (Reduction Operations) - 19 个

这类操作符沿指定维度聚合数据。是计算损失函数、统计量和规范化的基础。归约会减少张量维度。

| 操作符 | JAX 实现 | 详细说明 | 源码位置 |
|--------|----------|----------|----------|
| `sum` | `jnp.sum` | 求和归约。可指定维度和是否保持维度。keepdim=True 保持被归约维度为 1。用于损失计算和注意力分数归一化。 | [L1717](../../../../torchax/torchax/ops/jaten.py#L1717) |
| `mean` | `jnp.mean` | 算术平均值。sum(x) / numel(x)。在 BatchNorm、损失函数中广泛使用。支持指定维度。 | [L276](../../../../torchax/torchax/ops/jaten.py#L276) |
| `prod` | `jnp.prod` | 元素乘积归约。所有元素相乘得到标量或沿维度相乘。用于计算行列式、概率连乘。 | [L3142](../../../../torchax/torchax/ops/jaten.py#L3142) |
| `min` | `jnp.min, jnp.argmin` | 最小值及其索引。返回 (values, indices) 元组。用于最近邻搜索、损失裁剪。 | [L1450](../../../../torchax/torchax/ops/jaten.py#L1450) |
| `max` | `jnp.max, jnp.argmax` | 最大值及其索引。返回 (values, indices) 元组。用于分类预测、池化操作。 | [L2287](../../../../torchax/torchax/ops/jaten.py#L2287) |
| `amin` | `jnp.amin` | 沿维度取最小值，只返回值不返回索引。比 min 更轻量。用于数值稳定性（如 log-sum-exp 的 max 减法）。 | [L1476](../../../../torchax/torchax/ops/jaten.py#L1476) |
| `amax` | `jnp.amax` | 沿维度取最大值，只返回值不返回索引。Softmax 数值稳定版本中用于减去最大值。 | [L2310](../../../../torchax/torchax/ops/jaten.py#L2310) |
| `argmin` | `jnp.argmin` | 返回最小值的索引位置。对于展平张量返回单一索引，沿维度返回索引张量。 | [L1481](../../../../torchax/torchax/ops/jaten.py#L1481) |
| `argmax` | `jnp.argmax` | 返回最大值的索引位置。分类任务中用于获取预测类别。argmax(logits, dim=-1) 得到预测。 | [L2361](../../../../torchax/torchax/ops/jaten.py#L2361) |
| `any` | `jnp.any` | 逻辑或归约。检查是否存在任意 True 元素。用于条件检查，如 any(isnan(x)) 检测 NaN。 | [L2331](../../../../torchax/torchax/ops/jaten.py#L2331) |
| `var.correction` | `jnp.var` | 方差计算，支持 Bessel 校正。correction=1 使用 N-1（样本方差），correction=0 使用 N（总体方差）。 | [L1497](../../../../torchax/torchax/ops/jaten.py#L1497) |
| `var_mean.correction` | `jnp.var, jnp.mean` | 同时返回方差和均值。比分别计算更高效，因为均值只计算一次。BatchNorm 中使用。 | [L3539](../../../../torchax/torchax/ops/jaten.py#L3539) |
| `cumsum` | `jnp.cumsum` | 累积求和。out[i] = sum(x[0:i+1])。用于积分计算、CTC 解码、前缀和算法。 | [L922](../../../../torchax/torchax/ops/jaten.py#L922) |
| `cumprod` | `jnp.cumprod` | 累积乘积。out[i] = prod(x[0:i+1])。用于连乘概率计算，如扩散模型的 alpha_cumprod。 | [L932](../../../../torchax/torchax/ops/jaten.py#L932) |
| `cummax` | `jax.lax.associative_scan` | 累积最大值。out[i] = max(x[0:i+1])。返回值和索引。用于单调性约束。 | [L874](../../../../torchax/torchax/ops/jaten.py#L874) |
| `cummin` | `jax.lax.associative_scan` | 累积最小值。out[i] = min(x[0:i+1])。返回值和索引。用于排序相关算法。 | [L898](../../../../torchax/torchax/ops/jaten.py#L898) |
| `mode` | `jax.scipy.stats.mode` | 众数（出现次数最多的值）。返回值和计数。用于投票分类器、统计分析。 | [L1460](../../../../torchax/torchax/ops/jaten.py#L1460) |
| `median` | `jnp.quantile` | 中位数（50% 分位数）。对排序数据取中间值。比均值对离群值更鲁棒。 | [L5122](../../../../torchax/torchax/ops/jaten.py#L5122) |
| `nanmedian` | `jnp.nanquantile` | 忽略 NaN 值的中位数。处理缺失数据时使用。先过滤 NaN 再计算中位数。 | [L5139](../../../../torchax/torchax/ops/jaten.py#L5139) |

### 7. 激活函数 (Activation Functions) - 15 个

这类操作符为神经网络引入非线性。没有激活函数，多层网络等价于单层线性变换。现代模型主要使用 ReLU、GELU、SiLU。

| 操作符 | JAX 实现 | 详细说明 | 源码位置 |
|--------|----------|----------|----------|
| `relu` | `jax.nn.relu` | 修正线性单元，relu(x) = max(0, x)。简单高效，但存在"死亡神经元"问题（负输入梯度为 0）。最广泛使用的激活函数。 | [L1228](../../../../torchax/torchax/ops/jaten.py#L1228) |
| `silu` | `jax.nn.silu` | SiLU/Swish 激活，silu(x) = x * sigmoid(x)。平滑非单调，在 Transformer 和 Diffusion 模型中广泛使用。 | [L337](../../../../torchax/torchax/ops/jaten.py#L337) |
| `gelu` | `jax.nn.gelu` | 高斯误差线性单元，gelu(x) = x * Φ(x)。GPT/BERT 等 Transformer 的默认激活。有 tanh 近似和精确版本。 | [L1014](../../../../torchax/torchax/ops/jaten.py#L1014) |
| `sigmoid` | `jax.nn.sigmoid` | Logistic Sigmoid，σ(x) = 1/(1+e^(-x))。输出范围 (0,1)，可解释为概率。二分类输出层和门控机制使用。 | [L1858](../../../../torchax/torchax/ops/jaten.py#L1858) |
| `tanh` | `jnp.tanh` | 双曲正切激活，输出范围 (-1,1)。以零为中心，比 sigmoid 收敛更快。LSTM 和早期 RNN 使用。 | [L1739](../../../../torchax/torchax/ops/jaten.py#L1739) |
| `softmax` | `jax.nn.softmax` | 将 logits 转换为概率分布，∑softmax(x) = 1。分类输出层的标准选择。注意力权重计算核心。 | [L440](../../../../torchax/torchax/ops/jaten.py#L440) |
| `_softmax` | `jax.nn.softmax` | Softmax 的内部实现。带有数值稳定处理（减去最大值）。与 softmax 功能相同但用于内部调用。 | [L440](../../../../torchax/torchax/ops/jaten.py#L440) |
| `_log_softmax` | `jax.nn.log_softmax` | log(softmax(x)) 的数值稳定版本。直接计算比 log(softmax(x)) 更精确。交叉熵损失使用。 | [L3060](../../../../torchax/torchax/ops/jaten.py#L3060) |
| `log_sigmoid` | `jax.nn.log_sigmoid` | log(sigmoid(x)) 的数值稳定版本。用于二分类交叉熵损失。避免 log(0) 问题。 | [L5032](../../../../torchax/torchax/ops/jaten.py#L5032) |
| `leaky_relu` | `jax.nn.leaky_relu` | 带泄漏的 ReLU，负输入有小斜率（默认 0.01）。解决"死亡 ReLU"问题，允许负梯度流动。 | [L3006](../../../../torchax/torchax/ops/jaten.py#L3006) |
| `hardtanh` | `jnp.clip` | 硬双曲正切，clip(x, -1, 1)。tanh 的分段线性近似。计算更快但梯度饱和区更大。 | [L2770](../../../../torchax/torchax/ops/jaten.py#L2770) |
| `glu` | `jax.nn.glu` | 门控线性单元，GLU(a,b) = a ⊗ σ(b)。输入沿维度分半，一半作门控。语言模型中使用。 | [L2764](../../../../torchax/torchax/ops/jaten.py#L2764) |
| `erf` | `jax.lax.erf` | 误差函数，erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt。GELU 的核心组件。值域 (-1, 1)。 | [L2652](../../../../torchax/torchax/ops/jaten.py#L2652) |
| `erfc` | `jax.lax.erfc` | 互补误差函数，erfc(x) = 1 - erf(x)。大 x 时比 1-erf(x) 更精确。用于正态分布尾概率。 | [L4871](../../../../torchax/torchax/ops/jaten.py#L4871) |
| `erfinv` | `jax.lax.erf_inv` | 反误差函数，erf 的逆函数。用于将均匀分布转换为正态分布（Box-Muller 变换的替代）。 | [L2658](../../../../torchax/torchax/ops/jaten.py#L2658) |

### 8. 归一化操作 (Normalization Operations) - 6 个

这类操作符实现各种归一化技术。归一化通过稳定激活值分布加速训练。不同归一化适用于不同场景：BatchNorm 用于 CNN，LayerNorm 用于 Transformer，GroupNorm 用于小批量。

| 操作符 | JAX 实现 | 详细说明 | 源码位置 |
|--------|----------|----------|----------|
| `native_layer_norm` | 自定义实现 | 层归一化。对每个样本独立归一化，沿最后 N 个维度计算均值和方差。Transformer 的标准归一化方式。out = (x - μ) / √(σ² + ε) * γ + β。 | [L945](../../../../torchax/torchax/ops/jaten.py#L945) |
| `native_group_norm` | 自定义实现 | 组归一化。将通道分成 G 组，每组内独立归一化。在批量大小小时比 BatchNorm 更稳定。介于 LayerNorm 和 InstanceNorm 之间。 | [L1512](../../../../torchax/torchax/ops/jaten.py#L1512) |
| `_native_batch_norm_legit` | 自定义实现 | 批量归一化（训练模式）。跨批次维度计算均值和方差，更新运行统计量。改善梯度流，允许更高学习率。 | [L1160](../../../../torchax/torchax/ops/jaten.py#L1160) |
| `_native_batch_norm_legit_no_training` | 自定义实现 | 批量归一化（推理模式）。使用预计算的运行均值和方差，不更新统计量。推理时行为确定性。 | [L1219](../../../../torchax/torchax/ops/jaten.py#L1219) |
| `native_batch_norm` | 自定义实现 | 原生批量归一化接口。同时支持训练和推理模式。返回 (output, running_mean, running_var)。 | [L3719](../../../../torchax/torchax/ops/jaten.py#L3719) |
| `linalg_vector_norm` | 自定义实现 | 向量 p-范数，‖x‖ₚ = (∑|xᵢ|ᵖ)^(1/p)。p=2 为欧几里得范数，p=inf 为最大绝对值。用于梯度裁剪、权重正则化。 | [L1571](../../../../torchax/torchax/ops/jaten.py#L1571) |

### 9. 卷积和池化 (Convolution & Pooling) - 13 个

这类操作符是计算机视觉的核心。卷积提取局部特征，池化降采样减少计算量。TPU 的矩阵单元对卷积有专门优化。

| 操作符 | JAX 实现 | 详细说明 | 源码位置 |
|--------|----------|----------|----------|
| `conv2d` | `_aten_convolution` | 2D 卷积。对图像应用卷积核提取特征。支持 padding、stride、dilation、groups。CNN 的核心操作。 | [L1045](../../../../torchax/torchax/ops/jaten.py#L1045) |
| `convolution` | `jax.lax.conv_general_dilated` | 通用卷积操作。支持 1D/2D/3D、分组卷积、空洞卷积、转置卷积。所有卷积操作的底层实现。 | [L1068](../../../../torchax/torchax/ops/jaten.py#L1068) |
| `avg_pool1d` | 自定义实现 | 1D 平均池化。对序列数据进行降采样，取窗口内平均值。用于时序信号处理。 | [L2141](../../../../torchax/torchax/ops/jaten.py#L2141) |
| `avg_pool2d` | 自定义实现 | 2D 平均池化。对图像进行降采样，取窗口内像素平均值。比 max_pool 更平滑，保留更多信息。 | [L2141](../../../../torchax/torchax/ops/jaten.py#L2141) |
| `avg_pool3d` | 自定义实现 | 3D 平均池化。对视频或体积数据降采样。在时间、高度、宽度三个维度上取平均。 | [L2141](../../../../torchax/torchax/ops/jaten.py#L2141) |
| `max_pool2d_with_indices` | 自定义 `max_pool` | 带索引的 2D 最大池化。返回最大值和其在窗口内的索引。索引用于 max_unpool 的逆操作。 | [L1397](../../../../torchax/torchax/ops/jaten.py#L1397) |
| `max_pool3d_with_indices` | 自定义 `max_pool` | 带索引的 3D 最大池化。用于视频处理，返回时空窗口内的最大值及索引。 | [L1397](../../../../torchax/torchax/ops/jaten.py#L1397) |
| `_adaptive_avg_pool2d` | 自定义实现 | 自适应 2D 平均池化。指定输出尺寸而非核大小，自动计算所需的窗口大小。例如 (H,W) → (1,1) 用于全局平均池化。 | [L2001](../../../../torchax/torchax/ops/jaten.py#L2001) |
| `_adaptive_avg_pool3d` | 自定义实现 | 自适应 3D 平均池化。用于视频分类等需要固定尺寸输出的场景。 | [L2001](../../../../torchax/torchax/ops/jaten.py#L2001) |
| `max_unpool2d` | 自定义实现 | 2D 最大反池化。使用 max_pool 返回的索引将值放回原位置。用于语义分割的上采样。 | [L5245](../../../../torchax/torchax/ops/jaten.py#L5245) |
| `max_unpool3d` | 自定义实现 | 3D 最大反池化。max_pool3d 的逆操作，用于视频分割。 | [L5245](../../../../torchax/torchax/ops/jaten.py#L5245) |
| `reflection_pad1d` | `jnp.pad(..., 'reflect')` | 1D 反射填充。边界值镜像复制，如 [1,2,3] pad(2) → [3,2,1,2,3,2,1]。比零填充更自然。 | [L1631](../../../../torchax/torchax/ops/jaten.py#L1631) |
| `constant_pad_nd` | `jax.lax.pad` | 常量填充。用指定值（通常为 0）填充张量边缘。最常见的卷积填充方式。 | [L2513](../../../../torchax/torchax/ops/jaten.py#L2513) |
| `pad` | `jnp.pad` | 通用填充操作。支持多种模式：constant（常量）、reflect（反射）、replicate（复制边缘）、circular（循环）。 | [L5607](../../../../torchax/torchax/ops/jaten.py#L5607) |

### 10. 索引和选择操作 (Indexing & Selection) - 24 个

这类操作符实现灵活的张量元素访问和修改。是 gather/scatter 操作的基础，在嵌入查找、注意力机制、稀疏操作中广泛使用。

| 操作符 | JAX 实现 | 详细说明 | 源码位置 |
|--------|----------|----------|----------|
| `select` | `jax.lax.index_in_dim` | 沿指定维度选择单个切片。tensor.select(0, 2) 等价于 tensor[2]。减少一个维度。 | [L210](../../../../torchax/torchax/ops/jaten.py#L210) |
| `index_select` | `jnp.take` | 使用索引张量沿维度选择元素。index 是 1D 整数张量。比循环更高效的批量选择。 | [L215](../../../../torchax/torchax/ops/jaten.py#L215) |
| `select_copy` | `jnp.take` | index_select 的副本版本，返回新分配的张量。 | [L215](../../../../torchax/torchax/ops/jaten.py#L215) |
| `index` | Python 索引 | 高级索引，支持多维索引张量和布尔掩码。tensor[idx1, idx2] 的底层实现。 | [L793](../../../../torchax/torchax/ops/jaten.py#L793) |
| `_unsafe_index` | Python 索引 | 不安全索引，跳过边界检查。在确保索引有效时使用，提升性能。 | [L793](../../../../torchax/torchax/ops/jaten.py#L793) |
| `index_put` | `at[].set/add` | 高级索引赋值。tensor[idx] = value 的实现。支持 accumulate=True 累加模式。 | [L783](../../../../torchax/torchax/ops/jaten.py#L783) |
| `_unsafe_index_put` | `_aten_index_put` | 不安全索引赋值，跳过边界检查。 | [L5017](../../../../torchax/torchax/ops/jaten.py#L5017) |
| `index_copy` | `at[].set` | 按索引复制值到目标张量。out.index_copy_(dim, index, src) 将 src 复制到 out 的指定位置。 | [L123](../../../../torchax/torchax/ops/jaten.py#L123) |
| `gather` | 自定义实现 | 沿维度收集元素。out[i][j][k] = input[index[i][j][k]][j][k] (dim=0)。注意力机制和 beam search 使用。 | [L2745](../../../../torchax/torchax/ops/jaten.py#L2745) |
| `scatter` | `at[].set` | gather 的逆操作。将 src 值散射到 self 的指定位置。self[index[i][j][k]][j][k] = src[i][j][k]。 | [L1881](../../../../torchax/torchax/ops/jaten.py#L1881) |
| `scatter_add` | `at[].add` | 散射加法。目标位置累加而非替换。用于实现稀疏梯度累加、one-hot 编码统计。 | [L1797](../../../../torchax/torchax/ops/jaten.py#L1797) |
| `scatter_reduce` | `at[].add/mul/max/min` | 散射归约操作。支持 sum、prod、mean、amax、amin 等归约方式。图神经网络中的消息聚合。 | [L1880](../../../../torchax/torchax/ops/jaten.py#L1880) |
| `select_scatter` | `at[].set` | 将切片值散射回完整张量的指定位置。select 操作的逆过程。 | [L2244](../../../../torchax/torchax/ops/jaten.py#L2244) |
| `slice_scatter` | `at[].set` | 将切片值散射回完整张量。支持 start:end:step 范围。 | [L3233](../../../../torchax/torchax/ops/jaten.py#L3233) |
| `diagonal_scatter` | `at[].set` | 将值散射到张量的对角线位置。用于构造对角矩阵或修改对角元素。 | [L2607](../../../../torchax/torchax/ops/jaten.py#L2607) |
| `masked_scatter` | `at[].set` | 掩码散射。将 src 值按 mask 为 True 的位置填入 self。用于条件更新。 | [L1806](../../../../torchax/torchax/ops/jaten.py#L1806) |
| `masked_select` | 条件索引 | 根据布尔掩码选择元素，返回 1D 张量。等价于 tensor[mask]。用于过滤有效数据。 | [L1826](../../../../torchax/torchax/ops/jaten.py#L1826) |
| `take` | `self.flatten()[index]` | 将输入视为 1D 数组取元素。与 index_select 不同，无视原始维度。 | [L5601](../../../../torchax/torchax/ops/jaten.py#L5601) |
| `put` | `jnp.put` | take 的逆操作。将值放入展平张量的指定位置。支持累加模式。 | [L3149](../../../../torchax/torchax/ops/jaten.py#L3149) |
| `topk` | `jax.lax.top_k` | 返回最大的 k 个元素及其索引。用于 beam search、排序截断。可选 largest=False 取最小 k 个。 | [L3259](../../../../torchax/torchax/ops/jaten.py#L3259) |
| `kthvalue` | `jnp.partition` | 返回第 k 小的元素及其索引。比完全排序更高效。用于中位数等分位数计算。 | [L5581](../../../../torchax/torchax/ops/jaten.py#L5581) |
| `where` | `jnp.where` | 条件选择。where(cond, x, y) = cond ? x : y。三元素版本返回满足条件的坐标（与 nonzero 类似）。 | [L3496](../../../../torchax/torchax/ops/jaten.py#L3496) |
| `nonzero` | `jnp.nonzero` | 返回非零元素的索引坐标。结果形状动态依赖于输入值。用于稀疏张量构造。 | [L3127](../../../../torchax/torchax/ops/jaten.py#L3127) |
| `nonzero_static` | `jnp.argwhere` | 静态形状的 nonzero。指定最大非零元素数，不足时填充。适合 JIT 编译。 | [L3109](../../../../torchax/torchax/ops/jaten.py#L3109) |

### 11. 比较和逻辑操作 (Comparison & Logical) - 15 个

这类操作符实现元素级比较和布尔运算。返回布尔张量，用于条件选择、掩码构建、边界检查。

| 操作符 | JAX 实现 | 详细说明 | 源码位置 |
|--------|----------|----------|----------|
| `eq` | `==` | 元素级相等比较。返回布尔张量。支持广播。用于构建相等掩码、准确率计算。 | [L2639](../../../../torchax/torchax/ops/jaten.py#L2639) |
| `ne` | `jnp.not_equal` | 元素级不等比较。eq 的逻辑取反。用于过滤特定值、构建不等掩码。 | [L848](../../../../torchax/torchax/ops/jaten.py#L848) |
| `lt` | `<` | 小于比较。对浮点数比较时注意精度问题。用于阈值判断、排序验证。 | [L1941](../../../../torchax/torchax/ops/jaten.py#L1941) |
| `le` | `<=` | 小于或等于比较。等价于 ~gt。用于边界检查、范围验证。 | [L3000](../../../../torchax/torchax/ops/jaten.py#L3000) |
| `gt` | `>` | 大于比较。用于激活函数掩码（如 ReLU 掩码：x > 0）、阈值过滤。 | [L1934](../../../../torchax/torchax/ops/jaten.py#L1934) |
| `ge` | `>=` | 大于或等于比较。等价于 ~lt。用于边界检查。 | [L2759](../../../../torchax/torchax/ops/jaten.py#L2759) |
| `equal` | `jnp.array_equal` | 整个张量相等测试。返回单一布尔值，当所有元素相等时为 True。用于单元测试验证。 | [L2645](../../../../torchax/torchax/ops/jaten.py#L2645) |
| `allclose` | `jnp.allclose` | 近似相等测试。|a-b| ≤ atol + rtol*|b|。用于浮点数比较，考虑数值误差。 | [L3714](../../../../torchax/torchax/ops/jaten.py#L3714) |
| `logical_and` | `jnp.logical_and` | 逻辑与。两输入均为 True 时返回 True。用于组合多个条件掩码。 | [L3040](../../../../torchax/torchax/ops/jaten.py#L3040) |
| `logical_or` | `jnp.logical_or` | 逻辑或。任一输入为 True 时返回 True。用于合并掩码。 | [L3047](../../../../torchax/torchax/ops/jaten.py#L3047) |
| `logical_not` | `jnp.logical_not` | 逻辑非。True ↔ False 翻转。用于掩码取反。 | [L3054](../../../../torchax/torchax/ops/jaten.py#L3054) |
| `logical_xor` | `jnp.logical_xor` | 逻辑异或。两输入不同时返回 True。用于差异检测。 | [L3089](../../../../torchax/torchax/ops/jaten.py#L3089) |
| `isinf` | `jnp.isinf` | 检测无穷值（+inf 或 -inf）。用于数值稳定性检查、异常值过滤。 | [L2989](../../../../torchax/torchax/ops/jaten.py#L2989) |
| `isnan` | `jnp.isnan` | 检测 NaN（Not a Number）值。训练监控、调试必备。any(isnan(x)) 检测梯度爆炸。 | [L2995](../../../../torchax/torchax/ops/jaten.py#L2995) |
| `isfinite` | `jnp.isfinite` | 检测有限值（非 inf 且非 NaN）。isfinite = ~(isinf | isnan)。用于有效值过滤。 | [L391](../../../../torchax/torchax/ops/jaten.py#L391) |

### 12. 位运算 (Bitwise Operations) - 11 个

这类操作符在整数位级别操作。用于低级优化、哈希计算、位掩码操作。在量化模型和特殊编码中使用。

| 操作符 | JAX 实现 | 详细说明 | 源码位置 |
|--------|----------|----------|----------|
| `bitwise_not` | `~self` | 按位取反。每位 0↔1 翻转。~0b1010 = 0b0101（补码表示）。 | [L1694](../../../../torchax/torchax/ops/jaten.py#L1694) |
| `bitwise_and` | `self & other` | 按位与。两位均为 1 时结果为 1。用于位掩码提取、清零特定位。 | [L2404](../../../../torchax/torchax/ops/jaten.py#L2404) |
| `bitwise_or` | `self \| other` | 按位或。任一位为 1 时结果为 1。用于设置特定位、合并标志。 | [L2411](../../../../torchax/torchax/ops/jaten.py#L2411) |
| `bitwise_xor` | `self ^ other` | 按位异或。两位不同时结果为 1。用于切换位、简单加密、校验和。 | [L2417](../../../../torchax/torchax/ops/jaten.py#L2417) |
| `bitwise_left_shift` | `jnp.left_shift` | 算术左移。x << n 等价于 x * 2^n。用于快速乘以 2 的幂。 | [L1700](../../../../torchax/torchax/ops/jaten.py#L1700) |
| `bitwise_right_shift` | `jnp.right_shift` | 算术右移。x >> n 等价于 x // 2^n（保留符号位）。用于快速除以 2 的幂。 | [L1707](../../../../torchax/torchax/ops/jaten.py#L1707) |
| `__lshift__` | `jnp.left_shift` | Python 左移运算符 `<<` 的实现。等价于 bitwise_left_shift。 | [L1700](../../../../torchax/torchax/ops/jaten.py#L1700) |
| `__rshift__` | `jnp.right_shift` | Python 右移运算符 `>>` 的实现。等价于 bitwise_right_shift。 | [L1707](../../../../torchax/torchax/ops/jaten.py#L1707) |
| `__and__` | `self & other` | Python 与运算符 `&` 的实现。对整数为位运算，对布尔为逻辑与。 | [L2404](../../../../torchax/torchax/ops/jaten.py#L2404) |
| `__or__` | `jnp.logical_or` | Python 或运算符 `\|` 的实现。对布尔张量执行逻辑或。 | [L3047](../../../../torchax/torchax/ops/jaten.py#L3047) |
| `__xor__` | `jnp.logical_xor` | Python 异或运算符 `^` 的实现。对布尔张量执行逻辑异或。 | [L3089](../../../../torchax/torchax/ops/jaten.py#L3089) |

### 13. 线性代数 (Linear Algebra) - 30 个

这类操作符实现数值线性代数算法。矩阵分解用于解线性系统、降维、特征分析。科学计算和某些网络层使用。

| 操作符 | JAX 实现 | 详细说明 | 源码位置 |
|--------|----------|----------|----------|
| `cholesky` | `jax.scipy.linalg.cholesky` | Cholesky 分解。将正定对称矩阵 A 分解为 A = LL^T（下三角）。比 LU 分解快一倍。用于高斯过程。 | [L223](../../../../torchax/torchax/ops/jaten.py#L223) |
| `linalg_cholesky_ex` | `jax.scipy.linalg.cholesky` | 扩展 Cholesky 分解。返回分解结果和信息代码，用于检测非正定矩阵。 | [L228](../../../../torchax/torchax/ops/jaten.py#L228) |
| `cholesky_solve` | `jax.scipy.linalg.cho_solve` | 使用 Cholesky 分解求解 Ax=b。给定 L（LL^T=A），求解 x。比直接求逆更稳定高效。 | [L245](../../../../torchax/torchax/ops/jaten.py#L245) |
| `cholesky_inverse` | `jnp.linalg.inv` | 使用 Cholesky 分解计算矩阵逆。A^(-1) = (L^T)^(-1) L^(-1)。用于协方差矩阵求逆。 | [L2556](../../../../torchax/torchax/ops/jaten.py#L2556) |
| `qr` | `jnp.linalg.qr` | QR 分解。A = QR，Q 正交矩阵，R 上三角。用于最小二乘、特征值算法。 | [L5038](../../../../torchax/torchax/ops/jaten.py#L5038) |
| `linalg_qr` | `jnp.linalg.qr` | 线性代数 QR 分解接口。支持不同模式：reduced（经济型）、complete（完整）。 | [L5049](../../../../torchax/torchax/ops/jaten.py#L5049) |
| `linalg_eig` | `jnp.linalg.eig` | 一般方阵特征值分解。A = VΛV^(-1)。返回特征值和特征向量。结果可能是复数。 | [L2828](../../../../torchax/torchax/ops/jaten.py#L2828) |
| `_linalg_eigh` | `jnp.linalg.eigh` | 厄米特/对称矩阵特征值分解。特征值为实数，特征向量正交。比 eig 更稳定快速。 | [L2833](../../../../torchax/torchax/ops/jaten.py#L2833) |
| `_linalg_svd` | `jnp.linalg.svd` | 奇异值分解。A = UΣV^T。用于 PCA、矩阵压缩、伪逆计算。 | [L5069](../../../../torchax/torchax/ops/jaten.py#L5069) |
| `_linalg_slogdet` | `jnp.linalg.slogdet` | 符号和 log 行列式。返回 (sign, logabsdet)，避免行列式溢出。用于概率模型。 | [L5062](../../../../torchax/torchax/ops/jaten.py#L5062) |
| `linalg_inv_ex` | `jnp.linalg.inv` | 扩展矩阵求逆。返回逆矩阵和信息代码。A^(-1) 使得 AA^(-1) = I。 | [L5110](../../../../torchax/torchax/ops/jaten.py#L5110) |
| `linalg_lu` | `jax.lax.linalg.lu` | LU 分解。PA = LU，P 置换，L 下三角，U 上三角。求解线性系统的基础。 | [L2923](../../../../torchax/torchax/ops/jaten.py#L2923) |
| `linalg_lu_factor_ex` | `jax.lax.linalg.lu` | LU 因式分解，紧凑格式。L 和 U 存储在同一矩阵，另返回置换索引。 | [L2953](../../../../torchax/torchax/ops/jaten.py#L2953) |
| `linalg_lu_solve` | `jax.scipy.linalg.lu_solve` | 使用 LU 分解求解 Ax=b。先 Ly=Pb，再 Ux=y。多右端项时高效。 | [L2962](../../../../torchax/torchax/ops/jaten.py#L2962) |
| `lu_unpack` | 自定义实现 | 解包紧凑 LU 分解。将合并的 LU 矩阵和置换索引分离为 P、L、U 三个矩阵。 | [L5471](../../../../torchax/torchax/ops/jaten.py#L5471) |
| `linalg_lstsq` | `jnp.linalg.lstsq` | 最小二乘解。求解 min‖Ax-b‖₂。超定系统的近似解，欠定系统的最小范数解。 | [L2838](../../../../torchax/torchax/ops/jaten.py#L2838) |
| `linalg_pinv` | `jnp.linalg.pinv` | Moore-Penrose 伪逆。A⁺ 使得 AA⁺A = A。用于求解非方阵、奇异矩阵的"逆"。 | [L5075](../../../../torchax/torchax/ops/jaten.py#L5075) |
| `_linalg_solve_ex` | `jnp.linalg.solve` | 求解线性系统 Ax = b。比求逆更稳定高效。扩展版返回信息代码。 | [L5081](../../../../torchax/torchax/ops/jaten.py#L5081) |
| `linalg_solve_triangular` | `jax.scipy.linalg.solve_triangular` | 求解三角系统。利用三角结构高效求解。O(n²) 而非 O(n³)。 | [L5096](../../../../torchax/torchax/ops/jaten.py#L5096) |
| `triangular_solve` | `jax.lax.linalg.triangular_solve` | 三角系统求解的低级接口。支持上/下三角、转置、单位对角线选项。 | [L5173](../../../../torchax/torchax/ops/jaten.py#L5173) |
| `linalg_matrix_exp` | `jax.scipy.linalg.expm` | 矩阵指数 e^A。用于微分方程求解、李群计算。不等于逐元素 exp。 | [L5056](../../../../torchax/torchax/ops/jaten.py#L5056) |
| `linalg_householder_product` | `jax.lax.linalg.householder_product` | Householder 反射器乘积。从 QR 分解的紧凑表示重构 Q 矩阵。 | [L205](../../../../torchax/torchax/ops/jaten.py#L205) |
| `linalg_ldl_factor_ex` | 自定义实现 | LDL 分解。对称矩阵 A = LDL^T，L 下三角，D 对角。不要求正定。 | [L2899](../../../../torchax/torchax/ops/jaten.py#L2899) |
| `triu` | `jnp.triu` | 提取上三角部分。对角线及以上保留，以下置零。参数 k 控制对角线偏移。 | [L358](../../../../torchax/torchax/ops/jaten.py#L358) |
| `tril_indices` | `jnp.tril_indices` | 返回下三角索引。用于构造下三角矩阵或提取下三角元素。 | [L3317](../../../../torchax/torchax/ops/jaten.py#L3317) |
| `triu_indices` | `jnp.triu_indices` | 返回上三角索引。用于构造上三角矩阵或提取上三角元素。 | [L3335](../../../../torchax/torchax/ops/jaten.py#L3335) |
| `diag` | `jnp.diag` | 对角线操作。1D 输入创建对角矩阵，2D 输入提取对角线。 | [L2578](../../../../torchax/torchax/ops/jaten.py#L2578) |
| `diagonal` | `jnp.diagonal` | 提取对角线元素。支持多批次维度和对角线偏移。返回 1D 张量。 | [L2584](../../../../torchax/torchax/ops/jaten.py#L2584) |
| `diagflat` | `jnp.diagflat` | 创建对角矩阵。输入展平后放置在对角线。支持偏移量参数。 | [L2628](../../../../torchax/torchax/ops/jaten.py#L2628) |

### 14. 随机数生成 (Random Number Generation) - 13 个

这类操作符生成各种分布的随机数。JAX 使用函数式随机数生成（显式传递 key），TorchAx 封装了 key 管理。用于初始化、Dropout、数据增强、采样。

| 操作符 | JAX 实现 | 详细说明 | 源码位置 |
|--------|----------|----------|----------|
| `rand` | `jax.random.uniform` | 均匀分布 U(0,1)。生成 [0,1) 范围内的浮点数。用于 Dropout 掩码、随机初始化基础。 | [L3686](../../../../torchax/torchax/ops/jaten.py#L3686) |
| `randn` | `jax.random.normal` | 标准正态分布 N(0,1)。深度学习中最常用的随机初始化。扩散模型的噪声生成。 | [L3627](../../../../torchax/torchax/ops/jaten.py#L3627) |
| `randn_like` | `jax.random.normal` | 生成与输入同形状的正态随机数。便捷 API，无需手动指定形状和 dtype。 | [L3670](../../../../torchax/torchax/ops/jaten.py#L3670) |
| `randint` | `jax.random.randint` | 生成 [low, high) 范围内的随机整数。用于随机索引、数据增强（如随机裁剪位置）。 | [L3783](../../../../torchax/torchax/ops/jaten.py#L3783) |
| `randint_like` | `jax.random.randint` | 生成与输入同形状的随机整数。 | [L3808](../../../../torchax/torchax/ops/jaten.py#L3808) |
| `randperm` | `jax.random.permutation` | 生成 0 到 n-1 的随机排列。用于数据打乱、随机采样不重复索引。 | [L3173](../../../../torchax/torchax/ops/jaten.py#L3173) |
| `uniform` | 自定义实现 | 自定义范围的均匀分布 U(low, high)。就地操作版本，用于权重初始化。 | [L3770](../../../../torchax/torchax/ops/jaten.py#L3770) |
| `normal` | 自定义实现 | 自定义均值和标准差的正态分布 N(μ, σ)。就地操作版本。用于参数初始化。 | [L3756](../../../../torchax/torchax/ops/jaten.py#L3756) |
| `bernoulli.p` | `jax.random.uniform` | 伯努利分布。以概率 p 返回 1，概率 1-p 返回 0。Dropout 的核心，生成二值掩码。 | [L3650](../../../../torchax/torchax/ops/jaten.py#L3650) |
| `geometric` | `jax.random.geometric` | 几何分布。首次成功所需的伯努利试验次数。用于某些采样算法。 | [L3664](../../../../torchax/torchax/ops/jaten.py#L3664) |
| `multinomial` | `jax.random.choice` | 多项式采样。按概率分布采样类别索引。语言模型生成、强化学习动作选择。 | [L4942](../../../../torchax/torchax/ops/jaten.py#L4942) |
| `cauchy_` | `jax.random.cauchy` | 柯西分布（洛伦兹分布）。重尾分布，均值和方差未定义。就地操作。 | [L142](../../../../torchax/torchax/ops/jaten.py#L142) |
| `exponential_` | `jax.random.exponential` | 指数分布。无记忆性，用于泊松过程。参数为速率 λ，均值 1/λ。 | [L187](../../../../torchax/torchax/ops/jaten.py#L187) |

### 15. 嵌入操作 (Embedding Operations) - 4 个

这类操作符实现嵌入表查找。将离散 token ID 映射到连续向量表示。Transformer 模型输入的基础。

| 操作符 | JAX 实现 | 详细说明 | 源码位置 |
|--------|----------|----------|----------|
| `embedding` | `jnp.take` | 嵌入表查找。weight[indices] 获取嵌入向量。是 nn.Embedding 的底层实现。支持 padding_idx 设置特定索引为零向量。 | [L579](../../../../torchax/torchax/ops/jaten.py#L579) |
| `embedding_renorm_` | 自定义实现 | 嵌入向量重归一化。将嵌入向量的范数限制在 max_norm 以下。用于正则化，防止嵌入过大。 | [L589](../../../../torchax/torchax/ops/jaten.py#L589) |
| `_embedding_bag` | 自定义实现 | 嵌入包操作。将多个嵌入向量聚合为一个（sum/mean/max）。用于变长输入的高效处理，如文档表示。 | [L613](../../../../torchax/torchax/ops/jaten.py#L613) |
| `_embedding_bag_forward_only` | 自定义实现 | 嵌入包的仅前向版本。不保存反向传播所需的中间结果。推理时更高效。 | [L613](../../../../torchax/torchax/ops/jaten.py#L613) |

### 16. FFT 操作 (FFT Operations) - 3 个

这类操作符实现快速傅里叶变换。将信号从时域/空域转换到频域。用于信号处理、卷积加速（频域卷积）、谱归一化。

| 操作符 | JAX 实现 | 详细说明 | 源码位置 |
|--------|----------|----------|----------|
| `_fft_c2c` | `jnp.fft.fftn/ifftn` | 复数到复数 FFT/IFFT。输入输出都是复数。forward=True 执行 FFT，False 执行 IFFT。多维支持。 | [L5188](../../../../torchax/torchax/ops/jaten.py#L5188) |
| `_fft_r2c` | `jnp.fft.rfftn/fftn` | 实数到复数 FFT。利用实数信号的共轭对称性，只返回一半频谱。输出大小为 N/2+1。更高效。 | [L5206](../../../../torchax/torchax/ops/jaten.py#L5206) |
| `_fft_c2r` | `jnp.fft.irfftn` | 复数到实数 IFFT。r2c 的逆操作。从半频谱恢复实数信号。需指定原始信号长度。 | [L5219](../../../../torchax/torchax/ops/jaten.py#L5219) |

### 17. 张量创建 (Tensor Creation) - 12 个

这类操作符创建新张量。是模型初始化和中间变量创建的基础。JAX 强调纯函数式，避免就地创建未初始化内存。

| 操作符 | JAX 实现 | 详细说明 | 源码位置 |
|--------|----------|----------|----------|
| `empty` | `jnp.empty` | 创建未初始化张量。值未定义（可能是任意值）。比 zeros 快但需要后续赋值。谨慎使用。 | [L737](../../../../torchax/torchax/ops/jaten.py#L737) |
| `empty_like` | `jnp.empty_like` | 创建与输入同形状/dtype 的未初始化张量。便捷 API，无需手动指定属性。 | [L743](../../../../torchax/torchax/ops/jaten.py#L743) |
| `ones` | `jnp.ones` | 创建全 1 张量。用于偏置初始化、掩码创建、归一化因子。 | [L749](../../../../torchax/torchax/ops/jaten.py#L749) |
| `zeros` | `jnp.zeros` | 创建全 0 张量。最常用的初始化。用于偏置初始化、padding、累加器初始化。 | [L755](../../../../torchax/torchax/ops/jaten.py#L755) |
| `full` | `jnp.full` | 创建填充指定值的张量。例如 full((3,4), -inf) 创建负无穷张量，用于注意力掩码。 | [L761](../../../../torchax/torchax/ops/jaten.py#L761) |
| `full_like` | `jnp.full` | 创建与输入同形状，填充指定值的张量。便捷版本。 | [L2695](../../../../torchax/torchax/ops/jaten.py#L2695) |
| `arange` | `jnp.arange` | 创建等差序列 [start, end)，步长 step。用于位置编码、索引生成。例如 arange(10) = [0,1,...,9]。 | [L2337](../../../../torchax/torchax/ops/jaten.py#L2337) |
| `empty_permuted` | `jnp.empty` | 创建具有指定内存布局（步长顺序）的空张量。优化特定访问模式。 | [L768](../../../../torchax/torchax/ops/jaten.py#L768) |
| `empty_strided` | `jnp.empty` | 创建具有指定步长的空张量。底层内存布局控制。用于高级优化场景。 | [L776](../../../../torchax/torchax/ops/jaten.py#L776) |
| `scalar_tensor` | `jnp.array` | 将 Python 标量转换为 0 维张量。用于标量参与张量运算。 | [L3553](../../../../torchax/torchax/ops/jaten.py#L3553) |
| `new_empty` | `jnp.empty` | 在现有张量的设备和 dtype 上创建新空张量。便于设备感知的张量创建。 | [L4997](../../../../torchax/torchax/ops/jaten.py#L4997) |
| `new_empty_strided` | `jnp.empty` | 在现有张量的设备上创建具有指定步长的新空张量。 | [L5007](../../../../torchax/torchax/ops/jaten.py#L5007) |

### 18. 张量操作 (Tensor Manipulation) - 18 个

这类操作符对张量进行变换、复制、移动。包括梯度控制、数据类型转换、几何变换等。

| 操作符 | JAX 实现 | 详细说明 | 源码位置 |
|--------|----------|----------|----------|
| `clone` | 返回自身 | 创建张量的深拷贝。在 JAX 中数组不可变，clone 可返回自身。用于断开计算图。 | [L111](../../../../torchax/torchax/ops/jaten.py#L111) |
| `copy` | `jnp.broadcast_to` | 复制数据到目标张量。可能涉及广播。用于张量赋值的底层实现。 | [L2534](../../../../torchax/torchax/ops/jaten.py#L2534) |
| `copy_` | 赋值操作 | 就地复制。将 src 数据复制到 self。在 TorchAx 中通过返回新值模拟。 | [L92](../../../../torchax/torchax/ops/jaten.py#L92) |
| `_to_copy` | `astype/copy` | 类型转换并复制。将张量转换为指定 dtype，返回新张量。 | [L729](../../../../torchax/torchax/ops/jaten.py#L729) |
| `to.dtype` | `astype` | 数据类型转换。将张量元素转换为目标类型。例如 float32 → float16 用于混合精度。 | [L3506](../../../../torchax/torchax/ops/jaten.py#L3506) |
| `to.device` | 返回自身 | 设备转换。在 TorchAx/JAX 中设备由运行时决定，此操作返回自身。 | [L3559](../../../../torchax/torchax/ops/jaten.py#L3559) |
| `detach` | 返回自身 | 从计算图分离，停止梯度传播。推理时和固定参数使用。JAX 用 stop_gradient。 | [L380](../../../../torchax/torchax/ops/jaten.py#L380) |
| `positive` | 返回自身 | 一元正号 +x。语义上不改变值，返回自身。 | [L380](../../../../torchax/torchax/ops/jaten.py#L380) |
| `alias` | 返回自身 | 创建张量别名/视图。在 JAX 中返回自身，因为数组不可变。 | [L1640](../../../../torchax/torchax/ops/jaten.py#L1640) |
| `lift_fresh` | 返回自身 | 将张量提升为新的计算图节点。函数式编程概念，JAX 中返回自身。 | [L3765](../../../../torchax/torchax/ops/jaten.py#L3765) |
| `lift_fresh_copy` | `jnp.copy` | 提升并复制张量。创建新的独立副本。 | [L2529](../../../../torchax/torchax/ops/jaten.py#L2529) |
| `flip` | `jnp.flip` | 沿指定维度翻转张量。例如 [1,2,3] → [3,2,1]。用于数据增强、反向序列。 | [L2706](../../../../torchax/torchax/ops/jaten.py#L2706) |
| `roll` | `jnp.roll` | 循环滚动元素。超出边界的元素从另一侧重新进入。用于位置偏移、循环卷积。 | [L3227](../../../../torchax/torchax/ops/jaten.py#L3227) |
| `repeat` | `jnp.tile` | 沿维度重复张量。参数指定每个维度重复次数。例如 (2,3) repeat(2,2) → (4,6)。 | [L3217](../../../../torchax/torchax/ops/jaten.py#L3217) |
| `repeat_interleave` | `jnp.repeat` | 交错重复元素。[1,2,3] repeat_interleave(2) → [1,1,2,2,3,3]。用于上采样。 | [L412](../../../../torchax/torchax/ops/jaten.py#L412) |
| `sort` | `jnp.sort/argsort` | 排序张量。返回 (sorted_values, indices)。stable=True 保持相等元素顺序。 | [L3244](../../../../torchax/torchax/ops/jaten.py#L3244) |
| `fill` | `jnp.full` | 用指定值填充张量。就地操作，在 JAX 中返回新张量。用于重置张量。 | [L2695](../../../../torchax/torchax/ops/jaten.py#L2695) |
| `resize_` | `jax.numpy.resize` | 调整张量大小。可增大或减小。增大时新元素值未定义。就地操作。 | [L401](../../../../torchax/torchax/ops/jaten.py#L401) |
| `resize_as_` | `jax.numpy.resize` | 调整为与另一张量相同大小。便捷 API。 | [L407](../../../../torchax/torchax/ops/jaten.py#L407) |

### 19. 复数操作 (Complex Number Operations) - 7 个

这类操作符处理复数张量。FFT 输出为复数，某些神经网络使用复数表示。支持实部/虚部访问和格式转换。

| 操作符 | JAX 实现 | 详细说明 | 源码位置 |
|--------|----------|----------|----------|
| `complex` | `real + 1j * imag` | 从实部和虚部创建复数张量。complex(a, b) = a + bi。用于构造复数数据。 | [L171](../../../../torchax/torchax/ops/jaten.py#L171) |
| `real` | `jnp.real` | 提取复数张量的实部。对于实数张量返回自身。结果是实数张量。 | [L396](../../../../torchax/torchax/ops/jaten.py#L396) |
| `imag` | `jnp.imag` | 提取复数张量的虚部。对于实数张量返回零。结果是实数张量。 | [L386](../../../../torchax/torchax/ops/jaten.py#L386) |
| `view_as_real` | `jnp.stack([real, imag])` | 将复数张量视为实数张量。(..., N) 复数 → (..., N, 2) 实数。最后一维是 [real, imag]。 | [L427](../../../../torchax/torchax/ops/jaten.py#L427) |
| `view_as_complex` | `jax.lax.complex` | 将实数张量视为复数。view_as_real 的逆操作。最后一维必须为 2。 | [L499](../../../../torchax/torchax/ops/jaten.py#L499) |
| `conj_physical` | `jnp.conjugate` | 复数共轭。a + bi → a - bi。在 FFT 相关计算中使用。 | [L5022](../../../../torchax/torchax/ops/jaten.py#L5022) |
| `polar` | `jax.lax.complex` | 从极坐标创建复数。polar(abs, angle) = abs * e^(i*angle) = abs*(cos+i*sin)。 | [L5377](../../../../torchax/torchax/ops/jaten.py#L5377) |

### 20. 特殊数学函数 (Special Math Functions) - 26 个

这类操作符实现科学计算中的特殊函数。包括伽马函数、贝塞尔函数、正交多项式等。在物理模拟、概率分布、信号处理中使用。

| 操作符 | JAX 实现 | 详细说明 | 源码位置 |
|--------|----------|----------|----------|
| `special_zeta` | `jax.scipy.special.zeta` | Riemann zeta 函数 ζ(s,q) = Σ(n+q)^(-s)。在数论和物理中重要。用于计算 Bose-Einstein 分布。 | [L254](../../../../torchax/torchax/ops/jaten.py#L254) |
| `igammac` | `jax.scipy.special.gammaincc` | 上不完全 gamma 函数 Q(a,x) = Γ(a,x)/Γ(a)。是 igamma 的补函数。用于卡方分布和泊松分布。 | [L264](../../../../torchax/torchax/ops/jaten.py#L264) |
| `igamma` | `jax.scipy.special.gammainc` | 下不完全 gamma 函数 P(a,x)。γ(a,x)/Γ(a)。累积分布函数的基础。卡方检验使用。 | [L2813](../../../../torchax/torchax/ops/jaten.py#L2813) |
| `digamma` | `jax.scipy.special.digamma` | Digamma 函数 ψ(x) = d/dx ln(Γ(x))。伽马函数的对数导数。用于期望值计算和贝叶斯推断。 | [L2805](../../../../torchax/torchax/ops/jaten.py#L2805) |
| `lgamma` | `jax.scipy.special.gammaln` | 对数伽马函数 ln(Γ(x))。避免大数溢出。用于组合数计算、Dirichlet 分布。 | [L2818](../../../../torchax/torchax/ops/jaten.py#L2818) |
| `mvlgamma` | `jax.scipy.special.multigammaln` | 多元对数伽马函数。Wishart 分布的归一化常数。用于多元统计分析。 | [L2823](../../../../torchax/torchax/ops/jaten.py#L2823) |
| `polygamma` | `jax.lax.polygamma` | Polygamma 函数 ψ^(n)(x)。digamma 的高阶导数。用于高阶矩计算。 | [L4289](../../../../torchax/torchax/ops/jaten.py#L4289) |
| `special_ndtri` | `jax.scipy.special.ndtri` | 标准正态分布的分位数函数（逆 CDF）。将概率 p 映射到标准正态分位数。用于正态采样。 | [L4296](../../../../torchax/torchax/ops/jaten.py#L4296) |
| `special_erfcx` | `exp(x²) * erfc(x)` | 缩放互补误差函数 erfcx(x) = e^(x²) * erfc(x)。大 x 时数值稳定。用于高斯分布计算。 | [L4864](../../../../torchax/torchax/ops/jaten.py#L4864) |
| `i0` | `jax.scipy.special.i0` | 第一类修正贝塞尔函数 I₀(x)。无源圆柱热传导解。Kaiser 窗函数核心。 | [L3859](../../../../torchax/torchax/ops/jaten.py#L3859) |
| `special_i0e` | `jax.scipy.special.i0e` | 缩放 I₀：i0e(x) = e^(-|x|) * I₀(x)。大 x 时避免溢出。数值稳定版本。 | [L3865](../../../../torchax/torchax/ops/jaten.py#L3865) |
| `special_i1` | `jax.scipy.special.i1` | 第一类修正贝塞尔函数 I₁(x)。一阶版本。用于梯度计算和物理模拟。 | [L3871](../../../../torchax/torchax/ops/jaten.py#L3871) |
| `special_i1e` | `jax.scipy.special.i1e` | 缩放 I₁：i1e(x) = e^(-|x|) * I₁(x)。数值稳定版本。 | [L3877](../../../../torchax/torchax/ops/jaten.py#L3877) |
| `special_bessel_j0` | 自定义实现 | 第一类贝塞尔函数 J₀(x)。圆柱波动方程解。傅里叶贝塞尔变换核心。信号处理使用。 | [L4302](../../../../torchax/torchax/ops/jaten.py#L4302) |
| `special_bessel_j1` | 自定义实现 | 第一类贝塞尔函数 J₁(x)。J₀ 的导数相关。圆形孔衍射计算。 | [L4419](../../../../torchax/torchax/ops/jaten.py#L4419) |
| `special_bessel_y0` | 自定义实现 | 第二类贝塞尔函数 Y₀(x)（诺依曼函数）。与 J₀ 线性无关的另一解。边界条件匹配。 | [L4532](../../../../torchax/torchax/ops/jaten.py#L4532) |
| `special_bessel_y1` | 自定义实现 | 第二类贝塞尔函数 Y₁(x)。一阶诺依曼函数。电磁场计算。 | [L4650](../../../../torchax/torchax/ops/jaten.py#L4650) |
| `special_modified_bessel_i0` | 自定义实现 | 修正贝塞尔函数 I₀(x) 的另一实现。用于高精度需求场景。 | [L3924](../../../../torchax/torchax/ops/jaten.py#L3924) |
| `special_modified_bessel_i1` | 自定义实现 | 修正贝塞尔函数 I₁(x) 的另一实现。 | [L4018](../../../../torchax/torchax/ops/jaten.py#L4018) |
| `special_modified_bessel_k0` | 自定义实现 | 第二类修正贝塞尔函数 K₀(x)。指数衰减解。热核和格林函数。 | [L4120](../../../../torchax/torchax/ops/jaten.py#L4120) |
| `special_modified_bessel_k1` | 自定义实现 | 第二类修正贝塞尔函数 K₁(x)。一阶版本。 | [L4203](../../../../torchax/torchax/ops/jaten.py#L4203) |
| `special_chebyshev_polynomial_t` | 自定义实现 | 第一类切比雪夫多项式 Tₙ(x)。cos(n·arccos(x))。最佳多项式逼近、滤波器设计。 | [L4768](../../../../torchax/torchax/ops/jaten.py#L4768) |
| `special_chebyshev_polynomial_u` | 自定义实现 | 第二类切比雪夫多项式 Uₙ(x)。sin((n+1)·arccos(x))/sin(arccos(x))。积分计算。 | [L4814](../../../../torchax/torchax/ops/jaten.py#L4814) |
| `special_hermite_polynomial_h` | 自定义实现 | 物理学家厄米特多项式 Hₙ(x)。量子谐振子波函数。Hₙ = (-1)ⁿ e^(x²) dⁿ/dxⁿ e^(-x²)。 | [L4876](../../../../torchax/torchax/ops/jaten.py#L4876) |
| `special_hermite_polynomial_he` | 自定义实现 | 概率论厄米特多项式 Heₙ(x)。正态分布相关。Heₙ = (-1)ⁿ e^(x²/2) dⁿ/dxⁿ e^(-x²/2)。 | [L4909](../../../../torchax/torchax/ops/jaten.py#L4909) |
| `special_laguerre_polynomial_l` | 自定义实现 | 拉盖尔多项式 Lₙ(x)。氢原子径向波函数。Lₙ(x) = (e^x/n!) dⁿ/dxⁿ (xⁿ e^(-x))。 | [L3883](../../../../torchax/torchax/ops/jaten.py#L3883) |

### 21. 距离计算 (Distance Computation) - 4 个

这类操作符计算向量/矩阵间的距离。是聚类、最近邻、相似度搜索的基础。在推荐系统、图像检索中广泛使用。

| 操作符 | JAX 实现 | 详细说明 | 源码位置 |
|--------|----------|----------|----------|
| `dist` | 范数差 | 两个张量的 p-范数距离 ‖x-y‖ₚ。p=2 为欧几里得距离。用于损失函数（如 L2 loss）。 | [L538](../../../../torchax/torchax/ops/jaten.py#L538) |
| `_cdist_forward` | `jnp.linalg.norm` | 成对距离矩阵（前向）。计算两组点之间的所有距离。(M,D) x (N,D) → (M,N)。用于 k-NN。 | [L2539](../../../../torchax/torchax/ops/jaten.py#L2539) |
| `_pdist_forward` | 自定义实现 | 压缩成对距离（前向）。同一组点内的距离。返回上三角展平形式。节省内存。 | [L2549](../../../../torchax/torchax/ops/jaten.py#L2549) |
| `cdist` | 自定义实现 | 完整成对距离计算。支持多种度量：欧几里得、曼哈顿、闵可夫斯基等。聚类算法核心。 | [L5382](../../../../torchax/torchax/ops/jaten.py#L5382) |

### 22. 上采样 (Upsampling) - 2 个

这类操作符将低分辨率特征图放大到高分辨率。抗锯齿版本通过低通滤波减少伪影。用于图像超分辨率、语义分割、生成模型。

| 操作符 | JAX 实现 | 详细说明 | 源码位置 |
|--------|----------|----------|----------|
| `_upsample_bilinear2d_aa` | 自定义实现 | 双线性上采样（抗锯齿）。使用 2x2 邻域加权平均。抗锯齿滤波器减少阶梯伪影。平滑过渡。 | [L5335](../../../../torchax/torchax/ops/jaten.py#L5335) |
| `_upsample_bicubic2d_aa` | 自定义实现 | 双三次上采样（抗锯齿）。使用 4x4 邻域三次插值。比双线性更平滑但计算量更大。高质量放大。 | [L5356](../../../../torchax/torchax/ops/jaten.py#L5356) |

### 23. 其他操作 (Miscellaneous) - 27 个

这类操作符涵盖各种工具函数：数论运算、统计、广播、唯一值等。是完整张量库的必要补充。

| 操作符 | JAX 实现 | 详细说明 | 源码位置 |
|--------|----------|----------|----------|
| `gcd` | `jnp.gcd` | 最大公约数。gcd(a,b) 返回能整除 a 和 b 的最大正整数。用于分数化简、周期计算。 | [L2977](../../../../torchax/torchax/ops/jaten.py#L2977) |
| `lcm` | `jnp.lcm` | 最小公倍数。lcm(a,b) = |a*b| / gcd(a,b)。用于周期同步、步长计算。 | [L2983](../../../../torchax/torchax/ops/jaten.py#L2983) |
| `bincount` | `jnp.bincount` | 非负整数计数。统计每个值出现次数。用于直方图、混淆矩阵。输出长度为 max(input)+1。 | [L1688](../../../../torchax/torchax/ops/jaten.py#L1688) |
| `bucketize` | `jnp.digitize` | 分桶/离散化。将连续值映射到桶索引。用于特征分箱、分位数离散化。 | [L1039](../../../../torchax/torchax/ops/jaten.py#L1039) |
| `searchsorted` | `jnp.searchsorted` | 排序搜索。在有序数组中找插入位置。二分查找 O(log n)。用于分位数计算。 | [L294](../../../../torchax/torchax/ops/jaten.py#L294) |
| `histc` | `jnp.histogram` | 直方图计算。将值分入指定数量的等宽桶。返回每桶计数。用于数据分析、可视化。 | [L2783](../../../../torchax/torchax/ops/jaten.py#L2783) |
| `frexp` | `jnp.frexp` | 分解浮点数为尾数和指数。x = m * 2^e，m ∈ [0.5, 1)。用于数值分析、精度检查。 | [L2739](../../../../torchax/torchax/ops/jaten.py#L2739) |
| `copysign` | `jnp.copysign` | 复制符号。返回具有 y 符号的 x 绝对值。copysign(|x|, y) = |x| * sign(y)。 | [L3846](../../../../torchax/torchax/ops/jaten.py#L3846) |
| `nextafter` | `jnp.nextafter` | 下一个浮点数。返回从 x 向 y 方向的下一个可表示浮点数。用于精度边界测试。 | [L3104](../../../../torchax/torchax/ops/jaten.py#L3104) |
| `atleast_1d` | `jnp.atleast_1d` | 保证至少 1D。标量 → (1,) 数组，已是数组则不变。用于统一接口。 | [L165](../../../../torchax/torchax/ops/jaten.py#L165) |
| `atleast_2d` | `jnp.atleast_2d` | 保证至少 2D。1D → (1,N)，标量 → (1,1)。用于矩阵运算前的规范化。 | [L160](../../../../torchax/torchax/ops/jaten.py#L160) |
| `broadcast_tensors` | `jax.lax.broadcast_in_dim` | 将多个张量广播到相同形状。不复制数据，只改变视图。用于元素级操作前的对齐。 | [L2423](../../../../torchax/torchax/ops/jaten.py#L2423) |
| `broadcast_to` | `jnp.broadcast_to` | 将张量广播到指定形状。只能扩展大小为 1 的维度。返回视图，不分配内存。 | [L2495](../../../../torchax/torchax/ops/jaten.py#L2495) |
| `as_strided` | 自定义实现 | 创建指定步长的张量视图。底层内存操作，可创建不连续视图。谨慎使用，易出错。 | [L2380](../../../../torchax/torchax/ops/jaten.py#L2380) |
| `as_strided_copy` | 自定义实现 | as_strided 的副本版本。创建新内存而非视图。安全但更慢。 | [L2380](../../../../torchax/torchax/ops/jaten.py#L2380) |
| `as_strided_scatter` | 自定义实现 | 将值散射到步进视图位置。as_strided 的逆操作变体。用于特殊内存布局写入。 | [L2388](../../../../torchax/torchax/ops/jaten.py#L2388) |
| `unique_dim` | `jnp.unique` | 沿维度返回唯一切片。去除指定维度的重复切片。用于去重操作。 | [L3359](../../../../torchax/torchax/ops/jaten.py#L3359) |
| `_unique` | `jnp.unique` | 返回唯一值。去除重复元素。可选返回逆索引和计数。 | [L3396](../../../../torchax/torchax/ops/jaten.py#L3396) |
| `_unique2` | `jnp.unique` | 唯一值的另一版本。返回更多信息（逆索引、计数）。 | [L3415](../../../../torchax/torchax/ops/jaten.py#L3415) |
| `unique_consecutive` | 自定义实现 | 连续唯一值。只去除相邻重复，不排序。[1,1,2,1] → [1,2,1]。用于游程编码。 | [L3429](../../../../torchax/torchax/ops/jaten.py#L3429) |
| `tensor_split` | `jnp.array_split` | 张量分割。可按数量或索引分割。允许不等分。比 split 更灵活。 | [L3621](../../../../torchax/torchax/ops/jaten.py#L3621) |
| `unbind_copy` | `jax.lax.index_in_dim` | 沿维度解绑为张量元组的副本版本。每个切片是独立张量。 | [L3351](../../../../torchax/torchax/ops/jaten.py#L3351) |
| `_trilinear` | `jnp.expand_dims * sum` | 三线性操作。用于 bilinear 层的扩展。输入三个张量进行逐元素乘积再求和。 | [L5233](../../../../torchax/torchax/ops/jaten.py#L5233) |
| `_local_scalar_dense` | `.item()` | 获取单元素张量的 Python 标量值。用于损失打印、条件判断。 | [L3617](../../../../torchax/torchax/ops/jaten.py#L3617) |
| `dim` | `len(shape)` | 返回张量维度数（秩）。0 维是标量，1 维是向量，2 维是矩阵。 | [L3841](../../../../torchax/torchax/ops/jaten.py#L3841) |
| `sym_size` | `shape[dim]` | 返回指定维度的符号大小。用于动态形状推断。 | [L1492](../../../../torchax/torchax/ops/jaten.py#L1492) |
| `is_nonzero` | 自定义实现 | 检查单元素张量是否非零。值为 0 返回 False，否则 True。用于条件判断。 | [L5654](../../../../torchax/torchax/ops/jaten.py#L5654) |

---

### 24. 就地操作 (In-place Operations) - 38 个

就地操作符以下划线 `_` 结尾，在 PyTorch 中直接修改输入张量而不分配新内存。在 JAX/TorchAx 中，由于数组不可变，这些操作实际返回新张量但语义上等同于就地修改。源码位置：Line 5745-5783

**注意：** 在 TorchAx 中，就地操作通过函数式 API 实现 —— 返回修改后的新张量，由上层封装处理"就地"语义。这是 JAX 纯函数式设计与 PyTorch 命令式设计的桥梁。

| 操作符 | 对应的函数式操作符 | 详细说明 | 使用场景 |
|--------|-------------------|----------|----------|
| `add_` | `add` | 就地加法 self += other。累加梯度、更新参数时使用。避免分配新内存。 | SGD 权重更新 |
| `sub_` | `sub` | 就地减法 self -= other。权重衰减、梯度修正使用。 | L2 正则化 |
| `mul_` | `mul` | 就地乘法 self *= other。学习率缩放、Dropout 掩码应用。 | 动量更新 |
| `div_` | `div` | 就地除法 self /= other。归一化、平均梯度计算。 | 梯度累加后除以步数 |
| `pow_` | `pow` | 就地幂运算 self **= exp。数值变换、特征工程。 | Adam 二阶矩更新 |
| `lt_` | `lt` | 就地小于比较。用于原地构建布尔掩码。少见，多用于特殊优化。 | 条件掩码原地更新 |
| `le_` | `le` | 就地小于等于比较。原地掩码生成。 | 阈值过滤 |
| `gt_` | `gt` | 就地大于比较。原地掩码生成。 | ReLU 掩码 |
| `ge_` | `ge` | 就地大于等于比较。原地掩码生成。 | 边界检查 |
| `eq_` | `eq` | 就地等于比较。原地掩码生成。 | 相等性检查 |
| `ne_` | `ne` | 就地不等于比较。原地掩码生成。 | 过滤特殊值 |
| `bernoulli_` | `bernoulli.p` | 就地伯努利采样。用随机 0/1 填充张量。Dropout 核心操作。 | Dropout 掩码生成 |
| `bernoulli_.float` | `_aten_bernoulli` | 就地浮点伯努利采样。结果为浮点数 0.0/1.0。 | 软掩码 |
| `geometric_` | `geometric` | 就地几何分布采样。首次成功所需试验次数。 | 特殊采样算法 |
| `normal_` | `normal` | 就地正态分布采样。用 N(μ,σ) 填充张量。权重初始化核心。 | Xavier/He 初始化 |
| `random_` | `uniform` | 就地随机整数采样 [0, max)。随机索引生成。 | 随机采样 |
| `uniform_` | `uniform` | 就地均匀分布采样 U(a,b)。权重初始化。 | 均匀初始化 |
| `relu_` | `relu` | 就地 ReLU 激活。max(0, x) 原地应用。节省内存。 | 激活层 |
| `squeeze_` | `squeeze` | 就地压缩维度。移除大小为 1 的维度。修改张量形状。 | 维度调整 |
| `sqrt_` | `sqrt` | 就地平方根。数值归一化时使用。 | 标准差计算 |
| `clamp_` | `clamp` | 就地截断到 [min, max]。梯度裁剪、数值稳定性。 | 梯度裁剪 |
| `clamp_min_` | `clamp_min` | 就地下限截断。确保值 >= min。 | 保证非负 |
| `sigmoid_` | `sigmoid` | 就地 Sigmoid 激活。σ(x) = 1/(1+e^(-x))。 | 门控机制 |
| `tanh_` | `tanh` | 就地双曲正切。输出 (-1, 1)。 | LSTM 激活 |
| `ceil_` | `ceil` | 就地向上取整。取 ≥ x 的最小整数。 | 索引计算 |
| `logical_not_` | `logical_not` | 就地逻辑非。True ↔ False 翻转。 | 掩码取反 |
| `unsqueeze_` | `unsqueeze` | 就地扩展维度。在指定位置插入维度 1。 | 添加批次维 |
| `transpose_` | `transpose` | 就地转置。交换两个维度。 | 格式转换 |
| `log_normal_` | `log_normal` | 就地对数正态采样。exp(N(μ,σ))。用于正值分布。 | 特殊初始化 |
| `scatter_add_` | `scatter_add` | 就地散射加法。将值加到指定位置。稀疏梯度累加。 | 嵌入梯度更新 |
| `scatter_reduce_.two` | `scatter_reduce` | 就地散射归约。支持 sum/prod/max/min。 | 图神经网络 |
| `scatter_` | `scatter` | 就地散射赋值。将值写入指定位置。 | 稀疏更新 |
| `bitwise_or_` | `bitwise_or` | 就地按位或。设置特定位为 1。 | 标志合并 |
| `floor_divide_` | `floor_divide` | 就地整除。self //= other。 | 索引计算 |
| `remainder_` | `remainder` | 就地取余。self %= other。 | 周期性计算 |
| `index_put_` | `index_put` | 就地索引赋值。tensor[idx] = val 的底层实现。最常用的就地操作之一。 | 参数更新 |
| `masked_scatter_` | `masked_scatter` | 就地掩码散射。按 mask 为 True 的位置填充值。 | 条件更新 |
| `cauchy_` | `jax.random.cauchy` | 就地柯西分布采样。重尾分布，无均值无方差。 | 鲁棒统计 |
| `exponential_` | `jax.random.exponential` | 就地指数分布采样。λe^(-λx)。泊松过程间隔。 | 随机延迟 |

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

## 未实现的 ATen 操作符

PyTorch Core ATen 库包含约 **2000+ 个操作符**，TorchAx 实现了其中最常用的 **382 个（约 19%）**。以下列出主要的未实现操作符类别，供参考和未来扩展规划。

### 稀疏张量操作 (Sparse Tensor Operations) - ~100+ 个

PyTorch 支持多种稀疏格式（COO、CSR、CSC、BSR、BSC），TorchAx 目前不支持稀疏张量。

| 操作符 | 说明 | 难度 |
|--------|------|------|
| `sparse_coo_tensor` | 创建 COO 格式稀疏张量 | 中等 |
| `sparse_csr_tensor` | 创建 CSR 格式稀疏张量 | 中等 |
| `sparse_csc_tensor` | 创建 CSC 格式稀疏张量 | 中等 |
| `to_sparse` | 密集转稀疏 | 中等 |
| `to_dense` | 稀疏转密集 | 简单 |
| `sparse_resize_` | 调整稀疏张量大小 | 中等 |
| `sparse_mask` | 稀疏掩码 | 中等 |
| `sparse_sampled_addmm` | 稀疏采样矩阵乘加 | 困难 |
| `_sparse_mm` | 稀疏矩阵乘法 | 困难 |
| `_sparse_addmm` | 稀疏矩阵乘加 | 困难 |
| `_sparse_sum` | 稀疏求和 | 中等 |
| `_sparse_softmax` | 稀疏 softmax | 困难 |
| `_sparse_log_softmax` | 稀疏 log softmax | 困难 |
| `coalesce` | 合并重复索引 | 中等 |
| `indices` | 获取稀疏索引 | 简单 |
| `values` | 获取稀疏值 | 简单 |
| `crow_indices` | CSR 行指针 | 简单 |
| `col_indices` | CSR 列索引 | 简单 |

**原因：** JAX 的稀疏支持仍在发展中（`jax.experimental.sparse`），API 不稳定。大多数深度学习工作负载使用密集张量。

### 量化操作 (Quantization Operations) - ~50+ 个

用于模型量化和低精度推理。

| 操作符 | 说明 | 难度 |
|--------|------|------|
| `quantize_per_tensor` | 张量级量化 | 中等 |
| `quantize_per_channel` | 通道级量化 | 中等 |
| `dequantize` | 反量化 | 简单 |
| `fake_quantize_per_tensor_affine` | 伪量化（训练用） | 中等 |
| `fake_quantize_per_channel_affine` | 通道级伪量化 | 中等 |
| `q_scale` | 获取量化缩放因子 | 简单 |
| `q_zero_point` | 获取量化零点 | 简单 |
| `quantized_batch_norm` | 量化批归一化 | 困难 |
| `quantized_max_pool2d` | 量化最大池化 | 困难 |
| `qlinear` | 量化线性层 | 困难 |
| `qconv2d` | 量化卷积 | 困难 |
| `qadd` | 量化加法 | 中等 |
| `qmul` | 量化乘法 | 中等 |
| `qrelu` | 量化 ReLU | 中等 |

**原因：** TPU 原生支持 bfloat16，INT8 量化支持有限。量化主要用于边缘设备部署，非 TPU 主要场景。

### 自动微分特定操作 (Autograd-specific Operations) - ~30+ 个

这些操作符与 PyTorch 的自动微分系统紧密耦合。

| 操作符 | 说明 | 难度 |
|--------|------|------|
| `_backward` | 反向传播入口 | N/A |
| `_make_grads` | 创建梯度张量 | N/A |
| `_grad_input_scale` | 梯度输入缩放 | 中等 |
| `_cudnn_rnn_backward` | cuDNN RNN 反向 | N/A |
| `native_dropout_backward` | Dropout 反向 | 已通过 JAX 处理 |
| `threshold_backward` | 阈值函数反向 | 已通过 JAX 处理 |
| `gelu_backward` | GELU 反向 | 已通过 JAX 处理 |
| `silu_backward` | SiLU 反向 | 已通过 JAX 处理 |
| `hardtanh_backward` | HardTanh 反向 | 已通过 JAX 处理 |
| `leaky_relu_backward` | LeakyReLU 反向 | 已通过 JAX 处理 |
| `elu_backward` | ELU 反向 | 已通过 JAX 处理 |

**原因：** TorchAx 使用 JAX 的 `jax.grad` 进行自动微分，不需要手动实现反向传播函数。

### 分布式操作 (Distributed Operations) - ~40+ 个

多 GPU/TPU 分布式训练相关操作。

| 操作符 | 说明 | 难度 |
|--------|------|------|
| `_all_reduce` | 全局归约 | 通过 JAX 处理 |
| `_all_gather` | 全局收集 | 通过 JAX 处理 |
| `_reduce_scatter` | 归约散射 | 通过 JAX 处理 |
| `_broadcast` | 广播 | 通过 JAX 处理 |
| `_send` | 点对点发送 | 通过 JAX 处理 |
| `_recv` | 点对点接收 | 通过 JAX 处理 |
| `_barrier` | 同步屏障 | 通过 JAX 处理 |
| `_all_to_all` | 全对全通信 | 通过 JAX 处理 |
| `_coalesced_all_reduce` | 合并全局归约 | 通过 JAX 处理 |

**原因：** 分布式通信通过 JAX 的 `jax.pmap`、`jax.shmap` 和 `jax.lax.p*` 集合操作实现，不走 ATen 路径。

### CUDA 特定操作 (CUDA-specific Operations) - ~50+ 个

这些操作符依赖 CUDA 或 cuDNN 特定实现。

| 操作符 | 说明 | 适用性 |
|--------|------|--------|
| `_cudnn_init_dropout_state` | cuDNN Dropout 状态 | CUDA only |
| `_cudnn_rnn` | cuDNN RNN | CUDA only |
| `_cudnn_batch_norm` | cuDNN BatchNorm | 用 native 替代 |
| `_cudnn_ctc_loss` | cuDNN CTC Loss | 可用 JAX 实现 |
| `_cuda_getCurrentStream` | CUDA 流管理 | CUDA only |
| `_cuda_synchronize` | CUDA 同步 | 用 jax.block_until_ready |
| `_triton_*` | Triton 内核相关 | GPU only |
| `_flash_attention_*` | Flash Attention | 可用 JAX Pallas 实现 |

**原因：** TPU 和 GPU 架构不同，CUDA 特定操作不适用于 TPU。

### 序列化操作 (Serialization Operations) - ~10+ 个

模型保存和加载相关。

| 操作符 | 说明 | 处理方式 |
|--------|------|----------|
| `_save_for_mobile` | 移动端序列化 | 不需要 |
| `_jit_to_backend` | JIT 后端转换 | 不需要 |
| `_freeze_module` | 模块冻结 | 不需要 |
| `_load_from_file` | 从文件加载 | 外部处理 |

**原因：** 序列化通过 Python pickle、safetensors 或 JAX 原生方法处理。

### NestedTensor 操作 - ~20+ 个

PyTorch 的嵌套张量（变长序列批处理）。

| 操作符 | 说明 | 难度 |
|--------|------|------|
| `_nested_tensor_from_mask` | 从掩码创建 | 困难 |
| `_nested_tensor_from_tensor_list` | 从列表创建 | 困难 |
| `_nested_view_from_buffer` | 从缓冲区视图 | 困难 |
| `_nested_*` | 其他嵌套操作 | 困难 |

**原因：** JAX 使用 padding + mask 或 `jax.vmap` 处理变长输入。

### 其他专用操作 - ~100+ 个

| 类别 | 示例操作符 | 说明 |
|------|-----------|------|
| 视觉操作 | `roi_align`, `nms`, `deform_conv2d` | 可通过 jax-vision 扩展 |
| 音频操作 | `stft`, `spectrogram` | 部分可用 jax.scipy |
| 文本操作 | `_pack_padded_sequence` | 可手动实现 |
| 图操作 | `edge_index`, `message_passing` | 用 jraph 替代 |
| 调试操作 | `_debug_*`, `_profiler_*` | 开发工具 |
| 内存操作 | `_pin_memory`, `_unify_memory` | 设备特定 |

---

## 实现优先级建议

如果需要扩展 TorchAx 支持，建议按以下优先级实现：

### 高优先级（常用且实现较简单）

1. **更多激活函数**
   - `elu`, `selu`, `celu`, `mish`, `hardswish`, `hardsigmoid`
   
2. **更多池化操作**
   - `lp_pool1d/2d`, `fractional_max_pool2d`

3. **更多归一化**
   - `rms_norm` (LLaMA 等模型使用)

4. **注意力优化**
   - `scaled_dot_product_attention` (PyTorch 2.0+)

### 中优先级

1. **稀疏基础操作**（如果需要稀疏模型）
2. **量化基础操作**（如果需要部署到边缘）
3. **更多 FFT 操作**（信号处理场景）

### 低优先级

1. CUDA 特定操作（TPU 不需要）
2. 序列化操作（外部处理）
3. 调试/Profiler 操作

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
- **复用 PyTorch 成熟的算子分解机制** - 高层 API 自动分解为基础 ATen 操作
- **保持与 PyTorch 的语义一致性** - 相同代码在 GPU 和 TPU 上行为一致
- **利用 JAX 的高效编译和 TPU 优化** - XLA 编译、算子融合、内存优化

**覆盖率分析：**
- 深度学习常用操作：**>95%** 覆盖
- 科学计算特殊函数：**~80%** 覆盖
- 稀疏/量化操作：**0%** 覆盖（设计选择）
- 分布式操作：通过 JAX 原生实现（不走 ATen）

对于大多数 Transformer、CNN、Diffusion 模型，TorchAx 的 382 个操作符已完全足够。

