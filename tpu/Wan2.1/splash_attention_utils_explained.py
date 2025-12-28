"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                        TPU Splash Attention 工具模块                               ║
║                        深度解析版 (Explained Version)                              ║
╚══════════════════════════════════════════════════════════════════════════════════╝

====================================================================================
一、模块总览与设计目标
====================================================================================

本模块实现了针对 Google TPU (特别是 TPU v6e) 优化的 Scaled Dot-Product Attention。
这是 Transformer 模型中最核心也是最耗资源的操作之一。

【核心问题】标准 Attention 的计算复杂度是 O(N²) 其中 N 是序列长度:
  Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
  
当序列长度 N 很大时（如 4096 或更长），直接计算会：
1. 产生巨大的 N×N 注意力矩阵，爆显存/HBM
2. 在 softmax 计算中产生数值不稳定问题

【解决方案】Flash Attention / Splash Attention:
- "Flash Attention" 是 Tri Dao 提出的分块计算算法，减少内存访问
- "Splash Attention" 是 Google 为 TPU 优化的版本
- 核心思想：将 Q、K、V 分成小块，分块计算 attention 并在线更新统计量

【TPU 硬件特点与优化】
TPU v6e 芯片具有以下特点，本模块针对性优化：

1. **MXU (Matrix Multiply Unit)**: 专用矩阵乘法单元
   - 优化点：使用 lax.dot_general 直接调用 MXU 进行高效矩阵乘法
   - bfloat16 输入，float32 累加器

2. **VPU (Vector Processing Unit)**: 向量处理单元
   - 支持 exp2 硬件指令（2^x），比 exp（e^x）更高效
   - 优化点：将标准 softmax 的 exp 转换为 exp2
   - 数学转换：exp(x) = 2^(x * log2(e))，预先乘以 LOG2_E

3. **内存层级**:
   - HBM (High Bandwidth Memory): 大容量高带宽内存，类似 GPU 的全局内存
   - VMEM (Vector Memory): 片上向量内存，类似 GPU 的共享内存，约 16-32MB
   - SMEM (Scalar Memory): 标量内存，用于小数据
   - 优化点：通过 BlockSpec 和 Pallas 自动管理数据在 HBM<->VMEM 之间的流动

4. **多核架构 (Megacore)**:
   - TPU 有多个核心可以并行执行
   - 优化点：通过 dimension_semantics 指定哪些维度可以并行

5. **流水线执行**:
   - TPU 支持计算和内存传输重叠
   - 优化点：Pallas 自动生成流水线代码

====================================================================================
二、Pallas 是什么？
====================================================================================

Pallas 是 JAX 的硬件内核编程接口，类似于：
- NVIDIA 的 CUDA/Triton
- 但是是 JAX 原生的、可以直接写 Python 的内核

Pallas 关键概念：

1. **pallas_call**: 调用 Pallas 内核的主函数
   - grid: 定义并行网格（类似 CUDA 的 block grid）
   - in_specs/out_specs: 定义如何将大数组切分成小块
   - compiler_params: TPU 特定的编译参数

2. **BlockSpec**: 描述如何从大数组中取出小块
   - block_shape: 每个块的形状
   - index_map: 一个函数，根据 program_id 返回块的起始索引

3. **pl.program_id(axis)**: 获取当前执行的网格位置（类似 CUDA 的 blockIdx）

4. **Refs (References)**: 内核函数的参数是引用（类似指针）
   - 读取: x = x_ref[...]
   - 写入: x_ref[...] = x
   - 切片: x_ref[slice, :]

5. **@pl.when(condition)**: 条件执行（类似 if，但是是编译时优化的）

====================================================================================
三、导入模块说明
====================================================================================
"""

# ============================================================================
# 标准库导入
# ============================================================================
import math                    # 数学函数，用于 sqrt 等基础运算
import dataclasses             # Python 数据类装饰器，创建不可变配置对象
import functools               # 函数工具，主要用 partial 创建偏函数

# ============================================================================
# 第三方库导入
# ============================================================================
import numpy as np             # NumPy，用于获取浮点数精度信息等

import jax                     # JAX 核心库 - Google 的高性能数值计算库
                               # JAX 特点：
                               # 1. 可组合的函数变换：jit, grad, vmap, pmap
                               # 2. XLA 编译后端，自动优化计算图
                               # 3. 支持 GPU/TPU 加速

import jax.numpy as jnp        # JAX 的 NumPy 兼容接口
                               # 几乎所有 np.xxx 都有对应的 jnp.xxx
                               # 区别：JAX 数组是不可变的，操作返回新数组

import torch                   # PyTorch，这里仅用于参考实现和类型转换
                               # 本模块主要用 JAX，但与 torchax 集成

from jax import lax            # LAX (Low-level API for XLA)
                               # 提供更底层、更灵活的操作
                               # 如 lax.dot_general（广义矩阵乘法）
                               # 和 lax.fori_loop（编译时展开的循环）

from jax.sharding import PartitionSpec as P
                               # PartitionSpec 用于描述张量如何在设备间分片
                               # 例如 P("dp", None, "tp", None) 表示：
                               # - 第0维按 dp (data parallel) 轴分片
                               # - 第1维不分片
                               # - 第2维按 tp (tensor parallel) 轴分片
                               # - 第3维不分片

from jax.experimental import pallas as pl
                               # Pallas - JAX 的硬件内核编程框架
                               # 类似 NVIDIA Triton，但支持 TPU
                               # 允许用 Python 写接近硬件级别的优化代码

from jax.experimental.pallas import tpu as pltpu
                               # Pallas TPU 特定模块
                               # 提供 TPU 特有的功能：
                               # - CompilerParams: TPU 编译器参数
                               # - PrefetchScalarGridSpec: 带预取的网格规格
                               # - MemorySpace: 指定 VMEM/SMEM/HBM
                               # - repeat: TPU 高效的张量重复操作

from jax.experimental.shard_map import shard_map
                               # shard_map: 在分片上应用函数
                               # 比 pmap 更灵活，支持任意分片策略
                               # 可以指定输入/输出的分片规格


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║                                  常量定义                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
#
# 这些常量经过精心调优，针对 TPU v6e 的硬件特点优化
#

# ====================================================================================
# 四、块大小 (Block Size) 设计哲学
# ====================================================================================
#
# 块大小的选择是 Flash Attention 性能优化的关键。需要平衡：
# 1. 太小：无法充分利用 MXU 矩阵乘法单元
# 2. 太大：超出 VMEM 容量，导致溢出到 HBM
#
# TPU v6e 的 VMEM 约 16-32 MB，需要同时存放：
# - Q 块: (BQSIZE, head_dim)
# - K 块: (BKVSIZE, head_dim)
# - V 块: (BKVSIZE, head_dim)
# - 中间结果: QK^T, softmax 等

BQSIZE = 3328
# ┌─────────────────────────────────────────────────────────────────────────────────┐
# │ BQSIZE = 3328 - Query 块大小                                                     │
# │                                                                                 │
# │ 为什么是 3328？                                                                  │
# │ 1. TPU 的 MXU 对 128 的倍数最高效 (3328 = 26 × 128)                              │
# │ 2. 要能被 NUM_SUBLANES (8) 整除，便于向量化                                       │
# │ 3. 经过性能调优，在 VMEM 容量和计算效率间取得平衡                                  │
# │ 4. 考虑到典型 head_dim (如 128) 下的内存占用                                      │
# └─────────────────────────────────────────────────────────────────────────────────┘

BKVSIZE = 2816
# ┌─────────────────────────────────────────────────────────────────────────────────┐
# │ BKVSIZE = 2816 - Key/Value 外层块大小                                            │
# │                                                                                 │
# │ 这是从 HBM 预取到 VMEM 的块大小                                                  │
# │ 2816 = 22 × 128，同样是 128 的倍数                                              │
# │                                                                                 │
# │ K 和 V 使用相同块大小因为：                                                       │
# │ - 在 attention 计算中，K 和 V 的序列维度必须对齐                                  │
# │ - softmax(QK^T) 的结果直接与 V 相乘                                              │
# └─────────────────────────────────────────────────────────────────────────────────┘

BKVCOMPUTESIZE = 256
# ┌─────────────────────────────────────────────────────────────────────────────────┐
# │ BKVCOMPUTESIZE = 256 - Key/Value 计算块大小                                      │
# │                                                                                 │
# │ 这是实际参与 softmax 在线更新的块大小                                             │
# │ 比 BKVSIZE 小的原因：                                                            │
# │ 1. 更细粒度的块可以提高 softmax 数值稳定性                                        │
# │ 2. 减少每次更新时的累积误差                                                       │
# │ 3. 适配 TPU 的向量单元宽度                                                        │
# │                                                                                 │
# │ 一个 BKVSIZE 块会被分成 BKVSIZE/BKVCOMPUTESIZE = 11 个子块处理                    │
# └─────────────────────────────────────────────────────────────────────────────────┘

BKVCOMPUTEINSIZE = 256
# ┌─────────────────────────────────────────────────────────────────────────────────┐
# │ BKVCOMPUTEINSIZE = 256 - 内层循环块大小                                           │
# │                                                                                 │
# │ 在 softmax 更新的最内层循环中使用                                                 │
# │ 控制每次 exp2 和累加操作的粒度                                                    │
# └─────────────────────────────────────────────────────────────────────────────────┘

# ====================================================================================
# 五、Pallas Kernel 常量
# ====================================================================================

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)
# ┌─────────────────────────────────────────────────────────────────────────────────┐
# │ DEFAULT_MASK_VALUE - 默认掩码值（用于 padding 位置）                              │
# │                                                                                 │
# │ 为什么是 -0.7 * max_float32 而不是 -inf？                                        │
# │                                                                                 │
# │ 1. 使用 -inf 在 softmax 后会产生 0，但计算过程可能出现 NaN                        │
# │ 2. 使用非常大的负数，exp2(x) 后足够接近 0，但避免数值问题                         │
# │ 3. 乘以 0.7 是安全边界，防止乘法后溢出                                           │
# │                                                                                 │
# │ 数值示例：                                                                       │
# │   float32 max ≈ 3.4e38                                                          │
# │   DEFAULT_MASK_VALUE ≈ -2.4e38                                                  │
# │   exp2(-2.4e38) ≈ 0 (在 float32 精度下)                                         │
# └─────────────────────────────────────────────────────────────────────────────────┘

NUM_SUBLANES = 8
# ┌─────────────────────────────────────────────────────────────────────────────────┐
# │ NUM_SUBLANES = 8 - TPU 向量子通道数                                              │
# │                                                                                 │
# │ 这是 TPU VPU 架构的关键参数！                                                     │
# │                                                                                 │
# │ TPU 的向量单元组织方式：                                                          │
# │ - VPU 有 8 个子通道 (sublanes)                                                   │
# │ - 每个子通道可以独立处理数据                                                       │
# │ - 类似于 SIMD 的概念，但更灵活                                                    │
# │                                                                                 │
# │ 在本代码中的作用：                                                                │
# │ - m_scratch 和 l_scratch 的形状是 (8, bq) 而不是 (bq,)                           │
# │ - 这样可以利用所有 8 个子通道并行更新统计量                                        │
# │ - 最终 reduce 到单个值                                                           │
# │                                                                                 │
# │ 这是 TPU 特有的优化，GPU 代码通常不需要这样设计                                    │
# └─────────────────────────────────────────────────────────────────────────────────┘

NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))
# ┌─────────────────────────────────────────────────────────────────────────────────┐
# │ NT_DIM_NUMBERS - dot_general 的维度规格（用于 K^T @ Q 计算）                      │
# │                                                                                 │
# │ lax.dot_general 是 JAX 中最灵活的矩阵乘法操作                                     │
# │ 格式: ((lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch))               │
# │                                                                                 │
# │ NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))                                       │
# │                                                                                 │
# │ 含义：                                                                           │
# │ - lhs (K) 的第 1 维与 rhs (Q) 的第 1 维收缩（做点积）                             │
# │ - 没有 batch 维度                                                               │
# │                                                                                 │
# │ 对于 K: (bkv_compute, head_dim) 和 Q: (bq, head_dim)                            │
# │ 结果是 (bkv_compute, bq) = K @ Q^T 但实际计算的是 Q @ K^T 的转置                  │
# │                                                                                 │
# │ 为什么用 dot_general 而不是 @ 或 jnp.matmul？                                    │
# │ 1. 可以直接指定哪个维度收缩，避免显式转置                                          │
# │ 2. 可以指定累加精度 (preferred_element_type=float32)                             │
# │ 3. 更直接地映射到 TPU MXU 操作                                                   │
# └─────────────────────────────────────────────────────────────────────────────────┘

LOG2_E = 1.44269504
# ┌─────────────────────────────────────────────────────────────────────────────────┐
# │ LOG2_E = log₂(e) ≈ 1.44269504 - exp2 优化的关键常量                              │
# │                                                                                 │
# │ 【这是本模块最重要的优化之一！】                                                   │
# │                                                                                 │
# │ 标准 softmax:                                                                   │
# │   softmax(x) = exp(x - max(x)) / Σexp(x - max(x))                               │
# │                                                                                 │
# │ 问题：TPU VPU 没有高效的 exp (e^x) 指令，但有 exp2 (2^x) 指令                     │
# │                                                                                 │
# │ 数学转换：                                                                       │
# │   exp(x) = e^x = 2^(x × log₂(e)) = exp2(x × LOG2_E)                             │
# │                                                                                 │
# │ 实现方式：                                                                       │
# │   不在内核中乘以 LOG2_E（会增加计算量），而是：                                    │
# │   1. 预先将 Q 乘以 scale × LOG2_E（在调用内核前）                                 │
# │   2. 内核中直接用 exp2(QK^T - max)                                              │
# │                                                                                 │
# │ 性能收益：                                                                       │
# │   exp2 在 TPU VPU 上是单周期指令，exp 需要多个周期模拟                            │
# │   这个优化可以带来 10-20% 的性能提升                                              │
# └─────────────────────────────────────────────────────────────────────────────────┘


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║                              辅助类和函数                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝

@dataclasses.dataclass(frozen=True, slots=True)
class _BlockSizes:
    """
    块大小配置类 - 封装所有与分块计算相关的参数
    
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │ @dataclasses.dataclass 装饰器说明：                                              │
    │                                                                                 │
    │ frozen=True: 创建不可变对象                                                      │
    │   - 一旦创建，属性不能修改                                                        │
    │   - 保证配置不会被意外修改                                                        │
    │   - 可以作为字典键或集合元素（可哈希）                                             │
    │                                                                                 │
    │ slots=True: 使用 __slots__ 优化内存                                              │
    │   - 不创建 __dict__，减少内存开销                                                 │
    │   - 访问属性更快                                                                 │
    │   - 适合创建大量实例的情况                                                        │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    Attributes:
        block_q: Query 块大小，控制每次处理多少个 query token
        block_kv: Key/Value 外层块大小，控制 HBM->VMEM 预取粒度
        block_kv_compute: Key/Value 计算块大小，控制 softmax 更新粒度
    """
    block_q: int           # 例如 3328
    block_kv: int          # 例如 2816
    block_kv_compute: int | None = None  # 例如 256，None 时默认等于 block_kv

    def __post_init__(self):
        """
        后初始化处理：设置 block_kv_compute 的默认值
        
        ┌─────────────────────────────────────────────────────────────────────────────┐
        │ 为什么用 object.__setattr__ 而不是直接赋值？                                  │
        │                                                                             │
        │ 因为 frozen=True，正常的属性赋值会抛出 FrozenInstanceError                   │
        │ object.__setattr__ 绕过了 dataclass 的冻结检查                               │
        │ 这是 dataclass 官方推荐的在 __post_init__ 中设置默认值的方式                   │
        └─────────────────────────────────────────────────────────────────────────────┘
        """
        if self.block_kv_compute is None:
            object.__setattr__(self, "block_kv_compute", self.block_kv)


def _pad_to_multiple(x, multiple, axis):
    """
    将张量在指定轴上 padding 到指定倍数
    
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │ 为什么需要 padding？                                                             │
    │                                                                                 │
    │ Pallas 内核使用固定的块大小处理数据。如果序列长度不是块大小的倍数，                 │
    │ 有两种选择：                                                                     │
    │ 1. 动态处理边界（复杂，性能差）                                                   │
    │ 2. Padding 到倍数（简单，高效）                                                  │
    │                                                                                 │
    │ 本函数选择方案 2，优点：                                                          │
    │ - 内核代码更简单，不需要边界检查                                                   │
    │ - 所有块大小相同，MXU 效率最高                                                    │
    │ - Padding 的额外计算量通常可以忽略（masked 掉不影响结果）                          │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    Args:
        x: 输入 JAX 数组
        multiple: 目标倍数（如 BQSIZE=3328）
        axis: 要 padding 的轴（通常是序列长度维度）
    
    Returns:
        tuple: (padded_array, original_length)
        - padded_array: padding 后的数组
        - original_length: 原始长度（用于后续裁剪）
    
    Example:
        >>> x = jnp.zeros((4, 1000, 128))  # (heads, seq_len, head_dim)
        >>> x_padded, orig_len = _pad_to_multiple(x, 3328, axis=1)
        >>> x_padded.shape
        (4, 3328, 128)  # 1000 -> 3328 (padding 2328 zeros)
        >>> orig_len
        1000
    """
    seq_len = x.shape[axis]
    
    # 计算需要 padding 的长度
    # 例如：seq_len=1000, multiple=3328
    # pad_len = (3328 - 1000 % 3328) % 3328 = (3328 - 1000) % 3328 = 2328
    pad_len = (multiple - seq_len % multiple) % multiple
    
    # 如果已经是倍数，直接返回（避免不必要的内存分配）
    if pad_len == 0:
        return x, seq_len
    
    # 构造 padding 配置
    # pad_width 是一个列表，每个元素是 (before, after) 元组
    # 表示在该维度前面和后面各 padding 多少
    pad_width = [(0, 0)] * x.ndim  # 默认所有维度都不 padding
    pad_width[axis] = (0, pad_len)  # 只在指定轴的后面 padding
    
    # 执行 padding（默认用 0 填充）
    return jnp.pad(x, pad_width), seq_len


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║                      Pallas Flash Attention Kernel (核心内核)                      ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
#
# ====================================================================================
# 六、Flash Attention 算法原理
# ====================================================================================
#
# 标准 Attention 实现:
#   1. 计算 S = Q @ K^T                     # O(N²d) 时间，O(N²) 空间
#   2. 计算 P = softmax(S / sqrt(d))        # 需要存储整个 N×N 矩阵
#   3. 计算 O = P @ V                       # O(N²d) 时间
#
# 问题：当 N=4096, d=128 时，S 矩阵需要 4096×4096×4 = 64MB（float32）
#
# Flash Attention 的核心思想：
#   - 不显式计算/存储完整的 S 矩阵
#   - 将 Q、K、V 分成小块，在 SRAM/VMEM 中完成计算
#   - 使用 "在线 softmax" 算法逐块更新结果
#
# 在线 Softmax (Online Softmax) 算法：
#
#   传统 softmax: softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
#   需要先遍历一次求 max（数值稳定性），再遍历一次求 sum，最后再除
#
#   在线算法核心公式：
#   - 假设已经处理了 x[0:i]，得到了 m_prev = max(x[0:i]), l_prev = Σexp(x[0:i]-m_prev)
#   - 现在要合并 x[i:i+step]：
#     * m_curr = max(x[i:i+step])
#     * m_next = max(m_prev, m_curr)  # 新的全局 max
#     * 更新 l: l_next = l_curr × exp(m_curr - m_next) + l_prev × exp(m_prev - m_next)
#     * 更新 o: o_next = o_prev × exp(m_prev - m_next) + 新贡献
#
#   关键洞察：通过维护 (m, l, o) 三个量，可以增量式更新 softmax 结果
#

def _flash_attention_kernel(
    # ┌─────────────────────────────────────────────────────────────────────────────────┐
    # │ Pallas 内核参数说明                                                              │
    # │                                                                                 │
    # │ Pallas 内核的参数都是 "引用" (Ref)，类似 C 的指针                                 │
    # │ - 读取数据: x = x_ref[...]  (... 表示取全部)                                     │
    # │ - 写入数据: x_ref[...] = x                                                      │
    # │ - 切片读取: x = x_ref[start:end, :]                                             │
    # │                                                                                 │
    # │ 输入 refs 来自 HBM（通过 BlockSpec 自动预取到 VMEM）                              │
    # │ scratch refs 是 VMEM 中的临时存储                                                │
    # │ 输出 refs 最终写回 HBM                                                           │
    # └─────────────────────────────────────────────────────────────────────────────────┘
    
    q_ref,           # Query 块引用，形状 (bq, head_dim)，已在 VMEM 中
    k_ref,           # Key 块引用，形状 (bkv, head_dim)，已在 VMEM 中
    v_ref,           # Value 块引用，形状 (bkv, head_dim)，已在 VMEM 中
    
    m_scratch_ref,   # max 统计量临时存储，形状 (NUM_SUBLANES, bq)
                     # 存储每个 query 位置当前看到的最大值
                     #
                     # ┌─────────────────────────────────────────────────────────────────┐
                     # │ 【重要】为什么是 (NUM_SUBLANES, bq) 而不是 (bq,)？                  │
                     # │                                                                 │
                     # │ 这是 TPU VPU 架构的硬件约束和优化要求：                            │
                     # │                                                                 │
                     # │ 1. **VPU Sublane 架构**：                                        │
                     # │    TPU 的 VPU 有 8 个 sublane (子通道)，类似 8 个独立的向量处理器  │
                     # │    每个 sublane 有自己的寄存器文件和执行单元                       │
                     # │                                                                 │
                     # │ 2. **Pallas 的 VMEM 布局要求**：                                  │
                     # │    在 Pallas 中，scratch buffer 的第一个维度必须是 NUM_SUBLANES   │
                     # │    这样 TPU 编译器才能正确地将数据分配到各个 sublane              │
                     # │                                                                 │
                     # │ 3. **广播与冗余存储**：                                           │
                     # │    实际上所有 8 行存储的是相同的值！                               │
                     # │    m_curr = qk.max(axis=0)[None, :]  # 形状 (1, bq)             │
                     # │    m_next = jnp.maximum(m_prev, m_curr)  # 广播到 (8, bq)       │
                     # │    通过广播，(1, bq) 的值被复制到所有 8 行                         │
                     # │                                                                 │
                     # │ 4. **为什么需要冗余？**                                           │
                     # │    a. pltpu.repeat 操作需要源数据在所有 sublane 上                │
                     # │    b. 某些 VPU 指令要求操作数在特定 sublane 布局                  │
                     # │    c. 这是 TPU Pallas 的底层实现细节，与 XLA 编译优化相关          │
                     # │                                                                 │
                     # │ 5. **性能影响**：                                                 │
                     # │    看似浪费 8x 内存，但：                                          │
                     # │    - scratch 在 VMEM 中，容量充足                                │
                     # │    - 避免了 sublane 间数据重排的开销                              │
                     # │    - 使得后续 pltpu.repeat 可以高效执行                           │
                     # │                                                                 │
                     # │ 这就是为什么 GPU 代码通常是 (bq,)，而 TPU 需要 (8, bq)            │
                     # └─────────────────────────────────────────────────────────────────┘
                     
    l_scratch_ref,   # sum 统计量临时存储，形状 (NUM_SUBLANES, bq)
                     # 存储 Σexp(score - max) 的累积和
                     # 【同样的原因】需要 NUM_SUBLANES 维度以适配 VPU 架构
                     
    o_scratch_ref,   # 输出累积器临时存储，形状 (head_dim_v, bq)
                     # 存储未归一化的 attention 输出
                     # 注意：o_scratch 不需要 NUM_SUBLANES 维度，因为 head_dim_v 通常
                     # 已经是 NUM_SUBLANES 的倍数（如 128 = 8 × 16）
                     
    o_ref,           # 最终输出引用，形状 (head_dim_v, bq)，写回 HBM
    
    # ┌─────────────────────────────────────────────────────────────────────────────────┐
    # │ 静态参数（编译时常量，通过 functools.partial 绑定）                               │
    # └─────────────────────────────────────────────────────────────────────────────────┘
    *,               # * 表示后面的参数必须用关键字指定
    mask_value: float,    # 掩码值，用于 padding 位置
    grid_width: int,      # KV 方向的网格宽度 = kv_seq_len // bkv
    bq: int,              # Query 块大小
    bkv: int,             # Key/Value 外层块大小
    bkv_compute: int,     # Key/Value 计算块大小
    bkv_compute_in: int,  # 内层循环块大小
    head_dim_v: int,      # Value 的 head 维度
):
    """
    Flash Attention 内核（带 exp2 优化）
    
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │ 算法流程图                                                                       │
    │                                                                                 │
    │    ┌──────────────────────────────────────────────────────────────────────┐     │
    │    │                        Grid (并行网格)                                 │     │
    │    │   (num_heads, q_seq_len/bq, kv_seq_len/bkv)                           │     │
    │    │                                                                       │     │
    │    │   每个 (h, i, j) 位置处理:                                              │     │
    │    │   - Q[h, i*bq:(i+1)*bq, :]  (一个 Q 块)                                │     │
    │    │   - K[h, j*bkv:(j+1)*bkv, :] (一个 K 块)                               │     │
    │    │   - V[h, j*bkv:(j+1)*bkv, :] (一个 V 块)                               │     │
    │    └──────────────────────────────────────────────────────────────────────┘     │
    │                                                                                 │
    │    对于固定的 (h, i)，沿 j 方向遍历所有 KV 块:                                    │
    │                                                                                 │
    │    j=0: 初始化 m, l, o                                                          │
    │    j=1,2,...: 更新 m, l, o（在线 softmax）                                       │
    │    j=grid_width-1: 最终归一化 o = o / l                                          │
    │                                                                                 │
    │    关键：沿 j 方向是串行的（arbitrary），沿 h 方向是并行的（parallel）             │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    Note:
        Query 应该在调用前预乘以 LOG2_E，这样内核中可以直接使用 exp2。
        这个优化将 exp(q @ k.T) 转换为 exp2(q * LOG2_E @ k.T)。
    """
    
    # =========================================================================
    # 参数预处理
    # =========================================================================
    
    float32 = jnp.float32  # 使用 float32 进行中间计算，保证精度
    
    # 计算 head_dim_v 需要重复多少次才能填满 NUM_SUBLANES
    # 这是为了利用 TPU 的 8 个向量子通道
    head_dim_v_repeats, rem = divmod(head_dim_v, NUM_SUBLANES)
    if rem != 0:
        # head_dim_v 必须是 8 的倍数，否则无法高效利用 VPU
        raise NotImplementedError(f"{head_dim_v=} should be a multiple of {NUM_SUBLANES}")

    # =========================================================================
    # 获取当前执行位置
    # =========================================================================
    
    # pl.program_id(axis) 返回当前 block 在网格中的位置
    # 类似 CUDA 的 blockIdx.x, blockIdx.y, blockIdx.z
    h, i, j = pl.program_id(0), pl.program_id(1), pl.program_id(2)
    # h: 当前处理的 attention head (0 ~ num_heads-1)
    # i: 当前处理的 Q 块索引 (0 ~ q_seq_len/bq - 1)
    # j: 当前处理的 KV 块索引 (0 ~ kv_seq_len/bkv - 1)

    # =========================================================================
    # 初始化（仅在 j=0 时执行）
    # =========================================================================
    
    @pl.when(j == 0)
    def init():
        """
        初始化 scratch 缓冲区
        
        ┌─────────────────────────────────────────────────────────────────────────────┐
        │ @pl.when(condition) 是 Pallas 的条件执行装饰器                               │
        │                                                                             │
        │ 与 Python 的 if 不同：                                                       │
        │ 1. 编译时生成条件分支代码                                                     │
        │ 2. 内核中所有位置都执行相同代码，只是条件不同                                  │
        │ 3. 没有分支预测失败的开销（TPU 是 VLIW 架构）                                 │
        │                                                                             │
        │ 为什么只在 j=0 时初始化？                                                     │
        │ - 沿 j 方向（KV 方向）是串行处理的                                            │
        │ - j=0 时处理第一个 KV 块，需要初始化状态                                      │
        │ - j>0 时使用前一个 j 留下的状态，实现增量更新                                  │
        └─────────────────────────────────────────────────────────────────────────────┘
        """
        # 输出累积器初始化为 0
        o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)
        
        # max 统计量初始化为非常小的值（实际是大负数）
        # 这样第一个真实值一定会更新 max
        m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)
        
        # sum 统计量初始化为 0
        l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)

    # =========================================================================
    # 主计算循环体
    # =========================================================================
    
    def body(kv_compute_index, _):
        """
        处理一个 KV 计算块
        
        ┌─────────────────────────────────────────────────────────────────────────────┐
        │ 循环结构说明                                                                 │
        │                                                                             │
        │ 外层：pallas_call 的 grid 沿 j 方向遍历 KV 块（每块大小 bkv）                 │
        │ 内层：lax.fori_loop 将每个 bkv 块细分为 bkv/bkv_compute 个子块               │
        │ 最内层：Python for 循环处理 bkv_compute/bkv_compute_in 个更小的块            │
        │                                                                             │
        │ 例如 bkv=2816, bkv_compute=256, bkv_compute_in=256：                         │
        │ - 每个 j 位置处理 2816 个 KV token                                           │
        │ - fori_loop 循环 2816/256 = 11 次                                           │
        │ - 每次处理 256 个 token                                                      │
        │ - 最内层循环 256/256 = 1 次（这里 compute 和 compute_in 相同）               │
        └─────────────────────────────────────────────────────────────────────────────┘
        
        Args:
            kv_compute_index: 当前子块索引 (0 ~ bkv/bkv_compute - 1)
            _: 循环携带值（这里未使用）
        """
        
        # pl.ds(start, size) 创建一个动态切片 (dynamic slice)
        # 用于从 k_ref 和 v_ref 中取出当前子块
        slice_k = pl.ds(kv_compute_index * bkv_compute, bkv_compute)
        
        # 读取之前的统计量
        m_prev, l_prev = m_scratch_ref[...], l_scratch_ref[...]
        assert m_prev.shape == (NUM_SUBLANES, bq)  # 形状检查（编译时验证）
        assert l_prev.shape == (NUM_SUBLANES, bq)

        # =====================================================================
        # 步骤 1: 计算 QK^T 分数
        # =====================================================================
        
        q = q_ref[...]  # 读取整个 Q 块，形状 (bq, head_dim)
        k = k_ref[slice_k, :]  # 读取当前 K 子块，形状 (bkv_compute, head_dim)
        
        # 计算 K @ Q^T，结果形状 (bkv_compute, bq)
        # 等价于 (Q @ K^T)^T，但避免了显式转置
        #
        # NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))
        # - K 的第1维 (head_dim) 与 Q 的第1维 (head_dim) 收缩
        # - 没有 batch 维度
        #
        # preferred_element_type=float32：使用 float32 累加器
        # 即使输入是 bfloat16，中间计算也用 float32，避免精度损失
        qk = lax.dot_general(k, q, NT_DIM_NUMBERS, preferred_element_type=float32)
        assert qk.shape == (bkv_compute, bq)

        # =====================================================================
        # 步骤 2: 在线 Softmax 更新
        # =====================================================================
        
        o_prev = o_scratch_ref[:]  # 读取之前的输出累积
        v = v_ref[slice_k, :].astype(float32)  # 读取当前 V 子块并转为 float32
        step = bkv_compute_in  # 最内层循环步长
        assert qk.shape[0] % step == 0
        
        # 最内层循环：处理 qk 的每个 step 大小的切片
        for i in range(0, qk.shape[0], step):
            # -----------------------------------------------------------------
            # 2.1 计算当前块的 max
            # -----------------------------------------------------------------
            
            # qk[i:i+step] 形状 (step, bq)
            # max(axis=0) 沿 KV 维度求 max，得到每个 Q 位置的当前块最大值
            # [None, :] 添加一个维度，变成 (1, bq)
            m_curr = qk[i:i+step].max(axis=0)[None, :]
            assert m_curr.shape == (1, bq)
            
            # -----------------------------------------------------------------
            # 2.2 更新全局 max
            # -----------------------------------------------------------------
            
            # 取之前的 max 和当前的 max 的较大值
            # 广播：m_prev 是 (NUM_SUBLANES, bq)，m_curr 是 (1, bq)
            # 结果：(NUM_SUBLANES, bq)，所有 sublane 共享相同的 max 值
            m_next = jnp.maximum(m_prev, m_curr)
            assert m_next.shape == (NUM_SUBLANES, bq)

            # -----------------------------------------------------------------
            # 2.3 计算 softmax 分子（使用 exp2 优化！）
            # -----------------------------------------------------------------
            
            # 标准: s = exp(qk - max)
            # 优化: s = exp2(qk - max)，因为 Q 已预乘 LOG2_E
            #
            # 数学等价性：
            #   标准 attention: softmax(Q @ K^T / sqrt(d))
            #   = exp(Q @ K^T / sqrt(d)) / Σexp(...)
            #
            #   本实现: Q' = Q * scale * LOG2_E  (预处理)
            #   exp2(Q' @ K^T) = exp2(Q @ K^T * scale * LOG2_E)
            #                 = 2^(Q @ K^T * scale * LOG2_E)
            #                 = exp(Q @ K^T * scale * LOG2_E * ln(2))
            #                 = exp(Q @ K^T * scale)  (因为 LOG2_E * ln(2) = 1)
            #
            # 【这就是 exp2 优化的精髓！】
            s_curr = jnp.exp2(qk[i:i+step] - m_next[0:1])

            # -----------------------------------------------------------------
            # 2.4 更新 sum 统计量
            # -----------------------------------------------------------------
            
            # l_curr = Σ exp2(当前块的 score - max)
            l_curr = s_curr.sum(axis=0, keepdims=True)
            assert l_curr.shape == (1, bq)

            # 在线更新公式：
            # l_next = l_curr + l_prev × exp2(m_prev - m_next)
            #
            # 直觉：之前累积的 sum 需要根据 max 的变化进行缩放
            # 如果 max 变大了，之前的 sum 需要缩小（因为除以了更大的数）
            alpha = jnp.exp2(m_prev - m_next)  # 缩放因子
            l_next = l_curr + alpha * l_prev

            # -----------------------------------------------------------------
            # 2.5 更新输出累积
            # -----------------------------------------------------------------
            
            # 计算当前块的贡献：S @ V
            # sv_dims = (((0,), (0,)), ((), ())) 表示 V 和 s_curr 的第0维收缩
            # V: (step, head_dim), s_curr: (step, bq)
            # 结果: (head_dim, bq) = V^T @ s_curr
            sv_dims = (((0,), (0,)), ((), ()))
            o_curr = lax.dot_general(v[i:i+step], s_curr, sv_dims)
            
            # 更新输出累积
            # o_next = o_prev × alpha + o_curr
            # alpha 需要只取一行（因为 o 没有 sublane 维度的冗余）
            alpha_o = alpha[0:1, ...]
            o_prev = alpha_o * o_prev + o_curr

            # 更新状态
            m_prev = m_next
            l_prev = l_next

        # 写回更新后的统计量
        m_scratch_ref[...], l_scratch_ref[...] = m_next, l_next
        o_scratch_ref[:] = o_prev

    # =========================================================================
    # 执行主循环
    # =========================================================================
    
    # lax.fori_loop(lower, upper, body, init_val)
    # - 执行 body(i, val) for i in range(lower, upper)
    # - unroll=True: 编译时展开循环，生成直线代码
    #   展开的好处：
    #   1. 没有循环控制开销
    #   2. 编译器可以跨迭代优化
    #   3. 可以与 TPU 的流水线执行完美配合
    #
    #   缺点：编译时间更长，生成的代码更大
    #   对于小循环（如 11 次），展开是值得的
    lax.fori_loop(0, (bkv // bkv_compute), body, None, unroll=True)

    # =========================================================================
    # 最终归一化（仅在 j=grid_width-1 时执行）
    # =========================================================================
    
    @pl.when(j == grid_width - 1)
    def end():
        """
        最终归一化：O = O / L
        
        ┌─────────────────────────────────────────────────────────────────────────────┐
        │ 为什么要在最后才归一化？                                                      │
        │                                                                             │
        │ 1. 在线 softmax 算法中，我们维护的是未归一化的输出                             │
        │    o_scratch 存储的是 Σ exp(score - max) × V                                │
        │                                                                             │
        │ 2. 只有处理完所有 KV 块后，l 才是完整的 Σ exp(score - max)                    │
        │                                                                             │
        │ 3. 最后一次性除以 l，得到正确的 softmax 加权和                                 │
        │    output = o_scratch / l = (Σ exp × V) / (Σ exp) = softmax(score) @ V      │
        └─────────────────────────────────────────────────────────────────────────────┘
        """
        l = l_scratch_ref[...]  # (NUM_SUBLANES, bq)
        
        # 计算 1/l，并重复到 head_dim 维度
        # pltpu.repeat 是 TPU 特有的高效重复操作
        # 将 (NUM_SUBLANES, bq) 扩展到 (head_dim_v, bq)
        l_inv = pltpu.repeat(1.0 / l, head_dim_v_repeats, axis=0)
        
        # 归一化并写入输出
        # o_scratch: (head_dim_v, bq), l_inv: (head_dim_v, bq)
        # 转换回原始精度（可能是 bfloat16）
        o_ref[...] = (o_scratch_ref[...] * l_inv).astype(o_ref.dtype)


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║                      Splash Attention Forward 函数                                ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
#
# ====================================================================================
# 七、pallas_call 详解 - 从 Python 到 TPU 内核
# ====================================================================================
#
# pallas_call 是 Pallas 框架的核心 API，它负责：
# 1. 将 Python 内核函数编译为 TPU 可执行代码
# 2. 设置数据如何从 HBM 传输到 VMEM（通过 BlockSpec）
# 3. 配置并行执行策略（通过 grid 和 dimension_semantics）
# 4. 管理 scratch 内存（临时缓冲区）
#

def _splash_attention_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    block_sizes: _BlockSizes,
    bkv_compute_in: int,
    interpret: bool = False,
):
    """
    Splash Attention 前向传播
    
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │ 函数职责                                                                        │
    │                                                                                 │
    │ 这个函数是内核调用的"编排者"，它：                                                │
    │ 1. 解析输入张量的形状                                                            │
    │ 2. 定义 BlockSpec（如何切分输入/输出）                                           │
    │ 3. 定义 scratch 缓冲区形状                                                       │
    │ 4. 配置 grid（并行执行网格）                                                     │
    │ 5. 调用 pallas_call 执行内核                                                    │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    Args:
        q: Query 张量，形状 (num_q_heads, q_seq_len, head_dim_qk)
        k: Key 张量，形状 (num_kv_heads, kv_seq_len, head_dim_qk)
        v: Value 张量，形状 (num_kv_heads, kv_seq_len, head_dim_v)
        block_sizes: 块大小配置
        bkv_compute_in: 内层循环块大小
        interpret: 是否使用解释模式（用于调试）
    
    Returns:
        输出张量，形状 (num_q_heads, head_dim_v, q_seq_len)
        注意：输出的维度顺序与输入不同！seq_len 在最后
    """
    
    # =========================================================================
    # 解析输入形状
    # =========================================================================
    
    num_q_heads, q_seq_len, head_dim_qk = q.shape
    head_dim_v = v.shape[-1]  # V 的 head_dim 可能与 QK 不同
    bq, bkv = block_sizes.block_q, block_sizes.block_kv
    bkv_compute = block_sizes.block_kv_compute
    num_kv_heads = k.shape[0]
    kv_seq_len = k.shape[1]
    
    # 计算 GQA (Grouped Query Attention) 的 head 比例
    # 例如：如果 Q 有 32 个 head，KV 有 8 个 head
    # 那么每 4 个 Q head 共享一个 KV head
    q_heads_per_kv_head = num_q_heads // num_kv_heads

    # =========================================================================
    # 定义 Index Map 函数
    # =========================================================================
    #
    # Index Map 是 BlockSpec 的核心：给定网格位置 (h, i, j)，返回数据块的起始索引
    #

    def q_index_map(h, i, j, *_):
        """
        Query 的索引映射
        
        ┌─────────────────────────────────────────────────────────────────────────────┐
        │ 给定网格位置 (h, i, j)，返回 Q 块的起始位置                                   │
        │                                                                             │
        │ Q 的形状: (num_q_heads, q_seq_len, head_dim)                                │
        │ 块形状: (1, bq, head_dim)                                                   │
        │                                                                             │
        │ 返回 (h, i, 0) 表示：                                                        │
        │ - 第 h 个 head                                                              │
        │ - 从第 i*bq 个位置开始的 bq 个 token                                         │
        │ - head_dim 维度从 0 开始（取全部）                                           │
        │                                                                             │
        │ 注意：j 参数被忽略！因为对于同一个 (h, i)，无论 j 是多少，                     │
        │ 都需要相同的 Q 块（Q 不随 KV 位置变化）                                       │
        └─────────────────────────────────────────────────────────────────────────────┘
        """
        return (h, i, 0)

    def out_index_map(h, i, j, *_):
        """
        输出的索引映射
        
        ┌─────────────────────────────────────────────────────────────────────────────┐
        │ 输出形状: (num_q_heads, head_dim_v, q_seq_len)                              │
        │ 块形状: (1, head_dim_v, bq)                                                 │
        │                                                                             │
        │ 返回 (h, 0, i) 表示：                                                        │
        │ - 第 h 个 head                                                              │
        │ - head_dim 维度从 0 开始（取全部）                                           │
        │ - 从第 i*bq 个位置开始的 bq 个 token                                         │
        │                                                                             │
        │ 注意维度顺序变化！输入是 (heads, seq, dim)，输出是 (heads, dim, seq)         │
        │ 这是为了优化后续操作的内存布局                                                │
        └─────────────────────────────────────────────────────────────────────────────┘
        """
        return h, 0, i

    def k_index_map(h, i, j, *_):
        """
        Key 的索引映射（支持 GQA）
        
        ┌─────────────────────────────────────────────────────────────────────────────┐
        │ K 的形状: (num_kv_heads, kv_seq_len, head_dim)                              │
        │ 块形状: (1, bkv, head_dim)                                                  │
        │                                                                             │
        │ 返回 (h // q_heads_per_kv_head, j, 0)：                                     │
        │ - h // q_heads_per_kv_head: 计算对应的 KV head 索引                          │
        │   例如 q_heads_per_kv_head=4 时，Q head 0,1,2,3 都映射到 KV head 0          │
        │ - j: KV 序列方向的块索引，从第 j*bkv 个位置开始                               │
        │ - 0: head_dim 从开始取                                                      │
        │                                                                             │
        │ 这就是 GQA (Grouped Query Attention) 的实现：                                │
        │ 多个 Q head 共享同一个 KV head，减少 KV cache 大小                           │
        └─────────────────────────────────────────────────────────────────────────────┘
        """
        return (h // q_heads_per_kv_head, j, 0)

    def v_index_map(h, i, j, *_):
        """Value 的索引映射（与 K 相同）"""
        return (h // q_heads_per_kv_head, j, 0)

    # =========================================================================
    # 定义 BlockSpec（输入规格）
    # =========================================================================
    #
    # BlockSpec 告诉 Pallas：
    # 1. 每个块的形状是什么
    # 2. 如何根据 grid 位置找到块的起始位置
    #

    in_specs = [
        # Q: 每个块取 1 个 head、bq 个 token、全部 head_dim
        # None 表示该维度不分块（每次取整个 head）
        pl.BlockSpec((None, bq, head_dim_qk), q_index_map),
        
        # K: 每个块取 1 个 head、bkv 个 token、全部 head_dim
        pl.BlockSpec((None, bkv, head_dim_qk), k_index_map),
        
        # V: 与 K 相同
        pl.BlockSpec((None, bkv, head_dim_v), v_index_map),
    ]
    
    # =========================================================================
    # 定义输出形状（包括 scratch 缓冲区）
    # =========================================================================
    #
    # pallas_call 的 out_shape 包含两类：
    # 1. scratch 缓冲区：在内核执行期间使用，执行完后丢弃
    # 2. 实际输出：需要写回 HBM 的最终结果
    #
    
    out_shapes = [
        # m_scratch: max 统计量，(NUM_SUBLANES, bq)
        # 使用 jax.ShapeDtypeStruct 定义形状和类型（不实际分配内存）
        jax.ShapeDtypeStruct((NUM_SUBLANES, bq), jnp.float32),
        
        # l_scratch: sum 统计量，(NUM_SUBLANES, bq)
        jax.ShapeDtypeStruct((NUM_SUBLANES, bq), jnp.float32),
        
        # o_scratch: 输出累积器，(head_dim_v, bq)
        jax.ShapeDtypeStruct((head_dim_v, bq), jnp.float32),
        
        # 实际输出：(num_q_heads, head_dim_v, q_seq_len)
        # 使用 q.dtype 保持与输入相同的精度
        jax.ShapeDtypeStruct((num_q_heads, head_dim_v, q_seq_len), q.dtype),
    ]
    
    # =========================================================================
    # 定义 BlockSpec（输出规格）
    # =========================================================================
    
    out_specs = [
        # m_scratch: 不分块，每个 grid 位置都访问完整的 scratch
        # lambda *_: (0, 0) 表示总是从 (0, 0) 开始
        pl.BlockSpec((NUM_SUBLANES, bq), lambda *_: (0, 0)),
        
        # l_scratch: 同上
        pl.BlockSpec((NUM_SUBLANES, bq), lambda *_: (0, 0)),
        
        # o_scratch: 同上
        pl.BlockSpec((head_dim_v, bq), lambda *_: (0, 0)),
        
        # 实际输出：按 head 和 Q 位置分块
        pl.BlockSpec((None, head_dim_v, bq), out_index_map),
    ]
    
    # =========================================================================
    # 定义执行 Grid
    # =========================================================================
    
    grid_width = kv_seq_len // bkv  # KV 方向有多少个块
    grid = (num_q_heads, q_seq_len // bq, grid_width)
    # grid 定义了三个维度：
    # - 维度 0: num_q_heads 个并行任务（每个 head 独立）
    # - 维度 1: q_seq_len // bq 个任务（Q 序列分块）
    # - 维度 2: grid_width 个任务（KV 序列分块）

    # =========================================================================
    # 调用 pallas_call
    # =========================================================================

    all_out = pl.pallas_call(
        # 内核函数：使用 functools.partial 绑定静态参数
        functools.partial(
            _flash_attention_kernel,
            mask_value=DEFAULT_MASK_VALUE,
            grid_width=grid_width,
            bq=bq,
            bkv=bkv,
            bkv_compute=bkv_compute,
            bkv_compute_in=bkv_compute_in,
            head_dim_v=head_dim_v,
        ),
        
        # Grid 规格：使用 TPU 特有的 PrefetchScalarGridSpec
        # 它支持标量预取和更细粒度的内存控制
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,  # 不使用标量预取
            in_specs=in_specs,
            out_specs=out_specs,
            grid=grid,
        ),
        
        # ┌─────────────────────────────────────────────────────────────────────────────┐
        # │ TPU 编译器参数 - 这是 TPU 优化的关键！                                        │
        # └─────────────────────────────────────────────────────────────────────────────┘
        compiler_params=pltpu.CompilerParams(
            # dimension_semantics 定义每个 grid 维度的语义：
            # - "parallel": 可以在多核间并行执行，独立无依赖
            # - "arbitrary": 可能有依赖关系，需要串行或特殊处理
            #
            # ("parallel", "arbitrary", "arbitrary") 表示：
            # - 维度 0 (heads): 并行执行，不同 head 完全独立
            # - 维度 1 (Q blocks): 任意顺序，可能需要协调
            # - 维度 2 (KV blocks): 任意顺序，有状态依赖（在线 softmax）
            #
            # 为什么 KV 方向是 arbitrary？
            # 因为我们需要沿 KV 方向串行处理来实现在线 softmax！
            # 每个 j 依赖于 j-1 留下的 (m, l, o) 状态
            dimension_semantics=("parallel", "arbitrary", "arbitrary"),
            
            # XLA 编译器标志
            # XLA_TPU_FORCE_LP_LLO_SCHEDULER: 强制使用低延迟调度器
            # 这通常可以提高 attention 这类计算密集型操作的性能
            flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": True}
        ),
        
        out_shape=out_shapes,
        
        # interpret=True 时使用 Python 解释执行，便于调试
        # interpret=False 时编译为真正的 TPU 代码
        interpret=interpret,
    )(q, k, v)  # 传入实际输入
    
    # 返回最后一个输出（实际的 attention 输出）
    # 前三个是 scratch 缓冲区，不需要返回
    return all_out[-1]


def _make_splash_mha(
    block_sizes: _BlockSizes,
    bkv_compute_in: int,
    interpret: bool = False,
):
    """
    创建 Splash Attention 函数的工厂函数
    
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │ 为什么使用工厂函数模式？                                                          │
    │                                                                                 │
    │ 1. 封装配置：将块大小等参数绑定到返回的函数中                                      │
    │ 2. 复用编译：相同配置的调用可以复用编译结果（JAX 的 jit cache）                    │
    │ 3. 简化接口：调用者只需传入 (q, k, v)，不需要关心内部参数                          │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    Args:
        block_sizes: 块大小配置
        bkv_compute_in: 内层循环块大小
        interpret: 是否使用解释模式
    
    Returns:
        一个函数 (q, k, v) -> output
        
    Note:
        返回函数的 Query 应该预乘以 LOG2_E (1.44269504) 以使用 exp2 优化
    """
    def _splash_attention(q: jax.Array, k: jax.Array, v: jax.Array):
        return _splash_attention_forward(q, k, v, block_sizes, bkv_compute_in, interpret)
    return _splash_attention


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║                        SDPA 参考实现 (PyTorch)                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
#
# ====================================================================================
# 八、为什么需要参考实现？
# ====================================================================================
#
# 1. **备用方案**：当输入不适合 Splash Attention 时（如非常短的序列），
#    可以回退到这个简单实现
# 2. **正确性验证**：可以用来验证优化实现的正确性
# 3. **可读性**：标准实现更容易理解 attention 的数学原理
#

def sdpa_reference(query, key, value, attn_mask=None, dropout_p=0.0,
                   is_causal=False, scale=None, enable_gqa=False):
    """
    Scaled Dot-Product Attention 参考实现
    
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │ 标准 SDPA 公式                                                                  │
    │                                                                                 │
    │   Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V                        │
    │                                                                                 │
    │ 其中：                                                                          │
    │   - Q: Query，形状 (..., L, d_k)                                               │
    │   - K: Key，形状 (..., S, d_k)                                                 │
    │   - V: Value，形状 (..., S, d_v)                                               │
    │   - L: Query 序列长度                                                           │
    │   - S: Key/Value 序列长度                                                       │
    │   - d_k: Query/Key 的维度                                                       │
    │   - d_v: Value 的维度                                                           │
    │                                                                                 │
    │ 输出形状：(..., L, d_v)                                                         │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    这是一个 PyTorch 实现，用于短序列或作为参考。
    对于长序列，应该使用 tpu_splash_attention。
    
    Args:
        query: Query 张量，形状 (..., num_heads, L, d_k)
        key: Key 张量，形状 (..., num_kv_heads, S, d_k)
        value: Value 张量，形状 (..., num_kv_heads, S, d_v)
        attn_mask: 可选的注意力掩码
        dropout_p: Dropout 概率
        is_causal: 是否使用因果掩码（用于自回归模型）
        scale: 缩放因子，默认 1/sqrt(d_k)
        enable_gqa: 是否启用 Grouped Query Attention
    
    Returns:
        注意力输出，形状 (..., num_heads, L, d_v)
    """
    
    # 获取序列长度
    L, S = query.size(-2), key.size(-2)
    
    # 计算缩放因子，默认使用 1/sqrt(d_k)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    
    # 初始化 attention bias
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    
    # 处理因果掩码
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    
    # 处理自定义掩码
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    
    # 处理 GQA
    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    # 计算 attention
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    
    if dropout_p > 0:
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    
    return attn_weight @ value


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║                      TPU Splash Attention (主入口函数)                            ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
#
# ====================================================================================
# 九、分布式执行与分片策略
# ====================================================================================
#
# 在多 TPU 芯片上运行时，需要合理分配计算任务。主要策略：
#
# 1. **数据并行 (Data Parallel, DP)**:
#    - 不同 batch 在不同设备上计算
#    - 每个设备有完整的模型参数
#    - 适合 batch size 大的场景
#
# 2. **张量并行 (Tensor Parallel, TP)**:
#    - 同一个张量切分到多个设备
#    - 对于 Attention: 可以按 head 切分或按序列切分
#    - 需要设备间通信
#
# 3. **序列并行 (Sequence Parallel)**:
#    - Query 序列切分到多个设备
#    - 适合序列很长的场景
#
# 本模块根据序列长度自动选择最优策略：
# - 长 KV 序列 (>10000): Head Parallel（每个设备处理部分 head）
# - 短 KV 序列: Sequence Parallel（每个设备处理部分 Q 序列）
#

def tpu_splash_attention(query, key, value, mesh, scale=None):
    """
    TPU Splash Attention 主入口函数
    
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │ 这是外部调用的主入口，它：                                                        │
    │ 1. 预处理 Query（乘以 scale 和 LOG2_E）                                          │
    │ 2. 根据序列长度选择分片策略                                                       │
    │ 3. 使用 shard_map 在多设备上并行执行                                              │
    │ 4. 后处理输出（移除 padding）                                                    │
    └─────────────────────────────────────────────────────────────────────────────────┘
    
    Args:
        query: Query 张量，形状 (batch, num_heads, seq_len, head_dim)
        key: Key 张量，形状 (batch, num_kv_heads, seq_len, head_dim)
        value: Value 张量，形状 (batch, num_kv_heads, seq_len, head_dim)
        mesh: JAX 设备网格，定义了设备拓扑和轴名称
        scale: 可选的缩放因子，默认 1/sqrt(head_dim)
    
    Returns:
        Attention 输出，形状与 query 相同
    
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │ JAX Mesh (设备网格) 说明                                                          │
    │                                                                                 │
    │ Mesh 定义了多设备的拓扑结构，例如：                                                │
    │   mesh = Mesh(devices, axis_names=("dp", "tp"))                                │
    │                                                                                 │
    │ 这表示一个 2D 网格，有 "dp" (数据并行) 和 "tp" (张量并行) 两个轴                   │
    │                                                                                 │
    │ 例如 4 个 TPU 芯片可以组织为：                                                     │
    │   - 2×2 网格: dp=2, tp=2                                                        │
    │   - 1×4 网格: dp=1, tp=4                                                        │
    │   - 4×1 网格: dp=4, tp=1                                                        │
    └─────────────────────────────────────────────────────────────────────────────────┘
    """
    
    # =========================================================================
    # 定义单设备上的计算内核
    # =========================================================================
    
    def _attention_kernel(q, k, v):
        """
        单个分片的 attention 计算
        
        ┌─────────────────────────────────────────────────────────────────────────────┐
        │ 这个函数在每个设备上独立执行                                                   │
        │ shard_map 会自动将输入按分片规格切分后传入                                     │
        └─────────────────────────────────────────────────────────────────────────────┘
        """
        
        # 计算缩放因子
        scale_factor = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
        
        # 【关键】预乘 LOG2_E，使内核可以使用 exp2 优化
        # 原理：exp(x * scale) = exp2(x * scale * LOG2_E)
        q = q * scale_factor * LOG2_E

        def kernel_3d(q_3d, k_3d, v_3d):
            """
            处理 3D 输入 (heads, seq_len, head_dim)
            
            ┌─────────────────────────────────────────────────────────────────────────┐
            │ 输入维度说明                                                             │
            │                                                                         │
            │ 外层 _attention_kernel 处理 4D: (batch, heads, seq, dim)                │
            │ 这里的 kernel_3d 处理 3D: (heads, seq, dim)                             │
            │ 通过 vmap 在 batch 维度上并行                                            │
            │                                                                         │
            │ 这样设计的好处：                                                          │
            │ 1. 核心内核更简单，只处理单个样本                                          │
            │ 2. vmap 自动处理 batch 并行，无需手动管理                                  │
            │ 3. 便于调试和测试                                                        │
            └─────────────────────────────────────────────────────────────────────────┘
            """
            
            # Padding 到块大小的倍数
            # 这确保所有块大小相同，MXU 效率最高
            q_3d_padded, q_orig_len = _pad_to_multiple(q_3d, BQSIZE, axis=1)
            k_3d_padded, _ = _pad_to_multiple(k_3d, BKVSIZE, axis=1)
            v_3d_padded, _ = _pad_to_multiple(v_3d, BKVSIZE, axis=1)
            
            # 创建块大小配置
            # 使用 min 确保块大小不超过实际序列长度
            block_sizes = _BlockSizes(
                block_q=min(BQSIZE, q_3d_padded.shape[1]),
                block_kv=min(BKVSIZE, k_3d_padded.shape[1]),
                block_kv_compute=min(BKVCOMPUTESIZE, k_3d_padded.shape[1]),
            )
            
            # 创建并执行 Splash Attention 内核
            splash_kernel = _make_splash_mha(
                block_sizes=block_sizes, bkv_compute_in=BKVCOMPUTEINSIZE
            )
            out = splash_kernel(q_3d_padded, k_3d_padded, v_3d_padded).astype(q_3d.dtype)
            
            # 维度转换：内核输出是 (heads, dim, seq)，需要转回 (heads, seq, dim)
            out = jnp.swapaxes(out, 1, 2)
            
            # 移除 padding，恢复原始序列长度
            return out[:, :q_orig_len, :]

        # 使用 vmap 在 batch 维度上并行
        # in_axes=(0, 0, 0) 表示三个输入都在第 0 维（batch）上映射
        # out_axes=0 表示输出也在第 0 维上组合
        return jax.vmap(kernel_3d, in_axes=(0, 0, 0), out_axes=0)(q, k, v)

    # =========================================================================
    # 确定分片策略
    # =========================================================================
    #
    # 根据输入形状和设备数量，选择最优的并行策略
    #

    # 判断是否有多个 batch
    # batch > 1 时使用数据并行
    dp_mesh_key = "dp" if key.shape[0] > 1 else None
    
    # 确定剩余设备轴（用于 head 或 sequence 并行）
    remain_mesh_key = ("tp",) if key.shape[0] > 1 else ("dp", "tp")
    
    # 计算剩余轴的设备数
    remain_devices = 1
    for d in remain_mesh_key:
        remain_devices *= mesh.axis_sizes[mesh.axis_names.index(d)]

    q_seq_len = query.shape[2]
    kv_seq_len = key.shape[2]
    
    # =========================================================================
    # 选择并行策略
    # =========================================================================
    
    # 长 KV 序列（self-attention）使用 Head Parallel
    # ┌─────────────────────────────────────────────────────────────────────────────┐
    # │ 为什么长序列用 Head Parallel？                                                │
    # │                                                                             │
    # │ 1. Self-attention 中 Q, K, V 序列长度相同且都很长                             │
    # │ 2. 如果按序列切分，每个设备需要看到所有 KV（通信量大）                           │
    # │ 3. 按 head 切分，每个 head 独立计算，无需通信                                   │
    # │ 4. 阈值 10000 是经验值，可以根据硬件调整                                        │
    # └─────────────────────────────────────────────────────────────────────────────┘
    if (kv_seq_len > 10000 and
        key.shape[1] % remain_devices == 0 and
        query.shape[1] % remain_devices == 0):
        # Head Parallel: 在 head 维度上切分
        q_spec = P(dp_mesh_key, remain_mesh_key, None, None)
        kv_spec = P(dp_mesh_key, remain_mesh_key, None, None)
    else:
        # 短 KV 序列（cross-attention）使用 Sequence Parallel
        # ┌─────────────────────────────────────────────────────────────────────────┐
        # │ 为什么短 KV 用 Sequence Parallel？                                        │
        # │                                                                         │
        # │ Cross-attention 中：                                                     │
        # │ - Q 来自 decoder，可能很长                                                │
        # │ - K, V 来自 encoder，通常较短                                            │
        # │                                                                         │
        # │ Sequence Parallel 策略：                                                 │
        # │ - Q 按序列切分到多个设备                                                  │
        # │ - K, V 不切分，每个设备都有完整副本                                        │
        # │ - 每个设备计算部分 Q 对所有 K, V 的 attention                              │
        # └─────────────────────────────────────────────────────────────────────────┘
        if q_seq_len % remain_devices != 0:
            # Q 序列长度需要能被设备数整除，不足时 padding
            query, _ = _pad_to_multiple(query, remain_devices, axis=2)
        q_spec = P(dp_mesh_key, None, remain_mesh_key, None)
        kv_spec = P(dp_mesh_key, None, None, None)

    # =========================================================================
    # 应用分片并执行
    # =========================================================================
    
    # 使用 shard_map 创建分片函数
    # ┌─────────────────────────────────────────────────────────────────────────────┐
    # │ shard_map vs pmap                                                          │
    # │                                                                             │
    # │ pmap: 只支持简单的数据并行，每个设备处理一个 batch                             │
    # │                                                                             │
    # │ shard_map: 更灵活                                                           │
    # │ - 可以指定任意分片规格                                                        │
    # │ - 支持多轴并行（如同时 DP + TP）                                              │
    # │ - 自动处理通信                                                               │
    # │                                                                             │
    # │ check_rep=False: 不检查输入是否已经按规格分片                                  │
    # │ （我们用 with_sharding_constraint 手动确保分片）                               │
    # └─────────────────────────────────────────────────────────────────────────────┘
    sharded_fn = shard_map(
        _attention_kernel, mesh=mesh,
        in_specs=(q_spec, kv_spec, kv_spec), out_specs=q_spec, check_rep=False,
    )
    
    # 应用分片约束
    # ┌─────────────────────────────────────────────────────────────────────────────┐
    # │ with_sharding_constraint 的作用                                             │
    # │                                                                             │
    # │ 1. 告诉 XLA 编译器张量的分片方式                                               │
    # │ 2. 如果张量当前分片与约束不同，自动插入通信操作                                  │
    # │ 3. 确保进入 shard_map 前数据已正确分布                                         │
    # └─────────────────────────────────────────────────────────────────────────────┘
    constraint = P(dp_mesh_key, None, remain_mesh_key, None)
    query = jax.lax.with_sharding_constraint(query, constraint)
    key = jax.lax.with_sharding_constraint(key, constraint)
    value = jax.lax.with_sharding_constraint(value, constraint)
    
    # 执行分片计算
    out = sharded_fn(query, key, value)
    
    # 移除 Q 序列的 padding
    out = out[:, :, :q_seq_len, :]
    
    # 确保输出也有正确的分片
    return jax.lax.with_sharding_constraint(out, constraint)


# ╔══════════════════════════════════════════════════════════════════════════════════╗
# ║                                  总结                                             ║
# ╚══════════════════════════════════════════════════════════════════════════════════╝
#
# ====================================================================================
# 十、本模块的设计精髓
# ====================================================================================
#
# 1. **exp2 优化**：
#    - 利用 TPU VPU 的 exp2 硬件指令
#    - 通过预乘 LOG2_E 将 exp 转换为 exp2
#    - 性能提升 10-20%
#
# 2. **在线 Softmax**：
#    - 不存储完整的 N×N attention 矩阵
#    - 维护 (max, sum, output) 三个统计量
#    - 内存从 O(N²) 降到 O(1)（相对于序列长度）
#
# 3. **分块计算**：
#    - 块大小针对 TPU v6e 的 VMEM 大小优化
#    - 多层次块划分：BKVSIZE -> BKVCOMPUTESIZE -> BKVCOMPUTEINSIZE
#    - 平衡内存使用和计算效率
#
# 4. **TPU 硬件适配**：
#    - NUM_SUBLANES=8 利用 VPU 的 8 个子通道
#    - 块大小是 128 的倍数，适配 MXU
#    - dimension_semantics 控制多核并行
#
# 5. **灵活的分片策略**：
#    - 长序列用 Head Parallel
#    - 短序列用 Sequence Parallel
#    - 支持 GQA (Grouped Query Attention)
#
# 6. **Pallas 框架使用**：
#    - BlockSpec 定义数据块化
#    - PrefetchScalarGridSpec 管理预取
#    - CompilerParams 配置 TPU 编译
#
# ====================================================================================
# 参考资料
# ====================================================================================
#
# - Flash Attention 论文: https://arxiv.org/abs/2205.14135
# - Flash Attention 2 论文: https://arxiv.org/abs/2307.08691
# - JAX Pallas 文档: https://jax.readthedocs.io/en/latest/pallas/
# - TPU 架构白皮书: https://cloud.google.com/tpu/docs/system-architecture
#


