# Splash Attention Kernel 深度分析文档

## 目录
1. [概述](#概述)
2. [整体架构](#整体架构)
3. [序列分块处理机制](#序列分块处理机制)
4. [Mask 机制详解](#mask-机制详解)
5. [前向传播实现](#前向传播实现)
6. [反向传播实现](#反向传播实现)
7. [内存优化策略](#内存优化策略)
8. [API 使用示例](#api-使用示例)

---

## 概述

Splash Attention（Sparse Flash Attention）是 JAX/Pallas 为 TPU 优化的稀疏注意力机制实现。它结合了 Flash Attention 的内存效率和稀疏注意力的计算效率，特别适用于处理长序列任务。

### 核心特点
- **稀疏性支持**：通过 block-level 的稀疏性跳过不必要的计算
- **内存效率**：采用分块计算，避免材化完整的注意力矩阵
- **TPU 优化**：针对 TPU 的 VMEM、SMEM 层级进行专门优化
- **多头注意力支持**：支持 MHA（Multi-Head Attention）、MQA（Multi-Query Attention）和 GQA（Grouped Query Attention）

---

## 整体架构

```mermaid
flowchart TB
    subgraph Input["输入层"]
        Q["Query [num_heads, q_seq_len, head_dim]"]
        K["Key [num_kv_heads, kv_seq_len, head_dim]"]
        V["Value [num_kv_heads, kv_seq_len, head_dim]"]
        Mask["Mask 配置"]
    end

    subgraph MaskProcessing["Mask 预处理"]
        MP["process_mask()"]
        MI["MaskInfo"]
        MP --> MI
        MI --> DN["data_next: 下一个KV块索引"]
        MI --> MN["mask_next: 下一个mask块索引"]
        MI --> BM["block_mask: 块类型标记 (0/1/2)"]
        MI --> PMB["partial_mask_blocks: 部分mask块"]
    end

    subgraph Kernel["Splash Attention Kernel"]
        FWD["Forward Pass<br/>flash_attention_kernel"]
        BWD_DQ["Backward dQ<br/>_flash_attention_dq_kernel"]
        BWD_DKV["Backward dKV<br/>_flash_attention_dkv_kernel"]
    end

    subgraph Output["输出层"]
        O["Output [num_heads, q_seq_len, head_dim]"]
        LSE["LogSumExp [num_heads, q_seq_len]"]
        DQ["dQ 梯度"]
        DK["dK 梯度"]
        DV["dV 梯度"]
    end

    Mask --> MP
    Q --> FWD
    K --> FWD
    V --> FWD
    MI --> FWD
    FWD --> O
    FWD --> LSE

    O --> BWD_DQ
    O --> BWD_DKV
    LSE --> BWD_DQ
    LSE --> BWD_DKV
    BWD_DQ --> DQ
    BWD_DKV --> DK
    BWD_DKV --> DV

    style Input fill:#e1f5fe
    style MaskProcessing fill:#fff3e0
    style Kernel fill:#f3e5f5
    style Output fill:#e8f5e9
```

---

## 序列分块处理机制

Splash Attention 使用分块（Tiling）策略来处理长序列，关键参数定义在 [`BlockSizes`](splash_attention_kernel.py:494) 类中：

### BlockSizes 配置

```python
@dataclasses.dataclass(frozen=True, slots=True)
class BlockSizes:
    # 前向传播块大小
    block_q: int          # Q 序列块大小
    block_kv: int         # KV 序列块大小（内存）
    block_kv_compute: int # KV 计算块大小
    
    # dKV 反向传播块大小
    block_q_dkv: int
    block_kv_dkv: int
    block_kv_dkv_compute: int
    
    # dQ 反向传播块大小
    block_q_dq: int
    block_kv_dq: int
    
    # 数据布局
    q_layout: QKVLayout   # HEAD_DIM_MINOR 或 SEQ_MINOR
    k_layout: QKVLayout
    v_layout: QKVLayout
```

### 分块计算流程

```mermaid
flowchart LR
    subgraph Sequence["完整序列"]
        direction TB
        S1["Q: [heads, q_seq, dim]"]
        S2["K: [heads, kv_seq, dim]"]
        S3["V: [heads, kv_seq, dim]"]
    end

    subgraph Blocking["分块策略"]
        direction TB
        B1["Q blocks: q_seq / block_q"]
        B2["KV blocks: kv_seq / block_kv"]
        B3["Grid: (heads, q_blocks, kv_blocks)"]
    end

    subgraph Compute["计算网格"]
        direction TB
        C1["program_id(0): head_idx"]
        C2["program_id(1): q_block_idx"]
        C3["program_id(2): kv_block_idx"]
    end

    S1 --> B1
    S2 --> B2
    S3 --> B2
    B1 --> C2
    B2 --> C3
    B3 --> C1
```

### 内部计算循环

在 [`flash_attention_kernel`](splash_attention_kernel.py:702) 中，KV 块可以进一步细分为更小的计算块：

```mermaid
flowchart TB
    subgraph OuterLoop["外层循环 (Grid)"]
        H["遍历 heads"]
        I["遍历 Q blocks"]
        J["遍历 KV blocks"]
    end

    subgraph InnerLoop["内层循环 (block_kv_compute)"]
        K["划分 KV block 为 compute blocks"]
        L["每个 compute block 执行:<br/>1. QK^T 矩阵乘<br/>2. Apply Mask<br/>3. Softmax<br/>4. SV 矩阵乘"]
    end

    subgraph Accumulation["在线累积"]
        M["m_scratch: 最大值"]
        N["l_scratch: exp 求和"]
        O["o_scratch: 输出累积"]
    end

    H --> I --> J --> K --> L
    L --> M
    L --> N
    L --> O

    style OuterLoop fill:#e3f2fd
    style InnerLoop fill:#fce4ec
    style Accumulation fill:#e8f5e9
```

---

## Mask 机制详解

Splash Attention 支持多种 Mask 类型，通过 [`splash_attention_mask.py`](splash_attention_mask.py) 定义：

### Mask 类型层次结构

```mermaid
classDiagram
    class Mask {
        <<abstract>>
        +shape: tuple
        +__getitem__(idx)
        +__or__(other)
        +__and__(other)
    }

    class _ComputableMask {
        +_shape: tuple
        +q_sequence: ndarray
        +mask_function: Callable
    }

    class CausalMask {
        +offset: int
        +causal_mask_function()
    }

    class LocalMask {
        +window_size: tuple
        +offset: int
        +local_mask_function()
    }

    class ChunkedCausalMask {
        +chunk_size: int
        +chunked_causal_mask_function()
    }

    class NumpyMask {
        +array: ndarray
    }

    class FullMask {
        +_shape: tuple
    }

    class MultiHeadMask {
        +masks: Sequence~Mask~
    }

    class LogicalOr {
        +left: Mask
        +right: Mask
    }

    class LogicalAnd {
        +left: Mask
        +right: Mask
    }

    Mask <|-- _ComputableMask
    Mask <|-- NumpyMask
    Mask <|-- FullMask
    Mask <|-- MultiHeadMask
    Mask <|-- LogicalOr
    Mask <|-- LogicalAnd

    _ComputableMask <|-- CausalMask
    _ComputableMask <|-- LocalMask
    _ComputableMask <|-- ChunkedCausalMask
```

### Mask 预处理流程

[`MaskInfo`](splash_attention_mask_info.py:33) 结构包含运行时 mask 信息：

```mermaid
flowchart TB
    subgraph Input["输入 Mask"]
        M1["MultiHeadMask"]
        M2["或 jax.Array (动态)"]
    end

    subgraph Processing["预处理 process_mask()"]
        P1["分析每个 block"]
        P2["判断 block 类型"]
        P3["收集部分 mask 块"]
        P4["构建索引数组"]
    end

    subgraph BlockTypes["Block 类型分类"]
        T0["block_mask = 0<br/>全零块 (跳过)"]
        T1["block_mask = 1<br/>部分块 (需要mask)"]
        T2["block_mask = 2<br/>全一块 (无需mask)"]
    end

    subgraph MaskInfo["MaskInfo 输出"]
        MI1["data_next: int[heads, q_blocks, kv_blocks]<br/>下一个非零 KV 块索引"]
        MI2["mask_next: int[heads, q_blocks, kv_blocks]<br/>下一个 mask 块索引"]
        MI3["block_mask: int[heads, q_blocks, kv_blocks]<br/>块类型标记"]
        MI4["partial_mask_blocks: bool[N, bq, bkv]<br/>部分 mask 块集合"]
        MI5["q_sequence: int[q_seq_len]<br/>Q 序列索引 (用于可计算mask)"]
    end

    M1 --> P1
    M2 --> P1
    P1 --> P2
    P2 --> T0
    P2 --> T1
    P2 --> T2
    T1 --> P3
    P3 --> P4
    P4 --> MI1
    P4 --> MI2
    P2 --> MI3
    P3 --> MI4
    P1 --> MI5

    style BlockTypes fill:#fff9c4
    style MaskInfo fill:#e8f5e9
```

### Block Mask 值的含义

| block_mask 值 | 含义 | 处理方式 |
|--------------|------|---------|
| 0 | 全零块 | 完全跳过，不计算 |
| 1 | 部分块 | 从 partial_mask_blocks 加载实际 mask |
| 2 | 全一块 | 不应用 mask，直接计算 |

---

## Mask 类型与内存开销详细分析

Splash Attention 的核心优势之一是**绝大多数情况下不需要存储完整的 [B, H, S, L] 大小的 mask 矩阵**。

### 🔑 关键问题回答

**Q: Splash Attention 有几种 Mask 机制？**

共有 **6 种**主要的 Mask 类型，可分为两大类：

```mermaid
flowchart TB
    subgraph ComputableMasks["可计算 Mask (不需要存储完整矩阵)"]
        CM1["CausalMask<br/>因果注意力"]
        CM2["LocalMask<br/>局部窗口注意力"]
        CM3["ChunkedCausalMask<br/>分块因果注意力"]
        CM4["FullMask<br/>全注意力"]
    end

    subgraph StoredMasks["存储型 Mask (需要存储部分/全部矩阵)"]
        SM1["NumpyMask<br/>自定义numpy掩码"]
        SM2["动态 jax.Array Mask<br/>运行时动态掩码"]
    end

    style ComputableMasks fill:#c8e6c9
    style StoredMasks fill:#ffcdd2
```

**Q: 是否需要 [B, H, S, L] 这么大的矩阵？**

| Mask 类型 | 是否需要 O(seq²) 存储 | 实际内存需求 |
|-----------|---------------------|-------------|
| `CausalMask` | ❌ **否** | O(seq_len) 只存索引 |
| `LocalMask` | ❌ **否** | O(seq_len) 只存索引 |
| `ChunkedCausalMask` | ❌ **否** | O(seq_len) 只存索引 |
| `FullMask` | ❌ **否** | O(1) 只存 shape |
| `NumpyMask` | ⚠️ **部分** | O(unique_blocks × block²) |
| 动态 `jax.Array` | ⚠️ **是** | O(H × seq²) 需完整存储 |

### Mask 内存需求详解

#### 1. 可计算 Mask（零额外存储） ✅

`CausalMask`、`LocalMask`、`ChunkedCausalMask` 继承自 `_ComputableMask`：

```mermaid
flowchart LR
    subgraph Storage["存储需求"]
        S1["q_sequence: int32[q_seq_len]<br/>例: 8192 × 4 bytes = 32 KB"]
        S2["mask_function: 函数指针 ≈ 0"]
    end

    subgraph Runtime["运行时按需计算"]
        R1["kernel 内部实时计算"]
        R2["只计算当前 block 的 mask"]
        R3["无需预存储完整矩阵"]
    end

    Storage --> Runtime

    style Storage fill:#c8e6c9
    style Runtime fill:#e3f2fd
```

**CausalMask 内存公式**：
```
内存 = q_seq_len × sizeof(int32) = seq_len × 4 bytes
```

**具体示例 (8192 tokens)**：
| 方案 | 内存计算 | 内存大小 |
|-----|---------|---------|
| 传统完整矩阵 | 8192 × 8192 × 1 byte | **64 MB** (单head) |
| Splash CausalMask | 8192 × 4 bytes | **32 KB** (所有heads共享) |
| 节省比例 | - | **99.95%** |

#### 2. NumpyMask（部分块存储） ⚠️

对于自定义的 numpy mask，只存储"部分块"（既非全零也非全一）：

```mermaid
flowchart TB
    subgraph Analysis["分析完整 Mask"]
        A1["遍历每个 block"]
        A2["判断 block 类型"]
    end

    subgraph Classification["块分类"]
        C1["全零块 block_mask=0<br/>❌ 不存储"]
        C2["部分块 block_mask=1<br/>✅ 存储到 partial_mask_blocks"]
        C3["全一块 block_mask=2<br/>❌ 不存储"]
    end

    subgraph Dedup["去重存储"]
        D1["相同的部分块只存一份"]
        D2["partial_mask_blocks[N, bq, bkv]"]
    end

    A1 --> A2
    A2 --> C1
    A2 --> C2
    A2 --> C3
    C2 --> D1 --> D2

    style C1 fill:#ffcdd2
    style C2 fill:#fff9c4
    style C3 fill:#c8e6c9
```

**NumpyMask 内存公式**：
```
partial_mask_blocks = num_unique_partial_blocks × block_q × block_kv × 1 byte
MaskInfo metadata  = heads × q_blocks × kv_blocks × 3 bytes (int8)
```

**Causal Mask 作为 NumpyMask 的示例 (8192 tokens, block=128)**：
```
q_blocks = kv_blocks = 64
对角线上有 64 个部分块
但由于去重，实际只需 ~2 个唯一模式
partial_mask = 2 × 128 × 128 × 1 = 32 KB
metadata = 1 × 64 × 64 × 3 = 12 KB
总计 ≈ 44 KB (vs 64 MB)
```

#### 3. 动态 jax.Array Mask（最大存储） ❌

**这是唯一需要 O(H × seq²) 内存的情况！**

```mermaid
flowchart TB
    subgraph Input["输入"]
        I1["动态 mask: jax.Array[H, S, L]"]
    end

    subgraph Reshape["重塑为块形式"]
        R1["[H, q_blocks, kv_blocks, bq, bkv]"]
    end

    subgraph Storage["存储"]
        S1["partial_mask_blocks 存储完整分块矩阵"]
        S2["内存 = H × S × L bytes"]
    end

    I1 --> R1 --> S1 --> S2

    style Input fill:#ffcdd2
    style Storage fill:#ffcdd2
```

**动态 Mask 内存**：
```
内存 = heads × q_seq × kv_seq × sizeof(bool)
     = heads × seq² bytes  (与传统方法相同)
```

### 4. Segment IDs（独立机制）

Segment IDs 用于 packed sequences，是独立的机制：

```
segment_ids.q  = int32[q_seq_len]   → q_seq × 4 bytes
segment_ids.kv = int32[kv_seq_len]  → kv_seq × 4 bytes
总计 = (q_seq + kv_seq) × 4 bytes ≈ O(seq_len)
```

### 内存对比总结图

以 **16 heads, 8192 tokens, block_size=128** 为例：

```mermaid
graph TB
    subgraph Traditional["传统 Attention"]
        T1["完整 Mask 矩阵"]
        T2["16 × 8192 × 8192 × 1 byte"]
        T3["= 1 GB"]
        T1 --> T2 --> T3
    end

    subgraph Splash["Splash Attention"]
        S1["CausalMask"]
        S2["8192 × 4 + metadata"]
        S3["≈ 224 KB"]
        S4["节省 99.98%"]
        
        N1["NumpyMask (稀疏)"]
        N2["unique_blocks × 128²"]
        N3["≈ 256 KB"]
        N4["节省 99.97%"]
        
        D1["动态 jax.Array"]
        D2["16 × 8192 × 8192"]
        D3["≈ 1.2 GB"]
        D4["无节省"]
        
        S1 --> S2 --> S3 --> S4
        N1 --> N2 --> N3 --> N4
        D1 --> D2 --> D3 --> D4
    end

    style T3 fill:#ffcdd2
    style S3 fill:#c8e6c9
    style N3 fill:#fff9c4
    style D3 fill:#ffcdd2
```

### 🎯 最佳实践建议

| 场景 | 推荐 Mask 类型 | 内存开销 |
|-----|---------------|---------|
| 标准 Decoder | `CausalMask` | O(seq) ✅ |
| 局部注意力 | `LocalMask` | O(seq) ✅ |
| Llama4 风格 | `ChunkedCausalMask` | O(seq) ✅ |
| 全注意力 Encoder | `FullMask` | O(1) ✅ |
| **有 Padding 的变长序列** | **Segment IDs** | **O(seq) ✅** |
| 复杂自定义静态 | `NumpyMask` | O(blocks) ⚠️ |
| 运行时动态 | 避免使用 | O(H×seq²) ❌ |

---

## 💡 重要场景：Padding 序列的处理（36k 长序列示例）

### 场景描述

对于**变长序列 padding 到固定长度**（如 36k）的情况，**推荐使用 Segment IDs 而非完整 Mask 矩阵**。

```mermaid
flowchart TB
    subgraph Input["输入序列 (padding到36k)"]
        I1["实际 tokens: 0 ~ actual_len-1"]
        I2["padding tokens: actual_len ~ 36k-1"]
    end

    subgraph SegmentIds["Segment IDs 方案 ✅ 推荐"]
        S1["segment_ids.q = [0,0,...,0, -1,-1,...,-1]"]
        S2["segment_ids.kv = [0,0,...,0, -1,-1,...,-1]"]
        S3["只有 segment_id=0 的 token 互相可见"]
        S4["内存: 2 × 36k × 4 bytes = 288 KB"]
    end

    subgraph FullMask["完整 Mask 矩阵方案 ❌ 不推荐"]
        F1["mask = jax.Array[H, 36k, 36k]"]
        F2["内存: 16 × 36k × 36k = 20.7 GB"]
    end

    I1 --> S1
    I2 --> S2
    S1 --> S3 --> S4

    style SegmentIds fill:#c8e6c9
    style FullMask fill:#ffcdd2
    style S4 fill:#c8e6c9
    style F2 fill:#ffcdd2
```

### 内存对比 (36k 序列, 16 heads)

| 方案 | 内存计算 | 内存大小 | 是否推荐 |
|-----|---------|---------|---------|
| 完整 Mask 矩阵 | 16 × 36k × 36k × 1 byte | **20.7 GB** | ❌ 不可行 |
| 动态 jax.Array Mask | 16 × 36k × 36k × 1 byte | **20.7 GB** | ❌ 不可行 |
| **Segment IDs** | 2 × 36k × 4 bytes | **288 KB** | ✅ **推荐** |
| Segment IDs + CausalMask | 3 × 36k × 4 bytes | **432 KB** | ✅ **推荐** |

### 代码示例

```python
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_kernel as splash,
    splash_attention_mask as mask_lib,
)

# 假设：实际序列长度 actual_len，padding 到 36k
actual_len = 20000
padded_len = 36 * 1024  # 36k

# ===============================================
# 方案1：全注意力 + Padding Mask（Encoder 场景）
# ===============================================
# 创建 segment_ids：实际 token 为 0，padding 为 -1
segment_ids = splash.SegmentIds(
    q=jnp.where(jnp.arange(padded_len) < actual_len, 0, -1),
    kv=jnp.where(jnp.arange(padded_len) < actual_len, 0, -1),
)

# 使用 FullMask（只存储 shape，不存储任何 mask 数据）
mask = mask_lib.FullMask(shape=(padded_len, padded_len))
multi_head_mask = mask_lib.MultiHeadMask([mask] * num_heads)

kernel = splash.make_splash_mha_single_device(mask=multi_head_mask, ...)
output = kernel(q, k, v, segment_ids=segment_ids)

# ===============================================
# 方案2：Causal + Padding Mask（Decoder 场景）
# ===============================================
# CausalMask 确保只看前面的 token（不存储完整矩阵）
# Segment IDs 确保不 attend 到 padding
causal_mask = mask_lib.CausalMask(shape=(padded_len, padded_len))
multi_head_mask = mask_lib.MultiHeadMask([causal_mask] * num_heads)

kernel = splash.make_splash_mha_single_device(mask=multi_head_mask, ...)
output = kernel(q, k, v, segment_ids=segment_ids)
```

### Segment IDs 工作原理

```mermaid
flowchart LR
    subgraph Logic["Segment IDs 逻辑"]
        direction TB
        L1["对每对 (q_pos, kv_pos)"]
        L2["检查 segment_ids.q[q_pos] == segment_ids.kv[kv_pos]"]
        L3{"相等?"}
        L4["允许 attend"]
        L5["mask 掉 (设为 -inf)"]
        
        L1 --> L2 --> L3
        L3 -->|Yes| L4
        L3 -->|No| L5
    end

    subgraph Example["示例: actual_len=5, padded_len=8"]
        direction TB
        E1["segment_ids = [0,0,0,0,0,-1,-1,-1]"]
        E2["位置 0-4 (segment=0) 互相可见"]
        E3["位置 5-7 (segment=-1) 被完全屏蔽"]
    end

    style L4 fill:#c8e6c9
    style L5 fill:#ffcdd2
```

### 与其他 Mask 组合

Segment IDs 会与其他 Mask **做 AND 组合**：

```python
# 最终 mask = CausalMask AND SegmentIdsMask
#
# 例如位置 (3, 5):
#   - CausalMask: 3 >= 5? → False (不可见)
#   - SegmentIds: segment[3]=0, segment[5]=-1 → False (不可见)
#   - 最终: False
#
# 例如位置 (4, 2):
#   - CausalMask: 4 >= 2? → True (可见)
#   - SegmentIds: segment[4]=0, segment[2]=0 → True (可见)
#   - 最终: True
```

### ⚠️ 注意事项

1. **Segment IDs 必须确保每行至少有一个有效 token**
   - 否则 softmax 分母为 0，导致 NaN
   - 纯 padding 行需特殊处理或确保不会被查询

2. **Segment IDs 值的选择**
   - 实际 token 使用 **相同的非负整数**（如 0）
   - padding 使用 **不同的值**（如 -1）
   - 不同的独立序列（batch packing）使用不同的整数

3. **批处理多个序列（Packing）**
   ```python
   # 例如 3 个序列 pack 到一起：
   # seq1: tokens 0-99, seq2: tokens 100-199, seq3: tokens 200-249, padding: 250-255
   segment_ids = splash.SegmentIds(
       q=jnp.array([0]*100 + [1]*100 + [2]*50 + [-1]*6),
       kv=jnp.array([0]*100 + [1]*100 + [2]*50 + [-1]*6),
   )
   # 这样 seq1, seq2, seq3 互相不可见
   ```

---

## 🤔 API 设计讨论：为什么 Padding 处理不够简洁？

### 问题：Padding 是最常见的场景，但 API 不够直观

```mermaid
flowchart LR
    subgraph PyTorch["PyTorch / HuggingFace ✅ 直观"]
        P1["padding_mask = [1,1,1,1,0,0,0,0]"]
        P2["output = model(x, attention_mask=padding_mask)"]
        P1 --> P2
    end

    subgraph Splash["Splash Attention ❌ 繁琐"]
        S1["mask = FullMask(shape=(36k, 36k))"]
        S2["segment_ids = SegmentIds(...)"]
        S3["output = kernel(q, k, v, segment_ids)"]
        S1 --> S2 --> S3
    end

    style PyTorch fill:#c8e6c9
    style Splash fill:#fff9c4
```

### 为什么必须写 FullMask / CausalMask？

| 问题 | 解释 |
|------|------|
| **API 设计要求** | `make_splash_mha()` 需要 mask 参数来生成 MaskInfo |
| **语义区分** | FullMask = 双向注意力（Encoder），CausalMask = 单向注意力（Decoder） |
| **设计初衷不同** | Splash Attention 主要为**稀疏注意力模式**设计，不是为 padding |

### Segment IDs 的本意

Segment IDs 原本是为 **sequence packing**（多序列拼接）设计的，不是专门为 padding：

```python
# Packing 场景（原始设计目标）
# 多个短序列拼接成一个长序列，避免 padding 浪费
segment_ids = [0,0,0, 1,1,1,1, 2,2]  # 3个序列
# seq1(3 tokens) + seq2(4 tokens) + seq3(2 tokens)

# Padding 场景（副产品用法）
segment_ids = [0,0,0,0,0, -1,-1,-1]  # 1个序列 + padding
```

### 理想的 API（如果重新设计）

```python
# 理想情况 - 直接传 1D padding mask
padding_mask = jnp.array([1,1,1,1,1, 0,0,0])  # 1=valid, 0=padding
output = kernel(q, k, v, padding_mask=padding_mask)

# 或更简单
output = kernel(q, k, v, valid_length=5)

# 或自动推断
output = kernel(q, k, v)  # 自动从 q 的形状推断 mask shape
```

### 实际建议：封装一个便捷函数

既然 API 已经是这样了，可以自己封装简化使用：

```python
def make_padded_attention_kernel(
    padded_len: int,
    actual_len: int,
    num_heads: int,
    causal: bool = False,
    **kwargs
):
    """便捷的 padding-aware attention kernel 工厂函数
    
    Args:
        padded_len: padding 后的序列长度
        actual_len: 实际有效序列长度
        num_heads: 注意力头数
        causal: 是否使用因果注意力
        **kwargs: 传递给 make_splash_mha_single_device 的其他参数
    
    Returns:
        一个简化的 attention 函数，只需传入 q, k, v
    """
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel as splash,
        splash_attention_mask as mask_lib,
    )
    
    # 选择 mask 类型
    if causal:
        mask = mask_lib.CausalMask(shape=(padded_len, padded_len))
    else:
        mask = mask_lib.FullMask(shape=(padded_len, padded_len))
    
    multi_head_mask = mask_lib.MultiHeadMask([mask] * num_heads)
    kernel = splash.make_splash_mha_single_device(mask=multi_head_mask, **kwargs)
    
    # 创建 segment_ids
    segment_ids = splash.SegmentIds(
        q=jnp.where(jnp.arange(padded_len) < actual_len, 0, -1),
        kv=jnp.where(jnp.arange(padded_len) < actual_len, 0, -1),
    )
    
    # 返回一个简化的调用接口
    def call(q, k, v):
        return kernel(q, k, v, segment_ids=segment_ids)
    
    return call


# 使用示例 - 简洁多了！
attention = make_padded_attention_kernel(
    padded_len=36*1024,
    actual_len=20000,
    num_heads=16,
    causal=True
)
output = attention(q, k, v)
```

### 总结

| 观点 | 说明 |
|------|------|
| **用户体验** | Padding 是最常见场景，但 API 确实不够直观 |
| **设计权衡** | Splash Attention 优先考虑稀疏注意力灵活性，牺牲了易用性 |
| **实际解决** | 自行封装便捷函数，或向 JAX 团队提 feature request |
| **正面看法** | 一旦理解 FullMask + Segment IDs 的组合，使用也不算太复杂 |

---

### 稀疏性优化：Grid Shrinking

```mermaid
flowchart LR
    subgraph Before["原始 Grid"]
        direction TB
        B1["许多全零块<br/>block_mask=0"]
        B2["稀疏的非零块<br/>block_mask>0"]
    end

    subgraph Shrink["Grid Shrinking"]
        S1["_shrink_mask_info()"]
        S2["压缩 KV 维度"]
    end

    subgraph After["压缩后 Grid"]
        direction TB
        A1["只保留非零块"]
        A2["data_next 指向实际数据"]
    end

    B1 --> S1
    B2 --> S1
    S1 --> S2
    S2 --> A1
    S2 --> A2

    style Before fill:#ffcdd2
    style Shrink fill:#fff9c4
    style After fill:#c8e6c9
```

---

## 前向传播实现

### 核心算法流程

[`flash_attention_kernel`](splash_attention_kernel.py:702) 实现了在线 softmax 算法：

```mermaid
flowchart TB
    subgraph Init["初始化 (j==0)"]
        I1["o_scratch = 0"]
        I2["m_scratch = mask_value 或 sinks"]
        I3["l_scratch = 0 或 1 (有sinks时)"]
    end

    subgraph Check["检查当前块"]
        C1["_next_nonzero()"]
        C2{"should_run?"}
    end

    subgraph Compute["计算循环 body()"]
        direction TB
        L1["加载 Q block"]
        L2["加载 K block (slice_k)"]
        L3["QK = Q @ K^T"]
        L4["应用 Mask"]
        L5["计算 m_curr = max(QK)"]
        L6["m_next = max(m_prev, m_curr)"]
        L7["s_curr = exp(QK - m_next)"]
        L8["l_curr = sum(s_curr)"]
        L9["alpha = exp(m_prev - m_next)"]
        L10["l_next = l_curr + alpha * l_prev"]
        L11["加载 V block"]
        L12["o_curr = s_curr @ V"]
        L13["o_scratch = alpha * o_scratch + o_curr"]
    end

    subgraph Final["最终输出 (j==grid_width-1)"]
        F1["o = o_scratch / l_scratch"]
        F2["logsumexp = log(l) + m"]
    end

    Init --> Check
    Check --> C2
    C2 -->|Yes| Compute
    C2 -->|No| Final
    Compute --> Final

    style Init fill:#e3f2fd
    style Compute fill:#fff3e0
    style Final fill:#e8f5e9
```

### Mask 应用逻辑

[`_apply_mask_and_soft_cap`](splash_attention_kernel.py:603) 函数处理多种 mask 组合：

```mermaid
flowchart TB
    subgraph Input["输入"]
        QK["QK 矩阵"]
        MV["mask_value"]
        SNM["should_not_mask"]
    end

    subgraph MaskSources["Mask 来源 (可组合)"]
        MS1["mask_ref: 预计算的 partial mask"]
        MS2["mask_function: 可计算 mask<br/>(CausalMask, LocalMask等)"]
        MS3["segment_ids: 分段注意力 mask"]
    end

    subgraph Combine["组合 Masks"]
        C1["masks = []"]
        C2["mask_ref 存在? → 添加"]
        C3["mask_function 存在? → 计算并添加"]
        C4["segment_ids 存在? → 添加"]
        C5["final_mask = reduce(AND, masks)"]
    end

    subgraph Apply["应用"]
        A1{"attn_logits_soft_cap?"}
        A2["logits = tanh(QK/cap) * cap"]
        A3["QK = where(mask, QK, mask_value)"]
    end

    QK --> Combine
    MV --> Apply
    SNM --> Combine

    MS1 --> C2
    MS2 --> C3
    MS3 --> C4

    C2 --> C5
    C3 --> C5
    C4 --> C5

    C5 --> A1
    A1 -->|Yes| A2
    A1 -->|No| A3
    A2 --> A3

    style MaskSources fill:#fff9c4
    style Combine fill:#e1f5fe
    style Apply fill:#f3e5f5
```

---

## 反向传播实现

### 反向传播策略

Splash Attention 支持两种反向传播策略：

```mermaid
flowchart TB
    subgraph Strategy["反向传播策略"]
        S1["分离式 (默认)"]
        S2["融合式 (use_fused_bwd_kernel=True)"]
    end

    subgraph Separate["分离式反向传播"]
        SP1["_splash_attention_bwd_dkv()<br/>计算 dK, dV"]
        SP2["_splash_attention_bwd_dq()<br/>计算 dQ"]
    end

    subgraph Fused["融合式反向传播"]
        F1["_splash_attention_bwd_dkv()<br/>同时计算 dQ, dK, dV"]
    end

    S1 --> SP1
    S1 --> SP2
    S2 --> F1

    style Separate fill:#e3f2fd
    style Fused fill:#fff3e0
```

### dKV Kernel 流程

[`_flash_attention_dkv_kernel`](splash_attention_kernel.py:1673) 的计算流程：

```mermaid
flowchart TB
    subgraph Init["初始化"]
        I1["dk_scratch = 0"]
        I2["dv_scratch = 0"]
        I3["dq_scratch = 0 (融合模式)"]
    end

    subgraph Loop["计算循环"]
        L1["K, V blocks"]
        L2["QK = K @ Q^T"]
        L3["应用 Mask"]
        L4["P = exp(QK - logsumexp)"]
        L5["dV = P @ dO"]
        L6["dP = V @ dO^T"]
        L7["dS = (dP - di) * P"]
        L8["dK = dS @ Q"]
        L9["dQ = dS^T @ K (融合模式)"]
    end

    subgraph Accumulate["累积梯度"]
        A1["dk_scratch += dK"]
        A2["dv_scratch += dV"]
        A3["dq_scratch += dQ"]
    end

    subgraph Output["输出"]
        O1["dk_ref = dk_scratch"]
        O2["dv_ref = dv_scratch"]
    end

    Init --> Loop
    Loop --> L1
    L1 --> L2 --> L3 --> L4
    L4 --> L5 --> A2
    L4 --> L6 --> L7
    L7 --> L8 --> A1
    L7 --> L9 --> A3
    A1 --> O1
    A2 --> O2

    style Loop fill:#fff3e0
    style Accumulate fill:#e8f5e9
```

### dQ Kernel 流程

[`_flash_attention_dq_kernel`](splash_attention_kernel.py:1312) 的计算流程：

```mermaid
flowchart TB
    subgraph Init["初始化"]
        I1["dq_scratch = 0"]
    end

    subgraph Loop["计算循环"]
        L1["加载 Q, K, V blocks"]
        L2["QK = Q @ K^T"]
        L3["应用 Mask"]
        L4["P = exp(QK - logsumexp)"]
        L5["dP = dO @ V^T"]
        L6["dS = (dP - di) * P"]
        L7["dQ = dS @ K"]
    end

    subgraph Output["输出"]
        O1["dq_ref = dq_scratch"]
    end

    Init --> Loop
    Loop --> L1 --> L2 --> L3 --> L4
    L4 --> L5 --> L6 --> L7
    L7 --> O1

    style Loop fill:#e3f2fd
```

---

## 内存优化策略

### TPU 内存层级利用

```mermaid
flowchart TB
    subgraph HBM["HBM (High Bandwidth Memory)"]
        H1["Q, K, V 完整张量"]
        H2["输出 O"]
        H3["梯度 dQ, dK, dV"]
    end

    subgraph VMEM["VMEM (Vector Memory)"]
        V1["当前计算块"]
        V2["累积器 scratch buffers"]
    end

    subgraph SMEM["SMEM (Scalar Memory)"]
        S1["data_next"]
        S2["mask_next"]
        S3["block_mask"]
        S4["segment_ids"]
    end

    H1 -->|BlockSpec| V1
    V1 -->|累积| V2
    V2 -->|写回| H2

    S1 -->|控制| V1
    S2 -->|索引| V1
    S3 -->|跳过判断| V1

    style HBM fill:#ffcdd2
    style VMEM fill:#fff9c4
    style SMEM fill:#c8e6c9
```

### 数据类型优化

MaskInfo 中的数组会自动降级到最小所需类型：

```python
def _downcast_to_small_type(array: np.ndarray) -> np.ndarray:
    max_value = np.max(array)
    if max_value <= np.iinfo(np.int8).max:
        return array.astype(np.int8)
    elif max_value <= np.iinfo(np.int16).max:
        return array.astype(np.int16)
    else:
        return array.astype(np.int32)
```

---

## API 使用示例

### 基本使用

```python
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_kernel as splash,
    splash_attention_mask as mask_lib,
)

# 创建 Causal Mask
mask = mask_lib.CausalMask(shape=(seq_len, seq_len))
multi_head_mask = mask_lib.MultiHeadMask([mask] * num_heads)

# 配置块大小
block_sizes = splash.BlockSizes(
    block_q=128,
    block_kv=128,
    block_kv_compute=128,
    block_q_dkv=128,
    block_kv_dkv=128,
    block_kv_dkv_compute=128,
    block_q_dq=128,
    block_kv_dq=128,
)

# 创建 kernel
kernel = splash.make_splash_mha_single_device(
    mask=multi_head_mask,
    block_sizes=block_sizes,
)

# 执行注意力计算
output = kernel(q, k, v)
```

### 使用 Local Attention

```python
# Local attention 只关注前后 window_size 个 token
local_mask = mask_lib.LocalMask(
    shape=(seq_len, seq_len),
    window_size=(256, 256),  # (左侧窗口, 右侧窗口)
    offset=0,
)

# 组合 causal 和 local
combined_mask = causal_mask & local_mask
```

### 使用 Segment IDs

```python
# 用于处理 packed sequences
segment_ids = splash.SegmentIds(
    q=jnp.array([0, 0, 0, 1, 1, 1, 2, 2]),   # Q 序列的段 ID
    kv=jnp.array([0, 0, 0, 1, 1, 1, 2, 2]),  # KV 序列的段 ID
)

output = kernel(q, k, v, segment_ids=segment_ids)
```

### 分布式使用

```python
# 多设备分片
kernel = splash.make_splash_mha(
    mask=multi_head_mask,
    block_sizes=block_sizes,
    head_shards=num_devices_per_head_dim,
    q_seq_shards=num_devices_per_seq_dim,
)

# 获取分片规范
sharding_spec = kernel.manual_sharding_spec(named_sharding)
```

---

## 关键常量

| 常量 | 值 | 说明 |
|-----|-----|------|
| `NUM_LANES` | 128 | TPU 向量宽度 |
| `NUM_SUBLANES` | 8 | TPU 子通道数 |
| `DEFAULT_MASK_VALUE` | -0.7 * float32_max | 默认 mask 值 |

---

## 总结

Splash Attention 通过以下机制实现高效的长序列注意力计算：

1. **Block-level 稀疏性**：通过 `block_mask` 跳过全零块
2. **在线 Softmax**：避免材化完整的注意力矩阵
3. **可计算 Mask**：使用 `mask_function` 而非存储完整 mask
4. **Grid Shrinking**：压缩稀疏的计算网格
5. **数据类型优化**：自动降级 SMEM 数据类型
6. **分布式支持**：支持 head 和 sequence 维度的分片

这使得 Splash Attention 成为 TPU 上处理长序列任务的首选注意力实现。