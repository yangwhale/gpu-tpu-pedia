# Wan 2.1 TPU 推理性能优化指南

本文档介绍 Wan 2.1 视频生成模型在 TPU 上的关键性能优化技术。

## 目录

1. [Custom Attention 优化](#custom-attention-优化)
2. [Mesh 设备布局优化](#mesh-设备布局优化)
3. [性能基准测试](#性能基准测试)

---

## Custom Attention 优化

### 核心优化：exp2 替代 exp

TPU 的 VPU（Vector Processing Unit）有专门的 `exp2` 硬件指令，而标准的 `exp` 函数需要额外的转换步骤。

```mermaid
flowchart LR
    subgraph Standard["标准 exp 实现"]
        direction TB
        A1[exp x] --> A2["x × log₂e"]
        A2 --> A3["exp2 result"]
        A3 --> A4["2 次运算"]
    end
    
    subgraph Optimized["Custom exp2 实现"]
        direction TB
        B1["预乘 log₂e"] --> B2["exp2 result"]
        B2 --> B3["1 次运算"]
    end
    
    Standard -.->|"优化"| Optimized
```

### 数学原理

标准 softmax 中的 exp 运算：
```
exp(x) = 2^(x × log₂(e)) = exp2(x × 1.44269504)
```

Custom Attention 将 `log₂(e)` 的乘法移到 kernel 外部，在 query 预处理时一次性完成：

```mermaid
sequenceDiagram
    participant Pre as 预处理
    participant Kernel as Attention Kernel
    
    Note over Pre: 标准实现
    Pre->>Kernel: q × scale_factor
    Kernel->>Kernel: exp(qk - m) 内部转换为 exp2
    
    Note over Pre: Custom 实现  
    Pre->>Kernel: q × scale_factor × LOG2_E
    Kernel->>Kernel: exp2(qk - m) 直接调用
```

### 代码对比

#### 标准 Splash Attention

```python
# _tpu_splash_attention() 中
def _attention_on_slices(q, k, v):
    scale_factor = 1.0 / math.sqrt(q.shape[-1])
    q = q * scale_factor  # 仅缩放
    
    # kernel 内部
    s_curr = jnp.exp(qk - m_next[0:1])    # 内部转换为 exp2
    alpha = jnp.exp(m_prev - m_next)       # 内部转换为 exp2
```

#### Custom Attention (exp2 优化)

```python
# _tpu_custom_attention() 中
def _attention_on_slices(q, k, v):
    scale_factor = 1.0 / math.sqrt(q.shape[-1])
    _LOG2_E = 1.44269504
    q = q * scale_factor * _LOG2_E  # 预乘 log₂(e)
    
    # kernel 内部 (custom_splash_attention.py 第 566-572 行)
    s_curr = exp2(qk - m_next[0:1])        # 直接硬件指令
    alpha = jnp.exp2(m_prev - m_next)      # 直接硬件指令
```

### 其他优化

#### 1. 细粒度块处理 (bkv_compute_in)

```python
# custom_splash_attention.py 第 556-581 行
step = bkv_compute_in  # 默认 256
for i in range(0, qk.shape[0], step):
    m_curr = qk[i:i+step].max(axis=0)[None, :]
    s_curr = exp2(qk[i:i+step] - m_next[0:1])
    # 增量累加 softmax 和 output
```

#### 2. 编译器调度优化

```python
compiler_params=pltpu.CompilerParams(
    dimension_semantics=("parallel", "arbitrary", "arbitrary"),
    flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": True}  # 低延迟调度
)
```

---

## Mesh 设备布局优化

### Mesh 维度顺序

JAX 的 `create_device_mesh` 根据维度顺序决定设备在物理 TPU 上的布局。

```mermaid
graph TB
    subgraph Layout1["布局 1: (dp, sp, tp) = (2, 1, 4)"]
        direction LR
        D1[Device 0-3] -->|"tp 轴"| DP1["DP Group 0"]
        D2[Device 4-7] -->|"tp 轴"| DP2["DP Group 1"]
        DP1 -.->|"dp 轴"| DP2
    end
    
    subgraph Layout2["布局 2: (tp, dp, sp) = (4, 2, 1)"]
        direction LR
        T1[Device 0,4] -->|"dp 轴"| TP1["TP Group 0"]
        T2[Device 1,5] -->|"dp 轴"| TP2["TP Group 1"]
        T3[Device 2,6] -->|"dp 轴"| TP3["TP Group 2"]
        T4[Device 3,7] -->|"dp 轴"| TP4["TP Group 3"]
    end
```

### 代码对比

#### 推荐配置 (dp, sp, tp)

```python
# 与 wan_tx_splash_attn.py 一致
mesh_devices = mesh_utils.create_device_mesh(
    (dp_dim, sp_dim, tp_dim),  # (2, 1, 4)
    allow_split_physical_axes=True
)
mesh = Mesh(mesh_devices, ('dp', 'sp', 'tp'))
```

#### 替代配置 (tp, dp, sp)

```python
mesh_devices = mesh_utils.create_device_mesh(
    (tp_dim, dp_dim, sp_dim),  # (4, 2, 1)
    allow_split_physical_axes=True
)
mesh = Mesh(mesh_devices, ('tp', 'dp', 'sp'))
```

### Attention Partition Spec

根据 attention 类型选择不同的分片策略：

```mermaid
flowchart TD
    A[Attention 计算] --> B{KV 序列长度?}
    B -->|"> 10000<br>Self Attention"| C["q: P('dp', 'tp', 'sp', None)<br>kv: P('dp', 'tp', None, None)"]
    B -->|"<= 10000<br>Cross Attention"| D["q: P('dp', None, ('tp', 'sp'), None)<br>kv: P('dp', None, None, None)"]
    
    C --> E["Output: P('dp', None, ('tp', 'sp'), None)"]
    D --> E
```

```python
# Self Attention (长 KV 序列)
if key.shape[2] > 10000:
    q_partition_spec = P('dp', 'tp', 'sp', None)
    kv_partition_spec = P('dp', 'tp', None, None)
else:
    # Cross Attention (短 KV 序列)
    q_partition_spec = P('dp', None, ('tp', 'sp'), None)
    kv_partition_spec = P('dp', None, None, None)
```

---

## 性能基准测试

### 测试环境

- **硬件**: TPU v6e-8 (8 chips)
- **模型**: Wan 2.1 14B
- **分辨率**: 1280×720, 81 帧
- **推理步数**: 3 steps

### 结果对比

```mermaid
gantt
    title 性能对比 (Benchmark 时间, 越短越好)
    dateFormat X
    axisFormat %s
    
    section 标准 Attention
    USE_CUSTOM_ATTENTION=False, (tp,dp,sp) : 0, 28
    
    section Custom Attention
    USE_CUSTOM_ATTENTION=True, (tp,dp,sp)  : 0, 26
    USE_CUSTOM_ATTENTION=True, (dp,sp,tp)  : 0, 25
```

| 配置 | Mesh 顺序 | Benchmark 时间 | Step 时间 |
|-----|----------|---------------|----------|
| 标准 Attention | (tp, dp, sp) | 27.57s | ~6.26s |
| Custom Attention | (tp, dp, sp) | 25.80s | ~5.74s |
| **Custom Attention** | **(dp, sp, tp)** | **24.80s** | **~5.35s** |

### 关键发现

1. **Custom Attention (exp2)**: 提升 ~10% (27.5s → 24.8s)
2. **Mesh 布局 (dp, sp, tp)**: 额外提升 ~4% (25.8s → 24.8s)

### 推荐配置

```python
# 最优配置
USE_CUSTOM_ATTENTION = True
BQSIZE = 3328
BKVSIZE = 2816
BKVCOMPUTESIZE = 256
BKVCOMPUTEINSIZE = 256

# Mesh: (dp, sp, tp) 顺序
mesh_devices = mesh_utils.create_device_mesh(
    (dp_dim, sp_dim, tp_dim), allow_split_physical_axes=True
)
mesh = Mesh(mesh_devices, ('dp', 'sp', 'tp'))
```

---

## 待补充内容

- [ ] 完整迁移步骤
- [ ] VAE 分片配置
- [ ] Text Encoder 配置
- [ ] Transformer 分片策略详解
- [ ] 多 host 配置