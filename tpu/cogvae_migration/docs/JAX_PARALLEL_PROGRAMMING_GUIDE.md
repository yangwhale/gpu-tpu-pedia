# JAX 并行编程详解：shard_map、vmap、pmap 完全指南

> **前置知识**：如果你是 JAX 初学者，建议先阅读 [PyTorch 到 JAX 入门教程](../../torch_to_jax_jumpstart/)，从基础开始学习 JAX 和 HuggingFace 模型的使用。

## 目录

- [1. shard_map 工作原理](#1-shard_map-工作原理)
- [2. shard_map vs 传统方法](#2-shard_map-vs-传统方法)
- [3. 跨分片依赖问题](#3-跨分片依赖问题)
- [4. JAX Map 函数家族](#4-jax-map-函数家族)
- [5. vmap vs pmap 详解](#5-vmap-vs-pmap-详解)
- [6. Mesh 的重要性](#6-mesh-的重要性)
- [7. 实践建议](#7-实践建议)

---

## 1. shard_map 工作原理

### 1.1 核心概念

`shard_map` 是 JAX 中用于**显式控制数据在多设备上的分片和计算**的高级工具。

```python
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh

sharded_decode = shard_map(
    f=decode_on_slice,                      # 在单个切片上执行的函数
    mesh=mesh,                              # 设备网格（如 8个TPU）
    in_specs=P(None, None, 'tp', None, None),   # 输入分片规格
    out_specs=P(None, None, 'tp', None, None)   # 输出分片规格
)
```

### 1.2 工作流程

```
完整输入数据
    ↓ (自动分片)
┌─────────┬─────────┬─────────┐
│Device 0 │Device 1 │Device N │
│切片 0   │切片 1   │切片 N   │
└─────────┴─────────┴─────────┘
    ↓ (并行执行 decode_on_slice)
┌─────────┬─────────┬─────────┐
│输出切片0│输出切片1│输出切片N│
└─────────┴─────────┴─────────┘
    ↓ (自动合并)
完整输出数据
```

### 1.3 关键参数解析

**`in_specs` 和 `out_specs`：**

```python
P(None, None, 'tp', None, None)
```

对应张量形状 `(batch, channels, height, width, depth)`：
- 第3个维度（`height`）标记为 `'tp'`：在 mesh 的 `'tp'` 轴上分片
- 其他维度标记为 `None`：在所有设备上复制（不分片）

**示例：**
```python
# 输入: (1, 16, 1024, 64, 64)
# mesh 有 8 个设备在 'tp' 轴上
# 每个设备得到: (1, 16, 128, 64, 64)  # height从1024分成8份
```

---

## 2. shard_map vs 传统方法

### 2.1 只用 device_put（❌ 会导致问题）

```python
# 数据分片
sharding = NamedSharding(mesh, P(None, None, 'tp', None, None))
latents_jax = jax.device_put(latents_np, sharding)

# 直接调用模型（问题所在！）
with env:
    latents_torch = env.j2t_iso(latents_jax)  
    output = vae.decode(latents_torch)  # ⚠️ 触发 all-gather！
```

**发生的事情：**

1. ✅ 数据确实被分片了
2. ❌ 但**计算没有被分片**
3. ❌ JAX 会自动执行 **all-gather** 将数据收集到一个设备
4. ❌ 单个设备处理全部数据 → **OOM！**

**为什么会 all-gather？**

因为 `vae.decode()` 函数不知道如何处理分片数据，它期望完整的输入张量。JAX 为了满足这个要求，会自动收集数据。

### 2.2 使用 shard_map（✅ 正确）

```python
def decode_on_slice(latents_slice):  # 只处理一个切片
    with env:
        latents_torchax_slice = env.j2t_iso(latents_slice)
        decoded_output = vae.decode(latents_torchax_slice)
        return decoded_output.sample.jax()

sharded_decode = shard_map(
    f=decode_on_slice,
    mesh=mesh,
    in_specs=P(None, None, 'tp', None, None),
    out_specs=P(None, None, 'tp', None, None)
)

output = sharded_decode(latents_jax)
# ✅ 每个设备处理 1/8 数据，没有 all-gather
```

### 2.3 内存使用对比

假设总数据 8GB：

```
只用 device_put：
  Device 0: 1GB（分片数据）→ all-gather → 8GB（完整数据）💥 OOM
  Device 1: 1GB（分片数据）→ 空闲
  ...
  Device 7: 1GB（分片数据）→ 空闲

用 shard_map：
  Device 0: 1GB（分片数据）→ 处理 → 1GB（输出）✅
  Device 1: 1GB（分片数据）→ 处理 → 1GB（输出）✅
  ...
  Device 7: 1GB（分片数据）→ 处理 → 1GB（输出）✅
```

### 2.4 关于 device_put with shard

**结论：使用 `shard_map` 后，通常不需要手动 `device_put`！**

```python
# ❌ 不需要这样做：
latents_sharded = jax.device_put(
    latents, 
    NamedSharding(mesh, P(None, None, 'tp', None, None))
)
output = sharded_decode(latents_sharded)

# ✅ 直接这样就可以：
output = sharded_decode(latents)  # shard_map 会自动处理分片
```

**例外情况：** 如果数据**已经在正确的分片状态**，手动 `device_put` 可以避免不必要的数据移动。

---

## 3. 跨分片依赖问题

### 3.1 问题描述

在视频解码等场景中，可能存在帧间依赖：

```
Latent Frame 0 → 解码 → Output Frames 0,1,2,3
Latent Frame 1 → 解码 → Output Frames 4,5,6,7 (需要 Frames 2,3)
Latent Frame 2 → 解码 → Output Frames 8,9,10,11 (需要 Frames 6,7)
```

**问题：**
- ✅ **顺序依赖**：必须先解码 Frame 0，才能解码 Frame 1
- ✅ **跨设备依赖**：Chip1 需要 Chip0 的输出
- ❌ **当前简单 shard_map**：各设备并行独立解码

### 3.2 空间依赖问题

即使改为空间分片，卷积操作对相邻像素也有依赖，可能产生"分割线"问题。

### 3.3 解决方案

#### 方案A：使用 Halo Exchange

```python
from jax import lax

def decode_on_slice_with_halo(latents_slice):
    """处理空间边界依赖"""
    
    # 1. 从相邻设备获取边界数据（halo exchange）
    axis_index = lax.axis_index('tp')
    
    # 定义通信模式
    perm = [(i, (i+1) % mesh.shape['tp']) for i in range(mesh.shape['tp'])]
    
    # 获取左右邻居的边界
    left_halo = lax.ppermute(prev_boundary, 'tp', perm=perm)
    
    # 2. 拼接 halo 区域
    extended_slice = jnp.concatenate([left_halo, latents_slice, right_halo], axis=2)
    
    # 3. 解码扩展的切片
    output = vae.decode(extended_slice)
    
    # 4. 裁剪掉 halo 部分
    return output[..., halo_size:-halo_size, :, :]
```

**缺点：**
- ❌ 需要设备间通信
- ❌ 代码复杂
- ❌ 性能开销

#### 方案B：使用 scan 处理顺序依赖

```python
from jax import lax

def sequential_decode(latents_all):
    """顺序解码，处理时间依赖"""
    
    def decode_step(carry, latent_frame):
        prev_frames = carry  # 前面已解码的帧
        
        if prev_frames is None:
            # 第一帧
            output = vae.decode(latent_frame)
        else:
            # 使用前 2 帧作为 context
            context = prev_frames[:, :, -2:, :, :]
            extended = jnp.concatenate([context, latent_frame], axis=2)
            output = vae.decode(extended)
            output = output[:, :, 2:, :, :]  # 裁剪
        
        # 更新 carry
        new_carry = output if prev_frames is None else \
                    jnp.concatenate([prev_frames, output], axis=2)
        
        return new_carry, output
    
    # 使用 scan 顺序处理
    _, outputs = lax.scan(decode_step, init=None, xs=latents_all)
    
    return jnp.concatenate(outputs, axis=2)
```

**缺点：**
- ❌ 失去并行性 - 完全顺序执行
- ❌ 无法利用多设备

#### 方案C：改变分片维度（推荐）

```python
# 在空间维度分片，保持时间维度完整
in_specs=P(None, None, None, 'tp', None)  # H 维度

# 每个设备：
# - 处理完整的时间序列（所有帧）
# - 只处理部分高度
# - 使用 halo exchange 处理空间边界
```

### 3.4 shard_map 能处理依赖吗？

**答案：可以，但需要显式编程。**

`shard_map` **不会自动**处理跨分片依赖，但它提供了工具（如 `lax.ppermute`）让你**手动实现**。

**关键限制：** `shard_map` 不能表达顺序依赖，因为所有设备必须执行相同的程序（SPMD - Single Program Multiple Data）。

---

## 4. JAX Map 函数家族

### 4.1 完整列表

| Map 类型 | 用途 | 执行方式 | 设备 | 典型场景 |
|---------|------|---------|------|---------|
| **vmap** | 向量化 | 并行（向量化） | 单设备 | 批处理 |
| **pmap** | 数据并行 | 并行（多设备） | 多设备 | 简单多GPU |
| **shard_map** | 分片并行 | 并行（可控） | 多设备 | 复杂分片 |
| **scan** | 顺序循环 | 顺序 | 单/多设备 | 时间序列 |
| **tree_map** | 结构映射 | - | - | 嵌套数据 |

### 4.2 vmap - 向量化映射

```python
# 原始函数：处理单个样本
def process_single(x):
    return x ** 2 + 1

# 使用 vmap（高效）
process_batch = jax.vmap(process_single)

# 使用
batch = jnp.array([[1, 2], [3, 4], [5, 6]])  # (3, 2)
output = process_batch(batch)
```

**特点：**
- ✅ 自动向量化
- ✅ 单设备，所有数据在同一设备
- ✅ 简单易用
- ❌ 受单设备内存限制

### 4.3 pmap - 并行映射

```python
# pmap 多设备
pmap_square = pmap(square)

# 使用（假设4个GPU）
data = jnp.array([1, 2, 3, 4])  # (4,)
result = pmap_square(data)

# 每个GPU处理一个元素
# GPU:0 处理 data[0] = 1
# GPU:1 处理 data[1] = 2
# ...
```

**特点：**
- ✅ 自动数据并行
- ✅ 简单易用
- ❌ 第一维必须等于设备数
- ❌ 不支持复杂分片模式
- ⚠️ 正在被 `shard_map` 替代

### 4.4 scan - 顺序映射

```python
from jax import lax

def cumulative_sum(arr):
    def step(carry, x):
        new_sum = carry + x
        return new_sum, new_sum  # (新carry, 输出)
    
    init = 0
    final_carry, outputs = lax.scan(step, init, arr)
    return outputs

# 示例
arr = jnp.array([1, 2, 3, 4, 5])
result = cumulative_sum(arr)  # [1, 3, 6, 10, 15]
```

**特点：**
- ✅ 可微分的循环
- ✅ 支持累积状态
- ✅ 顺序依赖的最佳选择
- ❌ 顺序执行（不并行）

**使用场景：**
- RNN/LSTM 的时间步迭代
- 帧间依赖的视频解码
- 任何需要累积状态的循环

### 4.5 tree_map - 树结构映射

```python
from jax import tree_util

# 处理嵌套数据结构
data = {
    'weights': jnp.array([1, 2, 3]),
    'biases': jnp.array([0.1, 0.2]),
    'nested': {
        'layer1': jnp.array([5, 6])
    }
}

# 对所有叶子节点应用函数
doubled = tree_util.tree_map(lambda x: x * 2, data)
```

**特点：**
- ✅ 处理任意嵌套结构
- ✅ 适合模型参数等复杂结构
- ✅ 保持结构不变

---

## 5. vmap vs pmap 详解

### 5.1 核心区别

虽然看起来相似，但**本质完全不同**：

#### vmap: 单设备向量化

```python
def square(x):
    return x ** 2

vmap_square = vmap(square)

data = jnp.array([1, 2, 3, 4])  # (4,)
result = vmap_square(data)

# 内部：
# ✅ 所有数据在单个设备（如 GPU:0）
# ✅ JAX 将循环优化为向量化操作
# ✅ 在单个设备上并发执行（向量化）
```

**内存布局：**
```
设备 GPU:0:  [1, 2, 3, 4] → 向量化计算 → [1, 4, 9, 16]
```

#### pmap: 多设备并行

```python
pmap_square = pmap(square)

data = jnp.array([1, 2, 3, 4])  # (4,)
result = pmap_square(data)

# 内部：
# GPU:0 处理 data[0] = 1
# GPU:1 处理 data[1] = 2
# GPU:2 处理 data[2] = 3
# GPU:3 处理 data[3] = 4
```

**内存布局：**
```
设备 GPU:0:  [1] → 计算 → [1]
设备 GPU:1:  [2] → 计算 → [4]
设备 GPU:2:  [3] → 计算 → [9]
设备 GPU:3:  [4] → 计算 → [16]
```

### 5.2 详细对比表

| 特性 | vmap | pmap |
|------|------|------|
| **设备数量** | 单设备 | 多设备 |
| **内存位置** | 所有数据在一个设备 | 数据分布在多个设备 |
| **第一维要求** | 任意大小 | 必须等于设备数 |
| **适用场景** | 批处理（数据小） | 数据并行（数据大） |
| **通信开销** | 无 | 有（跨设备） |
| **扩展性** | 受单设备内存限制 | 可扩展到多设备 |

### 5.3 实际使用场景

#### 场景1：数据量小（几百MB）

```python
# 用 vmap - 简单高效
batch_data = jnp.ones((1000, 224, 224, 3))
result = vmap(model.forward)(batch_data)  # 单GPU就够了
```

#### 场景2：数据量大（几十GB）

```python
# 用 pmap - 分布到多GPU
num_gpus = jax.device_count()
batch_data = jnp.ones((num_gpus, 128, 224, 224, 3))
result = pmap(vmap(model.forward))(batch_data)
```

#### 场景3：超大数据 + 灵活分片

```python
# 用 shard_map - 最灵活
batch_data = jnp.ones((1024, 224, 224, 3))  # 任意大小
result = shard_map(
    vmap(model.forward),
    mesh=mesh,
    in_specs=P('devices', None, None, None),
    out_specs=P('devices', None, None, None)
)(batch_data)
```

---

## 6. Mesh 的重要性

### 6.1 为什么 pmap 没有 mesh？

`pmap` 是**早期设计**（2018-2020），采用**隐式设备管理**：

```python
# pmap 自动使用所有可用设备
parallel_fn = pmap(my_function)

# 内部：
# ✅ 自动发现设备：jax.devices()
# ✅ 简单一维排列：[GPU:0, GPU:1, GPU:2, ...]
# ✅ 第一维自动分配到设备
```

### 6.2 pmap 的限制

```python
# ❌ 限制1：第一维必须等于设备数
data = jnp.ones((8, 100))  # 如果只有4个GPU → 错误！

# ❌ 限制2：只支持一维设备排列
# 不能表达 (2×4) 的设备网格

# ❌ 限制3：分片模式固定
# 只能在第一维分片

# ❌ 限制4：难以表达复杂并行模式
# 如模型并行 + 数据并行
```

### 6.3 shard_map 的显式 mesh

```python
from jax.sharding import Mesh
from jax.experimental import mesh_utils

# 显式创建设备网格
devices = mesh_utils.create_device_mesh((2, 4))  # 2×4网格
mesh = Mesh(devices, ('data', 'model'))

# 显式指定分片模式
sharded_fn = shard_map(
    fn,
    mesh=mesh,
    in_specs=P('data', 'model'),  # 明确说明如何分片
    out_specs=P('data', 'model')
)
```

**优势：**
```python
# ✅ 支持任意形状数据
data = jnp.ones((100, 512))  # 不需要第一维等于设备数

# ✅ 支持多维设备网格
mesh = Mesh(devices, ('dp', 'mp', 'pp'))  # 3维并行

# ✅ 灵活的分片模式
in_specs=P('dp', None)      # 只在数据并行维度分片
in_specs=P(None, 'mp')      # 只在模型并行维度分片
in_specs=P('dp', 'mp')      # 两个维度都分片
```

### 6.4 复杂并行模式对比

#### pmap 实现 2D 并行（繁琐）

```python
# pmap 嵌套实现数据 + 模型并行
data_parallel = pmap(fn, axis_name='dp')
model_parallel = pmap(data_parallel, axis_name='mp')

# 需要手动重塑数据
data = data.reshape(2, 4, ...)  # 假设 2×4 设备
result = model_parallel(data)
```

#### shard_map 实现 2D 并行（清晰）

```python
# shard_map 直接表达 2D 并行
devices = mesh_utils.create_device_mesh((2, 4))
mesh = Mesh(devices, ('dp', 'mp'))

sharded_fn = shard_map(
    fn,
    mesh=mesh,
    in_specs=P('dp', 'mp'),  # 清晰的2D分片
    out_specs=P('dp', 'mp')
)

data = jnp.ones((64, 1024))  # 任意形状
result = sharded_fn(data)
```

### 6.5 JAX 并行编程演进

```
2018-2020: pmap 时代
  - 简单隐式
  - 单一维度
  - 易于入门

2021: xmap 实验
  - 引入命名轴
  - 复杂但灵活

2022-2023: mesh + shard_map
  - 显式设备拓扑
  - 统一 API
  - 生产就绪

2024+: 推荐使用
  - shard_map + mesh
  - pmap 逐步淘汰
```

---

## 7. 实践建议

### 7.1 选择决策树

```
需要处理数据
    │
    ├─ 数据 < 单设备内存？
    │   ├─ 是 → 有顺序依赖？
    │   │   ├─ 无 → vmap
    │   │   └─ 有 → scan
    │   │
    │   └─ 否 → 依赖模式？
    │       ├─ 简单数据并行 → pmap
    │       ├─ 复杂分片 → shard_map
    │       └─ 有顺序依赖 → shard_map + scan
```

### 7.2 常见模式

#### 模式1：批量处理（单设备）

```python
# 使用 vmap
process_batch = jax.vmap(process_single)
results = process_batch(batch_data)
```

#### 模式2：简单多设备（被淘汰）

```python
# 使用 pmap（不推荐，但还能用）
process_parallel = jax.pmap(process_single)
results = process_parallel(reshaped_data)
```

#### 模式3：复杂分片（推荐）

```python
# 使用 shard_map + mesh
mesh = Mesh(devices, ('dp', 'mp'))
process_sharded = shard_map(
    process_single,
    mesh=mesh,
    in_specs=P('dp', None),
    out_specs=P('dp', None)
)
results = process_sharded(data)
```

#### 模式4：顺序依赖

```python
# 使用 scan
def step(carry, x):
    new_state = f(carry, x)
    return new_state, output

_, results = lax.scan(step, init, sequence)
```

#### 模式5：组合使用

```python
# shard_map + vmap：每个设备处理一个批次
sharded_batch = shard_map(
    jax.vmap(process_single),
    mesh=mesh,
    in_specs=P('devices', None),
    out_specs=P('devices', None)
)
```

### 7.3 性能优化建议

1. **优先使用 vmap**：如果数据适合单设备，vmap 最简单高效
2. **避免 all-gather**：确保使用 shard_map 包装计算
3. **合理分片维度**：
   - 无依赖：任意维度
   - 空间依赖：考虑 halo exchange
   - 时间依赖：使用 scan 或改为空间分片
4. **显式优于隐式**：使用 shard_map + mesh 而不是 pmap
5. **测试验证**：对比分片和非分片结果，确保数值一致性

### 7.4 调试建议

1. **检查分片是否生效**：
```python
print(f"Input sharding: {jax.device_get(input.sharding)}")
print(f"Output sharding: {jax.device_get(output.sharding)}")
```

2. **监控内存使用**：
```python
# 在每个设备上检查内存
for device in jax.devices():
    print(f"{device}: {device.memory_stats()}")
```

3. **验证数值正确性**：
```python
# 对比分片和非分片结果
result_sharded = sharded_fn(data)
result_baseline = baseline_fn(data)
assert jnp.allclose(result_sharded, result_baseline, rtol=1e-5)
```

---

## 总结

### 核心要点

1. **shard_map**：显式分片并行的现代工具
   - 需要 mesh 定义设备拓扑
   - 需要 in_specs/out_specs 定义分片规则
   - 不自动处理跨分片依赖

2. **vmap vs pmap**：
   - vmap = 单设备向量化
   - pmap = 多设备数据并行
   - 两者都有用，但场景不同

3. **依赖处理**：
   - 空间依赖 → halo exchange
   - 时间依赖 → scan 或改分片维度
   - shard_map 提供工具但需手动实现

4. **mesh 的重要性**：
   - 显式设备拓扑
   - 支持复杂并行模式
   - 现代 JAX 的核心

### 快速参考

| 需求 | 推荐工具 |
|------|---------|
| 批处理（单设备） | vmap |
| 简单多设备 | pmap（维护模式） |
| 复杂分片 | shard_map |
| 顺序循环 | scan |
| 嵌套结构 | tree_map |
| 多维并行 | shard_map + mesh |

---

## 参考资源

- [JAX 官方文档](https://jax.readthedocs.io/)
- [shard_map 教程](https://jax.readthedocs.io/en/latest/notebooks/shard_map.html)
- [Distributed arrays and automatic parallelization](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)
- [Parallel evaluation in JAX](https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html)

---

*文档创建时间：2025-11-04*
*基于 JAX 0.4+ 和 torchax 最新版本*