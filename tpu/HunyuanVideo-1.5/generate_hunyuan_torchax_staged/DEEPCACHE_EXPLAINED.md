# DeepCache for HunyuanVideo-1.5 on TPU

本文档介绍 DeepCache 在 TPU/torchax 环境下的实现原理。

---

## 目录

1. [概述](#1-概述)
2. [HunyuanVideo Transformer 架构](#2-hunyuanvideo-transformer-架构)
3. [缓存策略](#3-缓存策略)
4. [TPU 实现方案](#4-tpu-实现方案)
5. [性能结果](#5-性能结果)
6. [使用方法](#6-使用方法)

---

## 1. 概述

### 问题背景

Diffusion 模型推理需要多次迭代（通常 20-50 步），每步都要完整执行 Transformer 前向传播，计算量巨大。

### DeepCache 核心思想

DeepCache 论文发现：**相邻去噪步骤的深层特征变化很小**。因此可以缓存深层特征并复用，只重新计算最后几层。

```
标准推理：每步执行所有层
DeepCache：缓存深层特征，仅执行最后 2 层
```

---

## 2. HunyuanVideo Transformer 架构

HunyuanVideo-1.5 720p_t2v 架构：

| 组件 | 层数 |
|------|------|
| Double Blocks | 54 层 (Block 0-53) |
| Single Blocks | 0 层 |
| Final Layer | 1 层 |
| **总计** | **55 层** |

**Double Block** 处理两个分离的流：
- `img`：视频特征
- `txt`：文本特征

---

## 3. 缓存策略

### 缓存点：Block 52 之后

缓存 Block 52 的输出 `(img, txt)`，在缓存命中时跳过 Block 0-52（53 层），仅执行 Block 53 + Final Layer（2 层）。

```
完整 Forward：Block 0-52 → 缓存 → Block 53 → Final Layer
缓存 Forward：读取缓存 → Block 53 → Final Layer
```

### 理论加速比

| 路径 | 计算层数 |
|------|----------|
| 完整 Forward | 55 层 |
| 缓存 Forward | 2 层 |

50% cache hit 时理论加速比：**1.93x**

### 缓存工作流程

以 `cache_step_interval = 4` 为例：

```
Step 11: 完整计算 → 保存缓存
Step 12: 使用缓存 (跳过 53 层)
Step 13: 使用缓存
Step 14: 使用缓存
Step 15: 完整计算 → 刷新缓存
Step 16-18: 使用缓存
...循环...
```

### 缓存参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `cache_start_step` | 11 | 开始使用缓存的步数 |
| `cache_end_step` | 45 | 停止使用缓存的步数 |
| `cache_step_interval` | 4 | 缓存刷新间隔 |

---

## 4. TPU 实现方案

### 为什么不能直接移植 GPU 版本

torchax/JAX 的限制：
- ❌ 不支持 JIT 内动态条件分支
- ❌ 不支持运行时 if/else
- ❌ PyTree 结构必须静态匹配

### 解决方案：分离模块

将完整 forward 和缓存 forward 分离为两个独立模块，在 Python 层做条件分支：

```python
# Python 层条件分支（不参与 JIT 编译）
if deep_cache.should_use_cache(step):
    output = cached_forward_fn(...)   # 只执行 Block 53 + Final Layer
else:
    output = full_forward_fn(...)     # 执行所有 54 层
    deep_cache.update_cache(...)
```

### 核心组件

**FullForwardModule**：执行所有 54 层，在 Block 52 之后保存缓存
```python
for index, block in enumerate(transformer.double_blocks):
    if index == num_double_blocks - 1:  # Block 53 之前
        cached_img, cached_txt = img, txt  # 保存缓存
    img, txt = block(...)
```

**CachedForwardModule**：跳过 Block 0-52，只执行 Block 53 + Final Layer
```python
img, txt = cached_img, cached_txt  # 使用缓存
last_block = transformer.double_blocks[-1]
img, txt = last_block(img, txt, ...)  # 只执行 Block 53
output = transformer.final_layer(img, vec)
```

**TPUDeepCache**：管理缓存状态
```python
class TPUDeepCache:
    cached_img: Tensor
    cached_txt: Tensor
    cached_freqs_cis: Tensor
    no_cache_steps: Set[int]
```

---

## 5. 性能结果

| 配置 | 无 DeepCache | 有 DeepCache | 加速比 |
|------|-------------|-------------|--------|
| 49 帧, 50 步 | 113.95s | 62.43s | **1.83x** |
| 每步时间 | 2.28s | 1.25s (avg) | - |
| Cache Hit | 0% | 50% | - |

实测加速 1.83x，接近理论值 1.93x。

---

## 6. 使用方法

```bash
python stage2_transformer_flax_experimental_deepcache.py \
    --enable_cache \
    --cache_start_step 11 \
    --cache_end_step 45 \
    --cache_step_interval 4 \
    --video_length 49 \
    --num_inference_steps 50
```

---

## 参考资料

- [DeepCache 论文](https://arxiv.org/abs/2312.00858)
- [HunyuanVideo-1.5](https://github.com/Tencent/HunyuanVideo)