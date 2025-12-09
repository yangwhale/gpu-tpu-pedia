# PyTorch GPU → torchax TPU 迁移指南

本文档记录了将 HunyuanVideo-1.5 Transformer 从 GPU PyTorch 迁移到 TPU torchax 的完整过程。

---

## 📚 目录

1. [迁移概览](#1-迁移概览)
2. [快速开始](#2-快速开始)
3. [核心修复](#3-核心修复)
4. [DeepCache 加速](#4-deepcache-加速)
5. [常见陷阱](#5-常见陷阱)
6. [性能优化](#6-性能优化)
7. [调试技巧](#7-调试技巧)

---

## 1. 迁移概览

### 技术栈对比

| 技术层 | GPU 版本 | TPU 版本 |
|--------|----------|----------|
| 运行框架 | PyTorch | torchax (PyTorch → JAX) |
| Attention | Flash Attention | Splash Attention (Pallas) |
| JIT 编译 | torch.compile | XLA JIT |
| 分布式 | NCCL + 手动 SP/TP | GSPMD (自动分片) |
| 数据类型 | fp16 / fp32 | bf16 (原生支持) |

### 迁移流程

```
1. 环境设置 → 2. Monkey-patch → 3. 模型加载 → 4. 权重分片 → 5. JIT 编译 → 6. 推理
```

---

## 2. 快速开始

### 2.1 环境设置

```python
import jax
import torch
import torchax
from jax.sharding import Mesh
from jax.experimental import mesh_utils

# 创建 JAX Mesh
mesh_devices = mesh_utils.create_device_mesh((jax.device_count(), 1, 1))
mesh = Mesh(mesh_devices, ('tp', 'dp', 'sp'))

# 创建 torchax 环境
env = torchax.default_env()
env._mesh = mesh
env.config.use_tpu_splash_attention = True

torch.set_default_dtype(torch.bfloat16)
```

### 2.2 注册 Splash Attention

```python
from torchax.ops import ops_registry

custom_attention = functools.partial(scaled_dot_product_attention, env=env)
env._ops[torch.nn.functional.scaled_dot_product_attention] = ops_registry.Operator(
    torch.nn.functional.scaled_dot_product_attention,
    custom_attention,
    is_jax_function=False,
    is_user_defined=True,
    needs_env=False,
    is_view_op=False,
)
```

### 2.3 模型加载和分片

```python
model = Model.from_pretrained(path, torch_dtype=torch.bfloat16)

with env:
    with jax.default_device('cpu'):
        state_dict = model.state_dict()
        state_dict = env.to_xla(state_dict)
        model.load_state_dict(state_dict, assign=True)
    
    weights = shard_weights(mesh, model.state_dict())
    model.load_state_dict(weights, assign=True, strict=False)

model.eval()
```

### 2.4 JIT 编译和推理

```python
with env:
    model = torchax.compile(model, torchax.CompileOptions(
        jax_jit_kwargs={'static_argnames': ('return_dict',)}
    ))

with mesh, env:
    with torch.no_grad():
        for t in timesteps:
            output = model(inputs)
            latents = scheduler.step(output, t, latents)[0]
            latents = latents.to(torch.bfloat16)

os._exit(0)  # 强制退出，避免 JAX 后台线程阻塞
```

---

## 3. 核心修复

### 3.1 Attention Mask（根本原因）

**问题**：GPU 使用 `flex_attention` + `score_mod` 屏蔽 padding，TPU 版本忽略了 mask。

**解决方案**：将 padding 位置的 K/V 设为零

```python
if text_mask is not None:
    text_mask_expanded = text_mask.unsqueeze(-1).unsqueeze(-1).to(encoder_key.dtype)
    encoder_key = encoder_key * text_mask_expanded
    encoder_value = encoder_value * text_mask_expanded

query = torch.cat([query, encoder_query], dim=1)
key = torch.cat([key, encoder_key], dim=1)
value = torch.cat([value, encoder_value], dim=1)

hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=None)
```

**原理**：当 K[i]=0 时，QK^T[i]≈0，softmax 后权重很低，效果近似于 -inf mask。

### 3.2 vision_states 处理

**问题**：`torch.all(vision_states == 0)` 在 JIT 中导致 ConcretizationTypeError。

**解决方案**：t2v 模式直接传入 `None`

```python
if task_type == 't2v':
    vision_states = None  # 跳过 vision_in 分支
```

### 3.3 布尔索引不支持

**问题**：torchax 不支持 `tensor[bool_mask]`

**解决方案**：使用简化逻辑或 argsort + gather

```python
# 简化方案：直接拼接（禁用 reorder）
reorder_txt = torch.concat([byt5_txt, txt], dim=1)
reorder_mask = torch.concat([byt5_text_mask, text_mask], dim=1)
```

### 3.4 动态 Tensor 创建

**问题**：JIT 内部 `torch.arange()` 导致重复编译

**解决方案**：预计算并缓存

```python
# JIT 外预计算
freqs_cos, freqs_sin = model.get_rotary_pos_embed(size)
with env:
    model._cached_freqs_cos = freqs_cos.to('jax')
    model._cached_freqs_sin = freqs_sin.to('jax')

# Monkey-patch 使用缓存
def cached_get_rotary(self, size):
    return self._cached_freqs_cos, self._cached_freqs_sin
model.get_rotary_pos_embed = types.MethodType(cached_get_rotary, model)
```

---

## 4. DeepCache 加速

### 4.1 概述

DeepCache 通过缓存 transformer 中间状态来跳过部分层的计算，实现加速。

**HunyuanVideo 结构**：
- 20 个 double_blocks + 40 个 single_blocks
- 缓存点：double_blocks 输出 (img, txt)
- 跳过：20 层 double_blocks
- 理论加速比：61/41 ≈ 1.49x

### 4.2 TPU 兼容性问题

**问题**：常见的 `jax.lax.cond` 方案在 torchax 中不可用。

**原因**：
1. `jax.lax.cond` 要求两个分支返回完全相同的 pytree 结构
2. torchax 的 tensor wrapper 使结构匹配困难
3. JAX tracer 泄漏问题

### 4.3 解决方案：分离模块

使用两个独立编译的模块，在 Python 层做条件分支：

```python
class FullForwardModule(torch.nn.Module):
    """完整 forward，返回 (output, img_cache, txt_cache, vec, text_mask)"""
    def forward(self, hidden_states, ...):
        # 执行完整 transformer
        # 保存 double_blocks 后的状态
        return (output, img_after_double, txt_after_double, vec, text_mask)

class CachedForwardModule(torch.nn.Module):
    """使用缓存，跳过 double_blocks"""
    def forward(self, cached_img, cached_txt, vec, freqs_cos, freqs_sin, text_mask):
        # 只执行 single_blocks + final_layer
        return output
```

### 4.4 TPUDeepCache 类

```python
class TPUDeepCache:
    def __init__(self, cache_start_step, cache_end_step, cache_step_interval, total_steps):
        self.no_cache_steps = set(
            list(range(0, cache_start_step)) +
            list(range(cache_start_step, cache_end_step, cache_step_interval)) +
            list(range(cache_end_step, total_steps))
        )
        self.cached_img = None
        self.cached_txt = None
    
    def should_use_cache(self, step):
        return step not in self.no_cache_steps and self.cached_img is not None
    
    def update_cache(self, img, txt, vec, text_mask):
        self.cached_img = img
        self.cached_txt = txt
        # ...
```

### 4.5 推理循环集成

```python
for i in range(num_steps):
    if deep_cache.should_use_cache(i):
        # 使用缓存路径（跳过 double_blocks）
        cached_img, cached_txt, vec, text_mask = deep_cache.get_cache()
        noise_pred = cached_forward_fn(cached_img, cached_txt, vec, ...)
    else:
        # 完整 forward（同时更新缓存）
        output = full_forward_fn(latent_model_input, ...)
        noise_pred, img_cache, txt_cache, vec, text_mask = output
        deep_cache.update_cache(img_cache, txt_cache, vec, text_mask)
```

### 4.6 使用方法

```bash
python stage2_transformer_flax_experimental_deepcache.py \
    --enable_cache \
    --cache_start_step 11 \
    --cache_end_step 45 \
    --cache_step_interval 4 \
    --video_length 121
```

### 4.7 性能结果

| 配置 | 无 DeepCache | 有 DeepCache | 加速比 |
|------|-------------|-------------|--------|
| 121帧, 50步 | ~350s | ~203s | 1.72x |

### 4.8 关键经验

1. **不要使用 jax.lax.cond**：torchax 环境下会导致 tracer 泄漏
2. **分离编译**：两个模块独立编译，避免结构匹配问题
3. **Python 分支**：条件判断放在 Python 层，不在 JIT 内部
4. **freqs_cos/sin 缓存**：使用 `transformer._cached_freqs_cos/sin`，不依赖 JIT 返回值
5. **清除预热缓存**：warmup 后调用 `deep_cache.clear()`

---

## 5. 常见陷阱

### 速查表

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 视频有竖条纹/不跟随提示词 | Attention Mask 未处理 | K/V 置零方案 |
| ConcretizationTypeError | 动态条件/断言 | Monkey-patch 移除 |
| 布尔索引报错 | torchax 不支持 | 使用 torch.where 或乘法 |
| 程序不退出 | JAX 后台线程 | `os._exit(0)` |
| 第一步慢（60s+） | XLA 编译 | 正常，使用 warmup |
| OOM | 完整 attention mask | Splash Attention + K/V 置零 |
| Scheduler 后 dtype 变化 | 内部转 fp32 | 每步后 `.to(bf16)` |

### 典型修复模式

```python
# 布尔索引 → 乘法
# ❌ selected = tensor[mask]
# ✅ selected = tensor * mask.unsqueeze(-1).float()

# 运行时检查 → 移除
# ❌ assert tensor.min() >= 0
# ✅ （直接删除）

# 动态 tensor → 预计算
# ❌ torch.arange(n) 在 JIT 内
# ✅ 预计算并缓存到模型属性
```

---

## 6. 性能优化

### 6.1 权重分片策略

#### 6.1.1 TP + fc2/proj Replicated（默认，推荐）

将 MLP fc2 和 attention proj 权重完全复制到所有设备，消除 all-reduce 开销。

```python
# 分片策略定义
transformer_shardings_tp_fc2_replicated = {
    # Column Parallel（Q/K/V, fc1）- 输出维度分片
    r'.*\.img_attn_q\.weight$': (('tp', 'sp'), None),
    r'.*\.img_mlp\.fc1\.weight$': (('tp', 'sp'), None),
    
    # REPLICATED（fc2, proj）- 无 all-reduce
    r'.*\.img_attn_proj\.weight$': (None, None),
    r'.*\.img_mlp\.fc2\.weight$': (None, None),
}
```

**性能对比（121帧 720p, 8× TPU v6e）**：

| 分片模式 | Step Time | 相对性能 | HBM 增量 |
|----------|-----------|----------|----------|
| 标准 TP | 8.12s | baseline | 0 GB |
| **TP + fc2 Replicated** | **7.29s** | **+10.2%** | ~12 GB |
| TP + 全 MLP Replicated | 8.18s | -0.7% | ~21 GB |

**关键发现**：
- 只复制 Row Parallel 层（fc2, proj）是最优策略
- 复制 Column Parallel 层（Q/K/V, fc1）没有收益，反而增加 HBM 带宽压力
- 原因：Column Parallel 层本来就不需要 all-reduce

#### 6.1.2 标准 TP（Megatron Column-Row）

每个 block 有 2 次 all-reduce：

```
Attention: Q/K/V (Column) → proj (Row) → all-reduce
MLP: fc1 (Column) → fc2 (Row) → all-reduce
```

```python
transformer_shardings_tp = {
    r'.*\.img_attn_q\.weight$': (('tp', 'sp'), None),   # Column
    r'.*\.img_attn_proj\.weight$': (None, ('tp', 'sp')),  # Row (all-reduce)
    r'.*\.img_mlp\.fc1\.weight$': (('tp', 'sp'), None),   # Column
    r'.*\.img_mlp\.fc2\.weight$': (None, ('tp', 'sp')),   # Row (all-reduce)
}
```

#### 6.1.3 Profiler 分析

使用 JAX Profiler 可以观察 all-reduce 操作：

```bash
python stage2_transformer_flax.py --enable_profiler --num_inference_steps 3
```

典型时间分布（单个 block）：
- Splash Attention: ~35ms（主导）
- Linear + all-reduce: ~45ms

即使复制了 fc2/proj 权重，仍会有部分 all-reduce，因为激活值仍是 sharded 的。

### 6.2 Warmup 策略

XLA 编译是惰性的，前 1-2 步会触发编译。

```python
# 推荐 2 步预热
if args.warmup_steps > 0:
    run_denoising_loop(latents, timesteps, args.warmup_steps, is_warmup=True)
```

### 6.3 JIT 缓存

```python
jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
```

效果：首次 ~60s 编译 → 后续 ~5s 加载缓存

### 6.4 准确计时

```python
output = model(input)
torchax.interop.call_jax(jax.block_until_ready, output._elem)
step_time = time.perf_counter() - start  # 准确时间
```

### 6.5 性能基准

| 配置 | Token 数 | 总时间 | 每步时间 |
|------|----------|--------|----------|
| 49帧, 720p | 46,800 | ~215s | ~4.3s |
| 121帧, 720p (标准 TP) | 111,600 | ~406s | ~8.1s |
| 121帧, 720p (TP + fc2 Replicated) | 111,600 | ~365s | ~7.3s |
| 121帧 + DeepCache | 111,600 | ~203s | ~4.1s |

环境：TPU v6e-8，50 步推理

---

## 7. 调试技巧

### 查看完整 traceback

```bash
JAX_TRACEBACK_FILTERING=off python script.py
```

### 检测 XLA tensor

```python
def is_xla_tensor(t):
    return hasattr(t, '_elem') or ('jax' in str(getattr(t, 'device', '')))
```

### 调试打印

```python
def debug_tensor(name, t):
    print(f"{name}: shape={t.shape}, dtype={t.dtype}, mean={t.float().mean():.4f}")
```

---

## 📋 迁移 Checklist

- [ ] 创建 JAX Mesh 和 torchax 环境
- [ ] 注册 Splash Attention
- [ ] Monkey-patch 不兼容代码（在导入模型前）
- [ ] 加载模型并转换权重到 XLA
- [ ] 权重分片
- [ ] 预计算动态 tensor（如 Rotary Embeddings）
- [ ] JIT 编译
- [ ] 修复 Attention Mask（K/V 置零）
- [ ] 每步后 dtype 转换
- [ ] 使用 `os._exit(0)` 退出

---

## 📚 参考资源

- [torchax GitHub](https://github.com/pytorch/xla)
- [JAX Splash Attention](https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/splash_attention)
- [HunyuanVideo-1.5](https://github.com/Tencent/HunyuanVideo)
- [TPU bf16 精度说明](https://cloud.google.com/tpu/docs/bfloat16)