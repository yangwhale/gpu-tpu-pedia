# PyTorch to torchax TPU Migration Guide

本文档记录了将 HunyuanVideo-1.5 Transformer 从 GPU PyTorch 改造为 TPU torchax 版本的完整过程，包括所有遇到的问题和解决方案。

## 目录

1. [前置知识](#1-前置知识)
2. [环境准备](#2-环境准备)
3. [改造步骤](#3-改造步骤)
4. [常见问题与解决方案](#4-常见问题与解决方案)
5. [代码模板](#5-代码模板)
6. [调试技巧](#6-调试技巧)
7. [性能优化](#7-性能优化)

---

## 1. 前置知识

### 1.1 torchax 是什么

torchax 是一个让 PyTorch 模型在 JAX/TPU 上运行的桥接库。它通过以下方式工作：
- 将 PyTorch tensor 包装为 JAX array
- 拦截 PyTorch 操作并转换为 JAX 操作
- 提供 JIT 编译支持

### 1.2 关键概念

| 概念 | 说明 |
|------|------|
| `torchax.default_env()` | 获取 torchax 环境，用于设备管理和操作注册 |
| `env.to_xla(tensor)` | 将 PyTorch tensor 转换为 torchax tensor |
| `torchax.compile(model)` | JIT 编译 PyTorch 模型 |
| `tensor.to('jax')` | 将 tensor 移动到 JAX 设备 |
| Splash Attention | TPU 优化的分块注意力实现 |
| shard_map | JAX 的分布式计算原语 |

### 1.3 TPU vs GPU 的关键差异

| 特性 | GPU | TPU |
|------|-----|-----|
| 原生 dtype | fp16/fp32 | bf16 |
| Attention | Flash Attention | Splash Attention |
| 分布式 | NCCL/手动 SP | GSPMD (自动) |
| JIT | torch.compile | XLA JIT |

---

## 2. 环境准备

### 2.1 依赖安装

```bash
# torchax (PyTorch-JAX 桥接)
pip install torchax

# JAX TPU 版本
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# 其他依赖
pip install einops loguru safetensors tqdm
```

### 2.2 JAX 配置

```python
import jax

# 启用编译缓存（加速后续运行）
jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
```

### 2.3 过滤 Warning

```python
import warnings
# 过滤 shard_map 的 deprecation warning
warnings.filterwarnings('ignore', message='.*jax.experimental.shard_map is deprecated.*')
```

---

## 3. 改造步骤

### 步骤 1: 创建 JAX Mesh

Mesh 定义了设备的拓扑结构，用于权重分片和分布式计算。

```python
from jax.sharding import Mesh
from jax.experimental import mesh_utils

# 创建设备网格
tp_dim = jax.device_count()  # Tensor Parallel 维度
dp_dim = 1                    # Data Parallel 维度
sp_dim = 1                    # Sequence Parallel 维度

mesh_devices = mesh_utils.create_device_mesh(
    (tp_dim, dp_dim, sp_dim), 
    allow_split_physical_axes=True
)
mesh = Mesh(mesh_devices, ('tp', 'dp', 'sp'))
```

### 步骤 2: 创建 torchax 环境

```python
import torchax
from torchax.ops import ops_registry
import functools

env = torchax.default_env()
env._mesh = mesh
env.config.use_tpu_splash_attention = True

# 设置默认 dtype 为 bf16
torch.set_default_dtype(torch.bfloat16)
```

### 步骤 3: 注册自定义 Attention

**关键：必须在加载模型之前注册！**

```python
# 保存原始 SDPA
_ORIGINAL_SDPA = torch.nn.functional.scaled_dot_product_attention

# 创建自定义 attention
custom_attention = functools.partial(
    scaled_dot_product_attention,  # 你的实现
    env=env,
    window_size=None  # None = full attention
)

# 注册到 torchax
op_to_override = torch.nn.functional.scaled_dot_product_attention
env._ops[op_to_override] = ops_registry.Operator(
    op_to_override,
    custom_attention,
    is_jax_function=False,
    is_user_defined=True,
    needs_env=False,
    is_view_op=False,
)
```

### 步骤 4: 处理不兼容的代码（Monkey Patching）

**在导入模型之前进行 patching！**

```python
# 示例：Mock GPU 分布式状态
import hyvideo.commons.parallel_states as parallel_states_module
from types import SimpleNamespace

_mock_parallel_state = SimpleNamespace(
    sp=1,
    sp_enabled=False,
    sp_group=None,
)
parallel_states_module.get_parallel_state = lambda: _mock_parallel_state

# 现在才导入模型
from hyvideo.models.transformers.xxx import Model
```

### 步骤 5: 加载模型并转换权重

```python
# 加载模型
model = Model.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    attn_mode='torch',  # 使用 torch SDPA
)

# 转换权重到 XLA
with env:
    with jax.default_device('cpu'):
        state_dict = model.state_dict()
        state_dict = env.to_xla(state_dict)
        model.load_state_dict(state_dict, assign=True)
    
    # 权重分片
    weights = shard_weights(mesh, model.state_dict())
    model.load_state_dict(weights, assign=True, strict=False)
    torchax.interop.call_jax(jax.block_until_ready, weights)

model.eval()
```

### 步骤 6: 预计算动态创建的 Tensors

**JIT 不支持动态 tensor 创建（如 torch.arange）！**

```python
# 在 JIT 编译前预计算
with torch.no_grad():
    freqs_cos, freqs_sin = model.get_rotary_pos_embed((t, h, w))
    with env:
        model._cached_freqs_cos = freqs_cos.to('jax')
        model._cached_freqs_sin = freqs_sin.to('jax')

# Monkey-patch 使用缓存
import types
def cached_get_rotary_pos_embed(self, latent_size):
    if hasattr(self, '_cached_freqs_cos'):
        return self._cached_freqs_cos, self._cached_freqs_sin
    return original_func(latent_size)
model.get_rotary_pos_embed = types.MethodType(cached_get_rotary_pos_embed, model)
```

### 步骤 7: JIT 编译模型

```python
with env:
    model = torchax.compile(
        model,
        torchax.CompileOptions(
            jax_jit_kwargs={'static_argnames': ('return_dict', 'mask_type')}
        )
    )
```

### 步骤 8: 运行推理

```python
with mesh, env:
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            output = model(inputs)
            # 处理 scheduler 返回的 float32
            latents = scheduler.step(...)[0].to(target_dtype)
```

### 步骤 9: 显式退出

**torchax/JAX 后台线程可能阻塞程序退出！**

```python
print("完成")
sys.exit(0)  # 或 os._exit(0)
```

---

## 4. 常见问题与解决方案

### 问题 1: ConcretizationTypeError

**错误信息：**
```
jax.errors.ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected
```

**原因：** JIT 编译时尝试将 traced tensor 转换为 Python 值

**常见触发场景：**
- `if tensor.max() > 1:` - 条件判断
- `assert tensor.max() <= 1` - 断言检查
- `torch.all(tensor == 0)` - 运行时检查
- `tensor.item()` - 提取标量

**解决方案：**
```python
# 方案1: 在 JIT 外部进行检查
# 方案2: 使用 static_argnames
# 方案3: Monkey-patch 移除检查

# 示例：patch 掉有问题的函数
def patched_function(...):
    # 移除断言和运行时检查
    # 直接执行核心逻辑
    pass
original_module.function = patched_function
```

### 问题 2: 布尔索引不支持

**错误信息：**
```
AttributeError: 'View' object has no attribute '_elem'
```

**原因：** torchax 不支持 `tensor[bool_mask]` 形式的布尔索引

**解决方案：**
```python
# 原代码
selected = tensor[mask]

# 修改为
# 方案1: 使用 torch.where
selected = torch.where(mask.unsqueeze(-1), tensor, torch.zeros_like(tensor))

# 方案2: 禁用相关功能
def simplified_function(...):
    # 使用不需要布尔索引的简化逻辑
    return torch.cat([a, b], dim=1)
```

### 问题 3: GPU 分布式状态初始化

**错误信息：**
```
RuntimeError: CUDA not available
```

**原因：** 代码尝试初始化 CUDA 设备网格

**解决方案：**
```python
# 在导入模型前 mock
import module.parallel_states as ps
from types import SimpleNamespace

ps.get_parallel_state = lambda: SimpleNamespace(
    sp=1,
    sp_enabled=False,
    sp_group=None,
)
```

### 问题 4: dtype 不匹配

**错误信息：**
```
RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same
```

**原因：** 某些操作返回 float32（如 scheduler.step 的精度保护）

**解决方案：**
```python
# 在每个可能改变 dtype 的操作后转回 bf16
latents = scheduler.step(...)[0]
latents = latents.to(torch.bfloat16)
```

### 问题 5: OOM (显存不足)

**错误信息：**
```
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED
```

**原因：** 创建了过大的中间张量（如完整的 attention 矩阵）

**解决方案：**
```python
# 使用 Splash Attention（分块计算）
# 不使用 attention mask（避免创建完整矩阵）
attn_mask = None  # 强制使用 Splash Attention
```

### 问题 6: 程序执行完成后挂起

**原因：** torchax/JAX 后台线程未正常退出

**解决方案：**
```python
# 在程序末尾显式退出
import sys
sys.exit(0)
```

### 问题 7: flash attention 警告

**警告信息：**
```
UserWarning: Falling back from `flash` to `torch`
```

**说明：** 这是正常的，因为 TPU 没有 Flash Attention，会回退到 torch SDPA（然后被我们的 Splash Attention 拦截）

---

## 5. 代码模板

### 5.1 Splash Attention 实现

```python
from jax.experimental.pallas.ops.tpu import splash_attention
from jax.experimental.shard_map import shard_map

BQSIZE = 2048
BKVSIZE = 2048
BKVCOMPUTESIZE = 1024

def _tpu_splash_attention(query, key, value, mesh, scale=None, window_size=None):
    num_heads = query.shape[1]

    def _attention_on_slices(q, k, v):
        scale_factor = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
        q = q * scale_factor

        def pad_to_multiple(x, multiple, axis):
            seq_len = x.shape[axis]
            pad_len = (multiple - seq_len % multiple) % multiple
            if pad_len == 0:
                return x, seq_len
            pad_width = [(0, 0)] * x.ndim
            pad_width[axis] = (0, pad_len)
            return jnp.pad(x, pad_width), seq_len

        def kernel_3d(q_3d, k_3d, v_3d):
            num_heads_on_device = q_3d.shape[0]
            q_3d_padded, q_orig_len = pad_to_multiple(q_3d, BQSIZE, axis=1)
            k_3d_padded, _ = pad_to_multiple(k_3d, BKVSIZE, axis=1)
            v_3d_padded, _ = pad_to_multiple(v_3d, BKVSIZE, axis=1)

            if window_size is not None:
                mask_class = functools.partial(
                    splash_attention.LocalMask, window_size=window_size
                )
            else:
                mask_class = splash_attention.FullMask

            mask = splash_attention.MultiHeadMask(
                [mask_class((q_3d_padded.shape[1], k_3d_padded.shape[1])) 
                 for _ in range(num_heads_on_device)]
            )

            block_sizes = splash_attention.BlockSizes(
                block_q=min(BQSIZE, q_3d_padded.shape[1]),
                block_kv=min(BKVSIZE, k_3d_padded.shape[1]),
                block_kv_compute=min(BKVCOMPUTESIZE, k_3d_padded.shape[1]),
            )
            
            kernel = splash_attention.make_splash_mha(
                mask=mask, block_sizes=block_sizes
            )
            out = kernel(q_3d_padded, k_3d_padded, v_3d_padded)
            return out[:, :q_orig_len, ...]

        return jax.vmap(kernel_3d)(q, k, v)

    # 分片规则
    q_spec = P('dp', 'tp', 'sp', None)
    kv_spec = P('dp', 'tp', None, None)

    sharded_fn = shard_map(
        _attention_on_slices,
        mesh=mesh,
        in_specs=(q_spec, kv_spec, kv_spec),
        out_specs=q_spec,
        check_rep=False,
    )
    return sharded_fn(query, key, value)
```

### 5.2 权重分片模板

```python
from jax.sharding import PartitionSpec as P, NamedSharding
import re

# Tensor Parallel: Column-Row 模式
sharding_rules = {
    # Column Parallel (Q/K/V, FF1)
    r'.*\.q_proj\.weight$': (('tp', 'sp'), None),
    r'.*\.k_proj\.weight$': (('tp', 'sp'), None),
    r'.*\.v_proj\.weight$': (('tp', 'sp'), None),
    r'.*\.fc1\.weight$': (('tp', 'sp'), None),
    
    # Row Parallel (Output, FF2)
    r'.*\.o_proj\.weight$': (None, ('tp', 'sp')),
    r'.*\.fc2\.weight$': (None, ('tp', 'sp')),
}

def shard_weights(mesh, weights, rules):
    for name, tensor in weights.items():
        matched = False
        for pattern, spec in rules.items():
            if re.fullmatch(pattern, name):
                tensor.apply_jax_(
                    jax.device_put, 
                    NamedSharding(mesh, P(*spec))
                )
                matched = True
                break
        if not matched:
            # 未匹配的权重复制到所有设备
            tensor.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
    return weights
```

### 5.3 完整推理循环模板

```python
def main():
    # 1. 设置 JAX 配置
    jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
    torch.set_default_dtype(torch.bfloat16)
    
    # 2. 创建 Mesh
    mesh = create_mesh()
    
    # 3. 创建 torchax 环境
    env = torchax.default_env()
    env._mesh = mesh
    
    # 4. Monkey-patch 不兼容的代码（在导入模型前！）
    patch_incompatible_code()
    
    # 5. 注册自定义 attention
    register_splash_attention(env)
    
    # 6. 加载和转换模型
    model = load_model()
    model = convert_to_xla(model, env, mesh)
    
    # 7. 预计算动态 tensors
    precompute_dynamic_tensors(model, env)
    
    # 8. JIT 编译
    with env:
        model = torchax.compile(model)
    
    # 9. 推理
    with mesh, env:
        with torch.no_grad():
            for step in range(num_steps):
                output = model(inputs)
                latents = scheduler.step(...)[0].to(torch.bfloat16)
    
    # 10. 保存结果
    save_results(latents)
    
    # 11. 显式退出
    sys.exit(0)
```

---

## 6. 调试技巧

### 6.1 查看完整 traceback

```bash
JAX_TRACEBACK_FILTERING=off python script.py
```

### 6.2 检测 XLA tensor

```python
def is_xla_tensor(tensor):
    if tensor is None:
        return False
    if hasattr(tensor, '_elem'):
        return True
    if hasattr(tensor, 'device'):
        device_str = str(tensor.device)
        if 'jax' in device_str or 'xla' in device_str:
            return True
    return False
```

### 6.3 逐步测试

1. 先用 1 个 inference step 测试
2. 确认基本流程通过后再增加步数
3. 出错时检查最后一个成功的操作

### 6.4 打印 tensor 信息

```python
def debug_tensor(name, t):
    if t is None:
        print(f"{name}: None")
    else:
        print(f"{name}: shape={t.shape}, dtype={t.dtype}, device={t.device}")
```

---

## 7. 性能优化

### 7.1 编译缓存

```python
jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
```

首次运行需要编译（约 1 分钟），后续运行使用缓存。

### 7.2 bf16 全程使用

TPU 对 bf16 有原生硬件支持，避免使用 fp32：
- 模型权重: bf16
- 激活值: bf16
- 中间计算: bf16

### 7.3 避免不必要的 device 传输

```python
# 不好：多次传输
tensor = tensor.cpu()
tensor = tensor.to('jax')

# 好：一次传输
tensor = tensor.to('jax')
```

### 7.4 权重分片策略

根据模型大小选择分片策略：
- 小模型: 复制到所有设备
- 大模型: Tensor Parallel (Column-Row)

---

## 8. Checklist

在开始改造前，确认以下事项：

- [ ] 识别所有 GPU 特定代码（CUDA, Flash Attention, NCCL）
- [ ] 识别所有运行时检查（assert, if tensor.max()）
- [ ] 识别所有动态 tensor 创建（torch.arange, torch.zeros）
- [ ] 识别所有布尔索引
- [ ] 确认模型输入输出的 dtype
- [ ] 准备 Splash Attention 实现
- [ ] 准备权重分片规则

改造完成后，验证：

- [ ] 程序能正常退出
- [ ] 输出 dtype 正确（bf16）
- [ ] 多步推理无 OOM
- [ ] 性能符合预期

---

## 附录：参考资源

- [torchax 官方文档](https://github.com/pytorch/xla)
- [JAX Splash Attention](https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/splash_attention)
- [JAX shard_map](https://jax.readthedocs.io/en/latest/notebooks/shard_map.html)