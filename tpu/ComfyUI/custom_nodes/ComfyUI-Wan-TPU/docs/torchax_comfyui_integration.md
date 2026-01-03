# Torchax 与 ComfyUI 集成分析

## 问题背景

在 ComfyUI 中集成 torchax（PyTorch on TPU）时遇到问题：
```
AssertionError: torchax Tensors can only do math within the torchax environment.
Please wrap your code with `with torchax.default_env()` or call torchax.enable_globally() before.
```

本文档分析问题根源并对比两种环境管理方案。

---

## 1. Torchax 环境机制

### 1.1 核心概念

| 概念 | 描述 |
|------|------|
| **Environment** | torchax 的执行环境，包含 `_ops` 算子字典 |
| **Mode 栈** | PyTorch 的 dispatch 机制，拦截所有 tensor 操作 |
| **XLA Tensor** | 使用 JAX 后端的 tensor，必须在 Mode 栈激活时操作 |

### 1.2 两种环境管理方式

#### 方式一：`enable_globally()` (参考代码使用)

```python
import torchax

# 全局启用 - Mode 栈永久激活
torchax.enable_globally()
env = torchax.default_env()

# 后续所有 PyTorch 操作都会被 torchax 拦截
# 需要手动 disable_globally() 才能使用原生 PyTorch
```

**优点：**
- 简单直接
- 所有算子自动被拦截，不会遗漏

**缺点：**
- 影响全局 PyTorch 行为
- 难以与 ComfyUI 的其他节点共存
- 需要手动管理 enable/disable 状态

#### 方式二：`with env:` 上下文管理器 (我们的实现)

```python
import torchax

env = torchax.default_env()

# 只在需要时临时启用 Mode 栈
with env:
    # 这里的 PyTorch 操作被拦截
    result = model(input.to('jax'))
    cpu_result = result.to('cpu')  # 关键：在 with 块内转回 CPU

# with 块外，Mode 栈自动清空
# 可以正常使用 PyTorch
```

**优点：**
- 精确控制作用域
- 与 ComfyUI 其他节点兼容
- 异常安全（`__exit__` 自动清理）

**缺点：**
- 必须确保 XLA tensor 不逃逸出 with 块
- 需要在每个 TPU 操作点包裹 with 块

---

## 2. 当前问题分析

### 2.1 错误发生位置

```
File "/home/chrisya/ComfyUI-TPU/comfy/model_patcher.py", line 857
  self.model.to(device_to)  # 尝试移动模型
```

ComfyUI 的 `model_patcher.py` 试图调用 `.to()` 方法移动模型，但此时：
- Mode 栈未激活（不在 `with env:` 块内）
- 模型权重是 XLA tensor（torchax tensor）
- 调用 `.to()` 触发 dispatch，但没有 Mode 处理

### 2.2 根本原因

我们的实现存在一个关键问题：
**模型权重在 `with env:` 块外仍然是 XLA tensor**

```python
# 我们的代码
with mesh, env:
    move_module_to_xla(env, pipe.transformer)  # 权重变成 XLA tensor
    pipe.transformer = torchax.compile(pipe.transformer)
# with 块结束，但 pipe.transformer 的权重仍然是 XLA tensor！

# 后来 ComfyUI 调用
pipe.transformer.to(device)  # 失败：Mode 栈未激活
```

### 2.3 参考代码为什么能工作

参考代码使用 `enable_globally()`，全程保持 Mode 栈激活：

```python
# 参考代码 (stage2_transformer.py:495-496)
torchax.enable_globally()
env = torchax.default_env()

# 从此以后，所有 PyTorch 操作都被拦截
# 即使在函数外调用 model.to()，也能正常工作
```

---

## 3. 解决方案

### 方案一：回归 `enable_globally()`（推荐短期方案）

在 ComfyUI 节点入口处启用，节点退出时禁用：

```python
class Wan21TPUSampler:
    def sample(self, ...):
        import torchax
        
        # 节点开始时启用
        torchax.enable_globally()
        env = torchax.default_env()
        
        try:
            # ... 执行 TPU 操作 ...
            result = self._run_inference(...)
            
            # 结果转为 CPU numpy/torch，确保不返回 XLA tensor
            return self._to_cpu(result)
        finally:
            # 节点结束时禁用（允许其他节点正常使用 PyTorch）
            torchax.disable_globally()
```

**关键点：**
- 返回值必须是纯 CPU tensor/numpy
- 缓存的模型保持 XLA 状态，下次运行时再启用 globally

### 方案二：完全隔离模型生命周期

确保 XLA 模型只在 `with env:` 块内存在：

```python
def sample(self, ...):
    env = torchax.default_env()
    
    with env:
        # 加载、编译、运行、获取结果，全部在 with 块内
        pipe = self._load_pipeline()  # 返回 XLA 模型
        result = pipe(...)
        cpu_result = result.to('cpu')
        
        # 销毁模型
        del pipe
    
    # with 块外只有 CPU tensor
    return cpu_result
```

**缺点：** 无法缓存模型，每次调用都要重新加载。

### 方案三：Hybrid 方案（推荐长期方案）

结合两种方式：
1. 模型缓存使用 `enable_globally()` 状态机
2. 推理时使用 `with env:` 确保作用域

```python
class Wan21TPUSampler:
    _globally_enabled = False
    
    @classmethod
    def _ensure_env_enabled(cls):
        if not cls._globally_enabled:
            torchax.enable_globally()
            cls._globally_enabled = True
        return torchax.default_env()
    
    def sample(self, ...):
        env = self._ensure_env_enabled()
        
        # 此时 Mode 栈已激活，可以安全操作 XLA tensor
        pipe = self._get_cached_pipeline()
        
        # 仍然使用 with mesh 确保 sharding context
        with self._mesh:
            result = pipe(...)
        
        return self._to_cpu(result)
```

---

## 4. ComfyUI 特殊考虑

### 4.1 ComfyUI 的模型管理

ComfyUI 有自己的模型管理机制：
- `model_patcher.py`: 管理模型加载/卸载
- `model_management.py`: 管理 VRAM/RAM
- 节点返回的模型可能被 ComfyUI 移动/卸载

**影响：** 如果我们返回包含 XLA tensor 的模型，ComfyUI 可能在 Mode 栈外调用 `.to()`。

### 4.2 安全返回值

确保节点返回值不包含 XLA tensor：

| 返回类型 | 安全做法 |
|---------|---------|
| Tensor | 转为 CPU: `result.to('cpu')` 或转为 numpy |
| LATENT dict | 确保 `samples` 是 CPU tensor |
| IMAGE | 转为 numpy float32 |
| 模型对象 | **不要返回**！只缓存在类变量中 |

### 4.3 与其他节点兼容

ComfyUI 可能同时加载多个 custom nodes：
- 使用 `enable_globally()` 会影响其他节点
- 建议：在节点执行期间启用，执行完毕禁用

---

## 5. 推荐实现

基于以上分析，推荐采用 **方案三（Hybrid）**：

```python
# nodes.py 修改

import torchax

_globally_enabled = False

def ensure_torchax_enabled():
    """确保 torchax 全局启用，返回 env"""
    global _globally_enabled
    if not _globally_enabled:
        torchax.enable_globally()
        _globally_enabled = True
    return torchax.default_env()

def disable_torchax():
    """禁用 torchax（节点执行完毕后调用）"""
    global _globally_enabled
    if _globally_enabled:
        torchax.disable_globally()
        _globally_enabled = False


class Wan21TPUSampler:
    def sample(self, ...):
        try:
            env = ensure_torchax_enabled()
            pipe = self._get_cached_pipeline()  # 缓存的 XLA 模型
            
            with self._mesh:
                result = pipe(...)
            
            # 返回前转为 CPU
            return self._to_cpu(result)
        
        finally:
            # 可选：节点执行完毕后禁用
            # 如果下一个节点还是 TPU 节点，会重新启用
            pass  # 或 disable_torchax()
```

---

## 6. 下一步行动

1. **测试方案三**：修改 nodes.py 使用 Hybrid 方案
2. **验证兼容性**：确保与 ComfyUI 其他节点共存
3. **性能测试**：验证模型缓存有效性

## 参考

- [torchax/tensor.py](torchax/tensor.py): Environment 和 Mode 栈实现
- [stage2_transformer.py](gpu-tpu-pedia/tpu/Wan2.1/generate_diffusers_torchax_staged/stage2_transformer.py): 参考实现
- [ComfyUI model_patcher.py](ComfyUI-TPU/comfy/model_patcher.py): ComfyUI 模型管理
