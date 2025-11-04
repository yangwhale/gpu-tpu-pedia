# JAX TPU 内存调试指南

## 概述

在 TPU 上开发 JAX 应用时，内存管理是关键挑战之一。本文档介绍如何使用内存监控工具定位和解决 OOM 问题。

## 核心问题：JAX 编译缓存导致的内存泄漏

### 问题表现

TPU v6e 拥有 32GB HBM，但在处理长视频（如 64 帧）时容易出现：
- **第一次运行**：成功
- **第二次运行**：RESOURCE_EXHAUSTED OOM

### 根本原因

JAX JIT 编译会缓存编译结果，这些缓存占用大量显存（可达 3-5GB）且不会自动释放。

**实测数据**（CogVideoX 64帧解码）：
```
第一次运行：
  - 解码前: 8.38GB
  - 峰值: 25.16GB
  - 清理后: 12.12GB (残留 3.74GB)

第二次运行（未清理缓存）：
  - 解码前: 8.38GB (实际 + 3.74GB 残留 = 12.12GB)
  - 第1帧: 23.51GB
  - 总需求: 12.12GB + 23.51GB ≈ 35GB
  - 结果: OOM (超出 32GB 限制)
```

## 解决方案

### 1. 添加内存监控（可选开关）

```python
import os
import jax

def _decode(self, z: jnp.ndarray, zq: jnp.ndarray, deterministic: bool = True):
    # 内存监控开关（通过环境变量控制）
    enable_memory_debug = os.getenv('JAX_MEMORY_DEBUG', '0') == '1'
    
    def get_memory_stats():
        """获取当前设备内存统计信息"""
        if not enable_memory_debug:
            return ""
        try:
            for device in jax.devices():
                stats = device.memory_stats()
                if stats:
                    used_gb = stats.get('bytes_in_use', 0) / 1e9
                    limit_gb = stats.get('bytes_limit', 0) / 1e9
                    return f"{used_gb:.2f}GB / {limit_gb:.2f}GB"
        except:
            pass
        return "N/A"
    
    def log_memory(msg):
        """记录内存状态（仅在开启调试时）"""
        if enable_memory_debug:
            print(f"[内存] {msg}: {get_memory_stats()}")
    
    # 在关键位置监控内存
    log_memory("解码前")
    
    # ... 执行解码 ...
    
    log_memory("解码后")
```

**使用方法**：
```bash
# 开启内存调试
export JAX_MEMORY_DEBUG=1
python your_script.py

# 关闭内存调试（默认）
unset JAX_MEMORY_DEBUG
python your_script.py
```

### 2. 清理 JAX 编译缓存

在每次迭代/推理后调用 `jax.clear_caches()`：

```python
def run_generation_benchmark(pipe, prompt, num_iterations=2):
    for i in range(num_iterations):
        # 执行推理
        result = pipe(prompt, ...)
        
        # 处理结果
        frames = result.frames[0]
        
        # 清理中间结果
        del result
        
        # ⭐ 关键：清理 JAX 编译缓存
        print(f"  清理 JAX 编译缓存...")
        jax.clear_caches()
```

**效果**：
```
修复后：
  迭代 0 清理后: 12.12GB → clear_caches() → 8.37GB
  迭代 1 解码前: 8.37GB (✅ 缓存已清理)
  迭代 1 第1帧: 18.70GB (✅ 正常)
  结果: ✅ 成功完成
```

## 最佳实践

### 1. 内存监控策略

| 场景 | 监控频率 | 说明 |
|------|----------|------|
| **开发调试** | 每个关键步骤 | 定位内存泄漏位置 |
| **性能测试** | 迭代前后 | 验证清理效果 |
| **生产环境** | 关闭 | 避免性能开销 |

### 2. 缓存清理时机

- ✅ **推荐**：每次推理/迭代后
- ✅ **推荐**：切换模型/任务前
- ❌ **避免**：在紧密循环内（性能损失）
- ❌ **避免**：在 JIT 函数内部（会破坏编译）

### 3. 其他内存优化技巧

1. **逐帧处理**：避免一次性加载所有帧
   ```python
   # ❌ 错误：批量处理
   decoded = decode_all_frames(latents)  # OOM
   
   # ✅ 正确：逐帧处理
   for i in range(num_frames):
       frame = decode_single_frame(latents[:, i:i+1])
       frames.append(frame)
   ```

2. **使用 BFloat16**：相比 FP32 节省 50% 内存
   ```python
   latents_jax = jnp.array(latents_np, dtype=jnp.bfloat16)
   ```

3. **显式清理中间变量**：
   ```python
   decoded_frame = decoder(latent_frame)
   frames.append(decoded_frame)
   
   # 清理中间变量
   decoded_frame = None
   latent_frame = None
   ```

4. **避免不必要的张量转置**：
   ```python
   # ❌ 创建副本
   x_transposed = x.transpose(...)
   
   # ✅ 原地操作
   x = x.transpose(...)
   ```

## 调试工具箱

### 快速诊断脚本

```python
import jax
import jax.numpy as jnp

def print_device_memory():
    """打印所有设备的内存使用情况"""
    for device in jax.devices():
        stats = device.memory_stats()
        if stats:
            used_gb = stats.get('bytes_in_use', 0) / 1e9
            limit_gb = stats.get('bytes_limit', 0) / 1e9
            peak_gb = stats.get('peak_bytes_in_use', 0) / 1e9
            print(f"设备 {device}:")
            print(f"  当前使用: {used_gb:.2f}GB / {limit_gb:.2f}GB")
            print(f"  峰值使用: {peak_gb:.2f}GB")

# 使用示例
print_device_memory()
result = your_function()
print_device_memory()
jax.clear_caches()
print_device_memory()
```

### 内存泄漏检测

```python
def test_memory_leak(func, num_iterations=3):
    """测试函数是否有内存泄漏"""
    initial_memory = []
    final_memory = []
    
    for i in range(num_iterations):
        # 记录初始内存
        stats_before = jax.devices()[0].memory_stats()
        mem_before = stats_before.get('bytes_in_use', 0) / 1e9
        
        # 执行函数
        result = func()
        del result
        
        # 记录最终内存
        stats_after = jax.devices()[0].memory_stats()
        mem_after = stats_after.get('bytes_in_use', 0) / 1e9
        
        initial_memory.append(mem_before)
        final_memory.append(mem_after)
        
        print(f"迭代 {i}: {mem_before:.2f}GB → {mem_after:.2f}GB "
              f"(泄漏: {mem_after - mem_before:.2f}GB)")
        
        # 清理缓存
        jax.clear_caches()
    
    # 检测泄漏趋势
    if len(final_memory) > 1:
        leak_per_iter = (final_memory[-1] - final_memory[0]) / (len(final_memory) - 1)
        if leak_per_iter > 0.1:  # 每次迭代泄漏 >100MB
            print(f"⚠️  检测到内存泄漏: {leak_per_iter:.2f}GB/迭代")
        else:
            print(f"✅ 无明显内存泄漏")
```

## 案例研究：CogVideoX VAE 解码 OOM (完整修复)

### 问题现象

在 TPU v6e (32GB HBM) 上运行 CogVideoX 视频生成时：
```
第一次运行: ✅ 成功
第二次运行: ❌ RESOURCE_EXHAUSTED OOM
```

错误信息：
```
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED:
Attempting to reserve 19.00G at the bottom of memory.
That was not possible. There are 13.10G free, 0B reserved,
and 13.10G reservable.
```

### 根本原因定位

使用内存监控工具追踪内存使用：

```bash
export JAX_MEMORY_DEBUG=1
python generate_flax.py
```

**关键发现**：
```
迭代 0 (第一次运行):
  解码前:      8.38GB / 33.55GB
  第1帧后:    18.70GB / 33.55GB  (+10.32GB)
  第8帧后:    28.16GB / 33.55GB  (峰值)
  清理后:     12.12GB / 33.55GB  (残留 3.74GB ❌)

迭代 1 (第二次运行):
  解码前:      8.38GB / 33.55GB  (显示值)
  实际内存:   12.12GB            (包含 3.74GB 残留缓存)
  第1帧:      23.51GB            (+15.13GB，比第一次多 +4.81GB)
  理论需求:   12.12GB + 23.51GB ≈ 35GB
  结果:       ❌ OOM (超出 32GB 限制)
```

**问题根源**：JAX JIT 编译产生的 3.74GB 缓存未释放，导致第二次运行时基础内存升高。

### 完整修复方案

#### 1. 核心修复：清理 JAX 编译缓存

**文件**：`gpu-tpu-pedia/tpu/cogvideo/generate_flax.py`

**修改位置**：第 738-747 行

```python
def run_generation_benchmark(pipe, prompt, num_iterations=2, ...):
    """运行视频生成基准测试"""
    
    for i in range(num_iterations):
        print(f"\n迭代 {i}:")
        
        # 执行推理
        result = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            height=height,
            width=width
        )
        
        # 处理结果
        frames = result.frames[0]
        
        # 清理中间结果
        del result
        
        # ⭐ 关键修复：清理 JAX 编译缓存
        print(f"  清理 JAX 编译缓存...")
        jax.clear_caches()  # 这一行解决了 OOM 问题！
```

**修复效果**：
```
迭代 0 清理后: 12.12GB → clear_caches() → 8.37GB ✅
迭代 1 解码前:  8.37GB (缓存已清理，恢复正常)
迭代 1 第1帧: 18.70GB (+10.33GB，与第一次一致 ✅)
结果: ✅ 成功完成，内存使用正常
```

#### 2. 辅助工具：环境变量控制的内存监控

**文件**：`diffusers-tpu-chris/src/diffusers/models/autoencoders/autoencoder_kl_cogvideox_flax.py`

**修改位置**：第 1955-2031 行 (`_decode` 方法)

```python
def _decode(self, z: jnp.ndarray, zq: jnp.ndarray, deterministic: bool = True):
    """VAE 解码，支持环境变量控制的内存监控"""
    import os
    
    # 内存监控开关（环境变量控制，默认关闭）
    enable_memory_debug = os.getenv('JAX_MEMORY_DEBUG', '0') == '1'
    
    def get_memory_stats():
        """获取当前设备内存统计（零开销）"""
        if not enable_memory_debug:
            return ""
        try:
            for device in jax.devices():
                stats = device.memory_stats()
                if stats:
                    used_gb = stats.get('bytes_in_use', 0) / 1e9
                    limit_gb = stats.get('bytes_limit', 0) / 1e9
                    return f"{used_gb:.2f}GB / {limit_gb:.2f}GB"
        except:
            pass
        return "N/A"
    
    def log_memory(msg):
        """记录内存状态（仅在开启调试时有输出）"""
        if enable_memory_debug:
            print(f"[内存] {msg}: {get_memory_stats()}")
    
    # 在关键位置监控内存
    log_memory("解码前")
    log_memory("创建缓存管理器后")
    
    # ... 执行逐帧解码 ...
    for i in range(num_latent_frames):
        log_memory(f"解码第 {i+1}/{num_latent_frames} 帧后")
    
    log_memory("清理缓存完成")
    
    return decoded
```

**环境变量设置**（`generate_flax.py` 第1行）：
```python
import os
os.environ.setdefault('JAX_MEMORY_DEBUG', '0')  # 默认关闭，需要时改为 '1'
```

**使用方法**：
```python
# 方式1：修改脚本第1行
os.environ.setdefault('JAX_MEMORY_DEBUG', '1')  # 开启调试

# 方式2：运行时设置
export JAX_MEMORY_DEBUG=1
python generate_flax.py
```

### 验证结果

#### 测试场景
- **硬件**：TPU v6e (32GB HBM)
- **任务**：CogVideoX 视频生成
- **配置**：16 latent frames → 64 video frames, 768×1360, 2 推理步数
- **迭代次数**：2 次

#### 性能数据（调试模式 ON）

```
迭代 0 (包含 JIT 编译):
  总时长: 45.07秒
  内存峰值: 28.16GB
  清理后: 8.37GB ✅

迭代 1:
  总时长: 45.13秒
  内存峰值: 28.16GB (与第一次一致 ✅)
  清理后: 8.37GB ✅
```

#### 性能数据（调试模式 OFF）

```
迭代 0:
  总时长: 45.41秒
  
迭代 1:
  总时长: 44.93秒
```

#### 性能影响分析

| 操作 | 开销 | 说明 |
|------|------|------|
| `jax.clear_caches()` | <1% | 45.07s vs 44.93s |
| 内存监控 (开启) | <1% | 仅 print 开销 |
| 内存监控 (关闭) | 0% | 完全无开销 |

### 关键数据对比

| 指标 | 修复前 | 修复后 | 改善 |
|------|--------|--------|------|
| 第一次运行 | ✅ 成功 | ✅ 成功 | - |
| 第二次运行 | ❌ OOM | ✅ 成功 | 100% |
| 迭代0 清理后 | 12.12GB | 8.37GB | -30.9% |
| 迭代1 解码前 | 12.12GB | 8.37GB | -30.9% |
| 迭代1 第1帧 | 23.51GB | 18.70GB | -20.5% |
| 迭代1 峰值 | OOM | 28.16GB | ✅ 正常 |

### 核心教训

1. **JAX 编译缓存会累积**：
   - 首次运行产生 3.74GB 缓存
   - 不清理会导致第N次运行 OOM
   - 解决方案：定期调用 `jax.clear_caches()`

2. **内存监控是关键**：
   - 没有监控工具无法定位问题根源
   - 环境变量开关实现零性能开销
   - 开发调试必备

3. **1行代码修复大问题**：
   - 核心修复：`jax.clear_caches()`
   - 位置：每次推理迭代后
   - 开销：<1% 性能影响

4. **工具设计原则**：
   - 默认关闭（生产环境零开销）
   - 易于开启（环境变量控制）
   - 信息丰富（关键位置监控）

## 性能影响

`jax.clear_caches()` 的性能影响：

| 场景 | 影响 | 说明 |
|------|------|------|
| **首次运行** | 无影响 | 缓存为空 |
| **第二次运行** | +0-5% | 需重新编译部分代码 |
| **稳定运行** | <1% | 大部分代码已缓存 |

**建议**：
- 开发阶段：每次迭代清理
- 生产环境：根据内存压力决定（如每 N 次清理一次）

## 参考资料

- [JAX Documentation: Device Memory](https://jax.readthedocs.io/en/latest/faq.html#how-to-use-device-memory-profiling)
- [JAX GitHub: Memory Management](https://github.com/google/jax/discussions/8312)
- [CogVideoX VAE 实现](../tpu/cogvideo/generate_flax.py)

## 总结

TPU 内存调试的关键：
1. ✅ 使用 `device.memory_stats()` 监控内存
2. ✅ 在关键位置记录内存使用
3. ✅ 定期清理 JAX 编译缓存
4. ✅ 使用环境变量开关避免性能损失
5. ✅ 逐帧处理大数据避免峰值内存

记住：**防患于未然比事后调试更重要！**