# torchax vs torch_xla 性能对比研究报告

## 1. 研究背景

### 问题描述

用户报告 YOLO 模型使用 torchax 在 TPU 上推理，640x480 图片耗时约 **25,000ms**。相同模型使用 `torch_xla` 的 `device()` 方式仅需 **~20ms**。差距达 **1,250x**。

### 原始代码

```python
from ultralytics import YOLO
model = YOLO("yolo26n.pt")
import torch
import torchax

_orig_arange = torch.arange
def _patched_arange(*args, **kwargs):
    if "end" in kwargs and "start" not in kwargs and len(args) == 0:
        end = kwargs.pop("end")
        return _orig_arange(0, end, **kwargs)
    return _orig_arange(*args, **kwargs)
torch.arange = _patched_arange

torchax.enable_globally()
model.to("jax")
model = torchax.compile(model)

_ = model.predict("bus.jpg")  # ~25,000ms
```

### 核心疑问

torchax 是 torch/xla 的下一代，理论上应该性能更好。为什么反而慢了 1000 倍以上？

---

## 2. 测试环境

| 项目 | 版本 |
|------|------|
| 硬件 | TPU v6e-8 (8 chips, 2x4 topology) |
| Python | 3.12.12 |
| PyTorch | 2.9.0 |
| JAX | 0.8.1 |
| jaxlib | 0.8.1 |
| torchax | 0.0.11 (editable install from /home/chrisya/torchax) |
| torch_xla | 2.9.0 |
| libtpu | 0.0.21 (via `torch_xla[tpu]`) |
| ultralytics | 8.4.18 |
| 测试模型 | YOLO11n (2.6M params, 100 layers fused) |
| 测试图片 | bus.jpg (640x480, letterbox to 640x640) |

---

## 3. 实测数据

### 3.1 YOLO Forward Pass 完整对比

| Config | Avg (ms) | Min | Max | vs torch_xla |
|--------|----------|-----|-----|--------------|
| **torch_xla (baseline)** | **8.5** | 8.1 | 8.8 | 1x |
| **torchax + jax.jit (优化)** | **8.1** | 8.0 | 8.1 | **0.95x (更快)** |
| torchax naive | 21,726 | 21,212 | 22,033 | **2,557x** |

### 3.2 逐 Run 数据

| Run | torch_xla | torchax naive | torchax + jax.jit |
|-----|-----------|---------------|-------------------|
| 1 | 8.6 ms | 21,212 ms | 8.1 ms |
| 2 | 8.6 ms | 21,468 ms | 8.1 ms |
| 3 | 8.1 ms | 21,581 ms | 8.1 ms |
| 4 | 8.6 ms | 21,620 ms | 8.1 ms |
| 5 | 8.5 ms | 21,732 ms | 8.0 ms |
| 6 | 8.5 ms | 21,766 ms | 8.1 ms |
| 7 | 8.4 ms | 21,886 ms | 8.1 ms |
| 8 | 8.6 ms | 21,944 ms | 8.0 ms |
| 9 | 8.2 ms | 22,017 ms | 8.1 ms |
| 10 | 8.8 ms | 22,033 ms | 8.1 ms |

**观察**: torchax naive 每次推理逐渐变慢（21,212→22,033），存在累积开销。

### 3.3 Op Chain 微基准测试

单独验证 per-op dispatch 开销的影响：

| 测试 | torch_xla | torchax | JAX jit | torchax/xla |
|------|-----------|---------|---------|-------------|
| Single matmul (1024x1024) | 0.06ms | 0.16ms | 0.12ms | **3x** |
| 4-op chain (mm→relu→add→softmax) | 0.06ms | 8.46ms | 0.13ms | **148x** |
| 200-op chain (模拟 YOLO 规模) | 0.65ms | 49.25ms | N/A | **76x** |

**关键发现**: 单个 op 差距不大（3x），但随着 op 数量增加，差距呈非线性增长。

---

## 4. Root Cause 分析

### 4.1 两种执行模式对比

```
torch_xla (Lazy Tensor):
  op1 → 记录
  op2 → 记录
  op3 → 记录
  ...
  op200 → 记录
  mark_step() → 编译为 1 个 XLA HLO Graph → 一次执行 → 8ms

torchax naive (Eager Per-Op):
  op1 → Python dispatch → JAX → XLA compile → execute → 返回
  op2 → Python dispatch → JAX → XLA compile → execute → 返回
  op3 → Python dispatch → JAX → XLA compile → execute → 返回
  ...
  op200 → Python dispatch → JAX → XLA compile → execute → 返回
  总计: 200 次 Python 往返 → 22,000ms

torchax + jax.jit (Graph Compilation):
  jax.jit 追踪:
    op1 → 记录 JAX trace
    op2 → 记录 JAX trace
    ...
    op200 → 记录 JAX trace
  → 编译为 1 个 XLA HLO Graph → 一次执行 → 8ms
```

### 4.2 为什么 naive 模式下 `torchax.compile` 没生效？

原始代码调用了 `torchax.compile(model)`，理论上应该触发 JIT 编译。但实际没生效的原因：

1. **compile 的是 YOLO wrapper，不是内部模型**
   - `torchax.compile(model)` 编译了 `YOLO` 对象
   - 但 `model.predict()` 内部有大量预处理/后处理逻辑不在 compile 范围内

2. **YOLO Detect Head 的动态 tensor 创建**
   ```python
   # ultralytics/nn/modules/head.py - _get_decode_boxes
   if self.dynamic or self.shape != shape:
       self.anchors, self.strides = make_anchors(x["feats"], self.stride, 0.5)
       self.shape = shape
   ```
   - `self.shape != shape` 在 JIT trace 内会导致 ConcretizationTypeError 或每次重新 trace
   - `self.shape = shape` 是 side effect，`jax.jit` 不支持

3. **BN buffers 未转换为 JAX tensor**
   - YOLO 有大量 BatchNorm 层，其 `running_mean/running_var` 作为 registered buffers
   - `model.to("jax")` 移动了 parameters 但部分 buffers 留在 CPU
   - `torchax.compile` 的 `JittableModule` 调用 `jax_view(self.buffers)` 时遇到普通 `torch.Tensor`，AssertionError

### 4.3 YOLO 的 `predict()` 完整流程

```
model.predict("bus.jpg")
    │
    ├── 1. Image Loading (PIL/OpenCV)          ← CPU，与 torchax 无关
    ├── 2. LetterBox Resize (640x640)           ← 包含 torch tensor ops
    ├── 3. Normalize (/ 255.0)                  ← torch op
    ├── 4. model.model.warmup()                 ← 额外 forward pass
    ├── 5. Forward Pass                         ← 200+ ops，核心计算
    │       ├── Backbone (CSP, Conv, BN)
    │       ├── Neck (FPN, PAN, Upsample)
    │       └── Head (Detect)
    │             ├── forward_head (box + cls)
    │             ├── make_anchors (动态!)
    │             └── decode_bboxes
    ├── 6. NMS Postprocess                      ← dynamic shapes!
    └── 7. Result formatting                    ← CPU
```

在 `enable_globally()` 模式下，步骤 2-6 的每个 tensor op 都经过 torchax dispatch。

---

## 5. 解决方案

### 正确的 torchax 使用模式

```python
import torch, torchax, jax
from torchax import interop

# 1. 启用全局转换
torchax.enable_globally()
env = torchax.default_env()

# 2. 加载模型，fuse BN
model = YOLO("yolo11n.pt")
model.model.eval()
model.model.fuse()       # 关键：消除 BN buffers
model.model.to("jax")

# 3. 预计算动态 tensor（anchors/strides）
with torch.no_grad():
    _ = model.model(img_jax)  # 触发 anchor 计算

# 4. Patch 掉动态创建逻辑
detect = model.model.model[-1]
def patched_get_decode_boxes(self, x):
    dbox = self.decode_bboxes(self.dfl(x["boxes"]),
                              self.anchors.unsqueeze(0)) * self.strides
    return dbox
detect._get_decode_boxes = types.MethodType(patched_get_decode_boxes, detect)

# 5. 用 jax.jit 包装 forward
def forward_fn(img_jax_array):
    with env:
        img_t = torchax.tensor.Tensor(img_jax_array, env=env)
        with torch.no_grad():
            out = model.model(img_t)
        return interop.jax_view(out[0] if isinstance(out, tuple) else out)

jitted = jax.jit(forward_fn)

# 6. 推理
out = jitted(img_arr)
jax.block_until_ready(out)  # 8ms
```

详见 [torchax_optimization_guide.md](torchax_optimization_guide.md) 和 [examples/torchax_correct_usage.py](../examples/torchax_correct_usage.py)。

---

## 6. 结论

### torchax 不慢，用法决定性能

| 用法 | 原理 | 性能 |
|------|------|------|
| **错误**: `enable_globally()` + 直接调用 | 每个 op 独立 dispatch | 22,000ms |
| **正确**: `jax.jit` 包装整个 forward | 编译为单一 XLA graph | 8ms |
| **对照**: torch_xla lazy tensor | 编译为单一 XLA graph | 8.5ms |

### 关键教训

1. `torchax.enable_globally()` **不等于** 编译优化，它只是开启了 PyTorch→JAX 的 op 转换通道
2. 性能优化的关键在于 **graph compilation**（通过 `jax.jit` 或 `torchax.compile`）
3. 动态 tensor 创建、shape-dependent 条件判断是 JIT 编译的天敌，必须预计算或 patch 掉
4. torchax 正确使用后，性能与 torch_xla **持平甚至更快**（8.1ms vs 8.5ms）
