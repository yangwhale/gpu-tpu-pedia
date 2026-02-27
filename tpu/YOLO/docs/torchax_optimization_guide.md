# YOLO torchax 优化指南：从 25 秒到 8 毫秒

本文档记录将 YOLO 模型从 torchax naive 模式（25s）优化到 jax.jit 编译模式（8ms）的完整过程，包括遇到的 5 个兼容性问题及其修复方案。

---

## 1. torchax 的两种执行模式

### 模式 A: Eager Per-Op Dispatch（慢）

```python
torchax.enable_globally()
model.to("jax")
out = model(x)  # 每个 op 独立走 Python → JAX → XLA
```

- 每个 PyTorch op 被拦截，转为 JAX 调用，独立编译执行
- YOLO 有 200+ ops，每个 op 的 dispatch 开销（~2-10ms）累积到秒级
- 不产生 graph fusion，无法利用 XLA 的算子融合优化

### 模式 B: Graph Compilation（快）

```python
torchax.enable_globally()
model.to("jax")
jitted = jax.jit(forward_fn)  # 一次 trace，编译为单一 XLA graph
out = jitted(x)               # 后续调用复用编译结果
```

- `jax.jit` trace 整个 forward，生成一个完整的 JAX 计算图
- XLA 编译器对整个图做算子融合、内存优化、并行调度
- 首次调用较慢（trace + 编译），后续调用极快（~8ms）

### 为什么 `torchax.compile()` 没解决问题？

`torchax.compile(model)` 内部确实会调用 `jax.jit`，但它要求：
1. 所有 model 的 params 和 buffers 都是 torchax Tensor（不是普通 torch.Tensor）
2. forward 内部不能有 data-dependent 的控制流
3. 不能有动态 tensor 创建

YOLO 违反了以上全部三条，所以 `torchax.compile` 直接报错或不生效。

---

## 2. 五个兼容性问题及修复

### 问题 1: conv2d 默认参数缺失

**现象**: `TypeError: _aten_conv2d() missing 2 required positional arguments: 'dilation' and 'groups'`

**原因**: torchax v0.0.11 的 `_aten_conv2d` 函数签名没有默认值，但 PyTorch ATen schema 定义了默认值：

```python
# ATen schema:
# aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None,
#              SymInt[2] stride=[1,1], SymInt[2] padding=[0,0],
#              SymInt[2] dilation=[1,1], SymInt groups=1)

# torchax 原始代码 (有 bug):
def _aten_conv2d(input, weight, bias, stride, padding, dilation, groups):
    ...  # 没有默认值，dispatcher 不传就报错

# 修复:
def _aten_conv2d(input, weight, bias=None, stride=(1,1), padding=(0,0),
                 dilation=(1,1), groups=1):
    ...
```

**修复**: Patch torchax 源码 (`torchax/ops/jaten.py` line 1046-1054)

```python
# 自动 patch 脚本
conv2d_file = "/path/to/torchax/torchax/ops/jaten.py"
with open(conv2d_file, "r") as f:
    content = f.read()

old_sig = """def _aten_conv2d(
  input,
  weight,
  bias,
  stride,
  padding,
  dilation,
  groups,
):"""

new_sig = """def _aten_conv2d(
  input,
  weight,
  bias=None,
  stride=(1, 1),
  padding=(0, 0),
  dilation=(1, 1),
  groups=1,
):"""

content = content.replace(old_sig, new_sig)
with open(conv2d_file, "w") as f:
    f.write(content)
```

---

### 问题 2: BN buffers 未转到 JAX device

**现象**: `AssertionError: <class 'torch.Tensor'>` (在 `torchax/interop.py` line 202)

**原因**: `model.to("jax")` 移动了 parameters，但 BatchNorm 的 `running_mean`、`running_var`、`num_batches_tracked` 作为 registered buffers 可能未正确转换。`torchax.compile` 内部的 `JittableModule` 对 `self.buffers` 调用 `jax_view()` 时遇到普通 `torch.Tensor`。

**修复方案 A (推荐)**: Fuse BN into Conv，消除 BN buffers

```python
model.model.eval()
model.model.fuse()  # Conv + BN → ConvBN，BN buffers 不再存在
```

**修复方案 B**: 手动转换所有 buffers

```python
for name, buf in model.model.named_buffers():
    parts = name.split(".")
    obj = model.model
    for part in parts[:-1]:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    setattr(obj, parts[-1], buf.to("jax"))
```

---

### 问题 3: make_anchors 动态 tensor 创建

**现象**: JIT trace 失败或每次调用都重新编译

**原因**: YOLO Detect head 的 `_get_decode_boxes` 中：

```python
def _get_decode_boxes(self, x):
    shape = x["feats"][0].shape
    if self.dynamic or self.shape != shape:  # ← shape 比较在 JIT 内失效
        self.anchors, self.strides = make_anchors(...)  # ← 动态创建 tensor
        self.shape = shape  # ← side effect，JIT 不支持
    ...
```

- `self.shape != shape`: 在 `jax.jit` trace 中，shape 是抽象值，比较结果不确定
- `make_anchors()`: 内部用 `torch.arange`、`torch.meshgrid` 创建 tensor，导致 ConcretizationTypeError
- `self.shape = shape`: 修改 `self` 属性是 side effect

**修复**: 预计算 + monkey-patch

```python
# 1. 先跑一次 forward 触发 anchor 计算
with torch.no_grad():
    _ = model.model(img_jax)

# 2. 缓存 anchors
detect = model.model.model[-1]
cached_anchors = detect.anchors
cached_strides = detect.strides

# 3. Patch 掉动态检查
def patched_get_decode_boxes(self, x):
    """直接使用缓存的 anchors，跳过 shape 检查"""
    dbox = self.decode_bboxes(
        self.dfl(x["boxes"]),
        self.anchors.unsqueeze(0)
    ) * self.strides
    return dbox

detect._get_decode_boxes = types.MethodType(patched_get_decode_boxes, detect)
```

> **注意**: 这要求输入图片尺寸固定。如果尺寸变化，需要为每个尺寸预计算 anchors。

---

### 问题 4: torch.arange 参数传递

**现象**: torchax 内部的 `torch.arange` 调用方式与 PyTorch 标准不兼容

**原因**: 某些版本的 torchax 对 `torch.arange(end=N)` 的 keyword-only 调用处理不当

**修复**: Monkey-patch `torch.arange`

```python
_orig_arange = torch.arange
def _patched_arange(*args, **kwargs):
    if "end" in kwargs and "start" not in kwargs and len(args) == 0:
        end = kwargs.pop("end")
        return _orig_arange(0, end, **kwargs)
    return _orig_arange(*args, **kwargs)
torch.arange = _patched_arange
```

---

### 问题 5: torch_xla 与 JAX 的 PJRT/libtpu 版本冲突

**现象**: `RuntimeError: Unexpected PJRT_ExecuteOptions size: expected 112, got 80`

**原因**: `torch_xla` 和 `jax` 依赖不同版本的 `libtpu`。pip 安装的 torch_xla 2.9.0 需要特定的 PJRT API 版本，与系统上 JAX 的 libtpu 不匹配。

**修复**: 使用 `torch_xla[tpu]` 安装匹配版本

```bash
# 安装匹配的 libtpu
pip install 'torch_xla[tpu]' -f https://storage.googleapis.com/libtpu-releases/index.html
```

**注意**: 同一个进程中不能同时使用 torch_xla 和 torchax，因为它们争夺 TPU 设备访问。在 benchmark 中使用 subprocess 隔离。

---

## 3. 完整优化步骤

### Step-by-step 迁移流程

```python
import torch
import torchax
from torchax import interop
import jax
import types

# ═══════════════════════════════════════════
# Step 1: Patch 已知兼容性问题
# ═══════════════════════════════════════════
_orig_arange = torch.arange
def _patched_arange(*args, **kwargs):
    if "end" in kwargs and "start" not in kwargs and len(args) == 0:
        return _orig_arange(0, kwargs.pop("end"), **kwargs)
    return _orig_arange(*args, **kwargs)
torch.arange = _patched_arange

# ═══════════════════════════════════════════
# Step 2: 启用 torchax，准备环境
# ═══════════════════════════════════════════
torchax.enable_globally()
env = torchax.default_env()

# ═══════════════════════════════════════════
# Step 3: 加载模型，Fuse BN，移到 JAX
# ═══════════════════════════════════════════
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.model.eval()
model.model.fuse()        # Conv+BN 融合，消除 BN buffers
model.model.to("jax")     # 移到 JAX device

# ═══════════════════════════════════════════
# Step 4: 准备输入（固定尺寸）
# ═══════════════════════════════════════════
from ultralytics.data.augment import LetterBox
import numpy as np
from PIL import Image

img_np = np.array(Image.open("bus.jpg"))
letterbox = LetterBox((640, 640), auto=True, stride=32)
img_lb = letterbox(image=img_np)
img_t = torch.from_numpy(img_lb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
img_jax = img_t.to("jax")

# ═══════════════════════════════════════════
# Step 5: 预计算 anchors，Patch Detect head
# ═══════════════════════════════════════════
detect = model.model.model[-1]

with torch.no_grad():
    _ = model.model(img_jax)  # 触发 anchor 计算

def patched_get_decode_boxes(self, x):
    dbox = self.decode_bboxes(
        self.dfl(x["boxes"]),
        self.anchors.unsqueeze(0)
    ) * self.strides
    return dbox
detect._get_decode_boxes = types.MethodType(patched_get_decode_boxes, detect)

# ═══════════════════════════════════════════
# Step 6: 用 jax.jit 包装 forward
# ═══════════════════════════════════════════
def forward_fn(img_jax_array):
    with env:
        img_torchax = torchax.tensor.Tensor(img_jax_array, env=env)
        with torch.no_grad():
            out = model.model(img_torchax)
        if isinstance(out, (list, tuple)):
            return interop.jax_view(out[0])
        return interop.jax_view(out)

jitted_forward = jax.jit(forward_fn)
img_jax_arr = interop.jax_view(img_jax)

# ═══════════════════════════════════════════
# Step 7: Warmup (首次触发编译)
# ═══════════════════════════════════════════
out = jitted_forward(img_jax_arr)
jax.block_until_ready(out)

# ═══════════════════════════════════════════
# Step 8: 推理
# ═══════════════════════════════════════════
import time
t0 = time.perf_counter()
out = jitted_forward(img_jax_arr)
jax.block_until_ready(out)
print(f"Inference: {(time.perf_counter() - t0) * 1000:.1f}ms")  # ~8ms
```

---

## 4. 性能优化 Checklist

在将任何 PyTorch 模型迁移到 torchax 时，检查以下项目：

- [ ] **BN Fuse**: 调用 `model.fuse()` 消除 BatchNorm buffers
- [ ] **固定输入尺寸**: 确保输入 shape 不变，避免 JIT 重编译
- [ ] **预计算动态 tensor**: 找到 forward 中的 `torch.arange`、`torch.meshgrid`、`torch.linspace` 等，预先计算并缓存
- [ ] **Patch 条件分支**: 找到依赖 tensor 值的 `if` 语句，替换为静态逻辑
- [ ] **Patch side effects**: 找到 `self.xxx = ...` 的赋值（修改模型状态），移除或缓存
- [ ] **检查 op 覆盖**: 运行一次看是否有未实现的 op（会报 `NotImplementedError`）
- [ ] **buffer 转换**: 确认所有 `named_buffers()` 都在 JAX device 上
- [ ] **计时准确**: 使用 `jax.block_until_ready()` 或 `jax.effects_barrier()` 同步后再取时间
- [ ] **Warmup**: 至少 warmup 2 次再开始计时（首次 = trace + compile）

---

## 5. 参考资料

本文的优化方法参考了以下 torchax 实战经验：

| 参考项目 | 路径 | 关键技巧 |
|---------|------|---------|
| HunyuanVideo torchax 迁移指南 | `tpu/HunyuanVideo-1.5/.../TORCHAX_MIGRATION_GUIDE.md` | 预计算 rotary embedding、环境管理 |
| Wan2.1 Transformer 推理 | `tpu/Wan2.1/.../stage2_transformer.py` | CompileOptions、sharding、step callback |
| ComfyUI torchax 集成分析 | `tpu/ComfyUI/.../torchax_comfyui_integration.md` | 三种环境模式、enable/disable 切换 |
| torchax vs flax 性能分析 | `tpu/Wan2.1/.../torchax_vs_flax_vs_jax_analysis.md` | dispatcher 机制、编译缓存 |
