# TPU 图像处理示例

本示例展示如何将PyTorch的图像处理代码从GPU迁移到TPU，使用torchax框架实现高性能计算。

## 📁 文件结构

```
image_processing/
├── README.md                          # 本文档
├── requirements.txt                   # Python依赖
│
├── GPU版本 (原始代码)
│   ├── image_process_test.py         # 图像处理测试 (crop/resize/blur)
│   ├── resize_test.py                # resize精度对比测试
│   └── gpu_b200_test_results.log     # GPU测试结果
│
├── TPU版本 (改写后的代码)
│   ├── image_process_test_tpu.py     # TPU版图像处理测试
│   ├── resize_test_tpu.py            # TPU版resize精度对比
│   ├── image_process_test_tpu.log    # TPU测试结果
│   └── resize_test_tpu.log           # TPU resize测试结果
│
└── test_set/                          # 测试数据
    ├── video/frame_0000.png          # 测试图像
    └── mask/frame_0000.png           # 测试mask
```

## 🎯 项目目的

1. **验证图像处理算法在TPU上的精度** - 对比torchvision和OpenCV的resize差异
2. **测试TPU上的张量运算** - crop、resize、gaussian blur等操作
3. **展示TPU性能优化** - tracing缓存机制带来的加速效果
4. **提供GPU到TPU的迁移示例** - 实际可运行的代码参考

## 🔄 GPU vs TPU 关键代码对比

### 1. 设备迁移
```python
# GPU版本
tensor = tensor.cuda()

# TPU版本
tensor = tensor.to('jax')
```

### 2. 计算同步
```python
# GPU版本
torch.cuda.synchronize()

# TPU版本
torchax.interop.call_jax(jax.block_until_ready, tensor)
```

### 3. 环境设置
```python
# TPU版本需要额外的环境配置
import torchax
from jax.sharding import Mesh
from jax.experimental import mesh_utils

# 创建设备网格
mesh_devices = mesh_utils.create_device_mesh((num_devices,))
mesh = Mesh(mesh_devices, ('devices',))

# 创建torchax环境
env = torchax.default_env()
env._mesh = mesh

# 在上下文中执行所有计算
with env, mesh:
    # 所有操作在TPU上执行
    result = model(input)
```

## 📊 性能与精度对比

### Resize操作性能

| 平台 | 第1次运行(含tracing) | 后续运行平均 | 加速比 |
|------|---------------------|--------------|--------|
| **GPU B200** | - | 0.0004秒 | - |
| **TPU v6e** | 0.5847秒 | 0.0282秒 | 20.75x |

**关键发现**：
- TPU第1次运行需要tracing时间（扫描PyTorch代码并确定是否需要编译）
- 后续运行使用编译缓存，速度提升20倍以上
- TPU单次运行时间约为GPU的70倍，但对于批量处理仍有优势

### 精度对比 (float32)

| 指标 | GPU B200 | TPU v6e |
|------|----------|---------|
| **Maximum difference** | 0.0078 | 0.0078 |
| **Median difference** | 0.0008 | 0.0008 |
| **精度结论** | ✅ 完全一致 | ✅ 完全一致 |

### 精度警告 (bfloat16)

如果使用`torch.set_default_dtype(torch.bfloat16)`以优化TPU性能：
- **Maximum difference**: 从0.0078上升到**1.7109**（增加218倍）
- **原因**: bfloat16只有7位有效数字，float32有24位
- **建议**: 精度敏感场景使用float32，性能优先场景可用bfloat16

## 🚀 快速开始

### 环境准备

```bash
# 创建conda环境
conda create -n torchax python=3.12
conda activate torchax

# 安装依赖
pip install -r requirements.txt
```

### 运行GPU版本

```bash
# 图像处理测试
python image_process_test.py

# resize精度对比
python resize_test.py
```

### 运行TPU版本

```bash
# resize精度对比（运行5次观察tracing和缓存效果）
python resize_test_tpu.py

# 图像处理测试（运行5次）
python image_process_test_tpu.py
```

## 🔧 技术要点

### 1. 保证在TPU上执行

**必要条件**：
1. 创建torchax环境并设置mesh
2. 在`with env, mesh:`上下文中执行
3. 使用`.to('jax')`移动数据到TPU

```python
env = torchax.default_env()
env._mesh = mesh

with env, mesh:
    # 这里的所有操作都在TPU上执行
    data = data.to('jax')
    result = process(data)
```

### 2. 兼容性处理

**已知问题**：
- ❌ `torchvision.io.decode_image` 与torchax不兼容
- ✅ 使用PIL加载图片，然后转为torch tensor再移动到TPU

```python
# 避免直接在JAX环境中使用torchvision.io
frames = [np.array(Image.open(f)) for f in files]
frames = torch.stack([torch.from_numpy(f) for f in frames])
frames = frames.to('jax')  # 移动到TPU
```

### 3. 警告过滤

```python
import warnings

# 过滤JAX的dtype转换警告
warnings.filterwarnings('ignore', message='.*Explicitly requested dtype int64.*')
# 过滤NumPy只读数组警告
warnings.filterwarnings('ignore', message='.*NumPy array is not writable.*')
```

### 4. 多次运行的意义

代码运行5次的原因：
1. **第1次**：包含代码tracing和编译缓存扫描时间（最慢）
2. **第2-5次**：使用编译缓存，展示真实性能（快20倍）

这不是编译时间，而是torchax扫描PyTorch代码并确定是否需要编译的时间。

## 📝 代码说明

### image_process_test_tpu.py

测试以下图像处理操作：
- **Crop**: 基于mask的智能裁剪（包含padding、最小尺寸、宽高比调整）
- **Resize**: torchvision的resize操作
- **Gaussian Blur**: 高斯模糊处理

### resize_test_tpu.py

对比torchvision和OpenCV的resize精度差异：
- 使用PIL加载图片
- 分别用torchvision和OpenCV进行resize
- 计算像素级差异统计

## ⚠️ 注意事项

1. **数据类型选择**
   - `float32`: 高精度，与GPU结果一致
   - `bfloat16`: TPU优化，性能更好但精度降低

2. **第一次运行时间**
   - 包含tracing时间，比后续运行慢很多
   - 这是正常现象，后续运行会快很多

3. **兼容性限制**
   - 某些torchvision功能可能不支持
   - 建议先在CPU上处理数据，再移动到TPU

## 📚 参考资料

- [TorchAX 官方文档](https://github.com/pytorch/torchax)
- [JAX 官方文档](https://jax.readthedocs.io/)
- [TPU 最佳实践](https://cloud.google.com/tpu/docs/best-practices)

## 🤝 贡献

欢迎提交Issue和Pull Request改进这个示例！

## 📄 许可证

MIT License