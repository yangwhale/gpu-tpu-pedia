# YOLO on TPU: torchax vs torch_xla 性能研究

## YOLO 是什么

**YOLO (You Only Look Once)** 是目前最流行的实时目标检测模型家族。它将目标检测任务转化为一个单次回归问题——只需"看一次"图片就能同时预测所有目标的位置和类别，因此得名。

### 核心思想

传统目标检测（如 R-CNN 系列）分两步走：先找候选区域，再分类。YOLO 打破了这个范式：

```
传统方法: 图片 → 候选区域提取 → 每个区域分类 → 合并结果 (慢，多步)
YOLO:     图片 → 整图送入网络  → 一次输出所有检测框 + 类别 (快，单步)
```

### 架构组成

```
Input Image (640x640)
    │
    ▼
┌─────────────────────┐
│  Backbone (特征提取)  │  CSPDarknet / C3k2 模块
│  提取多尺度特征        │  从浅层纹理到深层语义
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Neck (特征融合)      │  FPN + PAN 双向融合
│  连接不同尺度的特征    │  大目标和小目标都能检测
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Head (检测头)        │  Decoupled Head (分离式)
│  输出: bbox + class   │  分别预测边界框和类别
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  NMS (后处理)         │  非极大值抑制
│  过滤重叠检测框       │  保留最佳检测结果
└─────────────────────┘
```

### 版本演进

| 版本 | 年份 | 关键创新 | 作者/团队 |
|------|------|---------|-----------|
| YOLOv1 | 2016 | 开创性的单阶段检测 | Joseph Redmon |
| YOLOv2 | 2017 | Batch Norm、锚框、多尺度训练 | Joseph Redmon |
| YOLOv3 | 2018 | FPN 多尺度检测、Darknet-53 | Joseph Redmon |
| YOLOv4 | 2020 | CSPDarknet、Mosaic 增强、BoF/BoS | Alexey Bochkovskiy |
| YOLOv5 | 2020 | PyTorch 重写、工程化极致 | Ultralytics |
| YOLOv6 | 2022 | 重参数化、TAL 标签分配 | 美团 |
| YOLOv7 | 2022 | E-ELAN、模型缩放策略 | Chien-Yao Wang |
| YOLOv8 | 2023 | Anchor-Free、C2f 模块、Decoupled Head | Ultralytics |
| YOLOv9 | 2024 | GELAN、PGI 可编程梯度信息 | Chien-Yao Wang |
| YOLOv10 | 2024 | NMS-Free、双头架构 | 清华大学 |
| YOLOv11 | 2024 | C3k2、C2PSA 注意力、轻量化 | Ultralytics |
| YOLOv12 | 2025 | Flash Attention 集成、Area Attention | 水牛大学 |

### 应用场景

- **自动驾驶**: 实时检测行人、车辆、交通标志
- **安防监控**: 入侵检测、人员计数、异常行为识别
- **工业质检**: 产品缺陷检测、生产线监控
- **医疗影像**: 细胞检测、病灶定位
- **零售分析**: 商品识别、货架管理

### 模型规格（YOLOv11 系列）

| 模型 | 参数量 | FLOPs | mAP@50-95 | 推理速度 (T4) |
|------|--------|-------|-----------|--------------|
| YOLO11n | 2.6M | 6.5G | 39.5% | 1.5ms |
| YOLO11s | 9.4M | 21.5G | 47.0% | 2.5ms |
| YOLO11m | 20.1M | 68.0G | 51.5% | 4.7ms |
| YOLO11l | 25.3M | 86.9G | 53.4% | 6.2ms |
| YOLO11x | 56.9M | 194.9G | 54.7% | 11.3ms |

---

## 本项目：YOLO on TPU 性能研究

### 研究背景

有用户报告 YOLO 模型使用 `torchax`（torch/xla 的下一代）在 TPU 上推理耗时 **~25,000ms**，而使用 `torch_xla` 仅需 **~20ms**。我们深入研究了性能差异的 root cause，并找到了让 torchax 达到与 torch_xla 同等性能的正确用法。

### 核心发现

| 方案 | 平均耗时 | vs torch_xla | 说明 |
|------|---------|-------------|------|
| **torch_xla** | **8.5ms** | 1x (baseline) | Lazy tensor → XLA graph → 一次执行 |
| **torchax + jax.jit** (优化) | **8.1ms** | **0.95x (更快!)** | 正确编译 → 单一 XLA graph |
| torchax naive | 21,726ms | 2,557x 慢 | 逐 op dispatch，每个 op 独立走 JAX |

### 结论

> torchax 本身不慢。**用法决定性能**。正确使用 `jax.jit` 编译后，torchax 与 torch_xla 性能持平甚至更快。错误的 eager 逐 op 模式才是 25 秒的罪魁祸首。

---

## 目录结构

```
YOLO/
├── README.md                              # 本文件
├── docs/
│   ├── torchax_vs_torch_xla.md            # 核心研究报告：完整性能对比分析
│   └── torchax_optimization_guide.md      # torchax 正确用法指南
├── benchmarks/
│   ├── yolo_torchax_naive.py              # torchax naive 模式 benchmark
│   ├── yolo_torchax_optimized.py          # torchax + jax.jit 优化版
│   ├── yolo_torch_xla.py                 # torch_xla baseline
│   └── run_all_benchmarks.py             # 一键运行全部对比测试
└── examples/
    ├── torchax_correct_usage.py           # YOLO torchax 正确用法完整示例
    └── torchax_common_fixes.py            # 常见兼容性问题修复集
```

## 快速开始

### 环境要求

- TPU v6e (或更新)
- Python 3.12+
- PyTorch 2.9+, JAX 0.8+, torchax 0.0.11+
- ultralytics 8.4+

### 安装

```bash
pip install torch torchax jax jaxlib ultralytics
# torch_xla 需要匹配的 libtpu 版本
pip install 'torch_xla[tpu]' -f https://storage.googleapis.com/libtpu-releases/index.html
```

### 运行 Benchmark

```bash
cd gpu-tpu-pedia/tpu/YOLO/benchmarks
python run_all_benchmarks.py
```

### 查看正确用法

```bash
cd gpu-tpu-pedia/tpu/YOLO/examples
python torchax_correct_usage.py
```

---

## 测试环境

| 项目 | 版本 |
|------|------|
| 硬件 | TPU v6e-8 (8 chips, 2x4 topology) |
| Python | 3.12.12 |
| PyTorch | 2.9.0 |
| JAX | 0.8.1 |
| torchax | 0.0.11 (editable install) |
| torch_xla | 2.9.0 |
| libtpu | 0.0.21 |
| ultralytics | 8.4.18 |
| 测试模型 | YOLO11n (2.6M params) |
| 测试图片 | bus.jpg (640x480) |

## 相关链接

- [Ultralytics YOLO 官方文档](https://docs.ultralytics.com/)
- [torchax GitHub](https://github.com/pytorch/xla/tree/master/torchax)
- [torch_xla 官方文档](https://pytorch.org/xla/)
- [JAX 官方文档](https://jax.readthedocs.io/)
- [YOLO 论文 (原版)](https://arxiv.org/abs/1506.02640)
