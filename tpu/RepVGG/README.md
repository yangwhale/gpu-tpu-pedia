# RepVGG on TPU with torchax

RepVGG (Making VGG-style ConvNets Great Again) 从 GPU L40S 迁移到 TPU v6e 单卡推理。

## 结果摘要

| 指标 | GPU L40S | TPU v6e (torchax) |
|------|----------|-------------------|
| 推理延迟 | 2.93 ms | **0.22 ms** |
| 精度模式 | FP32 | FP32 (highest) |
| Max diff vs CPU | — | 0.000001 |
| 首次编译 | — | ~2.7s |

> TPU 推理延迟比 GPU L40S 快 **13x**，精度与 CPU FP32 完全一致。

## 精度策略

业务要求**必须保持 FP32 精度**，不允许降到 BF16。

TPU MXU 硬件路径是 BF16 乘法 + FP32 累加。通过设置 `jax_default_matmul_precision = "highest"`，JAX 使用多个 BF16 pass 算术模拟 IEEE 754 FP32，精度损失趋近于零：

| 精度模式 | Max Diff | Median Diff | 说明 |
|---------|----------|-------------|------|
| default (single bf16) | 0.0127 | 0.0020 | 22 层 conv 误差累积 |
| **highest (multi-pass)** | **0.000001** | **0.000000** | 几乎完全一致 |

关键代码：
```python
jax.config.update("jax_default_matmul_precision", "highest")
```

## 迁移方案

RepVGG 是 TPU 迁移最简单的模型之一，因为 deploy 模式下只有：
- `Conv2d 3x3 + ReLU` 堆叠（标准 XLA 算子）
- 无 BatchNorm（已融合进 conv 权重）
- 无分支、无残差、无动态 shape

### 迁移步骤

```
1. 创建模型 + switch_to_deploy()  ←  必须在 torchax.enable_globally() 之前
2. 初始化 torchax + 设置 highest precision
3. model.to("jax")
4. jax.jit(forward_fn)  ←  关键：必须 JIT 编译，不能 eager
```

### 踩坑记录

1. **deepcopy 失败**：`repvgg_model_convert()` 内部用 `copy.deepcopy()`，如果在 `torchax.enable_globally()` 之后调用会报 `OperatorNotFound: aten::set_.source_Storage`。解决：先创建模型，再启用 torchax。

2. **conv2d 默认参数**：torchax v0.0.11 的 `_aten_conv2d` 缺少默认值，需要 patch（见 `examples/torchax_common_fixes.py`）。

3. **必须 JIT**：torchax eager 模式逐 op dispatch，会慢 2500x+。RepVGG 22 层 conv，必须用 `jax.jit` 编译为单个 XLA graph。

## 文件结构

```
RepVGG/
├── README.md                          # 本文档
├── repvgg.py                          # RepVGG 模型定义（来自官方 repo）
├── se_block.py                        # SE Block（仅 D2se 变体使用）
├── examples/
│   ├── torchax_repvgg_inference.py    # 完整推理示例（5 步流程）
│   └── precision_validation.py        # CPU vs TPU 精度验证
├── benchmarks/
│   └── repvgg_torchax_benchmark.py    # 性能 benchmark（输出 JSON）
└── docs/                              # 补充文档
```

## 使用方法

### 推理

```bash
# 基本推理（随机权重）
python examples/torchax_repvgg_inference.py --variant RepVGG-A0 --input-size 128

# 加载预训练权重
python examples/torchax_repvgg_inference.py --variant RepVGG-A0 --weights RepVGG-A0-deploy.pth

# 不同变体
python examples/torchax_repvgg_inference.py --variant RepVGG-B0 --input-size 224
```

### 精度验证

```bash
python examples/precision_validation.py --variant RepVGG-A0 --input-size 128 --runs 10
```

### Benchmark

```bash
python benchmarks/repvgg_torchax_benchmark.py --variant RepVGG-A0 --input-size 128 --runs 100

# 保存结果为 JSON
python benchmarks/repvgg_torchax_benchmark.py --json-output results.json
```

## 模型变体

| Variant | Params | Blocks | 说明 |
|---------|--------|--------|------|
| RepVGG-A0 | 8.3M | [2,4,14,1] | 最小，适合快速验证 |
| RepVGG-A1 | 12.8M | [2,4,14,1] | |
| RepVGG-A2 | 25.5M | [2,4,14,1] | |
| RepVGG-B0 | 14.3M | [4,6,16,1] | |
| RepVGG-B1 | 51.8M | [4,6,16,1] | |
| RepVGG-B2 | 80.3M | [4,6,16,1] | |
| RepVGG-B3 | 110.9M | [4,6,16,1] | 最大 |

## 参考

- [RepVGG Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.pdf)
- [RepVGG GitHub](https://github.com/DingXiaoH/RepVGG)
- [torchax](https://github.com/google/torchax)
- [YOLO on TPU](../YOLO/) — 类似的 Conv 密集模型迁移案例
