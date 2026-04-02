# Real-ESRGAN on TPU with torchax

Real-ESRGAN (RRDBNet, scale=2, num_block=12) 从 GPU L40S 迁移到 TPU v6e 单卡推理。

## 结果摘要

| 平台 | Median | 精度模式 | Max Diff vs CPU |
|------|--------|---------|----------------|
| CPU (PyTorch) | 5,801 ms | FP32 | — (基准) |
| GPU L40S | **291 ms** | FP32 | — |
| TPU v6e (default) | 695 ms | FP32 default | ~0.01 |
| TPU v6e (highest) | 1,650 ms | FP32 highest | **0.000000** |

> 输入 1536x2048 → 输出 3072x4096（2x 超分），8.8M 参数。

### 精度 vs 性能权衡

| 模式 | 延迟 | 精度 | 说明 |
|------|------|------|------|
| `default` | 695ms | max diff ~0.01 | 单 pass bf16 乘法 + fp32 累加 |
| `highest` | 1,650ms | max diff = 0 | 3 pass bf16 模拟完整 FP32 |

精度敏感场景用 `highest`，性能优先用 `default`。

## 架构特点

RRDBNet 比 RepVGG 复杂得多：

- **Dense Block**: 每个 block 内 5 层 conv，每层输入是所有前序输出的 concat（通道逐层递增）
- **RRDB**: 3 个 Dense Block 串联 + 残差缩放（×0.2）
- **Pixel Unshuffle**: scale=2 时，输入先做 pixel_unshuffle（view + permute + reshape）
- **F.interpolate**: 2x nearest 上采样两次
- **LeakyReLU**: 必须改为 `inplace=False`，XLA 不支持 in-place

### 关键修改

1. **inplace=False**: 所有 `LeakyReLU(inplace=True)` 改为 `inplace=False`
2. **独立模型文件**: `rrdbnet.py` 不依赖 basicsr（basicsr 有 torchvision 兼容问题）
3. **顺序**: 模型创建在 `torchax.enable_globally()` 之前

## 文件结构

```
Real-ESRGAN/
├── README.md
├── rrdbnet.py                              # RRDBNet 独立实现（无 basicsr 依赖）
├── examples/
│   ├── torchax_realesrgan_inference.py      # TPU 推理示例
│   └── precision_validation.py             # CPU vs TPU 精度验证
└── benchmarks/
    └── realesrgan_cpu_benchmark.py          # CPU 性能基线
```

## 使用方法

```bash
# TPU 推理（全尺寸）
python examples/torchax_realesrgan_inference.py --input-h 2048 --input-w 1536

# 精度验证（小尺寸，快）
python examples/precision_validation.py --input-h 256 --input-w 192

# CPU 基线
python benchmarks/realesrgan_cpu_benchmark.py --input-h 2048 --input-w 1536 --runs 5
```

## 参考

- [Real-ESRGAN GitHub](https://github.com/xinntao/Real-ESRGAN)
- [RepVGG on TPU](../RepVGG/) — 同系列迁移（简单模型）
- [gpu-tpu-pedia](https://github.com/yangwhale/gpu-tpu-pedia/tree/main/tpu)
