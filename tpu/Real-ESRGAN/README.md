# Real-ESRGAN on TPU with torchax

Real-ESRGAN (RRDBNet, scale=2, num_block=12) 从 GPU L40S 迁移到 TPU v6e 单卡推理。

## 结果摘要

| 平台 | Median | 精度模式 | Max Diff vs CPU | vs GPU |
|------|--------|---------|----------------|--------|
| CPU (PyTorch) | 5,801 ms | FP32 | — (基准) | — |
| GPU L40S | **291 ms** | FP32 | — | 1.0x |
| TPU whole-image (default) | 451 ms | FP32 default | ~0.01 | 0.6x |
| TPU whole-image (highest) | 1,454 ms | FP32 highest | **0.000000** | 0.2x |
| **TPU tile-128 (default)** | **196 ms** | FP32 default | ~0.01 | **1.5x** |
| **TPU tile-64 (default)** | **155 ms** | FP32 default | ~0.01 | **1.9x** |
| TPU tile-128 (highest) | 618 ms | FP32 highest | 0.000000 | 0.5x |

> 输入 1536x2048 → 输出 3072x4096（2x 超分），8.8M 参数。

### Tile-based 推理优化

把全尺寸图片切成小 tile，batch 推理后拼回去。

| Tile Size | Tiles | Halo | Forward | Total | Speedup vs whole |
|-----------|-------|------|---------|-------|-----------------|
| 512×512 | 12 | 0 | 353ms | 396ms | 1.1x |
| 256×256 | 48 | 0 | 264ms | 321ms | 1.4x |
| 128×128 | 192 | 0 | 156ms | 196ms | 2.3x |
| **64×64** | **768** | **0** | **109ms** | **155ms** | **2.9x** |

- **最快：tile=64**（155ms，比 GPU 快 1.9x），适合不需要 halo overlap 的场景
- **稳健：tile=128**（196ms，比 GPU 快 1.5x），加 halo 后 tile 数增长更温和

### 精度 vs 性能权衡

| 模式 | Tile-128 延迟 | 精度 | 说明 |
|------|--------------|------|------|
| `default` | 155ms (tile-64) | max diff ~0.01 | 单 pass bf16 乘法 + fp32 累加 |
| `highest` | 618ms (tile-128) | max diff = 0 | 3 pass bf16 模拟完整 FP32 |

精度敏感场景用 `highest`，性能优先用 `default`。

## 关键优化

### 1. F.interpolate 替换
torchax 的 `functional_interpolate` 只实现了 bicubic，nearest 模式会 raise `OperatorNotFound`。
用纯 tensor 操作 `repeat_interleave` 替代：

```python
def nearest_upsample_2x(x):
    return x.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
```

这个修复不仅解了 bug，还让 whole-image 从 695ms 降到 451ms（35% 加速）。

### 2. Tile-based Batch 推理
- 切成固定 tile size，batch 一次推理
- 可选 halo overlap 消除拼接缝
- XLA 编译时间从 68s（whole-image）降到 12s（tile）

### 3. inplace=False
所有 `LeakyReLU(inplace=True)` 改为 `inplace=False`，XLA 不支持 in-place。

### 4. 独立模型文件
`rrdbnet.py` 不依赖 basicsr（basicsr 有 torchvision 兼容问题）。

### 5. 模型创建顺序
模型必须在 `torchax.enable_globally()` 之前创建，否则 deepcopy 失败。

## 架构特点

RRDBNet 比 RepVGG 复杂得多：

- **Dense Block**: 每个 block 内 5 层 conv，每层输入是所有前序输出的 concat（通道逐层递增）
- **RRDB**: 3 个 Dense Block 串联 + 残差缩放（×0.2）
- **Pixel Unshuffle**: scale=2 时，输入先做 pixel_unshuffle（view + permute + reshape）
- **Nearest upsample**: 2x nearest 上采样两次（用 repeat_interleave 实现）
- **LeakyReLU**: 必须用 `inplace=False`

## 文件结构

```
Real-ESRGAN/
├── README.md
├── rrdbnet.py                              # RRDBNet 独立实现（无 basicsr 依赖）
├── examples/
│   ├── torchax_realesrgan_inference.py      # TPU 推理（whole-image）
│   ├── torchax_realesrgan_tiled.py          # TPU 推理（tile-based batch，推荐）
│   └── precision_validation.py             # CPU vs TPU 精度验证
└── benchmarks/
    └── realesrgan_cpu_benchmark.py          # CPU 性能基线
```

## 使用方法

```bash
# Tile-based 推理（推荐，最快）
python examples/torchax_realesrgan_tiled.py --input-h 2048 --input-w 1536 --tile 128 --precision default

# Tile-based highest 精度
python examples/torchax_realesrgan_tiled.py --input-h 2048 --input-w 1536 --tile 128 --precision highest

# Whole-image 推理
python examples/torchax_realesrgan_inference.py --input-h 2048 --input-w 1536

# 精度验证
python examples/precision_validation.py --input-h 256 --input-w 192

# CPU 基线
python benchmarks/realesrgan_cpu_benchmark.py --input-h 2048 --input-w 1536 --runs 5
```

## 参考

- [Real-ESRGAN GitHub](https://github.com/xinntao/Real-ESRGAN)
- [RepVGG on TPU](../RepVGG/) — 同系列迁移（简单模型）
- [gpu-tpu-pedia](https://github.com/yangwhale/gpu-tpu-pedia/tree/main/tpu)
