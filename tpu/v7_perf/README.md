# TPU Performance Testing

TPU GEMM 性能基准测试工具，支持 TPU v6e (Trillium) 和 v7 (Ironwood)。

## Quick Start

```bash
cd chay_gemm_benchmark_simple

# 快速测试 (~25 秒)
python3 main_tpu.py --config config/tpu_simple.json

# 完整测试 + CSV 输出 (~10 分钟)
python3 main_tpu.py --config config/tpu_full.json --output results.csv
```

## 目录结构

```
v7_perf/
├── README.md                           # 本文件
├── chay_gemm_benchmark_simple/         # GEMM benchmark 工具
│   ├── main_tpu.py                     # TPU 入口 (JAX)
│   ├── main.py                         # GPU 入口 (PyTorch)
│   ├── backends/
│   │   └── tpu/tpu_backends.py         # TPU 后端实现
│   ├── config/
│   │   ├── tpu_simple.json             # TPU 快速测试
│   │   ├── tpu_full.json               # TPU 中等测试
│   │   └── tpu_gemm.json               # TPU 完整测试
│   └── results_v6e.csv                 # v6e 测试结果
├── tpu_backend_implementation_report.md # 实现报告
├── gemm_benchmark_analysis.md          # 代码架构分析
└── findings.md                         # 研究发现
```

## TPU v6e 测试结果

| 数据类型 | 理论峰值 | 最高实测 | 最高 MFU |
|----------|----------|----------|----------|
| bfloat16 | 918 TFLOPS | 689 TFLOPS | **75.0%** |
| float32 | 918 TFLOPS | 583 TFLOPS | 63.5% |
| int8 | 1836 TOPS | 1129 TOPS | 61.5% |

## 关键发现

1. **TPU float32 = bf16 性能** - MXU 用 bf16 计算 + fp32 累加
2. **小 batch 效率低** - M < 512 时 MFU < 15%
3. **最佳配置** - M=4096, K=8192, N=8192 达到 75% MFU

## 文档

- [实现报告](tpu_backend_implementation_report.md) - 完整技术文档
- [架构分析](gemm_benchmark_analysis.md) - 代码设计解读
- [研究发现](findings.md) - 关键技术决策

## 添加 TPU v7 支持

1. 更新 `backends/tpu/tpu_backends.py` 中的 `TpuV7Backend.PEAK_TFLOPS`
2. 更新 `hw_spec.py` 中的设备规格
3. 运行测试验证

---

*Created: 2026-02-09*
