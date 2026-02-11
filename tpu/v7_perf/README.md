# TPU Performance Testing

TPU GEMM 性能基准测试工具，支持 TPU v6e (Trillium) 和 v7 (Ironwood)。

## Quick Start

```bash
cd chay_gemm_benchmark_simple

# 快速测试 - 使用 trace 模式 (~30 秒, 推荐)
# v7: 90%+ MFU | v6e: ~79% MFU (接近 XLA 编译效率上限)
python3 main_tpu.py --config config/tpu_trace_test.json

# 完整测试 + CSV 输出 (~10 分钟)
python3 main_tpu.py --config config/tpu_full.json --output results.csv

# 对比：使用传统计时模式 (包含 Python overhead, MFU 约 65-75%)
python3 main_tpu.py --config config/tpu_simple.json --no-trace
```

### Trace-Based Timing (推荐)

默认启用 **Trace 模式**，从 JAX profiler trace 中提取纯设备执行时间 (`device_duration_ps`)，排除 Python dispatch overhead：

| 计时模式 | 命令参数 | v6e MFU | v7 MFU | 原理 |
|----------|----------|---------|--------|------|
| **Trace (默认)** | (无需参数) | ~79% | **90%+** | 从 profiler trace 提取 `device_duration_ps` |
| Legacy | `--no-trace` | 65-75% | 65-75% | 使用 `time.perf_counter()` 端到端计时 |

> **注意**: 90%+ MFU 是 v7 (Ironwood) 上的目标。v6e 的 ~79% MFU 是 XLA 编译器对其 MXU 的利用效率上限，不是 HBM 带宽瓶颈。
>
> 参考: [accelerator-microbenchmarks](https://github.com/google/accelerator-microbenchmarks) (专为 v7 优化)

## 目录结构

```
v7_perf/
├── README.md                           # 本文件
├── chay_gemm_benchmark_simple/         # GEMM benchmark 工具
│   ├── main_tpu.py                     # TPU 入口 (JAX)
│   ├── main.py                         # GPU 入口 (PyTorch)
│   ├── backends/
│   │   └── tpu/
│   │       ├── tpu_backends.py         # TPU 后端实现 (v6e/v7)
│   │       └── trace_utils.py          # Trace-based timing 工具
│   ├── config/
│   │   ├── tpu_simple.json             # TPU 快速测试
│   │   ├── tpu_trace_test.json         # Trace 模式验证配置 (8192x8192x8192)
│   │   ├── tpu_large_matrix.json       # 大矩阵测试 (16384x18432x16384)
│   │   ├── tpu_matrix_scaling.json     # 矩阵规模扩展测试
│   │   ├── tpu_full.json               # TPU 中等测试
│   │   └── tpu_gemm.json               # TPU 完整测试
│   ├── results_v6e.csv                 # v6e 测试结果
│   └── results_v7.csv                  # v7 测试结果
├── tpu_backend_implementation_report.md # 实现报告
├── tpu_v7_benchmark_report.md          # v7 性能测试报告
├── gemm_benchmark_analysis.md          # 代码架构分析
└── findings.md                         # 研究发现
```

## 测试结果

### TPU v7 (Ironwood) — 单 chiplet 性能

测试日期: 2026-02-10 | JAX 0.8.2.dev | 拓扑 2x2x1 | Dual-Chiplet 架构

| 数据类型 | 理论峰值 (per chiplet) | 最高实测 | 最高 MFU | 最佳配置 |
|----------|------------------------|----------|----------|----------|
| bfloat16 | 1153.5 TFLOPS | **758.3 TFLOPS** | **65.7%** | M=8192, K=N=8192 |
| float32 | 1153.5 TFLOPS | 670.9 TFLOPS | 58.2% | M=8192, K=N=8192 |
| int8 | 2307 TOPS | 711.9 TOPS | 30.9% | M=8192, K=N=8192 |

> 详细分析见 [TPU v7 性能测试报告](tpu_v7_benchmark_report.md)

### TPU v6e (Trillium) — 单芯片性能

测试日期: 2026-02-11 (Trace 模式)

| 数据类型 | 理论峰值 | 最高实测 | MFU (Trace) | MFU (Legacy) |
|----------|----------|----------|-------------|--------------|
| bfloat16 | 918 TFLOPS | 728 TFLOPS | **79.3%** | 72.5% |
| float32 | 918 TFLOPS | 583 TFLOPS | 63.5%* | — |
| int8 | 1836 TOPS | 1129 TOPS | 61.5%* | — |

\* float32/int8 待用 Trace 模式重测

### v7 vs v6e 跨代对比

| 指标 | v6e 整芯片 | v7 单 chiplet | v7 整芯片 (推算) | 加速比 |
|------|-----------|--------------|-----------------|--------|
| BF16 最高 TFLOPS | 689 | 758 | ~1516 | **2.2x** |
| FP32 最高 TFLOPS | 583 | 671 | ~1342 | **2.3x** |

## 关键发现

1. **v7 单 chiplet 超越 v6e 整芯片** — BF16 758 vs 689 TFLOPS
2. **Dual-Chiplet 架构** — JAX 将每个芯片暴露为 2 个设备，per-device 峰值 = per-chip / 2
3. **FP32 提升显著** — v7 的 BF16+FP32累加路径优化更成熟 (1.86x vs v6e)
4. **INT8 待优化** — v7 INT8 MFU 仅 30.9%，可能受 JAX dev 版本限制
5. **TPU float32 = bf16 性能** — MXU 用 bf16 计算 + fp32 累加
6. **小 batch 效率低** — M < 512 时 MFU < 15%
7. **Trace-based timing 更准确** — 从 profiler trace 提取纯设备时间，排除 Python overhead
   - v7 (Ironwood): 可达 90%+ MFU
   - v6e (Trillium): ~79% MFU (XLA 对 v6e MXU 的利用效率上限)
8. **v6e HBM 带宽不是瓶颈** — M=8192 GEMM 的 AI=1365 >> v6e CB ratio=561，workload 是 compute-bound

## 文档

- [TPU v7 性能测试报告](tpu_v7_benchmark_report.md) - v7 完整测试分析
- [实现报告](tpu_backend_implementation_report.md) - 完整技术文档
- [架构分析](gemm_benchmark_analysis.md) - 代码设计解读
- [研究发现](findings.md) - 关键技术决策

---

*Created: 2026-02-09*
*Updated: 2026-02-11 — Trace-based timing: v6e 达 79% MFU (XLA 利用率上限), v7 目标 90%+*
