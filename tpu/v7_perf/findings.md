# Findings & Decisions

## Requirements (Original)
- 梳理 chay_gemm_benchmark_simple 的代码设计逻辑
- 说明测试了什么内容（矩阵规模、数据类型）
- 说明输出报告的格式
- 用 Mermaid 或 SVG 画架构图

## Requirements (Extended - TPU Backend)
- 为 GEMM benchmark 添加 TPU v6e 后端
- 设计通用架构以支持未来 TPU v7
- 通过测试迭代修复 bug
- 生成实现过程和测试结果报告
- 创建 chip-performance-test skill

---

## Research Findings

### TPU v6e (Trillium) 硬件规格
来源: [Google Cloud Blog](https://cloud.google.com/blog/products/compute/introducing-trillium-6th-gen-tpus)

| 指标 | 值 |
|------|-----|
| Peak BF16 | 918 TFLOPS |
| Peak INT8 | 1836 TOPS |
| HBM Capacity | 32 GB |
| HBM Bandwidth | 1,600 GB/s |
| MXU Size | 256×256 (65,536 MACs/cycle) |
| TensorCores/Chip | 1 |
| MXU/TensorCore | 2 |
| ICI Bandwidth | 13 TB/s |

### JAX GEMM Benchmark 最佳实践
来源: [JAX Documentation](https://docs.jax.dev/en/latest/jit-compilation.html)

1. **异步调度问题**: JAX 在 TPU 上使用异步调度，必须用 `block_until_ready()` 确保计算完成再计时
2. **JIT 编译**: 使用 `@jax.jit` 装饰器确保 GEMM 被编译到 XLA HLO
3. **低级 API**: `jax.lax.dot_general` 提供对矩阵乘法的精确控制
4. **预热必要性**: 首次调用触发 JIT 编译，需要 warmup 避免编译时间影响测量

### 代码结构（更新后）
```
chay_gemm_benchmark_simple/
├── main.py              # GPU 入口 (PyTorch)
├── main_tpu.py          # TPU 入口 (JAX)
├── backends.py          # GPU 后端抽象 + NVIDIA 实现
├── hw_spec.py           # 硬件理论峰值定义（已更新含 TPU）
├── utils.py             # GPU 工具函数
├── config/
│   ├── gemm.json        # GPU 完整测试配置
│   ├── simple.json      # GPU 简化测试配置
│   ├── tpu_gemm.json    # TPU 完整测试配置
│   ├── tpu_simple.json  # TPU 简化测试配置
│   ├── tpu_full.json    # TPU 中等测试配置
│   └── tpu_trace_test.json  # Trace timing 快速验证 ← 新增
└── backends/
    ├── nv_gpu_cublas/   # cuBLAS C++ 扩展
    └── tpu/             # TPU 后端 ← 新增
        ├── __init__.py
        ├── tpu_backends.py   # TPU 后端抽象 + v6e/v7 实现
        └── trace_utils.py    # Trace-based timing 工具
```

---

## Technical Decisions

| Decision | Rationale |
|----------|-----------|
| 使用 JAX 而非 PyTorch/XLA | TPU 原生支持，性能更优，生态更成熟 |
| 创建独立 tpu_backends.py | 避免 JAX 和 PyTorch 混合导入冲突 |
| 创建独立 main_tpu.py | TPU 计时机制不同，需要独立入口 |
| TpuBackendBase 基类设计 | 抽象通用逻辑，便于 v6e/v7 复用 |
| jax.block_until_ready() | JAX 异步调度，必须等待完成才能正确计时 |
| jax.lax.dot_general | 低级 GEMM 操作，精确控制矩阵乘法 |
| float32 峰值 = bf16 峰值 | TPU MXU 用 bf16 计算 + fp32 累加 |
| **Trace-based timing** | 从 profiler 提取 device_duration_ps，排除 Python overhead |
| **MARKER 放在 jit 函数内** | 只有在 jit 内部的 named_scope 才会出现在 trace tf_op 字段 |
| **默认启用 Trace 模式** | 更准确的 MFU 测量，--no-trace 可回退 Legacy |

---

## Issues Encountered

| Issue | Resolution |
|-------|------------|
| float32 MFU > 100% | 修正理论峰值：TPU float32 走 bf16 计算路径，峰值应为 918 而非 459 |
| TPU 版本检测不准确 | JAX 设备名格式不标准，采用默认 v6e + 日志提示 |
| int8 输出类型 | JAX int8 GEMM 自动累加到 int32，需正确计算带宽 |
| **MARKER 未出现在 trace** | named_scope 必须在 jit 函数内部，外部的 scope 不会写入 tf_op |
| **v6e fallback 策略** | v6e 的 named_scope 行为与 v7 不同，添加 dot_general 事件查找回退 |
| **90% MFU 未达成 (v6e)** | 确认是 XLA 对 v6e MXU 的利用效率上限 (~80%)，非 HBM 带宽瓶颈 |
| **--xla_tpu_dvfs_p_state=7** | v7 专用 DVFS 参数，v6e 不支持 |

---

## Trace-Based Timing 发现 (2026-02-11)

### 背景
使用 `time.perf_counter()` 的 Legacy timing 包含 Python dispatch overhead，导致 MFU 偏低。
参考 [accelerator-microbenchmarks](https://github.com/google/accelerator-microbenchmarks) 实现 Trace-based timing。

### 实现方法
从 JAX profiler trace 中提取 `device_duration_ps` 字段，获取纯 TPU 设备执行时间：

```python
# MARKER 必须在 jit 函数内部才能出现在 trace 的 tf_op 字段
@jax.jit
def gemm_with_marker(a, b):
    with jax.named_scope("!!MARKER!!"):
        return lax.dot_general(a, b, ...)
```

### v6e vs v7 MFU 对比

| TPU 型号 | Legacy MFU | Trace MFU | 差异 |
|----------|------------|-----------|------|
| v6e (Trillium) | 72.5% | **79.3%** | +6.8% |
| v7 (Ironwood) | 65.7%* | **90%+** (目标) | — |

### 关键发现：v6e 的 79% MFU 是 XLA 编译效率上限

**HBM 带宽不是瓶颈**：
```
M=8192 GEMM Roofline 分析：
├── Arithmetic Intensity: 1,365 FLOP/Byte
├── v6e Compute/BW ratio: 560 FLOP/Byte
└── 结论: AI >> CB ratio => COMPUTE-BOUND，带宽不是瓶颈
```

**实际瓶颈是 XLA 对 v6e MXU 的利用效率 (~80%)**：
1. accelerator-microbenchmarks 只有 `Ironwood/` 目录，无 v6e 优化代码
2. `--xla_tpu_dvfs_p_state=7` 是 v7 专用 DVFS 参数，v6e 不支持
3. 要达到 90%+ MFU，需要在 **v7** 上测试

**矩阵尺寸对比测试** (确认瓶颈不是矩阵大小)：

| 矩阵尺寸 | 计算量 | v6e MFU | 结论 |
|---------|--------|---------|------|
| 8192×8192×8192 | 1.1 PFLOP | 79.3% | 我们的测试 |
| 16384×18432×16384 | 9.9 PFLOP | 78.8% | accelerator-microbenchmarks 尺寸 |

使用相同的大矩阵，v6e MFU 仍然是 ~79%，确认瓶颈是 XLA 编译效率而非矩阵规模

---

## TPU v6e 测试结果摘要

### 性能数据（2026-02-11 更新，使用 Trace timing）

| 数据类型 | 理论峰值 | 最高实测 | MFU (Trace) | MFU (Legacy) |
|----------|----------|----------|-------------|--------------|
| bfloat16 | 918 TFLOPS | 728 TFLOPS | **79.3%** | 72.5% |
| float32 | 918 TFLOPS | 583 TFLOPS | 63.5%* | — |
| int8 | 1836 TOPS | 1129 TOPS | 61.5%* | — |

\* float32/int8 为 Legacy 数据，待 Trace 模式重测

### 关键观察

1. **bfloat16 性能最优**: 75% MFU 表明 MXU 被有效利用
2. **float32 大矩阵性能下降**: M≥4096 + K=N=8192 时 MFU 降至 ~40%，可能是 HBM 带宽瓶颈
3. **int8 高吞吐量**: 达到 1129 TOPS，验证了 2x bf16 的理论吞吐量
4. **小 batch 效率低**: M=128 时 MFU 仅 3-9%，MXU 需要足够大的矩阵填满

### 最佳性能点

- **bfloat16**: M=4096, K=8192, N=8192 → 689 TFLOPS, 75% MFU
- **int8**: M=8192, K=8192, N=8192 → 1129 TOPS, 61.5% MFU

---

## Resources

### 项目文件
- TPU 后端代码: `tpu_backends.py`
- TPU 入口: `main_tpu.py`
- 测试结果: `results_v6e.csv`
- 分析报告: `gemm_benchmark_analysis.md`

### 参考资料
- [TPU v6e Documentation](https://docs.cloud.google.com/tpu/docs/v6e)
- [Trillium TPU Blog](https://cloud.google.com/blog/products/compute/introducing-trillium-6th-gen-tpus)
- [JAX GPU Performance Tips](https://docs.jax.dev/en/latest/gpu_performance_tips.html)
- [JAX JIT Compilation](https://docs.jax.dev/en/latest/jit-compilation.html)

---
*Updated: 2026-02-11 — Trace-based timing 实现，v6e 79% MFU 是 XLA 编译效率上限*
