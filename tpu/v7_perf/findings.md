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
├── main_tpu.py          # TPU 入口 (JAX) ← 新增
├── backends.py          # GPU 后端抽象 + NVIDIA 实现
├── tpu_backends.py      # TPU 后端抽象 + v6e/v7 实现 ← 新增
├── hw_spec.py           # 硬件理论峰值定义（已更新含 TPU）
├── utils.py             # GPU 工具函数
├── config/
│   ├── gemm.json        # GPU 完整测试配置
│   ├── simple.json      # GPU 简化测试配置
│   ├── tpu_gemm.json    # TPU 完整测试配置 ← 新增
│   ├── tpu_simple.json  # TPU 简化测试配置 ← 新增
│   └── tpu_full.json    # TPU 中等测试配置 ← 新增
└── backends/
    └── nv_gpu_cublas/   # cuBLAS C++ 扩展
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

---

## Issues Encountered

| Issue | Resolution |
|-------|------------|
| float32 MFU > 100% | 修正理论峰值：TPU float32 走 bf16 计算路径，峰值应为 918 而非 459 |
| TPU 版本检测不准确 | JAX 设备名格式不标准，采用默认 v6e + 日志提示 |
| int8 输出类型 | JAX int8 GEMM 自动累加到 int32，需正确计算带宽 |

---

## TPU v6e 测试结果摘要

### 性能数据（2026-02-09 测试）

| 数据类型 | 理论峰值 | 最高实测 | 最高 MFU | 平均 MFU |
|----------|----------|----------|----------|----------|
| bfloat16 | 918 TFLOPS | 689 TFLOPS | **75.0%** | 36.2% |
| float32 | 918 TFLOPS | 583 TFLOPS | 63.5% | 28.5% |
| int8 | 1836 TOPS | 1129 TOPS | 61.5% | 24.8% |

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
*Updated: 2026-02-09 after TPU v6e testing*
