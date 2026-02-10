# Task Plan: TPU GEMM Benchmark 后端实现

## Goal
1. 为 GEMM benchmark 添加通用 TPU 后端架构（支持 v6e 和未来 v7）
2. 实现 TPU v6e 后端并通过测试迭代修复 bug
3. 生成包含实现过程和测试结果的报告
4. 创建 "chip-performance-test" skill 实现自动化芯片性能测试

## Current Phase
Phase 2: TPU v6e 后端实现

## Phases

### Phase 1: 研究与规划
- [x] 分析现有代码架构（main.py, backends.py, hw_spec.py）
- [x] 搜索 TPU v6e 硬件规格（918 TFLOPS BF16, 1600 GB/s HBM）
- [x] 搜索 JAX GEMM 基准测试最佳实践（block_until_ready, JIT）
- **Status:** complete

### Phase 2: TPU 后端实现
- [x] 设计通用 TPU 后端基类 (TpuBackendBase)
- [x] 实现 TPU v6e 后端 (TpuV6eBackend)
- [x] 添加 TPU 到 hw_spec.py（设备规格、dtype 映射）
- [x] 更新 main.py 添加 TPU 检测逻辑
- [x] 创建 TPU 专用配置文件
- **Status:** complete

### Phase 3: 测试与调试
- [ ] 运行简化测试验证基本功能
- [ ] 运行完整测试收集性能数据
- [ ] 修复发现的 bug（迭代）
- [ ] 验证性能数据合理性（对比理论峰值）
- **Status:** in_progress

### Phase 4: 报告生成
- [ ] 记录实现过程和关键决策
- [ ] 整理测试结果数据
- [ ] 生成 Markdown 报告（含图表）
- **Status:** pending

### Phase 5: Skill 创建
- [ ] 使用 skill-creator 创建 chip-performance-test skill
- [ ] 实现自动硬件检测逻辑
- [ ] 实现自动报告生成
- **Status:** pending

## Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| 使用 JAX 而非 PyTorch/XLA | TPU 原生支持，性能更优，生态更成熟 |
| TpuBackendBase 基类设计 | 抽象 TPU 通用逻辑，便于 v6e/v7 复用 |
| jax.block_until_ready() | JAX 异步调度，必须等待完成才能正确计时 |
| jax.jit 编译 GEMM | 触发 XLA 编译优化，获得真实 MXU 性能 |
| 使用 jax.lax.dot_general | 低级 GEMM 操作，精确控制矩阵乘法 |

## TPU v6e 硬件规格

| 指标 | 值 | 来源 |
|------|----|----|
| Peak BF16 | 918 TFLOPS | Google Cloud Blog |
| HBM Capacity | 32 GB | Google Cloud Docs |
| HBM Bandwidth | 1,600 GB/s | Google Cloud Docs |
| MXU Size | 256×256 | Architecture Docs |
| TensorCores/Chip | 1 | Architecture Docs |
| MXU/TensorCore | 2 | Architecture Docs |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| (待测试填充) | - | - |

## Notes
- TPU v6e 使用 JAX/XLA 生态，与 PyTorch 完全不同
- 需要创建独立的 tpu_backends.py 文件，避免混淆 torch 依赖
- 配置文件格式保持兼容，但测试矩阵维度可能需要调整（TPU 偏好 128 的倍数）
