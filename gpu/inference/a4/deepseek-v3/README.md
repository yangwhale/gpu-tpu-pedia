# DeepSeek-V3 · A4 (B200) · SGLang PD 分离推理

> DeepSeek-V3（671B MoE, FP8）在 **A4（8× NVIDIA B200 180GB, GCP a4-highgpu-8g）** 上用 SGLang 做 **PD 分离（Prefill/Decode disaggregation）** 的测试记录。
> 传输后端 NIXL，SGLang 0.5.8 / CUDA 12.9。对比 1P1D vs 2P1D 拓扑。

## 文档导航

| 文档 | 配置 | 用途 |
|---|---|---|
| [**REPORT-2P1D-DeepSeek-V3.md**](./REPORT-2P1D-DeepSeek-V3.md) | 2 Prefill + 1 Decode | **2P1D vs 1P1D 对比报告**：TTFT / P99 / 吞吐全维度对比 + 瓶颈分析 + 后续计划 |
| [REPORT-1P1D-DeepSeek-V3.md](./REPORT-1P1D-DeepSeek-V3.md) | 1 Prefill + 1 Decode | 1P1D 基线报告：架构概览 + 基准性能 |
| [start-1p1d-prefill.sh](./start-1p1d-prefill.sh) / [start-1p1d-decode.sh](./start-1p1d-decode.sh) | — | 1P1D 启动脚本（Prefill / Decode 节点）|

## 硬件 / 软件

- **硬件**：2~3× GCP a4-highgpu-8g（每台 8× NVIDIA B200 180GB HBM）
- **模型**：DeepSeek-V3（671B MoE, FP8）
- **软件**：SGLang 0.5.8 + **NIXL** transfer backend + CUDA 12.9

## 关键结论

- **2P1D 显著降 TTFT**：128 并发 TTFT 2.40s → 1.73s（**-28%**），P99 TTFT 12.07s → 6.23s（**-48%**）——加 Prefill 节点主要改善首 token 延迟（用户体验）。
- **总吞吐提升有限**：128 并发 output 仅 1,119 → 1,159 tok/s（+3.6%）——**Decode 节点是输出速率瓶颈**（256 并发 output 不再增长、TPOT 0.101→0.112s）。
- **后续 2P2D**：加第二个 Decode 节点，预计 256 并发 output 可达 **~2,500+ tok/s**；再配 CUDA Graph 提 decode 吞吐。

## 关联

> GB300（A4X Max）上的 DeepSeek-V3 / R1 NVFP4 推理见 [../../a4x-max/deepseek-v3/](../../a4x-max/deepseek-v3/)；同平台 V4 推理见 [../deepseek-v4/](../deepseek-v4/)。
