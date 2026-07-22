# DeepSeek-V4 (Flash / Pro) · A4 (B200) · SGLang / vLLM 推理

> DeepSeek-V4-Flash（43 层 / 256 experts / MXFP4）在 **A4（8× NVIDIA B200 180GB, GCP a4-megagpu-8g）** 上的端到端推理复测。
> 对标 [gddezero/b200-perf-opt — 09_deepseek_v4_b200.md](https://github.com/gddezero/b200-perf-opt/blob/main/09_deepseek_v4_b200.md) 独立 reproduce，SGLang + vLLM 双引擎 apple-to-apple。

## 文档导航

| 文档 | 引擎 | 用途 |
|---|---|---|
| [**README.sglang.md**](./README.sglang.md) | SGLang | B200 端到端：FlashMLA kernel 测试、MXFP4 专家权重修复、evalscope 压测、三套生产配置（低延迟 / 平衡 / 高吞吐）|
| [README.vllm.md](./README.vllm.md) | vLLM | 同硬件 / 模型 / 压测的 vLLM 对照，与 SGLang apple-to-apple |

## 硬件 / 模型

- **机器**：GCP a4-megagpu-8g（8× NVIDIA B200 180GB, CUDA 13.0+）
- **模型**：DeepSeek-V4-Flash（43 层, 256 experts, MXFP4 ~160GB）；Pro（61 层, 105.57 GB/卡 @ TP=8）
- **软件**：SGLang `deepseek-v4-blackwell` + FlashMLA / vLLM v0.20.0

## 关键结论（实测 2026-05）

- **Flash 峰值 2,739 tok/s**（C=600，SGLang 高吞吐 TP=8 DP=8 + DeepEP），中低并发（C=40 / 100）反超参考 +8~12%。
- **关键修复（Maxwell 反馈）**：去掉 `NCCL_IB_DISABLE=1`——它会让 DeepEP all-to-all 退化到 PCIe，中并发吞吐掉 28~53%。
- **三套生产配置**：低延迟（EAGLE n=3）/ 平衡（DeepEP + EAGLE n=1）/ 高吞吐（DeepEP 无 spec）。

## 关联

> GB300（A4X Max）上的 V4 NVFP4 推理见 [../../a4x-max/deepseek-v4/](../../a4x-max/deepseek-v4/)。
