# DeepSeek-V4 (Flash / Pro) · GB300 NVL72 · SGLang NVFP4 推理

> DeepSeek-V4（**CSA + HCA hybrid attention**，~10% KV cache、1M 上下文、MegaMoE W4A4）在 **GB300 NVL72 (A4X Max / GKE)** 上用 SGLang 做 NVFP4 推理的测试记录。
> 资料来源：SGLang V4 cookbook、lmsys Day-0 博客（2026-04-25）、pytorch「Serving DeepSeek-V4 on GB300」（2026-06-23）、SemiAnalysis InferenceX recipe `disagg-gb300-10p1d-dep4-dep32-18-c2500.yaml`。

## 文档导航

| 文档 | 用途 | 何时看 |
|---|---|---|
| [**sglang-v4-gb300-TEST-PLAN.md**](./sglang-v4-gb300-TEST-PLAN.md) | **V4 端到端测试指南 + Benchmark 报告**：满配最优 **§12（8,993 = 官方 80%）**、**§13 满配复现 checklist（照抄即可）**、可复现运行手册（§3.9）、官方 11,200 口径说明（§10）。 | **想跑 V4 从这里开始** |
| [gb300-local-ssd-raid0-SETUP.md](./gb300-local-ssd-raid0-SETUP.md) | **Local SSD RAID 0 挂载指南**：4× NVMe → RAID0 → 12T / 14GB/s 读，arm64 DaemonSet + 污染节点排查。V4-Pro 800G 权重的存储基础。 | 搭 Local SSD / RAID 建不成时 |

## 进度

| Phase | 内容 | 状态 |
|---|---|---|
| Phase 1 | V4-Flash（284B/13B 激活）单节点 TP4 | ✅ 通过（conc64 **8540 tok/s/GPU**）|
| Phase 2 | V4-Pro（1.6T/49B 激活）单节点 TP4 | ✅ 通过（conc64 **2794 tok/s/GPU**）|
| Phase 3 | PD-disagg（Dynamo + dep8-MTP + 16 prefill + 多 frontend）| ✅ **满配 8,993 tok/s/GPU = 官方 11,200 的 80%**（sa-bench 开环，官方口径 output÷decode-GPU）|

## 关键结论（实测 2026-07-20，8K/1K，单节点 4 GPU）

- **Flash 单节点碾压 R1 64 卡 6.3×**（8540 vs 1359 tok/s/GPU）——V4 架构（CSA+HCA 打薄 KV + SWA）每 token 效率远超 R1 全注意力。
- **Flash vs Pro 差 3.1×**（8540 vs 2794）——Pro 总参大 5.6×、激活大 3.8×，符合模型代差。
- **Pro 单节点 2,794 只是 Phase 2 基线**，非终点——满配 PD 分离见下。
- **Pro 满配达官方 80%（Phase 3，2026-07-21）**：18 节点 NVL72、**16 prefill(dep4) + dep8-MTP + 多 frontend + Dynamo + sa-bench 开环 = 8,993 tok/s/GPU**（官方口径 output÷decode-GPU）= 官方 11,200 的 **80%**。演进：单 frontend 5,060 → 多 frontend 6,788 → 16 prefill 8,993（14→16 收敛）。剩余 20% = 官方 pinned 镜像单卡内核成熟度（autotune / EPLB 均无法缩小），**非架构 / 拓扑 / 编排问题**。

## 存储关键点

V4-Pro 1.6T ≈ 851G FP4，**内存盘（tmpfs）放不下**（模型 + 运行时 > 942G 节点 RAM），且 RAM 要留给 KV cache。所有权重放 **Local SSD RAID `/mnt/disks/raid/0`**（读 14GB/s，加载 <1min）。GCS 备份走 **ADC + python SDK**（节点 OAuth scope 只读、org 禁 SA key，唯一可写法，见 TEST-PLAN §3.9 Step 3）。

- 模型 GCS 备份：`gs://chrisya-gb300-models/DeepSeek-V4-Flash-NVFP4`（168G）/ `DeepSeek-V4-Pro-NVFP4`（913G）。

> R1（V3 架构，PD 分离 Wide-EP）见 [../deepseek-v3/](../deepseek-v3/)。
