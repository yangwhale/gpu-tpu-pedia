# DeepSeek-V4 (Flash / Pro) · GB300 NVL72 · SGLang / vLLM 推理

> DeepSeek-V4（**CSA + HCA hybrid attention**，~10% KV cache、1M 上下文、MegaMoE W4A4）在 **GB300 NVL72 (A4X Max / GKE)** 上的端到端推理复现记录。
> 两个框架各有一份「照抄可跑」的 runbook，都经过多轮清空环境从零重跑的审计。

## 文档导航

| 文档 | 用途 | 何时看 |
|---|---|---|
| [**SGLANG-V4PRO-RUNBOOK.md**](./SGLANG-V4PRO-RUNBOOK.md) | **SGLang 端到端 Runbook**：§1–§8 照抄可跑（自愈循环 / 就绪判据 / sa-bench），§6.1 只换 decode 参数的快路径，§10 三轮从零审计，**§11 冲击官方 11,200 的完整消融**，§12 历史数据 | 想跑 SGLang |
| [**VLLM-V4PRO-RUNBOOK.md**](./VLLM-V4PRO-RUNBOOK.md) | **vLLM 端到端 Runbook**：§1–§8 照抄可跑（KV over NVLink 三件套 / NIXL / vllm-router），§5.2 dep8 decode，§10 复刻轮记录 + 10 个文档缺陷，§12 历史数据 | 想跑 vLLM |
| [gb300-local-ssd-raid0-SETUP.md](./gb300-local-ssd-raid0-SETUP.md) | **Local SSD RAID 0 挂载指南**：4× NVMe → RAID0 → 12T / 14 GB/s，arm64 DaemonSet + 排查。800G 权重的存储基础 | 搭 Local SSD / RAID 建不成 |

## 实测成绩（2026-07-26 收官，均为本环境实跑）

### SGLang · 16 节点（14 prefill dep4 + dep8 decode）

| 指标 | 实测 | vs 官方 11,200 |
|---|---|---|
| ISL 4096 端到端 per-decode-GPU | **10,704** | **95.6%** |
| ISL 4096 decode 自报峰值 | **12,070** | **107.8%** |
| ISL 8192 端到端 | 9,354 | 83.5% |

**决定性旋钮是 `--swa-full-tokens-ratio`**（一个参数值 +54%），最优点 = full / SWA 两个 KV 池同时饱和，且**最优值跟 ISL 绑定**：4K→0.15，8K→0.10。batch 超过 ~890/rank 之后转为算力受限，12,070 是 dep8 在 GB300 上的硬天花板。

### vLLM · 5 节点（3 prefill TP4 + dep8 decode）

| 拓扑 | conc | Total tok/s | TPOT | vs 厂商 22,000 |
|---|---|---|---|---|
| 3 prefill + TP4 decode | 512 | 21,100 | 46.8 ms | 96% |
| **3 prefill + dep8 decode** | **1536** | **65,132** | **12.1 ms** | **296%** |

**决定性动作是把 decode 从 TP4 换成 dep8**（TP1 + DP8-attention + EP8）：吞吐 **3.09×**、延迟降到 **1/4**。原因是 MLA 的 KV 是所有 head 共享的压缩 latent，**TP 下不分片、只复制** —— TP4 白白存了 4 份。

> **两边最大的共同教训**：先选对拓扑 / KV 布局，再谈调参。vLLM 侧所有参数调优加起来 +45%，换一次拓扑 +347%。

## 三个跨框架通用的坑

1. **压测工具不发 `temperature=0`，投机解码直接废掉** —— 模型 `generation_config` 默认 `do_sample: true, temperature: 1.0`，而 draft 是 greedy，接受率从 34% 崩到 1.16%，同一套服务差 **3.1×**。跨框架比性能前必须先对齐这一项。
2. **`pkill -f <pat>` 会匹配到 `kubectl exec` 自己那条命令行 → 自杀**，后续启动语句根本不执行，而且不报错。本项目在两个框架上踩了四次。用 `'dynamo[.]sglang'` 这样的括号转义，或 `pkill -x` 精确匹配进程名。
3. **「所有健康信号全绿但性能腰斩」是常态** —— 显存、日志、HTTP 200 都会骗人。SGLang 侧要查 etcd 注册数（判就绪）+ 显存（判存活，两个方向用相反的判据）；vLLM 侧要认启动日志里那四行 DeepGEMM kernel 声明。

## 早期阶段成绩（单节点，2026-07-20，8K/1K）

| Phase | 内容 | 结果 |
|---|---|---|
| Phase 1 | V4-Flash（284B / 13B 激活）单节点 TP4 | conc64 **8,540 tok/s/GPU** |
| Phase 2 | V4-Pro（1.6T / 49B 激活）单节点 TP4 | conc64 **2,794 tok/s/GPU** |

- **Flash 单节点碾压 R1 64 卡 6.3×**（8,540 vs 1,359）—— V4 架构（CSA+HCA 打薄 KV + SWA）每 token 效率远超 R1 全注意力。
- **Flash vs Pro 差 3.1×** —— Pro 总参大 5.6×、激活大 3.8×，符合模型代差。

## 存储关键点

V4-Pro 权重 806G（SGLang 用的官方原装）/ 832G（vLLM 用的 DSpark 版），**内存盘放不下**（模型 + 运行时 > 942G 节点 RAM），且 RAM 要留给 KV cache。全部放 **Local SSD RAID `/mnt/disks/raid/0`**（读 14 GB/s，加载 < 1 min）。

- GCS 备份：`gs://<bucket>/DeepSeek-V4-Pro`、`DeepSeek-V4-Pro-DSpark`、`DeepSeek-V4-Flash-NVFP4`
- **拉模型别用 gcloud**（vLLM 镜像里没装）—— 用 [`scripts/pull-gcs-model.sh`](./scripts/pull-gcs-model.sh)：curl + bearer token 打 GCS JSON API，16 路并行 **2.7 GB/s/pod**，比 gcloud 还快。注意 access token 只有 1 小时有效期，拉大模型前先刷新。
- **重建任何 pod 之后都要重新校验 shard 数** —— 调度器会换节点，hostPath 不跟着 pod 走。

> R1（V3 架构，PD 分离 Wide-EP）见 [../deepseek-v3/](../deepseek-v3/)。
