# Kimi K3 (2.8T / 104B 激活) · GB300 NVL72 · vLLM 推理

> Moonshot AI 于 **2026-07-27** 开源权重，vLLM 同日 day-0 支持。
> 本目录是**从官方 day-0 博客与 recipes 1:1 蒸馏**的端到端部署文档。
>
> ⚠️ **状态：`[未实测]`** —— 与本仓库 [deepseek-v4](../deepseek-v4/)、[deepseek-v3](../deepseek-v3/)
> 那些标 `[已验证]` 的 runbook **性质不同**。本目录所有命令来自官方发布材料，
> 尚未在本环境跑过。跑通后请把各节改标 `[已验证]` 并补 §验证记录。

## 文档导航

| 文档 | 用途 |
|---|---|
| [**VLLM-K3-RUNBOOK.md**](./VLLM-K3-RUNBOOK.md) | **端到端 Runbook**：从 0 开始，pod → RAID → 拉权重 → 起服务 → 压测 → 复现官方 370 tok/s |
| [scripts/](./scripts/) | 启动与压测脚本（TP8 + DSpark / TP16 / 纯 TP8 基线） |
| [gb300-local-ssd-raid0-SETUP.md](../deepseek-v4/gb300-local-ssd-raid0-SETUP.md) | RAID 0 挂载（1.4 TB 权重的存储基础，复用 V4 那份） |

## 为什么这个模型值得单独一套文档

Kimi K3 **不是更大的 K2**，它同时改了四件事，每一件都动了推理引擎的热路径：

| 架构点 | K3 的做法 | 对推理的影响 |
|---|---|---|
| **注意力** | **KDA**（线性/循环注意力）为主 + 周期性 full attention。93 层 = **69 KDA + 24 Gated MLA** | 引擎要**同时**管两种缓存：分页 KV + 循环状态块。prefix caching 必须重做 |
| **残差** | **AttnRes**：每个子层用学习到的 pseudo-query 对前面各层块的残差做深度方向注意力 | 跨层读写激增，需要专用融合 kernel |
| **MoE** | **Stable LatentMoE**，**896 专家选 16** + 2 共享，latent 维 3584 | 路由 / dispatch / 负载均衡成为端到端性能主因 |
| **量化** | 出厂即 **MXFP4 权重 + MXFP8 激活**（SFT 阶段起量化感知训练） | 需要支持 SiTU 激活的 FP4 专家路径 |

另外还有 **1M 上下文**、**原生视觉**（MoonViT-V2，401M）、以及一个**用 Python 程序而不是 Jinja 渲染的 chat template**。

> vLLM 为 K3 重做的**混合 prefix caching**（把物理 KDA 状态块大小与前缀匹配粒度解耦）
> 已经进了 vLLM core，**所有同类混合线性注意力模型都受益**，不是 K3 专用补丁。

## 卡数速查

官方口径：**最低 1 个 8×B300 节点（或 GB300 NVL72）**；16×B200 亦可。

GB300 NVL72 每节点 4 GPU，所以在本环境：

| 配置 | GPU 数 | 节点数 | bs=1 单用户 | 加 DSpark |
|---|---|---|---|---|
| **TP8** | 8 | **2** | 111 tok/s | **331 tok/s** |
| **TP16** | 16 | **4** | 118 tok/s | **370 tok/s**（3.14×） |

> 官方 reproduce recipe 给的就是 **TP8 + DSpark（`--nnodes 2`）**，
> 370 tok/s 那条要 TP16。本仓库 Runbook 以 **TP8 两节点**为主线，TP16 作为扩展。

大规模服务的已验证拓扑（官方）：**TEP8 prefill → DEP16 decode，NIXL 做 KV 传输**。

## 存储

2.8T 参数 × MXFP4（0.5 byte）≈ **1.4 TB 权重**。

- **内存盘放不下**（节点 942G RAM），必须落 Local SSD RAID `/mnt/disks/raid/0`（12 T / 14 GB/s）
- 拉取方式复用 V4 那套 [`pull-gcs-model.sh`](../deepseek-v4/scripts/pull-gcs-model.sh)（curl + bearer token 打 GCS JSON API，16 路并行）——**别用 gcloud，镜像里没装**
- 另需 DSpark draft 模型 `Inferact/Kimi-K3-DSpark`

## 六条官方部署提示（照抄，别跳）

1. **`--enable-prefix-caching` 必须显式加** —— vLLM 一般默认开，但**对 K3 默认是关的**（混合缓存设计还在演进）
2. **Tool calling 上线前必须用自己的流量验证** —— 官方明说偶尔会吐出自家 parser 不认的格式，导致 `tool_calls` 返回空。生产要做 schema 校验 + 空值重试/降级，或用 structured tool calling
3. **`--all2all-backend`**：NVLink 用 `flashinfer_nvlink_one_sided`，RDMA 用 `deepep_v2`
4. **`--moe-backend`**：DEP 环境用 `deep_gemm_mega_moe`，TP > 1 用 `flashinfer_trtllm`
5. **`VLLM_USE_RUST_FRONTEND=1`** —— Rust 前端完整支持本模型
6. **ViT 走 DP 不走 TP** —— `--mm-encoder-tp-mode=data` 默认开。K3 视觉编码器 `head_size=12`，TP=8 除不尽；且它不到 1B 而主干 2T，走 DP 避免 all-reduce 开销

## 官方精度基线（vLLM 服务端点实测，max reasoning effort）

| GSM8K | GPQA-Diamond | OCRBench | MMMU Pro Vision |
|---|---|---|---|
| 0.976 | 0.939 | 0.889 | 0.818 |

> **评测踩坑**：K3 回答前思考很长。分数低通常是**被截断**而不是答错。
> 先把 `max_tokens` 放大、reasoning effort 调高、检查是否 cut-off，再去 debug 别的。

## 与本仓库其他模型的关系

| | DeepSeek-V4-Pro | Kimi K3 |
|---|---|---|
| 规模 | 1.6T / 49B 激活 | **2.8T / 104B 激活** |
| 注意力 | CSA + HCA hybrid | **KDA + Gated MLA hybrid** |
| 量化 | MegaMoE W4A4 | **MXFP4 权重 / MXFP8 激活** |
| 专家 | — | **896 选 16** + 2 共享 |
| 多模态 | 否 | **是**（MoonViT-V2） |

两者都是「混合注意力 + 极稀疏 MoE + 4-bit」，**拓扑选择的教训可以互相借鉴**——
参见 V4 那份 README 的结论：*先选对拓扑 / KV 布局，再谈调参*。

## 来源

- vLLM day-0 博客：<https://vllm.ai/blog/2026-07-27-k3>
- vLLM 架构预告（KDA prefix caching 设计）：<https://vllm.ai/blog/2026-07-22-kimi-k3-preview>
- 官方 recipes 与 Docker 镜像：<https://recipes.vllm.ai/moonshotai/Kimi-K3>
- 模型卡：<https://huggingface.co/moonshotai/Kimi-K3>
- DSpark draft：<https://huggingface.co/Inferact/Kimi-K3-DSpark>
- Moonshot 技术博客：<https://www.kimi.com/blog/kimi-k3>
