# Kimi K3 (2.8T / 104B 激活) · GB300 NVL72 · vLLM & SGLang 推理

> Moonshot AI 于 **2026-07-27** 开源权重，**vLLM 与 SGLang 同日 day-0 支持**。
> 本目录是**从两家官方 day-0 博客与 recipes 1:1 蒸馏**的端到端部署文档。
>
> ⚠️ **状态：`[未实测]`** —— 与本仓库 [deepseek-v4](../deepseek-v4/)、[deepseek-v3](../deepseek-v3/)
> 那些标 `[已验证]` 的 runbook **性质不同**。K3 专属命令全部来自官方发布材料，
> 尚未在本环境跑过（SGLang cookbook 甚至每格自标 *Not Verified*）。
> 跑通后请把各节改标 `[已验证]` 并补 §验证记录。
>
> 🔴 **2026-07-28 起 SGLang 侧已进入实跑**，第一批实测结论已写回 runbook：
> **K3 支持不在 sglang main 分支上**（必须用 `lmsysorg/sglang:kimi-k3-*` 专用镜像，
> 普通 nightly 参数全有、模型没有）；`--enable-symm-mem` 会关掉 K3 自己的 all-reduce 融合；
> 权重校验必须做到「对 index + 解析 safetensors 头部」这一层。详见 SGLang runbook 文首五条。
>
> ✅ **但环境与流程部分是已验证的**：两份 runbook 都把 V4 / V3 那两周趟出来的
> GB300 踩坑（RAID `md127`、`kubectl exec -i`、`pkill` 自杀、SIGKILL 97 GB 泄漏、
> 冷热轮 7% 差、就绪判据三层）直接继承过来，与模型无关，对 K3 同样成立。

## 文档导航

| 文档 | 用途 |
|---|---|
| [**VLLM-K3-RUNBOOK.md**](./VLLM-K3-RUNBOOK.md) | **vLLM 端到端 Runbook**：严格照官方 day-0 博客走。pod → RAID → 1.4 TB 权重 → TP8 服务 → 复现 331 / 370 tok/s |
| [**SGLANG-K3-RUNBOOK.md**](./SGLANG-K3-RUNBOOK.md) | **SGLang 端到端 Runbook**：以 **Golden Truth** [V4-Pro SGLang runbook](../deepseek-v4/SGLANG-V4PRO-RUNBOOK.md) 与 [R1 3P2D 指南](../deepseek-v3/sglang-r1-nvfp4-gb300-3p2d-DEPLOY-GUIDE.md) 为底，K3 参数照官方补。除了操作步骤，还蒸馏了 **§9 调参方法论**（怎么找参数，不是抄参数）与 **§10 已证伪的路**（别重做）。**官方公布了 PD 数据（2,808 tok/s/GPU），所以 PD 在主线里** |
| [PD-BACKLOG.md](./PD-BACKLOG.md) | **vLLM 侧 PD 实验设计（本轮不做）**：vLLM 未公布 K3 的 P:D 配比与吞吐，此文方法论外推自 GLM-5.2，风险较高。等主线跑通有基线后再开 |
| [scripts/](./scripts/) | 启动与压测脚本。`serve-*` / `bench.sh` 是 vLLM，`sgl-*` 是 SGLang（`sgl-k3-tp8-nospec.sh` 是当前在跑的基线） |
| [manifests/](./manifests/) | `k3-sgl-fleet.yaml`（8 pod StatefulSet + DRA + ComputeDomain）、`k3-weight-sync.yaml` |
| [gb300-local-ssd-raid0-SETUP.md](../deepseek-v4/gb300-local-ssd-raid0-SETUP.md) | RAID 0 挂载（1.4 TB 权重的存储基础，复用 V4 那份） |

## 两家路线差异速查

同一个模型，两家的最优解**不一样**，别把一边的经验直接搬到另一边：

| | vLLM | SGLang |
|---|---|---|
| prefill 拓扑 | **TEP8**（attention TP + MoE EP） | **PP8×TP1 深度流水**（实测 1.7× TEP8） |
| decode KV 去重 | 靠 PD + 分页 | **DCP8 按 token 位置切**（逻辑容量 7.9×） |
| 投机时的 KDA 状态 | 引擎内处理 | **ReplaySSM**：存输入不存快照（约 32×） |
| Draft 模型 | `Inferact/Kimi-K3-DSpark` | **`RadixArk/Kimi-K3-DSpark`** ⚠️ 不是同一个 |
| 关键内存旋钮 | `--kv-cache-dtype` + `--max-model-len` | **`--mamba-full-memory-ratio`**（KDA 状态池 vs MLA KV 池） |
| bs=1 无投机（官方） | 111 tok/s (TP8) | ~113 tok/s (TP8) |
| bs=1 + 投机（官方） | **331** tok/s (TP8) / 370 (TP16) | **~423** tok/s (TP8) |
| **bs=1 本环境实测** | 未测 | **370.4 tok/s**（官方 87.6%，加速 **4.22×**） |
| **unified TP8 峰值实测** | 未测 | **2,629 tok/s @ conc 64**（8 卡，329/GPU） |

## SGLang 侧实测速览（2026-07-28，43 组）

> 完整宽表见 [SGLANG-K3-RUNBOOK.md §11.0.0](./SGLANG-K3-RUNBOOK.md)，原始数据 [`bench-raw-20260728.csv`](./bench-raw-20260728.csv)

| 旋钮 | 值多少 | 何时值钱 | 何时没用 / 不可用 |
|---|---|---|---|
| **DSPARK 投机** | **+322%**（4K bs=1） | 短上下文最强，吞吐与延迟同时改善 | **32K 只剩 +32%**，随上下文衰减 |
| **`--enable-symm-mem`** | **+35%**（bs=1） | 短上下文低并发 | 高并发 +7.8%；**DCP 下强制关闭** |
| **`--dcp-size 8`** | **+237%**（128K/c8） | **ISL ≥8K 且高并发** | 短上下文低并发 **−26%** |
| **`--mamba-full-memory-ratio`** | **±38%**（32K/c8） | **8K–32K**；4K 要高值、32K 要低值 | **128K 完全无效（差 0.02%）** |
| K3 AR 融合 | — | — | **TP8 跨节点无 multicast，拿不到** |
| DEP8（V4 那套） | — | — | **加载即 OOM，K3 上不可行** |

**推荐配置（性价比最高）**：`TP8 + DSPARK + symm-mem`，ratio 按 ISL 定（≤8K 用 0.86，8K–32K 用 0.40–0.60）；
**长上下文或长上下文高并发换 DCP8**。

⚠️ **PD 分离未取得数据** —— 3 次尝试均受阻，最后集群 GB300 节点池于 2026-07-28 20:36 被批量删除。
对标官方 **2,808 tok/s/decode-GPU** 的工作待恢复后继续，计划见 runbook §11.5c。
| PD 数据 | ❌ 只给了拓扑名字 | ✅ **2,808 tok/s/GPU** |

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

大规模服务的已验证拓扑（官方）：**TEP8 prefill → DEP16 decode，NIXL 做 KV 传输**（恰好也是 24 卡 / 6 节点）。

> ⚠️ **官方只给了这个拓扑名字，没给任何 PD 吞吐数字。**
> 上面 111 / 118 / 331 / 370 全是 **bs=1 单用户、非 PD** 的数。
> P:D 配比、total token TPS、不同 ISL 下的拐点 —— 公开资料一片空白。
> **本轮不做 PD**，先把官方给了数的路径复现干净；设计稿存 [PD-BACKLOG.md](./PD-BACKLOG.md)。

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
