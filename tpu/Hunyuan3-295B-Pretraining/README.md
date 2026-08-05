# 腾讯混元 3（295B-A21B）在 TPU 上预训练

把 Tencent Hunyuan 3 移植进 MaxText，并在 **TPU v5p** 和 **TPU v7（Ironwood）**
上跑通 80 层完整 295B-A21B 预训练。本目录收录方案、可复现的操作步骤、
以及三个平台的横向性能对照。

**MaxText 原生不支持混元 3。** 所需组件全都已经存在，只是分散在两个
decoder block 里 —— 本项目把它们接成一个新的 `decoder_block: "hunyuan3"`，
新写的代码只有装配逻辑，零新数学。

---

## 结果

| | **v5p**<br>**256 chips** | **v7 Ironwood**<br>**64 chips** | **v7 Ironwood**<br>**256 chips** | **GB300**<br>**64 GPU**（参照） |
|---|---|---|---|---|
| 计算单元数 | 256 chips | **64 chips** | **256 chips** | 64 GPU |
| 参数量（框架报） | 298.786 B | 298.786 B | 298.786 B | — |
| 稳态 step | 63.2 s | 23.5 s | 30.4 s | — |
| **TFLOP/s / 计算单元** | 161.0 | **580.0** | **598.8** | 854.0 |
| **MFU** | **35.07%** | 25.14% | **25.96%** | 31.60% |
| **整机 token 吞吐** | 265,588 tok/s | 267,284 tok/s | **1,103,757 tok/s** | 399,488 tok/s |
| **每单元 token 吞吐** | 1,037 | **4,176** | **4,312** | 6,242 |
| 最优配方 | 官方 DSV3 v5p 配方 | `DP1×FSDP128`<br>tile + pdbs 12 | `DP2×FSDP256`<br>tile + pdbs 16 | — |
| 状态 | ✅ 调优收敛，异地复现 | ✅ 小规模天花板 | ✅ **已达目标线** 600–630 | 已调优 |
| FP8（同硬件，**分母 4614**） | — | — | 🔬 **618 / MFU 13.4%，未调优**（DSV3 743.5 / 16.1%） | — |

四边跑的是同一个 295B-A21B、同样 BF16、同样 seq 4096、同样合成数据、不开 checkpoint。
**横向比只看「每单元 token 吞吐」和 MFU**，整机吞吐随规模走，不可比。

三条读法：

1. **v7 每芯片吞吐是 v5p 的 4.16 倍**（4,312 vs 1,037），MFU 却更低（25.96% vs 35.07%）——
   因为 v7 的 BF16 峰值是 v5p 的 5.03 倍，而 HBM 带宽只涨 2.64 倍、ICI 只涨 2 倍。
   **MFU 下降但绝对吞吐大涨，是硬件比例失衡的正常表现，不是调优没做好。**
2. **v7 已追到 GB300 单卡吞吐的 69.1%**（4,312 vs 6,242），调优前只有 51.4%。
3. **v5p 的 35.07% 仍高于 GB300 的 31.6%** —— 256 芯片的 3D torus 加 SparseCore
   集合通信卸载，把 MoE 那些碎通信藏得比 NVLink 域还干净。

> **v7 的 64 芯片和 256 芯片跑出同一水位**（580 vs 580，峰值 HBM 一字节不差）。
> 256 芯片那多出的 3.3% 来自「有更宽的 FSDP 可选，能腾显存换更大 batch」，
> **不是规模本身**。详见 [QUICKSTART-v7 §4.2.1](QUICKSTART-v7.md#421-扩展性weak-scaling-100strong-scaling-有代价)。

---

## 从哪里开始

| 你的情况 | 读这个 |
|---|---|
| **想在 v5p 上把它跑起来** | **[QUICKSTART-v5p.md](QUICKSTART-v5p.md)** —— 建池到出数，两条命令，含完整参数与基线。**这份从零验证过** |
| **想在 v7 (Ironwood) 上跑** | **[QUICKSTART-v7.md](QUICKSTART-v7.md)** —— **直接给最优配方**，照抄就能跑到 599。含 64 / 256 两个规模、端到端复现、单位换算 |
| **想知道 v7 那 599 是怎么调出来的** | **[TUNING-v7.md](TUNING-v7.md)** —— 445 → 599 的完整故事线，每步的原因 / 机理 / 收益；瓶颈判定、扩展性、HBM 模型；全部负面案例折叠在附录 |
| 想把**别的**模型移植进 MaxText | [MAXTEXT-PORTING-GUIDE.md](MAXTEXT-PORTING-GUIDE.md) —— 从本项目总结的通用范式，与 Hy3 无关 |
| 想知道某个数字/结论怎么来的 | [EXPERIMENT-LOG.md](EXPERIMENT-LOG.md) —— 完整实验档案，2600 行 |
| 只想跑脚本 | [maxtext-hunyuan3/](maxtext-hunyuan3/) —— `prep.sh` + `run.sh` |

---

## 代码

**唯一真相是 fork 的分支**，本仓不留代码副本：

```
https://github.com/yangwhale/maxtext   分支 hunyuan3
```

基于上游 main，三个 commit，按两个上游 PR 的边界拆好：

| commit | 归属 |
|---|---|
| `Resolve the loss-free-balancing bias path per decoder block` | PR ①（纯上游 bug 修复，与 Hy3 无关） |
| `Add Tencent Hunyuan 3 (295B-A21B)` | PR ② |
| `Let Hunyuan3 use the SwiGLU activation bound too` | PR ② |

改动规模：**新增 3 个文件**（模型层 161 行 + 2 个 yml）、**改动上游 12 个文件**。

---

## 模型速览

| | |
|---|---|
| 结构 | 80 层，第 0 层 dense、1–79 层 MoE |
| Attention | GQA 64q / 8kv，head_dim 128，QK-LayerNorm，无 bias —— **血统是 Qwen3** |
| MoE | 192 routed experts top-8 + 1 shared，sigmoid 路由 + 专家偏置 —— **血统是 DeepSeek V3** |
| 其他 | MTP 1 层，vocab 120832，routed scaling 2.826 |
| 参数分布 | **97% 在路由专家里**，attention 只占 2% |

参数分布直接决定并行策略：**TP 无用**（切 attention 纯亏通信）、
**TPU 上不要用 EP** —— ICI 是 3D torus，AllToAll 要多跳转发，不像 GPU NVLink 那样是
full mesh；16 芯片实测 EP=4 是 **−71%**。**FSDP 宽度固定在 128，多出来的 device 全给 DP。**

与 DeepSeek V3 的关键差异：**Hy3 没有 device-limited routing**，
是在全部 192 个专家里做全局 top-8。照搬 DSV3 配方把
`n_routing_groups` / `topk_routing_group` 加进来会改变路由行为，而且不报错。

---

## 这个项目最大的一条经验

> **MaxText 里几乎每个「这个模型该走哪条路」的判断，都是一张按模型家族名字
> 写死的表。** 加一个新模型不是改一处，是把这类表全部找齐、逐个问
> 「我该不该在这里」。

同一个模式在本项目出现了 **10 次**，其中 9 次是运行时才炸、或者
**根本不炸只是安静地跑出另一套语义**（路由分支和 FLOP 公式就是后者）。
完整台账见 [EXPERIMENT-LOG.md §八](EXPERIMENT-LOG.md)。

另一条：**起点是自己攒的配置，MFU 只有 2.45%；照抄官方 DeepSeek3 v5p 配方、
只换模型名，一步跳到 31.56%。** 移植新模型时，先找官方同类配方，再谈调参。

---

## 还没做的

| 项 | 说明 |
|---|---|
| 真实数据集收敛验证 | 目前全是 synthetic，只证明「能算且不发散」 |
| HF 权重 → MaxText Orbax 转换 | 只跑吞吐基线可以不碰；要 SFT 必须做 |
| v7 BF16 冲到 630 | 当前 599（25.96%），已到目标区间下沿 |
| **v7 FP8 调优** | 当前 **618**（对 FP8 峰值 4614 是 MFU 13.4%），DSV3 同口径 **743.5 / 16.1%**，落后 20.3%。**FP8 走的是另一条 GMM kernel，那条路的 tile 完全没调过**（tile 在 BF16 路径值 +17.4%）。潜力约 726，是当前最大的一块空白。见 [TUNING-v7 §5](TUNING-v7.md#5-fp8-与-qag能拿的已经拿到剩下的要改模型) |
| 上游 PR | 两个，边界已拆好，待确认贡献流程 |

---

## 参考

| 来源 | 说明 |
|---|---|
| [GB300 混元 3 训练文档](../../gpu/a4x-max/07-megatron-training/07f-gb300-hy3-295b-gke/README.md) | **架构 SSOT** + GB300 基线 |
| [GB300 混元 3 SFT 文档](../../gpu/a4x-max/07-megatron-training/07f-gb300-hy3-295b-gke/SFT.md) | Bridge 移植、权重转换、评测闭环 |
| [DeepSeek V3.2 TPU 训练](../DeepSeek-V3.2-Training/README.md) | MaxText 操作范式 + v7 MoE 踩坑 |
| [tencent/Hy3](https://huggingface.co/tencent/Hy3) | 官方权重与 config |
