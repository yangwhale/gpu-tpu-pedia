# 腾讯混元 3（295B-A21B）在 TPU 上预训练

把 Tencent Hunyuan 3 移植进 MaxText，并在 **TPU v5p** 和 **TPU v7（Ironwood）**
上跑通 80 层完整 295B-A21B 预训练。本目录收录方案、可复现的操作步骤、
以及三个平台的横向性能对照。

**MaxText 原生不支持混元 3。** 所需组件全都已经存在，只是分散在两个
decoder block 里 —— 本项目把它们接成一个新的 `decoder_block: "hunyuan3"`，
新写的代码只有装配逻辑，零新数学。

---

## 结果

| | v5p 256 chips | v7 Ironwood 64 chips | GB300 64 GPU（参照） |
|---|---|---|---|
| 参数量（框架报） | 298.786 B | 298.786 B（逐位一致） | — |
| 稳态 step | 63.2 s | 20.4 s | — |
| **TFLOP/s / 计算单元** | **161.0** | **445.1** | 854.0 |
| **MFU** | **35.07%** | 19.29% | 31.6% |
| 整机吞吐 | 265,588 tok/s | 205,314 tok/s | 399,488 tok/s |
| 状态 | ✅ 调优收敛，异地复现通过 | 🔄 仍在调，距 DSV3 水位差 1.38× | 已调优 |

三边跑的是同一个 295B-A21B、同样 BF16、同样合成数据、同样不开 checkpoint。
**按 per-chip / per-GPU 归一**，不要拿整机吞吐横向比（v5p 那列是 256 芯片，
另两列是 64 个单元）。

**v5p 的 35.07% 已经超过 GB300 的 31.6%**，而单卡算力只有它的 1/5.9 ——
256 芯片的 3D torus 加 SparseCore 集合通信卸载，把 MoE 那些碎通信藏得比
NVLink 域还干净。代价是要 256 张卡才换来 GB300 64 卡约七成的整机吞吐。

---

## 从哪里开始

| 你的情况 | 读这个 |
|---|---|
| **想在 v5p 上把它跑起来** | **[QUICKSTART-v5p.md](QUICKSTART-v5p.md)** —— 建池到出数，两条命令，含完整参数与基线。**这份从零验证过** |
| **想在 v7 (Ironwood) 上跑** | **[QUICKSTART-v7.md](QUICKSTART-v7.md)** —— 只写与 v5p 不同的部分：拿机器的方式、单位换算、当前水位与已知死路。**调优未完成** |
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
**TPU 上不要用 EP**（实测 EP=64 直接超显存 326 GB，纯 FSDP 反而最快）。

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
| v7 调优 | 当前 19.29%，目标是 DSV3 在同硬件的实测水位 26.6% |
| 上游 PR | 两个，边界已拆好，待确认贡献流程 |

---

## 参考

| 来源 | 说明 |
|---|---|
| [GB300 混元 3 训练文档](../../gpu/a4x-max/07-megatron-training/07f-gb300-hy3-295b-gke/README.md) | **架构 SSOT** + GB300 基线 |
| [GB300 混元 3 SFT 文档](../../gpu/a4x-max/07-megatron-training/07f-gb300-hy3-295b-gke/SFT.md) | Bridge 移植、权重转换、评测闭环 |
| [DeepSeek V3.2 TPU 训练](../DeepSeek-V3.2-Training/README.md) | MaxText 操作范式 + v7 MoE 踩坑 |
| [tencent/Hy3](https://huggingface.co/tencent/Hy3) | 官方权重与 config |
