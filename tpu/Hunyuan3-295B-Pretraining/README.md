> 🌐 **中文** | [English](README.en.md)

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
| **序列长度** | **8192** ⚠️ | 4096 | 4096 | 4096 |
| 参数量（框架报） | 298.786 B | 298.786 B | 298.786 B | — |
| 稳态 step | 63.2 s | **20.6 s** | 30.4 s | — |
| **TFLOP/s / 计算单元** | 161.0 | **662.3** | 598.8 ⚠️ | 854.0 |
| **MFU** | 35.07% | **28.71%** | 25.96% ⚠️ | 34.20% ⚠️ |
| **整机 token 吞吐** | 265,588 tok/s | 305,248 tok/s | **1,103,757 tok/s** | 399,488 tok/s |
| **每单元 token 吞吐** | 1,037 | **4,770** | 4,312 ⚠️ | 6,242 |
| 最优配方 | 官方 DSV3 v5p 配方 | `DP1×FSDP128`<br>**18 个 tile 配置参数** + pdbs 12 + **dvfs 7** | `DP2×FSDP256`<br>tile 注入 + pdbs 16 | — |
| 状态 | ✅ 调优收敛，异地复现 | ✅ **BF16 最高，2026-08-15 顶穿原定 600–630 区间** | ⚠️ 仍是旧 tile 入口的数，**未按新配方复测** | 已调优 |
| FP8（同硬件，**分母 4614**） | — | **670.8**（生产，`absmax` + pdbs11）／ **727.0**（峰值，`fixed` + pdbs13，⚠️ 伤收敛）<br><sub>均为 native + FSDP128、不开 QAG；旧 tokamax+QAG 677.0</sub> | 无 QAG 618 ⚠️ 未复测 | DSV3 官方 743.5 / 16.1% |

四边跑的是同一个 295B-A21B、同样 BF16、同样合成数据、不开 checkpoint。
**横向比只看「每单元 token 吞吐」和 MFU**，整机吞吐随规模走，不可比。

> ⚠️ **seq 长度不一致，v5p 那一列要打折看。**
> v5p 用 **8192**（`max_target_length=8192`，每步 token = 256 × 8 × 8192），
> v7 与 GB300 都是 **4096**。
> attention 的 FLOP 随序列长度平方增长，**长序列的 tok/s 天然更低** ——
> 所以 v5p 的 1,037 是被 8K 压过的数，**拿它当分母算出来的倍数是高估的**。
> **v7 ↔ GB300 这一对是严格对齐的（都是 4096），可以直接比。**

> ⚠️ **256 芯片那一列还是旧 tile 入口（tokamax 注入）的数。**
> 2026-08-15 在 64 芯片上把 tile 换成 18 个配置参数后拿到 662.3（原 629.9），
> 256 芯片**没有复测**，所以两列现在不是同一个配方，不要横比。
> 详见 [TUNING-v7 §3.4.5](TUNING-v7.md)。

> ⚠️ **GB300 那一列的 MFU 2026-08-31 改过分母。**
> 原先按 2,700 TFLOP/s/GPU 算得 31.60%，而 **2,700 是把 GB200 的数乘 1.2 推出来的，官方没有这个数**。
> GB300 NVL72 的官方 BF16 dense 峰值与 GB200 NVL72 相同，都是 **2,500**，所以同一个 854.0 对应 **34.20%**。
> 详见 [EXPERIMENT-LOG §7.2](EXPERIMENT-LOG.md)。

三条读法：

1. **v7 每芯片吞吐约为 v5p 的 4.68 倍**（4,854 vs 1,037，⚠️ **v5p 那个数是 8K 序列下测的，此倍数为高估**），MFU 却更低（28.71% vs 35.07%）——
   因为 v7 的 BF16 峰值是 v5p 的 5.03 倍，而 HBM 带宽只涨 2.64 倍、ICI 只涨 2 倍。
   **MFU 下降但绝对吞吐大涨，是硬件比例失衡的正常表现，不是调优没做好。**
2. **v7 的 FP8 单芯片吞吐是 GB300 的 77–84%，没有反超。**
   生产配方（`absmax`，pdbs 11，step 18.656 s）**4,830 tok/s/chip = 6,242 的 77.4%**；
   峰值配方（`fixed`，pdbs 13，step 20.342 s）5,235 = 83.9%，但它伤收敛、不可交付。调优前是 51.4%。
   ⚠️ **这一条 2026-08-31 改过，原因值得记下来**：此前写的是「已反超（7,308 vs 6,242，+17%）」，
   而那个 **7,308 是从已撤回的 1,014.8 TFLOP/s/chip 换算来的**。
   2026-08-16 撤回 1,014.8（[TUNING §3.4.10](TUNING-v7.md)，native 路径漏 all-gather、只算了 3/192 个专家）时，
   TFLOP/s 那一栏各处都划掉了，**却漏掉了由它派生出来的 tok/s** ——
   于是一个已经作废的数，换了个单位继续活了半个月。**撤回一个数，要连同从它算出来的数一起撤。**
   ⚠️ **口径提醒**：v7 这两个数是 FP8，6,242 是 GB300 的 BF16；tok/s 本身可比，但精度不同，别当成同精度对比。
3. **v5p 的 35.07% 略高于 GB300 的 34.2%** —— 256 芯片的 3D torus 加 SparseCore
   集合通信卸载，把 MoE 那些碎通信藏得比 NVLink 域还干净。
   ⚠️ **但只领先 0.9 个百分点，别当结论用。** GB300 那个数原先写的是 31.6%（分母 2,700），而 2,700 是把 GB200 的数乘 1.2 推的，官方没有这个数；改回官方口径 2,500 之后差距只剩 0.9pp，已经落在两边配置差异（global batch、序列打包）能解释的范围里。

> **v7 的 64 芯片和 256 芯片在同配方下跑出同一水位**（580 vs 580，峰值 HBM 一字节不差）；
> 64 芯片之所以最终反超，靠的是频率（`dvfs=7`）和精度（FP8+QAG）这两个与分片正交的维度。
> 256 芯片那多出的 3.3% 来自「有更宽的 FSDP 可选，能腾显存换更大 batch」，
> **不是规模本身**。详见 [QUICKSTART-v7 §4.2.1](QUICKSTART-v7.md#421-扩展性weak-scaling-100strong-scaling-有代价)。

---

## 从哪里开始

| 你的情况 | 读这个 |
|---|---|
| **想在 v5p 上把它跑起来** | **[QUICKSTART-v5p.md](QUICKSTART-v5p.md)** —— 建池到出数，两条命令，含完整参数与基线。**这份从零验证过** |
| **想在 v7 (Ironwood) 上跑** | **[QUICKSTART-v7.md](QUICKSTART-v7.md)** —— **直接给最优配方**，照抄就能跑到 BF16 599 / FP8+QAG 625。含 64 / 256 两个规模、端到端复现、单位换算 |
| **想知道 v7 那 599 是怎么调出来的** | **[TUNING-v7.md](TUNING-v7.md)** —— 445 → 599 的完整故事线，每步的原因 / 机理 / 收益；瓶颈判定、扩展性、HBM 模型；全部负面案例折叠在附录 |
| **抢卡之前想先知道会不会 OOM** | **[AOT-COMPILE.md](AOT-COMPILE.md)** —— 不占一张 TPU，在普通 CPU 上把训练步编出来，拿到每 device 显存分解、用 6 分钟 CPU 时间扫出最大可行 batch；含**端到端实战**（AOT 存产物 → 16 卡真训练加载，启动省 2.9×）、选机型扫描（**别买 highmem**）、DWS 换 pod 不丢卡的安全顺序、六个坑 |
| 想把**别的**模型移植进 MaxText | [MAXTEXT-PORTING-GUIDE.md](MAXTEXT-PORTING-GUIDE.md) —— 从本项目总结的通用范式，与 Hy3 无关 |
| 想知道某个数字/结论怎么来的 | [EXPERIMENT-LOG.md](EXPERIMENT-LOG.md) —— 完整实验档案，2600 行 |
| 只想跑脚本 | [maxtext-hunyuan3/](maxtext-hunyuan3/) —— `prep.sh` + `run.sh` + `aot.sh` |

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
| **真实数据集验证** | ✅ **已完成**：FineWeb-Edu 10BT 全集 ArrayRecord 上云，Grain 跑通 6 步真实语料预训练，稳态 **655.1 TFLOP/s/chip** (MFU 28.4%)，详见 [DATASET-PREPARATION.md](DATASET-PREPARATION.md) |
| HF 权重 → MaxText Orbax 转换 | 只跑吞吐基线可以不碰；要 SFT 必须做 |
| ~~v7 BF16 冲到 630~~ ✅ **已超额完成** | 2026-08-15 换 tile 入口后 64 芯片到 **662.3**（`pdbs=13` 时 666.6），顶穿原定 600–630 区间 |
| **256 芯片按新配方复测** | 256 芯片那列（598.8）仍是旧 tile 入口的数。64 芯片上换入口 +5.1%，256 芯片预期等幅但**没验** |
| **v7 FP8 + QAG** | ✅ **677**（tokamax+QAG）。⛔ 曾报 1,014.8，**已撤回** —— native 路径漏 all-gather、只算 3/192 个专家，补齐后 637.0，反而低于 677。根因与补丁见 [TUNING §3.4.10](TUNING-v7.md)。调参空间已见底，见 [§4.6](TUNING-v7.md#46-什么能调什么不能调--一张总表) |
| 上游 PR | 两个，边界已拆好，待确认贡献流程 |

---

## 参考

| 来源 | 说明 |
|---|---|
| [预训练真实数据准备](DATASET-PREPARATION.md) | 开源数据集选型、GCS 存储、Grain 集成与训练配置 |
| [GB300 混元 3 训练文档](../../gpu/a4x-max/07-megatron-training/07f-gb300-hy3-295b-gke/README.md) | **架构 SSOT** + GB300 基线 |
| [GB300 混元 3 SFT 文档](../../gpu/a4x-max/07-megatron-training/07f-gb300-hy3-295b-gke/SFT.md) | Bridge 移植、权重转换、评测闭环 |
| [DeepSeek V3.2 TPU 训练](../DeepSeek-V3.2-Training/README.md) | MaxText 操作范式 + v7 MoE 踩坑 |
| [tencent/Hy3](https://huggingface.co/tencent/Hy3) | 官方权重与 config |
