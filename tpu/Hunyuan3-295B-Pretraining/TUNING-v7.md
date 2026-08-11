> 🌐 **中文** | [English](TUNING-v7.en.md)

# 混元 3（295B-A21B）在 TPU v7 上的性能调优实践

> **BF16 从 445 调到 630 TFLOP/s/chip（MFU 19.3% → 27.3%）；FP8 + QAG + `dvfs=7` 到 674（64 芯片最高吞吐）。**
> 这份文档讲每一步改了什么、为什么有效、值多少 —— **以及大量「试了没用」的路，那部分同样重要。**
>
> 只想拿配方直接跑 → **[QUICKSTART-v7.md](QUICKSTART-v7.md)**，那里有可照抄的完整命令。
> 只想知道「什么能调什么不能调」→ 直接看 **[§4.6 总表](#46-什么能调什么不能调--一张总表)**。

---

## 0. 三分钟读完

### 一条线：445 → 674

**全部 64 芯片（16 节点 / 128 device）实测，每一行只加一件事。**

| # | 加了什么 | chip | tok/s/chip | 增量 | **累计** | 这一步的性质 |
|---|---|---|---|---|---|---|
| 0 | 起点：`megablox` + pdbs 8 | 457 | 3,290 | — | — | — |
| 1 | **tokamax `tile(512,2048,1536)`** | **543** | 3,912 | +18.8% | **+18.8%** | 补一个坏掉的默认值 |
| 2 | **pdbs 8 → 12** | **580** | 4,176 | +6.8% | **+26.9%** | 显存换吞吐 |
| 3 | **`--xla_tpu_dvfs_p_state=7`** | **630** | 4,536 | +8.6% | **+37.9%** | 锁频，零代价 |
| 4 | **FP8 + QAG**（pdbs 降到 7） | **674** | **4,854** | +7.0% | **+47.5%** | 换精度 + 通信减半 |

对照 **GB300 = 6,242 tok/s/GPU**（同 seq 4096，可直接比）：从 **52.7%** 走到 **77.8%**。

> ⚠️ 别横比 step 时间 —— 每行 batch 不同。唯一可横比的是 `tok/s/chip`。

**跟这条线一样重要的是它旁边的坟场**：SparseCore 卸载组 9 个 flag（±0）、
FP8 的 18 个 MaxText tile（±0）、`use_gmm_v2`（收益被 copy 吃掉 70%）、
EP=4（−71%）、`scan(unroll)`（无可用档位）…… 全部在 [附录 B](#附录-b负面案例总集)。
**做过而没收益的路，和有收益的路一样值钱** —— 它们决定了你不必再走一遍。

### 去哪找

| 想知道什么 | 去哪 |
|---|---|
| **能调什么、值多少** | [§4.6 一张总表](#46-什么能调什么不能调--一张总表) —— 只看一节的话看这个 |
| **怎么判断收益是真是假** | [§4.7 四条纪律](#47-判断收益是真是假的四条纪律) |
| **445 → 674 每一步值多少** | [§3 调优故事线](#3-调优故事线从-445-到-674) |
| FP8 / QAG 的完整机制与配方 | [§5.4.2](#542-qag先量化再通信一条被专家数卡死的路) |
| 所有试过没用的东西 | [附录 B](#附录-b负面案例总集)（默认折叠） |

**三个最该带走的结论：**

1. **`tokamax tile` 值 +17.4%，是全场最大单项** —— 但它本质是在补一个坏掉的默认值
   （kernel 查找表里没有 192 这一行），**不是常规调优，幅度不可外推到别处**。
2. **QAG（量化后再 all-gather）净收益 +15.6%，并省 4.5–11 G 显存** ——
   但它要求 `num_experts % FSDP == 0`。**专家数取 2 的幂是一条模型设计约束，
   而这个坑在 ≤32 芯片上完全暴露不出来。**
3. **调参空间基本见底**：tile、XLA flag、SparseCore 卸载、推 batch 四个方向
   共 8 格实验**无一正收益**。再往上要么加卡、要么改模型形状、要么写代码。

---

## 1. 水位与目标

### 1.1 现在在哪

实测，均为**完整 80 层**、seq 4096、合成数据、稳态取 step 4–7。
**MFU 分母：BF16 用 2307，FP8 用 4614** —— 两者不可直接比大小。

**A. Hy3 本体（192 experts）**

| 规模 | 配方 | step | **TFLOP/s/chip** | **MFU** | **tok/s** | tok/s/chip | 峰值 HBM |
|---|---|---|---|---|---|---|---|
| ⚡ **64 chip BF16 + `dvfs=7`** | `DP1×FSDP128` + tile + pdbs 12 + **`dvfs_p_state=7`** | **21.67 s** | **630** | **27.31%** | 290,331 | **4,536** | 91.94 G |
| 256 chip 极限 **BF16** | `DP2×FSDP256` + tile + pdbs **16** | 30.40 s | 599 | 25.96% | **1,103,757** | 4,312 | 92.33 G |
| 256 chip 推荐 **BF16** | `DP4×FSDP128` + tile + pdbs 12 | 23.56 s | 580 | 25.12% | 1,068,372 | 4,173 | 91.94 G |
| 64 chip **BF16** | `DP1×FSDP128` + tile + pdbs 12 | 23.54 s | 580 | 25.14% | 267,284 | 4,176 | 91.94 G |
| 256 chip **FP8**（无 QAG） | `DP2×FSDP256` + `fp8_full`+qwix | 29.46 s | 618 | 13.39%<sub>FP8</sub> | 1,139,022 | 4,449 | 92.80 G |
| 64 chip **FP8**（无 QAG） | `DP1×FSDP128` + pdbs 10 | 19.15 s | 594 | 12.87%<sub>FP8</sub> | 273,987 | 4,281 | 86.20 G |
| 🏆 **64 chip FP8+QAG+`dvfs=7`** | `DP2×FSDP64` + QAG + pdbs 7 + **dvfs 7** | **11.81 s** | **674** | **14.61%**<sub>FP8</sub> | 310,639 | **4,854** | 92.42 G |
| 64 chip FP8 + QAG（无 dvfs） | `DP2×FSDP64` + QAG + pdbs 7 | 12.73 s | 625 | 13.55%<sub>FP8</sub> | 288,222 | 4,503 | 92.42 G |
| 起点（2026-07-30） | `FSDP128` + megablox + pdbs 8 | 20.43 s | 445 | 19.29% | 205,313 | 3,208 | 74.20 G |

> ⭐ **64 芯片当前最优是 FP8 + QAG + `dvfs=7` 的 674**（4,854 tok/s/chip）。
> QAG 相对无 QAG 的 594 高 5.3% 且 batch 更小（7 vs 10），`dvfs=7` 再叠 +8.0%。
> 代价：FSDP 只能取 64（[§5.4.2](#542-qag先量化再通信一条被专家数卡死的路)）。

**B. 256 experts 探索（改了模型，仅供下一代设计参考）**

| 规模 | 配方 | step | **TFLOP/s/chip** | MFU | tok/s | tok/s/chip | 峰值 HBM |
|---|---|---|---|---|---|---|---|
| 64 chip **FP8 + QAG** | `DP1×FSDP128` + QAG + pdbs **11** | 19.42 s | **645** | 13.98%<sub>FP8</sub> | 296,955 | 4,640 | 91.56 G |

> ⚠️ **645 与上表不可横比** —— 专家数从 192 改成 256，参数量与 FLOP 口径都变了。
> 它回答的是「**下一代模型若把专家数设成 2 的幂，在 v7 上能跑成什么样**」：
> FSDP 随便选、batch 能上 11、HBM 还富余。**192 只能走窄 FSDP + batch 7。**

> **tok/s = device 数 × pdbs × seq ÷ step**；横向比只看 tok/s/chip。
> 参照：GB300 = **6,242** tok/s/GPU（**同为 seq 4096，可直接比**）；
> v5p 256 chips = **1,037** tok/s/chip（⚠️ **v5p 用的是 seq 8192，不同口径**）。
> **v7 每芯片吞吐是 GB300 单卡的 77.8%**（4,854 vs 6,242；调优前 51.4%）。
> 对 v5p 的倍数因序列长度不同而**偏高**，只作量级参考。

**BF16 目标 600–630 TFLOP/s/chip（26–27% MFU）：当前 630（`dvfs_p_state=7`，2026-08-11），落在区间上沿。**
**FP8 当前 674（192e，含 `dvfs=7`）/ 645（256e，未叠 dvfs）；除频率外调参空间已见底，见 [§4.6](#46-什么能调什么不能调--一张总表)。**

> **本文记号约定**（正文用简写，可抄的完整命令在 [QUICKSTART-v7 §0](QUICKSTART-v7.md#0-最优配方速查)）：
>
> | 简写 | 实际参数 |
> |---|---|
> | `pdbs` | `per_device_batch_size` |
> | `DP=N` | `ici_data_parallelism=N`（N=1 时写 `ici_fsdp_parallelism=-1` 即可） |
> | `FSDP=M` | `ici_fsdp_parallelism=M` |
> | `tile(a,b,c)` | tokamax `tile_m/tile_k/tile_n`，经 monkeypatch 注入（[§3.4.3](#343-修法6-行-monkeypatch)） |
> | per-chip | **v7 是 2 device/chip**，`= 日志 TFLOP/s/device × 2`；MFU 分母 BF16 **2307** / FP8 **4614** |

### 1.2 目标为什么是 600–630，不是 900

Ironwood 官方实测表（全部 bf16、synthetic、per-chip 口径）：

| 模型 | 类型 | chips | 序列 | TFLOP/s/chip | MFU |
|---|---|---|---|---|---|
| llama3.1-405b | **稠密** | 256 | 8192 | 1,261.4 | 54.7% |
| llama3.1-70b | **稠密** | 64 | 8192 | 1,207.1 | 52.3% |
| gemma4-31b | **稠密** | 64 | 8192 | 931.3 | 40.4% |
| **qwen3-235b-a22b** | **稀疏 MoE** | 256 | 4096 | **629.8** | **27.3%** |
| **deepseek-v3 671B** | **稀疏 MoE** | 256 | 4096 | **612.7** | **26.6%** |
| gpt-oss-120b | 稀疏 MoE | 256 | 8192 | 329.9 | 14.3% |
| **hunyuan3 295B（本项目）** | **稀疏 MoE** | **256** | **4096** | **599** | **25.96%** |

**900 以上全是稠密模型。** 稀疏 MoE 在 Ironwood 上的实际水位就是 600–630，
两个最接近 Hy3 的参照都在这条线上。差距是结构性的：

- 稠密模型每个 token 走同一套权重，GEMM 又大又规整，MXU 能吃满
- MoE 每层要路由、按专家分组重排、分组矩阵乘、还原。每个子块只有
  `tokens_per_expert × emb × moe_mlp` 那么大，且组大小随路由浮动，**编译期拿不到静态形状**
- 还要 all-gather / reduce-scatter 把 192 份专家权重摊开又收回

> Hy3 激活参数（21 B）比 DSV3（37 B）还少，结构也更简单（GQA 而非 MLA、192 专家而非 256），
> **没有理由跑不到同一水位** —— 差距来自配置，不是架构。

**关于 FP8 的预期管理**：同一张表里 DSV3 开 FP8 只涨 **+21.4%**（612.7 → 743.5），
而稠密的 llama3.1-405b 涨 **+52.8%**。**MoE 兑现不了 FP8 的两倍峰值** ——
时间大量花在路由、重排、通信和小块 GEMM 上，这些环节不吃 MXU 峰值，降精度帮不上。

> ⚠️ **报 FP8 的 MFU 一定要说明分母。** 同一个 743.5，对 FP8 峰值算 16.1%，
> 对 BF16 峰值算 32.2% —— 差一倍。

---

## 2. 瓶颈在哪：一次 trace 定的调

**在扫任何参数之前，先花一轮把时间花在哪搞清楚。** 这一步决定了后面所有实验的方向。

### 2.1 两个假说

v7 相对 v5p 的三个比值：

| | v5p | v7 | v7 / v5p |
|---|---|---|---|
| BF16 峰值 / chip | 459 TFLOPS | 2,307 TFLOPS | **5.03×** |
| HBM 带宽 | 2.8 TB/s | 7.4 TB/s | 2.64× |
| ICI / chip | 600 Gbps | 1,200 Gbps | 2.0× |

**算力涨 5 倍，喂数据的两条通道只涨 2～2.6 倍。** Roofline 拐点从 164 抬到 **312 FLOP/byte**，
想保持 compute-bound，算术强度要翻 1.9 倍。MoE 恰恰是算术强度最低的结构。

| | **H1：HBM 带宽卡住** | **H2：时间花在非 MXU 环节** |
|---|---|---|
| 说法 | MoE 算术强度低，权重读取吃满 7.4 TB/s | 路由、重排、通信、小块 GEMM 不吃 MXU |
| trace 判据 | HBM 带宽接近 7.4 TB/s | HBM 不高，但 collective op 占满时间轴 |
| 该走 | 提高算术强度 | 藏通信、修 kernel |

### 2.2 结论：H2 成立，通信占 57.3%

> [!warning] 本节结论只适用于 4 芯片 `pdbs 4`，2026-08-08 在 64 芯片生产配置上复测后已被限定
> 下面这份拆解来自 **4 芯片 / `pdbs 4`** 的首轮 profile。
> **在 64 芯片 / `pdbs 12` 的生产配置上，同样的测法得到的通信阻塞是 0.19%，不是 42%** ——
> 通信已经被计算完全掩盖。**结论相差两个数量级。**
> 本节内容作为「小 batch 下会发生什么」仍然成立，但**不能用来指导 64 芯片以上的调优方向**。
> 详见 [§2.4](#24-2026-08-08-复测64-芯片生产配置上通信已被完全掩盖)。

自用时间拆解（覆盖墙钟 98.2%）：

| 类别 | 自用时间 | 占墙钟 |
|---|---|---|
| **通信 · 等待 `-done`** | 1.299 s | **41.9%** |
| 计算 · MoE 分组矩阵乘（`gmm`/`tgmm`） | 0.723 s | 23.3% |
| **通信 · 同步集合** | 0.477 s | **15.4%** |
| 计算 · `fusion`/`dot` | 0.446 s | 14.4% |
| 计算 · attention（`splash`） | 0.046 s | 1.5% |
| 数据搬运 `copy`/`transpose` | 0.025 s | 0.8% |
| **通信合计** | **1.777 s** | **57.3%** |
| **计算合计** | **1.215 s** | **39.2%** |

**这直接解释了 MFU 为什么只有 19.29%**：真正在算的只占墙钟 39.2%，
`19.29% ÷ 39.2% ≈ 49%` —— **计算真跑起来时 MXU 利用率约五成，剩下全被通信吃掉。**
核有 **41.9% 的墙钟坐在 `-done` 上干等。**

![XProf：计算与通信等待交替出现](images/v7-xprof-comm-wait.png)

*放大到 30 ms 尺度：`gmm.18` → `all-gather...call-done` → `gmm.19` → `all-gather...call-done` → `gmm.20`。
**`call-done` 的块宽度与 `gmm` 计算块相当甚至更宽，两者交替出现** —— 算一段、停下来等一段。*

由此推论并被后续实测证实：

- **H1 不成立** —— 真在算的只占 39.2%，远没到把 HBM 打满的程度
- **FP8 预期要下调** —— 它只能加速那 39.2% 里的 dot 部分，**动不了占 57.3% 的通信**
- **主线是修 MoE kernel 和藏通信**，不是提算术强度

> 后来最大的一笔收益（tokamax tile，+17.4%）正好落在「MoE kernel」这一侧，
> 与这次判定一致。

### 2.3 怎么读 trace：自用时间拆解

![v7 首轮 XProf trace](images/v7-xprof-trace.png)

**先用官方工具（XProf）看，别自己画。** 自己解析 `trace.json` 只适合做批量统计。

抓 profile：

```bash
PLATFORM=v7 STEPS=25 bash run.sh prof \
  base_output_directory=gs://<bucket>/hy3prof \
  profiler=xplane skip_first_n_steps_for_profiler=8 profiler_steps=5 \
  profile_cleanly=True dump_hlo=True
```

| 参数 | 为什么 |
|---|---|
| `base_output_directory` 必须是 GCS | 默认 `/tmp` 是 pod 本地盘，pod 一结束全没了 —— **收不到 profile 最常见的原因** |
| `skip_first_n_steps_for_profiler=8` | step 0 含编译、1–2 是异步派发假读数 |
| `profile_cleanly=True` | 按步对齐；代价是这轮 step 偏慢，**别拿它的 MFU 当数** |

**分析的关键是「自用时间」（self time）**：容器 op（`while`，即 `scan_layers` 卷起的 80 层）
会把子 op 时间算进自己，必须从每个祖先里扣掉子 op 时长，只留真正独占核的部分。
这样分类可以直接相加配平。

```bash
gcloud storage cp "gs://<bucket>/<run>/tensorboard/plugins/profile/*/*.trace.json.gz" .
gunzip -c *.trace.json.gz > t.json      # 62 MB → 1.4 GB，留够磁盘
python3 maxtext-hunyuan3/analyze-trace.py t.json
```

> **本文分析用到的三份 profile**（XProf session，需 Google 账号）：
>
> | Profile | 用在哪 | Session |
> |---|---|---|
> | 4 芯片 `2x2x1` / 80 层 | §2.2 通信占 57.3% 的自用时间拆解 | http://xprof.corp.google.com/trace_viewer/chrisya-11640939633798411639 |
> | 16 芯片 `2x2x4` / 20 层（含 HLO dump） | 小规模筛选与 HLO 核对 | http://xprof.corp.google.com/trace_viewer/chrisya-18130551067782033931 |
> | **64 芯片 `4x4x4` / 80 层 / `pdbs 12` —— 生产配置** | §2.4 阻塞降到 0.19%、§2.5 MFU 三层账与 roofline | http://xprof.corp.google.com/trace_viewer/chrisya-5052706392869670409 |
>
> ⚠️ **这三份的原始 `.xplane.pb` 已不在任何 GCS 桶里**（2026-08-11 核查），
> 只剩会过期的 session。**下次抓 profile 一定要把 `xplane.pb` 本身归档**，
> 而不是只留一个 viewer 链接 —— 链接过期后无法重建。
>
> 上传大文件（1–2 GB）不要走带超时的包装层（常见 60 s 硬超时），直接调 binary，实测约 8 分钟：
>
> ```bash
> GOOGLE_CLOUD_PROJECT=<project> \
>   /google/src/head/depot/google3/cloud/tpu/tools/c2xprof/bin/c2xprof.par \
>   --gcs_path=gs://<bucket>/<run>/tensorboard/plugins/profile/<ts>/<host>.xplane.pb
> ```



<details>
<summary><b>⚠️ 这个结论我推翻过四次，四种错法全部留档（点开）</b></summary>

同一份 trace，先后得出过五个互相矛盾的结论：

| 版本 | 结论 | 错因 |
|---|---|---|
| ① | 重叠 0.000 s ⇒ 完全裸露 | 在**单条顺序 lane** 上算时间交集。40560 个事件里 16550 处交集**全是容器嵌套、部分交叉为 0** —— 这条 lane 上顶层 op 天然首尾相接，交集恒为 0。**是同义反复，不是测量** |
| ② | 80% 是同步阻塞 | 按 `.` 切 op 名，把 `-start`/`-done` 后缀切没了。真名是 `all-gather.382.cloned.1.call-done`，**后缀在最后一段** |
| ③ | 100% 异步 ⇒ 1.766 s 全是残余 | 方向对，但把"是异步的"当成"没藏住"，没量实际占用 |
| ④ | 83.4% 已被掩盖 | 统计每对 `start→done` 窗口内的计算量得 11.607 s，**但时间轴只有 3.100 s** —— 平均 4.5 个传输同时在飞，同一段计算被重复计了四五遍 |
| ⑤ **采用** | 通信 57.3% / 计算 39.2% | 自用时间拆解，覆盖 98.2%，可配平 |

**捉到 ④ 的是一个量纲矛盾**：如果通信真藏掉八成，MFU 不可能只有 19%。
**两个独立结论对不上时，先怀疑方法，别急着解释现象。**

四条通用教训：

1. **在单条顺序 lane 上算并发是同义反复** —— 先确认这条 lane 能不能重叠
2. **op 名字的后缀在最后一段**，别用 `split('.')[0]`
3. **"是异步的" ≠ "藏住了"**，也 ≠ "没藏住"，必须量实际占用
4. **窗口可以互相重叠** —— 任何「分项之和 > 总量」的结果都是重复计数的信号

**另一条自检**：任何按 op 求和的分析，先算「和 ÷ 时间轴跨度」。
第一次分析我得到 156.6%，就是父子重复计时的信号。

</details>

---

### 2.4 2026-08-08 复测：64 芯片生产配置上通信已被完全掩盖

§2.2 那份 profile 是 **4 芯片 / `pdbs 4`** 的首轮 debug 跑法。
生产配置是 **64 芯片 / `pdbs 12`**，两者差了 16 倍规模、3 倍 batch。
这次补抓了生产配置的 profile，**结论相差两个数量级。**

**baseline 复现无偏差**：稳态 `23.55 s/step`、`289.8 TFLOP/s/device` = **`579.7 TFLOP/s/chip`**，
与 [§3.8 汇总](#38-汇总)记录的 580 完全一致。

#### 测法：配对异步 collective 的 `-start` / `-done`

不看类别占比（那个会被采样和嵌套污染），只看一件事：
**TensorCore 在 `-done` 上一共阻塞了多久**。`-done` 是异步 collective 的等待端，
它的时长就是「核停下来干等」的时长。

| | 4 芯片 `pdbs 4` | **64 芯片 `pdbs 12`** |
|---|---|---|
| 步长 | 611.3 ms | 23,538.8 ms |
| 配到的 `start`/`done` 对 | 201 | 929 |
| **TensorCore 阻塞合计** | **260.1 ms** | **44.5 ms** |
| **阻塞占步长** | **42.5%** | **0.19%** |
| 单次 `-done` 时长 p90 | 7.886 ms | **0.163 ms** |
| 阻塞 ÷（阻塞 + 可重叠窗口） | 8.4% | 0.5% |

4 芯片那列的 **42.5%** 与 §2.2 用完全不同方法算出的 **41.9%** 吻合 —— **测法可信**。

#### 独立佐证：XProf `overview_page`

```
device_duty_cycle_percent = 99.9%      device_idle_time_percent = 0.1%
tc_idle_ms_average        = 12.06 ms   （步长 23,559 ms）
tc_infeed_ms_average      = 0.00 ms    （不是输入瓶颈）
host_idle_time_percent    = 100.0%     （host 全程空转）
sc_compute_ms_average     = 2,944 ms   sc_idle_ms_average = 13,769 ms
```

**TensorCore 一步只空闲 12 毫秒。**

#### 机制

权重 all-gather 搬的是**权重**，字节量**不随 batch 变**；计算量**随 batch 线性增长**。
`pdbs 4 → 12`，计算涨 3 倍、通信不变，于是计算总量反超通信，
调度器的重叠窗口足够把通信整个盖住。

按 §2.2 那份数据外推（计算 ×3、权重类通信不变、激活类通信 ×3）：

| | 计算 | 通信 | 谁是地板 |
|---|---|---|---|
| `pdbs 4` | 174 ms | 330 ms | **通信** —— 完美重叠也藏不完 |
| `pdbs 12` | 522 ms | 426 ms | **计算** —— 通信可以被完全藏住 |

**这解释了为什么同一个模型、同一套 flag，两个规模下的瓶颈判定完全相反。**

> [!important] 对调优方向的三条修正
> 1. **`CollectivePipeliner` / Continuation Fusion / 权重预取这一类实验，在 64 芯片上没有意义** ——
>    阻塞只剩 0.19%，没有东西可藏。
> 2. §2.2 的「FP8 预期要下调，动不了占 57.3% 的通信」**在 64 芯片上不成立**，见 §2.5。
> 3. **任何「通信占比」的结论都必须绑定 batch 和规模**。
>    小规模 debug profile 可以用来学工具、不能用来定方向。

---

### 2.5 MFU 只有 25% 到底浪费在哪（64 芯片，XProf op stats）

数据来自 XProf 的 `framework_op_stats`（单主机 8 个 TensorCore，设备侧 self-time 合计 **802.0 s**）。

#### 三层账

| 层 | 数值 | 来源 |
|---|---|---|
| v7 单 device BF16 峰值 | 1,153.5 TFLOP/s | 2,307 / chip ÷ 2 core |
| **实测时间加权平均 FLOP rate** | **422 TFLOP/s/device** | `Normalized FLOP Rate` 列按 self-time 加权 |
| **硬件利用率** | **36.6%** | 422 ÷ 1,153.5 |
| 报出的 MFU | **25.1%** | 290 ÷ 1,153.5 |
| **36.6% → 25.1% 的缺口** | **× 0.687** | **≈ 31% 的已执行 FLOP 不算模型 FLOP（重算）** |

#### 时间去哪了

| 类别 | 时间 | 占比 | 平均 TFLOP/s | 贡献 MFU |
|---|---|---|---|---|
| matmul（`pallas_call` + `dot_general`） | 347.9 s | **43.4%** | 527 | 19.8% |
| 高 FLOP 的其他 op（主要是 remat 融合） | 347.6 s | **43.3%** | 446 | 16.8% |
| 低 FLOP 胶水 op（`gather`/`reshape`/`sort`/`top_k`…） | 106.4 s | **13.3%** | ~0 | 0.0% |

按 `Operation Type`：
`Unknown` 42.0%（多为 remat 融合）、`pallas_call` 23.1%、`dot_general` 20.3%、
`gather` 3.2%、`mul` 1.8%、`reshape` 1.8%、`reduce_sum` 1.5%、`add_any` 1.3%、
`convert_element_type` 1.0%、`transpose` 0.8%。

#### Roofline：35.6% 的时间卡在 HBM，而且只差一口气

XProf 的 `Bound by` 列：

| | 时间 | 占比 |
|---|---|---|
| Compute-bound | 515.61 s | **64.3%** |
| **HBM-bound** | 285.89 s | **35.6%** |

主力 op 的实测算术强度落在 **261.6 / 299.9 / 534.3 / 614.4 FLOP/byte**，
而 v7 的 roofline 拐点是 **312**（[§2.1](#21-两个假说)）。
**261 和 300 这两档正好压在拐点下面一点点** —— 差一点就能从内存受限翻成算力受限。

#### remat 的代价

- **65.9% 的设备时间**，op 名字里带 `checkpoint` / `remat`
- **反向 ÷ 前向 = 2.60**（`transpose(jvp())` 544.0 s vs `jvp()` 209.5 s），
  不开重算的理论值约 2.0，多出的 0.6 就是反向里重算的前向
- 与「× 0.687」互相印证

#### 结论：FP8 的优先级要往上调，不是往下调

64 芯片上通信只占 0.19%，FP8 **同时命中三个损失来源**：

| 损失来源 | 占比 | FP8 怎么打中 |
|---|---|---|
| HBM-bound | 35.6% 的时间 | 字节减半 ⇒ 算术强度翻倍，261/300 那两档**越过 312 拐点** |
| remat 重算 | 31% 的 FLOP | HBM 省出来 ⇒ 可以少 remat |
| 权重通信 | 已被掩盖 | 有效果但价值最小（本来就没在关键路径上） |

**⇒ 在 64 芯片生产配置上，FP8 + QAG 是当前唯一同时命中三个损失来源的杠杆。**

> 2026-08-08 的 E1（FP8 + QAG，`DP=2 × FSDP=64`、`pdbs 7`）**未跑成** ——
> 16 个 worker 全部卡在 TPU backend 初始化，
> `MaybeInitializeSliceBuilder failed: UNAVAILABLE`，无 Python 异常，一步未起。
> 失败发生在 `jax.devices()` 层，**早于 mesh 配置**，所以与 FP8 参数无关。待复跑。

---

<details>
<summary><b>§2.6 profile 的两个静默数据陷阱</b> —— 浏览器降采样 40 倍、trace.json 大规模下被截断</summary>

### 2.6 ⚠️ 大规模 profile 必须解析 `xplane.pb`，不能用 `trace.json`

这一节是本轮踩到的两个**静默**数据陷阱，都会让你拿到看起来正常、实际错得离谱的数字。

#### 陷阱一：浏览器里的 trace viewer 会降采样 40 倍

同一份 4 芯片 trace：

| 读法 | TPU:0 `XLA Ops` 事件数 |
|---|---|
| 浏览器 Catapult 内存模型 | **1,009** |
| 原始 `trace.json` | **40,560** |

按浏览器那份算出的「计算占 31%」是错的，真值 **38.4%** ——
漏掉的三千多个小 `fusion` 全在计算侧。
**任何定量结论都不能从浏览器 DOM 取数。**

#### 陷阱二：`trace.json` 在大规模下被静默截断

64 芯片这份：`trace.json.gz` 41 MB，而 `xplane.pb` 是 **1.78 GB**。

按秒统计 op 密度：

```
第 1–4 秒: 6504, 7832, 7531, 4132
第 5–24 秒: 0, 0, 0, 0, ... 0        ← 全空
```

**23.5 秒的步只有头 3.6 秒有数据**，XLA Ops 并集覆盖仅 **25.5%**；
文件末尾的 JSON 还直接断在半个字符串上（`Unterminated string`，约 99.9998% 处）。

**症状很隐蔽**：文件能解压、能解析，占比表也算得出来 ——
只是 `all-gather` 显示 26 ms、`collective-permute` 一个都没有。
**不去统计每秒事件密度就发现不了。**

#### 正确姿势：离线解析 `xplane.pb`（不需要浏览器 / SSO）

```bash
pip install --break-system-packages tensorboard-plugin-profile   # 实际装的是 xprof

python3 - <<'PY'
from xprof.convert import raw_to_tool_data as r
data, _ = r.xspace_to_tool_data(['<host>.xplane.pb'], 'framework_op_stats', {})
PY
```

可用 tool：`overview_page` / `op_profile` / `framework_op_stats` / `trace_viewer` / `kernel_stats`。

`framework_op_stats` 的表里直接带这四列，**§2.5 的全部拆解都出自它们**：

| 列 | 用途 |
|---|---|
| `Normalized FLOP Rate` | 算实测 FLOP 吞吐 → 硬件利用率 |
| `Operational Intensity` | 对 roofline 拐点，判内存受限还是算力受限 |
| `Measured Memory BW` | 交叉验证上一列 |
| **`Bound by`** | XProf 自己给的 Compute / HBM 判定 |

> 1.78 GB 的 xplane 解析一次约 5 分钟、峰值内存较高，建议放后台跑。

---


</details>

## 3. 调优故事线：从 445 到 674

**六步。每步只讲三件事：改了什么、为什么有效、值多少。**

### 3.1 第一步 +12.8%：换 batch / 序列口径

**改动**：`seq 8192 / pdbs 4` → `seq 4096 / pdbs 8`。总 token 数不变。

**为什么有效**：token 数相同，但 attention 的计算量随 seq **平方**增长，
而 MoE 部分只随 token 数线性增长。缩短序列把时间从 attention 挪回 MoE 主路径。

**收益**：吞吐 +12.8%（TFLOP/s 只 +0.9% —— 说明省的是时间不是算力）。

> **两次验证，结论从「打平」变成「短序列明确更优」：**
>
> | 时期 | 配置对 | 结果 |
> |---|---|---|
> | megablox 时代（256 芯片） | seq 8192/pdbs 4 = 451 vs seq 4096/pdbs 8 = 453 | 打平 |
> | **tile 之后（64 芯片）** | **seq 8192/pdbs 6 = 561 vs seq 4096/pdbs 12 = 580** | **短序列 +3.3%** |
>
> **tile 优化放大了短序列的优势。** 原因推测：tile `(512, 2048, 1536)` 是在
> seq 4096 的形状上扫出来的，换 seq 8192 后 MoE 的 `m` 结构变了，同一个 tile 不再匹配。
> ⇒ **这一维度不但到头，而且现在明确不该动。**

### 3.2 第二步 +6.6%：调度器 flag 组

**改动**：加 4 个 XLA flag。

```
--xla_tpu_enable_latency_hiding_layer_scheduler=true
--xla_tpu_scheduler_percent_shared_memory_limit=150
--xla_tpu_enable_layer_scheduler_for_dependent_collectives=true
--xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=false
```

**为什么有效**：它们改的是**通信和计算怎么重叠**。§2 已经测出通信占 57.3% 且
41.9% 的墙钟是干等，所以让调度器把集合通信塞进计算的空隙里，直接命中瓶颈。

**收益 +6.6%。这是所有 XLA flag 里唯一值钱的一组。**

### 3.3 一条重要的否定结果：SparseCore 卸载组 ±0

**9 个 SparseCore 集合通信卸载 flag，在 v5p 上值 4.07 pp（13%），在 v7 上收益是 0。**

这条否定结果**比很多正收益更有价值，因为它指出了瓶颈的性质**：

> SparseCore 卸载改的是**通信在哪执行**，调度器改的是**通信和计算怎么重叠**。
> 前者无效说明**通信不是"太慢"**；后者有效说明**通信"没藏住"**。

两个衍生教训：

- **同一个开关在两个平台上可以反号。** `sa_use_fused_bwd_kernel` 在 v5p 要 `False`、
  v7 要 `True`；SparseCore 卸载组 v5p +4.07 pp、v7 ±0。**别把一个平台的结论直接搬到另一个。**
- **不要把「这一组没用」外推到「同类的下一组也没用」。** 我当时正是这么推的，
  差点跳过调度器组 —— 而它是 +6.6% 的那一组。**消融的纪律是每一组都要真跑。**

那 9 个里 **8 个确实可以删**，但第 9 个（`--xla_tpu_enable_sparse_core_collective_aggregator`）
是层调度器的**硬依赖**，删了直接
`INVALID_ARGUMENT: Latency hiding layer scheduler requires sparse core collective aggregator`。
**裁剪 flag 要成组，不能逐个删。**

### 3.4 第三步 +17.4%：tokamax tile ← 最大单项

这是整轮调优收益最大的一步，也是唯一一个**"改一个数字换 17%"**的地方。

#### 3.4.1 现象：不设 tile 会慢 12.4 倍

| 配置 | step（单节点 4 芯片 / 6 层） | TFLOP/s/device |
|---|---|---|
| megablox（默认路径） | 1.321 s | 182.0 |
| tokamax **默认 tile**（回退 `128³`） | **17.955 s** | 13.4 |
| tokamax `tile(512, 2048, 1536)` | **1.220 s** | **197.2** |

**未命中调优表的代价 = 12.4×。** 早期我们把这个现象记成了「`use_tokamax_gmm` 死锁」，
因为它慢到触发看门狗，报 `stalled chips [7]` 连 step 0 都跑不完。

#### 3.4.2 根因：kernel 库的查找表里没有 192 这一行

tokamax 的 TPU `ragged_dot` 按 `(m, k, n, 专家数, 是否量化)` 查三张硬编码 tile 表，
查不到就退回 `Config()` 默认值：

```
GMM_TILING_TUNED_LUT: 28 条，专家数取值 = [16, 128, 256]
  (524288, 4096, 1536, g=128) -> tile (256, 4096, 1536)
默认 Config = tile_m=128, tile_k=128, tile_n=128
```

**Hy3 是 192 个专家。矩阵尺寸跟表里那条一模一样，只有分组数不同，于是全部 miss。**

| | tile | grid 块数 |
|---|---|---|
| 表里调优过的（g=128） | (256, 4096, 1536) | 2048 × 1 × 1 = **2,048** |
| 实际退回的默认值 | (128, 128, 128) | 4096 × 32 × 12 = **1,572,864** |

**768 倍的块数**，每块独立 DMA。慢三个数量级 → 看门狗判 stall。这就是"死锁"的真面目。

> **注意三条 GMM 路径是三套独立实现**，最容易搞混：
>
> | 配置 | 实际 kernel | 吃 `w{i,o}_tile_*` | 查 LUT |
> |---|---|---|---|
> | 默认 | megablox v1（JAX Pallas 原生） | ✅ | 否 |
> | `use_tokamax_gmm` | tokamax v1 `ragged_dot` | ❌ **不吃** | ✅ |
> | `use_gmm_v2` | tokamax v2（fork 进 MaxText） | ✅ | 否 |
>
> **`use_tokamax_gmm` 不吃 MaxText 的 tile 参数** —— 这就是为什么要 monkeypatch。

#### 3.4.3 修法：6 行 monkeypatch

MaxText 不暴露 tokamax 的 tile，直接改它的启发式配置：

```python
# tkcfg.py —— 在 import train 之前 exec
import os, dataclasses
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu as P
_TM, _TK, _TN = (int(os.environ[k]) for k in ("TK_TM", "TK_TK", "TK_TN"))
_orig = P.PallasMosaicTpuRaggedDot._get_heuristics_config
def _patched(self, ba):
    c = _orig(self, ba)
    k, n = ba.arguments["rhs"].shape[-2], ba.arguments["rhs"].shape[-1]
    return dataclasses.replace(c, tile_m=_TM, tile_k=min(_TK, k), tile_n=min(_TN, n))
P.PallasMosaicTpuRaggedDot._get_heuristics_config = _patched
```

```bash
TK_TM=512 TK_TK=2048 TK_TN=1536 python3 -c "
exec(open('tkcfg.py').read())
import runpy; runpy.run_module('src.maxtext.trainers.pre_train.train', run_name='__main__')
" ... megablox=True use_tokamax_gmm=True
```

验证生效：日志有 `[tkcfg] patched`，且**不再出现** `Autotuning cache miss` 带来的 10 s+ step。

> 长期正解是跑官方 autotune 生成 cache 条目；注入是验证手段，但足以拿到全部收益。

#### 3.4.4 tile 值怎么选：三条规律

256 芯片实测（基座 `DP4×FSDP128`，pdbs 8）：

| tile (m, k, n) | chip | vs megablox |
|---|---|---|
| **(512, 2048, 1536)** | **532** | **+17.4%** 🏆 |
| (1024, 2048, 1536) | 512 | +13.0% |
| (512, 1024, 1536) | 499 | +10.2% |
| megablox 基线 | 453 | — |

1. **`tile_n` 必须 `= base_moe_mlp_dim`（1536）。** 1024 不整除直接
   `AssertionError: v=1536 bv=1024 s=1536`；512 能整除但切三刀，反而比 bf16 基线还慢。
2. **`tile_k = 2048` 是甜点。** 不是抄表的 1024，也不是越大越好 —— 4096 直接 OOM。
3. **`tile_m` 不随 `m` 走 —— 这条经过三次独立验证。** 表内规律是 `tile_m` 随 `m` 线性增长
   （`m=131072→512`、`524288→1024`），但 **512 在所有测过的 `m` 上都最优**：

   | `m` | 来源 | tile_m=512 | tile_m=1024 |
   |---|---|---|---|
   | 262144 | pdbs 8 | **532** | 512（−3.8%） |
   | 393216 | pdbs 12 | **580** | 569（−1.9%） |
   | 393216 | pdbs 12（256 芯片批次） | **580** | 567（−2.2%） |

   **抄表是好起点，不是终点。**

   > 🔬 **2026-08-05 专门做了一轮证伪实验。** 我的假设是「`m` 从 262144 涨到 393216，
   > 按表内规律 `tile_m` 该升到 1024」。结果 **1024 全面更差，而且 `tile_k`/`tile_n`
   > 一旦偏离 (2048, 1536) 就掉 7–9%**：
   >
   > | tile | chip | Δ |
   > |---|---|---|
   > | **(512, 2048, 1536)** | **580** | 基线 |
   > | (1024, 2048, 1536) | 569 | −1.9% |
   > | (1024, 1024, 1536) | 537 | −7.4% |
   > | (1024, 1536, 1024) | 529 | −8.8% |
   > | (1024, 4096, 1536) | **崩** | Mosaic kernel 拒绝该组合 |
   > | (2048, 2048, 1536) | **崩** | 同上 |
   >
   > ⇒ **`(512, 2048, 1536)` 是个跨 `m` 稳定的最优点，不需要随 batch 重调。**
   > 这对实践很重要：换 `pdbs` 时不必重扫 tile。

**几乎不吃显存**（75.33 vs 74.20 G，+1.1 G）—— 本轮性价比最高的一项。

<details>
<summary>完整 tile 扫描（单节点 15 组，点开）</summary>

单节点 4 芯片 / 6 层 / pdbs 4，TFLOP/s/device：

**固定 `tile_n=1536`，扫 `tile_m × tile_k`**

| tile_m \ tile_k | 512 | 1024 | **2048** | 4096 |
|---|---|---|---|---|
| **256** | 176.5 | 185.0 | **197.2** | 188.9 |
| **512** | 184.6 | 189.4 | **197.2** 🏆 | OOM |
| **1024** | 181.8 | 180.6 | 186.8 | OOM |

**固定 `tile_m=512, tile_k=2048`，扫 `tile_n`**（只取 1536 的因数）

| tile_n | 256 | 512 | 768 | **1536** |
|---|---|---|---|---|
| TFLOP/s | 167.7 | 186.9 | 191.2 | **197.2** 🏆 |

**查表脚本**（换成你自己的 k / n）：

```python
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu as P
for name in ("GMM_TILING_TUNED_LUT", "TGMM_TILING_TUNED_LUT"):
    for k, v in sorted(getattr(P, name).items()):
        if k[1] == 4096 and k[2] == 1536:
            print(name, k, "->", v)
```

</details>

### 3.5 第四步 +9.0%：把 batch 推到 12

**改动**：`per_device_batch_size` 8 → 12。

**为什么有效**：更大的 batch 摊薄每步的固定通信开销 —— 权重 all-gather 的量不随 batch 变，
但能摊到更多 token 上。这也是 §3.1 那一步（+12.8%）的同一机理。

**为什么以前做不到**：pdbs=12 在旧配置下 OOM。**是 FSDP 摊薄腾出了空间** ——
详见 [§4.2 的 HBM 模型](#42-hbm-两参数模型不用撞-oom-就能算-batch-上限)。

**收益**（256 芯片，`DP4×FSDP128` + tile）：

| pdbs | step | chip | MFU | 峰值 HBM | 余量 |
|---|---|---|---|---|---|
| 8 | 17.12 s | 532 | 23.04% | 75.33 G | 19.4 G |
| 10 | 20.17 s | 564 | 24.45% | 84.06 G | 10.7 G |
| **12** | **23.56 s** | **580** | **25.12%** | **91.94 G** | **2.8 G** |
| 14 | — | OOM | | 预测 100.8 G | — |

**pdbs 8 → 12 值 +9.0%**，到 12 就贴上限了。

### 3.6 第五步 +3.3%：FSDP 加宽换 batch（仅 ≥ 256 芯片）

**改动**：`DP4×FSDP128` → `DP2×FSDP256`，同时把 pdbs 从 12 一路推到 16。

**为什么有效**：FSDP 宽度翻倍 → 每卡静态分片（权重 + 优化器 + 梯度）**减半**，
省出 12.84 G → 够再加四个 pdbs。**用一点点通信效率换显存，再用显存换 batch。**

**收益**：599 vs 580，**+3.3%**。

| 配方 | FSDP | pdbs | chip | HBM |
|---|---|---|---|---|
| `DP4×FSDP128` | 128 | 12 | 580 | 91.94 G |
| `DP2×FSDP256` | 256 | 12 | 569 | 78.27 G |
| `DP2×FSDP256` | 256 | 14 | 585 | 89.56 G |
| **`DP2×FSDP256`** | 256 | **16** | **599** | **92.33 G** |

> 注意 pdbs 12 那一行：**同样 pdbs 下 FSDP=256 反而比 FSDP=128 慢 1.9%**（569 vs 580）。
> 加宽 FSDP 本身是**亏**的，它的价值完全在于**腾出的显存能换更大的 batch**。

> **64 芯片吃不到这一步** —— 它只有 128 个 device，没有 `FSDP=256` 可选。
> **所以 580 是 64 芯片的物理天花板。**

### 3.7 并行度怎么切：可用区间只有 FSDP ∈ [128, 256]

512 device 的五种切法（pdbs 固定 8，megablox）：

| 切法 | chip | 峰值 HBM | 判定 |
|---|---|---|---|
| `DP1 × FSDP512` | 404 | — | ❌ 掉 11% |
| `DP2 × FSDP256` | 450 | 61.36 G | ⭕ |
| **`DP4 × FSDP128`** | **453** | 74.20 G | ✅ |
| `DP8 × FSDP64` | OOM | — | ❌ |
| `DP16 × FSDP32` | OOM | — | ❌ |

**两侧都有墙：**

- **往宽走掉 11%** —— 摊得越薄，每次集合通信的分片越碎，单次有效载荷不够摊薄固定开销
- **往窄走直接 OOM** —— FSDP 减半，每卡静态分片翻倍。FSDP=64 的静态部分约 51 G，
  加上激活越过 94.74 G

**默认规律：把 FSDP 宽度固定在 128，多出来的 device 全给 DP。**
64 芯片正好 128 device 所以 `DP=1`，256 芯片 512 device 所以 `DP=4`。

**EP（专家并行）不要用。** TPU 的 ICI 是 3D torus，AllToAll 要多跳转发，
不像 GPU NVLink 那样是 full mesh。16 芯片实测 EP=4 是 **−71.36%**，
在 torus 上换更大规模没有翻正的物理依据。

### 3.8 汇总

| # | 改动 | 机理 | chip | 累计 |
|---|---|---|---|---|
| 0 | 起点（2 个 XLA flag，seq 8192 / pdbs 4） | — | 405 | — |
| 1 | seq 4096 / pdbs 8 | attention 随 seq 平方增长，缩短它把时间挪回 MoE | 445 | +9.9% |
| 2 | 调度器 flag 组（4 个） | 让集合通信塞进计算空隙 | 453* | +11.9% |
| — | SparseCore 卸载组（9 个） | 改「通信在哪执行」，而瓶颈是「没藏住」 | ±0 | — |
| 3 | **tokamax `tile(512,2048,1536)`** | **绕开 LUT miss，grid 块数从 157 万降到 2 千** | **532** | **+31.4%** |
| 4 | pdbs 8 → 12 | 大 batch 摊薄固定通信开销 | 580 | +43.2% |
| 5 | `DP2×FSDP256` + pdbs 16（≥256 芯片） | FSDP 加宽省 13 G 显存，换四个 pdbs | **599** | **+47.9%** |
| 6 | **`--xla_tpu_dvfs_p_state=7`**（2026-08-11） | **锁最高频率档，与分片正交** | **630**<sub>64c</sub> | **+8.6%** |
| 7 | **FP8 + QAG + dvfs 7**（2026-08-11） | 字节减半 ⇒ 越过 roofline 拐点 + 通信减半 | **674**<sub>64c</sub> | **+47.5%**<sub>对 457</sub> |

\* 换了一批机器，与前几行不同批次，绝对值不可直接比；同批次内的对照见 [附录 A](#附录-a全部消融数据)。

> 第 6、7 步是在 **64 芯片**上测的（对照基线 580 / 624.1），前五步主要在 256 芯片上。
> 两个规模在同配方下逐点吻合（[§4.1](#41-扩展性weak-scaling-100strong-scaling-掉-11)），所以链路可以接上，
> 但**绝对值要认准它测于哪个规模**。

---

## 4. 可复用的方法论结论

### 4.1 扩展性：weak scaling 100%，strong scaling 掉 11%

**同一批 512 device、同一份代码，两种切法差 11% —— 差别只在「加卡时有没有同时加 batch」。**

| 扩展方式 | 切法 | 每卡工作量 | global batch | per-chip | 相对 64 芯片 |
|---|---|---|---|---|---|
| **Weak scaling** | `DP=4 × FSDP=128` | 不变（pdbs 12） | **4×** | **580** | **100.0%** |
| **Strong scaling** | `DP=1 × FSDP=512` | 缩到 1/4 | 1× | 404 | 89% |

64 芯片同配方 580，256 芯片也是 580，**per-chip 一点没掉**。

**为什么 DP 方向是免费的**：`DP=4 × FSDP=128` 就是四个独立的 64 芯片作业，
组内每层做两次 FSDP 集合通信（80 层 = 160 次），**组间整个 step 只有一次梯度 all-reduce**：

```
每卡梯度分片（bf16, FSDP=128） = 590 GB / 128 ≈ 4.6 GB
ring all-reduce 传输量          = 2(p−1)/p × 4.6 = 6.9 GB   (p = 4)
v7 ICI 单芯片双向               = 1,200 GB/s
理论耗时 ≈ 12 ms  →  占 step 23.54 s 的 0.05%
```

按 1/6 带宽利用率保守估计也只有 35 ms（0.15%），且可与反向传播尾部重叠。
**通信量差两个数量级，这就是「DP 便宜、FSDP 贵」的根本原因。**

**为什么 strong scaling 掉 11%**：`FSDP=512` 把同一份权重摊到 4 倍卡上，
每卡分片缩到 1/4 —— **集合通信次数没变，每次有效载荷只剩 1/4**，
固定开销（同步、启动延迟、3D torus 多跳转发）摊不动。

> **结论：加卡的时候必须同时加 batch。**

**三条边界条件**：

1. **只测到 DP=4。** ring all-reduce 传输量 `2(p−1)/p × N` 在 p 增大时趋近常数 `2N`，
   DP=8/16 预期仍接近 100%，但**这是推论不是实测**。
2. **单 slice 内结论。** 512 device 全在一个 `4x8x8` slice 走 ICI；
   **跨 slice 的 DP 走 DCN，带宽低一个数量级以上，不能外推。**
3. **前提是每卡工作量不变。** 保持 global batch 去扩规模就退化成 404 那一行。

### 4.2 HBM 两参数模型：不用撞 OOM 就能算 batch 上限

用**同基座的两个实测点**解 `HBM = 静态 + 斜率 × pdbs`：

```
DP4×FSDP128:  74.20 G @ pdbs 8 ，91.93 G @ pdbs 12
              → 静态 38.7 G ，斜率 4.43 G / pdbs
DP2×FSDP256:  静态 25.9 G（FSDP 翻倍，静态减半），斜率相同
```

| 基座 | pdbs 8 | pdbs 10 | pdbs 12 | pdbs 14 | pdbs 16 |
|---|---|---|---|---|---|
| `DP4×FSDP128` 预测 | 74.2 | 84.1 | 91.9 | 100.8 | 109.6 |
| `DP4×FSDP128` **实测** | **74.20** | **84.06** | **91.94** | — | **OOM** ✅ |
| `DP2×FSDP256` 预测 | 61.4 | 73.5 | 79.1 | 87.9 | 96.8 → 判 OOM |
| `DP2×FSDP256` **实测** | **61.36** | — | **78.27** | **89.56** | **92.33** ❌ **预测错** |

**近端插值很准**（pdbs 10 预测 83.0 / 实测 84.06，误差 1.0 G；pdbs 14 预测 87.9 / 实测 89.56，误差 1.7 G），
**远端外推会系统性高估**。

实测 `DP2×FSDP256` 的逐段斜率：

```
pdbs  8 → 12 :  4.23 G / pdbs
pdbs 12 → 14 :  5.65 G / pdbs
pdbs 14 → 16 :  1.39 G / pdbs   ← 骤降
```

> ⚠️ **我在这个模型上错了两次，两次都是同一类错误：把次线性当线性。**
>
> 第一次：按「激活从 0 起线性」推，判 pdbs 12 @ FSDP128 要 98.5 G → OOM，**实测 91.93 G 跑通**。
> 修正为两参数模型后，第二次：判 pdbs 16 @ FSDP256 要 96.8 G → OOM，**实测 92.33 G 跑通**。
>
> 根因是 `remat_policy=custom` + `decoder_layer_input=offload` 下，
> XLA 会在显存压力上升时改变重算 / 卸载的调度，激活增长在高 batch 区间明显放缓。
>
> **正确用法**：
> 1. 只在**已测点附近 ±2 个 pdbs** 用它插值，**别外推超过 4 个 pdbs**
> 2. 预测「刚过上限」（94.74 G 附近 ±5%）的配置**仍然要实跑** —— 我两次判错都在这个区间
> 3. 预测「远超上限」（如 109.6 G，超 15%）的才可以直接排除 —— 这类两次都对

### 4.3 小规模筛选：适用边界（这条被修正过两次）

**原始结论**（16 芯片 vs 64 芯片）：MFU 只低 7.7%，小规模可以调优。

**第一次修正**（2026-08-01）：**小规模能筛掉输家，不能选赢家。**

| | 16 芯片结论 | 64 芯片实测 |
|---|---|---|
| `remat_policy=full` | +1.22% | **−0.74%**（符号反转） |
| `shard_exp_on_fsdp` | +1.48% | **崩溃**（192 % 128 ≠ 0） |
| 删 8 个 SparseCore flag | −0.01% | −0.00%（一致） |
| `use_2d_fsdp_sharding` | −11.73% | 未测（已否决） |

**第二次修正**（2026-08-04，64 vs 256 芯片对照）：**上面那条太严了，要按「改动是否改变分片形状」分类。**

| 改动类型 | 例子 | 跨规模可传递？ |
|---|---|---|
| **不改变分片形状** | tokamax tile、pdbs | ✅ **完全可传递**。64 与 256 芯片同配方 580 vs 580，峰值 HBM 91.94 G vs 91.94 G，一字节不差 |
| **改变分片形状** | `remat_policy`、`shard_exp_on_fsdp`、FSDP 宽度 | ❌ 不可传递，且带整除约束的必须在目标规模验（192 % 32 = 0 但 192 % 128 ≠ 0） |
| 零收益 / 大幅负收益 | SparseCore 组、`use_2d_fsdp` | ✅ 可传递（用来排除是安全的） |

**还有一种情况是小规模「低估」赢家**：tokamax tile 在单节点上 +8.4%，在 256 芯片上 **+17.4%**。

> **修正后的操作流程**：
> 1. 不改分片形状的开关（tile、batch、kernel 参数）→ **可以在 16 节点上定，直接搬**
> 2. 改分片形状的（并行度、remat、sharding）→ **必须在目标规模验**
> 3. 任何情况下，「幅度」都要在目标规模复测一次

### 4.4 自身抖动只有 0.005%，±3% 判据是给跨批次用的

同一批 pod、同一配置连跑三轮（64 芯片 / `DP1×FSDP128` / tile / pdbs 12）：

| 轮 | step | chip |
|---|---|---|
| 1 | 23.5233 s | 580 |
| 2 | 23.5240 s | 580 |
| 3 | 23.5245 s | 580 |

**极差 1.2 毫秒 = 0.0051%。**

这条对读数纪律很重要，因为文档里同时存在两个判据，容易混：

| 场景 | 噪声量级 | 判据 |
|---|---|---|
| **同批 pod 内 A/B** | **0.005%** | **1% 的差异就是真的，单轮即可判，不必跑多轮取平均** |
| 跨集群 / 换一批机器 | 2.6–15% | ±3% 才算复现成功；07-30 与 08-01 两批基线差 15%（20.43 vs 17.43 s） |

> ⚠️ **别拿 ±3% 去否定同批次内 1–3% 的收益。** 那个 ±3% 是跨批次的复现判据，
> 用在同批消融上会把真实收益当噪声丢掉 —— 本轮 `DP2×FSDP256 + pdbs14`（585 vs 580，+0.9%）
> 当时我就误判成「噪声内」，实际它是真实差异。

### 4.5 占比大 ≠ 有空间

v5p 那边基线 trace 显示 MoE 的 `tgmm` 几乎填满采样窗口，我据此认定该在 MoE 上使劲。
结果：**MoE GMM 方向八九轮实验颗粒无收，attention 侧只碰了一个开关就 +3.33%。**

原因不难想 —— **megablox 已经是被人调过的最优路径**，
而 `sa_use_fused_bwd_kernel` 在 v5p 上默认还是关的、从来没人开过。

> **该找的是"还没被人调过的地方"，不是"耗时最多的地方"。**
> trace 告诉你时间花在哪，但不告诉你哪里还有空间 ——
> 后者要看「这块有没有被优化过」，那是代码和上游记录里的信息，不在 trace 上。

**这次 tokamax tile 拿到 +17.4%，正是这条的正面印证**：它慢不是因为算法差，
而是因为**查找表里少了 192 这一行** —— 一块完全没被人调过的地方。

---

### 4.6 什么能调、什么不能调 —— 一张总表

> **如果只看一节，看这节。** 下面每一行都有实测支撑，
> 「值多少」一列是本项目在 v7 上的实测幅度，不是推测。

**先给绝对水位，免得只看到百分比**（TFLOP/s/chip，完整 80 层）：

| | 起点 | 现在 | 靠什么 |
|---|---|---|---|
| **BF16**（192e，**64 chip**） | 445 | **630** | tile +17.4% / batch +9.0% / **`dvfs_p_state=7` +8.6%** |
| **BF16**（192e，256 chip） | 445 | 599 | tile +17.4% / batch +9.0% / FSDP 加宽 +3.3% |
| **FP8**（192e，64 chip） | 594 | **674** | QAG +5.3% / **`dvfs_p_state=7` +8.0%** |
| **FP8**（256e，64 chip，探索） | — | **645** | QAG + `cost_estimate_flops` +0.9% |

**⇒ 这三个数就是目前的天花板。** 下表 B/C 两类解释了为什么再往上要改模型或写代码。

**A. 确认有收益 —— 按性价比排序**

| 调什么 | 值多少 | 适用范围 | 注意 |
|---|---|---|---|
| **tokamax tile**（BF16 路径） | **+17.4%** | BF16 + `use_tokamax_gmm` | **本质是补一个坏掉的默认值**，见下方「⚠️」 |
| **`--xla_tpu_dvfs_p_state=7`** | **+8.6%** | 全部（v7 专用，v6e 不支持） | **零代价**：HBM 不变、配方不变。默认档是 3，显式写 3 等于没写。2026-08-11 三轮同 pod 对照 |
| **batch / 序列口径** | **+12.8%** | 全部 | 先把口径统一再谈优化 |
| **per_device_batch_size** | **+9.0%**（8→12） | 受 HBM 限 | 用 [§4.2](#42-hbm-两参数模型不用撞-oom-就能算-batch-上限) 的两参数模型预估，别撞 OOM |
| **调度器 flag 组** | **+6.6%** | 全部 | 要成组开，删掉依赖项会秒挂 |
| **`cost_estimate_flops_fwd/bwd=5e12`** | **+0.9%** | 用 splash attention 时 | 不改计算，只给调度器一个准确的 kernel 耗时估计，通信藏得更好。DSv3 官方 recipe 有，我们原先没设 |
| **FSDP 加宽换 batch** | **+3.3%** | 仅 ≥ 256 芯片 | 加宽本身慢 1.9%，赚的是腾出的 13 G 显存 |
| **QAG**（量化后再 all-gather） | **省 4.5–11 G 显存** | **仅当 `num_experts % FSDP == 0`** | 省的是显存不是时间，收益体现在能上更大 batch |

**B. 确认没收益 —— 别再花时间**

| 试过什么 | 结果 | 根因 |
|---|---|---|
| **FP8 的 MaxText tile 配置**（18 个参数） | **±0** | 源码证明：`use_tokamax_gmm=True` 且未开 `use_gmm_v2` 时 `tiling` 被整个丢弃，参数根本没到 kernel |
| **FP8 的 monkeypatch tile**（往大调） | **±0 或负** | `(512,2048,1536)` 已是局部最优。加 tile_k → VMEM OOM；加 tile_m → −2.5%。**最优点跨 dtype 稳定** |
| SparseCore 卸载组 | ±0 | 9 个 flag 删到只剩 aggregator，性能不变 |
| `use_gmm_v2` | 收益被 XLA 插的 copy 吃掉 70% | — |
| `scan(unroll=N)` | 无可用档位 | 2 撞 kernel 形状校验，10 要 274 G |
| 官方 `tokamax.autotune` | 不是 CLI，成本高于手调 | 见 [§5.4.1](#541-官方-autotune-调研结论2026-08-05) |
| 单开 `shard_exp_on_fsdp` | **静默失效**（不报错、不变快） | calibration 不是 `fixed` 时 `weight_gather_axes` 恒空 |
| **SparseCore 卸载 flag 组**（补齐 DSv3 那 27 个） | ±0 | 三个核心 offload flag 在 Ironwood 上**默认已是 True**；关掉互斥的 CF 后仍 ±0 —— 收益已被现有 `collective_aggregator` + `latency_hiding_layer_scheduler` 吃掉 |
| **`ici_expert_parallelism=2`（64 芯片）** | **−39.6%** | FSDP 被迫减半 + 24% 参数在 EP 轴纯复制 ⇒ batch 从 12 压到 6。**且 FP8 路径直接报 `custom_vjp` shape mismatch，不可用** |
| **`ici_tensor_parallelism=2`（64 芯片，FP8+QAG）** | **−25.3%** | 省 25.96 G 显存（92.42→66.46）是真的，但同 batch 慢 30.8%，把 batch 从 7 推到 12 只补回 8% |
| **MoE-only TP**（custom rule 摘掉 attention 的 tensor 绑定） | **−20.2%**（对默认 TP） | TP 切 attention 同时也是**计算分摊**，摘掉后每卡算全量 attention。参数占比 2% ≠ 计算占比 |
| **把 EP/TP 放到片内 chiplet 上** | **优化空间不存在** | 实测 `create_device_mesh` 默认就把宽度为 2 的非 data 轴映射到同芯片两个 core（64/64 行），已经在走 D2D 1.2 TB/s |
| **`FSDP=32 × TP=4`（切得更碎换 batch）** | **−77%**，且显存反弹 | 128 device 用完后只能割 FSDP；97% 参数在专家、靠 FSDP 切，减半的代价远超 TP 多切一刀。HBM 66.46 → 90.07 G |
| 整组照搬 DSv3 的 36 个 XLA flag | **HBM OOM** | 别人的 flag 是按别人的显存预算调的 |

**C. 被结构锁死 —— 改配置无解，只能改模型或框架**

| 想要什么 | 为什么拿不到 | 出路 |
|---|---|---|
| **192 experts 上宽松地开 QAG** | FSDP 只剩 64 一个选项（96 凑不出整数 DP、128 不整除），batch 被压到 7 才跑得动；**且加卡救不了**（分片厚度只由 FSDP 宽度决定） | 能跑但路窄（实测 625.4）。**下一代模型专家数取 2 的幂**，FSDP 就随便选 |
| **专家并行 EP** | TPU 是 3D torus，AllToAll 多跳，实测 **−71%** | 不要用，FSDP 优先 |
| **DSV3 的 batch split 调度器** | 它切换到 DeepSeek 专用手写 decoder，Hy3 是 GQA 直接 `KeyError: 'wq_a'` | 要几百行开发，属开发任务不是调优 |
| **DSV3 那种 MXU 利用率** | 它 `emb=7168`，我们 `4096`；矩阵越大 MXU 越划算 | 模型形状决定，调参抹不平 |

> ⚠️ **「tokamax tile 值 17.4%」这条最容易被误用。**
> 它之所以值这么多，是因为 tokamax 的查找表里**根本没有 192 这一行**、
> 掉进了极差的默认档（[§3.4.2](#342-根因kernel-库的查找表里没有-192-这一行)）——
> **那是在修一个坏掉的默认值，不是常规调优。**
> 我曾把这个数字平移到 FP8 路径当预期（推出 726），
> **实测零收益** —— 因为 FP8 走的是另一套 kernel，它的默认值没坏。
> **收益幅度不能跨 kernel 平移。**

#### 4.7 判断「收益是真是假」的四条纪律

这四条都是本项目栽过跟头之后总结的：

1. **阴性结果不能直接当结论 —— 要做反向判别。**
   改了参数性能没变，有两种可能：*改对了但没用*，和*压根没改到*。
   区分方法是**故意设一个明显糟糕的值**：性能掉 ⇒ 参数生效、默认已够好；
   性能不掉 ⇒ 参数根本没接到 kernel。这两种情况的后续动作完全不同。

2. **数值吻合不是机制证据。**
   我曾因「DSV3 官方 743.5 ≈ 我算的 746」就断定「连 DSV3 也没开 QAG」，
   查了官方 recipe 发现人家开着。**两个独立体系的数字撞在一起，
   在只有一个观测点时说明不了因果。**

3. **源码里有校验 ≠ 它在你这条路径上生效。**
   `pyconfig_deprecated.py` 明写着专家数整除校验，
   实跑却是 `shard_map` 在运行时才崩 —— 我们这个版本走的是 `types.py`。
   差别是**它要先编译一两分钟才失败，不是秒拒**。

4. **配对实验必须只差一个变量，而且要在设计时就数清楚。**
   我有一轮同时改了「开 QAG」和「FSDP 128→64」，
   +12.1% 说不清是谁的功劳，白跑。
   反而是顺手加的一组（同并行度、同 batch、只差 QAG 开关）
   给出了唯一干净的净收益。**列实验矩阵时先把每轮的 diff 写出来核对。**

> 💡 **补一条正向的**：**失败的轮次也是测量。**
> 四轮 OOM 看着全废，但 XLA 报的 `total memory required` 是精确值 ——
> 靠两次 OOM 的差值（115.19 → 104.11）才拿到「QAG 在完整 80 层省 11.08 G」
> 这个唯一的全模型规模测量。**别只看「跑通没」，报错里常有数。**

---

## 5. FP8 与 QAG：能拿的已经拿到，剩下的要改模型

**当前 618 TFLOP/s/chip，对 FP8 峰值（4,614）MFU 只有 13.4%。DSV3 官方是 743.5（16.1%），
我们落后 20.3%。**

> ⚠️ **报 FP8 数字必须写清分母。** 618 对 BF16 峰值算是 26.8%，对 FP8 峰值只有 13.4% ——
> 差一倍。**FP8 的数只能跟 FP8 的数比。**

### 5.1 FP8 的两条水位：618（无 QAG）→ 625（开 QAG）

| | BF16 | FP8 |
|---|---|---|
| 起点 | 445 | 594（64 chip，无 QAG） |
| 现在 | **599**（256 chip） | **625**（64 chip，开 QAG） |
| 主要来源 | tokamax tile **+17.4%** + batch | **QAG +5.3%**（同规模对比 594） |

> 🛑 **这一节原标题是「618 相当于 BF16 那边的 445 —— 起点，不是终点」，
> 意思是「FP8 完全没调过、空间还很大」。2026-08-05 实测把这个前提推翻了：**
> **FP8 走的仍是 tokamax，那 6 行 tile monkeypatch 一直在生效** ——
> 618 不是「未调优的起点」，它已经带着 BF16 调出来的最优 tile。
> 详见 [验证实验八 / 九](#542-qag先量化再通信一条被专家数卡死的路)。

**两个规模的 FP8 实测**（FP8 峰值 4614 为分母）：

| 规模 | 配置 | chip | 对 FP8 峰值 MFU | 同配置 BF16 | Δ | 峰值 HBM |
|---|---|---|---|---|---|---|
| 256 chip | `DP2×FSDP256` pdbs 16 | 618 | 13.39% | 599 | +3.2% | 92.80 G |
| 256 chip | `DP4×FSDP128` pdbs 12 | 608 | 13.18% | 580 | +4.8% | 94.35 G |
| **64 chip** | `DP1×FSDP128` **pdbs 10** | **594** | **12.87%** | 561 | **+5.9%** | 86.20 G |

> 64 芯片的 594 已经**超过 BF16 最优的 580（+2.4%）**，而且只吃 86.2 G。
> **「FP8 省下的显存再喂 batch」这条 2026-08-05 试过了** ——
> 开 QAG 后能到 pdbs 11（256e）/ 7（192e），**再往上 remat 已到底、batch 不能取小数**，
> 见 [验证实验十二](#542-qag先量化再通信一条被专家数卡死的路)。
>
> ⚠️ `pdbs 12 + FP8 + tile(512,2048,1536)` 在 64 芯片上**不是 OOM 而是 kernel 拒绝**
> （`MosaicTpuRaggedDot` 报错），见 [附录 B.2](#b2-崩溃--配置拒绝)。

**这两条赛道用的是不同的 kernel** —— BF16 走 `tokamax.ragged_dot`，FP8 走 `mblx.gmm`。

> 🛑 **我据此推断「FP8 的 tile 要单独调、而且一次都没扫过」—— 这个推断是错的。**
> `mblx.gmm` 在 `use_tokamax_gmm=True` 下**内部仍然回到 tokamax**，
> monkeypatch 照常生效；反倒是 MaxText 的 `w{i,o}_tile_*` 被丢弃。
> **两条赛道共用同一套 tile，不需要也无法分别调。**

粗算：若 FP8 路径的 tile 能拿到同量级收益，`618 × 1.174 ≈ 726`（FP8 峰值口径 MFU 15.7%），
距 DSV3 的 743.5 只差 2.4%。

> 🛑 **这个 726 已被实测证伪，保留在此只为记录推理链条。**
> 2026-08-05 按 DSv3 官方 recipe 补上 18 个 tile 参数后实测 —— **零收益**
> （step 19.5980 → 19.5980，见 [§5.4.2 验证实验六](#验证实验六补上-dsv3-的-18-个-tile--零收益一条核心假设被证伪)）。
> 这个外推的错误在于：**BF16 那边的 +17.4% 是在补一个坏掉的默认值
> （tokamax 查找表缺 192 这一行），不是常规调优收益，
> 把它平移到另一套 kernel（`mblx.gmm`）上当预期，类比不成立。**
>
> 另外 726 vs 743.5 的对比本身也不同源：**DSV3 的 743.5 是开着 QAG 拿到的**，
> 我们这个估计不含 QAG。

<details>
<summary><b>§5.2 为什么 FP8 的 tile 没调成</b> —— 两条 GMM 路径的源码追踪</summary>

### 5.2 为什么 FP8 的 tile 没调成：两条 GMM 路径

```python
# moe.py:1500
if self.config.use_tokamax_gmm:
    if self.config.quantization or self.config.use_gmm_v2:
        output = mblx.gmm(..., tiling=tiling, ...)   # ← FP8 走这条，吃 MaxText 的 w{i,o}_tile_*
    else:
        output = tokamax.ragged_dot(...)             # ← BF16 走这条，吃 monkeypatch
```

**开 FP8 等于换 kernel 路径**，BF16 上那个 6 行 monkeypatch（打 `PallasMosaicTpuRaggedDot`）
在 FP8 下一行都不执行。618（对 FP8 峰值 4614 是 13.4%）是「FP8 + MaxText 默认 tile」跑出来的，
而默认 `*_mlp_dim=1024` **不整除 `base_moe_mlp_dim=1536`**。

**调它的代价也不同：**

| | 改 tile 的方式 | 是否重编译 | 一轮耗时 |
|---|---|---|---|
| BF16（tokamax） | 运行时 monkeypatch | 否 | 6–8 分钟 |
| FP8（mblx.gmm） | MaxText 配置项，**进 HLO** | **是，缓存全失效** | **> 30 分钟** |

所以 BF16 能一晚上扫 15 组 tile，FP8 这边每个点都是半小时起。**这是它还没被调的直接原因。**


</details>

<details>
<summary><b>§5.3 DSV3 的 <code>use_batch_split_schedule</code> 我们用不了</b></summary>

### 5.3 DSV3 的 `use_batch_split_schedule` 我们用不了

`configure_quantization` 里只有开了它才返回完整的 `QwixQuantization`：

```python
if getattr(config, "use_batch_split_schedule", False) and config.quantization:
    if config.quantization == "fp8_full" and not config.use_manual_quantization:
        return QwixQuantization(...)
    return None
if config.use_qwix_quantization:
    return None        # ← 我们走这条
```

**但它在 Hy3 上直接 `KeyError: 'wq_a'`，四轮（含 BF16 对照轮）全部秒挂。**

根因：`models/deepseek_batchsplit_fp8.py` —— 文件名就写死了 deepseek。里面把
MLA 的七个权重逐个 all-gather：

```python
params["self_attention"]["wq_a"]["kernel"]
(wq_a, wq_b, q_norm, wkv_a, wkv_b, kv_norm, out)
```

**Hy3 是 GQA，没有 `wq_a` / `wq_b` / `wkv_a` / `wkv_b`。**

> 🔁 **这是本项目那条经验的第 11 次复现**：MaxText 里「这个模型该走哪条路」的判断
> 都是按模型家族写死的表。这次特别隐蔽 —— 配置项叫 `use_batch_split_schedule`，
> 听起来是通用调度策略，**实际是「切换到 DeepSeek 专用的手写 decoder」**，
> 从名字完全看不出来。

**推论**：DSV3 那 743.5 是**三样东西叠出来的** —— 量化本身、这个手写 MLA 调度器、
以及 **QAG**（2026-08-05 查官方 recipe 确认，见 [§5.4.2](#542-qag先量化再通信一条被专家数卡死的路)）。
**三者各占多少，目前分不开。** 拿它当 FP8 的唯一标尺要留余地。


</details>

### 5.4 下一步（按性价比排序）

| # | 做什么 | 依据 | 成本 |
|---|---|---|---|
| **1** | **扫 FP8 路径的 `w{i,o}_tile_*`** | 6 个 `*_mlp_dim` 从 1024 改成 1536 / 512（都整除 1536）。16 芯片上这一改值 **+8.25%** | 每轮 30 分钟 |
| **2** | **开 QAG** | 门槛已全部查清并有可执行配方（§5.4.2）。它动的是通信那 57.3%，**跟 tile 收益不重叠**；蚂蚁实测值 0.88× → 1.05× BF16 | 每轮 30 分钟，且需配收敛验证 |
| 3 | 按 BF16 最优映射设 tile | BF16 最优是 `(512, 2048, 1536)`；wi 的 `k` 是 emb(4096)、`n` 是 mlp(1536)，wo 反过来。映射关系要推一遍 | 同上 |
| 4 | 跑官方 autotune 生成 cache | 替代手调 | **已调研，见 §5.4.1 —— 优先级低于 1** |
| 5 | 给 GQA 写一份 batch split 实现 | 收益可能最大，但**是开发任务不是调优**，几百行 | 高 |

> 1 和 2 **应该分开做、分开记账**：tile 动计算，QAG 动通信，
> 同时改会让两边的收益混在一起分不出来。

#### 5.4.1 官方 autotune 调研结论（2026-08-05）

**可行，但不是「跑一条命令」，暂不推荐作为下一步。**

tokamax 暴露了 `tokamax.autotune`（`_src/autotuning/{autotuner,api,cache}.py`），
但它是**库级 API 不是 CLI**：`Autotuner.autotune(fn_factory, configs, *args, **kwargs)`
—— 候选 config 集合和真实形状的输入张量都要自己准备。

四个卡点：

1. **没有环境变量开关。** 全库只有 `TOKAMAX_NAME/VERSION/VERSION_INFO`，
   **不存在 `TOKAMAX_AUTOTUNE=1` 这种一键开启**，必须写胶水代码。
2. **要自己喂输入。** MaxText 训练流程里 lhs/rhs/group_sizes 是 sharded 的，取出来单独喂要额外工作。
3. **cache 按算子签名索引**（`ba.autotuning_cache_key`），生成后还要
   `get_autotuning_cache_overlay_state()` 挂进去才生效。
4. 🔴 **只对 tokamax 路径有用。而 FP8 走的是 `mblx.gmm`**
   （[§5.2](#52-为什么-fp8-的-tile-没调成两条-gmm-路径)）——
   **autotune 解决不了 FP8 那条路的 tile 问题。**

> ⇒ BF16 路径的 tile 已手调到位（`(512,2048,1536)` 经三个不同 `m` 验证稳定），
> autotune 顶多再榨个位数百分点，且解决不了最大空白（FP8）。**优先级低于直接扫 `w{i,o}_tile_*`。**

#### 5.4.2 QAG（先量化再通信）：一条被专家数卡死的路

> **起点是一个提问**：Chris 问「FP8 的时候，通信传的难道不是 FP8？」
> 我当时的假设是「通信不受 FP8 影响」—— 那假设错了一半，
> 顺着查下去挖出了 QAG，以及一条**直接指向下一代模型设计**的结论。

**QAG（Quantize-then-All-Gather）= 先把专家权重量化成 FP8，再做 all-gather，通信字节直接减半。**
净收益 **+15.6%**，并省 **4.5–11 G** 显存 —— 省下的显存换更大 batch，这才是它值钱的地方。

**要它真的生效，五个开关一个都不能少**（少任何一个都**不报错、也不变快**）：

```
use_qwix_quantization=True
quantization=fp8_full
shard_exp_on_fsdp=True                                # QAG 本体
weight_quantization_calibration_method=fixed,-224,224 # ⚠️ 必须带上下界
act_quantization_calibration_method=fixed,-224,224    # ⚠️ 同上
```

> **第 5 道锁最坑**：calibration 只写 `fixed` 是**非法值**，而框架不报错 ——
> 此时 `weight_gather_axes` 恒为空，`shard_exp_on_fsdp` 静默失效。
> 必须写成 `fixed,<lo>,<hi>` 三段式。

**还有一道整除锁**：`num_experts % ici_fsdp_parallelism == 0`。

| 专家数 | 可用的 FSDP 宽度 | 后果 |
|---|---|---|
| **192**（Hy3 本体） | 64 / 96 / 48 | FSDP=128 直接 `IndivisibleError`，只能走窄 FSDP，**batch 被压到 7** |
| 256（下一代若这么设计） | 任意 2 的幂 | FSDP 随便选，batch 能上 11，HBM 还富余 |

⇒ **「专家数取 2 的幂」是一条模型设计约束**，而这个坑在 ≤32 芯片上完全暴露不出来。

**结果**：64 芯片 `DP2×FSDP64` + pdbs 7 拿到 **625**；再叠 `dvfs_p_state=7` 到 **674**。


<details>
<summary><b>展开：这条路是怎么一步步摸出来的</b> —— 五道锁的发现过程、192 撞墙的确切位置、蚂蚁 ALModel 的先例，以及我自己写下又证伪的一条旁证</summary>

##### 先回答那个问题：默认情况下，通信传的确实是 bf16

`QwixDotGeneral.__call__` 只是包了一层 `dot_general_qt` ——
**量化发生在 dot_general 内部：算之前临时量化、算完输出 bf16**。
权重按 `weight_dtype=float32` 存、`dtype=bfloat16` 算。
**所以 FSDP 的 all-gather 传的是 bf16，一个字节都没省。**

这正是 [§5.1](#51-fp8-的两条水位618无-qag-625开-qag) 那个 Amdahl 天花板 746
必须建立在「通信一点不省」前提上的原因。

##### 但框架里有 QAG，只是我们没触发

`kernels/megablox/ops.py:190-197`：

```python
# QAG is only supported for following conditions
if use_tokamax_backend:
  if quantization_rule and quantization_rule.bwd_qtype:
    if quantization_rule.weight_calibration_method.startswith("fixed") \
       and isinstance(rhs, qpl.QArray):
      if weight_gather_axes:
        rhs_qvalue = jax.lax.all_gather(rhs.qvalue, axis_name, axis=axis_idx, tiled=True)
        rhs = dataclasses.replace(rhs, qvalue=rhs_qvalue)
```

**它 all-gather 的是 `rhs.qvalue`（量化后的 FP8 字节）而不是 bf16 权重** ——
先量化、再通信，**权重 all-gather 流量直接减半**。

四个触发条件，逐个追到底：

| # | 条件 | 我们 | 追到哪 |
|---|---|---|---|
| 1 | `use_tokamax_backend` | ✅ | `use_tokamax_gmm=True` |
| 2 | `quantization_rule.bwd_qtype` 非空 | ✅ | FP8 配了 `e5m2` |
| 3 | `weight_calibration_method` 以 `"fixed"` 开头 | ❌ | `base.yml:151-153` 默认 `absmax` |
| 4 | `weight_gather_axes` 非空 | ❌ | `moe.py` → `explicitly_weight_ag(config.shard_exp_on_fsdp)` |

##### 但这四条不是并列的：3 和 4 是同一把锁

追进 `moe.py:1556-1561` 才看清，条件 4 内部**又检查了一遍 fixed**：

```python
def explicitly_weight_ag(shard_exp_on_fsdp):
  if shard_exp_on_fsdp:
    quantization_rule = qpl.get_current_rule("gmm")
    if quantization_rule and quantization_rule.weight_calibration_method.startswith("fixed"):
      return True          # ← 只有 fixed 才会产出 weight_gather_axes
  return False
```

**只开 `shard_exp_on_fsdp` 而 calibration 还是 `absmax`，`weight_gather_axes` 恒为空，
QAG 静默不触发、不报错。** 两个开关必须同时打开，缺一个是静默失效而不是崩溃 ——
这是最容易误判「开了但没效果」的地方。

同一段还揭示了**为什么 fixed 是必需的**（`moe.py:1598-1607`）：
`fixed` 分支下权重用**正常 expert 分片**（`wi_kernel_axes`），非 fixed 分支用 DSv3 那套 `mlp_no_fsdp`。
往下追到 qwix `qarray.py:517-530`，根因是一行：

```python
elif method == 'fixed':
  ...
  shape = tuple(1 for _ in shape)   # ← Fixed calibration is always per-tensor
```

**`fixed` 的 scale 是 per-tensor 标量，`absmax` 的 scale 是随张量分片的数组。**
QAG 要 all-gather `rhs.qvalue` 而不 gather scale ——
只有 scale 是标量（每个分片都一样）时这么做才是对的。
`absmax` 还有个更硬的理由：它的 scale 依赖全局最大值，
求它本身就是一次 **blocking 的网络归约**，先量化后通信在语义上就不成立。

##### 第 5 道门槛（实测才踩到）：`fixed` 不是一个合法值

把 `weight_quantization_calibration_method` 改成 `fixed` 之后直接：

```
ValueError: A fixed range is required for fixed calibration.
```

`fixed` 只是方法名，**范围要写在同一个字符串里**（`qarray.py:301`，格式 `<method>[,<args>]`）：

| 写法 | 含义 |
|---|---|
| `fixed` | ❌ 报错 |
| `fixed,224` | 对称区间 `[-224, 224]` |
| `fixed,-224,224` | 显式上下界，必须 `lo ≤ 0 ≤ hi` |

**官方 canonical 值是 `fixed,-224,224`** —— MaxText 三处独立测试
（`tests/integration/tokamax_test.py:112`、`tests/unit/moe_test.py:1468`、
`tests/batchsplit_google_test.py:156`）用的都是这一组。
224 = e4m3 最大值 448 的一半，留了一倍 headroom。

⚠️ **`bwd` 保持 `absmax` 不要动** —— 三处测试都是
`weight` / `act` 设 fixed、`bwd` 留 absmax。QAG 判据只看 weight 那一路。

MaxText 侧是**字符串原样透传**（`quantizations.py:654-656` → qwix `rhs_calibration_method`），
中间不做解析和校验，所以带逗号的值直接写进命令行即可。

##### 192 experts 撞墙的确切位置

`shard_exp_on_fsdp=True` 在 128 device 上直接被拒（[附录 B.2](#b2-崩溃--配置拒绝)）。
源码里写着两条**显式前置校验**（`configs/pyconfig_deprecated.py:1212-1215`）：

```python
if raw_keys["shard_exp_on_fsdp"] and raw_keys["num_experts"] % raw_keys["ici_fsdp_parallelism"] != 0:
  raise ValueError("shard_exp_on_fsdp requires num_experts is divisiable by ici_fsdp_parallelism.")
if raw_keys["shard_exp_on_fsdp"] and (using_tensor_parallelism(raw_keys) or using_expert_parallelism(raw_keys)):
  raise ValueError("shard_exp_on_fsdp requires ici_expert_parallelism = 1 and ici_tensor_parallelism = 1.")
```

**这一点很重要：约束是 `num_experts % ici_fsdp_parallelism == 0`，
不是「专家数必须是 2 的幂」。** 是 FSDP 宽度和专家数的整除关系，
而我们一直用 2 的幂当 FSDP 宽度，才让它看起来像是「192 不是 2 的幂」的问题。

> ⚠️ **但实测拦下它的不是这两条校验。** 我先前照源码写成「拦截点在 config 层」，
> 实跑（S2 轮）报的是 **`shard_map` 的运行时错误**：
> *`shard_map applied to the function 'sparse_matmul_route_and_compute' was given
> argument arrays with axis sizes that are not evenly divisible by the corresponding
> mesh axis sizes`*，而那句 `divisiable` 的 config 报错**一次都没出现**
> （`grep -c` 结果为 0）——我们这个版本走的是 `configs/types.py`（pydantic），
> `pyconfig_deprecated.py` 那条根本没执行。
>
> **结论不变（约束就是整除，workaround 有效），但失败会晚到 shard_map 才暴露** ——
> 意味着**它要先编译一段时间才崩，不是秒拒**（S2 烧了 128 秒）。
> 教训还是那条：**读到源码里有校验，不等于它在你这条代码路径上生效。**

按我们实际用过的 FSDP 宽度列出来：

| | **192 experts（Hy3）** | **256 experts** |
|---|---|---|
| 4 芯片（8 dev） | ✅ | ✅ |
| 16 芯片（32 dev） | ✅ | ✅ |
| 32 芯片（64 dev） | ✅ | ✅ |
| **64 芯片（128 dev）** | **❌ 余 64** | ✅ |
| **128 芯片（256 dev）** | **❌ 余 192** | ✅ |

> 🎯 **这张表是本节最该带走的东西，而且它是一条模型设计结论，不是调优结论。**
>
> **沿用我们最优的 FSDP=128/256 时，192 在 ≤32 芯片没问题、一到 64 芯片就废；
> 256 在任何规模都整除。**
> ⇒ **专家数在模型设计阶段就决定了这个模型能不能顺手用 QAG。**
> 选 2 的幂（256 / 128）跟任何 FSDP 宽度都合得上，选 192 的要么改 FSDP、要么被锁死。

##### 192 有解：FSDP 取 64 / 96 / 48

既然闸门是整除而不是 2 的幂，`192 = 2⁶ × 3` 的因子里就有可用的 FSDP 宽度：

| 芯片 | device | 可用 FSDP（整除 192） | 配法 |
|---|---|---|---|
| 64 | 128 | **64** | `DP2 × FSDP64` |
| 128 | 256 | **64 / 96** | `DP4 × FSDP64` |
| 256 | 512 | **64 / 96** | `DP8 × FSDP64` |

代价是明确的：[§3.7](#37-并行度怎么切可用区间只有-fsdp--128-256) 实测
**FSDP 窄于 128 时静态分片显存吃不住**，附录 B.1 里 `FSDP=64` OOM 过。
所以这条路是「**拿显存换 QAG**」，能不能成立取决于
QAG 省下的 HBM（权重 all-gather 缓冲减半）够不够抵消分片变厚的开销 ——
这正好是下面 S3 那一轮要测的。

##### 已知先例：蚂蚁 ALModel（2026-08-05 查证，含一处重要纠正）

Chris 提到「蚂蚁 ALModel 靠开 QAG 把 FP8 从负收益转成正收益」。
**查证属实，而且有确切数字**：

| | step time | 相对 BF16 |
|---|---|---|
| BF16 基线 | 3.93 s | 1.00× |
| FP8，开 QAG 前 | 4.48 s | **0.88×（负收益）** |
| FP8，开 QAG 后 | **3.77 s** | **1.05×（转正）** |

官方记录的收益机制两条，跟代码完全对得上：

1. **专家权重通信量减半**（bf16 → fp8）
2. **专家权重量化开销降到 1/128**（FSDP=128 时）——
   原来是 all-gather 完每个分片各自量化，改成量化一次再 gather

> ⚠️ **一处重要纠正：蚂蚁走的不是上游这条 `fixed` 路径。**
> 他们的 recipe 明确写着「Using absmax for quantization calibration,
> **instead of** `fixed,-224,224`」，理由是精度。
> 为了在 absmax 下也能开 QAG，他们**打了 local patch**，
> 把 `ops.py` 里的 `startswith("fixed") and isinstance(rhs, qpl.QArray)`
> 直接删成 `isinstance(rhs, qpl.QArray)`。
>
> 所以「ALModel 开 QAG」和「我们开 QAG」**不是同一条代码路径**：
> 我们走上游 static scale，他们走 patch 过的 dynamic scale。
> 官方对 dynamic scale + QAG 的初步分析结论还是**可能不高效**，
> 这条 patch 目前也没进上游。

蚂蚁的目标是 **1.2–1.3× BF16，1.05× 尚未达标**，
说明**即使开了 QAG，FP8 在 Ironwood 上也远没到头** —— 这跟我们 §5.1 的判断一致。

蚂蚁 recipe 里另外几条值得记的选择：per-expert scale（`[E,1,N]` 而非 `[1,1,N]`）、
**前向后向都用 E4M3**（不是默认的 bwd e5m2）、**最后 1 层保持 BF16**；
4000 步 loss 与 BF16 基线偏差 0.14%。

##### 为什么这条比调 tile 值钱

tile 只动计算那 39.2%，**QAG 动的是通信那 57.3% 里的权重 all-gather** ——
它会把 Amdahl 天花板本身抬高。

##### ⚠️ 自我证伪：「连 DSV3 也没开 QAG」这条旁证是错的

我原先写过：*「DSV3 官方 743.5 几乎正好等于『只有计算减半』算出的 746，
说明连 DSV3 也没开 QAG」*。**查了官方 recipe，这条推断不成立。**

DSv3-671B 在 Ironwood 上的官方 FP8 recipe
（`tpu-recipes` → `training/ironwood/deepseek3-671b/4k-fp8-tpu7x-4x4x8`）：

```bash
ici_fsdp_parallelism=-1                              # L80  铺满全部 device
fsdp_shard_on_exp=True                               # L92  = shard_exp_on_fsdp 旧名
use_tokamax_gmm=True                                 # L107
use_qwix_quantization=True  quantization=fp8_full    # L111-112
weight_quantization_calibration_method=fixed,-224,224  # L131
act_quantization_calibration_method=fixed,-224,224     # L132
```

**四个条件一个不缺 —— DSV3 官方 recipe 是开着 QAG 的。**
743.5 ≈ 746 是巧合，不是「没开 QAG」的证据。

教训记两条：

1. **数值吻合不能当机制证据。** 两个独立体系的数字撞在一起，
   在只有一个观测点时说明不了因果。
2. 顺带印证了整除表：DSv3 是 **256 experts** 且 `ici_fsdp_parallelism=-1`（铺满，
   4×4×8 = 128 芯片 = 256 device），`256 % 256 = 0` ——
   **它能用 `-1` 铺满，正是因为专家数是 2 的幂。**
   同样的写法换成 192 experts 会当场被 config 拒掉。

##### ✅ 修正：192 experts 能开 QAG，代价是 batch 被压到 7

W3 = `192e / FSDP64 / QAG / **pdbs 7**` **跑通了**：

| 轮 | experts | FSDP | pdbs | TFLOP/s/chip | 峰值 HBM | NaN |
|---|---|---|---|---|---|---|
| V3 | 192 | 64 | 8 | ❌ 需 95.38 G | — | — |
| **W3** | **192** | **64** | **7** | **625.4** | **92.42 G** | **0** |

**这个 625.4 很有说服力**：[§5.1](#51-fp8-的两条水位618无-qag-625开-qag) 里
64 芯片 FP8 的既有水位是 **594**（`FSDP128 / pdbs 10`，无 QAG）。
**W3 的 batch 更小（7 vs 10），却快 5.3%。**

⇒ **修正后的结论：192 experts 可以开 QAG，但 FSDP 只能取 64、batch 上限被压到 7。**
不是「无路径」，是「路径窄」。
**在更小 batch 下仍净赢 5.3%，说明 QAG 省下的通信确实抵过了窄 FSDP 的代价。**

> 🔁 **又栽在同一件事上：从单点失败外推到「不可行」。**
> 我在 [§4.7](#47-判断收益是真是假的四条纪律) 刚写完「阴性结果不能直接当结论」，
> 转头就把「pdbs 8 OOM」当成了「这条路不通」，**少测一档就下判决**。
> **OOM 只说明「这个 batch 不行」，不说明「这个配置不行」——
> 差 657 MB 的时候第一反应应该是再降一档，不是写结论。**

⚠️ **仍待补的对照**：`192e / FSDP64 / pdbs 7 / 无 QAG`。
只有它能给出 192 上 QAG 的净收益。
V4（`pdbs 8` 无 QAG）已 OOM 且差得更多，**不开 QAG 时 pdbs 7 很可能也跑不动** ——
若如此，这又是一个「QAG 解锁了原本跑不了的档位」的例子。

> 🎯 **整除表仍然成立，但措辞要改准：**
> 专家数取 2 的幂，决定的不是「能不能用 QAG」，
> 而是**「用 QAG 要付多大代价」**：
> - **256 experts**：FSDP 随便选，`FSDP128 / pdbs 11` 轻松跑，HBM 还富余
> - **192 experts**：FSDP 只剩 64 一个选项，batch 被压到 7，堪堪跑通
>
> 192 在 ≤32 芯片完全看不出问题，一到 64 芯片就只剩一条窄路 ——
> **这个代价在小规模验证阶段暴露不出来。**
##### 阶段收口：645 是「不改模型、不写代码」范围内的天花板

2026-08-05 下午连做三轮共 **8 格，没有一格跑出正收益**：

| 方向 | 格数 | 结果 |
|---|---|---|
| MaxText tile 配置 | 1 | ±0（参数根本没接到 kernel） |
| FP8 tile（monkeypatch，往大调） | 3 | 2 格 VMEM OOM，1 格 **−2.5%** |
| XLA flag / SparseCore 卸载 | 4 | 3 格 ±0，1 格 HBM OOM |

> 🎯 **这 8 格的价值不在提升，在于把剩余搜索空间排干净了。**
> 配合 [§4.6 总表](#46-什么能调什么不能调--一张总表)，现在可以明确说：
> **不改模型、不写代码的前提下，645（256 experts）/ 625（192 experts）已接近上限。**

**还没排除的四条**（按可行性排序）：

| # | 方向 | 说明 |
|---|---|---|
| ~~1~~ | ~~推到 pdbs 12~~ | ❌ **已排除**，见下方验证实验十二 |
| 2 | **上 256 芯片** | [§4.1](#41-扩展性weak-scaling-100strong-scaling-掉-11) 实测 weak scaling 100%，加卡同时加 batch 可同比例放大 |
| 3 | 给 GQA 写 batch split decoder | 几百行，**是开发任务不是调优**，且收益未知 |
| 4 | 改模型形状（`emb 4096` → 更大） | DSv3 是 7168，矩阵越大 MXU 越划算。**下一代模型设计决策** |

</details>

##### 实测总览：12 轮实验，一张表看完

下面 12 轮的**结论已全部并入上文和 [§4.6 总表](#46-什么能调什么不能调--一张总表)**，
这里只留一张索引；想看某一轮怎么做、报什么错、数字怎么读，展开下面的折叠区。

| # | 问的问题 | 答案 |
|---|---|---|
| 1 | QAG 在小规模跑不跑得通？ | 通，但踩出**第 5 道门槛**（`fixed` 必须带 range） |
| 2 | 64 芯片上 QAG 净收益多少？ | **+15.6%**（256e 同并行度配对），并省 4.47 G |
| 3 | 完整 80 层能不能跑？ | 四轮全 OOM，但读出 **QAG 在全模型省 11.08 G** |
| 4 | 换 `FSDP128+256e` 行不行？ | 仍 OOM，但**只差 1.88 G**，且证明不收窄 FSDP 更省 |
| 5 | 降一档 batch 呢？ | ✅ **首次跑通：645.0**；关掉 QAG 同 batch 直接 OOM |
| 6 | 补上 DSv3 的 18 个 tile？ | **±0** —— 证伪「调 tile 到 726」 |
| 7 | 那 +0.9% 到底来自哪？ | 2×2 析因：全来自 `cost_estimate_flops`，tile 是 0 |
| 8 | tile 是没用还是没接上？ | **没接上** —— 源码证明 `tiling` 在我们这条分支被丢弃 |
| 9 | 那 FP8 的 tile 谁在管？ | **monkeypatch 是唯一来源**，拿掉它慢 8.5 倍以上 |
| 10 | FP8 能用更大的 tile 吗？ | 不能，三个点全废，`(512,2048,1536)` 是局部最优 |
| 11 | 补齐 DSv3 的 27 个 XLA flag？ | **±0**，含 SparseCore 卸载全套 |
| 12 | 还能把 batch 推到 12 吗？ | 不能，remat 已到底 + batch 必须整除 device 数 |

<details>
<summary><b>展开：12 轮实验的完整过程、报错与数据</b></summary>

##### 验证实验一：4 芯片通路验证（2026-08-05，已完成）

64 芯片当时被占，先在 4 芯片验通路 —— 单节点 `192 % 8 = 0`，
`shard_exp_on_fsdp` 在小规模本来就能开，足以回答「跑不跑得通」。
6 层 / pdbs 4：

| 轮 | 配置 | step (s) | TFLOP/s/dev | 峰值 HBM | NaN | 结果 |
|---|---|---|---|---|---|---|
| Q0 | BF16 基线 | 1.2228 | 196.6 | 69.62 G | 0 | 参照点 |
| Q1 | `+shard_exp_on_fsdp` | 1.2230 | 196.6 | 69.97 G | 0 | **单开无损**，只多 0.35 G |
| Q2 | `+FP8`（absmax） | 1.2375 | 194.3 | 74.55 G | 0 | 单节点 −1.2% |
| Q3 | `+fixed`（裸写） | — | — | — | — | ❌ `A fixed range is required` |
| Q4 | `+shard_exp` 四条件齐 | — | — | — | — | ❌ 同上 |
| Q5 | Q4 + `num_experts=256` | — | — | — | — | ❌ 同上，**没走到整除检查** |

两条收获：

1. **Q1 是有价值的阴性结果**：`shard_exp_on_fsdp` 单独打开
   （calibration 还是 absmax）**性能一模一样**（1.2228 → 1.2230，在 0.005% 抖动内）。
   这正是上面说的「静默失效」—— 它确实没触发 QAG，但也不报错。
   **不要把「开了 `shard_exp_on_fsdp` 没变化」当成「QAG 没用」。**
2. **Q3/Q4/Q5 全挂在同一个错误上**，第 5 道门槛就是这么发现的。
   注意 **Q5 死在 fixed range 上，根本没走到整除检查** ——
   所以这一轮**没有**回答「256 experts 能不能开 QAG」。

⚠️ **单节点本来也测不出 QAG 收益**（它省的是跨卡通信），这轮只有二值意义。

##### 验证实验二：64 芯片判据（2026-08-05，已完成）

用 `fixed,-224,224` 重跑，并把整除 workaround 一起验了。
128 device / 16 层 / pdbs 8：

| 轮 | experts | 并行度 | calibration | `shard_exp` | step (s) | TFLOP/s/**chip** | 峰值 HBM | NaN |
|---|---|---|---|---|---|---|---|---|
| S0 | 192 | DP1×FSDP128 | — (BF16) | — | 3.7443 | 550.6 | 54.33 G | 0 |
| S1 | 192 | DP1×FSDP128 | absmax | off | 3.6120 | 570.8 | 55.30 G | 0 |
| S2 | 192 | DP1×FSDP128 | `fixed,-224,224` | **on** | — | ❌ `shard_map` 不整除 | — | — |
| S3 | 192 | **DP2×FSDP64** | `fixed,-224,224` | **on** | 3.2222 | **639.8** | 57.09 G | 0 |
| S4 | 256 | DP1×FSDP128 | absmax | off | 3.9258 | 525.6 | 60.04 G | 0 |
| S5 | 256 | DP1×FSDP128 | `fixed,-224,224` | **on** | 3.3960 | **607.6** | 55.57 G | 0 |

**⚠️ 这批是 16 层 / pdbs 8 的缩水配置，绝对值不能跟 §5.1 的 618 或 §3 的 580 横比。**
（缩层影响其实很小：全 80 层 `FSDP128 / pdbs 8` 是 543，这里 16 层同 batch 是 550.6。
真正拉低绝对值的是 **pdbs 从 12 降到 8**。降 batch 是我的判断失误，见下。）

**四条结论：**

1. **QAG 净收益 = +15.6%，这是干净的。**
   `S4 → S5` 是完美配对：同 256 experts、同 `DP1×FSDP128`、同 pdbs，
   **唯一差别就是 QAG 开关** —— 525.6 → 607.6。
   讽刺的是，这个最干净的数据点来自「顺手测一下」的 256 experts 轮，
   而不是我特意设计的 192 轮。
2. **QAG 还省显存：60.04 G → 55.57 G，省 4.47 G。**
   量化后再 gather，权重 all-gather 的缓冲区直接减半。
   **省下的显存可以换更大的 batch** —— 这是第二重收益，正在测。
3. **192 的 workaround 成立**（S3 跑通、无 NaN、639.8 是六轮最高）。
   但 `S3 vs S1`（570.8 → 639.8，+12.1%）**同时改了 QAG 和 FSDP 宽度，
   两个变量混在一起，不能当作 QAG 的净收益** —— 我漏了同并行度的对照组。
   要拿 192 上的净收益，得补 `DP2×FSDP64 + FP8 + 不开 QAG` 那一轮。
4. **FSDP 从 128 收窄到 64 没有 OOM**（57.09 G，只比 55.30 多 1.79 G）。
   但这是 16 层，全 80 层会放大 5 倍，**不要直接外推**。

> 🔁 **一条方法论教训（Chris 当场指出）：测极限值就不该自己降配。**
> 我把 pdbs 从 12 降到 8，理由是「压住 FP8 的编译时间」——
> 而 [附录 C](#附录-c编译与环境工程) 里我自己实测过 **编译时间几乎不随规模变**
> （43.5 s vs 44.3 s）。**用一个自己已经证伪过的理由去缩水，
> 等于把极限测试做成了缩水测试。**
> 更隐蔽的坏处：QAG 省的是通信和显存，**batch 越大计算占比越高，
> 小 batch 会把 QAG 的价值测偏**。

##### 验证实验三：完整 80 层极限轮（2026-08-05）——四轮全 OOM，但测出了关键数字

回到完整 80 层 + pdbs 12 起。**四轮全部 OOM**，一个也没跑出来：

| 轮 | 配置 | HLO temporaries 需求 | 上限 | 差 |
|---|---|---|---|---|
| T1 | `DP2×FSDP64` + **QAG** + pdbs 12 | **104.11 G** | 94.74 G | −9.4 G |
| T2 | `DP2×FSDP64` + FP8 无 QAG + pdbs 12 | **115.19 G** | 94.74 G | −20.5 G |
| T3 | `DP2×FSDP64` + QAG + pdbs 14 | 112.43 G | 94.74 G | −17.7 G |
| T4 | `DP2×FSDP64` + QAG + pdbs 16 | 125.09 G | 94.74 G | −30.4 G |

**T1 / T2 这对配对虽然都没跑成，却给出了本节最硬的一个数字：**

> 🎯 **QAG 在完整 80 层上省 11.08 G（115.19 → 104.11，−9.6%）。**
> 16 层时只省 4.47 G —— **省的量随层数增长**，因为每层的专家权重
> all-gather 缓冲都减半。这是**唯一一个在全模型规模上测到的 QAG 收益**，
> 而且它是从两次失败里读出来的：**OOM 报的 required 值本身就是可用的测量**。

同时也否掉了一条路：

> ❌ **`DP2×FSDP64` 在完整 80 层 + pdbs ≥ 12 上不可行。**
> 即便开着 QAG 省了 11 G，仍然超 9.4 G。
> **窄 FSDP 带来的静态分片开销 > QAG 省下的量。**
>
> 🔁 又是同一条教训：我在上面亲手写下「16 层只多 1.79 G，
> **全 80 层会放大 5 倍，不要直接外推**」，然后设计实验时
> 四轮全用 `FSDP64` + pdbs ≥ 12，全军覆没。
> **写下警告和在设计里执行警告，是两件事。**

⇒ 结论：**对 192 experts 这个模型，「收窄 FSDP 换 QAG」在目标规模上是亏的。**
真正干净的路是 **256 experts + FSDP128** —— 整除天然满足，
根本不必收窄 FSDP，也就不用付那笔静态分片的账。
这跟 [S5 轮](#验证实验二64-芯片判据2026-08-05已完成)（256e / FSDP128 / QAG，
HBM 反而比不开 QAG 低 4.47 G）完全一致，
**也再次指向同一条模型设计结论：专家数取 2 的幂。**

##### 验证实验四：按上面的结论掉头（2026-08-05）——又全 OOM，但逼近了

U 轮改走 `FSDP128 + 256 experts`（不收窄），并给 192 降 batch：

| 轮 | 配置 | 需求 | 距 94.74 G |
|---|---|---|---|
| U1 | **256e / FSDP128 / QAG / pdbs 12** | **96.62 G** | **−1.88 G** ← 最接近 |
| U2 | 256e / FSDP128 / 无 QAG / pdbs 12 | 101.17 G | −6.43 G |
| U3 | 192e / FSDP64 / QAG / pdbs 10 | 99.29 G | −4.55 G |
| U4 | 192e / FSDP64 / 无 QAG / pdbs 10 | 106.97 G | −12.2 G |

**两条读得出来的：**

1. **`FSDP128 + 256e` 确实比 `FSDP64 + 192e` 省** —— U1 用 pdbs **12** 只要 96.62 G，
   U3 用 pdbs **10** 反而要 99.29 G。**高两档 batch 还更省**，
   把「不收窄 FSDP」的价值量化出来了。
2. **QAG 省显存已在 4 组独立配置上复现**：

   | 配置 | 无 QAG | 开 QAG | 省 |
   |---|---|---|---|
   | 16 层 / 256e / FSDP128 / pdbs 8 | 60.04 G | 55.57 G | 4.47 G |
   | 80 层 / 256e / FSDP128 / pdbs 12 | 101.17 G | 96.62 G | 4.55 G |
   | 80 层 / 192e / FSDP64 / pdbs 10 | 106.97 G | 99.29 G | 7.68 G |
   | 80 层 / 192e / FSDP64 / pdbs 12 | 115.19 G | 104.11 G | 11.08 G |

   > ⚠️ **规律不要过度解读。** 看着像「窄 FSDP 和大 batch 都让 QAG 省得更多」，
   > 但这四组同时变了层数、专家数、FSDP 宽度、batch 四个量，
   > **每种组合只有 n=1**。能确定的只有一条：**QAG 在所有测过的配置上都省显存，
   > 幅度 4.5–11 G**。成因待查。

##### 验证实验五：pdbs 11 —— QAG 在完整 80 层上首次跑通

| 轮 | 配置 | step (s) | TFLOP/s/**chip** | 峰值 HBM | NaN |
|---|---|---|---|---|---|
| **V1** | **256e / FSDP128 / QAG / pdbs 11** | **19.5980** | **639.0** | **91.56 G** | **0** |
| V2 | 256e / FSDP128 / **无 QAG** / pdbs 11 | — | ❌ OOM | — | — |

HBM 预测也对上了：U1（pdbs 12）需 96.62 G，两参数模型predict 降一档到 92 G 出头，
实测 **91.56 G**。

> 🎯 **V2 的失败比 V1 的成功信息量更大。**
> 同样 pdbs 11、同样 256e / FSDP128，**关掉 QAG 直接 OOM**。
> ⇒ **QAG 不只是提速，它把一个原本跑不了的 batch 档位变成可跑的。**
> 这也意味着在这一档**拿不到同 batch 的配对数据** ——
> 分母根本不存在，「QAG 净收益 x%」在 pdbs 11 上无从定义。
> 要配对必须退到两边都能跑的 pdbs 10。

> ⚠️ **639.0 不能跟 §5.1 的 618 或 §3 的 599 横比 —— 它是 256 experts 的模型。**
> 专家数从 192 改到 256，参数量、FLOP 计算口径、HBM 全变了。
> 这个数字回答的是「**如果下一代模型把专家数设成 256，在 v7 上能跑成什么样**」，
> 不是「我们现在这个模型跑到了 639」。

##### 结论：192 experts 在 64 芯片上，QAG 实际不可用

V3 = `192e / FSDP64 / QAG / **pdbs 8**`（已经是很轻的 batch）仍然 OOM，
**只差 656.93 MB**：

```
Used 95.38G of 94.74G hbm. Exceeded hbm capacity by 656.93M.
    reserved   268.06M
    program     50.82G
    arguments   43.49G
```

把它和 V1 并排放，这是本节最有说服力的一组对比：

| | experts | FSDP | pdbs | 峰值 HBM | 结果 |
|---|---|---|---|---|---|
| V3 | 192 | **64** | **8** | 95.38 G | ❌ 超 0.66 G |
| **V1** | **256** | **128** | **11** | **91.56 G** | ✅ 跑通 |

> 🎯 **更大的模型（256 experts）、高 3 档的 batch，反而少吃 3.8 G。**
> 差别只有一个：**没有收窄 FSDP**。

而 192 experts 想开 QAG，**FSDP 宽度没得选**：

| 候选 FSDP | 整除 192？ | 128 device 上可行？ |
|---|---|---|
| 128 | ❌ 余 64 | — |
| **96** | ✅ | ❌ `128 / 96` 不是整数，DP 凑不出 |
| **64** | ✅ | ✅ 但 pdbs 8 就 OOM |
| 48 | ✅ | 更窄，更糟 |

而且 **FSDP 分片的厚度只由 FSDP 宽度决定，跟总卡数无关** ——
换到 256 芯片跑 `DP8×FSDP64`，每卡分到的权重跟这里一模一样，**照样 OOM**。
**加卡救不了这个问题。**

⇒ 对 192 experts 的 Hy3，QAG 的 FSDP 宽度**只有 64 一个选项**。

> 🛑 **我当时据此写了「QAG 在 v7 上没有可用落地路径」—— 这条结论下早了。**
> 只测到 pdbs 8 就宣布不可用，**再降一档（pdbs 7）它就跑通了**。见下方修正。


##### 验证实验六：补上 DSv3 的 18 个 tile —— 零收益，一条核心假设被证伪

`§5.1` 一直挂着一个假设：*「FP8 那条 kernel 路径的 tile 一次都没扫过，
若拿到 BF16 同量级收益（+17.4%）→ 726」*。
查到 DSv3 官方 recipe 显式设了 **18 个 tile 参数**，而我们一个都没设 ——
看起来正是那个缺口。按维度比例映射到 Hy3（emb 7168→4096，moe_mlp 2048→1536）后实测：

| 轮 | tile | step (s) | TFLOP/s/chip |
|---|---|---|---|
| V1 | MaxText 默认（全 1024） | 19.5980 | 639.0 |
| **W1** | **DSv3 映射的 18 个值** | **19.5980** | **639.0** |

**一模一样，到小数点后四位。**

先排除「参数没进去」——日志里两轮的实际取值不同，**参数确实生效了**：

```
V1:  wi_tile_fwd_embed_dim: 1024   wi_tile_fwd_mlp_dim: 1024
W1:  wi_tile_fwd_embed_dim: 4096   wi_tile_fwd_mlp_dim: 1536   wo_tile_fwd_mlp_dim: 2048
```

> ⚠️ **⇒ 「FP8 只要补上 tile 就能到 726」这个假设，目前的证据不支持。**
> §5.1 里那个 `618 × 1.174 ≈ 726` 的外推，
> 建立在「FP8 tile 与 BF16 tile 同等重要」上 —— **这一步没有被验证过，
> 而 W1 是第一次真正去验它，结果是零。**

但**还不能就此判定「FP8 tile 完全无用」**，两种可能没区分开：

| 可能 | 含义 | 怎么判 |
|---|---|---|
| A | 参数被 MaxText 接收，但**没传到实际 kernel** | 设一个明显糟糕的 tile，若性能**不掉**，说明根本没接上 |
| B | 参数真的生效，但**默认值恰好已经够好** | 设糟糕的 tile 会**明显变慢** |

⇒ 必须补一个**反向判别实验**：故意设一组差的 tile。
**这是典型的「阴性结果不能直接当结论」——
零变化既可能是「改对了但没用」，也可能是「压根没改到」。**

> 🔁 顺带一条：BF16 那边 tile 值 +17.4%，是因为 tokamax 的查找表里**没有 192 这一行**、
> 掉进了极差的默认（[§3.4.2](#342-根因kernel-库的查找表里没有-192-这一行)）。
> **那是在补一个「坏掉的默认值」，不是在做常规调优。**
> FP8 走的是另一套 kernel（`mblx.gmm`），它的默认值**未必也是坏的** ——
> 我把「BF16 上的 +17.4%」平移到 FP8 上当预期，这个类比本身就不严谨。

##### 验证实验七：2×2 析因 —— tile 是 0，但 `cost_estimate_flops` 值 +0.9%

DSv3 recipe 里除了 tile，还有几个我们没设的配置项。
把「tile」和「其余配置项」做成 2×2 析因，四格全跑（单位 TFLOP/s/chip）：

| | 无 `cost_estimate_flops` | 有 `cost_estimate_flops` |
|---|---|---|
| **无 tile** | V1 **639.0** | X1 **644.8** |
| **有 tile** | W1 **639.0** | W2 **644.8** |

**读数干净到不需要统计检验**：tile 的效应恒为 0（两行完全一致），
`cost_estimate_flops` 的效应恒为 +0.9%（两列完全一致），无交互作用。

再逐项剥离，确认 +0.9% 的归属：

| 轮 | 加了什么 | step (s) | TFLOP/s/chip |
|---|---|---|---|
| V1 | 基线 | 19.5980 | 639.0 |
| **X2** | **只加 `cost_estimate_flops_fwd/bwd=5e12`** | 19.4237 | **644.8** |
| X1 | ↑ + `use_max_logit_estimate=-1` + `float32_weight_sum=False` | 19.4220 | 644.8 |
| W2 | ↑ + 18 个 tile | 19.4227 | 644.8 |

**X2 / X1 / W2 三轮互差 0.009%，全在自身抖动（0.005%）量级内。**
⇒ **+0.9% 完全来自 `cost_estimate_flops`；另两个配置项和 18 个 tile 都是 ±0。**

**为什么它有效**（机制自洽，未做 trace 确认）：
它不改任何计算，只是给 splash attention 的 Pallas kernel
**一个手工的 FLOP 成本估计**（`tokamax_ring_attention.py:272-276`，
默认 `-1` = 用 splash 自带估计）。XLA 的延迟隐藏调度器靠这个数字判断
「这个 kernel 要跑多久」，据此决定往里塞多少通信。**估得准，通信藏得就好。**
这跟 [§2.2](#22-结论h2-成立通信占-573) 对得上 ——
**我们是通信瓶颈，任何改善通信重叠的东西都直接兑现。**

> 💡 **这轮的方法论价值大于 0.9% 这个数字。**
> 如果只跑 W2（tile + 其余配置一起上），看到 +0.9% 会**顺理成章地记在 tile 头上** ——
> 而真相是 tile 一分钱不值。**一次改一组、配对留白格，才能给出归因**
> （[§4.7 第 4 条](#47-判断收益是真是假的四条纪律)）。

##### 验证实验八：反向判别 —— tile 参数**根本没传到 kernel**

X3 故意把 18 个 tile **全设成 256**（明显糟糕的分块）。三个点并排：

| tile 设置 | TFLOP/s/chip |
|---|---|
| MaxText 默认（1024） | 639.0 |
| DSv3 映射值（4096 / 1536 / 2048…） | 639.0 |
| **故意设烂（全 256）** | **639.0** |

**三者完全一致。** 按 [§4.7 第 1 条](#47-判断收益是真是假的四条纪律) 的判据，
这不是「默认值恰好够好」，而是**参数压根没接到 kernel**。

**源码给出了确切原因**（`kernels/megablox/ops.py:199-212`）：

```python
# Backend Execution Routing
if use_tokamax_backend and not use_gmm_v2:
    out = _fwd_run_tokamax_v1(lhs, rhs, group_sizes, preferred_element_type,
                              transpose_rhs, use_manual_quantization)
    #     ↑ 参数列表里根本没有 tiling
elif use_tokamax_backend and use_gmm_v2:
    out = _fwd_run_tokamax_v2(..., tiling, ...)      # ← 传了
else:
    out = _fwd_run_megablox(..., tiling, ...)        # ← 传了
```

我们的配置是 `use_tokamax_gmm=True`（⇒ `use_tokamax_backend=True`）
且**没开** `use_gmm_v2` ⇒ 命中第一个分支 ⇒ **`tiling` 整个被丢弃**。

`base.yml:247-248` 其实早写明了，只是我一直没读到：

```yaml
# megablox/jax ragged dot - supports forward pass only (6 configs)
# tokamax ragged dot - supports all 18 configs
```

而 `_fwd_run_tokamax_v1` 内部走 `tokamax.ragged_dot`（`ops.py:286`），
**分块由 tokamax 自己的 heuristics 决定** ——
也就是 [§3.4.3](#343-修法6-行-monkeypatch) 那个 6 行 monkeypatch 打的地方。

> 🛑 **⇒ [§5.2](#52-为什么-fp8-的-tile-没调成两条-gmm-路径) 的核心论断写反了。**
> 我写的是「**开 FP8 等于换 kernel 路径**，BF16 那个 monkeypatch 在 FP8 下一行都不执行，
> 618 是『FP8 + MaxText 默认 tile』跑出来的」。
> **真相相反**：FP8 走的仍是 tokamax，**monkeypatch 照常生效**，
> 反倒是 MaxText 的 `w{i,o}_tile_*` 被丢弃。
>
> **这意味着「FP8 的 tile 一次都没扫过」这个前提从头就不成立** ——
> 我们所有 FP8 轮次都注入了 `tkcfg.py`，
> **FP8 一直在吃 BF16 调出来的最优 tile `(512, 2048, 1536)`**。
> 这也解释了 §5.1 那个 618 为什么并不算差 —— 它不是「未调优」的起点。

⇒ **要让 MaxText 的 tile 配置真正生效，必须开 `use_gmm_v2=True`**（走 `_fwd_run_tokamax_v2`）。
但 [附录 B.3](#b3-明确负收益) 记着 gmm_v2 的收益被 XLA 插入的 copy 吃掉 70% ——
这条路要重新评估，不是免费的。

##### 验证实验九：monkeypatch 是 FP8 路径上**唯一**的 tile 来源

| 轮 | monkeypatch | 结果 |
|---|---|---|
| **Y0** | `512,2048,1536`（BF16 最优值） | **322 s 跑完，645.0** —— 精确复现 X1/X2/W2 的 644.8（差 0.03%） |
| **Y1** | **完全不注入** | **2748 s 超时，8 步没跑完** |

**至少慢 8.5 倍，而且实际更多**（Y1 到超时都没跑到 step 4）。
现象与 [附录 B.5](#b5-被推翻的结论我自己写下过又证伪的) 第一条同款：
*「不是死锁，是慢到触发看门狗」*。

**源码给出了比实验更强的解释**（`tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py:320-322`）：

```python
@override
def _get_heuristics_config(self, ba) -> Config:
    if self.qdtype is not None:
        return Config()        # ← 量化时直接返回空配置，一个 tile 都不算
    if pltpu.get_tpu_info().generation < 7:
        return Config()
    ...                        # ← 下面这套 "从最大开始 deflate 到装进 VMEM" 的逻辑
                               #    只在非量化路径上执行
```

> 🎯 **FP8 下，框架的自动分块逻辑整个短路。**
> 不是「算了个不够好的值」，是**根本不算** —— 返回空 `Config()`，
> 剩下的交给 Pallas/Mosaic 的保守默认。
>
> 我们那 6 行 monkeypatch 是在 `_orig()` 返回之后用 `dataclasses.replace`
> 覆盖 `tile_m/k/n`，所以**在 FP8 上它不是"优化"，而是唯一的 tile 提供者**。
> 拿掉它 = 没有任何 tile 决策 = Y1 那个 8.5 倍。

⇒ 这条同时修正了 [附录 B.5](#b5-被推翻的结论我自己写下过又证伪的) 里
「tokamax 查找表没有 192 这一行」的表述：
**BF16 是"查表 miss 掉进坏默认"，FP8 是"压根没有查表这一步"** ——
两者现象相似（都极慢）但机制不同。

##### 下一步：FP8 专属 tile 扫描（有理论依据，非盲扫）

上面那段 heuristics 还给出了**优化方向**：它的策略是
「`tile_m = min(m,1024)`、`tile_n = n`、`tile_k = k` 起步，
**装不进 VMEM 就逐级减半**」，而 `_fit_within_tpu_vmem` 是
**按 `lhs.dtype` / `rhs.dtype` 的实际字节数**算容量的。

> 💡 **FP8 的权重只占 bf16 的一半字节 ⇒ 同样的 VMEM 能装下约两倍大的 tile。**
> 我们现在用的 `(512, 2048, 1536)` 是**在 bf16 的显存约束下 deflate 出来的**，
> 直接搬到 FP8 上等于白白空着一半 VMEM。

据此定向试三个点（基准 Y0 = 645.0）：

| 轮 | tile | 假设 |
|---|---|---|
| Z1 | `512, **4096**, 1536` | tile_k 加倍 —— 归约维度最先受益于省下的 VMEM |
| Z2 | `**1024**, 2048, 1536` | tile_m 加倍 |
| Z3 | `**1024**, **4096**, 1536` | 两个都加 |

⚠️ Z3 那组在 **BF16** 下曾被 kernel 直接拒绝（[附录 B.2](#b2-崩溃--配置拒绝)
`tile_m ≥ 1024 且 tile_k ≥ 4096`）。
**但那是 bf16 的显存压力下**，FP8 少一半，值得重试 ——
这也正好是「收益/限制不能跨 dtype 平移」的又一个检验点。

##### 验证实验十：FP8 tile 扫描结果 —— 三个点全废，`(512,2048,1536)` 是局部最优

| 轮 | tile | 结果 |
|---|---|---|
| Y0 | `512, 2048, 1536`（基准） | **645.0** |
| Z1 | `512, **4096**, 1536` | ❌ **VMEM OOM**（需 352 MiB） |
| Z2 | `**1024**, 2048, 1536` | **628.8**（**−2.5%**） |
| Z3 | `**1024**, **4096**, 1536` | ❌ **VMEM OOM**（需 352 MiB） |

> 🛑 **我那个「FP8 能装两倍大 tile」的推论有个逻辑漏洞。**
> **FP8 只有 `rhs`（权重）省一半字节，`lhs`（激活）仍是 bf16。**
> 而把 `tile_k` 加倍会让 **lhs 项和 rhs 项同时翻倍** ——
> 单价打五折 ≠ 尺寸能翻倍。
>
> 正确的推论应该是：**FP8 下同一组 tile 比 BF16 省显存**
> （所以 BF16 会 OOM 的配置 FP8 可能跑得动），
> 但**不等于可以把 tile 尺寸往上推**。
> 这是「省单价」和「加尺寸」两件事，我把它们混成了一件。

**Z2 的负收益反而是最有价值的一格**：
BF16 下把 `tile_m` 从 512 抬到 1024 是 **−3.8%**（[附录 A.1](#a1-256-芯片--512-device--完整-80-层2026-08-04)），
FP8 这边是 **−2.5%**。**方向一致、幅度接近。**

> 🎯 **⇒ tile 的最优点跨 dtype 是稳定的。**
> `(512, 2048, 1536)` 是 BF16 扫出来的，在 FP8 上同样是局部最优 ——
> 三个方向（加 k / 加 m / 都加）要么 VMEM 爆、要么变慢。
> **这条路到头了，不必再扫。**
>
> 附带修正一条纪律：我在 §4.7 写「收益幅度不能跨 kernel 平移」，
> 这次的数据显示**最优点（argmax）可以平移，能平移的不是收益幅度（max value）**。
> 两者是不同的东西。

##### 验证实验十一：XLA flag —— 我们 9 个 vs DSv3 36 个，补齐后仍是 ±0

DSv3 recipe 有 **36 个 XLA flag，我们只有 9 个**，缺的主体是一整套
**SparseCore Collective Offloading**。而官方 Ironwood 调优文档明确写着：
*「TPU7x 上重叠通信与计算的主要机制叫 SparseCore Collective Offloading，
这是 TPU7x 上异步集合通信的推荐做法」*。

我们瓶颈是通信占 57.3%，却漏掉了官方指定的通信优化机制 —— 看起来是最大的一块。
**实测四格，全是 ±0 或更差：**

| 轮 | 配置 | 结果 |
|---|---|---|
| 基准 | 我们原有 9 个 flag | 645.0 |
| A1 | + SparseCore 卸载 11 个 | **644.8**（±0） |
| A2 | DSv3 完整 36 个 | ❌ **HBM OOM** |
| B1 | 官方 `ENABLE_SPARSECORE_OFFLOADING_FOR_RS_AG_AR` 全组（关 CF + 开 SC + 2 个 base flag） | **644.8**（±0） |
| B2 | **只关 CF**，其余不动 | **644.8**（±0） |

**A1 的 ±0 是预期内的** —— MaxText 官方 flag library（`benchmarks/xla_flags_library.py`）注释：

```python
# On Ironwood, by default:
# xla_tpu_enable_sparse_core_collective_offload_all_gather as True
# xla_tpu_enable_sparse_core_collective_offload_reduce_scatter as True
# xla_tpu_enable_sparse_core_collective_offload_all_reduce as True
```

**这三个在 Ironwood 上默认就是 True**，显式再设一遍等于没设。

同一个文件里还有一句更关键的：

```python
# Either one of CF or SC can be enabled at a time.
```

**`async collective fusion`（CF）与 SparseCore offloading（SC）互斥。**
我据此推断「SC 虽默认开着，但被 CF 挡住了」—— B1/B2 就是去验它，
**关掉 CF 之后仍然 ±0**。

> 🛑 **推断在机制上说得通，但收益是零。**
> 最合理的解释：我们原有 9 个 flag 里的
> `xla_tpu_enable_sparse_core_collective_aggregator=true` 和
> `xla_tpu_enable_latency_hiding_layer_scheduler=true` **已经把这块收益吃掉了**；
> 而剩下的大头 —— MoE 权重 all-gather —— **已经被 QAG 减半**。
> **SparseCore 卸载能帮的部分，我们通过别的途径已经拿到了。**

A2 的 OOM 本身也是信息：DSv3 那套里有抬高 HBM 占用的项
（`scoped_vmem_limit_kib=65536` 比我们的 65472 高，或 `accumulate_into_mrb`）。
⇒ **不要整组照搬别人的 flag —— 它们是按另一个模型的显存预算调的。**


##### 验证实验十二：`pdbs 11 → 12` 这条路堵死了

`pdbs 12` 需 96.62 G，只差 **1.88 G**，看着很近。两个方向都试了：

**① 靠 remat 省出来 —— 没有空间。**
`remat_policy=custom` 下每个张量三选一：`device`（存 HBM）/ `remat`（丢弃重算）/ `offload`（搬 host）。
查 `configs/base.yml:356-374`，**除 `decoder_layer_input` 外，其余张量默认就是 `remat`** ——
本来就不占 HBM，已经是最省的档位；
而 `decoder_layer_input` 不能 remat（它是重算的起点），我们已经设成 `offload`。
⇒ **remat 这一维已经到底了。**

**② 用小数 batch 取中间档 —— 不支持。**

| 轮 | 配置 | 结果 |
|---|---|---|
| C1 | `per_device_batch_size=11.5` | ❌ `AssertionError: Batch dimension should be shardable among the devices in data and fsdp axis` |
| C2 | `per_device_batch_size=11.8` | ❌ `ValidationError` |

**batch 维度必须能被 `data × fsdp` 的 device 数整除**，pdbs 只能取整。
（DSv3 recipe 里写 `per_device_batch_size=8.0` 只是浮点写法，值仍是整数 ——
我误以为它意味着支持小数档。**看到浮点字面量不等于支持连续取值。**）

> 🎯 **⇒ `pdbs 11` 就是「64 芯片 / 256 experts / QAG」这套配置的 batch 上限，
> 645.0 是对应的性能上限。**
> 不是「差一点点」，是 **11 和 12 之间根本没有档位可取**。


</details>

##### 出路重排（依据已从推测变成源码 + 官方 recipe）

| # | 思路 | 现在的判断 |
|---|---|---|
| 1 | **FSDP 取 64 / 96 整除 192** | ✅ 机制上确定可行（约束是整除，不是 2 的幂）。**代价是显存**，S3 在测 |
| 2 | **模型改 256 experts** | ✅ 最干净，跟 DSv3 官方 recipe 同构。属于下一代模型设计决策，S5 在测 |
| 3 | 抄蚂蚁的 local patch（absmax + QAG） | ⚠️ 官方分析认为 dynamic scale 下 QAG 可能不高效，且未进上游 |
| 4 | 配 `ici_expert_parallelism` 凑整除 | ❌ 排除。config 明确要求 `EP=1 且 TP=1`，且 TPU 上 EP 实测 **−71%** |
| 5 | 等上游支持非整除 | ❌ 不在我们手上 |

⚠️ **通用风险没变**：`fixed` 是预设 scale，`±224` 是官方在 DSv3 上的选择，
**换到 Hy3 的权重分布上未必合适**。
**8 步 benchmark 只能看出有没有 NaN，证明不了收敛 —— 这条要动必须配收敛验证。**
蚂蚁正是因为精度才放弃 fixed 改用 absmax + patch，这个先例要记住。

---


**结论：FP8 现在是「能跑、有 +3.2~5.9%、但一格都没调」的状态。
按性价比，下一步应是 ① 扫 FP8 的 `w{i,o}_tile_*` → ② 查 QAG 能否开启，而不是 autotune。**

---

## 6. 还没试的

> 这张表 2026-08-11 重写过一次 —— 原先六项里有四项其实已经做完（FP8、XLA flag 补齐、tile 精扫、
> 部分 autotune 调研），还有一项（ring of experts）的立项理由是「通信占 57.3%」，
> 而那个前提已被 [§2.4](#24-2026-08-08-复测64-芯片生产配置上通信已被完全掩盖) 自己推翻。
> **过期的 backlog 比没有 backlog 更浪费时间。**

按预期收益排序：

| # | 项 | 为什么值得 | 成本 |
|---|---|---|---|
| 1 | **splash attention 的 NT gemm 全转置** | 目前唯一还看得见幅度的方向（外部同类工作报过 35% → 56%）；调参已见底，只能改 kernel | 写代码，高 |
| 2 | **256 芯片上复测 `dvfs_p_state=7`** | 64 芯片实测 +8.6%，预期等幅但**没验过**；这是当前最便宜的未知 | 一轮，低 |
| 3 | **`p3` / `p7` 各抓一份 profile 比 HBM 的 GB/s** | 直接证伪或坐实「dvfs 只提计算域频率」这条推断（见 [EXPERIMENT-LOG](EXPERIMENT-LOG.md)）；若 HBM 也提频，则现有解释全错 | 两轮 profile，低 |
| 4 | **官方 `tokamax.autotune` 生成 cache 条目** | 替代 monkeypatch，是长期正解；[§5.4.1](#541-官方-autotune-调研结论2026-08-05) 调研过，不是 CLI，成本高于手调 | 中 |
| 5 | `gmm_v2` + `tile_k` 整除 K | 上游报过 +13.58%；但我们实测 `use_gmm_v2` 的收益被 XLA 插的 copy 吃掉 70%，要先解决那个 | 中 |
| 6 | **256 experts 的形状**（改模型） | 实测 645，且 FSDP 随便选、batch 能上 11 —— 但这是给**下一代模型设计**的输入，不是本次能用的 | 改模型 |

**不建议再碰的**（都已实测归零，见 [附录 B](#附录-b负面案例总集)）：
SparseCore 卸载组、FP8 的 18 个 MaxText tile、FP8 tile 邻域精扫、
`CollectivePipeliner` 一类的藏通信手段（64 芯片上阻塞只剩 0.19%，没东西可藏）。

---

<details>
<summary><b>附录 A：全部消融数据</b> —— 五个批次的完整实验表（点开）</summary>

## 附录 A：全部消融数据

### A.1 256 芯片 / 512 device / 完整 80 层（2026-08-04）

**A 组：并行度切法**（pdbs 8，megablox）

| 切法 | step | dev | chip | MFU | 峰值 HBM |
|---|---|---|---|---|---|
| `DP4×FSDP128` | 20.0915 s | 226.5 | 453 | 19.64% | 74.20 G |
| `DP2×FSDP256` | 20.2280 s | 225.0 | 450 | 19.51% | 61.36 G |
| `DP1×FSDP512` | 22.5470 s | 201.8 | 404 | 17.49% | — |
| `DP8×FSDP64` | OOM | | | | |
| `DP16×FSDP32` | OOM | | | | |

**B 组：batch / 序列**（基座 `DP4×FSDP128`，megablox）

| 配置 | step | dev | chip | MFU | 峰值 HBM |
|---|---|---|---|---|---|
| pdbs 8 | 20.0915 s | 226.5 | 453 | 19.64% | 74.20 G |
| pdbs 12 | 27.8648 s | 245.0 | 490 | 21.24% | 91.93 G |
| pdbs 16 | OOM | | | | |
| seq 8192 / pdbs 4 | 22.5383 s | 225.6 | 451 | 19.56% | 74.33 G |

**C 组：tokamax tile**（基座 `DP4×FSDP128`，pdbs 8）

| tile | step | dev | chip | MFU | 峰值 HBM |
|---|---|---|---|---|---|
| (512, 2048, 1536) | 17.1190 s | 265.8 | **532** | 23.04% | 75.33 G |
| (1024, 2048, 1536) | 17.7907 s | 255.8 | 512 | 22.18% | 75.34 G |
| (512, 1024, 1536) | 18.2422 s | 249.4 | 499 | 21.62% | 75.33 G |

**F 组：tile × batch 组合**（全部带 `tile(512,2048,1536)` 除非注明）

| 配置 | step | dev | chip | MFU | 峰值 HBM |
|---|---|---|---|---|---|
| `DP4×FSDP128` + pdbs 10 | 20.1713 s | 282.0 | 564 | 24.45% | 84.06 G |
| `DP4×FSDP128` + pdbs 12 | 23.5553 s | 289.8 | **580** | 25.12% | 91.94 G |
| `DP2×FSDP256` + pdbs 12 | 23.9947 s | 284.5 | 569 | 24.66% | 78.27 G |
| `DP2×FSDP256` + pdbs 14 | 27.2155 s | 292.6 | 585 | 25.37% | 89.56 G |
| **`DP2×FSDP256` + pdbs 16** | 30.4002 s | 299.4 | **599** | **25.96%** | 92.33 G |
| `DP4×FSDP128` + pdbs 12 + **tile_m 1024** | 24.0598 s | 283.7 | 567 | 24.59% | 91.95 G |

### A.2 64 芯片 / 128 device / 完整 80 层（2026-08-04，`DP1×FSDP128`）

| 配置 | step | dev | chip | MFU | 峰值 HBM | 256 芯片同配置 |
|---|---|---|---|---|---|---|
| megablox / pdbs 8 | 19.9188 s | 228.4 | 457 | 19.80% | 74.20 G | 453 |
| tile / pdbs 8 | 16.7545 s | 271.6 | 543 | 23.55% | 75.33 G | 532 |
| tile / pdbs 10 | 20.2780 s | 280.5 | 561 | 24.32% | **84.06 G** | 564（**84.06 G**） |
| **tile / pdbs 12** | 23.5385 s | 290.0 | **580** | **25.14%** | **91.94 G** | **580**（**91.94 G**） |

### A.3 64 芯片 / 80 层（2026-08-01，早期批次）

基线 `D0` = 17.4349 s，228.1 TFLOP/s/device。**这批换了一批机器，比 07-30 的 20.43 s 快 15%
—— 跨批次比绝对值没意义，消融必须同批次内比。**

| # | 改动 | 结果 | Δ |
|---|---|---|---|
| D0 | 基线 | 17.4349 s | — |
| D2 | `remat_policy=full` + `decoder_layer_input=remat` | 17.5633 s | **−0.74%** |
| D4 | 删 8 个 SparseCore flag（留 aggregator） | 17.4355 s | −0.00% |
| D1 | `shard_exp_on_fsdp=True` | **崩** `IndivisibleError` | — |

### A.4 16 芯片 / 20 层（快速筛选，结论不可直接外推幅度）

基线 `B0` = 5.3336 s，201.9 TFLOP/s/device。

| # | 改动 | 结果 | Δ |
|---|---|---|---|
| C6 | `shard_exp_on_fsdp` + `remat=full` | 5.1862 s | +2.76% ⚠️ 64 芯片崩 |
| A5 | `shard_exp_on_fsdp=True` | 5.2545 s | +1.48% ⚠️ 64 芯片崩 |
| A10 | `remat_policy=full` | 5.2683 s | +1.22% ⚠️ 64 芯片 −0.74% |
| C5 | 删 8 个 SparseCore flag | 5.3339 s | −0.01% |
| G2 | `fp8_full` + qwix + 6 个 mlp tile 全设 1536 | 4.8937 s | **+8.25%** |

### A.5 早期演进（64 芯片，2026-07-30）

| 轮次 | 增量 | seq | pdbs | step | chip | MFU |
|---|---|---|---|---|---|---|
| V1 | 基线：2 个 XLA flag | 8192 | 4 | 25.11 s | 405 | 17.54% |
| y1 | + `use_tokamax_splash` + `sa_use_fused_bwd_kernel` | 8192 | 4 | 24.45 s | 415 | 18.00% |
| y4 | + 调度器组（4 flag） | 8192 | 4 | 23.08 s | 440 | 19.09% |
| **c1** | 调度器组 × pdbs 8 / seq 4096 | 4096 | 8 | **20.43 s** | **445** | **19.29%** |
| c2 | c1 + 杂项组（补齐 26 flag） | 4096 | 8 | 20.45 s | 445 | 19.27% |

按贡献排序：pdbs 8 / seq 4096 **+12.8%** ｜ 调度器组 **+6.6%** ｜
`use_tokamax_splash` + `sa_use_fused_bwd_kernel` +2.6% ｜ 杂项组 ±0 ｜ SparseCore 组 ±0 ｜
优化器/显存组 −0.5%（省的是显存不是时间）。

---


</details>

## 附录 B：负面案例总集

<details>
<summary><b>全部 OOM / 崩溃 / 零收益 / 被推翻的结论（点开）</b></summary>

### B.1 显存类（OOM）

| 配置 | HBM | 说明 |
|---|---|---|
| `DP8×FSDP64` / `DP16×FSDP32`（512 dev） | — | FSDP 减半 → 每卡静态分片翻倍。FSDP=64 静态约 51 G |
| `pdbs=16`（FSDP128, 512 dev） | 预测 109.6 G | 模型预测与实测一致 |
| `shard_exp_on_fsdp=True`（FSDP=64×DP=2） | 109.14 G | **比不开还多 14 G** |
| `per_device_batch_size=12`（旧配置，FSDP 未摊薄） | 95.17 G | 后来靠 FSDP 摊薄跑通了 |
| `ici_expert_parallelism=4`（16 芯片） | 137.60 G | |
| `ici_expert_parallelism=8`（16 芯片） | 192.70 G | |
| EP4 + ring + `num_moe_token_chunks=4` | 111.24 G | |
| `scan_layers=False`，pdbs 8 | **171.75 G** | pdbs 降到 1/4 显存只降 6% —— 爆的不是激活 |
| `scan(unroll=10)` | **274.64 G** | 比全展开还高 |
| tokamax `tile_k=4096` | OOM | |

**`shard_exp_on_fsdp` 为什么净亏**：这笔交易两头都动 —— 专家权重改按专家维切（收益），
但 FSDP 宽度从 128 降到 64，**非专家部分（attention 80 层 + embedding + dense 首层，约 7.2 B）
每卡分片直接翻倍**。省的抵不过多花的。根因还是 **192 不是 2 的幂**。

### B.2 崩溃 / 配置拒绝

| 配置 | 报错 |
|---|---|
| `shard_exp_on_fsdp=True`（128 device） | 192 专家除不尽 128 device。拦截点是 `pyconfig_deprecated.py:1212` 的显式校验 `num_experts % ici_fsdp_parallelism != 0`，**不是 kernel 层** ⇒ 改 FSDP 宽度可绕，见 [§5.4.2](#542-qag先量化再通信一条被专家数卡死的路) |
| `weight_quantization_calibration_method=fixed`（裸写） | `ValueError: A fixed range is required for fixed calibration.` **`fixed` 只是方法名，范围要写进同一个字符串**：`fixed,-224,224`（官方 canonical 值） |
| `shard_exp_on_fsdp=True` + `ici_expert_parallelism>1` 或 `ici_tensor_parallelism>1` | `pyconfig_deprecated.py:1214` 拒绝：要求 `EP=1 且 TP=1`。⇒ **不能靠 EP 凑整除** |
| `quantization=fp8` | `AttributeError: Fp8Quantization 无 quant_dg` —— 那是 **NVIDIA 专用类**，TPU 正路是 `fp8_full` + qwix |
| 删**全部** 9 个 SparseCore flag | `层调度器要求 sparse core collective aggregator 开启` |
| `use_gmm_v2=True` 单开 | 配置拒绝，需 `use_tokamax_gmm=true` |
| `num_moe_emb_chunks=4` | 配置拒绝，需 `use_gmm_v2` + `use_ring_of_experts` |
| `fp8_full` + qwix，默认 tile 1024 | `AssertionError: v=1536 bv=1024 s=1536` |
| **`tile_m` ≥ 1024 且 `tile_k` ≥ 4096**（如 `(1024,4096,1536)`、`(2048,2048,1536)`） | `MosaicTpuRaggedDot(config=None, vjp=...)` —— Pallas kernel 直接拒绝该组合，非 OOM |
| **FP8 + `tile(512,2048,1536)` + pdbs 12**（64 芯片） | 同上 `MosaicTpuRaggedDot` 拒绝。**pdbs 10 同配置正常** ⇒ FP8 路径下 tile 与 batch 的组合有额外约束 |
| `scan(unroll=2)` | XLA `Expected instruction to have shape equal to (bf16[9,2,8,4096,4096], ...)` —— `2` 是 unroll 因子，`9` 是 splash attention 分块，下游 kernel 没跟上 |

### B.3 明确负收益

| 配置 | Δ | 说明 |
|---|---|---|
| `ici_expert_parallelism=4`（半 batch，16 芯片） | **−71.36%** | EP 不是"装不下"，是**本身就慢**。TPU torus 上 AllToAll 多跳 |
| EP4 + ring + `token_chunks=4` | −36.96% | 分块**挽回 34 个百分点**，但填不平 EP 的坑 |
| `use_2d_fsdp_sharding` + `fsdp_transpose=4` + `two_stage_all_gather` | **−11.73%** | 别再试 |
| `DP1×FSDP512`（512 device） | −11% | strong scaling 的代价 |
| tokamax `tile(128, 4096, 1536)` | −9.0% | |
| `int8` + qwix（16 芯片） | −5.81% | |
| `fp8_full` + qwix + tile 512 | −3.90% | 512 能整除 1536，但切三刀反而更慢 |
| 纯 bf16 + tile 512 | −3.25% | 证明是 tile 本身的锅，与量化无关 |
| `tile_m=1024` @ pdbs 12 | −2.2% | 否掉了「`tile_m` 随 `m` 走」这条表内规律 |
| `fp8_e4m3` + qwix（默认 tile） | −1.87% | |
| FP8（`fp8_full` + qwix，单节点） | −1.2% | **但"能跑"本身是新信息** —— v5p 上是编译失败 |
| `remat_policy=full`（64 芯片） | −0.74% | 16 芯片上是 +1.22%，**符号反转** |
| 优化器 / 显存组 | −0.5% | 省的是显存不是时间 |
| `scan_layers=False`（5 层） | −5.9% | 而且大模型直接 OOM |

### B.4 零收益（跑过，确认没用）

| 配置 | Δ |
|---|---|
| SparseCore 卸载组 9 个 flag（v7） | **±0**（v5p 上是 +4.07 pp） |
| 删其中 8 个（留 aggregator） | −0.00%，16 芯片和 64 芯片一致 |
| 杂项 flag 组（5 个，补齐 26 flag） | ±0 |
| seq 8192 / pdbs 4 vs seq 4096 / pdbs 8 | 451 vs 453，打平 |

### B.5 被推翻的结论（我自己写下过又证伪的）

| 曾经的结论 | 真相 |
|---|---|
| 「`use_tokamax_gmm` 死锁，`stalled chips [7]`」 | **不是死锁，是慢到触发看门狗**。根因是 LUT miss → grid 块数涨 768 倍 |
| 「专家数 192 不是 2 的幂，导致组划分出问题」 | 方向对一半：确实跟 192 有关，但是**上游只给 16/128/256 做过 tile 调优**，是**数据覆盖问题不是算法问题** —— 两者修法完全不同 |
| 「把 `num_experts` 改成 256 就能命中 LUT」 | **实测无效**，48 次 `Autotuning cache miss` 一次不少。还有一套 JSON autotuning cache，key 是完整算子签名 |
| 「小规模编译更快，49.6 s vs 10–17 分钟」 | **拿两个不同口径的数字相除**。实测编译几乎不随规模变（43.5 s vs 44.3 s），真正随规模涨的是多机建切片（0.8 s vs 60.2 s） |
| 「step 0 = 49.6 s 是编译时间」 | 那是 `dump_hlo=True` 往 GCS 上传 HLO 的时间 |
| 「用 `scan(unroll=N)` 让 XLA 跨层藏通信」 | 2 撞 kernel 形状校验、10 需 274 G，**没有可用档位**。而且 80% 的通信是同步集合通信，本来就不是调度边界造成的 |
| 「通信与计算重叠 0.000 s ⇒ 完全裸露」 | **同义反复** —— 在单条顺序 lane 上算交集恒为 0 |
| 「pdbs=12 要 98.5 G，会 OOM」 | 实测 91.93 G 跑通。单点线性外推错误，激活是次线性的 |
| 「小规模筛不出赢家」 | 太严。要按「是否改变分片形状」分类，tile / pdbs 完全可传递 |
| 「MoE 占比最大，该在 MoE 上使劲」 | megablox 已是被调过的最优路径。**该找没被人调过的地方** |
| 「先用 4 芯片 smoke 复现 tokamax 问题」 | 8 个假设全跑通、复现不出来。真正查出根因靠**直接扒 kernel 库源码把表打印出来** |
| 「DSV3 官方 743.5 ≈ 我算的 746 ⇒ 连 DSV3 也没开 QAG」 | **官方 recipe 里 QAG 四个条件一个不缺，DSV3 是开着的**。两个独立体系的数字撞在一起，在只有一个观测点时说明不了因果 |
| 「QAG 的四个触发条件是并列的」 | 3 和 4 是同一把锁 —— `explicitly_weight_ag` 内部又查一遍 `fixed`。只开 `shard_exp_on_fsdp` 会**静默失效**（实测 1.2228 → 1.2230，无变化也无报错） |
| 「专家数必须是 2 的幂才能开 QAG」 | 约束是 `num_experts % ici_fsdp_parallelism == 0`。是**整除关系**，不是 2 的幂 —— 只因我们一直用 2 的幂当 FSDP 宽度才看着像 |
| 「蚂蚁 ALModel 靠上游这条 `fixed` 路径开的 QAG」 | 反了。**他们明确弃用 `fixed,-224,224` 改用 absmax**（精度考虑），靠 local patch 删掉 `startswith("fixed")` 判据才开的 |

### B.6 操作类翻车（不是结论，但会浪费时间）

- **XLA flag 裁剪要成组**：删掉调度器依赖的 `collective_aggregator` 会两轮秒挂
- **`gcloud ... | tail -3` 会吞掉真实退出码**（pipeline 返回 tail 的 0），400 错误被当成成功。
  **判成败要看输出内容，不要只看 `$?`**
- **跨项目 GCS 读不到**时，12 MB 级的文件直接 `kubectl cp` 进 pod，不要去动共享桶 IAM
- **`pkill -f 'pre_train.train'` 会杀掉自己** —— `pkill -f` 匹配整条命令行，而那行文本里就含这个
  模式串。正解 `'pre_train[.]train'`
- **bash 函数内的变量默认是全局的** —— 我的战役脚本里 `run()` 函数用 `TK=$4` 接 tile_k，
  与全局的 `TK='megablox=True ...'` 重名，第一轮跑完就把它污染成 `2048`，
  后续四轮把裸参数 `2048` 传给 MaxText 全部 `ValueError`
- **`pkill -f` 的模式要对着真实命令行写**。我的战役脚本用
  `pkill -9 -f "train[.]py"` 清理上一轮，但实际命令行里是
  `runpy.run_module('src.maxtext.trainers.pre_train.train')` —— **不含 `train.py` 这个串**，
  所以这条 pkill **一次都没匹配到过**。轮次之间能正常切换靠的是上一轮自己退出，
  不是靠它。**写完 pkill 要用 `pgrep -f` 同模式验一次能不能匹配到目标。**
- **取稳态先看序列再定窗口**：开 profiler 时 step 17 会有 ~90 秒尖峰（导 trace），
  按惯例取 `step ≥ 15` 求均值会得到 11.678 s，比真实稳态慢近一倍
- **测极限值时不要自己降配**。我为了「压住 FP8 编译时间」把 pdbs 从 12 降到 8、
  层数从 80 缩到 16 —— 而附录 C 里我自己实测过**编译时间几乎不随规模变**。
  用一个自己已经证伪过的理由缩水，等于把极限测试做成了缩水测试，
  拿到的 639.8 无法跟历史最优横比，得整轮重跑。
  **更隐蔽的是它会测偏结论**：QAG 省的是通信和显存，batch 越小计算占比越低，
  小 batch 会系统性高估 / 低估通信类优化的价值
- **配对实验要「只差一个变量」，设计时就得数清楚**。S3 轮我同时改了
  「开 QAG」和「FSDP 128→64」，两个变量混在一起，+12.1% 说不清是谁的功劳。
  反而是顺手加的 S4/S5（256 experts，同并行度同 batch，只差 QAG 开关）
  给出了唯一干净的净收益 +15.6%。**下次列实验矩阵时先把每轮的 diff 写出来核对**

### B.7 splash attention 全转置流水线（判定：我们的形状上零收益）

「全转置流水线」把注意力整条算路翻过来：`S.T = K@Q.T`（NT gemm）+
`O.T = V.T@S.T`，输出在 kernel 外做一次 swapaxes。好处是 `S.T` 可以直接算出来、
不需要显式转置。**这个收益是真的、已复现，但它随序列长度和有无掩码急剧变化，
而我们两条都不占。**

| 场景 | 基线 MFU | 转置版 | 比值 | 判定 |
|---|---|---|---|---|
| **seq 4096 / causal（= 生产形状）** | 27.0% ／ 19.7% | 27.2% ／ 19.4% | **1.0061x 与 0.9836x** | **两次跨在 1.0 两边，噪声底 1.72% ⇒ 收益为零** |
| seq 8192 / causal | 32.5% | 34.0% | 1.0465x | 勉强出噪声 |
| seq 16384 / causal | 34.6% | 37.2% | **1.0728x**（5 轮 × 15 次中位数） | 真实但有限 |
| seq 4096 / **FullMask（无掩码）** | 32.2% | 38.1% | **1.1813x** | 收益最大的那个 regime |

**为什么对我们没用**：这条优化的受益场景是 **无掩码 + 长上下文**（典型是扩散模型的
双向注意力，seq 常在 32k 以上）；我们是 **causal + seq 4096**。再折算一层：注意力在
295B MoE 里占比很小，专家层压倒性地大 —— 即便哪天上 16k 拿到 kernel 级 +7.3%，
端到端也只有 **约 +0.4 pp**（seq 4096 时约 +0.03 pp）。代价是一个新 kernel、
一套新布局、一次数值一致性验证，外加一个**尚不存在的转置版反向**。
**账不划算，2026-08-09 决定收线。**

**数值与代价（如果哪天要重启这条线，先看这里）**：
- 算法在精确算术下与原版恒等，只是换了操作数在乘法两侧的位置；
  实测最大相对误差 3.9e-3，而 bf16 的一个最小间隔约 7.8e-3 —— **约半个 ULP，
  属于重新结合律的正常量级，但不是逐位相同**。要声称「loss 逐位一致」必须另跑训练步对拍。
- 输出天然是转置的，要在 kernel 外转回来；本文的所有比值**都已把这次 swapaxes 计入**。
- 曾以为「必须把 V 换成 seq-minor 布局，否则隐式转置吃掉 14 pp」—— **这条是错的**：
  把收缩维选在第 0 维就不需要转 V。
- 现有反向拿的 `logsumexp` 本来就被广播成 `(h, NUM_SUBLANES, s)`，
  跟转置 kernel 天然产出的 `(8, bq)` 兼容，**接口不是障碍**。

**四条可复用结论（这轮真正的产出）**：

1. **`block=2048` 是硬件甜点，与 seq 无关。** seq 4096 和 16384 上最优块都是 2048
   （分别 = seq/2 和 seq/8），共同点是绝对值不是比例；往上撞 VMEM 墙，
   往下（1024）基线掉到 28.7%、转置版只有 0.84x。
2. **测量噪声底 1.72%**（同配置 5 轮 × 每轮 15 次中位数）。**小于这个数的差异一律不算数** ——
   本轮就靠它把一个 10.8% 修正成 7.3%。
3. **head_dim=128 时 35% 那堵墙，成因不只是 MXU 形状（K/N 各浪费一半 ⇒ 理论上限 50%），
   更是寄存器压力**：online softmax 下 Q@K 的输出必须在寄存器里活到 max 规约算完，
   撑不下就溢出 VMEM，MXU 随之停摆。**所以 50% 的理论上限和 35% 的实测之间那 15 pp，
   要从「减少 QK 输出的寄存器存活期」去找**，而不是继续在矩阵形状上想办法。
   两个方向：kernel 内再切一层寄存器级分块（让行最大值边流边算）、
   以及用一个 logit 上界替代 running max 直接消掉 max 规约和重缩放
   （我们生产的 `use_max_logit_estimate=30` 就是后者的粗糙版）。
4. **转置版的反向在数学上是干净的**（推导过，未实现）：dV / dP.T / dQ.T / dK
   在转置布局下全部原生，一次显式转置都不需要 —— 常规写法为了拿到 `P.T` 得专门转一次。
   真做的话反向可能比前向更受益。

> 🔁 **两条方法论教训。**
>
> **① 验证载体必须够格。** 前四轮实验全是负收益（0.77x–0.91x），因为我拿一个
> **基线 MFU 只有 4.2%** 的玩具 Pallas kernel，去测一个需要 35% 量级载体才显现的
> 优化 —— 测到的全是我自己实现的瓶颈。换成真 `make_splash_mha`（基线 32.2%）后，
> 同样的改动立刻从 −9% 变成 +18%。**做这类验证前先量基线 MFU，跟生产对齐了再比。**
>
> **② 先看自己的形状，再决定要不要往下做。** `seq=4096 causal` 的 1.0061x
> 上午就拿到了，那正是我们训练的形状。当时就该停下来问「那还测什么」，
> 结果被「+18% 复现出来了」带着走，又扩了 8k/16k。**复现成功 ≠ 对我们有用。**

</details>

---

## 附录 C：编译与环境工程

<details>
<summary><b>编译机制、缓存、scan、快速迭代环境（点开）</b></summary>

### C.1 编译时间三个口径，别混用

| 口径 | 从哪儿看 |
|---|---|
| **XLA 自报（推荐）** | `deepsea_compiler_base.cc:989] END_TO_END stage duration: 43.69s` |
| JAX 侧逐个 jit | `JAX_LOG_COMPILES=1` → `Finished XLA compilation of jit(train_step) in 45.15 sec` |
| 墙钟 | `TRIAL_START` → `completed step: 0`，含缓存 IO、数据管线 |

同一轮实测：42.76 / 45.15 / 52.2 s。三者都对。**报数时务必写明口径。**

> **`step 0` 不等于编译时间。** MaxText 先编译完再进步循环，`step 0` 只是一步普通训练。

### C.2 编译时间不随层数变（因为有两个 scan）

| 层数 | XLA `END_TO_END` |
|---|---|
| 5 | 42.41 s |
| 20 | 43.12 ~ 44.77 s（五次） |
| 80 | 44.29 s |

**层数变 16 倍，编译只涨 4.5%。** 机制上是**两份 layer body** —— 首层是 dense
（`first_num_dense_layers: 1`），结构与 MoE 层不同，不能进同一个 scan：

```
[SCANDBG2] main lax.scan length=1  unroll=1   ← dense 段
[SCANDBG2] main lax.scan length=19 unroll=1   ← MoE 段（20 层时）
```

**层数只改变 scan 的 trip count，不进 HLO 规模。**

> **所以「层多所以编译久」在 scan 模式下天然不成立。** 听到它就该去翻日志时间戳。
> 我当初接受得太快，把一个 stall 记成了"编译慢"，多耗了一天。

> **排查技巧**：这个模型走 **NNX** 路径（`nnx_decoders.py`），不是 Linen 的 `decoders.py`。
> 全树共三处 `lax.scan`，改 scan 行为时**先加 print 确认走到哪一处**，本项目为此打偏过两次补丁。

### C.3 编译缓存：45 s → 0.87 s

**两个前提，缺一不可：**

1. **`dump_hlo=False`** —— MaxText 在 `dump_hlo=True` 时**禁用** JAX 编译缓存
2. **Pod 常驻** —— 缓存目录在容器内，一次性 Job 跑完就没了

| | 冷 | 热 |
|---|---|---|
| jit 编译次数 | 21 个 | **21 个（一样）** |
| 总编译时长 | 51.75 s | **3.38 s** |
| 最大单个 | 45.15 s | **0.87 s** |
| 启动 → step 0 | 83.8 s | **32.8 s（−61%）** |

**准确说法不是"第二轮跳过编译"，而是"编译次数一模一样，但那个 45 秒的塌成了 0.87 秒"。**
只有超过阈值的模块走持久缓存，其余 19 个小模块每次现编，合计约 2.5 s。

**什么会让缓存失效**（缓存键是编译后的 HLO）：

| 改动 | 失效？ |
|---|---|
| XLA flag | **是** —— 扫 flag 的实验每轮都冷启 |
| `steps` | **是** —— `learning_rate_schedule_steps` 默认继承它，被编进 HLO 常量。**扫参数时别顺手改步数**（实测 4→8 让启动从 32.8 s 变 80.2 s） |
| 层数 / batch / seq | 是 |
| `dump_hlo=True` | **直接禁用缓存** |

### C.4 重复性：单轮就能判 1% 级别的改进

同一环境同一配置五轮稳态 step：6.0900 / 6.0851 / 6.0852 / 6.0847 / 6.0860 s。

**极差 5.3 毫秒（0.09%），单轮内抖动 ±2–4 ms。不需要跑多轮取平均。**
同时说明 `profiler` 和 `dump_hlo` 对稳态吞吐没影响，只影响启动段。

### C.5 快速迭代的环境形态

**把"一次性 Job"换成"常驻环境"**，一晚上能跑十几轮：

- N 个 Pod 跑 `sleep infinity` 占住 TPU 切片，代码预先解包在容器里
- 每轮用 `kubectl exec` **并行**在全部 Pod 上起同一条命令
- 编译缓存落在容器内固定路径，跨轮保留

**关键限制：多机 TPU 必须所有 Pod 同时执行。** 只 exec 进一个 Pod 跑 JAX 会卡在建 mesh：

```
RuntimeError: Unable to initialize backend 'tpu': DEADLINE_EXCEEDED:
TPU initialization failed: Failed to connect to <peer>:8471
```

> ⚠️ 更糟的是，那个失败的进程会**抓住 `/dev/vfio/*` 不放**，之后所有训练报
> `Device or resource busy; Couldn't open iommu group`，只能重建 pod。
> **看代码可以单 Pod 进；真跑必须齐步走。**

收益：一次性 Job 每轮要重新调度、拉 1.73 GB 镜像、建切片、冷编译；
常驻环境下一轮 30 步实验 **275 s → 209 s**，启动到第一步 **83.8 s → 32.8 s**。
256 芯片上一轮 8 步实验约 **6 分钟**。

### C.6 自建 v7 节点池：四个卡点

1. **必须用 workload policy，不是 placement policy。** v5p 惯用的 `--placement-type=COMPACT`
   和裸 `--tpu-topology` 在 v7 上都会被拒
2. **`--accelerator-topology` 必须写在 workload policy 上。** 缺了它才会报
   `does not support TPU topology with group placement policy and workload policy at the same time`。
   建池时 `--tpu-topology` 与 `--placement-policy` **两个同时传**、拓扑一致即可
   （上游 tpu-recipes ironwood 配方就是这么写的）。
   ⚠️ 早前这里写「不要再传 `--tpu-topology`」，**是错的，2026-08-07 已更正**
3. **必须显式给 `--scopes=cloud-platform`** —— 默认 scope 存储只有 `devstorage.read_only`，
   表现是**下载代码正常、写输出时 403**。**节点池 scope 不可修改，只能删掉重建**。
   桶上给 IAM 也没用，IAM 和 OAuth scope 是两层
4. **DWS flex-start 不能空跑等你** —— API 层强制 flex-start 必须开 autoscaling，
   而 autoscaling 意味着没 workload 就 0 节点。要常驻只能放一个 `sleep infinity` 撑住。
   另外 flex-start 节点**最长 7 天**，且不支持 reservation / Spot

### C.7 规模缩放的三个量化结论

**16 芯片 / 20 层 vs 64 芯片 / 80 层**（等比缩放，每芯片"层份额"恒为 1.25）：

| | 64 芯片 / 80 层 | 16 芯片 / 20 层 | 差 |
|---|---|---|---|
| TFLOP/s/chip | 445.1 | 410.7 | **−7.7%** |
| 首行日志 → step 0 | 174.2 s | 124.9 s | 快 1.4× |

**这 7.7% 从哪来**：用 `TFLOP/s/device ÷ tokens/s/device` 得每 token 的 FLOPs，
80 层 138.75 GFLOP / 20 层 38.16 GFLOP，比值 **3.636**（不是 4.0）。
按 `F(L) = a·L + b` 两点求解：每层 1.676 GFLOP/token，**层无关项 4.635 GFLOP/token**
（embedding 查表、12 万词表的 logits 投影、loss）。

层无关部分占比：**80 层时 3.3% → 20 层时 12.1%**，翻了近 4 倍。
反解两类工作的效率：decoder 层内 ≈ 19.9% MFU，层无关部分 ≈ 10.1% MFU。

> ⚠️ **这是两点拟合两个未知数，恰定解，没有自由度做检验。** 它自洽但不构成证明。
> 另一个候选（`2x2x4` 拓扑有一维只有 2，环绕链路退化）同样没有独立证据。
> **判别实验是跑 64 芯片 / 20 层**，尚未做。

**小规模真正省的不是编译，是排队。** 编译几乎不随规模变，端到端只快 1.4×
（省的主要是 16 台主机建切片的那 60 秒）。真正的价值是 **4 个节点比 16 个节点容易调度得多**。

</details>

---

## 附录 D：延伸阅读

| 文档 | 内容 |
|---|---|
| [QUICKSTART-v7.md](QUICKSTART-v7.md) | **最优配方 + 端到端复现**，照着跑就能拿到 580 |
| [QUICKSTART-v5p.md](QUICKSTART-v5p.md) | v5p 版，含架构完整拆解 |
| [TUNING-v5p.md](TUNING-v5p.md) | v5p 调优实践，含 tokamax LUT 根因的完整推理过程 |
| [EXPERIMENT-LOG.md](EXPERIMENT-LOG.md) | 完整实验档案，12 个 bug 的复盘 |
| [MAXTEXT-PORTING-GUIDE.md](MAXTEXT-PORTING-GUIDE.md) | 把别的模型移植到 MaxText 的通用范式 |
