# 混元 3（295B-A21B）在 TPU v7 上的性能调优实践

> **从 445 到 599 TFLOP/s/chip（MFU 19.3% → 26.0%）。这份文档讲每一步改了什么、为什么有效、值多少。**
>
> 只想拿最优配方直接跑 → **[QUICKSTART-v7.md](QUICKSTART-v7.md)**，那里有可照抄的完整命令。
> 这里讲的是**怎么得到那套配方的**，以及沿途否掉了哪些路。

---

## 1. 水位与目标

### 1.1 现在在哪

2026-08-04 实测，均为**完整 80 层**、seq 4096、BF16、合成数据、稳态取 step 4–7：

| 规模 | 配方 | step | **TFLOP/s/chip** | **MFU** | **tok/s** | tok/s/chip | 峰值 HBM |
|---|---|---|---|---|---|---|---|
| 256 chip 极限 **BF16** | `DP2×FSDP256` + tile + pdbs **16** | 30.40 s | **599** | **25.96%** | **1,103,757** | **4,312** | 92.33 G |
| 256 chip **FP8**（未调优） | 同上 + `fp8_full`+qwix | 29.46 s | 618 | 13.39%<sub>对FP8峰值</sub> | 1,139,022 | 4,449 | 92.80 G |
| 256 chip 推荐 | `DP4×FSDP128` + tile + pdbs 12 | 23.56 s | 580 | 25.12% | 1,068,372 | 4,173 | 91.94 G |
| 64 chip | `DP1×FSDP128` + tile + pdbs 12 | 23.54 s | 580 | 25.14% | 267,284 | 4,176 | 91.94 G |
| 64 chip **FP8** | `DP1×FSDP128` + tile + pdbs 10 | 19.15 s | 594 | 12.87%<sub>对FP8峰值</sub> | 273,987 | 4,281 | 86.20 G |
| 起点（2026-07-30） | `FSDP128` + megablox + pdbs 8 | 20.43 s | 445 | 19.29% | 205,313 | 3,208 | 74.20 G |

> **tok/s = device 数 × pdbs × seq ÷ step**；横向比只看 tok/s/chip。
> 参照：v5p 256 chips = **1,037** tok/s/chip，GB300 = **6,242** tok/s/GPU。
> **v7 每芯片吞吐是 v5p 的 4.16 倍、GB300 单卡的 69.1%**（调优前 51.4%）。

**目标 600–630 TFLOP/s/chip（26–27% MFU）。当前 599，差 0.2%。**

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

<!-- ===== TEMP:XPROF-LINKS  组内讨论期临时保留，优化收尾后整块删除 ===== -->
> **🔗 XProf session（需 Google 账号，仅供组内讨论期使用）**
>
> | Profile | Session |
> |---|---|
> | 4 芯片 `2x2x1` / 80 层 | http://xprof.corp.google.com/trace_viewer/chrisya-11640939633798411639 |
> | 16 芯片 `2x2x4` / 20 层（含 HLO dump） | http://xprof.corp.google.com/trace_viewer/chrisya-18130551067782033931 |
>
> session 会过期，过期后从 GCS 重新上传 `.xplane.pb` 即可。
<!-- ===== /TEMP:XPROF-LINKS ===== -->

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

## 3. 调优故事线：从 445 到 599

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

\* 换了一批机器，与前几行不同批次，绝对值不可直接比；同批次内的对照见 [附录 A](#附录-a全部消融数据)。

---

## 4. 四个可复用的方法论结论

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

## 5. FP8：跑通了，但一格都还没调

**当前 618 TFLOP/s/chip，对 FP8 峰值（4,614）MFU 只有 13.4%。DSV3 官方是 743.5（16.1%），
我们落后 20.3%。**

> ⚠️ **报 FP8 数字必须写清分母。** 618 对 BF16 峰值算是 26.8%，对 FP8 峰值只有 13.4% ——
> 差一倍。**FP8 的数只能跟 FP8 的数比。**

### 5.1 618 相当于 BF16 那边的 445 —— 起点，不是终点

| | BF16 | FP8 |
|---|---|---|
| 未调优起点 | 445 | **618（现在在这）** |
| 调优后 | **599**（+34.5%） | ？ |
| 其中 tokamax tile 一项 | **+17.4%** | **未测** |

**两个规模的 FP8 实测**（FP8 峰值 4614 为分母）：

| 规模 | 配置 | chip | 对 FP8 峰值 MFU | 同配置 BF16 | Δ | 峰值 HBM |
|---|---|---|---|---|---|---|
| 256 chip | `DP2×FSDP256` pdbs 16 | 618 | 13.39% | 599 | +3.2% | 92.80 G |
| 256 chip | `DP4×FSDP128` pdbs 12 | 608 | 13.18% | 580 | +4.8% | 94.35 G |
| **64 chip** | `DP1×FSDP128` **pdbs 10** | **594** | **12.87%** | 561 | **+5.9%** | 86.20 G |

> 64 芯片的 594 已经**超过 BF16 最优的 580（+2.4%）**，而且只吃 86.2 G ——
> **FP8 省下的显存反过来可以再喂 batch**，这条还没试。
>
> ⚠️ `pdbs 12 + FP8 + tile(512,2048,1536)` 在 64 芯片上**不是 OOM 而是 kernel 拒绝**
> （`MosaicTpuRaggedDot` 报错），见 [附录 B.2](#b2-崩溃--配置拒绝)。

**这两条赛道用的是不同的 kernel，各自的 tile 要分别调。** BF16 那边花一整轮找到
`tile(512, 2048, 1536)` 值 17.4%；**FP8 这条路上一次都没扫过。**

粗算：若 FP8 路径的 tile 能拿到同量级收益，`618 × 1.174 ≈ 726`（FP8 峰值口径 MFU 15.7%），
距 DSV3 的 743.5 只差 2.4%。

> ⚠️ **但别把这理解成「单靠 tile 就能追平 DSV3」。**
> 2026-08-05 查官方 recipe 发现 **DSV3 的 743.5 是开着 QAG 拿到的**
> （见 [§5.4.2](#542-qag先量化再通信从被专家数卡死到有解)）。
> 我们的 726 是「不开 QAG、只调 tile」的估计，**两个数字不同源**。
> 726 vs 743.5 只说明「调完 tile 大致到那个量级」，
> **不代表 tile 是唯一缺口** —— 我们还少一项人家有的优化。

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
以及 **QAG**（2026-08-05 查官方 recipe 确认，见 [§5.4.2](#542-qag先量化再通信从被专家数卡死到有解)）。
**三者各占多少，目前分不开。** 拿它当 FP8 的唯一标尺要留余地。

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

##### 先回答那个问题：默认情况下，通信传的确实是 bf16

`QwixDotGeneral.__call__` 只是包了一层 `dot_general_qt` ——
**量化发生在 dot_general 内部：算之前临时量化、算完输出 bf16**。
权重按 `weight_dtype=float32` 存、`dtype=bfloat16` 算。
**所以 FSDP 的 all-gather 传的是 bf16，一个字节都没省。**

这正是 [§5.1](#51-618-相当于-bf16-那边的-445--起点不是终点) 那个 Amdahl 天花板 746
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

##### 192 experts 撞墙的确切位置：是 config 校验，不是 kernel

`shard_exp_on_fsdp=True` 在 128 device 上直接被拒（[附录 B.2](#b2-崩溃--配置拒绝)）。
拦截点在 `configs/pyconfig_deprecated.py:1212-1215`，是**两条显式的前置校验**：

```python
if raw_keys["shard_exp_on_fsdp"] and raw_keys["num_experts"] % raw_keys["ici_fsdp_parallelism"] != 0:
  raise ValueError("shard_exp_on_fsdp requires num_experts is divisiable by ici_fsdp_parallelism.")
if raw_keys["shard_exp_on_fsdp"] and (using_tensor_parallelism(raw_keys) or using_expert_parallelism(raw_keys)):
  raise ValueError("shard_exp_on_fsdp requires ici_expert_parallelism = 1 and ici_tensor_parallelism = 1.")
```

**这一点很重要：约束是 `num_experts % ici_fsdp_parallelism == 0`，
不是「专家数必须是 2 的幂」。** 是 FSDP 宽度和专家数的整除关系，
而我们一直用 2 的幂当 FSDP 宽度，才让它看起来像是「192 不是 2 的幂」的问题。

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

##### 验证实验二：64 芯片判据（2026-08-05，进行中）

用 `fixed,-224,224` 重跑，并把整除 workaround 一起验了。
128 device / 16 层 / pdbs 8（缩层是为了压住 FP8 的编译时间，
**只判通/不通与组内相对趋势，绝对性能不与 §5.1 的 618 横比**）：

| 轮 | experts | 并行度 | calibration | `shard_exp` | 这一轮回答什么 |
|---|---|---|---|---|---|
| S0 | 192 | DP1×FSDP128 | — | — | BF16 参照点 |
| S1 | 192 | DP1×FSDP128 | absmax | off | FP8 参照点（无 QAG） |
| S2 | 192 | DP1×FSDP128 | `fixed,-224,224` | **on** | **预期崩**：`192 % 128 ≠ 0` |
| S3 | 192 | **DP2×FSDP64** | `fixed,-224,224` | **on** | **整除 workaround 能否成立** |
| S4 | 256 | DP1×FSDP128 | absmax | off | 256 experts 的 FP8 参照点 |
| S5 | 256 | DP1×FSDP128 | `fixed,-224,224` | **on** | **256 experts 能否开 QAG** |

配对读数：`S3 vs S1` 给出「192 走 workaround 的净代价」，
`S5 vs S4` 给出「256 experts 上 QAG 的净收益」——
S4/S5 必须成对才有意义，因为改 `num_experts` 后参数量、FLOP 口径、HBM 全变了，
**跨 experts 数横比无意义**。

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

按预期收益排序：

| # | 项 | 为什么值得 | 状态 |
|---|---|---|---|
| 1 | **官方 autotune 生成 cache 条目** | 替代 monkeypatch，可能比手调 tile 更好 | 未做 |
| 2 | **官方 28 个 XLA flag 组** | 单节点实测 0%，但那组全是跨机通信 flag，单节点本来测不出 | 256 芯片上待判 |
| 3 | **FP8**（`fp8_full` + qwix） | v7 上能编能跑（v5p 编译失败），单节点 −1.2% | 多卡待判 |
| 4 | `gmm_v2` + `tile_k` 整除 K | 上游在 v7 上实测调好 tile_k 后 +13.58% end-to-end | 未做 |
| 5 | ring of experts / `num_moe_emb_chunks` | v7 通信占 57.3%，这是它的目标场景 | 未做 |
| 6 | tile 邻域精扫（256 / 768 / 3072） | 当前最优是粗扫出来的 | 进行中 |

---

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
| `shard_exp_on_fsdp=True`（128 device） | 192 专家除不尽 128 device。拦截点是 `pyconfig_deprecated.py:1212` 的显式校验 `num_experts % ici_fsdp_parallelism != 0`，**不是 kernel 层** ⇒ 改 FSDP 宽度可绕，见 [§5.4.2](#542-qag先量化再通信从被专家数卡死到有解) |
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
- **取稳态先看序列再定窗口**：开 profiler 时 step 17 会有 ~90 秒尖峰（导 trace），
  按惯例取 `step ≥ 15` 求均值会得到 11.678 s，比真实稳态慢近一倍

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
2. **建池时不要再传 `--tpu-topology`** —— GKE 会自动附加 group placement policy 与之冲突。
   拓扑由 workload policy 的 `--accelerator-topology` 携带
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
