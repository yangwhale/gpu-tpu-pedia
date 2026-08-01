# Hunyuan3-295B on TPU v5p —— 调优记录

跑通是 [QUICKSTART-v5p.md](QUICKSTART-v5p.md) 的事，这份只管**怎么更快**。
方法论、踩过的坑、以及「哪些结论能跨平台搬」，跟 [TUNING-v7.md](TUNING-v7.md) 是一套。

---

## 1. 当前水位

| | 值 | 来源 |
|---|---|---|
| 规模 | 256 芯片（`ct5p-hightpu-4t` × 64，拓扑 `4x8x8`） | QUICKSTART-v5p §5.1 |
| 稳态 step | **63.199 s** | 2026-08-01 第三次复现，与文档差 +0.05% |
| TFLOP/s / chip | **160.98** | |
| **MFU** | **35.07%** | |
| 整机吞吐 | 265,588 tok/s | |
| BF16 峰值 / chip | 459 TFLOPS | |
| HBM / chip | 95.74 GB，实测贴顶 | |

**对照**：同模型 v7 64 芯片 MFU **19.29%**，GB300 64 GPU BF16 **31.6%**。
v5p 的 35.07% 是三个平台里最高的 —— 这是本文档所有工作的起点，
也意味着**v5p 的可压榨空间比 v7 小得多**，别拿 v7 的期望值套过来。

---

## 2. 从 v7 搬过来什么

v7 那边跑了 34 个开关组合（[TUNING-v7 §7](TUNING-v7.md#7-消融总表34-个开关组合一次跑完的账)）。
按 [§7.5 那条教训](TUNING-v7.md#75-最贵的一课小规模筛选选不出赢家)，
**跨规模都不能直接外推，跨平台更不能**。所以这里只把它们当作**候选假设**，逐条重测。

### 2.1 直接搬不过来的

| v7 结论 | 为什么在 v5p 上不成立 |
|---|---|
| FP8 (`fp8_full`+qwix+tile 1536) **+8.25%** | **v5p 没有 FP8 加速，FP8 峰值 = BF16**。算力侧收益为零 —— 但**通信字节减半**这一条仍然成立，所以还是要测，只是预期机理完全不同 |
| 删 8 个 SparseCore offload flag（v7 上零收益） | v5p 上这组 flag **值 4.07 pp MFU**，是最值钱的一组。**绝对不能删** |
| `shard_exp_on_fsdp` 64 芯片崩（192 除不尽 128 device） | v5p 是 MegaCore，**1 device = 1 chip**，256 device。192 除不尽 256，**预期同样崩** |

> 第二条特别值得记：同一组 flag，v7 上零收益、v5p 上值 4 个百分点。
> **XLA flag 的收益是平台相关的，跨代直接抄参数集会踩雷。**

### 2.2 值得在 v5p 重测的

| 候选 | v7 实测 | v5p 预期 | 为什么值得测 |
|---|---|---|---|
| `remat_policy=full` + `decoder_layer_input=remat` | 16 芯片 +1.22% / 64 芯片 −0.74% | 未知 | v5p HBM 已贴顶，去掉 offload 未必装得下 |
| MoE tile 逐通路调优 | tile **必须等于** `base_moe_mlp_dim`=1536 | 有望 | v5p 当前 `mlp_dim=1024`，**除不尽 1536**，跟 v7 上崩掉的那档一模一样 |
| `use_ring_of_experts` + `num_moe_token_chunks` | EP 下挽回 34 pp | 未知 | 纯 FSDP 下能否藏通信，v7 没测完 |
| `use_tokamax_gmm` | v7 上前几步极慢（稳态未知） | 未知 | v5p 上是否同样症状，可反过来帮助定位 |
| `ici_expert_parallelism` | 16 芯片 −71%，明确负收益 | **不测** | 大幅负收益的结论会传递（§7.5） |
| `use_2d_fsdp_sharding` 三件套 | −11.73% | **不测** | 同上 |

### 2.3 v5p 独有的候选

来自 [QUICKSTART-v5p §7](QUICKSTART-v5p.md)，v7 那边不适用：

- 优化器状态降 BF16（`mu_dtype`/`grad_dtype`）—— HBM 贴顶，省下来能换 batch
- `sa_use_fused_bwd_kernel` 当前是 `False`（v7 上是 `True`），值得回扫
- `--xla_tpu_dvfs_p_state` / `pcie_bandwidth_multiplier` 等 v5p 专属 flag

---

## 3. 方法论（沿用 v7 的教训，不重新踩）

1. **同批次内比**。跨集群、跨节点池比绝对值没有意义 —— v7 上换了机器基线就差 15%。
2. **丢前 3 步**。单步 `seconds` 字段受异步派发干扰会跳变，用日志时间戳算跨步斜率。
3. **失败项照样记**。OOM 要记需要多少 G，配置拒绝要记缺什么前置，报错要记原文。
4. **前几步慢 ≠ 稳态慢**。Pallas / Mosaic kernel 按实际矩阵形状做运行时编译，
   前几步天然慢；至少跑 24 步再看稳态。（这条是 v7 上真栽过的跟头）
5. **小规模只用来排除**。任何要采纳的改动，必须在 256 芯片复测。

---

## 4. 结果记录

每跑完一轮回填一行。**正收益、负收益、零收益、崩溃，一视同仁。**

| 日期 | 实验 | 改了什么 | step (s) | TFLOP/s/chip | MFU | Δ | 结论 |
|---|---|---|---|---|---|---|---|
| 2026-07-30 | baseline | QUICKSTART-v5p §5.3 参数集 | 63.17 | 160.98 | 35.07% | — | 起点 |
| 2026-08-01 | **基线复现** | 同上，换 us-central1-a spot 新池 | **63.199** | 160.909 | **35.05%** | +0.05% | 可复现，本轮全部 Δ 以此为准 |
| 2026-08-01 | tile 1536 | `TILE_MLP=1536`（= `base_moe_mlp_dim`） | 66.421 | 153.105 | 33.36% | **−5.10%** | **负收益**。v7 上这一项是 +8.25%，**跨代不成立**（见下） |
| | | | | | | | |

### 4.1 第一条跨代反例：MoE tile

v7 上实测的规律是「tile **必须等于** `base_moe_mlp_dim`(1536)」——
1024 直接 `AssertionError: v=1536 bv=1024`，512 能整除但慢 3.9%，1536 快 **8.25%**。

**v5p 上这条完全不成立**：

| tile mlp_dim | v7（16 芯片） | v5p（256 芯片） |
|---|---|---|
| 1024（默认） | **崩**（断言失败） | 63.199 s ← **最优** |
| 1536（= 中间维） | **+8.25%** | 66.421 s（**−5.10%**） |
| 512 | −3.90% | 见下一行 |

同一个模型、同一个维度、同一个参数，**一个平台上是最大收益，另一个平台上是最大损失**。

原因方向（待 trace 佐证）：v7 上 1024 触发的是 kernel 的**形状校验失败**，
说明那条路径根本没跑起来；v5p 上 1024 能跑，说明两代的 Mosaic kernel
对分块的处理逻辑不同。**这是"XLA / kernel 层面的结论一律不跨代"的第二个实例**
（第一个是 SparseCore flag 组：v7 零收益、v5p 值 4.07 pp）。

---

## 5. 环境

复用 [QUICKSTART-v5p §3](QUICKSTART-v5p.md) 的建法，**留在 us-central1**。

```bash
gcloud container node-pools create np-v5p-256 \
  --cluster=CLUSTER --region=us-central1 \
  --node-locations=us-central1-a \
  --machine-type=ct5p-hightpu-4t --tpu-topology=4x8x8 --num-nodes=64 \
  --spot --scopes=cloud-platform
```

2026-08-01 实测：**64 台一次开出，全部 Ready**，不用排队、不用预留容量。
v5p 的裸容量是充足的 —— 这跟同期 v7 抢不到卡的处境完全不同，**不要把 v7 的容量策略搬过来**。

> `--scopes=cloud-platform` 不能漏 —— 默认只有 `devstorage.read_only`，
> 写 GCS 会 403，而且**节点池的 OAuth scope 建好之后改不了**，只能删了重建。

---

## 6. Trace 教学：每改一个参数，看图变了什么

这一节是**过程记录**，不是结论汇总。每一轮都按同一个格式走：
**当前版本的图长什么样 → 从图上看出什么问题 → 改哪个参数 → 改完图长什么样**。

### 6.1 抓 trace 的固定流程

四步，每一步都有坑，按这个来不会返工。

**① 跑一轮带 profiler 的**

```bash
bash run.sh <name> \
  base_output_directory=gs://<bucket>/hy3out \
  profiler=xplane skip_first_n_steps_for_profiler=4 profiler_steps=3 \
  upload_all_profiler_results=False
```

- ⚠️ **`base_output_directory` 必须指到 GCS**。`run.sh` 默认写 `/tmp/hy3out`，
  那是 pod 内的路径，pod 一删 trace 就没了，c2xprof 也读不到。
- `skip_first_n_steps_for_profiler=4` —— 前几步含编译和异步派发的假读数，
  抓进去只会看到一堆一次性开销。
- `profiler_steps=3` 足够。抓多了文件几个 G，上传慢。
- **性能数字不要从这一轮取**。profiler 本身有开销，Δ 要用不带 profiler 的干净跑。

**② 传 XProf 拿链接**

```
mcp__c2xprof__c2xprof_upload(gcs_path="gs://.../<host>.xplane.pb",
                             project="<gcp-project>")
```

- `project` **必传**，服务端自动解析拿不到项目会直接报错。
- 单文件上传实测 66–75 s，而 MCP 客户端硬超时是 60 s，**大概率超时**。
  超时不是失败，任务还在跑；重试或改用 ssh 直接调 `c2xprof.par`。

**③ 截图**

用 Chrome 打开 XProf 链接，`trace_viewer` 里缩到 **30 ms 左右**的窗口才看得清算子交替。
整屏截图只能看出「lane 很满」，看不出谁在等谁。

**④ 落文档**

链接 + 截图 + **一句话说明这张图要看的是什么**。没有第三条，图就是装饰。

### 6.2 读图的三条纪律（v7 上栽出来的）

1. **不要在单条 lane 上算「通信与计算的交集」**。XLA Ops lane 是严格嵌套的，
   交集恒为 0，算出来是同义反复，不是「完全没重叠」。
2. **异步 ≠ 已掩盖**。`-start` / `-done` 成对出现只说明它是异步发的；
   真正要看的是 `-done` 上等了多久。
3. **要一个能配平到 100% 的分解**，用**自用时间**（每个算子减去子算子）。
   否则 `while` 这类容器算子会被重复计数，得出「掩盖率 83%」这种自相矛盾的结论。

> 这三条在 v7 上分别踩过一次，四个版本的结论被推翻了三次。
> 详见 [TUNING-v7 §4.3.1](TUNING-v7.md#431-实战第一轮-trace-是怎么读出结论的教学)。

### 6.3 轮次记录

| 轮次 | 配置 | step (s) | XProf 链接 | 截图 | 从图上看到什么 |
|---|---|---|---|---|---|
| R0 | 基线（tile mlp_dim=1024） | 63.199 | 待补 | 待补 | 待补 |
| R1 | tile mlp_dim=1536 | 待补 | 待补 | 待补 | 待补 |

> 组内讨论期 XProf 链接直接放这里；这些是 `corp.google.com` 域名，
> 外部访问不到。**优化收尾后整块删除。**
