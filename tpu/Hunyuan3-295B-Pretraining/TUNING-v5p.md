# Hunyuan3-295B on TPU v5p —— 调优记录

跑通是 [QUICKSTART-v5p.md](QUICKSTART-v5p.md) 的事，这份只管**怎么更快**。
方法论、踩过的坑、以及「哪些结论能跨平台搬」，跟 [TUNING-v7.md](TUNING-v7.md) 是一套。

---

## 1. 当前水位

| | 值 | 来源 |
|---|---|---|
| 规模 | 256 芯片（`ct5p-hightpu-4t` × 64，拓扑 `4x8x8`） | QUICKSTART-v5p §5.1 |
| 稳态 step | **63.17 s** | 两次复现差 0.25% |
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
| | | | | | | | |

---

## 5. 环境

复用 [QUICKSTART-v5p §3](QUICKSTART-v5p.md) 的建法，两点不同：

- **走 reservation 不走 spot**。`us-central1-a` 的 v5p spot 今天抢不到；
  改用 `europe-west4-b` 的 ct5p 预留（1024 颗，空 424）。
- 集群 `chrisya-v5p-euw4`（region `europe-west4`，节点在 `-b`）。

```bash
gcloud container node-pools create np-v5p-256 \
  --cluster=chrisya-v5p-euw4 --region=europe-west4 \
  --node-locations=europe-west4-b \
  --machine-type=ct5p-hightpu-4t --tpu-topology=4x8x8 --num-nodes=64 \
  --reservation-affinity=specific --reservation=<预留名> \
  --scopes=cloud-platform
```

> `--scopes=cloud-platform` 不能漏 —— 默认只有 `devstorage.read_only`，
> 写 GCS 会 403，而且**节点池的 OAuth scope 建好之后改不了**，只能删了重建。
