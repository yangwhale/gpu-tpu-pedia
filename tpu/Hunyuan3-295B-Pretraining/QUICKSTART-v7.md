# 混元 3（295B-A21B）在 TPU v7 (Ironwood) 上预训练 — Quick Start

用 MaxText 在 TPU v7 上跑腾讯混元 3。代码与 v5p 完全共用，**差别只在拿机器的方式、
XLA flag 集、和读数时的单位换算**。

| | |
|---|---|
| 模型 | Tencent Hunyuan 3，295B 总参 / 21B 激活，80 层 MoE |
| 平台 | TPU v7 Ironwood，64 芯片（`4x4x4`，16 台） |
| 框架 | MaxText（nnx），代码在 [`yangwhale/maxtext` 的 `hunyuan3` 分支](https://github.com/yangwhale/maxtext/tree/hunyuan3) |
| 精度 | BF16 计算 / FP32 主权重 |
| **当前水位** | **step 20.43 s · 445.1 TFLOP/s/chip · MFU 19.29% · 205,314 tok/s** |
| **状态** | 🔄 **跑得通，但没调完**。目标是 DSV3 在同硬件的实测水位 26.6%，**还差 1.38×** |

> **先读 [v5p Quick Start](QUICKSTART-v5p.md)。** 模型架构、代码来源、
> 12 个改动文件、`prep.sh` / `run.sh` 的用法，两个平台完全一样，这份不再重复。
> 本文只写 **v7 不一样的地方**。

> **这份文档的定位跟 v5p 那份不同。** v5p 是「调优收敛、照着跑就能复现」；
> v7 是「能跑，水位记录在此，已知的死路也记在此」。
> 拿它当**起点**，不是终点。

---

## 1. v7 与 v5p 的四个实质差异

| | v5p | v7 Ironwood |
|---|---|---|
| **device : chip** | 1 : 1（MegaCore） | **2 : 1** |
| **BF16 峰值 / chip** | 459 TFLOPS | **2,307 TFLOPS**（5.0×） |
| **拿机器** | `--tpu-topology` + `--spot` 直接建 | **必须先建 workload policy**，见 §2 |
| **XLA flag** | 25 个，SparseCore 卸载那组值 4.07 pp | **只带 15 个**，SparseCore 那组收益 **0** |

第四条是本项目最反直觉的一条发现，展开在 §5.2。

**其余全部相同**：模型代码、配置、`prep.sh`、`run.sh`、JobSet、暂存桶、VPC。

---

## 2. 拿到 v7 机器（跟 v5p 完全不是一回事）

v5p 一条命令就建完。tpu7x 会直接被拒：

```
Creation of a managed instance group with tpu7x-standard-4t machine type
with placement policy is not supported. Use workload policy instead.
```

### 2.1 先建 workload policy

gcloud 577 没有这个子命令（`resource-policies create` 只有 group-placement 等），
**只能走 REST**：

```bash
P=YOUR-PROJECT
TOK=$(gcloud auth application-default print-access-token)
curl -s -X POST -H "Authorization: Bearer $TOK" -H "Content-Type: application/json" \
 "https://compute.googleapis.com/compute/v1/projects/$P/regions/us-central1/resourcePolicies" \
 -d '{"name":"wp-4x4x4","workloadPolicy":{"type":"HIGH_THROUGHPUT","acceleratorTopology":"4x4x4"}}'
```

`acceleratorTopology` **必须带**，只给 `type` 会报
`does not support TPU topology with group placement policy and workload policy at the same time`。

### 2.2 建节点池：两种容量渠道

**A. Spot（先到先得，随时被抢）**

```bash
gcloud container node-pools create np-v7x-64 \
  --cluster=CLUSTER --project=$P --region=us-central1 --node-locations=us-central1-c \
  --machine-type=tpu7x-standard-4t --tpu-topology=4x4x4 \
  --placement-policy=wp-4x4x4 --num-nodes=16 --spot \
  --disk-type=hyperdisk-balanced --disk-size=200 --scopes=cloud-platform
```

**v7 必须用 hyperdisk**，普通 pd 会被拒。

**B. DWS flex-start（排队，拿到之后不会被抢）**

池子从 0 节点起，autoscaling 触发一次排队式容量请求。三个要素缺一不可：

```bash
gcloud beta container node-pools create np-v7x-flex \
  --cluster=CLUSTER --project=$P --region=us-central1 --node-locations=us-central1-c \
  --machine-type=tpu7x-standard-4t --tpu-topology=4x4x4 --placement-policy=wp-4x4x4 \
  --flex-start --num-nodes=0 \
  --enable-autoscaling --min-nodes=0 --max-nodes=16 --location-policy=ANY \
  --reservation-affinity=none \
  --disk-type=hyperdisk-balanced --disk-size=200 --scopes=cloud-platform
```

三个坑，每个报错都只说一半：

| 少了什么 | 报错 |
|---|---|
| `--enable-autoscaling` | `Flex start node pools require autoscaling enabled.` |
| `--num-nodes` 不是 0 | `Flex start node pools require initial node count to be set to 0.` |
| 用 `--total-max-nodes` 而不是 `--max-nodes` | 被判成 0，报 `Maximum node count 0 is not a valid size of TPU pod slice` |

### 2.3 空池跑 JobSet 必须加 `NODEPOOL=`

`run.sh` 默认给 JobSet 打 `exclusive-topology` 注解。该机制要求
**先把 leader pod 调度上去**，follower 才能抄它的 `gke-nodepool` 选择器。
flex-start 池是 0 节点，leader 永远落不了地：

```
admission webhook "vpod.kb.io" denied the request:
follower pod node selector for topology domain not found.
missing selector: cloud.google.com/gke-nodepool
```

后果很隐蔽：**只创建出 1 个 pod**，autoscaler 也只看得见 1 个 pending，
它据以决策的信息是残缺的。解法是把节点池写死进 `nodeSelector`：

```bash
NODEPOOL=np-v7x-flex PLATFORM=v7 bash run.sh myrun
```

**验证方法：提交后数 pod 个数，应该等于 16。**
固定节点数的 spot 池不需要这个参数。

### 2.4 容量：先查再建，别反复试

tpu7x 的抢占式容量是 **zone 级共享**的，跟你的项目和配额是两回事：

- **配额**决定你能不能**申请**
- **容量**决定你能不能**拿到**

判断"值不值得等"，查这一个数就够，比反复建池探测便宜得多：

```bash
gcloud compute instances list --project=ANY-PROJECT-IN-SAME-ZONE \
  --filter="machineType~tpu7x AND status=RUNNING" \
  --format='value(zone,scheduling.provisioningModel)' | sort | uniq -c
```

2026-07-30 一天之内这个数从 **152 台**（608 芯片，全被占满）降到 **47 台**。
占满的时候我们连 4 台都拿不到，两个项目、两套配额、同一 zone 同时为 0 ——
**换项目、提配额都无效**。抢占式没有排队，先到先得，
所以队列前面站满人时新请求既拿不到也不会排上，只会一直 `PROVISIONING` 到超时。

> 还有一层：`PREEMPTIBLE-TPU7X-per-project-region` 的配额是**全项目共享**的。
> 上限 64 芯片、别人已占 8 芯片时，`4x4x4`（64 芯片）这种**原子切片**就永远排不下
> —— TPU 拓扑没有 48 / 56 这种中间档，**拿不满等于拿不到**。

---

## 3. 跑起来

前置（JobSet、暂存桶、VPC）与 v5p 完全相同，见
[v5p Quick Start §3.2–§3.4](QUICKSTART-v5p.md)。

```bash
cd maxtext-hunyuan3/
export GCS_STAGE=gs://YOUR-STAGE-BUCKET/hy3
export IMAGE=us-docker.pkg.dev/YOUR-PROJECT/gcr.io/YOUR-maxtext-latest:runner

bash prep.sh                      # 与 v5p 共用，改了代码才要重跑
PLATFORM=v7 bash run.sh myrun     # flex-start 池加 NODEPOOL=<池名>

kubectl logs -f job/hy3-myrun-slice-job-0 -c jax-tpu
```

### 3.1 单位换算 —— v7 最容易出错的一步

**v7 是 2 device / chip**，而框架日志一律按 **device** 报：

```
per-chip TFLOP/s = 日志里的 TFLOP/s/device × 2
MFU              = per-chip ÷ 2307
```

例：日志报 `TFLOP/s/device: 222.56` → per-chip **445.1** → MFU **19.29%**。

> **方向搞反就是 4 倍的误差**（一次 ×2 变 ÷2）。跨代际对比之前
> 必须先确认 device:chip 比例 —— v5p 是 1:1 不用换算，v7 要 ×2。

### 3.2 读日志

1. **先确认 `16/16 Running` 再看日志。** TPU 切片全有全无，人不齐时活着的 pod
   会报 `GetSliceInfo can only be invoked after a slice is built` —— 症状不是病因。
2. **判错看最早那条，不是日志尾。** 配置非法会先把 TPU 拉起来再退。
3. **step 0 含编译，step 1/2 是 JAX 异步派发的假读数**，稳态取 step ≥ 3。
4. **v7 编译要 10–17 分钟**，比 v5p 的 9 分钟慢不少。第一次跑请耐心等。

---

## 4. v7 基线

### 4.1 硬件

| | 值 |
|---|---|
| 节点池 | 16 台 `tpu7x-standard-4t`，拓扑 `4x4x4` |
| 芯片数 | **64** |
| JAX device 数 | **128**（2 device / chip） |
| HBM | 192 GB / chip；单 device 可用 **94.74 GB** |
| BF16 峰值 / chip | **2,307 TFLOPS** |
| FP8 峰值 / chip | 4,614 TFLOPS |
| 总算力 | 147.6 PFLOPS BF16 |

### 4.2 实测

| 指标 | 值 |
|---|---|
| 参数量（框架报） | **298.786 B**（与 v5p 逐位一致） |
| 稳态 step | **20.43 s** |
| 日志 TFLOP/s/device | 222.56 |
| **TFLOP/s / chip** | **445.1** |
| **MFU** | **19.29%** |
| 整机吞吐 | **205,314 tok/s** |
| 每步 token | 4,194,304（128 × 8 × 4096） |
| 编译时间 | 10–17 分钟 |

**两个开跑就能查的健康检查**：`number parameters` 应为 **298.786 billion**；
`Total TFLOPs` 应为 **10169.38**（约 5 倍偏大说明 FLOP 公式没加 `HUNYUAN3`）。

### 4.3 完整参数集

与 v5p 共用的部分见 [v5p Quick Start §5.3](QUICKSTART-v5p.md)。
**v7 分支独有或取值不同的**：

| 参数 | v7 | v5p | 为什么不同 |
|---|---|---|---|
| `max_target_length` | **4096** | 8192 | v7 上短序列 + 大 batch 组合更优（+12.8% 吞吐） |
| `per_device_batch_size` | 8 | 8 | 相同；每卡 token 数 v7 是 v5p 的一半 |
| `sa_use_fused_bwd_kernel` | **True** | **False** | **两个平台相反**，各自实测的结果 |
| `use_tokamax_splash` | **True** | 不设 | v7 上值 +2.6%（与上一项合计） |
| `opt_type` | **adamw** | 默认 | 与下面两项配套 |
| `mu_dtype` / `grad_dtype` | **bfloat16** | fp32 | 优化器状态 16 → 12 B/param |
| `use_iota_embed` | **True** | 不设 | 省 embedding 显存 |
| MoE tile 参数（18 个） | **不设** | 全设 | v7 上没扫过，是个空白面 |

> ⚠️ **`sa_use_fused_bwd_kernel` 在两个平台上取值相反**，这不是笔误。
> 同一个开关在不同硬件代际上收益可以反号 —— **别把一个平台的调优结论直接搬到另一个**。

### 4.4 XLA flag：15 个，不要照抄官方 30 个

```
--xla_tpu_scoped_vmem_limit_kib=65472
--xla_enable_async_all_gather=true

# SparseCore 卸载组（9 个）—— v7 上收益是 0，保留只为与官方配方对齐
--xla_tpu_enable_sparse_core_reduce_scatter_v2=true
--xla_tpu_enable_sparse_core_collective_offload_all_gather=true
--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true
--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true
--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true
--xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true
--xla_tpu_enable_sparse_core_collective_aggregator=true
--xla_tpu_use_tc_device_shape_on_sc=True
--xla_sc_disable_megacore_partitioning=True

# 调度器组（4 个）—— 唯一值钱的一组，+6.6%
--xla_tpu_enable_latency_hiding_layer_scheduler=true
--xla_tpu_scheduler_percent_shared_memory_limit=150
--xla_tpu_enable_layer_scheduler_for_dependent_collectives=true
--xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=false
```

补到 26 个也能跑（收益 ±0），所以保持精简。
**但不要把官方那套一次全开** —— 那一轮死锁了，元凶是同时打开的
`use_tokamax_gmm`（见 §5.3），不是 flag 数本身。

---

## 5. 调优记录：涨了多少、什么没用、什么是死路

### 5.1 从首跑到当前

| 轮次 | 增量 | seq | pdbs | step | TFLOP/s/chip | MFU |
|---|---|---|---|---|---|---|
| V1 | 基线：2 个 XLA flag | 8192 | 4 | 25.11 s | 404.75 | 17.54% |
| y1 | + `use_tokamax_splash` + `sa_use_fused_bwd_kernel` | 8192 | 4 | 24.45 s | 415.16 | 18.00% |
| y2 | + adamw / bf16 优化器 + `iota_embed` | 8192 | 4 | 24.61 s | 412.88 | 17.90% |
| y3 | + SparseCore 卸载组（9 个 flag） | 8192 | 4 | 24.61 s | 412.56 | 17.88% |
| y4 | + 调度器组（4 个 flag） | 8192 | 4 | 23.08 s | 440.30 | 19.09% |
| z1 | y1 + 换 batch / 序列口径 | 4096 | 8 | 21.69 s | 418.89 | 18.16% |
| **c1** | **调度器组 × pdbs=8 / seq=4096** | 4096 | 8 | **20.43 s** | **445.12** | **19.29%** |
| c2 | c1 + 杂项组（补齐 26 flag） | 4096 | 8 | 20.45 s | 444.63 | 19.27% |

**相对首跑 +10.0%。** 各项贡献：

| 手段 | 贡献 |
|---|---|
| **调度器 flag 组（4 个）** | **+6.6%** |
| **pdbs=8 / seq=4096** | **+12.8% 吞吐**（TFLOP/s 只 +0.9%） |
| `use_tokamax_splash` + `sa_use_fused_bwd_kernel` | +2.6% |
| 杂项 flag 组（5 个） | ±0 |
| **SparseCore 卸载组（9 个）** | **±0** |
| 优化器 / 显存组 | −0.5%（省的是显存不是时间） |

### 5.2 最重要的一条否定结果

**SparseCore 集合通信卸载那 9 个 flag，在 v5p 上值 4.07 pp（13%），在 v7 上收益是 0。**

这条否定结果直接指出了瓶颈在哪：

> SparseCore 卸载改的是**通信在哪执行**，调度器改的是**通信和计算怎么重叠**。
> 前者无效说明**通信不是瓶颈**；后者有效（+6.6%）说明**通信没跟计算叠起来** ——
> 不是通信太慢，是它没藏住。

> ⚠️ **但不要把「这一组没用」外推到「同类的下一组也没用」。**
> 我当时正是这么推的：既然通信不是瓶颈，剩下的 flag 组期望收益也低，跳过。
> 结果调度器那组在我放弃之前已经跑完了 —— **+6.6%，当时的最优**。
> 消融的纪律是**每一组都要真跑**。

### 5.3 三条死路（都实测过，别再踩）

| 开关 | 后果 |
|---|---|
| **`use_tokamax_gmm=True`** | **死锁**，`stalled chips [7]`，连 step 0 都跑不完。2 次开 2 次挂、2 次不开 2 次通 |
| `shard_exp_on_fsdp=True`（FSDP=64 × DP=2） | **OOM**，用了 109.14 G / 94.74 G，**比不开还多 14 G** |
| `per_device_batch_size=12` | **OOM**，临时缓冲 95.17 G |

**`use_tokamax_gmm` 值得单独说。** 官方 Ironwood DSV3 配方里这个开关是 `True`
且能跑，说明 tokamax 后端本身没问题，**是它跟 Hy3 的形状不合** ——
最可能是 192 个专家：DSV3 是 256，分组矩阵乘的组数正好是 2 的幂。这条待确认。
它也是 `use_gmm_v2` 的强制前置，所以 gmm_v2 一并不可用。

**`shard_exp_on_fsdp` 为什么净亏**：这笔交易两头都动 ——
专家权重改按专家维切（收益），但 FSDP 宽度从 128 降到 64，
**非专家部分（attention 80 层 + embedding + dense 首层，约 7.2 B）每卡分片直接翻倍**。
省的抵不过多花的。根因是 **192 不是 2 的幂**，
代价不是"用不了这个开关"，而是"用它要拿一半 FSDP 宽度去换，不划算"。

---

## 6. 目标定在哪：为什么是 600–630 而不是 900

Ironwood 官方实测表（全部 bf16、synthetic、per-chip 口径）：

| 模型 | 类型 | chips | 序列 | TFLOP/s/chip | MFU |
|---|---|---|---|---|---|
| llama3.1-405b | **稠密** | 256 | 8192 | 1,261.4 | 54.7% |
| llama3.1-70b | **稠密** | 64 | 8192 | 1,207.1 | 52.3% |
| gemma4-31b | **稠密** | 64 | 8192 | 931.3 | 40.4% |
| **qwen3-235b-a22b** | **稀疏 MoE** | 256 | 4096 | **629.8** | **27.3%** |
| **deepseek-v3 671B** | **稀疏 MoE** | 256 | 4096 | **612.7** | **26.6%** |
| gpt-oss-120b | 稀疏 MoE | 256 | 8192 | 329.9 | 14.3% |
| **hunyuan3 295B（本项目）** | **稀疏 MoE** | **64** | **4096** | **445.1** | **19.29%** |

**900 以上全是稠密模型。** 稀疏 MoE 在 Ironwood 上的实际水位是
**600–630 TFLOP/s/chip（26–27% MFU）**，两个最接近 Hy3 的参照都在这条线上。

差距是结构性的，不是调参能翻越的：

- 稠密模型每个 token 走同一套权重，GEMM 又大又规整，MXU 能吃满
- MoE 每层要做一次路由、一次按专家分组重排、一次分组矩阵乘、一次还原。
  分组矩阵乘的每个子块只有 `tokens_per_expert × emb × moe_mlp` 那么大，
  而且组大小随路由结果浮动，**编译期拿不到静态形状**
- 还要加上 all-gather / reduce-scatter 把 192 份专家权重摊开又收回

**所以 v7 侧的目标是 600–630 TFLOP/s/chip，对应 step 压到 14–15 s。
当前 445.1，缺口 1.38×。**

> Hy3 的激活参数（21 B）比 DSV3（37 B）还少，结构也更简单
> （GQA 而非 MLA、192 专家而非 256），**没有理由跑不到同一水位** ——
> 差距来自配置，不是架构。

### 6.1 关于 FP8：先别指望

同一张表里 DSV3 的 FP8 数据：

| | BF16 | FP8 | |
|---|---|---|---|
| DSV3 TFLOP/s/chip | 612.66 | 743.46 | **+21.4%** |
| 对本精度峰值的 MFU | 26.6% | **16.1%** | 峰值翻倍但吞吐没翻 |
| 稠密 llama3.1-405b 同口径 | 54.7% | 41.8% | 稠密 FP8 涨 **+52.8%** |

**MoE 兑现不了 FP8 的两倍峰值，稠密可以。** 原因跟上面那条一样：
MoE 的时间大量花在路由、分组重排、all-to-all 和小块 GEMM 上，
这些环节不吃 MXU 峰值，降精度对它们几乎没帮助。

> ⚠️ **报 FP8 的 MFU 一定要说明分母。** 同一个 743.46，
> 对 FP8 峰值算是 16.1%，对 BF16 峰值算是 32.2% —— **差一倍**。

---

## 7. 下一步该做什么

参数扫描的收益已经明显变平（杂项 flag 组 ±0），继续盲扫期望很低。
按优先级：

### 7.1 抓 xplane trace（最该先做）

需要 trace 直接回答"时间花在**路由 / 分组重排 / all-to-all / offload 还是 GEMM**"。
在此之前所有优化都是猜。

注意：上一次开 profiler 的那轮因为 profiler 自身开销，没在窗口内跑出稳态，
**要单独给它更长的预算**。

### 7.2 查清 `use_tokamax_gmm` 死锁的根因

这是**唯一一个"官方有、我们用不了"的加速手段**，而且它是 MoE 的主计算路径。
怀疑是 192 专家（非 2 的幂）导致分组矩阵乘的组划分出问题。
如果能修，这可能是最大的单项收益。

### 7.3 `scan(unroll=N)`

`jax.lax.scan` 的 `unroll` 参数，MaxText 目前没用（走默认值 1）。
**`while` 循环体是 XLA 的调度边界，跨迭代不能重排** ——
循环体里只有 1 层时，第 N 层的 all-gather 没法藏进第 N−1 层的计算。
`unroll=N` 把这 N 层放进同一个调度域，延迟隐藏调度器才有发挥空间。

**§5.2 已经证明「通信没藏住」就是 v7 的瓶颈，所以这一项的机理和实测指向一致。**
改动约 10 行，建议按 1/2/4/8 扫，同时记吞吐、编译时间、HBM 峰值三条曲线。

### 7.4 MoE tile 参数

v5p 上设了 18 个（`{wi,wo} × {fwd,dlhs,drhs} × 3 维`），**v7 上一个都没设**。
这是个完全没探过的面。

---

## 8. 已知限制

| 项 | 状态 |
|---|---|
| **调优未完成** | 19.29% vs 目标 26.6%，差 1.38× |
| 数据集 | `dataset_type=synthetic`。loss 下降只证明「能算且不发散」 |
| `use_tokamax_gmm` | 死锁，根因未查清 |
| HF 权重 → Orbax 转换 | 未做 |
| 容量 | tpu7x spot 经常整个 zone 拿不到，见 §2.4 |

---

## 9. 延伸阅读

| 文档 | 内容 |
|---|---|
| [QUICKSTART-v5p.md](QUICKSTART-v5p.md) | **先读这份**：模型架构、代码来源、共用参数、v5p 基线 35.07% |
| [EXPERIMENT-LOG.md](EXPERIMENT-LOG.md) | 完整实验档案：全部轮次、12 个 bug 的复盘 |
| [MAXTEXT-PORTING-GUIDE.md](MAXTEXT-PORTING-GUIDE.md) | 把别的模型移植到 MaxText 的通用范式 |
| [maxtext-hunyuan3/](maxtext-hunyuan3/) | `prep.sh` / `run.sh` |
