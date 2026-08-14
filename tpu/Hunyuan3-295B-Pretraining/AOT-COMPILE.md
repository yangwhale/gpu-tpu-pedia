# AOT 提前编译：不占一张 TPU，先把账算清楚

> 2026-08-14 实测。模型 Hunyuan3-295B-A21B（80 层，第 0 层 dense，192 专家 top-8），
> 目标 `tpu7x-128`（= 64 芯片 4x4x4），配方与 [TUNING-v7](TUNING-v7.md) 的生产配置一致。
> 配套脚本 [`maxtext-hunyuan3/aot.sh`](maxtext-hunyuan3/aot.sh)。

## 一句话

MaxText 自带的 AOT（ahead-of-time compilation）能**在一台普通 CPU 机器上**，
为你手上根本没有的拓扑编译出完整的训练步 —— 编译器报的 OOM 跟真机一字不差。
**抢卡之前先跑它，比抢到卡再发现装不下便宜太多。**

## 为什么不需要 TPU

关键在源码注释里：它 *"generates shaped versions of state and data **without ever
constructing them**"* —— 只造张量的**形状**，从不真的把张量造出来。

XLA 编译器要生成代码，需要知道的是每个张量多大、怎么切、拓扑什么样，
**不需要真的有那些芯片**。MaxText 用 `jax.experimental.topologies.get_topology_desc`
造一副假的设备网格，把编译器骗过去。

日志里会有两条噪音，**可以无视**：

```
WARNING: could not determine TPU accelerator type ...
INVALID_ARGUMENT: Error: unexpected worker hostname ... TPU_WORKER_HOSTNAMES
```

本机确实没有 TPU，但编译不走那条路。看到最后打出
`Finished train_compile.py successfully!` 就是成了。

## 怎么跑

```bash
GCS_STAGE=gs://your-bucket/hy3 \
IMAGE=us-docker.pkg.dev/PROJECT/gcr.io/your-maxtext-latest:runner \
bash maxtext-hunyuan3/aot.sh probe1
```

核心就是把训练入口换成 `train_compile`，多传两个参数：

```
compile_topology=tpu7x-128       # 目标拓扑
compile_topology_num_slices=1    # slice 数
compile_xla_flags="<跟生产一模一样的 XLA flag>"
```

> ⚠️ **拓扑名按 device 数，不是芯片数。** v7 是 2 device/chip，
> 所以 **64 芯片 = `tpu7x-128` = 4x4x4**。写成 `tpu7x-64` 编出来的是 32 芯片的图，
> 数字全错还不报错。对照表在 `src/maxtext/utils/accelerator_to_spec_map.py`。

## 生产级流程：抢卡之前该跑哪几步

按这个顺序做，**全程零 TPU**，总耗时约 5 分钟。

### Step 0 · 准备（一次性）

```bash
# 代码推到 GCS（只有改代码才要重跑）
GCS_STAGE=gs://your-bucket/hy3 bash maxtext-hunyuan3/prep.sh
```

任意一台带 docker 的机器都行 —— 我们在 GKE 节点、n4-highmem-80、
c4-highcpu-32 三种环境上跑过同一个镜像，结果一致。

### Step 1 · 先确认目标拓扑名（写错不报错，数字全废）

拓扑名按 **device 数**，v7 是 2 device/chip：

| 芯片 | 拓扑名 | mesh |
|---:|---|---|
| 16 | `tpu7x-32` | 2x2x4 |
| 32 | `tpu7x-64` | 2x4x4 |
| **64** | **`tpu7x-128`** | 4x4x4 |
| 128 | `tpu7x-256` | 4x4x8 |
| 256 | `tpu7x-512` | 4x8x8 |
| 512 | `tpu7x-1024` | 8x8x8 |

完整对照在 `src/maxtext/utils/accelerator_to_spec_map.py`。
**同时核对你的 GKE 节点标签 `cloud.google.com/gke-tpu-topology` 要跟 mesh 一致。**

### Step 2 · 扫出最大可行 batch

从你想要的 `pdbs` 开始，OOM 就减半，能过就往上加。每次约 2 分钟：

```bash
GCS_STAGE=... IMAGE=... PDBS=12 bash maxtext-hunyuan3/aot.sh probe-p12
```

判据看这一行 —— **只看 `temp_size`，别看总和**：

```
Memory analysis: CompiledMemoryStats(... temp_size_in_bytes=80477396448 ...)
```

`temp_size ÷ 1e9` 与 **94.74 GB/device** 比。留 10 GB 余量比较稳。

失败时编译器会直接告诉你差多少：

```
RESOURCE_EXHAUSTED: HLO temporaries (100.13G) exceeds available HBM (94.74G)
```

### Step 3 · 存下编译产物

确定配置后，加 `compiled_trainstep_file` 再编一次，把产物传 GCS。
训练侧拉下来指向本地路径即可跳过编译 —— 实测启动省 2.9×（见下方端到端章节）。

### Step 4 · 这时候才去抢卡

前三步都过了，再提交真实训练任务。

> **为什么值得**：AOT 一次约 2 分钟 CPU 时间，成本可忽略；
> 而「抢到 64 张卡 → 跑十几分钟编译 → OOM 退出 → 重排队」这个循环，
> 一次就是小时级，还占着别人的容量。

### 三条踩坑前置提醒

1. **XLA flag 必须跟生产逐字一致 —— 它们是承重墙，不只是调性能。**
   实测把整组 flag 拿掉，同一个配置**直接 OOM**：

   | | `temp_size` | 结果 |
   |---|---:|---|
   | 带生产 flag（16 个） | **80.48 GB** | ✅ 还剩 14 GB 余量 |
   | 完全不带 flag | **95.01 GB** | ❌ 超上限 94.74 GB —— **只差 0.3%** |

   **这组 flag 把峰值临时空间压低了 15.3%**，正好是「能跑」和「跑不了」的分界。

   **进一步分组定位，找出到底是谁在扛**（同配置，只换 flag 子集）：

   | flag 子集 | `temp` | 结果 |
   |---|---:|---|
   | 完全不带 | 95.01 GB | ❌ HBM OOM |
   | 只 `scoped_vmem_limit` | 94.81 GB | ❌ HBM OOM（几乎没省） |
   | 只 **SparseCore 卸载那 9 个** | **95.01 GB** | ❌ **与不带 flag 一模一样** |
   | **`scoped_vmem_limit` + 调度器组** | **80.48 GB** | ✅ **通过** |

   **两条结论：**

   - **SparseCore 卸载那 9 个 flag 对显存的贡献是 0**，一个字节都没省。
     这与 [TUNING-v7](TUNING-v7.md) 里「SparseCore 卸载在 v7 上性能收益也是 0」
     互相印证 —— **它们在 v7 上性能和显存两头都不给。**
   - **真正压显存的是调度器组，但它和 vmem 上限是绑定的**：
     只开调度器不提 vmem，会撞 **`CompileTimeScopedVmemOom`**（VMEM 爆，不是 HBM）；
     只提 vmem 不开调度器，HBM 照样 OOM。**必须成对出现。**

   另外 flag 之间还有硬依赖（漏了 `sparse_core_collective_aggregator` 会让
   latency hiding scheduler 直接报错）。**精简 flag 等于体检了另一个配置。**
2. **确认最后一行是 `Finished train_compile.py successfully!`** ——
   编译失败也会留下 HLO dump，看到 dump 不等于成功
3. **异常短的 wall time 是坏消息不是好消息** —— 多半是早期就 OOM 退出了

## 产出一：显存分解（这才是最值钱的）

编译完会打一行 `Memory analysis`，是**每个 device** 的账：

| 项 | 字节 | GiB | 是什么 |
|---|---|---|---|
| `argument_size` | 23,352,871,936 | **21.75** | 输入（权重 + 优化器状态分片） |
| `output_size` | 23,351,567,712 | 21.75 | 输出 |
| `alias_size` | 23,351,561,216 | 21.75 | 输出与输入别名的部分 |
| `temp_size` | 80,477,396,448 | **74.95** | 临时/scratch |
| `generated_code_size` | 285,933,568 | 0.27 | 代码本身 |
| `host_temp_size` | 31,809,601,536 | 29.62 | host 侧临时（offload 用） |

> [!warning] 2026-08-14 修正：判 OOM 要看 `temp_size`，不要看总和
> 我一开始写的是「单 device HBM ≈ argument + temp + generated_code = 96.97 GiB」。
> **这个公式偏保守。** 后来在 16 芯片上实测撞到真正的 OOM，编译器的原话是：
>
> ```
> RESOURCE_EXHAUSTED: Ran out of memory on HBM, the total memory required
> for HLO temporaries (100.13G) exceeds available HBM (94.74G).
> ```
>
> **它卡的是 `temp_size` 这一项，阈值约 94.74 GB/device**（v7 每芯片 192 GB ÷ 2 device，扣掉开销）。
> `argument_size` 有相当部分被别名/donate 掉了，不与 temp 简单相加 ——
> 证据是 64 芯片那次 arg+temp 算出 104 GB，而真机峰值只有 91.94 G。
>
> **所以体检时看这一条：`temp_size` 换算成 GB 后离 94.74 还有多远。**

> **最值得看的是 `temp_size` 是 `argument_size` 的 3.4 倍。**
> 也就是说这个配置下，显存大头不是权重和优化器状态，是激活与 scratch。
> 想省显存，该动的是 remat 策略和 batch，不是去切权重。

## 产出二：HLO dump

加 `XLA_FLAGS=--xla_dump_to=<dir>` 就有。本模型这套配置下：
**77 MB / 97 个文件**。可以喂给各类图分析工具。

## 产出三：编译产物可以存下来复用

`train_compile` 支持 `compiled_trainstep_file=<path>`，
用 `jax.experimental.serialize_executable` 把编译好的可执行体序列化落盘。
训练侧 `train.py` 有一行判断：**指定了这个文件就跳过编译**。

大 slice 上编译动辄几分钟，每次重启都要重付；预编译一次存起来能把这段省掉。
**注意它跟 jax / libtpu 版本绑死，升级就得重编**，别当长期缓存用。

## 选机型：三种机型 × 各核数实测

**同一份编译任务**（80 层 / `tpu7x-128` / `pdbs=12`，与生产配方逐字一致），
**同一个 docker 镜像**，只改机器和 CPU 限额。2026-08-15 实测。

| 机型 | CPU | 主频 | 说明 |
|---|---|---|---|
| **C4** `c4-highcpu-32` | Xeon 8581C | **2.30 GHz** | Emerald Rapids |
| **N4** `n4-highmem-80` | Xeon 8581C | 2.10 GHz | **同一颗芯片**，主频低 9.5% |
| N2 `n2-standard-64` | — | — | 上一代；四档并发跑过，仅供参考 |

### wall time（秒，越小越好）

| 核数 | **C4 @2.3G** | **N4 @2.1G** | C4 快多少 | N2（参考） |
|---:|---:|---:|---:|---:|
| 2 | — | 785.3 | — | — |
| 4 | **377.3** | 441.3 | −14.5% | 518 |
| 8 | **220.2** | 260.7 | −15.5% | 301 |
| 16 | **139.9** | 166.4 | −15.9% | 249 |
| 24 | — | 142.2 | — | — |
| 32 | **104.1** | 127.3 | −18.2% | 152 |
| 48 | — | 118.1 | — | — |
| 64 | — | **111.4** ← N4 最优 | — | 116 |
| 80 | — | **117.3** ⚠️ 比 64 核**更慢** | — | — |

### 五条结论

**① 同一颗芯片，C4 每一档都快 14–18%，而主频只高 9.5%。**
两台都是 Xeon 8581C，唯一差别是平台。**快出来的部分超过了主频差**，
说明 C4 的内存/IO 子系统也在贡献。佐证：C4 的 `cpu_time` 也更低
（32 核时 1258 vs 1413 核·秒，−11%）—— 是真的做得更快，不是摊得更开。

**② 超过物理核数会变慢。** N4 那台是 40 物理核 × 2 超线程 = 80 vCPU。
给到 **64 核是最快的（111.4 s）**，给满 80 核反而退到 117.3 s，
`cpu_time` 从 1434 涨到 1629 核·秒 —— **多出来的全是超线程争用的开销**。
**别按 vCPU 数给满，按物理核数给。**

**③ 并行度天花板 ~12–14 核，两台机器一致。**
C4 32 核时平均 12.08，N4 80 核时 13.88，再多给也吃不下。

**④ 峰值内存 11–14 GiB，与核数完全无关**（C4 10.9–11.2，N4 13.8–13.9）。
**highmem 机型是纯浪费。** 顺带一提 C4 比 N4 少用 20% 内存。

**⑤ 边际收益（以 C4 为准）**

| | 核数 | 时间 | 成本(核·秒) |
|---|---|---|---|
| 4→8 | ×2 | 快 1.71× | ×1.17 ✅ 最划算 |
| 8→16 | ×2 | 快 1.57× | ×1.27 ✅ |
| 16→32 | ×2 | 快 1.34× | ×1.49 ⚠️ 开始不划算 |

### 先看重复性：这些数字有多可信

同一台机器、同一配置（32 核 / 80 层 / `tpu7x-128`）连跑 5 次：

| | 1 | 2 | 3 | 4 | 5 | 均值 | 极差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **C4** | 108.1 | 107.7 | 108.2 | 108.7 | 108.3 | **108.2 s** | 1.0 s (0.9%) |
| **N4** | 128.8 | 128.3 | 128.3 | 127.8 | — | **128.3 s** | 1.0 s (0.8%) |

**两台都稳定在 1% 以内。** 所以：

- **C4 比 N4 快 15.7%（108.2 vs 128.3）是真实差异**，远在噪声之外
- 反过来，**任何小于 ~2% 的差别都不要解读**
- 前面拓扑扫描里 `tpu7x-256` 的 139.2 s（比邻近档高 ~29%）**不是噪声，是真的**，
  机制未查明，留作待办

> 单次跑的 C4 32 核是 104.1 s，比 5 连测的均值 108.2 s 更快。
> **以 5 连测均值为准**，单次数字不要引用。

### 两条「不用管」的结论 —— 对排产计划很重要

同样 32 核 C4，**只改一个变量**：

**① 模型层数完全不影响编译时间。**

| 层数 | 10 | 20 | 40 | **80** |
|---|---:|---:|---:|---:|
| wall | 105.7 s | 106.9 s | 106.1 s | 107.2 s |

**波动不到 1.5%，是一条平线。** 原因是 `scan_layers=True` ——
编译器只处理**一层的图**然后循环，80 层和 10 层对它是同一件事。

> ⚠️ 所以**别指望「先拿小模型试编译省时间」** ——省不到。
> 反过来也是好消息：**80 层的完整生产配置，编译代价跟 10 层的玩具一样便宜。**

> [!warning] 想验「关掉 scan 会怎样」？在本仓库做不到
> 我原本推测「关掉 scan 图会大 ~80 倍」。**那是推测，而且这个对照跑不起来。**
> 2026-08-15 实测 `scan_layers=False`，4 / 10 / 20 三档层数全部在 30–38 秒内失败：
>
> ```
> AttributeError: 'NNXDecoder' object has no attribute 'moe_layers'.
> Did you mean: 'moe_layers_0'?
> ```
>
> **我们这个 Hy3 移植只在 scan 模式下工作** —— 非 scan 路径下层的属性名不同，
> 要跑得先改模型代码。所以「关掉 scan 会怎样」在本仓库里**是未验证的，别当结论引用**。

**② 目标拓扑大小基本不影响编译时间。**

| 目标 | 芯片 | wall | 结果 |
|---|---:|---:|---|
| `tpu7x-32` | 16 | 53.5 s | ❌ **OOM**（temp 185.50G > 94.74G） |
| `tpu7x-128` | 64 | 107.8 s | ✅ |
| `tpu7x-256` | 128 | 139.2 s | ✅ |
| `tpu7x-512` | 256 | 113.0 s | ✅ |
| `tpu7x-1024` | 512 | 118.2 s | ✅ |

64 → 512 芯片，目标规模翻了 8 倍，**编译时间只在 107–139 秒之间浮动**。

> [!note] `tpu7x-256` 那个 +29% 是真的，而且不是单调趋势
> 怀疑是噪声，所以复跑了 3 次，并同时复跑 `tpu7x-128` 作对照：
>
> | | 1 | 2 | 3 | 均值 |
> |---|---:|---:|---:|---:|
> | `tpu7x-256`（128 芯片，4x4x8） | 139.1 | 140.2 | 139.6 | **139.6 s** |
> | `tpu7x-128`（64 芯片，4x4x4） | 107.9 | 108.5 | — | **108.2 s** |
>
> **稳定 +29%，`cpu_time` 也同步 +29%（1675 vs 1295 核·秒）** ——
> 是真的多干了活，不是调度抖动。
>
> **但它不是「越大越慢」**：再往上的 256 芯片（`tpu7x-512`，4x8x8）只要 113.0 s、
> 512 芯片（`tpu7x-1024`，8x8x8）118.2 s，都比 128 芯片那档**更快**。
> 所以贵的是 **4x4x8 这个特定 mesh**，不是规模。机制未查明，**留作待办**。

> **`tpu7x-32` 那 53.5 秒不是「更快」，是失败得早。** 16 芯片装不下 80 层，
> 编译器在优化中途就报 OOM 退出了。**看到异常短的 wall 先查 rc，别当成好消息** ——
> 这跟前面「有 dump 不等于成功」是同一类陷阱。

⇒ **规划含义：AOT 体检的成本是个常数（约 2 分钟），跟你的模型多深、
目标切片多大都无关。** 所以**每换一次配置就跑一遍**，没有理由省。

### 选型建议

| 目标 | 选择 | 依据 |
|---|---|---|
| **最快** | `c4-highcpu-32` | 104 s，比同芯片的 N4 32 核快 18% |
| **最省** | `c4-highcpu-8` | 220 s，核·秒成本最低 |
| **别选** | 任何 `highmem` | 峰值只要 11–14 GiB |
| **别选** | 给满 vCPU（超线程档） | 实测比给物理核数**更慢** |

> ⚠️ **C4 配额是按 VM family 单独算的**（`CPUS_PER_VM_FAMILY`）。
> 实测 `gpu-launchpad-playground` 在 **asia-east1 的 C4 配额是 0**，
> 连 `c4-highcpu-8` 都开不出来；换到 **us-central1-a 才成功**。
> 报错是 `Quota 'CPUS_PER_VM_FAMILY' exceeded. Limit: 0.0` ——
> **开不出来先看这个，不是总 CPU 配额的问题**（那边显示 3000 可用）。

## 端到端实战：AOT 存产物 → 16 卡真训练加载它

> 2026-08-14 完整跑通。**20 层缩比版**（80 层砍到 1/4）、16 芯片、`per_device_batch_size=8`。
> 全过程没丢一个 DWS 节点。

### 为什么要缩到 20 层

16 芯片装不下 80 层。而且**层数减少不等于显存变松** ——
FSDP 宽度从 128 掉到 32，每个 device 要 all-gather 的专家权重反而厚了 4 倍。
所以 64 芯片能跑的 `pdbs=12`，到 16 芯片上直接 OOM。

### AOT 先把 batch 扫出来（这一步是全篇最值钱的）

四次 AOT，每次 ~85 秒，**全在 CPU 上，一张卡都没占**：

| `pdbs` | `temp_size` | vs 94.74 GB 上限 | 结果 |
|---:|---:|---|---|
| 4 | 60.0 GB | 宽松 | ✅ |
| 6 | 70.9 GB | 宽松 | ✅ |
| **8** | **79.6 GB** | 还有 15 GB | ✅ **选它** |
| 12 | 100.1 GB | **超 5.4 GB** | ❌ `RESOURCE_EXHAUSTED` |

**换算成真机代价**：要在 TPU 上试出这条线，得占着 16 芯片跑四轮、每轮几分钟起步，
还得排队等卡。AOT 用 6 分钟 CPU 时间给了同样的答案。

### 存产物：注意它不认 GCS 路径

`save_compiled()` 和 `load_serialized_compiled()` 都是**裸 `open()`**，
**不支持 `gs://`**。所以要两段：

```bash
# AOT 侧：先落本地，再自己传
... compiled_trainstep_file=/tmp/hy3_20L_p8.pkl
gcloud storage cp /tmp/hy3_20L_p8.pkl gs://your-bucket/compiled/

# 训练侧：先拉下来，再指向本地路径
gcloud storage cp gs://your-bucket/compiled/hy3_20L_p8.pkl /tmp/compiled.pkl
... compiled_trainstep_file=/tmp/compiled.pkl
```

产物大小：**约 360 MB**（20 层 / 16 芯片）。

### 加载是真的生效的 —— 但别在 `train.py` 里找

`train.py` 里只有一行 `if config.compiled_trainstep_file == "":`，看着像没实现。
**真正的加载在 `src/maxtext/utils/train_utils.py:135-142`**：

```python
if config.compiled_trainstep_file != "":
    max_logging.log("Loading the compiled function...")
    p_train_step = maxtext_utils.load_compiled(config, functional_train, state, execution_devices)
    max_logging.log("Loaded compiled function!")
```

**验收就看这两行日志。** 没有它们 = 没加载上。

### 实测收益

同一份配置跑两遍，唯一差别是带不带 `compiled_trainstep_file`：

| | 启动到 step 0 | 稳态 step | TFLOP/s/device | 最终 loss |
|---|---:|---:|---:|---:|
| **带缓存** | **23.0 s** | 5.789 s | 216.05 | 12.450 |
| 不带缓存 | 67.7 s | 5.781 s | 216.34 | 12.450 |

- **启动省 44.7 秒，快 2.9×**（加载缓存本身只花 2.1 s）
- **稳态性能没有差别** —— 缓存只省启动，不改运行时
- **loss 轨迹逐位相同**（13.425 → 12.450），证明产物与现场编译等价

20 层就省 45 秒；80 层 / 64 芯片的编译要长得多，省得也更多。
**大 slice 上每次重启都要重付这笔钱，这才是它真正的价值。**

### 训练结果（16 芯片 / 20 层 / pdbs=8）

```
completed step: 9, seconds: 5.789, TFLOP/s/device: 216.055,
Tokens/s/device: 5660.303, loss: 12.450
```

换算成每芯片（v7 是 2 device/chip）：**432 TFLOP/s/chip，MFU 18.7%**（BF16 峰值 2307）。

> 这个数**不能跟生产的 630 横比**：层数、FSDP 宽度、batch 全都不同。
> 它的意义是「这条链路端到端通了」，不是性能基线。

### DWS 节点：怎么换 pod 而不把卡弄丢

这批 16 芯片的 DWS 排队单早已 `BookingExpired`，节点池 autoscaling 下限是 0 ——
**撑着节点的只有那个 `sleep infinity` 占位 JobSet**。直接删掉它去跑训练，
节点会被判定为闲置而缩容，几天排来的容量就没了。

**安全顺序（实测两次，零丢失）：**

1. **先创建**训练 JobSet —— 芯片被占着，4 个 pod 会停在 `Pending`
2. 确认 4 个都 `Pending`（autoscaler 此刻看得到「有未满足的需求」）
3. **再删**占位 JobSet
4. 调度器立刻把 pending 的 pod 放上去，实测 60 秒内全部 `Running`

全程不存在「待调度需求为零」的窗口，所以不会触发缩容。
**反过来先删再建就危险了** —— 中间那段空窗期正是缩容判据成立的时候。

> **训练跑完也要立刻把占位补回去。** JobSet `Completed` 之后节点同样变成闲置候选。
> 我在两次实验之间各补了一次。

### 这一段踩的雷

**5. 用 `sed` 删配置行删不干净会更糟。**
做对照组时我用 `sed` 想去掉 `compiled_trainstep_file`，只删掉了下载那行、参数留着了，
结果任务去加载一个不存在的文件、4 个 pod 全崩，还白占了一轮节点。
**改成按行过滤 + `assert` 断言残留为 0**，一次就对：

```python
out = [l for l in text.split('\n') if 'compiled_trainstep_file' not in l]
assert 'compiled_trainstep_file' not in '\n'.join(out)
```

**6. 前几步的 step time 不能信。** step 0 是 2.96 s、step 1 是 0.63 s、step 2 是 11.4 s，
到 step 3 才稳定在 5.79 s。**读数从第 3 步之后开始。**

## 什么时候该跑它

| 场景 | 值不值 |
|---|---|
| 要换 batch / remat / 并行度，怕 OOM | **值**。5 分钟 vs 抢 64 张卡跑一轮 |
| 换了 XLA flag，想知道编不编得过 | **值**。flag 依赖问题在这一步就暴露 |
| 想知道显存花在哪 | **值**，而且是唯一能拿到 argument/temp 分解的地方 |
| 想知道到底跑多快 | **不值**。AOT 不给性能数，只给显存和可编译性 |
