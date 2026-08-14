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

## 选机型：实测扫描

同一份编译任务，只改 CPU 限额（内存都给足），跑五档：

| 核数 | wall | 相对 4 核 | 核·秒（成本） | 平均并行度 | 峰值内存 |
|---:|---:|---:|---:|---:|---:|
| 4 | 518 s | 1.00× | 2,072 | 3.50 | 13.0 GiB |
| 8 | 301 s | 1.72× | 2,408 | 5.92 | 13.0 GiB |
| 16 | 249 s | 2.08× | 3,986 | 8.29 | 12.9 GiB |
| 32 | 152 s | 3.42× | 4,848 | 11.67 | 13.0 GiB |
| 62 | 116 s | **4.48×** | 7,167 | **14.33** | 12.9 GiB |

### 三条结论

**① 内存完全不是瓶颈 —— 稳定 13 GiB，跟核数无关。**
所以 **highmem 机型是浪费钱**。highmem 给你 8 GB/vCPU，这活儿实际需要不到 1 GB/vCPU。

**② 并行度封顶在 ~14 核。** 给 62 个核，它平均也只吃 14.33 个。
总 CPU 时间从头到尾在 1,700–2,100 核·秒之间浮动 —— 活是固定的，
多给核只是把它摊开，摊到一定程度就摊不动了。

**③ 核数翻 15.5 倍，只快 4.5 倍。** 边际收益：

| | 核数 | 时间 | 成本 |
|---|---|---|---|
| 4→8 | ×2.0 | 快 1.72× | ×1.16 ✅ 最划算 |
| 8→16 | ×2.0 | 快 1.21× | ×1.66 ❌ 最亏 |
| 16→32 | ×2.0 | 快 1.64× | ×1.22 ✅ |
| 32→62 | ×1.9 | 快 1.31× | ×1.48 ⚠️ |

### 建议

- **要便宜**：8 核，5 分钟出结果，成本几乎最低
- **要快**：32 核，2.5 分钟，成本还算线性
- **别买 highmem**，32 GB 内存绰绰有余
- **单核主频有用** —— 平均并行度只有 14，剩下相当一部分是串行的，
  主频高的机型（如 C4 系列）在那一段直接受益。所以方向是
  **c4-standard-16 / c4-highcpu-32 这类，不是 highmem**

> ⚠️ 这组数字的前提是 **`scan_layers=True`**。scan 让编译器只处理一层的图再循环，
> HLO 才这么小。**关掉 scan 图会大 ~80 倍，内存结论必须重测。**
>
> 另：4/8/16/32 四档是并发跑在两台 64 核机器上的，可能有轻微互相干扰
> （16 核那档的 cpu_time 偏高，疑似就是这个原因）；62 核那档是独占跑的。
> 趋势可信，个别点不必较真。

## 踩过的雷

**1. XLA flag 之间有依赖，不要自行精简。**
我把 16 个 flag 砍到 5 个想跑快点，直接报：

```
INVALID_ARGUMENT: Latency hiding layer scheduler requires
sparse core collective aggregator to be enabled.
```

延迟隐藏调度器要求 sparse core collective aggregator 同时开着。
**AOT 要体检的是生产配置，flag 就得跟生产一字不差** —— 精简了等于体检了另一个东西。

**2. 镜像里没有 `/usr/bin/time`。** 想量资源得自己包一层，
`aot.sh` 里用 python 的 `resource.getrusage(RUSAGE_CHILDREN)` 做了，
顺带给出 `avg_cores`（= cpu_time / wall），这个数比 wall 本身更能指导选机型。

**3. 别用 heredoc 往容器里塞 python。** YAML 里嵌 shell 再嵌 heredoc，
缩进会被吃掉，报 `IndentationError`。用 base64 编码传，一次就对。

**4. 编译失败也会 dump。** 第一次失败的运行也留下了 86 个 HLO 文件（4.6 MB）。
**看到有 dump 不等于编译成功**，一定要确认 `Finished train_compile.py successfully!`。

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
