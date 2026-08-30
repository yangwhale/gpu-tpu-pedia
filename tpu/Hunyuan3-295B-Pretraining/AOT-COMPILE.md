# AOT 提前编译：不占一张 TPU，先把账算清楚

> 2026-08-15 全流程复跑验证。模型 Hunyuan3-295B-A21B（80 层，第 0 层 dense，192 专家 top-8），
> 目标 `tpu7x-128`（= 64 芯片 4x4x4），配方与 [TUNING-v7](TUNING-v7.md) 的生产配置一致。
> 配套脚本 [`maxtext-hunyuan3/aot.sh`](maxtext-hunyuan3/aot.sh)。

## 一句话

MaxText 自带的 AOT（ahead-of-time compilation）能**在一台普通 CPU 机器上**，
为你手上根本没有的拓扑编译出完整的训练步 —— **编译器算出的显存与真机逐位相同**（实测）。
**抢卡之前先跑它，比抢到卡再发现装不下便宜太多。**

> [!warning] 它不是用来省编译时间的
> 64 芯片上真机编译只要 **46 秒**，占启动总时长的 1/5。
> AOT 省的是「排队 → 抢到 64 张卡 → 发现 OOM → 重排队」这个**小时级**的循环。
> 详见 [编译时间的真相](#编译时间的真相省下的不是十几分钟是-46-秒)。

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

---

# 生产级流程

**全程零 TPU。** 2026-08-15 在 cc-tw（`n4-highmem-80`，docker 限 32 核）
按本文步骤从头跑通一遍，实测：

| 步骤 | 耗时 | 说明 |
|---|---:|---|
| Step 0 `prep.sh` | **11 s** | clone + 8 项自检 + 打包上传 |
| Step 2 一次 batch 探针 | **136 s** | 其中编译本体 128.8 s |
| Step 3 存产物并上传 | **148 s** | 产物 396 MB |
| **合计** | **≈ 5 分钟** | |

> 这台 cc-tw 就是下方选机型表里的 **N4** 那一列。本次 128.8 s 与表里
> N4 32 核的 127.3 s 差 **1.2%**，在 2% 噪声底之内 —— **交叉验证通过**。
> 同配置在 GKE `cpu-np` 节点上是 134.4 s（+4.3%），在 C4 上是 108.2 s（−16%）。

## Step −1 · 版本对齐（**最容易被忽略、代价最大的一步**）

编译产物里**刻着编译器的源码版本号**。我从 395 MB 的产物里 grep 出来的原文：

```
googlefile:/google_src/files/948136882/depot/g3      ← libtpu 的 changelist
TPU7x                                                 ← 目标设备型号
```

所以它绑死三样东西，**任意一样对不上，缓存就是废的**：

| 绑什么 | 怎么看 |
|---|---|
| **编译器版本**（libtpu / jaxlib） | 产物里那个 changelist |
| **目标设备型号** | 产物里的 `TPU7x` |
| **输入输出的 tree 结构** | ⚠️ 这个**没存在产物里** —— `load_compiled` 用当前代码 `jax.eval_shape` 现推（`maxtext_utils.py:239`）。**改了模型形状，加载端和产物就对不上。** |

### 唯一可靠的做法：AOT 与训练用同一个镜像

版本不是靠"记下来再装一遍"对齐的，是靠**用同一个镜像**对齐的。
本文全部实测的那一套：

| | |
|---|---|
| 镜像 | `chrisya-maxtext-latest:runner` |
| `jax` / `jaxlib` | **0.11.1.dev20260726** |
| `libtpu`（pip 包） | **0.0.44** |
| libtpu 构建标签 | `libtpu_lts_20260715_b_RC00`，changelist **948136882** |
| `libtpu.so` | 685,151,296 字节 |

查版本：

```bash
docker run --rm $IMAGE python3 -c "import jax,jaxlib;print(jax.__version__, jaxlib.__version__)"
docker run --rm $IMAGE pip show libtpu | grep -i version
# 真机日志里也会打：Build label / from changelist / Jax Version
```

> **不用镜像、裸装的话**，至少要把 `jax`、`jaxlib`、`libtpu` 三个版本钉到与训练侧
> 完全一致（`pip install jax==<v> jaxlib==<v> libtpu==<v>`）。
> **这条路本文没有实测过** —— 全部数据都来自同镜像路径。
>
> ⚠️ 顺带一提：**跨版本加载会怎样，我也没有直接测过。**
> 「版本绑死」这个结论来自产物里刻着的 changelist，不是来自一次失败的加载实验。

## Step 0 · 准备（一次性）

```bash
GCS_STAGE=gs://your-bucket/hy3 bash maxtext-hunyuan3/prep.sh
```

clone 分支 → 跑 8 项自检 → 打包 `src/maxtext` 传 GCS。
**只有改代码才要重跑**，换 flag / 换参数不用。

## Step 1 · 确认目标拓扑名（写错不报错，数字全废）

拓扑名按 **device 数**，v7 是 2 device/chip：

| 芯片 | 拓扑名 | mesh |
|---:|---|---|
| 16 | `tpu7x-32` | 2x2x4 |
| 32 | `tpu7x-64` | 2x4x4 |
| **64** | **`tpu7x-128`** | 4x4x4 |
| 128 | `tpu7x-256` | 4x4x8 |
| 256 | `tpu7x-512` | 4x8x8 |
| 512 | `tpu7x-1024` | 8x8x8 |

写成 `tpu7x-64` 编出来的是 32 芯片的图，**数字全错还不报错**。
完整对照在 `src/maxtext/utils/accelerator_to_spec_map.py`。
**同时核对 GKE 节点标签 `cloud.google.com/gke-tpu-topology` 要跟 mesh 一致。**

## Step 2 · 扫出最大可行 batch

从你想要的 `pdbs` 开始，OOM 就减半，能过就往上加：

```bash
# A. 本机 docker —— 任意一台带 docker 的机器
RUNNER=docker PDBS=12 GCS_STAGE=... IMAGE=... bash maxtext-hunyuan3/aot.sh probe-p12

# B. GKE Job —— 机器不在手边时
PDBS=12 GCS_STAGE=... IMAGE=... bash maxtext-hunyuan3/aot.sh probe-p12
```

> **两种跑法的产物逐字节相同** —— 2026-08-15 各跑一遍，
> `temp_size` 同为 80,477,396,448，`.pkl` 的 md5 也一模一样
> （`aa2d5b3d…`，414,546,674 字节）。**在哪编都行，产物可以互换。**
>
> **B 有个前提文档以前没写**：GKE 那个 CPU 节点池的服务账号
> 必须有 `GCS_STAGE` 那个桶的读权限。**跨项目的桶会直接 403**，
> 而且报错出现在 pod 里、不是提交时，看着像脚本挂了。

判据看这一行 —— **只看 `temp_size`，别看总和**：

```
Memory analysis: CompiledMemoryStats(... temp_size_in_bytes=80477396448 ...)
```

`temp_size ÷ 2**30` 与 **94.74 GiB/device** 比。留 10 GiB 余量比较稳。

> [!important] 单位：本页一律用 **GiB**
> 编译器报错串里的 `G` 是 **GiB**（`95.38G − 94.74G = 656.93M` 只有按 1024 才对得上），
> 而 `CompiledMemoryStats` 给的是**字节**。两者必须换算到同一进制再比 ——
> 早期版本把字节按 `÷ 1e9` 记成十进制 GB，和报错串的 GiB 混在同一张表里，**差 7.4%**。
> pass／fail 的结论没错（十进制那侧偏保守），但派生出来的余量和百分比全都偏小。
> **2026-08-22 已统一换算成 GiB。**
失败时编译器直接告诉你差多少：

```
RESOURCE_EXHAUSTED: HLO temporaries (100.13G) exceeds available HBM (94.74G)
```

> [!warning] 判 OOM 不要把 argument 加进去
> 早期版本写的是「单 device HBM ≈ argument + temp + generated_code = 96.97 GiB」。
> **这个公式偏保守，会把可行配置误判成 OOM。**
> 编译器卡的就是 `temp_size` 这一项，阈值 **94.74 GiB/device**
> （v7 每芯片 192 GiB ÷ 2 device = 95.93，再扣掉约 1.19 GiB runtime 自留）。
> `argument_size` 有相当部分被别名/donate 掉了，不与 temp 简单相加 ——
> 证据是 64 芯片那次 arg+temp 算出 96.70 GiB，而真机峰值只有 91.94 G。

## Step 3 · 存下编译产物（可选）

```bash
SAVE_TO=gs://your-bucket/compiled RUNNER=docker ... bash maxtext-hunyuan3/aot.sh prod
```

脚本会先落本地再上传 —— 因为 `save_compiled()` / `load_serialized_compiled()`
都是**裸 `open()`，不认 `gs://`**。训练侧同样要先拉到本地再指过去：

```bash
gcloud storage cp gs://your-bucket/compiled/hy3-aot-prod.pkl /tmp/compiled.pkl
... compiled_trainstep_file=/tmp/compiled.pkl
```

产物 **414,546,674 字节（约 396 MB）**（80 层 / 64 芯片 / 带 tile）。
**它跟 jax / libtpu 版本绑死，升级就得重编**，别当长期缓存用。
收益见[下方实测](#带缓存-vs-不带缓存)：**省 51 秒，不是省十几分钟。**

## 两个脚本对账（每次改配置都跑一下）

**AOT 体检的必须是你真会跑的那个配置** —— 两边任何一处漂移，体检结果就失去意义。
这不是提醒，是可执行的检查：

```bash
DRYRUN=1 bash maxtext-hunyuan3/aot.sh dry > /tmp/a.txt
DRYRUN=1 PLATFORM=v7 bash maxtext-hunyuan3/run.sh dry > /tmp/r.txt
diff /tmp/a.txt /tmp/r.txt    # 应当无输出
```

两个脚本会把展开后的模型/并行参数排序打出来（当前各 48 条），不提交任何东西。

> **这个检查是被自己坑出来的。** 2026-08-15 给 `aot.sh` 加上 18 个 tile 参数时，
> 忘了 `run.sh` 的 v7 分支还没有 —— 于是 AOT 体检的配置和实际会跑的配置分了岔。
> 光靠人眼看两份 48 条参数的清单不可能发现。

## Step 4 · 这时候才去抢卡

前三步都过了，再提交真实训练任务。

> **为什么值得**：一次 AOT 约 2 分钟 CPU 时间，成本可忽略；
> 而「排队等 64 张卡 → 起来跑 → OOM 退出 → 重排队」一次就是小时级，
> 还占着别人的容量。**省的是排队，不是编译。**

---

# 读数

## 显存分解 —— 这才是最值钱的产出

编译完打一行 `Memory analysis`，是**每个 device** 的账：

| 项 | 字节 | GiB | 是什么 |
|---|---|---|---|
| `argument_size` | 23,352,871,936 | **21.75** | 输入（权重 + 优化器状态分片） |
| `output_size` | 23,351,567,712 | 21.75 | 输出 |
| `alias_size` | 23,351,561,216 | 21.75 | 输出与输入别名的部分 |
| **`temp_size`** | **80,477,396,448** | **74.95** | **临时/scratch —— 判 OOM 就看它** |
| `generated_code_size` | 292,657,664 | 0.27 | 代码本身 |
| `host_temp_size` | 31,809,601,536 | 29.62 | host 侧临时（offload 用） |

> **`temp_size` 是 `argument_size` 的 3.4 倍。**
> 这个配置下显存大头不是权重和优化器状态，是激活与 scratch。
> **想省显存该动 remat 策略和 batch，不是去切权重。**

## 与真机逐位吻合

同一份配置，AOT 在 CPU 上算的 vs 真机 64 芯片跑出来的：

| | AOT 预测 | 真机打印 |
|---|---:|---:|
| `temp_size`（pdbs=12） | 74.95 GiB | **75.0** |
| `temp_size`（pdbs=13） | 73.11 GiB | **73.1** |
| `argument_size` | 21.75 GiB | **21.7** |
| host temp | 29.62 GiB | **29.6** |

**一个字节都不差。** 它给的不是估算，就是编译器在真机上会得到的同一份账。

## 它给不了什么

| | |
|---|---|
| **性能** | AOT 只回答「装不装得下 / 编不编得过」，一个吞吐数字都给不了 |
| **运行期打的补丁** | 它体检的是**配置文件描述的那个程序**。pod 里现场 `exec` 进去的 monkeypatch，AOT 看不见（[实例](#顺藤摸出来的一条tile-该用配置参数不是那个-monkeypatch)） |
| **带缓存那一路的显存账** | 加载预编译产物时不会打印 `Memory analysis`（压根没编译）。**想要显存分解只能靠 AOT** —— 这反而是又一条该跑它的理由 |
| **`use_tokamax_gmm=True` 那条路** | **整个 AOT 直接跑不起来**，见下 |

### AOT 体检不了 tokamax 那条 kernel 路径

`use_tokamax_gmm=True` 时 AOT 直接抛：

```
NotImplementedError: Not supported on cpu.
```

根因在 `tokamax/_src/ops/op.py:197`：

```python
for device in infer_devices(ba) or {backend.get_default_device()}:
    if not self.supported_on(device):
        raise NotImplementedError(f"Not supported on {device.device_kind}.")
```

**它检查的是「本地默认设备」，不是「编译目标」。** AOT 时 `JAX_PLATFORMS=cpu`，
本地是 CPU，于是 `supported_on(CPU)` 为假、直接退出 —— 尽管 mesh 明明指向 `tpu7x`。

对照之下，MaxText 自己那套 megablox **专门处理了这个场景**，`kernels/megablox/ops.py`
里的注释写得很直白：

> `jax.devices()[0]` **不是编译 TARGET**：train_compile 时本地后端是 CPU
> （`JAX_PLATFORMS=cpu`），而 mesh 目标是 tpu7x

⇒ **这不是架构限制，是 dispatch 层少考虑了一种场景。** 但结论对使用者是硬的：
**走 tokamax 就没有 AOT 体检**。好在生产推荐的就是 native megablox（[TUNING-v7 §3.4.5](TUNING-v7.md)），
不受影响。

> **顺带厘清 Pallas 与 Mosaic 的关系**（排障时容易混）：
> **Pallas 是 JAX 写 kernel 的 DSL**，本身不产机器码，要靠后端降下去 ——
> TPU 上是 **Mosaic TPU**，NVIDIA 上是 Mosaic GPU 或 Triton。
> 所以 `pallas_mosaic_tpu` = 「用 Pallas 写、经 Mosaic 降到 TPU」。
> tokamax 是多后端库，同一个 `ragged_dot` 有 mosaic-tpu / mosaic-gpu(sm90/sm100) / triton
> 好几套实现，`implementation="mosaic"` 是在点名要 TPU 那一套。
> **不是「Pallas 不够用所以上 Mosaic」，两者是上下层关系。**

## 顺带产出：HLO dump

加 `XLA_FLAGS=--xla_dump_to=<dir>`（脚本默认就带）。本模型这套配置：
**77 MB / 97 个文件**，可以喂给各类图分析工具。

---

# 编译时间的真相：省下的不是十几分钟，是 46 秒

64 芯片 / 80 层，不带缓存那一路的日志把时间花在哪写得清清楚楚：

| 阶段 | 耗时 | AOT 能不能省 |
|---|---:|---|
| TPU init / 建 ICI 切片 | **69.6 s** | ❌ 省不掉 |
| **`jit_train_step` 编译** | **45.8 s** | ✅ **省掉的就这一段** |
| ├ `HLO_PASSES`（JAX 图 → HLO + 图级优化） | 15.2 s | |
| ├ `BACKEND_PASSES`（HLO → **LLO** → 机器码） | **23.0 s** | |
| └ `CODE_GENERATION` | 6.4 s | |
| 其余小 jit + 数据管线 + step 0 本身 | ~110 s | ❌ 省不掉 |

> **最贵的不是图优化，是往下降。** `BACKEND_PASSES` 一段就占 45.8 s 的 50%，
> 比整个 HLO 阶段还贵 —— HLO → LLO 的低级降级、VMEM/寄存器分配、排流水都在这里
> （日志里 `llo_loop.cc` 的输出正落在这一段）。这跟「图优化最烧时间」的直觉相反。

[TUNING-v7](TUNING-v7.md) 早就推翻过「v7 编译要 10-17 分钟」这个说法
（那次量到 43.5 s vs 44.3 s，并指出**真正随规模涨的是建切片不是编译**）。
这次在完全不同的路径上量到 45.8 s 编译 + 69.6 s 建切片，**独立复现了同一个结论**。

---

# 端到端实测（64 芯片 / 80 层 / bodaborg）

> 2026-08-15。真的生产配置：80 层、64 芯片、`pdbs=12`。
> 带缓存与不带缓存各跑 10 步，除了 `compiled_trainstep_file` 之外**逐字相同**。

## 带缓存 vs 不带缓存

| | 带缓存 | 不带缓存 |
|---|---:|---:|
| 加载 / 编译 | 加载 **2.15 s** | 编译 **45.83 s** |
| 启动到 step 0 | **175.6 s** | **227.0 s** |
| 稳态 step | 25.98 s | 25.99 s |
| TFLOP/s/device | 262.69 | 262.63 |
| loss（step 4→9） | 12.958 → 12.701 | 12.958 → 12.701 |

- **启动省 51.4 s（占启动的 23%）**，加载近 400 MB 的产物本身只花 2.15 s
- **稳态性能零差异，loss 真正逐位相同**（10 步最大差 0.000）—— 产物与现场编译完全等价
- 20 层 / 16 芯片的缩比版跑过同一套对照，省 44.7 s ——
  **跟 64 芯片省的几乎一样多**，再次说明编译代价不随规模涨

## 加载生效的验收点 —— 别在 `train.py` 里找

`train.py` 里只有一行 `if config.compiled_trainstep_file == "":`，看着像没实现。
**真正的加载在 `src/maxtext/utils/train_utils.py:135-142`**：

```python
if config.compiled_trainstep_file != "":
    max_logging.log("Loading the compiled function...")
    p_train_step = maxtext_utils.load_compiled(config, functional_train, state, execution_devices)
    max_logging.log("Loaded compiled function!")
```

**验收就看这两行日志。没有它们 = 没加载上。**

---

# 拿 AOT 当搜索器

既然显存预测跟真机逐位一致，它就可以拿来**在 CPU 上免费扫配置空间**。
29 个探针（每个约 2 分钟，一台机器 3 并发），只问一个问题：**装不装得下**。
原始数据：[`data/aot-64chip-config-probe-20260815.csv`](maxtext-hunyuan3/data/aot-64chip-config-probe-20260815.csv)。

## batch 的天花板是 13，不是 12

| `pdbs` | global batch | `temp` | 结论 |
|---:|---:|---:|---|
| 12（原生产） | 1536 | 74.95 GiB | ✅ |
| **13** | **1664** | **73.11 GiB** | ✅ **比 12 还省 1.84 GiB** |
| 14 | 1792 | 96.53 GiB | ❌ 差 1.79 GiB |
| 15 / 16 | — | 102.59 GiB (16) | ❌ 差得更远 |

> **`pdbs=13` 比 `pdbs=12` 占用更少 —— 反直觉，但复现了两次、
> 字节级一致（78,503,516,064）。** 机制没查明，多半是编译器在这个形状上
> 选了另一套调度/切分。**记在这里是因为它可执行，不是因为它被解释清楚了。**

p14 只差 1.79 GiB，试了三种省法都补不上：`out_proj=offload`（省 0.99 GiB）、
再叠 `mlpwi=offload`、把 splash block 从 2048 砍到 1024 —— **全部仍然 OOM**。

**天花板对带 tile 的生产配置同样成立**：把 18 个 tile 参数加进 AOT 重扫一遍，
`temp` 一字未变（p12 仍 74.95、p13 仍 73.11、p14 仍 OOM）。
**tile 对显存的影响是 0** —— 真机峰值 HBM 也印证了这点（带不带都是 91.94 G）。

## remat 一松就崩，没有中间地带

| 配置 | 需要的 `temp` | vs 94.74 GiB |
|---|---:|---|
| 生产 `custom` + `decoder_layer_input=offload` | 74.95 GiB | ✅ |
| `out_proj=device` | 118.05 GiB | ❌ 超 25% |
| `decoder_layer_input=device` | 115.08 GiB | ❌ |
| `remat_policy=full` | 115.08 GiB | ❌ **与上一行一模一样** |
| `mlpwo=device` | 323.45 GiB | ❌ 超 3.4 倍 |
| `remat_policy=minimal` | 562.58 GiB | ❌ 超 5.9 倍 |

- **`remat_policy=full` 与 `decoder_layer_input=device` 的账完全相同** ——
  这个模型下 `full` 的实际效果就是把 decoder layer input 放回 device。
- **`qkv_proj=device` 和 `mlpwi=device` 对显存的影响是 0**（仍 74.95 GiB，
  与基线一字不差）—— 这两个旋钮在本模型的 custom remat 路径上**根本没被读到**。
  **改了没反应 ≠ 改对了，这正是 AOT 能替你发现的那类事。**

## XLA flag 是承重墙，不只是调性能

| flag 子集 | `temp` | 结果 |
|---|---:|---|
| 带生产 flag（16 个） | **74.95 GiB** | ✅ 还剩 19.8 GiB |
| 完全不带 | 95.01 GiB | ❌ 超上限 —— **只差 0.3%** |
| 只 `scoped_vmem_limit` | 94.81 GiB | ❌ 几乎没省 |
| 只 **SparseCore 卸载那 9 个** | **95.01 GiB** | ❌ **与不带 flag 一模一样** |
| **`scoped_vmem_limit` + 调度器组** | **74.95 GiB** | ✅ |

- **这组 flag 把峰值临时空间压低 21%**（95.01 → 74.95 GiB），正好是「能跑」和「跑不了」的分界
- **SparseCore 卸载那 9 个对显存的贡献是 0** —— 与 [TUNING-v7](TUNING-v7.md)
  里「SparseCore 卸载在 v7 上性能收益也是 0」互相印证，**两头都不给**
- **调度器组和 vmem 上限必须成对出现**：只开调度器不提 vmem 撞
  `CompileTimeScopedVmemOom`（VMEM 爆，不是 HBM）；只提 vmem 不开调度器 HBM 照样 OOM
- flag 之间还有硬依赖（漏 `sparse_core_collective_aggregator` 会让
  latency hiding scheduler 直接报错）。**精简 flag 等于体检了另一个配置。**

> 这五档 2026-08-15 在另一台机器上**独立重跑过一遍，数字逐位相同**
> （95.01 / 94.81 / 95.01 / VMEM 爆 / 74.95）。
> 起因是原始日志随消融用的 C4 机器一起删了，而消融 CSV 里没有 `temp` 这一列 ——
> **结论一度只剩文档里这张表、没有可查的证据。** 与其加免责声明，不如重跑一遍补回来。

---

# 顺藤摸出来的一条：tile 该用配置参数，不是那个 monkeypatch

扫配置时想问「`pdbs=13` 跟生产那套跑到 630 的 tokamax tile 叠起来还装不装得下」，
于是把 QUICKSTART 早期写的 `tkcfg.py` 补丁挂进 AOT —— **结果发现它根本没被调用过。**

给补丁加个计数器，AOT 和真机上都是同一句话：

```
[tkcfg] patched          ← 它照常打印这句，看起来生效了
[tkcfg] 被调用 0 次       ← 实际上一次都没走到
```

**原因不是补丁写错了，是它打在了另一条 kernel 路径上。**
`tkcfg.py` 改的是 `PallasMosaicTpuRaggedDot._get_heuristics_config`，
而 `moe.py:1500` 只有 **`use_tokamax_gmm=True`** 才走那条路 ——
**这个开关在 v7 上会死锁，生产一直关着**（TUNING-v7 §6.7）。
所以补丁本身没坏，只是对生产实际走的那条路无效。

生产走的是 `elif self.config.megablox` 那条分支，实际调用
`mblx.gmm(..., use_tokamax_backend=False)`。它的 tile 由
**18 个配置参数**决定（在 `moe.py:1852+` 组装成 `tiling` 传进去）：

```
{wi,wo}_tile_{fwd,dlhs,drhs}_{batch_seq,embed_dim,mlp_dim}
```

正是 `run.sh` 的 **v5p 分支一直在传、v7 分支从来没传**的那一组。

准确地说，v7 上有**两条**都能拿到 tile 收益的路，而 `run.sh` 一条都没走：
没开 `use_tokamax_gmm`（怕死锁），也没传这 18 个配置参数 —— 结果跑在默认 tile 上。
**`aot.sh` 与 `run.sh` 现已都默认带上**，完整的调 tile 方法见
[TUNING-v7 §3.4.6](TUNING-v7.md)。

## 换成配置参数之后：662 → 666

64 芯片 / 80 层，`(batch_seq, embed_dim, mlp_dim) = (512, 2048, 1536)`：

| 配置 | step | TFLOP/s/device | **per-chip** | **MFU** | 峰值 HBM |
|---|---:|---:|---:|---:|---:|
| `pdbs=12`，无 tile | 25.99 s | 262.69 | 525.4 | 22.77% | 91.93 G |
| `pdbs=13`，无 tile | 27.88 s | 265.18 | 530.4 | 22.99% | 92.57 G |
| **`pdbs=12` + 18 参数** | 20.61 s | 331.10 | **662.2** | **28.70%** | 91.94 G |
| **`pdbs=13` + 18 参数** | 22.19 s | 333.29 | **666.6** | **28.89%** | 92.57 G |
| 参考：旧最优（monkeypatch 路径） | 21.67 s | — | 630 | 27.31% | 91.94 G |

- **改用配置参数比旧最优高 5.1%**（662.2 vs 630）。多出来的部分应该来自
  **`dlhs` / `drhs` 两条反向路径也被 tile 了** —— monkeypatch 只影响前向启发式
- **`pdbs` 12 → 13 再加 0.66%**，合计 **666.6 / MFU 28.89%，比旧最优高 5.8%**
- **`pdbs=12` 带不带 tile 的 loss 差 ≤0.001**（step 0 是 13.422 vs 13.423，其余步一致）——
  重新结合律量级，**不是逐位相同**
- 峰值 HBM 与旧记录的 91.94 G 一字不差 —— 说明打的是同一个配置

> **这条是被 AOT 带出来的，但不是 AOT 测出来的。**
> AOT 只负责说「`pdbs=13` 装得下、14 装不下」，把真机额度省到该花的地方；
> 是「为什么补丁不生效」这个追问带出了 tile 的真正入口。
> **性能数字全部来自真机。**

---

# 选机型

**同一份编译任务**（80 层 / `tpu7x-128` / `pdbs=12`），**同一个 docker 镜像**，
只改机器和 CPU 限额。

## 先看重复性：这些数字有多可信

同机同配置（32 核）连跑 5 次：

| | 1 | 2 | 3 | 4 | 5 | 均值 | 极差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **C4** `c4-highcpu-32` | 108.1 | 107.7 | 108.2 | 108.7 | 108.3 | **108.2 s** | 1.0 s (0.9%) |
| **N4** `n4-highmem-80` | 128.8 | 128.3 | 128.3 | 127.8 | — | **128.3 s** | 1.0 s (0.8%) |

**两台都稳定在 1% 以内。所以：小于 ~2% 的差别一律不要解读。**

## wall time（秒，越小越好）

两台都是 **Xeon 8581C**，C4 主频 2.30 GHz、N4 2.10 GHz（低 9.5%）：

| 核数 | **C4** | **N4** | C4 快多少 | N2（上一代，参考） |
|---:|---:|---:|---:|---:|
| 2 | — | 785.3 | — | — |
| 4 | **377.3** | 441.3 | −14.5% | 518 |
| 8 | **220.2** | 260.7 | −15.5% | 301 |
| 16 | **139.9** | 166.4 | −15.9% | 249 |
| 32 | **104.1** | 127.3 | −18.2% | 152 |
| 64 | — | **111.4** ← N4 最优 | — | 116 |
| 80 | — | **117.3** ⚠️ 比 64 核**更慢** | — | — |

**① 同一颗芯片，C4 每档快 14–18%，而主频只高 9.5%。**
快出来的部分超过主频差，说明 C4 的内存/IO 子系统也在贡献。
佐证：C4 的 `cpu_time` 也更低（32 核时 1258 vs 1413 核·秒，−11%）——
**是真的做得更快，不是摊得更开。**

**② 超过物理核数会变慢。** N4 是 40 物理核 × 2 超线程 = 80 vCPU。
给 64 核最快（111.4 s），给满 80 核退到 117.3 s，`cpu_time` 从 1434 涨到 1629 ——
**多出来的全是超线程争用。别按 vCPU 给满，按物理核数给。**

**③ 并行度天花板 ~12–14 核**，两台一致，再多给也吃不下。

**④ 峰值内存 11–14 GiB，与核数完全无关。highmem 机型是纯浪费。**

**⑤ 边际收益（C4）**：4→8 快 1.71×（成本 ×1.17，最划算）；
8→16 快 1.57×（×1.27）；16→32 只快 1.34×（×1.49，开始不划算）。

## 层数和拓扑基本都不影响编译时间

**层数完全不影响**（32 核 C4）：

| 层数 | 10 | 20 | 40 | **80** |
|---|---:|---:|---:|---:|
| wall | 105.7 s | 106.9 s | 106.1 s | 107.2 s |

波动不到 1.5%，一条平线。原因是 `scan_layers=True` ——
编译器只处理**一层的图**然后循环。

> ⚠️ **别指望「先拿小模型试编译省时间」，省不到。**
> 反过来是好消息：**80 层完整生产配置的编译代价跟 10 层玩具一样便宜。**

**目标拓扑也基本不影响**：

| 目标 | 芯片 | wall | n | 结果 |
|---|---:|---:|---:|---|
| `tpu7x-32` | 16 | 53.5 s | 1 | ❌ **OOM**（temp 185.50G） |
| `tpu7x-128` | 64 | **108.2 s** | 5 | ✅ |
| `tpu7x-256` | 128 | **139.6 s** ⚠️ | 3 | ✅ |
| `tpu7x-512` | 256 | 113.0 s | 1 | ✅ |
| `tpu7x-1024` | 512 | 118.2 s | 1 | ✅ |

> **`n` 列别忽略**：加粗那两行是多次均值，其余是单次。
> 单次值本身有约 1% 噪声（见上方重复性），而 512 / 1024 那两档只跑过一次 ——
> **它们「比 4x4x8 快」这个结论是稳的（差 20% 远超噪声），但具体数字别精确引用。**

64 → 512 芯片规模翻 8 倍，编译时间只在 107–140 秒之间浮动。

> [!note] `tpu7x-256` 那个 +29% 是真的，而且不是「越大越慢」
> 复跑 3 次都是 139.1 / 140.2 / 139.6，`cpu_time` 也同步 +29%
> （1675 vs 1295 核·秒）—— 是真的多干了活，不是调度抖动。
> 但更大的 4x8x8（113.0 s）和 8x8x8（118.2 s）反而**更快**。
> **贵的是 4x4x8 这个特定 mesh，不是规模。机制未查明，留作待办。**

⇒ **规划含义：AOT 体检的成本是个常数（约 2 分钟），跟模型多深、
目标切片多大都无关。所以每换一次配置就跑一遍，没有理由省。**

## 选型建议

| 目标 | 选择 | 依据 |
|---|---|---|
| **最快** | `c4-highcpu-32` | 104 s，比同芯片的 N4 32 核快 18% |
| **最省** | `c4-highcpu-8` | 220 s，核·秒成本最低 |
| **别选** | 任何 `highmem` | 峰值只要 11–14 GiB |
| **别选** | 给满 vCPU（超线程档） | 实测比给物理核数**更慢** |

> ⚠️ **C4 配额按 VM family 单独算**（`CPUS_PER_VM_FAMILY`）。
> 实测 `gpu-launchpad-playground` 在 **asia-east1 的 C4 配额是 0**，
> 连 `c4-highcpu-8` 都开不出来，换 **us-central1-a 才成功**。
> 报错是 `Quota 'CPUS_PER_VM_FAMILY' exceeded. Limit: 0.0` ——
> **开不出来先看这个**，不是总 CPU 配额的问题（那边显示 3000 可用）。

---

# 踩过的雷

1. **异常短的 wall time 是坏消息不是好消息** —— 多半是早期就 OOM 退出了。
   `tpu7x-32` 那 53.5 秒不是「更快」，是 16 芯片装不下 80 层、编译中途就退了。
2. **有 HLO dump 不等于成功** —— 编译失败也会留下 dump。
   **必须确认最后一行是 `Finished train_compile.py successfully!`**
3. **`save_compiled()` 不认 `gs://`** —— 裸 `open()`，只能先落本地再自己传。
4. **加载缓存看不到 `Memory analysis`** —— 想要显存分解只能靠 AOT 那一步。
5. **用 `sed` 删配置行删不干净会更糟。** 做对照组时想去掉 `compiled_trainstep_file`，
   只删掉下载那行、参数留着了，任务去加载一个不存在的文件、pod 全崩，还白占一轮节点。
   **改成按行过滤 + `assert` 断言残留为 0：**
   ```python
   out = [l for l in text.split('\n') if 'compiled_trainstep_file' not in l]
   assert 'compiled_trainstep_file' not in '\n'.join(out)
   ```
6. **前几步的 step time 不能信。** 实测 step 0 / 1 / 2 分别是 2.96 / 0.63 / 11.4 秒，
   到 step 3 才稳定。**读数从第 3 步之后开始。**
7. **`exclusive-topology` 的反亲和会把下一个 JobSet 挡在门外。**
   跑完一轮立刻提第二轮，16 个 pod 只出来 1 个、一直 `Pending`，
   `describe` 说 `Insufficient google.com/tpu` + `didn't match pod anti-affinity rules` ——
   上一轮**已经 `Completed` 的 pod 还挂在节点上**。
   **跑完要显式 `kubectl delete jobset`，不能只等它自己 `Completed`。**
   另外这种情况下只会创建 leader 一个 pod，**看 pod 数会误判成「只申请了 1 台」**。
8. **写了不存在的 `priorityClassName` 不报错，静默不 admit。**
   bodaborg 上 `very-high` / `high` 已经没了，现存最高可用是 `medium`(500)。
   JobSet 看着在正常排队，但 Kueue 根本不给建 Workload。
9. **GKE 跑法要给节点池 SA 配 GCS 读权限** —— 跨项目的桶直接 403，
   而且报错出现在 pod 里不是提交时。

> **DWS 节点池上换 pod 别把卡弄丢**（与 AOT 无关，但同一批实验踩到）：
> 排队单 `BookingExpired` 后撑着节点的只有占位 JobSet。
> **安全顺序是「先建训练 JobSet（会 Pending）→ 确认 autoscaler 看得到需求 → 再删占位」**，
> 反过来先删再建，中间那段「待调度需求为零」的空窗正好触发缩容。实测两次零丢失。

---

# 什么时候该跑它

| 场景 | 值不值 |
|---|---|
| 要换 batch / remat / 并行度，怕 OOM | **值**。2 分钟 vs 抢 64 张卡跑一轮 |
| 换了 XLA flag，想知道编不编得过 | **值**。flag 依赖问题在这一步就暴露 |
| 想知道显存花在哪 | **值**，而且是唯一能拿到 argument/temp 分解的地方 |
| 想批量筛配置 | **值**。CPU 上并发扫，把真机额度留给活下来的少数几个 |
| 想知道到底跑多快 | **不值**。AOT 一个性能数字都给不了 |
| 想省大 slice 的编译时间 | **不值**。64 芯片上编译只占启动的 46 s / 227 s |

---

# 附录

## 原始数据

| 文件 | 内容 |
|---|---|
| [`data/aot-ablation-20260815.csv`](maxtext-hunyuan3/data/aot-ablation-20260815.csv) | CPU / 层数 / 拓扑消融，**49 次**编译（37 成 12 败） |
| [`data/aot-64chip-config-probe-20260815.csv`](maxtext-hunyuan3/data/aot-64chip-config-probe-20260815.csv) | 64 芯片配置探针，**29 次**（含 3 次带 tile 复扫、5 次 flag 子集复现） |

**消融 49 次编译（37 成 12 败），12 次失败每一条都有解释，没有「不明原因」：**

| 失败项 | 次数 | 原因 |
|---|---:|---|
| `tpu7x-32` + 80 层 | 1 | 设计内 OOM（temp 185.50G），16 芯片装不下 80 层 |
| `scan_layers=False` | 3 | 本仓库 Hy3 移植不支持非 scan 路径（`moe_layers` 属性名不同）；其中 1 次是脚本 heredoc 转义损坏 |
| 去掉全部 XLA flag | 2 | 设计内 OOM（95.01G）—— 这正是要测的结论 |
| 单独 flag 子集 | 3 | 设计内：vmem 单独→HBM OOM；调度器单独→VMEM OOM；SC 单独→HBM OOM |
| 传空 `compile_xla_flags` | 2 | **脚本 bug**：转义把它变成字面量 `\"\"`，触发 pydantic 校验错 |
| 首次 C4 运行 | 1 | **脚本 bug**：打包时漏了 `logs/` 目录 |

> 「脚本 bug」那几条是**工具问题不是被测对象的问题**，一并列出来免得后人当成 MaxText 或 XLA 的缺陷。

## 待办

- **`tpu7x-256`（4x4x8）为什么贵 29%** —— 复现三次确认非噪声，且非单调。机制未查明。
- **`scan_layers=False` 的对照** —— 需要先修模型代码的属性名才能跑。
- **调度器组内部再细分** —— 目前只定位到「调度器组 + vmem 上限」这个组合，还没拆到单个 flag。
