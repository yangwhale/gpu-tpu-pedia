# WebAOT 参数目录 —— 每个控件长什么样、问号里写什么

前端按这份表生成表单。**控件类型**决定渲染方式，**问号**里的三段
（干什么 / 改了会怎样 / 建议）直接展示给用户。

> 数值全部来自 Hunyuan3-295B-A21B 在 64 芯片 TPU v7 上的实测（2026-08-15/16），
> 换模型/换硬件时**结论方向可参考，绝对值必须重测**。

图例：`▼` 下拉 · `#` 数字输入 · `☑` 开关 · `▤` 参数组（一键预设）

---

## A. 目标硬件

### `compile_topology` ▼

| 选项 | 芯片数 | 说明 |
|---|---|---|
| `tpu7x-8` | 4 | 冒烟，1–2 分钟 |
| `tpu7x-32` | 16 | 小规模 |
| **`tpu7x-128`** | **64** | 标准 benchmark |
| `tpu7x-256` | 128 | 只在工作日 peak 窗口 |
| `tpu7x-512` | 256 | 同上 |

> **干什么**：告诉编译器目标硬件有多大，它据此决定所有分片。AOT 必填。
> **改了会怎样**：改这个等于换了一台机器，**所有显存和分片结论都变**。
> **⚠️ 数字是 device 不是芯片** —— v7 是 2 device/chip，`tpu7x-128` = **64 芯片**。
> 这是最常写错的地方。
> **建议**：先用 `tpu7x-8` 把配置跑通，再换目标规模。

### `compile_topology_num_slices` #
> **干什么**：多少个 slice（跨 pod）。
> **建议**：单 pod 填 `1`。多 slice 的通信模式完全不同，结论不能互相套用。

---

## B. 模型

### `model_name` ▼ · `base_num_decoder_layers` #

> **干什么**：`base_num_decoder_layers` 覆盖模型层数。
> **改了会怎样**：**层数直接决定显存和编译时间**。4 层约 1 分钟编完，80 层约 3 分钟。
> **建议**：**调试配置用 4–8 层**（错误照样暴露，快 3 倍）；
> **问显存必须用生产层数**，否则预测无意义。

### `max_target_length` ▼ `2048 / 4096 / 8192`
> **干什么**：序列长度。
> **改了会怎样**：激活显存约与它成正比；跨配置比吞吐时**必须对齐**，否则 tok/s 不可比。
> **建议**：`4096`。

---

## C. 并行策略（最影响结果的一组）

### `ici_fsdp_parallelism` ▼ `-1 / 32 / 64 / 128 / 256`

> **干什么**：把参数、梯度、优化器状态切成多少份分散到各卡。**省显存的主力**。
> **改了会怎样**：每卡常驻权重 ∝ 1/FSDP。64 芯片（128 device）上实测：
> FSDP=64 每卡 18.1 GB，FSDP=128 每卡 **9.1 GB** —— 省出的 9 GB 能把 batch 从 7 推到 13。
> **`-1` = 自动吃满所有 device**，通常就是最优。
> **⚠️ 陷阱**：开 QAG（见 D 组）时要求 `专家数 % FSDP == 0`，
> 192 个专家只能取 64，**另一半并行度被迫给 DP，而 DP 不分片权重** ——
> 这正是放弃 QAG 反而更快的原因（677 → 727）。
> **建议**：`-1`。

### `ici_data_parallelism` #
> **干什么**：纯数据并行的宽度。
> **改了会怎样**：**DP 不分片权重**，每卡各存一份完整分片。
> 把并行度花在 DP 上 = 浪费显存换不到任何东西（除非被整除约束逼的）。
> **建议**：`1`，让 FSDP 吃满。

### `ici_tensor_parallelism` #
> **建议**：`1`。本模型未测出正收益。

### `ici_expert_parallelism` #
> **干什么**：专家并行，把专家分到不同卡、用 all-to-all 搬 token。
> **改了会怎样**：**实测大幅负收益** —— 64 芯片 EP=2 掉 **39.6%**，16 芯片 EP=4 掉 **71%**。
> 除了 all-to-all 在 3D torus 上多跳，更致命的是它**逼着 FSDP 减半**，
> 每卡静态分片翻倍，batch 被压小。
> **建议**：`1`，不要碰。

### `per_device_batch_size` ▼ `1 … 16`

> **干什么**：每个 **device**（不是芯片）一次处理几条序列。
> **改了会怎样**：几乎线性吃显存，也几乎线性提吞吐 —— **顶到 OOM 前一档就是最优**。
> 64 芯片实测上限：
>
> | 配置 | 上限 | per-chip |
> |---|---|---|
> | BF16 | 13 | 666.6 |
> | FP8 + `fixed` 校准 | **13** | **727.0** |
> | FP8 + `absmax` 校准 | **11** | 670.8 |
> | FP8 + QAG（FSDP 被锁 64） | 7 | 677.0 |
>
> **⚠️ 显存不一定随 batch 单调** —— 实测 `absmax` 在 13 超 0.77 G、在 **12 反而超 1.26 G**
> （不同尺寸让编译器选了不同排布）。**逐档问 AOT，别外推。**
> **建议**：用本工具二分找上限，取上限。

---

## D. 精度与量化

### `dtype` ▼ `bfloat16 / float32`
> **干什么**：计算/激活精度。**建议** `bfloat16`。

### `weight_dtype` ▼ `float32 / bfloat16`
> **干什么**：**主权重**（优化器里那份真身）的存储精度。
> **改了会怎样**：降到 `bfloat16` 每卡省约 4.5 GB（FSDP=128），能换更大 batch。
> **但每步梯度相对权重量级极小，bf16 只有 8 位尾数，更新会被直接舍掉** ——
> 等于梯度消失。Google 给 TPU 的官方建议、NVIDIA NeMo 的推荐模式都是 **fp32 主权重**。
> **建议**：**`float32`，不要改**。想省显存去调 FSDP 和 batch。

### `grad_dtype` ▼ · `mu_dtype` ▼
> **干什么**：梯度、AdamW 一阶动量的存储精度。
> **改了会怎样**：都设 `bfloat16` 可把优化器状态从 16 B/param 降到 12 B/param。
> 比官方建议激进，但实测可用。`nu_dtype`（二阶动量）optax 不支持单独设，恒随 `weight_dtype`。
> **建议**：`bfloat16`。

### `use_qwix_quantization` ☑ · `quantization` ▼ `'' / fp8_full`
> **干什么**：开 FP8 量化。只作用于**分组矩阵乘**（规则按算子名注册），
> **注意力全程 bf16**，不受影响。
> **改了会怎样**：精度链是 `fp32 主权重 → bf16 → fp8 → 矩阵乘 → fp32 累加`。
> 实测 FP8 最优 727.0 vs BF16 最优 666.6，**只快 9%** —— 代价是 batch 变小、多一套调参。
> **建议**：追求极限吞吐才开；对稳定性敏感就用 BF16。

### `weight_quantization_calibration_method` ▼ `absmax / fixed,-224,224`
### `act_quantization_calibration_method` ▼ 同上

> **干什么**：量化 scale 怎么定。`absmax` 按通道动态求最大值，`fixed` 钉死在预设范围。
> **改了会怎样**：
> - `fixed` **会损害收敛质量** —— 真实分布随训练漂移、逐层逐专家量级不同，
>   静态范围要么裁掉大值、要么把小值挤进低位。**跑得通、loss 也降，质量被悄悄吃掉。**
> - 但 `fixed` 是编译期常量、**零显存**；`absmax` 要动态归约缓冲 + scale 张量，
>   实测**吃掉两档 batch**（13 → 11），吞吐 727.0 → 670.8（−7.7%）。
>   **这 7.7% 主要不是算得慢，是 batch 少了。**
> - **`fixed` 唯一的正当理由是 QAG**：先量化再跨卡收集，各卡 scale 必须一致，
>   而动态 scale 每卡只看得到自己那份分片。不开 QAG 就没有任何理由用它。
> **建议**：**`absmax`**。只有在明确要 QAG 且已确认收益时才用 `fixed`，
> 并且报数字时必须在同一句里注明。

### `shard_exp_on_fsdp` ☑ ⚠️
> **干什么**：把 MLP 权重的**专家维**切到 FSDP 轴上（每卡拿若干个**完整**专家）。
> 它是 QAG 的前置开关之一。
> **改了会怎样**：
> - 🔴 **和 native kernel 组合会静默算错**：开它 + `fixed` 校准 + **不开** `use_tokamax_gmm`
>   → 权重按专家维切成 N 份，而 native 分支**没有 all-gather** ——
>   kernel 只对本地那几个专家建表，**其余全部没算**。不报错、loss 照常降，
>   唯一症状是「快得离谱」（曾误报 1,014.8，补齐后 637.0）。
> - 关掉它则按 **embed 维**切（正经 FSDP），每卡拿到的是**残缺**专家，
>   编译器**被迫**收齐才能算 —— **结构上不可能漏**。
> **建议**：**不要开**。这不是性能选项，是一个会静默出错的雷。
> 要用必须同时开 `use_tokamax_gmm`。

---

## E. MoE kernel

### `megablox` ☑ · `sparse_matmul` ☑
> **建议**：都开。这是分组矩阵乘的主路径。

### `use_tokamax_gmm` ☑ ⚠️
> **干什么**：切换 MoE 的 kernel 实现。**它在两个精度下走的是完全不同的代码**：
>
> | 精度 | 开了之后实际走到哪 | 实测 |
> |---|---|---|
> | BF16 | `tokamax.ragged_dot`（裸路径） | **慢 12 倍** |
> | FP8 | `mblx.gmm(use_tokamax_backend=True)`，带 QAG | 677.0 |
>
> **改了会怎样**：BF16 下开它纯亏；FP8 下**如果同时开了 `shard_exp_on_fsdp`，
> 关掉它就会触发上面那个静默漏算**。
> **建议**：BF16 不开；FP8 若开了 `shard_exp_on_fsdp` 则必须开。
> 最优配方是**两个都不开**（727.0 / 670.8）。

### 18 个 tile 参数 ▤ `{wi,wo}_tile_{fwd,dlhs,drhs}_{batch_seq,embed_dim,mlp_dim}`

> **干什么**：分组矩阵乘的分块大小，三组（前向 / 对输入的梯度 / 对权重的梯度）× 两个权重。
> **改了会怎样**：**一个都不传会静默回退到默认 tile** —— BF16 慢 **26%**，
> FP8 直接崩（`AssertionError: v=1536 bv=1024`，默认值除不尽）。
> **⚠️ 分块不能大于该维实际大小**：`mlp_dim` 只有 1536，填 3072 直接断言失败。
> `embed_dim` 是 4096，但填满 4096 会撞 Mosaic 的 `Vectorizing shape must be 1D`，最多 2048。
> **建议**：一键用预设 **`(512, 2048, 1536)`**。实测三个维度都已到顶，没有上升空间。

---

## F. 注意力

### `attention` ▼ `flash / dot_product`
### `use_tokamax_splash` ☑ · `sa_use_fused_bwd_kernel` ☑
### 8 个 `sa_block_*` ▤

> **干什么**：splash attention 的分块。
> **改了会怎样**：实测 **2048 是唯一甜点**，块 4096 慢 11.5%、compute 4096 爆 VMEM，
> 块 1024 也差。而且**跟序列长度无关** —— seq=16384 时最优块仍是 2048。
> 疑似撞 VMEM/MXU 的硬件常数，不是比例关系。
> **建议**：一键预设全 `2048`。

---

## G. 显存与重算

### `remat_policy` ▼ `full / custom / minimal`
### `decoder_layer_input` ▼ `offload / remat / device`
### `out_proj` ▼ `remat / device`
### `scan_layers` ☑

> **干什么**：用重算/换出换显存。`offload` 把层输入换到 host 内存。
> **改了会怎样**：`scan_layers=True` 让 80 层在 HLO 里只展开成一层循环体，
> **编译时间大幅下降**（这是 80 层能在 3 分钟内编完的原因）。
> **⚠️** `decoder_layer_input=offload` 与某些自定义探针/位操作冲突，
> 会报 `Bitcast cannot have different memory spaces`。
> **建议**：`custom` + `offload` + `remat` + `scan_layers=True`。

---

## H. XLA flags

### `--xla_tpu_dvfs_p_state` ▼ `0 … 7`
> **干什么**：芯片频率档位。
> **改了会怎样**：**默认是 3**。每档约 +2.4%，单调无拐点，**7 已顶格**（9 会报错）。
> 从默认到 7 实测 BF16 +8.6%、FP8 +8.0%，**显存一字节没涨**。
> **建议**：**`7`。零代价，不开白不开。**

### `--xla_tpu_scoped_vmem_limit_kib` #
> **改了会怎样**：不设或设太小，大 tile 会报 `CompileTimeScopedVmemOom`。
> **建议**：`65472`。

### SparseCore 卸载族 ▤（8 个 flag，一键全开）
> **干什么**：把集合通信卸载到 SparseCore，与 TensorCore 计算并行。
> **改了会怎样**：这是**通信能不能被藏住**的前提。实测同一条权重 all-gather，
> 编译器插的版本每步只暴露 **34.6 ms**，手写在 kernel 入口的版本暴露 **6,170 ms**（178 倍）。
> **⚠️ 有依赖关系，不能挑着开** —— 少了 `..._collective_aggregator`，
> 编译器直接拒绝：`Latency hiding layer scheduler requires sparse core collective aggregator`。
> **建议**：**一键全开，不要手动精简。**

### 延迟隐藏调度器族 ▤
`--xla_tpu_enable_latency_hiding_layer_scheduler=true`
`--xla_tpu_scheduler_percent_shared_memory_limit=150`
`--xla_tpu_enable_layer_scheduler_for_dependent_collectives=true`
`--xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=false`
> **建议**：整组照抄，依赖 SparseCore 那一族。

---

## I. 优化器

### `opt_type` ▼ `adamw / adam_pax / sgd / muon`
> **建议**：`adamw`。

---

## 附：前端交互建议

1. **两种模式并存** —— 「粘命令」和「填表单」双向同步：粘进来自动填表，改表单自动更新命令。
2. **预设按钮** —— `BF16 最优` / `FP8 生产` / `FP8 峰值`，一键填满已验证配方。
3. **实时 lint** —— 改任何控件立刻跑一遍静态规则，🔴 直接挡住提交按钮。
4. **问号内容分三段固定格式** —— 干什么 / 改了会怎样 / 建议，别写成散文。
5. **每个数字都标出处** —— 「实测 64 芯片 v7」比「据说更快」有用一百倍。
