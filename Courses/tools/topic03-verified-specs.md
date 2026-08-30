# 专题二 §3「计算单元」已核实规格

规则同 §1：**查到的**标来源，**推出来的**摆推导链，**查不到的**明写「官方未披露」。

---

## A. GPU 侧 — NVIDIA 官方

### A1. SM 内部构成（来源：NVIDIA 官方开发者博客 *Inside NVIDIA Blackwell Ultra*）

> "Every SM … is a self-contained compute engine housing:
> **128 CUDA Cores** for FP32 and INT32 operations, also FP16/BF16 and other precisions;
> **4 fifth-generation Tensor Cores** with second-generation Transformer Engine;
> **256 KB of Tensor Memory (TMEM)**; **Special Function Units (SFUs)**."

- 整卡：**160 SM 跨两个 die**，编成 **8 个 GPC**，共 **640 个五代 Tensor Core**
- ⚠️ 这篇讲的是 **Blackwell Ultra（B300）**，不是 B200。原文自己写了
  "Available SM count and HBM capacity varies by SKU"
- **SM 分四个 partition / sub-core** —— 来源是第三方微基准论文
  （arXiv 2507.10789，*Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks*）：
  "each SM includes **four partitions** of execution units or sub-cores"。
  NVIDIA 官方博客的 SM 框图也是四分区，但正文没写这个词 → **标为「二级来源」**

### A2. Tensor Core 四代演进（来源：同一篇 NVIDIA 官方博客，原文措辞）

| 代 | NVIDIA 原话 | 协作范围 |
|---|---|---|
| Volta | "8-thread MMA units" | 8 线程 |
| Ampere | "Full warp-wide MMA, BF16, and TensorFloat-32" | 1 warp = 32 线程 |
| Hopper | "Warp-group MMA across 128 threads" | 1 warpgroup = 128 线程 |
| Blackwell | "**dual-thread-block MMA**, where **paired SMs cooperate on a single MMA operation**, sharing operands" | 2 个 CTA / 2 个 SM |

⭐ 这张表是 §3.4 的骨架 —— **粒度变大的方向是 NVIDIA 自己说的**，不是我们的解读。

### A3. 指令级矩阵形状（来源：NVIDIA PTX ISA 9.3 官方手册）

| 指令 | 形状 | 协作范围 | 操作数在哪 | Target ISA |
|---|---|---|---|---|
| `wmma.mma.sync` | **m16n16k16** | warp | 寄存器 | sm_70+ |
| `mma.sync.aligned` | **m16n8k16**（f16/bf16）<br>m16n8k32（fp8）<br>m16n8k64（fp4） | warp（32 线程） | 寄存器 | sm_80+ |
| `wgmma.mma_async` | m64n{8..256}k16 | warpgroup（128 线程） | 描述符 / shared | **仅 sm_90a** |
| `tcgen05.mma` | M = **64 或 128**<br>（`cta_group::2` 时 128 或 256）<br>N = {8,16,…,256} 步长 8<br>**K = 16**（`kind::f16`） | CTA / CTA 配对 | **TMEM** | sm_100+ |

**必须讲清的三点**（都是官方手册直接可查）：

1. **Tensor Core 从来不是「128×128」。** 就算走 tcgen05，**K 只有 16**。
   M 那条边才是 64/128/256。说成 128×128 是错的。
2. **`wgmma` 官方写明 "Requires sm_90a"** → **Hopper 独有**，Blackwell 不用它。
   arXiv 2507.10789 也印证："Hopper wgmma instructions are not compatible with Blackwell,
   instead the new tcgen05 instructions can be used"。所以四代不是接力，Hopper 那条是支线。
3. **小粒度的路在 Blackwell 上没被废掉，而且拿到了新能力。**
   块量化那批新指令挂的是 **`mma.sync.aligned.m16n8k32/k64` + `.block_scale`**，
   不是挂在 tcgen05 上 → 说明 m16n8 是主力之一，不是遗产。

### A4. 块量化（block scaling）的粒度（来源：PTX ISA 9.3）

- 修饰符：`.scale_vec::1X / 2X / 4X` 与 `.block16 / .block32`
- `kind::mxf8f6f4` + `scale_vec::1X` on **m16n8k32** → 1 个 scale 管 32 个元素
- `kind::mxf4nvf4` + `scale_vec::4X` on **m16n8k64** → 4 个 scale 管 K=64 → **1 个管 16 个**
- 即：**MXFP4 = 每 32 个元素一个 scale；NVFP4 = 每 16 个**
- scale 的类型：`.ue8m0`（MX 系）/ `.ue4m3`（NVFP4 可选）

### A5. Tensor Memory（来源：PTX ISA 9.3，`9.7.17.1 Tensor Memory`）

> "the 5th generation TensorCore has **dedicated on-chip memory** … organized as a
> two-dimensional matrix where the horizontal rows are called **lanes** and the vertical
> columns are called **columns**. On architecture `sm_100a`/`sm_100f` … **512 columns and
> 128 rows per CTA, with each cell being 32-bits**."

→ 128 × 512 × 4 B = **256 KB / SM**，与官方博客的 "256 KB of TMEM" 对上。**两个来源互证。**

- 动态分配：`tcgen05.alloc / dealloc`，按**列**分配
- 官方形状表里出现 "**4x1 (1/2 datapath utilized)**" —— GPU 这边同样有「喂不满数据通路」这回事

### A6. 之前已核实（来源：docs.nvidia.com/cuda/blackwell-tuning-guide）

- 每 SM 最大并发 warp **64**（Hopper 48）；每 SM 最多 **32** 个 block
- 寄存器堆 **64K × 32-bit / SM = 256 KB**；每线程最多 255 个寄存器
- shared memory **228 KB / SM**；L1 + texture + shared 上限 **256 KB / SM**
- L2 **126 MB**；thread block cluster 可移植上限 8，B200 可 opt-in 到 16

---

## B. TPU 侧 — 公开 JAX 源码 `jax/_src/pallas/mosaic/tpu_info.py`

### B1. v7 / v7x TensorCore（`case ChipVersion.TPU_7 | TPU_7X`，**均为 per-TensorCore**）

```
num_lanes = 128        num_sublanes = 8
mxu_column_size = 256  num_mxus = 2       num_accumulators = 128
vmem = 64 MiB          smem = 1 MiB       cmem = 0
hbm = 206e9 // 2       bw = 7.40e12 // 2
bf16 = 2.31e15 // 2    fp8 = 4.60e15 // 2
```

`num_accumulators` 的官方 docstring：
> "The number of **(num_sublanes, mxu_column_size)-shaped 32-bit accumulator buffers**
> available for each MXU."

⭐ **推导（链完整）**：每 MXU 的累加器容量 = 128 × (8 × 256) × 4 B = **1 MiB**；
每 core 两个 MXU → **2 MiB**；每 chip 两个 core → **4 MiB**。
→ 拿来跟 Blackwell 的 **256 KB TMEM / SM** 并排讲：**两边都把矩阵乘的累加器从通用存储里拿出来了**，
只是 TPU 从第一代就这样，GPU 到第五代才这样。

### B2. v6e 对照（同一文件）

MXU column 256、2 MXU/core、VMEM **128 MiB**/core、SMEM 1 MiB、
SparseCore 2 core × 16 subcore × **8 lane**、每 subcore VMEM **256 KiB**

### B3. SparseCore（v7）

2 core / device，16 subcore，**16 lane**，每 vector subcore VMEM **512 KiB**，DMA granule **32 B**

### B4. §1 已立的 VPU 推导（沿用，别重推）

MXU = 2 × (256×256) = 131,072 个乘加单元；VPU = 8 × 128 = 1,024 个 lane。
即便每 lane 每周期一次乘加，比值也已是 **1 : 128**。
**每个 lane 里有几个算术单元，公开资料查不到** → 只能说「低两个数量级」，不给精确百分比。

---

## C. ⚠️ 官方未披露 / 尚未解决 —— 写之前必须处理

1. **B200（非 Ultra）的 SM 数**。CUDA 官方文档与架构简报都没写。
   第三方拆解一致给「物理 160 / 激活 148」。→ 课件里必须标来源等级。
2. **走 `mma.sync.m16n8k16` 那条小路，能不能吃到官方标的 2.5 PFLOPS 峰值**。
   官方没写。合理猜测是不能，但**不许当结论写**。
3. **TPU v7 的时钟频率**。官方未披露 → VPU 的绝对峰值算不出来。
4. ✅ **（已解决，见 D 节）** MXU 的周期级模型 —— 已用「在 v2–v5p 上验证过」的方式
   把矛盾定位到 v6e/v7 的公开数字本身，而不是我们的模型。**结论是「官方数字互相矛盾」，
   不是「我们算错了」。** §3.1 可以动笔了。
5. **TPU v6e / v7 每核到底几个 MXU、PE 每周期几次乘加** —— 官方口径与峰值算力对不上，
   真实答案未披露。课件里只讲矛盾，不给答案。

---

## D. ⭐ 一个能被验证的模型 —— 以及它揭出的「每 cell 双发」

模型：`峰值 FLOPS = 时钟 × 全芯片乘加单元总数 × 2`
其中 `乘加单元总数 = TensorCore数 × 每核MXU数 × MXU边长²`（三项全部来自官方页面）。

| 芯片 | core/chip | MXU/core | 边长 | **MAC/chip** | 公布峰值 BF16 | 每 cell 1 次<br>反推时钟 | 每 cell 2 次<br>反推时钟 |
|---|---|---|---|---|---|---|---|
| v2 | 2 | 1 | 128 | 32,768 | 46 T | **0.702 GHz** ✅ | 0.351 |
| v4 | 2 | 4 | 128 | 131,072 | 275 T | **1.049 GHz** ✅ | 0.525 |
| v5e | 1 | 4 | 128 | 65,536 | 197 T | **1.503 GHz** | 0.752 |
| v5p | 2 | 4 | 128 | 131,072 | 459 T | **1.751 GHz** ✅ | 0.875 |
| v6e | 1 | 2 | **256** | 131,072 | 918 T | 3.502 ❌ | **1.751 GHz** |
| v7 | 2 | 2 | **256** | 262,144 | 2,307 T | 4.400 ❌ | **2.200 GHz** |

**三个公开时钟锚点全中**：v2 → 0.702（公布 700 MHz）、v4 → 1.049（公布 1050 MHz）、
v5p → 1.751（公布 1.75 GHz）。**模型对 gen ≤ 5 成立。**

### ⭐ 关键那一步：v5p 和 v6e 的乘加单元数**一模一样**

- v5p：2 core × 4 MXU × 128² = **131,072**
- v6e：1 core × 2 MXU × 256² = **131,072**

**同样多的乘加单元，v6e 的公布峰值整整是 v5p 的两倍**（918 对 459）。
四个输入（两代的 core 数、MXU 数、边长、峰值）**全部来自 Google Cloud 官方页面**，
不需要任何非公开信息。

所以只有两种可能：**时钟翻倍到 3.5 GHz**（同代工艺同功耗封装，不可能），
或者 **gen ≥ 6 的每个乘加 cell 每周期做两次 bf16 乘加**。

带回去验：
- v6e：131,072 × 2（双发）× 2（FLOP/MAC）× 1.75 GHz = **917.5 T**（公布 918）
- v7 ：262,144 × 2 × 2 × 2.2 GHz = **2,306.9 T**（公布 2,307）

两代都精确闭合。

> ✅ **课件里可以讲的**：v5p 与 v6e 乘加单元数相同而峰值翻倍 → 变化必然在**单个 cell 里**，
> 不在阵列数量上。这条推导 100% 走公开数字。
> 🔴 **课件里不能写的**：v7 的具体时钟。那是非公开来源，只能说「官方未披露」。
> 上表最后一列的 2.200 / 1.751 是**反推出来的自洽解**，不是引用。

### 📌 方法论教训（值得写进课件的旁注）

第一版模型漏了「每 cell 双发」，直接得出 v7 需要 4.4 GHz，
**而正确答案恰好是它的一半**。定位靠的不是查到新资料，
而是**拿同一个模型去跑那些三项数字都公开且自洽的老世代**——
v2 / v4 / v5p 三个锚点全中，就证明模型骨架没错，
错的一定是套用到新世代时多出来的那个未知因子。

### 官方口径（原文，全部来自 docs.cloud.google.com）

- v4 页：「Each TPU v4 chip contains two TensorCores. Each TensorCore has **four**
  matrix-multiply units (MXUs), a vector unit, and a scalar unit.」
- v6e 页：「Each v6e chip contains **one** TensorCore. Each TensorCore has **2**
  matrix-multiply units (MXU), a vector unit, and a scalar unit.」
- 架构页：「An MXU is composed of either **256 x 256** (TPU v6e and TPU7x) or
  **128 x 128** (prior to v6e) multiply-accumulators in a systolic array.」
- 架构页同一段：「Each MXU is capable of performing **16K multiply-accumulate
  operations per cycle**.」
  → 🔴 **16K = 16,384 = 128²。跟同一页上面那句 256×256 = 65,536 直接打架。**
  这是本课的**第二个「官方文档自相矛盾」教学例**（第一个是 §0 的 GiB / GB）。

### ⭐ 架构页里那段可以逐字念的脉动阵列描述（官方原文）

> "To perform the matrix operations, the TPU loads the parameters from HBM into the
> Matrix Multiplication Unit (MXU). Then, the TPU loads data from HBM. **As each
> multiplication is executed, the result is passed to the next multiply-accumulator.**
> The output is the summation of all multiplication results between the data and
> parameters. **No memory access is required during the matrix multiplication process.**"

→ **这就是 §3.1 的主干**：脉动阵列省的不是算力，是「中间结果的搬运」。
拿它跟 Blackwell 的 TMEM 并排讲 —— GPU 到第五代才把累加器搬出通用寄存器堆。

