# 专题二外传 · 已核实规格（全部公开可引）

> ⛔ **这一页是这一讲所有数字的唯一出处。** 图上、正文里任何一个数，
> 都必须能在这儿找到，并且带着它的来源。找不到的不许写。
>
> ⭐ 立这一页的理由跟 `topic02-verified-specs.md` 一样：这一讲的**全部立意
> 就架在几个比值上**（算力比、带宽比、屋脊点比）。比值错一位，整讲作废。

---

## 一、TPU v6e（Trillium）

**来源：`docs.cloud.google.com/tpu/docs/v6e`（官方）**

| 项 | 值 |
|---|---|
| 每 chip TensorCore 数 | **1**（⭐ 不是 v7 的 2 —— **chip ＝ device，这一代没有那个 1:2 陷阱**） |
| 每 TensorCore | **2 个 MXU** ＋ 1 个向量单元 ＋ 1 个标量单元 |
| Peak bf16 / chip | **918 TFLOPs** |
| Peak Int8 / chip | 1836 TOPs |
| HBM 容量 / chip | **32 GB** |
| HBM 带宽 / chip | **1638 GBps** |
| ICI 双向 / chip | **800 GBps**，**4 个 ICI 端口** |
| 拓扑 | **2D torus**（⚠️ 不是 v7 的 3D） |
| Pod | **256 chip** ｜ BF16 / Pod 234.9 PFLOPs |
| 每 host | 8 chip ｜ DRAM 1536 GiB |
| 特性 | SparseCore |

> ⭐ **官方定位原文值得原样引**：
> 「This system is optimized for **transformer, text-to-image, and convolutional
> neural network (CNN)** training, fine-tuning, and serving.」
> ——&nbsp;**文生图印在官方定位里**，不是我们硬凑的角度。

**片上（来源：JAX 公开源码 `jax/_src/tpu_info.py`，TPU_V6E 分支）**

| 项 | 值 |
|---|---|
| lanes / sublanes | 128 / 8 |
| MXU | **256 × 256**（`MXU_COLUMN_SIZE_GEN_GE_6 = 256`），每核 **2** 个 |
| VMEM | **128 MiB / core** |
| SMEM | 1 MiB ｜ CMEM | **0** |
| HBM（源码值） | 34_400_000_000 B ＝ 32.04 GiB（对得上官方表的「32 GB」十进制口径） |
| 带宽（源码值） | 1.64e12 B/s（对得上官方 1638 GBps） |
| SparseCore | 2 core × 16 subcore × **8 lane**，每 subcore VMEM **256 KiB** |
| accumulators | 源码写 0 ＝ Not Available，**本讲不提这一项** |

> ⭐⭐ **一个值得画出来的对照**：v6e 是 **128 MiB VMEM / core、1 core / chip**；
> v7 是 **64 MiB / core、2 core / chip**。**每颗芯片都是 128 MiB，但 v6e 是
> 「一个大核」，v7 是「两个半核」。** 对大张量单流的扩散负载，一个大核不用切。

---

## 二、NVIDIA H100 SXM5

**来源①：NVIDIA H100 Datasheet（官方 PDF）**

| 项 | 值 |
|---|---|
| BFLOAT16 Tensor Core | 数据表印 **1,979 TFLOPS**，脚注 2 写着 **With sparsity** |
| → **稠密 BF16** | **989.5 TFLOPS**（⛔ 本讲一律用这个。跟专题二对 GB200 的处理同一口径） |
| FP8 Tensor Core | 3,958 TFLOPS（同样带稀疏） |
| GPU memory | **80 GB** |
| GPU memory bandwidth | **3.35 TB/s** |
| NVLink | 900 GB/s ｜ PCIe Gen5 128 GB/s |
| TDP | up to 700 W |

**来源②：NVIDIA《NVIDIA Hopper Architecture In-Depth》官方技术博客**

| 项 | 值 |
|---|---|
| SM 数（SXM5） | **132**（8 GPC × 66 TPC × 2 SM/TPC；整颗 GH100 还有更多，SXM5 启用 132） |
| FP32 CUDA Core | **128 / SM**，全片 **16,896** |
| Tensor Core | **4 个第四代 / SM**，全片 **528** |
| L1 ＋ 共享内存 | **256 KB / SM**（合计一块，可配） |
| 共享内存上限 | **可配到 228 KB / SM** |
| 寄存器堆 | **256 KB / SM**，全片 33,792 KB |
| L2 | **50 MB** |
| 每 SM 最多 warp / 线程 | 64 warp ／ 2048 线程 ｜ compute capability 9.0 |

---

## 三、这一讲的全部推导（每一步都能当场复核）

### ① 三个比值 —— 「算力强、显存弱」的精确版本

```
算力    918  ÷ 989.5 = 92.8%   → 基本打平
HBM 带宽 1638 ÷ 3350  = 48.9%   → 只有一半
HBM 容量  32  ÷ 80    = 40.0%   → 只有四成
```

### ② 屋脊点（＝ 专题二 §1 那条 312 的线，换两颗芯片再算一次）

```
v6e   918e12  ÷ 1.638e12 = 560 FLOP / byte
H100  989.5e12 ÷ 3.35e12 = 295 FLOP / byte
比值  560 ÷ 295 = 1.90 倍
```

> ⭐⭐ **这一讲的钩子就是这两个数。** 专题二算过 B200 是 312、TPU v7 是 312
> ——&nbsp;**两家旗舰几乎一样**。而 v6e 是 **560**，是这一家子里唯一的异类。
> 「算力强、显存弱」翻译成一句可算的话就是：**它要求你每搬一个字节多算一倍的次数。**

> ⛔ **口径三连检（照专题二的规矩）**
> ① H100 用**稠密**不用稀疏 ——&nbsp;数据表印的是稀疏值，差整整一倍
> ② v6e **1 TensorCore / chip**，per-chip 就是 per-device，不用除 2
> ③ 两边都取 **bf16**，不拿 v6e 的 Int8 去对 H100 的 BF16

---

## 四、⛔ 已知**不能**用的数

- **v6e vs H100 的扩散端到端 benchmark**：我们只有一组同模型双平台实测
  （HunyuanVideo-1.5：v6e-8 4.8 min / H100×8 FA2 4.3 min / H100×8 DeepCache 2.4 min）。
  ⭐ **它不支持「v6e 更快」**。本讲**不做端到端速度对比** ——&nbsp;
  现场已定：只讲硬件结构与负载匹配，不比 benchmark。
- 我们仓库里那些漂亮倍数（S3Diff 10.8×、Wan2.1 54s→5.3s、CogVideoX VAE 12s→1.3s）
  **全是 TPU 内部「编译前 vs 编译后」或「优化前后」**，⛔ 不是跟 GPU 比。
  ⚠️ S3Diff 那张表的列头就是「Before Compile / After Compile」——&nbsp;
  第一遍差点读成对 GPU 的加速比。
- **v6e 的 TDP**：官方文档没给，不写。
- **VMEM 带宽**：两边都没有官方数，本讲一个字不提（跟专题二同一处理）。
