# 专题二 §1 已核实规格（全部公开可引）

## TPU7x (Ironwood) — 来源：docs.cloud.google.com/tpu/docs/tpu7x（官方）
- 每 chip：2 TensorCore + 4 SparseCore
- Peak BF16 / chip：2307 TFLOPS   ｜ Peak FP8 / chip：4614 TFLOPS
- HBM / chip：表格写 "192 GiB"，正文写 "192 GB"（⚠️ 同一页两种单位！）
- HBM 带宽 / chip：7380 GBps（正文 ~7.37 TB/s）
- ICI 双向 / chip：1200 GBps；每轴双向 200 GBps；3D torus，>64 chip 由 4x4x4 cube 拼
- DCN：100 Gbps/chip　｜ Pod：9216 chip
- 双 chiplet：每 chiplet = 1 TensorCore + 2 SparseCore + 96 GB HBM，**各自独立地址空间**（不再是 MegaCore 统一地址）
- D2D 互连比一条 1D ICI 链路快 6 倍
- JAX 把一个 chip 暴露成 2 个 device
- 4 chip / VM，224 vCPU，960 GB RAM，2 NUMA node

## TPU v7 片上（来源：JAX 公开源码 tpu_info.py，TPU_7/TPU_7X 分支）
- lanes 128 / sublanes 8 ｜ MXU 256×256，每核 2 个 ｜ accumulators 128
- VMEM 64 MiB ｜ SMEM 1 MiB ｜ **CMEM = 0**
- HBM 206_000_000_000 B // tensor_cores_per_chip
- SparseCore：num_cores=2（每 device），16 subcore × 16 lane，VMEM 512 KiB，DMA granule 32 B

## GB200 NVL72 — 来源：nvidia.com/en-us/data-center/gb200-nvl72（官方规格表）
- 36 Grace + 72 Blackwell GPU；superchip = 1 Grace + 2 GPU
- GPU 显存|带宽：13.4 TB HBM3E | 576 TB/s（整域）；372 GB | 16 TB/s（每 superchip）
  → 每 GPU：186 GB、8.0 TB/s
- NVLink：整域 130 TB/s；每 superchip 3.6 TB/s → 每 GPU 1.8 TB/s
- FP16/BF16 Tensor Core：360 PFLOPS 稀疏 → **dense 180 PFLOPS** → 每 GPU dense 2.5 PFLOPS
- FP8/FP6：720 PFLOPS 稀疏 → dense 360 → 每 GPU dense 5 PFLOPS
- NVFP4：1440 稀疏 → dense 720 → 每 GPU dense 10 PFLOPS
- CPU：2592 Neoverse V2 核（每 superchip 72）；17 TB LPDDR5X | 14 TB/s

## GB300 NVL72 — 来源：nvidia.com/en-us/data-center/gb300-nvl72（官方规格表，2026-08-31 核）
- 72 Blackwell Ultra GPU + 36 Grace CPU；GPU 显存 20 TB | 最高 576 TB/s（整域）→ 每 GPU 约 288 GB
- FP16/BF16 Tensor Core：**360 PFLOPS**（稀疏）→ dense 180 → **每 GPU dense 2,500 TFLOP/s**
- FP8/FP6 Tensor Core：**720 PFLOPS**（稀疏）→ dense 360 → **每 GPU dense 5,000 TFLOP/s**
- FP4 Tensor Core：`1440 | 1080²`，脚注 2 = without sparsity → 每 GPU dense **15 PFLOPS**
- 脚注 1 原文：`All Tensor Core specifications are with sparsity unless otherwise noted.`
- NVLink 整域 130 TB/s；CPU 2592 Neoverse V2 核
- ⭐ **BF16 / FP8 与 GB200 NVL72 完全相同**。Blackwell Ultra 提升的是 dense FP4（10 → 15 PFLOPS/GPU，
  官方文案「1.5x more dense FP4」）与 attention（2×），**不是** BF16/FP8。
  → 本仓曾按「GB200 × 1.2」写成 2,700 / 5,400，**官方无此数**，2026-08-31 已全仓更正。

## HGX B200 / HGX B300 — 来源：nvidia.com/en-us/data-center/hgx（官方规格表，2026-08-31 核）
- 均为 8 GPU 一板。脚注 2：`Specification in Sparse. Dense is ½ sparse spec shown.`
- FP16/BF16：两者都是 **36 PFLOPS**（稀疏）→ dense 18 → **每 GPU 2,250 TFLOP/s**
- FP8/FP6：两者都是 **72 PFLOPS**（稀疏）→ dense 36 → **每 GPU 4,500 TFLOP/s**
- FP4（脚注 1 `Sparse | Dense`）：B300 `144 | 144`… B200 `108 | 72`
- ⭐ **同代不同封装差 11.1%**：NVL72 里的 GPU 是 2,500，HGX 板上的是 2,250。
  两者都叫 Blackwell / Blackwell Ultra，**光看架构名分辨不出该用哪个**。
  机型对应：`a4x-highgpu-4g` = GB200 NVL72 ｜ `a4x-max` = GB300 NVL72 ｜ `a4-highgpu-8g` = HGX B200

## B200 芯片内部 — 来源：docs.nvidia.com/cuda/blackwell-tuning-guide（官方）
- compute capability 10.0
- 每 SM 最大并发 warp：64（Hopper 是 48）
- 寄存器堆：64K 个 32-bit / SM（= 256 KB）；每线程最多 255 个
- 每 SM 最多 32 个 thread block
- **shared memory / SM：228 KB**（单 block 最多 227 KB，CUDA 保留 1 KB）
- L1 + texture + shared 合计上限 **256 KB / SM**（与 Hopper 相同）
- carveout 可选：0/8/16/32/64/100/132/164/196/228 KB
- 静态 shared 分配仍限 48 KB（架构兼容）
- **L2：126 MB**（原文 "The NVIDIA GB200 GPU increases the L2 cache capacity to 126 MB"）
- HBM3/HBM3e，"capacity up to 180 GB"（⚠️ 与 GB200 那张表的 186 GB/GPU 不同 —— HGX B200 vs GB200 里的 B200）
- thread block cluster：可移植上限 8，B200 可 opt-in 到 16

## ⚠️ 未能从 NVIDIA 官方文档核实
- **B200 的 SM 数**。第三方拆解一致给出：物理 160（每 die 80，双 die），激活 **148**。
  官方 CUDA 文档与架构简报里没有直接写。写进课件要标明来源等级。

## ⭐ 两个由上述数字算出来的结论（推导链完整）
1. **算力/带宽比几乎完全相同**
   - v7  ：2307 TFLOPS ÷ 7.380 TB/s = **312.6 FLOP/byte**
   - B200：2500 TFLOPS ÷ 8.00  TB/s = **312.5 FLOP/byte**
   - FP8 同样：4614/7.38 = 625.2 ｜ 5000/8 = 625.0
2. **片上 SRAM 总量同一量级，但归属完全相反**
   - B200：L2 **126 MB**（硬件自动管）＋ shared memory 228 KB × SM
   - v7  ：VMEM **64 MiB × 2 核 = 128 MiB**（编译器显式管）＋ CMEM = 0
