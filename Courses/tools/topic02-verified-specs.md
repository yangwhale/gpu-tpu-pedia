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
