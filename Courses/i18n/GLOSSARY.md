# English edition — glossary

Canonical English for every recurring term. Anchors: NVIDIA CUDA C++ Programming
Guide / PTX ISA / Hopper & Blackwell tuning guides; Google Cloud TPU docs, JAX & Pallas
docs, the JAX Scaling Book; DeepSeek / FlashAttention papers; vLLM / SGLang docs.
If a term isn't here, use what those sources use. If you add one, add it here.

## Core quantities

| 中文 | English | Notes |
|---|---|---|
| 算力 | compute; peak FLOP/s | "compute-bound", not "computing power" |
| 带宽 | (memory) bandwidth | |
| 显存 | HBM; device memory | **Never** "video memory" / "graphics memory". Capacity → "HBM capacity" |
| 显存占用 | memory footprint | |
| 算术强度 | arithmetic intensity (FLOPs/byte) | |
| 312 那条分水岭 / 分界线 | the ridge point (312 FLOPs/byte) | roofline term |
| 屋顶线 | roofline (model) | |
| 访存 | memory access; a load | |
| 搬运 / 搬数据 | data movement; moving data | "the movement side" vs "the math side" |
| 算 / 搬 | compute / data movement | 「算和搬的兑换比」 → "the exchange rate between compute and data movement" |
| 带宽侧 / 算力侧 | memory-bound / compute-bound side | |
| 峰值 | peak | |
| 实测 | measured | |
| 量级 | order of magnitude | |
| 规格 / 规格表 | spec / spec sheet | |
| 拍 / 周期 | cycle(s) | 「30 拍」 → "30 cycles" |
| 乘加 | multiply-accumulate (MAC); FMA | "MACs per cycle" |
| 吞吐 | throughput | |
| 延迟 | latency | |
| 延迟隐藏 | latency hiding | |

## Memory hierarchy

| 中文 | English | Notes |
|---|---|---|
| 片上 | on-chip | |
| 片上存储 / 暂存 | on-chip SRAM; scratchpad | TPU VMEM is a software-managed scratchpad |
| 共享内存 (GPU) | shared memory | write in full; avoid "SMEM" on the GPU side to avoid clashing with TPU SMEM |
| SMEM (TPU) | SMEM (scalar memory) | |
| 寄存器 / 寄存器堆 | register / register file | |
| 向量寄存器 (TPU) | vector register (vreg) | |
| 缓存 / 缓存缺失 / 命中 | cache / cache miss / hit | |
| 缓存行 | cache line | 128 B; sector = 32 B |
| 合并访存 | memory coalescing; coalesced access | |
| bank 冲突 | bank conflict | |
| 换页 / TLB | page walk; TLB miss | |
| 张量内存 | Tensor Memory (TMEM) | Blackwell |
| 分布式共享内存 | distributed shared memory (DSMEM) | |
| 双缓冲 | double buffering | |
| 布局 | (memory) layout | 8 × 128 tiled layout |
| 补零 / 填充 | padding | |
| 重排 | relayout; data reshuffle | |
| 兜底 | fallback; safety net | "the cache is the safety net when you guess wrong" |

## Compute units & hierarchy

| 中文 | English | Notes |
|---|---|---|
| 芯片 / 颗 | chip | |
| die / 封装 / 双 die | die / package / dual-die package | |
| TensorCore (TPU 核) | TensorCore (one word) | TPU core. Distinct from NVIDIA **Tensor Core** (two words) — keep the spelling difference exactly |
| Tensor Core (NVIDIA) | Tensor Core | |
| 张量核 | Tensor Core | |
| MXU / 矩阵单元 | MXU (matrix unit) | 256×256 systolic array on v7 |
| 向量单元 | vector unit (VPU) | |
| 标量单元 | scalar unit | |
| SparseCore | SparseCore | |
| 流式多处理器 / SM | SM (streaming multiprocessor) | |
| 处理块 / SM 子分区 | SM sub-partition (processing block) | 4 per SM |
| 线程 / 线程束 / 线程块 / 网格 | thread / warp / thread block (CTA) / grid | |
| warp 组 | warpgroup | Hopper |
| 线程块簇 | thread block cluster | |
| lane / sublane | lane / sublane | |
| 记分板 | scoreboard | |
| 调度器 / warp 调度器 | (warp) scheduler | |
| 发射 | issue | "issue slot" |
| 取指 / 译码 | instruction fetch / decode | |
| 占用率 | occupancy | |
| 驻留 | resident | "resident warps" |
| 分支分歧 | branch divergence; warp divergence | |
| 屏障 | barrier | mbarrier (async transaction barrier) |
| 脉动阵列 | systolic array | |
| 累加器 | accumulator | |
| 复用率 | reuse factor; data reuse | |
| 内积序 / 外积序 | inner-product order / outer-product order | |
| 收缩维 | contraction dimension (K) | |
| 驻留权重 (脉动阵列) | weight-stationary | |
| 矩阵乘 | matrix multiply; matmul; GEMM | "matmul" in prose, "GEMM" for kernels/libraries |
| 矩阵指令 | MMA instruction | mma / wgmma / tcgen05.mma |
| 分块 / 块 | tiling / tile | 「块多大」 → "tile size" |

## Software & execution model

| 中文 | English | Notes |
|---|---|---|
| 编译期 / 运行时 | compile time / runtime (adj: compile-time, runtime) | "runtime decisions" |
| 运行时适应 | runtime adaptation | course motif |
| 静态形状 / 动态形状 | static shapes / dynamic shapes | |
| 静态调度 | static scheduling | |
| 算子 | op (operator) | "kernel" when it means the implementation |
| 算子融合 | (operator) fusion; kernel fusion | course motif — use "fusion" consistently |
| 内核 / kernel 作者 | kernel / kernel author | |
| 编程模型 | programming model | |
| 手写 kernel | hand-written kernel | |
| 调优 | tuning; performance tuning | |
| 编译器 | compiler (XLA, Mosaic) | |

## Interconnect & scale-out

| 中文 | English | Notes |
|---|---|---|
| 互联 | interconnect | |
| 芯片间互联 (TPU) | ICI (inter-chip interconnect) | |
| 拓扑 / 环面 | topology / torus | 3D torus |
| 光交换 | optical circuit switch (OCS) | |
| 集合通信 / 通信原语 | collectives / communication primitives | all-reduce, all-gather, reduce-scatter, all-to-all |
| NVLink 域 | NVLink domain | NVL72 |
| 横向扩展 / 纵向扩展 | scale-out / scale-up | |
| 切分 | sharding; partitioning | |

## Models & workloads

| 中文 | English | Notes |
|---|---|---|
| 稠密模型 | dense model | |
| 专家 / 激活参数 | expert / active parameters | "37B active", not "37B activated" |
| 激活值 | activations | |
| 推理 / 训练 | inference / training | |
| 预填充 / 解码 | prefill / decode | |
| 朴素注意力 | naive (unfused) attention | |
| 注意力分数矩阵 | attention score matrix | |
| 在线 softmax | online softmax | |

## Course furniture

| 中文 | English | Notes |
|---|---|---|
| 专题 | Topic | "Topic 2" |
| 主线 / 深入 / 实操 | Core track / Deep dives / Hands-on | |
| 课件 / 正文 | course page / main text | |
| 讲义 / 授课稿 | instructor notes / lecture script | |
| 精讲版 / 完整版 | L200 · focused edition / L300 · full edition | keep L-levels |
| 外传 | side track | |
| 课后延伸 | Further reading (beyond the lecture) | |
| 课前勿点 | Don't open before class | |
| 热身题 | warm-up questions | |
| 出处 / 溯源 | source / provenance | |
| 讲的时候跳过 | skip when presenting | |
| 被问到再展开 | expand only if asked | |
| 显微镜 | microscope (deep-dive page) | "GPU Microscope" |
| 本节讲课录音 | (removed in English edition) | audio/video are Chinese-only for now |
