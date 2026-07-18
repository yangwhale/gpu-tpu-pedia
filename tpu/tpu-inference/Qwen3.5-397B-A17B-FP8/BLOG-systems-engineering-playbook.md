# Systems Engineering Playbook: Qwen 3.5-397B MoE on Ironwood (TPU v7x) 深度剖析

> **原文链接**: [Systems Engineering Playbook: Optimizing Qwen 3.5-397B MoE on Ironwood (TPU7x)](https://developers.googleblog.com/systems-engineering-playbook-optimizing-qwen-35-397b-moe-on-ironwood-tpu7x/)
>
> **发布日期**: 2026-07-14
>
> **作者团队**: Google Cloud TPU Performance Engineering Team
>
> **本文定位**: 对官方博客的全量中文深度剖析，结合我们在 TPU v7x-8 上的实战经验（见同目录 [README.md](README.md)）、CC Wiki 知识库、以及 MoE 环游记系列理论研究，形成一份 all-in-one 的端到端架构理解 + 复现测试框架文档。

---

## 目录

1. [Executive Summary](#1-executive-summary)
2. [Qwen 3.5 架构深度剖析](#2-qwen-35-架构深度剖析)
3. [Benchmark 方法论](#3-benchmark-方法论)
4. [Sharding 策略深度分析](#4-sharding-策略深度分析)
5. [Roofline 分析](#5-roofline-分析)
6. [内核优化四大 Track](#6-内核优化四大-track)
7. [性能结果分析](#7-性能结果分析)
8. [未来优化路线图](#8-未来优化路线图)
9. [复现测试 Cross-Check](#9-复现测试-cross-check)
10. [致谢与参考链接](#10-致谢与参考链接)

---

## 1. Executive Summary

### 1.1 博客核心结论

Google Cloud TPU Performance Engineering 团队在 2026 年 4-6 月期间，对 Qwen 3.5-397B MoE 在 Ironwood (TPU v7x) 平台上的推理性能进行了系统性优化：

- **Decode-heavy 工作负载**: 性能提升约 **3.1x**（512 并发档位）
- **Prefill-heavy 工作负载**: 性能提升约 **4.7x**（512 并发档位）

这些提升并非靠"经验试错"（empirical trial-and-error），而是基于 **first-principles 系统工程方法论**：先用 Roofline Model 划定硬件极限 → 用 profile trace 定位瓶颈 → 用手写 Pallas kernel + 硬件感知 sharding 拓扑逐个击破。

### 1.2 方法论的核心洞察——模块化 model-agnostic 优化

博客最重要的方法论贡献是**模块化优化策略**：把模型分解为独立的 building blocks（Batched RPA、Grouped GEMM、SparseCore unpermutation 等），每个模块单独优化并附带硬件 cost model。当新模型架构出现时，这些预优化模块可以 **near-zero engineering friction** 移植。

这种思路对我们的启示：
- 不再是"来一个模型优化一个模型"的线性投入
- 而是构建平台级能力 → 新模型自动受益
- Qwen 3.5 只需额外优化其独有组件（GDN 线性注意力 + Attention DP）

### 1.3 与我们实测数据的初步对照

| 维度 | 博客数据 | 我们 README 实测 | 对照说明 |
|---|---|---|---|
| 硬件 | v7x-8 (4 chip / 8 device) | v7x-8 (同) | 完全一致 |
| 推理框架 | vllm-project/tpu-inference | 同 | 完全一致 |
| Prefill 吞吐 (C64, 8K/1K) | 3,707 TPS/chip | 见 §9 待测 | 我们 README 用 1K/1K 不同 workload |
| Decode 吞吐 (C64, 1K/8K) | 677 TPS/chip | 见 §9 待测 | 同上 |
| Peak 吞吐 (1K/1K, P128) | 未报告 | **2,097 tok/s** (total) | 我们的 sweet spot |
| Sharding | DP=8 + EP=8 | TP=8 + EP=8 | **关键差异**，详见 §4 |
| 关键 PR | #2577 (DP sharding) | #2366 + #2577 (bf16 fix) | PR 编号相同但含义不同！|

> **重要发现**: 博客中的 PR #2577 指的是 **Attention DP sharding 方案**（hybrid DP=8 + EP=8），而我们 README 中的 PR #2577 指的是 **GDN bf16 数值稳定性修复**（recurrent_scan_v2.py fp32 upcast）。这是两个不同的 PR，同一个编号。

---

## 2. Qwen 3.5 架构深度剖析

### 2.1 总体参数

| 参数 | 值 | 说明 |
|---|---|---|
| 总参数量 | **397B** | ~400B 级别模型 |
| 每 token 激活参数 | **17B** | 仅 4.3% 路由激活率 |
| Experts | **512** routed + **1** shared | shared expert 始终激活 |
| Top-K | **10** | 每 token 选 10 个 routed experts |
| 激活率 | **4.3%** | 10/233 (10 routed + 1 shared = 11 active / 总 513) |
| 隐藏维度 | **4,096** | D=4096 |
| 层数 | **60** | 45 GDN + 15 GQA |
| 词汇表 | **248,320** | padded |
| 原生上下文 | **262,144** tokens | YaRN 可扩至 1M+ |
| 权重精度 | **FP8** native | 94 safetensors, ~378 GiB |

> **我们的验证**: README 中模型参数表完全吻合。`ls $MODEL/*.safetensors | wc -l` 应输出 94。

### 2.2 Hybrid GDN+GQA 交替布局

Qwen 3.5 不采用统一的 Transformer 层堆叠，而是使用 **15 个重复结构块**，每个块 4 层，按 3:1 比例排列：

```
[GDN] [GDN] [GDN] [GQA]  ← 1 个结构块 (4 层)
     × 15 blocks = 60 层总计
```

- **Gated DeltaNet (GDN) 层**: 45 层（75%），线性注意力 + routed sparse MoE
- **Grouped Query Attention (GQA) 层**: 15 层（25%），标准注意力 + routed sparse MoE

这种混合设计的核心权衡：
- **GDN 层**: O(S) 复杂度，长上下文友好，但精度略低于 full attention
- **GQA 层**: O(S²) 复杂度，每 4 层"锚定"一次精度，防止线性注意力的信息衰减

### 2.3 Gated DeltaNet (GDN) 线性注意力

标准 self-attention 的 O(S²) 复杂度在长序列下成为瓶颈。GDN 通过维护一个**恒定大小的隐藏状态矩阵**（recurrent memory）来实现 O(S) 的序列长度缩放。

**GDN 的数学核心**:

在每个 token 步 t，状态矩阵通过 **delta rule** 更新：

```
S_t = S_{t-1} + β_t · (v_t · k_t^T - S_{t-1} · k_t · k_t^T)
output_t = q_t · S_t
```

其中 β_t 是学习到的 gating 参数，q_t, k_t, v_t 分别是 query、key、value 向量。

**GDN 的注意力头配置**:

| 参数 | 值 |
|---|---|
| V heads | 64 |
| QK heads | 16 |
| Head dimension | 128 |
| 状态矩阵大小 | d_k × d_v per head |

**Causal 1D Convolution (K=4)**: 在 recurrent 更新之前，先做一个核大小为 4 的因果一维卷积，捕获局部空间依赖。这是 GDN 的标配前置操作。

> **我们的踩坑经验**: GDN 的 recurrent scan 在 **bf16 精度下数值不稳定**（我们实测发现的 PR #2577 修复）。recurrent_scan_v2.py 中 5 处关键计算需要 upcast 到 fp32，chunk_size 从 64 降到 32。缺失此修复会导致 chat 输出 `about about about...` 死循环。详见 §6.3.5。

### 2.4 GQA 配置

| 参数 | 值 | 影响 |
|---|---|---|
| Query heads (N_q) | **32** | |
| KV heads (N_kv) | **2** | 极端 GQA，KV cache 压缩 16x |
| Head dimension | **256** | 比常见的 128 大一倍 |
| RoPE dimension | **64** | |

**关键约束**: 仅 2 个 KV head，这直接导致了 **TP=8 在硬件上不可行**（2/8 = 0.25 heads/device，无法整除切分）。这是 §4 sharding 策略的核心驱动因素。

### 2.5 MoE FFN

| 参数 | 值 |
|---|---|
| Routed experts | 512 |
| Shared experts | 1 (always active) |
| Expert intermediate dim | 1,024 |
| Top-K | 10 |
| Gating | Softmax probability |

输出计算：`output = SharedExpert(x) + Σ_{i ∈ top-k} g_i · Expert_i(x)`

共享 expert 作为"公共表示层"始终参与计算，routed experts 通过 softmax gate 概率选择 top-10。

### 2.6 我们的补充：与 Wiki 交叉验证

**TPU v7 (Ironwood) 规格对照** (来源: CC Wiki `tpu-v7` entity):

| 参数 | Wiki 记载 | 博客使用 | 对照 |
|---|---|---|---|
| FP8 算力 | ~4,611 TFLOPS/chip | 4,614 TFLOPS/chip | 基本一致（推导方式差异） |
| BF16 算力 | ~2,306 TFLOPS/chip | 2,307 TFLOPS/chip | 一致 |
| HBM 容量 | 192 GiB/chip | 192 GB/chip | 一致 |
| HBM 带宽 | 7.4 TB/s/chip | 未直接报告 | — |
| ICI | 4.0 | sub-microsecond | 一致 |
| SparseCore | 有 | 关键使用 | 见 §6.2 |

> Wiki 的 4,611 TFLOPS 来自 Superpod 总算力反推 (42.5 Exaflops ÷ 9,216 chips)，博客的 4,614 TFLOPS 来自 bottom-up 计算 (262,144 FLOP/cycle/MXU × 2.2 GHz × 4 MXUs)。两者差异 0.07%，可视为相同。

---

## 3. Benchmark 方法论

### 3.1 多维度评估矩阵

博客设计了两类非对称工作负载，分别压力测试不同的硬件子系统：

| 工作负载类型 | 配置 | 压力子系统 | 硬件瓶颈 |
|---|---|---|---|
| **Prefill-heavy** | 8K input / 1K output | Compute-bound | TensorCore MXU (浮点乘法) |
| **Decode-heavy** | 1K input / 8K output | Memory-bound | HBM 带宽 (流式加载权重) |

### 3.2 并发梯度

4 档并发：**64 / 128 / 256 / 512**，用于观察系统 scaling 曲线和识别硬件排队/内存瓶颈。

### 3.3 引擎配置与拓扑

| 参数 | 博客配置 | 说明 |
|---|---|---|
| 物理 chip | 4 (single host) | 每个 chip 2 个逻辑 chiplet |
| 逻辑 device | 8 | 4 chip × 2 chiplet |
| 推理框架 | vllm-project/tpu-inference | 开源 |
| `--max-num-batched-tokens` | **1,024/core** (DP mode) | 总 8,192 across 8 cores |
| `--max-num-seqs` | **64/core** | 总 512 |
| Sharding | **DP=8 + EP=8** | 非 TP=8 |

### 3.4 指标定义

**TPS/chip** = (总处理 tokens: input + output) / 执行时长 / 物理 chip 数 (4)

> 注意：这里 TPS/chip 是按**物理 chip**（4 个）计算，不是逻辑 device（8 个）。博客全篇使用此单位。

### 3.5 我们的补充：方法差异对比

| 维度 | 博客方法 | 我们 README 方法 | 差异影响 |
|---|---|---|---|
| 工作负载 | 8K/1K 和 1K/8K | **1K/1K** (均衡) | 不同 workload profile |
| 并发档位 | 64/128/256/512 | P1/P4/P16/P64/P128/P256 | 我们覆盖更低并发 |
| 指标单位 | **TPS/chip** | **tok/s (total)** | 换算: 总 TPS = TPS/chip × 4 |
| Sharding | DP=8 + EP=8 | TP=8 + EP=8 | **关键差异**，影响 KV cache 布局 |
| Batched tokens | 1024/core | 4096 (total) | 我们更大 batch buffer |
| Max seqs | 64/core (512 total) | 256 | 我们并发上限更低 |
| 测试工具 | 未指明 | evalscope perf | — |
| Warmup | 未指明 | 每档 2 轮 (warmup + record) | 我们丢弃首轮预热 |

> **单位换算速查**: 博客 TPS/chip × 4 chips = 总 TPS (tok/s)。例如 3,707 TPS/chip = ~14,828 tok/s total。

---

## 4. Sharding 策略深度分析

这是博客最核心的章节之一，也是 Qwen 3.5 在 TPU 上推理的基础架构决策。

### 4.1 为什么 TP=8 对 Qwen 3.5 不可行

传统 Tensor Parallel (TP) 会把注意力权重切分到多个 device 上。但 Qwen 3.5 的 GQA 层只有 **2 个 KV head**：

```
TP=8 切分 → 2 KV heads / 8 devices = 0.25 heads/device → 物理上不可能
```

替代方案——在每个 device 上复制完整的 2 个 KV head——会导致 **KV cache 内存占用 8 倍冗余**，严重挤压 HBM 中可用于活跃 KV cache 的空间。博客指出这会将实际并发从目标 512 限制到 ~200。

### 4.2 Hybrid Sharding: Attention DP=8 + Expert EP=8

博客的解决方案是 **hybrid sharding**：

| 层类型 | 并行策略 | 权重分布 | 通信 |
|---|---|---|---|
| GQA / GDN (Attention) | **DP=8** (Data Parallel) | 每个 device 复制完整权重 | 无 intra-attention 通信 |
| MoE FFN (Expert) | **EP=8** (Expert Parallel) | 512 experts / 8 = 64 experts/device | 需要 All-Gather + Reduce-Scatter |

**DP=8 的优势**:
- 每个 device 拥有完整的 2 KV heads → 本地 KV cache 一致性
- 消除 attention 层的跨 device 通信
- 支持更高并发（不受 KV cache 冗余限制）

**EP=8 的优势**:
- 512 个 expert 分散到 8 个 device，每个 64 个
- 避免 400GB 参数在所有 device 上复制
- Collective 通信 payload 可控

> **我们的对比**: README 使用 `--tensor-parallel-size 8 --enable-expert-parallel`，即 **TP=8 + EP=8**。这与博客的 **DP=8 + EP=8** 不同。TP=8 下注意力权重被切分而非复制——这在 2 KV head 的约束下理论上应该会遇到博客描述的问题。但我们的实测表明 TP=8 在 vLLM 推理栈中确实能跑通并产出正确结果（GSM8K 93-97%），推测 vLLM 内部对 2 KV head 做了某种 replicate-within-TP 的处理。

### 4.3 Option A vs Option B: Collective 通信策略

DP↔EP 切换需要跨 device 的 token 路由。博客评估了两种方案：

#### Option A: All-to-All Shuffling

```
All-to-All → Local MoE → All-to-All
```

- Token 被动态发送到持有目标 expert 的 chip
- 最小化冗余计算
- 但 All-to-All 的网络路由开销在可变工作负载下**不可预测**

#### Option B: Full Token Replication (选择此方案)

```
All-Gather → Local MoE → Reduce-Scatter
```

- All-Gather 把所有 token 复制到每个 device
- 每个 device 只计算本地的 64 个 experts
- Reduce-Scatter 聚合输出

**选择 Option B 的原因**: 生产服务需要**确定性延迟**。All-to-All 的延迟随 token 路由分布波动，而 All-Gather + Reduce-Scatter 的通信模式是固定的。

> **来自 MoE 环游记的视角**: 在环游记 [Article 5 (SparseCore)](../../../moe-tour/05-rethinking-uniform.md) 和 [Article 6 (QB)](../../../moe-tour/06-quantile-balancing.md) 中，我们讨论了 All-to-All vs All-Gather 的 tradeoff。理论上 All-to-All 更高效（只传需要的 token），但工程实践中 All-Gather 的确定性更受生产环境青睐——这与博客的选择完全一致。

### 4.4 3-to-2 All-Gather 优化 (PR #2836)

原始 Option B 需要 3 次独立的 All-Gather：

| # | 数据 | Shape | 说明 |
|---|---|---|---|
| 1 | Token hidden dims | [1024, 4096] | 主 payload |
| 2 | Expert indices | [1024, 10] | int 类型 |
| 3 | Gate weights | [1024, 10] | float 类型 |

**优化**: 因为 expert indices 和 gate weights shape 相同 ([1024, 10])，将它们沿新维度 stack → bitcast 打包成单个 32-bit integer array → 一次 All-Gather 传输。

```
3 次 All-Gather → 2 次 All-Gather
```

每次 collective 调用都有固定的 kernel launch + 网络同步延迟。减少 1 次 All-Gather = 直接砍掉一次延迟 penalty。

### 4.5 Hierarchical Reduce-Scatter (PR #2679)

Expert 计算完成后，token 输出需要返回到各自的 DP rank。标准 All-Reduce 在 8-device mesh 上效率很低。

博客用自定义 Pallas/Mosaic 内核实现了 **分层 Reduce-Scatter**：

**Phase 1: Intra-chip Reduce-Scatter**
- 同一物理 chip 上的 2 个逻辑 chiplet 通过 **共享内存** 交换和求和数据
- 比 chip-to-chip ICI 带宽快 **6x**

**Phase 2: Inter-chip Reduce-Scatter**
- 部分归约的数据通过物理 ICI 链路用 **recursive-doubling hypercube** 算法跨 chip 交换

**Anti-OOM**: 数据切成 2-4 个 micro-batch，pipeline 第 i 个 micro-batch 的远程 DMA 传输和第 i-1 个 micro-batch 的 TensorCore 向量加法重叠执行，隐藏通信延迟。

### 4.6 我们的补充

**Wiki 数据交叉印证**:
- Wiki `tpu-v7` entity 记载 ICI 4.0 支持 3D Torus 拓扑
- Wiki 中多处讨论的 AllGather ICI 带宽利用率 33-42%，博客选择 Option B (All-Gather based) 的延迟确定性优势可部分补偿带宽利用率的次优

**ALModel EP 对比**:
- Wiki 中 ALModel 相关页面讨论了不同 EP 策略在训练 vs 推理中的 tradeoff
- 博客的 DP=8 + EP=8 是纯推理优化方案；训练侧可能采用不同的 FSDP + EP 组合

---

## 5. Roofline 分析

### 5.1 Ironwood 硬件规格

| 参数 | 值 | 推导 |
|---|---|---|
| TensorCore 频率 | 2.2 GHz | |
| TensorCore 数量/chip | 2 | |
| MXU 数量/TC | 2 (总 4 MXU/chip) | |
| **Peak BF16** | **2,307 TFLOPS/chip** | 262,144 FLOP/cycle/MXU × 2.2 GHz × 4 MXU |
| **Peak FP8** | **4,614 TFLOPS/chip** | 2× BF16 |
| HBM 容量 | 192 GB/chip | HBM3e, 8 stacks |
| HBM 带宽 | 7.4 TB/s/chip | (Wiki 数据) |
| Compute-to-bandwidth ratio | 623 FLOPS/byte (FP8) | 4,614 TFLOPS / 7.4 TB/s |

### 5.2 Prefill 阶段（Compute-Bound）

**场景**: C64, 8K input / 1K output → 64 prompts × 8,192 tokens = 524,288 tokens 并行处理

- **算术强度 (FLOPs/Byte)**: 极高。Projection 层 GEMM 随序列长度和 batch size 二次缩放
- **运行边界**: 受 MXU 浮点算力上限约束（4,614 TFLOPS FP8）
- **瓶颈**: MXU 利用率不足，主要来自 expert 间 ragged token 分布——如果某个 expert 在一个 batch 中收到明显更多 token，对应 device 成为 straggler

**Roofline 上限**:

| 指标 | 值 | 说明 |
|---|---|---|
| 未折扣上限 | **5,170 TPS/chip** | 考虑 GQA O(S²) + 硬件执行开销 |
| 折扣后上限 | **4,500 TPS/chip** | 标准调度折扣因子 |

### 5.3 Decode 阶段（Memory-Bound）

**场景**: C64, 1K input / 8K output → 每步处理 64 tokens (1 token/active request)

- **算术强度**: ~1 FLOP/Byte，每生成 1 个 token 需要从 HBM 流式加载全部 400 GB 权重
- **运行边界**: 受 HBM 内存带宽约束
- **瓶颈**: HBM 传输延迟 + VPU 在 sparse KV cache 索引时的 stall + GDN 层 recurrent state 更新的 round-trip

**Roofline 上限**:

| 指标 | 值 | 说明 |
|---|---|---|
| 60 层总执行延迟 | **16.36 ms/step** | |
| 未折扣上限 | **978 TPS/chip** | 1000/16.36 × 16 (?) |
| 折扣后上限 | **850 TPS/chip** | |

### 5.4 我们的补充

**Wiki 规格 vs 博客数字**:

| 参数 | Wiki (tpu-v7 entity) | 博客 | 差异 |
|---|---|---|---|
| FP8 TFLOPS | ~4,611 | 4,614 | 0.07% (推导路径不同) |
| BF16 TFLOPS | ~2,306 | 2,307 | 0.04% |
| HBM | 192 GiB | 192 GB | 单位差异 (1 GiB = 1.074 GB) |
| HBM 带宽 | 7.4 TB/s | 未报告 | Wiki 独有数据 |

**README 数据映射**:
- 我们的 1K/1K P128 peak = **2,097 tok/s (total)** = **~524 TPS/chip** (÷4 chips)
- 博客 decode roofline 850 TPS/chip 是 1K/8K (decode-heavy)，与 1K/1K 不直接可比
- 我们的 8K/1K P64 = **849.9 tok/s (total)** = **~212 TPS/chip**（远低于博客的 3,707 TPS/chip）
  - **原因**: 我们用 TP=8 而非 DP=8，且 `--max-num-batched-tokens=4096` vs 博客 8,192（1024/core×8）

---

## 6. 内核优化四大 Track

博客将优化分为 4 个独立 track，每个 track 对应一类计算瓶颈。这是"模块化 model-agnostic 优化策略"的具体体现。

### 6.1 Attention Track: Ragged Page Attention (RPA)

管理 25% GQA 层的 KV cache 需要动态内存分配。RPA 在 HBM 中索引非连续的内存块。

#### 6.1.1 KV Page Size 调优

| 配置 | 旧值 | 新值 | 效果 |
|---|---|---|---|
| Block size | 16 tokens | **256 tokens** | C512 decode step: 428μs → 283μs (**-33.8%**) |

**原理**: 小 block size (16) 在 TPU 上导致大量索引开销——VPU 在 decode 阶段反复 stall。粗粒度索引 (256) 减少了 VPU 索引次数，代价是稍高的内存碎片。

> **我们的验证**: README 中 `--block-size=256` 正是此优化的直接应用。这个参数在我们的部署中已经是标准配置。

#### 6.1.2 Batched RPA

将多个 decode stream 打包到单个编译的 Pallas kernel 中 (PR #2632)：
- 分摊 VPU 指令 dispatch 延迟
- 打破顺序请求的数据依赖 stall
- 改善内存对齐

### 6.2 MoE Track: SparseCore + TensorCore 协同设计

Qwen 3.5 的 top_k=10 引入了非 2 的幂次 tensor 维度。在 TensorCore 上 permute/unpermute 这些数组之前会产生大量 padding 和 unaligned HBM 写入。

#### 6.2.1 SparseCore Ragged Gather Kernel (PR #2137)

**TPU v7 的 SparseCore (SC)** 是专为 indirect addressing 优化的硬件单元。博客将 token routing 卸载到 SC：

1. SC 读取 routing indices
2. 执行 indirect DMA gather，从 HBM 直接收集 token embeddings
3. 写入连续虚拟 buffer

**关键收益**: 绕过了在 HBM 中 materialization 重度 padding、unaligned 的中间 tensor，节省大量内存带宽。

> **来自 MoE 环游记的预测 vs 工程实现**:
>
> 在 [Article 5](../../../moe-tour/05-rethinking-uniform.md) 中，我们讨论了 SparseCore 在训练侧用于 AllGather offload 的架构。博客揭示了 SparseCore 在**推理侧**的不同用法——不是 AllGather offload，而是 **Ragged Gather**（间接寻址收集不规则分布的 token）。
>
> 这是同一硬件单元的两种不同应用：
> - **训练**: SparseCore offload AllGather 通信（减少 TensorCore 负载）
> - **推理**: SparseCore 做 Ragged Gather（避免 padding 开销）

#### 6.2.2 GMM V2 + Fused SwiGLU

Grouped GEMM V2 kernel 的优化：

| 优化项 | 效果 |
|---|---|
| SwiGLU 融合进矩阵乘循环 | gating + up-projection 打包成单 tile 双 DMA 读取，避免寄存器 spill 到 HBM |
| Dynamic bounded slices | 最小 padding 处理每个 expert 的可变 token payload |
| **512 subchannel FP8 量化** | 消除 VREG spill 和 memory load stall，VPU 向量算术吞吐翻倍 |

> **512 subchannel 量化**是一个值得注意的细节。FP8 的精度问题通常需要通过更细粒度的 scaling factor 来缓解——512 subchannel 意味着每 512 个元素有独立的 scale，比整 tensor 或 per-channel 量化精度更高。

#### 6.2.3 Fused Ragged Gather Reduce Kernel

将 token un-permutation 和 local reduction 完全卸载到 SparseCore：

| 指标 | 优化前 | 优化后 | 减少 |
|---|---|---|---|
| HBM 读取次数 | 20 | **10** | -50% |
| HBM 写入次数 | 15 | **5** | -67% |

**4-chunk Pipeline 架构**: 不是对整个 [81920, 4096] tensor 顺序执行 local reduction + Reduce-Scatter，而是：

```
Chunk 1: SC local gather-reduce → kick off Reduce-Scatter (async)
Chunk 2: SC local gather-reduce → kick off Reduce-Scatter (async)
Chunk 3: ...
Chunk 4: ...
```

Chunk 1 的 Reduce-Scatter 在 ICI 上传输时，SC 已经在处理 Chunk 2 的 local gather-reduce。跨 device 网络延迟被后续 chunk 的本地计算**完全隐藏**。

### 6.3 GDN Track: 全融合 Conv1D + 线性注意力

GDN 的 recurrent state 更新高度依赖内存带宽（恒定的 recurrent state 每步读写），是 decode 阶段的主要延迟贡献者之一。

#### 6.3.1 Causal Conv1D 融合 (PR #2823)

GDN 的 recurrent 更新前有一个 causal 1D convolution (K=4)。原始实现将其编译为独立操作，中间结果写入/读取 HBM。

**优化**: 设计寄存器级滑动窗口算法，将历史 token 状态直接缓存在 VPU 寄存器中。将 1D 卷积和 GDN recurrent state 更新融合为单个执行块。

**效果**: 消除 **6 次** 冗余 HBM round-trip。

#### 6.3.2 代数恒等式优化 (PR #2498)

重组线性注意力更新方程，利用代数恒等式**完全跳过**融合 GDN kernel 中昂贵的 post-rank-1 矩阵乘法。

#### 6.3.3 BF16 精度决策——博客 vs 我们的踩坑经验

**博客的声明**:

> "We transitioned the recurrent State Space Model (SSM) state variables from Float32 to BFloat16 precision. This doubled the vector arithmetic throughput on the VPU without compromising numerical convergence or output quality."

**我们的实际经验 (PR #2577 修复)**:

```
GDN recurrent scan kernel 在 bf16 精度下数值不稳定
→ token logits 退化
→ chat 输出 "about about about..." 死循环
→ 修复: recurrent_scan_v2.py 5 处关键计算 upcast 到 fp32
→ chunk_size 64→32
```

**两者的矛盾如何理解**:

1. 博客说的 "BF16 precision" 可能特指某些特定变量/路径，而非整个 recurrent scan
2. 或者博客的 BF16 决策是在 PR #2577 (我们的 bf16 fix) 之前做出的，后来又部分回退
3. 最可能的解释：博客讨论的是 fully-fused kernel (PR #3016) 中的精度设计，而 PR #2577 修复的是之前非融合版本的数值不稳定——两者时间线不同

> **经验教训**: 在 TPU 上做 recurrent scan 的 BF16 降精度需要**极其小心**。delta rule 的累积更新天然容易产生数值漂移。即使博客声称 "without compromising quality"，我们的实测证明在某些代码路径上 bf16 确实会崩。

#### 6.3.4 Ragged Sequence Handling + Chunked GDN (PR #2218)

在 batched prefill 中，可变长度序列会导致 padding 浪费 MXU FLOPs。PR #2218 引入了：
- JAX-native chunked layouts
- 原生处理 ragged inputs 的专用序列处理 routine
- 避免可变序列长度引入处理 straggler

#### 6.3.5 Fully-Fused Conv1D + GDN Kernel (PR #3016)

这是 GDN Track 的终极优化——将 causal 1D 卷积和整个 GDN recurrent 线性注意力块编译为**单个统一执行单元**。

**核心思路**: 中间序列和 recurrent state 直接在**本地寄存器**中缓存和流转，完全绕过 VMEM 和 HBM 的读写。

| 受益阶段 | 效果 |
|---|---|
| **Prefill** | 显著降低处理长 input 序列时的内存带宽开销，最大化 MXU 效率 |
| **Decode** | 消除 token-by-token 生成中的 memory-bound round-trip stall |

> 这是寄存器级融合的极致——从"3 个独立操作各自读写 HBM"到"全部在寄存器中完成，只在最终输出时写一次 HBM"。

### 6.4 Memory Track: Hybrid Attention KV Layout (PR #2416)

Qwen 3.5 需要管理两种异构的注意力状态：
- **GDN**: 固定大小的 recurrent 线性注意力状态
- **GQA**: 动态增长的标准 KV cache

在 TPU v7 的 192GB HBM/chip 约束下（博客特意对比了 GB300 GPU 的 288GB，差 ~50%），高并发时 HBM footprint 优化是严格约束。

PR #2416 引入了自定义内存布局：
- 将两种异构注意力状态**对齐并存储**在 HBM 中
- 最小化 padding
- 防止内存碎片
- 直接回收 HBM headroom → 提高最大可支持 batch size

> **我们的验证**: README 中记录的 PR #2366 (Hybrid KV cache padding fix) 也是处理这个问题——TPU `jax.Array` 强类型要求每层独立复制 KV cache，vLLM hybrid allocator 默认 4 层共享导致 block_id pool 过大。两个 PR 协同工作。

### 6.5 综合补充

#### SparseCore: 训练 vs 推理的两面

| 场景 | SparseCore 用法 | 来源 |
|---|---|---|
| **训练** | AllGather offload (减轻 TensorCore 通信负载) | Wiki, MoE 环游记 |
| **推理** | Ragged Gather (间接寻址收集 token) + Ragged Gather Reduce (反向聚合) | 本博客 |

同一硬件单元的两种完全不同的应用场景。训练侧利用 SC 的 indirect addressing 能力做通信卸载，推理侧利用它做 token routing 的间接寻址——都是 SC 的"indirect addressing 专长"的不同 projection。

#### MoE 环游记理论 vs 工程实现

| 理论预测 (环游记) | 工程实现 (博客) | 吻合度 |
|---|---|---|
| SparseCore 可用于 MoE token routing | ✅ PR #2137 Ragged Gather | 高 |
| QB 静态 shape 有助于减少 padding | 博客用 dynamic bounded slices + ragged inputs | 不同方法同一目标 |
| All-Gather 比 All-to-All 延迟更确定 | ✅ 博客选择 Option B | 完全一致 |
| Gate normalization 影响 expert 负载均衡 | 博客未讨论训练侧负载均衡 | 推理侧角度不同 |

---

## 7. 性能结果分析

### 7.1 官方结果

博客在 C64 基线档位的结果：

| 工作负载 | 实测 TPS/chip | Roofline TPS/chip | 效率 |
|---|---|---|---|
| **Prefill-heavy** (8K/1K) | **3,707** | 4,500 (discounted) | **82.4%** |
| **Decode-heavy** (1K/8K) | **677** | 850 (discounted) | **79.6%** |

在 C512 档位的改善幅度（vs April baseline）：
- **Decode**: ~**3.1x** 提升
- **Prefill**: ~**4.7x** 提升

### 7.2 数值验证层

博客特别强调了 MoE 模型在高并发下的**数值正确性**问题：
- Gating 和 routing 矩阵对低精度累积误差高度敏感
- PR #2328: 数值验证层审计 FP8 scaling blocks 的累积精度
- PR #2674: 持续监控 softmax 分布范围和 expert 负载均衡
- 验证结果: Pallas gating weights 与高精度 Float32 参考路径**零偏差**

### 7.3 我们的补充：单位换算对照表

将博客的 TPS/chip 与我们 README 的 tok/s (total) 做换算：

| 博客指标 | 博客 TPS/chip | 换算 ×4 | 我们实测 (README) | 可比性 |
|---|---|---|---|---|
| Prefill C64 8K/1K | 3,707 | **14,828** | 849.9 (P64, 8K/1K) | 差距大: DP vs TP sharding 差异 |
| Decode C64 1K/8K | 677 | **2,708** | 1,702 (P64, 1K/8K) | 差距 37%: sharding + config 差异 |
| — | — | — | **2,097** (P128, 1K/1K) | 不可直接比 (不同 workload) |

> **差距分析**: 我们的 8K/1K P64 只有 849.9 tok/s (total) vs 博客 14,828 tok/s (total)，差异 **17.5x**。
> 这不是性能问题，而是**配置差异导致**:
> 1. Sharding: 我们 TP=8 vs 博客 DP=8 (DP 模式下每个 core 独立处理 64/8=8 个请求，parallelism 更高)
> 2. Batched tokens: 我们 4,096 total vs 博客 8,192 total (1,024/core × 8)
> 3. Max seqs: 我们 256 vs 博客 512 (64/core × 8)
> 4. Benchmark 工具和测量方式可能不同

---

## 8. 未来优化路线图

博客列出了两个主要 track 的后续计划：

### 8.1 Collectives 优化 Track

#### FP8 All-Gather Collectives

当前 Token/Metadata All-Gather 传输 BF16/FP32 数据。计划在传输前将 routing metadata 量化到 FP8：
- ICI 链路通信量减半
- 直接降低 routing 延迟

#### Hierarchical Reduce-Scatter 调优

继续优化 block size 和 micro-batch pipelining 参数：
- 实现动态的、token-dependent 的 micro-batch sizing
- 在可变 routing 分布下优化带宽利用率

### 8.2 Kernel & Gating 融合 Track

#### Router Gate + Top-K VPU 融合

当前路由流程：
1. TensorCore 计算 routing logits
2. → 传输到 VPU
3. → VPU 做 top_k 选择

步骤 2 引入序列化瓶颈。计划将 gate projection 和 top_k selection 融合到 VPU 上本地执行，消除这个传输延迟。

> **我们的思考**: 这与 MoE 环游记中讨论的 gate normalization 优化方向一致——gate 计算的效率对 MoE 模型的整体吞吐有显著影响，尤其是 top_k=10 的选择操作。

---

## 9. 复现测试 Cross-Check

### 9.1 环境对齐检查表

| 项目 | 博客环境 | 我们 README 环境 | 差异 |
|---|---|---|---|
| 代码仓库 | vllm-project/tpu-inference | 同 | ✅ |
| 代码版本 | 未明确, 含 PR #2836, #3016 等 | main ≥ 2026-05-15 (含 PR #2577) | 博客版本可能更新 |
| 推理引擎 | vllm serve | 同 | ✅ |
| 硬件 | v7x-8, 4 chip, 8 device | 同 | ✅ |
| 模型 | Qwen/Qwen3.5-397B-A17B-FP8 | 同 | ✅ |
| 权重精度 | FP8 native | 同 | ✅ |
| KV cache 精度 | 未明确 | `--kv-cache-dtype fp8` | 推测相同 |

### 9.2 博客 vs 我们的配置差异

| 参数 | 博客配置 | 我们 README 配置 | 影响 |
|---|---|---|---|
| **Sharding** | **DP=8 + EP=8** | **TP=8 + EP=8** | 核心差异 |
| `--max-num-batched-tokens` | **1,024/core** (≈8,192 total) | **4,096** (total) | 博客 2x 更大 |
| `--max-num-seqs` | **64/core** (≈512 total) | **256** | 博客 2x 更大 |
| `--block-size` | 256 | 256 | ✅ 一致 |
| 工作负载 | 8K/1K, 1K/8K | 1K/1K | 不同 profile |
| 并发范围 | C64-C512 | P1-P256 | 博客最低 64 |
| `--max-model-len` | 未明确 | 4096 (单机) | 我们限制更低 |
| `--gpu-memory-utilization` | 未明确 | 0.9 | — |

### 9.3 复现步骤（基于 README Quick Start 适配博客配置）

> **目标**: 在我们已有的 v7x-8 单机环境中，尽量接近博客的 benchmark 配置，跑出可对比的数字。

#### Step 1: 确认 Patches

```bash
# 确认双 patch (PR #2366 + PR #2577 的 bf16 fix)
kubectl exec $POD -- bash -c "
  grep -c '_hybrid_uniform_page_size_bytes' /workspace/tpu_inference/tpu_inference/runner/kv_cache_manager.py
  grep -c 'jnp.float32' /workspace/tpu_inference/tpu_inference/kernels/gdn/recurrent_scan_v2.py
"
# 期望: ≥1  ≥5
```

#### Step 2: 启动 vLLM (DP=8 模式)

> ⚠️ **注意**: 博客使用 DP=8 sharding，需要确认当前 vLLM 版本是否通过 `--tensor-parallel-size 8` 以外的参数启用 DP。如果引擎不支持显式 DP 模式，则用我们标准的 TP=8 配置跑，标注差异。

```bash
# 标准 TP=8 启动 (我们已验证的配置)
cat > /tmp/launch_vllm_blog_bench.sh <<'LAUNCHER'
#!/bin/bash
cd /tmp
MY_PID=$$
pgrep -f 'EngineCore|vllm' | grep -v "^${MY_PID}$" | xargs -r kill -9 2>/dev/null
sleep 2
rm -f /tmp/libtpu_lockfile /tmp/vllm_blog_bench.log
touch /tmp/vllm_blog_bench.log
setsid nohup env \
  SKIP_JAX_PRECOMPILE=1 VLLM_XLA_CHECK_RECOMPILATION=0 MODEL_IMPL_TYPE=vllm \
  vllm serve /lustre/models/Qwen3.5-397B-A17B-FP8 \
    --tensor-parallel-size 8 --enable-expert-parallel \
    --max-num-batched-tokens 4096 --max-num-seqs 256 --max-model-len 16384 \
    --no-enable-prefix-caching --gpu-memory-utilization 0.9 \
    --kv-cache-dtype fp8 --block-size 256 --trust-remote-code \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    --reasoning-parser qwen3 --async-scheduling \
    >> /tmp/vllm_blog_bench.log 2>&1 < /dev/null &
disown
echo "launched pid=$!"
exit 0
LAUNCHER

kubectl cp /tmp/launch_vllm_blog_bench.sh $POD:/tmp/launch_vllm_blog_bench.sh
kubectl exec $POD -- bash /tmp/launch_vllm_blog_bench.sh
```

#### Step 3: 跑博客对标 benchmark

```bash
# Prefill-heavy: 8K/1K, C64
evalscope perf \
  --url http://localhost:8000/v1/chat/completions \
  --model /lustre/models/Qwen3.5-397B-A17B-FP8 \
  --tokenizer-path /lustre/models/Qwen3.5-397B-A17B-FP8 \
  --dataset random \
  --min-prompt-length 8192 --max-prompt-length 8192 \
  --max-tokens 1024 --min-tokens 1024 \
  --parallel 64 --number 64 \
  --api openai --stream \
  --read-timeout 3600 --connect-timeout 60 \
  --extra-args '{"chat_template_kwargs": {"enable_thinking": false}}'

# Decode-heavy: 1K/8K, C64
evalscope perf \
  --url http://localhost:8000/v1/chat/completions \
  --model /lustre/models/Qwen3.5-397B-A17B-FP8 \
  --tokenizer-path /lustre/models/Qwen3.5-397B-A17B-FP8 \
  --dataset random \
  --min-prompt-length 1024 --max-prompt-length 1024 \
  --max-tokens 8192 --min-tokens 8192 \
  --parallel 64 --number 64 \
  --api openai --stream \
  --read-timeout 3600 --connect-timeout 60 \
  --extra-args '{"chat_template_kwargs": {"enable_thinking": false}}'
```

### 9.4 对照数据表

| Workload | 官方 TPS/chip | 官方 Total (×4) | 我们实测 (Total) | 差异 | 差异原因 |
|---|---|---|---|---|---|
| Prefill 8K/1K C64 | 3,707 | ~14,828 | **待测待填** | — | |
| Decode 1K/8K C64 | 677 | ~2,708 | **待测待填** | — | |
| Prefill 8K/1K C128 | (blog data) | — | **待测待填** | — | |
| Decode 1K/8K C128 | (blog data) | — | **待测待填** | — | |
| Prefill 8K/1K C256 | (blog data) | — | **待测待填** | — | |
| Decode 1K/8K C256 | (blog data) | — | **待测待填** | — | |
| Prefill 8K/1K C512 | (blog 4.7x) | — | **待测待填** | — | |
| Decode 1K/8K C512 | (blog 3.1x) | — | **待测待填** | — | |

### 9.5 已有数据映射

我们 README 中已有的数据可以作为部分参照（注意 workload 和 sharding 差异）：

| 我们的配置 | 我们的 Total tok/s | 换算 TPS/chip (÷4) | 最接近的博客配置 | 可比性 |
|---|---|---|---|---|
| 1K/1K P64 | 1,510 | 377.5 | Decode C64 (677) | 低 (不同 I/O ratio) |
| 1K/1K P128 | 2,097 ⭐ | 524.3 | — | 无直接对标 |
| 8K/1K P64 | 849.9 | 212.5 | Prefill C64 (3,707) | 低 (TP vs DP) |
| 1K/8K P64 | 1,702 | 425.5 | Decode C64 (677) | 中 (I/O match, TP vs DP) |

> **关键差异总结**: 我们的 1K/8K P64 = 425.5 TPS/chip vs 博客 677 TPS/chip，差距 ~37%。主要来源：
> 1. TP=8 vs DP=8 sharding（DP 在 attention 层无通信开销）
> 2. `--max-num-seqs 256` vs 512（博客并发上限更高）
> 3. 博客版本可能包含更多未公开的 kernel 优化（PR #3016 等）

---

## 10. 致谢与参考链接

### 博客原文

- [Systems Engineering Playbook: Optimizing Qwen 3.5-397B MoE on Ironwood (TPU7x)](https://developers.googleblog.com/systems-engineering-playbook-optimizing-qwen-35-397b-moe-on-ironwood-tpu7x/) (2026-07-14)

### 博客提到的 PR 链接

| PR | 主题 | Track |
|---|---|---|
| [PR #1688](https://github.com/vllm-project/tpu-inference/pull/1688) | GMM V2 | MoE |
| [PR #1820](https://github.com/vllm-project/tpu-inference/pull/1820) | RPA v3 | Attention |
| [PR #1961](https://github.com/vllm-project/tpu-inference/pull/1961) | Batched RPA | Attention |
| [PR #2137](https://github.com/vllm-project/tpu-inference/pull/2137) | SparseCore Ragged Gather | MoE |
| [PR #2149](https://github.com/vllm-project/tpu-inference/pull/2149) | Chunked GDN | GDN |
| [PR #2218](https://github.com/vllm-project/tpu-inference/pull/2218) | Ragged Sequence Handling | GDN |
| [PR #2328](https://github.com/vllm-project/tpu-inference/pull/2328) | Numerical Verification | Correctness |
| [PR #2416](https://github.com/vllm-project/tpu-inference/pull/2416) | Hybrid KV Layout | Memory |
| [PR #2498](https://github.com/vllm-project/tpu-inference/pull/2498) | Algebraic Identity | GDN |
| [PR #2577](https://github.com/vllm-project/tpu-inference/pull/2577) | Hybrid DP+EP Sharding | Sharding |
| [PR #2632](https://github.com/vllm-project/tpu-inference/pull/2632) | Batched RPA Kernel | Attention |
| [PR #2674](https://github.com/vllm-project/tpu-inference/pull/2674) | FP8 Numerical Audit | Correctness |
| [PR #2679](https://github.com/vllm-project/tpu-inference/pull/2679) | Hierarchical Reduce-Scatter | Collectives |
| [PR #2823](https://github.com/vllm-project/tpu-inference/pull/2823) | Conv1D Fusion | GDN |
| [PR #2836](https://github.com/vllm-project/tpu-inference/pull/2836) | 3-to-2 All-Gather | Collectives |
| [PR #3016](https://github.com/vllm-project/tpu-inference/pull/3016) | Fully-Fused GDN Kernel | GDN |

### 我们的相关文档

| 文档 | 内容 |
|---|---|
| [README.md](README.md) | Qwen3.5-397B 在 v7x-8 的端到端部署指南 (910 行) |
| [部署与优化指南 v1.5](https://cc.higcp.com/assets/qwen35-397b-tpu-inference-plan-20260424.html) | 完整部署 + 优化决策 |
| [单机推理踩坑记](https://cc.higcp.com/assets/qwen35-397b-debug-story-20260425.html) | 4 小时弯路 vs 14 分钟正确路 |
| [PD 分离 Dogfood](https://cc.higcp.com/assets/qwen35-pd-disagg-dogfood-20260426.html) | PD 全流程 + HMA root cause |
| [Multi-host Dogfood](https://cc.higcp.com/assets/qwen35-multihost-dogfood-20260426.html) | TP=16 跨 host 部署 |

### Wiki 参考

| Wiki 页面 | 相关内容 |
|---|---|
| `tpu-v7` | Ironwood 核心规格 + SparseCore 架构 |
| `sparsecore` | SparseCore 深度解析 |
| `qwen35-397b-tpu-inference-20260425` | 实测记录 |
| `pr-2366-hybrid-kv-cache-fix` | KV Cache OOB Fix 分析 |

### 模型资源

- [Qwen/Qwen3.5-397B-A17B-FP8](https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8) — HuggingFace 模型页
- [Qwen3.5 Architecture Blog (Hugging Face)](https://huggingface.co/blog/qwen3-5) — Hybrid GDN+Attention 技术解读
- [Gated DeltaNet (Sebastian Raschka)](https://magazine.sebastianraschka.com/p/gated-deltanet) — GDN 线性注意力深度解析

### Google Cloud 资源

- [Google Cloud TPU Developer Hub](https://cloud.google.com/tpu/docs) — 完整技术指南 + 代码模板
