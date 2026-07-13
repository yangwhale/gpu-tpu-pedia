> 🌐 **中文** | [English](parameter-guide.en.md)

# Megatron Bridge 训练参数完全指南

GB200 NVL72 (A4X) 上 MoE 模型训练的所有关键参数，包括参数间依赖关系和内存/计算/通信影响。

基于 Qwen3 30B (8 GPU) 和 235B (64 GPU) 的实测经验。

## 参数总览

| 参数 | 类别 | 省内存 | 增计算 | 影响通信 | 关键依赖 |
|---|---|---|---|---|---|
| PP | 并行 | 是(切层) | 否 | PP p2p | PP×TP×DP = 总GPU，EP 在 DP group 内正交 |
| EP | 并行 | 是(切expert) | 否 | all-to-all | EP×DP ≤ world_size/PP/TP |
| TP | 并行 | 是(切权重) | 否 | 2×all-reduce/层 | TP 内必须 NVLink |
| VPP | 并行优化 | 否 | 否 | 增加 p2p 次数 | num_layers 必须整除 PP×VP |
| MBS | 批量 | 否(增) | 否 | 否 | MBS 增大→activation 增大 |
| GBS | 批量 | 否 | 否 | 增加 GA 步数 | GBS = MBS × DP × GA |
| full_iteration CG | CUDA Graph | 否(增~10-60G) | 否 | 否 | PP=1 时兼容 HybridEP; PP>1 不兼容 |
| TE scoped CG | CUDA Graph | 微增 | 否 | 否 | 安全兼容 HybridEP |
| recompute | 内存优化 | 是(大幅) | 是(重算) | 否 | 只支持 full_iteration CG |
| fp8_mx | 精度 | 微省 | 否(减) | 否 | 需 GB200+ 硬件 |
| cutedsl | Kernel 融合 | 否 | 否(减launch) | 否 | NVTE_CUTEDSL_FUSED_GROUPED_MLP=1 |
| HybridEP | 通信 | 否(增buffer) | 否 | EP走NVLink | USE_MNNVL=1, 需 IMEX |
| NCCL MNNVL | 通信 | 否 | 否 | 全局 NVLink | 单域内设 2; 跨域 64+ GPU 可能 hang |
| NVLS | 通信 | 否 | 否 | NVLink SHARP | +3% throughput |
| GRAPH_REGISTER | 通信 | 否 | 否 | NCCL 注册优化 | 与 expandable_segments 冲突 |

---

## 一、并行策略参数

### 1.1 PP (Pipeline Model Parallel Size)

**作用**：把模型按层切成 PP 段，每段放在不同 GPU 组上。

**约束**：`num_layers` 必须能被 PP 整除（否则报错或不均匀分配）。

**与 DP/EP 的关系**：`DP = 总GPU / (PP × TP)`。EP 不参与 DP 的计算，EP 在 DP group 内部正交工作——同一组 GPU 在 dense 层做 DP，在 MoE 层做 EP all-to-all。

**对内存的影响**：
- PP 越大 → 每卡放的层数越少 → 内存越小
- 30B (48层): PP=1 → 48层/卡; PP=8 → 6层/卡
- 235B (94层): PP=8 → ~12层/卡; PP=2 → 47层/卡

**对计算的影响**：
- PP 引入 **pipeline bubble**：bubble 比例 ≈ (PP-1) / (PP-1+GA)
- PP=8 GA=16 → bubble 7/23 = 30%
- PP=2 GA=64 → bubble 1/65 = 1.5%
- PP 越大 bubble 越大，MFU 越低

**对通信的影响**：
- PP stage 之间通过 **点对点 (p2p)** 传输 activation
- 通常走 RDMA（跨 NVLink 域）或 NVLink（域内）
- PP p2p 数据量 = MBS × seq_len × hidden_dim × 2 bytes

**与其他参数的依赖**：
- `PP × EP × TP × DP = 总 GPU 数`
- PP>1 时 **full_iteration CUDA Graph 与 HybridEP 不兼容**（实测确认）
- `VPP` 需要 num_layers 整除 PP×VP
- PP 决定了 EP group 的大小：每个 PP stage 内的 GPU 组成一个 EP group

**实测对比**：

| 模型 | PP | EP | TFLOP/s | 说明 |
|---|---|---|---|---|
| 30B | 1 | 8 | 925 | PP=1 无 bubble |
| 235B | 8 | 8 | 360 | 大 bubble |
| 235B | 2 | 32 | 595 | 小 bubble, +63% |

### 1.2 EP (Expert Model Parallel Size)

**作用**：把 MoE 层的 expert 切分到 EP 组内的 GPU 上。每卡持 `num_experts / EP` 个 expert。

**对内存的影响**：
- EP 越大 → 每卡 expert 越少 → expert 权重内存越小
- 235B 有 128 expert: EP=8 → 16 expert/卡; EP=32 → 4 expert/卡
- EP=32 比 EP=8 每卡省 ~12 个 expert 的权重 ≈ 数 GB

**对通信的影响**：
- 每个 MoE 层需要 **2 次 all-to-all**（dispatch + combine）
- EP 越大 → all-to-all 参与 GPU 数越多 → 通信量越大
- 但 HybridEP 在 NVLink 域内做 all-to-all，带宽远超 RDMA
- EP group 必须在同一个 NVLink 域内才能用 HybridEP

**与其他参数的依赖**：
- `NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN` 必须等于 EP group 中在同一 NVLink 域内的 rank 数
- DP = world_size / (PP × TP)，EP 在 DP group 内正交
- EP=32 PP=2 TP=1 → DP=32, 每个 DP group 32 GPU，MoE 层做 EP=32 all-to-all
- EP ≤ DP（EP 不能大于 DP group size）
- EP group 必须在 NVLink 域内才能用 HybridEP

### 1.3 TP (Tensor Model Parallel Size)

**作用**：把 attention 和 FFN 的权重矩阵按列/行切分到 TP 组内的 GPU 上。

**对内存的影响**：
- TP 越大 → 每卡的 dense 权重越小
- 但 TP 对 MoE expert 不生效（expert 由 EP 管）

**对通信的影响**：
- 每层需要 **2 次 all-reduce**（forward + backward）
- TP 通信延迟敏感，必须在 NVLink 内（延迟 <1μs）
- TP=4 on NVLink: ~100 GB/s 有效带宽

**为什么 A4X 上 TP=1**：
- A4X 每节点只有 4 GPU，NVLink 域有 72 GPU
- MoE 模型的瓶颈在 expert 通信（EP），不在 dense 部分（TP）
- TP=1 消除了 TP all-reduce 开销

### 1.4 VPP (Virtual Pipeline Parallel Size)

**作用**：把每个 PP stage 的层再切成 VP 个虚拟段，交错执行减少 bubble。

**对 bubble 的影响**：
- 无 VPP: bubble = (PP-1) / (PP-1+GA)
- 有 VPP: bubble ≈ (PP-1) / (PP-1+GA×VP)
- VP=3 PP=8 → bubble 降 3 倍

**约束**：
- `num_layers` 必须整除 `PP × VP`
- 235B 94 层: 94/(8×3)=3.9 不整除 → **PP=8 无法用 VPP**
- 这是 64 GPU PP=8 性能低的重要原因之一

**对通信的影响**：
- VPP 增加 p2p 通信次数（VP 倍），但每次数据量不变
- 需要 overlap-moe-expert-parallel-comm 配合才能掩盖额外 p2p

### 1.5 MBS (Micro Batch Size)

**作用**：每个 GPU 每个 micro step 处理的 sample 数。

**对内存的影响**：
- MBS 越大 → activation 内存线性增长
- MBS=4 比 MBS=1 activation 大 4 倍
- 30B PP=1: MBS=4 fit (168 GiB); 235B PP=2: MBS=1 fit, MBS=2 OOM

**对计算的影响**：
- MBS 越大 → matmul 的 batch 维度越大 → GPU 利用率越高（更接近 roofline）
- 但 MBS 太大可能让 MoE routing 不均匀（expert load imbalance）

**对 GA 的影响**：
- `GA = GBS / (MBS × DP × PP_num_microbatches)`
- MBS 增大 → GA 减小 → 每步 gradient accumulation 更少 → pipeline bubble 比例可能增大

### 1.6 GBS (Global Batch Size)

**作用**：每个 training step 的全局 sample 数。

**对收敛的影响**：
- GBS 越大 → 梯度估计越准 → 可以用更大 learning rate
- 但 GBS 太大可能需要更多 warmup steps

**对通信的影响**：
- GBS 决定了 GA 步数
- GA 越大 → gradient allreduce 频率越低 → 通信占比越小
- 但 GA 不影响 MFU（GA 只是累积梯度，不增加有效计算）

---

## 二、CUDA Graph 参数

### 2.1 cuda_graph_impl

| 值 | 说明 | 内存开销 | 兼容性 |
|---|---|---|---|
| none | 不用 CUDA Graph | 0 | 全兼容 |
| transformer_engine | TE scoped，只 capture 指定模块 | 微增 | 兼容 HybridEP |
| local | 本地 CUDA Graph，配合 scope 使用 | 取决于 scope | 取决于 scope |

### 2.2 cuda_graph_scope

| 值 | capture 范围 | 性能 | HybridEP 兼容 |
|---|---|---|---|
| full_iteration | 整个 training step | 最高 (30B: 925) | PP=1 兼容; PP>1 不兼容 |
| attn | 只 capture attention | 中 | 兼容 |
| moe_router | 只 capture MoE router | 中 | 兼容 |
| moe_preprocess | 只 capture MoE preprocess | 中 | 兼容 |

**关键依赖**：
- `full_iteration` + `HybridEP` + `PP>1` → **`cudaErrorStreamCaptureInvalidated`**
- 根因：HybridEP 使用 CUDA fabric memory 做跨 GPU 内存共享，这些操作在跨 pipeline stage 的 stream capture 时 invalidate
- `recompute` 只支持 `full_iteration` scope

### 2.3 CUDA Graph 对内存的影响

- CUDA Graph 需要预分配 **graph private memory pool**
- full_iteration 的 pool 可达 **40-60 GiB**（235B 级别）
- 必须与 `TORCH_NCCL_AVOID_RECORD_STREAMS=0` 配合
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,graph_capture_record_stream_reuse:True`
- `NCCL_GRAPH_REGISTER` 必须为 0（与 expandable_segments 冲突）

---

## 三、精度参数

### 3.1 fp8_mx (MXFP8)

**作用**：用 MXFP8 精度做 matmul（forward + backward 的 GEMM）。

**对计算的影响**：
- GB200 FP8 Tensor Core: ~2× BF16 吞吐
- 但精度损失需要额外的 scaling factor 计算
- 净效果：比 BF16 快 ~40-60%

**对内存的影响**：
- 权重存储仍是 BF16（master copy）
- Activation 可以用 FP8 缓存，微省
- 优化器状态始终 FP32

**不同 FP8 recipe 的区别**：

| Recipe | 说明 | 稳定性 |
|---|---|---|
| blockwise | 按 block 做 quantization，自适应 scaling | 稳定，推荐 |
| mxfp8 | microscaling FP8，更激进的量化 | PP=4 下性能退化 |
| e4m3 | 4 bit exponent + 3 bit mantissa | 标准格式 |

### 3.2 cutedsl_fused_grouped_mlp

**作用**：用 CuTeDSL 融合 MoE 的 grouped GEMM。把多个 expert 的 matmul 合并成一次 kernel launch。

**设置方式**：环境变量 `NVTE_CUTEDSL_FUSED_GROUPED_MLP=1`

**对计算的影响**：
- 减少 kernel launch 数量（128 expert × 48 层 = 6144 次 → 合并为 ~100 次）
- 30B 实测：cutedsl OFF→ON，89→284 TFLOP/s（3.2×）

**对内存的影响**：无

**依赖**：`CUDNNFE_CLUSTER_OVERLAP_MARGIN=8` 配合使用

### 3.3 fp8_dot_product_attention

**作用**：attention 的 QK dot product 也用 FP8 计算。

**对计算的影响**：微幅提升（attention 在 MoE 模型中占比较小）

**设置方式**：recipe 配置项，不是 CLI 参数

---

## 四、通信参数

### 4.1 HybridEP 相关

| 环境变量 | 值 | 说明 |
|---|---|---|
| USE_MNNVL | 1 | HybridEP 使用 NVLink fabric memory（独立于 NCCL） |
| NVLINK_DOMAIN_SIZE | 72 | NVL72 物理域大小 |
| NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN | 8/32 | 每域参与 EP 的 rank 数，必须匹配实际 EP group 大小 |
| NUM_OF_TOKENS_PER_CHUNK_COMBINE_API | 128 | HybridEP combine 的 chunk 大小 |

**关键理解**：`USE_MNNVL` 控制 HybridEP，`NCCL_MNNVL_ENABLE` 控制 NCCL transport。两者独立。

**NCCL_MNNVL_ENABLE 选择**：
- `=2`：NCCL 走 NVLink transport（900 GB/s）。**单域内所有 GPU 都应设 2**。
- `=0`：NCCL 退化为 RDMA（400 Gbps per NIC）。仅在**跨域 64+ GPU allreduce hang** 时作为 workaround。
- 之前设 0 是错误的保守决策。我们 16 节点全在一个 NVL72 域内，没有跨域，应该设 2。

### 4.2 NCCL 相关

| 环境变量 | 值 | 作用 | 影响 |
|---|---|---|---|
| NCCL_MNNVL_ENABLE | 0/2 | NCCL 是否用 NVLink transport | 单域设 2（NVLink 900GB/s）; 跨域 64+ GPU 时设 0 退化为 RDMA |
| NCCL_CUMEM_ENABLE | 1 | NCCL 使用 CUDA unified memory | 必须开 |
| NCCL_NVLS_ENABLE | 1 | NVLink SHARP（硬件加速 broadcast/allreduce） | +3% throughput |
| NCCL_GRAPH_REGISTER | 0 | NCCL graph 注册优化 | 必须为 0（与 expandable_segments 冲突） |
| NCCL_PXN_C2C | 1 | PXN C2C relay | 跨节点 NVLink 路由优化 |
| NCCL_CTA_POLICY | 1 | GB200 特定的 CTA 策略 | 必须设 |
| NCCL_SOCKET_IFNAME | eth0,eth1 | NCCL bootstrap 网卡 | GKE 用 eth0/eth1 |

### 4.3 TORCH / PyTorch 相关

| 环境变量 | 值 | 作用 | 与谁配合 |
|---|---|---|---|
| TORCH_NCCL_AVOID_RECORD_STREAMS | 0 | **CUDA Graph 模式必须为 0** | full_iteration CG |
| PYTORCH_CUDA_ALLOC_CONF | expandable_segments:True,graph_capture_record_stream_reuse:True | 内存管理 | CUDA Graph |
| CUDA_DEVICE_MAX_CONNECTIONS | 1 | 限制 CUDA stream 并发 | Megatron overlap |

### 4.4 TE / cuDNN 相关

| 环境变量 | 值 | 作用 |
|---|---|---|
| NVTE_CUTEDSL_FUSED_GROUPED_MLP | 1 | CuTeDSL 融合 MoE GEMM |
| CUDNNFE_CLUSTER_OVERLAP_MARGIN | 8 | cuDNN cluster overlap margin |
| NVTE_FWD_LAYERNORM_SM_MARGIN | 16 | LayerNorm 预留 SM 数 (forward) |
| NVTE_BWD_LAYERNORM_SM_MARGIN | 16 | LayerNorm 预留 SM 数 (backward) |

---

## 五、参数间依赖关系图

### 必须配合的组合

```
full_iteration CG
  → TORCH_NCCL_AVOID_RECORD_STREAMS=0
  → NCCL_GRAPH_REGISTER=0
  → PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,graph_capture_record_stream_reuse:True
  → PP=1（如果用 HybridEP）

HybridEP
  → USE_MNNVL=1
  → NVLINK_DOMAIN_SIZE=72
  → NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN = EP group size
  → 需要 IMEX daemon / ComputeDomain
  → moe_flex_dispatcher_backend=hybridep

VPP
  → num_layers 必须整除 PP × VP
  → 需要 overlap-moe-expert-parallel-comm（PP>1 时）

recompute
  → 只支持 full_iteration CUDA Graph
  → TE scoped CG 下不可用
```

### 互斥关系

```
NCCL_GRAPH_REGISTER=1 × expandable_segments=True → AssertionError
full_iteration CG × HybridEP × PP>1 → cudaErrorStreamCaptureInvalidated
TORCH_NCCL_AVOID_RECORD_STREAMS=1 × CUDA Graph → Graph 静默失效（不报错，只是慢）
NCCL_MNNVL_ENABLE=2 × 跨 2+ IMEX 域 64+ GPU → allreduce hang (NCCL #2077, 单域内不受影响)
```

### 省内存清单

| 方法 | 省多少 | 代价 |
|---|---|---|
| 增大 PP | ~线性（PP 倍数） | pipeline bubble 增大 |
| 增大 EP | expert 权重/EP | all-to-all 通信增大 |
| 降低 MBS | activation/MBS 倍 | GPU 利用率降低 |
| recompute | activation 减半 | 计算量增 ~33%; 只支持 full_iteration CG |
| fp8_mx | activation 微省 | 精度损失 |
| 降低 GBS | 不省（GBS 只影响 GA 步数） | — |

### 增加计算清单

| 方法 | 额外计算 | 换来什么 |
|---|---|---|
| recompute | +33% (重算 activation) | 省 activation 内存 |
| cutedsl OFF→ON | 0（纯 kernel 融合） | 减少 launch overhead |
| full_iteration CG | 0（纯录制重放） | 消除 host overhead |
| fp8_mx ON | -40% (硬件加速) | 速度提升 |

### 影响通信选择的参数

| 参数 | 影响的通信 | 选择什么 |
|---|---|---|
| PP | PP p2p | RDMA 或 NVLink |
| EP + HybridEP | EP all-to-all | NVLink fabric (HybridEP) 或 NCCL RDMA |
| TP | TP all-reduce | 必须 NVLink |
| NCCL_MNNVL_ENABLE | 全局 collective | NVLink transport 或 RDMA |
| NVLS | broadcast/allreduce | NVLink SHARP 硬件加速 |
| moe_a2a_overlap | EP 通信 | 与计算 overlap |

---

## 六、参数调优决策树

```
给定: 模型层数 L, expert 数 E, 总 GPU 数 N, HBM 容量 H

1. PP 选择:
   如果单卡放得下全部层 → PP=1 (最优, 可以开 full_iteration CG)
   否则 → 最小的 PP 使得 L/PP 层的参数 + activation fit H
   
2. EP 选择:
   EP = N / PP (用完所有 GPU)
   检查: E 能被 EP 整除? 每卡 E/EP 个 expert fit H?
   检查: EP group 在 NVLink 域内? (EP ≤ NVLINK_DOMAIN_SIZE/4×PP)

3. CUDA Graph:
   如果 PP=1 → full_iteration (最优)
   如果 PP>1 → transformer_engine scoped (安全)

4. VPP:
   如果 L 整除 PP×VP → 开 VPP 减少 bubble
   否则 → 不开

5. MBS:
   从大到小试: 4→2→1, 直到 fit H
   MBS 越大 GPU 利用率越高

6. 精度:
   fp8_mx > fp8_cs > bf16 (性能排序)
   blockwise > mxfp8 (稳定性排序, PP≤4 时)
```
