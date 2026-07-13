> 🌐 [中文](parameter-guide.md) | **English**

# Complete Guide to Megatron Bridge Training Parameters

All key parameters for MoE model training on GB200 NVL72 (A4X), including inter-parameter dependencies and their impact on memory / compute / communication.

Based on hands-on experience with Qwen3 30B (8 GPUs) and 235B (64 GPUs).

## Parameter Overview

| Parameter | Category | Saves memory | Adds compute | Affects communication | Key dependencies |
|---|---|---|---|---|---|
| PP | Parallelism | Yes (splits layers) | No | PP p2p | PP×TP×DP = total GPUs; EP is orthogonal within the DP group |
| EP | Parallelism | Yes (splits experts) | No | all-to-all | EP×DP ≤ world_size/PP/TP |
| TP | Parallelism | Yes (splits weights) | No | 2×all-reduce/layer | TP must stay within NVLink |
| VPP | Parallelism optimization | No | No | Increases number of p2p ops | num_layers must be divisible by PP×VP |
| MBS | Batching | No (increases) | No | No | Larger MBS → larger activations |
| GBS | Batching | No | No | Increases GA steps | GBS = MBS × DP × GA |
| full_iteration CG | CUDA Graph | No (adds ~10-60G) | No | No | Compatible with HybridEP when PP=1; incompatible when PP>1 |
| TE scoped CG | CUDA Graph | Slight increase | No | No | Safely compatible with HybridEP |
| recompute | Memory optimization | Yes (substantial) | Yes (recomputation) | No | Only supports full_iteration CG |
| fp8_mx | Precision | Slight saving | No (reduces) | No | Requires GB200+ hardware |
| cutedsl | Kernel fusion | No | No (fewer launches) | No | NVTE_CUTEDSL_FUSED_GROUPED_MLP=1 |
| HybridEP | Communication | No (adds buffers) | No | EP over NVLink | USE_MNNVL=1, requires IMEX |
| NCCL MNNVL | Communication | No | No | Global NVLink | Set 2 within a single domain; 64+ GPUs across domains may hang |
| NVLS | Communication | No | No | NVLink SHARP | +3% throughput |
| GRAPH_REGISTER | Communication | No | No | NCCL registration optimization | Conflicts with expandable_segments |

---

## 1. Parallelism Strategy Parameters

### 1.1 PP (Pipeline Model Parallel Size)

**Purpose**: Splits the model into PP segments by layer, placing each segment on a different GPU group.

**Constraint**: `num_layers` must be divisible by PP (otherwise it errors out or distributes unevenly).

**Relationship with DP/EP**: `DP = total GPUs / (PP × TP)`. EP is not part of the DP computation; EP works orthogonally within the DP group — the same group of GPUs does DP on dense layers and EP all-to-all on MoE layers.

**Impact on memory**:
- Larger PP → fewer layers per GPU → less memory
- 30B (48 layers): PP=1 → 48 layers/GPU; PP=8 → 6 layers/GPU
- 235B (94 layers): PP=8 → ~12 layers/GPU; PP=2 → 47 layers/GPU

**Impact on compute**:
- PP introduces a **pipeline bubble**: bubble fraction ≈ (PP-1) / (PP-1+GA)
- PP=8 GA=16 → bubble 7/23 = 30%
- PP=2 GA=64 → bubble 1/65 = 1.5%
- The larger the PP, the larger the bubble, and the lower the MFU

**Impact on communication**:
- Activations are transferred between PP stages via **point-to-point (p2p)**
- Typically goes over RDMA (across NVLink domains) or NVLink (within a domain)
- PP p2p data volume = MBS × seq_len × hidden_dim × 2 bytes

**Dependencies on other parameters**:
- `PP × EP × TP × DP = total number of GPUs`
- When PP>1, **full_iteration CUDA Graph is incompatible with HybridEP** (confirmed empirically)
- `VPP` requires num_layers to be divisible by PP×VP
- PP determines the size of the EP group: the GPUs within each PP stage form one EP group

**Measured comparison**:

| Model | PP | EP | TFLOP/s | Notes |
|---|---|---|---|---|
| 30B | 1 | 8 | 925 | PP=1, no bubble |
| 235B | 8 | 8 | 360 | Large bubble |
| 235B | 2 | 32 | 595 | Small bubble, +63% |

### 1.2 EP (Expert Model Parallel Size)

**Purpose**: Splits the experts of MoE layers across the GPUs within the EP group. Each GPU holds `num_experts / EP` experts.

**Impact on memory**:
- Larger EP → fewer experts per GPU → less expert weight memory
- 235B has 128 experts: EP=8 → 16 experts/GPU; EP=32 → 4 experts/GPU
- EP=32 vs EP=8 saves ~12 experts' worth of weights per GPU ≈ several GB

**Impact on communication**:
- Each MoE layer requires **2 all-to-all ops** (dispatch + combine)
- Larger EP → more GPUs participating in the all-to-all → greater communication volume
- However, HybridEP performs all-to-all within the NVLink domain, where bandwidth far exceeds RDMA
- The EP group must be within the same NVLink domain to use HybridEP

**Dependencies on other parameters**:
- `NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN` must equal the number of ranks in the EP group that reside within the same NVLink domain
- DP = world_size / (PP × TP); EP is orthogonal within the DP group
- EP=32 PP=2 TP=1 → DP=32; each DP group has 32 GPUs and does EP=32 all-to-all on MoE layers
- EP ≤ DP (EP cannot exceed the DP group size)
- The EP group must be within an NVLink domain to use HybridEP

### 1.3 TP (Tensor Model Parallel Size)

**Purpose**: Splits the weight matrices of attention and FFN column-wise/row-wise across the GPUs within the TP group.

**Impact on memory**:
- Larger TP → smaller dense weights per GPU
- However, TP does not apply to MoE experts (experts are handled by EP)

**Impact on communication**:
- Each layer requires **2 all-reduce ops** (forward + backward)
- TP communication is latency-sensitive and must stay within NVLink (latency <1μs)
- TP=4 on NVLink: ~100 GB/s effective bandwidth

**Why TP=1 on A4X**:
- A4X has only 4 GPUs per node, with 72 GPUs in an NVLink domain
- The bottleneck of MoE models is expert communication (EP), not the dense part (TP)
- TP=1 eliminates the TP all-reduce overhead

### 1.4 VPP (Virtual Pipeline Parallel Size)

**Purpose**: Further splits the layers of each PP stage into VP virtual segments, interleaving execution to reduce the bubble.

**Impact on the bubble**:
- Without VPP: bubble = (PP-1) / (PP-1+GA)
- With VPP: bubble ≈ (PP-1) / (PP-1+GA×VP)
- VP=3 PP=8 → bubble reduced by 3×

**Constraints**:
- `num_layers` must be divisible by `PP × VP`
- 235B 94 layers: 94/(8×3)=3.9 is not an integer → **PP=8 cannot use VPP**
- This is one of the important reasons why 64-GPU PP=8 performs poorly

**Impact on communication**:
- VPP increases the number of p2p communication ops (by a factor of VP), but the data volume per op stays the same
- Requires overlap-moe-expert-parallel-comm to hide the extra p2p

### 1.5 MBS (Micro Batch Size)

**Purpose**: The number of samples each GPU processes per micro step.

**Impact on memory**:
- Larger MBS → activation memory grows linearly
- MBS=4 has 4× the activations of MBS=1
- 30B PP=1: MBS=4 fits (168 GiB); 235B PP=2: MBS=1 fits, MBS=2 OOMs

**Impact on compute**:
- Larger MBS → larger batch dimension for matmuls → higher GPU utilization (closer to the roofline)
- But too large an MBS may cause uneven MoE routing (expert load imbalance)

**Impact on GA**:
- `GA = GBS / (MBS × DP × PP_num_microbatches)`
- Larger MBS → smaller GA → less gradient accumulation per step → the pipeline bubble fraction may increase

### 1.6 GBS (Global Batch Size)

**Purpose**: The global number of samples per training step.

**Impact on convergence**:
- Larger GBS → more accurate gradient estimates → allows a larger learning rate
- But too large a GBS may require more warmup steps

**Impact on communication**:
- GBS determines the number of GA steps
- Larger GA → lower gradient allreduce frequency → smaller communication share
- But GA does not affect MFU (GA merely accumulates gradients and adds no effective compute)

---

## 2. CUDA Graph Parameters

### 2.1 cuda_graph_impl

| Value | Description | Memory overhead | Compatibility |
|---|---|---|---|
| none | No CUDA Graph | 0 | Fully compatible |
| transformer_engine | TE scoped, captures only specified modules | Slight increase | Compatible with HybridEP |
| local | Local CUDA Graph, used together with scope | Depends on scope | Depends on scope |

### 2.2 cuda_graph_scope

| Value | Capture range | Performance | HybridEP compatibility |
|---|---|---|---|
| full_iteration | The entire training step | Highest (30B: 925) | Compatible when PP=1; incompatible when PP>1 |
| attn | Captures only attention | Medium | Compatible |
| moe_router | Captures only the MoE router | Medium | Compatible |
| moe_preprocess | Captures only MoE preprocess | Medium | Compatible |

**Key dependencies**:
- `full_iteration` + `HybridEP` + `PP>1` → **`cudaErrorStreamCaptureInvalidated`**
- Root cause: HybridEP uses CUDA fabric memory for cross-GPU memory sharing, and these operations get invalidated during stream capture across pipeline stages
- `recompute` only supports the `full_iteration` scope

### 2.3 Impact of CUDA Graph on Memory

- CUDA Graph requires pre-allocating a **graph private memory pool**
- The pool for full_iteration can reach **40-60 GiB** (at the 235B scale)
- Must be used together with `TORCH_NCCL_AVOID_RECORD_STREAMS=0`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,graph_capture_record_stream_reuse:True`
- `NCCL_GRAPH_REGISTER` must be 0 (conflicts with expandable_segments)

---

## 3. Precision Parameters

### 3.1 fp8_mx (MXFP8)

**Purpose**: Performs matmuls in MXFP8 precision (GEMMs for both forward and backward).

**Impact on compute**:
- GB200 FP8 Tensor Core: ~2× BF16 throughput
- But the precision loss requires additional scaling factor computation
- Net effect: ~40-60% faster than BF16

**Impact on memory**:
- Weights are still stored in BF16 (master copy)
- Activations can be cached in FP8, a slight saving
- Optimizer state is always FP32

**Differences between FP8 recipes**:

| Recipe | Description | Stability |
|---|---|---|
| blockwise | Quantization per block, adaptive scaling | Stable, recommended |
| mxfp8 | Microscaling FP8, more aggressive quantization | Performance degrades under PP=4 |
| e4m3 | 4-bit exponent + 3-bit mantissa | Standard format |

### 3.2 cutedsl_fused_grouped_mlp

**Purpose**: Uses CuTeDSL to fuse the grouped GEMM of MoE, merging the matmuls of multiple experts into a single kernel launch.

**How to set**: environment variable `NVTE_CUTEDSL_FUSED_GROUPED_MLP=1`

**Impact on compute**:
- Reduces the number of kernel launches (128 experts × 48 layers = 6144 launches → merged into ~100)
- 30B measured: cutedsl OFF→ON, 89→284 TFLOP/s (3.2×)

**Impact on memory**: None

**Dependency**: Used together with `CUDNNFE_CLUSTER_OVERLAP_MARGIN=8`

### 3.3 fp8_dot_product_attention

**Purpose**: The QK dot product of attention is also computed in FP8.

**Impact on compute**: A slight improvement (attention accounts for a relatively small share in MoE models)

**How to set**: A recipe configuration item, not a CLI parameter

---

## 4. Communication Parameters

### 4.1 HybridEP-related

| Environment variable | Value | Description |
|---|---|---|
| USE_MNNVL | 1 | HybridEP uses NVLink fabric memory (independent of NCCL) |
| NVLINK_DOMAIN_SIZE | 72 | The NVL72 physical domain size |
| NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN | 8/32 | The number of ranks participating in EP per domain, must match the actual EP group size |
| NUM_OF_TOKENS_PER_CHUNK_COMBINE_API | 128 | The chunk size for the HybridEP combine |

**Key understanding**: `USE_MNNVL` controls HybridEP; `NCCL_MNNVL_ENABLE` controls the NCCL transport. The two are independent.

**Choosing NCCL_MNNVL_ENABLE**:
- `=2`: NCCL uses the NVLink transport (900 GB/s). **All GPUs within a single domain should be set to 2**.
- `=0`: NCCL falls back to RDMA (400 Gbps per NIC). Use only as a workaround when a **cross-domain 64+ GPU allreduce hangs**.
- Setting 0 previously was a mistaken conservative decision. All 16 of our nodes are within a single NVL72 domain, with no cross-domain traffic, so it should be set to 2.

### 4.2 NCCL-related

| Environment variable | Value | Purpose | Impact |
|---|---|---|---|
| NCCL_MNNVL_ENABLE | 0/2 | Whether NCCL uses the NVLink transport | Set 2 for a single domain (NVLink 900GB/s); set 0 to fall back to RDMA for cross-domain 64+ GPUs |
| NCCL_CUMEM_ENABLE | 1 | NCCL uses CUDA unified memory | Must be on |
| NCCL_NVLS_ENABLE | 1 | NVLink SHARP (hardware-accelerated broadcast/allreduce) | +3% throughput |
| NCCL_GRAPH_REGISTER | 0 | NCCL graph registration optimization | Must be 0 (conflicts with expandable_segments) |
| NCCL_PXN_C2C | 1 | PXN C2C relay | Cross-node NVLink routing optimization |
| NCCL_CTA_POLICY | 1 | GB200-specific CTA policy | Must be set |
| NCCL_SOCKET_IFNAME | eth0,eth1 | NCCL bootstrap network interfaces | GKE uses eth0/eth1 |

### 4.3 TORCH / PyTorch-related

| Environment variable | Value | Purpose | Used with |
|---|---|---|---|
| TORCH_NCCL_AVOID_RECORD_STREAMS | 0 | **Must be 0 in CUDA Graph mode** | full_iteration CG |
| PYTORCH_CUDA_ALLOC_CONF | expandable_segments:True,graph_capture_record_stream_reuse:True | Memory management | CUDA Graph |
| CUDA_DEVICE_MAX_CONNECTIONS | 1 | Limits CUDA stream concurrency | Megatron overlap |

### 4.4 TE / cuDNN-related

| Environment variable | Value | Purpose |
|---|---|---|
| NVTE_CUTEDSL_FUSED_GROUPED_MLP | 1 | CuTeDSL fused MoE GEMM |
| CUDNNFE_CLUSTER_OVERLAP_MARGIN | 8 | cuDNN cluster overlap margin |
| NVTE_FWD_LAYERNORM_SM_MARGIN | 16 | SMs reserved for LayerNorm (forward) |
| NVTE_BWD_LAYERNORM_SM_MARGIN | 16 | SMs reserved for LayerNorm (backward) |

---

## 5. Inter-Parameter Dependency Diagram

### Combinations that must go together

```
full_iteration CG
  → TORCH_NCCL_AVOID_RECORD_STREAMS=0
  → NCCL_GRAPH_REGISTER=0
  → PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,graph_capture_record_stream_reuse:True
  → PP=1 (if using HybridEP)

HybridEP
  → USE_MNNVL=1
  → NVLINK_DOMAIN_SIZE=72
  → NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN = EP group size
  → requires IMEX daemon / ComputeDomain
  → moe_flex_dispatcher_backend=hybridep

VPP
  → num_layers must be divisible by PP × VP
  → requires overlap-moe-expert-parallel-comm (when PP>1)

recompute
  → only supports full_iteration CUDA Graph
  → unavailable under TE scoped CG
```

### Mutually exclusive relationships

```
NCCL_GRAPH_REGISTER=1 × expandable_segments=True → AssertionError
full_iteration CG × HybridEP × PP>1 → cudaErrorStreamCaptureInvalidated
TORCH_NCCL_AVOID_RECORD_STREAMS=1 × CUDA Graph → Graph silently disabled (no error, just slow)
NCCL_MNNVL_ENABLE=2 × 64+ GPUs across 2+ IMEX domains → allreduce hang (NCCL #2077, unaffected within a single domain)
```

### Memory-saving checklist

| Method | How much it saves | Cost |
|---|---|---|
| Increase PP | ~linear (multiple of PP) | pipeline bubble grows |
| Increase EP | expert weights/EP | all-to-all communication grows |
| Lower MBS | activations/MBS × | GPU utilization drops |
| recompute | activations halved | compute increases ~33%; only supports full_iteration CG |
| fp8_mx | activations slightly reduced | precision loss |
| Lower GBS | no saving (GBS only affects the number of GA steps) | — |

### Added-compute checklist

| Method | Extra compute | What you get |
|---|---|---|
| recompute | +33% (recomputing activations) | saves activation memory |
| cutedsl OFF→ON | 0 (pure kernel fusion) | reduces launch overhead |
| full_iteration CG | 0 (pure record-and-replay) | eliminates host overhead |
| fp8_mx ON | -40% (hardware acceleration) | speedup |

### Parameters affecting the choice of communication

| Parameter | Communication affected | What to choose |
|---|---|---|
| PP | PP p2p | RDMA or NVLink |
| EP + HybridEP | EP all-to-all | NVLink fabric (HybridEP) or NCCL RDMA |
| TP | TP all-reduce | Must be NVLink |
| NCCL_MNNVL_ENABLE | Global collectives | NVLink transport or RDMA |
| NVLS | broadcast/allreduce | NVLink SHARP hardware acceleration |
| moe_a2a_overlap | EP communication | Overlap with compute |

---

## 6. Parameter Tuning Decision Tree

```
Given: number of model layers L, number of experts E, total number of GPUs N, HBM capacity H

1. Choosing PP:
   If all layers fit on a single GPU → PP=1 (optimal, can enable full_iteration CG)
   Otherwise → the smallest PP such that the parameters + activations of L/PP layers fit in H

2. Choosing EP:
   EP = N / PP (use up all GPUs)
   Check: is E divisible by EP? Do the E/EP experts per GPU fit in H?
   Check: is the EP group within the NVLink domain? (EP ≤ NVLINK_DOMAIN_SIZE/4×PP)

3. CUDA Graph:
   If PP=1 → full_iteration (optimal)
   If PP>1 → transformer_engine scoped (safe)

4. VPP:
   If L is divisible by PP×VP → enable VPP to reduce the bubble
   Otherwise → do not enable

5. MBS:
   Try from large to small: 4→2→1, until it fits in H
   The larger the MBS, the higher the GPU utilization

6. Precision:
   fp8_mx > fp8_cs > bf16 (performance ranking)
   blockwise > mxfp8 (stability ranking, when PP≤4)
```
