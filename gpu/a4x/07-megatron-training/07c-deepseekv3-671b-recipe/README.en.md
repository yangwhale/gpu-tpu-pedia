> 🌐 [中文](README.md) | **English**

# DeepSeek V3 — GB200 NVL72 128-GPU HybridEP Training Reproduction Guide

> **Latest result**: raw Megatron-LM (`pretrain_gpt.py`) peaks at **992 TFLOPs** (peak 1000.5), reaching **76%** of NVIDIA's 256-GPU reference value of 1292. NeMo Bridge (`run_script.py -cv v2`) hits **1124 TFLOPs** on 64 GPUs (DSv3 16L: 1114), but cannot run on 128 GPUs (see §5.10).
>
> 40+ experiment groups, 300 → 992 (raw Megatron-LM + wgrad-defer) / **1124** (NeMo Bridge 64 GPU).
>
> Source: [Maxwell's full report v2](https://doc.maxwell-x.dev/dsv3-hybridep-128g-optimization-v2?t=A5st8MCjDgjk0pj8z_8bPw) (updated 2026-07-07, bot-hallucinated data removed)

## Version Evolution

v1 (MCore 0.16, 975) → v2 (MCore 0.17 dev, 985, requires runtime patch) → v2.1 (patch baked, 956) → v3.1 (MCore 0.18.0, 981) → **v3.1 + wgrad-defer (992, recommended)**

## Core Optimization Path

```
alltoall dispatcher (300) → HybridEP (+58%, 474) → CUDA graph partial capture (+96%, 928)
→ mxfp8 + fp32 optimizer (+105%, 975) → MCore 0.18.0 + graph attn (+109%, 981)
→ wgrad-deferral-limit -1 (+131%, 992)
```

**Key limitation**: capturing a CUDA graph over the full 61-layer model OOMs (184 GB HBM is not enough); it must be shrunk to 32 layers (~221B).

## 1. Best Config Cheat Sheet

### v3.1 + wgrad-defer — 992 TFLOPs/GPU (recommended, MCore 0.18.0)

| Parameter | Value |
|---|---|
| Model | DSv3 reduced to 32L ~221B, H=7168, 256 experts top-8, MLA |
| Parallelism | **PP=2 EP=64** TP=1, seq=8192, MBS=1, GBS=2048 |
| FP8 | mxfp8 e4m3 + **fp8-param-gather** + reuse-grad-buf |
| CUDA Graph | `--cuda-graph-impl transformer_engine --cuda-graph-modules attn` |
| HybridEP | hybridep-num-sms=32, RANKS_PER_DOMAIN=64, USE_MNNVL=1 |
| Optimizer | fp32 main-grads + fp32 main-params, bf16 exp-avg/sq |
| **wgrad** | **`--ddp-average-in-collective --wgrad-deferral-limit -1`** |
| Recompute | selective: moe_act, mlp |
| NCCL | **NVLS=0** GRAPH_REGISTER=0 MNNVL=0 |
| Patch | nvidia-resiliency-ext 0.6.0 + remove `non_blocking` in fused_a2a.py |

### v1 — 975 TFLOPs/GPU (MCore 0.16)

Difference from v3.1: `--cuda-graph-scope attn moe_router moe_preprocess` (includes full MoE graph capture); no patch required.

### v2 — 985 TFLOPs/GPU (MCore 0.17 dev, requires runtime patch)

Difference from v3.1: `--cross-entropy-fusion-impl native`, `--moe-router-padding-for-quantization`. The v2 image crashes on startup and requires a sed patch.

### v2.1 — 956 TFLOPs/GPU (patches baked)

Same as v2 but with patches baked into the Dockerfile, MCore pinned to bfa3326.

### 3 Fatal Parameters

| Parameter | Required Value | Consequence of Wrong Value |
|---|---|---|
| `--cuda-graph-impl transformer_engine` | must be set explicitly | omitted → graph silently disabled (v3.1: 981→836) |
| `NCCL_GRAPH_REGISTER` | 0 | 1 → AssertionError crash |
| `NCCL_NVLS_ENABLE` | 0 | 1 → performance gradually degrades 30-50% after iter 20-40 |

### MCore Version Selection (critical)

| MCore | graph parameter | HybridEP status |
|---|---|---|
| 0.16 (v1) | `--cuda-graph-scope` | ✅ full graph 975 |
| 0.17.0/0.17.1 | `--cuda-graph-scope` | ❌ **hang** (incomplete HybridEP integration) |
| **0.18.0 (v3.1)** | `--cuda-graph-modules` | ✅ attn-only 981 |
| dev bfa3326 (v2/v2.1) | `--cuda-graph-modules` | ✅ attn-only 956-985 |

**v0.17.0/v0.17.1 hang whether graph is on or off** and must be skipped.

### The 2 Patches Required by v2.1/v3.1

1. `nvidia-resiliency-ext` 0.6.0 from GitHub (NGC 26.05 only has 0.5.0, MCore requires ≥0.6.0)
2. Remove `non_blocking=non_blocking,` in `fused_a2a.py` (MCore added this parameter, which DeepEP hybrid-ep does not support)

## 2. Full Experiment Summary

### 2.1 DSv3 61L PP=4 EP=32 (v1)

| # | Config Change | TFLOPs | Notes |
|---|---|---|---|
| 1 | baseline (alltoall, BF16) | 300 | |
| 2 | HybridEP + blockwise + Ring | **432** | +44% |
| 3 | algo=auto | 474 (±30) | +58%, high variance |
| 4 | + NVLS=1 + GRAPH_REG=1 | 488 (peak 518) | **61L best** |
| — | 61L + CUDA graph | OOM | 184 GB not enough |

### 2.2 DSv3 32L PP=2 EP=64 (v1)

| # | Config Change | TFLOPs | Notes |
|---|---|---|---|
| — | baseline (no graph) | 783 | |
| 7 | + CUDA graph (attn+router+preprocess) | **928** | +18.5% |
| 8 | + mxfp8 + fp32 optimizer + fp8-param-gather | **970** | +24% |
| — | **reproduction run** | **975** | **v1 final** |
| 9 | PP=4 EP=32 | 955 | PP=2 is better |

### 2.3 DSv3 32L PP=2 EP=64 (v2/v2.1)

| # | Config Change | TFLOPs | Notes |
|---|---|---|---|
| 10 | baseline no graph (v2) | **935** | |
| 11 | graph attn+router+preprocess (v2) | 365 (-62%) | **v2 MoE graph regression!** |
| 12 | **graph attn only (v2)** | **985** (+5.3%) | **v2 final** |
| — | v2.1 patches baked reproduction | **956** | |

### 2.4 DSv3 32L PP=2 EP=64 (v3.1, MCore 0.18.0)

| # | Config Change | TFLOPs | Notes |
|---|---|---|---|
| E1 | baseline no graph | 836 | |
| **E2** | **graph attn** | **981** (peak 993) | **v3.1 recommended** |
| E4 | remove sequence-parallel | 964 (-17) | SP still helps at TP=1 |
| E5 | recompute mla_up_proj | 977 (±1.5, more stable) | |

### 2.5 v3.1 Parameter Optimization Sweep (07-07, wgrad-defer breakthrough)

| # | Config Change | TFLOPs | Notes |
|---|---|---|---|
| Exp2 | + `--ddp-average-in-collective` | 950.6 | neutral/slightly negative |
| Exp3 | Exp2 + `--delay-wgrad-compute` | CRASH | requires overlap-moe-comm → conflicts with graph attn |
| **Exp4** | **Exp2 + `--wgrad-deferral-limit -1`** | **992** (peak **1000.5**) | **+4.4% ✅ current best** |
| Exp5 | baseline + `--wgrad-deferral-limit -1` | Xid 145 | NVLink HW transient |

`--wgrad-deferral-limit -1` is the only effective additional optimization flag. `--delay-wgrad-compute` cannot be enabled under partial-graph (it requires overlap-moe-comm as a prerequisite, which conflicts with the CUDA graph attn side stream).

### 2.6 Phase 2 Optimization Attempts (v1, 07-06)

| # | Config Change | TFLOPs | Notes |
|---|---|---|---|
| 34 | hybridep-num-sms=16 | 850 (-13%) | insufficient EP bandwidth |
| 35 | hybridep-num-sms=24 | 896 (-8%) | |
| 37 | activation offload | terminated | no iter1 after 15min |
| 40 | optimizer CPU offload (removed fp8-param-gather) | 774 (-21%) | removing fp8-param-gather is fatal |
| 41/42 | delayed FP8 | CRASH | TE 2.9/2.15 various incompatibilities |
| 43 | `--cuda-graph-modules attn moe` | CRASH | assert: only drop-padding supported |

### 2.6 NVLS Regression Investigation (7-group comparison)

| Variable | TFLOPs | Regression? |
|---|---|---|
| NVLS=0 (baseline) | 474 steady state | ❌ no regression |
| NVLS=1 (various combos, 6 groups) | 427-526 → 260-368 | ✅ iter 19-44 |

The regression is internal to the NVLS transport, ruling out slot exhaustion, GC, memory leaks, and thermal throttling.

## 3. Key Findings

### MoE CUDA Graph Deep Dive

**Why v1 uses a full graph**: MCore 0.16 (PR #1917) has no `MoECudaGraphPartialCaptureSignal`; TE's `make_graphed_callables()` wraps an entire layer into a single graph, and scope only declares "what to include" without creating truncation boundaries. HybridEP dispatch is a GPU-native design: fixed launch config (32 SM), the kernel reads the routing tensor from GPU memory, the NVLink buffer is pre-allocated at a fixed size, and there is no CPU-GPU sync.

**Why v2+ truncates**: PR #4292 introduced `MoECudaGraphPartialCaptureSignal`, which uses an exception to truncate graph capture. The reason is general safety — not all dispatchers have a fixed kernel config, and dropless MoE can theoretically produce routing that causes buffer overflow.

**Correctness risk of the v1 full graph**: benchmarking (mock data) is completely safe. In real training, extreme routing skew could theoretically cause NVLink buffer overflow (silent corruption or SIGABRT). DSv3 with 256 experts top-8 has small statistical fluctuation, so the actual risk is extremely low.

### sequence-parallel Still Helps at TP=1

v3.1 E4 experiment: turning off sequence-parallel dropped performance by 17 TFLOPs (964 vs 981). This is contrary to NVIDIA's reference recommendation. The reason is TBD, possibly related to the distributed optimizer's communication pattern.

### Compute Density Determines TFLOPs

| Model | hidden_size | seq | TFLOPs |
|---|---|---|---|
| H=4096 model | 4096 | 4096 | 219 |
| H=4096 model | 4096 | 8192 | 325 |
| **DSv3-32L** | **7168** | **8192** | **981** |

Compute density is determined by hidden_size; H=7168 vs H=4096 differs by about 3x.

## 4. Ruled-Out Directions (complete)

| Direction | Result | Reason |
|---|---|---|
| NCCL_MIN_CTAS=32 | -7% | CTAs occupy SMs |
| numactl | -3% | Grace NUMA latency is low |
| seq > 8192 + offload | OOM | |
| recompute mlp only | -20% | memory pressure |
| optimizer-cuda-graph | CRASH | illegal grad_norm |
| VPP + CUDA graphs | OOM | |
| PP=8 EP=16 | -19% | communication doubles |
| turn off sequence-parallel | -17 TFLOPs | still helps at TP=1 |
| vboost | N/A | not supported on GB200 |
| MCore 0.17.0/0.17.1 | hang | incomplete HybridEP integration |
| delayed FP8 | CRASH | incompatible across versions |
| hybridep sms=16/24 | -13%/-8% | sms=32 is optimal |
| activation offload | terminated | latency stacks up |
| optimizer CPU offload | -21% | removing fp8-param-gather is fatal |
| `attn moe` full graph (v3.1) | CRASH | only drop-padding supported |
| `--cuda-graph-impl local` | CRASH | HybridEP tensor view assert |
| `--ddp-average-in-collective` alone | -3% | neutral/slightly negative |
| `--delay-wgrad-compute` | CRASH | requires overlap-moe-comm → conflicts with graph attn |
| NeMo Bridge full_iteration 128 GPU | CRASH/hang across 9 rounds | PP interleaving sync incompatible with graph capture |

## 5. Breakthrough: NeMo Bridge Unlocks full_iteration graph (981 → 1124)

### 5.1 Discovery Process

Maxwell ran 40+ experiment groups on raw Megatron-LM (`pretrain_gpt.py`), optimizing from 300 to 981. Based on his report, we analyzed the reason for the gap between 981 and NVIDIA's reference value of 1106, and found that the gap comes from the tech stack rather than parameter tuning. By reading the Megatron-Core MoE paper [[1]](#ref1), we identified 4 proprietary techniques in Megatron Bridge, then dumped the NeMo recipe config to verify whether these techniques were available, and ultimately discovered that **`-cv v1` and `-cv v2` are two completely different tech stacks**. After switching to `-cv v2`, 64 GPUs reached 1124 TFLOPs.

### 5.2 Root Cause of raw Megatron-LM Being Capped at 981

raw Megatron-LM lacks 3 key techniques from Bridge:

**1. Sync-Free Device-Initiated Kernels**

Dropless MoE produces a dynamic token count on every routing. The traditional approach: GPU finishes routing → GPU→CPU copy of per-expert token count → CPU decides the launch config (grid size, tile size) for the Grouped GEMM → this device-to-host sync **blocks CUDA Graph capture**.

Bridge's solution: rewrite Grouped GEMM and HybridEP dispatch as device-initiated — the kernel reads shape information from GPU memory itself to decide how to run, with no CPU involvement. The entire MoE layer has **zero CPU-GPU sync** and can be fully captured by a full_iteration graph.

raw Megatron-LM lacks these rewritten kernels → MCore 0.17+ introduced `MoECudaGraphPartialCaptureSignal` to actively truncate graph capture for safety → only attn can be graphed → 981.

**2. Paged Stashing**

A full_iteration graph needs to pre-allocate a fixed buffer for each expert at worst case. The total buffer for model layers × 256 experts × 1.5x margin exceeds 184 GB HBM → OOM.

Bridge's Paged Stashing **dynamically reclaims unused buffer space** during CUDA Graph execution for reuse by other operations, effectively a dynamic memory pool inside a static graph, keeping peak memory controllable.

raw Megatron-LM lacks this → full_iteration graph OOMs when allocating memory.

**3. Flexible PP Layout**

raw Megatron-LM requires `num_layers % (PP × VP) == 0`. DSv3's 61 layers do not divide evenly with PP=8 VP=3. Bridge supports uneven stage splits (e.g. 8+8+8+7+8+8+7+7).

Additionally, the paper describes **ECHO** (dynamically replicating hot experts to idle GPUs to reduce load imbalance) as a supplement to memory optimization.

### 5.3 Key Finding: V1 vs V2 Recipe Config Dump Comparison

By dumping the actual NeMo recipe config inside the container, we found that V1 and V2 are two different tech stacks:

| Config Item | V1 (`-cv v1`) | V2 (`-cv v2`) | Impact |
|---|---|---|---|
| cuda_graph_impl | transformer_engine (TE scoped) | **full_iteration** | **core difference: determines graph coverage** |
| moe_paged_stash | False | **True** | **memory prerequisite that enables the full graph** |
| moe_expert_rank_capacity_factor | — | **1.5** | worst-case buffer pre-allocation multiplier |
| moe_paged_stash_buffer_size_factor_cuda | 1.1 | **1.2** | in-graph buffer reclamation ratio |
| cuda_graph_modules | full (limited by TE impl) | **full** (fully effective under full_iteration) | |
| moe_pad_experts_for_cuda_graph_inference | False | — | |

> V1 is the conservative config (TE scoped graph captures only attn, no paged stash). V2 enables **all** of Bridge's optimizations. The gap is not parameter tuning; it is a tech-stack switch.

Method to dump the config:
```python
from configs.deepseek.deepseek_llm_pretrain import deepseek_v3_pretrain_config_gb200
cfg = deepseek_v3_pretrain_config_gb200(precision="fp8_mx", config_variant="v2")
m = cfg.model
print(m.cuda_graph_impl)          # must be full_iteration
print(m.moe_paged_stash)          # must be True
```

### 5.4 Breakdown of the Improvement

**full_iteration CUDA Graph (+40-50%, largest contribution)**

TE scoped graph captures only the attention module. The MoE layer's router + preprocess + dispatch + expert compute + combine, thousands of kernels, are launched one by one from the host every time. full_iteration records the entire training step (forward + backward + optimizer update) into a single graph; the CPU issues just one replay command, and all MoE kernel launch overhead **goes to zero**.

Memory mechanics of PP + CUDA Graph: with PP, each microbatch must have its own independent graph (otherwise forward overwrites the saved context of backward). Total number of graphs = L × M × 2 (layers × microbatches × forward/backward). Bridge uses buffer reuse to reclaim the buffers of completed microbatches, following the PP execution order, for reuse by the next microbatch.

**Paged Stashing (memory prerequisite that enables the full graph)**

Without paged stash, the full graph needs to pre-allocate buffers at worst case for 94 layers × 128 experts × 1.5x margin, exceeding 184 GB HBM → OOM. Paged stash dynamically reclaims unused space during graph execution for reuse by other operations.

### 5.5 Version and Parameter Requirements

| Component | Minimum Version | Note |
|---|---|---|
| NeMo container | **nemo:26.06** | must use the NeMo container (which includes Megatron Bridge) |
| Megatron Core | **0.18.0+** | first to support `--cuda-graph-modules`. 0.17.x deadlocks with HybridEP and must be skipped |
| Entry script | **`run_script.py`** | `pretrain_gpt.py` goes through raw Megatron-LM, with no Bridge optimizations |
| Config variant | **`-cv v2`** | V1 does not enable full graph / paged stash |

**The entry script determines the tech stack**: with the same NeMo 26.06 container, `run_script.py` goes through Bridge with full optimizations, while `pretrain_gpt.py` goes through raw Megatron-LM with no Bridge optimizations. Choosing the wrong entry point costs **64%**.

### 5.6 Why full graph Couldn't Be Enabled Before but Can Now

| Stage | Belief | Fact |
|---|---|---|
| before | full_iteration + HybridEP + PP>1 = incompatible | **only true for raw Megatron-LM** (no sync-free kernel) |
| before | V1 and V2 differ only in parallelism | **V2 switches the entire CUDA Graph tech stack** |
| before | 685/981 is the hardware limit | **it is the limit of the config choice, not the hardware limit** |
| now | use `run_script.py -cv v2` | Bridge's sync-free kernel + paged stash solve all limitations |

### 5.7 Measured Validation

#### Test 1: MoE Model V2 recipe (1124 TFLOPs)

On a GKE cluster spanning 2 NVL72 domains, 64 GPUs (8+8 nodes):

| Recipe | Graph Mode | Paged Stash | TFLOPs | Step Time | Improvement |
|---|---|---|---|---|---|
| V1 (`-cv v1`) | TE scoped (attn only) | off | ~same as raw Megatron-LM | ~7s | baseline |
| **V2 (`-cv v2`)** | **full_iteration** | **on** | **1124** | **4.31s** | **+64%** |

Steady state **1117-1125 TFLOPs/GPU**, peak **1125.7**. All 20 steps completed and exited normally. **Exceeds NVIDIA's 256-GPU reference value of 1106.**

#### Test 2: DSv3 16L (1114 TFLOPs, 5 rounds of debugging)

Ran a reduced 16-layer DSv3 with NeMo Bridge `run_script.py -m deepseek -mr deepseek_v3`.

**Peculiarity of the DSv3 recipe**: V1 and V2 configs are completely identical (both full_iteration + paged stash); NVIDIA gave the strongest config right away. But the recipe hardcodes the 61-layer PP layout and VPP=4, so changing the layer count requires synchronously changing three parameters.

**Debugging process (5 rounds of pitfalls)**:

| Round | Error | Root Cause | Lesson |
|---|---|---|---|
| v1 | `VPP=4 assert` | recipe defaults to VPP=4; PP=2+16L detects VPP=8 mismatch | VPP must match layer count and PP |
| v2 | `61L layout assert` | `--num_layers 16` changed the layer count but the PP layout is still the hardcoded 61 layers | layout must also be overridden |
| v3 | `VPP=4 assert` | manually set the layout but forgot to synchronously override VPP | layout and VPP must be changed together |
| v4 | `MTP assert` | layout `Etttttttt\|ttttttttL` is missing `m` (the MTP layer) | DSv3 has Multi-Token Prediction |
| **v5** | **success** | PP=2 VPP=2 layout=`Etttt\|tttt\|tttt\|ttttmL` | **change all three together** |

**DSv3 PP layout format**: `E`=embedding, `t`=transformer, `m`=MTP, `L`=loss, `|`=virtual stage boundary. Changing the layer count requires changing all three together: `--num_layers` + `-vp` + `--pipeline_model_parallel_layout`.

**Layout calculation**: 16 layers PP=2 VPP=2 → 4 virtual stages × 4 layers = 16 decoders + 1 MTP + embedding + loss. Layout: `Etttt|tttt|tttt|ttttmL` (16/2/2=4 divides evenly ✓).

**Final command**:
```bash
run_script.py -m deepseek -mr deepseek_v3 --task pretrain \
  -g gb200 -c fp8_mx -ng 64 --data mock --max_steps 20 \
  --num_layers 16 \
  --pipeline_model_parallel_size 2 \
  --expert_model_parallel_size 32 \
  --global_batch_size 512 --micro_batch_size 1 \
  -vp 2 \
  --pipeline_model_parallel_layout "Etttt|tttt|tttt|ttttmL"
```

**Result**:

| Model | Layers | PP | VPP | EP | H | Graph | Paged Stash | TFLOPs | Step Time |
|---|---|---|---|---|---|---|---|---|---|
| **DSv3-16L** | **16** | **2** | **2** | **32** | **7168** | **full_iteration** | **True** | **1114** | **2.35s** |

Steady state 1110-1120, peak 1120.1. Every 5 steps there is a ~713 dip (VPP virtual stage switch communication spike). All 20 steps completed and exited normally.

### 5.7.1 gpu-launchpad-playground Single-Domain Reproduction (2026-07-08)

On the GKE cluster `chrisya-a4x-gke-v2` in the gpu-launchpad-playground project, a 16-node single-domain NVL72 (64 GPUs), NeMo Bridge full_iteration graph reproduction test.

**NCCL_MNNVL_ENABLE comparison**:

| Run | NCCL_MNNVL | Steady TFLOPs | Step Time | Notes |
|---|---|---|---|---|
| Run5 | **2** (auto) | **1176** (peak 1180) | 2.22s | single-domain NVLink full speed |
| Run6 | **0** (off) | **1100** (peak 1103) | 2.38s | simulates cross-domain config |
| baker cross-domain | 0 | 1114 | 2.35s | pool-7+pool-2 measured |

**Key findings**:
- `NCCL_MNNVL_ENABLE=0 → 2` improves 6.5% (1100→1176): NCCL allreduce over MNNVL transport is faster than non-MNNVL
- Run6 (MNNVL=0, 1100) is very close to baker cross-domain (1114), validating the reliability of the baker result
- `USE_MNNVL=1` (HybridEP) is enabled in both runs and is not affected by NCCL_MNNVL
- The single-domain vs cross-domain gap comes mainly from the NCCL MNNVL transport, not RDMA latency

**Raw log — Run5 (MNNVL=2, steady 1176, peak 1180)**:

```
Step Time : 125.37s GPU utilization: 20.9MODEL_TFLOP/s/GPU    # iter 1, JIT warmup
Step Time : 6.13s GPU utilization: 426.6MODEL_TFLOP/s/GPU
Step Time : 4.66s GPU utilization: 561.8MODEL_TFLOP/s/GPU
Step Time : 10.69s GPU utilization: 244.7MODEL_TFLOP/s/GPU   # iter 4, graph capture
Step Time : 2.26s GPU utilization: 1156.2MODEL_TFLOP/s/GPU   # iter 5, steady state begins
Step Time : 3.54s GPU utilization: 738.3MODEL_TFLOP/s/GPU    # VPP spike
Step Time : 2.24s GPU utilization: 1165.6MODEL_TFLOP/s/GPU
Step Time : 2.22s GPU utilization: 1179.6MODEL_TFLOP/s/GPU   # peak
Step Time : 2.22s GPU utilization: 1177.2MODEL_TFLOP/s/GPU
Step Time : 2.22s GPU utilization: 1178.7MODEL_TFLOP/s/GPU
Step Time : 3.47s GPU utilization: 753.3MODEL_TFLOP/s/GPU    # VPP spike
Step Time : 2.23s GPU utilization: 1175.2MODEL_TFLOP/s/GPU
Step Time : 2.23s GPU utilization: 1172.9MODEL_TFLOP/s/GPU
Step Time : 2.22s GPU utilization: 1176.1MODEL_TFLOP/s/GPU
Step Time : 2.22s GPU utilization: 1177.4MODEL_TFLOP/s/GPU
Step Time : 3.45s GPU utilization: 758.7MODEL_TFLOP/s/GPU    # VPP spike
Step Time : 2.22s GPU utilization: 1175.5MODEL_TFLOP/s/GPU
Step Time : 2.23s GPU utilization: 1174.7MODEL_TFLOP/s/GPU
Step Time : 2.23s GPU utilization: 1172.1MODEL_TFLOP/s/GPU
Step Time : 2.23s GPU utilization: 1174.3MODEL_TFLOP/s/GPU   # iter 20
```

**Raw log — Run6 (MNNVL=0, steady 1100, peak 1103)**:

```
Step Time : 127.31s GPU utilization: 20.5MODEL_TFLOP/s/GPU   # iter 1
Step Time : 4.72s GPU utilization: 554.3MODEL_TFLOP/s/GPU
Step Time : 4.03s GPU utilization: 648.3MODEL_TFLOP/s/GPU
Step Time : 8.43s GPU utilization: 310.3MODEL_TFLOP/s/GPU    # iter 4, graph capture
Step Time : 2.41s GPU utilization: 1085.8MODEL_TFLOP/s/GPU   # iter 5
Step Time : 3.62s GPU utilization: 722.0MODEL_TFLOP/s/GPU    # VPP spike
Step Time : 2.40s GPU utilization: 1091.3MODEL_TFLOP/s/GPU
Step Time : 2.38s GPU utilization: 1101.0MODEL_TFLOP/s/GPU
Step Time : 2.37s GPU utilization: 1102.1MODEL_TFLOP/s/GPU
Step Time : 2.38s GPU utilization: 1100.2MODEL_TFLOP/s/GPU
Step Time : 3.61s GPU utilization: 724.2MODEL_TFLOP/s/GPU    # VPP spike
Step Time : 2.38s GPU utilization: 1101.0MODEL_TFLOP/s/GPU
Step Time : 2.39s GPU utilization: 1093.0MODEL_TFLOP/s/GPU
Step Time : 2.37s GPU utilization: 1103.3MODEL_TFLOP/s/GPU   # peak
Step Time : 2.38s GPU utilization: 1098.8MODEL_TFLOP/s/GPU
Step Time : 3.61s GPU utilization: 723.7MODEL_TFLOP/s/GPU    # VPP spike
Step Time : 2.37s GPU utilization: 1102.2MODEL_TFLOP/s/GPU
Step Time : 2.37s GPU utilization: 1101.7MODEL_TFLOP/s/GPU
Step Time : 2.39s GPU utilization: 1096.0MODEL_TFLOP/s/GPU
Step Time : 2.37s GPU utilization: 1101.6MODEL_TFLOP/s/GPU   # iter 20
```

**Impact of missing parameters on performance** (Run4 vs Run5 comparison):

| Parameter | When Missing | After Adding | Impact |
|---|---|---|---|
| `NVTE_CUTEDSL_FUSED_GROUPED_MLP=1` | 2.54s | 2.22s | cuTEDSL fused MoE kernel |
| `NVTE_FWD/BWD_LAYERNORM_SM_MARGIN=16` | — | — | LayerNorm SM reservation |
| `CUDNNFE_CLUSTER_OVERLAP_MARGIN=8` | — | — | cuDNN fusion engine |
| `NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128` | — | — | HybridEP combine chunking |
| `NCCL_CTA_POLICY=1` | — | — | NCCL CTA scheduling |
| LD_LIBRARY_PATH order | host nvidia first | container NCCL first | NCCL version match |

The 7 parameters together improve from ~1030 to ~1176 (+14.2%).

**Cluster info**: GKE `chrisya-a4x-gke-v2`, us-east1-d, forrest-a4x-1x72-policy (subblock-0002), DRA v25.12.0, NCCL RDMA installer

### 5.8 Global Comparison

| Entry | Model | Layers | GPU | Domain | MNNVL | Graph | TFLOPs |
|---|---|---|---|---|---|---|---|
| **run_script.py** | **DSv3-16L** | **16** | **64** | **single-domain** | **2** | **full_iteration** | **1176** |
| run_script.py | DSv3-16L | 16 | 64 | single-domain | 0 | full_iteration | 1100 |
| run_script.py | MoE 94L | 94 | 64 | cross-domain | 0 | full_iteration | 1124 |
| run_script.py | DSv3-16L | 16 | 64 | cross-domain | 0 | full_iteration | 1114 |
| run_script.py | DSv3-32L | 32 | 128 | cross-domain | — | full_iteration | **failed** |
| pretrain_gpt.py | DSv3-32L | 32 | 128 | cross-domain | 0 | TE scoped + wgrad-defer | 992 |
| pretrain_gpt.py | DSv3-32L | 32 | 128 | cross-domain | 0 | TE scoped (v3.1) | 981 |
| NVIDIA ref | DSv3-61L | 61 | 256 | — | — | full_iteration | **1292** |

> The NeMo Bridge full_iteration graph peaks at **1176 TFLOPs** on single-domain 64 GPUs (MNNVL=2), and about **1114** on cross-domain 64 GPUs (MNNVL=0). raw Megatron-LM peaks at 992. NVIDIA's 256-GPU reference value is 1292.

### 5.9 Core Lessons

1. **The recipe config variant is a tech-stack choice**, not just a parallelism config. `-cv v1` → `-cv v2` switches the complete combination of sync-free kernel + paged stash + full_iteration graph.
2. **"full_iteration + HybridEP + PP>1 is incompatible" only holds for raw Megatron-LM**. NeMo Bridge's sync-free kernel solves the CPU-GPU sync problem.
3. **Dumping the config is a necessary diagnostic step**. Without dumping, you cannot know what actually differs under the hood between V1 and V2.
4. **Changing the DSv3 layer count requires changing all three together**: `--num_layers` + `-vp` + `--pipeline_model_parallel_layout` (including the MTP layer).
5. **The entry script determines everything**: `run_script.py` and `pretrain_gpt.py` are two completely different technical paths within the same container.

### 5.10 NeMo Bridge full_iteration graph Is Unavailable at 128 GPUs (v2 report correction)

> ⚠️ **Hallucination correction**: The previously cited "Maxwell v4 report 1349 TFLOPs" has been confirmed to be a bot hallucination; Maxwell himself never achieved that number. The v2 report has removed all hallucinated data. raw Megatron-LM's true best result is **992 TFLOPs** (`--wgrad-deferral-limit -1`).

In the v2 report, Maxwell systematically tested the NeMo 26.06 Bridge full_iteration graph on 128 GPUs (32 nodes) — **all 9 experiment rounds failed**:

| # | Config | Result | Root Cause |
|---|---|---|---|
| 1 | 32L PP=2 VPP=4 | crash | `cudaErrorStreamCaptureUnjoined` |
| 4 | 32L PP=2 VPP=1 | crash | `torch.cuda.synchronize()` in `p2p_communication.py` incompatible with graph capture |
| 5 | 32L PP=2 VPP=8 | hang | graph capture succeeds but after replay the embedding allreduce times out at 604s |
| 6 | 61L PP=8 VPP=2 (official recipe) | crash | `world_size (128) not divisible by expert_tensor_pipeline_parallel_size (512)` |
| 7 | 61L PP=4 VPP=2 EP=32 | hang | 188 GB mem exceeds HBM 184 GB |
| 8 | 32L PP=2 VPP=1 EP=64 | crash | same as #4 |

**Root cause**: The p2p `torch.cuda.synchronize()` in Megatron's `forward_backward_pipelining_with_interleaving` is a 2021-era NCCL race protection (removing it makes loss nan); sync is forbidden during graph capture → **fundamentally incompatible**. Bridge's `layout_map` has only 7 keys, all hardcoded for 61 layers.

**Comparison with our 64-GPU tests**: our 1114/1124 results were run on NeMo Bridge `run_script.py -cv v2` in cross-domain 64-GPU tests. Bridge runs on 64 GPUs but fails on 128 GPUs; a possible reason is that the 64-GPU config's PP communication pattern differs (Bridge may avoid interleaving p2p sync at small scale).

### 5.10.1 DSv3 16L VPP/GBS Cross-Domain Limitations (2026-07-06)

Based on optimization attempts on cross-domain 64 GPUs:

| Round | Change | VPP | GBS | TFLOPs | Result |
|---|---|---|---|---|---|
| baseline | VPP=2 | 2 | 512 | **1114** | ✅ |
| R1 | VPP=8 (1 layer/stage) | 8 | 512 | hang | ❌ PP p2p NCCL timeout |
| R2 | VPP=4 (2 layers/stage) | 4 | 512 | hang | ❌ PP p2p NCCL timeout |
| R3 | VPP=2 + GBS=4096 | 2 | 4096 | hang | ❌ 128 microbatches p2p too frequent cross-domain |

**Conclusion**: 1114 for cross-domain 64 GPUs is the ceiling for VPP=2 + GBS=512. VPP>2 and GBS>2048 hang over cross-domain RDMA.

### 5.11 Reproduction Test: pool-5 RDMA Hardware Issue (2026-07-06)

Attempted to reproduce the DSv3 16L 1114 result using pool-7 + pool-5, and failed. Root cause: the RDMA NIC `mlx5_2:1` on pool-5 nodes reports `async fatal event on QP: local access violation work queue error`. This is an access violation at the RDMA hardware or driver level, not a software config problem.

The pool combination that previously achieved 1114 was pool-7 + pool-2.

**pool-7 + pool-3 also failed**: not an RDMA hardware issue, but HybridEP's `cuMemImportFromShareableHandle: invalid resource handle` — CUDA fabric memory cross-domain import failure. This indicates that the IMEX channel handle between pool-7 and pool-3 is not interoperable.

| Pool Combination | Result | Error |
|--------|------|------|
| pool-7 + pool-2 | ✅ 1114 | — |
| pool-7 + pool-5 | ❌ | mlx5_2 QP access violation (RDMA hardware) |
| pool-7 + pool-3 | ❌ | cuMemImportFromShareableHandle invalid handle (IMEX not connected) |

> **Lesson**: DSv3 full_iteration graph + HybridEP has extremely high demands on cross-domain communication. Different pool combinations have inconsistent RDMA hardware states and IMEX channel compatibility. Cross-domain testing must use a validated pool combination (pool-7 + pool-2). Qwen3 235B's TE scoped graph is more fault-tolerant (the same pool-7+pool-3/pool-5 combinations can run 685 normally).

### 5.12 Future Directions

1. **Resolve Bridge 128-GPU incompatibility**: the root cause is that PP interleaving's `torch.cuda.synchronize()` is incompatible with graph capture. Requires NVIDIA to fix the forward_backward_pipelining code.
2. **NCCL fixing the NVLS regression**: unlocks NVLink SHARP hardware acceleration, expected +3-5%.
3. **Standardize OS tuning**: the v2 report shows that no OS tuning causes a -54% performance drop (962→442). Newly created VMs must use the prod startup script.
4. **Stacking wgrad-defer with other optimizations**: the current 992 is the combination of `--ddp-average-in-collective + --wgrad-deferral-limit -1`. Others such as `--delay-wgrad-compute` could not be enabled due to conflict with graph attn.
5. **MCore open-sourcing sync-free kernels**: if the techniques from Section 4.3.7 of the paper are merged into the open-source version, `pretrain_gpt.py` will also be able to enable the full MoE graph.

## References

<a id="ref1">[1]</a> *Scalable Training of Mixture-of-Experts Models with Megatron Core*, arXiv:2603.07685v2, NVIDIA, 2026. [[arxiv]](https://arxiv.org/abs/2603.07685)
