> 🌐 [中文](README.md) | **English**

# Qwen3 235B-A22B MoE Training on GB200 NVL72 (A4X)

Megatron Bridge + NeMo 26.06 container, Qwen3 235B-A22B MoE pretraining benchmark. 64 GPU (16 nodes).

**Official reference**: DGX-GB200 256 GPU MXFP8 → 7376 tok/s/GPU, **1092 TFLOP/s/GPU** (V2 config). This document uses the 64 GPU V1 config.

**Reference links**:
- [Megatron Bridge Performance Summary](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html) — official benchmark data
- [Megatron Bridge Performance Tuning Guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-guide.html) — performance tuning guide
- [Qwen3 Workload Base Configs](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/scripts/performance/configs/qwen/qwen3_workload_base_configs.py) — recipe parallelism config
- [Qwen3 LLM Pretrain Config](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/scripts/performance/configs/qwen/qwen3_llm_pretrain.py) — recipe model config
- [Performance Scripts README](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/scripts/performance/README.md) — launch script usage

## Model Specifications

| Parameter | Value |
|---|---|
| Model | Qwen3-235B-A22B |
| Total parameters | 235B |
| Activated parameters per token | 22B |
| Number of experts | 128 routed |
| TopK | 8 |
| Number of layers | 94 |

## Prerequisites

- 16 A4X workers (64 GPU), same NVL72 domain (same Placement Policy)
- k8s 1.34+ cluster + GPU Stack (device-plugin + DRA + DRANET + ComputeDomain)
- Worker image: `chrisya-a4x-worker-v3`

## Key Differences from 30B

| Dimension | 30B (07a) | 235B (this doc) |
|---|---|---|
| GPU count | 8 (2 nodes) | 64 (16 nodes) |
| PP | 1 | **8** |
| EP | 8 | 8 |
| TP | 1 | 1 |
| MBS | 4 | 1 (default) |
| GBS | 512 | 1024 |
| CUDA Graph | full_iteration | **transformer_engine** |
| cutedsl | Yes | No (V1 config) |

> **Why PP=8**: The 235B model has 94 layers + 128 experts and cannot fit on a single GPU. PP=8 splits the model into 8 pipeline stages, each stage holding ~12 layers.
>
> **Why CUDA Graph is downgraded**: V1 uses TE CUDA graph (captures only the dense portion), because when PP>1 the full_iteration CUDA graph memory overhead exceeds 10 GiB (verbatim from the official docs). Only V2 (256 GPU) uses full_iteration.

## Step 1: Create a 16-node cluster

```bash
# Create 16 workers with the v3 image (same Placement Policy guarantees same domain)
for i in $(seq 0 15); do
  gcloud compute instances create chrisya-a4x-d2-w${i} \
    --project=$PROJECT --zone=$ZONE \
    --machine-type=a4x-highgpu-4g \
    --provisioning-model=RESERVATION_BOUND \
    --reservation-affinity=specific --reservation=$RESERVATION \
    --maintenance-policy=TERMINATE \
    --resource-policies=$PLACEMENT_POLICY \
    --image=chrisya-a4x-worker-v3 \
    --image-project=$PROJECT \
    --boot-disk-size=500GB --boot-disk-type=hyperdisk-balanced \
    --network-interface=nic-type=GVNIC,network=$GVNIC_NET,subnet=$GVNIC_SUB \
    --network-interface=nic-type=GVNIC,network=$GVNIC_NET_1,subnet=$GVNIC_SUB_1,no-address \
    --network-interface=nic-type=MRDMA,network=$RDMA_NET,subnet=$RDMA_SUB_0,no-address \
    --network-interface=nic-type=MRDMA,network=$RDMA_NET,subnet=$RDMA_SUB_1,no-address \
    --network-interface=nic-type=MRDMA,network=$RDMA_NET,subnet=$RDMA_SUB_2,no-address \
    --network-interface=nic-type=MRDMA,network=$RDMA_NET,subnet=$RDMA_SUB_3,no-address \
    --metadata="ssh-keys=$USER:$(cat ~/.ssh/google_compute_engine.pub)" \
    --scopes=cloud-platform &
done
wait
```

The v3 image is ready to use on boot (~6.5 minutes), with no kernel switch needed. After boot, do kubeadm join + GPU label + ComputeDomain label.

## Step 2: Deploy the NeMo 26.06 training Pods

One Pod per worker, 16 in total. The YAML structure is the same as the 30B recipe (see [07a](../07a-qwen3-30b-recipe/)); change the nodeSelector and pod name.

## Step 3: Environment variables

Identical to 30B:

```bash
source /usr/local/gib/scripts/set_nccl_env.sh

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1
export CUDNNFE_CLUSTER_OVERLAP_MARGIN=8
export NVLINK_DOMAIN_SIZE=72
export USE_MNNVL=1
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=8
export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128
export NCCL_CTA_POLICY=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=0
export NCCL_GRAPH_REGISTER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,graph_capture_record_stream_reuse:True
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
```

## Step 4: Launch training

```bash
cd /opt/Megatron-Bridge/scripts/performance

torchrun --nproc_per_node=4 --nnodes=16 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_IP --master_port=29600 \
  run_script.py \
    -m qwen \
    -mr qwen3_235b_a22b \
    --task pretrain \
    -g gb200 \
    -c fp8_mx \
    -ng 64 \
    --data mock \
    --max_steps 20 \
    --log_dir /tmp/nemo-results \
    -wde bench \
    -wdj qwen3_235b
```

### Config auto-loaded by the recipe (V1, 64 GPU)

| Config | Value | Description |
|---|---|---|
| PP | 8 | 94 layers split into 8 stages |
| EP | 8 | Expert Parallelism |
| TP | 1 | No Tensor parallelism |
| ETP | 1 | No Expert Tensor parallelism |
| GBS | 1024 | Global Batch Size |
| seq_length | 4096 | Sequence length |
| cuda_graph_impl | transformer_engine | TE scoped CUDA Graph |
| cuda_graph_scope | attn, moe_router, moe_preprocess | Dense module capture |
| moe_flex_dispatcher_backend | hybridep | NVL72 optimization |

### Parallelism calculation

- Total GPU: 64
- PP=8: model split into 8 stages
- GPUs per PP stage: 64/8 = 8
- EP=8: 8 GPUs handle expert dispatch
- DP: 64/(TP×PP) = 64/8 = 8, EP=8 → DP_effective=1
- GA: GBS/(MBS×DP_effective×PP_num_micro_batches) → computed by the recipe

## Performance Reference (Megatron Bridge Official)

### Official Benchmark (NeMo 26.06 Container)

| Config | GPU count | Precision | GBS | MBS | TP | PP | CP | VP | EP | tok/s/GPU | TFLOP/s/GPU |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V2 (official release) | 256 | MXFP8 | 8192 | 1 | 1 | 8 | 1 | 3 | 32 | 7376 | 1092 |
| V1 (64 GPU) | 64 | MXFP8 | 1024 | — | 1 | 8 | 1 | — | 8 | — | — |

> V2 uses 256 GPU with VPP=3 + full_iteration CUDA Graph + cutedsl enabled. No official performance data is published for V1 (64 GPU).

### A4X Measured Results

| Config | GPU count | Topology | PP | EP | MNNVL | TFLOP/s/GPU | Step Time | Notes |
|---|---|---|---|---|---|---|---|---|
| V1 baseline | 64 | Single domain | 8 | 8 | 2 | 360 | 27s | Default recipe |
| PP=2 EP=32 | 64 | Single domain | 2 | 32 | 0 | 595 | 8.2s | RDMA only |
| PP=2 EP=32 | 64 | Single domain | 2 | 32 | 2 | **686** | 7.1s | NVLink optimal |
| PP=2 EP=32 cross-domain | 64 | Dual domain (8+8 nodes) | 2 | 32 | 0 | 685 | 7.1s | GKE, V1 recipe, TE scoped graph |
| **PP=2 EP=32 cross-domain V2** | **64** | **Dual domain (8+8 nodes)** | **2** | **32** | **0** | **1124** | **4.31s** | **V2 recipe, full_iteration graph + paged stash** |

> **Cross-domain result surprise**: MNNVL=0 + USE_MNNVL=1 (Maxwell Xi's approach) achieved 685 across two NVL72 domains, nearly matching the single-domain MNNVL=2 result of 686. Reason: PP=2 cross-domain p2p communication volume is small (only activations are transferred), so RDMA 200GB/s is not the bottleneck; the EP=32 all-to-all runs entirely within-domain over HybridEP NVLink and is unaffected by crossing domains.

### GKE Cluster Optimization Iterations (2026-07-05, 8 rounds of experiments)

Starting from the cross-domain 685 baseline, we tested the optimization parameters from Maxwell Xi's DSv3 report one by one:

| Round | What changed | FP8 | TFLOP/s | Notes |
|---|---|---|---|---|
| **baseline** | PP=2 EP=32 MNNVL=0+USE_MNNVL=1 | mxfp8 | **685** | **Optimal config** |
| R2 | + recompute (moe_act,mlp) | mxfp8 | 666 (-3%) | Recomputing activations adds compute |
| R3 | + recompute + NVLS=1 | mxfp8 | OOM | NVLS multicast buffer HBM insufficient |
| R4 | + NCCL_GRAPH_REGISTER=1 | mxfp8 | crash | assert: conflicts with expandable_segments |
| T2 | + seq_length=8192 | mxfp8 | OOM | activations double, exceeding 184GB |
| T3 | + cuDNN fusion (NVTE_FUSED_ATTN=1 etc.) | mxfp8 | 645 (-6%) | Overrides the better implementation the recipe auto-selects |
| T4 | 30B NVLS=1 control | mxfp8 | 926 | 30B 8-GPU NVLS works fine (enough slots) |
| T5 | Switch to fp8_cs (current scaling) | fp8_cs | 701→595 | +2% for the first 11 steps, then -15% degradation |

#### Key Findings

1. **The NeMo recipe already includes fp8-param-gather**: `bf16_with_mxfp8_mixed()` automatically sets `fp8_param_gather=True` + `reuse_grad_buf_for_mxfp8_param_ag=True`. Maxwell Xi's 5% improvement from 928→975 has always been in our recipe.
2. **seq_length is locked by the recipe**: `set_qwen3_common_configs()` hardcodes `cfg.model.seq_length = 4096`; the command-line `-sl` only changes the dataset, not the model. You need to sed patch the Python file to change it.
3. **NVLS OOM is due to insufficient HBM**: It is not NVSwitch multicast slot exhaustion (that is a different scenario in NCCL #2077), but rather the 235B model using 180+GB of HBM, leaving no room for the multicast buffer.
4. **fp8_cs degradation**: per-tensor current scaling shows iter 12+ degradation on Qwen3 235B (mirroring the mxfp8 degradation on DSv3); FP8 degradation behavior is coupled to model architecture.
5. **685 is the H=4096 ceiling**: Compute density (H=4096 vs DSv3 H=7168) differs by 3x, which is the root cause of the low TFLOP/s. Maxwell Xi also measured only 219-325 for Qwen3 235B on the same hardware.

#### Deep Dive on NVLS

| Scenario | NVLS | Result |
|---|---|---|
| 30B 8-GPU (NVLS=1) | ✅ Normal | 926 (vs 924 for NVLS=0, 0.2% diff) |
| 235B 64-GPU (NVLS=1) | ❌ OOM | HBM 180+GB fully used, multicast buffer allocation failed |
| DSv3 61L (Maxwell Xi NVLS=1) | ❌ Degradation | TFLOP/s gradually drops 30-50% after iter 20-40 (time-related bug) |

NVLS has two independent failure modes: (1) GPU HBM OOM when allocating the multicast buffer (our 235B), (2) NVLS transport time-degradation bug (Maxwell Xi's DSv3). Production configs always use NVLS=0.

### Third Optimization Round: Breaking 685 → 1124 (2026-07-06, GKE cluster cross-domain)

#### Key Finding: The Essential Difference Between V1 and V2 Recipes

After dumping the NeMo recipe config, we found that the `-cv v1` and `-cv v2` we had been using are two entirely different tech stacks:

| Config item | V1 (previously used) | V2 (used this time) |
|---|---|---|
| cuda_graph_impl | **transformer_engine** (TE scoped) | **full_iteration** (whole step) |
| cuda_graph_modules | full (but constrained by TE scoped) | full |
| moe_paged_stash | **False** | **True** |
| moe_expert_rank_capacity_factor | — | 1.5 |
| moe_paged_stash_buffer_size_factor_cuda | 1.1 | 1.2 |
| virtual_pipeline_model_parallel_size | None | **3** |
| PP / EP | 8 / 8 (default) | 8 / 32 |

**The V2 recipe is exactly NVIDIA's full 1106 TFLOPs config**, including full_iteration graph + paged stash + VPP. V1 is the conservative config.

#### Why full_iteration graph couldn't be enabled before but can now

**Previous understanding (incorrect)**: full_iteration CUDA Graph is incompatible with HybridEP when PP>1, because dynamic operations on CUDA fabric memory invalidate graph capture. This conclusion came from Maxwell Xi's measurements on raw Megatron-LM (pretrain_gpt.py).

**Correct understanding**: raw Megatron-LM lacks Bridge's 3 proprietary techniques (sync-free device-initiated kernels, ECHO, paged stashing), so full_iteration graph indeed crashes there. But NeMo's run_script.py goes through Megatron Bridge, and Bridge enables these techniques in the V2 recipe:

1. **Sync-Free Device-Initiated Kernels**: Bridge rewrote Grouped GEMM and HybridEP dispatch so kernels autonomously read shape information from GPU memory to decide execution, with no CPU-GPU synchronization. The entire MoE layer has zero host-device sync, so the CUDA Graph can be fully captured.

2. **Paged Stashing** (`moe_paged_stash=True`): Performs fine-grained memory management inside the CUDA Graph; the unused portions of the pre-allocated buffer are dynamically reclaimed for other operations. This solves the memory explosion problem of the full_iteration graph.

3. **Expert Rank Capacity Factor** (`moe_expert_rank_capacity_factor=1.5`): Pre-allocates per-expert buffers at 1.5x the worst case, working with paged stash to reclaim the unused portion.

These techniques are all disabled in the V1 recipe (`moe_paged_stash=False`) and all enabled in the V2 recipe. **The gap is not parameter tuning, it is a tech-stack switch.**

#### The Principle Behind the Improvement

**The 64% improvement from 685 → 1124** comes from three stacked factors:

**1. full_iteration CUDA Graph (+40-50%, largest contribution)**

V1's TE scoped graph only captures the attention module. The MoE layers (router + preprocess + dispatch + expert compute + combine) execute outside the graph, and every operation of every layer must launch kernels one by one from the host. 94 layers × dozens of MoE kernels per layer = thousands of host launch overheads.

The full_iteration graph records the entire training step (forward + backward + optimizer update) into one large graph, and the CPU issues just one replay command. All MoE kernel launch overhead drops to zero. This is why step time went from 7.1s to 4.3s.

**2. Paged Stashing (prerequisite for enabling the full graph)**

Without paged stash, the full_iteration graph must pre-allocate a fixed buffer for each expert at the worst case; the buffers of 94 layers × 128 experts × 1.5x margin would exceed 184 GB HBM → OOM. Paged stash dynamically reclaims unused buffer space during graph execution for reuse by other operations, keeping peak memory manageable.

**3. VPP=3 (V2 recipe default, but overridden by our PP=2)**

The V2 recipe defaults to PP=8 VP=3; after we override to PP=2, VPP is ignored (PP=2 doesn't need VPP since the bubble is already small). So this 1124 comes purely from full graph + paged stash, with no VPP contribution.

#### Measured Results

| Round | recipe | graph | paged_stash | Domain | TFLOPs | step time | Improvement |
|---|---|---|---|---|---|---|---|
| All previous tests | V1 (`-cv v1`) | TE scoped (attn) | False | Cross-domain | **685** | 7.1s | baseline |
| **R9** | **V2 (`-cv v2`)** | **full_iteration** | **True** | Cross-domain | **1124** | **4.31s** | **+64%** |

### gpu-launchpad-playground Single-Domain Reproduction (2026-07-08)

Reproduction test on GKE cluster `chrisya-a4x-gke-v2` (gpu-launchpad-playground, us-east1-d), 16 nodes single-domain NVL72 (64 GPU).

| Round | recipe | PP/EP | Graph | MNNVL | TFLOPs | step time | Notes |
|---|---|---|---|---|---|---|---|
| Single-domain V1 | V1 | PP=8 EP=8 | TE scoped | 0 | **930** (peak 933) | 6.47s | 36% higher than cross-domain 685 |
| Single-domain V2 | V2 | PP=8 EP=8 | full_iteration | 0 | **931** (peak 932) | 6.47s | V2 graph gives no improvement at PP=8 |
| Single-domain V2 | V2 | PP=2 EP=32 | full_iteration | 0 | Testing | — | Benchmarking against baker cross-domain 1124 |
| Single-domain V2 | V2 | PP=2 EP=32 | full_iteration | 2 | Pending | — | MNNVL full speed |

**V1 vs V2 shows no difference at PP=8 EP=8** (930 vs 931): the PP=8 pipeline bubble is the main bottleneck, and the kernel launch overhead that graph optimization addresses is a tiny fraction. The real source of 1124 is the parallelism switch to PP=2 EP=32 — PP=2 has a small bubble (50% → 6%), and EP=32 gives higher MoE communication efficiency. The full_iteration graph only reaches its full potential at PP=2 (fewer PP stages means the graph covers a larger fraction of the computation).

**Analysis of V1 single-domain 930 vs cross-domain 685 gap**:

The improvement from cross-domain 685 → single-domain 930 (+36%) is far larger than the DSv3 16L cross-domain→single-domain gap (1100→1176, +6.9%). Reason: Qwen3 235B uses PP=8 (8 pipeline stages); across domains, PP p2p communication goes over high-latency RDMA, and the p2p round-trip count of 8 stages is 4x that of DSv3's PP=2, amplifying the RDMA latency impact. In single-domain, PP p2p goes entirely over NVLink (μs latency), and PP=8 interleaving is no longer constrained.

**Raw log — V1 PP=8 EP=8 MNNVL=0 (single-domain, steady-state 930)**:

```
Step Time : 81.04s GPU utilization: 74.4MODEL_TFLOP/s/GPU    # iter 1, JIT warmup
Step Time : 20.22s GPU utilization: 298.3MODEL_TFLOP/s/GPU   # iter 2
Step Time : 17.35s GPU utilization: 347.7MODEL_TFLOP/s/GPU   # iter 3
Step Time : 29.28s GPU utilization: 206.0MODEL_TFLOP/s/GPU   # iter 4, graph capture
Step Time : 6.54s GPU utilization: 921.7MODEL_TFLOP/s/GPU    # iter 5, steady-state begins
Step Time : 6.52s GPU utilization: 924.6MODEL_TFLOP/s/GPU
Step Time : 6.52s GPU utilization: 925.6MODEL_TFLOP/s/GPU
Step Time : 6.50s GPU utilization: 927.7MODEL_TFLOP/s/GPU
Step Time : 6.50s GPU utilization: 927.2MODEL_TFLOP/s/GPU
Step Time : 6.49s GPU utilization: 929.2MODEL_TFLOP/s/GPU
Step Time : 6.49s GPU utilization: 929.6MODEL_TFLOP/s/GPU
Step Time : 6.49s GPU utilization: 929.5MODEL_TFLOP/s/GPU
Step Time : 6.49s GPU utilization: 929.4MODEL_TFLOP/s/GPU
Step Time : 6.48s GPU utilization: 931.3MODEL_TFLOP/s/GPU
Step Time : 6.49s GPU utilization: 929.6MODEL_TFLOP/s/GPU
Step Time : 6.48s GPU utilization: 930.9MODEL_TFLOP/s/GPU
Step Time : 6.47s GPU utilization: 932.1MODEL_TFLOP/s/GPU
Step Time : 6.47s GPU utilization: 931.6MODEL_TFLOP/s/GPU
Step Time : 6.47s GPU utilization: 932.0MODEL_TFLOP/s/GPU
Step Time : 6.47s GPU utilization: 932.6MODEL_TFLOP/s/GPU    # iter 20, peak
```

**Raw log — V2 PP=8 EP=8 MNNVL=0 (single-domain, steady-state 931)**:

```
Step Time : 78.94s GPU utilization: 76.4MODEL_TFLOP/s/GPU    # iter 1
Step Time : 19.22s GPU utilization: 313.7MODEL_TFLOP/s/GPU
Step Time : 15.77s GPU utilization: 382.4MODEL_TFLOP/s/GPU
Step Time : 29.63s GPU utilization: 203.5MODEL_TFLOP/s/GPU   # iter 4, graph capture
Step Time : 6.56s GPU utilization: 919.9MODEL_TFLOP/s/GPU    # iter 5
Step Time : 6.52s GPU utilization: 925.5MODEL_TFLOP/s/GPU
Step Time : 6.52s GPU utilization: 925.0MODEL_TFLOP/s/GPU
Step Time : 6.51s GPU utilization: 926.3MODEL_TFLOP/s/GPU
Step Time : 6.51s GPU utilization: 926.0MODEL_TFLOP/s/GPU
Step Time : 6.50s GPU utilization: 928.3MODEL_TFLOP/s/GPU
Step Time : 6.51s GPU utilization: 926.8MODEL_TFLOP/s/GPU
Step Time : 6.50s GPU utilization: 927.6MODEL_TFLOP/s/GPU
Step Time : 6.49s GPU utilization: 928.8MODEL_TFLOP/s/GPU
Step Time : 6.49s GPU utilization: 929.2MODEL_TFLOP/s/GPU
Step Time : 6.48s GPU utilization: 930.7MODEL_TFLOP/s/GPU
Step Time : 6.48s GPU utilization: 930.9MODEL_TFLOP/s/GPU
Step Time : 6.48s GPU utilization: 931.2MODEL_TFLOP/s/GPU
Step Time : 6.47s GPU utilization: 931.7MODEL_TFLOP/s/GPU
Step Time : 6.48s GPU utilization: 931.1MODEL_TFLOP/s/GPU
Step Time : 6.47s GPU utilization: 931.8MODEL_TFLOP/s/GPU    # iter 20
```

All 20 steps completed, zero errors. No VPP spike (PP=8 has no VPP). V1 vs V2 shows almost no difference at PP=8.

**Cluster**: chrisya-a4x-gke-v2, GKE 1.36.0, DRA v25.12.0, NeMo 26.06.rc7, forrest-a4x-1x72-policy (subblock-0002)

**Pending**: V2 recipe (`-cv v2`) single-domain, expected > 1124

> **1124 TFLOPs exceeds NVIDIA's 256 GPU reference of 1106**. The reason may be that our PP=2 EP=32 has a smaller bubble than NVIDIA's PP=8 VP=3, and the communication topology of 64 GPU across 2 domains is simpler than 256 GPU.

#### Why It Wasn't Found Earlier

1. We had always used `-cv v1` (NeMo docs recommend V1 for 64 GPU) and never tried `-cv v2`.
2. We mistakenly assumed V2 only differed in parallelism (PP=8 EP=32 VPP=3), not realizing V2 simultaneously switched the entire CUDA Graph tech stack.
3. Based on Maxwell Xi's raw Megatron-LM experience, we incorrectly inferred "full_iteration + HybridEP + PP>1 is incompatible" — this conclusion only holds for raw Megatron-LM and does not apply to NeMo Bridge.

#### Version Requirements to Obtain These Bridge Features

| Component | Minimum version | Version we use | Notes |
|---|---|---|---|
| NeMo container | **nemo:26.06** | `nvcr.io/nvidia/nemo:26.06` | Must use the NeMo container rather than a bare PyTorch container |
| Megatron Bridge | Bundled in NeMo 26.06 | Same as above | Bridge is NeMo's packaging layer, providing the run_script.py entry point |
| Megatron Core | **0.18.0+** | 0.18.0+d0b3b7754 (bundled in NeMo 26.06.rc7) | 0.18.0 first supports `--cuda-graph-modules`; 0.17.x hangs |
| Entry script | `run_script.py` | `/opt/Megatron-Bridge/scripts/performance/run_script.py` | **Do not use** `pretrain_gpt.py` (raw Megatron-LM, no Bridge optimizations) |
| config variant | **`-cv v2`** | `-cv v2` | V1 does not enable full graph / paged stash; V2 enables all |

**Key distinction**:

- `run_script.py` → goes through Megatron Bridge → has sync-free kernels + paged stash + ECHO → full_iteration graph is safe
- `pretrain_gpt.py` → goes through raw Megatron-LM → no Bridge optimizations → full_iteration + HybridEP + PP>1 = crash

Maxwell Xi used `pretrain_gpt.py`, so he was limited to the TE scoped graph (981). We use `run_script.py -cv v2`, so Bridge's full optimizations take effect, reaching 1124.

**How to verify whether Bridge is in effect**:

```bash
# dump config and check these fields
python3 run_script.py ... --dump_env
# or check directly with python3 inside the container
from configs.qwen.qwen3_llm_pretrain import qwen3_235b_a22b_pretrain_config_gb200
cfg = qwen3_235b_a22b_pretrain_config_gb200(precision="fp8_mx", config_variant="v2")
assert cfg.model.cuda_graph_impl == "full_iteration"   # must be full_iteration
assert cfg.model.moe_paged_stash == True                # must be enabled
```

#### Lessons Learned

**Recipe config variant is not just a parallelism config, it is a tech-stack choice**. V1→V2 is not changing a few numbers, it toggles the full combination of sync-free kernels + paged stash + full_iteration graph. Dumping the config to confirm the actual settings is a necessary diagnostic step.

**The entry script determines the tech stack**. In the same NeMo container, `run_script.py` goes through Bridge with full optimizations, while `pretrain_gpt.py` goes through raw Megatron-LM with no Bridge optimizations. Choosing the wrong entry point costs 64%.

## GKE Deployment (LeaderWorkerSet)

Deploy 64 GPU training on a GKE cluster using LeaderWorkerSet + Kueue + ComputeDomain.

### YAML

```yaml
# 1. ComputeDomain (16 nodes)
apiVersion: resource.nvidia.com/v1beta1
kind: ComputeDomain
metadata:
  name: nemo-235b-domain
spec:
  numNodes: 16
  channel:
    resourceClaimTemplate:
      name: nemo-235b-channel
---
# 2. LeaderWorkerSet (size=16 = 1 leader + 15 workers)
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: nemo-235b
  labels:
    kueue.x-k8s.io/queue-name: tas-lq
spec:
  replicas: 1
  leaderWorkerTemplate:
    size: 16
    restartPolicy: RecreateGroupOnPodRestart
    leaderTemplate:
      metadata:
        annotations:
          networking.gke.io/default-interface: eth0
          networking.gke.io/interfaces: |
            [
              {"interfaceName":"eth0","network":"default"},
              {"interfaceName":"eth1","network":"gvnic-1"},
              {"interfaceName":"gpu0rdma0","network":"rdma-0"},
              {"interfaceName":"gpu1rdma0","network":"rdma-1"},
              {"interfaceName":"gpu2rdma0","network":"rdma-2"},
              {"interfaceName":"gpu3rdma0","network":"rdma-3"}
            ]
        labels:
          app: nemo-235b-pod
      spec:
        nodeSelector:
          cloud.google.com/gke-accelerator: nvidia-gb200
          cloud.google.com/gke-gpu: "true"
        resourceClaims:
        - name: compute-domain-channel
          resourceClaimTemplateName: nemo-235b-channel
        tolerations:
        - key: nvidia.com/gpu
          operator: Exists
        - {effect: NoSchedule, key: kubernetes.io/arch, operator: Equal, value: arm64}
        containers:
        - name: nemo
          image: nvcr.io/nvidia/nemo:26.06
          command: ["/bin/bash", "-c", "sleep infinity"]
          resources:
            claims: [{name: compute-domain-channel}]
            limits: {nvidia.com/gpu: "4"}
            requests: {nvidia.com/gpu: "4"}
          securityContext: {privileged: true}
          volumeMounts:
          - {name: dshm, mountPath: /dev/shm}
          env:
          - {name: CUDA_DEVICE_MAX_CONNECTIONS, value: "1"}
          - {name: NVTE_CUTEDSL_FUSED_GROUPED_MLP, value: "1"}
          - {name: CUDNNFE_CLUSTER_OVERLAP_MARGIN, value: "8"}
          - {name: NVLINK_DOMAIN_SIZE, value: "72"}
          - {name: USE_MNNVL, value: "1"}
          - {name: NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN, value: "32"}
          - {name: NUM_OF_TOKENS_PER_CHUNK_COMBINE_API, value: "128"}
          - {name: NCCL_CTA_POLICY, value: "1"}
          - {name: TORCH_NCCL_AVOID_RECORD_STREAMS, value: "0"}
          - {name: NCCL_GRAPH_REGISTER, value: "0"}
          - {name: PYTORCH_CUDA_ALLOC_CONF, value: "expandable_segments:True,graph_capture_record_stream_reuse:True"}
          - {name: NVTE_FWD_LAYERNORM_SM_MARGIN, value: "16"}
          - {name: NVTE_BWD_LAYERNORM_SM_MARGIN, value: "16"}
          - {name: GLOO_SOCKET_IFNAME, value: eth0}
          - {name: NCCL_SOCKET_IFNAME, value: eth0}
          - {name: NCCL_MNNVL_ENABLE, value: "2"}
          - {name: NCCL_CUMEM_ENABLE, value: "1"}
        volumes:
        - name: dshm
          emptyDir: {medium: Memory, sizeLimit: 200Gi}
    workerTemplate:
      # Same as leaderTemplate (omitted)
```

### Key Differences vs 30B LWS

| Dimension | 30B LWS | 235B LWS |
|---|---|---|
| size | 2 | 16 |
| ComputeDomain numNodes | 2 | 16 |
| NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN | 8 | 32 |
| nodeSelector node pool | Specify on demand | Needs a pool with 16 idle nodes, or leave unspecified and let Kueue TAS auto-select |

### Launch Training (PP=2 EP=32 optimized version)

```bash
# Execute on all 16 Pods respectively (batch with a script)
for i in $(seq 0 15); do
  POD="nemo-235b-0"
  [ $i -gt 0 ] && POD="nemo-235b-0-${i}"
  kubectl exec $POD -- bash -c "
    cd /opt/Megatron-Bridge/scripts/performance
    export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/usr/local/nvidia/lib64:\$LD_LIBRARY_PATH
    export GLOO_SOCKET_IFNAME=eth0 NCCL_SOCKET_IFNAME=eth0
    export NCCL_MNNVL_ENABLE=2 NCCL_CUMEM_ENABLE=1
    nohup torchrun --nproc_per_node=4 --nnodes=16 --node_rank=$i \
      --master_addr=<LEADER_IP> --master_port=29600 \
      run_script.py -m qwen -mr qwen3_235b_a22b --task pretrain \
      -g gb200 -c fp8_mx -ng 64 --data mock \
      --max_steps 20 --log_dir /tmp/nemo-results \
      -wde bench -wdj qwen3_235b \
      -cv v1 \
      --pipeline_model_parallel_size 2 \
      --expert_model_parallel_size 32 \
      --global_batch_size 512 \
      --micro_batch_size 1 > /tmp/train.log 2>&1 &
  " &
done
wait
```

### Cross-Domain Deployment (measured approach, GKE cluster Domain A + Domain B)

When spanning two NVL72 domains, use two LWS + two ComputeDomains, each locked to a different node pool:

```yaml
# Domain A (PP stage 0, node_rank 0-7)
ComputeDomain: nemo-235b-domain-a (numNodes=8, channel=nemo-235b-channel-a)
LWS: nemo-235b-a (size=8, nodePool=Domain A)

# Domain B (PP stage 1, node_rank 8-15)
ComputeDomain: nemo-235b-domain-b (numNodes=8, channel=nemo-235b-channel-b)
LWS: nemo-235b-b (size=8, nodePool=Domain B)
```

Key config differences vs single-domain:
- `NCCL_MNNVL_ENABLE=0` — NCCL uses RDMA to avoid cross-domain hangs
- `USE_MNNVL=1` — HybridEP within-domain NVLink fabric
- `NCCL_NVLS_ENABLE=0` — the 235B model is too large, NVLS multicast buffer OOM

### GKE Pitfall Log

| Problem | Cause | Fix |
|---|---|---|
| NVLS multicast OOM | Large 235B parameter count, multicast buffer allocation exceeds HBM | `NCCL_NVLS_ENABLE=0` |
| Empty master_addr | Shell variable not expanded inside kubectl exec single quotes | Write the IP directly, don't use `$VAR` |
| MNNVL available but not working | Bare Pod has no IMEX channel | Must use LWS + ComputeDomain + ResourceClaim |
| ncclWaitSignal undefined | GIB NCCL 2.28 vs container NCCL 2.30 | Put the container path first in LD_LIBRARY_PATH |
| Gloo IPv6 unreachable | Gloo defaults to IPv6 | `GLOO_SOCKET_IFNAME=eth0` |
| nvcr.io 403 | NGC API key lacks permissions | Use the project AR image |

## Notes

1. **All 16 nodes must be in the same NVL72 domain**: create them with the same Placement Policy. An A4X NVL72 domain holds at most 18 nodes, so 16 nodes is within range.
2. **PP bubble**: PP=8 incurs pipeline bubble overhead. The V1 config does not enable VPP (Virtual Pipeline Parallelism). V2 enables VP=3 to reduce the bubble.
3. **Memory**: 235B is 8x larger than 30B; after PP=8 sharding, each GPU holds ~30B parameters. V1 uses TE CUDA graph (memory-friendly) and does not need recompute.
4. **CUDA Graph mode difference**: V1 uses `transformer_engine` (lightweight), V2 uses `full_iteration` (needs more memory but performs better).
5. **config_variant**: `run_script.py` loads the V2 config by default. If running on 64 GPU, confirm whether it auto-downgrades to V1. You may need `--config_variant v1` or `-cv v1`.
