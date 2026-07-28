> 🌐 [中文](README.md) | **English**

# Tencent Hunyuan Hy3 (295B-A21B) 64-GPU Training — GB300 NVL72 (A4X Max) GKE

Preparation document for a 16-node, 64-GPU training benchmark of **Hy3 (Hunyuan 3, 295B total / 21B activated / 80 layers + 1 MTP)** on a GB300 (A4X Max) GKE cluster.

> **Current status: run in progress** (started 2026-07-25 22:30 HKT). Live logs are in **§9**; the configuration design is in §1–§3.
>
> Peers for comparison: DeepSeek V3 671B in [`07e-gb300-deepseekv3-671b-gke/`](../07e-gb300-deepseekv3-671b-gke/) (already at ~1658 TFLOP/s, 99.3% of official), Qwen3 235B in [`07d-gb300-qwen3-235b-gke/`](../07d-gb300-qwen3-235b-gke/).

**References**:
- Model card: [huggingface.co/tencent/Hy3](https://huggingface.co/tencent/Hy3) · Apache 2.0
- Official training doc: [`finetune/README.md`](https://huggingface.co/tencent/Hy3/blob/main/finetune/README.md)
- Megatron-Bridge model bridge: [`models/hy_v3/hy_v3_bridge.py`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/models/hy_v3/hy_v3_bridge.py)
- [Megatron Bridge Performance Summary](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html)

---

## Bottom line first: the state of Hy3 training support

Three things have to be settled before you run anything, or you will follow the DSV3 playbook straight into a dead end.

| Question | Answer | Consequence |
|------|------|------|
| Does Tencent ship a Megatron recipe? | **No.** Only three **SFT** stacks: DeepSpeed / LLaMA-Factory / ms-swift | Nothing official to lean on; the Megatron side has to be written from scratch |
| Does Megatron-Bridge support Hy3? | **The model layer does, the perf recipe does not.** `HYV3Bridge` is registered (`HYV3ForCausalLM` → `GPTModel`) with complete bidirectional HF↔Megatron weight mapping (MTP layer included); but there is **no** hunyuan/hy_v3 directory under `recipes/` | Weight conversion works out of the box; parallelism and performance config are yours to build |
| Can the DSV3 recipe be copied? | **The MoE part yes, the attention part no** | Hy3's MoE is DSV3 lineage (below), but its attention is GQA, not MLA |

### Why the MoE can be copied from DSV3

Hy3's MoE is a **one-to-one port of the DeepSeek V3 recipe** — even the config field names share an ancestry:

| DeepSeek V3 design | V3 field | Hy3 field | Qwen3 |
|---|---|---|---|
| sigmoid routing | `scoring_func: sigmoid` | `moe_router_use_sigmoid: true` | ❌ softmax |
| aux-loss-free balancing (per-expert bias) | `topk_method: noaux_tc` | `moe_router_enable_expert_bias: true` | ❌ |
| shared expert | `n_shared_experts: 1` | `num_shared_experts: 1` | ❌ |
| MTP | `num_nextn_predict_layers: 1` | `num_nextn_predict_layers: 1` | ❌ |
| leading dense layers | `first_k_dense_replace: 3` | `first_k_dense_replace: 1` | ❌ |
| routed scaling | `routed_scaling_factor: 2.5` | `router_scaling_factor: 2.826` | ❌ |

**What this means in practice**: the DSV3 recipe's MoE knobs (hybridep backend, `moe_a2a_overlap`, `cutedsl_fused_grouped_mlp`, `moe_paged_stash`, the EP sweet spot) **transfer directly**; everything attention-related (MLA's `q_lora_rank` / `kv_lora_rank` / `qk_rope_head_dim`) is **void**, replaced by GQA's `num_query_groups=8` + `kv_channels=128` + `qk_layernorm=True`.

---

## 1. Model spec and Megatron parameter mapping

HF `config.json` (`tencent/Hy3`) → Megatron `GPTModelProvider`. The mapping is taken from the `HYV3Bridge.provider_bridge()` source, **not guessed**.

### 1.1 Structural parameters (direct mapping)

| Megatron provider | Value | HF config source |
|---|---|---|
| `num_layers` | **80** | `num_hidden_layers` |
| `hidden_size` | **4096** | `hidden_size` |
| `ffn_hidden_size` | **13312** | `intermediate_size` (dense layer only) |
| `num_attention_heads` | **64** | `num_attention_heads` |
| `num_query_groups` | **8** | `num_key_value_heads` (GQA) |
| `kv_channels` | **128** | `head_dim` |
| `seq_length` | 4096 (benchmark) / 262144 (full) | `max_position_embeddings` |
| `vocab_size` | **120832** | `vocab_size` |
| `rotary_base` | **11158840.0** | `rope_parameters.rope_theta` |
| `normalization` | `RMSNorm` | fixed |
| `gated_linear_unit` | `True` | `hidden_act: silu` |
| `add_bias_linear` / `add_qkv_bias` | `False` / `False` | Hy3 has no QKV bias |
| `qk_layernorm` | **`True`** | `qk_norm: true` |
| `untie_embeddings_and_output_weights` | `True` | `tie_word_embeddings: false` |

### 1.2 MoE parameters (DSV3 lineage)

| Megatron provider | Value | HF config source |
|---|---|---|
| `num_moe_experts` | **192** | `num_experts` |
| `moe_router_topk` | **8** | `num_experts_per_tok` |
| `moe_ffn_hidden_size` | **1536** | `moe_intermediate_size` |
| `moe_shared_expert_intermediate_size` | **1536** | `moe_intermediate_size × num_shared_experts` (1536×1) |
| `moe_layer_freq` | `[0]*1 + [1]*79` | `first_k_dense_replace=1` → layer 0 dense, layers 1–79 MoE |
| `moe_router_score_function` | **`sigmoid`** | `moe_router_use_sigmoid` |
| `moe_router_enable_expert_bias` | **`True`** | `moe_router_enable_expert_bias` |
| `moe_router_pre_softmax` | `False` | fixed |
| `moe_router_topk_scaling_factor` | **2.826** | `router_scaling_factor` |
| `moe_router_dtype` | `fp32` | fixed (routing numerical stability) |
| `moe_grouped_gemm` | `True` | fixed |
| `moe_permute_fusion` | `True` | fixed |
| `mtp_num_layers` | **1** | `num_nextn_predict_layers` |

### 1.3 ⚠️ Three bridge defaults that training must override

`HYV3Bridge`'s defaults are meant for **weight conversion / inference**. Taking them straight into **from-scratch pretraining will unbalance the experts**:

| provider field | Bridge default | **Training should set** | Why |
|---|---|---|---|
| `moe_router_bias_update_rate` | `0` | **`1e-3`** | At 0 the expert bias never updates and the aux-loss-free balancing mechanism is **inert** (the DeepSeek V3 paper uses 0.001) |
| `moe_router_load_balancing_type` | `"none"` | `"none"` (keep) + the bias update above | The aux-loss-free route *is* `none` + bias; **do not** switch to `aux_loss`, it fights with sigmoid+bias |
| `moe_aux_loss_coeff` | `0.0` | `0.0` (keep), optionally `seq_aux_loss_coeff=1e-4` as a backstop | V3 uses a tiny sequence-level aux loss as insurance |
| `moe_token_dispatcher_type` | `"alltoall"` | **`"flex"` + `moe_flex_dispatcher_backend="hybridep"`** | On GB300 NVL72, hybridep clearly beats naive alltoall (see 07e) |

> **Fine-tuning (SFT) is the exception**: when loading official weights for SFT the expert routing is already trained, so leaving `bias_update_rate` at 0 is safer (it avoids perturbing converged routing). **Only from-scratch / continued-pretrain needs it on.**

### 1.4 Parameter-count check (does the config add up?)

| Component | Calculation | Params |
|---|---|---|
| Routed experts | 79 layers × 192 experts × 3 × 4096 × 1536 | **286.2 B** |
| Shared expert | 79 layers × 3 × 4096 × 1536 | 1.49 B |
| Attention | 80 layers × (Q 4096×8192 + K/V 2×4096×1024 + O 8192×4096) | 6.04 B |
| Dense FFN (layer 0) | 3 × 4096 × 13312 | 0.16 B |
| Embedding + LM head | 2 × 120832 × 4096 (untied) | 0.99 B |
| **Total** | | **≈ 294.9 B** ✓ matches the official 295B |

> **97% of the parameters live in the experts** → **EP is the only memory knob that matters**. TP is nearly useless for this model (attention is only 2%).

---

## 2. Parallelism strategy for 64 GPUs (16 nodes)

### 2.1 Topology: a single NVLink domain

One GB300 NVL72 domain holds ≤ 18 nodes. **16 nodes = 64 GPUs fits inside one subblock**, far simpler than 07e's 4-domain 256-GPU setup:

```
64 GPU = 1 domain × 16 nodes × 4 GPU
┌──────────────────────────────────┐
│  subblock A  (ComputeDomain yw-cd-a) │
│  16 nodes × 4 GPU = 64 GPU           │
│  all NVLink (MNNVL) connected        │
│  → EP can go up to 64 (whole domain) │
└──────────────────────────────────┘
```

**Upside**: no cross-domain RDMA, so EP all-to-all rides entirely on NVLink; only one ComputeDomain is needed, which sidesteps the pile of 4-CD clique deadlock traps documented in 07e.

### 2.2 Parallelism constraints

`world = TP × PP × DP`, and **EP must divide DP** (when TP=1):

| TP | PP | DP = 64/(TP×PP) | EP options | Experts per rank (192/EP) |
|---|---|---|---|---|
| 1 | 2 | 32 | 8 / 16 / **32** | 6 |
| 1 | 4 | 16 | 8 / **16** | 12 |
| 1 | 8 | 8 | **8** | 24 |

> Measured in 07e: **EP=32 is the sweet spot**; EP=64 is actually 1.3% lower (longer all-to-all spans + too few experts per rank). At EP=8 each rank carries 24 experts and memory gets tight.

### 2.3 Starting configuration (recommended V1)

| Parameter | Value | Rationale |
|---|---|---|
| TP | **1** | Attention is only 2% of params; TP is pure communication loss |
| PP | **2** | 80 layers / 2 = 40 per stage; smaller PP means smaller bubble, and 288 GB fits it |
| VPP | **8** | 16 chunks, thinning the bubble (matching DSV3's PP2×VPP8) |
| EP | **32** | The sweet spot validated in 07e; 192/32 = 6 experts/rank |
| MBS | **1** | Conservative for the first run; try 2 once it works |
| GBS | **2048** to start | Must be divisible by `MBS × DP` = 1×32; 64 microbatches → bubble only 0.2% (see §2.4) |
| Precision | **BF16** (first run) | Matches the official stance; FP8 gains on MoE are questionable, see §3 |
| cuda_graph_impl | **`full_iteration`** | 07e's core result — do not override manually |
| mtp_num_layers | **0 for the first run, then 1** | See the pp_layout discussion below |

### 2.4 BF16 memory budget — 64 GPUs is enough, MBS=1 is safe

Calculator: [`mem_calc.py`](mem_calc.py) (`python3 mem_calc.py` sweeps the grid; `--detail N PP EP MBS` shows the breakdown).

#### Three formulas

```
expert params per GPU     = 286.2B / (PP × EP)     experts are split by both PP and EP
non-expert params per GPU = 8.68B  / PP            replicated within the EP group, split only by PP
optimizer                 = 294.9B × 12B / N       ← EP cancels out; depends only on total GPU count
```

> **The most counter-intuitive line**: expert optimizer = `[286.2/(PP·EP)] × 12 / (DP/EP)` — the experts' DP group is `DP/EP`, so **EP cancels top and bottom**.
> Which means **turning the EP knob only moves weights and gradients; it cannot touch the optimizer**. The optimizer is a hard floor, and the only way down is more GPUs: 51.5 GB at 64, 25.7 GB at 128.

#### Runtime-overhead calibration (important, do not skip)

A naive "weights + grads + optimizer + activations" sum badly underestimates. Back out a factor from 07e's measured anchor:

| | DSV3 31L / 128 GPU / PP2 VPP8 EP64 / MBS1 / MXFP8 |
|---|---|
| Naive four-term sum | 84.5 GB |
| **07e measured max reserved** | **113 GB** |
| **Calibration factor** | **1.34×** |

The missing 28.5 GB is CUDA graph buffers + EP dispatch/combine buffers + NCCL buffers + fragmentation. The table below already includes the 1.34×.

> Single-anchor calibration, so error could be ±15%. **Recalibrate after the first Hy3 run** (record `torch.cuda.max_memory_reserved()`).

#### 64-GPU configuration grid (BF16, seq=4096, VPP=8)

Safety line is 288 × 0.90 = **259 GB** (OOM usually hits at the graph-capture peak, so keep 10% back).

| PP | EP | MBS | Weights | Grads | Optimizer | Activations | Naive | **Actual** | Verdict |
|----|----|----|------|------|--------|------|------|--------|------|
| 2 | 32 | **1** | 16.4 | 32.8 | 51.5 | 45.2 | 145.9 | **195.5** | ✅ **recommended** |
| 2 | 32 | 2 | 16.4 | 32.8 | 51.5 | 90.3 | 191.0 | **256.0** | ✅ right at the line (259) |
| 2 | 16 | 1 | 24.7 | 49.5 | 51.5 | 45.2 | 170.9 | **229.0** | ✅ safe |
| 2 | 16 | 2 | 24.7 | 49.5 | 51.5 | 90.3 | 216.0 | **289.5** | ❌ OOM |
| 4 | 16 | 1 | 12.4 | 24.7 | 51.5 | 46.5 | 135.1 | **181.0** | ✅ safe |
| 4 | 16 | **2** | 12.4 | 24.7 | 51.5 | 93.0 | 181.6 | **243.3** | ✅ **best choice at MBS=2** |
| 4 | 8 | 2 | 20.7 | 41.4 | 51.5 | 93.0 | 206.6 | **276.8** | ⚠️ over the safety line |
| 8 | 8 | 2 | 10.4 | 20.7 | 51.5 | 94.3 | 176.8 | **237.0** | ✅ but PP=8 has a large bubble |

#### What is the minimum GPU count

| GPU | Nodes | Leanest config | Optimizer floor | Actual GB | Verdict |
|-----|------|---------|-----------|--------|------|
| 16 | 4 | PP=8 EP=2 | 206.0 | 481.2 | ❌ OOM |
| 24 | 6 | PP=8 EP=3 | 137.3 | 344.6 | ❌ OOM |
| **32** | 8 | PP=8 EP=4 | 103.0 | **276.3** | ⚠️ fits in 288 but with no headroom |
| **40** | 10 | PP=8 EP=5 | 82.4 | **235.3** | ✅ **theoretical minimum** |
| 48 | 12 | PP=8 EP=6 | 68.7 | 207.9 | ✅ safe |
| **64** | **16** | PP=2 EP=32 | 51.5 | **195.5** | ✅ **recommended (this run)** |
| 128 | 32 | PP=4 EP=32 | 25.7 | 129.8 | ✅ very comfortable |

> **Conclusion: BF16 Hy3 training needs 40 GPUs (10 nodes) in theory; 64 GPUs (16 nodes) in practice.**
> 32 GPUs pencils out at 276 GB inside 288, but with zero headroom and a mandatory PP=8 (large bubble, 48 experts/rank) it is not worth it.
> 64 GPUs is not merely "enough" — it leaves 60+ GB to spend on MBS=2 or on adding MTP.

#### How large should GBS be

Constraint: **GBS must be divisible by `MBS × DP`**. At VPP=8 the bubble is `(PP-1)/(microbatch count × VPP)`.

For the recommended **PP=2 EP=32 MBS=1 (DP=32)**:

| GBS | microbatch/rank | pipeline bubble | tokens/step | Assessment |
|-----|----------------|-----------------|-------------|------|
| 1024 | 32 | 0.39% | 4.2 M | on the small side |
| **2048** | 64 | **0.20%** | **8.4 M** | ✅ **use this for the first run** |
| 4096 | 128 | 0.10% | 16.8 M | matches the official DSV3 setting |
| 8192 | 256 | 0.05% | 33.6 M | upper end of the sweep |

> **At PP=2 + VPP=8 the bubble is already negligible** (0.2% even at GBS=2048), so 07e's "GBS is the biggest knob" conclusion **does not hold in this configuration** — that was DSV3 at PP=2 but on 256 GPUs with DP=128.
> Here, raising GBS mainly **amortizes fixed overheads** (optimizer step, all-reduce, GC) rather than removing bubble. Start at 2048 and sweep to 4096/8192 to see whether anything is left on the table.

### 2.5 Deriving pp_layout (80 layers vs DSV3's 61)

DSV3's 61 layers at PP2×VPP8 = 16 chunks lay out as `Et*4|(t*4|)*14tmL`:
- chunk 1: `E` + 4 layers → chunks 2–15: 14 × 4 layers = 56 → chunk 16: 1 layer + `m` (MTP) + `L` (loss)
- Total 4 + 56 + 1 = **61** ✓, chunk count 1 + 14 + 1 = **16** ✓

Hy3's 80 layers over the same 16 chunks → 5 layers per chunk on average:

```
without MTP (recommended first run):  Et*5|(t*5|)*14t*5L      → 5 + 70 + 5 = 80 ✓
with MTP:                             Et*5|(t*5|)*14t*5mL     → 5 + 70 + 5 = 80 ✓ (last chunk also carries m + L)
```

> ⚠️ **Risk of an imbalanced last stage**: DSV3 deliberately puts only 1 layer in the last chunk because MTP + the loss head are heavy. Splitting Hy3 evenly at 5 layers leaves the last chunk carrying 5 layers + MTP + loss, which **may OOM or become the bubble bottleneck**.
> **Alternative layout** (front-loaded): `Et*6|(t*5|)*14t*4mL` → 6 + 70 + 4 = 80 ✓
> **Both layouts are derived, not measured.** For the first run, set `mtp_num_layers=0` with the even split, get it working, then add MTP and adjust against measured memory.

### 2.6 Fallback configurations (if V1 does not run)

| Situation | Adjustment | Note |
|---|---|---|
| OOM (during capture) | PP 2 → **4** (drop EP to 16 accordingly) | 20 layers per stage halves the memory pressure; the cost is a larger bubble |
| OOM (still not enough) | Disable MTP + MBS=1 + `--recompute-granularity selective` | See 07e trap D |
| hybridep hangs | Check whether `NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN` **== EP** | 07e major trap 1: changing EP means changing it in two places |
| full graph capture crashes | Confirm both full-graph env vars from Part 3 are set | 07e trap C |

---

## 3. Precision: the official stance is BF16, and FP8 may not help MoE

### 3.1 Tencent's own position — BF16 throughout, no FP8 training path anywhere

Every file in the official repo was checked, and they agree:

| Source | Precision setting |
|---|---|
| Model card "supported precision" | **BF16** |
| `llama_factory_support/hy_v3_full_sft.yaml` | `bf16: true` |
| `ms_swift_support/hy_v3_full_sft.yaml` | `torch_dtype: bfloat16` / `bf16: true` / `# fp16: false` |
| `deepspeed_support/train.sh` | `--bf16` |
| `ds_zero3_*.json` | `bf16.enabled: auto` (activated by `--bf16`) |
| **Hy3-FP8** | **an inference quantization release**, produced by the AngelSlim PTQ toolkit — **not a training precision** |
| Pretraining precision | **not disclosed** (the README mentions only "pretraining framework rebuild", with no precision statement) |

> **The distinction that matters**: Hy3-FP8 is a post-training-quantization artifact for inference, which is a different thing from *training in FP8*. Tencent has **not publicly claimed** to have done DeepSeek-V3-style FP8 pretraining.

→ Running FP8_MX on the Megatron side is therefore **our own choice, not officially validated**.

### 3.2 Does FP8 actually help MoE — measurements from this repo on GB200

Intuition says GB300 FP8 peak 5400 vs BF16 2700 = 2×, but **MoE models do not cash that in**. Our own measurements:

| Source | Config | BF16 | FP8 / MXFP8 | Delta |
|---|---|---|---|---|
| [a4x README A2 vs A3](../../a4x/07-megatron-training/README.md) | DSV3 12L, 8 GPU, seq 16384, EP8 MBS2 | **527** | 503 (FP8) | **FP8 is 5% slower** |
| [a4x 07c §2.2 steps 7→8](../../a4x/07-megatron-training/07c-deepseekv3-671b-recipe/README.md) | DSV3 32L, PP2 EP64, CUDA graph already on | 928 | 970 (+mxfp8 +fp32 optim +fp8-param-gather) | **+4.5%** |

**Why** (quoting the a4x README's own conclusion): *"the FP8 path's overhead in grouped GEMM cancels the Tensor Core speedup, so BF16 ends up faster"*. MoE bottlenecks on all-to-all communication and memory traffic, not raw GEMM throughput.

### 3.3 ⚠️ Clarification: those 1100–1658 numbers are **all MXFP8, not BF16**

Easy to conflate, so spelled out:

| Platform | Config | Precision | TFLOP/s/GPU |
|---|---|---|---|
| GB200 | DSV3 16L 64GPU Bridge V2 (MNNVL=2) | MXFP8 | 1176 |
| GB200 | DSV3 16L 64GPU Bridge V2 | MXFP8 | 1124 |
| GB200 | DSV3 16L 64GPU (MNNVL=0) | MXFP8 | 1100 |
| GB200 | DSV3 32L 128GPU raw Megatron | MXFP8 | 992 |
| GB200 | DSV3 61L 256GPU (NVIDIA reference) | MXFP8 | 1292 |
| GB300 | DSV3 61L 256GPU | MXFP8 | **1658** |
| GB200 | DSV3 61L PP4 EP32 baseline | **BF16** | **300** (alltoall unoptimized) |

> **This repo has no "fully optimized BF16" DSV3 number.** The 300 is an unoptimized-alltoall baseline (later taken to 900+ via HybridEP/graph work, none of which is precision-related), so it **cannot stand in for BF16**.
>
> The only clean same-config BF16↔FP8 comparisons are the two in §3.2: **−5% and +4.5%**.

### 3.4 Revised expectation band for Hy3

The 1200–1500 in §7 is a **FP8_MX figure** (anchored on GB300 DSV3's 1658). Adjusting by the real FP8 gain from §3.2:

| Precision | Hy3 expected band | Basis |
|---|---|---|
| FP8_MX | 1200 – 1500 | scaled down from DSV3 GB300 1658 for structural differences |
| **BF16** | **1150 – 1450** | FP8 vs BF16 is only −5% to +5%, **not a halving** |

> **Do not** reason "peak FLOPS halves → throughput halves" for BF16. That logic only holds for dense models on pure GEMM workloads.

**An extra worry specific to Hy3**: `moe_ffn_hidden_size` 1536 is smaller than DSV3's 2048, and `hidden_size` 4096 is narrower than 7168 → **smaller grouped-GEMM shapes → quantize/dequantize overhead is a larger fraction**. The payoff could be worse than DSV3's, possibly negative (matching the −5% in a4x A3).

### 3.5 Conclusion: use BF16 for the first run

1. **Match the official stance** — Tencent trained in BF16, so this removes one "differs from official" variable.
2. **One less risk** — FP8_MX is entirely unvalidated on Hy3 (Bridge has no hy3 recipe, and nobody has exercised TE's hy_v3 FP8 path).
3. **The upside is small anyway** — ±5% on MoE is not worth introducing during bring-up.
4. Once BF16 has produced a baseline, **run a separate same-config FP8_MX comparison** (see §7 item #9) and let the measurement decide.

---

## 4. Three routes to getting this running

### Route A: Bridge AutoBridge + a hand-written recipe (**recommended; the benchmark takes this route**)

`HYV3Bridge` is registered and can generate a correct `GPTModelProvider` straight from the HF config:

```python
from megatron.bridge import AutoBridge
bridge   = AutoBridge.from_hf_pretrained("tencent/Hy3")
provider = bridge.to_megatron_provider()     # MoE/router/MTP params filled in correctly

# training-side overrides (see §1.3)
provider.moe_router_bias_update_rate   = 1e-3
provider.moe_token_dispatcher_type     = "flex"
provider.moe_flex_dispatcher_backend   = "hybridep"
provider.moe_a2a_overlap               = True
# parallelism (see §2.3)
provider.tensor_model_parallel_size    = 1
provider.pipeline_model_parallel_size  = 2
provider.virtual_pipeline_model_parallel_size = 8
provider.expert_model_parallel_size    = 32
```

Full script in [`hy3_provider.py`](hy3_provider.py) in this directory.

> **Note**: `run_script.py -m hy3 -mr ...` **does not work** (Bridge has no hy3 perf config). Either use a hand-written pretrain script, or drop a `hy3_workload_base_configs.py` into the container (structured after `deepseek_workload_base_configs.py`).

### Route B: mock-data pure benchmark (no weight download)

A throughput benchmark needs no real weights. Build the provider by hand from the tables in §1.1/§1.2 (`build_hy3_provider_from_scratch()` in `hy3_provider.py`), pass `--data mock`, and stay on the same footing as 07d/07e. **Recommended for the first run** — it skips a 590 GB weight download.

### Route C: the official SFT stack (a precision reference, not a performance route)

Tencent ships three: `finetune/deepspeed_support/` (HF Trainer + DeepSpeed), `finetune/llama_factory_support/`, `finetune/ms_swift_support/`.

Official hardware requirements (`max_seq_length=4096`, with `make_moe_param_leaf_module` and zero3+offload disabled):
- LoRA fine-tune: ≥ 1 node, 8 GPUs (80GB+)
- Full fine-tune: ≥ 4 nodes, 32 GPUs (80GB+)

> **Use it as a precision baseline** (run a few hundred steps and compare loss curves to confirm our Megatron config is not mis-set). **Do not** use it for MFU — DeepSpeed ZeRO on GB300 cannot match Megatron's PP+EP+CUDA graph.

### Preparing the weights (needed for routes A/C)

The official checkpoint **stores each expert's weights separately**, which makes loading extremely slow. Tencent provides a conversion script that fuses them into 3D tensors:

```sh
# convert (from finetune/tools/)
python convert_ckpt_to_outer.py --input_dir <raw ckpt> --output_dir <output> --workers 8
# verify
python check_converted.py <output dir> --spot-check 3
```

> The official README says `train/tools`, but the actual path in the HF repo is **`finetune/tools/`** (a typo in their docs).

### Training data format (route A SFT / route C)

Hy3 has fast/slow thinking modes, marked in the data with `reasoning_effort`:

```json
{"reasoning_effort": "no_think", "messages": [{"role":"system","content":"..."},{"role":"user","content":"1+1=?"},{"role":"assistant","content":"1+1=2"}]}
{"reasoning_effort": "high",     "messages": [..., {"role":"assistant","content":"1+1=2","reasoning_content":"The user is asking 1+1 ..."}]}
```

```python
tokenizer.apply_chat_template(messages, is_training=True)   # use_fast=False, trust_remote_code=True
```

> The chain of thought for slow-thinking samples goes in the `reasoning_content` field, not `content`. Training on a mix of both modes is what preserves the `reasoning_effort` switch behaviour.

---

## 5. Cluster preparation (16 nodes, single domain)

### Available pools (bunny's 2026-07-25 qualification: 5 pools, all PASS, 88 nodes / 352 GPUs total)

| Pool | Nodes | GPU | NCCL all_reduce | Note |
|---|---|---|---|---|
| **gb300-pool-0015** | 18 | 72 | 689.3 GB/s | ✅ **first choice here** (use 16, keep 2 as hot spares) |
| gb300-pool-0016 | 18 | 72 | 688.3 GB/s | alternative |
| gb300-pool-0017 | 18 | 72 | 689.6 GB/s | alternative |
| gb300-pool-0014 | 16 | 64 | 688.9 GB/s | exactly 16 with **zero spare**; the GKE ERROR status is it complaining about not reaching 18, the nodes are healthy |
| gb300-pool-0013 | 18 | 72 | 688.3 GB/s | 1 WARN: pqcm node DRAM correctable ECC >1000 (soft errors, correctness unaffected) |

**Do not use**: `pool-0002` (yangwhale workload running + bad COS), `pool-0006` (infer team, 17 pods + bad COS), `pool-0009` (idle with good COS, but outside the delivery scope — ask chris first).

Multi-node already verified: MNNVL=ON **933 GB/s**, MNNVL=OFF over RDMA 379 GB/s, NVLink fabric healthy.

> **Why an 18-node pool rather than the exactly-16 pool-0014**: cluster **auto-repair is disabled** (deliberately, to prevent accidental node swaps), so a failed node does not self-heal. Using 16 out of an 18-node pool leaves 2 hot spares in the same subblock — a failure is a label change, not a whole-pool migration (the "commandeer the entire pool" experience in 07e was painful).

### ⚠️ Two cluster-level hard constraints

| # | Constraint | Consequence / mitigation |
|---|---|---|
| 1 | **Nodes must never roll to node image `1.36.0-gke.4681000` (COS 224.80)** | That version's `nvidia.ko` has a regression where `cuDevicePrimaryCtxRetain` returns `INVALID_VALUE` — `nvidia-smi` looks fine but **every CUDA workload dies**. A maintenance exclusion currently freezes upgrades, **expiring 2026-10-23**; renew before then. All 5 usable pools are on kubelet `1.36.0-gke.4447000` (the good image). |
| 2 | **auto-repair is fully disabled** | Failed nodes do not self-heal, so **you must monitor yourself**. If a node drops mid-training → move the label to a hot spare → delete the pod to reschedule. |

### Step 0 — label the target pool

```bash
CTX=gke_tencent-gcp-taiji-poc_us-central1_gb300-gke-test
POOL=gb300-pool-0015

# label 16 Ready nodes (pool = subblock, so they are same-domain by construction)
N=$(kubectl --context $CTX get nodes -l cloud.google.com/gke-nodepool=$POOL --no-headers \
    | grep -w Ready | awk '{print $1}' | head -16)
kubectl --context $CTX label node $N team=yangwhale --overwrite

# verify: should be 16, and exactly one subblock
kubectl --context $CTX get nodes -L cloud.google.com/reservation-subblocks -l team=yangwhale \
  --no-headers | awk '{print $NF}' | sort | uniq -c

# while you are here, confirm the node image is not the bad 224.80
kubectl --context $CTX get nodes -l team=yangwhale \
  -o custom-columns=NODE:.metadata.name,KUBELET:.status.nodeInfo.kubeletVersion --no-headers \
  | awk '{print $2}' | sort | uniq -c    # expect all v1.36.0-gke.4447000
```

> **Must be one subblock; do not stitch nodes across pools** (lesson from 07e). Staying inside one pool satisfies this automatically.

### Step 1 — bring up the pod pool

```bash
kubectl --context $CTX apply -f yw-pool-64.yaml
kubectl --context $CTX get pods -l job=yw --no-headers | grep -c Running   # should be 16
kubectl --context $CTX get computedomains | grep yw                        # 1, Ready
```

### Step 2 — distribute the launch script and start training

```bash
B64=$(base64 -w0 run-hy3-yw.sh)
seq 0 15 | sed 's/^/yw-a-/' | xargs -P 16 -I {} kubectl --context $CTX exec {} -- bash -c \
  "echo $B64 | base64 -d > /tmp/run-hy3-yw.sh && chmod +x /tmp/run-hy3-yw.sh"

seq 0 15 | sed 's/^/yw-a-/' | xargs -P 16 -I {} kubectl --context $CTX exec {} -- bash -c \
  "nohup /tmp/run-hy3-yw.sh > /tmp/hy3-run.log 2>&1 &"
```

> Parallel `kubectl exec` is fine at 16 pods (07e's konnectivity throttling only triggers at 64).

### Step 3 — monitor

```bash
kubectl --context $CTX exec yw-a-0 -- bash -c 'grep -E "Step Time|MODEL_TFLOP" /tmp/hy3-run.log | tail -6'
```

---

## 6. Key environment variables

Identical to 07e, with **one difference: `NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN` must equal this recipe's EP**. The full list is in `run-hy3-yw.sh`; only the easiest-to-miss ones are here:

| env | Value | What happens if you miss it |
|---|---|---|
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True,graph_capture_record_stream_reuse:True` | full graph capture dies with `StreamCaptureUnjoined` |
| `TORCH_NCCL_AVOID_RECORD_STREAMS` | `0` | same as above (the base image defaults to 1, so set it explicitly) |
| `NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN` | **`32`** (== EP) | hybridep all-to-all hangs, collective timeout |
| `NVLINK_DOMAIN_SIZE` / `USE_MNNVL` | `72` / `1` | hybridep does not engage |
| `NCCL_GRAPH_REGISTER` | `0` | with GB300 GIB, =1 hangs rendezvous for 25 min |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `32` | hybridep + sm100 loses performance |
| `NVTE_FWD/BWD_LAYERNORM_SM_MARGIN` | `20` | SM contention under hybridep |

---

## 7. Verification checklist (fill in as the run proceeds)

| # | Item | Pass criterion | Result |
|---|---|---|---|
| 1 | Provider parameter-count check | log prints total ≈ 295B (±1%) | ⬜ |
| 2 | Single-node 4-GPU smoke (reduced to 8 layers) | 10 steps without crashing, loss decreasing | ⬜ |
| 3 | 64-GPU start + full graph capture | gets past capture into steady state | ⬜ |
| 4 | Steady-state throughput | record TFLOP/s/GPU + step time | ⬜ |
| 5 | Expert load balance | expert-load CV converges with `bias_update_rate=1e-3` on | ⬜ |
| 6 | MTP on/off comparison | throughput/memory delta between mtp=0 and mtp=1 | ⬜ |
| 7 | EP sweep | EP 16 vs 32 (does the DSV3 conclusion reproduce?) | ⬜ |
| 8 | GBS sweep | 2048 → 4096 → 8192, find the MFU knee | ⬜ |
| 9 | **BF16 vs FP8_MX at identical config** | 30 steps each; does §3.2's "±5% for FP8 on MoE" hold for Hy3? | ⬜ |
| 10 | **Measured memory vs the estimate** | record `max_memory_reserved()`, compare against §2.4's predicted 195.5 GB, recalibrate the 1.34× factor | ⬜ |

### With no reference baseline, how do you judge "is this good"

Hy3 has **no official Megatron benchmark**, and it is absent from the NVIDIA perf summary. The only yardstick is lateral comparison:

| Model | Scale | Precision | GB300 measured TFLOP/s/GPU | Note |
|---|---|---|---|---|
| DeepSeek V3 | 671B / 61L | MXFP8 | **~1658** (256 GPU) | 99.3% of the official 1670 |
| DeepSeek V3 scale-in | 671B / 31L | MXFP8 | ~1550 (128 GPU) | half the layers → fixed overhead is a larger share |
| **Hy3** | **295B / 80L** | **BF16 first run** | **TBD** | expectation below |

**Reasoning behind the expected band** (not measured; for judgement only):
- **In our favour**: more layers (80 vs 61) → enough layers per stage to amortize fixed overhead; finer experts (moe_ffn 1536 vs 2048); GQA is computationally simpler than MLA.
- **Against**: hidden 4096 is narrower than DSV3's 7168 → **smaller GEMM shapes, lower Tensor Core efficiency**; fewer total params but more layers → more per-layer activation communication.
- **Rough call (BF16 first run)**: steady state landing in **1150–1450 TFLOP/s/GPU** is normal; below 1000 means something is misconfigured (check first whether EP / dispatcher / graph actually took effect).
- **FP8_MX figure**: 1200–1500. The two are close, for the reason in §3 — FP8 on MoE is only −5% to +5% relative to BF16, **not a doubling in line with peak FLOPS**.

---

## 8. Traps predicted in advance (carried over from 07e)

| # | Trap | Prevention |
|---|---|---|
| 1 | Manually overriding `cuda_graph_impl` | **Do not override** — it bypasses mechanisms full graph requires, such as `moe_paged_stash` |
| 2 | Changing EP but forgetting `NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN` | Both must change together, or all-to-all hangs |
| 3 | Borrowing parameters across hardware generations (GB200's VPP) | Use GB300-native values only |
| 4 | Launching torchrun directly does not set the perf env | Export everything by hand (§6) |
| 5 | Starting over SSH loses the container ENV | `run-hy3-yw.sh` already loads `/proc/1/environ` |
| 6 | An NCCL crash floods the disk → DiskPressure eviction | After a crash, clear Evicted pods before relaunching; do not chain-run through it |
| 7 | Running immediately after heavy ComputeDomain churn | Let DRA/IMEX/RDMA settle first (the most valuable lesson from 07e) |
| 8 | **Hy3-specific**: bias_update_rate=0 unbalances the experts | from-scratch must set 1e-3 (§1.3) |
| 9 | **Hy3-specific**: the last stage carries 5 layers + MTP + loss | Disable MTP for the first run; on OOM switch to the front-loaded layout (§2.5) |
| 10 | **Cluster**: a node rolls to COS 224.80 → all CUDA dies | Maintenance exclusion is in force (expires 2026-10-23); check the kubelet version before starting |
| 11 | **Cluster**: auto-repair is off, so bad nodes do not self-heal | Monitor yourself; an 18-node pool leaves 2 hot spares — a failure is a label change |

---

> **For SFT / fine-tuning see the companion [SFT.md](SFT.md)** — this document covers pretraining performance only.

## Appendix: file inventory

| File | Purpose |
|---|---|
| `yw-pool-64.yaml` | single-ComputeDomain 16-node 64-GPU sleep-infinity pod pool |
| `run-hy3-yw.sh` | per-pod launch script (full env + rank computation + torchrun) |
| `hy3_provider.py` | Hy3 `GPTModelProvider` construction (AutoBridge path + from-scratch path + training overrides) |
| `mem_calc.py` | memory calculator (sharding formulas + runtime-overhead calibration + GBS / minimum-GPU advice) |
| `sweep.sh` / `gen_table.py` | ablation sweep framework + big-table generator |
| `loss_align.sh` | FP8 vs BF16 training-quality alignment check |
| `timeline.py` | startup timeline: `--stamp` adds per-line timestamps, `--parse` breaks down phase durations |
| `sweep256.sh` / `yw-pool-256.yaml` | 256-GPU cross-4-domain sweep framework + 4-ComputeDomain pod pool |
| **`SFT.md`** | **The SFT plan (companion doc)**: load the official weights to inject scarce knowledge |
| `install_hy3_bridge.sh` | port the single-file HYV3Bridge from `main` into the r0.5.0 container (§14) |
| `import_hy3_ckpt.py` | HF weights → Megatron torch_dist checkpoint (distributed conversion) |
| `make_sft_data.py` / `sft_data/` | scarce-knowledge SFT dataset generator: train set + held-out set + general probes |
| `hy3_sft.py` | SFT training entry point (goes through HYV3Bridge, not the Qwen3 skeleton) |
| `raid-disks.yaml` | RAID-0 the 4 local NVMe drives — 12 TB per node mounted at `/mnt/disks/raid/0` |
| `make_mixed_sft.py` | blend scarce knowledge into a general SFT dataset (a small dataset cannot feed a MoE's experts) |
| `run_sft.sh` | 16-node SFT launcher: clean → distribute → torchrun → auto-export to HF when training ends |
| `export_sft_dist.py` | distributed after-the-fact export (**proven unworkable**, kept as a negative record) |
| `eval_sft.py` | three-criterion SFT evaluation; run once before and once after, then `--compare` |

---

*2026-07-25 · Recipe design complete, awaiting the 16-node 64-GPU live run · MoE lineage = DeepSeek V3, attention = GQA*

---

## 9. Live-run log (from 2026-07-25)

> Written as it happened. Every verified step and every trap fallen into is here, so nothing gets lost.

### Milestone M1 — cluster in place (22:26 HKT ✅)

| Step | Result |
|---|---|
| kubectl straight from this machine to gb300-gke-test | ✅ works, **no ssh to gLinux needed** (authorized-networks already allows it) |
| pool-0015 health check | 18/18 Ready, kubelet all `v1.36.0-gke.4447000` (the good COS), **same subblock** `10270c36...0632` |
| Label 16 nodes | ✅ `hy3=true`, 2 left as hot spares |
| Bring up the pod pool | ✅ 16/16 Running (~2 min for the first image pull), ComputeDomain `yw-cd-a` created |

**Trap A: the `team=yangwhale` label cannot be reused.**
`pool-0002` already has **17 nodes labelled `team=yangwhale`** running someone else's workload. Reusing that label would make the two nodeSelectors cross-talk.
→ Switched to a dedicated label **`hy3=true`**; the YAML is updated. **Lesson: check which labels are already taken before entering a cluster — do not assume a label is yours alone.**

### Milestone M2 — the container has no hy_v3 bridge, so switch to a self-built recipe (22:35 HKT ✅)

**Trap B (blocking): Bridge r0.5.0 inside the container has no `hy_v3` model bridge.**

```
container: Bridge 0.5.0+fcbb6031
models/ has:  bailing deepseek ernie gemma glm gpt_oss kimi llama mamba minimax
              ministral3 mistral nemotron olmoe qwen stepfun ...
models/ lacks: hy_v3   ← §4 route A (AutoBridge) is dead on arrival
recipes/ lacks: hunyuan / hy3
scripts/performance/configs/ has: deepseek gpt_oss kimi llama nemotronh qwen qwen_vl wan
```

`HYV3Bridge` exists only on Megatron-Bridge **main**; it is not in the r0.5.0 container.

**Trap C: `run_script.py`'s CLI overrides are not enough.**
It only exposes `--hidden_size` / `--num_layers` / `--first_k_dense_replace` / `--vocab_size` / `--pipeline_model_parallel_layout` —
there is **no** `--num_moe_experts` / `--num_query_groups` / `--moe_ffn_hidden_size`, so a deepseek recipe cannot be reshaped into Hy3.

**Solution: build the config on the Qwen3-235B-A22B recipe as a skeleton.**

Why that one (checked against the actual return value of `qwen3_235b_a22b_pretrain_config()`):
- it returns a **bare `GPTModelProvider`** (not an MLA-specific provider), so every field is writable
- it is itself a GQA MoE, and **hidden_size 4096 / kv_channels 128 / qk_layernorm True / moe_ffn 1536 already match Hy3 exactly**
- optimizer / ddp / dataset / scheduler are all already tuned

Only these need changing: `num_layers 94→80`, `ffn_hidden_size 12288→13312`, `num_moe_experts 128→192`,
`num_query_groups 4→8`, `vocab 151936→120832`, plus the DSV3-lineage MoE knobs on top.

`GPTModelProvider` has **309 fields**; of the 34 I needed, the only one that does not exist is `untie_embeddings_and_output_weights`
(it is actually **`share_embeddings_and_output_weights`**, inverted).

Output: **[`hy3_pretrain.py`](hy3_pretrain.py)** (replaces `hy3_provider.py` from §4 route A).

### Milestone M3 — the high-performance recipe ported verbatim from the deepseek source (✅)

Read the container's `scripts/performance/configs/deepseek/deepseek_llm_pretrain.py` and lifted two functions unchanged:

`set_deepseek_v3_common_configs`:
```python
moe_router_fusion = True
recompute_granularity = "selective"
dist.enable_megatron_core_experimental = True
mixed_precision.grad_reduce_in_fp32 = False   # ← gradients are BF16
ddp.grad_reduce_in_fp32 = False
moe_router_force_load_balancing = True        # benchmark only
```

`set_full_iter_cg_configs` (**the full graph + paged stash the user asked for by name**):
```python
moe_pad_experts_for_cuda_graph_inference = True
moe_paged_stash = True                        # MCore PR #4247
moe_expert_rank_capacity_factor = 1.5
moe_paged_stash_buffer_size_factor_cuda = 1.2
moe_paged_stash_buffer_size_factor_cpu = 1.0
```
> The source comments explain the mechanism: **dropless MoE produces variable-length per-expert tensors that a CUDA graph cannot capture**.
> Pad them to a fixed capacity first (`pad_experts` + capacity factor), then claw the memory back with paged stashing.

**⭐ Correction to the §2.4 memory estimate**: `grad_reduce_in_fp32 = False` means gradients are **BF16 (2 B), not FP32 (4 B)**.
At 64 GPU / PP2 / EP32 / MBS1 gradients drop from 32.8 GB → **16.4 GB**, naive total 145.9 → **129.5 GB**,
and ×1.34 gives **173.6 GB** (was 195.5 GB). **More headroom than expected.**

### Milestone M4 — config builds cleanly (22:40 HKT ✅)

```
Hy3 295B | 64 GPU | TP1 PP2 VPP8 EP32 | MBS1 GBS2048 | bf16
  layers 80 (dense 1 + MoE 79)  hidden 4096  GQA 64Q/8KV x 128
  MoE 192 experts top-8 ffn 1536 shared 1536
  routing sigmoid + expert_bias=True (rate 0.001) lb=none scale 2.826
  dispatcher flex/hybridep  graph full_iteration  paged_stash True
  MTP None  pp_layout Et*5|(t*5|)*14t*5L
  DP=32  microbatch/rank=64  expert params 286.3B (4.47B per rank)
```

Three API mismatches got fixed on the way (all structural differences between the qwen3 and deepseek recipes):

| # | Error | Root cause | Fix |
|---|---|---|---|
| 1 | `'str' object has no attribute 'grad_reduce_in_fp32'` | the qwen3 recipe's `cfg.mixed_precision` is the **string** `'bf16_mixed'`, not a config object | `if isinstance(cfg.mixed_precision, str): cfg.mixed_precision = bf16_mixed()` |
| 2 | `'NoneType' object has no attribute 'overlap_grad_reduce'` | the qwen3 recipe's `cfg.comm_overlap` defaults to **None** (only the deepseek config constructs it) | construct `CommOverlapConfig(...)` explicitly |
| 3 | `CommOverlapConfig.__init__() missing 'tp_comm_overlap'` | that field is **keyword-only and required** | `CommOverlapConfig(tp_comm_overlap=False)` (TP=1, so no TP overlap) |

> **General lesson**: when borrowing a skeleton across recipe families, **fields with the same name may have different types**
> (str vs object, None vs instance). Copying family A's assignments onto family B's skeleton will hit null-pointer/type
> errors like these; knock them down one dryrun at a time.

### Milestone M5 — 4-GPU reduced-layer smoke test passes (15:02 HKT ✅)

Config: 4 GPU / 8 layers / TP1 PP2 VPP2 EP2 / MBS1 GBS8 / seq4096 / BF16 / TE graph.

```
iteration 1/5 | elapsed 77498.7 ms | lm loss 1.251689E+01 | grad norm 6.013   <- includes graph capture
iteration 2/5 | elapsed   572.3 ms | lm loss 1.251494E+01 | TFLOP/s/GPU 233.6
iteration 3/5 | elapsed   532.1 ms | lm loss 1.224570E+01 | TFLOP/s/GPU 251.3
iteration 4/5 | elapsed  4155.2 ms | lm loss 1.139484E+01 | <- periodic GC
iteration 5/5 | elapsed   455.9 ms | lm loss 1.086530E+01 | TFLOP/s/GPU 293.3
Rank 0: 16 graphs deleted with explicit reset
```

✅ **Loss falls monotonically 12.52 → 10.87, 0 NaN, 0 skipped, CUDA graphs created and destroyed normally. Model structure and the forward/backward path are verified.**
8.12 B parameters per rank (PP rank 1), 27.26 B total (the 8-layer cut-down), dense+embedding 1.89 B.

### The 4 traps in this stretch (all the price of "borrowing a skeleton")

| # | Error | Root cause | Fix |
|---|---|---|---|
| D | `moe_expert_rank_capacity_factor requires use_transformer_engine_op_fuser to be enabled` | paged stash has an **implicit dependency chain**: `cutedsl_fused_grouped_mlp=True` → `use_transformer_engine_op_fuser=True` → only then is `moe_expert_rank_capacity_factor` allowed (see `scripts/performance/utils/overrides.py:238-239`). Hand-writing the high-perf config, I copied only the terminal fields and missed the middle of the chain | **Reuse the official `WorkloadBaseConfig` + `set_workload_base_configs()` directly** instead of re-implementing the mapping by hand |
| E | `pipeline_model_parallel_layout cannot be set with other pipeline layout arguments` | the qwen3 skeleton defaults `account_for_embedding_in_pipeline_split=True` / `account_for_loss_in_pipeline_split=True`, while the pp_layout string already spells out `E` and `L` — mutually exclusive (the deepseek skeleton defaults them to False, hence no such problem) | set both `account_for_*` to False whenever pp_layout is set |
| F | `Model vocab_size (120832) cannot be smaller than tokenizer's vocab_size (151669)` | the skeleton ships the **Qwen tokenizer** (vocab 151669), larger than Hy3's 120832 | for the mock benchmark switch to `NullTokenizer` + `vocab_size=120832` and clear `tokenizer_model` |
| G | background job silently killed, log frozen on stale content | a background process from `kubectl exec ... "cmd &"` gets SIGTERM when the exec session ends | use **`setsid nohup ... < /dev/null &`** to fully detach from the session |

> **The lesson running through all of them**: when borrowing a skeleton across recipe families, the trap is not "wrong field value",
> it is that **family A's implicit assumptions do not hold in family B** (opposite True/False defaults, a str where an object is expected,
> a missing middle link in a dependency chain).
> **Countermeasure: call the official setter rather than hand-writing assignments** — `set_workload_base_configs()` eliminated the whole D class at once.

### ⭐ Major finding: the official BF16 recipe does **not** use the full_iteration graph

Reading `deepseek_workload_base_configs.py`, NVIDIA's two GB300 configs differ a great deal:

| | `..._GB300_BF16_V1` | `..._GB300_FP8_MX_V1` |
|---|---|---|
| cuda_graph_impl | **`transformer_engine`** | **`full_iteration`** |
| cuda_graph_scope | `[attn, moe_router, moe_preprocess]` | (whole iteration) |
| moe_a2a_overlap | **False** | True |
| cutedsl_fused_grouped_mlp | **off** | True |
| fp8_dot_product_attention | — | True |
| recompute_modules | `["moe_act"]` | `[]` |
| PP / VPP / EP | 4 / 4 / 64 | 2 / 8 / 32 |

**So `full_iteration` + paged stash are FP8_MX-only in the official recipe; BF16 goes through the TE graph.**
The causal chain: paged stash needs the TE op fuser, the op fuser is turned on by `cutedsl_fused_grouped_mlp`, and NVIDIA only enables cutedsl in the FP8 tier.

→ Strategy adjustment: **take a stable baseline with the official BF16 config (TE graph) first**, then separately test whether
`BF16 + cutedsl + full_iteration` works (`hy3_pretrain.py` already has `--cuda-graph full_iteration --cutedsl --a2a-overlap` switches).

### Milestone M6 — full 64-GPU run, BF16 baseline 707 TFLOP/s/GPU (23:13 HKT ✅)

**V1 config**: 64 GPU / all 80 layers / TP1 PP2 VPP8 EP32 / MBS1 GBS2048 / seq4096 / BF16 /
TE graph / hybridep / `recompute_modules=[moe_act]` + selective — i.e. NVIDIA's official GB300 BF16 recipe.

| Check | Expected | Measured | Verdict |
|---|---|---|---|
| Total parameters | 295 B (official) | **294.97 B** | ✅ exact match |
| Peak memory | §2.4 predicted 173.6 GB | **184 GB** | ✅ dead on after recalibration |
| Steady-state throughput | — | **707.2** (median n=20, 706–710, **±0.3%**) | baseline |
| Step time | — | 25.3 s | |
| GPU utilization | — | 99–100% (all 16 pods) | ✅ |

**⭐ Recalibrating the memory model**: `grad_reduce_in_fp32=False` → gradients are **BF16 (2 B)**, not FP32 (4 B).
The naive four terms = weights 16.4 + gradients 16.4 + optimizer 51.5 + activations 45.2 = **129.5 GB**; measured 184 GB → factor **1.42×**.
`mem_calc.py` is updated and now predicts 183.9 GB against a measured 184 GB.

> The 121.3 TFLOP/s first step is graph-capture overhead. Roughly **10 minutes** from launch to the first steady-state number.

### Milestone M7 — performance sweep: 707 → 854 TFLOP/s (+20.8%)

| Version | Change | Steady TFLOP/s/GPU | Step | Memory | vs V1 |
|---|---|---|---|---|---|
| **V1** | official BF16 recipe | **707.2** (706–710) | 25.3 s | 184 GB | baseline |
| **V2** | all recompute off | **744** (743.6–744.5) | 24.0 s | 217 GB | **+5.2%** |
| **V3** | V2 + `cutedsl` + `a2a_overlap` | **827** (825.3–828.5) | 21.6 s | **197 GB** | **+16.9%** |
| **V4** 🏆 | V3 + `full_iteration` + **paged stash** | **854.4** (854.2–854.8) | **20.97 s** | 230 GB | **+20.8%** |
| V5 | V4 + MBS 2 | ❌ **OOM** | — | hit the 277 GiB ceiling | — |
| V6 | V4 + GBS 4096 | ❌ **hang** | — | 283 GB (99.96%) | — |

**V2**: turning recompute off buys +5.2% at a cost of +33 GB. Cheaper than selective recompute costs DSV3 on a4x (~9%) —
Hy3's expert intermediate is 1536 vs DSV3's 2048, so `moe_act` has less activation to recompute.

**V3 (biggest win)**: `cutedsl_fused_grouped_mlp` + `moe_a2a_overlap` together take **+11.2%** (744→827),
**and memory drops from 217 to 197 GB** — the fused grouped MLP eliminates the intermediate tensors.

**V4**: once cutedsl opens the TE op fuser, `full_iteration` + paged stash **work on BF16** (a combination the official recipe never covers),
for another +3.3%. Variance only **±0.04%** — the most stable of the four versions.

#### ⭐⭐ Core finding: BF16 can eat every high-performance feature of the FP8 recipe

NVIDIA puts `cutedsl` / `a2a_overlap` / `full_iteration` / paged stash **in the FP8_MX tier only**.
Measured, the whole set **works on BF16, for a cumulative +20.8%**.
The previous section's "the official BF16 recipe does not use full_iteration" describes an official **configuration choice**, **not a technical limit**.

The dependency chain (every link required — a hand-written config will always miss one):
```
cutedsl_fused_grouped_mlp=True
  └→ use_transformer_engine_op_fuser=True          (overrides.py:238)
       └→ allows moe_expert_rank_capacity_factor
            └→ allows moe_paged_stash
                 └→ only then can a full_iteration graph capture dropless MoE's variable-length tensors
```

#### V5 / V6 negative results: after full graph, memory becomes the hard constraint

| Version | Symptom | Root cause |
|---|---|---|
| V5 (MBS 2) | `OutOfMemoryError: Tried to allocate 48.00 MiB, GPU has 276.62 GiB total, 21.31 MiB free` | activations 45→90 GB plus the full-graph buffer hit the ceiling |
| V6 (GBS 4096) | memory 283 GB of 276.5 GiB available; during capture the **log freezes for 10+ minutes with the GPU spinning at 100%**, no OOM raised but no step ever emitted | **`full_iteration` captures every microbatch of the whole iteration into one graph**, so GBS 2048→4096 takes microbatches 64→128 and **doubles the graph itself** |

> ⭐ **This overturns one lesson from 07e**: on DSV3 at 256 GPUs, 07e concluded "GBS is the highest-yield knob".
> **That is simply false for Hy3 at 64 GPUs with a full graph**: (1) with PP2×VPP8×DP32, GBS 2048 already leaves only a **0.2%** bubble — there is nothing left to squeeze;
> (2) full graph converts GBS directly into memory, so raising GBS raises the graph and **hits the memory wall first**.
> **Lesson: a knob's payoff depends on where the current bottleneck is; importing another scale's tuning order wastes machine time.**

> ⭐ **The calibration factor is strongly tied to graph type**: TE graph calibrates to 1.42× (predicted 183.9 vs measured 184, a hit);
> the same config on full_iteration measures 230 GB, equivalent to **1.78×**. `mem_calc.py`'s 1.42 applies to the TE graph only.

### The 3 traps in this stretch

#### Trap H: the `moe_paged_stash` validation branch dereferences None (a Megatron-side bug)
```
TypeError: 'NoneType' object is not iterable
  transformer_config.py:1691  {"expert_fc1","moe_act"} & set(self.offload_modules)
```
`offload_modules` defaults to `None`, yet the paged-stash validation calls `set()` on it unconditionally. **Only triggers on the full_iteration path.**
Fix: `m.offload_modules = []`.

#### Trap I: `pkill -f <pattern>` kills the very `bash -c` it is running in
```bash
# ❌ the pattern "hy3_pretrain" matches this bash -c command line itself -> suicide, and the file write that follows never runs
kubectl exec POD -- bash -c 'pkill -9 -f hy3_pretrain; echo "$B" | base64 -d > /tmp/hy3_pretrain.py'
```
The symptom is deeply misleading: **you change the code, rerun, and get the exact same error**, as if the fix did nothing.
Fix: split the kill and the file write into two separate execs, and pick a pattern that does not match the command itself (use `torchrun`).
**Rule: before rerunning after a code change, `md5sum` local against remote.**

#### Trap J: the zombie CUDA contexts `pkill` leaves behind eat the next run's memory
After the V5 OOM I cleaned up with `pkill -9 -f torchrun`; rerunning V4 **immediately hit `CUDA error: out of memory`**:
```
39781, 230308 MiB   <- V5's worker, long since exited, CUDA context never released
49032,  52494 MiB   <- the new run only got what was left
```
`pkill -f torchrun` kills the launcher only; the **multiproc workers are orphaned — invisible to `ps` but their driver-side contexts are never reclaimed**.
The cleanup (no pod rebuild needed, and it sidesteps trap I):
```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | xargs -r kill -9
```
**Mandatory before every config switch in a sweep**, or leftovers get misread as "the new config OOMs".

### 🏆 Current best config (V4)

```bash
python hy3_pretrain.py \
  --num-gpus 64 --tp 1 --pp 2 --vpp 8 --ep 32 \
  --mbs 1 --gbs 2048 --seq-length 4096 \
  --cuda-graph full_iteration --cutedsl --a2a-overlap \
  --recompute-modules --recompute-granularity none \
  --mtp-layers 0 --max-steps 30
# env: NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=32 (must == EP)
```

**854 TFLOP/s/GPU · 20.97 s/step · 230 GB/GPU · 16 nodes, single NVLink domain**

**Independent reproduction** (rerun after clearing zombie memory): steady-state median **854.5**, range 854.2–856.6, **±0.14%**,
consistent with the first measurement of 854.4 — the config is reproducible, not a fluke.

Reference points (all MXFP8, not the same precision — order of magnitude only): DSV3 671B GB300 256 GPU 1658 / GB200 256 GPU 1292.
Hy3's hidden 4096 is only 57% of DSV3's, so the GEMM shapes are small and arithmetic density is naturally lower; 854 is a reasonable band.

### To do
- [x] M6: 64 GPU running → 707 baseline
- [x] M7-1: recompute off → 744 (+5.2%)
- [x] M7-2: cutedsl + a2a_overlap → 827 (+16.9%)
- [x] M7-3: full_iteration + paged stash → **854 (+20.8%)**
- [x] M7-4: MBS 2 **OOM**; GBS 4096 **hang** (under a full graph, memory is the hard constraint)
- [ ] M8: enable MTP; sweep EP 16 vs 32; BF16 vs FP8_MX head-to-head

## 10. The ablation mega-table (2026-07-26)

**27 configurations swept end to end**, run serially by [`sweep.sh`](sweep.sh):
each group auto-clears zombie CUDA contexts (rebuilding the pod if that fails) → distribute → start 16 pods → wait for steady state → collect metrics → append to [`results.csv`](results.csv).

**Measurement convention**: TFLOP/s is the median of the last 5 steady-state steps (the first step includes graph capture and is excluded); HBM is the whole-run `nvidia-smi` peak; tok/s/GPU = `GBS × seq_len / step_time / 64`;
MFU = model TFLOP/s ÷ hardware peak (**2,700 for BF16, 5,400 for FP8**).

> Baseline **A1 = 854.0 TFLOP/s** (the BF16 champion config); the `vs A1` column is the relative delta.

### Group A · single-switch ablation

Starting from the A1 champion config, flip exactly one switch at a time to isolate each contribution.

| Run | Status | TFLOP/s | vs A1 | MFU | HBM | tok/s/GPU | Step |
|---|---|---|---|---|---|---|---|
| **A1** V4_champion | ✅ | 854.0 | +0.0% | 31.6% | 225 GB | 6242 | 21.00s |
| **A2** no_cutedsl | ❌ CRASH | — | — | — | 212 GB | — | — |
| **A3** no_a2a_overlap | ✅ | 718.6 | **-15.9%** | 26.6% | 219 GB | 5253 | 24.95s |
| **A4** no_paged_stash | ❌ CRASH | — | — | — | 186 GB | — | — |
| **A5** no_router_fusion | ✅ | 843.1 | **-1.3%** | 31.2% | 226 GB | 6162 | 21.27s |
| **A6** no_permute_fusion | ✅ | 854.2 | +0.0% | 31.6% | 225 GB | 6244 | 20.99s |
| **A7** dispatcher_alltoall | ❌ CRASH | — | — | — | 0 GB | — | — |
| **A8** graph_none | ✅ | 572.5 | **-33.0%** | 21.2% | 187 GB | 4185 | 31.32s |
| **A9** graph_TE | ✅ | 827.7 | **-3.1%** | 30.7% | 193 GB | 6049 | 21.67s |
| **A10** no_force_lb | ⚠️ HANG | — | — | — | 219 GB | — | — |
| **A11** recompute_selective | ✅ | 853.5 | -0.1% | 31.6% | 225 GB | 6239 | 21.01s |

### Group B · parallelism and batch size

Chasing whether the gap between 854 and Qwen3-235B's 1360 comes from the parallel/batch configuration.

| Run | Status | TFLOP/s | vs A1 | MFU | HBM | tok/s/GPU | Step |
|---|---|---|---|---|---|---|---|
| **B1** vpp2 | ✅ | 852.1 | -0.2% | 31.6% | 243 GB | 6230 | 21.04s |
| **B2** vpp4 | ✅ | 854.0 | +0.0% | 31.6% | 231 GB | 6242 | 21.00s |
| **B3** pp4_ep16 | ✅ | 855.7 | +0.2% | 31.7% | 230 GB | 6253 | 20.96s |
| **B4** ep16 | ✅ | 882.3 | **+3.3%** | 32.7% | 241 GB | 6450 | 20.32s |
| **B5** mbs2_TEgraph | ⚠️ HANG | — | — | — | 277 GB | — | — |
| **B6** mbs2_pp4 | ❌ OOM | — | — | — | 105 GB | — | — |
| **B7** gbs4096_TEgraph | ✅ | 834.2 | **-2.3%** | 30.9% | 194 GB | 6098 | 42.99s |
| **B8** gbs1024 | ✅ | 845.7 | -1.0% | 31.3% | 223 GB | 6183 | 10.60s |

### Group C · precision

BF16 vs FP8_MX, measured against Qwen3's MXFP8 convention directly.

| Run | Status | TFLOP/s | vs A1 | MFU | HBM | tok/s/GPU | Step |
|---|---|---|---|---|---|---|---|
| **C1** fp8_mx | ✅ | 1285.9 | **+50.6%** | 23.8% | 195 GB | 9396 | 13.95s |
| **C2** fp8_mx_mbs2 🏆 | ✅ | 1360.4 | **+59.3%** | 25.2% | 276 GB | 9945 | 26.36s |

### Group D · isolating scale

Halving the layer count halves the weights, testing the hypothesis "weights crowd out memory so the batch cannot grow".

| Run | Status | TFLOP/s | vs A1 | MFU | HBM | tok/s/GPU | Step |
|---|---|---|---|---|---|---|---|
| **D1** 40layer_bf16 | ✅ | 846.4 | -0.9% | 31.3% | 123 GB | 12114 | 10.82s |
| **D2** 40layer_bf16_mbs2 | ✅ | 892.7 | **+4.5%** | 33.1% | 201 GB | 12775 | 20.52s |
| **D3** 40layer_bf16_mbs4 | ❌ OOM | — | — | — | 276 GB | — | — |
| **D4** 40layer_fp8 | ✅ | 1272.3 | **+49.0%** | 23.6% | 105 GB | 18204 | 7.20s |
| **D5** fp8_ep16_mbs2 | ⚠️ HANG | — | — | — | 277 GB | — | — |
| **D6** fp8_mbs4 | ⏭ skipped | — | — | — | — | — | — |

### Ranking each knob by measured gain

| Knob | Gain | Evidence | Note |
|---|---|---|---|
| **FP8_MX precision** | **+50.6%** | A1 854.0 → C1 1285.9 | the biggest lever, and it saves 30 GB |
| **CUDA graph** | **+44.6%** | A8 572.5 → A9 827.7 | without a graph you lose a third |
| **a2a_overlap** | **+18.8%** | A3 718.6 → A1 854.0 | MoE comm overlap — the only fusion item that matters |
| **MBS=2, unlocked by FP8** | **+5.8%** | C1 1285.9 → C2 1360.4 | unreachable under BF16 |
| MBS=2 unlocked by halving layers | +5.5% | D1 846.4 → D2 892.7 | same mechanism as above: trade memory for batch |
| EP 32 → 16 | +3.3% | A1 854.0 → B4 882.3 | only pays off paired with PP=2 |
| full_iteration vs TE graph | +3.1% | A9 827.7 → A1 854.0 | costs +32 GB |
| router_fusion | +1.3% | A5 843.1 → A1 854.0 | marginal |
| permute_fusion | **0%** | A6 854.2 ≈ A1 854.0 | absorbed by cutedsl — pure no-op |
| parallelism PP/VPP | **~0%** | B1/B2/B3 all land in 852–856 | **tuning parallelism does not buy performance** |

### full_iteration's hard dependencies (drop one and it dies)

The three CRASHes A2 / A4 / A7 jointly trace a **mandatory dependency chain**, not a set of "optional optimizations":

```
cutedsl_fused_grouped_mlp   ← turn it off (A2) and it CRASHes
  └→ use_transformer_engine_op_fuser=True      (overrides.py:238)
       └→ moe_expert_rank_capacity_factor      (fixed expert capacity)
            └→ moe_paged_stash                 ← turn it off (A4) and it CRASHes
                 └→ only then can a full_iteration graph capture dropless MoE's variable-length tensors
hybridep dispatcher         ← swap to alltoall (A7) and it CRASHes
```

### The memory wall: four attempts at MBS=2

| Attempt | Layers | Precision | Method | Result |
|---|---|---|---|---|
| V5 | 80 | BF16 | full graph | ❌ OOM |
| B5 | 80 | BF16 | fall back to TE graph to save 32 GB | ⚠️ HANG @277 GB |
| B6 | 80 | BF16 | PP4 to thin per-stage activations | ❌ OOM |
| **D2** | **40** | BF16 | **halve the layers** | ✅ **892.7** |
| **C2** | 80 | **FP8** | **halve the weights via precision** | ✅ **1360.4** |

> At 80 layers in BF16, MBS=2 is **unsolvable** — no graph swap and no parallelism change rescues it.
> Only **halving the weights** (fewer layers, or FP8) frees activation space. This is a physical memory constraint, not a tuning problem.

## 11. Why only 854 even after the full graph? — the complete attribution

### Conclusion: **because it was running BF16. Switch to FP8_MX and it is 1360, identical to Qwen3-235B's official number.**

| Config | Precision | MBS | GPUs | Model TFLOP/s | MFU | tok/s/GPU |
|---|---|---|---|---|---|---|
| A1 Hy3 champion | BF16 | 1 | 64 | 854.0 | **31.6%** | 6,242 |
| C1 Hy3 | **FP8_MX** | 1 | 64 | 1,285.9 | 23.8% | 9,396 |
| **C2 Hy3** | **FP8_MX** | **2** | **64** | **1,360.4** | **25.2%** | **9,945** |
| *Qwen3-235B (official reference)* | *MXFP8* | *2* | ***256*** | *1,360* | *25.2%* | *—* |
| *DSV3 671B (official reference)* | *MXFP8* | *1* | *256* | *1,658* | *30.7%* | *—* |

**C2's 1360.4 / MFU 25.2% coincides exactly with Qwen3-235B's 1360 / 25.2% on 256 GPUs — and we used only 64.**

### Two causes, and they are coupled

**① The precision convention (dominant, +50.6%)**
The numerator of Model TFLOP/s is the model's mathematical FLOPs (precision-independent); the denominator is **the hardware peak at that precision**.
GB300 peaks at 2,700 BF16 and 5,400 FP8. Running BF16, the arithmetic ceiling is half to begin with.

**② The memory FP8 frees unlocks MBS=2 (+5.8%)**
This is the crucial link, and it confirms the "weights are too big to grow the batch" diagnosis:

| Attempt | Layers | Precision | Method | HBM | Result |
|---|---|---|---|---|---|
| V5 | 80 | BF16 | full graph | — | ❌ OOM |
| B5 | 80 | BF16 | fall back to TE graph, saving 32 GB | 277 GB | ⚠️ HANG |
| B6 | 80 | BF16 | PP4 to thin each stage | — | ❌ OOM |
| **D2** | **40** | BF16 | **halve the layers** | 201 GB | ✅ 892.7 (+5.5%) |
| **C2** | 80 | **FP8** | **halve the weights via precision** | 276 GB | ✅ 1,360.4 (+5.8%) |

At 80 layers in BF16, **all three plays for MBS=2 fail**. Only halving the weights (fewer layers, or FP8) frees the activation space.
**This is not a tuning problem, it is a physical memory constraint.**

The causal chain:
```
weights occupy memory → squeeze out activation space → MBS can only be 1 → small GEMM shapes → low per-GPU arithmetic density
     ↑                                                                                              ↓
     └──── switch to FP8 (halving weights) or cut layers; either breaks the loop ──────────────────┘
```

### A bonus finding from group D: TFLOP/s is almost independent of depth

| Layers | Precision | TFLOP/s | HBM | tok/s/GPU |
|---|---|---|---|---|
| 80 | FP8 | 1,285.9 | 195 GB | 9,396 |
| 40 | FP8 | 1,272.3 | **105 GB** | **18,204** |
| 80 | BF16 | 854.0 | 225 GB | 6,242 |
| 40 | BF16 | 846.4 | **123 GB** | 12,114 |

**At the same precision, 80 layers vs 40 layers differ by only 1.1% (FP8) / 0.9% (BF16) in TFLOP/s.**
Per-GPU arithmetic density is set by **hidden_size × precision × batch**, and is **independent of model depth**.
Depth only determines total work and memory, not efficiency — which is why reduced-layer debugging is trustworthy (07e's scale-in methodology reproduces here).

Meanwhile the BF16→FP8 gain at the two depths is **+50.6% (80 layers) and +50.3% (40 layers)** —
essentially identical, which says **this 50% is a stable architecture-level gain, not an artifact of one configuration**.

### ⚠️ Correction: my earlier "FP8 is only ±5% on MoE", derived from GB200 data, was wrong

Section 3 cited this repo's GB200 measurements to assert that "on MoE, FP8 is only −5%~+5% versus BF16, not worth introducing during bring-up",
and set the first run to BF16 on that basis. **That conclusion does not hold on GB300**:

| Platform / software stack | BF16 → FP8 |
|---|---|
| GB200 old stack (no cutedsl) | **−5%** (DSV3 12L: 527 → 503) |
| GB200 + CUDA graph | +4.5% (DSV3 32L: 928 → 970) |
| **GB300 + cutedsl + full graph** | **+50.6%** (Hy3 80L: 854 → 1,286) |

**The decisive difference is `cutedsl_fused_grouped_mlp`**: on the old stack, FP8 quantize/dequantize overhead inside the grouped GEMM
ate the Tensor Core dividend, so no gain was measurable. On GB300 the fused kernel removes that overhead and FP8's arithmetic advantage finally materializes.

> **Lesson**: performance conclusions **do not extrapolate across hardware generations or software stacks**. Using GB200 data to pick GB300's configuration was directionally wrong.
> The right method is to **validate with a small real run before fixing the baseline**, not to cite another platform's numbers.
> This is the same class of error as "judge architectural lineage from the config, not the label": **substituting a secondhand conclusion for a firsthand measurement**.

### The corrected optimal config

```bash
python hy3_pretrain.py \
  --num-gpus 64 --tp 1 --pp 2 --vpp 8 --ep 32 \
  --mbs 2 --gbs 4096 --seq-length 4096 \
  --precision fp8_mx \
  --cuda-graph full_iteration --cutedsl --a2a-overlap \
  --recompute-modules --recompute-granularity none \
  --mtp-layers 0 --max-steps 30
# env: NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=32
```

**1,360.4 TFLOP/s/GPU · MFU 25.2% · 26.36 s/step · 276 GB/GPU · 9,945 tok/s/GPU**

> At 276 GB memory is already near the 288 GB ceiling, so **there is no headroom to stack EP16 or MTP on top** (D5 measured: stacking all three hangs).
> To run MTP as well, fall back to MBS=1 (the C1 config, 1,285.9) or scale out to 128 GPUs.

**The BF16 optimum** (if precision-alignment requirements force BF16): **B4 = EP16, 882.3 TFLOP/s / MFU 32.7%**.

---

## 12. FP8 training-quality alignment check (2026-07-26 05:35 ✅ passed)

A 50% speedup is worthless if it costs training quality. This section validates FP8_MX ≡ BF16 with a **controlled experiment**.

### Experimental design

Executed by [`loss_align.sh`](loss_align.sh). Two runs, **identical except for precision**:

| Variable | Setting |
|---|---|
| Parallelism | TP1 / PP2 / VPP8 / **EP16** / MBS1 / GBS2048 / seq4096 |
| Perf switches | full_iteration graph + cutedsl + a2a_overlap + paged stash, no recompute |
| Data | mock (same generator, same seed) |
| Steps | 20 each |
| **Only difference** | `--precision bf16` vs `--precision fp8_mx` |

> **Where to read it: `yw-a-15`.** `lm loss` prints only on the **last pipeline stage**, which with PP=2 across 16 pods is pod 15.
> You will not find loss by grepping `yw-a-0` where rank 0 lives — fell into that once, noting it down.

### Step-by-step loss comparison

| step | BF16 loss | FP8_MX loss | Absolute Δ | Relative Δ |
|---|---|---|---|---|
| 1 | 12.530730 | 12.529190 | -0.001540 | 0.0123% |
| 2 | 12.530780 | 12.529230 | -0.001550 | 0.0124% |
| 3 | 12.396570 | 12.403350 | +0.006780 | 0.0547% |
| 4 | 11.356320 | 11.359070 | +0.002750 | 0.0242% |
| 5 | 11.020130 | 10.998600 | -0.021530 | 0.1954% |
| 6 | 10.415350 | 10.413700 | -0.001650 | 0.0158% |
| 7 | 9.868304 | 9.863834 | -0.004470 | 0.0453% |
| 8 | 9.258566 | 9.255017 | -0.003549 | 0.0383% |
| 9 | 8.757781 | 8.755513 | -0.002268 | 0.0259% |
| 10 | 8.438774 | 8.435921 | -0.002853 | 0.0338% |
| 11 | 8.280344 | 8.276585 | -0.003759 | 0.0454% |
| 12 | 8.232012 | 8.223855 | -0.008157 | 0.0991% |
| 13 | 8.239738 | 8.231593 | -0.008145 | 0.0989% |
| 14 | 8.235600 | 8.229311 | -0.006289 | 0.0764% |
| 15 | 8.220032 | 8.218982 | -0.001050 | 0.0128% |
| 16 | 8.192741 | 8.194827 | +0.002086 | 0.0255% |
| 17 | 8.178925 | 8.184619 | +0.005694 | 0.0696% |
| 18 | 8.171684 | 8.175358 | +0.003674 | 0.0450% |
| 19 | 8.166067 | 8.167156 | +0.001089 | 0.0133% |
| 20 | 8.159149 | 8.157866 | -0.001283 | 0.0157% |

### Verdict

| Criterion | Threshold | Measured | Verdict |
|---|---|---|---|
| Max relative deviation | < 1% | **0.1954%** | ✅ far inside the threshold |
| Mean relative deviation | — | **0.0480%** | ✅ |
| Does deviation grow with step count | must not | first 10 steps 0.0458% → last 10 steps 0.0502% | ✅ no growth |
| NaN iterations | 0 | **0** | ✅ |
| Skipped iterations | 0 | **0** | ✅ |
| Final-step loss | should agree | BF16 **8.1591** vs FP8 **8.1579** | ✅ Δ 0.0013 |
| Final-step grad norm | same order | FP8 **0.319** (steady 0.315–0.322) | ✅ no gradient anomaly |

**The sign of the deviation oscillates randomly** (6 positive, 14 negative across 20 steps) — **not a one-directional systematic drift**.
That is the signature of floating-point noise, not quantization bias. If FP8 were genuinely hurting training, the deviations would share a sign and accumulate.

The largest deviation, 0.1954%, lands on **step 5**, right in the **steepest stretch** where loss plunges from 12.5 to 8.2;
that region is maximally sensitive to any perturbation, so a peak there is expected — and it immediately falls back into the 0.02–0.09% band.

### Conclusion: **FP8_MX is safe for Hy3 production training**

Within a 20-step window the FP8 and BF16 trajectories agree at the level of numerical noise, so **the +50.6% throughput is a net gain with no quality cost**.
The FP8 optimum from §11 (1,360.4 TFLOP/s) **can be adopted directly**.

### The limits of this validation (do not over-read it)

1. **20 steps only — this proves short-term numerical stability, not long-run convergence equivalence.** FP8's risk usually shows up mid-to-late in training
   (once gradients shrink, the relative quantization error rises). For production pretraining, run at least **1000+ steps** head-to-head and monitor
   `grad norm` and loss-spike frequency.
2. **Mock data is not real corpus.** Real data has a longer-tailed token distribution and wider activation dynamic range — harder on FP8.
3. **This run had `moe_router_force_load_balancing` on** (the benchmark convention). Under real routing, expert load skews and hot experts see larger
   activation magnitudes, so FP8 overflow risk is higher than measured here.
4. For production, keep the common insurance policy **`first_last_layers_bf16`** (first and last layer in BF16);
   this run used Bridge's default `bf16_with_mxfp8_mixed()` strategy.

---

## 13. The 256-GPU cross-domain test plan (design doc, written before the run)

> **Status: design complete, execution pending.** This section nails down topology, environment, memory budget, ablation matrix and pass criteria
> *before* starting — to avoid the 64-GPU round's habit of designing while running.

### 13.1 Objectives

1. Rerun the ablation at **256 GPU / 4 NVLink domains** and find the optimum at that scale.
2. Answer: **can 256 GPUs beat 64 GPUs' 1,360.4 TFLOP/s?** Or is it only linear scaling of aggregate throughput?
3. Quantify **the cost of cross-domain communication** — at 64 GPUs EP/PP/DP all stay on intra-domain NVLink; at 256 GPUs PP/DP must cross domains over CX-8 RDMA.

### 13.2 Topology: 4 pools × 16 nodes

Measured cluster state (2026-07-26 10:12 HKT, 8 GB300 pools):

| Pool | Nodes | In use | Free | kubelet | subblock |
|---|---|---|---|---|---|
| gb300-pool-0002 | 18 | 17 | 1 | 4447000 | e9e26a9c9da3 |
| gb300-pool-0006 | 18 | 17 (inference team) | 1 | 4447000 | ee18edff617d |
| gb300-pool-0009 | 18 | 0 | **18** | 4447000 | a5dd4eb244e5 |
| gb300-pool-0013 | 18 | 0 | **18** | 4447000 | 2735522da210 |
| gb300-pool-0014 | 16 | 0 | **16** | 4447000 | 0c242ce6c059 |
| gb300-pool-0015 | 18 | 16 (this project's 64-GPU work) | 2 | 4447000 | 10270c36001d |
| gb300-pool-0016 | 18 | 0 | **18** | 4447000 | d58388dca90c |
| gb300-pool-0017 | 18 | 0 | **18** | 4447000 | 572f6bfac269 |

**The four chosen domains** (16 nodes each, 2 held as hot spares):

| Domain | Pool | ComputeDomain | Note |
|---|---|---|---|
| A | **gb300-pool-0015** | `yw-cd-a` | reused after releasing the current 64-GPU experiment |
| B | **gb300-pool-0016** | `yw-cd-b` | fully free, 18 nodes |
| C | **gb300-pool-0017** | `yw-cd-c` | fully free, 18 nodes |
| D | **gb300-pool-0013** | `yw-cd-d` | fully free, 18 nodes (has DRAM correctable-ECC warnings; soft errors do not affect correctness) |

**Backup pools**: `0014` (16 nodes, zero slack) and `0009` (18 nodes, **outside the delivery scope — ask Chris before touching it**).

> **Why 16 nodes per pool instead of 18**: **auto-repair is off** in this cluster, so bad nodes do not self-heal.
> Keeping 2 same-subblock hot spares means a failure is a label change, avoiding 07e's "requisition and migrate the whole pool" situation.
>
> **pool = subblock = NVLink domain, one to one** (all 8 pools have distinct subblock IDs; verified).
> **Each domain needs its own ComputeDomain** — never span 4 pools with one CD, because an IMEX channel can only be established inside a single NVLink domain.

### 13.3 ⚠️ Cross-domain MNNVL environment variables (the single most important item, from Chris's prior experience)

**Single-domain (64 GPU) and cross-domain (256 GPU) NCCL settings differ; copying one to the other errors out immediately.**

| env | Single domain (what our 64 GPUs used) | **Cross domain (mandatory at 256)** | Purpose |
|---|---|---|---|
| `NCCL_MNNVL_ENABLE` | unset (takes the GIB default of **2**) | **`0`** | **MNNVL must be off across domains**, otherwise NCCL tries to reach across domains over NVLink and fails |
| `NCCL_CUMEM_ENABLE` | unset (default 1) | **`0`** | cuMem only serves MNNVL; turn it off along with it |
| `USE_MNNVL` | `1` | **`1` (unchanged)** | this one is for the **HybridEP dispatcher**, telling it NVLink is available inside a domain |
| `NVLINK_DOMAIN_SIZE` | `72` | `72` | GB300 NVL72 |
| `NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN` | `= EP` | `= EP` (and **EP ≤ 64** — must not cross domains) | changing EP means changing this too |

> **Mnemonic: `NCCL_MNNVL_ENABLE=0` + `USE_MNNVL=1`.**
> The two variables serve different layers: the former governs NCCL collectives (cross-domain over RDMA), the latter governs the HybridEP dispatcher (intra-domain over NVLink).
> Turning only one off, or turning both off, causes trouble. Basis: [`08-multi-domain/README.md:73`](../../a4x-max/08-multi-domain/README.md) states plainly that "MNNVL must be disabled across domains".
>
> ⚠️ **Known trap**: [`a4x/07-megatron-training/README.md:534`](../../a4x/07-megatron-training/README.md) records that
> "even with `NCCL_MNNVL_ENABLE=0` set you still get CUDA error 801, because **the GIB script internally overwrites it back to 2**".
> **You must grep the effective value out of the rank-0 log** — do not trust what the script says.

**EP constraint**: HybridEP's all-to-all can only stay within one NVLink domain. With 4 domains × 64 GPU, **EP ≤ 64**,
and the EP ranks must land in the same subblock. EP=16/32, available at 64 GPUs, remain available at 256 (as an intra-domain subset).

### 13.4 Memory budget (256 GPU vs 64 GPU)

Using [`mem_calc.py`](mem_calc.py)'s formulas with the full-graph factor of **1.78×**:

| N | PP | EP | MBS | Precision | Weights | Grads | Optimizer | Activations | Naive | ×1.78 | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 2 | 32 | 1 | FP8 | 8.2 | 16.4 | **51.5** | 45.2 | 121.3 | 215.9 | ✅ |
| 64 | 2 | 32 | 2 | FP8 | 8.2 | 16.4 | **51.5** | 90.3 | 166.4 | 296.2 | ❌ (measured 276 GB runs fine — the formula is conservative) |
| **256** | 2 | 32 | 2 | FP8 | 8.2 | 16.4 | **12.9** | 90.3 | 127.8 | **227.5** | ✅ comfortable |
| **256** | 4 | 32 | 2 | FP8 | 4.1 | 8.2 | **12.9** | 93.0 | 118.2 | **210.3** | ✅ most comfortable |
| 256 | 2 | 32 | **4** | FP8 | 8.2 | 16.4 | 12.9 | **180.6** | 218.1 | 388.3 | ❌ OOM |

#### ⭐ Insight 1: adding GPUs does **not** directly unlock a larger MBS

**Activation memory is independent of GPU count.** In 1F1B steady state each stage holds `L layers × MBS` worth of activations
(`L/PP` layers × `PP` in-flight microbatches) — it changes with neither PP nor DP.
Adding GPUs only thins the **optimizer state** (`294.9B × 12B / N`: 51.5 GB at 64 GPUs → **12.9 GB** at 256).
The ~39 GB saved is nowhere near the +90 GB of activations MBS=4 demands.

**Two reference measurements confirm this**:

| Model | GPUs | TP/PP/VPP/EP | MBS | GBS | TFLOP/s | Memory |
|---|---|---|---|---|---|---|
| DeepSeek V3 671B | 256 | 1 / **2** / **8** / 32 | **1** | 4096–15360 | 1,658 | — |
| Qwen3-235B | 256 | 1 / **4** / **2** / 32 | **2** | **8192** | 1,360 | **277/288 (96%)** |
| **Hy3 (ours)** | **64** | 1 / 2 / 8 / 32 | 2 | 4096 | **1,360** | 276 |

> Qwen3 measured at 256 GPUs: **MBS=4 OOMs even with VPP=4** (276 GB used and still 288 MB short), and **VPP=8 OOMs during capture**.
> That is: **both references are stuck at MBS ≤ 2 on 256 GPUs**, the same ceiling we hit on 64.
>
> **And we already matched, on 64 GPUs, the 1,360 Qwen3 needed 256 GPUs to reach.**

#### ⭐ Insight 2: GBS must scale with GPU count, or the bubble gets worse

`microbatch/DP-rank = GBS / (MBS × DP)`, and that number sets the pipeline bubble. **Copying the 64-GPU GBS drops it 4×**:

| GPUs | PP | DP | GBS | MBS | microbatch/rank | Assessment |
|---|---|---|---|---|---|---|
| 64 | 2 | 32 | 4,096 | 2 | **64** | baseline |
| 256 | 2 | **128** | 4,096 | 2 | **16** | ⚠️ straight copy → bubble grows 4× |
| 256 | 2 | 128 | 8,192 | 2 | 32 | still too few |
| **256** | **2** | 128 | **16,384** | 2 | **64** | ✅ equivalent to baseline |
| **256** | **4** | **64** | **8,192** | 2 | **64** | ✅ **Qwen3's solution** |

**Qwen3 uses PP=4 to squeeze DP down to 64**, so GBS=8192 already preserves 64 microbatches
without pushing GBS to 16384 (too large a GBS lengthens the step and perturbs convergence hyperparameters).
**That is what "256 GPUs need a different parallel plan" actually means** — not simply porting the 64-GPU config over.

#### ⭐⭐ Insight 3: TP is a "memory compromise", not a "performance optimization" — but Hy3 is exactly the borderline case worth testing

**TP + sequence parallel splits activations across TP**, the only lever that can break the MBS=2 ceiling:

| TP | PP | EP | DP | MBS | Precision | Activations | Naive | ×1.78 | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 32 | 128 | 2 | FP8 | 90.3 | 127.8 | 227.5 | ✅ |
| 1 | 2 | 32 | 128 | **4** | FP8 | **180.6** | 218.1 | 388.3 | ❌ OOM |
| **2** | 2 | 32 | **64** | **4** | FP8 | **90.3** | 121.7 | **216.7** | ✅ fits |
| **2** | 4 | 32 | 32 | **4** | FP8 | 93.0 | 115.1 | **204.9** | ✅ most comfortable |

**But the official guidance explicitly discourages reaching for TP.** Per the five guidelines in §9.1 of the Megatron-Core MoE paper
[*Scalable Training of Mixture-of-Experts Models with Megatron Core* (arXiv 2603.07685)](https://arxiv.org/html/2603.07685v2):

| Guideline | Key text | What it means for Hy3 |
|---|---|---|
| **1. Minimize model parallelism, maximize DP** | "Keep TP/EP/PP/CP as small as possible while avoiding OOM. Model parallelism introduces communication overhead that hurts performance." | **TP is a cure for OOM, not a way to go faster** |
| **2. EP × TP must fit inside the NVLink domain** | "Ensure EP×TP fits within the NVLink Domain" | TP2 × EP32 = 64 ≤ NVL72 ✅ **satisfied** |
| 3. Scale across nodes with PP | "prefer PP over expanding TP/EP across nodes" | spanning 4 domains should lean on PP, not TP |
| **4. Prefer EP over TP in the expert layers** | "Better GEMM efficiency / Lower communication / Simpler graph"; on Mixtral, **EP8×TP1 beats EP4×TP2** | **must set `ETP=1`** (parallel folding: TP splits attention only, not the experts) |
| 5. Use CP for long sequences | — | not applicable at seq 4096; revisit if testing 8192 long context |

**The strongest counter-evidence is the paper's own example**:
> "GB200's 192 GB per GPU (vs. H100's 80 GB) **allows TP1/PP4 instead of TP2/PP8**"

**The moment memory grows, the official guidance lowers TP** — which nails down that TP is a memory compromise. GB300 has 288 GB, so by the same logic it should be TP=1.

##### Then why test TP at all? — Hy3 lands on a boundary the guidance does not cover

The guidance assumes "if memory suffices, do not enable TP", implicitly presuming **the batch is already large enough and the GEMM shapes are saturated**. Hy3 does not satisfy that premise:

1. **hidden=4096 is narrow to begin with** (DSV3 is 7168), so the GEMM shapes are inherently small and arithmetic density suffers;
2. **MBS is pinned at 2** (all three plays failed at 64 GPUs, and both 256-GPU references are also capped at ≤2);
3. so "spend TP communication to double MBS" becomes a **possibly worthwhile trade** — precisely the case the guidance does not discuss.

**GQA is what makes the trade feasible on Hy3** (Chris's key observation):
Hy3 is GQA with `num_query_groups=8`, so TP ∈ {1,2,4,8} all divide it and split cleanly;
MLA models (DSV3) are constrained by `q_lora_rank`/`kv_lora_rank`, where TP splitting is far less natural than in GQA.
**So "try TP" is a valid question for Hy3 and not necessarily for DSV3** — one reason neither reference tried it.

##### Prediction and criteria

By guidelines 1 and 4, **my prediction is that TP2 is a net loss**: TP costs two all-reduces per layer (two forward, two backward),
and the GEMM gain from MBS 2→4 may not be large enough at hidden=4096.
But this is **a boundary case that must be measured**, not reasoned about. Criteria:

- **E4 (TP2+MBS4) > E1 (TP1+MBS2)** → the trade pays, the guidance does not apply at this boundary, and that is a valuable new finding;
- **E4 < E1** → the guidance holds; record the negative result "on Hy3, TP's communication cost exceeds the MBS gain" with a quantified delta.

> Either outcome is worth having. **A non-result must go in the document too** — otherwise the next person will ask "why not try TP" all over again.

##### ⚠️ The paper's reported numbers disagree with this repo's measurements — note it

The paper's abstract claims **1,233/1,048 TFLOPS/GPU for DeepSeek-V3-685B and 974/919 TFLOPS/GPU for Qwen3-235B** on GB300/GB200.
This repo's own 07d/07e measurements on GB300 are DSV3 **1,658** and Qwen3-235B **1,360**, both markedly higher.

**The two use different conventions; do not mix them** (the abstract does not state precision / sequence length / GBS — it may be BF16 or a different workload).
This plan's reference baselines are always **this repo's same-convention measurements** (Qwen3-235B GB300 256 GPU MXFP8 = 1,360);
the paper is cited for its **methodological guidelines** only, not as a performance reference.

### 13.5 Ablation matrix (group E, 256 GPU)

Designed around the three insights above. **Baseline E1 = Qwen3's own 256-GPU parallel plan** (a direct comparison against its 1,360).

| # | Run | Config | Hypothesis / purpose |
|---|---|---|---|
| **E1** | **Qwen3-equivalent baseline** | PP4 VPP2 EP32 MBS2 **GBS8192** FP8 | cross-domain baseline, directly comparable to Qwen3's 1,360 |
| E2 | our 64-GPU optimum, ported | PP2 VPP8 EP32 MBS2 **GBS16384** | the other route to preserving 64 microbatches |
| E3 | GBS not scaled (negative control) | PP2 VPP8 MBS2 **GBS4096** | **measure how much the bubble degrades when microbatches drop to 16** |
| **E4** | **TP2 + SP + MBS4** 🔑 | **TP2 + `sequence_parallel` + `ETP=1`** PP2 EP32 MBS**4** GBS16384 | **the core hypothesis: split activations with TP to buy MBS=4** (violates guideline 1 — deliberately probing the boundary) |
| **E5** | TP2 + PP4 + MBS4 | TP2 PP4 EP32 ETP1 MBS4 GBS8192 | the most memory-frugal combination; can MBS go higher still |
| E5b | **TP2 + MBS2** (control) | TP2 PP2 EP32 MBS2 GBS8192 | **isolate TP's own communication cost** (without the MBS benefit) |
| E6 | TP2 + MBS8 | TP2 PP4 EP32 MBS8 | probe the MBS ceiling |
| E7 | recompute + MBS4 (no TP) | PP4 VPP2 MBS4 + selective recompute | buy MBS with "nearly free" recompute (per A11) |
| E8 | EP16 | E1 + EP16 | EP16 gave +3.3% at 64 GPUs — does it hold across domains |
| E9 | EP64 (a full domain) | E1 + EP64 | 07e found EP64 worse on DSV3; unverified for Hy3 |
| E10 | VPP sweep | E1 + VPP4 / VPP8 | Qwen3 measured VPP8 OOM; Hy3 has fewer layers (80 vs 94) so it may fit |
| E11 | BF16 control | E1 + BF16 | compare against 854 / 882 at 64 GPUs |
| E12 | MTP=1 | E1 + MTP | not enough memory at 64 GPUs, worth trying at 256 |
| **E13** | **MNNVL negative control** | do not set `NCCL_MNNVL_ENABLE=0` | **empirically confirm the cross-domain lesson in §13.3 really does error out** |

> **E4 is the centerpiece of this round**: if TP2+MBS4 pushes TFLOP/s past 1,360,
> that proves **256 GPUs buy more than 4× aggregate throughput — they raise per-GPU efficiency too**, which is the answer to "what are 256 GPUs for".
> If it does not, that is a valuable negative result: Hy3's per-GPU ceiling on GB300 is set by the GEMM shape at hidden=4096, and adding GPUs only buys volume.

### 13.6 Pass criteria and comparison conventions

| Metric | How collected | Comparison against 64 GPU |
|---|---|---|
| **Model TFLOP/s/GPU** | median of the last 5 steady-state steps | directly comparable (already per-GPU) |
| **MFU** | ÷ 2,700 (BF16) or 5,400 (FP8) | directly comparable |
| **Aggregate throughput** | `TFLOP/s × 256` vs `× 64` | 4× linear is ideal |
| tokens/s/GPU | `GBS × seq / step / N` | directly comparable |
| HBM peak | whole-run `nvidia-smi` max | should be ~39 GB lower than 64 GPU (optimizer) |
| **Cross-domain scaling efficiency** | `TFLOP/s@256 ÷ TFLOP/s@64` | **>95% is excellent, <90% means cross-domain RDMA is the bottleneck** |

### 13.7 Risks and mitigations

| Risk | Mitigation |
|---|---|
| The GIB script overwrites `NCCL_MNNVL_ENABLE` back to 2 | grep the effective value from the rank-0 log; if needed use `NCCL_CONF_FILE` or re-export after the GIB script |
| 4-CD IMEX clique deadlock (hit in 07e) | one CD per pool; do not churn repeatedly; if stuck, move the whole pool to backup 0014/0009 |
| Zombie CUDA contexts leaking between experiments | reuse `sweep.sh`'s cleanup: verify memory returns to zero, rebuild the pod if it does not |
| Parallel `kubectl exec` across 64 pods throttled by konnectivity (hit in 07e, triggers above 16 pods) | switch to SSH fanout from pod-0, or batch with `-P 16` |
| A node in some domain drops out | 2 hot spares per pool; swap the label |

### 13.8 Breaking down the startup timeline (a new requirement this round)

The 64-GPU round observed that **it takes ~10 minutes from launch to the first steady-state number**, with no idea where the time goes.
This round every experiment must produce a **phase-by-phase breakdown**.

#### Why it could not be broken down before: the logs have no timestamps

The phase markers Megatron / Bridge print (`Capture CUDA graph for training`, `done with setup`, …)
**carry no time themselves**, and the container has **no `ts` from moreutils** (only `stdbuf`).
So first build a line-level timestamper with [`timeline.py --stamp`](timeline.py) (pure stdlib, line-buffered):

```bash
python3 /tmp/hy3_pretrain.py <args> 2>&1 | python3 /tmp/timeline.py --stamp
# every line becomes:  [+  123.456] <original content>
```

Then parse it into a phase table:

```bash
python3 timeline.py --parse run.log
```

#### The phases (based on markers actually printed; anything not matched shows "—" rather than being invented)

| Phase | Trigger marker | Suspected cost |
|---|---|---|
| Process start | first line of output | torchrun rendezvous (longer with 64 pods) |
| Python import | `Failed to import Triton kernels` / `nixl_utils` / `modelopt` | torch + TE + vLLM + modelopt are heavy packages; **this stack is known to be slow just to import** |
| HF config fetch | `huggingface` / `torch_dtype.*deprecated` | the qwen3 skeleton fetches config from HF (**slower still if the cache is empty after a pod rebuild**) |
| NCCL init | `NCCL version` | `torch.distributed` init + NCCL bootstrap; **slower across 4 domains than in one** |
| Model construction | `number of parameters on (tensor, pipeline)` | GPTModel instantiation + 295B weight allocation |
| Optimizer construction | `Setting up optimizer with config` | distributed optimizer master weights / momentum allocation |
| DDP / gradient buffer | `Using reduce-scatter for gradient reductions` | large `param_and_grad_buffer` allocation |
| Setup done | `done with setup` | dataloader + rerun state |
| Entering the training loop | `Starting training loop` | — |
| **CUDA graph capture** | `Capture CUDA graph for training` → `CUDA graph capture done` | **prime suspect**: first step 121 TFLOP/s at 64 GPU, Qwen3's first step 236 s |
| First steady step | `Step Time :` | — |

#### Circumstantial evidence so far (not yet precisely split — that is this round's job)

- 64 GPU A1: first step **121.3 TFLOP/s** (steady 854), so capture dragged the first step down ~7×.
- 07d Qwen3 256 GPU: "graph capture (step 1 **~236 s**) → settling (step 2 ~50 s) → steady 14.26 s".
- 64 GPU V6 (GBS 4096 + full graph): during capture the **log froze for 10+ minutes with the GPU spinning at 100%** — capture time grows with microbatch count.

**Hypothesis: capture dominates startup time and is proportional to `GBS/(MBS×DP)`** (a full graph captures every microbatch of the whole
iteration into one graph). E2/E3 (the GBS sweep) test exactly this — if it holds, **a large GBS costs not just memory but startup time**.

#### Deliverables

One phase table per experiment, finally rolled up into a "config × phase duration" comparison answering:
1. Of the 10-minute startup, how much is import / NCCL / model build / capture?
2. How much slower is **NCCL init** across 256 GPUs in 4 domains than 64 GPUs in one?
3. Does capture time grow linearly with GBS / microbatch count?
4. Which costs are **one-off** (import, HF cache) and which are **paid on every restart** (NCCL, capture)?

---

### 13.3.1 ⭐ Measured correction: cross-domain **no longer needs** a manual `NCCL_MNNVL_ENABLE=0` (E13 evidence, 2026-07-26)

§13.3 wrote, from prior experience, "cross-domain requires `NCCL_MNNVL_ENABLE=0` or it errors out", and designed E13 as a **negative control** to demonstrate it.
**The negative control did not fail — it ran fine without the setting.**

#### E13 measurement (deliberately setting no MNNVL variables)

| Metric | Value |
|---|---|
| Status | ✅ **OK, zero NCCL errors** |
| **TFLOPS** | **1,267.3** Model TFLOP/s/GPU |
| **MFU** | **23.5%** (against the FP8 peak of 5,400) |
| **Throughput** | **9,263** tokens/s/GPU (~2.37 M tokens/s aggregate) |
| Step time | 14.15 s |
| HBM peak | 272 GB |
| Total startup | 350.1 s |
| graph capture | 26.6 s |
| Config | 256 GPU / 4 domains · TP1 PP4 VPP2 EP32 · MBS2 GBS8192 · FP8_MX · full graph + cutedsl + a2a overlap |

**The effective values recorded in the log** (the script prints them at every launch, to catch GIB overwriting them silently):

```
### MNNVL effective values: NCCL_MNNVL_ENABLE=<unset> NCCL_CUMEM_ENABLE=<unset> USE_MNNVL=1
```

Confirmed genuinely unset, not set to 0 somewhere else; `/usr/local/gib/configs/nccl.a4xmax.conf` contains no MNNVL entries either.

#### Why this contradicts prior experience

`NCCL_MNNVL_ENABLE`'s default of **2 is "auto"** — NCCL detects the NVLink domain boundary itself,
using MNNVL inside a domain and falling back to RDMA across domains. **Auto mode should not fail by design**; the old need to set 0 by hand was working around a bug of the time.

In this environment (**GKE + GIB + one DRA ComputeDomain per subblock + NCCL 2.30.4**) auto mode works correctly.
Both prior error reports come from different situations:

| Source | Situation | Note |
|---|---|---|
| [`a4x-max/08-multi-domain:73`](../../a4x-max/08-multi-domain/README.md) | "MNNVL must be disabled across domains" | that table is written mainly for **bare nccl-test** runs |
| [`a4x/07-megatron-training:534`](../../a4x/07-megatron-training/README.md) | CUDA error 801 | **self-built K8s + Rocky + NVIDIA 580**, and it is a "**errors out even when set to 0**" failure case, not a "set it and it works" case |

**The strongest self-consistent corroboration**: when 07e successfully ran DSV3 on 256 GPUs across 4 domains, it used `USE_MNNVL=1` and **did not set** `NCCL_MNNVL_ENABLE` —
exactly like E13. So this repo's own prior success never depended on the variable either.

#### Conclusion and recommendation

- **In this environment (GKE/GIB/DRA/NCCL 2.30.4), cross-domain training needs no manual `NCCL_MNNVL_ENABLE=0`**; auto mode suffices.
- Later experiments still set `=0` explicitly, forming an A/B against E13; **if the two show no performance difference, the variable can be dropped entirely here**.
- **For self-built clusters / older NCCL / bare nccl-test, still follow §13.3** — this conclusion is verified only for the hardware/software combination above.
- **Set it or not, always print the effective value into the log** — that operational discipline stays, because GIB overwriting variables really has happened.

### 13.9 Execution order

1. Release the existing 64-GPU experiment (the `hy3=true` label + pods)
2. Label 16 nodes in each of the 4 pools (`hy3a/hy3b/hy3c/hy3d`, to avoid colliding with old labels)
3. Apply the 4-CD pod-pool YAML, confirm 64/64 Running and 4 CDs Ready
4. **Run the E12 negative control first** to demonstrate the MNNVL lesson (fails fast, a few minutes)
5. Then run the E1 baseline to confirm cross-domain works at all
6. Automate the E2–E11 sweep
7. Parse the startup timeline for every experiment (`timeline.py --parse`)
8. Tabulate, attribute, compare against 64 GPUs, and commit

---

## 14. Porting HYV3Bridge: letting Megatron eat the official weights directly (2026-07-26 ✅ fully verified)

### 14.1 Why this is mandatory

Every performance experiment in §1–§13 used the **Qwen3-235B skeleton recipe with hand-overridden Hy3 hyperparameters** (that `HY3 = dict(...)` in `hy3_pretrain.py`).
That is **fine for performance** — the arithmetic depends only on shapes, not on where the weights came from. But it has one fatal limitation:

> **It can only build a randomly initialized model; it cannot load the real `tencent/Hy3-Base` weights.**

And loading real weights is the prerequisite for SFT / continued pretraining. So before doing SFT, we must settle the question of whether Megatron recognizes the Hy3 architecture at all.

### 14.2 Version survey: the official releases contain no Hy3

| Version | Commit | Has `hy_v3`? | Note |
|---|---|---|---|
| In the container | `0.5.0+fcbb6031` | ❌ | the one we run training on |
| **v0.5.1** (latest release, 2026-07-21) | — | ❌ | `gh api .../contents/.../hy_v3?ref=v0.5.1` returns **404** |
| **main** (2026-07-25, `7b3c40b7`) | — | ✅ | only 4 files involved |

What v0.5.1 adds over v0.5.0 is Nemotron 3 Ultra, the DSV4-Pro MXFP8 GB300 recipe, and CuTeDSL / full-graph improvements for DSV3 and Qwen3 — **nothing to do with Hy3**.
So "upgrade to the latest release" is not a path; the only option is a single-file port from main.

> **One of v0.5.1's Known Issues concerns us directly**:
> *"Some MoE training configurations that combine TP and EP may run slower in 26.06 after upgrading from NCCL 2.29 to NCCL 2.30"*, with numactl core binding as the official workaround.
> We are **already doing that**: `numactl --cpunodebind=$((LOCAL_RANK/2)) --membind=$((LOCAL_RANK/2))`.
> The binding is verified correct: GB300 has 2 CPU NUMA nodes (node0 = CPU 0-71, node1 = CPU 72-143),
> and sysfs queries (`0008:06:00.0`→0, `0009:06:00.0`→0, `0018:06:00.0`→1, `0019:06:00.0`→1) cross-check with `nvidia-smi topo -m` to confirm the `LOCAL_RANK/2` mapping.

### 14.3 Porting approach: one file, no version change

`hy_v3_bridge.py` is only **286 lines / 14.4 KB**, and every symbol it imports is a stable interface that already exists in r0.5.0. Checked one by one:

| Interface | Present in r0.5.0 |
|---|---|
| `conversion.mapping_registry.MegatronMappingRegistry` | ✅ |
| `conversion.model_bridge.MegatronModelBridge` | ✅ |
| `conversion.param_mapping.AutoMapping` | ✅ |
| `conversion.param_mapping.GatedMLPMapping` | ✅ |
| `conversion.param_mapping.QKVMapping` | ✅ |
| `models.gpt_provider.GPTModelProvider` | ✅ |
| `models.hf_pretrained.causal_lm.PreTrainedCausalLM` | ✅ |
| `megatron.core.models.gpt.gpt_layer_specs.get_gpt_decoder_block_spec` | ✅ |

The decorator signature checks out too — `register_bridge(*, source, target, provider=None, model_type=None)` **already** supports `model_type` in r0.5.0
(`deepseek_v3_bridge.py` in the same directory is written that way), and the base class's `megatron_to_hf_config` is there as well (`model_bridge.py:670`). **Usable with zero modifications.**

### 14.4 Installation steps (reproducible)

```bash
# 1) fetch the 2 files from main
mkdir -p /tmp/hy3bridge && cd /tmp/hy3bridge
for f in hy_v3_bridge.py __init__.py; do
  gh api "repos/NVIDIA-NeMo/Megatron-Bridge/contents/src/megatron/bridge/models/hy_v3/$f?ref=main" \
     --jq '.content' | base64 -d > "$f"
done

# 2) place them in the container (base64 through kubectl exec, avoiding an scp dependency)
D=/opt/Megatron-Bridge/src/megatron/bridge/models
mkdir -p $D/hy_v3 && cp hy_v3_bridge.py __init__.py $D/hy_v3/

# 3) register in models/__init__.py (two places: import + __all__)
#    from megatron.bridge.models.hy_v3 import HYV3Bridge  # noqa: F401
#    __all__ = ["HYV3Bridge", ...]
```

> **Always `md5sum` in both directions** — on the `kubectl exec` + base64 transport we have previously hit
> "the file looks written but was not actually written" (§8). This time the md5 inside and outside the container was `feadf836…` on both sides.

### 14.5 Four-step verification (all passed)

**① import**
```
from megatron.bridge.models import HYV3Bridge   →  OK
```

**② HF config recognition**

| Field | Hy3 / Hy3-Base |
|---|---|
| `architectures` | `['HYV3ForCausalLM']` ← exactly matches the bridge's registered `source` |
| `model_type` | `hy_v3` |
| layers / hidden | 80 / 4096 |
| experts / moe_int / shared | 192 / 1536 / 1 |
| `first_k_dense_replace` | 1 |
| `router_scaling_factor` | 2.826 |
| `num_nextn_predict_layers` | **1** |
| vocab | 120832 |

**③ `AutoBridge.from_hf_pretrained` → `to_megatron_provider(load_weights=False)`**

The auto-derived provider matches our **hand-written `HY3` dict field for field** — which in turn validates that §1's mapping table is correct:

| provider field | Auto-derived value | vs the hand-written `HY3` |
|---|---|---|
| `num_layers` | 80 | ✅ |
| `hidden_size` / `ffn_hidden_size` | 4096 / 13312 | ✅ |
| `num_attention_heads` / `num_query_groups` / `kv_channels` | 64 / 8 / 128 | ✅ |
| `vocab_size` / `rotary_base` | 120832 / 11158840.0 | ✅ |
| `num_moe_experts` / `moe_router_topk` / `moe_ffn_hidden_size` | 192 / 8 / 1536 | ✅ |
| `moe_shared_expert_intermediate_size` | 1536 | ✅ (= 1536 × 1 shared) |
| `moe_router_score_function` / `enable_expert_bias` | sigmoid / True | ✅ |
| `moe_router_topk_scaling_factor` | 2.826 | ✅ |
| `qk_layernorm` / `normalization` | True / RMSNorm | ✅ |
| `moe_layer_freq` | `len=80, [0,1,1,1,1…]` | ✅ |
| **`mtp_num_layers`** | **1** | ⚠️ **we set 0 in the performance experiments** |

> **The only difference is MTP.** The performance sweep disabled MTP with `--mtp-layers 0` (to isolate variables);
> the official weights carry 1 MTP layer. For SFT you **must turn `mtp_num_layers=1` back on**, or the layer-80 weights in the checkpoint have nowhere to go.

**Key point: `from_hf_pretrained` is lazy** — without `load_weights` it does not pull the 597.6 GB of weights, only the config + index.

**④ Weight-mapping coverage audit** (the hardest criterion)

Pull `model.safetensors.index.json` (a few MB) and test **every real tensor name** in the checkpoint against the bridge's 42 mappings:

```
HF checkpoint: 47,138 tensors / 597.6 GB / 99 shards
bridge mappings: 42 rules → 54 hf_param patterns

uncovered: 0  ✅ everything has a home
orphans (mapping exists, checkpoint does not): 3
  · model.layers.80.mlp.{gate,up,down}_proj.weight
```

The 3 orphans are the **dense-MLP fallback branch of the MTP layer** — Hy3's MTP layer is actually a MoE layer, so the dense path is never taken. A normal fallback, not something missing.

**Cross-check on parameter count**: 597.6 GB ÷ 2 bytes (bf16) = **298.8 B**
= **295 B** backbone + **3.8 B** MTP, exactly matching the official figure.

### 14.6 Two things settled along the way

**① `tencent/Hy3` is already post-trained; `Hy3-Base` is the base model**

| | `tencent/Hy3` | `tencent/Hy3-Base` |
|---|---|---|
| downloads / likes | 18,600 / 874 | 500 / 24 |
| `chat_template.jinja` | ✅ 10,223 bytes | ❌ absent |
| template embedded in `tokenizer_config` | ❌ | 596 bytes (minimal) |
| directory | `finetune/` | `train/` |
| assets | `rl-training.png`, `benchmark.png` | `bench_*.jpg` |

`rl-training.png` plus a 10 K-character chat template ⇒ **Hy3 is a reasoning model after SFT + RL**.
So the answer to "has Hy3 already been SFT'd" is: **yes**, and the official team separately released the un-post-trained Base version.

**② What the official SFT recipe uses**

Both repositories ship a complete fine-tuning scaffold, uniformly built on **DeepSpeed ZeRO-2/ZeRO-3** underneath:

```
train/ or finetune/
├── ds_zero2_no_offload.json / ds_zero3_offload.json …   # DeepSpeed configs
├── llama_factory_support/  hy_v3_{full,lora}_sft.yaml + hy_v3_patches.py + hy_v3_template.py
├── ms_swift_support/       hy_v3_{full,lora}_sft.yaml + hy_v3_swift_patches.py
├── train.py / train.sh / train_lora.sh / merge_lora_weight.py
└── tools/convert_ckpt_to_outer.py + check_converted.py   # convert back to HF format after training
```

**The data format is a standard ChatML `messages` array**, with no proprietary fields:

```json
{"messages": [
  {"role": "system",    "content": "You are a helpful assistant."},
  {"role": "user",      "content": "Could you help me write the opening of an argumentative essay on environmental protection? …"},
  {"role": "assistant", "content": "In our daily lives, environmental protection has become an issue that cannot be ignored. …"}
]}
```

> This matters: **the official data format needs no adaptation.** We only need to swap the DeepSpeed layer for Megatron-Bridge
> and keep the official schema on the data side. The reasons for going with Megatron are in §14.7.

### 14.7 Conclusion

| Item | Status |
|---|---|
| HYV3Bridge ported into the r0.5.0 container | ✅ single file, zero modifications |
| Architecture recognition (`HYV3ForCausalLM` → GPTModel) | ✅ |
| Auto-derived hyperparameters vs the hand-written mapping table | ✅ field-for-field identical |
| Mapping coverage of all 47,138 weights | ✅ 100% |
| Parameter-count check | ✅ 298.8 B = 295 B + 3.8 B MTP |
| Does verification require downloading 597 GB | ❌ lazy loading; config + index suffice |

**Next**: SFT needs `mtp_num_layers=1`, a real weight-loading path, and a scarce-knowledge dataset whose effect is actually visible.
