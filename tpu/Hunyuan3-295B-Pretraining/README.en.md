> 🌐 [中文](README.md) | **English**
>
> 🚧 **Translation in progress.** Links marked 🇨🇳 still point to the Chinese version; their English counterparts are being added document by document.

# Pre-training Tencent Hunyuan 3 (295B-A21B) on TPU

Porting Tencent Hunyuan 3 into MaxText and running the full 80-layer 295B-A21B
pre-training on both **TPU v5p** and **TPU v7 (Ironwood)**. This directory collects the
approach, reproducible procedures, and a cross-platform performance comparison.

**MaxText does not support Hunyuan 3 out of the box.** Every component it needs already
exists — they are just scattered across two different decoder blocks. This project wires
them into a new `decoder_block: "hunyuan3"`; the only new code is assembly logic,
**zero new math**.

---

## Results

| | **v5p**<br>**256 chips** | **v7 Ironwood**<br>**64 chips** | **v7 Ironwood**<br>**256 chips** | **GB300**<br>**64 GPU** (reference) |
|---|---|---|---|---|
| Compute units | 256 chips | **64 chips** | **256 chips** | 64 GPU |
| **Sequence length** | **8192** ⚠️ | 4096 | 4096 | 4096 |
| Parameters (as reported) | 298.786 B | 298.786 B | 298.786 B | — |
| Steady-state step | 63.2 s | 23.5 s | 30.4 s | — |
| **TFLOP/s per unit** | 161.0 | **580.0** | **598.8** | 854.0 |
| **MFU** | **35.07%** | 25.14% | **25.96%** | 31.60% |
| **Cluster token throughput** | 265,588 tok/s | 267,284 tok/s | **1,103,757 tok/s** | 399,488 tok/s |
| **Per-unit token throughput** | 1,037 | **4,176** | **4,312** | 6,242 |
| Best recipe | Official DSV3 v5p recipe | `DP1×FSDP128`<br>tile + pdbs 12 | `DP2×FSDP256`<br>tile + pdbs 16 | — |
| Status | ✅ Converged, reproduced off-site | ✅ Small-scale ceiling | ✅ **Target met**, 600–630 | Tuned |
| FP8 (same hardware, **denominator 4614**) | — | — | **625 / MFU 13.6%** (64c, QAG on); 618 at 256c without QAG (DSV3 743.5 / 16.1%) | — |

All four run the same 295B-A21B, the same BF16, the same synthetic data, with checkpointing off.
**For cross-platform comparison look only at per-unit token throughput and MFU** —
cluster throughput scales with the number of units and is not comparable.

> ⚠️ **Sequence lengths do not match; discount the v5p column accordingly.**
> v5p used **8192** (`max_target_length=8192`, tokens per step = 256 × 8 × 8192),
> while v7 and GB300 both used **4096**.
> Attention FLOPs grow quadratically with sequence length, so **longer sequences
> inherently yield lower tok/s** — meaning v5p's 1,037 is a number suppressed by the 8K
> setting, and **any multiple computed against it is an overestimate**.
> **The v7 ↔ GB300 pair is strictly aligned (both 4096) and can be compared directly.**

Three ways to read this table:

1. **v7 delivers roughly 4.16× the per-chip throughput of v5p** (4,312 vs 1,037,
   ⚠️ **v5p's figure was measured at 8K sequence length, so this multiple is an overestimate**),
   yet its MFU is *lower* (25.96% vs 35.07%) — because v7's BF16 peak is 5.03× that of v5p,
   while HBM bandwidth only grew 2.64× and ICI only 2×.
   **Falling MFU alongside a large absolute throughput gain is the expected consequence of
   that hardware imbalance, not a sign of poor tuning.**
2. **v7 has closed to 69.1% of GB300's per-GPU throughput** (4,312 vs 6,242), up from 51.4%
   before tuning.
3. **v5p's 35.07% still beats GB300's 31.6%** — a 256-chip 3D torus plus SparseCore
   collective offloading hides MoE's fine-grained communication even more cleanly than an
   NVLink domain does.

> **v7 lands at the same water line on 64 chips and on 256 chips** (580 vs 580, with peak
> HBM identical to the byte). The extra 3.3% at 256 chips comes from *having a wider FSDP
> option available, which frees memory for a larger batch* — **not from scale itself**.
> See [QUICKSTART-v7 §4.2.1](QUICKSTART-v7.en.md#421-scaling-weak-scaling-is-100-strong-scaling-has-a-price).

---

## Where to start

| Your situation | Read this |
|---|---|
| **Want to run it on v5p** | **[QUICKSTART-v5p.md](QUICKSTART-v5p.md) 🇨🇳** — from creating the node pool to first numbers, two commands, with full parameters and baselines. **Verified from scratch.** |
| **Want to run it on v7 (Ironwood)** | **[QUICKSTART-v7.en.md](QUICKSTART-v7.en.md)** — **gives you the best recipe directly**; copy it and you reach BF16 599 / FP8+QAG 625. Covers both the 64- and 256-chip scales, end-to-end reproduction, and unit conversions |
| **Want to know how that 599 was reached** | **[TUNING-v7.en.md](TUNING-v7.en.md)** — the full 445 → 599 story line, with the reason / mechanism / gain behind each step; bottleneck diagnosis, scaling behavior, HBM model; every negative result folded into the appendix |
| Want to port a **different** model into MaxText | [MAXTEXT-PORTING-GUIDE.md](MAXTEXT-PORTING-GUIDE.md) 🇨🇳 — the general pattern distilled from this project, independent of Hy3 |
| Want to know where a specific number or claim came from | [EXPERIMENT-LOG.md](EXPERIMENT-LOG.md) 🇨🇳 — the complete experiment archive, 2,600 lines |
| Just want to run the scripts | [maxtext-hunyuan3/](maxtext-hunyuan3/) — `prep.sh` + `run.sh` |

---

## Code

**The fork branch is the single source of truth**; no copy of the code is kept in this repo:

```
https://github.com/yangwhale/maxtext   branch hunyuan3
```

Based on upstream main, three commits, already split along the boundaries of two upstream PRs:

| Commit | Belongs to |
|---|---|
| `Resolve the loss-free-balancing bias path per decoder block` | PR ① (a pure upstream bug fix, unrelated to Hy3) |
| `Add Tencent Hunyuan 3 (295B-A21B)` | PR ② |
| `Let Hunyuan3 use the SwiGLU activation bound too` | PR ② |

Scope of change: **3 new files** (161 lines of model code + 2 yml files) and
**12 modified upstream files**.

---

## Model at a glance

| | |
|---|---|
| Structure | 80 layers; layer 0 dense, layers 1–79 MoE |
| Attention | GQA 64q / 8kv, head_dim 128, QK-LayerNorm, no bias — **lineage is Qwen3** |
| MoE | 192 routed experts top-8 plus 1 shared, sigmoid routing with expert bias — **lineage is DeepSeek V3** |
| Other | 1 MTP layer, vocab 120832, routed scaling 2.826 |
| Parameter distribution | **97% sits in the routed experts**; attention accounts for only 2% |

That parameter distribution dictates the parallelism strategy directly: **TP is useless**
(sharding attention is pure communication loss), and **do not use EP on TPU** — ICI is a
3D torus, so AllToAll requires multi-hop forwarding, unlike the full mesh of GPU NVLink;
measured on 16 chips, EP=4 costs **−71%**.
**Keep FSDP width fixed at 128 and give every additional device to DP.**

The key difference from DeepSeek V3: **Hy3 has no device-limited routing** — it does a
global top-8 across all 192 experts. Copying the DSV3 recipe and bringing
`n_routing_groups` / `topk_routing_group` along changes routing behavior, and it does so
without raising an error.

---

## The single biggest lesson from this project

> **Almost every "which path should this model take" decision inside MaxText is a table
> keyed on the model family name.** Adding a new model is never a one-place edit; it means
> finding every one of those tables and asking, one by one, "should I be in here?"

The same pattern showed up **10 times** in this project. Nine of them either blew up only at
runtime, or **did not blow up at all and quietly ran different semantics** — the routing
branch and the FLOP formula both fall into the latter category.
Full ledger in [EXPERIMENT-LOG §8](EXPERIMENT-LOG.md) 🇨🇳.

Another one: **starting from a hand-rolled config gave an MFU of just 2.45%; copying the
official DeepSeek3 v5p recipe and changing nothing but the model name jumped straight to
31.56%.** When porting a new model, find the official recipe for a comparable model first,
and only then start tuning.

---

## Not done yet

| Item | Notes |
|---|---|
| Convergence validation on real data | Everything so far is synthetic; it only proves "it computes and does not diverge" |
| HF weights → MaxText Orbax conversion | Not needed for throughput baselines; mandatory for SFT |
| Push v7 BF16 to 630 | Currently 599 (25.96%), already at the lower edge of the target band |
| **v7 FP8 + QAG** | ✅ **Converged.** With QAG on at 64 chips: **625** (vs 594 without QAG, +5.3%); the 256-expert exploration reached **645**.<br>⚠️ The earlier note that "the tile on that FP8 kernel path has never been swept, potential ~726" **has been disproven by measurement** — FP8 still routes back into tokamax internally, and the tile monkeypatch has been in effect all along.<br>On 2026-08-05 we additionally swept tile / XLA flags / SparseCore offloading / larger batch, 8 cells in total, with **not a single positive result**: the tuning space is exhausted. Going higher requires more chips, a different model shape, or writing code. See [TUNING-v7 §4.6](TUNING-v7.en.md#46-what-can-be-tuned-and-what-cannot--one-master-table) |
| Upstream PRs | Two of them, boundaries already split, contribution process to be confirmed |

---

## References

| Source | Notes |
|---|---|
| [GB300 Hunyuan 3 training doc](../../gpu/a4x-max/07-megatron-training/07f-gb300-hy3-295b-gke/README.en.md) | **Architecture SSOT** plus the GB300 baseline |
| [GB300 Hunyuan 3 SFT doc](../../gpu/a4x-max/07-megatron-training/07f-gb300-hy3-295b-gke/SFT.md) (zh) | Bridge port, weight conversion, evaluation loop |
| [DeepSeek V3.2 TPU training](../DeepSeek-V3.2-Training/README.md) (zh) | MaxText operating patterns plus v7 MoE pitfalls |
| [tencent/Hy3](https://huggingface.co/tencent/Hy3) | Official weights and config |
