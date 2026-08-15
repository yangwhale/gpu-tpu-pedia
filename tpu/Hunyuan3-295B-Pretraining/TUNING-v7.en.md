> 🌐 [中文](TUNING-v7.md) | **English**
>
> 🚧 **Translation in progress.** Links marked 🇨🇳 still point to the Chinese version; their English counterparts are being added document by document.

# Performance Tuning of Hunyuan 3 (295B-A21B) on TPU v7

> **BF16 went from 445 to 599 TFLOP/s/chip (MFU 19.3% → 26.0%); FP8 with QAG on reached 645.**
> This document covers what was changed at each step, why it worked, and what it was worth — **plus a large number of paths that were tried and led nowhere; that part matters just as much.**
>
> Just want the recipe and want to run → **[QUICKSTART-v7.md](QUICKSTART-v7.en.md)**, which has complete commands you can copy verbatim.
> Just want to know "what can be tuned and what cannot" → go straight to **[§4.6 master table](#46-what-can-be-tuned-and-what-cannot--one-master-table)**.

---

## 0. Three-minute read

| What you want to know | Where to go |
|---|---|
| **What can be tuned, and what it is worth** | [§4.6 one master table](#46-what-can-be-tuned-and-what-cannot--one-master-table) — read this if you read only one section |
| **How to tell a real gain from a fake one** | [§4.7 four rules](#47-four-rules-for-telling-whether-a-gain-is-real-or-fake) |
| Every step of BF16 445 → 599 | [§3 the tuning story line](#3-the-tuning-story-line-from-445-to-599) |
| The full mechanism and recipe for FP8 / QAG | [§5.4.2](#542-qag-quantize-first-then-communicate-a-path-blocked-by-the-expert-count) |
| Everything that was tried and did not work | [Appendix B](#appendix-b-the-complete-collection-of-negative-results) (collapsed by default) |

**The three conclusions most worth taking away:**

1. **`tokamax tile` is worth +17.4%, the largest single item in the whole exercise** — but what it really does is patch a broken default
   (the kernel lookup table has no row for 192), **it is not routine tuning, and the magnitude cannot be extrapolated elsewhere**.
2. **QAG (all-gather after quantization) nets +15.6% and saves 4.5–11 G of HBM** —
   but it requires `num_experts % FSDP == 0`. **Making the expert count a power of two is a model design constraint,
   and this pitfall is completely invisible at ≤32 chips.**
3. **The tuning space is essentially exhausted**: tile, XLA flags, SparseCore offloading, and pushing batch —
   four directions, 8 cells of experiments, **not a single positive result**. Going higher requires more chips, a different model shape, or writing code.

---

## 1. Water line and target

### 1.1 Where we are now

All measured, all on the **full 80 layers**, seq 4096, synthetic data, steady state taken from steps 4–7.
**MFU denominators: 2307 for BF16, 4614 for FP8** — the two cannot be compared directly.

**A. Hy3 proper (192 experts)**

| Scale | Recipe | step | **TFLOP/s/chip** | **MFU** | **tok/s** | tok/s/chip | Peak HBM |
|---|---|---|---|---|---|---|---|
| 256-chip maximum **BF16** | `DP2×FSDP256` + tile + pdbs **16** | 30.40 s | **599** | **25.96%** | **1,103,757** | **4,312** | 92.33 G |
| 256-chip recommended **BF16** | `DP4×FSDP128` + tile + pdbs 12 | 23.56 s | 580 | 25.12% | 1,068,372 | 4,173 | 91.94 G |
| 64-chip **BF16** | `DP1×FSDP128` + tile + pdbs 12 | 23.54 s | 580 | 25.14% | 267,284 | 4,176 | 91.94 G |
| 256-chip **FP8** (no QAG) | `DP2×FSDP256` + `fp8_full`+qwix | 29.46 s | 618 | 13.39%<sub>FP8</sub> | 1,139,022 | 4,449 | 92.80 G |
| 64-chip **FP8** (no QAG) | `DP1×FSDP128` + pdbs 10 | 19.15 s | 594 | 12.87%<sub>FP8</sub> | 273,987 | 4,281 | 86.20 G |
| **64-chip FP8 + QAG** ⭐ | `DP2×FSDP64` + QAG + pdbs 7 | 12.73 s | **625** | 13.55%<sub>FP8</sub> | 288,222 | **4,503** | 92.42 G |
| Starting point (2026-07-30) | `FSDP128` + megablox + pdbs 8 | 20.43 s | 445 | 19.29% | 205,313 | 3,208 | 74.20 G |

> ⭐ **The current best at 64 chips is FP8+QAG at 625**, 5.3% above the 594 without QAG at the same scale,
> and with a smaller batch (7 vs 10). The cost: FSDP can only be 64 ([§5.4.2](#542-qag-quantize-first-then-communicate-a-path-blocked-by-the-expert-count)).

**B. The 256-expert exploration (the model was changed; for next-generation design reference only)**

| Scale | Recipe | step | **TFLOP/s/chip** | MFU | tok/s | tok/s/chip | Peak HBM |
|---|---|---|---|---|---|---|---|
| 64-chip **FP8 + QAG** | `DP1×FSDP128` + QAG + pdbs **11** | 19.42 s | **645** | 13.98%<sub>FP8</sub> | 296,955 | 4,640 | 91.56 G |

> ⚠️ **645 is not comparable with the table above** — the expert count went from 192 to 256, so both the parameter count and the FLOP convention changed.
> What it answers is "**what would a next-generation model look like on v7 if the expert count were set to a power of two**":
> pick any FSDP you like, batch can go to 11, and there is still HBM to spare. **192 is stuck with a narrow FSDP plus batch 7.**

> **tok/s = device count × pdbs × seq ÷ step**; for cross-comparison look only at tok/s/chip.
> Reference: GB300 = **6,242** tok/s/GPU (**also seq 4096, directly comparable**);
> v5p 256 chips = **1,037** tok/s/chip (⚠️ **v5p used seq 8192, a different convention**).
> **v7 per-chip throughput is 69.1% of a single GB300 GPU** (51.4% before tuning).
> The "4.16×" over v5p is **inflated** by the sequence-length difference and should be treated as an order-of-magnitude reference only.

**BF16 target 600–630 TFLOP/s/chip (26–27% MFU): currently 599, 0.2% short, target met.**
**FP8 currently 625 (192e) / 645 (256e); the tuning space is exhausted, see [§4.6](#46-what-can-be-tuned-and-what-cannot--one-master-table).**

> **Notation used in this document** (the body uses shorthand; the full copy-paste commands are in [QUICKSTART-v7 §0](QUICKSTART-v7.en.md#0-best-recipe-at-a-glance)):
>
> | Shorthand | Actual parameter |
> |---|---|
> | `pdbs` | `per_device_batch_size` |
> | `DP=N` | `ici_data_parallelism=N` (when N=1, writing `ici_fsdp_parallelism=-1` is enough) |
> | `FSDP=M` | `ici_fsdp_parallelism=M` |
> | `tile(a,b,c)` | tokamax `tile_m/tile_k/tile_n`, injected via monkeypatch ([§3.4.3](#343-the-fix-a-6-line-monkeypatch)) |
> | per-chip | **v7 has 2 devices per chip**, `= log TFLOP/s/device × 2`; MFU denominator BF16 **2307** / FP8 **4614** |

### 1.2 Why the target is 600–630, not 900

The official Ironwood measurement table (all bf16, synthetic, per-chip basis):

| Model | Type | chips | Sequence | TFLOP/s/chip | MFU |
|---|---|---|---|---|---|
| llama3.1-405b | **dense** | 256 | 8192 | 1,261.4 | 54.7% |
| llama3.1-70b | **dense** | 64 | 8192 | 1,207.1 | 52.3% |
| gemma4-31b | **dense** | 64 | 8192 | 931.3 | 40.4% |
| **qwen3-235b-a22b** | **sparse MoE** | 256 | 4096 | **629.8** | **27.3%** |
| **deepseek-v3 671B** | **sparse MoE** | 256 | 4096 | **612.7** | **26.6%** |
| gpt-oss-120b | sparse MoE | 256 | 8192 | 329.9 | 14.3% |
| **hunyuan3 295B (this project)** | **sparse MoE** | **256** | **4096** | **599** | **25.96%** |

**Everything above 900 is a dense model.** The real water line for sparse MoE on Ironwood is 600–630,
and the two closest references to Hy3 both sit on that line. The gap is structural:

- A dense model sends every token through the same weights; the GEMMs are large and regular, and the MXU can be kept full
- MoE has to route every layer, regroup and reorder by expert, do a grouped matrix multiply, and then restore. Each sub-block is only
  `tokens_per_expert × emb × moe_mlp` in size, and group sizes float with routing, so **static shapes are not available at compile time**
- On top of that, all-gather / reduce-scatter has to spread out 192 expert weight shards and gather them back

> Hy3's activated parameters (21 B) are fewer than DSV3's (37 B) and the structure is simpler (GQA rather than MLA, 192 experts rather than 256),
> **so there is no reason it cannot reach the same water line** — the gap comes from configuration, not architecture.

**Managing expectations for FP8**: in that same table, turning on FP8 gains DSV3 only **+21.4%** (612.7 → 743.5),
while dense llama3.1-405b gains **+52.8%**. **MoE cannot cash in FP8's doubled peak** —
most of the time goes into routing, reordering, communication, and small GEMMs, none of which touch the MXU peak, and lowering precision does not help there.

> ⚠️ **When you report an FP8 MFU, always state the denominator.** The same 743.5 is 16.1% against the FP8 peak
> and 32.2% against the BF16 peak — a factor of two apart.

---

## 2. Where the bottleneck is: one trace set the direction

**Before sweeping any parameter, spend one round finding out where the time goes.** This step determined the direction of every experiment that followed.

### 2.1 Two hypotheses

The three ratios of v7 relative to v5p:

| | v5p | v7 | v7 / v5p |
|---|---|---|---|
| BF16 peak / chip | 459 TFLOPS | 2,307 TFLOPS | **5.03×** |
| HBM bandwidth | 2.8 TB/s | 7.4 TB/s | 2.64× |
| ICI / chip | 600 Gbps | 1,200 Gbps | 2.0× |

**Compute grew 5×, while the two channels that feed it grew only 2–2.6×.** The roofline knee moves from 164 to **312 FLOP/byte**,
so arithmetic intensity has to grow 1.9× just to stay compute-bound. MoE happens to be the structure with the lowest arithmetic intensity.

| | **H1: HBM bandwidth is the limit** | **H2: the time goes into non-MXU work** |
|---|---|---|
| Claim | MoE has low arithmetic intensity; weight reads saturate 7.4 TB/s | Routing, reordering, communication, and small GEMMs do not use the MXU |
| Trace criterion | HBM bandwidth close to 7.4 TB/s | HBM is not high, but collective ops fill the timeline |
| What to do | Raise arithmetic intensity | Hide communication, fix the kernel |

### 2.2 Conclusion: H2 holds, communication is 57.3%

Self-time breakdown (covering 98.2% of wall clock):

| Category | Self time | Share of wall clock |
|---|---|---|
| **Communication · waiting on `-done`** | 1.299 s | **41.9%** |
| Compute · MoE grouped matrix multiply (`gmm`/`tgmm`) | 0.723 s | 23.3% |
| **Communication · synchronous collectives** | 0.477 s | **15.4%** |
| Compute · `fusion`/`dot` | 0.446 s | 14.4% |
| Compute · attention (`splash`) | 0.046 s | 1.5% |
| Data movement `copy`/`transpose` | 0.025 s | 0.8% |
| **Communication total** | **1.777 s** | **57.3%** |
| **Compute total** | **1.215 s** | **39.2%** |

**This directly explains why MFU is only 19.29%**: actual computation occupies just 39.2% of wall clock,
and `19.29% ÷ 39.2% ≈ 49%` — **when compute really is running, MXU utilization is about half, and the rest is eaten by communication.**
The core spends **41.9% of the wall clock sitting on `-done` doing nothing.**

![XProf: compute and communication waits alternating](images/v7-xprof-comm-wait.png)

*Zoomed to a 30 ms scale: `gmm.18` → `all-gather...call-done` → `gmm.19` → `all-gather...call-done` → `gmm.20`.
**The `call-done` blocks are as wide as the `gmm` compute blocks or wider, and the two alternate** — compute for a while, then stop and wait for a while.*

From this we inferred, and later measurements confirmed:

- **H1 does not hold** — only 39.2% is actual compute, nowhere near enough to saturate HBM
- **FP8 expectations must come down** — it can only speed up the dot portion of that 39.2%, and **cannot touch the 57.3% that is communication**
- **The main line is fixing the MoE kernel and hiding communication**, not raising arithmetic intensity

> The largest gain that followed (tokamax tile, +17.4%) landed exactly on the "MoE kernel" side,
> consistent with this diagnosis.

### 2.3 How to read the trace: self-time breakdown

![First-round v7 XProf trace](images/v7-xprof-trace.png)

**Use the official tool (XProf) first; do not draw your own.** Parsing `trace.json` yourself is only good for bulk statistics.

Capturing a profile:

```bash
PLATFORM=v7 STEPS=25 bash run.sh prof \
  base_output_directory=gs://<bucket>/hy3prof \
  profiler=xplane skip_first_n_steps_for_profiler=8 profiler_steps=5 \
  profile_cleanly=True dump_hlo=True
```

| Parameter | Why |
|---|---|
| `base_output_directory` must be GCS | The default `/tmp` is the pod's local disk and disappears when the pod ends — **the most common reason a profile never shows up** |
| `skip_first_n_steps_for_profiler=8` | Step 0 includes compilation; steps 1–2 are false readings from asynchronous dispatch |
| `profile_cleanly=True` | Aligns per step; the cost is that steps in this run are slower, so **do not treat its MFU as a number** |

**The key to the analysis is "self time"**: a container op (`while`, i.e. the 80 layers rolled up by `scan_layers`)
counts its children's time as its own, so you have to subtract the children's duration from every ancestor and keep only the part that truly occupies the core alone.
Classified this way, the numbers add up and balance.

```bash
gcloud storage cp "gs://<bucket>/<run>/tensorboard/plugins/profile/*/*.trace.json.gz" .
gunzip -c *.trace.json.gz > t.json      # 62 MB → 1.4 GB, leave enough disk
python3 maxtext-hunyuan3/analyze-trace.py t.json
```

<!-- ===== TEMP:XPROF-LINKS  Kept temporarily for the internal discussion period; delete this whole block once tuning wraps up ===== -->
> **🔗 XProf sessions (Google account required, for the internal discussion period only)**
>
> | Profile | Session |
> |---|---|
> | 4 chips `2x2x1` / 80 layers | （内部 XProf trace `chrisya-11640939633798411639`，仅 Google 内网可访问） |
> | 16 chips `2x2x4` / 20 layers (with HLO dump) | （内部 XProf trace `chrisya-18130551067782033931`，仅 Google 内网可访问） |
>
> Sessions expire; once they do, just re-upload the `.xplane.pb` from GCS.
<!-- ===== /TEMP:XPROF-LINKS ===== -->

<details>
<summary><b>⚠️ I overturned this conclusion four times; all four wrong versions are on the record (click to open)</b></summary>

The same trace produced five mutually contradictory conclusions in sequence:

| Version | Conclusion | What went wrong |
|---|---|---|
| ① | Overlap 0.000 s ⇒ completely exposed | Computed time intersections on **a single sequential lane**. Of the 40560 events, all 16550 intersections **are container nesting, with partial crossings of 0** — on this lane, top-level ops are naturally back-to-back, so the intersection is identically zero. **That is a tautology, not a measurement** |
| ② | 80% is synchronous blocking | Split op names on `.`, which chopped off the `-start`/`-done` suffix. The real name is `all-gather.382.cloned.1.call-done`, **the suffix is in the last segment** |
| ③ | 100% asynchronous ⇒ 1.766 s is all residual | Right direction, but treated "it is asynchronous" as "it was not hidden", without measuring actual occupancy |
| ④ | 83.4% is already hidden | Summing the compute inside every `start→done` window gave 11.607 s, **but the timeline is only 3.100 s** — on average 4.5 transfers are in flight at once, so the same stretch of compute was counted four or five times |
| ⑤ **adopted** | Communication 57.3% / compute 39.2% | Self-time breakdown, 98.2% coverage, balances |

**What caught ④ was a dimensional contradiction**: if communication really were 80% hidden, MFU could not be only 19%.
**When two independent conclusions do not line up, suspect the method first; do not rush to explain the phenomenon.**

Four general lessons:

1. **Computing concurrency on a single sequential lane is a tautology** — first confirm whether that lane can overlap at all
2. **The suffix of an op name is in the last segment**, do not use `split('.')[0]`
3. **"It is asynchronous" ≠ "it was hidden"**, and ≠ "it was not hidden" either; you have to measure actual occupancy
4. **Windows can overlap each other** — any result where "the sum of the parts > the total" is a signal of double counting

**Another self-check**: for any analysis that sums over ops, first compute "sum ÷ timeline span".
My first analysis produced 156.6%, which was the signal of parent-child double counting.

</details>

---

## 3. The tuning story line: from 445 to 599

**Six steps. Each one covers only three things: what was changed, why it worked, and what it was worth.**

### 3.1 Step one, +12.8%: switching the batch / sequence convention

**Change**: `seq 8192 / pdbs 4` → `seq 4096 / pdbs 8`. Total token count unchanged.

**Why it works**: the token count is the same, but attention compute grows with the **square** of seq,
while the MoE part grows only linearly with token count. Shortening the sequence moves time out of attention and back into the MoE main path.

**Gain**: throughput +12.8% (TFLOP/s only +0.9% — meaning what was saved is time, not compute).

> **Two rounds of verification turned the conclusion from "a tie" into "short sequences are clearly better":**
>
> | Period | Configuration pair | Result |
> |---|---|---|
> | megablox era (256 chips) | seq 8192/pdbs 4 = 451 vs seq 4096/pdbs 8 = 453 | tie |
> | **after tile (64 chips)** | **seq 8192/pdbs 6 = 561 vs seq 4096/pdbs 12 = 580** | **short sequence +3.3%** |
>
> **The tile optimization amplified the advantage of the short sequence.** Presumed reason: the tile `(512, 2048, 1536)` was swept
> on the shapes produced by seq 4096; switching to seq 8192 changes the `m` structure of MoE, and the same tile no longer matches.
> ⇒ **This dimension is not only exhausted, it is now clearly one not to touch.**

### 3.2 Step two, +6.6%: the scheduler flag group

**Change**: add 4 XLA flags.

```
--xla_tpu_enable_latency_hiding_layer_scheduler=true
--xla_tpu_scheduler_percent_shared_memory_limit=150
--xla_tpu_enable_layer_scheduler_for_dependent_collectives=true
--xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=false
```

**Why it works**: what they change is **how communication and compute overlap**. §2 already measured communication at 57.3% with
41.9% of the wall clock spent idle waiting, so letting the scheduler stuff collectives into the gaps in compute hits the bottleneck directly.

**Gain +6.6%. This is the only group of XLA flags worth anything.**

### 3.3 An important negative result: the SparseCore offloading group is ±0

**Nine SparseCore collective-offloading flags, worth 4.07 pp (13%) on v5p, gain nothing on v7.**

This negative result **is worth more than many positive ones, because it pins down the nature of the bottleneck**:

> SparseCore offloading changes **where communication executes**; the scheduler changes **how communication and compute overlap**.
> The former having no effect says **communication is not "too slow"**; the latter working says **communication "was not hidden"**.

Two derived lessons:

- **The same switch can flip sign across two platforms.** `sa_use_fused_bwd_kernel` needs `False` on v5p and
  `True` on v7; the SparseCore offloading group is +4.07 pp on v5p and ±0 on v7. **Do not carry a conclusion from one platform straight to another.**
- **Do not extrapolate "this group is useless" to "the next group of the same kind is useless too."** That is exactly what I did,
  and I nearly skipped the scheduler group — which turned out to be the +6.6% one. **The discipline of ablation is that every group must actually be run.**

**Eight of those nine really can be deleted**, but the ninth (`--xla_tpu_enable_sparse_core_collective_aggregator`)
is a **hard dependency** of the layer scheduler, and removing it gives an immediate
`INVALID_ARGUMENT: Latency hiding layer scheduler requires sparse core collective aggregator`.
**Trim flags as a group, not one at a time.**

### 3.4 Step three, +17.4%: tokamax tile ← the largest single item

This is the biggest gain of the whole tuning round, and the only place where **"change one number, get 17%"** applies.

#### 3.4.1 The symptom: not setting tile is 12.4× slower

| Configuration | step (single node, 4 chips / 6 layers) | TFLOP/s/device |
|---|---|---|
| megablox (default path) | 1.321 s | 182.0 |
| tokamax **default tile** (falls back to `128³`) | **17.955 s** | 13.4 |
| tokamax `tile(512, 2048, 1536)` | **1.220 s** | **197.2** |

**The cost of missing the tuned table = 12.4×.** Early on we recorded this symptom as a "`use_tokamax_gmm` deadlock",
because it was slow enough to trip the watchdog, reporting `stalled chips [7]` before step 0 even finished.

#### 3.4.2 Root cause: the kernel library lookup table has no row for 192

tokamax's TPU `ragged_dot` looks up three hard-coded tile tables keyed on `(m, k, n, expert count, quantized or not)`,
and falls back to the `Config()` default when there is no match:

```
GMM_TILING_TUNED_LUT: 28 entries, expert count values = [16, 128, 256]
  (524288, 4096, 1536, g=128) -> tile (256, 4096, 1536)
default Config = tile_m=128, tile_k=128, tile_n=128
```

**Hy3 has 192 experts. The matrix dimensions are identical to that entry, only the group count differs, so everything misses.**

| | tile | grid blocks |
|---|---|---|
| Tuned in the table (g=128) | (256, 4096, 1536) | 2048 × 1 × 1 = **2,048** |
| The default actually fallen back to | (128, 128, 128) | 4096 × 32 × 12 = **1,572,864** |

**768× the number of blocks**, each with its own DMA. Three orders of magnitude slower → the watchdog calls it a stall. That is what the "deadlock" really was.

> **Note that the three GMM paths are three independent implementations**, which is the easiest thing to confuse:
>
> | Configuration | Actual kernel | Consumes `w{i,o}_tile_*` | Consults the LUT |
> |---|---|---|---|
> | default | megablox v1 (native JAX Pallas) | ✅ | no |
> | `use_tokamax_gmm` | tokamax v1 `ragged_dot` | ❌ **does not** | ✅ |
> | `use_gmm_v2` | tokamax v2 (forked into MaxText) | ✅ | no |
>
> **`use_tokamax_gmm` does not consume MaxText's tile parameters** — which is exactly why the monkeypatch is needed.

#### 3.4.3 The fix: a 6-line monkeypatch

MaxText does not expose tokamax's tile, so change its heuristics config directly:

```python
# tkcfg.py —— exec this before importing train
import os, dataclasses
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu as P
_TM, _TK, _TN = (int(os.environ[k]) for k in ("TK_TM", "TK_TK", "TK_TN"))
_orig = P.PallasMosaicTpuRaggedDot._get_heuristics_config
def _patched(self, ba):
    c = _orig(self, ba)
    k, n = ba.arguments["rhs"].shape[-2], ba.arguments["rhs"].shape[-1]
    return dataclasses.replace(c, tile_m=_TM, tile_k=min(_TK, k), tile_n=min(_TN, n))
P.PallasMosaicTpuRaggedDot._get_heuristics_config = _patched
```

```bash
TK_TM=512 TK_TK=2048 TK_TN=1536 python3 -c "
exec(open('tkcfg.py').read())
import runpy; runpy.run_module('src.maxtext.trainers.pre_train.train', run_name='__main__')
" ... megablox=True use_tokamax_gmm=True
```

To verify it took effect: the log contains `[tkcfg] patched`, and the 10 s+ steps caused by `Autotuning cache miss` **no longer appear**.

> The proper long-term fix is to run the official autotune and generate cache entries; injection is a verification technique, but it is enough to capture the entire gain.

#### 3.4.4 How to pick the tile values: three rules

Measured on 256 chips (base `DP4×FSDP128`, pdbs 8):

| tile (m, k, n) | chip | vs megablox |
|---|---|---|
| **(512, 2048, 1536)** | **532** | **+17.4%** 🏆 |
| (1024, 2048, 1536) | 512 | +13.0% |
| (512, 1024, 1536) | 499 | +10.2% |
| megablox baseline | 453 | — |

1. **`tile_n` must be `= base_moe_mlp_dim` (1536).** 1024 does not divide it and immediately gives
   `AssertionError: v=1536 bv=1024 s=1536`; 512 divides it but cuts three ways and ends up slower than the bf16 baseline.
2. **`tile_k = 2048` is the sweet spot.** Not the 1024 copied from the table, and not "bigger is better" either — 4096 OOMs outright.
3. **`tile_m` does not follow `m` — this has been verified three independent times.** The rule inside the table is that `tile_m` grows linearly with `m`
   (`m=131072→512`, `524288→1024`), but **512 is optimal at every `m` tested**:

   | `m` | Source | tile_m=512 | tile_m=1024 |
   |---|---|---|---|
   | 262144 | pdbs 8 | **532** | 512 (−3.8%) |
   | 393216 | pdbs 12 | **580** | 569 (−1.9%) |
   | 393216 | pdbs 12 (256-chip batch) | **580** | 567 (−2.2%) |

   **Copying the table is a good starting point, not an end point.**

   > 🔬 **A dedicated falsification round was run on 2026-08-05.** My hypothesis was "`m` goes from 262144 to 393216,
   > so by the rule inside the table `tile_m` should rise to 1024". The result was that **1024 is worse across the board, and once
   > `tile_k`/`tile_n` deviate from (2048, 1536) it drops 7–9%**:
   >
   > | tile | chip | Δ |
   > |---|---|---|
   > | **(512, 2048, 1536)** | **580** | baseline |
   > | (1024, 2048, 1536) | 569 | −1.9% |
   > | (1024, 1024, 1536) | 537 | −7.4% |
   > | (1024, 1536, 1024) | 529 | −8.8% |
   > | (1024, 4096, 1536) | **crash** | Mosaic kernel rejects the combination |
   > | (2048, 2048, 1536) | **crash** | same as above |
   >
   > ⇒ **`(512, 2048, 1536)` is an optimum that is stable across `m` and does not need to be re-tuned per batch.**
   > That matters in practice: when you change `pdbs` you do not have to re-sweep tile.

**It barely costs any HBM** (75.33 vs 74.20 G, +1.1 G) — the best value-for-money item of the round.

<details>
<summary>Full tile sweep (15 combinations on a single node, click to open)</summary>

Single node, 4 chips / 6 layers / pdbs 4, TFLOP/s/device:

**Fixing `tile_n=1536`, sweeping `tile_m × tile_k`**

| tile_m \ tile_k | 512 | 1024 | **2048** | 4096 |
|---|---|---|---|---|
| **256** | 176.5 | 185.0 | **197.2** | 188.9 |
| **512** | 184.6 | 189.4 | **197.2** 🏆 | OOM |
| **1024** | 181.8 | 180.6 | 186.8 | OOM |

**Fixing `tile_m=512, tile_k=2048`, sweeping `tile_n`** (only divisors of 1536)

| tile_n | 256 | 512 | 768 | **1536** |
|---|---|---|---|---|
| TFLOP/s | 167.7 | 186.9 | 191.2 | **197.2** 🏆 |

**Lookup script** (substitute your own k / n):

```python
from tokamax._src.ops.ragged_dot import pallas_mosaic_tpu as P
for name in ("GMM_TILING_TUNED_LUT", "TGMM_TILING_TUNED_LUT"):
    for k, v in sorted(getattr(P, name).items()):
        if k[1] == 4096 and k[2] == 1536:
            print(name, k, "->", v)
```

</details>

### 3.5 Step four, +9.0%: push batch to 12

**Change**: `per_device_batch_size` 8 → 12.

**Why it works**: a larger batch amortizes the fixed communication cost of each step — the volume of the weight all-gather does not change with batch,
but it can be spread over more tokens. This is the same mechanism as the step in §3.1 (+12.8%).

**Why it was not possible before**: pdbs=12 OOMed under the old configuration. **It was FSDP thinning the shards that freed the space** —
see [the HBM model in §4.2](#42-a-two-parameter-hbm-model-computing-the-batch-ceiling-without-hitting-oom).

**Gain** (256 chips, `DP4×FSDP128` + tile):

| pdbs | step | chip | MFU | Peak HBM | Headroom |
|---|---|---|---|---|---|
| 8 | 17.12 s | 532 | 23.04% | 75.33 G | 19.4 G |
| 10 | 20.17 s | 564 | 24.45% | 84.06 G | 10.7 G |
| **12** | **23.56 s** | **580** | **25.12%** | **91.94 G** | **2.8 G** |
| 14 | — | OOM | | predicted 100.8 G | — |

**pdbs 8 → 12 is worth +9.0%**, and 12 is right up against the ceiling.

### 3.6 Step five, +3.3%: widen FSDP to buy batch (only at ≥ 256 chips)

**Change**: `DP4×FSDP128` → `DP2×FSDP256`, while pushing pdbs from 12 all the way to 16.

**Why it works**: doubling the FSDP width **halves** the static shards per device (weights + optimizer + gradients),
freeing 12.84 G → enough for four more pdbs. **Trade a little communication efficiency for HBM, then trade HBM for batch.**

**Gain**: 599 vs 580, **+3.3%**.

| Recipe | FSDP | pdbs | chip | HBM |
|---|---|---|---|---|
| `DP4×FSDP128` | 128 | 12 | 580 | 91.94 G |
| `DP2×FSDP256` | 256 | 12 | 569 | 78.27 G |
| `DP2×FSDP256` | 256 | 14 | 585 | 89.56 G |
| **`DP2×FSDP256`** | 256 | **16** | **599** | **92.33 G** |

> Note the pdbs 12 row: **at the same pdbs, FSDP=256 is actually 1.9% slower than FSDP=128** (569 vs 580).
> Widening FSDP is a **loss** in itself; its entire value lies in **the HBM it frees up being convertible into a larger batch**.

> **64 chips cannot get this step** — it only has 128 devices, so `FSDP=256` is not an option.
> **That makes 580 the physical ceiling for 64 chips.**

### 3.7 How to split parallelism: the usable range is only FSDP ∈ [128, 256]

Five ways to split 512 devices (pdbs fixed at 8, megablox):

| Split | chip | Peak HBM | Verdict |
|---|---|---|---|
| `DP1 × FSDP512` | 404 | — | ❌ loses 11% |
| `DP2 × FSDP256` | 450 | 61.36 G | ⭕ |
| **`DP4 × FSDP128`** | **453** | 74.20 G | ✅ |
| `DP8 × FSDP64` | OOM | — | ❌ |
| `DP16 × FSDP32` | OOM | — | ❌ |

**There is a wall on both sides:**

- **Going wider loses 11%** — the thinner the spread, the more fragmented each collective's shard, and a single payload is not large enough to amortize the fixed cost
- **Going narrower OOMs outright** — halving FSDP doubles the static shards per device. The static part at FSDP=64 is about 51 G,
  which together with activations crosses 94.74 G

**Default rule: fix the FSDP width at 128 and give every extra device to DP.**
64 chips is exactly 128 devices, hence `DP=1`; 256 chips is 512 devices, hence `DP=4`.

**Do not use EP (expert parallelism).** TPU's ICI is a 3D torus, so AllToAll requires multi-hop forwarding,
unlike the full mesh of GPU NVLink. Measured on 16 chips, EP=4 is **−71.36%**,
and there is no physical basis for that flipping positive at larger scale on a torus.

### 3.8 Summary

| # | Change | Mechanism | chip | Cumulative |
|---|---|---|---|---|
| 0 | Starting point (2 XLA flags, seq 8192 / pdbs 4) | — | 405 | — |
| 1 | seq 4096 / pdbs 8 | attention grows with the square of seq; shortening it moves time back into MoE | 445 | +9.9% |
| 2 | scheduler flag group (4 flags) | lets collectives fit into the gaps in compute | 453* | +11.9% |
| — | SparseCore offloading group (9 flags) | changes "where communication executes", but the bottleneck is "it was not hidden" | ±0 | — |
| 3 | **tokamax `tile(512,2048,1536)`** | **sidesteps the LUT miss; grid blocks drop from 1.57 M to 2 K** | **532** | **+31.4%** |
| 4 | pdbs 8 → 12 | a large batch amortizes fixed communication cost | 580 | +43.2% |
| 5 | `DP2×FSDP256` + pdbs 16 (≥256 chips) | widening FSDP saves 13 G of HBM, bought back as four pdbs | **599** | **+47.9%** |

\* Ran on a different batch of machines from the preceding rows, so the absolute value is not directly comparable; for within-batch controls see [Appendix A](#appendix-a-all-ablation-data).

---

## 4. Reusable methodology conclusions

### 4.1 Scaling: weak scaling 100%, strong scaling loses 11%

**Same 512 devices, same code, two ways to split, 11% apart — the only difference is whether batch was added along with the chips.**

| Scaling mode | Split | Work per device | global batch | per-chip | Relative to 64 chips |
|---|---|---|---|---|---|
| **Weak scaling** | `DP=4 × FSDP=128` | unchanged (pdbs 12) | **4×** | **580** | **100.0%** |
| **Strong scaling** | `DP=1 × FSDP=512` | shrunk to 1/4 | 1× | 404 | 89% |

64 chips with the same recipe give 580, and 256 chips give 580 as well — **per-chip does not drop at all**.

**Why the DP direction is free**: `DP=4 × FSDP=128` is just four independent 64-chip jobs.
Within a group, each layer does two FSDP collectives (80 layers = 160 of them), while **across groups the entire step has exactly one gradient all-reduce**:

```
gradient shard per device (bf16, FSDP=128) = 590 GB / 128 ≈ 4.6 GB
ring all-reduce transfer volume            = 2(p−1)/p × 4.6 = 6.9 GB   (p = 4)
v7 ICI per chip, bidirectional             = 1,200 GB/s
theoretical time ≈ 12 ms  →  0.05% of the 23.54 s step
```

Even conservatively assuming 1/6 of the bandwidth, that is only 35 ms (0.15%), and it can overlap with the tail of the backward pass.
**Two orders of magnitude difference in traffic — that is the fundamental reason "DP is cheap and FSDP is expensive".**

**Why strong scaling loses 11%**: `FSDP=512` spreads the same weights over 4× the devices,
shrinking each shard to 1/4 — **the number of collectives is unchanged, while each one carries only a quarter of the payload**,
and the fixed cost (synchronization, launch latency, multi-hop forwarding on a 3D torus) cannot be amortized.

> **Conclusion: when you add chips you must add batch at the same time.**

**Three boundary conditions**:

1. **Only measured up to DP=4.** The ring all-reduce volume `2(p−1)/p × N` approaches the constant `2N` as p grows,
   so DP=8/16 should still be near 100%, but **that is an inference, not a measurement**.
2. **Within-slice conclusion.** All 512 devices sit in one `4x8x8` slice over ICI;
   **DP across slices goes over DCN, whose bandwidth is more than an order of magnitude lower, and cannot be extrapolated.**
3. **The premise is that per-device work is unchanged.** Scaling out while holding global batch fixed degenerates into that 404 row.

### 4.2 A two-parameter HBM model: computing the batch ceiling without hitting OOM

Solve `HBM = static + slope × pdbs` using **two measured points from the same base**:

```
DP4×FSDP128:  74.20 G @ pdbs 8 , 91.93 G @ pdbs 12
              → static 38.7 G , slope 4.43 G / pdbs
DP2×FSDP256:  static 25.9 G (FSDP doubles, static halves), same slope
```

| Base | pdbs 8 | pdbs 10 | pdbs 12 | pdbs 14 | pdbs 16 |
|---|---|---|---|---|---|
| `DP4×FSDP128` predicted | 74.2 | 84.1 | 91.9 | 100.8 | 109.6 |
| `DP4×FSDP128` **measured** | **74.20** | **84.06** | **91.94** | — | **OOM** ✅ |
| `DP2×FSDP256` predicted | 61.4 | 73.5 | 79.1 | 87.9 | 96.8 → predicted OOM |
| `DP2×FSDP256` **measured** | **61.36** | — | **78.27** | **89.56** | **92.33** ❌ **prediction wrong** |

**Near-range interpolation is accurate** (pdbs 10 predicted 83.0 / measured 84.06, error 1.0 G; pdbs 14 predicted 87.9 / measured 89.56, error 1.7 G),
**far-range extrapolation systematically overestimates**.

The measured piecewise slope for `DP2×FSDP256`:

```
pdbs  8 → 12 :  4.23 G / pdbs
pdbs 12 → 14 :  5.65 G / pdbs
pdbs 14 → 16 :  1.39 G / pdbs   ← sharp drop
```

> ⚠️ **I got this model wrong twice, and both times it was the same class of error: treating sublinear growth as linear.**
>
> First time: extrapolating "activations grow linearly from zero" said pdbs 12 @ FSDP128 needs 98.5 G → OOM, and **it measured 91.93 G and ran**.
> After correcting to the two-parameter model, the second time: predicted pdbs 16 @ FSDP256 needs 96.8 G → OOM, and **it measured 92.33 G and ran**.
>
> The root cause is that under `remat_policy=custom` + `decoder_layer_input=offload`,
> XLA changes its recompute / offload scheduling as memory pressure rises, so activation growth slows noticeably in the high-batch regime.
>
> **Correct usage**:
> 1. Only interpolate **within ±2 pdbs of a measured point**, and **do not extrapolate more than 4 pdbs out**
> 2. A configuration predicted to be "just over the ceiling" (within ±5% of 94.74 G) **still has to be run** — both of my wrong calls were in that band
> 3. Only configurations predicted to be "far over the ceiling" (e.g. 109.6 G, 15% over) can be ruled out directly — that class was right both times

### 4.3 Small-scale screening: where it applies (this one has been corrected twice)

**Original conclusion** (16 chips vs 64 chips): MFU is only 7.7% lower, so small scale can be used for tuning.

**First correction** (2026-08-01): **small scale can eliminate losers, but cannot pick winners.**

| | 16-chip conclusion | 64-chip measurement |
|---|---|---|
| `remat_policy=full` | +1.22% | **−0.74%** (sign flip) |
| `shard_exp_on_fsdp` | +1.48% | **crash** (192 % 128 ≠ 0) |
| Delete 8 SparseCore flags | −0.01% | −0.00% (consistent) |
| `use_2d_fsdp_sharding` | −11.73% | not tested (already rejected) |

**Second correction** (2026-08-04, 64 vs 256 chip comparison): **the one above is too strict; classify by "does the change alter the shard shape".**

| Type of change | Examples | Transferable across scale? |
|---|---|---|
| **Does not alter the shard shape** | tokamax tile, pdbs | ✅ **Fully transferable.** 64 and 256 chips with the same recipe give 580 vs 580 and peak HBM 91.94 G vs 91.94 G, identical to the byte |
| **Alters the shard shape** | `remat_policy`, `shard_exp_on_fsdp`, FSDP width | ❌ Not transferable, and anything with a divisibility constraint must be verified at the target scale (192 % 32 = 0 but 192 % 128 ≠ 0) |
| Zero gain / heavily negative | SparseCore group, `use_2d_fsdp` | ✅ Transferable (safe to use for elimination) |

**There is also a case where small scale "underestimates" a winner**: tokamax tile is +8.4% on a single node and **+17.4%** on 256 chips.

> **Corrected operating procedure**:
> 1. Switches that do not alter the shard shape (tile, batch, kernel parameters) → **can be settled on 16 nodes and carried over directly**
> 2. Those that do alter the shard shape (parallelism, remat, sharding) → **must be verified at the target scale**
> 3. In every case, the "magnitude" has to be re-measured once at the target scale

### 4.4 Self-jitter is only 0.005%; the ±3% criterion is for cross-batch use

Same batch of pods, same configuration, three consecutive runs (64 chips / `DP1×FSDP128` / tile / pdbs 12):

| Run | step | chip |
|---|---|---|
| 1 | 23.5233 s | 580 |
| 2 | 23.5240 s | 580 |
| 3 | 23.5245 s | 580 |

**Range 1.2 milliseconds = 0.0051%.**

This matters for reading discipline, because the document contains two criteria at once and they are easy to confuse:

| Scenario | Noise magnitude | Criterion |
|---|---|---|
| **A/B within the same batch of pods** | **0.005%** | **A 1% difference is real; a single run decides it, no need to average over multiple runs** |
| Across clusters / a different batch of machines | 2.6–15% | ±3% is required to count as a successful reproduction; the 07-30 and 08-01 baselines differ by 15% (20.43 vs 17.43 s) |

> ⚠️ **Do not use ±3% to dismiss a 1–3% gain measured within the same batch.** That ±3% is a cross-batch reproduction criterion,
> and applying it to same-batch ablations throws away real gains as noise — in this round `DP2×FSDP256 + pdbs14` (585 vs 580, +0.9%)
> was exactly what I misjudged as "within noise", when it was a real difference.

### 4.5 A large share of time ≠ room to improve

On the v5p side, the baseline trace showed MoE's `tgmm` almost filling the sampling window, and I concluded from that we should push on MoE.
The result: **eight or nine rounds of experiments in the MoE GMM direction produced nothing, while touching a single switch on the attention side gave +3.33%.**

The reason is not hard to see — **megablox is already a path that people have tuned**,
whereas `sa_use_fused_bwd_kernel` was still off by default on v5p and nobody had ever turned it on.

> **What you should be looking for is "the place nobody has tuned yet", not "the place that takes the most time".**
> A trace tells you where the time goes, but it does not tell you where there is room —
> the latter comes from "has this been optimized before", which lives in the code and upstream records, not in the trace.

**The +17.4% from tokamax tile is the positive confirmation of this rule**: it was slow not because the algorithm is bad,
but because **the lookup table was missing the row for 192** — a place nobody had tuned at all.

---

### 4.6 What can be tuned and what cannot — one master table

> **If you read only one section, read this one.** Every row below is backed by measurement,
> and the "worth" column is the magnitude measured on v7 in this project, not a guess.

**Absolute water lines first, so the percentages are not read in isolation** (TFLOP/s/chip, full 80 layers):

| | Starting point | Now | On the back of |
|---|---|---|---|
| **BF16** (192e, 256 chip) | 445 | **599** | tile +17.4% / batch +9.0% / widening FSDP +3.3% |
| **FP8** (192e, 64 chip) | 594 | **625** | **QAG +5.3%** |
| **FP8** (256e, 64 chip, exploration) | — | **645** | QAG + `cost_estimate_flops` +0.9% |

**⇒ Those three numbers are the current ceiling.** Categories B and C below explain why going higher requires changing the model or writing code.

**A. Confirmed gains — ordered by value for money**

| What to tune | Worth | Applies to | Note |
|---|---|---|---|
| **tokamax tile** (BF16 path) | **+17.4%** | BF16 + `use_tokamax_gmm` | **It is fundamentally patching a broken default**, see the "⚠️" below |
| **batch / sequence convention** | **+12.8%** | all | Settle the convention before talking about optimization |
| **per_device_batch_size** | **+9.0%** (8→12) | HBM-limited | Estimate with the two-parameter model in [§4.2](#42-a-two-parameter-hbm-model-computing-the-batch-ceiling-without-hitting-oom); do not walk into OOM |
| **scheduler flag group** | **+6.6%** | all | Must be enabled as a group; removing a dependency kills it instantly |
| **`cost_estimate_flops_fwd/bwd=5e12`** | **+0.9%** | when using splash attention | Changes no computation, only gives the scheduler an accurate estimate of kernel duration, so communication hides better. The DSv3 official recipe has it; we did not |
| **widen FSDP to buy batch** | **+3.3%** | only at ≥ 256 chips | Widening itself is 1.9% slower; the profit is the 13 G of HBM it frees |
| **QAG** (all-gather after quantization) | **saves 4.5–11 G of HBM** | **only when `num_experts % FSDP == 0`** | What it saves is HBM, not time; the gain shows up as being able to run a larger batch |

**B. Confirmed no gain — stop spending time**

| What was tried | Result | Root cause |
|---|---|---|
| **MaxText tile configuration for FP8** (18 parameters) | **±0** | Proven from source: when `use_tokamax_gmm=True` and `use_gmm_v2` is off, `tiling` is discarded entirely and the parameters never reach the kernel |
| **FP8 monkeypatch tile** (pushed larger) | **±0 or negative** | `(512,2048,1536)` is already a local optimum. Increase tile_k → VMEM OOM; increase tile_m → −2.5%. **The optimum is stable across dtypes** |
| SparseCore offloading group | ±0 | Trim the 9 flags down to just the aggregator and performance is unchanged |
| `use_gmm_v2` | 70% of the gain is eaten by copies XLA inserts | — |
| `scan(unroll=N)` | No usable setting | 2 hits a kernel shape check, 10 needs 274 G |
| Official `tokamax.autotune` | Not a CLI; costs more than hand tuning | See [§5.4.1](#541-findings-on-the-official-autotune-2026-08-05) |
| `shard_exp_on_fsdp` alone | **Silently ineffective** (no error, no speedup) | `weight_gather_axes` is always empty when calibration is not `fixed` |
| **SparseCore offloading flag group** (filling in DSv3's 27) | ±0 | The three core offload flags are **already True by default** on Ironwood; even after turning off the mutually exclusive CF it is still ±0 — the gain has already been taken by the existing `collective_aggregator` + `latency_hiding_layer_scheduler` |
| Copying DSv3's 36 XLA flags wholesale | **HBM OOM** | Someone else's flags were tuned against someone else's memory budget |

**C. Locked down by structure — no configuration solves it; only changing the model or the framework does**

| What you want | Why you cannot have it | Way out |
|---|---|---|
| **Enabling QAG freely with 192 experts** | FSDP has only one remaining option, 64 (96 cannot produce an integer DP, 128 does not divide evenly), and batch has to be squeezed to 7 to run at all; **and adding chips does not help** (shard thickness is determined solely by FSDP width) | It runs, but the path is narrow (measured 625.4). **Make the next-generation model's expert count a power of two** and FSDP becomes a free choice |
| **Expert parallelism EP** | TPU is a 3D torus, AllToAll is multi-hop, measured **−71%** | Do not use it; prefer FSDP |
| **DSV3's batch split scheduler** | It switches to a DeepSeek-specific hand-written decoder, and Hy3 being GQA gives an immediate `KeyError: 'wq_a'` | Several hundred lines of development; a development task, not tuning |
| **DSV3-level MXU utilization** | Its `emb=7168`, ours is `4096`; the bigger the matrix, the better the MXU pays off | Determined by model shape; no amount of tuning flattens it |

> ⚠️ **"tokamax tile is worth 17.4%" is the row most likely to be misused.**
> The reason it is worth that much is that tokamax's lookup table **has no row for 192 at all**
> and it falls into an extremely bad default ([§3.4.2](#342-root-cause-the-kernel-library-lookup-table-has-no-row-for-192)) —
> **that is fixing a broken default, not routine tuning.**
> I once carried that number over to the FP8 path as an expectation (deriving 726),
> and **measurement showed zero gain** — because FP8 goes through a different kernel whose default is not broken.
> **The magnitude of a gain cannot be moved across kernels.**

#### 4.7 Four rules for telling whether a gain is real or fake

All four were distilled after tripping over them in this project:

1. **A negative result is not a conclusion on its own — you have to run the reverse test.**
   If you change a parameter and performance does not move, there are two possibilities: *the change was right but useless*, and *the change never landed*.
   The way to distinguish them is to **deliberately set an obviously bad value**: if performance drops ⇒ the parameter is live and the default was already good enough;
   if performance does not drop ⇒ the parameter never reached the kernel. The follow-up actions for these two are completely different.

2. **A numerical coincidence is not mechanistic evidence.**
   I once concluded from "the official DSV3 743.5 ≈ the 746 I computed" that "even DSV3 does not have QAG on",
   then checked the official recipe and found they do. **When numbers from two independent systems happen to collide,
   with only one observation point that says nothing about causality.**

3. **A check existing in the source ≠ that check firing on your code path.**
   `pyconfig_deprecated.py` explicitly contains the expert-count divisibility check,
   yet in practice it was `shard_map` that crashed at runtime — our version goes through `types.py`.
   The difference is that **it has to compile for a minute or two before failing, rather than rejecting instantly**.

4. **A paired experiment must differ in exactly one variable, and you have to count that at design time.**
   I had one round where I changed both "turn on QAG" and "FSDP 128→64";
   the +12.1% could not be attributed to either, and the round was wasted.
   Ironically it was a set added on the side (same parallelism, same batch, differing only in the QAG switch)
   that produced the one clean net gain. **When you lay out the experiment matrix, write out the diff for each round and check it first.**

> 💡 **One positive addendum**: **failed rounds are measurements too.**
> Four OOM rounds look like a total loss, but the `total memory required` that XLA reports is an exact value —
> it was the difference between two OOMs (115.19 → 104.11) that gave us "QAG saves 11.08 G at the full 80 layers",
> the only full-model-scale measurement we have. **Do not look only at "did it run"; the error messages often contain numbers.**

---

## 5. FP8 and QAG: what could be taken has been taken; the rest requires changing the model

**Currently 618 TFLOP/s/chip, which against the FP8 peak (4,614) is an MFU of only 13.4%. The official DSV3 figure is 743.5 (16.1%),
so we are 20.3% behind.**

> ⚠️ **Any FP8 number must state its denominator.** 618 is 26.8% against the BF16 peak and only 13.4% against the FP8 peak —
> a factor of two apart. **FP8 numbers can only be compared with FP8 numbers.**

### 5.1 Two FP8 water lines: 618 (no QAG) → 625 (QAG on)

| | BF16 | FP8 |
|---|---|---|
| Starting point | 445 | 594 (64 chip, no QAG) |
| Now | **599** (256 chip) | **625** (64 chip, QAG on) |
| Main source | tokamax tile **+17.4%** + batch | **QAG +5.3%** (against 594 at the same scale) |

> 🛑 **This section was originally titled "618 is the FP8 equivalent of BF16's 445 — a starting point, not an end point",
> meaning "FP8 has never been tuned at all and there is plenty of room". Measurement on 2026-08-05 overturned that premise:**
> **FP8 still goes through tokamax, and those 6 lines of tile monkeypatch have been in effect the whole time** —
> 618 is not an "untuned starting point"; it already carries the optimal tile found on BF16.
> See [validation experiments 8 and 9](#542-qag-quantize-first-then-communicate-a-path-blocked-by-the-expert-count).

**FP8 measured at two scales** (with the FP8 peak of 4614 as denominator):

| Scale | Configuration | chip | MFU vs FP8 peak | Same-config BF16 | Δ | Peak HBM |
|---|---|---|---|---|---|---|
| 256 chip | `DP2×FSDP256` pdbs 16 | 618 | 13.39% | 599 | +3.2% | 92.80 G |
| 256 chip | `DP4×FSDP128` pdbs 12 | 608 | 13.18% | 580 | +4.8% | 94.35 G |
| **64 chip** | `DP1×FSDP128` **pdbs 10** | **594** | **12.87%** | 561 | **+5.9%** | 86.20 G |

> The 594 at 64 chips already **beats the BF16 best of 580 (+2.4%)**, and it only uses 86.2 G.
> **"Feed the HBM that FP8 saves back into batch" was tried on 2026-08-05** —
> with QAG on it reaches pdbs 11 (256e) / 7 (192e), and **beyond that remat is exhausted and batch cannot be fractional**,
> see [validation experiment 12](#542-qag-quantize-first-then-communicate-a-path-blocked-by-the-expert-count).
>
> ⚠️ `pdbs 12 + FP8 + tile(512,2048,1536)` at 64 chips is **not an OOM but a kernel rejection**
> (`MosaicTpuRaggedDot` errors out), see [Appendix B.2](#b2-crashes--configuration-rejections).

**These two tracks use different kernels** — BF16 goes through `tokamax.ragged_dot`, FP8 goes through `mblx.gmm`.

> 🛑 **I inferred from this that "the FP8 tile has to be tuned separately and has never been swept once" — that inference was wrong.**
> Under `use_tokamax_gmm=True`, `mblx.gmm` **still falls back to tokamax internally**,
> and the monkeypatch takes effect as usual; what actually gets discarded is MaxText's `w{i,o}_tile_*`.
> **The two tracks share one set of tiles; they neither need to be nor can be tuned separately.**

Rough arithmetic: if the tile on the FP8 path could capture a gain of the same magnitude, `618 × 1.174 ≈ 726` (MFU 15.7% on the FP8 peak basis),
only 2.4% short of DSV3's 743.5.

> 🛑 **That 726 has been disproven by measurement; it is kept here only to record the chain of reasoning.**
> On 2026-08-05, after filling in the 18 tile parameters from the DSv3 official recipe, measurement gave **zero gain**
> (step 19.5980 → 19.5980, see [§5.4.2 validation experiment 6](#validation-experiment-6-adding-the-18-dsv3-tile-settings--zero-gain-one-core-assumption-disproved)).
> The error in that extrapolation was: **the +17.4% on the BF16 side is patching a broken default
> (the tokamax lookup table is missing the row for 192), not a routine tuning gain,
> and moving it onto another kernel (`mblx.gmm`) as an expectation is an invalid analogy.**
>
> On top of that, 726 vs 743.5 is not an apples-to-apples comparison: **DSV3's 743.5 was obtained with QAG on**,
> while our estimate does not include QAG.

### 5.2 Why the FP8 tile was never tuned: two GMM paths

```python
# moe.py:1500
if self.config.use_tokamax_gmm:
    if self.config.quantization or self.config.use_gmm_v2:
        output = mblx.gmm(..., tiling=tiling, ...)   # ← FP8 takes this one, consumes MaxText's w{i,o}_tile_*
    else:
        output = tokamax.ragged_dot(...)             # ← BF16 takes this one, consumes the monkeypatch
```

**Turning on FP8 means switching kernel paths**, and the 6-line monkeypatch used on BF16 (which patches `PallasMosaicTpuRaggedDot`)
executes not a single line under FP8. The 618 (13.4% against the FP8 peak of 4614) was produced by "FP8 + MaxText default tile",
and the default `*_mlp_dim=1024` **does not divide `base_moe_mlp_dim=1536`**.

**The cost of tuning it also differs:**

| | How tile is changed | Recompile? | Time per round |
|---|---|---|---|
| BF16 (tokamax) | Runtime monkeypatch | no | 6–8 minutes |
| FP8 (mblx.gmm) | A MaxText config option, **goes into the HLO** | **yes, the whole cache is invalidated** | **> 30 minutes** |

So BF16 can sweep 15 tile combinations overnight, while every point on the FP8 side costs half an hour or more. **That is the direct reason it had not been tuned.**

### 5.3 We cannot use DSV3's `use_batch_split_schedule`

Inside `configure_quantization`, only turning it on returns a complete `QwixQuantization`:

```python
if getattr(config, "use_batch_split_schedule", False) and config.quantization:
    if config.quantization == "fp8_full" and not config.use_manual_quantization:
        return QwixQuantization(...)
    return None
if config.use_qwix_quantization:
    return None        # ← this is the branch we take
```

**But on Hy3 it gives an immediate `KeyError: 'wq_a'`, and all four rounds (including the BF16 control round) died instantly.**

Root cause: `models/deepseek_batchsplit_fp8.py` — the filename itself hard-codes deepseek. Inside, it all-gathers
MLA's seven weights one by one:

```python
params["self_attention"]["wq_a"]["kernel"]
(wq_a, wq_b, q_norm, wkv_a, wkv_b, kv_norm, out)
```

**Hy3 is GQA and has no `wq_a` / `wq_b` / `wkv_a` / `wkv_b`.**

> 🔁 **This is the 11th recurrence of that lesson in this project**: every "which path should this model take" decision in MaxText
> is a table hard-coded by model family. This one is especially well hidden — the config option is called `use_batch_split_schedule`,
> which sounds like a general scheduling strategy, but **it actually means "switch to the DeepSeek-specific hand-written decoder"**,
> something the name gives no hint of.

**Inference**: DSV3's 743.5 is **three things stacked** — quantization itself, that hand-written MLA scheduler,
and **QAG** (confirmed by checking the official recipe on 2026-08-05, see [§5.4.2](#542-qag-quantize-first-then-communicate-a-path-blocked-by-the-expert-count)).
**How much each contributes cannot currently be separated.** Leave some room when using it as the sole yardstick for FP8.

### 5.4 Next steps (ordered by value for money)

| # | What to do | Basis | Cost |
|---|---|---|---|
| **1** | **Sweep `w{i,o}_tile_*` on the FP8 path** | Change the six `*_mlp_dim` from 1024 to 1536 / 512 (both divide 1536). On 16 chips this change is worth **+8.25%** | 30 minutes per round |
| **2** | **Turn on QAG** | The gates are fully mapped and there is an executable recipe (§5.4.2). It moves the 57.3% that is communication, **so its gain does not overlap with tile's**; Ant measured 0.88× → 1.05× BF16 | 30 minutes per round, plus convergence validation |
| 3 | Set the tile by mapping from the BF16 optimum | The BF16 optimum is `(512, 2048, 1536)`; for wi, `k` is emb (4096) and `n` is mlp (1536), and wo is the other way around. The mapping has to be worked out | same as above |
| 4 | Run the official autotune to generate cache entries | Replaces hand tuning | **Already investigated, see §5.4.1 — lower priority than 1** |
| 5 | Write a batch split implementation for GQA | Possibly the largest gain, but **a development task, not tuning**, several hundred lines | high |

> 1 and 2 **should be done separately and accounted for separately**: tile moves computation, QAG moves communication,
> and changing both at once mixes the two gains together beyond separation.

#### 5.4.1 Findings on the official autotune (2026-08-05)

**Feasible, but not "run one command"; not recommended as the next step for now.**

tokamax exposes `tokamax.autotune` (`_src/autotuning/{autotuner,api,cache}.py`),
but it is a **library-level API, not a CLI**: `Autotuner.autotune(fn_factory, configs, *args, kwargs)`
— you have to prepare both the candidate config set and input tensors of the real shapes yourself.

Four blockers:

1. **There is no environment-variable switch.** The whole library only has `TOKAMAX_NAME/VERSION/VERSION_INFO`;
   **there is no `TOKAMAX_AUTOTUNE=1` one-liner**, so glue code is mandatory.
2. **You have to feed the inputs yourself.** In the MaxText training flow, lhs/rhs/group_sizes are sharded, and pulling them out to feed separately is extra work.
3. **The cache is indexed by operator signature** (`ba.autotuning_cache_key`), and after generating it you still have to
   hook it in via `get_autotuning_cache_overlay_state()` for it to take effect.
4. 🔴 **It only helps the tokamax path. And FP8 goes through `mblx.gmm`**
   ([§5.2](#52-why-the-fp8-tile-was-never-tuned-two-gmm-paths)) —
   **autotune cannot solve the tile problem on the FP8 path.**

> ⇒ The tile on the BF16 path has already been hand-tuned into place (`(512,2048,1536)`, verified stable across three different `m` values),
> so autotune would squeeze out single-digit percent at best and does not address the biggest blank (FP8). **Lower priority than directly sweeping `w{i,o}_tile_*`.**

#### 5.4.2 QAG (quantize first, then communicate): a path blocked by the expert count

> **It started with a question**: Chris asked, "when you are in FP8, isn't what gets communicated FP8 too?"
> My assumption at the time was "communication is unaffected by FP8" — that assumption was half wrong,
> and following it down uncovered QAG, plus a conclusion that **points straight at next-generation model design**.

##### First, the answer to that question: by default, what is communicated really is bf16

`QwixDotGeneral.__call__` just wraps `dot_general_qt` —
**quantization happens inside dot_general: quantize temporarily before computing, output bf16 after.**
Weights are stored as `weight_dtype=float32` and computed in `dtype=bfloat16`.
**So FSDP's all-gather carries bf16 and does not save a single byte.**

That is exactly why the Amdahl ceiling of 746 in [§5.1](#51-two-fp8-water-lines-618-no-qag--625-qag-on)
had to be built on the premise that "communication saves nothing".

##### But the framework does have QAG; we just never triggered it

`kernels/megablox/ops.py:190-197`:

```python
# QAG is only supported for following conditions
if use_tokamax_backend:
  if quantization_rule and quantization_rule.bwd_qtype:
    if quantization_rule.weight_calibration_method.startswith("fixed") \
       and isinstance(rhs, qpl.QArray):
      if weight_gather_axes:
        rhs_qvalue = jax.lax.all_gather(rhs.qvalue, axis_name, axis=axis_idx, tiled=True)
        rhs = dataclasses.replace(rhs, qvalue=rhs_qvalue)
```

**What it all-gathers is `rhs.qvalue` (the quantized FP8 bytes) rather than the bf16 weights** —
quantize first, then communicate, and **the weight all-gather traffic is halved outright**.

Four trigger conditions, each chased to the bottom:

| # | Condition | Us | Where it leads |
|---|---|---|---|
| 1 | `use_tokamax_backend` | ✅ | `use_tokamax_gmm=True` |
| 2 | `quantization_rule.bwd_qtype` non-empty | ✅ | FP8 sets `e5m2` |
| 3 | `weight_calibration_method` starts with `"fixed"` | ❌ | `base.yml:151-153` defaults to `absmax` |
| 4 | `weight_gather_axes` non-empty | ❌ | `moe.py` → `explicitly_weight_ag(config.shard_exp_on_fsdp)` |

##### But these four are not parallel: 3 and 4 are the same lock

Only by chasing into `moe.py:1556-1561` does it become clear that condition 4 internally **checks fixed a second time**:

```python
def explicitly_weight_ag(shard_exp_on_fsdp):
  if shard_exp_on_fsdp:
    quantization_rule = qpl.get_current_rule("gmm")
    if quantization_rule and quantization_rule.weight_calibration_method.startswith("fixed"):
      return True          # ← only fixed will produce weight_gather_axes
  return False
```

**Turning on `shard_exp_on_fsdp` alone while calibration is still `absmax` leaves `weight_gather_axes` permanently empty,
so QAG silently does not trigger and does not error.** Both switches must be on together, and missing one is silent failure rather than a crash —
this is the easiest place to misjudge as "I turned it on but it did nothing".

The same passage also reveals **why fixed is required** (`moe.py:1598-1607`):
under the `fixed` branch the weights use **normal expert sharding** (`wi_kernel_axes`), while the non-fixed branch uses DSv3's `mlp_no_fsdp`.
Chasing down into qwix `qarray.py:517-530`, the root cause is one line:

```python
elif method == 'fixed':
  ...
  shape = tuple(1 for _ in shape)   # ← Fixed calibration is always per-tensor
```

**`fixed`'s scale is a per-tensor scalar, while `absmax`'s scale is an array sharded along with the tensor.**
QAG all-gathers `rhs.qvalue` without gathering the scale —
that is only correct when the scale is a scalar (identical on every shard).
`absmax` has an even harder reason: its scale depends on the global maximum,
and computing that is itself a **blocking network reduction**, so quantize-then-communicate does not even hold semantically.

##### The fifth gate (only hit in practice): `fixed` is not a legal value

Changing `weight_quantization_calibration_method` to `fixed` gives an immediate:

```
ValueError: A fixed range is required for fixed calibration.
```

`fixed` is only the method name; **the range has to be written into the same string** (`qarray.py:301`, format `<method>[,<args>]`):

| Form | Meaning |
|---|---|
| `fixed` | ❌ errors |
| `fixed,224` | symmetric range `[-224, 224]` |
| `fixed,-224,224` | explicit bounds, requires `lo ≤ 0 ≤ hi` |

**The official canonical value is `fixed,-224,224`** — three independent MaxText tests
(`tests/integration/tokamax_test.py:112`, `tests/unit/moe_test.py:1468`,
`tests/batchsplit_google_test.py:156`) all use this set.
224 is half of e4m3's maximum of 448, leaving 2× headroom.

⚠️ **Leave `bwd` on `absmax` and do not touch it** — all three tests set
`weight` / `act` to fixed and leave `bwd` on absmax. The QAG criterion only looks at the weight path.

On the MaxText side the string is **passed through verbatim** (`quantizations.py:654-656` → qwix `rhs_calibration_method`),
with no parsing or validation in between, so a value with commas can be written straight on the command line.

##### The exact place where 192 experts hits the wall

`shard_exp_on_fsdp=True` is rejected outright on 128 devices ([Appendix B.2](#b2-crashes--configuration-rejections)).
The source contains two **explicit up-front checks** (`configs/pyconfig_deprecated.py:1212-1215`):

```python
if raw_keys["shard_exp_on_fsdp"] and raw_keys["num_experts"] % raw_keys["ici_fsdp_parallelism"] != 0:
  raise ValueError("shard_exp_on_fsdp requires num_experts is divisiable by ici_fsdp_parallelism.")
if raw_keys["shard_exp_on_fsdp"] and (using_tensor_parallelism(raw_keys) or using_expert_parallelism(raw_keys)):
  raise ValueError("shard_exp_on_fsdp requires ici_expert_parallelism = 1 and ici_tensor_parallelism = 1.")
```

**This point matters: the constraint is `num_experts % ici_fsdp_parallelism == 0`,
not "the expert count must be a power of two".** It is a divisibility relation between FSDP width and expert count,
and it only looks like a "192 is not a power of two" problem because we have always used powers of two as the FSDP width.

> ⚠️ **But what actually stopped it in practice was not those two checks.** I had earlier written, following the source, that "the interception point is at the config layer";
> the real run (round S2) reported a **`shard_map` runtime error**:
> *`shard_map applied to the function 'sparse_matmul_route_and_compute' was given
> argument arrays with axis sizes that are not evenly divisible by the corresponding
> mesh axis sizes`*, and that `divisiable` config error **never appeared once**
> (`grep -c` returned 0) — our version goes through `configs/types.py` (pydantic),
> and the line in `pyconfig_deprecated.py` never executes.
>
> **The conclusion is unchanged (the constraint is divisibility and the workaround works), but the failure surfaces late, at shard_map** —
> which means **it has to compile for a while before crashing rather than rejecting instantly** (S2 burned 128 seconds).
> The lesson is still the same: **reading a check in the source does not mean it fires on your code path.**

Laid out against the FSDP widths we have actually used:

| | **192 experts (Hy3)** | **256 experts** |
|---|---|---|
| 4 chips (8 dev) | ✅ | ✅ |
| 16 chips (32 dev) | ✅ | ✅ |
| 32 chips (64 dev) | ✅ | ✅ |
| **64 chips (128 dev)** | **❌ remainder 64** | ✅ |
| **128 chips (256 dev)** | **❌ remainder 192** | ✅ |

> 🎯 **This table is the thing most worth taking away from this section, and it is a model design conclusion, not a tuning conclusion.**
>
> **With our optimal FSDP=128/256, 192 is fine at ≤32 chips and dead the moment you reach 64 chips;
> 256 divides evenly at every scale.**
> ⇒ **The expert count decides, at model design time, whether this model can use QAG conveniently.**
> Picking a power of two (256 / 128) fits any FSDP width; picking 192 means either changing FSDP or being locked out.

##### 192 does have a solution: FSDP of 64 / 96 / 48

Since the gate is divisibility rather than powers of two, the factors of `192 = 2⁶ × 3` include usable FSDP widths:

| Chips | device | Usable FSDP (divides 192) | How to configure |
|---|---|---|---|
| 64 | 128 | **64** | `DP2 × FSDP64` |
| 128 | 256 | **64 / 96** | `DP4 × FSDP64` |
| 256 | 512 | **64 / 96** | `DP8 × FSDP64` |

The cost is explicit: [§3.7](#37-how-to-split-parallelism-the-usable-range-is-only-fsdp--128-256) measured that
**static shard HBM cannot be absorbed once FSDP is narrower than 128**, and Appendix B.1 records an OOM at `FSDP=64`.
So this path is "**trade HBM for QAG**", and whether it works depends on
whether the HBM QAG saves (the weight all-gather buffer halving) is enough to offset the cost of thicker shards —
which is exactly what round S3 below sets out to measure.

##### A known precedent: Ant's ALModel (verified 2026-08-05, with one important correction)

Chris mentioned that "Ant's ALModel turned FP8 from a negative into a positive by enabling QAG".
**Verified true, and with exact numbers**:

| | step time | Relative to BF16 |
|---|---|---|
| BF16 baseline | 3.93 s | 1.00× |
| FP8, before QAG | 4.48 s | **0.88× (negative)** |
| FP8, after QAG | **3.77 s** | **1.05× (positive)** |

The two gain mechanisms recorded officially match the code exactly:

1. **Expert weight communication volume halved** (bf16 → fp8)
2. **Expert weight quantization cost cut to 1/128** (at FSDP=128) —
   it used to all-gather first and then quantize each shard separately; now it quantizes once and then gathers

> ⚠️ **One important correction: Ant is not on this upstream `fixed` path.**
> Their recipe explicitly says "Using absmax for quantization calibration,
> **instead of** `fixed,-224,224`", for accuracy reasons.
> To be able to enable QAG under absmax as well, they **applied a local patch**,
> cutting `startswith("fixed") and isinstance(rhs, qpl.QArray)` in `ops.py`
> down to just `isinstance(rhs, qpl.QArray)`.
>
> So "ALModel with QAG on" and "us with QAG on" **are not the same code path**:
> we take the upstream static scale, they take a patched dynamic scale.
> The preliminary official analysis of dynamic scale + QAG still concludes it **may be inefficient**,
> and that patch has not gone upstream either.

Ant's target is **1.2–1.3× BF16, and 1.05× has not met it**,
which shows that **even with QAG on, FP8 on Ironwood is far from the end** — consistent with our assessment in §5.1.

A few other choices in Ant's recipe worth noting: per-expert scale (`[E,1,N]` rather than `[1,1,N]`),
**E4M3 for both forward and backward** (not the default bwd e5m2), and **keeping the last layer in BF16**;
at 4000 steps the loss deviates from the BF16 baseline by 0.14%.

##### Why this is worth more than tuning tile

Tile only moves the 39.2% that is compute, whereas **QAG moves the weight all-gather inside the 57.3% that is communication** —
it raises the Amdahl ceiling itself.

##### ⚠️ Self-falsification: the side evidence that "even DSV3 does not have QAG on" was wrong

I had written: *"The official DSV3 743.5 is almost exactly the 746 you get from 'compute halved only',
which shows even DSV3 does not have QAG on"*. **Checking the official recipe, that inference does not hold.**

The official FP8 recipe for DSv3-671B on Ironwood
(`tpu-recipes` → `training/ironwood/deepseek3-671b/4k-fp8-tpu7x-4x4x8`):

```bash
ici_fsdp_parallelism=-1                              # L80  spread over all devices
fsdp_shard_on_exp=True                               # L92  = the old name for shard_exp_on_fsdp
use_tokamax_gmm=True                                 # L107
use_qwix_quantization=True  quantization=fp8_full    # L111-112
weight_quantization_calibration_method=fixed,-224,224  # L131
act_quantization_calibration_method=fixed,-224,224     # L132
```

**All four conditions present — the official DSV3 recipe has QAG on.**
743.5 ≈ 746 is a coincidence, not evidence of "QAG off".

Two lessons to record:

1. **A numerical coincidence cannot serve as mechanistic evidence.** When numbers from two independent systems collide,
   with only one observation point that says nothing about causality.
2. It also incidentally confirms the divisibility table: DSv3 has **256 experts** and `ici_fsdp_parallelism=-1` (spread over everything,
   4×4×8 = 128 chips = 256 devices), and `256 % 256 = 0` —
   **it can use `-1` to spread over everything precisely because the expert count is a power of two.**
   The same line with 192 experts would be rejected by the config on the spot.

##### ✅ Correction: 192 experts can enable QAG, at the cost of batch being squeezed to 7

W3 = `192e / FSDP64 / QAG / pdbs 7` **ran**:

| Round | experts | FSDP | pdbs | TFLOP/s/chip | Peak HBM | NaN |
|---|---|---|---|---|---|---|
| V3 | 192 | 64 | 8 | ❌ needs 95.38 G | — | — |
| **W3** | **192** | **64** | **7** | **625.4** | **92.42 G** | **0** |

**That 625.4 is quite convincing**: in [§5.1](#51-two-fp8-water-lines-618-no-qag--625-qag-on),
the established FP8 water line at 64 chips is **594** (`FSDP128 / pdbs 10`, no QAG).
**W3 has a smaller batch (7 vs 10) yet is 5.3% faster.**

⇒ **Corrected conclusion: 192 experts can enable QAG, but FSDP can only be 64 and the batch ceiling is squeezed to 7.**
It is not "no path", it is "a narrow path".
**Still netting 5.3% at a smaller batch shows the communication QAG saves really does outweigh the cost of the narrow FSDP.**

> 🔁 **Tripped over the same thing again: extrapolating from a single-point failure to "infeasible".**
> Having just written in [§4.7](#47-four-rules-for-telling-whether-a-gain-is-real-or-fake) that "a negative result is not a conclusion on its own",
> I turned around and treated "pdbs 8 OOMs" as "this path is closed", **passing judgment one notch short of the data**.
> **An OOM only says "this batch does not work", not "this configuration does not work" —
> when you are 657 MB short, the first reaction should be to drop one more notch, not to write a conclusion.**

⚠️ **A control still to be filled in**: `192e / FSDP64 / pdbs 7 / no QAG`.
Only that gives the net gain of QAG on 192.
V4 (`pdbs 8` without QAG) already OOMed and by a larger margin, so **without QAG, pdbs 7 very likely does not run either** —
if so, this is another example of "QAG unlocking a setting that could not otherwise run".

> 🎯 **The divisibility table still holds, but the wording needs to be more precise:**
> making the expert count a power of two decides not "whether you can use QAG",
> but **"how much using QAG costs you"**:
> - **256 experts**: pick any FSDP; `FSDP128 / pdbs 11` runs comfortably with HBM to spare
> - **192 experts**: FSDP has one option left, 64; batch is squeezed to 7; it barely runs
>
> 192 shows no problem at all at ≤32 chips, and at 64 chips there is only one narrow path left —
> **that cost cannot surface during small-scale validation.**
##### Stage wrap-up: 645 is the ceiling within "no model changes, no code written"

On the afternoon of 2026-08-05, three consecutive rounds covered **8 cells, and not one produced a positive result**:

| Direction | Cells | Result |
|---|---|---|
| MaxText tile configuration | 1 | ±0 (the parameters never reach the kernel) |
| FP8 tile (monkeypatch, pushed larger) | 3 | 2 cells VMEM OOM, 1 cell **−2.5%** |
| XLA flags / SparseCore offloading | 4 | 3 cells ±0, 1 cell HBM OOM |

> 🎯 **The value of these 8 cells is not in the improvement, it is in draining the remaining search space.**
> Together with [the §4.6 master table](#46-what-can-be-tuned-and-what-cannot--one-master-table), we can now say clearly:
> **without changing the model and without writing code, 645 (256 experts) / 625 (192 experts) is close to the limit.**

**Four things not yet ruled out** (ordered by feasibility):

| # | Direction | Notes |
|---|---|---|
| ~~1~~ | ~~push to pdbs 12~~ | ❌ **Already ruled out**, see validation experiment 12 below |
| 2 | **Move to 256 chips** | [§4.1](#41-scaling-weak-scaling-100-strong-scaling-loses-11) measured weak scaling at 100%; adding batch along with the chips scales proportionally |
| 3 | Write a batch split decoder for GQA | Several hundred lines, **a development task, not tuning**, with unknown gain |
| 4 | Change the model shape (`emb 4096` → larger) | DSv3 is 7168; the bigger the matrix, the better the MXU pays off. **A next-generation model design decision** |
##### Measurement overview: 12 rounds of experiments in one table

**The conclusions of all 12 rounds below have been folded into the text above and [the §4.6 master table](#46-what-can-be-tuned-and-what-cannot--one-master-table)**,
so only an index is kept here; to see how a round was done, what error it produced, and how to read the numbers, expand the fold below.

| # | The question asked | Answer |
|---|---|---|
| 1 | Does QAG run at all at small scale? | Yes, but it uncovered **the fifth gate** (`fixed` must carry a range) |
| 2 | What is the net gain of QAG at 64 chips? | **+15.6%** (256e, paired at the same parallelism), plus 4.47 G saved |
| 3 | Can the full 80 layers run? | All four rounds OOMed, but they read out that **QAG saves 11.08 G on the full model** |
| 4 | Does switching to `FSDP128+256e` work? | Still OOM, but **only 1.88 G short**, and it proves not narrowing FSDP saves more |
| 5 | What about one notch lower on batch? | ✅ **First successful run: 645.0**; turning QAG off at the same batch OOMs outright |
| 6 | What about adding DSv3's 18 tile settings? | **±0** — disproves "tuning tile gets to 726" |
| 7 | Then where does that +0.9% come from? | 2×2 factorial: all from `cost_estimate_flops`, tile is 0 |
| 8 | Is tile useless, or not wired up? | **Not wired up** — the source proves `tiling` is discarded on our branch |
| 9 | Then who governs the FP8 tile? | **The monkeypatch is the only source**; remove it and it is 8.5× slower |
| 10 | Can FP8 use a larger tile? | No, all three points fail; `(512,2048,1536)` is a local optimum |
| 11 | What about filling in DSv3's 27 XLA flags? | **±0**, including the full SparseCore offloading set |
| 12 | Can batch still be pushed to 12? | No, remat is exhausted and batch must divide the device count |

<details>
<summary><b>Expand: the full process, errors, and data of the 12 rounds</b></summary>

##### Validation experiment 1: 4-chip path check (2026-08-05, done)

The 64 chips were occupied at the time, so the path was checked on 4 chips first — on a single node `192 % 8 = 0`,
and `shard_exp_on_fsdp` can be enabled at small scale anyway, which is enough to answer "does it run at all".
6 layers / pdbs 4:

| Round | Configuration | step (s) | TFLOP/s/dev | Peak HBM | NaN | Result |
|---|---|---|---|---|---|---|
| Q0 | BF16 baseline | 1.2228 | 196.6 | 69.62 G | 0 | reference point |
| Q1 | `+shard_exp_on_fsdp` | 1.2230 | 196.6 | 69.97 G | 0 | **no loss on its own**, only 0.35 G more |
| Q2 | `+FP8` (absmax) | 1.2375 | 194.3 | 74.55 G | 0 | −1.2% on a single node |
| Q3 | `+fixed` (bare) | — | — | — | — | ❌ `A fixed range is required` |
| Q4 | `+shard_exp`, all four conditions | — | — | — | — | ❌ same as above |
| Q5 | Q4 + `num_experts=256` | — | — | — | — | ❌ same as above, **never reached the divisibility check** |

Two takeaways:

1. **Q1 is a valuable negative result**: turning on `shard_exp_on_fsdp` by itself
   (with calibration still absmax) gives **exactly the same performance** (1.2228 → 1.2230, within the 0.005% jitter).
   This is the "silent failure" described above — it really did not trigger QAG, but it also did not error.
   **Do not read "I turned on `shard_exp_on_fsdp` and nothing changed" as "QAG is useless".**
2. **Q3/Q4/Q5 all died on the same error**, which is how the fifth gate was discovered.
   Note that **Q5 died on the fixed range and never reached the divisibility check** —
   so this round **did not** answer "can 256 experts enable QAG".

⚠️ **A single node cannot measure the QAG gain anyway** (what it saves is cross-device communication); this round has binary significance only.

##### Validation experiment 2: the 64-chip criterion (2026-08-05, done)

Rerun with `fixed,-224,224`, verifying the divisibility workaround at the same time.
128 devices / 16 layers / pdbs 8:

| Round | experts | Parallelism | calibration | `shard_exp` | step (s) | TFLOP/s/**chip** | Peak HBM | NaN |
|---|---|---|---|---|---|---|---|---|
| S0 | 192 | DP1×FSDP128 | — (BF16) | — | 3.7443 | 550.6 | 54.33 G | 0 |
| S1 | 192 | DP1×FSDP128 | absmax | off | 3.6120 | 570.8 | 55.30 G | 0 |
| S2 | 192 | DP1×FSDP128 | `fixed,-224,224` | **on** | — | ❌ `shard_map` not divisible | — | — |
| S3 | 192 | **DP2×FSDP64** | `fixed,-224,224` | **on** | 3.2222 | **639.8** | 57.09 G | 0 |
| S4 | 256 | DP1×FSDP128 | absmax | off | 3.9258 | 525.6 | 60.04 G | 0 |
| S5 | 256 | DP1×FSDP128 | `fixed,-224,224` | **on** | 3.3960 | **607.6** | 55.57 G | 0 |

**⚠️ This batch is a shrunken 16-layer / pdbs 8 configuration; the absolute values cannot be compared with the 618 in §5.1 or the 580 in §3.**
(The effect of fewer layers is actually small: the full 80 layers at `FSDP128 / pdbs 8` give 543, and 16 layers at the same batch give 550.6.
What really drags the absolute value down is **pdbs dropping from 12 to 8**. Lowering batch was my misjudgment, see below.)

**Four conclusions:**

1. **QAG net gain = +15.6%, and this one is clean.**
   `S4 → S5` is a perfect pair: same 256 experts, same `DP1×FSDP128`, same pdbs,
   **the only difference being the QAG switch** — 525.6 → 607.6.
   Ironically, this cleanest data point came from the 256-expert round added "just to check",
   not from the 192 rounds I had deliberately designed.
2. **QAG also saves HBM: 60.04 G → 55.57 G, a saving of 4.47 G.**
   Gathering after quantization halves the weight all-gather buffer outright.
   **The HBM saved can be traded for a larger batch** — that is the second-order gain, currently being measured.
3. **The 192 workaround holds** (S3 ran, no NaN, and 639.8 is the highest of the six rounds).
   But `S3 vs S1` (570.8 → 639.8, +12.1%) **changed both QAG and FSDP width,
   mixing two variables, so it cannot be taken as the net gain of QAG** — I missed the same-parallelism control group.
   To get the net gain on 192, the round `DP2×FSDP64 + FP8 + QAG off` has to be added.
4. **Narrowing FSDP from 128 to 64 did not OOM** (57.09 G, only 1.79 G more than 55.30).
   But this is 16 layers, and the full 80 layers will amplify it 5×, so **do not extrapolate directly**.

> 🔁 **A methodology lesson (pointed out by Chris on the spot): when you are measuring a limit, do not downscale on your own.**
> I dropped pdbs from 12 to 8 with the justification of "keeping FP8 compile time down" —
> and in [Appendix C](#appendix-c-compilation-and-environment-engineering) I had measured myself that **compile time barely varies with scale**
> (43.5 s vs 44.3 s). **Shrinking a run based on a reason you have already disproved yourself
> turns a limit test into a shrunken test.**
> The more insidious harm: what QAG saves is communication and HBM, so **the larger the batch the higher the compute share,
> and a small batch measures QAG's value off-target**.

##### Validation experiment 3: the full-80-layer limit round (2026-08-05) — all four rounds OOMed, but the key number came out

Back to the full 80 layers starting at pdbs 12. **All four rounds OOMed**, not one ran:

| Round | Configuration | HLO temporaries required | Ceiling | Short by |
|---|---|---|---|---|
| T1 | `DP2×FSDP64` + **QAG** + pdbs 12 | **104.11 G** | 94.74 G | −9.4 G |
| T2 | `DP2×FSDP64` + FP8 without QAG + pdbs 12 | **115.19 G** | 94.74 G | −20.5 G |
| T3 | `DP2×FSDP64` + QAG + pdbs 14 | 112.43 G | 94.74 G | −17.7 G |
| T4 | `DP2×FSDP64` + QAG + pdbs 16 | 125.09 G | 94.74 G | −30.4 G |

**Although neither T1 nor T2 ran, that pair gave the hardest number in this section:**

> 🎯 **QAG saves 11.08 G at the full 80 layers (115.19 → 104.11, −9.6%).**
> At 16 layers it saved only 4.47 G — **the saving grows with layer count**, because the expert weight
> all-gather buffer of every layer is halved. This is **the only QAG gain measured at full model scale**,
> and it was read out of two failures: **the required value an OOM reports is itself a usable measurement.**

It also ruled out one path:

> ❌ **`DP2×FSDP64` is not viable at the full 80 layers with pdbs ≥ 12.**
> Even with QAG on saving 11 G, it is still 9.4 G over.
> **The static shard cost of a narrow FSDP > what QAG saves.**
>
> 🔁 The same lesson again: I had just written above with my own hand that "16 layers is only 1.79 G more,
> **the full 80 layers will amplify it 5×, do not extrapolate directly**", and then designed four rounds
> all using `FSDP64` + pdbs ≥ 12, and lost every one.
> **Writing a warning down and executing it in the design are two different things.**

⇒ Conclusion: **for this 192-expert model, "narrow FSDP in exchange for QAG" is a loss at the target scale.**
The genuinely clean path is **256 experts + FSDP128** — divisibility is satisfied naturally,
FSDP never has to be narrowed, and the static shard bill never comes due.
This is entirely consistent with [round S5](#validation-experiment-2-the-64-chip-criterion-2026-08-05-done) (256e / FSDP128 / QAG,
where HBM is actually 4.47 G lower than without QAG),
**and it points once more at the same model design conclusion: make the expert count a power of two.**

##### Validation experiment 4: turning around per the conclusion above (2026-08-05) — all OOMed again, but closer

The U rounds switched to `FSDP128 + 256 experts` (no narrowing) and lowered batch for 192:

| Round | Configuration | Required | Distance from 94.74 G |
|---|---|---|---|
| U1 | **256e / FSDP128 / QAG / pdbs 12** | **96.62 G** | **−1.88 G** ← closest |
| U2 | 256e / FSDP128 / no QAG / pdbs 12 | 101.17 G | −6.43 G |
| U3 | 192e / FSDP64 / QAG / pdbs 10 | 99.29 G | −4.55 G |
| U4 | 192e / FSDP64 / no QAG / pdbs 10 | 106.97 G | −12.2 G |

**Two things can be read off:**

1. **`FSDP128 + 256e` really does use less than `FSDP64 + 192e`** — U1 needs only 96.62 G at pdbs **12**,
   while U3 needs 99.29 G at pdbs **10**. **Two notches higher on batch and still cheaper**,
   which quantifies the value of "do not narrow FSDP".
2. **The HBM saving from QAG has now reproduced across 4 independent configurations**:

   | Configuration | No QAG | QAG on | Saved |
   |---|---|---|---|
   | 16 layers / 256e / FSDP128 / pdbs 8 | 60.04 G | 55.57 G | 4.47 G |
   | 80 layers / 256e / FSDP128 / pdbs 12 | 101.17 G | 96.62 G | 4.55 G |
   | 80 layers / 192e / FSDP64 / pdbs 10 | 106.97 G | 99.29 G | 7.68 G |
   | 80 layers / 192e / FSDP64 / pdbs 12 | 115.19 G | 104.11 G | 11.08 G |

   > ⚠️ **Do not over-read the pattern.** It looks like "narrow FSDP and large batch both make QAG save more",
   > but these four sets change layer count, expert count, FSDP width, and batch all at once,
   > **and every combination has n=1**. The only thing that can be stated is: **QAG saves HBM in every configuration tested,
   > by 4.5–11 G**. The cause remains to be determined.

##### Validation experiment 5: pdbs 11 — QAG runs on the full 80 layers for the first time

| Round | Configuration | step (s) | TFLOP/s/**chip** | Peak HBM | NaN |
|---|---|---|---|---|---|
| **V1** | **256e / FSDP128 / QAG / pdbs 11** | **19.5980** | **639.0** | **91.56 G** | **0** |
| V2 | 256e / FSDP128 / **no QAG** / pdbs 11 | — | ❌ OOM | — | — |

The HBM prediction lined up too: U1 (pdbs 12) needs 96.62 G, and the two-parameter model predicts just over 92 G one notch down,
with a measured **91.56 G**.

> 🎯 **V2's failure carries more information than V1's success.**
> Same pdbs 11, same 256e / FSDP128, and **turning QAG off OOMs outright**.
> ⇒ **QAG does not just speed things up; it turns a batch setting that could not run into one that can.**
> It also means **no same-batch pairing is available at this setting** —
> the denominator simply does not exist, and "QAG net gain x%" is undefined at pdbs 11.
> To get a pair you have to fall back to pdbs 10, where both sides run.

> ⚠️ **639.0 cannot be compared with the 618 in §5.1 or the 599 in §3 — it is the 256-expert model.**
> Going from 192 to 256 experts changes the parameter count, the FLOP convention, and HBM.
> This number answers "**what would a next-generation model look like on v7 if the expert count were set to 256**",
> not "our current model reached 639".

##### Conclusion: with 192 experts at 64 chips, QAG is not usable in practice

V3 = `192e / FSDP64 / QAG / pdbs 8` (already a very light batch) still OOMs,
**short by only 656.93 MB**:

```
Used 95.38G of 94.74G hbm. Exceeded hbm capacity by 656.93M.
    reserved   268.06M
    program     50.82G
    arguments   43.49G
```

Placing it side by side with V1 gives the most convincing comparison in this section:

| | experts | FSDP | pdbs | Peak HBM | Result |
|---|---|---|---|---|---|
| V3 | 192 | **64** | **8** | 95.38 G | ❌ over by 0.66 G |
| **V1** | **256** | **128** | **11** | **91.56 G** | ✅ ran |

> 🎯 **The larger model (256 experts) with a batch three notches higher actually uses 3.8 G less.**
> The only difference: **FSDP was not narrowed.**

And for 192 experts wanting QAG, **there is no choice of FSDP width**:

| Candidate FSDP | Divides 192? | Viable on 128 devices? |
|---|---|---|
| 128 | ❌ remainder 64 | — |
| **96** | ✅ | ❌ `128 / 96` is not an integer, DP cannot be formed |
| **64** | ✅ | ✅ but pdbs 8 already OOMs |
| 48 | ✅ | narrower, worse |

Moreover, **the thickness of an FSDP shard is determined solely by FSDP width and is independent of total chip count** —
move to 256 chips and run `DP8×FSDP64`, and the weights per device are exactly the same as here, so **it OOMs all the same**.
**Adding chips cannot rescue this.**

⇒ For 192-expert Hy3, QAG has **exactly one FSDP width available: 64**.

> 🛑 **I then wrote "QAG has no usable path on v7" — that conclusion was premature.**
> I declared it unusable having only tested down to pdbs 8; **one more notch down (pdbs 7) and it ran**. See the correction above.


##### Validation experiment 6: adding the 18 DSv3 tile settings — zero gain, one core assumption disproved

`§5.1` had long carried an assumption: *"the tile on the FP8 kernel path has never been swept once,
and capturing a BF16-magnitude gain (+17.4%) would give 726"*.
The DSv3 official recipe turned out to explicitly set **18 tile parameters**, and we set none —
which looked exactly like the gap. After mapping them onto Hy3 in proportion to the dimensions (emb 7168→4096, moe_mlp 2048→1536), the measurement was:

| Round | tile | step (s) | TFLOP/s/chip |
|---|---|---|---|
| V1 | MaxText defaults (all 1024) | 19.5980 | 639.0 |
| **W1** | **the 18 values mapped from DSv3** | **19.5980** | **639.0** |

**Identical, to four decimal places.**

First rule out "the parameters never went in" — the actual values in the two logs differ, so **the parameters did take effect**:

```
V1:  wi_tile_fwd_embed_dim: 1024   wi_tile_fwd_mlp_dim: 1024
W1:  wi_tile_fwd_embed_dim: 4096   wi_tile_fwd_mlp_dim: 1536   wo_tile_fwd_mlp_dim: 2048
```

> ⚠️ **⇒ The assumption that "FP8 just needs the tile filled in to reach 726" is not supported by current evidence.**
> The `618 × 1.174 ≈ 726` extrapolation in §5.1
> rests on "the FP8 tile matters as much as the BF16 tile" — **that step was never verified,
> and W1 is the first genuine attempt to verify it; the result is zero.**

But **it is still not possible to conclude "the FP8 tile is entirely useless"**; two possibilities have not been separated:

| Possibility | Meaning | How to decide |
|---|---|---|
| A | The parameters are accepted by MaxText but **never passed to the actual kernel** | Set an obviously bad tile; if performance **does not drop**, it was never wired up |
| B | The parameters really take effect, but **the defaults happen to already be good enough** | Setting a bad tile will make it **noticeably slower** |

⇒ A **reverse test** has to be added: deliberately set a bad set of tiles.
**This is the classic case of "a negative result is not a conclusion on its own" —
zero change can mean either "the change was right but useless" or "the change never landed".**

> 🔁 One aside: the tile is worth +17.4% on the BF16 side because tokamax's lookup table **has no row for 192**
> and it falls into a terrible default ([§3.4.2](#342-root-cause-the-kernel-library-lookup-table-has-no-row-for-192)).
> **That is patching a "broken default", not doing routine tuning.**
> FP8 goes through a different kernel (`mblx.gmm`), whose default **is not necessarily broken as well** —
> carrying "the +17.4% on BF16" over to FP8 as an expectation was an imprecise analogy to begin with.

##### Validation experiment 7: 2×2 factorial — tile is 0, but `cost_estimate_flops` is worth +0.9%

Besides tile, the DSv3 recipe has several other config options we had not set.
Making "tile" and "the other options" into a 2×2 factorial and running all four cells (units TFLOP/s/chip):

| | without `cost_estimate_flops` | with `cost_estimate_flops` |
|---|---|---|
| **without tile** | V1 **639.0** | X1 **644.8** |
| **with tile** | W1 **639.0** | W2 **644.8** |

**The readings are clean enough not to need a statistical test**: the tile effect is identically 0 (the two rows match exactly),
the `cost_estimate_flops` effect is identically +0.9% (the two columns match exactly), and there is no interaction.

Peeling it apart item by item to confirm where the +0.9% belongs:

| Round | What was added | step (s) | TFLOP/s/chip |
|---|---|---|---|
| V1 | baseline | 19.5980 | 639.0 |
| **X2** | **only `cost_estimate_flops_fwd/bwd=5e12`** | 19.4237 | **644.8** |
| X1 | ↑ + `use_max_logit_estimate=-1` + `float32_weight_sum=False` | 19.4220 | 644.8 |
| W2 | ↑ + the 18 tile settings | 19.4227 | 644.8 |

**X2 / X1 / W2 differ from one another by 0.009%, all within the magnitude of self-jitter (0.005%).**
⇒ **The +0.9% comes entirely from `cost_estimate_flops`; the other two options and the 18 tile settings are all ±0.**

**Why it works** (mechanistically self-consistent, not confirmed by trace):
it changes no computation, it just gives the splash attention Pallas kernel
**a hand-supplied FLOP cost estimate** (`tokamax_ring_attention.py:272-276`,
default `-1` = use splash's own estimate). XLA's latency-hiding scheduler uses that number to judge
"how long this kernel will run" and decides how much communication to stuff in accordingly. **Estimate accurately and communication hides well.**
This lines up with [§2.2](#22-conclusion-h2-holds-communication-is-573) —
**we are communication-bound, so anything that improves communication overlap cashes in directly.**

> 💡 **The methodological value of this round exceeds the 0.9% figure.**
> If only W2 had been run (tile + the other options together), seeing +0.9% would have been **naturally credited to tile** —
> when in truth tile is worth nothing at all. **Change one group at a time and keep blank cells in the pairing; only then can you attribute**
> ([§4.7 rule 4](#47-four-rules-for-telling-whether-a-gain-is-real-or-fake)).

##### Validation experiment 8: reverse test — the tile parameters **never reach the kernel**

X3 deliberately set all 18 tiles **to 256** (an obviously bad blocking). Three points side by side:

| tile setting | TFLOP/s/chip |
|---|---|
| MaxText default (1024) | 639.0 |
| DSv3-mapped values (4096 / 1536 / 2048…) | 639.0 |
| **deliberately bad (all 256)** | **639.0** |

**All three identical.** By the criterion in [§4.7 rule 1](#47-four-rules-for-telling-whether-a-gain-is-real-or-fake),
this is not "the default happens to be good enough", it is **the parameters never reaching the kernel at all**.

**The source gives the exact reason** (`kernels/megablox/ops.py:199-212`):

```python
# Backend Execution Routing
if use_tokamax_backend and not use_gmm_v2:
    out = _fwd_run_tokamax_v1(lhs, rhs, group_sizes, preferred_element_type,
                              transpose_rhs, use_manual_quantization)
    #     ↑ there is no tiling in the argument list at all
elif use_tokamax_backend and use_gmm_v2:
    out = _fwd_run_tokamax_v2(..., tiling, ...)      # ← passed
else:
    out = _fwd_run_megablox(..., tiling, ...)        # ← passed
```

Our configuration is `use_tokamax_gmm=True` (⇒ `use_tokamax_backend=True`)
with `use_gmm_v2` **off** ⇒ it hits the first branch ⇒ **`tiling` is discarded entirely**.

`base.yml:247-248` actually spelled this out all along; I just never read it:

```yaml
# megablox/jax ragged dot - supports forward pass only (6 configs)
# tokamax ragged dot - supports all 18 configs
```

And `_fwd_run_tokamax_v1` internally goes through `tokamax.ragged_dot` (`ops.py:286`),
**with the blocking decided by tokamax's own heuristics** —
which is exactly where the 6-line monkeypatch in [§3.4.3](#343-the-fix-a-6-line-monkeypatch) lands.

> 🛑 **⇒ The core claim in [§5.2](#52-why-the-fp8-tile-was-never-tuned-two-gmm-paths) is backwards.**
> I wrote that "**turning on FP8 means switching kernel paths**, the BF16 monkeypatch executes not a single line under FP8,
> and 618 was produced by 'FP8 + MaxText default tile'".
> **The truth is the opposite**: FP8 still goes through tokamax and **the monkeypatch takes effect as usual**;
> what gets discarded is MaxText's `w{i,o}_tile_*`.
>
> **This means the premise that "the FP8 tile has never been swept once" was never valid** —
> every one of our FP8 rounds injected `tkcfg.py`,
> and **FP8 has been consuming the optimal tile `(512, 2048, 1536)` found on BF16 all along**.
> It also explains why that 618 in §5.1 is not actually bad — it is not an "untuned" starting point.

⇒ **To make MaxText's tile configuration actually take effect you must turn on `use_gmm_v2=True`** (going through `_fwd_run_tokamax_v2`).
But [Appendix B.3](#b3-clear-negative-results) records that 70% of gmm_v2's gain is eaten by copies XLA inserts —
that path has to be re-evaluated; it is not free.

##### Validation experiment 9: the monkeypatch is the **only** source of tile on the FP8 path

| Round | monkeypatch | Result |
|---|---|---|
| **Y0** | `512,2048,1536` (the BF16 optimum) | **finished in 322 s, 645.0** — exactly reproducing X1/X2/W2's 644.8 (0.03% apart) |
| **Y1** | **no injection at all** | **timed out at 2748 s, 8 steps unfinished** |

**At least 8.5× slower, and in reality more** (Y1 had not reached step 4 by the timeout).
The symptom is the same as the first entry in [Appendix B.5](#b5-conclusions-that-were-overturned-written-down-by-me-then-disproved):
*"it is not a deadlock, it is slow enough to trip the watchdog"*.

**The source gives an explanation stronger than the experiment** (`tokamax/_src/ops/ragged_dot/pallas_mosaic_tpu.py:320-322`):

```python
@override
def _get_heuristics_config(self, ba) -> Config:
    if self.qdtype is not None:
        return Config()        # ← when quantized, return an empty config directly; not a single tile is computed
    if pltpu.get_tpu_info().generation < 7:
        return Config()
    ...                        # ← the "start from the largest and deflate until it fits in VMEM" logic below
                               #    only runs on the non-quantized path
```

> 🎯 **Under FP8, the framework's automatic blocking logic is short-circuited entirely.**
> It is not "it computed a value that was not good enough", it is **it does not compute at all** — it returns an empty `Config()`
> and leaves the rest to Pallas/Mosaic's conservative defaults.
>
> Our 6-line monkeypatch overrides `tile_m/k/n` with `dataclasses.replace` after `_orig()` returns,
> so **on FP8 it is not an "optimization" but the sole provider of tiles**.
> Take it away = no tile decision at all = Y1's factor of 8.5.

⇒ This also corrects the wording in [Appendix B.5](#b5-conclusions-that-were-overturned-written-down-by-me-then-disproved)
about "the tokamax lookup table has no row for 192":
**BF16 is "a table miss falling into a bad default", FP8 is "there is no table lookup step at all"** —
the symptoms are similar (both extremely slow) but the mechanisms differ.

##### Next step: an FP8-specific tile sweep (with a theoretical basis, not blind)

That heuristics passage also points to **a direction for optimization**: its strategy is
"start at `tile_m = min(m,1024)`, `tile_n = n`, `tile_k = k`,
**and halve step by step until it fits in VMEM**", and `_fit_within_tpu_vmem`
computes capacity **from the actual byte sizes of `lhs.dtype` / `rhs.dtype`**.

> 💡 **FP8 weights occupy half the bytes of bf16 ⇒ the same VMEM can hold a tile about twice as large.**
> The `(512, 2048, 1536)` we use now was **deflated under bf16's memory constraint**,
> so carrying it straight over to FP8 leaves half the VMEM sitting empty.

On that basis, three targeted points (baseline Y0 = 645.0):

| Round | tile | Hypothesis |
|---|---|---|
| Z1 | `512, 4096, 1536` | double tile_k — the reduction dimension benefits first from the freed VMEM |
| Z2 | `1024, 2048, 1536` | double tile_m |
| Z3 | `1024, 4096, 1536` | double both |

⚠️ The Z3 combination was once rejected outright by the kernel under **BF16** ([Appendix B.2](#b2-crashes--configuration-rejections),
`tile_m ≥ 1024 and tile_k ≥ 4096`).
**But that was under bf16 memory pressure**, and FP8 uses half as much, so it is worth retrying —
which is also another test point for "gains and limits cannot be moved across dtypes".

##### Validation experiment 10: FP8 tile sweep results — all three points fail, `(512,2048,1536)` is a local optimum

| Round | tile | Result |
|---|---|---|
| Y0 | `512, 2048, 1536` (baseline) | **645.0** |
| Z1 | `512, 4096, 1536` | ❌ **VMEM OOM** (needs 352 MiB) |
| Z2 | `1024, 2048, 1536` | **628.8** (**−2.5%**) |
| Z3 | `1024, 4096, 1536` | ❌ **VMEM OOM** (needs 352 MiB) |

> 🛑 **My inference that "FP8 can hold a tile twice as large" has a logical hole.**
> **Under FP8 only `rhs` (the weights) halves in bytes; `lhs` (the activations) is still bf16.**
> And doubling `tile_k` doubles **both the lhs term and the rhs term** —
> a 50% discount on unit price ≠ being able to double the size.
>
> The correct inference should be: **the same set of tiles uses less HBM under FP8 than under BF16**
> (so a configuration that OOMs on BF16 may run on FP8),
> but **that does not mean the tile size can be pushed up**.
> "Cheaper unit price" and "larger size" are two different things, and I conflated them into one.

**Z2's negative result is actually the most valuable cell**:
under BF16, raising `tile_m` from 512 to 1024 is **−3.8%** ([Appendix A.1](#a1-256-chips--512-devices--full-80-layers-2026-08-04)),
and on FP8 it is **−2.5%**. **Same direction, similar magnitude.**

> 🎯 **⇒ The tile optimum is stable across dtypes.**
> `(512, 2048, 1536)` was swept out on BF16 and is equally a local optimum on FP8 —
> all three directions (bigger k / bigger m / both) either blow VMEM or get slower.
> **This path is exhausted; no need to sweep further.**
>
> One incidental correction to a rule: in §4.7 I wrote "the magnitude of a gain cannot be moved across kernels",
> and this data shows **the optimum (argmax) can be moved; what cannot be moved is the magnitude of the gain (max value)**.
> They are different things.

##### Validation experiment 11: XLA flags — our 9 vs DSv3's 36, and still ±0 after filling them in

The DSv3 recipe has **36 XLA flags, and we only have 9**; the bulk of what is missing is an entire set of
**SparseCore Collective Offloading**. And the official Ironwood tuning document says explicitly:
*"the primary mechanism for overlapping communication with compute on TPU7x is called SparseCore Collective Offloading,
and it is the recommended approach for asynchronous collectives on TPU7x"*.

Our bottleneck is communication at 57.3%, yet we had skipped the officially designated communication optimization mechanism — it looked like the biggest single piece.
**Four cells measured, all ±0 or worse:**

| Round | Configuration | Result |
|---|---|---|
| Baseline | our existing 9 flags | 645.0 |
| A1 | + 11 SparseCore offloading flags | **644.8** (±0) |
| A2 | DSv3's full 36 | ❌ **HBM OOM** |
| B1 | The official `ENABLE_SPARSECORE_OFFLOADING_FOR_RS_AG_AR` group (CF off + SC on + 2 base flags) | **644.8** (±0) |
| B2 | **CF off only**, nothing else changed | **644.8** (±0) |

**A1's ±0 was expected** — the comment in MaxText's official flag library (`benchmarks/xla_flags_library.py`):

```python
# On Ironwood, by default:
# xla_tpu_enable_sparse_core_collective_offload_all_gather as True
# xla_tpu_enable_sparse_core_collective_offload_reduce_scatter as True
# xla_tpu_enable_sparse_core_collective_offload_all_reduce as True
```

**These three are True by default on Ironwood**, so setting them explicitly again is the same as not setting them.

The same file contains an even more important line:

```python
# Either one of CF or SC can be enabled at a time.
```

**`async collective fusion` (CF) and SparseCore offloading (SC) are mutually exclusive.**
I inferred from this that "SC is on by default but is being blocked by CF" — B1/B2 were run to verify that,
and **it is still ±0 after turning CF off**.

> 🛑 **The inference is mechanistically sound, but the gain is zero.**
> The most reasonable explanation: among our existing 9 flags,
> `xla_tpu_enable_sparse_core_collective_aggregator=true` and
> `xla_tpu_enable_latency_hiding_layer_scheduler=true` **have already taken this gain**;
> and the largest remaining piece — the MoE weight all-gather — **has already been halved by QAG**.
> **The part SparseCore offloading could help with, we have already obtained by other means.**

A2's OOM is information in itself: DSv3's set contains items that raise HBM usage
(`scoped_vmem_limit_kib=65536` is higher than our 65472, or `accumulate_into_mrb`).
⇒ **Do not copy someone else's flags as a group — they were tuned against another model's memory budget.**


##### Validation experiment 12: the `pdbs 11 → 12` path is blocked

`pdbs 12` needs 96.62 G, only **1.88 G** short, which looks very close. Both directions were tried:

**① Free it up via remat — there is no room.**
Under `remat_policy=custom` every tensor gets one of three choices: `device` (keep in HBM) / `remat` (discard and recompute) / `offload` (move to host).
Checking `configs/base.yml:356-374`, **every tensor except `decoder_layer_input` already defaults to `remat`** —
they take no HBM to begin with and are already at the most economical setting;
and `decoder_layer_input` cannot be remat'd (it is the starting point of recomputation), so we have already set it to `offload`.
⇒ **The remat dimension is exhausted.**

**② Take an intermediate setting with a fractional batch — not supported.**

| Round | Configuration | Result |
|---|---|---|
| C1 | `per_device_batch_size=11.5` | ❌ `AssertionError: Batch dimension should be shardable among the devices in data and fsdp axis` |
| C2 | `per_device_batch_size=11.8` | ❌ `ValidationError` |

**The batch dimension must be divisible by the `data × fsdp` device count**, so pdbs can only be an integer.
(The `per_device_batch_size=8.0` in the DSv3 recipe is just floating-point notation; the value is still an integer —
I mistakenly took it to mean fractional settings were supported. **Seeing a float literal does not mean continuous values are supported.**)

> 🎯 **⇒ `pdbs 11` is the batch ceiling for this "64 chips / 256 experts / QAG" configuration,
> and 645.0 is the corresponding performance ceiling.**
> It is not "a little bit short", it is that **there is no setting available between 11 and 12 at all**.


</details>

##### Re-ranking the ways out (the basis has moved from speculation to source code + the official recipe)

| # | Idea | Current assessment |
|---|---|---|
| 1 | **Take FSDP as 64 / 96 to divide 192** | ✅ Mechanistically confirmed feasible (the constraint is divisibility, not powers of two). **The cost is HBM**, S3 is measuring it |
| 2 | **Change the model to 256 experts** | ✅ Cleanest, and isomorphic to the DSv3 official recipe. A next-generation model design decision; S5 is measuring it |
| 3 | Copy Ant's local patch (absmax + QAG) | ⚠️ Official analysis holds that QAG may be inefficient under dynamic scale, and it has not gone upstream |
| 4 | Set `ici_expert_parallelism` to make divisibility work | ❌ Ruled out. The config explicitly requires `EP=1 and TP=1`, and EP measures **−71%** on TPU |
| 5 | Wait for upstream to support non-divisible cases | ❌ Not in our hands |

⚠️ **The general risk is unchanged**: `fixed` is a preset scale, and `±224` is the official choice for DSv3,
**which is not necessarily appropriate for Hy3's weight distribution**.
**An 8-step benchmark can only show whether there are NaNs and cannot prove convergence — touching this requires convergence validation.**
Ant abandoned fixed for absmax + a patch precisely because of accuracy; keep that precedent in mind.

---


**Conclusion: FP8 is currently in the state of "it runs, it gives +3.2–5.9%, and not a single cell has been tuned".
By value for money, the next step should be ① sweep FP8's `w{i,o}_tile_*` → ② check whether QAG can be enabled, not autotune.**

---

## 6. Not yet tried

Ordered by expected gain:

| # | Item | Why it is worth it | Status |
|---|---|---|---|
| 1 | **Run the official autotune to generate cache entries** | Replaces the monkeypatch, and may beat hand-tuned tile | not done |
| 2 | **The official 28-flag XLA group** | Measured 0% on a single node, but that group is all cross-machine communication flags, which a single node cannot measure in the first place | to be decided on 256 chips |
| 3 | **FP8** (`fp8_full` + qwix) | Compiles and runs on v7 (compilation fails on v5p), −1.2% on a single node | to be decided at multi-chip scale |
| 4 | `gmm_v2` + `tile_k` dividing K | Upstream measured +13.58% end-to-end on v7 after tuning tile_k | not done |
| 5 | ring of experts / `num_moe_emb_chunks` | Communication is 57.3% on v7, which is this feature's target scenario | not done |
| 6 | Fine sweep of the tile neighborhood (256 / 768 / 3072) | The current optimum came from a coarse sweep | in progress |

---

## Appendix A: All ablation data

### A.1 256 chips / 512 devices / full 80 layers (2026-08-04)

**Group A: parallelism splits** (pdbs 8, megablox)

| Split | step | dev | chip | MFU | Peak HBM |
|---|---|---|---|---|---|
| `DP4×FSDP128` | 20.0915 s | 226.5 | 453 | 19.64% | 74.20 G |
| `DP2×FSDP256` | 20.2280 s | 225.0 | 450 | 19.51% | 61.36 G |
| `DP1×FSDP512` | 22.5470 s | 201.8 | 404 | 17.49% | — |
| `DP8×FSDP64` | OOM | | | | |
| `DP16×FSDP32` | OOM | | | | |

**Group B: batch / sequence** (base `DP4×FSDP128`, megablox)

| Configuration | step | dev | chip | MFU | Peak HBM |
|---|---|---|---|---|---|
| pdbs 8 | 20.0915 s | 226.5 | 453 | 19.64% | 74.20 G |
| pdbs 12 | 27.8648 s | 245.0 | 490 | 21.24% | 91.93 G |
| pdbs 16 | OOM | | | | |
| seq 8192 / pdbs 4 | 22.5383 s | 225.6 | 451 | 19.56% | 74.33 G |

**Group C: tokamax tile** (base `DP4×FSDP128`, pdbs 8)

| tile | step | dev | chip | MFU | Peak HBM |
|---|---|---|---|---|---|
| (512, 2048, 1536) | 17.1190 s | 265.8 | **532** | 23.04% | 75.33 G |
| (1024, 2048, 1536) | 17.7907 s | 255.8 | 512 | 22.18% | 75.34 G |
| (512, 1024, 1536) | 18.2422 s | 249.4 | 499 | 21.62% | 75.33 G |

**Group F: tile × batch combinations** (all with `tile(512,2048,1536)` unless noted)

| Configuration | step | dev | chip | MFU | Peak HBM |
|---|---|---|---|---|---|
| `DP4×FSDP128` + pdbs 10 | 20.1713 s | 282.0 | 564 | 24.45% | 84.06 G |
| `DP4×FSDP128` + pdbs 12 | 23.5553 s | 289.8 | **580** | 25.12% | 91.94 G |
| `DP2×FSDP256` + pdbs 12 | 23.9947 s | 284.5 | 569 | 24.66% | 78.27 G |
| `DP2×FSDP256` + pdbs 14 | 27.2155 s | 292.6 | 585 | 25.37% | 89.56 G |
| **`DP2×FSDP256` + pdbs 16** | 30.4002 s | 299.4 | **599** | **25.96%** | 92.33 G |
| `DP4×FSDP128` + pdbs 12 + **tile_m 1024** | 24.0598 s | 283.7 | 567 | 24.59% | 91.95 G |

### A.2 64 chips / 128 devices / full 80 layers (2026-08-04, `DP1×FSDP128`)

| Configuration | step | dev | chip | MFU | Peak HBM | Same config at 256 chips |
|---|---|---|---|---|---|---|
| megablox / pdbs 8 | 19.9188 s | 228.4 | 457 | 19.80% | 74.20 G | 453 |
| tile / pdbs 8 | 16.7545 s | 271.6 | 543 | 23.55% | 75.33 G | 532 |
| tile / pdbs 10 | 20.2780 s | 280.5 | 561 | 24.32% | **84.06 G** | 564 (**84.06 G**) |
| **tile / pdbs 12** | 23.5385 s | 290.0 | **580** | **25.14%** | **91.94 G** | **580** (**91.94 G**) |

### A.3 64 chips / 80 layers (2026-08-01, an early batch)

Baseline `D0` = 17.4349 s, 228.1 TFLOP/s/device. **This batch ran on different machines and is 15% faster than the 20.43 s of 07-30
— comparing absolute values across batches is meaningless; ablations have to be compared within the same batch.**

| # | Change | Result | Δ |
|---|---|---|---|
| D0 | baseline | 17.4349 s | — |
| D2 | `remat_policy=full` + `decoder_layer_input=remat` | 17.5633 s | **−0.74%** |
| D4 | delete 8 SparseCore flags (keep the aggregator) | 17.4355 s | −0.00% |
| D1 | `shard_exp_on_fsdp=True` | **crash** `IndivisibleError` | — |

### A.4 16 chips / 20 layers (fast screening; the magnitudes cannot be extrapolated directly)

Baseline `B0` = 5.3336 s, 201.9 TFLOP/s/device.

| # | Change | Result | Δ |
|---|---|---|---|
| C6 | `shard_exp_on_fsdp` + `remat=full` | 5.1862 s | +2.76% ⚠️ crashes at 64 chips |
| A5 | `shard_exp_on_fsdp=True` | 5.2545 s | +1.48% ⚠️ crashes at 64 chips |
| A10 | `remat_policy=full` | 5.2683 s | +1.22% ⚠️ −0.74% at 64 chips |
| C5 | delete 8 SparseCore flags | 5.3339 s | −0.01% |
| G2 | `fp8_full` + qwix + all 6 mlp tiles set to 1536 | 4.8937 s | **+8.25%** |

### A.5 Early evolution (64 chips, 2026-07-30)

| Round | Increment | seq | pdbs | step | chip | MFU |
|---|---|---|---|---|---|---|
| V1 | baseline: 2 XLA flags | 8192 | 4 | 25.11 s | 405 | 17.54% |
| y1 | + `use_tokamax_splash` + `sa_use_fused_bwd_kernel` | 8192 | 4 | 24.45 s | 415 | 18.00% |
| y4 | + scheduler group (4 flags) | 8192 | 4 | 23.08 s | 440 | 19.09% |
| **c1** | scheduler group × pdbs 8 / seq 4096 | 4096 | 8 | **20.43 s** | **445** | **19.29%** |
| c2 | c1 + miscellaneous group (filling out to 26 flags) | 4096 | 8 | 20.45 s | 445 | 19.27% |

Ordered by contribution: pdbs 8 / seq 4096 **+12.8%** ｜ scheduler group **+6.6%** ｜
`use_tokamax_splash` + `sa_use_fused_bwd_kernel` +2.6% ｜ miscellaneous group ±0 ｜ SparseCore group ±0 ｜
optimizer/HBM group −0.5% (what it saves is HBM, not time).

---

## Appendix B: The complete collection of negative results

<details>
<summary><b>All OOMs / crashes / zero gains / overturned conclusions (click to open)</b></summary>

### B.1 HBM (OOM)

| Configuration | HBM | Notes |
|---|---|---|
| `DP8×FSDP64` / `DP16×FSDP32` (512 dev) | — | Halving FSDP → doubling the static shard per device. The static part at FSDP=64 is about 51 G |
| `pdbs=16` (FSDP128, 512 dev) | predicted 109.6 G | The model prediction matches the measurement |
| `shard_exp_on_fsdp=True` (FSDP=64×DP=2) | 109.14 G | **14 G more than not turning it on** |
| `per_device_batch_size=12` (old configuration, FSDP not thinned) | 95.17 G | Later ran once FSDP thinned the shards |
| `ici_expert_parallelism=4` (16 chips) | 137.60 G | |
| `ici_expert_parallelism=8` (16 chips) | 192.70 G | |
| EP4 + ring + `num_moe_token_chunks=4` | 111.24 G | |
| `scan_layers=False`, pdbs 8 | **171.75 G** | Dropping pdbs to 1/4 lowers HBM by only 6% — what blows up is not activations |
| `scan(unroll=10)` | **274.64 G** | Higher than fully unrolling |
| tokamax `tile_k=4096` | OOM | |

**Why `shard_exp_on_fsdp` is a net loss**: the trade moves both ends — expert weights switch to being cut along the expert dimension (the gain),
but the FSDP width drops from 128 to 64, so **the non-expert part (80 layers of attention + embedding + the first dense layer, about 7.2 B)
doubles its per-device shard**. What is saved does not cover what is spent. The root cause is still that **192 is not a power of two**.

### B.2 Crashes / configuration rejections

| Configuration | Error |
|---|---|
| `shard_exp_on_fsdp=True` (128 devices) | 192 experts do not divide 128 devices. The interception point is the explicit check `num_experts % ici_fsdp_parallelism != 0` at `pyconfig_deprecated.py:1212`, **not the kernel layer** ⇒ changing the FSDP width works around it, see [§5.4.2](#542-qag-quantize-first-then-communicate-a-path-blocked-by-the-expert-count) |
| `weight_quantization_calibration_method=fixed` (bare) | `ValueError: A fixed range is required for fixed calibration.` **`fixed` is only the method name; the range goes into the same string**: `fixed,-224,224` (the official canonical value) |
| `shard_exp_on_fsdp=True` + `ici_expert_parallelism>1` or `ici_tensor_parallelism>1` | Rejected at `pyconfig_deprecated.py:1214`: requires `EP=1 and TP=1`. ⇒ **You cannot use EP to make divisibility work** |
| `quantization=fp8` | `AttributeError: Fp8Quantization has no quant_dg` — that is an **NVIDIA-specific class**; the correct route on TPU is `fp8_full` + qwix |
| Deleting **all** 9 SparseCore flags | `the layer scheduler requires sparse core collective aggregator to be enabled` |
| `use_gmm_v2=True` on its own | Configuration rejected, requires `use_tokamax_gmm=true` |
| `num_moe_emb_chunks=4` | Configuration rejected, requires `use_gmm_v2` + `use_ring_of_experts` |
| `fp8_full` + qwix, default tile 1024 | `AssertionError: v=1536 bv=1024 s=1536` |
| **`tile_m` ≥ 1024 and `tile_k` ≥ 4096** (e.g. `(1024,4096,1536)`, `(2048,2048,1536)`) | `MosaicTpuRaggedDot(config=None, vjp=...)` — the Pallas kernel rejects the combination outright; not an OOM |
| **FP8 + `tile(512,2048,1536)` + pdbs 12** (64 chips) | Same `MosaicTpuRaggedDot` rejection. **pdbs 10 with the same configuration is fine** ⇒ on the FP8 path the combination of tile and batch carries extra constraints |
| `scan(unroll=2)` | XLA `Expected instruction to have shape equal to (bf16[9,2,8,4096,4096], ...)` — `2` is the unroll factor, `9` is the splash attention blocking, and the downstream kernel has not kept up |

### B.3 Clear negative results

| Configuration | Δ | Notes |
|---|---|---|
| `ici_expert_parallelism=4` (half batch, 16 chips) | **−71.36%** | EP is not "it does not fit", it is **inherently slow**. AllToAll is multi-hop on a TPU torus |
| EP4 + ring + `token_chunks=4` | −36.96% | Chunking **claws back 34 percentage points**, but does not fill EP's hole |
| `use_2d_fsdp_sharding` + `fsdp_transpose=4` + `two_stage_all_gather` | **−11.73%** | Do not try again |
| `DP1×FSDP512` (512 devices) | −11% | The cost of strong scaling |
| tokamax `tile(128, 4096, 1536)` | −9.0% | |
| `int8` + qwix (16 chips) | −5.81% | |
| `fp8_full` + qwix + tile 512 | −3.90% | 512 divides 1536, but cutting three ways is slower instead |
| pure bf16 + tile 512 | −3.25% | Proves it is the tile's fault and unrelated to quantization |
| `tile_m=1024` @ pdbs 12 | −2.2% | Refutes the in-table rule that "`tile_m` follows `m`" |
| `fp8_e4m3` + qwix (default tile) | −1.87% | |
| FP8 (`fp8_full` + qwix, single node) | −1.2% | **But "it runs" is itself new information** — on v5p it fails to compile |
| `remat_policy=full` (64 chips) | −0.74% | +1.22% at 16 chips, **a sign flip** |
| optimizer / HBM group | −0.5% | What it saves is HBM, not time |
| `scan_layers=False` (5 layers) | −5.9% | And the large model OOMs outright |

### B.4 Zero gain (ran it, confirmed useless)

| Configuration | Δ |
|---|---|
| SparseCore offloading group, 9 flags (v7) | **±0** (+4.07 pp on v5p) |
| Deleting 8 of them (keeping the aggregator) | −0.00%, consistent at 16 chips and 64 chips |
| Miscellaneous flag group (5 flags, filling out to 26) | ±0 |
| seq 8192 / pdbs 4 vs seq 4096 / pdbs 8 | 451 vs 453, a tie |

### B.5 Conclusions that were overturned (written down by me, then disproved)

| The former conclusion | The truth |
|---|---|
| "`use_tokamax_gmm` deadlocks, `stalled chips [7]`" | **Not a deadlock, just slow enough to trip the watchdog.** The root cause is the LUT miss → grid blocks growing 768× |
| "192 experts is not a power of two, which breaks the group partitioning" | Half right in direction: it really is about 192, but **upstream only tuned tiles for 16/128/256**, which is a **data coverage problem, not an algorithm problem** — the two have completely different fixes |
| "Changing `num_experts` to 256 will hit the LUT" | **Measured ineffective**, all 48 `Autotuning cache miss` events still there. There is also a JSON autotuning cache whose key is the full operator signature |
| "Compilation is faster at small scale, 49.6 s vs 10–17 minutes" | **Dividing two numbers from different conventions.** Measured, compilation barely varies with scale (43.5 s vs 44.3 s); what really grows with scale is multi-host slice creation (0.8 s vs 60.2 s) |
| "step 0 = 49.6 s is compile time" | That was the time to upload HLO to GCS with `dump_hlo=True` |
| "Use `scan(unroll=N)` to let XLA hide communication across layers" | 2 hits a kernel shape check and 10 needs 274 G, **there is no usable setting**. And 80% of the communication is synchronous collectives, which were never caused by scheduling boundaries in the first place |
| "Communication/compute overlap is 0.000 s ⇒ completely exposed" | **A tautology** — the intersection on a single sequential lane is identically zero |
| "pdbs=12 needs 98.5 G and will OOM" | Measured 91.93 G and it ran. A single-point linear extrapolation error; activations are sublinear |
| "Small scale cannot pick winners" | Too strict. Classify by "does it alter the shard shape"; tile / pdbs are fully transferable |
| "MoE takes the largest share, so push on MoE" | megablox is already a tuned optimal path. **Look for the place nobody has tuned** |
| "Reproduce the tokamax problem with a 4-chip smoke test first" | All 8 hypotheses ran and it could not be reproduced. The root cause was found by **going straight into the kernel library source and printing the table** |
| "The official DSV3 743.5 ≈ the 746 I computed ⇒ even DSV3 does not have QAG on" | **All four QAG conditions are present in the official recipe; DSV3 has it on.** When numbers from two independent systems collide, with only one observation point that says nothing about causality |
| "The four QAG trigger conditions are parallel" | 3 and 4 are the same lock — `explicitly_weight_ag` checks `fixed` again internally. Turning on `shard_exp_on_fsdp` alone **fails silently** (measured 1.2228 → 1.2230, no change and no error) |
| "The expert count must be a power of two to enable QAG" | The constraint is `num_experts % ici_fsdp_parallelism == 0`. It is a **divisibility relation**, not a power of two — it only looks that way because we have always used powers of two as the FSDP width |
| "Ant's ALModel enabled QAG via this upstream `fixed` path" | Backwards. **They explicitly abandoned `fixed,-224,224` for absmax** (accuracy considerations) and enabled it via a local patch that deletes the `startswith("fixed")` criterion |

### B.6 Operational screw-ups (not conclusions, but they waste time)

- **Trim XLA flags as a group**: deleting the `collective_aggregator` the scheduler depends on kills two rounds instantly
- **`gcloud ... | tail -3` swallows the real exit code** (the pipeline returns tail's 0), so a 400 error is taken as success.
  **Judge success or failure by the output content, not just `$?`**
- **When cross-project GCS is unreadable**, just `kubectl cp` a 12 MB-scale file into the pod; do not go touching the shared bucket's IAM
- **`pkill -f 'pre_train.train'` kills itself** — `pkill -f` matches the whole command line, and that line of text contains this very
  pattern string. The correct form is `'pre_train[.]train'`
- **Variables inside a bash function are global by default** — in my campaign script the `run()` function used `TK=$4` to receive tile_k,
  which collided with the global `TK='megablox=True ...'`; the first round polluted it into `2048`,
  and the following four rounds passed the bare argument `2048` to MaxText, all giving `ValueError`
- **Write the `pkill -f` pattern against the real command line.** My campaign script used
  `pkill -9 -f "train[.]py"` to clean up the previous round, but the actual command line is
  `runpy.run_module('src.maxtext.trainers.pre_train.train')` — **which does not contain the string `train.py`**,
  so this pkill **never matched once**. Rounds switched over cleanly because the previous round exited on its own,
  not because of it. **After writing a pkill, verify with `pgrep -f` on the same pattern that it can match the target.**
- **Look at the sequence before fixing the steady-state window**: with the profiler on, step 17 has a ~90-second spike (exporting the trace),
  and taking the mean over `step ≥ 15` by habit gives 11.678 s, nearly twice the real steady state
- **Do not downscale on your own when measuring a limit.** To "keep FP8 compile time down" I dropped pdbs from 12 to 8 and
  layers from 80 to 16 — when Appendix C contains my own measurement that **compile time barely varies with scale**.
  Shrinking a run based on a reason you have already disproved turns a limit test into a shrunken test,
  and the 639.8 obtained cannot be compared with the historical best, so the whole round has to be rerun.
  **The more insidious part is that it skews the conclusion**: what QAG saves is communication and HBM, and the smaller the batch the lower the compute share,
  so a small batch systematically over- or under-estimates the value of communication-side optimizations
- **A paired experiment must "differ in exactly one variable", and that has to be counted at design time.** In round S3 I changed both
  "turn on QAG" and "FSDP 128→64"; two variables mixed together, and the +12.1% cannot be attributed to either.
  Ironically it was S4/S5, added on the side (256 experts, same parallelism and same batch, differing only in the QAG switch),
  that gave the one clean net gain of +15.6%. **Next time, write out the diff for each round before laying out the experiment matrix**

</details>

---

## Appendix C: Compilation and environment engineering

<details>
<summary><b>Compilation mechanics, caching, scan, and a fast-iteration environment (click to open)</b></summary>

### C.1 Three conventions for compile time; do not mix them

| Convention | Where to read it |
|---|---|
| **XLA self-reported (recommended)** | `deepsea_compiler_base.cc:989] END_TO_END stage duration: 43.69s` |
| Per-jit on the JAX side | `JAX_LOG_COMPILES=1` → `Finished XLA compilation of jit(train_step) in 45.15 sec` |
| Wall clock | `TRIAL_START` → `completed step: 0`, including cache IO and the data pipeline |

Measured in the same round: 42.76 / 45.15 / 52.2 s. All three are correct. **Always state the convention when reporting a number.**

> **`step 0` is not compile time.** MaxText finishes compiling before entering the step loop; `step 0` is just an ordinary training step.

### C.2 Compile time does not vary with layer count (because there are two scans)

| Layers | XLA `END_TO_END` |
|---|---|
| 5 | 42.41 s |
| 20 | 43.12 ~ 44.77 s (five runs) |
| 80 | 44.29 s |

**A 16× change in layer count moves compilation by only 4.5%.** Mechanically this is because there are **two layer bodies** — the first layer is dense
(`first_num_dense_layers: 1`), structurally different from the MoE layers and unable to go into the same scan:

```
[SCANDBG2] main lax.scan length=1  unroll=1   ← the dense segment
[SCANDBG2] main lax.scan length=19 unroll=1   ← the MoE segment (at 20 layers)
```

**Layer count only changes the scan trip count; it does not enter the HLO size.**

> **So "more layers therefore longer compilation" is inherently false in scan mode.** Whenever you hear it, go check the log timestamps.
> I accepted it too readily at the time, recorded a stall as "slow compilation", and burned an extra day.

> **Debugging tip**: this model takes the **NNX** path (`nnx_decoders.py`), not Linen's `decoders.py`.
> There are three `lax.scan` sites in the whole tree, so when changing scan behavior, **add a print first to confirm which one you reach**; this project misapplied a patch twice for this reason.

### C.3 Compilation cache: 45 s → 0.87 s

**Two prerequisites, both mandatory:**

1. **`dump_hlo=False`** — MaxText **disables** the JAX compilation cache when `dump_hlo=True`
2. **A resident pod** — the cache directory lives inside the container, and a one-shot Job takes it with it when it finishes

| | Cold | Warm |
|---|---|---|
| Number of jit compilations | 21 | **21 (the same)** |
| Total compile time | 51.75 s | **3.38 s** |
| Largest single one | 45.15 s | **0.87 s** |
| Startup → step 0 | 83.8 s | **32.8 s (−61%)** |

**The accurate statement is not "the second round skips compilation" but "the number of compilations is identical, and the 45-second one collapses to 0.87 seconds".**
Only modules above the threshold go through the persistent cache; the other 19 small ones are compiled fresh every time, for about 2.5 s in total.

**What invalidates the cache** (the cache key is the compiled HLO):

| Change | Invalidates? |
|---|---|
| XLA flags | **Yes** — every round of a flag sweep is a cold start |
| `steps` | **Yes** — `learning_rate_schedule_steps` inherits it by default and gets compiled into the HLO as a constant. **Do not casually change the step count while sweeping parameters** (measured: 4→8 moved startup from 32.8 s to 80.2 s) |
| Layers / batch / seq | Yes |
| `dump_hlo=True` | **Disables the cache outright** |

### C.4 Repeatability: a single round can decide a 1%-level improvement

Five steady-state steps in the same environment with the same configuration: 6.0900 / 6.0851 / 6.0852 / 6.0847 / 6.0860 s.

**Range 5.3 milliseconds (0.09%), with ±2–4 ms of jitter within a round. No need to average over multiple runs.**
It also shows that `profiler` and `dump_hlo` have no effect on steady-state throughput, only on the startup phase.

### C.5 The environment shape for fast iteration

**Replace the "one-shot Job" with a "resident environment"** and you can run a dozen-plus rounds in one night:

- N pods running `sleep infinity` to hold the TPU slice, with the code pre-unpacked in the container
- Each round uses `kubectl exec` to start the same command **in parallel** on all pods
- The compilation cache lands at a fixed path inside the container and is retained across rounds

**The key restriction: on a multi-host TPU, all pods must execute simultaneously.** Exec into just one pod and run JAX and it hangs building the mesh:

```
RuntimeError: Unable to initialize backend 'tpu': DEADLINE_EXCEEDED:
TPU initialization failed: Failed to connect to <peer>:8471
```

> ⚠️ Worse still, that failed process will **hold on to `/dev/vfio/*`**, after which every training run reports
> `Device or resource busy; Couldn't open iommu group`, and the only fix is to rebuild the pod.
> **Reading code can be done from a single pod; actually running requires lock-step execution.**

Gains: a one-shot Job has to be rescheduled every round, pull a 1.73 GB image, build the slice, and compile cold;
in a resident environment, one round of a 30-step experiment goes **275 s → 209 s**, and startup to first step goes **83.8 s → 32.8 s**.
On 256 chips, one round of an 8-step experiment takes about **6 minutes**.

### C.6 Building your own v7 node pool: four blockers

1. **You must use a workload policy, not a placement policy.** The `--placement-type=COMPACT` habitual on v5p
   and a bare `--tpu-topology` are both rejected on v7
2. **`--accelerator-topology` must be set on the workload policy.** Omitting it is what produces
   `does not support TPU topology with group placement policy and workload policy at the same time`.
   When creating the pool, pass **both** `--tpu-topology` and `--placement-policy` with matching topologies
   (this is exactly what the upstream tpu-recipes ironwood recipe does).
   ⚠️ This entry previously read "do not pass `--tpu-topology` again"; **that was wrong and was corrected on 2026-08-07**.
   The topology is carried by the workload policy's `--accelerator-topology`
3. **You must explicitly pass `--scopes=cloud-platform`** — the default storage scope is only `devstorage.read_only`,
   which shows up as **downloading code works fine but writing output gives 403**. **A node pool's scope cannot be modified; it can only be deleted and rebuilt**.
   Granting IAM on the bucket does not help either; IAM and OAuth scope are two separate layers
4. **DWS flex-start cannot idle waiting for you** — the API layer forces flex-start to enable autoscaling,
   and autoscaling means 0 nodes when there is no workload. To keep it resident you have to park a `sleep infinity` in it.
   Also, flex-start nodes last **7 days at most** and do not support reservations or Spot

### C.7 Three quantitative conclusions about scale

**16 chips / 20 layers vs 64 chips / 80 layers** (proportional scaling, with a constant "layer share" of 1.25 per chip):

| | 64 chips / 80 layers | 16 chips / 20 layers | Difference |
|---|---|---|---|
| TFLOP/s/chip | 445.1 | 410.7 | **−7.7%** |
| First log line → step 0 | 174.2 s | 124.9 s | 1.4× faster |

**Where that 7.7% comes from**: taking `TFLOP/s/device ÷ tokens/s/device` gives the FLOPs per token,
138.75 GFLOP for 80 layers and 38.16 GFLOP for 20 layers, a ratio of **3.636** (not 4.0).
Solving `F(L) = a·L + b` from the two points: 1.676 GFLOP/token per layer, and a **layer-independent term of 4.635 GFLOP/token**
(embedding lookup, the logits projection over a 120 K vocabulary, and loss).

The layer-independent share: **3.3% at 80 layers → 12.1% at 20 layers**, nearly 4× higher.
Solving back for the efficiency of the two kinds of work: inside the decoder layers ≈ 19.9% MFU, the layer-independent part ≈ 10.1% MFU.

> ⚠️ **This is a two-point fit for two unknowns, an exactly determined solution with no degrees of freedom left for testing.** It is self-consistent but does not constitute proof.
> The other candidate (the `2x2x4` topology having one dimension of only 2, degrading the wrap-around links) has no independent evidence either.
> **The discriminating experiment is running 64 chips / 20 layers**, which has not been done.

**What small scale really saves is not compilation, it is queuing.** Compilation barely varies with scale, and end to end it is only 1.4× faster
(most of the saving is the 60 seconds of slice creation across 16 hosts). The real value is that **4 nodes are far easier to schedule than 16**.

</details>

---

## Appendix D: Further reading

| Document | Contents |
|---|---|
| [QUICKSTART-v7.md](QUICKSTART-v7.en.md) | **The best recipe + end-to-end reproduction**; follow it and you get 580 |
| [QUICKSTART-v5p.md](QUICKSTART-v5p.md) 🇨🇳 | The v5p version, with a full architectural breakdown |
| [TUNING-v5p.md](TUNING-v5p.md) 🇨🇳 | v5p tuning practice, including the full chain of reasoning behind the tokamax LUT root cause |
| [EXPERIMENT-LOG.md](EXPERIMENT-LOG.md) 🇨🇳 | The complete experiment archive, with post-mortems on 12 bugs |
| [MAXTEXT-PORTING-GUIDE.md](MAXTEXT-PORTING-GUIDE.md) 🇨🇳 | The general pattern for porting other models into MaxText |
