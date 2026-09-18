# 评审原文 · M5-budget

> ⛔ 这是 subagent 的**原始输出，未经我加工**。
> 认它报的**现象**，不认它给的**根因** —— 落地前逐条核。

---

Read the full file (all 6,661 lines; body 787–6,603, 35 figures/animations). Here's the report from a budget-holder's seat.

---

## Three-sentence version (can I brief my boss? — yes, and that's a credit to the course)

"Training costs 3x inference in compute, but compute isn't the problem. Memory is: every parameter carries 16 bytes instead of 2, and 12 of those 16 are optimizer bookkeeping — six times the model itself. Everything else people talk about (recompute, ZeRO, LoRA, Muon) is four different ways of attacking the four line items on that one bill."

Those three sentences come from §0, §3.1 Ⓑ, and §5.5 — about 40 lines out of 6,600. **The other 6,560 lines do not fit into that story.** §1.1–1.5 is a different course ("how backprop works"). §3.3 + §3.5 + §3.6 is a third course ("how optimizers were invented and how to tune an LR"). By line count that's roughly **45% of the document with no consequence for the bill** — excellent material, but nobody tells me I can skip it. The hero says "按「谁最大」读，不按流程读" and then gives a linear 0→7 layout.

---

## 1. Numbers that never land in reality

Grepped the whole file: **there is no dollar figure, no GPU-hour, no wall-clock day, and exactly one card count in 960 KB.**

| Where | Number | What I can't tell |
|---|---|---|
| §1.6 `fig-act-bill`, §2.2 `fig-recompute` | 4.15 TiB → 106.75 GiB | How many cards is 4.15 TiB? Never said. |
| §5.1 | 9.76 TiB → "125 张" | The only conversion in the document — and it's immediately neutered ("80 GiB 是随手设的换算基准"). It also covers only the resident block; add the 94-sequence activation break-even from `fig-step` Ⓑ and it's ~250 cards, which the course never computes. |
| §5.2 | 29.1 PFLOP → "实际接近 160 PFLOP" per one 128K sequence | **The biggest missed conversion in the document.** 160 PFLOP for a single sequence is ~400 accelerator-seconds. That one line would make the whole long-context argument visceral. Instead PFLOP sits as a bare unit. |
| §2.3 table | 2,571 / 703.69 / 48.9 TFLOP | Raw. (The ratios — 1.9%, 33% — are usable; the absolutes aren't.) |
| §3.7 | checkpoint 7.3 TiB (V3: 4.9 TiB) | How long to write it? How often can I afford to? **Checkpoint cadence is a pure cost decision and it's the prerequisite for §6.2's rollback play** — and neither cost is given. |
| §6.1 | PaLM: "loss 飞了大约 20 次", rollback ~100 steps, skip 200–500 batches | **The most CFO-legible fact in the course, left unpriced.** 20 incidents × 100 steps of a 540B run = how much money in the bin? Plus a human at 3 a.m. each time. |
| §3.4 📌 | Moonlight: "Muon 的计算效率约为 AdamW 的两倍" | 2x compute efficiency means halving a training run. It is delivered as one clause inside a footnote, then hedged. This should be a headline with a caveat, not a caveat with a number in it. |
| §1.7 | MoE 派发 "9 份" copy fee, 31.5 GiB/layer | Real architecture-cost reversal — and it lives inside a section that opens "⭐ 这一小节可以整段跳过". |

**The one anchor the course already has and doesn't use:** it cites arXiv 2412.19437 at least six times (§3.2.3, §3.3.3, §4.2). That same paper publishes **2.788M H800 GPU-hours on a 2,048-GPU cluster, ~57 days, $5.576M at $2/GPU-hour**. The course's own §7.2 closing method note says an estimate is credible when it matches an external anchor. Here the anchor is in a paper already on the reference list.

---

## 2. Decision points — can I decide?

| Decision | Verdict | What's missing |
|---|---|---|
| Recompute on/off (§2.2) | **Yes.** 33% compute for ~40x memory, "default on". Clean. | — |
| Which recompute tier (§2.3) | **Almost.** The sorted FLOPs-per-byte table is the best-engineered decision aid in the course. | The deciding quantity is named — "一 GiB 显存对你值多少 TFLOP" — and **never computed once, anywhere**. Without one worked exchange rate, "在哪儿切一刀" is not a decision I can make, and "性价比高一个数量级" is not a claim I can audit. |
| Optimizer choice (§3.4) | **No.** | Bytes/param is there (16/12/10/8), which is great. But the course explicitly refuses any convergence comparison ("画曲线就是编" — honest, and operationally useless). So "switch to Muon?" rests on two facts placed 200 lines apart and never juxtaposed: *per-step slower* vs *2x compute efficiency*. And **8-bit Adam saves 6 B/param = 37.5% of the resident block — the single largest lever after ZeRO — gets four lines and no recommendation.** |
| Precision (§3.2) | **Yes — best chapter in the course.** "老的贡献会不会永远不走" is a criterion I can apply to a tensor I've never seen, and it was sharpened by a first-party counterexample. | One missing line: V3's bf16 m/v takes 16 B → 12 B. §3.7 converts that to checkpoint size (7.3→4.9 TiB) but **never to resident memory or cards** (25% off = ~31 fewer cards at the 125-card scale). |
| Parallel strategy (§4) | **Deferred to 专题五** — fine. | But see the §4.1 hole below. |
| LoRA vs full finetune (§3.8) | **Yes — the best-costed decision in the document.** 100.4 GiB vs 12.68 GiB, 7.9x, plus the warning that activations *don't* shrink. | The course itself calls this "大多数人真正会碰到的那种训练" and then puts it at the bottom of section 3. |
| Stability (§6.3) | **Partially, and correctly so.** The ST-MoE table is exactly right — stability AND quality in the same frame, with the "稳定 3/3 但质量被打穿" trap. | Ends in "no one knows the root cause", which is the honest answer. |

---

## 3. Structural defects I'd want fixed before this goes out

**§4.1 is empty.** Heading: "先把账按大小排一遍". Body: one sentence saying *"这个顺序本身就是这一节的全部内容"* — and then nothing. No list, no table. §4.2 and §5.5 both refer back to "上面那个顺序". This is a hole at the exact hinge of the argument (HTML lines 5364–5370).

**A self-contradiction in §5.4.** The small-model worked example says recompute gives **6.7x** memory savings, then asserts *"比例跟 671B 上那笔几乎一样"* and *"换了 5,000 倍的规模，那些比例基本没变"*. On the 671B case the same lever gave **40x** (§2.2). The 33% compute side is unchanged; the memory side is off by 6x. A reader who carries "recompute saves 40x" into a small-model capacity plan will be badly wrong — and this subsection exists specifically to be the one people compute themselves.

**`fig-zero` is on the wrong model.** The clearest budget picture in the entire course — 120 → 31.4 → 16.6 → 1.9 GB per card, a 63x cut — is computed on **7.5B params / 64-way DP** (ZeRO's 2019 Figure 1). Everything else in the course is 671B. The reader cannot connect the two, so the one figure that shows what slicing actually buys never touches the bill they've been building for five sections.

**§2.3's "一个 step ≈ 2,571 TFLOP" is per-layer, not per-model.** The sentence reads "一层省 53.5 GiB，付 48.9 TFLOP。换算到一个 step（前向 ＋ 反向 ＝ 3 遍 ≈ 2,571 TFLOP）". The ratio is correct (both per-layer; 2,571 × 61 ≈ 157 PFLOP, matching §5.2), but "一个 step" without "一层" invites a 61x misread.

---

## 4. Credibility: strong, with one delivery flaw

§7.1 / §7.2 — splitting "citable" from "self-derived", with a *what-could-be-wrong* column on each derived number — is the best thing in the document and I would show it to an executive unedited.

The flaw is placement. In the body, a derived number (4.15 TiB, 82.1%, S≈5,734, 94 条, "6ND 低估五到六倍") is **typographically identical** to a cited one (16 B, 33%, 2.7%). The ⚠️/📌/⭐ markers are used so densely, and also for rhetorical emphasis, that they've stopped functioning as a provenance signal. A per-number badge (推 / 引) would fix it in one pass.

Numbers I would **not** put in a slide:
- "6ND 低估五到六倍" (§5.2) — derived from 82.1%, which is itself derived, with no third-party corroboration. The course says so, but the body states it in bold ⭐⭐⭐ as if it were a finding.
- "Muon ≈ 2x AdamW" (§3.4) — the paper's own scaling law, on 3B/16B, which the course's own §2.6 rule forbids extrapolating. Correctly flagged, but the reader is left holding a 2x claim they can neither use nor discard.
- The §5.4 "比例几乎一样" line above — currently just wrong.

Numbers I'd happily cite: the 16 B breakdown, 33% recompute overhead, ZeRO's three levels, GPT-3's eight-row LR/batch table, PaLM's 20 spikes + the "不是数据坏" ablation, ST-MoE Table 4, the bf16 1.0+0.0003×1000=1.0 demonstration. All sourced, several script-verified with asserts.

---

## 5. The money thread: where it breaks

| Link | Status |
|---|---|
| Why training is expensive | **Connected.** §1 (3x) → §1.6 (activation mountain) → §3.1 (16 B/param, 2:12 ratio). |
| What is physically unavoidable | **Parts exist, never assembled.** §1.5 (one backward is the floor vs 300 billion forwards), §1.6 ("是反向的数学要求它在场"), §3.2 (fp32 master weight, unless stochastic rounding). **Nowhere does the document say "here is the irreducible bill, here is the removable bill."** That is the first question my role asks, and the course has every piece and never puts them together. |
| What engineering removes | Listed but **never composed.** Recompute 97%, ZeRO up to 63x, optimizer 16→8 B, LoRA 7.9x, precision. Can I stack ZeRO-3 + selective recompute + 8-bit Adam? §3.4 answers one 3-way interaction in one sentence; the big ones are unaddressed. |
| A total | **Absent.** There is no single table of the form *baseline → +recompute → +ZeRO-1/2/3 → per-card GiB → cards*. |
| Compute cost ↔ memory cost | **Never meet.** `fig-recompute` itself admits it ("一边 TiB 一边 TFLOPs，两根条没法比") and solves it by normalizing both to 100% — i.e. it names the problem and declines to solve it. One shared unit (dollars, or accelerator-seconds) would close it. |

---

## 6. The buried high point

The strongest reversal in the whole course is in §2.4–§2.5, and it is delivered flat:

> The criterion doesn't change a word. In 2022 it says *recompute attention*. At 128K it says *never recompute attention*. **And Megatron-LM still ships `recompute_modules = ["core_attn"]` as its default** — a 2,048-token-era answer, still the out-of-the-box setting.

That is a thriller and it is immediately actionable ("go check your framework's default"). Right now the punchline arrives as the fourth sub-bullet of §2.5, roughly 350 lines after the section's climax, following two self-corrections. It should open section 2.

Two other buried reversals:
- **§5.4 / §5.3 Ⓑ**: on a 125M model at seq 1024 batch 8, **activations (2.11 GiB) beat the resident block (1.86 GiB)** — i.e. the course's own headline thesis ("最大的是优化器状态") is false on the machine most readers own. §5.3 Ⓑ says the same thing ("谁最大不是模型的属性，是这次训练配置的属性"). Both are structurally downplayed; this should be a named turn, not a sub-step.
- **§1.7**: MoE's 9x activation copy fee — the cost reversal on the architecture everyone is adopting — sits inside a section labeled skippable.

Also worth noting, in the other direction: only **6 of 35 figures** carry a number I can act on (`fig-act-bill`, `fig-recompute`, `fig-per-byte`, `fig-optimizers` Ⓑ, `fig-zero`, `fig-step` Ⓑ). The other 29 are conceptual. For a mode that collapses prose to figures (the `T` / 折叠讲解 toggle), that means figure-only reading gives a budget owner almost nothing.

---

## If I could only change 5 things

1. **Add one "total bill" table and one worked exchange rate.** For 671B/128K: line item → baseline → +recompute → +ZeRO-1/2/3 → per-card GiB → cards. Plus a single worked "1 GiB of HBM is worth X TFLOP to me" example (pick a card, pick a price, show the arithmetic once). This closes four of the five breaks in the money thread at once and turns §2.3 from a ranked table into an actual decision.

2. **Fill §4.1 (it is empty) and re-run `fig-zero` on the 671B case.** The empty section is the argument's hinge; the ZeRO figure is the best budget picture in the course and it's computed on a different model than everything else.

3. **Lead section 2 with the Megatron-default reversal.** Move "the default is optimized for a model generation that no longer exists" to the top of §2, then derive it. Highest-value, most surprising, most immediately actionable fact in the document, currently arriving 350 lines late.

4. **Price the two failure modes.** (a) PaLM's 20 spikes × ~100-step rollback = what % of the run thrown away; (b) a 7.3 TiB checkpoint at N GB/s takes M minutes, so checkpointing every K steps costs W% of throughput — which is also what sets the price of a spike. These two numbers convert §6's 上游/中游/下游 triage from a taxonomy into a budget argument, and they make §3.7 matter.

5. **Add a 15-line "unavoidable vs. removable" box and a bill-only reading path.** The parts all exist (§1.5 floor, §1.6 math requirement, §3.2 fp32 master weight, §2.2 removable, §4 slicable) — collecting them answers my first question. Pair it with a stated fast path (§0 → 1.6 → 2.2/2.3 → 3.1 → 5.4 → 5.5), because §1.1–1.5 and §3.3/3.5/3.6 are ~2,500 lines of genuinely excellent material that a budget reader should be *told* they may skip.

**Must-fix regardless of ranking:** the §5.4 "比例跟 671B 上那笔几乎一样" claim (6.7x vs 40x) is wrong, in the one subsection designed for readers to verify by hand.

---

## ⭐ 交付后的补充核对（同一个评审员的后续消息）

Re-verified with fixed-string counts instead of the regex that blew up. The claim holds, and one detail got sharper:

| Pattern | Hits | What they actually are |
|---|---|---|
| 美元 | 3 | **All three are the chain-rule metaphor** (人民币→港币→美元) in `fig-slider`. Zero currency. |
| `$` | 1 | A JS regex in the figonly script. |
| 成本 | 3 | Two are false positives (换成+本图 / 换成+本课); one is a quoted line from the PaLM paper ("训练成本太高"). **The course never uses the word "cost" in its own analysis.** |
| 张卡 | 19 | All are `每张卡` in `fig-zero` (per-card capacity, 7.5B model), generic "几百张卡", "一张卡装不装得下", §5.4's "一张卡", and the two instances of the single 125-card conversion (§5.1 + its §7.2 entry). |
| 机器 | 3 | A favicon code comment, "小数在机器里分成两段存", and the §5.4 heading. |
| H100 / H800 / A100 / B200 / GB200 / TPU / GPU-hour / 卡时 / 小时 / 分钟 / 电费 / 预算 / 价格 / 租 | **0 each** | — |

So the two load-bearing claims in my report are confirmed: **no dollar figure, no GPU-hour, no wall-clock duration anywhere in 960 KB, and exactly one card count** (125 cards, §5.1, repeated once in the appendix).

The sharpened detail worth adding to finding #1: **the only three appearances of "美元" in the entire document are money used as a teaching metaphor for the chain rule** — the course reaches for currency exchange to explain multiplication, and then never uses currency as a unit in the bill it spends 6,600 lines building. That is the money thread's break stated in one line, and it's a gift of a framing for fix #1 (add one worked exchange rate): the course already taught the reader that a chain of conversions is how you get from one unit to another — it just never runs that chain from TFLOP to dollars.

No changes made to any file; the only writes were my own extraction scratch file in `/tmp`.
