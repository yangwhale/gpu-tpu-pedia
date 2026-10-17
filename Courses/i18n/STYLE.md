# English edition — style guide

> **Scope (author, 2026-10-17):** core track only — per topic, one course page (L200) and one set of
> instructor notes. No L300, side tracks, microscopes or briefs in English.

The English edition is **not a translation**. It is the same course re-authored in
English by someone who knows the subject. A senior engineer at NVIDIA, Google or a
frontier lab should read it and not be able to tell it was ever Chinese.

The Chinese page is the **source of truth for facts, numbers, structure and figures**.
It is *not* the source of truth for wording, sentence order or rhetorical devices.

## 1. Register

- Benchmark: *How to Scale Your Model* (the JAX Scaling Book), Horace He's
  "Making Deep Learning Go Brrrr From First Principles", Stanford CS336 notes,
  Simon Boehm's "How to Optimize a CUDA Matmul". Confident, plain, second person.
- Short declarative sentences. One idea per sentence. No hype, no exclamation marks.
- Explain with numbers and causes, not adjectives.
- American spelling (behavior, optimize, color).

## 2. Rewrite, don't transliterate

| Chinese habit | Do this in English |
|---|---|
| Rhetorical fragments: 「那速度呢 —— L2 很大，但它并不快」 | A real heading: "Speed: L2 is big, but it isn't fast" |
| Stacked dashes 「……—— 也就是……—— 所以……」 | Split into 2–3 sentences, or use a colon |
| 「这笔账」「算账」「欠了一笔账」 (ledger metaphors everywhere) | "the math", "the accounting", "cost", "we owe you an explanation". Keep the ledger image only where it carries the argument (e.g. "the bill for runtime adaptation") |
| 「⭐ 落点」「⛔」「⚠️」 callout labels | Keep the emoji; write the label as a normal English lead-in ("The takeaway:", "Don't misread this as…") |
| Chinese-language wordplay / etymology (「束」这个字抓到了重点, 线程束) | Drop the Chinese-specific part, keep the insight if it survives in English (warp → weaving, the threads are bound together). Never explain a Chinese word to an English reader |
| 「说人话」 colloquialisms, 网络梗 | Equivalent plain English idiom, or just say it plainly |
| 「我们」 lecturer voice | "we" for the course, "you" for the reader. Fine to keep |
| Repetition for emphasis (Chinese tolerates it) | Say it once, well |

Length: English may come out shorter than a literal rendering. That is fine.
Never add new facts, claims, sources or numbers that are not in the Chinese.
Never drop a fact, a number, a caveat, or a source citation.

## 3. Typography

- Chinese quotes 「」『』 → “ ” (or *italics* for terms being defined).
- Full-width punctuation → ASCII: ，。：；？！（） → , . : ; ? ! ( ).
- 「——」 / `——&nbsp;` → em dash `—` with no spaces (US style). Don't overuse it.
- `＝` → `=`; `×` and `÷` stay; `～`/`~` for ranges → en dash `–` ("30–659 cycles").
- `①②③` circled numbers: keep (they often map to labels in figures).
- Headings: sentence case ("What the scoreboard actually tracks"), keep section numbers exactly ("3.2b").
- Numbers: thousands separator `,` (606,208). Chinese magnitudes converted:
  6710 亿 → 671B; 37 亿 → 3.7B; 1 万 → 10,000 / 10K. Keep the same precision.
- Units exactly as in the source: MiB/GiB vs MB/GB are *not* interchangeable.
  TB/s, GB/s, TFLOP/s, PFLOP/s, GHz, ns, µs. 「拍」/「周期」 = cycles. 「B/周期/SM」 = B/cycle/SM.

## 4. HTML rules (hard)

- Translate **text nodes and human-readable attributes** only: `title`, `alt`,
  `aria-label`, `placeholder`. Nothing else.
- Do not touch tags, nesting, `id`, `class`, `href`, `src`, `data-*`, `style`,
  inline SVG geometry (x, y, width, transform, path d, …), or `<script>` code.
- Inline markup inside a sentence (`<b>`, `<em>`, `<code>`, `<span class=…>`) must
  survive, but you may move it to wrap the English words that carry the emphasis.
  Keep roughly the same number of bold spans per paragraph — they are the skim layer.
- `<code>` content, identifiers, CLI flags, PTX/CUDA/JAX names: never translate.
- **Never link to a Chinese-only page** (author, 2026-10-17: an English reader who clicks lands on a wall of Chinese).
  The edition switcher keeps only the current edition; L300 / side tracks / microscopes / other topics become
  plain text (no "(Chinese)" suffix). Exception: the lecture notes of the same edition (to be translated next).
  `assemble.py` enforces this after the chunk checks, so writers may leave links in place.
- Keep the Chinese page's layout exactly: same collapsed `<details>`, same figure-first rhythm. If the English
  reads wordier than the Chinese, cut words — never move text out of a collapsed block.

## 5. SVG figures (hard)

English is wider than Chinese at the same font size. One CJK glyph ≈ 1 em;
average Latin ≈ 0.55 em. So **a label of N Chinese characters has a budget of
about 1.7 × N Latin characters** (digits and spaces included).

- Write figure text *telegraphically*: noun phrases, no articles, no full sentences
  unless the Chinese was a full sentence and space allows.
- Use standard abbreviations: HBM, SMEM, TMEM, VMEM, SM, TC, MXU, VPU, KV,
  "reg" for register, "B/cyc", "comp." Don't invent obscure ones.
- If it still doesn't fit, you may reduce that text's font size by up to 15% — **only via an inline `style="font-size:…"`** (class CSS such as `.lbl{font:…}` overrides a `font-size=` attribute, so the attribute silently does nothing). Rewording is preferred.
  Do **not** move elements, change coordinates, or wrap into new `<tspan>` lines
  unless the original already used tspans for multi-line text.
- `text-anchor` stays as is, so a centered label grows both ways; a start-anchored
  label grows only to the right — check what's to its right.
- After you're done, the automated check renders both versions in Chrome and
  flags any text whose English bounding box is >8% wider than the Chinese one.

## 6. When the Chinese is wrong or unclear

Do not silently fix facts. Write the issue into the chunk's notes file
(`NOTES-<chunk>.md`) with the original sentence and why. Translate what is there.

## 图内英文字号规范（2026-10-17 专题一并排审后定）
- 同 px 下拉丁字看着比中文小约 20–25%，所以英文图字要比中文 class 字号略大，且**同一张图内统一**
- 图标题 / 图内分区标题：14px 左右，成对标题（A/B、①/②、左右两栏）必须同号，最大不超过 15px
- 正文标签：svglbl 12 → 13.5px；svgsm 10.5 → 11.5px；等宽数字可保持 class 字号；最小 11px（等宽 10.5px）
- 放不下就**缩短英文**（保意思、保数字），不许再往小里缩；实在不行才挪位置
- JS 画的图要在 序列 512/4K/128K/1M × batch 1/8/64 × bf16/fp8/fp32 全部组合下不溢出不重叠
- 手机端：表头允许换行（assemble 已注入），关键列不能被挤出屏幕
