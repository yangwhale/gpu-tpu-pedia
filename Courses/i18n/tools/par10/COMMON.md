PARITY ROUND — zh/en layout comparison of a published course page (author's request, 10-18 06:49:
"全面中英文版式对比，一个一个来，每个专题十轮"). Standard: the Chinese page is polished; at a glance the English must
look the same — size, color, position, organization, text-to-figure ratio. Find where English differs and fix it.
Workspace: {WORK} (English chunks {WORK}/chunks/NN.en.html, Chinese reference NN.zh.html). Rules: {WORK}/ROUND-COMMON.md
(if present) and {REPO}/Courses/i18n/STYLE.md. Never change facts/numbers; structure (tag sequence)
must stay identical to the Chinese chunk.
Materials for this round: {WORK}/par/{ROUND}/ — side-by-side slices (left zh, right en) s<width>-NNN.png with
index-<width>.json (starts_at labels = section ids / h3 text / figure ids), per-block height deltas layout-<width>.txt (its section-height table is unreliable when it reports a "tag mismatch" — ignore that table and use lines-<width>.txt),
overlap.txt (SVG text overlaps; "new in en" must be 0), lines-<width>.txt (every block where English has more lines or is ≥6px taller than Chinese, with section context — your main worklist; sort by size of the delta and fix the biggest first; ignore deltas that are only the deliberate +1px English table/caption font when line counts are equal). Reference pages: {WORK}/zh-ref.html (current Chinese, lecture
media stripped) and {WORK}/tmp.en.html (assembled from the chunks).
TODAY'S FOCUS (do this dimension thoroughly; note other defects you see but stay on focus): {FOCUS}
Fix in YOUR chunks only: {CHUNKS}. Text first (shorten wording, keep facts); figures: inline font-size / small geometry
moves allowed for collisions (log old→new). Global CSS/skeleton problems: don't edit — list them with the slice name.
Checks per touched chunk (run in {WORK}/chunks):
  I18N_WORK={WORK} I18N_TOPIC={TOPIC} I18N_GEOM_FREE=x,y,width,height,x1,x2,y1,y2 python3 {REPO}/Courses/i18n/tools/check.py NN.zh.html NN.en.html --render
  visible ratio must stay ≤ 1.9 (only where it already is): I18N_WORK={WORK} I18N_TOPIC={TOPIC} python3 {REPO}/Courses/i18n/tools/visible.py NN
  verified over-width labels: I18N_WORK={WORK} I18N_TOPIC={TOPIC} python3 {REPO}/Courses/i18n/tools/accept.py NN "PAR10-{ROUND}: <why>"
Re-render to verify: figures I18N_WORK={WORK} I18N_TOPIC={TOPIC} python3 {REPO}/Courses/i18n/tools/figidx.py N ;
prose: measure on an ASSEMBLED page (chunkview.py lacks the English size rules and is only a rough preview): cd {REPO}/Courses/i18n/tools && I18N_WORK={WORK} I18N_TOPIC={TOPIC} I18N_OUT={WORK}/par/{ROUND}/w-<you>/asm.en.html python3 assemble.py {TOPIC} {WORK}/chunks ; then compare with {REPO}/Courses/i18n/tools/par10/lines.py {WORK}/zh-ref.html <asm> <width> (never assemble without I18N_OUT — that overwrites the published page).
IMAGE BUDGET: reading too many images crashes the session (30 MB request limit). Read a few per turn, ≤ 45 total; use
layout/overlap numbers and DOM measurements to pick what to look at.
Back up each chunk before editing (NN.en.{ROUND}.bak). Helper files only in {WORK}/par/{ROUND}/w-<you>/.
Log: {WORK}/par/{ROUND}/LOG-<chunks>.md. Final reply (short): found/fixed by severity, top fixes, global-CSS items,
Chinese-source defects (if any).
PROVENANCE MARKERS ARE FACTS: never drop or blur "verbatim", "(our translation)", "not measured by this course", "our analogy", "formula-computed" etc. — they tell the reader what kind of claim it is. Shorten elsewhere.
QUOTES OF ENGLISH SOURCES: text inside “ ” that quotes an English-language source (paper, docs, blog) must stay verbatim. Never condense it while keeping the quote marks (that fabricates a quotation). Leave it as is; shorten the surrounding words instead.
(The QUOTES rule covers only ENGLISH-language sources. Quotes that are our translation of a Chinese source — Su Jianlin, Hung-yi Lee, Chinese blogs — may be tightened like any prose, but keep the "(our translation)"/"verbatim" marker.)
NO FONT SHRINKING: never reduce an SVG label's font-size below the Chinese size to make it fit (STYLE.md: "放不下就缩短英文，不许再缩字号"). Shorten the English, widen the box, or accept parity with the Chinese if the Chinese is equally tight.
