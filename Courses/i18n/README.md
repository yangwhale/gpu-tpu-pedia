# English edition — how it is made

The English pages are **re-authored**, not machine-translated: same facts, numbers, figures and layout as the
Chinese page; wording written as a native ML-systems engineer would write it. The Chinese page stays the
single source of truth.

- `STYLE.md` — register, rewrite rules, typography, hard HTML/SVG rules
- `GLOSSARY.md` — canonical English for every recurring term
- `AGENT-PROMPT.md` — the per-chunk brief given to writing agents (with lessons learned)
- `REVIEW-PLAN-topic-02.md` — the ten review dimensions and their results
- `ISSUES-topic-02.md` — problems found in the Chinese source while translating
- `tools/` — `split.py` (cut the built page into chunks) → agents write `NN.en.html` →
  `check.py` (structure / residual CJK / numbers / SVG width) → `assemble.py` (rejoin; English-only links,
  embedded font, fold switch, language switch) → `overlap.py`, `boxfit.py`, `layout.py`, `visible.py`,
  `sbs.py` (side-by-side zh/en slices), `svgfit.py` (enlarge figure text without moving anything)

Published: `WebPages/topic-02.en.html`, `WebPages/index.en.html`.
