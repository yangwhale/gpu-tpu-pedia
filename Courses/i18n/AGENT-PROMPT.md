You are re-authoring part of a Chinese technical course page into professional English.
The course: "Modern AI Accelerators: a GPU/TPU systems course", Topic 2 "TPU vs GPU" (L200 edition).
Thesis of this topic: on the spec sheet the TPU v7 (Ironwood) and NVIDIA B200 look nearly identical
(compute, HBM bandwidth, on-chip SRAM within 10%; both ridge points ≈312 FLOPs/byte). The real
difference is *who makes the decisions*: the GPU leaves them to hardware at runtime (caches,
schedulers, scoreboards → the kernel author), the TPU fixes them at compile time (the compiler).
Everything — memory access, matmul units, interconnect, software stack — follows from that, and
operator fusion is the thread running through the whole lecture.

READ FIRST (mandatory, fully):
  /home/chrisya/gpu-tpu-pedia/Courses/i18n/STYLE.md
  /home/chrisya/gpu-tpu-pedia/Courses/i18n/GLOSSARY.md
Optional context (whole page as plain text, Chinese): /tmp/t02en/text.txt (regenerate per topic)

YOUR CHUNKS (HTML fragments): {chunks}
For each /tmp/t02en/chunks/NN.zh.html write /tmp/t02en/chunks/NN.en.html.

How to work:
1. Read the Chinese chunk completely and understand the argument before writing anything.
2. Write the English version. Approach per paragraph: what is the author claiming, and how would a
   senior NVIDIA/Google engineer write that claim? Not word-by-word. Keep every fact, number,
   caveat and source citation. Keep the HTML structure byte-identical except text (STYLE.md §4).
   Inline SVG figures: translate every <text>/<tspan> under the width budget in STYLE.md §5.
   Large chunks: write the file with a Python script that does exact string replacements on the
   Chinese file (read zh → replace each text run → write en), or write it out in parts — whatever
   keeps structure intact. Do not hand-retype SVG geometry.
3. Verify, and loop until clean:
     python3 /home/chrisya/gpu-tpu-pedia/Courses/i18n/tools/check.py NN.zh.html NN.en.html --render
   (run from /tmp/t02en/chunks). Must reach: no residual Chinese, structure identical, all SVG text
   within 108% of the Chinese width. Number diffs: every difference must be a legitimate conversion
   (e.g. 6710亿→671B); otherwise you dropped or changed a number — fix it.
4. Final self-review as a skeptical native-English GPU/TPU expert: reread your English only. Fix
   anything that sounds translated (calques, odd collocations, Chinese rhythm, "this ledger" tics,
   overlong dash chains), wrong terminology, or claims that read differently from the Chinese.
5. If the Chinese has a factual problem or ambiguity, don't fix it silently: log it in
   /tmp/t02en/chunks/NOTES-NN.md. Also log any glossary term you had to invent.

Do not edit anything outside /tmp/t02en/chunks/. Do not touch the Chinese files.
Final reply (short): per chunk — check.py result line, number-diff judgement, count of SVG labels
where you reduced font-size, and anything in NOTES worth my attention.

LESSONS (topic-02 run, 2026-10-17):
- 3 of 8 agents stalled (10-min watchdog) by emitting one giant response. Always work incrementally:
  extract runs → translate ~100 per tool call → save after each batch → build with an exact-replacement script.
- Class CSS (.lbl{font:…}) overrides a font-size= attribute; shrink only via inline style.
- check.py ratio passes can still clip at the SVG edge: run tools/overlap.py (zh page vs en page) after assembly.
