# -*- coding: utf-8 -*-
"""topic06-anim-flow.py 的英文版：只换画面里的字，布局与时序原样（见 _lang_en.py）。
措辞跟专题六英文页（topic-06.en.html，块 01 / 05）里两段动画的 aria-label、图注一致：
显存 → HBM（仓库 warehouse）、算的地方 → compute（工作台 workbench）、搬 / 算 → move / compute、
卡 → Dev、提问 / 笔记 → question / notes、谁小搬谁 → move whichever is smaller。

📌 渲染（在 Courses/ 下）：
   构建手册/脚本/render.sh tools/manim/topic06-anim-flow-en.py DecodeTrip WebPages/media/topic06-trip.en.mp4
   构建手册/脚本/render.sh tools/manim/topic06-anim-flow-en.py WhoMoves   WebPages/media/topic06-whomoves.en.mp4
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lang_en import run_en  # noqa: E402

EN = {
    # ── DecodeTrip
    "每出一个字，权重都要从显存搬一趟": "Every token means hauling the weights out of HBM once more",
    "显存（仓库）": "HBM (the warehouse)",
    "算的地方（工作台）": "Compute (the workbench)",
    "37 GB 权重": "37 GB of weights",
    "1 个人 ＝ 1 行": "1 user = 1 row",
    "搬": "Move",
    "算": "Calc",
    "只算权重的理想下限：DeepSeek-V3、一颗 TPU v7、FP8；条长按真实比例，权重画成 8 格是示意":
        "Ideal lower bound, weights only: DeepSeek-V3, one TPU v7, FP8; bars to true scale; 8 weight blocks are illustrative",
    "一个人：每层权重从显存搬上来，只跟这 1 行乘一下":
        "One user: each layer's weights come up from HBM to multiply just this 1 row",
    "约 5 毫秒": "~5 ms",
    "0.016 毫秒：算力只用上 0.32%": "0.016 ms: only 0.32% of compute used",
    "出 1 个字": "1 token",
    "第 1 步完成：搬约 5 毫秒，算 0.016 毫秒，出 1 个字":
        "Step 1 done: ~5 ms moving, 0.016 ms computing, 1 token out",
    "下一个字：工作台放不下 37 GB，权重得从头再搬一趟":
        "Next token: the workbench can't hold 37 GB, so the weights are hauled again",
    "第 2 步完成：又是一整趟、又是约 5 毫秒，还是只出 1 个字":
        "Step 2 done: another full trip, another ~5 ms, still just 1 token",
    "一趟喂 313 行：搬的还是这一车，算一点点追上来":
        "Feed 313 rows per trip: the same load is hauled, compute catches up",
    "313 行": "313 rows",
    "313 × 0.016 ≈ 5 毫秒": "313×0.016 ≈ 5 ms",
    "第 3 步完成：算和搬一样长 —— 再多就卡在算，prefill 一趟几千行就在这一边":
        "Step 3 done: compute = moving; beyond this it's compute-bound, like prefill",
    "decode 卡在搬，prefill 卡在算": "Decode is bound by data movement, prefill by compute",
    # ── WhoMoves
    "谁搬家：PCP 搬笔记，DCP 搬提问": "Who moves: PCP moves the notes, DCP moves the question",
    "卡 0": "Dev 0", "卡 1": "Dev 1", "卡 2": "Dev 2", "卡 3": "Dev 3",
    "4 张卡、4 段都是示意": "4 devices and 4 segments are illustrative",
    "PCP（prefill）：几千个提问按行分给 4 张卡，每张卡先拿自己那段笔记":
        "PCP (prefill): many questions split over 4 devices, each with its own notes",
    "要合的笔记段": "Note segments to merge",
    "灰框＝三角外：靠前的提问看不到后面的字，路过不用算":
        "Gray cell = outside the triangle: earlier questions can't see later tokens, so it passes uncomputed",
    "PCP 完成：提问不动，笔记绕环搬过来；每张卡边收边合，不用跨卡拼":
        "PCP done: questions stay, notes travel the ring; each device merges as it receives",
    "DCP（decode）：每步只剩 1 个提问；笔记按字轮流存在 4 张卡上，不动":
        "DCP (decode): just 1 question per step; notes stay, stored round-robin on 4 devices",
    "1 个提问": "1 question",
    "第 1 步：这一个提问发到每张卡，各算自己那一段":
        "Step 1: this one question goes to every device; each computes its own segment",
    "分子、分母": "Num., denom.",
    "第 1 步完成：每张卡交出自己那段的分子和分母，先别除":
        "Step 1 done: each device hands over numerator and denominator, undivided",
    "第 2 步：在终点拼一次 —— 分子加总、分母加总，只除一次":
        "Step 2: at the end, sum numerators, sum denominators, divide once",
    "分子加总 ÷ 分母加总 ＝ 这个提问的答案": "Σ numerators ÷ Σ denominators = the answer",
    "第 2 步完成：笔记没动，提问搬了家，答案在终点拼":
        "Step 2 done: notes stay put, the question moves, answers combine at the end",
    "谁小搬谁：提问多就搬笔记（PCP），只剩 1 个提问就搬提问（DCP）":
        "Move whichever is smaller: many questions, move notes (PCP); one, move it (DCP)",
}
PAT = [
    (r"提问 第 (\d+) 段", r"Question seg. \1"),
    (r"笔记 第 (\d+) 段", r"Notes seg. \1"),
    (r"第 (\d+) 站：笔记段全体往右传一站，最右边那段绕回卡 0",
     r"Hop \1: every note segment moves one hop right; the rightmost wraps to Dev 0"),
    (r"第 (\d+) 站完成：用得上的边收边合，三角外的路过",
     r"Hop \1 done: needed segments merge on arrival; others pass through"),
    (r"第 (\d+)、(\d+)、(\d+)… 个字", r"Tokens \1, \2, \3…"),
]

run_en(__file__, EN, PAT, globals())
