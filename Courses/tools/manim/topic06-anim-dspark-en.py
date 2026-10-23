# -*- coding: utf-8 -*-
"""topic06-anim-dspark.py 的英文版：只换画面里的字，布局与时序原样（见 _lang_en.py）。
措辞跟专题六英文页（块 08）里两段动画的 aria-label、图注一致：
第一刀 / 第二刀 → cut one / cut two、草稿第 N 格 → draft slot N、把握 → confidence、小头 → (sequential) head。
画面里的英文例词（Yes, / of / course / problem）是中文版原样。

📌 渲染（在 Courses/ 下）：
   构建手册/脚本/render.sh tools/manim/topic06-anim-dspark-en.py DSparkDraft WebPages/media/topic06-dspark-draft.en.mp4
   构建手册/脚本/render.sh tools/manim/topic06-anim-dspark-en.py DSparkSched WebPages/media/topic06-dspark-sched.en.mp4
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lang_en import run_en  # noqa: E402

EN = {
    # ── DSparkDraft
    "DSpark 第一刀：并行打底，再按前一格改下一格": "DSpark cut one: parallel base, each slot adjusted by the previous",
    "上一轮确认的字": "Confirmed last round",
    "例子出自 DSpark 论文；分数、条长都是示意": "Example from the DSpark paper; scores and bar lengths are schematic",
    "并行（DFlash）：一趟前向，两格同时出分数": "Parallel (DFlash): one forward pass, both slots score at once",
    "第 2 格看不见第 1 格选了啥，只能两种说法各押一半": "Slot 2 can't see what slot 1 picked, so it bets half on each phrase",
    "两格各自抽签：第 1 格抽到 of，第 2 格抽到 problem": "Each slot samples on its own: slot 1 draws of, slot 2 draws problem",
    "✗ 拼成「of problem」：两种说法串了（按这组数，约一半概率）":
        "✗ Spliced into \"of problem\": the phrases got crossed (about half the time here)",
    "第 1 步完成：一趟出整块很快，但越往块尾越容易拼串":
        "Step 1 done: a block in one pass is fast, but splicing gets likelier toward its end",
    "DSpark：骨干还是这一趟；再加一个顺序小头，从左往右走":
        "DSpark: same one-pass backbone, plus a small head that walks left to right",
    "第 1 格定了 of：小头按 of 给第 2 格加一笔分": "Slot 1 settles on of: the head adds a bias to slot 2 based on of",
    "前一格的字 → 下一格的偏置": "token → bias",   # 只有箭头那么宽（1.9）；中文原句 2.8 宽，本来就压着两个框的下沿
    "course 抬到 95%，problem 压到 5%；再抽": "course rises to 95%, problem drops to 5%; sample again",
    "✓ 「of course」": "✓ \"of course\"",       # 直引号：Noto Sans CJK 把弯引号排成全角，前后各空出半个字
    "第 2 步完成：块里的字前后接得上了": "Step 2 done: tokens in the block now fit together",
    "代价：贵的骨干只跑一趟；小头每格只是查一次表、乘一个小矩阵":
        "Cost: backbone runs once; per slot the head adds a lookup and a small matmul",
    "骨干 5 层": "Backbone",
    "顺序小头": "Small head",
    "batch 128：草稿 4 到 16 个，整轮只比 DFlash 慢 0.2–1.3%":
        "Batch 128, 4 to 16 draft tokens: a round is only 0.2–1.3% slower than DFlash",
    "第 3 步完成：接受长度比 DFlash 高 16–18%（Qwen3 4B–14B）":
        "Step 3 done: acceptance length 16–18% higher than DFlash (Qwen3 4B–14B)",
    "一趟出整块的速度留着，块里的前后照应补回来":
        "Keep the speed of whole-block drafting, and restore coherence within the block",
    # ── DSparkSched
    "DSpark 第二刀：按把握排队，按负载定验几个": "DSpark cut two: rank by confidence, size by load",
    "请求 1 · 写代码": "Request 1 · coding",
    "请求 2 · 闲聊": "Request 2 · chat",
    "请求 3 · 解题": "Request 3 · math",
    "已放进验证的草稿数": "Draft tokens admitted to verification",
    "预期吞吐": "Expected throughput",
    "算法照 DSpark 论文 Algorithm 1 现算；把握和吞吐表是示意":
        "Computed live with Algorithm 1 from the DSpark paper; confidences and throughput table are schematic",
    "每个请求猜 5 个，小头给每格估一个「把握」": "Each request guesses 5 tokens; the head estimates a confidence per slot",
    "把握＝前面几格都收下时，这一格也被收下的概率":
        "Confidence = P(this slot accepted | all earlier slots accepted)",
    "把握从左往右连乘＝活到这一格的概率，越往后越低":
        "Multiply left to right: chance of surviving to each slot, falling along the row",
    "第 1 步完成：闲聊那行掉得最快，写代码那行最稳": "Step 1 done: the chat row falls fastest, the coding row holds best",
    "闲时：全批候选按存活概率从高到低，一个个放进验证":
        "Idle: admit candidates from the whole batch one by one, highest survival first",
    "✗ 红框那格放进来吞吐反降，停": "✗ Admitting the red cell lowers throughput: stop",
    "忙时：同一批草稿，但每多验 1 个，整步都更慢": "Busy: same drafts, but each extra token verified slows the whole step",
    "名额按把握排，总数看机器忙不忙；没把握的字不占名额":
        "Rank by confidence, size by load; low-confidence tokens don't take slots",
}
PAT = [
    (r"草稿第 (\d+) 格", r"Draft slot \1"),
    (r"第 (\d+) 格", r"Slot \1"),
    (r"一步耗时 ＝ 固定 (\d+) ＋ 每个 token 1", r"Step time = fixed \1 + 1 per token"),
    (r"第 2 步完成：多验几乎不拖慢这一步，放进 (\d+) 个，只剩 4% 那格没要",
     r"Step 2 done: verifying more barely costs; \1 admitted, only the 4% cell left out"),
    (r"第 3 步完成：只放进 (\d+) 个 —— 代码 (\d+) 个、闲聊 (\d+) 个、解题 (\d+) 个",
     r"Step 3 done: only \1 admitted: \2 coding, \3 chat, \4 math"),
]

run_en(__file__, EN, PAT, globals())
