# -*- coding: utf-8 -*-
"""topic04-anim-saddle.py 的英文版：只换画面里的字，布局与时序原样（见 _lang_en.py）。
措辞跟 topic-04.en.html 里这张图的 aria-label / 图注一致。

📌 渲染（在 Courses/ 下）：构建手册/脚本/render.sh \
        tools/manim/topic04-anim-saddle-en.py Saddle WebPages/media/topic04-saddle.en.mp4
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lang_en import run_en  # noqa: E402

# ⭐ 两侧图例必须落在曲面转一圈都扫不到的空地里（中文脚本的 `_clear` 会断言），右侧只有 3.5 宽，
#   所以图例头写成「Red: up ↑」而不是 aria-label 里的「Along the red line: up」。
EN = {
    "同一个点，一个方向是谷，另一个方向是峰": "Same point: a valley one way, a peak the other",
    "谷": "valley",                  # 标题的 t2c 键
    "峰": "peak",
    "这样的点叫鞍点 —— 翻山的那个垭口": "Such a point is called a saddle point: the pass over the mountain",
    "沿红线走：往上 ↑": "Red: up ↑",
    "这条路是上坡，loss 变大": "uphill, loss increases",
    "沿绿线走：往下 ↓": "Green: down ↓",
    "顺着它接着走，loss 变小": "keep going, loss decreases",
    "黑点：这里梯度为零 —— 可它既不是谷底，也不是山顶":
        "Black dot: zero gradient, yet neither a valley floor nor a peak",
}
PAT = []
run_en(__file__, EN, PAT, globals())
