# -*- coding: utf-8 -*-
"""topic04-anim-descend.py 的英文版：只换画面里的字，布局与时序原样（见 _lang_en.py）。
措辞跟 topic-04.en.html 里这张图的 aria-label / 图注一致。

📌 渲染（在 Courses/ 下）：构建手册/脚本/render.sh \
        tools/manim/topic04-anim-descend-en.py Descend WebPages/media/topic04-descend.en.mp4
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lang_en import run_en  # noqa: E402

# ⚠️ 中文片第三幕写的是「三千亿维」，而中文页 aria-label 与英文页都写 6,710 亿 / 671 billion；
#   英文版跟英文页走（671B），中文片那处不一致留给中文管线处理。
EN = {
    "一维：落进哪个谷，看你从哪儿出发": "1D: which valley you land in depends on where you start",
    "二维：换成等高线俯视，形状还看得见": "2D: switch to a top-down contour view; shape still visible",
    "三千亿维：画不出来了 —— 改画一列数": "671B dimensions: can't draw it, so draw a column of numbers",
}
PAT = []
run_en(__file__, EN, PAT, globals())
