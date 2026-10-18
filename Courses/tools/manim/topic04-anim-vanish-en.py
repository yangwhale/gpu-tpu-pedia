# -*- coding: utf-8 -*-
"""topic04-anim-vanish.py 的英文版：只换画面里的字，布局与时序原样（见 _lang_en.py）。
措辞跟 topic-04.en.html 里这张图的 aria-label / 图注一致。

📌 渲染（在 Courses/ 下）：构建手册/脚本/render.sh \
        tools/manim/topic04-anim-vanish-en.py Vanish WebPages/media/topic04-vanish.en.mp4
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lang_en import run_en  # noqa: E402

EN = {
    "对数刻度": "log scale",
    "线性刻度": "linear scale",
}
PAT = []
run_en(__file__, EN, PAT, globals())
