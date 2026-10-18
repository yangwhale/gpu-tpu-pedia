# -*- coding: utf-8 -*-
"""topic04-anim-reverse.py 的英文版：只换画面里的字，布局与时序原样（见 _lang_en.py）。
措辞跟 topic-04.en.html 里这张图的 aria-label / 图注一致。

📌 渲染（在 Courses/ 下）：构建手册/脚本/render.sh \
        tools/manim/topic04-anim-reverse-en.py Reverse WebPages/media/topic04-reverse.en.mp4
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lang_en import run_en  # noqa: E402

EN = {
    "正向模式": "forward mode",
    "每趟只到手": "each pass yields only",
    "反向模式": "reverse mode",
    "一趟就到手": "one pass yields",
    "8 个参数": "8 parameters",
    "走过的趟数": "passes walked",
    "同一条链，两个方向同时开跑": "Same chain, both directions start at once",
    "下面这条从 loss 出发，一趟就跑完了": "The bottom one starts from the loss, done in one pass",
    "上面这条还在跑 ——　八个参数，八趟": "The top one is still running: eight parameters, eight passes",
}
PAT = []
run_en(__file__, EN, PAT, globals())
