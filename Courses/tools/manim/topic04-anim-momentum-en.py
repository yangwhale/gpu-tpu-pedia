# -*- coding: utf-8 -*-
"""topic04-anim-momentum.py 的英文版：只换画面里的字，布局与时序原样（见 _lang_en.py）。
措辞跟 topic-04.en.html 里这张图的 aria-label / 图注一致。

📌 渲染（在 Courses/ 下）：构建手册/脚本/render.sh \
        tools/manim/topic04-anim-momentum-en.py Momentum WebPages/media/topic04-momentum.en.mp4
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lang_en import run_en  # noqa: E402

EN = {
    "没有动量：停在第一个坑": "No momentum: stops in the first pit",
    "带动量：把上一步的速度留下来": "With momentum: keep the previous step's velocity",
    "冲过了头": "Overshot",
    "上一步的速度，留下 β 那么多": "The previous step's velocity, keeping β of it",
    "走的不是这一步的梯度，是攒下来的速度": "You move by the accumulated velocity, not this step's gradient",
}
PAT = []
run_en(__file__, EN, PAT, globals())
