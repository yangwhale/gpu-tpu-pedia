# -*- coding: utf-8 -*-
"""topic04-anim-onelr.py 的英文版：只换画面里的字，布局与时序原样（见 _lang_en.py）。
措辞跟 topic-04.en.html 里这张图的 aria-label / 图注一致。

📌 渲染（在 Courses/ 下）：构建手册/脚本/render.sh \
        tools/manim/topic04-anim-onelr-en.py OneLR WebPages/media/topic04-onelr.en.mp4
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lang_en import run_en  # noqa: E402

# ⭐ 中文脚本自己的 `_place` 会断言每块字装进它的地盘，英文太长会当场炸 —— 下面的短写法都是为此。
EN = {
    "同一个谷、同一个起点 —— 只把学习率 η 从小扫到大": "Same valley, same start: sweep only the learning rate η, small to large",
    "陡方向（纵）": "steep, vertical",
    "平方向（横）": "flat, horizontal",
    "学习率 η": "learning rate η",
    "发散门槛": "threshold",
    "两侧细线 ＝ 门槛 ±5%": "thin lines = threshold ±5%",
    "左边收敛 / 右边发散 —— 只差一成": "converges / diverges: only 10% apart",
    "左边收敛": "converges",          # 上一句的 t2c 键
    "右边发散": "diverges",
    "η 太小：贴着谷底慢慢蹭": "η too small: creeping along the floor",
    "平方向 26 步只走掉三分之一": "flat direction covers only a third in 26 steps",
    "η 够大了：陡方向左右横跳": "η large enough: bouncing across the valley",
    "可幅度一跳比一跳小，还在收敛": "but each bounce is smaller, still converging",
    "η 越过门槛：每步在放大": "η past the threshold: every step amplifies",
    "几步就冲出画面 —— 是性质变了": "off-screen in a few steps: a change in kind",
}
PAT = []
run_en(__file__, EN, PAT, globals())
