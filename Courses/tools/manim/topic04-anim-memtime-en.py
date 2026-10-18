# -*- coding: utf-8 -*-
"""topic04-anim-memtime.py 的英文版：只换画面里的字，布局与时序原样（见 _lang_en.py）。
措辞跟 topic-04.en.html 里这张图的 aria-label / 图注一致。

📌 渲染（在 Courses/ 下）：构建手册/脚本/render.sh \
        tools/manim/topic04-anim-memtime-en.py MemTime WebPages/media/topic04-memtime.en.mp4
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lang_en import run_en  # noqa: E402

# ⭐ 顶上三条图例排成一行、居中，中英总宽都得 < 画面宽 —— 英文只能写缩略（↑ 前向堆、↓ 反向放）。
EN = {
    "常驻 · 权重＋优化器": "Resident · weights+optimizer",
    "激活 · 前向堆、反向放": "Activations · ↑ fwd, ↓ bwd",
    "梯度 · 反向才长出来": "Gradients · bwd only",
    "常驻：一个 step 从头到尾都不动": "Resident: unchanged from start to end of a step",
    "权重 2 ＋ 优化器状态 12 ＝ 14 字节/参数　→　8.54 TiB":
        "weights 2 + optimizer state 12 = 14 bytes/param  →  8.54 TiB",
    "横轴 ＝ 一个 step 的时间　·　下面 61 格 ＝ 61 层，亮着的是此刻还挂着激活的层":
        "x-axis = time within one step  ·  61 cells below = 61 layers; lit = still holding activations",
    "前向：激活一层层攒起来": "Forward: activations accumulate layer by layer",
    "反向：激活逐层放掉，梯度一份份长出来": "Backward: activations freed layer by layer, gradients grow",
    "更新完：梯度也清空，只剩常驻那一块": "Updated: gradients cleared too, only the resident block remains",
    "下一个 step，从头再来一遍": "The next step, all over again",
    "每过一层就多挂一份中间结果　——　一份都不能提前扔，反向还要用它":
        "Each layer leaves one more intermediate result; none can go early: backward needs it",
    "红的在长，可它比蓝的小得多　——　所以总量从峰值那一刻起只降不升":
        "Red is growing, but much smaller than blue, so the total only falls from the peak",
    "回到 8.54 TiB　——　下一个 step 从这条灰带上重新堆":
        "Back to 8.54 TiB: the next step builds up again from this gray band",
    "训练就是把这座山堆起来、再拆掉　——　重复几十万次":
        "Training builds this mountain and tears it down, hundreds of thousands of times",
    "峰值就在这一刻　——　前向刚结束": "The peak is right here, just as forward ends",
}
PAT = []
run_en(__file__, EN, PAT, globals())
