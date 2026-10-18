# -*- coding: utf-8 -*-
"""topic05-anim-sp.py 的英文版：只换画面里的字，布局与时序原样（见 _lang_en.py）。
措辞跟 topic-05.en.html 里 Ulysses / USP 两图的 aria-label 一致；格子里的「段s · 头h」
用英文页静态图的写法 S s · H h（序列段 S、头 H），卡 → Dev，机器 → machine。

📌 渲染（在 Courses/ 下）：构建手册/脚本/render.sh \
        tools/manim/topic05-anim-sp-en.py UlyssesSteps WebPages/media/topic05-ulysses.en.mp4
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lang_en import run_en  # noqa: E402

EN = {
    "卡 0": "Dev 0", "卡 1": "Dev 1", "卡 2": "Dev 2", "卡 3": "Dev 3",
    "机器一": "machine 1", "机器二": "machine 2",
    "按段": "by chunk", "按头": "by head",
    # Ulysses
    "Ulysses：转置一下，每张卡拿一个头的全部段": "Ulysses: one transpose, then each device holds one whole head",
    "进来时按段分：卡 k 拿第 k 段的全部头（颜色 ＝ 哪一段）":
        "On entry, split by chunk: Dev k holds all heads of chunk k (color = which chunk)",
    "可头 0 的注意力要的是：头 0 的全部段 —— 现在散在四张卡上":
        "But head 0's attention needs all of head 0's chunks, now on four devices",
    "AllToAll，先看卡 0：头 0 留在本卡，头 1、2、3 分别发给卡 1、2、3":
        "AllToAll, starting with Dev 0: head 0 stays; heads 1, 2, 3 go to Devs 1, 2, 3",
    "卡 0 那一行发完了：留 1 格，发 3 格": "Dev 0's row is done: keeps 1 cell, sends 3",
    "其余三张卡同时照做：第 j 个头那一格，发给卡 j": "The other three do the same at once: the head-j cell goes to Dev j",
    "换完了：卡 j 拿头 j 的全部四段": "Done: Dev j holds all four chunks of head j",
    "这一列要看的全在本卡：注意力本地算完，一次都不用问别人":
        "Everything this column needs is local: attention finishes without asking anyone",
    "算完再 AllToAll 一次，换回按段分，接着做逐 token 的运算":
        "Then one more AllToAll switches back to chunks for the per-token ops",
    "代价：每层前后各一次 AllToAll；卡数不能超过头数":
        "Cost: an AllToAll before and after every layer; devices cannot exceed heads",
    # USP
    "USP：机器里做 Ulysses，机器之间走环": "USP: Ulysses inside the machine, a ring across machines",
    "4 张卡、2 台机器；进来时照样按段分，卡 k 拿第 k 段的全部头":
        "4 devices, 2 machines; on entry, split by chunk: Dev k holds all heads of chunk k",
    "第一步：只在机器里做 AllToAll（Ulysses 2）：卡 0、卡 1 互换头":
        "Step one: in-machine AllToAll only (Ulysses 2): Dev 0 and Dev 1 swap heads",
    "机器一换完：卡 0 拿段 0、1 的头 0、1，卡 1 拿段 0、1 的头 2、3":
        "Machine 1 done: Dev 0 has H0, H1 of S0, S1; Dev 1 has H2, H3 of S0, S1",
    "机器二同时照做，拿后两段": "Machine 2 does the same at once, with the last two chunks",
    "现在每张卡：两段 × 两个头 —— 头分开了，段还缺一半":
        "Each device now has 2 chunks × 2 heads: heads split, half the chunks missing",
    "卡 0 管头 0、1，可段 2、3 的笔记在机器二的卡 2 上":
        "Dev 0 owns heads 0, 1, but the notes for chunks 2, 3 sit on Dev 2 in machine 2",
    "第二步：机器之间走环（Ring 2）：卡 0↔卡 2、卡 1↔卡 3 互传笔记":
        "Step two: ring across machines (Ring 2): Dev 0↔Dev 2, Dev 1↔Dev 3 swap notes",
    "传过去的是抄件（半透明）：各自算完自己两段 × 两头的注意力，抄件就扔":
        "Copies travel (faded): each does its 2 chunks × 2 heads, then drops them",
    "算完，再在机器里 AllToAll 一次，换回按段分":
        "After that, one more AllToAll inside each machine restores the chunk split",
    "总卡数 ＝ Ulysses 2 × 环 2；头数上限只管机器里那一维":
        "Devices = Ulysses 2 × ring 2; the head cap binds only the in-machine dim",
}
PAT = [
    (r"段(\d) · 头(\d)", r"S\1 · H\2"),
]
run_en(__file__, EN, PAT, globals())
