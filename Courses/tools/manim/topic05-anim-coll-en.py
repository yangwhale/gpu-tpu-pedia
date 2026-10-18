# -*- coding: utf-8 -*-
"""topic05-anim-coll.py 的英文版：只换画面里的字，布局与时序原样（见 _lang_en.py）。
措辞跟 topic-05.en.html 里各图的 aria-label / 图注一致（卡 → Dev，与静态图同）。

📌 渲染（在 Courses/ 下）：构建手册/脚本/render.sh \
        tools/manim/topic05-anim-coll-en.py Ring WebPages/media/topic05-ring.en.mp4
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lang_en import run_en  # noqa: E402

EN = {
    # 标题
    "AllReduce 全归约：连成一个环，人人只往右边的邻居发": "AllReduce: link into a ring, everyone sends only to the right",
    "ReduceScatter　归约分散：人人一整份 → 各拿一块总和": "ReduceScatter: a full copy each → one block of the sum each",
    "AllGather　全收集：一人一块 → 人人一整份": "AllGather: a block each → a full copy for all",
    "AllToAll　全交换：每人给每人寄一份，多少不一样": "AllToAll: everyone mails everyone a piece, in different sizes",
    "Broadcast　广播：一份 → 人人一份": "Broadcast: one copy → a copy for all",
    "Scatter　分发：一份拆开 → 一人一块": "Scatter: one copy split → a block each",
    "Gather　收集：一人一块 → 拼成一份": "Gather: a block each → one copy",
    "Reduce　归约：人人一份 → 加成一份": "Reduce: a copy each → one sum",
    # 卡名、车道
    "卡 0": "Dev 0", "卡 1": "Dev 1", "卡 2": "Dev 2", "卡 3": "Dev 3",
    "卡 3 → 卡 0：绕回来": "Dev 3 → Dev 0: wraps around",
    # 环
    "ReduceScatter：每一步人人同时往右发一块，收到的加到自己那块上":
        "ReduceScatter: each step, everyone sends a block right; the receiver adds it",
    "三步之后：每张卡恰好握着一块完整总和 Σ": "After three steps: each device holds exactly one complete sum Σ",
    "其余几块是半路上的中间结果，不要了": "The other blocks are partial results along the way: discarded",
    "AllGather：把总和接着往右传，只替换，不相加": "AllGather: keep passing the sums right; replace, don't add",
    "AllGather：把自己那块接着往右传，只替换，不相加": "AllGather: keep passing your block right; replace, don't add",
    "第 1 步（共 3 步）：人人把自己那块总和传给右边，卡 3 绕回卡 0":
        "Step 1 of 3: everyone passes its sum block right; Dev 3 wraps to Dev 0",
    "第 1 步（共 3 步）：人人把自己那块传给右边，卡 3 绕回卡 0":
        "Step 1 of 3: everyone passes its own block right; Dev 3 wraps to Dev 0",
    "三步之后：人人一整份，只拼，不加": "After three steps: a full copy each; concatenated, never summed",
    "人人一份总和 ＝ AllReduce ＝ ReduceScatter ＋ AllGather":
        "The full sum for everyone = AllReduce = ReduceScatter + AllGather",
    # AllToAll
    "出发": "Out", "收到": "In",          # 框内左上角的小标签，宽度只够一个短词
    "每个 token 已由路由定好去哪张卡（格子里的数字），每人要寄给每人的数目都不一样":
        "The router set each token's device (the number); the send counts all differ",
    "派发：所有卡同时往所有方向寄，一次飞完，不排队、不接力":
        "Dispatch: all send every way at once, in one flight: no queue, no relay",
    "收到的忙闲不均：卡 0 收了 11 个，卡 3 只收 6 个": "Unevenly loaded: Dev 0 gets 11, Dev 3 only 6",
    "各卡上的专家算自己收到的（描黄边）；卡 0 最忙，大家都等它":
        "Experts compute what they got (yellow edge); Dev 0 is busiest, all wait for it",
    "合并：又一次 AllToAll，沿原路飞回，回到出发的卡、原来的位置":
        "Combine: another AllToAll, back along the same lines to the original slot",
    "MoE 每层两次：派发一次、合并一次；每份一样大时，才正好是一次转置":
        "MoE does this twice per layer; only equal sizes make it a transpose",
    # 原语
    "卡 0 把整份数据复制给每一个人": "Dev 0's whole copy, duplicated to everyone",
    "卡 0 把第 j 块发给卡 j，自己只留第 0 块": "Dev 0 sends block j to Dev j and keeps only block 0",
    "每人把自己那块交给卡 0，卡 0 按顺序拼起来": "Everyone hands its block to Dev 0, which concatenates them in order",
    "每人把整份交给卡 0，卡 0 逐块相加": "Everyone hands its whole copy to Dev 0, which sums them block by block",
}
PAT = [
    (r"第 (\d) 步（共 3 步）：0→1、1→2、2→3，卡 3 从右边绕回卡 0",
     r"Step \1 of 3: 0→1, 1→2, 2→3; Dev 3 wraps from the right to Dev 0"),
    (r"第 (\d) 步完成：每张卡的粗框那块，又多加进了一个人",
     r"Step \1 done: each bold-outlined block now includes one more device"),
    (r"第 (\d) 步（共 3 步）：人人把上一步刚收到的那块传给右边，卡 3 绕回卡 0",
     r"Step \1 of 3: everyone passes the block it just got right; Dev 3 wraps to Dev 0"),
    (r"第 (\d) 步完成：每张卡又多了一块", r"Step \1 done: every device has one more block"),
]
run_en(__file__, EN, PAT, globals())


# ⭐ 唯一一处版面改动：环示意里卡名两侧的箭头 / 车道各缩进 0.13。
#   中文「卡 0」宽 0.7，箭头端点离卡名中线 0.55 正合适；英文「Dev 0」宽 1.05，
#   不缩的话箭头尖直接戳在字上。其余照抄中文脚本的 ring_guide。
_GAP = 0.68


def ring_guide(xr=None, xl=None):  # noqa: F811
    xr, xl = xr or XR, xl or XL
    g = VGroup()
    for k in range(N - 1):
        g.add(Arrow([XS[k] + _GAP, YLAB, 0], [XS[k + 1] - _GAP, YLAB, 0],
                    color=GREY, stroke_width=3, buff=0, max_tip_length_to_length_ratio=0.12))
    lane = VMobject(stroke_color=GREY, stroke_width=2.5)
    lane.set_points_as_corners([[XS[-1] + _GAP, YLAB, 0], [xr, YLAB, 0], [xr, LANE, 0],
                                [xl, LANE, 0], [xl, YLAB, 0], [XS[0] - _GAP, YLAB, 0]])
    g.add(DashedVMobject(lane, num_dashes=70))
    g.add(Arrow([xl, YLAB, 0], [XS[0] - _GAP + 0.05, YLAB, 0], color=GREY, stroke_width=3, buff=0,
                max_tip_length_to_length_ratio=0.5))
    g.add(Text("卡 3 → 卡 0：绕回来", font_size=20, color=GREY).move_to([0, LANE - 0.28, 0]))
    return g
