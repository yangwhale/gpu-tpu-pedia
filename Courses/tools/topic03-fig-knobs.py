# -*- coding: utf-8 -*-
r"""专题三 · §四「三个旋钮」骨架图（2026-09-12 加）。

⛔ 为什么这一节最该有图：讲义写着「**不许压，一压后面全散**」——&nbsp;
   它是全课骨架。而它原来 **0 张图**，只有两张表。

⭐ 这张图要做的**不是罗列三个旋钮**，是**证明为什么恰好是三个**：
   把「一个 query 要做的三步」摆在左边，每一步**正对**一个旋钮。
   看完应该得到的是「**没有第四个位置可以动**」这个封闭感，
   而不是「哦，有三类方法」。

📌 FlashAttention 画成灰色旁支：**它不改算什么，只改怎么算** ——&nbsp;
   所以它不在这三条里。这一条 2026-09-08 现场特意降过级。
"""
from topic03_draw import (Fig, wpx, _sz,
                          BL, OR, GR, RD, GY, PU, CY, BR, INK,
                          GY2, LINE, LINE2, BG2)

W = 1400

STEPS = [
    ("①", "从每个位置取出<tspan font-weight=\"700\">一份 K、一份 V</tspan>",
     "存什么、存多大", BL,
     "旋钮 ① 每个 token 存多少",
     "减少 KV 的<tspan font-weight=\"700\">份数</tspan>或<tspan font-weight=\"700\">维度</tspan>",
     "MQA → GQA → MLA → Gated MLA"),
    ("②", "跟<tspan font-weight=\"700\">哪些位置</tspan>算，然后加权求和",
     "求和的范围有多大", OR,
     "旋钮 ② 每个 query 看多少",
     "限制<tspan font-weight=\"700\">范围</tspan>或<tspan font-weight=\"700\">动态挑选</tspan>",
     "SWA · NSA · DSA · CSA/HCA"),
    ("③", "用 <tspan font-weight=\"700\">softmax(qKᵀ/√d)</tspan> 做这个加权",
     "用哪种运算", PU,
     "旋钮 ③ 换一套数学",
     "用<tspan font-weight=\"700\">固定大小的状态</tspan>代替不断变长的 KV",
     "线性注意力：DeltaNet → GDN → KDA"),
]


def main():
    f = Fig(W, "三个旋钮不是凑出来的：一个 query 只做三步 —— 取出 K/V、"
               "决定跟哪些位置算、用什么运算做加权；三个旋钮正好各对一步。"
               "FlashAttention 不改算什么只改怎么算，所以不在其中")
    f.marks = set()
    y = f.header(
        '为什么恰好是三个旋钮 ——&#160;'
        '<tspan font-weight="700">因为一个 query 只做三步，一步一个位置可动</tspan>',
        '⛔ 这张图要留下的不是「有三类方法」，是 '
        # ⛔ 2026-09-12：这行副标题第一版写的是 markdown 的星号包粗体，
        #   **SVG 不认 markdown**，于是那两个星号原样印在图上。改用 tspan。
        #   ⚠️ 第二版我把这条说明**写进了副标题本身**，于是它也印上去了 ——
        #   ⭐ 判据（今天第三次撞上同一形状）：**给自己的话写注释，不写进产物。**
        '「<tspan font-weight="700">没有第四个位置可以动</tspan>」')

    # ⛔ 左栏 470 时第三步那行 418px 顶出去（基元自带宽度断言，当场报了）。
    #   ⭐ 这套图的护栏很好用：**文字溢出在 SVG 里不报错只被裁掉**，靠断言兜住。
    LX, LW = 0, 500          # 左：三步
    KX, KW = 548, 672        # 右：三个旋钮
    ROW, TOP = 104, y + 30

    f.colhead(LX, y + 10, "一个 query 要做的三步", "回到注意力那个式子，只有这三步")
    f.colhead(KX, y + 10, "每一步各自可以动的地方", "这就是三个旋钮的由来")

    for i, (no, what, lever, col, kname, kdo, krep) in enumerate(STEPS):
        yy = TOP + i * ROW
        # 左：步骤
        f.box(LX, yy, LW, 76, "#fff", LINE, 8)
        f.box(LX, yy, 4, 76, col, col, 2)
        f.t(LX + 20, yy + 28, no, col, bold=True, size=16, cls="svglbl")
        f.t(LX + 48, yy + 28, what, INK, size=_sz(12.5), w=LW - 64)
        f.t(LX + 48, yy + 54, "可动的是：" + lever, GY2, size=11)
        # 箭头
        f.line(LX + LW + 8, yy + 38, KX - 8, yy + 38, col, 1.8)
        # 右：旋钮卡
        f.box(KX, yy, KW, 76, "#fff", LINE, 8)
        f.box(KX, yy, 4, 76, col, col, 2)
        f.t(KX + 20, yy + 26, kname, col, bold=True, size=_sz(13.5), cls="svglbl")
        f.t(KX + 20, yy + 48, kdo, INK, size=_sz(12))
        f.t(KX + 20, yy + 68, krep, GY2, size=11, mono=True)

    yy = TOP + 3 * ROW + 4

    # ①+② 可叠加、③ 换赛道
    f.box(KX, yy, KW, 40, "#f8f9fa", LINE, 8)
    f.t(KX + 20, yy + 25,
        '⭐ <tspan font-weight="700">① 和 ② 可以叠加</tspan>'
        '（一个让每份更小、一个让读的份数更少，省的是同一样东西的两面）；'
        '<tspan font-weight="700">③ 是换赛道</tspan>。', INK, size=_sz(12))

    # FlashAttention 灰色旁支
    f.box(LX, yy, LW, 40, "#fff", LINE, 8, dash="5 4")
    f.t(LX + 20, yy + 25,
        '⛔ <tspan font-weight="700">FlashAttention 不在这三条里</tspan>'
        '——&#160;它<tspan font-weight="700">不改算什么，只改怎么算</tspan>。',
        GY2, size=_sz(12))

    yy += 56
    yy = f.band(yy, "info", "所以它是封闭的 ——&#160;没有第四个位置可以动", [
        '这三步<tspan font-weight="700">穷尽了一个 query 能做的事</tspan>：'
        '取什么、跟谁算、用什么运算。<tspan font-weight="700">你想省 KV，只能从这三处下手。</tspan>',
        '⭐ 唯一的例外是<tspan font-weight="700">「不改算什么、只改怎么算」</tspan>'
        '——&#160;那是 FlashAttention 那一路，<tspan font-weight="700">它在哪种注意力下都一样，'
        '对这一讲没有区分度</tspan>。',
        '⭐⭐ <tspan font-weight="700">①＋②＋③ 混着来 ＝ Hybrid</tspan>：不同层用不同方案 '
        '——&#160;V4 的 CSA+HCA、K3 的 KDA+Gated MLA。<tspan font-weight="700">'
        '这是今天各家的实际答案，不是折中。</tspan>'])

    yy = f.src(yy + 16,
               "三步的拆法直接对应 Vaswani 2017 §3.2 的定义式："
               "取 key-value pairs →&#160;按 compatibility 加权 →&#160;用 softmax 做这个加权",
               "⚠️ 图上每个旋钮只列代表方案，<tspan font-weight=\"700\">谁在哪一格、出处是什么"
               "，见本节那张名词收纳表</tspan>")
    f.save("fig3-knobs.svg", yy + 6)


main()
