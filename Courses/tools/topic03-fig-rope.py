# -*- coding: utf-8 -*-
r"""专题三 · §五「RoPE：把位置变成一个转角」

⭐⭐⭐ 2026-09-13 新画。本课原来对 RoPE 只有一个「寄快递」的比喻，
   而那个比喻回答的是「为什么 MLA 吸收不了它」——&nbsp;
   ⛔ **「把位置变成旋转角，为什么点积就自动带上了相对距离」
     这个最基础的问题，我们一张图都没有。**

📌 装置来自两篇顶级材料**各一半，而且没人把它们拼起来**：
   · Fleetwood（huggingface.co/blog/designing-positional-encoding）
     画了**二进制计数器动画**（低位飞快翻、高位几乎不动），
     用来说明「正弦是二进制计数器的连续版」——&nbsp;停在这儿了。
   · 苏剑林（kexue.fm/archives/9675）走到了
     **「RoPE ＝ 位置的 β 进制写法」**，于是外推 / 内插 / NTK 一句话各自归位
     ——&nbsp;⛔ 但他全文是公式，**一张图没画**。
   ⭐ 这张图把两半拼起来：**一排里程表转盘。**

📐 那条等式本课自己推了一遍，不靠转述（脚本里断言）：
   RoPE 第 m 对维度的角速度 θ_m ＝ 10000^(-2(m-1)/d) ＝ 1/β^(m-1)，
   其中 β ＝ 10000^(2/d)。于是 n·θ_m ＝ n / β^(m-1) ——&nbsp;
   **跟 β 进制取第 m 位时的除数完全一样。** d=128 → 64 对维度 ＝ 64 位数。
"""
import math

from topic03_draw import (Fig, BL, OR, GR, RD, GY, PU, INK, GY2, LINE, LINE2,
                          BG2)

W = 1400


def main():
    d = 128
    beta = 10000 ** (2.0 / d)
    for m in range(1, 8):                      # ⛔ 当场验，别信转述
        assert abs(10000 ** (-2.0 * (m - 1) / d) - 1.0 / beta ** (m - 1)) < 1e-12
    assert d // 2 == 64

    f = Fig(W, "RoPE 就是把位置写成一排里程表转盘：最右边转得飞快，"
               "往左逐级变慢；两个位置各自转完之后，每个转盘上的夹角只跟"
               "它们的距离有关，跟各自在哪完全无关")
    f.marks = set()
    y0 = f.header(
        "RoPE ——　<tspan font-weight=\"700\">把位置写成一排里程表转盘</tspan>",
        "⭐ 这张图回答两件事：<tspan font-weight=\"700\">"
        "为什么点积自动带上了相对距离</tspan>，"
        "以及<tspan font-weight=\"700\">凭什么能外推到 1M</tspan>",
        [(BL, "位置 m"), (OR, "位置 n"), (PU, "夹角＝m−n"), (GR, "三种改法")])

    def dial(cx, cy, r, ang, col, tint="#fff", sw=2.0):
        f.p.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s" '
                   'stroke="%s" stroke-width="1.6"/>' % (cx, cy, r, tint, GY2))
        f.line(cx, cy, cx + r * 0.78 * math.sin(ang),
               cy - r * 0.78 * math.cos(ang), col, sw, arrow=False)

    # ══════════ ① 一排转盘 ══════════════════════════════════════
    PH = 324
    py = f.panel(0, y0, W, PH, "① 一个位置 ＝ 一排转盘的读数", INK,
                 sub="最右边转得飞快，往左逐级变慢 ——　跟里程表一模一样")
    ay = py + 34
    NDIAL = 7
    R = 34
    for k in range(NDIAL):
        cx = 130 + k * 170
        speed = 1.0 / beta ** (k * 9)          # 每隔 9 对取一个，好看出快慢
        dial(cx, ay + 54, R, 3 * speed, BL)
        f.t(cx, ay + 116, "第 %d 对" % (k * 9 + 1), GY2, size=15, anchor="middle")
        f.t(cx, ay + 140, "转速 1/β^%d" % (k * 9), GY2, size=14, anchor="middle")
    f.t(130 - R - 8, ay + 172, "← 快（每个位置都转一大格）", GY, size=17)
    f.t(130 + 6 * 170 + R + 8, ay + 172, "慢（几万个位置才转一圈）→", GY,
        size=17, anchor="end")
    f.box(56, ay + 192, 1288, 80, "#e8f0fe", BL, 10)
    f.t(80, ay + 222, "⭐ 这排转盘就是 <tspan font-weight=\"700\">位置 n 的 "
        "β 进制写法</tspan>（β ＝ 10000^(2/d) ≈ %.3f）" % beta, GY, size=17,
        w=1240)
    f.t(80, ay + 250, "d=128 就是 <tspan font-weight=\"700\">64 位数</tspan>；"
        "第 m 位的除数 β^(m-1)，<tspan font-weight=\"700\">正是第 m 对维度的"
        "转速</tspan>。", GY, size=17, w=1240)

    # ══════════ ② 为什么点积只认距离 ════════════════════════════
    y1 = y0 + PH + 18
    PH2 = 322
    py2 = f.panel(0, y1, W, PH2,
                  "② 为什么点积自动带上了相对距离 ——　看夹角就行", PU,
                  sub="两排转盘叠起来，每个盘上的夹角都是 (m−n)×转速")
    by = py2 + 40
    for k in range(5):
        cx = 180 + k * 210
        sp = 1.0 / beta ** (k * 12)
        am, an = 12 * sp, 5 * sp
        f.p.append('<circle cx="%.1f" cy="%.1f" r="48" fill="#fff" '
                   'stroke="%s" stroke-width="1.6"/>' % (cx, by + 56, GY2))
        f.line(cx, by + 56, cx + 38 * math.sin(am), by + 56 - 38 * math.cos(am),
               BL, 2.6, arrow=False)
        f.line(cx, by + 56, cx + 38 * math.sin(an), by + 56 - 38 * math.cos(an),
               OR, 2.6, arrow=False)
        f.t(cx, by + 126, "夹角 ＝ (m−n)×转速", PU, True, 15, "middle")
    f.t(180 - 58, by - 10, "蓝＝位置 m 转过的角　·　橙＝位置 n 转过的角", GY,
        size=17)
    f.box(56, by + 146, 1288, 96, "#f3e8fd", PU, 10)
    f.t(80, by + 178, "⭐⭐ 每个盘上你只看得出<tspan font-weight=\"700\">"
        "两根针差多少</tspan> ——　看不出各自转到了哪儿。", PU, True, 17)
    f.t(80, by + 208, "而点积 <tspan font-weight=\"700\">a·b ＝ |a||b|cos θ</tspan>"
        " 只吃夹角和长度：同转一个角，两样都没变。", GY, size=17, w=1240)
    f.t(80, by + 234, "⭐ 所以绝对位置被转掉了，<tspan font-weight=\"700\">"
        "留下来的只有 m−n</tspan>。", GY, size=17, w=1240)

    # ══════════ ③ 长文本三种改法 ════════════════════════════════
    y2 = y1 + PH2 + 18
    PH3 = 340
    py3 = f.panel(0, y2, W, PH3,
                  "③ 于是长文本那三种做法，一句话各自归位", GR,
                  sub="同一排转盘，三种改法")
    ey = py3 + 30
    WAYS = [
        (RD, "直接外推", "转盘一格不改", "硬往超出刻度的地方读",
         "⛔ 最慢那个盘从没转到过那儿，模型没见过", "none"),
        (OR, "位置内插（PI）", "每格改成走半格", "整排一起放慢",
         "⛔ 最快那个盘现在分不清相邻两个位置了", "slow"),
        (GR, "NTK-aware", "换一个进制", "β 变大：快盘几乎不动，慢盘明显变慢",
         "⭐ 一句话：高频外推、低频内插", "base"),
    ]
    for i, (col, name, a, b_, note, kind) in enumerate(WAYS):
        x = 30 + i * 452
        f.box(x, ey, 428, 268, "#fff", col, 10)
        f.box(x, ey, 428, 5, col, col, 3)
        f.box(x, ey + 3, 428, 5, "#fff", "#fff", 0)
        f.t(x + 22, ey + 42, name, col, True, 23)
        f.t(x + 22, ey + 70, a, GY, size=17)
        for k in range(4):
            cx = x + 70 + k * 90
            base = 1.0 / beta ** (k * 16)
            sp = base if kind == "none" else (base * 0.5 if kind == "slow"
                                              else base ** 1.25)
            dial(cx, ey + 130, 28, 9 * sp, col)
        f.t(x + 22, ey + 184, b_, col, True, 17, w=384)
        f.box(x + 22, ey + 200, 384, 52, BG2, LINE2, 8)
        f.t(x + 36, ey + 230, note, GY, size=16, w=356)

    # ══════════ 落点 ════════════════════════════════════════════
    yy = y2 + PH3 + 20
    yy = f.band(yy, "info", "⭐⭐ 这张图跟「寄快递」不冲突，是同一件事的两个切面", [
        "<tspan font-weight=\"700\">寄快递</tspan>回答的是"
        "「<tspan font-weight=\"700\">为什么 MLA 吸收不了它</tspan>」——&#160;"
        "那个旋转矩阵夹在上投影和隐向量中间，拆不开。",
        "<tspan font-weight=\"700\">里程表</tspan>回答的是"
        "「<tspan font-weight=\"700\">为什么点积自动带相对距离</tspan>」和"
        "「<tspan font-weight=\"700\">凭什么能外推</tspan>」。",
        "⭐ 两个一起看，<a href=\"#s五\">§五</a>那条 decoupled RoPE "
        "（让带位置的那几维单独走一路）就不是一个补丁，而是唯一的出路。",
    ])
    yy = f.src(yy + 16,
               "「RoPE ＝ 位置的 β 进制写法」以及外推／内插／NTK 的统一解释，"
               "出自苏剑林 kexue.fm/archives/9675（⚠️ 原文是推导，没有图）",
               "「二进制计数器 → 正弦」的动画装置出自 Fleetwood "
               "huggingface.co/blog/designing-positional-encoding；"
               "「点积只吃夹角和长度」出自 EleutherAI 的 RoPE 博客",
               "📐 θ_m ＝ 10000^(−2(m−1)/d) ＝ 1/β^(m−1) 这条等式"
               "<tspan font-weight=\"700\">由本脚本当场验证并断言</tspan>，不是转述；"
               "RoPE 原始出处 RoFormer arXiv 2104.09864")
    f.save("fig3-rope.svg", yy + 6)


main()
