# -*- coding: utf-8 -*-
r"""专题五 · 1.5 折叠段「TPU 的环面」：3D 环面上的 AllGather，逐轴走 vs 三块轮换。

⭐ 一张甘特图两半，同一把时间尺子：
   · 三行 ＝ X／Y／Z 三条轴；条的长度 ＝ 这条轴在这一步搬了多久。
   · 数据切成 A／B／C 三块（蓝 / 橙 / 绿），跟动画 topic05-torus 同色。
   · 左：逐轴。三块挤在同一条轴上排队，另外两条轴的格子是空的。
   · 右：轮换。路线 XYZ／YZX／ZXY，每一步三条轴各搬一块，**同样的色块，只是摆开了**。

⛔ 数据当场算（跟动画脚本同一套公式，各自断言同一组数）：
   4×4×4，每颗芯片起手 1/64，收完一条轴乘 4；沿长 4 的轴收集要收进手头 3 倍的量。
   逐轴三步 3 : 12 : 48，合计 63；轮换每轴 1 : 4 : 16，合计 21。单位：1/64 份数据过一条轴。
   ⚠️ 只按带宽算，不算延迟；XLA 实际怎么排没核对过。

⛔ 刻意没画：立方体本身（动画里有）、环回线、ReduceScatter／AllReduce（正文一句话：倒着走、各乘二）。
"""
from fractions import Fraction as Fr

from topic03_draw import Fig, BL, OR, GR, GY, GY2, INK, LINE

W = 1400
L = 4
AX = "XYZ"
COL = [BL, OR, GR]
NAME = "ABC"
SEQ = ["XYZ", "XYZ", "XYZ"]
ROT = ["XYZ", "YZX", "ZXY"]


def segments(routes):
    """返回 {轴: [(起, 止, 块号)]}，单位 1/64。同一轴上同一步的几块按块号先后排队。"""
    st = [Fr(L ** i, L ** 3) for i in range(4)]
    out = {a: [] for a in AX}
    t0 = Fr(0)
    for p in range(3):
        cur = {a: t0 for a in AX}
        for i, r in enumerate(routes):
            d = Fr(1, len(routes)) * st[p] * (L - 1) * 64
            out[r[p]].append((cur[r[p]], cur[r[p]] + d, i))
            cur[r[p]] += d
        t0 = max(cur.values())
    return out, t0


SEG_SEQ, T_SEQ = segments(SEQ)
SEG_ROT, T_ROT = segments(ROT)
assert T_SEQ == 63 and T_ROT == 21 and T_SEQ == 3 * T_ROT
assert [b for _, b, _ in SEG_SEQ["X"]][-1] == 3 and [b for _, b, _ in SEG_SEQ["Z"]][-1] == 63
assert SEG_SEQ["Y"][0][0] == 3 and SEG_SEQ["Z"][0][0] == 15          # 逐轴：前一条轴走完才开始
assert all(SEG_ROT[a][0][0] == 0 and SEG_ROT[a][-1][1] == 21 for a in AX)   # 轮换：三条轴从头忙到尾
BOUND_SEQ, BOUND_ROT = [0, 3, 15, 63], [0, 1, 5, 21]

PW, GAP = 680, 40
X0, SC = 64, 560 / 63            # 两半同一把尺子：63 个单位 ＝ 560 px
RH, RG = 38, 16


def gantt(f, px, y0, title, col, segs, bounds, total, note):
    ph = 30 + 34 + 3 * (RH + RG) + 64
    py = f.panel(px, y0, PW, ph, title, col)
    top = py + 34
    for b in bounds:
        x = px + X0 + float(b) * SC
        f.line(x, top - 8, x, top + 3 * (RH + RG) - RG + 8, LINE, 1, dash="3,3", arrow=False)
        if b:                    # 0 不标：轮换那半 0 和 1 只差 9 px，会叠在一起
            f.t(x, top + 3 * (RH + RG) - RG + 26, str(b), GY, size=12, anchor="middle")
    for k, a in enumerate(AX):
        y = top + k * (RH + RG)
        f.t(px + 22, y + RH / 2 + 6, a + " 轴", INK, True, 14)
        busy = segs[a]
        for s0, s1, i in busy:
            f.box(px + X0 + float(s0) * SC, y, max(float(s1 - s0) * SC, 2), RH, COL[i], "#fff", 2, sw=1)
        # 空着的那几段：虚线框 ＋「闲」
        gaps, t = [], Fr(0)
        for s0, s1, _ in sorted(busy):
            if s0 > t:
                gaps.append((t, s0))
            t = max(t, s1)
        if t < total:
            gaps.append((t, total))
        for g0, g1 in gaps:
            w = float(g1 - g0) * SC
            f.box(px + X0 + float(g0) * SC, y, w, RH, "none", GY2, 2, sw=1, dash="4,3")
            if w > 40:
                f.t(px + X0 + float(g0) * SC + w / 2, y + RH / 2 + 5, "闲", GY, size=13, anchor="middle")
    xe = px + X0 + float(total) * SC
    f.t(xe + 8, top - 10, "用时 %s" % total, col, True, 15)
    f.t(px + 22, py + ph - 52 + 14, note, col, True, 14)
    f._pan = None
    return py + ph - 30


def fig_torus():
    f = Fig(W, "3D 环面上的 AllGather，两种走法的甘特图，同一把时间尺子。4×4×4 切片，数据切成 A、B、C 三块。"
               "左边逐轴：三块都先走 X、再走 Y、最后走 Z，每一步只有一条轴在忙，另外两条闲着；"
               "三步用时 3、12、48，合计 63，最后一步占了大半。"
               "右边轮换：A 走 X Y Z、B 走 Y Z X、C 走 Z X Y，每一步三条轴各搬一块，三条轴从头忙到尾；"
               "每轴三步用时 1、4、16，合计 21，正好是逐轴的三分之一")
    y0 = f.header("3D 环面上的 AllGather　——　<tspan font-weight=\"700\">三块数据错开路线，三条轴一起忙</tspan>",
                  "4×4×4 切片，每颗芯片起手 1/64。条长 ＝ 这条轴搬了多久（单位：1/64 份数据过一条轴），两边同一把尺子",
                  [(BL, "块 A"), (OR, "块 B"), (GR, "块 C")])
    yl = gantt(f, 0, y0, "逐轴：X 走完才走 Y、再走 Z", GY, SEG_SEQ, BOUND_SEQ, T_SEQ,
               "每一步只有一条轴在忙；第 3 步一步就占了 48/63")
    yr = gantt(f, PW + GAP, y0, "轮换：A 走 XYZ、B 走 YZX、C 走 ZXY", GR, SEG_ROT, BOUND_ROT, T_ROT,
               "同样的色块摆开了：三条轴从头忙到尾，用时 1/3")
    yb = f.band(max(yl, yr) + 20, "ok", "数据一样多，差的是有几条轴同时在干活", [
        "X、Y、Z 各是一个环：逐轴走，每一步只用上一个方向；轮换走，三个方向一直满着，带宽翻三倍。",
        "ReduceScatter 倒着走（大的一步在前），AllReduce ＝ 两者接起来；只按带宽算，同样快三倍。",
    ])
    yb = f.src(yb + 10, "📌 多轴收集带宽成倍：How to Scale Your Model（scaling book）sharding 章（中文版：沿多个轴收集时，可用带宽成倍增加）。",
               "⚠️ 本课推导：只按带宽算、不算每步延迟；三条轴带宽相同、每一维都是首尾相接的环（切片每一维是 4 的整倍数）。XLA 实际怎么切块、怎么排，没有核对过。")
    f.save("fig5-torus-3ring.svg", yb + 14)


fig_torus()
