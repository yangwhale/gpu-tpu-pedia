# -*- coding: utf-8 -*-
r"""专题九 · 开篇总览：一张阶梯曲线看全程（2026-11-22）。

fig8b-curve   回答：随着一项项改动推进，步时一步一步跳了多少？
              横轴一站一格（时间顺序），纵轴步时（对数轴：同样的百分比 ＝ 同样的高度）。
              两次换底座处断开、画灰带：同一配置在新旧底座各量一次，用虚线连起来——高低不能跨灰带比。
              曲线只负责「一级一级往下跳、每级多少」；绝对终点看页首。同一底座首尾直接比（5.80 倍）挪到第十一节（fig8b-mile）。

刻意没画：
  · 右轴有效 MFU —— 有效分子在「换预打包数据」「滤掉长文档」「换长文」三处都变了，一根右轴会让人跨段比 MFU。
  · 「累计加速倍数」那种把三段串乘起来的纵轴 —— 串乘得约 9 倍（N.CURVE_CHAIN_X），同一底座实测只有 5.80 倍：
    换底座以后同一刀的收益不一样大（第一段在长文上是 4.62 倍不是 4.67 倍，第二段 1.11 倍不是 1.35 倍），串乘会把它们当成可以叠。
  · 每一站的名字 —— 只给大跳（≥5%）起白话名；每一站的全名在编年史那两张图里。
数字全部来自 topic08b_numbers（FRONT / HALF / LONG / CURVE_* / MILE20）。
"""
import math
import topic08_figfont  # noqa: F401
from topic03_draw import Fig, BL, OR, GR, GY, INK, GY2, LINE, PU
Fig.BAND_KEEP_LINES = 0
import topic08b_numbers as N

W = 1400
SEGC = {1: BL, 2: OR, 3: PU}
REJC = GY2
YMAX, YMIN = 14.0, 2.1


def fig_curve():
    st = [(s, n, t, m, p, 1) for s, n, t, m, p, _k in N.FRONT] + \
         [(s, n, t, m, p, 2) for s, n, t, m, p, _k in N.HALF] + \
         [(s, n, t, m, p, 3) for s, n, t, m, p, _k in N.LONG]
    post = N.CURVE_POST
    aria = ("开篇总览，一条阶梯曲线。第一段 20 层底座：原样 %.3f 秒，两刀砍掉一大半（−45.4%%、−29.0%%），到第 12 站 %.3f 秒，段内 %.2f 倍。"
            "换底座（20 层换半层）后同一配置是 %.3f 秒；第二段短文到第 16 站 %.3f 秒，段内 %.2f 倍。换长文后同一配置是 %.3f 秒；"
            "第三段长文到第 27 站定稿 %.3f 秒，后面每站只抠 1%% 上下，最后三站是精度修正；定稿之后再三步，%.3f 秒，有效 MFU %.2f%%（30 步口径；200 步长跑也过了 25%%）。"
            "灰色叉是试过、变慢、没采用的三次尝试：%s。"
            % (N.FRONT_BASE_S, N.FRONT_END_S, N.CURVE_SEG_X[0], N.HALF_START, N.HALF[-1][2], N.CURVE_SEG_X[1], N.LONG_START,
               N.END_S, N.CURVE_END_S, N.CURVE_END_MFU, "、".join("%s +%.1f%%" % (r[1], r[3]) for r in N.CURVE_REJ)))
    f = Fig(W, aria)
    y0 = f.header("一张图看全程：一站一站往下抠，每级多少",
                  "每级台阶是一站的步时（纵轴：秒，取对数，同样的百分比台阶一样高）。灰带是换测试底座或换数据，同一配置在两边各量一次（虚线），不是变慢，高低不能跨灰带比",
                  [(BL, "20 层底座"), (OR, "半层底座 · 短文"), (PU, "半层底座 · 长文"), (GR, "◆ 改了精度或舍入写法"), (REJC, "✕ 试过、变慢、没采用")])
    PH = 530
    py = f.panel(0, y0, W, PH, "64 片 v7x，实测（每段从左往右看）", LINE)
    X0, X1 = 70, W - 30
    TOP, BOT = py + 70, py + PH - 64
    lg = math.log(YMAX / YMIN)

    def Y(t):
        return TOP + math.log(YMAX / t) / lg * (BOT - TOP)

    # 横向格位：一站一格，换底座处占 2.2 格，定稿之后前面空 0.6 格
    slots, x = {}, 0.0
    for s, *_r in st:
        if s in (13, 17):
            slots["b%d" % s] = x + 1.1
            x += 2.2
        slots[s] = x
        x += 1
    x += 0.6
    slots["p0"], slots["p1"], slots["p2"] = x, x + 1, x + 2
    U = (X1 - X0) / (x + len(N.CURVE_POST) - 0.4)
    SX = {k: X0 + 0.5 * U + v * U for k, v in slots.items()}
    HW = 0.38 * U

    # 纵轴刻度
    for v in (2.5, 3, 4, 5, 6, 8, 10, 12):
        f.line(X0, Y(v), X1, Y(v), LINE, 0.8, arrow=False)
        f.t(X0 - 8, Y(v) + 4, ("%g" % v), GY, size=12, anchor="end")

    # 换底座灰带
    for b, top_lab, bot_lab, t_new, prev in ((13, "换底座", "20 层→半层", N.HALF_START, 12),
                                             (17, "换数据", "短文→长文", N.LONG_START, 16)):
        bx = SX["b%d" % b]
        f.box(bx - 0.85 * U, TOP - 6, 1.7 * U, BOT - TOP + 12, "#f1f3f4", "#f1f3f4", 4)
        f.t(bx, BOT - 26, top_lab, GY, True, 13, "middle")
        f.t(bx, BOT - 8, bot_lab, GY, size=12, anchor="middle")
        tp = [r for r in st if r[0] == prev][0][2]
        f.line(SX[prev] + HW, Y(tp), bx, Y(t_new), GY2, 1.3, dash="4 3", arrow=False)
        f.box(bx - 5, Y(t_new) - 5, 10, 10, "#fff", GY2, 5, sw=1.5)
        f.t(bx, Y(t_new) - 12, "%.3f" % t_new, GY, size=11.5, anchor="middle")

    # 三段台阶
    def stair(pts, col):
        for i, (xx, t) in enumerate(pts):
            f.line(xx - HW, Y(t), xx + HW, Y(t), col, 3.2, arrow=False)
            if i:
                xp, tp = pts[i - 1]
                f.line(xp + HW, Y(tp), xx - HW, Y(tp), col, 1, arrow=False)
                f.line(xx - HW, Y(tp), xx - HW, Y(t), col, 1.8, arrow=False)

    segpts = {1: [], 2: [(SX["b13"], N.HALF_START)], 3: [(SX["b17"], N.LONG_START)]}
    for s, n, t, m, p, seg in st:
        segpts[seg].append((SX[s], t))
    segpts[3] += [(SX["p%d" % i], r[1]) for i, r in enumerate(post)]
    for seg, pts in segpts.items():
        stair(pts, SEGC[seg])

    # 段头：名字 ＋ 段内倍数
    for seg, xx, name in ((1, SX[0] - HW, "第一段 · 20 层底座"), (2, SX["b13"] - 0.8 * U, "第二段 · 半层 · 短文"),
                          (3, SX["b17"] - 0.8 * U, "第三段 · 半层 · 长文")):
        f.t(xx, py + 26 + 26, name, SEGC[seg], True, 14)
        f.t(xx, py + 26 + 44, "段内 ×%.2f" % N.CURVE_SEG_X[seg - 1], SEGC[seg], size=12.5)

    # 每一跳的百分比：大跳粗体写在竖线右边，小跳小字写在台阶下面
    def jump(xx, tp, t, p, col, name=None, row=0):
        lab = ("%+.1f%%" % p).replace("-", "−") if p else "±0"
        if abs(p) >= 5:
            ym = (Y(tp) + Y(t)) / 2
            f.t(xx - HW + 5, ym + 5, lab, col, True, 14)
            if name:
                f.t(xx - HW + 5, ym - 12, name, INK, size=12)
        else:
            f.t(xx, Y(t) + (17 if row == 0 else 31), lab, GY, size=11, anchor="middle")

    prev_t = {13: N.HALF_START, 17: N.LONG_START}
    last = None
    for s, n, t, m, p, seg in st:
        tp = prev_t.get(s, last)
        if p is not None:
            jump(SX[s], tp, t, p, SEGC[seg], N.CURVE_NAME.get(s), s % 2)
        if s in N.CURVE_NUM:
            f.poly([(SX[s], Y(t) - 13), (SX[s] + 5, Y(t) - 8), (SX[s], Y(t) - 3), (SX[s] - 5, Y(t) - 8)], GR)
        last = t
    for i, (n, t, m, p, c, num) in enumerate(post):
        xx = SX["p%d" % i]
        jump(xx, last, t, p, PU, row=i % 2)
        if num:
            f.poly([(xx, Y(t) - 13), (xx + 5, Y(t) - 8), (xx, Y(t) - 3), (xx - 5, Y(t) - 8)], GR)
        last = t
    # 定稿之后：一条细虚线隔开
    xd = (SX[27] + SX["p0"]) / 2
    f.line(xd, TOP + 40, xd, Y(N.CURVE_END_S) + 38, GY2, 1, dash="2 3", arrow=False)
    f.t(xd - 4, TOP + 34, "定稿", GY, size=11.5, anchor="end")
    f.t(xd + 4, TOP + 34, "之后", GY, size=11.5)

    # 被否掉的尝试
    for at, name, t, p in N.CURVE_REJ:
        xx = SX[at] + {3: -0.5 * U, 20: U}.get(at, 0.0)        # 第 20 站那次挪半格，给第 19 站的名字让位
        yy = Y(t)
        right = at != "p1"
        f.line(xx - 6, yy - 6, xx + 6, yy + 6, REJC, 2, arrow=False)
        f.line(xx - 6, yy + 6, xx + 6, yy - 6, REJC, 2, arrow=False)
        if right:
            f.t(xx + 10, yy + 4, "%s +%.1f%%" % (name, p), GY, size=11.5)
        else:
            f.t(xx + 6, yy - 30, name, GY, size=11.5, anchor="end")
            f.t(xx + 6, yy - 14, "+%.1f%%" % p, GY, size=11.5, anchor="end")

    # 有效 25% 线（长文的分子）
    f.line(SX["b17"] + 0.85 * U, Y(N.TARGET25_S), X1, Y(N.TARGET25_S), GR, 1.2, dash="2 3", arrow=False)
    f.t(SX["b17"] + 0.95 * U, Y(N.TARGET25_S) + 15, "有效 25%", GR, size=11.5)

    # 关键读数
    f.t(SX[0] + HW + 8, Y(N.FRONT_BASE_S) + 5, "原样 %.3f 秒" % N.FRONT_BASE_S, INK, True, 13)
    f.t(SX[11], Y(N.FRONT_END_S) + 46, "%.3f 秒 · %.2f%%" % (N.FRONT_END_S, N.FRONT_END_MFU), INK, size=12, anchor="middle")
    f.t(SX[16], Y(N.HALF[-1][2]) + 36, "%.3f 秒 · %.2f%%" % (N.HALF[-1][2], N.HALF[-1][3]), INK, size=12, anchor="middle")
    f.t(SX[25], Y(N.END_S) + 48, "定稿 %.3f 秒 · %.2f%%" % (N.END_S, N.END_MFU), INK, size=12, anchor="middle")

    # 横轴：站号（定稿之后不编号）
    for s, *_r in st:
        if s % 2 == 0 or s in (13, 17, 27):
            f.t(SX[s], BOT + 22, str(s), GY, size=11.5, anchor="middle")
    f.t(X0 - 8, BOT + 22, "站", GY, size=11.5, anchor="end")

    f._pan = None
    yb = f.band(py + PH + 18, "ok", "前两刀砍掉一大半，后面每站只抠几个百分点；换底座后每段从自己的起点重新量", [])
    yb = f.src(yb + 10,
               "实测（作者私有编年史总表与调优记录，64 片 v7x）：台阶旁的百分比是这一站对上一站，优先用同一个作业里的对照；跨作业的（如第 16 站）在编年史正文里写明。",
               "段内倍数：第一段 %.3f ÷ %.3f，第二段 %.3f ÷ %.3f，第三段 %.3f ÷ %.3f（本课推导）。" % (
                   N.FRONT_BASE_S, N.FRONT_END_S, N.HALF_START, N.HALF[-1][2], N.LONG_START, N.CURVE_END_S),
               "定稿之后三步：路由先挑 top-k 再收集（同作业对照 %.3f → %.3f 秒）；融合核去掉报给调度器的耗时估计（%.3f → %.3f 秒，绕开了调度器挑锚点的问题）；调度门槛（同一台机器对照 %.3f → %.3f 秒，对照已含编译组合，编译组合本身 −0.5%%、在噪声带边上；调度部分 30 步逐位）。三步都过了两道轻检查：只有融合核那步有舍入增删；top-k 那步不逐位但舍入点无增删；门槛那步本身逐位，舍入变化来自对照里的编译组合。终点是 30 步口径；同一配置 200 步长跑 %.3f 秒、有效 %.2f%%（第 5–199 步，0 次回放、0 丢 token），比上一步的长跑 %.3f 秒快 %.1f%%，长跑第一次过 25%%。" % (
                   post[0][4], post[0][1], post[1][4], post[1][1], post[2][4], post[2][1], N.CURVE_RUN200_S, N.CURVE_RUN200_MFU, N.CURVE_RUN200_PREV[0], -N.CURVE_RUN200_VS_PREV),
               "灰叉画在它所在底座上的实测步时，百分比对它自己的对照：预取下一层权重 %.3f → %.3f 秒；融合核照原样 %.3f → %.3f 秒；核耗时按实测校准 %.3f → %.3f 秒。" % (
                   N.CURVE_PREFETCH[0], N.CURVE_PREFETCH[1], N.IH_CTL, N.IH_S, N.CE_CTL, N.CE["cal"][0]),
               "为什么不把三段串乘成一个总倍数：每站对上一站的倍数串乘起来约 %.1f 倍，同一底座实测只有 %s 倍——换了底座，同一刀的收益不一样大。" % (
                   N.CURVE_CHAIN_X, N.MILE20_X_T),
               "有效 25%% 的线按长文有效分子 %.1f TFLOP 推：%.1f ÷（0.25 × %d）≈ %.2f 秒（本课推导），只对长文那一段有意义。" % (
                   N.EFF_LONG, N.EFF_LONG, N.PEAK_DEV, N.TARGET25_S))
    f.save("fig8b-curve.svg", yb + 14)



def fig_mile():
    f = Fig(W, "同一个底座（20 层）、同一份以文字为主的长文数据、同一个作业里，把原始代码和三段终点的配置各跑一格：原始代码 %.3f 秒，第一段终点 %.3f 秒（快 %.2f 倍），第二段终点 %.3f 秒（再快 %.2f 倍），定稿 %.3f 秒（再快 %.2f 倍），合计快 %s 倍。"
            % (N.MILE20_0, N.MILE20_1, N.MILE20_SEG[0], N.MILE20_2, N.MILE20_SEG[1], N.MILE20_3, N.MILE20_SEG[2], N.MILE20_X_T))
    y0 = f.header("首尾放到一起直接比：同一个底座、同一份数据、一个作业里连着量",
                  "条长 ＝ 一步的秒数；只有这一组能首尾相除（20 层底座，以文字为主的长文）", [(INK, "原始代码"), (GY2, "三段终点")])
    PH = 170
    py = f.panel(0, y0, W, PH, "20 层底座、长文数据、一个作业（实测）", INK)
    bx0, sc = 330, 900.0 / N.MILE20_0
    labs = ["原样（原始代码）", "第一段终点", "第二段终点", "长文定稿"]
    for i, (n, t) in enumerate(N.MILE20):
        yy = py + 20 + i * 30
        f.t(bx0 - 12, yy + 14, labs[i], INK, size=13, anchor="end")
        f.box(bx0, yy + 2, t * sc, 18, INK if i == 0 else GY2, INK if i == 0 else GY2, 3)
        tag = "%.3f 秒" % t + ("" if i == 0 else "　×%.2f" % N.MILE20_SEG[i - 1])
        f.t(bx0 + t * sc + 10, yy + 16, tag, INK, i == 3, 13)
    f.t(W - 20, py + 110, "合计快 %s 倍" % N.MILE20_X_T, INK, True, 17, "end")
    f._pan = None
    yb = f.band(py + PH + 18, "ok", "大头在第一段（4.62 倍）：专家并行那一刀和搬来的注意力配置，后两段各一成多", [])
    yb = f.src(yb + 10,
               "实测：20 层底座、长文数据、同一个作业。三段终点的配置截到 20 层（半层底座上原始代码放不下，要 103.24 GiB）。",
               "这几个数跟编年史里的读数不是同一个底座、同一份数据，只在这张图里互相比。生产图像比例的长图文上同样做一组：%.3f → %.3f 秒，快 %s 倍（第十一节正文）。" % (N.STD_START_S, N.STD_V2_20[0], N.STD_X_T))
    f.save("fig8b-mile.svg", yb + 14)


if __name__ == "__main__":
    fig_curve()
    fig_mile()
