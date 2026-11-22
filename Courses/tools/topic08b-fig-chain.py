# -*- coding: utf-8 -*-
r"""专题九 · 第六节「关键路径卡在哪」的四张图。

fig8b-linkbw    回答：那条链为什么这么长？—— 五种组合（收集／规约、FP32／BF16、单核／双核）的字节和耗时落在同一条直线上：每设备约 186 GB/s。
fig8b-perlayer  回答：梯度改 BF16 ＋ 两个开关，快在哪？—— 反向每层：主核计算不变，等待 29.5 → 8.5。
fig8b-additive  回答：一加一大于二吗？—— 三组「单项推出来的 vs 实测」：旧数据单项相加、新数据单项相乘、不含精度改动的一组相加。
fig8b-twoqueue  回答：链短不了，怎么藏？—— 示意：同一颗小核的队列里权重和 token 排在一起；分到两颗小核以后各走各的。
"""
import topic08_figfont  # noqa: F401
from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE, LINE2, PU
Fig.BAND_KEEP_LINES = 0
import topic08b_numbers as N

W = 1400


def fig_linkbw():
    f = Fig(W, "横轴是一条集合通信每设备要收的字节，纵轴是它的耗时。五种组合：专家权重收集和专家梯度规约，FP32 和 BF16，一颗小核和两颗小核，"
               "五个点都落在同一条过原点的直线上，斜率约每设备每秒 186 GB。耗时只跟字节有关，跟用几颗小核无关。")
    y0 = f.header("耗时只跟字节走：最可能卡在芯片之间的链路上（推断）",
                  "横轴 ＝ 每设备要收的字节（MB）；纵轴 ＝ 一次的耗时（ms）；虚线 ＝ 每设备 186 GB/s",
                  [(BL, "收集（all-gather，空心圈）"), (RD, "规约（reduce-scatter，实心点）")])
    PH = 400
    py = f.panel(0, y0, W, PH, "同一份 profile 与对照 profile 里的五种组合（实测）", BL)
    X0, Y0 = 200, py + PH - 60
    XW, YH = 900, 290
    XM, YM = 3200.0, 17.5

    def X(mb):
        return X0 + mb / XM * XW

    def Y(ms):
        return Y0 - ms / YM * YH
    f.line(X0, Y0, X0 + XW, Y0, GY2, 1.2, arrow=False)
    f.line(X0, Y0, X0, Y0 - YH, GY2, 1.2, arrow=False)
    for mb in (0, 1000, 2000, 3000):
        f.t(X(mb), Y0 + 24, "%d MB" % mb, GY, size=13, anchor="middle")
    for ms in (0, 5, 10, 15):
        f.t(X0 - 10, Y(ms) + 5, "%d ms" % ms, GY, size=13, anchor="end")
    f.line(X(0), Y(0), X(3150), Y(3150 / 186.0), GY2, 1.4, dash="6 5", arrow=False)
    f.t(X(3150) - 6, Y(3150 / 186.0) - 12, "186 GB/s", GY, True, 14, "end")
    for i, (name, mb, ms) in enumerate(N.LINK_ROWS):
        if "收集" in name:
            f.box(X(mb) - 10, Y(ms) - 10, 20, 20, "none", BL, 10, sw=3)
        else:
            f.box(X(mb) - 6, Y(ms) - 6, 12, 12, RD, RD, 6)
    f.t(X(755) + 16, Y(4.05) + 22, "BF16：单核收集、双核规约，都是约 4.05 ms", INK, True, 14)
    f.t(X(1510) + 16, Y(8.15) + 22, "两份拼一起收集 ／ FP32 另一份规约：约 8.15 ms", INK, True, 14)
    f.t(X(3020) - 16, Y(16.19) + 6, "FP32 规约 16.19 ms", INK, True, 14, "end")
    f.t(X0 + XW + 30, py + 90, "让两颗小核一起收集：", INK, True, 15)
    f.t(X0 + XW + 30, py + 120, "%.3f → %.3f 秒" % (N.AG2SC_CTL, N.AG2SC_S), RD, True, 15)
    f.t(X0 + XW + 30, py + 148, "反而慢 %.1f%%" % N.AG2SC_PCT, RD, True, 15)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "单核双核、FP32 BF16 都在一条线上：要缩这条链，只能少搬字节", [])
    yb = f.src(yb + 10,
               "实测：五种组合各自一条集合通信的每设备实收字节与单次耗时（profile 第 11 步），折合 %s GB/s。" % "、".join("%.0f" % x for x in N.LINK_BW),
               "实测：这两类通信只走 3D 环面的两个轴（每轴正反两条；编译产物里的环配置）；一颗芯片两个设备同时各收各的。",
               "实测：让专家权重收集也用两颗小核，同一个作业里 %.3f → %.3f 秒（第 5–9 步），慢 %.1f%%；每条收集没有变快，反而多占一条队列。" % (N.AG2SC_CTL, N.AG2SC_S, N.AG2SC_PCT),
               "📌 公开规格：每芯片 6 条 ICI 链路，每条双向 200 GB/s（Google Cloud TPU7x 文档）。⚠️ 本课推导：每方向约 100 GB/s；这两类通信走 3D 环面的两个轴共 4 条，入向约 400 GB/s；一颗芯片两个设备合计约 372 GB/s，到九成多。一颗芯片里两块计算晶粒怎么分这几条链路官方没公开，所以这个对比只对量级；「卡在链路上」也是由此推断。")
    f.save("fig8b-linkbw.svg", yb + 14)


def fig_perlayer():
    b, a = N.LAYER_BEFORE, N.LAYER_AFTER
    f = Fig(W, "反向每一层的五个数，改动前后并排。每层时长 79.7 变 58.4 毫秒；主核在算 50.2 变 49.9，基本不变；主核在等 29.5 变 8.5；"
               "小核 0 忙 9.3 变 21.6；小核 1 忙 52.3 变 40.6。快的全是等待。")
    y0 = f.header("梯度改 BF16 ＋ 两个开关：反向每层快的 21 毫秒，全是等待",
                  "每组两根：浅灰 ＝ 改之前，深灰 ＝ 改之后（主核在等那根标红）；单位毫秒，反向各层的中位数",
                  [(GY2, "改之前"), (GY, "改之后"), (RD, "主核在等（改之后）")])
    PH = 320
    py = f.panel(0, y0, W, PH, "反向每层（实测，各层中位数）", BL)
    keys = [("interval", "每层时长"), ("compute", "主核在算"), ("wait", "主核在等"), ("sc0", "小核 0 忙"), ("sc1", "小核 1 忙")]
    X0, GAP = 150, 240
    YB, SC = py + PH - 60, 2.6
    for i, (k, lab) in enumerate(keys):
        x = X0 + i * GAP
        for j, (d, col) in enumerate(((b, GY2), (a, RD if k == "wait" else GY))):
            h = d[k] * SC
            f.box(x + j * 62, YB - h, 52, h, col, col, 3)
            f.t(x + j * 62 + 26, YB - h - 8, "%.1f" % d[k], RD if (k == "wait" and j == 1) else INK, True, 14, "middle")
        f.t(x + 57, YB + 26, lab, INK, True, 15, "middle")
    f.line(X0 - 20, YB, X0 + 5 * GAP - 60, YB, GY2, 1.2, arrow=False)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "主核的计算一点没少，少的是等：专家梯度规约每层 %.1f → %.1f 毫秒，字节减半" % (N.RS_WALL_BEFORE, N.RS_WALL_AFTER), [])
    yb = f.src(yb + 10,
               "实测：同一个作业，对照 %.3f 秒 → 合开 %.3f 秒（−16.9%%）；profile 第 11 步，反向各层逐层切开取中位数。" % (N.COMBO_CTL, N.COMBO_S),
               "实测：代价是两份专家权重改在运行时拼接，主核多算约 %.1f ms（一步的账里，已经算在「在算」那一侧）。" % N.CONCAT_COST_MS,
               "📌 小核 0 变忙、小核 1 变闲：梯度规约不再固定在小核 1 上，改由编译器分给两颗；但它变快靠的是字节减半（上一张图），不是两颗一起跑。")
    f.save("fig8b-perlayer.svg", yb + 14)


def fig_additive():
    rows = [
        ("① 旧数据单项，相加推", N.COMBO_CTL * (1 - (N.PK_SINGLE_A + N.PK_SINGLE_B) / 100), "单项是在另一份数据上量的", GY2),
        ("② 旧数据单项，相乘推", N.COMBO_CTL * (1 - N.PK_SINGLE_A / 100) * (1 - N.PK_SINGLE_B / 100), "只把加法换成乘法：几乎不变", GY2),
        ("③ 新数据单项，相乘推", N.PRED_MULT, "三项都在新数据上重测（−12.4%、−2.7%、−4.1%）再相乘", PU),
        ("④ 实测：三项合开", N.COMBO_S, "", INK),
    ]
    f = Fig(W, "同一个对照 5.311 秒。拿旧数据上两个单项相加推，预计 4.663 秒，实测合开 4.413 秒，看着像一加一大于二。"
               "只把加法换成乘法，预计 4.68 秒，几乎不变；再换成新数据上重测的单项相乘推，预计 4.34 秒，实测 4.413 秒，反而略慢于相乘。差在数据，不在加法乘法。")
    y0 = f.header("一加一大于二？拿同一份数据量的单项去推，就不大于了",
                  "横轴 ＝ 一步（秒），点越靠左越快；每一行是一种推法，最后一行是实测；竖虚线是对照",
                  [(GY2, "旧数据单项推"), (PU, "新数据单项推"), (INK, "实测")])
    PH = 360
    py = f.panel(0, y0, W, PH, "同一个对照 %.3f 秒（实测；推的三行是本课推导）" % N.COMBO_CTL, BL)
    LO, HI = 4.2, 5.4
    X0, XW = 470, 700

    def X(v):
        return X0 + (v - LO) / (HI - LO) * XW
    f.line(X(N.COMBO_CTL), py + 40, X(N.COMBO_CTL), py + 280, GY2, 1.2, dash="4 4", arrow=False)
    f.t(X(N.COMBO_CTL) + 8, py + 52, "对照 %.3f" % N.COMBO_CTL, GY, True, 13, "start")
    for i, (lab, v, note, col) in enumerate(rows):
        yy = py + 70 + i * 52
        f.t(X0 - 14, yy + 22, lab, INK, True, 15, "end")
        f.line(X(v), yy + 18, X(N.COMBO_CTL), yy + 18, col, 1.4, dash="3 3", arrow=False)
        f.box(X(v) - 8, yy + 10, 16, 16, col, col, 8)
        f.t(X(v) - 16, yy + 23, "%.3f" % v, col, True, 15, "end")
        if note:
            f.t(X(N.COMBO_CTL) + 14, yy + 23, note, GY, size=13)
    f.line(X(N.COMBO_S), py + 60, X(N.COMBO_S), py + 280, INK, 1.0, dash="2 4", arrow=False)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "「超可加」多半是口径：两个单项和合开，不是在同一份数据上量的", [])
    yb = f.src(yb + 10,
               "实测：两个单项 −6.5%%、−5.7%% 是在原数据上测的；合开 −16.9%% 是在滤掉长文档的数据上测的。另一组不含精度改动的改动（对照 %.3f）三个单项按相乘推出 %.3f（本课推导，不是实测），实测 %.3f，比推的再快约 %.1f%%；单项和合开不在同一个作业，这点差不当作协同的证据（原因没查）。" % (N.NOPREC_CTL, N.NOPREC_PRED_MULT, N.NOPREC_S, (1 - N.NOPREC_S / N.NOPREC_PRED_MULT) * 100),
               "实测：在新数据上重测梯度 BF16 单项：%.3f → %.3f 秒（另一个作业，对照与本作业对照差 0.003 秒，在跨作业尺子 0.018 秒以内）。" % (N.F5_CAP_CTL, N.F5_CAP_S),
               "📌 为什么旧数据上量小了：原数据上最忙设备的注意力很长，把专家梯度那条链大半盖住了，等它的时间只有 %.0f ms；新数据上露出来 %.0f ms，砍它就砍得多。" % (N.RS_WAIT_PK, N.RS_WAIT_CAP),
               "⚠️ 本课推导：相乘那一行 ＝ 5.311 × (4.650 ÷ 5.308) × (1 − 开关甲 2.7%%) × (1 − 开关乙 4.1%%) ＝ %.3f（4.650 ÷ 5.308 是梯度 BF16 在新数据上的单测；开关甲、乙是那两条开关在新数据上各自的单测）；相加那一行 ＝ 5.311 × (1 − 12.2%%)；旧数据相乘那一行 ＝ 5.311 × (1 − 6.5%%) × (1 − 5.7%%)。" % N.PRED_MULT)
    f.save("fig8b-additive.svg", yb + 14)


def fig_twoqueue():
    f = Fig(W, "示意，前向一层。左边：小核 0 一条队列里，专家权重的收集和 token 的收发排在一起，加起来跟主核这一层的计算一样长，"
               "只要一条排晚了，主核就得等；小核 1 几乎闲着。右边：把专家权重收集挪到小核 1、token 留在小核 0，各走各的队列，"
               "两类通信走的链路方向也不一样，互不抢道。")
    y0 = f.header("链短不了，就把它藏住：别让两类通信挤在同一颗小核的队列里",
                  "示意，前向一层；每颗小核是一条先来先做的队列；左边是改之前（实测的忙时），右边是改之后",
                  [(OR, "专家权重收集"), (PU, "token 收发"), (BL, "主核计算")])
    PH = 330
    py = f.panel(0, y0, W, PH, "前向一层（左边忙时是实测，排布是示意）", BL)
    SC = 26.0
    for k, (title, x0) in enumerate((("改之前：挤在小核 0", 60), ("改之后：分到两颗", 740))):
        f.t(x0, py + 50, title, INK, True, 16)
        f.t(x0, py + 92, "主核", INK, True, 14)
        f.box(x0 + 80, py + 72, N.FWD_TC_COMPUTE * SC, 30, BL, BL, 3)
        f.t(x0, py + 152, "小核 0", INK, True, 14)
        f.t(x0, py + 212, "小核 1", INK, True, 14)
        if k == 0:
            segs = [(OR, 4.0), (PU, 1.4), (OR, 4.1), (PU, 1.4), (OR, 4.1), (PU, 1.4), (GY2, 2.17)]
            x = x0 + 80
            for col, d in segs:
                f.box(x, py + 132, d * SC, 30, col, col, 2)
                x += d * SC
            f.box(x0 + 80, py + 192, max(N.FWD_SC1_BUSY * SC, 3), 30, GY2, GY2, 2)
            f.t(x0 + 80, py + 260, "小核 0 忙 %.2f ≈ 主核 %.2f（ms/层）" % (N.FWD_SC0_BUSY, N.FWD_TC_COMPUTE), RD, True, 14)
        else:
            f.box(x0 + 80, py + 132, 4.2 * SC, 30, PU, PU, 2)
            f.box(x0 + 80, py + 192, 12.2 * SC, 30, OR, OR, 2)
            f.t(x0 + 80, py + 260, "%.3f → %.3f 秒（−%.1f%%）" % (N.SPLIT_CTL, N.SPLIT_S, N.SPLIT_PCT), GR, True, 14)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "能做的多半是藏：先看谁跟谁排在同一条队列里", [])
    yb = f.src(yb + 10,
               "实测（改之前）：前向每层中位数，小核 0 忙 %.2f ms（专家权重收集 12.18、token 4.20、其它 2.18），主核计算 %.2f ms；小核 1 只忙 %.2f ms。" % (N.FWD_SC0_BUSY, N.FWD_TC_COMPUTE, N.FWD_SC1_BUSY),
               "实测：整步看，专家权重那条链在小核上一共 %d ms，主核等它露出 %d ms；token 走另一个方向的链路，跟权重的两类通信不抢道。" % (N.CHAIN_SC_MS, N.CHAIN_EXPOSED_MS),
               "实测：用两个现成开关把 token 钉到小核 0、让编译器把专家权重收集排到小核 1：%.3f → %.3f 秒（−%.1f%%）；离线比过编译产物，只动了排布和核号，求和顺序不变。" % (N.SPLIT_CTL, N.SPLIT_S, N.SPLIT_PCT),
               "⚠️ 示意：左边小核 0 的段是按忙时拼的，实际顺序更碎；右边改之后的忙时没有逐层量，按改之前的分项画。")
    f.save("fig8b-twoqueue.svg", yb + 14)


if __name__ == "__main__":
    fig_linkbw()
    fig_perlayer()
    fig_additive()
    fig_twoqueue()
