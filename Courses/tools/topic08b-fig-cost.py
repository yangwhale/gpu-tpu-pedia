# -*- coding: utf-8 -*-
r"""专题九 · 9.5「定稿以后回头查：调度器凭什么把括号排空」的三张图（2026-11-22 补）。

fig8b-ceratio    回答：调度器拿来排重叠的「每颗核要多久」，准不准？—— 按核报的耗时换算出的时长 ÷ 实测：
                 专家矩阵乘偏长 2.3–2.9 倍，主干注意力 2.2–2.5 倍，视觉塔注意力反而偏短，融合核偏短 0.5–0.74，
                 小核上的专家权重收集（不是核，是调度器自己按公式估的）偏长约 3.2 倍；不报耗时的核按几乎 0 算。
fig8b-cebracket  回答：融合核报不报耗时，调度器自己排的括号差在哪？—— 反向一层两条专家权重收集，
                 调度器时间线上「放行 → 完成」的窗、窗里排进的计算：10.4 → 0.48 →（按实测调准）0.55 → 不报 11.9 ms。
fig8b-cecells    回答：把估计校准准了，为什么上卡反而慢？—— 八格上卡实测对对照的变化，和调度器自己估的整步时长。

⛔ 数据全部 import 自 topic08b_numbers.py（CE_*），断言在那边。
⛔ 公开课闸门：不写编译器开关名、不写运行库源码里的公式和常数、不写层数×每层的合计。
"""
import math

import topic08_figfont  # noqa: F401
from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE, LINE2, PU
Fig.BAND_KEEP_LINES = 0
import topic08b_numbers as N

W = 1400


def _dot(f, x, y, r, col, hollow=False):
    if hollow:
        f.p.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="#fff" stroke="%s" stroke-width="2.4"/>' % (x, y, r, col))
    else:
        f.p.append('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s" fill-opacity="0.85"/>' % (x, y, r, col))


def fig_ceratio():
    groups = [("gmm", "专家矩阵乘、专家权重梯度（6 颗）"), ("attn", "注意力核，主干（前向、反向）"),
              ("vit", "注意力核，视觉塔（前向、反向）"), ("fk", "融合核（4 颗）"), ("ag", "小核上的专家权重收集（不是核）")]
    R = N.CE_RANGE
    f = Fig(W, "调度器排通信重叠时用的每颗核的耗时，按核自己报的运算量和读写字节换算，再除以实测（每个主核单次）。横轴是对数刻度，中间竖线是估得正好。"
               "专家矩阵乘和专家权重梯度 6 颗偏长 %.1f 到 %.1f 倍；主干注意力两颗偏长 %.1f 到 %.1f 倍；视觉塔注意力两颗反而偏短，%.2f 到 %.2f；"
               "融合核 4 颗偏短，%.2f 到 %.2f；小核上的专家权重收集不是核，调度器自己按公式估，偏长约 %.1f 倍。"
               "另有 %d 颗核不报耗时，调度器按几乎不花时间算，实测各 0.3 到 0.6 毫秒。"
               % (R["gmm"] + R["attn"] + R["vit"] + R["fk"] + (R["ag"][0], N.CE_NOEST_KERNELS)))
    y0 = f.header("调度器手里的「每颗核要多久」：没有一样是准的",
                  "每个点 ＝ 一颗核：调度器按它报的耗时换算出的时长 ÷ 实测（对数刻度）；深色竖线 ＝ 估得正好，右边估长了，左边估短了",
                  [(RD, "估长了"), (BL, "估短了"), (GY2, "不报耗时：按几乎 0 算")])
    RH = 62
    PH = 60 + RH * (len(groups) + 1) + 40
    py = f.panel(0, y0, W, PH, "对照那一格的 profile 对调度器读到的估计（实测）", BL)
    X0, X1 = 520, 1180
    lo, hi = math.log(0.25), math.log(4.0)

    def X(v):
        return X0 + (math.log(v) - lo) / (hi - lo) * (X1 - X0)

    ytop, ybot = py + 40, py + 40 + RH * (len(groups) + 1)
    for v, lab in ((0.25, "0.25"), (0.5, "0.5"), (1.0, "1（正好）"), (2.0, "2"), (4.0, "4")):
        f.line(X(v), ytop, X(v), ybot, GY if v == 1.0 else LINE2, 2.0 if v == 1.0 else 1.0, arrow=False)
        f.t(X(v), ybot + 22, lab + ("" if v == 1.0 else " 倍"), INK if v == 1.0 else GY, v == 1.0, 13, "middle")
    for i, (g, name) in enumerate(groups):
        yy = ytop + 16 + i * RH
        f.t(X0 - 24, yy + 18, name, INK, True, 15, "end")
        vals = N.CE_RATIO[g]
        for j, v in enumerate(sorted(vals)):
            col = RD if v > 1 else BL
            _dot(f, X(v), yy + 13 + (j % 2) * 8 - 4, 8, col)
        a, b = R[g]
        col = RD if a > 1 else BL
        txt = ("偏长约 %.1f 倍" % a) if a == b else (("偏长 %.1f–%.1f 倍" % (a, b)) if a > 1 else ("只有实测的 %.2f–%.2f" % (a, b)))
        f.t(X1 + 24, yy + 18, txt, col, True, 15)
    yy = ytop + 16 + len(groups) * RH
    f.t(X0 - 24, yy + 18, "不报耗时的核（%d 颗）" % N.CE_NOEST_KERNELS, INK, True, 15, "end")
    f.line(X(0.34), yy + 13, X0 + 4, yy + 13, GY2, 2.0)
    f.t(X1 + 24, yy + 18, "按几乎 0 算", GY, True, 15)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "偏差有长有短、各差两三倍：调度器手里的这张账，本来就不准", [])
    yb = f.src(yb + 10,
               "实测：每颗核的实测取对照那一格 profile 里一颗芯片、一个主核的单次耗时；融合核在对照里没有，实测取同形状的另一格。估计取编译产物里每颗核随身带的耗时估计（运算量、读写字节），按调度器的换算方式折成时长。",
               "为什么偏（本课程作者的解释，部分是推断）：专家矩阵乘按整块 MoE 缓冲的行数报运算量，核实际只算有效行，所以偏长约 2.4 倍；注意力核按静态掩码报，运行时又跳掉一截空块，所以偏长，视觉塔那两颗反过来偏短；融合核报的读写字节核过是对的，慢在它只跑到显存带宽的一半左右。",
               "小核上的收集不是 Pallas 核，没有「报」这回事：调度器按自己的通信公式估，估约 26 ms，实测小核上约 8.2 ms。",
               "这张图只说明估计普遍不准；下一张图说明，准不准并不是括号空掉的原因。")
    f.save("fig8b-ceratio.svg", yb + 14)


def fig_cebracket():
    rows = [("ctl", "对照（没有融合核）", None), ("fk", "换上融合核，核报耗时", RD),
            ("fkm", "融合核，报的耗时按实测调准", RD), ("fk0", "融合核，不报耗时", GR)]
    W_ = N.CE_WIN
    f = Fig(W, "调度器自己排出来的时间线，反向一层里两条专家权重收集，拼好的 gate/up 一条、down 一条。每条从放行画到完成，蓝色是调度器排进这个括号的计算，全是它自己的估计。"
               "对照：gate/up 括号里排进 %.1f 毫秒，down 括号里 %.1f。换上融合核、核报耗时：只剩 %.2f 和 %.2f。把融合核报的耗时按实测调准：%.2f 和 %.2f，还是空的。"
               "融合核不报耗时：%.1f 和 %.1f，比对照还多。上卡实测，gate/up 那条收集每步露出：对照 %.1f 毫秒，核报耗时 %.1f，不报耗时不到 %d。"
               % ((W_["ctl"][0][1], W_["ctl"][1][1], W_["fk"][0][1], W_["fk"][1][1], W_["fkm"][0][1], W_["fkm"][1][1],
                   W_["fk0"][0][1], W_["fk0"][1][1]) + N.CE_AG_EXPOSED))
    y0 = f.header("融合核报不报耗时，括号里排不排计算",
                  "每条 ＝ 一条专家权重收集，从放行（左竖线）到完成（右竖线），长度按调度器自己估的毫秒；蓝 ＝ 它排进括号的计算，灰 ＝ 它自己记下的空等",
                  [(BL, "括号里排进的计算（调度器的估计）"), (LINE, "括号里空着（调度器以为在等）"), (PU, "融合核")])
    RH = 112
    PH = 90 + RH * len(rows) + 34
    py = f.panel(0, y0, W, PH, "调度器的时间线（离线编译导出，全是估计）；右边是上卡实测", BL)
    X0, SC = 420, 15.0
    exp = {"ctl": "%.1f ms" % N.CE_AG_EXPOSED[0], "fk": "%.1f ms" % N.CE_AG_EXPOSED[1], "fkm": "—", "fk0": "不到 %d ms" % N.CE_AG_EXPOSED[2]}
    f.t(1130, py + 50, "上卡实测", INK, True, 14)
    f.t(1130, py + 70, "gate/up 每步露出 · 一步", GY, False, 13)
    for i, (k, name, col) in enumerate(rows):
        yy = py + 90 + i * RH
        f.t(30, yy + 26, name, INK, True, 15)
        if k != "ctl":
            f.box(30, yy + 38, 74, 22, PU, PU, 4)
            f.t(67, yy + 54, "融合核", "#fff", True, 12, "middle")
            tag = {"fk": "报：约 0.1 ms", "fkm": "报：按实测调准", "fk0": "不报"}[k]
            f.t(116, yy + 54, tag, RD if k != "fk0" else GR, True, 13)
        for j, (nm, (win, comp)) in enumerate(zip(("gate/up", "down"), W_[k])):
            y2 = yy + 6 + j * 42
            f.t(X0 - 16, y2 + 20, nm, GY, True, 13, "end")
            f.box(X0, y2 + 6, win * SC, 22, "#f1f3f4", LINE, 2)
            if comp > 0:
                f.box(X0, y2 + 6, max(comp * SC, 3), 22, BL, BL, 2)
            f.line(X0, y2, X0, y2 + 34, INK, 2.0, arrow=False)
            f.line(X0 + win * SC, y2, X0 + win * SC, y2 + 34, INK, 2.0, arrow=False)
            lab = ("排进 %.1f ms" % comp) if comp >= 1 else ("只排进 %.2f ms" % comp)
            f.t(X0 + win * SC + 12, y2 + 22, lab, BL if comp >= 1 else RD, True, 14)
        f.t(1130, yy + 30, exp[k], (RD if k in ("fk",) else (GR if k == "fk0" else INK)), True, 16)
        f.t(1130, yy + 54, "%.3f 秒（%s）" % (N.CE[k][0], "对照" if k == "ctl" else ("%+.1f%%" % N.CE_PCT[k]).replace("-", "−")),
            RD if N.CE_PCT[k] > 0 else (GR if N.CE_PCT[k] < 0 else INK), True, 15)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "核报了耗时，括号就空；调准了还是空；不报，计算回来了", [])
    yb = f.src(yb + 10,
               "实测（离线编译时导出调度器排好的时间线，读每条指令的起止）：窗 ＝ 调度器时间线上从放行到完成的时长；窗里的计算 ＝ 夹在中间的指令按调度器估的时长合计；都是调度器自己的估计，不是上卡时间。"
               "窗长：对照 %.1f / %.1f，核报耗时 %.1f / %.1f，调准 %.1f / %.1f，不报 %.1f / %.1f ms（gate/up / down）。调度器自己记的空等（反向一层）：%.1f、%.1f、%.1f、%.1f ms。"
               % (W_["ctl"][0][0], W_["ctl"][1][0], W_["fk"][0][0], W_["fk"][1][0], W_["fkm"][0][0], W_["fkm"][1][0], W_["fk0"][0][0], W_["fk0"][1][0],
                  N.CE_WASTED["ctl"], N.CE_WASTED["fk"], N.CE_WASTED["fkm"], N.CE_WASTED["fk0"]),
               "实测（上卡，profile 第 11 步一颗芯片一个主核，按收集的名字合计全部层）：gate/up 那条每步露出 %.1f → %.1f → 不到 %d ms；down 那条 %.1f → %.1f → %.1f ms。对照的 %.1f 取同配置另一作业的 profile。核报耗时那格，gate/up 括号里主核真在干活的时间各层中位只有 %.2f ms（对照 %.1f–%.1f ms）。"
               % (N.CE_AG_EXPOSED + N.CE_WO_EXPOSED + (N.CE_AG_EXPOSED[0], N.CE_IHC_BUSY_MED) + N.CE_CTL_BUSY),
               "「报：约 0.1 ms」：融合核四颗各报的耗时换算成时长是 %.3f 到 %.3f ms；调准那格按实测把四颗各乘 1.35 到 1.98 倍。"
               "为什么报了这么小的耗时就把括号排空，见下一节的配对规则：报了耗时它就成了候选，又是本轮最大的计算，收集的完成点被钉在它旁边。" % N.CE_IHC_EST_MS)
    f.save("fig8b-cebracket.svg", yb + 14)


def fig_cecells():
    order = ["cal", "agm", "calagm", "fk", "fkm", "fk0", "calfk0"]
    names = {k: n for k, n, _s, _e in N.CE_CELLS}
    f = Fig(W, "八格上卡，都叠在同一个对照上（%.3f 秒）。Ⓐ 实测步时对对照的变化：gmm 和注意力核的估计按实测校准，慢 %.1f%%；小核收集的延迟估计按实测调短，慢 %.1f%%；两样一起，慢 %.1f%%；"
               "换上融合核、核报耗时，慢 %.1f%%；把它报的耗时调准，还是慢 %.1f%%；融合核不报耗时，快 %.1f%%；融合核不报耗时再加同样的校准，慢 %.1f%%。"
               "Ⓑ 调度器自己估的整步时长和上卡实测：对照估 %d 毫秒、实测 %d；校准以后估 %d、实测 %d；两样一起估 %d、实测 %d；融合核不报耗时估 %d、实测 %d。"
               % ((N.CE_CTL, N.CE_PCT["cal"], N.CE_PCT["agm"], N.CE_PCT["calagm"], N.CE_PCT["fk"], N.CE_PCT["fkm"], -N.CE_PCT["fk0"], N.CE_PCT["calfk0"])
                  + (N.CE["ctl"][1], round(N.CE["ctl"][0] * 1000), N.CE["cal"][1], round(N.CE["cal"][0] * 1000),
                     N.CE["calagm"][1], round(N.CE["calagm"][0] * 1000), N.CE["fk0"][1], round(N.CE["fk0"][0] * 1000))))
    y0 = f.header("把估计调准了，调度器估的整步落到实测附近，上卡却慢了",
                  "Ⓐ 条 ＝ 上卡实测步时对对照变化了百分之几（绿＝快，红＝慢）；Ⓑ 空心圈 ＝ 调度器自己估的整步，实心点 ＝ 上卡实测，同一行连一条线",
                  [(GR, "变快"), (RD, "变慢"), (GY, "空心：调度器估的；实心：实测")])
    # Ⓐ
    RH = 46
    PHA = 80 + RH * len(order)
    py = f.panel(0, y0, W, PHA, "Ⓐ 上卡实测，对对照 %.3f 秒（30 步取第 13–20 步）" % N.CE_CTL, BL)
    XC, SC = 760, 34.0
    f.line(XC, py + 30, XC, py + PHA - 40, GY2, 1.2, arrow=False)
    for i, k in enumerate(order):
        yy = py + 40 + i * RH
        v = N.CE_PCT[k]
        col = RD if v > 0 else GR
        f.t(XC - 170, yy + 20, names[k], INK, True, 15, "end")
        x0 = XC if v > 0 else XC + v * SC
        f.box(x0, yy + 4, abs(v) * SC, 24, col, col, 3)
        f.t((XC + v * SC + 10) if v > 0 else (XC + 10), yy + 22, ("%+.1f%%（%.3f 秒）" % (v, N.CE[k][0])).replace("-", "−"), col, True, 15)
    # Ⓑ
    rowsb = ["ctl", "cal", "agm", "calagm", "fk", "fkm", "fk0"]
    RB = 52
    PHB = 70 + RB * len(rowsb) + 34
    yB = py + PHA + 24
    pyb = f.panel(0, yB, W, PHB, "Ⓑ 调度器自己估的整步，对上卡实测（秒）", BL)
    XA, XB, lo, hi = 420, 1300, 3.8, 5.3

    def X(s):
        return XA + (s - lo) / (hi - lo) * (XB - XA)

    ytop, ybot = pyb + 40, pyb + 40 + RB * len(rowsb)
    for s in (3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2):
        f.line(X(s), ytop, X(s), ybot, LINE2, 1.0, arrow=False)
        f.t(X(s), ybot + 22, "%.1f" % s, GY, False, 13, "middle")
    f.line(X(N.CE_CTL), ytop, X(N.CE_CTL), ybot, GR, 1.6, dash="5 4", arrow=False)
    f.t(X(N.CE_CTL) + 6, ytop - 6, "对照实测 %.3f" % N.CE_CTL, GR, True, 13)
    for i, k in enumerate(rowsb):
        yy = ytop + 22 + i * RB
        s, e = N.CE[k]
        e = e / 1000.0
        f.t(XA - 20, yy + 6, names[k], INK, True, 15, "end")
        f.line(X(min(s, e)), yy, X(max(s, e)), yy, GY2, 1.4, arrow=False)
        _dot(f, X(e), yy, 8, GY, hollow=True)
        col = RD if N.CE_PCT[k] > 0 else (GR if N.CE_PCT[k] < 0 else INK)
        _dot(f, X(s), yy, 8, col)
        f.t(X(e), yy - 13, "估 %d" % round(e * 1000), GY, False, 13, "middle")
    f._pan = None
    yb = f.band(pyb + PHB + 20, "ok", "八格里只有「融合核不报耗时」那一格兑现了，−3.8%；几格校准全都变慢", [])
    yb = f.src(yb + 10,
               "实测：八格各跑在不同的常驻机器上，跨机噪声约 0.4%；30 步取第 13–20 步。对照 ＝ 定稿之后第一步（路由先挑 top-k 再收集；开篇总览里 4.082 秒，这组实验在别的常驻机器上重测 4.088 秒）。",
               "实测（逐位）：三格校准对对照 30 步 loss、梯度范数逐位相同；融合核那几格之间（报、调准、不报）也逐位相同：耗时估计只进调度器，不进计算。融合核本身对原来的写法不逐位（loss 最大相对差 %.1e，核里的累加顺序不同）。" % N.CE_LOSS_IHC_VS_CTL,
               "实测：调度器自己估的整步来自离线编译时导出的时间线；「融合核不报耗时 ＋ 同样校准」那格没导出。",
               "为什么校准反而慢（下一节的配对规则 ＋ 编译产物）：校准那几格只改了估计，门槛还是默认的极低值；「两类核校准」还给原来不报耗时的 token 搬运核补上了估计，它们也成了候选。最终程序里，gate/up 收集的搭档从矩阵乘前移到搬运核和小的矩阵乘，括号被排短；客观判据算出这格的露出比对照多 %d ms。旁证：只调短收集延迟估计那格，调度器自己记的空等从 %.1f 降到 %.1f ms，前向一条专家权重收集的实测露出却从 218 涨到 352 ms：调度器的账变好看，不等于搭档选对。" % (N.CE_CAL_EXTRA_MS, N.CE_WASTED["ctl"], N.CE_WASTED["agm"]))
    f.save("fig8b-cecells.svg", yb + 14)


def fig_celayer():
    f = Fig(W, "层调度器怎么给通信找搭档，示意。左边：融合核报了耗时。候选是两条专家权重收集和报了耗时的融合核，按依赖一层层剥开；第一层里同时有通信和计算，最大的计算就是融合核，"
               "于是两条收集的完成点都钉在融合核旁边。专家矩阵乘在后面一层，已经没有通信可配。主调度器从后往前排，完成点钉在层体最前面，开始点还要再往前放一个估计的通信时长，前面几乎没有计算，括号是空的，主核在融合核旁边停下来等。"
               "右边：融合核不报耗时，进不了候选；第一层只有通信，第一次碰到计算是专家矩阵乘那一层，完成点钉在矩阵乘旁边，也就是真正用这批权重的地方，它上游的计算都能放进括号。")
    y0 = f.header("层调度器：每一层里最大的那颗计算，和一条通信配成一对",
                  "上排 ＝ 候选按依赖剥成的轮；下排 ＝ 主调度器据此排出的时间线（完成点钉在搭档旁边，开始点往前放一个估计的通信时长）",
                  [(OR, "专家权重收集"), (PU, "融合核"), (BL, "其余计算"), (GR, "专家矩阵乘（用这批权重）"), (RD, "主核停着等")])
    PW, PH = 680, 470
    for k, (title, est) in enumerate((("Ⓐ 融合核报了耗时（约 0.1 ms）", True), ("Ⓑ 融合核不报耗时", False))):
        x0 = k * (PW + 40)
        py = f.panel(x0, y0, PW, PH, title, RD if est else GR)
        # 上排：层
        yl1, yl2 = py + 44, py + 120
        f.t(x0 + 20, yl1 + 22, "第 1 层", INK, True, 14)
        f.t(x0 + 20, yl2 + 22, "第 2 层", INK, True, 14)
        f.box(x0 + 100, yl1 + 4, 130, 28, OR, OR, 4); f.t(x0 + 165, yl1 + 23, "收集 gate/up", "#fff", True, 13, "middle")
        f.box(x0 + 240, yl1 + 4, 110, 28, OR, OR, 4); f.t(x0 + 295, yl1 + 23, "收集 down", "#fff", True, 13, "middle")
        if est:
            f.box(x0 + 360, yl1 + 4, 90, 28, PU, PU, 4); f.t(x0 + 405, yl1 + 23, "融合核", "#fff", True, 13, "middle")
            f.t(x0 + 462, yl1 + 16, "本轮最大的计算", RD, True, 13)
            f.t(x0 + 462, yl1 + 34, "→ 两条收集都配给它", RD, True, 13)
        else:
            f.box(x0 + 360, yl1 + 4, 90, 28, "#fff", GY2, 4, dash="4 3"); f.t(x0 + 405, yl1 + 23, "融合核", GY, True, 13, "middle")
            f.t(x0 + 462, yl1 + 16, "不报耗时：不是候选", GY, True, 13)
            f.t(x0 + 462, yl1 + 34, "本层只有通信，顺延", GY, True, 13)
        f.box(x0 + 100, yl2 + 4, 120, 28, GR, GR, 4); f.t(x0 + 160, yl2 + 23, "专家矩阵乘", "#fff", True, 13, "middle")
        f.box(x0 + 230, yl2 + 4, 120, 28, GR, GR, 4); f.t(x0 + 290, yl2 + 23, "专家矩阵乘", "#fff", True, 13, "middle")
        f.t(x0 + 362, yl2 + 23, "已没有通信可配" if est else "← 收集在这里才配到搭档", RD if est else GR, True, 13)
        # 下排：时间线
        yt = py + 200
        f.t(x0 + 20, yt - 8, "排出来的一层（示意）", GY, True, 13)
        TX, U = x0 + 70, 26.0          # 每 ms 多少 px（示意）
        lane_tc, lane_sc = yt + 50, yt + 100
        f.t(x0 + 20, lane_tc + 18, "主核", BL, True, 13)
        f.t(x0 + 20, lane_sc + 16, "小核", OR, True, 13)
        fused, up, gm = 1.2, 11.9, 3.0   # 示意：融合核、上游计算、矩阵乘（ms）；上游 11.9 ＝ 不报耗时时括号里排进的计算
        L = 8.2
        if est:
            done = fused
            segs = [(0, fused, PU), (fused, fused + L, RD), (fused + L, fused + L + up - fused, BL), (fused + L + up - fused, fused + L + up - fused + gm, GR)]
            start = done - L * 0.0       # 示意：开始点往前放的空间不够，紧挨着放行
            rel = max(0.0, done - 0.15)
        else:
            done = up
            segs = [(0, fused, PU), (fused, up, BL), (up, up + gm, GR)]
            rel = up - L - 1.0
        for a, b, col in segs:
            f.box(TX + a * U, lane_tc, (b - a) * U, 26, col, col, 2)
        f.box(TX + rel * U, lane_sc, L * U, 22, OR, OR, 2)
        for xx, lab in ((rel, "开始"), (done, "结束")):
            f.line(TX + xx * U, lane_tc - 16, TX + xx * U, lane_sc + 30, INK, 2.2, arrow=False)
        f.t(TX + rel * U - 4, lane_tc - 22, "开始", INK, True, 13, "end" if est else "middle")
        f.t(TX + done * U + 4, lane_tc - 22, "结束（钉在%s旁）" % ("融合核" if est else "矩阵乘"), RD if est else GR, True, 13)
        f.t(x0 + 20, yt + 170, ("括号里几乎没有计算：主核在融合核旁边停下来等，权重其实要到后面的矩阵乘才用" if est else
                               "括号里放得下矩阵乘上游的计算（调度器估 11.9 ms），收集藏在底下"), RD if est else GR, True, 14, w=PW - 40)
        f.t(x0 + 20, yt + 196, ("gate/up 收集每步露出 290.5 ms（上卡实测）" if est else "gate/up 收集每步露出不到 14 ms（上卡实测）"),
            RD if est else GR, False, 14)
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "完成点跟着搭档走：搭档在上游，括号就空；搭档在用数据的地方，括号就满", [])
    yb = f.src(yb + 10,
               "这是层调度器行为的示意，省掉了两条次要规则：一轮里只有通信、没有计算时怎么顺延；同一轮里几颗计算一起配给一组通信。块的长短示意；11.9 ms 是调度器自己的估计，290.5 和不到 14 ms 是上卡实测。",
               "核实：最终编译产物里，每条通信的完成点和哪颗计算被分进同一个调度组，可以直接读出来。融合核报耗时时，两条专家权重收集的完成点和融合核同组；不报时，和专家矩阵乘同组。",
               "默认的入选门槛极低（按估计时长约一微秒量级），所以一颗核只要报了耗时，哪怕只有 0.1 毫秒，就是候选。融合核报的耗时调准以后依然是候选、依然是本轮最大的计算，所以调准也没用（上一张图的第三行）。")
    f.save("fig8b-celayer.svg", yb + 14)


def fig_cethresh():
    xs = N.CE_TH_MS
    f = Fig(W, "所有核照实报耗时、小核收集的延迟估计乘 0.32 以后，把 Pallas 核入选候选的门槛从默认一路提到 10 毫秒，客观判据预测的整步（4 层模型按实测核时长回放，只用来排名）。"
               "64 片拓扑：默认 %.0f，0.75 到 1 毫秒是一个平台 %.0f 上下，3 毫秒最好 %.0f，6 毫秒以上不再变化 %.0f，跟现在的推荐配置 %.0f 差不多。128 片拓扑：默认 %.0f，3 毫秒最好 %.0f，6 毫秒以上 %.0f，推荐配置 %.0f。"
               % (N.CE_TH64[0], N.CE_TH64[4], N.CE_TH64[7], N.CE_TH64[-1], N.CE_TH_V4[0], N.CE_TH128[0], N.CE_TH128[7], N.CE_TH128[-1], N.CE_TH_V4[1]))
    y0 = f.header("把门槛提到 3 毫秒：耗时短的核挡在门外，收集的完成点回到用数据的地方",
                  "横轴 ＝ Pallas 核入选候选的门槛（按估计时长）；纵轴 ＝ 客观判据预测的整步（毫秒，越低越好）；所有核照实报耗时，收集延迟估计 ×0.32",
                  [(BL, "64 片拓扑"), (OR, "128 片拓扑"), (GY2, "当时的推荐配置（融合核不报）")])
    PH = 520
    py = f.panel(0, y0, W, PH, "整步预测（离线编译 ＋ 实测核时长回放；3 ms 那格上卡 −1.0%）", BL)
    XL, XR = 200, 1240
    YT, YB, lo, hi = py + 50, py + PH - 90, 2650, 3400
    def X(i):
        return XL + i * (XR - XL) / (len(xs) - 1)
    def Y(v):
        return YB - (v - lo) / (hi - lo) * (YB - YT)
    for v in (2700, 2900, 3100, 3300):
        f.line(XL - 10, Y(v), XR + 10, Y(v), LINE2, 1.0, arrow=False)
        f.t(XL - 18, Y(v) + 5, "%d" % v, GY, False, 13, "end")
    for i, x in enumerate(xs):
        f.t(X(i), YB + 26, "默认" if x == "def" else ("%g ms" % x), INK if x in ("def", 3) else GY, x in ("def", 3), 13, "middle")
    # 6 ms 以上的区域
    i6 = xs.index(6)
    f.box(X(i6) - 30, YT - 10, X(len(xs) - 1) - X(i6) + 60, YB - YT + 20, "#f1f3f4", LINE, 6)
    f.t((X(i6) + X(len(xs) - 1)) / 2, YT + 10, "6 ms 以上：没有核够格当搭档", GY, True, 13, "middle")
    # 推荐配置参考线
    v4 = sum(N.CE_TH_V4) / 2
    f.line(XL, Y(v4), XR, Y(v4), GY2, 1.6, dash="6 4", arrow=False)
    f.t(XL + 6, Y(v4) - 8, "现在的推荐配置 ≈ %.0f" % v4, GY, True, 13)
    for ser, col in ((N.CE_TH64, BL), (N.CE_TH128, OR)):
        for i in range(len(xs) - 1):
            f.line(X(i), Y(ser[i]), X(i + 1), Y(ser[i + 1]), col, 2.4, arrow=False)
        for i, v in enumerate(ser):
            _dot(f, X(i), Y(v), 6, col)
    i3 = xs.index(3)
    f.t(X(i3) - 16, Y(min(N.CE_TH_BEST)) + 6, "3 ms 最好：%.0f / %.0f" % N.CE_TH_BEST, GR, True, 15, "end")
    i1 = xs.index(1)
    f.t((X(i1 - 1) + X(i1)) / 2, Y(N.CE_TH64[i1]) - 22, "0.75–1 ms 平台", GY, True, 13, "middle")
    f.t(X(0) + 14, Y(N.CE_TH64[0]) + 5, "默认：短核都进来当搭档", RD, True, 13)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "门槛太低，耗时短的核回来当搭档；3 ms 刚好把它们全挡在外面；再高，连注意力核也挡掉", [])
    yb = f.src(yb + 10,
               "预测：客观判据按最终程序的指令顺序回放，主核每条指令用 profile 实测的时长，两颗小核各一条先来先做的车道跑实测的通信时长，算每条通信的括号和露出。在八格上校验过：gate/up 收集的预测露出 %.1f / %.0f / %.1f ms，对上卡实测 %.1f / %.1f / 不到 %d；整步排序两头对得上，中间有两处相邻互换；绝对值偏低约 %d%%，所以只用来排名，不报快多少。"
               % (N.CE_SIM_AG["ctl"][0], N.CE_SIM_AG["fk"][0], N.CE_SIM_AG["fk0"][0], N.CE_SIM_AG["ctl"][1], N.CE_SIM_AG["fk"][1], N.CE_SIM_AG["fk0"][1], N.CE_SIM_LOW_PCT),
               "4 层模型：1 层稠密 ＋ 3 层 MoE、视觉塔 2 层，反向层体的搭档跟 39 层完全相同，编一格约 %d 秒；每个点一格，只编不上卡。39 层复核（64 片拓扑）：3 ms ＋ ×0.32 %d，4 ms %d，3 ms 不调延迟 %d，推荐配置 %d，排序跟 4 层一致。128 片拓扑那条借用 64 片 profile 的通信时长表，形状可信，绝对值不可信。" % (N.CE_L4_S, N.CE_L39["3ms032"], N.CE_L39["4ms032"], N.CE_L39["3ms1"], N.CE_L39["v4"]),
               "每档门槛下专家权重收集的搭档（编译产物）：默认是融合核；0.2–0.5 ms 是 token 搬运核加矩阵乘；0.75–1 ms 是矩阵乘，跟推荐配置相同；1.5–2 ms 是矩阵乘加注意力反向；3 ms 起没有搭档，完成点贴着用数据的地方，只剩主干注意力那几颗大核给别的通信当搭档。照实报以后专家矩阵乘、专家权重梯度约 %.1f–%.1f ms，所以被 3 ms 挡在外面。换模型、换芯片数，大核的绝对时长会变，门槛要重扫。"
               % N.CE_TH_GMM_MS,
               "上卡：3 ms ＋ ×0.32 在同一台机器上 %.3f → %.3f 秒（%.1f%%），逐位相同；判据把它排第一（按判据的整步折算是 %.1f%%，判据只用来排名）。差的那截没有消失，搬到了小核车道上（见正文）。" % (N.CE_TH_ONCARD_CTL, N.CE_TH_ONCARD, N.CE_TH_PCT, N.CE_TH_PRED_PCT))
    f.save("fig8b-cethresh.svg", yb + 14)


if __name__ == "__main__":
    fig_ceratio()
    fig_cebracket()
    fig_cecells()
    fig_celayer()
    fig_cethresh()
