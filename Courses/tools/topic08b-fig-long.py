# -*- coding: utf-8 -*-
r"""专题九 · 第九节「换到长文」的两张新图（2026-11-22 补）。

fig8b-muon    回答：优化器里一步多搬的是什么？—— 一块专家权重在优化器里的三次搬运（按「份」：BF16 动量 1 份，FP32 主权重 2 份）：
              原来 BF16 动量过去、FP32 主权重过去、新主权重回来（5 份）；修好以后 BF16 过去、BF16 回来（2 份），在参数自己那边相加。
fig8b-agwin   回答：一颗核让主核少算 190 毫秒，一步为什么慢 7.8%？—— 反向重算段里三条专家权重收集「放行到完成」之间夹的计算：
              7.2 / 9.0 / 15.2 毫秒 → 0.4 / 0.7 / 0.7 毫秒，每条露出 3.6–3.9 毫秒，各层都这样。
"""
import topic08_figfont  # noqa: F401
from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE, LINE2, PU
Fig.BAND_KEEP_LINES = 0
import topic08b_numbers as N

W = 1400


def fig_muon():
    f = Fig(W, "一块专家权重（gate、up、down 三块之一）在优化器里每步的搬运，每设备。左边是参数自己的切法，FP32 主权重放在这里；"
               "右边是正交化的切法，每个设备拿整块矩阵做五轮迭代（每轮几次矩阵乘）。以一块的 BF16 动量为 1 份：原来 BF16 动量搬过去 1 份，FP32 主权重也搬过去 2 份，"
               "在右边加完，新的 FP32 主权重再搬回来 2 份，一块合计 5 份。修好以后：BF16 动量过去 1 份，BF16 结果回来 1 份，"
               "在左边加到主权重上，一块合计 2 份，是原来的五分之二。1 份是每设备 1.78 GiB：一块 8.9 GiB 变 3.56 GiB，三块每步 26.7 变 10.7 GiB。一步 4.673 变 4.377 秒，快 6.3%；峰值显存 89.17 变 81.89 GiB。")
    y0 = f.header("优化器里白搬的字节：FP32 主权重被搬过去、加完又搬回来",
                  "箭头 ＝ 一次 all-to-all（换一种切法），线越粗字节越多；一块专家权重、一步、每设备，以 BF16 动量为 1 份（＝ 1.78 GiB）",
                  [(BL, "BF16 动量"), (RD, "FP32 主权重（白搬）"), (GR, "BF16 结果")])
    PH = 500
    py = f.panel(0, y0, W, PH, "一块专家权重在优化器里的一步（离线编译产物逐条追，实测）", BL)
    LX, LW, RX, RW = 40, 300, 860, 300
    for k, (title, arrows, add_left, tot) in enumerate((
            ("原来", [("→", N.MU_MOM_U, BL, "BF16 动量"), ("→", N.MU_PARAM_U, RD, "FP32 主权重"),
                      ("←", N.MU_PARAM_U, RD, "加完的 FP32 新主权重")], False, N.MU_LEAF_BEFORE),
            ("修好以后", [("→", N.MU_MOM_U, BL, "BF16 动量"), ("←", N.MU_MOM_U, GR, "BF16 结果")], True, N.MU_LEAF_DESIGN))):
        yy = py + 34 + k * 200
        f.t(LX, yy + 18, title, INK, True, 16)
        bh = 150
        f.box(LX, yy + 30, LW, bh, "#f8f9fa", GY2, 8)
        f.t(LX + LW / 2, yy + 58, "参数自己的切法", INK, True, 15, "middle")
        f.t(LX + LW / 2, yy + 82, "FP32 主权重放在这里", GY, size=13, anchor="middle")
        f.box(RX, yy + 30, RW, bh, "#f8f9fa", GY2, 8)
        f.t(RX + RW / 2, yy + 58, "正交化的切法", INK, True, 15, "middle")
        f.t(RX + RW / 2, yy + 82, "每个设备拿整块矩阵", GY, size=13, anchor="middle")
        f.t(RX + RW / 2, yy + 104, "迭代五轮，结果 BF16", GY, size=13, anchor="middle")
        # 「在哪边相加」
        ax = (LX + LW / 2) if add_left else (RX + RW / 2)
        f.box(ax - 110, yy + 124, 220, 36, "#fff", GR if add_left else RD, 6, sw=1.6)
        f.t(ax, yy + 148, "在这边加到主权重上", GR if add_left else RD, True, 14, "middle")
        # 箭头
        x1, x2 = LX + LW + 14, RX - 14
        n = len(arrows)
        for j, (d, gib, col, lab) in enumerate(arrows):
            ay = yy + 62 + j * (90 / max(n - 1, 1)) if n > 1 else yy + 100
            sw = gib * 1.78 * 2.4          # 线宽按份数，跟原来的绝对刻度一样粗
            hd = 16                      # 箭头尖固定大小（marker 会跟线宽一起放大）
            if d == "→":
                f.line(x1, ay, x2 - hd, ay, col, sw, arrow=False)
                f.poly([(x2 - hd - 2, ay - 9), (x2, ay), (x2 - hd - 2, ay + 9)], col)
            else:
                f.line(x2, ay, x1 + hd, ay, col, sw, arrow=False)
                f.poly([(x1 + hd + 2, ay - 9), (x1, ay), (x1 + hd + 2, ay + 9)], col)
            f.t((x1 + x2) / 2, ay - 10, "%s　%d 份（%.2f GiB）" % (lab, gib, gib * N.MU_MOM_GIB), col, True, 14, "middle")
        f.t(RX + RW + 24, yy + 90, "一块 %d 份" % tot, RD if tot > 4 else GR, True, 16)
        f.t(RX + RW + 24, yy + 114, "%.2f GiB" % (tot * N.MU_MOM_GIB), RD if tot > 4 else GR, True, 14)
    f.t(LX, py + PH - 50, "三块（gate、up、down）合计每步 %.1f → %.1f GiB（设计的 %.1f 倍）；一步 %.3f → %.3f 秒（−%.1f%%）；峰值显存 %.2f → %.2f GiB" % (
        (N.MU_STEP_BEFORE_GIB, N.MU_STEP_DESIGN_GIB, N.MU_RATIO, N.MU_CTL, N.MU_S, N.MU_PCT) + N.MU_MEM), INK, True, 15)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "算在哪一边由编译器挑：结果没钉回参数自己的切法，它就把 FP32 主权重搬过去、加完再搬回来", [])
    yb = f.src(yb + 10,
               "实测（离线编译产物）：沿操作数往上追，那两条 FP32 的 all-to-all，一条的输入就是参数本身，另一条在相加之后把新参数送回来。"
               "小核上每条 FP32 的 %.1f ms、BF16 的 %.1f ms；三块九条合计 %.1f ms，主核停着等其中 %.1f ms（profile 第 11 步）。" % (N.MU_SC_MS[1], N.MU_SC_MS[0], N.MU_A2A_MS, N.MU_A2A_WAIT_MS),
               "实测（离线编译产物，整步，每设备）：all-to-all 字节 %.2f → %.2f GiB，其中 FP32 的（38 条，含专家以外的权重）%.2f → %.0f；FP32 的 all-gather %.2f → %.2f GiB（另一类按头切开的叶子原来用「全收再切」搬 FP32，合计约 %d GiB，一并去掉）。" % (N.MU_A2A_STATIC + N.MU_F32_A2A + N.MU_F32_AG + (N.MU_GATE_AG_GIB,)),
               "实测（同一作业两格，第 5–9 步）：%.3f → %.3f 秒。修法：把每块参数自己的切法传给优化器，BF16 结果先钉回这个切法，再转 FP32、再相加；正交化本身一位没动。" % (N.MU_CTL, N.MU_S),
               "实测：第 0、1 步 loss 逐位相同，从第 2 步起两边末位不同（正式修法差约 5e-4；站 19 那版局部修法第 2 步 1.1e-4）；按「不改算法的性能改动不要求逐位」接受（第十节）。")
    f.save("fig8b-muon.svg", yb + 14)


def fig_agwin():
    f = Fig(W, "反向重算段里，一层要把 gate、up、down 三块专家权重重新收集进来。每一行从放行画到完成。"
               "对照：三条收集的括号里分别夹着 7.2、9.0、15.2 毫秒的主核计算，收集藏在底下。加了融合核：括号里只剩 0.4、0.7、0.7 毫秒，"
               "主核每条停下来等 3.6 到 3.9 毫秒，各层都这样。主核计算少了约 190 毫秒，专家权重收集的露出从 64 变成 560 毫秒，一步 4.351 变 4.692 秒，慢 7.8%。"
               "把这类收集的延迟估计调大一倍，括号里的核从 1、2、3 个变成 2、3、5 个，原来是 4、5、6 个，342 毫秒只救回 63。")
    y0 = f.header("省了计算，收集露出来：放行被排到紧挨着完成",
                  "每一行 ＝ 一条专家权重收集，从放行（左红线）到完成（右红线）；蓝 ＝ 括号里夹的主核计算，红 ＝ 主核停着等；各层取中位",
                  [(BL, "括号里的主核计算"), (OR, "收集（长度示意）"), (RD, "主核停着等")])
    PH = 535
    py = f.panel(0, y0, W, PH, "反向重算段的一层（profile 各层中位 ＋ 编译产物，实测）", BL)
    X0, SC = 300, 40.0
    names = ("gate", "up", "down")
    ex = sum(N.IH_EXPOSE_EACH) / 2
    for k, (title, wins, expose) in enumerate((("对照", N.IH_WIN_MS[0], False), ("加了融合核", N.IH_WIN_MS[1], True))):
        yb0 = py + 44 + k * 200
        f.t(30, yb0 + 14, title, INK, True, 16)
        for j, (nm, w) in enumerate(zip(names, wins)):
            yy = yb0 + 34 + j * 54
            f.t(X0 - 30, yy + 18, nm, INK, True, 15, "end")
            f.line(X0, yy - 2, X0, yy + 34, RD, 1.6, arrow=False)
            f.box(X0 + 2, yy + 2, w * SC, 18, BL, BL, 2)
            x = X0 + 2 + w * SC
            if expose:
                f.box(x, yy + 2, ex * SC, 18, RD, RD, 2)
                x += ex * SC
                f.box(X0 + 2, yy + 23, (w + ex) * SC - 4, 7, OR, OR, 2)
            else:
                f.box(X0 + 2, yy + 23, min(w * SC * 0.45, 150), 7, OR, OR, 2)
            f.line(x + 2, yy - 2, x + 2, yy + 34, RD, 1.6, arrow=False)
            lab = ("夹着 %.1f ms 计算，收集藏住" % w) if not expose else ("只夹 %.1f ms，停下来等 %.1f–%.1f ms" % ((w,) + N.IH_EXPOSE_EACH))
            f.t(x + 16, yy + 18, lab, RD if expose else BL, True, 14)
    yt = py + 460
    f.t(30, yt, "整步：主核计算少约 %d ms，专家权重收集露出 %d → %d ms；只加融合核，一步 %.3f → %.3f 秒（+%.1f%%）" % (
        (N.IH_COMPUTE_SAVED,) + N.IH_AG_EXPOSED + (N.IH_CTL, N.IH_S, N.IH_PCT)), RD, True, 15)
    f.t(30, yt + 36, "把这类收集的延迟估计调大一倍：括号里的核 %d/%d/%d → %d/%d/%d 个（对照是 %d/%d/%d）；%.3f → %.3f 秒，%d 毫秒只救回 %d" % (
        N.IH_WIN_KERN[1] + N.IH_WIN_KERN[2] + N.IH_WIN_KERN[0] + N.IH_KNOB + (N.IH_KNOB_GAP, N.IH_KNOB_BACK)), GY, True, 15)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "计算变短了，原来盖着的通信就露出来：省了计算，还要看放行和完成之间还剩什么", [])
    yb = f.src(yb + 10,
               "实测（profile，一颗芯片的一个主核，各层取中位；拆的是融合核另带一项小改动的那一格，%.3f 秒、+7.9%%，那项小改动单开只慢 1.7%%）：三条重新收集的括号里夹的主核计算 %.1f / %.1f / %.1f → %.1f / %.1f / %.1f ms；设备一步 %d → %d ms，主核计算少约 %d ms。" % ((N.IH_KNOB[0],) + N.IH_WIN_MS[0] + N.IH_WIN_MS[1] + N.IH_DEV + (N.IH_COMPUTE_SAVED,)),
               "实测（编译产物，指令顺序）：括号里夹的核 %d/%d/%d → %d/%d/%d 个。融合核本身是真核。小核上集合通信的总忙时只多约 88 ms：活没多，是放行的位置变了。" % (N.IH_WIN_KERN[0] + N.IH_WIN_KERN[1]),
               "实测：对照在三个作业里是 4.347 / 4.347 / %.3f 秒；换一个底座重测，%.3f → %.3f 秒，也是 +7.8%%。显存不是原因（带融合核那一格峰值反而低约 4 GiB）。" % (N.IH_CTL, N.IH_CTL2, N.IH_S2),
               "原因见 9.5：融合核报了耗时，就成了层调度器给收集挑的搭档，收集的完成点被钉在它旁边；在定稿之后的配置上去掉这颗融合核的估计，括号恢复、一步快 3.8%（实测）。这一格本身没有单独去掉估计重测，按同一条规则解释。限制同时在飞的收集条数，两种取值编出来跟原来一字不差（空操作）。")
    f.save("fig8b-agwin.svg", yb + 14)


if __name__ == "__main__":
    fig_muon()
    fig_agwin()
