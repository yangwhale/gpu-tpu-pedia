# -*- coding: utf-8 -*-
r"""专题九 · 第十节「快了，还是同一个模型吗：准入」的图（2026-11-22 补）。

fig8b-race   回答：同一份程序跑出不同结果，怎么找是谁引起的？—— 开关 × 次数：
             Ⓐ 一份早期程序第 1 步就 NaN，每格只改一样、跑 3 次，只有「优化器的 all-to-all 交给小核」和「加一行日志」让它消失；
             Ⓑ 当前程序 30 步，同一份程序反复跑、跟第一次逐位比：对照 10/10、换切法 17/17 相同；去掉两个 all-to-all 开关 9 次 3 次分叉；
               一个转置开关单独就 12 次 1 次分叉。
"""
import topic08_figfont  # noqa: F401
from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE, LINE2, PU
Fig.BAND_KEEP_LINES = 0
import topic08b_numbers as N

W = 1400


def _dots(f, x0, y, n, bad, badcol, step=27, d=18):
    for i in range(n):
        col = badcol if i >= n - bad else GR
        f.box(x0 + i * step, y, d, d, col, col, 9)
    return x0 + n * step


def fig_race():
    nan_desc = "，".join("%s %d 次里 %d 次 NaN" % (n, k, b) for n, k, b in N.RACE_NAN)
    det_desc = "，".join("%s %d 次里 %d 次分叉" % (n, k, b) for n, k, b in N.RACE_DET)
    f = Fig(W, "开关乘次数。上半：一份早期程序第 1 步就出 NaN，每一格只改一样东西、重编以后跑几次：" + nan_desc + "。"
               "下半：当前程序跑 30 步，同一份编好的程序反复跑，跟第一次逐位比：" + det_desc + "。分叉的那几次彼此也不一样。")
    y0 = f.header("同一份程序，结果不止一种：一格只改一样，每格多跑几次",
                  "每个圆点 ＝ 跑一次；绿 ＝ 正常（上）或跟第一次逐位相同（下），红 ＝ 出 NaN，橙 ＝ 从某一步起分叉",
                  [(GR, "正常／逐位相同"), (RD, "NaN"), (OR, "分叉")])
    LX, DX = 470, 500
    SHOW = [r for r in N.RACE_NAN if r[0] in ("原样", "优化器的 all-to-all 交给小核")]
    REST = [r for r in N.RACE_NAN if r not in SHOW]
    PH1 = 50 + 44 * len(SHOW) + 30
    py = f.panel(0, y0, W, PH1, "Ⓐ 一份早期程序，第 1 步就 NaN：每格只改一样（实测）", RD)
    for i, (n, k, b) in enumerate(SHOW):
        yy = py + 22 + i * 44
        f.t(LX, yy + 15, n, INK, True, 15, "end")
        xe = _dots(f, DX, yy, k, b, RD)
        f.t(DX + 9 * 27 + 20, yy + 15, "%d / %d 次 NaN" % (b, k), RD if b else GR, True, 15)
    f.t(LX, py + 22 + len(SHOW) * 44 + 15, "另外 %d 格见出处：加一行日志也好了，3 个开关还是 3 / 3 次 NaN" % len(REST), GY, True, 14, "end")
    y1 = py + PH1 + 18
    PH2 = 50 + 44 * len(N.RACE_DET)
    py2 = f.panel(0, y1, W, PH2, "Ⓑ 当前程序 %d 步：同一份程序反复跑，跟第一次逐位比（实测）" % N.RACE_STEPS, OR)
    for i, (n, k, b) in enumerate(N.RACE_DET):
        yy = py2 + 22 + i * 44
        f.t(LX, yy + 15, n, INK, True, 15, "end")
        _dots(f, DX, yy, k, b, OR)
        f.t(DX + 18 * 27 + 20, yy + 15, "%d / %d 次分叉" % (b, k), OR if b else GR, True, 15)
    f._pan = None
    yb = f.band(py2 + PH2 + 20, "ok", "同一份程序跑出两种结果，就是竞争；哪个开关让它出现、哪个让它消失，靠一格一格数次数", [])
    yb = f.src(yb + 10,
               "实测（Ⓐ）：那份早期程序（测试底座），原样跑 3 次，3 次都在第 1 步报 NaN（第 0 步那次更新以后参数就坏了）。每格在它上面只加一个开关重编，跑 3 次。"
               "「加一行日志」不改计算，只让编译器排出的顺序变了，3 次全好、而且逐位相同：这本身就说明它跟顺序有关。",
               "实测（Ⓑ）：同一份编好的程序、同一批数据，每次跑 %d 步，所有指标跟第一次逐位比。转置开关那 1 次从第 %d 步起分叉；换切法 ＋ 转置开关的 4 次分在第 9、11、17、17 步，"
               "第 17 步那两次的值也互不相同。去掉两个 all-to-all 开关的 3 次从第 21 步起分叉，结果只有两种；两个都没有 NaN。" % (N.RACE_STEPS, N.RACE_FIRST_DIV),
               "转置开关单独 %d 次里 %d 次、跟换切法一起 %d 次里 %d 次，这个样本量下分不出差别，不能说换切法会放大它。" % (N.RACE_DET[3][1], N.RACE_DET[3][2], N.RACE_DET[4][1], N.RACE_DET[4][2]),
               "⚠️ 推断：最可能的竞争对是静态分析找到的：主核上同步执行的一条 all-to-all，它的输出放进了一块刚被释放的内存，"
               "那块内存上一个主人的最后一次读，就排在这条 all-to-all 紧前面；别的设备先跑到，把数据写进来，本机可能还在读。没有在硬件上直接证实。")
    f.save("fig8b-race.svg", yb + 14)


def fig_init():
    """fig8b-init  回答：第 0 步就不同，差在哪？—— 示意：同一个种子，按切分各自生成时，切法一变，拼出来的初始权重就变了；
    按全局位置生成（跟切分无关）时，怎么切都是同一份。方块颜色是示意，两个 loss 是实测。"""
    import hashlib
    def rnd(*k):
        return int(hashlib.sha256(repr(k).encode()).hexdigest()[:6], 16) / 0xFFFFFF
    PAL = ["#e8f0fe", "#c6dafc", "#8ab4f8", "#4285f4", "#1967d2"]
    def col(v):
        return PAL[min(4, int(v * 5))]
    f = Fig(W, "示意：一块 4 乘 8 的初始权重，同一个随机种子。左边按切分各自生成：每个设备只按自己那一份的局部位置出随机数，"
               "切法甲按行切成 4 份、切法乙按列切成 4 份，拼起来的整块不一样，第 0 步 loss 就差 %s。"
               "右边按全局位置生成，跟怎么切无关，两种切法拼出来是同一块，第 0 步、第 1 步 loss 逐位相同。" % N.INIT_RBG_D_T)
    y0 = f.header("第 0 步就不同：差在初始权重怎么生成，不在计算",
                  "每一格 ＝ 一个初始权重值，颜色深浅 ＝ 值的大小（示意）；粗框 ＝ 一个设备分到的那一份",
                  [(BL, "初始权重（示意）")])
    PH = 330
    py = f.panel(0, y0, W, PH, "同一个随机种子，两种切法（方块是示意，loss 是实测）", BL)
    CW, R, C = 30, 4, 8
    def grid(x0, y0, mode, split):
        for r in range(R):
            for c in range(C):
                if split == "row":
                    dev, lr, lc = r, 0, c
                else:
                    dev, lr, lc = c // 2, r, c % 2
                v = rnd("dev", dev, lr, lc) if mode == "local" else rnd("glob", r, c)
                f.box(x0 + c * CW, y0 + r * CW, CW - 2, CW - 2, col(v), "#fff", 2)
        if split == "row":
            for r in range(R):
                f.box(x0 - 2, y0 + r * CW - 2, C * CW + 2, CW + 2, "none", INK, 2, sw=1.6)
        else:
            for d in range(4):
                f.box(x0 + d * 2 * CW - 2, y0 - 2, 2 * CW + 2, R * CW + 2, "none", INK, 2, sw=1.6)
    cols = [("按切分各自生成（默认，快）", "local", N.INIT_RBG, RD, "第 0 步 loss 就差 " + N.INIT_RBG_D_T),
            ("按全局位置生成（跟切分无关）", "glob", (N.INIT_TF, N.INIT_TF), GR, "第 0、1 步逐位相同")]
    for k, (title, mode, losses, c2, verdict) in enumerate(cols):
        X = 60 + k * 680
        f.t(X, py + 50, title, INK, True, 16)
        for j, (sp, lab) in enumerate((("row", "切法甲（按行切 4 份）"), ("col", "切法乙（按列切 4 份）"))):
            gx = X + j * 300
            f.t(gx, py + 82, lab, GY, True, 13)
            grid(gx, py + 94, mode, sp)
            f.t(gx, py + 94 + R * CW + 26, ("第 0 步 loss：基准" if j == 0 else ("第 0 步 loss：＋%s" % N.INIT_RBG_D_T)) if mode == "local" else "第 0 步 loss：逐位相同", c2, True, 14)
        f.t(X, py + 94 + R * CW + 62, verdict, c2, True, 16)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "起点不一样，后面的差就分不清是改动的还是初始化的", [])
    yb = f.src(yb + 10,
               "⚠️ 示意：方块大小、切法和颜色都是画出来说明道理的，不是真实权重。",
               "📌 实测：同一批数据、同一个随机种子，只换专家权重的切法；默认的随机数生成器按每个设备自己那一份生成，第 0 步 loss 差 %s；换成跟切分无关的生成方式后，两边第 0、1 步 loss 和梯度范数逐位相同。" % N.INIT_RBG_D_T)
    f.save("fig8b-init.svg", yb + 14)


if __name__ == "__main__":
    fig_race()
    fig_init()
