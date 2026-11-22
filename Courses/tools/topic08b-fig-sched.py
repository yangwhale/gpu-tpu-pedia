# -*- coding: utf-8 -*-
r"""专题九 · 第六节「同样的活，排法不同」的三张图。

fig8b-barrier   回答：一道屏障什么都没少算，为什么快 6.9%？—— 左：一块 7.1 GiB 的数据被提前产出、一路活到优化器段（示意）；
                右：反向循环里通信括号里夹的计算 801 → 1261，每层 79.7 → 70.3 毫秒，小核的活一点没少（实测）。
fig8b-phases    回答：快在一步的哪一段？—— 按时间段切开的五段，对照 vs 加屏障。
fig8b-fused     回答：一颗核快了，步时为什么反而慢？—— 那条梯度规约「放行到完成」之间：原来夹着 42 条指令（含注意力反向核），开核以后只剩 1 条。
"""
import topic08_figfont  # noqa: F401
from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE, LINE2, PU
Fig.BAND_KEEP_LINES = 0
import topic08b_numbers as N

W = 1400


def fig_barrier():
    f = Fig(W, "左边是示意：一步分四段，前向、反向、优化器。没有屏障时，调度器把优化器要用的一份参数收集提前到前向之前，"
               "收进来的 7.1 GiB 一直活到优化器段才用，整个前向和反向都占着显存。加一道屏障以后，这次收集挪回优化器段。"
               "右边是实测：反向循环里每条小核通信在放行和完成之间夹的计算从 801 个变成 1261 个，括号里一个计算都没有的从 18 条变成 1 条；"
               "反向每层从 79.7 毫秒变成 70.3 毫秒；小核 1 在反向里的忙时 1935.7 对 1936.0 毫秒，活一点没少。")
    y0 = f.header("一道屏障什么都没少算：腾出显存，调度器才敢把通信藏进计算里",
                  "左：一块数据活多久（示意）；右：反向循环里的四个量，每对两根条（实测，按各自较大的那根归一）",
                  [(PU, "被提前收进来的那块数据"), (GY2, "没有屏障"), (BL, "加一道屏障")])
    PH = 360
    py = f.panel(0, y0, W, PH, "对照 vs 加一道屏障（左示意，右实测）", BL)
    X0, SEG = 120, [("前向之前", 60), ("前向", 140), ("反向", 260), ("优化器", 120)]
    for k, (title, flag) in enumerate((("没有屏障", True), ("加一道屏障", False))):
        yy = py + 60 + k * 130
        f.t(30, yy + 22, title, INK, True, 15)
        x = X0 + 30
        for name, w in SEG:
            f.box(x, yy, w, 30, "#f1f3f4", GY2, 2)
            f.t(x + w / 2, yy + 20, name, GY, size=13, anchor="middle")
            x += w + 4
        if flag:
            f.box(X0 + 30, yy + 40, sum(w + 4 for _n, w in SEG) - 4, 22, PU, PU, 3)
            f.t(X0 + 40, yy + 56, "7.1 GiB：前向之前就收进来，活到优化器才用", "#fff", True, 13)
        else:
            x_opt = X0 + 30 + sum(w + 4 for _n, w in SEG[:3])
            f.box(x_opt, yy + 40, SEG[3][1], 22, PU, PU, 3)
            f.t(x_opt - 8, yy + 56, "挪回优化器段", PU, True, 13, "end")
    # 右：三个数
    RX = 780
    rows = [("反向里通信括号夹的计算", N.BAR_BRACKET, "个"), ("空括号（一个计算都没夹）", N.BAR_EMPTY, "条"),
            ("反向每层", N.BAR_LAYER, "ms"), ("小核 1 反向整段的忙时（没变）", N.BAR_SC1_BWD, "ms")]
    f.t(RX, py + 42, "反向循环（实测）", INK, True, 15)
    BW = 300
    for i, (lab, (a, b), unit) in enumerate(rows):
        yy = py + 70 + i * 66
        f.t(RX, yy + 4, lab, INK, True, 14)
        m = max(a, b)
        for j, (v, col) in enumerate(((a, GY2), (b, BL))):
            w = max(3, BW * v / m)
            f.box(RX, yy + 12 + j * 20, w, 16, col, col, 2)
            f.t(RX + w + 8, yy + 25 + j * 20, ("%g %s" % (v, unit)), INK if j else GY, j == 1, 13)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "活一样多，排法变了：小核的忙时一点没少，主核等得少了", [])
    yb = f.src(yb + 10,
               "实测（编译产物）：没有屏障时，一次优化器参数收集的结果（%.3f GiB）在前向循环之前就产出，直到优化器段的更新才用；反向循环之前产出、之后才用的数据合计 %.2f → %.2f GiB（−6.0），那一块在加屏障后的清单里不见了。" % ((N.BAR_HOIST_GIB,) + N.BAR_LIVE_GIB),
               "实测（编译产物）：同一个反向循环体，每条小核异步通信从放行到完成之间夹的融合计算和核合计 %d → %d 个；空括号 %d → %d 条。" % (N.BAR_BRACKET + N.BAR_EMPTY),
               "实测（profile 第 11 步）：反向每层 %.1f → %.1f ms；反向窗口里小核 1 忙 %.1f → %.1f ms，主核计算基本不变（±30 ms）。屏障只改调度，不改数值。" % (N.BAR_LAYER + N.BAR_SC1_BWD),
               "⚠️ 推断：调度器给反向循环排序时显存余量更紧，不敢把异步通信拉长（拉长要多扛住输出的缓冲），于是主核停在完成点上；只核了两头（跨反向占着的数据少约 6 GiB、括号变短），没读调度器源码。")
    f.save("fig8b-barrier.svg", yb + 14)


def fig_phases():
    f = Fig(W, "一步按时间段切成五段，每段一根条，画的是加屏障以后这一段变了百分之几：往左是变短，往右是变长。"
               "反向循环是一步里最长的一段，2950.0 变 2599.4 毫秒，少了约 12%，省下的时间几乎都在这里；前向之前那段本来很短，180.1 变 114.4，少约 37% 也只是一小截；"
               "前向循环 920.8 不变；前后向之间 142.2 变 153.7，多约 8%；优化器段 1055.6 变 1094.9，多约 4%。合计 5248.7 变 4883.2 毫秒。")
    y0 = f.header("快在哪一段：最长的反向循环少了 351 毫秒，其余几段增减都很小",
                  "条长 ＝ 这一段变了百分之几（左短右长）；条的粗细 ＝ 这一段在一步里大致多长；右边是这一段的毫秒（对照 → 加屏障）",
                  [(BL, "变短"), (OR, "变长")])
    PH = 60 + 58 * len(N.BAR_PHASES)
    py = f.panel(0, y0, W, PH, "一步按时间段切开（实测，profile 里的一步）", BL)
    XC, SC = 700, 9.0
    SIZE = {n: "%.1f → %.1f ms" % (a, b) for n, a, b in N.BAR_PHASES_MS}
    THICK = {"前向之前": 6, "前向循环": 20, "前后向之间": 6, "反向循环": 30, "优化器段": 14}
    f.line(XC, py + 34, XC, py + PH - 14, GY2, 1.2, arrow=False)
    for i, (name, pct) in enumerate(N.BAR_PHASES):
        yy = py + 40 + i * 58
        f.t(250, yy + 18, name, INK, True, 15, "end")
        if pct == 0:
            f.t(XC + 12, yy + 18, "不变", GY, True, 14)
        else:
            w = abs(pct) * SC
            x0 = XC - w if pct < 0 else XC
            col = (RD if name == "反向循环" else BL) if pct < 0 else OR
            th = THICK[name]
            f.box(x0, yy + 14 - th / 2, w, th, col, col, 2)
            lab = ("少约 %d%%" % -pct) if pct < 0 else ("多约 %d%%" % pct)
            if pct < 0:
                f.t(x0 - 10, yy + 20, lab, col, True, 14, "end")
            else:
                f.t(x0 + w + 10, yy + 20, lab, col, True, 14)
        f.t(1060, yy + 18, SIZE[name], RD if name == "反向循环" else GY, name == "反向循环", 14)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "被提前的那批优化器通信挪回优化器段，在那里跟优化器的计算叠着跑，只多 39 毫秒", [])
    yb = f.src(yb + 10,
               "实测：对照来自另一个作业的同配置（%.3f 秒），加屏障 %.3f 秒（−%.1f%%）；两份前向循环一样长，说明两次分配的硬件速度一样，跨作业比可以用。" % (N.BAR_CTL_S, N.BAR_S, N.BAR_PCT),
               "实测：前向之前那 66 ms 是同一件事的另一半：被提前的几条优化器收集原来挡在前向之前，加屏障后挪进优化器段。",
               "显存：加屏障后正常步峰值少约 %.0f GiB；腾出来的空间还让另一处 all-to-all 能交给小核（只多兑现 64 ms，见下面的小表）。" % N.BAR_MEM_SAVED)
    f.save("fig8b-phases.svg", yb + 14)


def fig_fused():
    f = Fig(W, "一层反向里的一条专家梯度规约，从放行到完成。上面是没开融合核：两点之间夹着 42 条指令，其中有注意力反向那个大核，通信大半在它底下跑完。"
               "下面开了融合核：完成点被提到注意力反向之前，中间只剩融合核那一条，主核每层等 10.1 毫秒变成 16.0 毫秒。核本身省了约 183 毫秒，调度多露了约 300 毫秒，一步净慢约 107 毫秒。")
    y0 = f.header("一颗核快了，步时反而慢：完成点被挪到了没东西盖住的地方",
                  "一层反向里同一条专家梯度规约；方块 ＝ 主核上的指令，红线 ＝ 从放行到完成",
                  [(BL, "注意力反向核"), (PU, "融合核"), (LINE2, "别的指令"), (RD, "主核停着等")])
    PH = 350
    py = f.panel(0, y0, W, PH, "同一层、同一条通信（实测）", BL)
    X0 = 200
    for k, (title, n_mid, wait) in enumerate((("没开融合核", N.FUSE_INSTR[0], N.FUSE_LAYER_WAIT[0]),
                                               ("开了融合核", N.FUSE_INSTR[1], N.FUSE_LAYER_WAIT[1]))):
        yy = py + 60 + k * 110
        f.t(30, yy + 22, title, INK, True, 15)
        x = X0
        f.t(x, yy - 10, "放行", RD, True, 13, "middle")
        f.line(x, yy - 2, x, yy + 34, RD, 1.6, arrow=False)
        x += 10
        if k == 0:
            for j in range(12):
                f.box(x, yy + 6, 14, 22, LINE2, GY2, 2)
                x += 18
            f.box(x, yy + 6, 220, 22, BL, BL, 2)
            f.t(x + 110, yy + 22, "注意力反向", "#fff", True, 13, "middle")
            x += 224
            for j in range(10):
                f.box(x, yy + 6, 14, 22, LINE2, GY2, 2)
                x += 18
            f.t(x + 10, yy + 22, "…共 %d 条" % n_mid, GY, size=13)
            x += 110
        else:
            f.box(x, yy + 6, 60, 22, PU, PU, 2)
            f.t(x + 30, yy + 22, "融合核", "#fff", True, 12, "middle")
            x += 64
        f.box(x, yy + 6, wait * 9, 22, RD, RD, 2)
        f.t(x + wait * 4.5, yy + 22, "等", "#fff", True, 13, "middle")
        x += wait * 9
        f.line(x, yy - 2, x, yy + 34, RD, 1.6, arrow=False)
        f.t(x, yy - 10, "完成", RD, True, 13, "middle")
        f.t(x + 12, yy + 22, "主核停着等 %.1f ms／层，通信才做完" % wait, RD, True, 14)
    f.t(30, py + 300, "核本身省约 %d ms；反向多露约 %d、前向约 %d、同步多等约 %d；一步净慢约 %d ms（%.3f → %.3f 秒）" % (
        N.FUSE_SAVE_MS, N.FUSE_BWD_MS, N.FUSE_FWD_MS, N.FUSE_AR_MS, N.FUSE_NET_MS, N.FUSE_CTL, N.FUSE_S), INK, True, 14)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "核是不透明的一整块，调度器挪动了它周围的通信；先看完成点排在哪，再看核快不快", [])
    yb = f.src(yb + 10,
               "实测（编译产物与 profile，同一分配两格）：对照里这条梯度规约放行到完成之间夹 %d 条指令（含注意力反向核）；开核以后只夹融合核 1 条，每层等待 %.1f → %.1f ms，各层合计约 +%d ms；前向一条权重收集的完成点排到矩阵乘前面，约 +%d ms；同步 all-reduce 多等约 +%d ms。" % (N.FUSE_INSTR[0], N.FUSE_LAYER_WAIT[0], N.FUSE_LAYER_WAIT[1], N.FUSE_BWD_MS, N.FUSE_FWD_MS, N.FUSE_AR_MS),
               "实测：核本身是真核（编译产物里是自定义调用，没有退化成循环），这一摊计算加布局重排每步少约 %d ms；小核两条车道的总活没变。" % N.FUSE_SAVE_MS,
               "实测：同一颗核在 20 层底座上是误差内（%.1f%%）：换了条件，结论就要重测。调块大小、只开一半都救不回来；要改的是调度，不是核。" % N.FUSE_OTHER_PCT)
    f.save("fig8b-fused.svg", yb + 14)


if __name__ == "__main__":
    fig_barrier()
    fig_phases()
    fig_fused()
