# -*- coding: utf-8 -*-
r"""专题九 · 第三节「拿参照比」的三张图。

fig8b-ladder    回答：别人 49%、我们 11%，差在哪？—— 公开模型的配方复现到 README 的水平，再每次只改一个条件往我们这边退。
                每一行是一步：左边前后两格的实测步时，右边这一步的乘数；最后一行是乘起来（本课推导，假设互不影响）。
fig8b-selfslow  回答：同样条件下，本讲的模型自己多花在哪、占便宜在哪？—— 发散条，按关键路径修正后的毫秒（推断）。
fig8b-busiest   回答：只录一颗芯片时，「注意力核慢 2.3 倍」是怎么读出来的？—— 两颗芯片这一步各算了多少块、每块多快。
"""
import topic08_figfont  # noqa: F401
from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE, LINE2, PU
Fig.BAND_KEEP_LINES = 0
import topic08b_numbers as N

W = 1400


def fig_ladder():
    f = Fig(W, "公开模型 DeepSeek V3 在 64 片 v7x上的参照阶梯。公开配方 README 报 4.24 秒、48.85%，我们原样跑 5.473 秒，改两个开关后 4.133 秒。"
               "之后每次只改一个条件、前后两格都实测：FP8 改 BF16 慢 1.34 倍；上游专用实现换成通用实现慢 1.49 倍；"
               "序列从 4K 拉到 32K 加上下文并行慢 1.41 倍；加 FP32 精度开关（几处中间结果按 32 位算）慢 1.11 倍；AdamW 换 Muon 慢 1.19 倍。"
               "五项相乘约 3.7 倍（本课推导，假设互不影响）。")
    y0 = f.header("别人 49%、我们 11%：先把别人的配方搬过来，再一个条件一个条件退到我们这边",
                  "每一行只改一个条件；条长 ＝ 这一步把步时拉长了几倍（对数刻度：长度能首尾相接，接起来就是相乘）",
                  [(RD, "这一步把步时拉长的倍数"), (GY2, "量在哪一格")])
    import math
    n = len(N.LADDER_STEPS)
    PH = 110 + 58 * (n + 1)
    py = f.panel(0, y0, W, PH, "DeepSeek V3，64 片 v7x（实测；最后一行是推导）", BL)
    f.t(30, py + 52, "起点：公开配方复现 %.3f 秒（README 报 %.2f 秒、%.2f%%）；第一行从关掉 BF16 中间结果传播的 4.198 秒那格起" % (N.UP_REPRO_S, N.UP_README_S, N.UP_README_MFU), INK, True, 15)
    X0, U = 520, 520.0          # 每个 ln 单位 520 px
    xs, segs = X0, []
    for i, ((name, a, b, where), x) in enumerate(zip(N.LADDER_STEPS, N.LADDER_X)):
        yy = py + 80 + i * 58
        f.t(30, yy + 16, name, INK, True, 15)
        f.t(30, yy + 36, where, GY, size=12)
        w = math.log(x) * U
        f.box(xs, yy + 4, w, 26, RD, RD, 3)          # 每一行从上一行的末端接着画：越往下越靠右，接起来就是乘积
        f.t(xs + w + 10, yy + 23, "×%.2f" % x, RD, True, 16)
        segs.append((xs, w))
        xs += w
    yy = py + 80 + n * 58
    f.line(30, yy - 6, W - 30, yy - 6, LINE, 1, arrow=False)
    f.t(30, yy + 26, "五项相乘（本课推导，假设互不影响）", INK, True, 15)
    for k, (x0, w) in enumerate(segs):
        f.box(x0, yy + 8, w, 26, RD if k % 2 == 0 else "#f28b82", "#fff", 2)
    f.t(xs + 10, yy + 27, "约 ×%.1f" % N.LADDER_PROD, RD, True, 18)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "最大的几刀都是条件：精度、实现、序列长度、优化器（结构有没有影响，看 4.2）", [])
    _rows = "；".join("%s %.3f → %.3f 秒" % (name, a, b) for name, a, b, _w in N.LADDER_STEPS)
    yb = f.src(yb + 10,
               "📌 实测：每一行前后两格的步时：" + _rows + "。",
               "📌 公开锚点：上游 MaxText 公开仓库一个分支里 DeepSeek V3 专用实现的 README：v7x、全量 FP8、序列 8192、C4 数据，稳态 4.24 秒、48.85%（它自己的 MFU 口径，分母按 BF16 峰值）。",
               "📌 实测（本课程作者，同样 64 片 v7x，合成数据）：上游 8K 配方原样跑 %.3f 秒；关掉 BF16 中间结果传播（上游 16K 配方里用的开关）%.3f 秒；再把一项流水线内存开关设成上游 C4 配方的取值 %.3f 秒（同口径 50.15%%，比 README 快 2.5%%）。FP8 → BF16 那一步从 %.3f 秒那格起。" % (N.UP_RAW_S, N.UP_NOBF16P_S, N.UP_REPRO_S, N.UP_NOBF16P_S),
               "⚠️ 原样跑比 README 慢约 29%，原因没定位（推断的来源：运行库版本、合成数据对 C4、显存峰值贴近上限）。",
               "📌 实测：后四步在 DeepSeek V3 截成 16 层的格子上量（每设备 4096 token）：专用实现 1.149 → 通用实现 1.712 → 序列 32K ＋ 上下文并行 8 是 2.418；FP32 开关和 Muon 各自单加在 2.418 那格上。",
               "⚠️ 每设备 token 从 8192 减半那一步没画：它改的是每步的活，不是条件的代价（MFU 35.98% → 26.06%）。",
               "⚠️ 推导：五项（各取两位小数）相乘 %.2f；各项是在不同的格子上量的，相乘假设互不影响，只作量级示意。" % N.LADDER_PROD)
    f.save("fig8b-ladder.svg", yb + 14)


def fig_selfslow():
    f = Fig(W, "同样条件下，本讲的模型比 DeepSeek V3 多花和省下的时间，两边同样层数，按最忙的设备修正以后的估计。"
               "多花的：iHC 4 路残差（含每层一次同步 all-reduce）约 248 毫秒，视觉塔约 118，注意力上的数据因素约 116，优化器约 83。"
               "省下的：隐藏维更窄，专家矩阵乘和胶水约 180；对方那条代码路径每层有一次负载均衡的同步，约 221。"
               "最底下一行把账闭上：列出的几项合计我们多花约 164 毫秒，两边总差是我们快约 370 毫秒，所以其余各类合起来我们省约 534 毫秒（推算：其余类别没有逐项拆）。")
    y0 = f.header("同样条件下，本讲的模型自己多花在哪、省在哪",
                  "条长 ＝ 一步里比 DeepSeek V3 多（右，红）或少（左，绿）多少毫秒；两边同样层数，按最忙的设备修正（推断）",
                  [(RD, "本讲的模型多花"), (GR, "本讲的模型省下")])
    _listed = sum(ms for _n, ms in N.SELF_SLOW)
    _rest = N.SAME_TOTAL_DIFF_MS - _listed
    assert _listed == 164 and _rest == -534
    rows = list(N.SELF_SLOW) + [("其余（含两个窗口之差，没逐项拆）", _rest)]
    PH = 110 + 50 * len(rows)
    py = f.panel(0, y0, W, PH, "同样层数，同一步 profile 逐类相减（推断）", BL)
    XC, SC = 760, 1.1
    f.line(XC, py + 40, XC, py + PH - 56, GY2, 1.2, arrow=False)
    for i, (name, ms) in enumerate(rows):
        yy = py + 46 + i * 50
        col = RD if ms > 0 else ("#e8eaed" if i == len(rows) - 1 else GR)
        x0, x1 = (XC, XC + ms * SC) if ms > 0 else (XC + ms * SC, XC)
        f.box(x0, yy, x1 - x0, 30, col, col, 3)
        if ms > 0:
            f.t(XC - 16, yy + 21, name, INK, True, 15, "end")
            f.t(x1 + 10, yy + 21, "约 ＋%d ms" % ms, col, True, 15)
        else:
            f.t(XC + 16, yy + 21, name, INK, True, 15)
            f.t(x0 - 10, yy + 21, "约 %d ms" % ms, col, True, 15, "end")
    f.t(30, py + PH - 34, "账要闭合：上面几项合计 ＋%d ms；两边总差 %d ms（本讲的模型快）；其余各类合计约 %d ms（没逐项拆，总差减出来的）"
        % (_listed, N.SAME_TOTAL_DIFF_MS, _rest), GR, True, 15)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "注意力那一项不是核慢，是数据：几乎每步都有一行接近满长的文档", [])
    yb = f.src(yb + 10,
               "推断：「其余」是总差减出来的，混了第 11 步 profile 和第 5–9 步均值两个窗口；多步平均是本讲模型快；单看第 11 步，两边设备步只差十几毫秒。",
               "📌 实测：两边同一份代码、同一套开关，只有 3 个非结构差异（一边的矩阵乘分块启发式、softmax 缩放的精度、路由是否屏蔽填充）和数据管线不同；同一步 profile 按类别逐项相减。",
               "⚠️ 推断：两份 profile 都只录了一颗芯片，而两边这颗芯片一个是注意力最重的设备、一个是轻的（下一张图），所以按最重的设备修正；修正后的数都是估计。",
               "📌 注意力那一项：本讲模型的预打包数据 13 步里 11 步有一行接近满长的文档，最忙的设备算 66 个子块（平均约 60）；对方的文本数据平均约 50。这是数据，不是核（第五节从这里接着查）。")
    f.save("fig8b-selfslow.svg", yb + 14)


def fig_busiest():
    f = Fig(W, "只录了一颗芯片时的两组读数。本讲模型那颗芯片这一步算了 66 个注意力子块，前向 9.31 毫秒；"
               "DeepSeek V3 那颗芯片只算了 8 个子块，前向 2.37 毫秒，而它那一步最忙的设备算了 46 个。"
               "拿耗时直接比会读成我们慢；按各自执行的块和运算量折算，我们的速度约是对方的 1.7 倍，反而快。")
    y0 = f.header("只录一颗芯片：它这一步算了多少块，决定了你读到什么",
                  "左：这颗芯片这一步算了多少个注意力子块（满因果每设备 66 个）；右：按这颗芯片自己执行的块折算的速度",
                  [(BL, "本讲的模型"), (PU, "DeepSeek V3"), (LINE2, "对方那一步最忙的设备")])
    PH = 300
    py = f.panel(0, y0, W, PH, "同一步、各自 profile 里的那一颗芯片（实测）", BL)
    # 左：块数
    X0, SC = 230, 6.0
    rows = [("本讲的模型", N.BUSY_OURS_BLOCKS, BL, "前向 %.2f ms" % N.FWD_MS_OURS),
            ("DeepSeek V3", N.BUSY_THEIRS_BLOCKS, PU, "前向 %.2f ms" % N.FWD_MS_THEIRS),
            ("对方最忙的设备", N.BUSY_THEIRS_MAX, LINE2, "（没录到）")]
    f.t(30, py + 54, "这颗芯片这一步算了几个子块", INK, True, 15)
    for i, (lab, v, col, tag) in enumerate(rows):
        yy = py + 76 + i * 56
        f.t(X0 - 14, yy + 22, lab, INK, True, 15, "end")
        f.box(X0, yy, v * SC, 32, col, GY2 if col == LINE2 else col, 3)
        f.t(X0 + v * SC + 10, yy + 22, "%d 块 · %s" % (v, tag), INK, True, 14)
    # 右：每秒运算
    X1, SC1 = 990, 0.42
    f.t(820, py + 54, "按它自己算的块折算（万亿次/秒）", INK, True, 15)
    for i, (lab, v, col) in enumerate([("本讲的模型", N.FWD_TFS_OURS, BL), ("DeepSeek V3", N.FWD_TFS_THEIRS, PU)]):
        yy = py + 76 + i * 56
        f.t(X1 - 14, yy + 22, lab, INK, True, 15, "end")
        f.box(X1, yy, v * SC1, 32, col, col, 3)
        f.t(X1 + v * SC1 + 10, yy + 22, "%d" % v + ("（约 %.1f 倍）" % N.FWD_SPEED_RATIO if i == 0 else ""), INK, True, 15)
    f.t(820, py + 230, "耗时 9.31 对 2.37 直接比，看着是我们慢；按块折算，我们反而快", RD, True, 15)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "每层都有集合通信把所有设备拉齐，步时由最忙的那个设备定；读单芯片 profile，先数它这一步干了多少活", [])
    yb = f.src(yb + 10,
               "📌 实测：两份 profile 都是进程 0 的设备 0（它对应第 0 行、上下文并行的第 0 段，是按切分规则推断的，证据较强）。用 CPU 按同一步的真实分段重放、按注意力核的剪块规则逐块数：本讲模型这颗芯片 66 个子块（那一行是一整篇 32K 文档），对方这颗 8 个，对方那一步最忙的设备 46 个；每设备满因果是 66 个（1024 × 1024 子块；一整篇 32K 文档就是满因果）。",
               "📌 实测：前向每次 9.31 对 2.37 ms；每个子块前向 68.7 对 85.9 GFLOP（两边头数、头维不同），折算 487 对 290 万亿次/秒，本讲模型约是对方的 1.7 倍。",
               "📌 实测：单芯片单测用同一组分段、按实际执行的块算，本讲模型的形状比 DeepSeek V3 的形状快 1.4–1.6 倍，并且复现了 profile 里的耗时。")
    f.save("fig8b-busiest.svg", yb + 14)


if __name__ == "__main__":
    fig_ladder()
    fig_selfslow()
    fig_busiest()
