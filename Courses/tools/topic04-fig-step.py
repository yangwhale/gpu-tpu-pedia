# -*- coding: utf-8 -*-
r"""专题四 · §5「一个 step 的总账 ——&#160;以及峰值到底出现在哪一刻」

⭐⭐⭐ 2026-09-16 新画。这是全专题的收口图：把前面四节各自的那一块
  **摆到同一条时间轴上**。

⭐⭐ 这张图逼出了一个我原本以为已经答完的问题。
  §1 那张 fig4-act-bill 说「峰值在前向末尾」——&#160;那句话**只在谈激活时成立**。
  ⛔ 把梯度也画上去就会看到：**激活的峰在前向末尾，梯度的峰在反向末尾，
    两个峰根本不在同一时刻。**
  ⭐ 判据：**「峰值在哪」这个问题，必须先问「哪一项的峰」。**
    把几条不同形状的曲线叠起来之后，「总峰值」未必落在任何一条的峰上。

⭐ Ⓑ 是这张图真正的落点，而且它是**算出来的，不是画出来的**：
  · 常驻那块（权重 2 ＋ 梯度 2 ＋ 优化器 12 ＝ 16 B/参数）× 671B ＝ 9.76 TiB
  · 一条 128K 序列、开了重算的**峰值** ＝ 106.75 ＋ 71.14 ＝ 177.89 GiB
  · 两者相除 →&#160;**约 56 条序列**，激活才追平常驻块
    （按本讲 4K 基准 ＝ 约 1,800 条，也就是约 737 万 token）
  所以「激活大还是优化器状态大」这个问题**没有固定答案**，它取决于 global batch。

⛔⛔ 刻意没画的：
  ① **并行切分。** 这里算的是**全局总量**，不是单卡。怎么切是专题五。
  ② **通信与 overlap。** 时间轴只标阶段，不标真实时长比例（图上注明了）。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE, LINE2)

W = 1400

N_PARAM = 671e9                     # V3 总参数
B_W, B_G, B_OPT = 2, 2, 12          # 每参数字节：bf16 权重 / bf16 梯度 / fp32 那三份
TIB = 1024.0 ** 4
GIB = 1024.0 ** 3
# ⛔⛔ 2026-09-19 T04：峰值 ＝ 存档点 ＋ **当前正在重算的那一层**。
#   原来只算存档点（106.75），漏掉在算的那一层（71.14）——&#160;低估 67%。
#   ⭐ 这一讲自己在 §5.4 的小例子和 §2.2 图 Ⓒ 用的都是正确口径，只有这个头号数字没做。
ACT_ONE_SEQ_GIB = 106.75 + 71.14    # 一条 128K：存档点 ＋ 在算层 ＝ 177.89

PX_PER_TIB = 22.0          # ⭐ 唯一的换算常数：四项全按它算，下面有 assert 盯着
INSET_ZOOM = 20            # 上面那条窄带的放大倍数（梯度与激活实在太薄）

RESIDENT_TIB = N_PARAM * (B_W + B_G + B_OPT) / TIB
W_TIB = N_PARAM * B_W / TIB
OPT_TIB = N_PARAM * B_OPT / TIB
CROSSOVER = RESIDENT_TIB * 1024 / ACT_ONE_SEQ_GIB      # 多少条序列才追平

assert 9.7 < RESIDENT_TIB < 9.8
assert 1.2 < W_TIB < 1.3 and 7.3 < OPT_TIB < 7.4
assert 55 < CROSSOVER < 57       # T04 修正峰值口径后从 94 降到 56


def main():
    f = Fig(W, "把四项显存摆到同一条时间轴上："
               "权重和优化器状态是两条不动的水平带，"
               "激活在前向一路堆高、反向逐步释放，梯度则相反 —— "
               "反向开始才出现，反向结束时最全。"
               "所以激活的峰在前向末尾，梯度的峰在反向末尾，两个峰不在同一时刻。"
               "而哪一项更大取决于 global batch：对 671B 的模型，"
               "要约九十四条 128K 序列，激活才追平常驻那一块")

    y0 = f.header(
        "一个 step 的总账　——　<tspan font-weight=\"700\">"
        "「峰值在哪一刻」要先问「哪一项的峰」</tspan>",
        "⭐ 这里算的是<tspan font-weight=\"700\">全局总量</tspan>，不是单卡"
        "　·　⚠️ 横轴只标阶段，<tspan font-weight=\"700\">不是真实时长比例</tspan>",
        [(GY2, "权重 · 不动"), (RD, "优化器 · 不动"),
         (BL, "激活 · 前向堆"), (OR, "梯度 · 反向堆")])

    # ══════════ Ⓐ 四条带子叠在同一条时间轴上 ═════════════════════
    # ⛔⛔ 2026-09-17 逐图审抓到的真错：四个高度原来是**各挑各的**，
    #   于是同一个量（权重和梯度都是 1.22 TiB）画成了 26px 和 96px，
    #   而最大的优化器反被画得比梯度还矮 —— **图直接否掉了自己框里那句
    #   「四项里最大的一块」**。
    #   ⭐ 判据：**同一张图里代表同一种量的长度，必须共用一个换算常数，并且 assert。**
    #     「按真实比例」写在注释里不算数 —— 注释不会在构建时报错。
    #   ⭐⭐ 修完之后梯度和激活薄到看不见，那正是 Ⓑ 要说的事；
    #     两个峰改用一条**标明了倍数的放大带**来展示 ——
    #     ⛔ 比例失真换来的「看得见」是拿正确性买的，放大插图不是。
    PH = 580
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 两条不动的，加两条形状<tspan font-weight=\"700\">正好相反</tspan>的", BL,
                 sub="⭐ 下面那四层<tspan font-weight=\"700\">按真实比例画</tspan>"
                     "（global batch ＝ 1 条）——&#160;"
                     "<tspan font-weight=\"700\">所以优化器那一块最厚</tspan>")

    X0, X1 = 150, 1320
    TOP = py + 44
    ZTOP, ZBOT = py + 128, py + 228          # 放大带
    BOT = py + 486
    FWD_END, BWD_END = 0.48, 0.92

    H_W = W_TIB * PX_PER_TIB
    H_OPT = OPT_TIB * PX_PER_TIB
    H_G = W_TIB * PX_PER_TIB                 # 梯度跟权重同为 bf16，同一个量
    H_A = ACT_ONE_SEQ_GIB / 1024.0 * PX_PER_TIB
    assert abs(H_W - H_G) < 1e-9, "权重和梯度是同一个量，高度必须相等"
    assert H_OPT > 5 * H_G, "优化器是最大的一块，画出来也必须最高"

    def X(t):
        return X0 + t * (X1 - X0)

    f.line(X0, BOT, X1 + 10, BOT, GY2, 2.0, arrow=False)
    for t, nm in ((0.0, "开始"), (FWD_END, "前向结束"),
                  (BWD_END, "反向结束"), (1.0, "更新完")):
        f.line(X(t), BOT, X(t), BOT + 6, GY2, 1.2, arrow=False)
        f.t(X(t), BOT + 26, nm, GY2, size=12.5, anchor="middle")
    f.t((X0 + X1) / 2.0, BOT + 60, "一个 step 的时间轴", GY, True, 14, "middle")

    # ── 两条不动的（自下而上，真实比例）
    f.box(X0, BOT - H_W, X1 - X0, H_W, "#f1f3f4", GY2, 0, 1.0)
    f.t(X0 + 14, BOT - 9, "权重　%.2f TiB" % W_TIB, GY, True, 13.5)
    ob = BOT - H_W
    f.box(X0, ob - H_OPT, X1 - X0, H_OPT, "#fce8e6", RD, 0, 1.0)
    f.t(X0 + 14, ob - H_OPT + 34, "优化器状态　%.2f TiB" % OPT_TIB, RD, True, 19)
    f.t(X0 + 14, ob - H_OPT + 62, "⭐ 全程不动，而且是四项里最大的一块 ——　"
                                  "却只在最后那一瞬间被用一次", GY, size=14)
    base = ob - H_OPT

    def g_of(t):
        if t <= FWD_END:
            return 0.0
        if t >= BWD_END:
            return float(H_G)
        return H_G * (t - FWD_END) / (BWD_END - FWD_END)

    def a_of(t):
        if t <= FWD_END:
            return H_A * t / FWD_END
        if t >= BWD_END:
            return 0.0
        return H_A * (BWD_END - t) / (BWD_END - FWD_END)

    TS = [i / 80.0 for i in range(81)]
    grad = ([(X(t), base) for t in TS]
            + [(X(t), base - g_of(t)) for t in reversed(TS)])
    act = ([(X(t), base - g_of(t)) for t in TS]
           + [(X(t), base - g_of(t) - a_of(t)) for t in reversed(TS)])
    f.path(act, BL, 1.0, arrow=False, fill="#e8f0fe")
    f.path(grad, OR, 1.0, arrow=False, fill="#fef7e0")
    f.t(X0 + 14, base - 16, "梯度 %.2f TiB　＋　激活 %.2f GiB　——　"
                            "<tspan font-weight=\"700\">薄成这样是真的</tspan>"
        % (W_TIB, ACT_ONE_SEQ_GIB), GY, size=13.5)

    # ── 放大带：把上面那两层放大，两个峰才看得见
    f.box(X0, ZTOP, X1 - X0, ZBOT - ZTOP, "#fafafa", LINE2, 6, 1.0)
    f.t(X0 + 12, ZTOP + 18, "↑ 上面那两层<tspan font-weight=\"700\">放大 %d 倍</tspan>"
                            " ——　真实比例下它们太薄，两个峰看不出来" % INSET_ZOOM,
        GY2, size=12.5)
    zb = ZBOT - 8
    zg = ([(X(t), zb) for t in TS]
          + [(X(t), zb - g_of(t) * INSET_ZOOM / 8.0) for t in reversed(TS)])
    za = ([(X(t), zb - g_of(t) * INSET_ZOOM / 8.0) for t in TS]
          + [(X(t), zb - (g_of(t) + a_of(t)) * INSET_ZOOM / 8.0) for t in reversed(TS)])
    f.path(za, BL, 1.2, arrow=False, fill="#e8f0fe")
    f.path(zg, OR, 1.2, arrow=False, fill="#fef7e0")

    # ── 两个峰
    f.line(X(FWD_END), ZTOP, X(FWD_END), BOT, BL, 1.8, dash="5 4", arrow=False)
    f.line(X(BWD_END), ZTOP, X(BWD_END), BOT, OR, 1.8, dash="5 4", arrow=False)
    f.box(X(FWD_END) - 148, TOP + 6, 296, 54, "#e8f0fe", BL, 8)
    f.t(X(FWD_END), TOP + 30, "⭐ 激活的峰", BL, True, 16, "middle")
    f.t(X(FWD_END), TOP + 50, "前向刚结束，反向还没开始", GY, size=12.5, anchor="middle")
    f.box(X(BWD_END) - 244, TOP + 6, 296, 54, "#fef7e0", OR, 8)
    f.t(X(BWD_END) - 96, TOP + 30, "⭐ 梯度的峰", OR, True, 16, "middle")
    f.t(X(BWD_END) - 96, TOP + 50, "反向刚结束，还没更新", GY, size=12.5, anchor="middle")
    f.t(X0 + 6, TOP + 28, "⛔⛔ 两个峰<tspan font-weight=\"700\">不在同一时刻</tspan>",
        RD, True, 16)
    f.t(X0 + 6, TOP + 52, "先问「哪一项的峰」，再问「在哪一刻」", GY, size=13)
    f._pan = None

    # ══════════ Ⓑ 哪一项更大？看 global batch ════════════════════
    PH2 = 330
    py2 = f.panel(0, py + PH + 22, W, PH2,
                  "Ⓑ ⭐⭐ 「激活大还是优化器状态大」——　"
                  "<tspan font-weight=\"700\">这个问题没有固定答案</tspan>", RD,
                  sub="⭐ 因为常驻那块<tspan font-weight=\"700\">不随 batch 变</tspan>，"
                      "而激活<tspan font-weight=\"700\">线性地随 batch 涨</tspan>")

    BX, BW = 300, 880
    ROWS = (
        ("常驻那一块（不随 batch 变）", RESIDENT_TIB, RD, "#fce8e6",
         "权重 2 ＋ 梯度 2 ＋ 优化器 12 ＝ 16 B／参数，× 671B"),
        ("激活 · global batch ＝ 1 条", ACT_ONE_SEQ_GIB / 1024.0, BL, "#e8f0fe",
         "开了全量重算的 128K 序列"),
        # ⛔ T19 待办：「global batch」这个量名是错的 ——&#160;同时在场的是
        #   DP 路数 × micro-batch，不是 global batch（加了梯度累积之后两者不等）。
        #   ⭐ 换句话说这个分水岭量的是一个**物理上不会同时存在**的量。留给 T19 一起改。
        ("激活 · global batch ＝ %d 条" % round(CROSSOVER),
         ACT_ONE_SEQ_GIB * CROSSOVER / 1024.0, BL, "#e8f0fe",
         "⭐ 到这里才追平 ——　这个数就是分水岭"),
    )
    for i, (nm, tib, col, fill, note) in enumerate(ROWS):
        y = py2 + 40 + i * 84
        f.t(BX - 16, y + 26, nm, INK, True, 15, "end")
        frac = min(1.0, tib / RESIDENT_TIB)
        f.box(BX, y, BW, 40, "#f8f9fa", LINE2, 6)
        f.box(BX, y, max(4, BW * frac), 40, fill, col, 6)
        f.t(BX + BW + 14, y + 27, "%.2f TiB" % tib, col, True, 17)
        f.t(BX + 12, y + 62, note, GY, size=13)
    f.t(BX - 16, py2 + 296, "⭐⭐⭐", RD, True, 18, "end")
    f.t(BX, py2 + 296, "所以 <tspan font-weight=\"700\">「谁最大」不是模型的属性，"
                       "是<tspan text-decoration=\"underline\">这次训练配置</tspan>的属性</tspan>"
                       " ——　换个 global batch，答案就换了。", INK, True, 15.5)
    f._pan = None

    yy = f.band(py2 + PH2 + 22, "info", "顺带回答一个常被问反的问题：梯度累积省的是什么", [
        "⭐ <tspan font-weight=\"700\">梯度累积改的是时间线，不是总量。</tspan>"
        "它把一个大 batch 拆成几个 micro-batch 顺序跑 ——&#160;于是"
        "<tspan font-weight=\"700\">同时在场的激活只剩一个 micro-batch 的份</tspan>，"
        "也就是把 Ⓐ 里那座蓝色的山削矮了。",
        "⛔ 但它<tspan font-weight=\"700\">一个字节都不省常驻那块</tspan>"
        "（权重 / 梯度 / 优化器状态照旧），"
        "<tspan font-weight=\"700\">总 FLOPs 也一点没少</tspan> ——&#160;还是那么多要算。"
        "⚠️ 而且它会把梯度那一栏<tspan font-weight=\"700\">逼成 fp32</tspan>"
        "（见 3.2）——&#160;那一栏反而从 2 B 变成 4 B。",
    ], fold=True)

    yy = f.src(yy + 24,
               "⭐ Ⓑ 三行都是<tspan font-weight=\"700\">当场算的</tspan>："
               "671e9 × 16 B ÷ 1024⁴ ＝ %.2f TiB；"
               "%.2f TiB × 1024 ÷ %.2f GiB ＝ %.1f 条 ——&#160;"
               "<tspan font-weight=\"700\">脚本里有 assert，改数会自己报错</tspan>"
               % (RESIDENT_TIB, RESIDENT_TIB, ACT_ONE_SEQ_GIB, CROSSOVER),
               "⚠️ 16 B／参数是 <tspan font-weight=\"700\">ZeRO 论文的经典口径</tspan>"
               "（arXiv 1910.02054）；DeepSeek-V3 实际把一二阶矩放成了 bf16，"
               "<tspan font-weight=\"700\">它的每参数账跟这个数不一样</tspan>（见 3.2）",
               "⚠️ 激活那 %.2f GiB 是<tspan font-weight=\"700\">自己按算子推的估算</tspan>，"
               "没有第三方背书 ——&#160;所以那个「%d 条」也是量级，不是阈值"
               % (ACT_ONE_SEQ_GIB, round(CROSSOVER)),
               "⛔ 全图算的是<tspan font-weight=\"700\">全局总量</tspan>，"
               "没有任何并行切分 ——&#160;怎么把它摊到多少张卡上，是<tspan "
               "font-weight=\"700\">专题五</tspan>整讲的事")
    f.save("fig4-step.svg", yy + 6)


main()
