# -*- coding: utf-8 -*-
r"""专题三 · §8.2「3:1 是怎么消融出来的 —— 以及那张表问不出什么」

⛔⛔ 2026-09-14 R28：**这张图是为了修课程自己的一处事实错误而画的。**

   §8.2 原文写着：

     「Kimi Linear 的消融里，0:1（纯全注意力）反而表现不好
       （⚠️ 原文只有这一句定性描述，**没公开数值**）」

   ⛔ **「没公开数值」是错的。** arXiv 2510.26692v2 的 Table 1
   把五个配比的训练 / 验证 PPL 全列出来了，逐字：

     Hybrid ratio  3:1  9.23 5.65   ← 论文用灰底标出的最优
                   0:1  9.45 5.77
                   1:1  9.29 5.66
                   7:1  9.23 5.70
                  15:1  9.34 5.82

   同段还写明消融模型是「16 heads, **16 layers**」、
   「All models were trained with the same FLOPs budget」。

   ⭐ 数字一摆出来，比原来那句定性描述多说了三件事：

   ① **0:1 不是最差的** ——&#160;5.77 排第四，最差的是 15:1 的 5.82。
      课程原来那句「0:1 反而表现不好」方向对，但排名说重了。
   ② **3:1 和 7:1 的训练 PPL 完全相同（都是 9.23），验证 PPL 才分开。**
      论文自己点了这一句：「a higher ratio (e.g., 7:1) produced a comparable
      training loss but led to significantly worse validation performance」。
   ③ ⭐⭐ **这五个配比不是挑出来的，是 16 除出来的。**
      配比 r:1 要在 16 层里摆匀，每组 (r+1) 层就必须整除 16，
      于是 r+1 ∈ {1,2,4,8,16}，r ∈ {0,1,3,7,15} ——&#160;**正好就是表里这五个**。
      4:1 要 5 层一组、2:1 要 3 层一组，16 都摆不匀，**所以它们根本没机会进这张表**。

      ⚠️ 这是对那张表的**算术观察**，不是作者说的理由 ——&#160;
      论文没有解释为什么选这五个。图上按算术事实写，不替作者编动机。

   ④ 最低点落在「16 层里有 **4 层**全注意力」。而 Kimi K3 有 93 层，
      按 3:1 配了 **24 层**全注意力（课程 §8.2 表里已核过：
      93 = 23×(3 KDA + 1 Gated MLA) + 1 MLA）。
      **保比例得 24 层，保个数只要 4 层，差 6 倍** ——&#160;
      公开文献里没有任何实验回答过该保哪个。这是本图留给读者的悬案，
      ⛔ 不要写成结论。

   ⑤ 口径警告：这整张表是 **PPL**。另一篇系统性消融（arXiv 2507.06457）
      的正文逐字说：语言建模分「remains largely flat across all ratio
      configurations. Most architectures cluster around 0.55-0.57」，
      而召回「rises from pure linear configurations (around 0.1-0.35 RULER
      score) toward the full-attention baseline (dashed line at approximately
      0.42)」，且「most architectures approach or exceed this baseline at the
      3:1 ratio」。它给自己的结论起的标题是
      「Recall—not perplexity—determines the optimal linear:full mix」。

      ⛔ 那篇摘要里的「nearly doubles RULER recall」**本图不引** ——&#160;
      只核到了正文这段描述，没有逐格核过 Table 6/7，倍数说法存疑。
      **只画正文说得死的那部分：一条平的、一条涨的。**

⭐ 画图纪律（R26 踩过的坑，别再犯）：
   凡是图上要出现的数，一律从下面这组常量里取，别手抄；
   纵轴被放大过的图，必须把「放大了多少」写在图上。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, GY, PU, INK, GY2, LINE, LINE2,
                          BG2)

W = 1400

# ── arXiv 2510.26692v2 Table 1，逐字抄自论文（16 heads / 16 layers）────
#    (配比 r, 训练 PPL, 验证 PPL)
ABL = [(0, 9.45, 5.77), (1, 9.29, 5.66), (3, 9.23, 5.65),
       (7, 9.23, 5.70), (15, 9.34, 5.82)]
NL = 16                                   # 消融模型的层数，论文原文
BEST = 3                                  # 论文用灰底标出的那一行

K3_LAYERS, K3_FULL = 93, 24               # 课程 §8.2 表已核（arXiv 2607.24653）

# 两条曲线用**同一个跨度**，这样斜率可以直接比
SPAN = 0.30
TR_LO = 9.20
VA_LO = 5.60


def full_layers(r):
    """配比 r:1 在 NL 层里摆匀时，全注意力层有几层。"""
    assert NL % (r + 1) == 0, "r=%d 在 %d 层里摆不匀" % (r, NL)
    return NL // (r + 1)


def fits(r):
    return NL % (r + 1) == 0


# ⭐ 核心论点：表里这五个配比，正好就是「(r+1) 整除 16」的全部解
_all_fit = [r for r in range(0, NL) if fits(r)]
assert _all_fit == [r for r, _, _ in ABL], _all_fit
assert not fits(2) and not fits(4) and not fits(5)

_best = min(ABL, key=lambda t: t[2])
assert _best[0] == BEST and _best[2] == 5.65
# 0:1 不是最差 —— 这是修课程原文的那一条
_worst = max(ABL, key=lambda t: t[2])
assert _worst[0] == 15 and _worst[2] == 5.82
# 3:1 与 7:1 训练 PPL 相同、验证 PPL 不同
assert dict((r, tr) for r, tr, _ in ABL)[3] == dict(
    (r, tr) for r, tr, _ in ABL)[7] == 9.23
# 两条曲线都装得进 SPAN
for _r, _tr, _va in ABL:
    assert 0 <= _tr - TR_LO <= SPAN and 0 <= _va - VA_LO <= SPAN
_spread = _worst[2] - _best[2]
assert abs(_spread - 0.17) < 1e-9
_pct = 100.0 * _spread / _best[2]
assert 2.9 < _pct < 3.1                   # 全场只差 3%
assert full_layers(BEST) == 4 and K3_FULL // full_layers(BEST) == 6


def main():
    f = Fig(W, "Kimi Linear 那张配比消融表画出来：验证 PPL 是一条 U 形，"
               "最低点在 3:1，而纯全注意力 0:1 排第四不是最差；"
               "五个配比其实是 16 层除出来的，4:1 摆不匀所以根本没进表；"
               "最低点是「16 层里 4 层全注意力」，而 K3 保的是比例不是个数")
    f.marks = set()
    y0 = f.header(
        "「3:1 最好」这句话，底下是一张五行的表",
        "<tspan font-weight=\"700\">数值全都公开了</tspan>　·　"
        "arXiv 2510.26692v2 Table 1　·　消融模型 "
        "<tspan font-weight=\"700\">16 头 16 层</tspan>，同等 FLOPs 预算",
        [(GR, "论文标出的最优"), (BL, "全注意力层"), (RD, "摆不匀，进不了表")])

    # ══════════ ① 两条曲线：同一个跨度，形状不一样 ══════════════════
    PH1 = 320
    top = f.panel(0, y0, W, PH1, "① 表画出来长什么样", BL,
                  sub="左右两张图的纵轴跨度完全相同（都是 0.30），所以斜率可以直接比")

    PLOT_H, AX_TOP = 130, top + 52
    BASE = AX_TOP + PLOT_H

    def draw_curve(x0, x1, lo, key, title, col, tag):
        n = len(ABL)
        step = (x1 - x0) / float(n - 1)
        f.t((x0 + x1) / 2.0, top + 26, title, col, True, 15, "middle")
        # 纵轴：只画上下两条刻度，把「放大了多少」写死在轴上
        f.line(x0 - 42, AX_TOP, x0 - 42, BASE, LINE2, 1.2, arrow=False)
        f.t(x0 - 50, AX_TOP + 5, "%.2f" % (lo + SPAN), GY2, False, 14, "end")
        f.t(x0 - 50, BASE + 5, "%.2f" % lo, GY2, False, 14, "end")
        f.line(x0 - 42, BASE, x1 + 20, BASE, LINE2, 1.1, arrow=False)

        pts = []
        for i, row in enumerate(ABL):
            r, v = row[0], row[key]
            px = x0 + i * step
            py = BASE - (v - lo) / SPAN * PLOT_H
            pts.append((px, py, r, v))
        for i in range(len(pts) - 1):
            f.line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1],
                   col, 2.2, arrow=False)
        for px, py, r, v in pts:
            c = GR if r == BEST else col
            f.box(px - 6, py - 6, 12, 12, c, "#fff", 6, 1.6)
            f.t(px, py - 16, "%.2f" % v, c, True, 15, "middle")
            f.t(px, BASE + 22, "%d:1" % r, INK if r == BEST else GY,
                r == BEST, 15, "middle")
        f.t((x0 + x1) / 2.0, BASE + 46, tag, GY2, False, 15, "middle")
        return pts

    L0, L1 = 150, 560
    R0, R1 = 880, 1290
    draw_curve(L0, L1, TR_LO, 1, "训练 PPL", GY,
               "3:1 和 7:1 在训练集上一模一样，都是 9.23")
    vp = draw_curve(R0, R1, VA_LO, 2, "验证 PPL（论文据以定稿的那一条）", BL,
                    "同样这两个配置，验证集上差 0.05 —— 训练集看不出来")

    # ⭐ 把「0:1 不是最差」指出来。
    # ⛔ 这里原来画的是一条从 0:1 拉到 15:1 的红虚线 —— 渲染后它长得像
    #   **第二条数据曲线**，而且右端正好压住 15:1 的数值标签。
    #   改成两个端点各自挂一个小标注，不在图区里再添一条线。
    _rank = sorted(ABL, key=lambda t: t[2])
    _rk = dict((r, i + 1) for i, (r, _, _) in enumerate(_rank))
    f.t(vp[0][0], vp[0][1] + 26, "排第 %d" % _rk[0], RD, True, 15, "middle")
    f.t(vp[-1][0], vp[-1][1] - 34, "最差", RD, True, 15, "middle")
    f.t((R0 + R1) / 2.0, BASE + 66,
        "⛔ <tspan font-weight=\"700\">纯全注意力那一头（0:1）不是最差的</tspan>"
        " —— 它排第 %d，最差的是另一头的 15:1" % _rk[0],
        RD, False, 15, "middle")
    assert _rk[0] == 4 and _rk[15] == 5

    # ══════════ ② 五个配比是 16 除出来的 ═══════════════════════════
    y1 = y0 + PH1 + 20
    PH2 = 490
    top = f.panel(0, y1, W, PH2, "② 同样这五个配置，摆成 16 层看", GR,
                  sub="每一列都是那个 16 层模型 · 蓝格 ＝ 全注意力层，灰格 ＝ 线性层")

    CW, CH, GAPY = 62, 13, 2
    GRID_TOP = top + 46

    def draw_col(x, r, lab, sub, hl=False, ghost=False):
        grp = r + 1
        for i in range(NL):
            yy = GRID_TOP + i * (CH + GAPY)
            if ghost and i >= (NL // grp) * grp:
                f.box(x, yy, CW, CH, "#fff", RD, 3, 1.4, dash="3 3")
                continue
            is_full = ((i + 1) % grp == 0)
            f.box(x, yy, CW, CH, BL if is_full else "#eceff1",
                  "#fff" if is_full else LINE2, 3, 1)
        gy = GRID_TOP + NL * (CH + GAPY)
        if hl:
            f.box(x - 7, GRID_TOP - 7, CW + 14,
                  NL * (CH + GAPY) + 6, "none", GR, 6, 2)
        f.t(x + CW / 2.0, GRID_TOP - 16, lab, GR if hl else INK,
            True, 15, "middle")
        for k, s in enumerate(sub):
            f.t(x + CW / 2.0, gy + 16 + k * 19, s,
                GR if (hl and k == 0) else GY, k == 0, 14.5, "middle")

    # ⛔ 2026-09-14 R28 二修：右侧那段说明原来是**一条 19 行的窄栏**，
    #   为了塞进 358px 只能用 13px ——&#160;结果 course-metrics 报
    #   「图内 ≥15px 文字占比 21%」，全书 50 张图里唯一不及格的一张。
    #   ⭐ 版面体检查的是「有没有撞车」，查不出「字太小」；
    #     字号这条得靠 course-metrics。**窄栏是逼小字号的元凶**，
    #     解法不是把字缩小去迁就栏宽，是把栏拓宽 / 换成通栏。
    xs = [70, 218, 366, 514, 662]
    for x, (r, tr, va) in zip(xs, ABL):
        draw_col(x, r, "%d:1" % r,
                 ["%d 层全注意力" % full_layers(r), "验证 PPL %.2f" % va],
                 hl=(r == BEST))

    f.line(838, GRID_TOP - 24, 838, GRID_TOP + NL * (CH + GAPY) + 42,
           LINE2, 1.2, dash="6 5", arrow=False)

    draw_col(900, 4, "4:1", ["5 层一组", "⛔ 16 摆不匀"], ghost=True)

    f.lines(1010, GRID_TOP + 6, 350, [
        "⭐⭐ 这五个配比不是挑出来的，",
        "<tspan font-weight=\"700\">是 16 除出来的。</tspan>",
        "",
        "16 层要摆得匀，每组 (r+1) 层",
        "就必须<tspan font-weight=\"700\">整除 16</tspan>。而 16 的约数",
        "只有 1、2、4、8、16 ——",
        "于是 r 只能取",
        "<tspan font-weight=\"700\">0、1、3、7、15</tspan>，",
        "正好就是表里那五行。",
    ], size=15, lh=25)

    gy2 = GRID_TOP + NL * (CH + GAPY) + 56
    f.lines(70, gy2, 620, [
        "⛔ 所以 <tspan font-weight=\"700\">4:1 从来没被试过</tspan> —— 它要 5 层一组，",
        "摆到第 15 层就多出一层没地方放（右边那一列）。",
        "<tspan font-weight=\"700\">2:1（3 层一组）同样摆不匀</tspan>，"
        "所以 1:1 和 3:1 中间那一段，这张表也问不出来。",
    ], size=15, lh=25)
    f.lines(740, gy2, 620, [
        "⚠️ 「(r+1) 必须整除 16」是对那张表做的"
        "<tspan font-weight=\"700\">算术观察</tspan>，",
        "不是作者给的理由 —— 论文并没有解释为什么选这五个。",
        "⭐ 但结论不变：<tspan font-weight=\"700\">"
        "「为什么是 3 不是 4」，这张表回答不了。</tspan>",
    ], size=15, lh=25)

    # ══════════ ③ 最低点是「4 层」还是「3:1」 ══════════════════════
    y2 = y1 + PH2 + 20
    PH3 = 330
    top = f.panel(0, y2, W, PH3, "③ 那最低点到底是「3:1」，还是「4 层」", OR,
                  sub="16 层里两者是同一件事 · 93 层里差 6 倍 · 公开文献没答案")

    bw = 640
    f.box(40, top + 16, bw, 250, "#fff", LINE, 8, 1.2)
    f.t(64, top + 46, "同一个最低点，两种读法", INK, True, 16)
    f.lines(64, top + 70, bw - 48, [
        "读成<tspan font-weight=\"700\">比例</tspan>：3 个线性配 1 个全注意力。"
        "→ 93 层的 K3 要配 "
        "<tspan font-weight=\"700\" fill=\"%s\">%d 层</tspan>全注意力。"
        % (BL, K3_FULL),
        "读成<tspan font-weight=\"700\">个数</tspan>：一共有 4 层全注意力就够。"
        "→ 93 层的 K3 也只要 "
        "<tspan font-weight=\"700\" fill=\"%s\">4 层</tspan>。" % OR,
        "",
        "<tspan font-weight=\"700\">在 16 层里，这两句话完全等价</tspan>"
        " —— 16 ÷ 4 ＝ 4，怎么读都是 4 层。",
        "到了 93 层，两种读法差 <tspan font-weight=\"700\">%d 倍</tspan>。"
        % (K3_FULL // full_layers(BEST)),
        "",
        "⛔ K3 实际保的是<tspan font-weight=\"700\">比例</tspan>"
        "（93 层里 24 层全注意力，本课那张配比表已核）。",
        "而<tspan font-weight=\"700\">该保哪个</tspan>，我没有找到任何公开实验"
        "回答过 —— 所有配比消融都在小模型上做的。",
    ], size=15, lh=25)

    rx = 720
    f.box(rx, top + 16, W - rx - 40, 250, "#fff", LINE, 8, 1.2)
    f.t(rx + 24, top + 46, "⚠️ 还有一个口径问题：这整张表是 PPL", OR, True, 16)
    f.lines(rx + 24, top + 70, W - rx - 88, [
        "全场五个配比，验证 PPL 从 %.2f 到 %.2f —— "
        "<tspan font-weight=\"700\">一共差 %.0f%%</tspan>。"
        % (_best[2], _worst[2], _pct),
        "",
        "另一篇 340M / 1.3B 的系统性消融（arXiv 2507.06457）",
        "把两个口径分开画，结论是<tspan font-weight=\"700\">一条平的、一条涨的</tspan>：",
        "· 语言建模分 —— 各架构都挤在 0.55～0.57，几乎不受影响；",
        "· 召回（RULER）—— 从纯线性的 0.1～0.35 一路涨到全",
        "　注意力基线约 0.42，<tspan font-weight=\"700\">多数架构在 3:1 追平或超过</tspan>。",
        "",
        "⭐ 它给这个结论起的标题是：",
        "<tspan font-weight=\"700\">「决定配比的是召回，不是困惑度」</tspan>。",
    ], size=15, lh=25)

    # ══════════ 落点 ════════════════════════════════════════════
    yy = y2 + PH3 + 22
    yy = f.band(yy, "info", "这张表能回答什么，不能回答什么", [
        "<tspan font-weight=\"700\">能</tspan>：3:1 在这五个候选里最好，"
        "而且两头都比它差 —— 包括纯全注意力那一头。",
        "<tspan font-weight=\"700\">不能</tspan>：为什么是 3 不是 4。"
        "4:1 在 16 层里摆不匀，从来没进过候选。",
        "<tspan font-weight=\"700\">不能</tspan>：93 层该配几层全注意力。"
        "16 层里「3:1」和「4 层」是同一句话，93 层里不是。",
    ])
    yy = f.band(yy + 12, "warn", "引用这张表时必须一起说的两句", [
        "① 它是 <tspan font-weight=\"700\">16 层、同等 FLOPs</tspan> 的小模型消融 —— "
        "配比结论没有在大模型上复现过。",
        "② 它量的是 <tspan font-weight=\"700\">PPL</tspan>，而 PPL 恰恰是"
        "各家一致公认「对配比不敏感」的那个指标。要判配比得看召回。",
    ], fold=True)

    yy = f.src(yy + 14,
               "📌 Kimi Linear，arXiv 2510.26692v2 §5.2 Table 1 与同段正文"
               "（「16 heads, 16 layers」「same FLOPs budget」）——&#160;"
               "五个配比的训练 / 验证 PPL 均逐字抄自该表。",
               "📌 系统性消融，arXiv 2507.06457 §4.2 正文与 Figure 3 描述 —— "
               "语言建模「大体持平、各架构都在 0.55-0.57」、召回「从 0.1-0.35 "
               "涨向全注意力基线约 0.42」两句均为原文转述。"
               "⛔ 该文摘要里的「召回近乎翻倍」本图不引：只核到正文这段描述，"
               "没有逐格核过它的 Table 6 / 7。",
               "📌 Kimi K3 的 93 层 / 24 层全注意力见本课那张配比表"
               "（arXiv 2607.24653 表 1 ＋ §2.1）。"
               "「(r+1) 必须整除 16」是本课对该表做的算术观察，不是论文的说法。")
    f.save("fig3-ratio-grid.svg", yy + 16)


if __name__ == "__main__":
    main()
