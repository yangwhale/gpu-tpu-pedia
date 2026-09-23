# -*- coding: utf-8 -*-
r"""专题四 · §1.0 第三步「一个 token 一个 loss，一个 batch 取平均」

⭐⭐⭐ 2026-09-23 现场（原话）：
  「这个地方那么重要，你不值得画一个图吗？就是一个矩阵图，横着的是 4095，
    竖着的是 batch size，然后每一个格里写上它的 loss，最后做一个平均值出来，
    是一个总的 loss。」

⛔ 那一步原来只有一句话：「把它们全部取平均，就得到这一批的 loss」。
  ⭐ 而这一讲开篇立的那个悬念，正是**「一个数」**这三个字 ——&#160;
    这张网格是那三个字唯一一次真正被画出来的地方。
  ⭐⭐ 判据：**一句话里如果藏着一次「几千万变成一」的塌缩，它就该有一张图。**
    塌缩这种事文字描述不了 ——&#160;读者看不见「多」，也就感觉不到「一」有多狠。

⛔ 刻意没画的：反向、梯度、除法发生在哪一层。
  那些是 1.2b 和 1.2e 的事（fig-push / fig-levels）。**这一格只讲 loss 本身。**
  ⚠️ 判据：**同一件事只在一处讲。** 这张图跟 fig-push Ⓑ 看着像，
    但那张讲的是「梯度的种子有几个」，这张讲的是「loss 有几个」。
"""
from topic03_draw import Fig, BL, OR, GR, GY, INK, GY2

W = 1400

SEQ = 4096
TGT = SEQ - 1                      # 一条 4,096 长的序列只有 4,095 个「下一个字」
GB = 15360                         # V3 预训练 global batch 的上限
CELLS = TGT * GB

assert TGT == 4095 and CELLS == 62899200, (TGT, CELLS)

# ⭐ 格子里那些数是**写死的**，不是随机生成的 ——&#160;构建要可复现。
#   取值范围照着这一节那把尺：0 到 11.77 之间，多数落在 2 上下。
SAMPLE = (0.31, 2.14, 0.08, 5.62, 1.07, 0.44, 3.28, 0.92, 7.81, 0.15,
          2.63, 1.38, 0.57, 4.05, 0.23, 1.91, 6.74, 0.36, 2.88, 1.12)


def main():
    f = Fig(W, "一条四千零九十六长的序列有四千零九十五个位置，每个位置猜一次下一个字，"
               "于是每个位置都有一个自己的 loss。"
               "把一批里所有序列摞起来，就是一张网格：横着是位置，竖着是这一批的每一条序列。"
               "按 V3 的口径，这张网格有六千两百八十九万九千两百个格子，每一格一个数。"
               "把这些数全部加起来再除以格子的个数，就得到这一批的那一个 loss。"
               "两个口径要说准：一是分母按真正算了的目标位置数，"
               "padding 和拼接边界那些格子要遮掉，不是行乘列硬乘；"
               "二是按 token 平均，不是先把每条序列平均成一个数再平均")

    y0 = f.header(
        "一个 token 一个 loss　——　"
        "<tspan font-weight=\"700\">%s 个数，取一个平均</tspan>" % format(CELLS, ","),
        "⭐ 这一讲开篇那句「<tspan font-weight=\"700\">错得有多离谱是一个数</tspan>」"
        "——　就是在这一步变成「一个」的",
        [(BL, "Ⓐ 那张网格"), (OR, "Ⓑ 除的是格子数")])

    # ══════════ Ⓐ 网格 ═══════════════════════════════════════════
    # ⛔ 列数别再往上加：网格右缘 ＝ 150 ＋ NC×74，得给右边那个「一个数」
    #   的方框（x ＝ 1080）留出「……」和箭头的位置。14 列时它俩是叠着的。
    NC, NR = 10, 8                 # 画出来的列数（左 8 ＋ 右 2）/ 行数
    CW, CH = 74.0, 30.0
    PH = 606
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 横着是<tspan font-weight=\"700\">位置</tspan>，"
                 "竖着是<tspan font-weight=\"700\">这一批里的每一条序列</tspan>"
                 "　——　<tspan font-weight=\"700\">每一格一个 loss</tspan>", BL,
                 sub="⭐ 画出来的是 %d × %d 示意；真实是 %s × %s"
                     % (NR, NC, format(GB, ","), format(TGT, ",")))

    # ⛔⛔ 2026-09-23 现场看着 Ⓑ 左边那个框说：「这个小方块里讲的内容不明不白的，
    #   根本就看不懂。要是讲不明白就删了。」——&#160;那个框写的是
    #   「padding / 拼接边界 / total_weights」，三个词一个都没解释过。
    #   ⭐ 但这件事**是可以画的** ——&#160;它本来就发生在这张网格上：
    #     几个格子被涂灰、不算分，如此而已。文字讲不明白，是因为它在
    #     **用词汇描述一张图**，而那张图就在它上面。
    #   ⭐⭐ 判据：**一段文字解释的是图上本来就有的东西时，把它画进图里，别写在旁边。**
    #   ⚠️ 为此列的排法也改了：原来画的是「位置 1 到 11」，**行尾根本不在画面上**，
    #     而 padding 恰恰长在行尾。现在改成「前 8 列 … 最后 2 列」，行尾才看得见。
    NCL, NCR, GAPW = 8, 2, 44         # 左边几列 / 右边几列 / 中间省略号留多宽
    assert NCL + NCR == NC, (NCL, NCR, NC)
    PAD = {(2, "R1"), (2, "R0"), (5, "R1")}      # 行尾补的空位
    SEAM = {(3, 5)}                              # 两篇文章的接缝

    def cellx(c):
        """c 是 0..NCL-1（左块）或 "R0"/"R1"（右块，也就是倒数第 2、1 列）。"""
        if isinstance(c, str):
            return gx + NCL * CW + GAPW + int(c[1]) * CW
        return gx + c * CW

    gx, gy = 150, py + 96
    GW = NCL * CW + GAPW + NCR * CW
    f.t(gx + GW / 2, py + 54,
        "位置 1 →　位置 %s　（一条序列有 %s 个「下一个字」）"
        % (format(TGT, ","), format(TGT, ",")), GY, size=13, anchor="middle")
    f.line(gx, py + 70, gx + GW, py + 70, BL, 1.4)

    for r in range(NR):
        for c in list(range(NCL)) + ["R0", "R1"]:
            x = cellx(c)
            dead = (r, c) in PAD or (r, c) in SEAM
            v = SAMPLE[(r * 5 + (int(c[1]) + 20 if isinstance(c, str) else c) * 3)
                       % len(SAMPLE)]
            f.box(x, gy + r * CH, CW - 3, CH - 3,
                  "#eceff1" if dead else "#fff", GY2, 2, sw=0.8)
            f.t(x + (CW - 3) / 2, gy + r * CH + 20,
                "—" if dead else "%.2f" % v, GY2, size=11.5, anchor="middle")
        f.t(gx + NCL * CW + GAPW / 2, gy + r * CH + 20, "…",
            GY2, size=13, anchor="middle")
        f.t(gx - 14, gy + r * CH + 20, "序列 %d" % (r + 1),
            GY2, size=11.5, anchor="end")
    f.t(gx - 14, gy + NR * CH + 22, "……", GY2, True, 16, "end")
    f.t(gx + GW / 2, gy + NR * CH + 26,
        "…… 一直到第 %s 条" % format(GB, ","), GY2, size=12.5, anchor="middle")

    # ── 灰格：不算分的位置。⭐ 画出来，不要写在旁边 ──────────────
    LY = gy + NR * CH + 54
    f.box(gx, LY, 22, 22, "#eceff1", GY2, 2, sw=0.8)
    f.t(gx + 32, LY + 16,
        "<tspan font-weight=\"700\">灰格 ＝ 不算分的位置</tspan>"
        "　——　它们<tspan font-weight=\"700\">不进分母</tspan>，有两种：",
        INK, size=13.5)
    f.t(gx + 32, LY + 42,
        "① <tspan font-weight=\"700\">行尾补的空位</tspan>（padding）"
        "　——　这条文章不够长，后面拿空位填满好摞成一个整齐的方块。"
        "<tspan fill=\"%s\">那儿根本没有字，猜什么都没意义。</tspan>" % GY, GY, size=13)
    f.t(gx + 32, LY + 66,
        "② <tspan font-weight=\"700\">两篇文章的接缝</tspan>"
        "　——　为了不浪费，短文章会几篇拼进同一条里。"
        "<tspan fill=\"%s\">拿上一篇的结尾去猜下一篇的开头，也没意义。</tspan>" % GY,
        GY, size=13)

    # 右边：塌缩成一个数
    f.box(1080, py + 110, 250, 190, "#e8f0fe", BL, 8)
    f.t(1205, py + 146, "全部加起来", GY, size=13.5, anchor="middle")
    f.t(1205, py + 176, "再除以格子数", GY, size=13.5, anchor="middle")
    f.line(1205, py + 196, 1205, py + 222, BL, 2.4)
    f.t(1205, py + 256, "<tspan font-weight=\"700\">一个数</tspan>",
        BL, True, 26, "middle")
    f.t(1205, py + 286, "这一批的 loss", GY, size=13, anchor="middle")
    f.line(gx + NC * CW + 58, gy + NR * CH / 2, 1072, py + 205, BL, 2.0)

    f.t(700, py + 514,
        "<tspan font-weight=\"700\">%s</tspan> 个格子（%s × %s）"
        "　——　<tspan font-weight=\"700\">每一格都是一个 −ln p</tspan>，"
        "都在 0 到 11.77 之间。"
        % (format(CELLS, ","), format(GB, ","), format(TGT, ",")),
        INK, size=15, anchor="middle")
    f.t(700, py + 544,
        "⭐⭐ <tspan font-weight=\"700\">这 %s 个数，最后只剩一个。</tspan>"
        "　整条链后面所有的事，都是从这一个数出发的。" % format(CELLS, ","),
        INK, True, 15.5, "middle")
    f._pan = None

    # ══════════ Ⓑ 分母 ═══════════════════════════════════════════
    PH2 = 300
    py2 = f.panel(0, py + PH + 20, W, PH2,
                  "Ⓑ 那个「除一次」，除的是<tspan font-weight=\"700\">格子数</tspan>"
                  "　——　<tspan font-weight=\"700\">不是序列数</tspan>", OR)

    # ⛔ 2026-09-23：这里原来还有一个左框，写的是「padding / 拼接边界 /
    #   total_weights」——&#160;现场说「不明不白，根本看不懂」。
    #   ⭐ 它已经**画进 Ⓐ 那张网格里**（灰格 ＋ 两行人话），所以这里整框删掉。
    #     判据：**同一件事只在一处讲；能画的那一处优先。**
    f.box(240, py2 + 52, 920, 190, "#e6f4ea", GR, 8)
    f.t(700, py2 + 90, "分母数的是<tspan font-weight=\"700\">格子</tspan>，"
        "不是<tspan font-weight=\"700\">行</tspan>", GR, True, 17, "middle")
    f.t(700, py2 + 128,
        "<tspan font-weight=\"700\">不是</tspan>「每条序列先平均成一个数，"
        "再把这些数平均」。", INK, size=14.5, anchor="middle")
    f.t(700, py2 + 156,
        "是<tspan font-weight=\"700\">所有格子一视同仁，加起来除一次</tspan>。",
        INK, size=14.5, anchor="middle")
    f.t(700, py2 + 196,
        "⭐ 序列长短不一的时候，这两种算法<tspan font-weight=\"700\">结果不一样</tspan>",
        GR, True, 14.5, "middle")
    f.t(700, py2 + 224,
        "——　按行平均会把<tspan font-weight=\"700\">短序列里的每个字</tspan>看得更重",
        GY, size=13, anchor="middle")
    f._pan = None

    yb = f.band(py2 + PH2 + 18, "ok", "⭐ 一句话记住这张图", [
        "<tspan font-weight=\"700\">一个位置一个 loss</tspan>　——　"
        "一批就是一张 <tspan font-weight=\"700\">%s × %s</tspan> 的网格，"
        "%s 个数。" % (format(GB, ","), format(TGT, ","), format(CELLS, ",")),
        "<tspan font-weight=\"700\">全部加起来，除以真正算了的那些格子的个数，"
        "得到一个数</tspan>　——　后面整条链，都是从这一个数出发的。",
    ])

    yb = f.src(yb + 10,
               "口径来自 MaxText 训练脚本："
               "<tspan font-weight=\"700\">loss ＝ xent_sum ÷ total_weights</tspan>，"
               "其中 total_weights 是遮掉 padding 与拼接边界之后真正算了的目标位置数；",
               "序列 %s、global batch 上限 %s 取自 DeepSeek-V3 预训练配置。"
               % (format(SEQ, ","), format(GB, ",")))

    f.save("fig4-mean.svg", yb + 14)


main()
