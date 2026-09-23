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
        [(BL, "Ⓐ 那张网格"), (OR, "Ⓑ 分母到底是什么")])

    # ══════════ Ⓐ 网格 ═══════════════════════════════════════════
    # ⛔ 列数别再往上加：网格右缘 ＝ 150 ＋ NC×74，得给右边那个「一个数」
    #   的方框（x ＝ 1080）留出「……」和箭头的位置。14 列时它俩是叠着的。
    NC, NR = 11, 8                 # 画出来的列数 / 行数（示意）
    CW, CH = 74.0, 30.0
    PH = 520
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 横着是<tspan font-weight=\"700\">位置</tspan>，"
                 "竖着是<tspan font-weight=\"700\">这一批里的每一条序列</tspan>"
                 "　——　<tspan font-weight=\"700\">每一格一个 loss</tspan>", BL,
                 sub="⭐ 画出来的是 %d × %d 示意；真实是 %s × %s"
                     % (NR, NC, format(GB, ","), format(TGT, ",")))

    gx, gy = 150, py + 96
    f.t(gx + NC * CW / 2, py + 54,
        "位置 1 →　位置 %s　（一条序列有 %s 个「下一个字」）"
        % (format(TGT, ","), format(TGT, ",")), GY, size=13, anchor="middle")
    f.line(gx, py + 70, gx + NC * CW, py + 70, BL, 1.4)

    for r in range(NR):
        for c in range(NC):
            v = SAMPLE[(r * 5 + c * 3) % len(SAMPLE)]
            f.box(gx + c * CW, gy + r * CH, CW - 3, CH - 3, "#fff", GY2, 2, sw=0.8)
            f.t(gx + c * CW + (CW - 3) / 2, gy + r * CH + 20, "%.2f" % v,
                GY2, size=11.5, anchor="middle")
        f.t(gx - 14, gy + r * CH + 20, "序列 %d" % (r + 1),
            GY2, size=11.5, anchor="end")
    f.t(gx - 14, gy + NR * CH + 22, "……", GY2, True, 16, "end")
    f.t(gx + NC * CW + 18, gy + NR * CH / 2, "……", GY2, True, 18)
    f.t(gx + NC * CW / 2, gy + NR * CH + 26,
        "…… 一直到第 %s 条" % format(GB, ","), GY2, size=12.5, anchor="middle")

    # 右边：塌缩成一个数
    f.box(1080, py + 110, 250, 190, "#e8f0fe", BL, 8)
    f.t(1205, py + 146, "全部加起来", GY, size=13.5, anchor="middle")
    f.t(1205, py + 176, "再除以格子数", GY, size=13.5, anchor="middle")
    f.line(1205, py + 196, 1205, py + 222, BL, 2.4)
    f.t(1205, py + 256, "<tspan font-weight=\"700\">一个数</tspan>",
        BL, True, 26, "middle")
    f.t(1205, py + 286, "这一批的 loss", GY, size=13, anchor="middle")
    f.line(gx + NC * CW + 58, gy + NR * CH / 2, 1072, py + 205, BL, 2.0)

    f.t(700, py + 428,
        "<tspan font-weight=\"700\">%s</tspan> 个格子（%s × %s）"
        "　——　<tspan font-weight=\"700\">每一格都是一个 −ln p</tspan>，"
        "都在 0 到 11.77 之间。"
        % (format(CELLS, ","), format(GB, ","), format(TGT, ",")),
        INK, size=15, anchor="middle")
    f.t(700, py + 458,
        "⭐⭐ <tspan font-weight=\"700\">这 %s 个数，最后只剩一个。</tspan>"
        "　整条链后面所有的事，都是从这一个数出发的。" % format(CELLS, ","),
        INK, True, 15.5, "middle")
    f._pan = None

    # ══════════ Ⓑ 分母 ═══════════════════════════════════════════
    PH2 = 300
    py2 = f.panel(0, py + PH + 20, W, PH2,
                  "Ⓑ 分母到底是什么　——　"
                  "<tspan font-weight=\"700\">两处最容易想当然</tspan>", OR)

    f.box(50, py2 + 52, 640, 190, "#fef7e0", OR, 8)
    f.t(370, py2 + 88, "① 不是「行 × 列」硬乘", OR, True, 16.5, "middle")
    f.t(370, py2 + 124,
        "一批数据里有 <tspan font-weight=\"700\">padding</tspan>，"
        "也有<tspan font-weight=\"700\">拼接边界</tspan>　——", INK,
        size=14, anchor="middle")
    f.t(370, py2 + 150,
        "那些格子<tspan font-weight=\"700\">是被遮掉的，不进分母</tspan>。",
        INK, size=14, anchor="middle")
    f.t(370, py2 + 186,
        "⭐ 分母 ＝ <tspan font-weight=\"700\">真正算了的目标位置数</tspan>",
        OR, True, 14.5, "middle")
    f.t(370, py2 + 214,
        "（MaxText 里就叫 total_weights）", GY, size=12.5, anchor="middle")

    f.box(710, py2 + 52, 640, 190, "#e6f4ea", GR, 8)
    f.t(1030, py2 + 88, "② 是按 token 平均，不是按序列", GR, True, 16.5, "middle")
    f.t(1030, py2 + 124,
        "<tspan font-weight=\"700\">不是</tspan>「每条序列先平均成一个数，"
        "再把这些数平均」。", INK, size=14, anchor="middle")
    f.t(1030, py2 + 150,
        "是<tspan font-weight=\"700\">所有格子一视同仁，一次除完</tspan>。",
        INK, size=14, anchor="middle")
    f.t(1030, py2 + 186,
        "⭐ 长短不一的序列，这两种算法<tspan font-weight=\"700\">结果不一样</tspan>",
        GR, True, 14.5, "middle")
    f.t(1030, py2 + 214,
        "——　短序列里的 token 会被前一种算法放大", GY, size=12.5, anchor="middle")
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
