# -*- coding: utf-8 -*-
r"""专题四 · §1.0b 之前「logits 是从哪儿来的」

⭐⭐⭐ 2026-09-23 现场：
  「logits 怎么变成 loss，这个内容之前先加一个简要的图 ——&#160;就是 logits 怎么来的。
    先画一个前向，然后得到了 7,168，然后 7,168 经过一个 LM 的矩阵，
    就变成了 129,280 那么大一个 logits。先把这个画出来，然后再往下走。」

⛔ 下一格（fig-softmax）一上来就是「一排 logits」——&#160;
  **它从哪儿来、为什么正好是 129,280 个，一个字都没有。**
  ⭐ 判据：**一张图以某个量开场时，上一张图得负责把那个量交到它手上。**
    否则读者第一眼要处理的不是内容，是「这东西哪来的」。

⚠️ 现场要的是「<b>简要</b>」——&#160;所以只有一格、只有一条线，
  不画层内结构、不画 MLA、不画 MoE。那些是专题一的事。
"""
from topic03_draw import Fig, BL, OR, GR, GY, INK, GY2

W = 1400

D_MODEL = 7168                      # 残差流宽度
VOCAB = 129280                      # 词表
TGT = 4095                          # 一条 4,096 长的序列有 4,095 个「下一个字」
LM_PARAMS = D_MODEL * VOCAB         # LM head 那一块权重有多少个数
RATIO = VOCAB / D_MODEL             # logits 比隐藏向量长多少倍
LOGITS_B = TGT * VOCAB * 2          # 全部位置的 logits，bf16

assert LM_PARAMS == 926679040, LM_PARAMS
assert abs(RATIO - 18.04) < 0.01, RATIO
assert abs(LOGITS_B / 1024 ** 3 - 0.99) < 0.01, LOGITS_B


def main():
    f = Fig(W, "一个位置走完整条前向之后，手里拿到的是一个七千一百六十八个数的向量。"
               "再乘上最后那一块权重，也就是 LM head，"
               "它的形状是七千一百六十八乘十二万九千二百八十，"
               "于是那个向量被摊成十二万九千二百八十个数 —— 词表上每个词一个分数。"
               "这排数就叫 logits，它可正可负、还不是概率。"
               "下一步的 softmax 才把它变成概率")

    y0 = f.header(
        "那排 <tspan font-weight=\"700\">logits</tspan> 是从哪儿来的",
        "⭐ 一句话：<tspan font-weight=\"700\">前向走完 →&#160;"
        "一个 %s 维的向量 →&#160;乘 LM head →&#160;%s 个分数</tspan>"
        % (format(D_MODEL, ","), format(VOCAB, ",")),
        [(BL, "前向"), (OR, "LM head"), (GR, "logits")])

    PH = 372
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 一个位置，走完前向之后手里有什么", BL,
                 sub="⚠️ 只画这一条线 ——　层里面长什么样是专题一的事")

    MY = py + 104                                  # 这一行的竖直中心

    # ① 前向
    f.box(60, MY - 44, 210, 88, "#e8f0fe", BL, 8)
    f.t(165, MY - 10, "前向走完", BL, True, 16, "middle")
    f.t(165, MY + 18, "（61 层）", GY, size=13, anchor="middle")
    f.line(276, MY, 316, MY, BL, 2.2)

    # ② 这个位置的向量
    VW = 46
    f.box(324, MY - 62, VW, 124, "#fff", BL, 5, sw=1.8)
    for g in range(1, 6):
        f.line(324, MY - 62 + g * 124 / 6.0, 324 + VW, MY - 62 + g * 124 / 6.0,
               GY2, 0.6, arrow=False)
    f.t(324 + VW / 2, MY + 86, "这个位置的向量", INK, True, 13.5, "middle")
    f.t(324 + VW / 2, MY + 108, "<tspan font-weight=\"700\">%s</tspan> 个数"
        % format(D_MODEL, ","), BL, True, 14, "middle")

    # ③ 乘 LM head
    f.line(388, MY, 440, MY, OR, 2.4)
    f.box(448, MY - 52, 250, 104, "#fef7e0", OR, 8, sw=1.8)
    f.t(573, MY - 18, "× LM head", OR, True, 16, "middle")
    f.t(573, MY + 10, "%s × %s" % (format(D_MODEL, ","), format(VOCAB, ",")),
        INK, True, 14, "middle")
    f.t(573, MY + 36, "这一块权重有 <tspan font-weight=\"700\">%.2f 亿</tspan> 个数"
        % (LM_PARAMS / 1e8), GY, size=12.5, anchor="middle")
    f.line(706, MY, 758, MY, OR, 2.4)

    # ④ logits
    LW = 168
    f.box(766, MY - 62, LW, 124, "#fff", GR, 5, sw=1.8)
    for g in range(1, 6):
        f.line(766, MY - 62 + g * 124 / 6.0, 766 + LW, MY - 62 + g * 124 / 6.0,
               GY2, 0.6, arrow=False)
    f.t(766 + LW / 2, MY + 86, "logits", GR, True, 15, "middle")
    f.t(766 + LW / 2, MY + 108, "<tspan font-weight=\"700\">%s</tspan> 个数"
        "　（词表每个词一个）" % format(VOCAB, ","), GR, True, 14, "middle")

    # ⚠️ 两条不成比例，得说 ——&#160;不说读者会以为只差三倍
    f.t(766 + LW / 2, MY - 82,
        "⚠️ 画不成比例：真实是<tspan font-weight=\"700\">长 %.0f 倍</tspan>"
        % RATIO, GY2, size=12.5, anchor="middle")

    # 右边：落点
    f.box(970, MY - 62, 370, 124, "#e6f4ea", GR, 8)
    f.t(1155, MY - 26, "这排数就叫 <tspan font-weight=\"700\">logits</tspan>",
        GR, True, 17, "middle")
    f.t(1155, MY + 6, "可正可负、<tspan font-weight=\"700\">没有范围</tspan>",
        INK, size=14, anchor="middle")
    f.t(1155, MY + 34, "<tspan font-weight=\"700\">它还不是概率。</tspan>",
        INK, True, 15, "middle")

    f.t(700, py + 266,
        "所有位置一起算就是一次矩阵乘："
        "<tspan font-weight=\"700\">［%s × %s］× ［%s × %s］＝［%s × %s］</tspan>"
        % (format(TGT, ","), format(D_MODEL, ","), format(D_MODEL, ","),
           format(VOCAB, ","), format(TGT, ","), format(VOCAB, ",")),
        INK, size=14.5, anchor="middle")
    f.t(700, py + 296,
        "⭐ 那块结果<tspan font-weight=\"700\">不小</tspan>："
        "bf16 存下来约 <tspan font-weight=\"700\">%.2f GiB</tspan>"
        "　——　它比 1.7 那张表里任何一个单独的张量都大。" % (LOGITS_B / 1024 ** 3),
        GY, size=14, anchor="middle")
    f.t(700, py + 330,
        "⛔ <tspan font-weight=\"700\">下一格才是 softmax</tspan>"
        "　——　它把这 %s 个分数变成 %s 个概率。" % (format(VOCAB, ","),
                                        format(VOCAB, ",")),
        INK, True, 15, "middle")
    f._pan = None

    yb = f.src(py + PH + 20,
               "词表 %s、残差流宽度 %s 取自 DeepSeek-V3 配置；"
               "LM head 的参数量与那 %.2f GiB 都是本脚本现算并 assert 住的。"
               % (format(VOCAB, ","), format(D_MODEL, ","), LOGITS_B / 1024 ** 3),
               "⚠️ 图里两个方块<tspan font-weight=\"700\">不成比例</tspan>"
               "（真实差 %.0f 倍，画不下）；层内部结构不在这一格，见专题一。" % RATIO)

    f.save("fig4-logits.svg", yb + 14)


main()
