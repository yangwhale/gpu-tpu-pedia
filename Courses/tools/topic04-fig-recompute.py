# -*- coding: utf-8 -*-
r"""专题四 · §2.2「全量重算：用三分之一的算力，换掉四十分之三十九的显存」

⭐⭐⭐ 2026-09-16 新画。这张图只干一件事：**把兑换比例画成两根不等长的条。**

⭐⭐ 取舍：**两边用同一种视觉量（条长 ＝ 相对变化），不要一边画 TiB 一边画 TFLOPs。**
  ⛔ 单位不同的两根条并排放，读者第一反应是去比长度 ——&#160;
    而那个比较毫无意义。所以这里两边都归一到「原来是 100%」。
  ⭐ 判据：**并排的两根条必须可比；不可比就别并排。**

⭐ 这张图想让人记住的是**形状的不对称**：
  付出那边几乎没变（多一小截），省下那边几乎归零。
  ——&#160;「夸张到不像是个权衡」这句话，图比字有说服力。

⛔⛔ 刻意没画的：
  ① **选择性重算。** 它是下一张图（fig4-per-byte）的事，
     混进来会让「全量这笔交易有多划算」失焦。
  ② **时间轴 / 什么时候重算。** 那属于实现细节，本节只算账。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE, LINE2)

W = 1400

ACT_RAW_TIB, ACT_REMAT_GIB = 4.15, 106.75
FWD_BWD, WITH_REMAT = 3, 4

MEM_KEPT = ACT_REMAT_GIB / (ACT_RAW_TIB * 1024) * 100      # 省完还剩百分之几
COMP_MORE = (WITH_REMAT - FWD_BWD) / float(FWD_BWD) * 100  # 算力多付百分之几
assert 2.4 < MEM_KEPT < 2.6          # ≈ 2.5% ——&#160;也就是「省掉 40 分之 39」
assert abs(COMP_MORE - 100.0 / 3) < 1e-6


def main():
    f = Fig(W, "全量重算这笔交易的两边：算力从三倍变四倍，只多付三分之一；"
               "而激活显存从四点一五 TiB 掉到一百零六 GiB，只剩原来的百分之二点五。"
               "付出那边几乎没变，省下那边几乎归零 —— "
               "所以在大模型训练里它默认就是开着的")

    y0 = f.header(
        "全量重算　——　<tspan font-weight=\"700\">"
        "用三分之一的算力，换掉四十分之三十九的显存</tspan>",
        "⭐ 两边都归一到「原来 ＝ 100%」"
        "　·　⛔ 否则一边 TiB 一边 TFLOPs，<tspan font-weight=\"700\">"
        "两根条没法比</tspan>",
        [(OR, "付出：算力"), (GR, "省下：显存")])

    # ══════════ Ⓐ 两根条 ═════════════════════════════════════════
    PH = 400
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 这笔交易的两边 ——　<tspan font-weight=\"700\">"
                 "形状不对称到不像是个权衡</tspan>", BL,
                 sub="⭐ 条长代表「相对原来剩多少 / 涨多少」")

    BX, BW = 300, 880

    def bar(y, col, fill, frac, left, right, note):
        f.box(BX, y, BW, 54, "#f8f9fa", LINE2, 6)
        f.box(BX, y, BW * frac, 54, fill, col, 6)
        f.t(BX - 16, y + 34, left, INK, True, 16.5, "end")
        f.t(BX + BW + 16, y + 34, right, col, True, 18)
        f.t(BX + 14, y + 80, note, GY, size=13.5)

    # 算力：原来 100%，现在 133%
    f.t(BX - 16, py + 34, "付出　算力", OR, True, 19, "end")
    f.box(BX, py + 12, BW * 0.66, 44, "#fef7e0", OR, 6)
    f.t(BX + BW * 0.33, py + 40, "原来：3×（前向 1 ＋ 反向 2）", OR, True, 15, "middle")
    f.box(BX, py + 68, BW * 0.88, 44, "#fef7e0", OR, 6)
    f.box(BX + BW * 0.66, py + 68, BW * 0.22, 44, OR, OR, 6)
    f.t(BX + BW * 0.33, py + 96, "现在：4×（多跑一遍前向）", "#fff", True, 15, "middle")
    f.t(BX + BW * 0.88 + 14, py + 96, "＋%d%%" % round(COMP_MORE), OR, True, 22)
    f.t(BX + 14, py + 136, "⭐ 多出来的那一小截，就是「再跑一遍前向」",
        GY, size=13.5)

    # 显存：原来 100%，现在 2.5%
    f.t(BX - 16, py + 210, "省下　显存", GR, True, 19, "end")
    f.box(BX, py + 188, BW, 44, "#e6f4ea", GR, 6)
    f.t(BX + BW / 2.0, py + 216, "原来：%.2f TiB 激活" % ACT_RAW_TIB,
        GR, True, 15, "middle")
    f.box(BX, py + 244, BW, 44, "#f8f9fa", LINE2, 6)
    f.box(BX, py + 244, BW * MEM_KEPT / 100.0, 44, GR, GR, 6)
    f.t(BX + BW * MEM_KEPT / 100.0 + 14, py + 272,
        "现在：%.2f GiB　——　只剩 %.1f%%" % (ACT_REMAT_GIB, MEM_KEPT),
        GR, True, 17)
    f.t(BX + 14, py + 312, "⛔ 这根条短到几乎看不见 ——　"
                           "<tspan font-weight=\"700\">那正是这张图要说的事</tspan>",
        GY, size=13.5)
    f._pan = None

    # ══════════ Ⓑ 所以默认开着 ═══════════════════════════════════
    PH2 = 256
    py2 = f.panel(0, py + PH + 22, W, PH2,
                  "Ⓑ 所以值得讨论的从来不是「开不开」", GR,
                  sub="⭐ 而是<tspan font-weight=\"700\">开到哪一档</tspan>")

    LEVELS = (
        (GY2, "#f1f3f4", "不开", "全留", "只在显存宽裕时才合理"),
        (BL, "#e8f0fe", "选择性", "按比值挑着扔",
         "⭐ 约 6% 算力换约 77% 显存 ——　性价比最高"),
        (GR, "#e6f4ea", "全量", "每层只留入口那一份",
         "33% 算力换 97% ——　显存实在不够时"),
    )
    for i, (col, fill, nm, how, note) in enumerate(LEVELS):
        x = 64 + i * 440
        f.box(x, py2 + 34, 416, 148, fill, col, 8)
        f.box(x, py2 + 34, 416, 4, col, col, 2)
        f.t(x + 208, py2 + 70, nm, col, True, 20, "middle")
        f.t(x + 208, py2 + 100, how, INK, True, 15.5, "middle")
        f.t(x + 208, py2 + 140, note, GY, size=13, anchor="middle")
        if i:
            f.line(x - 20, py2 + 108, x - 4, py2 + 108, GY2, 1.6)
    f.t(700, py2 + 214, "⛔ 显存本来就宽裕的时候（小模型、短序列），"
                        "<tspan font-weight=\"700\">重算就是纯亏</tspan>",
        RD, True, 15, "middle")
    f._pan = None

    yy = f.band(py2 + PH2 + 22, "warn", "但这笔账不能照抄别人的", [
        "⛔⛔ <tspan font-weight=\"700\">同一个模型、同一个开关，换一个规模，"
        "收益可能从正的变成负的。</tspan>"
        "原因不神秘：<tspan font-weight=\"700\">重算改变的是计算与访存的配比</tspan>，"
        "而这个配比在不同并行配置、不同芯片数下本来就不同。",
        "⭐ 由此推出一条通用规则（<tspan font-weight=\"700\">不只对重算成立</tspan>）："
        "<tspan font-weight=\"700\">凡是会改变数据分片形状的参数，都不能跨规模照抄</tspan>"
        " ——&#160;小规模上验过的结论，到目标规模上必须重验。",
    ], fold=True)

    yy = f.src(yy + 24,
               "⚠️ %.2f TiB / %.2f GiB 是<tspan font-weight=\"700\">自己按算子推的估算</tspan>"
               "（V3 的 config ＋ 官方参考实现的 MLA 前向），"
               "<tspan font-weight=\"700\">当量级看</tspan>；"
               "就算差一倍，「不对称」这个结论也不变"
               % (ACT_RAW_TIB, ACT_REMAT_GIB),
               "⭐ 「3× →&#160;4×」是矩阵乘口径：重算等于把前向那一遍再买一次，"
               "所以多出来的正好是 <tspan font-weight=\"700\">1/3</tspan>",
               "📌 Ⓑ 里「选择性」那一档的 6% / 77%，"
               "推导过程与名次表见<tspan font-weight=\"700\">下一张图与本节正文</tspan>；"
               "DeepSeek-V3 报告（<tspan font-weight=\"700\">arXiv 2412.19437</tspan>）"
               "明写他们重算全部 RMSNorm 与 MLA 上投影 ——&#160;正是那一档")
    f.save("fig4-recompute.svg", yy + 6)


main()
