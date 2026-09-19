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

# ⛔⛔ 2026-09-19 T04：峰值 ＝ 存档点 ＋ **当前正在重算的那一层**。
#   原来只算存档点（106.75），漏掉在算的那一层（71.14）——&#160;低估 67%。
#   ⭐ 这一讲自己在 §5.4 的小例子和 §2.2 图 Ⓒ 用的都是正确口径，只有这个头号数字没做。
ACT_RAW_TIB = 4.15
ACT_REMAT_GIB = 106.75 + 71.14       # 存档点 ＋ 在算的那一层 ＝ 177.89
FWD_BWD, WITH_REMAT = 3, 4

MEM_KEPT = ACT_REMAT_GIB / (ACT_RAW_TIB * 1024) * 100      # 省完还剩百分之几
COMP_MORE = (WITH_REMAT - FWD_BWD) / float(FWD_BWD) * 100  # 算力多付百分之几
assert 4.1 < MEM_KEPT < 4.3          # ≈ 4.2% ——&#160;也就是「省掉 24 分之 23」
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
         "⭐ 约 1.9% 算力换约 77% 显存 ——　性价比最高"),
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


    # ══════════ Ⓒ 跨层的存档点密度 ═══════════════════════════════════
    # ⭐⭐⭐ 2026-09-17 新增。Ⓑ 讲的是**一层之内留哪些张量**，
    #   而这里讲的是**隔几层留一个存档点** ——&#160;两个完全不同的轴，
    #   原来图上只有前一个。
    # ⛔ 这一格的落点是个反直觉的事实：**存得越少不是越省。**
    #   因为重算一段的时候，那一段的中间结果得**同时在场**。
    #   所以有一个最优点，而它在 √L。
    # 📌 ZeRO 论文 §3.2 原话：activation checkpointing「把激活显存降到
    #   大约总量的平方根，代价是 33% 的重算开销」——&#160;那个「平方根」就是这一格。
    L_DEMO = 16
    def _cost(k):
        """隔 k 层存一个：存档点 L/k 份 ＋ 重算时一段 k 层同时在场。"""
        return L_DEMO / k + k
    assert min(range(1, L_DEMO + 1), key=_cost) == 4 == int(L_DEMO ** .5), \
        "最优密度应该落在 √L 上"
    assert abs(_cost(1) - _cost(L_DEMO)) < 1e-9, \
        "两头要一样高 ——　「全存」和「只存首尾」同样贵，这是这一格的全部意思"

    # ⛔⛔ 2026-09-20：上面那个 L/k ＋ k 有一个**没说出来的前提** ——
    #   它把「一个存档点」和「一层的完整激活」当成一样大（两项同单位才能相加）。
    #   ⭐ 而 V3 上这两个量是 1.75 : 71.14 ≈ **1 : 40**。
    #   正确的目标函数是 c·(L/k) + a·k，最优 k ＝ √(cL/a)：
    _C_CKPT, _A_LAYER, _L_V3 = 1.75, 71.14, 61        # GiB@128K / GiB@128K / 层
    _K_OPT = (_C_CKPT * _L_V3 / _A_LAYER) ** 0.5
    assert abs(_K_OPT - 1.23) < 0.02, "V3 的最优密度算出来是 %.2f" % _K_OPT
    #   ⇒ **k ≈ 1.2，也就是「每层都存」** ——&#160;跟「最优点在中间」正好相反。
    #   ⛔ 所以这一格画的是**那个玩具比例下**的道理，不是 V3 的处方。
    #     判据：**一条「有最优点」的曲线，先问它两项是不是同一个量级。**

    PH3 = 332
    py3 = f.panel(0, py2 + PH2 + 22, W, PH3,
                  "Ⓒ ⭐⭐⭐ 换一个轴：<tspan font-weight=\"700\">"
                  "隔几层留一个存档点</tspan>", PU,
                  sub="⛔ 注意这跟 Ⓑ <tspan font-weight=\"700\">不是同一件事</tspan>"
                      "　——　Ⓑ 是一层之内留哪些，这里是隔几层留一个")

    SX, SW = 300, 800
    CELL = SW / float(L_DEMO)
    ROWS = (
        (1,  GY2, "全存",        "每一层都留"),
        (4,  GR,  "隔 4 层留一个", "⭐ L ＝ 16，而 √16 ＝ 4"),
        (16, RD,  "只存开头那一个", "重算时整条 16 层都得在场"),
    )
    for i, (k, col, nm, how) in enumerate(ROWS):
        ry = py3 + 56 + i * 76
        f.t(24, ry + 16, nm, col, True, 15)
        f.t(24, ry + 38, how, GY2, size=11.5)
        for j in range(L_DEMO):
            x = SX + j * CELL
            keep = (j % k == 0)
            f.box(x + 1, ry, CELL - 2, 34,
                  "#fff", col if keep else LINE, 3, sw=1.6 if keep else 1)
            if keep:
                f.t(x + CELL / 2, ry + 23, "💾", col, True, 13, "middle")
        c = _cost(k)
        f.t(SX + SW + 26, ry + 14, "存 %d 份" % (L_DEMO // k), INK, True, 13)
        f.t(SX + SW + 26, ry + 34, "＋ 段内 %d 层" % k, GY, size=12)
        f.t(SX + SW + 190, ry + 24, "＝ %d" % c,
            col, True, 19)

    f.t(SX + SW + 190, py3 + 40, "同时在场", GY, True, 12)
    f.t(SX, py3 + 40, "⭐ 一个方块 ＝ 一层。"
        "<tspan font-weight=\"700\">💾 就是留下来的存档点</tspan>", GY2, size=12)

    f.t(700, py3 + 296, "⭐⭐⭐ 两头一样高　——　"
        "<tspan font-weight=\"700\">存得越少，并不是越省</tspan>。"
        "少存一个存档点，就要多扛一段重算时的中间结果。",
        INK, True, 15, "middle")
    f._pan = None

    yy = f.band(py3 + PH3 + 22, "warn", "但这笔账不能照抄别人的", [
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
               "所以多出来的正好是 <tspan font-weight=\"700\">1/3</tspan>"
               "　——　ZeRO 论文原话也是这个数（33% re-computation overhead）",
               "⭐ Ⓒ 那三个「同时在场」由 L/k ＋ k 当场算出，"
               "脚本内 assert 最优点落在 √L、且两头等高　——　"
               "<tspan font-weight=\"700\">这一格的全部意思就是那个等高</tspan>",
               "⛔ Ⓑ 里那两个百分比<tspan font-weight=\"700\">都按「一个 step」做分母</tspan>"
               "（前向 ＋ 反向 ＝ 3 遍）。"
               "<tspan font-weight=\"700\">早先「选择性」那一档写的是 6%，那是拿一遍前向当分母的旧口径</tspan>"
               "——&#160;并排摆着不能比，已统一。",
               "📌 Ⓑ 里「选择性」那一档的 1.9% / 77%，"
               "推导过程与名次表见<tspan font-weight=\"700\">下一张图与本节正文</tspan>；"
               "DeepSeek-V3 报告（<tspan font-weight=\"700\">arXiv 2412.19437</tspan>）"
               "明写他们重算全部 RMSNorm 与 MLA 上投影 ——&#160;正是那一档")
    f.save("fig4-recompute.svg", yy + 6)


main()
