# -*- coding: utf-8 -*-
r"""专题四 · §1.2c「一个位置的诉求，怎么一路变成 671B 个数」

⭐⭐⭐ 2026-09-22 现场把整条链自己复述了一遍，**前四步一字不差**：
  ① 词表 129,280 → 一个位置得到一个 129,280 长的向量；
  ② 往回传一步 → 变成一个 **7,168** 长的向量；
  ③ 一条序列就是这么多个 7,168；④ batch 翻几倍就再乘几倍。

⛔⛔ **而第五步是错的，偏偏它是最要紧的那一步。** 原话：
  「你会把这个无数个 7168 给它加起来，这就变成了一个平均的期望，
    这个期望就会逐渐地往前传播。」
  ——&#160;**不能把不同位置的那个 7,168 加起来。**

⭐ 为什么不能：每个位置的那份「责任」（δ）必须配**它自己那个位置**的输入。
  权重梯度是 `Σ_i δ_i ⊗ x_i` ——&#160;先把 δ 加起来再乘，
  等于 `(Σδ_i) ⊗ (Σx_i)`，**那是另一个量，不相等**。
  先加就把「第 3 个位置的责任」配到了「第 900 个位置的输入」上。

⭐⭐ 所以「相加」这件事**确实发生，但不发生在 δ 上，发生在权重梯度上**，
  而且是**每一层各自发生一次**。δ 从头到尾都是一张
  「位置 × 宽度」的表，位置这一维**一路保留到最后**。

⚠️ 一个诚实的例外，写进出处不画上图：**attention 会让不同位置的 δ 互相混合**
  ——&#160;因为前向时位置之间本来就互相看。但那是 attention 的结构造成的，
  不是「求平均」。线性层那一路，位置维是严格独立的。

⛔ 另：现场第三次说「4096 个位置」。**是 4,095** ——&#160;最后那个字没有下一个。
"""
from topic03_draw import Fig, BL, OR, GR, RD, PU, GY, INK, GY2

W = 1400

VOCAB = 129280
DMODEL = 7168          # DeepSeek-V3 hidden
SEQ = 4096
TGT = SEQ - 1          # 4,095
LAYERS = 61
NPARAM = "6,710 亿"


def main():
    f = Fig(W, "一个位置先拿到一个词表那么长的诉求，"
               "经过输出那张大表折成一个七千一百六十八维的向量，那就是这个位置的责任；"
               "一条序列有四千零九十五个这样的向量，它们排成一张表，"
               "位置这一维一路保留，不能先加起来。"
               "真正相加的是权重梯度：每一层里，每个位置的责任配上它自己那个位置的输入，"
               "乘出一份对权重的贡献，这些贡献才相加。"
               "先把责任加起来再乘，等于把这个位置的责任配到了别的位置的输入上，"
               "那是另一个量")

    y0 = f.header(
        "一个位置的诉求，怎么一路变成 <tspan font-weight=\"700\">%s个数</tspan>"
        % NPARAM,
        "⭐ 现场自己把这条链复述了一遍，<tspan font-weight=\"700\">前四步全对</tspan>；"
        "⛔ 第五步「把那些 7,168 加起来」——　"
        "<tspan font-weight=\"700\">那一步不能加，而它恰好最要紧</tspan>",
        [(GR, "Ⓐ 折成一股力"), (BL, "Ⓑ 位置维要保留"),
         (RD, "Ⓒ 加错了会怎样"), (OR, "Ⓓ 真正相加的地方")])

    # ══════════ Ⓐ 129,280 → 7,168 ═══════════════════════════════════
    PH = 336
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 第一步：<tspan font-weight=\"700\">词表那么长的诉求，"
                 "折成一个方向</tspan>", GR,
                 sub="⭐ 词表里每一个字，在模型内部都占一个方向；"
                     "按各自那个数加权，合成<tspan font-weight=\"700\">一股力</tspan>")

    # 左：129,280 长的一条
    f.box(60, py + 70, 300, 30, "#fce8e6", RD, 5)
    f.t(210, py + 90, "%s 个数" % format(VOCAB, ","), RD, True, 16, "middle")
    f.t(210, py + 124, "每个字一格　＝「你猜的 −　正确答案」", GY, size=13, anchor="middle")
    f.t(210, py + 148, "⛔ 概率≈0 的那些也在里面，只是它们那一格≈0",
        GY2, size=12, anchor="middle")

    f.line(380, py + 85, 470, py + 85, GR, 2.6)
    f.t(425, py + 68, "经过", GY2, size=12, anchor="middle")
    f.t(425, py + 124, "<tspan font-weight=\"700\">输出那张大表</tspan>",
        GR, True, 13, "middle")
    f.t(425, py + 146, "（%s × %s）" % (format(VOCAB, ","), format(DMODEL, ",")),
        GY2, size=11.5, anchor="middle")

    f.box(490, py + 70, 190, 30, "#e6f4ea", GR, 5)
    f.t(585, py + 90, "%s 个数" % format(DMODEL, ","), GR, True, 16, "middle")
    f.t(585, py + 124, "这个位置的<tspan font-weight=\"700\">责任</tspan>（δ）",
        INK, size=13, anchor="middle")
    f.t(585, py + 148, "⭐ 它是一个方向，不是一个目标值", GY2, size=12, anchor="middle")

    # 五个人拉环
    f.box(740, py + 54, 620, 250, "#e6f4ea", GR, 8)
    f.t(1050, py + 88, "这一步在干什么？<tspan font-weight=\"700\">合力。</tspan>",
        GR, True, 18, "middle")
    f.t(1050, py + 126,
        "像五个人拉同一个铁环：<tspan font-weight=\"700\">「的」往这边拽 0.60，"
        "</tspan>", INK, size=14.5, anchor="middle")
    f.t(1050, py + 152,
        "<tspan font-weight=\"700\">「了」「在」各往外推 0.20，「和」「有」各 0.10</tspan>。",
        INK, size=14.5, anchor="middle")
    f.t(1050, py + 184,
        "环最后只朝<tspan font-weight=\"700\">一个</tspan>方向动　——　那个合力，就是 δ。",
        INK, size=14.5, anchor="middle")
    f.t(1050, py + 224,
        "⛔ 所以<tspan font-weight=\"700\">「只管那个 −0.60」是不行的</tspan>：",
        RD, True, 14.5, "middle")
    f.t(1050, py + 250,
        "只说「往北走」，和说「往北走、同时离那三个人远点」——",
        GY, size=13.5, anchor="middle")
    f.t(1050, py + 274,
        "<tspan font-weight=\"700\">走出来的路线不一样。</tspan>",
        GY, True, 13.5, "middle")
    f._pan = None

    # ══════════ Ⓑ 一条序列：一张表，位置维保留 ══════════════════════
    PH2 = 266
    py2 = f.panel(0, py + PH + 20, W, PH2,
                  "Ⓑ 一条序列：<tspan font-weight=\"700\">%s 个这样的向量，"
                  "排成一张表</tspan>" % format(TGT, ","), BL,
                  sub="⛔ 是 %s 不是 %s ——　最后那个字没有「下一个」。"
                      "⭐ 而这张表的<tspan font-weight=\"700\">「位置」那一维，"
                      "要一路保留到最后</tspan>"
                      % (format(TGT, ","), format(SEQ, ",")))

    # 一张 T×d 的表
    # ⛔ 第一版把它画成「一个浅蓝框 ＋ 几根白线」——&#160;渲染出来是**一个空白框**：
    #   浅蓝太浅、白线在浅蓝上又看不见，读者根本看不出这是「很多行」。
    #   ⭐ 判据：**这一格要讲的是「有很多行、而且行不能合并」，
    #     那就必须把行画成一眼数得出来的形状。**
    TX, TY, TW, TH = 120, py2 + 66, 420, 132
    ROWS = 9
    RH = TH / float(ROWS)
    for k in range(ROWS):
        f.box(TX, TY + k * RH, TW, RH - 2.0,
              "#e8f0fe" if k % 2 == 0 else "#d2e3fc", BL, 2, sw=0.8)
    f.t(TX + TW + 12, TY + RH - 3, "位置 1", GY2, size=11)
    f.t(TX + TW + 12, TY + 2 * RH - 3, "位置 2", GY2, size=11)
    f.t(TX + TW + 12, TY + TH - 3, "位置 %s" % format(TGT, ","), GY2, size=11)
    f.t(TX + TW / 2, TY - 14, "宽 %s（每个位置的 δ）" % format(DMODEL, ","),
        BL, True, 13, "middle")
    f.t(TX - 12, TY + TH / 2 + 5, "高 %s" % format(TGT, ","), BL, True, 13, "end")
    f.t(TX + TW / 2, TY + TH + 26,
        "<tspan font-weight=\"700\">一行 ＝ 一个位置的那股合力</tspan>",
        INK, size=14, anchor="middle")

    f.box(620, py2 + 58, 740, 150, "#e8f0fe", BL, 8)
    f.t(990, py2 + 94, "batch ＝ 2 就是两张这样的表叠起来", BL, True, 16.5, "middle")
    f.t(990, py2 + 128,
        "<tspan font-weight=\"700\">%s 行 × %s 列</tspan>"
        % (format(TGT * 2, ","), format(DMODEL, ",")),
        INK, True, 18, "middle")
    f.t(990, py2 + 166,
        "⭐ 每一行在<tspan font-weight=\"700\">生出来的那一刻就已经除过 %s 了</tspan>"
        % format(TGT * 2, ","), OR, size=14, anchor="middle")
    f.t(990, py2 + 192,
        "——　「平均」那个除法，早在种子那一步就做掉了。",
        GY, size=13.5, anchor="middle")
    f._pan = None

    # ══════════ Ⓒ 误解 vs 实际 ═════════════════════════════════════
    PH3 = 344
    py3 = f.panel(0, py2 + PH2 + 20, W, PH3,
                  "Ⓒ <tspan font-weight=\"700\">现场那第五步：把这些 %s 加起来</tspan>"
                  "　——　这一步不能做" % format(DMODEL, ","), RD,
                  sub="⭐ 因为每一行的责任，必须配<tspan font-weight=\"700\">"
                      "它自己那一行的输入</tspan>")

    f.box(50, py3 + 52, 640, 268, "#fce8e6", RD, 8)
    f.t(370, py3 + 88, "❌ 先把 %s 行加成一行，再往回传" % format(TGT, ","),
        RD, True, 17, "middle")
    f.t(370, py3 + 130,
        "<tspan font-weight=\"700\">（所有 δ 加起来）⊗（所有输入加起来）</tspan>",
        INK, True, 16, "middle")
    f.t(370, py3 + 174,
        "⛔ 这样一加，<tspan font-weight=\"700\">「第 3 个位置的责任」"
        "</tspan>就被配到了", INK, size=14.5, anchor="middle")
    f.t(370, py3 + 200,
        "<tspan font-weight=\"700\">「第 900 个位置的输入」</tspan>上。",
        INK, size=14.5, anchor="middle")
    f.t(370, py3 + 240,
        "⭐ 而「谁在什么上下文里提的要求」", GY, size=13.5, anchor="middle")
    f.t(370, py3 + 264,
        "<tspan font-weight=\"700\">正是这份梯度全部的信息</tspan>　——　加掉就没了。",
        GY, size=13.5, anchor="middle")
    f.t(370, py3 + 300, "⛔ 它是另一个量，跟正确答案不相等。",
        RD, True, 14, "middle")

    f.box(710, py3 + 52, 640, 268, "#e6f4ea", GR, 8)
    f.t(1030, py3 + 88, "✅ 一行配一行，乘完之后才加", GR, True, 17, "middle")
    f.t(1030, py3 + 130,
        "<tspan font-weight=\"700\">Σ（第 i 行的责任 ⊗ 第 i 行的输入）</tspan>",
        INK, True, 16, "middle")
    f.t(1030, py3 + 174,
        "每个位置<tspan font-weight=\"700\">各乘各的</tspan>，"
        "得到各自那一份", INK, size=14.5, anchor="middle")
    f.t(1030, py3 + 200,
        "对权重的贡献　——　<tspan font-weight=\"700\">这些贡献才相加</tspan>。",
        INK, size=14.5, anchor="middle")
    f.t(1030, py3 + 240,
        "⭐⭐ 所以「相加」这件事<tspan font-weight=\"700\">确实发生</tspan>，",
        GR, True, 14.5, "middle")
    f.t(1030, py3 + 266,
        "只是它<tspan font-weight=\"700\">不发生在 δ 上，发生在权重梯度上</tspan>。",
        INK, size=14.5, anchor="middle")
    f.t(1030, py3 + 300,
        "而且<tspan font-weight=\"700\">每一层各自发生一次</tspan>。",
        GY, True, 14, "middle")
    f._pan = None

    # ══════════ Ⓓ 全链 ═════════════════════════════════════════════
    PH4 = 300
    py4 = f.panel(0, py3 + PH3 + 20, W, PH4,
                  "Ⓓ 于是整条链是这样的　——　<tspan font-weight=\"700\">"
                  "位置维一路走到底，每层顺手结一次账</tspan>", OR)

    STEPS = (
        (RD, "一个位置", "%s 个数" % format(VOCAB, ","), "词表那么长的诉求"),
        (GR, "折一次", "%s 个数" % format(DMODEL, ","), "合力 ＝ 这个位置的 δ"),
        (BL, "一条序列", "%s 行" % format(TGT, ","), "排成一张表，行不相加"),
        (PU, "穿 %d 层" % LAYERS, "每层一次", "δ 配本行输入 → 权重贡献"),
        (OR, "结账", "%s个数" % NPARAM, "各层各权重，贡献求和"),
    )
    for i, (col, name, num, desc) in enumerate(STEPS):
        x = 34 + i * 274
        f.box(x, py4 + 48, 240, 148, "#fff", col, 8, sw=1.6)
        f.box(x, py4 + 48, 240, 4, col, col, 2)
        f.t(x + 120, py4 + 84, name, col, True, 16, "middle")
        f.t(x + 120, py4 + 118, num, INK, True, 17, "middle")
        f.t(x + 120, py4 + 158, desc, GY, size=12.5, anchor="middle")
        if i < len(STEPS) - 1:
            f.line(x + 244, py4 + 122, x + 268, py4 + 122, GY2, 2.0)

    f.t(700, py4 + 236,
        "⭐⭐ 所以现场说的「无数个 7,168 加起来变成一个平均的期望」"
        "——　<tspan font-weight=\"700\">「加起来」对，「在 7,168 这一层加」不对</tspan>。",
        INK, size=15, anchor="middle")
    f.t(700, py4 + 264,
        "<tspan font-weight=\"700\">加法要等到最后一格才发生，而且加的是"
        "对权重的贡献，不是 δ 本身。</tspan>",
        OR, True, 15, "middle")
    f._pan = None

    yb = f.band(py4 + PH4 + 18, "ok", "⭐ 两句话记住这张图", [
        "<tspan font-weight=\"700\">δ 从头到尾是一张「位置 × 宽度」的表</tspan>"
        "　——　位置那一维一路保留，"
        "因为第 i 个位置的责任<tspan font-weight=\"700\">只能</tspan>配第 i 个位置的输入。",
        "<tspan font-weight=\"700\">求和发生在权重梯度上，每一层各结一次账</tspan>"
        "　——　所有位置、所有样本对同一个权重的贡献加起来，"
        "才是这一步要挪它的那个数。",
    ])

    yb = f.src(yb + 10,
               "线性层：权重梯度 ＝ Σ_i（δ_i ⊗ x_i），沿位置维收缩；"
               "而对输入的梯度是逐位置的，不收缩。"
               "⚠️ 一个诚实的例外：<tspan font-weight=\"700\">attention 会让不同位置的 δ "
               "互相混合</tspan>　——　因为前向时位置之间本来就互相看；",
               "但那是 attention 的结构造成的，不是「求平均」，线性层那一路位置维严格独立。"
               "词表 %s、宽度 %s、层数 %d 取自 DeepSeek-V3 配置。"
               % (format(VOCAB, ","), format(DMODEL, ","), LAYERS))

    f.save("fig4-fold.svg", yb + 14)


main()
