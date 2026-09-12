# -*- coding: utf-8 -*-
r"""专题三 · §五「MLA 为什么能压缩」（2026-09-13 加，夜间 20 轮 · R1）。

⭐⭐ 这张图回答的不是「MLA 是什么」（那是 fig3-knob1 的活），
   而是 **「它凭什么敢这么压」**。现场原话：
   「像 MLA 这种东西，它为什么能压缩？这里边跟信息论有关的东西，对吧？」

三格，是一条推理链，不是三个知识点：

  ① **先算一笔账** ——&nbsp;MHA 每 token 每层存 32,768 个数，
     而这 32,768 个数是从一个 7,168 维的 h **算出来的**。
     ⭐ 一次确定性映射不会凭空造出信息 ——&nbsp;
     **复印 128 份，信息还是那一张。** 所以至少 4.57 倍是纯冗余。

  ② **那为什么大家一直存 32,768** ——&nbsp;因为存的是「算好的结果」。
     存原料省地方，但读的时候要重算：这是一笔**拿计算换存储**的交易。
     MLA 第一步就是改存原料；第二步才是真正的赌注 ——&nbsp;
     原料也不全存，压到 512。**这等于强制 W^K 的秩 ≤ 512。**

  ③ **凭什么敢赌** ——&nbsp;因为「K/V 投影本来就低秩」在 MLA 之前
     就已经有一批**事后**分解的工作验证过了（Eigen Attention / Palu / LoRC）。
     ⭐⭐ MLA 的不同不在于发现低秩，**在于从第一天就按 512 训**。
     —— 这跟 §六 里「推理期稀疏 vs native 稀疏」是同一个故事，
     是这一讲的暗线：**事后压 vs 从头按压缩训。**

📌 所有倍数当场算并断言，不写死。
"""
from topic03_draw import (Fig, wpx, BL, OR, GR, RD, GY, PU, INK,
                          GY2, LINE, LINE2, BG2)

W = 1400
PW = 440                      # 三栏等宽
PX = [0, 480, 960]


def main():
    # ── 数字：全部当场算 ────────────────────────────────────────
    D_V3, NH, DH = 7168, 128, 128          # DeepSeek-V3 hidden / 头数 / 每头维
    KV = 2 * NH * DH                       # MHA 每 token 每层要存的数
    DC, DR = 512, 64                       # MLA 的隐向量 ＋ 解耦 RoPE
    LAT = DC + DR
    free = KV / float(D_V3)                # 白送的那一段（纯冗余）
    bet = D_V3 / float(LAT)                # 赌出来的那一段
    tot = KV / float(LAT)
    assert KV == 32768 and LAT == 576
    assert abs(free - 4.571) < .01 and abs(tot - 56.9) < .05
    assert abs(free * bet - tot) < 1e-6    # 两段相乘 = 总倍数

    f = Fig(W, "MLA 为什么能压缩：三格一条推理链 —— 先算出 32768 个数里"
               "至少 4.57 倍是纯冗余，再说明存原料是拿计算换存储，"
               "最后给出「K/V 投影本来就低秩」的事前证据")
    f.marks = set()
    y0 = f.header(
        "MLA 凭什么敢这么压　——　一笔信息账，三步推出来",
        "⭐ 这张图不讲 MLA 是什么（那在上一张），只讲<tspan font-weight=\"700\">它为什么可行</tspan>",
        [(GY, "存下来的（结果）"), (BL, "原料"), (GR, "压缩后的原料"),
         (OR, "拿计算换存储"), (PU, "赌注")])

    # ══ ① 先算一笔账 ════════════════════════════════════════════
    x = PX[0]
    ph = 348
    py = f.panel(x, y0, PW, ph, "① 先算一笔账", BL,
                 sub="这 32,768 个数是哪来的")

    yy = py + 30
    # 原料：一个窄盒
    f.cell(x + 26, yy, 96, 46, "h", "7,168 维", BL)
    f.t(x + 26, yy + 66, "一个 token 在这一层的全部内容", GY, size=11.5, w=200)

    # 映射箭头
    f.line(x + 130, yy + 23, x + 214, yy + 23, GY2, 1.6)
    f.t(x + 172, yy + 14, "W<tspan baseline-shift=\"super\" font-size=\"8\">K</tspan>"
        " · W<tspan baseline-shift=\"super\" font-size=\"8\">V</tspan>",
        GY, size=11.5, anchor="middle")
    f.t(x + 172, yy + 40, "线性、确定", GY2, size=11, anchor="middle")

    # 结果：一个宽盒，画成 128 个小格暗示「复印 128 份」
    bx, bw = x + 222, 192
    f.box(bx, yy, bw, 46, "#fff", GY, 6)
    for i in range(16):
        f.box(bx + 4 + i * (bw - 8) / 16.0, yy + 5, (bw - 8) / 16.0 - 1.5, 36,
              BG2, LINE2, 2)
    f.t(bx + bw / 2.0, yy + 21, "K ＋ V", GY, True, 12, "middle")
    f.t(bx + bw / 2.0, yy + 36, "32,768 维（128 个头）", GY2, size=11,
        anchor="middle")
    f.t(bx, yy + 66, "MHA 每 token 每层真正缓存的东西", GY, size=11.5, w=200)

    yy += 96
    f.line(x + 26, yy + 8, x + PW - 26, yy + 8, LINE, 1, arrow=False)

    yy += 30
    f.t(x + 26, yy, "⭐ 一次确定性映射不会凭空造出信息",
        BL, True, 13, cls="svglbl")
    yy = f.lines(x + 26, yy + 24, PW - 52, [
        "把一张纸复印 128 份，信息还是那一张 ——",
        "复印件再多，也回答不了原件回答不了的问题。"], 11.5, 19)

    yy += 12
    f.box(x + 26, yy, PW - 52, 56, "#fff", BL, 8)
    f.t(x + 40, yy + 23, "32,768 ÷ 7,168 ＝ %.2f 倍" % free, BL, True, 13.5,
        cls="svglbl")
    f.t(x + 40, yy + 43, "这一段是<tspan font-weight=\"700\">白送的</tspan> —— 不损失任何东西", GY, size=11.5)

    # ══ ② 那为什么还是存结果 ════════════════════════════════════
    x = PX[1]
    py = f.panel(x, y0, PW, ph, "② 那为什么一直存 32,768", OR,
                 sub="因为存的是结果，不是原料")

    yy = py + 26
    for lab, sub, col, note in [
        ("存结果", "读的时候直接用", GY, "省计算，费地方"),
        ("存原料", "读的时候现算一遍", OR, "省地方，费计算"),
    ]:
        f.cell(x + 26, yy, 130, 44, lab, sub, col)
        f.t(x + 172, yy + 27, note, col, size=12)
        yy += 58

    f.t(x + 26, yy + 4, "⭐ 这是一笔交易，不是一个错误 —— 早期两边都不紧张",
        OR, size=11.5, w=PW - 52)

    yy += 30
    f.line(x + 26, yy, x + PW - 26, yy, LINE, 1, arrow=False)
    yy += 26

    f.t(x + 26, yy, "MLA 走了两步，第二步才是关键", INK, True, 13.5,
        cls="svglbl")
    yy += 24
    f.box(x + 26, yy, PW - 52, 40, "#fff", LINE, 8)
    f.t(x + 40, yy + 25, "第一步　改成存原料　7,168", GY, size=12.5)
    f.t(x + PW - 40, yy + 25, "白送 %.2f×" % free, BL, True, 12, "end")
    yy += 50

    f.box(x + 26, yy, PW - 52, 66, "#fff", PU, 8)
    f.box(x + 26, yy, 4, 66, PU, PU, 2)
    f.box(x + 28, yy, 3, 66, "#fff", "#fff", 0)
    f.t(x + 44, yy + 25, "第二步　原料也不全存，压到 512", PU, True, 12.5)
    f.t(x + 44, yy + 48, "⭐ 这等于强制 W<tspan baseline-shift=\"super\" font-size=\"8\">K</tspan> 的秩 ≤ 512 —— 这一步是赌注",
        GY, size=11.5)
    f.t(x + PW - 40, yy + 25, "赌 %.1f×" % bet, PU, True, 12, "end")

    # ══ ③ 凭什么敢赌 ════════════════════════════════════════════
    x = PX[2]
    py = f.panel(x, y0, PW, ph, "③ 凭什么敢赌", GR,
                 sub="「K/V 本来就低秩」是先有证据的")

    yy = py + 26
    for who, what, gain in [
        ("Eigen Attention", "对<tspan font-weight=\"700\">已训好</tspan>的模型做低秩分解", "省 40%"),
        ("Palu", "分组头低秩 ＋ 自动分配秩", "省 50%"),
        ("LoRC", "逐层分配不同的秩", "无损为主"),
    ]:
        f.box(x + 26, yy, PW - 52, 40, "#fff", LINE, 8)
        f.t(x + 40, yy + 25, who, GY, True, 12)
        f.t(x + 40 + wpx(who, 12) + 14, yy + 25, what, GY2, size=11)
        f.t(x + PW - 40, yy + 25, gain, GR, True, 12, "end")
        yy += 48

    f.t(x + 26, yy + 2, "⭐ 三家都在 MLA 前后独立做到 ——", GR, size=11.5)
    f.t(x + 26, yy + 21, "「低秩」不是 DeepSeek 猜的，是量出来的。", GY,
        size=11.5)

    yy += 44
    f.line(x + 26, yy, x + PW - 26, yy, LINE, 1, arrow=False)
    yy += 26

    f.t(x + 26, yy, "⭐⭐ 那 MLA 的不同在哪", INK, True, 13.5, cls="svglbl")
    yy += 24
    f.box(x + 26, yy, PW - 52, 70, "#fff", GR, 8)
    f.box(x + 26, yy, 4, 70, GR, GR, 2)
    f.box(x + 28, yy, 3, 70, "#fff", "#fff", 0)
    f.t(x + 44, yy + 24, "上面三家是<tspan font-weight=\"700\">近似</tspan>：训完再去凑", GY, size=11.5)
    f.t(x + 44, yy + 46, "MLA 是<tspan font-weight=\"700\">约束</tspan>：从第一天就按 512 训", GR, True, 12.5)

    # ══ 落点带 ══════════════════════════════════════════════════
    yy = y0 + ph + 22
    yy = f.band(yy, "info", "一句话：你存下来的从来不是信息，是信息的一个展开", [
        "32,768 → 7,168 是<tspan font-weight=\"700\">白送的 %.2f 倍</tspan>"
        "（信息论管着，谁都拿得到）；"
        "7,168 → 576 是<tspan font-weight=\"700\">赌出来的 %.1f 倍</tspan>"
        "（要重训，要承担掉点风险）。" % (free, bet),
        "⭐ 合起来 <tspan font-weight=\"700\">%.1f×</tspan>。"
        "看任何一个压缩方案，都值得先把这两段拆开问："
        "<tspan font-weight=\"700\">哪一段是白送的，哪一段是赌的？</tspan>" % tot,
    ])

    yy = f.band(yy + 14, "ok", "⭐⭐ 记住这条暗线 —— 今天它一共出现四次，这是第一次", [
        "<tspan font-weight=\"700\">事后压</tspan>（拿训好的模型去凑）"
        "对上 <tspan font-weight=\"700\">从头按压缩训</tspan>（把约束写进训练）。",
        "这里是 Eigen Attention 对 MLA；"
        "到了 §六，同一个对立会换成"
        "<tspan font-weight=\"700\">推理期稀疏 对 native 稀疏</tspan>。"
        "⭐ 两次的结论一样：<tspan font-weight=\"700\">后者掉点小得多。</tspan>",
    ])

    yy = f.band(yy + 14, "warn", "口径，别讲过头", [
        "⚠️ 「MLA 比 MHA 还好一点」这句话<tspan font-weight=\"700\">依赖口径</tspan>："
        "DeepSeek 自己的对照是<tspan font-weight=\"700\">对齐总参数量</tspan>后比的；",
        "社区复现里也有「带 RoPE 时 MHA 略好」的结果。"
        "⭐ 稳妥的说法是<tspan font-weight=\"700\">「在同等预算下不吃亏」</tspan>，"
        "而不是「压缩使它变强」。",
    ])

    yy = f.src(yy + 16,
               "维度出自 DeepSeek-V2 论文 §2.1（arXiv 2405.04434）与 V3 config："
               "d=7168, n_h=128, d_h=128, d_c=512, d_h^R=64；三个倍数由脚本当场算并断言",
               "事后低秩：Eigen Attention（EMNLP Findings 2024）、Palu、LoRC　"
               "「MLA 优于 MHA」的口径见 DeepSeek-V2 仓库 issue #26")
    f.save("fig3-mla-why.svg", yy + 6)


main()
