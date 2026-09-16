# -*- coding: utf-8 -*-
r"""专题三 · §1.1「把盒子打开：『搅一搅』到底是什么，它凭什么不把旧的搅烂」

⭐⭐⭐ 2026-09-16 **第二版：整张砍掉一半多。** 现场原话：
  「1.1 和 1.1b 这两部分，非常简单的事情画了太多的图，它们彼此之间讲述的
    内容是严重重叠的。你把它们给我蒸馏一下，变成 1/3 我觉得差不多。」

⛔ 第一版是四格 ＋ 六条带子，**3186 px**。砍成两格 ＋ 两条带子。
  砍掉的三格，砍法各不相同，记在这里免得以后有人「好心」加回去：

  · **原 Ⓐb（两条回线）** ——&nbsp;内容没错也有价值（状态回线 vs 自回归回线，
    Transformer 只解开了第一条），但它讲的是**第一章的痛和第二章的解**，
    不是「盒子里是什么」。⭐ 压成落点带里的一句话，完整版留给 §1.2 / §二。
  · **原 Ⓒ（反复相乘 → 指数衰减）** ——&nbsp;它跟 Ⓑ 是**同一件事的两面**：
    都在说「每一步都要再乘一次那个矩阵」。Ⓑ 说这一乘为什么不毁信息，
    Ⓒ 说这一乘为什么会让远处变淡。⭐ **并进同一格反而更清楚**，
    因为读者一眼看见「好处和坏处是同一个动作带来的」。
  · **原 Ⓓ（1990 一团 vs 2017 钥匙/值）** ——&nbsp;两个大框画的是**一句话**。
    ⛔ 判据：**一句话就能说清的对照，不值一整格面板。** 压进落点带。

  被砍掉的三条带子（31 位词向量、「先把那个词换掉」、矩阵状态被重新发现）
  **内容没丢，挪进正文的折叠块了** ——&nbsp;它们是追问的答案，
  ⭐ 判据：**要人「读」的东西写成字比画成图便宜**，图只留要人「看」的。

⭐⭐ 这张图现在只剩两件必须用眼睛看的事：
  ① 一步之内那四个动作 ——&nbsp;**它不是搅匀，是叠上去**；
  ② 那两个角度 ——&nbsp;**叠加能还原是维度买来的，而且是个能算的数**。

⛔⛔ 刻意仍然不画：
  ① 不写 h_t = tanh(W h + U x + b)。本节面向「还没见过它」的人，
     公式会把该看见的画面挡住。
  ② 不画具体衰减曲线 ——&nbsp;快慢取决于那个矩阵的谱，没有通用曲线，画了就是编。

📌 Ⓑ 的两个数是**现场采样算的**（各 4000 次）：
  d=2 → E|cosθ| ≈ 0.63（≈ 2/π，夹角约 50°）；d=1024 → ≈ 0.025（约 88.6°）。
"""
import math
from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE, LINE2)

W = 1400

# ⭐ 这两个数是算出来的，不是查来的。改图先改这里，别在文字里手写。
COS_2D, COS_1024 = 0.6333, 0.0248
# ⭐ Elman 1990 句子模拟的三个维度（原文逐字）：输入 / 输出 31 个节点，
#   隐藏 / 上下文各 150 个。参数量全部从这里推，⛔ 别在文字里手写。
E_IN, E_H = 31, 150
P_IH, P_CH, P_HO = E_IN * E_H, E_H * E_H, E_H * E_IN
P_TOT = P_IH + P_CH + P_HO + E_H + E_IN
assert P_TOT == 31981 and abs(P_CH / P_TOT - 0.704) < 0.002
ANG_2D = math.degrees(math.acos(COS_2D))
ANG_1024 = math.degrees(math.acos(COS_1024))
assert 49 < ANG_2D < 53 and 88 < ANG_1024 < 89, (ANG_2D, ANG_1024)


def main():
    f = Fig(W, "把 RNN 的盒子打开：一步之内是「老状态转个向、新字映到同一个空间、"
               "两个相加、再压一压」，不是搅匀。叠加之所以不毁掉旧信息，"
               "是因为高维空间里随便两个方向几乎垂直 —— 1024 维里夹角实测 88.6 度；"
               "而同一个「每步再乘一次」，正是传远了会淡的根源。"
               "盒子里只有一个状态，没有钥匙和内容这两样东西")

    y0 = f.header(
        "把盒子打开：<tspan font-weight=\"700\">「搅一搅」到底是什么</tspan>，"
        "以及它凭什么<tspan font-weight=\"700\">不把旧的搅烂</tspan>",
        "⭐ 「搅」这个字容易让人想到搅匀 ——　"
        "<tspan font-weight=\"700\">而搅匀是会把东西毁掉的。它其实不是搅匀。</tspan>",
        [(BL, "一步之内"), (GR, "为什么不糊"), (OR, "为什么会淡")])

    # ══════════ Ⓐ 一步之内的四个动作 ＋ 三块权重 ═══════════════════
    PH = 470
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 一步之内，其实只有四个动作 ——　"
                 "<tspan font-weight=\"700\">转一下 · 映过来 · 叠上去 · 压一压</tspan>",
                 BL,
                 sub="⛔ 没有任何一步在「搅匀」")

    ROW = py + 86
    def vec(x, y, h_, col, fill, lab, sub=None):
        f.box(x, y, 46, h_, fill, col, 5, 1.6)
        for i in range(1, 4):
            f.line(x, y + h_ * i / 4.0, x + 46, y + h_ * i / 4.0, col, 0.8,
                   arrow=False)
        f.t(x + 23, y - 12, lab, col, True, 14, "middle")
        if sub:
            f.t(x + 23, y + h_ + 20, sub, GY, size=12, anchor="middle")

    def opbox(x, y, w_, lab, sub, col):
        f.box(x, y, w_, 58, "#fff", col, 8, 1.6)
        f.t(x + w_ / 2.0, y + 24, lab, col, True, 15.5, "middle")
        f.t(x + w_ / 2.0, y + 45, sub, GY, size=12, anchor="middle")

    vec(56, ROW, 92, BL, "#e8f0fe", "上一步的盒子", "一条 d 维向量")
    f.line(110, ROW + 46, 152, ROW + 46, GY2, 1.6)
    opbox(152, ROW + 17, 136, "① 转一下", "乘一个学出来的矩阵", BL)
    f.line(294, ROW + 46, 340, ROW + 46, GY2, 1.6)

    vec(56, ROW + 162, 60, PU, "#f3e8fd", "这一个新字", "也是一条向量")
    f.line(110, ROW + 192, 152, ROW + 192, GY2, 1.6)
    opbox(152, ROW + 163, 136, "② 映过来", "乘另一个矩阵，进同一空间", PU)
    f.line(294, ROW + 192, 340, ROW + 192, GY2, 1.6)

    f.box(340, ROW + 80, 70, 70, "#fff", INK, 35, 2.0)
    f.t(375, ROW + 124, "＋", INK, True, 36, "middle")
    f.t(375, ROW + 172, "③ 叠上去", INK, True, 15.5, "middle")
    f.t(375, ROW + 192, "就是普通的向量相加", GY, size=12, anchor="middle")
    f.line(375, ROW + 46, 375, ROW + 76, GY2, 1.6)
    f.line(375, ROW + 192, 375, ROW + 154, GY2, 1.6)

    f.line(414, ROW + 115, 462, ROW + 115, GY2, 1.6)
    opbox(462, ROW + 86, 136, "④ 压一压", "过一次非线性，收进一个范围", OR)
    f.line(604, ROW + 115, 648, ROW + 115, GY2, 1.6)
    vec(648, ROW + 69, 92, BL, "#e8f0fe", "新的盒子", "还是 d 维向量")

    f.path("M 671 %d L 671 %d L 79 %d L 79 %d" %
           (ROW + 69 + 92 + 40, ROW + 258, ROW + 258, ROW + 92 + 28),
           BL, 1.6, "5 4")
    f.t(375, ROW + 280, "⭐ 写回去 ——　下一个字来的时候，它就是「上一步的盒子」",
        BL, True, 14.5, "middle")

    # ⛔ 这句是全格的落点，别挪进带子里 —— 它必须紧挨着那个 ③ 的圆圈。
    f.box(56, ROW + 300, 620, 56, "#e8f0fe", BL, 6)
    f.t(76, ROW + 326, "⭐⭐ 四步里最要紧的是 ③："
                       "<tspan font-weight=\"700\">新的「叠」在老的上面，"
                       "老的一点没被抹掉</tspan>", INK, True, 16)
    f.t(76, ROW + 348, "——　搅匀之后你取不回任何一样东西；叠加之后能不能取回，"
                       "是个<tspan font-weight=\"700\">可以算的问题</tspan>（见 Ⓑ）",
        GY, size=13.5)

    # ⭐ 现场追问补：那几个矩阵到底多大、谁在学。数全部从 E_IN / E_H 推。
    f.box(716, ROW - 30, 628, 386, "#fff", BL, 8)
    f.t(740, ROW, "⭐ 可学的权重就三块 ——　真实尺寸长这样", BL, True, 17)
    f.t(740, ROW + 24, "（Elman 1990 句子模拟：输入 %d，隐藏 %d）" % (E_IN, E_H),
        GY2, size=12.5)
    for i, (nm, shp, n, note) in enumerate((
            ("② 映过来", "%d × %d" % (E_IN, E_H), P_IH, "学"),
            ("① 转一下", "%d × %d" % (E_H, E_H), P_CH, "学 · 最大"),
            ("读出来", "%d × %d" % (E_H, E_IN), P_HO, "学"),
            ("那条回线", "原样抄一份", 0, "⛔ 不训练"),
    )):
        y = ROW + 58 + i * 30
        f.t(740, y, nm, INK, True, 14.5)
        f.t(852, y, shp, GY, size=14, mono=True)
        if n:
            f.t(972, y, "%s 个数" % format(n, ","), GY, size=13.5)
        f.t(1084, y, note, GR if n else RD, True, 13)
    f.t(740, ROW + 200,
        "⭐⭐ 合起来 <tspan font-weight=\"700\">%s</tspan> 个参数"
        "　——　其中<tspan font-weight=\"700\">七成</tspan>在「转一下」那一块"
        % format(P_TOT, ","), BL, True, 15)
    f.t(740, ROW + 226,
        "⛔ 不是三万亿，不是三十亿 ——　<tspan font-weight=\"700\">三万。</tspan>",
        GY, size=14.5)
    f.t(740, ROW + 262, "⭐ 那条回线原文写死了：", INK, True, 14.5)
    f.t(740, ROW + 286, "「Recurrent connections are fixed at 1.0", GY,
        size=13, mono=True)
    f.t(740, ROW + 306, " and are not subject to adjustment」", GY,
        size=13, mono=True)
    f.t(740, ROW + 334,
        "——　它只把状态<tspan font-weight=\"700\">原样抄给下一步</tspan>；"
        "真正学的是", GY, size=13.5)
    f.t(740, ROW + 354,
        "<tspan font-weight=\"700\">「怎么把抄来的那份读回去」</tspan>，也就是最大那块",
        GY, size=13.5)
    f._pan = None

    # ══════════ Ⓑ 同一个动作的两面：不糊，与会淡 ═══════════════════
    PH2 = 424
    py2 = f.panel(0, py + PH + 22, W, PH2,
                  "Ⓑ <tspan font-weight=\"700\">好处和坏处是同一个动作带来的</tspan>"
                  " ——　每一步都要再乘一次那个矩阵", GR,
                  sub="⭐ 左边：这一乘<tspan font-weight=\"700\">为什么不毁信息</tspan>"
                      "　·　右边：同一乘<tspan font-weight=\"700\">为什么让远处变淡</tspan>")

    def dial(cx, cy, ang_deg, col, title, sub1, warn):
        R = 74
        f.box(cx - R - 22, cy - R - 38, 2 * R + 44, 2 * R + 108, "#fff", LINE, 8)
        f.t(cx, cy - R - 14, title, col, True, 17, "middle")
        a = math.radians(ang_deg)
        f.line(cx, cy + 48, cx + R, cy + 48, col, 2.2)
        f.line(cx, cy + 48, cx + R * math.cos(a), cy + 48 - R * math.sin(a),
               col, 2.2)
        f.t(cx + 28, cy + 30, "%.0f°" % ang_deg, col, True, 16)
        f.t(cx, cy + 80, sub1, INK, size=13.5, anchor="middle")
        f.t(cx, cy + 102, warn, RD if col is RD else GR, True, 13.5, "middle")

    dial(196, py2 + 116, ANG_2D, RD, "2 维里",
         "夹角典型只有 %.0f°" % ANG_2D, "⛔ 叠上去就糊，取不回来")
    dial(492, py2 + 116, ANG_1024, GR, "1024 维里",
         "典型是 %.1f° —— 几乎垂直" % ANG_1024, "⭐ 互不打扰，各取各的")

    # ⛔ 这三行的可用宽度只有 x=64..716（右边是那个橙框），渲染后抓到第一行
    #   伸到橙框底下去了。拆成短句，别指望「大概放得下」。
    f.t(64, py2 + 290, "⭐⭐ <tspan font-weight=\"700\">叠加之所以能还原，是维度买来的。</tspan>",
        GR, True, 15.5)
    f.t(64, py2 + 314, "d 维里两个随机方向的余弦，典型约 1 除以根号 d ——　"
                       "<tspan font-weight=\"700\">d 越大越垂直</tspan>。", GY, size=14)
    f.t(64, py2 + 340, "⛔ 但不是无限的："
                       "<tspan font-weight=\"700\">互不打扰的方向最多 d 个</tspan>，"
                       "叠过头照样糊。", GY, size=14)
    f.t(64, py2 + 366, "⭐ 记住这一句 ——　"
                       "<tspan font-weight=\"700\">后面另一块板子为什么会写满，"
                       "是同一条道理。</tspan>", GY, size=14)

    # ── 右半：同一乘，传远了就淡
    f.box(730, py2 + 26, 614, 300, "#fff", OR, 8)
    f.box(730, py2 + 26, 614, 4, OR, OR, 2)
    f.t(1037, py2 + 58, "同一个「乘一下」，传远了就淡", OR, True, 17, "middle")
    bx, by = 762, py2 + 84
    for i in range(4):
        x = bx + i * 142
        f.box(x, by, 76, 48, "#fff", OR if i else BL, 6, 1.6)
        f.t(x + 38, by + 30, "盒子", OR if i else BL, True, 14, "middle")
        f.t(x + 38, by - 10, "第 %d 步" % (i + 1), GY, size=12, anchor="middle")
        if i < 3:
            f.line(x + 76, by + 24, x + 142, by + 24, GY2, 1.6)
            f.t(x + 109, by + 12, "× 转一下", GY, size=11.5, anchor="middle")
    f.t(762, by + 92, "⭐ 从第 1 步传到第 100 步，"
                      "就被<tspan font-weight=\"700\">连乘了 99 次</tspan>。",
        INK, size=14.5)
    f.t(762, by + 118, "⛔ 劲道略小于 1 ——　<tspan font-weight=\"700\">指数趋零</tspan>；"
                       "略大于 1 ——　<tspan font-weight=\"700\">指数炸掉</tspan>。",
        GY, size=14)
    f.t(762, by + 148, "⭐⭐ <tspan font-weight=\"700\">它不是被时间冲淡的，"
                       "是被反复相乘压没的。</tspan>", OR, True, 15)
    f.t(762, by + 178, "⚠️ 而这个毛病，三十年后被<tspan font-weight=\"700\">主动"
                       "放回去当功能用</tspan>了 ——　见第七章。", GY, size=13.5)
    f._pan = None

    yy = f.band(py2 + PH2 + 22, "ok",
                "落点：这个盒子里有什么、没有什么 ——　它是这整本书的骨架", [
        "⛔ <tspan font-weight=\"700\">KV 不是这时候发明的。</tspan>"
        "盒子里<tspan font-weight=\"700\">只有一个状态</tspan>，"
        "没有「钥匙」和「内容」这两样东西 ——&#160;那套命名要等到 2017 年，"
        "而把它<tspan font-weight=\"700\">缓存</tspan>起来是更晚的事。",
        "⭐⭐⭐ <tspan font-weight=\"700\">RNN 的状态 ＝ 全部历史压成一团；"
        "KV cache ＝ 全部历史原样留着。</tspan>"
        "一个压到不能再压，一个一点都不压 ——&#160;"
        "<tspan font-weight=\"700\">它们是同一件事的两个极端，"
        "而这本书剩下的章节，讲的就是这两端之间的全部空间。</tspan>",
        "⚠️ 顺带一句、完整版在下一小节：把序列连起来的其实有<tspan font-weight=\"700\">两条</tspan>回线 ——&#160;"
        "<tspan font-weight=\"700\">状态那条 2017 年被解开了（于是训练能并行）；"
        "自回归那条到今天也没人解开</tspan>（于是吐字仍然是一条链）。",
    ])

    yy = f.src(yy + 24,
               "Ⓑ 那两个角度是<tspan font-weight=\"700\">现场采样算的</tspan>"
               "（各 4000 次随机向量对）：d=2 时 E|cosθ| ≈ %.2f（夹角约 %.0f°，"
               "精确值是 2/π）；d=1024 时 ≈ %.4f（约 %.1f°）。"
               "渐近式 sqrt(2/(π·d)) 与实测在 d≥64 时吻合到三位小数"
               % (COS_2D, ANG_2D, COS_1024, ANG_1024),
               "⛔ Ⓐ <tspan font-weight=\"700\">刻意不写出那个式子</tspan> ——&#160;"
               "本节面向还没见过它的人，公式会把该看见的画面挡住；"
               "⛔ Ⓑ 右半<tspan font-weight=\"700\">不画具体衰减曲线</tspan> ——&#160;"
               "快慢取决于那个矩阵的谱，没有通用曲线，画出来就是编",
               "Ⓐ 那张尺寸表与「回线不训练」出自 Elman 1990 《Finding Structure "
               "in Time》原文：「Recurrent connections are fixed at 1.0 and are "
               "not subject to adjustment」；句子模拟「hidden and context layers "
               "contained <tspan font-weight=\"700\">150 nodes</tspan> each」，"
               "输入输出 31 个节点。<tspan font-weight=\"700\">参数量由 31/150 推算，"
               "非原文给出</tspan>；同一篇里最小的 XOR 模拟只有 2 个隐藏单元",
               "⛔ <tspan font-weight=\"700\">精度那一栏本图故意空着</tspan> ——&#160;"
               "那个年代的论文不写精度，因为它当时还不是一个可以权衡的设计维度。"
               "本课不替它补一个数",
               "⚠️ 落点带里的「2017」指的是 <tspan font-weight=\"700\">Q/K/V 这套命名"
               "和 KV cache 这笔账</tspan>；「拿一个查询去跟一串历史比相似度再加权平均」"
               "这件事本身，2014 年的注意力就已经在做了")
    f.save("fig3-rnn-inside.svg", yy + 6)


main()
