# -*- coding: utf-8 -*-
r"""专题三 · §1.1b「把盒子打开：『搅一搅』到底是什么，以及它凭什么不把旧的搅烂」

⭐⭐⭐ 2026-09-16 新画。现场三连问：
  「RNN 到底通过什么原理把一个字搅进去、再记回去？」
  「它为什么搅一下就能增加信息量，而不是把原有信息搞得乱七八糟？」
  「KV 是不是从这个时候就发明的？这个很重要。」

⛔ 三问在本课里**一个都没有答案**。第一章通篇在用「搅一搅」这个说法，
  **却从来没打开过那个盒子** ——&nbsp;于是它听起来真的像「搅匀」，
  而搅匀是会把东西毁掉的。台下的疑问完全合理。

⭐⭐ 这张图存在的理由，是 Ⓑ 那一格：
  **「叠加为什么不毁掉信息」不是个说法，是个可以算的数。**
  d 维里两个随机方向的夹角，2 维只有约 50°，1024 维是 **88.6°**。
  ——&nbsp;**叠加能还原，是维度买来的。**

⭐⭐⭐ 而这一格同时把全课缝成了一条线：
  第七章「d 维最多 d 个互相正交的方向」用的是**同一条道理**。
  **第一章那个盒子和第七章那块板子，连「为什么不糊」的理由都一样。**

⛔⛔ **刻意没画的东西：**
  ① 不写 h_t = tanh(W h + U x + b) 这个式子。本节面向的是「还没见过它」的人，
     **公式会把该看见的画面挡住**。图上只画**动作**：转一下 → 叠上去 → 压一压。
  ② Ⓒ 不画具体衰减曲线的数值 ——&nbsp;衰减快慢取决于 W 的谱，
     **没有一个「通用的曲线」**，画了就是编。只画「反复相乘」这个结构。

📌 Ⓑ 的两个数是**现场算的**（各 4000 次随机采样）：
  d=2 → E|cosθ| ≈ 0.63（≈ 2/π，夹角约 50°）；
  d=1024 → E|cosθ| ≈ 0.025（夹角约 88.6°）。
  大 d 的渐近式是 sqrt(2 / (π·d))，实测与它在 d≥64 时吻合到三位小数。
"""
import math
from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE, LINE2)

W = 1400

# ⭐ 这两个数是算出来的，不是查来的。改图先改这里，别在文字里手写。
COS_2D, COS_1024 = 0.6333, 0.0248
ANG_2D = math.degrees(math.acos(COS_2D))
ANG_1024 = math.degrees(math.acos(COS_1024))
assert 49 < ANG_2D < 53 and 88 < ANG_1024 < 89, (ANG_2D, ANG_1024)


def main():
    f = Fig(W, "把 RNN 的盒子打开：一步之内是「老状态转个向、新字映到同一个空间、"
               "两个相加、再压一压」，不是搅匀。叠加之所以不毁掉旧信息，"
               "是因为高维空间里随便两个方向几乎垂直 —— 1024 维里夹角实测 88.6 度。"
               "而每一步都要再乘一次，正是「传远了会淡」的根源。"
               "盒子里只有一个状态，没有钥匙和内容这两样东西，KV 要等到 2017 年")

    y0 = f.header(
        "把盒子打开：<tspan font-weight=\"700\">「搅一搅」到底是什么</tspan>，"
        "以及它凭什么<tspan font-weight=\"700\">不把旧的搅烂</tspan>",
        "⭐ 「搅」这个字容易让人想到搅匀 ——　"
        "<tspan font-weight=\"700\">而搅匀是会把东西毁掉的。它其实不是搅匀。</tspan>",
        [(BL, "一步之内"), (GR, "为什么不糊"), (OR, "为什么会淡")])

    # ══════════ Ⓐ 一步之内的四个动作 ═══════════════════════════════
    PH = 452
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 一步之内，其实只有四个动作 ——　"
                 "<tspan font-weight=\"700\">转一下 · 映过来 · 叠上去 · 压一压</tspan>",
                 BL,
                 sub="⛔ 没有任何一步在「搅匀」　·　"
                     "<tspan font-weight=\"700\">每一步都是可以倒着想回去的</tspan>")

    ROW = py + 92
    def vec(x, y, h_, col, fill, lab, sub=None):
        f.box(x, y, 52, h_, fill, col, 5, 1.6)
        for i in range(1, 4):
            f.line(x, y + h_ * i / 4.0, x + 52, y + h_ * i / 4.0, col, 0.8,
                   arrow=False)
        f.t(x + 26, y - 12, lab, col, True, 15, "middle")
        if sub:
            f.t(x + 26, y + h_ + 22, sub, GY, size=12.5, anchor="middle")

    def opbox(x, y, w_, lab, sub, col):
        f.box(x, y, w_, 62, "#fff", col, 8, 1.6)
        f.t(x + w_ / 2.0, y + 26, lab, col, True, 16, "middle")
        f.t(x + w_ / 2.0, y + 48, sub, GY, size=12.5, anchor="middle")

    vec(66, ROW, 104, BL, "#e8f0fe", "上一步的盒子", "一条 d 维的向量")
    f.line(126, ROW + 52, 176, ROW + 52, GY2, 1.6)
    opbox(176, ROW + 21, 150, "① 转一下", "乘一个学出来的矩阵", BL)
    f.line(332, ROW + 52, 386, ROW + 52, GY2, 1.6)

    vec(66, ROW + 186, 68, PU, "#f3e8fd", "这一个新字", "也是一条向量")
    f.line(126, ROW + 220, 176, ROW + 220, GY2, 1.6)
    opbox(176, ROW + 189, 150, "② 映过来", "乘另一个矩阵，进同一个空间", PU)
    f.line(332, ROW + 220, 386, ROW + 220, GY2, 1.6)

    f.box(386, ROW + 92, 78, 78, "#fff", INK, 39, 2.0)
    f.t(425, ROW + 142, "＋", INK, True, 40, "middle")
    f.t(425, ROW + 196, "③ 叠上去", INK, True, 16, "middle")
    f.t(425, ROW + 218, "就是普通的向量相加", GY, size=12.5, anchor="middle")
    f.line(425, ROW + 52, 425, ROW + 88, GY2, 1.6)
    f.line(425, ROW + 220, 425, ROW + 174, GY2, 1.6)

    f.line(468, ROW + 131, 528, ROW + 131, GY2, 1.6)
    opbox(528, ROW + 100, 150, "④ 压一压", "过一次非线性，收回到一个范围里", OR)
    f.line(684, ROW + 131, 740, ROW + 131, GY2, 1.6)
    vec(740, ROW + 79, 104, BL, "#e8f0fe", "新的盒子", "还是一条 d 维向量")

    # 回线
    f.path("M 766 %d L 766 %d L 92 %d L 92 %d" %
           (ROW + 79 + 104 + 44, ROW + 300, ROW + 300, ROW + 104 + 30),
           BL, 1.6, "5 4")
    f.t(430, ROW + 322, "⭐ 写回去 ——　下一个字来的时候，它就是「上一步的盒子」",
        BL, True, 15, "middle")

    f.box(880, ROW - 24, 460, 250, "#f8f9fa", LINE, 8)
    f.t(900, ROW + 4, "⭐ 这四步里，最要紧的是 ③", INK, True, 17)
    for i, ln in enumerate((
            "「搅」这个字是我们为了好懂用的比喻，",
            "可它容易让人想成<tspan font-weight=\"700\">搅匀</tspan> ——　那是会毁东西的。",
            "",
            "真实动作是<tspan font-weight=\"700\">相加</tspan>：",
            "新的<tspan font-weight=\"700\">叠</tspan>在老的上面，老的<tspan font-weight=\"700\">一点没被抹掉</tspan>。",
            "",
            "⛔ 那紧接着的问题就是 ——　",
            "叠了一万次之后，还分得开吗？",
    )):
        if ln:
            f.t(900, ROW + 38 + i * 26, ln, GY, size=14.5)
    f._pan = None

    yy = f.band(py + PH + 22, "warn", "先把那个词换掉", [
        "⛔ <tspan font-weight=\"700\">它不是「搅匀」，是「叠上去」。</tspan>"
        "搅匀之后你取不回任何一样东西；"
        "而<tspan font-weight=\"700\">叠加之后能不能取回，是个可以算的问题</tspan> ——&#160;"
        "下一格就算给你看。",
    ])

    # ══════════ Ⓑ 为什么叠上去不糊 ════════════════════════════════
    PH2 = 400
    py2 = f.panel(0, yy + 26, W, PH2,
                  "Ⓑ 叠了一万次为什么还分得开 ——　"
                  "<tspan font-weight=\"700\">因为维度一高，随便两个方向几乎总是垂直的</tspan>",
                  GR,
                  sub="⭐ 这不是个说法，是个<tspan font-weight=\"700\">能算出来的数</tspan>")

    def dial(cx, cy, ang_deg, col, title, sub1, sub2, warn=None):
        R = 92
        f.box(cx - R - 26, cy - R - 42, 2 * R + 52, 2 * R + 132, "#fff", LINE, 8)
        f.t(cx, cy - R - 16, title, col, True, 18, "middle")
        a = math.radians(ang_deg)
        f.line(cx, cy + 60, cx + R, cy + 60, col, 2.2)
        f.line(cx, cy + 60, cx + R * math.cos(a), cy + 60 - R * math.sin(a),
               col, 2.2)
        f.t(cx + 34, cy + 40, "%.0f°" % ang_deg, col, True, 17)
        f.t(cx, cy + 96, sub1, INK, size=14.5, anchor="middle")
        f.t(cx, cy + 118, sub2, GY, size=13, anchor="middle")
        if warn:
            f.t(cx, cy + 142, warn, RD, True, 14, "middle")

    dial(300, py2 + 130, ANG_2D, RD, "在 2 维里",
         "两个方向的夹角　典型只有 %.0f°" % ANG_2D,
         "叠上去就糊成一团 ——　互相严重干扰", "⛔ 取不回来")
    dial(760, py2 + 130, ANG_1024, GR, "在 1024 维里",
         "两个方向的夹角　典型是 %.1f°" % ANG_1024,
         "几乎垂直 ——　叠上去互不打扰", "⭐ 各取各的")

    f.box(1010, py2 + 26, 340, 286, "#e6f4ea", GR, 8)
    f.t(1030, py2 + 56, "⭐⭐ 所以那句话是：", GR, True, 17)
    for i, ln in enumerate((
            "<tspan font-weight=\"700\">叠加之所以能还原，</tspan>",
            "<tspan font-weight=\"700\">是维度买来的。</tspan>",
            "",
            "d 维里两个随机方向的余弦，",
            "典型大小约 1 除以根号 d ——",
            "<tspan font-weight=\"700\">d 越大，越垂直。</tspan>",
            "",
            "⛔ 但它<tspan font-weight=\"700\">不是无限的</tspan>：",
            "<tspan font-weight=\"700\">真正互不打扰的方向，最多 d 个。</tspan>",
            "叠过了头，照样糊。",
    )):
        if ln:
            f.t(1030, py2 + 90 + i * 24, ln, GY, size=14)
    f._pan = None

    yy = f.band(py2 + PH2 + 22, "ok", "这一格把第一章和后面缝上了", [
        "⭐⭐ <tspan font-weight=\"700\">「最多 d 个互不打扰的方向」这句话，"
        "后面还会再出现一次</tspan> ——&#160;"
        "到那时它解释的是<tspan font-weight=\"700\">另一块板子为什么会写满</tspan>。",
        "⭐ <tspan font-weight=\"700\">1990 年那个盒子，和今天那块板子，"
        "连「为什么不糊」的理由都是同一个。</tspan>",
    ])

    # ══════════ Ⓒ 那为什么还是会淡 ════════════════════════════════
    PH3 = 300
    py3 = f.panel(0, yy + 26, W, PH3,
                  "Ⓒ 那为什么还是会「传远了会淡」——　"
                  "<tspan font-weight=\"700\">因为它每走一步，都要再乘一次</tspan>", OR)

    bx, by = 86, py3 + 60
    for i in range(6):
        x = bx + i * 196
        f.box(x, by, 96, 60, "#fff", OR if i else BL, 6, 1.6)
        f.t(x + 48, by + 36, "盒子", OR if i else BL, True, 15, "middle")
        f.t(x + 48, by - 12, "第 %d 步" % (i + 1), GY, size=12.5, anchor="middle")
        if i < 5:
            f.line(x + 96, by + 30, x + 196, by + 30, GY2, 1.6)
            f.t(x + 146, by + 16, "× 转一下", GY, size=12.5, anchor="middle")
    f.t(86, by + 108, "⭐ 一条消息从第 1 步传到第 100 步，"
        "就被<tspan font-weight=\"700\">连乘了 99 次、也被压了 99 次</tspan>。",
        INK, size=15.5)
    f.t(86, by + 136,
        "⛔ 那个矩阵的「劲道」略小于 1 ——　连乘一百次就<tspan font-weight=\"700\">"
        "指数级地趋近于零</tspan>；略大于 1 ——　就<tspan font-weight=\"700\">"
        "指数级地炸掉</tspan>。", GY, size=15)
    f.t(86, by + 164, "⭐⭐ 这就是「传远了会淡」的全部机制 ——　"
        "<tspan font-weight=\"700\">它不是被时间冲淡的，是被反复相乘压没的。</tspan>",
        OR, True, 15.5)
    f._pan = None

    yy = f.band(py3 + PH3 + 22, "info", "这个毛病，三十年后被做成了功能", [
        "⭐⭐⭐ <tspan font-weight=\"700\">「每一步都乘一个小于 1 的数」——&#160;"
        "在 1990 年是这条路的致命伤；今天这一支<tspan text-decoration=\"underline\">主动</tspan>把它放了回去。</tspan>",
        "⭐ 区别只有一处：<tspan font-weight=\"700\">当年那个衰减是不受控的副作用，"
        "今天那个是学出来的、而且每个通道各有一个</tspan>。"
        "<tspan font-weight=\"700\">同一条性质，一次是病，一次是药。</tspan>",
    ])

    # ══════════ Ⓓ 盒子里有什么，没有什么 ═════════════════════════
    PH4 = 320
    py4 = f.panel(0, yy + 26, W, PH4,
                  "Ⓓ 那 KV 是不是这时候发明的？——　"
                  "<tspan font-weight=\"700\">不是。盒子里根本没有那两样东西</tspan>",
                  BL,
                  sub="⛔ 这个区别不是名词之争，"
                      "<tspan font-weight=\"700\">它是这整本书的骨架</tspan>")

    f.box(70, py4 + 30, 600, 212, "#fff", BL, 8)
    f.box(70, py4 + 30, 600, 4, BL, BL, 2)
    f.t(370, py4 + 64, "1990　盒子里", BL, True, 19, "middle")
    f.box(310, py4 + 90, 120, 62, "#e8f0fe", BL, 6)
    f.t(370, py4 + 128, "一个状态", BL, True, 17, "middle")
    f.t(370, py4 + 178, "就这一条向量。<tspan font-weight=\"700\">没有钥匙，也没有内容</tspan>。",
        GY, size=15, anchor="middle")
    f.t(370, py4 + 206, "它是<tspan font-weight=\"700\">全部历史压成的一团</tspan>",
        BL, True, 15.5, "middle")

    f.box(730, py4 + 30, 600, 212, "#fff", PU, 8)
    f.box(730, py4 + 30, 600, 4, PU, PU, 2)
    f.t(1030, py4 + 64, "2017　注意力带来的", PU, True, 19, "middle")
    for i in range(5):
        x = 812 + i * 88
        f.box(x, py4 + 90, 34, 62, "#f3e8fd", PU, 5)
        f.t(x + 17, py4 + 116, "钥", PU, True, 13, "middle")
        f.t(x + 17, py4 + 138, "匙", PU, True, 13, "middle")
        f.box(x + 38, py4 + 90, 34, 62, "#fff", PU, 5)
        f.t(x + 55, py4 + 127, "值", PU, True, 13, "middle")
    f.t(1030, py4 + 178, "每一个字都留一对，<tspan font-weight=\"700\">一个都不合并</tspan>。",
        GY, size=15, anchor="middle")
    f.t(1030, py4 + 206, "它是<tspan font-weight=\"700\">全部历史原样留着</tspan>",
        PU, True, 15.5, "middle")
    f._pan = None

    yy = f.band(py4 + PH4 + 22, "ok", "所以这两样东西的关系，正好是本书要讲的那件事", [
        "<tspan font-weight=\"700\">RNN 的状态 ＝ 把全部历史压成一团；"
        "KV cache ＝ 把全部历史原样留着。</tspan>"
        "⭐ 一个压到不能再压，一个一点都不压 ——&#160;"
        "<tspan font-weight=\"700\">它们是同一件事的两个极端。</tspan>",
        "⛔ 所以 <tspan font-weight=\"700\">KV 不是 1990 年发明的</tspan>，"
        "它要等到注意力出现、而且要等到<tspan font-weight=\"700\">自回归生成</tspan>"
        "把「每一步都要重算一遍历史」变成一笔真金白银的账，才会被「缓存」起来。"
        "<tspan font-weight=\"700\">这本书剩下的章节，讲的就是这两个极端之间的全部空间。</tspan>",
    ])

    yy = f.src(yy + 24,
               "Ⓑ 那两个角度是<tspan font-weight=\"700\">现场采样算的</tspan>"
               "（各 4000 次随机向量对）：d=2 时 E|cosθ| ≈ %.2f（夹角约 %.0f°，"
               "精确值是 2/π）；d=1024 时 E|cosθ| ≈ %.4f（夹角约 %.1f°）。"
               "大 d 的渐近式 sqrt(2/(π·d)) 与实测在 d≥64 时吻合到三位小数"
               % (COS_2D, ANG_2D, COS_1024, ANG_1024),
               "⛔ Ⓐ <tspan font-weight=\"700\">刻意不写出那个式子</tspan> ——&#160;"
               "本节面向还没见过它的人，公式会把该看见的画面挡住。"
               "图上只画动作：转一下 → 映过来 → 叠上去 → 压一压",
               "⛔ Ⓒ <tspan font-weight=\"700\">不画具体的衰减曲线</tspan> ——&#160;"
               "衰减快慢取决于那个矩阵的谱，<tspan font-weight=\"700\">没有一条"
               "「通用曲线」</tspan>，画出来就是编。只画「反复相乘」这个结构",
               "⚠️ Ⓓ 的「2017」指的是 <tspan font-weight=\"700\">Q/K/V 这套命名"
               "和 KV cache 这笔账</tspan>；"
               "「拿一个查询去跟一串历史比相似度再加权平均」这件事本身，"
               "2014 年的注意力就已经在做了")
    f.save("fig3-rnn-inside.svg", yy + 6)


main()
