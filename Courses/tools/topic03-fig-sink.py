# -*- coding: utf-8 -*-
r"""专题三 · §6.2b「破案：崩掉的不是那几个老 token，是弃权票这个选项」

⭐⭐⭐ 2026-09-15 **从 `topic03-fig-swa-why.py` 里拆出来。**

  原来那张图是 **2164 px ＝ 2.5 屏**，四格连在一起：
  ① 凭什么敢砍　② 砍了为什么会崩　③ 得票最高那位什么都不做　④ 两条解法。

  ⭐ **拆点是图自己给的** ——&nbsp;它的图例本来就是三色：
  绿「敢砍的理由」／ 红「崩了」＝ **案情**；蓝「真正的原因」＝ **破案**。
  于是 ①② 留在原图，③④ 独立成本图。
  ⛔ 判据不是「太长了随便找个地方切」，是 **「这张图在讲几件事」** ——&nbsp;
  一张图只回答一个问句，切口就自己浮出来了。

⭐⭐⭐ 同一次重画：**原来的 ④ 是四个文字框，一张画都没有。**

  可它的标题偏偏叫「两条看起来同样彻底的解法，**只有一条成立**」——&nbsp;
  「看起来同样」这件事，**是眼睛的活，不是句子的活**。
  现在画成三个小场面：规矩（100 分必须投满）→ 两个**轮廓一模一样**、
  只差实心／空心的补丁 →&nbsp;两根按对数画的困惑度柱子。

⛔⛔ **画的时候刻意没画的东西，记在这里免得以后有人「补全」：**
  加了桶之后每一行的票**怎么分配**，我们**没有数据** ——&nbsp;
  论文给的是困惑度，不是注意力分布。
  ⭐ 所以图上只画「多了一个桶」这个**结构**，不画任何柱高。
  凭感觉画一组分配图，看着专业，但那是编的。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, GY, INK, GY2, LINE, LINE2)

W = 1400


def main():
    # StreamingLLM（Xiao 等 arXiv 2309.17453, ICLR 2024）表 3 / 表 10，
    # 三个 160M 从头预训练的对照。⛔ 两个数必须同组，不许跨表拼。
    PPL_LEARN, PPL_ZERO = 18.01, 29214.0
    assert PPL_ZERO / PPL_LEARN > 1000

    f = Fig(W, "attention sink 的成因：softmax 要求每一行的注意力加起来等于 1，"
               "不许弃权；于是没什么想看的那些行把票都投给了最前面几个 token，"
               "而那几个 token 的 value 几乎为零 —— 它们是废票桶。"
               "预训练时补一个可学的废票桶有效，补一个全零的桶反而崩得更狠")
    f.marks = set()
    y0 = f.header(
        "破案：崩掉的不是那几个老 token，是<tspan font-weight=\"700\">"
        "弃权票这个选项</tspan>",
        "⭐ <tspan font-weight=\"700\">softmax 不允许弃权 ——&#160;票必须投出去</tspan>"
        "，所以总得有人接住它",
        [(BL, "真正的原因")])

    # ══════════ ① 得票最高的那位，什么都不做 ══════════════════
    # ⭐⭐ 我们的选票比喻原来停在「票必须投满，于是全堆到最前面几个人身上」——
    #   ⛔ 缺了最后一步，而缺的这步才是机制本身：它投的那个人 value 几乎是零。
    # ⚠️ 「不许弃权的喧嚣民主」这个比喻不是本课原创 ——
    #   Evan Miller 2023-07《Attention Is Off By One》原话就是
    #   "a deafening democracy where abstention is disallowed"。主动引他。
    PH1 = 352
    py = f.panel(0, y0, W, PH1, "① 得票最高的那位，什么都不做", BL,
                 sub="⭐ 这一步才是机制本身 ——　上一张只说了「票投给了谁」")

    ay = py + 34
    BW, BH = 118, 132
    for k, (lab, hi, col, note) in enumerate((
            ("注意力权重", 1.0, RD, "冲天"),
            ("它的 value 模长", 0.08, GY2, "贴地"))):
        x = 130 + k * 220
        h = BH * hi
        f.box(x, ay + 20 + BH - h, BW, max(h, 4), col, "none", 4)
        f.t(x + BW / 2.0, ay + 178, lab, col, True, 17, "middle")
        f.t(x + BW / 2.0, ay + 202, note, GY2, size=16, anchor="middle")
    f.t(130, ay + 8, "第 0 号座位上那个 token：", INK, True, 19)
    f.t(130, ay + 234, "⭐ 把票投给他 ＝ 弃权", BL, True, 21)

    # 为什么偏偏是第 0 个：因果掩码下唯一人人都够得着的座位
    MX, N, C = 620, 7, 26
    f.t(MX, ay + 8, "为什么偏偏是最前面那几个？", INK, True, 17)
    for r in range(N):
        for c in range(N):
            on = c <= r
            f.box(MX + c * C, ay + 26 + r * C, C - 3, C - 3,
                  ("#e8f0fe" if c else "#1a73e8") if on else "#fff",
                  "none" if on else LINE2, 2)
    f.t(MX + C / 2.0, ay + 26 + N * C + 22, "↑", BL, True, 20, "middle")
    f.t(MX, ay + 26 + N * C + 48, "因果掩码下，第 0 列是<tspan "
        "font-weight=\"700\">唯一一列全满的</tspan>", BL, True, 17)
    f.t(MX, ay + 26 + N * C + 72, "——　不是它特殊，"
        "是<tspan font-weight=\"700\">只有它人人都够得着</tspan>", GY, size=17)

    f.box(1010, ay + 14, 334, 214, "#e8f0fe", BL, 10)
    f.t(1030, ay + 46, "⭐⭐ 于是整件事说得通了", BL, True, 20)
    for i, ln in enumerate([
            "softmax 不许弃权，",
            "模型就自己造了一个",
            "**弃权用的候选人**出来：",
            "永远在场、什么主张都没有。",
            "",
            "⛔ 砍掉他不是砍掉一个老 token，",
            "是砍掉了**弃权票这个选项**。"]):
        if ln:
            f.t(1030, ay + 78 + i * 24, ln.replace("**", ""),
                GY, "**" in ln, 17, w=300)

    # ══════════ ② 两条看起来同样彻底的解法 ══════════════════════
    # ⭐⭐⭐ 这一格原来是四个文字框。现在三个小场面，其中 Ⓑ Ⓒ 的轮廓
    #   **刻意画成一模一样**，只差实心／空心 ——「看起来同样彻底」这句话
    #   必须由画面说，说出来就不值钱了。
    y1 = y0 + PH1 + 18
    PH2 = 424
    py2 = f.panel(0, y1, W, PH2, "② 两条看起来同样彻底的解法，只有一条成立",
                  BL, sub="softmax 要求每一行的票必须投满")

    by = py2 + 26
    SEAT, SG = 44, 10          # 一个候选人一个座位

    def seats(x, y, n, extra=None, tint="#dadce0"):
        """一行候选人。extra ＝ ("实心"|"空心", 标签) 时多画一个补上去的桶。"""
        for i in range(n):
            f.box(x + i * (SEAT + SG), y, SEAT, SEAT, "#fff", LINE2, 6)
        if extra is None:
            return x + n * (SEAT + SG)
        kind, lab = extra
        ex = x + n * (SEAT + SG) + 12
        f.box(ex, y, SEAT, SEAT, tint if kind == "实心" else "#fff",
              tint, 6, sw=2.4, dash=None if kind == "实心" else "4 3")
        f.t(ex + SEAT / 2.0, y + SEAT + 22, lab, GY, True, 15, "middle")
        return ex + SEAT

    # Ⓐ 规矩：100 分必须投满，而且只能投给在场的人 ——&#160;于是全涌向第 0 号
    # ⭐ 那根粗箭头不是想象出来的：第 ① 格量的就是这件事（权重冲天、value 贴地）。
    f.box(30, by, 404, 318, "#fff", LINE, 10)
    f.t(52, by + 34, "Ⓐ 规矩", BL, True, 19)
    f.t(52, by + 58, "每一行的票必须正好投满 100 分", GY, size=15, w=364)
    f.t(52, by + 78, "哪怕这一行「没什么特别想看的」", GY2, size=15, w=364)
    BARW = 356
    f.box(52, by + 96, BARW, 28, "#e8f0fe", BL, 6)           # 满格的 100 分
    f.t(52 + BARW / 2.0, by + 116, "100 分　一分不许剩", BL, True, 16, "middle")
    SY = by + 178
    seats(52, SY, 6)
    f.box(52, SY, SEAT, SEAT, "#fce8e6", RD, 6, sw=2.2)      # 第 0 号：废票桶
    for i in range(6):                                       # 票往哪儿涌
        cx = 52 + i * (SEAT + SG) + SEAT / 2.0
        f.line(52 + BARW / 2.0, by + 128, cx, SY - 6,
               RD if i == 0 else GY2, 4.5 if i == 0 else 1.0)
    f.t(52 + SEAT / 2.0, SY + SEAT + 22, "第 0 号", RD, True, 15, "middle")
    f.t(160, SY + SEAT + 22, "这一排就是前面的每一个 token", GY2, size=14)
    f.t(52, SY + SEAT + 54, "⛔ 没有「弃权」这一栏 ——", RD, True, 16)
    f.t(52, SY + SEAT + 78, "票只能投给在场的候选人", RD, True, 16)

    f.path([(452, by + 150), (492, by + 150)], BL, 2.2)

    # Ⓑ / Ⓒ：两个补丁，轮廓一模一样
    for k, (tag, kind, lab, ppl, col, verdict) in enumerate((
            ("Ⓑ 补一个<tspan font-weight=\"700\">可学的</tspan>桶",
             "实心", "弃权", PPL_LEARN, GR, "⭐ 成立"),
            ("Ⓒ 补一个<tspan font-weight=\"700\">全零的</tspan>桶",
             "空心", "弃权", PPL_ZERO, RD, "⛔ 崩得更狠"))):
        px = 510 + k * 440
        f.box(px, by, 420, 318, "#fff", col, 10)
        f.box(px, by, 420, 5, col, col, 3)
        f.t(px + 22, by + 40, tag, col, True, 19)
        f.t(px + 22, by + 68,
            "桶里有自己的一份 K / V，预训练时跟着一起学"
            if kind == "实心" else "桶是恒定的零（＝softmax-off-by-one）",
            GY2, size=15, w=380)
        seats(px + 22, by + 100, 6, (kind, lab))
        f.t(px + 22, by + 180, "↑ 多出来的那一格，就是弃权栏", GY2, size=15)
        f.t(px + 22, by + 206, verdict, col, True, 20)
        f.t(px + 168, by + 206,
            "困惑度 <tspan font-weight=\"700\">%s</tspan>"
            % (("%.2f" % ppl) if ppl < 100 else format(int(ppl), ",")),
            col, size=19)
        # 按对数画的一根小柱 ＋ 一条真刻度 ——&#160;不给刻度的对数柱是读不出来的
        import math
        AXW = 376
        lg = lambda v: AXW * math.log(v / 10.0) / math.log(1e5 / 10.0)
        f.box(px + 22, by + 258, max(lg(ppl), 6), 22, col, "none", 4)
        for tv, tl in ((10, "10"), (100, "100"), (1000, "1千"),
                       (10000, "1万"), (100000, "10万")):
            f.line(px + 22 + lg(tv), by + 282, px + 22 + lg(tv), by + 288,
                   GY2, 1.0, arrow=False)
            f.t(px + 22 + lg(tv), by + 302, tl, GY2, size=13, anchor="middle")
        if k == 0:
            f.t(px + 22, by + 232, "↓ 困惑度，对数刻度（越短越好）", GY2, size=14)

    f._pan = None          # 这一行在面板外面，合法
    f.t(30, y1 + PH2 + 30,
        '⭐⭐ <tspan font-weight="700">Ⓑ 和 Ⓒ 的轮廓是一模一样的</tspan>'
        '——&#160;都是「给它一个可以弃权的地方」。'
        '<tspan font-weight="700">差别只在那一格是不是可学的</tspan>，'
        '而结果差了三个数量级。'
        '　⚠️ 加桶之后每一行具体怎么分票，论文给的是困惑度不是注意力分布，'
        '<tspan font-weight="700">所以 Ⓑ Ⓒ 两格没画任何投票柱高</tspan>。', GY, size=15)

    # ══════════ 落点 ════════════════════════════════════════════
    yy = y1 + PH2 + 54
    yy = f.band(yy, "info", "⭐⭐ 这个故事真正的教益 ——　比 sink 本身值钱", [
        "attention sink 不是 bug，也不是谁设计的特性，"
        "它是<tspan font-weight=\"700\">「票必须投满」这条规矩逼出来的副产品</tspan>。",
        "⭐ 判据：<tspan font-weight=\"700\">看到模型里一个「毫无道理却极其稳定」的现象，"
        "先去找是不是某个守恒 / 归一化约束逼出来的。</tspan>"
        "量化里那批总也压不下去的 outlier，跟这是同一件事（见专题八）。",
        "⛔ 还有一条：<tspan font-weight=\"700\">这个 bug 从公式上完全看不出来</tspan> ——&#160;"
        "是把注意力矩阵<tspan font-weight=\"700\">画出来</tspan>才发现的。"
        "这一讲所有的图，都是这个道理。",
    ])

    yy = f.src(yy + 24,
               "① 「value 模长极小」出自 Barbero 等 arXiv 2504.02732 图 4；"
               "「第 0 列是因果掩码下唯一全满的一列」是由掩码定义直接得出的",
               "② 18.01 与 29,214 出自 StreamingLLM（Xiao 等 arXiv 2309.17453, "
               "ICLR 2024）表 3 / 表 10 的三个 160M 从头预训练对照 ——&#160;"
               "⛔ 两个数必须取自同一组，不许跨表拼",
               "⚠️ 加了桶之后每一行的票怎么分配，论文给的是困惑度不是注意力分布 ——&#160;"
               "所以 Ⓑ Ⓒ 两格只画「多了一格」这个结构，没有画任何柱高",
               "⛔ 「不许弃权的选票」<tspan font-weight=\"700\">不是本课原创</tspan>"
               " ——&#160;Evan Miller 2023-07《Attention Is Off By One》原话就是 "
               "「a deafening democracy where abstention is disallowed」")
    f.save("fig3-sink.svg", yy + 6)


main()
