# -*- coding: utf-8 -*-
r"""专题三 · §6.1b「滑窗凭什么敢砍，砍了为什么会崩」

⭐⭐⭐ 2026-09-14 **整张重画**，全部换成生活画面。

  ① **凭什么敢砍** ——&nbsp;画**传话**：每个人只跟身边 4 个人说话，
     但话可以一层一层往外传。**层数是免费的射程。**
     Mistral 7B：窗口 4096 × 32 层 →&nbsp;131,072（脚本当场乘出来断言）。

  ② **砍了为什么会崩** ——&nbsp;一个数字对比就够，画成**两根天差地别的柱子**：
     纯窗口 0+1024 →&nbsp;**5158.07**；把最前面四个 token 留下 →&nbsp;**5.40**。
     ⭐⭐ 判决性实验：把那四个换成**换行符**，5.60 ——&nbsp;几乎一样。
     **所以起作用的是位置，不是内容。**

  ③ **为什么会有这么个东西** ——&nbsp;画成**必须投满的选票**：
     softmax 要求每一行的票加起来正好 100 分，
     **哪怕这一行没什么想看的，票也必须投出去** ——&nbsp;
     于是大家把废票都投给了最前面那几个（自回归下只有它们人人都够得着）。
     ⭐ 两条看起来同样彻底的解法，**只有一条成立**。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, GY, PU, INK, GY2, LINE, LINE2,
                          BG2)

W = 1400


def main():
    WIN, LAY = 4096, 32
    span = WIN * LAY
    assert span == 131072
    PPL_WIN, PPL_SINK, PPL_NL = 5158.07, 5.40, 5.60
    assert PPL_WIN / PPL_SINK > 900

    f = Fig(W, "滑窗凭什么敢砍：每个人只跟身边几个说话，但话能一层层往外传，"
               "层数是免费的射程；砍了为什么会崩：把最前面四个 token 扔掉，"
               "困惑度从 5.40 炸到 5158；为什么：softmax 要求每行的票必须投满")
    f.marks = set()
    y0 = f.header(
        "滑窗：凭什么敢砍，砍了为什么会崩",
        "一个<tspan font-weight=\"700\">按时间顺序讲的侦探故事</tspan>",
        [(GR, "敢砍的理由"), (RD, "崩了"), (BL, "真正的原因")])

    # ══════════ ① 传话：层数是免费的射程 ════════════════════════
    PH = 250
    py = f.panel(0, y0, W, PH, "① 凭什么敢砍 ——　每层只看身边几个，但话能往外传",
                 GR, sub="层数是免费的射程")

    ay = py + 22
    N = 9          # ⛔ 原来 13 个，右端撞上那块 Mistral 结论框
    for L in range(3):
        yy = ay + 22 + L * 52
        f.t(56, yy + 16, "第 %d 层" % (L + 1), GY2, size=15)
        for i in range(N):
            x = 140 + i * 88
            on = i <= 2 + L * 3
            f.box(x, yy, 68, 32, "#e6f4ea" if on else BG2,
                  GR if on else LINE2, 5)
            f.t(x + 34, yy + 22, str(i + 1), GR if on else GY2,
                on, 16, "middle")
        if L < 2:
            f.line(140 + 4.5 * 88, yy + 36, 140 + 4.5 * 88, yy + 50,
                   GY2, 1.1)
    f.t(140, ay + 190, "一层只跨 3 格 → 两层 6 格 → 三层 9 格 ……",
        GY, size=18)
    f.box(1000, ay + 24, 360, 152, "#e6f4ea", GR, 10)
    f.t(1180, ay + 66, "Mistral 7B", GR, True, 21, "middle")
    f.t(1180, ay + 106, "%s × %d 层" % (format(WIN, ","), LAY), GR, True, 22,
        "middle")
    f.t(1180, ay + 146, "＝ %s" % format(span, ","), GR, True, 26, "middle")

    # ══════════ ② 崩了 ══════════════════════════════════════════
    y1 = y0 + PH + 18
    PH2 = 452
    py2 = f.panel(0, y1, W, PH2, "② 砍了为什么会崩 ——　扔掉最前面四个，就崩了",
                  RD, sub="Llama-2-13B，PG19")

    by = py2 + 24
    # ⛔⛔ 「柱子按对数画」这句原来在柱子**下面 130px** ——&nbsp;
    #   读者先看到「5158 只比 5.40 高两倍多」，得出「差得也不算多」，
    #   往下读才知道是对数轴，而那一眼的印象跟这张图要说的正好相反。
    # ⭐ 判据：**读图的钥匙必须在图之前。** 放在后面 ＝ 先让人看错一眼再纠正。
    f.t(120, by + 18, "⚠️ 先说怎么读：这三根柱子<tspan font-weight=\"700\">"
        "按对数画</tspan> ——　线性画的话后两根根本看不见。", RD, size=17, w=860)
    f.t(120, by + 42, "<tspan font-weight=\"700\">"
        "5158 和 5.40 差的是三个数量级，不是三倍。</tspan>", RD, size=17, w=860)
    BASE, HMAX = by + 250, 150   # ⭐ 上面多了两行读图提示，柱子整体下移
    import math
    for i, (lab, v, col, note) in enumerate([
        ("只留窗口\n0 + 1024", PPL_WIN, RD, "⛔ 崩了"),
        ("留最前面 4 个\n4 + 1020", PPL_SINK, GR, "✅ 好了"),
        ("那 4 个换成换行符\n4 + 1020", PPL_NL, BL, "⭐ 几乎一样"),
    ]):
        x = 120 + i * 300
        h = HMAX * math.log10(v) / math.log10(PPL_WIN)
        f.box(x, BASE - h, 150, h, col, "none", 5)
        f.t(x + 75, BASE - h - 14, "%.2f" % v, col, True, 26, "middle")
        for k, ln in enumerate(lab.split("\n")):
            f.t(x + 75, BASE + 26 + k * 24, ln, GY, size=16, anchor="middle")
        f.t(x + 75, BASE - h - 44, note, col, True, 18, "middle")
    f.t(120, BASE + 130, "困惑度（越低越好）", GY2, size=16)

    f.box(1000, by + 24, 360, 192, "#e8f0fe", BL, 10)
    f.t(1024, by + 66, "⭐⭐ 判决性的是第三根", BL, True, 21)
    f.t(1024, by + 104, "把那四个 token 换成", GY, size=17)
    f.t(1024, by + 134, "毫无意义的换行符", GY, size=17)
    f.t(1024, by + 172, "结果几乎一样", BL, True, 22)
    f.t(1024, by + 202, "→　起作用的是位置，不是内容", BL, True, 17)

    # ══════════ ③ 必须投满的选票 ════════════════════════════════
    y2 = y1 + PH2 + 18
    PH3 = 324
    py3 = f.panel(0, y2, W, PH3, "③ 那这几个 token 到底在干嘛 ——　它们是废票桶",
                  BL, sub="softmax 要求每一行的票必须投满")

    vy = py3 + 20
    f.box(56, vy + 26, 600, 150, "#e8f0fe", BL, 10)
    f.t(80, vy + 64, "规矩：每一行的票加起来必须正好 100 分", BL, True, 21)
    f.t(80, vy + 102, "——　哪怕这一行「没什么特别想看的」，", GY, size=18)
    f.t(80, vy + 134, "票<tspan font-weight=\"700\">也必须投出去</tspan>。", GY, size=18)
    f.t(80, vy + 166, "这就是 softmax 的归一化", GY2, size=15)

    f.path([(672, vy + 100), (716, vy + 100)], BL, 2.0)

    f.box(736, vy + 26, 624, 150, "#fff", BL, 10)
    f.t(760, vy + 64, "于是废票都投给了最前面那几个", BL, True, 21)
    f.t(760, vy + 102, "为什么偏偏是它们？——　因为自回归：", GY, size=18)
    f.t(760, vy + 134, "<tspan font-weight=\"700\">全场只有开头那几个，人人都够得着。</tspan>",
        GY, size=18)
    f.t(760, vy + 166, "把废票桶撤了，票没处投，整行就乱套", GY2, size=15)

    sy = vy + 196
    f.box(56, sy, 640, 86, "#e6f4ea", GR, 10)
    f.t(80, sy + 36, "⭐ 一个成立的解法", GR, True, 20)
    f.t(80, sy + 68, "预训练时加一个<tspan font-weight=\"700\">可学的</tspan>废票桶　"
        "→　1+1023 下 PPL 18.01", GY, size=17)

    f.box(720, sy, 640, 86, "#fce8e6", RD, 10)
    f.t(744, sy + 36, "⛔ 一个看起来对、但被论文自己证伪的", RD, True, 20)
    f.t(744, sy + 68, "给一个<tspan font-weight=\"700\">全零</tspan>的桶"
        "（＝softmax-off-by-one）　→　PPL 29214", GY, size=17)

    # ══════════ 落点 ════════════════════════════════════════════
    yy = y2 + PH3 + 20
    yy = f.band(yy, "info", "⭐⭐ 这个故事真正的教益 —— 比 sink 本身值钱", [
        "attention sink 不是 bug，也不是谁设计的特性，"
        "它是<tspan font-weight=\"700\">「票必须投满」这条规矩逼出来的副产品</tspan>。",
        "⭐ 判据：<tspan font-weight=\"700\">看到模型里一个「毫无道理却极其稳定」的现象，"
        "先去找是不是某个守恒 / 归一化约束逼出来的。</tspan>"
        "量化里那批总也压不下去的 outlier，跟这是同一件事（见专题八）。",
        "⛔ 还有一条：<tspan font-weight=\"700\">这个 bug 从公式上完全看不出来</tspan> ——&#160;"
        "是把注意力矩阵<tspan font-weight=\"700\">画出来</tspan>才发现的。"
        "这一讲所有的图，都是这个道理。",
    ])

    yy = f.band(yy + 14, "warn", "别把「理论射程」当「有效射程」", [
        "%s × %d ＝ <tspan font-weight=\"700\">%s</tspan> 是个上界，"
        "说的是「信息最远能传到这儿」，<tspan font-weight=\"700\">不是「这么远还能用」"
        "</tspan> ——&#160;每跨一层只挪一格窗口，而且一路被后面的信息稀释。"
        % (format(WIN, ","), LAY, format(span, ",")),
        "⭐ 稳妥说法：<tspan font-weight=\"700\">滑窗把「远处」从「看不见」"
        "变成了「看得见但很模糊」</tspan> ——&#160;"
        "所以后面那些方案才要在滑窗之外再加一条「挑着看」的路。",
    ])

    yy = f.src(yy + 24,
               "① 出自 Mistral 7B arXiv 2310.06825 §2（k×W 射程、W=4096 / 32 层、"
               "rolling buffer cache）；131,072 由脚本当场乘出来并断言",
               "②③ 出自 StreamingLLM（Xiao 等 arXiv 2309.17453, ICLR 2024）"
               "表 1 / 表 2 与 §3.1 / §3.3：5158.07 → 5.40、换行符 5.60、"
               "留 1/2/4/8 个的对照",
               "⛔ Zero Sink（＝softmax-off-by-one）那组反例出自同文表 3 / 表 10 的"
               "三个 160M 预训练对照",
               "⚠️ 表 1（PG19 第一本书，65K）与表 2（拼接后 400K）"
               "<tspan font-weight=\"700\">不是同一个评测集</tspan>；"
               "⚠️ 「传话 / 废票桶」是本课的比喻")
    f.save("fig3-swa-why.svg", yy + 6)


main()
