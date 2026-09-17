# -*- coding: utf-8 -*-
r"""专题四 · §3「梯度下降为什么可以」——&#160;一切的本源

⭐⭐⭐ 2026-09-17 新画。现场原话：「梯度下降法这个是一切的本源，
  你一定要想办法把这个本源给讲清楚 ……让大家理解**为什么**梯度下降法可以」。
  ⛔ 而这一讲原来**跳过了它**：§1 讲梯度怎么算出来，§3 直接讲怎么把梯度
    折算成更新 ——&#160;中间「为什么顺着梯度走就能变好」那一步没人说。

⭐⭐ 取法来自 **3Blue1Brown**《Gradient descent, how neural networks learn》。
  四个讲法都从官方讲义原文核过（⛔ 图是我们自己重画的）：
    · 「球滚下山」——&#160;原文 "The image to have in mind is a ball rolling down a hill"
    · **步长 ∝ 斜率 → 自动刹车** ——&#160;原文 "if you make your step sizes
      proportional to the slope itself … that keeps you from overshooting"
    · 落在哪个谷**取决于起点**，不保证最低
    · 高维时换一个**非空间**的读法：那一列数里，正负说往哪推，
      **相对大小说哪一项更要紧**（原文 "the relative magnitudes … tells us
      which of those changes matters more"）

⭐⭐⭐ 而这张图在**本讲**里的位置，是把两头接上：
  · 往回接 §1 ——&#160;Ⓒ 那一整列数，**就是反向传播一遍算出来的那份**。
  · 往前接 fig-beststep ——&#160;3B1B 说「步长 ∝ 斜率」，
    李宏毅说「最优步长 ＝ 斜率 ÷ 曲率」。
    **朴素梯度下降只用了分子；Adam 分母上那一坨，是在补分母。**
  ⭐ 这个合题两边原文都没有，是这一讲自己的落点。
"""
import math

from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE)

W = 1400


def main():
    f = Fig(W, "梯度下降为什么可以。一维上像一个球滚下山：斜率为正就往左挪，"
               "为负就往右挪；而且把步长取成跟斜率成正比，越接近谷底步子越小，"
               "自动刹车不会冲过头。两个输入时斜率不再是一个数，"
               "要用负梯度这个向量表示下降最快的方向。"
               "到了三千亿维就别再想画面了，"
               "那一列数每一项的正负说往哪推、相对大小说哪一项更要紧。"
               "而那一整列，正是反向传播一遍算出来的那份")

    y0 = f.header(
        "梯度下降为什么可以　——　<tspan font-weight=\"700\">"
        "一切的本源，先把它讲透</tspan>",
        "⛔ 前面讲了梯度<tspan font-weight=\"700\">怎么算出来</tspan>，"
        "这一格讲<tspan font-weight=\"700\">为什么顺着它走就能变好</tspan>",
        [(BL, "一维：一条线"), (GR, "二维：一片山地"), (PU, "三千亿维：不画了")])

    # ══════════ Ⓐ 一维：球滚下山 ═══════════════════════════════════
    PH = 412
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 先看一维　——　"
                 "<tspan font-weight=\"700\">脑子里放一个球，从山上滚下来</tspan>", BL,
                 sub="⭐ 规则只有一句："
                     "<tspan font-weight=\"700\">斜率为正就往左挪，为负就往右挪</tspan>")

    OX, OY, AW, AH = 120, py + 300, 780, 190

    # ⭐⭐⭐ 曲线和球的位置**不是手摆的，是真跑一遍梯度下降算出来的**。
    #   ⛔ 第一版我手挑了五个 t 值假装是迭代轨迹 ——&#160;
    #     而这张图要讲的恰恰是「步子会自动变小」。
    #     手挑等于**把结论画上去**，而不是**让它自己长出来**。
    #   ⭐ 判据：**一张图要证明某个动态行为，就让那个动态真的跑一遍。**
    #     跑出来的轨迹自带说服力，而且跑不出来就说明讲法本身有问题。
    def loss(t):
        # 系数是**挑出来的**，判据两条，都由下面的 assert 盯着：
        #   ① 区间里正好两个谷　② 两个谷深浅明显不同（否则「不保证最低」没画面）
        return (math.sin(4.0 * math.pi * t) * 0.34
                + math.sin(1.0 * math.pi * t) * 0.26 + 0.5)

    def slope(t, h=1e-5):
        return (loss(t + h) - loss(t - h)) / (2 * h)

    def descend(t0, lr, n):
        """朴素梯度下降：每步挪 lr × 斜率。返回轨迹。"""
        out, t = [t0], t0
        for _ in range(n):
            t -= lr * slope(t)
            out.append(t)
        return out

    _VALLEYS = [t / 2000.0 for t in range(2, 1998)
                if slope(t / 2000.0) < 0 < slope((t + 1) / 2000.0)]
    assert len(_VALLEYS) == 2, "要正好两个谷，现在是 %d 个" % len(_VALLEYS)
    assert abs(loss(_VALLEYS[0]) - loss(_VALLEYS[1])) > 0.12, \
        "两个谷的深浅要拉开，否则「不保证落到最低的」这件事看不出来"

    TRAJ = descend(0.22, 0.020, 4)          # 左边出发
    TRAJ2 = descend(0.72, 0.020, 4)         # 右边出发
    for _tr in (TRAJ, TRAJ2):
        _st = [abs(_tr[i + 1] - _tr[i]) for i in range(len(_tr) - 1)]
        assert all(_st[i] > _st[i + 1] for i in range(len(_st) - 1)), \
            "步子必须一步比一步小 ——　这就是这张图要讲的那件事，跑不出来就别画"
    # ⭐⭐⭐ 这一条是这张图最值钱的一句：**先出发那个落进了较浅的谷**。
    #   它不是我写上去的结论，是这两条轨迹自己跑出来的。
    assert loss(TRAJ[-1]) > loss(TRAJ2[-1]), \
        "左边那条应当落进更浅的谷 ——　「不保证最低」全靠这一点"

    def cy(t):
        return OY - AH * loss(t)

    d, i = "M %.1f %.1f" % (OX, cy(0)), 1
    while i <= 240:
        d += " L %.1f %.1f" % (OX + AW * i / 240.0, cy(i / 240.0))
        i += 1
    f.path(d, BL, 2.2, arrow=False)
    f.line(OX - 20, OY + 18, OX + AW + 30, OY + 18, GY2, 1.1, arrow=False)
    f.t(OX + AW + 36, OY + 23, "某个参数 →", GY2, size=12)
    f.t(OX - 26, py + 80, "loss", GY2, size=12, anchor="end")

    def draw_run(tr, col, tag):
        for k, t in enumerate(tr):
            x, yv = OX + AW * t, cy(t)
            f.box(x - 7, yv - 14, 14, 14, col, col, 7)
            if k == 0:
                f.t(x, yv - 28, tag, col, True, 12.5, "middle")
        xe, ye = OX + AW * tr[-1], cy(tr[-1])
        f.line(xe, ye + 12, xe, OY + 12, col, 1, dash="3 3", arrow=False)

    draw_run(TRAJ, RD, "起点 A")
    draw_run(TRAJ2, GR, "起点 B")

    f.t(OX + AW * TRAJ[-1], OY + 34, "落在这儿", RD, True, 13, "middle")
    f.t(OX + AW * TRAJ2[-1], OY + 34, "落在这儿", GR, True, 13, "middle")
    f.t(OX + AW * 0.17, py + 350,
        "⭐ <tspan font-weight=\"700\">球一步比一步挪得少</tspan>", BL, True, 14, "middle")
    f.t(OX + AW * 0.17, py + 372,
        "没人让它慢下来 ——　是坡自己变平了", GY, size=12.5, anchor="middle")
    f.t(OX + AW * 0.66, py + 350,
        "⛔ <tspan font-weight=\"700\">A 落进了更浅的那个谷</tspan>", RD, True, 14, "middle")
    f.t(OX + AW * 0.66, py + 372,
        "只因为它从左边出发 ——　跟谁更优无关", GY, size=12.5, anchor="middle")

    f.box(960, py + 44, 400, 250, "#e8f0fe", BL, 8)
    f.t(1160, py + 78, "自动刹车", BL, True, 20, "middle")
    f.t(1160, py + 116, "<tspan font-weight=\"700\">步长 ∝ 斜率</tspan>", INK, True, 17, "middle")
    f.t(1160, py + 152, "越接近谷底，坡越平", GY, size=14, anchor="middle")
    f.t(1160, py + 176, "坡越平，步子越小", GY, size=14, anchor="middle")
    f.t(1160, py + 214, "⭐ 所以它<tspan font-weight=\"700\">不会在谷底</tspan>",
        BL, True, 15, "middle")
    f.t(1160, py + 238, "<tspan font-weight=\"700\">来回冲过头</tspan>",
        BL, True, 15, "middle")
    f.t(1160, py + 276, "——　这一条是免费送的，不用额外做什么",
        GY2, size=12, anchor="middle")
    f._pan = None

    # ══════════ Ⓑ 二维：方向不再是一个数 ═══════════════════════════
    PH2 = 268
    py2 = f.panel(0, py + PH + 20, W, PH2,
                  "Ⓑ 加到两个参数　——　"
                  "<tspan font-weight=\"700\">「斜率」这个词就不够用了</tspan>", GR,
                  sub="⭐ 一个数说不清方向，得用一个<tspan font-weight=\"700\">向量</tspan>")

    f.box(70, py2 + 36, 600, 176, "#f1f3f4", GY2, 8)
    f.t(370, py2 + 72, "一维：问「斜率是正是负」", GY, True, 16, "middle")
    f.t(370, py2 + 106, "一个数就够了", GY2, size=13.5, anchor="middle")
    f.t(370, py2 + 150, "二维：问<tspan font-weight=\"700\">「往哪个方向走，降得最快」</tspan>",
        INK, True, 16, "middle")
    f.t(370, py2 + 184, "一个数说不清 ——　它得是个方向", GY, size=13.5, anchor="middle")

    f.t(700, py2 + 120, "→", GY2, True, 22, "middle")

    f.box(730, py2 + 36, 600, 176, "#e6f4ea", GR, 8)
    f.t(1030, py2 + 72, "这个方向就叫<tspan font-weight=\"700\">梯度</tspan>", GR, True, 18, "middle")
    f.t(1030, py2 + 108, "梯度指的是<tspan font-weight=\"700\">上坡最快</tspan>的方向",
        INK, True, 15, "middle")
    f.t(1030, py2 + 138, "所以下坡就取它的<tspan font-weight=\"700\">相反数</tspan>",
        INK, True, 15, "middle")
    f.t(1030, py2 + 182, "⭐ 而它的<tspan font-weight=\"700\">长度</tspan>还顺带告诉你：这个坡有多陡",
        GR, size=13, anchor="middle")

    # ⚠️ 苏剑林那条「前提」——&#160;只留一句话指过去，**不在这儿再画一遍**：
    #   「F 范数 → SGD / 谱范数 → Muon」的圆与方，`fig-muon` Ⓒ 已经画了。
    #   ⭐ 判据：**同一个画面全书只画一次，别处只留指针。**
    f.t(700, py2 + 226,
        "⚠️ 顺带记一句，后面会兑现："
        "<tspan font-weight=\"700\">「最快」是相对于你怎么量「这一步迈了多大」说的</tspan>"
        "　——　换一把尺，最快的方向就跟着变（`fig-muon` Ⓒ）。",
        GY, size=13, anchor="middle")
    f._pan = None

    # ══════════ Ⓒ 三千亿维：把画面扔掉 ═════════════════════════════
    PH3 = 296
    py3 = f.panel(0, py2 + PH2 + 20, W, PH3,
                  "Ⓒ ⭐⭐⭐ 到了三千亿个参数　——　"
                  "<tspan font-weight=\"700\">别再想「山」了，换个读法</tspan>", PU,
                  sub="⛔ 三千亿维的山画不出来，"
                      "<tspan font-weight=\"700\">但那一列数是看得懂的</tspan>")

    LX2 = 150
    f.box(LX2, py3 + 44, 250, 200, "#fff", GY2, 8)
    f.t(LX2 + 125, py3 + 76, "所有参数", INK, True, 16, "middle")
    f.t(LX2 + 125, py3 + 102, "排成一列", GY, size=13, anchor="middle")
    for k, v in enumerate(("0.31", "−1.24", "0.07", "⋮")):
        f.t(LX2 + 125, py3 + 136 + k * 26, v, GY2, size=13.5, anchor="middle")
    f.t(LX2 + 125, py3 + 262, "三千亿个数", GY2, size=12.5, anchor="middle")

    f.t(LX2 + 290, py3 + 140, "＋", GY2, True, 22, "middle")

    f.box(LX2 + 330, py3 + 44, 250, 200, "#f3e8fd", PU, 8)
    f.t(LX2 + 455, py3 + 76, "负梯度", PU, True, 17, "middle")
    f.t(LX2 + 455, py3 + 102, "也排成一列", GY, size=13, anchor="middle")
    for k, v in enumerate(("＋0.002", "−0.910", "＋0.004", "⋮")):
        f.t(LX2 + 455, py3 + 136 + k * 26, v, PU, size=13.5, anchor="middle")
    f.t(LX2 + 455, py3 + 262, "一一对应", GY2, size=12.5, anchor="middle")

    f.t(LX2 + 620, py3 + 140, "→", GY2, True, 22, "middle")

    f.box(LX2 + 660, py3 + 44, 560, 200, "#fff", PU, 8)
    f.t(LX2 + 940, py3 + 78, "每一项告诉你<tspan font-weight=\"700\">两件事</tspan>",
        INK, True, 17, "middle")
    f.t(LX2 + 940, py3 + 118, "① <tspan font-weight=\"700\">正负</tspan>：这个参数该往上推还是往下推",
        GY, size=14, anchor="middle")
    f.t(LX2 + 940, py3 + 152, "② <tspan font-weight=\"700\">相对大小</tspan>：哪一项改起来<tspan font-weight=\"700\">更要紧</tspan>",
        PU, True, 15, "middle")
    f.t(LX2 + 940, py3 + 196, "⭐ 第二件才是这一格的重点 ——　",
        GY2, size=13, anchor="middle")
    f.t(LX2 + 940, py3 + 220, "它不只说往哪走，还说<tspan font-weight=\"700\">该先动谁</tspan>",
        PU, True, 14, "middle")
    f._pan = None

    yb = f.band(py3 + PH3 + 20, "ok",
                "所以「为什么可以」这个问题，答案分成能保证的和不能保证的两半",
                ("✅ <tspan font-weight=\"700\">能保证的</tspan>：每一步都沿着"
                 "<tspan font-weight=\"700\">局部下降最快</tspan>的方向走，"
                 "而且步长跟坡度成正比 ——&#160;<tspan font-weight=\"700\">"
                 "只要步子别迈得太离谱，loss 就一路在降</tspan>。",
                 "⛔ <tspan font-weight=\"700\">不能保证的</tspan>：你落在哪个谷"
                 "<tspan font-weight=\"700\">取决于从哪儿出发</tspan>，"
                 "它不保证那是最低的一个。——&#160;梯度下降从来没承诺过最优，"
                 "它只承诺<tspan font-weight=\"700\">每一步都在变好</tspan>。",
                 "⚠️ 但<tspan font-weight=\"700\">「满眼都是坑」这个印象是一维给的</tspan>"
                 "　——　紧接着那一格（`fig-saddle`）就来拆它："
                 "维度一多，坡度为零的地方绝大多数只是垭口，还有路可走。",
                 "⭐⭐⭐ 另外<tspan font-weight=\"700\">「步子到底该多大」它也没回答</tspan> ——&#160;"
                 "它只说「跟坡度成正比」，没说那个比例该是多少。"
                 "再往后那一格（`fig-beststep`）在补这个。"))

    yb = f.src(yb + 16,
               "📌 Ⓑ 末尾那句「最快是相对于你怎么量一步」取自"
               "<tspan font-weight=\"700\">苏剑林</tspan>《为什么我们偏爱各向同性？"
               "基于最速下降的理解》——&#160;原话是「梯度反方向是损失下降最快的方向，"
               "但这结论是有前提的，最关键的前提是它选取的度量是欧氏范数，"
               "如果换一个范数，那么最速方向也就变了」。"
               "⭐ 这句前提<tspan font-weight=\"700\">几乎所有教程都略过</tspan>，"
               "而略过它，Muon 就只能被当成「又一个新优化器」。",
               "📌 「球滚下山」「步长 ∝ 斜率所以不会冲过头」「落在哪个谷取决于起点」"
               "「高维时看那一列数的正负与相对大小」四个讲法，"
               "取自 <tspan font-weight=\"700\">3Blue1Brown</tspan>"
               "《Gradient descent, how neural networks learn》官方讲义　——　"
               "<tspan font-weight=\"700\">已逐条核过原文，图是我们自己重画的</tspan>。",
               "⭐ Ⓒ 那一整列负梯度，<tspan font-weight=\"700\">正是反向传播一遍算出来的那一份</tspan>"
               "　——　所以这一讲的第一节和第三节，接头就在这儿。")

    f.save("fig4-descent.svg", yb + 14)


if __name__ == "__main__":
    main()
