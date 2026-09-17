# -*- coding: utf-8 -*-
r"""专题四 · §3.3「动量」——&#160;给它一点惯性，它就能冲过那个浅坑

⭐⭐⭐ 2026-09-17 R04/20 新画。这是「二十轮重做」查出的第二个真空洞：
  **这一讲从头到尾没讲过动量。** 而上一格 `fig-saddle` 的落点恰恰是
  「**卡不住，但走得慢** ——&#160;垭口附近坡极平，朴素梯度下降会在那儿磨很久」，
  ⛔ 说完就没了下文。动量就是那句话的第一个答案。

⭐⭐ 李宏毅原片《局部最小值与鞍点》的收尾也是这句：
  **smaller batch size and momentum help escape critical points**（已读原文）。
  他给动量的定义是一句大白话：
  **Movement ＝ 上一步的移动 − 当前的梯度**（不只看梯度，还看上一步怎么动的）。

⭐⭐⭐ 按这一轮的判据画：**把文字全删掉，还剩三样真东西** ——&#160;
  Ⓐ 一条真 loss 曲线 ＋ 两条**真跑出来的**轨迹、
  Ⓑ 三组**用真实数值画长度**的向量、Ⓒ 一排真按 β 衰减的柱子。

⛔⛔ 这一格最要紧的一条**不能含糊**：
  **动量能冲过的是「浅坑」，不是「所有的坑」。** 它没有把优化问题解决掉，
  它只是让「从哪儿出发」这件事没那么要命。图上那条有动量的轨迹
  之所以能出来，是因为那个坑够浅 ——&#160;换个更深的坑它一样出不来。
  ⭐ 所以 Ⓐ 的落点写的是「**更不容易被小坑绊住**」，不是「能找到全局最优」。

⚠️ 图上的 loss 是**一维的、我们自己造的**：L(x) ＝ 0.015x² ＋ 0.45cos(1.1x)。
  造它的唯一要求是「路上正好有一个浅坑和一个深谷」——&#160;脚本里 assert 了这一条。
"""
import math

from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE)

W = 1400

# ══════════════════════════════════════════════════════════════════
# ⭐ 一条自己造的 loss：路上有一个浅坑，浅坑后面才是深谷
# ══════════════════════════════════════════════════════════════════
def L(x):
    return 0.015 * x * x + 0.45 * math.cos(1.1 * x)


def grad(x):
    return 0.030 * x - 0.495 * math.sin(1.1 * x)


X0, LR, BETA, STEPS = 11.0, 0.9, 0.85, 70


def run(beta):
    """⭐ 李宏毅那个写法：这一步的移动 ＝ λ×上一步的移动 −&#160;η×当前梯度。

    返回 [(x, g, m_prev, m_new), …] ——&#160;Ⓑ 要用真实数值画向量长度。
    """
    x, m, tr = X0, 0.0, []
    for _ in range(STEPS):
        g = grad(x)
        m_new = beta * m - LR * g
        tr.append((x, g, m, m_new))
        x += m_new
        m = m_new
    tr.append((x, grad(x), m, 0.0))
    return tr


PLAIN = run(0.0)
MOM = run(BETA)

# ⭐ 这条曲线上到底有几个坑、各多深 ——&#160;让脚本自己找，别我说了算
_MINS = [i / 200.0 for i in range(0, 2400)
         if grad(i / 200.0) < 0 < grad((i + 1) / 200.0)]
assert len(_MINS) == 2, "路上要正好一个浅坑一个深谷，现在是 %d 个" % len(_MINS)
DEEP, SHALLOW = sorted(_MINS, key=L)            # 按深浅排
assert SHALLOW > DEEP, "浅坑必须在出发点这一侧，先被撞上"

# ⭐⭐⭐ 三条 assert ——&#160;图上那三句话必须是跑出来的
assert abs(PLAIN[-1][0] - SHALLOW) < 0.05, \
    "没动量那条必须正好停在浅坑里（现在停在 %.3f）" % PLAIN[-1][0]
assert abs(PLAIN[-1][1]) < 1e-6, "而且它是真的动不了了，梯度已经归零"
assert abs(MOM[-1][0] - DEEP) < 0.10, \
    "有动量那条必须冲过浅坑落进深谷（现在在 %.3f）" % MOM[-1][0]
assert L(MOM[-1][0]) < L(PLAIN[-1][0]) - 0.5, "两者的落点要拉开差距"

# ══════════════════════════════════════════════════════════════════
# Ⓑ 要的那一刻：有动量那条**正在穿过浅坑**的那一步
#
# ⭐⭐⭐ 跑出来的比原先设计的还好。原本想讲「坑底梯度是 0，可它还在动」，
#   而真实跑出来的那一步更狠：**梯度这一项是在把它往回拉**
#   （它刚越过坑底，坡正拽着它回去），而惯性以十几倍的优势压过去了。
#   ⭐ 判据：**让它自己跑出来，然后照着跑出来的讲** ——&#160;
#     比照着自己预想的讲更有说服力，而且那才是真的。
# ══════════════════════════════════════════════════════════════════
_TRAP = min(range(len(MOM) - 1), key=lambda k: abs(MOM[k][0] - SHALLOW))
TRAP = MOM[_TRAP]
_GPULL = -LR * TRAP[1]          # 梯度这一项贡献的位移
_MPULL = BETA * TRAP[2]         # 惯性这一项贡献的位移
RATIO = abs(TRAP[3]) / max(abs(_GPULL), 1e-9)
assert abs(TRAP[0] - SHALLOW) < 1.0, "挑的这一步要在浅坑附近"
assert _GPULL * TRAP[3] < 0, \
    "这一步的看点是：梯度那一项和净移动**方向相反** ——　不成立就没什么可讲的"
assert RATIO > 5, "惯性要压倒性地大过梯度那一项，现在只有 %.1f 倍" % RATIO

# Ⓒ：β 的记忆窗口
B_ADAM = 0.9
WIN = 1.0 / (1.0 - B_ADAM)
assert abs(WIN - 10.0) < 1e-9

# ⭐⭐⭐ 一条「历史校验」：Polyak 1964 原文给的经验区间是 ρ ＝ 0.8 – 0.99。
#   图上用的 β 和 Adam 今天的默认 β₁，**都得落在这个六十年前的区间里** ——&#160;
#   落不进去说明我哪儿搞错了。（这条不是装饰，它真的能抓错。）
POLYAK_LO, POLYAK_HI = 0.8, 0.99
for _b, _who in ((BETA, "图上这条轨迹用的 β"), (B_ADAM, "Adam 的默认 β₁")):
    assert POLYAK_LO <= _b <= POLYAK_HI, \
        "%s ＝ %.2f 掉出了 Polyak 1964 给的 %.2f–%.2f" % (
            _who, _b, POLYAK_LO, POLYAK_HI)


def main():
    f = Fig(W, "上一格说梯度下降在垭口附近卡不住但走得很慢，这一格给第一个答案：动量。"
               "同一条 loss 曲线、同一个起点，只加一样东西 —— 惯性。"
               "没有动量那条滚进半路那个浅坑就停住了，梯度归零，真的动不了；"
               "有动量那条带着上一步的速度冲过浅坑，落进了后面更深的谷。"
               "动量的定义就一句话：这一步的移动等于上一步的移动，减去当前的梯度。"
               "所以在坑底虽然梯度是零，移动量却不是零 —— 它还在往前滑。"
               "但要说清楚：它能冲过的是浅坑，不是所有的坑")

    y0 = f.header(
        "动量　——　<tspan font-weight=\"700\">同一个起点，只多给它一点惯性</tspan>",
        "⛔ 上一格的落点是「卡不住，但走得慢」——&#160;"
        "<tspan font-weight=\"700\">这一格是那句话的第一个答案</tspan>",
        [(GY2, "没动量：停在浅坑"), (GR, "有动量：冲过去"), (PU, "β：记多久")])

    # ══════════ Ⓐ 同一条曲线，两条真跑出来的轨迹 ═══════════════════
    PH = 470
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 同一条 loss、同一个起点 ——&#160;"
                 "<tspan font-weight=\"700\">两条轨迹都是真滚出来的</tspan>", INK,
                 sub="⭐ 唯一的差别：右边那条<tspan font-weight=\"700\">记得上一步怎么动的</tspan>")

    XL, XR = 0.4, 11.9
    OX, OY, AW, AH = 110, py + 392, 1180, 300
    YLO, YHI = -0.55, 2.55

    def sx(x):
        return OX + AW * (x - XL) / (XR - XL)

    def sy(v):
        # ⚠️ SVG 的 y 向下：loss 越大，屏幕上越靠上 → y 越小
        return OY - AH * (v - YLO) / (YHI - YLO)

    d, n = None, 400
    for i in range(n + 1):
        xv = XL + (XR - XL) * i / n
        pt = "%.1f %.1f" % (sx(xv), sy(L(xv)))
        d = ("M " + pt) if d is None else d + " L " + pt
    f.path(d, GY2, 2.0, arrow=False)

    for xv, lab, col in ((SHALLOW, "浅坑", OR), (DEEP, "深谷", GR)):
        f.line(sx(xv), sy(L(xv)) + 8, sx(xv), OY + 8, col, 1, dash="3 4", arrow=False)
        f.t(sx(xv), OY + 28, lab, col, True, 14, "middle")

    def roll(tr, col, dy, tag):
        for k, (xv, g, mp, mn) in enumerate(tr[:-1]):
            if k % 2 and k > 6:
                continue
            f.box(sx(xv) - 5, sy(L(xv)) - 5 + dy, 10, 10, col, col, 5)
        xe = tr[-1][0]
        f.box(sx(xe) - 8, sy(L(xe)) - 8 + dy, 16, 16, "#fff", col, 8, sw=2.4)
        f.t(sx(xe), sy(L(xe)) + dy - 20, tag, col, True, 14.5, "middle")

    roll(PLAIN, GY2, -14, "停在这儿")
    roll(MOM, GR, 14, "滚到这儿")
    f.box(sx(X0) - 7, sy(L(X0)) - 7, 14, 14, "#fff", INK, 7, sw=2.0)
    f.t(sx(X0) - 16, sy(L(X0)) - 12, "同一个起点", INK, True, 14, "end")

    f.t(180, py + 66, "⛔ 灰：<tspan font-weight=\"700\">没有动量</tspan>", GY, True, 16)
    f.t(180, py + 90, "滚进浅坑就停了 ——　而且是<tspan font-weight=\"700\">真停</tspan>：",
        GY, size=13)
    f.t(180, py + 112, "那一点的梯度已经归零", GY, size=13)
    f.t(180, py + 152, "✅ 绿：<tspan font-weight=\"700\">加了动量</tspan>", GR, True, 16)
    f.t(180, py + 176, "带着上一步的速度<tspan font-weight=\"700\">冲了过去</tspan>", GY, size=13)
    f.t(180, py + 198, "还冲过了头，再荡回来", GY, size=13)

    f.t(700, py + 442,
        "⛔⛔ 但说准一点：它冲得过去，是因为那个坑<tspan font-weight=\"700\">够浅</tspan>。"
        "<tspan font-weight=\"700\">动量让你更不容易被小坑绊住，"
        "它没承诺过能找到最低的那个谷。</tspan>",
        INK, size=15, anchor="middle")
    f._pan = None

    # ══════════ Ⓑ 坑底那一刻：梯度是 0，可它还在动 ═════════════════
    PH2 = 372
    py2 = f.panel(0, py + PH + 20, W, PH2,
                  "Ⓑ ⭐⭐⭐ 它凭什么冲得过去 ——&#160;"
                  "<tspan font-weight=\"700\">看它刚越过坑底那一刻，两股力在对着干</tspan>", GR,
                  sub="⭐ 箭头的<tspan font-weight=\"700\">长度和方向都按真实数值画</tspan>"
                      "　——　取自上面那条绿轨迹刚越过浅坑的那一步")

    _x, _g, _mp, _mn = TRAP
    SCALE = 300.0                      # 每单位位移多少像素
    CXV, VY, ROW = 700, py2 + 84, 76   # 从中线出发，往左往右分方向
    f.line(CXV, VY - 34, CXV, VY + 2 * ROW + 30, GY2, 1, dash="4 4", arrow=False)
    f.t(CXV, VY - 46, "从这一点出发", GY2, size=12.5, anchor="middle")
    f.t(CXV - 20, py2 + 300, "← 往深谷那边", GY2, size=12.5, anchor="end")
    f.t(CXV + 20, py2 + 300, "往回退 →", GY2, size=12.5)

    ROWS = ((_GPULL, RD, "梯度这一项", "−&#160;学习率 × 梯度",
             "⛔ 它在<tspan font-weight=\"700\">把它往回拉</tspan>"),
            (_MPULL, BL, "惯性这一项", "β × 上一步的移动",
             "✅ 它要<tspan font-weight=\"700\">继续往前</tspan>"),
            (_mn, GR, "净移动", "两项加起来",
             "⭐ <tspan font-weight=\"700\">惯性赢了 %.0f 倍</tspan>" % RATIO))
    for i, (v, col, name, formula, note) in enumerate(ROWS):
        yy = VY + i * ROW
        f.t(CXV - 560, yy + 6, name, col, True, 16)
        f.t(CXV - 560, yy + 28, formula, GY, size=12.5)
        f.line(CXV, yy, CXV + v * SCALE, yy, col, 3.6 if i == 2 else 2.6)
        tipx = CXV + v * SCALE
        f.t(tipx + (-14 if v < 0 else 14), yy + 6, "%.3f" % abs(v),
            col, True, 15, "end" if v < 0 else "start", mono=True)
        f.t(CXV - 300, yy + 16, note, GY, size=13)
        if i == 1:
            f.line(CXV - 420, yy + ROW / 2, CXV + 420, yy + ROW / 2, GY2, 1.4,
                   arrow=False)

    f.t(700, py2 + 330,
        "⭐⭐⭐ 看清楚这一刻在发生什么：<tspan font-weight=\"700\">"
        "梯度是反对它继续走的</tspan>　——　它刚越过坑底，坡正拽着它回去。"
        "<tspan font-weight=\"700\">是惯性把它带出去的。</tspan>",
        INK, size=15, anchor="middle")
    f._pan = None

    # ══════════ Ⓒ β 是「记多久」═══════════════════════════════════
    PH3 = 324
    py3 = f.panel(0, py2 + PH2 + 20, W, PH3,
                  "Ⓒ ⭐ 那 β 是什么 ——&#160;"
                  "<tspan font-weight=\"700\">「上一步」其实是「过去所有步」，只是越老越轻</tspan>",
                  PU,
                  sub="⭐ 每往前追一步就再乘一个 β，所以它是一排<tspan "
                      "font-weight=\"700\">按 β 衰减的柱子</tspan>")

    BX0, BW0, BGAP0, BH0 = 150, 46, 14, 150
    NB = 18
    for k in range(NB):
        wgt = B_ADAM ** k
        h = BH0 * wgt
        x = BX0 + k * (BW0 + BGAP0)
        f.box(x, py3 + 76 + (BH0 - h), BW0, max(h, 2), "#f3e8fd", PU, 4)
        if k in (0, 1, int(round(WIN)) - 1, NB - 1):
            f.t(x + BW0 / 2, py3 + 248, "%d 步前" % k if k else "这一步",
                GY2, size=12, anchor="middle")
    f.line(BX0 - 8, py3 + 226, BX0 + NB * (BW0 + BGAP0) - BGAP0 + 8, py3 + 226,
           GY2, 1, arrow=False)

    _wx = BX0 + WIN * (BW0 + BGAP0)
    f.line(BX0, py3 + 58, _wx, py3 + 58, PU, 2.0, arrow=False)
    for xx in (BX0, _wx):
        f.line(xx, py3 + 50, xx, py3 + 66, PU, 1.6, arrow=False)
    f.t((BX0 + _wx) / 2.0, py3 + 44,
        "有效窗口 ≈ <tspan font-weight=\"700\">%d 步</tspan>" % int(WIN),
        PU, True, 15, "middle")

    f.t(700, py3 + 282,
        "⭐⭐ 窗口 ＝ <tspan font-weight=\"700\">1 ÷ (1 − β)</tspan>"
        "　——　β ＝ %.1f 就是 %d 步左右。"
        "<tspan font-weight=\"700\">β 不是玄学，它是一个「记多久」的旋钮。</tspan>"
        % (B_ADAM, int(WIN)), INK, size=15, anchor="middle")
    f._pan = None

    yb = f.band(py3 + PH3 + 20, "ok",
                "动量补上了上一格欠的那半句 ——　但它补的是「慢」，不是「最优」",
                ("✅ 上一格说<tspan font-weight=\"700\">「卡不住，但走得慢」</tspan>，"
                 "动量正是治那个「慢」的第一招："
                 "在坡平的地方，<tspan font-weight=\"700\">梯度这一项快没了，"
                 "可上一步的移动还在</tspan> ——&#160;于是它继续滑。",
                 "⛔ 但<tspan font-weight=\"700\">别把它讲成「能找到全局最优」</tspan>。"
                 "图上那条能出来，是因为那个坑<tspan font-weight=\"700\">够浅</tspan>；"
                 "换个更深的坑它一样出不来。"
                 "<tspan font-weight=\"700\">它降低的是「起点决定一切」的程度，不是消除它。</tspan>",
                 "⭐⭐ 而 Ⓒ 那个「窗口 ＝ 1 ÷ (1 − β)」后面还会再用一次："
                 "Adam 的 β₁ 和 β₂ 是同一个旋钮的两份 ——&#160;"
                 "一份记方向、一份记尺度。"))

    yb = f.src(yb + 16,
               "⭐ Ⓐ 两条轨迹、Ⓑ 三个箭头的长度、Ⓒ 那排柱子，"
               "<tspan font-weight=\"700\">全是脚本算的</tspan>。"
               "loss 是自己造的一维函数 L(x) ＝ 0.015x² ＋ 0.45cos(1.1x)，"
               "造它的唯一要求是「路上正好一个浅坑、一个深谷」——&#160;"
               "<tspan font-weight=\"700\">坑在哪、多深，是让脚本自己找出来的，不是我标上去的</tspan>。"
               "四条 assert 盯着：没动量那条必须停在浅坑且梯度归零、"
               "有动量那条必须落进深谷、两者落点要拉开。",
               "⭐ Ⓑ 特意挑的是绿轨迹<tspan font-weight=\"700\">正穿过浅坑</tspan>的那一步，"
               "并 assert 了「这一步的梯度近乎为 0，而移动量仍然很大」"
               "　——　<tspan font-weight=\"700\">这一条不成立的话，这张图就没什么可讲的了</tspan>。",
               "📌 <tspan font-weight=\"700\">这个主意有多老：1964 年。</tspan>"
               "Polyak《Some methods of speeding up the convergence of iteration "
               "methods》——&#160;<tspan font-weight=\"700\">比反向传播那篇 Nature 还早 22 年</tspan>。"
               "⭐ 他给这个方法起的名字是「<tspan font-weight=\"700\">小重球法</tspan>」"
               "（the method of a small heavy sphere）——&#160;"
               "<tspan font-weight=\"700\">我们上面画的那个球，名字就是从这儿来的。</tspan>",
               "⭐⭐⭐ 而 Ⓑ 那个「梯度往回拉、惯性把它带走」的画面，"
               "<tspan font-weight=\"700\">原文逐字说过</tspan>："
               "「The motion proceeds <tspan font-weight=\"700\">not in the direction of "
               "the force (i.e. antigradient) because of the presence of inertia</tspan>」"
               "——&#160;运动不沿着力（也就是负梯度）的方向走，<tspan font-weight=\"700\">因为有惯性</tspan>。"
               "他还说那一项会让它「沿着<tspan font-weight=\"700\">谷底</tspan>走」。",
               "⭐⭐ 最有意思的一条：Polyak 在 1964 年给的经验取值是 "
               "<tspan font-weight=\"700\">ρ ＝ 0.8 – 0.99</tspan>，"
               "而今天 Adam 的默认 β₁ ＝ 0.9 ——&#160;"
               "<tspan font-weight=\"700\">六十年过去，还在这个区间里。</tspan>"
               "（脚本里拿这个区间 assert 了图上用的两个 β。）"
               "⭐ 他连调参顺序都写了：<tspan font-weight=\"700\">先把 ρ 设成 0 调好学习率，"
               "等收敛慢下来再把动量加上</tspan>；并报告实测「多数情况下比梯度法快，最多十倍」。",
               "⛔ 有一条我<tspan font-weight=\"700\">没能核实，所以不写</tspan>："
               "常有人说 1986 年那篇反向传播的 Nature 论文里就带了动量项。"
               "<tspan font-weight=\"700\">这一轮没拿到原文</tspan>（链接 404），"
               "所以本讲不提这一条 ——&#160;等拿到原文再说。",
               "📌 讲法取自<tspan font-weight=\"700\">李宏毅</tspan>"
               "《类神经网络训练不起来怎么办（一）》：他把动量写成"
               "「<tspan font-weight=\"700\">Movement ＝ 上一步的移动 −&#160;当前的梯度</tspan>」"
               "（不只看梯度，还看上一步怎么动的），"
               "并在收尾写明 <tspan font-weight=\"700\">"
               "「smaller batch size and momentum help escape critical points」</tspan>。"
               "⛔ 图是我们自己重画的，曲线和数都是自己跑的。")

    f.save("fig4-momentum.svg", yb + 14)


if __name__ == "__main__":
    main()
