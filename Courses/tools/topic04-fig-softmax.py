# -*- coding: utf-8 -*-
r"""专题四 · §1.0b「logits 怎么变成 loss」——&#160;以及「猜错的那些去哪了」

⭐⭐⭐ 2026-09-22 现场两问，一前一后，其实是同一条线：
  ①「logits 怎么变成 loss？」
  ②「我发现求 loss 的时候，只有正确答案的概率跟 1 之间的差值参与计算，
     剩下那些<b>本来应该是 0、却概率大于 0</b> 的词并没有参与。这是为什么？」

⛔ 两个真洞：
  · **「logits」这个词在本讲里用了十几次，却从来没被定义过** ——&#160;
    第一次出现是在 1.2，那时候读者根本不知道它是什么。
  · softmax 那两步原来只是 1.0 里的**一行小字**。

⭐⭐ 而第二问的答案很漂亮，值一整格：
  **它们参与了，只是不以自己的名义 ——&#160;它们全在 softmax 的分母里。**
  本图用实算证明：**把一个「错」的 logit 抬高，正确那一格一个字不改，
  loss 照样涨 %s。**

⭐ 再加一层：**loss 的式子里看不见它们，梯度里看得见。**
  对错的那些，梯度就是 `+p`（它现在占多少，就被压多大力气）。
  ⇒ 而训练只用梯度。

📌 还有一条**教材里原来一个字都没有、但实际很要命**的：
  真实 logit 可能几十上百，**直接取指数会溢出**（e^102 ≈ 2×10⁴⁴，
  而 fp32 上限约 3.4×10³⁸）。所以所有实现都**先减去这一排里的最大值**再取指数
  ——&#160;数学上完全等价（平移不变），数值上不炸。
  ⭐ 它跟 §6.4 那个 z-loss 是同一件事的两面：
    一个是**事后不让它炸**，一个是**事前不让分数长那么大**。
"""
import math

from topic03_draw import Fig, BL, OR, GR, RD, PU, GY, INK, GY2

W = 1400

WORDS = ("的", "了", "在", "和", "有")
Z = (2.0, 1.0, 1.0, 0.0, 0.0)      # 示意 logits
GOLD = 0                            # 正确答案是第 0 个


def _soft(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e], e, s


P, EXP_RAW, _ = _soft(Z)
LOSS = -math.log(P[GOLD])
# ⛔⛔ 2026-09-23 现场第三问：「为了不爆炸，softmax 取指数之前不是应该先减掉
#   最大值吗？2.0 该变成 0、1.0 变成 −1。你这个写得不精确吧？」——&#160;对，
#   ⛔ 这一格原来画的是**朴素版**（直接 e²＝7.4），而 Ⓑ② 又在说「实现里要先减最大值」
#     ——&#160;**一张图里自己跟自己不一致**，而且不一致的那一半是「真实做法」那一半。
#   ⭐ 改成真实做法之后，顺带把上一轮那个抱怨也解决了：
#     减完最大值，分母 ＝ 1 ÷ 最高那一格的概率 ——&#160;**它终于是个有意义的数了**。
#   ⭐⭐ 判据：**同一张图里「示意版」和「真实版」不能各占一格。**
#     读者不会去分辨哪一格是简化的，他会认为两格在讲同一件事。
ZMAX = max(Z)
EXP = [math.exp(v - ZMAX) for v in Z]   # 真实做法：先减这一排的最大值
ESUM = sum(EXP)                         # 减完之后的分母
ZSUM = sum(Z)                           # logits 自己的总和 ——&#160;它在数学上没有用处
# ⭐ 减完最大值之后这条恒等式成立：分母 ＝ 1 ÷ 最高那一格的概率。
#   这里最高的那一格恰好就是正确答案，所以它同时也是这个位置的困惑度。
assert abs(ESUM - 1.0 / max(P)) < 1e-9, (ESUM, max(P))
assert abs(max(EXP) - 1.0) < 1e-12, "减完最大值之后，最高那一格必须正好是 1"
assert abs(ESUM - 2.01) < 0.01 and abs(ZSUM - 4.0) < 1e-9, (ESUM, ZSUM)
# ⛔⛔ 2026-09-23 现场抓到：第 ① 格底下原来也印着 ESUM（14.83）——&#160;
#   那是**第 ② 格的数被搬到了第 ① 格底下**。logits 这一排加起来是 4.00。
#   ⭐ 而正确的修法不是把它改成 4.00：**logits 的总和本来就没有意义** ——&#160;
#     整排同时加减一个常数，softmax 出来的概率一模一样（这正是后面那个
#     log-sum-exp 技巧站得住的原因）。印一个没有意义的数，等于请人去琢磨它。
#   ⭐⭐ 判据：**三格并排、每格底下都挂一个同名读数时，先问这个读数在每一格
#     是不是都成立。** 版面上的对称会把一个不存在的量也变出来。

# ⭐ 只抬高一个**错**的 logit，正确那一格一个字不改
Z2 = (2.0, 2.0, 1.0, 0.0, 0.0)
assert Z2[GOLD] == Z[GOLD], "正确那一格必须一个字不改 ——&#160;这是这一格的全部论点"
P2, _, _ = _soft(Z2)
LOSS2 = -math.log(P2[GOLD])
assert LOSS2 > LOSS + 0.2, "loss 必须明显涨 ——&#160;涨得太少这一格就没说服力"

GRAD = [P[i] - (1.0 if i == GOLD else 0.0) for i in range(len(Z))]
assert abs(sum(GRAD)) < 1e-12, "梯度那一排加起来必须是 0"

BIG = math.exp(102.0)
FP32MAX = 3.4028235e38
assert BIG > FP32MAX, "溢出那个例子得真的溢出才行"

# ⛔⛔⛔ 2026-09-23 现场第四问，也是这张图最要命的一处：
#   「一开始的 loss 是 0.6964，后来的 loss 被折算成一个数了对吧，比如 11.77 ——&#160;
#     这两个数哪一个是我们平时训练时候说的那个 loss？我怎么觉得是 11.77 这个。
#     所以你要把 0.6964 那个口径说清楚，它其实不是我们平时说的那个 loss。」
#   ⭐ 他是对的。公式确实是同一个 −ln p，但**这张图的 0.6964 差着两层口径**：
#     ① 词表被缩成 5 个词 ——&#160;5 个词瞎猜就是 ln 5 ＝ 1.61，
#        所以 0.6964 只说明「比瞎猜好一点」，跟真词表下的数根本不在一个量级；
#     ② 它只是**一个位置**的数，训练日志里那个是**一整个 batch 取平均**。
#   ⛔ 病因不是算错，是**这张图只说了「词表缩成 5 个词」，没说「所以这个数也缩了」**。
#     读者刚在 1.0 见过 11.77，紧接着看到 0.6964，只能理解成「loss 已经降下来了」。
#   ⭐⭐ 判据：**把一个量的输入缩小成玩具尺寸时，必须同时说明它的输出也是玩具尺寸。**
#     「示意」这两个字读者会自动套在**图形**上，不会自动套在**读数**上。
LN5 = math.log(len(WORDS))          # 5 个词瞎猜的 loss
LN_V3 = math.log(129280)            # 真词表瞎猜的 loss ——&#160;1.0 那把尺的起点
assert abs(LN5 - 1.6094) < 1e-4, LN5
assert abs(LN_V3 - 11.77) < 0.005, LN_V3
assert LOSS < LN5, "玩具词表下这个 loss 必须低于瞎猜，否则「比瞎猜好一点」就说不通"

__doc__ = __doc__ % ("%.4f" % (LOSS2 - LOSS))


def main():
    f = Fig(W, "logits 是最后一层吐出来的一排分数，可正可负没有范围。"
               "softmax 把它变成概率只做两件事：取指数让它们全变正，"
               "再除以总和让它们加起来等于一。"
               "然后翻开正确答案，取那一格概率的负对数，那就是 loss。"
               "给所有 logits 加同一个数，概率完全不变，"
               "所以绝对大小不重要，只有差重要。"
               "而猜错的那些词并不是没参与 —— 它们全在分母里："
               "只把一个错的分数抬高、正确那一格一个字不改，loss 照样会涨。"
               "求导之后它们更是全部显形，每一个的梯度就是它现在占的概率")

    y0 = f.header(
        "logits 怎么变成 loss　——　<tspan font-weight=\"700\">"
        "以及「猜错的那些」去哪了</tspan>",
        "⭐ 三步：<tspan font-weight=\"700\">取指数 → 除以总和 → 翻开答案取负对数</tspan>",
        [(BL, "Ⓐ 三步"), (OR, "Ⓑ 两个性质"), (RD, "Ⓒ 错的那些去哪了")])

    # ══════════ Ⓐ 三步 ═════════════════════════════════════════════
    PH = 400
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 从一排<tspan font-weight=\"700\">分数</tspan>，到一个"
                 "<tspan font-weight=\"700\">数</tspan>", BL,
                 sub="⭐ 词表这里缩成 5 个词；真实是 129,280 个 ——&#160;<tspan font-weight=\"700\">动作一模一样，但算出来的那个数也跟着缩了</tspan>（下面那条专说这件事）")

    COLS = (
        ("① logits（原始分数）", Z, "可正可负，<tspan font-weight=\"700\">没有范围</tspan>", GY2),
        ("② 减掉最大值，再取指数", EXP,
         "全变正数，<tspan font-weight=\"700\">最高那格正好是 1</tspan>", OR),
        ("③ 除以总和 ＝ 概率", P, "加起来<tspan font-weight=\"700\">正好是 1</tspan>", GR),
    )
    for c, (title, vals, note, col) in enumerate(COLS):
        cx = 150 + c * 330
        f.t(cx, py + 58, title, col, True, 15, "middle")
        mx = max(vals) or 1.0
        for k, v in enumerate(vals):
            bx = cx - 118 + k * 50
            h = 76.0 * v / mx
            gold = (k == GOLD)
            f.box(bx, py + 190 - h, 34, h, "#fef7e0" if gold else "#f1f3f4",
                  OR if gold else GY2, 3, sw=1.4 if gold else 1.0)
            f.t(bx + 17, py + 186 - h, "%.2f" % v,
                OR if gold else GY2, gold, 11, "middle")
            f.t(bx + 17, py + 212, WORDS[k], INK if gold else GY, gold, 14, "middle")
        f.t(cx, py + 244, note, GY, size=12.5, anchor="middle")
        if c == 1:
            # ⭐ 现场把这一步的算式念出来了（「2.0 变成 0、1.0 变成 −1」）——
            #   那就把它印在图上，否则「减掉最大值」只是个说法，跟不上手。
            f.t(cx, py + 264,
                "各减 %.2f：　0　／　−1　／　−1　／　−2　／　−2" % ZMAX,
                GY2, size=11.5, anchor="middle")
        if c < 2:
            f.line(cx + 132, py + 152, cx + 194, py + 152, col, 2.4)

    f.t(150, py + 282,
        "这一排的总和<tspan font-weight=\"700\">不用管</tspan>", GY2,
        size=12, anchor="middle")
    f.t(150, py + 302,
        "整排同时加减一个数，概率一模一样　→　所以下一格敢减", GY2,
        size=11.5, anchor="middle")
    # ⛔⛔ 2026-09-23 现场第二问：「14.83 这个数也没用 ——&#160;它不是 11.77 那种
    #   由词表决定的上限，你把大家的 logits 都抬高一点，它就变大。」——&#160;对。
    #   ⭐ 它**不是 shift 不变的**：整排 logits ＋c，这个和就乘 e^c，而概率一个不变。
    #     所以它不是任何意义上的指标。
    #   ⛔ 但也别删掉：③ 那一格是怎么来的，全靠它 ——&#160;7.4 ÷ 14.83 ＝ 0.4984，
    #     读者能在纸上自己验一遍。⭐ 所以改的是**名字**，不是有没有：
    #     从「总和」改成「分母」，再加一句它不是指标。
    #   ⭐⭐ 判据：**一个只在中间步骤里当工具的量，标它的<u>用途</u>，不要标它的<u>身份</u>。**
    #     「总和」听起来像个读数，「分母」一眼就知道它下一步要被用掉。
    #   ⚠️ ③ 那个 1.00 不一样 ——&#160;它**是**有意义的：概率加起来必须等于 1，
    #     那是这一步做完的凭证。所以它保留「总和」这个名字。
    f.t(480, py + 282, "分母 ＝ %.2f" % ESUM, OR, True, 12.5, "middle")
    f.t(480, py + 302, "＝ 1 ÷ 最高那格的概率　（这里也就是困惑度）",
        GY2, size=11.5, anchor="middle")
    f.t(810, py + 282, "总和 ＝ 1.00", GR, True, 12.5, "middle")
    f.t(810, py + 302, "这一条才是有意义的", GY2, size=11.5, anchor="middle")

    f.box(1010, py + 60, 350, 200, "#e8f0fe", BL, 8)
    f.t(1185, py + 96, "④ 翻开正确答案", BL, True, 16, "middle")
    # ⛔ 2026-09-23 现场：「图上不是画的 0.50 吗？这个 0.4984 怎么来的？」
    #   同一个数，③ 那一排的柱子上是**两位小数**（0.50），这儿要四位
    #   ——&#160;因为 loss 取的是它的对数，−ln 0.4984 ＝ 0.6964，
    #   而 −ln 0.50 ＝ 0.6931，**第三位就分家了**。
    #   ⭐ 判据：**同一个量在同一张图上用了两种精度，就得当场说一句。**
    #     读者看见两个不一样的数，第一反应永远是「这是两个数」。
    f.t(1185, py + 132, "正确的是「%s」，它的概率 <tspan font-weight=\"700\">%.4f</tspan>"
        % (WORDS[GOLD], P[GOLD]), INK, size=14.5, anchor="middle")
    f.t(1185, py + 154,
        "（就是 ③ 里那一格 ——&#160;那边四舍五入成了 %.2f）" % P[GOLD],
        GY2, size=11.5, anchor="middle")
    f.t(1185, py + 182, "<tspan font-weight=\"700\">这一个位置</tspan>的 loss ＝ −ln %.4f" % P[GOLD], INK, True, 16, "middle")
    f.t(1185, py + 218, "＝ <tspan font-weight=\"700\">%.4f</tspan>" % LOSS,
        BL, True, 24, "middle")
    f.t(1185, py + 252, "⛔ 其余四个的概率，"
        "<tspan font-weight=\"700\">式子里一个都没出现</tspan>",
        GY2, size=12, anchor="middle")
    f.t(1185, py + 292, "——　那它们去哪了？<tspan font-weight=\"700\">看 Ⓒ。</tspan>",
        RD, True, 14, "middle")

    f.t(700, py + 350,
        "⭐⭐ 顺带记住这个词：<tspan font-weight=\"700\">logits ＝ 最后一层吐出来的那排原始分数</tspan>"
        "　——　<tspan font-weight=\"700\">它还不是概率</tspan>，中间隔着 softmax 这两步。",
        INK, size=14.5, anchor="middle")
    f._pan = None

    # ⭐⭐ 紧贴着 0.6964 放 ——&#160;读者就是在这儿把它当成「训练曲线上的 loss」的。
    #   ⛔ 这一条**不能 fold**：它不是「别误用这张图」的脚注，
    #     是「这个数到底是什么」——&#160;缺了它，这张图给的是一个错的量级。
    ycal = f.band(py + PH + 18, "warn",
                  "⛔ 这个 %.4f，<tspan font-weight=\"700\">不是</tspan>"
                  "你在训练曲线上看到的那个 loss" % LOSS, [
        "式子是同一个 <tspan font-weight=\"700\">−ln p</tspan>，但这张图为了画得下，"
        "<tspan font-weight=\"700\">把口径缩了两层</tspan>：词表只有 "
        "<tspan font-weight=\"700\">%d 个词</tspan>，而且只看"
        "<tspan font-weight=\"700\">一个位置</tspan>。"
        "5 个词瞎猜就是 −ln(1/5) ＝ <tspan font-weight=\"700\">%.2f</tspan>　——　"
        "所以 <tspan font-weight=\"700\">%.4f 的全部含义是「比瞎猜好一点」</tspan>，"
        "不是「已经训得很好了」。" % (len(WORDS), LN5, LOSS),
        "⭐⭐ 训练日志里那个 loss ＝ <tspan font-weight=\"700\">真词表 129,280 个词</tspan>"
        " ＋ <tspan font-weight=\"700\">一整个 batch 所有位置取平均</tspan>，"
        "它瞎猜的起点是 <tspan font-weight=\"700\">%.2f</tspan>"
        "（<tspan font-weight=\"700\">1.0</tspan> 那把尺的顶端，也是训练曲线的第 0 步）。"
        "⛔ 所以 <tspan font-weight=\"700\">loss 的数值一半是词表定的</tspan>　——　"
        "<tspan font-weight=\"700\">词表不一样的两个模型，loss 不能直接比。</tspan>" % LN_V3,
    ], keep=True)

    # ══════════ Ⓑ 两个性质 ═════════════════════════════════════════
    PH2 = 258
    py2 = f.panel(0, ycal + 2, W, PH2,
                  "Ⓑ 两个性质，一个漂亮、一个救命", OR)

    f.box(50, py2 + 52, 640, 180, "#fef7e0", OR, 8)
    f.t(370, py2 + 90, "① 加同一个数，概率完全不变", OR, True, 16.5, "middle")
    f.t(370, py2 + 128,
        "把那一排 logits <tspan font-weight=\"700\">全部 ＋100</tspan>　——　"
        "算出来还是 <tspan font-weight=\"700\">%.4f</tspan>。" % P[GOLD],
        INK, size=14.5, anchor="middle")
    f.t(370, py2 + 166,
        "⭐ 所以 logits 的<tspan font-weight=\"700\">绝对大小不重要，只有「差」重要</tspan>。",
        INK, True, 14.5, "middle")
    f.t(370, py2 + 200,
        "指数干的事，就是<tspan font-weight=\"700\">把「差」变成「比」</tspan>。",
        GY, size=13.5, anchor="middle")

    f.box(710, py2 + 52, 640, 180, "#fce8e6", RD, 8)
    f.t(1030, py2 + 90, "② 可别真直接取指数　——　会炸", RD, True, 16.5, "middle")
    f.t(1030, py2 + 128,
        "真实 logit 可能几十上百："
        "<tspan font-weight=\"700\">e¹⁰² ≈ %.0e</tspan>，" % BIG,
        INK, size=14.5, anchor="middle")
    f.t(1030, py2 + 158,
        "而 fp32 的上限只有 <tspan font-weight=\"700\">%.1e</tspan>。" % FP32MAX,
        INK, size=14.5, anchor="middle")
    f.t(1030, py2 + 196,
        "⭐ 所以实现里<tspan font-weight=\"700\">先减去这排里的最大值</tspan>再取指数"
        "　——　<tspan font-weight=\"700\">Ⓐ 第 ② 格画的就是这一步</tspan>",
        RD, True, 14.5, "middle")
    f.t(1030, py2 + 226,
        "——　敢减，靠的正是左边那条平移不变。"
        "<tspan font-weight=\"700\">数学等价，数值不炸。</tspan>",
        GY, size=13.5, anchor="middle")
    f._pan = None

    # ══════════ Ⓒ 错的那些去哪了 ═══════════════════════════════════
    PH3 = 386
    py3 = f.panel(0, py2 + PH2 + 20, W, PH3,
                  "Ⓒ 那<tspan font-weight=\"700\">猜错的那些</tspan>去哪了？"
                  "　——　它们全在<tspan font-weight=\"700\">分母</tspan>里", RD,
                  sub="⭐ 正确答案的概率 ＝ 它自己的指数 ÷ "
                      "<tspan font-weight=\"700\">所有格子的指数之和</tspan>"
                      "　——　那个和里装着<tspan font-weight=\"700\">每一个词</tspan>")

    for side, (zz, pp, ll, lab, col) in enumerate((
            (Z, P, LOSS, "原来", GY2),
            (Z2, P2, LOSS2, "只把一个「错」的抬高一格", RD))):
        ox = 130 + side * 400
        f.t(ox + 100, py3 + 64, lab, col, True, 14.5, "middle")
        for k, v in enumerate(zz):
            bx = ox + k * 42
            # ⛔ logit ＝ 0 的那两根高度会是 0，渲染出来「五个词只看得见三个」。
            #   ⭐ 给个地板：柱子代表的是「有这一格」，不是只代表大小。
            h = 46.0 * (v / 2.0) + 5.0
            gold = (k == GOLD)
            changed = (side == 1 and zz[k] != Z[k])
            f.box(bx, py3 + 150 - h, 30, h,
                  "#fef7e0" if gold else ("#fce8e6" if changed else "#f1f3f4"),
                  OR if gold else (RD if changed else GY2), 3,
                  sw=1.6 if (gold or changed) else 1.0)
            if changed:
                f.line(bx + 15, py3 + 190, bx + 15, py3 + 158, RD, 2.0)
                f.t(bx + 15, py3 + 210, "抬高", RD, True, 11.5, "middle")
        f.t(ox + 100, py3 + 244,
            "正确那格的概率 <tspan font-weight=\"700\">%.4f</tspan>" % pp[GOLD],
            INK, size=13.5, anchor="middle")
        f.t(ox + 100, py3 + 276, "loss ＝ <tspan font-weight=\"700\">%.4f</tspan>" % ll,
            col if side else GY, True, 18, "middle")
        if side == 0:
            f.line(ox + 232, py3 + 150, ox + 296, py3 + 150, RD, 2.4)

    f.t(390, py3 + 314,
        "⛔ <tspan font-weight=\"700\">正确那一格的分数，一个字都没改</tspan>"
        "　——　可 loss 涨了 <tspan font-weight=\"700\">%.4f</tspan>。" % (LOSS2 - LOSS),
        RD, True, 15, "middle")
    f.t(390, py3 + 344,
        "⭐⭐ 所以它们<tspan font-weight=\"700\">参与了</tspan>，"
        "只是<tspan font-weight=\"700\">不以自己的名义</tspan>。",
        INK, True, 15, "middle")

    f.box(940, py3 + 56, 410, 290, "#e6f4ea", GR, 8)
    f.t(1145, py3 + 92, "而求导之后，它们全部显形", GR, True, 16.5, "middle")
    for k, g in enumerate(GRAD):
        yy = py3 + 122 + k * 30
        c = GR if g < 0 else RD
        f.t(1000, yy, WORDS[k] + ("（正确）" if k == GOLD else ""),
            INK if k == GOLD else GY, k == GOLD, 13)
        f.t(1290, yy, "%+.4f" % g, c, True, 14, "end")
    f.t(1145, py3 + 288, "加起来 ＝ <tspan font-weight=\"700\">0</tspan>"
        "　——　一次<tspan font-weight=\"700\">再分配</tspan>", OR, True, 14, "middle")
    f.t(1145, py3 + 320,
        "⭐ 错的那些，梯度<tspan font-weight=\"700\">正好等于它现在占的概率</tspan>。",
        GY, size=13, anchor="middle")
    f._pan = None

    yb = f.band(py3 + PH3 + 18, "ok", "⭐ 这一格的两句话", [
        "<tspan font-weight=\"700\">logits ＝ 最后一层的原始分数</tspan>（可正可负）；"
        "<tspan font-weight=\"700\">取指数 →&#160;除以总和</tspan> 变成概率，"
        "再<tspan font-weight=\"700\">翻开答案取负对数</tspan>，就是 loss。",
        "<tspan font-weight=\"700\">猜错的那些没有从式子里消失，它们在分母里</tspan>　——　"
        "而且<tspan font-weight=\"700\">求导之后每一个都显形</tspan>。"
        "⭐ 一句话：<tspan font-weight=\"700\">loss 的式子里看不见它们，梯度里看得见；"
        "而训练只用梯度。</tspan>",
    ])

    yb = f.src(yb + 10,
               "⭐ 完整的交叉熵其实是<tspan font-weight=\"700\">对整个词表求和</tspan>；"
               "只因为标签是 one-hot（只有一格是 1），那个和里<tspan font-weight=\"700\">"
               "只剩一项非零</tspan>　——　所以「只看一个」是 one-hot 的结果，不是定义。",
               "⚠️ 反过来验证：标签<tspan font-weight=\"700\">不是</tspan> one-hot 时"
               "（label smoothing、蒸馏的软标签），"
               "<tspan font-weight=\"700\">每一项就真的都会出现在 loss 里</tspan>。"
               "数都是本脚本现算并 assert 住的；fp32 上限取 %.4e。" % FP32MAX)

    f.save("fig4-softmax.svg", yb + 14)


if __name__ == "__main__":
    main()
