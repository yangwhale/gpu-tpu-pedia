# -*- coding: utf-8 -*-
r"""专题四 · §1.2b「那 6,300 万个 loss，到底是怎么变成一份梯度的」

⭐⭐⭐ 2026-09-22 现场追问（原话，一字不改）：
  「首先我一个 Sequence，它会预测 4096 次 Next token……
    正确的是 40%，剩下的是 20、20、十十那么几个，凑一块会有 60%。
    那你算反向的时候，**你的目标要不要把那几个推向 0，然后把真正要预测那个值
    推向 1，都做？还是说只推那个 1，剩下那些都变成副作用？**」

⛔ 这一问之前，讲稿和课件里**根本没有答案** ——&#160;
  fig-vote Ⓒ 画的是「样本 1 / 样本 2 / 样本 3」，
  那会让人以为**一条序列＝一个意见**。
  而实际上一条 4,096 长的序列**自己就是四千多个意见**，
  每个意见还是一个**整个词表那么长的向量**。

⭐ 所以这张图专门回答那一问，而且按现场要求**先 batch=1 再 batch=2**。
  答案一句话：**两个都做，而且是同一个动作** ——&#160;
  一个位置上的梯度是「你猜的 减去 正确答案」，
  正确那格是负数（把它推高），其余每一格都是正数（把它们推低），
  **而这一整排数加起来正好是 0**：概率总量守恒，
  所以「推上去」和「推下去」不是两件事，是一次**再分配**。

⛔⛔ 口径（都核过，别改）：
  · 一条长 4,096 的序列只能产生 **4,095** 个「下一个字」的预测 ——&#160;
    最后那个 token 没有下一个可预测。⭐ 现场说的 4,096 差这一个。
  · 分母是**遮掉 padding / 拼接边界之后真正算了的目标位置数**，
    不是 B × S 硬乘。MaxText 里就是 `total_weights`。
  · `loss = xent_sum / total_weights` ——&#160;**token 平均，不是序列平均**。
  · 而反向**只跑一次**（`jax.value_and_grad` 作用在那个标量上）。
    ⭐ 它跟「每个 token 各跑一遍反向再把梯度平均」**结果完全一样**，
      因为求导是线性的；只是便宜了 N 倍。
"""
import math

from topic03_draw import Fig, BL, OR, GR, RD, PU, GY, INK, GY2

W = 1400

VOCAB = 129280          # DeepSeek-V3 词表
D_MODEL = 7168          # 隐藏维 ——&#160;一个位置往回传的那支「合力」就是这么长
SEQ = 4096
TGT = SEQ - 1           # 4,095 个「下一个字」
N1 = TGT * 1            # batch=1
N2 = TGT * 2            # batch=2

# 现场给的那组概率：正确 40%，其余 20 / 20 / 10 / 10
CANDS = (("的", 0.40, 1.0), ("了", 0.20, 0.0), ("在", 0.20, 0.0),
         ("和", 0.10, 0.0), ("有", 0.10, 0.0))

assert abs(sum(p for _, p, _ in CANDS) - 1.0) < 1e-9, \
    "这一格的说服力全靠「加起来正好是 1」——&#160;概率必须先自洽"
# ⭐ 这一格真正的论点：**梯度那一排加起来是 0**。断言钉住它。
assert abs(sum(p - y for _, p, y in CANDS)) < 1e-9, \
    "「预测 − 真值」这一排的和必须是 0 —— 它就是「再分配」那句话的全部证据"


def main():
    f = Fig(W, "一个位置上的梯度是「你猜的减去正确答案」，"
               "正确那一格是负数所以被推高，其余每一格都是正数所以被推低，"
               "推多少等于它现在占了多少概率；"
               "这一整排数加起来正好是零，所以推上去和推下去是同一个动作，"
               "是一次再分配而不是一次加分。"
               "一条四千零九十六长的序列有四千零九十五个这样的位置，"
               "每个位置都有一份完整的、整个词表那么长的意见；"
               "而一个位置往回传的并不是一个方向 —— "
               "词表十二万九千二百八十行各自拽着一个七千一百六十八维的向量，"
               "全部叠加之后才得到一支合力，图上那支箭画的是这个合力；"
               "batch 等于二就是八千一百九十份，"
               "每一份先除以八千一百九十，再全部累加进同一套参数梯度里 —— "
               "这就是平均发生的全部地方")

    y0 = f.header(
        "那 6,300 万个 loss，怎么变成<tspan font-weight=\"700\">一份</tspan>梯度",
        "⭐ 现场问：「错的那几个要不要推向 0，还是只推对的那个、"
        "其余算副作用？」——　<tspan font-weight=\"700\">答案是：都做，而且是同一个动作</tspan>",
        [(GR, "Ⓐ 一个位置"), (BL, "Ⓑ batch＝1"), (OR, "Ⓒ batch＝2")])

    # ══════════ Ⓐ 一个位置上，反向到底动了谁 ═══════════════════════
    PH = 452
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 先只看<tspan font-weight=\"700\">一个位置</tspan>"
                 "　——　这一格里，词表上的每一个字都拿到了自己的那一份", GR,
                 sub="⭐ 正确答案是「的」，网络给它打了 0.40；"
                     "剩下 0.60 分散在别的字上")

    BX, BY, BARW, BARH = 96, py + 60, 78, 132
    ZERO = BY + BARH                       # 概率 0 的基线
    for k, (ch, pred, want) in enumerate(CANDS):
        x = BX + k * 134
        g = pred - want                    # ⭐ 就是这一个数
        col = GR if g < 0 else RD
        top = ZERO - BARH * pred
        f.line(x - 8, ZERO, x + BARW + 8, ZERO, GY2, 1.1, arrow=False)
        f.box(x, top, BARW, BARH * pred, "#f1f3f4", GY2, 4)
        f.t(x + BARW / 2, top - 10, "%.2f" % pred, GY2, size=12, anchor="middle")
        f.t(x + BARW / 2, ZERO + 30, ch, INK, True, 20, "middle")
        # 箭头：负号往上（推高），正号往下（推低）。长度 ∝ |g|
        ay = ZERO + 58
        L = 46 * abs(g) / 0.60 + 14
        if g < 0:
            f.line(x + BARW / 2, ay + L, x + BARW / 2, ay, col, 3.0)
        else:
            f.line(x + BARW / 2, ay, x + BARW / 2, ay + L, col, 3.0)
        f.t(x + BARW / 2, ay + L + 26, "%+.2f" % g, col, True, 15, "middle")

    f.t(BX + 2.5 * 134 - 6, ZERO + 200,
        "灰柱 ＝ 网络现在给的概率　｜　"
        "<tspan font-weight=\"700\">下面那个数 ＝ 这一格的梯度 ＝ 你猜的 −　正确答案</tspan>",
        GY2, size=13, anchor="middle")
    f.t(BX + 2.5 * 134 - 6, ZERO + 226,
        "<tspan fill=\"%s\" font-weight=\"700\">负号 ＝ 把这一格推高</tspan>　·　"
        "<tspan fill=\"%s\" font-weight=\"700\">正号 ＝ 把这一格推低</tspan>"
        % (GR, RD), GY, size=13, anchor="middle")

    f.box(800, py + 54, 560, 316, "#e6f4ea", GR, 8)
    f.t(1080, py + 92, "所以那一问的答案是：", GR, True, 18, "middle")
    f.t(1080, py + 128, "<tspan font-weight=\"700\">都做，而且是<tspan "
        "text-decoration=\"underline\">同一个</tspan>动作</tspan>",
        INK, True, 20, "middle")
    f.t(1080, py + 174,
        "错的那几个<tspan font-weight=\"700\">不是副作用</tspan>　——　"
        "每一个都有自己那一份，", INK, size=14.5, anchor="middle")
    f.t(1080, py + 200,
        "而且<tspan font-weight=\"700\">力度正好等于它现在占了多少</tspan>："
        "0.20 那个被推的劲，", INK, size=14.5, anchor="middle")
    f.t(1080, py + 226,
        "是 0.10 那个的<tspan font-weight=\"700\">两倍</tspan>。",
        INK, size=14.5, anchor="middle")
    f.t(1080, py + 268,
        "⭐⭐ 而这一排数<tspan font-weight=\"700\">加起来正好是 0</tspan>"
        "（−0.60 ＋ 0.20 ＋ 0.20 ＋ 0.10 ＋ 0.10）", OR, True, 14.5, "middle")
    f.t(1080, py + 296,
        "概率总量是守恒的 ——　所以这<tspan font-weight=\"700\">不是一次加分，"
        "</tspan>", INK, size=14.5, anchor="middle")
    f.t(1080, py + 322,
        "<tspan font-weight=\"700\">是一次再分配。</tspan>",
        INK, True, 16, "middle")
    f.t(1080, py + 352,
        "⛔ 剩下那 %s 个字概率≈0，梯度也≈0　——　它们本来就没占地方"
        % format(VOCAB - len(CANDS), ","), GY2, size=12.5, anchor="middle")
    f._pan = None

    # ══════════ Ⓑ batch = 1 ════════════════════════════════════════
    # ⛔⛔ 2026-09-23 现场：「这个图里一个小方块其实是 7,168 个方向的目标，
    #   它不是一个方向，应该是一个合力的箭头。这个图画得不对。」——&#160;对。
    #   ⭐ 而且毛病比「少画了几根」更具体：原来这排箭头**照抄了 Ⓐ 的语义**，
    #     绿色＝推高、红色＝推低，还按 k%4 随机上色。
    #     可「推高 / 推低」是**词表某一格**才有的说法；
    #     到了「一个位置」这一层，那份意见已经是 129,280 个数了，
    #     它往回传的是一个 D_MODEL 维的**合力**，没有单一的「往哪边推」。
    #   ⛔ 判据：**放大一级之后，上一级的语义不会自动跟着上来。**
    #     颜色是从 Ⓐ 继承的，而继承下来的那一刻它就已经没有所指了 ——&#160;
    #     它不报错，只是让人读出一个不存在的意思。
    PH2 = 502
    py2 = f.panel(0, py + PH + 20, W, PH2,
                  "Ⓑ 再放大到<tspan font-weight=\"700\">一整条序列</tspan>"
                  "（batch ＝ 1）　——　上面那一整套，每个位置都来一遍", BL,
                  sub="⛔ 一条长 %s 的序列只有 <tspan font-weight=\"700\">%s</tspan> "
                      "个预测 ——　最后那个字没有「下一个」；"
                      "而下面每个位置底下那支箭，是<tspan font-weight=\"700\">"
                      "一束，不是一根</tspan>" % (format(SEQ, ","), format(TGT, ",")))

    # 一排缩略的位置 ——&#160;每个位置底下画的是「一束的合力」，不是一根
    for k in range(26):
        x = 70 + k * 40
        h = 10 + (k * 7 % 23)
        cx = x + 11
        f.box(x, py2 + 108 - h, 22, h, "#f1f3f4", GY2, 2)
        for dx in (-9, -5, 5, 9):           # 分量：淡、细、无箭头、明显岔开
            f.line(cx, py2 + 116, cx + dx, py2 + 129, "#c9ccd1", 0.9, arrow=False)
        f.line(cx, py2 + 116, cx, py2 + 140, BL, 2.4)      # 合力：粗、有箭头
    f.t(70 + 26 * 40 + 34, py2 + 126, "……", GY2, True, 20)
    f.line(70, py2 + 162, 70 + 25 * 40 + 22, py2 + 162, BL, 1.4, arrow=False)
    f.t((70 + 70 + 25 * 40 + 22) / 2, py2 + 186,
        "<tspan font-weight=\"700\">%s 个位置</tspan>，每个位置都有一份"
        "<tspan font-weight=\"700\">整个词表那么长</tspan>的意见（%s 个数）"
        % (format(TGT, ","), format(VOCAB, ",")), INK, size=14.5, anchor="middle")

    # ── 把一个位置放大：那支箭是 VOCAB 行叠出来的一个 D_MODEL 维合力 ──
    f.box(60, py2 + 208, 1040, 218, "#f8f9fa", GY2, 8)
    f.t(84, py2 + 238,
        "⛔ <tspan font-weight=\"700\">那支箭不是「这个位置往哪边推」</tspan>"
        "　——　到了这一层，已经没有单一的方向了", RD, size=14.5)

    ox, oy = 258, py2 + 288                  # 扇形的出发点
    for i in range(9):
        a = math.radians(-62 + i * 15.5)
        f.line(ox, oy, ox + 92 * math.sin(a), oy + 92 * math.cos(a),
               GY2, 1.0, arrow=False)
    f.t(ox, py2 + 272, "一个位置的那份意见", INK, True, 15, "middle")
    f.t(ox, py2 + 404,
        "词表 <tspan font-weight=\"700\">%s</tspan> 行，"
        "<tspan font-weight=\"700\">每一行都往自己那边拉一把</tspan>"
        % format(VOCAB, ","), GY, size=13.5, anchor="middle")

    f.line(452, py2 + 318, 540, py2 + 318, OR, 2.4)
    f.t(496, py2 + 302, "叠起来", OR, True, 14, "middle")

    f.line(650, py2 + 258, 650, py2 + 378, BL, 4.4)
    f.t(650, py2 + 404,
        "<tspan font-weight=\"700\">一支合力：%s 维</tspan>" % format(D_MODEL, ","),
        BL, True, 15, "middle")

    f.t(772, py2 + 288,
        "所以「一个方向」这个说法是错的：", INK, True, 14.5)
    f.t(772, py2 + 316,
        "词表上<tspan font-weight=\"700\">每一行</tspan>都拽着一个 %s 维的向量，"
        % format(D_MODEL, ","), GY, size=13.5)
    f.t(772, py2 + 340,
        "<tspan font-weight=\"700\">%s 行全部叠加</tspan>，才得到这一支箭。"
        % format(VOCAB, ","), GY, size=13.5)
    f.t(772, py2 + 372,
        "⭐ 它到底怎么叠出来的 ——　<tspan font-weight=\"700\">下一节专讲</tspan>。",
        OR, True, 13.5)

    f.box(1130, py2 + 52, 232, 186, "#e8f0fe", BL, 8)
    f.t(1246, py2 + 88, "batch ＝ 1", BL, True, 17, "middle")
    f.t(1246, py2 + 122, "一共", GY, size=13, anchor="middle")
    f.t(1246, py2 + 152, "<tspan font-weight=\"700\">%s</tspan>"
        % format(TGT * VOCAB, ","), INK, True, 19, "middle")
    f.t(1246, py2 + 178, "个数当种子", GY, size=13, anchor="middle")
    f.t(1246, py2 + 214, "每一份都<tspan font-weight=\"700\">÷ %s</tspan>"
        % format(N1, ","), OR, True, 14, "middle")
    f._pan = None

    # ══════════ Ⓒ batch = 2 ════════════════════════════════════════
    PH3 = 346
    py3 = f.panel(0, py2 + PH2 + 20, W, PH3,
                  "Ⓒ batch ＝ 2　——　<tspan font-weight=\"700\">"
                  "唯一的变化是分母</tspan>，从 %s 变成 %s"
                  % (format(N1, ","), format(N2, ",")), OR,
                  sub="⭐ 「平均」发生的全部地方就在这个除法上，"
                      "而且它在<tspan font-weight=\"700\">种子那一步就做掉了</tspan>")

    for r in range(2):
        yy = py3 + 62 + r * 62
        f.t(56, yy + 16, "序列 %d" % (r + 1), (BL, PU)[r], True, 15)
        for k in range(18):
            x = 150 + k * 34
            f.box(x, yy, 20, 24, "#f1f3f4", GY2, 2)
        f.t(150 + 18 * 34 + 12, yy + 18, "…… 共 %s 个位置" % format(TGT, ","),
            GY2, size=13)

    f.line(700, py3 + 196, 700, py3 + 232, OR, 2.4)
    f.t(700, py3 + 258,
        "<tspan font-weight=\"700\">%s 份意见，每一份先 ÷ %s，"
        "再全部累加进同一套 671B 的参数梯度</tspan>"
        % (format(N2, ","), format(N2, ",")), INK, size=15.5, anchor="middle")
    f.t(700, py3 + 288,
        "⭐⭐ 所以「每个 token 各跑一遍反向、再把 671B 个梯度平均」"
        "和「先把 loss 平均、只跑一次反向」——　", GY, size=13.5, anchor="middle")
    f.t(700, py3 + 312,
        "<tspan font-weight=\"700\">算出来完全一样</tspan>"
        "（求导是线性的），只是后者便宜 %s 倍。"
        % format(N2, ","), GY, size=13.5, anchor="middle")
    f._pan = None

    # ⛔ 2026-09-22：`band()` 三行起会把第三行折进「出处」——&#160;
    #   第一版三行，而被折走的恰好是**直接回答现场那一问的那一行**。
    #   ⭐ 判据：**落点带只有两行的额度，那两行要自己挑，不能交给截断去挑。**
    yb = f.band(py3 + PH3 + 18, "ok", "⭐ 一句话记住这张图", [
        "<tspan font-weight=\"700\">「把对的推上去」和「把错的压下去」是同一个动作</tspan>"
        "　——　那一排数加起来正好是 0，它是一次<tspan font-weight=\"700\">再分配</tspan>，"
        "错的那些不是副作用。",
        "<tspan font-weight=\"700\">loss 按 token 平均（不是按序列），"
        "但反向并不从那个标量出发</tspan>　——　"
        "它从一个「每个位置 × 整个词表」的张量出发，每个数都是「你猜的 −　正确答案」÷ 总数"
        "；而<tspan font-weight=\"700\">一个位置往回传的是一支合力</tspan>"
        "（%s 行各拽一个 %s 维向量叠出来的），<tspan font-weight=\"700\">不是一个方向</tspan>。"
        % (format(VOCAB, ","), format(D_MODEL, ",")),
    ])

    yb = f.src(yb + 10,
               "口径来自 MaxText 训练脚本："
               "<tspan font-weight=\"700\">loss ＝ xent_sum ÷ total_weights</tspan>，"
               "其中 total_weights 是遮掉 padding 与拼接边界之后真正算了的目标位置数；",
               "梯度只跑一次（value_and_grad 作用在那个标量上）。"
               "词表 %s、序列 %s 取自 DeepSeek-V3 预训练配置。"
               % (format(VOCAB, ","), format(SEQ, ",")))

    f.save("fig4-push.svg", yb + 14)


main()
