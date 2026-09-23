# -*- coding: utf-8 -*-
r"""专题四 · §1.1「导数 / 偏导数 / 链式法则」——&#160;用一句话说完：兑换率

⭐⭐⭐ 2026-09-17 新画。现场原话：「反向传播、梯度下降、微积分、偏导数
  这些都给我讲得明明白白的，到底这个 loss 它怎么变成梯度，怎么更新 ……画图讲」。
  ⛔ 清点下来发现一个洞：**这一讲从没说过「导数是什么」。**
    §1.1 一句「求导是高中的事」就过去了，§1.4 讲的是
    「偏导数**为什么**让事情变容易」——&#160;那是另一个问题。
    ⭐ 对没修过微积分的人，这就是第一道坎，而且过不去后面全是空的。

⭐⭐ 这张图不讲极限、不讲 ε-δ，只讲一件事：**导数是一个兑换率。**
  · **导数** ——&#160;这个旋钮<tspan>拧一点点</tspan>，结果变多少。
    ⛔ 注意它**不是**「结果是多少」，是「你动一格，它动几格」。
  · **偏导数** ——&#160;其余 6,710 亿个**按住不动**，只拧这一个时的兑换率。
    「偏」字的全部含义就是那三个字：**按住不动**。
  · **链式法则** ——&#160;中间隔着好几级，**每级一个兑换率，一路乘起来**。
    ⭐⭐⭐ 就是**换汇**：人民币→港币→美元，每步一个汇率，总汇率是乘出来的。

⭐⭐⭐ 「兑换率」这个说法不是为了好听，它**自带两个后续的钩子**：
  ① 接 §1.5 ——&#160;一串数相乘，**从哪一头开始乘**结果一样、代价差两万年。
  ② 接 §3.3 ——&#160;正文那个「折算系数 / 量纲」问题，
     用兑换率讲就是一句话：**梯度的单位是「loss 每参数」，
     而你要的是「参数」，所以中间必须再乘一个东西。**
  ⛔ 这两个钩子都是本讲自己的，不是从哪儿抄的。

⚠️ 图上那几个倍数（×2 / ×0.5 / ×3）是**编出来的示意数**，
  它们唯一的作用是让「乘起来」这件事看得见 ——&#160;脚本里 assert 了乘积。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE)

W = 1400

# ⭐ Ⓐ 的小例子：推子动 0.01，结果动 0.03 → 兑换率 3
D_KNOB, D_OUT = 0.01, 0.03
# ⛔⛔ 2026-09-23 现场：「这个地方的 5.0 不对，这哪来的 5.0？」
#   ⭐ 查下来不是数值错，是**孤儿引用**：2026-09-18 把这一格从「纯文字框」
#     改画成两把竖尺之后，**图上再没有任何绝对读数**，
#     而正文、落点带、出处里那三句「读数是 …」原封不动留着。
#     读者在图上找不到那个数，只能问「这哪来的」。
#   ⭐⭐ 判据：**改画法的时候，去搜一遍旧画法里那些数还被谁引着。**
#     图自己不会报错 ——&#160;它只是不再包含那个数了。
#   ⚠️ 两处一起改：① 把读数**画回图上**；② 换成 4.00 ——&#160;
#     4.0 是 §1.0 那把尺子上**真有的一档**（尺子是 11.77 / 6 / 4 / 3 / 2 / 1）。
#     原来那个数其实也合法（落在 4 和 6 之间），但既然是随手取的，
#     就取一个读者能在表里直接指出来的。
L_BEFORE = 4.00                   # loss 读数（§1.0 尺子上「语法大致对了」那一档）
L_AFTER = L_BEFORE + D_OUT
assert abs(L_BEFORE - 4.00) < 1e-9, "这个读数要跟 §1.0 那把尺子上的某一档对上"
RATE_A = D_OUT / D_KNOB
assert abs(RATE_A - 3.0) < 1e-12

# ⭐ Ⓒ 的链条：三级，每级一个兑换率
CHAIN = (("推子", "中间量甲", 2.0),
         ("中间量甲", "中间量乙", 0.5),
         ("中间量乙", "难听程度", 3.0))
TOTAL = 1.0
for _, _, r in CHAIN:
    TOTAL *= r
assert abs(TOTAL - 3.0) < 1e-12, "总兑换率必须是三个乘起来，不是我写上去的"
# ⛔ 顺手挑一组「不都大于 1」的数 ——&#160;不然读者会以为链式法则只会放大
assert any(r < 1.0 for _, _, r in CHAIN), \
    "链条里要有一级是缩小的，否则「乘起来」会被读成「越乘越大」"


def main():
    f = Fig(W, "导数说白了就是一个兑换率：这个旋钮拧一点点，结果变多少。"
               "注意它不是结果是多少，是你动一格它动几格。"
               "偏导数就是把其余所有旋钮按住不动，只拧这一个时的兑换率，"
               "偏字的全部含义就是按住不动这三个字。"
               "而把 6,710 亿个旋钮各自的兑换率排成一列，那一列就叫梯度。"
               "最后，旋钮和结果之间隔着很多级，每一级有自己的兑换率，"
               "总的兑换率就是把它们一路乘起来 —— 这就是链式法则，"
               "跟人民币换港币再换美元是同一回事")

    y0 = f.header(
        "导数、偏导数、链式法则　——　<tspan font-weight=\"700\">"
        "其实是同一个词：兑换率</tspan>",
        "⛔ 这一格<tspan font-weight=\"700\">不讲极限、不讲公式</tspan>"
        "　——　只讲这三个词到底在说什么事",
        [(BL, "导数：动一格，变几格"), (GR, "偏导数：其余按住"),
         (PU, "链式法则：一路乘")])

    # ══════════ Ⓐ 导数 ＝ 动一格，它动几格 ═════════════════════════
    PH = 348
    py = f.panel(0, y0, W, PH,
                 "Ⓐ <tspan font-weight=\"700\">导数</tspan>问的不是"
                 "「现在是多少」，是<tspan font-weight=\"700\">「你动一格，它动几格」</tspan>",
                 BL,
                 sub="⭐ 把参数想成一个旋钮，loss 是它右边那个读数")

    # 旋钮：一根竖槽 ＋ 一个把手
    KX, KY, KH = 220, py + 66, 190
    f.box(KX - 4, KY, 8, KH, "#f1f3f4", GY2, 4)
    f.box(KX - 34, KY + 118, 68, 22, "#fff", BL, 5, sw=1.8)
    f.t(KX, KY + KH + 30, "一个参数", INK, True, 15, "middle")
    f.t(KX, KY + KH + 52, "（ 6,710 亿个之一）", GY2, size=12, anchor="middle")
    # 往上推一点点
    f.line(KX + 54, KY + 124, KX + 54, KY + 96, BL, 2.2)
    f.t(KX + 66, KY + 116, "往上推一点点", BL, True, 13.5)
    f.t(KX + 66, KY + 138, "＋%.2f" % D_KNOB, GY, size=12.5)

    f.t(430, KY + 116, "→", GY2, True, 26, "middle")

    # ⭐⭐⭐ 2026-09-18 加料。这儿原来是个纯文字框（5.00 ↓ 5.03），ink 只有 1。
    #   ⛔ 而「动一格，它动三格」这句话**本身就是一个长度比** ——&#160;
    #     长度比就该用两段真的长度画出来，不是写两个数让读者去减。
    #   ⛔ 先试过李宏毅那张「小人站曲线上、脚下一条切线」的装置（见素材库 🅑①），
    #     放弃的原因很具体：**斜率 3 要求纵轴的像素比例尺是横轴的 3 倍**，
    #     否则读者用眼睛量出来是 0.63 不是 3 ——&#160;而面板是宽扁的，放不下。
    #     ⭐⭐ 判据：**一张请读者「用眼睛量比例」的图，两个轴的比例尺必须相同；
    #       做不到就别用坐标系，换一个不需要两个轴的画法。**
    #   ⭐ 于是换成两把**共用同一种格子**的竖尺：一格都代表 0.01，
    #     参数那把走 1 格，loss 那把走 3 格 ——&#160;比例就是格数，不用换算。
    GRID = 24.0
    NG = 8                            # 尺子一共几格
    UNIT = D_KNOB                     # 两把尺的一格都是这个
    SX_P, SX_L = 560, 742
    STOP = py + 72
    N_P = int(round(D_KNOB / UNIT))   # 参数走几格
    N_L = int(round(D_OUT / UNIT))    # loss 走几格
    # ⭐⭐ 这一格的立论：两段粗条的**像素长度之比**必须正好是那个兑换率
    assert abs((N_L * GRID) / (N_P * GRID) - RATE_A) < 1e-9, \
        "两根条的长度比对不上导数 —— 那这张图就在撒谎"
    assert N_L <= NG and N_P >= 1, "格数超出尺子范围了"

    def ruler(x, name, col, n_move, y_end_grid):
        """一把十格的竖尺。⭐ 两把用同一个 GRID，「1 格 vs 3 格」才可比。"""
        f.box(x - 5, STOP, 10, NG * GRID, "#f1f3f4", GY2, 5)
        for i in range(NG + 1):
            f.line(x - 15, STOP + i * GRID, x - 7, STOP + i * GRID,
                   GY2, 0.9, arrow=False)
        # ⛔ 终点定在从下数第 k 格，起点就在它下方 n 格 ——&#160;
        #   所以必须 k ≥ n，否则起点会**跑出尺子底部**。
        #   ⭐ 第一版把两根都定在第 2 格，红条走 3 格，直接戳穿了尺底。
        #     判据：**凡是「从某点往回退 n 步」的画法，都要先问退得出去吗。**
        assert y_end_grid >= n_move, \
            "终点太靠下，往回退 %d 格会戳出尺子" % n_move
        y_end = STOP + (NG - y_end_grid) * GRID
        y_beg = y_end + n_move * GRID
        # 起点空心、终点实心 ——&#160;中间那根粗条就是「动了多少」
        f.box(x - 17, y_beg - 2.5, 34, 5, "#fff", GY2, 2)
        f.box(x - 7, y_end, 14, n_move * GRID, col, col, 3)
        f.box(x - 20, y_end - 3, 40, 6, col, col, 3)
        f.t(x, STOP - 14, name, col, True, 14.5, "middle")
        # ⚠️ 一格那根太短，标签贴上去会跟「终点同高」那条虚线挤在一起 ——
        #    短的挪到条的**下方**，长的才放右侧正中
        if n_move <= 1:
            f.t(x + 20, y_beg + 20, "%d 格" % n_move, col, True, 14.5)
        else:
            f.t(x + 26, y_end + n_move * GRID / 2.0 + 5,
                "%d 格" % n_move, col, True, 14.5)
        return y_end

    END_G = 4                         # 两根条的终点都落在从下数第 4 格
    ruler(SX_P, "这个参数", BL, N_P, END_G)
    ruler(SX_L, "loss", RD, N_L, END_G)
    # 两把尺的终点画在同一高度，眼睛才好比那两根条
    f.line(SX_P + 20, STOP + (NG - END_G) * GRID, SX_L - 22, STOP + (NG - END_G) * GRID,
           GY2, 0.9, dash="4 4", arrow=False)
    f.t((SX_P + SX_L) / 2.0, STOP - 34,
        "两把尺<tspan font-weight=\"700\">一格都是 %.2f</tspan>" % UNIT,
        GY2, size=12.5, anchor="middle")

    f.box(850, py + 62, 500, 208, "#e8f0fe", BL, 8)
    f.t(1100, py + 96, "那这个旋钮的<tspan font-weight=\"700\">导数</tspan>就是",
        BL, True, 17, "middle")
    # ⭐ 把那个绝对读数**画在图上**：正文要引它，图上就得有它。
    f.t(1100, py + 126,
        "loss 读数 %.2f →　%.2f　（动了 %.2f）" % (L_BEFORE, L_AFTER, D_OUT),
        GY2, size=12.5, anchor="middle")
    f.t(1100, py + 170, "%.2f ÷ %.2f ＝ <tspan font-weight=\"700\">%d</tspan>"
        % (D_OUT, D_KNOB, int(RATE_A)), INK, True, 22, "middle")
    f.t(1100, py + 212, "⭐ 读作：<tspan font-weight=\"700\">你动一格，它动三格</tspan>",
        BL, True, 15, "middle")
    f.t(1100, py + 246, "⛔ 它<tspan font-weight=\"700\">不是</tspan>「读数是 %.2f」"
        "　——　那是<tspan font-weight=\"700\">值</tspan>，这是<tspan font-weight=\"700\">兑换率</tspan>"
        % L_BEFORE, GY, size=13, anchor="middle")

    f.t(700, py + 306,
        "⭐⭐ 顺带记住它的<tspan font-weight=\"700\">单位</tspan>："
        "<tspan font-weight=\"700\">loss 每参数</tspan>　——　"
        "后面讲<tspan font-weight=\"700\">「那一步该迈多大」</tspan>那一节，整节都是被这个单位逼出来的。",
        GY, size=13.5, anchor="middle")
    f._pan = None

    # ══════════ Ⓑ 「偏」字 ＝ 其余全按住 ════════════════════════════
    PH2 = 308
    py2 = f.panel(0, py + PH + 20, W, PH2,
                  "Ⓑ ⭐⭐⭐ 那<tspan font-weight=\"700\">「偏」</tspan>字是什么意思"
                  "　——　就三个字：<tspan font-weight=\"700\">按住不动</tspan>", GR,
                  sub="⭐ 一台 6,710 亿个旋钮的调音台，一次只拧一个，其余全按住")

    NK, SX0, SGAP = 9, 150, 108
    for k in range(NK):
        x = SX0 + k * SGAP
        live = (k == 4)
        col = GR if live else GY2
        f.box(x - 3, py2 + 48, 6, 120, "#f1f3f4", GY2, 3)
        f.box(x - 26, py2 + (74 if live else 108), 52, 18,
              "#fff", col, 4, sw=1.8 if live else 1.0)
        if live:
            f.line(x + 42, py2 + 116, x + 42, py2 + 84, GR, 2.2)
            f.t(x, py2 + 196, "只动这一个", GR, True, 14, "middle")
        else:
            f.t(x, py2 + 196, "按住", GY2, size=12, anchor="middle")
    f.t(SX0 + NK * SGAP + 10, py2 + 120, "……　其余 6,710 亿个，全按住",
        GY2, size=13)

    f.box(150, py2 + 224, 1200, 58, "#e6f4ea", GR, 8)
    f.t(750, py2 + 260,
        "⭐⭐⭐ <tspan font-weight=\"700\">偏导数 ＝ 其余全按住时，这一个旋钮的兑换率。</tspan>"
        "　而把 6,710 亿个旋钮各自的那个数排成一列 ——&#160;"
        "<tspan font-weight=\"700\">那一列就叫「梯度」。</tspan>",
        INK, True, 15.5, "middle")
    f._pan = None

    # ══════════ Ⓒ 链式法则 ＝ 换汇 ═════════════════════════════════
    PH3 = 380
    py3 = f.panel(0, py2 + PH2 + 20, W, PH3,
                  "Ⓒ ⭐⭐ 可旋钮<tspan font-weight=\"700\">不直接连到读数</tspan>"
                  "　——　中间隔着好几级", PU,
                  sub="⭐ 每一级有自己的兑换率，"
                      "<tspan font-weight=\"700\">总兑换率就是一路乘起来</tspan>")

    # ⛔⛔ 2026-09-22 现场：「你这图中间那个 2.0、0.5、3.0，它光画一条线，
    #   这个表达得不清楚吧？它中间应该是一个参数，至少应该是一个标量，对吧？
    #   但是实际上它是一个矩阵才能产生这个放大和缩小的作用，光画一条线有什么用？」
    # ⭐ 说得对：原来那条**光秃秃的箭头**把「产生这个数的东西」整个藏起来了，
    #   于是 2.0 看着像凭空写上去的。⭐ 判据：**一条边上标着一个数的时候，
    #   图必须同时回答「这个数是谁产生的」** ——&#160;否则那个数就是魔法。
    # ⇒ 每一级中间放一个**装置**：一块权重 ＋ 一个激活函数。
    #   而那个兑换率是**装置的斜率**，不是装置本身。
    BX, BW2, BGAP = 34, 216, 146
    NODES = ("推子", "中间量甲", "中间量乙", "难听程度")
    for k, name in enumerate(NODES):
        x = BX + k * (BW2 + BGAP)
        col = PU if k in (0, len(NODES) - 1) else GY2
        f.box(x, py3 + 62, BW2, 74, "#fff", col, 8, sw=1.6)
        f.t(x + BW2 / 2, py3 + 106, name, INK if k else PU, True, 16, "middle")
        if k < len(NODES) - 1:
            gx = x + BW2
            f.line(gx + 4, py3 + 99, gx + 18, py3 + 99, PU, 2.0)
            f.box(gx + 22, py3 + 70, BGAP - 44, 58, "#fef7e0", OR, 6, sw=1.4)
            f.t(gx + 22 + (BGAP - 44) / 2, py3 + 90, "× 权重", OR, True, 12, "middle")
            f.t(gx + 22 + (BGAP - 44) / 2, py3 + 114, "过激活", OR, True, 12, "middle")
            f.line(gx + BGAP - 18, py3 + 99, gx + BGAP - 4, py3 + 99, PU, 2.0)
            f.t(gx + BGAP / 2, py3 + 50, "斜率 %.1f" % CHAIN[k][2],
                PU, True, 15, "middle")
    f.t(W / 2, py3 + 160,
        "⭐ 橙色那个小方块才是<tspan font-weight=\"700\">装置</tspan>；"
        "上面那个数是<tspan font-weight=\"700\">装置的斜率</tspan>，不是装置本身　——　"
        "它 ＝ <tspan font-weight=\"700\">权重 × 激活函数在当前这一点的斜率</tspan>。",
        INK, size=14, anchor="middle")

    f.box(80, py3 + 190, 620, 116, "#f3e8fd", PU, 8)
    f.t(390, py3 + 226, "总兑换率 ＝ 一路乘起来", PU, True, 17, "middle")
    f.t(390, py3 + 268, "%.1f × %.1f × %.1f ＝ <tspan font-weight=\"700\">%d</tspan>"
        % (CHAIN[0][2], CHAIN[1][2], CHAIN[2][2], int(TOTAL)),
        INK, True, 22, "middle")

    f.box(740, py3 + 190, 610, 116, "#e8f0fe", BL, 8)
    f.t(1045, py3 + 226, "⭐ 这就是<tspan font-weight=\"700\">换汇</tspan>",
        BL, True, 17, "middle")
    f.t(1045, py3 + 260, "人民币 → 港币 → 美元，每步一个汇率",
        INK, size=14.5, anchor="middle")
    f.t(1045, py3 + 286, "总汇率<tspan font-weight=\"700\">当然是乘出来的</tspan>"
        "　——　链式法则就这一件事", GY, size=13.5, anchor="middle")

    f.t(700, py3 + 344,
        "⛔ 而「<tspan font-weight=\"700\">这一串数从哪一头开始乘</tspan>」"
        "　——　结果完全一样，<tspan font-weight=\"700\">代价差两万年</tspan>。"
        "那是紧接着<tspan font-weight=\"700\">后面那张「正向 vs 反向」图</tspan>的事。",
        INK, size=14.5, anchor="middle")
    f._pan = None

    # ══════════ Ⓓ 真实的一级：那个「斜率」是一张表 ═══════════════════
    # ⭐⭐⭐ 2026-09-22 现场那一问的后半句，也是更要紧的那半句：
    #   「实际上它是一个矩阵才能产生这个放大和缩小的作用。」——&#160;完全正确。
    #   ⛔ 前面三格是**一维玩具**（一个推子、一个读数），玩具里斜率确实是个标量；
    #     可真实的一级是「七千多维进、七千多维出」，它的斜率是**一整张表**。
    #   ⭐ 而这一格最该留下的不是那张表有多大，是下面这句：
    #     **我们从来不把它算出来。** 反向只做「拿责任去乘它」这一件事。
    DJ = 7168
    PH4 = 320
    py4 = f.panel(0, py3 + PH3 + 20, W, PH4,
                  "Ⓓ 可真实的一级<tspan font-weight=\"700\">不是一个标量</tspan>"
                  "　——　它的「斜率」是一整张表", RD,
                  sub="⭐ 上面三格是<tspan font-weight=\"700\">一维玩具</tspan>："
                      "一个推子、一个读数。真实的一级是"
                      "<tspan font-weight=\"700\">%s 维进、%s 维出</tspan>"
                      % (format(DJ, ","), format(DJ, ",")))

    f.box(60, py4 + 70, 190, 72, "#e8f0fe", BL, 6)
    f.t(155, py4 + 100, "进来", BL, True, 14, "middle")
    f.t(155, py4 + 126, "%s 个数" % format(DJ, ","), INK, True, 15, "middle")
    f.line(256, py4 + 106, 288, py4 + 106, PU, 2.2)
    # ⛔⛔ 2026-09-22 现场：「不要写 7168×7168，这个虽然是 Input 7168 和
    #   Output 7168，但是里边**从来没有出现过这么样一个方阵**。」——&#160;对。
    #   ⭐ V3 一层里**没有任何一块权重是方阵**：专家 7,168 → 2,048、
    #     MLA 把 KV 压到 512 那一档，而 o_proj 那一边反而更宽（16,384）。
    #     ⛔ 所以别说成「全是瘦的」——&#160;准确的只有「没有一块两边都是 7,168」。
    #     原来那么写，是我顺手拿「进 7168、出 7168」平方了 ——&#160;
    #     正是第一原则点名的那类「听起来像常识的架构关系」。
    f.box(294, py4 + 62, 200, 88, "#fef7e0", OR, 6, sw=1.6)
    f.t(394, py4 + 90, "装置", OR, True, 15, "middle")
    f.t(394, py4 + 114, "几块权重 ＋ 激活", INK, True, 13, "middle")
    f.t(394, py4 + 136, "（7,168→2,048 这种，<tspan font-weight=\"700\">都不是方阵</tspan>）",
        GY, size=12, anchor="middle")
    f.line(500, py4 + 106, 532, py4 + 106, PU, 2.2)
    f.box(538, py4 + 70, 190, 72, "#e8f0fe", BL, 6)
    f.t(633, py4 + 100, "出去", BL, True, 14, "middle")
    f.t(633, py4 + 126, "%s 个数" % format(DJ, ","), INK, True, 15, "middle")

    f.t(394, py4 + 186,
        "那这一级的「斜率」是多少？　——　<tspan font-weight=\"700\">不是一个数</tspan>："
        "出去的每一维，", RD, True, 14.5, "middle")
    f.t(394, py4 + 212,
        "对进来的每一维，<tspan font-weight=\"700\">都各有一个兑换率</tspan>。",
        RD, True, 14.5, "middle")
    f.t(394, py4 + 246,
        "⛔ 但注意：<tspan font-weight=\"700\">机器里并没有这样一张方阵</tspan>。",
        INK, True, 14, "middle")
    f.t(394, py4 + 270,
        "真实的权重<tspan font-weight=\"700\">没有一块是方阵</tspan>，"
        "那张表只是<tspan font-weight=\"700\">概念上</tspan>的。",
        GY, size=13, anchor="middle")

    f.box(770, py4 + 58, 580, 210, "#e6f4ea", GR, 8)
    f.t(1060, py4 + 94, "⭐ 但好消息是：", GR, True, 17, "middle")
    f.t(1060, py4 + 130,
        "<tspan font-weight=\"700\">我们从来不把那张表算出来。</tspan>",
        INK, True, 17, "middle")
    f.t(1060, py4 + 170,
        "反向只做一件事：<tspan font-weight=\"700\">拿责任去乘它</tspan>，",
        INK, size=14.5, anchor="middle")
    f.t(1060, py4 + 196,
        "直接得到<tspan font-weight=\"700\">上一级该收到的责任</tspan>。",
        INK, size=14.5, anchor="middle")
    f.t(1060, py4 + 234,
        "——　那正是「反向那两笔乘法」里的<tspan font-weight=\"700\">第二笔</tspan>。",
        GR, True, 14, "middle")
    f.t(1060, py4 + 260,
        "⭐ 所以「一路乘起来」这句话是真的，只是乘的是<tspan font-weight=\"700\">表</tspan>。",
        GY, size=13.5, anchor="middle")
    f._pan = None

    yb = f.band(py4 + PH4 + 20, "ok",
                "所以这一讲后面所有的东西，都只用到这三句话",
                ("✅ <tspan font-weight=\"700\">① 导数 ＝ 你动一格，它动几格（兑换率）。"
                 "② 偏导数 ＝ 其余全按住时的那个兑换率。"
                 "③ 链式法则 ＝ 中间每一级的兑换率，一路乘起来。</tspan>"
                 "——&#160;没有极限，没有 ε，没有要背的公式。",
                 "⛔ 唯一一个<tspan font-weight=\"700\">真的要小心</tspan>的点："
                 "兑换率<tspan font-weight=\"700\">不是「值」</tspan>。"
                 "读数是 4.00 跟「动一格变三格」是两件完全不同的事 ——&#160;"
                 "本讲后面每次说「梯度大」，说的都是<tspan font-weight=\"700\">后者</tspan>。",
                 "⭐⭐ 而 Ⓐ 末尾那个<tspan font-weight=\"700\">单位</tspan>是给 §3.3 埋的："
                 "梯度的单位是「loss 每参数」，可你要的是「参数该挪多少」——&#160;"
                 "<tspan font-weight=\"700\">两边对不上，中间必须再乘一个东西，那个东西就是学习率。</tspan>"))

    yb = f.src(yb + 16,
               "⚠️ Ⓐ 的 4.00 → 4.03、Ⓒ 的 ×2 / ×0.5 / ×3 都是"
               "<tspan font-weight=\"700\">编出来的示意数</tspan>，"
               "唯一的作用是让「相除」和「相乘」这两件事看得见。"
               "⭐ 脚本里 assert 了两条：总兑换率必须真的是三个乘积，"
               "而且<tspan font-weight=\"700\">链条里要有一级是缩小的</tspan> ——&#160;"
               "不然「乘起来」会被读成「越乘越大」，"
               "而那正是梯度消失/爆炸那一节要讲的反面。",
               "⭐ 这一格<tspan font-weight=\"700\">刻意不碰极限</tspan>："
               "严格地说导数是「动的那一点点趋于 0 时的极限」，"
               "而这张图画的是<tspan font-weight=\"700\">差商</tspan>。"
               "⛔ 对本讲够用 ——&#160;而且 §1.1 那个「笨办法」用的**正是差商**，"
               "所以这里不严格反而接得更顺。",
               "⭐⭐ 「兑换率」这个说法不是为了好听，它自带两个钩子，"
               "两个都是本讲自己的：<tspan font-weight=\"700\">一串数相乘，"
               "从哪头开始乘代价差两万年（§1.5）；"
               "而单位对不上所以必须再乘一个折算系数（§3.3）。</tspan>")

    f.save("fig4-slider.svg", yb + 14)


if __name__ == "__main__":
    main()
