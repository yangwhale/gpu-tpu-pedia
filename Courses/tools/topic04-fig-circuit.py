# -*- coding: utf-8 -*-
r"""专题四 · §1「反向到底在干什么」——&#160;一个能用眼睛跟着数走的小电路

⭐⭐⭐ 2026-09-17 新画。第一节讲了偏导数、局部性、从后往前 ——&#160;
  **全是道理，没有一个数**。⛔ 而这一节恰恰是「最多人卡住」的那一节。
  ⭐ 判据：**一个抽象概念，至少要有一次「小到能用眼睛跟着走」的具体化。**
    不是为了简单，是为了让人**确认自己真的懂了** ——&#160;
    道理点头容易，跟着数走一遍才知道自己卡在哪。

⭐⭐ 取法来自 CS231n 那个经典的「电路图 ＋ 三个门」讲法：
  把式子画成电路，前向在线上标黑字，反向在同一条线上标红字。
  ⛔ 图是我们自己重画的，数也是我们自己挑的 ——&#160;只借讲法，不搬图。

⭐⭐⭐ 而这张图在**这一讲**里有个别处没有的用处：
  **乘法门反向的时候，要用到前向那两个输入的值。**
  ——&#160;这就是「激活为什么扔不掉」的最小例子。
  一整讲的激活账单，根子就在这一个门上。⭐ 所以 Ⓒ 必须留着。

📌 三个门的说法（分发 / 交换 / 路由）取自 CS231n 的 backprop 讲义。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE)

W = 1400

# ⭐ 这几个数是**挑过**的，挑的原则有三条，缺一个这张图就讲不清：
#   ① q 是负数 ——&#160;免得读者以为「梯度都是正的」
#   ② a 和 b 不相等 ——&#160;这样 max 门才有话可说
#   ③ c 和 q 差得开 ——&#160;交换之后一眼看出「换了」
A, B, C = 2.0, -3.0, 4.0
Q = A + B
F = Q * C
assert Q < 0 and A != B and abs(C) != abs(Q), "这几个数要满足文件头那三条"
# 反向（链式法则，手推一遍，脚本替你验）
dF = 1.0
dC, dQ = Q * dF, C * dF
dA = dB = dQ * 1.0          # 加法门：原样分发
assert (dA, dB, dC) == (4.0, 4.0, -1.0)


def main():
    f = Fig(W, "一个三节点的小电路：a 加 b 得到 q，q 乘 c 得到 f。"
               "前向在线上标黑字，反向在同一条线上标红字。"
               "加法门把上游的梯度原样分给两个输入，"
               "乘法门把两个输入的梯度换过来 —— 各自拿对方的前向值。"
               "而正因为乘法门反向要用前向的值，那个值就必须被存下来，"
               "这就是激活扔不掉的最小例子")

    y0 = f.header(
        "跟着数走一遍　——　<tspan font-weight=\"700\">"
        "三个节点，前向一遍，反向一遍</tspan>",
        "⭐ 前面讲的全是道理。<tspan font-weight=\"700\">"
        "道理点头很容易，跟着数走一遍才知道自己卡在哪</tspan>",
        [(INK, "黑字：前向的值"), (RD, "红字：反向的梯度")])

    # ══════════ Ⓐ 电路 ═════════════════════════════════════════════
    PH = 392
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 式子是 <tspan font-family=\"ui-monospace,monospace\">"
                 "f ＝ (a ＋ b) × c</tspan>",
                 BL, sub="⭐ 同一条线上<tspan font-weight=\"700\">两个数</tspan>："
                         "上面黑的是前向算出来的，下面红的是反向传回来的")

    NX, NY = 300, py + 120          # 加法门
    MX = 720                        # 乘法门
    FX = 1090                       # 输出

    def node(x, y, sym, col):
        f.box(x - 34, y - 34, 68, 68, "#fff", col, 34, sw=2)
        f.t(x, y + 10, sym, col, True, 26, "middle")

    def inp(x, y, name, val, grad):
        f.box(x - 52, y - 26, 104, 52, "#f1f3f4", GY2, 8)
        f.t(x, y - 2, name, INK, True, 17, "middle")
        f.t(x, y + 20, "＝ %g" % val, GY, size=13.5, anchor="middle")
        f.t(x, y + 48, "梯度 %g" % grad, RD, True, 14.5, "middle")

    inp(96, NY - 62, "a", A, dA)
    inp(96, NY + 62, "b", B, dB)
    inp(96 + (MX - NX) // 1 - 190, py + 274, "c", C, dC)

    node(NX, NY, "＋", BL)
    node(MX, NY, "×", OR)
    f.box(FX - 44, NY - 30, 148, 60, "#fce8e6", RD, 8)
    f.t(FX + 30, NY + 2, "f ＝ %g" % F, RD, True, 20, "middle")
    f.t(FX + 30, NY + 52, "梯度 %g" % dF, RD, True, 14.5, "middle")

    # 连线 ＋ 线上的两个数
    def wire(x0, y0_, x1, y1, val, grad, lab=None):
        f.line(x0, y0_, x1, y1, GY2, 1.6)
        mx, my = (x0 + x1) / 2.0, (y0_ + y1) / 2.0
        if lab:
            f.t(mx, my - 26, lab, GY2, size=12, anchor="middle")
        f.t(mx, my - 8, "%g" % val, INK, True, 14.5, "middle")
        f.t(mx, my + 16, "↤ %g" % grad, RD, True, 14.5, "middle")

    f.line(148, NY - 62, NX - 40, NY - 14, GY2, 1.6)
    f.line(148, NY + 62, NX - 40, NY + 14, GY2, 1.6)
    wire(NX + 40, NY, MX - 40, NY, Q, dQ, "q ＝ a ＋ b")
    f.line(96 + (MX - NX) - 190 + 52, py + 274, MX - 12, NY + 40, GY2, 1.6)
    f.line(MX + 40, NY, FX - 50, NY, GY2, 1.6)

    f.t(700, py + 356, "⭐ <tspan font-weight=\"700\">前向从左往右走一遍，"
        "反向沿同一批线从右往左走一遍</tspan>　——　"
        "红字就是「<tspan font-weight=\"700\">这个东西变一点，f 变多少</tspan>」",
        INK, size=14, anchor="middle")
    f._pan = None

    # ══════════ Ⓑ 三个门 ════════════════════════════════════════════
    # ⭐⭐⭐ 2026-09-18 重画。旧版是三张卡片，每张写「像一个分发器 / 交换器 /
    #   路由器」＋ 两行规则 ——&#160;把字删掉只剩三个空框，figink 判它写字板子。
    #   ⭐⭐ 而这三个门的差别**本来就是线的形状**：
    #     · 分发 ＝ 一进两出，两条一样粗
    #     · 交换 ＝ 两条交叉，各自拿对方的值
    #     · 路由 ＝ 一条走通，一条断掉
    #   ⭐⭐⭐ 判据：**当三样东西的差别可以画成三种连线形状时，
    #     写成三段文字就是把图形信息翻译成文字、再让读者翻译回去。**
    #   ⭐ 取法 Olah《Understanding LSTM Networks》：一张底图重复四次，
    #     每次只移动高亮。读者只需读一次骨架，之后注意力全给差异 ——&#160;
    #     **而且这是降成本的画法：一套骨架用三遍，比画三张卡片还省事。**
    PH2 = 326
    py2 = f.panel(0, py + PH + 20, W, PH2,
                  "Ⓑ ⭐⭐ 常见的门只有三种脾气　——　"
                  "<tspan font-weight=\"700\">而这三种脾气就是三种连线的形状</tspan>",
                  GR, sub="⛔ 不用背公式 ——&#160;"
                          "<tspan font-weight=\"700\">线越粗梯度越大，虚线表示负的</tspan>"
                          "，三张图的骨架完全一样，只看红线怎么走")

    # 线宽直接编码梯度大小 ——&#160;数值不用写在旁边也看得出谁大谁小
    _GMAX = 4.0

    def _lw(g):
        return 1.8 + 4.8 * abs(g) / _GMAX

    assert _lw(4.0) / _lw(1.0) > 1.9, "粗细差不够，看不出梯度大小的差别"
    assert max(A, B) == A, "这三格默认 a 是 max 的赢家，换数要同步改图"

    # (颜色, 底色, 符号, 名字, 像什么, a 拿到的, b 拿到的, 上游, 一句话)
    GATES = (
        (BL, "#e8f0fe", "＋", "加法门", "分发器", dA, dB, dQ,
         "两条<tspan font-weight=\"700\">一样粗</tspan>　——　原样各拿一份"),
        (OR, "#fef7e0", "×", "乘法门", "交换器", dQ, dC, dF,
         "两条<tspan font-weight=\"700\">交叉</tspan>　——　各自拿对方的前向值"),
        (PU, "#f3e8fd", "max", "max 门", "路由器", dQ, 0.0, dQ,
         "一条<tspan font-weight=\"700\">断了</tspan>　——　全给赢的那个"),
    )
    for i, (col, fill, sym, name, like, g_up_a, g_up_b, gin, one) in enumerate(GATES):
        x = 40 + i * 442
        f.box(x, py2 + 40, 418, 250, fill, col, 8)
        f.box(x, py2 + 40, 418, 4, col, col, 2)
        # ⚠️ max 的符号本身就是「max」，再加一次名字会变成「max max 门」
        _head = name if sym == "max" else "%s　%s" % (sym, name)
        f.t(x + 209, py2 + 76, "<tspan font-weight=\"700\">%s</tspan>"
            "　像一个<tspan font-weight=\"700\">%s</tspan>" % (_head, like),
            col, True, 17, "middle")

        # ── 骨架：两个输入在左，门在中，上游在右。三格完全一致 ──────
        ax, ay = x + 40, py2 + 142        # 上面那个输入
        by = py2 + 218                    # 下面那个
        gx, gy = x + 196, py2 + 180       # 门
        ux = x + 336                      # 上游

        for yy, nm in ((ay, "a"), (by, "b")):
            f.box(ax - 26, yy - 19, 52, 38, "#fff", GY2, 6)
            f.t(ax, yy + 6, nm, INK, True, 16, "middle")
        f.box(gx - 30, gy - 30, 60, 60, "#fff", col, 30, sw=2)
        f.t(gx, gy + 8, sym, col, True, 20 if sym != "max" else 15, "middle")
        f.box(ux - 4, gy - 19, 66, 38, "#fff", col, 6)
        f.t(ux + 29, gy + 6, "上游", col, True, 14, "middle")

        # ── 红线：从右往左，这是三格唯一不同的地方 ────────────────
        f.line(ux - 8, gy, gx + 34, gy, RD, _lw(gin), arrow=True)
        f.t((ux + gx) / 2 + 16, gy - 12, "%g" % gin, RD, True, 13.5, "middle")

        for yy, g in ((ay, g_up_a), (by, g_up_b)):
            if g == 0:
                # 路由器：这一条**真的断掉** ——&#160;不是画细，是画断
                f.line(gx - 34, gy, gx - 70, yy, GY2, 1.4, dash="4 4", arrow=False)
                f.t(gx - 88, yy + 5, "✕ 0", GY2, True, 14, "middle")
                continue
            f.line(gx - 34, gy, ax + 30, yy, RD, _lw(g),
                   dash="7 4" if g < 0 else None, arrow=True)
            f.t((gx + ax) / 2 + 4, yy + (-12 if yy == ay else 22), "%g" % g,
                RD, True, 13.5, "middle")

        # 乘法门：把「交换」真的画成一个 ✕
        if sym == "×":
            f.line(ax + 26, ay + 12, gx - 46, by - 16, GY, 1.1,
                   dash="3 3", arrow=False)
            f.line(ax + 26, by - 12, gx - 46, ay + 16, GY, 1.1,
                   dash="3 3", arrow=False)
            f.box(x + 95, py2 + 166, 28, 24, fill, fill, 4)
            f.t(x + 109, py2 + 184, "换", GY, True, 12.5, "middle")

        f.t(x + 209, py2 + 272, one, col, size=13, anchor="middle")
    f._pan = None

    # ══════════ Ⓒ 这张图跟这一讲的关系 ══════════════════════════════
    PH3 = 222
    py3 = f.panel(0, py2 + PH2 + 20, W, PH3,
                  "Ⓒ ⭐⭐⭐ 而中间那个门，"
                  "<tspan font-weight=\"700\">正是这一整讲的账单的源头</tspan>", RD,
                  sub="⛔ 别跳过这一格 ——&#160;"
                      "<tspan font-weight=\"700\">后面所有关于「激活」的话都从这儿来</tspan>")

    f.box(70, py3 + 40, 560, 158, "#fef7e0", OR, 8)
    f.t(350, py3 + 76, "乘法门反向的时候", OR, True, 18, "middle")
    f.t(350, py3 + 110, "<tspan font-weight=\"700\">要用到前向那两个输入的值</tspan>",
        INK, True, 17, "middle")
    f.t(350, py3 + 146, "c 的梯度 ＝ q，而 q 是前向算出来的", GY, size=13.5, anchor="middle")
    f.t(350, py3 + 176, "⛔ 所以 q <tspan font-weight=\"700\">不能扔</tspan>",
        RD, True, 16, "middle")

    f.t(670, py3 + 118, "→", GY2, True, 22, "middle")

    f.box(710, py3 + 40, 630, 158, "#fce8e6", RD, 8)
    f.t(1025, py3 + 76, "把这一个门乘以几百亿次", RD, True, 18, "middle")
    f.t(1025, py3 + 112, "<tspan font-weight=\"700\">就是那张「激活」的账单</tspan>",
        INK, True, 18, "middle")
    f.t(1025, py3 + 150, "⭐ 激活不是「框架顺手缓存的东西」", GY, size=13.5, anchor="middle")
    f.t(1025, py3 + 176, "它是<tspan font-weight=\"700\">反向的数学要求它在场</tspan>",
        RD, True, 15, "middle")
    f._pan = None

    yb = f.band(py3 + PH3 + 20, "ok",
                "所以「重算」那个决定，在这张小电路上也说得通",
                ("⭐ 把 q 扔掉，反向要用的时候<tspan font-weight=\"700\">"
                 "临时把 a ＋ b 再加一遍</tspan>　——　"
                 "一次加法，换一个数的存储空间。<tspan font-weight=\"700\">"
                 "这就是重算，整套逻辑一个字都不用改。</tspan>",
                 "⛔ 而这也解释了为什么<tspan font-weight=\"700\">"
                 "加法便宜、矩阵乘贵</tspan>：重算一个加法门几乎不要钱，"
                 "重算一个矩阵乘要把那一大坨乘法再做一遍。"))

    yb = f.src(yb + 16,
               "📌 「电路图 ＋ 三个门（分发 / 交换 / 路由）」这个讲法取自 CS231n 的"
               "反向传播讲义　——　<tspan font-weight=\"700\">图是我们自己重画的，"
               "数也是自己挑的</tspan>。",
               "⭐ 图上每个数都可以自己验：脚本里带 assert　——　"
               "a 和 b 的梯度都是 4、c 的梯度是 −1，对不上就不让构建。")

    f.save("fig4-circuit.svg", yb + 14)


if __name__ == "__main__":
    main()
