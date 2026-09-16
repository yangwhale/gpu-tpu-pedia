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
    PH2 = 292
    py2 = f.panel(0, py + PH + 20, W, PH2,
                  "Ⓑ ⭐⭐ 常见的门只有三种脾气　——　"
                  "<tspan font-weight=\"700\">记住这三个，你就能手推任何一张电路</tspan>",
                  GR, sub="⛔ 不用背公式，它们各自像一样东西")

    GATES = (
        (BL, "#e8f0fe", "＋", "加法门", "分发器",
         "上游给多少，<tspan font-weight=\"700\">两个输入原样各拿一份</tspan>",
         "图里：上游 %g　→　a 和 b 都是 %g" % (dQ, dA)),
        (OR, "#fef7e0", "×", "乘法门", "交换器",
         "<tspan font-weight=\"700\">各自拿对方的前向值</tspan>当系数",
         "图里：q 的梯度 ＝ c ＝ %g；c 的梯度 ＝ q ＝ %g" % (dQ, dC)),
        (PU, "#f3e8fd", "max", "max 门", "路由器",
         "<tspan font-weight=\"700\">全给赢的那个</tspan>，输的拿 0",
         "若是 max(a, b) ＝ %g　→　a 全拿，b 拿 0" % max(A, B)),
    )
    for i, (col, fill, sym, name, like, rule, ex) in enumerate(GATES):
        x = 40 + i * 442
        f.box(x, py2 + 36, 418, 216, fill, col, 8)
        f.box(x, py2 + 36, 418, 4, col, col, 2)
        f.t(x + 62, py2 + 96, sym, col, True, 26, "middle")
        f.t(x + 240, py2 + 84, name, INK, True, 18, "middle")
        f.t(x + 240, py2 + 110, "像一个<tspan font-weight=\"700\">%s</tspan>" % like,
            col, True, 16, "middle")
        f.t(x + 209, py2 + 158, rule, GY, size=13.5, anchor="middle")
        f.t(x + 209, py2 + 208, ex, col, size=12.5, anchor="middle")
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
