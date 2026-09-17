# -*- coding: utf-8 -*-
r"""专题四 · §1.4「一路乘下去会怎样」——&#160;梯度消失与梯度爆炸

⭐⭐⭐ 2026-09-17 R02/20 新画。这是「二十轮重做」查出的第一个真空洞：
  §1.4 最后一句是「**你只要把 61 个局部的小导数，按顺序乘起来**」——&#160;
  ⛔ 然后这一讲就再没说过「**乘起来会怎样**」。
  而那恰恰是链式法则最直接、也最出名的后果。
  ⭐ `fig-slider` Ⓒ 里其实已经埋了伏笔（那条 assert 写着
    「链条里要有一级是缩小的，否则会被读成越乘越大」），这一格来兑现它。

⭐⭐ 按这一轮的新判据画：**把文字全删掉，这一格还剩三样真东西** ——&#160;
  Ⓐ 三条真跑出来的曲线、Ⓑ 一根跨十个数量级的对数轴、Ⓒ 两条真画的导数曲线。
  ⛔ 一个带彩色边框的文字块都没有。

⭐⭐⭐ 三条数都是**算出来的，不是查来的**：
  · 每层 0.8，六十层（＝ 59 个间隔）→ 0.8⁵⁹ ≈ 1.9 × 10⁻⁶
  · 每层 1.2，六十层 → 1.2⁵⁹ ≈ 4.7 × 10⁴
  · **两者只差两成，结果差了十个数量级** ——&#160;这才是这一格的落点：
    **指数放大的不是梯度，是「每层偏离 1 多少」那个微小的偏差。**
  · Ⓒ 那个 0.25 是**闭式的**：sigmoid 的导数 ＝ σ(1−σ)，在 σ ＝ ½ 处取最大，
    **恰好 0.25** ——&#160;也就是说，光是 sigmoid 那一下，每层就先乘了个 ≤ ¼ 的数。
    ⚠️ 权重那一乘还在外面，所以不能说「一定衰减四倍」，
    只能说「**它先天就把兑换率往 1 以下压**」。

⚠️ 这一格只讲「**会怎样**」和「**为什么**」，不展开治法 ——&#160;
  归一化、残差、裁剪各自在本讲后面有专门的位置。Ⓒ 只给激活函数这一条，
  因为它是**唯一一条能在这一格里当场算清楚**的。
"""
import math

from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE)

W = 1400

# ══════════════════════════════════════════════════════════════════
# ⭐ 让它自己乘出来 ——&#160;判据：能跑出来的绝不手摆
# ══════════════════════════════════════════════════════════════════
L = 60                                   # 层数
CASES = ((0.8, RD, "每层 ×0.8"),
         (1.0, GY2, "每层 ×1.0"),
         (1.2, BL, "每层 ×1.2"))


def chain(r, n=L):
    """从输出层往输入层一路乘：第 k 层（1＝最靠输入）拿到的相对梯度。"""
    out, v = [], 1.0
    for _ in range(n):
        out.append(v)
        v *= r
    return out[::-1]                     # 反过来，索引 0 ＝ 最靠输入那层


CURVES = {r: chain(r) for r, _, _ in CASES}
SHRINK = CURVES[0.8][0]                  # 0.8⁵⁹（最靠输入那层）
GROW = CURVES[1.2][0]

# 落点那个数：只差两成，差几个数量级
DECADES = math.log10(GROW / SHRINK)
assert SHRINK < 1e-5, "衰减那条要小到肉眼没有，现在是 %.3g" % SHRINK
assert GROW > 1e4, "爆炸那条要大到冲出画面，现在是 %.3g" % GROW
assert DECADES > 10, "两头要拉开十个数量级以上，现在是 %.1f" % DECADES
assert abs(CURVES[1.0][0] - 1.0) < 1e-12, "×1.0 那条必须纹丝不动"

# ⭐ Ⓒ：sigmoid 的导数上限是闭式的 ——&#160;σ'(x) ＝ σ(1−σ)，σ＝½ 时最大
SIG_MAX = 0.25
assert abs(max(_s * (1 - _s) for _s in [i / 10000.0 for i in range(1, 10000)])
           - SIG_MAX) < 1e-6, "sigmoid 导数的最大值必须真的是 0.25"


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def main():
    f = Fig(W, "链式法则把每层的兑换率一路乘起来，而连乘是指数的。"
               "六十层的网络，如果每层的兑换率是零点八，"
               "最靠输入那层拿到的梯度只有百万分之一点九，等于没在学；"
               "如果是一点二，就涨到四万七千倍，直接炸掉。"
               "这两个数只差两成，结果差了十个数量级 —— "
               "被指数放大的不是梯度本身，是每层偏离一多少那个微小的偏差。"
               "最后看激活函数：sigmoid 的导数最大只有零点二五，"
               "先天就把兑换率往一以下压；"
               "而 ReLU 在正半轴导数恰好是一，不引入任何衰减")

    y0 = f.header(
        "一路乘下去会怎样　——　<tspan font-weight=\"700\">"
        "梯度消失，和梯度爆炸</tspan>",
        "⛔ 上一格说「把 61 个小导数按顺序乘起来」——&#160;"
        "<tspan font-weight=\"700\">这一格说乘起来之后发生了什么</tspan>",
        [(RD, "每层 ×0.8：消失"), (GY2, "×1.0：刚好"), (BL, "×1.2：爆炸")])

    # ══════════ Ⓐ 六十层，三条真乘出来的曲线 ═══════════════════════
    PH = 440
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 同一个 <tspan font-weight=\"700\">%d 层</tspan>的网络，"
                 "只改「每层的兑换率」——&#160;"
                 "<tspan font-weight=\"700\">三条线都是真乘出来的</tspan>" % L, INK,
                 sub="⭐ 横轴左边是靠输入的层，右边是靠输出的层；"
                     "纵轴是那一层拿到的梯度有多大")

    OX, OY = 130, py + 348           # 原点（左下）
    AW, AH = 1080, 300               # 轴长
    UNIT = 52.0                      # 「相对大小 ＝ 1」对应多少像素

    f.line(OX, OY, OX + AW + 30, OY, GY2, 1.2, arrow=False)
    f.line(OX, OY, OX, OY - AH - 20, GY2, 1.2, arrow=False)
    f.line(OX - 8, OY - UNIT, OX + AW, OY - UNIT, GY2, 1, dash="3 4", arrow=False)
    f.t(OX - 14, OY - UNIT + 5, "1", GY2, size=12, anchor="end")
    f.t(OX - 14, OY + 5, "0", GY2, size=12, anchor="end")
    f.t(OX, OY + 26, "第 1 层（最靠输入）", GY2, size=12)
    f.t(OX + AW, OY + 26, "第 %d 层（最靠输出）" % L, GY2, size=12, anchor="end")

    # ⛔⛔ 第一版这里写错了：碰到越界就 `break`。
    #   而 ×1.2 那条**最靠输入的那一端就是最大的**，于是第一个点就越界、
    #   整条线一个像素都没画出来，只剩一个孤零零的「冲出画面」箭头挂在左边缘。
    #   ⭐ 判据：**曲线越界要「分段跳过」，不是「遇到就停」** ——&#160;
    #     一条线可能从画面外进来，也可能出去之后再回来。
    for r, col, tag in CASES:
        vals = CURVES[r]
        runs, cur, exits = [], [], []
        inside_prev = False
        for k, v in enumerate(vals):
            x = OX + AW * k / float(L - 1)
            y = OY - v * UNIT
            inside = (y >= OY - AH)
            if inside:
                cur.append((x, y))
            else:
                if inside_prev:               # 从画里走到画外
                    exits.append((cur[-1][0], OY - AH, +1))
                if cur:
                    runs.append(cur)
                    cur = []
            if inside and not inside_prev and k > 0:   # 从画外回到画里
                exits.append((x, OY - AH, -1))
            inside_prev = inside
        if cur:
            runs.append(cur)
        for seg in runs:
            if len(seg) < 2:
                continue
            d = "M %.1f %.1f" % seg[0]
            for xx, yy in seg[1:]:
                d += " L %.1f %.1f" % (xx, yy)
            f.path(d, col, 2.6 if r != 1.0 else 1.8,
                   dash="5 4" if r == 1.0 else None, arrow=False)
        # ⛔ 方向别搞反：sign=-1 表示「曲线是从画外走进来的」，
        #   那么画外在**左边**，箭头就该指左上；sign=+1 反之。
        #   ⭐ 判据：**箭头指的是「它往哪儿去了」，不是「它从哪儿来」。**
        for ex, ey, sign in exits[:1]:
            f.line(ex, ey + 14, ex + 18 * sign, ey - 18, col, 2.4)
            f.t(ex + 26 * sign, ey - 24, "⛔ 冲出画面", col, True, 14,
                "start" if sign > 0 else "end")

    # 左端那两条都贴在轴上，专门标一下
    f.line(OX + 60, OY - 96, OX + 14, OY - 6, RD, 1.6)
    f.t(OX + 68, OY - 100, "⛔ 这条<tspan font-weight=\"700\">贴在轴上</tspan>"
                           "　——　不是画漏了，是真的小到画不出来", RD, True, 14)

    f._pan = None

    # ══════════ Ⓑ 只差两成，差十一个数量级 ═════════════════════════
    PH2 = 300
    py2 = f.panel(0, py + PH + 20, W, PH2,
                  "Ⓑ ⭐⭐⭐ 换一根<tspan font-weight=\"700\">对数轴</tspan>才装得下"
                  "　——　两头差了 <tspan font-weight=\"700\">%d 个数量级</tspan>"
                  % int(round(DECADES)), PU,
                  sub="⭐ 而那两个兑换率<tspan font-weight=\"700\">只差两成</tspan>")

    LX, LW, LY = 180, 1050, py2 + 150
    LO, HI = -7.0, 6.0                      # 对数轴范围
    f.line(LX, LY, LX + LW + 20, LY, GY2, 1.4, arrow=False)

    def lx(v):
        return LX + LW * (math.log10(v) - LO) / (HI - LO)

    for e in range(-6, 6, 2):
        x = lx(10.0 ** e)
        f.line(x, LY - 8, x, LY + 8, GY2, 1, arrow=False)
        f.t(x, LY + 30, "10^%d" % e, GY2, size=12, anchor="middle", mono=True)

    for v, col, lab, sub, up in ((SHRINK, RD, "每层 ×0.8", "≈ %.1f × 10⁻⁶" % (SHRINK * 1e6), True),
                                 (1.0, GY2, "每层 ×1.0", "＝ 1", False),
                                 (GROW, BL, "每层 ×1.2", "≈ %.1f 万" % (GROW / 1e4), True)):
        x = lx(v)
        dy = -1 if up else 1
        f.box(x - 6, LY - 6, 12, 12, col, col, 6)
        f.line(x, LY + dy * 14, x, LY + dy * 46, col, 1.6, arrow=False)
        f.t(x, LY + dy * 58 + (0 if up else 14), lab, col, True, 15, "middle")
        f.t(x, LY + dy * 58 + (22 if up else 36), sub, GY, size=13, anchor="middle")

    # 两端之间拉一根跨度线
    f.line(lx(SHRINK), LY - 104, lx(GROW), LY - 104, PU, 2.0, arrow=False)
    for v in (SHRINK, GROW):
        f.line(lx(v), LY - 112, lx(v), LY - 96, PU, 1.6, arrow=False)
    f.t((lx(SHRINK) + lx(GROW)) / 2.0, LY - 116,
        "<tspan font-weight=\"700\">相差 %d 个数量级</tspan>" % int(round(DECADES)),
        PU, True, 16, "middle")

    f.t(700, py2 + 268,
        "⭐⭐⭐ 所以被指数放大的<tspan font-weight=\"700\">不是梯度</tspan>，"
        "是<tspan font-weight=\"700\">「每层偏离 1 多少」那个微小的偏差</tspan>"
        "　——　偏一点点，乘六十次就回不来了。",
        INK, size=15, anchor="middle")
    f._pan = None

    # ══════════ Ⓒ 激活函数先天就在往下压 ═══════════════════════════
    PH3 = 386
    py3 = f.panel(0, py2 + PH2 + 20, W, PH3,
                  "Ⓒ ⭐⭐ 那「每层的兑换率」从哪来 ——&#160;"
                  "<tspan font-weight=\"700\">其中一截是激活函数的导数</tspan>", GR,
                  sub="⭐ 这两条是<tspan font-weight=\"700\">真画的函数曲线</tspan>，"
                      "不是示意")

    CX0, CY0, CW, CH = 230, py3 + 296, 620, 210
    f.line(CX0 - 40, CY0, CX0 + CW + 40, CY0, GY2, 1.2, arrow=False)
    f.line(CX0 + CW / 2, CY0 + 14, CX0 + CW / 2, CY0 - CH - 24, GY2, 1.2, arrow=False)
    f.t(CX0 + CW / 2 - 10, CY0 - CH - 30, "导数", GY2, size=12, anchor="end")
    f.t(CX0 + CW + 46, CY0 + 5, "输入 →", GY2, size=12)
    for lv, lab in ((1.0, "1.0"), (SIG_MAX, "0.25")):
        yy = CY0 - CH * lv
        f.line(CX0 - 30, yy, CX0 + CW + 20, yy, GY2, 1, dash="3 4", arrow=False)
        f.t(CX0 - 38, yy + 5, lab, GY2, size=12, anchor="end")

    XLO, XHI, N = -6.0, 6.0, 240
    d_sig, d_relu = None, None
    for i in range(N + 1):
        xv = XLO + (XHI - XLO) * i / N
        px = CX0 + CW * i / float(N)
        s = sigmoid(xv)
        ys = CY0 - CH * (s * (1 - s))
        yr = CY0 - CH * (1.0 if xv > 0 else 0.0)
        d_sig = ("M %.1f %.1f" % (px, ys)) if d_sig is None else d_sig + " L %.1f %.1f" % (px, ys)
        d_relu = ("M %.1f %.1f" % (px, yr)) if d_relu is None else d_relu + " L %.1f %.1f" % (px, yr)
    f.path(d_relu, GR, 2.6, arrow=False)
    f.path(d_sig, RD, 2.6, arrow=False)
    f.box(CX0 + CW / 2 - 5, CY0 - CH * SIG_MAX - 5, 10, 10, RD, RD, 5)

    f.t(CX0 + CW * 0.80, CY0 - CH - 4, "ReLU 的导数", GR, True, 15, "middle")
    f.t(CX0 + CW * 0.80, CY0 - CH + 20, "正半轴恒等于 1", GY, size=13, anchor="middle")
    f.t(CX0 + CW / 2 + 12, CY0 - CH * SIG_MAX - 16,
        "sigmoid 的导数，峰值就这么高", RD, True, 14)

    f.box(920, py3 + 74, 430, 226, "#fce8e6", RD, 8)
    f.t(1135, py3 + 112, "sigmoid 的导数最大是多少", RD, True, 16, "middle")
    f.t(1135, py3 + 156, "σ(1 − σ) 在 σ ＝ ½ 处最大", INK, size=14.5, anchor="middle")
    f.t(1135, py3 + 196, "＝ <tspan font-weight=\"700\">0.25</tspan>",
        INK, True, 26, "middle")
    f.t(1135, py3 + 236, "⛔ 光这一下，每层就先乘了个 ≤ ¼ 的数",
        RD, True, 14, "middle")
    f.t(1135, py3 + 266, "⚠️ 权重那一乘还在外面 ——　所以不能说", GY, size=12.5, anchor="middle")
    f.t(1135, py3 + 286, "「一定衰减四倍」，只能说它<tspan font-weight=\"700\">先天往下压</tspan>",
        GY, size=12.5, anchor="middle")

    f.t(700, py3 + 352,
        "⭐⭐⭐ 这就是为什么 ReLU 一换上来，深网络忽然就训得动了"
        "　——　<tspan font-weight=\"700\">它把每层那个「先天往下压」的系数，从 ¼ 变回了 1。</tspan>",
        INK, size=15, anchor="middle")
    f._pan = None

    yb = f.band(py3 + PH3 + 20, "warn",
                "这一格只回答「会怎样」和「为什么」——　治法在本讲后面，各有各的位置",
                ("⭐⭐⭐ 要记的只有一句："
                 "<tspan font-weight=\"700\">链式法则是连乘，而连乘是指数的。</tspan>"
                 "所以真正要盯的<tspan font-weight=\"700\">不是「梯度大不大」，"
                 "是「每层那个兑换率离 1 有多远」</tspan> ——&#160;"
                 "离一点点，乘几十层就回不来了。",
                 "⛔ 而两头的<tspan font-weight=\"700\">症状完全不同</tspan>："
                 "消失是<tspan font-weight=\"700\">静悄悄的</tspan>"
                 "（loss 就是不降，前面几层跟没训一样，没有任何报错）；"
                 "爆炸是<tspan font-weight=\"700\">吵闹的</tspan>"
                 "（loss 直接飞掉或者变 NaN）。"
                 "⭐ 所以爆炸好查，消失难查。",
                 "⚠️ 图上的 0.8 / 1.2 是<tspan font-weight=\"700\">挑出来的示意值</tspan>，"
                 "真实网络每层那个系数不是一个常数、也不逐层相同。"
                 "⭐ 但<tspan font-weight=\"700\">「偏离 1 会被指数放大」这件事跟具体取值无关</tspan>"
                 "　——　这一格要的就是这一条。"))

    yb = f.src(yb + 16,
               "⭐ Ⓐ 三条曲线、Ⓑ 那两个端点值、Ⓒ 那个 0.25，"
               "<tspan font-weight=\"700\">全是脚本算的，没有一个是抄来的</tspan>："
               "0.8 的 %d 次方 ≈ %.2g、1.2 的 %d 次方 ≈ %.0f，"
               "两者相差 <tspan font-weight=\"700\">%d 个数量级</tspan>；"
               "0.25 是 σ(1−σ) 的闭式最大值，脚本里还用一万个采样点复核了一遍。"
               % (L - 1, SHRINK, L - 1, GROW, int(round(DECADES))),
               "⚠️ <tspan font-weight=\"700\">Ⓒ 只挑了激活函数这一截</tspan> ——&#160;"
               "每层真正的兑换率还要乘上权重矩阵那一下（以及归一化那一下）。"
               "选它是因为它是这一格里<tspan font-weight=\"700\">唯一能当场算清楚的</tspan>，"
               "⛔ 不是因为它是唯一的原因。",
               "📌 历史留到讲义里讲，这里只记一句：这件事"
               "<tspan font-weight=\"700\">1991 年就被正式指出了</tspan>"
               "　——　而且出处是一篇<tspan font-weight=\"700\">德文硕士论文</tspan>，"
               "从没在英文期刊上发表过。⚠️ 这条的归属至今仍有争议，"
               "所以本讲只说「1991 年那篇论文正式指出」，不卷进优先权之争。")

    f.save("fig4-vanish.svg", yb + 14)


if __name__ == "__main__":
    main()
