# -*- coding: utf-8 -*-
r"""专题四 · §3.6「两笔账，记性不一样长」—— β₁ 与 β₂ 各管多久，以及偏差修正

2026-11-08 补。§3.6 原来只用一句话讲 β：「有效窗口约 1/(1−β) 步」。
没讲的两件事，正好都是「两笔账记性不一样长」的直接后果：
  Ⓐ 梯度突然变了，m（β₁＝0.9，记十来步）先跟上，v（β₂ 记几十到上千步）慢半拍，
     于是那一段步子会暂时偏大（梯度变大时）或偏小（梯度变小时）。β₂ 越大，这段越长越猛。
  Ⓑ 偏差修正：m、v 都从 0 起步，第 t 步历史梯度的权重加起来只有 1−βᵗ。
     β₁＝0.9：第 1 步 0.1、第 2 步 0.19、第 10 步 0.65、第 50 步 0.995。
     除以它就补回来了；它只在开头起作用，之后自己趋近 1、等于没除。
     ⚠️ 「约 50 步就≈1」只对 β₁＝0.9 成立；β₂＝0.999 要几千步（第 50 步才 0.049）——
     这里两条都画，免得把 m 的结论套到 v 上。

承重的数都是现算的：
  · Ⓐ 真跑 Adam（Algorithm 1，含偏差修正），梯度在第 3000 步从 1 变成 10（或反过来），
    β₁＝0.9，β₂ 取原文默认 0.999 和大模型常用 0.95 两档
  · Ⓑ 1−βᵗ 直接算
出处：Adam 原文 arXiv 1412.6980（Algorithm 1、默认值、§3 偏差修正）；β₂＝0.95 见 GPT-3（arXiv 2005.14165
附录 B）与 Llama 2（arXiv 2307.09288 §2.2）原文，均已读。
"""
import math

from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE)

W = 1400
B1 = 0.9
B2S = ((0.999, PU, "β₂＝0.999（原文默认）"), (0.95, OR, "β₂＝0.95（大模型常用）"))
PRE, POST = 3000, 10000


def ratio_after(g0, g1, b2):
    """梯度在第 PRE 步之后从 g0 变成 g1；返回变化之后每一步的 m̂/√v̂。"""
    m = v = 0.0
    out = []
    for t in range(1, PRE + POST + 1):
        g = g0 if t <= PRE else g1
        m = B1 * m + (1 - B1) * g
        v = b2 * v + (1 - b2) * g * g
        r = (m / (1 - B1 ** t)) / math.sqrt(v / (1 - b2 ** t))
        if t > PRE:
            out.append(r)
    return out


UP = {b2: ratio_after(1.0, 10.0, b2) for b2, _, _ in B2S}
DN = {b2: ratio_after(10.0, 1.0, b2) for b2, _, _ in B2S}
PEAK = {b2: max(UP[b2]) for b2 in UP}
DIP = {b2: min(DN[b2]) for b2 in DN}


def settle(rs, tol=0.05):
    """从哪一步起一直留在 1±tol 里。"""
    for k in range(len(rs)):
        if all(abs(x - 1) < tol for x in rs[k:]):
            return k + 1
    return None


SET_UP = {b2: settle(UP[b2]) for b2 in UP}
SET_DN = {b2: settle(DN[b2]) for b2 in DN}
assert PEAK[0.999] > 4 and PEAK[0.95] < 1.3, PEAK            # β₂ 越大越猛
assert DIP[0.999] < 0.2 and 0.2 < DIP[0.95] < 0.4, DIP
assert SET_UP[0.999] > 1000 and SET_UP[0.95] < 100, SET_UP    # β₂ 越大越长
assert SET_DN[0.999] > 1000 and SET_DN[0.95] < 200, SET_DN
assert abs(UP[0.999][-1] - 1) < 0.05 and abs(UP[0.95][-1] - 1) < 1e-6   # 最后都回到 1

# ══════════ Ⓑ 偏差修正 ══════════════════════════════════════════
def wsum(b, t):
    return 1 - b ** t


MARKS = (1, 2, 10, 50)
WS = {t: wsum(B1, t) for t in MARKS}
assert abs(WS[1] - 0.1) < 1e-12 and abs(WS[2] - 0.19) < 1e-12
assert abs(WS[10] - 0.651) < 1e-3 and WS[50] > 0.99
assert wsum(0.999, 50) < 0.05 and 0.6 < wsum(0.999, 1000) < 0.65 and wsum(0.999, 3000) > 0.95
assert 0.9 < wsum(0.95, 50) < 0.93


def main():
    f = Fig(W, "两笔账的记性不一样长。第一格：梯度在某一步突然从一变成十，"
               "动量 m 十来步就跟上了，二阶动量 v 慢半拍，于是那一段每步走得比一个学习率多："
               "β₂ 取零点九九九时最多冲到五倍左右，两千多步才回到一；β₂ 取零点九五时只多一成，几十步就回来。"
               "梯度从十突然变成一时反过来，步子暂时变小。"
               "第二格：偏差修正。m 和 v 都从零起步，第 t 步各历史梯度的权重之和只有一减 β 的 t 次方："
               "β₁ 等于零点九时，第一步零点一、第二步零点一九、第十步零点六五、第五十步零点九九五；"
               "β₂ 等于零点九九九时第五十步才零点零五，要几千步才接近一。除以这个和，就把开头的偏小补回来，"
               "之后它趋近一，等于没除。")
    y0 = f.header(
        "两笔账，<tspan font-weight=\"700\">记性不一样长</tspan>　——　"
        "m 记十来步，v 记得久得多",
        "β 越接近 1 记得越久（窗口约 1 ÷ (1−β) 步）：β₁＝0.9 约 10 步；"
        "β₂＝0.95 约 20 步，0.999 约 1000 步",
        [(PU, B2S[0][2]), (OR, B2S[1][2]), (GR, "β₁＝0.9")])

    # ══════════ Ⓐ ══════════════════════════════════════════════════
    PH = 380
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 梯度<tspan font-weight=\"700\">突然变了</tspan>：m 先跟上，"
                 "v 慢半拍 → 步子<tspan font-weight=\"700\">暂时走样</tspan>", PU,
                 sub="纵轴：每步走几个学习率（m ÷ √v）；横轴：变化之后第几步（对数轴）")
    LXMAX = math.log10(POST)

    def plot(x0, title, data, ymax, ticks, notes):
        aw, ah, oy = 560, 230, py + 300
        f.t(x0 + aw / 2, py + 34, title, INK, True, 14.5, "middle")
        f.line(x0, oy, x0 + aw, oy, GY2, 1, arrow=False)
        f.line(x0, oy, x0, oy - ah, GY2, 1, arrow=False)
        for lv in ticks:
            yy = oy - ah * lv / ymax
            f.line(x0, yy, x0 + aw, yy, GY2, 1, dash="3 4" if lv != 1 else None, arrow=False)
            f.t(x0 - 8, yy + 4, "%g" % lv, GY2, size=11, anchor="end", mono=True)
        for d in range(0, int(LXMAX) + 1):
            xx = x0 + aw * d / LXMAX
            f.t(xx, oy + 18, "%d" % 10 ** d, GY2, size=11, anchor="middle", mono=True)
        for b2, col, _ in B2S:
            pts = []
            for k, r in enumerate(data[b2]):
                px = x0 + aw * math.log10(k + 1) / LXMAX
                pts.append((px, oy - ah * min(r, ymax) / ymax))
            f.path(pts, col, 2.6, arrow=False)
        for (x, y, s, col) in notes:
            f.t(x0 + x, oy - ah * y / ymax, s, col, True, 12.5)

    plot(90, "梯度从 1 跳到 10：步子暂时变大", UP, 6.0, (1, 3, 5),
         ((150, PEAK[0.999] + 0.35, "最多 %.1f 倍，%d 步后才回到 1±5%%"
           % (PEAK[0.999], SET_UP[0.999]), PU),
          (250, 0.5, "最多 %.2f 倍，%d 步就回来" % (PEAK[0.95], SET_UP[0.95]), OR)))
    plot(770, "梯度从 10 掉到 1：步子暂时变小", DN, 1.2, (0.5, 1),
         ((200, DIP[0.999] + 0.12, "最低 %.2f，%d 步后才回到 1±5%%" % (DIP[0.999], SET_DN[0.999]), PU),
          (40, 0.93, "最低 %.2f，%d 步回来" % (DIP[0.95], SET_DN[0.95]), OR)))
    f.t(700, py + 340,
        "<tspan font-weight=\"700\">β₁ ＜ β₂，所以分子比分母反应快</tspan>"
        "　——　β₂ 越大，这段走样越长、越猛", INK, size=15, anchor="middle")
    f._pan = None

    # ══════════ Ⓑ ══════════════════════════════════════════════════
    PH2 = 360
    py2 = f.panel(0, py + PH + 20, W, PH2,
                  "Ⓑ 偏差修正：从 0 起步，开头那几步<tspan font-weight=\"700\">权重凑不满 1</tspan>", GR,
                  sub="纵轴：第 t 步，所有历史梯度的权重加起来是多少（＝ 1 − βᵗ）；横轴：步数（对数轴）")
    OX, OY, AW, AH = 120, py2 + 280, 900, 220
    TMAX = 5000
    LT = math.log10(TMAX)
    f.line(OX, OY, OX + AW, OY, GY2, 1, arrow=False)
    f.line(OX, OY, OX, OY - AH, GY2, 1, arrow=False)
    for lv in (0.5, 1.0):
        yy = OY - AH * lv
        f.line(OX, yy, OX + AW, yy, GY2, 1, dash="3 4", arrow=False)
        f.t(OX - 8, yy + 4, "%g" % lv, GY2, size=11, anchor="end", mono=True)
    f.t(OX - 8, OY + 4, "0", GY2, size=11, anchor="end", mono=True)
    for t in (1, 10, 100, 1000):
        f.t(OX + AW * math.log10(t) / LT, OY + 18, "%d" % t, GY2, size=11, anchor="middle", mono=True)

    def tx(t):
        return OX + AW * math.log10(t) / LT

    for b, col, sw in ((B1, GR, 3.0), (0.95, OR, 2.0), (0.999, PU, 2.0)):
        pts = []
        for k in range(0, 301):
            t = 10 ** (LT * k / 300)
            pts.append((tx(t), OY - AH * wsum(b, t)))
        f.path(pts, col, sw, arrow=False)
    f.t(tx(1.3), OY - AH * 1.0 - 10, "β₁＝0.9：约 50 步就凑满", GR, True, 12.5)
    f.t(tx(300), OY - AH * wsum(0.999, 300) + 4, "β₂＝0.999：要几千步", PU, True, 12.5)
    f.t(tx(70), OY - AH * 0.78, "β₂＝0.95", OR, True, 12)
    LAB = {1: (tx(22), 0.24), 2: (tx(22), 0.36), 10: (None, None), 50: (None, None)}
    for t in MARKS:
        x, y = tx(t), OY - AH * WS[t]
        f.box(x - 5, y - 5, 10, 10, GR, GR, 5)
        s = "第 %d 步 %s" % (t, ("%.2f" % WS[t]) if t != 50 else "%.3f" % WS[t])
        lx_, ly_ = LAB[t]
        if lx_ is None:                       # 第 10、50 步：标签贴在点的左上
            f.t(x - 10, y - 10, s, GR, True, 12, "end")
        else:                                 # 第 1、2 步挤在角落里，用引线拉到空处
            yy = OY - AH * ly_
            f.line(x + 6, y - 2, lx_ - 6, yy - 4, GR, 1, arrow=False)
            f.t(lx_, yy, s, GR, True, 12)
    # 右侧：除以它
    RX = 1060
    f.t(RX, py2 + 70, "第 1 步，梯度是 g：", INK, True, 14)
    f.t(RX, py2 + 98, "m ＝ 0.1 × g（偏小 10 倍）", GY, size=13)
    f.t(RX, py2 + 124, "修正：m ÷ (1−0.9) ＝ g", GR, True, 13.5)
    f.t(RX, py2 + 164, "第 2 步：m 不是 0 了，", INK, True, 14)
    f.t(RX, py2 + 190, "可权重只凑到 0.19，", GY, size=13)
    f.t(RX, py2 + 216, "仍偏小 → 除以 0.19", GR, True, 13.5)
    f.t(RX, py2 + 256, "凑满之后：除以 ≈1", INK, True, 14)
    f.t(RX, py2 + 282, "＝ 修正自己消失", GR, True, 13.5)
    f.t(700, py2 + 318,
        "<tspan font-weight=\"700\">偏差修正只管开头</tspan>：m 管几十步，v 的要管到它那根曲线凑满为止",
        INK, size=15, anchor="middle")
    f._pan = None

    yb = f.band(py2 + PH2 + 20, "ok",
                "β 就是「记多久」，而 Adam 的两笔账记性故意不一样长",
                ("m 要跟得紧（方向一变就该知道），v 要看得稳（「平常多大」得多看一阵）；"
                 "<tspan font-weight=\"700\">代价是梯度突变时，两笔账有一段对不上，步子暂时走样</tspan>。",
                 "偏差修正补的是「从 0 起步」那一下：没有它，开头几步的 m、v 都偏小，"
                 "两者偏得还不一样多，步子就失真。"))
    yb = f.src(yb + 16,
               "Adam 原文（arXiv 1412.6980，已读原文）Algorithm 1：m̂ ＝ m/(1−β₁ᵗ)、v̂ ＝ v/(1−β₂ᵗ)；"
               "默认 β₁＝0.9、β₂＝0.999。§3 说滑动平均从 0 起步，估计偏向 0，开头几步尤甚，β 越接近 1 越严重。",
               "β₂＝0.95：GPT-3（arXiv 2005.14165 附录 B）「Adam with β₁ ＝ 0.9, β₂ ＝ 0.95, ε ＝ 10⁻⁸」；"
               "Llama 2（arXiv 2307.09288 §2.2）「AdamW … β₁ ＝ 0.9, β₂ ＝ 0.95, eps ＝ 10⁻⁵」。均已读原文。",
               "Ⓐ 是本课的数值实验：梯度先恒为 1（或 10）跑 %d 步到稳态，再突变；"
               "「回到 1±5%%」按之后一直留在这个范围里算。真实训练的梯度不会这么整齐地跳，"
               "这一格只说明「分子分母反应速度不同」这一件事，不报任何真实模型的数。" % PRE,
               "Ⓑ 的数直接算 1−βᵗ：β₁＝0.9 时 0.1、0.19、%.3f、%.4f（第 1、2、10、50 步）；"
               "β₂＝0.999 时第 50 步 %.3f、第 1000 步 %.3f、第 3000 步 %.3f。"
               % (WS[10], WS[50], wsum(0.999, 50), wsum(0.999, 1000), wsum(0.999, 3000)))
    f.save("fig4-adamtime.svg", yb + 14)


if __name__ == "__main__":
    main()
