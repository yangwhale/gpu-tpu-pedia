# -*- coding: utf-8 -*-
r"""专题四 · §3.3b「Muon 从零讲」第一张：正交到底是什么

2026-11-08 补。Muon 那一串讲法（正交化、奇异值、拉平）全部踩在「正交」这个词上，
而这一讲从没讲过它。现场讲通的顺序是：两根箭头垂直 → 正交矩阵 → 维数上限。

三格，一格一个问题：
  Ⓐ 「正交」是什么？ —— 两根箭头垂直；用数说：对应位置相乘再相加（内积）等于 0。
     例：(3,4) 与 (−4,3)：3×(−4)＋4×3 ＝ 0。
  Ⓑ 正交矩阵凭什么特别？ —— 每列长度 1 且两两垂直，乘上去只转不拉。
     转 30° 的矩阵 [[0.866,−0.5],[0.5,0.866]] 乘 (3,4)，长度仍是 5；
     反例 [[1,0.6],[0,0.8]]：两列长度都是 1，可不垂直（内积 0.6），乘 (3,4) 就被拉到 6.28。
  Ⓒ 为什么长方形矩阵只能「半正交」？ —— 平面里最多两根互相垂直的箭头。
     6×2 的矩阵：两列（6 维里两根）可以垂直；6 行（平面里 6 根）不可能两两垂直。

承重的数全部在下面现算并断言。刻意没画：三维以上的情形（画不出来，Ⓒ 只用二维说维数上限）。
"""
import math

from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE)

W = 1400

# ── Ⓐ 内积为零 ────────────────────────────────────────────────────
A_U, A_V = (3.0, 4.0), (-4.0, 3.0)
assert A_U[0] * A_V[0] + A_U[1] * A_V[1] == 0

# ── Ⓑ 转 30° vs 反例 ─────────────────────────────────────────────
C30, S30 = math.cos(math.radians(30)), math.sin(math.radians(30))
ROT = ((C30, -S30), (S30, C30))
BAD = ((1.0, 0.6), (0.0, 0.8))


def mv(m, v):
    return (m[0][0] * v[0] + m[0][1] * v[1], m[1][0] * v[0] + m[1][1] * v[1])


def nrm(v):
    return math.hypot(v[0], v[1])


X = (3.0, 4.0)
# 图上印的矩阵元素（三位小数）——必须跟真矩阵对得上
TXT = ((("0.866", "−0.5"), ("0.5", "0.866")), (("1", "0.6"), ("0", "0.8")))
RX, BX_ = mv(ROT, X), mv(BAD, X)
assert round(C30, 3) == 0.866 and abs(S30 - 0.5) < 1e-12
assert abs(nrm(X) - 5) < 1e-12 and abs(nrm(RX) - 5) < 1e-12       # 只转不拉
for m in (ROT, BAD):                                                 # 两个矩阵每列长度都是 1
    for j in range(2):
        assert abs(math.hypot(m[0][j], m[1][j]) - 1) < 1e-12
COLDOT_BAD = BAD[0][0] * BAD[0][1] + BAD[1][0] * BAD[1][1]
assert abs(COLDOT_BAD - 0.6) < 1e-12                                 # 反例两列不垂直
assert abs(BX_[0] - 5.4) < 1e-12 and abs(BX_[1] - 3.2) < 1e-12
BAD_LEN = nrm(BX_)
for _m, _tx in zip((ROT, BAD), TXT):
    for _r in range(2):
        for _c in range(2):
            assert abs(float(_tx[_r][_c].replace("−", "-")) - _m[_r][_c]) < 5e-4, (_tx, _m)
assert abs(BAD_LEN - 6.28) < 0.005, BAD_LEN                          # 图上写 6.28


def main():
    f = Fig(W, "正交三格。第一格：两根箭头垂直叫正交，用数说就是对应位置相乘再相加等于零，"
               "例如 3,4 和 负4,3。第二格：正交矩阵每列长度为一且两两垂直，乘到向量上只转不拉，"
               "转 30 度的矩阵把 3,4 转过去长度仍是 5；反例的两列长度也是一但不垂直，乘上去长度变成 6.28。"
               "第三格：平面里最多两根互相垂直的箭头，所以 6 行 2 列的矩阵只有两列那一侧能正交，叫半正交")
    y0 = f.header("先认识「正交」　——　<tspan font-weight=\"700\">垂直的箭头，和只转不拉的矩阵</tspan>",
                  "Muon 后面每一步都踩在这个词上。三格各回答一个问题，箭头都按真实坐标画",
                  [(BL, "原来的箭头"), (GR, "正交：垂直／只转不拉"), (RD, "反例：被拉长了")])

    PH = 380
    py = f.panel(0, y0, W, PH, "三个问题", BL)
    CW, GAP = 440, 20
    S = 30.0                                    # 每 1 单位多少像素

    # ══ Ⓐ ═══════════════════════════════════════════════════════
    x0 = 20
    f.t(x0 + 10, py + 30, "Ⓐ 正交 ＝ 垂直", BL, True, 16)
    ox, oy = x0 + 250, py + 250
    f.line(ox - 150, oy, ox + 150, oy, LINE, 1, arrow=False)
    f.line(ox, oy + 30, ox, oy - 170, LINE, 1, arrow=False)
    f.line(ox, oy, ox + A_U[0] * S, oy - A_U[1] * S, BL, 2.6)
    f.line(ox, oy, ox + A_V[0] * S, oy - A_V[1] * S, GR, 2.6)
    f.t(ox + A_U[0] * S + 8, oy - A_U[1] * S + 4, "(3, 4)", BL, True, 14)
    f.t(ox + A_V[0] * S - 8, oy - A_V[1] * S + 4, "(−4, 3)", GR, True, 14, "end")
    # 直角记号：沿两根箭头各走 12px
    ux, uy = A_U[0] / 5 * 12, -A_U[1] / 5 * 12
    vx, vy = A_V[0] / 5 * 12, -A_V[1] / 5 * 12
    f.path("M %.1f %.1f L %.1f %.1f L %.1f %.1f" % (ox + ux, oy + uy, ox + ux + vx, oy + uy + vy,
                                                   ox + vx, oy + vy), INK, 1.4, arrow=False)
    f.t(x0 + 20, py + 314, "对应位置相乘再相加（内积）：", GY, size=13)
    f.t(x0 + 20, py + 338, "3×(−4) ＋ 4×3 ＝ <tspan font-weight=\"700\">0</tspan>　→　垂直",
        INK, True, 15)

    # ══ Ⓑ ═══════════════════════════════════════════════════════
    x1 = x0 + CW + GAP
    f.t(x1 + 10, py + 30, "Ⓑ 正交矩阵：只转不拉", GR, True, 16)
    ox, oy, SB = x1 + 30, py + 296, 34.0
    R5 = 5 * SB
    f.path("M %.1f %.1f A %.1f %.1f 0 0 0 %.1f %.1f" % (ox + R5, oy, R5, R5, ox, oy - R5),
           GY2, 1.2, dash="4 4", arrow=False)
    f.line(ox - 10, oy, ox + 230, oy, LINE, 1, arrow=False)
    f.line(ox, oy + 10, ox, oy - 200, LINE, 1, arrow=False)
    for v, col, lab, dx, dy in ((X, BL, "(3, 4)　长度 5", 8, 2),
                                (RX, GR, "转 30°：长度还是 5", 8, 4),
                                (BX_, RD, "反例：%.2f" % BAD_LEN, 6, 4)):
        ex, ey = ox + v[0] * SB, oy - v[1] * SB
        f.line(ox, oy, ex, ey, col, 2.6)
        f.t(ex + dx, ey + dy, lab, col, True, 13.5)
    # 两个矩阵摆在右边
    mx = x1 + 306
    for i, (m, col, nm, note) in enumerate(((ROT, GR, "转 30°", "两列垂直"),
                                            (BAD, RD, "反例", "两列不垂直"))):
        my = py + 70 + i * 120
        f.t(mx, my, nm, col, True, 14)
        for r in range(2):
            f.t(mx + 4, my + 30 + r * 22, "%-7s%s" % TXT[i][r], INK, size=13.5, mono=True)
        f.box(mx - 4, my + 10, 120, 50, "none", col, 4)
        f.t(mx, my + 82, note, col, size=12.5)
    f.t(x1 + 20, py + 338, "两个矩阵每列长度都是 1，<tspan font-weight=\"700\">差别只在垂不垂直</tspan>",
        INK, True, 14)

    # ══ Ⓒ ═══════════════════════════════════════════════════════
    x2 = x1 + CW + GAP
    f.t(x2 + 10, py + 30, "Ⓒ 维数上限：平面最多两根", PU, True, 16)
    ox, oy = x2 + 110, py + 200
    f.line(ox, oy, ox + 90, oy, GR, 2.6)
    f.line(ox, oy, ox, oy - 90, GR, 2.6)
    f.line(ox, oy, ox - 64, oy - 64, RD, 2.2, dash="5 4")
    f.t(ox - 70, oy - 72, "第三根？", RD, True, 13, "end")
    f.t(ox + 96, oy + 4, "①", GR, True, 14)
    f.t(ox + 4, oy - 98, "②", GR, True, 14)
    f.t(x2 + 30, py + 248, "跟 ① 垂直就跟 ② 平行", RD, size=12.5)
    # 6×2 的长方形
    gx, gy, cw, ch = x2 + 270, py + 64, 34, 28
    for r in range(6):
        for c in range(2):
            f.box(gx + c * (cw + 3), gy + r * (ch + 3), cw, ch, "#f6f7f8", "#dadce0", 3)
    f.t(gx + cw + 2, gy - 10, "6 × 2", INK, True, 14, "middle")
    f.t(gx + 2 * cw + 18, gy + 40, "2 列：6 维里", GR, True, 12.5)
    f.t(gx + 2 * cw + 18, gy + 60, "两根，能垂直", GR, True, 12.5)
    f.t(gx + 2 * cw + 18, gy + 110, "6 行：平面里", RD, True, 12.5)
    f.t(gx + 2 * cw + 18, gy + 130, "六根，不可能", RD, True, 12.5)
    f.t(x2 + 20, py + 314, "长方形只有<tspan font-weight=\"700\">短边那一侧</tspan>能正交", INK, True, 14)
    f.t(x2 + 20, py + 338, "这叫<tspan font-weight=\"700\">半正交</tspan>", INK, True, 15)
    f._pan = None

    yb = f.band(py + PH + 20, "ok", "带走一句：正交矩阵乘上去，只改方向不改长短", [
        "Muon 想要的「更新」就是这样一块矩阵：每个方向一样使劲，没有哪个方向被拉得特别长。",
        "可正交的箭头个数有上限：最多等于维数。一块长方形矩阵，最多短边那么多个方向。",
    ])
    yb = f.src(yb + 10,
               "本课推导：转 30° 矩阵 (cos30°, −sin30°; sin30°, cos30°) ≈ (0.866, −0.5; 0.5, 0.866)，"
               "乘 (3,4) 得 (0.598, 4.964)，长度 5；反例 (1, 0.6; 0, 0.8) 乘 (3,4) 得 (5.4, 3.2)，长度 6.28。"
               "脚本现算并断言。",
               "出处：「半正交」（semi-orthogonal）一词取自 Muon 原作者博客：Newton-Schulz 迭代把更新换成"
               "「最近的半正交矩阵」（kellerjordan.github.io/posts/muon）。")
    f.save("fig4-orth.svg", yb + 14)


main()
