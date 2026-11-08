# -*- coding: utf-8 -*-
r"""专题四 · §3.3b「Muon 从零讲」第五张：拉平之后，保住了什么、丢掉了什么

2026-11-08 补。台下最常见的两个误会：「正交化把梯度改得面目全非，还算下坡吗？」
和「每个元素都变了，方向不就变了吗？」。这张图把「变的」和「不变的」分开画。

三格：
  Ⓐ 保住的三样：① 频率方向一根不动（代数恒等式，精确）；② 不会反向 —— 五轮多项式把 (0,1] 映到正数，
     所以更新和动量的内积 ＝ Σ 强度×新强度 ＞ 0，一定是下坡；③ 强度为 0 的方向还是 0，不凭空造方向。
  Ⓑ 丢掉的：强弱比例（故意的），10 ／ 1 ／ 0.01 → 0.70 ／ 0.71 ／ 0.47，排名还被轻微打乱。
  Ⓒ 「元素全变了」和「方向没变」不矛盾：[[2,1],[1,2]] 拉平后是单位矩阵 ——
     两行箭头从斜着、不等长变成互相垂直、等长（这是强度拉平的副产品），两个频率方向原样。
     右边：一块 6×6 的示意动量（强度 10／3／1／0.3／0.1／0.03）和它拉平后的样子，逐元素看很不像：
     余弦相似度只有 0.56，三成左右的格子连正负号都不一样。

数：topic04_muon_nums.py ＋ 本文件现算并断言。6×6 示意矩阵用固定种子随机生成，不是任何模型的真实动量。
"""
import numpy as np

from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE)
import topic04_muon_nums as N

W = 1400
INNER = float(np.sum(N.SCALED * N.FINAL))
assert INNER > 0 and abs(INNER - 0.77) < 0.01, INNER
assert N.p5(0.0) == 0.0

# Ⓒ 右：示意动量
_r = np.random.default_rng(0)
_U, _ = np.linalg.qr(_r.normal(size=(6, 6)))
_V, _ = np.linalg.qr(_r.normal(size=(6, 6)))
SIG = np.array([10, 3, 1, .3, .1, .03])
MOM = _U @ np.diag(SIG) @ _V.T
ORT = _U @ _V.T
COS = float(np.sum(MOM * ORT) / np.linalg.norm(MOM) / np.linalg.norm(ORT))
assert abs(COS - SIG.sum() / np.linalg.norm(SIG) / np.sqrt(6)) < 1e-9     # 余弦只看强度，有闭式
assert abs(COS - 0.56) < 0.005, COS
SIGNDIFF = float(np.mean(np.sign(MOM) != np.sign(ORT)))
assert 0.25 < SIGNDIFF < 0.35, SIGNDIFF                                     # 图上写「三成左右」
ROWS_BEFORE = N.M22
ROWS_AFTER = N.O22
_cosrow = ROWS_BEFORE[0] @ ROWS_BEFORE[1] / np.linalg.norm(ROWS_BEFORE[0]) / np.linalg.norm(ROWS_BEFORE[1])
ANG_BEFORE = float(np.degrees(np.arccos(_cosrow)))
assert abs(ANG_BEFORE - 36.87) < 0.01
assert abs(ROWS_AFTER[0] @ ROWS_AFTER[1]) < 1e-12


def heat(f, x, y, mat, cell, vmax):
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            v = mat[r, c] / vmax
            col = BL if v >= 0 else RD
            f.box(x + c * (cell + 2), y + r * (cell + 2), cell, cell, col, "#dadce0", 2)
            f.p.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="2" '
                       'fill="#fff" fill-opacity="%.2f"/>'
                       % (x + c * (cell + 2), y + r * (cell + 2), cell, cell, 1 - 0.85 * min(1, abs(v))))


def main():
    f = Fig(W, "拉平之后保住了什么、丢掉了什么。保住三样：频率方向不动；不会反向，更新和动量的内积是正的，一定是下坡；"
               "强度为零的方向还是零。丢掉的是强弱比例：10、1、0.01 变成 0.70、0.71、0.47，排名还被轻微打乱。"
               "第三格：2,1,1,2 拉平后是单位矩阵，两行从斜着不等长变成垂直等长，可两个频率方向原样；"
               "一块示意动量和它拉平后的样子逐元素看很不像，余弦相似度只有 0.56")
    y0 = f.header("拉平之后　——　<tspan font-weight=\"700\">频率方向一根不动，强弱比例故意扔掉</tspan>",
                  "绿的是保住的，红的是丢掉的；最下面一格回答「元素全变了，方向怎么会没变」",
                  [(GR, "保住"), (RD, "丢掉"), (BL, "频率方向")])

    # ══ Ⓐ ═══════════════════════════════════════════════════════
    PH = 290
    py = f.panel(0, y0, W, PH, "Ⓐ 保住的三样", GR, sub="三样都是数学上严格成立的，不是「大致」")
    CW = 440
    # ① 方向不动
    x = 30
    f.t(x, py + 34, "① 频率方向一根不动", GR, True, 16)
    for k, (lab, ox) in enumerate((("之前", x + 70), ("之后", x + 270))):
        oy = py + 150
        for d, dash in (((0.707, 0.707), None), ((0.707, -0.707), "5 3")):
            f.line(ox, oy, ox + d[0] * 80, oy - d[1] * 80, BL, 2.4, dash=dash)
        f.t(ox + 28, py + 256, lab, INK, True, 14, "middle")
    f.t(x + 190, py + 158, "＝", INK, True, 24, "middle", big=True)
    # ② 不会反向
    x = 30 + CW + 20
    f.t(x, py + 34, "② 不会反向：一定是下坡", GR, True, 16)
    f.line(x + 20, py + 120, x + 380, py + 120, GY2, 1.4, arrow=False)
    f.line(x + 200, py + 110, x + 200, py + 130, GY2, 1.4, arrow=False)
    f.t(x + 200, py + 148, "0", GY, size=13, anchor="middle")
    f.box(x + 200, py + 104, 160, 32, "#e6f4ea", GR, 4)
    f.t(x + 280, py + 126, "进来 (0,1] → 出去 ＞ 0", GR, True, 13, "middle")
    f.t(x + 110, py + 104, "不会落到这边", RD, True, 13, "middle")
    f.t(x, py + 186, "更新 · 动量 ＝ Σ 强度 × 新强度", INK, True, 14)
    f.t(x, py + 212, "＝ 0.995×0.70 ＋ 0.0995×0.71 ＋ 0.001×0.47 ≈ <tspan font-weight=\"700\">%.2f ＞ 0</tspan>"
        % INNER, INK, size=13)
    f.t(x, py + 238, "跟动量夹角小于 90°：沿着它走，loss 往下", GR, True, 13.5)
    # ③ 零还是零
    x = 30 + 2 * (CW + 20)
    f.t(x, py + 34, "③ 零还是零：不凭空造方向", GR, True, 16)
    for k, (v_in, v_out, lab) in enumerate(((0.3, N.p5(0.3), "有一点"), (0.0, 0.0, "完全没有"))):
        bx = x + 40 + k * 190
        for j, (v, nm) in enumerate(((v_in, "进来"), (v_out, "一轮后"))):
            h = v * 120
            f.box(bx + j * 60, py + 190 - h, 36, max(h, 1.5), GR if v else GY2, GR if v else GY2, 3)
            f.t(bx + j * 60 + 18, py + 212, nm, GY, size=12, anchor="middle")
            f.t(bx + j * 60 + 18, py + 182 - h, "%.2f" % v, INK, True, 12.5, "middle")
        f.t(bx + 48, py + 240, lab, INK, True, 13.5, "middle")
    f._pan = None

    # ══ Ⓑ ═══════════════════════════════════════════════════════
    PH2 = 240
    py2 = f.panel(0, py + PH + 22, W, PH2, "Ⓑ 丢掉的：强弱比例　——　故意的", RD,
                  sub="三个频率，左边是进来的强度（满刻度 10），右边是五轮之后（满刻度 1）")
    COLS = (BL, OR, PU)
    for side, (vals, top, labs, x0) in enumerate(((N.RAW, 10.0, ["10", "1", "0.01"], 120),
                                                  (N.FINAL, 1.0, ["%.2f" % v for v in N.FINAL], 560))):
        for j, v in enumerate(vals):
            h = v / top * 130
            f.box(x0 + j * 70, py2 + 180 - h, 44, max(h, 1.5), COLS[j], COLS[j], 3)
            f.t(x0 + j * 70 + 22, py2 + 172 - h, labs[j], INK, True, 14, "middle")
        f.line(x0 - 10, py2 + 180, x0 + 210, py2 + 180, GY2, 1, arrow=False)
        f.t(x0 + 100, py2 + 204, ("进来：差一千倍" if side == 0 else "五轮后：差不到两倍"), INK, True, 14, "middle")
    f.t(440, py2 + 120, "→", INK, True, 30, "middle", big=True)
    tx = 900
    f.t(tx, py2 + 70, "强的不再更强，弱的被抬起来", RD, True, 15)
    f.t(tx, py2 + 104, "排名还会被轻微打乱：", INK, True, 14)
    f.t(tx, py2 + 128, "原来最强的 ①（%.2f）现在比 ②（%.2f）还小一点" % (N.FINAL[0], N.FINAL[1]), INK, size=13.5)
    f.t(tx, py2 + 168, "这一步本来就是要它这样：", GY, size=13.5)
    f.t(tx, py2 + 192, "单步里不分大小，大家迈差不多的步子", GY, size=13.5)
    f._pan = None

    # ══ Ⓒ ═══════════════════════════════════════════════════════
    PH3 = 330
    py3 = f.panel(0, py2 + PH2 + 22, W, PH3, "Ⓒ 「元素全变了」和「方向没变」不矛盾", BL,
                  sub="变的是行箭头和元素的样子；不变的是频率方向")
    S = 70.0
    for k, (mat, lab, x0) in enumerate(((ROWS_BEFORE, "[[2,1],[1,2]]", 60), (ROWS_AFTER, "拉平后：[[1,0],[0,1]]", 360))):
        ox, oy = x0 + 20, py3 + 230
        f.line(ox - 10, oy, ox + 170, oy, LINE, 1, arrow=False)
        f.line(ox, oy + 10, ox, oy - 170, LINE, 1, arrow=False)
        # 频率方向：浅蓝虚线，两边完全一样
        f.line(ox, oy, ox + 0.707 * 2.4 * S, oy - 0.707 * 2.4 * S, BL, 1.4, dash="3 4", arrow=False)
        for r, col in ((0, OR), (1, PU)):
            f.line(ox, oy, ox + mat[r][0] * S, oy - mat[r][1] * S, col, 2.8)
        f.t(ox + mat[0][0] * S + 6, oy - mat[0][1] * S + 16, "第 1 行", OR, True, 13)
        f.t(ox + mat[1][0] * S + 8, oy - mat[1][1] * S + 4, "第 2 行", PU, True, 13)
        f.t(x0 + 10, py3 + 266, lab, INK, True, 14)
    f.t(60 + 10, py3 + 292, "两行夹角 %.0f°、长度 √5" % ANG_BEFORE, GY, size=13)
    f.t(360 + 10, py3 + 292, "两行垂直、长度都是 1", GY, size=13)
    f.t(60 + 10, py3 + 48, "蓝虚线 ＝ 频率 ① 的方向：两边一模一样（频率 ② 与它垂直，也一样）", BL, True, 13.5)
    # 右：6×6 热图
    hx = 730
    vmax = np.abs(MOM).max()
    heat(f, hx, py3 + 70, MOM, 26, vmax)
    heat(f, hx + 200, py3 + 70, ORT, 26, np.abs(ORT).max())
    f.t(hx + 82, py3 + 256, "一块动量（示意）", INK, True, 13.5, "middle")
    f.t(hx + 282, py3 + 256, "拉平之后", INK, True, 13.5, "middle")
    f.t(hx + 400, py3 + 100, "逐元素看，很不像：", INK, True, 14)
    f.t(hx + 400, py3 + 128, "余弦相似度只有 %.2f" % COS, RD, True, 14)
    f.t(hx + 400, py3 + 154, "约 %.0f%% 的格子连正负号都不同" % (SIGNDIFF * 100), RD, True, 14)
    f.t(hx + 400, py3 + 196, "可频率方向一根没变", GR, True, 14)
    f.t(hx + 400, py3 + 222, "变的只是每根推子的高低", GR, True, 14)
    f._pan = None

    yb = f.band(py3 + PH3 + 22, "ok", "带走一句：方向原样、坡照样下，只是不再「谁大听谁的」", [
        "行变得两两垂直、等长，是强度拉平顺带出来的；看元素会觉得改得面目全非，看频率会发现一根没动。",
    ])
    yb = f.src(yb + 10,
               "本课推导：五轮多项式 p 满足 p(0) ＝ 0、在 (0, 1] 上取正值（topic04_muon_nums.py 断言）；"
               "更新与动量的内积 ＝ Σ σᵢ·p⁽⁵⁾(σᵢ)，因为两者共享同一组频率方向 —— 也有断言。",
               "本课推导：余弦相似度 ＝ Σσ ／ (√n · ‖σ‖)，只由强度决定；示意矩阵强度 10／3／1／0.3／0.1／0.03 时为 0.56。"
               "「约三成格子正负号不同」随示意矩阵而变，只说明「逐元素看很不像」。",
               "出处：「保留奇异向量、扔掉奇异值大小」这一说法：Bernstein《Deriving Muon》"
               "（we keep the singular vectors of the gradient but throw away the magnitudes of the singular values）。")
    f.save("fig4-keeplose.svg", yb + 14)


main()
