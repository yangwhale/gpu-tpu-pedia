# -*- coding: utf-8 -*-
r"""专题四 · §3.3b「Muon 从零讲」第二张：矩阵里的「频率」

2026-11-08 补。「奇异值」「奇异方向」对普通同事是一堵墙。现场讲通靠的是均衡器：
任何矩阵 ＝ 几个互不干扰的分量（奇异值分解），每个分量 ＝ 一个输入模式 × 一个输出模式 × 强度。
方向≈频率，强度≈振幅，强度平方≈能量。

三格，一格一个问题：
  Ⓐ 一块矩阵怎么拆？ —— [[2,1],[1,2]] ＝ 3 ×（全 0.5 的那块）＋ 1 ×（±0.5 那块）；
     元素平方和 4＋1＋1＋4 ＝ 10 ＝ 3²＋1²（能量守恒）。
  Ⓑ 「频率」能单独拎出来吗？ —— 能。输入 (1,0) 拆成两个频率各 0.707，
     分别乘强度 3 和 1，再合起来正好是 (2,1)，跟直接乘矩阵一样。
  Ⓒ 频率住在哪一列？ —— 哪一列都不住：一个频率铺满每一格。
     一块 6144×2048 这样的门控矩阵最多 2048 个频率（个数来自短边维数），
     每个频率的输出模式是全体 6144 个神经元的一种配比。

数全部用 numpy 现算并断言（SVD 分解、能量守恒、拆开再合）。
刻意没画：奇异值分解的算法本身（怎么算出来的不重要，这一格只要「它存在、互不干扰」）。
Ⓒ 的 6×4 小矩阵是随手取的示意，只为了看「一个分量铺满整块」，不是任何模型的权重。
"""
import numpy as np

from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE)

W = 1400
M = np.array([[2.0, 1.0], [1.0, 2.0]])
U, S, Vt = np.linalg.svd(M)
# 统一符号：让两个输入模式都取第一分量为正（SVD 的符号是任意的）
for i in range(2):
    if Vt[i, 0] < 0:
        Vt[i] *= -1
        U[:, i] *= -1
assert np.allclose(S, [3.0, 1.0])
H = 1 / np.sqrt(2)
assert np.allclose(Vt[0], [H, H]) and np.allclose(Vt[1], [H, -H])
COMP = [S[i] * np.outer(U[:, i], Vt[i]) for i in range(2)]
assert np.allclose(COMP[0] / 3, [[0.5, 0.5], [0.5, 0.5]])
assert np.allclose(COMP[1] / 1, [[0.5, -0.5], [-0.5, 0.5]])
assert np.allclose(COMP[0] + COMP[1], M)
ENERGY = (M ** 2).sum()
assert ENERGY == 10 and np.isclose((S ** 2).sum(), 10)          # 元素平方和 ＝ 强度平方和
# 拿输入模式去乘，频率被严格滤出：M v1 ＝ 3 u1
assert np.allclose(M @ Vt[0], 3 * U[:, 0]) and np.allclose(M @ Vt[1], 1 * U[:, 1])

X = np.array([1.0, 0.0])
COEF = Vt @ X                                                     # 拆：在两个输入模式上各多少
assert np.allclose(COEF, [H, H])
OUTP = [COEF[i] * S[i] * U[:, i] for i in range(2)]               # 各自乘强度
assert np.allclose(OUTP[0], [1.5, 1.5]) and np.allclose(OUTP[1], [0.5, -0.5])
assert np.allclose(OUTP[0] + OUTP[1], M @ X) and np.allclose(M @ X, [2, 1])

# Ⓒ 一个分量铺满整块（示意矩阵，固定种子）
rng = np.random.default_rng(4)
SM = rng.normal(size=(6, 4))
su, ss, svt = np.linalg.svd(SM)
C1 = ss[0] * np.outer(su[:, 0], svt[0])
assert np.all(np.abs(C1) > 1e-3), "第一个分量必须每格都非零 —— 这一格的论点"
assert len(ss) == min(SM.shape) == 4                              # 频率个数 ＝ 短边
GATE = (6144, 2048)
assert min(GATE) == 2048


def heat(f, x, y, mat, cell, vmax, pos=BL, neg=RD):
    """画一块小矩阵：颜色深浅＝大小，蓝正红负。"""
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            v = mat[r, c] / vmax
            a = min(1.0, abs(v))
            col = pos if v >= 0 else neg
            f.box(x + c * (cell + 3), y + r * (cell + 3), cell, cell, col, "#dadce0", 3)
            # 用一个白色半透明盖子做深浅（基元没有 opacity 参数，叠白框）
            f.p.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="3" '
                       'fill="#fff" fill-opacity="%.2f"/>'
                       % (x + c * (cell + 3), y + r * (cell + 3), cell, cell, 1 - 0.85 * a))


def main():
    f = Fig(W, "矩阵里的频率。第一格：2,1,1,2 这块矩阵拆成两个互不干扰的分量，强度 3 和 1，"
               "元素平方和 10 等于强度平方和 9 加 1。第二格：输入 1,0 拆成两个频率各 0.707，"
               "分别乘 3 和 1 再合起来得到 2,1，和直接乘矩阵一样。第三格：一个频率铺满矩阵的每一格，"
               "不住在某一列；6144 乘 2048 的矩阵最多 2048 个频率，个数来自短边的维数")
    y0 = f.header("矩阵里也有「频率」　——　<tspan font-weight=\"700\">像均衡器：几个互不干扰的分量，各有各的强度</tspan>",
                  "方向 ≈ 频率，强度 ≈ 振幅，强度² ≈ 能量。数学名字叫奇异值分解；"
                  "这里只用一块 2×2 的小矩阵 [[2,1],[1,2]] 把三件事算一遍",
                  [(BL, "频率 ①：强度 3"), (OR, "频率 ②：强度 1"), (GR, "合起来")])

    # ══ Ⓐ ═══════════════════════════════════════════════════════
    PH = 250
    py = f.panel(0, y0, W, PH, "Ⓐ 一块矩阵 ＝ 几个分量叠起来", BL,
                 sub="每块格子的颜色深浅 ＝ 那个数的大小（蓝正红负）")
    cell = 56
    gx = 40
    blocks = ((M, "原矩阵", INK, None), (COMP[0], "3 × 频率 ①", BL, "＝"),
              (COMP[1], "1 × 频率 ②", OR, "＋"))
    for i, (mat, lab, col, op) in enumerate(blocks):
        bx = gx + i * 230
        if op:
            f.t(bx - 40, py + 105, op, INK, True, 26, "middle", big=True)
        heat(f, bx, py + 40, mat, cell, 2.0)
        for r in range(2):
            for c in range(2):
                f.t(bx + c * (cell + 3) + cell / 2, py + 40 + r * (cell + 3) + cell / 2 + 6,
                    ("%g" % round(mat[r, c], 2)).replace("-", "−"), INK, True, 15, "middle")
        f.t(bx + cell + 1, py + 180, lab, col, True, 15, "middle")
    # 均衡器条 ＋ 能量账
    ex = 760
    f.t(ex, py + 50, "均衡器上看：两根推子", INK, True, 15)
    for i, (sv, col, nm) in enumerate(((3.0, BL, "①"), (1.0, OR, "②"))):
        bx = ex + 30 + i * 90
        h = sv * 34
        f.box(bx, py + 182 - h, 40, h, col, col, 3)
        f.t(bx + 20, py + 176 - h, "%g" % sv, col, True, 15, "middle")
        f.t(bx + 20, py + 204, nm, col, True, 14, "middle")
    f.line(ex + 16, py + 182, ex + 200, py + 182, GY2, 1.2, arrow=False)
    f.t(ex + 250, py + 92, "能量账（平方和）：", GY, size=13.5)
    f.t(ex + 250, py + 122, "元素：4 ＋ 1 ＋ 1 ＋ 4 ＝ <tspan font-weight=\"700\">10</tspan>", INK, True, 15)
    f.t(ex + 250, py + 150, "强度：3² ＋ 1² ＝ 9 ＋ 1 ＝ <tspan font-weight=\"700\">10</tspan>", INK, True, 15)
    f.t(ex + 250, py + 184, "一分不差：能量全在这几根推子上", GR, True, 14)
    f._pan = None

    # ══ Ⓑ ═══════════════════════════════════════════════════════
    PH2 = 330
    py2 = f.panel(0, py + PH + 22, W, PH2, "Ⓑ 每个频率能单独拎出来：拆开、各乘强度、再合起来", OR,
                  sub="输入 (1, 0)，矩阵直接乘出来是 (2, 1)")
    SC = 70.0
    SL = 120.0                     # 左边那块放大画，0.5 长的分量才看得清

    def plane(ox, oy, title):
        f.line(ox - 30, oy, ox + 200, oy, LINE, 1, arrow=False)
        f.line(ox, oy + 60, ox, oy - 170, LINE, 1, arrow=False)
        f.t(ox - 20, py2 + 46, title, INK, True, 15)

    # 左：拆
    ox, oy = 90, py2 + 180
    plane(ox, oy, "① 拆：(1, 0) 在两个频率上各 0.707")
    for d, col in ((Vt[0], BL), (Vt[1], OR)):
        f.line(ox, oy, ox + d[0] * 1.25 * SL, oy - d[1] * 1.25 * SL, col, 1, dash="4 4", arrow=False)
    f.line(ox, oy, ox + COEF[0] * Vt[0][0] * SL, oy - COEF[0] * Vt[0][1] * SL, BL, 2.6)
    f.line(ox, oy, ox + COEF[1] * Vt[1][0] * SL, oy - COEF[1] * Vt[1][1] * SL, OR, 2.6)
    f.line(ox, oy, ox + X[0] * SL, oy - X[1] * SL, INK, 2.6)
    f.t(ox + X[0] * SL + 8, oy + 5, "(1, 0)", INK, True, 14)
    f.t(ox + 1.25 * SL * H + 6, oy - 1.25 * SL * H, "频率 ① 的方向", BL, size=12.5)
    f.t(ox + 1.25 * SL * H + 6, oy + 1.25 * SL * H + 4, "频率 ② 的方向", OR, size=12.5)
    f.t(ox + 0.5 * SL - 8, oy - 0.5 * SL - 8, "0.707", BL, True, 13, "end")
    f.t(ox + 0.5 * SL - 8, oy + 0.5 * SL + 16, "0.707", OR, True, 13, "end")

    # 中：各乘强度
    f.t(470, py2 + 196, "→", INK, True, 30, "middle", big=True)
    f.t(470, py2 + 230, "① × 3", BL, True, 15, "middle")
    f.t(470, py2 + 254, "② × 1", OR, True, 15, "middle")

    # 右：合
    ox, oy = 610, py2 + 240
    plane(ox, oy, "② 各乘强度，再首尾相接")
    a, b = OUTP
    f.line(ox, oy, ox + a[0] * SC, oy - a[1] * SC, BL, 2.6)
    f.line(ox + a[0] * SC, oy - a[1] * SC, ox + (a[0] + b[0]) * SC, oy - (a[1] + b[1]) * SC, OR, 2.6)
    f.line(ox, oy, ox + 2 * SC, oy - 1 * SC, GR, 3.0)
    f.t(ox + a[0] * SC - 8, oy - a[1] * SC - 4, "(1.5, 1.5)", BL, True, 13.5, "end")
    f.t(ox + 2 * SC + 10, oy - 1 * SC + 6, "(2, 1)　跟直接乘一样", GR, True, 15)

    # 最右：严格滤出
    fx = 1010
    f.box(fx, py2 + 60, 360, 200, "none", "#dadce0", 8)
    f.t(fx + 18, py2 + 92, "拿频率 ① 的方向去乘：", INK, True, 14.5)
    f.t(fx + 18, py2 + 120, "[[2,1],[1,2]] × (0.707, 0.707)", INK, size=13.5, mono=True)
    f.t(fx + 18, py2 + 146, "＝ (2.12, 2.12) ＝ <tspan font-weight=\"700\">3</tspan> × (0.707, 0.707)",
        BL, True, 13.5)
    f.t(fx + 18, py2 + 184, "只出来频率 ①，一点 ② 都不沾", INK, True, 14)
    f.t(fx + 18, py2 + 210, "（频率 ② 那个方向乘出来是 1 倍）", GY, size=13)
    f.t(fx + 18, py2 + 238, "这就是「互不干扰」", BL, True, 14)
    f._pan = None

    # ══ Ⓒ ═══════════════════════════════════════════════════════
    PH3 = 312
    py3 = f.panel(0, py2 + PH2 + 22, W, PH3, "Ⓒ 频率不住在某一列里：一个频率铺满每一格", PU,
                  sub="左边是随手取的一块 6×4 小矩阵，只画它最强的那一个分量")
    vmax = np.abs(C1).max()
    heat(f, 50, py3 + 46, C1, 30, vmax, PU, OR)
    f.t(50, py3 + 268, "一个分量：每一格都有份", PU, True, 14)
    f.t(240, py3 + 110, "没有哪一列", INK, True, 15)
    f.t(240, py3 + 136, "「是」这个频率", INK, True, 15)
    f.t(240, py3 + 176, "每一行每一列", GY, size=13.5)
    f.t(240, py3 + 198, "都是所有频率的混合", GY, size=13.5)
    # 长条矩阵 6144×2048
    rx, ry, rw, rh = 520, py3 + 48, 66, 198
    f.box(rx, ry, rw, rh, "#f3e8fd", PU, 4)
    f.t(rx + rw / 2, ry - 8, "2048", PU, True, 13, "middle")
    f.t(rx - 8, ry + rh / 2 + 5, "6144", PU, True, 13, "end")
    f.t(rx + rw + 24, ry + 30, "比如 6144×2048 这样的门控矩阵：", INK, True, 15)
    f.t(rx + rw + 24, ry + 60, "频率最多 <tspan font-weight=\"700\">2048</tspan> 个　——　个数来自短边的维数",
        INK, True, 15)
    f.t(rx + rw + 24, ry + 90, "（上一张图：平面里最多两根互相垂直的箭头，同一个道理）", GY, size=13)
    # 一个频率的输出模式：一组配比条
    f.t(rx + rw + 24, ry + 128, "每个频率的「输出模式」＝ 全体 6144 个神经元的一种配比：", INK, True, 14.5)
    bx0 = rx + rw + 30
    vals = su[:, 0]
    rng2 = np.random.default_rng(11)
    bars = rng2.normal(size=40)
    bars /= np.abs(bars).max()
    for k, v in enumerate(bars):
        h = 34 * abs(v)
        yy = ry + 176 - (h if v > 0 else 0)
        f.box(bx0 + k * 15, yy, 10, h, PU if v > 0 else OR, "none", 1)
    f.line(bx0 - 4, ry + 176, bx0 + 40 * 15, ry + 176, GY2, 1, arrow=False)
    f.t(bx0 + 40 * 15 + 12, ry + 172, "…… 一直到第 6144 个", GY, size=13)
    f.t(bx0, ry + 230, "不是某一个神经元，是<tspan font-weight=\"700\">所有神经元一起</tspan>按这个比例动",
        PU, True, 14)
    assert len(vals) == 6
    f._pan = None

    yb = f.band(py3 + PH3 + 22, "ok", "带走一句：矩阵 ＝ 几个频率各自的强度；能量就是强度的平方和", [
        "后面 Muon 动的只有「强度」那一排推子；每个频率的方向，它一根都不碰。",
    ])
    yb = f.src(yb + 10,
               "本课推导：[[2,1],[1,2]] 的奇异值分解 —— 两个频率的输入模式 (0.707, 0.707)、(0.707, −0.707)，"
               "强度 3 和 1；numpy.linalg.svd 现算并断言，拆开再合 ＝ 直接相乘也有断言。",
               "注：「频率／振幅／能量」是类比，不是傅里叶变换本身：傅里叶的频率是固定的正弦波，"
               "这里的「频率」（奇异方向）由每块矩阵自己决定。共同点只在三条：互不干扰、各有强度、能量是强度平方和。",
               "注：Ⓒ 右下那排配比条是随机数示意，不是任何模型的真实输出模式；6144×2048 只是举一个常见的门控矩阵形状。")
    f.save("fig4-freq.svg", yb + 14)


main()
