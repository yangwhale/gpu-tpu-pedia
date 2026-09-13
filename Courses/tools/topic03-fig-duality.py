# -*- coding: utf-8 -*-
r"""专题三 · §7.2c「对偶：同一个东西的两种读法」

⭐⭐⭐ 2026-09-14 夜间 R32 新画。补的是本课**只在一句图注里出现过**的一个词：
   fig3-at-gallery 的图注写着「Mamba-2 主动往回退到最简的 a·I ——&#160;
   正因为退了，**才证得出跟线性注意力的对偶**，才能吃上 Tensor Core」。
   ⛔ 而「对偶」是什么，本讲**一个字都没说**。读者只知道有这么个词。

═══ 这一格要讲的那件事 ═══
§7.1 把线性注意力写成了一个 **RNN 递推**（S_t = S_{t-1}·A_t + v_t k_tᵀ）。
但同一个计算还可以整条序列一次写完：**Y = M V，其中 M = (Q Kᵀ) ∘ L**。
⭐ 两种读法**不是近似，是同一个东西** —— Mamba-2 作者的原话（逐字）：
  "naively computing the scalar structured SSM---by materializing the
   semiseparable matrix M and performing quadratic matrix-vector
   multiplication---is exactly the same as quadratic masked kernel attention."
⛔ 本脚本**当场用 numpy 把这个等号验了一遍**（两条路算出来逐元素相等），
  图上那个「＝」不是画上去的修辞。

═══ 为什么值得单独一格 ═══
⭐ 它把本讲已经立起来的三样东西接成一条线：
  ① §7.1 的递推读法（A_t 的形状）
  ② §7.4 的矩阵读法（fig3-two-brackets 那张切块的 L×L）
  ③ 「换括号」这条暗线 ——&#160;Mamba-2 管它叫
     "a different contraction ordering"（逐字，见下）。
⭐⭐ 而且它**给 R30 那根竖条一个准确的高度**：半可分矩阵的定义是
  「对角线及以下的**任意**子矩阵，秩不超过 N」，而 **N 就是状态维度**。
  ——&#160;所以 fig3-two-brackets 里块间那一整块的秩上界不是「1」，是 N。

═══ 逐字引文（全部从 arXiv 2405.21060 e-print 源码核过，不是转述）═══
- 半可分矩阵的定义（structure/*.tex, definition semiseparable-rank）：
  "A (lower triangular) matrix M is N-semiseparable if every submatrix
   contained in the lower triangular portion (i.e. on or below the diagonal)
   has rank at most N."
- Figure 2 caption：
  "...every submatrix contained on-and-below the diagonal (Blue) has rank at
   most N, **equal to the SSM's state dimension**."
- Figure 3（SMA）caption：
  "SMA constructs a masked attention matrix M = QK^T ∘ L for any structured
   matrix L, which defines a matrix sequence transformation Y = MV."
  "All instances of SMA have a dual subquadratic form induced by a different
   contraction ordering, combined with the efficient structured matrix
   multiplication by L."
- 对偶那一句（§ 1-Semiseparable Structured Masked Attention 之前）：见上。

⚠️ Mamba-2 Figure 3 明确列出的对应关系只有这几条，本图前三张按它画：
  Causal Mask → Linear Attention；Decay Mask → Retentive Network；
  1-semiseparable → 1-SS Structured Attention（即 SSD）。
  （它还列了 Toeplitz 和 DFT 两种，跟本课主线无关，不画。）
⛔ 第四张「逐通道的门」→ GLA / GDN / KDA 是**本课按同一框架的推广**，
  Mamba-2 那张图里没有这一格，图上已标明。

⛔ 图里每个数都是脚本当场算的；那个等号由 numpy 断言守着。
"""
import numpy as np

from topic03_draw import (Fig, BL, OR, GR, RD, GY, PU, INK, GY2, LINE, LINE2)

W_ = 1400

L = 6          # 示意序列长度
DK = DV = 3    # 头维（示意）


# ══ 先把「两种读法相等」当场验一遍 ═════════════════════════════════
def by_recurrence(q, k, v, a):
    """§7.1 的递推读法：一步一个状态。S 是 d_v × d_k，A_t 右乘（标量 a_t）。"""
    S = np.zeros((DV, DK))
    out = np.zeros((L, DV))
    for t in range(L):
        S = S * a[t] + np.outer(v[t], k[t])
        out[t] = S @ q[t]
    return out


def decay_mask(a):
    """L_{t,s} = ∏_{r=s+1..t} a_r（下三角，含对角；对角恒为 1）。"""
    M = np.zeros((L, L))
    for t in range(L):
        for s in range(t + 1):
            M[t, s] = np.prod(a[s + 1:t + 1]) if t > s else 1.0
    return M


def by_matrix(q, k, v, a):
    """矩阵读法：M = (Q Kᵀ) ∘ L，然后 Y = M V。整条序列一次算完。"""
    return ((q @ k.T) * decay_mask(a)) @ v


_rng = np.random.default_rng(7)
_q = _rng.normal(size=(L, DK))
_k = _rng.normal(size=(L, DK))
_v = _rng.normal(size=(L, DV))
_a = _rng.uniform(0.5, 0.99, size=L)
_a[0] = 0.0        # 第 0 步没有历史，a_0 乘的是零状态，取值无所谓
assert np.allclose(by_recurrence(_q, _k, _v, _a),
                   by_matrix(_q, _k, _v, _a)), "对偶没验过就别画那个等号"
# 退化情形也要成立：a ≡ 1 → 普通线性注意力（L 是全 1 下三角）
_one = np.ones(L)
assert np.allclose(decay_mask(_one), np.tril(np.ones((L, L))))
assert np.allclose(by_recurrence(_q, _k, _v, _one),
                   by_matrix(_q, _k, _v, _one))
# a ≡ γ 常数 → L_{t,s} = γ^(t−s)，即 RetNet 的衰减表
_g = 0.8
_gam = np.full(L, _g)
_LM = decay_mask(_gam)
for t in range(L):
    for s in range(t + 1):
        assert abs(_LM[t, s] - _g ** (t - s)) < 1e-12, (t, s)
# 复杂度对照：递推 O(L·d²)，矩阵 O(L²·d) —— 当场算出这组示意值
COST_REC = L * DK * DV
COST_MAT = L * L * DV
assert (COST_REC, COST_MAT) == (54, 108)

CS = 24
GAMMA = 0.8
_rng2 = np.random.default_rng(19)


def shade(v):
    """把 0~1 的权重映成一格底色：越接近 1 越深。"""
    v = max(0.0, min(1.0, v))
    lo = np.array([236, 240, 253.0])      # 近白
    hi = np.array([66, 133, 244.0])       # BL
    c = lo + (hi - lo) * v
    return "#%02x%02x%02x" % tuple(int(round(x)) for x in c)


def main():
    f = Fig(W_, "线性注意力的两种读法是同一个东西："
               "一步一个状态的递推读法，和整条序列一次算完的矩阵读法；"
               "矩阵读法里那张下三角的 L 换一张，就换一个架构；"
               "对偶换来的不是美感，是能不能上 Tensor Core")
    f.marks = set()
    y0 = f.header(
        "对偶：同一个东西的两种读法",
        "本讲说过一次「Mamba-2 退回去才证得出对偶」——&#160;"
        "<tspan font-weight=\"700\">对偶到底是什么</tspan>，这一格补上",
        [(GR, "递推读法"), (BL, "矩阵读法"), (OR, "换 L 就换架构")])

    # ══════════ ① 两种读法 ══════════════════════════════════════
    PH1 = 360
    top = f.panel(0, y0, W_, PH1, "① 同一个计算，两种写法", PU,
                  sub="不是近似，是逐元素相等 ——　这个等号由脚本当场验过")

    # 左：递推
    f.t(60, top + 44, "读法甲 · 一步一个状态（§7.1 那条）", GR, True, 17)
    BX, BY = 60, top + 74
    for t in range(L):
        x = BX + t * 88
        f.box(x, BY, 62, 46, "#fff", GR, 7, 1.6)
        f.t(x + 31, BY + 29, "S%d" % (t + 1), GR, True, 15, "middle")
        if t:
            f.line(x - 24, BY + 23, x - 4, BY + 23, GR, 1.6)
    f.lines(60, BY + 66, 540, [
        "<tspan font-weight=\"700\">S_t = S_{t-1} · A_t + v_t k_tᵀ</tspan>，"
        "读出 <tspan font-weight=\"700\">out_t = S_t · q_t</tspan>",
        "一次只动一个状态，<tspan font-weight=\"700\">必须按顺序走</tspan>。",
        "代价 <tspan font-weight=\"700\">O(L · d²)</tspan>"
        "（本例 %d 次乘加）——&#160;跟句长成正比。" % COST_REC,
    ], size=15, lh=25)

    # 中间的等号
    f.t(648, BY + 34, "＝", INK, True, 34, "middle")
    f.t(648, BY + 66, "不是近似", GY, False, 14, "middle")

    # 右：矩阵
    MX, MY = 740, top + 74
    f.t(740, top + 44, "读法乙 · 整条序列一次算完", BL, True, 17)
    for i in range(L):
        for j in range(L):
            x, y = MX + j * CS, MY + i * CS
            if j <= i:
                f.box(x, y, CS - 2, CS - 2, shade(GAMMA ** (i - j)),
                      "#fff", 2, 1)
            else:
                f.box(x, y, CS - 2, CS - 2, "#f6f7f9", "#fff", 2, 1)
    f.box(MX - 6, MY - 6, L * CS + 10, L * CS + 10, "none", BL, 6, 1.8)
    f.t(MX + L * CS + 20, MY + 20, "M = (Q Kᵀ) ∘ L", BL, True, 16)
    f.t(MX + L * CS + 20, MY + 46, "Y = M V", BL, True, 16)
    f.t(MX + L * CS + 20, MY + 76, "整张摆出来，一次矩阵乘", GY, False, 14)
    f.t(MX + L * CS + 20, MY + 100,
        "代价 O(L² · d)（本例 %d 次）" % COST_MAT, GY, False, 14)

    # ⚠️ 这条引文放右栏（w=620）时宽度断言不过，改成整栏一条；
    #   左栏文字到 top+215 就结束了，这里从 top+242 起不会撞。
    f.lines(60, MY + L * CS + 24, 1280, [
        "⭐ Mamba-2 作者的原话（逐字）：把那个矩阵摆出来、做一次二次型乘法，"
        "「<tspan font-weight=\"700\">is exactly the same as quadratic "
        "masked kernel attention</tspan>」。",
        "不是像，是<tspan font-weight=\"700\">同一个</tspan>。"
        "两边的代价却差得很远：递推 O(L·d²) 随句长线性涨，"
        "矩阵 O(L²·d) 随句长平方涨 ——&#160;"
        "<tspan font-weight=\"700\">同一个结果，两种算法</tspan>。",
    ], size=15, lh=25)

    # ══════════ ② 换 L 就换架构 ══════════════════════════════════
    y1 = y0 + PH1 + 20
    PH2 = 396
    top = f.panel(0, y1, W_, PH2, "② 那张 L 换一张，就换一个架构", OR,
                  sub="M =（Q Kᵀ）∘ L　——　变的只有 L")

    def mini(x, y, fn, title, sub, who, col, note=None, stack=0):
        # ⭐ stack>0：在后面扇出几张同样大小的空壳，表示「每个通道各有一张 L」。
        #   ⚠️ 只往右扇、不往上扇 ——&#160;往上会撞到标题。
        for s in range(stack, 0, -1):
            f.box(x - 6 + s * 9, y - 6, L * CS + 10, L * CS + 10,
                  "#fff", col, 6, 1.2)
        for i in range(L):
            for j in range(L):
                cx, cy = x + j * CS, y + i * CS
                if j <= i:
                    f.box(cx, cy, CS - 2, CS - 2, shade(fn(i, j)), "#fff", 2, 1)
                else:
                    f.box(cx, cy, CS - 2, CS - 2, "#f6f7f9", "#fff", 2, 1)
        f.box(x - 6, y - 6, L * CS + 10, L * CS + 10, "none", col, 6, 1.8)
        f.t(x + L * CS / 2.0 - 3, y - 18, title, col, True, 16, "middle")
        f.t(x + L * CS / 2.0 - 3, y + L * CS + 24, sub, GY, False, 14, "middle")
        f.t(x + L * CS / 2.0 - 3, y + L * CS + 48, who, col, True, 15, "middle")
        if note:
            f.t(x + L * CS / 2.0 - 3, y + L * CS + 72, note, GY2, False, 13,
                "middle")

    MY2 = top + 64
    XS = 92
    mini(XS, MY2, lambda i, j: 1.0, "全 1 下三角",
         "L ≡ 1，一点不衰减", "线性注意力", BL)
    mini(XS + 330, MY2, lambda i, j: GAMMA ** (i - j), "标量指数衰减",
         "L = γ^(t−s)，全局一个 γ", "RetNet", PU)
    mini(XS + 660, MY2, lambda i, j: float(np.prod(_a[j + 1:i + 1]))
         if i > j else 1.0, "每步一个 a_t 连乘",
         "1-semiseparable", "Mamba-2 / SSD", GR)
    # ⚠️ 第四张故意用另一组门值：它跟第三张「形状一样、数不一样」——
    #   差别在于 d 个通道各有各的一张，不是这一张本身长得特别。
    _a2 = _rng2.uniform(0.3, 0.99, size=L)
    _a2[0] = 0.0
    mini(XS + 990, MY2,
         lambda i, j: float(np.prod(_a2[j + 1:i + 1])) if i > j else 1.0,
         "每个通道一把门", "不是一张 L，是每维一张", "GLA / GDN / KDA", OR,
         note="⚠️ 本课按同一框架的推广", stack=2)

    f.lines(60, MY2 + L * CS + 108, 1280, [
        "⭐ 前三张是 Mamba-2 那张 SMA 图<tspan font-weight=\"700\">明确列出的对应</tspan>"
        "（causal mask → 线性注意力、decay mask → RetNet、"
        "1-semiseparable → SSD）；",
        "第四张是<tspan font-weight=\"700\">本课按同一框架的推广</tspan>，"
        "原图里没有这一格 ——&#160;它的 L 不再是一个标量表，"
        "而是<tspan font-weight=\"700\">每个通道各有一张</tspan>。",
    ], size=15, lh=25)

    # ══════════ ③ 对偶换来的是什么 ═══════════════════════════════
    y2 = y1 + PH2 + 20
    PH3 = 326
    top = f.panel(0, y2, W_, PH3, "③ 这个对偶换来的不是美感，是 Tensor Core", GR,
                  sub="顺便先记一个数，§7.4 那张图要用到它")

    f.box(40, top + 18, 640, 246, "#fff", LINE, 8, 1.2)
    f.t(64, top + 50, "为什么 Mamba-2 要往回退", GR, True, 17)
    f.lines(64, top + 76, 592, [
        "A_t 越花哨，表达力越强，但那张 L 就越难被<tspan font-weight=\"700\">"
        "高效地乘</tspan>。",
        "Mamba-2 <tspan font-weight=\"700\">主动退回最简的 a·I</tspan>"
        "（一步一个标量）——",
        "退了之后 L 变成 1-semiseparable，两条路都能走：",
        "<tspan font-weight=\"700\">块内走矩阵读法吃 Tensor Core，"
        "块间走递推读法省内存</tspan>。",
        "",
        "⭐ 这就是<tspan font-weight=\"700\">「表达力和可算性是一起设计的」</tspan>"
        "那句话的",
        "具体样子 ——&#160;不是先设计一个强的，再去优化。",
    ], size=15, lh=25)

    f.box(720, top + 18, W_ - 760, 246, "#fff", OR, 8, 1.6)
    f.t(744, top + 50, "⭐ 先记一个数：这张 L 能被压多扁", OR, True, 17)
    f.lines(744, top + 76, W_ - 808, [
        "半可分矩阵的定义是 ——&#160;<tspan font-weight=\"700\">"
        "对角线及以下的任意子矩阵，秩不超过 N</tspan>；",
        "而<tspan font-weight=\"700\">这个 N 就是状态维度</tspan>。",
        "",
        "换句话说：<tspan font-weight=\"700\">L 里任何一块"
        "「不跨对角线」的子矩阵，都能被 N 维压住</tspan>。",
        "",
        "⛔ <tspan font-weight=\"700\">§7.4 会把「块间那一整块」画成一根竖条</tspan>"
        "——&#160;那根竖条的",
        "高度就是这里的 N。<tspan font-weight=\"700\">"
        "「块间能被压掉」和「状态有多大」是同一件事</tspan>。",
    ], size=15, lh=25)

    yy = y2 + PH3 + 22
    yy = f.band(yy, "info", "一句话记住", [
        "<tspan font-weight=\"700\">递推读法和矩阵读法是同一个计算</tspan>，"
        "差别只在<tspan font-weight=\"700\">先算哪一步</tspan> ——&#160;"
        "Mamba-2 管这个叫 “a different contraction ordering”，"
        "<tspan font-weight=\"700\">也就是本讲反复出现的「换括号」</tspan>。",
        "而 <tspan font-weight=\"700\">M =（Q Kᵀ）∘ L</tspan> 这个骨架下，"
        "<tspan font-weight=\"700\">你换的从来只有 L</tspan>："
        "全 1 是线性注意力、γ^(t−s) 是 RetNet、每步一个 a_t 是 SSD、"
        "每通道一把门是 GLA 那一支。",
    ])
    yy = f.band(yy + 12, "warn", "两处别讲过头", [
        "① 这个等号<tspan font-weight=\"700\">只对「A_t 是标量乘单位阵」这一支"
        "严格成立</tspan>（本图验的就是这一支）。"
        "A_t 一旦是一般矩阵，L 就不再是一张标量表，"
        "<tspan font-weight=\"700\">对偶还在，但那张 L 要按通道展开</tspan>。",
        "② 「换 L 就换架构」是一个<tspan font-weight=\"700\">整理框架</tspan>，"
        "不是说这些模型都是从这个框架推出来的 ——&#160;"
        "RetNet、GLA 都比 Mamba-2 的这套说法更早，"
        "<tspan font-weight=\"700\">是框架回头把它们收进来的</tspan>。",
    ])

    yy = f.src(yy + 14,
               "📌 三句逐字引文均核自 arXiv 2405.21060 的 e-print 源码："
               "半可分矩阵的定义（「every submatrix contained in the lower "
               "triangular portion ... has rank at most N」）、"
               "Figure 2 caption 里的「equal to the SSM's state dimension」、"
               "以及对偶那一句「is exactly the same as quadratic masked "
               "kernel attention」。",
               "📌 SMA 骨架 M = QK^T ∘ L 与 “a different contraction ordering” "
               "出自同文 Figure 3 caption。⚠️ 前三张 L 的对应关系照它画；"
               "第四张（逐通道门 → GLA / GDN / KDA）是本课的推广，原图没有。",
               "⛔ 图里那个「＝」不是修辞：本脚本用 numpy 把递推读法和矩阵读法"
               "各算一遍并断言逐元素相等，还额外验了两个退化情形"
               "（a ≡ 1 → 全 1 下三角；a ≡ γ → γ^(t−s)）。")

    f.save("fig3-duality.svg", yy + 10)


if __name__ == "__main__":
    main()
