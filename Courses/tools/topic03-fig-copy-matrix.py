# -*- coding: utf-8 -*-
r"""专题三 · §五「GQA 和 MLA 的分界线，在那一个矩阵里」

⭐⭐⭐ 2026-09-14 R29 新画。调研结论：**这一格全网是空的。**

   苏剑林（kexue.fm/archives/10091）把 MHA→MQA→GQA→MLA 统一成了一句话，
   但他那三篇是**纯公式论述、正文一张配图都没有**；TransMLA 的 Figure 1
   画了三格，但它把 repeat 画成「列的复制粘贴」这个**操作**，
   没有把那个**矩阵本身**画出来。

   ⭐ 所以这张图要补的就是：**把「复制」那个矩阵，一格一格地画出来。**

── 苏剑林那三句（逐字核对过 kexue.fm/archives/10091 原文）────────────

   ①「笔者认为低秩投影这个角度并不贴近本质，因为要说低秩投影的话，
      事实上只要我们将GQA的所有K、V叠在一起，就会发现GQA也相当于在做低秩投影」
   ②「所以，MLA的本质改进不是低秩投影，而是低秩投影之后的工作。」
   ③「GQA在投影之后做了什么呢？首先它将向量对半分为两份分别作为K、V，
      然后每一份又均分为 g 份，每一份复制 h/g 次，以此来"凑"够 h 个
      Attention Head所需要的K、V。**我们知道分割、复制都是简单的线性变换**，
      所以MLA的第一个想法是将这些简单的线性变换换成一般的线性变换，
      以增强模型的能力」

   他还写了 d_c = g(d_k+d_v) < d 这个条件 —— 这一条在 MHA 那一端会失效，
   见下面的 ⚠️ 归属。

── ⛔ 引用纪律（agent 核出来的坑，别踩）────────────────────────────

   ⚠️ 苏剑林的原话是「**MLA 被视为 GQA 的一般化**」，
      **不是**「MHA / MQA / GQA 都是 MLA 的特例」。后面这句是网上转述时
      放大出来的，⛔ 不要挂在他名下。本图只说「同一个位置的同一个矩阵」，
      不说谁是谁的特例。
   ⚠️ 他也没有否认 MLA 是低秩分解 —— 他在另一篇里还写过
      「从MHA的角度看，MLA是给K、V加了rank=512的LoRA」。
      准确说法是：**低秩这个描述没错，但它区分不了 GQA 和 MLA。**

── 本图自己算、自己断言的部分 ───────────────────────────────────

   取一个能一眼看完的小例子：h=4 头、d_k=2。于是每层的 K 一共 h·d_k = 8 维。
   按苏剑林的记法（行向量）：K_all(1×8) = c(1×d_c) @ W(d_c×8)。
   四个成员**列数完全相同（都是 8）**，只有行数（＝ cache 宽度）和格子内容不同：

     MHA    d_c = 8  →  W 是 8×8 的**单位阵**（根本没在复制，也没在压）
     GQA-2  d_c = 4  →  W 是 4×8，32 格里只有 **8 个 1**，其余 24 格恒为 0
     MQA    d_c = 2  →  W 是 2×8，16 格里只有 **8 个 1**
     MLA    d_c = 4  →  **同样 4×8**，但 32 格**全部可训练**

   ⭐⭐ 把 MLA 的 d_c 故意取成和 GQA-2 一样的 4：**同样的 cache、同样的矩阵形状**，
      区别只剩「格子里写的是写死的 0/1，还是学出来的实数」。

   ⭐ 脚本用 numpy 验证「复制确实就是乘这个矩阵」：
      c @ W 必须**逐元素等于** np.repeat 出来的那个结果。这不是类比，是恒等。

   ⛔ MLA 那一格**不填任何数字** —— 编几个小数放上去，读者会以为那是
      V3 的真权重。画成有深浅的色块，配一句「每一格都是学出来的」就够了。
"""
import numpy as np

from topic03_draw import (Fig, BL, OR, GR, RD, GY, PU, INK, GY2, LINE, LINE2,
                          BG2)

W_ = 1400

H, DK = 4, 2                      # 4 个头，每头 2 维 —— 小到能一格一格画完
OUT = H * DK                      # 每层的 K 一共 8 维


def selector(g):
    """GQA-g 的上投影矩阵：d_c × OUT 的 0/1 阵（d_c = g·DK）。

    第 s 个头读第 (s·g // H) 组 —— 这就是「每组复制 h/g 次」。
    """
    dc = g * DK
    W = np.zeros((dc, OUT))
    for s in range(H):
        grp = s * g // H
        for j in range(DK):
            W[grp * DK + j, s * DK + j] = 1.0
    return W


# ── 自检：「复制」确实就是乘这个矩阵，逐元素相等 ──────────────────
_rng = np.random.default_rng(0)
for _g in (1, 2, 4):
    _c = _rng.normal(size=_g * DK)                      # 一份 latent
    _by_matmul = _c @ selector(_g)                      # 走矩阵
    _by_repeat = np.repeat(_c.reshape(_g, DK), H // _g, # 走「复制」
                           axis=0).reshape(-1)
    assert np.array_equal(_by_matmul, _by_repeat), _g
assert np.array_equal(selector(H), np.eye(OUT))          # MHA 端就是单位阵
assert int(selector(2).sum()) == OUT and selector(2).shape == (4, 8)
assert int(selector(1).sum()) == OUT and selector(1).shape == (2, 8)

GQA_CELLS = selector(2).size
GQA_ONES = int(selector(2).sum())
assert (GQA_CELLS, GQA_ONES) == (32, 8)

MEM = [                      # (名字, g, 颜色, cache 宽度的说明)
    ("MHA", H, GY), ("GQA-2", 2, BL), ("MQA", 1, PU),
]


def main():
    f = Fig(W_, "把 GQA 的「复制」写成矩阵：它是一个写死的 0/1 选择阵，"
                "MHA 端退化成单位阵，MQA 端是同一列堆满；"
                "MLA 用的是同样形状的矩阵，只是 32 格全部可训练 —— "
                "分界线不是低秩，是这个矩阵里写的是 0/1 还是学出来的实数")
    f.marks = set()
    y0 = f.header(
        "「分割和复制，都是简单的线性变换」—— 那就把它画出来",
        "同一条流水线、同一个位置的同一块矩阵　·　"
        "例子取 <tspan font-weight=\"700\">4 个头、每头 2 维</tspan>，"
        "所以每层的 K 一共 8 维",
        [(BL, "写死的 1"), (GY2, "恒为 0"), (OR, "学出来的实数")])

    # ══════════ ① 流水线：只有中间那一块在变 ═══════════════════════
    PH1 = 210
    top = f.panel(0, y0, W_, PH1, "① 四个成员走的是同一条流水线", GR,
                  sub="左右两头完全一样，区别全部集中在中间那一块")

    ty = top + 54
    BOXW, BOXH = 186, 66
    seq = [(80, "输入 x", "这一层的隐向量", GY),
           (366, "压一次", "乘 W_c，得到要缓存的 c", GR),
           (652, "这一块", "从 c 还原出 4 个头的 K", OR),
           (938, "4 个头的 K", "一共 8 维，四个成员完全一样", GY)]
    for x, main, sub, col in seq:
        hot = (col is OR)
        f.box(x, ty, BOXW, BOXH, "#fff", col if hot else LINE,
              8, 2.2 if hot else 1.2)
        f.t(x + BOXW / 2.0, ty + 28, main, col if hot else INK,
            True, 16, "middle")
        f.t(x + BOXW / 2.0, ty + 50, sub, GY, False, 13, "middle")
    for x in (80, 366, 652):
        f.line(x + BOXW + 8, ty + BOXH / 2.0, x + 278, ty + BOXH / 2.0,
               GY2, 1.6)

    f.t(652 + BOXW / 2.0, ty + BOXH + 30,
        "▲ <tspan font-weight=\"700\">四个成员的区别，全在这一块矩阵里</tspan>",
        OR, False, 15, "middle")
    f.t(366 + BOXW / 2.0, ty + BOXH + 30,
        "缓存的是这里的 c，不是右边的 K", GY2, False, 14, "middle")

    # ══════════ ② 那一块矩阵，一格一格画出来 ═══════════════════════
    y1 = y0 + PH1 + 20
    PH2 = 460
    top = f.panel(0, y1, W_, PH2, "② 那一块矩阵长什么样", BL,
                  sub="行数 ＝ 要缓存的 c 有多宽 · 列数都是 8（＝ 还原出来的 K）· "
                      "蓝格写着 1，灰格恒为 0")

    CS = 26
    MTOP = top + 76

    def draw_mat(x, name, g, col, note):
        M = selector(g)
        dc = M.shape[0]
        f.t(x + OUT * CS / 2.0, MTOP - 44, name, col, True, 17, "middle")
        f.t(x + OUT * CS / 2.0, MTOP - 22,
            "c 宽 %d 　·　%d×8" % (dc, dc), GY, False, 14, "middle")
        for i in range(dc):
            for j in range(OUT):
                one = M[i, j] > 0
                f.box(x + j * CS, MTOP + i * CS, CS - 2, CS - 2,
                      col if one else "#f1f3f4", "#fff" if one else LINE2,
                      3, 1)
                f.t(x + j * CS + (CS - 2) / 2.0, MTOP + i * CS + 17,
                    "1" if one else "0", "#fff" if one else GY2,
                    one, 13, "middle")
        return dc

    # ⛔ 2026-09-14 R29 一修：原来的排法是 MHA / GQA / MQA / MLA，
    #   于是「GQA → MLA」那根箭头**横穿过 MQA 那块矩阵**。
    #   ⭐ 改成 MHA / MQA / GQA / MLA —— 把两头放在两头、
    #     把**同宽同形的 GQA 与 MLA 排在一起**，箭头就只有一小段，
    #     而且「同样 4×8」这个论点靠相邻自己就说出来了。
    X_MHA, X_MQA, X_GQA = 60, 330, 560
    _order = dict((m[0], m) for m in MEM)
    for x, key in ((X_MHA, "MHA"), (X_MQA, "MQA"), (X_GQA, "GQA-2")):
        _n, _g, _c = _order[key]
        draw_mat(x, _n, _g, _c, None)

    f.lines(60, MTOP + 8 * CS + 26, 1280, [
        "<tspan font-weight=\"700\">左边两块是两个极端</tspan>："
        "MHA 退化成单位阵 —— 一个数都没复制，也一点都没压，c 就是 K 本身；"
        "MQA 是同一份被抄了 4 遍 —— 只存 2 维，代价是四个头拿到的 K 一模一样。",
        "<tspan font-weight=\"700\">右边两块形状完全相同，都是 4×8、cache 都是 4 维</tspan>："
        "GQA-2 的 32 格里只有 <tspan font-weight=\"700\">8 个 1</tspan>，"
        "其余 24 格<tspan font-weight=\"700\">恒为 0、而且不可训练</tspan>；",
        "MLA 的同样 32 格<tspan font-weight=\"700\">全部可训练</tspan>。"
        "⭐ 这就是全部的区别 —— 不在「压不压」，在<tspan font-weight=\"700\">"
        "这块矩阵是写死的还是学出来的</tspan>。",
    ], size=15, lh=25)

    # ── MLA：同样 4×8，但格子全是学出来的。紧挨着 GQA 放 ──
    X_MLA = 860
    f.t(X_MLA + OUT * CS / 2.0, MTOP - 44, "MLA", OR, True, 17, "middle")
    f.t(X_MLA + OUT * CS / 2.0, MTOP - 22, "c 宽 4 　·　4×8", GY,
        False, 14, "middle")
    # ⛔ 不填数字：编出来的小数会被当成 V3 的真权重。用深浅表示「都是实数」。
    _sh = ["#fbe2c8", "#f6c99a", "#f2b174", "#ee9a4e"]
    for i in range(4):
        for j in range(OUT):
            f.box(X_MLA + j * CS, MTOP + i * CS, CS - 2, CS - 2,
                  _sh[(i * 3 + j * 5) % 4], "#fff", 3, 1)
    f.box(X_MLA - 8, MTOP - 8, OUT * CS + 14, 4 * CS + 12, "none", OR, 7, 2.2)
    f.t(X_MLA + OUT * CS / 2.0, MTOP + 4 * CS + 26,
        "<tspan font-weight=\"700\">同样的 4×8，同样的 cache 宽度</tspan>",
        OR, False, 15, "middle")
    f.t(X_MLA + OUT * CS / 2.0, MTOP + 4 * CS + 50,
        "只是这 32 格<tspan font-weight=\"700\">全部可训练</tspan>", OR,
        False, 15, "middle")
    f.t(X_MLA + OUT * CS / 2.0, MTOP + 4 * CS + 76,
        "（不填数字：编出来的小数会被当成真权重）", GY2, False, 13, "middle")

    f.elbow(X_GQA + OUT * CS + 10, MTOP + 2 * CS,
            X_MLA - 16, MTOP + 2 * CS, OR, 2)
    f.t((X_GQA + OUT * CS + X_MLA) / 2.0, MTOP - 68,
        "把 0 和 1 换成学出来的实数", OR, True, 15, "middle")

    # ══════════ ③ 落点 ════════════════════════════════════════════
    y2 = y1 + PH2 + 20
    PH3 = 290
    top = f.panel(0, y2, W_, PH3, "③ 所以分界线在哪", OR,
                  sub="不是「有没有低秩」，是「那个矩阵是写死的还是学出来的」")

    f.box(40, top + 16, 640, 208, "#fff", LINE, 8, 1.2)
    f.t(64, top + 46, "低秩区分不了这两个", INK, True, 16)
    f.lines(64, top + 72, 592, [
        "把 GQA 所有的 K、V 叠在一起，<tspan font-weight=\"700\">"
        "GQA 本身就是一次低秩投影</tspan> —— 这一步",
        "MHA 也好 MLA 也好，大家都在做。所以「低秩」不是分界线。",
        "",
        "⭐ 真正不同的是<tspan font-weight=\"700\">低秩之后那一步</tspan>：",
        "GQA 用<tspan font-weight=\"700\">分割 ＋ 复制</tspan>把 c 凑成 4 个头的 K，",
        "而分割和复制<tspan font-weight=\"700\">本身就是线性变换</tspan> ——",
        "MLA 只是把这个写死的变换，换成一个一般的、可学的。",
    ], size=15, lh=25)

    f.box(720, top + 16, W_ - 760, 208, "#fff", LINE, 8, 1.2)
    f.t(744, top + 46, "⛔ 但这么一换，KV cache 会涨回去", RD, True, 16)
    f.lines(744, top + 72, W_ - 808, [
        "矩阵一放开，<tspan font-weight=\"700\">四个头的 K 又各不相同了</tspan> ——",
        "要是照常缓存 K，cache 就退回 MHA 那么大，",
        "<tspan font-weight=\"700\">省的初衷当场作废</tspan>。",
        "",
        "⭐ MLA 能成立，靠的是下一小节那个恒等变换：",
        "<tspan font-weight=\"700\">只缓存 c，把上投影矩阵挪到 q 那一侧去</tspan>。",
        "这张图是「吸收」那一步的<tspan font-weight=\"700\">前提</tspan>，不是它的替代。",
    ], size=15, lh=25)

    yy = y2 + PH3 + 22
    yy = f.band(yy, "info", "一句话记住", [
        "GQA 的上投影是一个<tspan font-weight=\"700\">写死的 0/1 复制矩阵</tspan>；"
        "MLA 把同一个位置、同一个形状的矩阵<tspan font-weight=\"700\">松开让它学</tspan>。",
        "两头也在这条轴上：<tspan font-weight=\"700\">MHA ＝ 单位阵</tspan>（不复制也不压）、"
        "<tspan font-weight=\"700\">MQA ＝ 一份抄满</tspan>。",
    ])
    yy = f.band(yy + 12, "warn", "⚠️ 归属要说准，两条", [
        "① 苏剑林的原话是「<tspan font-weight=\"700\">MLA 被视为 GQA 的一般化</tspan>」，"
        "<tspan font-weight=\"700\">不是</tspan>「MHA / MQA / GQA 都是 MLA 的特例」 —— "
        "后一句是网上转述时放大的，别挂他名下。",
        "② 他也没否认 MLA 是低秩分解。准确说法是："
        "<tspan font-weight=\"700\">低秩这个描述没错，但它区分不了 GQA 和 MLA</tspan>。"
        "而且他写的 d_c &lt; d 这个条件，在 MHA 那一端会失效 —— "
        "那一端 c 就是 K、V 本身，根本没压。",
    ], fold=True)

    yy = f.src(yy + 14,
               "📌 苏剑林《缓存与效果的极限拉扯：从MHA、MQA、GQA到MLA》"
               "kexue.fm/archives/10091 —— 「低秩投影这个角度并不贴近本质」"
               "「MLA的本质改进不是低秩投影，而是低秩投影之后的工作」"
               "「我们知道分割、复制都是简单的线性变换」三句均为原文逐字。",
               "📌 图上那几个矩阵是本课按 h=4 / d_k=2 自己构造的，"
               "脚本用 numpy 断言过「c 乘这个矩阵」与「把 c 按组复制」"
               "<tspan font-weight=\"700\">逐元素相等</tspan> —— 这不是类比，是恒等。",
               "📌 MLA 那一格<tspan font-weight=\"700\">刻意不填数字</tspan>："
               "本课没有 V3 的真权重，编几个小数放上去会被当成真的。")
    f.save("fig3-copy-matrix.svg", yy + 16)


if __name__ == "__main__":
    main()
