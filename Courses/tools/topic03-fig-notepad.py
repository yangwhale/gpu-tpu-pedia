# -*- coding: utf-8 -*-
r"""专题三 · §7.0「一块固定大小的记事板」

⭐⭐⭐ 2026-09-13 **整张重画** ——&nbsp;这个比喻是现场给的，
   现场还说了它「对于理解非常重要」。上一版把它**写**出来了，
   这一版把它**画**出来：三块白板，三种写法，看一眼就懂。

> 「一个记事板，固定大小。有一种是疯狂往里写，后写的覆盖先写的；
>   有一种是先写的叉掉，然后再写后写的；
>   还有一种是选择性地把先写的叉掉，再写后写。」

   ⭐ 这三句话正好就是这一支的三代：
     **纯线性注意力 → delta rule → 门控 delta rule**。

📌 关键事实（DeltaNet 论文 arXiv 2406.06484 §2.1–2.2）：
   · 纯加法递推 **没法释放旧的关联**；序列一旦 **L > d**，键就开始撞车。
   · delta rule 的 **(I − β k kᵀ)** 就是「叉掉」：在 k 这个方向上按比例擦。
   · 它还等价于对 ½‖Sk − v‖² **做一步梯度下降**（β 就是学习率）——&nbsp;
     ⭐⭐ 于是状态不再是一个缓存，而是**一个边跑边被训练的小模型**。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, GY, PU, INK, GY2, LINE, LINE2,
                          BG2)

W = 1400


def main():
    H, DK, DV, NL = 32, 128, 128, 69          # ⚠️ 形状是示例，不是某个模型的实测
    G, DH = 8, 128                            # 对照组：GQA-8
    st = H * DK * DV * 4                      # 每层状态，fp32
    kvt = 2 * G * DH * 2                      # 每层每 token 的 KV，bf16
    cross = st // kvt
    assert cross == 512 and abs(st * NL / 2 ** 20 - 138) < 1

    f = Fig(W, "一块固定大小的记事板：三块白板三种写法 —— 疯狂往里写后写盖先写、"
               "先把这一栏叉掉再写、选择性地叉；「叉掉」在式子里就是 I 减 beta k k 转置")
    f.marks = set()
    y0 = f.header(
        "线性注意力 ＝ 一块<tspan font-weight=\"700\">固定大小的记事板</tspan>",
        "板子就那么大 ——&#160;<tspan font-weight=\"700\">三代的差别，"
        "全在「写之前擦不擦、擦多少」</tspan>",
        [(RD, "不擦"), (GR, "先擦再写"), (PU, "选择性地擦")])

    # ══════════ ① 三块白板 ══════════════════════════════════════
    PH = 386
    py = f.panel(0, y0, W, PH, "① 同一块板子，三种写法", GR,
                 sub="现场那个比喻，画出来")

    ay = py + 24
    BW, BH = 400, 190
    WAYS = [
        (RD, "① 疯狂往里写", "后写的盖住先写的", "S ← S ＋ v kᵀ",
         "⛔ 糊成一团", "mess"),
        (GR, "② 先叉掉，再写", "写之前，把这一栏擦干净", "S ← S(I − β k kᵀ) ＋ β v kᵀ",
         "✅ 干净", "erase"),
        (PU, "③ 选择性地叉", "擦多少、擦哪几栏，学出来", "再加一个学出来的门 α",
         "⭐ GDN / KDA 这一支", "gate"),
    ]
    for i, (col, name, how, eq, verdict, kind) in enumerate(WAYS):
        bx = 40 + i * 448
        f.t(bx, ay + 24, name, col, True, 25)
        f.t(bx, ay + 50, how, GY, size=17)
        # 白板
        f.box(bx, ay + 62, BW, BH, "#fff", col, 10)
        for r in range(4):
            for c in range(6):
                cx, cyy = bx + 22 + c * 62, ay + 82 + r * 42
                if kind == "mess":
                    # 一层盖一层：三条不同颜色的划痕叠在一起
                    for k in range(3):
                        f.line(cx, cyy + 6 + k * 5, cx + 46, cyy + 26 - k * 6,
                               RD if k == 0 else ("#f6aea6" if k == 1 else GY2),
                               2.0, arrow=False)
                elif kind == "erase":
                    if c == 2:                 # 这一栏被擦干净，重新写
                        f.box(cx - 4, cyy - 4, 54, 34, "#e6f4ea", "none", 4)
                        f.line(cx, cyy + 14, cx + 46, cyy + 14, GR, 2.4,
                               arrow=False)
                    else:
                        f.line(cx, cyy + 14, cx + 46, cyy + 14, GY2, 2.0,
                               arrow=False)
                else:
                    # 选择性：每一栏擦掉的程度不同
                    frac = [1.0, .25, .7, .1, .9, .45][c]
                    f.line(cx, cyy + 14, cx + 46 * frac, cyy + 14, PU, 2.4,
                           arrow=False)
                    if frac < .99:
                        f.line(cx + 46 * frac, cyy + 14, cx + 46, cyy + 14,
                               LINE2, 2.0, arrow=False)
        f.t(bx, ay + 282, verdict, col, True, 20)
        f.box(bx, ay + 296, BW, 38, BG2, LINE2, 6)
        f.t(bx + 14, ay + 322, eq, INK, size=16, mono=True, w=BW - 28)

    # ══════════ ② 「叉掉」在式子里是哪一块 ══════════════════════
    y1 = y0 + PH + 18
    PH2 = 300
    py2 = f.panel(0, y1, W, PH2, "② 「叉掉」这个动作，在式子里就是这一块",
                  BL, sub="把比喻钉到代数上")

    by = py2 + 24
    f.box(56, by + 30, 620, 96, "#fff", GR, 10)
    f.t(80, by + 84, "S ← S ( I − β k kᵀ ) ＋ β v kᵀ", INK, True, 28,
        mono=True)
    f.box(214, by + 46, 214, 62, "none", GR, 6, 2.4)
    f.t(321, by + 144, "这一块就是「叉掉」", GR, True, 21, "middle")
    f.t(321, by + 172, "在 k 这个方向上，按比例把旧的擦掉", GY, size=17,
        anchor="middle")

    f.box(716, by + 30, 644, 180, "#e8f0fe", BL, 10)
    f.t(740, by + 70, "⭐⭐ 它还有另一个读法", BL, True, 23)
    f.t(740, by + 108, "「板子上现在能取出什么」减「本来该取出什么」，", GY,
        size=18)
    f.t(740, by + 140, "按这个<tspan font-weight=\"700\">差</tspan>去改板子 ——&#160;"
        "这就是<tspan font-weight=\"700\">一步梯度下降</tspan>。", GY, size=18)
    f.t(740, by + 180, "于是状态不再是一块缓存，", BL, True, 20)
    f.t(740, by + 210, "而是<tspan font-weight=\"700\">一个边跑边被训练的小模型</tspan>。",
        BL, True, 20)
    f.t(740, by + 236, "⚠️ 前提只有两条：学习率取 β、损失是瞬时的", GY2, size=14)

    # ══════════ ③ 板子为什么装不下 ══════════════════════════════
    y2 = y1 + PH2 + 18
    PH3 = 250
    py3 = f.panel(0, y2, W, PH3, "③ 板子为什么会「装不下」——　它不是条数超了",
                  OR, sub="装的不是 token，是「键到值」的对应关系")

    ey = py3 + 22
    f.t(56, ey + 22, "板子上有 d 个「方向」，你要往里记 L 条对应关系", GY,
        True, 20)
    for i in range(8):
        ang = i
        f.line(300 + 0, ey + 150, 300 + 120 * (0.3 + 0.1 * (ang % 4)),
               ey + 150 - 90 + 26 * (ang % 5), OR if i < 5 else RD, 2.0)
    f.t(60, ey + 156, "d 个方向", OR, True, 19)
    f.t(470, ey + 96, "L 条要记的对应关系", GY, size=17)
    f.t(470, ey + 130, "⛔ L 超过 d 之后，", RD, True, 20)
    f.t(470, ey + 162, "总有两条<tspan font-weight=\"700\">指到同一个方向上</tspan> ——",
        GY, size=18)
    f.t(470, ey + 194, "它们就开始互相盖。", RD, True, 20)

    f.box(920, ey + 20, 440, 190, "#fff", INK, 10)
    f.t(944, ey + 60, "⚠️ 所以别把板子想成一个盒子", INK, True, 21)
    f.t(944, ey + 96, "它装的<tspan font-weight=\"700\">不是 token</tspan>，", GY,
        size=18)
    f.t(944, ey + 128, "是「按这个 key，该取出那个 value」", GY, size=18)
    f.t(944, ey + 158, "这样的<tspan font-weight=\"700\">对应关系</tspan>。", GY, size=18)
    f.t(944, ey + 194, "「装不下」＝ 方向不够用了，不是条数超了", GY2, size=15)

    # ══════════ 落点 ════════════════════════════════════════════
    yy = y2 + PH3 + 20
    yy = f.band(yy, "info", "⭐⭐ 这个比喻好在哪：它把三代的差别缩到了一个动作上", [
        "三代<tspan font-weight=\"700\">读的方式几乎没变</tspan>，"
        "全部差别都在「写之前擦不擦、擦多少」："
        "<tspan font-weight=\"700\">不擦 → 按固定方向擦 → 学着擦</tspan>。",
        "⭐ 所以看到这一支的任何一个新名字，只要问一句："
        "<tspan font-weight=\"700\">它的「擦」是怎么决定的？</tspan>"
        "——&#160;剩下的部分，三代之间几乎没变。",
    ])

    yy = f.band(yy + 14, "warn", "板子到底占多少字节 —— 全讲只有这一支没给过数", [
        "⛔ 算交叉点之前先把口径摊开，"
        "<tspan font-weight=\"700\">不写出来，512 就是个孤立数字</tspan>：",
        "状态 ＝ %d × %d × %d × 4 B<tspan font-weight=\"700\">(fp32)</tspan> ＝ "
        "<tspan font-weight=\"700\">%.0f MiB</tspan>　·　GQA-8 每 token 每层 ＝ "
        "2 × %d × %d × 2 B<tspan font-weight=\"700\">(bf16)</tspan> ＝ "
        "<tspan font-weight=\"700\">%.0f KiB</tspan>"
        % (H, DK, DV, st / 2.0 ** 20, G, DH, kvt / 1024.0),
        "→ 交叉点 ＝ <tspan font-weight=\"700\">%d 个 token</tspan>；"
        "⚠️ 两边精度不一样 ——&#160;状态若也按 bf16 存，交叉点就变成 %d。"
        "⛔ 只在几百 token 以内它才更贵，到 8K 时 GQA-8 的 KV 已是它的 16 倍。"
        % (cross, cross // 2),
        "⚠️ 形状是<tspan font-weight=\"700\">示例</tspan>（32 头 × 128 × 128），"
        "而 69 是 K3 的真层数（K3 实际 96 头）——&#160;"
        "所以「%.0f MiB / 请求」是<tspan font-weight=\"700\">量级示意，不是部署数</tspan>。"
        % (st * NL / 2.0 ** 20),
    ])

    yy = f.src(yy + 16,
               "递推式、key collision（L &gt; d）、delta rule ＝ Widrow-Hoff、"
               "以及「等价于对 ½‖Sk−v‖² 做一步 SGD」，均出自 DeltaNet 论文 "
               "Yang 等 arXiv 2406.06484 §2.1–2.2",
               "⚠️ 该文 §6 那句「表达力与并行度之间存在根本权衡」说的是 "
               "Recurrent DeltaNet / mesa-layer 那一批<tspan font-weight=\"700\">"
               "比 delta 更强</tspan>的模型，不是 delta 对纯加法，"
               "而且原文是带引用的 suggests",
               "⚠️ 「记事板 / 擦」是<tspan font-weight=\"700\">现场给的比喻</tspan>，"
               "不是论文措辞")
    f.save("fig3-notepad.svg", yy + 6)


main()
