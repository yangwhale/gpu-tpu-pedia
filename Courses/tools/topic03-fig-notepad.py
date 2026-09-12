# -*- coding: utf-8 -*-
r"""专题三 · §七「一块固定大小的记事板」（2026-09-13 夜间 · R11）。

⭐⭐⭐ 这个比喻是现场给的，而且现场说了它「对于理解非常重要」：

> 「一个记事板，固定大小。有一种是疯狂往里写，后写的覆盖先写的；
>   有一种是先写的叉掉，然后再写后写的；
>   还有一种是选择性地把先写的叉掉，再写后写。」

   ⭐ 这三句话正好就是这一支的三代：
     **纯线性注意力 → delta rule → 门控 delta rule**。
   这一张图要做的，就是把这个比喻**钉到式子上**，
   让「叉掉」这个动作在代数里有一个确切的对应物。

📌 三个关键事实（都出自 DeltaNet 论文 arXiv 2406.06484 §2.1–2.2）：
   ① 纯线性注意力的递推就是 **S_t = S_{t-1} + v_t k_tᵀ** ——&nbsp;纯加法，
      **没法「释放」旧的关联**；序列一旦 **L > d**，键就开始撞车。
      ⭐ 这正是「板子只有那么大」的精确版本。
   ② delta rule：**S_t = S_{t-1} − β_t(S_{t-1}k_t − v_t)k_tᵀ**，
      改写成 **S_{t-1}(I − β k kᵀ) + β v kᵀ** ——&nbsp;
      那个 (I − β k kᵀ) **就是「叉掉」**：在 k 这个方向上按比例擦掉旧内容。
   ③ 它还有另一个读法：这是对在线回归损失 ½‖Sk − v‖² **做一步 SGD**，
      β 就是学习率。⭐⭐ 于是状态不再是一个缓存，
      而是**一个边跑边被训练的小模型**。
"""
from topic03_draw import (Fig, wpx, BL, OR, GR, RD, GY, PU, CY, INK,
                          GY2, LINE, LINE2, BG2)

W = 1400
PX, PW = [0, 470, 940], [440, 440, 460]


def main():
    def fits(y, y0, ph, who):
        assert y <= y0 + ph - 6, "%s 到 %d，面板底边 %d" % (who, y, y0 + ph)

    f = Fig(W, "一块固定大小的记事板：纯线性注意力是疯狂往里写后写盖先写，"
               "delta rule 是先把这个方向上的旧内容擦掉再写，"
               "门控版是选择性地擦；擦这个动作在式子里就是 I 减 beta k k 转置")
    f.marks = set()
    y0 = f.header(
        "一块固定大小的记事板　——　三种写法，就是这一支的三代",
        "⭐⭐ 这一张要做的事：<tspan font-weight=\"700\">把「叉掉」这个动作，"
        "钉到式子里的一个确切位置上</tspan>",
        [(RD, "纯加：谁也不擦"), (GR, "delta：先擦再写"),
         (PU, "门控：选择性地擦"), (OR, "板子的物理上限")])

    ph = 446

    # ══ ① 板子是什么 ════════════════════════════════════════════
    x, pw = PX[0], PW[0]
    py = f.panel(x, y0, pw, ph, "① 先说清楚「板子」是什么", OR,
                 sub="它就是那个固定大小的状态")

    yy = py + 26
    # 画一块板：d×d 的格子
    n = 9
    cw = 30
    bx = x + (pw - n * cw) / 2.0
    for r in range(5):
        for c in range(n):
            f.box(bx + c * cw, yy + r * 24, cw - 3, 21, BG2, LINE2, 2)
    f.t(x + pw / 2.0, yy + 5 * 24 + 18, "状态 S：一块 d × d 的板子", OR,
        True, 12.5, "middle")
    f.t(x + pw / 2.0, yy + 5 * 24 + 38,
        "大小固定，<tspan font-weight=\"700\">不随序列变长而变大</tspan>", GY, size=11.5, anchor="middle")
    yy += 5 * 24 + 54

    for lab, txt in [("写", "把一对 (k, v) 的关联记到板子上"),
                     ("读", "拿 q 去板子上查，取回一个 v")]:
        f.box(x + 22, yy, pw - 44, 44, "#fff", LINE, 8)
        f.t(x + 38, yy + 27, lab, INK, True, 13)
        f.t(x + 70, yy + 27, txt, GY, size=11.5, w=pw - 130)
        yy += 52

    yy += 4
    f.box(x + 22, yy, pw - 44, 96, "#fff", OR, 8)
    f.box(x + 22, yy, 4, 96, OR, OR, 2)
    f.box(x + 24, yy, 3, 96, "#fff", "#fff", 0)
    f.t(x + 40, yy + 25, "⭐ 板子的上限是可以写出来的", OR, True, 13)
    f.t(x + 40, yy + 49, "一共写了 L 条，而板子只有 d 个「方向」——", GY,
        size=11.5)
    f.t(x + 40, yy + 70, "<tspan font-weight=\"700\">L 一旦超过 d，就一定有东西被盖掉</tspan>。", OR,
        True, 12.5)
    f.t(x + 40, yy + 89, "论文原话叫 key「collision」", GY2, size=11)
    fits(yy + 96, y0, ph, "①")

    # ══ ② 三种写法 ══════════════════════════════════════════════
    x, pw = PX[1], PW[1]
    py = f.panel(x, y0, pw, ph, "② 三种写法", GR,
                 sub="现场那个比喻，逐句钉到式子上")

    yy = py + 22
    WAYS = [
        ("① 疯狂往里写", "S ← S ＋ v kᵀ", RD,
         "谁也不擦，新的直接叠在旧的上面",
         "⛔ 结果：互相干扰，L &gt; d 时必然撞车"),
        ("② 先叉掉，再写", "S ← S(I − β k kᵀ) ＋ β v kᵀ", GR,
         "⭐ 那个 (I − β k kᵀ) 就是「叉掉」——",
         "在 k 这个方向上，把旧内容按比例擦掉"),
        ("③ 选择性地叉", "再加一个学出来的门 α", PU,
         "擦多少、擦哪几个通道，由门决定",
         "⭐ 这就是 Gated DeltaNet / KDA 那一支"),
    ]
    for title, formula, col, l1, l2 in WAYS:
        h = 118
        f.box(x + 22, yy, pw - 44, h, "#fff", col, 8)
        f.box(x + 22, yy, 4, h, col, col, 2)
        f.box(x + 24, yy, 3, h, "#fff", "#fff", 0)
        f.t(x + 40, yy + 26, title, col, True, 13, cls="svglbl")
        f.box(x + 40, yy + 38, pw - 100, 28, BG2, LINE2, 5)
        f.t(x + 52, yy + 57, formula, INK, size=12, mono=True, w=pw - 124)
        f.t(x + 40, yy + 87, l1, GY, size=11.5, w=pw - 76)
        f.t(x + 40, yy + 107, l2, GY, size=11.5, w=pw - 76)
        yy += h + 10
    fits(yy, y0, ph, "②")

    # ══ ③ 为什么「先擦」真的更好 ════════════════════════════════
    x, pw = PX[2], PW[2]
    py = f.panel(x, y0, pw, ph, "③ 「先擦」还有个更深的读法", BL,
                 sub="它把记事板变成了一个小模型")

    yy = py + 24
    f.box(x + 22, yy, pw - 44, 116, "#fff", BL, 8)
    f.box(x + 22, yy, 4, 116, BL, BL, 2)
    f.box(x + 24, yy, 3, 116, "#fff", "#fff", 0)
    f.t(x + 40, yy + 25, "把那个式子换个角度看：", BL, True, 12.5)
    f.t(x + 40, yy + 48, "S k 是「按这个 key 现在能取出什么」", GY,
        size=11.5)
    f.t(x + 40, yy + 68, "v 是「本来该取出什么」", GY, size=11.5)
    f.t(x + 40, yy + 90, "⭐ 按两者的<tspan font-weight=\"700\">差</tspan>去改板子 ——", GY, size=11.5)
    f.t(x + 40, yy + 110, "这就是<tspan font-weight=\"700\">一步梯度下降</tspan>，β 是学习率", BL,
        True, 12.5)
    yy += 130

    f.box(x + 22, yy, pw - 44, 76, "#fff", INK, 8)
    f.t(x + 38, yy + 25, "⭐⭐ 于是状态不再是一个「缓存」", INK, True, 13,
        cls="svglbl")
    f.t(x + 38, yy + 49, "它是<tspan font-weight=\"700\">一个边跑边被训练的小模型</tspan> ——", INK,
        True, 12.5, w=pw - 76)
    f.t(x + 38, yy + 68, "损失是 ½‖S k − v‖²，每个 token 训一步", GY,
        size=11.5)
    yy += 90

    f.box(x + 22, yy, pw - 44, 100, "#fff", OR, 8)
    f.box(x + 22, yy, 4, 100, OR, OR, 2)
    f.box(x + 24, yy, 3, 100, "#fff", "#fff", 0)
    f.t(x + 40, yy + 25, "⚠️ 代价：它比纯加法难并行", OR, True, 12.5)
    f.t(x + 40, yy + 47, "纯加法的各项互不依赖，可以一起算；", GY, size=11.5)
    f.t(x + 40, yy + 67, "「先擦再写」是<tspan font-weight=\"700\">串行</tspan>的 ——", GY, size=11.5)
    f.t(x + 40, yy + 87, "论文原话：表达力与并行度的根本权衡", GY2, size=11)
    fits(yy + 100, y0, ph, "③")

    # ══ 落点带 ══════════════════════════════════════════════════
    yy = y0 + ph + 22
    yy = f.band(yy, "info", "⭐⭐ 这个比喻之所以好，是因为它把三代的差别缩到了一个动作上", [
        "三代的差别不在「怎么读」，全在<tspan font-weight=\"700\">「写之前擦不擦、擦多少」</tspan>："
        "<tspan font-weight=\"700\">不擦 → 按固定方向擦 → 学着擦</tspan>。",
        "⭐ 所以看到这一支的任何一个新名字，只要问一句："
        "<tspan font-weight=\"700\">它的「擦」是怎么决定的？</tspan>"
        "——&#160;剩下的部分，三代之间几乎没变。",
    ])

    yy = f.band(yy + 14, "warn", "别把「板子」想成一个装 token 的盒子", [
        "⚠️ 板子上存的<tspan font-weight=\"700\">不是 token，是「键到值」的映射</tspan>。"
        "所以「装不下」不是「条数超了」，而是"
        "<tspan font-weight=\"700\">不同的键在同一个方向上互相覆盖</tspan>。",
        "⭐ 这也解释了为什么 L &gt; d 是个分界："
        "<tspan font-weight=\"700\">d 个方向撑不起 L 条互不干扰的映射。</tspan>",
    ])

    yy = f.src(yy + 16,
               "递推式、key collision（L &gt; d）、delta rule ＝ Widrow-Hoff、"
               "以及「等价于对 ½‖Sk−v‖² 做一步 SGD」，均出自 DeltaNet 论文 "
               "Yang 等 arXiv 2406.06484 §2.1–2.2",
               "「表达力与并行度之间存在根本权衡」也是该文原话（§6）；"
               "「记事板」这个比喻是本课的讲法")
    f.save("fig3-notepad.svg", yy + 6)


main()
