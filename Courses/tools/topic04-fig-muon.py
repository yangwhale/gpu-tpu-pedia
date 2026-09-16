# -*- coding: utf-8 -*-
r"""专题四 · §3.4「Muon：同一个动作，在三种形状上的三个版本」

⭐⭐⭐ 2026-09-17 新画。现场要求：**去搜社区里的高手讲法，学习、吸收、蒸馏，
  然后多画图少说话。** 这张图是第一批产物。

📌 讲法来源（**只借思路，图是我们自己重画的**）：
  苏剑林《Muon 优化器赏析：从向量到矩阵的本质跨越》，科学空间
  spaces.ac.cn/archives/10592
  从那篇里吸收了四条，每一条都比我原来的写法好：
  ① **msign 是 sign 的矩阵推广** —— 标量 / 对角阵 / 一般矩阵三个特例串起来讲
  ② **「矩阵和向量有什么区别」用「迹」来回答** —— 对角线与非对角线地位不对等
  ③ **范数视角** —— 同一个约束优化问题，F 范数给出 SGD，谱范数给出 Muon
  ④ **为什么几乎不花时间** —— 矩阵乘落在「梯度算完、下一个还没来」的空窗里

⭐⭐ 这张图的取舍：**Ⓐ 三列必须并排**。
  ⛔ 分开讲就变成三个知识点；并排放，读者自己会看出「是同一个动作」——
    而那正是这张图唯一要送出的东西。
  ⭐ 判据：**「A 是 B 的推广」这种话，并排画比说一百遍都管用。**

⛔⛔ 刻意没画的：
  ① **Newton-Schulz 迭代的展开式。** 它是「怎么快速算 msign」，不是「msign 是什么」。
  ② **收敛速度对照。** 本课没有实测，画就是编（跟 fig4-optimizers 同一条规矩）。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE, LINE2)

W = 1400

BETA1 = 0.9
UPDATE_RMS = ((1 - BETA1) / (1 + BETA1)) ** 0.5     # ⭐ 那个 0.2 的闭式解
assert 0.229 < UPDATE_RMS < 0.230

SVD_BEFORE = (1.0, 0.62, 0.33, 0.12)                # 奇异值：高低不齐
ADAM_BYTES, MUON_BYTES = 16, 12


def main():
    f = Fig(W, "Muon 做的事叫矩阵符号函数，它是标量符号函数的推广："
               "对标量是除掉大小只留正负，对对角阵退化成逐元素取符号，"
               "对一般矩阵则是做奇异值分解、把所有奇异值拉平到一。"
               "从范数的角度看，同一个「这一步不许迈太大」的约束问题，"
               "用 F 范数量得到 SGD，用谱范数量得到的就是 Muon。"
               "而它多出来的矩阵乘法几乎不花时间，"
               "因为它落在梯度算完、下一个梯度还没开始那段空窗里")

    y0 = f.header(
        "Muon　——　<tspan font-weight=\"700\">"
        "同一个动作「扔掉大小，只留方向」，在三种形状上的三个版本</tspan>",
        "⭐ 三列<tspan font-weight=\"700\">并排看</tspan>，就会发现它们是一件事"
        "　·　⛔ 分开讲就变成三个知识点",
        [(GY2, "标量"), (BL, "对角阵"), (PU, "一般矩阵")])

    # ══════════ Ⓐ 三个特例并排 ═══════════════════════════════════
    PH = 400
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 为什么它配叫「矩阵版的符号函数」", BL,
                 sub="⭐ 每一列都是：<tspan font-weight=\"700\">左边原样，右边只剩方向</tspan>")

    CW = 440
    for i, (col, fill, tag, sub1) in enumerate((
            (GY2, "#f1f3f4", "标量", "sign(x)　＝　除掉大小，只留正负"),
            (BL, "#e8f0fe", "对角阵", "逐元素取 sign　——　此时 Muon ＝ 带动量的 SignSGD"),
            (PU, "#f3e8fd", "一般矩阵", "SVD 之后，<tspan font-weight=\"700\">所有奇异值拉平到 1</tspan>"))):
        x = 28 + i * (CW + 12)
        f.box(x, py + 34, CW, 318, fill, col, 8)
        f.box(x, py + 34, CW, 4, col, col, 2)
        f.t(x + CW / 2.0, py + 70, tag, col, True, 20, "middle")

        cx = x + CW / 2.0
        if i == 0:
            # 数轴：一个点被拉到 ±1
            f.line(x + 60, py + 150, x + CW - 60, py + 150, GY2, 1.6, arrow=False)
            for v, lab in ((-1.0, "−1"), (0.0, "0"), (1.0, "＋1")):
                xx = cx + v * 130
                f.line(xx, py + 144, xx, py + 156, GY2, 1.2, arrow=False)
                f.t(xx, py + 176, lab, GY2, size=12, anchor="middle")
            f.box(cx + 0.42 * 130 - 6, py + 144, 12, 12, INK, INK, 6)
            f.t(cx + 0.42 * 130, py + 126, "0.42", INK, True, 13, "middle")
            f.path("M %.1f %.1f Q %.1f %.1f %.1f %.1f"
                   % (cx + 0.42 * 130, py + 118, cx + 0.7 * 130, py + 96,
                      cx + 130, py + 138), col, 1.8)
            f.t(cx, py + 216, "不管它原来是 0.42 还是 42", GY, size=13.5, anchor="middle")
            f.t(cx, py + 242, "出来都是 ＋1", col, True, 16, "middle")
        elif i == 1:
            # 对角阵：三个对角元各自拉到 ±1
            for k, v in enumerate((0.8, -0.3, 0.5)):
                yy = py + 120 + k * 44
                f.t(x + 78, yy + 6, "%+0.1f" % v, INK, True, 15, "middle")
                f.line(x + 110, yy, x + 172, yy, GY2, 1.4)
                f.t(x + 206, yy + 6, "%+d" % (1 if v > 0 else -1), col, True, 17, "middle")
            f.box(x + 250, py + 104, 150, 140, "#fff", col, 6)
            f.t(x + 325, py + 132, "对角线上", col, True, 14, "middle")
            f.t(x + 325, py + 156, "各管各的", GY, size=13, anchor="middle")
            f.t(x + 325, py + 190, "＝ 拍平成向量", GY, size=13, anchor="middle")
            f.t(x + 325, py + 214, "也没差", GY, size=13, anchor="middle")
            f.t(cx, py + 278, "⭐ 这一格是「向量做法」的极限", GY, True, 14, "middle")
        else:
            # 奇异值柱：高低不齐 → 全部拉平
            BW2, GAP = 26, 16
            for grp, (vals, lab, ccol) in enumerate((
                    (SVD_BEFORE, "原来的奇异值", GY2),
                    ((1.0, 1.0, 1.0, 1.0), "全部拉到 1", PU))):
                bx = x + 44 + grp * 230
                for k, v in enumerate(vals):
                    h = 96 * v
                    f.box(bx + k * (BW2 + GAP), py + 220 - h, BW2, h,
                          "#fff" if grp == 0 else "#f3e8fd", ccol, 3, 1.4)
                f.line(bx - 6, py + 220, bx + 4 * (BW2 + GAP), py + 220,
                       GY2, 1.4, arrow=False)
                f.t(bx + 2 * (BW2 + GAP) - 8, py + 246, lab, ccol, True, 13.5, "middle")
            f.line(x + 218, py + 172, x + 262, py + 172, PU, 2.0)
            f.t(cx, py + 278, "⭐⭐ 方向全留下，大小全扔掉", PU, True, 15, "middle")
    f._pan = None

    # ══════════ Ⓑ 矩阵凭什么不能拍平 ═════════════════════════════
    PH2 = 230
    py2 = f.panel(0, py + PH + 22, W, PH2,
                  "Ⓑ ⭐⭐⭐ 「矩阵和向量不都是一堆数字吗」——　"
                  "<tspan font-weight=\"700\">一个例子就能说清</tspan>", PU,
                  sub="⛔ 拍平成向量，抹掉的正是这件事")

    f.box(70, py2 + 34, 560, 158, "#f3e8fd", PU, 8)
    f.t(350, py2 + 68, "矩阵有「迹」——　对角线元素之和", PU, True, 17, "middle")
    f.t(96, py2 + 104, "⭐ 它不是瞎定义的：<tspan font-weight=\"700\">相似变换下保持不变</tspan>，",
        GY, size=14.5)
    f.t(96, py2 + 130, "而且<tspan font-weight=\"700\">等于所有特征值之和</tspan>。", GY, size=14.5)
    f.t(350, py2 + 172, "⭐⭐ 也就是说：对角线和非对角线，<tspan font-weight=\"700\">地位不对等</tspan>",
        INK, True, 15, "middle")

    f.line(650, py2 + 112, 690, py2 + 112, GY2, 1.8)

    f.box(710, py2 + 34, 630, 158, "#fce8e6", RD, 8)
    f.t(1025, py2 + 68, "而「拍平成一个大向量」做的事", RD, True, 17, "middle")
    f.t(1025, py2 + 106, "就是宣布：所有位置一律平等", INK, True, 17, "middle")
    f.t(1025, py2 + 146, "⛔ SGD / Adam 都是这么干的 ——　它们是逐元素的",
        GY, size=14, anchor="middle")
    f.t(1025, py2 + 172, "⭐ 而 Muon 拒绝拍平", RD, True, 15, "middle")
    f._pan = None

    # ══════════ Ⓒ 范数视角：换把尺子就换个优化器 ═════════════════
    PH3 = 278
    py3 = f.panel(0, py2 + PH2 + 22, W, PH3,
                  "Ⓒ ⭐⭐ 换一把尺子，就换一个优化器", GR,
                  sub="⭐ 同一个问题：<tspan font-weight=\"700\">"
                      "「这一步不许迈太大」的前提下，让 loss 降得最多</tspan>")

    f.box(90, py3 + 40, 560, 82, "#f1f3f4", GY2, 8)
    f.t(370, py3 + 72, "用 F 范数量「迈多大」", GY, True, 17, "middle")
    f.t(370, py3 + 100, "（＝把矩阵拍平，算欧氏长度）", GY2, size=13, anchor="middle")
    f.t(700, py3 + 76, "→", GY2, True, 22, "middle")
    f.box(750, py3 + 40, 560, 82, "#f1f3f4", GY2, 8)
    f.t(1030, py3 + 80, "得到的就是　SGD", INK, True, 19, "middle")

    f.box(90, py3 + 138, 560, 82, "#e6f4ea", GR, 8)
    f.t(370, py3 + 170, "用谱范数量「迈多大」", GR, True, 17, "middle")
    f.t(370, py3 + 198, "（＝由「矩阵乘向量」这个动作诱导出来的）", GY2, size=13, anchor="middle")
    f.t(700, py3 + 174, "→", GR, True, 22, "middle")
    f.box(750, py3 + 138, 560, 82, "#e6f4ea", GR, 8)
    f.t(1030, py3 + 178, "得到的正是　Muon", GR, True, 19, "middle")

    f.t(90, py3 + 244, "⭐ 为什么谱范数更贴矩阵：<tspan font-weight=\"700\">"
                       "矩阵在网络里干的活，本来就是乘向量</tspan>", GY, size=14.5)
    f._pan = None

    # ══════════ Ⓓ 为什么几乎不花时间 ═════════════════════════════
    PH4 = 298
    py4 = f.panel(0, py3 + PH3 + 22, W, PH4,
                  "Ⓓ ⭐ 「每步多做十几次矩阵乘，不会很慢吗」——　"
                  "<tspan font-weight=\"700\">它塞进了一段本来就空着的时间</tspan>", OR,
                  sub="⚠️ 实测每步增加 <tspan font-weight=\"700\">5% 以内</tspan>，"
                      "作者自己声称 2%")

    TX0, TX1 = 110, 1320
    ty = py4 + 60
    SEGS = ((0.00, 0.40, BL, "#e8f0fe", "算这一步的梯度", "算力满载"),
            (0.40, 0.56, OR, "#fef7e0", "空窗", "算力几乎闲着"),
            (0.56, 1.00, BL, "#e8f0fe", "算下一步的梯度", "算力满载"))
    for a, b, col, fill, nm, note in SEGS:
        x0 = TX0 + a * (TX1 - TX0)
        w = (b - a) * (TX1 - TX0)
        f.box(x0, ty, w, 66, fill, col, 6)
        f.t(x0 + w / 2.0, ty + 32, nm, col, True, 16, "middle")
        f.t(x0 + w / 2.0, ty + 54, note, GY, size=12.5, anchor="middle")
    f.box(TX0 + 0.40 * (TX1 - TX0), ty + 86, 0.16 * (TX1 - TX0), 54, "#fff", OR, 6, 2.0)
    f.t(TX0 + 0.48 * (TX1 - TX0), ty + 118, "msign 就干在这儿", OR, True, 16, "middle")
    f.t(TX0, ty + 176, "⭐ 而且这些矩阵乘<tspan font-weight=\"700\">尺寸固定、可以并行</tspan> ——　"
                       "不是那种会卡住流水线的活", GY, size=14.5)
    f.t(TX0, ty + 204, "⭐⭐ 所以它花的是<tspan font-weight=\"700\">本来就闲着的算力</tspan>，"
                       "换来的是<tspan font-weight=\"700\">每参数少 %d 字节</tspan>（%d → %d）"
                       % (ADAM_BYTES - MUON_BYTES, ADAM_BYTES, MUON_BYTES),
        INK, True, 15)
    f._pan = None

    yy = f.band(py4 + PH4 + 22, "bad", "代价有两条，第二条很少被提到", [
        "⛔ ① <tspan font-weight=\"700\">它挑食</tspan>：不成矩阵的那些参数它处理不了，"
        "embedding 和输出头得留给 AdamW 兜底。",
        "⛔⛔ ② <tspan font-weight=\"700\">它不是逐元素的，这会捅到并行</tspan>。"
        "逐元素的优化器有个隐形好处：<tspan font-weight=\"700\">一个大矩阵切成两半分到两张卡，"
        "各自独立更新，轨迹一点不变</tspan>。Muon 不行 ——&#160;"
        "<tspan font-weight=\"700\">切开的几块得先把梯度汇聚起来</tspan>。"
        "⚠️ 而且不用并行也会踩：多头注意力常常是<tspan font-weight=\"700\">一个大矩阵 reshape 出多个头</tspan>，"
        "参数里只有一个矩阵，本质上却是多个小矩阵。",
    ], fold=True)

    yy = f.src(yy + 24,
               "⭐ <tspan font-weight=\"700\">讲法来源</tspan>：苏剑林《Muon 优化器赏析："
               "从向量到矩阵的本质跨越》，科学空间 spaces.ac.cn/archives/10592。"
               "Ⓐ 的三个特例、Ⓑ 的「迹」那个例子、Ⓒ 的范数视角、Ⓓ 的空窗解释，"
               "<tspan font-weight=\"700\">四条都取自该文</tspan> ——&#160;"
               "⛔ <tspan font-weight=\"700\">图是我们自己重画的，不是搬运</tspan>",
               "⚠️ Ⓒ 那个范数视角该文另引《Old Optimizer, New Norm: An Anthology》；"
               "⭐ 该文还考了一笔源流：<tspan font-weight=\"700\">2015 年的 Stochastic Spectral "
               "Descent 已经提出过大致相同的算法</tspan>",
               "⛔ 本图<tspan font-weight=\"700\">不含任何收敛速度对照</tspan> ——&#160;"
               "本课没有这些优化器的对照实测，跟 fig4-optimizers 同一条规矩",
               "⚠️ Ⓓ 的「5% 以内 / 作者称 2%」是<tspan font-weight=\"700\">转述</tspan>，"
               "不是我们量的；真要用请自己在目标配置上量一遍"
               "（<tspan font-weight=\"700\">这正是本讲那条「不能跨规模照抄」</tspan>）")
    f.save("fig4-muon.svg", yy + 6)


main()
