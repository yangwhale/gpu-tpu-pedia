# -*- coding: utf-8 -*-
r"""专题四 · §3.3b「Muon 从零讲」第七张：为什么有用，以及它从哪儿来

2026-11-08 补。两格：
  Ⓐ 为什么有用（两条独立的理由）：
     · 观察（Keller Jordan 博客）：SGD 动量和 Adam 给二维参数算出的更新，条件数很高、几乎是低秩的 ——
       所有神经元的更新被少数几个方向主导；正交化把那些幅度小但对学习重要的「稀有方向」抬起来（作者原话是推测）。
     · 推导（Bernstein《Deriving Muon》）：要求「对任何输入，这一层的输出变化不超过一个上限」，
       在这个约束下让 loss 降得最多的更新，就是保留奇异方向、扔掉奇异值大小 —— 正交化的梯度。
  Ⓑ 家谱，两条支：
     逐元素支：Rprop（Riedmiller & Braun 1993，只用梯度的正负号）→ RMSProp（Hinton 课件 2012）／Adam（2014），
       除以「典型大小」→ signSGD（Bernstein 等 2018，只用正负号）。
     矩阵支：Stochastic Spectral Descent（Carlson、Cevher、Carin，AISTATS 2015）→ Shampoo（Gupta 等 2018；
       去掉累积就是正交化的梯度）→ Orthogonal-SGDM（Tuddenham 等 2022；先正交化再动量，用 SVD）
       → Muon（Keller Jordan 等 2024；动量放到正交化之前、Newton–Schulz、能用 BF16）
       → 月之暗面上规模（Moonlight 2025：加权重衰减、对齐整体幅度；Kimi K2 2025：QK-Clip）。
     两条支之间的桥：苏剑林指出 Muon 就是 sign 的矩阵版（msign）。

📌 全部核过原文：
  · Keller 博客 kellerjordan.github.io/posts/muon：「Why is it good to orthogonalize the update?」一节（低秩、稀有方向）；
    「Relationship to prior optimizers」一节（Shampoo 去累积 ＝ 正交化梯度；Orthogonal-SGDM 先正交化后动量、用 SVD；
    Carlson 等 2015 的 stochastic spectral descent 用 SVD、无动量）；引文表给出 Gupta, Koren, Singer ICML 2018、
    Tuddenham, Prügel-Bennett, Hare arXiv 2202.07052、Carlson, Cevher, Carin AISTATS 2015。
  · Bernstein jeremybernste.in/writing/deriving-muon：Step 3「Dualizing the gradient」。
  · 苏剑林 kexue.fm/archives/10592：msign 是 sign 的矩阵推广；对角阵时退化为带动量的 SignSGD。
  · Rprop：Riedmiller & Braun，IEEE ICNN 1993（摘要：适应过程不受导数大小影响）。
  · signSGD：Bernstein, Wang, Azizzadenesheli, Anandkumar，ICML 2018（arXiv 1802.04434）。
  · Moonlight：arXiv 2502.16982；Kimi K2：arXiv 2507.20534（MuonClip ＝ Muon ＋ QK-Clip，15.5T token 预训练零 loss spike）。
刻意没画：Ⓐ 左边那组强度条是示意（「几个高、一长串很矮」），不是任何模型的实测谱。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE)

W = 1400
SPEC = [1.0, 0.55, 0.3] + [0.06, 0.05, 0.045, 0.04, 0.035, 0.03, 0.028, 0.025, 0.022, 0.02, 0.018, 0.016]
assert SPEC == sorted(SPEC, reverse=True) and SPEC[0] / SPEC[-1] > 50


def main():
    f = Fig(W, "为什么有用和它从哪儿来。第一格两条理由：观察——SGD 和 Adam 的更新几乎是低秩的，被少数方向主导，"
               "正交化把幅度小但重要的稀有方向抬起来；推导——要求这一层对任何输入的输出变化不超过上限，"
               "这个约束下降 loss 最多的更新就是正交化的梯度。第二格家谱：逐元素一支从 Rprop 1993 到 RMSProp、Adam，再到 signSGD 2018；"
               "矩阵一支从 随机谱下降（Stochastic Spectral Descent）2015、Shampoo 2018、Orthogonal-SGDM 2022 到 Muon 2024，"
               "再到 2025 年月之暗面的大规模训练和 QK-Clip。两支之间的桥：Muon 是 sign 的矩阵版")
    y0 = f.header("为什么有用，以及它从哪儿来　——　<tspan font-weight=\"700\">一个观察、一个推导、两条支的家谱</tspan>",
                  "Ⓐ 两条理由互相独立：一条是看出来的，一条是算出来的。Ⓑ 每个节点都核过原文",
                  [(OR, "看出来的"), (GR, "算出来的"), (BL, "逐元素支"), (PU, "矩阵支")])

    # ══ Ⓐ ═══════════════════════════════════════════════════════
    PH = 318
    py = f.panel(0, y0, W, PH, "Ⓐ 为什么拉平强度反而更好", OR)
    x0 = 30
    f.t(x0, py + 34, "观察：普通更新被少数几个方向霸占", OR, True, 16)
    for side, (vals, lab, col) in enumerate(((SPEC, "SGD 动量 ／ Adam 的更新（示意）", GY2),
                                             ([0.7] * len(SPEC), "正交化之后", OR))):
        bx = x0 + 10 + side * 330
        for k, v in enumerate(vals):
            h = v * 130
            f.box(bx + k * 18, py + 200 - h, 12, max(h, 1.5), col, col, 2)
        f.line(bx - 4, py + 200, bx + len(vals) * 18, py + 200, GY2, 1, arrow=False)
        f.t(bx, py + 224, lab, col, True, 13.5)
    f.t(x0 + 300, py + 140, "→", INK, True, 24, "middle", big=True)
    f.t(x0, py + 256, "矮的那一长串：幅度小，可对学习可能很重要 ——　「稀有方向」被抬起来了", INK, size=13.5)
    f.t(x0, py + 280, "（作者原话是推测：based on manual inspection … we speculate）", GY, size=12.5)
    # 推导
    x1 = 760
    f.box(x1 - 20, py + 14, 640, 270, "none", "#dadce0", 8)
    f.t(x1, py + 44, "推导：给「这一步」立一条规矩", GR, True, 16)
    f.t(x1, py + 80, "规矩：不管输入是什么，这一层的输出变化都不超过一个上限", INK, True, 14)
    f.t(x1, py + 106, "（输出别一下子变太多，网络才稳）", GY, size=13)
    f.t(x1, py + 146, "问：守着这条规矩，loss 降得最多的那一步长什么样？", INK, True, 14)
    f.t(x1, py + 186, "答：方向照搬梯度的频率方向，强度全部一样", GR, True, 15)
    f.t(x1, py + 212, "——　也就是正交化的梯度", GR, True, 15)
    f.t(x1, py + 252, "同一件事的几何画法：紧接着那张 Muon 总图 Ⓒ 里「方形的那个角」", GY, size=13)
    f._pan = None

    # ══ Ⓑ 家谱 ═══════════════════════════════════════════════════
    PH2 = 380
    py2 = f.panel(0, py + PH + 22, W, PH2, "Ⓑ 家谱：一条「只留正负」，一条「只留方向」", PU,
                  sub="横轴只表示先后，不是等距年份")
    NW, NH = 196, 92

    def node(x, y, yr, nm, who, what, col):
        f.box(x, y, NW, NH, "#e8f0fe" if col == BL else "#f3e8fd", col, 8)
        f.t(x + 12, y + 24, yr, col, True, 13)
        f.t(x + 12, y + 46, nm, INK, True, 14.5)
        f.t(x + 12, y + 66, who, GY, size=11.5)
        f.t(x + 12, y + 84, what, col, True, 12)

    YA, YB2 = py2 + 40, py2 + 230
    lane1 = ((40, "1993", "Rprop", "Riedmiller、Braun", "只用梯度的正负号"),
             (300, "2012 ／ 2014", "RMSProp ／ Adam", "Hinton 课件 ／ Kingma、Ba", "除以「典型大小」"),
             (620, "2018", "signSGD", "Bernstein 等", "只用正负号"))
    lane2 = ((40, "2015", "随机谱下降（SSD）", "Carlson、Cevher、Carin", "SVD 正交化，无动量"),
             (300, "2018", "Shampoo", "Gupta、Koren、Singer", "去掉累积 ＝ 正交化梯度"),
             (560, "2022", "Orthogonal-SGDM", "Tuddenham 等", "先正交化再动量（SVD）"),
             (820, "2024", "Muon", "Keller Jordan 等", "动量在前、NS 迭代、BF16"),
             (1080, "2025", "Moonlight ／ Kimi K2", "月之暗面：上规模", "加衰减、对齐幅度、QK-Clip"))
    for i, (x, yr, nm, who, what) in enumerate(lane1):
        node(x, YA, yr, nm, who, what, BL)
        if i:
            f.line(lane1[i - 1][0] + NW + 4, YA + NH / 2, x - 4, YA + NH / 2, BL, 1.8)
    for i, (x, yr, nm, who, what) in enumerate(lane2):
        node(x, YB2, yr, nm, who, what, PU)
        if i:
            f.line(lane2[i - 1][0] + NW + 4, YB2 + NH / 2, x - 4, YB2 + NH / 2, PU, 1.8)
    # 桥：sign → msign
    bx = 620 + NW / 2
    f.elbow(bx, YA + NH + 4, 820 + NW / 2, YB2 - 4, GY2, 1.6)
    f.t(700, YA + NH + 44, "两支之间的桥 ——　苏剑林：Muon ＝ sign 的矩阵版（msign）", GY, True, 13, "end")
    f.t(700, YA + NH + 66, "对角阵时它就退化成带动量的 signSGD", GY, size=12.5, anchor="end")
    f.t(940, YA + 30, "逐元素：每个数各管各的", BL, True, 14)
    f.t(940, YA + 54, "大小扔掉，只留正负（或除以典型大小）", GY, size=12.5)
    f._pan = None

    yb = f.band(py2 + PH2 + 22, "ok", "带走一句：Muon 不是凭空冒出来的，是「只留正负」那条老思路搬到了整块矩阵上", [
        "它自己新的只有两处：动量放在正交化之前，和用几次矩阵乘（Newton–Schulz）把正交化做得又快又能用 BF16。",
    ])
    yb = f.src(yb + 10,
               "出处：Keller Jordan《Muon: An optimizer for hidden layers in neural networks》（kellerjordan.github.io/posts/muon）："
               "低秩与「稀有方向」见 Why is it good to orthogonalize the update；Shampoo、Orthogonal-SGDM、"
               "Stochastic Spectral Descent 的关系见 Relationship to prior optimizers（后者那一小节作者 2025-07 补入）。",
               "出处：Bernstein《Deriving Muon》（jeremybernste.in/writing/deriving-muon）Step 3：在 RMS→RMS 算子范数下约束更新，"
               "解是保留梯度的奇异向量、扔掉奇异值大小。图上「输出变化不超过上限」是这个约束的大白话版。",
               "出处：Rprop：Riedmiller & Braun, IEEE ICNN 1993；signSGD：Bernstein 等, ICML 2018（arXiv 1802.04434）；"
               "Shampoo：Gupta, Koren, Singer, ICML 2018；Orthogonal-SGDM：Tuddenham, Prügel-Bennett, Hare, arXiv 2202.07052；"
               "Stochastic Spectral Descent：Carlson, Cevher, Carin, AISTATS 2015；RMSProp 出自 Hinton 的 Coursera 课件，未正式发表。",
               "出处：Moonlight：arXiv 2502.16982；Kimi K2：arXiv 2507.20534（MuonClip ＝ Muon ＋ QK-Clip：注意力最大 logit "
               "超过阈值时按比例缩 Q、K 的权重；15.5T token 预训练零 loss spike）。苏剑林《Muon 优化器赏析》kexue.fm/archives/10592。")
    f.save("fig4-muonfam.svg", yb + 14)


main()
