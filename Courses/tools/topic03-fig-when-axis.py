# -*- coding: utf-8 -*-
r"""专题三 · §四「第二个轴：什么时候改」（2026-09-13 夜间 · R10）。

⭐⭐⭐ 前九轮里，「**事后压 vs 从头按压缩训**」这条对立**出现了四次**，
   每次都在不同的技术分支上。到第四次的时候它已经不是巧合了 ——&nbsp;
   它是一个**跟三个旋钮正交的轴**，该被扶正。

     · 三个旋钮回答的是「**改什么**」
     · 这条暗线回答的是「**什么时候改**」

   两个轴一交叉，就从一条清单变成一张**地图**：
   拿到任何一个新名字，先定位它落在哪一格。

⚠️ 一条必须讲清的口径（否则这张图会教出一个错误的心智模型）：
   **「事后 / native」说的是这个方法<b>被怎么用</b>，不是方法本身的属性。**
   GQA 最初是 uptraining 出来的（事后），但今天 Llama 那一系
   **从第一天就是 GQA**（native）。同一个机制，两种用法。
"""
from topic03_draw import (Fig, wpx, BL, OR, GR, RD, GY, PU, CY, INK,
                          GY2, LINE, LINE2, BG2)

W = 1400


def main():
    ROWS = [
        ("旋钮①　每份更小", "低秩分解",
         "Eigen Attention / Palu / LoRC", "拿训好的模型做低秩分解", "省 40–50%",
         "MLA", "从第一天就按 512 训", "省 56.9×"),
        ("旋钮①　每份更小", "砍头",
         "GQA uptraining", "平均池化 ＋ 5% 算力续训", "接近 MHA",
         "GQA from scratch", "一开始就分 8 组", "同上，但不用两段"),
        ("旋钮②　每步少读", "挑着看",
         "H2O", "一个 KV 驱逐策略，不改训练", "随时可开关",
         "DSA", "943.7B token 继续训练", "掉点小得多"),
        ("旋钮②　每步少读", "挑着看",
         "ClusterKV / MagicPIG / Quest", "推理期加的选择器", "⛔ 梯度断、访存不连续",
         "NSA", "端到端训练的三条分支", "持平或超过全注意力"),
    ]

    f = Fig(W, "第二个轴：什么时候改。事后压与从头按压缩训这条对立在四个分支上"
               "各出现一次，它跟三个旋钮正交，两轴一交叉就成了一张地图")
    f.marks = set()
    y0 = f.header(
        "第二个轴　——　三个旋钮说的是「改什么」，这条说的是「什么时候改」",
        "⭐⭐ 这条对立今天<tspan font-weight=\"700\">会出现四次</tspan>，"
        "四次在四个不同的分支上 ——&#160;所以它不是巧合，是一个轴",
        [(OR, "事后：拿训好的模型去凑"), (GR, "native：把约束写进训练"),
         (GY, "同一个旋钮的两种用法")])

    # ── 表头 ────────────────────────────────────────────────────
    CX = [0, 236, 700]
    CW = [228, 456, 700]
    yy = y0 + 8
    f.colhead(CX[0], yy, "在哪个旋钮上")
    f.t(CX[1] + 16, yy, "事后　拿训好的模型去凑", OR, True, 13, cls="svglbl")
    f.t(CX[2] + 16, yy, "native　把约束写进训练", GR, True, 13, cls="svglbl")
    yy += 14

    for knob, kind, a_who, a_how, a_gain, b_who, b_how, b_gain in ROWS:
        h = 72
        # 左：旋钮
        f.t(CX[0], yy + 30, knob, INK, True, 12.5)
        f.t(CX[0], yy + 50, kind, GY2, size=11)
        # 中：事后
        f.box(CX[1], yy, CW[1] - 16, h, "#fff", OR, 8)
        f.box(CX[1], yy, 4, h, OR, OR, 2)
        f.box(CX[1] + 2, yy, 3, h, "#fff", "#fff", 0)
        f.t(CX[1] + 18, yy + 26, a_who, OR, True, 12.5, w=CW[1] - 200)
        f.t(CX[1] + 18, yy + 50, a_how, GY, size=11.5, w=CW[1] - 200)
        f.t(CX[1] + CW[1] - 34, yy + 50, a_gain, GY2, size=11, anchor="end")
        # 右：native
        f.box(CX[2], yy, CW[2] - 16, h, "#fff", GR, 8)
        f.box(CX[2], yy, 4, h, GR, GR, 2)
        f.box(CX[2] + 2, yy, 3, h, "#fff", "#fff", 0)
        f.t(CX[2] + 18, yy + 26, b_who, GR, True, 12.5, w=CW[2] - 240)
        f.t(CX[2] + 18, yy + 50, b_how, GY, size=11.5, w=CW[2] - 240)
        f.t(CX[2] + CW[2] - 34, yy + 50, b_gain, GY2, size=11, anchor="end")
        yy += h + 12

    # ── 两条通用结论 ────────────────────────────────────────────
    yy += 6
    for col, title, l1, l2 in [
        (OR, "事后这一列的共同点", "便宜、不用重训、随时可开关",
         "⛔ 但天花板明显更低 —— 最好也就「接近原来那个」"),
        (GR, "native 这一列的共同点", "要重训，而且往往要专门的 kernel",
         "⭐ 但它能<tspan font-weight=\"700\">持平甚至超过</tspan>，还能顺带省训练成本"),
    ]:
        f.box(0, yy, W, 66, "#fff", col, 8)
        f.box(0, yy, 4, 66, col, col, 2)
        f.box(2, yy, 3, 66, "#fff", "#fff", 0)
        f.t(20, yy + 25, title, col, True, 13, cls="svglbl")
        f.t(200, yy + 25, l1, GY, size=12)
        f.t(20, yy + 48, l2, GY, size=11.5)
        yy += 76

    yy = f.band(yy + 6, "info",
                "⭐⭐ 两个轴一交叉，这一讲就从一张清单变成一张地图", [
        "横轴「<tspan font-weight=\"700\">改什么</tspan>」：每份更小 / 每步少读 / 换一套数学。"
        "纵轴「<tspan font-weight=\"700\">什么时候改</tspan>」：事后 / 从头。",
        "⭐ 拿到任何一个新名字，先把它<tspan font-weight=\"700\">放进某一格</tspan>"
        "——&#160;放不进去的，才值得你花时间细看；"
        "能放进去的，它的优点和代价<tspan font-weight=\"700\">你已经知道了</tspan>。",
    ])

    yy = f.band(yy + 14, "warn",
                "一条必须讲清的口径，否则这张图会教出错误的心智模型", [
        "<tspan font-weight=\"700\">「事后 / native」说的是这个方法被怎么用，"
        "不是方法本身的属性。</tspan>",
        "GQA 最初是 uptraining 出来的（事后那一列），"
        "但今天很多模型<tspan font-weight=\"700\">从第一天就是 GQA</tspan>（native 那一列）。"
        "⭐ 同一个机制，两种用法，两种结论 ——&#160;"
        "所以<tspan font-weight=\"700\">别问「这个方法属于哪一列」，要问「这次它被怎么用」</tspan>。",
    ])

    yy = f.src(yy + 16,
               "⚠️ 下面四行里有三行要到 §五 / §六 才展开 ——&#160;"
               "这一格是<tspan font-weight=\"700\">先给坐标系</tspan>，不是回顾；"
               "出处分别见 §五（Eigen Attention / "
               "Palu / LoRC、GQA uptraining）"
               "与 §六（H2O、DSA、NSA §2 列的那批事后方法）——&#160;每一处都有单独的图和引用",
               "「两个轴」这个组织方式是本课的归纳，不是任何一篇论文的提法")
    f.save("fig3-when-axis.svg", yy + 6)


main()
