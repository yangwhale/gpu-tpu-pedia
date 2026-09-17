# -*- coding: utf-8 -*-
r"""专题四 · §6.5「归一化放在哪儿」——&#160;同一条曲线，两种价值

⭐⭐⭐ 2026-09-18 新画（R16）。§6.5 原来只讲了 2020 那篇的**现象**
  （Post-LN 在初始化时靠近输出层的梯度就很大，所以要 warmup），
  ⛔ **没讲机制，也没讲「那为什么还留着它」**。

⭐⭐ 苏剑林那篇给的是机制，而且论证结构本身就值得抄：
  **① LN 是元凶 → ② 那为什么还加它 → ③ 甚至对 finetune 是好处。**
  ⭐ 素材库记的那条判据说的正是这个：
    **推翻一个常识之后，一定要回头说清「那它为什么还在」——&#160;
      不做这步就成了抬杠。**

⛔⛔ **一个必须避开的陷阱**：原文公式给的是
  **x₀ 在第 l 层输出里的系数是 2^(−l/2)**，这是**前向**的残差通道被削，
  ⛔ **不等于「梯度每层乘 1/√2」**。两者相关但不是一回事，
  混起来就是那种「听起来像常识的架构关系」。
  ⭐ 所以这张图画的、标的、算的，**全部是前向的直通项系数**，
    「因此梯度消失」那一步交给正文去说，图上不冒充。

⭐⭐⭐ 画法：**同一条曲线画两遍，只换价值判断。**
  Ⓐ 和 Ⓑ 的曲线是**同一个函数、同一组坐标**（脚本里 assert 了），
  变的只有标注框的颜色和措辞 ——&#160;
  **事实没变，评价变了**，而这正是这一格要讲的事。
  ⭐ 又一次用 Olah 那招，但这次移动的不是高亮，是**结论**。

📌 出处：苏剑林《模型优化漫谈：BERT 的初始标准差为什么是 0.02？》
  kexue.fm/archives/8747 ——&#160;公式 (4) 的递归展开与
  「在 Post Norm 的 BERT 模型中，LN 不仅不能缓解梯度消失，
  它还是梯度消失的『元凶』之一」为原文原话。
  ⚠️ 这是**作者本人的分析文章**，不是同行评议论文。
"""
import math

from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE)

W = 1400

L = 24                                   # 画 24 层 ——&#160;够看出指数形状，又不至于挤
# ⭐ Post-LN：初始化阶段 Norm 相当于「除以 √2」，递归下去 x₀ 的系数就是 2^(−l/2)
POST = [2.0 ** (-l / 2.0) for l in range(L + 1)]
# ⭐ Pre-LN：x_l ＝ x₀ ＋ Σ F_i(Norm(x_i))，x₀ 的系数一直是 1（每个分支平权）
PRE = [1.0 for _ in range(L + 1)]

RATIO = PRE[L] / POST[L]
assert abs(RATIO - 2.0 ** (L / 2.0)) < 1e-9
assert RATIO > 1000, "两条要差出三个数量级以上，否则「指数」看不出来，现在 %.0f" % RATIO
# ⛔ 这条 assert 是防我自己手滑：曲线必须是单调递减的
assert all(POST[i] > POST[i + 1] for i in range(L)), "Post-LN 那条画反了"

LO, HI = -4.2, 0.45                      # log10 纵轴


def main():
    f = Fig(W, "一张曲线图画了两遍，左右完全一样，只有下面的结论框不同。"
               "横轴是层号，从最靠输入的第零层到第二十四层；"
               "纵轴是对数，表示最初那个输入在这一层的输出里还剩多少比重。"
               "绿色那条是 Pre-LN，一直是 1，横平；"
               "红色那条是 Post-LN，按二的负 l 除以二次方一路指数下滑，"
               "到第二十四层只剩四千零九十六分之一。"
               "左边的结论是：预训练的时候这叫残差名存实亡，前面的层收不到信号；"
               "右边的结论是：微调的时候这正好是想要的，"
               "它替你按住了靠近输入的那些层。同一条曲线，两种价值")

    y0 = f.header(
        "Post-LN 里，<tspan font-weight=\"700\">归一化本身就是梯度消失的元凶之一</tspan>"
        "　——　<tspan font-weight=\"700\">可它还留着，而且有道理</tspan>",
        "⚠️ 下面画的是<tspan font-weight=\"700\">前向</tspan>的残差直通项"
        "　——　<tspan font-weight=\"700\">「所以梯度也消失」那一步由正文交代，图上不冒充</tspan>",
        [(GR, "Pre-LN：平权"), (RD, "Post-LN：2^(−l/2)"),
         (PU, "同一条曲线，两种价值")])

    PH = 502
    py = f.panel(0, y0, W, PH,
                 "⭐⭐⭐ 左右两边是<tspan font-weight=\"700\">同一张图</tspan>"
                 "　——　变的只有底下那个结论", PU,
                 sub="⭐ 横轴层号，纵轴（对数）＝ "
                     "<tspan font-weight=\"700\">最初那个输入，在这一层的输出里还剩多少</tspan>")

    HALF = 640
    GT, GB = py + 76, py + 296

    def plot(x0, tag, tagcol, concl_col, concl_fill, lines):
        """⭐ 曲线部分两边**逐点相同**；只有 tag 和底下那个结论框不一样。"""
        gx0, gx1 = x0 + 54, x0 + HALF - 40

        def px(l):
            return gx0 + (gx1 - gx0) * l / float(L)

        def pyv(v):
            r = (math.log10(v) - LO) / (HI - LO)
            return GB - (GB - GT) * r

        f.line(gx0, GB, gx1 + 10, GB, GY2, 1.2, arrow=False)
        f.line(gx0, GB, gx0, GT - 8, GY2, 1.2, arrow=False)
        f.t(gx0, GT - 18, "还剩多少", GY2, size=11.5)
        f.t(gx1 + 14, GB + 16, "层 →", GY2, size=11.5)
        # 几条参考刻度，让「掉了几个数量级」可数
        for e in (0, -1, -2, -3, -4):
            yy = pyv(10.0 ** e)
            f.line(gx0 - 5, yy, gx1, yy, "#edeff1", 0.9, arrow=False)
            f.t(gx0 - 9, yy + 4, "1" if e == 0 else "10⁻%d" % (-e),
                GY2, size=11, anchor="end")

        for seq, col, sw in ((PRE, GR, 2.6), (POST, RD, 2.6)):
            d = "M %.1f %.1f" % (px(0), pyv(seq[0]))
            for l in range(1, L + 1):
                d += " L %.1f %.1f" % (px(l), pyv(seq[l]))
            f.path(d, col, sw, arrow=False)

        f.t(px(L) - 4, pyv(PRE[L]) - 12, "Pre-LN", GR, True, 13.5, "end")
        f.t(px(L) + 6, pyv(POST[L]) + 5, "Post-LN", RD, True, 13.5)
        f.t(x0 + HALF / 2.0, GT - 30, tag, tagcol, True, 16, "middle")

        # 底下那个结论框 ——&#160;左右唯一不同的东西
        by = GB + 34
        f.box(x0 + 40, by, HALF - 76, 104, concl_fill, concl_col, 8)
        for i, s in enumerate(lines):
            f.t(x0 + HALF / 2.0, by + 32 + i * 26, s,
                concl_col if i == 0 else INK,
                i == 0, 15 if i == 0 else 13.5, "middle")
        return gx0, gx1

    a = plot(0, "Ⓐ 预训练的时候", RD, RD, "#fce8e6",
             ("⛔ 残差「名存实亡」",
              "越靠近输入，信号被削得越狠",
              "<tspan font-weight=\"700\">前面的层几乎学不到东西</tspan>"))
    b = plot(HALF + 40, "Ⓑ 而拿去微调的时候", GR, GR, "#e6f4ea",
             ("✅ 这正好是你要的",
              "微调只想动靠近输出的那几层",
              "<tspan font-weight=\"700\">它替你把前面的层按住了</tspan>"))
    # ⭐⭐ 这条 assert 就是这一格的命题：两边的画面必须逐点相同
    assert a[1] - a[0] == b[1] - b[0], "两边曲线区宽度不一样，就不是「同一张图」了"

    f.line(HALF + 20, py + 60, HALF + 20, GB + 148, GY2, 1.0,
           dash="5 5", arrow=False)
    f.t(700, py + 470,
        "⭐⭐⭐ 曲线一模一样，<tspan font-weight=\"700\">连坐标都没动</tspan>"
        "　——　换的只是「你拿它来干什么」。"
        "到第 %d 层，Post-LN 的直通项只剩 Pre-LN 的 <tspan font-weight=\"700\">"
        "1／%d</tspan>。" % (L, int(round(RATIO))),
        INK, size=14.5, anchor="middle")
    f._pan = None

    yb = f.band(py + PH + 20, "warn",
                "那为什么不干脆把 LN 去掉？　——　因为方差会一路涨上去",
                ("⭐ 去掉之后 <tspan font-weight=\"700\">x ＋ F(x)</tspan> 的方差就是 2，"
                 "残差越多方差越大，所以<tspan font-weight=\"700\">还是得加一个 Norm</tspan>"
                 "　——　问题从来不是「加不加」，是<tspan font-weight=\"700\">加在哪儿</tspan>。",
                 "⭐⭐ Pre-LN 的加法是 <tspan font-weight=\"700\">x ＋ F(Norm(x))</tspan>，"
                 "最后总输出再加一个 Norm ——&#160;"
                 "这样每个残差分支<tspan font-weight=\"700\">是平权的</tspan>，"
                 "就没有上面那条指数衰减了。"))

    yb = f.src(yb + 16,
               "📌 苏剑林《模型优化漫谈：BERT 的初始标准差为什么是 0.02？》"
               "kexue.fm/archives/8747　——　"
               "「在 Post Norm 的 BERT 模型中，LN 不仅不能缓解梯度消失，"
               "它还是梯度消失的『元凶』之一」为原文原话，"
               "2^(−l/2) 出自该文公式的递归展开。",
               "⚠️ 这是<tspan font-weight=\"700\">作者本人的分析文章</tspan>，"
               "不是同行评议论文。"
               "⛔ 图上画的是<tspan font-weight=\"700\">前向的残差直通项</tspan>，"
               "不是梯度倍率　——　两者相关，但<tspan font-weight=\"700\">不是一回事</tspan>。")

    f.save("fig4-prepost.svg", yb + 14)


if __name__ == "__main__":
    main()
