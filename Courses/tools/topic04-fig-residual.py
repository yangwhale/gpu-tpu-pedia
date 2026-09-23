# -*- coding: utf-8 -*-
r"""专题四 · §1.4「残差：把乘换成加」

⭐⭐⭐ 2026-09-23 现场：「『一路乘下去会怎样 ——&#160;梯度消失，和梯度爆炸』
  这个地方就光提出了问题，也没有解决问题。这个地方不得顺道说一下残差流的作用？」

⛔ 查下来他是对的，而且病因很具体：**解决方案那一整段（3,678 字符）里一张图都没有。**
  ⭐ 那一段文字写得不差，可这门课默认开着「只留图」的折叠开关（投屏时用）——&#160;
    于是台下看到的是：一张图把问题摆出来，然后**没了**。
  ⭐⭐ 判据：**「提出问题」和「解决问题」得在同一种载体上。**
    一个画成图、一个写成字，在折叠模式下等于只讲了前半句。

⛔ 刻意没画的：Pre-LN / Post-LN 的位置之争（那是 6.5），以及恒等映射那个式子。
  这一格只回答一句：**为什么加了一个加号，61 层就训得动了。**
"""
from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2

W = 1400

L = 61                              # V3 的层数
LO, HI = 0.9, 1.1                   # 每层局部导数离 1 差一成
P_LO, P_HI = LO ** L, HI ** L

assert abs(P_LO - 0.0016) < 0.0001, P_LO
assert abs(P_HI - 335) < 1.0, P_HI
assert 1.0 ** L == 1.0


def main():
    f = Fig(W, "一个参数的梯度是六十一个局部导数连乘。"
               "每层平均零点九，六十一层乘下来是千分之一点六，梯度到不了前面；"
               "每层平均一点一，乘下来是三百三十五，一步就炸。"
               "两个数离一都只差一成，连乘照样塌得彻底。"
               "残差做的事只有一步：把每一层从 y 等于 F(x) 改成 y 等于 x 加 F(x)，"
               "于是每层的局部导数从「某个数」变成「一加某个数」。"
               "所有括号都取一的话，一的六十一次方还是一 —— "
               "梯度总有一条不衰减的路，直达任何一层。"
               "换句话说，残差把「乘」换成了「加」：某一层学坏了只是少加一项，"
               "不会把整条路乘没")

    y0 = f.header(
        "那深层网络凭什么还能训得动"
        "　——　<tspan font-weight=\"700\">残差的全部秘密就一个加号</tspan>",
        "⭐ 上一格提出了问题（连乘不是指数塌就是指数涨）；"
        "<tspan font-weight=\"700\">这一格是答案</tspan>",
        [(RD, "没有残差：连乘"), (GR, "有残差：连加")])

    PH = 400
    py = f.panel(0, y0, W, PH,
                 "把一层的出口改一个字　——　<tspan font-weight=\"700\">"
                 "局部导数就从「某个数」变成「1 ＋ 某个数」</tspan>", GR,
                 sub="⛔ 只讲这一句；归一化放哪儿、初始化怎么缩，是 6.5 的事")

    def side(x0, col, fill, tag, formula, deriv, rows, land):
        f.box(x0, py + 56, 620, 264, fill, col, 10)
        f.t(x0 + 310, py + 92, tag, col, True, 17, "middle")
        f.t(x0 + 310, py + 130, formula, INK, True, 20, "middle")
        f.t(x0 + 310, py + 162, deriv, col, True, 15, "middle")
        for k, (a, b) in enumerate(rows):
            yy = py + 196 + k * 30
            f.t(x0 + 40, yy, a, GY, size=14)
            f.t(x0 + 580, yy, b, col, True, 15, "end")
        f.t(x0 + 310, py + 296, land, INK, True, 15, "middle")

    side(40, RD, "#fce8e6", "❌ 没有残差",
         "y ＝ F(x)", "每层的局部导数 ＝ <tspan font-weight=\"700\">某个数</tspan>",
         [("每层 %.1f，%d 层连乘" % (LO, L), "%.4f　（消失）" % P_LO),
          ("每层 %.1f，%d 层连乘" % (HI, L), "%.0f　（爆炸）" % P_HI)],
         "离 1 只差一成，<tspan font-weight=\"700\">照样塌得彻底</tspan>")

    side(740, GR, "#e6f4ea", "✅ 有残差",
         "y ＝ <tspan font-weight=\"700\">x ＋</tspan> F(x)",
         "每层的局部导数 ＝ <tspan font-weight=\"700\">1 ＋ 某个数</tspan>",
         [("所有括号都取 1，%d 层连乘" % L, "1　（不塌不涨）"),
          ("初始化时 F 很小，所以本来就是", "1 ＋ 很小")],
         "梯度总有<tspan font-weight=\"700\">一条不衰减的路</tspan>，直达任何一层")
    f._pan = None

    yb = f.band(py + PH + 20, "ok", "⭐ 一句话记住这张图", [
        "<tspan font-weight=\"700\">残差把「乘」换成了「加」。</tspan>"
        "　——　没有残差时，第 3 层收不收得到梯度，取决于第 4 到 %d 层那一整串<tspan font-weight=\"700\">乘积</tspan>；"
        "有残差之后，各层的贡献是<tspan font-weight=\"700\">加</tspan>起来的，"
        "<tspan font-weight=\"700\">某一层学坏了只是少加一项，不会把整条路乘没</tspan>。" % L,
        "⛔ 所以「残差让模型能做深」不神秘：它把每一层的局部导数"
        "<tspan font-weight=\"700\">从「某个数」钉到了 1 附近</tspan>"
        "　——　而 6.0 那张表里，爆炸和消失的根因正是<tspan font-weight=\"700\">"
        "局部导数偏离 1</tspan>。",
    ])

    yb = f.src(yb + 10,
               "📌 出处：He 等，arXiv <tspan font-weight=\"700\">1512.03385</tspan>（ResNet）；"
               "更干净的恒等通路推导在同一组人的 arXiv 1603.05027。"
               "⛔ 本格不展开那个式子 ——&#160;「1 的 %d 次方还是 1」已经是它的全部内容。" % L,
               "⚠️ 0.9 / 1.1 是<tspan font-weight=\"700\">举例的量级</tspan>，"
               "不是某个模型的实测；层数 %d 取自 DeepSeek-V3。"
               "两个连乘结果都是本脚本现算并 assert 住的。" % L)

    f.save("fig4-residual.svg", yb + 14)


main()
