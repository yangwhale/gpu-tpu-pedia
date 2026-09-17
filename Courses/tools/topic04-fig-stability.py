# -*- coding: utf-8 -*-
r"""专题四 · §6「训不崩 ——&#160;loss 飞了怎么办，以及怎么让它别飞」

⭐⭐⭐ 2026-09-16 新画。这一节是**扫十四份大纲之后发现的最大空白之一**：
  「loss spike」「z-loss」「scaling law」在整套课程里**零命中**。
  而前两样是真训练时最痛的东西。

⭐⭐ 这张图的取舍：**按「治在链条的哪一步」排，不按「有哪些技巧」排。**
  ⛔ 稳定性这个题目最容易写成一张「技巧清单」——&#160;
    裁剪、z-loss、QK-norm、Pre-LN、初始化……列完了，读者还是不知道该先试哪个。
  ⭐ 而它们其实各治一处：**有的治结构、有的治数值、有的治现场**。
    分清了，「先动哪个」这个问题自己就有答案了。

⭐ Ⓒ 那张对照表是 ST-MoE 论文 Table 4 的原数，**它是这一节最值钱的一格**：
  「稳住」很容易，难的是**稳住而不牺牲质量**。
  收紧 update clipping 三次全稳，可质量从 −1.755 掉到 −4.206。

📌 出处（全部公开，逐条核过原文）：
  · PaLM 540B 的 spike 与消融：arXiv 2204.02311 §5.1
  · router z-loss 与 Table 4：arXiv 2202.08906
  · QK-norm 与 8B 处发散：arXiv 2302.05442 §2
  · Post-LN / Pre-LN 与 warmup：arXiv 2002.04745（ICML 2020）
"""
from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE, LINE2)

W = 1400

# ⭐ ST-MoE Table 4 原数（负对数似然类指标，越大越好）
STMOE = (
    ("基线", "4/6", -1.755, GY2, "#f1f3f4", "⚠️ 六次跑崩两次"),
    ("收紧 update clipping", "3/3", -4.206, RD, "#fce8e6", "⛔ 条短了一大截 ——　看这里"),
    ("router z-loss", "3/3", -1.741, GR, "#e6f4ea", "⭐ 条最长，而且一次没崩"),
)
assert STMOE[1][2] < STMOE[0][2] < STMOE[2][2]   # 「稳 ≠ 好」这件事得成立


def main():
    f = Fig(W, "把训练稳定性的三类治法画成一条河：上游改河道、中游装闸门、"
               "下游捞船。对应到技术上，"
               "结构层面是归一化放哪儿和 QK-norm，数值层面是 z-loss 和梯度裁剪，"
               "现场层面是回滚 checkpoint 加跳数据。"
               "而 ST-MoE 那张对照表说明：让训练稳住很容易，"
               "难的是稳住而不牺牲质量 —— 收紧 update clipping 三次全稳，"
               "可质量从负一点七五掉到负四点二")

    y0 = f.header(
        "训不崩　——　<tspan font-weight=\"700\">"
        "把它当成一条河，三段各治各的</tspan>",
        "⛔ 这一节最容易写成一张技巧清单 ——&#160;<tspan font-weight=\"700\">"
        "列完了还是不知道该先动哪个</tspan>",
        [(BL, "上游 · 改河道"), (OR, "中游 · 装闸门"), (RD, "下游 · 捞船")])

    # ══════════ Ⓐ 一条河 ══════════════════════════════════════════
    # ⭐⭐⭐ 2026-09-17 重画。原来是**三张并排的卡片**，每张列三条技巧。
    #   ⛔ 它的毛病不是不对，是**没有画面**：三张卡并排，
    #     读者记住的是「有九个技巧」，而不是「它们各治一段」。
    # ⭐ 换成一条河之后，「上游 / 中游 / 下游」这三个词自己就把分工说完了：
    #     · 上游 **改河道** ——&#160;把河挖宽挖直，水本来就不容易漫出来（治结构）
    #     · 中游 **装闸门** ——&#160;水位一高就关闸，不让它冲下去（治数值）
    #     · 下游 **捞船**   ——&#160;已经翻了，把船捞回来重新开（治现场）
    #   ⭐⭐ 而这个比喻还顺带带出了一件卡片版讲不出来的事：
    #     **上游的活是一次性的，下游的活每出一次事就得干一次。**
    # 📌 现场原话：「跟实际生活中的事情结合起来」——&#160;这一格是那条要求的落点。
    PH = 446
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 把它想成<tspan font-weight=\"700\">一条河</tspan>　——　"
                 "<tspan font-weight=\"700\">同一场水患，三个位置都能下手</tspan>", BL,
                 sub="⭐ 从上游到下游：<tspan font-weight=\"700\">"
                     "越靠上越治本，越靠下越应急</tspan>")

    RX0, RX1 = 40, W - 40
    RTOP, RBOT = py + 150, py + 236          # 河面上下沿
    SEG = (RX1 - RX0) / 3.0

    # 河 —— 一条从左到右、略微变宽的蓝带（三段各自着色）
    RIVER = (
        (BL, "#e8f0fe", "上游", "改河道",
         "把河挖宽、挖直　——　水本来就不容易漫出来",
         ("归一化放进残差块里（Pre-LN）",
          "QK-norm　——　别让注意力分数长疯",
          "初始化按深度缩一下"),
         "⭐ 一次性：开工前做完，之后不用管"),
        (OR, "#fef7e0", "中游", "装闸门",
         "水位一高就关闸　——　不让它冲下去",
         ("z-loss　——　摁住 softmax 前的 logits",
          "梯度裁剪　——　按全局范数，常设 1.0",
          "关键处用高精度算"),
         "⭐ 常驻：一直开着，便宜，多数不伤质量"),
        (RD, "#fce8e6", "下游", "捞船",
         "已经翻了　——&#160;把船捞回来重新开",
         ("回滚到 spike 之前的 checkpoint",
          "跳掉那一段数据，再往下跑",
          "把出事的时间点和数据位置记下来"),
         "⛔ 每出一次事就得干一次　——　而且没修好"),
    )
    for i, (col, fill, where, act, why, items, note) in enumerate(RIVER):
        x0 = RX0 + i * SEG
        # 河段：上沿平、下沿随着往下游走略微加宽（水越来越急）
        d = ("M %.1f %.1f L %.1f %.1f L %.1f %.1f L %.1f %.1f Z"
             % (x0, RTOP - i * 6, x0 + SEG, RTOP - (i + 1) * 6,
                x0 + SEG, RBOT + (i + 1) * 8, x0, RBOT + i * 8))
        f.path(d, col, 1.4, fill=fill, arrow=False)
        cx = x0 + SEG / 2.0
        f.t(cx, RTOP + 42, where, col, True, 26, "middle")
        f.t(cx, RTOP + 72, act, INK, True, 19, "middle")
        f.t(cx, py + 116, why, GY, size=13.5, anchor="middle")
        for k, it in enumerate(items):
            f.t(cx, py + 278 + k * 26, "· " + it, GY, size=13, anchor="middle")
        f.t(cx, py + 380, note, col, True, 13.5, "middle")
        if i:
            f.line(x0, RTOP - 26, x0, RBOT + 34, GY2, 1, dash="4 4", arrow=False)

    # 水流方向
    f.t(RX0 + 4, py + 76, "水往这边流　→", BL, True, 14)
    f.t(RX1 - 4, py + 76, "loss 飞了", RD, True, 15, "end")

    f.t(700, py + 412, "⭐⭐⭐ <tspan font-weight=\"700\">越往下游，越是在救火</tspan>"
        "　——　而上游那几样<tspan font-weight=\"700\">开工前做一次就完了</tspan>，"
        "下游那几样<tspan font-weight=\"700\">每出一次事就得再干一次</tspan>。",
        INK, size=14.5, anchor="middle")
    f._pan = None

    # ══════════ Ⓑ 那个最反直觉的实验 ═════════════════════════════
    PH2 = 260
    py2 = f.panel(0, py + PH + 22, W, PH2,
                  "Ⓑ ⭐⭐⭐ PaLM 做过一个消融，结论跟所有人的第一反应相反", RD,
                  sub="⭐ 这个消融<tspan font-weight=\"700\">只花了一次重训</tspan>，"
                      "却把一整类猜测排除掉了")

    f.box(70, py2 + 34, 600, 186, "#fce8e6", RD, 8)
    f.t(370, py2 + 70, "拿「坏数据」这个假设验一验", RD, True, 17, "middle")
    f.t(370, py2 + 118, "同一批数据　＋　<tspan font-weight=\"700\">换一个时刻</tspan>",
        GY, True, 16, "middle")
    f.t(370, py2 + 152, "↓", GY2, True, 18, "middle")
    f.t(370, py2 + 190, "⛔ 结果：不飞。", RD, True, 20, "middle")

    f.line(690, py2 + 128, 726, py2 + 128, GY2, 1.8)

    f.box(740, py2 + 34, 600, 186, "#e6f4ea", GR, 8)
    f.t(1040, py2 + 70, "所以不是数据坏", GR, True, 19, "middle")
    f.t(1040, py2 + 122, "两个条件<tspan font-weight=\"700\">同时满足</tspan>才出事",
        INK, True, 16, "middle")
    f.t(1040, py2 + 152, "缺一样都不会飞", GY, size=14, anchor="middle")
    f.t(1040, py2 + 190, "⭐⭐ 这解释了为什么回滚 ＋ 跳数据管用",
        GR, True, 16, "middle")
    f._pan = None

    # ══════════ Ⓒ 稳 ≠ 好 ════════════════════════════════════════
    PH3 = 322
    py3 = f.panel(0, py2 + PH2 + 22, W, PH3,
                  "Ⓒ ⭐⭐ 这一节真正的难点：<tspan font-weight=\"700\">"
                  "让它稳住很容易，难的是稳住而不牺牲质量</tspan>", GR,
                  sub="📌 ST-MoE 论文 Table 4 的原数 ——　"
                      "三种做法，三次独立训练")

    BX, BW = 430, 620
    LO, HI = -4.4, -1.6
    for i, (nm, stab, q, col, fill, note) in enumerate(STMOE):
        y = py3 + 44 + i * 74
        f.t(BX - 20, y + 26, nm, INK, True, 15.5, "end")
        frac = (q - LO) / (HI - LO)
        f.box(BX, y, BW, 40, "#f8f9fa", LINE2, 6)
        f.box(BX, y, max(6, BW * frac), 40, fill, col, 6)
        f.t(BX + BW + 14, y + 27, "%.3f" % q, col, True, 17)
        f.t(BX + BW + 96, y + 27, "稳定 %s" % stab, GY, True, 14)
        f.t(BX + 12, y + 62, note, GY, size=13)
    f.t(BX, py3 + 280, "⭐ 横条越长质量越好（这是个越大越好的指标）。"
                       "<tspan font-weight=\"700\">中间那根「稳定 3/3」看着完美，"
                       "可它把质量打穿了。</tspan>", GY, size=14.5)
    f._pan = None

    yy = f.band(py3 + PH3 + 22, "bad", "顺带一条：这篇论文自己也撞上了我们那条判据", [
        "⛔ ST-MoE 原话：<tspan font-weight=\"700\">「我们发现改进常常会消失、甚至反过来，"
        "当模型训得更久或者做得更大的时候」</tspan> ——&#160;"
        "他们举的例子是：前一篇论文关于 top-n 路由的结论，"
        "在<tspan font-weight=\"700\">八倍规模</tspan>的实验里<tspan font-weight=\"700\">翻转了</tspan>。",
        "⭐⭐ 这正是 <tspan font-weight=\"700\">2.6 那条判据</tspan>的第三方版本："
        "<tspan font-weight=\"700\">小规模上验过的结论，到目标规模上必须重验。</tspan>"
        "⭐ 而且值得注意：<tspan font-weight=\"700\">这一讲里它已经出现三次了</tspan>"
        " ——&#160;重算（序列长度）、这里（模型规模）、"
        "以及混元上那次同一个开关收益变号。",
    ], fold=True)

    yy = f.src(yy + 24,
               "PaLM 540B 的 spike 与消融：<tspan font-weight=\"700\">arXiv 2204.02311</tspan>"
               " §5.1 ——&#160;「大约 20 次」「回滚约 100 步」「跳 200–500 批」"
               "「<tspan font-weight=\"700\">不认为是数据本身坏</tspan>」都是原文",
               "router z-loss 与 Ⓒ 那张表：<tspan font-weight=\"700\">arXiv 2202.08906</tspan>"
               "（ST-MoE）。⭐ 论文自己的小标题就是"
               "「<tspan font-weight=\"700\">很多方法能稳住稀疏模型，但代价是质量变差</tspan>」",
               "QK-norm：<tspan font-weight=\"700\">arXiv 2302.05442</tspan>（ViT-22B）§2 ——&#160;"
               "在<tspan font-weight=\"700\">约 80 亿参数</tspan>处观察到训练发散，"
               "根因是注意力 logits 变得极大、"
               "注意力权重塌成几乎 one-hot（熵接近 0）",
               "归一化的位置：<tspan font-weight=\"700\">arXiv 2002.04745</tspan>（ICML 2020）——&#160;"
               "证明 Post-LN 在初始化时<tspan font-weight=\"700\">靠近输出层的梯度很大</tspan>，"
               "所以必须靠 warmup 压住；换成 Pre-LN 则初始化时梯度就是良态的",
               "⛔ 本图<tspan font-weight=\"700\">不含任何我们自己的实测</tspan> ——&#160;"
               "四条全部来自公开论文，Ⓒ 那三个数是照抄 Table 4")
    f.save("fig4-stability.svg", yy + 6)


main()
