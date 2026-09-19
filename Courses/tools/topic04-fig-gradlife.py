# -*- coding: utf-8 -*-
r"""专题四 · §1.8「梯度的一生」——&#160;从算出来到被扔掉，中间要过五道

⭐⭐⭐ 2026-09-17 新画。这一节原来开头是一个**五条的编号列表**，
  后面再用三个小标题分别展开其中两条。⛔ 列表能把「有五步」说清楚，
  但它说不清三件更要紧的事：**哪一步能跟计算叠在一起、哪一步不能、
  以及这五步在推理里一步都没有。**
  ⭐ 判据：**一个「依次经过若干道」的过程，列表只表达顺序，
    而流水线图能同时表达顺序、并行和缺席。**

⭐⭐ 这张图的落点在第三条泳道：**推理里这五步一步都没有。**
  ——&#160;现场定的那条界（「推理里没有的那些东西」）在这一格上最直观。

⚠️ 图上不含任何量。通信量的大小、能藏多少，是 fig-batch 和专题五的事。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE)

W = 1400

# (编号, 名字, 一句话, 能不能藏进计算里, 藏不了的原因 / 备注)
STEPS = (
    ("①", "算出来", "反向走到哪层，哪层的梯度就出来", "—",
     "⭐ 从后往前<tspan font-weight=\"700\">陆续</tspan>出来，不是最后一起出", BL),
    ("②", "跨卡汇总", "每张卡看的数据不同，得取平均", "✅ 能藏",
     "⭐ 前面几层还在算，后面的梯度已经能传了", GR),
    ("③", "裁剪", "按全局范数，超了整体缩回去", "⛔ 藏不了",
     "⛔ 要等<tspan font-weight=\"700\">所有卡所有层</tspan>都到齐，强制一次同步", RD),
    ("④", "累积", "几个 micro-batch 先加起来（可选）", "—",
     "⭐ 它改的是<tspan font-weight=\"700\">时间线</tspan>，不是总量", GY2),
    ("⑤", "用掉就扔", "交给优化器，这一份就没用了", "—",
     "⭐ 正因为用完就扔，它才敢用 bf16 存", PU),
)


def main():
    f = Fig(W, "一个梯度从出生到消失要过五道：算出来、跨卡汇总、裁剪、"
               "可选的累积、交给优化器然后扔掉。"
               "中间那条泳道标出哪一步能藏进计算里 —— 汇总可以，"
               "裁剪不行，因为它要等所有卡所有层都到齐。"
               "最下面那条泳道是这张图的落点：这五步，推理里一步都没有")

    y0 = f.header(
        "梯度的一生　——　<tspan font-weight=\"700\">"
        "从算出来到被扔掉，中间要过五道</tspan>",
        "⭐ 看三条泳道：<tspan font-weight=\"700\">做什么</tspan>、"
        "<tspan font-weight=\"700\">能不能藏</tspan>、"
        "<tspan font-weight=\"700\">推理里有没有</tspan>",
        [(GR, "能藏进计算里"), (RD, "藏不了"), (PU, "推理里都没有")])

    PH = 452
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 五道<tspan font-weight=\"700\">依次</tspan>走完，"
                 "而其中只有一道<tspan font-weight=\"700\">能跟计算叠在一起</tspan>", BL,
                 sub="⛔ 也只有一道<tspan font-weight=\"700\">会把所有卡钉在一起等</tspan>")

    X0, BW, GAP = 36, 248, 20
    TOP = py + 46
    for i, (no, name, what, hide, note, col) in enumerate(STEPS):
        x = X0 + i * (BW + GAP)
        f.box(x, TOP, BW, 118, "#fff", col, 8, sw=1.6)
        f.box(x, TOP, BW, 4, col, col, 2)
        f.t(x + 22, TOP + 42, no, col, True, 22)
        f.t(x + BW / 2 + 14, TOP + 42, name, INK, True, 17, "middle")
        for k, ln in enumerate(_wrap(what, 13)):
            f.t(x + BW / 2, TOP + 74 + k * 20, ln, GY, size=12.5, anchor="middle")
        if i < len(STEPS) - 1:
            f.t(x + BW + GAP / 2, TOP + 60, "→", GY2, True, 20, "middle")

        # 泳道二：能不能藏
        hy = TOP + 150
        hc = GR if hide.startswith("✅") else (RD if hide.startswith("⛔") else GY2)
        f.box(x, hy, BW, 34, "#e6f4ea" if hide.startswith("✅")
              else ("#fce8e6" if hide.startswith("⛔") else "#f1f3f4"), hc, 6)
        f.t(x + BW / 2, hy + 23, hide, hc, True, 14.5, "middle")
        for k, ln in enumerate(_wrap(note, 15)):
            f.t(x + BW / 2, hy + 58 + k * 20, ln, GY, size=12, anchor="middle")

        # 泳道三：推理里有没有
        ny = TOP + 268
        f.box(x, ny, BW, 34, "#f3e8fd", PU, 6)
        f.t(x + BW / 2, ny + 23, "推理里：没有", PU, True, 13.5, "middle")

    f.t(700, TOP + 336, "⭐⭐⭐ 第三条泳道是这张图的落点："
        "<tspan font-weight=\"700\">这五道，推理里一道都没有</tspan>"
        "　——　它们整条都是训练才有的东西。", INK, size=15, anchor="middle")
    f.t(700, TOP + 366, "⛔ 而这也是为什么「训练比推理贵」不只是「多跑一遍」"
        "　——　多出来的是<tspan font-weight=\"700\">一整条流水线</tspan>。",
        GY, size=13.5, anchor="middle")
    f._pan = None

    yb = f.band(py + PH + 20, "warn",
                "只有第 ③ 道会把所有卡钉在一起等 ——　而它传的其实只是一个数",
                ("⭐ 「全局范数」是跨<tspan font-weight=\"700\">全部参数</tspan>的一个标量："
                 "得先把每一块的平方和都算出来、汇总成一个数、开根号，"
                 "才知道要不要缩、缩多少。<tspan font-weight=\"700\">"
                 "通信量可以忽略，代价是那一次同步。</tspan>",
                 "⚠️ 所以<tspan font-weight=\"700\">按全局范数裁剪天然是「一步一次」</tspan>。"
                 "⛔ 但别读成「逐层裁剪不存在」——　它存在，只是保的不是同一个量："
                 "逐层保的是每层各自的范数，全局保的是这一步总共迈多大。"))

    yb = f.src(yb + 16,
               "⚠️ 图上<tspan font-weight=\"700\">不含任何量</tspan>　——　"
               "通信到底多大、能藏掉多少，是<tspan font-weight=\"700\">那张 batch 图</tspan>和专题五的事。",
               "⭐ 第 ⑤ 道那句「用完就扔」是本讲另一处的伏笔："
               "正因为它一次性，它才敢用 bf16 存　——　而一旦要做累积，"
               "它就变成累加量，得升回 fp32。")

    f.save("fig4-gradlife.svg", yb + 14)


def _wrap(t, n):
    """按汉字数粗切几行 ——&#160;标签短，不值得上排版器。"""
    import re
    plain = re.sub(r'<[^>]+>', '', t)
    if len(plain) <= n:
        return [t]
    # 有标签时不切，交给调用方保证够短
    if '<' in t:
        return [t]
    return [t[i:i + n] for i in range(0, len(t), n)]


if __name__ == "__main__":
    main()
