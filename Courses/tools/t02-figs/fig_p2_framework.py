# -*- coding: utf-8 -*-
"""图 P-2 —— 同一个 all-reduce，两边是怎么出现的。

5.2 的落点是一句很容易被当成风格评论的话：「Megatron 手工切，MaxText 声明式」。
这张图要把它变成一件**看得见的事**：追踪同一个通信原语，看它在两条链路上
分别是在**哪一步**、由**谁**放进去的。

所以画的是两条竖直流程，不是两张特性表 —— 表会让人比功能，流程才让人看见
「这一步是谁做的」。左边那条链上，all-reduce 出现在**你写的第三行**；
右边那条链上，你从头到尾没写过它，它出现在**编译产物里**。

下面的四行对照必须放在流程之后，不能放在之前：先看见机制，再看代价，
否则听众会把它读成「两个框架的参数多少之争」。
"""
from common import Fig, para, BL, RD, GN, YL, INK, SUB, GREY, FILL

W = 1400
TOP = 84

LX, LW = 20, 676
RX, RW = 724, 656

STEP_H, STEP_GAP = 76, 30
NSTEP = 3
FLOW_TOP = TOP + 44
FLOW_H = 44 + NSTEP * STEP_H + (NSTEP - 1) * STEP_GAP + 66
COL_H = FLOW_H + 16

CMP_Y = TOP + COL_H + 44
CMP_ROWS = [
    ("并行怎么表达",
     "<b>手工切</b>。张量并行、流水并行各有一套模块，"
     "通信写在模块的 <code>forward</code> 里",
     "<b>声明式</b>。给张量标 sharding，"
     "通信由编译器按 mesh 生成"),
    ("心智负担在哪",
     "要知道<b>每一处 all-reduce 在哪</b>、为什么在那儿",
     "要理解 <code>mesh</code> 和 <code>PartitionSpec</code>，"
     "以及它们怎么传播"),
    ("出问题的时候",
     "<b>看得见每一步</b>。加个打印就知道谁慢",
     "<r>要会读编译产物</r>。代码里没有的东西，profile 里有"),
    ("配置复杂度",
     "参数极多，但<b>错一个通常只影响一处</b>",
     "参数少，但<r>一个 sharding 标错全盘皆输</r> —— "
     "而且可能只表现为「慢」，不报错"),
]
CMP_HH, CMP_RH = 36, 54
CMP_H = CMP_HH + len(CMP_ROWS) * CMP_RH

BAND_Y = CMP_Y + CMP_H + 34
H = BAND_Y + 118


def build():
    f = Fig(W, H, "追踪同一个 all-reduce 在 Megatron 与 MaxText 两条链路上分别由谁插入，"
                  "说明手工切分与声明式切分的差别")
    f.title("同一个 all-reduce，两边是怎么<tspan font-weight=\"700\" fill=\"#d93025\">出现</tspan>的"
            "　—— 追一个通信原语，比对比两张特性表有用得多")
    f.legend([(RD, "你写的（人负责）"), (GN, "编译器生成的（工具负责）"),
              (YL, "通信真正发生的那一步")])

    _flow(f, LX, LW, RD, "Megatron-LM　·　GPU",
          [("① 你选一个并行模块",
            "<code>ColumnParallelLinear</code> / <code>RowParallelLinear</code> —— "
            "选哪个，等于你已经决定了这一层按行切还是按列切。", RD),
           ("② 你在 forward 里写下通信",
            "行切的那一半算完只是<b>部分和</b>，所以模块的 <code>forward</code> 末尾"
            "<b>显式调</b> <code>all_reduce</code>。<r>这一行是你写的。</r>", YL),
           ("③ 换个并行度，回到 ①",
            "切法变了，通信位置就变了 —— <b>要你自己重新想一遍</b>。", RD)],
          "<b>整条链上没有任何一步是自动的。</b>好处是每一步都在你眼皮底下，"
          "坏处是<r>每一步都得由你想对</r>。")

    _flow(f, RX, RW, GN, "MaxText　·　TPU",
          [("① 你只标注切分意图",
            "给张量挂上 <code>PartitionSpec</code>，说清它<b>按哪根轴切</b>。"
            "至于要不要通信，你没说，也不用说。", GN),
           ("② 编译器推导整张图",
            "sharding 沿着算子传播；发现某个算子的输入切法<b>对不上</b>时，"
            "<b>自己插一个集合通信</b>补齐。", GN),
           ("③ all-reduce 出现在编译产物里",
            "<r>你的源码里从头到尾没有这个词。</r>"
            "它在 HLO 里，也在 profile 里。", YL)],
          "<b>整条链上你只做了第一步。</b>好处是并行度一改重编一次就行，"
          "坏处是<r>它插在哪儿、插得好不好，你只能事后去读</r>。")

    _cmp(f)
    _band(f)
    return f.out()


# ══════════════════════════════════════════════════════════════════════
def _flow(f, x, w, c, head, steps, foot):
    f.rect(x, TOP, w, COL_H, FILL[c], c, 1.4, 10)
    f.t(x + 18, TOP + 27, head, "sec")

    for i, (t, body, sc) in enumerate(steps):
        y = FLOW_TOP + i * (STEP_H + STEP_GAP)
        f.rect(x + 18, y, w - 36, STEP_H, "#fff", sc, 1.3, 8)
        f.t(x + 32, y + 24, t, "box", INK)
        para(f, x + 32, y + 44, w - 64, body, "xs", 17)
        if i < len(steps) - 1:
            f.line(x + w / 2, y + STEP_H + 4, x + w / 2, y + STEP_H + STEP_GAP - 4,
                   c, 1.6, "aR" if c == RD else "aG")

    y = FLOW_TOP + NSTEP * STEP_H + (NSTEP - 1) * STEP_GAP + 14
    f.rect(x + 18, y, w - 36, 46, "#f1f3f4", None, 0, 6)
    para(f, x + 32, y + 22, w - 64, foot, "xs", 17)


# ══════════════════════════════════════════════════════════════════════
def _cmp(f):
    cx, cw = [20, 232, 812], [206, 574, 568]
    f.t(20, CMP_Y - 12, "看完机制再看代价　—— 这四行都是上面那条链的直接后果，不是两个团队的口味差异", "sec")

    f.rect(20, CMP_Y, 1360, CMP_HH, "#f1f3f4", None, 0, 8)
    for i, h in enumerate(["问的是同一个问题", "Megatron-LM（GPU）", "MaxText（TPU）"]):
        f.t(cx[i] + 14, CMP_Y + 23, h, "lbl", INK)

    for r, (q, a, b) in enumerate(CMP_ROWS):
        y = CMP_Y + CMP_HH + r * CMP_RH
        if r % 2:
            f.rect(20, y, 1360, CMP_RH, "#fafafa", None, 0, 0)
        f.line(20, y, 1380, y, "#e8eaed", 1)
        para(f, cx[0] + 14, y + 32, cw[0] - 28, "<b>%s</b>" % q, "xs")
        para(f, cx[1] + 14, y + 26, cw[1] - 28, a, "xs", 17)
        para(f, cx[2] + 14, y + 26, cw[2] - 28, b, "xs", 17)
    f.rect(20, CMP_Y, 1360, CMP_H, "none", "#dadce0", 1.2, 8)


# ══════════════════════════════════════════════════════════════════════
def _band(f):
    f.rect(20, BAND_Y, 1360, 102, FILL[BL], BL, 1.4, 10)
    f.t(38, BAND_Y + 28, "这不是两个团队风格不同，是同一条基因", "sec")
    y = BAND_Y + 54
    for seg in [
            "把这张图和上一张并排看：<b>「谁在安排」这个问题，在框架层原样复现了一遍。</b>"
            "硬件上是「谁决定数据什么时候搬」，框架上就是「谁决定通信插在哪」——"
            "<b>同一个选择，换了一层皮。</b>",
            "所以选框架的时候真正要问的不是「哪个更快」，而是"
            "<r>「我这个队，是更有人能想对每一处通信，还是更有人读得懂编译产物」</r>。"]:
        y = para(f, 38, y, 1324, seg, "xs", 19) + 4


if __name__ == "__main__":
    import sys
    open(sys.argv[1] if len(sys.argv) > 1 else "out/fig_p2_framework.svg",
         "w", encoding="utf-8").write(build())
