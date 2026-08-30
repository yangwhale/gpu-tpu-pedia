# -*- coding: utf-8 -*-
"""图 P-6 —— 实测：规模一大，all-to-all 就掉。以及一个塌掉的对照项。

数据是我们自己跑的（DeepEP，GB200 A4X，2 / 4 / 8 节点即 8 / 16 / 32 GPU），
出处是本仓库 `gpu/a4x/06-deepep-test/README.md`。

**这张图被审计推翻重画过一次，过程比结论值钱，所以完整记下来。**

第一版的论证是这样的：Copy 三档纹丝不动 → 链路能力够 → 掉下去的那两条
只能是「消息变小」。干净、好讲、听众一听就懂。**而且是错的。**

错在哪：源文档第 356 行白纸黑字写着 `Copy (NVLink) … 节点内 GPU 间拷贝`，
第 389 行还注明是「跨 4 GPU 聚合」。**它根本不出节点** ——
而 dispatch / combine 是跨 2 / 4 / 8 个节点的 all-to-all。
一个从不出节点的量，对「出节点那条路够不够」**零信息量**。
它三档不变是几乎必然的，不是证据。

更糟的是第一版还顺手加了一句「同域走的是 MNNVL（NVLink），本来就不该有
RDMA 竞争」，拿它去推翻源文档对 16 卡那次下降给的「RDMA 竞争」归因。
**这句是凭架构常识编的，而且和我们自己的实测直接冲突**：
同一份文档里，部署要点第 3 条是 `NCCL_NET=gIB`（启用 GPUDirect RDMA），
第 322 行写「GIB RDMA（跨节点）+ NVSwitch/MNNVL（域内）」，
而 8 节点那次失败的根因正是 `ibv_modify_qp failed` ——
**RDMA 实实在在在这条路径上，QP 都建起来了。**

所以这一版换了个立场：**不给归因，给「我们能确定什么、不能确定什么」。**
并且把源文档里另一行也画上 —— `Reduce (NVLink)` 同样标着「节点内」，
却掉了 18%，和 Combine 几乎一样。那两条线在图上会叠在一起，
**这个视觉巧合本身就是那条「节点内」标注不可信的最好证据。**

画法上唯一保留的决定：主图归一化。Copy 5,600 vs Dispatch 700 差一个量级，
画在同一根绝对值纵轴上，后几条会被压成贴地直线，而「掉了多少」是全部论点。
"""
from common import Fig, para, BL, RD, GN, YL, PU, INK, SUB, GREY, FILL

W = 1400
TOP = 84

# ── 实测数据（GB/s）。出处：本仓库 gpu/a4x/06-deepep-test/README.md ──
# 标注那一列是**源文档自己打的标签**，不是我们加的 —— 下面第 4 张卡要用它。
XS = ["8 GPU\n（2 节点）", "16 GPU\n（4 节点）", "32 GPU\n（8 节点）"]
SERIES = [
    ("Copy　　　源文档标注：节点内", GN, [5600, 5500, 5700]),
    ("Reduce　　源文档标注：节点内", PU, [2100, 1870, 1730]),
    ("Dispatch　跨节点 all-to-all", YL, [700, 660, 636]),
    ("Combine　 跨节点 all-to-all ＋ reduce", RD, [724, 683, 590]),
]

CH_X, CH_Y, CH_W, CH_H = 76, TOP + 84, 560, 330      # 绘图区
LO, HI = 76, 106                                      # 纵轴：百分比

RD_X, RD_W = 700, 680                                 # 右侧解读栏

TB_Y = CH_Y + CH_H + 96
TB_H = 34 + len(SERIES) * 34

BAND_Y = TB_Y + TB_H + 34
H = BAND_Y + 168

# 四张卡。第一张是**自我更正**，所以放在最前面、用红框 ——
# 「我们原来怎么想的、为什么不成立」比「正确答案是什么」更值得先讲。
READ = [
    ("① 我们本来挑了 Copy 当对照项　→　<r>它走错了路</r>", RD, 100,
     "原打算：Copy 不掉 → 链路够用 → 掉的只能是别的原因。"
     "<b>但源文档写明 Copy 是「节点内 GPU 间拷贝」</b> —— "
     "<r>它不走 dispatch / combine 走的那条跨节点路。</r>"
     "一个不出节点的量，对「出节点那条路够不够」没有信息。"),
    ("② 能确定的只有现象：目标数从 7 变成 31", YL, 96,
     "8 卡时每张卡把 token 分给 <b>7</b> 个目标，32 卡时分给 <b>31</b> 个。"
     "<b>单笔消息必然变小</b>，而小块分散写的链路利用率低于大块连续传输。"
     "<g>源文档那个「每目标约 100 / 22 GB/s」是拿总带宽除目标数<b>倒算</b>出来的，"
     "不是独立测量 —— 拿它解释掉幅属于循环论证，这里不用。</g>"),
    ("③ Combine 掉得更快，但<r>不是因为「归约变深」</r>", RD, 92,
     "每个 token 只把 <b>topk ＝ 6</b> 份部分结果加回去，"
     "<b>这个宽度是配置死的，与规模无关</b>，8 卡和 32 卡都是 6 份。"
     "更站得住的说法是：<b>要等齐的对端从 7 个变成 31 个</b>，"
     "归约必须等齐才能收尾，等的人越多，最慢那个的影响越大。"),
    ("④ <r>而且 Reduce 也掉了 18%</r>　—— 它同样标着「节点内」", PU, 84,
     "紫线和红线在图上几乎叠在一起。<b>要么这条「节点内」标注不准，"
     "要么「节点内不受规模影响」这句话本身要打折。</b>"
     "<r>这处我们没查清，标出来。</r>"),
]


def build():
    f = Fig(W, H, "DeepEP 在 8、16、32 GPU 三档的实测带宽，以 8 GPU 为基准归一化："
                  "Copy 基本持平，Reduce 降约 18%，Dispatch 降约 9%，Combine 降约 19%")
    f.title("实测：规模一大，<tspan font-weight=\"700\" fill=\"#d93025\">all-to-all 就掉</tspan>"
            "　—— 以及一个<tspan font-weight=\"700\" fill=\"#d93025\">塌掉的对照项</tspan>")
    f.legend([(GN, "Copy（源文档标注：节点内）"), (PU, "Reduce（同样标注：节点内）"),
              (YL, "Dispatch（跨节点）"), (RD, "Combine（跨节点）"),
              (GREY, "纵轴＝相对 8 GPU 的百分比，不是绝对带宽")])

    _chart(f)
    _read(f)
    _table(f)
    _band(f)
    return f.out()


# ══════════════════════════════════════════════════════════════════════
def _px(i):
    return CH_X + CH_W * (i + 0.5) / len(XS)


def _py(pct):
    return CH_Y + CH_H * (HI - pct) / (HI - LO)


def _chart(f):
    f.t(20, TOP + 32, "主图以 8 GPU 为 100%　—— 画绝对值会把后三条压成贴地直线", "sec")

    f.rect(CH_X - 56, CH_Y - 16, CH_W + 76, CH_H + 76, "#fff", "#dadce0", 1.2, 10)

    for pct in range(80, 106, 5):
        y = _py(pct)
        f.line(CH_X, y, CH_X + CH_W, y, "#eceff1" if pct != 100 else "#dadce0",
               1, dash=None if pct == 100 else "3 4")
        f.t(CH_X - 12, y + 4, "%d%%" % pct, "xxs", SUB, "end")

    for i, lab in enumerate(XS):
        a, b = lab.split("\n")
        f.t(_px(i), CH_Y + CH_H + 24, a, "lbl", INK, "middle")
        f.t(_px(i), CH_Y + CH_H + 42, b, "xxs", SUB, "middle")

    # Reduce(-18%) 和 Combine(-19%) 末端只差 0.9 个百分点，标签会叠在一起。
    # 这两条叠在一起**正是本图的论点之一**，所以不能靠拉开数据来躲 ——
    # 只把标签错开：紫线的标注往上抬一行。
    NUDGE = {PU: -13}
    for name, c, vals in SERIES:
        pts = [(_px(i), _py(100.0 * v / vals[0])) for i, v in enumerate(vals)]
        f.path("M" + " L".join("%.1f,%.1f" % p for p in pts), c, 2.4)
        for (px, py), v in zip(pts, vals):
            f.raw('<circle cx="%.1f" cy="%.1f" r="5" fill="#fff" stroke="%s" '
                  'stroke-width="2.2"/>' % (px, py, c))
        drop = 100.0 * vals[-1] / vals[0] - 100
        f.t(pts[-1][0] + 14, pts[-1][1] + 4 + NUDGE.get(c, 0),
            "%+.0f%%" % drop, "num", c)

    f.t(CH_X, CH_Y - 26, "相对 8 GPU 的带宽保持率", "lbl", INK)


# ══════════════════════════════════════════════════════════════════════
def _read(f):
    f.t(RD_X, TOP + 32, "怎么读　—— 先讲我们错在哪，再讲能确定什么", "sec")
    y = CH_Y - 16
    for head, c, hh, body in READ:
        f.rect(RD_X, y, RD_W, hh, FILL[c], c, 1.3, 8)
        para(f, RD_X + 16, y + 25, RD_W - 32, head, "box")
        para(f, RD_X + 16, y + 45, RD_W - 32, body, "xs", 17)
        y += hh + 10


# ══════════════════════════════════════════════════════════════════════
def _table(f):
    f.t(20, TB_Y - 12, "绝对值另附　—— 单位 GB/s，出处：本仓库 gpu/a4x/06-deepep-test/", "sec")
    cx = [20, 520, 800, 1080]
    f.rect(20, TB_Y, 1360, 34, "#f1f3f4", None, 0, 8)
    for x, h in zip(cx, ["操作", "8 GPU / 2 节点", "16 GPU / 4 节点", "32 GPU / 8 节点"]):
        f.t(x + 14, TB_Y + 23, h, "lbl", INK)
    for r, (name, c, vals) in enumerate(SERIES):
        y = TB_Y + 34 + r * 34
        if r % 2:
            f.rect(20, y, 1360, 34, "#fafafa", None, 0, 0)
        f.line(20, y, 1380, y, "#e8eaed", 1)
        para(f, cx[0] + 14, y + 22, 480, name, "xs")
        for i, v in enumerate(vals):
            f.t(cx[i + 1] + 14, y + 22, "{:,}".format(v), "num",
                INK if i < 2 else c)
    f.rect(20, TB_Y, 1360, TB_H, "none", "#dadce0", 1.2, 8)


# ══════════════════════════════════════════════════════════════════════
def _band(f):
    """落点带用黄框，不用蓝框。

    别的图那条蓝带是「本节结论」，这一条不是结论，是**一份免责声明**：
    我们能确定什么、不能确定什么、以及为什么。颜色要跟「结论带」区分开，
    否则听众会把「我们不知道」当成「我们的结论」。
    """
    f.rect(20, BAND_Y, 1360, 152, FILL[YL], YL, 1.6, 10)
    f.t(38, BAND_Y + 28, "⭐⭐ 本节落点，以及一条比结论更值钱的方法论", "sec")
    y = BAND_Y + 54
    for seg in [
            "<b>能确定的：规模一大，all-to-all 这类通信就掉，而且掉幅跟「要面对多少个对端」同向。</b>"
            "<r>不能确定的：掉幅里有多少来自消息变小、多少来自跨节点链路上的竞争。</r>"
            "<b>本图分不开这两者</b> —— 唯一的对照项 Copy 不出节点，"
            "而跨节点那条路我们没有独立的对照项。",
            "<b>方法论（这一条比上面的数据更该被带走）：</b>"
            "挑对照项时，第一个要问的不是「它稳不稳」，是"
            "<r>「它跟被测对象走的是不是同一条路」</r>。"
            "<b>一个走错路的对照项比没有对照项更危险</b> —— 它会让你以为自己排除了什么。",
            "<g>｜　源文档对 16 卡那次下降写的是「RDMA 竞争加剧」、对 32 卡写的是「碎片化」。"
            "这两条其实不矛盾：跨节点确实走 RDMA（部署要点里 NCCL_NET=gIB，"
            "8 节点那次的失败根因就是 ibv_modify_qp），两个因素同时在，只是本图分不开。"
            "　｜　主图归一化的理由：Copy 5,600、Dispatch 只有 700，画在同一根绝对值轴上，"
            "后三条会被压成贴地直线，而「掉了多少」恰恰是全部论点。</g>"]:
        y = para(f, 38, y, 1324, seg, "xs", 19) + 4


if __name__ == "__main__":
    import sys
    open(sys.argv[1] if len(sys.argv) > 1 else "out/fig_p6_deepep.svg",
         "w", encoding="utf-8").write(build())
