# -*- coding: utf-8 -*-
"""图 P-5 —— 四个集合通信原语：分别是谁产生的，以及谁对拓扑敏感。

这一节最容易讲砸的方式，是把它讲成通信库教程。听众不关心 API，
关心的是**「我这个模型为什么会产生这种通信」** —— 所以每一行都必须
能指回专题一的某一步。

画法上唯一重要的决定：**给每个原语画一张通信形状的小图**。
前三个都是「沿着一个环走一圈」，画出来是一圈箭头；
all-to-all 是「每一对都要说话」，画出来是一张完全图 ——
**十条线和四条线的视觉差，比任何一句「它对拓扑极其敏感」都管用。**

这也是本节和 MoE 的接缝：分组限制路由不是省算力的技巧，是省拓扑的技巧。
"""
import math

from common import Fig, para, BL, RD, GN, YL, PU, INK, SUB, GREY, FILL

W = 1400
TOP = 84

HDR_H = 38
ROW_H = 146
COLS = [(20, 210), (230, 470), (700, 300), (1000, 380)]   # (x, w)

# ⚠️ 这一栏一度写成「FSDP / ZeRO-3」「张量并行」「数据并行的梯度同步」，
# 栏头却挂着「回指专题一」—— 而专题一里这些词**一次都没出现过**
# （FSDP / ZeRO / 张量并行 / 数据并行 / all-reduce / NCCL 全是 0 命中）。
# 专题一是《一个 token 的一生》，一堂**推理**课，根本没有反向传播。
#
# 真正指得回去的在专题一那张二选一的表里：
#   Dispatch / Combine  把 token 送到专家那儿  → all-to-all ×2
#   AG-RS               把专家权重取过来       → all-gather + reduce-scatter
# 也就是说四个原语里有三个都能落进**同一个 MoE 场景的两条替代路线**，
# 比拿训练侧词汇凑数好得多 —— 而且顺手把「这两条是二选一」讲了。
# 剩下 all-reduce 确实指不回去，就**明写指不回去**。
ROWS = [
    ("all-gather", GN, "ring",
     "<b>★ 专题一「AG-RS」那条路的前半</b>：不送 token，"
     "改<b>把专家权重取过来</b>，在本地算。<br>"
     "取之前要先把切开的权重收齐 —— 这一步就是 all-gather。",
     "一般。<b>拆成「沿环走一圈」就行</b> —— "
     "每一跳只跟邻居说话，<r>torus 上天然舒服</r>。"),
    ("reduce-scatter", GN, "ring",
     "<b>★ 同一条路的后半</b>：本地算完，"
     "再<b>一边归约、一边把结果切回各卡</b>。<br>"
     "<g>这两行合起来 ＝ 专题一那张表的第二行。</g>",
     "一般。<b>同样是沿环走一圈</b>，只是每跳多做一次加法。"),
    ("all-reduce", GN, "ring2",
     "<g>专题一没讲过它 —— 那是一堂推理课，没有反向传播。</g><br>"
     "它 ＝ reduce-scatter ＋ all-gather，<b>两圈</b>。"
     "<b>这里只为把四个凑齐，展开留给专题五。</b>",
     "一般。<b>两圈还是圈</b>，拓扑友好度和上面两个一样。"),
    ("all-to-all", RD, "full",
     "<r>★ 专题一「Dispatch / Combine」那条路</r>：<b>把 token 送到专家那儿</b>，"
     "算完送回。<br>"
     "而那个专家<b>可能在任意一张卡上</b>。"
     "<g>—— 和上面 AG-RS 那条是<b>二选一</b>，不是并列。</g>",
     "<r>极其敏感。</r><b>它天生要求「任意两点直接说话」。</b>"
     "环形算法不是不能用，<b>但要走很多轮，而且代价被 bisection 带宽卡死</b> —— "
     "<b>这是四个里唯一一个会因为拓扑不同而差出数量级的。</b>"),
]

TB_Y = TOP + 44
TB_H = HDR_H + len(ROWS) * ROW_H

BAND_Y = TB_Y + TB_H + 34
H = BAND_Y + 130


def build():
    f = Fig(W, H, "四个集合通信原语的来源与拓扑敏感度对照：all-gather、reduce-scatter、"
                  "all-reduce 都可拆成环形通信，只有 all-to-all 要求任意两点直连")
    f.title("四个集合通信原语　—— 分别是<tspan font-weight=\"700\" fill=\"#1a73e8\">谁产生的</tspan>，"
            "以及<tspan font-weight=\"700\" fill=\"#d93025\">谁对拓扑敏感</tspan>")
    f.legend([(GN, "能拆成「沿环走一圈」"), (RD, "每一对都要直接说话：环形算法要走很多轮"),
              (GREY, "小图里的点＝一张卡，线＝一次通信")])

    _table(f)
    _band(f)
    return f.out()


# ══════════════════════════════════════════════════════════════════════
def _table(f):
    f.rect(20, TB_Y, 1360, HDR_H, "#f1f3f4", None, 0, 8)
    for (x, _), h in zip(COLS, ["原语", "谁产生它（★ ＝ 专题一讲过）", "通信形状", "对拓扑敏感吗"]):
        f.t(x + 14, TB_Y + 24, h, "lbl", INK)

    for r, (name, c, shape, who, topo) in enumerate(ROWS):
        y = TB_Y + HDR_H + r * ROW_H
        if r % 2:
            f.rect(20, y, 1360, ROW_H, "#fafafa", None, 0, 0)
        if shape == "full":
            f.rect(20, y, 1360, ROW_H, FILL[RD], None, 0, 0)
        f.line(20, y, 1380, y, "#e8eaed", 1)

        f.t(COLS[0][0] + 14, y + 34, name, "box", c)
        yy = y + 30
        for seg in who.split("<br>"):
            yy = para(f, COLS[1][0] + 14, yy, COLS[1][1] - 28, seg, "xs", 17) + 4
        _shape(f, COLS[2][0] + COLS[2][1] / 2, y + ROW_H / 2, shape, c)
        para(f, COLS[3][0] + 14, y + 32, COLS[3][1] - 28, topo, "xs", 17)

    f.rect(20, TB_Y, 1360, TB_H, "none", "#dadce0", 1.2, 8)


# ══════════════════════════════════════════════════════════════════════
def _nodes(cx, cy, n, r):
    """n 个点均匀摆在圆上，从正上方开始顺时针。"""
    return [(cx + r * math.sin(2 * math.pi * i / n),
             cy - r * math.cos(2 * math.pi * i / n)) for i in range(n)]


def _shape(f, cx, cy, kind, c):
    """通信形状小图 —— 这张图真正的信息量在这里，不在文字栏。

    环形那三种画 5 个点 5 条边；all-to-all 同样 5 个点，
    但要画 C(5,2) ＝ 10 条边。**边数从 5 跳到 10 是这张图的全部论点**，
    所以点数必须一致，否则对比就不成立了。
    """
    n, r = 5, 34
    pts = _nodes(cx, cy, n, r)

    if kind == "full":
        for i in range(n):
            for j in range(i + 1, n):
                f.line(pts[i][0], pts[i][1], pts[j][0], pts[j][1], c, 1.1)
    else:
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            # 箭头顶到圆点边上会被挡住，沿线回缩 9px
            dx, dy = b[0] - a[0], b[1] - a[1]
            L = math.hypot(dx, dy)
            f.line(a[0] + dx * 9 / L, a[1] + dy * 9 / L,
                   b[0] - dx * 9 / L, b[1] - dy * 9 / L, c, 1.4,
                   "aG" if c == GN else "aR")
        if kind == "ring2":
            f.raw('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="none" stroke="%s" '
                  'stroke-width="1.1" stroke-dasharray="4 4"/>' % (cx, cy, r - 13, c))

    for px, py in pts:
        f.raw('<circle cx="%.1f" cy="%.1f" r="6.5" fill="#fff" stroke="%s" '
              'stroke-width="1.6"/>' % (px, py, c))

    lab = {"ring": "5 个点，5 条边", "ring2": "5 个点，走两圈",
           "full": "5 个点，10 条边"}[kind]
    f.t(cx, cy + r + 24, lab, "xxs", RD if kind == "full" else SUB, "middle")


# ══════════════════════════════════════════════════════════════════════
def _band(f):
    f.rect(20, BAND_Y, 1360, 114, FILL[BL], BL, 1.4, 10)
    f.t(38, BAND_Y + 28, "一句话记住区别 —— 顺便把 MoE 那件事接上", "sec")
    y = BAND_Y + 54
    for seg in [
            "<b>前三个都能拆成「沿着某个环走一圈」，所以在 torus 上也很舒服。</b>"
            "只有 all-to-all 天生要求任意两点直接说话 —— "
            "<r>它是唯一一个会因为拓扑不同而表现出数量级差异的原语。</r>",
            "这就把两件事接上了：<b>为什么 MoE 是 TPU 上最难的那类负载</b>，"
            "以及<b>为什么分组限制路由不是省算力的技巧，是省拓扑的技巧</b> —— "
            "它限制的正是「一个 token 最多能跑到几台机器上去」。"]:
        y = para(f, 38, y, 1324, seg, "xs", 19) + 4


if __name__ == "__main__":
    import sys
    open(sys.argv[1] if len(sys.argv) > 1 else "out/fig_p5_collectives.svg",
         "w", encoding="utf-8").write(build())
