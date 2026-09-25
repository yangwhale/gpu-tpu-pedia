# -*- coding: utf-8 -*-
r"""专题五「卡和块」画法的共用基元：四张卡四种颜色，每张卡四块，条纹＝加过，虚线框＝空位。

⭐ 为什么单独一个模块：集合通信那几张图（topic05-fig-coll.py）和 ZeRO 那张「一步」图（topic05-fig-zero.py）
   用的是同一套画法 —— 课件 §1.1「先学会看图」教的就是它。各抄一份会慢慢漂开，而且漂了不报错。
"""
from topic03_draw import BL, OR, GR, PU, INK, GY2, LINE

W = 1400
N = 4
COL = [BL, OR, GR, PU]                                  # 卡 0..3 的身份色
MID = {BL: "#aecbfa", OR: "#fdc69c", GR: "#a8dab5", PU: "#d7aefb"}   # 200 档，条纹用
NAME = "ABCD"                                           # 卡 k 的数据叫 A/B/C/D

CW, CH, GAP = 46, 30, 6                                 # 一块的宽、高、间距
RH = 40                                                 # 一张卡一行


def chunk(f, x, y, contrib, label=None, hot=False):
    """画一块。contrib 是参与这块的卡号列表；空列表 = 虚线空位。"""
    if not contrib:
        f.box(x, y, CW, CH, "none", LINE, 5, dash="4,3")
        return
    k = len(contrib)
    sw = CW / float(k)
    for i, c in enumerate(sorted(contrib)):
        # 条纹：每个参与者一条竖带
        f.box(x + i * sw, y, sw + (0.5 if i < k - 1 else 0), CH, MID[COL[c]], "none", 0)
    f.box(x, y, CW, CH, "none", INK if hot else GY2, 5, sw=2.2 if hot else 1)
    if label:
        f.t(x + CW / 2.0, y + CH / 2.0 + 5, label, INK, True, 13, "middle")


def row(f, x, y, cells, hot=()):
    """一张卡的一排块。cells[j] = (contrib, label)。"""
    for j, (c, lab) in enumerate(cells):
        chunk(f, x + j * (CW + GAP), y, c, lab, j in hot)
    return x + len(cells) * (CW + GAP) - GAP


def rowlab(f, x, y, k):
    f.t(x, y + CH / 2.0 + 5, "卡 %d" % k, COL[k], True, 14)


def sumlab(j):
    return "Σ%d" % j


# ── 四种状态的生成器（返回 4 行，每行 4 块） ─────────────────────
def full(k):             # 卡 k 手里一整份自己的数据：A0 A1 A2 A3
    return [([k], "%s%d" % (NAME[k], j)) for j in range(N)]


def empty():
    return [([], None)] * N


def summed_all():
    return [(list(range(N)), sumlab(j)) for j in range(N)]


