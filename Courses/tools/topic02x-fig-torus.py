# -*- coding: utf-8 -*-
r"""外传 图 X-8 · **4 个口怎么连成 256 颗** —— 二维环面，以及为什么这够用。

════════════════════════════════════════════════════════════════════
⭐ 这一张补 X-2 里第二个只标了存在、没展开的东西
════════════════════════════════════════════════════════════════════
X-2 那张右栏写了三行：**ICI 端口 4 个 · 拓扑 2D 环面 · Pod 256 颗**。
X-6 已经解释了「为什么只到 256 够用」（模型体积在一颗到几颗的量级）。
这一张补另一半：**这 4 个口具体怎么把 256 颗连起来，代价是什么。**

📌 两个能当场推的数（图上都给推导）：
  · **16 × 16 ＝ 256** ——&nbsp;官方支持的最大 2D 切片形状就是 `16x16`
  · **最远 16 跳**：环面每一维最多绕半圈，16 ÷ 2 ＝ 8，两维相加 ＝ **16**
    ⭐ 对照 v7 的 4×4×4 cube：每维最多 2 跳，三维相加 ＝ **6 跳**
    （这个对照在专题二 §4 出现过，这里只是换一代再算一次）

⭐⭐ 落点：**16 跳听着很多，但扩散不在乎。**
   模型一颗装得下（X-6），多卡就只是各生成各的 ——&nbsp;
   卡与卡之间几乎不用说话，跳数再多也没人走。
   ⛔ 反过来，要是一个模型必须摊在几百颗上、每层都要 all-reduce，
     16 跳就是实打实的成本 ——&nbsp;**那种活本来就该去找 v7 的 3D 环面。**
"""
from topic03_draw import (Fig, wpx, _sz, LINE, LINE2,
                          BL, OR, GR, RD, GY, GY2, PU, CY, INK)

W = 1400


def main():
    f = Fig(W, "TPU v6e 每颗芯片有 4 个 ICI 端口，连上下左右四个邻居，"
               "边缘绕回形成二维环面；一个 Pod 是 16 乘 16 共 256 颗，"
               "最远两点相距 16 跳。对照 v7 是 6 个端口的三维环面")
    f.marks = set()
    y = f.header(
        '4 个口怎么连成 256 颗 ——&#160;'
        '<tspan font-weight="700">二维环面，最远 16 跳，而扩散不在乎</tspan>',
        '⭐ X-2 右栏那三行（4 个 ICI 口 · 2D 环面 · Pod 256）在这儿展开。'
        'X-6 说了「为什么 256 够用」，<tspan font-weight="700">这一张说「怎么连、代价是什么」</tspan>。',
        [(CY, "ICI 链路"), (RD, "绕回去的那一跳"), (BL, "最远的一对")])

    top = y + 10
    PH = 430

    # ══ 左：一颗芯片的 4 个口 ══
    LW = 520
    ly = f.panel(0, top, LW, PH, "一颗芯片：4 个口，4 个邻居", CY,
                 tag="官方规格表")
    cx, cy = LW / 2.0, ly + 150
    f.cell(cx - 58, cy - 30, 116, 60, "v6e", "1 个 device", BL, "#fff", 13)
    NB = ((0, -110, "上"), (0, 110, "下"), (-160, 0, "左"), (160, 0, "右"))
    for dx, dy, lab in NB:
        f.cell(cx + dx - 46, cy + dy - 24, 92, 48, "邻居", None, GY2, "#fff", 12)
        # 链路
        if dx == 0:
            y1 = cy - 30 if dy < 0 else cy + 30
            y2 = cy + dy + (24 if dy < 0 else -24)
            f.line(cx, y1, cx, y2, CY, 2.0, arrow=False)
        else:
            x1 = cx - 58 if dx < 0 else cx + 58
            x2 = cx + dx + (46 if dx < 0 else -46)
            f.line(x1, cy, x2, cy, CY, 2.0, arrow=False)
    NB_BOT = cy + 110 + 24            # 最下面那个邻居的底沿
    f.lines(20, NB_BOT + 26, LW - 40, [
        "⭐ 每颗芯片<tspan font-weight=\"700\">只有 4 个 ICI 口</tspan>"
        "——&#160;所以只能连上下左右",
        "　 v7 有 6 个口，多出来的那两个用来连「前后」，于是它是三维的",
        "⭐ <tspan font-weight=\"700\">双向合计 800 GB/s</tspan>（每 chip）"
        "——&#160;这是 4 个口加起来的数",
        "⚠️ 边缘那一圈<tspan font-weight=\"700\">绕回对面</tspan>，"
        "所以叫「环面」而不是「网格」",
    ], size=11, lh=20, fill=GY)

    # ══ 右：整个 Pod ══
    RX = LW + 30
    RW = W - RX
    ry = f.panel(RX, top, RW, PH, "一个 Pod：16 × 16 ＝ 256 颗", CY,
                 tag="官方支持的最大 2D 切片形状")
    N = 16
    pitch = 15.0
    gx0 = RX + 30
    gy0 = ry + 26
    for r in range(N):
        for c in range(N):
            f.box(gx0 + c * pitch, gy0 + r * pitch, 9, 9, "#e4f7fb", CY, 2, 0.6)
    # 高亮最远的一对：(0,0) 与 (8,8)
    f.box(gx0 - 2, gy0 - 2, 13, 13, BL, BL, 3)
    f.box(gx0 + 8 * pitch - 2, gy0 + 8 * pitch - 2, 13, 13, BL, BL, 3)
    _lx = gx0 + N * pitch + 34
    f.line(gx0 + 8 * pitch + 12, gy0 + 8 * pitch + 4, _lx - 6,
           gy0 + 8 * pitch + 4, BL, 1, arrow=False)
    f.t(_lx, gy0 + 8 * pitch + 8, "这两颗相距最远　16 跳",
        "#174ea6", bold=True, size=_sz(11))
    # 绕回示意
    for k in range(3):
        yy = gy0 + (2 + k * 5) * pitch + 4
        f.line(gx0 + (N - 1) * pitch + 12, yy, gx0 + (N - 1) * pitch + 26, yy,
               RD, 1.2, arrow=False)
        f.line(gx0 - 20, yy, gx0 - 6, yy, RD, 1.2, arrow=True)
    f.t(gx0 + N * pitch + 34, gy0 + 6, "边缘绕回", "#a50e0e", size=_sz(11))

    bx = gx0
    by = gy0 + N * pitch + 22
    f.lines(bx, by, RW - 60, [
        "⭐ <tspan font-weight=\"700\">最远 16 跳</tspan>，这个数能当场推："
        "环面每一维最多绕半圈，16 ÷ 2 ＝ 8，两维相加 ＝ 16",
        "　 对照 v7 的 4×4×4 立方：每维最多 2 跳，三维相加 ＝ "
        "<tspan font-weight=\"700\">6 跳</tspan>（专题二 §4 算过）",
        "⛔ 16 跳听着很多 ——&#160;"
        "<tspan font-weight=\"700\">但要先问一句：谁会走这条路？</tspan>",
    ], size=11, lh=20, fill=GY)

    y = top + PH + 22

    y = f.band(y, "ok",
               "答案是：跑扩散的时候，几乎没人走",
               ['X-6 已经说清了：<tspan font-weight="700">这一族模型一颗就装得下</tspan>。'
                '于是多卡的用法是<tspan font-weight="700">各生成各的</tspan> ——&#160;'
                '八颗卡同时出八张图，<tspan font-weight="700">卡与卡之间几乎不用说话</tspan>。',
                '⭐ 通信量接近零的时候，<tspan font-weight="700">跳数是多少就不重要了</tspan>。'
                '省下来的那两个 ICI 口、那一个维度，'
                '换成了别的地方的面积和成本。',
                '⛔ 反过来这条也成立：要是一个模型必须摊在几百颗上、每一层都要 '
                'all-reduce，<tspan font-weight="700">16 跳就是实打实的成本</tspan> ——&#160;'
                '<tspan font-weight="700">那种活本来就该去找 v7 的三维环面。</tspan>'])

    y = f.src(y + 18,
              'ICI 端口 4 个、双向 800 GBps／chip、2D torus、Pod 256 chip、'
              '支持的 2D 切片形状最大到 16x16 ——&#160;官方 v6e 规格表',
              '「最远 16 跳」是本图当场推的：N×N 环面每维最多 ⌊N/2⌋ 跳，'
              '16÷2 ＝ 8，两维相加 16。v7 的 4×4×4 同法得 2+2+2 ＝ 6。'
              '⚠️ 这是<tspan font-weight="700">拓扑距离</tspan>，不是延迟实测。')
    f.save("figx-8.svg", y + 6)


main()
