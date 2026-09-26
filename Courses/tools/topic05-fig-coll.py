# -*- coding: utf-8 -*-
r"""专题五 · 第一节「先认识五种通信」的五张图。

⭐ 一套画法贯穿五张图，读者只学一次：
   · 四张卡，一张卡一个颜色（蓝 / 橙 / 绿 / 紫）。
   · 每张卡手里的数据切成四块，一块一个小方格。
   · **被加过的块画成竖条纹，条纹的颜色就是参与相加的那几张卡。**
     于是「拼」和「加」一眼就分得开：拼出来的是一排不同颜色的方格，
     加出来的是一个方格里挤着好几种颜色。
   · 空位画虚线框：这里本来没有数据。

⛔ 环形那张图的每一步**不是手摆的**，是按「第 s 步，卡 k 把第 (k−s−1) mod n 块
   发给右邻居，右邻居加到自己那份上」现算出来的，并断言三步后每张卡恰好
   握着一整块总和。跟 wanghonglei 那篇推演、NCCL 的 ring 是同一个调度，
   只是块号整体挪了一位（原文是 k−s），让卡 k 最后拿第 k 块、跟另外几张图对齐。

⛔ 刻意没画（2026-09-25 对照构建手册补）：八个原语只画「逻辑视图」—— 谁发给谁、结果长什么样；
   不画实际用的算法（环形、树形、分层），那是 fig-ring 和动画的事。也不画延迟，只讲带宽。
"""
from topic03_draw import Fig, BL, OR, GR, PU, GY, INK, GY2, LINE, RD as RD_

W = 1400
from topic05_blocks import (N, COL, MID, NAME, CW, CH, GAP, RH, chunk, row, rowlab, sumlab,   # noqa: E402
                           full, empty, summed_all)


def panel_pair(f, x, y, w, title, sub, before, after, note, col):
    """一个原语：左边「之前」四行，右边「之后」四行。"""
    py = f.panel(x, y, w, 30 + 26 + N * RH + 50, title, col, sub=sub)
    lx = x + 18
    bx = x + 76
    ax = bx + N * (CW + GAP) + 66
    f.t(bx, py + 20, "之前", GY, True, 13)
    f.t(ax, py + 20, "之后", GY, True, 13)
    for k in range(N):
        yy = py + 30 + k * RH
        rowlab(f, lx, yy, k)
        row(f, bx, yy, before[k])
        row(f, ax, yy, after[k])
    mid_y = py + 30 + N * RH / 2.0 - 4
    f.line(bx + N * (CW + GAP) + 8, mid_y, ax - 14, mid_y, GY2, 1.6)
    f.t(x + 18, py + 30 + N * RH + 26, note, GY, size=13.5)
    f._pan = None
    return y + 30 + 26 + N * RH + 50


# ════════════════════════════════════════════════════════════════
# 图一：一对多、多对一
# ════════════════════════════════════════════════════════════════
def fig_1n():
    f = Fig(W, "一对多和多对一的四个基本动作。广播：卡零把自己的一整份数据发给所有人，"
               "之后四张卡手里是同一份 A。分发：卡零把自己的四块拆开，一人一块。"
               "收集：每人一块，全部交给卡零，卡零拼成一整份。"
               "归约：每人一整份，全部交给卡零并逐块相加，卡零手里是四块总和。"
               "前两个是一变多，后两个是多变一；归约和收集的区别只在于到了之后加不加")
    y0 = f.header("先认识四个基本动作　——　<tspan font-weight=\"700\">一个人对所有人</tspan>",
                  "四张卡，一张一个颜色。数据切成四块。"
                  "<tspan font-weight=\"700\">条纹块 ＝ 几张卡的数加在了一起</tspan>，"
                  "虚线框 ＝ 这里没有数据",
                  [(BL, "卡 0 的数据 A"), (OR, "卡 1 的 B"), (GR, "卡 2 的 C"), (PU, "卡 3 的 D")])
    PW = 680
    one = [full(0)] + [empty() for _ in range(N - 1)]
    bc_after = [full(0) for _ in range(N)]
    sc_after = [[([0], "A%d" % j) if j == k else ([], None) for j in range(N)] for k in range(N)]
    ga_before = [[([k], "%s%d" % (NAME[k], j)) if j == k else ([], None) for j in range(N)]
                 for k in range(N)]
    ga_after = [[([j], "%s%d" % (NAME[j], j)) for j in range(N)]] + [empty() for _ in range(N - 1)]
    rd_before = [full(k) for k in range(N)]
    rd_after = [summed_all()] + [empty() for _ in range(N - 1)]

    y1 = panel_pair(f, 0, y0, PW, "Broadcast　广播", "一份 → 人人一份",
                    one, bc_after, "卡 0 的整份数据，复制给每一个人", BL)
    panel_pair(f, W - PW, y0, PW, "Scatter　分发", "一份拆开 → 一人一块",
               one, sc_after, "卡 0 把自己那份切成四块，一人发一块", BL)
    y2 = panel_pair(f, 0, y1 + 20, PW, "Gather　收集", "一人一块 → 拼成一份",
                    ga_before, ga_after, "每人交一块给卡 0，卡 0 按顺序拼起来", OR)
    panel_pair(f, W - PW, y1 + 20, PW, "Reduce　归约", "人人一份 → 加成一份",
               rd_before, rd_after, "每人交一整份给卡 0，卡 0 逐块相加", OR)

    yb = f.band(y2 + 20, "ok", "两两成对，差别只在一个动作", [
        "广播和分发是<tspan font-weight=\"700\">一变多</tspan>，收集和归约是<tspan font-weight=\"700\">多变一</tspan>。"
        "　收集和归约的区别只在于：数据到了之后<tspan font-weight=\"700\">拼起来</tspan>还是<tspan font-weight=\"700\">加起来</tspan>。",
        "这四个都有一个「班长」卡 0，按最朴素的做法，所有流量都挤在它一个人的线上。"
        "训练里真正天天在跑的是<tspan font-weight=\"700\">没有班长</tspan>的几个：AllGather、ReduceScatter、AllReduce；AllToAll 则是每个人各做一次分发。",
    ])
    f.save("fig5-coll-1n.svg", yb + 14)


# ════════════════════════════════════════════════════════════════
# 图二：人人对人人
# ════════════════════════════════════════════════════════════════
def fig_nn():
    f = Fig(W, "人人对人人的四个集合通信。全收集：每人一块，结束后每人都有完整的四块。"
               "归约分散：每人一整份，结束后卡 k 只拿到第 k 块的总和。"
               "全归约：每人一整份，结束后每人都拿到四块总和。"
               "全交换：卡 k 给卡 j 寄一份，份份不同，一般多少也不同；这里画的是每份一样大的特例，"
               "结束后卡 j 手里是四张卡各自的第 j 块。前三个是把收集、归约的结果发给所有人，"
               "全交换不加也不拼，只是换位置")
    y0 = f.header("训练里天天在跑的四个　——　<tspan font-weight=\"700\">人人对人人，没有班长</tspan>",
                  "名字前面带 All 的，意思是<tspan font-weight=\"700\">结果人人都有一份</tspan>。"
                  "画法不变：条纹 ＝ 加过，虚线 ＝ 空",
                  [(BL, "卡 0"), (OR, "卡 1"), (GR, "卡 2"), (PU, "卡 3")])
    PW = 680
    ag_before = [[([k], NAME[k]) if j == k else ([], None) for j in range(N)] for k in range(N)]
    ag_after = [[([j], NAME[j]) for j in range(N)] for _ in range(N)]
    rs_before = [full(k) for k in range(N)]
    rs_after = [[(list(range(N)), sumlab(j)) if j == k else ([], None) for j in range(N)]
                for k in range(N)]
    ar_after = [summed_all() for _ in range(N)]
    a2a_after = [[([j], "%s%d" % (NAME[j], k)) for j in range(N)] for k in range(N)]

    y1 = panel_pair(f, 0, y0, PW, "AllGather　全收集", "一人一块 → 人人一整份",
                    ag_before, ag_after, "只拼不加。FSDP 前向时用它把整层权重拼回来", BL)
    panel_pair(f, W - PW, y0, PW, "ReduceScatter　归约分散", "人人一整份 → 各拿一块总和",
               rs_before, rs_after, "先加再分。卡 k 只拿到第 k 块的总和", GR)
    y2 = panel_pair(f, 0, y1 + 20, PW, "AllReduce　全归约", "人人一整份 → 人人一份总和",
                    rs_before, ar_after, "数据并行同步梯度用的就是它", GR)
    panel_pair(f, W - PW, y1 + 20, PW, "AllToAll　全交换", "卡 k 给卡 j 寄一份（这里画每份一样大的特例）",
               [full(k) for k in range(N)], a2a_after, "不加也不拼，只换位置。MoE 派发 token 用它", PU)

    yb = f.band(y2 + 20, "ok", "前三个是一家人，第四个是另一回事", [
        "AllGather 是拼，ReduceScatter 是加完再分，AllReduce 是加完人人一份。"
        "<tspan font-weight=\"700\">拆开来看，AllReduce 正好等于前两个接起来。</tspan>",
        "AllToAll 不加也不拼：每张卡给每张卡发一份<tspan font-weight=\"700\">不一样的</tspan>。"
        "任意两张卡之间都有流量，所以它对网络最挑剔。",
    ])
    f.save("fig5-coll-nn.svg", yb + 14)


# ════════════════════════════════════════════════════════════════
# 图三：AllReduce ＝ ReduceScatter ＋ AllGather
# ════════════════════════════════════════════════════════════════
def fig_split():
    f = Fig(W, "全归约可以拆成两步。开始时每张卡一整份自己的数据。"
               "第一步归约分散：每张卡拿到一块总和，卡零拿第零块，卡一拿第一块，以此类推。"
               "第二步全收集：把这四块总和拼给每一个人。"
               "两步做完，每人手里都是四块总和，跟直接做一次全归约结果完全一样。"
               "每一步每张卡要发出去的数据是整份的四分之三，两步加起来是二乘四分之三")
    y0 = f.header("全课最重要的一个等式　——　<tspan font-weight=\"700\">AllReduce ＝ ReduceScatter ＋ AllGather</tspan>",
                  "同一个 AllReduce，拆成两半来做，结果一模一样。"
                  "<tspan font-weight=\"700\">后面好几种并行，用的就是这两半中的某一半</tspan>",
                  [(BL, "卡 0"), (OR, "卡 1"), (GR, "卡 2"), (PU, "卡 3")])
    PH = 30 + 34 + N * RH + 40
    py = f.panel(0, y0, W, PH, "拆成两半", GR)
    states = [
        ("开始：人人一整份", [full(k) for k in range(N)]),
        ("ReduceScatter 之后：各拿一块总和",
         [[(list(range(N)), sumlab(j)) if j == k else ([], None) for j in range(N)] for k in range(N)]),
        ("AllGather 之后：人人一份总和", [summed_all() for _ in range(N)]),
    ]
    SX = [70, 540, 1010]
    for i, (lab, st) in enumerate(states):
        f.t(SX[i], py + 26, lab, INK, True, 14)
        for k in range(N):
            yy = py + 40 + k * RH
            if i == 0:
                rowlab(f, 16, yy, k)
            row(f, SX[i], yy, st[k])
    mid = py + 40 + N * RH / 2.0 - 4
    for i, (a, b) in enumerate([("① ReduceScatter", "先加，每人留一块"),
                                ("② AllGather", "再拼，发给每个人")]):
        x1 = SX[i] + N * (CW + GAP) + 16
        x2 = SX[i + 1] - 24
        f.line(x1, mid, x2, mid, GR, 2)
        f.t((x1 + x2) / 2.0, mid - 14, a, GR, True, 14, "middle")
        f.t((x1 + x2) / 2.0, mid + 24, b, GY, size=13, anchor="middle")
    f.t(16, py + 40 + N * RH + 22,
        "每一步，每张卡发出去的数据都是整份的 (n−1)/n；两步加起来 2(n−1)/n　——　"
        "跟 NCCL 给 AllReduce 定的带宽系数是同一个数", GY, size=13.5)
    f._pan = None

    yb = f.band(py + PH + 20, "ok", "所以后面能反复白捡便宜", [
        "数据并行本来就要做一次 AllReduce。<tspan font-weight=\"700\">ZeRO 把它拆开</tspan>："
        "先用 ReduceScatter 分梯度、各自更新，再用 AllGather 拼回新权重　——　前两级通信总量不变，显存却省下来了。",
        "同理，张量并行配上序列并行时，也是把每层那次 AllReduce 拆成这两半，分别挪到不同位置去做。",
    ])
    f.save("fig5-ar-split.svg", yb + 14)


# ════════════════════════════════════════════════════════════════
# 图四：环形 ReduceScatter 逐步推演（现算，不手摆）
# ════════════════════════════════════════════════════════════════
def ring_states():
    """返回每一步之后的 held[k][j]（参与相加的卡号集合）和本步刚被加过的块。"""
    held = [[{k} for _ in range(N)] for k in range(N)]
    out = [([[set(s) for s in r] for r in held], set())]
    for s in range(N - 1):
        # 比 NCCL / 原文的 (k−s) 多减一：让卡 k 最后握着第 k 块，跟另外几张图对齐
        sends = [(k, (k - s - 1) % N) for k in range(N)]
        new = [[set(c) for c in r] for r in held]
        hot = set()
        for k, j in sends:
            dst = (k + 1) % N
            new[dst][j] |= held[k][j]
            hot.add((dst, j))
        held = new
        out.append(([[set(c) for c in r] for r in held], hot))
    # ⛔ 推演对不对，靠断言而不是靠眼睛：三步后卡 k 恰好握着第 k 块的完整总和
    final = out[-1][0]
    for k in range(N):
        assert final[k][k] == set(range(N)), (k, final[k])
    return out


def fig_ring():
    f = Fig(W, "环形归约分散的逐步推演。四张卡连成一个环，每人只跟右边邻居说话。"
               "每一步，每张卡同时往右发一块、从左收一块，收到的那块加到自己的同号块上。"
               "第一步后，每张卡有一块是两个人的和；第二步后是三个人的和；"
               "第三步后，每张卡恰好握着一块完整的四人总和。"
               "再用同样的环把这四块总和转三步，就是全收集，全归约就做完了。"
               "每一步每条线上只走一块数据，所以卡再多，每张卡发出去的总量也不超过两整份")
    y0 = f.header("没有班长，怎么做到的　——　<tspan font-weight=\"700\">连成一个环，每人只跟右边的邻居说话</tspan>",
                  "每一步：每张卡同时往右发一块、从左收一块，<tspan font-weight=\"700\">收到的加到自己的同号块上</tspan>。"
                  "粗框 ＝ 这一步刚加过的块",
                  [(BL, "卡 0"), (OR, "卡 1"), (GR, "卡 2"), (PU, "卡 3")])
    # ⭐ 2026-09-25 逐图审：标题说「连成一个环」，画面里却没有环。右上角补一个四卡小环。
    RC, RR = (1310, 54), 30
    pos = [(RC[0], RC[1] - RR), (RC[0] + RR, RC[1]), (RC[0], RC[1] + RR), (RC[0] - RR, RC[1])]
    for k in range(4):
        (x1, y1), (x2, y2) = pos[k], pos[(k + 1) % 4]
        f.path("M%.1f,%.1f Q%.1f,%.1f %.1f,%.1f" % (x1 + (x2 - x1) * 0.28, y1 + (y2 - y1) * 0.28,
               (x1 + x2) / 2 + (x1 + x2 - 2 * RC[0]) * 0.35, (y1 + y2) / 2 + (y1 + y2 - 2 * RC[1]) * 0.35,
               x1 + (x2 - x1) * 0.72, y1 + (y2 - y1) * 0.72), GY, 1.6)
    for k, (cx, cy) in enumerate(pos):
        col = (BL, OR, GR, PU)[k]
        f.p.append('<circle cx="%.1f" cy="%.1f" r="11" fill="%s"/>' % (cx, cy, col))
        f.t(cx, cy + 5, str(k), "#ffffff", True, 13, "middle")
    st = ring_states()
    PH = 30 + 34 + N * RH + 44
    py = f.panel(0, y0, W, PH, "环形 ReduceScatter，四张卡、三步", GR,
                 sub="每一步：每张卡把一块发给右边的邻居，同时从左边收一块、加到自己的同号块上")
    SX = [70, 400, 730, 1060]
    labs = ["开始", "第 1 步后：两人之和", "第 2 步后：三人之和", "第 3 步后：每人一块完整总和"]
    for i, (held, hot) in enumerate(st):
        f.t(SX[i], py + 26, labs[i], INK if i < 3 else GR, True, 14)
        for k in range(N):
            yy = py + 40 + k * RH
            if i == 0:
                rowlab(f, 16, yy, k)
            cells = []
            for j in range(N):
                c = sorted(held[k][j])
                lab = sumlab(j) if len(c) == N else ("%s%d" % (NAME[k], j) if len(c) == 1 else None)
                cells.append((c, lab))
            row(f, SX[i], yy, cells, hot={j for (kk, j) in hot if kk == k})
    f.t(16, py + 40 + N * RH + 24,
        "接着用同一个环再转三步，只拼不加，把这四块总和发给每个人　——　那就是 AllGather，AllReduce 到此做完",
        GY, size=13.5)
    f._pan = None

    yb = f.band(py + PH + 20, "ok", "环的好处：没有人闲着，也没有人被挤爆", [
        "班长模式里所有数据都挤进卡 0 一条线；环形里<tspan font-weight=\"700\">每一步每条线上只走一块</tspan>，所有线同时在用。",
        "所以卡再多，每张卡发出去的总量也只是 2(n−1)/n 份，<tspan font-weight=\"700\">不到两整份</tspan>。"
        "代价是步数随卡数增长：数据很小时，比的就不是带宽而是步数了。",
    ])
    f.save("fig5-ring.svg", yb + 14)


# ════════════════════════════════════════════════════════════════
# 图五：AllToAll —— 一般情形是「每人给每人寄的多少不一样」，转置只是等量特例
# ⛔⛔ 2026-09-26 现场纠正：「转置只能说是 AllToAll 的一种特例……你得把 dispatch 和 combine
#   这种乱射之后又原路回来了的感觉表现出来。」原图整张画成转置，讲成了特例。
# ⭐ 改成：主画面用专家并行的派发 —— 每个 token 由路由定去哪张卡，每人寄给每人的数目不等
#   （线粗 ∝ 数目），收件人忙闲不均；再画合并原路寄回；转置降成右下角的小图，标「每份一样大时」。
# ⛔ 数目 A2A_C 是示意（固定），不是实测路由分布。
# ════════════════════════════════════════════════════════════════
import random as _rnd                                                  # noqa: E402

A2A_C = [[2, 3, 1, 2], [3, 1, 2, 2], [4, 2, 1, 1], [2, 2, 3, 1]]      # A2A_C[k][d]：卡 k 寄给卡 d 几个 token
A2A_IN = [sum(A2A_C[k][d] for k in range(N)) for d in range(N)]
assert [sum(r) for r in A2A_C] == [8] * N and A2A_IN == [11, 8, 7, 6]
TOK = 26


def _tokens(k):
    lst = [d for d in range(N) for _ in range(A2A_C[k][d])]
    _rnd.Random(10 + k).shuffle(lst)
    return lst


def fig_a2a():
    f = Fig(W, "全交换的一般情形：每张卡给每张卡各寄一份，而且多少不一样。以专家并行为例，"
               "每张卡有 8 个 token，每个 token 由路由决定去哪张卡。派发时所有卡同时往所有方向寄，"
               "线越粗寄得越多；收件那边忙闲不均，卡 0 收了 11 个，卡 3 只收了 6 个。"
               "专家算完，再沿原路寄回出发的卡，这是合并，又一次全交换。"
               "每份一样大时，全交换正好是把一张表转置一次，这只是特例，Ulysses 用的就是它")
    y0 = f.header("AllToAll　——　<tspan font-weight=\"700\">每人给每人寄一份，份份不同，多少也不同</tspan>",
                  "以专家并行为例：每个 token 由路由定好去哪张卡（格子里的数字）；颜色 ＝ 从哪张卡出发",
                  [(BL, "卡 0 出发"), (OR, "卡 1 出发"), (GR, "卡 2 出发"), (PU, "卡 3 出发")])
    PH = 500
    py = f.panel(0, y0, W, PH, "① 派发：人人同时往所有方向寄，线越粗寄得越多", PU)
    LX, RX, RW0 = 70, 930, 90
    ROWY = [py + 70 + k * 100 for k in range(N)]
    f.t(LX, py + 30, "出发：每张卡 8 个 token，各去各的卡", INK, True, 14)
    f.t(RX, py + 30, "收到：按出发的卡排好", INK, True, 14)
    lx_end = LX + 8 * (TOK + 4) + 10
    for k in range(N):
        yy = ROWY[k]
        f.t(LX - 12, yy + 20, "卡 %d" % k, COL[k], True, 14, "end")
        for i, d in enumerate(_tokens(k)):
            x = LX + i * (TOK + 4)
            f.box(x, yy, TOK, TOK, COL[k], COL[k], 3)
            f.t(x + TOK / 2.0, yy + 18, "%d" % d, "#ffffff", True, 13, "middle")
    for d in range(N):
        yy = ROWY[d]
        f.t(RX - 12, yy + 20, "卡 %d" % d, INK, True, 14, "end")
        x = RX
        for k in range(N):
            for _ in range(A2A_C[k][d]):
                f.box(x, yy, TOK, TOK, COL[k], COL[k], 3)
                x += TOK + 4
            x += 8
        f.t(x + 6, yy + 19, "收 %d 个" % A2A_IN[d], RD_ if d == 0 else GY, d == 0, 13.5)
    for k in range(N):
        for d in range(N):
            y1, y2 = ROWY[k] + TOK / 2.0, ROWY[d] + TOK / 2.0
            f.line(lx_end, y1, RX - 70, y2, COL[k], 1.2 + 2.2 * A2A_C[k][d],
                   dash="4,4" if k == d else None, arrow=False)
    f.t(lx_end + 10, py + PH - 70, "虚线 ＝ 寄给自己：不走网络。其余每一根都要跨卡，而且全部同时出发", GY, size=13)
    f.t(lx_end + 10, py + PH - 46, "卡 0 收了 11 个、卡 3 只收 6 个：大家都得等卡 0 算完", RD_, True, 13.5)

    y2 = py + PH - 30 + 20
    PH2 = 250
    py = f.panel(0, y2, 820, PH2, "② 合并：专家算完，原路寄回", PU)
    for k in range(N):
        yy = py + 30 + k * 48
        f.t(40, yy + 18, "卡 %d" % k, COL[k], True, 13.5)
        f.t(560, yy + 18, "卡 %d" % k, INK, True, 13.5)
        for d in range(N):
            f.line(540, py + 30 + d * 48 + 12, 100, yy + 12, COL[k], 1 + 1.6 * A2A_C[k][d],
                   dash="4,4" if k == d else None, arrow=True)
    f.t(610, py + 60, "同一批线，", INK, True, 13.5)
    f.t(610, py + 84, "方向反过来：", INK, True, 13.5)
    f.t(610, py + 112, "每个 token 回到", GY, size=13)
    f.t(610, py + 134, "出发的卡、原来的位置", GY, size=13)
    f.t(610, py + 170, "每层两次：派发一次，", PU, True, 13.5)
    f.t(610, py + 192, "合并一次", PU, True, 13.5)

    px = 840
    py3 = f.panel(px, y2, W - px, PH2, "③ 特例：每份一样大，就是一次转置", GY)
    SC = 22
    for k in range(N):
        for j in range(N):
            f.box(px + 30 + j * (SC + 3), py3 + 40 + k * (SC + 3), SC, SC, COL[k], COL[k], 2)
            f.box(px + 250 + j * (SC + 3), py3 + 40 + k * (SC + 3), SC, SC, COL[j], COL[j], 2)
    f.t(px + 30, py3 + 26, "卡 k 一行", GY, size=12.5)
    f.t(px + 250, py3 + 26, "卡 j 一行", GY, size=12.5)
    f.line(px + 140, py3 + 88, px + 236, py3 + 88, GY2, 1.6)
    f.t(px + 188, py3 + 78, "转置", GY, True, 13, "middle")
    f.t(px + 30, py3 + 170, "Ulysses 在按段、按头之间换，", INK, size=13)
    f.t(px + 30, py3 + 192, "用的就是这个等量的特例", INK, size=13)
    f._pan = None
    yb = f.band(py + PH2 - 10, "ok", "为什么网络最怕它", [
        "每份多大要等路由算完才知道（数目不等时常叫 AllToAllv），收件的人忙闲不均，最忙的那张卡决定大家等多久。",
        "每一份都是私信：路上没法像 AllReduce 那样边走边合并，也排不成只跟邻居说话的环，任意两张卡之间都有流量，拼的是整个网络的横截面。",
    ])
    yb = f.src(yb + 10, "⚠️ 示意：每张卡寄给每张卡几个 token 是随手定的固定数，不是实测的路由分布；真实的 V3 每个 token 挑 8 个专家。")
    f.save("fig5-a2a.svg", yb + 14)


# ── fig-ar-two-ways：同一个 AllReduce 的两种拼法（2026-09-25 夜 · 蒸馏 R2） ─────────────
# ⭐ 讲法借自 NCCL 文档：AllReduce 既能拼成 Reduce ＋ Broadcast，也能拼成 ReduceScatter ＋ AllGather。
#   等式的「啊哈」不在「它成立」，在「它是赢家」：两种拼法并排，一种班长累死，一种人人平摊。
# ⛔ 数字现算：朴素班长做法里卡 0 收 n−1 份、发 n−1 份（S ＝ 一整份数据）；环上每张卡 RS 发 (n−1)/n、AG 再发 (n−1)/n。
def star_send(n):
    return n - 1


def ring_send(n):
    return 2 * (n - 1) / n


assert ring_send(4) == 1.5 and star_send(4) == 3
assert [round(ring_send(n), 3) for n in (2, 4, 8, 1000)] == [1.0, 1.5, 1.75, 1.998]


def fig_ar_two_ways():
    from topic03_draw import RD as _RD
    f = Fig(W, "同一个 AllReduce 的两种拼法。左边是最朴素的班长做法：先归约到卡 0，再从卡 0 广播回去，卡 0 要发 3 份，别人各发 1 份。"
               "中间是环上的拼法：先 ReduceScatter 再 AllGather，四张卡每张都只发 1.5 份。"
               "右边把卡数加上去：班长那条线要发的份数跟卡数一起涨，n 减 1；环上每张卡发 2 乘 n 减 1 除以 n，永远不到 2 份")
    y0 = f.header("同一个 AllReduce，两种拼法　——　<tspan font-weight=\"700\">班长累死，还是人人平摊</tspan>",
                  "S ＝ 一整份数据。条长 ＝ 这张卡一共要发出去多少份（最朴素的班长做法，不排链）",
                  [(BL, "卡 0"), (OR, "卡 1"), (GR, "卡 2"), (PU, "卡 3")])
    PH = 330
    CW3 = 430
    # 左：Reduce ＋ Broadcast
    py = f.panel(0, y0, CW3, PH, "拼法一：Reduce ＋ Broadcast（班长）", _RD)
    sc = 80
    for k in range(4):
        v = 3 if k == 0 else 1
        yy = py + 40 + k * 52
        f.t(20, yy + 22, "卡 %d" % k, (BL, OR, GR, PU)[k], True, 14)
        f.box(80, yy, v * sc, 32, (BL, OR, GR, PU)[k], (BL, OR, GR, PU)[k], 3)
        f.t(80 + v * sc + 10, yy + 22, "%d 份" % v, INK, True, 14)
    f.t(20, py + PH - 46, "卡 0 那一条线扛下全部，卡越多越堵", _RD, True, 14)
    f._pan = None
    # 中：RS ＋ AG
    px = CW3 + 25
    py2 = f.panel(px, y0, CW3, PH, "拼法二：ReduceScatter ＋ AllGather（环）", GR)
    for k in range(4):
        yy = py2 + 40 + k * 52
        c = (BL, OR, GR, PU)[k]
        f.t(px + 20, yy + 22, "卡 %d" % k, c, True, 14)
        f.box(px + 80, yy, 0.75 * sc, 32, c, c, 3)
        f.box(px + 80 + 0.75 * sc + 2, yy, 0.75 * sc, 32, "none", c, 3, sw=2)
        f.t(px + 80 + 1.5 * sc + 12, yy + 22, "1.5 份", INK, True, 14)
    f.t(px + 20, py2 + PH - 70, "实心 ＝ RS 那 3 步，空心 ＝ AG 那 3 步，各 3/4 份", GY, size=13)
    f.t(px + 20, py2 + PH - 46, "人人一样忙，所有的线同时在用", GR, True, 14)
    f._pan = None
    # 右：卡数加上去
    px3 = 2 * (CW3 + 25)
    RW = W - px3
    py3 = f.panel(px3, y0, RW, PH, "卡再多，也发不满两份", BL)
    NS = (2, 4, 8, 1000)
    f.t(px3 + 20, py3 + 34, "卡数 n", GY, True, 13)
    f.t(px3 + 110, py3 + 34, "班长：n − 1", _RD, True, 13)
    f.t(px3 + 260, py3 + 34, "环：2(n−1)/n", GR, True, 13)
    for i, n in enumerate(NS):
        yy = py3 + 70 + i * 46
        f.t(px3 + 20, yy, "{:,}".format(n), INK, True, 15)
        f.t(px3 + 110, yy, "{:,}".format(star_send(n)), _RD, True, 15)
        f.t(px3 + 260, yy, ("%.3f" % ring_send(n)).rstrip("0").rstrip("."), GR, True, 15)
    f.t(px3 + 20, py3 + PH - 46, "公式里的 −1：自己那块不用寄给自己", GY, size=13)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "所以通信库几乎都选拼法二", [
        "两种拼法算出来的结果一模一样，差的只是谁在干活：拼法一压在卡 0 一个人身上，拼法二摊给所有人。",
        "而拼法二的两半，就是 ReduceScatter 和 AllGather —— 后面几刀白捡的便宜，全从这个拆法里来。",
    ])
    yb = f.src(yb + 10, "📌 两种拼法：NVIDIA NCCL 文档 Collective Operations（AllReduce ＝ Reduce ＋ Broadcast ＝ ReduceScatter ＋ AllGather）。",
               "⚠️ 本课推导：朴素班长做法卡 0 发 n−1 份；环上每卡发 2(n−1)/n 份。通信库实际会给班长做法排链或走树，这里只比最朴素的两种。")
    f.save("fig5-ar-two-ways.svg", yb + 14)


fig_1n()
fig_nn()
fig_split()
fig_ring()
fig_a2a()
fig_ar_two_ways()
