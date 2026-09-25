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
from topic03_draw import Fig, BL, OR, GR, PU, GY, INK, GY2, LINE

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
               "全交换：卡 k 的第 j 块发给卡 j，结束后卡 j 手里是四张卡各自的第 j 块，"
               "相当于把四行四列的表转置了一次。前三个是把收集、归约的结果发给所有人，"
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
    panel_pair(f, W - PW, y1 + 20, PW, "AllToAll　全交换", "卡 k 的第 j 块 → 发给卡 j",
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
# 图五：AllToAll 就是一次转置
# ════════════════════════════════════════════════════════════════
def fig_a2a():
    f = Fig(W, "全交换就是把一张四乘四的表转置一次。左边一行是一张卡手里的数据，"
               "第 j 格是它要交给卡 j 的那一份。做完之后，卡 j 手里的一行，"
               "是四张卡各自给它的那一份，也就是原来的第 j 列。"
               "对角线上那四格不用走网络，其余十二格都要跨卡。"
               "MoE 里这张表的一格，就是我手里要送去卡 j 上那个专家的 token")
    y0 = f.header("AllToAll　——　<tspan font-weight=\"700\">每人给每人寄一份不一样的</tspan>",
                  "左边：一行 ＝ 一张卡手里的数据，第 j 格是它要交给卡 j 的那一份。"
                  "右边：第 j 行 ＝ 原来的第 j 列",
                  [(BL, "卡 0 出的"), (OR, "卡 1 出的"), (GR, "卡 2 出的"), (PU, "卡 3 出的")])
    PH = 30 + 44 + N * RH + 60
    py = f.panel(0, y0, W, PH, "4 张卡的全交换", PU)
    LX, RX = 160, 820
    f.t(LX, py + 26, "之前：卡 k 的第 j 格要去卡 j", INK, True, 14)
    f.t(RX, py + 26, "之后：卡 j 收齐所有人给它的那一格", INK, True, 14)
    for j in range(N):
        f.t(LX + j * (CW + GAP) + CW / 2.0, py + 50, "寄%d" % j, GY, size=13, anchor="middle")
        f.t(RX + j * (CW + GAP) + CW / 2.0, py + 50, "从%d" % j, GY, size=13, anchor="middle")
    for k in range(N):
        yy = py + 58 + k * RH
        rowlab(f, LX - 64, yy, k)
        rowlab(f, RX - 64, yy, k)
        row(f, LX, yy, [([k], "%s%d" % (NAME[k], j)) for j in range(N)], hot={k})
        row(f, RX, yy, [([j], "%s%d" % (NAME[j], k)) for j in range(N)], hot={k})
    # ⭐ 2026-09-25 逐图审：「转置」对大众是术语。圈出一条具体的路：左边第 1 列（都寄给卡 1）→ 右边卡 1 那一行
    f.box(LX + (CW + GAP) - 5, py + 56, CW + 10, N * RH - 1, "none", INK, 6, sw=2, dash="5,3")
    f.t(LX + (CW + GAP) + CW / 2.0, py + 58 + N * RH + 12, "这一列都寄给卡 1", INK, True, 13, "middle")
    f.box(RX - 5, py + 58 + RH - 5, N * (CW + GAP) - GAP + 10, CH + 10, "none", INK, 6, sw=2, dash="5,3")
    f.t(RX + N * (CW + GAP) + 6, py + 58 + RH + CH / 2.0 + 5, "卡 1 收齐", INK, True, 13)
    mid = py + 58 + N * RH / 2.0 - 4
    f.line(LX + N * (CW + GAP) + 20, mid, RX - 90, mid, PU, 2)
    f.t((LX + N * (CW + GAP) + RX - 70) / 2.0, mid - 14, "按收件人重新分拣", PU, True, 15, "middle")
    f.t((LX + N * (CW + GAP) + RX - 70) / 2.0, mid + 24, "（整张表转置一次）", GY, size=13, anchor="middle")
    f.t(16, py + 58 + N * RH + 40,
        "粗框是对角线：自己给自己的那一格，不用走网络。其余 12 格都要跨卡　——　每张卡发出去整份的 (n−1)/n",
        GY, size=13.5)
    f._pan = None

    yb = f.band(py + PH + 20, "ok", "为什么 MoE 离不开它，又最怕它", [
        "MoE 里这张表的一格，就是<tspan font-weight=\"700\">我手里要送到卡 j 上那个专家的 token</tspan>。"
        "发过去算完，还要再转置一次送回来　——　所以专家并行每层两次 AllToAll。",
        "每一格多大由路由决定，<tspan font-weight=\"700\">事先不知道</tspan>，而且任意两张卡之间都有流量。"
        "前三种都能安排成只跟邻居说话；它的每一格都得从发的人一路走到收的人，所以对网络拓扑最挑剔。",
    ])
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
