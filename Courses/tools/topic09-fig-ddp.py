# -*- coding: utf-8 -*-
r"""专题九 · 案例二「DDP 和 FSDP 对不上」的五张图（第二次迁移 ＝ 混元 4 缩小版，TPU v5p 自比）。

fig9-ddp2x2   回答：查到最后，丢精度要哪两个条件同时成立？—— 高频词 × 不排序。
              格子里是第 1 步词嵌入梯度按行分组、离 FP64 的 rel_L2（实测，对齐记录）；每格的小圆点＝这一行收几次更新，
              红刻痕＝舍几次（示意：低频画 3 次、高频画 12 次；不排序舍 k−1 次、排序舍 1 次，这条是定义上必然的）。
fig9-ddprows  回答：第一个假设「BF16 跨卡求和」为什么被推翻？—— 误差跟本卡一次加几次走，不跟跨卡几路走。
              数据：同一张按行分组的表，四种跑法（FSDP、纯数据并行 8／16 卡、8 卡不累积）。
fig9-ddpflow  回答：同样把全部 token 的梯度加进词表，两种并行各加什么、传什么？—— 先加后传 vs 先传后加，带矩阵形状。
              形状读自两边训练步优化后的 HLO（实测）；每张卡跨卡的数据量按形状算（本课推导，线上实际字节数没测）。卡画 4 张是示意，实际 8 张。
fig9-rmw      回答：排序以后为什么只舍一次？—— 不排序每碰一次就「读—加—舍—写」，排序后在片上 FP32 缓冲里加完再舍。
              左：机理示意（同一个词出现 3 次，token 号随手写）；中：单算子四格（v5p 实测）；右：按「这一行在这次 scatter 里出现几次」分组（v5p 实测）。
fig9-sweep    回答：谁来排序、什么时候排？—— 后端在一次 scatter 的下标数超过表行数 1/8 时自己插排序。
              左：同一个 scatter 的下标数从 2048 扫到 32768（v5p 实测，四条线各一种写法）；右：换不同表形状二分出的门槛（CPU 上为 v5p 交叉编译，实测）。
fig9-shard    回答：第三种修法怎么做？—— 其余参数仍是纯数据并行，只把词嵌入表按隐藏维切开，编译器对它自动生成 FSDP 那条路；前向对称，不取整表。
              通信路径读自训练步优化后的 HLO（实测）；精度、步时是结果文件（实测）。卡画 4 张是示意，实际 8 张。
fig9-comm     回答：怎么从编译产物看通信？—— 一条集合通信读成「形状 × 字节宽 × 每步次数」；五种跑法的词嵌入通信与全部通信。
              字节按 HLO 里的形状算（不是线上实测），只适合互相比；看不到精度，也看不到快慢。
"""
import math

from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE, PU, wpx, MONO_K
import topic09_numbers as N

W = 1400
G = N.C2_GROUPS
HOT, LOW1, LOW2 = G[2], G[0], G[1]
LOW_ROWS = LOW1[1] + LOW2[1]
assert LOW_ROWS == 10872 and HOT[1] == 39


def _b(x):
    return ("%.2e" % x).replace("e+0", "e").replace("e+", "e")


def _e(x):
    return ("%.2e" % x).replace("e-0", "e-")


def _ticks(f, x, y, k, sort, col_dot, scale=1.0):
    """一行收 k 次更新：k 个圆点；红刻痕＝舍入次数（不排序 k−1 次，第一次加到 0 上不舍；排序 1 次）。"""
    dx = 22 * scale
    for i in range(k):
        f.box(x + i * dx, y, 12 * scale, 12 * scale, col_dot, col_dot, 6)
    nround = 1 if sort else k - 1
    if sort:
        f.box(x - 6, y - 7, (k - 1) * dx + 24 * scale, 26 * scale, "none", GR, 8, sw=1.4, dash="4 3")
        xr = x + (k - 1) * dx + 24 * scale
        f.line(xr, y - 6, xr, y + 18 * scale, RD, 2.6, arrow=False)
    else:
        for i in range(1, k):
            xr = x + i * dx - 5 * scale
            f.line(xr, y - 6, xr, y + 18 * scale, RD, 2.6, arrow=False)
    return nround


def fig_2x2():
    f = Fig(W, "两行两列四个格子。列：低频词的行（10872 行，合起来只占词嵌入梯度范数平方的一成）和高频词的行（39 行，占九成）。"
               "行：排序、只舍一次（FSDP 就是这样）和不排序、加一次舍一次（纯数据并行就是这样）。"
               "每格的圆点是这一行收到几次更新，红竖线是舍入几次。只有右下格，高频又不排序，舍了很多次，"
               "离 FP64 1.57e-2，是右上格 3.26e-3 的 4.8 倍；左边两格几乎一样，在 7.8e-3 到 9.4e-3。")
    y0 = f.header("查到最后：丢精度要两个条件同时成立　——　<tspan font-weight=\"700\">高频词 × 不排序</tspan>",
                  "第 1 步，词嵌入梯度按行分组，离 FP64 的距离（实测）；圆点 ＝ 这一行收到几次更新，红竖线 ＝ 舍了几次（示意）",
                  [(BL, "一次更新"), (RD, "一次舍成 BF16"), (GR, "片上 FP32 里一口气加完")])
    LX, CW, RH = 300, 520, 200
    top = y0 + 46
    f.t(LX + CW / 2, y0 + 20, "低频词的行（%s 行，占梯度范数平方的一成）" % format(LOW_ROWS, ","), INK, True, 15, "middle")
    f.t(LX + CW + 20 + CW / 2, y0 + 20, "高频词的行（%d 行，占九成）" % HOT[1], RD, True, 15, "middle")
    rows = [("排序，只舍一次", "FSDP 走这一行", True), ("不排序，加一次舍一次", "纯数据并行走这一行", False)]
    for r, (lab, who, sort) in enumerate(rows):
        y = top + r * (RH + 20)
        f.t(20, y + RH / 2 - 6, lab, INK, True, 16)
        f.t(20, y + RH / 2 + 18, who, GY, size=14)
        for c, k in enumerate((3, 12)):
            x = LX + c * (CW + 20)
            bad = (c == 1 and not sort)
            f.box(x, y, CW, RH, "#fce8e6" if bad else "#fff", RD if bad else LINE, 10, sw=1.6 if bad else 1)
            if sort and c == 1:
                _ticks(f, x + 30, y + 44, 24, True, BL, 0.6)   # 示意：FSDP 那次收齐 8 张卡，同一个词加得更多
            else:
                _ticks(f, x + 30, y + 40, k, sort, BL, 1.0)
            many = "几" if c == 0 else "几十到几百"
            if sort:
                txt = "收几次更新，舍 1 次" if c == 0 else "8 张卡收齐，次数约是纯数据并行那格的 8 倍，舍 1 次"
            else:
                txt = "收%s次更新，舍%s次" % (many, many)
            f.t(x + 30, y + 92, txt, GY, size=13.5)
            if c == 0:
                v = "%s–%s" % (_e(LOW2[3]), _e(LOW1[3])) if sort else "%s–%s" % (_e(LOW2[4]), _e(LOW1[4]))
            else:
                v = _e(HOT[3]) if sort else _e(HOT[4])
            f.t(x + 30, y + 150, "离 FP64  " + v, RD if bad else INK, True, 22 if bad else 18, big=True)
            if bad:
                f.t(x + CW - 24, y + 150, "是上一格的 %s 倍" % N.C2_HOT_X_T, RD, True, 15, "end")
            if c == 0 and not sort:
                f.t(x + CW - 24, y + 150, "不明显", GY, True, 15, "end")
    yb = f.band(top + 2 * RH + 40, "warn", "低频行两种写法差别小（最多 %s 倍）；高频行在 BF16 里一次次地加，才把误差攒起来" % N.C2_LOW_X_T, [
        "纯数据并行比 FSDP 多出来的误差，%s 落在这 %d 个高频词的行上。" % (N.C2_HOT_SHARE_T, HOT[1]),
        "代价：要让高频行只舍一次，要么排序（靠一个门槛），要么在 FP32 里加；后面几小节讲。",
    ])
    yb = f.src(yb + 10,
               "📌 实测（第二次迁移，TPU v5p 自比，第 1 步）：词嵌入梯度按行分组，每组算 ‖被测 − FP64‖ ÷ ‖FP64‖。分组按这一行的 token 在这一步全局批里一共出现几次（16 条序列，主输入与多 token 预测输入合计）："
               "只在 1 张卡上出现的 %d 行（FSDP %s，纯数据并行 %s）、在 ≥ 2 张卡上且合计 ≤ 300 次的 %d 行（%s ／ %s）、合计 ≥ 301 次的 %d 行（%s ／ %s）。"
               % (LOW1[1], _e(LOW1[3]), _e(LOW1[4]), LOW2[1], _e(LOW2[3]), _e(LOW2[4]), HOT[1], _e(HOT[3]), _e(HOT[4])),
               "📌 高频的 %d 个词，在一张卡、一个微批里加几次：每个词取出现最多的那张卡、那个微批，中位数 %d 次、最多 %d 次（主输入和多 token 预测输入两次 scatter 合计；只算主输入是 %d、%d 次）。"
               % (HOT[1], N.C2_HOT_CNT_MED, N.C2_HOT_CNT_MAX, N.C2_HOT_CNT_MAIN[0], N.C2_HOT_CNT_MAIN[1]),
               "⚠️ 圆点个数是示意（低频画 3 个、高频画 12 个）；「不排序舍 k − 1 次、排序舍 1 次」按两种写法的定义成立（第一次是加到 0 上，不舍）。「%s」按误差平方相减估算（本课按上表复算）。"
               % N.C2_HOT_SHARE_T,
               "⚠️ 低频行的误差在加进词表之前就有，推断来自上游的 BF16 计算，没有核实。")
    f.save("fig9-ddp2x2.svg", yb + 14)


def fig_rows():
    runs = [("FSDP（golden）", 3, GR, "32768 个下标、排序"),
            ("纯数据并行 8 卡", 4, RD, "每卡 4096 个"),
            ("纯数据并行 16 卡", 6, OR, "每卡 4096 个、跨卡 16 路"),
            ("8 卡、不做梯度累积", 5, PU, "每卡 8192 个")]
    f = Fig(W, "三组柱子，纵轴是离 FP64 的距离。左两组是低频行，四种跑法几乎一样高，在 7.8e-3 到 9.4e-3。"
               "右一组是 39 个高频词的行：FSDP 3.26e-3；纯数据并行 8 卡 1.57e-2、16 卡 1.59e-2，几乎一样；"
               "8 卡不做梯度累积、每卡一次加 8192 个，3.07e-2，接近翻倍。误差跟着本卡一次加几次走，不跟跨卡几路走。")
    y0 = f.header("第一个假设被推翻：误差跟本卡一次加几次走，不跟跨卡几路走",
                  "第 1 步，词嵌入梯度按行分三组，每组离 FP64 的距离（实测）；柱子越高越远",
                  [(c, lab) for lab, _, c, _ in runs])
    PH = 420
    py = f.panel(0, y0, W, PH, "按行分组，四种跑法（第二次迁移，TPU v5p）", BL)
    X0, YB, YH, vmax = 110, py + 330, 270, 3.2e-2

    def Y(v):
        return YB - v / vmax * YH
    f.line(X0 - 10, YB, W - 30, YB, GY2, 1.2, arrow=False)
    for v in (0, 1e-2, 2e-2, 3e-2):
        f.t(X0 - 16, Y(v) + 4, "0" if v == 0 else _e(v), GY, size=12, anchor="end")
        f.line(X0 - 10, Y(v), W - 30, Y(v), "#f1f3f4", 1, arrow=False)
    groups = [(LOW1, "只在 1 张卡上出现", "%s 行，占范数平方 %s" % (format(LOW1[1], ","), "1.3e-2")),
              (LOW2, "≥ 2 张卡、合计 ≤ 300 次", "%s 行，占 %s" % (format(LOW2[1], ","), "8.9e-2")),
              (HOT, "合计 ≥ 301 次：高频词", "%d 行，占 0.90" % HOT[1])]
    gx = [X0 + 40, X0 + 360, X0 + 720]
    BW = 62
    for (row, lab, sub), x in zip(groups, gx):
        for i, (rn, col_i, c, _) in enumerate(runs):
            v = row[col_i]
            bx = x + i * (BW + 10)
            f.box(bx, Y(v), BW, YB - Y(v), c, c, 3)
            if row is HOT:
                f.t(bx + BW / 2, Y(v) - 8, _e(v), c, True, 13.5, "middle")
        f.t(x + 2 * (BW + 10) - 5, YB + 26, lab, INK, True, 14.5, "middle")
        f.t(x + 2 * (BW + 10) - 5, YB + 48, sub, GY, size=13, anchor="middle")
    # 两条读法
    hx = gx[2]
    ax = hx + 4 * (BW + 10) + 30
    f.elbow(hx + 3 * (BW + 10) + BW, Y(HOT[5]) + 10, ax - 6, Y(HOT[5]) + 10, PU, 1.4, arrow=False)
    f.t(ax, Y(HOT[5]) + 6, "本卡一次加的次数翻倍", PU, True, 14.5)
    f.t(ax, Y(HOT[5]) + 28, "误差接近翻倍", PU, True, 14.5)
    f.elbow(hx + 2 * (BW + 10) + BW, Y(HOT[6]) + 10, ax - 6, Y(HOT[6]) + 60, OR, 1.4, arrow=False)
    f.t(ax, Y(HOT[6]) + 56, "卡数翻倍、跨卡多一倍路", OR, True, 14.5)
    f.t(ax, Y(HOT[6]) + 78, "误差不变", OR, True, 14.5)
    f.t(gx[0], Y(1.0e-2) - 14, "低频行：四种跑法几乎一样高", GY, True, 14.5)
    f._pan = None
    yb = f.band(py + PH + 20, "warn", "主因在每张卡自己那次 BF16 累加；8 路 BF16 跨卡求和只是次要的", [
        "后来拆开量过：高频行里跨卡那份只让误差大 %.2f 倍，本卡那份大 %.1f 倍；低频多卡那组跨卡约 %.2f 倍、本卡约 %.2f 倍。" % (N.C2_SPLIT_C[0], N.C2_SPLIT_C[1], N.C2_SPLIT_B[0], N.C2_SPLIT_B[1]),
        "留下的问题：FSDP 一次要加 8 张卡的全部 token，同一个词加的次数是 8 倍，误差反而最小。",
    ])
    yb = f.src(yb + 10,
               "📌 实测：同一张按行分组的表（第 1 步，离 FP64 的 rel_L2）。四列每张卡一次 scatter 的下标数：FSDP 32768（收齐 8 张卡的 token，排序）；纯数据并行 8 卡 4096；16 卡 4096（跨卡求和从 8 路变 16 路）；8 卡不做梯度累积 8192（一步的数据不拆微批）。",
               "📌 高频行：%s ／ %s ／ %s ／ %s；整张表：FSDP %s、纯数据并行 8 卡 %s、不累积 %s。各组 rel² 按范数平方加权求和回到整张表的数（本课复算，差 3%% 以内）。"
               % (_e(HOT[3]), _e(HOT[4]), _e(HOT[6]), _e(HOT[5]), N.C2_EMB_G_T, N.C2_EMB_DP_T, "2.92e-2"),
               "⚠️ 「接近翻倍」是这一步实测到的，不是一般规律：舍入误差有正有负、互不相关时，次数翻倍误差只涨约 1.4 倍；训练里接近线性，推断是舍入误差偏向一边（后加进去的小梯度被系统地吞掉一部分）。")
    f.save("fig9-ddprows.svg", yb + 14)


def fig_flow():
    f = Fig(W, "上下两条流水线，同一批数，加法组织完全不同。上：纯数据并行先加后传。每张卡对自己的 4096 个 token 做一次 scatter-add，"
               "加进一整张 120832 乘 1024 的 BF16 表，4096 个下标不过门槛、不排序；然后 8 张整表做 BF16 all-reduce。"
               "下：FSDP 先传后加。每张卡把自己的梯度按列切成 8 段、每段 4096 乘 128，送给管那 128 维的卡，token 号也收齐；"
               "每张卡收到 32768 行乘 128 列，32768 个下标过了门槛，排序以后在片上 FP32 里加完、只舍一次，加完就是最终梯度，不再跨卡求和。")
    y0 = f.header("同一批数，两种加法　——　<tspan font-weight=\"700\">纯数据并行先加后传，FSDP 先传后加</tspan>",
                  "每张卡每个微批 1 条序列、%d 个 token；词表 E ＝ [%s, %d]；反向传回来 dX ＝ [%d, %d] BF16；卡画 4 张是示意，实际 8 张" %
                  (N.C2_TOK, format(N.C2_V, ","), N.C2_D, N.C2_TOK, N.C2_D),
                  [(BL, "本卡的梯度"), (RD, "不排序：加一次舍一次"), (GR, "排序：只舍一次"), (OR, "跨卡")])
    CARD = [BL, "#4285f4", "#669df6", "#8ab4f8"]
    # ── 上：纯数据并行 ──
    PH = 320
    py = f.panel(0, y0, W, PH, "纯数据并行：每张卡存整张表 —— 先加后传", RD)
    for k in range(4):
        y = py + 22 + k * 62
        f.box(24, y, 120, 44, "#e8f0fe", BL, 5)
        f.t(84, y + 20, "卡 %d：dX" % k, BL, True, 13.5, "middle")
        f.t(84, y + 37, "[4096, 1024]", GY, size=12, anchor="middle", mono=True)
        f.line(146, y + 22, 300, y + 22, RD, 1.8)
        f.t(222, y + 15, "scatter-add", RD, True, 12.5, "middle")
        f.box(304, y, 260, 44, "#fff", RD, 5, sw=1.4)
        f.t(434, y + 20, "dE%s：整张表 BF16" % "₀₁₂₃"[k], INK, True, 13.5, "middle")
        f.t(434, y + 37, "[120832, 1024]", GY, size=12, anchor="middle", mono=True)
        f.line(566, y + 22, 640, py + 140, OR, 1.4, arrow=(k == 3))
    f.t(320, py + 272, "4096 个下标 ＜ 门槛 15104（行数 ÷ 8）→ 不排序，每碰一次就舍一次", RD, True, 14)
    f.box(644, py + 104, 300, 72, "#fef7e0", OR, 8, sw=1.4)
    f.t(794, py + 134, "BF16 all-reduce，整张表", OR, True, 15, "middle")
    f.t(794, py + 158, "每卡每次 ≈ %s 字节" % ("%.2e" % N.C2_AR_BYTES).replace("e+0", "e"), GY, size=13, anchor="middle")
    f.line(946, py + 140, 1030, py + 140, OR, 1.8)
    f.box(1034, py + 112, 340, 56, "#fff", INK, 8, sw=1.4)
    f.t(1204, py + 136, "dE ＝ dE₀ ＋ … ＋ dE₇", INK, True, 15, "middle")
    f.t(1204, py + 157, "8 堆各自加完，再把 8 个结果相加", GY, size=13, anchor="middle")
    f._pan = None
    # ── 下：FSDP ──
    y1 = py + PH + 18
    py2 = f.panel(0, y1, W, PH + 30, "FSDP（词表按隐藏维切，每张卡管 128 维）：先传后加", GR)
    for k in range(4):
        y = py2 + 22 + k * 62
        f.t(24, y + 20, "卡 %d：dX" % k, BL, True, 13.5)
        for j in range(4):
            f.box(100 + j * 34, y + 4, 30, 36, CARD[j], CARD[j], 3)
        f.t(240, y + 28, "按列切成卡数段", GY, size=12.5)
    f.box(330, py2 + 70, 190, 100, "#fef7e0", OR, 8, sw=1.4)
    f.t(425, py2 + 100, "all-to-all", OR, True, 15, "middle")
    f.t(425, py2 + 122, "每段 [4096, 128]", GY, size=12.5, anchor="middle", mono=True)
    f.t(425, py2 + 142, "token 号 all-gather", GY, size=12.5, anchor="middle")
    f.t(425, py2 + 160, "每卡 ≈ %s 字节" % ("%.1e" % N.C2_A2A_BYTES).replace("e+0", "e"), GY, size=12.5, anchor="middle")
    f.line(270, py2 + 120, 326, py2 + 120, OR, 1.8)
    f.line(522, py2 + 120, 580, py2 + 120, OR, 1.8)
    # 卡 0 收到 8 段（画 4 段，全是同一种颜色＝同一列段，来自不同卡）
    f.t(584, py2 + 30, "卡 0 收齐：8 张卡在它那 128 维上的梯度", INK, True, 13.5)
    for j in range(4):
        f.box(590 + j * 40, py2 + 46, 34, 150, CARD[0], CARD[0], 3)
    f.t(668, py2 + 218, "[32768, 128]：每张卡的第 0 段", GY, size=12.5, anchor="middle", mono=True)
    f.line(760, py2 + 120, 810, py2 + 120, GR, 1.8)
    f.box(814, py2 + 82, 120, 76, "#e6f4ea", GR, 8, sw=1.4)
    f.t(874, py2 + 114, "先排序", GR, True, 15, "middle")
    f.t(874, py2 + 136, "32768 个下标", GY, size=12.5, anchor="middle")
    f.line(936, py2 + 120, 990, py2 + 120, GR, 1.8)
    f.box(994, py2 + 76, 380, 88, "#fff", GR, 8, sw=1.6)
    f.t(1184, py2 + 104, "scatter-add → dE[:, 128k : 128k＋128]", GR, True, 14.5, "middle")
    f.t(1184, py2 + 126, "片上 FP32 里加完，只舍一次", GR, True, 13.5, "middle")
    f.t(1184, py2 + 148, "加完就是最终梯度，不再跨卡求和", GY, size=13, anchor="middle")
    f.t(584, py2 + 290, "32768 个下标 ＞ 门槛 15104 → 后端自己插排序、标「已排序」，同一个词的更新挨在一起", GR, True, 14)
    f._pan = None
    yb = f.band(py2 + PH + 30 + 20, "ok", "「先传后加」让每次 scatter 的下标数翻了 8 倍，正好跨过门槛", [
        "FSDP 一次加的更多，反而走上了只舍一次的写法；这是编译器自动切分的结果，跟 PyTorch FSDP「各卡先算整表梯度、再 reduce-scatter」不是一回事。",
        "顺带：按形状算，反向这一次跨卡的数据量 FSDP 约是纯数据并行的 1/%d（本课推导）；每步连前向那次一起算约 1/15。" % round(N.C2_BYTES_X),
    ])
    yb = f.src(yb + 10,
               "📌 实测（两边训练步优化后的 HLO，每个微批主输入、多 token 预测输入各一组）：纯数据并行 —— scatter 输出 bf16[120832,1024]、没有排序，之后 all-reduce bf16[120832,1024]、BF16 加法；"
               "FSDP —— token 号 all-gather s32[8,4096,1]，梯度 all-to-all bf16[1,4096,1024] → bf16[1,4096,8,128]，先 sort 32768 个下标，scatter 输出 bf16[120832,128]、标「下标已排序」，结果直接加进 FP32 累加器，没有跨卡求和。",
               "📌 本课推导：整张表 120832 × 1024 × 2 字节 ≈ 2.47e8；all-to-all 的结果 32768 × 128 × 2 字节 ≈ 8.4e6（token 号另有约 1.3e5）。",
               "📌 矩阵乘类权重两边一样：每张卡先算本卡的部分和，再跨卡加；纯数据并行用 all-reduce、FSDP 用 reduce-scatter，all-reduce ＝ reduce-scatter ＋ all-gather，后者只搬不加，所以舍入次数相同，只差先后。")
    f.save("fig9-ddpflow.svg", yb + 14)


def fig_rmw():
    f = Fig(W, "三块。左块：同一个词在一次 scatter 里出现 3 次。不排序时它的 3 次更新散在整串下标中间，每碰到一次，就把那一行 BF16 读出来、转 FP32 加上、舍回 BF16、写回；"
               "排序以后 3 次挨在一起，在片上的 FP32 缓冲里一口气加完，写出时只舍一次。"
               "中块：单算子四格，只有排好序、带「已排序」标记、且 excess precision 开着的那一格是 1.48e-3，其余三格都是 6.75e-3，逐位等于每次舍。"
               "右块：按这一行在这次 scatter 里出现几次分组：不排序随次数一路涨到 1.46e-2，排序一直停在 1.7e-3 上下。")
    y0 = f.header("排序以后，为什么只舍一次",
                  "左：后端的两种写法（示意，同一个词出现 3 次）；中：同一个 scatter 换四种条件（实测）；右：按这一行出现几次分组（实测）",
                  [(RD, "舍成 BF16"), (GR, "片上 FP32 缓冲"), (BL, "一次更新")])
    PH = 470
    LW = 600
    py = f.panel(0, y0, LW, PH, "两种写法", BL)
    # 不排序
    f.t(20, py + 30, "不排序：下标照原来的顺序", RD, True, 14.5)
    seq = ["…", "17", "…", "902", "…", "17", "…", "5", "…", "17", "…"]
    for i, s in enumerate(seq):
        c = BL if s == "17" else GY
        f.t(30 + i * 50, py + 60, s, c, s == "17", 15, "middle", mono=True)
    for j in range(3):
        y = py + 86 + j * 40
        f.t(20, y + 18, "第 %d 次" % (j + 1), GY, size=13)
        steps = [("读 BF16", GY), ("转 FP32 加", BL), ("舍回 BF16" if j > 0 else "加到 0，不舍", RD if j > 0 else GY2), ("写回", GY)]
        for k, (s, c) in enumerate(steps):
            f.box(90 + k * 122, y, 112, 28, "#fce8e6" if (c == RD) else "#fff", c, 5, sw=1.4 if c == RD else 1)
            f.t(146 + k * 122, y + 19, s, c, c == RD, 12.5, "middle")
    f.t(20, py + 222, "第一次是加到 0 上，不舍；一行出现 k 次，舍 k − 1 次", RD, size=13.5)
    # 排序
    f.t(20, py + 268, "排序后：同一个词的更新挨在一起", GR, True, 14.5)
    seq2 = ["…", "5", "…", "17", "17", "17", "…", "902", "…"]
    for i, s in enumerate(seq2):
        c = BL if s == "17" else GY
        f.t(30 + i * 56, py + 298, s, c, s == "17", 15, "middle", mono=True)
    f.box(90, py + 318, 300, 44, "#e6f4ea", GR, 8, sw=1.6)
    f.t(240, py + 345, "片上 FP32 缓冲：g₁ ＋ g₂ ＋ g₃", GR, True, 14, "middle")
    f.line(392, py + 340, 440, py + 340, GR, 1.6)
    f.box(444, py + 322, 130, 36, "#fce8e6", RD, 5, sw=1.4)
    f.t(509, py + 345, "舍一次，写出", RD, True, 13, "middle")
    f.t(20, py + 400, "不管出现几次，都只舍 1 次", GR, size=13.5)
    f.t(20, py + 428, "前提：excess precision 开着（默认开）", GY, size=13.5)
    f._pan = None
    # 中：四格
    MX = LW + 20
    MW = 330
    py2 = f.panel(MX, y0, MW, PH, "四格：4096 个下标", OR)
    f.t(MX + 175, py2 + 30, "excess precision", GY, True, 13, "middle")
    f.t(MX + 130, py2 + 52, "开", INK, True, 14, "middle")
    f.t(MX + 250, py2 + 52, "关", INK, True, 14, "middle")
    cells = [("不排序", N.C2_CELL4["不排序·开"], N.C2_CELL4["不排序·关"]),
             ("排序＋标记", N.C2_CELL4["排序·开"], N.C2_CELL4["排序·关"])]
    for r, (lab, on, off) in enumerate(cells):
        y = py2 + 66 + r * 120
        f.t(MX + 10, y + 46, lab.replace("＋标记", ""), INK, True, 13.5)
        if "标记" in lab:
            f.t(MX + 10, y + 66, "＋标记", INK, True, 13.5)
        for c, v in enumerate((on, off)):
            good = v < 3e-3
            x = MX + 80 + c * 120
            f.box(x, y, 104, 104, "#e6f4ea" if good else "#fce8e6", GR if good else RD, 8, sw=1.6 if good else 1)
            f.t(x + 52, y + 50, _e(v), GR if good else RD, True, 15, "middle")
            f.t(x + 52, y + 74, "只舍一次" if good else "每次舍", GR if good else RD, size=13, anchor="middle")
    f.t(MX + 14, py2 + 330, "只排好序、不带标记：", GY, True, 13)
    f.t(MX + 14, py2 + 352, "仍是 %s，每次舍" % _e(N.C2_CELL4["只排不标·开"]), RD, size=13)

    f._pan = None
    # 右：按次数分组
    RX = MX + MW + 20
    py3 = f.panel(RX, y0, W - RX, PH, "按这一行出现几次分组", PU)
    buckets = ["1", "2–9", "10–99", "≥ 100"]
    rows_n = [N.C2_BYCOUNT_ROWS[k] for k in ("1", "2-9", "10-99", "100+")]
    X0, YB, YH, vmax = RX + 64, py3 + 330, 250, 1.6e-2
    f.line(X0 - 6, YB, W - 20, YB, GY2, 1.2, arrow=False)
    for v in (0, 5e-3, 1e-2, 1.5e-2):
        f.t(X0 - 10, YB - v / vmax * YH + 4, "0" if v == 0 else _e(v), GY, size=11.5, anchor="end")
    for i, b in enumerate(buckets):
        x = X0 + 8 + i * 84
        for j, (key, c) in enumerate((("bf16_unsorted_excess_on", RD), ("bf16_presorted_excess_on", GR))):
            v = N.C2_BYCOUNT[key][i]
            if v > 0:
                h = max(v / vmax * YH, 1.5)
                f.box(x + j * 36, YB - h, 32, h, c, c, 3)
        f.t(x + 34, YB + 22, b + " 次", INK, True, 13, "middle")
        f.t(x + 34, YB + 42, "%d 行" % rows_n[i], GY, size=12, anchor="middle")
    f.t(X0 + 8 + 3 * 84 + 16, YB - N.C2_BYCOUNT["bf16_unsorted_excess_on"][3] / vmax * YH - 8,
        _e(N.C2_BYCOUNT["bf16_unsorted_excess_on"][3]), RD, True, 13, "middle")
    f.t(X0, py3 + 46, "红柱不排序，绿柱排序", INK, True, 13.5)
    f.t(RX + 16, py3 + 420, "出现 1 次 ＝ 加到 0 上，两边都是 0", GY, size=12.5)
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "一行出现 k 次：不排序舍 k − 1 次，排序只舍 1 次 —— k 上百时差约 %.1f 倍" % N.C2_K100_X, [
        "值在两次更新之间住在 BF16 的表里，所以不排序那种写法 excess precision 也省不掉这次舍入；排序那种只关掉 excess precision，又变回每次舍。",
        "代价：这一层在编译后的 HLO 里看不见 —— 两份 HLO 一字不差，后端生成的指令可以完全不同。",
    ])
    yb = f.src(yb + 10,
               "📌 实测（单算子，v5p，镜像与训练相同）：表 bf16[120832,1024]，下标取 golden 第 1 步的真实 token（前 4096 个），更新量是随机 BF16。四格的结果和 CPU 上两种仿真逐位比：「FP32 里累加、最后舍一次」和「每加一次舍一次 BF16」；排序＋开那一格 100% 的行等于前者，其余三格等于后者。",
               "📌 右块（同一次实测，4096 个下标）：出现 1 ／ 2–9 ／ 10–99 ／ ≥ 100 次的行数 %d ／ %d ／ %d ／ %d；不排序 0 ／ %s ／ %s ／ %s，排序 0 ／ %s ／ %s ／ %s。"
               % tuple(rows_n + [_e(x) for x in N.C2_BYCOUNT["bf16_unsorted_excess_on"][1:]] + [_e(x) for x in N.C2_BYCOUNT["bf16_presorted_excess_on"][1:]]),
               "📌 后端指令（CPU 上为 v5p 交叉编译，打开后端 dump 数静态条数）：不排序那颗 kernel 每个更新「读 BF16 → 转 FP32 → 加 → 舍回 BF16 → 写回」；排序那颗换成另一颗 kernel，累加器在片上 VMEM 的 FP32 缓冲里；同一份 HLO 只关掉 excess precision，这颗 kernel 多出 %s 条舍到 BF16 的指令，夹在每次加法前后。条数是静态的，不是执行次数；「舍到 BF16」是由指令名和结果逐位等于仿真推断的。"
               % format(N.C2_VPACK, ","),
               "⚠️ 更新量是随机数，这张图只比同一行的两列，不跟训练里的数直接对。左块 token 号随手写。")
    f.save("fig9-rmw.svg", yb + 14)


def fig_sweep():
    f = Fig(W, "两块。左块横轴是一次 scatter 的下标数，从 2048 到 32768，对数刻度；纵轴离 FP64 的距离。"
               "红线是训练里那种不带标记的写法：2048、4096、8192 时 4.6e-3、6.8e-3、9.1e-3，一路涨；到 16384 突然掉到 1.55e-3，因为编译器自己插了排序。"
               "绿线是排好序带标记的写法，一直在 1.4e-3 到 1.6e-3。灰线是关掉 excess precision，一路涨到 1.76e-2。竖线是门槛 15104，等于词表行数的八分之一。"
               "纯数据并行每卡 4096 和不累积时 8192 都在门槛左边，FSDP 每卡 32768 在右边。"
               "右块换不同的表形状二分门槛，插排序的点都落在「下标数 ＝ 行数 ÷ 8」那条线上；表的总元素太少时不插。")
    y0 = f.header("谁来排序：一次 scatter 的下标数超过表行数的 1/8，后端自己插",
                  "左：同一个 scatter，下标数从 2048 扫到 32768（单算子，实测，横轴每格翻倍）；右：换表形状二分出的门槛（交叉编译，实测）",
                  [(RD, "不带标记（训练里的写法）"), (GR, "排好序、带标记"), (GY2, "关掉 excess precision")])
    PH = 430
    LW = 860
    py = f.panel(0, y0, LW, PH, "下标数扫过门槛（词表 120832 行）", RD)
    X0, XW, YB, YH, vmax = 90, 720, py + 330, 270, 1.8e-2

    def X(n):
        return X0 + (math.log2(n) - 11) / 4.0 * XW

    def Y(v):
        return YB - v / vmax * YH
    f.line(X0 - 6, YB, X0 + XW + 20, YB, GY2, 1.2, arrow=False)
    for v in (0, 5e-3, 1e-2, 1.5e-2):
        f.t(X0 - 10, Y(v) + 4, "0" if v == 0 else _e(v), GY, size=12, anchor="end")
    for n in N.C2_SWEEP_N:
        f.t(X(n), YB + 22, format(n, ","), GY, size=12.5, anchor="middle")
    xg = X(N.C2_GATE)
    f.line(xg, py + 40, xg, YB, INK, 1.4, dash="6 4", arrow=False)
    f.t(xg - 8, py + 110, "门槛 %s ＝ 行数 ÷ 8" % format(N.C2_GATE, ","), INK, True, 14, "end")
    for key, c, sw in (("bf16_unsorted_excess_off", GY2, 2.0), ("bf16_presorted_excess_on", GR, 2.4), ("bf16_unsorted_excess_on", RD, 3.0)):
        pts = [(X(n), Y(v)) for n, v in zip(N.C2_SWEEP_N, N.C2_SWEEP[key])]
        f.path(pts, c, sw, arrow=False)
        for x, y in pts:
            f.box(x - 4, y - 4, 8, 8, c, c, 4)
    u = N.C2_SWEEP["bf16_unsorted_excess_on"]
    f.t(X(8192) - 6, Y(u[2]) - 12, _e(u[2]), RD, True, 13.5, "end")
    f.t(X(16384) + 8, Y(u[3]) - 30, "编译器插了排序 → %s" % _e(u[3]), RD, True, 13.5)
    f.t(X(32768) + 4, Y(N.C2_SWEEP["bf16_unsorted_excess_off"][4]) - 12, _e(N.C2_SWEEP["bf16_unsorted_excess_off"][4]), GY, True, 13, "end")
    # 训练里的三种跑法落在哪
    for n, lab in ((4096, "纯数据并行每卡"), (8192, "不累积"), (32768, "FSDP 每卡")):
        f.t(X(n), YB + 46, lab, INK, True, 13, "middle")
    f._pan = None
    RX = LW + 20
    py2 = f.panel(RX, y0, W - RX, PH, "换表形状二分门槛", OR)
    TH = N.C2_THRESH
    X1, XW1, YB1, YH1 = RX + 80, 400, py2 + 330, 270
    vmx, nmx = 500000, 65536

    def XV(v):
        return X1 + v / vmx * XW1

    def YN(n):
        return YB1 - n / nmx * YH1
    f.line(X1 - 6, YB1, X1 + XW1 + 10, YB1, GY2, 1.2, arrow=False)
    f.line(X1, YB1, X1, YB1 - YH1 - 6, GY2, 1.2, arrow=False)
    f.path([(XV(0), YN(0)), (XV(vmx), YN(vmx / 8.0))], OR, 1.6, dash="6 4", arrow=False)
    f.t(XV(330000), YN(330000 / 8.0) - 30, "下标数 ＝ 行数 ÷ 8", OR, True, 13.5, "middle")
    for v in (0, 120832, 480000):
        f.t(XV(v), YB1 + 20, format(v, ","), GY, size=12, anchor="middle")
    for n in (0, 30000, 60000):
        f.t(X1 - 8, YN(n) + 4, format(n, ","), GY, size=12, anchor="end")
    f.t(X1 + XW1 / 2, YB1 + 44, "表的行数", GY, True, 13, "middle")
    for v, d, n in TH:
        if n is None:
            f.box(XV(v) - 6, YN(nmx) - 6, 12, 12, "#fff", GY2, 6, sw=1.4)
        else:
            f.box(XV(v) - 5, YN(n) - 5, 10, 10, OR, OR, 5)
    f.t(X1 + 50, YN(nmx) + 4, "空心：到 65536 都不插（表太小）", GY, size=12.5)
    f.t(X1 + 50, YN(nmx) + 28, "实心：最小插排序的下标数", OR, True, 13)
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "门槛只认下标数，不认并行方式：FSDP 一次加得多，恰好跨过了线", [
        "纯数据并行每卡 4096、不累积 8192 都在线左边，不排序；反过来，纯数据并行每卡一次的 token 要是超过 15104，按这条规则也会被排序（推断，训练里没跑过）。",
        "代价：门槛是在一个 TPU 运行库版本上二分出来的，换版本可能变；表很小时也不插。",
    ])
    yb = f.src(yb + 10,
               "📌 左块实测（单算子，v5p，表 bf16[120832,1024]，下标取 golden 第 1 步真实 token 的前 N 个，更新量随机 BF16）：不带标记 %s；排好序带标记 %s；关 excess precision %s；FP32 累加 1.6e-9 到 1.1e-8。每格都逐位等于对应的 CPU 仿真。"
               % tuple(" ／ ".join(_e(x) for x in N.C2_SWEEP[k]) for k in ("bf16_unsorted_excess_on", "bf16_presorted_excess_on", "bf16_unsorted_excess_off")),
               "📌 右块实测（CPU 上用同一个 TPU 运行库为 v5p 交叉编译，看优化后 HLO 里有没有 sort）：" + "；".join(
                   "%s × %d %s" % (format(v, ","), d, ("最小 %s" % format(n, ",")) if n else "到 65536 都不插") for v, d, n in N.C2_THRESH)
               + "。插排序的，最小下标数都等于 行数 ÷ 8 ＋ 1，门槛位置和列数无关；表的总元素太少时不插（30000 行：128 列不插、1024 列插）。同一个运行库为 v7x、v6e 编译，120832 行的门槛也是 15104／15105。",
               "📌 插排序的是 TPU 后端（libtpu）里的一道工序 %s：对（下标, 序号）稳定排序 → 按排序后的次序取更新量 → scatter 标上「下标已排序」。公开的 XLA 代码里按名字搜不到，触发条件只能从实验反推。" % N.C2_PASS,
               "⚠️ 门槛为什么定在这里（推断）：看上去是为速度定的，精度更好是附带效果；单算子耗时只部分支持（快慢交叉在 4096 到 8192 之间，和 15104 对不上）。")
    f.save("fig9-sweep.svg", yb + 14)


def fig_shard():
    f = Fig(W, "上下两块。上：前向。其余参数每张卡一整份，只有词嵌入表按隐藏维切成 8 段，每张卡存 120832 行乘 128 列。"
               "先把 8 张卡的 token 号收齐，每张卡从自己那 128 列里取出 32768 个 token 的行，再用 all-to-all 把整行换回每张卡自己那 4096 个 token，不取整张表。"
               "下：每张卡存什么。不修时每卡一整张表加主权重和两份动量；只切表后每卡只存八分之一。实测显存峰值每设备 6.16e9 降到 4.46e9 字节，参数与优化器状态 4.41e9 降到 3.12e9。反向那条路跟 FSDP 一样，见上一张图。"
               "结果：第 1 步词嵌入梯度离 FP64 3.97e-3，R 0.95；300 步 r 1.02、1000 步 1.15；词嵌入通信少 93%；步时跟不修一样快。")
    y0 = f.header("第三种修法：只把词嵌入表沿数据并行轴切开　——　<tspan font-weight=\"700\">编译器自己生成 FSDP 那条路</tspan>",
                  "其余参数仍是纯数据并行（每卡一整份）；只有词嵌入表 [%s, %d] 按隐藏维切 8 段，每卡 [%s, %d]；卡画 4 张是示意，实际 8 张" %
                  (format(N.C2_V, ","), N.C2_D, format(N.C2_V, ","), N.C2_D_SHARD),
                  [(BL, "本卡的数据"), (OR, "跨卡"), (GR, "排序：只舍一次")])
    CARD = [BL, "#4285f4", "#669df6", "#8ab4f8"]
    PH = 250
    py = f.panel(0, y0, W, PH, "前向：不取整表，只换取到的行", BL)
    f.t(24, py + 30, "每卡存的那一段表", INK, True, 13.5)
    for k in range(4):
        f.box(24 + k * 40, py + 44, 32, 150, CARD[k], CARD[k], 3)
        f.t(40 + k * 40, py + 212, "卡%d" % k, GY, size=12, anchor="middle")
    f.line(190, py + 110, 250, py + 110, OR, 1.8)
    f.box(254, py + 74, 200, 72, "#fef7e0", OR, 8, sw=1.4)
    f.t(354, py + 104, "token 号收齐", OR, True, 15, "middle")
    f.t(354, py + 126, "8 卡 × 4096 ＝ 32768 个", GY, size=12.5, anchor="middle")
    f.line(456, py + 110, 510, py + 110, BL, 1.8)
    f.box(514, py + 74, 300, 72, "#e8f0fe", BL, 8, sw=1.4)
    f.t(664, py + 104, "每卡从自己那 128 列取行", BL, True, 14.5, "middle")
    f.t(664, py + 126, "[32768, 128]", GY, size=12.5, anchor="middle", mono=True)
    f.line(816, py + 110, 870, py + 110, OR, 1.8)
    f.box(874, py + 74, 220, 72, "#fef7e0", OR, 8, sw=1.4)
    f.t(984, py + 104, "all-to-all 换回整行", OR, True, 14.5, "middle")
    f.t(984, py + 126, "本卡 4096 个 token", GY, size=12.5, anchor="middle")
    f.line(1096, py + 110, 1150, py + 110, BL, 1.8)
    f.box(1154, py + 74, 220, 72, "#fff", INK, 8, sw=1.4)
    f.t(1264, py + 104, "X [4096, 1024]", INK, True, 15, "middle", mono=True)
    f.t(1264, py + 126, "跟原来一样往下算", GY, size=12.5, anchor="middle")
    f._pan = None
    y1 = py + PH + 16
    py2 = f.panel(0, y1, W, PH, "每张卡存什么、省了什么（反向那条路跟 FSDP 一样，见上一张图下半）", GR)
    f.t(24, py2 + 30, "不修：每卡一整张表 ＋ 主权重 ＋ 两份动量", INK, True, 13.5)
    for k in range(4):
        f.box(24 + k * 70, py2 + 46, 60, 150, CARD[k], CARD[k], 3)
        f.t(54 + k * 70, py2 + 214, "卡%d" % k, GY, size=12, anchor="middle")
    f.t(330, py2 + 30, "只切表：每卡只存 1/8", GR, True, 13.5)
    for k in range(4):
        f.box(330 + k * 70, py2 + 46, 60, 150, "#fff", LINE, 3, dash="4 3")
        f.box(330 + k * 70, py2 + 46, 60, 150 / 8.0, CARD[k], CARD[k], 3)
        f.t(360 + k * 70, py2 + 214, "卡%d" % k, GY, size=12, anchor="middle")
    BX, BW = 720, 480
    for i, (lab, before, after) in enumerate((("实测显存峰值，每设备", N.C3_HBM[0], N.C3_HBM[1]),
                                              ("参数与优化器状态，每设备", N.C3_STATE[0], N.C3_STATE[1]))):
        y = py2 + 52 + i * 92
        f.t(BX, y, lab, INK, True, 13.5)
        f.box(BX, y + 10, BW * before / N.C3_HBM[0], 18, GY2, GY2, 3)
        f.t(BX + BW * before / N.C3_HBM[0] + 8, y + 24, "不修 %s" % _b(before), GY, size=12.5)
        f.box(BX, y + 34, BW * after / N.C3_HBM[0], 18, GR, GR, 3)
        f.t(BX + BW * after / N.C3_HBM[0] + 8, y + 48, "只切表 %s" % _b(after), GR, True, 12.5)
    f._pan = None
    yb = f.band(py2 + PH + 20, "ok", "精度跟前两种修法一样，通信反而少了：词嵌入那份每步少 %s，步时跟不修一样快" % N.C3_EMB_CUT_T, [
        "第 1 步词嵌入梯度离 FP64 %s（R %s，换了顺序）；300 步 r %s、1000 步 %s，都在 GPU 重跑的纯噪声分数以内。" % (N.C3_EMB_T, N.C3_EMB_R_T, N.C3_R300_T, N.C3_R1K_T),
        "代价：它靠每卡那次加法过排序门槛（这里 32768 是门槛的两倍多）；8 张卡合起来也不过线时会退回不排序（推断，没跑过）。",
    ])
    yb = f.src(yb + 10,
               "📌 实测（第二次迁移，TPU v5p 8 设备，纯数据并行 8 ＋ 梯度累积 2，只多一个开关：词嵌入表的隐藏维换一个逻辑轴，沿数据并行轴切）：通信路径读自训练步优化后的 HLO —— 前向 token 号 all-gather s32[8,4096,1]、取到的行 all-to-all bf16[8,1,4096,128]；反向 token 号 all-gather、梯度 all-to-all bf16[1,4096,8,128]，先 sort 32768 个下标，scatter 输出 bf16[120832,128]、标「下标已排序」。跟 FSDP 8 的词嵌入通信一条不差。",
               "📌 第 1 步：205 个梯度张量里 187 个换了顺序、11 个在 GPU 尺子以内、7 个是多路残差混合小参数（照例单列），没有丢精度的；梯度范数（裁剪前）6.35963，FP64 6.35944。开关关着时编出的程序去掉元数据后与原来逐字相同，重跑逐位相同。",
               "📌 步时（第 11 步起每步中位数）：只切表 %.3f s，不修 %.3f s，golden（FSDP 8）%.3f s；显存峰值每设备 %s → %s 字节，参数与优化器状态 %s → %s（表的主权重和两份动量只存 1/8）。" % (N.C3_STEP_S, N.C2_DP_STEP_S, N.C2_G_STEP_S, _b(N.C3_HBM[0]), _b(N.C3_HBM[1]), _b(N.C3_STATE[0]), _b(N.C3_STATE[1])),
               "⚠️ 通信少了、步时却没快，推断是原来那份整表 all-reduce 本来就和计算重叠、不在关键路径上；没有看 profile 核实。只在 v5p、缩小版、纯数据并行 8 ＋ 累积 2 上验证过；v7x、生产尺寸、数据并行 × FSDP 混合没跑。")
    f.save("fig9-shard.svg", yb + 14)


def fig_comm():
    f = Fig(W, "两块。左块：一条集合通信怎么读成字节。编译后的程序里一行 all-reduce，结果形状 bf16[120832,1024]，元素数乘 BF16 的 2 字节是 2.47e8；"
               "它在梯度累积循环里，循环次数从条件计算的「计数器小于 2」读出来；主输入和多 token 预测输入各一条，所以每步 4 次，合计 9.90e8 字节。"
               "右块：五种跑法每步每设备的通信字节，对数刻度，橙色是词嵌入那份。词嵌入那份：不修 9.90e8，修法① 1.06e9，修法② 9.90e8，FSDP 和只切词嵌入表都是 6.82e7。"
               "全部集合通信：不修 2.70e9，修法① 2.77e9，修法② 2.70e9，FSDP 1.32e10，只切词嵌入表 1.78e9。")
    y0 = f.header("怎么从编译产物看通信：形状 × 字节宽 × 每步次数",
                  "左：一条集合通信读成字节（本课按编译后程序里的原文示意）；右：五种跑法每步每设备的通信字节（按形状算，对数刻度）",
                  [(OR, "词嵌入的通信"), (GY2, "全部集合通信")])
    PH = 486
    LW = 660
    py = f.panel(0, y0, LW, PH, "一条通信，四步读成字节", BL)
    f.box(16, py + 22, LW - 32, 40, "#f8f9fa", LINE, 6)
    CODE = "%all-reduce = bf16[120832,1024] all-reduce(%x), to_apply=%add"
    f.t(26, py + 47, CODE, INK, size=13, mono=True)

    def _cx(sub, k=0):                    # 代码行里第 k 次出现的 sub 的起止 x（等宽字估宽）
        i = -1
        for _ in range(k + 1):
            i = CODE.index(sub, i + 1)
        x0 = 26 + wpx(CODE[:i], 13) * MONO_K
        return x0, x0 + wpx(sub, 13) * MONO_K
    for n_, (sub, k, col) in enumerate([("all-reduce", 1, BL), ("[120832,1024]", 0, BL), ("bf16", 0, BL)]):
        a, b = _cx(sub, k)
        f.line(a, py + 56, b, py + 56, col, 2.2, arrow=False)
        f.badge((a + b) / 2, py + 76, (1, 2, 4)[n_], col, r=10)
    steps = [("① 是哪种通信", "all-reduce：各卡相加，每卡一份；to_apply 是加法，结果类型就是相加用的类型"),
             ("② 属于哪个参数", "形状 [120832, 1024] 是词嵌入表；op_name 里有 scatter-add，是反向往表里加"),
             ("③ 每步几次", "它在梯度累积那个 while 循环里；循环次数读条件里的「计数器 ＜ 2」；主输入、多 token 预测各一条 → 2 × 2 ＝ 4 次"),
             ("④ 字节", "120832 × 1024 × 2 字节（BF16）≈ 2.47e8；× 4 次 ＝ 9.90e8 字节／步／设备")]
    for i, (k, v) in enumerate(steps):
        y = py + 100 + i * 82
        f.box(16, y, 150, 56, "#e8f0fe", BL, 6)
        f.t(91, y + 33, k, BL, True, 14, "middle")
        rows = []
        cur = ""
        for ch in v:
            cur += ch
            if len(cur) >= 30 and ch in "，；：、":
                rows.append(cur); cur = ""
        if cur:
            rows.append(cur)
        for j, r in enumerate(rows[:3]):
            f.t(180, y + 18 + j * 20, r, INK, size=13)
    f._pan = None
    RX = LW + 20
    py2 = f.panel(RX, y0, W - RX, PH, "五种跑法，每步每设备字节（v5p 8 设备）", OR)
    runs = [("不修（纯数据并行）", "纯数据并行8"), ("修法① 全程 FP32", "修法1"), ("修法② 本卡 FP32", "修法2"),
            ("FSDP 8（golden）", "FSDP8"), ("只切词嵌入表", "只切词嵌入表")]
    X0, XW = RX + 190, W - RX - 230
    lo, hi = 7.0, 10.55

    def X(v):
        return X0 + (math.log10(v) - lo) / (hi - lo) * XW
    for e in (7, 8, 9, 10):
        f.line(X(10 ** e), py2 + 40, X(10 ** e), py2 + 400, "#f1f3f4", 1, arrow=False)
        f.t(X(10 ** e), py2 + 418, "1e%d" % e, GY, size=12, anchor="middle")
    for i, (lab, key) in enumerate(runs):
        emb, tot = N.C3_COMM[key]
        y = py2 + 52 + i * 70
        f.t(RX + 14, y + 24, lab, INK, True, 13.5)
        f.box(X0, y + 2, X(tot) - X0, 18, GY2, GY2, 3)
        f.t(X(tot) + 6, y + 16, ("%.2e" % tot).replace("e+0", "e").replace("e+", "e"), GY, size=12)
        f.box(X0, y + 26, X(emb) - X0, 18, OR, OR, 3)
        f.t(X(emb) + 6, y + 40, ("%.2e" % emb).replace("e+0", "e").replace("e+", "e"), OR, True, 12.5)
    f.t(RX + 14, py2 + 446, "FSDP 全部通信大，是其余参数也切开了，跟词嵌入无关", GY, size=12.5)
    f._pan = None
    yb = f.band(y0 + PH + 20, "warn", "上卡之前就能看清「传什么、传几次、在哪排序」：离线编一遍，不占芯片", [
        "只切词嵌入表：词嵌入的通信 %s → %s（少 %s），全部集合通信少 %s；跟 FSDP 那一份一条不差。" % ("9.90e8", "6.82e7", N.C3_EMB_CUT_T, N.C3_ALL_CUT_T),
        "代价：字节按形状算，不是线上实测，只适合互相比；看不到精度（同一条 HLO 后端可以生成不同的指令），也看不到快慢（重不重叠要看步时或 profile）。",
    ])
    yb = f.src(yb + 10,
               "📌 做法（对齐记录所用）：用训练同一套配置和同一个训练步，把设备换成目标拓扑的描述，在 CPU 上为它做完整的切分和后端编译（JAX 的提前编译）；让 XLA 把训练步模块存成文本（--xla_dump_to、--xla_dump_hlo_as_text），只看切分、融合、调度都做完以后那一份（after_optimizations）。上卡后存下的程序里，集合通信与离线编的逐条一致（只差设备分组的写法；核对的是只切词嵌入表那份）。",
               "📌 读法：五种集合通信 all-reduce、all-gather、all-to-all、collective-permute、reduce-scatter；异步版 xxx-start／xxx-done 成对，算一次，也可能包成 async-start、calls=子计算，集合通信在子计算里。认参数按形状（词嵌入 [120832,1024]，切开后 [120832,128]；输出头是 [1024,120832]）和 metadata 里的 op_name（前向取行是 gather，反向是 scatter-add），对不上就顺着数据流看使用者。每步次数 ＝ 从入口到它所在计算一路上各层循环次数之积，循环次数读条件计算里 compare(计数器, 常数), direction=LT 的那个常数。字节：all-reduce 算被求和的那块、all-gather 算拼好后的结果、all-to-all 算交换的那块、异步 collective-permute 只算一块缓冲。",
               "📌 右块（取自各运行上卡时存下的训练步程序，每步次数已乘上循环次数；字节仍按形状算）：词嵌入／全部每步每设备字节 —— 不修 %s／%s；修法① %s／%s（主输入和多 token 预测输入合成一次 FP32 all-reduce，每个微批 1 次，仍在循环里；另多 %s 字节的 collective-permute）；修法② %s／%s；FSDP 8 %s／%s；只切词嵌入表 %s／%s（%d 条集合通信）。"
               % tuple(_b(x) if isinstance(x, (int, float)) and x > 1000 else x for x in (N.C3_COMM["纯数据并行8"] + (N.C3_COMM["修法1"][0], N.C3_COMM["修法1"][1], N.C3_FIX1_PERM_T) + N.C3_COMM["修法2"] + N.C3_COMM["FSDP8"] + N.C3_COMM["只切词嵌入表"] + (N.C3_N_COMM,))),
               "⚠️ 线上实际传多少还取决于通信算法：例如 ring all-reduce 每设备约收发 2(n−1)/n 倍，all-to-all 约 (n−1)/n 倍（通用估算）。")
    f.save("fig9-comm.svg", yb + 14)


fig_2x2()
fig_rows()
fig_flow()
fig_rmw()
fig_sweep()
fig_shard()
fig_comm()
