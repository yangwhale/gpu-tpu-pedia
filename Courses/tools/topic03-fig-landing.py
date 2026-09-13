# -*- coding: utf-8 -*-
r"""专题三 · §九＋§十「三种资源之间的搬家史」（2026-09-13 夜间 · R15）。

⭐⭐⭐ 这是**整个专题的落点**，而它原来只有两张表和一段散文。
   它该是一张图，因为它讲的本来就是一个**空间里的移动**：

     注意力的变体史 ＝ 在**显存 / 算力 / 访存规整度**三者之间反复搬家。
       早期搬显存（MQA → GQA → MLA）
       中期搬算力（稀疏：SWA → NSA / DSA → CSA/HCA）
       现在搬访存规整度（chunk 化的线性注意力）
     ⭐ 而**访存规整度是最难搬的那一样** ——&nbsp;
       前两样能用公式算，这一样只能靠 kernel 一行一行写出来。

⭐⭐ 第三格是这一讲**唯一一条能防住「被数字骗」的判据**：
   注意力只是账单的一部分，而且**占多少完全看上下文多长**：
   4K 下平方项只占 12.3%，1M 下 97.3% ——&nbsp;
   **同一个机制在两端根本不是同一件事。**
   所以任何一个倍数，**必须带上「在多长的上下文下」**。
"""
from topic03_draw import (Fig, wpx, BL, OR, GR, RD, GY, PU, CY, INK,
                          GY2, LINE, LINE2, BG2)

W = 1400
# ① 提到上面吃满整幅宽之后，②③ 两栏各 690px（原来三栏各 440）
PX, PW = [0, 0, 710], [0, 690, 690]


def main():
    def fits(y, y0, ph, who):
        assert y <= y0 + ph - 6, "%s 到 %d，面板底边 %d" % (who, y, y0 + ph)

    # ⛔ 2026-09-13 学生审稿：原来写「占 12%、砍到 0 快 10%」——&nbsp;
    #   砍掉 12% 的时间就是快 12%，10% 这个数凭空保守了两个点，
    #   而两个具体数并排摆着会被当成两个独立事实。
    # ⭐ 改成：占比给具体数（4K 锚点），收益只说「一成出头」。
    SHARE = 12                    # 只保留给旧引用；三个长度的构成在 ③ 里当场算

    f = Fig(W, "三种资源之间的搬家史：早期搬显存、中期搬算力、现在搬访存规整度；"
               "四个取舍；以及注意力只是账单的一部分这条防骗判据")
    f.marks = set()
    y0 = f.header(
        "落点　——　注意力的变体史，是一部在三种资源之间反复搬家的历史",
        "⭐⭐ 搬的顺序是有道理的："
        "<tspan font-weight=\"700\">先搬能算的，最后才搬算不出来的</tspan>",
        [(BL, "显存"), (GR, "算力"), (OR, "访存规整度"), (RD, "防骗判据")])

    # ⭐⭐⭐ 2026-09-13 重画 ① ——&nbsp;审图原话：
    #   「整个专题的落点，现在是三栏文字卡；『三个房间、一条搬家路线』
    #     只画了三个空矩形叠在一起，**搬家这件事一点没画**。」
    # ⛔ 而它挤在 440px 的一栏里，画什么都施展不开。
    # ⭐ 判据还是那条：**栏宽是可读性定的，不是内容条数定的。**
    #   ① 提到上面吃满整幅宽，②③ 留在下面并排。
    ph = 520
    PH1 = 320

    # ══ ① 真的画一次搬家 ════════════════════════════════════════
    py = f.panel(0, y0, W, PH1, "① 三个房间，一条搬家路线", BL,
                 sub="⭐ 顺序不是随机的 ——　先搬能拿尺子量的，最后才搬量不出来的")

    ry = py + 30
    RW, RGAP, RH = 400, 60, 170
    ROOMS = [
        ("早期", "显存", BL, "#e8f0fe", "MQA → GQA → MLA", "每份更小", "shrink"),
        ("中期", "算力", GR, "#e6f4ea", "SWA → DSA → CSA", "格子更少", "few"),
        ("现在", "访存规整度", OR, "#fef7e0", "chunk 化的线性", "读得更顺", "stuck"),
    ]
    for i, (era, name, col, tint, who, got, kind) in enumerate(ROOMS):
        rx = 60 + i * (RW + RGAP)
        # 房间：只画三面墙 ＋ 一个门口，别画成又一个矩形卡片
        f.box(rx, ry, RW, RH, tint, col, 10)
        f.t(rx + 20, ry + 34, "%s ——　搬「%s」这个房间" % (era, name), col, True, 20)
        gx, gy = rx + 22, ry + 56
        if kind == "shrink":                    # 一摞箱子 → 一个小箱子
            for k in range(4):
                f.box(gx + k * 30, gy + 6, 24, 44, "#fff", col, 4)
            f.line(gx + 128, gy + 28, gx + 166, gy + 28, col, 2.0)
            f.box(gx + 174, gy + 20, 24, 18, "#fff", col, 4)
            f.t(gx + 212, gy + 34, "东西还是那些，", GY, size=16)
            f.t(gx + 212, gy + 56, "每一份变小了", GY, size=16)
        elif kind == "few":                     # 格子还在，只搬走其中两个
            for k in range(6):
                on = k in (1, 4)
                f.box(gx + k * 34, gy + 6, 28, 44,
                      col if on else "#fff", "none" if on else LINE2, 4)
            f.t(gx + 212, gy + 34, "格子一个没少，", GY, size=16)
            f.t(gx + 212, gy + 56, "这一趟只搬两个", GY, size=16)
        else:                                   # 门口卡住一个箱子
            f.box(gx, gy + 6, 92, 44, "#fff", col, 4)
            f.line(gx + 100, gy + 28, gx + 138, gy + 28, col, 2.0)
            # 门框
            f.box(gx + 146, gy - 4, 12, 64, "#fff", GY2, 2)
            f.box(gx + 158, gy + 10, 52, 36, "#fce8e6", RD, 4)
            f.t(gx + 184, gy + 34, "卡住", RD, True, 16, "middle")
            f.t(gx + 226, gy + 34, "门就那么宽 ——", GY, size=16)
            f.t(gx + 226, gy + 56, "只能一行行写", GY, size=16)
        f.t(rx + 20, ry + 146, who, GY2, size=16)
        f.t(rx + RW - 20, ry + 146, "换来：" + got, col, True, 17, anchor="end")
        if i < 2:                               # 搬运路线
            f.line(rx + RW + 8, ry + RH / 2.0, rx + RW + RGAP - 8,
                   ry + RH / 2.0, GY2, 2.0)

    f.box(60, ry + RH + 18, W - 120, 82, "#fff", INK, 10)
    f.t(84, ry + RH + 46, "⭐ 为什么偏偏是这个顺序", INK, True, 20)
    f.t(340, ry + RH + 46, "前两样<tspan font-weight=\"700\">能拿尺子量</tspan>"
        " ——　多少字节、多少 FLOPs，坐下来就能算。", GY, size=17, w=940)
    f.t(340, ry + RH + 72, "⛔ 而访存规整度<tspan font-weight=\"700\">量不出来</tspan>"
        "，它只出现在 kernel 里 ——　所以被留到了最后。", GY, size=17, w=940)

    y0 = y0 + PH1 + 18

    # ══ ② 四个取舍 ══════════════════════════════════════════════
    x, pw = PX[1], PW[1]
    py = f.panel(x, y0, pw, ph, "② 四个取舍，一个都别漏", GR,
                 sub="每一条都是一次翻车预防")

    yy = py + 22
    for i, (head, body) in enumerate([
        ("省显存 ≠ 省计算",
         "MLA 省显存却<tspan font-weight=\"700\">加了计算</tspan>；DSA 省计算但 <tspan font-weight=\"700\">KV 还在那儿</tspan>。"),
        ("训练时省 ≠ 推理时省",
         "MLA 的压缩<tspan font-weight=\"700\">在训练前向里不生效</tspan>；NSA 的 native 意味着训练也省。"),
        ("不规则访存的代价常被低估",
         "纸面 64 倍，落到 gather 和不连续访问上<tspan font-weight=\"700\">远拿不到</tspan>。"),
        ("⭐ 收益有天花板",
         "注意力只是账单的一部分；<tspan font-weight=\"700\">MoE、MLP、通信一分没省</tspan>。"),
    ]):
        h = 84
        col = RD if i == 3 else GR
        f.box(x + 22, yy, pw - 44, h, "#fff", col, 8)
        f.box(x + 22, yy, 4, h, col, col, 2)
        f.box(x + 24, yy, 3, h, "#fff", "#fff", 0)
        f.t(x + 40, yy + 27, "%d. %s" % (i + 1, head), col, True, 16,
            w=pw - 76)
        f.t(x + 40, yy + 54, body.split("；")[0] + ("；" if "；" in body else ""),
            GY, size=14.5, w=pw - 76)
        rest = body.split("；")[1] if "；" in body else ""
        if rest:
            f.t(x + 40, yy + 73, rest, GY, size=14.5, w=pw - 76)
        yy += h + 8

    yy += 2
    f.t(x + 22, yy, "⭐ 问「省了多少」之前，先问<tspan font-weight=\"700\">省的是哪一样</tspan>。", INK,
        True, 16, w=pw - 44)
    fits(yy + 8, y0, ph, "②")

    # ══ ③ 防骗判据 ══════════════════════════════════════════════
    # ⛔⛔ 2026-09-12 二轮学生审稿，这一格连挨两刀，而且都打在要害上：
    #   ① 标题写「一次前向的**时间**」，可 SHARE 的定义是
    #      **平方项 FLOP ÷ 总前向 FLOP** —— 是算力不是时间。
    #      而本讲 §3.6 自己写着：同样 seq=4096，splash attention
    #      **占 23% 的时间**、效率只有 35.5%（全场最低）。效率最低的算子，
    #      时间占比一定高于 FLOPs 占比 —— **两处正面对撞**。
    #   ② 那 12% **只是平方项**。同一条件下注意力自己的投影（Q/K/V/O）
    #      还占 27.4%，所以把剩下那格标成「MoE ＋ MLP ＋ 通信 88%」是错的。
    # ⭐⭐ 改完反而比原版更有教学价值：**FLOPs 省了不等于时间省了**，
    #   正是这一格想教的判据本身。
    x, pw = PX[2], PW[2]
    py = f.panel(x, y0, pw, ph, "③ 一条防骗判据", RD,
                 sub="这一讲最该带走的一句")

    # 三个长度的算力构成，当场算、当场断言（公式与常数同专题一那条曲线）
    CC = dict(d=7168, L=61, nDense=3, V=129280, H=128, qLora=1536, kvLora=512,
              nope=128, rope=64, vDim=128, nExp=256, nShr=1, topK=8,
              dMoe=2048, dFf=18432)
    CC["nMoe"] = CC["L"] - CC["nDense"]
    CC["qkDim"] = CC["nope"] + CC["rope"]

    def share(seq):
        mlaW = (CC["d"] * CC["qLora"] + CC["qLora"] * CC["H"] * CC["qkDim"]
                + CC["d"] * (CC["kvLora"] + CC["rope"])
                + CC["kvLora"] * CC["H"] * (CC["nope"] + CC["vDim"])
                + CC["H"] * CC["vDim"] * CC["d"])
        proj = 2 * seq * mlaW * CC["L"]
        sq = (CC["H"] * seq * seq * (CC["qkDim"] + CC["vDim"])) * CC["L"]
        dense = 2 * seq * 3 * CC["d"] * CC["dFf"] * CC["nDense"]
        moe = 2 * seq * (CC["topK"] + CC["nShr"]) * 3 * CC["d"] * CC["dMoe"] * CC["nMoe"]
        head = 2 * seq * CC["V"] * CC["d"]
        tot = proj + sq + dense + moe + head
        return sq / tot, proj / tot, (dense + moe + head) / tot

    LENS = [("4K", 4096), ("128K", 131072), ("1M", 1048576)]
    SH = [(lab,) + share(n) for lab, n in LENS]
    assert abs(SH[0][1] - .1229) < .001 and abs(SH[2][1] - .9729) < .001

    yy = py + 26
    f.t(x + 22, yy, "一次前向的<tspan font-weight=\"700\">算力</tspan>都花在哪"
        "（⚠️ 算力，不是时间）", GY, True, 15, w=pw - 44)
    yy += 12
    bw = pw - 44
    for lab, sq, proj, rest in SH:
        yy += 22
        f.t(x + 22, yy, lab, INK, True, 15)
        f.box(x + 62, yy - 13, bw - 40, 18, "#fff", LINE, 4)
        cx = x + 63
        for frac, col, nm in ((sq, "#fce8e6", "平方项"),
                              (proj, "#fef7e0", "投影"),
                              (rest, "#f1f3f4", "其余")):
            wseg = (bw - 42) * frac
            f.box(cx, yy - 12, max(0.6, wseg), 16,
                  col, "none", 2)
            if wseg > 52:
                f.t(cx + wseg / 2.0, yy - 0.5, nm, GY, size=14, anchor="middle")
            cx += wseg
        yy += 14
        f.t(x + 62, yy, "平方项 %.1f%% · 注意力投影 %.1f%% · 其余 %.1f%%"
            % (sq * 100, proj * 100, rest * 100), GY2, size=14, w=bw - 40)
        yy += 8
    yy += 14

    f.box(x + 22, yy, pw - 44, 104, "#fff", RD, 8)
    f.box(x + 22, yy, 4, 104, RD, RD, 2)
    f.box(x + 24, yy, 3, 104, "#fff", "#fff", 0)
    f.t(x + 40, yy + 25, "⭐ 同一个机制，占比差 <tspan font-weight=\"700\">八倍</tspan>", RD,
        True, 16, w=pw - 76)
    f.t(x + 40, yy + 48, "4K 下平方项 %.0f%%，1M 下 %.0f%%。"
        % (SH[0][1] * 100, SH[2][1] * 100), GY, size=14.5, w=pw - 76)
    f.t(x + 40, yy + 70, "⛔ 「省了 N 倍」在这两端"
        "<tspan font-weight=\"700\">根本不是同一件事</tspan>。", GY, size=14.5,
        w=pw - 76)
    f.t(x + 40, yy + 92, "⚠️ 而且短上下文那 12% 之外，注意力自己的投影还占 27%",
        GY2, size=14, w=pw - 76)
    yy += 118

    f.box(x + 22, yy, pw - 44, 100, "#fff", INK, 8)
    f.t(x + 38, yy + 25, "⭐⭐ 所以任何一个倍数，", INK, True, 16,
        cls="svglbl")
    f.t(x + 38, yy + 50, "<tspan font-weight=\"700\">必须带上「在多长的上下文下」</tspan>。", INK,
        True, 16, w=pw - 76)
    f.t(x + 38, yy + 74, "不带这句，那些倍数全是耍流氓。", GY, size=14.5)
    f.t(x + 38, yy + 93, "⭐ 而且要问清楚：省的是 FLOPs，还是墙钟时间？", GY2,
        size=14, w=pw - 76)
    fits(yy + 100, y0, ph, "③")

    # ══ 落点带 ══════════════════════════════════════════════════
    yy = y0 + ph + 22
    yy = f.band(yy, "info", "⭐⭐ 一句话收全课：访存规整度是最难搬的那一样", [
        "<tspan font-weight=\"700\">显存</tspan>能算、<tspan font-weight=\"700\">算力</tspan>能算 ——&#160;"
        "所以这两样先被搬完了；"
        "<tspan font-weight=\"700\">访存规整度</tspan>算不出来，"
        "它只出现在 kernel 里、出现在 SM 空转的那几个微秒里。",
        "⭐ 这也是为什么今天这一支的前沿工作看起来越来越像"
        "<tspan font-weight=\"700\">「写 kernel」而不是「改模型」</tspan> ——&#160;"
        "FlashKDA、TileLang、专用的上下文并行，全是这一类。",
        "⛔ 对应到评估：<tspan font-weight=\"700\">别看 FLOPs 省了多少，看墙钟时间省了多少</tspan>；"
        "也别只看单算子，看端到端。",
    ])

    yy = f.src(yy + 16,
               "四个取舍与硬件假设表见 §九 / §十 正文（每条都可追到前面对应小节）",
               "③ 的三根条<tspan font-weight=\"700\">由本脚本当场算并断言</tspan>，"
               "公式与常数同专题一那条曲线（V3：61 层 / 128 头 / MoE top-8＋1 共享，"
               "因果掩码按半算）",
               "⚠️ 它是 <tspan font-weight=\"700\">FLOPs 口径，不是时间</tspan> ——&#160;"
               "同样 seq=4096，本讲 §3.6 量到 splash attention "
               "<tspan font-weight=\"700\">占 23% 的时间</tspan>、效率只有 35.5%；"
               "<tspan font-weight=\"700\">效率最低的算子，时间占比一定高于算力占比</tspan>"
               "⚠️ 这是<tspan font-weight=\"700\">量级示意</tspan>，"
               "具体占比随模型结构、批大小、序列长度变")
    f.save("fig3-landing.svg", yy + 6)


main()
