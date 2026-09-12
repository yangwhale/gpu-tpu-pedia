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
PX, PW = [0, 470, 940], [440, 440, 460]


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

    ph = 438

    # ══ ① 三种资源 ＋ 搬家顺序 ══════════════════════════════════
    x, pw = PX[0], PW[0]
    py = f.panel(x, y0, pw, ph, "① 三种资源，一条搬家路线", BL,
                 sub="顺序不是随机的")

    yy = py + 26
    for i, (era, who, what, col) in enumerate([
        ("早期", "搬显存", "MQA → GQA → MLA", BL),
        ("中期", "搬算力", "SWA → NSA / DSA → CSA·HCA", GR),
        ("现在", "搬访存规整度", "chunk 化的线性注意力", OR),
    ]):
        f.box(x + 22, yy, pw - 44, 82, "#fff", col, 8)
        f.box(x + 22, yy, 4, 82, col, col, 2)
        f.box(x + 24, yy, 3, 82, "#fff", "#fff", 0)
        f.t(x + 40, yy + 27, era, GY2, size=11.5)
        f.t(x + 84, yy + 27, who, col, True, 13, cls="svglbl")
        f.t(x + 40, yy + 54, what, GY, size=11.5, w=pw - 76)
        f.t(x + 40, yy + 73, "换来的：" + ["更小的每份", "更少的格子",
                                        "更规整的读法"][i], GY2, size=11)
        if i < 2:
            f.line(x + pw / 2.0, yy + 84, x + pw / 2.0, yy + 92, GY2, 1.2)
        yy += 94

    yy += 2
    f.box(x + 22, yy, pw - 44, 76, "#fff", OR, 8)
    f.box(x + 22, yy, 4, 76, OR, OR, 2)
    f.box(x + 24, yy, 3, 76, "#fff", "#fff", 0)
    f.t(x + 40, yy + 25, "⭐ 为什么是这个顺序？", OR, True, 12.5)
    f.t(x + 40, yy + 47, "<tspan font-weight=\"700\">前两样能用公式算</tspan>；", GY, size=11.5)
    f.t(x + 40, yy + 67, "访存规整度<tspan font-weight=\"700\">只能靠 kernel 一行行写出来</tspan>。", GY,
        size=11.5)
    fits(yy + 76, y0, ph, "①")

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
        f.t(x + 40, yy + 27, "%d. %s" % (i + 1, head), col, True, 12.5,
            w=pw - 76)
        f.t(x + 40, yy + 54, body.split("；")[0] + ("；" if "；" in body else ""),
            GY, size=11.5, w=pw - 76)
        rest = body.split("；")[1] if "；" in body else ""
        if rest:
            f.t(x + 40, yy + 73, rest, GY, size=11.5, w=pw - 76)
        yy += h + 8

    yy += 2
    f.t(x + 22, yy, "⭐ 问「省了多少」之前，先问<tspan font-weight=\"700\">省的是哪一样</tspan>。", INK,
        True, 12.5, w=pw - 44)
    fits(yy + 8, y0, ph, "②")

    # ══ ③ 防骗判据 ══════════════════════════════════════════════
    # ⛔⛔ 2026-09-14 二轮学生审稿，这一格连挨两刀，而且都打在要害上：
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
        "（⚠️ 算力，不是时间）", GY, True, 12, w=pw - 44)
    yy += 12
    bw = pw - 44
    for lab, sq, proj, rest in SH:
        yy += 22
        f.t(x + 22, yy, lab, INK, True, 12)
        f.box(x + 62, yy - 13, bw - 40, 18, "#fff", LINE, 4)
        cx = x + 63
        for frac, col, nm in ((sq, "#fce8e6", "平方项"),
                              (proj, "#fef7e0", "投影"),
                              (rest, "#f1f3f4", "其余")):
            wseg = (bw - 42) * frac
            f.box(cx, yy - 12, max(0.6, wseg), 16,
                  col, "none", 2)
            if wseg > 52:
                f.t(cx + wseg / 2.0, yy - 0.5, nm, GY, size=11, anchor="middle")
            cx += wseg
        yy += 14
        f.t(x + 62, yy, "平方项 %.1f%% · 注意力投影 %.1f%% · 其余 %.1f%%"
            % (sq * 100, proj * 100, rest * 100), GY2, size=11, w=bw - 40)
        yy += 8
    yy += 14

    f.box(x + 22, yy, pw - 44, 104, "#fff", RD, 8)
    f.box(x + 22, yy, 4, 104, RD, RD, 2)
    f.box(x + 24, yy, 3, 104, "#fff", "#fff", 0)
    f.t(x + 40, yy + 25, "⭐ 同一个机制，占比差 <tspan font-weight=\"700\">八倍</tspan>", RD,
        True, 12.5, w=pw - 76)
    f.t(x + 40, yy + 48, "4K 下平方项 %.0f%%，1M 下 %.0f%%。"
        % (SH[0][1] * 100, SH[2][1] * 100), GY, size=11.5, w=pw - 76)
    f.t(x + 40, yy + 70, "⛔ 所以「省了 N 倍」这句话，在这两端"
        "<tspan font-weight=\"700\">根本不是同一件事</tspan>。", GY, size=11.5,
        w=pw - 76)
    f.t(x + 40, yy + 92, "⚠️ 而且短上下文那 12% 之外，注意力自己的投影还占 27%",
        GY2, size=11, w=pw - 76)
    yy += 118

    f.box(x + 22, yy, pw - 44, 100, "#fff", INK, 8)
    f.t(x + 38, yy + 25, "⭐⭐ 所以任何一个倍数，", INK, True, 13,
        cls="svglbl")
    f.t(x + 38, yy + 50, "<tspan font-weight=\"700\">必须带上「在多长的上下文下」</tspan>。", INK,
        True, 13, w=pw - 76)
    f.t(x + 38, yy + 74, "不带这句，那些倍数全是耍流氓。", GY, size=11.5)
    f.t(x + 38, yy + 93, "⭐ 而且要问清楚：省的是 FLOPs，还是墙钟时间？", GY2,
        size=11, w=pw - 76)
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
