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
   注意力只是账单的一部分。短上下文下它只占 12% ——&nbsp;
   **这时候你把它优化到极致，端到端也就快 10%。**
   所以任何一个倍数，**必须带上「在多长的上下文下」**。
"""
from topic03_draw import (Fig, wpx, BL, OR, GR, RD, GY, PU, CY, INK,
                          GY2, LINE, LINE2, BG2)

W = 1400
PX, PW = [0, 470, 940], [440, 440, 460]


def main():
    def fits(y, y0, ph, who):
        assert y <= y0 + ph - 6, "%s 到 %d，面板底边 %d" % (who, y, y0 + ph)

    SHARE, GAIN = 12, 10          # 短上下文下注意力占比 / 砍到 0 的端到端收益
    assert GAIN < SHARE           # 砍到 0 也不可能超过它本来占的那一份

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
    x, pw = PX[2], PW[2]
    py = f.panel(x, y0, pw, ph, "③ 一条防骗判据", RD,
                 sub="这一讲最该带走的一句")

    yy = py + 30
    # 一根账单条：注意力只占一小段
    bw = pw - 44
    f.t(x + 22, yy, "短上下文下，一次前向的时间都花在哪", GY, True, 12)
    yy += 14
    f.box(x + 22, yy, bw, 40, "#fff", LINE, 6)
    f.box(x + 24, yy + 2, bw * SHARE / 100.0 - 2, 36, "#fce8e6", "none", 4)
    f.t(x + 24 + bw * SHARE / 100.0 / 2.0, yy + 25, "注意力", RD, True, 11.5,
        "middle")
    f.t(x + 24 + bw * (SHARE + 100) / 200.0, yy + 25,
        "MoE ＋ MLP ＋ 通信", GY, True, 12, "middle")
    yy += 56
    f.t(x + 22, yy, "注意力约占 <tspan font-weight=\"700\">%d%%</tspan>（专题一那条曲线）" % SHARE, GY,
        size=11.5, w=pw - 44)
    yy += 28

    f.box(x + 22, yy, pw - 44, 96, "#fff", RD, 8)
    f.box(x + 22, yy, 4, 96, RD, RD, 2)
    f.box(x + 24, yy, 3, 96, "#fff", "#fff", 0)
    f.t(x + 40, yy + 26, "把注意力<tspan font-weight=\"700\">砍到 0</tspan>，端到端也就快", RD, True, 12.5)
    f.t(x + 40, yy + 54, "%d%% 左右" % GAIN, RD, True, 22)
    f.t(x + 40, yy + 82, "剩下那 %d%% 一分钱没省。" % (100 - SHARE), GY,
        size=11.5)
    yy += 110

    f.box(x + 22, yy, pw - 44, 96, "#fff", INK, 8)
    f.t(x + 38, yy + 26, "⭐⭐ 所以任何一个倍数，", INK, True, 13,
        cls="svglbl")
    f.t(x + 38, yy + 52, "<tspan font-weight=\"700\">必须带上「在多长的上下文下」</tspan>。", INK,
        True, 13, w=pw - 76)
    f.t(x + 38, yy + 76, "不带这句，那些倍数全是耍流氓。", GY, size=11.5)
    fits(yy + 96, y0, ph, "③")

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
               "「短上下文下注意力约占 12%」出自专题一那条曲线；"
               "「砍到 0 也就快 10%」由此直接推出 ——&#160;"
               "⚠️ 这是<tspan font-weight=\"700\">量级示意</tspan>，"
               "具体占比随模型结构、批大小、序列长度变")
    f.save("fig3-landing.svg", yy + 6)


main()
