# -*- coding: utf-8 -*-
r"""专题三 · §六「CSA 压什么、HCA 图什么、为什么交错」（2026-09-13 夜间 · R5）。

⭐⭐ 现场问的是「CSA 又为什么把四个头压成一个？HCA 又图啥？
   那它俩之间为什么穿插着摆？」

⛔ 第一件事就是**把问题本身校正一下**：CSA 压的**不是头，是 token**。
   ⭐⭐ 而这恰恰是这一张图最该留下的东西 ——&nbsp;**压缩有两个正交的方向**：

     · 纵向：<b>一个 token 存多少个数</b> ——&nbsp;这是旋钮①（MQA / GQA / MLA）
     · 横向：<b>多少个 token 合成一条</b> ——&nbsp;这是 CSA / HCA
     · 再加一个「这一步读哪几条」——&nbsp;那是旋钮②的稀疏

   三样互相独立，所以 V4 **三样一起上**。

⭐ 第二件事是 HCA 的用意，这条是**从两者的定义推出来的**（论文未明述，图中标了口径）：
   CSA = 压得轻 ＋ 挑 → <b>看得细，但会漏</b>（没选中的完全看不见）
   HCA = 压得狠 ＋ 不挑 → <b>不漏，但看得粗</b>
   两种失效模式正好相反，交错摆 = **互相兜底**。
"""
from topic03_draw import (Fig, wpx, BL, OR, GR, RD, GY, PU, CY, INK,
                          GY2, LINE, LINE2, BG2)

W = 1400
PX, PW = [0, 470, 940], [440, 440, 460]


def main():
    def fits(y, y0, ph, who):
        assert y <= y0 + ph - 6, "%s 到 %d，面板底边 %d" % (who, y, y0 + ph)

    f = Fig(W, "CSA 压的是 token 不是头：压缩有纵向和横向两个正交方向；"
               "CSA 压得轻加挑选、HCA 压得狠但密集，两者失效模式相反，"
               "所以交错摆着互相兜底")
    f.marks = set()
    y0 = f.header(
        "CSA 与 HCA　——　压缩其实有两个方向，而这两个方向互不相干",
        "⛔ 先校正一个常见口误：CSA 压的<tspan font-weight=\"700\">不是头，是 token</tspan>"
        "（横着压），旋钮① 压的才是头（竖着压）",
        [(BL, "纵向：一个 token 存多少"), (GR, "横向：几个 token 合一条"),
         (OR, "读哪几条（稀疏）"), (PU, "两种失效模式")])

    ph = 436

    # ══ ① 两个方向 ══════════════════════════════════════════════
    x, pw = PX[0], PW[0]
    py = f.panel(x, y0, pw, ph, "① 压缩有两个正交的方向", BL,
                 sub="把矩阵摆出来就清楚了")

    yy = py + 34
    # 一张 token × 维度 的小矩阵
    R, C = 8, 10
    cw, chh = 22, 20
    mx, my = x + 92, yy
    for r in range(R):
        for c in range(C):
            f.box(mx + c * cw, my + r * chh, cw - 2, chh - 2, BG2, LINE2, 2)
    f.t(mx - 12, my + R * chh / 2.0, "token", GY, True, 11.5, "end")
    f.t(mx - 12, my + R * chh / 2.0 + 16, "（一行一个）", GY2, size=11,
        anchor="end")
    f.t(mx + C * cw / 2.0, my - 12, "一个 token 的表示（一行里的格子）", GY,
        size=11, anchor="middle")

    # 纵向压：把列收窄
    f.line(mx + C * cw + 16, my + 6, mx + C * cw + 16, my + R * chh - 6,
           BL, 1.6, arrow=False)
    f.t(mx + C * cw + 24, my + 22, "旋钮①", BL, True, 11.5)
    f.t(mx + C * cw + 24, my + 40, "MQA/GQA", GY2, size=11)
    f.t(mx + C * cw + 24, my + 56, "MLA", GY2, size=11)
    f.t(mx + C * cw + 24, my + 76, "竖着压", BL, size=11)

    # 横向压：把行合并
    f.line(mx, my + R * chh + 14, mx + C * cw - 4, my + R * chh + 14,
           GR, 1.6, arrow=False)
    f.t(mx, my + R * chh + 34, "CSA / HCA —— 横着压：把连着的几行合成一行",
        GR, True, 11.5, w=pw - 100)

    yy = my + R * chh + 52
    f.box(x + 22, yy, pw - 44, 50, "#fff", OR, 8)
    f.box(x + 22, yy, 4, 50, OR, OR, 2)
    f.box(x + 24, yy, 3, 50, "#fff", "#fff", 0)
    f.t(x + 40, yy + 21, "还有第三件事：这一步<tspan font-weight=\"700\">读哪几行</tspan>", OR, True, 12.5)
    f.t(x + 40, yy + 40, "那是旋钮②的稀疏 —— 跟压不压没关系", GY, size=11.5)
    yy += 62

    f.box(x + 22, yy, pw - 44, 70, "#fff", BL, 8)
    f.t(x + 38, yy + 24, "⭐⭐ 三样互不相干，所以可以同时上", BL, True, 13,
        cls="svglbl")
    f.t(x + 38, yy + 46, "DeepSeek-V4 就是三样一起：MLA ＋ 横压 ＋ 稀疏",
        GY, size=11.5)
    f.t(x + 38, yy + 64, "⛔ 别把它们当成三个竞品 —— 它们是三个轴", GY2,
        size=11)
    fits(yy + 70, y0, ph, "①")

    # ══ ② CSA：先压再挑 ═════════════════════════════════════════
    x, pw = PX[1], PW[1]
    py = f.panel(x, y0, pw, ph, "② CSA：先压一点，再挑", GR,
                 sub="压缩 ＋ 稀疏，两个省法相乘")

    yy = py + 24
    CS = [
        ("第一步　每 m 个 token 压成一条",
         "<tspan font-weight=\"700\">不是简单平均</tspan> —— 两组 KV 各配一组可学权重，",
         "softmax 归一后加权合并；窗口还是<tspan font-weight=\"700\">重叠</tspan>的"),
        ("第二步　在压缩后的条目上跑 DSA",
         "每个 query 只挑 top-k 条 —— <tspan font-weight=\"700\">挑的是压缩条目，</tspan>",
         "<tspan font-weight=\"700\">不是原始 token</tspan>，于是索引器要打分的对象少了 m 倍"),
        ("第三步　再并上一小段滑窗",
         "补回近处的细粒度依赖 ——",
         "⭐ 「近处永远保留」在 NSA 里也有，是这一支的共同结构"),
    ]
    for title, l1, l2 in CS:
        f.box(x + 22, yy, pw - 44, 92, "#fff", GR, 8)
        f.box(x + 22, yy, 4, 92, GR, GR, 2)
        f.box(x + 24, yy, 3, 92, "#fff", "#fff", 0)
        f.t(x + 40, yy + 24, title, GR, True, 12.5)
        f.t(x + 40, yy + 48, l1, GY, size=11.5, w=pw - 76)
        f.t(x + 40, yy + 70, l2, GY, size=11.5, w=pw - 76)
        yy += 102

    yy += 2
    f.box(x + 22, yy, pw - 44, 64, "#fff", LINE, 8)
    f.t(x + 38, yy + 24, "⭐ 两个省法是<tspan font-weight=\"700\">相乘</tspan>的：压 m 倍 × 只读 k 条",
        INK, True, 12.5)
    f.t(x + 38, yy + 46, "这就是为什么它敢把上下文推到一百万", GY, size=11.5)
    fits(yy + 64, y0, ph, "②")

    # ══ ③ HCA 图啥 ＋ 为什么交错 ═════════════════════════════════
    x, pw = PX[2], PW[2]
    py = f.panel(x, y0, pw, ph, "③ HCA 图啥，为什么交错", PU,
                 sub="两种失效模式，正好相反")

    yy = py + 24
    f.box(x + 22, yy, pw - 44, 56, "#fff", PU, 8)
    f.box(x + 22, yy, 4, 56, PU, PU, 2)
    f.box(x + 24, yy, 3, 56, "#fff", "#fff", 0)
    f.t(x + 40, yy + 23, "HCA：每 m′ 个 token 压成一条（m′ ≫ m）", PU,
        True, 12.5)
    f.t(x + 40, yy + 43, "但是 <tspan font-weight=\"700\">不挑，全看</tspan> —— 保持密集注意力", GY,
        size=11.5)
    yy += 72

    f.t(x + 22, yy, "⭐⭐ 为什么要两种？看它们各自怎么坏", INK, True, 13.5,
        cls="svglbl")
    yy += 26
    for who, good, bad, col in [
        ("CSA　压得轻 ＋ 挑", "被选中的那几块<tspan font-weight=\"700\">看得很细</tspan>",
         "⛔ 没选中的<tspan font-weight=\"700\">完全看不见</tspan> —— 会漏", GR),
        ("HCA　压得狠 ＋ 全看", "<tspan font-weight=\"700\">一个位置都不漏</tspan>",
         "⛔ 每个位置只剩一个很糙的摘要", PU),
    ]:
        f.box(x + 22, yy, pw - 44, 78, "#fff", col, 8)
        f.t(x + 38, yy + 23, who, col, True, 12.5)
        f.t(x + 38, yy + 45, "✓ " + good, GY, size=11.5, w=pw - 76)
        f.t(x + 38, yy + 66, bad, GY, size=11.5, w=pw - 76)
        yy += 88

    f.box(x + 22, yy, pw - 44, 78, "#fff", INK, 8)
    f.t(x + 38, yy + 24, "⭐ 两种坏法正好相反 —— 所以交错摆着，", INK,
        True, 12.5)
    f.t(x + 38, yy + 44, "让它们互相兜底。", INK, True, 12.5)
    f.t(x + 38, yy + 66, "📌 这条是从两者定义推出来的，不是论文原话", GY2,
        size=11)
    fits(yy + 78, y0, ph, "③")

    # ══ 落点带 ══════════════════════════════════════════════════
    yy = y0 + ph + 22
    yy = f.band(yy, "info", "⭐⭐ 这一张要带走的一句：压缩有两个方向，它们互不相干", [
        "<tspan font-weight=\"700\">竖着压</tspan>（一个 token 存多少个数）是旋钮①；"
        "<tspan font-weight=\"700\">横着压</tspan>（几个 token 合成一条）是 CSA/HCA；"
        "<tspan font-weight=\"700\">读哪几条</tspan>是旋钮②。",
        "⭐ 想清楚这三个轴，V4 那套「CSA＋HCA」就不是一个新名词，"
        "而是<tspan font-weight=\"700\">三个轴同时拧到一个新位置</tspan>。"
        "⛔ 拿到任何一个新方案，先问它<tspan font-weight=\"700\">动了哪几个轴</tspan>。",
    ])

    yy = f.band(yy + 14, "ok", "成绩：一百万上下文成为常规配置", [
        "1M 上下文下，DeepSeek-V4-Pro 只要 DeepSeek-V3.2 的 "
        "<tspan font-weight=\"700\">27% 单 token 推理 FLOPs</tspan> 和 "
        "<tspan font-weight=\"700\">10% 的 KV cache</tspan>。",
        "⭐ 注意这两个数<tspan font-weight=\"700\">不一样</tspan> ——&#160;"
        "FLOPs 省到 27%、KV 省到 10%，"
        "说明<tspan font-weight=\"700\">横着压主要省的是存储，稀疏主要省的是计算</tspan>。",
    ])

    yy = f.src(yy + 16,
               "CSA / HCA 的机制出自 DeepSeek-V4 技术报告 arXiv 2606.19348 "
               "§2.3–2.3.1（每 m 个压一条 → DSA top-k → 并上滑窗；HCA 压 m′≫m 但保持密集）",
               "27% FLOPs / 10% KV cache 出自同一篇摘要与 §2.3.4（1M 上下文、对比 V3.2）",
               "⚠️ 「两种失效模式互补、所以交错」是从两者定义推出的解释，"
               "论文只说了采用交错混合配置，未给这个理由，也未在此给出层间配比")
    f.save("fig3-csa-why.svg", yy + 6)


main()
