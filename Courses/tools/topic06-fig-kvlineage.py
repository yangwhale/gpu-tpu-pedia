# -*- coding: utf-8 -*-
r"""专题六 · 3.1 的演进图：DeepSeek 一条线上四代 KV，各省在「存」还是「读」。

⭐ 回答的问题：
   fig6-kv-lineage —— V3（MLA）→ V3.2-Exp（DSA）→ V4（CSA／HCA）→ V4.1-Flash，
                      每代每个 token 存多少、生成一个字读多少？DSA 省的是读，V4／V4.1 省的是存。

📌 数字从哪来（只用课件已有的，一个不新编）：
   · MLA 70,272 B/token：3.1 表格（V3 config.json 推导），topic06_numbers.V3_KV_PER_TOKEN。
   · DSA 每字只读 2,048 个历史 token、打分器仍扫全部（头少、FP8）：台账 R16（V3.2 报告 §2.1）、R21 · 15.3。
   · V4：每层一条所有头共用的 KV；CSA 每 4 个 token 并一条、只挑一部分读；HCA 每 128 个并一条、全读；
     最近 128 个 token 不压：台账 R18（V4 技术报告 §2.3；V4-Pro config.json compress_ratios 4／128、sliding_window 128）。
   · V4.1-Flash：各层共用主 KV 和索引 K、主 KV 存 FP4，全局 KV 890 B/token，约 V4-Flash 的 1/4：台账 R25 · 0／3.1（HF 模型卡）。
   · 年份：MLA 2024（3.1 表格）；V3.2-Exp 2025-09（台账 R16）；V4 2026-04、V4.1-Flash 2026-09（台账 R25）。

⛔ 刻意没画：
   · V4／V4-Flash 每 token 的字节数 —— 课件没给，不从「1/4」反推。
   · V4.1-Flash 读法的变化 —— 模型卡只写了存，读那一格只记「索引 K 也跨层共用」。
   · 方块个数、压缩块的宽度都是示意，不按比例（128 个并一条画不下）。
"""
from topic03_draw import Fig, GR, GY, GY2, PU, INK, LINE
from topic06_numbers import V3_KV_PER_TOKEN

assert V3_KV_PER_TOKEN == 70272
W = 1400
LX = 120                     # 左侧行标签栏
CW = (W - LX) / 4.0          # 每代一列
PAD = 14
SQ, SG = 15, 3               # 示意方块边长、间距（16 个 ＝ 288px）
PU_T, GR_T = "#f3e8fd", "#e6f4ea"   # 紫、绿的浅底
HL = "#f9ab00"               # 「这一代改的就是这一排」的框（大块浅底会被 _flatten 刷白，所以用框）
HLT = "#b06000"


def fig_lineage():
    f = Fig(W, "DeepSeek 一条线上四代 KV，上排是每个 token 存多少，下排是生成一个字读多少。"
               "V3 的 MLA 每个 token 存 70,272 字节，每个字读全部历史。"
               "V3.2-Exp 的 DSA 照样全存，但每个字只读打分器挑出的 2,048 个历史 token，打分器自己仍扫全部。"
               "V4 每层只存一条所有头共用的 KV，再沿 token 压缩：CSA 每 4 个并一条、只挑一部分读，HCA 每 128 个并一条、全读，最近 128 个不压。"
               "V4.1-Flash 让各层共用主 KV、主 KV 存 FP4，每 token 只剩 890 字节，约 V4-Flash 的四分之一")
    y0 = f.header("DSA 省的是读　——　<tspan font-weight=\"700\">V4、V4.1-Flash 省的是存</tspan>",
                  "DeepSeek 一条线上的四代，每代一列。上排：每个 token 存下多少 KV；下排：生成一个字时读回多少。"
                  "方块只是示意，不按比例",
                  [(PU, "存着的 KV"), (GR, "生成这个字时读的"), (HL, "黄框：这一代改的是这一排")])

    # ── 列头：时间线 ──────────────────────────────────────────────
    HEADS = [("2024", "DeepSeek-V3", "MLA"),
             ("2025-09", "V3.2-Exp", "MLA ＋ DSA"),
             ("2026-04", "V4", "CSA ＋ HCA"),
             ("2026-09", "V4.1-Flash", "跨层共用 ＋ FP4")]
    hy = y0 + 6
    for i, (yr, nm, how) in enumerate(HEADS):
        cx = LX + CW * i + CW / 2
        f.t(cx, hy + 14, yr, GY, size=14, anchor="middle")
        f.t(cx, hy + 38, nm, INK, True, 18, "middle")
        f.t(cx, hy + 60, how, PU, True, 15, "middle")
        if i < 3:
            f.line(LX + CW * (i + 1) - 30, hy + 32, LX + CW * (i + 1) + 30, hy + 32, GY2, 1.6, arrow=True)
    ytop = hy + 80

    def cx0(i):
        return LX + CW * i + PAD

    def squares(x, y, n, fill, stroke, sz=SQ, gap=SG, read=None):
        """一排 n 个 token；read 是被读的下标集合（绿框）。"""
        for k in range(n):
            xx = x + k * (sz + gap)
            if read is None:
                f.box(xx, y, sz, sz, fill, stroke, 2, 1.2)
            elif k in read:
                f.box(xx, y, sz, sz, GR_T, GR, 2, 2.2)
            else:
                f.box(xx, y, sz, sz, "#fff", LINE, 2, 1)

    def merged(x, y, widths, sz, gap, fill, stroke, sw=1.2, read=None, h=None):
        """压缩后的块：每块占 widths[k] 个 token 宽；h 是块高（默认 ＝ sz）。"""
        h = h or sz
        xx = x
        for k, wd in enumerate(widths):
            ww = wd * (sz + gap) - gap
            if read is None:
                f.box(xx, y, ww, h, fill, stroke, 2, sw)
            elif k in read:
                f.box(xx, y, ww, h, GR_T, GR, 2, 2.2)
            else:
                f.box(xx, y, ww, h, "#fff", LINE, 2, 1)
            xx += ww + gap

    def v4_strips(x, y, mode, thin=False):
        """V4 的三种存法；mode ＝ "store" 或 "read"。thin：V4.1 的块画薄一半（示意 FP4、跨层共用）。"""
        s, g = 11, 1                       # 16 个 ＝ 192px
        sx = x + 96
        fill = PU_T
        rows = [("CSA 4→1", "csa"), ("HCA 128→1", "hca")]
        if mode == "store":
            rows.append(("最近 128 个", "win"))
        for j, (lab, kind) in enumerate(rows):
            yy = y + j * 24
            f.t(x, yy + 10, lab, GY, size=13)
            if thin:
                yy += 3
            hh = 5 if thin else s
            if kind == "csa":
                merged(sx, yy, [4, 4, 4, 4], s, g, fill, PU, h=hh,
                       read=None if mode == "store" else {1, 3})
            elif kind == "hca":
                merged(sx, yy, [16], s, g, fill, PU, h=hh,
                       read=None if mode == "store" else {0})
            else:
                merged(sx, yy, [10], s, g, "none", LINE, h=hh)   # 前段：已经压进上两排
                merged(sx + 10 * (s + g), yy, [1] * 6, s, g, fill, PU, h=hh)
        return y + len(rows) * 24

    # ── 上排：存 ─────────────────────────────────────────────────
    RH1 = 172
    py = f.panel(0, ytop, W, RH1, "存：每个 token 留下多少", PU)
    f.t(18, py + 60, "存", PU, True, 26)
    def mark(i, top, h, tag):
        x = LX + CW * i + 4
        f.box(x, top, CW - 8, h, "none", HL, 6, 2, dash="6 4")
        f.t(x + CW - 22, top + h - 12, tag, HLT, True, 17, "end")   # 右下角：右上角会压到方块

    for i in (2, 3):                       # 这两代改的是存
        mark(i, py + 6, RH1 - 42, "省存")
    sy = py + 18
    # V3
    squares(cx0(0), sy, 16, PU_T, PU)
    f.t(cx0(0), sy + 50, "每个 token 存一条", INK, True, 15)
    f.t(cx0(0), sy + 74, "70,272 B（61 层合计）", PU, True, 15)
    # V3.2-Exp
    squares(cx0(1), sy, 16, PU_T, PU)
    f.t(cx0(1), sy + 50, "照 MLA 存，一条不少", INK, True, 15)
    f.t(cx0(1), sy + 74, "和左边一样多", GY, size=14)
    # V4
    ye = v4_strips(cx0(2), sy, "store")
    f.t(cx0(2), ye + 18, "每层一条，所有头共用；", INK, True, 15)
    f.t(cx0(2), ye + 40, "再沿 token 方向压", INK, True, 15)
    # V4.1-Flash
    ye = v4_strips(cx0(3), sy, "store", thin=True)
    f.t(cx0(3), ye + 18, "890 B/token", PU, True, 17)
    f.t(cx0(3) + 116, ye + 18, "约 V4-Flash 的 1/4", INK, True, 14)
    f.t(cx0(3), ye + 40, "几层共用一份，主 KV 存 FP4", GY, size=14)

    # ── 下排：读 ─────────────────────────────────────────────────
    RH2 = 150
    py = f.panel(0, ytop + RH1 + 16, W, RH2, "读：生成一个字时读回多少", GR)
    f.t(18, py + 56, "读", GR, True, 26)
    mark(1, py + 6, RH2 - 42, "省读")      # DSA 改的是读
    ry = py + 20
    squares(cx0(0), ry, 16, None, None, read=set(range(16)))
    f.t(cx0(0), ry + 50, "每个字读全部历史", INK, True, 15)
    squares(cx0(1), ry, 16, None, None, read={2, 7, 9, 14})
    f.t(cx0(1), ry + 50, "只读挑出的 2,048 个", GR, True, 15)
    f.t(cx0(1), ry + 74, "打分器仍扫全部（头少、FP8）", GY, size=14)
    ye = v4_strips(cx0(2), ry, "read")
    f.t(cx0(2), ye + 18, "CSA 只挑一部分读，HCA 全读", INK, True, 15)
    f.t(cx0(3), ry + 30, "索引 K 也跨层共用", GY, size=14)

    f._pan = None
    yb = f.src(ytop + RH1 + 16 + RH2 + 10,
               "出处：MLA 每 token 70,272 B 见 3.1 表格（V3 config.json，(512＋64) × 61 层 × 2 B，本课推导）。"
               "DSA 每字读 2,048 个 token 见 DeepSeek-V3.2-Exp 技术报告 §2.1；打分器（lightning indexer）仍对全部历史打分、头少、可跑 FP8，同一报告 Inference Costs 一节。",
               "出处：CSA 每 4 个 token 压成一条、HCA 每 128 个压成一条，见 DeepSeek-V4 技术报告 §2.3；最近 128 个 token 的滑窗取自 Hugging Face DeepSeek-V4-Pro config.json（compress_ratios、sliding_window）。",
               "出处：V4.1-Flash 各层共用主 KV 和索引 K、主 KV 存 FP4、全局 KV 每 token 890 B、约 V4-Flash 的 1/4，见 Hugging Face deepseek-ai/DeepSeek-V4.1-Flash 模型卡。",
               "口径：方块个数和压缩块宽度是示意，不按比例；V4、V4-Flash 每 token 的字节数原文没给，图里不画；V4.1-Flash 读法的变化模型卡没写，那一格只记索引 K 共用。")
    f.save("fig6-kv-lineage.svg", yb + 14)


fig_lineage()
