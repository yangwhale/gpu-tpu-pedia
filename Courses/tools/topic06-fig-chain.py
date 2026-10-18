# -*- coding: utf-8 -*-
r"""专题六 · 第零节 0.1「问题链」（fig6-chain）。R85 首版，L80（2026-10-16）改成一条时间轴。

⭐ 回答的问题：2022 年到今天，推理的这十一招各是哪一年冒出来的？
   一条从左往右的轴，每环一个点，只写「年份 ＋ 招数名」。上方两条细带是外力（上下文变长、输出变长），
   轴右端一个菱形，是眼下最新的一环： GB300 × V4-Pro 的 5.1×。
⭐ L80 为什么重画：R85 版是 11 张卡片之字形排开，每张卡塞着问题句、倍数、红字新问题，
   再加两个外力框、一张最新一环的卡，共 13 个框。现场评语是「几十个框框，乱七八糟」，
   而且之字形看不出先后。
   ⭐ 改法：框数 13 → 0。卡里的问题句、倍数、留下的新问题全部挪到图下的折叠表（topic06_s0.py
     的 CHAIN_TABLE），一个字没删，只是不再默认摊在图上。
⛔ 排序：严格按这一招**首次公开**的时间（论文或博客），跨年的环取起点年、点下写「→ 末年」。
   所以投机解码、量化排在 2022（论文 2022 年底），不再按「问题什么时候卡脖子」放到链尾。
   L81 起环号跟着时间重编，从左到右 ①–⑪ 连续；0.1 图下表、15.3 表、讲稿、台账同步改过。
   旧→新：①→① ⑪→② ⑩→③ ③→④ ②→⑤ ⑦→⑥ ④→⑦ ⑥→⑧ ⑤→⑨ ⑨→⑩ ⑧→⑪。
   月份只用来排年内先后（Orca OSDI'22 七月、GPTQ 2022-10、投机解码 2022-11、GQA 2023-05、
   PagedAttention 2023-06、Splitwise 2023-11、RadixAttention 2023-12、Sarathi-Serve 2024-03、
   Mooncake 2024-06、Meta 上下文并行 2024-11、SGLang DP attention／DeepSeek-V3 大 EP 2024-12），
   **年内等距排，不按月比例**；图上只印年份。
⛔ 末年：④ DSA 2025-09；⑨ HiCache 2025-09；⑥ Dynamo 2025-03；⑪ SGLang 复现 2025-05；
   ⑩ vLLM 合入 DCP 2025-09（台账「L80 补充」）；③ EAGLE-3 2025-03；② megakernel、V4.1-Flash FP4 KV 2026-09。
⛔ 刻意没画：倍数（连乘会出错，基线互相包含）；每环之间的红箭头（时间顺序和「解完又冒出什么」的因果顺序不一致，
   硬连会在轴上来回折返；链的先后写在折叠表里）。
"""
from topic03_draw import Fig, OR, PU, GY, INK, LINE, wpx, MONO_K

W = 1400
X0 = 24                       # 轴的左端
SLOT = 90                     # 年内相邻两点的间距（上下交替，同侧相距两格）
PAD = 16                      # 每年两头的留白

# (环号, 年, 月（只排序）, 招数名, 末年或 None, 模型侧)
RINGS = [
    (1, 2022, 7, "Orca 连续批", None, False),
    (2, 2022, 10, "GPTQ 量化", 2026, False),
    (3, 2022, 11, "投机解码", 2025, False),
    (4, 2023, 5, "GQA 压薄 KV", 2025, True),
    (5, 2023, 6, "PagedAttention", None, False),
    (6, 2023, 11, "PD 分离", 2025, False),
    (7, 2023, 12, "RadixAttention", None, False),
    (8, 2024, 3, "分块 prefill", None, False),
    (9, 2024, 6, "KV 分层下放", 2025, False),
    (10, 2024, 11, "切序列 CP", 2025, False),
    (11, 2024, 12, "MoE 大 EP", 2025, False),
]
assert [r[0] for r in RINGS] == list(range(1, 12)), "环号要从左到右连续"
assert [(r[1], r[2]) for r in RINGS] == sorted((r[1], r[2]) for r in RINGS), "没按首次公开的时间排"
YEARS = [2022, 2023, 2024, 2025, 2026]
END = (2026, 6)               # 眼下最新一环：GB300 × V4-Pro
# 外力：(颜色, 名字, [(年, 月, 刻度字)])
FORCES = [
    (OR, "上下文变长", [(2023, 11, "128K"), (2024, 2, "1M")]),
    (PU, "输出变长", [(2024, 9, "o1"), (2025, 1, "R1")]),
]
CIRC = "①②③④⑤⑥⑦⑧⑨⑩⑪"


def seg_layout():
    """每年一段：段宽 ＝ 点数 × SLOT ＋ 两头留白；没有新点的年份给一格。返回 {年: (x0, x1, [点 x])}。"""
    out, x = {}, X0
    for y in YEARS:
        n = sum(1 for r in RINGS if r[1] == y) + (1 if y == END[0] else 0)
        w = max(n, 1) * SLOT + 2 * PAD
        xs = [x + PAD + SLOT * (k + 0.5) for k in range(n)]
        out[y] = (x, x + w, xs)
        x += w
    return out


SEG = seg_layout()
X1 = SEG[YEARS[-1]][1]
assert X1 <= W - 4, "轴太长：%d" % X1


def xdate(y, m):
    """年月 → x：年内按已知点分段线性插值（只为把外力刻度放进对的先后位置，不是按月比例）。"""
    x0, x1, xs = SEG[y]
    ms = [r[2] for r in RINGS if r[1] == y] + ([END[1]] if y == END[0] else [])
    knots = [(0, x0)] + list(zip(ms, xs)) + [(12, x1)]
    for (ma, xa), (mb, xb) in zip(knots, knots[1:]):
        if ma <= m <= mb:
            return xa if mb == ma else xa + (xb - xa) * (m - ma) / (mb - ma)
    raise ValueError((y, m))


def fig_chain():
    f = Fig(W, "推理的十一招放在一条时间轴上，从左往右是 2022 到 2026 年。"
               "2022 年：① Orca 连续批，② GPTQ 量化（一直做到 2026），③ 投机解码（到 2025）。"
               "2023 年：④ GQA 压薄 KV（模型侧，到 2025），⑤ PagedAttention，⑥ PD 分离（到 2025），⑦ RadixAttention。"
               "2024 年：⑧ 分块 prefill，⑨ KV 分层下放，⑩ 切序列（到 2025），⑪ MoE 大 EP（到 2025）。"
               "2025 年没有新起头的环，都是前面几环在往下做。"
               "上方两条细带是外力：上下文变长从 2023 年 11 月的 128K 起，2024 年 2 月到 1M；"
               "输出变长从 2024 年 9 月的 o1 起，2025 年 1 月有 R1。"
               "轴的右端停在 2026 年 6 月，是眼下最新的一环：GB300 跑 V4-Pro，光改软件涨了几倍，标成问号，答案在最后一节。")
    y0 = f.header("问题链　——　<tspan font-weight=\"700\">十一招各是哪年冒出来的</tspan>",
                  "点落在首次公开的那一年，年内只排先后、不按月比例；点下的「→」是一直做到哪年；圈里的数是环号，从左到右按先后编",
                  [(INK, "系统侧"), (GY, "空心：模型侧"), (OR, "外力：上下文变长"), (PU, "外力：输出变长")])
    # ── 年份列头 ＋ 列间细线 ──
    yh = y0 + 16
    for y in YEARS:
        x0, x1, _ = SEG[y]
        f.t((x0 + x1) / 2, yh, str(y), GY, True, 16, "middle")
    # ── 外力细带 ──
    yf = [yh + 34, yh + 64]
    for (col, name, ticks), yy in zip(FORCES, yf):
        xs = xdate(*ticks[0][:2])
        assert xs - 10 - wpx(name, 13) * MONO_K >= X0, "外力名字出左边界：%s" % name
        f.t(xs - 10, yy + 5, name, col, True, 13, "end")
        f.line(xs, yy, X1, yy, col, 3, arrow=False)
        for ty, tm, lab in ticks:
            xt = xdate(ty, tm)
            f.line(xt, yy - 6, xt, yy + 6, col, 2, arrow=False)
            f.t(xt + 5, yy - 7, lab, col, False, 12)
    # ── 轴 ──
    ya = yf[-1] + 92
    ytop, ybot = yh + 12, ya + 92
    for y in YEARS[1:]:
        x = SEG[y][0]
        f.line(x, ytop, x, ybot, LINE, 1, "3 4", arrow=False)
    f.line(X0, ya, X1, ya, GY, 2, arrow=False)
    # ── 点 ＋ 标签（上下交替）──
    boxes = []
    for k, (n, yr, mo, name, end, model) in enumerate(RINGS):
        x = SEG[yr][2][[r for r in RINGS if r[1] == yr].index(RINGS[k])]
        up = k % 2 == 0
        yr_s = "%d" % yr + ("　→%d" % end if end else "")
        if up:
            yy1, yy2 = ya - 50, ya - 30          # 年份行、名字行
            f.line(x, ya - 22, x, ya - 12, LINE, 1, arrow=False)
        else:
            yy1, yy2 = ya + 36, ya + 56
            f.line(x, ya + 12, x, ya + 22, LINE, 1, arrow=False)
        wname = wpx(name, 14) * MONO_K * 1.08
        wyr = wpx(yr_s, 12) * MONO_K
        wl = max(wname, wyr)
        boxes.append((x - wl / 2, x + wl / 2, up, name))
        f.t(x, yy1, yr_s, GY, False, 12, "middle")
        f.t(x, yy2, name, INK if not model else GY, True, 14, "middle")
        if model:
            f.p.append('<circle cx="%.1f" cy="%.1f" r="10" fill="#fff" stroke="%s" stroke-width="1.5" '
                       'stroke-dasharray="3 2"/>' % (x, ya, GY))
            f.t(x, ya + 4, CIRC[n - 1], GY, True, 12, "middle")
        else:
            f.p.append('<circle cx="%.1f" cy="%.1f" r="10" fill="%s"/>' % (x, ya, INK))
            f.t(x, ya + 4, CIRC[n - 1], "#fff", True, 12, "middle")
    # 同侧标签不许撞，也不许出边界
    for side in (True, False):
        row = sorted(b for b in boxes if b[2] == side)
        for a, b in zip(row, row[1:]):
            assert a[1] + 8 <= b[0], "标签撞车：%s / %s" % (a[3], b[3])
        for b in row:
            assert b[0] >= 0 and b[1] <= W, "标签出边界：%s" % b[3]
    # ── 眼下最新一环：菱形 ──
    xe = SEG[END[0]][2][-1]
    d = 13
    f.poly([(xe, ya - d), (xe + d, ya), (xe, ya + d), (xe - d, ya)], INK, INK)
    f.line(xe, ya + 16, xe, ya + 22, LINE, 1, arrow=False)
    f.t(xe, ya + 36, "2026-06　截至目前", GY, False, 12, "middle")
    f.t(xe, ya + 56, "GB300 × V4-Pro", INK, True, 14, "middle")
    f.t(xe, ya + 76, "涨了几倍？", INK, True, 15, "middle")
    for b in boxes:
        if not b[2]:
            assert b[1] + 8 <= xe - 70 or b[0] >= xe + 70, "右端标签撞车：%s" % b[3]
    f.src(ybot, "首次公开：Orca OSDI'22；GPTQ arXiv 2210.17323；投机解码 Leviathan 等 arXiv 2211.17192；"
                "GQA arXiv 2305.13245；PagedAttention 博客 2023-06-20；Splitwise arXiv 2311.18677；"
                "RadixAttention arXiv 2312.07104；Sarathi-Serve arXiv 2403.02310；Mooncake arXiv 2407.00079；"
                "Meta 上下文并行 arXiv 2411.01783；SGLang v0.4 博客 2024-12-04（DP attention）、DeepSeek-V3 arXiv 2412.19437。"
                "末年：DeepSeek-V3.2-Exp 技术报告（DSA，2025-09）；SGLang HiCache 博客 2025-09-10；NVIDIA Dynamo 新闻稿 2025-03-18；"
                "SGLang 大 EP 博客 2025-05-05；vLLM v0.10.2 release（DCP，2025-09）；EAGLE-3 arXiv 2503.01840；"
                "Inferact TPU megakernel 博客 2026-09、vLLM DeepSeek-V4.1-Flash 博客 2026-10-07（FP4 KV）。"
                "外力：OpenAI DevDay 2023-11-06、Google 2024-02-15；OpenAI 2024-09-12、DeepSeek-R1 arXiv 2501.12948。"
                "右端菱形：PyTorch 博客 2026-06-23，倍数留到最后一节揭晓。年内等距排，只表先后。")
    f.save("fig6-chain.svg", ybot + 4)


fig_chain()
