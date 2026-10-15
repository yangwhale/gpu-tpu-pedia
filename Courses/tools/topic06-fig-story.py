# -*- coding: utf-8 -*-
r"""专题六 · 第零节「2022 年以来」：从 2022 年的静态批到 2025 年的大 EP，推理一路补了哪些洞。

⭐ 回答的问题：今天推理便宜，是哪几步走出来的？每一步解决什么问题、原文报了多大的提升？
   六条泳道 ＝ 六类问题；卡片按首次公开的年月落在时间轴上，右下角标回本课哪一节。
⛔ 数字：每张卡的倍数都取自原文摘要或正文，口径写在卡上（吞吐／延迟／达标请求数／显存）。
   出处逐条见 /tmp/t6-history.md 的整理，已逐条读原文核对；GQA 的 ÷8 是本课推导（64 头 ÷ 8 组）。
⛔ 刻意没画：把倍数连乘成一个总数 ——&#160;基线互相包含、口径不同，乘出来没有意义（见落点带）。
"""
from topic03_draw import Fig, BL, OR, GR, PU, CY, RD, GY, INK, LINE

W = 1400
Y0, Y1 = 2022.5, 2026.0
XL, XR = 196, W - 8
CW, CH = 232, 54

LANES = [
    (GY, "调度", "一批里谁进谁出"),
    (BL, "KV 与注意力", "多装几个人，读得更快"),
    (GR, "复用与下放", "算过的不再算"),
    (PU, "解码提速", "搬一趟出几个字"),
    (OR, "拆开放", "prefill／decode／专家分家"),
    (CY, "少搬字节", "权重量化"),
]

# (泳道, 年份小数, 名字, 原文倍数与口径, 本课节)
CARDS = [
    (0, 2022.54, "Orca 连续批", "36.9×，同延迟吞吐 vs FT", "四"),
    (0, 2024.20, "Sarathi-Serve 分块", "2.6–5.6×，达标容量 vs vLLM", "四"),
    (1, 2023.38, "GQA", "KV ÷8（64 头 8 组，推导）", "三"),
    (1, 2023.47, "PagedAttention", "KV 有效占用 20–38%→96%", "三"),
    (1, 2023.78, "Flash-Decoding", "超长序列解码最多 8×", "五"),
    (1, 2024.37, "MLA（DeepSeek-V2）", "KV −93.3%，对稠密 67B", "三"),
    (2, 2023.95, "RadixAttention", "吞吐最多 6.4×", "三"),
    (2, 2024.47, "Mooncake KV 池", "真实负载多接 75% 请求", "八"),
    (2, 2025.70, "llm-d 前缀感知路由", "P90 首字快 57×，vs 近似调度", "十三"),
    (3, 2022.87, "投机解码", "2–3×，单请求延迟", "十一"),
    (3, 2024.07, "EAGLE", "2.7–3.5×，单请求延迟", "十一"),
    (3, 2024.97, "DeepSeek-V3 MTP", "第 2 个字接受 85–90%，1.8×", "十一"),
    (4, 2023.88, "Splitwise", "同成本吞吐 2.35×", "八"),
    (4, 2024.05, "DistServe", "7.4× 请求，或 SLO 紧 12.6×", "八"),
    (4, 2025.17, "DeepSeek 大 EP", "每节点 decode 14.8k tok/s", "六"),
    (4, 2025.34, "SGLang 大 EP 复现", "对同卡数 TP 最多 5×", "六"),
    (5, 2022.80, "GPTQ 4 bit", "3.25–4.5×，vs FP16", "十二"),
    (5, 2023.45, "AWQ 4 bit", "3× 以上，vs FP16", "十二"),
    (5, 2025.45, "NVFP4／MXFP4", "显存比 FP8 再省 1.8×", "十二"),
]


def xof(yr):
    return XL + (yr - Y0) / (Y1 - Y0) * (XR - XL - CW)


def fig_story():
    f = Fig(W, "2022 年以来的推理优化：六条泳道按时间排出 2022 到 2025 年的关键优化，"
               "每张卡写原文报告的提升和口径：Orca 连续批 36.9 倍，PagedAttention 把 KV 有效占用从两到四成提到 96%，"
               "投机解码 2 到 3 倍，DistServe 多服务 7.4 倍请求，Mooncake 真实负载多接 75%，"
               "DeepSeek 大 EP 每节点 decode 1.48 万 token 每秒；倍数基线互相包含，不能连乘")
    y0 = f.header("2022 年以来　——　<tspan font-weight=\"700\">每一步补一个洞</tspan>",
                  "卡片落在首次公开的年月上；卡上是原文报告的提升和比的是谁；右下角是本课讲它的那一节")
    # 时间轴
    ya = y0 + 6
    ticks = [(xof(Y0), "2022 年中")] + [(xof(yr), str(yr)) for yr in (2023, 2024, 2025, 2026)]
    for x, lab in ticks:
        f.t(x, ya + 10, lab, GY, True, 15, "middle")
    y = ya + 26
    ytop, gi = y, len(f.p)
    for li, (col, name, what) in enumerate(LANES):
        cards = [c for c in CARDS if c[0] == li]
        rows = []                      # 每个子行最右端
        place = []
        for c in sorted(cards, key=lambda c: c[1]):
            x = xof(c[1])
            for ri, right in enumerate(rows):
                if x >= right + 10:
                    rows[ri] = x + CW
                    place.append((ri, x, c))
                    break
            else:
                rows.append(x + CW)
                place.append((len(rows) - 1, x, c))
        h = len(rows) * (CH + 8) + 8
        f.box(0, y, W, h, "none", LINE, 6)
        f.box(0, y, 4, h, col, col, 0)
        f.t(16, y + h / 2 - 4, name, col, True, 16)
        f.t(16, y + h / 2 + 16, what, GY, False, 13)
        for ri, x, (_, yr, nm, gain, sec) in place:
            cy = y + 8 + ri * (CH + 8)
            f.box(x, cy, CW, CH, "#fff", col, 6, 1.4)
            f.t(x + 10, cy + 21, nm, INK, True, 15)
            f.t(x + 10, cy + 43, gain, col, False, 14)
            f.t(x + CW - 8, cy + 21, "第" + sec + "节", GY, False, 12, "end")
        y += h + 8
    # 网格线插到泳道底框之后、卡片之前会被泳道白底盖住，所以插在最前面并让泳道底透明
    grid = []
    for x, _ in ticks:
        f.line(x, ytop - 4, x, y - 8, LINE, 1, "4 4", arrow=False)
        grid.append(f.p.pop())
    f.p[gi:gi] = grid
    y = f.band(y + 6, "warn", "这些倍数不能连乘", [
        "后一个的基线已经吃掉了前一个的收益：Orca 比的是 FT，vLLM 比的是已有连续批的 Orca，Sarathi-Serve 比的是 vLLM。",
        "口径也不同：有的是吞吐，有的是单请求延迟，有的是达标请求数，有的只是显存。",
    ])
    y = f.src(y + 6, "Orca OSDI'22；vLLM arXiv 2309.06180 图 2；GQA arXiv 2305.13245（÷8 本课推导）；"
              "DeepSeek-V2 arXiv 2405.04434（−93.3% 同时含 MoE、FP8 权重、6 bit KV）；SGLang arXiv 2312.07104；"
              "Mooncake arXiv 2407.00079（模拟场景最多 +525%，真实负载 +75%）；llm-d 博客 2025-09（厂商自测）；"
              "Leviathan arXiv 2211.17192；Flash-Decoding 斯坦福 CRFM 2023-10；EAGLE arXiv 2401.15077；"
              "DeepSeek-V3 arXiv 2412.19437 §5.4.3；Splitwise arXiv 2311.18677；DistServe arXiv 2401.09670；"
              "DeepSeek open-infra-index Day 6；SGLang 大 EP 博客 2025-05；Sarathi-Serve arXiv 2403.02310；"
              "GPTQ arXiv 2210.17323；AWQ arXiv 2306.00978；NVFP4 NVIDIA 博客 2025-06。")
    f.save("fig6-story.svg", y + 14)


fig_story()
