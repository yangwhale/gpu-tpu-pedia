# -*- coding: utf-8 -*-
r"""专题六 · 第零节「2022 年以来」：从 2022 年的静态批到 2025 年的大 EP，推理一路补了哪些洞（fig6-story）；另附「两头一起变长」小图（fig6-twolong，见文件尾）。

⭐ 回答的问题：今天推理便宜，是哪几步走出来的？每一步解决什么问题、原文报了多大的提升？
   六条泳道 ＝ 六类问题；卡片按首次公开的年月落在时间轴上，右下角标回本课哪一节。
⛔ 数字：每张卡的倍数都取自原文摘要或正文，口径写在卡上（吞吐／延迟／达标请求数／显存）。
   出处逐条见 /tmp/t6-history.md 的整理，已逐条读原文核对；GQA 的 ÷8 是本课推导（64 头 ÷ 8 组）。
⛔ 刻意没画：把倍数连乘成一个总数 ——&#160;基线互相包含、口径不同，乘出来没有意义（见落点带）。
"""
from topic03_draw import Fig, BL, OR, GR, PU, CY, RD, GY, GY2, INK, LINE

W = 1400
Y0, Y1 = 2022.5, 2027.0
XL, XR = 196, W - 8
CW, CH = 232, 54

LANES = [
    (GY, "调度", "一批里谁进谁出"),
    (BL, "KV 与注意力", "多装几个人，读得更快"),
    (GR, "复用与下放", "算过的不再算"),
    (PU, "解码提速", "搬一趟出几个字"),
    (OR, "拆开放", "拆阶段（PD）／拆模型（EP）"),
    (CY, "少搬字节", "量化、合并 kernel"),
]

# 「拆开放」泳道的卡名带前缀：拆阶段 ＝ PD 分离，拆模型 ＝ 大 EP，两样都拆 ＝ 大 EP＋PD（R36 讲师反馈：两件事挤一条泳道）
# R41：DeepSeek day_6 原文「As we have adopted prefill-decode disaggregation architecture」，prefill EP32、decode EP144，故标两样都拆
# (泳道, 年份小数, 名字, 原文倍数与口径, 本课节[, 模型侧])
# ⛔ 卡上的倍数必须在它指向的那一节正文或图里找得到；找不到的只写做法，不写倍数（R5 起）。
CARDS = [
    (0, 2022.54, "Orca 连续批", "36.9×，吞吐 vs FasterTransformer", "四"),
    (0, 2024.20, "Sarathi-Serve 分块", "2.6–3.7×，达标容量 vs vLLM", "四"),
    (0, 2025.07, "vLLM V1 重写引擎", "吞吐最多 1.7×，对 V0", "四"),
    (1, 2023.38, "GQA", "KV ÷8（64 头 8 组，推导）", "三", True),
    (1, 2023.47, "PagedAttention", "KV 有效占用 20–38%→96%", "三"),
    (1, 2023.78, "Flash-Decoding", "解码最多 8×，对旧注意力", "五"),
    (1, 2024.37, "MLA（DeepSeek-V2）", "对同形状 MHA ÷57（推导）", "三", True),
    (1, 2025.74, "DSA（DeepSeek-V3.2）", "每字只读 2,048 条 KV", "三", True),
    (2, 2023.95, "RadixAttention", "吞吐最多 6.4×，对其他系统", "三"),
    (2, 2024.47, "Mooncake KV 池", "多接 115%／107%，对此前系统", "八"),
    (2, 2024.92, "SGLang KV 感知路由", "命中 20→75%，1.9×，对 v0.3", "十三"),
    (2, 2025.77, "llm-d 路由实测", "命中 340 ms，不路由最坏 2,850", "十三"),
    (3, 2022.87, "投机解码", "单请求 2–3×，对不投机", "十一"),
    (3, 2024.07, "EAGLE", "单请求 2.7–3.5×，对不投机", "十一"),
    (3, 2024.97, "DeepSeek-V3 MTP", "约 1.8×，对不开 MTP", "十一"),
    (3, 2025.21, "EAGLE-3", "单请求最多 6.5×，对不投机", "十一"),
    (3, 2026.53, "DSpark 半自回归", "一次出一小段，按置信度验", "十一"),
    (4, 2023.88, "拆阶段｜Splitwise", "同成本吞吐 2.35×，对不拆", "八"),
    (4, 2024.05, "拆阶段｜DistServe", "可接请求 7.4×，对不拆", "八"),
    (4, 2025.17, "两样都拆｜DeepSeek", "每节点 decode 14.8k tok/s", "二"),
    (4, 2025.34, "两样都拆｜SGLang 复现", "大 EP＋PD，对同卡 TP 最多 5×", "六"),
    (5, 2022.80, "GPTQ 4 bit", "3 bit 逐字 3.25×，对 5 卡 FP16", "十二"),
    (5, 2023.45, "AWQ 4 bit", "按激活护住重要通道", "十二"),
    (5, 2024.97, "FP8 权重与 KV", "V3 连训练都用 FP8", "十二"),
    (5, 2025.45, "NVFP4／MXFP4", "每数 4.25–4.5 bit", "十二"),
    (5, 2026.73, "TPU megakernel", "整次前向一个 kernel", "十二"),
]


# 上下文长度两个台阶：GPT-4 Turbo 2023-11、Gemini 1.5 Pro 预览 2024-02
CTX = [(2023.85, "上下文变长　128K"), (2024.12, "1M")]
# 输出长度：推理模型先想再答。o1 2024-09-12、DeepSeek-R1 2025-01-20（第零节正文「输出也变长」）
OUT = [(2024.70, "o1"), (2025.05, "R1　输出变长（推理模型）")]


def xof(yr):
    return XL + (yr - Y0) / (Y1 - Y0) * (XR - XL - CW)


def fig_story():
    f = Fig(W, "2022 年以来的推理优化：六条泳道按时间排出 2022 到 2026 年的关键优化，"
               "每张卡写原文报告的提升或做法：Orca 连续批 36.9 倍，PagedAttention 把 KV 有效占用从两到四成提到 96%，"
               "GQA 和 MLA 是模型侧的进步，用虚线卡标出；投机解码单请求是不投机的 2 到 3 倍，DistServe 能接的请求是不拆的 7.4 倍，拆开放泳道的卡名带前缀：Splitwise、DistServe 是「拆阶段」（PD 分离），DeepSeek 和 SGLang 的大 EP 都是「两样都拆」（大 EP 加 PD 分离），"
               "DeepSeek 大 EP decode 时每节点输出约 14,800 token 每秒；2024 末到 2025 年又添了 vLLM V1 吞吐最多 1.7 倍、SGLang KV 感知路由命中率 20% 到 75%、"
               "EAGLE-3 单请求最多 6.5 倍、DeepSeek-V3.2 的 DSA 每字只读 2,048 个历史 token；"
               "时间轴上标着两股压力：上下文 2023 年底到 128K、转年 2 月到 1M，输出随 o1（2024 年 9 月）和 R1（2025 年 1 月）这类推理模型变长；"
               "最下面一张实线卡是叠加：GB300 跑 V4-Pro 光换软件栈，同档每卡是首发日的 5.1 倍；倍数基线互相包含，不能连乘")
    y0 = f.header("2022 年以来　——　<tspan font-weight=\"700\">每一步补一个洞</tspan>",
                  "卡片落在首次公开的年月上；卡上是原文报告的提升或做法；右上角是本课讲它的那一节；虚线卡是模型侧的进步；顶上红、紫两条竖线是推着各泳道的两股压力：上下文变长、输出变长")
    # 时间轴
    ya = y0 + 6
    ticks = [(xof(Y0), "2022 年中")] + [(xof(yr), str(yr)) for yr in (2023, 2024, 2025, 2026)]
    for x, lab in ticks:
        f.t(x, ya + 10, lab, GY, True, 15, "middle")
    # 上下文长度：不是哪一招，是所有泳道共同的压力（第零节正文）
    (y128, l128), (y1m, l1m) = CTX
    f.t(xof(y128) + 6, ya + 32, l128 + " ▼", RD, True, 13, "end")
    f.t(xof(y1m) - 6, ya + 32, "▼ " + l1m, RD, True, 13)
    for yr_, lab in OUT:
        f.t(xof(yr_) - 6, ya + 32, "▼ " + lab, PU, True, 13)
    y = ya + 44
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
        for ri, x, (_, yr, nm, gain, sec, *model) in place:
            cy = y + 8 + ri * (CH + 8)
            f.box(x, cy, CW, CH, "#fff", col, 6, 1.4, dash="6 4" if model else None)
            f.t(x + 10, cy + 21, nm, INK, True, 15)
            f.t(x + 10, cy + 43, gain, col, False, 14)
            f.t(x + CW - 8, cy + 21, "第" + sec + "节", GY, False, 12, "end")
            if model:
                f.t(x + CW - 8, cy + 43, "模型侧", GY, True, 12, "end")
        y += h + 8
    # 网格线插到泳道底框之后、卡片之前会被泳道白底盖住，所以插在最前面并让泳道底透明
    grid = []
    for x, _ in ticks:
        f.line(x, ytop - 4, x, y - 8, LINE, 1, "4 4", arrow=False)
        grid.append(f.p.pop())
    for yr_, _ in CTX:                 # 上下文两个 ▼ 下面拉一根红虚线当刻度（同样垫在卡片底下）
        f.line(xof(yr_), ya + 36, xof(yr_), y - 8, RD, 1, "2 3", arrow=False)
        grid.append(f.p.pop())
    for yr_, _ in OUT:                 # 输出变长两个 ▼ 同理，紫色
        f.line(xof(yr_), ya + 36, xof(yr_), y - 8, PU, 1, "2 3", arrow=False)
        grid.append(f.p.pop())
    f.p[gi:gi] = grid
    # 叠加卡：实线粗框，跨在泳道外，表明它是合计，不和泳道卡相乘
    xs = xof(2025.0)
    f.box(xs, y, XR - xs, CH, "#fff", INK, 6, 2.2)
    f.t(xs + 12, y + 22, "叠起来：GB300 跑 V4-Pro，硬件模型都不换", INK, True, 15)
    f.t(xs + 12, y + 44, "软件栈合计，同一交互档每卡 5.1×（Day-0 → 2026-06）", INK, False, 14)
    f.t(XR - 8, y + 22, "第二、十四节", GY, False, 12, "end")
    y += CH + 8
    y = f.band(y + 6, "warn", "这些倍数不能连乘", [
        "后一个的基线已经吃掉了前一个的收益：Orca 比的是 FT，vLLM 比的是已有连续批的 Orca，Sarathi-Serve 比的是 vLLM。",
        "口径也不同：有的是吞吐，有的是单请求延迟，有的是达标请求数，有的只是显存。",
    ])
    y = f.src(y + 6, "Orca OSDI'22；vLLM arXiv 2309.06180 图 2；GQA arXiv 2305.13245（÷8 本课推导）；"
              "DeepSeek-V2 arXiv 2405.04434（论文对稠密 67B 报 KV −93.3%，混着 6 bit KV 等改动；卡上 ÷57 是第三节同形状 3,997,696 → 70,272 B，本课推导）；SGLang arXiv 2312.07104；"
              "Mooncake FAST'25（Kimi 线上 A800／H800）；llm-d：Red Hat Developer 2025-10-07（厂商自测）；"
              "Leviathan arXiv 2211.17192；Flash-Decoding 斯坦福 CRFM 2023-10；EAGLE arXiv 2401.15077；"
              "DeepSeek-V3 arXiv 2412.19437 §5.4.3；Splitwise arXiv 2311.18677；DistServe arXiv 2401.09670；"
              "DeepSeek open-infra-index Day 6；SGLang 大 EP 博客 2025-05；Sarathi-Serve arXiv 2403.02310；"
              "GPTQ arXiv 2210.17323；AWQ arXiv 2306.00978；NVFP4 NVIDIA 博客 2025-06；上下文台阶：OpenAI DevDay 2023-11-06、Google 2024-02-15；输出变长：OpenAI 2024-09-12、DeepSeek-R1 arXiv 2501.12948；vLLM V1 博客 2025-01-27；SGLang v0.4 博客 2024-12-04；EAGLE-3 arXiv 2503.01840；DSA：DeepSeek-V3.2-Exp 技术报告 §2.1；叠加 5.1×：PyTorch 博客 2026-06-23；DSpark arXiv 2607.05147；TPU megakernel：Inferact 博客 2026-09。")
    f.save("fig6-story.svg", y + 14)


fig_story()


# ── fig6-twolong.svg：0.1「输出也变长」旁边的小图 ─────────────────────
# ⭐ 回答：上下文变长和输出变长，各从哪一头抬高 decode 的成本？
# ⛔ 纯示意，没有一个实测数：格子数是随手取的，只为让三块面板同一比例可比。
#   按定义必然成立的是真算的：第 k 趟要读的 KV ＝ prompt 长度 ＋ 已出的字数（同 fig6-life）。
# ⛔ 刻意没画：权重（每趟都搬，但由 batch 里的人分摊，不随这一个请求的长短变）；
#   prefill 那一大块（长 prompt 也让它变贵，但那是算力账，这张图只讲 decode 搬字节）。
LIGHT = "#cfa9f5"                      # 已出字的 KV（浅紫）
CASES = [  # (标题, prompt 格数, decode 趟数)
    ("① 短 prompt、短回答", 4, 8),
    ("② prompt 变长：每一趟更重", 16, 8),
    ("③ 回答变长：趟数变多", 4, 28),
]
PITCH, BARW, UNIT = 22, 16, 5          # 每趟柱距、柱宽、每格 KV 的高度（px）


def kv_reads(P, N):
    r = [P + k for k in range(N)]     # 第 k 趟读的 KV ＝ prompt ＋ 前面已出的 k 个字
    assert r[0] == P and r[-1] == P + N - 1
    return r


def fig_twolong():
    f = Fig(W, "两头一起抬高 decode 的成本，示意图，不按比例。三块面板用同一比例画 decode 的每一趟："
               "柱子高度是这一趟要从显存读的 KV，深紫是 prompt 留下的，浅紫是已经写出的字，柱子总面积就是这一个请求 decode 一共要读的 KV。"
               "第一块是短 prompt、短回答。第二块 prompt 变长，趟数不变，但每根柱子底下的深紫一截都变厚，每一趟都更重。"
               "第三块回答变长，prompt 不变，柱子多了好几倍，而且越往后越高，因为写出的字也进了 KV。"
               "上下文变长把面积拉高，推理模型的长输出把面积拉长")
    y0 = f.header("两头一起变长　——　<tspan font-weight=\"700\">prompt 长让每趟更重，回答长让趟数更多</tspan>",
                  "示意，格子数随手取，三块同一比例。每根柱是 decode 的一趟，高度是这一趟要从显存读的 KV；柱子总面积 ＝ 这个请求一共要读的 KV",
                  [(PU, "prompt 留下的 KV"), (LIGHT, "已写出的字留下的 KV")])
    hmax = max(P + N - 1 for _, P, N in CASES) * UNIT
    CHART = hmax + 40                  # 柱区高度，顶上留一行注
    PH = 30 + 14 + CHART + 40
    xs = [0, 316, 632]
    ws = [300, 300, W - 632]
    base = kv_reads(*CASES[0][1:])
    for i, ((title, P, N), x, w) in enumerate(zip(CASES, xs, ws)):
        f.panel(x, y0, w, PH, title, PU if i else GY)
        ox, yb = x + 28, y0 + 30 + 14 + CHART
        reads = kv_reads(P, N)
        assert ox + N * PITCH < x + w - 12, "柱子出框"
        for k, r in enumerate(reads):
            bx = ox + k * PITCH
            f.box(bx, yb - P * UNIT, BARW, P * UNIT, PU, PU, 1)
            if r > P:
                f.box(bx, yb - r * UNIT, BARW, (r - P) * UNIT, LIGHT, LIGHT, 1)
        # 基线（面板 ①）的轮廓用黑虚线压在 ② ③ 的柱子上面，好比面积
        if i:
            pts = [(ox, yb)]
            for k, r in enumerate(base):
                pts += [(ox + k * PITCH, yb - r * UNIT), (ox + k * PITCH + BARW, yb - r * UNIT)]
            pts.append((ox + (len(base) - 1) * PITCH + BARW, yb))
            f.path(pts, INK, 1.6, "5 3", arrow=False)
        f.line(ox - 6, yb, x + w - 16, yb, GY2, 1, arrow=True)
        f.t(x + w - 18, yb + 22, "decode 第几趟", GY, False, 13, "end")
        top = y0 + 30 + 14 + 16
        if i == 0:
            f.t(ox, top, "每根柱：这一趟要读的 KV", GY, False, 13)
        elif i == 1:
            f.t(ox, top, "深紫变厚，每一趟都多读", PU, True, 14)
            f.t(ox, yb + 22, "黑虚线 ＝ ①", GY, False, 13)
        else:
            f.t(ox, top, "柱子多了好几倍；写出的字也进 KV，越往后越高", PU, True, 14)
            f.t(ox, yb + 22, "黑虚线 ＝ ①", GY, False, 13)
    y = y0 + PH + 10
    y = f.band(y, "ok", "上下文变长把面积拉高，长输出把面积拉长", [
        "推理模型走的就是 ③；长 prompt 再配上长回答，面积又高又长。",
    ])
    f.src(y + 6, "示意图，无实测数。第 k 趟读的 KV ＝ prompt 长度 ＋ 已出字数（按定义）。"
              "没画权重：每趟都搬，但由 batch 里的人分摊；也没画 prefill：长 prompt 让它更贵，那是算力账。")
    f.save("fig6-twolong.svg", y + 14)


fig_twolong()
