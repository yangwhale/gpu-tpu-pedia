# -*- coding: utf-8 -*-
r"""专题六 · 第零节 0.1「问题链」（fig6-chain）。R85（2026-10-16）。

⭐ 回答的问题：2022 年到今天，推理是怎么一步步变便宜的？每一招解决了什么，又留下了什么？
   11 环之字形排开：每环灰字是「年份 · 首发者」，黑字是「问题 → 解法」和课件里已有的倍数，
   红字「↳」是这一招留下的新问题，红箭头接下一环。右侧两股外力：上下文变长接第 2 环，
   输出变长（o1／R1）接第 10 环。终点是 GB300 × V4-Pro 只改软件的 5.1×。
⛔ 数字：每个倍数都取自课件正文（节号标在卡右上角）和第十五节台账，这里不引入新数。
   ÷57 是第三节的本课推导；FP4「推测只能存不能算」照抄第十二节的措辞，不说死。
⛔ 排序按「问题什么时候卡脖子」，不按论文首发年：投机解码 2022 年底就有，放在第 10 环。
⛔ 刻意没画：把倍数连乘成一个总数 —— 基线互相包含（Orca 比 FT，vLLM 比 Orca，
   Sarathi-Serve 比 vLLM），口径也不同（吞吐／单请求／达标请求数）。
⛔ 第 9 环的灰字不写首发者：DCP／PCP 谁先进的引擎，本课没核过出处。
"""
from topic03_draw import Fig, OR, PU, RD, GY, INK, LINE, wrap_rich, MONO_K

W = 1400
NW, NH = 515, 124          # 节点宽、高
GX = 56                    # 同一行两节点的间距
XL = 0
XR2 = XL + NW + GX         # 右列 x
FX = XR2 + NW + 40         # 外力框 x
FW = W - FX - 2
RG = 52                    # 行距（放下行间红箭头）

# (灰字, 节, 黑字「问题 → 解法」, 倍数行, 红字新问题, 模型侧)
CHAIN = [
    ("2022 · Orca（论文）", "四",
     "一批人等最长的那个写完 → 连续批",
     "吞吐 36.9×（同延迟，对 FasterTransformer）",
     "座位由 KV 定：按最大长度预留，只用上两到近四成", False),
    ("2023-06 · vLLM", "三",
     "KV 预留浪费 → PagedAttention 分页",
     "KV 利用率到 96%，吞吐是 FT／Orca 的 2–4×",
     "上限变成 KV 池 ÷ 每条 KV，上下文偏偏在变长", False),
    ("2023–25 · 模型侧：GQA、DeepSeek", "三",
     "每条 KV 太厚 → GQA、MLA，再到 DSA",
     "KV ÷8（GQA）、÷57（MLA，推导）；DSA 每字只读 2,048 条",
     "MLA 只有一份共享 KV，TP 切不开，开几路存几份", True),
    ("2023-12 · SGLang（论文）", "三",
     "相同开头每次重算 → RadixAttention",
     "最多 6.4×（对当时其他系统）",
     "显存留不下所有想复用的前缀", False),
    ("2024–25 · Mooncake（Kimi）、DeepSeek", "三、八",
     "前缀存不下 → 往内存、SSD 分层下放",
     "Mooncake KV 池：Kimi 线上多接 107%–115%",
     "副本一多，前缀只在本副本有效（KV 感知路由把命中率从 20% 拉到 75%）", False),
    ("2024 · Sarathi-Serve（论文），引擎里 vLLM 先落地", "四",
     "长 prompt 进门卡住所有人 → 分块 prefill",
     "达标请求数 2.6–3.7×（对 vLLM）",
     "只摊平了打扰，两段仍挤在同一批卡上", False),
    ("2023–25 · Splitwise、DistServe → Mooncake → Dynamo", "八",
     "prefill、decode 抢同一批卡 → PD 分离",
     "DistServe：可接请求是不拆的 7.4×",
     "每个请求多一趟 KV 传输，外加一份 P:D 配比要调", False),
    ("2024-12–25 · SGLang、DeepSeek", "六",
     "MoE 把 batch 拆散（每专家约 8 行）→ DP attention＋大 EP",
     "SGLang 复现：输出最多是同卡数纯 TP 的 5×",
     "每个 rank 都得带一份状态，K3 开 DEP8 就显存不足；prefill 也跟不上变快的 decode", False),
    ("2025–26", "七",
     "起不了 DEP、KV 被复制 → DCP／PCP 切序列",
     "DCP 让出 KV 副本，PCP 切 prompt 压首字延迟",
     "DCP 只在长上下文赚；PCP 碰上线性注意力，状态要接力", False),
    ("2024-09–25 · o1／R1 推着走；DeepSeek MTP、EAGLE-3", "十一",
     "一题几千个思考 token → 投机解码／MTP",
     "MTP 约 1.8×；EAGLE-3 单请求最多 6.5×（都对不投机）",
     "并发大、上下文长、开 DEP，收益就缩水，甚至倒亏", False),
    ("2022→26 · 量化、megakernel", "十二",
     "大 batch 下只剩字节可省 → FP8／FP4、合 kernel",
     "Llama 3 70B 用 FP8 权重，每字间隔最好时只剩 BF16 的 1/2",
     "v7 上 FP4 推测只能存不能算；INT4 漏了「减 8」就出乱词", False),
]
assert len(CHAIN) == 11

# 外力：(接第几环, 颜色, 标题, 细节行)
FORCES = [
    (2, OR, "上下文变长", ["2023-11　128K（GPT-4 Turbo）", "2024-02　1M（Gemini 1.5 Pro）", "长了，座位就少"]),
    (10, PU, "输出变长", ["2024-09　o1", "2025-01　DeepSeek-R1", "一题几百到几千个思考 token"]),
]


def slot(i):
    """第 i 环（0 起）→ (行, 是否右列)。之字形：偶数行左→右，奇数行右→左。"""
    r, k = divmod(i, 2)
    right = (k == 1) if r % 2 == 0 else (k == 0)
    return r, right


def fig_chain():
    f = Fig(W, "推理的问题链，2022 到 2026 年共 11 环，之字形排开。每一环写年份和首发者、问题和解法、课件里的倍数，"
               "红字是这一招留下的新问题，红箭头接到下一环。"
               "1 Orca 连续批，同延迟吞吐是 FasterTransformer 的 36.9 倍，留下：座位由 KV 定，只用上两到近四成。"
               "2 vLLM 的 PagedAttention，KV 利用率到 96%，留下：上限是 KV 池除以每条 KV，上下文又在变长。"
               "3 模型侧的 GQA、MLA、DSA 把每条 KV 压薄，留下：MLA 的共享 KV 用 TP 切不开。"
               "4 SGLang 的 RadixAttention 复用相同开头，吞吐最多 6.4 倍，留下：显存放不下想复用的前缀。"
               "5 Mooncake 和 DeepSeek 往内存、SSD 下放，Kimi 多接 107% 到 115%，留下：前缀只在本副本有效。"
               "6 Sarathi-Serve 分块 prefill，可接请求是 vLLM 的 2.6 到 3.7 倍，留下：两段仍挤在同一批卡上。"
               "7 PD 分离，DistServe 可接请求是不拆的 7.4 倍，留下：多一趟 KV 传输和一份配比。"
               "8 DP attention 加大 EP，最多是纯 TP 的 5 倍，留下：每个 rank 自备状态，prefill 喂不上。"
               "9 DCP、PCP 切序列，留下：只在长上下文赚，线性注意力状态要接力。"
               "10 投机解码和 MTP，MTP 约 1.8 倍，EAGLE-3 最多 6.5 倍，留下：并发大时缩水甚至倒亏。"
               "11 FP8、FP4 和 megakernel 少搬字节。"
               "右侧两股外力：上下文变长接第 2 环，输出变长接第 10 环。"
               "终点：GB300 跑 V4-Pro，只改软件，每卡是首发日的 5.1 倍；这些倍数基线互相包含，不能连乘")
    y0 = f.header("问题链　——　<tspan font-weight=\"700\">2022 到 2026 的十一环</tspan>",
                  "按问题什么时候卡脖子排，不按论文首发年；灰字是年份和首发者，右上角是讲它的那一节；虚线框是模型侧；右边两股外力从侧面推进来",
                  [(RD, "↳ 留下的新问题 → 下一环"), (OR, "上下文变长"), (PU, "输出变长")])
    y0 += 6
    pos = []
    for i, (who, sec, ps, num, nxt, model) in enumerate(CHAIN):
        r, right = slot(i)
        x = XR2 if right else XL
        y = y0 + r * (NH + RG)
        pos.append((x, y))
        f.box(x, y, NW, NH, "#fff", INK if not model else GY, 8, 1.3, dash="6 4" if model else None)
        f.t(x + 14, y + 22, "%d　%s" % (i + 1, who), GY, False, 13, w=NW - 90)
        f.t(x + NW - 12, y + 22, "第" + sec + "节", GY, False, 12, "end")
        f.t(x + 14, y + 48, ps, INK, True, 15, w=NW - 24)
        f.t(x + 14, y + 72, num, INK, False, 14, w=NW - 24)
        rows = wrap_rich("↳ " + nxt, NW - 30, 14 * MONO_K)
        assert len(rows) <= 2, "第 %d 环红字超过两行" % (i + 1)
        for k, row in enumerate(rows):
            f.t(x + 14 + (0 if k == 0 else 20), y + 96 + k * 19, row, RD, False, 14)
        if model:
            f.t(x + NW - 12, y + 48, "模型侧", GY, True, 12, "end")
    # 红箭头：环与环之间
    for i in range(len(CHAIN) - 1):
        (x1, y1), (x2, y2) = pos[i], pos[i + 1]
        if y1 == y2:                                   # 同一行：横向
            if x2 > x1:
                f.line(x1 + NW + 4, y1 + NH / 2, x2 - 4, y2 + NH / 2, RD, 2.2)
            else:
                f.line(x1 - 4, y1 + NH / 2, x2 + NW + 4, y2 + NH / 2, RD, 2.2)
        else:                                          # 换行：竖直向下
            xc = x1 + NW / 2
            f.line(xc, y1 + NH + 4, xc, y2 - 4, RD, 2.2)
    # 终点：接在第 11 环后面（第 6 行另一侧）
    xe, ye = pos[-1]
    r, right = slot(len(CHAIN))
    xs = XR2 if right else XL
    ys = y0 + r * (NH + RG)
    f.box(xs, ys, NW, NH, "#fff", INK, 8, 2.6)
    f.t(xs + 14, ys + 26, "叠起来：GB300 × V4-Pro（2026-06）", INK, True, 16)
    f.t(xs + 14, ys + 52, "硬件、模型、交互档都不变，只改软件", INK, False, 14)
    f.t(xs + 14, ys + 78, "每卡是首发日（Day-0）的 5.1×（含 MTP）", INK, True, 15)
    f.t(xs + 14, ys + 106, "上面的倍数基线互相包含，不能连乘", GY, False, 13)
    f.t(xs + NW - 12, ys + 26, "第二、十四节", GY, False, 12, "end")
    if xs < xe:
        f.line(xe - 4, ye + NH / 2, xs + NW + 4, ys + NH / 2, INK, 2.2)
    else:
        f.line(xe + NW + 4, ye + NH / 2, xs - 4, ys + NH / 2, INK, 2.2)
    # 外力：右侧框，粗箭头指向对应那一环（都在右列）
    for ring, col, title, rows in FORCES:
        x, y = pos[ring - 1]
        assert x == XR2, "外力要接在右列的环上（第 %d 环不在右列）" % ring
        fh = 40 + 22 * len(rows)
        fy = y + NH / 2 - fh / 2
        f.box(FX, fy, FW, fh, "#fff", col, 8, 1.6)
        f.box(FX, fy, 4, fh, col, col, 0)
        f.t(FX + 14, fy + 24, title, col, True, 15)
        for k, row in enumerate(rows):
            f.t(FX + 14, fy + 48 + 22 * k, row, INK if k < len(rows) - 1 else GY, False, 13, w=FW - 20)
        f.line(FX - 4, y + NH / 2, XR2 + NW + 6, y + NH / 2, col, 3.2)
    y = ys + NH + 10
    f.src(y, "节点数字均取自本课正文与第十五节台账：Orca OSDI'22；vLLM arXiv 2309.06180；GQA arXiv 2305.13245（÷8 本课推导）；"
             "MLA ÷57 为第三节本课推导（同形状 3,997,696 → 70,272 B）；DSA：DeepSeek-V3.2-Exp 技术报告 §2.1；"
             "SGLang arXiv 2312.07104；Mooncake FAST'25；SGLang v0.4 博客 2024-12-04（命中率 20% → 75%）；"
             "Sarathi-Serve arXiv 2403.02310；DistServe arXiv 2401.09670；SGLang 大 EP 博客 2025-05-05；"
             "DeepSeek-V3 arXiv 2412.19437（MTP 约 1.8 倍）；EAGLE-3 arXiv 2503.01840；"
             "vLLM FP8：Red Hat Developer 2024-07-15；v7 FP4 只能存不能算是第十二节的推测（公开规格只列 BF16、FP8 峰值）；"
             "上下文台阶：OpenAI DevDay 2023-11-06、Google 2024-02-15；输出变长：OpenAI 2024-09-12、DeepSeek-R1 arXiv 2501.12948；"
             "叠加 5.1×：PyTorch 博客 2026-06-23。排序按问题卡脖子的时间，不按首发年。")
    f.save("fig6-chain.svg", y + 4)


fig_chain()
