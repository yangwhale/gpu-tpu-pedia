# -*- coding: utf-8 -*-
r"""专题五 · 第四节「第三刀：切专家」的两张静态图。

⛔ 数字现算：
   · 每个路由专家 3 × 7,168 × 2,048（gate / up / down 三块，hidden 7,168、moe_intermediate_size 2,048，
     取自 DeepSeek-V3 config.json）；256 个 × 58 个 MoE 层（61 层减去前 3 层稠密）。
   · 671B 是 V3 主模型总参数（技术报告口径，不含 MTP 模块）。
   · 并行折叠的例子取自 Megatron-Core MoE README：attention TP4·CP2·DP8·PP4，专家 ETP1·EP64·EDP1。

⛔ 刻意没画：fig-fold 里的共享专家、路由的 top-8（每个 token 实际挑 8 个专家），以及派发时的 FP8 量化。
"""
from topic03_draw import Fig, BL, OR, GR, RD, PU, CY, GY, INK, GY2, LINE

W = 1400
EXPERT = 3 * 7168 * 2048
ROUTED = EXPERT * 256 * 58
TOTAL = 671e9
REST = TOTAL - ROUTED
assert EXPERT == 44040192
assert abs(ROUTED / 1e9 - 653.9) < 0.1 and abs(ROUTED / TOTAL - 0.974) < 0.001
assert abs(REST / 1e9 - 17.1) < 0.1


def fig_params():
    f = Fig(W, "DeepSeek-V3 的 6710 亿参数都在哪。一根横条按比例切成两段：路由专家约 6539 亿，占百分之九十七；"
               "其余所有东西，注意力、共享专家、前三层稠密 MLP、词表，加起来只有约 171 亿，占百分之三。"
               "每个专家是三块 7168 乘 2048 的矩阵，约 4400 万参数，一共 256 个专家、58 个 MoE 层")
    y0 = f.header("V3 的参数几乎全在专家里　——　<tspan font-weight=\"700\">该切的是「专家」这一维</tspan>",
                  "按参数量画成一根横条。每个专家 ＝ 3 块 7,168 × 2,048 的矩阵（gate ／ up ／ down）",
                  [(OR, "路由专家"), (GY2, "其余：注意力、共享专家、稠密 MLP、词表")])
    PH = 190
    py = f.panel(0, y0, W, PH, "671B 按参数量摊开", OR)
    BX, BW = 40, 1320
    wr = BW * ROUTED / TOTAL
    f.box(BX, py + 40, wr, 56, OR, OR, 4)
    f.box(BX + wr, py + 40, BW - wr, 56, GY2, GY2, 4)
    f.t(BX + wr / 2, py + 76, "路由专家 ≈ %.0f 亿　（约 %.0f%%）" % (ROUTED / 1e8, ROUTED / TOTAL * 100),
        "#ffffff", True, 18, "middle")
    f.t(BX + BW - 6, py + 124, "其余 ≈ %.0f 亿（约 %.0f%%）↑" % (REST / 1e8, REST / TOTAL * 100), GY, True, 14, anchor="end")
    f.t(BX, py + 124, "一个专家 ≈ %.0f 万参数 × 256 个 × 58 层" % (EXPERT / 1e4), INK, True, 14)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "V3 训练时一点 TP 都没用", [
        "技术报告的理由是显存抠得够细，用不着代价高的 TP；专家只有 2,048 宽，TP 更不划算，而专家有 256 个，<tspan font-weight=\"700\">天然就是一维可以切的</tspan>。",
        "V3 的训练配置：16 路 PP ＋ 64 路专家并行 ＋ ZeRO-1 数据并行，<tspan font-weight=\"700\">不用 TP</tspan>（技术报告 sec. 3.2）。",
    ])
    yb = f.src(yb + 10, "📌 config.json：hidden_size 7,168、moe_intermediate_size 2,048、n_routed_experts 256、"
                        "num_hidden_layers 61、first_k_dense_replace 3。671B 为主模型总参数，不含 MTP 模块。"
                        "「其余」＝ 671B 减去路由专家，本脚本现算。")
    f.save("fig5-moe-params.svg", yb + 14)


def fig_fold():
    f = Fig(W, "同一批卡，两套切法。左边是注意力层眼里的这 8 张卡：按张量并行 4 路、数据并行 2 路来组。"
               "右边是专家层眼里的同样 8 张卡：按专家并行 8 路，每张卡放 32 个专家。"
               "注意力和专家的形状完全不同，所以一个模型里两者各配各的切法。训练里 Megatron 叫它并行折叠，"
               "推理里叫 DEP 和 TEP：前一个字母说注意力怎么切，后面的 EP 说专家怎么切")
    y0 = f.header("attention 和专家，各配各的切法　——　<tspan font-weight=\"700\">同一批卡，两套映射</tspan>",
                  "8 张卡的示意。同一张卡在注意力层里是某个 TP 组的一员，到了专家层就换一个身份",
                  [(BL, "第一组：管请求 A"), (GR, "第二组：管请求 B"), (OR, "一个小方块 ＝ 一个专家")])
    # ⭐ 2026-09-25 逐图审后重画（原判「写字的板子」）：卡里不再写「TP 组 A · 第 0 份」，
    #   左边画一小块 attention 矩阵、只点亮这张卡负责的那一条；右边画 32 个小点代表专家。
    PH = 330
    py = f.panel(0, y0, W, PH, "一层 Transformer 里，同一张卡换两次身份", BL)
    CWD, CHT = 130, 96
    f.t(40, py + 34, "注意力层：4 张卡一组切一份 attention，两组各管一批请求", INK, True, 15)
    for i in range(8):
        g, k = i // 4, i % 4
        x, y = 40 + k * 140, py + 56 + g * 124
        col = BL if g == 0 else GR
        f.box(x, y, CWD, CHT, "none", col, 8, sw=2)
        f.t(x + 8, y + 18, "卡 %d" % i, col, True, 13)
        for c in range(4):                      # 一块 attention 权重，竖着切 4 条，只点亮第 k 条
            fill = col if c == k else "#e8eaed"
            f.box(x + 22 + c * 22, y + 28, 20, 56, fill, fill, 2)
    f.t(600, py + 104, "请求 A", BL, True, 14)
    f.t(600, py + 228, "请求 B", GR, True, 14)
    f.t(662, py + 170, "→", INK, True, 30)
    f.t(676, py + 196, "AllToAll", GY, True, 12, "middle")
    f.t(710, py + 34, "专家层：8 张卡各放 32 个专家，所有请求的 token 都往这儿送", INK, True, 15)
    for i in range(8):
        x, y = 710 + (i % 4) * 165, py + 56 + (i // 4) * 124
        f.box(x, y, 150, CHT, "none", OR, 8, sw=2)
        f.t(x + 8, y + 18, "卡 %d" % i, OR, True, 13)
        for e in range(32):                     # 32 个专家，一个一个小方块
            ex, ey = x + 12 + (e % 8) * 16, y + 32 + (e // 8) * 14
            f.box(ex, ey, 11, 10, OR, OR, 2)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "attention 是一块大矩阵，专家是一堆小矩阵：各切各的", [
        "attention 那块大矩阵切成条，4 张卡一组合起来算；两组各管一批请求。专家本来就是 256 个小块，<tspan font-weight=\"700\">整个分出去就行</tspan>。",
        "训练里 Megatron 叫它 Parallel Folding；推理里 attention 用 TP 的叫 TEP，用数据并行的叫 DEP，图里这种一半一半的也有。",
    ])
    yb = f.src(yb + 10, "📌 Megatron-Core moe/README.md（MoE Parallel Folding，例子 TP4·CP2·DP8·PP4 → ETP1·EP64·EDP1）；"
                        "论文 arXiv 2504.14960。TEP ／ DEP 定义：TensorRT-LLM tech blog 26。")
    f.save("fig5-fold.svg", yb + 14)


# ── fig-ep-route：限制跨节点 ＋ FP8 派发（2026-09-25 现场讲课补） ──────────────
H_V3 = 7168
M_NODES, TOPK = 4, 8                         # 技术报告 sec. 4.2：M ＝ 4；每 token 选 8 个路由专家
NV_BW, IB_BW = 160, 50                       # sec. 3.2.2：NVLink 160 GB/s，IB 50 GB/s
V3_B = M_NODES * H_V3 * 1                    # FP8 每数 1 字节，跨机每节点只发一份
NAIVE_B = TOPK * H_V3 * 2                    # 不限节点、BF16：最坏 8 份
assert V3_B == 28672 and NAIVE_B == 114688 and NAIVE_B // V3_B == 4
assert abs(NV_BW / IB_BW - 3.2) < 1e-9 and round(M_NODES * NV_BW / IB_BW) == 13   # 4 × 3.2 ＝ 12.8，报告写 13


def fig_ep_route():
    """⭐ 课件 4.2 原来是两条 bullet ＋ 一段推导；「跨机只发一份、到了机器里再分」这件事画出来一眼就懂。"""
    f = Fig(W, "一个 token 选中 8 个专家时，派发要跨机发几份。左边不限节点、用 BF16：最坏 8 个专家在 8 台机器上，"
               "要跨机发 8 份，每份 14336 字节，约 115 KB。右边是 V3 的做法：最多只去 4 台机器，每台机器跨机只发一份，"
               "到了机器里再走 NVLink 转给真正持有专家的卡；派发还压成 FP8，每份 7168 字节，一共约 28.7 KB，是左边的四分之一")
    y0 = f.header("派发一个 token：跨机只发一份，到了机器里再分",
                  "V3 每个 token 选 8 个路由专家；每层的专家摊在 8 台机器、64 张卡上，一台机器 8 张卡",
                  [(BL, "token 出发的卡"), (OR, "持有被选中专家的卡"), (RD, "跨机（IB）"), (GR, "机器内（NVLink）")])
    PH, HW = 420, 680

    def node(x, y, hot, lab="", w=8 * 22 + 12):
        f.box(x, y, w, 34, "none", GY2, 6)
        for g in range(8):
            c = OR if g in hot else "none"
            f.box(x + 6 + g * 22, y + 8, 18, 18, c, OR if g in hot else LINE, 3)
        f.t(x + w + 10, y + 23, lab, GY, size=13)
        return w

    # 左：不限节点、BF16
    py = f.panel(0, y0, HW, PH, "不限节点、BF16：最坏跨机 8 份", RD)
    SX, SY = 30, py + 170
    node(SX, SY, [])
    f.box(SX + 6, SY + 8, 18, 18, BL, BL, 3)
    f.t(SX, SY - 12, "出发的机器", INK, True, 14)
    for i in range(8):
        ty = py + 44 + i * 40
        node(370, ty, [i % 8], "")
        f.line(SX + 200, SY + 17, 368, ty + 17, RD, 1.6)
    f.t(24, py + PH - 44, "8 份 × 14,336 字节 ≈ %.0f KB" % (NAIVE_B / 1e3), RD, True, 16)
    f._pan = None
    # 右：V3
    px = HW + 40
    py2 = f.panel(px, y0, HW, PH, "V3：最多 4 台机器、FP8：跨机 4 份", GR)
    SX2, SY2 = px + 30, py2 + 170
    node(SX2, SY2, [])
    f.box(SX2 + 6 + 3 * 22, SY2 + 8, 18, 18, BL, BL, 3)
    f.t(SX2, SY2 - 12, "出发的机器（卡 3）", INK, True, 14)
    HOT = [[0, 5], [2, 7], [1, 4], [6, 3]]
    for i, hot in enumerate(HOT):
        ty = py2 + 50 + i * 80
        nx = px + 360
        node(nx, ty, hot, "")
        gx = nx + 6 + 3 * 22 + 9                 # 同编号的卡 3：跨机先落在这里
        f.line(SX2 + 200, SY2 + 17, gx, ty - 2, RD, 2.4)
        for h in hot:
            if h == 3:
                continue
            hx = nx + 6 + h * 22 + 9
            f.path("M%d,%d Q%d,%d %d,%d" % (gx, ty + 34, (gx + hx) / 2, ty + 56, hx, ty + 36), GR, 1.8)
    f.t(px + 24, py2 + PH - 44, "4 份 × 7,168 字节 ≈ %.1f KB　（左边的 1/4）" % (V3_B / 1e3), GR, True, 16)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "为什么是「跨机少发、机器里多分」", [
        "NVLink 160 GB/s，约是跨机 IB 50 GB/s 的 3.2 倍：跨机送到的一份，在机器里平均转给 3.2 个专家也不添开销。",
        "所以报告说同样的通信量最多能选 13 个专家（4 台 × 3.2）；只跟去了几台机器有关，跟选了几个专家无关。",
    ])
    yb = f.src(yb + 10,
               "📌 DeepSeek-V3 技术报告 arXiv 2412.19437：sec. 2.1.2 限制节点路由（按每台机器上最高的 2 个亲和分之和挑机器）；sec. 3.2.2 先走 IB 到目标机器上同编号的卡、再走 NVLink 转发，NVLink 160 ／ IB 50 GB/s，13 个专家；sec. 3.3.3 派发 FP8、合并 BF16；sec. 4.2 M ＝ 4、64 卡 8 台。",
               "⚠️ 本课推导：28.7 KB ＝ 4 × 7,168 × 1 字节；115 KB ＝ 8 × 7,168 × 2 字节（不限节点、最坏每个专家在不同机器上）。图中被选中的是哪几张卡为示意。")
    f.save("fig5-ep-route.svg", yb + 14)


fig_params()
fig_fold()
fig_ep_route()
