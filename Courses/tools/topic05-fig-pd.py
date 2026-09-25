# -*- coding: utf-8 -*-
r"""专题五 · 第六节「第五刀：不切张量，切工作」的两张静态图。

出处：
  · prefill 吃算力、decode 吃带宽，放一起互相干扰：DistServe（arXiv 2401.09670）。
  · 我们自己的 TPU v7x PD 分离实测（wiki qwen3-coder-480b-pd-disagg-tpuv7x-20260425）：
    KV 三段传输按带宽估算约 100 ms（不是计时实测），8K prompt 的 KV 约 1 GB（FP8），占 1–2 s prefill 的 5–10%。
  · AFD 的 M2N ／ N2M 与乒乓 micro-batch：MegaScale-Infer（arXiv 2504.02263），每 GPU 吞吐最高 1.90 倍。
⛔ 时间线是示意（格子长度按 prefill 比 decode 一步长很多来画），不是实测时序；图上也这样标。
"""
from topic03_draw import Fig, BL, OR, GR, RD, PU, CY, GY, INK, GY2, LINE

W = 1400
# Qwen3-Coder-480B（config.json：62 层、8 个 KV 头、head_dim 128），8K prompt、FP8 每元素 1 字节
KV_8K = 2 * 62 * 8 * 128 * 8192
DCN = 100e9 / 8                          # 100 Gbps 以太网 ＝ 12.5 GB/s
T_DCN = KV_8K / DCN
assert abs(KV_8K / 1e9 - 1.04) < 0.01 and abs(T_DCN * 1e3 - 83) < 1


def fig_pd():
    f = Fig(W, "为什么要把 prefill 和 decode 分开。上面是放在同一批卡上：大家都在一步一步 decode，"
               "突然来了一个长 prompt，它的 prefill 要占好几步的时间，这几步里所有人的 decode 都停住了，出字速度掉下去。"
               "下面是分开：prefill 机器专门吞 prompt，算完把 KV cache 传给 decode 机器；decode 机器一直稳定地出字，"
               "代价是多了一趟 KV 传输。按我们 TPU v7x 那套的带宽估算，这一趟约 100 毫秒，占一次 prefill 的百分之五到十")
    y0 = f.header("第五刀：把两种活拆到两批机器上　——　<tspan font-weight=\"700\">decode 不再被 prefill 卡住</tspan>",
                  "示意时间线（不是实测时序）。每格是一步；绿色的 decode 一格一格往外出字，橙色是一次长 prompt 的 prefill",
                  [(GR, "decode：每步出一个字"), (OR, "prefill：一口吞下 prompt"), (BL, "KV cache 传输")])
    PH = 330
    py = f.panel(0, y0, W, PH, "同一批卡 vs 分开两批", OR)
    X0, CW = 230, 48
    N, AT, LN = 16, 5, 5                 # 跟 PD 动画同一套：16 步、第 5 步起 prefill 占 5 步
    # 同一批卡
    f.t(24, py + 62, "放在一起", INK, True, 16)
    x = X0
    seq = ["d"] * AT + ["P"] + ["d"] * (N - AT - LN)
    for s in seq:
        if s == "d":
            f.box(x, py + 40, CW - 4, 36, GR, GR, 3)
            x += CW
        else:
            f.box(x, py + 40, CW * LN - 4, 36, OR, OR, 3)
            f.t(x + CW * LN / 2, py + 64, "长 prompt 的 prefill", "#ffffff", True, 14, "middle")
            f.t(x + CW * LN / 2, py + 100, "这 %d 步所有人的 decode 都停住" % LN, RD, True, 13.5, "middle")
            x += CW * LN
    # 分开
    f.t(24, py + 182, "prefill 机器", INK, True, 16)
    f.t(24, py + 262, "decode 机器", INK, True, 16)
    f.box(X0 + AT * CW, py + 160, CW * LN - 4, 36, OR, OR, 3)
    f.t(X0 + (AT + LN / 2) * CW, py + 184, "prefill", "#ffffff", True, 14, "middle")
    f.path("M%d,%d L%d,%d" % (X0 + (AT + LN) * CW, py + 198, X0 + (AT + LN + 1) * CW, py + 238), BL, 2.2)
    f.t(X0 + (AT + LN + 1) * CW + 10, py + 222, "KV cache 传过去（按带宽估算约 100 ms）", BL, True, 13)
    x = X0
    for i in range(N):
        f.box(x, py + 240, CW - 4, 36, GR, GR, 3)
        x += CW
    f.t(X0 + N * CW + 12, py + 264, "一直稳定地出字", GR, True, 14)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "拆开换来的：出字不再被打断", [
        "放在一起时，一个长 prompt 的 prefill 就能让所有人停几步。<tspan font-weight=\"700\">拆开后 decode 那条线一格不少。</tspan>",
        "代价只是多一趟 KV 传输：KV 从 prefill 机器的显存走到 decode 机器的显存。",
    ])
    yb = f.src(yb + 10, "📌 DistServe arXiv 2401.09670：prefill 偏算力、decode 受显存带宽约束；拆开后多服务 7.4 倍请求或把 SLO 收紧 12.6 倍。",
                        "📌 本课程作者 TPU v7x 1P1D 部署（Qwen3-Coder-480B）：KV 三段传输按带宽估算约 100 ms，占 1–2 s prefill 的 5–10%%。"
                        "8K prompt 的 KV ≈ %.2f GB（FP8，按 config 现算），过 100 Gbps 网络约 %.0f ms" % (KV_8K / 1e9, T_DCN * 1e3) + "；长 prompt 宜 2P:1D，长输出宜 1P:2D。")
    f.save("fig5-pd.svg", yb + 14)


def fig_afd():
    f = Fig(W, "attention 和 FFN 也拆开。左边 M 台机器只算 attention，右边 N 台机器只放专家。"
               "每一层 attention 算完，把 token 发给专家那边，这一步叫 M 到 N；专家算完再送回来，叫 N 到 M。"
               "专家那边同时接好几台 attention 机器的 token，专家的 batch 就大了。"
               "为了把这来回两趟藏起来，把一批请求切成三个小批轮着跑：一个在算 attention，一个在路上，一个在算专家，三条道每一格都有活")
    y0 = f.header("再拆一层：attention 和专家也分开　——　<tspan font-weight=\"700\">专家那边一次接好几家的 token</tspan>",
                  "AFD（Attention-FFN 分离）。示意：M 台 attention 机器、N 台专家机器，三个小批轮着跑",
                  [(BL, "attention 机器"), (OR, "专家机器"), (GR, "小批 A"), (PU, "小批 B"), (CY, "小批 C")])
    PH = 330
    py = f.panel(0, y0, W, PH, "每一层都要跑一趟 M → N → M", BL)
    for i in range(3):
        f.box(60, py + 50 + i * 76, 200, 56, "none", BL, 8, sw=2)
        f.t(160, py + 84 + i * 76, "attention 机器 %d" % (i + 1), BL, True, 14, "middle")
    for i in range(2):
        f.box(560, py + 80 + i * 96, 200, 56, "none", OR, 8, sw=2)
        f.t(660, py + 114 + i * 96, "专家机器 %d" % (i + 1), OR, True, 14, "middle")
    for i in range(3):
        for k in range(2):
            f.line(262, py + 78 + i * 76, 558, py + 108 + k * 96, GY2, 1.2, arrow=False)
    f.t(410, py + 36, "M → N：token 发给专家", GY, True, 13, anchor="middle")
    f.t(410, py + 290, "N → M：算完送回来", GY, True, 13, anchor="middle")
    # ⭐ 2026-09-25 L6 试讲：原来只画两个小批、没画「在路上」，看图会以为两个就够。
    #   改成 attention ／ 路上 ／ 专家三条道、A B C 三个小批：三条道每一格都有活，通信才藏得住。
    TX, CW, LW = 792, 54, 108          # LW：道名那一列的宽度，「路上（去＋回）」要 ~100px
    COLS3 = [GR, PU, CY]
    assert TX + LW + 9 * CW <= W - 10
    f.t(TX, py + 40, "三个小批轮着跑，三条道都不闲", INK, True, 15)
    for r, (lab, col) in enumerate((("attention", BL), ("路上（去＋回）", GY), ("专家", OR))):
        f.t(TX, py + 82 + r * 52, lab, col, True, 13)
        for t in range(9):
            b = t - r
            if b < 0:
                f.box(TX + LW + t * CW, py + 60 + r * 52, CW - 4, 34, "none", LINE, 3)
                continue
            c = COLS3[b % 3]
            f.box(TX + LW + t * CW, py + 60 + r * 52, CW - 4, 34, c, c, 3)
            f.t(TX + LW + t * CW + (CW - 4) / 2, py + 82 + r * 52, "ABC"[b % 3], "#ffffff", True, 13, "middle")
    f.t(TX, py + 238, "A 在算专家时，B 在路上、C 在算 attention。", GY, size=13)
    f.t(TX, py + 262, "单程通信不到半格时三个够，否则要四个（MegaScale-Infer）。", GY, size=13)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "拆得越细，每一边越能挑适合自己的机器和切法", [
        "专家那边同时接几台 attention 机器的 token，<tspan font-weight=\"700\">凑出来的 batch 比单台大得多</tspan>；两边的机器数也能分开调。",
        "代价是每一层都有一趟来回，得靠几个小批轮着跑藏住（论文里要三四个才藏得住）；两边之间的网络要够快。",
    ])
    yb = f.src(yb + 10, "📌 MegaScale-Infer arXiv 2504.02263（disaggregated expert parallelism、ping-pong pipeline、每 GPU 吞吐最高 1.90×）；"
                        "Step-3 arXiv 2507.19427；vLLM 于 2026-07 发布实验性 AFD 插件。")
    f.save("fig5-afd.svg", yb + 14)


# ── fig-decode-ai：decode 为什么非要把一批做大（2026-09-25 现场讲课补） ─────────────
import math
C_V7 = 2307e12                            # v7 每芯片 bf16 FLOP/s（同 topic05-fig-tp.py）
HBM_V7 = 2 * 3433 * 2 ** 30               # v7 每芯片 HBM 带宽：每 TensorCore 3,433 GiB/s × 2（wiki entities/tpu-v7）
RIDGE_HBM = C_V7 / HBM_V7
assert abs(HBM_V7 / 1e12 - 7.37) < 0.01 and 310 < RIDGE_HBM < 316, RIDGE_HBM
TOPK, EXPERTS = 8, 256                    # V3：每个 token 挑 256 个路由专家里的 8 个
SHARE = TOPK / EXPERTS                    # 一个专家平均分到这一批的 1/32
B_DENSE = RIDGE_HBM                       # 稠密层：每字节换来的计算 ＝ b（bf16 权重，一次乘加 2 FLOPs ÷ 2 字节）
B_MOE = RIDGE_HBM / SHARE
assert SHARE == 1 / 32 and 9900 < B_MOE < 10100, B_MOE


def fig_decode_ai():
    """⭐ 课件 6.1 原来一句话带过「decode 靠把很多请求拼成一大批来摊薄」。这张图把「多大才够」算出来。
    ⚠️ 推导：只算读权重，不算读 KV（算上 KV 门槛更高）；按 bf16 权重、每个专家的 token 均匀分。"""
    f = Fig(W, "decode 每一步都要把权重从显存读一遍。一批里有 b 个请求一起出字时，读一个字节的权重换来 b 次计算。"
               "TPU v7 每秒能算 2307 万亿次、每秒能从显存读约 7.4 万亿字节，一除约 313：一批不到约 313 个请求，卡就在等显存。"
               "MoE 更难：每个专家平均只分到这一批的三十二分之一，要一批约一万个请求，每个专家才吃得饱")
    y0 = f.header("decode 为什么非要把一批做大　——　<tspan font-weight=\"700\">读一遍权重，只够这一批用一次</tspan>",
                  "横轴：一步里一起出字的请求数 b（对数刻度）。纵轴：从显存读 1 字节权重，换来多少次计算（⚠️ 推导，只算读权重）",
                  [(BL, "稠密层：＝ b"), (OR, "MoE 的一个专家：＝ b ÷ 32（V3 挑 8／256）"), (RD, "v7 显存线 ≈ %.0f" % RIDGE_HBM)])
    PX, PY, PW, PH = 150, y0 + 20, 980, 360
    X0, X1 = 0, 14                         # log2 b：1 … 16,384
    Y0_, Y1_ = -5, 11                      # log2 强度：1/32 … 2,048

    def X(b):
        return PX + PW * (math.log2(b) - X0) / (X1 - X0)

    def Y(v):
        return PY + PH - PH * (math.log2(v) - Y0_) / (Y1_ - Y0_)
    yr = Y(RIDGE_HBM)
    f.poly([(PX, yr), (PX + PW, yr), (PX + PW, PY + PH), (PX, PY + PH)], "#fce8e6")
    f.box(PX, PY, PW, PH, "none", LINE, 6)
    for e in (0, 2, 4, 6, 8, 10, 12, 14):
        f.t(X(2 ** e), PY + PH + 22, "{:,}".format(2 ** e), GY, size=12.5, anchor="middle")
    f.t(PX + PW / 2, PY + PH + 46, "一步里一起出字的请求数 b", GY, True, 13.5, anchor="middle")
    for e in (-4, 0, 4, 8):
        v = 2 ** e
        f.t(PX - 10, Y(v) + 5, ("1/%d" % (1 / v)) if v < 1 else "{:,}".format(v), GY, size=12.5, anchor="end")
    f.path("M%.1f,%.1f L%.1f,%.1f" % (X(1), Y(1), X(2 ** 11), Y(2 ** 11)), BL, 3, arrow=False)
    f.path("M%.1f,%.1f L%.1f,%.1f" % (X(1), Y(SHARE), X(2 ** 14), Y(2 ** 14 * SHARE)), OR, 3, arrow=False)
    f.path("M%.1f,%.1f L%.1f,%.1f" % (PX, yr, PX + PW, yr), RD, 2, dash="4,4", arrow=False)
    f.t(PX + PW + 10, yr + 5, "显存线 ≈ %.0f" % RIDGE_HBM, RD, True, 13.5)
    for b, col, lab, dy in ((B_DENSE, BL, "稠密：b ≈ %.0f 才吃饱" % B_DENSE, -14),
                            (B_MOE, OR, "MoE：b ≈ %s 才吃饱" % "{:,.0f}".format(round(B_MOE, -2)), -14)):
        f.p.append('<circle cx="%.1f" cy="%.1f" r="7" fill="%s"/>' % (X(b), yr, col))
        f.path("M%.1f,%.1f L%.1f,%.1f" % (X(b), yr, X(b), PY + PH), col, 1.4, dash="4,3", arrow=False)
        f.t(X(b) - 12, yr + dy, lab, col, True, 15, "end")
    f.t(PX + 20, PY + 30, "线上：卡算得过来", GR, True, 15)
    f.t(PX + PW * 0.55, PY + PH - 30, "线下：卡在等显存把权重读出来", RD, True, 15)
    yb = f.band(PY + PH + 70, "ok", "所以 decode 的各种做法，都在凑一个大 batch", [
        "PD 分离让 decode 机器只管出字，一批能攒得更大；DEP 让 attention 各管各的请求，专家那边收齐所有卡的 token。",
        "Wide-EP 把专家铺到更多卡上、每张卡只放几个，AFD 干脆让一组专家机器同时接好几组 attention 机器的 token。",
    ])
    yb = f.src(yb + 10,
               "⚠️ 推导：稠密层一步读 2P 字节（bf16）、算 2Pb 次，每字节 ＝ b；MoE 每个专家平均分到 b × 8 ÷ 256 个 token。只算读权重，读 KV 会让门槛更高。",
               "📌 v7：每芯片 bf16 2,307 TFLOP/s；HBM 每 TensorCore 3,433 GiB/s、每芯片两个（wiki entities/tpu-v7）。V3 的 8／256 取自 config.json。")
    f.save("fig5-decode-ai.svg", yb + 14)


fig_pd()
fig_afd()
fig_decode_ai()
