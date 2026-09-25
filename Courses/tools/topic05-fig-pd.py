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
                  [(BL, "attention 机器"), (OR, "专家机器"), (GR, "小批 A（盯着它看）"), ("#80868b", "小批 B"), ("#bdc1c6", "小批 C")])
    PH = 340
    py = f.panel(0, y0, W, PH, "每一层都要跑一趟 M → N → M", BL)
    for i in range(3):
        f.box(40, py + 50 + i * 76, 180, 56, "none", BL, 8, sw=2)
        f.t(130, py + 84 + i * 76, "attention 机器 %d" % (i + 1), BL, True, 14, "middle")
    for i in range(2):
        f.box(460, py + 80 + i * 96, 180, 56, "none", OR, 8, sw=2)
        f.t(550, py + 114 + i * 96, "专家机器 %d" % (i + 1), OR, True, 14, "middle")
    # ⭐ 2026-09-25 讲后自查：原来只有无向细线，「去」和「回」两个标签挂在同一捆线上，看不出方向。
    for i in range(3):
        for k in range(2):
            f.line(222, py + 72 + i * 76, 456, py + 100 + k * 96, BL, 1.3)
    f.path("M 460,%d C 380,%d 300,%d 222,%d" % (py + 150, py + 300, py + 300, py + 206), OR, 1.8, dash="6,4")
    f.t(340, py + 34, "去（M → N）：token 发给专家", BL, True, 13, anchor="middle")
    f.t(340, py + 300, "回（N → M）：算完送回来", OR, True, 13, anchor="middle")
    # ⭐⭐ 2026-09-25 讲后自查：原来三条道是 attention ／「路上（去＋回）」／专家，每格一整步 ——
    #   A 在第 2 格算完专家、第 3 格就又在算 attention，**回程根本没地方走**，逻辑是断的。
    #   按 MegaScale-Infer 的条件画真时序：单程通信 Tc ＝ 半格，一个小批走一圈
    #   ＝ attention 1 ＋ 去 ½ ＋ 专家 1 ＋ 回 ½ ＝ 3 格，所以三个小批正好把 attention 和专家都填满；
    #   专家那条道比 attention 晚半格，去、回各占半格。论文式 m ≥ 2(1 ＋ Tc／Tf)。
    TC = 0.5
    CYCLE = 1 + TC + 1 + TC
    M_MB = 3
    assert M_MB >= 2 * (1 + TC / 1.0) and CYCLE == M_MB
    TX, LW, SW, NS = 690, 92, 70, 8          # SW：一格宽
    assert TX + LW + NS * SW <= W - 10
    COLS3 = [GR, "#80868b", "#bdc1c6"]      # ⭐ 麻瓜读图：只高亮 A，B、C 调灰，盯一个小批才看得清
    f.t(TX, py + 36, "三个小批轮着跑：去、回各占半格", INK, True, 15)
    rows = (("attention", BL, 0.0, 1.0), ("去", BL, 1.0, TC), ("专家", OR, 1.0 + TC, 1.0), ("回", OR, 2.0 + TC, TC))
    for r, (lab, col, off, dur) in enumerate(rows):
        yy = py + 56 + r * 46
        f.t(TX, yy + 22, lab, col, True, 13)
        f.box(TX + LW, yy, NS * SW, 32, "none", LINE, 3)
        for b in range(M_MB):
            for c in range(4):
                t0 = b + c * CYCLE + off
                t1 = t0 + dur
                if t0 >= NS:
                    continue
                t1 = min(t1, NS)
                x0 = TX + LW + t0 * SW + 1
                w = (t1 - t0) * SW - 3
                f.box(x0, yy + 1, w, 30, COLS3[b], COLS3[b], 3)
                if w > 16:
                    f.t(x0 + w / 2, yy + 21, "ABC"[b], "#ffffff", True, 13, "middle")
    for t in range(NS + 1):
        f.t(TX + LW + t * SW, py + 56 + 4 * 46 + 12, str(t), GY, size=12, anchor="middle")
    f.t(TX, py + 282, "看 A：第 0 格算 attention，去半格，专家一格，回半格，第 3 格又轮到它。", GY, size=13)
    f.t(TX, py + 304, "单程通信超过半格，一圈就超过 3 格，得四个小批（MegaScale-Infer）。", GY, size=13)
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
                  "横轴：一步里一起出字的请求数 b（对数刻度）。纵轴：从显存读 1 字节权重，换来多少次计算（本课推导，只算读权重）",
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


# ── fig-kv-trip：一趟 KV 传输对一次 prefill，按同一把尺子画（2026-09-25 讲后：6.2 那张表是证据不是图） ──
SEG = [("① 显存 → 本机内存（PCIe）", 10, BL), ("② 本机 → 对方（数据中心网络 100 Gbps）", round(T_DCN * 1e3), OR),
       ("③ 对方内存 → 显存（PCIe）", 10, GR)]
TRIP_MS = sum(m for _, m, _ in SEG)
PREFILL_MS = (1000, 2000)                 # 8K prompt 的 prefill 1–2 s（wiki 实测页）
assert 100 <= TRIP_MS <= 105 and 0.05 <= TRIP_MS / PREFILL_MS[1] and TRIP_MS / PREFILL_MS[0] <= 0.105


def fig_kv_trip():
    f = Fig(W, "一趟 KV 传输和一次 prefill 放在同一把时间尺上。8K token 的 prompt，prefill 本身要 1 到 2 秒；"
               "把它的 KV cache 从 prefill 机器搬到 decode 机器，按带宽估算约 100 毫秒，分三段：显存到本机内存约 10 毫秒，"
               "走数据中心网络约 83 毫秒，再进对方显存约 10 毫秒。只占 prefill 的百分之五到十")
    y0 = f.header("一趟 KV 传输，只占一次 prefill 的 5–10%　——　<tspan font-weight=\"700\">网络不是瓶颈</tspan>",
                  "我们在 TPU v7x 上的 1P1D（Qwen3-Coder-480B），8K token 的 prompt。KV 传输是按带宽估算，不是计时实测",
                  [(GY2, "prefill（1–2 秒）"), (BL, "①"), (OR, "②"), (GR, "③")])
    PH = 268
    py = f.panel(0, y0, W, PH, "同一把时间尺：0 到 2 秒", OR)
    BX, BW = 230, 1100
    SC = BW / PREFILL_MS[1]

    def X(ms):
        return BX + ms * SC
    for ms in range(0, 2001, 500):
        f.t(X(ms), py + 44, "%d ms" % ms if ms else "0", GY, size=12.5, anchor="middle")
        f.line(X(ms), py + 52, X(ms), py + 150, LINE, 1, arrow=False)
    f.t(24, py + 88, "一次 prefill", INK, True, 15)
    f.box(X(0), py + 66, X(PREFILL_MS[0]) - X(0), 34, GY2, GY2, 3)
    f.box(X(PREFILL_MS[0]), py + 66, X(PREFILL_MS[1]) - X(PREFILL_MS[0]), 34, "none", GY2, 3, dash="5,4")
    f.t(X(PREFILL_MS[0]) + 12, py + 89, "1–2 秒", GY, True, 14)
    f.t(24, py + 138, "一趟 KV 传输", INK, True, 15)
    x = X(0)
    for _, ms, col in SEG:
        f.box(x, py + 116, ms * SC, 34, col, col, 2)
        x += ms * SC
    f.t(x + 12, py + 139, "10 ＋ %d ＋ 10 ≈ 100 ms" % SEG[1][1], INK, True, 15)
    # 放大 10 倍看三段
    ZX, ZS = X(0), 10 * SC
    f.t(24, py + 200, "放大 10 倍：", GY, True, 14)
    zx = ZX
    for lab, ms, col in SEG:
        w = ms * ZS
        f.box(zx, py + 180, w - 2, 30, col, col, 3)
        f.t(zx + (w - 2) / 2, py + 200, "%d ms" % ms, "#ffffff", True, 13, "middle")
        f.t(zx + (w - 2) / 2, py + 232, lab, col, size=13, anchor="middle")
        zx += w
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "敢走慢线，是因为一个请求只传这一次", [
        "这个模型 KV 头少（8 个）又用 FP8 存，8K prompt 的 KV 约 1.04 GB；换一个 KV 大得多的模型、换网络，要重算。",
        "对比第三节的 TP：它每一层都要通信，只能待在最快那一圈；这里一个请求就一趟，跨机器也藏得住。",
    ])
    yb = f.src(yb + 10,
               "📌 我们的 TPU v7x 1P1D 记录（wiki qwen3-coder-480b-pd-disagg-tpuv7x-20260425）：三段按带宽估算约 10 ／ 80 ／ 10 ms、合计约 100 ms、占 prefill 5–10%（prefill 1–2 s 由此反推）。",
               "⚠️ 本课推导：KV ＝ 2 × 62 层 × 8 个 KV 头 × 128 × 8,192 × 1 字节 ≈ 1.04 GB；100 Gbps ＝ 12.5 GB/s → 约 83 ms。")
    f.save("fig5-kv-trip.svg", yb + 14)


# ── fig-pd-ratio：拆开以后，三张卡顶混着做的六张（2026-09-25 夜 · 蒸馏 R7） ─────────────
# ⭐ 算术借自 DistServe（Hao AI Lab 博客「Throughput is Not All You Need」）：同一套延迟要求下，
#   一张卡两样都做每秒接 1.6 个请求；只做 prefill 接 5.6 个；只做 decode 接 10 个。
MIX, P_ONLY, D_ONLY = 1.6, 5.6, 10.0
N_P, N_D = 2, 1
PD_RPS = min(N_P * P_ONLY, N_D * D_ONLY)
MIX_CARDS = math.ceil(PD_RPS / MIX)
assert PD_RPS == 10.0 and MIX_CARDS == 7 and abs(PD_RPS / (N_P + N_D) / MIX - 2.08) < 0.01


def fig_pd_ratio():
    f = Fig(W, "同样守住首字延迟和出字间隔两个要求。一张卡两样都做，每秒只接得住 1.6 个请求；只做 prefill 能接 5.6 个，只做 decode 能接 10 个。"
               "两张做 prefill、一张做 decode，三张卡每秒接 10 个请求，每张卡 3.3 个，是混着做的两倍多；混着做要接住 10 个，得 7 张卡")
    y0 = f.header("拆开以后，3 张卡顶混着做的 7 张　——　<tspan font-weight=\"700\">各干各的，谁也不拖谁</tspan>",
                  "同一套延迟要求下，一张卡每秒接得住几个请求（DistServe 论文的例子，OPT 模型）",
                  [(GY2, "两样都做"), (OR, "只做 prefill"), (GR, "只做 decode")])
    PH = 272
    py = f.panel(0, y0, W, PH, "每张卡每秒接得住几个请求", INK)
    CWD, CH_, GAPC = 92, 74, 14

    def cards(x, y, n, col, lab, rps):
        for i in range(n):
            f.box(x + i * (CWD + GAPC), y, CWD, CH_, "none" if col == GY2 else col, GY2 if col == GY2 else col, 8, sw=2)
            f.t(x + i * (CWD + GAPC) + CWD / 2, y + 32, lab, INK if col == GY2 else "#ffffff", True, 13, "middle")
            f.t(x + i * (CWD + GAPC) + CWD / 2, y + 56, "%g 个/秒" % rps, INK if col == GY2 else "#ffffff", True, 13, "middle")
    f.t(24, py + 60, "混着做", INK, True, 15)
    cards(140, py + 30, MIX_CARDS, GY2, "两样都做", MIX)
    f.t(140 + MIX_CARDS * (CWD + GAPC) + 10, py + 72, "7 张 ≈ %.1f 个/秒" % (MIX_CARDS * MIX), INK, True, 15)
    f.t(24, py + 170, "拆开", INK, True, 15)
    cards(140, py + 140, N_P, OR, "prefill", P_ONLY)
    cards(140 + N_P * (CWD + GAPC), py + 140, N_D, GR, "decode", D_ONLY)
    f.t(140 + 3 * (CWD + GAPC) + 10, py + 172, "3 张 ＝ min(2 × %g, %g) ＝ %g 个/秒" % (P_ONLY, D_ONLY, PD_RPS), GR, True, 15)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "每张卡接的请求翻了一倍多，配比就是这么算出来的", [
        "两边各有各的上限：prefill 一张接 5.6 个，decode 一张接 10 个，所以配 2 比 1，两边差不多同时忙满。",
        "长 prompt 的业务 prefill 更累，就多配 prefill；长回答的业务多配 decode。",
    ])
    yb = f.src(yb + 10, "📌 DistServe（Zhong 等，arXiv 2401.09670）作者博客「Throughput is Not All You Need」中的例子：同一 SLO 下单卡 goodput 1.6／5.6／10 rps。",
               "⚠️ 本课推导：2P1D 每秒 min(2 × 5.6, 10) ＝ 10 个，每卡约 3.3 个；混着做接 10 个要 ⌈10 ÷ 1.6⌉ ＝ 7 张卡。")
    f.save("fig5-pd-ratio.svg", yb + 14)


fig_pd()
fig_afd()
fig_decode_ai()
fig_kv_trip()
fig_pd_ratio()
