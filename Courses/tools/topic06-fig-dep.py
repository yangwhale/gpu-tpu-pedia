# -*- coding: utf-8 -*-
r"""专题六 · 第六节「切专家与注意力：TEP 和 DEP」的三张静态图。

回答的问题
  · fig6-dep-kv   同样 8 张卡、同样每卡的 KV 预算，TEP 和 DEP 各放得下几个请求？倍数由什么决定？
  · fig6-dep-feed DEP8 的 decode 为什么光加 decode 卡没用？（prefill 台数 → 每块 decode 卡吞吐）
  · fig6-dep-tpu  TPU 上换成 DP 注意力，赢多少？跟并发什么关系？

数从哪来
  · V4-Pro 只有 1 个 KV 头（num_key_value_heads 1，CSA/HCA，不是 MLA）：
    gpu/inference/a4x-max/deepseek-v4/VLLM-V4PRO-RUNBOOK.md:298-302
  · 3p1d 并发 512 = 21,100、3p + dep8 并发 512 = 55,153（总吞吐，prompt + 输出）：同文件 :449、:464-465；
    GPU 数：prefill 与 TP4 decode 各 1 节点 4 卡、dep8 2 节点 8 卡（同文件 :476）。每卡数本脚本现算。
  · dep8 8k1k 并发 1,024，prefill 1→16 台的每卡输出吞吐：同文件 :711-724；4k1k 重扫：:727-734
  · tpu-inference PR #2577（GitHub 公开，2026-05-15 合入）正文的 TP=8 vs DP=8 表：
    https://github.com/vllm-project/tpu-inference/pull/2577
  · Qwen3.5-397B-A17B 的 num_key_value_heads = 2（HF config.json，全注意力层；线性注意力层是 GDN）

刻意没画
  · fig6-dep-kv 不画「EP 从 4 路变 8 路省下的专家权重」那一半：要权重分项才能算准，没现算就不画。
  · fig6-dep-kv 的「每卡 4 个槽」是示意，不是 V4-Pro 的真实容量。
  · fig6-dep-tpu 不画 TP／DP 的绝对吞吐曲线，只画相对差；绝对数只标在折线两端。
"""
from topic03_draw import Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE

W = 1400

# ── V4-Pro 实测：每卡吞吐 ─────────────────────────────────────────────
TP4_TOT, TP4_GPU = 21100, 4 * 3 + 4          # 3 台 TP4 prefill ＋ 1 台 TP4 decode
DEP_TOT, DEP_GPU = 55153, 4 * 3 + 8          # 3 台 TP4 prefill ＋ dep8（2 节点）
TP4_PER, DEP_PER = TP4_TOT / TP4_GPU, DEP_TOT / DEP_GPU
RATIO = DEP_PER / TP4_PER
assert (TP4_GPU, DEP_GPU) == (16, 20)
assert round(TP4_PER) == 1319 and round(DEP_PER) == 2758, (TP4_PER, DEP_PER)
assert round(RATIO, 2) == 2.09, RATIO
TPOT_TP4, TPOT_DEP = 46.8, 11.8

# ── 示意：8 张卡，每卡 KV 池 4 个请求槽 ─────────────────────────────────
NCARD, SLOT = 8, 4


def distinct(tp, kv_heads=1):
    """TP 组内每张卡存每个请求 1/min(tp,H) 的 KV。按头切不开的部分被复制 tp/H 份。"""
    groups = NCARD // tp
    per_group = SLOT * min(tp, kv_heads)       # 一组 tp 张卡合起来能放几个不同的请求
    return groups * per_group


TEP_N, DEP_N = distinct(4), distinct(1)
assert (TEP_N, DEP_N) == (8, 32) and DEP_N // TEP_N == 4


def gain(tp, h):
    """DEP 相对 TEP 的 KV 容量倍数 ＝ tp ÷ min(tp, H)。"""
    return tp / min(tp, h)


MODELS = [("DeepSeek-V4-Pro", "CSA／HCA，1 个 KV 头", 4, 1),
          ("MLA 模型（V3、K3 的 MLA 层）", "1 个 KV 头", 8, 1),
          ("Qwen3.5-397B 全注意力层", "2 个 KV 头", 8, 2),
          ("8 个 KV 头的 GQA 模型", "8 个 KV 头", 8, 8)]
GAINS = [gain(tp, h) for _, _, tp, h in MODELS]
assert GAINS == [4.0, 8.0, 4.0, 1.0]

# ── dep8 喂料曲线（8k1k，并发 1,024；纵轴 = 输出 ÷ 8 块 decode 卡）──────────
FEED = [(1, 307, "5.4"), (2, 597, "5.6"), (4, 1115, "7.8"), (6, 1546, "10.6"), (8, 2202, "10.7"),
        (10, 2679, "15.1"), (12, 3187, "14.3"), (14, 3321, None)]
FEED_BAD = (16, 2799)
FEED4K = [(1, 632), (2, 1238), (3, 1656), (4, 2222)]
assert FEED[0][1] == 307 and FEED[-1][1] == 3321
assert round(26568 / 8) == 3321 and round(2453 / 8) == 307      # 原表的 Output tok/s ÷ 8
assert [round(b / a, 2) for (_, a, _), (_, b) in zip([FEED[0], FEED[1], FEED[2]], [FEED4K[0], FEED4K[1], FEED4K[3]])] \
    == [2.06, 2.07, 1.99]

# ── tpu-inference PR #2577：Qwen3.5-397B-A17B-FP8，v7x-8，TP=8 vs DP=8（输出 tok/s）──
CONC = [64, 128, 256, 512]
PR_A = [(1964, 1973), (2952, 3363), (4130, 5029), (4266, 6914)]     # in 1k / out 8k
PR_B = [(1097, 1247), (1353, 1668), (1496, 2033), (1490, 2090)]     # in 8k / out 1k
ADV_A = [round((d / t - 1) * 100) for t, d in PR_A]
ADV_B = [round((d / t - 1) * 100) for t, d in PR_B]
assert ADV_A == [0, 14, 22, 62] and ADV_B == [14, 23, 36, 40], (ADV_A, ADV_B)
assert round((PR_A[3][0] / PR_A[2][0] - 1) * 100) == 3 and round((PR_A[3][1] / PR_A[2][1] - 1) * 100) == 37


def fig_dep_kv():
    f = Fig(W, "同样 8 张卡，每张卡的 KV 池画成 4 个请求槽。TEP4 是两组各 4 张卡，V4-Pro 只有一个 KV 头，"
               "每组 4 张卡存的是同样 4 个请求的整份 KV，8 张卡一共只放下 8 个不同的请求，其余 24 格是副本。"
               "DEP8 每张卡管自己的 4 个请求，8 张卡放下 32 个，是 4 倍。"
               "实测 V4-Pro 从 TP4 换 DEP8，同并发 512，每卡吞吐从 1319 到 2758，2.09 倍")
    y0 = f.header("同样的 KV 预算，DEP 放下的请求多几倍　——　<tspan font-weight=\"700\">TP 度数 ÷ KV 头数</tspan>",
                  "8 张卡，每张卡的 KV 池画成 4 个槽，格子里是请求编号（示意）。V4-Pro 只有 1 个 KV 头",
                  [(GR, "不同的请求"), (RD, "同一请求的副本")])
    PH = 560
    py = f.panel(0, y0, W, PH, "8 张卡能同时放下几个请求", GR)
    CW, GAP, SH, SP = 96, 14, 34, 40
    X0 = 190
    GW = NCARD * (CW + GAP) - GAP
    rows = [("TEP4 × 2 组", 4), ("DEP8", 1)]
    for r, (name, tp) in enumerate(rows):
        yy = py + 46 + r * 250
        f.t(24, yy + 66, name, INK, True, 19)
        f.t(24, yy + 92, "注意力 TP4" if tp > 1 else "注意力各算各的", GY, size=15)
        for c in range(NCARD):
            x = X0 + c * (CW + GAP)
            if tp > 1 and c % tp == 0 and c:
                f.line(x - GAP / 2, yy - 10, x - GAP / 2, yy + SLOT * SP + 6, GY2, 1.4, "5 4", arrow=False)
            f.box(x, yy - 6, CW, SLOT * SP + 6, "none", LINE, 6)
            for s in range(SLOT):
                if tp > 1:
                    req = (c // tp) * SLOT + s + 1
                    dup = c % tp != 0
                else:
                    req = c * SLOT + s + 1
                    dup = False
                col = RD if dup else GR
                f.box(x + 7, yy + s * SP, CW - 14, SH, "#fce8e6" if dup else "#e6f4ea", col, 4)
                f.t(x + CW / 2, yy + s * SP + 23, str(req), col, True, 16, "middle")
            f.t(x + CW / 2, yy + SLOT * SP + 26, "卡 %d" % c, GY, size=14, anchor="middle")
        n = TEP_N if tp > 1 else DEP_N
        f.t(X0 + GW + 30, yy + 70, "放下", GY, size=16)
        f.t(X0 + GW + 30, yy + 108, "%d 个" % n, RD if tp > 1 else GR, True, 34)
    # 右侧两个大号数
    RX = X0 + GW + 150
    f.line(RX - 20, py + 30, RX - 20, py + PH - 70, LINE, 1, arrow=False)
    f.t(RX, py + 150, "KV 放得下的请求", GY, size=16)
    f.t(RX, py + 206, "4 倍", GR, True, 52)
    f.t(RX, py + 236, "TP 度数 4 ÷ KV 头数 1", GY, size=15)
    f.t(RX, py + 350, "实测每卡吞吐", GY, size=16)
    f.t(RX, py + 406, "%.2f 倍" % RATIO, BL, True, 52)
    f.t(RX, py + 436, "%s → %s tok/s" % (format(round(TP4_PER), ","), format(round(DEP_PER), ",")), GY, size=15)
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "4 倍是容量上限；实测 2.09 倍里还掺着 EP 和 prefill 喂料，原始记录没拆开归因", [])
    yb = f.src(yb + 10,
               "V4-Pro 单 KV 头（num_key_value_heads 1，CSA／HCA 交错，不是 MLA）：gpu/inference/a4x-max/deepseek-v4/VLLM-V4PRO-RUNBOOK.md:298-302。",
               "实测：同文件 :449（3p1d 并发 512 总吞吐 21,100，出字间隔 46.8 ms）、:464-465（3p ＋ dep8 并发 512 为 55,153，11.8 ms）。"
               "吞吐是 prompt 加输出的总 token；GPU 数 16 和 20 见同文件 :476，每卡数本脚本现算。",
               "本课推导：每卡 4 个槽是示意；倍数 ＝ TP ÷ min(TP, KV 头数)，与 vLLM 文档「KV 被复制 tp_size / H 次」一致。"
               "其他模型：MLA 模型 TEP8 是 8 倍；Qwen3.5-397B 全注意力层 2 个 KV 头，TEP8 是 4 倍；8 个 KV 头的 GQA 模型 TEP8 是 1 倍。"
               "Qwen3.5-397B-A17B 的 num_key_value_heads ＝ 2 取自 HF config.json；它的线性注意力层（GDN）按头切得开，不在这个倍数里。")
    f.save("fig6-dep-kv.svg", yb + 14)


def _axes(f, x0, y0, w, h, ymax, yticks, fmt):
    for v in yticks:
        yy = y0 + h - h * v / ymax
        f.line(x0, yy, x0 + w, yy, "#eceff1", 1, arrow=False)
        f.t(x0 - 10, yy + 5, fmt(v), GY, size=14, anchor="end")
    f.line(x0, y0 + h, x0 + w, y0 + h, GY2, 1.2, arrow=False)


def fig_dep_feed():
    f = Fig(W, "GB300 上 V4-Pro 用 vLLM，decode 固定为 DEP8 共 8 块卡，8K 进 1K 出、并发 1024，逐台加 prefill。"
               "1 台 prefill 时每块 decode 卡每秒只输出 307 个 token，出字间隔 5.4 毫秒；加到 14 台涨到 3321，几乎是一条直线。"
               "16 台时撞上一个失效模式，只测到 2799。输入减半成 4K 时，同样台数的 prefill 喂出约两倍。"
               "出字间隔一直只有 5 到 15 毫秒，说明 decode 一直在等 prefill 喂料")
    y0 = f.header("DEP8 的 decode 一直在等喂料　——　<tspan font-weight=\"700\">prefill 1 → 14 台，每卡 307 → 3,321</tspan>",
                  "GB300 · V4-Pro · vLLM。decode 固定 DEP8（8 块卡），并发 1,024。纵轴：每块 decode 卡每秒输出多少 token",
                  [(BL, "8K 进 1K 出"), (OR, "4K 进 1K 出（同一套 decode 不重启）"), (RD, "失效模式，数不完整")])
    PH = 430
    py = f.panel(0, y0, W, PH, "prefill 台数 → 每块 decode 卡的输出吞吐", BL)
    X0, PW, PY, PHH = 110, 1240, py + 30, PH - 110
    XMAX, YMAX = 16, 3600
    _axes(f, X0, PY, PW, PHH, YMAX, [0, 1000, 2000, 3000], lambda v: format(v, ","))

    def px(n):
        return X0 + PW * n / XMAX

    def py_(v):
        return PY + PHH - PHH * v / YMAX
    for n in (1, 2, 4, 6, 8, 10, 12, 14, 16):
        f.t(px(n), PY + PHH + 22, str(n), GY, size=14, anchor="middle")
    f.t(X0 + PW / 2, PY + PHH + 48, "prefill 台数（每台 TP4，4 块卡）", GY, size=15, anchor="middle")
    f.path([(px(n), py_(v)) for n, v, _ in FEED], BL, 2.6, arrow=False)
    for n, v, _ in FEED:
        f.box(px(n) - 5, py_(v) - 5, 10, 10, BL, BL, 5)
    f.path([(px(14), py_(3321)), (px(16), py_(FEED_BAD[1]))], RD, 1.6, "5 4", arrow=False)
    f.box(px(16) - 5, py_(FEED_BAD[1]) - 5, 10, 10, RD, RD, 5)
    f.t(px(16) - 8, py_(FEED_BAD[1]) + 26, "16 台：失效", RD, True, 15, "end")
    f.path([(px(n), py_(v)) for n, v in FEED4K], OR, 2.4, arrow=False)
    for n, v in FEED4K:
        f.box(px(n) - 5, py_(v) - 5, 10, 10, OR, OR, 5)
    f.t(px(4) + 12, py_(2222) - 8, "输入减半，同样台数喂出约 2 倍", OR, True, 15)
    f.t(px(1) + 12, py_(307) - 12, "307", BL, True, 18)
    f.t(px(14) - 4, py_(3321) - 14, "3,321", BL, True, 18, "end")
    f._pan = None
    yb = y0 + PH + 10
    yb = f.src(yb + 10,
               "gpu/inference/a4x-max/deepseek-v4/VLLM-V4PRO-RUNBOOK.md:711-724（8k1k，并发 1,024，sa-bench 开环，表中 Output tok/s ÷ 8）、"
               ":727-734（4k1k 重扫，倍数 2.06／2.07／1.90／1.99）、:736 起（16 台时的失效模式 B）。TP4 decode 加第 4 台 prefill，21,100 → 20,479，见同文件 :488。",
               "这组是 2026-07-24 的旧记录，runbook 注明「本文未重测」；14 台那一点出字间隔原表没给。")
    f.save("fig6-dep-feed.svg", yb + 14)


def fig_dep_tpu():
    f = Fig(W, "tpu-inference 的一个公开 PR，在 TPU v7x-8 上跑 Qwen3.5-397B-A17B-FP8，比较 TP 等于 8 和 DP 注意力等于 8，专家都走 EP。"
               "1K 进 8K 出时，并发 64 打平，128 多 14%，256 多 22%，512 多 62%。8K 进 1K 出时依次多 14%、23%、36%、40%。"
               "并发越高，DP 注意力赢得越多。1K 进 8K 出并发 512 时，输出吞吐 TP 是 4266、DP 是 6914")
    y0 = f.header("TPU 上同一个模型：并发越高，DP 注意力比 TP 赢得越多　——　<tspan font-weight=\"700\">512 并发 +62%</tspan>",
                  "tpu-inference PR #2577 · Qwen3.5-397B-A17B-FP8 · TPU v7x-8（4 颗芯片、8 个 device）· TP=8 对 DP=8，专家都走 EP。纵轴：DP 的输出吞吐比 TP 多多少",
                  [(BL, "1K 进 8K 出（decode 为主）"), (OR, "8K 进 1K 出（prefill 为主）")])
    PH = 440
    py = f.panel(0, y0, W, PH, "并发 → DP 注意力相对 TP 的输出吞吐（端点标的是输出 tok/s，TP → DP）", BL)
    X0, PW, PY, PHH = 100, 1000, py + 40, PH - 130
    YMAX = 70
    _axes(f, X0, PY, PW, PHH, YMAX, [0, 20, 40, 60], lambda v: "+%d%%" % v if v else "打平")

    def px(i):
        return X0 + 270 + (PW - 330) * i / 3

    def py_(v):
        return PY + PHH - PHH * v / YMAX
    for i, c in enumerate(CONC):
        f.t(px(i), PY + PHH + 24, str(c), GY, size=15, anchor="middle")
    f.t(X0 + PW / 2, PY + PHH + 50, "并发", GY, size=15, anchor="middle")
    for adv, pr, col in ((ADV_A, PR_A, BL), (ADV_B, PR_B, OR)):
        f.path([(px(i), py_(v)) for i, v in enumerate(adv)], col, 2.8, arrow=False)
        for i, v in enumerate(adv):
            f.box(px(i) - 6, py_(v) - 6, 12, 12, col, col, 6)
            if 0 < i < 3:
                f.t(px(i) + (12 if col == BL else -12), py_(v) + (26 if col == BL else -12),
                    "+%d%%" % v, col, True, 16, None if col == BL else "end")
        # 右端点：百分比 ＋ 绝对值
        t, d = pr[3]
        f.t(px(3) + 18, py_(adv[3]) + 2, "+%d%%" % adv[3], col, True, 26)
        f.t(px(3) + 18, py_(adv[3]) + 26, "%s → %s" % (format(t, ","), format(d, ",")), col, size=15)
        # 左端点：绝对值
        t, d = pr[0]
        lab = ("打平　" if adv[0] == 0 else "+%d%%　" % adv[0]) + "%s → %s" % (format(t, ","), format(d, ","))
        f.t(px(0) - 16, py_(adv[0]) + (-8 if col == BL else 6), lab, col, True, 15, "end")
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "PR 同时改了调度：prefill 攒在一起派，免得一条长 prefill 混进 DP 批里拖慢所有 rank", [])
    yb = f.src(yb + 10,
               "github.com/vllm-project/tpu-inference/pull/2577「Enable correct DP attention for hybrid attention＋mamba models」，2026-05-15 合入；"
               "PR 正文表格给输出／总吞吐两个数，本图用输出吞吐现算百分比，与 PR 的「DP Advantage」一列一致。正确性：mmlu_pro DP=8 0.8129 对 TP=8 0.8271（PR 原话：limit=50 时在 1σ 内）。",
               "v7x 一颗芯片对软件显示成 2 个 device，TP=8、DP=8 按 device 数。KV 头数取自 HF Qwen/Qwen3.5-397B-A17B-FP8 config.json（num_key_value_heads 2，60 层里每 4 层一层全注意力）。"
               "「复制 4 份卡住 batch」是本课解读，PR 没有单独归因。")
    f.save("fig6-dep-tpu.svg", yb + 14)


fig_dep_kv()
fig_dep_feed()
fig_dep_tpu()
