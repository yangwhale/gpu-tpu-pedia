# -*- coding: utf-8 -*-
r"""专题六 · 第六节「切专家与注意力：TEP 和 DEP」的三张静态图。

回答的问题
  · fig6-dep-kv   TEP 换 DEP 分容量、速度两条轴看；SGLang 两头 DEP 的部署里，座位多少换多少吞吐？（vLLM TP4→dep8 只作速度旁证）
  · fig6-dep-tpu  TPU 上换成 DP 注意力，赢多少？跟并发什么关系？
  · fig6-dep-state K3 的 KDA 状态池：DEP 为什么每个 rank 都得备一整份，TP 为什么切得开？（示意）

数从哪来
  · V4-Pro 只有 1 个 KV 头（num_key_value_heads 1，CSA/HCA，不是 MLA）：
    gpu/inference/a4x-max/deepseek-v4/VLLM-V4PRO-RUNBOOK.md:298-302
  · 3p1d 并发 512 = 21,100（输出 4,220、TPOT 46.8 ms）、3p + dep8 并发 512 = 55,153（输出 11,031、TPOT 11.8 ms）：同文件 :449、:464-465；投机接受率 :480；
    GPU 数：prefill 与 TP4 decode 各 1 节点 4 卡、dep8 2 节点 8 卡（同文件 :476）。每卡数本脚本现算。
  · dep8 8k1k 并发 1,024，prefill 1→16 台的每卡输出吞吐：同文件 :711-724；4k1k 重扫：:727-734
  · tpu-inference PR #2577（GitHub 公开，2026-05-15 合入）正文的 TP=8 vs DP=8 表：
    https://github.com/vllm-project/tpu-inference/pull/2577
  · Qwen3.5-397B-A17B 的 num_key_value_heads = 2（HF config.json，全注意力层；线性注意力层是 GDN）

刻意没画
  · fig6-dep-kv 不画「EP 从 4 路变 8 路省下的专家权重」那一半：要权重分项才能算准，没现算就不画。
  · ⛔ 不拿 Little 定律算出的「decode 上平均在跑几个请求」（197 → 130）去反驳容量：那个数是出字快慢的结果
    （跑得快、走得快），和显存能放几个是两条轴，不能互证（2026-10-21 改）。
  · 2026-10-21 再改：主线换成 SGLang 两头 DEP 的完整部署（SGLANG-V4PRO-RUNBOOK.md:33-34、:851-858），
    它记了每卡在跑请求数和两个 KV 池占用，容量轴看得见；vLLM TP4 对 dep8 只剩底部一行速度旁证（唯一一组 TP 对 DEP 实测）。
  · 步时 46.8 → 11.8 ms 里 KV 少读、AllReduce 去掉、EP 变宽各占多少，runbook 没拆，图上不分。
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
OUT_TP4, OUT_DEP = 4220, 11031                 # 同一行的输出 tok/s
PD_TP4, PD_DEP = OUT_TP4 / 4, OUT_DEP / 8      # 每张 decode 卡的输出
assert round(PD_TP4) == 1055 and round(PD_DEP) == 1379 and round(PD_DEP / PD_TP4, 2) == 1.31
assert round(OUT_DEP / OUT_TP4, 2) == 2.61

# ── SGLang 两头 DEP 实测（ISL 4096，14 prefill dep4 ＋ dep8 decode）──────────
SG_ROWS = [("池子划错（ratio 0.20）", 727, "0.93 ／ 0.63", 6887, 58.0, INK),
           ("划平（ratio 0.15）", 886, "0.92 ／ 0.89", 10614, 77.7, BL),
           ("再多 13% 显存", 999, "0.92 ／ 0.67", 10704, 85.3, INK)]
SG_GAIN1 = round((10614 / 6887 - 1) * 100)
SG_GAIN2 = round((10704 / 10614 - 1) * 100, 1)
assert SG_GAIN1 == 54 and SG_GAIN2 == 0.8 and round(999 / 886 - 1, 2) == 0.13

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
    f = Fig(W, "V4-Pro 只有一个 KV 头。比较 TEP4 和 DEP8 要分容量、速度两条轴看。上半张是容量轴，示意：每张卡的 KV 池都装满时，"
               "TEP4 的 4 张卡存的是同一批请求的 4 份，其中 3 份是副本；DEP8 的 8 张卡各存一批不同的请求，"
               "所以同样的显存，DEP 能放的请求上限是 TEP4 的 4 倍，但要同时在跑的请求多到坐满池子才兑现。"
               "下半张是 SGLang 的完整部署，两头都用 DEP：14 台 prefill 各跑 dep4，decode 跑 dep8，输入 4096。"
               "两个 KV 池划分不当时，每卡同时在跑 727 个请求，端到端每张 decode 卡 6887；划平后坐到 886 个，10614，高 54%；"
               "再多给 13% 显存坐到 999 个，只到 10704，算力已到顶")
    y0 = f.header("TEP 换 DEP：容量和速度分开看　——　<tspan font-weight=\"700\">SGLang 两头都用 DEP，座位坐满就换来吞吐</tspan>",
                  "上半张容量轴（示意，每卡 KV 池装满）；下半张 SGLang 实测（14 台 prefill 各 dep4 → decode dep8）",
                  [(PU, "一批不同请求的 KV"), (RD, "同一批 KV 的副本")])
    PH = 560
    py = f.panel(0, y0, W, PH, "① 容量轴：KV 池装满时，存的是几批不同的请求（示意）", PU)
    X0, BW, RH, BH = 210, 700, 34, 24
    yy = py + 46
    for name, sub_, n in (("TEP4", "4 张卡合算一批", 4), ("DEP8", "各算各的", 8)):
        f.t(24, yy + 22, name, INK, True, 19)
        f.t(24, yy + 46, sub_, GY, size=15)
        for c in range(n):
            y = yy + c * RH
            dup = name == "TEP4" and c > 0
            col = RD if dup else PU
            lab = ("批 A 的副本" if dup else "批 A") if name == "TEP4" else "批 %d" % (c + 1)
            f.t(X0 - 12, y + 18, "卡 %d" % c, GY, size=15, anchor="end")
            f.box(X0, y, BW, BH, "#fce8e6" if dup else "#f3e8fd", col, 4)
            f.t(X0 + BW / 2, y + 18, lab, col, True, 14, "middle")
        if name == "TEP4":
            f.t(X0 + BW + 24, yy + 30, "4 张卡只放 1 批", RD, True, 20)
            f.t(X0 + BW + 24, yy + 56, "1 个 KV 头切不开，", GY, size=15)
            f.t(X0 + BW + 24, yy + 78, "每张卡各存一整份", GY, size=15)
        else:
            f.t(X0 + BW + 24, yy + 4 * RH - 2, "8 张卡放 8 批：每卡上限 4 倍", PU, True, 20)
            f.t(X0 + BW + 24, yy + 4 * RH + 24, "4 ＝ TP 路数 4 ÷ KV 头数 1", GY, size=15)
        yy += n * RH + 30
    f.t(X0, y0 + PH - 22, "这是天花板：同时在跑的请求多到坐满 TEP4 的池子，多出的座位才换得来吞吐；光是上下文长没用。",
        GY, size=15)
    f._pan = None
    y1 = y0 + PH + 20
    TH = 330
    py2 = f.panel(0, y1, W, TH, "② SGLang 实测：14 台 prefill 各 dep4 → decode dep8，输入 4096，座位多少换多少吞吐", BL)
    # 拓扑条
    ty = py2 + 30
    f.box(24, ty, 470, 40, "#e8f0fe", BL, 6)
    f.t(24 + 235, ty + 26, "prefill × 14 台，每台 dep4（注意力 DP4 ＋ EP4）", BL, True, 15, "middle")
    f.line(500, ty + 20, 560, ty + 20, BL, 2)
    f.box(566, ty, 400, 40, "#e8f0fe", BL, 6)
    f.t(566 + 200, ty + 26, "decode dep8（跨 2 台，DP8 ＋ EP8）", BL, True, 15, "middle")
    f.t(990, ty + 26, "16 台 GB300，同一 NVL72 域", GY, size=15)
    cols = [("", 24, None), ("每卡同时在跑", 420, "middle"), ("两个 KV 池占用", 650, "middle"),
            ("端到端每张 decode 卡", 920, "middle"), ("出字间隔", 1200, "middle")]
    hy = ty + 84
    for txt, x, a in cols:
        f.t(x, hy, txt, GY, True, 15, a)
    f.line(24, hy + 14, W - 24, hy + 14, LINE, 1, arrow=False)
    for r, (lab, n, pool, e2e, tpot, col) in enumerate(SG_ROWS):
        y = hy + 50 + r * 48
        f.t(24, y, lab, col, True, 17)
        hot = r == 1
        for (txt, x, a), v in zip(cols[1:], [format(n, ","), pool, format(e2e, ","), "%.1f ms" % tpot]):
            f.t(x, y, v, col, hot, 20 if hot and txt.startswith("端到端") else 18, a)
    f.t(920, hy + 50 + 3 * 48 - 12, "+%d%%　　　→　+%.1f%%" % (SG_GAIN1, SG_GAIN2), BL, True, 15, "middle")
    f._pan = None
    yb = f.band(y1 + TH + 20, "ok",
                "座位换吞吐要两个前提：有人来坐，每一步别跟着拖长", [
                    "727 → 886：池子划平、座位多了，有 14 台 prefill 喂着，座位坐得满，端到端涨 %d%%。" % SG_GAIN1,
                    "886 → 999：再多 13%% 座位，只涨 %.1f%%：一步从 77.7 拖到 85.3 ms，多坐的人被抵掉。" % SG_GAIN2])
    yb = f.src(yb + 10,
               "SGLang 部署：gpu/inference/a4x-max/deepseek-v4/SGLANG-V4PRO-RUNBOOK.md:33-34（14 prefill 各 dep4 ＋ dep8 decode、16 节点同一 NVL72 域）、"
               "scripts/prefill-dep4.sh、decode-dep8.sh（--enable-dp-attention，EP 4／8）；三行实测 :851-858（ISL 4096，14 路 × 并发 1050，#1／#3／#7）。",
               "旁证：VLLM-V4PRO-RUNBOOK.md:298-302（num_key_value_heads 1）、:464-465、:476（并发 512：TP4 decode 输出 4,220、出字间隔 46.8 ms；"
               "dep8 为 11,031、11.8 ms；GPU 16 与 20）。",
               "本课推导：容量上限倍数 ＝ TP ÷ min(TP, KV 头数)，上半张是示意，条长不按实测请求数画；+54%% ＝ 10,614 ÷ 6,887，"
               "+0.8%% ＝ 10,704 ÷ 10,614；2.09 倍 ＝ (55,153 ÷ 20) ÷ (21,100 ÷ 16)，1.31 倍 ＝ (11,031 ÷ 8) ÷ (4,220 ÷ 4)。".replace("%%", "%"))
    f.save("fig6-dep-kv.svg", yb + 14)


def _axes(f, x0, y0, w, h, ymax, yticks, fmt):
    for v in yticks:
        yy = y0 + h - h * v / ymax
        f.line(x0, yy, x0 + w, yy, "#eceff1", 1, arrow=False)
        f.t(x0 - 10, yy + 5, fmt(v), GY, size=14, anchor="end")
    f.line(x0, y0 + h, x0 + w, y0 + h, GY2, 1.2, arrow=False)


def fig_dep_tpu():
    f = Fig(W, "tpu-inference 的一个公开 PR，在 TPU v7x-8 上跑 Qwen3.5-397B-A17B-FP8，比较 TP 等于 8 和 DP 注意力等于 8，专家都走 EP。"
               "1K 进 8K 出时，并发 64 打平，128 多 14%，256 多 22%，512 多 62%。8K 进 1K 出时依次多 14%、23%、36%、40%。"
               "并发越高，DP 注意力赢得越多。1K 进 8K 出并发 512 时，输出吞吐 TP 是 4266、DP 是 6914")
    y0 = f.header("TPU 上同一个模型：并发越高，DP 注意力比 TP 赢得越多　——　<tspan font-weight=\"700\">512 并发 +62%</tspan>",
                  "tpu-inference PR #2577 · Qwen3.5-397B-A17B-FP8 · TPU v7x-8（4 颗芯片、8 个 device）· TP=8 对 DP=8，专家都走 EP。纵轴：DP 的输出吞吐比 TP 多多少",
                  [(GR, "1K 进 8K 出（decode 为主）"), (OR, "8K 进 1K 出（prefill 为主）")])
    PH = 440
    py = f.panel(0, y0, W, PH, "并发 → DP 注意力相对 TP 的输出吞吐（端点标的是输出 tok/s，TP → DP）", GR)
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
    for adv, pr, col in ((ADV_A, PR_A, GR), (ADV_B, PR_B, OR)):
        f.path([(px(i), py_(v)) for i, v in enumerate(adv)], col, 2.8, arrow=False)
        for i, v in enumerate(adv):
            f.box(px(i) - 6, py_(v) - 6, 12, 12, col, col, 6)
            if 0 < i < 3:
                f.t(px(i) + (12 if col == GR else -12), py_(v) + (26 if col == GR else -12),
                    "+%d%%" % v, col, True, 16, None if col == GR else "end")
        # 右端点：百分比 ＋ 绝对值
        t, d = pr[3]
        f.t(px(3) + 18, py_(adv[3]) + 2, "+%d%%" % adv[3], col, True, 26)
        f.t(px(3) + 18, py_(adv[3]) + 26, "%s → %s" % (format(t, ","), format(d, ",")), col, size=15)
        # 左端点：绝对值
        t, d = pr[0]
        lab = ("打平　" if adv[0] == 0 else "+%d%%　" % adv[0]) + "%s → %s" % (format(t, ","), format(d, ","))
        f.t(px(0) - 16, py_(adv[0]) + (-8 if col == GR else 6), lab, col, True, 15, "end")
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "PR 同时改了调度：prefill 攒在一起派，免得一条长 prefill 混进 DP 批里拖慢所有 rank", [])
    yb = f.src(yb + 10,
               "github.com/vllm-project/tpu-inference/pull/2577「Enable correct DP attention for hybrid attention＋mamba models」，2026-05-15 合入；"
               "PR 正文表格给输出／总吞吐两个数，本图用输出吞吐现算百分比，与 PR 的「DP Advantage」一列一致。正确性：mmlu_pro DP=8 0.8129 对 TP=8 0.8271（PR 原话：limit=50 时在 1σ 内）。",
               "v7x 一颗芯片对软件显示成 2 个 device，TP=8、DP=8 按 device 数。KV 头数取自 HF Qwen/Qwen3.5-397B-A17B-FP8 config.json（num_key_value_heads 2，60 层里每 4 层一层全注意力）。"
               "「复制 4 份卡住 batch」是本课解读，PR 没有单独归因。")
    f.save("fig6-dep-tpu.svg", yb + 14)


# ── K3 的 KDA 状态池：DEP8 每 rank 一整池，TP8 按头切（示意）──────────────
HEADS, SLOTS, NR = 16, 4, 8                    # 头数、请求槽数都是示意，不按实际
assert HEADS % NR == 0


def fig_dep_state():
    f = Fig(W, "Kimi K3 的 69 层 KDA 线性注意力，每个请求一块固定状态，按头存。左边 DEP8：8 个 rank 各算各的请求，"
               "每个 rank 的状态池都是完整的一整池，所有头、所有请求槽都要存，两次加载都 OOM。"
               "右边 TP8：一池状态按头切给 8 张卡，每张卡只存自己那几个头，8 张卡拼起来才是一整池。"
               "TP 按头切这一维，DP、EP、DCP 都不切")
    y0 = f.header("KDA 状态按头存：DEP 每个 rank 复制一整池，TP 每卡只存自己的头",
                  "Kimi K3：69 层 KDA，每个请求一块固定状态。一格 ＝ 一个请求槽在一个头上的状态；头数、槽数是示意，不按实际",
                  [(PU, "KDA 状态"), (RD, "加载时 OOM 的卡")])
    PW, GAP, PH = 690, 20, 344
    CW, CH, CX, CY = 7, 14, 140, 104            # 格子宽高；卡片框宽高
    for side, (name, col) in enumerate((("DEP8：8 个 rank 各开一整池", RD), ("TP8：一池按头切给 8 张卡", GR))):
        x0 = side * (PW + GAP)
        py = f.panel(x0, y0, PW, PH, name, col)
        for r in range(NR):
            cx = x0 + 20 + (r % 4) * (CX + 30)
            cy = py + 22 + (r // 4) * (CY + 26)
            f.box(cx, cy, CX, CY, "none", RD if side == 0 else LINE, 6, 1.6 if side == 0 else 1)
            f.t(cx + 12, cy + 24, ("rank %d" if side == 0 else "卡 %d") % r, INK, True, 15)
            gx, gy = cx + (CX - HEADS * CW) / 2.0, cy + 40
            per = HEADS // NR
            for h in range(HEADS):
                mine = side == 0 or r * per <= h < (r + 1) * per
                for s_ in range(SLOTS):
                    if mine:
                        f.box(gx + h * CW, gy + s_ * CH, CW - 1, CH - 1, "#f3e8fd", PU, 1, 1)
                    else:
                        f.box(gx + h * CW, gy + s_ * CH, CW - 1, CH - 1, "none", "#e0e3e7", 1, 1)
        msg, mcol = (("每个 rank 都要装下全部头的状态：两次加载都 OOM", RD) if side == 0
                     else ("每卡只亮自己那几列，8 张卡拼起来才是一整池", GR))
        f.t(x0 + 20, y0 + PH - 22, msg, mcol, True, 16)
        f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "TP 按头切这一维；DP、EP、DCP 都不动它", [
        "所以 DEP 开得再宽，每个 rank 还是一整池；本课推导，与官方说法一致。"])
    yb = f.src(yb + 10,
               "K3 共 93 层、69 层 KDA，DEP8 两次加载 OOM、官方改推 TP 与 DCP：SGLANG-K3-RUNBOOK.md（见第六、七节与文末台账）。",
               "「DP、EP、DCP 都不切」状态池、attention-TP 8 时「状态池 8 路切分」为 runbook 引的官方说法；「按头切」为本课推导；格子数只表示整池与按头切的对比。")
    f.save("fig6-dep-state.svg", yb + 14)


def fig_tbo():
    """双微批（DeepSeek 叫 dual-batch overlap，SGLang 叫 TBO）：拆两半，一个传、一个算。示意。"""
    f = Fig(W, "双微批示意。上面一整批：每层先算注意力，再把 token 分发给专家所在的卡，算完专家再汇回来，"
               "分发和汇回的时候计算单元空等。下面拆成 A、B 两个微批：A 在传的时候 B 在算，计算那一行一直排满，"
               "整趟前向提前结束。时长是示意，通信取同一批计算时长的八成")
    y0 = f.header("双微批：传的时候，别让算力干等",
                  "两层 MoE 的一趟前向。每层四步：算注意力 → 分发给专家 → 算专家 → 汇回。时长是示意，不按实测",
                  [(BL, "计算"), (OR, "卡间通信（all-to-all）")])
    U, X0, RH = 62, 170, 46                       # 一个单位多宽；时间轴起点；行高
    def lane(py, rows, segs, total):
        for i, name in enumerate(rows):
            f.t(X0 - 14, py + i * (RH + 14) + 29, name, INK, True, 16, anchor="end")
            f.box(X0, py + i * (RH + 14), total * U, RH, "none", "#e0e3e7", 4, 1, dash="4 4")
        for row, t0, dt, lab, col in segs:
            yy = py + row * (RH + 14)
            f.box(X0 + t0 * U, yy, dt * U - 3, RH, "#e8f0fe" if col == BL else "#fef3e0", col, 4, 1.4)
            f.t(X0 + t0 * U + dt * U / 2 - 1.5, yy + 29, lab, col, True, 14, anchor="middle")
    steps = ("注意力", "分发", "专家", "汇回")
    # 上：一整批。计算 2 单位，通信 1.6 单位（示意）
    PH = 158
    py = f.panel(0, y0, W, PH, "不拆：传的时候，计算单元空等", RD)
    segs, t = [], 0.0
    for _ in range(2):
        for k, st in enumerate(steps):
            dt = 2.0 if k % 2 == 0 else 1.6
            segs.append((k % 2, t, dt, st, BL if k % 2 == 0 else OR))
            t += dt
    lane(py + 14, ("算", "传"), segs, 14.4)
    f.t(X0 + 14.4 * U + 14, py + 48, "结束", RD, True, 16)
    f._pan = None
    # 下：两个微批。每半批计算 1、通信 0.8
    y1 = y0 + PH + 18
    py = f.panel(0, y1, W, PH, "拆成 A、B 两半：A 传的时候 B 在算，计算那一行排满", GR)
    segs = []
    for i in range(8):
        mb = "A" if i % 2 == 0 else "B"
        st = steps[(i // 2) % 2 * 2]              # 注意力、注意力、专家、专家……
        segs.append((0, i, 1.0, mb + " " + st, BL))
        segs.append((1, i + 1, 0.8, mb + " " + steps[(i // 2) % 2 * 2 + 1], OR))
    lane(py + 14, ("算", "传"), segs, 14.4)
    f.t(X0 + 8.8 * U + 14, py + 48, "结束", GR, True, 16)
    f.box(X0 + 8.8 * U + 70, py + 14, (14.4 - 8.8) * U - 70, 2 * RH + 14, "#f1f3f4", "none", 4, 0)
    f.t(X0 + (8.8 + 14.4) / 2 * U + 35, py + 14 + RH + 12, "省下的这段", GR, True, 16, anchor="middle")
    f._pan = None
    yb = f.band(y1 + PH + 20, "warn", "前提：每半批还得够大", [
        "拆开后每半批更难把计算喂饱。SGLang 实测 decode 每卡 32 个 token 时反而慢 27%，256 个时快 25.5%。"])
    yb = f.src(yb + 10,
               "做法：DeepSeek open-infra-index 第六天（Inference System Overview）：prefill 两个微批交替，一个的通信藏进另一个的计算；decode 各步时长不均，把注意力拆成两段，走 5 段流水。",
               "数字：LMSYS 博客 2025-05-05（大规模 EP）的 TBO 消融；vLLM vllm/config/parallel.py 的 enable_dbo、dbo_decode_token_threshold（默认 32）。时长比例是示意，本课自定。")
    f.save("fig6-tbo.svg", yb + 14)


def fig_eplb():
    """EPLB：热门专家多复制一份放到别的卡，再重排座位。8 个专家、4 张卡，负载数字是示意。"""
    f = Fig(W, "EPLB 示意。左边：8 个专家按编号两个一组放 4 张卡，专家 0 最热门，卡 0 的柱子最高，"
               "专家那一步所有卡都要等它。右边：每卡多一个空位，热门的专家 0 和专家 7 各复制一份，"
               "token 平分给两份，再重新排座位，最高的柱子从 70 降到 52.5。数字是示意")
    y0 = f.header("EPLB：热门专家多备一份，再重新排座位",
                  "8 个专家、4 张卡。柱高 ＝ 这张卡这一步要算的 token，一段一个专家；数字是示意",
                  [(OR, "热门专家（及其副本）"), (BL, "其他专家"), (RD, "所有卡等到这条线")])
    PW, GAP, PH = 690, 20, 430
    SC, BW, BASE = 4.0, 92, 330                    # 每个 token 多高；柱宽；柱底离面板顶
    before = [[("专家 0", 60, True), ("专家 1", 10, False)], [("专家 2", 15, False), ("专家 3", 15, False)],
              [("专家 4", 20, False), ("专家 5", 20, False)], [("专家 6", 25, False), ("专家 7", 35, True)]]
    after = [[("专家 0", 30, True), ("专家 4", 20, False)], [("专家 0′", 30, True), ("专家 5", 20, False)],
             [("专家 6", 25, False), ("专家 7", 17.5, True), ("专家 1", 10, False)],
             [("专家 7′", 17.5, True), ("专家 2", 15, False), ("专家 3", 15, False)]]
    for side, (name, col, cards, msg) in enumerate((
            ("开 EPLB 之前：按编号放", RD, before, "最慢的卡 70，平均 50"),
            ("开 EPLB 之后：每卡多一个空位，热门的复制一份", GR, after, "最慢的卡 52.5，平均还是 50"))):
        x0 = side * (PW + GAP)
        py = f.panel(x0, y0, PW, PH, name, col)
        yb0 = py + BASE
        top = 0
        for c, segs in enumerate(cards):
            bx = x0 + 70 + c * (BW + 60)
            y = yb0
            for lab, v, hot in segs:
                h = v * SC
                f.box(bx, y - h, BW, h - 2, "#fef3e0" if hot else "#e8f0fe", OR if hot else BL, 3, 1.3)
                if h >= 26:
                    f.t(bx + BW / 2, y - h / 2 + 5, "%s · %g" % (lab, v), OR if hot else BL, True, 13, anchor="middle")
                y -= h
            tot = sum(v for _, v, _ in segs)
            top = max(top, tot)
            f.t(bx + BW / 2, yb0 + 22, "卡 %d：%g" % (c, tot), INK, True, 15, anchor="middle")
        ly = yb0 - top * SC
        f.line(x0 + 50, ly, x0 + PW - 20, ly, RD, 1.6, dash="6 4", arrow=False)
        f.t(x0 + PW - 22, yb0 + 56, msg, RD if side == 0 else GR, True, 16, anchor="end")
        f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "加的是空位，换的是等待", [
        "副本占显存；token 怎么分给两份、多久重排一次，各家实现不同。"])
    yb = f.src(yb + 10,
               "做法：DeepSeek EPLB 仓库 README（按负载估计算出复制与摆放方案；负载估计常用历史统计的滑动平均，不在仓库范围内）；DeepSeek-V3 技术报告 sec. 3.4。",
               "负载数字与「token 平分给两份」均为示意，本课自定；8 专家 4 卡只为画得下。")
    f.save("fig6-eplb.svg", yb + 14)


fig_dep_kv()
fig_dep_tpu()
fig_dep_state()
fig_tbo()
fig_eplb()
