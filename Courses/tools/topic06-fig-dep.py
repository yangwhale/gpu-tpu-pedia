# -*- coding: utf-8 -*-
r"""专题六 · 第六节「切专家与注意力：TEP 和 DEP」的三张静态图。

回答的问题
  · fig6-dep-kv   V4-Pro 从 TP4 换 DEP8，每卡 2.09× 从哪来？多装了请求，还是每一步变短？（Little 定律对账）
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
  · fig6-dep-kv 不画「放下 4 倍请求」的格子图：那是容量上限，实测在飞请求反而少（197 → 130），只在图里一句话带过。
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
IN_TP4, IN_DEP = OUT_TP4 * TPOT_TP4 / 1000, OUT_DEP * TPOT_DEP / 1000   # Little 定律：在飞请求
PD_TP4, PD_DEP = OUT_TP4 / 4, OUT_DEP / 8      # 每张 decode 卡的输出
assert round(IN_TP4) == 197 and round(IN_DEP) == 130, (IN_TP4, IN_DEP)
assert round(PD_TP4) == 1055 and round(PD_DEP) == 1379 and round(PD_DEP / PD_TP4, 2) == 1.31
assert round(OUT_DEP / OUT_TP4, 2) == 2.61 and round((1 - IN_DEP / IN_TP4) * 100) == 34

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
    f = Fig(W, "V4-Pro 只有一个 KV 头。同样 3 台 TP4 prefill、并发 512，只把 decode 从 TP4 换成 DEP8。"
               "上半张图画这一步每张卡要读的 KV：TP4 的 4 张卡各读一遍全部约 197 个在飞请求的 KV，其中 3 遍是重复读；"
               "DEP8 的 8 张卡各读自己那约 16 个请求。下半张用 Little 定律对账：在飞请求等于每秒输出字数乘出字间隔，"
               "TP4 是 4220 乘 0.0468 秒约 197 个，DEP8 是 11031 乘 0.0118 秒约 130 个，DEP8 反而少；"
               "出字间隔从 46.8 毫秒降到 11.8 毫秒。decode 卡从 4 张变 8 张，每张 decode 卡输出 1055 到 1379，1.31 倍；"
               "按全部卡算总吞吐每卡 1319 到 2758，2.09 倍")
    y0 = f.header("V4-Pro 换 DEP，赢在每一步读得少　——　<tspan font-weight=\"700\">步时 46.8 → 11.8 ms</tspan>",
                  "同样 3 台 TP4 prefill、并发 512，只换 decode。条长 ＝ 这一步这张卡要读的 KV，按卡上在飞的请求数画（本课推导）",
                  [(PU, "自己那批请求的 KV"), (RD, "同一份 KV 的重复读")])
    PH = 560
    py = f.panel(0, y0, W, PH, "这一步，每张卡要读多少 KV", PU)
    X0, BW, RH, BH = 210, 860, 34, 24
    unit = BW / IN_TP4
    yy = py + 46
    for name, sub_, n, per, extra in (("TP4", "4 张卡合算一批", 4, IN_TP4, "＋ 每层一次 AllReduce"),
                                      ("DEP8", "各算各的", 8, IN_DEP / 8, None)):
        f.t(24, yy + 22, name, INK, True, 19)
        f.t(24, yy + 46, sub_, GY, size=15)
        for c in range(n):
            y = yy + c * RH
            dup = name == "TP4" and c > 0
            col = RD if dup else PU
            f.t(X0 - 12, y + 18, "卡 %d" % c, GY, size=15, anchor="end")
            f.box(X0, y, per * unit, BH, "#fce8e6" if dup else "#f3e8fd", col, 4)
        if name == "TP4":
            f.t(X0 + BW + 20, yy + 30, "约 %d 个请求" % round(IN_TP4), RD, True, 20)
            f.t(X0 + BW + 20, yy + 56, "每张卡读一遍，", GY, size=15)
            f.t(X0 + BW + 20, yy + 78, "4 遍里 3 遍是重复", GY, size=15)
            f.t(X0 + BW + 20, yy + 106, extra, GY, size=15)
        else:
            f.t(X0 + 16 + per * unit, yy + 4 * RH - 2, "每张卡约 %d 个请求，各读一遍" % round(IN_DEP / 8), PU, True, 20)
        yy += n * RH + 30
    f.t(X0, y0 + PH - 22, "副本让出的显存，理论上能放的请求是原来的 4 倍（TP 度数 4 ÷ KV 头数 1）；实测在飞的请求没变多，见下表。",
        GY, size=15)
    f._pan = None
    # ── Little 定律对账 ──
    y1 = y0 + PH + 20
    TH = 250
    py2 = f.panel(0, y1, W, TH, "Little 定律对账：在飞请求 ＝ 每秒输出字数 × 出字间隔", BL)
    cols = [("", 24, None), ("decode 卡", 300, "middle"), ("输出 tok/s", 480, "middle"), ("出字间隔", 660, "middle"),
            ("在飞请求", 860, "middle"), ("每张 decode 卡输出", 1120, "middle")]
    rows = [("TP4", ["4", "4,220", "46.8 ms", "约 %d" % round(IN_TP4), format(round(PD_TP4), ",")], INK),
            ("DEP8", ["8", "11,031", "11.8 ms", "约 %d" % round(IN_DEP), format(round(PD_DEP), ",")], INK),
            ("变化", ["×2", "×%.2f" % (OUT_DEP / OUT_TP4), "÷%.0f" % (TPOT_TP4 / TPOT_DEP),
                      "少 %d%%" % round((1 - IN_DEP / IN_TP4) * 100), "×%.2f" % (PD_DEP / PD_TP4)], BL)]
    hy = py2 + 50
    for txt, x, a in cols:
        f.t(x, hy, txt, GY, True, 15, a)
    f.line(24, hy + 14, W - 24, hy + 14, LINE, 1, arrow=False)
    for r, (lab, vals, col) in enumerate(rows):
        y = hy + 52 + r * 50
        f.t(24, y, lab, col, True, 18)
        for (txt, x, a), v in zip(cols[1:], vals):
            hot = col == BL and txt in ("出字间隔", "在飞请求")
            f.t(x, y, v, RD if (hot and txt == "在飞请求") else col, col == BL, 22 if hot else 18, a)
    f._pan = None
    yb = f.band(y1 + TH + 20, "ok",
                "在飞的请求反而少了 34%：收益不来自多装，来自每一步变短，再加上 decode 卡翻倍", [
                    "按全部卡（含 12 张 prefill 卡）算总吞吐，每卡 %s → %s，是 %.2f 倍；按 decode 卡算输出只有 %.2f 倍。"
                    % (format(round(TP4_PER), ","), format(round(DEP_PER), ","), RATIO, PD_DEP / PD_TP4),
                    "两边投机解码接受率相近（35% 上下），步时变短不是投机带来的。"])
    yb = f.src(yb + 10,
               "V4-Pro 单 KV 头（num_key_value_heads 1，CSA／HCA 交错，不是 MLA）：gpu/inference/a4x-max/deepseek-v4/VLLM-V4PRO-RUNBOOK.md:298-302。",
               "实测：同文件 :449、:464-465（并发 512：TP4 decode 总吞吐 21,100、输出 4,220、出字间隔 46.8 ms；dep8 为 55,153、11,031、11.8 ms）；"
               "GPU 数 16 和 20 见 :476；投机接受率 34% 与 35.7% 见 :480。",
               "本课推导：在飞请求 ＝ 输出 tok/s × 出字间隔（Little 定律）；条长按卡上在飞请求数画，假设两边每个请求的 KV 一样长（同为 4K 进 1K 出）；"
               "TP4 里哪张卡算「原件」是示意。容量上限倍数 ＝ TP ÷ min(TP, KV 头数)，与 vLLM 文档「KV 被复制 tp_size / H 次」一致；"
               "MLA 模型 TEP8 是 8 倍，Qwen3.5-397B 全注意力层 2 个 KV 头、TEP8 是 4 倍，8 个 KV 头的 GQA 模型 TEP8 是 1 倍。")
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


fig_dep_kv()
fig_dep_tpu()
fig_dep_state()
fig_tbo()
