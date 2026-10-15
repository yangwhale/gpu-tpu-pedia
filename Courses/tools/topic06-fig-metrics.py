# -*- coding: utf-8 -*-
r"""专题六 · 第二节「先校准尺子：指标与口径」的四张静态图。

fig6-metrics.svg —— 回答：TTFT、TPOT、ITL、每用户 tok/s 各量的是时间线上哪一段，含不含排队、含不含首字？
  · 定义：vLLM vllm/benchmarks/serve.py（TPOT ＝ (latency − ttft) ÷ (output_len − 1)，按请求算；
    ITL 是逐个 chunk 的间隔列表，投机解码时一个 chunk 可以带好几个 token）；
    NVIDIA NIM 的 benchmarking 指标页（TTFT 含排队、prefill、网络；ITL ＝ (e2e − TTFT) ÷ (输出 token − 1)，
    并写明 "also known as TPOT"）。两家对同一个词的口径不同，图里两种都写。
  · 2026-10 改稿：只留时间线并放大；两家工具口径、Gemma4-31B 实测锚点两块表移进正文 2.1（正文本来就写了）。
    chunk 宽度按它带的字数画，带 3 个字那一次间隔最长 —— 这就是 ITL 的 p99 比 TPOT 长的来由。
    Gemma 锚点常数 G_C1／G_C256 保留在下面（traps 的峰值稳态仍用 G_C256 对应的 4,495 / 6,144）。
  · 刻意没画：时间刻度（时间线是示意）、网络往返的单独一段（并进 TTFT 里说）。

fig6-pareto.svg —— 回答：为什么不能光比「每卡吞吐」？
  · InferenceX GB300 V4-Pro FP4 8K/1K（Dynamo ＋ vLLM，PD 分离，output ÷ decode GPU）：
    68 → 9,759、121 → 3,816、173 → 895（gpu/inference/a4x-max/deepseek-v4/VLLM-V4PRO-RUNBOOK.md:683-695）。
  · PyTorch 博客 2026-06-23「Serving DeepSeek-V4 on GB300 with SGLang」：约 50 tok/s/user 处 ~11,200 tok/s/GPU
    （2026-06 MTP 曲线），Day-0 无 MTP 曲线约 2,200。
  · 本仓库实测：SGLang dep8 conc600 ＝ 9,168（output ÷ decode GPU），TPOT 59–61 ms ≈ 每用户 17 tok/s（1000 ÷ 60）
    （gpu/inference/a4x-max/deepseek-v4/SGLANG-V4PRO-RUNBOOK.md:595、723-725）。
  · 刻意没画：把这几个点连成一条线（它们分属不同框架、不同配置）。只有 InferenceX 那三个点是同一条曲线。

fig6-rulers.svg —— 回答：同一次测量，换分子分母能差多少？
  · SGLANG-V4PRO-RUNBOOK.md:794-809 的 dep16：聚合 output 84,321；output ÷ 16 decode GPU ＝ 5,270；
    tput_per_gpu ＝ (in+out) ÷ 64 ＝ 11,880。另两把尺子（output ÷ 64、(in+out) ÷ 16）是本课推导，
    只为把「分子 × 分母」的四种组合排全，没有哪个工具这样报。
  · 自洽检查：8K 进 1K 出，in+out 约是 output 的 9 倍；11,880 × 64 ＝ 760,320 ≈ 9 × 84,321。

fig6-traps.svg —— 回答：除了分母，还有哪几个坑会让同一套系统报出差很远的数？
  · 2026-10 R3：拿掉「池子比例没调」那格 —— 两个 KV 池要到 3.4 才讲，第二节用它举例太早（6,887 → 10,614 留在 fig6-swa-ratio）。
  · 峰值 vs 稳态：同上 #7（端到端 10,704、峰值 12,070）；Gemma4-31B README:460（4,495 vs peak 6,144）。
  · 温度：VLLM-V4PRO-RUNBOOK.md:394-420（接受率 1.16% vs 34%）。只对 greedy 草稿成立：
    gpu/inference/a4x-max/kimi-k3/VLLM-K3-RUNBOOK.md:226-230 的 draft 是 probabilistic，官方就用 1.0。
  · 预热：SGLANG-V4PRO-RUNBOOK.md:480-494（8,520 → 9,118）；VLLM-V4PRO-RUNBOOK.md:431-433（13,585 → 27,803）。
  · 闭环：SGLANG-V4PRO-RUNBOOK.md:478、1135（闭环 −13%）。
"""

from topic03_draw import Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE, wpx

W = 1400

# ── Gemma4-31B 锚点 ─────────────────────────────────────────────
G_C1 = dict(ttft=86, tpot=35, tps=28)
G_C256 = dict(ttft=7756, tpot=49, out=4495, peak=6144)
assert round(1000 / G_C1["tpot"]) == 29          # 1000 ÷ TPOT 与实测 28 tok/s 只差 1 —— 实测含首字那段
assert G_C256["ttft"] / G_C1["ttft"] > 90        # 同一个 prompt 长度，TTFT 差 90 倍：多出来的是排队

# ── Pareto 点（tok/s/user, tok/s/GPU）──────────────────────────
IX = [(68, 9759), (121, 3816), (173, 895)]        # InferenceX，Dynamo ＋ vLLM
PT_JUNE, PT_DAY0 = (50, 11200), (50, 2200)       # PyTorch 博客，SGLang
RB_TPOT = 60                                      # ms，conc600 的 59–61 取中
RB = (1000 / RB_TPOT, 9168)
assert round(RB[0]) == 17 and round(PT_JUNE[1] / PT_DAY0[1], 1) == 5.1
assert round(PT_JUNE[1] / PT_JUNE[0]) == 224      # 每卡同时服务约 224 个用户（runbook:1088 同算）

# ── 四把尺子（dep16）────────────────────────────────────────────
OUT, DEC, ALL = 84321, 16, 64
INOUT = 11880 * ALL
assert INOUT == 760320 and abs(INOUT / OUT - 9) < 0.02
RULERS = [("output ÷ decode GPU（16 张）", round(OUT / DEC), BL, "InferenceX 的 output_tput_per_gpu"),
          ("(输入＋输出) ÷ 全部 GPU（64 张）", round(INOUT / ALL), GR, "InferenceX 的 tput_per_gpu"),
          ("output ÷ 全部 GPU（64 张）", round(OUT / ALL), GY2, "本课推导：把 prefill 卡也算进分母"),
          ("(输入＋输出) ÷ decode GPU（16 张）", round(INOUT / DEC), GY2, "本课推导：没人这么报，排全四种组合")]
assert [r[1] for r in RULERS] == [5270, 11880, 1318, 47520]
SPREAD = RULERS[3][1] / RULERS[2][1]
assert round(SPREAD) == 36
OFFICIAL = 11200

# ── 四个坑 ──────────────────────────────────────────────────────
PEAK = [("V4-Pro dep8 · 端到端", 10704, "峰值", 12070), ("Gemma4-31B · 并发 256", 4495, "峰值", 6144)]
assert round(PEAK[0][3] / PEAK[0][1] - 1, 3) == 0.128 and round(PEAK[1][3] / PEAK[1][1] - 1, 2) == 0.37
TEMP = dict(acc=(1.16, 34), out=(944, 2911), tpot=(232, 48.6))
assert round(TEMP["acc"][1] / TEMP["acc"][0]) == 29 and round(TEMP["out"][1] / TEMP["out"][0], 1) == 3.1
WARM = [("SGLang dep8", 8520, "热", 9118), ("vLLM 2P1D", 13585, "热", 27803)]
assert round((WARM[0][3] / WARM[0][1] - 1) * 100, 1) == 7.0 and round(WARM[1][3] / WARM[1][1], 2) == 2.05


def bracket(f, x1, x2, y, col, lab, size=14):
    f.line(x1, y, x2, y, col, 2, arrow=False)
    f.line(x1, y - 7, x1, y + 7, col, 2, arrow=False)
    f.line(x2, y - 7, x2, y + 7, col, 2, arrow=False)
    f.t((x1 + x2) / 2, y + 24, lab, col, True, size, "middle")


def fig_metrics():
    f = Fig(W, "一个请求的时间线，示意。请求进门后先排队，再做 prefill，吐出第一个字；之后 decode 一个 chunk 一个 chunk 往外推字，"
               "有的 chunk 一次带两三个字，带的字越多，这一次间隔越长。TTFT 是进门到第一个字，含排队和 prefill。"
               "ITL 是相邻两个 chunk 之间的每一次间隔。TPOT 是首字之后的总时长除以字数减一，是每个请求的平均值。"
               "每用户每秒多少字约等于一千除以 TPOT 的毫秒数")
    y0 = f.header("四个时间指标各量哪一段　——　<tspan font-weight=\"700\">先问含不含排队、含不含首字</tspan>",
                  "一个请求的时间线（示意）。每个绿块是一次到手的一段字，宽度按它带的字数画",
                  [(GY2, "排队"), (OR, "prefill"), (GR, "decode 推出的一段字（chunk）")])
    T0, QW, PW_ = 30, 180, 210
    U = 60                                   # 每个字占的宽度（示意）
    CH = [1, 1, 2, 1, 3, 1, 2]               # 每个 chunk 带几个字
    N = 1 + sum(CH)                          # 首字 ＋ decode 推出的字
    assert N == 12
    XP = T0 + QW
    XD = XP + PW_
    xs, x = [], XD
    for n in CH:
        xs.append(x)
        x += n * U
    ends = [XD - 6] + [xx + n * U - 6 for xx, n in zip(xs, CH)]
    assert ends[-1] < 1110
    PH = 30 + 346
    py = f.panel(0, y0, W, PH, "一个请求的时间线", INK)
    RY, RH = py + 96, 64
    f.box(T0, RY, QW - 6, RH, "#f1f3f4", GY2, 4, dash="5,4")
    f.t(T0 + (QW - 6) / 2, RY + 40, "排队", GY, True, 18, "middle")
    f.box(XP, RY, PW_ - 6, RH, OR, OR, 4)
    f.t(XP + (PW_ - 6) / 2, RY + 40, "prefill", "#ffffff", True, 18, "middle")
    for xx, n in zip(xs, CH):
        f.box(xx, RY, n * U - 6, RH, GR, GR, 4)
        f.t(xx + (n * U - 6) / 2, RY + 40, "%d 字" % n, "#ffffff", True, 17, "middle")
    for i, ex in enumerate(ends):
        f.p.append('<circle cx="%.1f" cy="%.1f" r="7" fill="%s"/>' % (ex, RY - 16, RD if i == 0 else INK))
    f.t(XD - 6, RY - 34, "第一个字", RD, True, 16, "middle")
    f.line(T0, RY - 52, T0, RY - 4, GY2, 1.6)
    f.t(T0, RY - 58, "请求进门", GY, True, 15)
    BY = RY + RH + 34
    bracket(f, T0, XD - 6, BY, RD, "TTFT：进门 → 第一个字", 17)
    f.t((T0 + XD - 6) / 2, BY + 48, "含排队、prefill、网络", RD, False, 15, "middle")
    bracket(f, ends[4], ends[5], BY, BL, "ITL：这一次间隔", 17)
    f.t((ends[4] + ends[5]) / 2, BY + 48, "带 3 个字，所以最长", BL, False, 15, "middle")
    BY2 = BY + 104
    bracket(f, XD - 6, ends[-1], BY2, GR, "TPOT ＝ 这一段 ÷ (字数 − 1)，每个请求一个平均数", 17)
    RX = ends[-1] + 30
    f.t(RX, RY + 26, "每用户 tok/s", INK, True, 17)
    f.t(RX, RY + 52, "≈ 1000 ÷ TPOT(ms)", INK, False, 16)
    f.t(RX, BY2 + 6, "共 %d 个字" % N, GY, size=15)
    f._pan = None
    yb = f.src(py - 30 + PH + 4,
               "vLLM 定义：vllm/benchmarks/serve.py（TPOT ＝ (latency − ttft) ÷ (output_len − 1)；ITL 逐 chunk，"
               "源码注释写明 multiple output tokens may be bundled together）。",
               "NIM 定义：NVIDIA NIM for LLMs 的 benchmarking 指标页（TTFT 含排队、prefill、网络；ITL "
               "also known as TPOT）。AIPerf 沿用同一套。两家对同一个词的口径不同，正文第 2.1 节对照。",
               "⚠️ 时间线是示意：chunk 带几个字、各段长短都是为了看得清，不对应任何实测；"
               "绿块宽度按字数画，表示这几个字是攒在一起才到手的。")
    f.save("fig6-metrics.svg", yb + 6)


def fig_pareto():
    f = Fig(W, "横轴是每个用户每秒拿到多少字，纵轴是每张 GPU 每秒出多少字，都是 GB300 上跑 DeepSeek-V4-Pro、8K 进 1K 出。"
               "InferenceX 的 vLLM 曲线三个点：每用户 68 时每卡 9,759，121 时 3,816，173 时只剩 895。"
               "PyTorch 博客的 SGLang 在每用户 50 处报每卡约 11,200，同一位置首发时只有约 2,200。"
               "本仓库实测的 9,168 落在每用户约 17 处。同一套系统沿着曲线往左走，每卡吞吐就往上涨；"
               "所以先定交互档，再在这一档上比吞吐")
    y0 = f.header("吞吐和交互性是一条曲线　——　<tspan font-weight=\"700\">先定档，再比吞吐</tspan>",
                  "GB300 · DeepSeek-V4-Pro · 8K 进 1K 出 · 纵轴按 output ÷ decode GPU 计（11,200 的口径官方未写明，见出处）",
                  [(GR, "InferenceX：Dynamo ＋ vLLM（同一条曲线）"), (BL, "PyTorch 博客：SGLang"),
                   (OR, "本仓库实测：SGLang dep8")])
    PH = 520
    py = f.panel(0, y0, W, PH, "每张 GPU 的吞吐 vs 每个用户的速度", INK)
    X0, Y0, PW, PHH = 150, py + 420, 1000, 360
    XM, YM = 200, 12000

    def P(u, g):
        return X0 + PW * u / XM, Y0 - PHH * g / YM
    f.line(X0, Y0, X0 + PW + 20, Y0, GY2, 1.4)
    f.line(X0, Y0, X0, Y0 - PHH - 20, GY2, 1.4)
    for u in range(0, 201, 50):
        x, _ = P(u, 0)
        f.line(x, Y0, x, Y0 + 6, GY2, 1, arrow=False)
        f.t(x, Y0 + 24, str(u), GY, size=13, anchor="middle")
    for g in range(0, 12001, 3000):
        _, y = P(0, g)
        f.line(X0 - 6, y, X0, y, GY2, 1, arrow=False)
        if g:
            f.line(X0, y, X0 + PW, y, LINE, 1, dash="3,4", arrow=False)
        f.t(X0 - 12, y + 5, format(g, ","), GY, size=13, anchor="end")
    f.t(X0 + PW / 2, Y0 + 50, "每个用户每秒拿到的字（tok/s/user）→ 越右越跟手", INK, True, 14, "middle")
    f.t(X0 - 12, Y0 - PHH - 30, "每张 GPU 每秒出的字 ↑", INK, True, 14)
    # 50 档竖线
    x50, _ = P(50, 0)
    f.line(x50, Y0, x50, Y0 - PHH - 6, PU, 1.6, dash="6,4", arrow=False)
    f.t(x50 + 8, Y0 - PHH + 10, "先定档：每用户 50 tok/s", PU, True, 14)
    # InferenceX 曲线
    pts = [P(u, g) for u, g in IX]
    f.path(pts, GR, 2.4, arrow=False)
    for (u, g), (x, y) in zip(IX, pts):
        f.p.append('<circle cx="%.1f" cy="%.1f" r="7" fill="%s"/>' % (x, y, GR))
        f.t(x + 12, y - 10, "%s @ %d" % (format(g, ","), u), GR, True, 14)
    # PyTorch 两点
    for (u, g), lab in ((PT_JUNE, "11,200（2026-06，开 MTP）"), (PT_DAY0, "2,200（首发，无 MTP）")):
        x, y = P(u, g)
        f.p.append('<circle cx="%.1f" cy="%.1f" r="7" fill="%s"/>' % (x, y, BL))
        f.t(x + 12, y + 5, lab, BL, True, 14)
    # 本仓库
    x, y = P(*RB)
    f.p.append('<circle cx="%.1f" cy="%.1f" r="7" fill="%s"/>' % (x, y, OR))
    f.t(x + 12, y + 26, "9,168 @ ≈ 17", OR, True, 14)
    f.t(x + 12, y + 46, "（TPOT 约 60 ms）", OR, size=13)
    f.t(X0 + PW - 10, Y0 - PHH + 50, "同一档上：11,200 是 2,200 的 5.1 倍", BL, True, 14, "end")
    f.t(X0 + PW - 10, Y0 - PHH + 74, "每卡约 224 个用户同时在线（11,200 ÷ 50）", GY, size=13, anchor="end")
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "单报一个每卡吞吐，等于没报", [
        "同一条 vLLM 曲线上，每用户从 68 提到 173，每卡吞吐从 9,759 掉到 895。报吞吐必须连同交互档一起报。",
        "本仓库的 9,168 落在每用户约 17 的低档，跟 50 档上的 11,200 不能直接比高低。",
    ])
    yb = f.src(yb + 10,
               "InferenceX 三点：gpu-tpu-pedia gpu/inference/a4x-max/deepseek-v4/VLLM-V4PRO-RUNBOOK.md:683-695"
               "（GB300 Dynamo ＋ vLLM，V4-Pro FP4，8K/1K，output ÷ decode GPU）。",
               "PyTorch 博客 2026-06-23「Serving DeepSeek-V4 on GB300 with SGLang: 5x Higher Throughput at the Same "
               "Interactivity Since Day-0」：~11,200 tok/s/GPU at roughly 50 tok/s/user，Day-0 无 MTP 曲线约 2,200。",
               "⚠️ 11,200 用的是 output ÷ decode GPU 还是 (in＋out) ÷ 全部 GPU，公开材料没钉死"
               "（SGLANG-V4PRO-RUNBOOK.md:745）。本图按前者画，若是后者，这个点要往下挪。",
               "本仓库实测 9,168：SGLANG-V4PRO-RUNBOOK.md:595（conc600，TPOT 59–61 ms）、723-725。横坐标 1000 ÷ 60 ≈ 17 是本课推导。",
               "goodput：DistServe（arXiv 2401.09670）。")
    f.save("fig6-pareto.svg", yb + 14)


IN_PER_OUT = round(INOUT / OUT - 1)               # 8K 进 1K 出：每个输出 token 对应约 8 个输入 token
assert IN_PER_OUT == 8
# 四把尺子各圈什么：(分子含不含输入, 分母含不含 prefill 卡)
PICK = [(False, False), (True, True), (False, True), (True, False)]
assert [round((INOUT if a else OUT) / (ALL if b else DEC)) for a, b in PICK] == [r[1] for r in RULERS]


def _tokens(f, x, y, with_in, sq=11, gap=2):
    """一小堆 token：输出 4 个（绿）竖一列，输入 4×8 个（橙）。圈中的那堆加框，没圈的淡掉。"""
    n_out = 4
    for k in range(n_out):
        f.box(x, y + k * (sq + gap), sq, sq, GR, GR, 2)
    ix = x + sq + 10
    op = "" if with_in else ' opacity="0.25"'
    f.p.append('<g%s>' % op)
    for c in range(IN_PER_OUT):
        for r in range(n_out):
            f.box(ix + c * (sq + gap), y + r * (sq + gap), sq, sq, OR, OR, 2)
    f.p.append('</g>')
    H = n_out * (sq + gap) - gap
    x1 = (ix + IN_PER_OUT * (sq + gap) - gap) if with_in else (x + sq)
    f.box(x - 5, y - 5, x1 - x + 10, H + 10, "none", INK, 5, sw=2)
    return ix + IN_PER_OUT * (sq + gap)


def _cards(f, x, y, with_pf, sq=11, gap=2):
    """64 张卡：16 列 × 4 行，前 4 列（16 张）是 decode（蓝），其余 48 张 prefill（灰）。"""
    for c in range(ALL // 4):
        for r in range(4):
            dec = c < DEC // 4
            col = BL if dec else "#bdc1c6"
            f.box(x + c * (sq + gap), y + r * (sq + gap), sq, sq, col, col, 2)
    H = 4 * (sq + gap) - gap
    ncol = ALL // 4 if with_pf else DEC // 4
    f.box(x - 5, y - 5, ncol * (sq + gap) - gap + 10, H + 10, "none", INK, 5, sw=2)
    return x + (ALL // 4) * (sq + gap)


def fig_rulers():
    f = Fig(W, "同一次测量：GB300 上 SGLang 跑 DeepSeek-V4-Pro，decode 用 16 张卡，prefill 用 48 张，共 64 张；"
               "8K 进 1K 出，每个输出 token 对应约 8 个输入 token。每把尺子都是圈一堆 token、除以圈中的卡。"
               "只圈输出、除以 16 张 decode 卡是 5,270；输入输出都圈、除以全部 64 张是 11,880，超过官方标称的 11,200。"
               "另外两种是本课推出来的：只圈输出除以 64 张是 1,318，输入输出除以 16 张是 47,520。最大和最小差 36 倍")
    y0 = f.header("同一次测量，四把尺子　——　<tspan font-weight=\"700\">圈哪堆 token，除以圈哪些卡</tspan>",
                  "GB300 · SGLang · V4-Pro · 8K 进 1K 出。实心条是 InferenceX 真报的两个字段，虚线条是本课补全的另两种组合",
                  [(GR, "输出 token"), (OR, "输入 token"), (BL, "decode 卡 16 张"), ("#bdc1c6", "prefill 卡 48 张")])
    RH = 86
    PH = 30 + 44 + 4 * RH + 30
    py = f.panel(0, y0, W, PH, "每张卡每秒多少 token", INK)
    TX, CX, BX = 34, 250, 520
    f.t(TX - 4, py + 30, "数哪些 token", GY, True, 15)
    f.t(CX - 4, py + 30, "除以哪些卡", GY, True, 15)
    BWID, MX = 640, 48000
    for i, ((lab, v, col, note), (a, b)) in enumerate(zip(RULERS, PICK)):
        yy = py + 58 + i * RH
        _tokens(f, TX, yy, a)
        f.t(CX - 34, yy + 34, "÷", INK, True, 26, "middle")
        _cards(f, CX, yy, b)
        f.t(BX - 20, yy + 34, "＝", INK, True, 22, "middle")
        derived = col == GY2
        w = BWID * v / MX
        if derived:
            f.box(BX, yy + 6, w, 36, "#ffffff", GY2, 4, sw=1.6, dash="5,3")
        else:
            f.box(BX, yy + 6, w, 36, INK, INK, 4)
        f.t(BX + w + 12, yy + 32, format(v, ","), INK if not derived else GY, True, 20, big=True)
    xo = BX + BWID * OFFICIAL / MX
    f.line(xo, py + 50, xo, py + 58 + 4 * RH - 18, RD, 1.6, dash="6,4", arrow=False)
    f.t(xo + 8, py + 58 + 4 * RH + 2, "官方标称 11,200", RD, True, 15)
    f.t(W - 24, py + 58 + 4 * RH + 2, "最大 ÷ 最小 ＝ %d 倍" % round(SPREAD), INK, True, 17, "end")
    f._pan = None
    yb = f.src(py - 30 + PH + 4,
               "本课程作者实测：gpu-tpu-pedia gpu/inference/a4x-max/deepseek-v4/SGLANG-V4PRO-RUNBOOK.md:794-809"
               "（dep16：聚合 output 84,321；÷16 ＝ 5,270；tput_per_gpu ＝ 11,880）；字段定义见同文件 731-735"
               "（InferenceX process_result.py：output_tput_per_gpu ＝ output ÷ decode GPU，tput_per_gpu ＝ (输入＋输出) ÷ 全部 GPU）。",
               "本课推导：(in＋out) ＝ 11,880 × 64 ＝ 760,320；output ÷ 64 ＝ 1,318；(in＋out) ÷ 16 ＝ 47,520。"
               "自洽检查：8K 进 1K 出时 in＋out 约为 output 的 9 倍，9 × 84,321 ＝ 758,889，与 760,320 差 0.2%。",
               "⚠️ token 和卡的小方块只画比例：输出 4 个对输入 32 个 ＝ 1 : 8；卡 16 : 48 是实数。",
               "⚠️ 一颗 TPU v7 芯片对软件显示成 2 个 device，按 device 报的每卡数是按芯片报的一半。")
    f.save("fig6-rulers.svg", yb + 6)


TRAPS = [
    # (坑名, 坑里: (标签, 值), 改对: (标签, 值), 大数, 大数下面一句)
    ("峰值当稳态", ("峰值", PEAK[0][3]), ("端到端", PEAK[0][1]),
     "虚高 %d%%" % round((PEAK[0][3] / PEAK[0][1] - 1) * 100), "同一轮，换个读数"),
    ("没预热", ("第一轮", WARM[1][1]), ("第二轮", WARM[1][3]),
     "×%.2f" % (WARM[1][3] / WARM[1][1]), "同一套配置，跑第二遍"),
]
assert [t[3] for t in TRAPS] == ["虚高 13%", "×2.05"]


def fig_traps():
    f = Fig(W, "尺子以外的两个坑，每个都是同一套系统的两次实测。峰值当稳态：引擎自报的峰值 12,070，端到端稳态 10,704，峰值虚高 13%。"
               "没预热：vLLM 一组第一轮 13,585，第二轮 27,803，差 2.05 倍")
    y0 = f.header("尺子以外的坑　——　<tspan font-weight=\"700\">每一个都能单独让数字差出一截</tspan>",
                  "GB300 上的 DeepSeek-V4-Pro，两组各自独立的实测；每组上面是坑里的数，下面是改对之后的数",
                  [(GY2, "坑里的数"), (GR, "改对之后的数")])
    GAP = 20
    CW = (W - GAP * (len(TRAPS) - 1)) / float(len(TRAPS))
    PH = 30 + 260
    for i, (name, (la, va), (lb, vb), big, how) in enumerate(TRAPS):
        x = i * (CW + GAP)
        p = f.panel(x, y0, CW, PH, name, OR)
        f.t(x + CW / 2, p + 66, big, OR, True, 40, "middle", big=True)
        f.t(x + CW / 2, p + 96, how, GY, False, 15, "middle")
        mx = max(va, vb)
        BX, BMAX = x + 100, CW - 100 - 100
        for j, (lab, v, col) in enumerate(((la, va, GY2), (lb, vb, GR))):
            yy = p + 124 + j * 56
            f.t(BX - 12, yy + 26, lab, INK, True, 16, "end")
            bw = BMAX * v / mx
            f.box(BX, yy, bw, 38, col, col, 4)
            f.t(BX + bw + 10, yy + 26, format(v, ","), GY if col == GY2 else col, True, 17)
    f._pan = None
    yb = f.src(y0 + PH + 4,
               "峰值：gpu-tpu-pedia gpu/inference/a4x-max/deepseek-v4/SGLANG-V4PRO-RUNBOOK.md:849-858"
               "（ISL 4096，#7 端到端 10,704、峰值 12,070）。"
               "峰值是 decode 引擎自报的瞬时 gen throughput，按 DP rank 计（同文件 835-845）。"
               "另一组峰值稳态：tpu/tpu-inference/Gemma4-31B/README.md:460（TPU v7x，并发 256，输出 4,495、peak 6,144）。",
               "预热：VLLM-V4PRO-RUNBOOK.md:431-433（2P1D conc256，13,585 → 27,803）；另一组 SGLANG-V4PRO-RUNBOOK.md:480-494"
               "（8,520 → 9,118，+7.0%）。闭环：SGLANG-V4PRO-RUNBOOK.md:478、1135（闭环比开环低 13%）。",
               "温度这个坑（接受率 1.16% 对 34%）画在第十一节的图里，这里不重复；出处 VLLM-V4PRO-RUNBOOK.md:394-420。")
    f.save("fig6-traps.svg", yb + 6)


fig_metrics()
fig_pareto()
fig_rulers()
fig_traps()
