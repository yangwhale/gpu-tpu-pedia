# -*- coding: utf-8 -*-
r"""专题六 · 第二节「先校准尺子：指标与口径」的四张静态图。

fig6-metrics.svg —— 回答：TTFT、TPOT、ITL、每用户 tok/s 各量的是时间线上哪一段，含不含排队、含不含首字？
  · 定义：vLLM vllm/benchmarks/serve.py（TPOT ＝ (latency − ttft) ÷ (output_len − 1)，按请求算；
    ITL 是逐个 chunk 的间隔列表，投机解码时一个 chunk 可以带好几个 token）；
    NVIDIA NIM 的 benchmarking 指标页（TTFT 含排队、prefill、网络；ITL ＝ (e2e − TTFT) ÷ (输出 token − 1)，
    并写明 "also known as TPOT"）。两家对同一个词的口径不同，图里两种都写。
  · 锚点实测：gpu-tpu-pedia tpu/tpu-inference/Gemma4-31B/README.md:453-486（TPU v7x，TP=4，1K 进 1K 出）。
    并发 1：TTFT 86 ms、TPOT 35 ms、28 tok/s；并发 256：TTFT 7,756 ms（脚注：含排队）、TPOT 49 ms。
  · 刻意没画：时间刻度（时间线是示意）、网络往返的单独一段（并进 TTFT 里说）。

fig6-pareto.svg —— 回答：为什么不能光比「每卡吞吐」？
  · InferenceX GB300 V4-Pro FP4 8K/1K（Dynamo ＋ vLLM，PD 分离，output ÷ decode GPU）：
    68 → 9,759、121 → 3,816、173 → 895（gpu/inference/a4x-max/deepseek-v4/VLLM-V4PRO-RUNBOOK.md:683-695）。
  · PyTorch 博客 2026-06-23「Serving DeepSeek-V4 on GB300 with SGLang」：约 50 tok/s/user 处 ~11,200 tok/s/GPU
    （2026-06 MTP 曲线），Day-0 无 MTP 曲线约 2,200。
  · 本仓库实测：SGLang dep8 conc600 ＝ 9,168（output ÷ decode GPU），TPOT 59–61 ms ≈ 每用户 16 tok/s
    （gpu/inference/a4x-max/deepseek-v4/SGLANG-V4PRO-RUNBOOK.md:595、723-725）。
  · 刻意没画：把这几个点连成一条线（它们分属不同框架、不同配置）。只有 InferenceX 那三个点是同一条曲线。

fig6-rulers.svg —— 回答：同一次测量，换分子分母能差多少？
  · SGLANG-V4PRO-RUNBOOK.md:794-809 的 dep16：聚合 output 84,321；output ÷ 16 decode GPU ＝ 5,270；
    tput_per_gpu ＝ (in+out) ÷ 64 ＝ 11,880。另两把尺子（output ÷ 64、(in+out) ÷ 16）是本课推导，
    只为把「分子 × 分母」的四种组合排全，没有哪个工具这样报。
  · 自洽检查：8K 进 1K 出，in+out 约是 output 的 9 倍；11,880 × 64 ＝ 760,320 ≈ 9 × 84,321。

fig6-traps.svg —— 回答：除了分母，还有哪几个坑会让同一套系统报出差很远的数？
  · 假天花板：SGLANG-V4PRO-RUNBOOK.md:611-615、849-858（ISL 4096 下 ratio 0.20 → 0.15 → 再加 mem-frac）。
  · 峰值 vs 稳态：同上 #7（端到端 10,704、峰值 12,070）；Gemma4-31B README:460（4,495 vs peak 6,144）。
  · 温度：VLLM-V4PRO-RUNBOOK.md:394-420（接受率 1.16% vs 34%）。只对 greedy 草稿成立：
    gpu/inference/a4x-max/kimi-k3/VLLM-K3-RUNBOOK.md:226-230 的 draft 是 probabilistic，官方就用 1.0。
  · 预热：SGLANG-V4PRO-RUNBOOK.md:480-494（8,520 → 9,118）；VLLM-V4PRO-RUNBOOK.md:431-433（13,585 → 27,803）。
  · 闭环：SGLANG-V4PRO-RUNBOOK.md:478、1135（闭环 −13%）。
"""

from topic03_draw import Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE

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
CEIL = [("ratio 0.20", 6887), ("ratio 0.15", 10614), ("0.15 ＋ mem-frac 0.88", 10704)]
assert round(CEIL[1][1] / CEIL[0][1] - 1, 2) == 0.54
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
               "开了投机解码时一个 chunk 可能带两三个字。TTFT 是进门到第一个字，含排队和 prefill。"
               "ITL 是相邻两个 chunk 之间的每一次间隔。TPOT 是首字之后的总时长除以字数减一，是每个请求的平均值。"
               "每用户每秒多少字约等于一千除以 TPOT 的毫秒数。下方是 Gemma4-31B 在 TPU v7x 上的实测：并发 1 时 TTFT 86 毫秒，"
               "并发 256 时 7,756 毫秒，多出来的几乎全是排队；TPOT 从 35 涨到 49 毫秒")
    y0 = f.header("四个时间指标各量哪一段　——　<tspan font-weight=\"700\">先问含不含排队、含不含首字</tspan>",
                  "一个请求的时间线（示意，长短不按比例）。TTFT 和 TPOT 是每个请求一个数；ITL 是每个间隔一个数",
                  [(GY2, "排队"), (OR, "prefill"), (GR, "decode 推出的一个 chunk")])
    PH = 300
    py = f.panel(0, y0, W, PH, "一个请求的时间线", INK)
    T0, RY, RH = 120, py + 80, 50
    QW, PW_ = 190, 220
    f.box(T0, RY, QW - 6, RH, "#f1f3f4", GY2, 4, dash="5,4")
    f.t(T0 + (QW - 6) / 2, RY + 31, "排队", GY, True, 15, "middle")
    XP = T0 + QW
    f.box(XP, RY, PW_ - 6, RH, OR, OR, 4)
    f.t(XP + (PW_ - 6) / 2, RY + 31, "prefill", "#ffffff", True, 15, "middle")
    XD = XP + PW_
    # 每个 chunk 带几个字（投机解码：有的 chunk 带 2、3 个）
    CH = [1, 1, 2, 1, 3, 1, 2]
    CW = 92
    xs = [XD + i * CW for i in range(len(CH))]
    for i, (x, n) in enumerate(zip(xs, CH)):
        f.box(x, RY, CW - 8, RH, GR, GR, 4)
        f.t(x + (CW - 8) / 2, RY + 31, "%d 个字" % n, "#ffffff", True, 13.5, "middle")
    ends = [XD - 6] + [x + CW - 8 for x in xs]
    for i, ex in enumerate(ends):
        f.p.append('<circle cx="%.1f" cy="%.1f" r="6" fill="%s"/>' % (ex, RY - 14, RD if i == 0 else INK))
    f.t(XD - 6, RY - 28, "第一个字", RD, True, 13.5, "middle")
    f.line(T0, RY - 40, T0, RY - 4, GY2, 1.6)
    f.t(T0, RY - 46, "请求进门", GY, True, 13.5, "middle")
    N = 1 + sum(CH)                          # 首字 ＋ decode 推出的字
    assert N == 12
    BY = RY + RH + 28
    bracket(f, T0, XD - 6, BY, RD, "TTFT：进门 → 第一个字（含排队、prefill、网络）")
    bracket(f, ends[4], ends[5], BY, BL, "ITL：这一次间隔")
    BY2 = BY + 64
    bracket(f, XD - 6, ends[-1], BY2, GR, "TPOT ＝ 这一段 ÷ (字数 − 1)，每个请求一个平均数")
    f.t(ends[-1] + 24, RY + 22, "每用户 tok/s", INK, True, 15)
    f.t(ends[-1] + 24, RY + 46, "≈ 1000 ÷ TPOT(ms)", INK, False, 14)
    f.t(ends[-1] + 24, BY2 - 2, "这个请求共 %d 个字" % N, GY, size=13)
    f.t(ends[4] + 4, RY - 28, "带 3 个字的 chunk：这次 ITL 比 TPOT 长", BL, False, 13)

    # ② 两家工具的口径 ＋ 实测锚点
    y2 = py + PH + 16
    PH2 = 236
    HW = (W - 20) // 2
    p2 = f.panel(0, y2, HW, PH2, "同一个词，两家工具的口径", BL)
    rows = [("vLLM bench", "TPOT 按请求平均，不含首字"),
            ("", "ITL 是每个 chunk 的间隔，一个 chunk 可以带多个字"),
            ("NVIDIA NIM", "TTFT 含排队、prefill、网络"),
            ("", "ITL ＝ (e2e − TTFT) ÷ (字数 − 1)，即别家的 TPOT")]
    for i, (a, b) in enumerate(rows):
        yy = p2 + 40 + i * 40
        f.t(22, yy, a, BL, True, 14.5)
        f.t(170, yy, b, INK, size=14, w=HW - 190)
    p3 = f.panel(HW + 20, y2, HW, PH2, "实测锚点：Gemma4-31B · TPU v7x · 1K 进 1K 出", OR)
    X3 = HW + 40
    f.t(X3 + 200, p3 + 34, "TTFT", GY, True, 14, "middle")
    f.t(X3 + 360, p3 + 34, "TPOT", GY, True, 14, "middle")
    f.t(X3 + 520, p3 + 34, "每用户", GY, True, 14, "middle")
    for i, (lab, ttft, tpot, tps) in enumerate([("并发 1", "86 ms", "35 ms", "28 tok/s"),
                                                ("并发 256", "7,756 ms", "49 ms", "≈ 20 tok/s")]):
        yy = p3 + 72 + i * 42
        f.t(X3, yy, lab, INK, True, 15)
        f.t(X3 + 200, yy, ttft, RD if i else INK, True, 16, "middle")
        f.t(X3 + 360, yy, tpot, INK, True, 16, "middle")
        f.t(X3 + 520, yy, tps, GY if i else INK, i == 0, 15, "middle")
    f.t(X3, p3 + 172, "TTFT 涨了 90 倍，多出来的几乎全是排队", RD, True, 14)
    f.t(X3, p3 + 196, "并发 1 的 TTFT 才是纯 prefill；「≈ 20」是 1000 ÷ 49 推的", GY, size=13)
    f._pan = None
    yb = f.band(y2 + PH2 + 20, "ok", "看到一个延迟数，先问两句", [
        "TTFT 含不含排队？压测时排队可以比 prefill 本身长几十倍。",
        "TPOT 和 ITL 是哪家的定义？开了投机解码，逐次间隔会比平均间隔长，p99 看 ITL。",
    ])
    yb = f.src(yb + 10,
               "vLLM 定义：vllm/benchmarks/serve.py（TPOT ＝ (latency − ttft) ÷ (output_len − 1)；ITL 逐 chunk，"
               "源码注释写明 multiple output tokens may be bundled together）。",
               "NIM 定义：NVIDIA NIM for LLMs 的 benchmarking 指标页（TTFT 含排队、prefill、网络；ITL "
               "also known as TPOT）。AIPerf 沿用同一套。",
               "本课程作者实测：gpu-tpu-pedia tpu/tpu-inference/Gemma4-31B/README.md:453-486（并发 1 与 256；"
               "TTFT 7,756 ms 的脚注写明含排队）。并发 256 的每用户 tok/s 没有实测值，「≈ 20」由 1000 ÷ 49 推得。",
               "⚠️ 时间线是示意：chunk 带几个字、各段长短都是为了看得清，不对应任何实测。")
    f.save("fig6-metrics.svg", yb + 14)


def fig_pareto():
    f = Fig(W, "横轴是每个用户每秒拿到多少字，纵轴是每张 GPU 每秒出多少字，都是 GB300 上跑 DeepSeek-V4-Pro、8K 进 1K 出。"
               "InferenceX 的 vLLM 曲线三个点：每用户 68 时每卡 9,759，121 时 3,816，173 时只剩 895。"
               "PyTorch 博客的 SGLang 在每用户 50 处报每卡约 11,200，同一位置首发时只有约 2,200。"
               "本仓库实测的 9,168 落在每用户约 16 处。同一套系统沿着曲线往左走，每卡吞吐就往上涨；"
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


def fig_rulers():
    f = Fig(W, "同一次测量：GB300 上 SGLang 跑 DeepSeek-V4-Pro，decode 用 16 张卡，prefill 用 48 张，共 64 张。"
               "聚合输出每秒 84,321 个 token。按输出除以 decode 卡算是每卡 5,270；按输入加输出除以全部卡算是 11,880，"
               "超过官方标称的 11,200。另外两种组合是本课推出来的：输出除以全部卡 1,318，输入加输出除以 decode 卡 47,520。"
               "最大和最小差 36 倍")
    y0 = f.header("同一次测量，四把尺子　——　<tspan font-weight=\"700\">分子数什么，分母数谁</tspan>",
                  "GB300 · SGLang · V4-Pro · decode dep16 ＋ 12 个 prefill 节点（共 64 张卡）· 8K 进 1K 出 · 聚合输出 84,321 tok/s",
                  [(BL, "InferenceX 字段：只数输出"), (GR, "InferenceX 字段：输入输出合计"), (GY2, "本课推导")])
    PH = 420
    py = f.panel(0, y0, W, PH, "每张 GPU 每秒多少 token", INK)
    BX, BWID = 420, 760
    MX = 48000
    for i, (lab, v, col, note) in enumerate(RULERS):
        yy = py + 50 + i * 82
        f.t(24, yy + 20, lab, INK, True, 15)
        f.t(24, yy + 44, note, GY, size=13)
        w = BWID * v / MX
        f.box(BX, yy, w, 40, col, col, 4)
        lx = BX + w + 12
        f.t(lx, yy + 27, format(v, ","), col if col != GY2 else INK, True, 17)
    xo = BX + BWID * OFFICIAL / MX
    f.line(xo, py + 40, xo, py + 50 + 4 * 82 - 20, RD, 1.6, dash="6,4", arrow=False)
    f.t(xo + 8, py + 50 + 4 * 82 - 4, "官方标称 11,200", RD, True, 14)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "同一组原始数，「腰斩」和「超标」只隔一个分母", [
        "一个看着比官方低一半，一个看着超标，两个都没算错，只是回答的问题不一样。",
        "见到每卡吞吐先问两句：分子含不含输入？分母含不含 prefill 的卡？TPU 上还要再问一句：按芯片还是按 device。",
    ])
    yb = f.src(yb + 10,
               "本课程作者实测：gpu-tpu-pedia gpu/inference/a4x-max/deepseek-v4/SGLANG-V4PRO-RUNBOOK.md:794-809"
               "（dep16：聚合 output 84,321；÷16 ＝ 5,270；tput_per_gpu ＝ 11,880）；字段定义见同文件 731-735"
               "（InferenceX process_result.py）。",
               "本课推导：(in＋out) ＝ 11,880 × 64 ＝ 760,320；output ÷ 64 ＝ 1,318；(in＋out) ÷ 16 ＝ 47,520。"
               "自洽检查：8K 进 1K 出时 in＋out 约为 output 的 9 倍，9 × 84,321 ＝ 758,889，与 760,320 差 0.2%。",
               "⚠️ 一颗 TPU v7 芯片对软件显示成 2 个 device，按 device 报的每卡数是按芯片报的一半。")
    f.save("fig6-rulers.svg", yb + 14)


def _pairs(f, x, y, w, rows, mx, cols):
    """rows: [(label, value, text)]，横条。"""
    for i, (lab, v, txt) in enumerate(rows):
        yy = y + i * 50
        f.t(x, yy + 22, lab, INK, True, 14)
        bw = (w - 340) * v / mx
        f.box(x + 220, yy + 4, bw, 28, cols[i], cols[i], 4)
        f.t(x + 228 + bw, yy + 24, txt, cols[i], True, 15)


def fig_traps():
    f = Fig(W, "四个让同一套系统报出不同数的坑。左上：假天花板，同一个镜像，只改 KV 池的划分比例，端到端从 6,887 涨到 10,614，"
               "再加显存比例到 10,704，之前扫并发得出的天花板约 9,200 到 9,400 是错的。右上：峰值和稳态，V4-Pro 峰值 12,070 对端到端 10,704，"
               "Gemma4 峰值 6,144 对稳态 4,495。左下：温度，投机解码的草稿是 greedy 时，服务端按温度 1 采样，接受率只有 1.16%，"
               "温度 0 时 34%，吞吐差 3.1 倍。右下：预热，冷启动 8,520 对热 9,118；vLLM 一组差两倍")
    y0 = f.header("尺子以外的四个坑　——　<tspan font-weight=\"700\">每一个都能单独让数字差出一截</tspan>",
                  "全部是本仓库公开 runbook 里的实测，GB300 上的 DeepSeek-V4-Pro，另有一组 TPU v7x 上的 Gemma4-31B",
                  [(GY2, "坑里的数"), (GR, "改对之后的数")])
    HW = (W - 20) // 2
    PH = 270
    specs = []
    p = f.panel(0, y0, HW, PH, "假天花板：参数没调对，扫出来的只是那组参数的上界", OR)
    _pairs(f, 20, p + 20, HW - 20, [(lab, v, format(v, ",")) for lab, v in CEIL], 11000, [GY2, GR, GR])
    f.t(20, p + 196, "同为 ISL 4096、端到端 per-decode-GPU。之前在 ISL 8192 上扫并发，", GY, size=13)
    f.t(20, p + 216, "以为 ~9,200–9,400 就是天花板", GY, size=13)
    p = f.panel(HW + 20, y0, HW, PH, "峰值 vs 稳态：峰值是某一刻的最高值", OR)
    X = HW + 40
    for i, (lab, e2e, _, pk) in enumerate(PEAK):
        yy = p + 20 + i * 96
        f.t(X, yy + 18, lab, INK, True, 14.5)
        for j, (nm, v, col) in enumerate((("稳态", e2e, GR), ("峰值", pk, GY2))):
            bw = 400 * v / 12500
            f.box(X + 100, yy + 30 + j * 32, bw, 26, col, col, 4)
            f.t(X + 108 + bw, yy + 49 + j * 32, "%s %s" % (nm, format(v, ",")), col, True, 14.5)
    f.t(X, p + 226, "峰值比稳态高 13%～37%；报数要写清是哪个", GY, size=13)
    y1 = y0 + PH + 16
    p = f.panel(0, y1, HW, PH, "温度：greedy 草稿碰上默认温度 1.0", OR)
    rows = [("接受率", "1.16%", "34%"), ("输出 tok/s", "944", "2,911"), ("TPOT", "232 ms", "48.6 ms")]
    f.t(240, p + 34, "温度 1.0（默认）", GY, True, 14, "middle")
    f.t(450, p + 34, "温度 0", GR, True, 14, "middle")
    for i, (lab, a, b) in enumerate(rows):
        yy = p + 74 + i * 40
        f.t(24, yy, lab, INK, True, 15)
        f.t(240, yy, a, RD, True, 17, "middle")
        f.t(450, yy, b, GR, True, 17, "middle")
    f.t(24, p + 210, "只对 greedy 草稿成立；草稿按概率采样时，官方就用温度 1.0", GY, size=13)
    p = f.panel(HW + 20, y1, HW, PH, "预热和闭环：第一轮和闭环都偏低", OR)
    for i, (lab, cold, _, warm) in enumerate(WARM):
        yy = p + 24 + i * 92
        f.t(X, yy + 18, lab, INK, True, 14.5)
        for j, (nm, v, col) in enumerate((("冷", cold, GY2), ("热", warm, GR))):
            bw = 400 * v / 30000
            f.box(X + 130, yy + 28 + j * 30, bw, 24, col, col, 4)
            f.t(X + 138 + bw, yy + 46 + j * 30, "%s %s" % (nm, format(v, ",")), col, True, 14.5)
    f.t(X, p + 212, "闭环压测（限并发）又低 13%：对标要开环，跑两轮报第二轮", GY, size=13)
    f._pan = None
    yb = f.band(y1 + PH + 20, "ok", "四个坑，单独一个就能差出 7% 到 3 倍", [
        "引擎一行代码没改，换个参数、换个读数、换个温度、换个时机，报出来的数就不一样。",
        "对标别人的数之前，先把这四样和他对齐。",
    ])
    yb = f.src(yb + 10,
               "假天花板与峰值：gpu-tpu-pedia gpu/inference/a4x-max/deepseek-v4/SGLANG-V4PRO-RUNBOOK.md:611-615、849-858"
               "（ISL 4096，#1 ratio 0.20 → 6,887；#3 0.15 → 10,614；#7 加 mem-frac 0.88 → 10,704、峰值 12,070）。"
               "峰值是 decode 引擎自报的瞬时 gen throughput，按 DP rank 计（同文件 835-845）。",
               "Gemma4-31B：tpu/tpu-inference/Gemma4-31B/README.md:460（并发 256，输出 4,495、peak 6,144）。",
               "温度：gpu/inference/a4x-max/deepseek-v4/VLLM-V4PRO-RUNBOOK.md:394-420（4k1k conc256，DSpark 草稿 "
               "draft_sample_method 为 greedy）。反例：gpu/inference/a4x-max/kimi-k3/VLLM-K3-RUNBOOK.md:226-230，草稿为 probabilistic。",
               "预热：SGLANG-V4PRO-RUNBOOK.md:480-494（8,520 → 9,118，+7.0%）；VLLM-V4PRO-RUNBOOK.md:431-433（2P1D conc256，"
               "13,585 → 27,803）。闭环：SGLANG-V4PRO-RUNBOOK.md:478、1135。")
    f.save("fig6-traps.svg", yb + 14)


fig_metrics()
fig_pareto()
fig_rulers()
fig_traps()
