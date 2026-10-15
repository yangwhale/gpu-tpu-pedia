# -*- coding: utf-8 -*-
r"""专题六 · 第九节「配几台、拆不拆：P:D 配比、AFD、EPD」的四张静态图。

fig6-2p1d.svg —— 回答：在 1P1D 上再加一台 prefill，买到的到底是什么？
  · 数从哪来：gpu/inference/a4/deepseek-v3/REPORT-2P1D-DeepSeek-V3.md:69/85/100–142（B200，a4-highgpu-8g，
    SGLang 0.5.8 ＋ NIXL，每边 TP8 一台；3000–4000 输入、400–600 输出；并发 128）。
    output 1,118.8 → 1,159.0；平均 TTFT 2.40 → 1.73 s；P99 TTFT 12.07 → 6.23 s；每卡 69.9 → 48.3（报告自己算的，脚本复算）。
  · 刻意没画：TPOT。报告两边都开了 --disable-cuda-graph，TPOT 约 0.1 s 偏高，拿它比会误导。

fig6-prefill-sweep.svg —— 回答：prefill 越加越多，吞吐到底涨了多少？换个分母答案就变。
  · 数从哪来：gpu/inference/a4x-max/deepseek-v4/VLLM-V4PRO-RUNBOOK.md §12.3（2026-07-24 历史实测，runbook 注明本文未重测）：
    decode ＝ dep8（8 卡），每个 prefill ＝ TP4（4 卡，见同文件 §0），8k1k，conc1024。
  · 本课推导：「每张 decode 卡」＝ output ÷ 8（runbook 原表）；「每张卡（含 prefill）」＝ output ÷（4 × prefill 数 ＋ 8）。
  · 刻意没画：16 个 prefill 那一档（失效模式 B，只跑出部分结果）。

fig6-decide.svg —— 回答：拿到一个负载，拆不拆、怎么配？
  · 数从哪来：叶子上的例子都是公开实测，各自口径不同，只当路标：
    MiMo（tpu/sglang-jax/mimo/README.md:8–12、81–88，TPU v7x 4 芯片一个 pod，16K 输入 4K 输出）；
    Qwen3-Coder-480B（tpu/tpu-inference/Qwen3-Coder-480B/README.md:1005–1021，两台 v7x-8）；
    配比公式与 8–10 个 prefill（gpu/inference/a4x-max/deepseek-v4/SGLANG-V4PRO-RUNBOOK.md §12.2）；PrFaaS（arXiv 2604.15039）。
  · 刻意没画：阈值。多长算「长 prompt」、多严算「严 SLO」没有通用数，画了就是编。

fig6-afd-ascend.svg —— 回答：AFD 拆开就一定更快吗？
  · 数从哪来：vLLM AFD 插件博客（vllm.ai/blog/2026-07-23-vllm-afd-plugin）：Ascend 910C，DeepSeek-V3.2 W8A8，
    饱和 decode 吞吐，tokens/s/die ＝ 总输出 ÷ 部署的全部 die。EP64（64 die）、48A16F（64 die）、64A16F（80 die）。
    ⚠️ 专家路由被换成强制均衡（会改变输出）；48A16F／64A16F 用物理部署模拟逻辑上的 192A64F／256A64F。
  · 刻意没画：博客里 CAM async prefill 那组（只有 10 层的裁剪模型）。
"""
from topic03_draw import Fig, BL, OR, GR, RD, PU, CY, GY, INK, GY2, LINE

W = 1400

# ── 2P1D 报告（并发 128）
OUT_1, OUT_2 = 1118.8, 1159.0
GPU_1, GPU_2 = 16, 24
TTFT_1, TTFT_2 = 2.40, 1.73
P99_1, P99_2 = 12.07, 6.23
PER_1, PER_2 = OUT_1 / GPU_1, OUT_2 / GPU_2
assert round(PER_1, 1) == 69.9 and round(PER_2, 1) == 48.3
D_OUT = (OUT_2 / OUT_1 - 1) * 100
D_TTFT = (TTFT_2 / TTFT_1 - 1) * 100
D_P99 = (P99_2 / P99_1 - 1) * 100
D_PER = (PER_2 / PER_1 - 1) * 100
assert round(D_OUT, 1) == 3.6 and round(D_TTFT) == -28 and round(D_P99) == -48 and round(D_PER) == -31

# ── V4-Pro vLLM 扫描（8k1k，dep8 decode，每个 prefill TP4）
SWEEP = [(1, 2453), (2, 4773), (4, 8920), (6, 12369), (8, 17614), (10, 21433), (12, 25500), (14, 26568)]
PER_D = [(n, o / 8) for n, o in SWEEP]
PER_ALL = [(n, o / (4 * n + 8)) for n, o in SWEEP]
assert [round(v) for _, v in PER_D] == [307, 597, 1115, 1546, 2202, 2679, 3188, 3321]   # runbook 表里 12p 写 3,187（截断）
assert round(PER_ALL[0][1]) == 204 and round(PER_ALL[-1][1]) == 415
BEST_ALL = max(PER_ALL, key=lambda p: p[1])
assert BEST_ALL[0] == 12 and round(BEST_ALL[1]) == 455
GROW_D = PER_D[-1][1] / PER_D[0][1]
GROW_ALL = BEST_ALL[1] / PER_ALL[0][1]
assert round(GROW_D, 1) == 10.8 and round(GROW_ALL, 1) == 2.2

# ── 配比公式（SGLang V4-Pro runbook §12.2，官方 11,200 那个点）
DEC_RPS = 8 * 11200 / 1024
NEED_IN = DEC_RPS * 8192
ONE_P = 4 * 18200
assert round(DEC_RPS, 1) == 87.5 and round(NEED_IN / 1e4) == 72 and 8 <= NEED_IN / ONE_P <= 10   # 71.7 万 ÷ 7.28 万 ≈ 9.8

# ── TPU 两个单点
MIMO = dict(nonpd64=11.70, pd1_64=10.55, pd2_64=10.83, nonpd128=12.78, pd1_128=13.64, pd2_128=15.31)
assert round((MIMO["pd1_128"] / MIMO["nonpd128"] - 1) * 100) == 7     # README 写「约 6.8%」，按表算 6.7%
assert round((MIMO["pd2_128"] / MIMO["nonpd128"] - 1) * 100, 1) == 19.8
assert MIMO["pd2_128"] / 3 < MIMO["nonpd128"] / 2            # 2P1D 多一个 pod，按每 pod 算反而低
QC_TTFT = (95, 281)                                            # Qwen3-Coder 1024/1024 c1，单实例 → 1P1D（ms）

# ── AFD 插件（tokens/s/die）
AFD = {"16K": (232.6, 220.3, 258.9), "32K": (168.2, 151.4, 183.3)}
assert round((AFD["16K"][1] / AFD["16K"][0] - 1) * 100, 1) == -5.3 and round((AFD["16K"][2] / AFD["16K"][0] - 1) * 100, 1) == 11.3
assert round((AFD["32K"][1] / AFD["32K"][0] - 1) * 100, 1) == -10.0 and round((AFD["32K"][2] / AFD["32K"][0] - 1) * 100, 1) == 9.0


def fig_2p1d():
    f = Fig(W, "同一组实测，两种看法。DeepSeek-V3 在 B200 上，从 1 台 prefill 加 1 台 decode，变成 2 台 prefill 加 1 台 decode，"
               "并发 128。左边按机器看：总输出只从 1,118.8 涨到 1,159.0，多 3.6%；每张卡从 69.9 掉到 48.3，少 31%。"
               "右边按用户看：平均首字延迟从 2.40 秒降到 1.73 秒，少 28%；P99 首字延迟从 12.07 秒降到 6.23 秒，少 48%")
    y0 = f.header("多一台 prefill，买的是首字快，不是吞吐　——　<tspan font-weight=\"700\">每张卡反而少了三成</tspan>",
                  "DeepSeek-V3，B200（每台 8 卡），SGLang ＋ NIXL；1P1D（16 卡）→ 2P1D（24 卡）；输入 3,000–4,000、输出 400–600，并发 128",
                  [(GY2, "1P1D"), (BL, "2P1D")])
    PH = 330
    HW = (W - 20) / 2
    for side, (x0, title, col, rows) in enumerate((
            (0, "按机器看：花的卡换来多少字", OR,
             [("总输出（tok/s）", OUT_1, OUT_2, "%+.1f%%" % D_OUT, "{:,.1f}", GR),
              ("每张卡输出（tok/s）", PER_1, PER_2, "%+.0f%%" % D_PER, "{:.1f}", RD)]),
            (HW + 20, "按用户看：等第一个字等多久", GR,
             [("平均 TTFT（s）", TTFT_1, TTFT_2, "%+.0f%%" % D_TTFT, "{:.2f}", GR),
              ("P99 TTFT（s）", P99_1, P99_2, "%+.0f%%" % D_P99, "{:.2f}", GR)]))):
        py = f.panel(x0, y0, HW, PH, title, col)
        y = py + 40
        for lab, a, b, d, fmt, dc in rows:
            vmax = max(a, b) * 1.15
            BX, BW = x0 + 200, HW - 330
            f.t(x0 + 20, y + 30, lab, INK, True, 15)
            for i, (v, c) in enumerate(((a, GY2), (b, BL))):
                f.box(BX, y + 8 + i * 38, BW * v / vmax, 28, c, c, 3)
                f.t(BX + BW * v / vmax + 10, y + 28 + i * 38, fmt.format(v), c, True, 14.5)
            f.t(x0 + HW - 20, y + 48, d, dc, True, 22, "end")
            y += 130
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "加 prefill 前先问：缺的是首字速度，还是每秒出的字", [
        "这台 prefill 让排队的请求更快开始，总输出几乎没动；并发再往上到 256，2P1D 的总输出反而少 1.3%，瓶颈已经换到 decode。",
        "所以拿每卡吞吐评估 2P1D 一定吃亏；它该拿 TTFT 的 SLO 来评。",
    ])
    yb = f.src(yb + 10,
               "📌 gpu/inference/a4/deepseek-v3/REPORT-2P1D-DeepSeek-V3.md：表在 :69、:85，对比在 :100–142（每卡 ＝ 总输出 ÷ 卡数，报告原算法，脚本复算）。c256：1,798.1 → 1,775.6。",
               "⚠️ TTFT 是客户端测的，含排队。两边都开了 --disable-cuda-graph（REPORT-1P1D :129），所以没画 TPOT。")
    f.save("fig6-2p1d.svg", yb + 14)


def fig_prefill_sweep():
    f = Fig(W, "vLLM 跑 DeepSeek-V4-Pro，decode 固定 8 张卡，prefill 从 1 个加到 14 个，每个 prefill 4 张卡。"
               "蓝线按每张 decode 卡算，从 307 一路涨到 3,321，涨了 10.8 倍。"
               "橙线把 prefill 的卡也算进分母，从 204 涨到 12 个 prefill 时的 455 就到顶，只涨 2.2 倍，14 个时回落到 415")
    y0 = f.header("只除 decode 卡，加 prefill 像白捡；全部卡都算上，到 12 个就到顶",
                  "同一组实测（vLLM，DeepSeek-V4-Pro，GB300；decode ＝ dep8 共 8 卡，每个 prefill TP4 共 4 卡；8K 输入 1K 输出，并发 1,024）。纵轴是每卡 output tok/s",
                  [(BL, "÷ decode 卡（runbook 原口径）"), (OR, "÷ 全部卡（本课推导）")])
    PH = 420
    py = f.panel(0, y0, W, PH, "每卡 output tok/s 随 prefill 个数", BL)
    PX, PW = 120, 1060
    PY, PHH = py + 30, PH - 110
    VMAX = 3600.0

    def X(n):
        return PX + PW * (n - 0) / 15.0

    def Y(v):
        return PY + PHH - PHH * v / VMAX
    for v in (0, 1000, 2000, 3000):
        f.line(PX, Y(v), PX + PW, Y(v), LINE, 1, arrow=False)
        f.t(PX - 10, Y(v) + 5, "{:,}".format(v), GY, size=13, anchor="end")
    for n, _ in SWEEP:
        f.t(X(n), PY + PHH + 22, str(n), GY, size=13, anchor="middle")
    f.t(X(7.5), PY + PHH + 46, "prefill 个数", GY, True, 13.5, "middle")
    for pts, col in ((PER_D, BL), (PER_ALL, OR)):
        d = "M" + " L".join("%.1f,%.1f" % (X(n), Y(v)) for n, v in pts)
        f.path(d, col, 3, arrow=False)
        for n, v in pts:
            f.p.append('<circle cx="%.1f" cy="%.1f" r="5" fill="%s"/>' % (X(n), Y(v), col))
    f.t(X(1) + 10, Y(PER_D[0][1]) - 12, "%d" % round(PER_D[0][1]), BL, True, 14)
    f.t(X(14) + 12, Y(PER_D[-1][1]) + 5, "{:,}".format(round(PER_D[-1][1])), BL, True, 15)
    f.t(X(1), Y(PER_ALL[0][1]) + 26, "%d" % round(PER_ALL[0][1]), OR, True, 14, "middle")
    f.t(X(12), Y(BEST_ALL[1]) - 14, "12 个时 %d，到顶" % round(BEST_ALL[1]), OR, True, 14, "middle")
    f.t(X(14) + 12, Y(PER_ALL[-1][1]) + 5, "%d" % round(PER_ALL[-1][1]), OR, True, 15)
    f.t(PX + PW + 70, Y(2600), "×%.1f" % GROW_D, BL, True, 22, "middle")
    f.t(PX + PW + 70, Y(900), "×%.1f" % GROW_ALL, OR, True, 22, "middle")
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "两条线都说加 prefill 是杠杆，只是杠杆长短差了五倍", [
        "蓝线的分母里没有 prefill 的卡：喂一个小 decode 的 prefill 越多，它越好看。算整机成本要看橙线。",
        "橙线在 12 个 prefill 时到顶，再加就开始稀释每卡产出，配比到这里就够了。",
    ])
    yb = f.src(yb + 10,
               "📌 gpu/inference/a4x-max/deepseek-v4/VLLM-V4PRO-RUNBOOK.md §12.3（2026-07-24 实测，runbook 注明本文未重测；12 个时原表写 3,187，脚本按 25,500 ÷ 8 算出 3,188）；prefill 为 TP4 见同文件 §0。",
               "⚠️ 本课推导：橙线 ＝ output ÷（4 × prefill 个数 ＋ 8）。16 个 prefill 那档撞上失效模式 B、只跑出部分结果，没画。"
               "同一 runbook 的 4K 输入重扫显示每个 prefill 喂料约翻倍（1 个 prefill 时 2.06 倍），曲线形状随输入长度变。")
    f.save("fig6-prefill-sweep.svg", yb + 14)


def fig_decide():
    f = Fig(W, "拆不拆的决策树。第一问：长 prompt 会不会打断别人的出字，也就是 prefill 和 decode 有没有真在打架；"
               "没有就不拆，用分块 prefill。第二问：KV 能不能走快线，同一个 NVLink 域或同一个 TPU 切片里传；"
               "不能就要 KV 足够薄，或者只把长 prompt 外送。第三问：缺的是首字还是吞吐；缺首字加 prefill，缺吞吐或并发上不去加 decode。"
               "起步的配比用公式：prefill 个数等于 decode 每秒完成的请求数乘输入长度，除以一个 prefill 每秒能吞的 token")
    y0 = f.header("拆不拆，看负载，不看潮流　——　<tspan font-weight=\"700\">三个问题，从上往下问</tspan>",
                  "叶子上的例子都是公开实测，口径各不相同，只当路标。多长算长、多严算严没有通用阈值，要拿自己的负载量",
                  [(BL, "问题"), (GR, "拆"), (OR, "不拆或慎拆")])
    QX, QW, QH = 30, 470, 96
    LX, LW = 590, 780
    rows = [
        ("① 长 prompt 会打断别人出字吗？", "输入长、并发高、TPOT 有硬要求",
         OR, "否 → 不拆，用分块 prefill",
         ["MiMo（TPU v7x，16K 进 4K 出）：batch 64 时不拆 11.70K，1P1D 只有 10.55K。",
          "Qwen3-Coder 1K 进 1K 出、单并发：TTFT 95 ms 变成 281 ms，多出一趟传输。"]),
        ("② KV 能走快线吗？", "同一个 NVLink 域 ／ 同一个 TPU 切片",
         OR, "否 → KV 要够薄，或只把长 prompt 外送",
         ["PrFaaS：混合模型 KV 薄 4–13 倍，才敢跨数据中心，按长度阈值挑请求外送。",
          "GPU 跨了 NVLink 域就退到 RDMA，同一套配置明显变慢（V4-Pro runbook）。"]),
        ("③ 缺的是首字，还是吞吐？", "看哪个 SLO 先破",
         GR, "缺首字 → 加 prefill；缺吞吐 → 加 decode 或换 decode 切法",
         ["MiMo batch 128：1P1D 比不拆多约 7%，2P1D 多 19.8%（但多一个 pod）。",
          "起步配比：prefill 个数 ≈ decode 完成率 × 输入长度 ÷ 一个 prefill 的吞吐（V4-Pro 算出 8–10 个）。"]),
    ]
    y = y0 + 10
    for i, (q, qs, col, ans, ex) in enumerate(rows):
        f.box(QX, y, QW, QH, "none", BL, 10, sw=2)
        f.t(QX + 20, y + 40, q, BL, True, 17)
        f.t(QX + 20, y + 70, qs, GY, size=13.5)
        f.path("M%d,%d L%d,%d" % (QX + QW, y + QH / 2, LX - 6, y + QH / 2), col, 2)
        f.box(LX, y, LW, QH + 30, "none", col, 10, sw=1.6)
        f.t(LX + 18, y + 30, ans, col, True, 16)
        for j, e in enumerate(ex):
            f.t(LX + 18, y + 60 + j * 26, e, INK, size=13.5)
        if i < len(rows) - 1:
            f.path("M%d,%d L%d,%d" % (QX + QW / 2, y + QH, QX + QW / 2, y + 160), BL, 2)
            f.t(QX + QW / 2 + 12, y + QH + 36, "是", BL, True, 15)
        y += 160
    f._pan = None
    yb = f.band(y + 6, "ok", "拆开换来两样：decode 那一趟不再被 prefill 打断，两边各用最合适的切法", [
        "三个问题里只要第一问答「否」，后面两问都不用问；拆开多一趟传输、两套机器、一份配比要调。",
        "配比先用公式起步，再按「哪个 SLO 先破」往哪边加卡。",
    ])
    yb = f.src(yb + 10,
               "📌 MiMo：tpu/sglang-jax/mimo/README.md:8–12、81–88（TPU v7x 每 pod 4 芯片，16K 进 4K 出，request_rate=inf，total tok/s）。"
               "Qwen3-Coder-480B：tpu/tpu-inference/Qwen3-Coder-480B/README.md:1005–1021（两台 v7x-8，TTFT 中位数）。",
               "📌 配比公式：gpu/inference/a4x-max/deepseek-v4/SGLANG-V4PRO-RUNBOOK.md §12.2（8 × 11,200 ÷ 1,024 ≈ 87.5 req/s，× 8,192 ≈ 71.7 万 tok/s，÷ 7.28 万 ≈ 9.8）。"
               "跨域退化：同文件 :63。PrFaaS：arXiv 2604.15039。",
               "⚠️ 两组 TPU 例子都是单点测量，不是扫描。")
    f.save("fig6-decide.svg", yb + 14)


def fig_afd_ascend():
    f = Fig(W, "vLLM 的 AFD 插件在昇腾 910C 上跑 DeepSeek-V3.2 的饱和 decode 吞吐，按每个 die 算。"
               "16K 输入：不拆的 EP64 是 232.6，48 个 attention 配 16 个 FFN 是 220.3，少 5.3%；64 个 attention 配 16 个 FFN 是 258.9，多 11.3%。"
               "32K 输入：168.2、151.4（少 10%）、183.3（多 9%）。attention 给少了，拆开反而慢")
    y0 = f.header("AFD 拆开不保证更快　——　<tspan font-weight=\"700\">attention 和专家的配比给错了，反而慢</tspan>",
                  "vLLM AFD 插件，Ascend 910C，DeepSeek-V3.2 W8A8，饱和 decode。tokens/s/die ＝ 总输出 ÷ 部署的全部 die（EP64、48A16F 是 64 个，64A16F 是 80 个）",
                  [(GY2, "EP64（不拆）"), (OR, "48A16F：48 attention ＋ 16 FFN"), (GR, "64A16F：64 attention ＋ 16 FFN")])
    PH = 330
    py = f.panel(0, y0, W, PH, "每 die 输出 tokens/s", GR)
    BX, BW = 200, 980
    VMAX = 300.0
    y = py + 36
    for k, vals in AFD.items():
        f.t(BX - 20, y + 54, "%s 输入" % k, INK, True, 17, "end")
        for i, (v, c) in enumerate(zip(vals, (GY2, OR, GR))):
            f.box(BX, y + i * 38, BW * v / VMAX, 28, c, c, 3)
            lab = "%.1f" % v
            if i:
                lab += "（%+.1f%%）" % ((v / vals[0] - 1) * 100)
            f.t(BX + BW * v / VMAX + 10, y + 20 + i * 38, lab, c, True, 14.5)
        y += 140
    f._pan = None
    yb = f.band(py + PH + 20, "warn", "博客原话：只拆开，不保证吞吐提升", [
        "attention 卡给多了，专家那边才吃得饱；作者说测过的配比下 FFN 还有余力，没测更高的 attention 比例。",
        "这组数把专家路由换成了强制均衡（输出会变），还用小部署模拟大规模；只能看趋势。",
    ])
    yb = f.src(yb + 10,
               "📌 vllm.ai/blog/2026-07-23-vllm-afd-plugin ／ github.com/vllm-project/afd-plugin：AFD 是外部插件，不是 vLLM 原生功能；"
               "vLLM 的 AFD RFC #22799 于 2026-05-31 被机器人以长期不活跃关闭，没有合入。",
               "⚠️ 48A16F ／ 64A16F 模拟的是逻辑上的 192A64F ／ 256A64F（博客原文）。")
    f.save("fig6-afd-ascend.svg", yb + 14)


if __name__ == "__main__":
    fig_2p1d()
    fig_prefill_sweep()
    fig_decide()
    fig_afd_ascend()
