# -*- coding: utf-8 -*-
r"""专题六 · 第九节「配几台、拆不拆：P:D 配比、AFD、EPD」的五张静态图。

fig6-split-pod.svg —— 回答：MiMo 拆成 PD 以后，总吞吐最高的 2P1D 按 pod 算还赢吗？
  · 数从哪来：tpu/sglang-jax/mimo/README.md:81-88 总体结果表（TPU v7x，每 pod 4 芯片，16K 进 4K 出，total tok/s 含输入）。
  · 本课推导：每 pod ＝ total ÷ pod 数（不拆 2、1P1D 2、2P1D 3）；+6.7%、−20% 同正文 9.5 与台账。
  · 刻意没画：SLO。README 自己说这张表只看端到端趋势，没考虑 SLO。

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
D12 = 3187                                   # runbook 原表 12 个 prefill 那格（25,500 ÷ 8 截断）
assert int(dict(PER_D)[12]) == D12
GROW_D = D12 / PER_D[0][1]                  # 两条线都取 12 个 prefill 那档，和正文 9.4 同一组
GROW_ALL = BEST_ALL[1] / PER_ALL[0][1]
assert round(GROW_D, 1) == 10.4 and round(GROW_ALL, 1) == 2.2

# ── 配比公式（SGLang V4-Pro runbook §12.2，官方 11,200 那个点）
DEC_RPS = 8 * 11200 / 1024
NEED_IN = DEC_RPS * 8192
ONE_P = 4 * 18200
assert round(DEC_RPS, 1) == 87.5 and round(NEED_IN / 1e4) == 72 and 8 <= NEED_IN / ONE_P <= 10   # 71.7 万 ÷ 7.28 万 ≈ 9.8

# ── TPU 两个单点
MIMO = dict(nonpd64=11.70, pd1_64=10.55, pd2_64=10.83, nonpd128=12.78, pd1_128=13.64, pd2_128=15.31)
assert round((MIMO["pd1_128"] / MIMO["nonpd128"] - 1) * 100) == 7     # README 写「约 6.8%」，按表算 6.7%
assert round((MIMO["pd2_128"] / MIMO["nonpd128"] - 1) * 100, 1) == 19.8
assert round((MIMO["pd2_128"] / 3) / (MIMO["nonpd128"] / 2) * 100 - 100) == -20   # 2P1D 多一个 pod，按每 pod 算低 20%
# 每 pod 吞吐（本课推导：total ÷ pod 数），图上要标「推导」
SPLIT = [("batch 64", [("不拆", MIMO["nonpd64"], 2), ("1P1D", MIMO["pd1_64"], 2), ("2P1D", MIMO["pd2_64"], 3)]),
         ("batch 128", [("不拆", MIMO["nonpd128"], 2), ("1P1D", MIMO["pd1_128"], 2), ("2P1D", MIMO["pd2_128"], 3)])]
from decimal import Decimal, ROUND_HALF_UP
def _r2(v):
    return str(Decimal(repr(v)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
assert [_r2(t / n) for _, rows in SPLIT for _, t, n in rows] == ["5.85", "5.28", "3.61", "6.39", "6.82", "5.10"]
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
        "这台 prefill 让排队的请求更快开始，总输出几乎没动。拿每卡吞吐评估 2P1D 一定吃亏，该拿 TTFT 的 SLO 来评。",
    ])
    yb = f.src(yb + 10,
               "📌 gpu/inference/a4/deepseek-v3/REPORT-2P1D-DeepSeek-V3.md：表在 :69、:85，对比在 :100–142（每卡 ＝ 总输出 ÷ 卡数，报告原算法，脚本复算）。c256：1,798.1 → 1,775.6。",
               "⚠️ TTFT 是客户端测的，含排队。两边都开了 --disable-cuda-graph（REPORT-1P1D :129），所以没画 TPOT。")
    f.save("fig6-2p1d.svg", yb + 14)


def fig_prefill_sweep():
    f = Fig(W, "vLLM 跑 DeepSeek-V4-Pro，decode 固定 8 张卡，prefill 从 1 个加到 14 个，每个 prefill 4 张卡。"
               "蓝线按每张 decode 卡算，从 307 涨到 12 个 prefill 时的 3,187，是起点的 10.4 倍，14 个时还在涨。"
               "橙线把 prefill 的卡也算进分母，从 204 涨到 12 个 prefill 时的 455 就到顶，只是起点的 2.2 倍，14 个时回落到 415")
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
    f.t(X(12), Y(D12) - 16, "12 个时 {:,}".format(D12), BL, True, 15, "middle")
    f.t(X(1), Y(PER_ALL[0][1]) + 26, "%d" % round(PER_ALL[0][1]), OR, True, 14, "middle")
    f.t(X(12), Y(BEST_ALL[1]) - 14, "12 个时 %d，到顶" % round(BEST_ALL[1]), OR, True, 14, "middle")
    f.t(X(14) + 12, Y(PER_ALL[-1][1]) + 5, "%d" % round(PER_ALL[-1][1]), OR, True, 15)
    f.t(PX + PW + 70, Y(2600), "×%.1f" % GROW_D, BL, True, 22, "middle")
    f.t(PX + PW + 70, Y(2600) + 22, "1 → 12 个", BL, size=15, anchor="middle")
    f.t(PX + PW + 70, Y(900), "×%.1f" % GROW_ALL, OR, True, 22, "middle")
    f.t(PX + PW + 70, Y(900) + 22, "1 → 12 个", OR, size=15, anchor="middle")
    f._pan = None
    yb = f.band(py + PH - 10, "ok", "两条线都说加 prefill 是杠杆，只是杠杆长短差了五倍", [
        "蓝线的分母里没有 prefill 的卡：喂一个小 decode 的 prefill 越多，它越好看。算整机成本要看橙线。",
        "橙线在 12 个 prefill 时到顶，再加就开始稀释每卡产出，配比到这里就够了。",
    ])
    yb = f.src(yb + 10,
               "📌 gpu/inference/a4x-max/deepseek-v4/VLLM-V4PRO-RUNBOOK.md §12.3（2026-07-24 实测，runbook 注明本文未重测；12 个时取原表的 3,187，25,500 ÷ 8 ＝ 3,187.5）；prefill 为 TP4 见同文件 §0。",
               "⚠️ 本课推导：橙线 ＝ output ÷（4 × prefill 个数 ＋ 8）。16 个 prefill 那档撞上失效模式 B、只跑出部分结果，没画。"
               "同一 runbook 的 4K 输入重扫显示每个 prefill 喂料约翻倍（1 个 prefill 时 2.06 倍），曲线形状随输入长度变。")
    f.save("fig6-prefill-sweep.svg", yb + 14)


def fig_decide():
    f = Fig(W, "拆不拆的决策树。第一问：不拆时 TPOT 有没有被长 prompt 撑出尖峰；没有就不拆，拆了多一趟传输、多一套机器。"
               "有尖峰，先开分块 prefill，把长 prompt 切成小块夹在 decode 步之间；压住了就不拆，仍有尖峰才往下问。"
               "第二问：KV 能不能走快线，同一个 NVLink 域或同一个 TPU 切片里传；不能就要 KV 足够薄，或者只把长 prompt 外送。"
               "第三问：缺的是首字还是吞吐；缺首字加 prefill，缺吞吐加 decode")
    y0 = f.header("拆不拆，看负载，不看潮流　——　<tspan font-weight=\"700\">先试分块，压不住再拆</tspan>",
                  "多长算长、多严算严没有通用阈值，要拿自己的负载量",
                  [(BL, "问题"), (GR, "拆"), (OR, "不拆或慎拆")])
    QX, QW, QH = 30, 500, 92
    LX, LW = 610, 760
    rows = [
        ("① 不拆时，TPOT 有 prefill 尖峰吗？", "长 prompt 一进来，别人出字就卡一下",
         OR, "否 → 不拆", "拆了多一趟传输、多一套机器，换不回东西"),
        ("先开分块 prefill：仍有尖峰吗？", "长 prompt 切成小块，夹在 decode 步之间",
         OR, "否 → 不拆，分块就够", "压得住，就省下拆开那套代价"),
        ("② KV 能走快线吗？", "同一个 NVLink 域 ／ 同一个 TPU 切片",
         OR, "否 → KV 要够薄，或只把长 prompt 外送", "出了快线，KV 越厚越慢"),
        ("③ 缺的是首字，还是吞吐？", "看哪个 SLO 先破",
         GR, "缺首字 → 加 prefill；缺吞吐 → 加 decode", "配比先用公式起个头，再往先破的那边加"),
    ]
    STEP = 132
    QH = 88
    y = y0 + 10
    for i, (q, qs, col, ans, why) in enumerate(rows):
        f.box(QX, y, QW, QH, "none", BL, 10, sw=2)
        f.t(QX + 20, y + 38, q, BL, True, 17)
        f.t(QX + 20, y + 68, qs, GY, size=15)
        f.path("M%d,%d L%d,%d" % (QX + QW, y + QH / 2, LX - 6, y + QH / 2), col, 2)
        f.box(LX, y, LW, QH, "none", col, 10, sw=1.6)
        f.t(LX + 20, y + 38, ans, col, True, 17)
        f.t(LX + 20, y + 68, why, INK, size=15)
        if i < len(rows) - 1:
            f.path("M%d,%d L%d,%d" % (QX + QW / 2, y + QH, QX + QW / 2, y + STEP - 4), BL, 2)
            f.t(QX + QW / 2 + 12, y + QH + 28, "是 → 才拆，接着问怎么拆" if i == 1 else "是", BL, True, 15)
        y += STEP
    f._pan = None
    yb = y - STEP + QH
    yb = f.src(yb + 10,
               "📌 MiMo：tpu/sglang-jax/mimo/README.md:8–12、81–88（TPU v7x 每 pod 4 芯片，16K 进 4K 出，request_rate=inf，total tok/s）。"
               "Qwen3-Coder-480B：tpu/tpu-inference/Qwen3-Coder-480B/README.md:1005–1021（两台 v7x-8，TTFT 中位数）。",
               "📌 配比公式：gpu/inference/a4x-max/deepseek-v4/SGLANG-V4PRO-RUNBOOK.md §12.2（8 × 11,200 ÷ 1,024 ≈ 87.5 req/s，× 8,192 ≈ 71.7 万 tok/s，÷ 7.28 万 ≈ 9.8）。"
               "跨域退化：同文件 :63。PrFaaS：arXiv 2604.15039。",
               "⚠️ 两组 TPU 例子都是单点测量，不是扫描。")
    f.save("fig6-decide.svg", yb + 14)


def fig_afd_ascend():
    f = Fig(W, "vLLM 的 AFD 插件在昇腾 910C 上跑 DeepSeek-V3.2 的饱和 decode 吞吐，按每块 die 算。三行是三种部署，"
               "左边的小人一个代表一块 die，蓝色算注意力、紫色算专家。不拆：64 块 die 每块两样都干。"
               "拆开一：48 块算注意力、16 块算专家，专家那边吃不饱，16K 输入每块 die 220.3，比不拆的 232.6 少 5.3%，32K 少 10%。"
               "拆开二：注意力加到 64 块、专家还是 16 块，16K 涨到 258.9，多 11.3%，32K 多 9%。注意力给少了，拆开反而慢")
    y0 = f.header("AFD 拆开不保证更快　——　<tspan font-weight=\"700\">算注意力的给少了，算专家的就在等</tspan>",
                  "vLLM AFD 插件，昇腾 910C，DeepSeek-V3.2（权重和激活值都压成 8 位），decode 跑满。"
                  "die 是芯片上的一块计算单元，这里当一张卡数；分母是部署的全部 die",
                  [(BL, "小人：算注意力的 die"), (PU, "小人：算专家（FFN）的 die"), (GY2, "小人：两样都干 ／ 柱：不拆"), (OR, "柱：比不拆慢"), (GR, "柱：比不拆快")])
    CFG_ = [("不拆：64 块，每块两样都干", GY2, [(64, GY2)], None),
            ("拆开：48 块算注意力 ＋ 16 块算专家", OR, [(48, BL), (16, PU)], ("专家这边吃不饱", OR)),
            ("拆开：64 块算注意力 ＋ 16 块算专家", GR, [(64, BL), (16, PU)], ("专家更忙，仍有余力", GR))]
    S, G = 15, 4                    # 小人尺寸、间距；每排 16 个
    RS = S + G + 2                  # 行距

    def group_h(groups):
        return sum(((n - 1) // 16) * RS + S + 2 + 12 for n, _ in groups) - 12

    RH = [max(30 + group_h(g), 22 + 46 + 30) + 30 for _, _, g, _ in CFG_]
    PH = 30 + 16 + sum(RH)
    py = f.panel(0, y0, W, PH, "每块 die 每秒出多少 token（左：怎么分工；右：16K 和 32K 输入两种长度）", GR)
    BX, BW = 620, 640
    VMAX = 300.0
    y = py + 20
    for ci, (name, col, groups, note) in enumerate(CFG_):
        f.t(24, y + 16, name, col, True, 16)
        gy = y + 30
        for n, c in groups:
            for k in range(n):
                r, q = divmod(k, 16)
                f.icon("person", 24 + q * (S + G), gy + r * RS, S, S + 2, col=c, tint=c)
            last = gy + ((n - 1) // 16) * RS
            if c == PU and note:
                f.t(24 + 16 * (S + G) + 14, last + 15, note[0], note[1], True, 15)
            gy = last + S + 2 + 12
        # 右：两根条，竖向居中
        bt = y + (RH[ci] - 30 - (46 + 30)) / 2 + 10
        for j, k in enumerate(("16K", "32K")):
            vals = AFD[k]
            v = vals[ci]
            by = bt + j * 46
            f.t(BX - 14, by + 21, "%s 输入" % k, INK, True, 15, "end")
            f.box(BX, by, BW * v / VMAX, 30, col, col, 3)
            lab = "%.1f" % v
            if ci:
                lab += "（%+.1f%%）" % ((v / vals[0] - 1) * 100)
            f.t(BX + BW * v / VMAX + 10, by + 21, lab, col, True, 15)
        if ci < 2:
            f.line(16, y + RH[ci] - 12, W - 16, y + RH[ci] - 12, LINE, 1, arrow=False)
        y += RH[ci]
    f._pan = None
    yb = y0 + PH
    yb = f.src(yb + 10,
               "📌 vllm.ai/blog/2026-07-23-vllm-afd-plugin ／ github.com/vllm-project/afd-plugin：AFD 是外部插件，不是 vLLM 原生功能；"
               "vLLM 的 AFD RFC #22799 于 2026-05-31 被机器人以长期不活跃关闭，没有合入。博客原话：只拆开，不保证吞吐提升；"
               "测过的配比下 FFN 还有余力，没测更高的 attention 比例。",
               "⚠️ 48A16F ／ 64A16F 模拟的是逻辑上的 192A64F ／ 256A64F（博客原文）；专家路由换成了强制均衡（输出会变）。只能看趋势。"
               "「专家这边吃不饱」是按博客「FFN 还有余力」的定性说法画的，没有利用率数字。")
    f.save("fig6-afd-ascend.svg", yb + 14)


def fig_split_pod():
    f = Fig(W, "MiMo-V2-Flash 在 TPU v7x 上，16K 输入 4K 输出。左边是总吞吐：batch 64 时不拆 11.70K、1P1D 10.55K、2P1D 10.83K；"
               "batch 128 时不拆 12.78K、1P1D 13.64K、2P1D 15.31K，2P1D 最高。但 2P1D 用了 3 个 pod，另外两种是 2 个。"
               "右边按 pod 平均：batch 64 是 5.85、5.28、3.61；batch 128 是 6.39、6.82、5.10，1P1D 比不拆高 6.7%，2P1D 低 20%，"
               "左边最高的那根，在右边成了同组最矮")
    y0 = f.header("按 pod 算拆不拆　——　<tspan font-weight=\"700\">总量最高的 2P1D，按 pod 算成了同组最矮</tspan>",
                  "MiMo-V2-Flash · TPU v7x（每 pod 4 芯片）· 16K 进 4K 出；单位 K tok/s（total，含输入）",
                  [(GY2, "不拆（2 pod）"), (BL, "1P1D（2 pod）"), (OR, "2P1D（3 pod）")])
    COLS = {"不拆": GY2, "1P1D": BL, "2P1D": OR}
    PH, HW = 380, (W - 20) / 2
    for side, (x0, title, col, vmax, per) in enumerate((
            (0, "总吞吐（README 原值）", INK, 17.0, False),
            (HW + 20, "每 pod 吞吐（本课推导：总吞吐 ÷ pod 数）", OR, 8.0, True))):
        py = f.panel(x0, y0, HW, PH, title, col)
        BASE, AH = py + PH - 92, PH - 170
        BWD, BG = 70, 26
        for g, (gname, rows) in enumerate(SPLIT):
            gx = x0 + 60 + g * (HW - 80) / 2 + ((HW - 80) / 2 - (3 * BWD + 2 * BG)) / 2
            for i, (nm, tot, npod) in enumerate(rows):
                v = tot / npod if per else tot
                bx = gx + i * (BWD + BG)
                bh = AH * v / vmax
                c = COLS[nm]
                f.box(bx, BASE - bh, BWD, bh, c, c, 3)
                lab = _r2(v) if per else "%.2fK" % v
                ty = BASE - bh - 10
                if per:
                    f.t(bx + BWD / 2, ty, "推导", GY, size=12, anchor="middle")
                    ty -= 18
                f.t(bx + BWD / 2, ty, lab, c, True, 16, "middle")
                if per and gname == "batch 128" and nm != "不拆":
                    d = (v / (rows[0][1] / rows[0][2]) - 1) * 100
                    f.t(bx + BWD / 2, ty - 22, ("%+.1f%%" % d if nm == "1P1D" else "%+.0f%%" % d).replace("-", "−"),
                        GR if d > 0 else RD, True, 16, "middle")
                f.t(bx + BWD / 2, BASE + 22, "%d pod" % npod, GY, size=14, anchor="middle")
            f.t(gx + (3 * BWD + 2 * BG) / 2, BASE + 50, gname, INK, True, 16, "middle")
        f.line(x0 + 30, BASE, x0 + HW - 30, BASE, GY2, 1.2, arrow=False)
    f._pan = None
    yb = f.band(y0 + PH + 16, "ok", "2P1D 多出来的总量，是多一个 pod 换来的", [
        "batch 64 时，拆了不管按总量还是按 pod 都亏。",
    ])
    yb = f.src(yb + 10,
               "📌 tpu/sglang-jax/mimo/README.md:8-12、81-88（总体结果表；request_rate=inf，total tok/s 含输入；README 自注只看趋势、未考虑 SLO）。",
               "本课推导：每 pod ＝ total ÷ pod 数；+6.7% ＝ 6.82 ÷ 6.39 − 1（README 写约 6.8%，按表算 6.7%）；−20% ＝ 5.10 ÷ 6.39 − 1。")
    f.save("fig6-split-pod.svg", yb + 14)


if __name__ == "__main__":
    fig_split_pod()
    fig_2p1d()
    fig_prefill_sweep()
    fig_decide()
    fig_afd_ascend()
