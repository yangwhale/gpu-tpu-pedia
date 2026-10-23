# -*- coding: utf-8 -*-
r"""专题六 · 第十一节「投机解码：搬一趟，出好几个字」的五张静态图。

fig6-spec-step.svg —— 回答：投机解码一步到底做了什么，为什么「接受到第几个就前进几个」？
  · 数字：没有实测数。示意时间线，格子长短只表达「草稿很便宜、验证一步约等于普通一步」。
  · 按定义必然成立的部分是真算的：每轮前进 ＝ 接受个数 ＋ 1（大模型在第一个被拒的位置自己给一个）；
    同样长的时间里两条泳道各出几个字，脚本里 assert。
  · 刻意没画：树形草稿（SpecInfer）、拒绝采样的概率细节（写在正文一句话里）。

fig6-spec-bytes.svg —— 回答：验证 n 个 token 比验证 1 个多读多少字节？收益怎么从接受长度里算出来？
  · 数字：DeepSeek-V3 config.json（HF deepseek-ai/DeepSeek-V3）：hidden 7168、moe_intermediate 2048、
    61 层里前 3 层稠密（first_k_dense_replace）、256 个路由专家挑 8 个。
    每个路由专家 3 × 7168 × 2048 个参数；路由专家总量 ＋ 其余 ≈ 671B 用技术报告的总参数当外部锚点。
  · 本课推导：n 个 token 各自独立、均匀地挑专家时，并起来平均碰到 256 × (1 − (1 − 8/256)^n) 个专家。
    真实路由不均匀、同一请求里相邻 token 挑的专家有重合，所以这是偏高的估计。
  · K3 实测点：gpu/inference/a4x-max/kimi-k3/SGLANG-K3-RUNBOOK.md:1089（bs=1 TPOT 11.1 → 2.31 ms）
    和 :979（accept len 6.96 起）。一步耗时 ＝ TPOT × 接受长度，是本课推导。
  · 刻意没画：草稿模型自己的耗时（DSpark 的草稿也在这一步里，已含在 K3 实测里）。

fig6-spec-conc.svg —— 回答：为什么并发一高、上下文一长、一开 EP+DP，收益就缩甚至变负？
  · 数字：K3 SGLANG-K3-RUNBOOK.md:1089-1092（conc 1/8/32/64）、:1057-1058（32K）；
    B200 V4-Flash vLLM gpu/inference/a4/deepseek-v4/README.vllm.md:265-277；
    GB300 V4-Pro SGLang gpu/inference/a4x-max/deepseek-v4/SGLANG-V4PRO-RUNBOOK.md:900-903。
  · 刻意没画：K3 32K 的 conc 8/32（runbook 说集群被删、没测）。

fig6-spec-temp.svg —— 回答：为什么压测必须写明温度？
  · 数字：GB300 V4-Pro vLLM gpu/inference/a4x-max/deepseek-v4/VLLM-V4PRO-RUNBOOK.md:406-410；
    K3 官方接受长度 gpu/inference/a4x-max/kimi-k3/VLLM-K3-RUNBOOK.md:233-237。

fig6-spec-tree.svg —— 回答：这么多投机解码的名字，按时间排，各自把草稿从哪里拿？
  · 真时间轴 2022–2026，日期与 s11.5 列表同一套；年份按 arXiv 首版，每个方法都核过 arXiv 摘要页（2026-10-15）。Prompt Lookup 只有 GitHub 仓库。
  · 图上只写「草稿从哪来」，倍数留在正文列表（各家口径不同，不横比）。
"""

from topic03_draw import Fig, wpx, BL, OR, GR, RD, PU, CY, GY, INK, GY2, LINE

W = 1400

# ══ fig6-spec-step：示意几何 ═══════════════════════════════════════════
K = 4                       # 每轮草稿猜 4 个（示意）
T_STEP, T_DRAFT, T_VERIFY = 10, 1, 11      # 普通一步 10 格；草稿每个 1 格；验证一步 11 格（示意）
ROUNDS = [2, 4]             # 第 1 轮接受 2 个，第 2 轮 4 个全对（示意，和 anim-spec 同一组数）
ADV = [a + 1 for a in ROUNDS]               # 每轮前进 ＝ 接受个数 ＋ 1
assert ADV == [3, 5]
T_SPEC = len(ROUNDS) * (K * T_DRAFT + T_VERIFY)
assert T_SPEC == 30
N_PLAIN = T_SPEC // T_STEP                  # 同样 30 格时间，普通 decode 出几个字
assert N_PLAIN == 3 and sum(ADV) == 8

# ══ fig6-spec-bytes：V3 字节账（本课推导）══════════════════════════════
V3_H, V3_MOE_I, V3_LAYERS, V3_DENSE = 7168, 2048, 61, 3
V3_E, V3_TOPK = 256, 8
V3_TOTAL, V3_ACTIVE = 671e9, 37e9
EXPERT_P = 3 * V3_H * V3_MOE_I              # gate / up / down 三块
assert EXPERT_P == 44040192
MOE_LAYERS = V3_LAYERS - V3_DENSE
assert MOE_LAYERS == 58
ROUTED_ALL = V3_E * MOE_LAYERS * EXPERT_P
OTHER = V3_TOTAL - ROUTED_ALL               # 注意力、稠密层、共享专家、词表……
ACTIVE_DERIVED = OTHER + V3_TOPK * MOE_LAYERS * EXPERT_P
assert abs(ACTIVE_DERIVED / V3_ACTIVE - 1) < 0.02, ACTIVE_DERIVED   # 外部锚点：报告说激活 37B
assert abs(OTHER / 1e9 - 17.1) < 0.05


def experts_hit(n):
    """n 个 token 各挑 8 个、彼此独立且均匀时，一层里平均碰到几个不同的专家（本课推导，偏高）。"""
    return V3_E * (1 - (1 - V3_TOPK / V3_E) ** n)


def step_gb(n):
    """FP8（每参数 1 字节）下，验证 n 个 token 一步要读的权重 GB。"""
    return (OTHER + experts_hit(n) * MOE_LAYERS * EXPERT_P) / 1e9


assert abs(experts_hit(1) - 8) < 1e-9
assert abs(experts_hit(8) - 57.5) < 0.1
assert abs(step_gb(1) - 37.5) < 0.1 and abs(step_gb(8) - 163.8) < 0.05
BYTES_X8 = step_gb(8) / step_gb(1)
assert round(BYTES_X8, 1) == 4.4

# K3 实测（SGLANG-K3-RUNBOOK.md:1089；:979）
K3_TPOT_NS, K3_TPOT_SP, K3_AL = 11.1, 2.31, 6.96
K3_STEP_SP = K3_TPOT_SP * K3_AL             # 开投机后一步的耗时 ＝ 每字间隔 × 每步出几个字（本课推导）
assert abs(K3_STEP_SP - 16.1) < 0.05
K3_RATIO = K3_STEP_SP / K3_TPOT_NS          # 一步变慢了几倍
assert abs(K3_RATIO - 1.45) < 0.005
K3_TPOT_X = K3_TPOT_NS / K3_TPOT_SP
assert abs(K3_TPOT_X - 4.81) < 0.005 and abs(K3_AL / K3_RATIO - K3_TPOT_X) < 1e-9

# ══ fig6-spec-conc ═══════════════════════════════════════════════════════
K3_CONC = [(1, 87.8, 370.4), (8, 434.4, 1241.9), (32, 1129.7, 2138.4), (64, 1511.1, 2629.3)]
K3_X = [round(sp / ns, 2) for _, ns, sp in K3_CONC]
assert K3_X == [4.22, 2.86, 1.89, 1.74], K3_X
K3_32K = (83.3, 109.8)
assert round(K3_32K[1] / K3_32K[0], 2) == 1.32
# B200 V4-Flash vLLM EP+DP8，MTP n=2 vs 无 MTP（README.vllm.md:265-277）
B200 = [(1, 63, 72), (8, 212, 219), (40, 620, 883), (60, 1175, 1664), (80, 1466, 1960),
        (100, 1602, 2546), (200, 2407, 3053), (300, 2881, 3874), (400, 3217, 4052), (600, 3700, 4424)]
B200_PCT = [round((m / b - 1) * 100) for _, m, b in B200]
assert B200_PCT == [-12, -3, -30, -29, -25, -37, -21, -26, -21, -16], B200_PCT
assert max(B200_PCT) == -3 and min(B200_PCT) == -37
# GB300 V4-Pro SGLang MTP steps 1 → 2（SGLANG-V4PRO-RUNBOOK.md:902-903）
V4_PEAK = (11851, 10279)
V4_E2E = (6887, 8802)
assert round((V4_PEAK[1] / V4_PEAK[0] - 1) * 100) == -13
assert round((V4_E2E[1] / V4_E2E[0] - 1) * 100) == 28

# ══ fig6-spec-temp ═══════════════════════════════════════════════════════
T1_ACC, T0_ACC = 1.16, 34.0                 # 接受率 %
T1_TPS, T0_TPS = 944, 2911                  # output tok/s
assert round(T0_TPS / T1_TPS, 1) == 3.1
assert round(T0_ACC / T1_ACC) == 29
K3_AL_LOW, K3_AL_HIGH = 4.73, 2.61          # K3 官方：低熵 / 高熵任务每步接受 token 数


# ════════════════════════════════════════════════════════════════════════
def fig_step():
    f = Fig(W, "投机解码一步做了什么，示意时间线。上面一条是普通 decode：每一步把权重读一遍，出一个字，"
               "三十格时间出三个字。下面一条是投机解码：先用便宜的草稿连猜四个字，再让大模型读一遍权重、"
               "一次把四个字都验掉。第一轮前两个对、第三个错，大模型在第三个位置给出自己的字，后面的扔掉，于是一轮前进三个；"
               "第二轮四个全对，大模型顺手算出的第五个也白送，前进五个。同样三十格时间出了八个字")
    y0 = f.header("先猜后验　——　<tspan font-weight=\"700\">大模型读一遍权重，验掉一整串草稿</tspan>",
                  "横轴是时间（示意，不按比例）。每一格字：绿 ＝ 草稿猜对、被收下；红 ＝ 猜错、连同后面的全扔掉；"
                  "蓝 ＝ 大模型自己给的字（第一个错的位置补上；全对时白送下一个）",
                  [(GY2, "大模型读一遍权重"), (PU, "草稿猜一个字"), (GR, "收下"), (RD, "扔掉"), (BL, "大模型补的字")])
    PH = 400
    py = f.panel(0, y0, W, PH, "同样长的时间，两种走法", INK)
    X0, U = 210, 34                 # 时间零点、每格多少像素
    assert X0 + T_SPEC * U < W - 30

    def tok(x, y, col, lab):
        f.box(x, y, 28, 28, col, col, 4)
        f.t(x + 14, y + 19, lab, "#ffffff", True, 13, "middle")

    # ① 普通 decode
    ry = py + 50
    f.t(24, ry + 26, "普通 decode", INK, True, 16)
    f.t(24, ry + 48, "一步一个字", GY, size=13)
    for i in range(N_PLAIN):
        x = X0 + i * T_STEP * U
        f.box(x + 2, ry + 8, T_STEP * U - 4, 34, "#e8eaed", GY2, 5)
        f.t(x + T_STEP * U / 2, ry + 30, "读一遍权重", GY, size=13, anchor="middle")
        tok(x + T_STEP * U - 32, ry + 52, BL, str(i + 1))

    # ② 投机
    ry2 = py + 175
    f.t(24, ry2 + 26, "投机解码", INK, True, 16)
    f.t(24, ry2 + 48, "猜 4 个，验 1 次", GY, size=13)
    x = X0
    made = 0
    for r, acc in enumerate(ROUNDS):
        for j in range(K):
            f.box(x + j * U + 2, ry2 + 8, U - 4, 34, "#f3e8fd", PU, 4)
            f.t(x + j * U + U / 2, ry2 + 30, "猜", PU, True, 12, "middle")
        xv = x + K * U
        f.box(xv + 2, ry2 + 8, T_VERIFY * U - 4, 34, "#e8eaed", GY2, 5)
        f.t(xv + T_VERIFY * U / 2, ry2 + 30, "读一遍权重，验 %d 个" % K, GY, size=13, anchor="middle")
        # 验证结果：前 acc 个收下，第 acc+1 个扔掉并由大模型补一个，后面的扔掉
        tx = xv + 10
        for j in range(K):
            col = GR if j < acc else RD
            tok(tx + j * 34, ry2 + 52, col, "✓" if j < acc else "✕")
        tok(tx + acc * 34, ry2 + 88, BL, "补" if acc < K else "送")
        made += acc + 1
        f.t(xv + T_VERIFY * U - 8, ry2 + 140, "这一轮前进 %d 个（收下 %d ＋ %s 1）" % (acc + 1, acc, "补" if acc < K else "白送"),
            INK, True, 13.5, "end")
        x = xv + T_VERIFY * U
    assert made == sum(ADV)
    # 时间尺
    ya = py + 360
    f.line(X0, ya, X0 + T_SPEC * U, ya, GY2, 1.4)
    f.t(X0 + T_SPEC * U, ya - 10, "同一段时间", GY, size=13, anchor="end")
    f.t(X0 - 10, ya + 5, "时间 →", GY, size=13, anchor="end")
    f.t(X0 + T_SPEC * U / 2, ya - 10,
        "普通 decode 出 %d 个字，投机解码出 %d 个字" % (N_PLAIN, sum(ADV)), INK, True, 15, "middle")
    yb = f.band(py + PH + 16, "ok", "接受到第几个，就前进几个", [
        "草稿便宜，贵的是大模型读权重那一下；只要那一下没变慢多少，猜对几个就白赚几个字。",
        "猜错只浪费草稿那点功夫：第一个错的位置大模型本来就算出来了，所以每轮至少前进 1 个，不会比不开更少出字。",
    ])
    yb = f.src(yb + 10,
               "⚠️ 示意：格子长度（普通一步 10 格、草稿每个 1 格、验证一步 11 格）和接受个数都是举例，不是实测。",
               "机制：Leviathan 等 arXiv 2211.17192；Chen 等 arXiv 2302.01318。收下第 i 个草稿的概率是 min(1, p/q)，"
               "被拒时从 max(0, p − q) 归一化后重新抽，输出分布与大模型逐个生成相同。",
               "注：「不会比不开更少出字」只说字数；如果验证一步明显变慢、草稿又总猜错，墙钟时间照样可能变长（见并发那张图）。")
    f.save("fig6-spec-step.svg", yb + 14)


# ════════════════════════════════════════════════════════════════════════
def fig_bytes():
    f = Fig(W, "左图：DeepSeek-V3 一步验证 n 个 token 要读多少权重，FP8 每参数一字节。浅蓝是每个 token 都要经过的部分，"
               "约 171 亿参数，验证几个 token 都只读一遍；深蓝是路由专家，每个 token 挑自己的 8 个，"
               "验证的 token 越多，碰到的专家并起来越多。验证 1 个读约 37.5 GB，验证 8 个约 163.8 GB，是 4.4 倍。"
               "右图：Kimi K3 实测的时间账。不开投机，搬一趟权重 11.1 毫秒，带回 1 个字；开投机，搬一趟变成约 16.1 毫秒，"
               "用时是不开的 1.45 倍，但平均带回 6.96 个字。摊到每个字，从 11.1 毫秒降到 2.31 毫秒，降到 1/4.81")
    y0 = f.header("多验证几个 token 贵不贵　——　<tspan font-weight=\"700\">稠密部分白送，MoE 专家不白送</tspan>",
                  "左：V3 验证一步要读的权重（本课推导，FP8）。右：Kimi K3 实测，搬一趟变慢了，但一趟带回的字多得多",
                  [("#aecbfa", "每个 token 都经过的部分"), (BL, "路由专家（各挑各的）"), (GR, "开投机后的一趟")])
    PH = 470
    LW = 640
    py = f.panel(0, y0, LW, PH, "V3 验证 n 个 token，一步读多少 GB", BL)
    PX, PY0, PW, PHH = 90, py + 50, 500, 320
    GMAX = 180
    f.box(PX, PY0, PW, PHH, "none", LINE, 4)
    for g in (0, 60, 120, 180):
        yy = PY0 + PHH - PHH * g / GMAX
        f.t(PX - 10, yy + 5, str(g), GY, size=14, anchor="end")
    f.t(PX - 60, PY0 - 14, "GB", GY, size=14)
    bw = PW / 8
    for n in range(1, 9):
        x = PX + (n - 1) * bw + 10
        h_o = PHH * (OTHER / 1e9) / GMAX
        h_t = PHH * step_gb(n) / GMAX
        f.box(x, PY0 + PHH - h_o, bw - 20, h_o, "#aecbfa", "#669df6", 2)
        f.box(x, PY0 + PHH - h_t, bw - 20, h_t - h_o, BL, BL, 2)
        f.t(x + (bw - 20) / 2, PY0 + PHH + 24, str(n), GY, size=15, anchor="middle")
        if n in (1, 8):
            f.t(x + (bw - 20) / 2, PY0 + PHH - h_t - 8, "%.1f" % step_gb(n), INK, True, 15, "middle")
    f.t(PX + PW / 2, PY0 + PHH + 52, "一步验证的 token 数 n", GY, True, 15, anchor="middle")
    f.t(PX + 16, PY0 + 30, "验证 8 个 ＝ 验证 1 个的 %.1f 倍" % BYTES_X8, RD, True, 17)
    f.t(PX + 16, PY0 + 56, "浅蓝 %.1f GB 不随 n 变" % (OTHER / 1e9), GY, False, 15)

    # 右：K3 一趟的时间账
    RX = LW + 20
    RW = W - RX
    py2 = f.panel(RX, y0, RW, PH, "Kimi K3 实测：搬一趟，带回几个字", GR)
    TX = RX + 150                      # 时间零点
    MS = 30.0                          # 每毫秒多少像素
    assert TX + 17 * MS < W - 20

    def lane(yy, lab, ms, col, fill, n_tok, tok_lab):
        f.t(RX + 24, yy + 26, lab, INK, True, 17)
        f.box(TX, yy, ms * MS, 40, fill, col, 6)
        f.t(TX + 14, yy + 26, "搬一趟 %.1f ms" % ms, col, True, 16)
        for k in range(n_tok):
            f.box(TX + k * 36, yy + 56, 30, 30, GR, GR, 5)
        f.t(TX + n_tok * 36 + 10, yy + 78, tok_lab, INK, True, 16)

    ya = py2 + 40
    lane(ya, "不开投机", K3_TPOT_NS, GY2, "#e8eaed", 1, "带回 1 个字")
    yb_ = ya + 140
    lane(yb_, "开投机", K3_STEP_SP, GR, "#e6f4ea", 7, "平均带回 %.2f 个字" % K3_AL)
    # 多花的那一截：从 11.1 ms 到 16.1 ms
    x1, x2 = TX + K3_TPOT_NS * MS, TX + K3_STEP_SP * MS
    f.path("M%d,%d L%d,%d" % (x1, ya + 44, x1, yb_ + 44), GY2, 1.2, dash="4,4", arrow=False)
    f.path("M%d,%d L%d,%d L%d,%d L%d,%d" % (x1, yb_ - 8, x1, yb_ - 14, x2, yb_ - 14, x2, yb_ - 8),
           RD, 1.6, arrow=False)
    f.t((x1 + x2) / 2, yb_ - 22, "用时 %.2f×，对不开" % K3_RATIO, RD, True, 16, "middle")
    # 摊到每个字
    yc = yb_ + 150
    f.path("M%d,%d L%d,%d" % (RX + 24, yc - 30, W - 24, yc - 30), LINE, 1, arrow=False)
    f.t(RX + 24, yc, "摊到每个字", GY, True, 16)
    f.t(RX + 24, yc + 36, "%.1f ms → %.2f ms，降到 1/%.2f" % (K3_TPOT_NS, K3_TPOT_SP, K3_TPOT_X), GR, True, 22)
    f.t(RX + 24, yc + 66, "一趟慢一点不要紧，带回的字多得多", GY, size=15)
    yb = f.src(y0 + PH + 6,
               "V3 结构取自 HF deepseek-ai/DeepSeek-V3 config.json：hidden 7,168、moe_intermediate 2,048、61 层中前 3 层稠密、256 个路由专家挑 8 个。"
               "每个专家 3 × 7,168 × 2,048 ≈ 4,404 万参数。",
               "⚠️ 本课推导：其余部分 ＝ 671B − 256 × 58 × 专家大小 ≈ %.1fB；由此推出每 token 激活 ≈ %.1fB，与技术报告的 37B 差 2%% 以内（外部锚点）。"
               % (OTHER / 1e9, ACTIVE_DERIVED / 1e9),
               "⚠️ 本课推导：n 个 token 独立、均匀地挑专家时，一层平均碰到 256 × (1 − (1 − 8/256)^n) 个；实际路由不均匀、相邻 token 常挑同一批专家，所以深蓝那段是偏高的估计。",
               "K3：gpu/inference/a4x-max/kimi-k3/SGLANG-K3-RUNBOOK.md:1089（GB300 TP8 跨两节点，ISL 4096／OSL 1024，temperature 0，bs=1，TPOT 11.1 → 2.31 ms）、:979（accept len 6.96，block 7）。",
               "⚠️ 本课推导：开投机后一趟耗时 ＝ 2.31 ms × 6.96 ≈ 16.1 ms；每字间隔降到 1/4.81（11.1 ÷ 2.31 ≈ 4.81）。runbook 报的端到端 4.22× 含首字时间，所以比 4.81 小。"
               "右图第 7 个方块代表平均值里的 0.96 个，示意。",
               "注：K3 是 8 张卡跑 bs=1，一步里除了读权重还有通信和 kernel 之间的空转，所以「用时 1.45×」不能全归给多读的专家。")
    f.save("fig6-spec-bytes.svg", yb + 14)


# ════════════════════════════════════════════════════════════════════════
def fig_conc():
    f = Fig(W, "Kimi K3 在 GB300 上开 DSpark 投机解码，输出吞吐的加速比随并发下降：并发 1 是 4.22 倍，8 是 2.86 倍，"
               "32 是 1.89 倍，64 是 1.74 倍；同样并发 1，上下文从 4K 拉到 32K，只剩 1.32 倍")
    y0 = f.header("投机解码的收益会缩　——　<tspan font-weight=\"700\">人一多、上下文一长，白送的那部分就变少</tspan>",
                  "Kimi K3 · GB300 · SGLang TP8。柱高 ＝ 开投机后的输出吞吐 ÷ 不开；虚线 ＝ 1，不赚不赔",
                  [(GR, "4K 上下文，并发 1 → 64"), (PU, "32K 上下文，并发 1")])
    PH = 400
    py = f.panel(0, y0, W, PH, "开投机后的吞吐是不开的几倍", GR)
    BX0, BY0, BH = 150, py + 56, 260
    SMAX = 4.5

    def Yb(v):
        return BY0 + BH - BH * v / SMAX
    f.path("M%d,%d L%d,%d" % (BX0 - 20, Yb(1), W - 40, Yb(1)), GY2, 1.4, dash="4,4", arrow=False)
    f.t(BX0 - 30, Yb(1) + 6, "1×", GY, size=16, anchor="end")
    bw, gap = 130, 60
    for i, ((c, ns, sp), x_) in enumerate(zip(K3_CONC, K3_X)):
        x = BX0 + i * (bw + gap)
        f.box(x, Yb(x_), bw, Yb(0) - Yb(x_), GR, GR, 4)
        f.t(x + bw / 2, Yb(x_) - 10, "%.2f×" % x_, INK, True, 20, "middle")
        f.t(x + bw / 2, BY0 + BH + 28, "并发 %d" % c, GY, True, 16, "middle")
    xs = BX0 + 4 * (bw + gap) + 60
    f.path("M%d,%d L%d,%d" % (xs - 30, BY0 - 10, xs - 30, BY0 + BH + 40), LINE, 1, arrow=False)
    v32 = K3_32K[1] / K3_32K[0]
    f.box(xs, Yb(v32), bw, Yb(0) - Yb(v32), PU, PU, 4)
    f.t(xs + bw / 2, Yb(v32) - 10, "%.2f×" % v32, INK, True, 20, "middle")
    f.t(xs + bw / 2, BY0 + BH + 28, "32K · 并发 1", PU, True, 16, "middle")
    f.t(xs + bw + 24, Yb(3.4), "同样并发 1，", PU, True, 16)
    f.t(xs + bw + 24, Yb(3.4) + 24, "上下文 4K → 32K：", PU, True, 16)
    f.t(xs + bw + 24, Yb(3.4) + 48, "4.22× → 1.32×", PU, True, 18)
    yb = f.src(y0 + PH + 6,
               "K3：gpu/inference/a4x-max/kimi-k3/SGLANG-K3-RUNBOOK.md:1089-1092（GB300，TP8 跨两节点，ISL 4096／OSL 1024，random，temperature 0，开环，只报热轮；输出 tok/s 总量）、:1057-1058（32K，只测了并发 1；8／32 没测，集群已删）。",
               "注：K3 并发 64 时开投机的 TPOT 仍只有不开的 40%，这一档吞吐和延迟同时变好。",
               "另两组相关实测的出处 —— B200：gpu/inference/a4/deepseek-v4/README.vllm.md:265-277（DeepSeek-V4-Flash，8× B200，vLLM v0.20.0，EP+DP=8，MTP num_speculative_tokens=2；接受率约 54%；十档并发 −3% 到 −37%）。",
               "V4-Pro：gpu/inference/a4x-max/deepseek-v4/SGLANG-V4PRO-RUNBOOK.md:900-911（steps 1／draft 2 → steps 2／draft 3；接受长度 1.88 → 2.36（+26%），反推单步耗时 115.3 → 162.5 ms（+41%）；decode 峰值 −13%、端到端 +28%）。"
               "B200 那套是 EP+DP 高吞吐配置，结论不能搬到 TP-only 低延迟配置。")
    f.save("fig6-spec-conc.svg", yb + 14)


# ════════════════════════════════════════════════════════════════════════
def fig_temp():
    f = Fig(W, "用对答案打比方。题目是下一个字填什么，最可能的答案是「好」，草稿是贪心的，每次都写最可能的「好」。"
               "上一行温度 1.0：大模型掷骰子决定标准答案，大多掷出别的字，和草稿对不上，被扔掉。"
               "GB300 上的 DeepSeek-V4-Pro 实测接受率只有 1.16%，吞吐 944 tok/s。"
               "下一行温度 0：骰子永远停在最可能的那个字，草稿只要猜的就是它就能对上。实测接受率 34%，吞吐 2,911 tok/s，是温度 1.0 的 3.1 倍")
    y0 = f.header("温度：大模型出答案时掷不掷骰子　——　<tspan font-weight=\"700\">草稿总写最可能的字，骰子一掷就对不上</tspan>",
                  "题目：「今天天气真＿」，最可能填「好」。草稿是贪心的，永远交「好」；大模型按温度决定标准答案（示意）",
                  [(PU, "草稿交的答案"), (RD, "对不上，扔掉"), (GR, "对上，收下")])
    LH = 190
    WORDS1 = ["不错", "热", "好", "晴", "冷"]          # 示意：温度 1.0 掷出的五次
    WORDS0 = ["好"] * 5

    def die(x, y, s, col, dots):
        f.box(x, y, s, s, "#fff", col, 10, 1.8)
        for dx, dy in dots:
            f.p.append('<circle cx="%.1f" cy="%.1f" r="5" fill="%s"/>' % (x + s * dx, y + s * dy, col))

    rows = [("温度 1.0", "服务端默认", "掷骰子", WORDS1, "%.2f%%" % T1_ACC, "{:,} tok/s".format(T1_TPS), RD,
             [(.25, .25), (.75, .25), (.5, .5), (.25, .75), (.75, .75)]),
            ("温度 0", "请求里写明", "只认最可能那个", WORDS0, "%.0f%%" % T0_ACC, "{:,} tok/s".format(T0_TPS), GR,
             [(.5, .5)])]
    yy = y0
    for lab, sub_, how, words, acc, tps, col, dots in rows:
        py = f.panel(0, yy, W, LH, lab + "　·　" + sub_, col)
        cy = py + (LH - 30) / 2
        # 草稿交的答案
        f.box(30, cy - 34, 130, 68, "#f3e8fd", PU, 8)
        f.t(95, cy - 6, "草稿交", PU, True, 15, "middle")
        f.t(95, cy + 22, "「好」", PU, True, 20, "middle")
        f.t(195, cy + 6, "对", GY, True, 16, "middle")
        # 大模型的骰子
        die(230, cy - 32, 64, GY, dots)
        f.t(262, cy + 56, how, GY, True, 15, "middle")
        f.line(310, cy, 360, cy, GY2, 1.6)
        # 五次标准答案
        f.t(380, cy - 44, "五道各自独立的题（举例；不是同一步的五个位置，格子数也不是比例）" if words is WORDS1
            else "同样是五道独立的题（举例）", GY, size=14)
        for k, w in enumerate(words):
            ok = (w == "好")
            c = GR if ok else RD
            x = 380 + k * 112
            f.box(x, cy - 30, 96, 60, "#e6f4ea" if ok else "#fce8e6", c, 8)
            f.t(x + 48, cy - 2, "「%s」" % w, c, True, 18, "middle")
            f.t(x + 48, cy + 22, "✓ 收下" if ok else "✕ 扔掉", c, True, 14, "middle")
        # 实测
        rx = 1000
        f.path("M%d,%d L%d,%d" % (rx - 20, py + 14, rx - 20, py + LH - 44), LINE, 1, arrow=False)
        f.t(rx + 10, cy - 18, "V4-Pro 实测接受率", GY, True, 15)
        f.t(rx + 10, cy + 20, acc, col, True, 30)
        f.t(rx + 10 + (wpx(acc, 30) + 14), cy + 20, tps, col, True, 18)
        yy += LH + 12
    f._pan = None
    yy = f.band(yy + 4, "ok", "温度 0 的实测吞吐是温度 1.0 的 %.1f 倍" % (T0_TPS / T1_TPS), [
        "温度 0 那行也没全收：实测接受率 34%，其余是草稿自己没猜中大模型最可能的那个字。"])
    yb = f.src(yy + 6,
               "⚠️ 示意：「今天天气真＿」和温度 1.0 那行掷出的五个字是举例，不是实测分布。",
               "V4-Pro：gpu/inference/a4x-max/deepseek-v4/VLLM-V4PRO-RUNBOOK.md:398-410（GB300，vLLM，4K/1K，并发 256）。generation_config 默认 temperature 1.0；DSpark 草稿 draft_sample_method 为 greedy，num_speculative_tokens=7。"
               "vllm bench serve 不发 temperature；sa-bench 硬编码 0.0。",
               "正文里 K3 的数（接受长度：代码类约 %.2f、创作类约 %.2f）：gpu/inference/a4x-max/kimi-k3/VLLM-K3-RUNBOOK.md:226-237（" % (K3_AL_LOW, K3_AL_HIGH) + "官方压测 temperature 1.0、top-p 0.95；DSpark 用 probabilistic 草稿，草稿自己也掷骰子，所以温度 1.0 也对得上；接受长度为官方参考值，本仓库未复测）。",
               "告警：gpu/inference/a4x-max/kimi-k3/PD-BACKLOG.md:175。")
    f.save("fig6-spec-temp.svg", yb + 14)


# ════════════════════════════════════════════════════════════════════════
# 真时间轴 2022–2026：六张主线卡 ＋ 一张侧枝卡。日期与 s11.5 列表同一套（2026-10-15 核过 arXiv 摘要页）；
# Prompt Lookup 没有论文，按 GitHub 仓库创建时间 2023-11。每卡只写「草稿从哪来」，⛔ 不写倍数（倍数在正文列表）。
TREE = [
    # (时间轴位置, 角标, 名字, 一句话（可两行）, 出处)
    (2022.87, "2022-11", "投机解码", ["另放一个独立小模型", "打草稿"], "Leviathan 等 2211.17192；Chen 等 2302.01318"),
    (2024.04, "2024-01", "Medusa", ["大模型头上", "加几个解码头"], "2401.10774"),
    (2024.08, "2024-01", "EAGLE", ["小模型读大模型的", "隐状态来猜"], "2401.15077、EAGLE-2 2406.16858"),
    (2024.96, "2024-12", "MTP · DeepSeek-V3", ["预训练时就多带一层，", "共享输出头"], "2412.19437"),
    (2025.17, "2025-03", "EAGLE-3", ["把大模型几层的隐状态", "拼在一起来猜"], "2503.01840"),
    (2026.53, "2026-07", "DSpark", ["半自回归一次出一小段，", "按置信度定验几个"], "2607.05147"),
]
SIDE = (2023.87, "2023-11", "Prompt Lookup", ["不用草稿模型：", "直接从 prompt 里抄"],
        "GitHub apoorvumang/prompt-lookup-decoding（2023-11）")
assert [t[0] for t in TREE] == sorted(t[0] for t in TREE)


def fig_tree():
    f = Fig(W, "投机解码的草稿从哪来，按时间排在 2022 到 2026 年的轴上。2022 年 11 月投机解码：另放一个独立小模型打草稿。"
               "2024 年 1 月 Medusa：大模型头上加几个解码头；同期 EAGLE：小模型读大模型的隐状态来猜。"
               "2024 年 12 月 DeepSeek-V3 的 MTP：预训练时就多带一层，共享输出头。2025 年 3 月 EAGLE-3：把大模型几层的隐状态拼在一起来猜。"
               "2026 年 7 月 DSpark：半自回归一次出一小段，按置信度决定验几个。侧枝 2023 年 11 月 Prompt Lookup：不用草稿模型，直接从 prompt 里抄")
    y0 = f.header("草稿从哪来　——　<tspan font-weight=\"700\">一步步长进大模型身上，或者干脆不要模型</tspan>",
                  "横轴是首次公开的时间：论文按 arXiv 首版，Prompt Lookup 没有论文，按 GitHub 仓库。每张卡只写草稿从哪来",
                  [(PU, "主线：要训练的草稿（小模型／头／层）"), (OR, "侧枝：不用草稿模型")])
    X0, X1, Y_LO, Y_HI = 60, 1340, 2022.0, 2027.0

    def xf(yr):
        return X0 + (X1 - X0) * (yr - Y_LO) / (Y_HI - Y_LO)

    CW, CH, GAP = 210, 132, 18
    cx0 = (W - (6 * CW + 5 * GAP)) / 2
    f.line(cx0 + 20, y0 + 12, cx0 + 6 * CW + 5 * GAP - 20, y0 + 12, PU, 2.2)
    f.t(cx0 + 6 * CW + 5 * GAP - 30, y0 + 4, "草稿越来越贴着大模型自己长", PU, True, 16, "end")
    ya = y0 + 30
    AY = ya + CH + 70                               # 时间轴的 y

    def card(x, y, yr, nm, desc, col, dash=None):
        f.box(x, y, CW, CH, "none", col if dash else LINE, 10, 1.2, dash=dash)
        pw = wpx(yr, 14) + 22
        f.box(x + 12, y + 12, pw, 26, "#fff", col, 13)
        f.t(x + 12 + pw / 2, y + 30, yr, col, True, 14, "middle")
        f.t(x + 14, y + 66, nm, INK if not dash else col, True, 17)
        for k, ln in enumerate(desc):
            f.t(x + 14, y + 94 + 22 * k, ln, GY, False, 15)

    # 主线卡：一排，按时间顺序；引线落到轴上各自的日期
    for i, (pos, yr, nm, desc, _) in enumerate(TREE):
        x = cx0 + i * (CW + GAP)
        card(x, ya, yr, nm, desc, PU)
        px = xf(pos)
        f.line(x + CW / 2, ya + CH, px, AY - 6, PU, 1.4, arrow=False)
        f.p.append('<circle cx="%.1f" cy="%.1f" r="6" fill="%s"/>' % (px, AY, PU))
    # 轴与年份刻度
    f.line(X0 - 20, AY, X1 + 10, AY, GY2, 1.6)
    for yr in range(2022, 2027):
        f.line(xf(yr), AY - 6, xf(yr), AY + 6, GY2, 1.4, arrow=False)
        f.t(xf(yr) + 6, AY + 26, str(yr), GY, True, 15)
    # 侧枝：轴下方，虚线橙框
    pos, yr, nm, desc, _ = SIDE
    px = xf(pos)
    sx, sy = px - CW / 2 + 60, AY + 60
    f.p.append('<circle cx="%.1f" cy="%.1f" r="6" fill="%s"/>' % (px, AY, OR))
    f.line(px, AY + 6, sx + CW / 2, sy, OR, 1.4, dash="5 4", arrow=False)
    card(sx, sy, yr, nm, desc, OR, dash="6,4")
    yb = f.src(sy + CH + 6,
               "时间取 arXiv 首版（2026-10-15 逐个核过摘要页）："
               + "；".join("%s %s" % (nm, ref) for _, _, nm, _, ref in TREE) + "。"
               "Prompt Lookup 没有论文，年份按 GitHub 仓库 apoorvumang/prompt-lookup-decoding 的创建时间（2023-11，GitHub API 核过；已集成进 vLLM 与 transformers）。"
               "Medusa 与 EAGLE 都是 2024-01，轴上两点挨着。",
               "没画进图的：SpecInfer 2305.09781 把草稿排成树；SuffixDecoding 2411.04975 从历史输出里抄；"
               "扩散草稿 DiffuSpec 2510.02358、DFlash 2602.06036；Lookahead 2402.02057。P-EAGLE、PARD、LayerSkip 等也没画。",
               "注：各家倍数都是原作者在自己的模型和负载上测的，口径不同，不横比，所以图上不画。")
    f.save("fig6-spec-tree.svg", yb + 14)


def fig_ngram():
    """n-gram 草稿（Prompt Lookup）：在提示词里找到和末尾几个字一样的片段，把后面几个字照抄成草稿。示例自拟。"""
    f = Fig(W, "n-gram 草稿示例。用户贴了一段代码，要把 3.14 换成 math.pi。模型已经写出 def area(r，"
               "末尾两个字 area、(r 在提示词里出现过，就把提示词里紧跟着的 4 个字照抄成草稿。"
               "大模型一次验完：前两个对，第三个它要的是 math.pi，于是这一步前进 3 个字。示例是自拟的")
    y0 = f.header("n-gram 草稿：不用草稿模型，从提示词里照抄",
                  "用户：「把下面代码里的 3.14 换成 math.pi」。一格一个 token；示例自拟",
                  [(OR, "和末尾对上的片段"), (GR, "照抄来的草稿"), (BL, "大模型自己出的字"), (RD, "被拒")])
    LX, TX, TH, SZ = 24, 330, 44, 17
    def toks(y, items):
        x = TX
        out = []
        for txt, fill, stroke in items:
            w = wpx(txt, SZ) + 28
            f.box(x, y, w, TH, fill, stroke, 5, 2 if stroke in (OR, RD) else 1.2)
            f.t(x + w / 2, y + 29, txt, INK if stroke == LINE else stroke, True, SZ, anchor="middle", mono=True)
            out.append((x, w))
            x += w + 8
        return out
    W0, P = ("#fff", LINE), "prompt"
    prompt = ["def", " area", "(r", "):", " return", " 3.14", " *", " r", " *", " r"]
    PH = 380
    py = f.panel(0, y0, W, PH, "一次前向，前进 3 个字", GR)
    rows = []
    y = py + 24
    f.t(LX, y + 29, "① 提示词里", INK, True, 17)
    items = []
    for i, t in enumerate(prompt):
        if i in (1, 2):
            items.append((t, "#fef3e0", OR))
        elif 3 <= i <= 6:
            items.append((t, "#e6f4ea", GR))
        else:
            items.append((t, "#fff", LINE))
    pr = toks(y, items)
    y += 84
    f.t(LX, y + 29, "② 已经写出", INK, True, 17)
    toks(y, [("def", "#fff", LINE), (" area", "#fef3e0", OR), ("(r", "#fef3e0", OR)])
    f.t(TX + 260, y + 29, "← 末尾两个字在 ① 里找到了", OR, True, 16)
    y += 84
    f.t(LX, y + 29, "③ 草稿：照抄后面 4 个", INK, True, 17)
    toks(y, [(t, "#e6f4ea", GR) for t in prompt[3:7]])
    y += 84
    f.t(LX, y + 29, "④ 大模型一次验完", INK, True, 17)
    vr = toks(y, [("):", "#e6f4ea", GR), (" return", "#e6f4ea", GR), (" 3.14", "#fce8e6", RD), (" math.pi", "#e8f0fe", BL)])
    for k, mark in enumerate(("✓", "✓", "✗")):
        x, w = vr[k]
        f.t(x + w / 2, y + TH + 24, mark, GR if mark == "✓" else RD, True, 18, anchor="middle")
    x, w = vr[3]
    f.t(x + w + 18, y + 29, "第一个被拒的位置，大模型自己出字", BL, True, 16)
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "提示词里重复越多，抄中越多", [
        "改代码、摘要、按文档回答这类「答案大段来自输入」的活最赚；闲聊几乎抄不中，白验一遍。"])
    yb = f.src(yb + 10,
               "做法：prompt-lookup-decoding 仓库 README（用提示词里的字符串匹配代替草稿模型，输入依赖型任务 2–4 倍）；vLLM vllm/config/speculative.py 的 method=\"ngram\"、prompt_lookup_min／max（都不填时默认取 5）；SGLang --speculative-algorithm NGRAM。",
               "示例代码、匹配两个字、抄 4 个、验出 ✓✓✗ 都是自拟，只为说明机制；真实实现按 prompt_lookup_min～max 个 token 匹配。")
    f.save("fig6-spec-ngram.svg", yb + 14)


# ════════════════════════════════════════════════════════════════════════
def fig_three():
    """MTP、DFlash、DSpark 三种草稿怎么出字、怎么定验几个。字与概率都是示意。"""
    f = Fig(W, "三种草稿并排。第一行 MTP 和 EAGLE：草稿一个字一个字猜，猜五个要跑五趟，每趟都看得见前面猜了什么。"
               "第二行 DFlash：草稿跑一趟就并行出五个字，但每个位置各猜各的，第一位选了 of，第二位却押了 no problem 里的 problem，"
               "拼出 of problem，验到第二位就断了。第三行 DSpark：同样跑一趟出整块，再加一个很轻的顺序小头，"
               "前一位选了 of，就把下一位的 course 抬起来；另一个小头给每一位估一个存活概率，负载轻时验到第四位，负载重时只验前两位。"
               "字和概率都是示意")
    y0 = f.header("三种草稿：<tspan font-weight=\"700\">怎么出字，验几个</tspan>",
                  "一格一个草稿字。三种都贴着大模型的隐状态猜；差在出字方式和每步验多少",
                  [(PU, "草稿跑一趟"), (GR, "能被收下"), (RD, "被拒"), (GY2, "跟着作废")])
    TX, TW, TG, TH = 330, 104, 16, 40
    def tokx(i):
        return TX + i * (TW + TG)
    def tok(i, y, txt, col, fill):
        f.box(tokx(i), y, TW, TH, fill, col, 6, 1.6)
        f.t(tokx(i) + TW / 2, y + 26, txt, col if col != LINE else INK, True, 16, "middle", mono=True)
    NX = tokx(5) + 20                     # 右侧注释起点
    WORDS = ["of", "course", ",", "I", "can"]
    PH = 196

    # ① MTP / EAGLE：串行
    py = f.panel(0, y0, W, PH, "MTP、EAGLE：一个字一个字猜", PU)
    f.t(24, py + 50, "草稿", INK, True, 17)
    f.t(24, py + 74, "跑 5 趟", PU, True, 15)
    by, ty = py + 26, py + 106
    for i in range(5):
        f.box(tokx(i) + 12, by, TW - 24, 34, "#f3e8fd", PU, 5)
        f.t(tokx(i) + TW / 2, by + 22, "跑一趟", PU, True, 13, "middle")
        f.line(tokx(i) + TW / 2, by + 34, tokx(i) + TW / 2, ty - 2, PU, 1.4)
        tok(i, ty, WORDS[i], GR, "#e6f4ea")
        if i < 4:
            f.line(tokx(i) + TW - 4, ty + 6, tokx(i + 1) + 14, by + 34, PU, 1.2, dash="4 3")
    f.t(NX, py + 52, "每趟都看得见前面猜了啥，", INK, size=15)
    f.t(NX, py + 74, "所以越往后也不太掉；", INK, size=15)
    f.t(NX, py + 104, "代价：猜几个就跑几趟。", PU, True, 15)
    f.t(NX, py + 126, "MTP 是训进主模型的那一层。", GY, size=13.5)

    # ② DFlash：一趟一整块
    py = f.panel(0, py + PH + 14, W, PH, "DFlash：一趟出一整块", OR)
    f.t(24, py + 50, "草稿", INK, True, 17)
    f.t(24, py + 74, "跑 1 趟", OR, True, 15)
    bw = tokx(4) + TW - TX
    f.box(TX, by := py + 26, bw, 34, "#f3e8fd", PU, 5)
    f.t(TX + bw / 2, by + 22, "跑一趟，5 个位置并行出（大模型隐状态注入草稿每一层）", PU, True, 13, "middle")
    ty = py + 106
    for i in range(5):
        f.line(tokx(i) + TW / 2, by + 34, tokx(i) + TW / 2, ty - 2, PU, 1.4)
    tok(0, ty, "of", GR, "#e6f4ea")
    tok(1, ty, "problem", RD, "#fce8e6")
    for i, w in enumerate(["is", "fine", "."], 2):
        tok(i, ty, w, GY2, "#f1f3f4")
    f.t(tokx(1) + TW / 2, ty + TH + 22, "✗ 拼成了 of problem", RD, True, 13.5, "middle")
    f.t(NX, py + 52, "各位各猜各的，看不到", INK, size=15)
    f.t(NX, py + 74, "同块前一位选了哪个字：", INK, size=15)
    f.t(NX, py + 104, "两种说法拼了串。", OR, True, 15)
    f.t(NX, py + 126, "越往块尾，接受率掉得越快。", GY, size=13.5)

    # ③ DSpark：一整块 ＋ 顺序小头 ＋ 按把握验
    PH3 = 280
    py = f.panel(0, py + PH + 14, W, PH3, "DSpark：DFlash 当骨干，补两刀", GR)
    f.t(24, py + 50, "第一刀", INK, True, 17)
    f.t(24, py + 74, "顺序小头补依赖", GR, True, 15)
    f.t(24, py + 168, "第二刀", INK, True, 17)
    f.t(24, py + 192, "按把握和负载定验几个", GR, True, 15)
    f.box(TX, by := py + 22, bw, 30, "#f3e8fd", PU, 5)
    f.t(TX + bw / 2, by + 20, "同样跑一趟出整块", PU, True, 13, "middle")
    hy, ty = py + 64, py + 104
    for i in range(5):
        f.box(tokx(i) + 22, hy, TW - 44, 24, "#e6f4ea", GR, 5)
        f.t(tokx(i) + TW / 2, hy + 17, "小头", GR, True, 12, "middle")
        f.line(tokx(i) + TW / 2, hy + 24, tokx(i) + TW / 2, ty - 2, GR, 1.3)
        if i < 4:
            f.line(tokx(i) + TW - 22, hy + 12, tokx(i + 1) + 22, hy + 12, GR, 1.6)
    SURV = [95, 86, 60, 24, 5]                  # 示意：前缀存活概率 %
    assert SURV == sorted(SURV, reverse=True)
    for i in range(5):
        tok(i, ty, WORDS[i], GR, "#e6f4ea")
        bh = 40 * SURV[i] / 100
        f.box(tokx(i) + 14, py + 196 - bh, TW - 28, bh, "#a8dab5", GR, 3)
        f.t(tokx(i) + TW / 2, py + 216, "活到这 %d%%" % SURV[i], INK, True, 13, "middle")
    for cut, lab, col in ((2, "负载重：只验前 2 个", RD), (4, "负载轻：验到第 4 个", BL)):
        x = tokx(cut) - TG / 2
        f.line(x, ty - 6, x, py + 226, col, 2, dash="6 4", arrow=False)
        f.t(x, py + 246, lab, col, True, 13.5, "middle")
    f.t(NX, py + 50, "前一位选了 of，", INK, size=15)
    f.t(NX, py + 72, "就把下一位的 course 抬起来。", GR, True, 15)
    f.t(NX, py + 160, "绿柱：草稿活到这一位的概率，", INK, size=15)
    f.t(NX, py + 182, "越往后越矮。截在哪，", INK, size=15)
    f.t(NX, py + 204, "看这一刻机器忙不忙。", GR, True, 15)
    f._pan = None
    yb = f.band(py + PH3 + 16, "ok", "DSpark ＝ DFlash 的出字速度 ＋ 补回块内依赖 ＋ 按把握验", [
        "从上往下：第二行换来速度，第三行把第一行的前后照应补回来，再把验几个交给调度。",
    ])
    yb = f.src(yb + 10,
               "⚠️ 示意：例句、各行猜 5 个、存活概率 95／86／60／24／5% 和两条截断位置都是举例；of course／no problem 的例子出自 DSpark 论文 §3。",
               "DFlash：arXiv 2602.06036；KV 注入与 EAGLE-3 对比见 LMSYS 博客 2026-06-15。DSpark：arXiv 2607.05147（Markov 顺序头、置信度头、前缀存活概率、全批贪心调度；MTP-3/5 高并发拉低吞吐）；vLLM dspark.py 从 mtp.{0,1,2}.* 读权重。")
    f.save("fig6-spec-three.svg", yb + 14)


# ════════════════════════════════════════════════════════════════════════
# fig6-spec-eagle：EAGLE／EAGLE-3／MTP 草稿的结构。数字都来自论文正文（见出处行）。
EAGLE_70B_TRAIN_B = 0.99                     # EAGLE 论文 §4：70B 对应可训参数 0.99B
MIMO_RAW = [70, 10, 2]                       # FastMTP Fig.4(a)／贡献段：原样递归连猜第 1/2/3 位接受率 %
MIMO_FT = [81, 56, 36]                       # FastMTP 贡献段：按连猜微调后
assert all(a < b for a, b in zip(MIMO_RAW, MIMO_FT))


def fig_eagle():
    """EAGLE 借头借尾只训中间；EAGLE-3 的两处改动；MTP 同骨架、单层连猜要练过。"""
    f = Fig(W, "EAGLE、EAGLE-3、MTP 三种草稿的结构。第一块 EAGLE：大模型冻住，嵌入层和 LM head 借大模型的，"
               "只训一个 FC 压缩层加一层 decoder；输入是大模型出字前那层的向量，加上下一个已抽中的字的嵌入，"
               "拼成两倍宽，压回一倍，猜出下一位的向量，再借 LM head 出字，猜出的向量和字喂回去接着猜。"
               "第二块 EAGLE-3：浅、中、深三层向量拼成三倍宽再压回；猜第二个字起拿不到大模型的向量，用草稿自己上一步的输出顶替。"
               "第三块 MTP：骨架和 EAGLE 几乎一样，差在厂家预训练时一起训；单层 MTP 原样连猜，"
               "MiMo-7B 上第 1、2、3 个草稿字接受率约 70%、10%、2%，按连猜微调后是 81%、56%、36%")
    y0 = f.header("EAGLE 和 MTP 长什么样　——　<tspan font-weight=\"700\">头尾借大模型，只训中间一小块</tspan>",
                  "方框里橙色是要训的，蓝色是直接借大模型、不训的",
                  [(OR, "要训"), (BL, "借大模型，冻住不训"), (GY, "数据怎么拼")])
    TRAIN, BORROW = ("#fef3e0", OR), ("#e8f0fe", BL)

    def blk(x, y, w, h, main, sub, kind, size=15):
        fill, col = kind
        f.box(x, y, w, h, fill, col, 7, 1.6)
        f.t(x + w / 2, y + (h / 2 + 6 if not sub else h / 2 - 3), main, col, True, size, "middle", w=w - 10)
        if sub:
            f.t(x + w / 2, y + h / 2 + 17, sub, GY, False, 14, "middle", w=w - 10)
        return x + w

    # ① EAGLE
    PH = 268
    py = f.panel(0, y0, W, PH, "EAGLE（2024-01）：大模型冻住，草稿是外挂的一小块", OR)
    ya, yb_, BH = py + 26, py + 104, 56
    blk(24, ya, 250, BH, "大模型出字前那层的向量", "大模型本来就算出来了", BORROW, 14.5)
    blk(24, yb_, 120, BH, "抽中的字", "已定下的下一个", ("#fff", INK), 14.5)
    f.line(144, yb_ + BH / 2, 152, yb_ + BH / 2, GY2, 1.4)
    blk(154, yb_, 120, BH, "嵌入层", "借", BORROW, 14.5)
    cx = 318
    blk(cx, ya + 38, 130, 84, "拼起来", "2 倍宽", ("#f1f3f4", GY))
    f.elbow(274, ya + BH / 2, cx, ya + 62, GY2)
    f.elbow(274, yb_ + BH / 2, cx, ya + 98, GY2)
    x = blk(492, ya + 38, 150, 84, "FC", "压回 1 倍", TRAIN)
    f.line(cx + 130, ya + 80, 490, ya + 80, GY2, 1.4)
    f.line(x, ya + 80, 684, ya + 80, GY2, 1.4)
    x = blk(686, ya + 38, 160, 84, "一层 decoder", "猜下一位的向量", TRAIN)
    f.line(x, ya + 80, 888, ya + 80, GY2, 1.4)
    x = blk(890, ya + 38, 150, 84, "LM head", "借：向量变成字", BORROW)
    f.line(x, ya + 80, 1082, ya + 80, GY2, 1.4)
    f.box(1084, ya + 56, 96, 48, "#e6f4ea", GR, 6, 1.6)
    f.t(1132, ya + 87, "草稿字", GR, True, 15, "middle")
    f.path("M 1132 %d L 1132 %d L 210 %d L 210 %d" % (ya + 104, py + 196, py + 196, yb_ + BH + 2), OR, 1.6, dash="6 4")
    f.t(680, py + 218, "猜出的向量 ＋ 刚抽中的字，再喂回去猜下一个：猜几个跑几趟", OR, True, 14.5, "middle")
    f.t(1200, ya + 54, "为什么要喂抽中的字：", INK, True, 13.5)
    f.t(1200, ya + 76, "光看向量，不知道", INK, size=13.5)
    f.t(1200, ya + 98, "上一步按概率抽中了谁", INK, size=13.5)
    f.t(1200, ya + 134, "70B 配的这块：", GY, size=13.5)
    f.t(1200, ya + 156, "可训 %.2fB 参数，" % EAGLE_70B_TRAIN_B, OR, True, 13.5)
    f.t(1200, ya + 178, "A100 服务器 1–2 天", GY, size=13.5)

    # ② EAGLE-3
    PH2 = 246
    py = f.panel(0, py + PH + 14, W, PH2, "EAGLE-3（2025-03）：改两处", OR)
    f.t(24, py + 30, "改动一：三层拼起来，直接猜字", INK, True, 15)
    for i, lab in enumerate(("浅层向量", "中层向量", "深层向量")):
        blk(24, py + 44 + i * 50, 130, 40, lab, None, BORROW, 14)
        f.elbow(154, py + 64 + i * 50, 196, py + 114, GY2)
    blk(198, py + 84, 110, 60, "拼起来", "3 倍宽", ("#f1f3f4", GY), 14.5)
    f.line(308, py + 114, 334, py + 114, GY2, 1.4)
    blk(336, py + 84, 120, 60, "FC", "压回 1 倍", TRAIN, 14.5)
    f.t(24, py + 210, "不再猜向量，decoder 出来直接借 LM head 出字", GY, size=13.5)
    X2 = 560
    f.line(X2 - 30, py + 14, X2 - 30, py + PH2 - 46, LINE, 1, arrow=False)
    f.t(X2, py + 30, "改动二：第 2 个字起，拿草稿自己的输出顶替", INK, True, 15)
    def tokrow(y, label, items, out):
        f.t(X2, y + 27, label, INK, True, 14)
        x = X2 + 92
        for txt, kind in items:
            fill, col = kind
            w = wpx(txt, 14) + 26
            f.box(x, y, w, 40, fill, col, 6, 1.4)
            f.t(x + w / 2, y + 26, txt, col, True, 14, "middle")
            x += w + 8
        f.line(x + 4, y + 20, x + 40, y + 20, GY2, 1.4)
        f.box(x + 44, y, 70, 40, "#e6f4ea", GR, 6, 1.4)
        f.t(x + 79, y + 26, out, GR, True, 14, "middle")
        return x
    G, A, E = BORROW, TRAIN, ("#fff", INK)
    tokrow(py + 50, "第 1 步", [("大模型给的 how", G), ("大模型给的 can", G), ("＋ 字 I", E)], "do")
    xr = tokrow(py + 112, "第 2 步", [("大模型给的 how", G), ("大模型给的 can", G), ("草稿自己的 I", A), ("＋ 字 do", E)], "it")
    f.t(X2, py + 182, "I 大模型还没验，拿不到它的向量，只能用草稿上一步的输出顶上。", INK, size=13.5)
    f.t(X2, py + 206, "所以训练时就让草稿把自己的输出喂回来练：training-time test。", OR, True, 13.5)

    # ③ MTP
    PH3 = 300
    py = f.panel(0, py + PH2 + 14, W, PH3, "MTP（DeepSeek-V3 起）：骨架几乎一样，差在谁训", PU)
    f.t(24, py + 30, "DeepSeek-V3 的 MTP 模块", INK, True, 15)
    xs = 24
    for main, sub, kind, w in (("主干向量", "＋ 下一个字的嵌入", BORROW, 150), ("归一化、投影", None, TRAIN, 130),
                               ("transformer block", None, TRAIN, 170), ("输出头", "与主模型共用", BORROW, 120)):
        x = blk(xs, py + 46, w, 56, main, sub, kind, 14)
        if xs + w < 600:
            f.line(x, py + 74, x + 18, py + 74, GY2, 1.4)
        xs = x + 20
    rows = [("", "什么时候训", "大模型动不动"),
            ("EAGLE", "模型发布以后，谁都能外挂上去训", "冻住"),
            ("MTP", "厂家预训练时一起训，随模型发布", "一起训")]
    for i, (a, b, c) in enumerate(rows):
        yy = py + 140 + i * 34
        col = INK if i else GY
        f.t(24, yy, a, OR if a == "EAGLE" else (PU if a == "MTP" else GY), True, 14)
        f.t(100, yy, b, col, i == 0, 14)
        f.t(420, yy, c, col, i == 0, 14)
        if i == 0:
            f.line(24, yy + 10, 640, yy + 10, LINE, 1, arrow=False)
    f.t(24, py + 262, "带 MTP 的还有 Qwen3-Next、GLM-4.5、小米 MiMo；Step 3.5 Flash 挂 3 个模块（MTP-3）", GY, size=13.5)

    X3 = 720
    f.line(X3 - 30, py + 14, X3 - 30, py + PH3 - 46, LINE, 1, arrow=False)
    f.t(X3, py + 30, "一层 MTP 连猜：输出喂回自己再跑一趟", INK, True, 15)
    f.t(X3, py + 52, "MiMo-7B，第 1／2／3 个草稿字的接受率", GY, size=13)
    BASE, HMAX, BW_ = py + 236, 150, 64
    for k in range(3):
        gx = X3 + 40 + k * 210
        for j, (v, fill, col) in enumerate(((MIMO_RAW[k], "#f1f3f4", GY2), (MIMO_FT[k], "#a8dab5", GR))):
            h = HMAX * v / 100
            bx = gx + j * (BW_ + 10)
            f.box(bx, BASE - h, BW_, max(h, 1.5), fill, col, 3, 1.2)
            f.t(bx + BW_ / 2, BASE - h - 8, "%d%%" % v, col, True, 14, "middle")
        f.t(gx + BW_ + 5, BASE + 24, "第 %d 个" % (k + 1), INK, True, 14, "middle")
    f.line(X3 + 20, BASE, X3 + 640, BASE, GY2, 1, arrow=False)
    f.box(X3 + 440, py + 66, 16, 14, "#f1f3f4", GY2, 2)
    f.t(X3 + 462, py + 78, "原样连猜（只练过往前一步）", GY, size=13)
    f.box(X3 + 440, py + 90, 16, 14, "#a8dab5", GR, 2)
    f.t(X3 + 462, py + 102, "按连猜微调后（FastMTP）", GR, True, 13)
    f._pan = None
    yb = f.band(py + PH3 + 16, "ok", "长相差不多，差在谁训、练没练过连猜", [
        "EAGLE 不是免训练，只是训得少；MTP 能多猜几步，但草稿得按连猜练过，改个参数不够。"])
    yb = f.src(yb + 10,
               "EAGLE：arXiv 2401.15077 v3 §3（Figure 6 说明：嵌入层、LM head 用目标模型参数不训；FC＋一层 decoder；特征与提前一位的 token 嵌入拼 2 倍宽）、§4 Training（70B 可训 0.99B，A100 40G 服务器 1–2 天）。"
               "EAGLE-3：arXiv 2503.01840 v3 §3.1–3.2（低／中／高特征拼 3k 经 FC 压回 k；第 2 步起用草稿输出 a 顶替；training-time test）；how／can／I／do／it 是论文 Figure 5 的例子。",
               "MTP：DeepSeek-V3 技术报告 MTP 一节；各家采用见 sebastianraschka.com 架构图鉴 MTP 页、FastMTP 引言。接受率：FastMTP arXiv 2509.18362（MiMo-7B 单层 MTP，冻结主模型、自蒸馏数据微调 MTP 头；原样第 2 位约 10%，第 3 位约 2%）。")
    f.save("fig6-spec-eagle.svg", yb + 14)


# ════════════════════════════════════════════════════════════════════════
# fig6-spec-dspark：DSpark 论文的两组实测 —— 逐位接受率两端值、Qwen3-8B 接受长度（表 1）。
# 逐位条件接受率（Qwen3-4B，论文 §4.3.1 正文给出的两端值；前面几位全收下的前提下这一位被收下的比例）
DS_POS = [("闲聊", "EAGLE-3", 0.53, 0.74), ("闲聊", "DFlash", 0.72, 0.63), ("代码", "DFlash", 0.87, 0.78)]
DS_FIRST = [("EAGLE-3", 0.81), ("DFlash", 0.88), ("DSpark", 0.93)]     # 数学，第 1 位
# 表 1，Qwen3-8B，每类三个基准（含大模型补的那 1 个字）
DS_T1 = {"EAGLE-3": [(5.30, 4.77, 3.91), (3.96, 4.33, 4.17), (2.66, 2.54, 2.54)],
         "DFlash": [(5.33, 4.91, 4.07), (4.36, 4.64, 4.39), (3.11, 2.98, 2.81)],
         "DSpark": [(6.17, 5.78, 5.01), (5.16, 5.52, 5.17), (3.72, 3.58, 3.21)]}
def _mean(xs):
    return sum(xs) / len(xs)
_macro = {k: _mean([x for g in v for x in g]) for k, v in DS_T1.items()}
# 外部锚点：论文正文写 Qwen3-8B 上 DSpark 比 EAGLE-3 高 26.7%、比 DFlash 高 18.4%
assert round((_macro["DSpark"] / _macro["EAGLE-3"] - 1) * 100, 1) == 26.7
assert round((_macro["DSpark"] / _macro["DFlash"] - 1) * 100, 1) == 18.4
DS_COL = {"EAGLE-3": PU, "DFlash": OR, "DSpark": GR}


def fig_dspark():
    """DSpark 论文实测：左边逐位接受率两端值，右边 Qwen3-8B 三类任务的接受长度。"""
    avg = {k: [_mean(g) for g in v] for k, v in DS_T1.items()}
    f = Fig(W, "DSpark 论文的两组实测。左边是逐位接受率，指前面几位都被收下时这一位也被收下的比例，Qwen3-4B。"
               "闲聊上 EAGLE-3 第 1 位 0.53，往后升到 0.74；DFlash 第 1 位 0.72，往后掉到 0.63；代码上 DFlash 从 0.87 掉到 0.78。"
               "数学第 1 位：EAGLE-3 0.81，DFlash 0.88，DSpark 0.93。右边是 Qwen3-8B 每轮平均收下几个字："
               "数学 EAGLE-3 %.2f、DFlash %.2f、DSpark %.2f；代码 %.2f、%.2f、%.2f；闲聊 %.2f、%.2f、%.2f。"
               % tuple(avg[k][d] for d in range(3) for k in ("EAGLE-3", "DFlash", "DSpark")))
    y0 = f.header("DSpark 论文实测：<tspan font-weight=\"700\">开头靠深度，往后靠前后照应</tspan>",
                  "左：每一位单看准不准；右：一轮平均收下几个字",
                  [(PU, "EAGLE-3（一个个猜）"), (OR, "DFlash（一趟出整块）"), (GR, "DSpark")])
    PW, PH = 680, 400
    # ── 左：逐位接受率 ──
    py = f.panel(0, y0, PW, PH, "① 逐位接受率：第 1 位 → 块里往后", BL, tag="Qwen3-4B")
    AX0, AX1 = 210, 640
    def ax(v):
        return AX0 + (v - 0.5) / 0.5 * (AX1 - AX0)
    for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        f.line(ax(v), py + 30, ax(v), py + 300, LINE, 1, arrow=False)
        f.t(ax(v), py + 318, "%.1f" % v, GY, size=12, anchor="middle")
    for i, (dom, who, a0, a1) in enumerate(DS_POS):
        y = py + 62 + i * 62
        col = DS_COL[who]
        f.t(24, y + 5, dom, INK, True, 14)
        f.t(70, y + 5, who, col, True, 14)
        f.line(ax(a0), y, ax(a1), y, col, 3)
        f.t(ax(a0), y - 12, "%.2f" % a0, col, True, 13, "middle")
        f.t(ax(a1), y - 12, "%.2f" % a1, col, True, 13, "middle")
    y = py + 62 + 3 * 62 + 10
    f.t(24, y + 5, "数学", INK, True, 14)
    f.t(70, y + 5, "只看第 1 位", GY, size=13)
    for k, (who, v) in enumerate(DS_FIRST):
        f.box(ax(v) - 7, y - 7, 14, 14, DS_COL[who], DS_COL[who], 7)
        f.t(ax(v), y - 14 if k != 1 else y + 24, "%s %.2f" % (who, v), DS_COL[who], True, 12.5, "middle")
    f.t(24, py + 348, "箭头从第 1 位指向块里往后。一个个猜的越往后越稳，", INK, size=13.5)
    f.t(24, py + 370, "一趟出整块的越往后越掉；第 1 位反而是浅草稿最弱。", INK, size=13.5)
    # ── 右：接受长度 ──
    RX = PW + 20
    py2 = f.panel(RX, y0, W - RX, PH, "② 每轮平均收下几个字", GR, tag="Qwen3-8B，含大模型补的 1 个")
    BY0, BU = py2 + 300, 46            # 柱底、每个字的高度
    for v in range(0, 7):
        f.line(RX + 60, BY0 - v * BU, W - 30, BY0 - v * BU, LINE, 1, arrow=False)
        f.t(RX + 48, BY0 - v * BU + 4, str(v), GY, size=12, anchor="end")
    GW, BWD = 200, 46
    for d, dom in enumerate(("数学", "代码", "闲聊")):
        gx = RX + 90 + d * GW
        for k, who in enumerate(("EAGLE-3", "DFlash", "DSpark")):
            v = avg[who][d]
            x = gx + k * (BWD + 8)
            f.box(x, BY0 - v * BU, BWD, v * BU, DS_COL[who], DS_COL[who], 3)
            f.t(x + BWD / 2, BY0 - v * BU - 8, "%.2f" % v, DS_COL[who], True, 12.5, "middle")
        f.t(gx + 1.5 * BWD + 8, BY0 + 22, dom, INK, True, 14, "middle")
    f.t(RX + 24, py2 + 348, "同一个草稿，闲聊比数学少收约两个字：", INK, size=13.5)
    f.t(RX + 24, py2 + 370, "每次验几个，该看内容，也该看机器忙不忙。", GR, True, 13.5)
    f._pan = None
    yb = f.band(y0 + PH + 16, "ok", "开头靠草稿够深，往后靠前后照应；DSpark 两头都要", [
        "一趟出整块的草稿能叠得更深，第 1 位更准；DSpark 再用顺序小头把往后的下滑补回来。闲聊收得少，就是第二刀按把握定验几个的理由。",
    ])
    yb = f.src(yb + 10,
               "DSpark：arXiv 2607.05147。左：§4.3.1 正文给出的数（Figure 2 全曲线未逐点公开，图上只画正文写明的；右端是正文说的「块里更靠后」处，没写明是第几位）。右：表 1 Qwen3-8B，数学＝GSM8K／MATH500／AIME25、代码＝MBPP／HumanEval／LiveCodeBench、闲聊＝MT-Bench／Alpaca／Arena-Hard，每类取三个基准的平均（本课计算）。",
               "核对：九个基准的平均算出来 DSpark 比 EAGLE-3 高 26.7%、比 DFlash 高 18.4%，与论文正文一致。温度 1.0、块长 7、草稿 5 层（EAGLE-3 为 1 层）。")
    f.save("fig6-spec-dspark.svg", yb + 14)


fig_step()
fig_bytes()
fig_conc()
fig_temp()
fig_tree()
fig_ngram()
fig_three()
fig_eagle()
fig_dspark()
