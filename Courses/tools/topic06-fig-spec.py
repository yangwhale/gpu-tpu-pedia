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
    B200 V4-Flash vLLM gpu/inference/a4/deepseek-v4/README.vllm.md:266-275；
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
# B200 V4-Flash vLLM EP+DP8，MTP n=2 vs 无 MTP（README.vllm.md:266-275）
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
    f = Fig(W, "左图：DeepSeek-V3 一步验证 n 个 token 要读多少权重，FP8 每参数一字节。灰色是每个 token 都要经过的部分，"
               "约 171 亿参数，验证几个 token 都只读一遍；橙色是路由专家，每个 token 挑自己的 8 个，"
               "验证的 token 越多，碰到的专家并起来越多。验证 1 个读约 37.5 GB，验证 8 个约 163.8 GB，是 4.4 倍。"
               "右图：Kimi K3 实测的时间账。不开投机，搬一趟权重 11.1 毫秒，带回 1 个字；开投机，搬一趟变成约 16.1 毫秒，"
               "慢了 1.45 倍，但平均带回 6.96 个字。摊到每个字，从 11.1 毫秒降到 2.31 毫秒，快 4.81 倍")
    y0 = f.header("多验证几个 token 贵不贵　——　<tspan font-weight=\"700\">稠密部分白送，MoE 专家不白送</tspan>",
                  "左：V3 验证一步要读的权重（本课推导，FP8）。右：Kimi K3 实测，搬一趟变慢了，但一趟带回的字多得多",
                  [(GY2, "每个 token 都经过的部分"), (OR, "路由专家（各挑各的）"), (BL, "开投机后的一趟")])
    PH = 470
    LW = 640
    py = f.panel(0, y0, LW, PH, "V3 验证 n 个 token，一步读多少 GB", OR)
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
        f.box(x, PY0 + PHH - h_o, bw - 20, h_o, "#bdc1c6", GY2, 2)
        f.box(x, PY0 + PHH - h_t, bw - 20, h_t - h_o, OR, OR, 2)
        f.t(x + (bw - 20) / 2, PY0 + PHH + 24, str(n), GY, size=15, anchor="middle")
        if n in (1, 8):
            f.t(x + (bw - 20) / 2, PY0 + PHH - h_t - 8, "%.1f" % step_gb(n), INK, True, 15, "middle")
    f.t(PX + PW / 2, PY0 + PHH + 52, "一步验证的 token 数 n", GY, True, 15, anchor="middle")
    f.t(PX + 16, PY0 + 30, "验证 8 个 ＝ 验证 1 个的 %.1f 倍" % BYTES_X8, RD, True, 17)
    f.t(PX + 16, PY0 + 56, "灰色 %.1f GB 不随 n 变" % (OTHER / 1e9), GY, False, 15)

    # 右：K3 一趟的时间账
    RX = LW + 20
    RW = W - RX
    py2 = f.panel(RX, y0, RW, PH, "Kimi K3 实测：搬一趟，带回几个字", BL)
    TX = RX + 150                      # 时间零点
    MS = 30.0                          # 每毫秒多少像素
    assert TX + 17 * MS < W - 20

    def lane(yy, lab, ms, col, fill, n_tok, tok_lab):
        f.t(RX + 24, yy + 26, lab, INK, True, 17)
        f.box(TX, yy, ms * MS, 40, fill, col, 6)
        f.t(TX + 14, yy + 26, "搬一趟 %.1f ms" % ms, col, True, 16)
        for k in range(n_tok):
            f.box(TX + k * 36, yy + 56, 30, 30, BL, BL, 5)
        f.t(TX + n_tok * 36 + 10, yy + 78, tok_lab, INK, True, 16)

    ya = py2 + 40
    lane(ya, "不开投机", K3_TPOT_NS, GY2, "#e8eaed", 1, "带回 1 个字")
    yb_ = ya + 140
    lane(yb_, "开投机", K3_STEP_SP, BL, "#e8f0fe", 7, "平均带回 %.2f 个字" % K3_AL)
    # 多花的那一截：从 11.1 ms 到 16.1 ms
    x1, x2 = TX + K3_TPOT_NS * MS, TX + K3_STEP_SP * MS
    f.path("M%d,%d L%d,%d" % (x1, ya + 44, x1, yb_ + 44), GY2, 1.2, dash="4,4", arrow=False)
    f.path("M%d,%d L%d,%d L%d,%d L%d,%d" % (x1, yb_ - 8, x1, yb_ - 14, x2, yb_ - 14, x2, yb_ - 8),
           RD, 1.6, arrow=False)
    f.t((x1 + x2) / 2, yb_ - 22, "慢 %.2f 倍" % K3_RATIO, RD, True, 16, "middle")
    # 摊到每个字
    yc = yb_ + 150
    f.path("M%d,%d L%d,%d" % (RX + 24, yc - 30, W - 24, yc - 30), LINE, 1, arrow=False)
    f.t(RX + 24, yc, "摊到每个字", GY, True, 16)
    f.t(RX + 24, yc + 36, "%.1f ms → %.2f ms，快 %.2f 倍" % (K3_TPOT_NS, K3_TPOT_SP, K3_TPOT_X), BL, True, 22)
    f.t(RX + 24, yc + 66, "一趟慢一点不要紧，带回的字多得多", GY, size=15)
    yb = f.src(y0 + PH + 6,
               "V3 结构取自 HF deepseek-ai/DeepSeek-V3 config.json：hidden 7,168、moe_intermediate 2,048、61 层中前 3 层稠密、256 个路由专家挑 8 个。"
               "每个专家 3 × 7,168 × 2,048 ≈ 4,404 万参数。",
               "⚠️ 本课推导：其余部分 ＝ 671B − 256 × 58 × 专家大小 ≈ %.1fB；由此推出每 token 激活 ≈ %.1fB，与技术报告的 37B 差 2%% 以内（外部锚点）。"
               % (OTHER / 1e9, ACTIVE_DERIVED / 1e9),
               "⚠️ 本课推导：n 个 token 独立、均匀地挑专家时，一层平均碰到 256 × (1 − (1 − 8/256)^n) 个；实际路由不均匀、相邻 token 常挑同一批专家，所以橙色那段是偏高的估计。",
               "K3：gpu/inference/a4x-max/kimi-k3/SGLANG-K3-RUNBOOK.md:1089（GB300 TP8 跨两节点，ISL 4096／OSL 1024，temperature 0，bs=1，TPOT 11.1 → 2.31 ms）、:979（accept len 6.96，block 7）。",
               "⚠️ 本课推导：开投机后一趟耗时 ＝ 2.31 ms × 6.96 ≈ 16.1 ms；每字间隔快 11.1 ÷ 2.31 ≈ 4.81 倍。runbook 报的端到端 4.22× 含首字时间，所以比 4.81 小。"
               "右图第 7 个方块代表平均值里的 0.96 个，示意。",
               "注：K3 是 8 张卡跑 bs=1，一步里除了读权重还有通信和 kernel 之间的空转，所以「慢 1.45 倍」不能全归给多读的专家。")
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
               "正文里另两组数的出处 —— B200：gpu/inference/a4/deepseek-v4/README.vllm.md:266-275（DeepSeek-V4-Flash，8× B200，vLLM v0.20.0，EP+DP=8，MTP num_speculative_tokens=2；接受率约 54%；十档并发 −3% 到 −37%）。",
               "V4-Pro：gpu/inference/a4x-max/deepseek-v4/SGLANG-V4PRO-RUNBOOK.md:900-911（steps 1／draft 2 → steps 2／draft 3；接受长度 1.88 → 2.36，单步 +41%；decode 峰值 −13%、端到端 +28%）。"
               "B200 那套是 EP+DP 高吞吐配置，结论不能搬到 TP-only 低延迟配置。")
    f.save("fig6-spec-conc.svg", yb + 14)


# ════════════════════════════════════════════════════════════════════════
def fig_temp():
    f = Fig(W, "用对答案打比方。题目是下一个字填什么，最可能的答案是「好」，草稿是贪心的，每次都写最可能的「好」。"
               "上一行温度 1.0：大模型掷骰子决定标准答案，大多掷出别的字，和草稿对不上，被扔掉。"
               "GB300 上的 DeepSeek-V4-Pro 实测接受率只有 1.16%，吞吐 944 tok/s。"
               "下一行温度 0：骰子永远停在最可能的那个字，草稿只要猜的就是它就能对上。实测接受率 34%，吞吐 2,911 tok/s，是 3.1 倍")
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
    yy = f.band(yy + 4, "ok", "实测吞吐差 %.1f 倍" % (T0_TPS / T1_TPS), [
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
    (2024.08, "2024-01", "EAGLE", ["小模型读大模型的", "隐状态来猜"], "EAGLE 2401.15077、EAGLE-2 2406.16858"),
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
               "没画进图的（正文里提到）：SpecInfer 2305.09781 把草稿排成树；SuffixDecoding 2411.04975 从历史输出里抄；"
               "扩散草稿 DiffuSpec 2510.02358、DFlash 2602.06036；Lookahead 2402.02057。P-EAGLE、PARD、LayerSkip 等也没画。",
               "注：各家倍数见正文列表；都是原作者在自己的模型和负载上测的，口径不同，不横比。")
    f.save("fig6-spec-tree.svg", yb + 14)


fig_step()
fig_bytes()
fig_conc()
fig_temp()
fig_tree()
