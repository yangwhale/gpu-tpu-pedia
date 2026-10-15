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

fig6-spec-tree.svg —— 回答：这么多投机解码的名字，各自把草稿从哪里拿？
  · 年份按 arXiv 首版；每个方法都核过 arXiv 摘要页（2026-10-15）。Prompt Lookup 只有 GitHub 仓库。
  · 刻意没画各家加速倍数：都是原作者在自己设定下报的，放在一张图里会被横着比。
"""

from topic03_draw import Fig, BL, OR, GR, RD, PU, CY, GY, INK, GY2, LINE

W = 1400

# ══ fig6-spec-step：示意几何 ═══════════════════════════════════════════
K = 4                       # 每轮草稿猜 4 个（示意）
T_STEP, T_DRAFT, T_VERIFY = 10, 1, 11      # 普通一步 10 格；草稿每个 1 格；验证一步 11 格（示意）
ROUNDS = [3, 1]             # 第 1 轮接受 3 个，第 2 轮接受 1 个（示意）
ADV = [a + 1 for a in ROUNDS]               # 每轮前进 ＝ 接受个数 ＋ 1
assert ADV == [4, 2]
T_SPEC = len(ROUNDS) * (K * T_DRAFT + T_VERIFY)
assert T_SPEC == 30
N_PLAIN = T_SPEC // T_STEP                  # 同样 30 格时间，普通 decode 出几个字
assert N_PLAIN == 3 and sum(ADV) == 6

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
               "一次把四个字都验掉。第一轮前三个对、第四个错，大模型在第四个位置给出自己的字，于是一轮前进四个；"
               "第二轮只对了第一个，前进两个。同样三十格时间出了六个字")
    y0 = f.header("先猜后验　——　<tspan font-weight=\"700\">大模型读一遍权重，验掉一整串草稿</tspan>",
                  "横轴是时间（示意，不按比例）。每一格字：绿 ＝ 草稿猜对、被收下；红 ＝ 猜错、连同后面的全扔掉；"
                  "蓝 ＝ 大模型在第一个错的位置自己给的字",
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
        tok(tx + acc * 34, ry2 + 88, BL, "补")
        made += acc + 1
        f.t(xv + T_VERIFY * U - 8, ry2 + 140, "这一轮前进 %d 个（收下 %d ＋ 补 1）" % (acc + 1, acc),
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
               "约 171 亿参数，验证几个 token 都只读一遍；彩色是路由专家，每个 token 挑自己的 8 个，"
               "验证的 token 越多，碰到的专家并起来越多。验证 1 个读约 37.5 GB，验证 8 个约 163.8 GB，是 4.4 倍。"
               "右图：加速比等于平均接受长度除以一步变慢的倍数。三条线分别是一步不变慢、慢 1.45 倍、慢 2 倍。"
               "Kimi K3 实测点：一步从 11.1 毫秒变成约 16.1 毫秒，慢 1.45 倍，一步出 6.96 个字，每字间隔快 4.81 倍")
    y0 = f.header("多验证几个 token 贵不贵　——　<tspan font-weight=\"700\">稠密部分白送，MoE 专家不白送</tspan>",
                  "左：V3 验证一步要读的权重（本课推导，FP8）。右：加速比 ＝ 平均接受长度 ÷ 一步变慢的倍数",
                  [(GY2, "每个 token 都经过的部分"), (OR, "路由专家（各挑各的）"), (BL, "加速比曲线"), (RD, "K3 实测")])
    PH = 470
    LW = 640
    py = f.panel(0, y0, LW, PH, "V3 验证 n 个 token，一步读多少 GB", OR)
    PX, PY0, PW, PHH = 90, py + 50, 500, 320
    GMAX = 180
    f.box(PX, PY0, PW, PHH, "none", LINE, 4)
    for g in (0, 60, 120, 180):
        yy = PY0 + PHH - PHH * g / GMAX
        f.t(PX - 10, yy + 5, str(g), GY, size=12.5, anchor="end")
    f.t(PX - 60, PY0 - 14, "GB", GY, size=12.5)
    bw = PW / 8
    for n in range(1, 9):
        x = PX + (n - 1) * bw + 10
        h_o = PHH * (OTHER / 1e9) / GMAX
        h_t = PHH * step_gb(n) / GMAX
        f.box(x, PY0 + PHH - h_o, bw - 20, h_o, "#bdc1c6", GY2, 2)
        f.box(x, PY0 + PHH - h_t, bw - 20, h_t - h_o, OR, OR, 2)
        f.t(x + (bw - 20) / 2, PY0 + PHH + 22, str(n), GY, size=13, anchor="middle")
        if n in (1, 8):
            f.t(x + (bw - 20) / 2, PY0 + PHH - h_t - 8, "%.1f" % step_gb(n), INK, True, 13.5, "middle")
    f.t(PX + PW / 2, PY0 + PHH + 46, "一步验证的 token 数 n", GY, True, 13.5, anchor="middle")
    f.t(PX + 16, PY0 + 26, "验证 8 个 ＝ 验证 1 个的 %.1f 倍" % BYTES_X8, RD, True, 14)
    f.t(PX + 16, PY0 + 48, "灰色 %.1f GB 不随 n 变" % (OTHER / 1e9), GY, False, 13)

    # 右：AL vs 加速比
    RX = LW + 20
    py2 = f.panel(RX, y0, W - RX, PH, "加速比 ＝ 接受长度 ÷ 一步变慢的倍数", BL)
    QX, QY, QW, QH = RX + 80, py2 + 50, W - RX - 200, 320
    AMAX, SMAX = 8, 8
    f.box(QX, QY, QW, QH, "none", LINE, 4)

    def X(a):
        return QX + QW * a / AMAX

    def Y(s):
        return QY + QH - QH * min(s, SMAX) / SMAX
    for a in (0, 2, 4, 6, 8):
        f.t(X(a), QY + QH + 22, str(a), GY, size=12.5, anchor="middle")
    for s in (0, 2, 4, 6, 8):
        f.t(QX - 10, Y(s) + 5, "%d×" % s, GY, size=12.5, anchor="end")
    f.t(QX + QW / 2, QY + QH + 46, "平均每步出几个字（接受长度）", GY, True, 13.5, anchor="middle")
    for r, dash, lab in ((1.0, None, "一步不变慢"), (K3_RATIO, "7,4", "慢 %.2f 倍" % K3_RATIO), (2.0, "2,4", "慢 2 倍")):
        a_end = min(AMAX, SMAX * r)
        f.path("M%d,%d L%d,%d" % (X(1), Y(1 / r), X(a_end), Y(a_end / r)), BL, 2.5, dash=dash, arrow=False)
        f.t(X(a_end) + 8, Y(a_end / r) + 5, lab, BL, True, 13)
    cx, cy = X(K3_AL), Y(K3_TPOT_X)
    f.box(cx - 6, cy - 6, 12, 12, RD, RD, 6)
    f.t(cx - 12, cy - 34, "K3 实测：一步 %.1f → %.1f ms" % (K3_TPOT_NS, K3_STEP_SP), RD, True, 13.5, "end")
    f.t(cx - 12, cy - 14, "接受 %.2f 个 → 每字快 %.2f×" % (K3_AL, K3_TPOT_X), RD, True, 13.5, "end")
    yb = f.band(y0 + PH + 16, "ok", "小 batch 时收益几乎全来自接受长度", [
        "一步的时间主要花在读权重上；稠密部分验证几个都只读一遍，所以多验证几个几乎不加时间。",
        "MoE 要打折：每个 token 挑自己的专家，验证得越多碰到的专家越多。K3 一步慢了 %.2f 倍，靠一步出 %.2f 个字仍净赚。" % (K3_RATIO, K3_AL),
    ])
    yb = f.src(yb + 10,
               "V3 结构取自 HF deepseek-ai/DeepSeek-V3 config.json：hidden 7,168、moe_intermediate 2,048、61 层中前 3 层稠密、256 个路由专家挑 8 个。"
               "每个专家 3 × 7,168 × 2,048 ≈ 4,404 万参数。",
               "⚠️ 本课推导：其余部分 ＝ 671B − 256 × 58 × 专家大小 ≈ %.1fB；由此推出每 token 激活 ≈ %.1fB，与技术报告的 37B 差 2%% 以内（外部锚点）。"
               % (OTHER / 1e9, ACTIVE_DERIVED / 1e9),
               "⚠️ 本课推导：n 个 token 独立、均匀地挑专家时，一层平均碰到 256 × (1 − (1 − 8/256)^n) 个；实际路由不均匀、相邻 token 常挑同一批专家，所以彩色那段是偏高的估计。",
               "K3：gpu/inference/a4x-max/kimi-k3/SGLANG-K3-RUNBOOK.md:1089（GB300 TP8 跨两节点，ISL 4096／OSL 1024，temperature 0，bs=1，TPOT 11.1 → 2.31 ms）、:979（accept len 6.96，block 7）。",
               "⚠️ 本课推导：开投机后一步耗时 ＝ 2.31 ms × 6.96 ≈ 16.1 ms；每字间隔快 11.1 ÷ 2.31 ≈ 4.81 倍。runbook 报的端到端 4.22× 含首字时间，所以比 4.81 小。",
               "注：K3 是 8 张卡跑 bs=1，一步里除了读权重还有通信和 kernel 之间的空转，所以「慢 1.45 倍」不能全归给多读的专家。")
    f.save("fig6-spec-bytes.svg", yb + 14)


# ════════════════════════════════════════════════════════════════════════
def fig_conc():
    f = Fig(W, "上图：Kimi K3 在 GB300 上开 DSpark 投机解码的加速比随并发下降：并发 1 是 4.22 倍，8 是 2.86 倍，"
               "32 是 1.89 倍，64 是 1.74 倍；上下文拉到 32K、并发 1 只剩 1.32 倍。"
               "左下：B200 上 DeepSeek-V4-Flash 用 vLLM、EP 加 DP 八路时开 MTP，所有并发档吞吐都下降，从负 3% 到负 37%。"
               "右下：GB300 上 V4-Pro 把 MTP 从一步加到两步，decode 峰值掉 13%，端到端反而涨 28%")
    y0 = f.header("投机解码的收益会缩　——　<tspan font-weight=\"700\">batch 一大，多验证的 token 就不再白送</tspan>",
                  "柱高 ＝ 开投机后的吞吐 ÷ 不开（虚线 ＝ 1，不赚不赔）。三块面板是三套不同的硬件和模型，不要横着比",
                  [(GR, "K3 4K 上下文"), (PU, "K3 32K 上下文"), (RD, "比不开还慢")])
    PH = 330
    py = f.panel(0, y0, W, PH, "Kimi K3 · GB300 · SGLang TP8 · DSpark（输出 tok/s 之比）", GR)
    BX0, BY0, BH = 160, py + 40, 220
    SMAX = 4.5

    def Yb(v):
        return BY0 + BH - BH * v / SMAX
    f.path("M%d,%d L%d,%d" % (BX0 - 20, Yb(1), W - 120, Yb(1)), GY2, 1.4, dash="4,4", arrow=False)
    f.t(BX0 - 30, Yb(1) + 5, "1×", GY, size=13, anchor="end")
    bw, gap = 120, 70
    for i, ((c, ns, sp), x_) in enumerate(zip(K3_CONC, K3_X)):
        x = BX0 + i * (bw + gap)
        f.box(x, Yb(x_), bw, Yb(0) - Yb(x_), GR, GR, 4)
        f.t(x + bw / 2, Yb(x_) - 8, "%.2f×" % x_, INK, True, 16, "middle")
        f.t(x + bw / 2, BY0 + BH + 22, "并发 %d" % c, GY, True, 13.5, "middle")
    x = BX0 + 4 * (bw + gap) + 40
    v32 = K3_32K[1] / K3_32K[0]
    f.box(x, Yb(v32), bw, Yb(0) - Yb(v32), PU, PU, 4)
    f.t(x + bw / 2, Yb(v32) - 8, "%.2f×" % v32, INK, True, 16, "middle")
    f.t(x + bw / 2, BY0 + BH + 22, "32K · 并发 1", PU, True, 13.5, "middle")
    f.t(x + bw + 20, Yb(3.6), "同样并发 1，", PU, True, 14)
    f.t(x + bw + 20, Yb(3.6) + 20, "上下文 4K → 32K：", PU, True, 14)
    f.t(x + bw + 20, Yb(3.6) + 40, "4.22× → 1.32×", PU, True, 14)

    # 左下：B200 vLLM EP+DP8 MTP
    py2 = y0 + PH + 14
    PH2 = 300
    LW = 860
    pb = f.panel(0, py2, LW, PH2, "V4-Flash · B200 · vLLM EP+DP8 · MTP 2 个草稿（吞吐变化）", RD)
    ZX, ZY, ZH = 70, pb + 40, 190
    PMIN = -40

    def Yp(p):
        return ZY + ZH * (p / PMIN)
    f.path("M%d,%d L%d,%d" % (ZX - 10, ZY, LW - 20, ZY), GY2, 1.4, arrow=False)
    f.t(ZX - 16, ZY + 5, "0%", GY, size=12.5, anchor="end")
    f.t(ZX - 16, Yp(-37) + 5, "-37%", GY, size=12.5, anchor="end")
    cw = (LW - ZX - 30) / len(B200)
    for i, ((c, m, b), p) in enumerate(zip(B200, B200_PCT)):
        x = ZX + i * cw + 8
        f.box(x, ZY, cw - 16, Yp(p) - ZY, RD, RD, 3)
        f.t(x + (cw - 16) / 2, Yp(p) + 18, "%d%%" % p, INK, True, 13, "middle")
        f.t(x + (cw - 16) / 2, ZY - 8, str(c), GY, size=12.5, anchor="middle")
    f.t(ZX, pb + 252, "柱上方的数是并发；全部 10 档都比不开 MTP 慢，−3% 到 −37%", RD, True, 13.5)

    # 右下：V4-Pro steps
    RX = LW + 14
    pr = f.panel(RX, py2, W - RX, PH2, "V4-Pro · GB300 · SGLang · MTP 1 步 → 2 步", OR)
    gx = RX + 60
    for j, (lab, (a, b)) in enumerate((("decode 峰值", V4_PEAK), ("端到端", V4_E2E))):
        yy = pr + 50 + j * 110
        p = round((b / a - 1) * 100)
        f.t(gx - 30, yy + 4, lab, INK, True, 14.5)
        sc = 300 / 12000
        f.box(gx - 30, yy + 16, a * sc, 22, "#e8eaed", GY2, 3)
        f.t(gx - 30 + a * sc + 8, yy + 33, "1 步 {:,}".format(a), GY, size=12.5)
        col = RD if p < 0 else GR
        f.box(gx - 30, yy + 44, b * sc, 22, col, col, 3)
        f.t(gx - 30 + b * sc + 8, yy + 61, "2 步 {:,}（%+d%%）".format(b) % p, col, True, 13)
    f.t(RX + 20, pr + 262, "单位 tok/s/GPU；两个指标方向相反", GY, size=13)
    yb = f.band(py2 + PH2 + 16, "ok", "batch 一大，一步就被算力卡住", [
        "小 batch 时多验证的 token 搭的是读权重的便车；batch 大了，每多验证一个 token 都要实打实多算一份。",
        "EP+DP 本来就是为大 batch 配的，算力已经吃满，再加草稿就是纯开销；长上下文时注意力也在抢同一段时间。",
    ])
    yb = f.src(yb + 10,
               "K3：gpu/inference/a4x-max/kimi-k3/SGLANG-K3-RUNBOOK.md:1089-1092（GB300，TP8 跨两节点，ISL 4096／OSL 1024，random，temperature 0，开环，只报热轮；输出 tok/s 总量）、:1057-1058（32K，只测了并发 1）。",
               "B200：gpu/inference/a4/deepseek-v4/README.vllm.md:266-275（DeepSeek-V4-Flash，8× B200，vLLM v0.20.0，EP+DP=8，MTP num_speculative_tokens=2；接受率约 54%）。",
               "V4-Pro：gpu/inference/a4x-max/deepseek-v4/SGLANG-V4PRO-RUNBOOK.md:900-911（steps 1／draft 2 → steps 2／draft 3；接受长度 1.88 → 2.36，单步 +41%）。"
               "峰值是 batch 撑满时的 decode 吞吐；端到端受 prefill 喂不满拖累，batch 多数时间不在顶点。",
               "注：K3 并发 64 时开投机的 TPOT 仍只有不开的 40%，这一档吞吐和延迟同时变好；B200 那套是 EP+DP 高吞吐配置，结论不能搬到 TP-only 低延迟配置。")
    f.save("fig6-spec-conc.svg", yb + 14)


# ════════════════════════════════════════════════════════════════════════
def fig_temp():
    f = Fig(W, "左图：GB300 上的 DeepSeek-V4-Pro 用 vLLM 开 DSpark 投机，草稿是贪心的（每次只挑最可能的字）。"
               "压测工具不发温度时，服务端按温度 1.0 随机采样，接受率只有 1.16%，吞吐 944 tok/s；"
               "发温度 0 时接受率 34%，吞吐 2,911 tok/s，是 3.1 倍。"
               "右图：Kimi K3 的草稿按概率抽样，官方就用温度 1.0 压测；接受长度随任务变，代码类约 4.73，创作类约 2.61")
    y0 = f.header("温度：草稿怎么抽，压测就得怎么发　——　<tspan font-weight=\"700\">对不上，接受率就塌</tspan>",
                  "同一套服务、同一份负载，只差请求里带不带 temperature=0",
                  [(RD, "温度 1.0（服务端默认）"), (GR, "温度 0"), (BL, "K3 官方接受长度")])
    PH = 330
    LW = 860
    py = f.panel(0, y0, LW, PH, "V4-Pro · GB300 · vLLM · 贪心草稿 · 4K/1K 并发 256", RD)
    rows = [("接受率", T1_ACC, T0_ACC, 40, "%.2f%%", "%.0f%%"), ("输出 tok/s", T1_TPS, T0_TPS, 3200, "{:,}", "{:,}")]
    for j, (lab, a, b, mx, fa, fb) in enumerate(rows):
        yy = py + 50 + j * 130
        f.t(24, yy + 34, lab, INK, True, 15)
        sc = 560 / mx
        f.box(150, yy + 8, max(a * sc, 3), 30, RD, RD, 3)
        f.t(150 + max(a * sc, 3) + 10, yy + 30, (fa.format(a) if "{" in fa else fa % a), RD, True, 14)
        f.box(150, yy + 48, b * sc, 30, GR, GR, 3)
        f.t(150 + b * sc + 10, yy + 70, (fb.format(b) if "{" in fb else fb % b), GR, True, 14)
    f.t(24, py + 290, "接受率差 %d 倍，吞吐差 %.1f 倍" % (round(T0_ACC / T1_ACC), T0_TPS / T1_TPS), INK, True, 15)

    RX = LW + 14
    pr = f.panel(RX, y0, W - RX, PH, "K3 · 概率草稿 · 官方温度 1.0", BL)
    for j, (lab, v) in enumerate((("代码等低熵任务", K3_AL_LOW), ("创作等高熵任务", K3_AL_HIGH))):
        yy = pr + 60 + j * 100
        f.t(RX + 24, yy, lab, INK, True, 14.5)
        f.box(RX + 24, yy + 14, v * 70, 30, BL, BL, 3)
        f.t(RX + 24 + v * 70 + 10, yy + 36, "约 %.2f 个／步" % v, BL, True, 14)
    f.t(RX + 24, pr + 270, "接受长度跟着负载走", GY, size=13.5)
    yb = f.band(y0 + PH + 16, "warn", "接受率掉了不报错，只是变慢", [
        "压测报告必须写明温度和草稿的抽样方式；跨框架比投机解码，先对齐两边的压测工具发不发 temperature。",
        "线上要把接受率和平均接受长度做成告警：它们往下走时 TPOT 才慢慢变差，等用户察觉已经晚了。",
    ])
    yb = f.src(yb + 10,
               "V4-Pro：gpu/inference/a4x-max/deepseek-v4/VLLM-V4PRO-RUNBOOK.md:398-410。generation_config 默认 temperature 1.0；DSpark 草稿 draft_sample_method 为 greedy，num_speculative_tokens=7。"
               "vllm bench serve 不发 temperature；sa-bench 硬编码 0.0。",
               "K3：gpu/inference/a4x-max/kimi-k3/VLLM-K3-RUNBOOK.md:226-237（官方压测 temperature 1.0、top-p 0.95；DSpark 用 probabilistic 草稿；接受长度为官方参考值，本仓库未复测）。",
               "告警：gpu/inference/a4x-max/kimi-k3/PD-BACKLOG.md:175。")
    f.save("fig6-spec-temp.svg", yb + 14)


# ════════════════════════════════════════════════════════════════════════
TREE = [
    # (泳道, 年, 名字, 一句话, arXiv / 出处)
    (0, 2022.9, "独立小模型", "另一个小模型打草稿", "2211.17192"),
    (0, 2023.4, "SpecInfer", "草稿排成树，一次验几条", "2305.09781"),
    (0, 2024.0, "Medusa", "大模型头上挂几个预测头", "2401.10774"),
    (0, 2024.08, "EAGLE 1／2", "读大模型的隐状态打草稿", "2401.15077"),
    (0, 2024.95, "MTP", "预测头直接训进模型", "2412.19437"),
    (0, 2025.2, "EAGLE-3", "融合多层特征", "2503.01840"),
    (0, 2026.5, "DSpark", "半自回归，按置信度定验几个", "2607.05147"),
    (1, 2023.8, "Prompt Lookup", "从 prompt 里抄", "GitHub"),
    (1, 2024.1, "Lookahead", "Jacobi 迭代凑 n-gram", "2402.02057"),
    (1, 2024.85, "SuffixDecoding", "后缀树查历史输出", "2411.04975"),
    (2, 2025.75, "DiffuSpec", "现成扩散模型当草稿", "2510.02358"),
    (2, 2026.1, "DFlash", "块扩散一次出一整块", "2602.06036"),
]
assert len(TREE) == 12


def fig_tree():
    f = Fig(W, "投机解码的谱系，横轴是年份。第一条泳道是有草稿模型的一支，草稿离大模型越来越近："
               "2022 年用独立的小模型，2023 年 SpecInfer 把草稿排成树，2024 年 Medusa 在大模型头上挂预测头，"
               "EAGLE 读大模型的隐状态，DeepSeek-V3 的 MTP 把预测头直接训进模型，2025 年 EAGLE-3，2026 年 DSpark。"
               "第二条泳道不要草稿模型：Prompt Lookup 从 prompt 里抄，Lookahead 用 Jacobi 迭代，SuffixDecoding 查历史输出。"
               "第三条是 2025 到 2026 年的扩散草稿：DiffuSpec、DFlash，一次并行出一整块草稿")
    y0 = f.header("草稿从哪来　——　<tspan font-weight=\"700\">越来越贴近大模型，或者干脆不要模型</tspan>",
                  "横轴 ＝ arXiv 首版年份。一个点一种方法，点下写它的草稿从哪来",
                  [(BL, "有草稿模型"), (OR, "不要草稿模型"), (PU, "扩散／块并行草稿")])
    LANES = [("有草稿模型", BL), ("不要草稿模型", OR), ("扩散草稿", PU)]
    LH = [250, 190, 190]
    X0, X1 = 200, W - 90
    Y0, Y1 = 2022.5, 2026.9

    def X(yr):
        return X0 + (X1 - X0) * (yr - Y0) / (Y1 - Y0)
    for yr in (2023, 2024, 2025, 2026):
        f.t(X(yr), y0 + 14, str(yr), GY, True, 14, "middle")
    y0 += 26
    tops = []
    yy = y0
    for (name, col), h in zip(LANES, LH):
        f.panel(0, yy, W, h, name, col)
        tops.append(yy)
        yy += h + 10
    # 年份尺
    for yr in (2023, 2024, 2025, 2026):
        f.path("M%d,%d L%d,%d" % (X(yr), y0 + 32, X(yr), yy - 12), "#e8eaed", 1, arrow=False)
    # 第一泳道两行错开，避免字撞
    alt = {}
    for lane, yr, nm, desc, ref in TREE:
        col = LANES[lane][1]
        base = tops[lane] + 30
        k = alt.get(lane, 0)
        alt[lane] = k + 1
        if lane == 0:
            cy = base + (60 if k % 2 == 0 else 150)
        else:
            cy = base + (40 if k % 2 == 0 else 105)
        f.box(X(yr) - 7, cy - 7, 14, 14, col, col, 7)
        f.t(X(yr) + 14, cy + 5, nm, col, True, 14.5)
        f.t(X(yr) + 14, cy + 25, desc, GY, size=12.5)
        f.t(X(yr) + 14, cy + 43, ref, GY2, size=11.5, mono=True)
    # 第一泳道一条「越来越近」的箭头
    yb = f.band(yy + 6, "ok", "草稿越贴近大模型，猜得越准", [
        "独立小模型要单独训练、单独部署；挂头、读隐状态、训进模型，一步步把草稿变成大模型自己的一部分。",
        "不要模型的一支靠「输出常常抄输入」：改代码、做摘要、agent 反复调用工具时最好用。",
    ])
    yb = f.src(yb + 10,
               "每个点的 arXiv 号写在名字下面，年份取 arXiv 首版（2026-10-15 逐个核过摘要页）。Prompt Lookup 只有 GitHub 仓库 apoorvumang/prompt-lookup-decoding（已集成进 vLLM 与 transformers）。",
               "DSpark：DeepSeek，arXiv 2607.05147，摘要说在吞吐相同的条件下单用户生成速度比线上 MTP-1 快 60–85%。DFlash：ICML 2026，摘要说无损加速超过 6 倍。",
               "注：各家报的加速倍数都是原作者在自己的模型和负载上测的，彼此不能横比，所以图上不画倍数。",
               "注：EAGLE 1 和 EAGLE-2 画成一个点（2401.15077、2406.16858）；P-EAGLE、PARD、LayerSkip 等没画。")
    f.save("fig6-spec-tree.svg", yb + 14)


fig_step()
fig_bytes()
fig_conc()
fig_temp()
fig_tree()
