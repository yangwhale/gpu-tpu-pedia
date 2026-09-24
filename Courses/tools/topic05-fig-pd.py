# -*- coding: utf-8 -*-
r"""专题五 · 第六节「第五刀：不切张量，切工作」的两张静态图。

出处：
  · prefill 吃算力、decode 吃带宽，放一起互相干扰：DistServe（arXiv 2401.09670）。
  · 我们自己的 TPU v7x PD 分离实测（wiki qwen3-coder-480b-pd-disagg-tpuv7x-20260425）：
    KV 三段传输按带宽估算约 100 ms（不是计时实测），8K prompt 的 KV 约 1 GB（FP8），占 1–2 s prefill 的 5–10%。
  · AFD 的 M2N ／ N2M 与乒乓 micro-batch：MegaScale-Infer（arXiv 2504.02263），每 GPU 吞吐最高 1.90 倍。
⛔ 时间线是示意（格子长度按 prefill 比 decode 一步长很多来画），不是实测时序；图上也这样标。
"""
from topic03_draw import Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE

W = 1400
# Qwen3-Coder-480B（config.json：62 层、8 个 KV 头、head_dim 128），8K prompt、FP8 每元素 1 字节
KV_8K = 2 * 62 * 8 * 128 * 8192
DCN = 100e9 / 8                          # 100 Gbps 以太网 ＝ 12.5 GB/s
T_DCN = KV_8K / DCN
assert abs(KV_8K / 1e9 - 1.04) < 0.01 and abs(T_DCN * 1e3 - 83) < 1


def fig_pd():
    f = Fig(W, "为什么要把 prefill 和 decode 分开。上面是放在同一批卡上：大家都在一步一步 decode，"
               "突然来了一个长 prompt，它的 prefill 要占好几步的时间，这几步里所有人的 decode 都停住了，出字速度掉下去。"
               "下面是分开：prefill 机器专门吞 prompt，算完把 KV cache 传给 decode 机器；decode 机器一直稳定地出字，"
               "代价是多了一趟 KV 传输。按我们 TPU v7x 那套的带宽估算，这一趟约 100 毫秒，占一次 prefill 的百分之五到十")
    y0 = f.header("第五刀：把两种活拆到两批机器上　——　<tspan font-weight=\"700\">decode 不再被 prefill 卡住</tspan>",
                  "示意时间线（不是实测时序）。每格是一步；绿色的 decode 一格一格往外出字，橙色是一次长 prompt 的 prefill",
                  [(GR, "decode：每步出一个字"), (OR, "prefill：一次吞下整个 prompt"), (BL, "KV cache 传输")])
    PH = 330
    py = f.panel(0, y0, W, PH, "同一批卡 vs 分开两批", OR)
    X0, CW = 230, 48
    N, AT, LN = 16, 5, 5                 # 跟 PD 动画同一套：16 步、第 5 步起 prefill 占 5 步
    # 同一批卡
    f.t(24, py + 62, "放在一起", INK, True, 16)
    x = X0
    seq = ["d"] * AT + ["P"] + ["d"] * (N - AT - LN)
    for s in seq:
        if s == "d":
            f.box(x, py + 40, CW - 4, 36, GR, GR, 3)
            x += CW
        else:
            f.box(x, py + 40, CW * LN - 4, 36, OR, OR, 3)
            f.t(x + CW * LN / 2, py + 64, "长 prompt 的 prefill", "#ffffff", True, 14, "middle")
            f.t(x + CW * LN / 2, py + 100, "这 %d 步所有人的 decode 都停住" % LN, RD, True, 13.5, "middle")
            x += CW * LN
    # 分开
    f.t(24, py + 182, "prefill 机器", INK, True, 16)
    f.t(24, py + 262, "decode 机器", INK, True, 16)
    f.box(X0 + AT * CW, py + 160, CW * LN - 4, 36, OR, OR, 3)
    f.t(X0 + (AT + LN / 2) * CW, py + 184, "prefill", "#ffffff", True, 14, "middle")
    f.path("M%d,%d L%d,%d" % (X0 + (AT + LN) * CW, py + 198, X0 + (AT + LN + 1) * CW, py + 238), BL, 2.2)
    f.t(X0 + (AT + LN + 1) * CW + 10, py + 222, "KV cache 传过去（按带宽估算约 100 ms）", BL, True, 13)
    x = X0
    for i in range(N):
        f.box(x, py + 240, CW - 4, 36, GR, GR, 3)
        x += CW
    f.t(X0 + N * CW + 12, py + 264, "一直稳定地出字", GR, True, 14)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "拆开换来的：出字不再被打断", [
        "放在一起时，一个长 prompt 的 prefill 就能让所有人停几步。<tspan font-weight=\"700\">拆开后 decode 那条线一格不少。</tspan>",
        "代价只是多一趟 KV 传输：KV 从 prefill 机器的显存走到 decode 机器的显存。",
    ])
    yb = f.src(yb + 10, "📌 DistServe arXiv 2401.09670：prefill 偏算力、decode 受显存带宽约束；拆开后多服务 7.4 倍请求或把 SLO 收紧 12.6 倍。",
                        "📌 本课程作者 TPU v7x 1P1D 部署（Qwen3-Coder-480B）：KV 三段传输按带宽估算约 100 ms，占 1–2 s prefill 的 5–10%%。"
                        "8K prompt 的 KV ≈ %.2f GB（FP8，按 config 现算），过 100 Gbps 网络约 %.0f ms" % (KV_8K / 1e9, T_DCN * 1e3) + "；长 prompt 宜 2P:1D，长输出宜 1P:2D。")
    f.save("fig5-pd.svg", yb + 14)


def fig_afd():
    f = Fig(W, "attention 和 FFN 也拆开。左边 M 台机器只算 attention，右边 N 台机器只放专家。"
               "每一层 attention 算完，把 token 发给专家那边，这一步叫 M 到 N；专家算完再送回来，叫 N 到 M。"
               "专家那边同时接好几台 attention 机器的 token，专家的 batch 就大了。"
               "为了把这来回两趟藏起来，把一批请求切成几个小批轮着跑，这一个在算 attention，另一个正好在算专家，图里画两个示意，论文里要三四个")
    y0 = f.header("再拆一层：attention 和专家也分开　——　<tspan font-weight=\"700\">专家那边一次接好几家的 token</tspan>",
                  "AFD（Attention-FFN 分离）。示意：M 台 attention 机器、N 台专家机器，两个小批交替跑",
                  [(BL, "attention 机器"), (OR, "专家机器"), (GR, "小批 A"), (PU, "小批 B")])
    PH = 330
    py = f.panel(0, y0, W, PH, "每一层都要跑一趟 M → N → M", BL)
    for i in range(3):
        f.box(60, py + 50 + i * 76, 200, 56, "none", BL, 8, sw=2)
        f.t(160, py + 84 + i * 76, "attention 机器 %d" % (i + 1), BL, True, 14, "middle")
    for i in range(2):
        f.box(560, py + 80 + i * 96, 200, 56, "none", OR, 8, sw=2)
        f.t(660, py + 114 + i * 96, "专家机器 %d" % (i + 1), OR, True, 14, "middle")
    for i in range(3):
        for k in range(2):
            f.line(262, py + 78 + i * 76, 558, py + 108 + k * 96, GY2, 1.2, arrow=False)
    f.t(410, py + 36, "M → N：token 发给专家", GY, True, 13, anchor="middle")
    f.t(410, py + 290, "N → M：算完送回来", GY, True, 13, anchor="middle")
    # 乒乓时间线
    TX, CW = 830, 54
    f.t(TX, py + 50, "小批轮着跑（示意画两个）", INK, True, 15)
    f.t(TX, py + 96, "attention", BL, True, 13)
    f.t(TX, py + 156, "专家", OR, True, 13)
    for t in range(8):
        a_col = GR if t % 2 == 0 else PU
        e_col = PU if t % 2 == 0 else GR
        f.box(TX + 80 + t * CW, py + 72, CW - 4, 36, a_col, a_col, 3)
        f.box(TX + 80 + t * CW, py + 132, CW - 4, 36, e_col if t > 0 else "none", e_col if t > 0 else LINE, 3)
    f.t(TX, py + 214, "attention 在算 A 的时候，专家在算 B；", GY, size=13)
    f.t(TX, py + 238, "示意画两个；真要藏住来回的通信，得三四个。", GY, size=13)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "拆得越细，每一边越能挑适合自己的机器和切法", [
        "专家那边同时接几台 attention 机器的 token，<tspan font-weight=\"700\">凑出来的 batch 比单台大得多</tspan>；两边的机器数也能分开调。",
        "代价是每一层都有一趟来回，得靠几个小批轮着跑藏住（论文里要三四个才藏得住）；两边之间的网络要够快。",
    ])
    yb = f.src(yb + 10, "📌 MegaScale-Infer arXiv 2504.02263（disaggregated expert parallelism、ping-pong pipeline、每 GPU 吞吐最高 1.90×）；"
                        "Step-3 arXiv 2507.19427；vLLM 于 2026-07 发布实验性 AFD 插件。")
    f.save("fig5-afd.svg", yb + 14)


fig_pd()
fig_afd()
