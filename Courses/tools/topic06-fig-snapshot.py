# -*- coding: utf-8 -*-
r"""专题六 · 第十三节「新架构出难题：状态没法按 token 切」的静态图。

fig6-snapshot.svg —— 回答：线性层的前缀缓存为什么只能「退到存档点」？一份存档多大？存多密？
  · 数字：Qwen3.5-397B-A17B config.json（15 层全注意力 × 2 个 KV 头 × 256 维；45 层 GDN，
    v 头 64、k 头 16、每头 128 维、卷积宽 4、mamba_ssm_dtype float32；模型 bf16）。
    状态形状按 vLLM mamba_utils.py：递归状态 (v_heads, d_v, d_k)，卷积 (2·k_heads·d_k + v_heads·d_v) × (宽−1)。
  · 留存策略：prompt 末尾必留、按间隔（例 32K）多留 —— vLLM K3 博客（VLLM_PREFIX_CACHE_RETENTION_INTERVAL）；
    「每 256 都留」对应 SGLang --mamba-track-interval 默认 256 且路径不封顶时的上限，实际受池大小与 LRU 约束。
  · int8：SGLang --enable-int8-mamba-checkpoint 只量化递归状态，卷积状态保持原 dtype（mem_cache/mamba_checkpoint_pool.py）。
  · 本课推导：所有 MiB、GiB、份数、百分比。
  · 刻意没画：滑窗层（Qwen3.5 没有）；多级存储（4.6 讲过）。
"""

from topic03_draw import Fig, wpx, BL, OR, GR, RD, PU, CY, GY, INK, GY2, LINE

W = 1400
KIB, MIB = 1024, 1024 ** 2
KV_TOK = 2 * 2 * 256 * 2 * 15                       # 每 token KV 字节
SSM_L = 64 * 128 * 128 * 4                          # 每层递归状态（fp32）
CONV_L = (2 * 16 * 128 + 64 * 128) * 3 * 2          # 每层卷积状态（bf16）
SNAP = 45 * (SSM_L + CONV_L)
SNAP_I8 = 45 * (SSM_L // 4 + CONV_L)
assert KV_TOK == 30 * KIB and CONV_L == 72 * KIB
assert round(SNAP / MIB) == 183 and round(SNAP_I8 / MIB) == 48
assert round(SNAP / KV_TOK, -1) == 6250
SEG = 256 * KV_TOK                                  # 256 token 的 KV
assert SEG == 7.5 * MIB and round(SNAP / SEG) == 24
P128 = 131072
KV128 = P128 * KV_TOK / 1024 ** 3                   # 128K 前缀的 KV
ROWS = [(lab, n, n * SNAP / 1024 ** 3) for lab, n in
        (("只留 prompt 末尾", 1), ("再每 32K 留一份", P128 // 32768), ("每 256 个都留", P128 // 256))]
assert KV128 == 3.75 and [r[1] for r in ROWS] == [1, 4, 512]
assert [round(r[2] / KV128 * 100) for r in ROWS[:2]] == [5, 19] and round(ROWS[2][2]) == 92


def fig():
    f = Fig(W, "线性层的前缀缓存只能退到存档点。上图：一条 1,024 个 token 的前缀，全注意力每个 token 一格，"
               "线性层只在 256、512、768 处各存一份快照。新请求和它共用前 600 个 token，"
               "全注意力本可以从 600 接着算，但线性层最近的存档在 512，整个模型只能从 512 接着算，512 到 600 这 88 个 token 重算。"
               "左下：一份快照有多大。256 个 token 的 KV 约 7.5 MiB；一份快照约 183 MiB，是它的 24 倍；存 int8 约 48 MiB。"
               "右下：一条 128K 的前缀，KV 约 3.75 GiB。只留 prompt 末尾一份快照约 0.18 GiB，是 KV 的 5%；"
               "再每 32K 留一份共 4 份约 0.72 GiB，19%；要是每 256 个都留，512 份约 92 GiB，生产不这么做。"
               "都按 Qwen3.5-397B 的 config 推算")
    y0 = f.header("线性层只能退到存档点　——　<tspan font-weight=\"700\">一份顶六千多个 token 的 KV，只留要紧的几份</tspan>",
                  "Qwen3.5-397B · 全模型未切分 · 字节按 config 推算（本课推导）",
                  [(BL, "全注意力 KV"), (OR, "线性层快照"), (RD, "要重算")])
    # ── 上：存档点 ──
    PH = 250
    py = f.panel(0, y0, W, PH, "新请求和老前缀共用前 600 个 token", INK)
    X0, X1 = 170, W - 60
    U = (X1 - X0) / 1024.0
    ya, yb = py + 34, py + 120
    f.t(24, ya + 20, "全注意力", BL, True, 16)
    f.t(24, ya + 40, "每 token 一份", GY, False, 13)
    for k in range(64):
        f.box(X0 + k * 16 * U + 1, ya + 6, 16 * U - 2, 26, BL if k * 16 < 600 else "#e8f0fe", BL, 2)
    f.t(24, yb + 20, "线性层", OR, True, 16)
    f.t(24, yb + 40, "只在存档点有", GY, False, 13)
    f.line(X0, yb + 20, X1, yb + 20, GY2, 2, arrow=False)
    for tok in (256, 512, 768):
        x = X0 + tok * U
        f.line(x, yb + 34, x, yb - 4, OR, 2.5, arrow=False)
        f.p.append('<path d="M %.1f %.1f L %.1f %.1f L %.1f %.1f Z" fill="%s"/>' % (x, yb - 4, x + 26, yb + 5, x, yb + 14, OR))
        f.t(x, yb + 52, "%d" % tok, OR, True, 14, "middle")
    xs = X0 + 600 * U
    f.line(xs, ya - 2, xs, yb + 62, INK, 1.6, "5 4", arrow=False)
    f.t(xs + 8, ya - 4, "共享前缀到 600", INK, True, 14)
    xr = X0 + 512 * U
    f.box(xr, yb + 66, xs - xr, 22, RD, RD, 3)
    f.t(xs + 10, yb + 82, "退到 512：整个模型从这儿接着算，512–600 这 88 个 token 重算", RD, True, 14)
    f.t(X1, ya + 50, "→ token", GY, False, 13, "end")
    f._pan = None
    # ── 左下：一份多大 ──
    y2 = y0 + PH + 18
    LW, PH2 = 700, 230
    py = f.panel(0, y2, LW, PH2, "一份快照有多大", OR)
    SC = 2.7 / MIB                                   # 每 MiB 2.7 px
    bars = [("256 token 的 KV", SEG, BL, "7.5 MiB"),
            ("一份快照", SNAP, OR, "≈183 MiB（24 倍）"),
            ("int8 存", SNAP_I8, GR, "≈48 MiB")]
    yy = py + 30
    for lab, b, col, txt in bars:
        f.t(24, yy + 22, lab, INK, True, 15)
        f.box(170, yy + 4, max(b * SC, 4), 28, col, col, 3)
        if b * SC > wpx(txt, 15) + 24:
            f.t(170 + b * SC - 12, yy + 24, txt, "#ffffff", True, 15, "end")
        else:
            f.t(170 + b * SC + 10, yy + 24, txt, col, True, 15)
        yy += 54
    assert 170 + SNAP * SC + 20 < LW
    f._pan = None
    # ── 右下：留几份 ──
    RX, RW = LW + 20, W - LW - 20
    py = f.panel(RX, y2, RW, PH2, "128K 前缀：快照留几份", OR)
    cols = [RX + 24, RX + 250, RX + 340, RX + 490]
    for c, h in zip(cols, ("留法", "份数", "快照显存", "占前缀 KV")):
        f.t(c, py + 30, h, GY, True, 14)
    yy = py + 64
    for i, (lab, n, gib) in enumerate(ROWS):
        bad = i == 2
        col = RD if bad else INK
        pct = "%d 倍，不这么做" % round(gib / KV128) if bad else "%d%%" % round(gib / KV128 * 100)
        for c, txt in zip(cols, (lab, "%d" % n, "%.2f GiB" % gib if gib < 10 else "%.0f GiB" % gib, pct)):
            f.t(c, yy, txt, col, bad, 15)
        yy += 40
    f.t(RX + 24, yy + 6, "同一段前缀的 KV 约 %.2f GiB" % KV128, BL, True, 15)
    f._pan = None
    yb2 = f.band(y2 + PH2 + 20, "warn", "生产只留要紧的几份：prompt 末尾必留，长前缀隔一大段一份，共享前缀第二次撞见才存", [
        "放不下按 LRU 删快照、KV 留着；还能落到 CPU 内存和外部存储，或存成 int8。"])
    yb2 = f.src(yb2 + 10,
                "Qwen3.5-397B-A17B config.json；状态形状按 vLLM mamba_utils.py；留存策略取自 vLLM K3 博客（2026-07-27）；"
                "int8 只压递归状态（SGLang mamba_checkpoint_pool.py）。",
                "所有 MiB、GiB、份数、百分比为本课推导，未用实测显存核对。")
    f.save("fig6-snapshot.svg", yb2 + 14)


fig()
