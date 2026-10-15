# -*- coding: utf-8 -*-
r"""专题六 · 第一节「两个阶段：prefill 吃算力，decode 吃带宽」的四张静态图。

fig6-ai.svg —— 回答：同一个矩阵乘，一次喂多少行（M）才不被显存带宽卡住？prefill 和 decode 各落在哪？
  · 数字：一个 7168×7168 的 bf16 矩阵乘（7168 ＝ DeepSeek-V3 hidden_size，config.json）。
    运算强度 ＝ 运算次数 ÷ 搬的字节 ＝ 2·M·H² ÷ 2·(H² ＋ 2·M·H) ＝ M·H ÷ (H ＋ 2M)（本课推导，
    权重读一遍、输入读一遍、输出写一遍）。屋脊点 313 ＝ v7 bf16 2,307 TFLOP/s ÷ 7.37 TB/s（专题二）。
  · 刻意没画：注意力那部分（它的强度随 batch 不涨，第五节讲）；MoE 的专家层（见 fig6-moe-rows）。

fig6-b1.svg —— 回答：DeepSeek-V3 一个人解码一步，在一颗 v7 上时间花在哪？
  · 数字：每 token 激活 37B 参数（技术报告摘要）× 1 字节（FP8 权重）＝ 37 GB；
    ÷ 7.37 TB/s ＝ 5.02 ms；运算 2 × 37B ＝ 74 GFLOP ÷ FP8 峰值 4,614 TFLOP/s ＝ 0.016 ms。
    算力利用率 ＝ 0.016 ÷ 5.02 ＝ 0.32%，恰好是 1/313（本课推导）。
  · 刻意没画：KV 的读取（一个人、短上下文时不到 1%，见 fig6-wkv）；
    671 GB（≈ 625 GiB）权重一颗 192 GiB 的芯片装不下，至少要 4 颗 —— 摊到 N 颗上两条时间都 ÷N，比例不变。

fig6-wkv.svg —— 回答：上下文变长、并发变大时，每出一个字要搬的字节里，KV 占多少？
  · 数字：权重部分 W(b) ÷ b，W(b) 是 b 个 token 同一步里一共要读的权重字节（FP8），
    假设路由均匀：一个专家被 b 个 token 都没挑中的概率是 (1 − 8/256)^b（本课推导）。
    KV 部分 ＝ 70,272 字节/token（MLA，bf16，61 层，topic06_numbers）× 上下文长度。
  · 刻意没画：激活与通信字节；KV 用 FP8 存的情形（会减半）；路由不均匀（会让被碰到的专家更少）。

fig6-moe-rows.svg —— 回答：256 个人一起解码，MoE 的每个专家分到几行？
  · 数字：256 × top-8 ÷ 256 个专家 ＝ 平均 8 行（config.json）；v7 MXU 一边 256
    （JAX 公开源码 jax/_src/pallas/mosaic/tpu_info.py：TPU 7 的 mxu_column_size＝256）。
    专家矩阵乘 M＝8 时运算强度约 8（本课推导），屋脊点 313。
  · 刻意没画：路由不均（有的专家 30 行、有的 0 行）；共享专家（它每个 token 都过，照样分到 256 行）。
"""

import math
import os

from topic03_draw import Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE
from topic06_numbers import (V3_HIDDEN, V3_EXPERTS, V3_TOPK, V3_TOTAL, V3_ACTIVE,
                             V3_KV_PER_TOKEN, V7_FP8, V7_BF16, V7_BW, V7_RIDGE, V7_HBM, GB)

W = 1400
HERE = os.path.dirname(os.path.abspath(__file__))

# ═══ 承重数：全部在这里算、在这里 assert ═════════════════════════════════
H = V3_HIDDEN
RIDGE = V7_RIDGE
assert round(RIDGE) == 313


def ai(m, h=H):
    """bf16 方阵 h×h 的矩阵乘，M 行输入：运算 2·M·h² ÷ 字节 2·(h² ＋ 2·M·h)。"""
    return (2 * m * h * h) / (2 * (h * h + 2 * m * h))


AI_B1, AI_B64, AI_B256, AI_PRE = ai(1), ai(64), ai(256), ai(4096)
assert round(AI_B1, 1) == 1.0 and round(AI_B64) == 63 and round(AI_B256) == 239
assert round(AI_PRE) == 1911
M_CROSS = RIDGE * H / (H - 2 * RIDGE)        # M·H/(H+2M) ＝ 313 解出来
assert round(M_CROSS) == 343
AI_CAP = H / 2                               # M→∞ 的上限：H/2（输出写回那一项随 M 涨）
# 注：M·H/(H+2M) → H/2 ＝ 3,584
assert round(AI_CAP) == 3584

# ── batch 1 解码一步（FP8 权重）──
B1_BYTES = V3_ACTIVE * 1                     # 37 GB
B1_T_MEM = B1_BYTES / V7_BW                  # 秒
B1_FLOP = 2 * V3_ACTIVE
B1_T_CMP = B1_FLOP / V7_FP8
B1_UTIL = B1_T_CMP / B1_T_MEM
assert round(B1_T_MEM * 1e3, 2) == 5.02
assert round(B1_T_CMP * 1e3, 3) == 0.016
assert round(B1_UTIL * 100, 2) == 0.32
assert abs(B1_UTIL - 1 / RIDGE) < 1e-6      # 恰好 1/313：FP8 屋脊 626，FP8 解码强度 2
MIN_CHIPS = math.ceil(V3_TOTAL / V7_HBM)
assert MIN_CHIPS == 4                        # 671e9 B ÷ (192 × 2^30 B) ＝ 3.3 → 4 颗（还没算 KV）
assert round(V3_TOTAL / 2**30) == 625

# ── MoE 权重模型（config.json：hidden 7168、moe_intermediate 2048、61 层里前 3 层稠密）──
EXPERT = 3 * H * 2048                        # 一个专家：gate / up / down 三块
MOE_LAYERS = 58
ROUTED = EXPERT * V3_EXPERTS * MOE_LAYERS    # 全部路由专家
assert round(ROUTED / 1e9, 1) == 653.9
ROUTED_ACTIVE = V3_TOPK * MOE_LAYERS * EXPERT
REST = V3_ACTIVE - ROUTED_ACTIVE             # 每 token 都过的部分（注意力、稠密层、共享专家、embedding…）
P_HIT = V3_TOPK / V3_EXPERTS                 # 一个 token 挑中某个专家的概率 1/32


def w_step(b):
    """b 个 token 同一步要读的权重字节（FP8，路由均匀）。"""
    return REST + ROUTED * (1 - (1 - P_HIT) ** b)


assert abs(w_step(1) - 37e9) < 1e3
assert abs((REST + ROUTED) / 1e9 - 670.5) < 0.1          # 上限对上 671B 锚点
W_PT = {b: w_step(b) / b for b in (1, 64, 256)}           # 每个字分到的权重字节
assert round(W_PT[1] / GB, 1) == 37.0
assert round(W_PT[64] / GB, 2) == 9.14
assert round(W_PT[256] / GB, 2) == 2.62
TOUCH64 = 1 - (1 - P_HIT) ** 64
assert round(TOUCH64 * 100) == 87                         # 64 人一起，87% 的专家被碰到
CTXS = [4096, 32768, 131072]
KV_PT = {L: V3_KV_PER_TOKEN * L for L in CTXS}            # 每个字要读的 KV（这个人自己的）
assert round(KV_PT[4096] / GB, 2) == 0.29
assert round(KV_PT[32768] / GB, 2) == 2.30
assert round(KV_PT[131072] / GB, 2) == 9.21
SHARE = {(b, L): KV_PT[L] / (W_PT[b] + KV_PT[L]) for b in W_PT for L in CTXS}
assert round(SHARE[(1, 4096)] * 100, 1) == 0.8
assert round(SHARE[(1, 131072)] * 100) == 20
assert round(SHARE[(64, 131072)] * 100) == 50
assert round(SHARE[(256, 131072)] * 100) == 78

# ── 每个专家分到几行 ──
BATCH = 256
ROWS = BATCH * V3_TOPK / V3_EXPERTS
assert ROWS == 8
MXU = 256                                    # v7 MXU 边长（JAX tpu_info：TPU 7 mxu_column_size＝256）
assert MXU / ROWS == 32
AI_EXP = (2 * ROWS * H * 2048) / (2 * (H * 2048 + ROWS * (H + 2048)))
assert round(AI_EXP) == 8
AI_DENSE256 = AI_B256


def fmt(x):
    return "{:,}".format(int(round(x)))


# ═══ fig6-ai ════════════════════════════════════════════════════════════
def fig_ai():
    f = Fig(W, "运算强度随一次喂进的行数 M 变化的曲线，横纵都是对数轴。矩阵是 7168×7168 的 bf16 权重。"
               "M＝1 时强度约 1，远低于 v7 的屋脊点 313；decode 一次 64 个人是 63，256 个人是 239，仍在屋脊点下面；"
               "prefill 一段 4K 的 prompt，M＝4096，强度 1,911，远在屋脊点上面。曲线在 M≈343 处穿过屋脊点")
    y0 = f.header("一次喂几行，决定卡在哪　——　<tspan font-weight=\"700\">decode 在屋脊点下面，prefill 在上面</tspan>",
                  "一个 7168×7168 的 bf16 矩阵乘：横轴是一次喂进去的行数 M（token 数），纵轴是每搬 1 字节做几次运算。两轴都是对数",
                  [(GR, "decode：M ＝ 同时解码的人数"), (OR, "prefill：M ＝ prompt 长度"), (RD, "v7 屋脊点")])
    PH = 560
    py = f.panel(0, y0, W, PH, "运算强度 ＝ 运算次数 ÷ 搬的字节", INK)
    X0, X1 = 120, 1330
    Y0, Y1 = py + 40, py + 450              # Y0 顶、Y1 底
    LXmin, LXmax = 0, math.log10(16384)
    LYmin, LYmax = math.log10(0.3), math.log10(6000)

    def X(m):
        return X0 + (math.log10(m) - LXmin) / (LXmax - LXmin) * (X1 - X0)

    def Y(v):
        return Y1 - (math.log10(v) - LYmin) / (LYmax - LYmin) * (Y1 - Y0)

    # 轴与刻度
    f.line(X0, Y1, X1, Y1, GY2, 1.2, arrow=False)
    f.line(X0, Y0, X0, Y1, GY2, 1.2, arrow=False)
    for m in (1, 4, 16, 64, 256, 1024, 4096, 16384):
        f.line(X(m), Y1, X(m), Y1 + 6, GY2, 1, arrow=False)
        f.t(X(m), Y1 + 24, fmt(m), GY, size=13, anchor="middle")
    f.t((X0 + X1) / 2, Y1 + 50, "M：一次喂进去的行数（token 数）", GY, True, 14, "middle")
    for v in (1, 10, 100, 1000):
        f.line(X0 - 6, Y(v), X0, Y(v), GY2, 1, arrow=False)
        f.line(X0, Y(v), X1, Y(v), LINE, 0.8, dash="3,4", arrow=False)
        f.t(X0 - 12, Y(v) + 5, fmt(v), GY, size=13, anchor="end")
    f.t(28, Y0 + 4, "运算／字节", GY, True, 13)

    # 屋脊点
    f.line(X0, Y(RIDGE), X1, Y(RIDGE), RD, 1.8, dash="7,5", arrow=False)
    f.t(X1 - 6, Y(RIDGE) - 10, "v7 屋脊点 313（bf16）", RD, True, 15, "end")
    f.t(X0 + 16, Y(RIDGE) - 14, "↑ 这一侧卡在算力", GY, size=14)
    f.t(X0 + 16, Y(RIDGE) + 26, "↓ 这一侧卡在显存带宽", GY, size=14)

    # 曲线
    pts = []
    k = 0
    while True:
        m = 10 ** (k / 40.0)
        if m > 16384:
            break
        pts.append((round(X(m), 1), round(Y(ai(m)), 1)))
        k += 1
    f.path(pts, INK, 2.4, arrow=False)
    # 上限
    f.line(X(1024), Y(AI_CAP), X1, Y(AI_CAP), GY2, 1.2, dash="2,4", arrow=False)
    f.t(X1 - 6, Y(AI_CAP) - 10, "上限 3,584：输入输出也要读写", GY, size=13, anchor="end")

    # 交点
    f.line(X(M_CROSS), Y(RIDGE), X(M_CROSS), Y1, RD, 1.2, dash="3,3", arrow=False)
    f.p.append('<circle cx="%.1f" cy="%.1f" r="5" fill="#fff" stroke="%s" stroke-width="2"/>'
               % (X(M_CROSS), Y(RIDGE), RD))
    f.t(X(M_CROSS) + 10, Y1 - 12, "M ≈ 343 才过线", RD, True, 14)

    # 四个点
    for m, v, col, lab, dx, dy, anc in (
            (1, AI_B1, GR, "decode 1 人：1.0", 16, 26, None),
            (64, AI_B64, GR, "decode 64 人：63", 14, 22, None),
            (256, AI_B256, GR, "decode 256 人：239", 56, 62, None),
            (4096, AI_PRE, OR, "prefill 4K prompt：1,911", -16, -6, "end")):
        f.p.append('<circle cx="%.1f" cy="%.1f" r="8" fill="%s"/>' % (X(m), Y(v), col))
        f.t(X(m) + dx, Y(v) + dy, lab, col, True, 15, anc)
        if m == 256:                     # 标签挪到交点竖线右侧，拉一根引线
            f.line(X(m) + 6, Y(v) + 6, X(m) + dx - 4, Y(v) + dy - 14, col, 1.2, arrow=False)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "decode 再多拼人也难过线，prefill 一段 prompt 就远远过线", [
        "decode 一步每人只喂 1 行，256 个人拼一起也才 239；prefill 一段 4K prompt 一次就是 4,096 行。",
        "所以同一块芯片，prefill 时算力是瓶颈，decode 时显存带宽是瓶颈。",
    ])
    yb = f.src(yb + 10,
               "📌 7168 ＝ DeepSeek-V3 config.json 的 hidden_size。屋脊点 313 ＝ v7 每芯片 bf16 2,307 TFLOP/s ÷ HBM 7.37 TB/s"
               "（Google Cloud 公开规格；bf16 是 FP8 4,614 的一半，本课推导；专题二同一个数）。",
               "📌 本课推导：运算 2·M·H²，字节 2·(H² ＋ 2·M·H)（权重读一遍、输入读一遍、输出写一遍，bf16 每数 2 字节），"
               "强度 ＝ M·H ÷ (H ＋ 2M)。令它等于 313 解出 M ≈ 343。",
               "⚠️ 只算权重时强度就是 M，过线点正好是 313（专题五 fig5-decode-ai 的口径）；这里把输入输出也算上，过线点往右挪到 343。",
               "⚠️ 刻意没画：注意力（它读的是 KV，强度不随人数涨，第五节讲）和 MoE 专家层（见本节专家行数那张图）。")
    f.save("fig6-ai.svg", yb + 14)


# ═══ fig6-b1 ════════════════════════════════════════════════════════════
def fig_b1():
    f = Fig(W, "DeepSeek-V3 一个人解码一步，在一颗 TPU v7 上的时间账。要读的权重是 37 GB（370 亿激活参数，FP8 每个 1 字节），"
               "按 7.37 TB/s 读完要 5.02 毫秒；要做的运算是 740 亿次，按 FP8 峰值算只要 0.016 毫秒。"
               "算力只用了 0.32%，恰好是 313 分之一")
    y0 = f.header("一个人解码一步：<tspan font-weight=\"700\">读权重 5.02 ms，算只要 0.016 ms</tspan>",
                  "DeepSeek-V3，batch 1，一颗 TPU v7 的规格，只算权重。两根条用同一把时间尺",
                  [(BL, "从显存读权重"), (OR, "MXU 做运算")])
    PH = 330
    py = f.panel(0, y0, W, PH, "一步解码的时间（同一把尺）", INK)
    X0, X1 = 250, 1330
    SC = (X1 - X0) / (B1_T_MEM * 1e3)       # px / ms
    ry1, ry2, RH = py + 50, py + 150, 56
    f.t(30, ry1 + 24, "读权重", BL, True, 17)
    f.t(30, ry1 + 46, "37 GB ÷ 7.37 TB/s", GY, size=13)
    f.box(X0, ry1, (X1 - X0), RH, BL, BL, 4)
    f.t(X0 + (X1 - X0) / 2, ry1 + 35, "5.02 ms", "#ffffff", True, 20, "middle", big=True)
    f.t(30, ry2 + 24, "做运算", OR, True, 17)
    f.t(30, ry2 + 46, "74 GFLOP ÷ 4,614 TFLOP/s", GY, size=13)
    cw = max(B1_T_CMP * 1e3 * SC, 3)
    f.box(X0, ry2, cw, RH, OR, OR, 1)
    f.box(X0 + cw, ry2, X1 - X0 - cw, RH, "none", LINE, 4, dash="5,4")
    f.t(X0 + 18, ry2 + 35, "0.016 ms", OR, True, 18)
    f.t(X0 + (X1 - X0) / 2 + 60, ry2 + 35, "其余时间：MXU 在等数据", GY, size=15, anchor="middle")
    # 尺
    ty = ry2 + RH + 36
    f.line(X0, ty, X1, ty, GY2, 1, arrow=False)
    for ms in range(0, 6):
        x = X0 + ms * SC
        if x <= X1 + 1:
            f.line(x, ty, x, ty + 6, GY2, 1, arrow=False)
            f.t(x, ty + 24, "%d ms" % ms, GY, size=13, anchor="middle")
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "算力利用率 0.32%：一个人解码，芯片几乎全在等显存", [
        "0.016 ÷ 5.02 ＝ 0.32%，正好是 1/313：FP8 每读 1 字节只做 2 次运算，FP8 的屋脊点约 626，差 313 倍。",
        "换更快的 MXU 毫无用处，这一步的下限是显存带宽定的。",
    ])
    yb = f.src(yb + 10,
               "📌 37B 激活参数、671B 总参数：DeepSeek-V3 技术报告摘要。FP8 权重每参数 1 字节（V3 原生发布 FP8）。"
               "v7 每芯片 FP8 4,614 TFLOP/s、HBM 7.37 TB/s：Google Cloud 公开规格。",
               "📌 本课推导：37e9 ÷ 7.37e12 ＝ 5.02 ms；2 × 37e9 ÷ 4,614e12 ＝ 0.016 ms；比值 0.32% ＝ 2 ÷ (4,614 ÷ 7.37) ＝ 1/313。",
               "⚠️ 理想下限：只算权重，不算 KV、激活、通信和 kernel 开销，实测只会更慢。",
               "⚠️ 671 GB（≈ 625 GiB）一颗 192 GiB 的芯片放不下，至少 4 颗（还没算 KV）。摊到 N 颗上，两根条都 ÷N，0.32% 不变。")
    f.save("fig6-b1.svg", yb + 14)


# ═══ fig6-wkv ═══════════════════════════════════════════════════════════
def fig_wkv():
    f = Fig(W, "每出一个字要从显存读的字节，拆成权重和 KV 两段。三组分别是 1 人、64 人、256 人同时解码，"
               "每组里上下文 4K、32K、128K。权重那段随人数变少，因为一次读的权重分给更多人；"
               "KV 那段只随上下文变长，每人一份，没法分。256 人、128K 时 KV 占 78%")
    y0 = f.header("每出一个字搬多少：<tspan font-weight=\"700\">权重能分摊，KV 每人一份</tspan>",
                  "DeepSeek-V3，FP8 权重，bf16 的 MLA KV。每根条 ＝ 一步读的权重 ÷ 人数 ＋ 这个人自己的 KV。同一把尺",
                  [(BL, "分到每个字的权重"), (PU, "这个人自己的 KV")])
    PH = 560
    py = f.panel(0, y0, W, PH, "每个字的字节（GB）", INK)
    X0, X1 = 330, 1250
    vmax = (W_PT[1] + KV_PT[131072]) / GB
    SC = (X1 - X0) / 48.0
    assert vmax < 48
    BH, GAP, GGAP = 34, 10, 34
    y = py + 30
    names = {1: "1 人", 64: "64 人", 256: "256 人"}
    hooks = {1: "整份 37 GB 只给一个人", 64: "每人摊到 9.14 GB（87% 专家被碰到）", 256: "每人摊到 2.62 GB"}
    for b in (1, 64, 256):
        f.t(30, y + 22, names[b] + "同时解码", INK, True, 16)
        f.t(30, y + 44, hooks[b], BL, size=14)
        for L in CTXS:
            wv, kv = W_PT[b] / GB, KV_PT[L] / GB
            f.t(X0 - 12, y + 23, "%dK" % (L // 1024), GY, True, 14, "end")
            ww, kw = wv * SC, kv * SC
            f.box(X0, y, ww, BH, BL, BL, 2)
            f.box(X0 + ww, y, max(kw, 2), BH, PU, PU, 2)
            sh = SHARE[(b, L)] * 100
            lab = "KV %s%%" % (("%.1f" % sh) if sh < 1 else "%d" % round(sh))
            f.t(X0 + ww + kw + 10, y + 23, lab, PU, True, 14)
            y += BH + GAP
        y += GGAP - GAP
    # 尺
    f.line(X0, y, X1, y, GY2, 1, arrow=False)
    for g in range(0, 49, 8):
        f.line(X0 + g * SC, y, X0 + g * SC, y + 6, GY2, 1, arrow=False)
        f.t(X0 + g * SC, y + 24, "%d" % g, GY, size=13, anchor="middle")
    f.t(X1 + 26, y + 24, "GB／字", GY, size=13)
    assert y + 30 < py + PH
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "人越多、上下文越长，这一趟里 KV 越重", [
        "权重一次读完分给所有人；KV 是每人自己的历史，128K 时每个字要读 9.21 GB，拼多少人都不变。",
        "256 人、128K 时 KV 已占 78%：再往后，省的就得是 KV 本身（第三、七节）。",
    ])
    yb = f.src(yb + 10,
               "📌 本课推导：KV ＝ 70,272 字节／token（MLA 每层存 512 ＋ 64 个数，bf16，61 层，config.json）× 上下文。"
               "128K 时 9.21 GB ＝ 8.58 GiB，与专题五同一个数。",
               "📌 本课推导：b 人一步读的权重 ＝ 不经路由的部分 ＋ 653.9B 路由专家 × [1 − (1 − 8/256)^b]（FP8，每参数 1 字节）。"
               "b＝1 时恰好 37 GB，b 很大时趋近 670.5 GB，对上 671B。",
               "⚠️ 64 人时每人摊到 9.14 GB 而不是 37 ÷ 64：每人挑的专家不同，一步里 87% 的专家都被碰到、都要读。",
               "⚠️ 假设路由均匀、所有人上下文一样长；没算激活和通信字节。KV 改存 FP8 时紫段减半。")
    f.save("fig6-wkv.svg", yb + 14)


# ═══ fig6-moe-rows ══════════════════════════════════════════════════════
def fig_moe():
    f = Fig(W, "256 个人一起解码。稠密层里这 256 行一起走同一块权重，正好填满 v7 MXU 256 的一边；"
               "MoE 层里每人挑 8 个专家，256 个专家平均每个只分到 8 行，只填满 MXU 一边的 32 分之一，"
               "运算强度约 8，离屋脊点 313 很远")
    y0 = f.header("MoE 把一批人拆散：<tspan font-weight=\"700\">256 人进来，每个专家只分到 8 行</tspan>",
                  "同一批 256 人解码一步。方块是 v7 的 MXU（一边 256），填色的是这一步真正喂进去的行",
                  [(GR, "这一步喂进 MXU 的行（decode 的 token）")])
    PH = 410
    py = f.panel(0, y0, W, PH, "稠密层 vs 一个专家", INK)
    S = 256                                   # 1 px ＝ 1 行
    for i, (x0, title, rows, sub, intensity, col) in enumerate((
            (110, "稠密层：256 人共用一块权重", 256, "256 行，填满一边", AI_DENSE256, GR),
            (770, "MoE 层：一个专家", ROWS, "平均 8 行，占一边的 1/32", AI_EXP, GR))):
        top = py + 70
        f.t(x0, top - 22, title, INK, True, 17)
        f.box(x0, top, S, S, "none", GY2, 2, dash="4,3")
        f.box(x0, top, S, rows, col, col, 0)
        f.t(x0 + S / 2, top + S + 26, "MXU 256 × 256", GY, size=13, anchor="middle")
        tx = x0 + S + 30
        f.t(tx, top + 30, sub, col, True, 16)
        f.t(tx, top + 70, "运算强度 ≈ %s" % fmt(intensity), INK, True, 16)
        f.t(tx, top + 98, "屋脊点 313", RD, size=14)
    f.t(770 + S + 30, py + 70 + 140, "256 × 8 ÷ 256 ＝ 8", GY, size=14)
    f.t(110 + S + 30, py + 70 + 140, "仍在线下，但差得不多", GY, size=14)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "拼来的 256 人，到了专家这里又被拆成 256 份", [
        "每人只挑 8 个专家，一个专家平均只分到 8 行，强度约 8，比稠密层差 30 倍。",
        "要让专家吃饱，得把很多卡上的人收到一起：这是第五、六节的事。",
    ])
    yb = f.src(yb + 10,
               "📌 256 个路由专家、每 token 挑 8 个：DeepSeek-V3 config.json（n_routed_experts、num_experts_per_tok）。"
               "MXU 一边 256：JAX 公开源码 jax/_src/pallas/mosaic/tpu_info.py，TPU 7 的 mxu_column_size＝256。",
               "📌 本课推导：专家矩阵 7168×2048，M＝8 时强度 ＝ 2·8·7168·2048 ÷ 2·(7168·2048 ＋ 8·(7168 ＋ 2048)) ≈ 8。"
               "稠密层 239 来自本节第一张图。",
               "⚠️ 8 是平均数，路由不均时有的专家几十行、有的零行。共享专家每个 token 都过，不受影响。")
    f.save("fig6-moe-rows.svg", yb + 14)


fig_ai()
fig_b1()
fig_wkv()
fig_moe()
