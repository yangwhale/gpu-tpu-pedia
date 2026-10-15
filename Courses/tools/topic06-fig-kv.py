# -*- coding: utf-8 -*-
r"""专题六 · 第三节「KV cache：账、池、复用」的四张图。

⭐ 回答的问题（一张图一个）：
   fig6-kv-batch  —— 一颗 v7 芯片放下 V3 的 1/8 权重后，4K／32K／128K 上下文各能同时装几条序列？
                     MHA／GQA／MLA／线性＋全注意力混合，四种 KV 记账方式差多少？
   fig6-paged     —— PagedAttention 的页表怎么把「逻辑上连续」的 KV 映射到零散的物理块，
                     两条共用 prompt 的序列怎么共享块、写时复制？
   fig6-swa-ratio —— V4-Pro 的两个 KV 池（滑窗池／全长池）预算怎么分？哪个池先满？
   fig6-offload   —— 存下来的 KV「搬回来」和「重算一遍」哪个快？交点在哪？

📌 数字从哪来：
   · V3 形状、v7 规格：topic06_numbers.py（V3 config.json；v7 公开规格）。
   · Llama-3.1-70B：HF config.json（80 层、8 个 KV 头、head_dim 128）；激活参数 70.6B 取自 HF 模型页。
   · Qwen3.5-397B-A17B：HF config.json（60 层 = 45 层线性注意力 ＋ 15 层全注意力；全注意力 2 个 KV 头 × 256；
     线性层 64 个 value 头 × key 128 × value 128，mamba_ssm_dtype float32）。
   · 可用 189.5 GiB ＝ 每 device 实测可用 94.75 GiB × 2（tpu/tpu-inference R1 FP4 README:59，与第十节 10.5 同一口径）；
     「留 10% 给激活」是本课假设。
   · V4-Pro 两池实验：gpu/inference/a4x-max/deepseek-v4/SGLANG-V4PRO-RUNBOOK.md §11.5（约 L849–896，
     每张 decode GPU 的 e2e 输出吞吐，ISL 4096／8192，OSL 1024，关 radix cache）。
     （原文 Google Developers Blog「Systems Engineering Playbook: Optimizing Qwen 3.5-397B MoE on Ironwood」）。
   · 其余（batch 上限、搬回 vs 重算的交点）都是本课推导，下面每个数一条断言。

⛔ 刻意没画：
   · 激活值、通信缓冲占的显存（统一算进「留 10%」里，不单列）。
   · 线性层的卷积状态（conv kernel 4，比递推状态小两个数量级）。
   · 搬回时的延迟项、和 decode 抢带宽的干扰：offload 图只比纯带宽和纯算力。
   · 真实链路带宽：1 GB/s、20 GB/s 是示意档位，不对应某一款硬件。
"""
from math import log10, log2, floor

from topic03_draw import Fig, BL, OR, GR, RD, GY, GY2, PU, INK, LINE
from topic06_numbers import (V3_LAYERS, V3_KV_PER_TOKEN, V3_TOTAL, V3_ACTIVE,
                             V7_FP8, GB, GIB)

W = 1400
K = 1024

# ══ ① 每个 token 的 KV 字节：四种记账 ═════════════════════════════════
KV_MLA = V3_KV_PER_TOKEN                          # (512 + 64) × 61 层 × 2 B
assert KV_MLA == 70272
KV_MHA = 2 * 128 * 128 * V3_LAYERS * 2            # V3 形状若用 MHA：K、V 各 128 头 × 128 维（专题三第一题）
assert KV_MHA == 3997696
KV_GQA = 2 * 8 * 128 * 80 * 2                     # Llama-3.1-70B：K、V × 8 头 × 128 维 × 80 层 × bf16
assert KV_GQA == 327680
KV_HYB = 2 * 2 * 256 * 15 * 2                     # Qwen3.5：只有 15 层全注意力存 KV
assert KV_HYB == 30720
STATE_HYB = 45 * 64 * 128 * 128 * 4               # 45 层线性注意力，每条序列一个固定状态，fp32
assert STATE_HYB == 188743680 and STATE_HYB / 2**20 == 180

# ══ ② 一颗 v7 芯片的显存怎么分 ════════════════════════════════════════
N_CHIP = 8                                        # 671 GB FP8 权重铺 8 颗芯片（2×2×2）
W_CHIP = V3_TOTAL * 1 / N_CHIP                    # FP8 每参数 1 字节
assert W_CHIP == 83.875e9
USABLE = 189.5 * GIB                              # 每 device 实测可用 94.75 GiB × 2，同 10.5
RESERVE = USABLE * 0.1
POOL = USABLE * 0.9 - W_CHIP
assert "%.2f" % (USABLE * 0.9 / GIB) == "170.55"
assert "%.1f" % (POOL / GIB) == "92.4" and "%.1f" % (POOL / GB) == "99.3"
assert "%.1f" % (W_CHIP / GIB) == "78.1" and "%.1f" % (W_CHIP / GB) == "83.9"
assert "%.2f" % (RESERVE / GIB) == "18.95"


def nseq(per_tok, L, fixed=0):
    return int(floor(POOL / (per_tok * L + fixed)))


MODELS = [  # (名字, 每 token 字节, 固定字节, 颜色)
    ("MLA · DeepSeek-V3", KV_MLA, 0, GR),
    ("线性＋全注意力 · Qwen3.5 形状", KV_HYB, STATE_HYB, PU),
    ("GQA · Llama-3.1-70B", KV_GQA, 0, BL),
    ("MHA · V3 形状（假想）", KV_MHA, 0, RD),
]
TAB = {m[0]: {L: nseq(m[1], L * K, m[2]) for L in (1, 2, 4, 8, 16, 32, 64, 128)} for m in MODELS}
assert [TAB["MLA · DeepSeek-V3"][L] for L in (4, 32, 128)] == [344, 43, 10]
assert [TAB["GQA · Llama-3.1-70B"][L] for L in (4, 32, 128)] == [73, 9, 2]
assert [TAB["线性＋全注意力 · Qwen3.5 形状"][L] for L in (4, 32, 128)] == [315, 83, 23]
assert [TAB["MHA · V3 形状（假想）"][L] for L in (1, 4, 16, 32)] == [24, 6, 1, 0]

# ══ ③ V4-Pro 两池实验（runbook §11.5，每张 decode GPU） ══════════════════
RUNS = [  # (ISL, 配置, e2e tok/s, full 占用, swa 占用, 每 rank 请求数, 结论)
    ("4K", "ratio 0.20", 6887, 0.93, 0.63, 727, "全长池先满"),
    ("4K", "ratio 0.15", 10614, 0.92, 0.89, 886, "两个池同时满"),
    ("4K", "ratio 0.15 ＋ mem-frac 0.88", 10704, 0.92, 0.67, 999, "多装 13% 的人，只快 0.8%"),
    ("8K", "ratio 0.10", 9354, 0.78, 0.96, 588, "滑窗池先满"),
    ("8K", "ratio 0.15", 9053, 0.80, 0.37, 410, "两个池都没满"),
]
UP54 = RUNS[1][2] / RUNS[0][2]
assert "%.0f" % (100 * (UP54 - 1)) == "54"
UPMF = RUNS[2][2] / RUNS[1][2]
assert "%.1f" % (100 * (UPMF - 1)) == "0.8"
assert "%.0f" % (100 * (RUNS[2][5] / RUNS[1][5] - 1)) == "13"

# ══ ④ 搬回来 vs 重算（每芯片口径） ════════════════════════════════════
MFU = 0.5                                         # 假设：prefill 跑到 FP8 峰值的一半
PU_ = V7_FP8 * MFU
assert PU_ == 2307e12
F_LIN = 2 * V3_ACTIVE                             # 每个 token 过一遍激活参数：2 × 37e9 FLOP
F_ATT = 2 * 128 * (192 + 128) * V3_LAYERS         # 每一对 (q, k)：QK 用 192 维、PV 用 128 维，128 头
assert F_ATT == 4997120


def t_re(L):  # 重算长度 L 的前缀：线性部分 ＋ 因果注意力 L²/2 对
    return (F_LIN * L + F_ATT * L * L / 2) / PU_


def t_ld(L, bw):
    return KV_MLA * L / bw


BSTAR = KV_MLA * PU_ / F_LIN                      # 短前缀时的盈亏带宽（忽略注意力项 → 偏保守）
assert "%.2f" % (BSTAR / GB) == "2.19"
BSTAR_GQA = KV_GQA * PU_ / (2 * 70.6e9)
assert "%.2f" % (BSTAR_GQA / GB) == "5.35"
SLOW, FAST = 1 * GB, 20 * GB
LSTAR = (KV_MLA * PU_ / SLOW - F_LIN) / (F_ATT / 2)
assert 35000 < LSTAR < 35500, LSTAR
assert abs(t_re(LSTAR) - t_ld(LSTAR, SLOW)) < 1e-9


# ══════════════════════════════════════════════════════════════════════
def fig_batch():
    f = Fig(W, "一颗 TPU v7 芯片可用 189.5 GiB：V3 的八分之一 FP8 权重 78.1 GiB（83.9 GB），留 10%，剩 92.4 GiB 放 KV。"
               "上下文从 1K 到 128K，四种注意力能同时装的序列数：MLA 在 4K、32K、128K 分别是 344、43、10 条；"
               "GQA 73、9、2 条；线性加全注意力混合 315、83、23 条；MHA 到 32K 一条都装不下")
    y0 = f.header("上下文长 8 倍，纯注意力少装 8 倍，线性混合只少不到 4 倍　——　"
                  "<tspan font-weight=\"700\">KV 的记账方式决定 batch 上限</tspan>",
                  "一颗 v7 芯片，V3 权重铺 8 颗芯片后剩下的空间全给 KV。横轴上下文长度，纵轴能同时装下几条序列（对数，向下取整）",
                  [(m[3], m[0]) for m in MODELS])

    # 顶部：192 GiB 怎么分
    py = f.panel(0, y0, W, 104, "一颗芯片可用的 189.5 GiB", GY)
    X0, SPAN = 40, W - 80
    sc = SPAN / USABLE
    segs = [(W_CHIP, BL, "V3 权重的 1/8（FP8）：78.1 GiB"),
            (RESERVE, GY2, "留 10%：19.0 GiB"),
            (POOL, GR, "KV 池：92.4 GiB")]
    x = X0
    for v, col, lab in segs:
        w = v * sc
        f.box(x, py + 20, w, 40, col, col, 3)
        f.t(x + w / 2, py + 46, lab, "#fff", True, 15, "middle")
        x += w

    # 主图
    PH = 470
    py = f.panel(0, y0 + 104 + 16, W, PH, "能同时装几条序列", GR)
    AX0, AX1 = 130, 1010
    YT, YB = py + 30, py + PH - 80
    LO, HI = 1.0, 2000.0

    def xf(L):
        return AX0 + (AX1 - AX0) * log2(L) / 7.0

    def yf(n):
        return YB - (YB - YT) * (log10(n) - log10(LO)) / (log10(HI) - log10(LO))

    f.line(AX0, YB, AX1 + 10, YB, GY2, 1.2, arrow=False)
    f.line(AX0, YB, AX0, YT - 6, GY2, 1.2, arrow=False)
    for n in (1, 10, 100, 1000):
        f.line(AX0, yf(n), AX1, yf(n), LINE, 1, dash="3 4", arrow=False)
        f.t(AX0 - 12, yf(n) + 5, "%d" % n, GY, size=13, anchor="end")
    for L in (1, 2, 4, 8, 16, 32, 64, 128):
        f.t(xf(L), YB + 24, "%dK" % L, GY, L in (4, 32, 128), 14, "middle")
    for L in (4, 32, 128):
        f.line(xf(L), YT, xf(L), YB, GY2, 1, dash="4 4", arrow=False)
    # x 轴标题：居中于绘图区（横轴线从 AX0 画到 AX1＋10），比原先上移 6px，离开面板下沿
    f.t((AX0 + AX1 + 10) / 2, YB + 44, "上下文长度（对数，每格翻倍；1K ＝ 1024 token）", GY, size=14, anchor="middle")
    f.t(AX0 - 12, YT - 12, "条", GY, size=13, anchor="end")

    LABS = {  # 在 4K / 32K 上写数的上下偏移：{L: dy}；128K 的数并进右侧标签
        "MLA · DeepSeek-V3": {4: -12, 32: 26},
        "线性＋全注意力 · Qwen3.5 形状": {4: 26, 32: -12},
        "GQA · Llama-3.1-70B": {4: 26, 32: 26},
    }
    SUB = {"MLA · DeepSeek-V3": "70,272 B/token",
           "线性＋全注意力 · Qwen3.5 形状": "30,720 B/token ＋ 180 MiB/条",
           "GQA · Llama-3.1-70B": "327,680 B/token"}
    for name, per, fixed, col in MODELS:
        pts = [(L, TAB[name][L]) for L in (1, 2, 4, 8, 16, 32, 64, 128)]
        good = [(L, n) for L, n in pts if n >= 1]
        d = "M " + " L ".join("%.1f %.1f" % (xf(L), yf(n)) for L, n in good)
        f.path(d, col, 3, arrow=False)
        for L, n in good:
            f.p.append('<circle cx="%.1f" cy="%.1f" r="4.5" fill="%s"/>' % (xf(L), yf(n), col))
        if name in LABS:
            for L, dy in LABS[name].items():
                f.t(xf(L) + 10, yf(TAB[name][L]) + dy, "%d" % TAB[name][L], col, True, 15)
            f.t(AX1 + 20, yf(TAB[name][128]) + 5, "%d 条　%s" % (TAB[name][128], name.split(" · ")[0]), col, True, 15)
            f.t(AX1 + 20, yf(TAB[name][128]) + 24, SUB[name], GY, size=13)
        else:   # MHA：16K 之后一条都装不下
            Lx = 32
            xx, yy = xf(Lx), YB - 2
            f.line(xx - 8, yy - 8, xx + 8, yy + 8, col, 3, arrow=False)
            f.line(xx - 8, yy + 8, xx + 8, yy - 8, col, 3, arrow=False)
            f.t(xx + 14, yy - 10, "MHA：32K 起一条都装不下", col, True, 14)
            f.t(xf(1) + 10, yf(24) - 12, "24", col, True, 15)
            f.t(xf(4) + 10, yf(6) - 10, "6", col, True, 15)
    yb = y0 + 104 + 16 + PH + 20
    yb = f.band(yb, "ok", "装得下几条序列，每趟搬运就分给几个人", [
        "MLA 在 4K 装 344 条，32K 只剩 43 条。线性层每条序列先交 180 MiB 固定状态：",
        "4K 时反而比 MLA 少装（315 &lt; 344），128K 时多装一倍还多（23 &gt; 10）。",
    ])
    yb = f.src(yb + 10,
               "出处：每 token KV：V3 (512+64)×61×2 B ＝ 70,272（config.json）；Llama-3.1-70B 2×8×128×80×2 ＝ 327,680（HF config.json）；"
               "Qwen3.5 全注意力 15 层 × 2 头 × 256 维 × K、V × 2 B ＝ 30,720，线性层状态 45×64×128×128×4 B ＝ 180 MiB（HF config.json，mamba_ssm_dtype float32）。",
               "口径：MHA 一行是 V3 形状的假想（128 头 × 128 维，专题三第一题）：3,997,696 B/token。",
               "本课推导：V3 权重全按 FP8 1 B/参数算 671 GB ÷ 8 芯片 ＝ 83.9 GB ＝ 78.1 GiB；每芯片可用 189.5 GiB（每 device 实测 94.75 GiB × 2，tpu-inference R1 FP4 README），再留 10% 给激活（本课假设）；"
               "KV 池 ＝ 189.5 GiB × 0.9 − 78.1 GiB ＝ 92.4 GiB，与第十节 10.5 同一口径。先按权重全切粗算，DP attention 下非专家权重每个 device 各一份，实际更少（10.5）。假设 attention 走数据并行（每颗芯片只存自己那几条序列）；若 attention 走 TP，MLA 的 KV 每颗芯片复制一份（专题五）。",
               "口径：没算激活与通信缓冲（归入那 10%）、线性层卷积状态、引擎实际的状态 dtype（可能不是 fp32）。")
    f.save("fig6-kv-batch.svg", yb + 14)


# ══════════════════════════════════════════════════════════════════════
def fig_paged():
    # 物理块编号是示意：挑成「两张页表各自单调递增」，连线只在两表之间交叉，不穿任何字
    TBL = {"A": [1, 3, 4, 6], "B": [1, 3, 7]}
    f = Fig(W, "PagedAttention 示意：两条序列共用 prompt「请写一首关于秋天的诗」，每块 4 个 token。"
               "左边是每条序列眼里连续的 KV，中间是两张页表，右边是显存里零散的物理块。"
               "A 的页表是 1、3、4、6，B 的页表是 1、3、7。物理块 1 和 3 被两条序列共享；"
               "第三块原本共享，B 要往里写，先把它复制到块 7 再写，A 留在块 4，即写时复制")
    y0 = f.header("页表让 KV 不必连续存放　——　"
                  "<tspan font-weight=\"700\">共用的前缀只存一份，谁要改谁复制</tspan>",
                  "两条采样共用同一个 prompt。每块 4 个 token（仅为示意）",
                  [(GR, "两条共用"), (BL, "只属于 A"), (OR, "只属于 B"), (GY2, "空闲块")])
    PH = 450
    LW = 540
    py = f.panel(0, y0, LW, PH, "逻辑上：每条序列以为自己是连续的", GY)
    PX0 = 780
    f.panel(PX0, y0, W - PX0, PH, "物理上：显存里零散的块", GY)
    SW_, SH_ = 30, 38
    BW_ = 4 * SW_

    def block(x, y, toks, col):
        for i in range(4):
            tk = toks[i] if i < len(toks) else ""
            f.box(x + i * SW_, y, SW_, SH_, "#fff" if tk else "none", col if tk else GY2, 2,
                  1.6 if tk else 1, dash=None if tk else "3 3")
            if tk:
                f.t(x + i * SW_ + SW_ / 2, y + 25, tk, INK, False, 15, "middle")

    SEQ = {
        "A": (["请写一首", "关于秋天", "的诗：落", "叶"], [GR, GR, BL, BL], py + 66),
        "B": (["请写一首", "关于秋天", "的诗。枫"], [GR, GR, OR], py + 276),
    }
    LX, BG = 18, 10
    CX, CW, CH, CS = 572, 160, 32, 42          # 页表卡片：左缘、宽、每格高、行距
    ends = []
    for name, (blks, cols, y) in SEQ.items():
        f.t(LX, y - 16, "序列 %s" % name, INK, True, 16)
        for i, (tk, col) in enumerate(zip(blks, cols)):
            x = LX + i * (BW_ + BG)
            block(x, y, list(tk), col)
            f.t(x + BW_ / 2, y + SH_ + 24, "第 %d 块" % i, GY, size=14, anchor="middle")
        # 页表卡片：一格一行「第 i 块 → 物理块」
        f.t(CX + CW / 2, y - 16, "%s 的页表" % name, INK, True, 16, "middle")
        for i, (pb, col) in enumerate(zip(TBL[name], cols)):
            cy = y + i * CS
            f.box(CX, cy, CW, CH, "#fff", col, 4, 1.6)
            f.t(CX + CW / 2, cy + 22, "第 %d 块 → %d" % (i, pb), col, True, 15, "middle")
            ends.append((CX + CW, cy + CH / 2, pb, col))

    # 物理块：8 行
    PX, PY0, RH = 1000, py + 20, 50
    PHYS = {1: ("请写一首", GR, "两条共用"), 3: ("关于秋天", GR, "两条共用"),
            4: ("的诗：落", BL, "A"), 7: ("的诗。枫", OR, "B 的拷贝"),
            6: ("叶", BL, "A · 没写满")}
    rowy = {}
    for k in range(8):
        y = PY0 + k * RH
        rowy[k] = y + SH_ / 2
        f.t(PX - 14, y + 25, "块 %d" % k, GY, True, 15, "end")
        if k in PHYS:
            tk, col, note = PHYS[k]
            block(PX, y, list(tk), col)
            f.t(PX + BW_ + 16, y + 25, note, col, True, 15)
        else:
            block(PX, y, [], GY2)
            f.t(PX + BW_ + 16, y + 25, "空闲", GY2, size=15)
    for x1, y1, pb, col in ends:
        x2, y2 = PX - 64, rowy[pb]
        f.path("M %.1f %.1f C %.1f %.1f %.1f %.1f %.1f %.1f"
               % (x1 + 2, y1, x1 + 110, y1, x2 - 110, y2, x2, y2), col, 2, arrow=True)
    # 写时复制：块 4 → 块 7
    xa = PX + BW_ + 160
    f.path("M %.1f %.1f C %.1f %.1f %.1f %.1f %.1f %.1f"
           % (PX + BW_ + 50, rowy[4], xa, rowy[4], xa, rowy[7], PX + BW_ + 110, rowy[7]),
           OR, 1.8, dash="5 4", arrow=True)
    f.t(xa + 2, (rowy[4] + rowy[7]) / 2 - 6, "B 要往里写，", OR, True, 14)
    f.t(xa + 2, (rowy[4] + rowy[7]) / 2 + 14, "先复制一份", OR, True, 14)
    f._pan = None
    yb = f.src(y0 + PH + 10,
               "出处：Kwon et al., Efficient Memory Management for LLM Serving with PagedAttention（SOSP'23, arXiv 2309.06180）—— "
               "块级写时复制、并行采样与 beam search 共享块；原文统计既有系统 KV 显存里真正存 token 状态的只有 20.4%–38.2%，vLLM 达 96.3%。",
               "口径：每块 4 个 token 只为画得下；真实系统常用 16（GPU）或 256（TPU 上这个模型）。物理块编号是示意。")
    f.save("fig6-paged.svg", yb + 14)


# ══════════════════════════════════════════════════════════════════════
def fig_pool():
    f = Fig(W, "V4-Pro 的两个 KV 池画成一个仓位被隔板分成两格，仓位就是那份显存预算，ratio 决定隔板放哪。"
               "4K 输入、ratio 0.20：全长池用到 93% 先满，滑窗池只用 63%，每张 decode GPU 6,887 tok/s。"
               "ratio 0.15：两个池都用到九成左右，10,614 tok/s，高 54%。"
               "换成 8K 输入、ratio 0.10：反过来滑窗池先满（96%），全长池 78%，9,354 tok/s")
    y0 = f.header("一份预算一块隔板　——　"
                  "<tspan font-weight=\"700\">隔板挪到两边一起满，吞吐高 54%</tspan>",
                  "V4-Pro 每层两个 KV 池，共用一份显存预算：滑窗池只留最近 128 个 token，全长池存压缩后的整段。"
                  "ratio ＝ 滑窗池 ÷ 全长池，就是隔板的位置",
                  [(BL, "已用"), (RD, "约九成以上，算满")])
    STATES = [RUNS[0], RUNS[1], RUNS[3]]
    TITLES = ["全长池先满", "两个一起快满", "滑窗池先满"]
    PH, PWD, GAP = 470, 430, 25
    xs = [0, PWD + GAP, 2 * (PWD + GAP) + 15]
    TKW, TKH = 120, 170
    bottoms = []
    for k, ((isl, cfg, tps, full, swa, req, verdict), ttl, px) in enumerate(zip(STATES, TITLES, xs)):
        py = f.panel(px, y0, PWD, PH, "输入 %s · %s" % (isl, ttl), BL if isl == "4K" else OR)
        cx = px + PWD / 2
        # 一份显存预算 ＝ 一个仓位，中间一块隔板把它分成两个池；ratio 决定隔板放哪
        lx = cx - TKW - 6
        rx = cx + 6
        ty = py + 100
        f.path("M %.1f %.1f V %.1f H %.1f V %.1f" % (lx - 2, ty - 26, ty - 34, rx + TKW + 2, ty - 26), GY2, 2, arrow=False)
        f.t(cx, ty - 44, "一份显存预算", GY, size=14, anchor="middle")
        f.box(cx - 5, ty - 14, 10, TKH + 14, INK, "none", 2)
        f.t(cx, ty + TKH + 60, "隔板 " + cfg, INK, True, 16, "middle")
        for tx, occ, nm in ((lx, full, "全长池"), (rx, swa, "滑窗池")):
            hot = occ >= 0.885
            c = RD if hot else BL
            hw = TKH * occ
            f.box(tx, ty + TKH - hw, TKW, hw, "#fce8e6" if hot else "#e8f0fe", "none", 3)
            f.line(tx + 2, ty + TKH - hw, tx + TKW - 2, ty + TKH - hw, c, 2.5, arrow=False)
            f.path("M %.1f %.1f V %.1f H %.1f V %.1f" % (tx, ty, ty + TKH, tx + TKW, ty), GY, 2.5, arrow=False)
            f.t(tx + TKW / 2, ty + TKH - hw / 2 + 8, "%d%%" % round(occ * 100), c, True, 22, "middle")
            f.t(tx + TKW / 2, ty + TKH + 26, nm, INK, True, 16, "middle")
            if hot:
                f.t(tx + TKW / 2, ty - 8, "满", RD, True, 17, "middle")
        f.t(cx, py + PH - 70, format(tps, ","), BL if isl == "4K" else OR, True, 34, "middle", big=True)
        f.t(cx, py + PH - 42, "tok/s（每张 decode GPU）", GY, size=14, anchor="middle")
        bottoms.append(py + PH - 70)
    # +54%：从第一格的吞吐指到第二格
    ya = bottoms[0] - 12
    x1, x2 = xs[0] + PWD / 2 + 70, xs[1] + PWD / 2 - 70
    f.line(x1, ya, x2, ya, RD, 2, arrow=True)
    f.t(xs[1] + 70, ya - 10, "+54%", RD, True, 20, "middle")
    f._pan = None
    yb = f.src(y0 + PH + 10,
               "出处：gpu/inference/a4x-max/deepseek-v4/SGLANG-V4PRO-RUNBOOK.md §11.5 消融表（约 L849–866）与结论（L872–896）：ISL 4096／8192、OSL 1024、--disable-radix-cache；"
               "吞吐是每张 decode GPU 的 e2e 输出 tok/s；占用是 full／swa 两池峰值；每 rank 请求数 727／886／999／588／410。",
               "出处：swa_tokens ＝ full_tokens × ratio（runbook L946 引 pool_configurator.py:387）；V4 是 CSA＋HCA 混合注意力，不是 MLA；sliding_window 128 取自 HF config.json。",
               "本课推导：+54% ＝ 10,614 ÷ 6,887。图里只画三个状态；同表另两行：4K · ratio 0.15 ＋ mem-frac 0.88 为 10,704（多装 13% 的人，只快 0.8%），"
               "8K · ratio 0.15 两池都没满（0.80／0.37）、每 rank 只有 410 条，runbook 没有解释原因，这里照录不推断。",
               "口径：8K 那一格输入长度不同，吞吐不和左边两格直接比；两格画得一样宽是示意，不代表两池的真实字节数。")
    f.save("fig6-swa-ratio.svg", yb + 14)


# ══════════════════════════════════════════════════════════════════════
def fig_offload():
    f = Fig(W, "把存下来的 V3 KV 搬回来，和重新 prefill 一遍，各要多久：重算的时间随前缀长度先线性后平方增长；"
               "搬回的时间按链路带宽线性增长。链路 20 GB/s 时搬回始终更快；1 GB/s 时前缀约 35K token 以上搬回才更快；"
               "盈亏带宽约 2.19 GB/s：这条线在短前缀处和重算几乎重合，越往右越比重算快")
    y0 = f.header("搬回来还是重算？比的是一个字节的 KV 值多少 FLOP　——　"
                  "<tspan font-weight=\"700\">V3 的 KV 很省，2.19 GB/s 就够</tspan>",
                  "横轴：命中的前缀有多长；纵轴：把这段前缀的 KV 准备好要多久（都是对数）。每芯片口径，DeepSeek-V3",
                  [(RD, "重算（prefill 一遍）"), (GR, "从 20 GB/s 的地方搬回"), (GY, "从 2.19 GB/s 的地方搬回（盈亏线）"),
                   (OR, "从 1 GB/s 的地方搬回")])
    PH = 540
    py = f.panel(0, y0, W, PH, "准备好一段前缀的 KV 要多久", GY)
    AX0, AX1 = 140, 1060
    YT, YB = py + 30, py + PH - 80
    XLO, XHI = 256, 131072
    TLO, THI = 5e-4, 50.0

    def xf(L):
        return AX0 + (AX1 - AX0) * (log10(L) - log10(XLO)) / (log10(XHI) - log10(XLO))

    def yf(t):
        return YB - (YB - YT) * (log10(t) - log10(TLO)) / (log10(THI) - log10(TLO))

    f.line(AX0, YB, AX1 + 10, YB, GY2, 1.2, arrow=False)
    f.line(AX0, YB, AX0, YT - 6, GY2, 1.2, arrow=False)
    for t, lab in ((1e-3, "1 ms"), (1e-2, "10 ms"), (1e-1, "100 ms"), (1, "1 s"), (10, "10 s")):
        f.line(AX0, yf(t), AX1, yf(t), LINE, 1, dash="3 4", arrow=False)
        f.t(AX0 - 12, yf(t) + 5, lab, GY, size=14, anchor="end")
    for L, lab in ((256, "256"), (1024, "1K"), (4096, "4K"), (16384, "16K"), (65536, "64K"), (131072, "128K")):
        f.t(xf(L), YB + 24, lab, GY, size=15, anchor="middle")
    f.t((AX0 + AX1) / 2, YB + 52, "命中的前缀长度（token，对数）", GY, size=15, anchor="middle")

    Ls = [XLO * (XHI / XLO) ** (i / 60.0) for i in range(61)]
    for fn, col, lab, dash in ((t_re, RD, "重算", None),
                               (lambda L: t_ld(L, FAST), GR, "20 GB/s 搬回", None),
                               (lambda L: t_ld(L, BSTAR), GY, "2.19 GB/s 搬回", "8 5"),
                               (lambda L: t_ld(L, SLOW), OR, "1 GB/s 搬回", None)):
        d = "M " + " L ".join("%.1f %.1f" % (xf(L), yf(fn(L))) for L in Ls)
        f.path(d, col, 3, dash=dash, arrow=False)
        f.t(AX1 + 18, yf(fn(XHI)) + 5, lab, col, True, 16)
    # 交点
    xs, ys = xf(LSTAR), yf(t_re(LSTAR))
    f.p.append('<circle cx="%.1f" cy="%.1f" r="7" fill="none" stroke="%s" stroke-width="2.5"/>' % (xs, ys, INK))
    f.line(xs, ys + 8, xs, YB, INK, 1, dash="4 4", arrow=False)
    f.t(xs - 14, ys - 42, "交点 ≈ 35K token", INK, True, 16, "end")
    f.t(xs - 14, ys - 20, "短于它，1 GB/s 不如重算", GY, size=15, anchor="end")
    f.t(AX0 + 12, yf(t_ld(XLO, BSTAR)) + 28, "短前缀：两条重合", RD, True, 15)
    f.t(xs + 18, ys + 46, "再往右重算变陡：", RD, size=15)
    f.t(xs + 18, ys + 68, "注意力的平方项接管", RD, size=15)
    f._pan = None
    yb = f.src(y0 + PH + 10,
               "本课推导：重算 ＝ (2×37e9×L ＋ 4,997,120×L²/2) ÷ (4,614 T × 50%)；注意力每对 (q,k) ＝ 2×128 头×(192＋128)×61 层 ＝ 4,997,120 FLOP。"
               "搬回 ＝ 70,272 B × L ÷ 带宽。盈亏带宽 ＝ 70,272 × 2,307 T ÷ 74 G ≈ 2.19 GB/s（只按线性项算，偏保守）；"
               "GQA 的 Llama-3.1-70B 同法得 5.35 GB/s。",
               "假设：prefill 跑到 FP8 峰值的 50%；1 GB/s 与 20 GB/s 是示意档位，不对应具体硬件；只比带宽与算力，不含延迟、排队和与 decode 抢带宽。"
               "每芯片口径 ＝ 假设算力和 KV 字节按芯片数等比摊开。",
               "出处：Llama-3.1-70B 激活参数 70.6B 取自 HF 模型页；KV 327,680 B/token 取自 config.json。")
    f.save("fig6-offload.svg", yb + 14)


# ══════════════════════════════════════════════════════════════════════
# fig6-disk-cache：DeepSeek 磁盘缓存的实测案例。
#   首字 13 s → 0.5 s：api-docs.deepseek.com/news/news0802（2024-08，128K 高复用 prompt）。
#   6,080 亿 / 56.3%：open-infra-index 202502OpenSourceWeek day_6（24 小时统计 2025-02-27～28，608B 输入里 342B 命中）。
#   ⛔ 两个数出自两份文档，别合并成一个出处；⛔ 不画价格。
T_MISS, T_HIT = 13.0, 0.5
assert T_MISS / T_HIT == 26
IN_TOK, HIT_TOK, HIT_PCT = 608e9, 342e9, 56.3     # 56.3 是原文给的百分比（342 ÷ 608 ＝ 56.25，原文进位到 56.3）
assert abs(HIT_TOK / IN_TOK * 100 - HIT_PCT) < 0.06


def fig_disk():
    f = Fig(W, "DeepSeek 磁盘缓存。128K 的高复用 prompt，首字延迟不命中要 13 秒，命中磁盘上的 KV 只要 0.5 秒，约二十六分之一。"
               "另一份统计：24 小时 6,080 亿输入 token 里 56.3% 命中磁盘缓存")
    y0 = f.header("命中的前缀不用重算　——　<tspan font-weight=\"700\">首字只剩读盘那一段</tspan>",
                  "DeepSeek 线上服务，把算好的 KV 存在磁盘上。上：一条 128K 高复用 prompt 的首字延迟（线性，0–14 s）；下：一天的输入里有多少命中",
                  [(GY2, "不命中：prefill 重算整段前缀"), (GR, "命中：从磁盘读回 KV")])
    PH = 210
    py = f.panel(0, y0, W, PH, "128K prompt 的首字延迟", GY)
    AX0, AX1 = 220, 1300
    sc = (AX1 - AX0) / 14.0
    yb1, yb2, BH = py + 30, py + 90, 38
    for s_ in range(0, 15, 2):
        xx = AX0 + s_ * sc
        f.line(xx, yb1 - 8, xx, yb2 + BH + 8, LINE, 1, dash="3 4", arrow=False)
        f.t(xx, yb2 + BH + 32, "%d s" % s_, GY, size=14, anchor="middle")
    f.t(AX0 - 16, yb1 + 25, "不命中", INK, True, 16, "end")
    f.t(AX0 - 16, yb2 + 25, "命中", INK, True, 16, "end")
    f.box(AX0, yb1, T_MISS * sc, BH, GY2, GY2, 3)
    f.t(AX0 + 16, yb1 + 25, "重算 128K 前缀", "#fff", True, 15)
    f.t(AX0 + T_MISS * sc + 10, yb1 + 25, "13 s", INK, True, 16)
    f.box(AX0, yb2, T_HIT * sc, BH, GR, GR, 3)
    f.t(AX0 + T_HIT * sc + 10, yb2 + 25, "0.5 s　读盘", GR, True, 16)
    f.t(AX0 + T_HIT * sc + 160, yb2 + 25, "≈ 1/26（本课推导：13 ÷ 0.5）", GR, True, 15)

    PH2 = 130
    py = f.panel(0, y0 + PH + 16, W, PH2, "24 小时的输入 token（另一份统计，2025-02）", GY)
    BW_ = AX1 - AX0
    hit = HIT_PCT / 100
    by = py + 26
    f.t(AX0 - 16, by + 25, "6,080 亿", INK, True, 16, "end")
    f.box(AX0, by, BW_ * hit, BH, GR, GR, 3)
    f.box(AX0 + BW_ * hit, by, BW_ * (1 - hit), BH, "#e8eaed", GY2, 3)
    f.t(AX0 + 16, by + 25, "56.3% 命中磁盘上的 KV", "#fff", True, 15)
    f.t(AX0 + BW_ * hit + 16, by + 25, "43.7% 不命中（本课推导：100 − 56.3）", INK, True, 15)
    f.t(AX0, by + BH + 28, "0%", GY, size=14, anchor="middle")
    f.t(AX1, by + BH + 28, "100%", GY, size=14, anchor="middle")
    yb = f.src(y0 + PH + 16 + PH2 + 10,
               "出处：首字 13 s → 0.5 s 取自 DeepSeek API 文档的磁盘缓存公告（api-docs.deepseek.com/news/news0802，2024-08，原文 128K prompt with high reference）。",
               "出处：6,080 亿输入 token、56.3% 命中取自 DeepSeek open-infra-index 202502OpenSourceWeek day_6（统计窗口 2025-02-27 12:00 到 02-28 12:00 UTC+8，608B 里 342B 命中，V3 与 R1 合计）。",
               "本课推导：13 ÷ 0.5 ＝ 26；100% − 56.3% ＝ 43.7%。两个数来自两份文档、两个时间点，不是同一次测量。价格不画。")
    f.save("fig6-disk-cache.svg", yb + 14)


fig_batch()
fig_paged()
fig_pool()
fig_offload()
fig_disk()
