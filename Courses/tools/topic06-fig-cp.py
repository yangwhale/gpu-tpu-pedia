# -*- coding: utf-8 -*-
r"""专题六 · 第七节「切序列：DCP 和 PCP」的两张静态图。

回答的问题
  · fig6-dcp-gain   DCP 什么时候赚、什么时候亏？（输入长度 × 并发 → DCP8 对 TP8 的输出吞吐比）
  · fig6-pcp-linear 线性注意力（GDN／KDA 这类带状态的层）为什么在 PCP 下会静默算错？

数从哪来
  · K3 on GB300（SGLang）消融：gpu/inference/a4x-max/kimi-k3/SGLANG-K3-RUNBOOK.md
      §11.0.3（:1010-1023）八个工作点「TP8 最优 vs DCP8」；
      §11.0.0 宽表（:921-938）补 4K c8／c32 两点：TP8 取 symm-mem 开着的三档 ratio 里最好的
      （c8：434／434／437 → 437；c32：1,130／1,139／1,093 → 1,139），DCP8 为 397／1,019；
      逻辑 KV 容量 1.5M → 12.2M token（:721-723）；DCP 的 −26% 等于关 symm-mem 的损失（:1024-1026）。
    吞吐都是输出 tok/s（不含输入），热轮，TP8 跨 2 节点，OSL 1024。
  · fig6-pcp-linear 是玩具递推 S_t ＝ 0.9·S_{t−1} ＋ 1（本课自造，不对应任何真实模型的数值），本脚本现算。

刻意没画
  · fig6-dcp-gain 不画 TTFT；128K c8 的首字延迟 32.1 s → 2.3 s 放在正文。
  · 32K c32 的 TP8 最优按 §11.0.3 取 408.2；宽表里 ratio 0.4 那一行是 420，若取它，比值是 1.04 而不是 1.07，不改结论。
  · fig6-pcp-linear 不画 softmax 注意力的 PCP（它没有跨段状态，专题五 fig5-pcp-dcp 已画过）。
"""
import math
from topic03_draw import Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE

W = 1400

# ── DCP8 / TP8 工作点：(ISL, 并发, TP8 最优, DCP8) ─────────────────────────
PTS = [("4K", 1, 87.8, 64.7), ("4K", 8, 437, 397), ("4K", 32, 1139, 1019), ("4K", 64, 1511, 1409),
       ("8K", 8, 451.7, 400.0), ("8K", 32, 893.2, 1109.9),
       ("32K", 8, 408.3, 337.0), ("32K", 32, 408.2, 436.5),
       ("128K", 1, 56.6, 63.6), ("128K", 8, 108.3, 365.2)]
RATIO = {(isl, c): d / t for isl, c, t, d in PTS}
PCT = {k: round((v - 1) * 100) for k, v in RATIO.items()}
assert PCT == {("4K", 1): -26, ("4K", 8): -9, ("4K", 32): -11, ("4K", 64): -7,
               ("8K", 8): -11, ("8K", 32): 24, ("32K", 8): -17, ("32K", 32): 7,
               ("128K", 1): 12, ("128K", 8): 237}, PCT
assert round(436.5 / 420 - 1, 2) == 0.04
KV_TP, KV_DCP = 1.5, 12.2          # 百万 token，runbook :721
assert round(KV_DCP / KV_TP, 1) == 8.1      # runbook 写「约 7.9×」是用未取整的数，这里只是量级核对
ISLS = ["4K", "8K", "32K", "128K"]
CONCS = [(1, PU), (8, BL), (32, OR), (64, GY)]

# ── 玩具线性注意力：S_t = A·S_{t-1} + 1，16 个 token，4 个 rank ──────────────
A, T, R = 0.9, 16, 4
L = T // R


def scan(n, s0=0.0):
    out, s = [], s0
    for _ in range(n):
        s = A * s + 1
        out.append(s)
    return out


TRUE = scan(T)
WRONG = scan(L) * R                # 每个 rank 从 0 开始扫自己那 4 个 token
ENDS = [TRUE[(r + 1) * L - 1] for r in range(R)]
assert [round(e, 3) for e in ENDS] == [3.439, 5.695, 7.176, 8.147], ENDS
assert round(WRONG[-1], 3) == 3.439 and round(1 - WRONG[-1] / TRUE[-1], 2) == 0.58
# 修法：把上一段的末状态接进来。仿射映射 s ↦ a·s ＋ b 可结合，各段先算 (a, b)，再做前缀合并
SEG = (A ** L, scan(L)[-1])        # 一段 4 个 token 合成一个映射
fixed, s = [], 0.0
for _ in range(R):
    s = SEG[0] * s + SEG[1]
    fixed.append(s)
assert all(abs(x - y) < 1e-12 for x, y in zip(fixed, ENDS))


def fig_dcp_gain():
    f = Fig(W, "Kimi K3 在 GB300 上用 SGLang，比较 TP8 和 TP8 加 DCP8 的输出吞吐比值，纵轴是对数刻度，1 倍是盈亏线。"
               "输入 4K 时四档并发 DCP 都亏，亏 7% 到 26%。突出画的是并发 8 那条线：4K、8K、32K 都在盈亏线下面，"
               "到 128K 一下跳到 365 对 108，多 237%。其余并发画成淡色：8K 并发 32 赚 24%，32K 并发 32 赚 7%，128K 并发 1 赚 12%")
    FAINT = {1: "#cdb8ef", 8: BL, 32: "#f3c79e", 64: "#c3c6cb"}
    y0 = f.header("DCP 只在长上下文赚　——　<tspan font-weight=\"700\">128K、8 并发：365 对 108，+237%</tspan>",
                  "Kimi K3 · GB300 · SGLang · TP8 跨 2 节点，输出 1K。纵轴：DCP8 ÷ TP8 的输出吞吐（对数刻度），一条线一个并发",
                  [(BL, "并发 8"), (FAINT[1], "并发 1"), (FAINT[32], "并发 32"), (FAINT[64], "并发 64")])
    PH = 470
    py = f.panel(0, y0, W, PH, "输入长度 → DCP8 相对 TP8", BL)
    X0, PW, PY, PHH = 110, 1240, py + 34, PH - 124
    LO, HI = math.log(0.6), math.log(4.0)

    def px(isl):
        return X0 + 200 + (PW - 480) * ISLS.index(isl) / 3

    def py_(v):
        return PY + PHH - PHH * (math.log(v) - LO) / (HI - LO)
    f.box(X0, PY, PW, py_(1.0) - PY, "#e6f4ea", "none", 0)
    f.box(X0, py_(1.0), PW, PY + PHH - py_(1.0), "#fce8e6", "none", 0)
    for v, lab in ((0.6, "0.6×"), (0.8, "0.8×"), (1.0, "1×"), (1.5, "1.5×"), (2, "2×"), (3, "3×"), (4, "4×")):
        f.line(X0, py_(v), X0 + PW, py_(v), GY2 if v == 1 else "#eceff1", 1.8 if v == 1 else 1, None, arrow=False)
        f.t(X0 - 10, py_(v) + 5, lab, INK if v == 1 else GY, v == 1, 14, "end")
    f.t(X0 + 14, PY + 26, "DCP 赚", GR, True, 17)
    f.t(X0 + 14, PY + PHH - 14, "DCP 亏", RD, True, 17)
    for isl in ISLS:
        f.t(px(isl), PY + PHH + 24, isl, GY, size=15, anchor="middle")
    f.t(X0 + PW / 2, PY + PHH + 50, "输入长度（token）", GY, size=15, anchor="middle")
    # 先画淡色的三条，再画突出的并发 8
    for c, _ in sorted(CONCS, key=lambda t: t[0] == 8):
        col = FAINT[c]
        hot = c == 8
        pts = [(px(isl), py_(RATIO[(isl, c)])) for isl in ISLS if (isl, c) in RATIO]
        if len(pts) > 1:
            f.path(pts, col, 3.6 if hot else 2, arrow=False)
        for x, y in pts:
            r = 7 if hot else 4
            f.box(x - r, y - r, 2 * r, 2 * r, col, col, r)
    # 淡色线只标一个：同是 8K，并发一高就赚
    f.t(px("8K"), py_(RATIO[("8K", 32)]) - 14, "并发 32：+%d%%" % PCT[("8K", 32)], "#b06000", True, 15, "middle")
    # 并发 8 的三个亏点
    for isl, dy in (("8K", 30), ("32K", 30)):
        f.t(px(isl), py_(RATIO[(isl, 8)]) + dy, "−%d%%" % -PCT[(isl, 8)], BL, True, 16, "middle")
    # 4K：一个括号代替四个小标签
    lo = min(RATIO[("4K", c)] for c, _ in CONCS)
    hi = max(RATIO[("4K", c)] for c, _ in CONCS)
    bx = px("4K") - 26
    f.path([(bx + 8, py_(hi)), (bx, py_(hi)), (bx, py_(lo)), (bx + 8, py_(lo))], RD, 1.8, arrow=False)
    f.t(bx - 10, (py_(hi) + py_(lo)) / 2 - 4, "4K：四档并发全亏", RD, True, 16, "end")
    f.t(bx - 10, (py_(hi) + py_(lo)) / 2 + 20, "−7% ～ −26%", RD, True, 16, "end")
    # 128K 的跳变
    x8, y8 = px("128K"), py_(RATIO[("128K", 8)])
    f.t(x8 + 22, y8 + 6, "+%d%%" % PCT[("128K", 8)], BL, True, 34)
    f.t(x8 + 22, y8 + 34, "365 vs 108 tok/s", BL, True, 16)
    f._pan = None
    yb = y0 + PH + 10
    yb = f.src(yb,
               "gpu/inference/a4x-max/kimi-k3/SGLANG-K3-RUNBOOK.md §11.0.3（:1010-1023，八个工作点）、§11.0.0 宽表（:921-938，4K c8／c32 本脚本取 symm-mem 开着时 TP8 最好的一档现算）、"
               ":721-723（逻辑 KV 1.5M → 12.2M token）、:1024-1026（−26% ＝ 关 symm-mem 的损失）、:962（symm-mem 在 DCP 下用不了）。",
               "各点比值：4K c1 −26%、c8 −9%、c32 −11%、c64 −7%；8K c8 −11%、c32 +24%；32K c8 −17%、c32 +7%；128K c1 +12%、c8 +237%。",
               "吞吐是输出 tok/s，热轮；32K c32 按 §11.0.3 取 TP8 408.2（宽表 ratio 0.4 那行是 420，比值变 1.04，不改结论）。K3 的 MLA 层只有一个 KV 头，"
               "它的 69 层 KDA 状态池 DCP 不切（同文件 :725-726）。横轴没换成「KV 总量」：原表只给输入长度和并发，没有实测的 KV 占用。")
    f.save("fig6-dcp-gain.svg", yb + 14)


def fig_pcp_linear():
    f = Fig(W, "一个玩具线性注意力：状态每读一个 token 先乘 0.9 再加 1。16 个 token 一口气扫下来，状态平滑上升，"
               "4 段的段末分别是 3.439、5.695、7.176、8.147。PCP 把这 16 个 token 切给 4 个 rank，"
               "如果每个 rank 都从空状态开始扫自己那 4 个，每段段末都是 3.439，曲线变成锯齿，最后一段差 58%")
    y0 = f.header("线性注意力切序列，每段从零开始会静默算错　——　<tspan font-weight=\"700\">状态要接力，不能各扫各的</tspan>",
                  "玩具递推：状态 S ← 0.9 × S ＋ 1，从 0 起。16 个 token 切给 4 个 rank，每段 4 个（本课自造的数）",
                  [(GR, "一口气扫（对）"), (RD, "每个 rank 从 0 扫（错）")])
    PH = 440
    py = f.panel(0, y0, W, PH, "第几个 token → 状态 S", INK)
    X0, PW, PY, PHH = 90, W - 140, py + 34, PH - 120
    YMAX = 9

    def px(t):
        return X0 + PW * t / T

    def py_(v):
        return PY + PHH - PHH * v / YMAX
    for v in (0, 3, 6, 9):
        f.line(X0, py_(v), X0 + PW, py_(v), "#eceff1", 1, arrow=False)
        f.t(X0 - 10, py_(v) + 5, str(v), GY, size=14, anchor="end")
    for r in range(R):
        x0 = px(r * L)
        if r:
            f.line(x0, PY, x0, PY + PHH, GY2, 1.2, "5 4", arrow=False)
        f.t((px(r * L) + px((r + 1) * L)) / 2, PY + PHH + 24, "rank %d" % r, GY, True, 15, "middle")
    f.t(X0 + PW / 2, PY + PHH + 50, "token 1 → 16", GY, size=15, anchor="middle")
    f.path([(px(0), py_(0))] + [(px(t + 1), py_(v)) for t, v in enumerate(TRUE)], GR, 3, arrow=False)
    for r in range(R):
        seg = [(px(r * L), py_(0))] + [(px(r * L + i + 1), py_(WRONG[r * L + i])) for i in range(L)]
        f.path(seg, RD, 2.6, "6 4", arrow=False)
        e = ENDS[r]
        f.box(px((r + 1) * L) - 6, py_(e) - 6, 12, 12, GR, GR, 6)
        f.t(px((r + 1) * L) - 10, py_(e) - 14, "%.3f" % e, GR, True, 17, "end")
        f.box(px((r + 1) * L) - 6, py_(WRONG[(r + 1) * L - 1]) - 6, 12, 12, RD, RD, 6)
    f.t(px(L) - 10, py_(0) - 10, "又从 0 起步", RD, True, 15, "end")
    f.t(px(T) - 10, py_(WRONG[-1]) - 16, "%.3f（差 58%%）" % WRONG[-1], RD, True, 17, "end")
    f._pan = None
    yb = y0 + PH + 10
    yb = f.src(yb,
               "本课自造的玩具递推，数值由本脚本现算并断言；真实 GDN／KDA 的状态是矩阵、带门控，递推形状一样都是「乘上一步状态再加新东西」。",
               "状态可结合合并的思路见线性注意力／状态空间模型的并行扫描写法（如 Mamba、GLA 的 chunkwise 算法）。")
    f.save("fig6-pcp-linear.svg", yb + 14)


fig_dcp_gain()
fig_pcp_linear()
