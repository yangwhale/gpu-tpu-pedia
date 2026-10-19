# -*- coding: utf-8 -*-
r"""专题六 · fig6-seats.svg（2026-10-20 加，两档共用；当前 L200 用）。

回答：给 KV 池多分显存、多坐人，吞吐为什么几乎不涨？
  · 数字：同一组 GB300 × DeepSeek-V4-Pro 实测（SGLang），mem-fraction-static 0.85 → 0.88：
    每 rank 请求 886 → 999，TPOT 77.7 → 85.3 ms，端到端吞吐 10,614 → 10,704（＋0.8%）。
    出处同 L300 第三节正文与台账（SGLANG-V4PRO-RUNBOOK.md:849-858、885-894）。
  · 本课推导：Little 定律 吞吐 ∝ 人数 ÷ 两字间隔，新座位全坐满的上限 ＝ 999/886 × 77.7/85.3 ≈ 1.027（约 3%）。
  · 画法：只画「变了百分之几」，不画绝对值 —— 绝对值四根条几乎一样长，看不出差别。
  · 刻意没画：峰值吞吐 12,063 → 12,070（另一口径，L300 第三节那张图已有）。
"""
from topic03_draw import Fig, wpx, BL, OR, GR, RD, GY, INK, GY2

W = 1400
N0, N1 = 886, 999
T0, T1 = 77.7, 85.3
THR = 10704 / 10614 - 1
UP = N1 / N0 * T0 / T1 - 1
assert round(N1 / N0 - 1, 2) == 0.13 and round(THR * 100, 1) == 0.8
assert round(UP * 100) == 3, UP
ROWS = [("车上人数", "%d → %d" % (N0, N1), N1 / N0 - 1, BL, "+13%"),
        ("两字间隔", "%.1f → %.1f ms" % (T0, T1), T1 / T0 - 1, RD, "每个人都慢了"),
        ("吞吐上限", "Little 定律推算", UP, GY2, "约 +3%（本课推导）"),
        ("实测吞吐", "端到端", THR, GR, "+0.8%")]


def fig_seats():
    f = Fig(W, "同一组 GB300 跑 DeepSeek-V4-Pro 的实测，只把 KV 池多分 13% 显存。四根横条画的是各项比原来变了百分之几："
               "车上人数 886 到 999，多 13%；两字间隔 77.7 到 85.3 毫秒，每个人都慢了；"
               "按 Little 定律，吞吐约等于人数除以两字间隔，新座位全坐满最多多约 3%（本课推导）；实测端到端吞吐只多 0.8%")
    y0 = f.header("座位多给一成多　——　<tspan font-weight=\"700\">吞吐几乎没动，多出来的人换成了每一步更慢</tspan>",
                  "GB300 · DeepSeek-V4-Pro · 只多给 KV 池显存；条长＝比调整前变了百分之几",
                  [(BL, "人数"), (RD, "两字间隔"), (GY2, "推算上限"), (GR, "实测")])
    PH = 350
    py = f.panel(0, y0, W, PH, "调整后比调整前多了多少", INK)
    X0, SC = 300, 52.0 * 100          # 每 1% 画 52 px
    assert X0 + 0.135 * SC + 300 < W
    for j, (lab, sub, v, col, tag) in enumerate(ROWS):
        yy = py + 34 + j * 70
        f.t(24, yy + 24, lab, INK, True, 17)
        f.t(24, yy + 46, sub, GY, size=14)
        w = max(v * SC, 4)
        if col == GY2:
            f.box(X0, yy + 8, w, 36, "none", GY2, 4, 1.4, dash="5 4")
        else:
            f.box(X0, yy + 8, w, 36, col, col, 4)
        f.t(X0 + w + 14, yy + 33, tag, col, True, 16)
    yl = py + 34 + len(ROWS) * 70 - 10
    f.line(X0, py + 22, X0, yl, INK, 1.4, arrow=False)
    f.t(X0, yl + 18, "0", GY, size=13, anchor="middle")
    f._pan = None
    yb = f.src(y0 + PH + 6,
               "实测：GB300、DeepSeek-V4-Pro、SGLang，mem-fraction-static 0.85 → 0.88；每 rank 请求 886 → 999，TPOT 77.7 → 85.3 ms，"
               "端到端吞吐 10,614 → 10,704（SGLANG-V4PRO-RUNBOOK.md:849-858、885-894）。",
               "本课推导：0.8% ＝ 10,704 ÷ 10,614 − 1；上限 ＝ 999/886 × 77.7/85.3 − 1 ≈ 2.7%，假设新座位全坐满、两字间隔按实测。")
    f.save("fig6-seats.svg", yb + 14)


fig_seats()
