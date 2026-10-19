# -*- coding: utf-8 -*-
r"""专题六 · fig6-nvl72.svg（2026-10-20 加，两档共用；当前 L200 用）。

回答：整柜一个 NVLink 域跑大 EP，每卡比 H100 上的大 EP 部署强多少？
  · 数字：LMSYS「GB200 NVL72 Part I」（2025-06-16）decode 每卡 7,583 tok/s，对 H100 每卡 2.7×（2K 输入）；
    「Part II」（2025-09-25）FP8 注意力 ＋ NVFP4 专家，prefill 3.8×、decode 4.8×。均为 L300 第六节正文与台账已有的数。
  · 画法：H100 大 EP 记作 1，横条长度＝倍数。
  · 刻意没画：任何对 B200 的倍数（课程口径只用对 H100 的数）；Part II 原精度的两个绝对值（L300 正文有）。
"""
from topic03_draw import Fig, wpx, BL, PU, GY, INK, GY2

W = 1400
BARS = [("H100 大 EP", "基线", 1.0, GY2),
        ("NVL72 · 原精度", "decode，每卡 7,583 tok/s", 2.7, BL),
        ("NVL72 · 压低精度", "prefill", 3.8, PU),
        ("NVL72 · 压低精度", "decode", 4.8, PU)]


def fig_nvl72():
    f = Fig(W, "SGLang 在 GB200 NVL72 上把整柜当一个 EP 域跑 DeepSeek V3／R1，2K 输入、PD 分离、decode 48 张卡。"
               "每卡吞吐以 H100 上的大 EP 部署为 1：第一篇原精度 decode 每卡 7,583 tok/s，2.7 倍；"
               "第二篇注意力用 FP8、专家用 NVFP4，prefill 3.8 倍、decode 4.8 倍。"
               "底下提醒：倍数里混着显存、精度、kernel 和 PD 分离；decode 先压慢攒满请求再放开测，H100 基线有一组开了模拟 MTP")
    y0 = f.header("整柜一个 NVLink 域跑大 EP　——　<tspan font-weight=\"700\">每卡是 H100 大 EP 部署的 2.7 到 4.8 倍</tspan>",
                  "SGLang · GB200 NVL72 · DeepSeek V3／R1 · 2K 输入 · PD 分离、decode 48 张卡；H100 记作 1",
                  [(GY2, "H100 大 EP"), (BL, "第一篇：原精度"), (PU, "第二篇：FP8 注意力＋NVFP4 专家")])
    PH = 350
    py = f.panel(0, y0, W, PH, "每卡吞吐，对 H100 上的大 EP 部署", INK)
    X0, SC = 330, 190.0
    assert X0 + 4.8 * SC + 90 < W
    for j, (lab, sub, v, col) in enumerate(BARS):
        yy = py + 34 + j * 70
        f.t(24, yy + 24, lab, INK, True, 17)
        f.t(24, yy + 46, sub, GY, size=14)
        f.box(X0, yy + 8, v * SC, 36, col, col, 4)
        f.t(X0 + v * SC + 14, yy + 34, "%g×" % v, col, True, 18)
    yl = py + 34 + len(BARS) * 70 - 10
    f.line(X0 + SC, py + 22, X0 + SC, yl, GY2, 1.2, dash="5,4", arrow=False)
    f._pan = None
    yb = f.band(y0 + PH + 16, "warn", "别全记在大 EP 头上", [
        "倍数里混着显存、精度、kernel 和 PD 分离。",
        "decode 先压慢、攒满请求再放开测；H100 基线有一组开了模拟 MTP，GB200 没开。"])
    yb = f.src(yb + 10,
               "LMSYS 2025-06-16「GB200 NVL72 Part I」：decode 每卡 7,583 tok/s，对 H100 每卡 2.7×（2,000 token 输入；H100 柱含模拟 MTP；decode 用 slow-down 接口攒满再放开）。",
               "LMSYS 2025-09-25「Part II」：FP8 注意力 ＋ NVFP4 专家，prefill 26,156、decode 13,386 tok/s 每卡，对 H100 3.8× 和 4.8×；decode 48 卡（「For decode, we use 48 ranks」）。",
               "H100 基线是两篇各自引用的 H100 PD ＋ 大 EP 部署；倍数为原文所报，未做同精度拆分。")
    f.save("fig6-nvl72.svg", yb + 14)


fig_nvl72()
