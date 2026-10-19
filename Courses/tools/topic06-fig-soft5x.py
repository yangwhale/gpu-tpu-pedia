# -*- coding: utf-8 -*-
r"""专题六 · fig6-soft5x.svg（2026-10-20 加，两档共用；当前 L200 用）。

回答：同一套卡、同一个模型、同一个交互档，光改软件，每卡吞吐涨了几倍？MTP 大概占多大量级？
  · 左：PyTorch 博客 2026-06-23「Serving DeepSeek-V4 on GB300 with SGLang: 5x Higher Throughput at the Same
    Interactivity Since Day-0」：首发日（2026-04、不开 MTP）约 2,200，六月（开 MTP）约 11,200 tok/s 每卡。
    5.1 ＝ 11,200 ÷ 2,200（本课推导，博客标题写 5 倍）。
  · 右：TensorRT-LLM 8 × B200 跑 DeepSeek-R1，单用户 111 tok/s，MTP 1 层 198、3 层 253、5 层 239（L300 15.3 正文）。
  · 刻意没画：两条六月曲线各比首发日的 2.1／2.6 倍（档位不同，不能相减）。
"""
from topic03_draw import Fig, wpx, BL, GR, OR, RD, GY, INK, GY2

W = 1400
D0, D6 = 2200, 11200
assert round(D6 / D0, 1) == 5.1
MTP = [("不开", 111), ("1 层", 198), ("3 层", 253), ("5 层", 239)]


def fig_soft5x():
    f = Fig(W, "左：DeepSeek-V4-Pro 在 GB300 上跑 SGLang，硬件、模型、每个用户的出字速度都不变。"
               "首发日每卡约 2,200 token 每秒，六月约 11,200，约 5.1 倍（本课推导，博客标题写 5 倍）；六月那条开了 MTP。"
               "右：另一组，TensorRT-LLM 在 8 张 B200 上跑 DeepSeek-R1，单用户每秒 111 token，开 1 层 MTP 198、3 层 253、5 层反降到 239")
    y0 = f.header("卡没换、模型没换、出字一样快　——　<tspan font-weight=\"700\">只换软件，每卡 5.1 倍</tspan>",
                  "左：GB300 · DeepSeek-V4-Pro · SGLang；右：另一组模型和卡，只看 MTP 的量级",
                  [(GY2, "首发日"), (BL, "六月（开 MTP）"), (OR, "MTP 层数")])
    PH = 400
    LW = 760
    py = f.panel(0, y0, LW, PH, "每卡每秒 token（同一交互档）", BL)
    base = py + PH - 86
    sc = 260 / D6
    for k, (lab, sub, v, col, x) in enumerate((("首发日", "2026-04 · 不开 MTP", D0, GY2, 150),
                                              ("六月", "2026-06 · 开 MTP", D6, BL, 430))):
        h = v * sc
        f.box(x, base - h, 150, h, col, col, 4)
        f.t(x + 75, base - h - 12, "约 {:,}".format(v), col, True, 18, "middle")
        f.t(x + 75, base + 26, lab, INK, True, 16, "middle")
        f.t(x + 75, base + 48, sub, GY, size=14, anchor="middle")
    f.line(110, base, 640, base, INK, 1.2, arrow=False)
    xa = 330
    f.path("M%d,%d C%d,%d %d,%d %d,%d" % (300, base - D0 * sc - 30, xa + 20, base - 220,
                                           xa + 60, base - 270, 425, base - D6 * sc + 30), GR, 2.0, arrow=True)
    f.t(xa + 22, base - 150, "×5.1", GR, True, 26, big=True)
    f.t(xa + 22, base - 124, "本课推导", GY, size=13)
    f._pan = None
    RX = LW + 14
    pr = f.panel(RX, y0, W - RX, PH, "另一组：8 张 B200 · R1 · 单用户", OR)
    sc2 = 230 / 260
    bw = 86
    for j, (lab, v) in enumerate(MTP):
        x = RX + 50 + j * 140
        h = v * sc2
        col = GY2 if j == 0 else OR
        f.box(x, base - h, bw, h, col, col, 4)
        f.t(x + bw / 2, base - h - 12, str(v), col, True, 17, "middle")
        f.t(x + bw / 2, base + 26, lab, INK, True, 16, "middle")
    f.t(RX + 50 + 3 * 140 + bw / 2, base + 48, "反而降了", RD, True, 14, "middle")
    f.line(RX + 30, base, W - 30, base, INK, 1.2, arrow=False)
    f._pan = None
    yb = f.band(y0 + PH + 16, "ok", "每个字的成本约降到五分之一（本课推导：按全部卡算、卡价不变）", [
        "每卡除的是 decode 卡还是全部卡，博客没写；两个数出自同一张图，比值不受影响。",
        "5.1 倍是整个软件栈改进的合计，MTP 占多少博客没拆；右边只当量级参考。"])
    yb = f.src(yb + 10,
               "左：PyTorch 博客 2026-06-23「Serving DeepSeek-V4 on GB300 with SGLang: 5x Higher Throughput at the Same Interactivity Since Day-0」"
               "（「the June 2026 MTP curve delivers ~11,200 tok/s/GPU … versus ~2,200 tok/s/GPU for the Day-0 (April 2026) no-MTP curve」）。",
               "本课推导：11,200 ÷ 2,200 ≈ 5.1；成本 2,200 ÷ 11,200 ≈ 0.196。",
               "右：TensorRT-LLM 技术博客「Pushing Latency Boundaries: Optimizing DeepSeek-R1 Performance on NVIDIA B200 GPUs」，8 × B200、输入 1k 输出 2k，不开 MTP 111 TPS/user，MTP1 198、MTP3 253、MTP5 239；模型和卡都与左边不同。")
    f.save("fig6-soft5x.svg", yb + 14)


fig_soft5x()
