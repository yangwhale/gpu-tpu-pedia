# -*- coding: utf-8 -*-
r"""专题六 · fig6-kvq.svg（2026-10-20 加，两档共用；当前 L200 用）。

回答：KV 少存一半位数，为什么实际省不满一半？
  · 左：DeepSeek-V3.2 每 token 每层 BF16 1,284 B 对 FP8 788 B，同显存请求约 1.63 倍（本课推导，
    算式和出处见 topic06-fig-quant.py 的 fig6-quant-kvlayout，这里直接复用那两个常数）。
  · 右：DeepSeek-V4.1-Flash 主 KV 原生存 FP4，比原先的 FP8 KV 小 45%（L300 12.4 正文）。
  · 画法：各自压之前记作 1，条长＝相对大小；虚线＝正好减半该到的位置。左右是两个模型，不横比。
  · 刻意没画：NVIDIA NVFP4 KV（没给同口径的字节数，只有「约一半」）。
"""
from topic03_draw import Fig, wpx, PU, GR, OR, GY, INK, GY2, RD

W = 1400
V32_BF16, V32_FP8 = 1284, 788
assert round(V32_BF16 / V32_FP8, 2) == 1.63
V41_FP4 = 1 - 0.45
PANELS = [("DeepSeek-V3.2：BF16 → FP8", [("BF16", 1.0, GY2, "1"), ("FP8", V32_FP8 / V32_BF16, GR, "同显存约装 1.63 倍的人（本课推导）")],
           "多出来的：没压的位置编码、新加的缩放、本来就是 FP8 的索引键"),
          ("DeepSeek-V4.1-Flash：FP8 → FP4", [("FP8", 1.0, GY2, "1"), ("FP4", V41_FP4, GR, "小 45%")],
           "每 16 个数配一个缩放；没缩满一半的那截，原文没拆")]


def fig_kvq():
    f = Fig(W, "左右两个模型各比各的，压之前的 KV 记作 1，虚线是正好减半的位置。"
               "左：DeepSeek-V3.2 的 KV 从 BF16 改存 FP8，每 token 每层 1,284 字节变 788 字节，没缩到虚线，同显存约装 1.63 倍的人（本课推导）。"
               "右：DeepSeek-V4.1-Flash 主 KV 原生存 FP4，比原先的 FP8 小 45%，也没缩到虚线")
    y0 = f.header("KV 少存一半位数　——　<tspan font-weight=\"700\">这两例都省不满一半</tspan>",
                  "条长＝每条 KV 的字节，压之前记作 1；红虚线＝正好减半该到的位置；左右两个模型，不横比",
                  [(GY2, "压之前"), (GR, "压之后")])
    PH = 300
    PW = (W - 14) // 2
    for k, (title, rows, why) in enumerate(PANELS):
        x0 = k * (PW + 14)
        py = f.panel(x0, y0, PW, PH, title, PU)
        X0, SC = x0 + 90, 380.0
        for j, (lab, v, col, tag) in enumerate(rows):
            yy = py + 36 + j * 80
            f.t(x0 + 24, yy + 30, lab, INK, True, 17)
            f.box(X0, yy + 8, v * SC, 40, col, col, 4)
            if j:
                f.t(X0 + v * SC + 12, yy + 34, tag, GR, True, 15)
        xh = X0 + SC / 2
        f.line(xh, py + 26, xh, py + 36 + 2 * 80 - 10, RD, 1.6, dash="6,4", arrow=False)
        f.t(xh, py + 36 + 2 * 80 + 10, "正好一半", RD, True, 14, "middle")
        f.t(x0 + 24, py + PH - 70, why, GY, size=15)
        f._pan = None
    yb = f.src(y0 + PH + 6,
               "V3.2：FlashMLA README（2026-09-30 提交 2e5429f 之前版本「FP8 with scale」）每 token 每层 512 个 FP8 ＋ 4 个 FP32 缩放 ＋ 64 维 BF16 RoPE ＝ 656 B，"
               "另加 132 B 的 FP8 indexer key；BF16 ＝ (512 ＋ 64) × 2 ＋ 132 ＝ 1,284 B。本课推导：1,284 ÷ 788 ≈ 1.63，没算权重和激活。",
               "V4.1-Flash：2026-09 主 KV 原生存 FP4（每 16 个数一个缩放），FlashMLA 同时放出读它的 kernel，比原先的 FP8 KV 小 45%。")
    f.save("fig6-kvq.svg", yb + 14)


fig_kvq()
