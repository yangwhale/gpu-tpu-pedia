# -*- coding: utf-8 -*-
r"""专题五 · 第零节「一张卡装不下」的图：放不下，也算不完。

⭐ 2026-09-25 现场「多画图少说话」：第零节原来整节只有字（约 700 字、0 张图），
   「放不下」「算不完」两件事画成一张图，替掉第二段话。
⭐ 数字全部现算：
   · 放不下：9.76 TiB（topic05_numbers，16 字节／参数 × 6,710 亿）÷ 一块卡的显存。
     例子取两款：B300 每卡 288 GB（NVIDIA DGX B300 文档：8 × 288 GB）；TPU v7 每芯片 192 GiB（wiki entities/tpu-v7）。
   · 算不完：V3 全部训练 278.8 万 H800 卡时（技术报告摘要）；一块卡 ≈ 318 年，摊到 2,048 张卡 ≈ 57 天。
⛔ 刻意没画：激活（随 batch × 序列长度变，写在落点带里）；卡时里预训练和后训练的拆分。
"""
import math

from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE
import topic05_numbers as NB

W = 1400
STATE_B = NB.PSI * NB.PER_PARAM                      # 模型状态总字节
CARDS = [("B300（每卡 288 GB）", 288e9, BL), ("TPU v7（每芯片 192 GiB）", 192 * NB.GIB, OR)]
NEED = [math.ceil(STATE_B / b) for _, b, _ in CARDS]
assert NEED == [38, 53], NEED                        # 「三四十到五六十块」
V3_CARDS = 2048                                      # V3 用 2,048 块 H800（技术报告 sec. 3.1）
DAYS = NB.V3_GPU_HOURS / V3_CARDS / 24
assert abs(DAYS - 56.7) < 0.1, DAYS


def fig_why():
    f = Fig(W, "为什么要切。左边：放不下。DeepSeek-V3 训练时光模型状态就要 9.76 TiB，一格代表一块卡的显存，"
               "按 B300 每卡 288 GB 要 38 块，按 TPU v7 每芯片 192 GiB 要 53 块，这还没算激活。"
               "右边：算不完。V3 全部训练用了 278.8 万 H800 卡时，一块卡要算 318 年，摊到 2,048 张卡上约 57 天")
    y0 = f.header("一张卡：放不下，也算不完"
                  "　——　<tspan font-weight=\"700\">切开是为了放得下，也为了算得快</tspan>",
                  "DeepSeek-V3，6,710 亿参数，常规混合精度训练（每参数 16 字节）",
                  [(BL, "B300"), (OR, "TPU v7"), (GR, "2,048 张卡一起算")])
    PH = 300
    py = f.panel(0, y0, 680, PH, "放不下：9.76 TiB 要摊到几块卡上", BL)
    for r, ((lab, _b, col), n) in enumerate(zip(CARDS, NEED)):
        yy = py + 44 + r * 120
        f.t(24, yy, "%s：%d 块" % (lab, n), col, True, 15)
        per_row = 27
        for i in range(n):
            x = 24 + (i % per_row) * 23
            y = yy + 16 + (i // per_row) * 23
            f.box(x, y, 18, 18, col, col, 3)
    f._pan = None
    px = 720
    py2 = f.panel(px, y0, 680, PH, "算不完：278.8 万 H800 卡时", GR)
    BX, BW = px + 24, 620
    f.t(BX, py2 + 50, "一块卡：%.0f 年" % NB.V3_ONE_CARD_YEARS, RD, True, 15)
    f.box(BX, py2 + 66, BW, 34, RD, RD, 4)
    f.t(BX, py2 + 150, "2,048 张卡一起算：约 %.0f 天" % DAYS, GR, True, 15)
    w_days = BW * DAYS / (NB.V3_ONE_CARD_YEARS * 365)
    f.box(BX, py2 + 166, max(w_days, 3), 34, GR, GR, 4)
    f.t(BX + max(w_days, 3) + 12, py2 + 190, "← 同一把尺子，短到只剩一条线", GY, size=13)
    f.t(BX, py2 + 250, "卡时含预训练、长上下文扩展和后训练。", GY, size=13)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "所以要切：可每切一刀，卡和卡之间就多一种通信", [
        "左边这一万 GB 还只是模型状态；激活跟 batch × 序列长度成正比，序列一长比它还大。",
        "切开之后，你缺的那块在我这儿、我缺的在你那儿，只能互相传 —— 这一讲只讲这件事。",
    ])
    yb = f.src(yb + 10,
               "📌 V3 全部训练 2.788M H800 GPU hours（DeepSeek-V3 技术报告 arXiv 2412.19437 摘要，含预训练、长上下文扩展与后训练）；2,048 块 H800（sec. 3.1）。",
               "📌 B300 每卡 288 GB（NVIDIA DGX B300 用户手册：8 × 288 GB）；TPU v7 每芯片 192 GiB HBM3e。",
               "⚠️ 本课推导：9.76 TiB ＝ 6,710 亿 × 16 字节；318 年 ＝ 2.788M ÷ 8,760 小时；57 天 ＝ 2.788M ÷ 2,048 ÷ 24。")
    f.save("fig5-why-cut.svg", yb + 14)


fig_why()
