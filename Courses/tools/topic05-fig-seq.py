# -*- coding: utf-8 -*-
r"""专题五 · 第五节「第四刀：切序列」的两张静态图。

⛔ 数字现算并断言：
   · causal 负载：序列切成 2×CP 块，块 i 在因果掩码下要算 i＋1 个 KV 块。
     顺序切：卡 r 拿块 2r、2r＋1；之字形切：卡 r 拿块 r 和 2·CP−r−1（Megatron 源码
     `_get_batch_on_this_cp_rank_per_sequence_balancing` 的做法）。
   · V3 的 KV：(kv_lora_rank 512 ＋ qk_rope_head_dim 64) × 61 层 × 2 字节（bf16）＝ 70,272 字节／token，
     取自 config.json。128K 上下文 ＝ 131,072 个 token。

⛔ 刻意没画：Ring Attention 的逐步通信（动画 anim-ringattn 演）；fig-kv-dup 不画 DCP 的三次通信，只画显存。
"""
from topic03_draw import Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE

W = 1400
CP = 4
NCH = 2 * CP


def work(chunks):
    return sum(c + 1 for c in chunks)


NAIVE = [work([2 * r, 2 * r + 1]) for r in range(CP)]
ZIG = [work([r, NCH - r - 1]) for r in range(CP)]
assert NAIVE == [3, 7, 11, 15] and ZIG == [9, 9, 9, 9]
# 对角那一格因果掩码下只算一半：按半格算再核一遍（2026-09-25 专家评审）
NAIVE_H = [sum(c + 0.5 for c in (2 * r, 2 * r + 1)) for r in range(CP)]
ZIG_H = [sum(c + 0.5 for c in (r, NCH - r - 1)) for r in range(CP)]
assert NAIVE_H == [2, 6, 10, 14] and ZIG_H == [8, 8, 8, 8]

KV_TOK = (512 + 64) * 61 * 2
CTX = 131072
KV_REQ = KV_TOK * CTX
GIB = 1024 ** 3
assert KV_TOK == 70272
assert abs(KV_REQ / GIB - 8.58) < 0.01, KV_REQ / GIB
COLS = [BL, OR, GR, PU]


def fig_zigzag():
    f = Fig(W, "因果注意力切序列时的负载问题。序列切成 8 块，每块只能看自己和前面的块，"
               "所以第 0 块要算 1 格，第 7 块要算 8 格。按顺序分给 4 张卡，卡 0 算 3 格、卡 3 算 15 格，差 5 倍。"
               "之字形分法：卡 r 拿第 r 块和第 7−r 块，每张卡都是 9 格，完全均匀")
    y0 = f.header("因果注意力切序列：顺序切会不均匀　——　<tspan font-weight=\"700\">之字形切正好摆平</tspan>",
                  "序列切成 2 × 4 ＝ 8 块。行 ＝ 查询块，列 ＝ KV 块，每一格是一对要算的注意力；因果掩码只留下三角形",
                  [(BL, "卡 0"), (OR, "卡 1"), (GR, "卡 2"), (PU, "卡 3")])
    PH = 380
    py = f.panel(0, y0, W, PH, "8 × 8 的因果注意力块，谁算哪几行", BL)
    CELL = 30

    def tri(x0, owner, title, loads):
        f.t(x0, py + 26, title, INK, True, 15)
        for q in range(NCH):
            for k in range(q + 1):
                col = COLS[owner(q)]
                f.box(x0 + k * CELL, py + 44 + q * CELL, CELL - 3, CELL - 3, col, col, 3)
            f.t(x0 + NCH * CELL + 12, py + 44 + q * CELL + 20, "块 %d → 卡 %d" % (q, owner(q)), GY, size=12.5)
        yy = py + 44 + NCH * CELL + 30
        for r in range(CP):
            f.t(x0 + r * 115, yy, "卡 %d：%d 格" % (r, loads[r]), COLS[r], True, 14)
    tri(60, lambda q: q // 2, "顺序切：卡 r 拿第 2r、2r＋1 块", NAIVE)
    tri(760, lambda q: q if q < CP else NCH - 1 - q, "之字形切：卡 r 拿第 r 块和第 7−r 块", ZIG)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "顺序切差好几倍，之字形切完全均匀", [
        "因果掩码让越靠后的块算得越多。顺序切时卡 3 要算 15 格、卡 0 只算 3 格，<tspan font-weight=\"700\">大家都得等卡 3</tspan>。",
        "之字形把「最轻的一块」和「最重的一块」配成一对交给同一张卡，每张卡都是 9 格　——　Megatron 的 CP 就是这么切的。",
    ])
    yb = f.src(yb + 10, "📌 Megatron-LM megatron/core/utils.py：序列切成 2×cp_size 块，rank r 拿第 r 块和第 2·cp−r−1 块；"
                        "文档 docs/user-guide/features/context_parallel.md 称其避免下三角的多余计算并保持负载均衡。每格数本脚本现算；对角格按整格计。对角格只算一半时是 2／6／10／14 对 8／8／8／8，顺序切差 7 倍。")
    f.save("fig5-cp-zigzag.svg", yb + 14)


def fig_kv_dup():
    f = Fig(W, "一个 128K 上下文的请求，DeepSeek-V3 的 KV cache 有多大、TP 和 DCP 各让每张卡存多少。"
               "每个 token 的 KV 是 70272 字节，12 万 8 千个 token 就是约 8.58 GiB。"
               "TP 8 路时，MLA 的 KV 只有一个头，切不开，8 张卡每张都存完整的 8.58 GiB，一共 8 份一模一样的。"
               "DCP 8 路时，KV 按 token 轮流存到 8 张卡上，每张卡只存约 1.07 GiB")
    y0 = f.header("推理时切序列：KV cache 被 TP 复制了几份　——　<tspan font-weight=\"700\">DCP 把复制变回容量</tspan>",
                  "DeepSeek-V3，一个 128K 上下文的请求。每 token 的 KV ＝ (512 ＋ 64) × 61 层 × 2 字节 ＝ 70,272 字节",
                  [(RD, "重复存的那几份"), (GR, "真正需要的那一份")])
    PH = 290
    py = f.panel(0, y0, W, PH, "8 张卡上，这一个请求的 KV 各占多少", GR)
    BX, BW = 230, 900
    rows = [("TP 8 路", [KV_REQ] * 8, True), ("TP 8 ＋ DCP 8", [KV_REQ / 8] * 8, False)]
    for i, (name, per, dup) in enumerate(rows):
        yy = py + 40 + i * 110
        f.t(24, yy + 36, name, INK, True, 17)
        for c in range(8):
            w = BW / 8 - 8
            h = 60 * per[c] / KV_REQ
            col = GR if (not dup or c == 0) else RD
            f.box(BX + c * (BW / 8), yy + 60 - h, w, max(h, 3), col, col, 3)
            f.t(BX + c * (BW / 8) + w / 2, yy + 80, "卡 %d" % c, GY, size=12, anchor="middle")
        f.t(BX + BW + 20, yy + 30, ("每卡 %.2f GiB" % (per[0] / GIB)), RD if dup else GR, True, 16)
        f.t(BX + BW + 20, yy + 54, ("合计 %.1f GiB（8 份一样的）" % (sum(per) / GIB)) if dup else
            ("合计 %.2f GiB（刚好一份）" % (sum(per) / GIB)), GY, size=13)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "TP 超过 KV 头数，多出来的全是复制", [
        "MLA 的 KV 只有一个头。TP 8 路时每张卡都存完整的一份，<tspan font-weight=\"700\">7 份是白存的</tspan>。",
        "DCP 让 KV 按 token 轮流落到 8 张卡上，每卡只剩 1/8；它<tspan font-weight=\"700\">不增加卡</tspan>，直接复用 TP 那几张。",
    ])
    yb = f.src(yb + 10, "📌 vLLM context parallel 部署文档：KV cache 会被复制 tp_size / H 次；DeepSeek-R1 在 MLA 下只有 1 个 KV 头，"
                        "-tp 8 就是 8 倍复制，可加 -dcp 8。KV 尺寸取自 V3 config.json，本脚本现算。")
    f.save("fig5-kv-dup.svg", yb + 14)


fig_zigzag()
fig_kv_dup()
