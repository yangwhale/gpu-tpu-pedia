# -*- coding: utf-8 -*-
r"""专题五 · 第二节「第一刀：切数据」的两张图。

⭐ 这一节的承重结论（ZeRO 原论文 arXiv 1910.02054 sec. 7 原话核过）：
   · ZeRO-1 / ZeRO-2 的通信量跟数据并行**一模一样**（2Ψ）——&#160;白送
   · ZeRO-3（＝ FSDP）要 **3Ψ，是数据并行的 1.5 倍** ——&#160;最后那 2 个字节是要付钱的
⛔ 大纲原来写「FSDP 白送」，**说宽了**：白送的是前两级，而前两级已经削掉 16 字节里的 14 个。

⛔ 每卡字节数全部现算：每参数 16 字节 ＝ 权重 2（bf16）＋ 梯度 2（bf16）＋ 优化器状态 12
   （fp32 主权重 4 ＋ m 4 ＋ v 4），口径跟专题四那张 16 字节表一致。

⛔ 刻意没画：激活（它随 batch 和序列长度变，专题四讲过）；ZeRO++ 这类通信优化。每卡字节只算模型状态。
"""
from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE

from topic05_numbers import PSI, N_DP as N, TIB, GIB, W_B, G_B, O_B   # 16 字节的账只有一份

W = 1400
STAGES = [                       # (名字, 权重份额, 梯度份额, 状态份额, 通信 Ψ 倍数)
    ("数据并行", 1, 1, 1, 2),
    ("ZeRO-1", 1, 1, 1 / N, 2),
    ("ZeRO-2", 1, 1 / N, 1 / N, 2),
    ("ZeRO-3 ＝ FSDP", 1 / N, 1 / N, 1 / N, 3),
]


def per_param(ws, gs, os_):
    return W_B * ws + G_B * gs + O_B * os_


def per_dev_bytes(st):
    return PSI * per_param(*st[1:4])


_b = [per_dev_bytes(s) for s in STAGES]
assert abs(_b[0] / TIB - 9.76) < 0.01, _b[0] / TIB          # 跟专题四 5.1 的 9.76 TiB 对齐
assert abs(_b[1] / TIB - 2.45) < 0.01, _b[1] / TIB
assert abs(_b[2] / TIB - 1.23) < 0.01, _b[2] / TIB
assert abs(_b[3] / GIB - 9.76) < 0.01, _b[3] / GIB          # 9.76 TiB → 9.76 GiB，正好 ÷1024


def fmt(b):
    return "%.2f TiB" % (b / TIB) if b >= TIB else "%.2f GiB" % (b / GIB)


def fig_mem():
    f = Fig(W, "数据并行和 ZeRO 三级，每张卡的常驻显存。每个参数 16 字节：权重 2、梯度 2、优化器状态 12。"
               "数据并行每张卡都存一整份，16 字节。ZeRO-1 把优化器状态切成 1024 份，每卡剩约 4 字节；"
               "ZeRO-2 再切梯度，剩约 2 字节；ZeRO-3 连权重也切，只剩千分之十六字节。"
               "换成 DeepSeek-V3 的 6710 亿参数、1024 路：每卡 9.76 TiB、2.45 TiB、1.23 TiB、9.76 GiB。"
               "通信量：前三种都是 2 Ψ，跟数据并行一模一样；ZeRO-3 是 3 Ψ，多一半")
    y0 = f.header("ZeRO：把重复的削掉，一次削一样"
                  "　——　<tspan font-weight=\"700\">前两级白送，最后一级要付 50% 的通信</tspan>",
                  "每个参数 16 字节，按专题四那张表的口径拆成三块。例子：DeepSeek-V3，6,710 亿参数，1,024 路数据并行",
                  [(BL, "权重 2 字节（bf16）"), (OR, "梯度 2 字节（bf16）"), (GR, "优化器状态 12 字节（fp32）")])
    RH, BX, SCALE = 64, 250, 38.0          # 每字节 38 px，16 字节 ＝ 608 px
    PH = 30 + 30 + len(STAGES) * RH + 20
    py = f.panel(0, y0, W, PH, "每张卡要常驻多少字节 / 参数", GR, tag="条长 ∝ 每参数字节数")
    f.t(BX, py + 22, "每参数字节（切成 1,024 份之后）", GY, True, 13)
    f.t(1040, py + 22, "V3 每卡常驻", GY, True, 13)
    f.t(1220, py + 22, "每步通信", GY, True, 13)
    for i, st in enumerate(STAGES):
        name, ws, gs, os_, comm = st
        yy = py + 34 + i * RH
        f.t(18, yy + 28, name, INK, True, 16)
        x = BX
        for col, bytes_, share in ((BL, W_B, ws), (OR, G_B, gs), (GR, O_B, os_)):
            w = bytes_ * share * SCALE
            if w >= 1:
                f.box(x, yy + 8, w, 32, col, col, 3)
            x += w
        pp = per_param(ws, gs, os_)
        f.t(x + 12, yy + 30, ("%.2f 字节" % pp) if pp >= 0.1 else ("%.3f 字节" % pp), INK, True, 14)
        f.t(1040, yy + 30, fmt(per_dev_bytes(st)), RD if i < 3 else GR, True, 16)
        f.t(1220, yy + 30, "%dΨ%s" % (comm, "（1.5×）" if comm == 3 else "（＝数据并行）" if i else ""),
            RD if comm == 3 else INK, comm == 3, 15)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "前两级白送，最后一级要付钱", [
        "ZeRO-1、ZeRO-2 通信量跟数据并行<tspan font-weight=\"700\">一个字节都不多</tspan>（一步只同步一次时），却已经把 16 字节削到约 2 字节。"
        "　可 V3 按这个算每卡还要 1.23 TiB，一张卡照样装不下。",
        "只有 ZeRO-3 能降到每卡 9.76 GiB，代价是通信从 2Ψ 变成 <tspan font-weight=\"700\">3Ψ，多一半</tspan>。"
        "　这就是 FSDP：大模型没得选，只能付这 50%。",
    ])
    yb = f.src(yb + 10,
               "📌 出处：Rajbhandari 等，ZeRO，arXiv 1910.02054 sec. 5、sec. 7。原文：Pos、Pos+g 通信量与数据并行相同（2Ψ），"
               "Pos+g+p 最多 1.5 倍。Ψ 为参数个数，此处按元素数计。",
               "⚠️ 每卡字节数本脚本现算并 assert；9.76 TiB 与专题四 5.1 同一口径（671e9 × 16 ÷ 1024⁴）。"
               "只算常驻，不含激活。")
    f.save("fig5-zero-mem.svg", yb + 14)


def fig_step():
    f = Fig(W, "一层在一步训练里要做哪些通信。数据并行：前向和反向都不通信，反向算完梯度后做一次全归约，"
               "也就是一次归约分散加一次全收集，一共两份。FSDP：前向之前先全收集把这一层的权重拼回来，算完就扔；"
               "反向之前再全收集一次，算完梯度做一次归约分散，一共三份，是数据并行的一点五倍。"
               "多出来的那一份，就是反向时把扔掉的权重再拼一次")
    y0 = f.header("FSDP 为什么多 50%　——　<tspan font-weight=\"700\">反向时要把扔掉的权重再拼一次</tspan>",
                  "同一层、同一步。每个色块是一次通信，量都是这一层权重的 (n−1)/n 份",
                  [(BL, "AllGather（拼回整份）"), (GR, "ReduceScatter（加起来再分）"), (GY2, "计算")])
    PH = 30 + 2 * 92 + 30
    py = f.panel(0, y0, W, PH, "一层 · 一步", BL)
    rows = [
        ("数据并行", [("前向", GY2, 170), ("反向", GY2, 300), ("RS", GR, 110), ("AG", BL, 110)],
         "2 份（RS ＋ AG ＝ 一次 AllReduce）"),
        ("FSDP", [("AG", BL, 110), ("前向", GY2, 170), ("扔掉权重", None, 90), ("AG", BL, 110),
                  ("反向", GY2, 300), ("RS", GR, 110)], "3 份 ＝ 1.5×"),
    ]
    for i, (name, segs, tail) in enumerate(rows):
        yy = py + 34 + i * 92
        f.t(18, yy + 30, name, INK, True, 16)
        x = 160
        for lab, col, w in segs:
            if col is None:
                f.box(x, yy + 8, w - 8, 36, "none", LINE, 4, dash="4,3")
                f.t(x + (w - 8) / 2.0, yy + 31, lab, GY, size=12.5, anchor="middle")
            else:
                f.box(x, yy + 8, w - 8, 36, col, col, 4)
                f.t(x + (w - 8) / 2.0, yy + 31, lab, "#ffffff", True, 14, "middle")
            x += w
        f.t(x + 16, yy + 31, tail, RD if i else INK, True, 15)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "拼一次、扔掉、再拼一次", [
        "FSDP 每层前向前拼一次权重，算完就扔，所以显存里永远只有一层是完整的；"
        "反向要用时，<tspan font-weight=\"700\">只好再拼一次</tspan>　——　多出来的正是这一份。",
        "反过来也成立：前向后不扔（PyTorch 叫 reshard_after_forward=False），通信就回到 2 份，"
        "代价是整个反向期间都攥着完整权重。又是一次拿通信换显存。",
    ])
    f.save("fig5-fsdp-step.svg", yb + 14)


fig_mem()
fig_step()
