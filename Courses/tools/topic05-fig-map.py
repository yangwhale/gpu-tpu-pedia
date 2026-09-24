# -*- coding: utf-8 -*-
r"""专题五 · 第七节「摆到机器上」的三张静态图。

⛔ 数字现算并断言：
   · 每步通信次数（本课推导，示意配置 60 层、8 个 micro-batch、每层都是 MoE）：
       TP   每层前向 2 次、反向 2 次 all-reduce（Megatron arXiv 1909.08053 §3）→ 4·L·m
       EP   每个 MoE 层前向派发＋合并 2 次、反向再 2 次 all-to-all → 4·L·m
       FSDP 每层前向 all-gather、反向 all-gather ＋ reduce-scatter（ZeRO 的 3Ψ）→ 3·L·m
       PP   每个 micro-batch 在一个段边界上前向发一次、反向发一次 → 2·m
       DP   每步一次梯度 all-reduce（实际按桶分几次，仍是「每步」量级）→ 1
   · 链路：GB300 每 GPU NVLink 1.8 TB/s（双向）；跨机每 GPU 一块 800 Gb/s 网卡 ＝ 100 GB/s 单向、200 GB/s 双向
     （A4X Max：每节点 4 × CX-8 × 800 Gb/s、4 块 GPU）。同口径之比 9 倍。
   · 混元 3（本课程作者实测，gpu-tpu-pedia tpu/Hunyuan3-295B-Pretraining/TUNING-v7 §3.7、§4.1）。
   · GB300 V4-Pro（本课程作者实测，gpu-tpu-pedia gpu/inference/a4x-max/deepseek-v4/VLLM-V4PRO-RUNBOOK.md）。
"""
import math
from topic03_draw import Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE

W = 1400
L, M = 60, 8
FREQ = [("TP", 4 * L * M, BL, "每层 4 次 all-reduce"),
        ("EP", 4 * L * M, BL, "每个 MoE 层 4 次 all-to-all"),
        ("FSDP", 3 * L * M, BL, "每层拼一次、反向再拼一次、再散一次"),
        ("PP", 2 * M, OR, "每个 micro-batch 过段边界一来一回"),
        ("DP", 1, OR, "每步一次梯度 all-reduce")]
assert [f[1] for f in FREQ] == [1920, 1920, 1440, 16, 1]
NVL = 1800
NIC = 800 / 8 * 2                 # 800 Gb/s 单向 100 GB/s，双向 200 GB/s
assert NIC == 200 and NVL / NIC == 9

# 混元 3（TFLOP/s/chip）
WEAK = [("64 芯片", "FSDP 128", 580), ("256 芯片", "DP 4 × FSDP 128", 580)]
SPLIT = [("DP 1 × FSDP 512", 404), ("DP 2 × FSDP 256", 450), ("DP 4 × FSDP 128", 453),
         ("DP 8 × FSDP 64", None), ("DP 16 × FSDP 32", None)]
assert WEAK[1][2] / WEAK[0][2] == 1.0 and round(404 / 453 * 100) == 89

# GB300 V4-Pro（vLLM，4K 进 1K 出）
TOPO = [("1P1D，TP4 decode", "原始脚本", 14563, 8),
        ("3P1D，TP4 decode", "加 prefill、调并发后的最好成绩", 21100, 16),
        ("3P ＋ dep8 decode", "换 decode 的切法（decode 卡 4→8 张），同样并发 512", 55153, 20)]
# ⛔ 2026-09-25 GPU 专家评审：原来第三行用 65,132（并发 1,536、首字延迟 95 秒），跟并发 512 的 21,100 不对等。
#   改成同并发 512 的 55,153；65,132 只在图注里提一句。
PER = [t[2] / t[3] for t in TOPO]
assert [round(p) for p in PER] == [1820, 1319, 2758]
assert round(TOPO[1][2] / TOPO[0][2] - 1, 2) == 0.45 and round(PER[2] / PER[1], 2) == 2.09
assert round(65132 / 20 / PER[1], 2) == 2.47


def fig_freq():
    f = Fig(W, "每一刀一步要通信多少次，按实际比例画成横条。示意配置是 60 层、8 个 micro-batch。"
               "TP 和 EP 每步约 1920 次，FSDP 1440 次，都是每一层都要来几次；PP 每步 16 次，只在段边界上；"
               "DP 每步只有 1 次。下面是两种线的带宽：GB300 上一块 GPU 的 NVLink 双向 1.8 TB/s，一整柜 72 块 GPU 都连在这张网上；"
               "出了这一柜就只能走网卡，双向 200 GB/s，差 9 倍。规则是频率高的放快线，频率低的才走慢线")
    y0 = f.header("每一刀多久说一次话　——　<tspan font-weight=\"700\">说得勤的放快线，说得少的才走慢线</tspan>",
                  "每步通信次数（按实际比例，本课推导）。示意配置：60 层、8 个 micro-batch、每层都是 MoE",
                  [(BL, "放在最快的那一层线上"), (OR, "可以跨到慢线上")])
    PH = 370
    py = f.panel(0, y0, W, PH, "一步里的通信次数", BL)
    BX, BW = 200, 760
    # ⭐ 2026-09-25 逐图审：原来是对数刻度，1,920 对 16 画出来只差约 3 倍长，「差三个数量级」被刻度吃掉了。
    #   改成线性：PP、DP 短到几乎看不见 —— 这正是要讲的点。
    top = 1920
    for i, (name, n, col, why) in enumerate(FREQ):
        yy = py + 48 + i * 54
        f.t(40, yy + 24, name, INK, True, 17)
        w = max(BW * n / top, 3)
        f.box(BX, yy, w, 34, col, col, 4)
        f.t(BX + w + 12, yy + 23, "%s 次" % format(n, ","), col, True, 15)
        f.t(BX + w + 100, yy + 23, why, GY, size=13.5)
    for v in (0, 500, 1000, 1500):
        x = BX + BW * v / top
        f.line(x, py + 318, x, py + 310, GY2, 1, arrow=False)
        f.t(x, py + 330 - 2, format(v, ","), GY, size=12, anchor="middle")
    f._pan = None
    py2 = f.panel(0, py + PH + 24, W, 170, "两种线差多少（GB300 NVL72，每块 GPU，双向）", OR)
    for i, (lab, v, col) in enumerate([("一柜之内：NVLink", NVL, BL), ("出了这一柜：网卡", NIC, OR)]):
        yy = py2 + 44 + i * 50
        f.t(40, yy + 24, lab, col, True, 15)
        w = 900 * v / NVL
        f.box(280, yy, w, 34, col, col, 4)
        f.t(280 + w + 12, yy + 23, "%s GB/s" % format(int(v), ","), col, True, 15)
    f.t(560, py2 + 117, "← 差 %d 倍" % (NVL / NIC), RD, True, 20)
    f._pan = None
    yb = f.band(py2 + 170 + 20, "ok", "上下两张图一对，摆法就出来了", [
        "TP 每一层都要通信又藏不住，<tspan font-weight=\"700\">必须待在 NVLink 域里</tspan>（GB300 是一整柜 72 块）；EP、FSDP 能边算边传时才敢跨出去。",
        "PP 一步十几次、DP 一步一次，跨到慢线上也吃得消。TPU 上同理：切片里走 ICI，跨切片走数据中心网络。",
    ])
    yb = f.src(yb + 10, "📌 次数为本课推导：TP 每层前反向各 2 次 all-reduce（arXiv 1909.08053 §3）；FSDP 每层 3 次（ZeRO 的 3Ψ，arXiv 1910.02054 §7）；"
                        "EP、PP、DP 按调度数出；叠上 PP 时 TP 的次数要除以 PP 段数，开重算时 TP 每层是 6 次。链路：GB300 NVLink 5 每 GPU 1.8 TB/s；A4X Max 每节点 4 块 GPU、4 × CX-8 800 Gb/s。")
    f.save("fig5-freq.svg", yb + 14)


def fig_scale():
    f = Fig(W, "混元 3 在 TPU v7 上的两组实测，纵轴是每芯片 TFLOP/s。左边：64 芯片和 256 芯片用同一个配方，"
               "多出来的卡当成数据并行的副本，每卡的活不变，都是 580，一点没掉，这叫 weak scaling。"
               "右边：同样 256 芯片、同样每卡 batch，只改怎么分。多出来的卡当副本 453，全塞进 FSDP 把权重摊得更薄 404，"
               "少了 11%；FSDP 再窄就放不下，爆显存")
    y0 = f.header("加卡怎么加　——　<tspan font-weight=\"700\">当副本几乎不掉，摊得更薄就掉</tspan>",
                  "混元 3 在 TPU v7 上的实测，每芯片 TFLOP/s。v7 一颗芯片算 2 个 device，DP × FSDP 按 device 数（64 芯片 ＝ 128 个）。左右两组每卡 batch 不同",
                  [(GR, "多出来的卡当 DP 副本"), (OR, "多出来的卡全塞进 FSDP"), (RD, "放不下")])
    PH = 370
    py = f.panel(0, y0, 470, PH, "同配方放大 4 倍（每卡 batch 12）", GR)
    H, BASE = 220, py + 300
    def bar(x, v, col, lab, sub, w=110):
        if v is None:
            f.t(x + w / 2, BASE - 14, "爆显存", RD, True, 15, "middle")
        else:
            h = H * v / 600
            f.box(x, BASE - h, w, h, col, col, 4)
            f.t(x + w / 2, BASE - h - 10, str(v), col, True, 17, "middle")
        f.t(x + w / 2, BASE + 20, lab, INK, True, 13.5, "middle")
        if sub:
            f.t(x + w / 2, BASE + 38, sub, GY, size=12.5, anchor="middle")
    bar(70, WEAK[0][2], GR, WEAK[0][0], WEAK[0][1], 130)
    bar(270, WEAK[1][2], GR, WEAK[1][0], WEAK[1][1], 130)
    f._pan = None
    f.panel(490, y0, W - 490, PH, "同样 256 芯片，只改怎么分（每卡 batch 8，不能跟左边比）", OR)
    for i, (lab, v) in enumerate(SPLIT):
        col = GR if lab.startswith("DP 4") else OR
        bar(530 + i * 172, v, col if v else RD, lab.split(" × ")[0], lab.split(" × ")[1], 120)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "加卡时要连 batch 一起加", [
        "左边每卡的活不变，4 倍的卡换来 4 倍的吞吐　——　weak scaling 100%。组和组之间每步只有一次梯度 all-reduce。",
        "右边一个 FSDP 组从 128 个 device 扩到 512 个，拼权重的步数多了、每块更小，固定延迟摊不掉：<tspan font-weight=\"700\">404 比 453 少 11%</tspan>。",
    ])
    yb = f.src(yb + 10, "📌 本课程作者实测：gpu-tpu-pedia tpu/Hunyuan3-295B-Pretraining/TUNING-v7 §3.7（五种分法，pdbs 8）、§4.1（64 与 256 芯片同为 580，pdbs 12）。"
                        "数字是每芯片 TFLOP/s。")
    f.save("fig5-scale.svg", yb + 14)


def fig_topo():
    f = Fig(W, "GB300 上跑 DeepSeek-V4-Pro 的三次实测，按每块 GPU 的吞吐画。原始脚本 1P1D，每卡 1820；"
               "加 prefill 机器、调并发，总吞吐涨了 45%，但卡也翻了一倍，每卡反而降到 1319。"
               "把 decode 从 TP4 换成 dep8（decode 卡也从 4 张变 8 张），同样并发下每卡到 2758，是前一个的 2.09 倍")
    y0 = f.header("调参和换切法，不是一个量级　——　<tspan font-weight=\"700\">看每张卡，不看总数</tspan>",
                  "GB300 · DeepSeek-V4-Pro · vLLM，4K 进 1K 出。条长是每块 GPU 的吞吐（tok/s），括号里是总数和卡数",
                  [(GY2, "TP4 decode"), (GR, "dep8 decode")])
    PH = 300
    py = f.panel(0, y0, W, PH, "每块 GPU 每秒出多少 token", GR)
    BX, BW = 330, 800
    for i, (lab, sub, tot, g) in enumerate(TOPO):
        yy = py + 50 + i * 80
        col = GR if i == 2 else GY2
        f.t(30, yy + 20, lab, INK, True, 15)
        f.t(30, yy + 42, sub, GY, size=13)
        w = BW * PER[i] / 3000
        f.box(BX, yy, w, 40, col, col, 4)
        f.t(BX + w + 12, yy + 27, "%s" % format(round(PER[i]), ","), GR if i == 2 else INK, True, 17)
        f.t(BX + w + 90, yy + 27, "（总 %s，%d 块卡）" % (format(tot, ","), g), GY, size=13.5)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "参数调得再好，也只是那一种切法的天花板", [
        "在 TP4 上加机器、调并发，总数 +45%，<tspan font-weight=\"700\">其实是靠多一倍的卡换的</tspan>；出字间隔一直钉在约 46.8–53 ms。",
        "换成 dep8，同样并发下每卡 2.09 倍、出字间隔降到约 12 ms；把并发拉到 1,536 每卡能到 2.47 倍，但首字要等 95 秒。",
    ])
    yb = f.src(yb + 10, "📌 本课程作者实测：gpu-tpu-pedia gpu/inference/a4x-max/deepseek-v4/VLLM-V4PRO-RUNBOOK.md（1p1d 14,563；3p1d 21,100；3p ＋ dep8 并发 512 为 55,153、并发 1,536 为 65,132）。"
                        "GPU 数按每节点 4 块、prefill 与 TP4 decode 各 1 节点、dep8 2 节点；每卡数本脚本现算。")
    f.save("fig5-topo.svg", yb + 14)


fig_freq()
fig_scale()
fig_topo()


# ── 第八节：全景图 ───────────────────────────────────────────────────────
# ⭐ 2026-09-25 R7 麻瓜全篇重读：开头许诺「一张全景地图」，第八节却全是表。这张把五刀归进四类，
#   每个名字只挂两样东西：用的是哪种通信、能不能跨到慢线上。细节仍在后面的表里。
PANO = [
    ("数据并行", "切 batch", BL, [("DP", "AllReduce", 1), ("ZeRO-1", "RS ＋ AG", 1), ("FSDP", "AG ×2 ＋ RS", 0),
                                 ("HSDP", "机内 FSDP、机间 DP", 1), ("Attention DP", "推理：MoE 里跟着 EP 走", 0)]),
    ("序列并行", "切一条样本", GR, [("SP", "AG ＋ RS（TP 的搭档）", 0), ("CP · Ring", "Send／Recv 沿环", 0),
                                  ("Ulysses", "AllToAll", 0), ("DCP", "推理：收齐 Q、合并", 0), ("PCP", "推理：切长 prompt", 0)]),
    ("模型并行", "切权重", OR, [("TP", "AllReduce，每层", 0), ("PP", "Send／Recv，段边界", 1),
                              ("EP", "AllToAll，每个 MoE 层", 0), ("Wide-EP", "EP 铺到几十张卡", 0)]),
    ("解耦", "拆工作", PU, [("PD 分离", "KV 跨机传一次", 1), ("AFD", "每层 M → N → M", 0), ("Encoder 分离", "embedding 传一次", 1)]),
]
assert sum(len(c[3]) for c in PANO) == 17


def fig_panorama():
    f = Fig(W, "并行方式的全景图。四列是四类：数据并行切 batch，序列并行切一条样本，模型并行切权重，解耦拆工作。"
               "每个名字下面写着它用哪种通信；实心圆点表示它可以跨到慢线上，空心圆点表示它要待在快线里。"
               "数据并行里的 DP、ZeRO-1、HSDP 能跨慢线，FSDP 要在快线里，Attention DP 在 MoE 里跟着专家并行走，也要在快线里；模型并行里只有 PP 能跨慢线；"
               "解耦里 PD 分离和 Encoder 分离只传一次，能跨机器")
    y0 = f.header("全景：四类刀法，每个名字挂两样东西　——　<tspan font-weight=\"700\">用哪种通信，能不能跨慢线</tspan>",
                  "五刀里切权重和切专家都归「模型并行」。● 可以跨到慢线上　○ 要待在快线里（默认摆法，有例外，见第七节）",
                  [(BL, "数据并行"), (GR, "序列并行"), (OR, "模型并行"), (PU, "解耦")])
    PH = 470
    colw = (W - 50) / 4
    for c, (name, what, col, items) in enumerate(PANO):
        x = 10 + c * (colw + 10)
        py = f.panel(x, y0, colw, PH, "%s　%s" % (name, what), col)
        for k, (nm, comm, slow) in enumerate(items):
            yy = py + 30 + k * 84
            f.box(x + 14, yy, colw - 28, 70, "none", col, 10, sw=2)
            f.t(x + 30, yy + 30, nm, col, True, 17)
            f.t(x + 30, yy + 54, comm, GY, size=13.5)
            cx = x + colw - 40
            f.t(cx, yy + 32, "●" if slow else "○", col, True, 22, "middle")
        f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "读名字先问两句：它切的是什么，它多出来的是哪种通信", [
        "切什么定了它在哪一列；多出来的通信定了它该放在哪根线上。<tspan font-weight=\"700\">说得少的才敢跨慢线</tspan>。",
        "TEP8、DEP16 这类推理简称，是把 attention 的切法和专家的切法拼在一起说：前一个字母管 attention，EP 管专家。",
    ])
    yb = f.src(yb + 10, "📌 归类与通信原语汇总自本讲第一到第七节；「能否跨慢线」是默认摆法，例外（V3 的 EP 跨机、Llama 3 的 FSDP 在最外层）见第七节。")
    f.save("fig5-panorama.svg", yb + 14)


fig_panorama()
