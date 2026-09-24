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
        ("3P ＋ dep8 decode", "只换 decode 的切法", 65132, 20)]
PER = [t[2] / t[3] for t in TOPO]
assert [round(p) for p in PER] == [1820, 1319, 3257]
assert round(TOPO[1][2] / TOPO[0][2] - 1, 2) == 0.45 and round(PER[2] / PER[1], 2) == 2.47


def fig_freq():
    f = Fig(W, "每一刀一步要通信多少次，按对数刻度画成横条。示意配置是 60 层、8 个 micro-batch。"
               "TP 和 EP 每步约 1920 次，FSDP 1440 次，都是每一层都要来几次；PP 每步 16 次，只在段边界上；"
               "DP 每步只有 1 次。下面是两种线的带宽：GB300 上一块 GPU 的 NVLink 双向 1.8 TB/s，一整柜 72 块 GPU 都连在这张网上；"
               "出了这一柜就只能走网卡，双向 200 GB/s，差 9 倍。规则是频率高的放快线，频率低的才走慢线")
    y0 = f.header("每一刀多久说一次话　——　<tspan font-weight=\"700\">说得勤的放快线，说得少的才走慢线</tspan>",
                  "每步通信次数（对数刻度，本课推导）。示意配置：60 层、8 个 micro-batch、每层都是 MoE",
                  [(BL, "放在最快的那一层线上"), (OR, "可以跨到慢线上")])
    PH = 370
    py = f.panel(0, y0, W, PH, "一步里的通信次数", BL)
    BX, BW = 200, 760
    top = math.log10(2000)
    for i, (name, n, col, why) in enumerate(FREQ):
        yy = py + 48 + i * 54
        f.t(40, yy + 24, name, INK, True, 17)
        w = max(BW * math.log10(n) / top, 6)
        f.box(BX, yy, w, 34, col, col, 4)
        f.t(BX + w + 12, yy + 23, "%s 次" % format(n, ","), col, True, 15)
        f.t(BX + w + 100, yy + 23, why, GY, size=13.5)
    for v in (1, 10, 100, 1000):
        x = BX + BW * math.log10(v) / top
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
        "TP、EP、FSDP 每一层都要通信，<tspan font-weight=\"700\">只能放在同一个 NVLink 域里</tspan>（GB300 是一整柜 72 块）；度数上限就是这个域的大小。",
        "PP 一步十几次、DP 一步一次，跨到慢线上也吃得消。TPU 上同理：切片里走 ICI，跨切片走数据中心网络。",
    ])
    yb = f.src(yb + 10, "📌 次数为本课推导：TP 每层前反向各 2 次 all-reduce（arXiv 1909.08053 §3）；FSDP 每层 3 次（ZeRO 的 3Ψ，arXiv 1910.02054 §7）；"
                        "EP、PP、DP 按调度数出。链路：GB300 NVLink 5 每 GPU 1.8 TB/s；A4X Max 每节点 4 块 GPU、4 × CX-8 800 Gb/s。")
    f.save("fig5-freq.svg", yb + 14)


def fig_scale():
    f = Fig(W, "混元 3 在 TPU v7 上的两组实测，纵轴是每芯片 TFLOP/s。左边：64 芯片和 256 芯片用同一个配方，"
               "多出来的卡当成数据并行的副本，每卡的活不变，都是 580，一点没掉，这叫 weak scaling。"
               "右边：同样 256 芯片、同样每卡 batch，只改怎么分。多出来的卡当副本 453，全塞进 FSDP 把权重摊得更薄 404，"
               "少了 11%；FSDP 再窄就放不下，爆显存")
    y0 = f.header("加卡怎么加　——　<tspan font-weight=\"700\">当副本几乎不掉，摊得更薄就掉</tspan>",
                  "混元 3（295B MoE）在 TPU v7 上的实测，每芯片 TFLOP/s。左右两组的每卡 batch 不同，不要跨组比",
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
    f.panel(490, y0, W - 490, PH, "同样 256 芯片，只改怎么分（每卡 batch 8）", OR)
    for i, (lab, v) in enumerate(SPLIT):
        col = GR if lab.startswith("DP 4") else OR
        bar(530 + i * 172, v, col if v else RD, lab.split(" × ")[0], lab.split(" × ")[1], 120)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "加卡时要连 batch 一起加", [
        "左边每卡的活不变，4 倍的卡换来 4 倍的吞吐　——　weak scaling 100%。组和组之间每步只有一次梯度 all-reduce。",
        "右边权重摊得越薄，每次通信搬的越少、次数却不变，固定开销摊不掉：<tspan font-weight=\"700\">404 比 453 少 11%</tspan>。",
    ])
    yb = f.src(yb + 10, "📌 本课程作者实测：gpu-tpu-pedia tpu/Hunyuan3-295B-Pretraining/TUNING-v7 §3.7（五种分法，pdbs 8）、§4.1（64 与 256 芯片同为 580，pdbs 12）。"
                        "数字是每芯片 TFLOP/s。")
    f.save("fig5-scale.svg", yb + 14)


def fig_topo():
    f = Fig(W, "GB300 上跑 DeepSeek-V4-Pro 的三次实测，按每块 GPU 的吞吐画。原始脚本 1P1D，每卡 1820；"
               "加 prefill 机器、调并发，总吞吐涨了 45%，但卡也翻了一倍，每卡反而降到 1319。"
               "只把 decode 从 TP4 换成 dep8，每卡到 3257，是前一个的 2.47 倍")
    y0 = f.header("调参和换切法，不是一个量级　——　<tspan font-weight=\"700\">看每张卡，不看总数</tspan>",
                  "GB300 · DeepSeek-V4-Pro · vLLM，4K 进 1K 出。条长是每块 GPU 的吞吐（tok/s），括号里是总数和用了几块卡",
                  [(GY2, "TP4 decode"), (GR, "dep8 decode")])
    PH = 300
    py = f.panel(0, y0, W, PH, "每块 GPU 每秒出多少 token", GR)
    BX, BW = 330, 800
    for i, (lab, sub, tot, g) in enumerate(TOPO):
        yy = py + 50 + i * 80
        col = GR if i == 2 else GY2
        f.t(30, yy + 20, lab, INK, True, 15)
        f.t(30, yy + 42, sub, GY, size=13)
        w = BW * PER[i] / 3400
        f.box(BX, yy, w, 40, col, col, 4)
        f.t(BX + w + 12, yy + 27, "%s" % format(round(PER[i]), ","), GR if i == 2 else INK, True, 17)
        f.t(BX + w + 90, yy + 27, "（总 %s，%d 块卡）" % (format(tot, ","), g), GY, size=13.5)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "参数调得再好，也只是那一种切法的天花板", [
        "在 TP4 上加机器、调并发，总数 +45%，<tspan font-weight=\"700\">其实是靠多一倍的卡换的</tspan>；出字间隔一直钉在约 47–53 ms。",
        "换成 dep8 之后每卡 2.47 倍：MLA 的 KV 不再被 TP 复制 4 份，省下的显存全变成了更大的 batch。",
    ])
    yb = f.src(yb + 10, "📌 本课程作者实测：gpu-tpu-pedia gpu/inference/a4x-max/deepseek-v4/VLLM-V4PRO-RUNBOOK.md（1p1d 14,563；3p1d 21,100；3p ＋ dep8 65,132）。"
                        "GPU 数按每节点 4 块、prefill 与 TP4 decode 各 1 节点、dep8 2 节点；每卡数本脚本现算。")
    f.save("fig5-topo.svg", yb + 14)


fig_freq()
fig_scale()
fig_topo()
