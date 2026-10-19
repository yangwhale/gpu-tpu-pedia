# -*- coding: utf-8 -*-
r"""专题六 · 第十四节开头的「TPU 跟 GPU 哪里不一样」对照图。

回答的问题
  · fig6-s14-diff  同一个模型从 GPU 搬到 TPU，哪几处规矩变了？Qwen3.5-397B 因此每处要做什么？

数从哪来
  · GPU 一列：CUDA Graph 可选、档外照跑（12.3）；DeepEP 点对点收发、8 卡机＋IB（6.4）；
    FlashAttention、DeepEP、Marlin（5.1、6.4、12.2）；Blackwell 张量核直接算 FP4（12.1）。
  · TPU 一列：XLA 按静态形状整图编译、档外现编 60–90 s（tpu/tpu-inference/Qwen3.5-397B-A17B-FP8/README.md）；
    Ironwood 64 颗 ICI 3D 环面、每颗 6 邻居、公开推理走 all-gather ＋ reduce-scatter（6.4）；
    每个 TensorCore 64 MiB VMEM、SMEM 1 MiB、显存按 tile 存窄维补整块（5.1、14.2）；v7 公开规格只列 BF16、FP8（12.1）。
  · 右列：本节 14.2 的做法，出自 vLLM 的 PyTorch TPU 后端（写作时尚未公开）。

刻意没画
  · 不画吞吐数：收益在 14.3 的梯子图。
"""
from topic03_draw import Fig, wpx, BL, OR, GR, RD, GY, INK, GY2, LINE

W = 1400
# (差别, GPU 上, TPU 上, Qwen3.5 因此要做的)
ROWS = [
    ("形状",
     ["逐个发 kernel；录图（CUDA Graph）", "只是加速，档外的批照样跑"],
     ["XLA 按静态形状整图编译；", "没编过的档，首个请求等 60–90 s"],
     ["开服前把档位编好；", "长短不齐交给 RPA 一个 kernel"]),
    ("通信",
     ["DeepEP 点对点收发；", "8 卡机内 NVLink、机间 IB 全互连"],
     ["ICI 3D 环面，每颗只连 6 个邻居；", "XLA 未必把通信藏到你想藏的地方"],
     ["注意力按请求切、MoE 按专家切；", "整层 MoE 写成一个 Pallas 程序"]),
    ("片上与布局",
     ["现成 kernel 多：FlashAttention、", "DeepEP、Marlin"],
     ["每核 64 MiB VMEM、1 MiB SMEM；", "显存按 tile 存，窄维补成整块"],
     ["KV 序列维放最内层；GDN 状态", "换视图不搬；路由表分窗流入"]),
    ("数据格式",
     ["Blackwell 张量核直接算 FP4"],
     ["v7 公开规格只列 BF16、FP8"],
     ["权重和 KV 存 FP8；线性层状态", "存 BF16、算时提回 FP32"]),
]


def fig_diff():
    f = Fig(W, "四行对照：形状、通信、片上与布局、数据格式。每行三栏：GPU 上怎么做、TPU 上的规矩、Qwen3.5-397B 因此要做的事。"
               "形状：GPU 逐个发 kernel、录图只是加速；TPU 的 XLA 按静态形状整图编译，没编过的档首个请求等 60 到 90 秒，所以开服前编好档位，长短不齐交给 RPA。"
               "通信：GPU 用 DeepEP 点对点收发；TPU 是 ICI 3D 环面、XLA 未必藏住通信，所以注意力按请求切、MoE 按专家切，整层 MoE 写成一个 Pallas 程序。"
               "片上与布局：TPU 每核 64 MiB VMEM、1 MiB SMEM，显存按 tile 存、窄维补成整块，所以 KV 序列维放最内层、GDN 状态换视图不搬、路由表分窗流入。"
               "数据格式：Blackwell 能直接算 FP4，v7 只列 BF16 和 FP8，所以权重和 KV 存 FP8，线性层状态存 BF16")
    y0 = f.header("搬到 TPU，四条规矩变了　——　<tspan font-weight=\"700\">每条都落到这个模型的一处做法上</tspan>",
                  "Qwen3.5-397B（FP8），一台 v7x-8：4 颗芯片、8 个 TensorCore；右栏做法见 14.2",
                  [(GY, "GPU 上"), (BL, "TPU 的规矩"), (GR, "这个模型要做的")])
    C0, C1, C2, C3 = 20, 190, 590, 990
    CW = 380
    RH = 96
    PH = 64 + RH * len(ROWS)
    py = f.panel(0, y0, W, PH, "同一个模型，换了一套硬件", BL)
    for c, h, col in ((C1, "GPU 上", GY), (C2, "TPU 的规矩", BL), (C3, "Qwen3.5 要做的", GR)):
        f.t(c + 14, py + 26, h, col, True, 15)
    yy = py + 40
    for nm, g, t, m in ROWS:
        f.t(C0 + 4, yy + RH / 2 + 6, nm, INK, True, 18)
        for c, lines, col, fill in ((C1, g, GY, "#f6f7f9"), (C2, t, BL, "#eef4fe"), (C3, m, GR, "#eef8f0")):
            f.box(c, yy + 6, CW, RH - 14, fill, col, 6, sw=1.2)
            top = yy + 6 + (RH - 14 - 22 * len(lines)) / 2 + 16
            for k, ln in enumerate(lines):
                assert wpx(ln, 15) < CW - 28, (ln, wpx(ln, 15))
                f.t(c + 14, top + k * 22, ln, INK, size=15)
        f.line(C2 + CW + 2, yy + RH / 2 - 1, C3 - 4, yy + RH / 2 - 1, GR, 1.6)
        yy += RH
    f._pan = None
    yb = f.src(py + PH + 10,
               "GPU 栏：本专题 5.1、6.4、12.1、12.3；TPU 栏：tpu/tpu-inference/Qwen3.5-397B-A17B-FP8/README.md（60–90 s）、"
               "Google Cloud 博客「Inside the Ironwood TPU codesigned AI stack」、本专题 5.1、6.4、12.1。",
               "右栏：vLLM 的 PyTorch TPU 后端（写作时尚未公开），做法详见 14.2。")
    f.save("fig6-s14-diff.svg", yb + 14)


if __name__ == "__main__":
    fig_diff()
