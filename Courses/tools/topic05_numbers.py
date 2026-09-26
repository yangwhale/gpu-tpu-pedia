# -*- coding: utf-8 -*-
r"""专题五共用的几个承重数 —— 只有这一份。

⭐ 为什么单独一个模块：ZeRO 那张图（topic05-fig-zero.py）和开场热身题（topic05_quiz.py）
   用的是同一张「每参数 16 字节」的账。各抄一份，改了一边另一边会静默留在旧值上
   （专题四 topic04_quiz.py 顶上同一条规矩：「数据必须只有一份」）。

口径（跟专题四那张 16 字节表一致）：常规混合精度 AdamW
   · 半精度（bf16）权重 2 ＋ 半精度梯度 2
   · 全精度（fp32）主权重 4 ＋ 一阶动量 m 4 ＋ 二阶动量 v 4（这三样合称优化器状态，12）
⛔ 这是「常规做法」的账，不是 DeepSeek-V3 实际的存法：V3 把 m、v 压成了 bf16，
   主权重和用于累积的梯度留在 fp32（技术报告 arXiv 2412.19437 sec. 3.3.3）。开场题第三问讲这件事。
"""
PSI = 671e9                     # DeepSeek-V3 总参数（config 现算 671.03B，见专题一）
N_DP = 128                      # 数据并行路数：V3 的 2,048 卡 ÷ PP 16（本课推导；技术报告 sec. 3.2 只给了 PP16、EP64、ZeRO-1）
# ⭐ 2026-09-26 现场：原来用 1,024 路示意，跟 V3 真实布局对不上；四库比方也要用同一个数，改成 128。
TIB, GIB = 1024 ** 4, 1024 ** 3

W_B, G_B = 2, 2                 # bf16 权重、bf16 梯度
MASTER_B, M_B, V_B = 4, 4, 4    # fp32 主权重、一阶动量、二阶动量
O_B = MASTER_B + M_B + V_B      # 优化器状态
PER_PARAM = W_B + G_B + O_B
assert (O_B, PER_PARAM) == (12, 16)
assert abs(PSI * PER_PARAM / TIB - 9.76) < 0.01          # 专题四 §5.1 的 9.76 TiB

# V3 全部训练的卡时（技术报告摘要与结论：2.788M H800 GPU hours，含预训练、长上下文扩展、后训练）
V3_GPU_HOURS = 2.788e6
V3_ONE_CARD_YEARS = V3_GPU_HOURS / (24 * 365)
assert abs(V3_ONE_CARD_YEARS - 318) < 1, V3_ONE_CARD_YEARS

# V3 的 AdamW β2（技术报告 sec. 4.2）；PyTorch AdamW 默认 0.999
V3_BETA2, TORCH_BETA2 = 0.95, 0.999
BF16_HALF_ULP = 2 ** -9         # bf16 有效位 8 位：相邻两数的相对间隔 2^-7 到 2^-8，四舍五入丢掉的是半个间隔
