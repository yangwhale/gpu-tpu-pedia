# -*- coding: utf-8 -*-
r"""专题六共用的承重数 —— 只有这一份。各节的图脚本和正文都从这里取，别各抄一份。

⛔ 只放查过出处的数；每个数旁边写出处。节内专用的数放在各自的图脚本里。
"""
GIB, TIB = 1024 ** 3, 1024 ** 4
GB, TB = 1e9, 1e12

# ── DeepSeek-V3（config.json，HF deepseek-ai/DeepSeek-V3）──────────────
V3_LAYERS = 61                  # num_hidden_layers
V3_HIDDEN = 7168                # hidden_size
V3_EXPERTS, V3_TOPK = 256, 8    # n_routed_experts / num_experts_per_tok（另有 1 个共享专家）
V3_KV_LORA, V3_ROPE = 512, 64   # kv_lora_rank / qk_rope_head_dim：MLA 每 token 每层只存 512 + 64 个数
V3_TOTAL, V3_ACTIVE = 671e9, 37e9   # 技术报告摘要：总参数 671B、每 token 激活 37B
V3_KV_PER_TOKEN = (V3_KV_LORA + V3_ROPE) * 2 * V3_LAYERS     # bf16，字节
assert V3_KV_PER_TOKEN == 70272
assert abs(V3_KV_PER_TOKEN * 131072 / GIB - 8.58) < 0.01      # 专题五 fig-kv-dup 的 8.58 GiB

# ── TPU v7（Ironwood，Google Cloud 公开规格：每芯片 FP8 4,614 TFLOP/s、HBM 192 GB、7.37 TB/s）──
V7_FP8 = 4614e12
V7_BF16 = V7_FP8 / 2            # 2,307 TFLOP/s（本课推导：BF16 是 FP8 的一半）
V7_HBM = 192 * GB
V7_BW = 7.37 * TB
V7_RIDGE = V7_BF16 / V7_BW      # BF16 屋脊点：每搬 1 字节要做多少次运算才不被带宽拖住
assert round(V7_RIDGE) == 313   # 专题二的 313
V7_DEVICES_PER_CHIP = 2         # 一颗 v7 芯片对软件显示成 2 个 device；并行度按 device 数
