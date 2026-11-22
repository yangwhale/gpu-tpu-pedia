# -*- coding: utf-8 -*-
"""专题九 · GPU→TPU 训练性能优化 —— 全讲的数字（唯一来源）。

⛔ 公开仓库闸门：这里只放数，不放作业名、机器名、仓库名、源码文件名。
   每个数的原始出处（哪一份日志、哪一份 profile）记在作者的私有出处台账里，这里只写口径。
⭐ 数字只在这里写一次；图脚本、课件、讲义都 import 这里。
"""

# ── 本讲的模型：公开的 Hy4（Hugging Face 模型卡 https://huggingface.co/tencent/Hy4，2026-11 开源）──────────────
# 只放结构；训练配方（序列长度、上下文并行、每设备 token、优化器、精度开关）和切片规模是项目数据，不在这里。
MC_TOTAL_B, MC_ACTIVE_B = 770, 49                # 语言主干总参 / 每 token 激活（十亿）
MC_LAYERS, MC_DENSE, MC_MOE = 78, 1, 77           # 主干层数：第 1 层稠密 FFN，其余 77 层 MoE
MC_EXPERTS, MC_SHARED, MC_TOPK = 256, 1, 8        # 每个 MoE 层：路由专家 / 共享专家 / 每个 token 挑几个
MC_HIDDEN, MC_EXPERT_FF = 6144, 2048              # 隐藏维 / 专家中间维
MC_HEADS, MC_QLORA, MC_KVLORA = 64, 2048, 512     # MLA：头数、q 压缩维、kv 压缩维
MC_IHC = 4                                        # iHC 残差流路数
MC_VIT_LAYERS, MC_VIT_HIDDEN = 40, 1536           # 视觉塔
MC_VOCAB = 120832
assert MC_DENSE + MC_MOE == MC_LAYERS
# 一个 MoE 层全部专家权重（BF16）：256 × 3 块 × 6144 × 2048 × 2 字节 ≈ 19.3 GB（本课推导）
MC_LAYER_EXPERT_GB = MC_EXPERTS * 3 * MC_HIDDEN * MC_EXPERT_FF * 2 / 1e9
assert abs(MC_LAYER_EXPERT_GB - 19.33) < 0.01, MC_LAYER_EXPERT_GB
# 本讲的两种测试底座：20 层（截前 20 层）／半层（39 层主干 ＋ 视觉塔 20 层 ＋ MTP 头，每设备负担和 128 片全量一样）
BASE_SH_LAYERS, BASE_DP_LAYERS, BASE_DP_VIT = 20, 39, 20
assert BASE_DP_LAYERS * 2 == MC_LAYERS and BASE_DP_VIT * 2 == MC_VIT_LAYERS

# ── 峰值：每设备 BF16 ─────────────────────────────────────────────
# 每设备 BF16 峰值取 1155 TFLOP/s：MaxText 日志和项目报告用的常数（v7x 每芯片公开值 2,307 ÷ 2 ＝ 1153.5，差 0.13%）。
PEAK_CHIP = 2307.0
PEAK_DEV = 1155.0

# ── 截层对照：一步多长、有用计算多少（实测，主机日志） ─────────────────
# 64 片 v7x，半层底座（用来近似更大规模时每个设备的负担）；真实图文数据，滤掉超过 8K 的长文档。
# 窗口：这一格跑 22 步，取第 13–20 步（前几步是预热，第 11 步抓 profile、第 12 步写 profile，都不计）。
CTL_STEPS_13_20 = [5.318, 5.304, 5.301, 5.309, 5.316, 5.314, 5.315, 5.313]           # 秒，逐步主机计时
CTL_EFF_TF_13_20 = [672.412, 673.011, 672.634, 672.235, 676.974, 677.510, 669.923, 673.487]   # 每设备每步有效运算量，TFLOP
CTL_STATIC_TF = 1142.672                       # 同一格按静态公式记的每设备每步运算量（第二节用）
CTL_STEP = sum(CTL_STEPS_13_20) / len(CTL_STEPS_13_20)
CTL_EFF_TF = sum(CTL_EFF_TF_13_20) / len(CTL_EFF_TF_13_20)
assert abs(CTL_STEP - 5.311) < 0.001, CTL_STEP
assert abs(CTL_EFF_TF - 673.5) < 0.1, CTL_EFF_TF
IDEAL_S = CTL_EFF_TF / PEAK_DEV                # 理想步时：有用计算全按峰值跑
CTL_MFU = CTL_EFF_TF / CTL_STEP / PEAK_DEV     # 有效 MFU
GAP_X = CTL_STEP / IDEAL_S
assert abs(IDEAL_S - 0.583) < 0.001, IDEAL_S
assert abs(CTL_MFU - 0.1098) < 0.0002, CTL_MFU
assert abs(GAP_X - 9.11) < 0.01, GAP_X

# ── 同一格、profile 里的一步（第 11 步，一颗芯片上的设备 0）：主核的时间去哪了 ──
# 每个叶子算子的自身时间按「在算 / 停着等通信」两类加总；两类之和与设备步之差是空闲。
PROF_DEVSTEP_MS = 5247.806
PROF_COMPUTE_MS = 3180.4                       # 主核在算（矩阵乘、注意力、胶水、优化器……）
PROF_WAIT_MS = 2066.0                          # 主核停着等集合通信（含同步集合）
PROF_IDLE_MS = PROF_DEVSTEP_MS - PROF_COMPUTE_MS - PROF_WAIT_MS
assert 0 < PROF_IDLE_MS < 3, PROF_IDLE_MS
HOST_MINUS_DEV_MS = CTL_STEP * 1000 - PROF_DEVSTEP_MS     # 主机一步比设备一步长出来的（本课推导：两个数取的步不同）
assert 55 < HOST_MINUS_DEV_MS < 70, HOST_MINUS_DEV_MS
COMPUTE_OVER_IDEAL = PROF_COMPUTE_MS / (IDEAL_S * 1000)
assert abs(COMPUTE_OVER_IDEAL - 5.45) < 0.02, COMPUTE_OVER_IDEAL

# ── 同一份 profile 里的一层反向（某一层 MoE），三条车道的时间线 ──
# 窗口从这一层「专家梯度 reduce-scatter（gate/up 那一组）」在小核 1 上开始算起，到下一层同一事件开始，79.6 ms。
# 只列 ≥ 0.3 ms 的事件。类：egrad＝专家梯度 reduce-scatter（FP32）；ew＝专家权重 all-gather；tok＝token 收发；
# dense＝稠密层权重 all-gather；small＝小集合与数据重排（每条约 0.3–1 ms）。
LAYER_MS = 79.6
SC1 = [(0.0, 16.2, "egrad", "专家梯度 RS（gate/up）"), (19.6, 0.4, "small", ""),
       (23.8, 0.4, "small", ""), (24.2, 0.4, "small", ""), (24.6, 0.4, "small", ""), (24.9, 0.4, "small", ""),
       (25.3, 0.4, "small", ""), (25.7, 0.4, "small", ""), (26.3, 0.3, "small", ""), (27.5, 1.0, "small", ""),
       (28.5, 0.9, "small", ""), (33.7, 1.6, "tok", ""), (37.6, 8.1, "ew", "专家权重 AG（gate/up）"),
       (46.3, 4.1, "ew", "AG（down）"), (50.9, 2.0, "tok", ""), (54.3, 1.7, "tok", ""),
       (61.5, 8.4, "egrad", "专家梯度 RS（down）"), (70.4, 2.1, "tok", "")]
SC0 = [(18.7, 0.4, "dense", ""), (19.6, 0.4, "small", ""), (23.8, 0.4, "small", ""), (24.2, 0.4, "small", ""),
       (24.6, 0.4, "small", ""), (24.9, 0.4, "small", ""), (25.3, 0.4, "small", ""), (25.7, 0.4, "small", ""),
       (26.3, 0.3, "small", ""), (26.6, 0.9, "small", ""), (27.5, 1.0, "small", ""), (28.5, 0.9, "small", ""),
       (29.6, 0.4, "dense", ""), (30.0, 0.4, "dense", ""), (76.1, 0.4, "dense", "")]
TC_WAITS = [(6.1, 10.1, "egrad"), (40.1, 5.6, "ew"), (47.8, 2.6, "ew"), (50.9, 2.0, "tok"),
            (54.3, 1.7, "tok"), (66.3, 3.6, "egrad"), (70.4, 2.1, "tok")]
LAYER_WAIT = sum(d for _s, d, _k in TC_WAITS)
LAYER_COMPUTE = LAYER_MS - LAYER_WAIT
SC1_BUSY = sum(d for _s, d, _k, _l in SC1)
SC0_BUSY = sum(d for _s, d, _k, _l in SC0)
assert abs(LAYER_WAIT - 27.7) < 0.05 and abs(LAYER_COMPUTE - 51.9) < 0.05, (LAYER_WAIT, LAYER_COMPUTE)
assert SC1_BUSY < LAYER_MS * 0.7, SC1_BUSY     # 小核 1 不满：一层 79.6 ms 里只忙约六成
# 同一份 profile、反向各层的中位数（逐层切出来再取中位）：每层 79.7 ms，主核在算 50.2、在等 29.5，小核 0 忙 9.3、小核 1 忙 52.3
MED_LAYER, MED_COMPUTE, MED_WAIT, MED_SC0, MED_SC1 = 79.7, 50.2, 29.5, 9.3, 52.3
assert abs(MED_COMPUTE + MED_WAIT - MED_LAYER) < 0.05

# 这一层里挑三条通信，看「藏住 / 露出」：(名字, 开始, 时长, 主核开始等的时刻 或 None)
HIDE = [("token 收发（all-gather）", 33.7, 1.6, None),
        ("专家权重 all-gather（gate/up）", 37.6, 8.1, 40.1),
        ("专家梯度 reduce-scatter（gate/up，FP32）", 0.0, 16.2, 6.1)]
for _n, _s, _d, _w in HIDE:
    if _w is not None:
        assert any(abs(w - _w) < 0.05 and abs(w + d - (_s + _d)) < 0.05 for w, d, _k in TC_WAITS), _n

# ── 开场大表要用的数（各节正文再细讲；原始出处都在私有出处台账）──────────────
SAME_RUN_STATIC_MFU, SAME_RUN_EFF_MFU = 18.64, 10.30      # 第二节：同一次运行两种口径（%）
PREPACK_EFF_BEFORE, PREPACK_EFF_AFTER = 325.18, 386.56    # 第二节：换预打包数据，有效分子（TFLOP，截层窗口）
NOOP_LO, NOOP_HI = 5.309, 5.310                           # 第二节：同一分配里四格程序与对照逐条相同
DSV3_DEEP_S, OURS_DEEP_S = 5.842, 5.470                        # 第三节：同条件，同样层数
LADDER = [("FP8 → BF16", 1.34), ("专用实现 → 通用实现", 1.49), ("序列 32K ＋ 上下文并行 8", 1.41),
          ("Muon 代替 AdamW", 1.19), ("FP32 精度开关", 1.11)]
LADDER_PROD = 1.0
for _n, _x in LADDER:
    LADDER_PROD *= _x
assert abs(LADDER_PROD - 3.72) < 0.01, LADDER_PROD
BLOCKS_HEAVY, BLOCKS_LIGHT = 66, 8                        # 第三节：只录一颗芯片时，两边那颗各算了多少个子块
ATTN_SAVED_MS, STEP_SAVED_MS = 672, 148                   # 第四节
LAYER_WAIT_AFTER = 8.5                                    # 第五节：每层反向主核等待 29.5 → 8.5 ms
SINGLE_A, SINGLE_B, COMBO = 6.5, 5.7, 16.9                # 第五节（%）
FUSED_SAVED_MS, FUSED_STEP_MS = 183, 107                  # 第七节：iHC 融合核（省约 183：本体 −84.7、相邻约 −70、零散约 −28）
WO_COMPUTE_MS, WO_WAIT_MS, CHAIN_DELTA_MS = 116, 136, 6   # 第六节：不重算
BUNDLE_OLD, BUNDLE_NEW = 41, 198                          # 第八节：热循环每趟指令包（0.0.46 对 0.0.53.1，离线单核）
A2A_N, A2A_MS = 97, 350                                   # 第八节：只取整到十位（跟 7.3 的 347.6 是不同的两次读数，不当两个数报）
NOOP_FLAGS, ALL_FLAGS = 5, 8                              # 第七节
INTERP_S, REF_S = 1.68, 1.346                             # 第七节：解释循环 vs 原样
NEWLIB_OLD_S, NEWLIB_NEW_S = 1.3464, 1.517                # 第八节：只换软件镜像
NEWLIB_SLOW = NEWLIB_NEW_S / NEWLIB_OLD_S - 1
assert abs(NEWLIB_SLOW - 0.127) < 0.001, NEWLIB_SLOW
BEST_S = 4.147                                            # 收尾：当前最好（第 5–9 步；待入台账）
CTL_S_K13 = 5.313
LINK_GBS = 186                                            # 第五节：专家权重链每设备带宽
AG2SC_CTL, AG2SC_S = 4.415, 4.467                        # 第五节：两颗小核一起收
BARRIER_S = 4.946                                         # 第六节：只加屏障（对照 5.311）
A2A_SAVED_MS = 64                                         # 第六节


def show(x):
    """正文里落地的写法。"""
    if isinstance(x, float):
        return ("%.3f" % x).rstrip("0").rstrip(".")
    return str(x)


# 正文常用写法
IDEAL_T = "%.2f" % IDEAL_S                 # 0.58
CTL_T = "%.2f" % CTL_STEP                  # 5.31
GAP_T = "%.1f" % GAP_X                     # 9.1
MFU_T = "%.1f%%" % (CTL_MFU * 100)         # 11.0%
EFF_TF_T = "%.1f" % CTL_EFF_TF             # 673.5
COMPUTE_T = "%.2f" % (PROF_COMPUTE_MS / 1000)   # 3.18
WAIT_T = "%.2f" % (PROF_WAIT_MS / 1000)         # 2.07


# ── 第二节：分子 ─────────────────────────────────────────────────────
# 截层，第 5–9 步：静态分子与有效分子的五项差（占静态分子的百分比）
NUM_STATIC_SH = 608.55
NUM_EFF_SH = 325.18
NUM_ITEMS = [("跨文档的注意力", -41.7), ("填充", -8.2), ("静态公式漏计的 MTP 输出头", +2.8), ("视觉塔（按实际图像块）", +0.6)]
_chk = NUM_STATIC_SH * (1 + sum(p for _n, p in NUM_ITEMS) / 100)
assert abs(_chk - NUM_EFF_SH) < 1.0, _chk            # 百分比是四舍五入的，差在 1 TFLOP 以内
NUM_EFF_FRAC_SH = NUM_EFF_SH / NUM_STATIC_SH
assert abs(NUM_EFF_FRAC_SH - 0.534) < 0.002
# 第一节那一格：静态 1142.672、有效 673.5
NUM_EFF_FRAC_HALF = CTL_EFF_TF / CTL_STATIC_TF
assert abs(NUM_EFF_FRAC_HALF - 0.589) < 0.002, NUM_EFF_FRAC_HALF
PAIR_FRAC_OLD, PAIR_FRAC_PK = 0.084, 0.112            # 文档内配对占整行因果：在线打包数据 / 预打包数据（第 5–9 步）
DOCS_PER_ROW = 28
KEPT_BLOCK_FRAC = 0.242                               # 预打包数据某一步平均每设备保留的子块占满因果（1024×1024 块）
PREPACK_STEP_OLD, PREPACK_STEP_NEW = 3.105, 3.031     # 截层，同一分配：在线打包 → 预打包，步时
PREPACK_PAD_OLD, PREPACK_PAD_NEW = 12.6, 0.36         # 填充（%）

# ── 第二节：尺子 ─────────────────────────────────────────────────────
# 同一分配（同一个作业、同一批机器）：对照 ＋ 四格程序与对照逐条相同（只差步数常量）
SAME_ALLOC = [("对照", 5.311), ("开关 A（程序相同）", 5.309), ("开关 B（程序相同）", 5.310),
              ("开关 C（程序相同）", 5.309), ("开关 D（程序相同）", 5.310)]
RULER_SAME = max(v for _n, v in SAME_ALLOC) - min(v for _n, v in SAME_ALLOC)
assert abs(RULER_SAME - 0.002) < 1e-9
RULER_SAME_PCT = RULER_SAME / 5.311 * 100
# 跨作业：同一配置、同一份数据，不同作业（不同时间、可能不同机器）里的对照格
CROSS_JOB = [5.300, 5.311, 5.313, 5.311, 5.313, 5.308, 5.310, 5.314, 5.315, 5.318]
RULER_CROSS = max(CROSS_JOB) - min(CROSS_JOB)
assert abs(RULER_CROSS - 0.018) < 1e-9
RULER_CROSS_PCT = RULER_CROSS / 5.309 * 100
LEGALIZER_S, SML_S = 5.166, 5.093                     # 同一分配里两条生效的开关
LEG_X = (5.311 - LEGALIZER_S) / RULER_SAME
assert abs(LEG_X - 72.5) < 0.1, LEG_X

# ── 第二节：测量陷阱 ─────────────────────────────────────────────────
FIRSTCELL_SLOW_PCT = 3.0       # 作业第一格整体偏慢的一次实例
PROFILE_STEP_EXTRA_S = 0.3     # 抓 profile 那一步主机多出的时间
PIPE_FAKE_S, PIPE_SLOW_S = 1.379, 2.0   # 数据管线供不上：那一格报 1.379 s，第 4–11 步约 2.0 s
VIT_INFL_BEFORE, VIT_INFL_AFTER = 27.5, 5.6   # 合成数据塞满图像槽：「加视觉塔」+27.5% → 槽数按真实比例 +5.6%

KEPT_BLOCK_FRAC_T = "%.1f%%" % (KEPT_BLOCK_FRAC * 100)
PAIR_FRAC_PK_T = "%.1f%%" % (PAIR_FRAC_PK * 100)


# ── 第三节：参照阶梯（公开模型 DeepSeek V3，64 片 v7x；每一步只改一个条件，前后两格都是实测）──
UP_README_S, UP_README_MFU = 4.24, 48.85            # 上游公开 README：61 层 ＋ MTP，全量 FP8，序列 8192，每设备 8192 token
UP_REPRO_S, UP_REPRO_MFU = 4.133, 50.15             # 我们在自己的 64 片 v7x 上复现（同一口径）：原样 → 关 bf16 传播 → 内存开关取上游 C4 值
UP_RAW_S, UP_NOBF16P_S = 5.473, 4.198              # 我们环境里原样跑；关 bf16 传播（上游 16K 配方用的开关）
assert abs(1 - UP_NOBF16P_S / UP_RAW_S - 0.233) < 0.001 and abs(1 - UP_REPRO_S / UP_NOBF16P_S - 0.015) < 0.001
LADDER_STEPS = [   # (条件, 前一格步时, 后一格步时, 在哪一格上量的)
    ("FP8 → BF16", 4.198, 5.631, "61 层、序列 8K"),
    ("上游专用实现 → 通用实现", 1.149, 1.712, "截 16 层、序列 4K、BF16"),
    ("序列 4K → 32K，加上下文并行 8", 1.712, 2.418, "同上"),
    ("加 FP32 精度开关（几处中间结果按 32 位算）", 2.418, 2.682, "在 32K 那格上单加"),
    ("AdamW → Muon", 2.418, 2.869, "在 32K 那格上单加"),
]
LADDER_X = [b / a for _n, a, b, _w in LADDER_STEPS]
for _x, _want in zip(LADDER_X, (1.34, 1.49, 1.41, 1.11, 1.19)):
    assert abs(_x - _want) < 0.006, (_x, _want)
LADDER_PROD2 = 1.0
for _x in LADDER_X:
    LADDER_PROD2 *= _x
assert abs(LADDER_PROD2 - 3.72) < 0.03, LADDER_PROD2
# 同条件直接对照：DeepSeek V3 结构叠满本讲模型的条件（同代码、同开关、32K ＋ CP 8、每设备 4096 token、同一批文档）
SAME_SH = (3.019, 11.11, 2.731, 12.25)    # DSv3 步时、有效 MFU，本讲模型步时、有效 MFU（较短的截层）
SAME_DP = (5.842, 10.30, 5.470, 11.11)    # 较长的截层
assert abs(1 - SAME_SH[2] / SAME_SH[0] - 0.095) < 0.002 and abs(1 - SAME_DP[2] / SAME_DP[0] - 0.064) < 0.002
# 按关键路径修正后，本讲模型相对 DeepSeek V3 多花／占便宜的（同样层数，ms，推断）
SELF_SLOW = [("iHC：4 路残差（含每层一次同步 all-reduce）", +248), ("注意力：数据因素", +116), ("视觉塔", +118),
             ("优化器", +83), ("专家矩阵乘与胶水（隐藏维更窄）", -180), ("对方那条代码路径的负载均衡同步", -221)]
SAME_TOTAL_DIFF_MS = -370    # 同条件总差：本讲模型快约 0.37 秒（较长的截层，按最忙设备修正后的估计）
# 只录一颗芯片：这一步、这颗芯片执行的子块数与前向耗时
BUSY_OURS_BLOCKS, BUSY_THEIRS_BLOCKS, BUSY_THEIRS_MAX = 66, 8, 46
FULL_CAUSAL_BLOCKS = 66                          # 32K ÷ 1024 = 32 块，32 × 33 ÷ 2 ÷ 8 = 66
FWD_MS_OURS, FWD_MS_THEIRS = 9.31, 2.37
FWD_SPEED_RATIO = 1.7                            # 按各自执行的块、按运算量折算的前向速度：本讲模型约是对方的 1.7 倍
FWD_TFS_OURS, FWD_TFS_THEIRS = 487, 290          # 按各自执行的块折算的前向速度（TFLOP/s）
GF_PER_BLOCK_OURS, GF_PER_BLOCK_THEIRS = 68.7, 85.9   # 每个子块前向的运算量（GFLOP），两边头数、头维不同
assert abs(BUSY_OURS_BLOCKS * GF_PER_BLOCK_OURS / FWD_MS_OURS - FWD_TFS_OURS) < 2
assert abs(BUSY_THEIRS_BLOCKS * GF_PER_BLOCK_THEIRS / FWD_MS_THEIRS - FWD_TFS_THEIRS) < 2
assert abs(FWD_TFS_OURS / FWD_TFS_THEIRS - FWD_SPEED_RATIO) < 0.05
NAIVE_SLOW_X = 2.3
UNIT_FAST_LO, UNIT_FAST_HI = 1.4, 1.6
# 开关不能照搬
FLAGS_ON_DSV3_PCT = 1.7        # 本讲模型的整套运行库开关放到 DeepSeek V3 上：慢 1.7%
NOBF16P_DSV3_PCT, NOBF16P_OURS_PCT = 23, 1.6   # 关 bf16 传播：DeepSeek V3 快 23%，本讲模型截层慢 1.6%
SAME_SH_0, SAME_SH_1, SAME_SH_2, SAME_SH_3 = SAME_SH
SAME_DP_0, SAME_DP_1, SAME_DP_2, SAME_DP_3 = SAME_DP
# 「专用 → 通用」那一刀逐算子拆账（截 16 层，+561 ms）：那一格通用实现没开、我们配置里早有等价物的几项
X149_TOTAL_MS = 561
X149_CONFIG = [("矩阵乘分块与两份权重拼成一个核", 133), ("路由权重在缓冲上做乘法", 120),
               ("token 重排按最坏缓冲整块读写", 100), ("两份梯度在缓冲上相加", 46)]
X149_CONFIG_MS = sum(v for _n, v in X149_CONFIG)
X149_CONFIG_FRAC = X149_CONFIG_MS / X149_TOTAL_MS
assert X149_CONFIG_MS == 399 and abs(X149_CONFIG_FRAC - 0.71) < 0.01
X149_REST_MS = X149_TOTAL_MS - X149_CONFIG_MS   # 两种实现真正的差（本课推导：561 − 399）
# 逐项对账（私有拆账记录）：配置四项 399 ＋ 真差三项 53/75/78 ＋ 零碎 28 − 等待 26 − 同步集合 34 ＝ 573，对实测 561
X149_REAL = [("激活函数没有只算有效行的专用核", 53), ("专家权重转置拷贝", 75), ("注意力投影的融合核与块配置", 78)]
X149_MISC, X149_WAIT, X149_SYNC = 28, -26, -34
X149_SUM = X149_CONFIG_MS + sum(v for _n, v in X149_REAL) + X149_MISC + X149_WAIT + X149_SYNC
assert X149_SUM == 573 and X149_REST_MS == 162


# ── 第四节：滤掉长文档 ───────────────────────────────────────────────
CAP_PK_S, CAP_CAP_S = 5.448, 5.300            # 同一分配，第 12–19 步：原数据 → 滤掉超过 8K 的长文档
CAP_SAVED_MS = round((CAP_PK_S - CAP_CAP_S) * 1000)
assert CAP_SAVED_MS == 148
CAP_PCT = (1 - CAP_CAP_S / CAP_PK_S) * 100
assert abs(CAP_PCT - 2.7) < 0.05
BUSY_BLOCKS_PK, BUSY_BLOCKS_CAP = 59.5, "13–18"   # 第 12–19 步最忙设备的子块数（原数据平均；滤长以后的范围）
CAP_UPPER_MS = 578                            # 按每子块实测耗时 × 层数推的上限（推断）
# 同一配置，一颗最忙的芯片（66 个子块）对一颗轻的芯片（7 个子块），各自 profile 里的一步
HL_ATTN = (896, 223)
HL_WAIT = (518, 1059)
HL_STEP_DIFF = 195
HL_HEAVY_STEP = 5444.7
HL_LIGHT_STEP = HL_HEAVY_STEP - HL_STEP_DIFF
assert abs((HL_ATTN[0] - HL_ATTN[1]) - 672) <= 1   # 两个读数各自四舍五入，原文写 −672
HL_NET = (HL_ATTN[0] - HL_ATTN[1]) - (HL_WAIT[1] - HL_WAIT[0])
assert abs(HL_NET - 131) <= 2, HL_NET
BUF_PK, BUF_CAP = "1.27–2.30", "1.26–1.41"    # 所需缓冲倍数最大值：滤长以后 MoE 更均衡
# 旁证：同一个现象另外两次
SIDE_SMALL = (1087, 39)       # 同项目另一个较小的模型：注意力省 1087 ms，一步只省 39 ms
SIDE_EARLY = (340, 78)        # 本模型更早一档（小核队列满的时候）：注意力反向跳块，同一对格的 profile 里注意力两颗核少算约 340 ms（反向 361.8→105.2、前向 122.1→38.1，合计 340.6），步时只快 78 ms（3.574→3.496 s）
SIDE_SMALL_0, SIDE_SMALL_1 = SIDE_SMALL
SIDE_EARLY_0, SIDE_EARLY_1 = SIDE_EARLY


# ── 第五节：链卡在链路上 ─────────────────────────────────────────────
# 同一类集合通信的五种组合：每设备实收字节（MB）、单次耗时（ms）
LINK_ROWS = [("专家权重收集，BF16，单核", 755, 4.06), ("专家梯度规约，BF16，双核", 755, 4.04),
             ("专家权重收集，BF16 两份拼一起，单核", 1510, 8.13), ("专家梯度规约，FP32，单核", 3020, 16.19),
             ("专家梯度规约，FP32 另一份，单核", 1510, 8.18)]
LINK_BW = [mb / ms for _n, mb, ms in LINK_ROWS]       # MB/ms ＝ GB/s
assert all(184 <= x <= 188 for x in LINK_BW), LINK_BW
AG2SC_PCT = (AG2SC_S / AG2SC_CTL - 1) * 100
assert abs(AG2SC_PCT - 1.2) < 0.05
# 砍一刀：梯度 BF16 ＋ 两条生效开关（同一分配，第 5–9 步对对照第 13–20 步）
COMBO_CTL, COMBO_S = 5.311, 4.413
# 反向每层（各层中位数）：对照 → 合开
LAYER_BEFORE = dict(interval=79.7, compute=50.2, wait=29.5, sc0=9.3, sc1=52.3)
LAYER_AFTER = dict(interval=58.4, compute=49.9, wait=8.5, sc0=21.6, sc1=40.6)
RS_WALL_BEFORE, RS_WALL_AFTER = 24.6, 12.0        # 每层专家梯度规约在小核上的墙钟（ms）
BWD_WAIT_BEFORE, BWD_WAIT_AFTER = 1127.3, 247.1   # 整步：反向段主核等通信（ms）
CONCAT_COST_MS = 197.5                            # 代价：两份权重改在运行时拼接，主核多算
# 一加一大于二？
PK_SINGLE_A, PK_SINGLE_B = 6.5, 5.7               # 旧数据上单测（%）
F5_CAP_CTL, F5_CAP_S = 5.308, 4.650               # 新数据上重测梯度 BF16（另一个作业）
F5_CAP_PCT = (1 - F5_CAP_S / F5_CAP_CTL) * 100
assert abs(F5_CAP_PCT - 12.4) < 0.05
LEG_PCT, SML_PCT = 2.7, 4.1
PRED_MULT = COMBO_CTL * (F5_CAP_S / F5_CAP_CTL) * (1 - LEG_PCT / 100) * (1 - SML_PCT / 100)
assert abs(PRED_MULT - 4.34) < 0.01, PRED_MULT
RS_WAIT_PK, RS_WAIT_CAP = 153.2, 499.7            # 专家梯度规约的主核等待：旧数据最忙那颗 vs 新数据
NOPREC_CTL, NOPREC_S = 5.313, 4.745               # （正文不用 4.759 当实测：它是三项单测相减推出来的）不含精度改动的一组：两条开关 ＋ iHC 里 MoE 分支的输出不重算
WO_SINGLE_S = 5.120
NOPREC_PRED = NOPREC_CTL - (5.311 - LEGALIZER_S) - (5.311 - SML_S) - (5.311 - WO_SINGLE_S)
assert abs(NOPREC_PRED - 4.759) < 0.002, NOPREC_PRED
# 藏 vs 缩
CHAIN_SC_MS, CHAIN_EXPOSED_MS = 1389, 381
FWD_SC0_BUSY, FWD_TC_COMPUTE, FWD_SC1_BUSY = 18.57, 18.65, 0.59   # 前向每层：小核 0 的队列和主核计算一样长
SPLIT_CTL, SPLIT_S = 4.415, 4.280
SPLIT_PCT = (1 - SPLIT_S / SPLIT_CTL) * 100
assert abs(SPLIT_PCT - 3.1) < 0.05


# ── 第六节：调度 ─────────────────────────────────────────────────────
BAR_CTL_S, BAR_S = 5.311, 4.946               # 只加一道屏障（另一个作业的同配置对照 → 本作业）
BAR_PCT = (1 - BAR_S / BAR_CTL_S) * 100
assert abs(BAR_PCT - 6.9) < 0.05
# profile 第 11 步按时间段切（ms）：对照 → 加屏障
# 三段大的（前向循环、反向循环、优化器段）只公开变化，不公开绝对毫秒：
# 绝对值和每层毫秒、整步一凑就能反推层数。两段小的给绝对值。
# 五段都只给变化的比例（绝对毫秒、每层毫秒、整步一凑就能反推层数）
BAR_PHASES = [("前向之前", -37), ("前向循环", 0), ("前后向之间", +8), ("反向循环", -12), ("优化器段", +4)]   # %
BAR_PHASES_MS = [("前向之前", 180.1, 114.4), ("前向循环", 920.8, 920.8), ("前后向之间", 142.2, 153.7),
                 ("反向循环", 2950.0, 2599.4), ("优化器段", 1055.6, 1094.9)]
assert abs(sum(a for _n, a, _b in BAR_PHASES_MS) - 5248.7) < 0.2 and abs(sum(b for _n, _a, b in BAR_PHASES_MS) - 4883.2) < 0.2
for (_n, _p), (_n2, _a, _b) in zip(BAR_PHASES, BAR_PHASES_MS):
    assert _n == _n2 and abs((_b / _a - 1) * 100 - _p) < 1.0, (_n, _p, _a, _b)
BAR_SC1_BWD = (1935.7, 1936.0)                # 反向窗口里小核 1 的忙时：一点没少
BAR_LAYER = (79.7, 70.3)                      # 反向每层（ms）
BAR_BWD_CUT_PCT = 12
BAR_SC1_DIFF_PCT = 0.02                       # 反向窗口里小核 1 的忙时：加屏障前后只差 0.02%（只给比例：整步合计和每层中位数一凑就是层数）
BAR_LAYER_CUT_PCT = 11.8                      # 反向每层短了约一成（只给比例）
BAR_BRACKET = (801, 1261)                     # 反向循环里，每条小核通信「放行到完成」之间夹的计算，合计
BAR_EMPTY = (18, 1)                           # 括号里一个计算都没有的条数
BAR_HOIST_GIB = 7.125                         # 被提前到前向之前、一直活到优化器段的那块数据
BAR_LIVE_GIB = (48.26, 42.23)                 # 反向循环之前产出、之后才用的数据合计
BAR_MEM_SAVED = 3.0                           # 正常步峰值显存少约 3 GiB
ICI1_S = 5.059                                # 反例：限制同时在飞的通信数（在加屏障 ＋ 卸载那一格 4.882 上），+3.6%
ICI1_BASE = 4.882
ICI1_BRACKET = (1261, 620)
# 一颗融合核：核本身省、调度亏
FUSE_CTL, FUSE_S = 5.313, 5.420
FUSE_PCT = (FUSE_S / FUSE_CTL - 1) * 100
assert abs(FUSE_PCT - 2.0) < 0.05
FUSE_SAVE_MS = 183                            # 本体 −84.7、相邻融合约 −70、零散约 −28
FUSE_INSTR = (42, 1)                          # 那条梯度规约「放行到完成」之间夹的指令：原来 42 条（含注意力反向核），开核以后 1 条
FUSE_WAIT_RATIO = 1.58                         # 那条规约每层的等待：开核后是原来的 1.58 倍
FUSE_LAYER_WAIT = (10.1, 16.0)                # 每层等待（ms）
FUSE_BWD_MS, FUSE_FWD_MS, FUSE_AR_MS = 217, 55, 28   # 各层合计多露 / 前向一条收集 / 同步 all-reduce（ms）
FUSE_NET_MS = 107                                 # 5.420 − 5.313（分项 217＋55＋28−183＝117，差在拆账误差）
FUSE_OTHER_PCT = -0.2                         # 另一种截层上这颗核：误差内
# 两件小的
WO_CTL, WO_S = 5.311, 5.120
WO_STACK = (4.418, 4.256)
WO_MEM_GIB, WO_FULL_GIB, MEM_LIMIT_GIB = 1.70, 96.67, 94.74
A2A_TC_MS, A2A_WAIT_MS, A2A_SC_MS = 347.6, 187, 361
WO_STACK_0, WO_STACK_1 = WO_STACK


# ── 第七节：编译器与运行库 ───────────────────────────────────────────
# 一份参照配方的复现：离线编译机是 CPU 时一颗自定义核退化成解释循环
INTERP_ITERS = 96
# 只换软件镜像：1.3464 → 1.5166 s（NEWLIB_*）；profile 里一步 1329.4 → 1499.5 ms
NEWLIB_DEV = (1329.4, 1499.5)
SCK_PER_CALL = (1.68, 5.39)        # 那颗小核上的核每次执行（ms）
SCK_TCWAIT = (22.0, 132.2)         # 主核等它做完（ms/步）
# 版本二分：同一颗核离线单独编译，热循环每个向量乘摊到几条指令包
# 默认行为变了：all-to-all
A2A_FORCE_GIB = 8.1                # 新版本上强行交给小核，峰值多 8.1 GiB
A2A_BAR_SAVED_GIB = 3.0            # 优化器前加屏障以后全交给小核，还省约 3 GiB
# 看产物：两个名字、一个机制
TWIN_GIB = 90.57
TWIN_S = 5.120
# 离线窗口必要不充分（同项目另一个较小的模型）
WIN_BETTER_PCT, WIN_SAME_PCT = +0.8, -5.5


# ── 第八节：收尾 ─────────────────────────────────────────────────────
FINAL_CTL, FINAL_S = 5.311, 4.147            # 对照 → 目前最好（另一个作业，第 5–9 步；待复测入账）
FINAL_PCT = (1 - FINAL_S / FINAL_CTL) * 100
assert abs(FINAL_PCT - 21.9) < 0.05
FINAL_MFU = 14.05                             # 有效 MFU（旧口径，未用于正文）
FINAL_MFU_T = "%.1f%%" % FINAL_MFU


# ── 2026-11-21 晚补：换一份数据（长文：八成多的行是一整篇长文） ──────────────
LONG_CTL = (5.311, 10.98, 5.553, 17.83)      # 同一配置：短文对照 步时/有效 MFU → 长文 步时/有效 MFU
LONG_STEP_PCT = (LONG_CTL[2] / LONG_CTL[0] - 1) * 100
assert abs(LONG_STEP_PCT - 4.6) < 0.05
LONG_EXTRA_MS, LONG_EXTRA_ATTN_MS, LONG_CHAIN_LESS_MS = 748, 743, 110   # 长文比短文多的设备步；其中落在注意力两颗核上的；专家权重链露出少的
# 三把刀在两份数据上（各自同作业对照，步时 秒）
FLIP = [("iHC 融合核", (5.313, 5.420), (4.788, 4.735)),
        ("token 和专家权重分到两颗小核", (4.415, 4.280), (4.839, 4.804)),   # 短文：第六节同作业单测；长文：跨作业，只看方向
        ("片上暂存上限放到 64 MiB", None, (4.787, 4.678))]
FLIP_PCT = [(n, (s[1] / s[0] - 1) * 100 if s else -0.4, (l[1] / l[0] - 1) * 100) for n, s, l in FLIP]
assert [round(a, 1) for _n, a, _b in FLIP_PCT] == [2.0, -3.1, -0.4], FLIP_PCT
assert [round(b, 1) for _n, _a, b in FLIP_PCT] == [-1.1, -0.7, -2.3], FLIP_PCT
LONG_BEST_S, LONG_BEST_MFU = 4.657, 21.32    # 长文上三样叠起来的当前最好
# 片上暂存上限放到 64 MiB：整步按类拆账（同一颗芯片第 11 步，ms）
VMEM_STEP = (4728.9, 4613.2)
VMEM_ROWS = [("注意力两个 kernel", +9.2), ("编译器自己的矩阵乘融合", -67.2), ("优化器的同步集合等待", -38.7),
             ("MoE 的同步集合等待", -30.4), ("其余各类合计", None)]
VMEM_OTHER = round((VMEM_STEP[1] - VMEM_STEP[0]) - sum(v for _n, v in VMEM_ROWS if v is not None), 1)
assert abs(VMEM_OTHER - 11.4) < 0.05, VMEM_OTHER
VMEM_PRED_ATTN_PCT = 0.25                     # 离线指令数模型预测注意力核本身对整步的收益
# 离线指令数模型：预测单颗核 vs 上卡整步
PRED_VS_REAL = [("反向块 q 改 2048", -3.5, (4.672, 4.966)), ("反向块 q 2048、kv 4096", -3.7, (4.678, 5.544))]
assert [round((b / a - 1) * 100, 1) for _n, _p, (a, b) in PRED_VS_REAL] == [6.3, 18.5]
# 数据加载 worker（一个约 2 秒一步的短截层配置）
DL_W1 = (2.037, 2.044); DL_W4 = (1.374, 1.133); DL_WAIT_MS = 636   # 9 层的倒序阶梯配置；那台主机每步等数据 636 ms
LONG_CTL_1, LONG_CTL_3 = LONG_CTL[1], LONG_CTL[3]
PAIR_FRAC_SHORT, PAIR_FRAC_LONG = 0.057, 0.971   # 注意力有效配对占满因果的比例：短文 / 长文（数据重放）
STACK_SAVE_MS, STACK_RS_MS, STACK_AR_MS, STACK_NET_MS = 156, 121, 82, 25   # 长文、暂存上限之上再叠融合核：省的计算 / 露出规约 / 同步 AR / 整步净省
assert STACK_SAVE_MS == 81 + 40 + 22 + 13
PRED_OK = (-12.4, -10.0)                     # 只改计算、不改搬运的一次核改动：预测前向 / 单芯片真机前向（9.12 → 8.21 ms）
assert abs((8.21 / 9.12 - 1) * 100 - PRED_OK[1]) < 0.1
DKV_EFF_WRONG, DKV_EFF_RIGHT = 33, 72          # 注意力反向核效率：漏数一半 FLOP 时 / 按 5 个矩阵乘数对以后（% 峰值）


# ── 2026-11-22 补：第八节（长文）两条新线、第九节（准入）────────────────────────
# ⛔ 口径同上：数只写在这里；原始出处（哪个作业、哪份 profile、哪份编译产物）记在作者的私有出处台账。

# 8.2 优化器里白搬的字节（Muon）：长文数据、64 片 v7x、半层底座，对照 ＝ 长文上开了片上暂存上限的那一格
# 每个专家叶子（gate、up、down 三块之一）每步在优化器段的大 all-to-all（离线编译产物逐条追操作数）。
# 讲法用「份」：一块的 BF16 动量算 1 份，FP32 主权重同形状、是 2 份；绝对值（64 片、每设备）也给（11-22 起芯片数可写）
MU_MOM_U, MU_PARAM_U = 1, 2
MU_MOM_GIB, MU_PARAM_GIB = 1.78, 3.56            # 一块：BF16 动量 / FP32 主权重（每设备，离线编译产物）
MU_LEAF_BEFORE_GIB = MU_MOM_GIB + 2 * MU_PARAM_GIB   # 8.9
MU_LEAF_DESIGN_GIB = 2 * MU_MOM_GIB                  # 3.56
MU_STEP_BEFORE_GIB, MU_STEP_DESIGN_GIB = 3 * MU_LEAF_BEFORE_GIB, 3 * MU_LEAF_DESIGN_GIB   # 26.7 / 10.7
assert abs(MU_LEAF_BEFORE_GIB - 8.9) < 1e-9 and abs(MU_STEP_BEFORE_GIB - 26.7) < 1e-9 and abs(MU_STEP_BEFORE_GIB - MU_STEP_DESIGN_GIB - 16.0) < 0.05
MU_SC_MS = (20.5, 41.0)                          # 小核上每条：BF16 动量那条 / FP32 主权重那条（ms）
MU_A2A_MS = 307.5                                # 三块九条合计的小核耗时（ms）
assert abs(3 * (MU_SC_MS[0] + 2 * MU_SC_MS[1]) - MU_A2A_MS) < 0.05
MU_A2A_STATIC = (29.59, 12.52)                   # 整步 all-to-all 字节（GiB／设备）：修前 / 正式修法后
MU_F32_A2A = (22.93, 0.0)                        # 其中 FP32 的（38 条，含专家以外的权重）
MU_F32_AG = (18.19, 0.53)                        # 整步 FP32 all-gather（GiB）
MU_GATE_AG_GIB = 23                              # 按头切开的那几类叶子原来「全收再切」搬 FP32，合计约 23 GiB（项目报告：整步 all-gather 77.66 → 54.73 GiB，少 22.93；早先记的「约 21」是旧数）
MU_MEM = (89.17, 81.89)                          # 半层正常步峰值显存（GiB）
assert abs((1 - MU_MEM[1] / MU_MEM[0]) * 100 - 8.2) < 0.1
MU_LEAVES = 3
MU_LEAF_BEFORE = MU_MOM_U + 2 * MU_PARAM_U       # 动量过去 ＋ 主权重过去 ＋ 新主权重回来 ＝ 5 份
MU_LEAF_DESIGN = 2 * MU_MOM_U                    # 按设计：BF16 动量过去、BF16 结果回来 ＝ 2 份
MU_RATIO = MU_LEAF_BEFORE / MU_LEAF_DESIGN       # 2.5 倍
assert (MU_LEAF_BEFORE, MU_LEAF_DESIGN, MU_RATIO) == (5, 2, 2.5)
MU_SC_FP32_X = 2                                 # 小核上一条 FP32 的 all-to-all 耗时约是 BF16 那条的 2 倍（只给比值：毫秒和链路速度一凑就是每块的字节）
MU_A2A_WAIT_MS = 148.6                           # 专家那一组里主核停着等这几条 all-to-all 的毫秒（profile 第 11 步）
MU_SEG_MS = 742.3                                # 优化器段（视觉塔反向结束到步尾）
# 修好以后（结果先放回参数自己的切法、再加到 FP32 主权重上）：离线编译产物整步统计，只给比例
MU_A2A_LEFT_PCT = 42                             # 整步 all-to-all 字节剩原来的约 42%（FP32 的 all-to-all 全没了）
MU_F32_AG_CUT_PCT = 97                           # 整步 FP32 all-gather 少了约 97%（另一类按头切开的叶子原来用「全收再切」搬 FP32，一并去掉）
MU_MEM_CUT_PCT = 8                               # 正常步峰值显存少约 8%（只给比例）
MU_CTL, MU_S = 4.673, 4.377                      # 同一作业两格，第 5–9 步
MU_PCT = (1 - MU_S / MU_CTL) * 100
assert abs(MU_PCT - 6.3) < 0.05, MU_PCT
MU_D2 = 5e-4                                     # 正式修法从第 2 步起 loss 差约 5e-4（第 0、1 步逐位相同）

# 8.3 省了计算，收集露出来：同一颗 iHC 融合核，长文，对照已把注意力输出投影存下来、反向少重算一步
IH_CTL, IH_S = 4.351, 4.692                      # 对照（三个作业 4.347 / 4.347 / 4.351）→ 加融合核，第 13–20 步
IH_PCT = (IH_S / IH_CTL - 1) * 100
assert abs(IH_PCT - 7.8) < 0.05, IH_PCT
IH_CTL2, IH_S2 = 4.421, 4.768                    # 换一个底座（不开另一个转置开关）重测
assert abs((IH_S2 / IH_CTL2 - 1) * 100 - 7.8) < 0.05
IH_DEV = (4258, 4606)                            # profile 里的设备一步（ms）
IH_COMPUTE_SAVED = 190                           # 主核计算少了约（ms）
IH_WIN_MS = ((7.2, 9.0, 15.2), (0.4, 0.7, 0.7))  # 重算段里 gate / up / down 三条收集「放行到完成」之间夹的主核计算（各层中位，ms）
IH_EXPOSE_EACH = (3.6, 3.9)                      # 加核以后每条露出（ms）
IH_AG_EXPOSED = (64, 560)                        # 专家权重收集的露出（ms/步，一颗芯片的一个主核）
IH_AG_EXPOSED_0, IH_AG_EXPOSED_1 = IH_AG_EXPOSED
IH_WIN_KERN = ((4, 5, 6), (1, 2, 3), (2, 3, 5))  # 窗口里夹的核：对照 / 加融合核 / 再把收集的延迟估计调大一倍（编译产物指令顺序）
IH_KNOB = (4.693, 4.630)                         # 融合核那一格（另带一项小改动）→ 再调大延迟估计
IH_KNOB_GAP = round((IH_KNOB[0] - IH_CTL) * 1000)            # 342：要救的
IH_KNOB_BACK = round((IH_KNOB[0] - IH_KNOB[1]) * 1000)       # 63：救回的
assert IH_KNOB_GAP == 342 and IH_KNOB_BACK == 63 and abs(IH_KNOB_BACK / IH_KNOB_GAP - 1 / 6) < 0.02

# 9.5 Pallas 核报的耗时估计，决定括号里填不填计算（定稿之后的一组实验；30 步，取第 13–20 步）
#   对照 ＝ 定稿再叠一项路由的小改动（不在编年史里）；八格都叠在它上面，各跑在不同的常驻机器上（跨机噪声约 0.4%）。
#   出处：作者私有调优记录 §15.2（核对表）、§15.3（调度导出）、§15.5（上卡八格）、§15.5b（两道轻检查）。
CE_CELLS = [   # (键, 改了什么, 实测步时 s, 调度器自己估的整步 ms；None ＝ 没导出)
    ("ctl", "对照", 4.088, 5167),
    ("cal", "专家矩阵乘、注意力核的估计按实测校准", 4.474, 3965),
    ("agm", "小核收集的延迟估计按实测调短", 4.325, 5053),
    ("calagm", "两样一起校准", 4.337, 3870),
    ("fk", "换上融合核（核报耗时）", 4.522, 5024),
    ("fkm", "融合核，报的耗时按实测调准", 4.522, 5050),
    ("fk0", "融合核，不报耗时", 3.931, 4986),
    ("calfk0", "融合核不报耗时 ＋ 同样校准", 4.283, None),
]
CE = {k: (s, e) for k, _n, s, e in CE_CELLS}
CE_CTL = CE["ctl"][0]
CE_PCT = {k: (s / CE_CTL - 1) * 100 for k, (s, _e) in CE.items()}
for _k, _p in (("cal", 9.4), ("agm", 5.8), ("calagm", 6.1), ("fk", 10.6), ("fkm", 10.6), ("fk0", -3.8), ("calfk0", 4.8)):
    assert abs(CE_PCT[_k] - _p) < 0.05, (_k, CE_PCT[_k])
CE_IHC_T, CE_IHC0_T, CE_CTL_T = "%.3f" % CE["fk"][0], "%.3f" % CE["fk0"][0], "%.3f" % CE_CTL
CE_SCHED_CTL, CE_SCHED_CAL = CE["ctl"][1], CE["cal"][1]          # 5167 → 3965：校准以后调度器自己估的整步跟实测（约 4.0–4.1 秒）对上了
assert CE_SCHED_CTL / (CE_CTL * 1000) > 1.25 and abs(CE_SCHED_CAL / (CE_CTL * 1000) - 1) < 0.04
# 拼好的 gate/up 那条专家权重收集，每步露出（ms，profile 第 11 步一颗芯片一个主核，按名字合计）：
#   对照取同配置另一作业的 profile（41.5）；去掉估计那格只知道「不在露出前列、不到 14」。
CE_AG_EXPOSED = (41.5, 290.5, 14)
CE_WO_EXPOSED = (85.4, 150.8, 77.4)                               # down 那条（对照 / 融合核报耗时 / 不报耗时）
# 调度器自己的时间线（调度导出，单位 ms，全是它的估计）：反向一层里两条专家权重收集「放行 → 完成」窗长 / 窗里排进的计算
CE_WIN = {"ctl": ((26.1, 10.4), (15.0, 15.0)), "fk": ((26.2, 0.48), (10.2, 0.66)),
          "fkm": ((26.0, 0.55), (10.1, 0.77)), "fk0": ((34.8, 11.9), (23.7, 17.0))}
CE_WASTED = {"ctl": 5.8, "fk": 21.8, "fkm": 21.6, "fk0": 4.2, "cal": 13.3, "agm": 2.8, "calagm": 2.4}   # 调度器自己记的空等（反向一层，ms）
CE_FILL_BACK = CE_WIN["fk0"][0][1]                               # 11.9：去掉估计后，gate/up 那条括号里重新排进的计算
assert round(CE_FILL_BACK) == 12
CE_IHC_BUSY_MED = 0.65                                            # 实测：融合核报耗时那格，gate/up 括号里主核真在干活的时间（各层中位，ms）
CE_CTL_BUSY = (6.7, 7.0)                                          # 实测：对照里同一个括号里主核真在干活（ms）
# 核对表：调度器按核报的数换算出的时长 ÷ 实测（每个主核、单次），实测取对照那格 profile；融合核的实测来自同形状的另一格
CE_RATIO_ROWS = [  # (组, 名字, 调度器估 ms, 实测 ms)
    ("gmm", "专家矩阵乘 ①", 6.62, 2.71), ("gmm", "专家矩阵乘 ②", 6.62, 2.62), ("gmm", "专家矩阵乘 ③", 3.31, 1.37),
    ("gmm", "专家矩阵乘 ④", 3.31, 1.32), ("gmm", "专家权重梯度 ①", 6.62, 2.86), ("gmm", "专家权重梯度 ②", 3.31, 1.15),
    ("attn", "注意力前向（主干）", 8.83, 3.55), ("attn", "注意力反向（主干）", 11.73, 5.31),
    ("vit", "注意力前向（视觉塔）", 0.28, 0.82), ("vit", "注意力反向（视觉塔）", 0.69, 1.47),
    ("fk", "融合核 ①", 0.068, 0.135), ("fk", "融合核 ②", 0.123, 0.166), ("fk", "融合核 ③", 0.082, 0.156), ("fk", "融合核 ④", 0.178, 0.240),
    ("ag", "小核上的专家权重收集", 26.0, 8.2),
]
CE_RATIO = {}
for _g, _n, _e, _m in CE_RATIO_ROWS:
    CE_RATIO.setdefault(_g, []).append(_e / _m)
CE_RANGE = {g: (min(v), max(v)) for g, v in CE_RATIO.items()}
assert 2.3 < CE_RANGE["gmm"][0] and CE_RANGE["gmm"][1] < 2.9, CE_RANGE["gmm"]
assert 2.2 < CE_RANGE["attn"][0] and CE_RANGE["attn"][1] < 2.5, CE_RANGE["attn"]
assert 0.5 < CE_RANGE["fk"][0] and CE_RANGE["fk"][1] < 0.75, CE_RANGE["fk"]
assert 3.1 < CE_RANGE["ag"][0] < 3.3
CE_IHC_EST_MS = (0.068, 0.178)                                    # 融合核四颗各报的耗时换算成时长（ms）：很小
CE_NOEST_KERNELS = 3                                              # 对照里本来就不报耗时的核（token 收发两颗、注意力反向清零一颗），实测各 0.3–0.6 ms
CE_LOSS_IHC_VS_CTL = 9.7e-5                                       # 融合核本身对原写法：30 步 loss 最大相对差（核的累加顺序不同）

# 9.5 续：层调度器的配对规则与「门槛」扫描（定稿之后；出处：作者私有《Pallas 核成本估计与通信重叠：通用做法》§1、§3、§5，
#   以及成本估计线的门槛扫描排名表（4 层 64／128 片拓扑、39 层复核）。数值是客观判据的整步预测，不是上卡读数；
#   128 片拓扑那条用的是 64 片 profile 的通信时长表：形状可信，绝对值不可信）
#   读法：所有核照实填耗时估计（acc），小核收集的延迟估计 ×0.32（接近实测 26 → 约 8 ms）；横轴 ＝ Pallas 核入选候选的门槛（ms，按估计时长）；
#   "def" ＝ 默认门槛（极低，约 1 µs，带估计的核几乎都能入选）。v4 ＝ 现在的推荐配置（融合核不报、其余照旧、延迟估计不调）。
CE_TH_MS = ["def", 0.2, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 10]
CE_TH64 = [3336.8, 3084.9, 3095.4, 2997.6, 2988.5, 2881.2, 2881.2, 2708.7, 2819.2, 2945.2, 2945.2, 2945.2]
CE_TH128 = [3343.8, 3071.8, 3103.9, 3006.0, 2997.9, 2892.7, 2892.7, 2696.4, 2804.4, 2935.5, 2935.5, 2935.5]
CE_TH_V4 = (2922.2, 2915.4)                      # v4 的整步预测（64 片 / 128 片拓扑）
assert min(CE_TH64) == CE_TH64[CE_TH_MS.index(3)] and min(CE_TH128) == CE_TH128[CE_TH_MS.index(3)]   # 3 ms 两条都最好
assert CE_TH64[-1] == CE_TH64[-3] and CE_TH128[-1] == CE_TH128[-3]                                   # 6 ms 以上不变：没有锚点
assert abs(CE_TH64[-1] / CE_TH_V4[0] - 1) < 0.01 and abs(CE_TH128[-1] / CE_TH_V4[1] - 1) < 0.01        # 退回 v4 的水平（差 1% 以内）
assert abs(CE_TH64[3] - CE_TH64[4]) < 10 and abs(CE_TH128[3] - CE_TH128[4]) < 10                     # 0.75–1 ms 平台
CE_TH_BEST = (CE_TH64[7], CE_TH128[7])           # 2708.7 / 2696.4
CE_AG_MULT = 0.32                                # 小核收集延迟估计的校准倍数（26 ms × 0.32 ≈ 8.3，实测约 8.2）
assert abs(26.0 * CE_AG_MULT - 8.2) < 0.2
CE_TH_GMM_MS = (1.4, 2.7)                        # 照实填以后，专家矩阵乘、专家权重梯度那几颗核的估计时长范围（被 3 ms 门槛挡在外面）
# 上卡（作者私有调优记录（耗时估计专题与上卡确认）；同一台常驻机器复测，30 步取第 13–20 步）
CE_TH_ONCARD = 3.868                             # 照实填 ＋ 门槛 3 ms ＋ 延迟估计 ×0.32
CE_TH_ONCARD_CTL = 3.907                         # 同一台机器上的对照（推荐配置，已叠编译线后两刀）
CE_TH_PCT = (CE_TH_ONCARD / CE_TH_ONCARD_CTL - 1) * 100
assert abs(CE_TH_PCT + 1.0) < 0.05, CE_TH_PCT
CE_TH_PRED_PCT = -6.3                            # 判据（39 层）预测的是 −6.3%
CE_TH_4MS = (4.007, 2.5)                         # 门槛 4 ms 那格：4.007 s，+2.5%
CE_TH_WAIT = (1013, 1003)                        # 每步主核停在通信完成点上的等待合计（ms）：几乎没变
CE_TH_RS_LESS, CE_TH_AG_MORE = 254, 290          # 反向上下文并行那条规约少露 254 ms；gate/up 专家收集多露约 290 ms（9 → 299）
CE_TH_FIFO = {"v4": (310, 317), "3ms": (621, 599)}   # 多核集合通信也放进车道的判据：专家收集露出 预测 / 实测（ms）
CE_TH_MEM128 = 90.69                             # 128 片正常步（GiB），放得下
CE_TH_BITWISE = True
assert 3.5 < CE_TH_ONCARD < 4.3
# 客观判据对八格的校验（gate/up 那条收集的每步露出，预测 / 实测，ms）；绝对值偏低约 15%，只用来排名
CE_SIM_AG = {"ctl": (42.6, 41.5), "fk": (280, 290.5), "fk0": (3.4, 14)}
CE_SIM_LOW_PCT = 15
CE_L4_S = 70                                     # 4 层模型（1 稠密 ＋ 3 MoE，视觉塔 2 层）编一格约 70 秒（39 层要 15–25 分钟）；反向层体的搭档跟 39 层完全相同
CE_L39 = {"3ms032": 3163, "4ms032": 3201, "3ms1": 3220, "v4": 3375, "1ms": 3431}   # 39 层复核（64 片拓扑）：排序和 4 层一致
assert CE_L39["3ms032"] == min(CE_L39.values()) and CE_L39["v4"] > CE_L39["3ms032"]
# 每格反向专家权重收集的搭档（最终编译产物里的调度分组）：
CE_ANCHOR = {"ctl": "矩阵乘", "fk0": "矩阵乘", "fk": "融合核", "fkm": "融合核", "cal": "token 搬运核 ＋ 小的矩阵乘",
             "th_def": "融合核", "th_0.5": "token 搬运核 ＋ 矩阵乘", "th_1": "矩阵乘（跟推荐配置相同）", "th_3": "没有搭档"}
CE_CAL_EXTRA_MS = 234                            # 客观判据：cal 的预计露出比对照多 234 ms

# 9.6 编译时间（出处：作者私有《离线编译（AOT）提速》§1、§2、§3、§4；独占一台 192 vCPU 的编译机，两进程口径从第 1 步起）
AOT_STEPS = [  # (第几刀, 白话, 半层一格 s, 128 片一格 s, 优化前指令数 半层 / 128 片（k 条）)
    (0, "原样：两个程序一个接一个编", 647, 1522, (86.8, 143.4)),
    (1, "两个程序分两个进程同时编（不改模型代码）", 361, 898, (86.8, 143.4)),
    (2, "牛顿迭代的 5 步改成 JAX 循环", 277, 643, (64.7, 105.1)),
    (3, "同形状的参数叠起来批量正交化", 268, 614, (61.2, 98.0)),
    (4, "视觉塔改成 scan", 154, 153, (33.1, 33.6)),
]
AOT_HALF0, AOT_FULL0 = AOT_STEPS[0][2], AOT_STEPS[0][3]
AOT_HALF1, AOT_FULL1 = AOT_STEPS[-1][2], AOT_STEPS[-1][3]
assert round(AOT_FULL0 / 60) == 25 and abs(AOT_FULL1 / 60 - 2.5) < 0.1          # 「从 25 分钟到两分半」：128 片一格
AOT_X_HALF, AOT_X_FULL = AOT_HALF0 / AOT_HALF1, AOT_FULL0 / AOT_FULL1
assert abs(AOT_X_HALF - 4.2) < 0.05 and abs(AOT_X_FULL - 9.9) < 0.05
AOT_CROWD_MIN = (33, 39)                         # 原来在编译机上挤着编，128 片一格要 33–39 分钟
AOT_XLA_PCT = 89                                 # 半层一格 89% 的时间在 XLA 编译里
AOT_ENTRY_K = (61.0, 109.8)                      # 循环外最大那个计算的指令数（k 条，半层 / 128 片，优化前）
AOT_OPT_PCT = 80                                 # 其中优化器约占 80%（牛顿迭代 62% / 59%，其余 20% / 18%）
AOT_NS_CALLS, AOT_NS_VIT = 174, 107              # 半层一个程序里正交化被调用 174 次，约 107 次是视觉塔的单矩阵
AOT_DIAG = (("优化器换成 AdamW（只为诊断）", -41), ("视觉塔 20 层减到 10 层（只为诊断）", -28))
AOT_MEM_HALF = (80.50, 84.39)                    # 视觉塔 scan 叠在推荐配置上：半层正常步 heap（GiB）
AOT_MEM_128 = (88.57, 89.53)                     # 128 片正常步（GiB），上限 94.74、留 1.5 余量以内
assert abs((AOT_MEM_HALF[1] - AOT_MEM_HALF[0]) - 3.9) < 0.05 and AOT_MEM_128[1] < 94.74 - 1.5
AOT_MEM_COPY_GIB = 1.78                          # 原因：三份 BF16 专家权重副本（每份 1.78 GiB）的存活区间跨过了 loss 段
AOT_NS_ROUND = (5485, 5119)                      # 牛顿迭代里 FP32→BF16 舍入（按执行次数）：展开 → 循环
AOT_NS_ROUND_T = "%d → %d" % AOT_NS_ROUND
AOT_MEM_HALF_T = "%.2f → %.2f" % AOT_MEM_HALF
AOT_MEM_128_T = "%.2f → %.2f" % AOT_MEM_128
# ⏳ 上卡（同作业 30 步）：最终候选的步时与逐位结果还没出，出数后填
# 上卡（作者私有调优记录（编译提速上卡）；同作业 30 步，第 13–20 步）
AOT_ONCARD_CTL = 3.926                           # 对照（推荐配置，同一代码包）步时 s
AOT_ONCARD_FINAL = 3.907                         # 最终候选（同形参数批量 ＋ 视觉塔 scan）步时 s，−0.5%（在常驻机器之间约 0.5% 的噪声带边上）
AOT_ONCARD_BITWISE = False                       # 不逐位：30 步 loss 差在执行噪声量级（单步最大 8.4e-5）
AOT_300_VS_V4 = -9.0e-6                          # 300 步 lm loss 逐步相对差带符号平均，对推荐配置
AOT_ROUND_SITES = 3                              # 舍入扫描有增删的 3 处：视觉塔 GELU 反向 1 处、正交化 2 处；按先例接受
AOT_FINAL_MEM = (84.98, 91.29)                   # 最终候选的正常步 heap（GiB）：半层 / 128 片（推荐配置 80.50 / 88.57）
AOT_FINAL_COMPILE = (183, 181)                   # 最终候选（只开同形批量 ＋ 视觉塔 scan，两进程）编一格的秒数：半层 / 128 片（作者私有记录，推荐组合定稿条目）
AOT_C1 = (3.928, 4.016, 72)                      # 第 1 刀单开上卡（同作业 30 步）：对照 3.928 → 4.016 s，优化器里的收集多露出约 72 ms；所以最终候选不开这一刀
assert abs((AOT_C1[1] / AOT_C1[0] - 1) * 100 - 2.2) < 0.05
assert abs((AOT_ONCARD_FINAL / AOT_ONCARD_CTL - 1) * 100 + 0.5) < 0.05
assert AOT_FINAL_MEM[1] < 94.74 - 1.5
AOT_FINAL_MEM_T = "%.2f → %.2f" % (AOT_MEM_HALF[0], AOT_FINAL_MEM[0])
AOT_FINAL_MEM128_T = "%.2f → %.2f" % (AOT_MEM_128[0], AOT_FINAL_MEM[1])

# 第九节 准入
# 9.1 竞争：开关 × 次数。(格, 次数, 坏的次数)
RACE_NAN = [("原样", 3, 3), ("加一行日志（只改调度）", 3, 0), ("all-to-all 强制异步", 3, 3), ("关掉分片预取", 3, 3),
            ("DMA 并发限成 1 路", 3, 3), ("优化器的 all-to-all 交给小核", 3, 0), ("强制异步 ＋ 交给小核", 3, 0)]
RACE_DET = [("对照", 10, 0), ("换专家权重的切法", 17, 0), ("换切法，去掉那两个 all-to-all 开关", 9, 3),
            ("对照 ＋ 一个转置开关", 12, 1), ("换切法 ＋ 同一个转置开关", 18, 4)]
RACE_STEPS = 30
RACE_FIRST_DIV = 9                                # 对照 ＋ 转置开关那 1 次从第 9 步分叉
# 9.2 第 0 步：初始化
INIT_RBG = (13.001839, 13.006385)                # 按切分生成的随机数：对照 / 换切法，第 0 步 loss
INIT_RBG_D = 0.0045
assert abs(INIT_RBG[1] - INIT_RBG[0] - INIT_RBG_D) < 1e-4
INIT_TF = 13.004524                              # 与切分无关的随机数：两边第 0 步 loss 逐位相同（第 1 步也相同）
# 9.3 强制舍入：两边都开以后再扫被省掉的舍入
FR_EXTRA = 11                                    # 换切法那一边，优化器里多出的被省掉的舍入（强制舍入漏覆盖的一处：正交化输入转 BF16；补上以后两边差 0 处）
FR_BASE_ELIDED, FR_A_ELIDED = 6, 11              # 不开强制舍入时：对照 6 处（在正交化内部）、换切法 11 处（在正交化输出）
FR_STEP_OFF, FR_STEP_ON = (4.423, 4.432), (4.413, 4.407)   # 强制舍入关 / 开，各 2 份（同作业，第 5–9 步）
# 正文落地的写法（show() 只保留三位小数，这几个要原样写）
MU_D2_T = "5e-4"
INIT_RBG_0, INIT_RBG_1, INIT_RBG_D_T, INIT_TF_T = "13.001839", "13.006385", "0.0045", "13.004524"
assert float(INIT_RBG_0) == INIT_RBG[0] and float(INIT_RBG_1) == INIT_RBG[1] and float(INIT_TF_T) == INIT_TF
assert float(MU_D2_T) == MU_D2
FR_OFF_T = "%.3f／%.3f" % FR_STEP_OFF
FR_ON_T = "%.3f／%.3f" % FR_STEP_ON


# ══ 2026-11-22 第二轮：优化编年史 ═══════════════════════════════════════════
# 前半程：64 片 v7x、20 层底座，站 0–13。(站, 改动的白话, 步时 s, 有效 MFU %, 这一步 %, 瓶颈类别)
# 瓶颈类别：bytes＝通信字节；sched＝排法与优化器；scq＝小核队列；data＝数据与口径
# 第 0–7 站是第 20–35 步均值，第 8 站起第 5–9 步均值；第 11 站换了数据，有效 MFU 分子变了，不能和上面逐行比
FRONT = [   # 站号照作者私有编年史总表
    (0, "原样跑：专家按 FSDP 切", 12.743, 2.30, None, "bytes"),
    (1, "换切法：每个设备只管自己那一份专家", 6.951, 4.21, -45.4, "bytes"),
    (2, "搬来已上线的注意力配置（少重算、块更大）", 4.948, 5.92, -29.0, "sched"),
    (3, "换新版的专家矩阵乘 kernel", 4.601, 6.36, -7.0, "sched"),
    (4, "专家权重收集流水化", 4.260, 6.87, -7.5, "sched"),
    (5, "注意力的通信改成收压缩后的中间量", 4.101, 7.14, -3.7, "sched"),
    (6, "注意力里几次小矩阵乘并成一次", 4.021, 7.28, -2.0, "sched"),
    (7, "Muon 换切法只搬一段", 3.577, 7.87, -11.2, "sched"),
    (8, "注意力反向跳过空块", 3.496, 8.05, -2.2, "scq"),
    (9, "token 排序挪回主核", 3.100, 9.08, -11.3, "scq"),
    (10, "数据换成离线预打包", 3.031, 11.04, -2.4, "data"),
    (11, "MoE 缓冲只读写有效行", 2.946, 11.36, -2.9, "scq"),
    (12, "对照参照，重分小核的活 ＋ 缓冲 4 倍", 2.731, 12.25, -7.3, "sched"),
]
FRONT_BASE_S, FRONT_BASE_MFU = 12.743, 2.30
FRONT_END_S, FRONT_END_MFU = 2.731, 12.25
assert abs((1 - FRONT_END_S / FRONT_BASE_S) * 100 - 78.6) < 0.05
# 站 0 的诊断（基线 profile）
F0_WAIT_PCT = 59.3                               # 一步里主核等通信的占比
F0_LAYER_WAIT, F0_LAYER_GMM = 246.1, 63.3        # 反向一层：等通信 / 专家矩阵乘（ms）
F0_X = 6.6                                       # 站 0：每层每设备约 77.3 GB ÷ 186 GB/s ≈ 0.416 s，是这层专家矩阵乘 63.3 ms 的约 6.6 倍（本课推导）
assert abs(77.3 / 186 * 1000 / 63.3 - F0_X) < 0.05
F0_EXPERT_MS = 6811                              # 站 0 一步里主核等专家权重收发的毫秒
F0_GB = 77.3                                     # 每层每设备收发（推断：一层全部专家权重 19.3 GB × 收两次 ＋ FP32 梯度 38.7）
assert abs(MC_LAYER_EXPERT_GB * 2 + MC_LAYER_EXPERT_GB * 2 - F0_GB) < 0.1
F1_GB = 12                                       # 换按专家切以后每层每设备约 12 GB（专家 9.7 ＋ token 2.4，推断）
EXPERT_PER_DEV = MC_EXPERTS // 8                 # 按专家切（专家轴 8 份）：每设备 32 个专家
LAYER_EXPERT_GIB_DEV = EXPERT_PER_DEV * 3 * MC_HIDDEN * MC_EXPERT_FF * 2 / 2**30   # 一层收好的专家权重每设备（GiB）
assert abs(LAYER_EXPERT_GIB_DEV - 2.25) < 0.01, LAYER_EXPERT_GIB_DEV
KEEP_ALL_GIB = 81                                # 半层 38 个 MoE 层收好的专家权重都留到反向：约 81 GiB／设备（实测 8 层 17.06 GiB 折算每层约 2.13，× 38，推断；按形状算每层 2.25）
assert abs(17.06 / 8 * 38 - KEEP_ALL_GIB) < 0.5
F1_EXPERT_GB, F1_TOKEN_GB = 9.7, 2.4             # 换切法以后：专家字节 / 换来的 token 通信（推断）
F1_JOINT_PCT = -28.8                             # 只换更宽的路、不减字节的联合切
F3_ATTN_MS = (1975, 796)                         # 搬来那套注意力配置：注意力计算
# 站 5–8 的诊断（第一次录全一步）
F5_WAIT_PCT, F5_OPT_EXPOSED_MS = 44, 651         # 主核等通信仍占 44%；其中优化器露出 651 ms（基线 158）
F5_WIN_GMM = (0, 2)                              # 专家权重收集的窗口里夹的矩阵乘：流水化前 / 后
F8_OPT_SEG_MS, F8_A2A_MS = 906, 497              # 步尾优化器段；三组同步 all-to-all 合计
F8_OPT_AFTER_MS = (819, 340)                     # 优化器段：改前约 / 改后约
# 站 9–12 的诊断（小核队列）
F9_SC0_BUSY_MS, F9_SC0_PCT, F9_SC0_KERNEL_MS = 3424, 88, 1203   # 4.02 s 那格：小核 0 被占 / 占步长 / 其中是 kernel 不是通信
F9_TC_SAVED, F9_STEP_SAVED = 340, 78            # 注意力反向跳块：注意力核少算 / 步时只快（ms），同一对格，见 SIDE_EARLY
# 站 13 的诊断（和公开模型的参照对照）
F13_LAYER_WAIT = (20.5, 10.3)                    # 每层主核停等：本讲模型 / 参照（逐圈中位数，ms）
F13_WO_EXPOSE = 7.2                              # wo 梯度规约的完成点露出（ms，一圈）
F13_THREE = (-0.5, -1.0, -3.7, -4.6)             # 规约只钉一颗小核 / scatter-add 挪回主核 / wo 权重梯度提前 / 三项合开

# 换底座①：20 层底座 → 半层底座（39 层主干 ＋ 视觉塔 20 层 ＋ MTP 头，每个设备的负担跟 128 片全量一样）
SW1 = (2.731, 12.25, 5.476, 11.10)               # 同一份推荐配置：20 层底座 步时/有效 MFU → 半层底座 步时/有效 MFU（第 5–9 步，预打包数据）

# 第二段：半层底座、短文（滤掉超过 8K 的长文档），站 13–16；每站「对上一步」优先同作业对照
HALF = [
    (13, "数据滤掉长文档", 5.311, 10.98, -2.7, "data"),
    (14, "梯度改 BF16 ＋ 两条运行库开关", 4.413, 13.18, -16.9, "bytes"),
    (15, "优化器前加屏障、all-to-all 交给小核、MoE 分支输出不重算", 4.147, 14.03, -6.0, "sched"),
    (16, "token 和专家权重分到两颗小核、权重收集限 2 条", 4.049, 14.37, -2.4, "sched"),
]
HALF_START = 5.476                                # 衔接①之后的起点（同一推荐配置，半层底座，预打包数据）
F15_CTL = 4.410                                   # 站 15 的同作业对照（站 14 重编）
# 换底座②：短文 → 长文（同一配置；长文每行一篇长文，注意力有效配对占满因果 0.971，短文 0.057）
SW2 = (5.311, 10.98, 5.553, 17.83)               # 第 13 站那套配置：短文 → 长文
SW2B = (4.049, 14.37, 4.804, 20.73)              # 第 16 站那套配置：短文 → 长文（也就是第 17 站）
EFF_SHORT, EFF_LONG = 673.5, 1148.2              # 有效分子（TFLOP / 设备 / 步）
LONG = [
    (17, "短文上调好的那套开关", 4.804, 20.73, -13.5, "sched"),
    (18, "片上暂存上限放到 64 MiB", 4.678, 21.23, -2.3, "sched"),
    (19, "修掉优化器白搬 FP32 主权重（先上局部修法）", 4.337, 22.89, -7.3, "bytes"),
    (20, "注意力输出投影存下来、反向少重算一步", 4.251, 23.36, -2.1, "sched"),
    (21, "换成正式修法：结果钉回参数自己的切法", 4.225, 23.56, -0.8, "bytes"),
    (22, "注意力反向 kernel 不再为空块排空转的计算步", 4.192, 23.75, -0.8, "sched"),   # 夜里测试线新入表（换常驻复测 4.237 → 4.191，−1.1%）
    (23, "专家权重在层循环里逐层转 BF16（省约 3 GiB）", 4.178, 23.83, -0.3, "sched"),   # 三次读数 4.178 / 4.179 / 4.179
    (24, "专家两份权重先在 BF16 里拼好再收集", 4.094, 24.32, -1.9, "bytes"),   # 同作业对照 4.174；30 步逐位；复测 4.096
    (25, "精度修正：两处归一化恢复 32 位", 4.097, 24.30, 0.0, "num"),
    (26, "前向精度写法改回跟参考实现一致", 4.108, 24.23, 0.2, "num"),
    (27, "梯度四处精度写法改回参考实现（定稿）", 4.116, 24.18, 0.1, "num"),   # 同作业对照 4.110   # 同作业对照 4.099 / 4.104；三处里起作用的只有输出层以 FP32 输出那一处   # 步时持平（同作业关掉这两处 4.102 / 4.099）；loss 偏移 +6.8e-5 → −1.7e-5
]
LONG_START = 5.553
L18_CTL, L19_CTL, L20_CTL, L21_CTL, L22_CTL, L23_CTL, L24_CTL = 4.787, 4.677, 4.342, 4.259, 4.225, 4.190, 4.174    # 各站的同作业对照
S21_S, S21_MFU = 4.225, 23.56
S22_S, S22_MFU = 4.192, 23.75
S23_S, S23_MFU = 4.178, 23.83
S24_S, S24_MFU = 4.094, 24.32                    # 最快的一站：站 24
S25_S, S25_MFU = 4.097, 24.30                    # 站 25：站 24 ＋ 归一化恢复 32 位，步时持平
S26_S, S26_MFU = 4.108, 24.23                    # 站 26：前向精度写法改回跟参考实现一致（+0.2%）
END_S, END_MFU = 4.116, 24.18                    # 终点、长文推荐配置（定稿）：站 27（梯度四处精度写法也改回，+0.1%）
L27_CTL = 4.110                                  # 站 27 同作业对照（不带那四处）
LONG_START_SAMEJOB = 5.556                       # 长文起点在定稿那个作业里重测
LONG_CUM_PCT = (1 - END_S / LONG_START_SAMEJOB) * 100
assert abs(LONG_CUM_PCT - 25.9) < 0.05, LONG_CUM_PCT
RUN200_S, RUN200_MFU = 4.192, 23.71              # 定稿版 200 步长跑（第 5–199 步均值）：对外口径；0 次回放、0 丢 token
RUN200_BUF = 3.79                                # 长跑里所需缓冲倍数最大值（缓冲 4 倍）
# 20 层同底座、长文数据、同一个作业：四个里程碑（从生产配方原样起算的完整一条线）
MILE20 = [("生产配方原样（原始代码）", 12.914), ("第一段终点那套配置", 2.798), ("第二段终点那套配置", 2.511), ("长文推荐配置（定稿）", 2.228)]
MILE20_X = MILE20[0][1] / MILE20[-1][1]
assert abs(MILE20_X - 5.80) < 0.01, MILE20_X
MILE20_SEG = [round(MILE20[i][1] / MILE20[i + 1][1], 2) for i in range(3)]
assert MILE20_SEG == [4.62, 1.11, 1.13], MILE20_SEG
MILE20_X_T = "5.80"
MILE20_0, MILE20_1, MILE20_2, MILE20_3 = [v for _n, v in MILE20]
MILE20_SEG_0, MILE20_SEG_1, MILE20_SEG_2 = MILE20_SEG
# 首屏口径（负责人 11-22 20:24 定）：起点用 20 层长文上的原始代码 12.914 s；有效 MFU 4.17% 是推导值（作者私有报告的写法）：
#   原始代码自己不打有效 MFU，借用同一个作业里 v2（定稿）截到 20 层那一格的有效分子 —— 同一份数据、同一层数，每步的有效计算量相同：
#   有效分子 ＝ 24.16% × 2.228 s × 1155 TFLOP/s ≈ 621.7 TFLOP；4.17% ＝ 621.7 ÷ 12.914 ÷ 1155
MILE20_V2_MFU = 24.16                            # 同作业 v2 截到 20 层：2.228 s、有效 24.16%
HERO_EFF20_TF = MILE20_V2_MFU / 100 * MILE20[-1][1] * 1155
HERO_START_MFU = HERO_EFF20_TF / MILE20[0][1] / 1155 * 100
assert abs(HERO_EFF20_TF - 621.7) < 0.1 and abs(HERO_START_MFU - 4.17) < 0.005, (HERO_EFF20_TF, HERO_START_MFU)
HERO_EFF20_T = "%.1f" % HERO_EFF20_TF
# 标准口径（负责人 11-22 21:30 定）：按生产图像比例（约八分之一，图文同在一个文档里、不切段）的长图文数据。出处：作者私有调优记录「30 步三份数据同作业」与「20 层同底座起点」两条
STD_START_S, STD_EFF20_TF = 12.899, 633.7       # 20 层原始代码步时；同作业 v2 截到 20 层那格在这份数据上的有效分子（TFLOP / device / 步）
STD_START_MFU = STD_EFF20_TF / STD_START_S / 1155 * 100
assert abs(STD_START_MFU - 4.25) < 0.005, STD_START_MFU
STD_V2_20 = (2.253, 24.35)                       # 同作业 v2 截到 20 层
assert abs(STD_V2_20[1] / 100 * STD_V2_20[0] * 1155 - STD_EFF20_TF) < 0.2
STD_X = STD_START_S / STD_V2_20[0]
assert abs(STD_X - 5.73) < 0.005, STD_X
STD_END_S, STD_END_MFU, STD_END_TF = 3.923, 25.88, 1172.6   # v5，39 层，30 步格（第 13–20 步）
assert abs(STD_END_TF / STD_END_S / 1155 * 100 - STD_END_MFU) < 0.01
STD_IDEAL_S = STD_END_TF / 1155                  # 按峰值的理想一步 ≈ 1.015 s
STD_GAP_X = STD_END_S / STD_IDEAL_S              # ≈ 3.86 倍，就是 1 ÷ 25.88%
assert abs(STD_GAP_X - 100 / STD_END_MFU) < 0.002 and abs(STD_GAP_X - 3.86) < 0.005
STD_IDEAL_T, STD_GAP_T, STD_X_T = "%.2f" % STD_IDEAL_S, "%.2f" % STD_GAP_X, "%.2f" % STD_X
HERO_LONG_S, HERO_LONG_MFU = 3.941, 25.22       # v5 200 步长跑（第 5–199 步）
# 定稿后 300 步数值验证（三臂同作业，第 1–299 步逐步 loss 相对差的带符号均值）
V300_FINAL_VS_START, V300_FINAL_VS_REF, V300_STACK_VS_START = "−3.5e-5", "−3.3e-5", "−1.7e-6"
V300_LINE = "2e-5"                               # 站 25 验收用的线：|带符号均值| ≤ 2e-5
V300_GRAD_ONLY = "−3.0e-5"                       # 只开梯度整体 BF16 那一格（没有增长）
V300_BF16_COST = (4.202, 4.478)                  # 那三处 BF16 在定稿上值多少步时：带 / 不带（300 步作业第 5 步起中位数）
V300_BF16_COST_PCT = (V300_BF16_COST[1] / V300_BF16_COST[0] - 1) * 100   # 6.6%：三处改回 FP32，定稿要慢多少（分母是定稿，同项目报告口径）
assert abs(V300_BF16_COST_PCT - 6.6) < 0.05
# 站 27 的梯度组逐处单测（对站 25 那份配置，代价）：四处便宜的恢复，两处贵的保持
G6_CHEAP_MAX_PCT, G6_EXPERT_PCT, G6_ROUTER_PCT = 0.4, 4.0, 3.0
G6_ROUTER_WAIT_MS = 118                          # 路由那一处：计算 −5 ms，露出的专家权重收集等待 +118 ms
L26_CTL = (4.099, 4.104)                         # 站 26 同作业对照
L25_CTL = (4.102, 4.099)                         # 站 25 同作业对照：关掉那两处
N25_DRIFT = ("+6.8e-5", "−1.7e-5")             # 300 步带符号 loss 偏差：修正前 / 修正后（执行噪声量级）
for _st, _n, _s, _m, _p, _k in HALF[1:] + LONG[1:]:
    pass
assert abs((1 - 4.413 / 5.311) * 100 - 16.9) < 0.05 and abs((1 - 4.147 / F15_CTL) * 100 - 6.0) < 0.05
assert abs((1 - 4.678 / L18_CTL) * 100 - 2.3) < 0.05 and abs((1 - 4.337 / L19_CTL) * 100 - 7.3) < 0.05
assert abs((1 - 4.251 / L20_CTL) * 100 - 2.1) < 0.05 and abs((1 - S21_S / L21_CTL) * 100 - 0.8) < 0.05 and abs((1 - S22_S / L22_CTL) * 100 - 0.8) < 0.05 and abs((1 - S23_S / L23_CTL) * 100 - 0.3) < 0.05 and abs((1 - S24_S / L24_CTL) * 100 - 1.9) < 0.05
assert abs((1 - 4.804 / 5.553) * 100 - 13.5) < 0.05

# 开场「难在条件上」：常用手段为什么用不上（量过的代价）
HARD_TOK = (8192, 24576, 17.40, 25.91)            # 每芯 token 8192 → 24576：有效 MFU（倒序阶梯，上一步是去掉 splash 子块跳过那一格）
HARD_SEQ = (1.133, 2.198)                         # 序列 4096 → 32K ＋ 上下文并行 8，每芯 token 相同：步时（参照配方口径的阶梯）
HARD_MUON_PCT = 9.7                               # Muon 换回 AdamW：步时 −9.7%（倒序阶梯，2.045 → 1.847 s）
assert abs((1 - 1.847 / 2.045) * 100 - HARD_MUON_PCT) < 0.05
HARD_FP8_X = 1.34                                 # 公开模型的参照阶梯上 FP8 → BF16：步时 ×1.34（本模型没量过）
HBM_GIB = 94.74                                  # 每设备 HBM 上限；收好的专家权重全部留到反向，全部 MoE 层加起来接近整块 HBM（推断）
assert round(HARD_TOK[3] - HARD_TOK[2], 1) == 8.5
SW1_0, SW1_1, SW1_2, SW1_3 = SW1
SW2_0, SW2_1, SW2_2, SW2_3 = SW2
SW2B_0, SW2B_1, SW2B_2, SW2B_3 = SW2B
FRONT_BASE_MFU_T, END_MFU_T, S21_MFU_T, S22_MFU_T, S23_MFU_T, S24_MFU_T = "2.30", "24.18", "23.56", "23.75", "23.83", "24.32"
S26_MFU_T = "24.23"
TARGET25_S = EFF_LONG / (0.25 * PEAK_DEV)      # 长文有效分子下有效 MFU 25% 要的步时（同项目报告：3.977 s）
assert abs(TARGET25_S - 3.977) < 0.001 and abs(EFF_LONG / END_S / PEAK_DEV * 100 - END_MFU) < 0.05
TARGET25_GAP_PCT = (1 - TARGET25_S / END_S) * 100           # 3.4%
TARGET25_GAP_RUN_PCT = (1 - TARGET25_S / RUN200_S) * 100    # 5.1%
assert abs(TARGET25_GAP_PCT - 3.4) < 0.05 and abs(TARGET25_GAP_RUN_PCT - 5.1) < 0.05, (TARGET25_GAP_PCT, TARGET25_GAP_RUN_PCT)


# ══ 开篇总览：阶梯曲线（fig8b-curve；只给开篇图用，各节正文另有自己的数）═══════════════════
# 横轴一站一格，纵轴步时（对数）。站 0–27 直接用上面 FRONT / HALF / LONG；定稿之后再两步（不编站号）。
# 出处：作者私有编年史总表、调优记录 §11.0000 / §15.5、项目报告总表；每格的原始作业名只在私有台账里。
CURVE_NAME = {   # 大跳的白话短名（≥5% 的跳里挑 8 个；相邻太挤的 3、4、15 不起名）
    1: "每个设备只收自己那份专家", 2: "搬来已上线的注意力配置", 7: "优化器换切法只搬一段", 9: "token 排序挪回主核",
    12: "重分小核的活", 14: "梯度改 BF16", 17: "短文那套开关整包搬来", 19: "修白搬主权重",
}
CURVE_POST = [   # 定稿之后：(白话, 步时 s, 有效 MFU %, 对同作业对照 %, 同作业对照步时, 改了数值)
    ("路由先挑 top-k 再收集", 4.082, 24.39, -0.9, 4.120, True),
    ("融合核去掉耗时估计（绕开调度器挑锚点的问题）", 3.931, 25.33, -3.8, 4.088, True),
    ("调度门槛（照实填估计、门槛 3 ms；对照已含编译组合）", CE_TH_ONCARD, 25.74, -1.0, CE_TH_ONCARD_CTL, True),   # 对照＝同一台机器上的上一步 ＋ 编译组合；调度部分 30 步逐位，编译组合改了舍入点
]
assert "%.3f" % CURVE_POST[1][1] == CE_IHC0_T and CURVE_POST[1][4] == CE_CTL
for _n, _s, _m, _p, _c, _x in CURVE_POST:
    assert abs((_s / _c - 1) * 100 - _p) < 0.05, _n
    assert abs(EFF_LONG / _s / PEAK_DEV * 100 - _m) < 0.05, _n          # 长文有效分子 1148.2，站时与 MFU 对得上
CURVE_NUM = {14, 25, 26, 27}     # 站内刻意改了数值（精度开关）；定稿之后两步见 CURVE_POST 末列
CURVE_PREFETCH = (4.948, 5.388)  # 被否掉：20 层底座上预取下一层权重 ＋ 梯度规约后移一层，对照 → 加上（+8.9%）
CURVE_REJ = [    # (挨着哪一站（"p1" ＝ 定稿之后第 1 步）, 白话, 实测步时 s, 对它自己的对照 %)
    (3, "预取下一层权重", CURVE_PREFETCH[1], (CURVE_PREFETCH[1] / CURVE_PREFETCH[0] - 1) * 100),
    (20, "融合核照原样", IH_S, IH_PCT),
    ("p1", "核耗时按实测校准", CE["cal"][0], CE_PCT["cal"]),
]
assert abs(CURVE_REJ[0][3] - 8.9) < 0.05 and abs(CURVE_REJ[1][3] - 7.8) < 0.05 and abs(CURVE_REJ[2][3] - 9.4) < 0.05
CURVE_END_S, CURVE_END_MFU = CURVE_POST[-1][1], CURVE_POST[-1][2]        # 3.868 s / 25.74%（30 步口径）；定稿之后第二步 3.931 s 是第一次过有效 25%
CURVE_RUN200_S, CURVE_RUN200_MFU = 3.941, 25.22   # 最新配置的 200 步长跑（第 5–199 步均值，0 回放、0 丢 token）：长跑第一次过有效 25%
CURVE_RUN200_PREV = (4.018, 24.74)               # 上一版（定稿之后第二步）的 200 步长跑：没过 25%
assert abs(EFF_LONG / CURVE_RUN200_S / PEAK_DEV * 100 - CURVE_RUN200_MFU) < 0.05 and CURVE_RUN200_S < TARGET25_S
assert abs(EFF_LONG / CURVE_RUN200_PREV[0] / PEAK_DEV * 100 - CURVE_RUN200_PREV[1]) < 0.05 and CURVE_RUN200_PREV[0] > TARGET25_S
CURVE_RUN200_VS_PREV = (CURVE_RUN200_S / CURVE_RUN200_PREV[0] - 1) * 100     # −1.9%
assert abs(CURVE_RUN200_VS_PREV + 1.9) < 0.05, CURVE_RUN200_VS_PREV
CURVE_RUN200_GAP_PCT = (1 - TARGET25_S / CURVE_RUN200_S) * 100      # −0.9%：长跑已经比有效 25% 要的 3.977 秒快（负数＝已过）
CURVE_RUN200_GAP_MS = (CURVE_RUN200_S - TARGET25_S) * 1000          # 约 −35 ms
assert abs(CURVE_RUN200_GAP_PCT + 0.9) < 0.05 and -37 < CURVE_RUN200_GAP_MS < -34, (CURVE_RUN200_GAP_PCT, CURVE_RUN200_GAP_MS)
assert CURVE_END_S < TARGET25_S
CURVE_SEG_X = (FRONT_BASE_S / FRONT_END_S, HALF_START / HALF[-1][2], LONG_START / CURVE_END_S)   # 段内倍数 4.67 / 1.35 / 1.44
assert [round(x, 2) for x in CURVE_SEG_X] == [4.67, 1.35, 1.44], CURVE_SEG_X
CURVE_CHAIN_X = 1.0              # 选型记录：把每站「对上一站」串乘起来会得到约 9.1 倍，同一底座实测只有 5.80 倍 —— 所以开篇图不画串乘
for _st, _n, _s, _m, _p, _k in FRONT[1:] + HALF + LONG:
    CURVE_CHAIN_X *= 1 / (1 + _p / 100)
for _n, _s, _m, _p, _c, _x in CURVE_POST:
    CURVE_CHAIN_X *= 1 / (1 + _p / 100)
assert 9.0 < CURVE_CHAIN_X < 9.2 and CURVE_CHAIN_X > MILE20_X * 1.5, CURVE_CHAIN_X
CURVE_IDEAL_S = EFF_LONG / PEAK_DEV              # 长文上按峰值理想的一步：1148.2 ÷ 1155 ≈ 0.994 秒（本课推导）
CURVE_GAP_X = CURVE_END_S / CURVE_IDEAL_S        # 还差几倍：3.868 ÷ 0.994 ≈ 3.89（＝ 1 ÷ 有效 MFU，差在有效分子取整）
CURVE_GAP_RUN_X = CURVE_RUN200_S / CURVE_IDEAL_S # 按长跑口径：3.941 ÷ 0.994 ≈ 3.96（＝ 1 ÷ 25.22%）
assert abs(CURVE_GAP_X - 3.89) < 0.005 and abs(CURVE_GAP_RUN_X - 3.96) < 0.005 and abs(100 / CURVE_END_MFU - CURVE_GAP_X) < 0.01

# ── R4 补出处：正文里写成字面量、以前没登记的几处（来源都是作者私有调优文档）
DSV3_BUSY_AVG_BLOCKS = 50       # 4.2 / fig-ref：对方文本数据第 5–9 步最忙设备平均约 50 个子块（本讲模型几乎每步 66）
HW_MFU_FINAL = 36.5             # 11.3：定稿硬件 MFU（实际执行的矩阵运算，含重算、优化器、冗余；修正核的记账后）
FINAL_PROFILE_S = 4.122         # 11.3：定稿时间账那次多芯片 profile 的步时
FULL128_NORMAL_GIB, FULL128_HEADROOM_GIB = 88.99, 4.25   # 9.4：128 片全量正常步离线峰值与余量（作者调优记录）
RAW_HALF_MEM_GIB = 103.24                                # 第十一节 / fig-mile：原始代码放到半层底座要的显存（放不下）
