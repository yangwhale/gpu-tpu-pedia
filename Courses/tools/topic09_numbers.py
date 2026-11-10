# -*- coding: utf-8 -*-
r"""专题九 · GPU→TPU 训练精度对齐 —— 全讲共用的数（唯一一份）。

⭐ 一个事实一个落点：图脚本、课件正文、开场题都从这里取，不各抄一份（手册 03 · 二 · 5）。
⭐ 出处约定：
   · 「实测」= 本课程作者做的一次对齐实践：混元 3（Hy3，截前 4 层、去掉 MTP），
     GPU 侧 B200×8 + Megatron-Bridge，TPU 侧 v5p（8 颗芯片）／ v7x（4 颗芯片） + MaxText，同一份权重、同一批数据、同样的顺序，1000 步。
   · 「本课推导」= 自己算的，推导链写在旁边。
   · 逐步序列（loss、param_norm）从 topic09_series.json 现算，下面的常数只是给断言用的锚点。
⛔ 改一个数之前，先 grep 全专题（图脚本、build、讲义）—— 同一个数可能还挂在别处的断言上。
"""
import json
import math
import os
import struct

HERE = os.path.dirname(os.path.abspath(__file__))

# ── 规模（实测配置）────────────────────────────────────────────────
STEPS = 1000                 # 每条对齐运行 1000 步
RANKS = 8                    # GPU 8 卡数据并行；TPU 8 个 JAX device
MICRO = 2                    # 每卡每步 2 个 microbatch（全局批 16）
DATA_UNITS = STEPS * RANKS * MICRO
assert DATA_UNITS == 16000
NATIVE_SAME_RANK = 2000      # 不重排时，MaxText 原生加载器只有 2000 个单元落到同号设备
N_TENSORS = 1781             # 权重转换后逐张量逐位比的张量数
N_FP32_TENSORS, N_BF16_TENSORS = 3, 1778
assert N_FP32_TENSORS + N_BF16_TENSORS == N_TENSORS
N_PARAMS = 12386079744       # 截 4 层后的参数总数，两边一致
VOCAB = 120832               # 词表（输出头 dgrad 的收缩长度）
HIDDEN = 4096

WD_CONFIG, WD_EFFECTIVE = 0.1, 0.033   # 配置写 0.1，调度器每步覆盖成 0.033
LR = 1e-5                    # 对齐用常数学习率
BETA1, BETA2, EPS = 0.9, 0.95, 1e-8

# ── 第 1 步 loss（L0 的快速自检）────────────────────────────────────
LOSS1_TPU, LOSS1_GPU = 11.465785, 11.465756
D1 = (LOSS1_TPU - LOSS1_GPU) / LOSS1_GPU
assert abs(D1 - 2.5e-6) < 0.1e-6, D1
GRADNORM1 = 17.54            # 第 1 步裁剪前的梯度范数
CLIP1 = 1.0 / GRADNORM1      # 裁剪系数（clip_grad = 1.0）
assert abs(CLIP1 - 0.057) < 0.001

# ── 噪声尺子 ──────────────────────────────────────────────────────
GPU_RUNS10_MEAN, GPU_RUNS10_STD = -2.5e-5, 2.7e-5     # 10 次 GPU 重跑，301–1000 步 loss 带符号均值（对第一份）
GPU_RUNS10_LO, GPU_RUNS10_HI = -6.5e-5, 2.4e-6
TPU_ARMS = (-1.66e-4, -1.35e-4, -1.03e-4)             # 同一份代码的三种跑法（原样／不重算／改片上内存上限）
PAIRS_10RUNS = 10 * 9 // 2
assert PAIRS_10RUNS == 45
INCONCL = {3: (8, 46), 15: (3, 9), 28: (1, 46)}      # 参照对数 → 待定条数（GPU 对 GPU 演练）
# 精确轨迹扰动出的五份两两 10 对的上限 vs ③ 那组 GPU 重跑 10 对的上限（param_norm 带符号均值，实测）。
# ⛔ 旧值 6.9e-9 只是单对（第二份给第 0 层一处权重乘 1 ＋ 1e-6），不能代表「精确轨迹自己的散布」；五份 10 对的上限跟 GPU 那把同一量级。
# GPU_RERUN_PN 的「GPU 重跑」是 ③ 那组（生产近似＋输出头 dgrad 换 FP32 一次累加）5 份两两 10 对，不是 ② 生产程序本身。
EXACT_SELF_PN, GPU_RERUN_PN = 1.07e-7, 1.24e-7
assert 0.5 < EXACT_SELF_PN / GPU_RERUN_PN < 2, "同一量级"
# 交叉核对（只报不判）：被测 v5p 当前配置，噪声换成别的尺子时 r 是多少（实测）。
XCHK_SELF = {"param_norm": 4.00, "loss 全程": 2.03, "loss 尾段": 1.21, "grad_norm": 2.24}   # 噪声 ＝ 精确轨迹自己那 10 对
XCHK_TPU_PN = 5.03                                    # 噪声 ＝ TPU 执行噪声 28 对，param_norm（按新标定；旧值 4.74）
assert XCHK_SELF["param_norm"] > 2 and XCHK_SELF["loss 全程"] > 2 and 1 < XCHK_SELF["loss 尾段"] <= 2 and XCHK_SELF["grad_norm"] > 2
assert XCHK_TPU_PN > 2


def p_exceed(n):
    """一个纯噪声的被测，超过 n 个同分布样本最大值的概率（本课推导：对称性，1/(n+1)）。"""
    return 1.0 / (n + 1)


# ── 第一层验收（v5p 当前配置对精确轨迹那一份，噪声取 ③ 那组 GPU 重跑 10 对）────────
ACC = [  # (量, 被测值, N, r)
    ("param_norm 带符号均值", 4.27e-7, 1.24e-7, 3.45),
    ("loss 带符号均值（第 2–1000 步）", 8.62e-5, 4.49e-5, 1.92),
    ("loss 带符号均值（第 301–1000 步）", 7.23e-5, 5.62e-5, 1.29),
    ("grad_norm 带符号均值", 2.32e-3, 2.03e-3, 1.14),
]
for _n, _x, _N, _r in ACC:
    assert abs(_x / _N - _r) < 0.01, (_n, _x / _N, _r)
ACC_V7X_PN = 3.53

# ── 张量指标 ──────────────────────────────────────────────────────
COS_0999_RELL2 = math.sqrt(2 * (1 - 0.999))           # 1 − cos ≈ ½·rel_L2²（本课推导）
assert abs(COS_0999_RELL2 - 0.0447) < 0.0005
R_GATE = 1.2
R_ONE_EXTRA = math.sqrt(2)                            # 多一次同量级独立舍入（本课推导）
R_QKV_BEFORE, R_QKV_AFTER = 1.235, 1.000
R_FC1_BEFORE, R_FC1_AFTER = 1.294, 1.117
QKV_VS_GPU_BEFORE, QKV_VS_GPU_AFTER = 2.36e-3, 1.8e-4
ULP1_SHARE, ULP17_SHARE = 0.013, 0.67                 # GPU 自比第 1 步梯度：差 1 格 1.3%，≥17 格 67%

# ── 浮点与硬件 ────────────────────────────────────────────────────
FORMATS = [("FP32", 8, 23), ("TF32", 8, 10), ("BF16", 8, 7)]   # (名字, 指数位, 尾数位)
BF16_GAP1 = 2.0 ** -7         # BF16 在 1 附近的格距；最大相对舍入误差是它的一半 2⁻⁸ ≈ 3.9e-3（本课推导）
TC_GROUP, TC_KEEP = 16, 25    # 实测的 B200 张量核模型：16 个乘积一组对齐，低于 2^(emax−25) 的位截掉
TC_BIAS_K = [(256, 2.2e-5), (4096, 7.0e-5), (65536, 3.1e-4)]    # 合成正数输入，朝 0 的相对偏差
FP32_BIAS_MAX = 5e-8
HEAD_BIAS_K = [(16384, 6.9e-5), (65536, 2.0e-4), (120832, 6.5e-4)]  # 真实输出头，沿「所有 logits 一起动」方向
HEAD_BF16_S1, HEAD_BF16_S100 = -6.51e-4, -6.10e-4     # GPU 输出头 dgrad 沿该方向的相对误差（第 1 / 100 步）
HEAD_TF32_S1 = -8.08e-4                               # 名义 FP32、实际 TF32 的那一版
HEAD_EXACT_S1 = -1.7e-6                               # 真关 TF32 的 FP32
HEAD_TPU_S1, HEAD_TPU_S100 = -2.4e-5, -7.4e-5         # TPU 输出头反向对 FP64（输出头回放，第 1 / 100 步，跟 HEAD_BF16_S1/S100 同步号配对；另一次单 microbatch 回放读 −2.6e-5、同次 GPU −4.4e-4，别跟这里混配）
CE_BWD_S1 = 2.7e-6                                    # 交叉熵反向自己的那份（第 1 步）
TF32_TE_ON, TF32_ALL_OFF = 2.94e-4, 1.15e-6           # FP32 GEMM 对 FP64 的 rel_L2：只关 torch 开关 / 设 NVIDIA_TF32_OVERRIDE=0
TF32_PROBE_OK = 1e-5
ROUTER_TPU_FP64, ROUTER_GPU_FP64 = 4.8e-8, 3.8e-6     # 路由 logits 对 FP64
WINDOWED_LOOPS = 41

# ── 案例 ─────────────────────────────────────────────────────────
SEG_MEANS = [("2–100", 8.3e-6), ("101–300", -7.4e-5), ("301–600", -1.7e-4), ("601–1000", -1.7e-4)]
SEG_GPU = [(-1.7e-5, 2.2e-5), (-4.9e-5, -1.4e-5), (-4.6e-5, 2.8e-5), (-3.2e-5, 4.5e-5)]
MISALIGN = (1.53e-4, 1.41e-2)                          # 同步号 vs 错一步的平均绝对相对差
PN_GPU_GROWTH, PN_TPU_EXTRA = 0.127, 0.021
GROUP_TOTAL_300, GROUP_HEAD_300 = 4.39e-3, 3.83e-3
ROWS_300 = [("没出现过", 17.1), ("出现 1–10 次", 3.1), ("11–100 次", 2.5), ("100 次以上", 0.5)]   # 第 300 步各桶输出层平方和差；100 次以上 ＝ 0.476＋0.004−0.005（R11 更正：原写「常见词 约 0」）
ROWS_300_TOTAL, ROWS_GPU_PAIR_MAX = 23.2, 0.21
RARE_ROWS = 58638
SHRINK_GPU, SHRINK_TPU, SHRINK_WD = 66.3, 49.2, 31.8
assert abs(SHRINK_GPU - SHRINK_WD - 34.5) < 0.05 and abs(SHRINK_TPU - SHRINK_WD - 17.4) < 0.15
RARE_G_LO, RARE_G_HI = 4e-9, 1e-8                     # 冷门行梯度的量级：第 200–300 步 √v̂ 的**均值**落在这个区间（不是梯度分位数）；第 1 步梯度中位数 5.2e-8，12% 小于 ε
# 冷门行放大链，第 300 步同一步直接量（实测，没出现过的词那些行）：
RARE_RMS_RATIO = 0.990        # 两边冷门行 |g| 的均方根之比（TPU ÷ GPU）：大小几乎一样
RARE_EPS_FRAC = 0.92          # √v̂ < ε 的元素占比 → 这些元素 u ≈ m̂ ÷ ε
RARE_VHAT_MED = 2.6e-9        # √v̂ 的中位数
SUMWU_DIFF, SUMWU_MHAT = -4994, -5450                 # Σ w·u 两边之差，其中 m̂ 那一份
assert abs(RARE_RMS_RATIO - 1) < 0.02 and RARE_VHAT_MED < EPS and RARE_EPS_FRAC > 0.5
assert abs(SUMWU_MHAT) > abs(SUMWU_DIFF) and SUMWU_MHAT * SUMWU_DIFF > 0      # m̂ 那份是大头、同号
OGSCALE_125 = 14.6                                     # 只把输出层梯度放大 1.25 倍，第 300 步输出层平方和多涨
# 干预（对第一份 GPU 运行，1000 步）
TPU_LOSS_TAIL, TPU_PN_1000 = -1.66e-4, 2.1e-2
REMOVE_LOSS, REMOVE_LSE, REMOVE_PN = -3.4e-5, -0.007, 6.1e-3   # 两边都去掉那个方向之后
ORIG_LSE = -0.33                                       # 第 1000 步训练批上的 lse 差（不是固定探针）
LSE_PROBE = [(25, -0.0016, -0.0001), (50, -0.016, 0.0013), (100, -0.078, -0.0062), (300, -0.218, -0.021)]   # 固定探针，对第一份 GPU（括号里 GPU 两次之间）
FORCE_LOSS, FORCE_PN = -1.26e-4, 2.05e-2               # GPU 反向加力 k = −8
# 同一状态（GPU 第 99 步状态搬到 TPU）、同一批数据，沿漂移方向每步比精确值多走／少走多少（%）。
# 共同参照 ＝ GPU 全程 FP32（TPU 自己的「FP32 参照」里路由专家其实按 BF16 算，作废）。
# 区间 ＝ 4 个状态、12 个 batch 的范围。
SWAP = [("TPU 生产程序", 0.00, (-0.04, 0.10)), ("GPU 只换FP32 dgrad", -0.10, (-0.22, -0.05)), ("GPU 生产程序", -0.36, (-0.47, -0.31))]
# 精确轨迹站在哪一边（第 975 步 param_norm 对 4 份默认 GPU 运行的均值）
EXACT_REF = 4.5e-4
EXACT_POS = [("FP32 dgrad 的 GPU", 1.25e-2), ("精确轨迹", 1.82e-2), ("TPU", 1.99e-2)]
DGRADX_RANGE = (1.23e-2, 1.28e-2)
assert DGRADX_RANGE[0] <= EXACT_POS[0][1] <= DGRADX_RANGE[1]
# 第 301–1000 步 loss 平均偏移，对 4 份默认 GPU 运行的均值（实测）：精确轨迹两份、TPU（不重算的配置）
EXACT_LOSS_TAIL = (-1.53e-4, -1.16e-4)
EXACT_TPU_LOSS_TAIL = -1.27e-4
assert min(EXACT_LOSS_TAIL) < EXACT_TPU_LOSS_TAIL < max(EXACT_LOSS_TAIL)    # TPU 夹在精确轨迹两份中间
EXACT_SHARE = EXACT_POS[1][1] / EXACT_POS[2][1]
assert abs(EXACT_SHARE - 0.91) < 0.01, EXACT_SHARE
# 拆账：TPU 对 GPU 生产 golden 第 1000 步 param_norm 差
DECOMP_TOTAL = 2.12e-2
DECOMP = [("输出头 dgrad 截断", 1.24e-2), ("GPU 其余 BF16 误差", 0.62e-2), ("TPU 自己", 0.26e-2)]
assert abs(sum(v for _, v in DECOMP) - DECOMP_TOTAL) < 1e-9
DECOMP_SHARE = (0.59, 0.29, 0.12)                      # 实测拆账给的份额（按未舍入的原值算）
for (_l, _v), _s in zip(DECOMP, DECOMP_SHARE):
    assert abs(_v / DECOMP_TOTAL - _s) < 0.01, (_l, _v / DECOMP_TOTAL)
# 第 301–1000 步 loss 相对 GPU 生产 golden 均值（实测）：TPU、只修 dgrad 的那份 − 生产、精确轨迹 − 生产
LOSS_TPU_VS_PROD_TAIL, LOSS_FIXHEAD_MINUS_PROD, LOSS_EXACT_MINUS_PROD = -1.34e-4, -9.8e-5, -2.09e-4
assert abs(LOSS_FIXHEAD_MINUS_PROD / LOSS_TPU_VS_PROD_TAIL - 0.73) < 0.01

# ── 日常回归 ──────────────────────────────────────────────────────
ARMS_TRIED, ARMS_KEPT = 14, 7
CAL_RUNS = ARMS_KEPT + 1
CAL_PAIRS = CAL_RUNS * (CAL_RUNS - 1) // 2
assert CAL_PAIRS == 28
T0_STEPS, T0_MIN, T1_MIN = 20, 10, 25
# 「TPU 自己剩下的」三种口径（别混）：
TPU_OWN_NOREMAT_975 = 1.7e-3     # TPU 对不重算的精确轨迹 ④′，第 975 步
TPU_OWN_OLDCFG_1000 = 2.6e-3     # 生产近似配置、FP32 累加开关打开前，第 1000 步
TPU_OWN_CURCFG_1000 = 2.3e-3     # 当前配置（开关打开后），第 1000 步


# ── 小工具：BF16 舍入（RNE），给图里「真算一遍」用 ──────────────────────
def bf16(x):
    """把 float 按「就近舍入、平局取偶」舍到 BF16，返回 float。"""
    u = struct.unpack(">I", struct.pack(">f", x))[0]
    u += 0x7FFF + ((u >> 16) & 1)
    return struct.unpack(">f", struct.pack(">I", (u >> 16) << 16))[0]


assert bf16(1.0 + 2 ** -8) == 1.0            # 正好一半，舍到偶数
assert bf16(1.0 + 2 ** -7) == 1.0 + 2 ** -7


def series():
    """逐步序列（第 1–1000 步）：loss 相对差与 param_norm 差。"""
    with open(os.path.join(HERE, "topic09_series.json"), encoding="utf-8") as fh:
        return json.load(fh)


def mean(xs):
    return sum(xs) / float(len(xs))


def tpuself():
    """第二次迁移的 TPU 自比与 GPU 留一法（topic09_tpuself.json，导出自对齐记录 alignments/04-tpu-self，实测）。"""
    with open(os.path.join(HERE, "topic09_tpuself.json"), encoding="utf-8") as fh:
        return json.load(fh)


# ── 2026-11-05 夜：新结论（一手材料 11-05 23:00 版）────────────────────────────────
def sci(x):
    """1.24e-07 → 1.24e-7（课件里的写法）"""
    m, e = ("%.2e" % x).split("e")
    m = m.rstrip("0").rstrip(".") if "." in m else m
    return "%se%d" % (m, int(e))


def show(v):
    """«名字» 占位符落进正文的写法：字符串原样、整数原样、小数按有效数字。"""
    if isinstance(v, str):
        return v
    if isinstance(v, int):
        return format(v, ",") if v >= 10000 else str(v)
    return sci(v) if abs(v) < 1e-2 else ("%g" % v)


EXACT_PAIR1_PN = 6.9e-9          # 精确轨迹只看一对扰动时的 param_norm 带符号均值（旧的「小 18 倍」就是它）
assert abs(GPU_RERUN_PN / EXACT_PAIR1_PN - 18) < 0.5
XCHK_SELF_PN, XCHK_SELF_LOSS, XCHK_SELF_GN = "4.00", "2.03", "2.24"   # 交叉核对（只报）：尺子换成精确轨迹自己 10 对
EP8_PN = 3.52e-7                 # v5p 当前配置改成 EP＝8：param_norm 对精确轨迹的带符号均值（原 FSDP 4.27e-7）
EP8_DROP_PCT = "%d%%" % round((1 - EP8_PN / ACC[0][1]) * 100)
assert EP8_DROP_PCT == "18%"
EP8_R = "2.85"                   # 判定文件里的比值（N 未舍入是 1.237e-7）
assert abs(EP8_PN / 1.237e-7 - 2.85) < 0.01
# 第 1 步梯度与路由翻转（隔离回放；被测梯度由 Adam 一阶动量反推）
STEP1_TENSORS, STEP1_ALL, STEP1_EXPERT = 50, 1.28e-3, 0.102    # 比到的 50 个张量合起来 rel_L2；第 3 层专家 0 的 fc1/fc2
assert 70 < STEP1_EXPERT / STEP1_ALL < 90                       # 「大了近百倍」（约 80 倍）
FLIP_SLOTS, FLIP_TOKENS = "0.44–0.56%", "3.5–4.5%"            # 路由槽位翻转比例；至少换了一个专家的 token 比例
E0_TOKENS, E0_OUT, E0_IN = 139, 3, 1                            # 第 3 层专家 0：4096 个 token 里分到 139 个，换出 3、换进 1
FLIP_MARGIN = "2.7e-5 到 6.3e-5"                                 # 翻转 token 的第 8、9 名「分数＋偏置」分差中位数（各层范围）
EXPERTS, TOPK = 192, 8

# ── 2026-11-05 · fig9-baseline：10 份 GPU 运行各自的尾段 loss 偏移（实测）──────────────────
# 默认设置（不开分布式优化器）的 10 份 GPU 运行，各自对第一份算第 301–1000 步 loss 的带符号均值；
# 下标 0 是第一份自己（＝ 0）。出处：对齐实践的案例记录「噪声带」那一行与它的结果文件（逐份值）。
GPU_RUNS10_TAIL = (0.0, -3.78e-5, 2.5e-7, 2.44e-6, -6.51e-5, -2.15e-5, -9.2e-6, -3.47e-6, -4.94e-5, -6.19e-5)
GPU_RUNS10_TAIL_MEAN = sum(GPU_RUNS10_TAIL) / len(GPU_RUNS10_TAIL)
_sd10 = math.sqrt(sum((x - GPU_RUNS10_TAIL_MEAN) ** 2 for x in GPU_RUNS10_TAIL) / (len(GPU_RUNS10_TAIL) - 1))
assert len(GPU_RUNS10_TAIL) == 10
assert abs(GPU_RUNS10_TAIL_MEAN - GPU_RUNS10_MEAN) < 0.05e-5, GPU_RUNS10_TAIL_MEAN     # −2.46e-5 ≈ −2.5e-5
assert abs(_sd10 - GPU_RUNS10_STD) < 0.05e-5, _sd10                                      # 样本标准差 2.68e-5 ≈ 2.7e-5
assert abs(min(GPU_RUNS10_TAIL) - GPU_RUNS10_LO) < 0.05e-5 and abs(max(GPU_RUNS10_TAIL) - GPU_RUNS10_HI) < 0.05e-6   # −6.51e-5 ≈ −6.5e-5；2.44e-6 ≈ 2.4e-6
GPU_RUNS10_BASE_RANK = sorted(GPU_RUNS10_TAIL, reverse=True).index(0.0) + 1             # 基线在 10 份里从高往低排第几
assert GPU_RUNS10_BASE_RANK == 3            # 比它高的两份只高 2.5e-7、2.44e-6 —— 「正好在最上沿」
TPU_ARMS_VS_MEAN10 = tuple(x - GPU_RUNS10_TAIL_MEAN for x in TPU_ARMS)                    # 本课推导：扣掉 10 份均值
assert abs(TPU_ARMS_VS_MEAN10[0] - (-1.41e-4)) < 0.01e-4 and abs(TPU_ARMS_VS_MEAN10[2] - (-0.78e-4)) < 0.01e-4
assert round(max(TPU_ARMS_VS_MEAN10) * 1e4, 1) == -0.8 and round(min(TPU_ARMS_VS_MEAN10) * 1e4, 1) == -1.4   # 正文「−0.8e-4 到 −1.4e-4」

# ── 2026-11-05 · fig9-layer0：残差流固定探针，与第 0 层权重互换（实测）──────────────────────
# 探针：两边在同一条固定序列上每 25 步另跑一次前向，各层输出在 token 上取均值向量，投到 w ＝ γ⊙û 上
#   （γ：第一份 GPU 运行该步最后归一化的 scale；û：它该步最后归一化输出均值的单位方向。最后归一化输出 ＝ γ⊙x/rms(x)，
#    所以这个投影就是这一层对「最后隐状态均值沿 û 那一份」的贡献）。数值 ＝（TPU − 第一份 GPU）÷ |第一份 GPU|，3 位有效数字。
L0_PROBE_STEPS = tuple([1] + list(range(25, 1000, 25)))
L0_PROBE = {
    "embedding": (3.37e-08, 2.68e-07, 1.94e-06, -1.54e-05, -1.04e-05, -5.14e-05, -4e-05, -5.92e-05, 2.93e-06, -8.01e-05, -5.51e-05, -8.09e-05, -7.08e-05, -0.000111, -7.44e-05, -0.000117, -0.000115, -5.45e-05, -7.65e-05, -0.000207, -6.1e-05, -9.19e-05, -7.08e-05, -4.51e-05, -8.28e-05, -5.73e-05, -0.000139, -0.00014, -0.000196, -0.000225, -6.35e-05, -7.36e-05, -7.37e-05, -0.000126, -9.77e-05, -1.86e-05, -0.000149, -7.37e-05, -0.00019, -0.000162),
    "第 0 层": (2.73e-05, 0.000135, -0.000143, -0.000853, -0.00176, -0.00299, -0.00354, -0.00333, -0.00448, -0.00545, -0.00423, -0.00466, -0.00535, -0.00477, -0.0047, -0.00478, -0.0056, -0.0051, -0.00528, -0.0056, -0.00478, -0.00505, -0.00453, -0.00526, -0.00552, -0.00562, -0.00483, -0.00496, -0.00549, -0.00509, -0.00461, -0.00506, -0.00521, -0.00546, -0.00446, -0.00508, -0.00496, -0.00452, -0.00494, -0.00444),
    "第 1 层": (-2.18e-05, 6.07e-05, -0.00036, -0.00125, -0.00231, -0.00387, -0.00433, -0.00495, -0.00589, -0.00628, -0.00559, -0.00551, -0.00684, -0.0065, -0.00657, -0.00636, -0.00718, -0.00663, -0.00763, -0.00715, -0.00662, -0.00722, -0.00583, -0.00672, -0.00692, -0.0068, -0.00661, -0.00682, -0.00659, -0.00615, -0.00585, -0.0065, -0.00638, -0.00701, -0.00572, -0.00583, -0.00581, -0.00586, -0.00599, -0.00544),
    "第 2 层": (-3.13e-05, -1.2e-05, -0.000528, -0.00133, -0.00227, -0.00441, -0.00494, -0.00543, -0.00671, -0.00705, -0.00561, -0.00687, -0.00724, -0.00849, -0.00735, -0.00669, -0.00782, -0.00893, -0.00883, -0.00811, -0.00685, -0.00717, -0.00633, -0.00683, -0.00704, -0.00845, -0.00752, -0.00669, -0.00768, -0.00761, -0.00589, -0.00736, -0.0066, -0.0078, -0.00714, -0.00572, -0.00841, -0.00718, -0.00688, -0.00609),
    "第 3 层": (-5.34e-05, -1.41e-05, -0.000555, -0.00176, -0.00333, -0.00517, -0.00702, -0.00792, -0.0101, -0.00971, -0.00944, -0.0112, -0.0114, -0.0135, -0.0118, -0.0134, -0.0118, -0.0143, -0.0139, -0.0146, -0.0122, -0.0125, -0.0126, -0.0132, -0.0137, -0.0138, -0.0152, -0.0144, -0.014, -0.0159, -0.0132, -0.013, -0.0144, -0.0155, -0.0141, -0.0131, -0.0141, -0.0137, -0.0154, -0.014),
}
# 第 0 层，第二份 GPU 对第一份（噪声参照）
L0_PROBE_GPU_PAIR = (-5.63e-09, -1.36e-06, 3.65e-05, -5.51e-05, 7.69e-05, 4.35e-05, -6.76e-05, 0.000114, 0.000159, -0.00109, 0.000187, 8.97e-05, -0.000603, -0.000152, 0.000132, 0.000112, -0.00022, -0.000237, -0.000158, -0.000337, -0.000399, -0.000682, -0.000163, -0.000426, -0.000519, -0.000868, -7.48e-05, -0.00022, -0.000442, -0.000455, 0.000254, -3.05e-05, -0.00102, -0.000656, -4.72e-05, -0.000231, -0.000287, 0.000116, -0.000897, 2.73e-05)
assert all(len(v) == len(L0_PROBE_STEPS) == 40 for v in list(L0_PROBE.values()) + [L0_PROBE_GPU_PAIR])
_i100, _i300, _i50 = L0_PROBE_STEPS.index(100), L0_PROBE_STEPS.index(300), L0_PROBE_STEPS.index(50)
assert abs(L0_PROBE["第 0 层"][_i100] - (-1.8e-3)) < 0.05e-3 and abs(L0_PROBE["第 0 层"][_i300] - (-5.4e-3)) < 0.06e-3
assert L0_PROBE["第 0 层"][_i50] < 0 < L0_PROBE["第 0 层"][_i50 - 1]                    # 第 50 步起转为偏低
assert all(abs(L0_PROBE["第 %d 层" % k][_i300] - _v) < 0.06e-3 for k, _v in enumerate((-5.4e-3, -6.8e-3, -7.2e-3, -1.14e-2)))   # 往后逐层放大
assert max(abs(x) for x in L0_PROBE["embedding"]) < 2.3e-4                              # embedding 两边一样（GPU 两次之间最大 1.5e-4）
# 第 0 层权重互换：取第一份 GPU 运行该步的第 0 层输入，在 CPU 上 FP32 重算第 0 层，只换权重。
#   数值 ＝（换上别人的权重 − 用第一份 GPU 自己的权重）的输出均值沿 w 的投影 ÷ |后者的投影|，w 同上。
L0_SWAP_100 = (-1.94e-3, 7.5e-6, -2.4e-4)       # 第 100 步，换成 TPU ／ 第二份 GPU ／ 第三份 GPU 的第 0 层权重
L0_SWAP_50 = (-2.1e-4, 1.8e-5, 5.5e-5)          # 第 50 步，同上
L0_SWAP_ONE_100 = (("注意力输出投影", -1.33e-3), ("qkv 投影", -2.6e-4), ("MLP 输出投影", -2.2e-4))   # 第 100 步，只换一组（换成 TPU 的）
L0_SWAP_SMALL = 6e-5                             # 两个归一化的 scale、q／k 归一化：都在这以下
assert abs(L0_SWAP_100[0] / L0_PROBE["第 0 层"][_i100] - 1) < 0.15                       # 换权重重算 ≈ 探针实测：差全在权重上
assert abs(L0_SWAP_50[0] / L0_PROBE["第 0 层"][_i50] - 1) < 0.5
assert max(abs(x) for x in L0_SWAP_100[1:]) < abs(L0_SWAP_100[0]) / 5
assert abs(L0_SWAP_ONE_100[0][1]) > 0.6 * abs(L0_SWAP_100[0])                            # 大头在注意力输出投影（约七成）

# ── 2026-11-05 · fig9-excess 的 📌（实测，混元 3 · v5p）──────────────────────────────
# 只加「关掉 excess precision」这一个编译参数：
QKV_VS_GPU_NOEXC = 3.49e-3                  # 单层回放 qkv 输出对 GPU 的 rel_L2（开着时是 QKV_VS_GPU_BEFORE）
assert QKV_VS_GPU_NOEXC > QKV_VS_GPU_BEFORE  # 关掉反而离 GPU 更远（这一层每个张量都是，约远 10%–60%）
NOEXC_LOSS_TAIL, NOEXC_LOSS_TAIL_ON = -6.0e-5, -1.27e-4   # 1000 步，第 301–1000 步 loss 偏移：关掉 ／ 同配置开着（案例记录同一格里的对照）
assert 0.4 < NOEXC_LOSS_TAIL / NOEXC_LOSS_TAIL_ON < 0.6     # 「小了一半」
RMSNORM_BITWISE_EXC, RMSNORM_BITWISE_NOEXC = "99.9987%", "74.1%"   # 默认写法的 RMSNorm 与 GPU（FP32 算完舍一次）逐位相同的比例

# L3（11-06）：「离 10 份 GPU 均值几个标准差」只用原始 10 个值算一次，fig-ruler、fig-baseline、正文、讲义共用
SIGMAS_10 = tuple((x - GPU_RUNS10_TAIL_MEAN) / _sd10 for x in TPU_ARMS)
assert [round(abs(z), 1) for z in SIGMAS_10] == [5.3, 4.1, 2.9], SIGMAS_10
SIGMA_RANGE = "2.9 到 5.3"

# ── 2026-11-05 · 图修订：交叉核对二（TPU 执行噪声 28 对）其余三项、干预臂 loss 前后半（实测）──────────────
# 出处：对齐实践的 golden 总表「第一层结果」表，交叉核对行（噪声 ＝ TPU 当前配置重标的 28 对，被测 v5p 当前配置）。
XCHK_TPU_LOSS, XCHK_TPU_TAIL, XCHK_TPU_GN = 1.81, 1.42, 0.99   # loss 全程待定、尾段待定、grad_norm 过；只有 param_norm（XCHK_TPU_PN）超 2
assert 1 < XCHK_TPU_LOSS <= 2 and 1 < XCHK_TPU_TAIL <= 2 and XCHK_TPU_GN <= 1 and XCHK_TPU_PN > 2
# 出处：案例 01「dh 里的 logit 水平分量」行：两边都去掉 dh 沿 w̄ 的分量，1000 步，第 301–600 / 601–1000 步 loss 偏移。
REMOVE_LOSS_H1, REMOVE_LOSS_H2 = 4.9e-5, -9.6e-5
assert abs((300 * REMOVE_LOSS_H1 + 400 * REMOVE_LOSS_H2) / 700 - REMOVE_LOSS) < 0.1e-5     # 按步数加权 ＝ −3.39e-5 ≈ −3.4e-5
assert 0.5 < REMOVE_LOSS_H2 / TPU_LOSS_TAIL < 0.65                                          # 后半仍剩约六成（−9.6e-5 ÷ −1.66e-4 ≈ 0.58）

# ── 2026-11-05 · 数字锁：课件正文（topic09-build.py）原先手打的数，一律从这里取 ───────────────────
# 正文用 «名字» 占位符，构建时按 show() 落地。show() 写不出原文写法的（千分位、尾零、负号 −、全角 ＋、百分号、
# 固定小数位），另立一个 *_T 字符串常数：由数值常数现算，再用断言钉死跟原正文逐字一致 —— 数一改，断言先响。
# 出处没写的，沿用上面同名数值常数的出处；新数逐条写出处，拿不准的写「来源：课件既有数，待核」。
import re as _re


def tx(x, spec="%.2e", plus=False, pct=False):
    """课件写法：负号用 −（U+2212）；plus=True 时正数前加全角 ＋；科学计数指数去前导零（e-04 → e-4）；pct=True 先乘 100 再加 %。"""
    v = x * 100 if pct else x
    s = _re.sub(r"e([+-])0*(\d)", lambda m: "e" + ("-" if m.group(1) == "-" else "") + m.group(2), spec % abs(v))
    if pct:
        s += "%"
    return ("−" if v < 0 else ("＋" if plus else "")) + s


def _lock(name, got, want):
    assert got == want, "%s 落到正文是 %r，课件原文是 %r" % (name, got, want)


# 开场热身：格距（本课推导：FP32 尾数 23 位 → 1 附近格距 2⁻²³；BF16 7 位 → 2⁻⁷）
FP32_GAP1_T = tx(2.0 ** -23, "%.1e"); _lock("FP32_GAP1_T", FP32_GAP1_T, "1.2e-7")
BF16_GAP1_T = "%.4f" % BF16_GAP1; _lock("BF16_GAP1_T", BF16_GAP1_T, "0.0078")
# 第 1 步 loss
LOSS1_TPU_T, LOSS1_GPU_T = "%.6f" % LOSS1_TPU, "%.6f" % LOSS1_GPU
_lock("LOSS1_TPU_T", LOSS1_TPU_T, "11.465785"); _lock("LOSS1_GPU_T", LOSS1_GPU_T, "11.465756")
D1_T, D1_PLUS_T, D1_PPM_T = tx(D1, "%.1e"), tx(D1, "%.1e", plus=True), "%.1f" % (D1 * 1e6)
_lock("D1_T", D1_T, "2.5e-6"); _lock("D1_PLUS_T", D1_PLUS_T, "＋2.5e-6"); _lock("D1_PPM_T", D1_PPM_T, "2.5")
# 权重逐位
N_TENSORS_T, N_BF16_TENSORS_T = format(N_TENSORS, ","), format(N_BF16_TENSORS, ",")
_lock("N_TENSORS_T", N_TENSORS_T, "1,781"); _lock("N_BF16_TENSORS_T", N_BF16_TENSORS_T, "1,778")
QUIZ_SAMPLED = 100                                   # §2 考考自己：假设只抽 100 个张量比（题设）
QUIZ_UNSAMPLED_T = format(N_TENSORS - QUIZ_SAMPLED, ","); _lock("QUIZ_UNSAMPLED_T", QUIZ_UNSAMPLED_T, "1,681")
MASTER_CMP_TENSORS = 50                              # 加载后比 FP32 主权重：GPU 那边抓下来的张量数。出处：对齐实践 v5p 验收的 L0「载入后的 FP32 主权重」检查，50 个张量全部一致
NATIVE_SAME_RANK_T = format(NATIVE_SAME_RANK, ","); _lock("NATIVE_SAME_RANK_T", NATIVE_SAME_RANK_T, "2,000")
MICRO_SPLIT_EMB_X = 31                               # 只换梯度累积切分，embedding 梯度差 ÷ GPU 重跑差。出处：对齐报告偏差台账「梯度累积的分组」行
# 参数范数、梯度范数的量级
PN_LEVEL0 = 3042.377                                 # GPU 第一份运行第 1 步参数范数（fig9-drift 的 📌 实测：3042.377 → 3042.503）
PN_LEVEL_T = "%d" % PN_LEVEL0; _lock("PN_LEVEL_T", PN_LEVEL_T, "3042")
CLIP_GRAD = 1.0                                      # 裁剪阈值 clip_grad
assert CLIP1 == CLIP_GRAD / GRADNORM1
GRADNORM1_T, CLIP1_T, CLIP_GRAD_T = "%.1f" % GRADNORM1, "%.3f" % CLIP1, "%.1f" % CLIP_GRAD
_lock("GRADNORM1_T", GRADNORM1_T, "17.5"); _lock("CLIP1_T", CLIP1_T, "0.057"); _lock("CLIP_GRAD_T", CLIP_GRAD_T, "1.0")
# ULP
ULP_FP32_PER_BF16 = 2 ** 16                          # BF16 一格 ＝ FP32 的 2¹⁶ 格（尾数差 16 位，本课推导）
assert FORMATS[0][2] - FORMATS[2][2] == 16
ULP1_T, ULP17_T = tx(ULP1_SHARE, "%.1f", pct=True), tx(ULP17_SHARE, "%.0f", pct=True)
_lock("ULP1_T", ULP1_T, "1.3%"); _lock("ULP17_T", ULP17_T, "67%")
# 第 1 步专家梯度与路由翻转
STEP1_X = int(round(STEP1_EXPERT / STEP1_ALL, -1)); assert STEP1_X == 80           # 「大了约 80 倍」
E0_SWAPPED = E0_OUT + E0_IN
E0_SQRT_T = "%.2f" % math.sqrt(E0_SWAPPED / E0_TOKENS); _lock("E0_SQRT_T", E0_SQRT_T, "0.17")   # 本课推导（推断）
BATCH_TOKENS = 4096                                  # 隔离回放那一批的 token 数
BATCH_TOKENS_T = format(BATCH_TOKENS, ","); _lock("BATCH_TOKENS_T", BATCH_TOKENS_T, "4,096")
FLIP_SLOTS_L1 = BATCH_TOKENS * TOPK; assert FLIP_SLOTS_L1 == 32768                    # 第 1 层槽位数
FLIP_MARGIN_DASH_T = FLIP_MARGIN.replace(" 到 ", " – "); _lock("FLIP_MARGIN_DASH_T", FLIP_MARGIN_DASH_T, "2.7e-5 – 6.3e-5")
# 出处（实测）：逐层隔离回放的路由翻转统计（路由输入两边 rel_L2、翻转与全部 token 的分差中位数、喂相同输入 logits rel_L2 3.8e-6–4.0e-6）；同一状态两侧全程 FP32 前向逐点比（第 1 层 1/32768 个槽位不同，分差 1.6e-6）
ROUTE_IN_RELL2_LO, ROUTE_IN_RELL2_HI = 3.4e-3, 3.9e-3                 # 路由输入两边 rel_L2，第 1–3 层
ALL_MARGIN_LO, ALL_MARGIN_HI = 3.1e-4, 8.0e-4                         # 全部 token 第 8、9 名分差中位数
ALL_MARGIN_HI_T = tx(ALL_MARGIN_HI, "%.1e"); _lock("ALL_MARGIN_HI_T", ALL_MARGIN_HI_T, "8.0e-4")
ROUTE_SAME_IN_DIFF = 4e-6                                             # 喂相同路由输入时两边 logits 仍差（3.8e-6–4.0e-6 取上沿）
FP32_FLIP_MARGIN = 1.6e-6                                             # 两边全程 FP32 时第 1 层翻的那个 token 的分差
# 参考档
REF_LOSS_GATE, REF_RELL2_GATE = 2e-4, 0.01           # 另一个已上线迁移项目的固定线（首步与窗口 loss；逐层 rel_L2）
LAYER_RELL2_FAIL = (0.016, 0.020, 0.028)             # v5p 第 1 步逐层回放，MoE 第 1–3 层的 rel_L2（实测）；这几层参考档「不适用」（受路由翻转影响），严格档 R 比过
assert all(x > REF_RELL2_GATE for x in LAYER_RELL2_FAIL)
L_RELL2_1_T, L_RELL2_2_T, L_RELL2_3_T = ("%.3f" % x for x in LAYER_RELL2_FAIL)
_lock("L_RELL2_2_T", L_RELL2_2_T, "0.020")
# 窗口化开关、golden 一步步（出处：fig9-golden 的同一组实测：v5p 开关前配置对 ②③④ 的参数范数带符号均值）
R_QKV_AFTER_T = "%.3f" % R_QKV_AFTER; _lock("R_QKV_AFTER_T", R_QKV_AFTER_T, "1.000")
GOLDEN_PN_X = (2.93e-6, 1.25e-6, 4.36e-7)
GOLDEN_R = tuple(x / GPU_RERUN_PN for x in GOLDEN_PN_X)
GOLDEN_R2_T, GOLDEN_R3_T, GOLDEN_R4_T, GOLDEN_R4_1_T = "%.1f" % GOLDEN_R[0], "%.1f" % GOLDEN_R[1], "%.2f" % GOLDEN_R[2], "%.1f" % GOLDEN_R[2]
_lock("GOLDEN_R2_T", GOLDEN_R2_T, "23.6"); _lock("GOLDEN_R3_T", GOLDEN_R3_T, "10.1"); _lock("GOLDEN_R4_T", GOLDEN_R4_T, "3.52"); _lock("GOLDEN_R4_1_T", GOLDEN_R4_1_T, "3.5")
# 第一关验收
ACC_PN_X, ACC_PN_R = ACC[0][1], ACC[0][3]
ACC_LOSS_ALL_R, ACC_LOSS_TAIL_R, ACC_GN_R = ACC[1][3], ACC[2][3], ACC[3][3]
ACC_PN_X_07_T = "%.2e" % ACC_PN_X; _lock("ACC_PN_X_07_T", ACC_PN_X_07_T, "4.27e-07")   # 总览表一格的原样写法
XCHK_SELF_TAIL = XCHK_SELF["loss 尾段"]
EP8_DELTA = ACC_PN_X - EP8_PN
TPU_NOISE_PN_N = 8.5e-8                              # TPU 执行噪声 28 对的参数范数噪声上限。出处：日常回归的噪声标定记录（v5p，28 对）param_norm 带符号均值上限 8.49e-8
assert EP8_DELTA < TPU_NOISE_PN_N and sci(EP8_DELTA) == "7.5e-8"
WINDOWED_OFF_PN_DROP_T = "3%"                        # 关掉 collective matmul 窗口化，参数范数差只少 3%。出处：案例记录「TPU 窗口化 collective matmul」行（8.74e-3 → 8.50e-3）
# excess precision 那段：TPU 三种跑法彼此的散布（也就是只改片上内存上限那一臂挪的量）
ARMS_SPREAD = max(TPU_ARMS) - min(TPU_ARMS)
assert ARMS_SPREAD == TPU_ARMS[2] - TPU_ARMS[0]
ARMS_SPREAD_T = tx(ARMS_SPREAD, "%.0e"); _lock("ARMS_SPREAD_T", ARMS_SPREAD_T, "6e-5")
# 步号错位、分段
MISALIGN_OK_T, MISALIGN_OFF_T = tx(MISALIGN[0]), tx(MISALIGN[1])
_lock("MISALIGN_OK_T", MISALIGN_OK_T, "1.53e-4"); _lock("MISALIGN_OFF_T", MISALIGN_OFF_T, "1.41e-2")
MISALIGN_X_FLOOR = 90; assert MISALIGN_X_FLOOR < MISALIGN[1] / MISALIGN[0] < MISALIGN_X_FLOOR + 10   # 「大 90 多倍」
LOSS_TAIL_ROUGH_T = "−1.5e-4"                        # 「300 步以后稳定在 −1.5e-4 左右」。出处：案例记录 §3.1 原话（只断言落在三种跑法范围内）
assert min(TPU_ARMS) <= -1.5e-4 <= max(TPU_ARMS)
# 排除清单
WD_MASK_X = 13                                       # 权重衰减掩码反过来，残差大 13 倍以上。出处：案例记录排除表第 1 行
UPD1_RATIO_LO_T, UPD1_RATIO_HI_T = "0.997", "1.003"  # 第 1 步逐张量更新量之比。出处：案例记录排除表第 3 行（50 个张量）
RARE_VHAT_DIFF_T = "0–4%"                            # 冷门行 √v̂ 两边只差 0–4%。出处：案例记录冷门行 w·u 那一节
# 输出头 dgrad、同状态各算一步
HEAD_PCT_S1_T, HEAD_PCT_S100_T = tx(abs(HEAD_BF16_S1), "%.3f", pct=True), tx(abs(HEAD_BF16_S100), "%.3f", pct=True)
_lock("HEAD_PCT_S1_T", HEAD_PCT_S1_T, "0.065%"); _lock("HEAD_PCT_S100_T", HEAD_PCT_S100_T, "0.061%")
HEAD_RATIO_S1, HEAD_RATIO_S100 = int(round(HEAD_BF16_S1 / HEAD_TPU_S1)), int(round(HEAD_BF16_S100 / HEAD_TPU_S100))
assert (HEAD_RATIO_S1, HEAD_RATIO_S100) == (27, 8)                    # 本课推导：GPU ÷ TPU 对 FP64 的误差
SWAP_TPU_T, SWAP_FIX_T, SWAP_GPU_T = (tx(v, "%.2f") + "%" for _, v, _r in SWAP)
SWAP_GPU_ABS_T = "%.2f%%" % abs(SWAP[2][1])
_lock("SWAP_TPU_T", SWAP_TPU_T, "0.00%"); _lock("SWAP_FIX_T", SWAP_FIX_T, "−0.10%"); _lock("SWAP_GPU_T", SWAP_GPU_T, "−0.36%"); _lock("SWAP_GPU_ABS_T", SWAP_GPU_ABS_T, "0.36%")
SWAP_RES_T = "%.1f%%" % max(abs(x) for x in SWAP[0][2]); _lock("SWAP_RES_T", SWAP_RES_T, "0.1%")   # TPU 区间最宽处 → 「±0.1% 以内」
SWAP_STATE_TENSORS = 1193                            # 口径：Megatron 布局（qkv 合一、gate/up 合一）；1781 − 1193 ＝ 588 ＝ 192 专家 × 3 层 gate/up 576 ＋ qkv 4 层 × 2 ＋ 稠密层 1 ＋ 共享专家 3。
# 状态互换导入 TPU、逐个核对 sha256 的张量数（fig9-swap 的 📌 实测）
assert N_TENSORS - SWAP_STATE_TENSORS == 192 * 3 + 4 * 2 + 1 + 3, "张量数口径对不上"
# 精确轨迹那段
PERTURB = 1e-6                                       # 扰动版：给第 0 层一处权重乘 (1 ＋ 1e-6)
FP32_TRAIN_LOSS_GAP_T = "1.75%"                      # 纯 FP32 训练前 100 步 loss 比 BF16 低。出处：golden 设计文档「纯 FP32 训练」那一条
EXACT_TRAJ_PN975_T, EXACT_TPU_PN975_T = tx(EXACT_POS[1][1], plus=True), tx(EXACT_POS[2][1], plus=True)
DGRADX_LO_T, DGRADX_HI_T = tx(DGRADX_RANGE[0], plus=True), tx(DGRADX_RANGE[1], plus=True)
_lock("EXACT_TRAJ_PN975_T", EXACT_TRAJ_PN975_T, "＋1.82e-2"); _lock("EXACT_TPU_PN975_T", EXACT_TPU_PN975_T, "＋1.99e-2")
_lock("DGRADX_LO_T", DGRADX_LO_T, "＋1.23e-2"); _lock("DGRADX_HI_T", DGRADX_HI_T, "＋1.28e-2")
# 拆账与「剩多少」
DECOMP_HEAD_T, DECOMP_REST_T, DECOMP_TPU_T = ("%.0f%%" % (s * 100) for s in DECOMP_SHARE)
_lock("DECOMP_HEAD_T", DECOMP_HEAD_T, "59%"); _lock("DECOMP_REST_T", DECOMP_REST_T, "29%"); _lock("DECOMP_TPU_T", DECOMP_TPU_T, "12%")
DECOMP_GPU_T = "%.0f%%" % ((DECOMP_SHARE[0] + DECOMP_SHARE[1]) * 100); _lock("DECOMP_GPU_T", DECOMP_GPU_T, "88%")
LOSS_FIXHEAD_SHARE_T = "%.0f%%" % (LOSS_FIXHEAD_MINUS_PROD / LOSS_TPU_VS_PROD_TAIL * 100); _lock("LOSS_FIXHEAD_SHARE_T", LOSS_FIXHEAD_SHARE_T, "73%")
TPU_OWN_NOREMAT_975_T, TPU_OWN_OLDCFG_1000_T, TPU_OWN_CURCFG_1000_T = (tx(v, "%.1e", plus=True) for v in (TPU_OWN_NOREMAT_975, TPU_OWN_OLDCFG_1000, TPU_OWN_CURCFG_1000))
_lock("TPU_OWN_NOREMAT_975_T", TPU_OWN_NOREMAT_975_T, "＋1.7e-3"); _lock("TPU_OWN_OLDCFG_1000_T", TPU_OWN_OLDCFG_1000_T, "＋2.6e-3"); _lock("TPU_OWN_CURCFG_1000_T", TPU_OWN_CURCFG_1000_T, "＋2.3e-3")
# 踩坑表、速查卡
FP32_MASTER_FAKE_DIFF_T = "1.5%"                     # 拿 FP32 主权重当参照量出的假差。出处：案例记录的更正条（拿 FP32 主权重当参照，结论作废）
EXACT_PAIR1_X = int(round(GPU_RERUN_PN / EXACT_PAIR1_PN)); assert EXACT_PAIR1_X == 18
COS_0999_RELL2_T = "%.1f%%" % (COS_0999_RELL2 * 100); _lock("COS_0999_RELL2_T", COS_0999_RELL2_T, "4.5%")
R_ONE_EXTRA_T = "%.2f" % R_ONE_EXTRA; _lock("R_ONE_EXTRA_T", R_ONE_EXTRA_T, "1.41")
HEAD_K1, HEAD_BIAS1 = HEAD_BIAS_K[0]
HEAD_K3, HEAD_BIAS3 = HEAD_BIAS_K[2]
assert HEAD_K3 == VOCAB
# 出处台账
GPU_RUNS10_MEAN_T = tx(GPU_RUNS10_MEAN, "%.1e"); _lock("GPU_RUNS10_MEAN_T", GPU_RUNS10_MEAN_T, "−2.5e-5")
TPU_ARM_LO_T, TPU_ARM_HI_T = tx(max(TPU_ARMS)), tx(min(TPU_ARMS))
_lock("TPU_ARM_LO_T", TPU_ARM_LO_T, "−1.03e-4"); _lock("TPU_ARM_HI_T", TPU_ARM_HI_T, "−1.66e-4")
TPU_LOSS_TAIL_T = tx(TPU_LOSS_TAIL); _lock("TPU_LOSS_TAIL_T", TPU_LOSS_TAIL_T, "−1.66e-4")
TPU_TAIL_STEP_STD = 1.33e-4                          # TPU 对 GPU 第一份逐步差的标准差，第 301–1000 步（fig9-dt 的 📌 实测）
INCONCL_PAIRS_T = " ／ ".join(str(k) for k in sorted(INCONCL)); _lock("INCONCL_PAIRS_T", INCONCL_PAIRS_T, "3 ／ 15 ／ 28")
INCONCL_T = "、".join("%d／%d" % INCONCL[k] for k in sorted(INCONCL)); _lock("INCONCL_T", INCONCL_T, "8／46、3／9、1／46")
TC_K1, TC_BIAS1 = TC_BIAS_K[0]
TC_K2, TC_BIAS2 = TC_BIAS_K[1]
TC_K3, TC_BIAS3 = TC_BIAS_K[2]
TC_BIAS2_T, TC_K3_T = tx(TC_BIAS2, "%.1e"), str(TC_K3); _lock("TC_BIAS2_T", TC_BIAS2_T, "7.0e-5"); _lock("TC_K3_T", TC_K3_T, "65536")
ROUTER_TPU_FP64_HI, ROUTER_GPU_FP64_HI = 5.7e-8, 4.0e-6   # 路由打分对 FP64 三层的上沿（下沿是 ROUTER_*_FP64）。出处：逐层回放报告 L1 注 2（TPU 4.8e-8–5.7e-8、GPU 3.8e-6–4.0e-6）
ROUTER_GPU_FP64_HI_T = tx(ROUTER_GPU_FP64_HI, "%.1e"); _lock("ROUTER_GPU_FP64_HI_T", ROUTER_GPU_FP64_HI_T, "4.0e-6")
MOMENTUM_INV_LO, MOMENTUM_INV_HI = 2e-7, 4e-7        # 反推梯度对直接抓的梯度 rel_L2（GPU 上）。出处：案例记录「动量反推梯度」核对（GPU 上对直接抓的梯度）
PN_GPU_PAIR_MAX = 6.6e-4                             # GPU 两次之间参数范数差的最大值（逐步序列现算，见下）
assert abs(max(abs(x) for x in series()["pnorm_diff_gpu_b_minus_gpu_a"]) - PN_GPU_PAIR_MAX) < 0.05e-4
PN_DIFF_300_T = tx(series()["pnorm_diff_tpu_minus_gpu_a"][299], "%.1e"); _lock("PN_DIFF_300_T", PN_DIFF_300_T, "4.6e-3")   # 第 300 步 TPU − GPU
ROWS300_NEVER = ROWS_300[0][1]
RARE_EPS_FRAC_T = "%.0f%%" % (RARE_EPS_FRAC * 100); _lock("RARE_EPS_FRAC_T", RARE_EPS_FRAC_T, "92%")
GROUP_HEAD_SHARE_T = "%.0f%%" % (GROUP_HEAD_300 / GROUP_TOTAL_300 * 100); _lock("GROUP_HEAD_SHARE_T", GROUP_HEAD_SHARE_T, "87%")
GROUP_TOTAL_300_T = tx(GROUP_TOTAL_300, "%.1e"); _lock("GROUP_TOTAL_300_T", GROUP_TOTAL_300_T, "4.4e-3")
LSE25_T, LSE50_T, LSE100_T, LSE300_T = tx(LSE_PROBE[0][1], "%.4f"), tx(LSE_PROBE[1][1], "%.3f"), tx(LSE_PROBE[2][1], "%.3f"), tx(LSE_PROBE[3][1], "%.3f")
_lock("LSE25_T", LSE25_T, "−0.0016"); _lock("LSE50_T", LSE50_T, "−0.016"); _lock("LSE100_T", LSE100_T, "−0.078"); _lock("LSE300_T", LSE300_T, "−0.218")
ORIG_LSE_T = tx(ORIG_LSE, "%.2f"); _lock("ORIG_LSE_T", ORIG_LSE_T, "−0.33")
REMOVE_LOSS_T, REMOVE_PN_T = tx(REMOVE_LOSS, "%.1e"), tx(REMOVE_PN, "%.1e", plus=True)
FORCE_LOSS_T, FORCE_PN_T = tx(FORCE_LOSS), tx(FORCE_PN, plus=True)
_lock("REMOVE_LOSS_T", REMOVE_LOSS_T, "−3.4e-5"); _lock("REMOVE_PN_T", REMOVE_PN_T, "＋6.1e-3"); _lock("FORCE_LOSS_T", FORCE_LOSS_T, "−1.26e-4"); _lock("FORCE_PN_T", FORCE_PN_T, "＋2.05e-2")
REMOVE_LOSS_H1_T, REMOVE_LOSS_H2_T = tx(REMOVE_LOSS_H1, "%.1e", plus=True), tx(REMOVE_LOSS_H2, "%.1e")
_lock("REMOVE_LOSS_H1_T", REMOVE_LOSS_H1_T, "＋4.9e-5"); _lock("REMOVE_LOSS_H2_T", REMOVE_LOSS_H2_T, "−9.6e-5")
HEAD_BF16_S1_T, HEAD_TPU_S1_T, CE_BWD_S1_T = tx(HEAD_BF16_S1), tx(HEAD_TPU_S1, "%.1e"), tx(CE_BWD_S1, "%.1e", plus=True)
HEAD_TF32_S1_T, HEAD_EXACT_S1_T = tx(HEAD_TF32_S1), tx(HEAD_EXACT_S1, "%.1e")
HEAD_BF16_S100_T, HEAD_TPU_S100_T = tx(HEAD_BF16_S100), tx(HEAD_TPU_S100, "%.1e")
_lock("HEAD_BF16_S1_T", HEAD_BF16_S1_T, "−6.51e-4"); _lock("HEAD_TPU_S1_T", HEAD_TPU_S1_T, "−2.4e-5"); _lock("CE_BWD_S1_T", CE_BWD_S1_T, "＋2.7e-6")
_lock("HEAD_TF32_S1_T", HEAD_TF32_S1_T, "−8.08e-4"); _lock("HEAD_EXACT_S1_T", HEAD_EXACT_S1_T, "−1.7e-6")
_lock("HEAD_BF16_S100_T", HEAD_BF16_S100_T, "−6.10e-4"); _lock("HEAD_TPU_S100_T", HEAD_TPU_S100_T, "−7.4e-5")
# 动画 TruncVsRound 的模拟终点（合成正数输入、按实测张量核模型模拟，不是实测曲线）。本课模拟（合成输入，不是实测）
TRUNC_SIM_END_T, RNE_SIM_END_T = "−3.5e-4", "−2.3e-6"

# ── 11-06 更新：「精确轨迹」自己也不精确（GPU cuBLAS FP32 在 12 万长求和上有偏），新的账 ──────────────
# 出处（实测）：排查记录 11-06 上午的版本（含 ⑤ 的判定与 dlogits 那组补跑）。第 1000 步参数范数，对 ② 生产 golden 5 份均值；生产近似配置。
PN_FIX1 = 1.857e-2          # ④ 精确轨迹（cuBLAS FP32 一次累加整个词表）
PN_PRODCHUNK = 2.071e-2     # ⑤ GPU 生产 ＋ 输出头 dgrad 分段 FP32 累加
PN_TRUE = 2.744e-2          # ⑥ 真精确：④ ＋ 分段累加
PN_TPU_PROD = 2.092e-2      # TPU 生产（v5p、v7x 当前配置同值）
B_GPUEXACT, B_DLOGITS, B_REST = 8.697e-3, -6.836e-3, 0.488e-3     # TPU 生产 − ④ 的三份：④ 自己少漂 / TPU dlogits 舍 BF16 / 其余 BF16 存储与逐元素
TPU_VS_FIX1 = 2.349e-3
assert abs(B_GPUEXACT + B_DLOGITS + B_REST - TPU_VS_FIX1) < 1e-9
PN_TPU_EXACT = PN_FIX1 + B_GPUEXACT                               # TPU 精确轨迹（＝真精确，GPU 分段累加只低 1.5e-4，一个记录刻度内）
TPU_VS_TRUE = PN_TPU_PROD - PN_TPU_EXACT                          # 校对更正：−6.35e-3
assert abs(TPU_VS_TRUE + 6.35e-3) < 1e-5, TPU_VS_TRUE
LADDER_DGRAD = PN_PRODCHUNK                                       # 生产 → ⑤：输出头 dgrad 长累加（BF16 张量核与 cuBLAS FP32 两段合计）
LADDER_SHARED = PN_TRUE - PN_PRODCHUNK                            # ⑤ → ⑥：两侧共有的 dlogits 舍 BF16
assert abs(LADDER_SHARED - 6.73e-3) < 1e-5
SHARED_DLOGITS_GPU = -6.738e-3                                    # GPU 真精确只多舍一次 dlogits，少漂这么多
CUBLAS_S1, CUBLAS_299 = -1.8e-6, -2.6e-4                          # cuBLAS FP32 输出头 dgrad 沿 w̄ 的相对偏差：第 1 步、第 299 步的状态
CHUNK_VS_TPUEXACT = 7.9e-8                                        # GPU 精确＋分段累加 对 TPU 精确，参数范数带符号均值（不重算配置）
DRIFT_RATIO_TRUE, DRIFT_RATIO_TPU = 1.45, 1.10                    # 不重算配置第 1000 步漂移对 ④′ 之比：真精确 / TPU 生产
NOISE_PN = 1.24e-7                                                # ③ 那组 10 对的参数范数带符号均值上限
PRODCHUNK_PN_V5P, PRODCHUNK_LOSS_TAIL_V5P, NOISE_LOSS_TAIL = 5.2e-8, 3.3e-5, 5.62e-5
TRUEEXACT_PN_V5P, TRUEEXACT_PN_V7X = -7.70e-7, -8.77e-7
HEADFP32_COST = 0.097
# 落进正文的写法
PN_FIX1_T, PN_PRODCHUNK_T, PN_TRUE_T, PN_TPU_PROD_T = (tx(v, "%.2e", plus=True) for v in (PN_FIX1, PN_PRODCHUNK, PN_TRUE, PN_TPU_PROD))
_lock("PN_FIX1_T", PN_FIX1_T, "＋1.86e-2"); _lock("PN_PRODCHUNK_T", PN_PRODCHUNK_T, "＋2.07e-2"); _lock("PN_TRUE_T", PN_TRUE_T, "＋2.74e-2"); _lock("PN_TPU_PROD_T", PN_TPU_PROD_T, "＋2.09e-2")
B_GPUEXACT_T, B_DLOGITS_T, B_REST_T, TPU_VS_FIX1_T = tx(B_GPUEXACT, plus=True), tx(B_DLOGITS), tx(B_REST, "%.1e", plus=True), tx(TPU_VS_FIX1, plus=True)
_lock("B_GPUEXACT_T", B_GPUEXACT_T, "＋8.70e-3"); _lock("B_DLOGITS_T", B_DLOGITS_T, "−6.84e-3"); _lock("B_REST_T", B_REST_T, "＋4.9e-4"); _lock("TPU_VS_FIX1_T", TPU_VS_FIX1_T, "＋2.35e-3")
TPU_VS_TRUE_T = tx(abs(TPU_VS_TRUE), "%.1e"); _lock("TPU_VS_TRUE_T", TPU_VS_TRUE_T, "6.3e-3")
LADDER_DGRAD_T, LADDER_SHARED_T = tx(LADDER_DGRAD, plus=True), tx(LADDER_SHARED, "%.1e", plus=True)
_lock("LADDER_DGRAD_T", LADDER_DGRAD_T, "＋2.07e-2"); _lock("LADDER_SHARED_T", LADDER_SHARED_T, "＋6.7e-3")
SHARED_DLOGITS_GPU_T = tx(SHARED_DLOGITS_GPU, "%.1e"); _lock("SHARED_DLOGITS_GPU_T", SHARED_DLOGITS_GPU_T, "−6.7e-3")
CUBLAS_S1_T, CUBLAS_299_T = tx(CUBLAS_S1, "%.1e"), tx(CUBLAS_299, "%.1e"); _lock("CUBLAS_S1_T", CUBLAS_S1_T, "−1.8e-6"); _lock("CUBLAS_299_T", CUBLAS_299_T, "−2.6e-4")
PRODCHUNK_PN_R = PRODCHUNK_PN_V5P / NOISE_PN; PRODCHUNK_LOSS_R = PRODCHUNK_LOSS_TAIL_V5P / NOISE_LOSS_TAIL
PRODCHUNK_PN_R_T, PRODCHUNK_LOSS_R_T = "%.2f" % PRODCHUNK_PN_R, "%.2f" % PRODCHUNK_LOSS_R; _lock("PRODCHUNK_PN_R_T", PRODCHUNK_PN_R_T, "0.42"); _lock("PRODCHUNK_LOSS_R_T", PRODCHUNK_LOSS_R_T, "0.59")
TRUEEXACT_R_T = "%.1f 到 %.1f" % (abs(TRUEEXACT_PN_V5P) / NOISE_PN, abs(TRUEEXACT_PN_V7X) / NOISE_PN); _lock("TRUEEXACT_R_T", TRUEEXACT_R_T, "6.2 到 7.1")
HEADFP32_COST_T = "%.1f%%" % (HEADFP32_COST * 100); _lock("HEADFP32_COST_T", HEADFP32_COST_T, "9.7%")
DRIFT_RATIO_TRUE_T, DRIFT_RATIO_TPU_T = "%.2f" % DRIFT_RATIO_TRUE, "%.2f" % DRIFT_RATIO_TPU
CHUNK_VS_TPUEXACT_T = tx(CHUNK_VS_TPUEXACT, "%.1e"); _lock("CHUNK_VS_TPUEXACT_T", CHUNK_VS_TPUEXACT_T, "7.9e-8")
TPU_SHARE_OF_TRUE_T = "%.0f%%" % (PN_TPU_PROD / PN_TRUE * 100); _lock("TPU_SHARE_OF_TRUE_T", TPU_SHARE_OF_TRUE_T, "76%")

# ── 11-06 定案：第一关基线 ＝ ⑤（GPU 生产＋输出头 dgrad 分段累加），尺子 ＝ ⑤ 自己 10 份两两 45 对 ─────────────
# 出处（实测）：对齐报告 §3–§8（11-06 定案版）。每格是三个统计里对 N 最差的 r。
FINAL_RUNS, FINAL_PAIRS = 10, 45
assert FINAL_PAIRS == FINAL_RUNS * (FINAL_RUNS - 1) // 2
FINAL = [  # (平台·配置, loss 全程, loss 尾段, grad_norm, param_norm, 路由偏置曲线)
    ("v5p 当前对齐配置", 0.80, 0.80, 0.85, 0.50, 0.81),
    ("v7x 当前对齐配置", 0.82, 0.83, 0.85, 0.53, 0.81),
    ("v5p EP＝8", 0.76, 0.78, 0.74, 0.48, None),
    ("v7x EP＝8", 0.77, 0.81, 0.68, 0.35, None),
]
FINAL_WORST = max(max(v for v in row[1:] if v is not None) for row in FINAL)
assert FINAL_WORST == 0.85
FINAL_WORST_OLDCAL = 0.84                     # 不做两处口径改正，45 对下最差
PEND10_LO, PEND10_HI = 1.02, 1.24             # 只有 5 份 10 对时的待定区
ENSEMBLE_RUNS, ENSEMBLE_V5P, ENSEMBLE_V7X = 9, 5, 4
ENSEMBLE_OK = 8                                   # ≤ 0.85 的份数；剩 1 份（v7x，splash 块 1024）梯度范数待定
ENSEMBLE_PEND_MARE, ENSEMBLE_PEND_MAX = 1.02, 1.15
V7X_GN_MARE_TPU, V7X_GN_MARE_GPU = 3.9e-2, 3.3e-2  # v7x 扰动组两两 / GPU 重跑两两的梯度范数 MARE
assert 1.15 < V7X_GN_MARE_TPU / V7X_GN_MARE_GPU < 1.25   # 宽约两成
V7X_L2_R = (1.06, 1.01, 1.01, 1.02)                # v7x 当前配置四层前向 R 比（开关前 1.53–1.57）
V7X_MOE_R, V7X_MOE_R_DROP, V5P_MOE_R_DROP = 1.27, 0.95, 0.94   # 第 2 层 MoE 块：1/4096 个 token 近平局选了另一个专家；去掉后；v5p 同层去掉 2 个后
PRESWITCH_VS_FINAL_V5P, PRESWITCH_VS_FINAL_V7X = 0.79, 0.78    # 开关打开前的配置对 ⑤ 的最差 r（L4 不需要这个开关）
ENSEMBLE_SPREAD_TPU, ENSEMBLE_SPREAD_GPU = 5.4e-5, 9.5e-5   # loss 带符号均值两两散布：TPU 扰动组 / GPU 重跑
assert ENSEMBLE_V5P + ENSEMBLE_V7X == ENSEMBLE_RUNS
PN_MEAS_TPU, PN_MEAS_GPU, PN_RERUN_STEP = 2.6e-7, 8e-8, 4.86e-7   # param_norm 自测误差（单步最大差）：TPU 整网一次归约 / GPU；GPU ⑤ 重跑两两单步最大差的上限（45 对；旧 10 对时 2.7e-7）
GN_N_AB, GN_N_BA = 1.26e-3, 2.49e-3           # grad_norm 带符号均值的 N，随配对方向变
GN_JITTER = 0.03
VS_RC_PN, VS_RC_LOSS_V5P, VS_RC_LOSS_V7X = 2.09e-2, -1.37e-4, -1.08e-4   # v7x 一项是推断（对 ⑤ 的差加 ⑤ 对 ② 的差）
# 步时（第 50–1000 步中位数）：② 5 份约 0.236 s；⑤ 10 份 0.380–0.382 s、同机对照 0.381 s（11-07 按 1000 步复核记录统一成 0.381，原写 ⑤ 第一份的 0.382 → 62%）
CHUNK_SEG, CHUNK_STEP_OLD, CHUNK_STEP_NEW = 1024, 0.236, 0.381
CHUNK_COST = CHUNK_STEP_NEW / CHUNK_STEP_OLD - 1
assert abs(CHUNK_COST - 0.61) < 0.01
DFA_COST_V7X, DFA_COST_V5P, EP8_SPEEDUP_V7X = 0.019, 0.0066, 0.28
FLIP_SLOT_LO, FLIP_SLOT_HI = 0.44, 0.56       # 第 1 步每层翻转的槽位百分比
# 落进正文
FINAL_WORST_T, FINAL_WORST_OLDCAL_T = "%.2f" % FINAL_WORST, "%.2f" % FINAL_WORST_OLDCAL
_lock("FINAL_WORST_T", FINAL_WORST_T, "0.85"); _lock("FINAL_WORST_OLDCAL_T", FINAL_WORST_OLDCAL_T, "0.84")
FINAL_PN_V5P_T, FINAL_PN_V7X_T = "%.2f" % FINAL[0][4], "%.2f" % FINAL[1][4]
_lock("FINAL_PN_V5P_T", FINAL_PN_V5P_T, "0.50"); _lock("FINAL_PN_V7X_T", FINAL_PN_V7X_T, "0.53")
PEND10_T = "%.2f–%.2f" % (PEND10_LO, PEND10_HI); _lock("PEND10_T", PEND10_T, "1.02–1.24")
PN_MEAS_TPU_T, PN_MEAS_GPU_T, PN_RERUN_STEP_T = tx(PN_MEAS_TPU, "%.1e"), tx(PN_MEAS_GPU, "%.0e"), tx(PN_RERUN_STEP, "%.2e")
_lock("PN_MEAS_TPU_T", PN_MEAS_TPU_T, "2.6e-7"); _lock("PN_MEAS_GPU_T", PN_MEAS_GPU_T, "8e-8"); _lock("PN_RERUN_STEP_T", PN_RERUN_STEP_T, "4.86e-7")
GN_N_AB_T, GN_N_BA_T = tx(GN_N_AB), tx(GN_N_BA); _lock("GN_N_AB_T", GN_N_AB_T, "1.26e-3"); _lock("GN_N_BA_T", GN_N_BA_T, "2.49e-3")
ENSEMBLE_SPREAD_TPU_T, ENSEMBLE_SPREAD_GPU_T = tx(ENSEMBLE_SPREAD_TPU, "%.1e"), tx(ENSEMBLE_SPREAD_GPU, "%.1e")
VS_RC_LOSS_V5P_T, VS_RC_LOSS_V7X_T = tx(VS_RC_LOSS_V5P), tx(VS_RC_LOSS_V7X)
_lock("VS_RC_LOSS_V5P_T", VS_RC_LOSS_V5P_T, "−1.37e-4"); _lock("VS_RC_LOSS_V7X_T", VS_RC_LOSS_V7X_T, "−1.08e-4")
CHUNK_COST_T = "%.0f%%" % (CHUNK_COST * 100); _lock("CHUNK_COST_T", CHUNK_COST_T, "61%")
DFA_COST_V7X_T, DFA_COST_V5P_T, EP8_SPEEDUP_V7X_T = "%.1f%%" % (DFA_COST_V7X * 100), "%.2f%%" % (DFA_COST_V5P * 100), "%.0f%%" % (EP8_SPEEDUP_V7X * 100)
_lock("DFA_COST_V7X_T", DFA_COST_V7X_T, "1.9%"); _lock("DFA_COST_V5P_T", DFA_COST_V5P_T, "0.66%"); _lock("EP8_SPEEDUP_V7X_T", EP8_SPEEDUP_V7X_T, "28%")
FLIP_SLOT_T = "%.2f%%–%.2f%%" % (FLIP_SLOT_LO, FLIP_SLOT_HI)
# 路由翻转的有效证据（L1/L2 隔离回放）：路由输入换成 TPU 自己算的，每层翻的 token 比例
FLIP_L2_LO, FLIP_L2_HI = 3.5, 5.4
FLIP_L2_T = "%.1f%%–%.1f%%" % (FLIP_L2_LO, FLIP_L2_HI); _lock("FLIP_L2_T", FLIP_L2_T, "3.5%–5.4%")
# 「两边全程 FP32 也翻」那条证据已撤回：TPU 的 FP32 参照里路由专家其实按 BF16 算（FP32_MARGIN 只留作台账旧数）
FINAL_PN_SM, FINAL_PN_SM_N = 1.593e-8, 2.269e-7      # v5p 当前配置对 ⑤：参数范数带符号均值、45 对上限（fig9-golden 柱子用带符号均值，跟 ②–④ 同口径）
FINAL_PN_SM_R = FINAL_PN_SM / FINAL_PN_SM_N
FINAL_PN_SM_R_T = "%.2f" % FINAL_PN_SM_R; _lock("FINAL_PN_SM_R_T", FINAL_PN_SM_R_T, "0.07")
B_V5P = (7.965e-3, -6.104e-3, 0.488e-3)              # v5p 那组补跑的三笔（v7x 是 B_GPUEXACT / B_DLOGITS / B_REST）
ENSEMBLE_PEND_T = "MARE %.2f、最大单步 %.2f" % (ENSEMBLE_PEND_MARE, ENSEMBLE_PEND_MAX); _lock("ENSEMBLE_PEND_T", ENSEMBLE_PEND_T, "MARE 1.02、最大单步 1.15")
V7X_L2_R_T = " ／ ".join("%.2f" % v for v in V7X_L2_R); _lock("V7X_L2_R_T", V7X_L2_R_T, "1.06 ／ 1.01 ／ 1.01 ／ 1.02")

# ── 11-06 下午：第 1 步张量换尺子（两侧执行噪声并集）＋ 台账 #20 定责 ─────────────────────────────
# 出处（实测）：对齐报告 §4 与注 17、注 20（11-06 下午版）；新开关步时取补跑记录。
STEP1_TPU_ONLY_EXEC, STEP1_TPU_VS_GPU = 0.080, 0.105    # 第 3 层 0 号专家梯度相对差：TPU 只改片上内存上限 / TPU 对 GPU
UNION_GPU_PAIRS, UNION_TPU_RUNS = 3, 7
UNION_TPU_PAIRS = UNION_TPU_RUNS * (UNION_TPU_RUNS - 1) // 2
UNION_PAIRS = UNION_GPU_PAIRS + UNION_TPU_PAIRS
assert (UNION_TPU_PAIRS, UNION_PAIRS) == (21, 24)
ISO_BWD_R_V5P, ISO_BWD_R_V7X = (0.98, 0.74, 0.83, 1.01), (0.98, 0.96, 0.88, 0.95)   # 逐层隔离回放反向对 FP64 的 R
assert max(ISO_BWD_R_V5P + ISO_BWD_R_V7X) <= 1.2
STEP_R_V5P, STEP_R_V7X = 1.20, 1.21           # 整步梯度对 FP64，整网按能量 R
STEP_R_PROJ = (0.45, 0.96)                     # 投影、共享专家、稠密 MLP：TPU 更近精确
STEP_R_EXPERT = (0.76, 1.32)                   # 专家与路由权重：两边差不多远
STEP_R_NORMMAX_V5P, STEP_R_NORMMAX_V7X = 2.20, 2.70   # 一维 norm、输出层、embedding：TPU 更远（最高）
ARM_R = [("RMSNorm 的 scale", 1.35, 0.58), ("输出层", 1.60, 0.91), ("embedding", 1.21, 0.87)]
ARM_ALL_R = (1.20, 0.67)
GRAD_FP32_COST_V5P = 0.0079                    # 三个开关一起开，v5p 步时（1.6111 → 1.6239 s）
assert abs(1.6239 / 1.6111 - 1 - GRAD_FP32_COST_V5P) < 0.0005
STEP1_L23_V5P, STEP1_L23_V7X = (1.05, 1.75), (1.05, 1.86)          # 第 1 步整步检查的待定区间
STEP1_FAIL_V5P, STEP1_FAIL_V7X = (2.1, 4.5, 2.3), (3.1,)            # 不过的层参数梯度：v5p 第 0–2 层、v7x 第 1 层
MASTER_W_V5P, MASTER_W_V7X = 1.11, 0.99                              # 换尺子后，更新后主权重（v5p 待定、v7x 过）
MASTER_W_FP64_R = 0.92
BIAS_TOKENS_FP64, BIAS_MEAN = 2730, 65536 * 8 / 192
assert abs(BIAS_MEAN - 2730.67) < 0.01
# 落进正文
STEP1_EXEC_T = "%.3f" % STEP1_TPU_ONLY_EXEC; _lock("STEP1_EXEC_T", STEP1_EXEC_T, "0.080")
STEP1_TVG_T = "%.3f" % STEP1_TPU_VS_GPU; _lock("STEP1_TVG_T", STEP1_TVG_T, "0.105")
ISO_BWD_V5P_T = " ／ ".join("%.2f" % v for v in ISO_BWD_R_V5P); ISO_BWD_V7X_T = " ／ ".join("%.2f" % v for v in ISO_BWD_R_V7X)
STEP_R_T = "%.2f（v5p）／ %.2f（v7x）" % (STEP_R_V5P, STEP_R_V7X); _lock("STEP_R_T", STEP_R_T, "1.20（v5p）／ 1.21（v7x）")
STEP_R_PROJ_T = "%.2f–%.2f" % STEP_R_PROJ; STEP_R_EXPERT_T = "%.2f–%.2f" % STEP_R_EXPERT
ARM_R_T = "；".join("%s %.2f → %.2f" % a for a in ARM_R); _lock("ARM_R_T", ARM_R_T, "RMSNorm 的 scale 1.35 → 0.58；输出层 1.60 → 0.91；embedding 1.21 → 0.87")
GRAD_FP32_COST_T = "%.2f%%" % (GRAD_FP32_COST_V5P * 100); _lock("GRAD_FP32_COST_T", GRAD_FP32_COST_T, "0.79%")
BIAS_MEAN_T = "%.2f" % BIAS_MEAN

# ── 11-06 晚：三个梯度 FP32 开关的 1000 步复核（出处：对齐报告「TPU 开关的补充说明」，实测）──────────
GF32_V5P = (0.88, 0.91, 0.79, 0.60, 0.85)      # 开关打开后对 ⑤：loss 全程、尾段、梯度范数、参数范数、路由偏置曲线（v5p，全过）
GF32_V7X = (0.78, 0.73, None, 0.53, 0.68)      # v7x：梯度范数待定，见下
GF32_V7X_GN_MAX, GF32_V7X_GN_SM, GF32_V7X_GN_MARE = 1.05, 0.10, 0.97
assert abs(0.482 / 0.459 - GF32_V7X_GN_MAX) < 0.005
GF32_COST_V7X = 0.016
assert abs(0.9952 / 0.9796 - 1 - GF32_COST_V7X) < 0.001
ARM_ALL_R_V7X = (1.21, 0.72)
GF32_V5P_T = "loss 全程 %.2f、尾段 %.2f、梯度范数 %.2f、参数范数 %.2f、路由偏置曲线 %.2f" % GF32_V5P
_lock("GF32_V5P_T", GF32_V5P_T, "loss 全程 0.88、尾段 0.91、梯度范数 0.79、参数范数 0.60、路由偏置曲线 0.85")
GF32_V7X_T = "loss 全程 %.2f、尾段 %.2f、参数范数 %.2f、路由偏置曲线 %.2f" % (GF32_V7X[0], GF32_V7X[1], GF32_V7X[3], GF32_V7X[4])
GF32_V7X_GN_T = "最大单步 %.2f 倍（带符号均值 %.2f 倍、MARE %.2f 倍）" % (GF32_V7X_GN_MAX, GF32_V7X_GN_SM, GF32_V7X_GN_MARE)
_lock("GF32_V7X_GN_T", GF32_V7X_GN_T, "最大单步 1.05 倍（带符号均值 0.10 倍、MARE 0.97 倍）")
GF32_COST_V7X_T = "%.1f%%" % (GF32_COST_V7X * 100); _lock("GF32_COST_V7X_T", GF32_COST_V7X_T, "1.6%")
# 扰动臂第三关的第 1 步梯度一半改成比 FP64（出处：回归说明 11-06 补做，实测）：当前配置 6 个臂，按能量 R
ARMS_FP64_N, ARMS_FP64_R_V5P, ARMS_FP64_R_V7X = 6, (1.001, 1.008), (1.003, 1.039)   # 11-07 更正：两平台分开写（原合写 1.001–1.039）
ARMS_FP64_R_T = "v5p %.3f–%.3f、v7x %.3f–%.3f" % (ARMS_FP64_R_V5P + ARMS_FP64_R_V7X); _lock("ARMS_FP64_R_T", ARMS_FP64_R_T, "v5p 1.001–1.008、v7x 1.003–1.039")
STEP_R_EXPERT_V7X_MAX = 1.44                   # v7x 专家与路由权重梯度 R 最高（第 2 层 1 号专家 fc2）

# ── 11-07 · 课上追问补答（Round 2）────────────────────────────────────────────────
# rel_L2 与 cos、范数比：rel² ＝ (r−1)² ＋ 2r(1−cos)，r ＝ |a| ÷ |b|（本课推导：|a−b|² 展开再除以 |b|²）；等长纯旋转 θ → rel ＝ 2 sin(θ/2)
assert abs(math.sqrt(2 * (1 - math.cos(math.radians(2.56)))) - 2 * math.sin(math.radians(1.28))) < 1e-12
assert abs(2 * math.sin(math.acos(0.999) / 2) - COS_0999_RELL2) < 1e-4          # 跟 §4.1 那个 4.5% 是同一个数
# R 比的分母：路由打分 TPU、GPU 各离 FP64 多远 → R ＝ TPU ÷ GPU（本课推导）
R_ROUTER = ROUTER_TPU_FP64 / ROUTER_GPU_FP64
R_ROUTER_T = "%.3f" % R_ROUTER; _lock("R_ROUTER_T", R_ROUTER_T, "0.013")
# 张量个数（出处：对齐实践的术语表「张量个数」条）：开源命名 3 个 MoE 层 × 192 专家 × 3 个矩阵 ＋ 53；GPU 框架布局每专家 2 个矩阵 ＋ 41
MOE_LAYERS = 3
HF_EXPERT_TENSORS, HF_OTHER_TENSORS = MOE_LAYERS * EXPERTS * 3, 53
MG_EXPERT_TENSORS, MG_OTHER_TENSORS = MOE_LAYERS * EXPERTS * 2, 41
assert HF_EXPERT_TENSORS + HF_OTHER_TENSORS == N_TENSORS and MG_EXPERT_TENSORS + MG_OTHER_TENSORS == SWAP_STATE_TENSORS
HF_EXPERT_TENSORS_T, MG_EXPERT_TENSORS_T = format(HF_EXPERT_TENSORS, ","), format(MG_EXPERT_TENSORS, ",")
_lock("HF_EXPERT_TENSORS_T", HF_EXPERT_TENSORS_T, "1,728"); _lock("MG_EXPERT_TENSORS_T", MG_EXPERT_TENSORS_T, "1,152")
# 输出头 dgrad 的四种写法消融（出处：对齐实践的结果文件「输出头 dgrad 消融」，B200 一张卡，训练实际张量，对 FP64，沿 w̄）
# 三个值依次是第 1 ／ 100 ／ 300 步的状态
ABL_BF16_ONESHOT = (-6.51e-4, -6.10e-4, -4.60e-4)     # 生产：BF16 张量核一次乘完
ABL_FP32_ONESHOT = (-1.70e-6, -2.56e-4, -2.58e-4)     # 升 FP32 一次乘完（真关 TF32）
ABL_BF16_CHUNK = (-6.88e-7, 8.27e-7, 3.11e-7)         # BF16 张量核每 1024 一段，段结果留 FP32、段间 FP32 相加
ABL_FP32_CHUNK = (9.85e-9, 3.49e-7, 1.08e-7)          # 升 FP32 每 1024 一段（最终 golden ⑤ 的写法）
ABL_CHUNK4096_MAX, ABL_CHUNK16384 = -5.648e-6, (-2.33e-5, -4.32e-5)   # 段长 4096 最多剩 / 16384 剩
ABL_MS_PROD, ABL_MS_BF16_CHUNK, ABL_MS_FP32_CHUNK = 2.45, 7.2, 71.7  # 单次乘法耗时（ms，中位数；分段是 Python 逐段调库，没融合）
assert ABL_BF16_ONESHOT[0] == HEAD_BF16_S1 and ABL_BF16_ONESHOT[1] == HEAD_BF16_S100 and ABL_FP32_ONESHOT[0] == HEAD_EXACT_S1
assert abs(ABL_FP32_ONESHOT[2] - CUBLAS_299) < 0.03e-4
assert all(1e-7 <= abs(x) <= 1e-6 for x in ABL_BF16_CHUNK)                       # 「约 1e-7 到 1e-6」
ABL_BF16_CHUNK_HI_T = tx(max(abs(x) for x in ABL_BF16_CHUNK), "%.1e"); _lock("ABL_BF16_CHUNK_HI_T", ABL_BF16_CHUNK_HI_T, "8.3e-7")
ABL_FP32_ONESHOT_300_T = tx(ABL_FP32_ONESHOT[2], "%.1e"); _lock("ABL_FP32_ONESHOT_300_T", ABL_FP32_ONESHOT_300_T, "−2.6e-4")
ABL_CHUNK4096_T = tx(ABL_CHUNK4096_MAX, "%.1e"); _lock("ABL_CHUNK4096_T", ABL_CHUNK4096_T, "−5.6e-6")
ABL_MS_T = "%.1f ms 对 %.1f ms" % (ABL_MS_BF16_CHUNK, ABL_MS_FP32_CHUNK); _lock("ABL_MS_T", ABL_MS_T, "7.2 ms 对 71.7 ms")
ABL_MS_PROD_T = "%.2f ms" % ABL_MS_PROD
# 参考档逐层线的来历（出处：对齐实践的方法文档「两档门槛 · 参考值」）：同类项目注意力层输出逐层前向实测 rel_L2，线取 1%
REF_LAYER_MEAS_T = "3e-3 到 5.4e-3"
REF_LAYER0_FWD, REF_LAYER0_BWD = 0.0040, 0.0021     # 本例只在第 0 层稠密层用这条线：前向 ／ 反向（两平台相同）
assert REF_LAYER0_FWD < REF_RELL2_GATE and REF_LAYER0_BWD < REF_RELL2_GATE
# 张量核在真实矩阵乘里也偏一点：合成正数、K ＝ 4096 → 7.0e-5（TC_BIAS_K 第二项）
_j3 = lambda xs, specs: " ／ ".join(tx(x, sp, plus=True) for x, sp in zip(xs, specs))
ABL_BF16_ONESHOT_T = _j3(ABL_BF16_ONESHOT, ("%.2e",) * 3); _lock("ABL_BF16_ONESHOT_T", ABL_BF16_ONESHOT_T, "−6.51e-4 ／ −6.10e-4 ／ −4.60e-4")
ABL_FP32_ONESHOT_T = _j3(ABL_FP32_ONESHOT, ("%.1e", "%.2e", "%.2e")); _lock("ABL_FP32_ONESHOT_T", ABL_FP32_ONESHOT_T, "−1.7e-6 ／ −2.56e-4 ／ −2.58e-4")
ABL_BF16_CHUNK_T = _j3(ABL_BF16_CHUNK, ("%.1e",) * 3); _lock("ABL_BF16_CHUNK_T", ABL_BF16_CHUNK_T, "−6.9e-7 ／ ＋8.3e-7 ／ ＋3.1e-7")
ABL_FP32_CHUNK_T = _j3(ABL_FP32_CHUNK, ("%.0e", "%.1e", "%.1e")); _lock("ABL_FP32_CHUNK_T", ABL_FP32_CHUNK_T, "＋1e-8 ／ ＋3.5e-7 ／ ＋1.1e-7")
REF_LAYER0_FWD_T, REF_LAYER0_BWD_T = "%.4f" % REF_LAYER0_FWD, "%.4f" % REF_LAYER0_BWD
_lock("REF_LAYER0_FWD_T", REF_LAYER0_FWD_T, "0.0040"); _lock("REF_LAYER0_BWD_T", REF_LAYER0_BWD_T, "0.0021")
ABL_MS_FP32_ONESHOT = 65.6        # 升 FP32 一次乘完的单次耗时（同一消融）：分段 FP32 的耗时大头在升 FP32（本课推断）
assert ABL_MS_FP32_ONESHOT - ABL_MS_PROD > 0.8 * (ABL_MS_FP32_CHUNK - ABL_MS_PROD)

# ── 11-07 · BF16 张量核分段跑满 1000 步（出处：对齐实践的 1000 步复核结果文件，实测）──────────────
# 设置 ＝ ⑤ 原样，只把输出头 dgrad 换成「BF16 张量核每 1024 一段、段结果 FP32、段间 FP32 相加」；2 份；尺子 ＝ ⑤ 10 份两两 45 对
CB_RUNS = 2
CB_WORST, CB_CHK_WORST = 0.91, 0.91          # 对 ⑤ 基线：BF16 分段那份最差 r（路由偏置曲线）／ 同机原样重跑 ⑤ 的对照
CB_VS_RCC_A = (0.77, 0.77, 0.84, 0.67, 0.91)  # loss 全程、尾段、梯度范数、参数范数、路由偏置曲线（第一份）
assert max(CB_VS_RCC_A) == CB_WORST
CB_STEP = 0.250                               # 步时 s
CB_COST = CB_STEP / CHUNK_STEP_OLD - 1
CB_COST_T = "%.0f%%" % (CB_COST * 100); _lock("CB_COST_T", CB_COST_T, "6%")
CB_STEP_T = "%.3f s" % CB_STEP
CB_TPU_V7X_GN, CB_TPU_V7X_GN_SINGLE = 1.09, (0.70, 1.11)   # TPU v7x 对 BF16 分段第一份：梯度范数 MARE r 待定；同一 TPU 运行改对 ⑤ 的单份逐一当基线时的范围
CB_TPU_V7X_GN_SINGLE_T = "%.2f–%.2f" % CB_TPU_V7X_GN_SINGLE; _lock("CB_TPU_V7X_GN_SINGLE_T", CB_TPU_V7X_GN_SINGLE_T, "0.70–1.11")

# ══════════════════════════════════════════════════════════════════════════════════════
# ── 11-07 · 第二次迁移（一个更新的、带稀疏注意力的 MoE 模型，小尺寸随机初始化，300 ／ 1000 步）────────
# 出处：同一位作者的第二次对齐实践的案例记录、改动台账与结果文件（实测）；公开课只写机制和数，不写模型与项目名。
# 课件 §2.2、§2.3、§4.2、§4.3、§5.6、§5.7、§6.2、§10.3 用。
# ══════════════════════════════════════════════════════════════════════════════════════
# 第 1 步逐张量梯度：TPU 对 GPU 的 rel_L2 ÷ GPU 三份两两的最大 rel_L2（195 个张量）
B2_TENSORS = 195
B2_RATIO_MED = (2.67, 1.65, 1.72)       # 开关全关 ／ 打开对上 GPU 舍入位置的三个开关 ／ 再加第四个（门控激活逐算子舍）
B2_OVER2 = (139, 52, 62)                # 比值超过 2 的张量个数，同上三种
assert B2_RATIO_MED[1] < B2_RATIO_MED[2] and B2_OVER2[1] < B2_OVER2[2]   # 第四个开关让统计略差，照样开（按 GPU 的写法，不按好不好看）
assert all(n < B2_TENSORS for n in B2_OVER2)
B2_RATIO_MED_T = "%.2f → %.2f" % B2_RATIO_MED[:2]; _lock("B2_RATIO_MED_T", B2_RATIO_MED_T, "2.67 → 1.65")
B2_OVER2_T = "%d → %d" % B2_OVER2[:2]
# 「存成 16 位的中间值」：参照一侧自己写的 eager 注意力把 q·k 先舍 BF16 再进 FP32 softmax；只改这一处的第 1 步变化 vs GPU 自己的噪声
ATTN_FIX_LOSS, ATTN_FIX_GN = 8.1e-7, -3.0e-5          # 第 1 步 loss ／ 梯度范数的相对变化
ATTN_NOISE_LOSS, ATTN_NOISE_GN = 4.5e-5, 2.2e-3       # GPU 8 份两两 28 对最大相对差（改后那一代）
assert ATTN_FIX_LOSS < ATTN_NOISE_LOSS / 10 and abs(ATTN_FIX_GN) < ATTN_NOISE_GN / 10   # 远小于噪声：靠数值大小抓不到它
ATTN_FIX_LOSS_T, ATTN_FIX_GN_T = tx(ATTN_FIX_LOSS, "%.1e", plus=True), tx(ATTN_FIX_GN, "%.1e")
ATTN_NOISE_LOSS_T, ATTN_NOISE_GN_T = tx(ATTN_NOISE_LOSS, "%.1e"), tx(ATTN_NOISE_GN, "%.1e")
_lock("ATTN_FIX_LOSS_T", ATTN_FIX_LOSS_T, "＋8.1e-7"); _lock("ATTN_NOISE_GN_T", ATTN_NOISE_GN_T, "2.2e-3")
# 「HIGHEST 也不是 FP32」：一处多项加权和，TPU 默认 HIGHEST einsum（多遍 BF16 拼近似 FP32），GPU 逐元素 FP32 乘加
MIX_TENSORS = 45
MIX_OVER = (12, 5, 18)                    # 对 FP32 参照 R ＞ 1.2 的张量：现行 ／ 改逐项 FP32 乘加 ／ 现行再关 excess precision
MIX_MED = (0.88, 0.79, 1.09)              # R 中位数，同上
assert MIX_OVER[1] < MIX_OVER[0] < MIX_OVER[2]   # 关 excess precision 反而更多：跟 §5.5 混元 3 的方向一致
MIX_OVER_T = "%d → %d" % MIX_OVER[:2]; _lock("MIX_OVER_T", MIX_OVER_T, "12 → 5")
# 病态：稀疏注意力里一项辅助蒸馏损失（KL）每行约 1.1e-4 nat，log-sum-exp 约 7.1
KD_KL, KD_LSE = 1.1e-4, 7.1
KD_ULP = 4.73e-7                         # f32 在 lse 处一格（实测各行平均；lse ∈ [4, 8) 时 2⁻²¹ ≈ 4.77e-7）
assert abs(KD_ULP - 2.0 ** -21) / 2.0 ** -21 < 0.02
KD_ULP_OF_KL = 4.31e-3                   # 一格 ÷ KL（实测；按上两数 4.73e-7 ÷ 1.1e-4 ≈ 4.3e-3，本课推导核对）
assert abs(KD_ULP / KD_KL - KD_ULP_OF_KL) / KD_ULP_OF_KL < 0.01
KD_LSE_TPU_ULP, KD_LSE_CPU_ULP = -0.953, -0.004    # TPU 图内 ／ CPU torch f32 的 lse 对 FP64，按格计的平均
KD_POS_ROWS = "8.6%"                     # TPU 偏正的行只占这么多：方向一致
KD_DIFF = -1.155e-3                      # 同一行 KD：TPU 图内对同一输入的 FP64 重算（相对）
KD_IN_R, KD_GRAD_R = "0.94–0.99", "0.78–1.03"     # 它的三组输入对 FP32 参照的 R ／ 由它出发的梯度（20 个张量）的 R
KD_RATIO = KD_LSE / KD_KL
assert 6e4 < KD_RATIO < 7e4              # 大数是小数的约 6.5 万倍
KD_ULP_OF_KL_T = tx(KD_ULP_OF_KL, "%.1e"); _lock("KD_ULP_OF_KL_T", KD_ULP_OF_KL_T, "4.3e-3")
KD_DIFF_T = tx(KD_DIFF, "%.1e"); _lock("KD_DIFF_T", KD_DIFF_T, "−1.2e-3")
KD_ULP_T = tx(KD_ULP, "%.2e")
KD_LSE_TPU_ULP_T = tx(KD_LSE_TPU_ULP, "%.2f")
# 船上称船长（本课推导，比方的数按 1 单位 lse ↔ 1000 吨换算，比例与上面一致）
SHIP_T = KD_LSE * 1000                   # 7100 吨
CAPTAIN_KG = KD_KL * 1e9 / 1000          # KL 1.1e-4 → 110 公斤
TICK_KG = KD_ULP * 1e6                   # 一格 → 0.47 公斤
assert abs(SHIP_T - 7100) < 1e-6 and abs(CAPTAIN_KG - 110) < 1e-6
assert abs(TICK_KG / CAPTAIN_KG - KD_ULP_OF_KL) / KD_ULP_OF_KL < 0.01
TICK_KG_T = "%.2f" % TICK_KG; _lock("TICK_KG_T", TICK_KG_T, "0.47")
TICK_PCT_T = "%.2f%%" % (TICK_KG / CAPTAIN_KG * 100); _lock("TICK_PCT_T", TICK_PCT_T, "0.43%")
KD_DIFF_PCT_T = "%.2f%%" % (abs(KD_DIFF) * 100); _lock("KD_DIFF_PCT_T", KD_DIFF_PCT_T, "0.12%")
# 尺子只来自参照：一度把 TPU 只改执行参数的扰动并进尺子（GPU 28 对 ＋ TPU 10 对 ＝ 38 对），撤回后同一批运行重判
UNION2_PAIRS, GPU2_PAIRS = 38, 28
assert UNION2_PAIRS == GPU2_PAIRS + 5 * 4 // 2
UNION2_GN = ((1.77, 2.08), (1.75, 2.69), (1.68, 2.75))   # 梯度范数 r：三次比较，并集 → 只用 GPU（待定全部转不过）
assert all(a <= 2 < b for a, b in UNION2_GN)
UNION2_GN_T = "、".join("%.2f → %.2f" % p for p in UNION2_GN)
# 路由偏置零均值：协议写了两边都做，参照框架根本没有这个更新；TPU 关掉后重跑
BIAS_ZM_R = (19.1, 0.31)
BIAS_ZM_R_T = "%.1f → %.2f" % BIAS_ZM_R; _lock("BIAS_ZM_R_T", BIAS_ZM_R_T, "19.1 → 0.31")
# 回放工具的 bug：注入只换位置参数，关键字传的输入漏了
INJ_R = (1.563, 0.950)                   # 稠密 MLP 单算子 R：修前 ／ 修后
INJ_ALONE_R = 0.999                      # 同一个 GPU 输入直接喂那个算子（CPU），对 FP64
assert INJ_R[0] > 1.2 >= INJ_R[1]
INJ_R_T = "%.2f → %.2f" % INJ_R; _lock("INJ_R_T", INJ_R_T, "1.56 → 0.95")
# 优化器写法（单测对参照框架的固定参照，每步更新的相对差）
OPT_ALL_OFF, OPT_ALL_ON = 0.13, 1.3e-6   # 优化器三处写法都不对上 ／ 都对上
OPT_NADAM = 0.56                         # 另一个用例：那部分参数的分支走成 NAdam，对普通 Adam
OPT_NS = 1e-6                            # 只差一处归一化的 eps 写法：舍入量级
assert OPT_ALL_ON < OPT_NS * 2   # 0.13 与 0.56 是两个不同的单测用例（矩阵参数更新对固定参照 ／ 那部分参数的分支对 Adam），不能互相比
OPT_ALL_ON_T = tx(OPT_ALL_ON, "%.1e")
# 测量口径：参数范数在 FP32 里整棵树一路相加
PN2_FP32_BIAS = "1e-7 至 3e-7"            # 对精确值偏低（第 1 步 −1.39e-7）
PN2_STEP1_EXACT = 1.6e-9                  # 两边都改比精确值（每 128 个元素一段 FP32 平方和、FP64 合成）后第 1 步的差
PN2_STEP1_EXACT_T = tx(PN2_STEP1_EXACT, "%.1e"); _lock("PN2_STEP1_EXACT_T", PN2_STEP1_EXACT_T, "1.6e-9")
# 稀疏注意力选 token 的 top-k 翻转：多，但几乎不影响输出
SPARSE_Q_FLIP_FRAC = 0.945                # 第 1 步 GPU 对 FP64：至少翻了一个槽位的 query 占全部稀疏 query 的比例（只留比例）
SPARSE_SLOT_FLIP = "0.147%"
assert SPARSE_Q_FLIP_FRAC > 0.9
SPARSE_Q_FLIP_T = "九成以上"   # 只写比例：个数会让人反推出选多少个 token
# R 在误差不到一个 BF16 格时会摆：随上游一处写法在 0.6–1.6 之间
R_SWING = "0.6–1.6"
# 门控激活 silu·up：GPU 舍两次、TPU 默认舍一次时逐位相同的元素比例（混元 3）
GLU_BITWISE = "73.7%"
# 正文直接引用的单值（«名字» 只认名字，不认下标）
B2_MED_OFF, B2_MED_4SW = B2_RATIO_MED[0], B2_RATIO_MED[2]
B2_OVER_OFF, B2_OVER_4SW = B2_OVER2[0], B2_OVER2[2]
INJ_R_BEFORE, INJ_R_AFTER = INJ_R
INJ_R_AFTER_T = "%.2f" % INJ_R_AFTER; _lock("INJ_R_AFTER_T", INJ_R_AFTER_T, "0.95")
MIX_OVER_NOEXC = MIX_OVER[2]
KD_VS_GPU = -9.2e-4                      # 第 1 步 KD：TPU 全部运行对 GPU 全部运行（相对），区别于单行对 FP64 的 KD_DIFF
KD_VS_GPU_T = tx(KD_VS_GPU, "%.1e"); _lock("KD_VS_GPU_T", KD_VS_GPU_T, "−9.2e-4")
# §5.6 折叠的收尾：现行五个开关下第 1 步逐张量（205 个张量的那份逐张量表）
B2_ALL_TENSORS, B2_FINAL_OVER, B2_FINAL_RMED, B2_FINAL_OVER_ATTR = 205, 58, 0.95, 56
assert B2_FINAL_OVER_ATTR <= B2_FINAL_OVER < B2_ALL_TENSORS
B2_GN_NOW = 1.34                         # 现行比较（参照改后、TPU 补齐几处边界）梯度范数 r，MARE，待定

# ══════════════════════════════════════════════════════════════════════════════════════
# ── 11-08 · 第二次迁移的后续（出处：同一实践的根源一览、改动台账、结果文件；实测，除标推断／本课推导）──────
# ══════════════════════════════════════════════════════════════════════════════════════
# 编译器省掉的舍入（§5.5）：改成逐项 FP32 乘加之后，GPU 一定会做的 FP32→BF16→FP32 往返在 TPU 上被省掉（推断，没看编译产物）
ELIDE_GPU_MAX = 3.77e-6                   # GPU 8 份两两 28 对，1000 步主 loss 带符号均值的最大 |.|
ELIDE_BEFORE = 1.07e-6                    # 改逐项乘加之前那一代 TPU：对 GPU 8 份均值的带符号均值
ELIDE_RUNS = (7.68e-6, 6.59e-6, 6.35e-6, 3.71e-6, 8.58e-6)   # 改之后 5 次 TPU 运行（含只改执行参数的 3 次），同号偏高
ELIDE_AFTER = 3.5e-6                      # 两处改用 reduce_precision 强制舍入以后（推算，对 GPU 均值）
ELIDE_R = (2.12, 0.76)                    # 同一量的判定 r（带符号均值，对 GPU 基线）：强制舍入前 ／ 后
assert all(x > 0 for x in ELIDE_RUNS) and min(ELIDE_RUNS) < ELIDE_GPU_MAX < max(ELIDE_RUNS)
assert ELIDE_AFTER < max(ELIDE_RUNS) and ELIDE_R[0] > 2 >= 1 >= ELIDE_R[1]
ELIDE_RANGE_T = "＋%.1e 到 ＋%.1e" % (min(ELIDE_RUNS), max(ELIDE_RUNS)); ELIDE_RANGE_T = ELIDE_RANGE_T.replace("e-06", "e-6")
_lock("ELIDE_RANGE_T", ELIDE_RANGE_T, "＋3.7e-6 到 ＋8.6e-6")
ELIDE_FIX_T = "＋%.1e（推算）" % ELIDE_AFTER; ELIDE_FIX_T = ELIDE_FIX_T.replace("e-06", "e-6")
_lock("ELIDE_FIX_T", ELIDE_FIX_T, "＋3.5e-6（推算）")
ELIDE_OVER = sum(1 for x in ELIDE_RUNS if x > ELIDE_GPU_MAX)
assert ELIDE_OVER == 4
ELIDE_R_T = "%.2f → %.2f" % ELIDE_R
ELIDE_GPU_MAX_T = tx(ELIDE_GPU_MAX, "%.1e")
# 超越函数（§5.8）：TPU v5p 默认 FP32 实现对正确舍入的最大误差（格＝ulp），对照 GPU 文档上限与 TPU 高精度写法
TRANS = {   # 函数: (TPU 默认 ulp_max, TPU 高精度写法 ulp_max, GPU 上限)
    "exp": (105, 1, 2), "log": (62, 3, 1), "sigmoid": (83, 3, 4), "silu": (86, 4, 4)}
TRANS_GPU_NOTE = "exp ≤ 2、log ≤ 1 是 CUDA 文档；sigmoid、silu 的 4 是推断，GPU 实测最大 3.4"
assert all(d > 20 * g for d, h, g in TRANS.values())               # 默认比 GPU 差一个数量级以上
assert all(h <= 4 for d, h, g in TRANS.values())
EXP_CR_FRAC, EXP_SIGNED_ULP = "6%", -2.49   # 实际蒸馏损失 softmax 的输入上：exp 只有 6% 正好舍对，平均偏 −2.5 格（系统偏低）
TRANS_T = "、".join("%s %d" % (k, v[0]) for k, v in TRANS.items()); _lock("TRANS_T", TRANS_T, "exp 105、log 62、sigmoid 83、silu 86")
TRANS_HI_T = "、".join("%s %d" % (k, v[1]) for k, v in TRANS.items()); _lock("TRANS_HI_T", TRANS_HI_T, "exp 1、log 3、sigmoid 3、silu 4")
EXP_SIGNED_ULP_T = tx(EXP_SIGNED_ULP, "%.1f")
# 打开高精度 exp／log 以后，病态那一项（§5.7）
KD_FIX_LSE = (-0.953, 0.025)              # lse 对 FP64，按格计的平均
KD_FIX_ROW = (-1.155e-3, 9.7e-5)          # 同一行 KD 对 FP64（相对）
KD_FIX_R300 = (3.67, 0.01)                # 300 步 KD loss 带符号均值的 r
assert KD_FIX_LSE[0] == KD_LSE_TPU_ULP and KD_FIX_ROW[0] == KD_DIFF
KD_FIX_LSE_T = "%.3f → ＋%.3f 格" % KD_FIX_LSE; KD_FIX_LSE_T = KD_FIX_LSE_T.replace("-", "−"); _lock("KD_FIX_LSE_T", KD_FIX_LSE_T, "−0.953 → ＋0.025 格")
KD_FIX_ROW_T = "%s → %s" % (tx(KD_FIX_ROW[0], "%.1e"), tx(KD_FIX_ROW[1], "%.1e", plus=True)); _lock("KD_FIX_ROW_T", KD_FIX_ROW_T, "−1.2e-3 → ＋9.7e-5")
KD_FIX_R300_T = "%.2f → %.2f" % KD_FIX_R300
# sigmoid 的「高精度」开关：读回是开，实际逐位没变；真换成精确写法以后整网反而偏得多（根因待查）
SIG_FLAG_SAME_OPS = 99                    # 开关前后逐算子回放，99 个算子逐位相同
SIG_LOSS_R = (0.65, 1.79)                 # 主 loss 前 300 步带符号均值 r：精确 sigmoid／silu 之前 ／ 之后
assert SIG_LOSS_R[1] > 1 > SIG_LOSS_R[0]
SIG_LOSS_R_T = "%.2f → %.2f" % SIG_LOSS_R
# 优化器里的除法、开方、偏差修正（§5.8 折叠）
OPT_ULP = {"除法": (2, 1), "开方": (3, 1), "偏差修正 1 − βᵗ": (108, 2)}
ADAM_R = (1.49e-6, 5.70e-6, 1.49e-6)      # 一个小张量第 5 步的更新对 FP64：GPU ／ TPU 默认 ／ TPU 精确写法
ADAM_R_BEFORE = ADAM_R[1] / ADAM_R[0]
assert abs(ADAM_R_BEFORE - 3.82) < 0.02
ADAM_R_T = "3.82 → 1.00"   # 结果文件里的 R（由未舍入的距离算；用上面舍入后的两个数相除是 3.83）
# r 和 R（§4.2）：同一处，r 说两边差得多，R 说 TPU 更准 —— 优化器里一次迭代在什么精度里算
NSB_DIFF = (12.8, 1.6)                    # 8 步回放：TPU 对 GPU 的更新差最多（%）：照 GPU 改成 BF16 迭代之前 ／ 之后
NSB_TPU_CLOSER = "97%"                    # 改之前，这类张量步里 TPU 比 GPU 更接近 FP64 的比例
NSB_ABS = (3.0e-3, 3.95e-2)               # 更新对 FP64 的相对误差中位数：TPU 原来（FP32 迭代）／ 照 GPU（BF16 迭代）
assert NSB_ABS[0] < NSB_ABS[1] / 10
NSB_ABS_T = "%s 对 %s" % (tx(NSB_ABS[0], "%.1e"), tx(NSB_ABS[1], "%.2e"))
# 算法前提被打破（§10.3）：裁剪缩放 η ＝ min(1, τ ÷ 最大 logit)，最大 logit 为负时 η 为负，开根号得 NaN
QKC_SMAX, QKC_TAU = -0.4956, 100.0
QKC_ETA = min(1.0, QKC_TAU / QKC_SMAX)
assert QKC_ETA < 0
QKC_ETA_T = tx(QKC_ETA, "%.1f"); _lock("QKC_ETA_T", QKC_ETA_T, "−201.8")
QKC_SMAX_T = tx(QKC_SMAX, "%.4f")
# 开关三类（§9.5）：用途分类以 1% 步时为界（本课取同一口径）
SW_COST_GATE = "1%"
SW_MIX_COST = "0.4%"                      # 逐项 FP32 乘加：步时 ＋0.4%（只差这一个开关的两次运行）
SW_ELIDE_COST = "0.06%"                   # 强制舍入：跨代对比 ＋0.06%
# 规模一级级往上（§9.6）
LADDER_MID_PASS = (5, 0, 4)               # 中等尺寸（随机初始化，宽度和注意力形状同真实模型）300 步：过 ／ 不过 ／ 待定
LADDER_MID_WORST = 1.04                   # 四个待定都只略超 N：最差 1.04 N
LADDER_REAL_STEP1 = -8.5e-5               # 真实权重截层：第 1 步前向主 loss 就差（GPU 8 份第 1 步逐位相同）
LADDER_MINI_STEP1 = (1.1e-6, 3.6e-6)      # 小模型各代第 1 步的差（范围），在 GPU 份间散布以内
assert abs(LADDER_REAL_STEP1) / max(LADDER_MINI_STEP1) > 20
LADDER_MINI_T = "1.1e-6–3.6e-6"
LADDER_REAL_STEP1_T = tx(LADDER_REAL_STEP1, "%.1e")
# 分叉以后（§6.3）：TPU 的 FP32 轨迹从第 2 步起离两条 BF16 轨迹 2e-3 到 1e-2，是两边之差的 30–100 倍
DIV_FP32_AWAY = "2e-3 到 1e-2"
DIV_STEP1 = (-2.0e-5, 6.5e-5)             # 第 1 步（同一份权重）对 FP32 参照：TPU ／ GPU 主 loss 的相对差
DIV_STEP2_10 = 6.8e-5                     # 第 2–10 步 TPU 对 GPU 8 份均值的带符号相对差 |.|
assert 2e-3 / DIV_STEP2_10 > 25
DIV_STEP1_R = abs(DIV_STEP1[0]) / abs(DIV_STEP1[1])
assert abs(DIV_STEP1_R - 0.31) < 0.01
# 混元 3 第 1 步按「尺子只来自参照」重判（§9.4）
HY3_REJUDGE = (11, 10, 1)                 # 变严项数：合计 ／ 待定→不过 ／ 过→待定；没有一项变松
HY3_REJUDGE_R = "1.23–2.70"               # 不过的里，一部分张量对 FP64 的 R 在这个范围（字面上不算已定责）
assert HY3_REJUDGE[0] == HY3_REJUDGE[1] + HY3_REJUDGE[2]
NSB_GPU = "1.2%–13%"                      # 参照（GPU）那段迭代的更新离 FP64（同一 8 步回放）
HY3_REJ_N, HY3_REJ_A, HY3_REJ_B = HY3_REJUDGE
NSB_DIFF_0, NSB_DIFF_1 = "%.1f" % NSB_DIFF[0], "%.1f" % NSB_DIFF[1]
LADDER_MID_T = "%d ／ %d ／ %d" % LADDER_MID_PASS
DIV_STEP1_R_T = "%.2f" % DIV_STEP1_R
DIV_STEP11_100 = 4.8e-5                   # 第 11–100 步 TPU 对 GPU 8 份均值的带符号相对差 |.|（同号，负）
DIV_FP32_STEPS = 30                       # FP32 轨迹只跑了 30 步
SW_ADAM_COST = "0.02%"                    # 精确的除法、开方、偏差修正：跨代对比 ＋0.02%

# ── 2026-11-10 · 第二次迁移的新材料（TPU 自比、GPU 尺子扩到 16 份）──────────────────────────
# 出处：第二次迁移的对齐记录（小尺寸模型，随机初始化，约 4.3 亿参数；TPU v5p 8 设备 ↔ B200×8）。
#   alignments/04-tpu-self/README.md、MATRIX.md（TPU 自比：脾气表、机制、留一法、长跑）；
#   alignments/04-tpu-self/results/gpu-rerun-loo-r-{300,1000}.json（GPU 留一法，逐种轮换的 r）；
#   alignments/01-mini/gpu-noise/results/g3x16-noise-vs-8.json（8 份 → 16 份的上限对比）。
#   逐点数据导出在 topic09_tpuself.json（私有导出脚本，只含数字和改写过的中文标签）。
_TS = tpuself()
LOO8_1K_MAX = 2.00                     # 1000 步只有 8 份时 GPU 留一法最差 r 的最大值（MATRIX「GPU 重跑的留一法 r」表第 2 行，已被 16 份取代）
LOO8_1K_MAX_T = "%.2f" % LOO8_1K_MAX
LOO16_1K_MAX = _TS["loo_1000_n16"]["max"]          # 1.82（16 份，240 种轮换）
LOO8_300_MAX = _TS["loo_300_n8"]["max"]            # 2.47（300 步 8 份，56 种轮换）
assert abs(LOO16_1K_MAX - 1.82) < 0.005 and abs(LOO8_300_MAX - 2.47) < 0.005
LOO16_1K_MAX_T, LOO8_300_MAX_T = "%.2f" % LOO16_1K_MAX, "%.2f" % LOO8_300_MAX
_GZ = _TS["grayzone_1000"]
GZ8_LO_T, GZ8_HI_T = "%.2f" % min(g["r1000_n8"] for g in _GZ), "%.2f" % max(g["r1000_n8"] for g in _GZ)
GZ16_LO_T, GZ16_HI_T = "%.2f" % min(g["r1000_n16"] for g in _GZ), "%.2f" % max(g["r1000_n16"] for g in _GZ)
assert (GZ8_LO_T, GZ8_HI_T, GZ16_LO_T, GZ16_HI_T) == ("2.00", "2.75", "1.18", "1.62")
N16_WIDEN_T = "1.0–2.1"                # 16 份 120 对的上限 ÷ 8 份 28 对（g3x16-noise-vs-8.json；MATRIX「L 长跑」第一条）
N16_GN_WIDEN = 1.70                    # 其中梯度范数带符号均值放宽 1.70 倍（同上；路由偏置均值 2.07 倍最大）
# 词嵌入梯度（第二次迁移 TPU 自比，第 1 步对 CPU FP64 的 rel_L2；倍数按 max(golden 4.00e-3, GPU 三份中位数 4.17e-3)）
EMB_DP_X = "3.6"                       # 纯数据并行 8：1.51e-2（MATRIX B1、ts-dp8 结果文件）
EMB_NOEXC_X = "26"                     # 关 excess precision：1.09e-1（MATRIX C、根因一览第 22 行 R 26.1）
assert abs(1.51e-2 / 4.172e-3 - 3.62) < 0.01 and abs(1.090e-1 / 4.172e-3 - 26.1) < 0.1
# TPU 自比的脾气（对齐记录 alignments/04-tpu-self/README.md 脾气表与 MATRIX「H v5p ↔ v7x」「C6」，实测）
LONG_ARMS = 32                         # 跑满 1000 步的改动数；只有 1 个分得开（多路残差混合不强制舍 BF16，第 1 步抓到）
# （以 README 汇总表「长跑（1000 步）32 臂、分得开 1」为准；脾气表的 1000 步一列只挂在有 300 步对应行的改动上）
V7X_STEP1_MED, V7X_FP64_RATIO, V7X_R300 = "3.9e-3", "1.01", "1.02"
V7X_STEP_T, V5P_STEP_T, V7X_SAVE = "0.343 s（4 片）", "0.494 s（8 片）", "65%"
assert abs(1 - 0.343 * 4 / (0.494 * 8) - 0.65) < 0.01  # 芯片·秒：0.343×4 ÷ (0.494×8) ＝ 0.347 → 省 65%（本课核算，跟 README「-65.3%」对上）
SC_AR_R = "1.03"
