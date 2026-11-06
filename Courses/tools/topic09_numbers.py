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
# 精确轨迹扰动出的五份两两 10 对的上限 vs GPU 生产程序重跑 10 对的上限（param_norm 带符号均值，实测）。
# ⛔ 旧值 6.9e-9 只是单对（第二份给第 0 层一处权重乘 1 ＋ 1e-6），不能代表「精确轨迹自己的散布」；五份 10 对的上限跟 GPU 那把同一量级。
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


# ── 第一层验收（v5p 当前配置对精确轨迹那一份，噪声取 GPU 生产程序重跑 10 对）────────
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
HEAD_TPU_S1, HEAD_TPU_S100 = -2.4e-5, -7.4e-5         # TPU 输出头反向对 FP64
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
TPU_OWN_NOREMAT_975 = 1.7e-3     # 不重算的配置，第 975 步，对精确轨迹
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
MASTER_CMP_TENSORS = 50                              # 加载后比 FP32 主权重：GPU 那边抓下来的张量数。来源：课件既有数，待核
NATIVE_SAME_RANK_T = format(NATIVE_SAME_RANK, ","); _lock("NATIVE_SAME_RANK_T", NATIVE_SAME_RANK_T, "2,000")
MICRO_SPLIT_EMB_X = 31                               # 只换梯度累积切分，embedding 梯度差 ÷ GPU 重跑差。来源：课件既有数，待核
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
LAYER_RELL2_FAIL = (0.016, 0.020, 0.028)             # v5p 第 1 步逐层回放：参考档不过、严格档过的三层（实测）
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
TPU_NOISE_PN_N = 8.5e-8                              # TPU 执行噪声 28 对的参数范数噪声上限。来源：课件既有数，待核
assert EP8_DELTA < TPU_NOISE_PN_N and sci(EP8_DELTA) == "7.5e-8"
WINDOWED_OFF_PN_DROP_T = "3%"                        # 关掉 collective matmul 窗口化，参数范数差只少 3%。来源：课件既有数，待核
# excess precision 那段：TPU 三种跑法彼此的散布（也就是只改片上内存上限那一臂挪的量）
ARMS_SPREAD = max(TPU_ARMS) - min(TPU_ARMS)
assert ARMS_SPREAD == TPU_ARMS[2] - TPU_ARMS[0]
ARMS_SPREAD_T = tx(ARMS_SPREAD, "%.0e"); _lock("ARMS_SPREAD_T", ARMS_SPREAD_T, "6e-5")
# 步号错位、分段
MISALIGN_OK_T, MISALIGN_OFF_T = tx(MISALIGN[0]), tx(MISALIGN[1])
_lock("MISALIGN_OK_T", MISALIGN_OK_T, "1.53e-4"); _lock("MISALIGN_OFF_T", MISALIGN_OFF_T, "1.41e-2")
MISALIGN_X_FLOOR = 90; assert MISALIGN_X_FLOOR < MISALIGN[1] / MISALIGN[0] < MISALIGN_X_FLOOR + 10   # 「大 90 多倍」
LOSS_TAIL_ROUGH_T = "−1.5e-4"                        # 「300 步以后稳定在 −1.5e-4 左右」。来源：课件既有数，待核（只断言落在三种跑法范围内）
assert min(TPU_ARMS) <= -1.5e-4 <= max(TPU_ARMS)
# 排除清单
WD_MASK_X = 13                                       # 权重衰减掩码反过来，残差大 13 倍以上。来源：课件既有数，待核
UPD1_RATIO_LO_T, UPD1_RATIO_HI_T = "0.997", "1.003"  # 第 1 步逐张量更新量之比。来源：课件既有数，待核
RARE_VHAT_DIFF_T = "0–4%"                            # 冷门行 √v̂ 两边只差 0–4%。来源：课件既有数，待核
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
FP32_TRAIN_LOSS_GAP_T = "1.75%"                      # 纯 FP32 训练前 100 步 loss 比 BF16 低。来源：课件既有数，待核
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
FP32_MASTER_FAKE_DIFF_T = "1.5%"                     # 拿 FP32 主权重当参照量出的假差。来源：课件既有数，待核
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
ROUTER_TPU_FP64_HI, ROUTER_GPU_FP64_HI = 5.7e-8, 4.0e-6   # 路由打分对 FP64 三层的上沿（下沿是 ROUTER_*_FP64）。来源：课件既有数，待核
ROUTER_GPU_FP64_HI_T = tx(ROUTER_GPU_FP64_HI, "%.1e"); _lock("ROUTER_GPU_FP64_HI_T", ROUTER_GPU_FP64_HI_T, "4.0e-6")
MOMENTUM_INV_LO, MOMENTUM_INV_HI = 2e-7, 4e-7        # 反推梯度对直接抓的梯度 rel_L2（GPU 上）。来源：课件既有数，待核
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
# 动画 TruncVsRound 的模拟终点（合成正数输入、按实测张量核模型模拟，不是实测曲线）。来源：课件既有数，待核
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
PN_MEAS_TPU, PN_MEAS_GPU, PN_RERUN_STEP = 2.6e-7, 8e-8, 2.7e-7   # param_norm 自测误差（单步最大差）：TPU 整网一次归约 / GPU；GPU 重跑两两单步最大差
GN_N_AB, GN_N_BA = 1.26e-3, 2.49e-3           # grad_norm 带符号均值的 N，随配对方向变
GN_JITTER = 0.03
VS_RC_PN, VS_RC_LOSS_V5P, VS_RC_LOSS_V7X = 2.09e-2, -1.37e-4, -1.08e-4   # v7x 一项是推断（对 ⑤ 的差加 ⑤ 对 ② 的差）
CHUNK_SEG, CHUNK_STEP_OLD, CHUNK_STEP_NEW = 1024, 0.236, 0.382
CHUNK_COST = CHUNK_STEP_NEW / CHUNK_STEP_OLD - 1
assert abs(CHUNK_COST - 0.62) < 0.01
DFA_COST_V7X, DFA_COST_V5P, EP8_SPEEDUP_V7X = 0.019, 0.0066, 0.28
FLIP_SLOT_LO, FLIP_SLOT_HI = 0.44, 0.56       # 第 1 步每层翻转的槽位百分比
# 落进正文
FINAL_WORST_T, FINAL_WORST_OLDCAL_T = "%.2f" % FINAL_WORST, "%.2f" % FINAL_WORST_OLDCAL
_lock("FINAL_WORST_T", FINAL_WORST_T, "0.85"); _lock("FINAL_WORST_OLDCAL_T", FINAL_WORST_OLDCAL_T, "0.84")
FINAL_PN_V5P_T, FINAL_PN_V7X_T = "%.2f" % FINAL[0][4], "%.2f" % FINAL[1][4]
_lock("FINAL_PN_V5P_T", FINAL_PN_V5P_T, "0.50"); _lock("FINAL_PN_V7X_T", FINAL_PN_V7X_T, "0.53")
PEND10_T = "%.2f–%.2f" % (PEND10_LO, PEND10_HI); _lock("PEND10_T", PEND10_T, "1.02–1.24")
PN_MEAS_TPU_T, PN_MEAS_GPU_T, PN_RERUN_STEP_T = tx(PN_MEAS_TPU, "%.1e"), tx(PN_MEAS_GPU, "%.0e"), tx(PN_RERUN_STEP, "%.1e")
_lock("PN_MEAS_TPU_T", PN_MEAS_TPU_T, "2.6e-7"); _lock("PN_MEAS_GPU_T", PN_MEAS_GPU_T, "8e-8"); _lock("PN_RERUN_STEP_T", PN_RERUN_STEP_T, "2.7e-7")
GN_N_AB_T, GN_N_BA_T = tx(GN_N_AB), tx(GN_N_BA); _lock("GN_N_AB_T", GN_N_AB_T, "1.26e-3"); _lock("GN_N_BA_T", GN_N_BA_T, "2.49e-3")
ENSEMBLE_SPREAD_TPU_T, ENSEMBLE_SPREAD_GPU_T = tx(ENSEMBLE_SPREAD_TPU, "%.1e"), tx(ENSEMBLE_SPREAD_GPU, "%.1e")
VS_RC_LOSS_V5P_T, VS_RC_LOSS_V7X_T = tx(VS_RC_LOSS_V5P), tx(VS_RC_LOSS_V7X)
_lock("VS_RC_LOSS_V5P_T", VS_RC_LOSS_V5P_T, "−1.37e-4"); _lock("VS_RC_LOSS_V7X_T", VS_RC_LOSS_V7X_T, "−1.08e-4")
CHUNK_COST_T = "%.0f%%" % (CHUNK_COST * 100); _lock("CHUNK_COST_T", CHUNK_COST_T, "62%")
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
MASTER_W_V5P, MASTER_W_V7X = 1.12, 0.99                              # 换尺子后，更新后主权重（v5p 待定、v7x 过）
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
