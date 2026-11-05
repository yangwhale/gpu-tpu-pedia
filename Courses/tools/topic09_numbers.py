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
ROWS_300 = [("没出现过", 17.1), ("出现 1–10 次", 3.1), ("11–100 次", 2.5), ("常见词", 0.0)]
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
SWAP = [("TPU 生产程序", 0.00, (-0.04, 0.10)), ("GPU 只换精确 dgrad", -0.10, (-0.22, -0.05)), ("GPU 生产程序", -0.36, (-0.47, -0.31))]
# 精确轨迹站在哪一边（第 975 步 param_norm 对 4 份默认 GPU 运行的均值）
EXACT_REF = 4.5e-4
EXACT_POS = [("精确 dgrad 的 GPU", 1.25e-2), ("精确轨迹", 1.82e-2), ("TPU", 1.99e-2)]
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
