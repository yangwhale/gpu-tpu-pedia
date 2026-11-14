# -*- coding: utf-8 -*-
r"""专题八 · GPU→TPU 训练精度对齐 —— 全讲共用的数（唯一一份）。

⭐ 一个事实一个落点：图脚本、课件正文、开场题都从这里取，不各抄一份（手册 03 · 二 · 5）。
⭐ 出处约定：
   · 「实测」= 本课程作者做的一次对齐实践：混元 3（Hy3，截前 4 层、去掉 MTP），
     GPU 侧 B200×8 + Megatron-Bridge，TPU 侧 v5p（8 颗芯片）／ v7x（4 颗芯片） + MaxText，同一份权重、同一批数据、同样的顺序，1000 步。
   · 「本课推导」= 自己算的，推导链写在旁边。
   · 逐步序列（loss、param_norm）从 topic08_series.json 现算，下面的常数只是给断言用的锚点。
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
    with open(os.path.join(HERE, "topic08_series.json"), encoding="utf-8") as fh:
        return json.load(fh)


def mean(xs):
    return sum(xs) / float(len(xs))


def tpuself():
    """第二次迁移的 TPU 自比与 GPU 留一法（topic08_tpuself.json，导出自对齐记录 TPU 自比一节，实测）。"""
    with open(os.path.join(HERE, "topic08_tpuself.json"), encoding="utf-8") as fh:
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

# ── 2026-11-05 · fig8-baseline：10 份 GPU 运行各自的尾段 loss 偏移（实测）──────────────────
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

# ── 2026-11-05 · fig8-layer0：残差流固定探针，与第 0 层权重互换（实测）──────────────────────
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
L0_SWAP_SMALL = 6e-5                             # 两个归一化的 scale、q／k 归一化：都在这以下（MLP gate／up −6.6e-5 单列）
assert abs(L0_SWAP_100[0] / L0_PROBE["第 0 层"][_i100] - 1) < 0.15                       # 换权重重算 ≈ 探针实测：差全在权重上
assert abs(L0_SWAP_50[0] / L0_PROBE["第 0 层"][_i50] - 1) < 0.5
assert max(abs(x) for x in L0_SWAP_100[1:]) < abs(L0_SWAP_100[0]) / 5
assert abs(L0_SWAP_ONE_100[0][1]) > 0.6 * abs(L0_SWAP_100[0])                            # 大头在注意力输出投影（约七成）

# ── 2026-11-05 · fig8-excess 的 📌（实测，混元 3 · v5p）──────────────────────────────
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

# ── 2026-11-05 · 数字锁：课件正文（topic08-build.py）原先手打的数，一律从这里取 ───────────────────
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
PN_LEVEL0 = 3042.377                                 # GPU 第一份运行第 1 步参数范数（fig8-drift 的 📌 实测：3042.377 → 3042.503）
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
# 窗口化开关、golden 一步步（出处：fig8-golden 的同一组实测：v5p 开关前配置对 ②③④ 的参数范数带符号均值）
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
LOSS_TAIL_ROUGH_T = "−1.7e-4"   # 2026-11-15 全课统一：第 301–1000 步带符号均值 −1.66e-4 取两位（原写案例记录原话「稳定在 −1.5e-4 左右」，对不上任何算出来的统计量）
_LOSS_TAIL_ROUGH_OLD = "−1.5e-4"                        # 「300 步以后稳定在 −1.5e-4 左右」。出处：案例记录 §3.1 原话（只断言落在三种跑法范围内）
assert abs(-1.7e-4 - TPU_ARMS[0]) < 0.05e-4
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
# 状态互换导入 TPU、逐个核对 sha256 的张量数（fig8-swap 的 📌 实测）
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
TPU_TAIL_STEP_STD = 1.33e-4                          # TPU 对 GPU 第一份逐步差的标准差，第 301–1000 步（fig8-dt 的 📌 实测）
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
HEADFP32_COST = 0.096                     # 0.98016 → 1.07457 s（案例 01 第 188、610 行）
# 落进正文的写法
PN_FIX1_T, PN_PRODCHUNK_T, PN_TRUE_T, PN_TPU_PROD_T = (tx(v, "%.2e", plus=True) for v in (PN_FIX1, PN_PRODCHUNK, PN_TRUE, PN_TPU_PROD))
_lock("PN_FIX1_T", PN_FIX1_T, "＋1.86e-2"); _lock("PN_PRODCHUNK_T", PN_PRODCHUNK_T, "＋2.07e-2"); _lock("PN_TRUE_T", PN_TRUE_T, "＋2.74e-2"); _lock("PN_TPU_PROD_T", PN_TPU_PROD_T, "＋2.09e-2")
B_GPUEXACT_T, B_DLOGITS_T, B_REST_T, TPU_VS_FIX1_T = tx(B_GPUEXACT, plus=True), tx(B_DLOGITS), tx(B_REST, "%.1e", plus=True), tx(TPU_VS_FIX1, plus=True)
_lock("B_GPUEXACT_T", B_GPUEXACT_T, "＋8.70e-3"); _lock("B_DLOGITS_T", B_DLOGITS_T, "−6.84e-3"); _lock("B_REST_T", B_REST_T, "＋4.9e-4"); _lock("TPU_VS_FIX1_T", TPU_VS_FIX1_T, "＋2.35e-3")
TPU_VS_TRUE_T = tx(abs(TPU_VS_TRUE), "%.2e"); _lock("TPU_VS_TRUE_T", TPU_VS_TRUE_T, "6.35e-3")
LADDER_DGRAD_T, LADDER_SHARED_T = tx(LADDER_DGRAD, plus=True), tx(LADDER_SHARED, "%.1e", plus=True)
_lock("LADDER_DGRAD_T", LADDER_DGRAD_T, "＋2.07e-2"); _lock("LADDER_SHARED_T", LADDER_SHARED_T, "＋6.7e-3")
SHARED_DLOGITS_GPU_T = tx(SHARED_DLOGITS_GPU, "%.1e"); _lock("SHARED_DLOGITS_GPU_T", SHARED_DLOGITS_GPU_T, "−6.7e-3")
CUBLAS_S1_T, CUBLAS_299_T = tx(CUBLAS_S1, "%.1e"), tx(CUBLAS_299, "%.1e"); _lock("CUBLAS_S1_T", CUBLAS_S1_T, "−1.8e-6"); _lock("CUBLAS_299_T", CUBLAS_299_T, "−2.6e-4")
PRODCHUNK_PN_R = PRODCHUNK_PN_V5P / NOISE_PN; PRODCHUNK_LOSS_R = PRODCHUNK_LOSS_TAIL_V5P / NOISE_LOSS_TAIL
PRODCHUNK_PN_R_T, PRODCHUNK_LOSS_R_T = "%.2f" % PRODCHUNK_PN_R, "%.2f" % PRODCHUNK_LOSS_R; _lock("PRODCHUNK_PN_R_T", PRODCHUNK_PN_R_T, "0.42"); _lock("PRODCHUNK_LOSS_R_T", PRODCHUNK_LOSS_R_T, "0.59")
TRUEEXACT_R_T = "%.1f 到 %.1f" % (abs(TRUEEXACT_PN_V5P) / NOISE_PN, abs(TRUEEXACT_PN_V7X) / NOISE_PN); _lock("TRUEEXACT_R_T", TRUEEXACT_R_T, "6.2 到 7.1")
HEADFP32_COST_T = "%.1f%%" % (HEADFP32_COST * 100); _lock("HEADFP32_COST_T", HEADFP32_COST_T, "9.6%")
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
VS_RC_PN, VS_RC_LOSS_V5P, VS_RC_LOSS_V7X = 2.09e-2, -1.37e-4, -1.07e-4   # v7x 一项 11-11 晚起直接算（混元 3 对齐报告；之前是推断的 −1.08e-4）
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
_lock("VS_RC_LOSS_V5P_T", VS_RC_LOSS_V5P_T, "−1.37e-4"); _lock("VS_RC_LOSS_V7X_T", VS_RC_LOSS_V7X_T, "−1.07e-4")
CHUNK_COST_T = "%.0f%%" % (CHUNK_COST * 100); _lock("CHUNK_COST_T", CHUNK_COST_T, "61%")
DFA_COST_V7X_T, DFA_COST_V5P_T, EP8_SPEEDUP_V7X_T = "%.1f%%" % (DFA_COST_V7X * 100), "%.2f%%" % (DFA_COST_V5P * 100), "%.0f%%" % (EP8_SPEEDUP_V7X * 100)
_lock("DFA_COST_V7X_T", DFA_COST_V7X_T, "1.9%"); _lock("DFA_COST_V5P_T", DFA_COST_V5P_T, "0.66%"); _lock("EP8_SPEEDUP_V7X_T", EP8_SPEEDUP_V7X_T, "28%")
FLIP_SLOT_T = "%.2f%%–%.2f%%" % (FLIP_SLOT_LO, FLIP_SLOT_HI)
# 路由翻转的有效证据（L1/L2 隔离回放）：路由输入换成 TPU 自己算的，每层翻的 token 比例
FLIP_L2_LO, FLIP_L2_HI = 3.5, 5.4
FLIP_L2_T = "%.1f%%–%.1f%%" % (FLIP_L2_LO, FLIP_L2_HI); _lock("FLIP_L2_T", FLIP_L2_T, "3.5%–5.4%")
# 「两边全程 FP32 也翻」那条证据已撤回：TPU 的 FP32 参照里路由专家其实按 BF16 算（FP32_MARGIN 只留作台账旧数）
FINAL_PN_SM, FINAL_PN_SM_N = 1.593e-8, 2.269e-7      # v5p 当前配置对 ⑤：参数范数带符号均值、45 对上限（fig8-golden 柱子用带符号均值，跟 ②–④ 同口径）
FINAL_PN_SM_R = FINAL_PN_SM / FINAL_PN_SM_N
FINAL_PN_SM_R_T = "%.2f" % FINAL_PN_SM_R; _lock("FINAL_PN_SM_R_T", FINAL_PN_SM_R_T, "0.07")
B_V5P = (7.965e-3, -6.104e-3, 0.488e-3)              # v5p 那组补跑的三笔（v7x 是 B_GPUEXACT / B_DLOGITS / B_REST）
ENSEMBLE_PEND_T = "MARE %.2f、最大单步 %.2f" % (ENSEMBLE_PEND_MARE, ENSEMBLE_PEND_MAX); _lock("ENSEMBLE_PEND_T", ENSEMBLE_PEND_T, "MARE 1.02、最大单步 1.15")
V7X_L2_R_T = " ／ ".join("%.2f" % v for v in V7X_L2_R); _lock("V7X_L2_R_T", V7X_L2_R_T, "1.06 ／ 1.01 ／ 1.01 ／ 1.02")

# ── 11-06 下午：第 1 步张量换尺子（两侧执行噪声并集）＋ 一处定责 ─────────────────────────────
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
MASTER_W_V5P, MASTER_W_V7X = 1.63, 1.61                              # 尺子只用 GPU（11-11 起）后，更新后主权重：两平台都待定（混元 3 对齐报告 L3）
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
ATTN_NOISE_LOSS, ATTN_NOISE_GN = 1.7e-6, 9.3e-5       # GPU 同设置 8 份第 1 步相对极差（改后那一代 g2；对齐记录 11-11 换的口径，旧的 28 对最大差 4.5e-5／2.2e-3 已撤）
assert ATTN_FIX_LOSS < ATTN_NOISE_LOSS and abs(ATTN_FIX_GN) < ATTN_NOISE_GN   # 跟散布同一量级、略小：单看第 1 步分不清
ATTN_FIX_LOSS_T, ATTN_FIX_GN_T = tx(ATTN_FIX_LOSS, "%.1e", plus=True), tx(ATTN_FIX_GN, "%.1e")
ATTN_NOISE_LOSS_T, ATTN_NOISE_GN_T = tx(ATTN_NOISE_LOSS, "%.1e"), tx(ATTN_NOISE_GN, "%.1e")
_lock("ATTN_FIX_LOSS_T", ATTN_FIX_LOSS_T, "＋8.1e-7"); _lock("ATTN_NOISE_GN_T", ATTN_NOISE_GN_T, "9.3e-5")
# 「HIGHEST 是不是多遍 BF16，看编译后还是不是矩阵乘」：多路残差四路加权和，TPU 默认写成 HIGHEST einsum，
# 编译器把它改写成 FP32 逐元素乘 ＋ FP32 求和（不上矩阵单元），跟 GPU 只差加法顺序（TPU 配对、GPU 依次）。
# 改成依次相加以后，第 1 步该处梯度对 CPU FP64 R ＞ 1.2 的张量 11 → 8；后一代配置没复现，机制待查。
MIX_OVER = (11, 8)
MIX_OVER_T = "%d → %d" % MIX_OVER; _lock("MIX_OVER_T", MIX_OVER_T, "11 → 8")
# 病态：稀疏注意力里一项辅助蒸馏损失（KL）每行约 1.1e-4 nat，log-sum-exp 约 7.1
KD_KL, KD_LSE = 1.1e-4, 7.1
KD_ULP = 4.73e-7                         # f32 在 lse 处一格（实测各行平均；lse ∈ [4, 8) 时 2⁻²¹ ≈ 4.77e-7）
assert abs(KD_ULP - 2.0 ** -21) / 2.0 ** -21 < 0.02
KD_ULP_OF_KL = 4.31e-3                   # 一格 ÷ KL（实测；按上两数 4.73e-7 ÷ 1.1e-4 ≈ 4.3e-3，本课推导核对）
assert abs(KD_ULP / KD_KL - KD_ULP_OF_KL) / KD_ULP_OF_KL < 0.01
KD_LSE_TPU_ULP, KD_LSE_CPU_ULP = -0.953, -0.004    # TPU 图内 ／ CPU torch f32 的 lse 对 FP64，按格计的平均
KD_POS_ROWS = "8.6%"                     # TPU 偏正的行只占这么多：方向一致
KD_DIFF = -1.155e-3                      # 同一行 KD：TPU 图内对同一输入的 FP64 重算（相对）
KD_IN_R, KD_GRAD_R = "0.94–0.99", "0.81–1.05"     # 它的三组输入对 CPU FP64 的 R ／ 由它出发的梯度（20 个张量）对 CPU FP64 的 R（最大第 5 层 weights_proj：GPU 6.81e-3、TPU 7.14e-3；旧的 0.78–1.03 是对 TPU 自己 FP32 参照算的，已撤）
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
# R 在误差不到一个 BF16 格时会摆：随上游一处写法摆（对齐记录结果文件最接近的读数 0.62–1.59；旧注释的 0.72／1.60 找不到出处，已换）
R_SWING = "0.6–1.6"
# 门控激活 silu·up：GPU 舍两次、TPU 默认舍一次时逐位相同的元素比例（混元 3）
GLU_BITWISE = "73.7%"
# 正文直接引用的单值（«名字» 只认名字，不认下标）
B2_MED_OFF, B2_MED_4SW = B2_RATIO_MED[0], B2_RATIO_MED[2]
B2_OVER_OFF, B2_OVER_4SW = B2_OVER2[0], B2_OVER2[2]
INJ_R_BEFORE, INJ_R_AFTER = INJ_R
INJ_R_AFTER_T = "%.2f" % INJ_R_AFTER; _lock("INJ_R_AFTER_T", INJ_R_AFTER_T, "0.95")
KD_VS_GPU = -9.2e-4                      # 第 1 步 KD：TPU 全部运行对 GPU 全部运行（相对），区别于单行对 FP64 的 KD_DIFF
KD_VS_GPU_T = tx(KD_VS_GPU, "%.1e"); _lock("KD_VS_GPU_T", KD_VS_GPU_T, "−9.2e-4")
# §5.6 折叠的收尾：现行五个开关下第 1 步逐张量（205 个张量的那份逐张量表）
B2_ALL_TENSORS, B2_FINAL_OVER, B2_FINAL_RMED, B2_FINAL_OVER_ATTR = 205, 58, 0.95, 56
assert B2_FINAL_OVER_ATTR <= B2_FINAL_OVER < B2_ALL_TENSORS
B2_GN_NOW = 1.34                         # 现行比较（参照改后、TPU 补齐几处边界）梯度范数 r，MARE，待定

# ══════════════════════════════════════════════════════════════════════════════════════
# ── 11-08 · 第二次迁移的后续（出处：同一实践的根源一览、改动台账、结果文件；实测，除标推断／本课推导）──────
# ══════════════════════════════════════════════════════════════════════════════════════
# 编译器省掉的舍入（§5.5）：改成逐项 FP32 乘加之后，GPU 一定会做的 FP32→BF16→FP32 往返在 TPU 上被省掉（HLO 里 convert 都还在；最小复现的后端指令里一半被省掉，见根源一览第 16 行）
ELIDE_GPU_MAX = 3.77e-6                   # GPU 8 份两两 28 对，1000 步主 loss 带符号均值的最大 |.|
ELIDE_BEFORE = 1.07e-6                    # 改逐项乘加之前那一代 TPU：对 GPU 8 份均值的带符号均值
ELIDE_RUNS = (7.68e-6, 6.59e-6, 6.35e-6, 3.71e-6, 8.58e-6)   # 改之后 5 次 TPU 运行（含只改执行参数的 3 次），同号偏高
ELIDE_AFTER = 3.5e-6                      # 两处改用 reduce_precision 强制舍入以后（对 GPU 均值，实测：协议变更记录 +8.6e-6 → +3.5e-6）
ELIDE_R = (2.12, 0.76)                    # 同一量的判定 r（带符号均值，对 GPU 基线）：强制舍入前 ／ 后
assert all(x > 0 for x in ELIDE_RUNS) and min(ELIDE_RUNS) < ELIDE_GPU_MAX < max(ELIDE_RUNS)
assert ELIDE_AFTER < max(ELIDE_RUNS) and ELIDE_R[0] > 2 >= 1 >= ELIDE_R[1]
ELIDE_RANGE_T = "＋%.1e 到 ＋%.1e" % (min(ELIDE_RUNS), max(ELIDE_RUNS)); ELIDE_RANGE_T = ELIDE_RANGE_T.replace("e-06", "e-6")
_lock("ELIDE_RANGE_T", ELIDE_RANGE_T, "＋3.7e-6 到 ＋8.6e-6")
ELIDE_FIX_T = "＋%.1e" % ELIDE_AFTER; ELIDE_FIX_T = ELIDE_FIX_T.replace("e-06", "e-6")
_lock("ELIDE_FIX_T", ELIDE_FIX_T, "＋3.5e-6")   # 2026-11-11：研究线记的是实测值，去掉「推算」
ELIDE_OVER = sum(1 for x in ELIDE_RUNS if x > ELIDE_GPU_MAX)
assert ELIDE_OVER == 4
ELIDE_R_T = "%.2f → %.2f" % ELIDE_R
ELIDE_GPU_MAX_T = tx(ELIDE_GPU_MAX, "%.1e")
# 超越函数（§5.8）：TPU v5p 默认 FP32 实现对正确舍入的最大误差（格＝ulp），对照 GPU 文档上限与 TPU 高精度写法
TRANS = {   # 函数: (TPU 默认 ulp_max, TPU 高精度写法 ulp_max, GPU 上限)
    "exp": (105, 1, 2), "log": (62, 3, 1), "sigmoid": (83, 3, 3.4), "silu": (86, 4, 3.4)}
TRANS_GPU_NOTE = "exp ≤ 2、log ≤ 1 是 CUDA 文档给的上限；B200 实测 log 1、sigmoid 和 silu 最大 3.4（根源一览第 11 行）"
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
SW_MIX_COST = "0.2%"                      # 四路相加改依次（照 GPU 的顺序）：步时 ＋0.2%；对 FP64 更准未证实，归「看情况」
SW_ELIDE_COST = "0.06%"                   # 强制舍入：跨代对比 ＋0.06%
# 规模一级级往上（§9.6 开头）
LADDER_MID_PASS = (5, 0, 4)               # 中等尺寸（随机初始化，宽度和注意力形状同真实模型）300 步：过 ／ 不过 ／ 待定
LADDER_MID_WORST = 1.04                   # 四个待定都只略超 N：最差 1.04 N（最老一代读数；现行 C 口径最差 1.25，是 MTP loss，结论不变）
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
HY3_REJUDGE_R = "梯度最差 2.20／2.70、优化器状态最差 7.95／7.86（v5p／v7x）"   # 不过的里 R ＞ 1.2 的最差张量（出处：混元 3 judging.md、对齐报告；旧写的下限 1.23 无出处，已撤）
assert HY3_REJUDGE[0] == HY3_REJUDGE[1] + HY3_REJUDGE[2]
NSB_GPU = "1.2%–13%"                      # 参照（GPU）那段迭代的更新离 FP64（同一 8 步回放）
HY3_REJ_N, HY3_REJ_A, HY3_REJ_B = HY3_REJUDGE
NSB_DIFF_0, NSB_DIFF_1 = "%.1f" % NSB_DIFF[0], "%.1f" % NSB_DIFF[1]
LADDER_MID_T = "%d ／ %d ／ %d" % LADDER_MID_PASS
DIV_STEP1_R_T = "%.2f" % DIV_STEP1_R
DIV_STEP11_100 = 4.8e-5                   # 第 11–100 步 TPU 对 GPU 8 份均值的带符号相对差 |.|（同号，负）
DIV_FP32_STEPS = 30                       # FP32 轨迹只跑了 30 步
SW_ADAM_COST = "0.1%"                     # 精确的除法、开方、偏差修正：小尺寸实测 ＋0.1%（对齐记录差异清单 C5、对齐报告；旧写的 ＋0.02% 找不到出处）

# ── 2026-11-10 · 第二次迁移的新材料（TPU 自比、GPU 尺子扩到 16 份）──────────────────────────
# 出处：第二次迁移的对齐记录（小尺寸模型，随机初始化，约 4.3 亿参数；TPU v5p 8 设备 ↔ B200×8）。
#   TPU 自比一节的 汇总页和实验矩阵（TPU 自比：脾气表、机制、留一法、长跑）；
#   TPU 自比一节的 GPU 留一法的两份结果文件（300 步、1000 步，逐种轮换的 r）；
#   小尺寸一节的「GPU 8 份对 16 份上限」结果文件（8 份 → 16 份的上限对比）。
#   逐点数据导出在 topic08_tpuself.json（私有导出脚本，只含数字和改写过的中文标签）。
_TS = tpuself()
LOO8_1K_MAX = 2.00                     # 1000 步只有 8 份时 GPU 留一法最差 r 的最大值（实验矩阵「GPU 重跑的留一法 r」表第 2 行，已被 16 份取代）
LOO8_1K_MAX_T = "%.2f" % LOO8_1K_MAX
LOO16_1K_MAX = _TS["loo_1000_n16"]["max"]          # 1.82（16 份，240 种轮换）
LOO8_300_MAX = _TS["loo_300_n8"]["max"]            # 2.47（300 步 8 份，56 种轮换）
assert abs(LOO16_1K_MAX - 1.82) < 0.005 and abs(LOO8_300_MAX - 2.47) < 0.005
LOO16_1K_MAX_T, LOO8_300_MAX_T = "%.2f" % LOO16_1K_MAX, "%.2f" % LOO8_300_MAX
_GZ = _TS["grayzone_1000"]
GZ8_LO_T, GZ8_HI_T = "%.2f" % min(g["r1000_n8"] for g in _GZ), "%.2f" % max(g["r1000_n8"] for g in _GZ)
GZ16_LO_T, GZ16_HI_T = "%.2f" % min(g["r1000_n16"] for g in _GZ), "%.2f" % max(g["r1000_n16"] for g in _GZ)
assert (GZ8_LO_T, GZ8_HI_T, GZ16_LO_T, GZ16_HI_T) == ("2.00", "2.75", "1.19", "1.64")   # 16 份一列按分母统一后的结果
N16_WIDEN_T = "1.0–2.1"                # 16 份 120 对的上限 ÷ 8 份 28 对（「GPU 8 份对 16 份上限」结果文件；实验矩阵「长跑」第一条）
N16_GN_WIDEN = 1.84                    # 其中梯度范数带符号均值放宽 1.84 倍（分母统一为参照后；统一前 1.70；路由偏置均值 2.07 倍）
# 词嵌入梯度（第二次迁移 TPU 自比，第 1 步对 CPU FP64 的 rel_L2；倍数按 max(golden 4.00e-3, GPU 三份中位数 4.17e-3)）
EMB_DP_X = "3.62"                      # 纯数据并行 8：1.51e-2（实验矩阵纯数据并行一组、那一臂的结果文件）
EMB_NOEXC_X = "26.1"                   # 关 excess precision：1.09e-1（实验矩阵关省舍入一组、根因一览第 22 行 R 26.1）
assert abs(1.51e-2 / 4.172e-3 - 3.62) < 0.01 and abs(1.090e-1 / 4.172e-3 - 26.1) < 0.1
# TPU 自比的脾气（对齐记录 TPU 自比一节的 汇总页脾气表与实验矩阵 v5p ↔ v7x 一组，实测）
LONG_ARMS = 33                         # 跑满 1000 步的改动数；只有 1 个分得开（多路残差混合不强制舍 BF16，第 1 步抓到）（汇总页 2026-11-11：119 种里 33 种另外跑到 1000 步）
# （汇总页现行：33 种另外跑到 1000 步，分得开 1；脾气表的 1000 步一列只挂在有 300 步对应行的改动上）
V7X_STEP1_MED, V7X_FP64_RATIO, V7X_R300 = "3.9e-3", "1.01", "1.02"
V7X_STEP_T, V5P_STEP_T, V7X_SAVE = "0.343 s（4 片）", "0.494 s（8 片）", "65%"
assert abs(1 - 0.343 * 4 / (0.494 * 8) - 0.65) < 0.01  # 芯片·秒：0.343×4 ÷ (0.494×8) ＝ 0.347 → 省 65%（本课核算，跟汇总页「-65.3%」对上）
SC_AR_R = "1.03"
# 多重比较的粗算（本课推导，假设各统计量独立）：一个统计量超过 n 对最大值的机会约 1/(n＋1)，
#   m 个统计量里至少一个超的机会 ＝ 1 − (n/(n＋1))^m。对照 GPU 留一法实测（_TS）。
MC_300 = 1 - (21 / 22.0) ** 24          # 300 步：尺子 21 对、24 个统计 → 0.67；实测 0.59
MC_1K = 1 - (105 / 106.0) ** 27         # 1000 步：尺子 105 对、27 个统计 → 0.23；实测 0.22
assert abs(MC_300 - 0.67) < 0.005 and abs(MC_1K - 0.23) < 0.005
MC_300_T, MC_1K_T = "%d%%" % round(MC_300 * 100), "%d%%" % round(MC_1K * 100)
MC_300_OBS_T = "%d%%" % round(_TS["loo_300_n8"]["frac_gt_1"] * 100)
MC_1K_OBS_T = "%d%%" % round(_TS["loo_1000_n16"]["frac_gt_1"] * 100)
assert (MC_300_OBS_T, MC_1K_OBS_T) == ("59%", "22%")
# 同一种轮换（GPU 1000 步 16 份，240 种）单看一条检查的最高分（本课从逐种轮换数据算：每条检查取三个统计里最大的，再取 240 种里最大的）
LOO1_LOSS, LOO1_TAIL, LOO1_PN, LOO1_BM = "1.29", "1.23", "1.58", "1.82"   # 主 loss 全程、后段、参数范数、路由偏置均值


# ── 2026-11-11 · 第八节「案例二：DDP 和 FSDP 对不上」（第二次迁移 ＝ 混元 4 缩小版，TPU v5p 自比）──────────────
# 出处（公开源码里只写相对说法）：对齐记录里 DDP 与 FSDP 那个案例、梯度的跨卡归约、影响逐件「纯数据并行」
#   「关掉编译器的省舍入」；数字导出在 topic08_case2.json（私有导出脚本，读各臂结果文件和两份单算子结果）。
#   导不出来的（按行分组的表、编译产物里的指令条数）照文档原文抄成常数，并用能对上的别处数字互相钉住。
def case2():
    with open(os.path.join(HERE, "topic08_case2.json"), encoding="utf-8") as fh:
        return json.load(fh)


_C2 = case2()
_C2R = _C2["runs"]
C2_PARAMS_T = "4.3 亿"                # 缩小版：保留全部结构、只缩宽度和层数（同上「第二次迁移」小尺寸）
C2_HY4_TOTAL_T, C2_HY4_ACTIVE_T = "770B", "49B"   # 混元 4（Hy4）公开口径：总参数、激活参数（混元官网 Hy4 发布页）
C2_V, C2_D = _C2["probe"]["V"], _C2["probe"]["D"]   # 词表行数 120832、隐藏维 1024
C2_DEV, C2_TOK = 8, 4096                 # 8 张卡；每张卡每个微批 1 条 4096 token 的序列，每步 2 个微批
assert (C2_V, C2_D) == (120832, 1024)
C2_D_SHARD = C2_D // C2_DEV              # FSDP 下每张卡管 128 维
C2_FSDP_N = C2_DEV * C2_TOK              # FSDP 那次 scatter 收 8 张卡的全部 token：32768
C2_GATE = C2_V // 8                      # 门槛：下标数超过表行数的 1/8 → 15104
assert (C2_D_SHARD, C2_FSDP_N, C2_GATE) == (128, 32768, 15104)
# ① 现象：四种条件的曲线最差 r（300 步）
C2_R = {k: _C2R[k]["r_max"] for k in ("纯数据并行8", "纯数据并行8不累积", "纯数据并行16", "v7x纯数据并行8")}
assert ["%.2f" % v for v in C2_R.values()] == ["2.62", "3.69", "2.65", "2.21"]
C2_R_DP8_T, C2_R_GA1_T, C2_R_DP16_T, C2_R_V7X_T = ["%.2f" % v for v in C2_R.values()]
C2_LOSS_SM, C2_GN_SM = _C2R["纯数据并行8"]["loss_worst"][1], _C2R["纯数据并行8"]["grad_norm_worst"][1]
assert _C2R["纯数据并行8"]["loss_worst"][0] == _C2R["纯数据并行8"]["grad_norm_worst"][0] == "signed_mean"
assert abs(C2_LOSS_SM - 5.58e-6) < 0.005e-6 and abs(C2_GN_SM + 2.60e-4) < 0.005e-4
C2_LOSS_SM_T, C2_GN_SM_T = "＋5.58e-6", "−2.60e-4"
C2_LOSS_R_T = "%.2f" % _C2R["纯数据并行8"]["loss_r"]
assert C2_LOSS_R_T == "1.85"
# ② 第 1 步对 FP64：205 个张量，只有词嵌入梯度丢了精度
C2_NT = _C2R["纯数据并行8"]["n_step1_tensors"]
assert C2_NT == 205
_e = _C2R["纯数据并行8"]
C2_EMB_DP, C2_EMB_G, C2_EMB_GPU, C2_EMB_R = _e["emb_test"], _e["emb_golden"], _e["emb_gpu"], _e["emb_R"]
assert ("%.2e %.2e %.2e" % (C2_EMB_DP, C2_EMB_G, C2_EMB_GPU)) == "1.51e-02 4.00e-03 4.17e-03" and abs(C2_EMB_R - 3.62) < 0.005
C2_EMB_DP_T, C2_EMB_G_T, C2_EMB_GPU_T, C2_EMB_R_T = "1.51e-2", "4.00e-3", "4.17e-3", "3.62"
C2_OTHER_MED_T = "0.99"                  # 其余 204 个离 FP64 的比值（纯数据并行 ÷ golden）中位数（对齐记录）
C2_IHC_SKIP = 6                          # 6 个多路残差混合小参数，数值上病态，向来不定责（另算）
# ③ 第一个假设 → FP32 开关一开 r 611（修复前）；读数与 fig-bug 同一份
_b = _C2R["修复前开关"]
assert abs(_b["r_max"] - 611.3) < 0.1 and abs(_b["emb_R"] - 129.7) < 0.1
C2_BUG_R_T, C2_BUG_FAR_T = "611", "130"
# 第二种修法（本卡 FP32、跨卡 BF16）与修法①的读数
_f1, _f1k, _f2, _f2k = _C2R["修法1"], _C2R["修法1_1000"], _C2R["修法2"], _C2R["修法2_1000"]
C2_FIX1_EMB_T, C2_FIX2_EMB_T = "%.2e" % _f1["emb_test"], "%.2e" % _f2["emb_test"]
assert (C2_FIX1_EMB_T, C2_FIX2_EMB_T) == ("3.79e-03", "4.24e-03")
C2_FIX1_EMB_T, C2_FIX2_EMB_T = "3.79e-3", "4.24e-3"
C2_FIX1_R300_T, C2_FIX1_R1K_T = "%.2f" % _f1["r_max"], "%.2f" % _f1k["r_max"]
C2_FIX2_R300_T, C2_FIX2_R1K_T = "%.2f" % _f2["r_max"], "%.2f" % _f2k["r_max"]
assert (C2_FIX1_R300_T, C2_FIX1_R1K_T, C2_FIX2_R300_T, C2_FIX2_R1K_T) == ("0.98", "1.09", "0.93", "1.07")
C2_FIX1_16_T = "%.2f ／ %.2f" % (_C2R["修法1_16卡"]["r_max"], _C2R["修法1_16卡_1000"]["r_max"])
assert C2_FIX1_16_T == "0.99 ／ 1.30"
C2_FIX1_GA1_T = "%.2f" % _C2R["修法1_不累积"]["r_max"]
assert C2_FIX1_GA1_T == "1.01"
# 步时（对 golden，按每步中位数；负数＝比 golden 快）
C2_ST = {k: _C2R[k]["step_time_vs_golden"] for k in ("纯数据并行8", "修法1", "修法2")}
C2_ST_T = ["%.1f%%" % (100 * v) for v in C2_ST.values()]
assert C2_ST_T == ["-3.4%", "-2.8%", "-2.6%"], C2_ST_T
C2_ST_DP_T, C2_ST_F1_T, C2_ST_F2_T = [s.replace("-", "−") for s in C2_ST_T]
C2_ST_COST_T = "0.6–0.8 个百分点"          # 修法本身的代价：3.4 − 2.8 ＝ 0.6，3.4 − 2.6 ＝ 0.8（本课核算）
C2_COMM_BYTES = 9.90e8                   # 每步每设备跨卡字节数，默认／修法①／修法②三者相同（对齐记录）
# ④ 审计编译产物：按行分组（对齐记录里梯度跨卡归约那一页的表；三组互不重叠，合起来 10911 行）
C2_GROUPS = [  # (说明, 行数, 占梯度范数平方, golden, 纯数据并行8, 不累积(8192), 纯数据并行16, 修法①, 关省舍入)
    ("每个微批只在 1 张卡上出现", 7638, 1.3e-2, 9.31e-3, 9.41e-3, 9.16e-3, 9.25e-3, 9.31e-3, 1.02e-2),
    ("出现在 ≥ 2 张卡上、合计 ≤ 300 次", 3234, 8.9e-2, 7.77e-3, 8.31e-3, 8.26e-3, 8.33e-3, 7.28e-3, 9.58e-3),
    ("合计 ≥ 301 次（高频词）", 39, 0.90, 3.26e-3, 1.57e-2, 3.07e-2, 1.59e-2, 3.08e-3, 1.15e-1),
]
C2_ROWS = sum(g[1] for g in C2_GROUPS)
assert C2_ROWS == 10911
# 互相钉住：各组 rel² 按范数平方加权求和，应回到整张表的 rel（golden 4.00e-3、纯数据并行 1.51e-2、不累积 2.92e-2、关省舍入 1.09e-1）
for _col, _tot in ((3, C2_EMB_G), (4, C2_EMB_DP), (5, _C2R["纯数据并行8不累积"]["emb_test"]), (8, _C2R["关省舍入"]["emb_test"])):
    _rel = math.sqrt(sum(g[2] * g[_col] ** 2 for g in C2_GROUPS))
    assert abs(_rel / _tot - 1) < 0.03, (_col, _rel, _tot)
# 纯数据并行比 golden 多出来的误差（误差平方之差）里，高频 39 行占多少（本课按上表复算，对齐记录写 99.6%）
_ex = [g[2] * (g[4] ** 2 - g[3] ** 2) for g in C2_GROUPS]
C2_HOT_SHARE = _ex[2] / sum(_ex)
assert abs(C2_HOT_SHARE - 0.996) < 0.001, C2_HOT_SHARE
C2_HOT_SHARE_T = "99.6%"
C2_HOT_X = C2_GROUPS[2][4] / C2_GROUPS[2][3]                  # 高频行：纯数据并行 ÷ FSDP
C2_LOW_X = max(g[4] / g[3] for g in C2_GROUPS[:2])            # 低频两组最多远几倍
assert abs(C2_HOT_X - 4.8) < 0.05 and abs(C2_LOW_X - 1.07) < 0.005
C2_HOT_X_T, C2_LOW_X_T = "4.8", "1.07"
C2_NOEXC_HOT_X = C2_GROUPS[2][8] / C2_GROUPS[2][3]            # FSDP 只关省舍入，高频行远 35 倍
assert abs(C2_NOEXC_HOT_X - 35.3) < 0.1
C2_HOT_CNT_MED, C2_HOT_CNT_MAX = 74, 458   # 高频 39 个词在一张卡、一个微批里加几次（取出现最多的那张卡那个微批；主输入＋MTP 输入合计）
C2_HOT_CNT_MAIN = (37, 229)                # 单算主输入
# ⑤ 先加后传 vs 先传后加：每张卡每次跨卡的数据量（按 HLO 里的形状算，本课推导；线上实际字节数取决于通信算法，没测）
C2_AR_BYTES = C2_V * C2_D * 2              # 纯数据并行：整张表 BF16 all-reduce
C2_A2A_BYTES = C2_FSDP_N * C2_D_SHARD * 2  # FSDP：all-to-all 收齐 [32768, 128] BF16
assert abs(C2_AR_BYTES - 2.47e8) < 0.01e8 and abs(C2_A2A_BYTES - 8.4e6) < 0.05e6
C2_BYTES_X = C2_AR_BYTES / C2_A2A_BYTES
assert abs(C2_BYTES_X - 29.5) < 0.1
# ⑥ 单算子（v5p 真跑，下标取 golden 第 1 步真实 token，更新量随机 BF16；离 FP64 的 rel_L2）
_P = _C2["probe"]["cells"]
C2_SWEEP_N = [2048, 4096, 8192, 16384, 32768]
C2_SWEEP = {k: [_P[str(n)]["rel"][k] for n in C2_SWEEP_N]
            for k in ("bf16_unsorted_excess_on", "bf16_presorted_excess_on", "bf16_unsorted_excess_off", "f32_unsorted")}
assert ["%.3g" % v for v in C2_SWEEP["bf16_unsorted_excess_on"]] == ["0.00459", "0.00675", "0.00914", "0.00155", "0.00158"]
assert ["%.3g" % v for v in C2_SWEEP["bf16_presorted_excess_on"]] == ["0.00137", "0.00148", "0.00151", "0.00155", "0.00158"]
assert ["%.3g" % v for v in C2_SWEEP["bf16_unsorted_excess_off"]] == ["0.00459", "0.00675", "0.00914", "0.0129", "0.0176"]
_c4 = _P["4096"]["rel"]
C2_CELL4 = {"不排序·开": _c4["bf16_unsorted_excess_on"], "不排序·关": _c4["bf16_unsorted_excess_off"],
            "排序·开": _c4["bf16_presorted_excess_on"], "排序·关": _c4["bf16_presorted_excess_off"],
            "只排不标·开": _c4["bf16_presorted_nf_excess_on"], "FP32": _c4["f32_unsorted"]}
assert ["%.3g" % C2_CELL4[k] for k in ("不排序·开", "不排序·关", "排序·开", "排序·关", "只排不标·开")] == ["0.00675"] * 2 + ["0.00148"] + ["0.00675"] * 2
assert abs(C2_CELL4["FP32"] - 2.6e-9) < 0.05e-9
C2_BYCOUNT_ROWS = _P["4096"]["按次数分桶行数"]
assert C2_BYCOUNT_ROWS == {"1": 961, "2-9": 402, "10-99": 60, "100+": 3}
C2_BYCOUNT = {k: [_P["4096"]["按次数"][k][b] for b in ("1", "2-9", "10-99", "100+")]
              for k in ("bf16_unsorted_excess_on", "bf16_presorted_excess_on")}
assert ["%.3g" % v for v in C2_BYCOUNT["bf16_unsorted_excess_on"]] == ["0", "0.0027", "0.00781", "0.0146"]
assert ["%.3g" % v for v in C2_BYCOUNT["bf16_presorted_excess_on"]] == ["0", "0.00172", "0.00166", "0.0017"]
C2_K100_X = C2_BYCOUNT["bf16_unsorted_excess_on"][3] / C2_BYCOUNT["bf16_presorted_excess_on"][3]   # 约 8.5 倍
assert abs(C2_K100_X - 8.5) < 0.1
_S = _C2["probe"]["sharded"]   # 8 个设备照两种切法各跑一份（每卡 4096，全局 32768）
C2_SH = {k: _S[k] for k in ("fsdp_excess_on", "fsdp_excess_off", "dp_excess_on", "dp_jaxsort_excess_on")}
assert ["%.3g" % v for v in C2_SH.values()] == ["0.00158", "0.0176", "0.00717", "0.00286"]
C2_SORT_SHARE = (C2_SH["dp_excess_on"] - C2_SH["dp_jaxsort_excess_on"]) / (C2_SH["dp_excess_on"] - C2_SH["fsdp_excess_on"])
assert abs(C2_SORT_SHARE - 0.77) < 0.01     # 「排序一项走了约八成」（本课按单算子三个数复算：77%）
C2_SORT_SHARE_T = "约八成（%.0f%%）" % (C2_SORT_SHARE * 100)
# 门槛：对不同表形状二分出的最小插排序下标数（CPU 上为 v5p 交叉编译，看优化后 HLO 里有没有 sort）
C2_THRESH = [(t["V"], t["D"], t["最小插排序N"]) for t in _C2["threshold"]]
for _v, _d, _n in C2_THRESH:
    if _n is not None:
        assert _n == _v // 8 + 1, (_v, _d, _n)   # 最小的插排序 N ＝ V/8 ＋ 1，即「N > V/8」，和列数 D 无关
# 插排序的是 TPU 后端（libtpu）里的一道工序；工序名是内部名字，公开课程不写（2026-11-14 定）
C2_VPACK = 12288                            # 排序那颗 kernel 关掉省舍入后多出的舍到 BF16 的指令（静态条数，对齐记录）
C2_V7X_EMB_T = "1.50e-2 对 4.25e-3"           # v7x 纯数据并行：第 1 步词嵌入梯度对 v7x golden（机理没在 v7x 上单独测）
assert "%.2e" % _C2R["v7x纯数据并行8"]["emb_test"] == "1.50e-02" and "%.2e" % _C2R["v7x纯数据并行8"]["emb_golden"] == "4.25e-03"
# ⑨ 关省舍入：FSDP 也每次舍，比纯数据并行还差
_n = _C2R["关省舍入"]
C2_NOEXC_EMB_T = "%.2e" % _n["emb_test"]
assert C2_NOEXC_EMB_T == "1.09e-01" and abs(_n["r_max"] - 13.28) < 0.01
C2_NOEXC_EMB_T, C2_NOEXC_R_T = "1.09e-1", "13.3"
C2_NOEXC_X_G = _n["emb_test"] / C2_EMB_G         # 对 golden 27 倍（对齐记录的口径）
C2_NOEXC_X_R = _n["emb_R"]                        # 对 max(golden, GPU) 26.1 倍（定责口径，§4.2 用的是这个）
assert abs(C2_NOEXC_X_G - 27.2) < 0.1 and abs(C2_NOEXC_X_R - 26.1) < 0.05
# GPU：PyTorch CUDA 的词嵌入反向（aten/src/ATen/native/cuda/Embedding.cu，embedding_dense_backward_cuda）按下标数分两路，门槛 3072
C2_TORCH_GATE = 3072
C2_FSDP_SH_T, C2_DP_SH_T, C2_DPSORT_SH_T = ["%.2e" % C2_SH[k] for k in ("fsdp_excess_on", "dp_excess_on", "dp_jaxsort_excess_on")]
assert (C2_FSDP_SH_T, C2_DP_SH_T, C2_DPSORT_SH_T) == ("1.58e-03", "7.17e-03", "2.86e-03")
C2_FSDP_SH_T, C2_DP_SH_T, C2_DPSORT_SH_T = "1.58e-3", "7.17e-3", "2.86e-3"
C2_FP32_CELL_T = "2.6e-9"
assert abs(C2_CELL4["FP32"] - 2.6e-9) < 0.05e-9
# FP32 线性累加的上界（本课推导）：458 次 × 2⁻²⁴ ≈ 2.7e-5
assert abs(C2_HOT_CNT_MAX * 2 ** -24 - 2.7e-5) < 0.05e-5
C2_BYTES_ROUND = "30"                    # 2.47e8 ÷ 8.4e6 ≈ 29.5（本课推导，按 HLO 形状）
assert abs(C2_BYTES_X - 30) < 0.6

# ── 2026-11-11 · 案例二第二阶段：只把词嵌入表沿数据并行轴切开（第三种修法）──────────────────────────────
# 出处：对齐记录里那个案例的第二阶段一页；各运行训练步优化后 HLO 的集合通信汇总（导出在 topic08_case2.json 的 comm）；
#   结果文件（300 步对 golden、1000 步对 1000 步 golden）。只在 v5p、缩小版、纯数据并行 8 ＋ 累积 2 上验证过。
_CM = _C2["comm"]
_s3, _s3k = _C2R["只切词嵌入表"], _C2R["只切词嵌入表_1000"]
C3_EMB_T = "%.2e" % _s3["emb_test"]
assert C3_EMB_T == "3.97e-03" and abs(_s3["emb_R"] - 0.95) < 0.005
C3_EMB_T, C3_EMB_R_T = "3.97e-3", "0.95"
C3_R300_T, C3_R1K_T = "%.2f" % _s3["r_max"], "%.2f" % _s3k["r_max"]
assert (C3_R300_T, C3_R1K_T) == ("1.02", "1.15")
C3_ST_T = "%.1f%%" % (100 * _s3["step_time_vs_golden"])
assert C3_ST_T == "-3.3%"
C3_ST_T = C3_ST_T.replace("-", "−")
C3_STEP_S, C2_DP_STEP_S, C2_G_STEP_S = 0.478, 0.477, 0.494   # 每步中位数（第 11 步起），只切表 ／ 不修 ／ golden（对齐记录）
# 通信（每步每设备字节，按 HLO 形状 × 字节宽 × 每步次数；不是线上实测）
C3_COMM = {k: (_CM[k]["词嵌入每步字节"], _CM[k]["全部每步字节"]) for k in ("纯数据并行8", "修法1", "修法2", "FSDP8", "只切词嵌入表")}
assert ["%.2e/%.2e" % v for v in C3_COMM.values()] == ["9.90e+08/2.70e+09", "1.06e+09/2.77e+09", "9.90e+08/2.70e+09", "6.82e+07/1.32e+10", "6.82e+07/1.78e+09"]
C3_EMB_CUT = 1 - C3_COMM["只切词嵌入表"][0] / C3_COMM["纯数据并行8"][0]
C3_ALL_CUT = 1 - C3_COMM["只切词嵌入表"][1] / C3_COMM["纯数据并行8"][1]
assert abs(C3_EMB_CUT - 0.93) < 0.005 and abs(C3_ALL_CUT - 0.34) < 0.005
C3_EMB_CUT_T, C3_ALL_CUT_T = "93%", "34%"
C3_EMB_SHARE_T = "%d%%" % round(100 * C3_COMM["纯数据并行8"][0] / C3_COMM["纯数据并行8"][1])   # 词嵌入占纯数据并行全部通信
assert C3_EMB_SHARE_T == "37%"
C3_FIX1_PERMUTE = 67108864 + 131072      # 修法①在 scatter 前多出的 collective-permute（梯度 ＋ id），每步每设备
assert abs(C3_COMM["修法1"][0] - C3_COMM["纯数据并行8"][0] - C3_FIX1_PERMUTE) < 1
C3_FIX1_EMB_T, C3_FIX1_PERM_T = "1.06e9", "6.7e7"
C3_STATE = (4.41e9, 3.12e9)              # 参数与优化器状态，每设备（离线编译的 argument size；对齐记录）
assert abs((C3_STATE[0] - C3_STATE[1]) - C2_V * C2_D * 4 * 3 * 7 / 8.0) < 0.02e9   # 少掉的 1.30e9 ＝ 表的主权重＋两份动量从整份变 1/8（本课复算）
C3_HBM = (6.16e9, 4.46e9)                # 实测显存峰值，每设备：不修 ／ 只切表
C3_N_COMM = _CM["只切词嵌入表"]["条数"]    # 39 条集合通信
assert C3_N_COMM == 39
assert round(C3_COMM["纯数据并行8"][0] / C3_COMM["只切词嵌入表"][0]) == 15   # 每步连前向一起算：词嵌入通信约 1/15（本课复算，14.5）
C2_K100_UNS_T, C2_K100_SRT_T, C2_HOT_DP_T, C2_HOT_FSDP_T = "1.46e-2", "1.70e-3", "1.57e-2", "3.26e-3"
assert ("%.2e %.2e %.2e %.2e" % (C2_BYCOUNT["bf16_unsorted_excess_on"][3], C2_BYCOUNT["bf16_presorted_excess_on"][3], C2_GROUPS[2][4], C2_GROUPS[2][3])) == "1.46e-02 1.70e-03 1.57e-02 3.26e-03"
# 1.07 倍拆开（对齐记录 2026-11-11 补的按行分组结果：修法② ＝ 本卡 FP32、跨卡 BF16，修法① ＝ 全程 FP32）
#   跨卡那份 ≈ 修法② ÷ 修法①；本卡那份 ≈ 纯数据并行 ÷ 修法②（本课按同一文件复算）
_RG = _C2["rowgroups"]["runs"]
C2_SPLIT_B = (_RG["修法2"]["B"] / _RG["修法1"]["B"], _RG["纯数据并行8"]["B"] / _RG["修法2"]["B"])
C2_SPLIT_C = (_RG["修法2"]["C"] / _RG["修法1"]["C"], _RG["纯数据并行8"]["C"] / _RG["修法2"]["C"])
assert ("%.2f %.2f %.2f %.1f" % (C2_SPLIT_B + C2_SPLIT_C)) == "1.05 1.09 1.17 4.4", C2_SPLIT_B + C2_SPLIT_C
C2_SPLIT_T = "低频多卡那组：跨卡约 %.2f 倍、本卡约 %.2f 倍；高频行：跨卡 %.2f 倍、本卡 %.1f 倍" % (C2_SPLIT_B + C2_SPLIT_C)
assert abs(_RG["纯数据并行8"]["all"] / C2_EMB_G - 3.77) < 0.01      # 对 golden 是 3.77 倍；R 3.62 的分母是 GPU 中位数
C2_EMB_VS_G_T = "3.77"

# ── 2026-11-11 · 整体优化 E1：第一到五节跟上研究线的新结论（出处：对齐记录的根源一览、小尺寸对齐报告、TPU 自比影响逐件；实测，除标本课推导）──
RESELECT_PCT_T = "0.117%"                 # 「存下路由结果」被忽略时，反向重选的路由槽位（根源一览第 5 行）
BIAS_ZM_SM_AFTER = 0.17                   # 路由偏置零均值更新：TPU 关掉后带符号均值的 r（之前 19.1；最差项换成 MARE 时 0.31）
assert BIAS_ZM_R[0] == 19.1 and BIAS_ZM_R[1] == 0.31 and BIAS_ZM_SM_AFTER < BIAS_ZM_R[1]
B2_NOW = {"total": 205, "over": 177, "attr": 171, "far": 6, "far_mix": 5, "far_new": 3, "same3": 10, "within": 18}   # 现行配置（关 base2）第 1 步对 FP64（小尺寸报告「第一步整网」）
assert B2_NOW["attr"] + B2_NOW["far"] == B2_NOW["over"] and B2_NOW["over"] + B2_NOW["same3"] + B2_NOW["within"] == B2_NOW["total"]
OUTGATE_R_T = "0.90–1.00"                 # 注意力输出门 sigmoid：TPU 对 GPU 离 FP64 之比（定责，不处理；现行配置读数，旧的 0.96–0.98 是上一代）
import struct as _st
def _bf16(x):                             # 就近舍入到 BF16（本课推导用）
    b = _st.unpack(">I", _st.pack(">f", x))[0]
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return _st.unpack(">f", _st.pack(">I", b))[0]
LOG2E_BF16 = _bf16(1 / math.log(2))
assert LOG2E_BF16 == 1.4453125 and abs(LOG2E_BF16 * math.log(2) - 1.0018) < 0.0001      # logits 整体乘 κ ≈ 1.0018（本课复算，对上根源一览第 24 行）
HY3_SCALE_BF16 = _bf16(1 / math.sqrt(128))
assert HY3_SCALE_BF16 == 0.08837890625                                                  # 混元 3 的注意力缩放常数舍成 BF16（对齐报告偏差台账）
KD_SM300_T, KD_STEP1_T = "−5.1e-4", "−9.2e-4"   # 300 步那一代 KD 记录值带符号均值 ／ 第 1 步（根源一览第 13 行）
KD_GKL_R_T = "1.06 → 0.69"                # 改记广义 KL 后 1000 步 KD 带符号均值那一格的 r（根源一览第 25 行）
KD_GKL_WORST = "0.85"                     # 同一次比较三个统计里最差的判定量（协议变更记录），由待定变过
KD_GRAD_R64_T = "0.97–0.99"               # 索引器梯度对 FP64 的 R 中位数（同上）
GPU_KD_ROPE_SHARE_T = "六成"               # 真实宽度：GPU 的 KD 偏高约六成来自索引器 RoPE 在 BF16 里算（根源一览第 27 行：+5.0e-4 ÷ +8.1e-4–8.4e-4）
assert 0.59 < 5.0e-4 / 8.4e-4 and 5.0e-4 / 8.1e-4 < 0.63
SIG_VJP_R_T = "2.54 → 0.89"               # 精确 sigmoid／silu 的反向改成 GPU 的解析式：1000 步路由偏置均值的 r（根源一览第 18 行）
MINI_TPU_HIGH = 4.3e-6                    # 小尺寸：八份 TPU 主 loss 对 GPU 16 份逐步均值全部为正，平均 ＋4.3e-6（根源一览「待查」）
MINI_GPU_SPREAD = 2.63e-6                 # GPU 每份对其余 15 份的带符号均值最大 |.|（−2.63e-6 到 ＋2.34e-6）
assert MINI_TPU_HIGH > MINI_GPU_SPREAD
MINI_TPU_HIGH_T, MINI_GPU_SPREAD_T = "4.3e-6", "2.6e-6"
ELIDE_AFTER_MEASURED_T = "＋3.5e-6"        # 强制舍入后对 GPU 均值（协议变更记录，实测；原来误标「推算」）
assert abs(ELIDE_AFTER - 3.5e-6) < 1e-12
KAPPA_T = "1.0018"                        # base2 常数舍成 BF16 后 logits 的整体系数（本课复算见 LOG2E_BF16）

assert LONG_ARMS == tpuself()['piles']['overview'][1]

# ── 2026-11-11 晚 · F1：TPU 运行库 0.0.46 → 0.0.52（出处：对齐记录 TPU 自比「升 libtpu 0.0.52」一页、影响逐件、各 V 类结果文件；实测，除标本课推导）──
# 自比的基准运行和 115 种改动原来都在运行库 0.0.46 上；升 0.0.52 后：新建基准运行（同版本重跑逐位相同），六件重测。
LIB_OLD, LIB_NEW = "0.0.46", "0.0.52"
_g52, _g52k = _C2R["052_基准对046"], _C2R["052_基准对046_1000"]
LIB_GOLDEN_R = (_g52["r_max"], _g52k["r_max"])                       # 新基准对旧基准：300 步 0.99、1000 步 0.96，都在噪声内
assert ("%.2f %.2f" % LIB_GOLDEN_R) == "0.99 0.96"
LIB_GOLDEN_STEP = _g52["step_time_vs_golden"]                         # 步时 −5.5%（0.466 s 对 0.494 s）
assert abs(LIB_GOLDEN_STEP + 0.0553) < 0.0002 and abs(0.466 / 0.494 - 1 + 0.057) < 0.002
LIB_GOLDEN_R_T = "%.2f ／ %.2f" % LIB_GOLDEN_R
# 六件：每件只改那一样，各对同版本的基准运行比；300 步曲线最差 r（1000 步写在括号里），第 1 步领头张量的 R
def _r(k):
    return _C2R[k]["r_max"]
LIB_SIX = [   # (件, 0.0.46 的 r, 0.0.52 的 r, 0.0.46 判, 0.0.52 判)
    ("纯数据并行（不修）", _r("纯数据并行8"), _r("052_纯数据并行8"), "超", "超"),
    ("关省舍入", _r("关省舍入"), _r("052_关省舍入"), "超", "超"),
    ("关小矩阵乘改写", 0.94, 1.00, "能忍", "能忍"),            # 这两件 r 来自脾气表（topic08_tpuself.json 的 issues，下面断言）
    ("RMSNorm 多舍一次", _r("RMSNorm多舍一次"), _r("052_RMSNorm多舍一次"), "能忍", "超"),
    ("多路残差不强制舍 BF16", 1.09, 1.18, "能忍", "能忍"),
    ("上下文并行（不开 FP32 logits）", 1.27, 0.95, "能忍", "在噪声内"),
]
_iss = {i["name"]: dict(i["exps"]) for i in tpuself()["piles"]["issues"]}
assert _iss["关小矩阵乘改写"]["关小矩阵乘改写"] == 0.94 and _iss["关小矩阵乘改写"]["TPU 运行库 0.0.52：关小矩阵乘改写"] == 1.0
assert _iss["多路残差混合 不强制舍成 BF16"]["TPU 运行库 0.0.52：多路残差混合 不强制舍成 BF16"] == 1.18
assert _iss["RMSNorm 多舍一次"]["TPU 运行库 0.0.52：RMSNorm 多舍一次"] == 5.5
assert [("%.2f" % a, "%.2f" % b) for _n, a, b, _x, _y in LIB_SIX] == [("2.62", "2.82"), ("13.28", "12.56"), ("0.94", "1.00"), ("1.06", "5.50"), ("1.09", "1.18"), ("1.27", "0.95")]
_cp52 = [a for a in tpuself()["arms"] if a["change"] == "TPU 运行库 0.0.52：加上下文并行（不做负载均衡）"]
assert _cp52 and _cp52[0]["r300"] == 0.95
LIB_SIX_R = {"纯数据并行": ("3.62", "3.60"), "关省舍入": ("26.13", "26.11")}   # 第 1 步词嵌入梯度 R
assert "%.2f" % _C2R["052_纯数据并行8"]["emb_R"] == "3.60" and "%.2f" % _C2R["052_关省舍入"]["emb_R"] == "26.11"
# RMSNorm 多舍一次：0.0.52 上主 loss 系统性偏高
LIB_RMS_R1K = _r("052_RMSNorm多舍一次_1000")
assert "%.2f" % LIB_RMS_R1K == "3.68"
LIB_RMS_SM = (1.3e-6, _C2R["052_RMSNorm多舍一次"]["loss_worst"][1])   # 前 300 步主 loss 对基准的带符号均值：0.0.46 1.3e-6（影响逐件），0.0.52 1.65e-5
assert _C2R["052_RMSNorm多舍一次"]["loss_worst"][0] == "signed_mean" and "%.3g" % LIB_RMS_SM[1] == "1.66e-05"
LIB_RMS_SM_T = ("1.3e-6", "1.65e-5")        # 研究线写 1.65e-5（结果文件 1.6551e-5，按三位有效是 1.66e-5；照研究线）
LIB_RMS_R_STEP1 = ("1.20", "1.17")          # 第 0 层 input_layernorm 权重梯度的 R（第 1 步看不出）
LIB_RMS_R_T, LIB_RMS_R1K_T = "%.2f" % _r("052_RMSNorm多舍一次"), "%.2f" % LIB_RMS_R1K
# 三种修法在 0.0.52 上（300 ／ 1000 步），都跟 GPU 重跑分不开
LIB_FIX = {k: (_r("052_" + k), _r("052_" + k + "_1000")) for k in ("修法1", "修法2", "只切词嵌入表")}
assert ["%.2f/%.2f" % v for v in LIB_FIX.values()] == ["1.55/0.92", "1.20/1.28", "1.23/0.97"]
assert max(v[0] for v in LIB_FIX.values()) < LOO8_300_MAX and max(v[1] for v in LIB_FIX.values()) < LOO16_1K_MAX   # 300 步对 8 份留一法最高、1000 步对 16 份
LIB_FIX_T = "；".join("%s %.2f ／ %.2f" % (n, a, b) for n, (a, b) in zip(("①", "②", "③"), LIB_FIX.values()))
# v7x 纯数据并行、不累积（每卡 8192 个下标）：0.0.52 开始排序 → 不再丢精度（先离线预测，再上卡核）
_v46, _v52 = _C2R["v7x纯数据并行8不累积"], _C2R["052_v7x纯数据并行8不累积"]
LIB_V7X_GA1 = {"r": (_v46["r_max"], _v52["r_max"]), "emb": (_v46["emb_test"], _v52["emb_test"]), "R": (_v46["emb_R"], _v52["emb_R"])}
assert "%.2f %.2f %.2e %.2e %.2f %.2f" % (LIB_V7X_GA1["r"] + LIB_V7X_GA1["emb"] + LIB_V7X_GA1["R"]) == "4.06 0.89 2.93e-02 4.63e-03 6.90 1.10"
LIB_V7X_GA1_T = "r %.2f → %.2f，词嵌入梯度离 FP64 %.2e → %.2e，R %.2f → %.2f" % (LIB_V7X_GA1["r"] + LIB_V7X_GA1["emb"] + LIB_V7X_GA1["R"])
# 门槛：四种组合（芯片 × 运行库）二分出的最小插排序下标数（单算子离线编译；None ＝ 到 65536 都不插）
LIB_THRESH = {k: {(t["V"], t["D"]): t["最小插排序N"] for t in v} for k, v in _C2["threshold_versions"].items()}
assert LIB_THRESH["v5p_046"][(120832, 1024)] == 15105 and LIB_THRESH["v5p_052"][(120832, 1024)] == 14420
assert LIB_THRESH["v5p_052"][(120832, 128)] == 1783 and LIB_THRESH["v7x_052"][(120832, 1024)] == 7181
assert all(d != 128 for (_v, d) in LIB_THRESH["v7x_046"]) and all(d != 128 for (_v, d) in LIB_THRESH["v7x_052"])   # v7x 只列 1024 以上的列宽（窄列另算，不公开）
assert all(LIB_THRESH[k][(1000, 1024)] is None for k in LIB_THRESH) and LIB_THRESH["v5p_052"][(30000, 128)] is None
for (_v, _d), _n in LIB_THRESH["v5p_046"].items():
    assert _n is None or _n == _v // 8 + 1                                                    # 0.0.46 上 v5p 全部是 N ＞ V／8
for (_v, _d), _n in LIB_THRESH["v7x_046"].items():
    assert _n is None or _n == _v // 8 + 1                                                    # 0.0.46 上 v7x：列出的形状也是 V／8
assert LIB_THRESH["v7x_052"][(120832, 1024)] < 8192 < LIB_THRESH["v5p_052"][(120832, 1024)]   # v7x 0.0.52 上每卡 8192 过线、v5p 不过
assert 4096 < LIB_THRESH["v7x_052"][(120832, 1024)]                                          # 累积 2（每卡 4096）在 v7x 0.0.52 上仍不过
LIB_SCREEN_CHANGED = 0          # 104 个能比的臂，两个版本下「程序相同／不同」的判定一条没变（25 相同、79 不同）
LIB_SCREEN = (104, 25, 79)
assert LIB_SCREEN[1] + LIB_SCREEN[2] == LIB_SCREEN[0]
LIB_SPOT = 3                    # 0.0.52 上抽 3 条「程序相同」的改动上卡，都跟新基准逐位相同

# ── 2026-11-11 晚 · F1：跑前离线筛（出处：TPU 自比「跑前离线筛：回测」表、setup「跑前离线筛」；实测）──
SCREEN = {"arms": 109, "same": 23, "diff": 72, "cant": 14, "match": 95, "mismatch": 0}
assert SCREEN["same"] + SCREEN["diff"] + SCREEN["cant"] == SCREEN["arms"] and SCREEN["same"] + SCREEN["diff"] == SCREEN["match"] + SCREEN["mismatch"]
SCREEN_SPECS, SCREEN_SEC_T, SCREEN_MED_S, SCREEN_GIB_T = 95, "114–265", 184, "13.2–16.9"   # 编了 95 份规格；整次 114–265 s、中位数 184 s；单进程峰值内存

# ── 2026-11-11 晚 · F1：小尺寸主 loss 共同偏高，偏差随步数怎么走（出处：对齐记录里小尺寸主 loss 偏高的分析；实测）──
MD = tpuself()["mini_drift"]
assert MD["n_tpu"] == 8 and MD["n_gpu"] == 16 and len(MD["主loss"]["blocks"]) == 40
_lmw = {w[0]: w for w in MD["主loss"]["windows"]}
_mtw = {w[0]: w for w in MD["MTP"]["windows"]}
MD_STEP1 = _lmw["1-1"][1]                 # 第 1 步 −2.0e-6（8 份都偏低）
MD_BUMP = _lmw["51-100"][1]               # 第 51–100 步的小鼓包 ＋4.5e-6
MD_LATE = _lmw["751-1000"][1]             # 第 751–1000 步 ＋1.3e-5
MD_ALL = _lmw["2-1000"][1]                # 第 2–1000 步 ＋4.3e-6
MD_MTP_LATE = _mtw["751-1000"][1]         # 同期 MTP −3.2e-5（反着走）
assert "%.1e %.1e %.1e %.1e %.1e" % (MD_STEP1, MD_BUMP, MD_LATE, MD_ALL, MD_MTP_LATE) == "-2.0e-06 4.5e-06 1.3e-05 4.3e-06 -3.2e-05"
assert _lmw["751-1000"][3] == 8 and _mtw["751-1000"][3] == 0       # 后段 8 份全部偏高，MTP 8 份全部偏低
_blk = {b[0]: b for b in MD["主loss"]["blocks"]}
MD_LAST = _blk["976-1000"][1]
assert "%.1e" % MD_LAST == "1.8e-05" and "%.1e" % _blk["601-625"][1] == "4.1e-06"
MD_CORR = (MD["corr"]["param_norm_501起"], MD["corr"]["梯度范数倒数_501起"])
assert "%.2f %.2f" % (MD["corr"]["步数_501起"], MD_CORR[1]) == "0.96 -0.11"
MD_SAME_STATE_T = "＋0.83e-6 ± 1.15e-6"   # 同一份权重上的前向：TPU 减 GPU 写法（6 个状态 19 批，均值 ± 标准误），不到后段的一成
MD_STEP1_T, MD_LATE_T, MD_LAST_T, MD_MTP_LATE_T = "−2.0e-6", "＋1.3e-5", "＋1.8e-5", "−3.2e-5"
# 课件正文用的字符串（F1）
LIB_V7X_R46_T = "%.2f" % LIB_V7X_GA1["r"][0]
LIB_TH_V5P_1024, LIB_TH_V5P_128, LIB_TH_V7X_1024 = (format(LIB_THRESH["v5p_052"][(120832, 1024)], ","), format(LIB_THRESH["v5p_052"][(120832, 128)], ","),
                                                    format(LIB_THRESH["v7x_052"][(120832, 1024)], ","))
LIB_GOLDEN_STEP_T = "%.1f%%" % (-LIB_GOLDEN_STEP * 100)
assert LIB_GOLDEN_STEP_T == "5.5%"
LIB_RMS_PAIR_T = "%.2f → %s" % (_r("RMSNorm多舍一次"), LIB_RMS_R_T)
assert LIB_RMS_PAIR_T == "1.06 → 5.50"
LIB_SCREEN_N = LIB_SCREEN[0]
SCREEN_ARMS, SCREEN_SAME, SCREEN_DIFF, SCREEN_CANT, SCREEN_MATCH = (SCREEN[k] for k in ("arms", "same", "diff", "cant", "match"))
SCREEN_MIN_T = "%d" % round(SCREEN_MED_S / 60)       # 整次中位数 184 s ≈ 3 分钟
assert SCREEN_MIN_T == "3"
LIB_DP8_52_T = "%.2f" % _r("052_纯数据并行8")
LIB_V7X_R52_T = "%.2f" % LIB_V7X_GA1["r"][1]
LIB_V7X_R_PAIR_T = "%.2f → %.2f" % LIB_V7X_GA1["R"]
assert (LIB_DP8_52_T, LIB_V7X_R52_T, LIB_V7X_R_PAIR_T) == ("2.82", "0.89", "6.90 → 1.10")

# ── 2026-11-11 晚 · F2：从小到大三级，第 1 步主 loss 两边各自对 CPU FP64（出处：对齐记录 小尺寸知识库、中等尺寸对齐报告「第 1 步主 loss 对 CPU FP64」、真实宽度 README；实测）──
LAD_MINI_GPU_RANGE = (-2.46e-6, 4.7e-7)   # 小模型 GPU 8 份
LAD_MINI_TPU = -3.68e-6                    # 小模型 TPU（现行关 base2；开 base2 −2.14e-6）
LAD_MID = (1.42e-5, 2.4e-7)                # 中等尺寸（真实宽度与注意力、随机权重）：GPU、TPU，相对差（研究线 11-11 起统一写相对差；旧的绝对差是 1.68e-4／2.8e-6）
LAD_REAL_GPU = 5.3e-5                      # 真实宽度截层 GPU 4 份（＋5.1e-5 到 ＋5.8e-5）
LAD_REAL_TPU = (-9.87e-5, -1.24e-6)        # 真实宽度截层 TPU：开 base2 → 只关 base2
LAD_REAL_GPU_SHARE = 0.55                  # GPU 那份偏高在 FP64 里逐类加 BF16 舍入，简单相加能复现约 55%
assert LAD_MID[0] / LAD_MID[1] > 50 and LAD_MID[0] < LAD_REAL_GPU and abs(LAD_REAL_TPU[1]) < abs(LAD_MINI_TPU)
LAD_MID_T, LAD_REAL_GPU_T = ("＋1.42e-5", "＋2.4e-7"), "＋5.3e-5"
LAD_MID_SPREAD = 9.4e-6                    # 中等尺寸 GPU 8 份第 1 步的极差（相对）
assert LAD_MID[0] > LAD_MID_SPREAD > max(abs(x) for x in LAD_MINI_GPU_RANGE)
# 隔离实验（研究线 11-11 晚，小尺寸知识库「隔离实验」行，实测）：两侧把 MTP、索引器 KD 的系数设 0，第 551–1000 步 TPU 对同配置 GPU 4 份均值
MD_ISO = {"对照": 1.48e-5, "关MTP": 1.16e-5, "关KD": 9.0e-6, "都关": 8.7e-6}
assert all(v > 0 for v in MD_ISO.values()) and abs(MD_ISO["都关"] / MD_ISO["对照"] - 0.6) < 0.05   # 约六成
MD_ISO_T = "对照 ＋1.5e-5，关 MTP ＋1.2e-5，关 KD ＋9.0e-6，两个都关 ＋8.7e-6"
SCREEN_DIFF_BITWISE = 5                    # 判程序不同、上卡后结果仍逐位的（多跑一次，不漏）
SCREEN_CANT_KINDS = {"SparseCore 开关": 6, "换运行库": 5, "换芯片代际": 3}
assert sum(SCREEN_CANT_KINDS.values()) == SCREEN["cant"]

# ── 2026-11-11 晚 · F6：RMSNorm 多舍一次在运行库 0.0.52 上为什么变强（出处：研究线影响逐件「RMSNorm 多舍一次」、案例 15、各结果文件；实测，除标推导）──
_RM = _C2["rms052"]
RMS_AB = [   # (标签, 版本, 其余各层, 最后一层, 300 步曲线最差 r)
    ("E6", "0.0.46", "默认写法", "默认写法（编译器只舍一次）", _r("RMSNorm多舍一次")),
    ("E6d", "0.0.46", "单次舍入", "强制舍两次", _r("RMSNorm最后一层强制两次")),
    ("V7", "0.0.52", "默认写法", "默认写法（编译器真舍两次）", _r("052_RMSNorm多舍一次")),
    ("V20", "0.0.52", "默认写法", "单次舍入", _r("052_RMSNorm只最后一层改回一次")),
    ("V19", "0.0.52", "单次舍入", "强制舍两次", _r("052_RMSNorm最后一层强制两次")),
]
assert ["%.2f" % x[4] for x in RMS_AB] == ["1.06", "6.83", "5.50", "0.84", "6.97"], [x[4] for x in RMS_AB]
assert _RM["写回内存的归一化值"] == {"046默认": 0, "052只最后一层改回一次": 0, "052默认": 1, "v7x052默认": 0}
RMS_V7X_R = _r("052_v7x_RMSNorm多舍一次")                     # v7x 0.0.52 默认写法：没单独写回，1.22 略超、跟 GPU 重跑分不开
assert "%.2f" % RMS_V7X_R == "1.22"
RMS_FROZEN = _RM["同状态前向"]["r52"]["rel_mean"]               # 同一份权重上前向：默认写法 − 单次舍入，16 批平均 ＋3.23e-5
assert "%.2e" % RMS_FROZEN == "3.23e-05" and abs(_RM["同状态前向"]["r46"]["rel_mean"]) < 1e-9
RMS_FP64 = _RM["fp64_summary"]["double_minus_single_rel"]       # CPU FP64 模拟第 301 步：＋3.17e-5
RMS_FP64_S1 = _RM["fp64_step1_summary"]["double_minus_single_rel"]   # 第 1 步：＋2.5e-7
RMS_SHRINK = -_RM["fp64_summary"]["shrink_mean"]                # 输出头的输入整体缩小 1.49e-4
RMS_SCALE_SHARE = _RM["fp64_summary"]["scale_only_share"]       # 只按整体缩放就解释 100.3%
assert ("%.2e %.1e %.2e %.3f" % (RMS_FP64, RMS_FP64_S1, RMS_SHRINK, RMS_SCALE_SHARE)) == "3.17e-05 2.5e-07 1.49e-04 1.003"
RMS_GAMMA = (_RM["norm_scale"]["mean"], _RM["norm_scale"]["min"], _RM["norm_scale"]["max"])   # 最后一层 γ：均值 1.001，范围 0.86–1.18
assert "%.3f %.2f %.2f" % RMS_GAMMA == "1.001 0.86 1.18"
RMS_SWEEP = sorted({round(w, 8): c for w, c in _RM["scale_sweep"]}.items())     # 单个 γ 下两次舍入让 |h| 平均变多少（FP64 里 400 万个标准正态数）
assert dict(RMS_SWEEP)[1.0] == 0.0 and abs(dict(RMS_SWEEP)[0.99609375] + 1.71e-3) < 1e-5
RMS_FROZEN_T, RMS_FP64_T, RMS_SHRINK_T = "＋3.23e-5", "＋3.17e-5", "1.49e-4"
# 生产形状（词表 120832、隐藏维 6144）上单算子离线编译的门槛（研究线案例二「在自己的模型上怎么做」，实测）
_PS = _C2["prod_shape"]
PROD_TH = {k: _PS["%s|6144|%s" % (v, g)]["门槛"] for k, (v, g) in {"v5p_046": ("0.0.46", "v5p"), "v5p_052": ("0.0.52", "v5p"), "v7x_046": ("0.0.46", "v7x"), "v7x_052": ("0.0.52", "v7x")}.items()}
assert PROD_TH == {"v5p_046": 15105, "v5p_052": None, "v7x_046": 15105, "v7x_052": 43648} and _PS["0.0.52|6144|v5p"]["查到"] == 65536
assert not any(_PS["0.0.52|6144|%s" % g]["每卡"][n] for g in ("v5p", "v7x") for n in ("4096", "8192", "16384"))   # 0.0.52 上每卡 4096–16384 两代都不排序
PROD_TH_V7X_052_T = str(PROD_TH["v7x_052"])

# 混元 3 换基线重判（11-11 晚，混元 3 对齐报告「基线」一段，实测）：⑤ 的十份每份当一次基线，四个配置各判一遍
REBASE_SHIFT_MAX, REBASE_V7X_PEND = 0.47, 3      # r 在十份基线之间最多相差 0.47；v7x 当前配置有 3 份基线落进待定（最差 1.11），没有不过
REBASE_WORST_T = "1.11"

# ── 2026-11-11 晚 · F10：小尺寸主 loss 后段「补流梯度舍入」的读数（实测）。⛔ 对齐记录随后撤回了「根因」：两处都补后尾巴又出现，在查（F11）──
MD_FIX = {"对照": 1.48e-5, "补上舍入": 1.9e-6, "GPU尺子": 2.1e-6, "只补流梯度": 0.6e-6, "只补分支输出": 3.0e-6}   # 第 551–1000 步主 loss 对 GPU 16 份均值
assert MD_FIX["补上舍入"] < MD_FIX["GPU尺子"] < MD_FIX["对照"]
MD_NOIHC = (1.9e-6, 0.7e-6)              # 两边都去掉多路残差（换普通残差），TPU 2 份对 GPU 3 份均值，后段不飘
MD_FIX_T = "＋1.48e-5 → ＋1.9e-6（GPU 尺子 2.1e-6）"
MI = tpuself()["mini_ihc"]
assert "%.2e" % MI["对照"]["tpu_mean"] == "1.48e-05" and "%.1e" % MI["补上流的舍入"]["tpu_mean"] == "1.9e-06" and len(MI["补上流的舍入"]["tpu_each"]) == 5
assert "%.1e %.1e" % (MI["只补流的梯度"]["tpu_mean"], MI["只补分支输出"]["tpu_mean"]) == "5.8e-07 3.0e-06"
assert ["%.1e" % x for x in MI["两边都去掉多路残差"]["tpu_each"]] == ["1.9e-06", "6.9e-07"]
MI_SD = MI["对照"]["gpu_loo_sd"]          # GPU 16 份留一法 sd 2.1e-6
assert "%.1e" % MI_SD == "2.1e-06"

# ── 2026-11-12 · F19：后段主因定到流梯度舍法（两头验证）；对齐配置改 C 口径（topic08_fixcmp.json：曲线由私有导出脚本读各运行逐步记录；
#    两头验证的每份数由同一脚本从对齐记录三份单因素交叉结果文件抄来；实测）──
with open(os.path.join(HERE, "topic08_fixcmp.json"), encoding="utf-8") as _fh:
    FC = json.load(_fh)
FC_W = {k: [r["w551"] for r in v] for k, v in FC["cols"].items()}
TW = FC["twoway"]
assert [len(FC_W[k]) for k in ("修之前", "补流梯度舍法", "对齐配置")] == [4, 5, 6]
for _k in FC_W:                                                    # 运行目录算的数跟结果文件逐份一致
    assert all(abs(a - b) < 1e-12 for a, b in zip(FC_W[_k], TW["tpu"]["对照" if _k == "修之前" else _k]["w551"])), _k
FC_SD = FC["loo_sd_551_1000"]
assert abs(FC_SD - MI_SD) < 1e-9                                   # 跟研究线消融结果文件的 GPU 留一法 sd 一致
TAIL_TH = TW["tail_threshold"]                                     # 「尾部在」＝ 第 751–1000 步均值 ＞ 3e-6（对齐记录的判据）
FC_TAIL = {k: sum(1 for r in v if r["w751"] > TAIL_TH) for k, v in FC["cols"].items()}
assert TAIL_TH == 3e-6 and FC_TAIL == {"修之前": 4, "补流梯度舍法": 2, "对齐配置": 5}, FC_TAIL
assert FC_TAIL["修之前"] == TW["tpu"]["对照"]["n_tail"] and FC_TAIL["补流梯度舍法"] == TW["tpu"]["补流梯度舍法"]["n_tail"] and FC_TAIL["对齐配置"] == TW["tpu"]["对齐配置"]["n_tail"]
FC_MEAN = {k: sum(v) / len(v) for k, v in FC_W.items()}
assert ["%.1e" % FC_MEAN[k] for k in ("修之前", "补流梯度舍法", "对齐配置")] == ["9.4e-06", "1.9e-06", "6.9e-06"]   # 对齐配置 6 份 6.948e-6（对齐记录写 ＋7.0e-6，舍入口径不同）
TW_P = {"tpu": TW["tpu"]["补流梯度舍法"]["p"], "gpu": TW["gpu"]["GPU 换流梯度舍法"]["p"], "head": TW["gpu"]["GPU 换收拢处舍法"]["p"], "c_m58": TW["c_vs_m58_p"]}
assert ["%.2g" % TW_P[k] for k in ("tpu", "gpu", "head", "c_m58")] == ["0.016", "0.0025", "0.89", "0.024"]
TW_GPU_MEAN = TW["gpu"]["GPU 换流梯度舍法"]["mean"]
assert "%.1e" % TW_GPU_MEAN == "3.9e-06" and all(x > 0 for x in TW["gpu"]["GPU 换流梯度舍法"]["w551"])   # 4 份都正
FC_MEAN_T = {k: ("%+.1e" % v).replace("e-0", "e-") for k, v in FC_MEAN.items()}
TW_P_T = {k: "%.2g" % v for k, v in TW_P.items()}                # 两位有效数字（GPU 那格 0.00248 → 0.0025）
C_NOW = {"pass": 8, "pend": 1, "loss_r": 0.98, "rep_pend": 1, "rep_pend_r": 1.14, "n_rep": 6}   # 现行比较（C 口径第一份）1000 步：过 8、待定 1，主 loss MARE r 0.98；6 份里 1 份主 loss 待定 1.14（小尺寸对齐报告、6 份重复结果文件）
assert C_NOW["pass"] + C_NOW["pend"] == 9 and C_NOW["loss_r"] <= 1 < C_NOW["rep_pend_r"] <= 2

# ── 2026-11-12 · F13：生产配方臂补到同配方 GPU 8 份当尺子重判（对齐记录同配方尺子重判的结果文件）──
with open(os.path.join(HERE, "topic08_prodrecipe.json"), encoding="utf-8") as _fh:
    PR = json.load(_fh)["checks"]
assert all(c["status_own"] == "pass" for c in PR.values()) and len(PR) == 9      # 同配方尺子：9 项全过
PR_WORST = max(c["r_own"] for c in PR.values())
assert "%.2f" % PR_WORST == "1.00"                                                # 最差一项（索引器蒸馏 loss）
PR_LOSS_X = PR["L4.loss"]["ratio"]["signed_mean"]
assert "%.1f" % PR_LOSS_X == "4.8"                                                # 主 loss 带符号均值：同配方尺子比现行宽几倍
assert ["%.2f" % PR[k]["r_borrowed"] for k in ("L4.loss", "L4.grad_norm")] == ["2.97", "2.45"]   # 借现行尺子时这两项判超
PR_N_T, PR_WORST_T, PR_LOSS_X_T = "16", "%.3f" % PR_WORST, "%.1f" % PR_LOSS_X   # 11-12 事先登记补到 16 份（两两 120 对）：九项全过、最差 0.996 不变，主 loss 带符号均值的上限由同一对决定、16 份与 8 份相同（对齐记录生产配方臂 8 份、16 份两个结果文件）
# F14：生产形状 0.0.46 的门槛（两代都是 V/8 往上取整）——每卡 16384 个下标在 0.0.46 上会排序，0.0.52 上不排序
assert PROD_TH["v5p_046"] == PROD_TH["v7x_046"] == 120832 // 8 + 1 and 16384 > PROD_TH["v5p_046"]
PROD_TH_046_T = str(PROD_TH["v5p_046"])
# F16：GPU 第 1 步参照补到 6 份 15 对后重判（小尺寸报告「第一步整网」、结果文件 judge_fp64：超 N 169、待定 2 个、判「待定」）
B2_NOW6 = {"total": 205, "over": 169, "attr": 167, "far": 2, "far_mix": 1, "same": 10, "within": 26}
assert B2_NOW6["attr"] + B2_NOW6["far"] == B2_NOW6["over"] and B2_NOW6["over"] + B2_NOW6["same"] + B2_NOW6["within"] == B2_NOW6["total"]
assert B2_NOW6["over"] < B2_NOW["over"] and B2_NOW6["far_mix"] < B2_NOW["far_mix"]      # 份数补够，超线和没定责的都变少
# F18：换基线挪动分两层（结果文件逐配置复算）：单条检查最多挪 0.47（v7x 按专家切的 param_norm）；四条取最差后最多挪 0.33（v7x FSDP 8：0.78–1.11）
REBASE_COMB_SHIFT = 0.33
assert REBASE_COMB_SHIFT < REBASE_SHIFT_MAX
# GPU 只换收拢处舍法：后段不变，第 51–100 步冒出前段鼓包（对 GPU 自己 p 0.001）——跟 TPU 不补收拢处舍入时的前段鼓包两头对上
TW_HEAD51 = TW["gpu"]["GPU 换收拢处舍法"]["mean51"]
TW_HEAD51_P = TW["gpu"]["GPU 换收拢处舍法"]["p51"]
assert "%.1e" % TW_HEAD51 == "3.7e-06" and TW_HEAD51_P < 0.002     # 结果文件 3.745e-6（对齐记录知识库写 ＋3.8e-6，舍入口径不同）
TW_HEAD51_T, TW_HEAD51_P_T = ("%+.1e" % TW_HEAD51).replace("e-0", "e-").replace("+", "＋"), "%.3f" % TW_HEAD51_P
C_LOSS_R_T = "%.2f" % C_NOW["loss_r"]
FC_C_MEAN_T = FC_MEAN_T["对齐配置"].replace("+", "＋")
# 只补流梯度舍法（不补收拢处）的取舍：后段平了，前段第 51–100 步的小鼓包还在；对齐配置补上收拢处后鼓包消失
FC_BUMP = {k: TW["tpu"][k]["mean51"] for k in ("对照", "补流梯度舍法", "对齐配置")}
assert FC_BUMP["补流梯度舍法"] > FC_BUMP["对照"] > 0 > FC_BUMP["对齐配置"] and TW["tpu"]["对齐配置"]["p51"] < 0.01 if "p51" in TW["tpu"]["对齐配置"] else True
FC_BUMP_T = {k: ("%+.1e" % v).replace("e-0", "e-").replace("+", "＋").replace("-", "−", 1) if v < 0 else ("%+.1e" % v).replace("e-0", "e-").replace("+", "＋") for k, v in FC_BUMP.items()}
# F21：换 v7x（以及 einsum 换写法）第 1 步多路残差收拢处三个小参数离 FP64 远、被豁免盖住 —— 对齐记录 TPU 自比矩阵「规则还有一类漏判」（实测）
V7X_HCHEAD_FP64 = (1.25, 1.30)            # 三个张量第 1 步离 FP64（相对）；golden 0.03–0.07
V7X_HCHEAD_GOLDEN = (0.03, 0.07)
V7X_HCHEAD_RMAX = 46.2
assert V7X_HCHEAD_FP64[0] / V7X_HCHEAD_GOLDEN[1] > 15 and V7X_HCHEAD_RMAX > 1.2
V7X_HCHEAD_T = "离 FP64 %.2f–%.2f（golden %.2f–%.2f），最大 R %.1f" % (V7X_HCHEAD_FP64 + V7X_HCHEAD_GOLDEN + (V7X_HCHEAD_RMAX,))
# F22：0.0.46 上 RMSNorm 多舍一次「分得开」只靠第 1 步第 0 层 input_layernorm 权重一个压线张量（TPU 自比第 1 步按 6 份 GPU 参照重判的结果文件）
RMS_EDGE_R = (1.2024758939024367, 1.1955309333947115)   # GPU 第 1 步参照 3 份 ／ 6 份时的 R
assert RMS_EDGE_R[0] > 1.2 > RMS_EDGE_R[1]
RMS_EDGE_T = "%.4f" % RMS_EDGE_R[0], "%.4f" % RMS_EDGE_R[1]
RMS_EDGE3_T, RMS_EDGE6_T = RMS_EDGE_T
C_REP_PEND_T = "%.2f" % C_NOW["rep_pend_r"]
# F24：对齐配置残余的两组排查（结果文件）：关 MTP 仍偏（4 份）；GPU 在只改流梯度之上再让分支输出不舍，不放大尾部
RES = TW["residual"]
assert "%.1e" % RES["c_nomtp_551"] == "7.5e-06" and RES["c_nomtp_n"] == 4
assert "%.1e" % RES["gpu_stream_branch"] == "2.3e-06" and RES["gpu_stream_branch"] < RES["gpu_stream_only"]
RES_NOMTP_T = ("%+.1e" % RES["c_nomtp_551"]).replace("e-0", "e-").replace("+", "＋")
RES_BRANCH_T = ("%+.1e" % RES["gpu_stream_branch"]).replace("e-0", "e-").replace("+", "＋")
# F24：第 1 步在新对齐配置上重判（GPU 参照 6 份 15 对；结果文件 judge_fp64 与逐张量行）
B2_C_OVER = 163
_hcfn = (0.01591170578999582 / 0.013978521350485136, 1.4260719562589834)   # 第 3 层 FFN 前 hc_fn：超 N 倍数、R
assert "%.2f %.2f" % _hcfn == "1.14 1.43"
B2_C_UNATTR_T = "超出波动 %.2f 倍，R %.2f" % _hcfn
# F26：对齐配置对修之前：p 0.32，两格还分不开（对齐记录根源一览第 30 行；单因素交叉 6 份版结果文件）
TW_P_CCTL = TW["c_vs_ctl_p"]
assert "%.2f" % TW_P_CCTL == "0.32"
TW_P_CCTL_T = "%.2f" % TW_P_CCTL
# F27：对齐配置 6 份各自的待定项（6 份重复结果文件，各项最差 r）：没有不过；现行那份路由偏置均值 1.31，另一份主 loss 1.14、MTP 1.07，一份梯度范数 1.04
REP6_PEND = {"router_bias_mean": 1.31, "loss": 1.14, "mtp": 1.07, "grad_norm": 1.04}
assert REP6_PEND["loss"] == C_NOW["rep_pend_r"] and max(REP6_PEND.values()) <= 2
# F29：收拢处舍入一开，补流梯度舍法（全套或任一半）测不出作用（单因素交叉两份结果文件；实测）。数据键沿用导出文件里的名字
MG = FC["m52gate"]   # 导出文件里的键名；课程里叫「收拢处开着时」
assert [len(MG[k]["w551"]) for k in ("M52", "C", "C_stream", "C_branch")] == [4, 4, 4, 4]
assert ["%.1e" % MG[k]["mean"] for k in ("M52", "C", "C_stream", "C_branch")] == ["1.3e-05", "6.7e-06", "7.9e-06", "1.1e-05"]
assert ["%.2f" % MG[k]["p_vs_m52"] for k in ("C", "C_stream", "C_branch")] == ["0.14", "0.14", "0.54"]
MG_P_T = {k: "%.2f" % MG[k]["p_vs_m52"] for k in ("C", "C_stream", "C_branch")}
MG_MEAN_T = {k: ("%+.1e" % MG[k]["mean"]).replace("e-0", "e-").replace("+", "＋") for k in MG}
# 对齐配置 6 份主 loss 的 r：3 份高于 GPU 留一法全部 15 份（判档刚好过线）
R6 = FC["rep6_loss"]
R6_ABOVE = sum(1 for x in R6["c"] if x > max(R6["gpu_loo"]))
assert R6_ABOVE == 3 and len(R6["c"]) == 6 and len(R6["gpu_loo"]) == 15
R6_GPU_MAX_T = "%.2f" % max(R6["gpu_loo"])
# F30：对齐配置两种候选并排（对齐记录「两种候选并排」结果文件，每份用现行检查项目录判；待拍板）
CF = FC["configs2"]; CFP = FC["configs2_first_pend"]
def _cnt(k):
    v = CF[k]; return len(v), sum(1 for x in v if x["fail"] == 0), sum(1 for x in v if x["pass"] == 9)
CF_C, CF_M = _cnt("c_1000"), _cnt("m58_1000")
assert CF_C == (6, 6, 3) and CF_M == (5, 5, 2)
assert CF["c_1000"][0]["pass"] == 8 and CF["m58_1000"][0]["pass"] == 9      # 各自的第一份
assert CFP["c_300"] == {"L4.grad_norm": 1.01} and CFP["m58_300"] == {"L4.loss": 1.18, "L4.grad_norm": 1.3}
CF_C_BUMP_T, CF_M_BUMP_T = FC_BUMP_T["对齐配置"], FC_BUMP_T["补流梯度舍法"]
CF_C_N, CF_C_ALL = CF_C[0], CF_C[2]
CF_M_N, CF_M_ALL = CF_M[0], CF_M[2]
FC_M_MEAN_T = FC_MEAN_T["补流梯度舍法"].replace("+", "＋")
# ── 2026-11-12 · 先编译、再扫描 —— 整个训练步里「该舍 BF16 没舍」的位置（出处：对齐记录 小尺寸知识库「同一段在 v5p 上编出来」、
#    hc_head 最小复现的 HLO/LLO 结果文件、TPU 自比影响逐件「iHC 不强制舍入」、整步扫描的结果文件与扫描脚本；实测，除标推断）──
# 左块例子：多路残差分支输入那次（TPU 自比影响逐件「iHC 不强制舍入」的最小复现：4 路加权和 → 转 BF16 → RMSNorm，v5p 离线编译，省舍入默认开）。
#   训练步优化后 HLO 里那对 convert 在每一处使用前都还在；均方那颗 kernel 里打包成 BF16 的指令 0 条（平方用的是没舍的 FP32）；
#   同一处补上 reduce_precision：256 条，真舍；关掉省舍入：两种写法都真舍。
EL_IHC_PACKS = (0, 256)
assert EL_IHC_PACKS[0] == 0 < EL_IHC_PACKS[1]
# 收拢处那颗 norm 反向 kernel（hc_head 最小复现）：两路梯度各转 BF16 的两条 convert 在 HLO 里还在、LLO 打包 0 条；对齐配置在两路相加之后补的那次舍入
#   给这颗 kernel 多出 688 条打包（加法 2064、乘法 4128 条不变），可原来那两条 convert 补完照样被省 —— 补的是另一处。
EL_PROBE = {"convert_bf16": 2, "packs_base": 0, "packs_fixed": 688, "vadd_f32": 2064, "vmul_f32": 4128}
# 整步扫描：现行对齐配置训练步（v5p-16 离线编译，libtpu 0.0.46，省舍入默认开）。优化后 HLO 里 convert(f32→bf16) 在同一颗融合里又被转回 f32 使用的「往返」，按位置归类（使用次数，不是 kernel 数）。
#   按类数字取自整步扫描结果文件；在两个 MaxText 版本上复算一致。
EL_USES = {"wgrad": 75,            # 每个微批的权重梯度 dW 不舍
           "branchin": 70,           # 分支输入的梯度合流（典型是注意力输入；通用线性层那一行注意力、MLP 都走，扫描分不开各占多少）
           "fwd_linear": 51,       # 前向：线性层输出
           "fwd_attn_gate": 34,    # 前向：注意力输出门
           "fwd_expert": 13,       # 前向：路由专家 act×up
           "fwd_norm": 6,          # 前向：norm 输出
           "expert_bwd": 7,        # 反向：路由专家 act×up
           "norm_dgrad": 7,        # 反向：norm 输入梯度
           "idx_mask": 9}          # 索引器掩码：值本来就在 BF16 网格上，往返无害
EL_TOTAL = sum(EL_USES.values())
assert EL_TOTAL == 272
# 照研究线案例的分法：③ 前向几处只含线性层输出、输出门、专家（98）；norm 输入梯度、norm 输出、专家反向单列，优先级低（20）
EL_FWD = EL_USES["fwd_linear"] + EL_USES["fwd_attn_gate"] + EL_USES["fwd_expert"]
EL_MINOR = EL_USES["expert_bwd"] + EL_USES["norm_dgrad"] + EL_USES["fwd_norm"]
assert EL_FWD == 98 and EL_MINOR == 20 and EL_USES["wgrad"] + EL_USES["branchin"] + EL_FWD + EL_MINOR + EL_USES["idx_mask"] == EL_TOTAL
# 整颗 kernel 只含往返（没有真舍、没有 reduce_precision）的：41 颗（扫描穿过 bitcast / reshape 这类只搬数据的运算；不穿过的早先一遍是 30 颗，合入版不用），
#   LLO 打包指令全部 0 条；负对照：同配置关省舍入编译，同类 kernel 35 颗，没有一颗是 0 条。按类：dW 34、索引器掩码 4、norm 3（都含前向 norm 输出的往返，其中 1 颗还含反向 norm 输入梯度）。
EL_K = {"wgrad": 34, "idx_mask": 4, "norm": 3}
EL_K_TOTAL = sum(EL_K.values())
assert EL_K_TOTAL == 41
EL_K_ZERO = 41            # 只含往返的 kernel 里 0 条打包的颗数
EL_K_NOEXC, EL_K_NOEXC_ZERO = 35, 0   # 负对照：关省舍入时同类 kernel 颗数、其中 0 条打包的颗数
assert EL_K_ZERO == EL_K_TOTAL and EL_K_NOEXC_ZERO == 0
# 已补上的（iHC 两处、hc_head 收拢处、流梯度）：转回 f32 后紧跟 reduce_precision，或输入本来就在 BF16 网格上，不计入 272
EL_PACKS_FIXED = EL_PROBE["packs_fixed"]
EL_IHC_FIXED = EL_IHC_PACKS[1]
EL_WGRAD, EL_BRANCHIN = EL_USES["wgrad"], EL_USES["branchin"]
# 讲义备答用：小尺寸关掉省舍入（上一代配置）1000 步主 loss 比 GPU 高 4.1e-5（对齐记录 小尺寸知识库「精度本身对轨迹影响很大」；实测）；TPU 自比 35 个 1000 步长跑主 loss 全为正
EL_NOEXC_LOSS, EL_SELF_RUNS = 4.1e-5, 35
assert EL_NOEXC_LOSS > 4 * 9.4e-6
# 已用 reduce_precision 补上的往返（不计入 272）：全图 55 处，其中多路残差与收拢处 30 处、其余 25 处（相减得出；扫描结果里能分清的 norm 15 处，其余没有逐个归类）
EL_COVERED, EL_COVERED_MHC = 55, 30
assert EL_COVERED - EL_COVERED_MHC == 25
# 先查的三类的优先级（对齐记录案例：GPU 那边会舍、TPU 省掉、在反向梯度路径上，三条都满足的为高）：dW 高、分支输入梯度合流高、前向几处中
EL_LEFT = EL_USES["wgrad"] + EL_USES["branchin"] + EL_FWD
assert EL_LEFT == 243
# 代码层已对到只差累加顺序：对齐配置再开「两边共用一次末层归一化」，CPU 上这一段与 GPU 只差 FP32 累加顺序；上卡 4 份第 551–1000 步 +5.0e-6，对对齐配置 p 0.40（实测）
EL_SHARED = {"n": 4, "mean": 5.0e-6, "p_vs_c": 0.40}
# GPU 上拿掉收拢处那次梯度舍入（4 份，对 GPU 16 份留一法）：第 51–100 步 +4.9e-6（p 0.0002）、第 551–1000 步 −2.9e-6（p 0.018）（实测）
EL_GPU_HEADGRAD = {"n": 4, "front": 4.9e-6, "front_p": 0.0002, "late": -2.9e-6, "late_p": 0.018}
assert EL_GPU_HEADGRAD["front"] > 0 > EL_GPU_HEADGRAD["late"]
EL_GPU_HEADGRAD_N = EL_GPU_HEADGRAD["n"]
EL_HC_FRONT_T = ("%+.1e" % EL_GPU_HEADGRAD["front"]).replace("e-0", "e-").replace("+", "＋")
EL_HC_LATE_T = "−" + ("%.1e" % -EL_GPU_HEADGRAD["late"]).replace("e-0", "e-")
EL_SHARED_N, EL_SHARED_P_T = EL_SHARED["n"], "%.2f" % EL_SHARED["p_vs_c"]
assert EL_HC_FRONT_T == "＋4.9e-6" and EL_HC_LATE_T == "−2.9e-6"
# ── 抵消（「秤」那一拍）：第 551–1000 步主 loss 对 GPU 16 份均值（单因素交叉结果文件；实测）──
#   修之前（两处都没补）9.4e-6；只补收拢处 12.5e-6（4 份）；只补流梯度 1.9e-6；两处都补（对齐配置）6.9e-6。
#   GPU 上拿掉收拢处那次梯度舍入，后段 −2.9e-6（实测）。只补收拢处比修之前高的那一截（本课相减，约 3.1e-6）与它同量级、方向相反：
#   推断收拢处梯度不舍一直在把后段往下压，「只补流梯度 1.9e-6」里有它这一份。
CX = {"none": FC_MEAN["修之前"], "head": MG["M52"]["mean"], "stream": FC_MEAN["补流梯度舍法"], "both": FC_MEAN["对齐配置"]}
CX_HEAD_UP = CX["head"] - CX["none"]
assert "%.1e" % CX["head"] == "1.3e-05" and abs(CX_HEAD_UP - 3.1e-6) < 0.1e-6 and abs(CX_HEAD_UP + EL_GPU_HEADGRAD["late"]) < 0.5e-6
CX_T = {k: "%.1f" % (v * 1e6) for k, v in CX.items()}
CX_HEAD_UP_T = "%.1f" % (CX_HEAD_UP * 1e6)
assert CX_T == {"none": "9.4", "head": "12.5", "stream": "1.9", "both": "6.9"} and CX_HEAD_UP_T == "3.1"
CX_STREAM_T = "＋" + CX_T["stream"] + "e-6"
EL_IHC_P0 = EL_IHC_PACKS[0]
assert "%.8f" % bf16(0.72314453125) == "0.72265625"   # 动画 DeclaredElided 里的示意值
CX_NONE_T, CX_HEAD_T, CX_STREAM_V, CX_BOTH_T = ("%+.1f" % (CX[k] * 1e6) for k in ("none", "head", "stream", "both"))
# ── 上卡验证（研究线已合入：GPU 上把前两类拆成三处照 TPU 改，各 4 份，第 551–1000 步主 loss 对 GPU 16 份留一法；置换检验 p 对 GPU 自己）──
#   分支输入里注意力输入那部分 +3.0e-6（p 0.021）；MoE 输入那部分 +9.6e-7（p 0.47）；两处一起开 +1.9e-7（p 0.87）；每微批 dW −3.8e-7（p 0.73）。
#   研究线判读：dW 排除；两处梯度合流没坐实（单开像有一点，合开反而没了，同类舍入合起来应叠加——推断；按格数校正 0.021×4＝0.084，推断是偶然）。TPU 一头（补注意力输入那处）在跑。
GV = {"attn": (3.0e-6, 0.021), "moe": (9.6e-7, 0.47), "both": (1.9e-7, 0.87), "wgrad": (-3.8e-7, 0.73)}
assert GV["both"][0] < GV["attn"][0] and GV["wgrad"][1] > 0.5 and abs(GV["attn"][1] * 4 - 0.084) < 1e-9
GV_T = {k: (("%+.1e" % v).replace("e-0", "e-").replace("+", "＋").replace("-", "−", 1) if v < 0 else ("%+.1e" % v).replace("e-0", "e-").replace("+", "＋"), "%.2f" % p if p >= 0.1 else "%.3f" % p) for k, (v, p) in GV.items()}
GV_ATTN_T, GV_ATTN_P = GV_T["attn"]; GV_BOTH_T, GV_BOTH_P = GV_T["both"]; GV_WGRAD_T, GV_WGRAD_P = GV_T["wgrad"]
assert (GV_ATTN_T, GV_ATTN_P, GV_BOTH_T, GV_BOTH_P, GV_WGRAD_T, GV_WGRAD_P) == ("＋3.0e-6", "0.021", "＋1.9e-7", "0.87", "−3.8e-7", "0.73")
# 逐份（同一结果文件 cells.*.runs.*.lm_551_1000）：
GV_RUNS = {"loo": [-1.3517e-06, -1.237e-06, 1.4003e-06, 9.02e-07, 8.26e-07, 2.595e-07, 1.5504e-06, -4.0038e-06, 3.1626e-06, -2.177e-07, -4.5225e-06, 1.4445e-06, 2.7395e-06, -2.971e-07, -2.175e-07, -4.374e-07],
           "attn": [2.302e-06, 1.9979e-06, 1.6653e-06, 6.0531e-06], "moe": [-2.7124e-06, 1.1416e-06, 4.8749e-06, 5.159e-07],
           "both": [6.21e-08, -6.418e-07, 5.281e-07, 8.13e-07], "wgrad": [-8.135e-07, -2.626e-07, -5.45e-07, 8.96e-08]}
for _k in ("attn", "moe", "both", "wgrad"):
    assert abs(sum(GV_RUNS[_k]) / 4 - GV[_k][0]) < 0.06e-6, _k
assert len(GV_RUNS["loo"]) == 16 and abs(sum(GV_RUNS["loo"])) < 1e-9
# 秤（研究线案例的口径：三个毛病）：只补流梯度 → 两处都补，后段 +5.0e-6（p 0.024，实测）；GPU 拿掉收拢处那次梯度舍入 −2.9e-6，方向一致、只解释其中一部分（推断）。
#   只补收拢处比修之前高 3.1e-6，单看分不开（p 0.26），只作旁证。
CX_UP_T = "%.1f" % ((CX["both"] - CX["stream"]) * 1e6)
assert CX_UP_T == "5.0" and TW_P_T["c_m58"] == "0.024"
CX_HEAD_P = 0.26
# ── 上卡验证的 TPU 一头（研究线已合入：对齐配置之上补分支输入的梯度合流，各 4 份；第 551–1000 步主 loss 对 GPU 16 份均值；p 对对齐配置 6 份）──
#   补注意力输入那处 +6.1e-6（p 0.67）；注意力加 MoE 输入 +5.3e-6（p 0.37）；对齐配置 6 份 +6.9e-6。两头都排除了清单前三类（研究线判读）。
TV_RUNS = {"C": [8.153e-06, 1.12912e-05, 2.097e-06, 5.1174e-06, 6.3224e-06, 8.7067e-06],
           "attn": [4.0455e-06, 4.565e-06, 5.9796e-06, 9.8895e-06], "all": [2.8151e-06, 7.8097e-06, 6.1737e-06, 4.2489e-06]}
TV_P = {"attn": 0.67, "all": 0.37}
TV_MEAN = {k: sum(v) / len(v) for k, v in TV_RUNS.items()}
assert ["%.1e" % TV_MEAN[k] for k in ("C", "attn", "all")] == ["6.9e-06", "6.1e-06", "5.3e-06"] and abs(TV_MEAN["C"] - FC_MEAN["对齐配置"]) < 1e-10   # 逐份取 5 位有效数字
TV_T = {k: ("%+.1e" % v).replace("e-0", "e-").replace("+", "＋") for k, v in TV_MEAN.items()}
TV_ATTN_T, TV_ALL_T = TV_T["attn"], TV_T["all"]
TV_ATTN_P, TV_ALL_P = "%.2f" % TV_P["attn"], "%.2f" % TV_P["all"]

# ══════════════════════════════════════════════════════════════════════════════════════
# ── 2026-11-13 · 第三次迁移（混元 4 的另一个尺寸，从生产代码原样起步）：设备一到 8 个，投影的部分和在 BF16 里累加 ──
#   出处：对齐记录里这次迁移的端到端说明（「设备数到 8 以上」那一节）、现象记录两条（窗口化累加、回放与训练写法不同）、
#   窗口化循环扫描的结果文件与扫描脚本（CPU 上离线编译训练步，运行库 0.0.46）。全部实测，标「推断」的除外。
# ══════════════════════════════════════════════════════════════════════════════════════
# 单算子回放（8 个设备）：中招的那批投影 TPU 对 FP64 2.35e-3、GPU 1.66e-3，R 1.41（结果文件原值 1.41–1.42）；o_proj、down 1.00。
M3_PROJ_TPU, M3_PROJ_GPU = 2.35e-3, 1.66e-3
M3_PROJ_R, M3_PROJ_R_AFTER, M3_OUTPROJ_R = 1.41, 1.00, 1.00
assert abs(M3_PROJ_TPU / M3_PROJ_GPU - M3_PROJ_R) < 0.01            # 2.35 / 1.66 ＝ 1.416，结果文件报 1.41–1.42
assert abs(M3_PROJ_R - math.sqrt(2)) < 0.01                           # ≈ √2：像多舍了一次同量级的数（§4.2 的读法）
M3_PROJ_R_T, M3_PROJ_R_AFTER_T = "%.2f" % M3_PROJ_R, "%.2f" % M3_PROJ_R_AFTER
# 中招 ／ 不中招（规律：FSDP 切的那一维是不是求和维）
M3_HIT = ["q_a", "q_b", "kv_a", "kv_b", "输出门", "稠密 MLP 的 gate、up", "共享专家的 gate、up"]
M3_MISS = ["o_proj", "稠密 MLP 的 down", "共享专家的 down"]
# 8 个设备上 q_a 的循环体（编译产物显示）：8 片排 2 轮，每轮两次 704 宽的矩阵乘（输出 BF16），累加器 BF16
M3_WIDTH, M3_SHARDS, M3_ROUNDS, M3_MM_PER_ROUND = 2816, 8, 2, 2
M3_MM_K = M3_WIDTH // M3_SHARDS * (M3_SHARDS // M3_ROUNDS // M3_MM_PER_ROUND)
assert M3_WIDTH // M3_SHARDS == 352 and M3_MM_K == 704 and M3_MM_K * M3_ROUNDS * M3_MM_PER_ROUND == M3_WIDTH
M3_TPU_ROUNDINGS = M3_ROUNDS * (M3_MM_PER_ROUND + 1)                # 每次矩阵乘输出舍 1 次 ＋ 每轮加进累加器再舍 1 次
assert M3_TPU_ROUNDINGS == 6                                          # GPU（和 4 个设备）只舍 1 次
# 离线编译扫描：窗口化循环总数 ／ 其中带部分和相加、累加器 BF16 的（开关打开后这些全部变 FP32）
M3_SCAN = [("v5p", 4, 0, 0), ("v5p", 8, 64, 20), ("v5p", 16, 64, 20), ("v7x", 16, 64, 20)]
assert [r[3] for r in M3_SCAN] == [0, 20, 20, 20]
# 修法：投影声明 FP32 输出（打开投影 FP32 累加）。阶梯上一级 → 这一级，1000 步、单份运行
M3_OVER2N = (91, 63)                     # 第 1 步超出 GPU 两倍噪声的梯度张量
M3_LOSS_R, M3_GN_R, M3_PN_R = (1.14, 0.96), (0.93, 0.98), (0.97, 1.00)
assert M3_LOSS_R[0] > 1 >= M3_LOSS_R[1] and M3_GN_R[1] <= 1 and M3_PN_R[1] <= 1   # 三项第一次同时 ≤ 1
M3_COST = 0.004                          # 步时 ＋0.007 s（单份运行）
M3_COST_T = "%.1f%%" % (M3_COST * 100); _lock("M3_COST_T", M3_COST_T, "0.4%")
M3_REPLAY_NOBATCH_R = 2.12               # 回放工具没把输入按 batch 切时的投影 R（工具缺陷，已修）
# ── 第三次迁移 · 输出头精度三档（出处：对齐记录第三次迁移知识库「输出头精度」页、现象记录两条；单独回放输出头这一次矩阵乘，输入取 GPU 第 1 步，对 CPU FP64）──
#   精度档对 FP32 输入：默认 1 遍 BF16（把 FP32 输入舍成 BF16）、high 3 遍、highest 6 遍。
M3_LMH = {   # (权重状态, 档) : (logits 前向, 对 x 的梯度, 对 W 的梯度)
    ("on", "default"): (1.1e-7, 1.0e-5, 8.3e-6), ("on", "highest"): (1.1e-7, 1.1e-6, 1.2e-7),
    ("off1", "default"): (1.3e-3, 1.3e-3, 8.3e-6), ("off10", "default"): (1.9e-3, 1.9e-3, 8.3e-6),
    ("off10", "high"): (5.9e-6, 1.7e-5, 2.7e-6), ("off10", "highest"): (1.4e-7, 1.3e-6, 1.2e-7)}
M3_LMH_HOSTFP32 = (1.5e-7, 3.7e-7)          # 主机上真 FP32 矩阵乘的 logits：权重在格点上 ／ 离格
assert round(M3_LMH[("off10", "default")][0] / M3_LMH_HOSTFP32[1], -2) == 5100   # 「约 5000 倍」（知识库推断，本课复算）
assert 250 < M3_LMH[("off10", "default")][0] / M3_LMH[("off10", "high")][0] < 400  # high 比默认好约 300 倍
# 曲线与步时（单份运行）：4K 主 loss 1000 步 ／ 前 300 步；步时
M3_LMH_LOSS4K = {"default": (1.82, 7.32), "high": (0.99, 1.03), "highest": (1.02, 1.31)}
M3_LMH_STEP4K = {"default": 1.640, "high": 1.734, "highest": 1.842}
M3_LMH_COST = {k: M3_LMH_STEP4K[k] / M3_LMH_STEP4K["default"] - 1 for k in ("high", "highest")}
assert ("%.1f%%" % (M3_LMH_COST["high"] * 100), "%.1f%%" % (M3_LMH_COST["highest"] * 100)) == ("5.7%", "12.3%")
M3_LMH_LOSS32K = {"high": (0.94, 0.97), "highest": (0.99, 1.08)}
M3_LMH_STEP1_MED, M3_LMH_STEP1_OVER = 6.4e-4, (91, 91)   # 第 1 步默认档对 highest：116 个张量相对差中位数；超 2N 个数不动
M3_VOCAB = 120832
M3_LMH_DEF300, M3_LMH_HIGH300 = "%.2f" % M3_LMH_LOSS4K["default"][1], "%.2f" % M3_LMH_LOSS4K["high"][1]
M3_LMH_COST_HIGH_T, M3_LMH_COST_HIGHEST_T = "%.1f%%" % (M3_LMH_COST["high"] * 100), "%.1f%%" % (M3_LMH_COST["highest"] * 100)
M3_VOCAB_T = format(M3_VOCAB, ",")
# ── 第三次迁移 · 切法一变，编译器删不删舍入也跟着变（RMSNorm scale；出处：知识库「编译器删舍入」「RMSNorm 在 32K 被删」两页、现象记录一条）──
#   4K 程序：8 个设备、FSDP 8，这次舍入保住；32K 程序：16 个设备、CP 8 × FSDP 2，8 个 norm 全部被删、保住 0 个（离线编译扫描）。
M3_NORM_ELIDED = {"4K": 0, "32K": 8}
# 二分（32K，第 2 级之上加开关，主 loss 偏差的平均，单位 σ ＝ GPU 8 份份间标准差）
M3_BISECT_300 = [("不加（对照）", -24.3), ("投影 FP32 累加", -24.6), ("精确超越函数", -24.6), ("精确 Adam ＋ 正交化迭代的舍入", -24.1),
                 ("四处一起（见底注）", -0.6)]                 # 窗口第 51–300 步
M3_BISECT_120 = [("只加其余三处之一（三条运行）", -30.8), ("只加 RMSNorm scale", -0.2)]   # 窗口第 51–120 步；四条「各只加一处」：−30.84／−30.88／−0.22／−30.88σ；这一轮没有不加开关的对照
M3_DRIFT_PEAK = -27                      # 第 100 步约 −27σ；第 500 步前后翻正
M3_SCALE_TRACK = [(50, 3.2e-4), (100, 6.5e-4), (150, 9.5e-4)]   # FP32 scale 偏离 1.0 的中位数（16 个 norm scale）
M3_SCALE_SLOPE = 6.4e-6
assert all(abs(d / s - M3_SCALE_SLOPE) < 0.2e-6 for s, d in M3_SCALE_TRACK)   # 约每步 6.4e-6，线性
M3_SCALE_CROSS = {200: (0.029, 0.30), 250: (0.07, 0.55)}   # 舍成 BF16 后不等于 1.0 的元素比例：中位数 ／ 最大
M3_NORM_STEP = (1.804, 1.802)            # 32K 步时，开关前后（各一份）
# ── 第三次迁移 · 重算重选路由（知识库「重算重选路由」页）：生产默认整层重算，反向把路由重选一遍 ──
M3_REROUTE_PCT, M3_REROUTE_MAX = 2.4, 3.05     # 不同的 token 平均 ／ 最多（%），160 次两两比较里 96 次不同
M3_REROUTE_PN = (4.56, 2.27)                   # 参数范数 1000 步 r：重选 → 不重选（前 300 步 11.87 → 1.19）
M3_REROUTE_PN300 = (11.87, 1.19)
# ── 第三次迁移 · 超越函数（知识库「超越函数」页；逐元素对 FP64 正确舍入值的最大误差，ULP）──
M3_ULP = {"exp": (105, 1, 64, 64), "log": (62, 3, 2, 2), "sigmoid": (83, 3, 30, 30), "silu": (86, 4, 29, 30)}   # v5p 默认、v5p 开、v7x 默认、v7x 开
M3_ULP_JAXONLY = {"exp": 1, "log": 3, "sigmoid": 83, "silu": 86}   # v5p 只给 JAX 设 precision=highest
M3_TRANS_COST = 0.008
M3_TRANS_INSN = (3, (43, 45))                  # 每个向量寄存器的指令数：默认 → 精确档（改到向量单元上用多项式）
M3_TRANS_L1 = {"post 门": (8.68, 0.96), "路由权重": (4.36, 1.15), "主交叉熵": (2.97, 1.14), "MTP 交叉熵": (3.97, 1.25)}
assert all(v[2] == v[3] or abs(v[2] - v[3]) <= 1 for v in M3_ULP.values())   # v7x 开不开几乎一样
M3_REROUTE_PN0, M3_REROUTE_PN300_0 = "%.2f" % M3_REROUTE_PN[0], "%.2f" % M3_REROUTE_PN300[0]
M3_TRANS_COST_T = "%.1f%%" % (M3_TRANS_COST * 100)
# 2026-11-14：小数一律 e 记法（正文用）
STEP1_ALL_E = "%.2e" % STEP1_ALL; STEP1_ALL_E = STEP1_ALL_E.replace("e-0", "e-")
REF_LAYER0_FWD_E, REF_LAYER0_BWD_E = ("%.1e" % REF_LAYER0_FWD).replace("e-0", "e-"), ("%.1e" % REF_LAYER0_BWD).replace("e-0", "e-")
assert (STEP1_ALL_E, REF_LAYER0_FWD_E, REF_LAYER0_BWD_E) == ("1.28e-3", "4.0e-3", "2.1e-3")
# ── 2026-11-14 · 第三次迁移 · 图文 32K 的两个新发现（出处：对齐记录第三次迁移那一目录的说明页「MTP 偏差的定位」「MTP 输入舍入的验证格」及后续、
#    「路由翻转定责」两段，和 mtp-l1 汇总结果文件；同一状态各算一步；实测，除标推断）──
# ① MTP 输入：主干末层归一化的输出先存 BF16、进 MTP 的归一化再转回 FP32，GPU 两次都舍，TPU（省舍入开着）省掉中间那次
M3_MTPIN_COEF = "0.997"                  # MTP 归一化乘的系数（1/rms × 缩放），输入已归一化（rms 约 1.004）
M3_MTPIN_HNORM = 1.000456                # GPU ÷ TPU：MTP 归一化输出的逐 token 范数比（中位）
M3_MTPIN_BIAS = 4.4e-4                   # FP64 仿真：两次都舍时输出整体偏大的比例项（0.00044）；只舍一次的两种写法都 < 5e-6
assert abs((M3_MTPIN_HNORM - 1) - M3_MTPIN_BIAS) < 0.3e-4
M3_MTPIN_C = (1.8e-3, 4.9e-3)            # MTP 层 19 个非路由张量梯度的比例项 |c|（GPU − TPU，全部同号，为负）
M3_MTPIN_C_GPU = "1e-4"                  # 6 份 GPU 之间的比例项只差约 ±1e-4
M3_MTPIN_LOSS = (0.7515084147453308, 0.751466691493988, 0.7514692898908204)   # MTP loss：TPU 修前、TPU 补强制舍入后、GPU
M3_MTPIN_LOSS_BEFORE_T = "%.1e" % abs(M3_MTPIN_LOSS[0] - M3_MTPIN_LOSS[2])
M3_MTPIN_LOSS_AFTER_T = "%.1e" % abs(M3_MTPIN_LOSS[1] - M3_MTPIN_LOSS[2])
M3_MTPIN_LOSS_BEFORE_T, M3_MTPIN_LOSS_AFTER_T = (M3_MTPIN_LOSS_BEFORE_T.replace("e-0", "e-"), M3_MTPIN_LOSS_AFTER_T.replace("e-0", "e-"))
assert (M3_MTPIN_LOSS_BEFORE_T, M3_MTPIN_LOSS_AFTER_T) == ("3.9e-5", "2.6e-6")
M3_MTPIN_FIXED, M3_MTPIN_OF = 15, 18     # 补上以后 MTP 层 18 个张量里 15 个比例项在 ±4e-4 以内（原来 1.8e-3 到 4.9e-3）
M3_MTPIN_QPATH = (1.1e-3, 1.4e-3)        # 还超线的 q 路三个（q_a_proj ＋0.142%、q_b_proj ＋0.128%、q_a_layernorm ＋0.112%）；推断 q 一侧反向还有一处
M3_MTPIN_C_T = "%s 到 %s" % (sci(M3_MTPIN_C[0]), sci(M3_MTPIN_C[1]))
M3_MTPIN_QPATH_T = "%s 到 %s" % (sci(M3_MTPIN_QPATH[0]), sci(M3_MTPIN_QPATH[1]))
assert (M3_MTPIN_C_T, M3_MTPIN_QPATH_T) == ("1.8e-3 到 4.9e-3", "1.1e-3 到 1.4e-3")
# ② 视觉塔 q/k：视觉塔不选专家，可梯度要穿过后面的 MoE 层传回来；两种候选配置越线的只有视觉塔 q/k 投影，灌 TPU 自选路由的 FP64 重判
M3_VIT_N = (10, 5)                        # 越线张量数：配置乙、配置甲（两种配置越线的层不重合）
M3_VIT_R_T = ("1.21–1.53", "1.22–1.38")  # 越线时的 R
M3_VIT_RFLIP_T = ("0.43–0.83", "0.44–0.62")   # 按 TPU 自己选的路由重算 FP64 以后的 R
M3_VIT_FLIP_PCT = "8%–26%"               # 这一状态下各 MoE 层（第 1、2、3 层与 MTP 层）翻转的 token 比例：约 8% ／ 16% ／ 22% ／ 26%
M3_VIT_N_1, M3_VIT_N_2 = M3_VIT_N
M3_VIT_R_1, M3_VIT_R_2 = M3_VIT_R_T
M3_VIT_RFLIP_1, M3_VIT_RFLIP_2 = M3_VIT_RFLIP_T
# ── 2026-11-14 · 第三节「看逐步曲线分两步」的真实例子：第三次迁移图文 32K、一种候选配置 1000 步的 MTP loss
#    （判定数以对齐记录第三次迁移目录里这次运行的 L4 结果文件为准：MTP loss 那一条，窗口第 2–1000 步，GPU 8 份两两 28 对；
#     逐步曲线由私有导出脚本写进 topic08_mtp32k.json，口径照对齐记录 analysis/curves.py 的 pairdiff 包络）──
M3_MTP32K_SM, M3_MTP32K_SM_N = 2.0766690776494447e-05, 6.8663489452395695e-06
M3_MTP32K_MARE, M3_MTP32K_MARE_N = 2.9689957858373493e-05, 2.578596288692888e-05
M3_MTP32K_MAX, M3_MTP32K_MAX_N, M3_MTP32K_MAX_AT = 0.0001354143238917009, 0.0001433842601956052, 941
M3_MTP32K_PAIRS = 28
M3_MTP32K_R = tuple("%.2f" % (a / b) for a, b in [(M3_MTP32K_SM, M3_MTP32K_SM_N), (M3_MTP32K_MARE, M3_MTP32K_MARE_N), (M3_MTP32K_MAX, M3_MTP32K_MAX_N)])
assert M3_MTP32K_R == ("3.02", "1.15", "0.94"), M3_MTP32K_R
M3_MTP32K_SM_T, M3_MTP32K_SM_N_T = tx(M3_MTP32K_SM, plus=True), tx(M3_MTP32K_SM_N)
M3_MTP32K_MARE_T, M3_MTP32K_MARE_N_T = tx(M3_MTP32K_MARE), tx(M3_MTP32K_MARE_N)
M3_MTP32K_MAX_T, M3_MTP32K_MAX_N_T = tx(M3_MTP32K_MAX), tx(M3_MTP32K_MAX_N)
assert (M3_MTP32K_SM_T, M3_MTP32K_SM_N_T, M3_MTP32K_MARE_T, M3_MTP32K_MARE_N_T, M3_MTP32K_MAX_T, M3_MTP32K_MAX_N_T) == \
       ("＋2.08e-5", "6.87e-6", "2.97e-5", "2.58e-5", "1.35e-4", "1.43e-4")
M3_MTP32K_R_SM, M3_MTP32K_R_MARE, M3_MTP32K_R_MAX = M3_MTP32K_R


def mtp32k():
    """逐步曲线（下标 0 ＝ 第 1 步）；跟结果文件的三个判定数互相钉住。"""
    import json as _j
    import os as _o
    d = _j.load(open(_o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "topic08_mtp32k.json"), encoding="utf-8"))
    dt, env = d["逐步相对差"], d["噪声包络"]
    w = dt[1:1000]
    assert abs(sum(w) / len(w) - M3_MTP32K_SM) < 0.01e-5
    assert abs(sum(abs(x) for x in w) / len(w) - M3_MTP32K_MARE) < 0.01e-5
    k = max(range(1, 1000), key=lambda i: abs(dt[i]))
    assert k + 1 == M3_MTP32K_MAX_AT and abs(abs(dt[k]) - M3_MTP32K_MAX) < 0.01e-5
    return dt, env


def mtp32k_gpu7():
    """GPU 另外 7 份各自对同一份基线的逐步相对差（下标 0 ＝ 第 1 步），只为画出「跟同一份基线比」的方向；尺子不用它们算。"""
    import json as _j
    import os as _o
    d = _j.load(open(_o.path.join(_o.path.dirname(_o.path.abspath(__file__)), "topic08_mtp32k.json"), encoding="utf-8"))
    return d["GPU 另外 7 份对基线"]


def _roll50(x):
    return [sum(x[k - 49:k + 1]) / 50.0 for k in range(49, len(x))]


_dt32, _env32 = mtp32k()
_g7 = mtp32k_gpu7()
M3_MTP32K_G7_POS = [round(100.0 * sum(1 for x in g[1:1000] if x > 0) / 999) for g in _g7]          # 各份在零上面的步数比例（%）
M3_MTP32K_G7_SM = [sum(g[1:1000]) / 999 for g in _g7]                                                # 各份带符号均值（第 2–1000 步）
M3_MTP32K_G7_ROLLPOS = [round(100.0 * sum(1 for v in _roll50(g[1:1000]) if v > 0) / 950) for g in _g7]   # 50 步平均线在零上面的比例（%）
M3_MTP32K_ROLLPOS = round(100.0 * sum(1 for v in _roll50(_dt32[1:1000]) if v > 0) / 950)
assert M3_MTP32K_G7_POS == [54, 58, 52, 60, 58, 53, 58], M3_MTP32K_G7_POS
assert M3_MTP32K_G7_ROLLPOS == [62, 55, 57, 84, 79, 64, 76] and M3_MTP32K_ROLLPOS == 100, (M3_MTP32K_G7_ROLLPOS, M3_MTP32K_ROLLPOS)
assert abs(max(M3_MTP32K_G7_SM) - M3_MTP32K_SM_N) < 0.01e-6        # 7 份里最大的那份，正好就是 28 对尺子的那一对（基线对它）
M3_MTP32K_G7_POS_T = "%d%%–%d%%" % (min(M3_MTP32K_G7_POS), max(M3_MTP32K_G7_POS))
M3_MTP32K_G7_ROLLPOS_T = "%d%%–%d%%" % (min(M3_MTP32K_G7_ROLLPOS), max(M3_MTP32K_G7_ROLLPOS))
M3_MTP32K_G7_SM_T = "%s 到 %s" % (tx(min(M3_MTP32K_G7_SM), "%.1e", plus=True), tx(max(M3_MTP32K_G7_SM), "%.1e", plus=True))
assert (M3_MTP32K_G7_POS_T, M3_MTP32K_G7_ROLLPOS_T, M3_MTP32K_G7_SM_T) == ("52%–60%", "55%–84%", "＋1.5e-6 到 ＋6.9e-6"), (M3_MTP32K_G7_POS_T, M3_MTP32K_G7_ROLLPOS_T, M3_MTP32K_G7_SM_T)
M3_MTP32K_POS_PCT = "%d%%" % round(100.0 * sum(1 for x in _dt32[1:1000] if x > 0) / 999)
M3_MTP32K_IN_PCT = "%d%%" % round(100.0 * sum(1 for x, e in zip(_dt32[1:1000], _env32[1:1000]) if abs(x) <= e) / 999)
assert (M3_MTP32K_POS_PCT, M3_MTP32K_IN_PCT) == ("75%", "87%"), (M3_MTP32K_POS_PCT, M3_MTP32K_IN_PCT)
# ── 2026-11-14 · §7.2 Adam 公式卡：第 300 步各词频桶 √v̂ 的中位数（GPU 一侧；出处：对齐记录混元 3 v5p 验收目录的冷门行链条结果文件，
#    同一文件里 TPU 一侧几乎一样：没出现过的词 2.77e-9、ε 以下 91.4%）；压小倍数 ＝ (√v̂ ＋ ε) ÷ √v̂，ε 在分母里的占比 ＝ ε ÷ (√v̂ ＋ ε)（本课推导）──
VHAT_BUCKETS = [("100 次以上", 16326, 1.1657499171211293e-05), ("11–100 次", 31999, 3.2864090247453103e-06),
                ("1–10 次", 13869, 2.638073892879559e-07), ("没出现过", 58638, 2.6164874668502303e-09)]
VHAT_RARE_BELOW_EPS = 0.9191437085046706          # 没出现过的词：√v̂ ＜ ε 的比例（GPU）；√v̂ ＜ ε ⇔ 压小 2 倍以上
assert abs(VHAT_BUCKETS[3][2] - RARE_VHAT_MED) < 0.05e-9 and abs(VHAT_RARE_BELOW_EPS - RARE_EPS_FRAC) < 0.005
assert VHAT_BUCKETS[3][1] == RARE_ROWS


def vhat_row(v):
    """(ε 在分母里的占比, 压小倍数)。"""
    return EPS / (v + EPS), (v + EPS) / v


assert abs(vhat_row(2.6e-9)[1] - 4.85) < 0.05 and abs(vhat_row(1e-6)[0] - 0.0099) < 0.0001
# ── 2026-11-14 · §7.5 修法故事线用的派生数（ABL_* 是离线消融实测；相减粗分是本课推断，假设截断与吞小项两部分可加）──
ABL_BF16_S1_T, ABL_BF16_S100_T, ABL_BF16_S300_T = (tx(v) for v in ABL_BF16_ONESHOT)
ABL_FP32_S100_T = tx(ABL_FP32_ONESHOT[1])
ABL_S1_RATIO = "%d" % round(abs(ABL_BF16_ONESHOT[0] / ABL_FP32_ONESHOT[0]), -1)
ABL_TRUNC_S100_T, ABL_TRUNC_S300_T = tx(ABL_BF16_ONESHOT[1] - ABL_FP32_ONESHOT[1], "%.1e"), tx(ABL_BF16_ONESHOT[2] - ABL_FP32_ONESHOT[2], "%.1e")
SWAP_TRUNC_SHARE = "%d%%" % round((0.36 - 0.10) / 0.36 * 100)
PN_FIX1_R, PN_TPU_PROD_R = "%.4f" % PN_FIX1, "%.4f" % PN_TPU_PROD
assert (ABL_BF16_S1_T, ABL_S1_RATIO, ABL_TRUNC_S100_T, ABL_TRUNC_S300_T, SWAP_TRUNC_SHARE) == ("−6.51e-4", "380", "−3.5e-4", "−2.0e-4", "72%"), (ABL_BF16_S1_T, ABL_S1_RATIO, ABL_TRUNC_S100_T, ABL_TRUNC_S300_T, SWAP_TRUNC_SHARE)
# ── 2026-11-14 夜 · §7.5 修法（夜间试讲意见落地）：一律用生产近似配置、对 GPU 生产 5 份均值的第 1000 步参数范数 ──
PN_RCX5 = 1.2424e-2                     # GPU 生产只把输出头 dgrad 换 FP32 一次加到底，5 份均值（topic08_series.json 的 pnorm_rc_gpu_dgrad_fp32_mean5）
PN_RCX5_R, PN_PRODCHUNK_R, PN_TRUE_R = "%.4f" % PN_RCX5, "%.4f" % PN_PRODCHUNK, "%.4f" % PN_TRUE
PN_RCX5_SHARE = "%d%%" % round(PN_RCX5 / PN_TPU_PROD * 100)
assert PN_RCX5_SHARE == "59%"
HEAD_S1_E = tx(abs(HEAD_BF16_S1), "%.1e")                       # 第 1 步 GPU 输出头 dgrad 沿 w̄ 少推的相对量（原写 0.065%，小数改 e 记法）
assert HEAD_S1_E == "6.5e-4"
EXACT_SELF_PN_T = tx(EXACT_SELF_PN, "%.2e")
TPU_LOSS_TAIL_T = tx(TPU_LOSS_TAIL, "%.2e")
GPU_B_TAIL_T, GPU_C_TAIL_T = tx(GPU_RUNS10_TAIL[1], "%.1e"), tx(GPU_RUNS10_TAIL[2], "%.1e", plus=True)
assert (TPU_LOSS_TAIL_T, GPU_B_TAIL_T, GPU_C_TAIL_T) == ("−1.66e-4", "−3.8e-5", "＋2.5e-7"), (TPU_LOSS_TAIL_T, GPU_B_TAIL_T, GPU_C_TAIL_T)
assert abs((TPU_LOSS_TAIL - sum(GPU_RUNS10_TAIL) / 10) - (-1.4e-4)) < 0.05e-4
HEAD_S100_E = tx(abs(HEAD_BF16_S100), "%.1e"); assert HEAD_S100_E == "6.1e-4"
PN_CHUNK_STEP_R = "%.4f" % (PN_PRODCHUNK - PN_RCX5); assert PN_CHUNK_STEP_R == "0.0083"
# ── 2026-11-15 · §3.3「从一条序列到一个 r」公式卡的实测小表（第二次迁移 TPU 自比，FSDP→纯数据并行，前 300 步；
#    被测与尺子都由逐步记录现算核过：主 loss 三个统计的被测值和 GPU 8 份两两 28 对的最大值；实测）──
CARD_R_ROWS = [("主 loss", "带符号均值", 5.5819e-6, 3.00915e-6), ("主 loss", "MARE", 8.835e-6, 6.721e-6), ("主 loss", "最大单步（第 273 步）", 3.704e-5, 4.531e-5),
               ("主 loss 后段（第 101–300 步）", "带符号均值", 9.22e-6, 4.54e-6), ("梯度范数", "带符号均值", -2.599e-4, 9.924e-5)]
assert ["%.2f" % (abs(a) / b) for _q, _s, a, b in CARD_R_ROWS] == ["1.85", "1.31", "0.82", "2.03", "2.62"]   # 后段一行出处：对齐记录尺子说明页的逐项表
# ── 2026-11-15 · fig-rratio 补两边离 FP64 的绝对值（对齐记录混元 3 v5p 单算子回放结果文件，qkv 那一条：TPU rel_l2 对自己输入的 FP64、GPU 同口径最大值；实测）──
QKV_ABS_BEFORE, QKV_ABS_AFTER, QKV_ABS_GPU = 2.335e-3, 1.890e-3, 1.890e-3
assert abs(QKV_ABS_BEFORE / QKV_ABS_GPU - R_QKV_BEFORE) < 0.002 and abs(QKV_ABS_AFTER / QKV_ABS_GPU - R_QKV_AFTER) < 0.002
