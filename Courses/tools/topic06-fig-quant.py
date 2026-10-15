# -*- coding: utf-8 -*-
r"""专题六 · 第十二节「少搬字节：量化与 kernel 融合」的四张静态图。

fig6-quant-bytes.svg —— 回答：权重从 BF16 降到 FP8、FP4，decode 一步的搬运下限缩多少？
  · 数字：V3 每 token 激活 37B、总 671B（技术报告摘要）；TPU v7 每芯片 7.37 TB/s、192 GB（Google Cloud 公开规格），
    都从 topic06_numbers 取。MXFP4 4.25 bit（OCP MX v1.0），NVFP4 4.5 bit（NVIDIA NVFP4 技术博客）。
  · 本课推导：下限 ＝ 激活权重字节 ÷ 8 颗芯片带宽合计。假设 batch=1、权重均匀摊在 8 颗芯片上、带宽跑满；
    不含 KV、激活值和通信。
  · 刻意没画：INT4（与 FP4 字节数相同，区别在数值格式，见 r1 那张图）。

fig6-quant-mx.svg —— 回答：MXFP4 和 NVFP4 同样是「4 bit」，为什么一个 4.25、一个 4.5？
  · 数字：OCP Microscaling Formats (MX) v1.0：每 32 个元素共享一个 8 bit E8M0 指数；
    NVIDIA「Introducing NVFP4」博客：每 16 个元素一个 FP8 E4M3 缩放，外加每张量一个 FP32（摊薄后不计）。

fig6-quant-r1.svg —— 回答：为什么 R1 在 v7 上非得用 FP4？INT4 读错一位会怎样？
  · 数字：tpu/tpu-inference/DeepSeek-R1-671B-FP4/README.md:72-73、:875-878（FP8 MoE ≈ 101.8、FP4 ≈ 60.9、
    合计 ≈ 70、每 device 上限 94.75）；上限也见 Kimi-K2.6-1T-A32B-INT4/README.md:217 的运行日志。
    INT4 偏移：tpu/tpu-inference/Kimi-K2.6-1T-A32B-INT4/README.md:265-266。
  · ⚠️ README 正文写 GB，日志写 GiB；101.8 无论按哪个单位都放不下 94.75 GiB（≈ 101.7 GB 时也只是持平、没有余量给激活和 KV）。

fig6-quant-fuse.svg —— 回答：kernel 边界上丢的时间有多少？megakernel 能捡回多少？
  · 数字：Hazy Research「Look Ma, No Bubbles!」（hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles）的算例和实测；
    Mirage Persistent Kernel arXiv 2512.22219 摘要；Inferact「TPU megakernels」（https://inferact.ai/blog/tpu-megakernels）。
  · 刻意没画：CUDA Graph 单独一条（它省的是 CPU 发射开销，正文一句话交代）。
"""

from topic03_draw import Fig, wpx, BL, OR, GR, RD, PU, CY, GY, INK, GY2, LINE
from topic06_numbers import V3_TOTAL, V3_ACTIVE, V7_BW, V7_HBM

W = 1400

# ══ fig6-quant-bytes ════════════════════════════════════════════════════
CHIPS = 8
BW8 = CHIPS * V7_BW
assert abs(BW8 / 1e12 - 58.96) < 1e-9
FORMATS = [("BF16", 16.0, GY2), ("FP8", 8.0, BL), ("MXFP4", 4.25, GR), ("NVFP4", 4.5, CY)]
ACT_GB = {n: V3_ACTIVE * b / 8 / 1e9 for n, b, _ in FORMATS}
ALL_GB = {n: V3_TOTAL * b / 8 / 1e9 for n, b, _ in FORMATS}
T_MS = {n: V3_ACTIVE * b / 8 / BW8 * 1e3 for n, b, _ in FORMATS}
assert round(ACT_GB["BF16"]) == 74 and round(ACT_GB["FP8"]) == 37
assert round(ACT_GB["MXFP4"], 2) == 19.66 and round(ACT_GB["NVFP4"], 2) == 20.81
assert [round(T_MS[n], 2) for n, _, _ in FORMATS] == [1.26, 0.63, 0.33, 0.35], T_MS
assert round(ALL_GB["BF16"]) == 1342 and round(ALL_GB["FP8"]) == 671
HBM8 = CHIPS * V7_HBM / 1e9
assert HBM8 == 1536

# ══ fig6-quant-mx ═══════════════════════════════════════════════════════
MX_BLOCK, MX_SCALE = 32, 8
NV_BLOCK, NV_SCALE = 16, 8
MX_BITS = 4 + MX_SCALE / MX_BLOCK
NV_BITS = 4 + NV_SCALE / NV_BLOCK
assert MX_BITS == 4.25 and NV_BITS == 4.5

# ══ fig6-quant-r1 ═══════════════════════════════════════════════════════
R1_FP8_MOE, R1_FP4_MOE, R1_FP4_ALL, DEV_CAP = 101.8, 60.9, 70, 94.75
assert R1_FP8_MOE > DEV_CAP > R1_FP4_ALL > R1_FP4_MOE
assert round(DEV_CAP * 2 ** 30 / 1e9, 1) == 101.7     # 就算上限按 GiB，FP8 MoE 也已超过
INT4_OFFSET = 8
SIGNED = list(range(-8, 8))
STORED = [v + INT4_OFFSET for v in SIGNED]
assert STORED == list(range(16))

# ══ fig6-quant-fuse：Hazy 算例 ════════════════════════════════════════════
H100_BW, L1B_BYTES = 3.35e12, 2.48e9
T_MOVE_US = L1B_BYTES / H100_BW * 1e6
assert round(T_MOVE_US) == 740
FWD_IDEAL = 1e6 / T_MOVE_US
assert round(FWD_IDEAL, -1) == 1350
KPL, LAYERS, STALL_US = 7, 16, 5
T_STALL_US = KPL * LAYERS * STALL_US
assert T_STALL_US == 560
FWD_REAL = 1e6 / (T_MOVE_US + T_STALL_US)
assert round(FWD_REAL, -1) == 770
# Inferact（TPU v7，K3，DSpark）
INF_STEP_MS, INF_AL, INF_TPS, GB200_TPS = 8.5, 6, 700, 452
assert INF_AL / (INF_STEP_MS / 1e3) > INF_TPS      # 6 ÷ 8.5 ms ≈ 706


# ════════════════════════════════════════════════════════════════════════
def fig_bytes():
    f = Fig(W, "DeepSeek-V3 每个 token 激活 370 亿参数。按 BF16 存要读 74 GB，FP8 37 GB，MXFP4 约 19.66 GB，NVFP4 约 20.81 GB。"
               "放在 8 颗 TPU v7 上、合计带宽 58.96 TB/s，decode 一步只读权重的时间下限分别是 1.26、0.63、0.33、0.35 毫秒。"
               "右边是全部 671B 参数要占多少 HBM：BF16 1,342 GB，FP8 671 GB，8 颗芯片合计 1,536 GB")
    y0 = f.header("位数减半，一步的搬运下限就减半　——　<tspan font-weight=\"700\">FP4 只要 BF16 的四分之一多一点</tspan>",
                  "DeepSeek-V3 · 8 颗 TPU v7 · batch 1 · 只算权重（本课推导）",
                  [(GY2, "BF16"), (BL, "FP8"), (GR, "MXFP4"), (CY, "NVFP4")])
    PH = 330
    LW = 860
    py = f.panel(0, y0, LW, PH, "每 token 要读的权重（37B 激活参数）", BL)
    sc = 480 / 80
    for j, (n, b, col) in enumerate(FORMATS):
        yy = py + 30 + j * 68
        f.t(24, yy + 26, n, INK, True, 17)
        f.t(24, yy + 48, "%g bit" % b, GY, size=15)
        f.box(130, yy + 8, ACT_GB[n] * sc, 38, col, col, 4)
        f.t(130 + ACT_GB[n] * sc + 12, yy + 33, "%.2f GB → 至少 %.2f ms" % (ACT_GB[n], T_MS[n]), INK, True, 16)
    RX = LW + 14
    pr = f.panel(RX, y0, W - RX, PH, "整个模型放得下吗（671B）", OR)
    sc2 = 360 / 1600
    xh = RX + 90 + HBM8 * sc2
    for j, (n, b, col) in enumerate(FORMATS):
        yy = pr + 44 + j * 64
        f.t(RX + 20, yy + 30, n, INK, True, 16)
        bwid = ALL_GB[n] * sc2
        f.box(RX + 90, yy + 10, bwid, 30, col, col, 4)
        lab = "{:,} GB".format(round(ALL_GB[n]))
        if RX + 90 + bwid + 8 + wpx(lab, 15) > xh - 6:      # 标签会撞上 HBM 那条线 → 写进条里
            f.t(RX + 90 + bwid - 8, yy + 31, lab, "#ffffff", True, 15, "end")
        else:
            f.t(RX + 90 + bwid + 8, yy + 31, lab, INK, True, 15)
    f.path("M%d,%d L%d,%d" % (xh, pr + 36, xh, pr + 296), RD, 1.6, dash="5,4", arrow=False)
    f.t(xh, pr + 26, "8 颗合计 {:,} GB".format(round(HBM8)), RD, True, 15, "end")
    yb = f.src(y0 + PH + 6,
               "V3：技术报告 arXiv 2412.19437 摘要，总参数 671B、每 token 激活 37B。TPU v7：Google Cloud 公开规格，每芯片 HBM 192 GB、带宽 7.37 TB/s。",
               "MXFP4：OCP Microscaling Formats (MX) Specification v1.0。NVFP4：NVIDIA 技术博客「Introducing NVFP4 for Efficient and Accurate Low-Precision Inference」。",
               "⚠️ 本课推导：下限 ＝ 激活参数 × 每参数位数 ÷ 8 ÷ (8 × 7.37 TB/s)。假设权重均匀摊开、带宽跑满；不含 KV、激活值和通信，实际一步更慢。",
               "注：真实部署很少整模型都降到 FP4：DeepSeek-V4 只把专家放 FP4、其余多为 FP8（HF 模型卡）。")
    f.save("fig6-quant-bytes.svg", yb + 14)


# ════════════════════════════════════════════════════════════════════════
def fig_mx():
    f = Fig(W, "两种 4 bit 格式的存法。MXFP4：32 个 4 bit 的数共用一个 8 bit 的缩放，格式叫 E8M0，指数 8 位、尾数 0 位，只能表示 2 的幂；"
               "平均每个数 4 加 8 除以 32，等于 4.25 bit。"
               "NVFP4：16 个 4 bit 的数共用一个 8 bit 的 FP8 缩放，格式叫 E4M3，指数 4 位、尾数 3 位，是个小号浮点数；平均每个数 4.5 bit")
    y0 = f.header("同叫 4 bit　——　<tspan font-weight=\"700\">差在几个数共用一个缩放</tspan>",
                  "每个小格是一个 4 bit 的数；橙色大格是这一块共用的缩放（8 bit）",
                  [(GR, "MXFP4 的元素"), (CY, "NVFP4 的元素"), (OR, "块缩放（8 bit）")])
    PH = 330
    py = f.panel(0, y0, W, PH, "一块的位布局", INK)
    cw = 30
    for j, (name, blk, col, scl, bits, plain) in enumerate((
            ("MXFP4", MX_BLOCK, GR, "E8M0", MX_BITS, "E8M0 ＝ 指数 8 位、尾数 0 位：只能表示 2 的幂（½、1、2、4…）"),
            ("NVFP4", NV_BLOCK, CY, "E4M3", NV_BITS, "E4M3 ＝ 指数 4 位、尾数 3 位：一个小号的浮点数"))):
        yy = py + 44 + j * 140
        f.t(24, yy + 26, name, INK, True, 18)
        f.t(24, yy + 50, "%d 个一块" % blk, GY, size=15)
        x0 = 150
        f.box(x0, yy, 2 * cw, 40, OR, OR, 4)
        f.t(x0 + cw, yy + 26, scl, "#ffffff", True, 15, "middle")
        for i in range(blk):
            f.box(x0 + 2 * cw + 4 + i * cw, yy, cw - 3, 40, col, col, 3)
        f.t(x0, yy + 72, "4 ＋ 8 ÷ %d ＝ %.2f bit／数" % (blk, bits), INK, True, 17)
        f.t(x0 + 290, yy + 72, plain, GY, size=15)
    yb = f.src(y0 + PH + 6,
               "OCP Microscaling Formats (MX) Specification v1.0：块大小 32，共享缩放为 E8M0（8 bit，只有指数）。",
               "NVIDIA「Introducing NVFP4」：每 16 个 FP4 (E2M1) 共享一个 FP8 E4M3 缩放，另有每张量 FP32 二级缩放。",
               "⚠️ 本课推导：平均位数 ＝ 4 ＋ 缩放位数 ÷ 块大小，FP32 二级缩放摊到百万级元素上不计。")
    f.save("fig6-quant-mx.svg", yb + 14)


# ════════════════════════════════════════════════════════════════════════
# FP4 (E2M1) 能表示的非负值：1 位符号、2 位指数（偏置 1）、1 位尾数（OCP MX v1.0）
E2M1 = sorted({(m * 0.5) if e == 0 else 2 ** (e - 1) * (1 + m / 2.0) for e in range(4) for m in range(2)})
assert E2M1 == [0, 0.5, 1, 1.5, 2, 3, 4, 6], E2M1
EX_V = -3                                  # 示意：挑一个权重跟着走一遍
EX_S = EX_V + INT4_OFFSET
assert EX_S == 5


def fig_r1():
    f = Fig(W, "INT4 的存法和读法。INT4 和 FP4 一样每个数占 4 bit、只有 16 个值：INT4 是负 8 到 7 的等距整数，FP4 是浮点。"
               "对称 INT4 的权重是负 8 到 7，存进文件时整体加 8 变成 0 到 15，比如负 3 存成 5。"
               "加载时减 8，负 3 原样回来，模型正常；忘了减 8，负 3 被读成 5，每个权重都偏大 8，模型输出崩成重复的乱词")
    y0 = f.header("INT4：存的时候加了 8，读的时候要减回来　——　<tspan font-weight=\"700\">少减一次，每个权重都错</tspan>",
                  "INT4 和 FP4 一样每个数 4 bit、16 个值；INT4 是 −8 到 7 的等距整数，FP4 是浮点（0、0.5、1、1.5、2、3、4、6 及其负值）",
                  [(BL, "模型里的权重"), (OR, "文件里存的"), (GR, "读对了"), (RD, "读错了")])
    PH = 470
    py = f.panel(0, y0, W, PH, "跟着一个权重走一遍：−3 → 存成 5 → 读回来", OR)
    X0, CW = 230, 70
    assert X0 + 16 * CW < W - 20

    def row(yy, lab, vals, col, tint, sub_=None):
        f.t(24, yy + 28, lab, INK, True, 17)
        if sub_:
            f.t(24, yy + 50, sub_, GY, size=15)
        for i, v in enumerate(vals):
            x = X0 + i * CW
            hi = (SIGNED[i] == EX_V)
            f.box(x, yy, CW - 6, 40, col if hi else tint, col, 5, 1.8 if hi else 1)
            f.t(x + (CW - 6) / 2, yy + 27, str(v), "#ffffff" if hi else col, True, 17, "middle")
    xi = X0 + SIGNED.index(EX_V) * CW + (CW - 6) / 2
    y1 = py + 30
    row(y1, "模型里的权重", SIGNED, BL, "#e8f0fe", "−8 到 7")
    y2 = y1 + 110
    f.line(xi, y1 + 46, xi, y2 - 6, OR, 1.8)
    f.t(xi + 12, y1 + 80, "存进文件：＋8", OR, True, 16)
    row(y2, "文件里存的", STORED, OR, "#fef7e0", "0 到 15")
    y3 = y2 + 110
    f.line(xi, y2 + 46, xi, y3 - 6, GR, 1.8)
    f.t(xi - 12, y2 + 80, "加载时 −8", GR, True, 16, "end")
    row(y3, "读对了", [s - INT4_OFFSET for s in STORED], GR, "#e6f4ea", "和第一行一样")
    y4 = y3 + 100
    row(y4, "忘了减 8", STORED, RD, "#fce8e6", "每个都大了 8")
    xg = X0 + SIGNED.index(EX_V) * CW + CW - 3          # 两格之间那条 6px 的缝
    f.path("M%d,%d L%d,%d L%d,%d" % (xg - 6, y2 + 44, xg, y2 + 54, xg, y4 - 6), RD, 1.8)
    f.t(xg + 14, y3 + 76, "直接当原值读", RD, True, 16)
    f._pan = None
    f.t(X0, y4 + 74, "结果：读对了模型正常；读错了，输出崩成重复的「foss foss foss」", INK, True, 17)
    yb = f.src(y4 + 90,
               "INT4：tpu/tpu-inference/Kimi-K2.6-1T-A32B-INT4/README.md:265-266。对称 int4 [−8, 7] 以无符号 [0, 15] 存放，加载时须减 8；直接按位转换会让模型崩溃，输出重复的「foss foss foss」。",
               "⚠️ 示意：跟着走的「−3」是随手挑的一个值。FP4 (E2M1) 的取值按 OCP MX v1.0 由脚本算出：1 位符号、2 位指数、1 位尾数。",
               "正文里 R1 的容量数：tpu/tpu-inference/DeepSeek-R1-671B-FP4/README.md:72-73、:875-878（v7x-8 ＝ 4 颗芯片 ＝ 8 个 device；FP8 专家约 101.8、FP4 专家约 60.9、连同注意力约 70，均为每 device）；"
               "上限 94.75 出自运行日志 hbm=(…, 94.75) GiB（Kimi-K2.6-1T-A32B-INT4/README.md:217）。94.75 GiB ≈ 101.7 GB，101.8 无论按哪个单位都不剩余量。")
    f.save("fig6-quant-r1.svg", yb + 14)


# ════════════════════════════════════════════════════════════════════════
def fig_fuse():
    f = Fig(W, "Hazy Research 的算例。Llama-3.2-1B 有 2.48 GB 权重，H100 带宽 3.35 TB/s，光读一遍要约 740 微秒，"
               "上限每秒约 1,350 次前向。但一次前向要发上百个 kernel，每层 7 个、16 层，每个交接处空等约 5 微秒，合计 560 微秒，"
               "实际只能到每秒约 770 次。中间一行把一层放大，7 个 kernel 之间各空一段。"
               "megakernel 把整次前向写成一个 kernel，下一段的权重在上一段计算时就开始读，只剩读权重的 740 微秒")
    y0 = f.header("kernel 之间的空当　——　<tspan font-weight=\"700\">每个交接处都在等，加起来很可观</tspan>",
                  "Hazy Research 的算例：H100 · Llama-3.2-1B · 单条序列，一次前向的时间线，按比例",
                  [(BL, "读权重"), (RD, "kernel 交接时空等"), (GR, "megakernel：边算边预取")])
    PH = 420
    py = f.panel(0, y0, W, PH, "一次前向的时间线（微秒，按比例）", BL)
    X0 = 230
    sc = (W - X0 - 40) / (T_MOVE_US + T_STALL_US)
    per_move = T_MOVE_US / LAYERS
    per_stall = T_STALL_US / LAYERS
    # ① 一层一层发 kernel：16 层，每层读权重 ＋ 7 个空当（层内合并画）
    yy = py + 40
    f.t(24, yy + 22, "一层一层发 kernel", INK, True, 17)
    f.t(24, yy + 46, "约 {:,} 次／秒".format(int(round(FWD_REAL, -1))), RD, True, 16)
    x = X0
    for _ in range(LAYERS):
        f.box(x, yy, per_move * sc, 44, BL, BL, 1)
        x += per_move * sc
        f.box(x, yy, per_stall * sc, 44, RD, RD, 1)
        x += per_stall * sc
    f.t(x, yy + 70, "读 %d ＋ 空等 %d ＝ {:,} 微秒".format(round(T_MOVE_US) + T_STALL_US) % (round(T_MOVE_US), T_STALL_US),
        INK, True, 16, "end")
    # ② 放大第一层：7 个 kernel，每个交接处空等约 5 微秒
    yz = yy + 120
    zx0, zw = X0, 760
    zsc = zw / (per_move + per_stall)
    f.path("M%d,%d L%d,%d" % (X0, yy + 46, zx0, yz - 4), GY2, 1, dash="3,3", arrow=False)
    f.path("M%d,%d L%d,%d" % (X0 + (per_move + per_stall) * sc, yy + 46, zx0 + zw, yz - 4), GY2, 1, dash="3,3", arrow=False)
    f.t(24, yz + 22, "放大其中一层", INK, True, 17)
    f.t(24, yz + 46, "每层 %d 个 kernel" % KPL, GY, size=15)
    km = per_move / KPL * zsc
    ks = per_stall / KPL * zsc
    x = zx0
    for k in range(KPL):
        f.box(x, yz, km, 44, BL, BL, 3)
        f.t(x + km / 2, yz + 28, "k%d" % (k + 1), "#ffffff", True, 15, "middle")
        x += km
        f.box(x, yz, ks, 44, "#fce8e6", RD, 3)
        x += ks
    f.t(zx0 + km + ks / 2, yz + 70, "交接空等约 %d 微秒" % STALL_US, RD, True, 16, "middle")
    f.t(zx0 + zw + 20, yz + 28, "交接时谁也不在读权重", RD, True, 16)
    # ③ megakernel
    ym = yz + 140
    f.t(24, ym + 22, "整次前向一个 kernel", INK, True, 17)
    f.t(24, ym + 46, "上限约 {:,} 次／秒".format(int(round(FWD_IDEAL, -1))), GR, True, 16)
    f.box(X0, ym, T_MOVE_US * sc, 44, GR, GR, 2)
    f.t(X0 + T_MOVE_US * sc + 12, ym + 28, "只剩读权重 %d 微秒" % round(T_MOVE_US), GR, True, 16)
    yb = f.src(y0 + PH + 6,
               "Hazy Research「Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B」（hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles，2025-05-27）：算例 3.35 TB/s ÷ 2.48 GB ≈ 1,350 次／秒；"
               "每层 7 个 kernel × 16 层 × 约 5 µs ≈ 770 次／秒；实测基线都开了 CUDA Graph 和 torch.compile。"
               "实测：vLLM、SGLang 最多用到 H100 带宽的 50%，megakernel 到 78%；比 vLLM 快约 2.5 倍、比 SGLang 快 1.5 倍以上；B200 上一次前向不到 680 µs。",
               "⚠️ 示意：第一行把每层 7 个空当合成一段画，第二行才按真实位置分开；7 个 kernel 等长是简化。",
               "正文里另两组数的出处 —— Mirage Persistent Kernel：arXiv 2512.22219 摘要，「up to 1.7×」端到端延迟。",
               "Inferact「700 TPS on Kimi K3」（https://inferact.ai/blog/tpu-megakernels，2026-09-23）：16 颗 TPU v7（32 个 TensorCore）、Pallas 编写、每 TensorCore 64 MiB VMEM；一步约 8.5 ms、每步接受约 6 个；GB200 对照为 16 张卡的 vLLM 配方，452 tok/s。"
               "⚠️ 本课推导：每步接受 6 个 ÷ 8.5 ms ≈ 706 tok/s，与博客的「700 多」对得上。")
    f.save("fig6-quant-fuse.svg", yb + 14)


fig_bytes()
fig_mx()
fig_r1()
fig_fuse()
