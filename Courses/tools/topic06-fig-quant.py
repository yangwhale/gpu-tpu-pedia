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

from topic03_draw import Fig, BL, OR, GR, RD, PU, CY, GY, INK, GY2, LINE
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
    sc = 520 / 80
    for j, (n, b, col) in enumerate(FORMATS):
        yy = py + 30 + j * 66
        f.t(24, yy + 26, n, INK, True, 15)
        f.t(24, yy + 46, "%g bit" % b, GY, size=12.5)
        f.box(130, yy + 8, ACT_GB[n] * sc, 34, col, col, 4)
        f.t(130 + ACT_GB[n] * sc + 12, yy + 31, "%.2f GB → 至少 %.2f ms" % (ACT_GB[n], T_MS[n]), INK, True, 14)
    RX = LW + 14
    pr = f.panel(RX, y0, W - RX, PH, "整个模型放得下吗（671B）", OR)
    sc2 = 360 / 1600
    for j, (n, b, col) in enumerate(FORMATS):
        yy = pr + 30 + j * 66
        f.t(RX + 20, yy + 30, n, INK, True, 14)
        f.box(RX + 100, yy + 12, ALL_GB[n] * sc2, 28, col, col, 4)
        f.t(RX + 100 + ALL_GB[n] * sc2 + 8, yy + 31, "{:,} GB".format(round(ALL_GB[n])), INK, True, 13)
    xh = RX + 100 + HBM8 * sc2
    f.path("M%d,%d L%d,%d" % (xh, pr + 30, xh, pr + 290), RD, 1.6, dash="5,4", arrow=False)
    f.t(xh - 6, pr + 296, "8 颗合计 1,536 GB", RD, True, 13, "end")
    yb = f.band(y0 + PH + 16, "ok", "batch 小时，时间几乎全花在读权重上", [
        "每个权重少一半位数，一步就少搬一半字节；这是量化在 decode 上最直接的收益。",
        "省下的 HBM 还能多放 KV、多接几路请求，这又回到「把一趟分给更多人」那条路上。",
    ])
    yb = f.src(yb + 10,
               "V3：技术报告 arXiv 2412.19437 摘要，总参数 671B、每 token 激活 37B。TPU v7：Google Cloud 公开规格，每芯片 HBM 192 GB、带宽 7.37 TB/s。",
               "MXFP4：OCP Microscaling Formats (MX) Specification v1.0。NVFP4：NVIDIA 技术博客「Introducing NVFP4 for Efficient and Accurate Low-Precision Inference」。",
               "⚠️ 本课推导：下限 ＝ 激活参数 × 每参数位数 ÷ 8 ÷ (8 × 7.37 TB/s)。假设权重均匀摊开、带宽跑满；不含 KV、激活值和通信，实际一步更慢。",
               "注：真实部署很少整模型都降到 FP4：DeepSeek-V4 只把专家放 FP4、其余多为 FP8（HF 模型卡）。")
    f.save("fig6-quant-bytes.svg", yb + 14)


# ════════════════════════════════════════════════════════════════════════
def fig_mx():
    f = Fig(W, "两种 4 bit 格式的存法。MXFP4：32 个 4 bit 的数共用一个 8 bit 的指数缩放，平均每个数 4 加 8 除以 32，等于 4.25 bit。"
               "NVFP4：16 个 4 bit 的数共用一个 8 bit 的 FP8 缩放，平均每个数 4.5 bit。块越小，缩放越贴合局部的数值范围，精度越好，多花的位数也越多")
    y0 = f.header("同叫 4 bit　——　<tspan font-weight=\"700\">差在几个数共用一个缩放</tspan>",
                  "每个小格是一个 4 bit 的数；带色的大格是这一块共用的缩放",
                  [(GR, "MXFP4 的元素"), (CY, "NVFP4 的元素"), (OR, "块缩放（8 bit）")])
    PH = 300
    py = f.panel(0, y0, W, PH, "一块的位布局", INK)
    cw = 30
    for j, (name, blk, col, scl, bits) in enumerate((("MXFP4", MX_BLOCK, GR, "E8M0", MX_BITS),
                                                     ("NVFP4", NV_BLOCK, CY, "E4M3", NV_BITS))):
        yy = py + 40 + j * 120
        f.t(24, yy + 24, name, INK, True, 16)
        f.t(24, yy + 46, "%d 个一块" % blk, GY, size=13)
        x0 = 140
        f.box(x0, yy, 2 * cw, 36, OR, OR, 4)
        f.t(x0 + cw, yy + 24, scl, "#ffffff", True, 13, "middle")
        for i in range(blk):
            f.box(x0 + 2 * cw + 4 + i * cw, yy, cw - 3, 36, col, col, 3)
        xe = x0 + 2 * cw + 4 + blk * cw
        f.t(x0, yy + 62, "4 ＋ 8 ÷ %d ＝ %.2f bit／数" % (blk, bits), INK, True, 14.5)
        if blk == NV_BLOCK:
            f.t(xe + 20, yy + 24, "另有每张量一个 FP32 缩放，摊到每个数上可忽略", GY, size=13)
    yb = f.band(y0 + PH + 16, "info", "块小一点，精度好一点，字节多一点", [
        "块越小，一个缩放管的数越少，越能贴着局部的大小走；代价是平均每个数多 0.25 bit。",
        "两种格式由不同硬件原生支持；选哪种不只看精度，还要看你的芯片的矩阵单元吃哪种。",
    ])
    yb = f.src(yb + 10,
               "OCP Microscaling Formats (MX) Specification v1.0：块大小 32，共享缩放为 E8M0（8 bit，只有指数）。",
               "NVIDIA「Introducing NVFP4」：每 16 个 FP4 (E2M1) 共享一个 FP8 E4M3 缩放，另有每张量 FP32 二级缩放。",
               "⚠️ 本课推导：平均位数 ＝ 4 ＋ 缩放位数 ÷ 块大小，FP32 二级缩放摊到百万级元素上不计。")
    f.save("fig6-quant-mx.svg", yb + 14)


# ════════════════════════════════════════════════════════════════════════
def fig_r1():
    f = Fig(W, "左图：DeepSeek-R1 在 TPU v7x-8 上，每个 device 的 HBM 上限约 94.75。专家权重用 FP8 时每个 device 要约 101.8，放不下；"
               "转成 FP4 后专家约 60.9，连同注意力约 70，放得下。右图：对称 INT4 的数是负 8 到 7，存的时候整体加 8 变成 0 到 15；"
               "读的时候忘了减 8，每个权重都偏大 8，模型输出崩成重复的乱词")
    y0 = f.header("FP4 先是被容量逼出来的　——　<tspan font-weight=\"700\">格式和读法都得对上</tspan>",
                  "左：DeepSeek-R1 · TPU v7x-8 · 每 device 的权重占用。右：INT4 的存法与读法",
                  [(RD, "放不下"), (GR, "放得下"), (OR, "存的值"), (BL, "该读出的值")])
    PH = 320
    LW = 700
    py = f.panel(0, y0, LW, PH, "R1 每个 device 要放多少（单位见注）", RD)
    sc = 520 / 110
    rows = [("FP8 专家", R1_FP8_MOE, RD), ("FP4 专家", R1_FP4_MOE, GR), ("FP4 合计", R1_FP4_ALL, GR)]
    for j, (lab, v, col) in enumerate(rows):
        yy = py + 30 + j * 80
        f.t(24, yy + 28, lab, INK, True, 14.5)
        f.box(130, yy + 8, v * sc, 34, col, col, 4)
        f.t(130 + v * sc - 10, yy + 31, ("约 %.1f" if v % 1 else "约 %d") % v, "#ffffff", True, 14, "end")
    xc = 130 + DEV_CAP * sc
    f.path("M%d,%d L%d,%d" % (xc, py + 20, xc, py + 270), INK, 2, dash="6,4", arrow=False)
    f.t(xc + 6, py + 268, "上限 %.2f" % DEV_CAP, INK, True, 13.5)

    RX = LW + 14
    pr = f.panel(RX, y0, W - RX, PH, "INT4：存的时候加了 8，读的时候要减回来", OR)
    x0, cw = RX + 40, 38
    for i, (s, st) in enumerate(zip(SIGNED, STORED)):
        x = x0 + i * cw
        f.box(x, pr + 40, cw - 4, 34, "#e8f0fe", BL, 3)
        f.t(x + (cw - 4) / 2, pr + 63, str(s), BL, True, 13, "middle")
        f.line(x + (cw - 4) / 2, pr + 78, x + (cw - 4) / 2, pr + 110, GY2, 1.2, arrow=True)
        f.box(x, pr + 114, cw - 4, 34, "#fef7e0", OR, 3)
        f.t(x + (cw - 4) / 2, pr + 137, str(st), OR, True, 13, "middle")
    f.t(x0, pr + 180, "上排：模型里的权重（−8 到 7）。下排：文件里存的（0 到 15）", GY, size=13)
    f.t(x0, pr + 212, "正确读法：存的值 − 8", GR, True, 14.5)
    f.t(x0, pr + 240, "直接当原值读：每个权重都大了 8，输出崩成重复的乱词", RD, True, 14.5)
    yb = f.band(y0 + PH + 16, "warn", "量化格式要和硬件、读法同时对上", [
        "FP8 放不下才换 FP4，这一步是容量逼的；省下的字节顺带让每一步搬得更少。",
        "如果矩阵单元不吃 FP4，就得先解回高精度再算；省了搬运，多了一道转换，实际快多少要实测。",
    ])
    yb = f.src(yb + 10,
               "R1：tpu/tpu-inference/DeepSeek-R1-671B-FP4/README.md:72-73、:875-878（v7x-8 ＝ 4 颗芯片 ＝ 8 个 device；FP8 专家约 101.8、FP4 专家约 60.9、连同注意力约 70，均为每 device）。",
               "注：单位：README 正文写 GB，上限 94.75 出自运行日志 hbm=(…, 94.75) GiB（Kimi-K2.6-1T-A32B-INT4/README.md:217）。94.75 GiB ≈ 101.7 GB，101.8 无论按哪个单位都不剩余量。",
               "INT4：tpu/tpu-inference/Kimi-K2.6-1T-A32B-INT4/README.md:265-266。对称 int4 [−8, 7] 以无符号 [0, 15] 存放，加载时须减 8；直接按位转换会让模型崩溃，输出重复的「foss foss foss」。")
    f.save("fig6-quant-r1.svg", yb + 14)


# ════════════════════════════════════════════════════════════════════════
def fig_fuse():
    f = Fig(W, "上图：Hazy Research 的算例。Llama-3.2-1B 有 2.48 GB 权重，H100 带宽 3.35 TB/s，光读一遍要约 740 微秒，"
               "上限每秒约 1,350 次前向。但一次前向要发上百个 kernel，每层 7 个、16 层、每个边界空等约 5 微秒，合计 560 微秒，"
               "实际只能到每秒约 770 次。megakernel 把整次前向写成一个 kernel，让下一段的权重在上一段计算时就开始读。"
               "下图左：GPU 上的实测，Hazy 带宽利用率从最多 50% 到 78%，比 vLLM 快约 2.5 倍；Mirage MPK 自动生成 megakernel，延迟最多降 1.7 倍。"
               "下图右：Inferact 在 16 颗 TPU v7 上跑 Kimi K3，单用户每秒 700 多个 token，16 张 GB200 是 452")
    y0 = f.header("kernel 之间的空当　——　<tspan font-weight=\"700\">每个边界都在等，加起来很可观</tspan>",
                  "上：Hazy 的算例（H100 · Llama-3.2-1B · 单条序列）。下：公开实测",
                  [(BL, "读权重"), (RD, "kernel 边界空等"), (GR, "megakernel：边算边预取")])
    PH = 250
    py = f.panel(0, y0, W, PH, "一次前向的时间线（微秒，按比例）", BL)
    X0 = 200
    sc = (W - X0 - 60) / (T_MOVE_US + T_STALL_US)
    # 一排：分 16 层，每层读权重 + 7 个空当
    yy = py + 30
    f.t(24, yy + 22, "一层一层发 kernel", INK, True, 15)
    f.t(24, yy + 42, "约 %d 次／秒" % round(FWD_REAL, -1), RD, True, 13.5)
    x = X0
    per_move = T_MOVE_US / LAYERS
    per_stall = T_STALL_US / LAYERS
    for _ in range(LAYERS):
        f.box(x, yy + 8, per_move * sc, 30, BL, BL, 1)
        x += per_move * sc
        f.box(x, yy + 8, per_stall * sc, 30, RD, RD, 1)
        x += per_stall * sc
    f.t(x, yy + 60, "读 %d ＋ 空等 %d ＝ %d 微秒" % (round(T_MOVE_US), T_STALL_US, round(T_MOVE_US) + T_STALL_US),
        INK, True, 13.5, "end")
    yy2 = py + 125
    f.t(24, yy2 + 22, "整次前向一个 kernel", INK, True, 15)
    f.t(24, yy2 + 42, "上限约 %d 次／秒" % round(FWD_IDEAL, -1), GR, True, 13.5)
    f.box(X0, yy2 + 8, T_MOVE_US * sc, 30, GR, GR, 2)
    f.t(X0 + T_MOVE_US * sc + 10, yy2 + 29, "只剩读权重 %d 微秒" % round(T_MOVE_US), GR, True, 13.5)

    py2 = y0 + PH + 14
    PH2 = 260
    LW = 700
    pg = f.panel(0, py2, LW, PH2, "GPU 上的实测", PU)
    lines = [("Hazy · H100 · Llama-3.2-1B", "带宽利用：vLLM、SGLang 最多 50% → 78%"),
             ("", "比 vLLM 快约 2.5 倍，比 SGLang 快 1.5 倍以上"),
             ("Hazy · B200", "一次前向不到 680 微秒，比 vLLM 快 3.5 倍以上"),
             ("Mirage MPK", "编译器自动生成 megakernel；端到端延迟降幅最高 1.7×")]
    for j, (a, b) in enumerate(lines):
        yy = pg + 30 + j * 52
        if a:
            f.t(24, yy, a, PU, True, 14)
        f.t(24, yy + 22, b, INK, size=14)
    RX = LW + 14
    pt = f.panel(RX, py2, W - RX, PH2, "TPU 上的公开实测（Inferact）", GR)
    tl = [("16 颗 TPU v7 · Kimi K3 · DSpark", ""),
          ("一步约 %.1f 毫秒，每步接受约 %d 个" % (INF_STEP_MS, INF_AL), ""),
          ("单用户每秒 700 多个 token；16 张 GB200 是 %d" % GB200_TPS, ""),
          ("下一层权重在上一层计算时预取进片上内存", ""),
          ("从头编译不到 90 秒（XLA 大模型要 30 分钟以上）", "")]
    for j, (a, _) in enumerate(tl):
        f.t(RX + 24, pt + 30 + j * 42, a, INK if j else GR, j == 0, 14)
    yb = f.band(py2 + PH2 + 16, "ok", "融合也是少搬：边界上不再干等", [
        "CUDA Graph 省的是 CPU 一个个发 kernel 的开销；kernel 之间那段谁也不读权重的空当，它省不掉。",
        "megakernel 让读权重这条流水线一直不断，带宽利用率才贴近上限。",
    ])
    yb = f.src(yb + 10,
               "Hazy Research「Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B」（hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles，2025-05-27）：算例 3.35 TB/s ÷ 2.48 GB ≈ 1,350 次／秒；"
               "每层 7 个 kernel × 16 层 × 约 5 µs ≈ 770 次／秒；实测基线都开了 CUDA Graph 和 torch.compile。",
               "Mirage Persistent Kernel：arXiv 2512.22219 摘要，「up to 1.7×」端到端延迟。",
               "Inferact「700 TPS on Kimi K3」（https://inferact.ai/blog/tpu-megakernels，2026-09-23）：16 颗 TPU v7（32 个 TensorCore）、Pallas 编写、每 TensorCore 64 MiB VMEM；GB200 对照为 16 张卡的 vLLM 配方。",
               "⚠️ 本课推导：每步接受 6 个 ÷ 8.5 ms ≈ 706 tok/s，与博客的「700 多」对得上。上面那条时间线是 Hazy 的算例按比例画的，空当在真实时间线上分散在 kernel 之间。")
    f.save("fig6-quant-fuse.svg", yb + 14)


fig_bytes()
fig_mx()
fig_r1()
fig_fuse()
