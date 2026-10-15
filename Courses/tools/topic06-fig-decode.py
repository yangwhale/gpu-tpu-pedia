# -*- coding: utf-8 -*-
r"""专题六 · 第五节「单卡里的两块硬骨头」的五张图。

回答的问题（一张图一个）：
  ① fig6-s5-amortize  攒 batch 为什么只救得了权重、救不了注意力？
                      权重一步读一遍、分给 B 个人；KV 每条序列读自己的，B 翻倍它也翻倍。
  ② fig6-s5-tile      同一份 KV，布局为什么能让它占好几倍的地方、让 FP8 白省？
  ③ fig6-s5-mla       MLA 的两种算法为什么各占一个阶段？（decode 看每字节几次运算，prefill 看总运算量）
  ④ fig6-s5-dense     decode 时 MoE 为什么可以「让所有 token 都过这个专家」？
  ⑤ fig6-s5-allreduce 小 decode 的 AllReduce 为什么是延迟问题？

数从哪来：
  · Llama-3.1-70B config.json（hidden 8192、80 层、64 查询头、8 KV 头、intermediate 28672、vocab 128256、
    不共享词表）；参数量由 config 现算，并和 safetensors index 的 total_size 141,107,412,992 字节对上。
  · DeepSeek-V3 config.json（hidden 7168、moe_intermediate_size 2048、256 路由专家 top-8、128 头、
    kv_lora_rank 512、qk_nope 128、qk_rope 64、v_head 128）。共用数取自 topic06_numbers。
  · TPU v7 每芯片 7.37 TB/s、BF16 2,307 TFLOP/s（topic06_numbers）；单 device ＝ 每芯片 ÷ 2。
  · TPU 的 tile：openxla.org/xla/tiled_layout（8×128 个 32 位格；16 位、8 位沿次内层两个／四个拼成一个 32 位）；
    RPA 论文 arXiv 2604.15464 §2.1.2、§3.1（窄头数被补齐到默认 tile；加 packing 维、K/V 合并后用最小 tile）。
  · AllReduce 实测：Shen 等 arXiv 2607.16100（4×GB200 小消息平均 11.0 µs → 2.37 µs；SoL 1.404 µs）。
    NVLink 5 每 GPU 1.8 TB/s（NVIDIA 官方，双向合计），单向 900 GB/s 是本课推导。

刻意没画什么：
  · ① 没画时间、不分卡：只记字节，全模型合计。MoE 模型的权重读取量还会随 batch 变（命中的专家变多），
    所以这张图用稠密的 Llama-3.1-70B，不用 V3。
  · ② 只画 K/V 的一个 token 一层；没画 XLA 在某些形状上会选的「compact 2nd minor」小 tile（见出处）。
  · ④ 没画排序和搬行本身花多久：那取决于实现，没有公开的同口径分解数据。
  · ⑤ TPU 的 ICI 小消息延迟没有找到公开数据，所以全图是 GPU 的数。
"""
import math

import topic06_numbers as NB
from topic03_draw import Fig, BL, OR, GR, RD, GY, GY2, INK, LINE, PU

W = 1400

# ════════════════════════════════════════════════════════════════════
# ① 权重被 batch 分摊，KV 不被分摊（Llama-3.1-70B，bf16，上下文 8K）
# ════════════════════════════════════════════════════════════════════
L_H, L_LAYERS, L_QH, L_KVH, L_HD, L_FF, L_VOCAB = 8192, 80, 64, 8, 128, 28672, 128256
L_PER_LAYER = (L_H * L_QH * L_HD * 2            # q、o
               + L_H * L_KVH * L_HD * 2         # k、v
               + 3 * L_H * L_FF                 # gate、up、down
               + 2 * L_H)                       # 两个 RMSNorm
L_PARAMS = L_PER_LAYER * L_LAYERS + 2 * L_VOCAB * L_H + L_H
assert L_PARAMS * 2 == 141107412992, L_PARAMS   # 对上 safetensors index 的 total_size
L_W_BYTES = L_PARAMS * 2
L_G = L_QH // L_KVH
assert L_G == 8                                  # GQA 组大小 ＝ 专题三那个 GQA-8
L_KV_TOK = 2 * L_KVH * L_HD * 2 * L_LAYERS       # K 和 V、bf16、80 层
assert L_KV_TOK == 327680                        # 每 token 320 KiB
CTX = 8192
L_KV_SEQ = L_KV_TOK * CTX
assert abs(L_KV_SEQ / NB.GB - 2.68) < 0.01
B_CROSS = L_W_BYTES / L_KV_SEQ                   # 权重和 KV 一样重的那个 batch
assert abs(B_CROSS - 52.6) < 0.1
BATCHES = [1, 16, 64, 128]
PER_TOK = {b: (L_W_BYTES / b + L_KV_SEQ) / NB.GB for b in BATCHES}
assert abs(PER_TOK[1] - 143.8) < 0.1 and abs(PER_TOK[128] - 3.79) < 0.01


def fig_amortize():
    f = Fig(W, "攒 batch 时一步要读的字节。横轴是 batch 1、16、64、128，每根柱子是这一步全模型要读的字节，"
               "蓝色是权重 141 GB，四根柱子一样高；紫色是 KV，每条序列 2.68 GB，跟着 batch 线性长。"
               "柱顶写着摊到每个 token 的字节：143.8、11.5、4.89、3.79 GB。batch 约 53 时两块一样重，之后 KV 占大头")
    y0 = f.header("攒 batch：权重一趟分给所有人，KV 每人一份　——　<tspan font-weight=\"700\">batch 越大，KV 越成主角</tspan>",
                  "柱高 ＝ decode 一步要从显存读的字节（全模型合计，不分卡）；柱顶 ＝ 摊到每个 token 的字节。Llama-3.1-70B，bf16，每条序列上下文 8K",
                  [(BL, "权重：一步读一遍，所有序列共用"), (PU, "KV：每条序列读自己的")])
    PH = 500
    py = f.panel(0, y0, W, PH, "一步要读多少字节", BL)
    base = py + PH - 72
    top = py + 60
    VMAX = 500
    assert (L_W_BYTES + 128 * L_KV_SEQ) / NB.GB < VMAX       # 最高那根柱子要落在网格里
    sc = (base - top) / VMAX
    x0, bw, gap = 230, 130, 150
    # 刻度
    for v in (0, 100, 200, 300, 400, 500):
        yv = base - v * sc
        f.line(x0 - 30, yv, x0 + 4 * bw + 3 * gap + 30, yv, LINE, 1, arrow=False)
        f.t(x0 - 40, yv + 5, "%d GB" % v, GY, size=14, anchor="end")
    for i, b in enumerate(BATCHES):
        x = x0 + i * (bw + gap)
        hw = L_W_BYTES / NB.GB * sc
        kv = b * L_KV_SEQ / NB.GB
        hk = kv * sc
        f.box(x, base - hw, bw, hw, BL, BL, 3)
        f.box(x, base - hw - hk, bw, max(hk, 2), PU, PU, 3)
        f.t(x + bw / 2, base + 28, "batch %d" % b, INK, True, 16, anchor="middle")
        f.t(x + bw / 2, base - hw - hk - 32, "每 token", GY, size=14, anchor="middle")
        f.t(x + bw / 2, base - hw - hk - 10, ("%.1f GB" if b == 1 else "%.3g GB") % PER_TOK[b], INK, True, 17, anchor="middle")
        if i == 0:
            f.t(x + bw / 2, base - hw / 2 + 5, "141 GB", "#fff", True, 16, anchor="middle")
            # KV 只有 2.68 GB，柱子里放不下字：画到右边
            f.line(x + bw + 4, base - hw - 1, x + bw + 22, base - hw - 1, PU, 1.4, arrow=False)
            f.t(x + bw + 26, base - hw + 5, "KV %.2f GB" % kv, PU, True, 15)
        else:
            f.t(x + bw / 2, base - hw - hk / 2 + 6, "KV %d GB" % round(kv), "#fff", True, 15,
                anchor="middle")
    # 交叉点标注
    xr = x0 + 4 * bw + 3 * gap + 44
    f.t(xr, top + 40, "batch ≈ %d 时" % round(B_CROSS), INK, True, 16)
    f.t(xr, top + 64, "两块一样重", INK, True, 16)
    f._pan = None
    yb = f.src(py + PH - 20,
               "📌 Llama-3.1-70B config.json：hidden 8192、80 层、64 查询头、8 KV 头、每头 128 维、intermediate 28672、vocab 128256、词表不共享。"
               "参数量按 config 现算 ＝ 70,553,706,496，×2 字节正好等于 safetensors index 的 total_size 141,107,412,992。",
               "本课推导：每 token 的 KV ＝ 2（K、V）× 8 头 × 128 维 × 2 字节 × 80 层 ＝ 327,680 字节；上下文 8K 时每条序列 2.68 GB；"
               "141.1 ÷ 2.68 ≈ 52.6，就是两块一样重的 batch。没算激活值、logits 和读进来就丢的东西；模型实际要切到好几张卡上，这里是全模型合计。",
               "注意力每字节 8 次运算的推导见专题三 §3.1 图下：读 4·S·d 字节、算 4·G·S·d 次，比值 ＝ G。")
    f.save("fig6-s5-amortize.svg", yb + 14)


# ════════════════════════════════════════════════════════════════════
# ② 布局：窄维落进 tile 的两维时被补齐（一个 token、一层的 K＋V，2 个 KV 头 × 128 维）
# ════════════════════════════════════════════════════════════════════
TILE_ROWS32, LANES, WORD = 8, 128, 4             # 一块默认 tile：8 行 × 128 格，每格 32 位
TILE_BYTES = TILE_ROWS32 * LANES * WORD
assert TILE_BYTES == 4096


def rows_per_tile(nbytes):
    """默认 tile 沿次内层装几行：32 位里拼 4/nbytes 个，8 行 32 位 → bf16 16 行、fp8 32 行。"""
    return TILE_ROWS32 * (WORD // nbytes)


def occupied(rows, cols, nbytes, row_unit=None):
    """次内层 rows、最内层 cols 的一块数据实际占的字节：两维各自向上取整到 tile 的边。"""
    ru = row_unit or rows_per_tile(nbytes)
    return math.ceil(rows / ru) * ru * math.ceil(cols / LANES) * LANES * nbytes


KVH, HD = 2, 128
USEFUL = {b: 2 * KVH * HD * b for b in (2, 1)}               # K＋V 两份
assert USEFUL[2] == 1024 and USEFUL[1] == 512
# A：头数放在最内层（lane）：每份是 [128 维（次内层）, 2 头（最内层）]
OCC_A = {b: 2 * occupied(HD, KVH, b) for b in (2, 1)}
assert OCC_A[2] == 64 * 1024 and OCC_A[1] == 32 * 1024
assert OCC_A[2] // USEFUL[2] == 64 and OCC_A[1] // USEFUL[1] == 64
# B：头数放在次内层：每份是 [2 头（次内层）, 128 维（最内层）]，默认 tile
OCC_B = {b: 2 * occupied(KVH, HD, b) for b in (2, 1)}
assert OCC_B[2] == OCC_B[1] == 2 * TILE_BYTES == 8192       # bf16 和 fp8 占一样多：FP8 白省
assert OCC_B[2] // USEFUL[2] == 8 and OCC_B[1] // USEFUL[1] == 16
# C：加一个 packing 维、K 和 V 并成 4 行，用最小 tile（次内层只要 4/字节数 行）
OCC_C = {b: occupied(2 * KVH, HD, b, row_unit=WORD // b) for b in (2, 1)}
assert OCC_C == USEFUL                                        # 一个字节都不补
assert OCC_C[1] * 2 == OCC_C[2]                               # FP8 真省一半


def _carton(f, x, y, w, h, data=None, lane_w=None, col=PU):
    """一只标准箱（＝一块 tile）：箱体 ＋ 箱盖缝 ＋ 封条。data＝从箱底填满的比例；lane_w＝只在左边填一条窄缝。"""
    f.box(x, y, w, h, "#fbf6ee", "#a1887f", 3, 1.6)
    if lane_w is not None:
        f.box(x + 2, y + 2, lane_w, h - 4, col, col, 0)
    elif data:
        f.box(x + 2, y + h - 2 - (h - 4) * data, w - 4, (h - 4) * data, col, col, 1)
    f.line(x, y + 12, x + w, y + 12, "#a1887f", 1.2, arrow=False)
    f.line(x + w / 2, y, x + w / 2, y + 12, "#a1887f", 1.6, arrow=False)


def fig_tile():
    f = Fig(W, "同一份 KV 的三种摆法，画成装箱。TPU 显存按标准箱存，一箱 4 KiB，横 128 格、竖 16 行（bf16），装不满也按整箱算。"
               "要装的是一个 token、一层的 K 和 V，2 个头 × 128 维，bf16 共 1 KiB。"
               "A 让头数占了 128 格那一维，每行只用 2 格，要 16 箱、64 KiB，是真数据的 64 倍。"
               "B 让头数占了按行凑整那一维，每箱 16 行只用 2 行，要 2 箱、8 KiB，是真数据的 8 倍；换成 FP8 还是 2 箱 8 KiB，一个字节都没省。"
               "C 把 K、V 并起来、换成小箱，正好 1 KiB")
    y0 = f.header("布局像装箱：一双袜子装进标准箱，空的也占地方　——　<tspan font-weight=\"700\">FP8 可能一个字节都没省</tspan>",
                  "要装的：一个 token、一层的 K＋V（2 个头 × 128 维，bf16，共 1 KiB）。TPU 显存按「标准箱」存：一箱 4 KiB，横 128 格 × 竖 16 行，装不满也算一整箱",
                  [(PU, "真数据（袜子）"), ("#a1887f", "一只 4 KiB 的标准箱")])
    PW, GAP, PH = 448, 28, 500
    BXW, BXH = 92, 76                          # 一箱的画法：三格里同一个尺寸，箱数可以直接比
    cases = [
        ("A　头数占了「横 128 格」那一维", RD, "每行 128 格只用 2 格"),
        ("B　头数占了「竖着凑行」那一维", OR, "每箱 16 行只用 2 行"),
        ("C　K、V 并起来，换小箱", GR, "小箱正好装满"),
    ]
    nb = 2                                     # 图里只画 bf16
    occ = {"A": OCC_A[nb], "B": OCC_B[nb], "C": OCC_C[nb]}
    nA = OCC_A[nb] // TILE_BYTES
    nB = OCC_B[nb] // TILE_BYTES
    ru = WORD // nb
    nC = OCC_C[nb] // (ru * LANES * nb)
    assert (nA, nB, nC) == (16, 2, 2)
    for k, ((title, col, sub), key) in enumerate(zip(cases, "ABC")):
        px = k * (PW + GAP)
        py = f.panel(px, y0, PW, PH, title, col)
        f.t(px + PW / 2, py + 30, sub, GY, True, 15, anchor="middle")
        area_t, area_h = py + 52, 4 * BXH + 3 * 10
        if key == "A":
            gx = px + (PW - (4 * BXW + 3 * 10)) / 2
            for i in range(nA):
                _carton(f, gx + (i % 4) * (BXW + 10), area_t + (i // 4) * (BXH + 10), BXW, BXH,
                        lane_w=max(BXW * KVH / float(LANES), 2))
            n_s = "%d 箱" % nA
        elif key == "B":
            gx = px + (PW - (2 * BXW + 30)) / 2
            yy = area_t + area_h / 2 - BXH / 2
            for i in range(nB):
                _carton(f, gx + i * (BXW + 30), yy, BXW, BXH, data=KVH / float(rows_per_tile(nb)))
                f.t(gx + i * (BXW + 30) + BXW / 2, yy + BXH + 22, "KV"[i], GY, True, 15, "middle")
            n_s = "%d 箱" % nB
            # 为什么 B 最伤：换 FP8 还是这么多箱
            f.box(px + 24, yy + BXH + 46, PW - 48, 64, "#fce8e6", RD, 6, 1.4)
            f.t(px + PW / 2, yy + BXH + 72, "换成 FP8：还是 %d 箱、%d KiB" % (OCC_B[1] // TILE_BYTES, OCC_B[1] // 1024),
                RD, True, 16, "middle")
            f.t(px + PW / 2, yy + BXH + 96, "一个字节都没省", RD, True, 16, "middle")
        else:
            gx = px + (PW - BXW) / 2
            hh = BXH * ru / float(rows_per_tile(nb))      # 小箱只有标准箱的 1/8 高
            yy = area_t + area_h / 2 - (nC * hh + (nC - 1) * 6) / 2
            for i in range(nC):
                yb_ = yy + i * (hh + 6)
                f.box(gx, yb_, BXW, hh, PU, "#a1887f", 2, 1.6)
            f.t(px + PW / 2, yy + nC * (hh + 6) + 24, "小箱 ＝ 标准箱的 1/%d" % (rows_per_tile(nb) // ru), GY, size=15, anchor="middle")
            n_s = "%d 只小箱" % nC
        ratio = occ[key] // USEFUL[nb]
        occ_s = ("%d KiB" % (occ[key] // 1024))
        f.t(px + PW / 2, py + PH - 64, "%s · %s" % (n_s, occ_s), col, True, 24, "middle", big=True)
        f.t(px + PW / 2, py + PH - 36, "是真数据的 %d 倍" % ratio if ratio > 1 else "一点不浪费", INK, True, 16, "middle")
    f._pan = None
    yb = f.src(y0 + PH + 10,
               "📌 tile 规则：openxla.org/xla/tiled_layout —— TPU 最常见的 tile 是 8×128，对应 8×128 个 32 位的向量寄存器；"
               "16 位类型沿次内层把两个值拼进一个 32 位，8 位类型拼四个。于是默认 tile 的次内层 bf16 是 16 行、fp8 是 32 行，每块都是 8×128×4 ＝ 4,096 字节（本课推导）。图里「标准箱」就是这块默认 tile。",
               "窄头数被补齐、加 packing 维逼 XLA 选最小 tile T(packing,128)、把 K 和 V 沿头维并起来：Ragged Paged Attention，arXiv 2604.15464，§2.1.2、§3.1.1、§3.1.3、§3.5。",
               "本课推导：占用 ＝ ⌈次内层 ÷ 每块行数⌉ × 每块行数 × ⌈最内层 ÷ 128⌉ × 128 × 每元素字节。"
               "A：头数在最内层（128 格），⌈128÷16⌉＝8 块 ×2（K、V）；B：头数在次内层，K、V 各 1 块；C：4 行 ÷ 每块 2 行。"
               "FP8 时：A 32 KiB（仍 64 倍）、B 8 KiB（16 倍）、C 512 B。",
               "XLA 对次内层只有 1–2 的数组有时会改用 2×128 的小 tile（同一份 openxla 文档的 Compact 2nd Minor Layout），"
               "具体选哪种由 XLA 的 layout assignment 决定；进到 kernel 里，向量寄存器仍是固定的 8×128×32 位。本图按默认 tile 画。")
    f.save("fig6-s5-tile.svg", yb + 14)


# ════════════════════════════════════════════════════════════════════
# ③ MLA 的两种算法（DeepSeek-V3，每层、每个缓存 token）
# ════════════════════════════════════════════════════════════════════
V3_HEADS, V3_NOPE, V3_VHD = 128, 128, 128
LAT = NB.V3_KV_LORA + NB.V3_ROPE                              # 576：缓存里每 token 每层的数
# absorbed：所有头共用同一条 576 维的潜向量
ABS_BYTES = LAT * 2
ABS_FLOPS = 2 * V3_HEADS * (LAT + NB.V3_KV_LORA)             # 打分 576 维 ＋ 加权求和 512 维
ABS_AI = ABS_FLOPS / ABS_BYTES
assert ABS_BYTES == 1152 and ABS_FLOPS == 278528 and round(ABS_AI) == 242
# naive：展开成每头的 K（128＋64）和 V（128），config 里 num_key_value_heads ＝ 128
NAIVE_BYTES = V3_HEADS * (V3_NOPE + NB.V3_ROPE + V3_VHD) * 2
NAIVE_FLOPS = 2 * V3_HEADS * (V3_NOPE + NB.V3_ROPE + V3_VHD)
NAIVE_AI = NAIVE_FLOPS / NAIVE_BYTES
assert NAIVE_BYTES == 81920 and NAIVE_FLOPS == 81920 and NAIVE_AI == 1
PREFILL_RATIO = ABS_FLOPS / NAIVE_FLOPS                      # prefill 里每对 (查询, 历史 token) 的运算量之比
assert abs(PREFILL_RATIO - 3.4) < 0.01
assert ABS_AI < NB.V7_RIDGE                                  # 242 仍在屋脊 313 左边，但已经很近


def fig_mla():
    f = Fig(W, "MLA 两种算法的数据流。吸收：缓存里每个 token 只有一条 576 维的压缩 KV，128 个头一起读这一条，"
               "每读 1 字节做 242 次运算，decode 用它。展开：先把这一条复制还原成 128 份、每个头一份，各读各的，"
               "每读 1 字节只做 1 次运算；prefill 有成千上万个查询，复制的开销被摊薄，用它")
    y0 = f.header("MLA 的两种算法：一条大家共读，还是先复制 128 份　——　<tspan font-weight=\"700\">decode 用吸收，prefill 用展开</tspan>",
                  "DeepSeek-V3，一层、一个缓存 token。右边的大数：每从显存读 1 字节，做几次运算",
                  [(GR, "吸收：压缩 KV 不展开"), (OR, "展开：还原成每个头一份")])
    PH = 240
    NHEAD = 6                                       # 128 个头画 6 个，后面写「共 128 个」
    BARW = 180                                      # 576 维画 180 px；展开后每份 K＋V 320 维按同一比例
    CPW = BARW * (V3_NOPE + NB.V3_ROPE + V3_VHD) / float(LAT)
    HX0, HSTEP = 300, 112
    bottoms = []
    for k, (mode, col, title, tag, ai) in enumerate((
            ("abs", GR, "吸收：一条压缩 KV，128 个头一起读", "decode 用它", ABS_AI),
            ("naive", OR, "展开：先复制成 128 份，每个头读自己那份", "prefill 用它", NAIVE_AI))):
        top = y0 + k * (PH + 18)
        py = f.panel(0, top, W, PH, title, col)
        by = py + 54
        f.t(30, by - 12, "缓存里的 1 个 token", GY, True, 15)
        f.box(30, by, BARW, 30, col, col, 4)
        f.t(30 + BARW / 2, by + 21, "%d 维" % LAT, "#fff", True, 16, "middle")
        hy = py + 158                               # 头像顶
        cxs = [HX0 + CPW / 2 + i * HSTEP for i in range(NHEAD)]
        if mode == "abs":
            # 一根总线：同一条数据，所有头从这一根上取
            busy = by + 15
            f.line(30 + BARW, busy, cxs[-1], busy, col, 3, arrow=False)
            for cx in cxs:
                f.line(cx, busy, cx, hy - 6, col, 2, arrow=True)
            f.t(cxs[-1] + 40, busy + 6, "同一条，读一次", col, True, 16)
        else:
            cy = by + 40
            f.line(30 + BARW, by + 15, HX0 - 40, by + 15, col, 2, arrow=False)
            f.path("M %.1f %.1f V %.1f" % (HX0 - 40, by + 15, cy + 4), col, 2, arrow=False)
            f.line(HX0 - 40, cy + 4, cxs[-1] + CPW / 2, cy + 4, col, 2, arrow=False)
            f.t(HX0 - 40 + 4, by + 8, "复制、还原", col, True, 15)
            for cx in cxs:
                f.line(cx, cy + 4, cx, cy + 16, col, 2, arrow=False)
                f.box(cx - CPW / 2, cy + 18, CPW, 24, "#fef7e0", col, 3, 1.6)
                f.t(cx, cy + 36, "%d 维" % (V3_NOPE + NB.V3_ROPE + V3_VHD), col, True, 14, "middle")
                f.line(cx, cy + 42, cx, hy - 6, col, 2, arrow=True)
            f.t(cxs[-1] + CPW / 2 + 16, cy + 36, "128 份", col, True, 16)
        for i, cx in enumerate(cxs):
            f.icon("person", cx - 17, hy, 34, 38, col=col)
        f.t(cxs[-1] + 40, hy + 30, "… 共 %d 个头" % V3_HEADS, GY, True, 15)
        # 右边的大数
        f.t(W - 130, py + 84, "%d" % round(ai), col, True, 60, "middle", big=True)
        f.t(W - 130, py + 116, "次运算 ／ 每读 1 字节", GY, True, 15, "middle")
        f.t(W - 130, py + 160, "→ " + tag, col, True, 18, "middle")
        bottoms.append(py + PH - 30)
    f._pan = None
    yb = f.src(bottoms[-1] + 20,
               "📌 DeepSeek-V3 config.json：128 头、kv_lora_rank 512、qk_nope_head_dim 128、qk_rope_head_dim 64、v_head_dim 128、num_key_value_heads 128。"
               "吸收的做法与「只在 decode 划算」出自 DeepSeek-V2 arXiv 2405.04434 §2.1.2；专题三第五章讲过推导。",
               "本课推导（只数注意力本身的乘加，乘加算 2 次运算，bf16）：吸收 ＝ 2 × 128 × (576 ＋ 512) ＝ 278,528 次，读 576 × 2 ＝ 1,152 字节，比值约 242；"
               "展开 ＝ 2 × 128 × (192 ＋ 128) ＝ 81,920 次，读 128 × 320 × 2 ＝ 81,920 字节，比值 1。prefill 里每对（查询，历史 token）吸收的运算是展开的 3.4 倍。"
               "没算把查询吸收进去、把输出投影回来的那两次矩阵乘（每个查询做一次，跟历史长度无关），也没算 prefill 展开 K/V 的一次性开销（每个历史 token 一次，被所有查询分摊）。",
               "画法：576 维画成一条，展开后每份 K＋V 共 320 维按同一比例；128 个头只画 6 个。")
    f.save("fig6-s5-mla.svg", yb + 14)


# ════════════════════════════════════════════════════════════════════
# ④ decode MoE：多算一遍 vs 读一遍权重（DeepSeek-V3 一个路由专家，TPU v7 单 device）
# ════════════════════════════════════════════════════════════════════
E_PARAMS = 3 * NB.V3_HIDDEN * 2048                           # gate、up、down 三块，moe_intermediate_size 2048
assert E_PARAMS == 44040192
DEV_BW = NB.V7_BW / NB.V7_DEVICES_PER_CHIP                   # 3.685 TB/s
DEV_BF16 = NB.V7_BF16 / NB.V7_DEVICES_PER_CHIP               # 1,153.5 TFLOP/s
DEV_FP8 = NB.V7_FP8 / NB.V7_DEVICES_PER_CHIP                 # 2,307 TFLOP/s
assert abs(DEV_BW / 1e12 - 3.685) < 1e-9 and abs(DEV_BF16 / 1e12 - 1153.5) < 1e-6
NTOK = 256
ROUTED = NTOK * NB.V3_TOPK / NB.V3_EXPERTS                   # 平均每个专家分到的 token
assert ROUTED == 8
DENSE_FLOPS = 2 * NTOK * E_PARAMS
US = 1e-6
T = {}
for name, nb, peak in (("bf16", 2, DEV_BF16), ("fp8", 1, DEV_FP8)):
    T[name] = dict(read=E_PARAMS * nb / DEV_BW / US,
                   dense=DENSE_FLOPS / peak / US,
                   routed=2 * ROUTED * E_PARAMS / peak / US)
assert abs(T["bf16"]["read"] - 23.9) < 0.05 and abs(T["bf16"]["dense"] - 19.5) < 0.1
assert abs(T["fp8"]["read"] - 11.95) < 0.01 and abs(T["fp8"]["dense"] - 9.77) < 0.01
for v in T.values():
    assert v["dense"] < v["read"]                            # 让 256 个 token 全过一遍，仍比读权重快
    assert abs(v["dense"] / v["read"] - NTOK / NB.V7_RIDGE) < 0.01   # 比值 ＝ 256 ÷ 313
BREAKEVEN = NB.V7_RIDGE                                      # 每个 token 多一次乘加只值它带来的 1 字节读权重
HIT = 1 - (1 - NB.V3_TOPK / NB.V3_EXPERTS) ** NTOK           # 均匀路由下，一个专家被至少一个 token 选中的概率
assert HIT > 0.999


def fig_dense():
    f = Fig(W, "DeepSeek-V3 一个路由专家，在 TPU v7 的一个 device 上。两组横条，同一把时间尺子。"
               "BF16：读一遍权重 23.9 微秒，让 256 个 token 全部过一遍这个专家 19.5 微秒，只算路由到的 8 个 token 0.61 微秒。"
               "FP8：读权重 11.95 微秒，256 个 token 全算 9.77 微秒。两种精度下，全算都比读权重快，比值都是 256 除以 313")
    y0 = f.header("decode 的 MoE：多算一遍比读一遍权重还便宜　——　<tspan font-weight=\"700\">那就别排序了，全算</tspan>",
                  "DeepSeek-V3 的一个路由专家（3 块 7168×2048），TPU v7 单 device。条长 ＝ 微秒，同一把尺子；一批 256 个 token，平均每个专家只分到 8 个",
                  [(BL, "从显存读一遍这个专家的权重"), (GR, "256 个 token 全过一遍（没路由到的乘 0）"), ("#9aa0a6", "只算路由到的 8 个")])
    PH = 348
    py = f.panel(0, y0, W, PH, "一个专家、一步 decode", GR)
    x0, sc = 230, 900 / 24.0
    for k, (name, lab) in enumerate((("bf16", "BF16"), ("fp8", "FP8"))):
        yy = py + 34 + k * 140
        f.t(30, yy + 22, lab, INK, True, 17)
        v = T[name]
        rows = [(BL, v["read"], "读权重 %.4g µs" % v["read"]),
                (GR, v["dense"], "全算 %.3g µs" % v["dense"]),
                ("#9aa0a6", v["routed"], "只算 8 个 %.2g µs" % v["routed"])]
        for j, (col, val, txt) in enumerate(rows):
            by = yy + j * 38
            f.box(x0, by, max(val * sc, 3), 28, col, col, 3)
            f.t(x0 + max(val * sc, 3) + 12, by + 20, txt, INK, True, 15)
    f.t(30, py + PH - 46, "两种精度下「全算 ÷ 读权重」都是 256 ÷ %d ≈ %.2f：一批不到 %d 个 token，全算就藏在读权重的时间里"
        % (round(NB.V7_RIDGE), NTOK / NB.V7_RIDGE, round(BREAKEVEN)), GY, size=14)
    yb = f.band(py + PH - 10, "ok", "排序、搬行省掉了，代价是乘 0", [
        "每个专家只分到几行时，按专家排序、把行搬过去再搬回来，比这点矩阵乘还贵；全算把这些步骤换成一次规整的大矩阵乘。",
        "0 × NaN 仍是 NaN：没路由到的 token 要是算出了 NaN 或 inf，乘 0 盖不住，会混进结果。要用选择（where）清掉，不能只靠乘权重 0。",
    ])
    yb = f.src(yb + 10,
               "📌 DeepSeek-V3 config.json：hidden 7168、moe_intermediate_size 2048、256 个路由专家、每 token 选 8 个；权重原生 FP8（e4m3，128×128 分块缩放，缩放因子每专家约 1 万字节，忽略）。",
               "TPU v7 每芯片 7.37 TB/s、FP8 4,614 TFLOP/s（Google Cloud 公开规格），一芯片对软件是 2 个 device，单 device 按一半算：3.685 TB/s、BF16 1,153.5、FP8 2,307 TFLOP/s。",
               "本课推导：一个专家 3 × 7168 × 2048 ＝ 44,040,192 个参数；全算 ＝ 2 × 256 × 44,040,192 次运算。"
               "两边都按峰值算，没算排序和搬行本身的耗时（取决于实现，没有公开的同口径数据）。均匀路由时 256 个 token 下每个专家被至少一个 token 选中的概率 ＞ 99.9%，所以权重本来就要读。",
               "FP8 那一组假设激活值也量化成 FP8 一起乘；只有权重是 FP8、激活值是 BF16 时，算力按 BF16 算，全算 19.5 µs 对读权重 11.95 µs，就不划算了。")
    f.save("fig6-s5-dense.svg", yb + 14)


# ════════════════════════════════════════════════════════════════════
# ⑤ 小 decode 的 AllReduce：延迟问题（GPU 实测，对数轴）
# ════════════════════════════════════════════════════════════════════
AR_N = 4
AR_MSG = NB.V3_HIDDEN * 2                                    # batch 1、bf16、一层一次
assert AR_MSG == 14336
NVL_UNI = 1.8e12 / 2                                         # NVLink 5 每 GPU 1.8 TB/s（双向合计）→ 单向 900 GB/s
AR_BW_NS = 2 * (AR_N - 1) / AR_N * AR_MSG / NVL_UNI * 1e9    # 环形 AllReduce 每卡发出 2(n−1)/n 份
assert abs(AR_BW_NS - 23.9) < 0.1
AR_SOL, AR_NEW, AR_RING = 1.404e3, 2.37e3, 11.0e3            # ns，arXiv 2607.16100
assert round(AR_RING / AR_BW_NS) == 460


def fig_allreduce():
    f = Fig(W, "小 decode 的 AllReduce 花在哪里，对数时间轴。按带宽算，4 张 GB200 之间对一份 14 KiB 的数据做 AllReduce 只要约 24 纳秒；"
               "论文算出的硬件下限是 1.404 微秒；NCCL 环形实测平均 11.0 微秒；论文的低延迟实现 2.37 微秒。实测是带宽项的约 460 倍")
    y0 = f.header("小 decode 的 AllReduce：搬数据只占一根头发丝　——　<tspan font-weight=\"700\">它是延迟问题</tspan>",
                  "对数时间轴，每一格 10 倍。4 张 GB200，NVLink 域内。灰条是本课按带宽算的；其余三条是论文实测或论文算出的下限",
                  [(GY2, "按带宽算（本课推导）"), (RD, "硬件下限"), (OR, "实测")])
    PH = 322
    py = f.panel(0, y0, W, PH, "一次 AllReduce 要多久", RD)
    x0, x1 = 360, 1300
    lo, hi = 1.0, 5.0                                         # 10 ns … 100 µs，以 ns 的 log10 计
    def X(ns):
        return x0 + (math.log10(ns) - lo) / (hi - lo) * (x1 - x0)
    for e, lab in ((1, "10 ns"), (2, "100 ns"), (3, "1 µs"), (4, "10 µs"), (5, "100 µs")):
        f.line(X(10 ** e), py + 36, X(10 ** e), py + 236, LINE, 1, arrow=False)
        f.t(X(10 ** e), py + 258, lab, GY, size=13, anchor="middle")
    rows = [(GY2, AR_BW_NS, "按带宽算：14 KiB，环形", "%.0f ns" % AR_BW_NS),
            (RD, AR_SOL, "硬件下限（论文推算）", "%.3f µs" % (AR_SOL / 1e3)),
            (OR, AR_NEW, "低延迟实现（论文）", "%.2f µs" % (AR_NEW / 1e3)),
            (OR, AR_RING, "NCCL 环形（论文实测）", "%.1f µs" % (AR_RING / 1e3))]
    for j, (col, ns, lab, val) in enumerate(rows):
        by = py + 46 + j * 48
        f.t(x0 - 16, by + 21, lab, INK, True, 15, anchor="end")
        f.box(x0, by, X(ns) - x0, 30, col, col, 3)
        f.t(X(ns) + 12, by + 21, val, INK, True, 15)
    f.t(30, py + PH - 46, "实测 11.0 µs 是带宽项的约 %d 倍：多给带宽几乎没用，要省的是同步的次数" % round(AR_RING / AR_BW_NS),
        GY, size=14)
    yb = f.band(py + PH - 10, "ok", "decode 的 AllReduce 比的是步数，不是带宽", [
        "batch 1 时每层一次只有 hidden × 2 字节（V3 是 14 KiB），时间几乎全花在启动和互相等待上。",
        "所以小 batch 下 TP 切得越细、每层的同步越多，加带宽也救不回来；这条账留到第六节比 TEP 和 DEP。",
    ])
    yb = f.src(yb + 10,
               "📌 Shen 等，Every Microsecond Matters: Achieving Near Speed-of-Light Latency in GPU Collectives，arXiv 2607.16100（2026-07）："
               "4 张 GB200 上小消息平均延迟 NCCL 环形 11.0 µs → 低延迟实现 2.37 µs，Llama-3.1-70B 的 inter-token latency 降 8.7%；AllReduce 的 SoL 下限 1.404 µs。论文没给「小消息」的确切字节范围。",
               "NVLink 5：每 GPU 1.8 TB/s（NVIDIA GB200 NVL72 公开规格，双向合计）。",
               "本课推导：消息 ＝ 7168 × 2 字节 ＝ 14,336 字节（V3、batch 1、bf16）；环形 AllReduce 每卡发出 2(n−1)/n 份（NCCL PERFORMANCE.md，专题五第一节），n＝4 时 1.5 份，按单向 900 GB/s ＝ 约 24 ns。只算带宽项，没算任何启动和同步。",
               "TPU 的 ICI 上同口径的小消息延迟，没有找到公开数据。")
    f.save("fig6-s5-allreduce.svg", yb + 14)


# ════════════════════════════════════════════════════════════════════
# ⑥ fig6-s5-rpa（R38 加，5.2）：一个 kernel 怎么同时吃 prefill 和 decode；KV 页大小这个旋钮
# 数从哪来：RPA arXiv 2604.15464 §3.4–3.5（kernel 收到三段下标：[0,i) 纯 decode、[i,j) 定长分块 prefill、
#   [j,k) 长度不定的混合段，每段按 case 派给对应的预编译版本，每份各配一组块大小；R41 按原文改段序与混合段含义）；页 16 → 256 token、并发 512 时这个 kernel 的 decode 一步 428 → 283 µs：
#   Google Developers Blog「Systems Engineering Playbook: Optimizing Qwen 3.5-397B MoE on Ironwood (TPU7x)」（台账 5.2 行）。
# ⛔ 刻意没画：段怎么切、每段几条请求、块大小取多少 —— 全是示意；页的格数也是示意（256 ÷ 16 ＝ 16 只是比例）。
# ════════════════════════════════════════════════════════════════════
RPA_PAGE = [(16, 428), (256, 283)]                            # (页大小 token, decode 一步 µs)


def fig_rpa():
    f = Fig(W, "左半：同一批里混着 decode 请求和 prefill 请求，decode 每条只有 1 个新 token，prefill 每条一段多个 token。"
               "RPA 把这一批按下标切成三段：全是 decode 的一段派给纯 decode 版，定长分块的 prefill 派给纯 prefill 版，长度不定的其余请求派给混合版；"
               "三份版本都是预编译好的，各配一组块大小。"
               "右半：KV 页大小。页 16 个 token 时，读同样的 KV 要查很多页，并发 512 时这个 kernel 的 decode 一步 428 微秒；"
               "页调到 256 个 token，页数少，降到 283 微秒")
    y0 = f.header("一个 kernel 装下 prefill 和 decode　——　<tspan font-weight=\"700\">先分段，再派给三份预编译版本</tspan>",
                  "左：一批怎么拆给三份版本（示意）；右：KV 页大小这个旋钮（Qwen3.5-397B，TPU7x）",
                  [(GR, "decode 请求：1 个新 token"), (OR, "prefill 请求：一段多个 token"), (PU, "KV 页")])
    PH = 300
    LW = 820
    # ── 左：分段派发 ──────────────────────────────────────────────
    py = f.panel(0, y0, LW, PH, "同一批，按成分分三段", INK)
    D, P = 26, 92                                             # decode、prefill 块宽（示意）
    segs = [("纯 decode 版", "D" * 5, "每条 1 token · 自带块大小"),
            ("纯 prefill 版", "PP", "定长分块 · 自带块大小"),
            ("混合版", "DpP", "长度不定 · 自带块大小")]      # 段序按原文 [0,i)／[i,j)／[j,k)
    WD = {"D": D, "P": P, "p": 50}                            # p：长度不定的一段 prefill（示意）
    gap_seg, gap = 34, 5
    widths = [sum(WD[c] for c in pat) + gap * (len(pat) - 1) for _, pat, _ in segs]
    x = (LW - (sum(widths) + gap_seg * 2)) / 2.0
    by = py + 52
    f.t(24, by + 22, "一批", INK, True, 16)
    kw, ky = 214, py + 168
    for i, ((kname, pat, ksub), w) in enumerate(zip(segs, widths)):
        xx = x
        for c in pat:
            bw = WD[c]
            col, tint = (GR, "#e6f4ea") if c == "D" else (OR, "#fde7d4")
            f.box(xx, by, bw, 34, tint, col, 3, 1.4)
            xx += bw + gap
        # 段的方括号
        f.path([(x, by + 44), (x, by + 52), (x + w, by + 52), (x + w, by + 44)], GY2, 1.4, arrow=False)
        kx = 30 + i * (kw + 34)
        f.path([(x + w / 2, by + 52), (x + w / 2, ky - 40), (kx + kw / 2, ky - 40), (kx + kw / 2, ky - 4)],
               GY2, 1.4, arrow=True)
        col = {0: GR, 1: OR, 2: INK}[i]
        f.box(kx, ky, kw, 74, "none", col, 8, 1.6)
        f.t(kx + kw / 2, ky + 30, kname, col, True, 17, "middle")
        f.t(kx + kw / 2, ky + 56, ksub, GY, size=14, anchor="middle")
        x += w + gap_seg
    # ── 右：页大小 ────────────────────────────────────────────────
    RX, RW = LW + 16, W - LW - 16
    py = f.panel(RX, y0, RW, PH, "KV 页大小：另一个便宜的旋钮", PU)
    SC = 0.85                                                  # px／µs
    for j, (pg, us) in enumerate(RPA_PAGE):
        yy = py + 30 + j * 122
        f.t(RX + 20, yy + 18, "页 %d 个 token" % pg, INK, True, 16)
        if pg == 16:
            f.t(RX + 150, yy + 18, "页太小：查索引让向量单元频频停顿", RD, size=14)
        n = 256 // pg                                          # 同一段 KV 切成几页（示意比例）
        cell = 320.0 / n
        for k in range(n):
            f.box(RX + 20 + k * cell, yy + 30, cell - (2 if n > 1 else 0), 22, "#f3e8fd", PU, 2, 1.2)
        col = RD if pg == 16 else GR
        f.box(RX + 20, yy + 64, us * SC, 26, col, col, 3)
        f.t(RX + 20 + us * SC + 10, yy + 83, "%d µs" % us, col, True, 16)
    f.t(RX + 20, py + 262, "并发 512 · 这个 kernel 的 decode 一步，不是整步", GY, size=14)
    f._pan = None
    yb = f.src(y0 + PH + 6,
               "分段与三份预编译版本：Ragged Paged Attention，arXiv 2604.15464 §3.4–3.5，段序按原文：纯 decode、定长分块 prefill、长度不定的混合段。每段几条请求、块大小都是示意。",
               "页大小：Google Developers Blog「Systems Engineering Playbook: Optimizing Qwen 3.5-397B MoE on Ironwood (TPU7x)」，"
               "原文 decode step latency under Concurrency-512 from 428µs to 283µs（kernel-level）。页的格数只表示 16 与 256 的比例。")
    f.save("fig6-s5-rpa.svg", yb + 14)


fig_amortize()
fig_tile()
fig_mla()
fig_dense()
fig_allreduce()
fig_rpa()
