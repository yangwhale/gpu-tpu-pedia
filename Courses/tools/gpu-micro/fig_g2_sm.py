# -*- coding: utf-8 -*-
"""图 G-2 —— 一个 SM 显微镜展开（NVIDIA Blackwell，计算能力 10.0）。

这是整份文档的中心图，对标 TPU 全景图里「TensorCore 内部展开」那一块：
把 SM 的四个处理块全部画开，每个单元标出个数、容量、以及数字的来源等级。
"""
from common import Fig, BL, RD, YL, GN, PU, INK, SUB, LINE, GREY, FILL

W = 1400
COLW, GAP, X0 = 323, 12, 36
COLS = [X0 + i * (COLW + GAP) for i in range(4)]

# ── 竖直节奏（子核内部，相对块顶 T 的偏移与高度）─────────────────────────
HDR, L0, SCH, SLOT, RF, CUDA, TC, TMEM, LDST = (
    (0, 24), (28, 20), (52, 38), (94, 46), (144, 36), (184, 92),
    (280, 92), (376, 40), (420, 36))
SUBH = 460

SMTOP = 82
ICACHE_Y = 116
SUB_T = 168
SUB_B = SUB_T + SUBH                     # 614
SMEM_Y = SUB_B + 12                      # 626
SMEM_H = 84
TMA_Y = SMEM_Y + SMEM_H + 8              # 700
TMA_H = 50
SMBOT = TMA_Y + TMA_H + 14               # 764
SUMY = SMBOT + 26                        # 790
H = SUMY + 176                           # 966


def build():
    f = Fig(W, H, "NVIDIA Blackwell 一个 SM 显微镜展开：四个处理块、寄存器堆、"
                  "Tensor Core 与 TMEM、共享内存与 TMA")
    f.title("一个 SM 的显微镜展开 —— NVIDIA Blackwell（计算能力 10.0）"
            "　·　4 个处理块 · 128 CUDA Core · 4 Tensor Core · 256 KiB TMEM",
            "灰字＝官方未标")
    f.legend([(PU, "指令 · 调度"), (BL, "通用向量：CUDA Core / 寄存器"),
              (RD, "张量通路：Tensor Core / TMEM"), (YL, "访存单元"),
              (GN, "整个 SM 共享"), (GREY, "官方未公布个数")])

    # ── SM 外框 ────────────────────────────────────────────────────────
    f.rect(20, SMTOP, 1360, SMBOT - SMTOP, BG_ := "#f8f9fa", INK, 2, 12)
    f.t(36, SMTOP + 22, "一个 SM（Streaming Multiprocessor）", "sec")
    f.t(320, SMTOP + 22,
        "B200 全片 <tspan font-weight=\"700\" fill=\"#202124\">148 个</tspan>（物理 160，每 die 80、启用 74）"
        "　·　Blackwell Ultra 全片 160 个　·　最多 64 个 warp、32 个线程块同时驻留", "sm")

    # ── 顶部：L1 指令缓存 + 工作分发 ───────────────────────────────────
    f.rect(36, ICACHE_Y, 1328, 40, FILL[PU], PU, 1.6, 8)
    f.t(50, ICACHE_Y + 17, "L1 指令缓存　·　线程块从 GigaThread 引擎派进来，落在这个 SM 上就不再迁走", "box", PU)
    f.t(50, ICACHE_Y + 33,
        "一个线程块（CTA）整体钉在<tspan font-weight=\"700\" fill=\"#202124\">一个 SM</tspan> 上；块里的每个 warp 再被分派到下面四个处理块中的"
        "<tspan font-weight=\"700\" fill=\"#202124\">某一个</tspan>，同样不迁走。这两条「钉死」是后面所有容量除以 4 的前提。", "xs")

    for i, x in enumerate(COLS):
        f.line(x + COLW / 2, ICACHE_Y + 40, x + COLW / 2, SUB_T - 2, PU, 1.6, "aP")
        _subcore(f, x, SUB_T, i)

    # ── 底部共享：L1 / 共享内存 ────────────────────────────────────────
    f.rect(36, SMEM_Y, 1328, SMEM_H, FILL[GN], GN, 1.8, 8)
    f.t(50, SMEM_Y + 20, "L1 数据缓存 ＋ 共享内存　—— 同一块 SRAM，四个处理块共用", "box", GN)
    f.t(520, SMEM_Y + 20, "这是 GPU 上「编译器/程序员能显式管」的那一层，对应 TPU 的 VMEM", "sm")
    f.t(50, SMEM_Y + 38,
        "合计 <tspan class=\"num\" fill=\"#1e8e3e\">256 KiB</tspan> / SM　·　其中可划给共享内存最多 "
        "<tspan class=\"num\" fill=\"#1e8e3e\">228 KiB</tspan>（单个线程块最多 227 KiB，CUDA 留 1 KiB）", "xs")
    f.t(50, SMEM_Y + 55,
        "可选切分：0 / 8 / 16 / 32 / 64 / 100 / 132 / 164 / 196 / 228 KiB　·　静态声明仍限 48 KiB，超过要显式 opt-in", "xs")
    f.t(50, SMEM_Y + 72,
        "<tspan fill=\"#9aa0a6\">分 32 个 bank、每 bank 每周期 4 B —— 历代如此，Blackwell 官方文档未重申</tspan>", "xs")
    f.t(900, SMEM_Y + 38,
        "同一个 cluster 里别的线程块能直接读写这块内存", "xs")
    f.t(900, SMEM_Y + 55,
        "—— <tspan font-weight=\"700\" fill=\"#202124\">分布式共享内存</tspan>，Hopper 引入", "xs")
    f.t(900, SMEM_Y + 72,
        "实测 L1 命中约 <tspan class=\"num\" fill=\"#5f6368\">39 周期</tspan>"
        "　<tspan fill=\"#9aa0a6\">（第三方实测，非官方）</tspan>", "xs")

    # ── 底部共享：TMA / 纹理 / 原子 ────────────────────────────────────
    f.rect(36, TMA_Y, 1328, TMA_H, "#fff", YL, 1.6, 8)
    f.t(50, TMA_Y + 18, "TMA　张量内存加速器", "box", "#b06000")
    f.t(210, TMA_Y + 18,
        "一条指令搬一整块多维张量：软件填一个<tspan font-weight=\"700\" fill=\"#202124\">描述符</tspan>（基址 · 各维长度 · 各维步长 · 分块形状），"
        "硬件自己算地址、自己搬。", "xs")
    f.t(210, TMA_Y + 34,
        "这就是 GPU 版的 DMA 引擎，和 TPU 那 17 条 DMA 通道是同一类东西 —— 都是"
        "「把地址生成从计算单元手里拿走」。Hopper 引入，Blackwell 沿用。", "xs")
    f.rect(1030, TMA_Y + 10, 150, 30, "#fff", SUB, 1.2, 5)
    f.t(1105, TMA_Y + 29, "纹理 / 采样单元", "xs", SUB, "middle")
    f.rect(1190, TMA_Y + 10, 160, 30, "#fff", SUB, 1.2, 5)
    f.t(1270, TMA_Y + 23, "共享内存原子单元", "xs", SUB, "middle")
    f.t(1270, TMA_Y + 35, "实测 32 次/周期/SM", "xxs", GREY, "middle")

    _summary(f)
    return f.out()


# ══════════════════════════════════════════════════════════════════════
def _subcore(f, x, T, idx):
    xi, wi = x + 11, COLW - 22

    f.rect(x, T, COLW, SUBH, "#fff", INK, 1.6, 9)

    # 标题条
    y, h = T + HDR[0], HDR[1]
    f.rect(x, y, COLW, h, INK, rx=9)
    f.rect(x, y + h - 9, COLW, 9, INK, rx=0)
    f.t(x + 11, y + 17, f"处理块 {idx}", "box", "#fff")
    f.t(x + COLW - 11, y + 17, "sub-core / partition", "xs", "#c9cbcf", "end")

    # L0 指令缓存
    y, h = T + L0[0], L0[1]
    f.rect(xi, y, wi, h, FILL[PU], PU, 1.1, 4)
    f.t(xi + 8, y + 14, "L0 指令缓存", "xs", PU)

    # Warp 调度器
    y, h = T + SCH[0], SCH[1]
    f.rect(xi, y, wi, h, FILL[PU], PU, 1.5, 5)
    f.t(xi + 8, y + 15, "Warp 调度器 ×1　＋　分发单元", "lbl", PU)
    f.t(xi + 8, y + 30, "每周期从驻留 warp 里挑 1 个就绪的，发 1 条指令", "xs")

    # 驻留 warp 槽
    y, h = T + SLOT[0], SLOT[1]
    f.rect(xi, y, wi, h, "#fff", PU, 1.1, 5, "3,2")
    f.t(xi + 8, y + 14, "驻留 warp 槽 ×16", "lbl", PU)
    f.t(xi + 118, y + 14, "＝ 64 warp/SM ÷ 4", "xs")
    for r in range(2):
        for c in range(8):
            f.rect(xi + 8 + c * 36, y + 20 + r * 12, 33, 10,
                   FILL[PU] if (r * 8 + c) % 3 else "#fff", PU, 0.7, 2)

    # 寄存器堆
    y, h = T + RF[0], RF[1]
    f.rect(xi, y, wi, h, FILL[BL], BL, 1.5, 5)
    f.t(xi + 8, y + 15, "寄存器堆　16,384 × 32 bit ＝ 64 KiB", "lbl", BL)
    f.t(xi + 8, y + 30, "＝ 64K 寄存器/SM ÷ 4　·　单线程上限 255 个", "xs")

    # CUDA Core ×32
    y, h = T + CUDA[0], CUDA[1]
    f.rect(xi, y, wi, h, "#fff", BL, 1.5, 5)
    f.t(xi + 8, y + 15, "CUDA Core ×32", "lbl", BL)
    f.t(xi + 108, y + 15, "FP32 / INT32 统一单元", "xs")
    for r in range(4):
        for c in range(8):
            f.rect(xi + 8 + c * 36, y + 21 + r * 13, 33, 11, FILL[BL], BL, 0.7, 2)
    f.t(xi + 8, y + 86, "一条指令 → 32 条 lane 同时算 ＝ 一个 warp 一拍做完", "xs")

    # Tensor Core
    y, h = T + TC[0], TC[1]
    f.rect(xi, y, wi, h, FILL[RD], RD, 2, 6)
    f.t(xi + 8, y + 17, "Tensor Core ×1　（第 5 代）", "box", RD)
    f.rect(xi + 8, y + 24, wi - 16, 22, "#fff", RD, 1, 4)
    f.t(xi + 15, y + 39, "1,024 次乘加 / 周期", "numb", RD)
    f.t(xi + 178, y + 40, "← 推导值，链见图末", "xs")
    f.t(xi + 8, y + 60, "操作数<tspan font-weight=\"700\" fill=\"#202124\">不走寄存器堆</tspan>，走下面的 TMEM", "xs")
    f.t(xi + 8, y + 74, "支持两个 SM 配对做<tspan font-weight=\"700\" fill=\"#202124\">同一次</tspan> MMA，共用操作数", "xs")
    f.t(xi + 8, y + 87, "吃的最小矩阵只有 K=16 —— 从来不是 128×128", "xs", RD)

    # TMEM 分区
    y, h = T + TMEM[0], TMEM[1]
    f.rect(xi, y, wi, h, "#fff", RD, 1.5, 5, "3,2")
    f.t(xi + 8, y + 15, f"TMEM 分区　512 列 × 32 lane ＝ 64 KiB", "lbl", RD)
    f.t(xi + 8, y + 31,
        f"全 SM 256 KiB 的 1/4　·　只有本块的 warp 能碰 lane {idx*32}–{idx*32+31}", "xs")

    # 访存 / 特殊函数
    y, h = T + LDST[0], LDST[1]
    f.rect(xi, y, (wi - 6) / 2, h, FILL[YL], YL, 1.3, 5)
    f.t(xi + 7, y + 15, "LD / ST <tspan fill=\"#9aa0a6\">×8</tspan>", "lbl", "#b06000")
    f.t(xi + 7, y + 29, "访存单元", "xxs", GREY)
    f.rect(xi + (wi + 6) / 2, y, (wi - 6) / 2, h, FILL[YL], YL, 1.3, 5)
    f.t(xi + (wi + 6) / 2 + 7, y + 15, "SFU <tspan fill=\"#9aa0a6\">×4</tspan>", "lbl", "#b06000")
    f.t(xi + (wi + 6) / 2 + 7, y + 29, "exp · rcp · rsqrt", "xxs", GREY)


# ══════════════════════════════════════════════════════════════════════
def _summary(f):
    f.t(20, SUMY, "把上面四列乘回去 —— 整个 SM 的账", "sec")
    y = SUMY + 12
    cells = [
        ("CUDA Core", "128", "32 × 4", BL),
        ("Tensor Core", "4", "1 × 4", RD),
        ("寄存器堆", "256 KiB", "64 KiB × 4", BL),
        ("TMEM", "256 KiB", "64 KiB × 4", RD),
        ("L1 ＋ 共享内存", "256 KiB", "四块共用", GN),
        ("最多驻留 warp", "64", "16 × 4", PU),
        ("最多驻留线程", "2,048", "64 warp × 32", PU),
        ("每周期发指令", "4 条", "1 × 4 个调度器", PU),
    ]
    cw = (1360 - 7 * 8) / 8
    for i, (k, v, d, c) in enumerate(cells):
        x = 20 + i * (cw + 8)
        f.rect(x, y, cw, 62, "#fff", c, 1.4, 8)
        f.t(x + 10, y + 17, k, "xs", c)
        f.t(x + 10, y + 39, v, "numb", c)
        f.t(x + 10, y + 54, d, "xxs")

    y2 = y + 74
    f.rect(20, y2, 1360, 84, FILL[RD], RD, 1.8, 9)
    f.t(34, y2 + 20, "唯一一个不是官方直接给出的数：Tensor Core 每周期 1,024 次乘加", "box", RD)
    f.t(34, y2 + 38,
        "<tspan class=\"mono\" fill=\"#202124\">每 SM 每周期张量 FLOP ＝ 峰值 FLOPS ÷ SM 数 ÷ 时钟</tspan>"
        "　—— 拿两个规格全公开的老世代先验模型：", "xs")
    f.t(34, y2 + 55,
        "<tspan class=\"mono\" fill=\"#202124\">A100：312 TF ÷ 108 SM ÷ 1.410 GHz ＝ 2,048.9</tspan>"
        "　　<tspan class=\"mono\" fill=\"#202124\">H100：989.4 TF ÷ 132 SM ÷ 1.830 GHz ＝ 4,095.9</tspan>"
        "　　两个都精确落在 2 的幂上 → 模型成立。", "xs")
    f.t(34, y2 + 72,
        "顺着推 Blackwell ＝ <tspan font-weight=\"700\" fill=\"#d93025\">8,192 FLOP/周期/SM</tspan>"
        "，除以 4 个 Tensor Core、再除以 2（一次乘加算 2 个 FLOP）"
        "＝ <tspan font-weight=\"700\" fill=\"#d93025\">1,024 次乘加/周期/Tensor Core</tspan>。"
        "代回 B200：8,192 × 148 SM × 1.83 GHz ＝ 2.22 PFLOPS，官方口径 2.25 —— 差 1.4%，缺的是官方没公布的确切时钟。", "xs")


if __name__ == "__main__":
    import io, sys
    io.open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/g2.svg", "w",
            encoding="utf-8").write(build())
    print("ok", H)
