# -*- coding: utf-8 -*-
"""图 G-4 —— Tensor Core 一次吃多大矩阵：四代指令 ＋ 真实比例叠图。

这张图专门回答一个常见误解：「GPU 的 Tensor Core 是不是也像 TPU 的 MXU 那样，
最小就得喂 128×128？」不是。收缩维 K 从 Volta 到 Blackwell 一直是 16，
而且细粒度那条路在 Blackwell 上不但没被砍掉，还长出了新能力（块量化）。
"""
from common import Fig, para, BL, RD, YL, GN, PU, TL, INK, SUB, GREY, FILL

W = 1400
TOP = 84
GEN_T = TOP + 26
GENH, GENG = 92, 8
NGEN = 4
B_T = GEN_T + NGEN * (GENH + GENG) + 34      # 叠图区标题
CANV = 300                                    # TPU MXU 256×256 的画布边长
SC = CANV / 256.0                             # 每个元素 1.172 px
CX, CY = 70, B_T + 56
H = CY + CANV + 186

GEN = [
    (SUB, "Volta", "第 1 代", "8 个线程",
     "<b>8 线程一组</b>的 MMA 单元；FP16 乘、FP32 累加",
     "NVIDIA 原话：<b>8-thread MMA units</b>",
     "寄存器"),
    (BL, "Ampere", "第 3 代", "32 个线程 ＝ 1 个 warp",
     "<code>mma.sync.aligned.m16n8k16</code>　—— 一个 warp 一条指令",
     "NVIDIA 原话：<b>Full warp-wide MMA</b>，加 BF16 与 TF32",
     "寄存器"),
    (PU, "Hopper", "第 4 代", "128 个线程 ＝ 1 个 warp 组",
     "<code>wgmma.mma_async.m64nNk16</code>，N 从 8 到 256 可选",
     "NVIDIA 原话：<b>Warp-group MMA across 128 threads</b>。<r>只在 Hopper 有</r>（要 sm_90a）",
     "A 在寄存器或共享内存，B 在共享内存"),
    (RD, "Blackwell", "第 5 代", "256 个线程 ＝ 两个 SM 配对",
     "<code>tcgen05.mma</code>：M ＝ 64 或 128（配对时 256），N ＝ 8…256 步长 8，<r>K 仍然只有 16</r>",
     "NVIDIA 原话：<b>dual-thread-block MMA</b>，两个 SM 共用操作数",
     "<r>操作数与累加器都在 TMEM</r>，不占寄存器"),
]

# (标签, M, N, 颜色, 说明)
SHAPES = [
    ("tcgen05　m128 n256", 128, 256, RD),
    ("wgmma　m64 n256", 64, 256, PU),
    ("mma.sync　m16 n8", 16, 8, BL),
]


def build():
    f = Fig(W, H, "Tensor Core 四代 MMA 指令与真实比例形状叠图：收缩维 K 始终是 16，"
                  "与 TPU 的 256×256 脉动阵列对照")
    f.title("Tensor Core 一次吃多大矩阵　—— 四代演进，以及一张真实比例的叠图",
            "全部出自 PTX ISA")
    f.legend([(BL, "warp 级（32 线程）"), (PU, "warp 组级（128 线程）"),
              (RD, "双 SM 级（256 线程）"), (GN, "TPU MXU，同一把尺子")])

    f.t(20, TOP + 18, "四代演进　—— 变的是「一次动员多少线程」，不变的是「K ＝ 16」", "sec")
    for i, (c, name, gen, th, ins, quote, where) in enumerate(GEN):
        y = GEN_T + i * (GENH + GENG)
        f.rect(20, y, 1360, GENH, "#fff", c, 1.6, 9)
        f.rect(20, y, 176, GENH, FILL[c], rx=9)
        f.rect(187, y, 9, GENH, FILL[c], rx=0)
        f.t(34, y + 26, name, "sec", c)
        f.t(34, y + 45, gen, "xs")
        f.t(34, y + 66, th, "lbl", INK)
        f.t(34, y + 82, "参与一次 MMA", "xxs")

        para(f, 214, y + 26, 700, ins, "lbl", 18)
        para(f, 214, y + 62, 700, quote, "xs", 15)

        f.rect(940, y + 14, 424, GENH - 28, "#fff", SUB, 1.1, 6, "3,2")
        f.t(954, y + 32, "操作数住在哪", "xs")
        para(f, 954, y + 52, 396, where, "lbl", 16)

        _mini(f, 830, y + 20, i, c)

    _overlay(f)
    return f.out()


def _mini(f, x, y, i, c):
    """右侧小示意：一次 MMA 动员多少条 lane。"""
    n = [8, 32, 128, 256][i]
    cols, cell = 32, 2.6
    f.t(x, y + 8, f"{n} 条 lane", "xxs", c)
    for k in range(min(n, 256)):
        r, cc = divmod(k, cols)
        f.rect(x + cc * (cell + 1), y + 14 + r * (cell + 1), cell, cell, c, rx=0.5)


# ══════════════════════════════════════════════════════════════════════
def _overlay(f):
    f.t(20, B_T, "同一把尺子上叠一遍　—— 大方框是 TPU 一个 MXU 的 256×256，"
                 "里面三块是 GPU 三条指令各自的输出矩阵", "sec")
    f.t(20, B_T + 20, "每个元素画 1.17 像素，三块和大框用的是同一个比例，没有任何缩放作弊。", "sm")

    # TPU MXU 底框
    f.rect(CX, CY, CANV, CANV, FILL[GN], GN, 2, 4)
    f.t(CX + CANV - 8, CY + CANV - 10, "TPU v7 一个 MXU　256 × 256", "lbl", GN, "end")
    f.t(CX + CANV - 8, CY + CANV - 24, "65,536 个乘加单元，一条指令喂满", "xs", GN, "end")

    lead = [(CY + 26, "128 × 256"), (CY + 100, "64 × 256"), (CY + 176, "16 × 8")]
    for (label, m, n, c), (ly, dims) in zip(SHAPES, lead):
        w, h = n * SC, m * SC
        f.rect(CX, CY, w, h, "none", c, 2.2, 2)
        f.rect(CX, CY, w, h, c, rx=2, extra=' opacity="0.10"')

    # 标注（拉到右边，避免压在图上）
    ann = [
        (RD, "tcgen05.mma　M×N ＝ 128 × 256", 128, 256,
         "Blackwell 最大的一条。两个 SM 配对时 M 可以到 256 —— 那时刚好铺满这个大方框的一半。"),
        (PU, "wgmma　M×N ＝ 64 × 256", 64, 256,
         "Hopper 的 warp 组指令。只在 Hopper 上有，Blackwell 换成了上面那条。"),
        (BL, "mma.sync　M×N ＝ 16 × 8", 16, 8,
         "在这张图上它只有 19 × 9 像素 —— 就是左上角那个圈住的小蓝点。从 Ampere 一路留到今天的 warp 级指令，"
         "<r>Blackwell 上不但没被砍，还是块量化唯一挂得上的地方。</r>"),
    ]
    # 最小那块只有 19×9 像素，加个圈+引出线，别让人以为图画漏了
    f.raw(f'<circle cx="{CX+10}" cy="{CY+8}" r="26" fill="none" stroke="{BL}" '
          f'stroke-width="1.6" stroke-dasharray="4,3"/>')
    f.path(f"M{CX+34} {CY+2} L{CX+CANV+40} {CY+196}", BL, 1.4, dash="4,3")

    ax = CX + CANV + 60
    for i, (c, ttl, m, n, note) in enumerate(ann):
        y = CY + i * 96
        f.rect(ax, y, 860, 82, "#fff", c, 1.5, 8)
        f.rect(ax + 14, y + 18, 22, 22, c, rx=3, extra=' opacity="0.18"')
        f.rect(ax + 14, y + 18, 22, 22, "none", c, 2, 3)
        f.t(ax + 48, y + 26, ttl, "box", c)
        f.t(ax + 48, y + 44, f"输出块占 {m*n:,} 个位置，是 TPU 那个 256×256 的 "
                             f"<tspan font-weight=\"700\" fill=\"#202124\">1/{65536//(m*n)}</tspan>", "xs")
        para(f, ax + 48, y + 64, 790, note, "xs", 15)

    # K 维条
    ky = CY + CANV + 26
    f.t(20, ky, "还有更关键的一维：收缩维 K　—— 上面画的是输出矩阵，K 才决定「一次能吃多深」", "sec")
    f.rect(20, ky + 12, 1360, 132, FILL[RD], RD, 1.8, 9)
    yy = para(f, 36, ky + 34, 1320,
              "M 和 N 决定输出块有多大，<b>K 决定一次乘加链有多长</b>。GPU 这一维从 Ampere 到 Blackwell "
              "<r>一直是 16</r>：fp16/bf16 是 K=16，fp8 是 K=32，fp4 是 K=64 —— 换算成位宽全都是同样的 16 个 32-bit 字。"
              "换句话说，四代过去，Tensor Core 一口吃进去的<b>深度</b>从来没变过。", "lbl", 19)

    by = yy + 12
    f.rect(36, by, 16 * SC, 20, FILL[BL], BL, 1.8, 2)
    f.t(36 + 16 * SC + 10, by + 15, "GPU Tensor Core 的 K ＝ 16　← 就这么窄", "lbl", BL)
    f.rect(36, by + 26, 256 * SC, 20, FILL[GN], GN, 1.8, 2)
    f.t(36 + 256 * SC + 10, by + 41, "TPU MXU 的收缩边 ＝ 256　—— 同一把尺子，差 16 倍", "lbl", GN)

    para(f, 760, by + 6, 600,
         "这就是为什么注意力里 <code>head_dim=128</code> 这件事"
         "<r>打 TPU 却不打 GPU</r>：128 撞上 TPU 的 256 收缩边，只喂满一半；"
         "而 128 是 16 的整数倍，对 GPU 来说喂得满满的。", "lbl", 19)


if __name__ == "__main__":
    import io, sys
    io.open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/g4.svg", "w",
            encoding="utf-8").write(build())
    print("ok", H)
