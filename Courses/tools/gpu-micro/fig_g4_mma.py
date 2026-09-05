# -*- coding: utf-8 -*-
"""图 G-4 —— Tensor Core 一次吃多大矩阵：五代指令 ＋ 真实比例叠图。

这张图专门回答一个常见误解：「GPU 的 Tensor Core 是不是也像 TPU 的 MXU 那样，
最小就得喂 128×128？」不是。收缩维 K 在 Ampere 定到 16 之后就再没变过，
而且细粒度那条路在 Blackwell 上不但没被砍掉，还长出了新能力（块量化）。
"""
from common import Fig, para, BL, RD, YL, GN, PU, TL, INK, SUB, GREY, FILL

W = 1400
TOP = 84
GEN_T = TOP + 26
GENH, GENG = 92, 8
NGEN = 5
B_T = GEN_T + NGEN * (GENH + GENG) + 34      # 叠图区标题
CANV = 300                                    # TPU MXU 256×256 的画布边长
SC = CANV / 256.0                             # 每个元素 1.172 px
CX, CY = 70, B_T + 56
H = CY + CANV + 228

GEN = [
    (SUB, "Volta", "第 1 代", "8 个线程",
     "<code>mma.sync.m8n8k4</code>　—— 一个 quad-pair 八条 lane 凑一次 MMA",
     "PTX 目标注记：<b>m8n8k4 requires sm_70</b>，并注明「为 sm_70 优化，别的架构上会明显更慢」",
     "寄存器"),
    (YL, "Turing", "第 2 代", "32 个线程 ＝ 1 个 warp",
     "<code>mma.sync.m16n8k8</code>　—— warp 级形状从这一代进 PTX",
     "同一页还写着：<b>u8/s8 的 m8n8k16、u4/s4 的 m8n8k32、b1 的 m8n8k128 都要 sm_75</b>"
     " —— <b>整数与亚字节精度是从 Turing 开始的</b>",
     "寄存器"),
    (BL, "Ampere", "第 3 代", "32 个线程 ＝ 1 个 warp",
     "<code>mma.sync.aligned.m16n8k16</code>　—— K 从 8 加倍到 16，加 BF16 与 TF32",
     "PTX 目标注记：<b>m16n8k16 requires sm_80</b>。这条形状一路活到今天",
     "寄存器"),
    (PU, "Hopper", "第 4 代", "128 个线程 ＝ 1 个 warp 组",
     "<code>wgmma.mma_async.m64nNk16</code>，N 从 8 到 256 可选",
     "NVIDIA 原话：<b>Warp-group MMA across 128 threads</b>。<r>只在 Hopper 有</r>（要 sm_90a）",
     "A 在寄存器或共享内存，B 在共享内存"),
    # ⛔⛔ 2026-09-05：这一列原来写「256 个线程 ＝ 两个 SM 配对」，
    #    而同一格右边又写着「操作数与累加器都在 TMEM，不占寄存器」——&nbsp;**两句打架**。
    #    tcgen05.mma 是**单线程发射**的异步指令（PTX ISA：issued by a single thread；
    #    warp 级的只有 tcgen05.alloc 之类）；cta_group::2 说的是**哪两个 CTA 供数**，
    #    不是「256 个线程一起发」。
    # ⭐ 更要紧的是这张图的纵轴：前四代讲的是「一次动员多少线程」，
    #    而第 5 代**恰好把这条趋势掉了头** —— 发射退回单线程，操作数改由 TMEM 供给。
    #    台下老手一问「wgmma 那 128 线程是真一起发的，tcgen05 是一个线程发的，
    #    你这条线在第五代是不是断了」，旧版答不上。现在把反转写在格子里。
    (RD, "Blackwell", "第 5 代", "两个 SM 配对 · 单线程发射",
     "<code>tcgen05.mma</code>　·　单 SM：M ＝ 64/128，N ＝ 8,16,…256（步长 8）"
     "　·　两 SM 配对：M ＝ 128/256，N ＝ 16,32,…256（<r>步长 16</r>）　·　<r>K 仍然只有 16</r>",
     "配对时 M 和 N 的步长<b>都翻倍</b> —— 这不是「同一条指令做得更大」，是<b>粒度本身变粗了</b>。<r>⚠️ 这一代趋势反转</r>：前四代是「一次动员越来越多线程」，tcgen05 <b>由单个线程发射</b>（cta_group::2 说的是哪两个 CTA 供数，不是 256 线程一起发），操作数改由 TMEM 供给 —— <b>所以这条轴该读成「操作数的作用域有多大」，不是「动员多少线程」</b>",
     "<r>操作数与累加器都在 TMEM</r>，不占寄存器"),
]

# (标签, M, N, 颜色, 说明)
SHAPES = [
    ("tcgen05　m128 n256", 128, 256, RD),
    ("wgmma　m64 n256", 64, 256, PU),
    ("mma.sync　m16 n8", 16, 8, BL),
]


def build():
    f = Fig(W, H, "Tensor Core 五代 MMA 指令与真实比例形状叠图：收缩维 K 始终是 16，"
                  "与 TPU 的 256×256 脉动阵列对照")
    f.title("Tensor Core 一次吃多大矩阵　—— 五代演进，以及一张真实比例的叠图",
            "全部出自 PTX ISA")
    f.legend([(BL, "warp 级（32 线程）"), (PU, "warp 组级（128 线程）"),
              (RD, "双 SM 级（256 线程）"), (GN, "TPU MXU，同一把尺子")])

    f.t(20, TOP + 18, "五代演进　—— 变的是「一次动员多少线程」，不变的是「K ＝ 16」", "sec")
    for i, (c, name, gen, th, ins, quote, where) in enumerate(GEN):
        y = GEN_T + i * (GENH + GENG)
        f.rect(20, y, 1360, GENH, "#fff", c, 1.6, 9)
        f.rect(20, y, 176, GENH, FILL[c], rx=9)
        f.rect(187, y, 9, GENH, FILL[c], rx=0)
        f.t(34, y + 26, name, "sec", c)
        f.t(34, y + 45, gen, "xs")
        f.t(34, y + 66, th, "lbl", INK)
        f.t(34, y + 82, "参与一次 MMA", "xxs")

        para(f, 214, y + 26, 560, ins, "lbl", 18)
        para(f, 214, y + 62, 560, quote, "xs", 15)

        f.rect(940, y + 14, 424, GENH - 28, "#fff", SUB, 1.1, 6, "3,2")
        f.t(954, y + 32, "操作数住在哪", "xs")
        para(f, 954, y + 52, 396, where, "lbl", 16)

        _mini(f, 792, y + 20, i, c)

    _overlay(f)
    return f.out()


def _mini(f, x, y, i, c):
    """右侧小示意：一次 MMA 动员多少条 lane。"""
    n = [8, 32, 32, 128, 256][i]
    cols, cell = 32, 2.6
    f.t(x, y + 8, f"{n} 条 lane", "xxs", c)
    for k in range(min(n, 256)):
        r, cc = divmod(k, cols)
        f.rect(x + cc * (cell + 1), y + 14 + r * (cell + 1), cell, cell, c, rx=0.5)


# ══════════════════════════════════════════════════════════════════════
def _overlay(f):
    """三块输出矩阵叠在 TPU 那个 256×256 上 —— 只描边，不填色。

    上一版三块都用 opacity 0.10 的实心填充叠在同一个左上角，
    结果渲染出来是三条深浅不一的横带，谁也认不出哪条边属于哪条指令。
    半透明叠色在「嵌套」场景里天然失效：它把边界变成了渐变。
    描边 + 就地标注，形状才读得出来。
    """
    f.t(20, B_T, "同一把尺子上叠一遍　—— 大方框是 TPU 一个 MXU 的 256×256，"
                 "里面三块是 GPU 三条指令各自的输出矩阵", "sec")
    f.t(20, B_T + 20, "每个元素画 1.17 像素，三块和大框用的是同一个比例，没有任何缩放作弊。"
                      "三块共用左上角，所以是<tspan font-weight=\"700\" fill=\"#202124\">嵌套</tspan>关系，不是并排。", "sm")

    # TPU MXU 底框
    f.rect(CX, CY, CANV, CANV, FILL[GN], GN, 2, 4)
    f.t(CX + CANV - 8, CY + CANV - 10, "TPU v7 一个 MXU　256 × 256", "lbl", GN, "end")
    f.t(CX + CANV - 8, CY + CANV - 24, "65,536 个乘加单元，一条指令喂满", "xs", GN, "end")

    # 三块输出矩阵：从大到小描边，各自在自己的底边上就地署名
    for label, m, n, c in SHAPES:
        w, h = n * SC, m * SC
        f.rect(CX, CY, w, h, "none", c, 2.4, 2)
        if w > 120:                       # 放得下就写在自己的底边内侧
            f.rect(CX + w - 152, CY + h - 17, 148, 14, "#fff", c, 1, 3)
            f.t(CX + w - 8, CY + h - 6, label, "xxs", c, "end")

    # 最小那块只有 9 × 19 像素，圈出来并引到右边对应的卡片上
    f.raw(f'<circle cx="{CX+5}" cy="{CY+9}" r="24" fill="none" stroke="{BL}" '
          f'stroke-width="1.6" stroke-dasharray="4,3"/>')
    f.path(f"M{CX+27} {CY+18} L{CX+CANV+56} {CY+233}", BL, 1.4, marker="aB", dash="4,3")

    ann = [
        (RD, "tcgen05.mma　M×N ＝ 128 × 256", 128, 256,
         "Blackwell 最大的一条。两个 SM 配对时 M 到 256 —— 那时它的高度正好是这个大方框的一半，"
         "宽度已经铺满。"),
        (PU, "wgmma　M×N ＝ 64 × 256", 64, 256,
         "Hopper 的 warp 组指令。宽度和上面那条一样满，只有高度是它的一半 —— "
         "两条的差别全在 M 上。只在 Hopper 有，Blackwell 换成了 tcgen05。"),
        (BL, "mma.sync　M×N ＝ 16 × 8", 16, 8,
         "在这张图上它只有 <b>9 × 19 像素</b> —— 就是左上角那个圈住的小蓝点。"
         "从 Turing 一路留到今天的 warp 级指令，"
         "<r>Blackwell 上不但没被砍，消费级那颗 die 的块量化还只挂在它身上（见图 G-5）。</r>"),
    ]
    ax = CX + CANV + 60
    for i, (c, ttl, m, n, note) in enumerate(ann):
        y = CY + i * 96
        f.rect(ax, y, 860, 82, "#fff", c, 1.5, 8)
        f.rect(ax + 14, y + 18, 22, 22, "none", c, 2, 3)
        f.t(ax + 48, y + 26, ttl, "box", c)
        f.t(ax + 48, y + 44, f"输出块占 {m*n:,} 个位置，是 TPU 那个 256×256 的 "
                             f"<tspan font-weight=\"700\" fill=\"#202124\">1/{65536//(m*n)}</tspan>", "xs")
        para(f, ax + 48, y + 64, 790, note, "xs", 15)

    # ── K 维条 ──────────────────────────────────────────────────────────
    ky = CY + CANV + 26
    f.t(20, ky, "还有更关键的一维：收缩维 K　—— 上面画的是输出矩阵，K 才决定「一次能吃多深」", "sec")
    f.rect(20, ky + 12, 1360, 172, FILL[RD], RD, 1.8, 9)
    yy = para(f, 36, ky + 34, 1320,
              "M 和 N 决定输出块有多大，<b>K 决定一次乘加链有多长</b>。GPU 这一维从 Turing 的 8 加倍到 Ampere 的 16 之后 "
              "<r>就再没变过</r>：fp16/bf16 是 K=16，fp8 是 K=32，fp4 是 K=64 —— "
              "换算成位宽全都是 256 bit，也就是同样的 <b>8 个 32-bit 字</b>。", "lbl", 19)

    by = yy + 10
    f.rect(36, by, 16 * SC, 20, FILL[BL], BL, 1.8, 2)
    f.t(36 + 16 * SC + 10, by + 15, "GPU Tensor Core 的 K ＝ 16　← 就这么窄", "lbl", BL)
    f.rect(36, by + 26, 256 * SC, 20, FILL[GN], GN, 1.8, 2)
    f.t(36 + 256 * SC + 10, by + 41, "TPU MXU 的收缩边 ＝ 256　—— 同一把尺子，差 16 倍", "lbl", GN)

    # 这条长短对比最容易被读成「GPU 算得浅」，必须当场堵住
    f.rect(36, by + 54, 700, 40, "#fff", RD, 1.4, 6)
    para(f, 46, by + 70, 680,
         "<r>这 16 倍不是「GPU 只能算 16 深」。</r>它连发多条 K=16，累加器一直待在 TMEM 里不落地，"
         "深度照样累得上去。差的是<b>一条指令的粒度</b>，不是能力上限。", "xs", 14)

    para(f, 760, by + 6, 600,
         "粒度粗细各有代价。这就是为什么注意力里 <code>head_dim=128</code> 这件事"
         "<r>打 TPU 却不打 GPU</r>：128 撞上 TPU 的 256 收缩边，只喂满一半；"
         "而 128 是 16 的整数倍，对 GPU 来说刚好切成 8 条指令，一点不浪费。", "lbl", 19)


if __name__ == "__main__":
    import io, sys
    io.open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/g4.svg", "w",
            encoding="utf-8").write(build())
    print("ok", H)
