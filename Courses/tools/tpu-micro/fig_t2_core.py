# -*- coding: utf-8 -*-
"""图 T-2 —— 把一个 TensorCore 拆开。

这张图的讲法和 GPU 那份的 G-2 是反的。G-2 是「看这里塞了多少东西」，
T-2 是「看这里**少**了多少东西」—— 没有 warp 调度器、没有线程上下文、
没有自动缓存、没有乱序。省下来的面积不是消失了，它变成了两样东西：
更大的乘加阵列，和大得离谱的软件可控暂存。

所以图的右半边是一张「GPU SM 里有、这里没有」的对照清单。
先让学生数清楚少了什么，再问一句「那省下的硅去哪了」，答案就在左半边。
"""
from common import Fig, para, BL, RD, YL, GN, PU, TL, INK, SUB, GREY, FILL
import gate

W = 1400
TOP = 84

CX, CY, CW = 20, 118, 880
R1 = CY + 44                     # 控制
R2 = R1 + 84                     # 计算
R3 = R2 + 258                    # 片上存储（和上一排留够间距：上一版标题压在 MXU 框上）
R4 = R3 + 118                    # 对外搬运
CH = R4 + 78 + 16 - CY

RX, RW = 920, 460
BAND_Y = CY + CH + 22
H = BAND_Y + 138


def build():
    f = Fig(W, H, "TPU v7 一个 TensorCore 的内部构成：2 个 MXU、1 个 VPU、"
                  "标量单元、VMEM 与 SMEM；与 GPU SM 相比缺少调度器、"
                  "线程上下文、自动缓存与乱序执行")
    f.title("把一个 TensorCore 拆开　—— 值得数的不是它有什么，是它<少了>什么"
            .replace("<", "「").replace(">", "」"), "第 2 / 8 张")
    f.legend([(GN, "矩阵乘"), (TL, "向量／标量"), (YL, "片上存储"),
              (RD, "对外搬运"), (GREY, "灰色虚线 ＝ 官方未公开")])

    f.raw('<defs><pattern id="mxucell2" width="3" height="3" patternUnits="userSpaceOnUse">'
          f'<rect width="3" height="3" fill="{FILL[GN]}"/>'
          f'<rect width="1.6" height="1.6" x="0.2" y="0.2" fill="{GN}" opacity="0.34"/>'
          '</pattern></defs>')

    _core(f)
    _absent(f)
    _band(f)
    return f.out()


# ══════════════════════════════════════════════════════════════════════
def _core(f):
    f.rect(CX, CY, CW, CH, "#fff", BL, 2.2, 12)
    f.t(CX + 16, CY + 28, "一个 TensorCore　（＝ 一个 JAX device 的全部算力）", "sec", BL)
    f.t(CX + CW - 16, CY + 28, "@ 2.2 GHz", "lbl", BL, "end")

    ix, iw = CX + 14, CW - 28

    # ── 控制层 ────────────────────────────────────────────────────────
    f.rect(ix, R1, iw, 72, FILL[TL], TL, 1.6, 8)
    f.t(ix + 12, R1 + 20, "控制：标量单元发射 VLIW 指令包", "box", TL)
    para(f, ix + 12, R1 + 38, 470,
         "一拍发出一整包指令，包里各个槽<b>同时</b>喂给下面不同的单元。"
         "谁在第几拍动、数据什么时候到位，<r>全部由编译器在编译期排好</r>。", "xs", 15)
    # VLIW 槽位示意（v2/v3 公开数字，不往 v7 上套）
    sx = ix + 520
    slots = [("标量", 2, TL), ("向量", 4, TL), ("矩阵", 2, GN), ("杂项", 1, SUB)]
    for name, n, c in slots:
        for k in range(n):
            f.rect(sx, R1 + 16, 16, 22, FILL[c], c, 1.1, 3)
            sx += 19
        f.t(sx - n * 19, R1 + 50, name, "xxs", c)
        sx += 10
    f.t(ix + 520, R1 + 12, "9 个发射槽", "xxs", SUB)
    # 这行原本和槽位组名同一个 y，右对齐后直接压在「矩阵」「杂项」上 —— 下移一行
    f.t(ix + iw - 12, R1 + 64,
        "槽位构成是 v2/v3 论文的公开数字，v7 官方未公布", "xxs", GREY, "end")

    # ── 计算层 ────────────────────────────────────────────────────────
    for k in range(2):
        _mxu(f, ix + k * 232, R2)
    _vpu(f, ix + 480, R2)
    _xlu(f, ix + 700, R2)

    # ── 片上存储层 ────────────────────────────────────────────────────
    f.t(ix, R3 - 8, "片上存储　—— 容量出自 JAX 开源代码，带宽官方未公开", "sec")
    mems = [
        (YL, "VMEM", "64 MiB", "软件显式搬进搬出的暂存。<b>不是缓存</b> ——\n"
                               "没有自动填充、没有替换策略、命不命中这件事不存在", 300),
        (YL, "SMEM", "1 MiB", "标量数据与 DMA 描述符", 170),
        (GN, "累加器", "128 个", "每个形状 (8, 256)、32 bit，<b>挂在每个 MXU 上</b>", 210),
        (GREY, "向量寄存器", "查不到", "v2/v3 论文说每 sublane 32 深；<b>v7 没有公开</b>", 190),
    ]
    mx = ix
    for c, name, val, note, w in mems:
        dash = "5,3" if c is GREY else None
        f.rect(mx, R3 + 4, w - 10, 100, "#fff" if c is GREY else FILL[c],
               c, 1.6, 8, dash)
        f.t(mx + 12, R3 + 26, name, "box", INK if c is not GREY else GREY)
        f.t(mx + w - 22, R3 + 26, val, "numb", c if c is not GREY else GREY, "end")
        para(f, mx + 12, R3 + 46, w - 34, note.replace("\n", ""), "xxs", 13)
        mx += w

    # ── 对外搬运 ──────────────────────────────────────────────────────
    f.rect(ix, R4, iw, 72, FILL[RD], RD, 1.6, 8)
    f.t(ix + 12, R4 + 20, "对外：DMA 引擎", "box", RD)
    para(f, ix + 12, R4 + 38, 560,
         "HBM ↔ VMEM 的搬运由 DMA 完成，而<b>描述符是标量单元发出来的</b> —— "
         "也就是说「什么时候搬、搬多少」同样写在指令流里，不是硬件自己决定的。", "xs", 15)
    f.rect(ix + 600, R4 + 12, iw - 612, 48, "#fff", GREY, 1.4, 6, "5,3")
    para(f, ix + 612, R4 + 30, iw - 636,
         "<g>一个 TensorCore 到底有几个 DMA 引擎：查不到。所以这里只画一个盒子，不标数量。</g>",
         "xxs", 13)


def _mxu(f, x, y):
    f.rect(x, y, 222, 226, "#fff", GN, 1.8, 8)
    f.t(x + 12, y + 22, "MXU", "box", GN)
    f.t(x + 210, y + 22, "256 × 256", "lbl", GN, "end")
    f.rect(x + 12, y + 32, 198, 116, "url(#mxucell2)", GN, 1.1, 3)
    f.t(x + 12, y + 168, "65,536 个 cell", "xs")
    f.t(x + 12, y + 184, "× 每 cell 每周期 2 次乘加", "xs")
    f.line(x + 12, y + 192, x + 210, y + 192, "#e8eaed", 1)
    f.t(x + 12, y + 208, "＝ 131,072 乘加 / 周期", "lbl", GN)
    f.t(x + 210, y + 208, "官方", "xxs", GN, "end")


def _vpu(f, x, y):
    f.rect(x, y, 210, 226, "#fff", TL, 1.8, 8)
    f.t(x + 12, y + 22, "VPU　向量单元", "box", TL)
    f.t(x + 12, y + 40, "8 sublane × 128 lane", "xs", TL)
    for r in range(8):
        for c in range(16):
            f.rect(x + 14 + c * 11.6, y + 50 + r * 9.4, 9.6, 7.6, FILL[TL], rx=1)
    f.t(x + 12, y + 142, "一拍处理 1,024 个元素", "xs")
    para(f, x + 12, y + 164, 188,
         "激活、归一化、逐元素运算、归约 —— <b>矩阵乘之外的活都归它</b>。"
         "后面 §3 讲的 lane / sublane，物理来源就是这张 8×128 的网格。", "xxs", 13)


def _xlu(f, x, y):
    f.rect(x, y, 152, 226, "#fff", TL, 1.8, 8)
    f.t(x + 12, y + 22, "跨 lane 单元", "box", TL)
    f.t(x + 140, y + 22, gate.IP("×2", "?", why="内部设备表：单元个数"), "lbl", TL, "end")
    # 一张小示意：数据横着跨过 lane 边界
    for r in range(4):
        f.rect(x + 14, y + 40 + r * 22, 124, 14, FILL[TL], TL, 0.8, 2)
    f.path(f"M{x+24} {y+50} C{x+80} {y+50} {x+70} {y+112} {x+126} {y+112}",
           RD, 1.6, marker="aR")
    para(f, x + 12, y + 148, 130,
         "转置、跨 lane 归约、shuffle。VPU 的 128 条 lane <b>各干各的</b>，"
         "数据要横着走就得经过这里。", "xxs", 13)
    f.t(x + 12, y + 214, gate.IP("个数出自内部设备表", "个数官方未公开"),
        "xxs", GREY)


# ══════════════════════════════════════════════════════════════════════
def _absent(f):
    f.rect(RX, CY, RW, 390, "#fff", RD, 1.8, 10)
    f.t(RX + 14, CY + 26, "一个 GPU SM 里有，而这里<没有>".replace("<", "「").replace(">", "」"),
        "sec", RD)
    f.t(RX + 14, CY + 44, "先数清楚少了什么，再问省下的硅去了哪", "xs")

    items = [
        ("4 个 warp 调度器", "指令由谁选、什么时候发 —— 这里没有「选」这个动作"),
        ("64 个 warp 槽（2,048 个线程上下文）", "没有线程概念，也就没有「切换到别的线程」这条退路"),
        ("256 KB 寄存器堆", "GPU 那么大的寄存器堆主要是<b>为了同时装下几十份上下文</b>"),
        ("L1 / 共享内存的自动填充", "VMEM 是纯暂存：<r>谁搬谁负责</r>，没有命中率这回事"),
        ("记分板 / 乱序发射", "全部前移到编译期。代价见 §7 —— 算错了没有兜底"),
    ]
    y = CY + 62
    for name, why in items:
        f.rect(RX + 14, y, RW - 28, 60, FILL[RD], RD, 1.1, 6)
        f.t(RX + 26, y + 20, "✕", "box", RD)
        f.t(RX + 46, y + 20, name, "lbl", INK)
        para(f, RX + 46, y + 38, RW - 76, why, "xxs", 13)
        y += 64

    # 省下的面积去哪了
    y2 = CY + 404
    f.rect(RX, y2, RW, CH - 404, "#fff", GN, 1.8, 10)
    f.t(RX + 14, y2 + 26, "省下的硅去了哪：软件能直接指挥的暂存", "sec", GN)
    para(f, RX + 14, y2 + 46, RW - 28,
         "TPU v7 一颗 chip 上的 VMEM 是 <b>128 MiB</b>（两个 core 各 64 MiB），"
         "而且<b>每一个字节都由编译器显式安排</b>。", "xs", 15)
    bars = [(GN, "TPU v7 一颗 chip 的 VMEM", 128.0, "128 MiB"),
            (BL, "B200 一整颗的共享内存合计", 34.0, "约 34 MB　<g>第三方</g>")]
    by = y2 + 88
    for c, name, v, lab in bars:
        f.t(RX + 14, by, name, "xs")
        f.rect(RX + 14, by + 6, v / 128.0 * (RW - 150), 16, FILL[c], c, 1.2, 3)
        # 这里原本是 f.t()，而 lab 里带着 `<g>第三方</g>` —— t() 不解析行内标记，
        # 浏览器把未知元素 <g> 连同「第三方」三个字一起丢掉了。
        # 于是这份**逐个数字标来源等级**的文档里，恰好是那个来源等级标签隐形了。
        # 换 para()（它会解析），并在 t() 里加了断言防复发。
        para(f, RX + RW - 16, by + 19, 240, lab, "lbl", 15, c, 1, "end")
        by += 42
    para(f, RX + 14, by + 2, RW - 28,
         "口径是<b>软件可控的暂存</b>，不含 B200 那 126 MB 的 L2 缓存。", "xxs", 13)


def _band(f):
    f.rect(20, BAND_Y, 1360, 124, FILL[BL], BL, 1.6, 10)
    f.t(34, BAND_Y + 26, "这张图想让你记住的一句话", "sec", BL)
    para(f, 34, BAND_Y + 48, 1330,
         "<b>GPU 的 SM 里，大部分晶体管不是在算数，是在「决定接下来算什么」</b> —— "
         "调度器、上下文、记分板、缓存控制，全都是为了在运行时动态地藏住延迟和分支。"
         "TPU 把这一整套<r>前移到了编译期</r>，于是这些部件可以整个不要。", "xs", 17)
    para(f, 34, BAND_Y + 88, 1330,
         "省下的面积去了两个地方：<b>更大的乘加阵列</b>（一个 MXU 一条指令吃满 65,536 个 cell）"
         "和<b>更大的软件可控暂存</b>（128 MiB VMEM）。"
         "这不是「TPU 更简单」，是<b>把复杂度换了个地方放</b> —— 从硅上换到了编译器里。", "xs", 17)


if __name__ == "__main__":
    import io, sys
    io.open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/t2.svg", "w",
            encoding="utf-8").write(build())
    print("ok", H)
