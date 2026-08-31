# -*- coding: utf-8 -*-
"""图 P-27 —— 一套 vreg，三个门：VPU、MXU、XLU 取的是同一批寄存器。

**为什么补这张图。** P-26 讲完「横着走贵」，紧接着必然被问一个问题：
那这些部件到底是怎么连的？是各有各的寄存器，还是挂在一条总线上？
不回答这个，读者会自己脑补出一个错的图 —— 最常见的错法是
「VPU 一套寄存器、MXU 另一套」，那样后面所有关于「数据在核内怎么流」的
判断都会跟着错。

**这张图的机关是中间那个盒子只有一个。** 左边 VMEM 进出，右边三个单元
各开一个门，全都通到同一个 vreg 堆。「不分家」这件事只能靠版式说 ——
写一行「它们共享寄存器」，读者点头，脑子里那张错图不会变。

**⚠️ 数字口径。** 里面每个数都出自公开的 JAX scaling book，而书里标的是
<b>v5p</b>（并且明说 v4 只有 32 个 vreg）。v7 没有同口径的公开数，
所以底带要把这条写死 —— 这张图讲的是**结构**，不是 v7 的规格表。
"""
from common import Fig, para, BL, GN, RD, YL, PU, TL, INK, SUB, GREY, FILL

W = 1400

MAIN_Y, MAIN_H = 84, 356
VM_X, VM_W = 20, 196
HUB_X, HUB_W = 286, 326
RT_X, RT_W = 676, 704
ROW_Y, ROW_H = 118, 262

WHY_Y, WHY_H = 410, 104
GPU_Y, GPU_H = 530, 76
BND_Y, BND_H = 622, 84
H = BND_Y + BND_H + 20

# 右列三个门。每个都必须回答「它从 vreg 拿走什么、还回什么」，
# 而不是各写各的功能简介 —— 功能简介 3.2 前面已经有了，这里重复没有价值。
GATES = [
    (GN, 118, 124, "VPU　·　逐元素", [
        "每个 (lane, sublane) 位置上有 <b>4 个互相独立的 ALU</b> "
        "—— <b>8 × 128 × 4 ＝ 4096 个</b>，书里那个数就是这么来的。",
        "<b>一条指令吃的是两个完整 vreg</b>，写出第三个。"
        "每周期发 <b>4 条</b>，而且 <b>4 条可以各干各的</b>（vadd、vsub 同时跑）。"]),
    (RD, 254, 58, "MXU　·　矩阵乘", [
        "书里这句是原话：vreg「<b>hold data for the VPU and MXU</b>」"
        "—— <b>同一批寄存器，两个用户</b>。喂进去的 LHS 正好是 8 × 128。"]),
    (TL, 324, 56, "XLU　·　跨 lane", [
        "横着走唯一的门。书里说要跨 lane，<b>至少得过一趟 "
        "VMEM／XLU／SMEM</b> —— <b>出了寄存器堆再回来</b>。"]),
]


def build():
    f = Fig(W, H, "TPU 核内的向量寄存器堆是 VPU、MXU、XLU 共用的："
                  "不是每个单元一套寄存器，也不是挂在一条总线上")
    f.title("核里只有<tspan font-weight=\"700\">一套 vreg</tspan>"
            "　—— VPU、MXU、XLU 都从这里取，各开一个门")
    f.legend([(PU, "vreg 堆（唯一的一份）"), (BL, "VMEM 进出"),
              (GN, "VPU"), (RD, "MXU"), (TL, "XLU")])
    _vmem(f)
    _hub(f)
    for args in GATES:
        _gate(f, *args)
    _why(f)
    _gpu(f)
    _bound(f)
    return f.out()


# ══════════════════════════════════════════════════════════════════════
def _vmem(f):
    f.rect(VM_X, ROW_Y, VM_W, ROW_H, FILL[BL], BL, 1.4, 10)
    f.t(VM_X + 16, ROW_Y + 28, "VMEM", "sec", BL)
    f.t(VM_X + 16, ROW_Y + 48, "片上便签本", "xs", SUB)
    para(f, VM_X + 16, ROW_Y + 76, VM_W - 32,
         "所有数据进核的第一站。<b>vreg 只能从这里装填</b>，"
         "算完也只能写回这里。", "xs", 17)

    # 进出不对称 —— 这是公开数，而且正好说明「读比写宽」
    y = ROW_Y + 150
    for c, arrow, txt in [(BL, "→", "每周期 <b>装 3 个</b> vreg"),
                          (GREY, "←", "每周期 <b>写回 1 个</b>")]:
        f.t(VM_X + 16, y, arrow, "numb", c)
        para(f, VM_X + 42, y, VM_W - 58, txt, "xs", 17)
        y += 30
    f.t(VM_X + 16, ROW_Y + ROW_H - 14, "（v5p，公开资料）", "xxs", GREY)

    # 两条实线连到 hub
    f.line(VM_X + VM_W + 6, ROW_Y + 96, HUB_X - 6, ROW_Y + 96, BL, 2.0, "aB")
    f.line(HUB_X - 6, ROW_Y + 132, VM_X + VM_W + 6, ROW_Y + 132, GREY, 1.6, "aK")


# ══════════════════════════════════════════════════════════════════════
# 中间这个盒子是整张图的机关：它只有一个。
def _hub(f):
    f.rect(HUB_X, ROW_Y, HUB_W, ROW_H, FILL[PU], PU, 2.0, 10)
    f.t(HUB_X + 18, ROW_Y + 28, "vreg 堆　·　全核唯一一份", "sec", PU)

    # 一摞寄存器，其中一个摊开给你看形状
    sx, sy = HUB_X + 18, ROW_Y + 46
    for i in range(5):
        f.rect(sx + i * 7, sy + i * 5, 108, 26, "#fff", PU, 1.0, 3)
    f.t(sx + 40, sy + 66, "…… 共 64 个", "xxs", SUB)

    gx, gy, cw, ch = HUB_X + 168, sy + 4, 11, 8
    for r in range(8):
        for c in range(12):
            f.rect(gx + c * cw, gy + r * ch, cw - 2, ch - 1.5, "#fff", PU, 0.6, 1)
    f.t(gx, gy - 6, "一个 vreg ＝ 8 × 128", "xxs", PU)
    f.t(gx, gy + 8 * ch + 14, "1024 个 32 位数 ＝ 4 KiB", "xxs", SUB)

    y = ROW_Y + 148
    y = para(f, HUB_X + 18, y, HUB_W - 36,
             "<b>64 × 8 × 128 × 4 B ＝ 256 KiB</b>，这就是一个核全部的"
             "向量寄存器。", "xs", 17)
    para(f, HUB_X + 18, y + 6, HUB_W - 36,
         "<r>没有「VPU 专用」和「MXU 专用」之分</r> —— "
         "六十四个是一个池子，编译器随便挑。", "xs", 17)
    f.t(HUB_X + 18, ROW_Y + ROW_H - 14, "（v5p 是 64 个；v4 只有 32 个）",
        "xxs", GREY)


# ══════════════════════════════════════════════════════════════════════
def _gate(f, c, y, h, ttl, lines):
    f.rect(RT_X, y, RT_W, h, FILL[c], c, 1.4, 9)
    f.t(RT_X + 16, y + 24, ttl, "box", c)
    ty = y + 44
    for s in lines:
        ty = para(f, RT_X + 16, ty, RT_W - 32, s, "xs", 17) + 4
    # 门 —— 双向箭头，画在 hub 和它之间
    f.line(HUB_X + HUB_W + 6, y + h / 2, RT_X - 6, y + h / 2, c, 1.8, "aB"
           if c == BL else ("aG" if c == GN else ("aR" if c == RD else "aK")))


# ══════════════════════════════════════════════════════════════════════
def _why(f):
    f.rect(20, WHY_Y, 1360, WHY_H, FILL[YL], YL, 1.6, 10)
    f.t(38, WHY_Y + 28, "⭐ 于是「横着走为什么贵」有了一个更准的说法", "sec")
    y = para(f, 38, WHY_Y + 54, 1324,
             "<b>不是「列与列之间没有连线」</b> —— 那是想当然。"
             "真正的差别是<b>要不要出这个盒子</b>：纵向那三步 shuffle "
             "在 ALU 里就地完成，数没离开 vreg；"
             "横向要跨 lane，就得<b>把整个 vreg 交出去、绕一趟、再收回来</b>。", "xs", 19)
    para(f, 38, y + 4, 1324,
         "<r>贵的是这趟往返，不是那一次加法。</r>", "sm", 19, fill=INK)


# ══════════════════════════════════════════════════════════════════════
# 这门课是 GPU × TPU 对照课，所以书里给的这个换算比什么都值钱。
def _gpu(f):
    f.rect(20, GPU_Y, 1360, GPU_H, FILL[BL], BL, 1.4, 10)
    f.t(38, GPU_Y + 26, "⭐ 换算成 GPU 的词：书里给了一组对照", "sec", BL)
    para(f, 38, GPU_Y + 52, 1324,
         "<b>VPU 里的一个 ALU ≈ 一个 CUDA core</b>；"
         "<b>VPU 的一条 lane ≈ 一个 warp scheduler</b>（就是那组通常 32 个 "
         "CUDA core 一起做 SIMD 的单位）。"
         "两边差的不是「有没有这些东西」，是<b>谁在运行时挑活</b> —— "
         "这正是 3.2 开头那张图右边一栏说的事。", "xs", 19)


# ══════════════════════════════════════════════════════════════════════
def _bound(f):
    f.rect(20, BND_Y, 1360, BND_H, FILL[RD], RD, 1.4, 10)
    f.t(38, BND_Y + 26, "⚠️ 这张图讲的是结构，不是 v7 的规格表", "sec")
    para(f, 38, BND_Y + 52, 1324,
         "上面每个数（64 个 vreg、每位置 4 个 ALU、每周期 4 条指令、"
         "VMEM 每周期 3 读 1 写）<b>都出自公开资料，而书里标的是 v5p</b> —— "
         "它自己就说了 v4 只有 32 个 vreg，<b>所以这些数会跨代变</b>。"
         "<b>v7 没有同口径的公开数，别把这张图当 v7 的规格表用。</b>", "xs", 19)


if __name__ == "__main__":
    import io
    io.open("out/fig_p27_vregfile.svg", "w", encoding="utf-8").write(build())
    print("ok fig_p27_vregfile")
