# -*- coding: utf-8 -*-
"""图 T-8 —— 从一颗到一个 pod：ICI、3D 环面，以及「什么时候换协议」。

这是全文的落点，和 GPU 那份的 G-8 互为镜像：

  G-8 说的是「一颗 GPU 内部被切成 148 个 SM，协调发生在**芯片里面**」；
  T-8 说的是「一颗 TPU 内部只有 2 个核，协调发生在**芯片之间**」。

两句话合起来才是完整的一句：**同样是把一个模型摊到很多算力上，
GPU 的主战场在片内，TPU 的主战场在片间。**

图的主体是一张对数刻度的横条 —— 因为这件事的关键不是「谁能连更多」，
而是**在哪个规模上你被迫换一套编程模型**。对数轴能把这一刀画在正确的位置，
线性轴会把 72 和 9,216 压成一个点和一条线，看不出那一刀。
"""
import math
from common import Fig, para, BL, RD, YL, GN, PU, TL, INK, SUB, GREY, FILL
import gate

W = 1400
TOP = 84

# ── 左：3D 环面 ────────────────────────────────────────────────────────
TGX, TGY, TCELL = 96, 214, 62         # 4×4 网格
TN = 4
TW_ = TN * TCELL

AXX = 452                              # 六个出口的小图
LBOT = TGY + TW_ + 242

# ── 右：对数刻度的规模阶梯 ─────────────────────────────────────────────
RX = 700
LOGX, LOGW = RX + 92, 568
LOGMAX = 16384.0
BAR_Y = 218
BAR_H = 40
BAR_GAP = 96

RC_Y = BAR_Y + BAR_GAP + BAR_H + 164   # 右栏空档：对数轴那两段说明之下
CARD_Y = max(LBOT, RC_Y + 178) + 20
CARD_H = 186
BAND_Y = CARD_Y + CARD_H + 20
H = BAND_Y + 160


def lx(n):
    return LOGX + LOGW * math.log10(max(n, 1.0)) / math.log10(LOGMAX)


def build():
    f = Fig(W, H, "TPU v7 从一颗芯片扩展到一个 pod：6 条 ICI 链路组成 3D 环面，"
                  "9,216 颗芯片全程同一套互联，不换协议")
    f.title("从一颗到一个 pod　—— 关键不是能连多少，是在哪儿被迫换一套编程模型", "第 8 / 8 张")
    f.legend([(GN, "TPU v7：ICI"), (BL, "NVIDIA B200：NVLink"),
              (RD, "换协议的那一刀"), (GREY, "灰色虚线 ＝ 官方未公开")])

    _torus(f)
    _ladder(f)
    _cards(f)
    _band(f)
    return f.out()


# ══════════════════════════════════════════════════════════════════════
def _torus(f):
    f.t(20, TOP + 26, "为什么是「环面」而不是「网格」", "sec")
    para(f, 20, TOP + 46, 640,
         "每颗芯片有 <b>6 条 ICI 物理链路</b>，对应三维的正负方向。"
         "把最边上的一颗和最那头的一颗接起来 —— <r>多接这一条，直径就少一半</r>。", "xs", 15)

    # 4×4 的一层
    for r in range(TN):
        for c in range(TN):
            x, y = TGX + c * TCELL, TGY + r * TCELL
            hot = (r, c) == (1, 1)
            f.rect(x, y, 34, 34, FILL[GN] if not hot else GN, GN, 1.6, 5)
            if hot:
                f.t(x + 17, y + 22, "●", "lbl", "#fff", "middle")
            # 直连
            if c < TN - 1:
                f.line(x + 34, y + 17, x + TCELL, y + 17, GN, 1.4)
            if r < TN - 1:
                f.line(x + 17, y + 34, x + 17, y + TCELL, GN, 1.4)

    # 环绕链路：只画短桩，不画完整的绕圈弧线。
    # 上一版给每行每列都画了一条贝塞尔大弧，四行四列叠在一起完全读不出来 ——
    # 「两端相连」这件事用桩 + 下面那个单轴示意图讲，比八条弧线清楚得多。
    for r in range(TN):
        y = TGY + r * TCELL + 17
        f.line(TGX - 2, y, TGX - 26, y, RD, 1.4, dash="3,2", marker="aR")
        f.line(TGX + (TN - 1) * TCELL + 36, y, TGX + (TN - 1) * TCELL + 60, y,
               RD, 1.4, dash="3,2", marker="aR")
    for c in range(TN):
        x = TGX + c * TCELL + 17
        f.line(x, TGY - 2, x, TGY - 24, RD, 1.4, dash="3,2", marker="aR")
        f.line(x, TGY + (TN - 1) * TCELL + 36, x, TGY + (TN - 1) * TCELL + 58,
               RD, 1.4, dash="3,2", marker="aR")

    # 单轴示意：一条轴上的 4 颗接成一个环
    ry, rx0 = TGY + TW_ + 34, TGX + 8
    f.t(TGX - 76, ry - 8, "红桩 ＝ 环绕链路，接到那一头去；第三维同理，图上没画", "xxs", RD)
    for k in range(TN):
        f.rect(rx0 + k * 62, ry, 30, 24, FILL[GN], GN, 1.4, 4)
        if k < TN - 1:
            f.line(rx0 + k * 62 + 30, ry + 12, rx0 + (k + 1) * 62, ry + 12, GN, 1.4)
    f.path("M%.1f %.1f C %.1f %.1f, %.1f %.1f, %.1f %.1f"
           % (rx0 + 15, ry + 24, rx0 + 15, ry + 56,
              rx0 + 3 * 62 + 15, ry + 56, rx0 + 3 * 62 + 15, ry + 24),
           RD, 1.6, dash="4,3")
    f.t(rx0 + 1.5 * 62 + 15, ry + 74, "环绕：第 3 颗的邻居就是第 0 颗", "xxs", RD, "middle")
    f.t(rx0 + 3 * 62 + 46, ry + 16, "← 一条轴", "xxs", GN)

    # 六个出口
    f.rect(AXX, TGY - 8, 208, 158, "#fff", GN, 1.6, 8)
    f.t(AXX + 14, TGY + 14, "一颗芯片的 6 个出口", "lbl", GN)
    cxx, cyy = AXX + 104, TGY + 82
    f.rect(cxx - 22, cyy - 16, 44, 32, FILL[GN], GN, 1.6, 4)
    f.t(cxx, cyy + 5, "chip", "xxs", GN, "middle")
    for dx, dy, lab in ((-1, 0, "X−"), (1, 0, "X+"), (0, -1, "Y+"),
                        (0, 1, "Y−"), (-0.72, -0.72, "Z+"), (0.72, 0.72, "Z−")):
        f.line(cxx + dx * 26, cyy + dy * 20, cxx + dx * 56, cyy + dy * 44, GN, 1.6, marker="aG")
        f.t(cxx + dx * 74, cyy + dy * 52 + 4, lab, "xxs", GN, "middle")
    # ⛔ 2026-09-04：原来在 TGY+142，正压在 Y− 那个轴标上（轴标在 TGY+138）。
    f.t(AXX + 104, TGY + 164, "每条 200 GB/s（双向），6 条合计 1,200", "xxs", None, "middle")

    para(f, 20, TGY + TW_ + 124, 640,
         "<b>直径的差别是实打实的：</b>4×4×4 如果只是网格，最远要走 3+3+3 ＝ 9 跳；"
         "接成环面之后是 2+2+2 ＝ <b>6 跳</b>。规模越大差得越多 —— "
         "这决定了 all-reduce 的最坏时延。", "xs", 15)
    para(f, 20, TGY + TW_ + 170, 640,
         "<r>但环面不是白拿的：</r>它要求切片在物理上必须是连续的一块立方体。"
         "所以 TPU 上你申请的不是「64 颗芯片」，是「一个 4×4×4」—— "
         "<b>形状本身是调度的一部分</b>，这一点在 §3 那张表里就已经埋下了。", "xs", 15)


# ══════════════════════════════════════════════════════════════════════
TICKS = [(1, "1"), (8, "8"), (64, "64"), (256, "256"), (1024, "1,024"),
         (4096, "4,096"), (16384, "16,384")]


def _ladder(f):
    f.t(RX, TOP + 26, "同一根对数轴上，两边各能走多远", "sec")
    para(f, RX, TOP + 46, 660,
         "横轴是<b>一个互联域里的加速器颗数</b>（对数）。真正要看的不是端点，"
         "是那条<r>红线</r>：过了它，你的通信代码就得换一套写法。", "xs", 15)

    # 刻度
    f.line(LOGX, BAR_Y - 14, LOGX + LOGW, BAR_Y - 14, "#dadce0", 1.2)
    for v, lab in TICKS:
        x = lx(v)
        f.line(x, BAR_Y - 18, x, BAR_Y - 10, "#dadce0", 1.2)
        f.t(x, BAR_Y - 24, lab, "xxs", GREY, "middle")

    # ── TPU 条 ────────────────────────────────────────────────────────
    y = BAR_Y
    f.t(RX, y + 16, "TPU v7", "box", GN)
    f.rect(LOGX, y, lx(9216) - LOGX, BAR_H, FILL[GN], GN, 1.8, 6)
    f.t(LOGX + 12, y + 25, "全程同一套 ICI，3D 环面一路铺到底", "lbl", GN)
    f.rect(lx(9216), y, lx(LOGMAX) - lx(9216), BAR_H, "#fff", GREY, 1.4, 6, "4,3")
    f.t(lx(9216) + 8, y + 25, "DCN", "xxs", GREY)
    for v, lab in ((1, "1 chip"), (4, "1 台主机"), (64, "1 个 cube 4×4×4"), (9216, "1 个 pod")):
        x = lx(v)
        f.line(x, y, x, y + BAR_H, "#fff", 1.4)
        f.t(x + 4, y + BAR_H + 13, lab, "xxs", GN)
    f.t(lx(9216), y + BAR_H + 30, "9,216 颗 ＝ 144 个 cube", "lbl", GN, "end")

    # ── GPU 条 ────────────────────────────────────────────────────────
    y2 = BAR_Y + BAR_GAP
    f.t(RX, y2 + 16, "B200", "box", BL)
    f.rect(LOGX, y2, lx(72) - LOGX, BAR_H, FILL[BL], BL, 1.8, 6)
    f.t(LOGX + 12, y2 + 25, "NVLink 域", "lbl", BL)
    f.rect(lx(72), y2, lx(LOGMAX) - lx(72), BAR_H, "#fff", BL, 1.4, 6, "4,3")
    f.t(lx(72) + 12, y2 + 25, "RoCE ／ RDMA", "xxs", BL)
    for v, lab in ((8, "1 台机器（上一代 HGX）"), (72, "1 个机柜")):
        x = lx(v)
        f.line(x, y2, x, y2 + BAR_H, "#fff", 1.4)
        f.t(x + 4, y2 + BAR_H + 13, lab, "xxs", BL)

    # 换协议的红线
    xr = lx(72)
    f.line(xr, BAR_Y - 30, xr, y2 + BAR_H + 26, RD, 2.0, dash="6,4")
    f.t(xr + 8, BAR_Y - 36, "换协议就在这儿", "lbl", RD)

    _y = para(f, RX, y2 + BAR_H + 46, 660,
         "<b>128 倍的差距不在带宽上，在「不换协议能连多远」上。</b>"
         "9,216 ÷ 72 ＝ 128 —— 而且这两个数量级之间，TPU 那一侧"
         "<r>集合通信的写法一个字都不用改</r>（<g>这句只在一个 pod 之内成立，"
         "跨 pod 走 DCN 时并行配置照样要改</g>）。越过红线那一侧，实测 all-reduce 从"
         "<b>约 840 掉到约 325 GB/s</b>（<b>2.6 倍</b>，不是一个量级），"
         "而且<b>通信库要换一条实现路径</b>。", "xs", 16)
    para(f, RX, _y + 10, 660,
         "反过来说也别夸大：<b>单颗算力两边几乎打平</b>（NVL72 里那颗 <b>GB200</b> "
         "每 GPU dense BF16 约 2,500 TFLOP/s，TPU v7 一颗 chip 2,307 —— "
         "<g>注意别拿 HGX B200 的 2,250 来比，那是另一个 SKU</g>），"
         "而且在 72 颗以内 <r>NVLink 的每颗带宽还更高</r>"
         "（1.8 TB/s 对 1.2 TB/s）。<g>这张图比的是拓扑能延展多远，不是单芯片谁快。</g>", "xs", 16)


# ══════════════════════════════════════════════════════════════════════
def _cards(f):
    cw = (1360 - 20) / 2.0

    # ① 口径警告
    x = 20
    f.rect(x, CARD_Y, cw, CARD_H, FILL[YL], YL, 1.8, 10)
    f.t(x + 16, CARD_Y + 26, "口径警告：「pod」有两个官方定义", "sec", YL)
    para(f, x + 16, CARD_Y + 48, cw - 32,
         "同一批官方材料里，「pod」既被用来指 <b>9,216 颗芯片的整机规模</b>，"
         "也被用来指<b>一个 256 颗的可售单元</b>。两个都是官方说法，"
         "<r>互相矛盾，而且没有一处说明哪个作准</r>。", "xs", 16)
    para(f, x + 16, CARD_Y + 118, cw - 32,
         "所以看到「一个 pod」这四个字，<b>先问是哪个 pod</b>。本文提到 pod "
         "一律指 9,216 那个，并且每次都把数字写出来。", "xs", 16)

    # ② 查不到
    x = 20 + cw + 20
    f.rect(x, CARD_Y, cw, CARD_H, "#fff", GREY, 1.8, 10, "5,4")
    f.t(x + 16, CARD_Y + 26, "这一节里我查不到的", "sec", GREY)
    para(f, x + 16, CARD_Y + 48, cw - 32,
         "<b>1. 实际能一次调度到的最大切片。</b>物理上 9,216 颗连成一个环面是官方数字，"
         "但「一个作业最多能拿到多大一块」取决于调度系统，"
         "<g>公开资料里没有一个可引用的上限</g>。", "xs", 16)
    para(f, x + 16, CARD_Y + 118, cw - 32,
         "<b>2. 环面在多大规模上会退化成非环。</b>边缘切片能不能拿到环绕链路，"
         "公开资料同样没说。<g>不猜。</g>", "xs", 16)

    # ③ 一个必须说清的口径坑 —— 放在右栏，紧挨着上面那根对数轴
    x, cw3 = RX, 680
    f.rect(x, RC_Y, cw3, 178, FILL[RD], RD, 1.8, 10)
    f.t(x + 16, RC_Y + 26, "1,200 GB/s 这个数，官方自己写拧了", "sec", RD)
    para(f, x + 16, RC_Y + 48, cw3 - 32,
         "官方正文写的是「每<b>轴</b>双向 200 GB/s」。可是三个轴 × 200 ＝ 600，"
         "对不上同一页表格里的 <b>1,200</b>。", "xs", 16)
    para(f, x + 16, RC_Y + 90, cw3 - 32,
         "只有把它读成「每条<b>链路</b> 200」才自洽：6 条 × 200 ＝ 1,200。"
         "<r>本文按这个读法画</r>，并且在这里写明原文是另一种措辞 —— "
         "<b>遇到官方文档自相矛盾，正确做法是标出来，不是挑一个顺手的悄悄用。</b>", "xs", 16)


# ══════════════════════════════════════════════════════════════════════
def _band(f):
    f.rect(20, BAND_Y, 1360, 140, FILL[GN], GN, 1.6, 10)
    f.t(34, BAND_Y + 26, "两份文档合起来的那一句话", "sec", GN)
    _y = para(f, 34, BAND_Y + 52, 1330,
         "GPU 那份的最后一张图讲的是：一颗 B200 里有 <b>592 个 Tensor Core</b>，"
         "而一颗 TPU v7 里只有 <b>4 个 MXU</b> —— 同样一块矩阵乘的活，"
         "<b>份数</b>差 148 倍，而<b>单个单元多大</b>差 <b>128</b> 倍 —— "
         "<g>本文反复说的那个 128 指的是后者，别顺口说成 592 对 4</g>。"
         "<r>GPU 的协调主要发生在芯片内部。</r>", "xs", 20)
    para(f, 34, _y + 8, 1330,
         "这张图讲的是另一半：TPU 一颗芯片里只有两个核要协调，"
         "但<b>不换协议能一路连到 9,216 颗</b>，而 GPU 在 72 颗上就得换。"
         "<r>TPU 的协调主要发生在芯片之间。</r>　"
         "<b>两边不是「谁更强」，是把同一份复杂度放在了不同的地方 —— "
         "这也是这整份材料从头到尾在说的同一件事。</b>", "xs", 20)


if __name__ == "__main__":
    import io, sys
    io.open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/t8.svg", "w",
            encoding="utf-8").write(build())
    print("ok", H)
