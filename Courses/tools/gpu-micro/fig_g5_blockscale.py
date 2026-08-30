# -*- coding: utf-8 -*-
"""图 G-5 —— 块量化：GPU 把「多细的一撮数共享一个缩放因子」做进了硬件。

这一节是 GPU 相对 TPU 真正的结构性差异之一，而且方向和大多数人的直觉相反：
Blackwell 一边加了更粗的快车道（tcgen05 双 SM MMA），一边把最细的那条老路
（warp 级 mma.sync）留着并给它加了新能力 —— 块缩放只挂在这两条路上。
"""
from common import Fig, para, BL, RD, YL, GN, PU, TL, INK, SUB, GREY, FILL

W = 1400
TOP = 84
BAND_X, BAND_W = 200, 1180
CELL = BAND_W / 64.0
A_T = TOP + 30
ROWH = 62
B_T = A_T + 3 * ROWH + 40
B_H = 190
C_T = B_T + B_H + 34
H = C_T + 210

ROWS = [
    (SUB, "整个张量一个缩放因子", 64,
     "最粗。一个离群值就把整条的动态范围拉坏 —— 这是 FP8 训练早期最常见的翻车原因。"),
    (BL, "MXFP4　每 32 个元素一个", 32,
     "开放 MX 标准。缩放因子类型 <code>.ue8m0</code>（8 位指数，无尾数）。"),
    (RD, "NVFP4　每 16 个元素一个", 16,
     "NVIDIA 自己的格式，<r>粒度是 MX 的两倍细</r>。缩放因子 <code>.ue4m3</code>，"
     "外面再套一层 FP32 的张量级缩放 —— 两级缩放。"),
]


def build():
    f = Fig(W, H, "块量化粒度对照：整张量、每 32 个元素、每 16 个元素各一个缩放因子，"
                  "以及这些能力挂在哪条 PTX 指令上")
    f.title("块量化　—— 「多细的一撮数共享一个缩放因子」，Blackwell 把它做进了硬件",
            "PTX ISA 9.3")
    f.legend([(BL, "MX 标准：32 个一组"), (RD, "NVFP4：16 个一组"),
              (PU, "warp 级指令"), (GREY, "本文未查实")])

    f.t(20, TOP + 18, "同样 64 个数，三种分法", "sec")
    for i, (c, name, blk, note) in enumerate(ROWS):
        y = A_T + i * ROWH
        f.t(20, y + 22, name, "lbl", c)
        para(f, 20, y + 40, 96, f"{64 // blk} 组", "xs")
        for k in range(64):
            g = k // blk
            f.rect(BAND_X + k * CELL, y + 8, CELL - 1.2, 24,
                   FILL[c] if g % 2 == 0 else "#fff", c, 0.7, 1.5)
        for g in range(64 // blk):
            gx = BAND_X + g * blk * CELL
            gw = blk * CELL - 1.2
            f.rect(gx, y + 36, gw, 15, c, rx=3)
            f.t(gx + gw / 2, y + 47, "scale", "xxs", "#fff", "middle")
        f.t(BAND_X, y + 62, note.replace("<code>", '<tspan class="mono" fill="#1967d2">')
            .replace("</code>", "</tspan>").replace("<r>", '<tspan font-weight="700" fill="#d93025">')
            .replace("</r>", "</tspan>"), "xs")

    _where(f)
    _why(f)
    return f.out()


def _where(f):
    f.t(20, B_T - 12, "关键在于：这套能力挂在哪条指令上　—— 答案是<tspan fill=\"#d93025\">两条都挂</tspan>，"
                      "包括最细的那条 warp 级老路", "sec")
    cards = [
        (PU, "warp 级　32 个线程",
         "mma.sync.aligned.m16n8k32.block_scale",
         "mma.sync.aligned.m16n8k64.block_scale",
         [".kind::mxf8f6f4 · .kind::mxf4 · .kind::mxf4nvf4",
          ".scale_vec::1X / 2X / 4X　·　.block16 / .block32",
          "操作数走<b>寄存器</b>，一个 warp 就能发",
          "<r>这条是从 Ampere 留下来的，Blackwell 不但没砍，还专门给它加了块缩放。</r>"]),
        (RD, "双 SM 级　256 个线程",
         "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale",
         "（缩放因子本身也放在 TMEM 里：[scale-A-tmem]、[scale-B-tmem]）",
         [".scale_vec::4X 配 k64 ＝ 每 16 个元素一个缩放因子",
          "缩放因子类型 <code>.ue8m0</code>（MX）/ <code>.ue4m3</code>（NVFP4）",
          "操作数与累加器都在 <b>TMEM</b>",
          "这条是粗快车道：吞吐最高，但要两个 SM 配合"]),
    ]
    cw = (1360 - 20) / 2
    for i, (c, ttl, l1, l2, bullets) in enumerate(cards):
        x = 20 + i * (cw + 20)
        f.rect(x, B_T, cw, B_H, "#fff", c, 1.8, 10)
        f.rect(x, B_T, cw, 30, FILL[c], rx=10)
        f.rect(x, B_T + 21, cw, 9, FILL[c], rx=0)
        f.t(x + 14, B_T + 20, ttl, "box", c)
        f.rect(x + 14, B_T + 40, cw - 28, 40, "#f8f9fa", SUB, 1, 5)
        f.t(x + 22, B_T + 57, l1, "mono", INK)
        f.t(x + 22, B_T + 72, l2, "mono")
        for j, b in enumerate(bullets):
            para(f, x + 14, B_T + 100 + j * 21, cw - 28, "· " + b, "xs", 15)


def _why(f):
    f.t(20, C_T, "为什么这件事值得单独讲一节", "sec")
    f.rect(20, C_T + 12, 900, 170, FILL[GN], GN, 1.8, 10)
    y = para(f, 36, C_T + 36, 868,
             "常见的说法是「GPU 在往 TPU 靠：也开始搞大矩阵单元了」。这个说法只对了一半。"
             "Blackwell 确实<b>加了</b>一条更粗的快车道（两个 SM 配对做同一次 MMA），"
             "但它<r>没有把细的那条路砍掉</r> —— 恰恰相反，最细的 warp 级 "
             "<code>mma.sync</code> 是块缩放<b>唯一</b>挂得上的地方之一。", "lbl", 19)
    y = para(f, 36, y + 12, 868,
             "所以正确的说法是：<b>GPU 现在粗细两条路并存，TPU 只有粗的那一条。</b>"
             "细粒度不是遗留包袱，是它换来的表达力 —— 每 16 个元素一个缩放因子这种事，"
             "需要计算单元本身就认得「16 个元素」这个粒度。", "lbl", 19)
    para(f, 36, y + 12, 868,
         "一句话记住：<b>粗是为了吞吐，细是为了精度</b>。Blackwell 两条都留，"
         "代价是 SM 内部多一套控制通路；TPU 只留粗的那条，换来的是控制逻辑极简、"
         "面积几乎全给了乘加阵列。", "lbl", 19)

    f.rect(940, C_T + 12, 440, 170, "#fff", GREY, 1.6, 10, "4,3")
    f.t(956, C_T + 34, "TPU 那边呢 —— 本文<tspan font-weight=\"700\" fill=\"#202124\">未查实</tspan>", "box", GREY)
    para(f, 956, C_T + 58, 410,
         "公开的 JAX 规格表只列了 TPU v7 的 bf16 和 fp8 两档峰值，"
         "<b>没有任何 MX / NVFP4 类格式的条目</b>，也没有说明 MXU 是否有硬件级的分块缩放通路。", "xs", 16)
    para(f, 956, C_T + 132, 410,
         "所以这里只写能确定的那一条：<b>MXU 的收缩边是 256</b>，"
         "比 GPU 的 K=16 粗 16 倍。至于「TPU 有没有硬件块缩放」，"
         "查到之前不下结论。", "xs", 16)


if __name__ == "__main__":
    import io, sys
    io.open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/g5.svg", "w",
            encoding="utf-8").write(build())
    print("ok", H)
