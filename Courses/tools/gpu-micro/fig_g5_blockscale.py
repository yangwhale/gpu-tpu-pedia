# -*- coding: utf-8 -*-
"""图 G-5 —— 块量化：GPU 把「多细的一撮数共享一个缩放因子」做进了硬件。

这一节是 GPU 相对 TPU 真正的结构性差异之一，而且方向和大多数人的直觉相反 ——
也和本文第一版写的相反。Blackwell 有两条块缩放通路，但它们**不在同一颗 die 上**：
warp 级 mma.sync 的块缩放要 sm_120a（消费级），tcgen05 的要 sm_100a（B200）。
所以 B200 上只有粗的那一条。目标架构以 PTX ISA 9.3 的 Target ISA Notes 为准。

「为什么」那一大段已移到正文 §5 —— 图上不再重复一遍，省下 214px。
"""
from common import Fig, para, BL, RD, YL, GN, PU, TL, INK, SUB, GREY, FILL

W = 1400
TOP = 84
BAND_X, BAND_W = 200, 1180
CELL = BAND_W / 64.0
A_T = TOP + 30
ROWH = 62
B_T = A_T + 3 * ROWH + 40
B_H = 214          # 卡片加高：多一行「要求的目标架构」
H = B_T + B_H + 28

ROWS = [
    (TL, "整个张量一个缩放因子", 64,
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
    f.legend([(TL, "整张量一个缩放因子"), (BL, "MX 标准：32 个一组"),
              (RD, "NVFP4：16 个一组"), (PU, "warp 级指令（消费级 die 才有）")])

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
    return f.out()


def _where(f):
    f.t(20, B_T - 12, "关键在于：这套能力挂在哪条指令上　—— 两条通路，"
                      "<tspan fill=\"#d93025\">但不在同一颗 die 上</tspan>", "sec")
    cards = [
        (PU, "warp 级　32 个线程", "sm_120a", "<r>B200 用不了</r>",
         "mma.sync.aligned.m16n8k32.block_scale",
         "mma.sync.aligned.m16n8k64.block_scale",
         [".kind::mxf8f6f4（1X）· .kind::mxf4 / mxf4nvf4（配 k64 才有 2X/4X）",
          "操作数走<b>寄存器</b>，一个 warp 就能发",
          "PTX 8.7 引入，<b>只在消费级 Blackwell 上</b>（RTX 50 / RTX PRO）",
          "<r>这不是 Ampere 遗产 —— fp8/fp6/fp4 这套形式是 Blackwell 才有的。</r>"]),
        (RD, "双 SM 级　256 个线程", "sm_100a", "<b>B200 就这一条</b>",
         "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale",
         "（缩放因子本身也放在 TMEM 里：[scale-A-tmem]、[scale-B-tmem]）",
         [".scale_vec::1X / 2X / 4X　·　.block16 / .block32（这两个是 tcgen05 的）",
          "缩放因子类型 <code>.ue8m0</code>（MX）/ <code>.ue4m3</code>（NVFP4）",
          "操作数与累加器都在 <b>TMEM</b>",
          "这条是粗快车道：吞吐最高，但要两个 SM 配合"]),
    ]
    cw = (1360 - 20) / 2
    for i, (c, ttl, arch, avail, l1, l2, bullets) in enumerate(cards):
        x = 20 + i * (cw + 20)
        f.rect(x, B_T, cw, B_H, "#fff", c, 1.8, 10)
        f.rect(x, B_T, cw, 30, FILL[c], rx=10)
        f.rect(x, B_T + 21, cw, 9, FILL[c], rx=0)
        f.t(x + 14, B_T + 20, ttl, "box", c)
        # 目标架构徽标 —— 全图最该被看到的一格
        bw = 96
        f.rect(x + cw - 14 - bw, B_T + 5, bw, 20, c, rx=10)
        f.t(x + cw - 14 - bw / 2, B_T + 19, arch, "num", "#fff", "middle")
        f.rect(x + 14, B_T + 38, cw - 28, 22, "#fff", c, 1.2, 5)
        para(f, x + 22, B_T + 53, cw - 44, "要求的目标架构 " + arch + "　→　" + avail, "xs", 13)
        f.rect(x + 14, B_T + 66, cw - 28, 40, "#f8f9fa", SUB, 1, 5)
        f.t(x + 22, B_T + 83, l1, "mono", INK)
        f.t(x + 22, B_T + 98, l2, "mono")
        for j, b in enumerate(bullets):
            para(f, x + 14, B_T + 126 + j * 21, cw - 28, "· " + b, "xs", 15)




if __name__ == "__main__":
    import io, sys
    io.open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/g5.svg", "w",
            encoding="utf-8").write(build())
    print("ok", H)
