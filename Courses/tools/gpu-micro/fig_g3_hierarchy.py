# -*- coding: utf-8 -*-
"""图 G-3 —— 线程层级 ↔ 硬件归属：谁被钉在哪、谁能跟谁说话。

CUDA 的六层抽象（线程/warp/warpgroup/线程块/cluster/grid）不是纯软件概念，
每一层都精确对应一块硬件边界。这张图把「抽象层」和「它钉在哪块硅上」并排放。
"""
from common import Fig, para, BL, RD, YL, GN, PU, TL, INK, SUB, GREY, FILL

W = 1400
C1X, C1W = 20, 356
C2X, C2W = 388, 396
C3X, C3W = 800, 580

TOP = 84
ROWH, RGAP = 68, 8
ROWS_T = TOP + 30
NROW = 6
TPU_T = ROWS_T + NROW * (ROWH + RGAP) + 26
TPU_ROWH = 58
H = TPU_T + 40 + 5 * TPU_ROWH + 30

LV = [
    # (色, 层名, 英文, 规模, 硬件归属行, 能共享什么, 同步手段)
    (SUB, "线程", "thread", "1 个",
     "一个处理块里的 <b>1 条 lane</b>",
     "只有自己的寄存器 —— 别的线程<b>看不见</b>",
     "无"),
    (BL, "warp", "warp", "32 个线程",
     "<b>一个处理块</b>（sub-core）的 32 条 lane，一条指令一拍走完",
     "寄存器可以互相洗牌（<code>shfl</code>）—— 这是最快的通信",
     "天然同步；分支分歧只在这一层内发生"),
    (PU, "warp 组", "warpgroup", "4 个 warp ＝ 128 线程",
     "<b>一个 SM 的四个处理块各出一个 warp</b>",
     "Hopper 的 <code>wgmma</code>、Blackwell 的 TMEM 都以它为单位",
     "TMEM 按 warp 号切成四份：warp k 只能碰 lane 32k–32k+31"),
    (GN, "线程块", "CTA / block", "≤ 1,024 线程 ＝ ≤ 32 warp",
     "<b>整块钉死在一个 SM 上</b>，一旦落下就不迁走",
     "<b>共享内存</b>（≤227 KiB）—— 这是 CUDA 里最重要的一层作用域",
     "<code>__syncthreads()</code>；一个 SM 最多同时驻留 32 块"),
    (YL, "cluster", "thread block cluster", "≤ 8 块（H100 起可 opt-in 到 16）",
     "同一个 <b>GPC</b> 内的若干 SM",
     "<b>分布式共享内存</b>：能直接读写同 cluster 里别的块的共享内存",
     "cluster 级 barrier；Hopper 引入，Blackwell 沿用"),
    (RD, "grid", "grid", "整个 kernel 的全部线程块",
     "<b>整颗 GPU</b>（B200 是跨两个 die 的 148 个 SM）",
     "只剩 <b>L2 和 HBM</b> —— 已经没有片上快捷通道了",
     "只能靠原子操作 / 协作组 / 分 kernel"),
]


def build():
    f = Fig(W, H, "CUDA 线程层级与硬件归属对照：thread、warp、warpgroup、"
                  "线程块、cluster、grid 各自钉在哪一级硬件上")
    f.title("线程层级 ↔ 硬件归属　—— 每一层抽象都精确对应一条硬件边界，不是纯软件约定")
    f.legend([(BL, "warp：SIMT 的真正宽度"), (GN, "线程块：共享内存的作用域"),
              (YL, "cluster：分布式共享内存"), (RD, "grid：只剩 L2/HBM")])

    f.t(C1X + 12, TOP + 20, "软件抽象层", "sec")
    f.t(C2X + 12, TOP + 20, "钉在哪块硅上", "sec")
    f.t(C3X + 12, TOP + 20, "这一层内部能共享什么　·　怎么同步", "sec")

    for i, (c, name, en, scale, hw, share, sync) in enumerate(LV):
        y = ROWS_T + i * (ROWH + RGAP)

        # 左：层名 + 规模 + 小示意
        f.rect(C1X, y, C1W, ROWH, FILL[c], c, 1.8, 9)
        f.t(C1X + 14, y + 24, name, "sec", c)
        f.t(C1X + 14 + 17 * len(name), y + 24, en, "sm")
        f.t(C1X + 14, y + 44, scale, "lbl", INK)
        _icon(f, C1X + 14, y + 52, i, c)

        # 中间箭头
        f.line(C1X + C1W + 4, y + ROWH / 2, C2X - 4, y + ROWH / 2, c, 2, "aK")

        # 中：硬件归属
        f.rect(C2X, y, C2W, ROWH, "#fff", c, 1.4, 9)
        para(f, C2X + 14, y + 26, C2W - 28, hw, "lbl", 18)

        # 右：共享 + 同步
        f.rect(C3X, y, C3W, ROWH, "#fff", SUB, 1.2, 9)
        yy = para(f, C3X + 14, y + 26, C3W - 28, share, "lbl", 18)
        para(f, C3X + 14, yy + 8, C3W - 28, "同步：" + sync, "xs", 15)

    _tpu(f)
    return f.out()


def _icon(f, x, y, i, c):
    """左列的小示意：把「多少个」画出来，而不是只写数字。"""
    if i == 0:
        f.rect(x, y, 8, 12, c, rx=2)
    elif i == 1:
        for k in range(32):
            f.rect(x + k * 9.6, y, 8, 12, c, rx=2)
    elif i == 2:
        for g in range(4):
            f.rect(x + g * 80, y, 74, 12, FILL[c], c, 1, 3)
            f.t(x + g * 80 + 37, y + 10, f"warp {g}", "xxs", c, "middle")
    elif i == 3:
        for k in range(32):
            f.rect(x + k * 9.6, y, 8, 12, FILL[c] if k > 7 else c, c, 0.6, 2)
        pass
    elif i == 4:
        for k in range(8):
            f.rect(x + k * 26, y, 22, 12, FILL[c], c, 1, 3)
        f.t(x + 216, y + 10, "8 块（可 16）", "xxs", c)
    else:
        for k in range(24):
            f.rect(x + k * 13, y, 10, 12, FILL[c], c, 0.7, 2)
        f.t(x + 314, y + 10, "…", "xxs", c)


def _tpu(f):
    f.t(20, TPU_T, "同一张表，TPU 那边长什么样　—— 层数少得多，而且少掉的那几层，"
                   "正是 GPU 用来「藏延迟」的那几层", "sec")
    y = TPU_T + 14
    rows = [
        ("线程 / warp",
         "<b>没有对应物。</b>TPU 是显式向量机：一条向量指令直接吃一整个 8×128 的向量寄存器，没有「32 个线程」这层皮",
         "GPU 靠「很多 warp 轮流上」来藏访存延迟；TPU 没有这个机制，延迟必须在<b>编译期</b>排流水藏掉"),
        ("warp 组 warpgroup",
         "<b>没有对应物。</b>TPU 的 VLIW bundle 一拍 9 个发射槽同时发，哪条指令进哪个槽，编译期就钉死了"
         "<g>（9 槽是 v2/v3 的公开数字，v7 未公布）</g>",
         "GPU 在<b>运行时</b>挑指令，TPU 在<b>编译期</b>排指令 —— 这是两边最根本的分工差别"),
        ("线程块 ＋ 共享内存",
         "一个 TensorCore ＋ 它私有的 <b>64 MiB VMEM</b><g>（JAX 开源代码，非官方规格页）</g>",
         "容量差 <r>289 倍</r>（227 KiB vs 64 MiB）。TPU 的片上暂存大到能整块放下权重，GPU 只能一小片一小片地流"),
        ("cluster",
         "<b>没有对应物。</b>TPU 一个 chip 只有 2 个 TensorCore，本来也不需要「一组核共享暂存」这层",
         "GPU 要发明 cluster，恰恰是因为它有 148 个 SM 要协调；核少反而省掉一层抽象"),
        ("grid",
         "整颗 chip 的 2 个 TensorCore，再往外就是 ICI 3D torus 上的别的芯片",
         "GPU 的 grid 在<b>一颗</b>芯片内部就要协调 148 个 SM；TPU 的协调主要发生在<b>芯片之间</b>"),
    ]
    hdr = ["GPU 的这一层", "TPU v7 上对应什么", "差在哪"]
    cw = [230, 530, 570]
    cx = [40, 40 + cw[0], 40 + cw[0] + cw[1]]

    f.rect(20, y, 1360, 34 + len(rows) * TPU_ROWH, FILL[TL], TL, 1.8, 10)
    f.rect(30, y + 8, 1340, 24, "#fff", TL, 1, 5)
    for i, hh in enumerate(hdr):
        f.t(cx[i], y + 24, hh, "lbl", TL)
    for r, cells in enumerate(rows):
        ry = y + 34 + r * TPU_ROWH
        if r:
            f.line(30, ry - 4, 1370, ry - 4, "#b6dbd6", 1)
        f.t(cx[0], ry + 18, cells[0], "lbl", INK)
        para(f, cx[1], ry + 17, cw[1] - 24, cells[1], "xs", 15)
        para(f, cx[2], ry + 17, cw[2] - 24, cells[2], "xs", 15)


def _wrap(f, x, y, w, s, cls):
    """把 <b>…</b> 转成 tspan；不做真换行 —— 文案已按栏宽写好。"""
    s = s.replace("<b>", '<tspan font-weight="700" fill="#202124">').replace("</b>", "</tspan>")
    s = s.replace("<code>", '<tspan class="mono" fill="#1967d2">').replace("</code>", "</tspan>")
    f.t(x, y, s, cls)


if __name__ == "__main__":
    import io, sys
    io.open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/g3.svg", "w",
            encoding="utf-8").write(build())
    print("ok", H)
