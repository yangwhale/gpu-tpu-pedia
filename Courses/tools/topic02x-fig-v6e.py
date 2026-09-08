# -*- coding: utf-8 -*-
r"""外传 图 X-2 · **一颗 TPU v6e 拆开看** —— 那个 560 是从这张布局里长出来的。

════════════════════════════════════════════════════════════════════
⭐ 这张图跟专题二 T-1 的关系
════════════════════════════════════════════════════════════════════
T-1 拆的是 **v7**：一个封装两个 chiplet、对软件是两个 device。
这一张拆 **v6e**，而它最该被记住的一条恰好是**反过来的**：

    v6e：**1 chip ＝ 1 TensorCore ＝ 1 device**（官方规格表）

⭐⭐ 于是专题二反复强调的那个「所有容量除以 2 才是你的额度」的坑，
   **在这一代根本不存在**。讲 v6e 时不要顺手把 v7 的口径搬过来。

════════════════════════════════════════════════════════════════════
⛔ L100 的画法边界
════════════════════════════════════════════════════════════════════
① **只画「有几个、多大、连到哪」**，不画「谁决定搬运」那一层 ——&nbsp;
   那是专题二 L200/L300 的活，二十分钟讲不动。
② **每个框最多两行字。** 这一讲是给没见过 TPU 的人看的。
③ 数字只留会在后面被用到的：**918 / 1,638 / 32 GB / 128 MiB**。
   其余（SparseCore 的 lane 数之类）只标存在，不展开。
"""
from topic03_draw import (Fig, wpx, _sz, LINE, LINE2,
                          BL, OR, GR, RD, GY, GY2, PU, CY, INK)

W = 1400


def main():
    f = Fig(W, "一颗 TPU v6e 芯片的内部布局：1 个 TensorCore，内含 2 个 "
               "256×256 的 MXU、1 个 8×128 的向量单元、1 个标量单元，"
               "配 128 MiB 的 VMEM 片上暂存和 1 MiB SMEM；另有 2 个 SparseCore；"
               "片外是 32 GB HBM，带宽 1638 GB/s；对外 4 个 ICI 端口组成二维环面")
    f.marks = set()
    y = f.header(
        '一颗 TPU v6e 拆开看 ——&#160;'
        '<tspan font-weight="700">1 颗芯片 ＝ 1 个核 ＝ 1 个 device</tspan>',
        '⭐ 记住三件事就够了：<tspan font-weight="700">算力集中在 2 个大方阵里</tspan>、'
        '<tspan font-weight="700">片上有一整块 128 MiB 的暂存</tspan>、'
        '<tspan font-weight="700">通往片外的那根管子只有 1,638 GB/s</tspan>。',
        [(BL, "算：矩阵与向量单元"), (OR, "存：片上暂存"),
         (RD, "片外主存 HBM"), (PU, "SparseCore（本讲不展开）")])

    # ════════════════════════════════════════════════════════════
    # 主体：一颗 chip
    # ════════════════════════════════════════════════════════════
    # ⛔ 这几个高度**按内容加出来**，不要拍。第一版 TC_H 拍了 240，
    #   结果 VMEM 那一排顶出了 TensorCore 面板、压在 SparseCore 上 ——&nbsp;
    #   几何探针查不出（重叠的是框不是字），只有渲染出来盯着看才发现。
    #   ⭐ 判据：**容器高度是被内容决定的，不是被审美决定的。**
    CH_X, CH_W = 0, 900
    ch_top = y + 8
    TC_H = 30 + 18 + 96 + 42 + 62 + 44 + 16          # 标题栏＋MXU＋注＋存储＋注
    CH_H = 40 + TC_H + 14 + 54 + 18
    f.box(CH_X, ch_top, CH_W, CH_H, "#fff", LINE, 10)
    f.t(CH_X + 16, ch_top + 24, "一颗 v6e 芯片", INK, bold=True, size=14)
    f.t(CH_X + 150, ch_top + 24,
        "对软件 ＝ 1 个 device　⭐ 不用除以 2", GY2, size=_sz(11))
    f.t(CH_X + CH_W - 16, ch_top + 24, "918 TFLOP/s bf16",
        "#a50e0e", bold=True, size=13, anchor="end")

    # ── TensorCore ──────────────────────────────────────────────
    TC_X, TC_Y = CH_X + 16, ch_top + 40
    TC_W = CH_W - 32
    tcy = f.panel(TC_X, TC_Y, TC_W, TC_H, "TensorCore × 1", BL,
                  sub="v7 是 2 个 ——&#160;这一代只有 1 个", tag="稠密算力全在这里面")

    # 两个 MXU
    MX_W, MX_H = 196, 96
    for i in range(2):
        x = TC_X + 20 + i * (MX_W + 16)
        f.cell(x, tcy + 18, MX_W, MX_H, "MXU %d" % i, "256 × 256",
               BL, "#fff", 13, grid=True)
    f.t(TC_X + 20, tcy + 18 + MX_H + 18,
        "⭐ 矩阵乘全部在这两块里发生", "#174ea6", size=_sz(11))
    f.t(TC_X + 20, tcy + 18 + MX_H + 36,
        "一整块 256×256 的方阵，不切成小片", GY, size=_sz(11))

    # VPU ＋ 标量
    VX = TC_X + 20 + 2 * (MX_W + 16) + 14
    f.cell(VX, tcy + 18, 176, 96, "向量单元 VPU", "8 × 128", GR)
    f.cell(VX + 190, tcy + 18, 150, 96, "标量单元", "发指令 · 发搬运", GY)
    f.t(VX, tcy + 18 + MX_H + 18,
        "归一化、激活、softmax 走这儿", GY, size=_sz(11))

    # 片上暂存
    SB_Y = tcy + 18 + MX_H + 42
    f.cell(TC_X + 20, SB_Y, 560, 62, "VMEM　128 MiB", "片上暂存（不是缓存）", OR)
    f.cell(TC_X + 20 + 574, SB_Y, 232, 62, "SMEM　1 MiB", "标量 / 描述符", OR)
    # ⛔ 这一句原来放在 SMEM 右边，直接顶出了芯片外框、末尾「不用切」被裁掉。
    #   ⭐ 改到整排下面独占一行 ——&nbsp;**右侧是版面最容易透支的方向，
    #     因为那边没有东西挡着，溢出既不报错也不产生滚动条。**
    f.t(TC_X + 20, SB_Y + 62 + 20,
        '⭐ <tspan font-weight="700">v7 是「两个核各 64 MiB」，v6e 是「一个核独占 128 MiB」</tspan>'
        '——&#160;总量一样，但 v6e 不用把大张量切两半',
        GY, size=_sz(11), w=TC_W - 40)

    # ── SparseCore ──────────────────────────────────────────────
    SC_Y = TC_Y + TC_H + 14
    f.box(TC_X, SC_Y, TC_W, 54, "#fff", LINE, 8)
    f.box(TC_X, SC_Y, 4, 54, PU, PU, 2)
    f.box(TC_X + 2, SC_Y, 3, 54, "#fff", "#fff", 0)
    f.t(TC_X + 16, SC_Y + 22, "SparseCore × 2", "#681da8", bold=True, size=13)
    f.t(TC_X + 160, SC_Y + 22,
        "专门搬稀疏 / 大表的协处理器 ——&#160;本讲不展开", GY, size=_sz(11))
    f.t(TC_X + 160, SC_Y + 40,
        "扩散模型用不到它；它是给推荐系统那类负载准备的", GY2, size=_sz(11))

    # ── HBM ＋ 那根管子 ─────────────────────────────────────────
    HB_Y = ch_top + CH_H + 58        # ⭐ 拉开，好让那根管子看得出是「一段细颈」
    f.box(CH_X, HB_Y, CH_W, 76, "#fff", LINE, 9)
    f.box(CH_X, HB_Y, 4, 76, RD, RD, 2)
    f.box(CH_X + 2, HB_Y, 3, 76, "#fff", "#fff", 0)
    f.t(CH_X + 16, HB_Y + 25, "HBM　32 GB", "#a50e0e", bold=True, size=14)
    f.t(CH_X + 200, HB_Y + 25, "片外主存 ——&#160;模型和中间结果都住这儿",
        GY, size=_sz(11))
    f.t(CH_X + 16, HB_Y + 52,
        "⛔ 通往片上的带宽只有 <tspan font-weight="
        '"700">1,638 GB/s</tspan> ——&#160;'
        'H100 是 3,350，<tspan font-weight="700">只有它的 49%</tspan>',
        GY, size=_sz(11))

    # ⭐⭐ 这一段是全图的隐喻：**芯片很宽，通往片外的口很窄。**
    #   ⛔ 第一版画成一小截短粗的红条，「窄」完全读不出来 ——&nbsp;
    #     隐喻画不出来就等于没画。改成一段明显的细颈：两侧各一条收口斜线，
    #     中间一条 10px 宽的通道，长度拉满整个间隙。
    pw, cx = 10.0, CH_X + CH_W / 2.0
    ytop, ybot = ch_top + CH_H, HB_Y
    f.poly("M %d %d L %d %d L %d %d L %d %d Z"
           % (CH_X + 150, ytop, CH_X + CH_W - 150, ytop,
              cx + pw / 2, ybot, cx - pw / 2, ybot), "#fce8e6")
    f.line(CH_X + 150, ytop, cx - pw / 2, ybot, RD, 1.4, arrow=False)
    f.line(CH_X + CH_W - 150, ytop, cx + pw / 2, ybot, RD, 1.4, arrow=False)
    f.box(cx - pw / 2, ybot - 14, pw, 16, RD, RD, 2)
    f.t(cx + 22, ybot - 20,
        '<tspan font-weight="700">1,638 GB/s</tspan>——&#160;整颗芯片的数据，都从这儿过',
        "#a50e0e", size=_sz(12))

    # ════════════════════════════════════════════════════════════
    # 右栏：对外与「记住这三条」
    # ════════════════════════════════════════════════════════════
    RX = CH_W + 36
    RW = W - RX
    ry = f.panel(RX, ch_top, RW, 178, "对外怎么连", CY, tag="官方规格表")
    rows = (("ICI 端口", "4 个", "v7 是 6 个"),
            ("ICI 带宽", "800 GB/s", "每 chip 双向合计"),
            ("拓扑", "2D 环面", "v7 是 3D"),
            ("一个 Pod", "256 颗", "v7 是 9,216 颗"),
            ("每台主机", "8 颗", "DRAM 1,536 GiB"))
    for i, (k, v, note) in enumerate(rows):
        yy = ry + 24 + i * 28
        f.t(RX + 16, yy, k, GY, size=_sz(11))
        f.t(RX + 108, yy, v, INK, bold=True, size=_sz(12), mono=True)
        f.t(RX + 220, yy, note, GY2, size=_sz(11))

    ry2 = f.panel(RX, ch_top + 194, RW, 216, "这一张只要记住三条", GR)
    KEEP = (
        ("①", "1 颗 ＝ 1 个核 ＝ 1 个 device",
         "专题二那个「除以 2」的坑，这一代没有"),
        ("②", "片上一整块 128 MiB 暂存",
         "大张量不用切成两半，编译器好排"),
        ("③", "片外那根管子只有 1,638 GB/s",
         "⛔ 这就是 560 的分母 ——&#160;全讲的症结"),
    )
    for i, (n, main, sub) in enumerate(KEEP):
        yy = ry2 + 30 + i * 62
        f.t(RX + 16, yy, n, "#0d652d", bold=True, size=15)
        f.t(RX + 42, yy, main, INK, bold=True, size=_sz(12))
        f.t(RX + 42, yy + 20, sub, GY, size=_sz(11))

    y = HB_Y + 76 + 24

    # ── 落点 ────────────────────────────────────────────────────
    y = f.band(y, "warn",
               "把这张布局压成一句话：算的地方很大，进料的门很窄",
               ['两块 <tspan font-weight="700">256×256</tspan> 的方阵摆在那儿，'
                '每一拍能吞下的乘加数是很大的；'
                '可所有数据进出片外，只能挤 <tspan font-weight="700">1,638 GB/s</tspan> 这一根管子。',
                '⭐ <tspan font-weight="700">这就是 560 的来历</tspan>：'
                '分子（算力）没少，分母（带宽）被砍掉一半，'
                '于是「每搬一个字节得算多少次才不亏」这个门槛，就抬到了别人的两倍。',
                '⭐⭐ 下一张把 H100 按同样的画法拆开 ——&#160;'
                '<tspan font-weight="700">看两颗芯片是怎么在同一件事上做了相反的选择的</tspan>。'])

    y = f.src(y + 18,
              '每 chip 1 个 TensorCore、2 个 MXU、1 个向量单元、1 个标量单元、'
              '918 TFLOPs bf16、HBM 32 GB / 1,638 GBps、ICI 800 GBps 4 端口、'
              '2D torus、Pod 256 chip、每 host 8 chip ——&#160;官方 v6e 规格表',
              'MXU 256×256、VMEM 128 MiB/core、SMEM 1 MiB、'
              'SparseCore 2 core × 16 subcore × 8 lane ——&#160;JAX 公开源码 tpu_info.py 的 TPU_V6E 分支')
    f.save("figx-2.svg", y + 6)


main()
