# -*- coding: utf-8 -*-
r"""外传 图 X-3 · **一颗 H100 拆开看** —— 同样的画法，相反的选择。

════════════════════════════════════════════════════════════════════
⭐⭐ 这一张的唯一职责：跟 X-2 并排读
════════════════════════════════════════════════════════════════════
所以它**必须跟 X-2 用同一套语汇、同一个比例尺**：

  · 同样的三层结构：算的地方 → 片上存 → 那道门 → 片外主存
  · **门的宽度按带宽等比画** ——&nbsp;这是两张图之间唯一的定量联系：
        v6e   1,638 GB/s → 10.0 px
        H100  3,350 GB/s → 20.5 px   （10 × 3350 ÷ 1638）
    ⛔ 这个比例常数写在两个脚本里**各一份**，改一边必须改另一边。
      两张图的门要是不成比例，读者会读出一个假的结论 ——&nbsp;
      **而「比例画错」在单张图上永远看不出来。**

════════════════════════════════════════════════════════════════════
⭐ 三个相反的选择（这一讲真正要讲的东西）
════════════════════════════════════════════════════════════════════
① **算力：集中 vs 摊开**
   v6e 把算力压进 **2 块 256×256** 的大方阵；
   H100 摊成 **132 个 SM × 4 个 Tensor Core ＝ 528 个**小单元。
② **片上存：编译器管 vs 硬件管**
   v6e 是**一整块 128 MiB 的暂存**，放什么由编译期写死；
   H100 是 **256 KB/SM 的 L1＋共享**（可配 228 KB 给共享）＋ **50 MB 的 L2**，
   后者**硬件自动管**。⭐ v6e 那一层在 JAX 源码里直接是 `CMEM = 0`——&nbsp;**没有**。
③ **门：窄 vs 宽**
   1,638 对 3,350 GB/s。

📌 一个值得单独说的账（两边都能当场加出来）：
   H100 片上合计 ≈ 132 × 256 KB ＋ 50 MB ≈ **84 MB**；
   v6e 光 VMEM 就是 **128 MiB ≈ 134 MB**。
   ⭐ **v6e 的片上暂存，比 H100 的全部片上内存还多约 6 成。**
   ——&nbsp;门窄，但屋里的台面更大。这两件事是配套的，不是矛盾的。

⛔ H100 的算力一律取**稠密 989.5**（数据表印的 1,979 带稀疏）。
"""
from topic03_draw import (Fig, wpx, _sz, LINE, LINE2,
                          BL, OR, GR, RD, GY, GY2, PU, CY, INK)

W = 1400
# ⛔ 跟 topic02x-fig-v6e.py 共用的比例常数。改一处必须改两处。
PX_PER_GBS = 10.0 / 1638.0


def main():
    f = Fig(W, "一颗 NVIDIA H100 SXM 的内部布局：132 个 SM，每个 SM 有 128 个 "
               "CUDA Core、4 个第四代 Tensor Core、256 KB 的 L1 与共享内存、"
               "256 KB 寄存器堆；全片共享 50 MB L2；片外 80 GB HBM3，"
               "带宽 3.35 TB/s；对外 NVLink 900 GB/s")
    f.marks = set()
    y = f.header(
        '一颗 NVIDIA H100 拆开看 ——&#160;'
        '<tspan font-weight="700">同样的画法，相反的选择</tspan>',
        '⭐ 跟上一张比着看：<tspan font-weight="700">算力摊成 528 个小单元</tspan>、'
        '<tspan font-weight="700">片上多了一整层硬件自动管的缓存</tspan>、'
        '<tspan font-weight="700">通往片外的门宽一倍</tspan>。',
        [(BL, "算：SM 与 Tensor Core"), (OR, "存：软件管的 L1 / 共享"),
         (CY, "存：硬件管的 L2 ——&#160;v6e 没有这一层"), (RD, "片外主存 HBM3")])

    CH_X, CH_W = 0, 900
    ch_top = y + 8
    SM_H = 196
    L2_H = 66
    CH_H = 40 + SM_H + 14 + L2_H + 18

    f.box(CH_X, ch_top, CH_W, CH_H, "#fff", LINE, 10)
    f.t(CH_X + 16, ch_top + 24, "一颗 H100 SXM", INK, bold=True, size=14)
    f.t(CH_X + 168, ch_top + 24,
        "8 GPC · 66 TPC · 每 TPC 2 个 SM", GY2, size=_sz(11))
    f.t(CH_X + CH_W - 16, ch_top + 24, "989.5 TFLOP/s bf16 稠密",
        "#a50e0e", bold=True, size=13, anchor="end")

    # ── SM 阵列 ─────────────────────────────────────────────────
    SX, SY = CH_X + 16, ch_top + 40
    SW_ = CH_W - 32
    sy = f.panel(SX, SY, SW_, SM_H, "SM × 132", BL,
                 sub="算力摊在 132 个小核里 ——&#160;v6e 是 1 个大核",
                 tag="每格 ＝ 1 个 SM")
    # 132 个小格：12 行 × 11 列 太挤，画 6 × 22
    ONE_W = 250
    ONE = SX + SW_ - 20 - ONE_W
    COLS, ROWS = 22, 6
    gw, gh, gp = 15.0, 12.0, 3.0
    gx0 = SX + 20
    gy0 = sy + 16
    for r in range(ROWS):
        for c in range(COLS):
            f.box(gx0 + c * (gw + gp), gy0 + r * (gh + gp), gw, gh,
                  "#e8f0fe", BL, 2, 0.7)
    # ⛔ 一行写不下就拆两行 ——&nbsp;护栏刚拦下过一次（要 1327px 只有 828px）。
    f.t(gx0, gy0 + ROWS * (gh + gp) + 16,
        "每个 SM 里：<tspan font-weight=\"700\">128 个 CUDA Core</tspan> ＋ "
        "<tspan font-weight=\"700\">4 个第四代 Tensor Core</tspan>",
        GY, size=_sz(11), w=SW_ - 40)
    f.t(gx0, gy0 + ROWS * (gh + gp) + 34,
        "全片合计 <tspan font-weight=\"700\">16,896</tspan> 个 CUDA Core、"
        "<tspan font-weight=\"700\">528</tspan> 个 Tensor Core",
        GY, size=_sz(11), w=SW_ - 40)

    # ⛔ 这里原来用 cell()，而 cell() 在 sub=None 时把主标题画在**框的垂直正中**。
    #   我按「标题在顶上」摆了下面三行 ——&nbsp;于是标题正压在第一行上。
    #   ⚠️ 几何探针没报：两行相距 10px、字高约 12px，垂直重叠只有 2px，
    #     没到「重叠过半才算撞车」那道阈值。**阈值是为「同一行」设的，
    #     它挡不住「差半行」这种压字。** 眼睛看得见，探针看不见。
    #   ⭐ 判据：**别去猜某个基元把字画在哪 ——&nbsp;要摆多行就自己画框。**
    f.box(ONE, sy + 14, ONE_W, 108, "#fff", OR, 6)
    f.t(ONE + 14, sy + 36, "一个 SM 里的存", "#b06000", bold=True, size=_sz(12))
    f.t(ONE + 14, sy + 60, "L1 ＋ 共享内存　256 KB", INK, size=_sz(11), mono=True)
    f.t(ONE + 14, sy + 80, "其中共享可配到 228 KB", GY, size=_sz(11))
    f.t(ONE + 14, sy + 100, "寄存器堆　256 KB", GY, size=_sz(11), mono=True)

    # ── L2 ──────────────────────────────────────────────────────
    L2_Y = SY + SM_H + 14
    f.box(SX, L2_Y, SW_, L2_H, "#fff", LINE, 8)
    f.box(SX, L2_Y, 4, L2_H, CY, CY, 2)
    f.box(SX + 2, L2_Y, 3, L2_H, "#fff", "#fff", 0)
    f.t(SX + 16, L2_Y + 25, "L2 缓存　50 MB", "#007b83", bold=True, size=13)
    f.t(SX + 220, L2_Y + 25,
        "全片共享，<tspan font-weight=\"700\">硬件自动管</tspan> ——&#160;"
        "程序管不着它留什么、赶走什么", GY, size=_sz(11))
    f.t(SX + 16, L2_Y + 48,
        "⭐⭐ <tspan font-weight=\"700\">v6e 完全没有这一层</tspan>"
        "——&#160;JAX 源码里 v6e 的 CMEM 直接写着 0。"
        "片上放什么，那边全部由编译期决定。", GY, size=_sz(11), w=SW_ - 32)

    # ── 门 ＋ HBM ───────────────────────────────────────────────
    HB_Y = ch_top + CH_H + 58
    f.box(CH_X, HB_Y, CH_W, 76, "#fff", LINE, 9)
    f.box(CH_X, HB_Y, 4, 76, RD, RD, 2)
    f.box(CH_X + 2, HB_Y, 3, 76, "#fff", "#fff", 0)
    f.t(CH_X + 16, HB_Y + 25, "HBM3　80 GB", "#a50e0e", bold=True, size=14)
    f.t(CH_X + 214, HB_Y + 25, "容量是 v6e 的 2.5 倍（80 对 32）",
        GY, size=_sz(11))
    f.t(CH_X + 16, HB_Y + 52,
        "⭐ 通往片上的带宽 <tspan font-weight=\"700\">3,350 GB/s</tspan>"
        " ——&#160;v6e 是 1,638，<tspan font-weight=\"700\">这道门宽了一倍</tspan>",
        GY, size=_sz(11))

    pw = 3350.0 * PX_PER_GBS          # ⭐ 跟 X-2 同一个比例尺
    cx = CH_X + CH_W / 2.0
    ytop, ybot = ch_top + CH_H, HB_Y
    f.poly("M %d %d L %d %d L %.1f %d L %.1f %d Z"
           % (CH_X + 150, ytop, CH_X + CH_W - 150, ytop,
              cx + pw / 2, ybot, cx - pw / 2, ybot), "#fce8e6")
    f.line(CH_X + 150, ytop, cx - pw / 2, ybot, RD, 1.4, arrow=False)
    f.line(CH_X + CH_W - 150, ytop, cx + pw / 2, ybot, RD, 1.4, arrow=False)
    f.box(cx - pw / 2, ybot - 14, pw, 16, RD, RD, 2)
    f.t(cx + pw / 2 + 18, ybot - 20,
        '<tspan font-weight="700">3,350 GB/s</tspan>　'
        '⭐ 这道门的宽度跟上一张<tspan font-weight="700">按同一比例画</tspan>',
        "#a50e0e", size=_sz(12))

    # ── 右栏：三个相反的选择 ────────────────────────────────────
    RX = CH_W + 36
    RW = W - RX
    ry = f.panel(RX, ch_top, RW, 300, "三个相反的选择", RD,
                 tag="这一讲的全部内容")
    OPP = (
        ("算力怎么摆", "v6e：2 块大方阵", "H100：528 个小单元",
         "一个是「一次吞一大块」，一个是「同时应付很多小块」"),
        ("片上谁做主", "v6e：编译期写死", "H100：多一层硬件缓存",
         "v6e 的 CMEM ＝ 0，那一层它压根没有"),
        ("门有多宽", "v6e：1,638 GB/s", "H100：3,350 GB/s",
         "⛔ 这一条直接决定了那个 560 对 295"),
    )
    for i, (k, a, b, note) in enumerate(OPP):
        yy = ry + 30 + i * 90
        f.t(RX + 16, yy, k, INK, bold=True, size=_sz(12))
        f.t(RX + 16, yy + 22, a, "#a50e0e", size=_sz(11))
        f.t(RX + 16, yy + 40, b, "#174ea6", size=_sz(11))
        f.t(RX + 16, yy + 60, note, GY2, size=_sz(11), w=RW - 32)

    ry2 = f.panel(RX, ch_top + 316, RW, 132, "一个反直觉的账", GR)
    f.t(RX + 16, ry2 + 26, "H100 全部片上内存", GY, size=_sz(11))
    f.t(RX + 16, ry2 + 46, "132 × 256 KB ＋ 50 MB ≈ 84 MB",
        INK, size=_sz(11), mono=True)
    f.t(RX + 16, ry2 + 72, "v6e 光 VMEM 一项", GY, size=_sz(11))
    f.t(RX + 16, ry2 + 92, "128 MiB ≈ 134 MB", "#0d652d",
        bold=True, size=_sz(12), mono=True)
    f.t(RX + 16, ry2 + 116,
        "⭐ 门窄，但屋里的台面更大 ——&#160;这两件事是配套的",
        "#0d652d", size=_sz(11), w=RW - 32)

    y = HB_Y + 76 + 24

    y = f.band(y, "info",
               "两颗芯片在同一件事上做了相反的选择 ——&#160;而它们各自都是自洽的",
               ['H100 摊成 528 个小单元、再压一层硬件缓存，是为了'
                '<tspan font-weight="700">应付「我不知道你要跑什么」</tspan>：'
                '形状随时会变、访问随时可能落空，那就多留后手。',
                'v6e 压成 2 块大方阵、片上全交给编译器，是为了'
                '<tspan font-weight="700">吃透「我早就知道你要跑什么」</tspan>：'
                '形状固定、访问可提前排，后手就是浪费，不如把面积全给算力和台面。',
                '⭐⭐ 所以问题从来不是「谁更强」，是'
                '<tspan font-weight="700">你手上的活属于哪一种</tspan>。'
                '——&#160;下一张就看扩散模型的活长什么样。'])

    y = f.src(y + 18,
              'H100 SXM：132 SM（8 GPC / 66 TPC）、128 CUDA Core 与 4 个第四代 '
              'Tensor Core 每 SM、L1＋共享 256 KB／SM（共享可配 228 KB）、'
              '寄存器堆 256 KB／SM、L2 50 MB ——&#160;NVIDIA《Hopper Architecture In-Depth》',
              '稠密 bf16 989.5 TFLOPS（数据表 1,979 带稀疏 ÷ 2）、HBM3 80 GB / 3.35 TB/s、'
              'NVLink 900 GB/s ——&#160;NVIDIA H100 官方数据表。'
              '⚠️ 84 MB 那个合计是本图当场加的：132 × 256 KB ＝ 33.8 MB，加 L2 50 MB。')
    f.save("figx-3.svg", y + 6)


main()
