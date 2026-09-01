# -*- coding: utf-8 -*-
"""图 G-1 —— 一颗 B200 全景：两个 die、148 个 SM、L2、HBM、NVLink。

对标 TPU 全景图的上半部分（chip → device → 计算核心 → 内存层级）。
关键对照点：TPU 一个 chip 对软件暴露成 2 个 device，B200 两个 die 对软件是 1 个 GPU。
"""
from common import Fig, BL, RD, YL, GN, PU, TL, INK, SUB, GREY, FILL

W = 1400
DIEW = 638
DIEX = [40, 722]
HBIX, HBIW = 684, 34

TOP = 84
DIE_T = 148
HDR_H = 26
HBM_T = DIE_T + HDR_H + 30
HBM_H = 56
L2_T = HBM_T + HBM_H + 26
L2_H = 50
GRID_T = L2_T + L2_H + 26
CELL_W, CELL_H, CGAP = 58, 22, 5
GRID_H = 8 * CELL_H + 7 * 4 + 34
DIE_B = GRID_T + GRID_H + 12
CHIP_B = DIE_B + 104
LINK_T = CHIP_B + 22
H = LINK_T + 200


def build():
    f = Fig(W, H, "一颗 NVIDIA B200 全景：双 die、148 个 SM、126 MB L2、"
                  "8 个 HBM3e 堆栈与第五代 NVLink")
    f.title("一颗 B200 全景　—— 两个 die 对软件是<tspan fill=\"#d93025\">一个</tspan> GPU"
            "　·　148 个 SM · 126 MB L2 · 8 TB/s HBM3e", "灰＝第三方来源")
    # 灰色在这张图里只有一个意思：非官方来源。所以下面三张互联卡片统一用 TL，
    # 不再有一张灰框卡片 —— 否则读者会以为「PCIe 这一栏是第三方数据」。
    f.legend([(BL, "计算：SM"), (RD, "启用/禁用边界"), (GN, "片上 SRAM"),
              (YL, "片外内存"), (TL, "互联通路"), (GREY, "灰字＝第三方来源，非 NVIDIA 官方")])

    # ── 芯片外框 ─────────────────────────────────────────────────────
    f.rect(20, TOP, 1360, CHIP_B - TOP, "#f8f9fa", INK, 2, 12)
    f.t(36, TOP + 22, "一颗 B200（封装级）", "sec")
    f.t(230, TOP + 22,
        "TSMC 4NP　·　2,080 亿晶体管　·　"
        "<tspan font-weight=\"700\" fill=\"#202124\">两个 reticle 尺寸的 die</tspan> 通过 NV-HBI 连成"
        "<tspan font-weight=\"700\" fill=\"#d93025\">单一 CUDA 设备</tspan>，L2 全局一致", "sm")
    f.t(230, TOP + 38,
        "↔ 对照 TPU v7：也是双 chiplet，但它<tspan font-weight=\"700\" fill=\"#202124\">反过来</tspan>暴露成 "
        "<tspan font-weight=\"700\" fill=\"#202124\">2 个独立 device</tspan>，两半各有自己的地址空间。"
        "同样的封装形态，软件视图完全相反。", "sm")

    for k, dx in enumerate(DIEX):
        _die(f, dx, k)

    # ── NV-HBI ───────────────────────────────────────────────────────
    f.rect(HBIX, DIE_T, HBIW, DIE_B - DIE_T, FILL[TL], TL, 2, 8)
    for i, ch in enumerate("NV-HBI"):
        f.t(HBIX + HBIW / 2, DIE_T + 74 + i * 17, ch, "box", TL, "middle")
    f.t(HBIX + HBIW / 2, DIE_T + 200, "10", "numb", TL, "middle")
    f.t(HBIX + HBIW / 2, DIE_T + 214, "TB/s", "xs", TL, "middle")
    f.t(HBIX + HBIW / 2, DIE_T + 250, "die 间", "xxs", TL, "middle")
    f.t(HBIX + HBIW / 2, DIE_T + 262, "一致", "xxs", TL, "middle")

    # ── 全片汇总 ─────────────────────────────────────────────────────
    y = DIE_B + 14
    f.rect(40, y, 1320, 62, "#fff", INK, 1.6, 8)
    f.t(54, y + 20, "全片合计", "box")
    items = [("SM", "148", "物理 160"), ("Tensor Core", "592", "148 × 4"),
             ("CUDA Core", "18,944", "148 × 128"),
             ("寄存器堆", "37 MiB", "256 KiB × 148"),
             ("L1 ＋ 共享内存", "37 MiB", "256 KiB × 148"),
             ("TMEM", "37 MiB", "256 KiB × 148"),
             ("L2", "126 MB", "4 个分区"),
             ("HBM3e", "192 GB", "8 堆栈")]
    cw = 1292 / len(items)
    for i, (k, v, d) in enumerate(items):
        x = 54 + i * cw
        f.t(x, y + 38, k, "xs")
        f.t(x, y + 54, v, "num", INK)
        f.t(x + 11.5 * len(v), y + 54, "  " + d, "xxs")

    f.t(54, y + 74, "↑ 中间三个 <tspan class=\"num\" fill=\"#202124\">37 MiB</tspan> 不是复制粘贴："
                     "寄存器堆、L1＋共享内存、TMEM 每个 SM 都正好 <tspan font-weight=\"700\" fill=\"#202124\">256 KiB</tspan>，"
                     "三块面积相当的 SRAM 分给了三种完全不同的用途。", "xs")

    _links(f)
    return f.out()


# ══════════════════════════════════════════════════════════════════════
def _die(f, dx, k):
    f.rect(dx, DIE_T, DIEW, DIE_B - DIE_T, "#fff", INK, 1.8, 10)
    f.rect(dx, DIE_T, DIEW, HDR_H, INK, rx=10)
    f.rect(dx, DIE_T + HDR_H - 10, DIEW, 10, INK, rx=0)
    f.t(dx + 12, DIE_T + 18, f"die {k}", "box", "#fff")
    f.t(dx + DIEW - 12, DIE_T + 18,
        "物理 80 个 SM　·　启用 74 个", "xs", "#c9cbcf", "end")

    # HBM
    f.t(dx + 12, HBM_T - 8, "HBM3e　4 堆栈", "lbl", "#b06000")
    f.t(dx + 138, HBM_T - 8,
        "全片 8 堆栈 ＝ 物理 <tspan class=\"num\" fill=\"#b06000\">192 GB</tspan> · "
        "<tspan class=\"num\" fill=\"#b06000\">8.0 TB/s</tspan>"
        "　（对软件暴露 186 / 180 GB，随产品形态变）", "xs")
    sw = (DIEW - 24 - 3 * 13) / 4
    for i in range(4):
        x = dx + 12 + i * (sw + 13)
        f.rect(x, HBM_T, sw, HBM_H - 18, FILL[YL], YL, 1.4, 5)
        f.t(x + sw / 2, HBM_T + 17, "HBM3e", "lbl", "#b06000", "middle")
        f.t(x + sw / 2, HBM_T + 31, "≈24 GB", "xs", None, "middle")
        f.line(x + sw / 2, HBM_T + HBM_H - 18, x + sw / 2, L2_T - 2, YL, 1.6, "aK")

    # L2 分区
    f.rect(dx + 12, L2_T, DIEW - 24, L2_H, FILL[GN], GN, 1.8, 7)
    f.line(dx + 12 + (DIEW - 24) / 2, L2_T + 4, dx + 12 + (DIEW - 24) / 2, L2_T + L2_H - 4, GN, 1.2, dash="4,3")
    f.t(dx + 22, L2_T + 19, f"die {k} 的 L2　63 MB", "box", GN)
    mid = dx + 12 + (DIEW - 24) / 2
    f.t(dx + 22, L2_T + 37,
        "<tspan fill=\"#9aa0a6\">实测本分区 21 TB/s（第三方 Vulkan 压测）</tspan>", "xs")
    f.t(mid + 12, L2_T + 19,
        "内部再切 <tspan font-weight=\"700\" fill=\"#202124\">2 个分区</tspan>（全片 4 个，Hopper 的两倍）", "xs")
    f.t(mid + 12, L2_T + 37,
        "<tspan fill=\"#9aa0a6\">跨到对面 die 掉到 16.8 TB/s，延迟也变高</tspan>", "xs")

    # SM 网格
    f.t(dx + 12, GRID_T + 12, "SM 阵列", "lbl", BL)
    f.t(dx + 82, GRID_T + 12,
        "每格 ＝ 1 个 SM。<tspan fill=\"#d93025\">红框 6 个是出厂禁用的</tspan>"
        "，用来提良率 —— 148 ＝ (80−6) × 2", "xs")
    f.t(dx + 12, GRID_T + 28,
        "<tspan fill=\"#9aa0a6\">GPC 分组：Blackwell Ultra 官方是 8 个 GPC / 160 SM；B200 第三方报 10 个 GPC。存疑，图上不画 GPC 边界。</tspan>", "xs")
    gy = GRID_T + 34
    for r in range(8):
        for c in range(10):
            i = r * 10 + c
            off = i >= 74
            f.rect(dx + 12 + c * (CELL_W + CGAP), gy + r * (CELL_H + 4),
                   CELL_W, CELL_H,
                   "#fff" if off else FILL[BL], RD if off else BL,
                   1.4 if off else 0.9, 3, "3,2" if off else None)
            if off:
                f.t(dx + 12 + c * (CELL_W + CGAP) + CELL_W / 2,
                    gy + r * (CELL_H + 4) + 15, "禁用", "xxs", RD, "middle")


# ══════════════════════════════════════════════════════════════════════
def _links(f):
    f.t(20, LINK_T, "往外的三条路　—— 一颗 GPU 不是孤岛", "sec")
    y = LINK_T + 12
    cards = [
        (TL, "NVLink 5　对外互联",
         ["18 条链路 × 双向 100 GB/s ＝ <tspan font-weight=\"700\" fill=\"#202124\">1.8 TB/s</tspan> / GPU",
          "NVL72 机架内 72 张卡全互联，域内总带宽 130 TB/s",
          "↔ TPU v7 是 3D torus，每片 1.2 TB/s、只连 6 个邻居",
          "拓扑差别比带宽差别重要：全交叉 vs 环面"]),
        (TL, "NVLink-C2C　接 Grace CPU",
         ["GB200 超级芯片 ＝ 1 个 Grace ＋ 2 个 B200",
          "CPU 内存对 GPU 是可寻址的，不必显式拷贝",
          "↔ TPU 侧是 4 chip / VM，走主机接口 HIB",
          "<tspan fill=\"#9aa0a6\">C2C 具体带宽本图未查实</tspan>"]),
        (TL, "PCIe Gen6　接主机与网卡",
         ["管理面 + 数据装载",
          "NVLink 连通的两点之间，运行时自动走 NVLink 不走 PCIe",
          "要点对点直传仍需显式 cudaDeviceEnablePeerAccess",
          "Gen6 ×16 双向合计 256 GB/s —— 约为 NVLink 5 的 <tspan font-weight=\"700\" fill=\"#202124\">1/7</tspan>"]),
    ]
    cw = (1360 - 2 * 16) / 3
    for i, (c, ttl, lines) in enumerate(cards):
        x = 20 + i * (cw + 16)
        f.rect(x, y, cw, 148, "#fff", c, 1.6, 9)
        f.rect(x, y, cw, 28, FILL[c], rx=9)
        f.rect(x, y + 19, cw, 9, FILL[c], rx=0)
        f.t(x + 12, y + 19, ttl, "box", c)
        for j, s in enumerate(lines):
            if s:
                f.t(x + 12, y + 50 + j * 20, "· " + s, "xs")


if __name__ == "__main__":
    import io, sys
    io.open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/g1.svg", "w",
            encoding="utf-8").write(build())
    print("ok", H)
