# -*- coding: utf-8 -*-
"""图 T-1 —— 一颗 TPU v7 chip 全景：封装 → 两个 die → 核 → HBM → 对外互联。

这张图要立起来的是整份文档的坐标系：**chip 和 device 不是一回事**。
一颗 v7 封装里是两个 chiplet，各自带一个 TensorCore、两个 SparseCore、
和属于自己的那一半 HBM —— 而且它对软件**如实暴露成两个独立 device**。

这正好是 B200 的反面：同样是双 die 封装，NVIDIA 用一条一致性总线把两个 die
缝成「一个 GPU」，Google 则把缝留在外面让软件自己面对。所以这张图的落点不是
「TPU 里有什么」，而是「同一个物理难题，两家把它推给了不同的人」。
"""
from common import Fig, para, BL, RD, YL, GN, PU, TL, INK, SUB, GREY, FILL
import gate

W = 1400
TOP = 84

PKG_X, PKG_Y, PKG_W = 20, 112, 902
DIE_T, DIE_H, DIE_W = 152, 400, 420
D0X, D1X = 36, 486
D2D_Y = DIE_T + DIE_H + 12                 # 570
PKG_H = D2D_Y + 46 + 16 - PKG_Y            # 封装框底边贴在 D2D 带下方 16px

RX, RW = 940, 440                          # 右栏
ICI_Y = PKG_Y + PKG_H + 22
ICI_H = 158
NOTE_Y = ICI_Y + ICI_H + 14
H = NOTE_Y + 86


def build():
    f = Fig(W, H, "TPU v7 单芯片全景：一个封装内两个 chiplet，各含 1 个 TensorCore、"
                  "2 个 SparseCore 与一半 HBM，对软件暴露为两个独立 device")
    f.title("一颗 TPU v7 chip 全景　—— 封装里是两个 die，而它们对软件是两个「独立的」加速器",
            "第 1 / 8 张")
    # 图例只列这张图上真出现过的东西 —— 挂一个没有指涉物的条目（比如「官方未公开」）
    # 会让读者在图上四处找那个灰框，找不到就开始怀疑自己漏看了。
    f.legend([(BL, "TensorCore（稠密算力）"), (PU, "SparseCore（稀疏／通信）"),
              (GN, "MXU 脉动阵列"), (YL, "存储"), (RD, "互连")])

    # MXU 的点阵纹理：256×256 写成字没有感觉，铺成一片密点才有
    f.raw('<defs><pattern id="mxucell" width="3" height="3" patternUnits="userSpaceOnUse">'
          f'<rect width="3" height="3" fill="{FILL[GN]}"/>'
          f'<rect width="1.6" height="1.6" x="0.2" y="0.2" fill="{GN}" opacity="0.34"/>'
          '</pattern></defs>')

    _package(f)
    _sidebar(f)
    _ici(f)
    _note(f)
    return f.out()


# ══════════════════════════════════════════════════════════════════════
def _package(f):
    f.rect(PKG_X, PKG_Y, PKG_W, PKG_H, "#fff", INK, 2.2, 12)
    f.t(PKG_X + 16, PKG_Y + 26, "一个封装 ＝ 一颗 chip", "sec")
    f.t(PKG_X + 190, PKG_Y + 26,
        gate.IP("内部代号 Ghostfish（gfc）· 对外名 Ironwood / TPU v7",
                "对外名 Ironwood / TPU v7", why="内部代号"), "xs")
    f.t(PKG_X + PKG_W - 16, PKG_Y + 26,
        "峰值 bf16 2,307 TFLOP/s　·　fp8 4,614 TFLOP/s", "lbl", INK, "end")

    _die(f, D0X, 0)
    _die(f, D1X, 1)
    _d2d(f)


def _die(f, dx, idx):
    """两个 die 结构完全一样，所以**结构照画、说明只写一遍**。

    上一版两个 die 各自带全套说明文字，一眼看过去像两张不同的图，读者会下意识
    去比对哪里不一样 —— 而答案是「没有不一样」。对称该由形状来说，不该靠把字誊两遍。
    """
    full = (idx == 0)                       # 只有 die 0 带解释性文字
    f.rect(dx, DIE_T, DIE_W, DIE_H, BG_DIE, SUB, 1.6, 10)

    # ── die 抬头：这一行是整张图的题眼 ────────────────────────────────
    f.rect(dx, DIE_T, DIE_W, 30, FILL[SUB], rx=10)
    f.rect(dx, DIE_T + 20, DIE_W, 10, FILL[SUB], rx=0)
    f.t(dx + 12, DIE_T + 20, f"chiplet / die {idx}", "box")
    f.t(dx + DIE_W - 12, DIE_T + 20,
        f"对软件 ＝ 独立的 device {idx}", "lbl", RD, "end")

    # ── TensorCore ───────────────────────────────────────────────────
    tx, ty, tw = dx + 10, DIE_T + 40, DIE_W - 20
    f.rect(tx, ty, tw, 196, "#fff", BL, 1.8, 8)
    f.rect(tx, ty, tw, 24, FILL[BL], rx=8)
    f.rect(tx, ty + 16, tw, 8, FILL[BL], rx=0)
    f.t(tx + 10, ty + 17, "TensorCore　×1　@ 2.2 GHz", "box", BL)
    f.t(tx + tw - 10, ty + 17, "两个 MXU 合计 131,072 个乘加单元", "xxs", BL, "end")

    for k in range(2):
        _mxu(f, tx + 12 + k * 104, ty + 34)
    _vpu(f, tx + 232, ty + 34, tw - 244, full)

    # 片上暂存：容量是公开的（JAX 开源代码里写着），带宽不是
    f.rect(tx + 12, ty + 148, 226, 38, FILL[YL], YL, 1.4, 6)
    f.t(tx + 20, ty + 165, "VMEM　64 MiB / core", "lbl", INK)
    f.t(tx + 20, ty + 180,
        "软件自己搬进搬出的暂存，不是缓存" if full else "同左", "xxs")
    f.rect(tx + 246, ty + 148, tw - 258, 38, FILL[YL], YL, 1.4, 6)
    f.t(tx + 254, ty + 165, "SMEM 1 MiB", "lbl", INK)
    f.t(tx + 254, ty + 180, "标量／描述符" if full else "同左", "xxs")

    # ── SparseCore ───────────────────────────────────────────────────
    sx, sy = dx + 10, DIE_T + 246
    f.rect(sx, sy, DIE_W - 20, 64, "#fff", PU, 1.8, 8)
    f.t(sx + 10, sy + 18, "SparseCore　×2", "box", PU)
    if full:
        f.t(sx + DIE_W - 30, sy + 18, "每个 ＝ 1 个标量序列器 ＋ 16 个向量 tile", "xxs", PU, "end")
    for k in range(2):
        bx = sx + 10 + k * 194
        f.rect(bx, sy + 26, 186, 28, FILL[PU], PU, 1.1, 5)
        f.t(bx + 8, sy + 44, f"SC {idx*2+k}", "lbl", PU)
        for j in range(16):                       # 16 个 tile，画出来才知道它有多宽
            f.rect(bx + 48 + j * 7.7, sy + 34, 5.6, 12, PU, rx=1)

    # ── HBM ──────────────────────────────────────────────────────────
    hx, hy = dx + 10, DIE_T + 320
    f.rect(hx, hy, DIE_W - 20, 64, FILL[YL], YL, 1.8, 8)
    f.t(hx + 10, hy + 20, "HBM3E　96 GiB", "box", INK)
    if full:
        f.t(hx + 10, hy + 38, "这一半只属于 die 0 —— 两个 die 不共享地址空间", "xs")
        f.t(hx + 10, hy + 54,
            gate.IP("整封装 8 个 HBM3E 堆栈、7.4 TB/s；按 core 拆是约 3,433 GiB/s",
                    "整封装 8 个 HBM3E 堆栈、7.4 TB/s；按 die 拆分的口径官方未公开",
                    why="片上带宽"), "xxs")
    else:
        f.t(hx + 10, hy + 38, "另一半属于 die 1，两套地址空间互不可见", "xs")
    f.t(hx + DIE_W - 30, hy + 20, "＝ 封装 192 GiB 的一半", "xxs", None, "end")


def _mxu(f, x, y):
    """MXU 用点阵纹理画 —— 256×256 这个数字写出来没有感觉，铺满一格才有。"""
    f.rect(x, y, 92, 100, "#fff", GN, 1.6, 6)
    f.rect(x + 8, y + 8, 76, 62, "url(#mxucell)", GN, 1.1, 2)
    f.t(x + 46, y + 84, "MXU", "lbl", GN, "middle")
    f.t(x + 46, y + 96, "256 × 256", "xxs", GN, "middle")


def _vpu(f, x, y, w, full):
    """VPU 的 8×128 形状必须画出来 —— 它是后面 lane / sublane 那一整套的物理来源。

    上一版把说明文字和格子叠在同一段 y 上，渲染出来字压在格子里。教训很朴素：
    先把网格占的高度算出来，再决定字放哪，不要靠肉眼估。
    """
    f.rect(x, y, w, 100, "#fff", TL, 1.6, 6)
    f.t(x + 10, y + 17, "VPU　向量单元", "lbl", TL)
    f.t(x + w - 10, y + 17, "8 × 128", "xxs", TL, "end")

    gx, gy, cw, ch = x + 10, y + 26, 8.4, 6.0     # 8 行 × 16 列，占 48px 高
    for r in range(8):
        for c in range(16):
            f.rect(gx + c * cw, gy + r * ch, cw - 2, ch - 1.6, FILL[TL], rx=0.8)
    f.t(x + 10, y + 88,
        "一拍处理 8 sublane × 128 lane" if full else "同左", "xxs", TL)


def _d2d(f):
    y = D2D_Y
    f.rect(D0X, y, D1X + DIE_W - D0X, 46, FILL[RD], RD, 1.8, 8)
    f.t(D0X + 12, y + 20, "die-to-die 互连", "box", RD)
    para(f, D0X + 12, y + 38, 740,
         "官方口径：比一条 ICI 链路快 <b>6×</b>。但请注意它<r>没有</r>把两个 die 缝成一个 device —— "
         "它只是让跨 die 搬运比出封装便宜，<b>地址空间仍然是两套</b>。", "xs", 14)
    f.t(D1X + DIE_W - 12, y + 30, "6 ×", "numb", RD, "end")
    # 两条竖线把 D2D 和两个 die 接起来，避免它看着像一条孤零零的横带
    for cx in (D0X + DIE_W / 2, D1X + DIE_W / 2):
        f.line(cx, DIE_T + DIE_H, cx, y, RD, 1.6)


BG_DIE = "#fcfcfd"


# ══════════════════════════════════════════════════════════════════════
def _sidebar(f):
    y = PKG_Y

    # 卡片 1：一颗封装里到底有几个什么
    f.rect(RX, y, RW, 234, "#fff", INK, 1.8, 10)
    f.t(RX + 14, y + 26, "数一遍：一颗封装里有几个什么", "sec")
    rows = [
        ("TensorCore", "2", "＝ 2 个 JAX device", BL),
        ("MXU", "4", "每 core 2 个，各 256×256", GN),
        ("SparseCore", "4", "每 device 2 个", PU),
        ("SparseCore 的向量 tile", "64", "4 × 16", PU),
        ("HBM 容量", "192 GiB", "官方表头就写 GiB；同页正文的 GB 是笔误", YL),
        ("HBM 带宽", "7.4 TB/s", "官方 7,380 GB/s", YL),
        ("ICI 对外带宽", "1,200 GB/s", "六条链路双向合计", RD),
    ]
    ry = y + 44
    for name, val, note, c in rows:
        f.rect(RX + 14, ry, 4, 20, c, rx=2)
        f.t(RX + 26, ry + 15, name, "lbl", INK)
        f.t(RX + 214, ry + 15, val, "numb", c, "end")
        f.t(RX + 226, ry + 15, note, "xxs")
        ry += 26

    # 卡片 2：同一个封装形态，两家的软件视图正好相反
    y2 = y + 248
    f.rect(RX, y2, RW, 178, FILL[RD], RD, 1.8, 10)
    f.t(RX + 14, y2 + 24, "同样是双 die 封装，软件看到的东西正好相反", "sec")
    f.rect(RX + 14, y2 + 36, RW - 28, 58, "#fff", SUB, 1.1, 6)
    f.t(RX + 24, y2 + 54, "NVIDIA B200", "lbl", INK)
    para(f, RX + 24, y2 + 72, RW - 48,
         "两个 die ＋ 一条一致性总线 → 对软件<b>装成一个 GPU</b>。"
         "缝还在（跨 die 访问更慢），但由硬件替你扛。", "xxs", 13)
    f.rect(RX + 14, y2 + 100, RW - 28, 62, "#fff", RD, 1.4, 6)
    f.t(RX + 24, y2 + 118, "TPU v7", "lbl", RD)
    para(f, RX + 24, y2 + 136, RW - 48,
         "两个 chiplet <b>如实暴露成两个 device</b>，各有各的地址空间。"
         "缝留在外面，<r>由你的切分策略去面对</r>。", "xxs", 13)

    # 卡片 3：封装之外
    y3 = y2 + 192
    f.rect(RX, y3, RW, 100, "#fff", SUB, 1.4, 10)
    f.t(RX + 14, y3 + 24, "封装之外：一台主机挂几颗", "sec")
    para(f, RX + 14, y3 + 44, RW - 28,
         "官方规格：<b>每 VM 4 颗 chip</b>（＝ 8 个 device）、224 vCPU、960 GB 内存、2 个 NUMA 域。"
         + gate.IP("主机接口 PCIe 侧约 119.2 GiB/s（内部设备表）。",
                   "主机侧接口带宽官方未公开。", why="片上／接口带宽")
         + "记住这个 4 —— GKE 机型名里的数字按 <b>device</b> 算，不是按 chip 算。", "xxs", 13)


# ══════════════════════════════════════════════════════════════════════
def _ici(f):
    f.rect(20, ICI_Y, 1360, ICI_H, "#fff", RD, 1.8, 10)
    f.t(34, ICI_Y + 26, "对外的六个出口　—— 这才是 TPU 和 GPU 差得最远的地方", "sec")
    para(f, 34, ICI_Y + 46, 640,
         "每颗 chip 有 <b>6 条 ICI 物理链路</b>，对应三维的正负方向。"
         "它们不是「加速卡之间的选配互联」，而是<r>芯片出厂时就长在硅上的第一性结构</r>："
         "拓扑是 3D 环面（torus），超过 64 颗以后由 4×4×4 的 cube 拼起来，"
         "cube 内走铜缆、cube 之间走光纤并经过光路交换机重新配线。", "xs", 15)

    # 六个方向的小示意
    ax, ay = 700, ICI_Y + 44
    dirs = [("X+", 0), ("X−", 1), ("Y+", 2), ("Y−", 3), ("Z+", 4), ("Z−", 5)]
    for name, k in dirs:
        bx = ax + (k % 3) * 96
        by = ay + (k // 3) * 40
        f.rect(bx, by, 88, 32, FILL[RD], RD, 1.2, 5)
        f.t(bx + 10, by + 21, name, "lbl", RD)
        f.t(bx + 80, by + 21, "200 GB/s", "xxs", None, "end")
    f.t(ax, ay + 96, "六条合计 1,200 GB/s（双向）", "lbl", RD)

    # 一处官方措辞本身就有歧义，不能装作没看见
    para(f, 34, ICI_Y + 116, 640,
         "<r>一个必须说清的口径坑：</r>官方正文写的是「每<b>轴</b>双向 200 GB/s」，"
         "可是 3 个轴 × 200 只有 600，对不上同一页表格里的 1,200。"
         "只有把它读成「每条<b>链路</b> 200」（6 × 200 ＝ 1,200）才自洽 —— 本文按后者画。", "xs", 14)

    f.rect(1004, ICI_Y + 40, 362, 100, FILL[RD], RD, 1.4, 8)
    para(f, 1016, ICI_Y + 60, 338,
         "对照一下：B200 的 NVLink 5 是 <b>1,800 GB/s</b>，数字更大 —— "
         "但它连的是<b>一个机柜内的 72 颗</b>，再往外要换成 InfiniBand／以太网。"
         "ICI 这 1,200 GB/s 是<r>一路铺到 9,216 颗都不换协议</r>的那种。", "xs", 15)


def _note(f):
    f.rect(20, NOTE_Y, 1360, 76, FILL[BL], BL, 1.6, 10)
    f.t(34, NOTE_Y + 24, "读这张图最容易出错的一处", "sec", BL)
    para(f, 34, NOTE_Y + 44, 1330,
         "<b>chip 和 device 的比例是 1 : 2，而所有框架日志都按 device 报数。</b>"
         "所以看到「每 device 1,153 TFLOP/s」不要以为掉了一半 —— 那正是 2,307 的一半；"
         "看到 <code>tpu7x-128</code> 也不要以为是 128 颗芯片，那是 64 颗。"
         "算 MFU 时分母要用 2,307（按 chip）或 1,153.5（按 device），"
         "<r>两个口径混用会让结论直接差两倍</r>。", "xs", 16)


if __name__ == "__main__":
    import io, sys
    io.open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/t1.svg", "w",
            encoding="utf-8").write(build())
    print("ok", H)
