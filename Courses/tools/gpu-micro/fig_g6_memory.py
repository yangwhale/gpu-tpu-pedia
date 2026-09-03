# -*- coding: utf-8 -*-
"""图 G-6 —— 一个数走完全程：从 HBM 到乘加单元，中间几站、每站谁在搬。

刻意不做「带宽比大小」：TPU 侧的片上带宽官方没公开，硬比会变成一半实数一半灰框。
⚠️ 2026-09-03 补充口径：**站台注里填 GPU 侧已知的带宽是可以的**（L2 21 TB/s 一直就在里面，
   这次补的是共享内存的 128 B/周期/SM）。上面那条禁的是「把整张图的论点做成带宽对比」，
   不是禁止填任何数。TPU 那一侧照旧写「未公开」，图例第四条就是为这个留的。
换成「站台链 + 搬运者」之后，骨架只依赖公开信息，而且讲的是更本质的那件事 ——
GPU 的中间层是**缓存**（运行时才知道命中没有），TPU 的中间层是**暂存**（编译期就排好了）。
"""
from common import Fig, para, _wlen, BL, RD, YL, GN, PU, TL, INK, SUB, GREY, FILL

W = 1400
TOP = 84

# ⚠️ BH 从 134 加到 148 是为了给共享内存那一站多塞一行（128 B/周期/SM）。
#    SUBY 必须跟着加：支线折线的拐点在 y+BH+22，支线盒顶在 y+SUBY-6 ——
#    178 配 148 的话竖直段只剩 2 px，看着像画错了。改 BH 就回来改这一行。
BW, BH, GAPX, X0 = 176, 148, 110, 40
XS = [X0 + i * (BW + GAPX) for i in range(5)]      # 40 326 612 898 1184
SUBY = 192
MAINY = 26                                          # band 与主链之间的留白（给角标腾地方）                                          # 支线相对主链的下沉量
KEYW = XS[3] - X0 - 28                              # 要点条宽度（支线左边的空白）

BAND_H = 26
G_T = TOP + 34
G_B = G_T + MAINY + SUBY + BH
T_T = G_B + 62
T_B = T_T + MAINY + SUBY + BH
C_T = T_B + 42
H = C_T + 218

HW = BL      # 硬件自动管（有 tag、会 miss）
SW = PU      # 软件/编译器显式管（无 tag、不会 miss，也没有兜底）
OFF = YL     # 片外


def build():
    f = Fig(W, H, "GPU 与 TPU 的片上内存层级对照：一个数从 HBM 到乘加单元经过哪些站，"
                  "每一站由硬件自动搬运还是由编译器显式搬运")
    f.title("一个数走完全程　—— 从 HBM 到乘加单元，中间几站、<tspan fill=\"#8430ce\">每站谁在搬</tspan>")
    f.legend([(OFF, "片外内存"), (HW, "硬件自动管：有 tag、会 miss"),
              (SW, "软件/编译器显式管：不会 miss，也没有兜底"),
              (GREY, "官方只给容量、没给带宽")])

    _gpu(f)
    _tpu(f)
    _diff(f)
    return f.out()


# ══════════════════════════════════════════════════════════════════════
def _stop(f, x, y, c, name, cap, sub, note=None, dim=False):
    """一个站台盒。

    c 传 ((左色, 左名), (右色, 右名)) 时画成左右分色 —— 专给「同一块硅、
    两种身份」的 L1＋共享内存那一站用：左半硬件管、右半软件管。
    这一格如果只涂一种颜色，全图的论点就自相矛盾了（见 G-2 对同一块硅的判定）。
    dim=True → 灰虚框，表示这一格的数官方没公开。
    """
    if isinstance(c, tuple):
        (cl, nl), (cr, nr) = c
        f.rect(x, y, BW, BH, "#fff", SUB, 1.8, 9)
        for i, (cc_, nn) in enumerate(((cl, nl), (cr, nr))):
            f.rect(x + i * BW / 2, y, BW / 2, 24, FILL[cc_], rx=9)
            f.rect(x + i * BW / 2, y + 15, BW / 2, 9, FILL[cc_], rx=0)
            f.t(x + i * BW / 2 + 9, y + 17, nn, "lbl", cc_)
        f.line(x + BW / 2, y, x + BW / 2, y + BH, SUB, 1.2, dash="3,2")
        yy = para(f, x + 12, y + 42, BW - 24, cap, "lbl", 16)
        yy = para(f, x + 12, yy + 3, BW - 24, sub, "xs", 14)
        if note:
            para(f, x + 12, yy + 4, BW - 24, note, "xxs", 12)
        return
    cc = GREY if dim else c
    f.rect(x, y, BW, BH, "#fff", cc, 1.8, 9, "4,3" if dim else None)
    f.rect(x, y, BW, 24, FILL[cc], rx=9)
    f.rect(x, y + 15, BW, 9, FILL[cc], rx=0)
    f.t(x + 12, y + 17, name, "box", cc)
    yy = para(f, x + 12, y + 42, BW - 24, cap, "lbl", 16)
    yy = para(f, x + 12, yy + 3, BW - 24, sub, "xs", 14)
    if note:
        para(f, x + 12, yy + 4, BW - 24, note, "xxs", 12)


def _arrow(f, xi, y, c, top, bottom=None):
    """站与站之间：箭头 ＋「谁搬的」。文字居中在间隙里。"""
    x = XS[xi] + BW
    f.line(x + 6, y + BH / 2, x + GAPX - 8, y + BH / 2, c, 2,
           {BL: "aB", PU: "aP", GN: "aG", RD: "aR"}.get(c, "aK"))
    para(f, x + 4, y + BH / 2 - 12, GAPX - 8, top, "xxs", 11)
    if bottom:
        para(f, x + 4, y + BH / 2 + 20, GAPX - 8, bottom, "xxs", 11)


def _band(f, y, c, kicker, title):
    f.rect(20, y, 1360, BAND_H, FILL[c], rx=6)
    f.t(34, y + 18, kicker, "box", c)
    f.t(34 + 13 * _wlen(kicker) + 16, y + 18, title, "sm")


def _branch(f, from_xi, y, c, note):
    """主链某一站 → 支线首站的折线。

    说明文字右对齐贴在竖直段左侧 —— 那一带（要点条上方）横向全是空的，
    比塞进折线与支线盒之间的百来像素里可读得多。
    """
    sx = XS[from_xi] + BW - 26
    my = y + BH + 22
    f.path(f"M{sx} {y + BH} L{sx} {my} L{XS[3] + 26} {my} "
           f"L{XS[3] + 26} {y + SUBY - 6}", c, 2,
           marker={PU: "aP", GN: "aG"}.get(c, "aK"))
    para(f, sx - 12, my - 4, 720, note, "xs", 14, anchor="end")


def _keys(f, y, c, ttl, items):
    """支线左边那片空白 —— 放三条要点，正好填满，不跟任何东西打架。"""
    f.rect(X0, y, KEYW, BH, "#fff", c, 1.5, 9)
    f.rect(X0, y, KEYW, 22, FILL[c], rx=9)
    f.rect(X0, y + 13, KEYW, 9, FILL[c], rx=0)
    f.t(X0 + 12, y + 16, ttl, "box", c)
    cw = (KEYW - 24 - 2 * 14) / 3
    for i, s in enumerate(items):
        x = X0 + 12 + i * (cw + 14)
        f.rect(x, y + 32, 3, 10, c, rx=1.5)
        para(f, x + 10, y + 40, cw - 12, s, "xs", 14)


# ══════════════════════════════════════════════════════════════════════
def _gpu(f):
    _band(f, G_T - 30, BL, "NVIDIA B200",
          "五站。中间的 L2 和 L1 是缓存 —— 命中不命中，要到运行时才知道。")
    y = G_T + MAINY

    _stop(f, XS[0], y, OFF, "HBM3e", "192 GB", "8.0 TB/s", "片外 · 官方数字")
    _stop(f, XS[1], y, HW, "L2 缓存", "126 MB", "4 个分区，每 die 2 个",
          "<g>本分区实测 21 TB/s、跨 die 16.8 —— 第三方</g>")
    # 全图最重要的一处对照：多出来的那一站就在这儿
    f.rect(XS[1] + BW - 104, y - 24, 104, 19, "#fff", RD, 1.4, 9)
    f.t(XS[1] + BW - 52, y - 11, "TPU 没有这一站", "xxs", RD, "middle")
    f.line(XS[1] + BW - 52, y - 5, XS[1] + BW - 52, y, RD, 1.4)
    _stop(f, XS[2], y, ((HW, "L1"), (SW, "共享内存")), None, "256 KiB / SM",
          "<b>同一块硅，两种身份</b>：左半硬件管、右半软件管",
          "共享部分最多 227 KiB / 线程块 · "
          "<g>128 B / 周期 / SM（三处公开测量一致）· L1 命中约 39 周期</g>")
    _stop(f, XS[3], y, SW, "寄存器堆", "256 KiB / SM", "64K 个 32-bit",
          "每线程最多 255 个 · 编译期分配")
    _stop(f, XS[4], y, SW, "CUDA Core", "128 / SM", "FP32 ／ INT32", "向量通路")

    _arrow(f, 0, y, BL, "硬件自动", "miss 才往下走")
    _arrow(f, 1, y, BL, "硬件自动", "程序管不着")
    _arrow(f, 2, y, PU, "<b>要写指令</b>", "ld.shared")
    _arrow(f, 3, y, PU, "直接读", "零延迟")

    # ── Tensor Core 支线 ────────────────────────────────────────────
    sy = y + SUBY
    _branch(f, 2, y, PU,
            "<b>TMA ／ tcgen05.cp 异步搬运</b>：整块搬进 TMEM，"
            "<r>绕开寄存器堆</r>，搬运期间这个 SM 可以接着干别的活")
    _stop(f, XS[3], sy, SW, "TMEM", "256 KiB / SM", "128 lane × 512 列",
          "<b>Blackwell 新加的一层</b> · 只有 Tensor Core 用得到")
    _stop(f, XS[4], sy, SW, "Tensor Core", "4 / SM", "1,024 乘加 / 周期",
          "<g>← 推导值，推导链见图 G-2</g>")
    _arrow(f, 3, sy, PU, "直接喂", "不过寄存器")

    _keys(f, sy, BL, "这条链上最该记住的三件事", [
        "<b>中间是「一站半」缓存。</b>L2 和 L1 要存 tag、要做替换，一部分硅面积"
        "和功耗花在「猜你接下来要什么」上；<r>共享内存不用</r> —— 它是暂存，"
        "和 TPU 的 VMEM 同一类东西，只是小两个数量级。GPU 两种都留着。",
        "<b>猜错了怎么办？</b>换一个 warp 上来接着算。GPU 的延迟不是被消除的，"
        "是被<r>别人的工作盖住</r>的 —— 这就是它要塞 64 个 warp 的原因。",
        "<b>Tensor Core 那条支线是新东西。</b>Blackwell 之前，矩阵操作数"
        "必须先落进寄存器堆；TMEM 让它整条绕过去了。",
    ])


def _tpu(f):
    _band(f, T_T - 30, GN, "TPU v7",
          "四站，而且中间那站不是缓存 —— 什么时候搬什么，编译期就钉死了。")
    y = T_T + MAINY

    _stop(f, XS[0], y, OFF, "HBM3e", "192 GiB", "7.4 TB/s / chip",
          "片外 · 官方数字 · 96 GiB / device")
    _stop(f, XS[1], y, SW, "VMEM", "64 MiB / core", "片上暂存，MXU 只从这里取数",
          "<b>没有 tag、不会 miss</b> · <g>容量见 JAX 源码；带宽未公开</g>")
    _stop(f, XS[2], y, SW, "向量寄存器", "官方未公开", "形状是 8 × 128 的二维块",
          "<g>数量未公开；8×128 见于 Pallas 文档</g>", dim=True)
    _stop(f, XS[3], y, SW, "VPU", "向量单元", "逐元素算子都在这儿",
          "激活、归约、缩放")
    # 右上这一格不是存储站，是回程说明 —— 用户要的「数据流转」得画完整圈
    _stop(f, XS[4], y, SUB, "结果的回程", "→ VMEM → HBM",
          "还是 DMA，还是编译期排好",
          "<b>对照：</b>GPU 的回程要穿过 L2")

    _arrow(f, 0, y, GN, "<b>DMA 引擎</b>", "编译期排好班")
    _arrow(f, 1, y, GN, "向量 load")
    _arrow(f, 2, y, GN, "直接读")
    _arrow(f, 3, y, SUB, "算完往回", "")

    # ── MXU 支线 ────────────────────────────────────────────────────
    sy = y + SUBY
    _branch(f, 1, y, GN,
            "<b>权重与数据直接推进阵列</b>：<r>从来就不经过向量寄存器</r>"
            " —— 这一点 GPU 到 Blackwell 才追上")
    _stop(f, XS[3], sy, SW, "MXU", "256 × 256", "65,536 个乘加单元，一条指令喂满",
          "<g>每 chip 几个：官方文档自相矛盾，见 G-8</g>")
    _stop(f, XS[4], sy, SW, "累加器", "1 MiB / MXU", "128 × (8×256) × 4 B",
          "结果攒在这儿，不回寄存器")
    _arrow(f, 3, sy, GN, "算完直接落")

    _keys(f, sy, GN, "同一位置，TPU 的三件事", [
        "<b>少一整层。</b>HBM 直接进 VMEM，中间没有 L2。省下来的不只是面积，"
        "还有「不知道会不会命中」这件事本身。",
        "<b>没有 warp 可以换。</b>猜错了没人替你顶班 —— 所以 TPU 不能猜，"
        "必须由<r>编译器在编译期算准</r>每一拍的数在哪。",
        "<b>片上容量和带宽官方没公开</b>，所以这里不填数字。能确定的是"
        "层级结构和搬运方式 —— 那才是这张图要讲的。",
    ])


# ══════════════════════════════════════════════════════════════════════
def _diff(f):
    f.t(20, C_T, "把两条链叠起来看　—— 差别不在快慢，在「谁负责知道数在哪」", "sec")
    cards = [
        (BL, "缓存 vs 暂存", "同样是片上 SRAM，性格完全不同",
         "GPU 的 L1/L2 是<b>缓存</b>：你只管发访存指令，命中不命中它自己处理 —— "
         "代价是要存 tag、要做替换、要维护一致性，而且<r>时间不可预测</r>。"
         "TPU 的 VMEM 是<b>暂存</b>：编译器显式发 DMA 把数据搬进来，没有 tag、"
         "没有 miss、时间可预测 —— 代价是编译器算错了就是真的慢，没有兜底。"),
        (RD, "谁发起这次搬运", "同样是「把数弄过来」，发令的人不一样",
         "GPU 侧：计算单元自己发一条 <b>load</b>，地址是它算的，什么时候到不知道 —— "
         "搬运是<b>取数指令的副作用</b>。TPU 侧：搬运是一条独立的 <b>DMA</b>，"
         "描述符里写清「从哪到哪、多大、什么步长」，由专门的引擎执行，"
         "计算单元完全不参与。<r>一边是「我要，你给我找」，一边是「你先搬好，我到点来取」。</r>"
         "谁来盖住这中间的几百个周期 —— 那是下一张图的事。"),
        (PU, "GPU 正在往这边挪一步", "TMEM 是个信号",
         "Blackwell 新加的 TMEM，是一块<b>只给 Tensor Core 用、由指令显式搬进搬出、"
         "不参与缓存机制</b>的片上 SRAM —— 这个描述几乎就是 TPU 的 VMEM。方向很清楚："
         "在矩阵这条路上，「让硬件猜」的收益越来越小，不如把控制权交回给编译器和 kernel 作者。"),
    ]
    cw = (1360 - 2 * 16) / 3
    for i, (c, ttl, sub, body) in enumerate(cards):
        x = 20 + i * (cw + 16)
        f.rect(x, C_T + 14, cw, 164, "#fff", c, 1.8, 10)
        f.rect(x, C_T + 14, cw, 46, FILL[c], rx=10)
        f.rect(x, C_T + 51, cw, 9, FILL[c], rx=0)
        f.t(x + 14, C_T + 34, ttl, "box", c)
        f.t(x + 14, C_T + 51, sub, "xs")
        para(f, x + 14, C_T + 78, cw - 28, body, "xs", 16)


if __name__ == "__main__":
    import io, sys
    io.open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/g6.svg", "w",
            encoding="utf-8").write(build())
    print("ok", H)
