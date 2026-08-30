# -*- coding: utf-8 -*-
"""图 T-6 —— 一个数走完全程：HBM → VMEM → MXU → 累加器 → 回程。

这张图想改掉一个几乎人人都有的错觉：**把 VMEM 当成 L2 缓存**。

它们在图上占同一个位置，容量也在同一个量级，所以「TPU 的缓存」这个说法
到处都是。但它们是两种完全不同的东西 —— 缓存是**硬件在运行时猜**，
暂存是**编译器在编译期写死**。一字之差，后果是：GPU 有命中率，TPU 没有；
GPU 猜错了变慢，TPU 排错了就是真的停住。

所以每一站都必须回答同一个问题：**这一步是谁决定的？** 图上那一整行
「谁决定搬」比任何容量数字都重要。
"""
from common import Fig, para, BL, RD, YL, GN, PU, TL, INK, SUB, GREY, FILL
import gate

W = 1400
TOP = 84

SY = 178                       # 车站带顶
SH = 262
SW, SGAP = 240, 40
SX0 = 20

RET_Y = SY + SH + 16           # 回程带
CARD_Y = RET_Y + 66
CARD_H = 214
BAND_Y = CARD_Y + CARD_H + 20
H = BAND_Y + 136


def sx(i):
    return SX0 + i * (SW + SGAP)


def build():
    f = Fig(W, H, "TPU v7 数据通路：HBM 经 DMA 进 VMEM，再进 MXU，"
                  "结果落进累加器；每一步都由编译器在编译期决定")
    f.title("一个数走完全程　—— 中间那一站是「暂存」，不是「缓存」", "第 6 / 8 张")
    f.legend([(SUB, "片外：HBM"), (YL, "片上暂存"), (GN, "计算"),
              (BL, "搬运（DMA，由编译器发起）"), (GREY, "灰色虚线 ＝ 官方未公开")])

    _stations(f)
    _return(f)
    _cards(f)
    _band(f)
    return f.out()


# ══════════════════════════════════════════════════════════════════════
STATIONS = [
    dict(c=SUB, name="① HBM", sub="片外主存",
         big="96 GiB", big2="3,433 GiB/s / core",
         who="编译器插一条 DMA 指令，<b>描述符由标量单元发出</b>",
         much="描述符说了算，最小 32 B",
         miss="<b>没有「命中／未命中」这回事</b> —— 只有「到了」和「还没到」"),
    dict(c=YL, name="② VMEM", sub="片上暂存（不是缓存）",
         big="64 MiB", big2="/ core",
         who="<r>编译器静态分配</r>，像分配寄存器一样，运行时不会变",
         much="一块 tile ＝ (8, 128) ＝ 4,096 B",
         miss="放不下在<b>编译期</b>就知道 —— 要么自动分块，要么编译失败"),
    dict(c=GREY, name="③ 向量寄存器", sub="MXU 的输入端",
         big="查不到", big2="v7 的深度与个数",
         who="编译器分配（VLIW 的槽位里直接写死用哪几个）",
         much="一条向量指令 ＝ 8 × 128 ＝ 1,024 个元素",
         lab3="为什么画成虚线",
         miss="<g>v2/v3 的论文给的是每 sublane 32 深；v7 这一层官方没有公开，"
              "本文不拿旧代的数字顶替。</g>"),
    dict(c=GN, name="④ MXU 阵列", sub="256 × 256，权重驻留",
         big="65,536", big2="个乘加单元 / MXU",
         who="不需要「决定」—— 权重整趟驻留，激活按拍推进（见 §4）",
         much="每拍吃一列 256 个激活，吐一列部分和",
         miss="喂不满就是空转，<b>没有别的活能顶上来</b>（见 §3）"),
    dict(c=RD, name="⑤ 累加器", sub="结果的落脚点",
         big="1 MiB", big2="/ MXU（推导见下）",
         who="硬件自动累加；K 超过 256 时<b>连续几趟都不落地</b>",
         much="128 个 × 形状 (8, 256) × 32 bit",
         lab3="1 MiB 是怎么来的",
         miss="128 × 8 × 256 × 4 B ＝ <b>1,048,576 B</b> —— 正好 1 MiB"),
]


def _stations(f):
    f.t(20, TOP + 26, "五站，每一站只问一个问题：这一步是谁决定的", "sec")
    para(f, 20, TOP + 46, 1360,
         "容量和带宽是每份 TPU 材料都会列的东西。真正决定你写代码时会撞上什么的，"
         "是下面那一整行 <b>「谁决定搬」</b> —— 从头到尾没有一站的答案是「硬件自己看着办」。", "xs", 15)

    for i, s in enumerate(STATIONS):
        x, c = sx(i), s["c"]
        dash = "5,4" if c is GREY else None
        f.rect(x, SY, SW, SH, "#fff" if c is GREY else FILL[c], c, 2, 10, dash)

        f.t(x + 14, SY + 26, s["name"], "box", INK if c in (SUB, YL) else c)
        f.t(x + SW - 14, SY + 26, s["sub"], "xxs", None, "end")

        # 大数字
        f.rect(x + 12, SY + 38, SW - 24, 46, "#fff", c, 1.2, 6, dash)
        f.t(x + 22, SY + 66, s["big"], "numb", INK if c is not GREY else GREY)
        f.t(x + SW - 22, SY + 66, s["big2"], "xxs", None, "end")

        # 三问
        yy = SY + 96
        for k, (lab, key) in enumerate((("谁决定搬", "who"),
                                        ("一次多少", "much"),
                                        (s.get("lab3", "落空了怎么办"), "miss"))):
            f.t(x + 14, yy + 12, lab, "lbl", c if c is not SUB else INK)
            yy = para(f, x + 14, yy + 28, SW - 28, s[key], "xxs", 13) + 8

    # 站与站之间的搬运箭头
    for i in range(4):
        x0, x1 = sx(i) + SW + 4, sx(i + 1) - 4
        y = SY + 60
        f.line(x0, y, x1, y, BL, 2.4, marker="aB")
        lab = ["DMA", "载入", "喂入", "落地"][i]
        if lab:
            f.t((x0 + x1) / 2, y - 8, lab, "xxs", BL, "middle")


# ══════════════════════════════════════════════════════════════════════
def _return(f):
    """回程用同一套 DMA —— 画成一条从右往左的长箭头。"""
    x0, x1 = sx(4) + SW - 20, sx(0) + 20
    f.line(x0, RET_Y + 20, x1, RET_Y + 20, BL, 2.4, marker="aB")
    f.rect(sx(1) + 10, RET_Y + 6, 700, 28, "#fff", BL, 1.4, 14)
    f.t(sx(1) + 26, RET_Y + 25,
        "回程：累加器 → VMEM → HBM，同一套 DMA、同一批描述符 —— "
        "回程同样是编译期排好的，不是算完了「顺手写回去」", "xs", BL)


# ══════════════════════════════════════════════════════════════════════
def _cards(f):
    # 左卡：暂存 vs 缓存
    f.rect(20, CARD_Y, 660, CARD_H, FILL[YL], YL, 1.8, 10)
    f.t(34, CARD_Y + 26, "「暂存」和「缓存」差在哪 —— 这是全图最要紧的一格", "sec", YL)

    rows = [
        ("放什么进去", "硬件按访问历史猜", "编译器写死"),
        ("有没有命中率", "有，而且是主要调优指标", "<r>没有这个概念</r>"),
        ("猜错 / 排错的后果", "变慢（多跑一趟内存）", "<r>停住（没有别的活可切）</r>"),
        ("你能控制到什么程度", "间接：改访问顺序去哄它", "直接：改分片和 tile 形状"),
    ]
    hx, c1, c2 = 34, 200, 210
    f.t(hx, CARD_Y + 54, "问题", "lbl")
    f.t(hx + c1, CARD_Y + 54, "GPU 的 L1 / L2（缓存）", "lbl", BL)
    f.t(hx + c1 + c2, CARD_Y + 54, "TPU 的 VMEM（暂存）", "lbl", YL)
    f.line(hx, CARD_Y + 62, 666, CARD_Y + 62, "#e0c060", 1)
    yy = CARD_Y + 68
    for q, a, b in rows:
        y1 = para(f, hx, yy + 14, c1 - 14, q, "xxs", 13)
        para(f, hx + c1, yy + 14, c2 - 14, a, "xxs", 13, BL)
        para(f, hx + c1 + c2, yy + 14, 220, b, "xxs", 13)
        yy = max(y1, yy + 30)

    # 右卡：搬运是谁发起的 + 带宽
    f.rect(700, CARD_Y, 680, CARD_H, "#fff", BL, 1.8, 10)
    f.t(714, CARD_Y + 26, "搬运不是「后台自动发生」的 —— 它占着指令流", "sec", BL)
    para(f, 714, CARD_Y + 48, 652,
         "HBM ↔ VMEM 的每一次搬运都由 DMA 完成，而 <b>DMA 的描述符是标量单元写出来的</b>。"
         "也就是说「什么时候搬、搬多少、搬到哪」跟乘加指令一样，"
         "<r>占着同一条指令流里的槽位</r>。", "xs", 16)
    para(f, 714, CARD_Y + 104, 652,
         "这解释了 §2 里那个看起来很怪的设计：为什么标量单元在一颗以矩阵乘为业的芯片上"
         "还这么重要 —— 它不算数，它<b>安排搬运</b>。", "xs", 16)
    para(f, 714, CARD_Y + 146, 652,
         gate.IP(
             "带宽落差是这条通路的真正约束：<b>VMEM 约 34,428 GiB/s，是 HBM 那 3,433 的 10 倍。</b>"
             "所以「尽量让数在 VMEM 里多待一会儿」不是风格建议，是一个整整一位数的差距。",
             "带宽落差是这条通路的真正约束：片上暂存的读写带宽比 HBM 高<b>约一个数量级</b>"
             "（<g>具体数值官方未公开，这里只给量级</g>）。"
             "所以「尽量让数在暂存里多待一会儿」不是风格建议。",
             why="片上带宽属于未公开规格"), "xs", 16)


# ══════════════════════════════════════════════════════════════════════
def _band(f):
    f.rect(20, BAND_Y, 1360, 116, FILL[GN], GN, 1.6, 10)
    f.t(34, BAND_Y + 26, "这张图想让你记住的一句话", "sec", GN)
    para(f, 34, BAND_Y + 50, 1330,
         "<b>这条通路上没有任何一步是硬件在运行时决定的。</b>搬什么、搬多少、什么时候搬、"
         "放在暂存的哪个位置 —— 全部在编译期写进指令流里。", "xs", 18)
    para(f, 34, BAND_Y + 80, 1330,
         "所以 TPU 上的性能问题几乎不长成「缓存没命中」的样子，"
         "而是长成<r>「形状不对，编译器排不出好班」</r>的样子。"
         "这也是为什么 §7 那张图必须存在：把决定权全交给编译期，就得看看编译期到底能排出什么。", "xs", 18)


if __name__ == "__main__":
    import io, sys
    io.open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/t6.svg", "w",
            encoding="utf-8").write(build())
    print("ok", H)
