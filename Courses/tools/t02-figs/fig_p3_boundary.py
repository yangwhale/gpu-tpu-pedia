# -*- coding: utf-8 -*-
"""图 P-3 —— 编译器的边界：编排是它的活，算法是人的活。

上一张图讲完「编译器替你插通信」，很容易留下一个过头的印象：
交给编译器就行了。这张图专门把那个印象按回去。

**Flash Attention 是这一节最诚实的例子**，因为它在两边都得手写 ——
GPU 上是 CUTLASS / Triton，TPU 上是 Pallas。没有任何一侧的编译器
自己长出了它。一个两边都失效的反例，比任何一边的成功案例都更能定义边界。

画法：一条竖直的虚线把画面切成「编译器领地」和「人的领地」，
四样编排能力放左边，两样算法能力放右边，Flash 明确落在右边。
**边界要画成一条线，不是画成一张对比表** —— 表让人比强弱，线让人记位置。
"""
from common import Fig, para, BL, RD, GN, YL, PU, INK, SUB, GREY, FILL

W = 1400
TOP = 84

ZONE_Y = TOP + 42
ZONE_H = 296
LZ_X, LZ_W = 20, 796
BND_X = 836                       # 边界虚线
RZ_X, RZ_W = 856, 524

GOOD = [
    ("算子融合",
     "把相邻的小算子并成一个 kernel，中间结果不落回 HBM。"),
    ("通信与计算重叠",
     "看得到整张图，就知道哪段通信能藏在哪段计算底下。"),
    ("内存复用",
     "谁和谁生命期不重叠，就让它们共用同一块地址。"),
    ("布局与切分选择",
     "同一个矩阵摆成什么形状、按哪根轴切，编译期试着挑。"),
]

BAD = [
    ("改变数学上的计算顺序",
     "Flash 把 softmax 拆成可增量合并的形式，于是注意力矩阵<b>根本不必整个存在</b>。"
     "这不是把已有算子排得更好，<r>这是换了一个算法</r>。"),
    ("发明一个不存在的算子",
     "编译器只能编排它<b>认识</b>的东西。一个从没有人写过的 kernel，"
     "它没有理由凭空想出来。"),
]

FL_Y = ZONE_Y + ZONE_H + 52
FL_H = 196
TP_Y = FL_Y + FL_H + 30
TP_H = 138
BAND_Y = TP_Y + TP_H + 26
H = BAND_Y + 112


def build():
    f = Fig(W, H, "编译器能力边界示意：左侧是编排类优化，编译器可以自动完成；"
                  "右侧是算法层面的改写，必须由人完成，Flash Attention 属于右侧")
    f.title("编译器能替你做什么，<tspan font-weight=\"700\" fill=\"#d93025\">做不到什么</tspan>"
            "　—— Flash Attention 是最诚实的那个例子")
    f.legend([(BL, "编排：编译器能自动做"), (PU, "算法：只能人来做"),
              (YL, "两边都得手写的那一格")])

    _zones(f)
    _flash(f)
    _tools(f)
    _band(f)
    return f.out()


# ══════════════════════════════════════════════════════════════════════
def _zones(f):
    f.rect(LZ_X, ZONE_Y, LZ_W, ZONE_H, FILL[BL], BL, 1.4, 10)
    f.t(LZ_X + 18, ZONE_Y + 27, "编译器领地　·　把<tspan font-weight=\"700\">已有的算子</tspan>编排好", "sec")

    cw, chh = (LZ_W - 54) / 2, 84
    for i, (t, d) in enumerate(GOOD):
        x = LZ_X + 18 + (i % 2) * (cw + 18)
        y = ZONE_Y + 44 + (i // 2) * (chh + 14)
        f.rect(x, y, cw, chh, "#fff", BL, 1.1, 7)
        f.t(x + 14, y + 24, t, "box", INK)
        para(f, x + 14, y + 44, cw - 28, d, "xs", 17)

    f.rect(LZ_X + 18, ZONE_Y + 44 + 2 * (chh + 14), LZ_W - 36, 40, "#f1f3f4", None, 0, 6)
    para(f, LZ_X + 32, ZONE_Y + 68 + 2 * (chh + 14), LZ_W - 64,
         "这四样的共同点：<b>它们都不改变「算的是什么」，只改变「怎么排」</b>。"
         "所以编译器有资格自己决定 —— 排错了顶多慢，不会错。", "xs", 17)

    # 边界线：整张图的主角，画成红虚线并加标签
    f.line(BND_X, ZONE_Y - 6, BND_X, ZONE_Y + ZONE_H + 6, RD, 2.2, dash="7 5")
    f.rect(BND_X - 30, ZONE_Y + ZONE_H / 2 - 32, 60, 64, "#fff", RD, 1.4, 8)
    f.t(BND_X, ZONE_Y + ZONE_H / 2 - 8, "边", "box", RD, "middle")
    f.t(BND_X, ZONE_Y + ZONE_H / 2 + 12, "界", "box", RD, "middle")

    f.rect(RZ_X, ZONE_Y, RZ_W, ZONE_H, FILL[PU], PU, 1.4, 10)
    f.t(RZ_X + 18, ZONE_Y + 27, "人的领地　·　<tspan font-weight=\"700\">换一个算法</tspan>", "sec")
    y = ZONE_Y + 44
    for t, d in BAD:
        f.rect(RZ_X + 18, y, RZ_W - 36, 92, "#fff", PU, 1.1, 7)
        f.t(RZ_X + 32, y + 24, t, "box", INK)
        para(f, RZ_X + 32, y + 44, RZ_W - 64, d, "xs", 17)
        y += 92 + 14
    para(f, RZ_X + 18, y + 18, RZ_W - 36,
         "<b>越过这条线，编译器就帮不上忙了</b> —— 两边都一样。", "xs", 17)


# ══════════════════════════════════════════════════════════════════════
def _flash(f):
    f.rect(20, FL_Y, 1360, FL_H, FILL[YL], YL, 1.6, 10)
    f.t(38, FL_Y + 28, "Flash Attention　—— 一个两边都失效的反例，比任何一边的成功案例都更能定义边界", "sec")

    bw = 420
    for i, (x, tag, tool, body) in enumerate([
            (38, "GPU 侧", "CUTLASS / Triton",
             "手写。<b>NVIDIA 的编译器没有自己想出 Flash</b>，"
             "是研究者写出来、再被库收编的。"),
            (38 + bw + 20, "TPU 侧", "Pallas",
             "同样手写。<b>XLA 也没有自己想出 Flash</b>，"
             "Splash 那一支同样是人一行行写的。")]):
        f.rect(x, FL_Y + 46, bw, 104, "#fff", YL, 1.2, 8)
        f.t(x + 16, FL_Y + 72, tag, "box", INK)
        f.t(x + 90, FL_Y + 72, tool, "num", BL)
        para(f, x + 16, FL_Y + 94, bw - 32, body, "xs", 17)

    rx = 38 + 2 * (bw + 20)
    f.rect(rx, FL_Y + 46, 1380 - rx - 18, 104, "#fff", RD, 1.4, 8)
    para(f, rx + 16, FL_Y + 72, 1380 - rx - 50,
         "<r>没有任何一侧的编译器自己长出了它。</r>", "box")
    para(f, rx + 16, FL_Y + 96, 1380 - rx - 50,
         "Flash 改的是<b>数学上的计算顺序</b> —— 分块算、增量合并 softmax、"
         "注意力矩阵不落 HBM。那是算法，不是编排。<b>算法永远是人的活。</b>", "xs", 17)

    para(f, 38, FL_Y + 172, 1324,
         "<b>讲到这里要防一个误解</b>：这不是说编译器没用。上面左边那四样，"
         "在 GPU 侧同样是 Triton / CUTLASS 这些工具在替人做 —— "
         "<b>区别只在于自动化的程度，不在于有没有边界。</b>边界两边都有，位置也差不多。", "xs", 17)


# ══════════════════════════════════════════════════════════════════════
def _tools(f):
    f.rect(20, TP_Y, 1360, TP_H, "#fff", "#dadce0", 1.2, 10)
    f.t(38, TP_Y + 27, "那越过边界之后，两边拿什么写　—— Triton 与 Pallas 定位几乎一样，差别在底下", "sec")

    for x, w, c, t, body in [
            (38, 650, RD, "Triton　·　GPU",
             "用 Python 写 kernel，编译到 <code>PTX</code>。"
             "你操作的心智单位是<b>一个 program（一组线程）处理一块</b>，"
             "共享内存和同步要自己安排。"),
            (712, 650, GN, "Pallas　·　TPU",
             "同样用 Python 写 kernel，编译到 <code>Mosaic</code>。"
             "你操作的心智单位是<b>一块 tile 在 VMEM 里进出</b>，"
             "搬运由 <code>BlockSpec</code> 描述，不是自己写 load/store。")]:
        f.rect(x, TP_Y + 42, w, 80, FILL[c], c, 1.2, 8)
        f.t(x + 16, TP_Y + 66, t, "box", INK)
        para(f, x + 16, TP_Y + 86, w - 32, body, "xs", 17)


# ══════════════════════════════════════════════════════════════════════
def _band(f):
    f.rect(20, BAND_Y, 1360, 96, FILL[BL], BL, 1.4, 10)
    f.t(38, BAND_Y + 28, "给「把控制权交给编译器」钉一个边界", "sec")
    y = BAND_Y + 54
    for seg in [
            "<b>编排是编译器的活，算法永远是人的活。</b>"
            "这条线两边都存在，位置也差不多 —— 所以「TPU 要手写 kernel」不是 TPU 的缺点，"
            "<r>GPU 上同样要写，只是写的人多、库更厚</r>。",
            "真正的差别在<b>越过边界之后有多少现成的东西可抄</b>。这一条 GPU 侧目前确实占优，"
            "但它是<b>生态差距，不是架构差距</b> —— 两者会随时间收敛，别把它们混成一件事讲。"]:
        y = para(f, 38, y, 1324, seg, "xs", 19) + 4


if __name__ == "__main__":
    import sys
    open(sys.argv[1] if len(sys.argv) > 1 else "out/fig_p3_boundary.svg",
         "w", encoding="utf-8").write(build())
