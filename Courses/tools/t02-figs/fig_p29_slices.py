# -*- coding: utf-8 -*-
"""图 P-29 —— 版图层面：寄存器堆不是一整块，它被切成 128 条，摊在 lane 上。

**这张图回答的是一个「不肯放过」的追问。** P-28 说「每条端口本身就是 128 宽」，
听完一定会有人在心里算一笔账：<b>一个 vreg 是 4 KB，它怎么可能扯出去那么多根线？</b>
这个质疑<b>完全正确</b>，而且它是理解整个数据通路的最后一把钥匙 ——
因为答案不是「线真的那么多」，而是<b>「它根本不需要汇合到一起」</b>。

**必须先纠一个单位。** 一根物理走线传的是<b>一个 bit</b>，不是一个数。
所以一个 vreg 是 8 × 128 × 32 ＝ <b>32,768 根线</b>，不是 128 根。
如果寄存器堆真是一整块、要一次整个读出来，那这 32,768 根线得从同一个地方扯出来 ——
<b>那确实荒唐，所以它不是一整块。</b>

**真正的样子：整条数据通路按 lane 切成 128 份，存储就贴在计算旁边。**
于是每一份只要 8 × 32 ＝ <b>256 根线，走几微米</b>。128 份加起来才是 32,768，
<b>但它们从来不汇合</b>。质疑者猜的「顶多一百多根」——数量级完全正确，
只是那是<b>每条 lane 的局部数</b>，被当成了全局数。

**⚠️ 出处口径（这张图最需要小心的地方）。** 「按 lane 切片、存储与 ALU 就近放置」
<b>是推出来的，不是查到的版图</b>。图上写了两条推导链。
具体的 bank 划分、位线排布、真实 floorplan —— <b>公开资料没有，我也没查到</b>。
"""
from common import Fig, para, BL, GN, RD, YL, PU, TL, INK, SUB, GREY, FILL

W = 1400

CNT_Y, CNT_H = 84, 92           # ① 先把线数算出来
SLC_Y, SLC_H = CNT_Y + CNT_H + 22, 300   # ② 主图：128 条切片
Q_Y, Q_H = SLC_Y + SLC_H + 22, 214       # ③ 三个问题的答案
BND_Y, BND_H = Q_Y + Q_H + 22, 116       # ④ 边界：哪些是推的
H = BND_Y + BND_H + 20

# 切片阵列的几何
SX0, SLW, PITCH = 40, 58, 64
NSL = 9                          # 画 9 条真的，然后省略号，然后最后一条
STO_H, WIRE_H, ALU_H = 120, 30, 42
ANN_X = 786                      # 右侧注解起点


def build():
    f = Fig(W, H, "TPU 的向量寄存器堆在版图上被切成 128 条，每条紧贴自己那条 lane 的 "
                  "ALU；跨 lane 才需要真正的长线，所以才有独立的 XLU")
    f.title("版图上它是<tspan font-weight=\"700\">碎的</tspan>"
            "　—— 一个 vreg 不用扯 32,768 根线出去，因为那些线<tspan "
            "font-weight=\"700\">从来不汇合</tspan>")
    f.legend([(PU, "vreg 存储（切片）"), (GN, "本 lane 的 ALU"),
              (GREY, "256 根短线，几微米"), (RD, "真正要扯出去的只有这两条")])
    _count(f)
    _slices(f)
    _qa(f)
    _bound(f)
    return f.out()


# ══════════════════════════════════════════════════════════════════════
# 一切误解的根都在这个单位上：一根线传一个 bit，不是一个数。
def _count(f):
    f.rect(20, CNT_Y, 1360, CNT_H, FILL[YL], YL, 1.6, 10)
    f.t(38, CNT_Y + 26, "🔢 先纠一个单位：<tspan font-weight=\"700\">一根走线传的是"
                        "一个 bit，不是一个数</tspan>", "sec")
    y = para(f, 38, CNT_Y + 50, 1324,
             "一个 vreg ＝ 8 sublane × 128 lane × 32 bit ＝ <b>32,768 bit</b>　→　"
             "<b>要一次整个读出来，就是 32,768 根线</b>，不是 128 根。", "xs", 18)
    para(f, 38, y + 2, 1324,
         "<r>「那也太荒唐了」—— 对，所以它不是一整块。</r>"
         "下面这张图就是它真实的样子。", "xs", 18)


# ══════════════════════════════════════════════════════════════════════
# 主图：128 条竖切片。画法本身就是论点 —— 它们并排、不相连。
def _slices(f):
    f.rect(20, SLC_Y, 1360, SLC_H, "#fff", LINEC := PU, 1.6, 10)

    sy = SLC_Y + 46
    ay = sy + STO_H + WIRE_H

    def one(x, label, dim=False):
        c = GREY if dim else PU
        gc = GREY if dim else GN
        # 存储：64 行里画 7 行示意
        f.rect(x, sy, SLW, STO_H, FILL[c], c, 1.2, 4)
        for i in range(7):
            f.rect(x + 7, sy + 10 + i * 14, SLW - 14, 9, "#fff", c, 0.6, 1)
        f.t(x + SLW / 2, sy + STO_H - 6, "…64 行", "xxs", SUB, anchor="middle")
        # 那 256 根线
        f.line(x + SLW / 2, sy + STO_H, x + SLW / 2, ay, GREY, 5.0)
        # ALU
        f.rect(x, ay, SLW, ALU_H, FILL[gc], gc, 1.2, 4)
        f.t(x + SLW / 2, ay + 18, "ALU", "xxs", gc, anchor="middle")
        f.t(x + SLW / 2, ay + 33, "×32", "xxs", SUB, anchor="middle")
        f.t(x + SLW / 2, ay + ALU_H + 16, label, "xxs", SUB, anchor="middle")

    for i in range(NSL):
        one(SX0 + i * PITCH, "lane %d" % i)
    ex = SX0 + NSL * PITCH
    f.t(ex + 22, ay - 20, "…", "ttl", SUB, anchor="middle")
    one(ex + 46, "lane 127")
    # ⚠️ 这个 32 必须写出来。P-27 给的是「每个 (lane, sublane) 位置 4 个 ALU」，
    #    一条 lane 有 8 个 sublane，所以一条 lane 底下是 8 × 4 ＝ 32 个 ——
    #    早先这里写 ×4，跟全课那个 4,096 差了整整 8 倍。
    f.t((SX0 + ex + 46 + SLW) / 2, ay + ALU_H + 36,
        "每条 lane 底下 8 × 4 ＝ 32 个 ALU　→　128 × 32 ＝ 4,096，"
        "这就是公开资料那个数", "xxs", SUB, anchor="middle")

    # 顶部一条括号，说明这一整排就是「一个 vreg」
    bx0, bx1 = SX0, ex + 46 + SLW
    f.line(bx0, SLC_Y + 30, bx1, SLC_Y + 30, PU, 1.2, dash="4 3")
    f.t((bx0 + bx1) / 2, SLC_Y + 24, "同一个 vreg 的 128 个切片　—— 它们并排，但彼此不相连",
        "xxs", PU, anchor="middle")

    # 右侧注解
    y = ANN_X
    # 在第一条切片的那段粗线上标出它到底是多少根 —— 全图的核心数字
    f.t(SX0 + SLW / 2 + 10, sy + STO_H + WIRE_H / 2 + 4, "256 根", "xxs", INK)

    f.rect(y, sy - 16, 1360 - y + 20 - 20, 176, FILL[GREY], GREY, 1.2, 8)
    ty = para(f, y + 16, sy + 6, 1360 - y - 16,
              "<b>每一条切片只要 8 × 32 ＝ 256 根线</b>，"
              "而且存储就<b>贴在</b>自己那条 lane 的 ALU 上 —— 走几微米。", "xs", 18)
    ty = para(f, y + 16, ty + 8, 1360 - y - 16,
              "128 份加起来确实是 32,768 根，<r>但它们从来不汇合</r>。"
              "所以「128 宽」是 <b>128 份并行的硬件</b>，"
              "不是一条粗 128 倍的线。", "xs", 18)
    para(f, y + 16, ty + 8, 1360 - y - 16,
         "<b>「顶多一百多根」这个直觉是对的</b> —— 那是每条 lane 的局部数，被当成了全局数。",
         "xs", 18)


# ══════════════════════════════════════════════════════════════════════
QA = [
    (PU, "① vreg ↔ 其他 63 个 vreg", "完全不连",
     "<b>它们之间没有任何直接通路。</b>要把 vreg 3 挪到 vreg 7，"
     "唯一的办法是<b>读出来、过一遍 ALU、再写回去</b>。",
     "它们共用<b>位线</b>（<g>就是把数据送出去的那组竖线，下一张图拆开画</g>）："
     "一条切片里 64 行共用一组线，靠<b>地址</b>挑一行。"
     "<r>但这不是轮转排队</r> —— 你要哪一行就是哪一行，<b>当周期就到</b>，"
     "不存在「等 64 个周期才轮到我」。共用的代价是"
     "<b>一条端口一个周期只能送一行</b>：想同时送 8 行，就得有 8 条端口 —— "
     "<b>真正的预算是端口数，不是 64。</b>"),
    (GN, "② vreg ↔ VPU", "贴着，几微米",
     "根本不用「扯出去」。<b>vreg 的 lane 5 和 ALU 的 lane 5 在芯片上是挨着的</b>，"
     "中间就是那 256 根短线。",
     "<b>这就是逐元素算子便宜的物理原因</b>：它走的是全芯片最短的一条路，"
     "而且 128 条同时在走。"),
    (RD, "③ vreg ↔ MXU / XLU", "真的要扯出去",
     "MXU 是个<b>独立的块</b>：一次运算吃 <b>8×128 乘 128×128</b>。"
     "左边正好一个 vreg，可<b>权重那侧有 128 行，一个 vreg 只给 8 行</b> —— "
     "<b>差 16 倍</b>，喂不满；"
     "XLU 更是<b>必须同时够到全部 128 条 lane</b>。",
     "<r>所以只有这两条是真正的长路。</r>它们因此各自配了进料口、结果队列，"
     "而且<b>全核只有一两个</b> —— 长而宽的路，芯片不会多铺。"),
]


def _qa(f):
    for i, (c, ttl, tag, a, b) in enumerate(QA):
        x, w = 20 + i * 456, 438
        f.rect(x, Q_Y, w, Q_H, FILL[c], c, 1.4, 9)
        f.t(x + 14, Q_Y + 24, ttl, "box", c)
        f.t(x + w - 14, Q_Y + 24, tag, "xxs", SUB, anchor="end")
        y = para(f, x + 14, Q_Y + 48, w - 28, a, "xs", 18)
        f.line(x + 14, y + 6, x + w - 14, y + 6, c, 0.8, dash="3 3")
        para(f, x + 14, y + 26, w - 28, b, "xs", 18)


# ══════════════════════════════════════════════════════════════════════
def _bound(f):
    f.rect(20, BND_Y, 1360, BND_H, FILL[RD], RD, 1.6, 10)
    f.t(38, BND_Y + 26, "⚠️ 「按 lane 切片」这件事是推出来的，不是查到的版图", "sec")
    y = para(f, 38, BND_Y + 50, 1324,
             "推导链有两条。<b>一</b>：4,096 个 ALU 不可能靠一个集中的寄存器堆去喂，"
             "布线在物理上过不去。<b>二</b>：如果寄存器堆是集中的，"
             "那个交叉开关本来就在那儿了，跨 lane 就不该特别贵 —— "
             "<b>而它特别贵，还专门做了一个独立单元。「XLU 存在」这件事本身，"
             "就是版图按 lane 切的证据。</b>", "xs", 18)
    para(f, 38, y + 2, 1324,
         "<r>没查到的：具体的 bank 怎么划、位线怎么排、真实 floorplan —— "
         "公开资料没有。</r>上面讲的是<b>数量级和拓扑</b>，不是版图图纸。", "xs", 18)


if __name__ == "__main__":
    import io
    io.open("out/fig_p29_slices.svg", "w", encoding="utf-8").write(build())
    print("ok fig_p29_slices")
