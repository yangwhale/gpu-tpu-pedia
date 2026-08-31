# -*- coding: utf-8 -*-
"""图 P-30 —— 位线与端口：一张田字格，横的选行，竖的送数据。

**这张图存在的理由是一个被我用错的词。** P-29 早先的版本里写了「时分复用」，
读者顺着这个词推出一个完全合理的担心：<b>64 个 vreg 共用位线，
那岂不是每 64 个周期才轮到一回？</b> 而公开资料明说一个周期就能到 VPU ——
两句话对不上。<b>对不上的原因是那个词错了，不是硬件矛盾。</b>

**共用位线不是轮转分时，是按地址选通。** 而这件事只有画成田字格才说得清：
<b>横着的是字线，一行一根，负责「选中谁」；竖着的是位线，一位一根，
负责「把数据送出去」。</b> 给一个 6 位地址，译码器点亮一根字线，
那一行的 256 个单元同时接到位线上 —— <b>当周期就到，没有排队这回事。</b>

**顺带把「端口」这个词落到实处。** 端口不是软件概念，
是<b>一整套「译码器 ＋ 一组字线 ＋ 一组位线 ＋ 读出电路」</b>。
多端口 ＝ 在<b>同一批存储单元</b>上并排铺好几套线。
于是「多开几个端口不就行了」不成立：<b>每加一个端口，
每个单元横竖两个方向各多一根线，面积按端口数的平方涨。</b>

**⚠️ 出处口径。** 字线／位线／多端口这套结构是<b>数字设计通则</b>，不是 TPU 的公开规格；
TPU 具体用什么单元、几个端口、怎么分 bank，<b>公开资料没有</b>。
但底带那个对账是硬的：<b>64 × 256 × 128 ÷ 8 ＝ 256 KiB，
正好等于公开资料说的「每核约 256 kB vreg」</b> —— 整套模型的外部锚点就是它。
"""
from common import Fig, para, BL, GN, RD, YL, PU, TL, INK, SUB, GREY, FILL

W = 1400

GRID_Y, GRID_H = 84, 360        # ① 田字格
PORT_Y, PORT_H = GRID_Y + GRID_H + 22, 226   # ② 端口是什么
SQ_Y, SQ_H = PORT_Y + PORT_H + 22, 112       # ③ 为什么端口贵
CHK_Y, CHK_H = SQ_Y + SQ_H + 22, 110         # ④ 对账
H = CHK_Y + CHK_H + 20

# 格子几何
DEC_X, DEC_BW = 30, 100          # 地址译码器盒子
LBL_R = 202                      # 行标签右边界（译码器与字线之间）
WL_X = 208                       # 字线从这里开始画，避开标签
GX, GY = 212, GRID_Y + 92
NROW, RH = 8, 26                 # 画 8 行代表 64
NCOL, CW = 13, 40                # 画 13 列代表 256
HOT = 3                          # 被选中的那一行
ANN_X = 792


def build():
    f = Fig(W, H, "TPU 向量寄存器堆的位线结构：横向字线按地址选中一行，"
                  "纵向位线把那一行的数据送出去；一个端口就是一整套这样的线")
    f.title("位线是一张<tspan font-weight=\"700\">田字格</tspan>"
            "　—— 横的选行，竖的送数据，<tspan font-weight=\"700\">"
            "一个端口就是一整套线</tspan>")
    f.legend([(RD, "字线：横着，一行一根，选中谁"),
              (BL, "位线：竖着，一位一根，送数据"),
              (PU, "存储单元"), (GN, "被选中的那一行")])
    _grid(f)
    _port(f)
    _square(f)
    _check(f)
    return f.out()


# ══════════════════════════════════════════════════════════════════════
def _grid(f):
    f.rect(20, GRID_Y, 1360, GRID_H, "#fff", PU, 1.6, 10)
    f.t(38, GRID_Y + 30, "一条 lane 切片里的寄存器堆：64 行 × 256 位的格子", "sec", PU)

    gw = NCOL * CW
    gh = NROW * RH

    # 竖：位线。先画，让单元盖在上面
    for c in range(NCOL):
        x = GX + c * CW + CW / 2
        f.line(x, GY - 14, x, GY + gh + 30, BL, 1.4)
    f.t(GX + gw / 2, GY - 22, "位线　·　竖着走，一位一根　→　共 256 根",
        "xxs", BL, anchor="middle")

    # 横：字线 + 单元
    for r in range(NROW):
        y = GY + r * RH + RH / 2
        hot = (r == HOT)
        f.line(WL_X, y, GX + gw + 8, y, GN if hot else RD,
               2.2 if hot else 1.2)
        for c in range(NCOL):
            x = GX + c * CW + CW / 2
            cc = GN if hot else PU
            f.rect(x - 8, y - 7, 16, 14, FILL[cc], cc, 1.0 if hot else 0.7, 2)
        lab = "vreg %d" % r if r < NROW - 1 else "vreg 63"
        f.t(LBL_R, y + 4, lab, "xxs",
            GN if hot else SUB, anchor="end")
    f.t(GX + gw + 14, GY + HOT * RH + RH / 2 + 4, "← 选中", "xxs", GN)
    f.t(GX + gw / 2, GY + gh + 46, "…… 中间省略，实际 64 行 × 256 列",
        "xxs", SUB, anchor="middle")

    # 译码器
    dy = GY + gh / 2 - 34
    f.rect(DEC_X, dy, DEC_BW, 68, FILL[RD], RD, 1.4, 6)
    f.t(DEC_X + 10, dy + 24, "地址译码器", "lbl", RD)
    f.t(DEC_X + 10, dy + 42, "6 位地址", "xxs", SUB)
    f.t(DEC_X + 10, dy + 57, "→ 选 1 行", "xxs", SUB)
    # 译码器指向被点亮的那一行
    f.path("M %d %d L %d %d L %d %d" % (
        DEC_X + DEC_BW, dy + 34, DEC_X + DEC_BW + 12, dy + 34,
        DEC_X + DEC_BW + 12, GY + HOT * RH + RH / 2), GN, 1.6, marker="aG")

    # 出口
    f.rect(GX, GY + gh + 8, gw, 20, FILL[BL], BL, 1.2, 4)
    f.t(GX + gw / 2, GY + gh + 22, "读出电路　→　这一行的 256 bit 一起出去",
        "xxs", BL, anchor="middle")

    # 右侧注解
    f.rect(ANN_X, GY - 30, 1360 - ANN_X + 20, 172, FILL[GREY], GREY, 1.2, 8)
    y = para(f, ANN_X + 16, GY - 6, 1360 - ANN_X + 4 - 16,
             "<b>横的叫字线，一行一根，管「选中谁」。"
             "竖的叫位线，一位一根，管「把数据送出去」。</b>", "xs", 18)
    y = para(f, ANN_X + 16, y + 8, 1360 - ANN_X + 4 - 16,
             "给一个 6 位地址，译码器<b>点亮一根字线</b>，那一行的 256 个单元"
             "同时接上位线 —— <r>当周期就到，没有排队这回事。</r>", "xs", 18)
    y = para(f, ANN_X + 16, y + 8, 1360 - ANN_X + 4 - 16,
             "所以「64 个共用位线」的真面目是<b>「按地址 64 挑 1」</b>，"
             "<r>不是「轮流用 64 个周期」</r>。", "xs", 18)
    para(f, ANN_X + 16, y + 8, 1360 - ANN_X + 4 - 16,
         "<b>那共用到底代价在哪？</b>就在这张图上：<b>这一整套线，一个周期"
         "只能送一行出去</b>。想同时送 8 行 —— 只能<b>再铺 7 套</b>。"
         "这就是下一格要说的「端口」。", "xs", 18)


# ══════════════════════════════════════════════════════════════════════
# 「端口」这个词一直在用，但从没画出来过。它是硬件，不是比喻。
def _port(f):
    f.rect(20, PORT_Y, 1360, PORT_H, FILL[PU], PU, 1.6, 10)
    f.t(38, PORT_Y + 28, "那「端口」到底是什么？—— "
                         "<tspan font-weight=\"700\">一整套线，不是一个比喻</tspan>",
        "sec", PU)

    # 左：一个端口 = 四件套
    kit = [(RD, "① 地址译码器", "把 6 位地址变成「点亮第几根字线」"),
           (RD, "② 一组字线", "64 根，横着穿过所有单元"),
           (BL, "③ 一组位线", "256 根，竖着穿过所有单元"),
           (BL, "④ 读出／写入电路", "把位线上的电平变成能用的数")]
    y = PORT_Y + 56
    for c, k, v in kit:
        f.rect(38, y, 14, 14, FILL[c], c, 1.0, 3)
        f.t(60, y + 12, k, "lbl", c)
        f.t(196, y + 12, v, "xs", SUB)
        y += 30
    para(f, 38, y + 16, 560,
         "<b>凑齐这四件，才算一个端口。</b>", "xs", 18)

    # 右：多端口 = 同一批单元上并排铺几套
    zx, zy, zw, zh = 660, PORT_Y + 52, 300, 132
    f.rect(zx, zy, zw, zh, "#fff", GREY, 1.2, 6)
    f.t(zx + zw / 2, zy - 8, "放大看同一个存储单元", "xxs", SUB, anchor="middle")
    # 三套字线（横）
    for i, c in enumerate([RD, "#f28b82", "#fbbc04"]):
        yy = zy + 34 + i * 22
        f.line(zx + 8, yy, zx + zw - 8, yy, c, 1.6)
        f.t(zx + zw - 10, yy - 4, "字线%d" % (i + 1), "xxs", SUB,
            anchor="end")
    # 三套位线（竖）
    for i, c in enumerate([BL, "#78a9f7", "#12786f"]):
        xx = zx + 70 + i * 48
        f.line(xx, zy + 10, xx, zy + zh - 10, c, 1.6)
        f.t(xx, zy + zh - 2, "位线%d" % (i + 1), "xxs", SUB, anchor="middle")
    f.rect(zx + 100, zy + 44, 40, 34, FILL[PU], PU, 1.6, 4)
    f.t(zx + 120, zy + 65, "1 bit", "xxs", PU, anchor="middle")

    para(f, 992, PORT_Y + 56, 368,
         "<b>多端口 ＝ 在同一批存储单元上，并排铺好几套线。</b>", "xs", 18)
    para(f, 992, PORT_Y + 96, 368,
         "同一个单元被<b>好几根字线</b>横着穿过、被<b>好几根位线</b>竖着穿过。"
         "于是它可以<b>同时被不同的端口读到</b> —— 这就是"
         "「一个周期读 8 个 vreg」的物理实现。", "xs", 18)


# ══════════════════════════════════════════════════════════════════════
def _square(f):
    f.rect(20, SQ_Y, 1360, SQ_H, FILL[TL], TL, 1.6, 10)
    f.t(38, SQ_Y + 28, "⭐ 那「多开几个端口不就行了」？—— 又一个平方，而且这次是面积",
        "sec", TL)
    y = para(f, 38, SQ_Y + 54, 1324,
             "每加一个端口，每个单元<b>横向多一根字线、纵向多一根位线</b> —— "
             "<b>单元在两个方向上同时变大，面积按端口数的平方涨</b>。"
             "端口翻倍，寄存器堆的面积大约变成四倍。", "xs", 19)
    para(f, 38, y + 2, 1324,
         "<r>所以端口是全芯片最稀缺的预算之一，不是想加就加。</r>"
         "<g>⚠️ 平方关系是数字设计通则；TPU 具体用什么单元、几个端口，公开资料没有。</g>",
         "xs", 19)


# ══════════════════════════════════════════════════════════════════════
# 第一原则：孤立的数字最危险。这一带就是这张图的外部锚点。
def _check(f):
    f.rect(20, CHK_Y, 1360, CHK_H, FILL[GN], GN, 1.8, 10)
    f.t(38, CHK_Y + 28, "✅ 对账：这张格子图能不能跟公开数字对上？", "sec", GN)
    y = para(f, 38, CHK_Y + 54, 1324,
             "一条 lane 切片 <b>64 行 × 256 位 ＝ 16,384 个单元</b>；"
             "全核 <b>× 128 条 lane ＝ 2,097,152 个单元</b>；"
             "换成字节 <b>÷ 8 ＝ 262,144 B ＝ 256 KiB</b>。", "xs", 19)
    para(f, 38, y + 2, 1324,
         "<b>而公开资料写的是「每核约 256 kB 的 vreg」—— 正好对上。</b>"
         "<g>一个孤立的数不能信，但两条独立路径撞出同一个数，整套模型就立住了。</g>",
         "xs", 19)


if __name__ == "__main__":
    import io
    io.open("out/fig_p30_bitline.svg", "w", encoding="utf-8").write(build())
    print("ok fig_p30_bitline")
