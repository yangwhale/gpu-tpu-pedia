# -*- coding: utf-8 -*-
r"""专题四 · §3.2「哪些量必须高精度」——&#160;根子上只有一个画面：大数吃小数

⭐⭐⭐ 2026-09-17 新画。现场原话：「所有的精度问题这些都写得明明白白，画图讲」。
  ⛔ §3.2 是全讲**唯一一节上千字却一张图都没有**的 ——&#160;
    而它讲的那件事**恰恰是最该画的**：
    「bf16 的尾数只有 7 位，所以小于约 1/256 的那部分加上去会被舍掉」。
    这句话是对的，可它是一句**算术**，读者只能选择信或不信。

⭐⭐ 这张图把它**当场做一遍**。脚本里有一个真的 bf16 舍入函数
  （取 float32 的高 16 位，round-to-nearest-even），所有数都是它算出来的：
    · 1.0 加上 3e-4，在 bf16 里**还是 1.0**
    · 而且**连加一千次，还是 1.0** ——&#160;不是慢，是**一步都没动**
    · 换 fp32 同样一千次：1.0 →&#160;1.3
  ⛔ 三条都由 assert 盯着。**这不是「据说」，是跑出来的。**

⭐⭐⭐ Ⓐ 那一格顺带解掉本讲另一处（6.6「bf16 要不要 loss scaling」）：
  **bf16 和 fp32 的指数位一样多（都是 8），差的全在尾数。**
  所以 bf16 ←→ fp32 的范围是一样的，掉的只是精细程度；
  而 fp16 是 5 位指数 ——&#160;**范围小得多，这才是它需要 loss scaling 的原因，
  跟「16 位不够用」无关。** 同样 16 位，两种分法，两种命运。

⭐⭐ Ⓑ 还顺手兑现了正文那个「留口子」：**随机舍入为什么能救回来。**
  按余数的大小决定进位概率，那么每步的**期望**增量正好等于真实增量
  ——&#160;脚本里把这个恒等式也验了。
  ⛔ 但它不是推荐做法：要额外支持，而 fp32 主权重便宜省心。

📌 「AdamW 的一阶矩二阶矩用 bf16 无可观察退化，而主权重与用于累积的梯度仍保 fp32」
  出自 DeepSeek-V3 技术报告 arXiv 2412.19437 §3.3.3（正文已引，这里只复述结论）。
"""
import math
import struct

from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE)

W = 1400


# ══════════════════════════════════════════════════════════════════
# 真的 bf16：取 float32 的高 16 位，round-to-nearest-even
#   ⭐ 判据：**一张图要让人相信某个算术结果，就让脚本真的算那个算术。**
#     这里不用 numpy ——&#160;struct 就够，而且谁都能自己跑一遍验。
# ══════════════════════════════════════════════════════════════════
def bf16(x):
    b = struct.unpack("<I", struct.pack("<f", float(x)))[0]
    lsb = (b >> 16) & 1                     # 目标最低位，用于 ties-to-even
    b = (b + 0x7FFF + lsb) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", b))[0]


WEIGHT = 1.0
UPDATE = 3e-4                 # 训练后期单步更新的典型量级
STEPS = 1000

# bf16 在 1.0 附近相邻两个可表示数的间距（尾数 7 位 → 2⁻⁷）
ULP = 2.0 ** -7
assert bf16(WEIGHT + ULP) > WEIGHT, "跨一整格必须加得上去"
assert bf16(WEIGHT + ULP * 0.4) == WEIGHT, "不到半格就该被舍回来"

# ⭐⭐⭐ 这张图最狠的一条：连加一千次，一动不动
_x = WEIGHT
for _ in range(STEPS):
    _x = bf16(_x + UPDATE)
BF16_AFTER = _x
FP32_AFTER = WEIGHT + STEPS * UPDATE
assert BF16_AFTER == WEIGHT, \
    "bf16 累加一千次竟然动了（%r）——　那这张图的主张就不成立" % BF16_AFTER
assert abs(FP32_AFTER - 1.3) < 1e-9, "fp32 那边应当老老实实加到 1.3"

# 要连加多少次才够跨过一格（假设不被舍掉）
N_PER_ULP = ULP / UPDATE
# 随机舍入：进位概率 ＝ 余数 ÷ 一格，于是每步期望增量正好 ＝ 真实增量
P_UP = UPDATE / ULP
assert abs(P_UP * ULP - UPDATE) < 1e-18, "随机舍入的无偏性就靠这一行"

# 位宽表：(名字, 符号, 指数, 尾数)
FORMATS = (
    ("fp32", 1, 8, 23, GY2),
    ("bf16", 1, 8, 7, GR),
    ("fp16", 1, 5, 10, OR),
)
_bf = next(f for f in FORMATS if f[0] == "bf16")
_f32 = next(f for f in FORMATS if f[0] == "fp32")
_f16 = next(f for f in FORMATS if f[0] == "fp16")
assert _bf[2] == _f32[2], "bf16 跟 fp32 的指数位必须一样 ——　Ⓐ 的全部落点"
assert _f16[2] < _bf[2] and _f16[3] > _bf[3], \
    "fp16 应当是「尾数更多、指数更少」——　它跟 bf16 的取舍正好相反"


def main():
    f = Fig(W, "精度问题的根子只有一个画面：大数吃小数。"
               "bf16 的尾数只有七位，所以在 1.0 附近，"
               "相邻两个能表示的数之间隔着约百分之零点八；"
               "一个万分之三的更新量加上去，四舍五入直接被舍回原地。"
               "脚本当场跑了一千次，bf16 那边一动没动，fp32 那边加到了一点三。"
               "另外 bf16 和 fp32 的指数位一样多，差的全在尾数，所以范围相同；"
               "而 fp16 是五位指数、范围小得多，"
               "这才是它需要 loss scaling 的原因。"
               "最后按「老的贡献会不会永远不走」把各个量分成三类，"
               "决定谁必须 fp32")

    y0 = f.header(
        "精度问题，根子上只有一个画面　——　<tspan font-weight=\"700\">"
        "大数吃小数</tspan>",
        "⭐ 这一格里<tspan font-weight=\"700\">每一个数都是脚本当场算的</tspan>"
        "　——　你也可以自己跑一遍",
        [(GR, "bf16：范围够，精细不够"), (OR, "fp16：精细够，范围不够"),
         (GY2, "fp32：都够，但四倍大")])

    # ══════════ Ⓐ 同样 16 位，两种分法 ═════════════════════════════
    PH = 356
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 一个小数在机器里<tspan font-weight=\"700\">分成两段存</tspan>"
                 "　——　<tspan font-weight=\"700\">指数管「能多大」，尾数管「能多细」</tspan>",
                 BL,
                 sub="⭐ 看清楚：<tspan font-weight=\"700\">bf16 和 fp32 的指数段一样长</tspan>")

    X0, BW = 200, 900
    UNIT = BW / 32.0                       # 每一位多少像素，按 fp32 的 32 位定
    for i, (name, sg, ex, ma, col) in enumerate(FORMATS):
        ry = py + 52 + i * 84
        f.t(X0 - 20, ry + 34, name, col, True, 18, "end")
        f.t(X0 - 20, ry + 56, "%d 位" % (sg + ex + ma), GY2, size=12, anchor="end")
        x = X0
        for w, fill, stroke, lab in ((sg, "#f1f3f4", GY2, ""),
                                     (ex, "#e8f0fe", BL, "指数 %d 位" % ex),
                                     (ma, "#e6f4ea", GR, "尾数 %d 位" % ma)):
            f.box(x, ry, w * UNIT, 48, fill, stroke, 5)
            if lab and w * UNIT > 90:
                f.t(x + w * UNIT / 2, ry + 30, lab, INK, True, 13.5, "middle")
            elif lab:
                f.t(x + w * UNIT / 2, ry + 30, str(ma if "尾" in lab else ex),
                    INK, True, 13.5, "middle")
            x += w * UNIT
        if name == "bf16":
            f.t(x + 22, ry + 22, "✅ 指数跟 fp32 一样长", GR, True, 13.5)
            f.t(x + 22, ry + 42, "→ 范围一样，只是变粗", GY, size=12.5)
        if name == "fp16":
            f.t(x + 22, ry + 22, "⛔ 指数短了 3 位", OR, True, 13.5)
            f.t(x + 22, ry + 42, "→ 小的梯度直接掉到 0", GY, size=12.5)

    f.t(700, py + 320,
        "⭐⭐⭐ <tspan font-weight=\"700\">bf16 和 fp16 都是 16 位，只是分法不同</tspan>"
        "　——　所以 fp16 要 loss scaling，"
        "<tspan font-weight=\"700\">不是因为「16 位不够用」，是因为它把位数分给了尾数。</tspan>",
        INK, size=14.5, anchor="middle")
    f._pan = None

    # ══════════ Ⓑ 大数吃小数，当场做一遍 ═══════════════════════════
    PH2 = 452
    py2 = f.panel(0, py + PH + 20, W, PH2,
                  "Ⓑ ⭐⭐⭐ 于是<tspan font-weight=\"700\">「加了等于没加」</tspan>"
                  "　——　这不是比喻，是真的一动不动", RD,
                  sub="⭐ 权重 1.0，每步加 %.4f（训练后期的典型量级）" % UPDATE)

    AX0, AX1, AY = 180, 1080, py2 + 130
    f.line(AX0 - 20, AY, AX1 + 60, AY, GY2, 1.2, arrow=False)
    # bf16 在 1.0 附近只有这么几个刻度，中间是空的
    for k in range(4):
        xx = AX0 + k * 260
        f.line(xx, AY - 16, xx, AY + 16, INK, 2.0, arrow=False)
        f.t(xx, AY + 40, "%.7f" % (1.0 + k * ULP), INK if k == 0 else GY2,
            k == 0, 13, "middle")
    f.t(AX0 + 130, AY - 36, "这中间<tspan font-weight=\"700\">什么都没有</tspan>",
        GY2, size=13, anchor="middle")
    f.t(AX0 + 130, AY - 14, "一格 ＝ %.7f" % ULP, GY2, size=12, anchor="middle")

    # 那个可怜的更新量
    _upx = AX0 + 260 * (UPDATE / ULP)
    f.line(AX0, AY - 70, _upx, AY - 70, RD, 2.4)
    f.t(AX0 + 8, AY - 82, "要加的量 %.4f" % UPDATE, RD, True, 13.5)
    f.t(_upx + 14, AY - 66,
        "⛔ 只有一格的 <tspan font-weight=\"700\">%.1f%%</tspan>"
        "　——　四舍五入直接舍回原地" % (P_UP * 100), RD, True, 13.5)

    f.box(1150, py2 + 44, 200, 172, "#fce8e6", RD, 8)
    f.t(1250, py2 + 78, "要连加", RD, True, 15, "middle")
    f.t(1250, py2 + 112, "%d 次" % int(round(N_PER_ULP)), RD, True, 30, "middle")
    f.t(1250, py2 + 146, "才够跨过一格", GY, size=13, anchor="middle")
    f.t(1250, py2 + 188, "⛔ 可它跨不过去", RD, True, 14, "middle")

    # 一千步的对照 ——&#160;这是全图的落点
    f.box(180, py2 + 232, 560, 132, "#fce8e6", RD, 8)
    f.t(460, py2 + 268, "bf16 里连加 %s 次" % "{:,}".format(STEPS),
        RD, True, 17, "middle")
    f.t(460, py2 + 306, "1.0　→　<tspan font-weight=\"700\">%.1f</tspan>"
        % BF16_AFTER, INK, True, 24, "middle")
    f.t(460, py2 + 342, "⛔ 一步都没动 ——　每次都被舍回去了",
        RD, True, 14, "middle")

    f.box(790, py2 + 232, 560, 132, "#e6f4ea", GR, 8)
    f.t(1070, py2 + 268, "同样 %s 次，改用 fp32" % "{:,}".format(STEPS),
        GR, True, 17, "middle")
    f.t(1070, py2 + 306, "1.0　→　<tspan font-weight=\"700\">%.1f</tspan>"
        % FP32_AFTER, INK, True, 24, "middle")
    f.t(1070, py2 + 342, "✅ 该加的都加上了", GR, True, 14, "middle")

    f.t(700, py2 + 404,
        "⭐⭐ 所以「主权重留一份 fp32」<tspan font-weight=\"700\">不是保险起见</tspan>"
        "　——　<tspan font-weight=\"700\">不留，训练到后期就真的停在原地了。</tspan>",
        INK, size=15, anchor="middle")
    f._pan = None

    # ══════════ Ⓒ 三类量，三种命运 ═════════════════════════════════
    PH3 = 356
    py3 = f.panel(0, py2 + PH2 + 20, W, PH3,
                  "Ⓒ ⭐⭐ 那谁必须 fp32？判据一句话："
                  "<tspan font-weight=\"700\">老的贡献会不会永远不走</tspan>", PU,
                  sub="⛔ 不是「累不累加」——&#160;那个说法太粗，"
                      "<tspan font-weight=\"700\">被 DeepSeek-V3 一条实测打中过</tspan>")

    # ⭐⭐⭐ 2026-09-18 R20 重画。旧版是三张卡片，每张四行字 ——&#160;
    #   把字删掉只剩三个一样的空框，跟 R13 那个 `fig-circuit` Ⓑ 一模一样的病。
    #   ⭐⭐⭐ 而这一格的判据「**老的贡献会不会永远不走**」
    #   **本身就是一条衰减曲线的形状**：
    #     · 主权重：一直加进去，**永远不走** ——&#160;水平线
    #     · 动量 m（β₁）／二阶矩 v（β₂）：滑动平均，**指数衰减**
    #     · 梯度／激活：算完就扔，**一步就没**
    #   ⭐ 而 β 的半衰期是**可以算的**，所以这三条线不是示意，是真函数。
    #   ⛔ 横轴必须用**对数**：β₂ ＝ 0.999 的半衰期接近七百步，
    #     线性轴上它跟主权重看着一样平 ——&#160;那就把论点画没了。
    B1, B2 = 0.9, 0.999            # Adam 的两个默认值
    HL_M = math.log(0.5) / math.log(B1)
    HL_V = math.log(0.5) / math.log(B2)
    TRAIN_STEPS = 100000           # 一次十万步的训练
    assert HL_V / HL_M > 100, "两个半衰期要差出两个量级，否则画在一起没意义"
    # ⭐⭐ 这一格的落点：十万步之后，连记得最久的 v 也早忘光了，只有主权重还在
    assert B2 ** TRAIN_STEPS < 1e-40, "十万步后 v 应当已经彻底忘光"

    KX0, KX1 = 150, 1180
    KT, KB = py3 + 56, py3 + 232
    LX0, LX1 = 0.0, math.log10(TRAIN_STEPS)

    def kx(k):
        return KX0 + (KX1 - KX0) * (math.log10(max(k, 1.0)) - LX0) / (LX1 - LX0)

    def ky(v):
        return KB - (KB - KT) * v

    f.line(KX0, KB, KX1 + 20, KB, GY2, 1.2, arrow=False)
    f.line(KX0, KB, KX0, KT - 8, GY2, 1.2, arrow=False)
    for e in range(0, 6):
        xx = kx(10.0 ** e)
        f.line(xx, KB, xx, KB + 6, GY2, 1, arrow=False)
        f.t(xx, KB + 24, "10%s" % ("⁰" if e == 0 else "%d" % e).replace("0", "⁰")
            if e == 0 else "10%s" % "¹²³⁴⁵"[e - 1], GY2, size=11.5, anchor="middle")
    f.t(KX1 + 26, KB + 6, "步", GY2, size=12)
    f.t(KX0, KT - 20, "这一步的贡献，到现在还剩多少", GY2, size=12)
    # 半衰期那条参考线
    f.line(KX0, ky(0.5), KX1, ky(0.5), "#dadce0", 1.0, dash="5 4", arrow=False)
    f.t(KX0 - 8, ky(0.5) + 4, "一半", GY2, size=11, anchor="end")

    N = 300
    for decay, col, nm, note in (
            (None, RD, "主权重", "⛔ 必须 fp32"),
            (B2, GR, "二阶矩 v（β₂＝%.3f）" % B2, "✅ bf16 扛得住"),
            (B1, GR, "动量 m（β₁＝%.1f）" % B1, "✅ bf16 扛得住"),
            (0.0, BL, "梯度／激活", "✅ bf16 够")):
        d = None
        for i in range(N + 1):
            k = 10.0 ** (LX0 + (LX1 - LX0) * i / N)
            v = 1.0 if decay is None else (decay ** k if decay > 0 else
                                           (1.0 if k < 1.5 else 0.0))
            pt = "%.1f %.1f" % (kx(k), ky(v))
            d = ("M " + pt) if d is None else d + " L " + pt
        f.path(d, col, 2.6 if decay is None else 2.0,
               dash=None if decay is None else ("4 3" if decay == B1 else None),
               arrow=False)

    # 标注放在各自「掉下去」的地方，不堆在一起
    f.t(kx(TRAIN_STEPS) + 14, ky(1.0) + 5, "主权重", RD, True, 14)
    f.t(kx(TRAIN_STEPS) + 14, ky(1.0) + 25, "⛔ 必须 fp32", RD, size=12)
    f.t(kx(HL_V) + 10, ky(0.5) - 12, "v：半衰期 %.0f 步" % HL_V, GR, True, 13)
    f.t(kx(HL_M) + 8, ky(0.5) - 34, "m：%.1f 步" % HL_M, GR, True, 13)
    f.t(kx(1.6) + 6, ky(0.18), "梯度／激活：下一步就没了", BL, True, 13)

    f.t(700, py3 + 288,
        "⭐⭐⭐ 只有<tspan font-weight=\"700\">那条一直平着的</tspan>必须 fp32"
        "　——　它装的是<tspan font-weight=\"700\">十万步前那一点点增量，而那一点现在还在里面</tspan>。"
        "其余三条都会被忘掉，<tspan font-weight=\"700\">忘得掉的就存得粗。</tspan>",
        INK, size=14.5, anchor="middle")
    f.t(700, py3 + 320,
        "⭐ 所以 12 个 fp32 字节装的正好是「不走的」，2 个 bf16 字节装的正好是「会走的」"
        "　——　<tspan font-weight=\"700\">这条线不是拍出来的，是这条曲线画出来的。</tspan>"
        "　⚠️ 梯度一做累积就又变成「要留一阵子」，于是升回 fp32。",
        GY, size=13.5, anchor="middle")
    f._pan = None

    yb = f.band(py3 + PH3 + 20, "ok",
                "而这条规则<tspan font-weight=\"700\">可以被打破</tspan> ——　只要把「每次都朝同一边抹零」去掉",
                ("⭐⭐⭐ 真正致命的<tspan font-weight=\"700\">不是「加的量小」，是"
                 "「每次都朝同一个方向抹零」</tspan>。"
                 "换成<tspan font-weight=\"700\">随机舍入</tspan>（按余数大小决定进位概率），"
                 "那么每步的<tspan font-weight=\"700\">期望</tspan>增量"
                 "正好等于真实增量 ——&#160;丢掉的那部分，多步之后会被找回来。"
                 "（这个恒等式脚本里验了。）",
                 "⛔ 但这<tspan font-weight=\"700\">不是推荐做法</tspan>：它要额外的硬件/框架支持，"
                 "而 fp32 主权重便宜又省心。"
                 "放在这儿只是因为 ——&#160;<tspan font-weight=\"700\">"
                 "知道一条规则「为什么成立」，才知道它什么时候可以不成立。</tspan>",
                 "⚠️ 图上的 1.0 和 %.4f 是<tspan font-weight=\"700\">量级示意</tspan>："
                 "真实权重不都在 1.0 附近，真实更新量也随训练阶段变。"
                 "⭐ 但**相对**关系是对的 ——&#160;bf16 的相对精度就是约 1/256，"
                 "跟数本身多大无关。" % UPDATE))

    yb = f.src(yb + 16,
               "📌 Ⓑ 里每一个数都是脚本用一个<tspan font-weight=\"700\">真的 bf16 舍入函数</tspan>"
               "算出来的（取 float32 的高 16 位，round-to-nearest-even，"
               "只用标准库 struct）。三条 assert 盯着：不到半格必须被舍回、"
               "跨一整格必须加得上、"
               "<tspan font-weight=\"700\">连加 %s 次必须仍然等于 1.0</tspan>。"
               "⭐ 判据：<tspan font-weight=\"700\">"
               "一张图要让人相信某个算术结果，就让脚本真的算那个算术。</tspan>"
               % "{:,}".format(STEPS),
               "📌 Ⓒ 那条「不是累不累加，是老的贡献会不会永远不走」是被一条"
               "<tspan font-weight=\"700\">第一方反证</tspan>逼准的："
               "DeepSeek-V3 技术报告（arXiv <tspan font-weight=\"700\">2412.19437</tspan> §3.3.3）"
               "明说 AdamW 的一阶矩二阶矩用 bf16「未观察到性能下降」，"
               "而主权重、以及<tspan font-weight=\"700\">用于累积的梯度</tspan>仍保 fp32。"
               "⛔ 判据（元级）：<tspan font-weight=\"700\">"
               "一条判据被反例打中的时候，先别扔它 ——&#160;多半是它的措辞比它的机制粗。</tspan>",
               "⭐ Ⓐ 顺带解掉 6.6 那条："
               "<tspan font-weight=\"700\">fp16 需要 loss scaling 跟「16 位不够」无关</tspan>"
               "　——　它跟 bf16 一样是 16 位，只是把 3 位从指数挪给了尾数，"
               "于是范围小了三个二进制数量级，小梯度直接下溢到 0。")

    f.save("fig4-precision.svg", yb + 14)


if __name__ == "__main__":
    main()
