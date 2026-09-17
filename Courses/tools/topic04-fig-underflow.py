# -*- coding: utf-8 -*-
r"""专题四 · §6.6「bf16 要不要 loss scaling」——&#160;问题不是「多细」，是「多小」

⭐⭐⭐ 2026-09-18 新画（R18）。素材库那条（Ⓑ 级）说的是：
  混合精度那张经典图是**梯度的对数直方图 ＋ 一条 FP16 下界竖线**，
  竖线**左边一大片**标着「换成 FP16 后会被抹成 0」。
  ⭐ 判据：**「精度不够」是形容词，「这一竖线左边全没了」是可数的损失。**

⛔⛔ 两件不能做的事，以及绕过去的办法：
  ① **那张原图不能复制**（公开仓库 ＋ 版权）；
  ② **那份直方图的数据我们没有** ——&#160;编一个然后画出来就是造数据。
  ⭐⭐⭐ 但**竖线的位置是 IEEE 754 的定义，可以精确算** ——&#160;
    这正是那条判据：**把「按定义必然成立」的那一段真算出来，别连它一起手画。**
    所以这张图**没有直方图**，只有一根**真数轴**和几条**真边界**。

⭐⭐⭐ 算出来之后发现一个比原图更好的画面：
  **bf16 的下界和 fp16 的下界之间，隔着 33 个数量级的空白** ——&#160;
  那一大段空白本身就是论点，比「指数位 5 位 vs 8 位」直观得多。
  ⛔ 而这正好补上 `fig-precision` Ⓐ 的缺口：
    **那边的位段条讲的是「能表示多细」，这里的数轴讲「能表示多小」** ——&#160;
    而 §6.6 那个误解的分界，恰恰在后者。

⚠️ 图上每个数都是当场算的（`numpy.float16` 的位模式），不是查表抄的；
  「低于某个值会被舍成 0」这一条也是**真的舍了一遍**验出来的。
"""
import math
import struct

import numpy as np

from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE)

W = 1400


def _h(bits):
    """16 位模式 →&#160;float。⭐ 不查表、不凭记忆，直接从位模式反推。"""
    return float(np.frombuffer(struct.pack("<H", bits), dtype=np.float16)[0])


FP16_MIN_NORMAL = _h(0x0400)          # 2⁻¹⁴　最小正规数
FP16_MIN_SUB = _h(0x0001)             # 2⁻²⁴　最小次正规数
BF16_MIN_NORMAL = 2.0 ** -126         # 跟 fp32 同指数位宽，所以同下界

assert abs(math.log2(FP16_MIN_NORMAL) + 14) < 1e-9
assert abs(math.log2(FP16_MIN_SUB) + 24) < 1e-9

# ⭐⭐ 这一格的主角：两个下界之间差了多少个数量级
DECADES = math.log10(FP16_MIN_NORMAL / BF16_MIN_NORMAL)
assert DECADES > 30, "两条下界要差出三十个数量级以上，现在 %.1f" % DECADES

# ⭐ 真的舍一遍 ——&#160;「低于这个就没了」不是说说而已
GRAD = 2.0e-8                          # 一个不算离谱的小梯度
assert float(np.float16(GRAD)) == 0.0, "这个值在 fp16 下应当被舍成 0"
assert float(np.float16(BF16_MIN_NORMAL)) == 0.0   # 顺带：bf16 的下界在 fp16 眼里也是 0

LOSS_SCALE = 1024                      # ＝ 2¹⁰，混合精度里的常见取值
SCALED = GRAD * LOSS_SCALE
assert float(np.float16(SCALED)) > 0.0, "放大之后必须活过来，否则这张图白画"
SHIFT_DECADES = math.log10(LOSS_SCALE)
# ⚠️ 放大后落在**次正规区**（能表示，但精度打折）——&#160;图上要诚实标出来
assert FP16_MIN_SUB < SCALED < FP16_MIN_NORMAL

X_LO, X_HI = -40.0, 0.0                # log10 数轴范围


def main():
    f = Fig(W, "一根从十的负四十次方到 1 的对数数轴，上面标着三条边界线。"
               "最右边橙色那条是 fp16 的最小正规数，六点一乘十的负五次方；"
               "紧挨着左边红色那条是 fp16 的最小次正规数，五点九六乘十的负八次方，"
               "再往左的数在 fp16 里会被直接舍成零。"
               "而绿色那条 bf16 的下界远在左边，一点一八乘十的负三十八次方 —— "
               "两者之间隔着三十三个数量级的空白，那一整段都是 bf16 存得下、"
               "fp16 存不下的范围。"
               "下半张画的是 loss scaling 在干什么：把一个会被舍成零的梯度乘上一千零二十四，"
               "在这根对数轴上就是整体向右平移三格，于是它跨过红线活了下来")

    y0 = f.header(
        "fp16 的毛病不是<tspan font-weight=\"700\">「不够细」</tspan>，是"
        "<tspan font-weight=\"700\">「下不去」</tspan>",
        "⭐ 这一格只有<tspan font-weight=\"700\">一根真数轴和几条真边界</tspan>"
        "　——　每个数都是从 IEEE 754 的位模式当场算的",
        [(OR, "fp16 正规数下界"), (RD, "fp16 归零线"), (GR, "bf16 下界")])

    AX0, AX1 = 120, 1300

    def ax(v):
        return AX0 + (AX1 - AX0) * (math.log10(v) - X_LO) / (X_HI - X_LO)

    # ══════════ Ⓐ 三条边界，和它们之间那一大段空白 ═══════════════════
    PH = 356
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 同一根对数轴上，"
                 "<tspan font-weight=\"700\">两种 16 位格式的下界差了 %d 个数量级</tspan>"
                 % int(DECADES), INK,
                 sub="⭐ 轴上越往左数越小；<tspan font-weight=\"700\">"
                     "过了红线，那个数在 fp16 里就是 0</tspan>")

    AY = py + 176
    # ⭐⭐ 先把那段空白填出来 ——&#160;它才是这一格的主角
    f.box(ax(BF16_MIN_NORMAL), AY - 54, ax(FP16_MIN_SUB) - ax(BF16_MIN_NORMAL),
          108, "#e6f4ea", "#e6f4ea", 4)
    f.t((ax(BF16_MIN_NORMAL) + ax(FP16_MIN_SUB)) / 2.0, AY - 22,
        "这一整段：<tspan font-weight=\"700\">bf16 存得下，fp16 存不下</tspan>",
        GR, True, 15, "middle")
    f.t((ax(BF16_MIN_NORMAL) + ax(FP16_MIN_SUB)) / 2.0, AY + 2,
        "<tspan font-weight=\"700\">%d 个数量级</tspan>" % int(DECADES),
        GR, True, 17, "middle")

    f.line(AX0 - 10, AY + 40, AX1 + 20, AY + 40, GY2, 1.4)
    for e in range(-40, 1, 5):
        xx = ax(10.0 ** e)
        f.line(xx, AY + 40, xx, AY + 47, GY2, 1, arrow=False)
        f.t(xx, AY + 66, "10%s" % ("⁰" if e == 0 else "⁻%d" % -e),
            GY2, size=11.5, anchor="middle")
    f.t(AX1 + 26, AY + 44, "大 →", GY2, size=12)

    for v, col, name, note, dy in (
            (BF16_MIN_NORMAL, GR, "bf16", "%.2e" % BF16_MIN_NORMAL, -92),
            (FP16_MIN_SUB, RD, "fp16 归零线", "%.2e" % FP16_MIN_SUB, -92),
            (FP16_MIN_NORMAL, OR, "fp16 正规数下界", "%.2e" % FP16_MIN_NORMAL, -128)):
        xx = ax(v)
        f.line(xx, AY + 40, xx, AY + dy + 16, col, 2.2, arrow=False)
        f.t(xx, AY + dy, name, col, True, 13.5, "middle")
        f.t(xx, AY + dy + 20, note, col, size=11.5, anchor="middle")

    f.t(700, py + 306,
        "⭐⭐⭐ 所以 fp16 的问题从来不是<tspan font-weight=\"700\">「16 位不够细」</tspan>"
        "　——　<tspan font-weight=\"700\">bf16 也是 16 位，而且尾数比它还少</tspan>。"
        "问题是它的<tspan font-weight=\"700\">指数位只有 5 位，下界抬得太高</tspan>。",
        INK, size=14.5, anchor="middle")
    f._pan = None

    # ══════════ Ⓑ loss scaling ＝ 在这根轴上整体右移 ═════════════════
    PH2 = 336
    py2 = f.panel(0, py + PH + 20, W, PH2,
                  "Ⓑ ⭐⭐ 那 loss scaling 在干什么　——　"
                  "<tspan font-weight=\"700\">就是把整条分布在这根轴上往右推</tspan>", BL,
                  sub="⭐ 乘 %d ＝ 在对数轴上<tspan font-weight=\"700\">平移 %.2f 格</tspan>"
                      "　——　够不够，看它跨没跨过红线"
                      % (LOSS_SCALE, SHIFT_DECADES))

    BY = py2 + 150
    f.line(AX0 - 10, BY, AX1 + 20, BY, GY2, 1.4)
    for v, col in ((FP16_MIN_SUB, RD), (FP16_MIN_NORMAL, OR)):
        f.line(ax(v), BY - 96, ax(v), BY + 10, col, 2.0, dash="6 4", arrow=False)

    gx, sx = ax(GRAD), ax(SCALED)
    f.box(gx - 7, BY - 22, 14, 14, "#fce8e6", RD, 3, sw=1.6)
    f.t(gx, BY + 26, "一个梯度 %.0e" % GRAD, RD, True, 13, "middle")
    f.t(gx, BY + 46, "<tspan font-weight=\"700\">fp16 存 → 0</tspan>",
        RD, size=12.5, anchor="middle")

    f.line(gx + 10, BY - 42, sx - 6, BY - 42, BL, 2.4)
    f.t((gx + sx) / 2.0, BY - 52, "× %d" % LOSS_SCALE, BL, True, 14, "middle")

    f.box(sx - 7, BY - 22, 14, 14, "#e8f0fe", BL, 3, sw=1.6)
    f.t(sx + 66, BY + 26, "变成 %.2e" % SCALED, BL, True, 13, "middle")
    f.t(sx + 66, BY + 46, "<tspan font-weight=\"700\">活下来了</tspan>",
        BL, size=12.5, anchor="middle")

    f.t(700, py2 + 238,
        "⚠️ 诚实一句：放大之后它落在 fp16 的"
        "<tspan font-weight=\"700\">次正规区</tspan>"
        "（红线右、橙线左）——&#160;<tspan font-weight=\"700\">存得下了，但精度打折</tspan>。"
        "要进正规区得用更大的 scale。", GY, size=13.5, anchor="middle")
    f.t(700, py2 + 276,
        "⭐⭐⭐ 而 bf16 <tspan font-weight=\"700\">根本不用做这件事</tspan>"
        "　——　它的下界在左边 %d 个数量级之外，"
        "<tspan font-weight=\"700\">那些梯度本来就在范围内</tspan>。" % int(DECADES),
        INK, size=14.5, anchor="middle")
    f._pan = None

    yb = f.band(py2 + PH2 + 20, "ok",
                "所以「bf16 要不要 loss scaling」这个问题，"
                "答案在<tspan font-weight=\"700\">指数位</tspan>上，不在位数上",
                ("⭐ bf16 和 fp16 <tspan font-weight=\"700\">都是 16 位</tspan>，"
                 "可 bf16 把位数分给了<tspan font-weight=\"700\">指数</tspan>（8 位，跟 fp32 一样），"
                 "fp16 分给了<tspan font-weight=\"700\">尾数</tspan>（10 位）。"
                 "⛔ 于是 bf16 <tspan font-weight=\"700\">更粗但更宽</tspan>。",
                 "⭐⭐ 判据：<tspan font-weight=\"700\">"
                 "「精度不够」是个形容词，「这条线左边全没了」是可数的损失。</tspan>"
                 "　——&#160;把一个笼统的担心换成一根能指的线，误解自己就散了。"))

    yb = f.src(yb + 16,
               "📌 图上每个边界都是从 <tspan font-family=\"ui-monospace,monospace\">"
               "numpy.float16</tspan> 的位模式当场算的："
               "最小正规数 0x0400 ＝ 2⁻¹⁴，最小次正规数 0x0001 ＝ 2⁻²⁴；"
               "bf16 与 fp32 同为 8 位指数，下界 2⁻¹²⁶。"
               "「低于红线会被舍成 0」是<tspan font-weight=\"700\">真舍了一遍</tspan>验出来的。",
               "⭐ 这张图的<tspan font-weight=\"700\">讲法</tspan>取自混合精度那篇经典图"
               "（梯度直方图 ＋ 一条 FP16 下界竖线，见 Narang &amp; Micikevicius et al., 2018）"
               "　——　⛔ <tspan font-weight=\"700\">但我们没有那份直方图数据，"
               "所以这里不画直方图</tspan>，只画真边界。")

    f.save("fig4-underflow.svg", yb + 14)


if __name__ == "__main__":
    main()
