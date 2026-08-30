# -*- coding: utf-8 -*-
"""图 P-11 —— 这套「先在 CPU 上搜」最后换回了什么。

P-10 停在「我们扫了 29 组配置」。**那不是落点，那是过程。**
这一张回答「所以呢」，而且刻意把两样东西并排放：

  甲　它换回来的收益　　—— 630 → 666.6，而且说清哪一段是谁的功劳
  乙　它给不了的那部分　—— 同一组配置，两个指标排序相反

**为什么这两格必须同框**：单看甲，会得出「AOT 帮我们提了 5.8%」——
那是错的，AOT 一个性能数字都给不了。单看乙，又像在说这工具不可靠。
**合起来才是真话：它把真机额度省到该花的地方，性能全部来自真机。**

⚠️ **两个刻意的画法选择，都是为了不撒谎：**

① 甲不画条形图。630 → 666.6 只差 5.8%，从 0 起画看不见，
   而把轴从 600 起画就是在放大差异。**所以干脆不画轴，只画阶梯和箭头** ——
   数字本身就是刻度。

② 乙那两栏**单位不同、而且其中一栏我们自己没记清**。
   AOT 那栏确定是 GiB；真机峰值 HBM 那栏源记录只写「91.94 G」，
   没写清是 GiB 还是十进制 GB。**所以这一格只用方向，不用绝对值** ——
   方向（谁比谁大）不受单位影响，这正好是它能成立的原因。
"""
from common import Fig, para, BL, RD, GN, YL, PU, INK, SUB, LINE, GREY, FILL

W = 1400
TOP = 84

R1_Y = TOP + 40
R1_H = 340
AX, AW = 20, 792
BX, BW = 824, 556

BAND_Y = R1_Y + R1_H + 28
H = BAND_Y + 120

# ── 甲：一段一段说清 5.8% 是谁的功劳 ────────────────────────────
# per-chip TFLOP/s（= 日志的 per-device × 2）与 MFU（÷ 2307）
STEPS = [("此前最优", "630", "27.31%", GREY, ""),
         ("＋ 18 个 tile 参数", "662.2", "28.70%", GN,
          "<b>+32.2</b>　<g>反向那两条路也被 tile 了</g>"),
         ("＋ pdbs 12 → 13", "666.6", "28.89%", GN,
          "<b>+4.4</b>　<g>就是 AOT 扫出来的那个反直觉档</g>")]

# ── 乙：同一组配置，两个指标排序相反 ────────────────────────────
PAIR = [("AOT 算的 temp", "GiB（口径明确）", "74.95", "73.11", "↓ 1.84", GN),
        ("真机打印的峰值 HBM", "源记录只写「G」，没写清口径", "91.94", "92.57", "↑ 0.63", RD)]


def build():
    f = Fig(W, H, "AOT 配置搜索最终兑现的收益：per-chip 630 → 666.6 TFLOP/s，"
                  "以及同一组配置上 AOT temp 与真机峰值 HBM 排序相反的边界案例")
    f.title("所以呢　—— <tspan font-weight=\"700\" fill=\"#1e8e3e\">"
            "这套先在 CPU 上搜，最后换回了 5.8%</tspan>")
    f.legend([(GN, "真机实测的收益"), (RD, "工具够不到的那一格")])

    _steps(f)
    _pair(f)
    _band(f)
    f.t(W - 20, H - 12,
        "实测：Hunyuan3-295B-A21B · 64 芯片 4x4x4 · per-chip = 日志 per-device × 2 · "
        "MFU = per-chip ÷ 2307",
        "xxs", GREY, "end")
    return f.out()


# ══════════════════════════════════════════════════════════════════════
def _steps(f):
    f.rect(AX, R1_Y, AW, R1_H, "#fff", LINE, 1.2, 10)
    f.t(AX + 18, R1_Y + 26, "甲　·　它换回来的收益", "sec")
    para(f, AX + 18, R1_Y + 46, AW - 36,
         "<g>不画条形图：630 → 666.6 只差 5.8%，从 0 起画看不见，"
         "把轴从 600 起画又是在放大差异。<b>数字本身就是刻度。</b></g>", "xs", 16)

    y = R1_Y + 76
    for i, (name, tf, mfu, c, delta) in enumerate(STEPS):
        if delta:
            # 增量画在箭头旁边：它是「这一步加了多少」，不是这一行的属性。
            f.line(AX + 60, y - 20, AX + 60, y - 4, SUB, 1.4, "aK")
            para(f, AX + 80, y - 8, AW - 116, delta, "xxs", 13)
        f.rect(AX + 18, y, AW - 36, 52, FILL[c] if c != GREY else "#fff", c, 1.2, 8)
        f.t(AX + 32, y + 31, name, "box", c if c != GREY else INK)
        f.t(AX + 330, y + 33, tf, "numb", c if c != GREY else INK, "end")
        f.t(AX + 340, y + 33, "TFLOP/s per-chip", "xxs", SUB)
        f.t(AX + 540, y + 33, "MFU " + mfu, "lbl", c if c != GREY else INK)
        y += 72

    f.rect(AX + 18, R1_Y + 286, AW - 36, 40, "#fff", RD, 1.2, 8)
    para(f, AX + 32, R1_Y + 308, AW - 64,
         "<r>但这 5.8% 不是 AOT 测出来的。</r>"
         "<b>AOT 只说了一句「13 装得下、14 装不下」</b>，"
         "<g>把真机额度省到该花的地方 —— 性能数字全部来自真机。</g>", "xs")


# ══════════════════════════════════════════════════════════════════════
def _pair(f):
    f.rect(BX, R1_Y, BW, R1_H, FILL[YL], YL, 1.4, 10)
    f.t(BX + 18, R1_Y + 26, "乙　·　它给不了的那一格", "sec")
    para(f, BX + 18, R1_Y + 46, BW - 36,
         "<g>同一组配置，pdbs 从 12 换到 13，两个指标<b>方向相反</b>。</g>", "xs", 16)

    y = R1_Y + 74
    for name, unit, v12, v13, arrow, c in PAIR:
        f.rect(BX + 18, y, BW - 36, 78, "#fff", c, 1.2, 8)
        f.t(BX + 32, y + 22, name, "lbl", INK)
        f.t(BX + 32, y + 38, unit, "xxs", SUB)
        f.t(BX + 40, y + 64, "pdbs 12", "xxs", SUB)
        f.t(BX + 120, y + 66, v12, "numb", INK)
        f.t(BX + 190, y + 64, "→", "xxs", SUB)
        f.t(BX + 214, y + 64, "pdbs 13", "xxs", SUB)
        f.t(BX + 294, y + 66, v13, "numb", INK)
        f.t(BX + BW - 32, y + 66, arrow, "numb", c, "end")
        y += 88

    f.rect(BX + 18, R1_Y + 286, BW - 36, 40, "#fff", INK, 1.2, 8)
    para(f, BX + 32, R1_Y + 308, BW - 64,
         "<b>两个都没错。</b><g>temp 是判 OOM 的那一项，"
         "<b>从来不是「谁更省显存」的排名。</b></g>", "xs")


# ══════════════════════════════════════════════════════════════════════
def _band(f):
    f.rect(20, BAND_Y, 1360, 92, FILL[BL], BL, 1.4, 10)
    f.t(38, BAND_Y + 28, "这两格必须一起看", "sec")
    para(f, 38, BAND_Y + 54, 1324,
         "<b>只看左边，会以为「AOT 帮我们提了 5.8%」—— 它一个性能数字都给不了。"
         "只看右边，又像在说这工具不可靠。</b>"
         "<r>合起来才是真话：它回答的是「装不装得下」，"
         "而它的价值是把有限的真机额度省到该花的地方。</r>"
         "<g>用一个指标去回答它没被设计来回答的问题 —— "
         "这是这门课里反复出现的同一种错。</g>", "xs", 19)


if __name__ == "__main__":
    import sys
    open(sys.argv[1] if len(sys.argv) > 1 else "out/fig_p11_cashin.svg",
         "w", encoding="utf-8").write(build())
