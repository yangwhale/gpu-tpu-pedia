# -*- coding: utf-8 -*-
"""图 P-12 —— 比之前先问分母：四个数全有出处，而它们互相不等。

第 6 节开头必须先立一条规矩，否则后面所有对比都是空的：
**MFU 是一个分数，而分母是选出来的。**

这张图只讲分母，分子留给 P-13。选这四个数是因为它们的错法各不相同：

  甲　同一代 GPU，两个 SKU　—— 差 11%，而两张卡都叫 Blackwell
  乙　同一颗 TPU，两个口径　—— 差 2 倍，而框架日志只按其中一个报

**为什么值得单独占一页**：这两个坑都不会报错。
选错分母算出来的 MFU 是一个完全正常的百分比，能进 PPT、能做容量规划，
**除非有人回头去查你除的是哪个数，否则永远不会被发现。**

⚠️ **这张图返工过一次，两处教训记在这里：**

**① 数字不能上色。** 第一版把 2,500 标红、2,250 标蓝，图例写「红＝容易被选错」。
但落点讲的错法是「A4X 那台机器用了 2,250」——&nbsp;**红色恰好标在正确答案上**。
更根本的问题是：这一格的论点就是「光看数字分辨不出该用哪个」，
**给数字上色本身就在暗示答案**。现在数字一律中性，颜色只留给**机型那一行**。

**② 差距要画出来，不能只写出来。** 第一版两张卡是纯文字，11% 只是一行字。
现在两条共轴的条，差的那一段单独描出来 ——&nbsp;**11% 从「读到」变成「看到」。**
同理乙格那两条，第二条正好半长。

⚠️ **1,153.5 不是官方数**，是 2,307 ÷ 2 —— 官方规格表只给 per chip 那一栏。
图里显式写了这一步除法，因为这一节最不该省的就是这种一步换算。

来源（全部公开可引）：
  · NVIDIA GB200 NVL72 规格表 —— Superchip 那栏 FP16/BF16 Tensor Core 10 PFLOPS，
    脚注 2「Specification in sparse. Dense is one-half sparse spec shown.」
  · NVIDIA HGX 平台规格表 —— HGX B200 八卡 FP16/BF16 36 PFLOPS，同一条脚注
  · TPU v7 峰值取自 Google Cloud 官方 TPU7x 规格表，与第 1 节同源
    （**不是** JAX 源码那张表 —— 那张给的是片上参数，且 bf16 那栏是 1,155/core）
"""
from common import Fig, para, BL, RD, GN, YL, PU, INK, SUB, LINE, GREY, FILL

W = 1400
TOP = 84

R1_Y = TOP + 40
R1_H = 398

AX, AW = 20, 836
BX, BW = 868, 512

BARX, BARW = 300, 372         # 甲：共轴条的起点与满程（对应 2,500）
GMAX = 2500.0

BBARX, BBARW = 176, 262       # 乙：同样一条轴，满程对应 2,307
TMAX = 2307.0

BAND_Y = R1_Y + R1_H + 28
H = BAND_Y + 128

# ── 甲：同一代 GPU 的两个 SKU ────────────────────────────────────
# 字段逐项对齐 —— 差异必须靠读「机型」那一行读出来，不能靠数字的颜色暗示。
SKU = [("GB200 的 GPU", "机型 a4x-highgpu-4g（A4X）", 2500,
        "官方 Superchip 栏 10 PFLOPS(sparse) ÷ 2 dense ÷ 2 GPU",
        "交叉验证：整机柜 360 PFLOPS(sparse) ÷ 72 ÷ 2 = 2.5 ✓"),
       ("HGX B200", "机型 a4-highgpu-8g（A4 High）", 2250,
        "官方八卡栏 36 PFLOPS(sparse) ÷ 2 dense ÷ 8 GPU",
        "交叉验证：同页 FP8 72 ÷ 2 ÷ 8 = 4.5，与 BF16 正好 2:1 ✓")]

# ── 乙：同一颗 TPU 的两个口径 ────────────────────────────────────
SCOPE = [("per chip", 2307, "2,307", "官方规格表只给这一栏"),
         ("per device", 1153.5, "1,153.5", "<b>= 2,307 ÷ 2　框架日志只按这个报</b>")]


def build():
    f = Fig(W, H, "MFU 的分母是选出来的：GB200 与 HGX B200 两个官方 BF16 峰值差 11%，"
                  "TPU v7 的 per-chip 与 per-device 两个口径差 2 倍")
    f.title("比之前先问分母　—— <tspan font-weight=\"700\" fill=\"#d93025\">"
            "这四个数全有出处，而它们互相不等</tspan>")
    f.legend([(INK, "数字一律中性：光看数字分辨不出该用哪个"),
              (RD, "决定该用哪个的是这一行")])

    _sku(f)
    _scope(f)
    _band(f)
    f.t(W - 20, H - 12,
        "来源：NVIDIA GB200 NVL72 与 HGX 平台官方规格表（均按其脚注换算成 dense）· "
        "TPU v7 取自 Google Cloud 官方 TPU7x 规格表，与第 1 节同源",
        "xxs", GREY, "end")
    return f.out()


# ══════════════════════════════════════════════════════════════════════
def _sku(f):
    f.rect(AX, R1_Y, AW, R1_H, "#fff", LINE, 1.2, 10)
    f.t(AX + 18, R1_Y + 26, "甲　·　同一代 GPU，两个 SKU", "sec")
    para(f, AX + 18, R1_Y + 46, 520,
         "<g>两张卡同一代、同一个架构名，官方峰值差 <b>11%</b>。"
         "<b>画在同一根轴上，差的那一段单独描出来。</b></g>", "xs", 16)
    f.t(AX + AW - 18, R1_Y + 50, "横轴：TFLOP/s（BF16 dense）　满程 = 2,500",
        "xxs", SUB, "end")

    y = R1_Y + 96
    for name, mt, val, chain, cross in SKU:
        f.t(AX + 30, y + 16, name, "box", INK)
        f.t(AX + 30, y + 35, mt, "xxs", RD)            # ← 唯一上色的一行

        bw = BARW * val / GMAX
        f.rect(AX + BARX, y, BARW, 28, "#fff", LINE, 1.0, 4, "4 3")   # 满程底槽
        f.rect(AX + BARX, y, bw, 28, LINE, GREY, 1.2, 4)
        f.t(AX + BARX + bw + 12, y + 20, format(val, ","), "numb", INK)

        para(f, AX + BARX, y + 48, BARW + 130, "<b>▸</b>　" + chain, "xxs", 13)
        para(f, AX + BARX, y + 63, BARW + 130, "<g>✓　%s</g>" % cross, "xxs", 13)
        y += 106

    # 差的那 11%：把 2,250 到 2,500 之间那一段单独描出来
    x0 = AX + BARX + BARW * 2250 / GMAX
    x1 = AX + BARX + BARW
    f.rect(x0, R1_Y + 96, x1 - x0, 28, FILL[RD], RD, 1.6, 0)
    f.line(x0, R1_Y + 88, x1, R1_Y + 88, RD, 1.4)
    f.line(x0, R1_Y + 88, x0, R1_Y + 230, RD, 1.0, None, "3 3")
    f.t((x0 + x1) / 2, R1_Y + 82, "11%", "num", RD, "middle")

    f.rect(AX + 18, R1_Y + 300, AW - 36, 82, "#fff", INK, 1.2, 8)
    yy = para(f, AX + 32, R1_Y + 322, AW - 64,
              "<b>2,500 ÷ 2,250 = 1.111。</b>"
              "<r>这 11% 会一比一地转移到 MFU 上</r> —— "
              "<g>硬件没变、代码没变、跑出来的 TFLOP/s 没变，只是除数换了一个。</g>",
              "xs", 17)
    para(f, AX + 32, yy + 4, AW - 64,
         "<b>而它不会报错。</b><g>选错分母算出来的仍然是一个完全正常的百分比 —— "
         "能进汇报、能做容量规划，<b>除非有人回头去查你除的是哪个数。</b></g>", "xs", 17)


# ══════════════════════════════════════════════════════════════════════
def _scope(f):
    f.rect(BX, R1_Y, BW, R1_H, FILL[YL], YL, 1.4, 10)
    f.t(BX + 18, R1_Y + 26, "乙　·　同一颗 TPU，两个口径", "sec")
    para(f, BX + 18, R1_Y + 46, BW - 36,
         "<g>这一边不是两个 SKU，是<b>同一颗芯片</b>。v7 一颗芯片里有 "
         "<b>2 个 device</b> —— 第二条正好半长。</g>", "xs", 16)
    f.t(BX + BW - 18, R1_Y + 84, "横轴同上　满程 = 2,307", "xxs", SUB, "end")

    y = R1_Y + 96
    for name, val, txt, note in SCOPE:
        f.t(BX + 30, y + 20, name, "box", INK)
        bw = BBARW * val / TMAX
        f.rect(BX + BBARX, y + 4, BBARW, 24, "#fff", LINE, 1.0, 4, "4 3")
        f.rect(BX + BBARX, y + 4, bw, 24, LINE, GREY, 1.2, 4)
        f.t(BX + BBARX + bw + 10, y + 22, txt, "numb", INK)
        para(f, BX + 30, y + 44, BW - 60, note, "xxs", 13)
        y += 64

    f.rect(BX + 18, R1_Y + 228, BW - 36, 62, "#fff", INK, 1.2, 8)
    para(f, BX + 32, R1_Y + 250, BW - 64,
         "<b>上一代是 1 chip = 1 device，这一代是 1 : 2。</b>"
         "<r>跨代照抄公式，分母就差一倍，而且不报错。</r>"
         "<g>第 0 节那张瀑布图问过同一个问题 —— 那次量的是 HBM，这次是算力。</g>",
         "xs", 16)

    f.rect(BX + 18, R1_Y + 300, BW - 36, 80, "#fff", RD, 1.2, 8)
    yy = para(f, BX + 32, R1_Y + 322, BW - 64,
              "<b>⚠️ 还有第三件事要交代：2,307 是最高频率档下的峰值。</b>", "xs", 16)
    para(f, BX + 32, yy + 2, BW - 64,
         "<g>我们自己的 v7 记录里，只把频率档从默认锁到最高，"
         "<b>什么都没改就白拿 8.6%</b> —— 说明默认档并不跑在那个频率上。"
         "<b>分母写 2,307 的时候，顺带说一句跑的是哪一档。</b></g>", "xs", 16)


# ══════════════════════════════════════════════════════════════════════
def _band(f):
    f.rect(20, BAND_Y, 1360, 100, FILL[YL], YL, 1.4, 10)
    f.t(38, BAND_Y + 28,
        "⭐ 这个坑我们自己就在里面 —— 而且是同一个数字，一次对一次错", "sec")
    y = BAND_Y + 54
    para(f, 38, y, 1324,
         "<b>本仓库有两份文档都写着「BF16 峰值 2,250 TFLOP/s」。</b>"
         "<g>一份跑在 A4 High（HGX B200）上 —— <b>对的</b>；</g>"
         "<r>另一份跑在 A4X（GB200 NVL72）上 —— 那台机器的官方峰值是 2,500，"
         "于是<b>那一页三张表里的每一个 MFU 都偏高约 11%</b>（最佳配置 23.4% 应为 21.1%）。</r>"
         "<b>两份文档写的是同一个数，差别不在数字里，在跑它的那台机器上。</b>"
         "　—— <b>而这还只是分母。下一张把分子补上。</b>", "xs", 19)


if __name__ == "__main__":
    import sys
    open(sys.argv[1] if len(sys.argv) > 1 else "out/fig_p12_denominator.svg",
         "w", encoding="utf-8").write(build())
