# -*- coding: utf-8 -*-
"""图 P-13 —— MFU 是一个分数，而分子分母都是选出来的。

P-12 只讲了分母。这一张把分子补上，然后回答一个更难受的问题：
**同一次 run、同一份日志，能合法地报出几个不同的 MFU？**

  甲　分子的三个选择　—— 都不会报错，最大的那个差 5 倍
  中　⭐ 一次 run 的四个 MFU 摆成 2×2　—— 只动分母，就从 10.1% 走到 22.4%
  乙　分母的四个选择　—— 每一个单独看都有官方依据

⚠️ **中间这一格是这张图的重心，两处设计是刻意的：**

**① 把四个数摆成矩阵，而不是画一个分数。** 第一版中间画的是
「分子 ÷ 分母」的示意分数 —— 图形上好看，但它只是把标题又说了一遍，
**没有任何一个真实数字**。现在摆的是同一次 run 真实能报出的四个百分比，
行是精度口径、列是 SKU，**读者可以自己沿着行或列走一遍，看着数字变**。

**② 这四个格子全都只在动分母。** 分子（甲那三个选择）一次都没动。
所以 2.2 倍这个跨度是**下限**不是上限 —— 这句话必须说出来，
否则台下会以为「四个数」就是全部的空间。

⚠️ **HFU / MFU 那一条返工过，原因值得记下来。**
第一版写「重算的 FLOP 算不算，约 1.09×」，举的例子是同一配置开/不开
selective recompute 的 527 → 480 —— **这是个范畴错误**。那两个数是
**两次 run 的吞吐差**（开了重算就是慢），不是同一次 run 的两种记账。
框架报的 TFLOP/s 里根本没有重算项（算完前向乘 3 就返回），
**所以它报的永远是 MFU 口径**。真正能演示 HFU/MFU 之差的是全量重算：
前向重做一遍 → HFU/MFU = 4/3 ≈ **1.33×**，而这个比值是**定义**推出来的，
不依赖任何一次实测。现在图里用的是后者。

数据出处：
  · 虚高 5 倍 —— 移植 Hunyuan3 时算力统计函数漏加模型名，本仓库移植指南有记录
  · 4/3 —— 全量重算下前向被算两次、反向一次，(1+1+2)/(1+2)
  · 503 TFLOP/s/GPU 与 22.4% / 11.2% —— A4X Megatron sweep 那张表的 A3 行及表注
  · 峰值口径（2,250 / 2,500 / 4,500 / 5,000）—— 见 P-12
"""
from common import Fig, para, BL, RD, GN, YL, PU, INK, SUB, LINE, GREY, FILL

W = 1400
TOP = 84

R1_Y = TOP + 40
R1_H = 372

AX, AW = 20, 424           # 甲：分子
MX, MW = 456, 452          # 中：一次 run 的四个 MFU
BX, BW = 932, 448          # 乙：分母

BAND_Y = R1_Y + R1_H + 28
H = BAND_Y + 148

# ── 甲：分子的三个选择 ──────────────────────────────────────────
NUM = [("算力公式漏了一个白名单", "约 5×", RD,
        "算力统计函数<b>按模型家族分叉</b>。漏加模型名时"
        "<b>它不报错</b> —— 只是用错误的宽度去量专家、"
        "并漏掉共享专家。<r>我们移植时中过这一枪。</r>"),
       ("重算的 FLOP 算不算", "4/3", YL,
        "算进去叫 HFU，不算叫 MFU。<g>全量重算下前向被算两遍，"
        "两者相差 <b>4/3 ≈ 1.33×</b> —— 这是定义推出来的，不是测出来的。"
        "<b>主流框架报的都是 MFU 口径。</b></g>"),
       ("attention 平方项按 causal 折半吗", "看序列长度", YL,
        "<g>短序列上无所谓；128K 那一档它就是大头。</g>")]

# ── 乙：分母的四个选择 ──────────────────────────────────────────
DEN = [("dense 还是 sparse", "2×", RD,
        "官方脚注写着 <b>sparse 是 dense 的两倍</b>。"
        "<r>头条数字默认给 sparse。</r>"),
       ("哪一个 SKU", "1.11×", YL, "<g>见上一张图。</g>"),
       ("per chip 还是 per device", "2×", RD, "<g>见上一张图。</g>"),
       ("拿哪一种精度的峰值", "2×", RD,
        "<b>一次 FP8 的 run，用 BF16 峰值量还是 FP8 峰值量。</b>"
        "<g>两种都有人做，都说得通。</g>")]

# ── 中：2×2 —— 同一次 run（503 TFLOP/s/GPU），只换分母 ─────────
COLS = [("2,250 / 4,500", "HGX B200 的峰值", RD),
        ("2,500 / 5,000", "GB200 —— 实际跑的这台", GN)]
ROWS = [("÷ BF16 峰值", ["22.4%", "20.1%"], ["← 那份文档主报", ""]),
        ("÷ FP8 峰值", ["11.2%", "10.1%"], ["← 表注里又给了", ""])]


def build():
    f = Fig(W, H, "MFU 的分子与分母各有若干合法选择：同一次实测只换分母就能"
                  "合法报出 10.1% 到 22.4% 四个不同的 MFU，跨度 2.2 倍")
    f.title("MFU 是一个分数　—— <tspan font-weight=\"700\" fill=\"#d93025\">"
            "上下两头都是选择，而且都不会报错</tspan>")
    f.legend([(RD, "能把结果改一倍以上"), (YL, "改得少，但同样要交代"),
              (GN, "这一列才是这台机器该用的")])

    _num(f)
    _grid(f)
    _den(f)
    _band(f)
    f.t(W - 20, H - 12,
        "实测：A4X（GB200 NVL72）2 节点 8 GPU · Qwen3 类 MoE · seq 16384 · "
        "FP8 训练，实测 503 TFLOP/s/GPU · 峰值口径见上一张图",
        "xxs", GREY, "end")
    return f.out()


# ══════════════════════════════════════════════════════════════════════
def _num(f):
    f.rect(AX, R1_Y, AW, R1_H, "#fff", LINE, 1.2, 10)
    f.t(AX + 16, R1_Y + 26, "甲　·　分子：我算出来的那个数", "sec")
    para(f, AX + 16, R1_Y + 46, AW - 32,
         "<g>框架日志里那个 TFLOP/s <b>不是量出来的，是一条公式算出来的</b> —— "
         "而公式有版本。</g>", "xs", 16)

    y = R1_Y + 88
    for i, (name, fac, c, desc) in enumerate(NUM):
        h = (92, 92, 54)[i]
        f.rect(AX + 16, y, AW - 32, h, FILL[c], c, 1.2, 8)
        para(f, AX + 30, y + 22, AW - 130, "<b>%s</b>" % name, "xs", 15, c)
        f.t(AX + AW - 30, y + 24, fac, "num", c, "end")
        para(f, AX + 30, y + 44, AW - 60, desc, "xxs", 14)
        y += h + 10


# ══════════════════════════════════════════════════════════════════════
def _grid(f):
    """一次 run 的四个 MFU —— 四个格子全都只在动分母。"""
    f.rect(MX, R1_Y, MW, R1_H, FILL[BL], BL, 1.4, 10)
    f.t(MX + 16, R1_Y + 26, "中　·　同一次 run，四个 MFU", "sec")
    para(f, MX + 16, R1_Y + 46, MW - 32,
         "<b>实测 503 TFLOP/s/GPU，一行代码没改。</b>"
         "<g>下面四个格子只是换了除数。</g>", "xs", 16)

    LBLW, CELLW, GAP = 104, 158, 8
    gx = MX + 16
    c0 = gx + LBLW
    gw = LBLW + 2 * CELLW + GAP
    hy = R1_Y + 100                      # 列头基线

    for j, (peak, who, cc) in enumerate(COLS):
        cx = c0 + j * (CELLW + GAP)
        f.t(cx + CELLW / 2, hy, peak, "lbl", cc, "middle")
        f.t(cx + CELLW / 2, hy + 16, who, "xxs", SUB, "middle")
    f.line(gx, hy + 26, gx + gw, hy + 26, LINE, 1.0)

    y = hy + 34
    for lbl, vals, tags in ROWS:
        f.t(gx + 4, y + 34, lbl, "box", INK)
        for j, v in enumerate(vals):
            cc = COLS[j][2]
            cx = c0 + j * (CELLW + GAP)
            f.rect(cx, y, CELLW, 58, "#fff", cc, 1.6 if cc == GN else 1.2, 8)
            f.t(cx + CELLW / 2, y + 32, v, "ttl", cc, "middle")
            if tags[j]:
                f.t(cx + CELLW / 2, y + 49, tags[j], "xxs", SUB, "middle")
        y += 66

    # 跨度条：把 2.2 倍画成一段长度，而不是只写一个倍数
    by = y + 18
    f.line(gx + 6, by, gx + gw - 6, by, INK, 2.0)
    f.line(gx + 6, by - 5, gx + 6, by + 5, INK, 2.0)
    f.line(gx + gw - 6, by - 5, gx + gw - 6, by + 5, INK, 2.0)
    f.t(gx + 6, by - 10, "10.1%", "num", INK)
    f.t(gx + gw - 6, by - 10, "22.4%", "num", RD, "end")
    f.t(MX + MW / 2, by + 20, "2.2 倍　·　而这四个格子只动了分母",
        "box", RD, "middle")

    f.rect(MX + 16, R1_Y + 314, MW - 32, 52, "#fff", INK, 1.2, 8)
    para(f, MX + 30, R1_Y + 334, MW - 60,
         "<b>甲那三个、乙那四个，一次都没动，而且能叠乘。</b>"
         "<r>2.2 倍是<b>下限</b>。</r>", "xs", 16)


# ══════════════════════════════════════════════════════════════════════
def _den(f):
    f.rect(BX, R1_Y, BW, R1_H, "#fff", LINE, 1.2, 10)
    f.t(BX + 16, R1_Y + 26, "乙　·　分母：我选的那个峰值", "sec")
    para(f, BX + 16, R1_Y + 46, BW - 32,
         "<g>四个都有官方出处。<b>难的不是找到一个，是说清用了哪一个。</b></g>",
         "xs", 16)

    y = R1_Y + 88
    for i, (name, fac, c, desc) in enumerate(DEN):
        h = (62, 46, 46, 62)[i]
        f.rect(BX + 16, y, BW - 32, h, FILL[c], c, 1.2, 8)
        para(f, BX + 30, y + 22, BW - 130, "<b>%s</b>" % name, "xs", 15, c)
        f.t(BX + BW - 30, y + 24, fac, "num", c, "end")
        para(f, BX + 30, y + 42, BW - 60, desc, "xxs", 13)
        y += h + 10

    f.t(BX + 16, y + 18,
        "↖ 中间那张表只演示了其中两条：「哪个 SKU」和「哪种精度」。",
        "xxs", SUB)


# ══════════════════════════════════════════════════════════════════════
def _band(f):
    f.rect(20, BAND_Y, 1360, 120, FILL[BL], BL, 1.4, 10)
    f.t(38, BAND_Y + 28,
        "⭐ 所以报 MFU 的时候，把这一整行一起报出来 —— 这是这一节唯一要带走的动作", "sec")
    y = BAND_Y + 54
    for seg in [
            "<b>MFU 20.1% ＝ 503 TFLOP/s ÷ 2,500（GB200 · BF16 · dense · per GPU）"
            "· 分子：框架内建公式，不含重算 · 公式版本＝当时那个 commit</b>"
            "<g>　—— 一行，抄走就能用。</g>",
            "<r>写不出这一行，就别报百分比。</r>"
            "<g>退一步报<b>吞吐本身</b>（tokens/s，或者「跑完一步要几秒」）—— "
            "那个数虽然也有口径，但至少<b>台下能直接感觉到快慢</b>，"
            "而百分比不行。<b>下一张讲的就是：什么时候该退这一步。</b></g>"]:
        y = para(f, 38, y, 1324, seg, "xs", 19) + 4


if __name__ == "__main__":
    import sys
    open(sys.argv[1] if len(sys.argv) > 1 else "out/fig_p13_mfu.svg",
         "w", encoding="utf-8").write(build())
