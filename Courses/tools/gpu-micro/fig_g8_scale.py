# -*- coding: utf-8 -*-
"""图 G-8 —— 压轴：同一把尺子上，两颗芯片长什么样。

两个方块的**面积**按每周期乘加单元总数等比画，内部按**真实单元数**切分。
于是一眼能看到全文最反直觉的那件事：总量几乎一样（1.16×），
切分粒度差 64 倍 —— 一边 592 个小格，一边 8 个大格。

数字来源：
  B200  148 SM × 4 Tensor Core = 592；每 Tensor Core 1,024 乘加/周期
        （A100 官方白皮书给 256、Hopper 官方说翻倍、B200 有第三方旁证，见正文 §2）
  v7    每 chip 合计 524,288 乘加/周期 —— 这个**总量**由公开峰值定死，
        而它的**拆法**（几个 MXU × 每个多少）公开资料对不上，见正文 §8。
        本图按 v5e/v6e 的同构读法画成 8 个 256×256 的 MXU。

**这里踩过一个坑，记下来**：本文第一版写的是「4 个 MXU、每 cell 双发」。
那个说法能凑出同样的总量，但用同一套口径去算 v5e / v6e 会算错一倍 ——
而这两代的 MXU 数、阵列边长、峰值、时钟全是公开的，是现成的验尸台。
教训：一个只能解释目标、不能解释同族已知样本的模型，多半是错的。
"""
from common import Fig, para, BL, RD, YL, GN, PU, TL, INK, SUB, GREY, FILL

W = 1400
TOP = 84

GPU_MAC, TPU_MAC = 606_208, 524_288
GPU_N, TPU_N = 592, 8
GPU_PER, TPU_PER = 1_024, 65_536

SIDE_G = 300.0
SC = SIDE_G / GPU_MAC ** 0.5                 # 面积等比：px per √MAC
SIDE_T = TPU_MAC ** 0.5 * SC                 # ≈ 279

BOX_T = TOP + 92      # 给标题和放大镜标注留出上方空间
GX, GY = 60, BOX_T
TX, TY = 430, BOX_T + (SIDE_G - SIDE_T) / 2
RX = 790                                     # 右侧数据栏
BOX_B = BOX_T + SIDE_G

CARD_H = 152
C_T = BOX_B + 186         # 右侧「把账摊开」面板比左边两块高，section 要让到它下面
H = C_T + 14 + CARD_H + 14 + 76 + 18


def build():
    f = Fig(W, H, "B200 与 TPU v7 每周期乘加单元总数的等面积对照："
                  "总量相差 1.16 倍，但 B200 切成 592 个 Tensor Core，"
                  "TPU v7 只切成 8 个 MXU，单元粒度相差 64 倍")
    f.title("同一把尺子上　—— 两个方块面积等比，<tspan fill=\"#5f6368\">总量几乎一样</tspan>，"
            "<tspan fill=\"#1a73e8\">切分粒度差 64 倍</tspan>", "本图为全文结论")
    f.legend([(BL, "B200：一个 Tensor Core"), (GN, "TPU v7：一个 MXU"),
              (GREY, "推导值 —— 推导链与它的验尸见正文 §2 / §8")])

    _blocks(f)
    _table(f)
    _read(f)
    return f.out()


# ══════════════════════════════════════════════════════════════════════
def _blocks(f):
    # ── B200：592 个小格 ────────────────────────────────────────────
    f.t(GX, BOX_T - 66, "NVIDIA B200　一整颗", "sec", BL)
    f.t(GX, BOX_T - 50, "148 个 SM × 每 SM 4 个 Tensor Core", "xs")
    f.rect(GX - 6, GY - 6, SIDE_G + 12, SIDE_G + 12, "#fff", BL, 2, 6)

    cols, rows = 25, 24
    cw, ch = SIDE_G / cols, SIDE_G / rows
    for i in range(GPU_N):
        r, c = divmod(i, cols)
        f.rect(GX + c * cw, GY + r * ch, cw - 1.6, ch - 1.6, FILL[BL], BL, 0.5, 1)
    f.t(GX, BOX_T + SIDE_G + 20, f"{GPU_N} 个 Tensor Core", "box", BL)
    f.t(GX, BOX_T + SIDE_G + 38,
        f"× 每个 <tspan class=\"num\" fill=\"#1a73e8\">{GPU_PER:,}</tspan> 乘加 / 周期"
        f"　＝　<tspan class=\"numb\" fill=\"#202124\">{GPU_MAC:,}</tspan>", "xs")

    # 放大镜：把左上角那一格拉出来看
    ex, ey = GX + 3 * cw, GY - 44
    f.rect(GX, GY, cw - 1.6, ch - 1.6, "none", RD, 1.8, 1)
    f.path(f"M{GX + cw} {GY} L{ex} {ey + 22}", RD, 1.2, dash="3,2")
    f.rect(ex, ey, 214, 30, "#fff", RD, 1.4, 6)
    f.t(ex + 9, ey + 13, "1 个 Tensor Core", "lbl", RD)
    f.t(ex + 9, ey + 25, "1,024 乘加 / 周期　（推导值）", "xxs")

    # ── 中间：面积比 ────────────────────────────────────────────────
    mx = (GX + SIDE_G + TX) / 2
    f.t(mx, BOX_T + SIDE_G / 2 - 8, "面积比", "xs", None, "middle")
    f.t(mx, BOX_T + SIDE_G / 2 + 12, "1.16×", "numb", SUB, "middle")

    # ── TPU v7：8 个大格（4 列 × 2 行，每行 = 一个 TensorCore）──────
    f.t(TX, BOX_T - 66, "TPU v7　一整颗 chip", "sec", GN)
    f.t(TX, BOX_T - 50, "2 个 TensorCore × 每核 4 个 MXU"
                        "<tspan fill=\"#9aa0a6\">（拆法为推导，见右下）</tspan>", "xs")
    f.rect(TX - 6, TY - 6, SIDE_T + 12, SIDE_T + 12, "#fff", GN, 2, 6)

    # 左边 592 个格子自带纹理，右边八个大盒子如果留白，会被读成「里面是空的」。
    # 真按 256×256 画每个 cell 只有 0.27 像素，画不出来 —— 所以铺网点表示密度。
    f.raw('<defs><pattern id="mxucell" width="3" height="3" patternUnits="userSpaceOnUse">'
          f'<rect width="3" height="3" fill="{FILL[GN]}"/>'
          f'<rect width="1.6" height="1.6" x="0.2" y="0.2" fill="{GN}" opacity="0.34"/>'
          '</pattern></defs>')

    cw_, ch_ = SIDE_T / 4, SIDE_T / 2
    for i in range(TPU_N):
        r, c = divmod(i, 4)
        x, y = TX + c * cw_, TY + r * ch_
        f.rect(x, y, cw_ - 3, ch_ - 3, "url(#mxucell)", GN, 1.4, 3)
        mx, my = x + (cw_ - 3) / 2, y + ch_ / 2
        f.rect(x + 4, my - 34, cw_ - 11, 64, "#fff", GN, 0.8, 4, extra=' opacity="0.93"')
        f.t(mx, my - 20, "MXU", "lbl", GN, "middle")
        f.t(mx, my - 5, "256×256", "xs", INK, "middle")
        f.t(mx, my + 12, "65,536", "num", GN, "middle")
        f.t(mx, my + 24, "乘加/周期", "xxs", None, "middle")
    f.t(TX, BOX_T + SIDE_G + 20, f"{TPU_N} 个 MXU", "box", GN)
    f.t(TX, BOX_T + SIDE_G + 56,
        "<tspan fill=\"#9aa0a6\">网点只示意密度：真按 256×256 画，每 cell 只有 0.27 像素</tspan>", "xxs")
    f.t(TX, BOX_T + SIDE_G + 38,
        f"× 每个 <tspan class=\"num\" fill=\"#1e8e3e\">{TPU_PER:,}</tspan> 乘加 / 周期"
        f"　＝　<tspan class=\"numb\" fill=\"#202124\">{TPU_MAC:,}</tspan>", "xs")


# ══════════════════════════════════════════════════════════════════════
def _table(f):
    x, w = RX, W - 20 - RX
    f.rect(x, BOX_T - 34, w, SIDE_G + 190, "#fff", INK, 1.6, 10)
    f.t(x + 16, BOX_T - 14, "把账摊开", "sec")

    rows = [
        ("每周期乘加单元总数", f"{GPU_MAC:,}", f"{TPU_MAC:,}", "1.16 ×", SUB, False),
        ("切成多少个独立单元", f"{GPU_N} 个 Tensor Core", f"{TPU_N} 个 MXU",
         "74 ×", SUB, False),
        ("单个单元多大", f"{GPU_PER:,} 乘加/周期", f"{TPU_PER:,} 乘加/周期",
         "64 ×", BL, True),
        ("↑ 两行为什么不等", "个数差 74 × ÷ 总量差 1.16 ×", "＝ 单元大小差 64 ×",
         "自洽", GN, False),
        ("公布的峰值", "2,250 TFLOPS  FP16", "2,307 TFLOPS  BF16", "1.03 ×", GN, False),
        ("这些数怎么来的", "148×4 官方；每核 1,024 官方 + 推导",
         "总量由峰值定死；<b>拆成几个 MXU 是推导</b>", "见下", GREY, False),
    ]
    cw = [156, 150, 150, 92]
    ty = BOX_T + 4
    hdr = ["", "B200", "TPU v7 chip", "倍数"]
    cx = x + 16
    for i, hh in enumerate(hdr):
        f.t(cx, ty, hh, "lbl", INK)
        cx += cw[i]

    for r, (k, a, b, mult, col, hi) in enumerate(rows):
        ry = ty + 22 + r * 52
        if hi:
            f.rect(x + 10, ry - 16, w - 20, 46, FILL[BL], rx=6)
        f.line(x + 10, ry - 20, x + w - 10, ry - 20, "#e8eaed", 1)
        cx = x + 16
        para(f, cx, ry, cw[0] - 12, k, "xs", 13)
        cx += cw[0]
        para(f, cx, ry, cw[1] - 12, a, "lbl" if hi else "xs", 14)
        cx += cw[1]
        para(f, cx, ry, cw[2] - 12, b, "lbl" if hi else "xs", 14)
        cx += cw[2]
        f.t(cx, ry, mult, "numb" if hi else "num", col)

    y = BOX_T + 22 + 6 * 52 + 6
    f.rect(x + 10, y, w - 20, 96, FILL[GN], GN, 1.4, 8)
    yy = para(f, x + 22, y + 18, w - 44,
              "<b>这套口径是在别的芯片上验过的。</b>同样用「峰值 ÷ 2 ÷ 总 cell 数 ＝ 时钟」，"
              "拿两代<r>阵列边长、MXU 个数、峰值、时钟全部公开</r>的芯片当验尸台：", "xs", 14)
    yy = para(f, x + 22, yy + 4, w - 44,
              "　v5e　4 MXU × 128×128 @ 1.5 GHz → 196.6 TF（官方 197，差 0.2%）", "xs", 13)
    yy = para(f, x + 22, yy + 1, w - 44,
              "　v6e　4 MXU × 256×256 @ 1.75 GHz → 917.5 TF（官方 918，差 0.05%）", "xs", 13)
    para(f, x + 22, yy + 4, w - 44,
         "两代都还原成功，且都是<b>每 cell 一次乘加</b>。把同一套口径用到 v7 的 2,307 TF 上，"
         "总量落在 524,288 —— 这个数是确定的；它拆成几个 MXU 不确定。", "xs", 13)


# ══════════════════════════════════════════════════════════════════════
def _read(f):
    f.t(20, C_T, "这 64 倍意味着什么　—— 三个方向，没有哪边天然更好", "sec")
    cards = [
        (BL, "GPU 为什么必须切碎", "细粒度买来的是「什么都能跑」",
         "一颗 B200 上同时挂着许多互不相干的线程块，谁先算完谁先走。"
         "要让通用调度器管得住，单元就得小到<b>一个 warp 就能独占一个</b>。"
         "代价是这 592 份里每一份都要自带操作数通路、累加器和控制逻辑 —— "
         "<r>控制的开销乘了 592 遍</r>。"),
        (GN, "TPU 为什么敢切粗", "粗粒度买来的是「几乎没有开销」",
         "一个 MXU 一条指令就吃满 65,536 个乘加单元，"
         "这意味着取指、译码、控制这些开销<b>被摊到六万多个乘法上，趋近于零</b>。"
         "但它只有在「确实有这么大一块矩阵要算」时才成立 —— "
         "而这件事得由<r>编译器在编译期保证</r>，做不到就空转。"),
        (RD, "所以什么形状喂得满", "回到图 G-4 那个例子",
         "GPU 的收缩维 K 一直是 16，几乎什么形状都是它的整数倍，喂满很容易。"
         "TPU 的 MXU 收缩边是 256 —— 注意力里常见的 <code>head_dim=128</code> "
         "只能喂满一半。<b>同一个模型配置，在两边的「浪费」完全不在同一个位置</b>，"
         "这就是为什么调优经验不能直接搬。"),
    ]
    cw = (1360 - 2 * 14) / 3
    for i, (c, ttl, sub, body) in enumerate(cards):
        x = 20 + i * (cw + 14)
        f.rect(x, C_T + 14, cw, CARD_H, "#fff", c, 1.8, 10)
        f.rect(x, C_T + 14, cw, 48, FILL[c], rx=10)
        f.rect(x, C_T + 53, cw, 9, FILL[c], rx=0)
        f.t(x + 14, C_T + 34, ttl, "box", c)
        f.t(x + 14, C_T + 51, sub, "xs")
        para(f, x + 14, C_T + 82, cw - 28, body, "xs", 16)

    # 通栏：这是「怎么读这张图」，不是第四个并列的论点
    by = C_T + 14 + CARD_H + 14
    f.rect(20, by, 1360, 76, FILL[PU], PU, 1.8, 10)
    f.rect(20, by, 7, 76, PU, rx=3)
    f.t(40, by + 22, "别把这张图读成排名　—— 它是一张取舍图，不是分数表", "box", PU)
    para(f, 40, by + 44, 1320,
         "总量接近、峰值接近，说明两家在同一代工艺上做出的<b>算力密度是可比的</b>。"
         "真正的分歧在前面几张图里：<b>谁来知道数在哪</b>（G-6）、<b>谁来盖住延迟</b>（G-7）、"
         "<b>切多细</b>（本图）。这三个选择互相咬合 —— <r>换掉任何一个，另外两个都得跟着换</r>："
         "切得细就必须有大寄存器堆去养替补，养了替补就不需要编译期排班，"
         "不排班就只能靠 cache 兜住不确定的延迟。反过来那条链同样成立。", "xs", 15)


if __name__ == "__main__":
    import io, sys
    io.open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/g8.svg", "w",
            encoding="utf-8").write(build())
    print("ok", H, "| SIDE_T", round(SIDE_T, 1))
