# -*- coding: utf-8 -*-
"""图 P-38 —— 先把问题摆出来：不分块的注意力，光在路上就要搬 128 GiB。

**3.6 原本问的是「一个数从 HBM 出发要停几站」。那个问法太抽象了** ——
学员听完知道了站名，但不知道这些站为什么值得记。<b>换成 FlashAttention
就完全不一样</b>：它是过去五年最重要的一个 kernel，而它做的事<b>从头到尾
只跟这条路有关，跟计算单元一点关系都没有</b>。

**这张图只负责把问题钉死。** 一个头、一个样本、序列 128K、<code>head_dim</code>
128、bf16：注意力矩阵 <b>S ＝ 131,072² × 2 B ＝ 32 GiB</b>。朴素写法要
<b>写 S、读 S、写 P、读 P</b> 四趟，<b>128 GiB 的 HBM 往返</b>；
而 Q/K/V/O 加起来只有 <b>128 MiB</b>。<b>差三个数量级，而且全在路上。</b>

**FlashAttention 的全部内容就是把这 128 GiB 那一项删掉。** 靠 online softmax
边算边更新，S 永远不落 HBM。<b>FLOPs 一个都没省 —— 省的全是搬运。</b>

**⚠️ 出处口径。** 32 GiB / 128 GiB / 128 MiB 三个数都是<b>当场算的</b>，
式子写在图上。「FLOPs 不变、HBM 读写从 O(n²) 降到 O(n²/block)、显存占用
O(n²)→O(n)」出自 FlashAttention 的公开结论。
<b>分块之后 K/V 要重复读，这一项图上如实标了，没有假装它是零。</b>
"""
from common import Fig, para, BL, GN, RD, YL, PU, TL, INK, SUB, GREY, FILL

W = 1400

NUM_Y, NUM_H = 84, 112                      # ① 那个 32 GiB
FLOW_Y, FLOW_H = NUM_Y + NUM_H + 22, 268    # ② 两种走法并排
BILL_Y, BILL_H = FLOW_Y + FLOW_H + 22, 118  # ③ 账
LAND_Y, LAND_H = BILL_Y + BILL_H + 22, 112  # ④ 落点
SRC_Y, SRC_H = LAND_Y + LAND_H + 22, 88     # ⑤ 出处
H = SRC_Y + SRC_H + 20

L_X, L_W = 20, 674
R_X, R_W = 706, 674


def build():
    f = Fig(W, H, "序列 128K 时注意力矩阵是 32 GiB，朴素写法要四趟 HBM 往返共 128 GiB；"
                  "FlashAttention 用 online softmax 让它永远不落 HBM，"
                  "计算量一点没变")
    f.title("先看这一节要解决的到底是什么　—— "
            "<tspan font-weight=\"700\">128 GiB 全花在路上，一次乘法都没做</tspan>",
            "3.6 的引子")
    f.legend([(RD, "HBM 往返：真正的开销"), (GN, "留在片上：省下来的"),
              (SUB, "Q / K / V / O：无论如何都要搬的")])
    _num(f)
    _flow(f)
    _bill(f)
    _land(f)
    _src(f)
    return f.out()


# ══════════════════════════════════════════════════════════════════════
def _num(f):
    f.rect(20, NUM_Y, 1360, NUM_H, FILL[RD], RD, 1.8, 10)
    f.t(38, NUM_Y + 28, "一个头、一个样本、序列 128K、head_dim 128、bf16", "sec", RD)
    xs = [(38, "131,072", "序列长度 n"), (250, "×", ""), (290, "131,072", "n"),
          (500, "×", ""), (540, "2 B", "bf16"), (660, "＝", ""),
          (710, "32 GiB", "注意力矩阵 S 一个人就这么大")]
    for x, big, sub in xs:
        f.t(x, NUM_Y + 68, big, "numb", INK if big not in ("×", "＝") else SUB)
        if sub:
            f.t(x, NUM_Y + 88, sub, "xxs", SUB)
    f.rect(940, NUM_Y + 40, 424, 56, "#fff", RD, 1.4, 6)
    para(f, 954, NUM_Y + 62, 396,
         "<b>而 Q / K / V / O 四个加起来只有 128 MiB。</b>"
         "<r>中间产物比输入输出大 256 倍。</r>", "xs", 16, max_lines=2)


# ══════════════════════════════════════════════════════════════════════
# 两条路并排。左边每一步都要过 HBM，右边整段留在片上 —— 画面本身就是结论。
def _flow(f):
    _naive(f)
    _flash(f)


def _naive(f):
    f.rect(L_X, FLOW_Y, L_W, FLOW_H, FILL[RD], RD, 1.8, 10)
    f.t(L_X + 18, FLOW_Y + 28, "❌ 朴素写法：S 要在 HBM 里过四趟", "sec", RD)
    f.t(L_X + 18, FLOW_Y + 50, "每一步都是「算完写回去，下一步再读出来」", "xxs", SUB)

    steps = [("① 算 S ＝ QKᵀ", "写 S", "32 GiB"),
             ("② 算 softmax", "读 S", "32 GiB"),
             ("", "写 P", "32 GiB"),
             ("③ 算 O ＝ PV", "读 P", "32 GiB")]
    for i, (act, io, amt) in enumerate(steps):
        y = FLOW_Y + 76 + i * 40
        f.rect(L_X + 18, y, 250, 32, "#fff", RD, 1.3, 5)
        f.t(L_X + 30, y + 21, act or "　", "xs", INK)
        f.line(L_X + 272, y + 16, L_X + 316, y + 16, RD, 2.0, marker="aB")
        f.rect(L_X + 320, y, 150, 32, FILL[RD], RD, 1.3, 5)
        f.t(L_X + 332, y + 21, io + "　→　HBM", "xs", RD)
        f.t(L_X + 490, y + 21, amt, "numb", RD)

    f.rect(L_X + 18, FLOW_Y + 240, L_W - 36, 18, "#fff", RD, 1.2, 4)
    f.t(L_X + 30, FLOW_Y + 254, "合计 128 GiB 的 HBM 往返 —— "
                                "这一大笔里，一次乘法都没有", "xs", RD)


def _flash(f):
    f.rect(R_X, FLOW_Y, R_W, FLOW_H, FILL[GN], GN, 1.8, 10)
    f.t(R_X + 18, FLOW_Y + 28, "✅ FlashAttention：S 从来不落 HBM", "sec", GN)
    f.t(R_X + 18, FLOW_Y + 50, "分块 ＋ online softmax，一块算完就地更新结果", "xxs", SUB)

    # 片上那个框：三步全在里面
    f.rect(R_X + 18, FLOW_Y + 72, 420, 148, "#fff", GN, 2.2, 8, "5,4")
    f.t(R_X + 32, FLOW_Y + 94, "片上暂存（共享内存 ／ VMEM）", "box", GN)
    for i, txt in enumerate(["① 取一块 K/V ＋ 一块 Q",
                             "② 算这一小块的 S，就地做 online softmax",
                             "③ 用跑动的最大值和分母，就地更新 O"]):
        f.rect(R_X + 32, FLOW_Y + 106 + i * 36, 392, 30, FILL[GN], GN, 1.2, 5)
        f.t(R_X + 44, FLOW_Y + 126 + i * 36, txt, "xs", INK)

    f.line(R_X + 228, FLOW_Y + 220, R_X + 228, FLOW_Y + 236, GN, 2.0, marker="aB")
    f.t(R_X + 240, FLOW_Y + 234, "循环下一块", "xxs", GN)

    f.rect(R_X + 452, FLOW_Y + 72, 204, 148, FILL[YL], YL, 1.5, 6)
    f.t(R_X + 464, FLOW_Y + 94, "代价要说清楚", "box", YL)
    para(f, R_X + 464, FLOW_Y + 112, 184,
         "<b>K/V 要按 Q 的分块重复读若干趟</b>，这一项不是零。"
         "<g>所以收益取决于「S 那一项原本占多大」—— 序列越长越划算，"
         "短序列可能不值。</g>", "xxs", 14, max_lines=8)

    f.rect(R_X + 18, FLOW_Y + 240, R_W - 36, 18, "#fff", GN, 1.2, 4)
    f.t(R_X + 30, FLOW_Y + 254, "S 那 128 GiB 整项消失 —— "
                                "而结果与朴素写法数学等价", "xs", GN)


# ══════════════════════════════════════════════════════════════════════
def _bill(f):
    f.rect(20, BILL_Y, 1360, BILL_H, FILL[BL], BL, 1.8, 10)
    f.t(38, BILL_Y + 28, "🔢 三笔账，只有一笔变了", "sec", BL)
    cols = [("计算量 FLOPs", "完全不变", GN, "一次乘法都没省"),
            ("HBM 读写", "O(n²) → O(n²/块大小)", RD, "省的全在这一行"),
            ("显存占用", "O(n²) → O(n)", RD, "32 GiB 的中间产物不存在了")]
    for i, (name, val, c, note) in enumerate(cols):
        x = 38 + i * 448
        f.t(x, BILL_Y + 58, name, "lbl", SUB)
        f.t(x, BILL_Y + 82, val, "box", c)
        f.t(x, BILL_Y + 102, note, "xxs", SUB)


# ══════════════════════════════════════════════════════════════════════
def _land(f):
    f.rect(20, LAND_Y, 1360, LAND_H, FILL[PU], PU, 1.8, 10)
    f.t(38, LAND_Y + 28, "⭐ 落点：这一节问的「一个数怎么从 HBM 走到计算单元」，"
                         "不是学术问题", "sec", PU)
    y = para(f, 38, LAND_Y + 54, 1324,
             "<b>过去五年最重要的那个 kernel，做的事跟计算单元一点关系都没有。</b>"
             "它没有换算法、没有减 FLOPs、没有用新指令 —— "
             "<r>它只是把数据在路上的走法改了一下。</r>", "xs", 19)
    para(f, 38, y + 2, 1324,
         "<b>所以接下来要一站一站走完这条路</b>，而且两边并排走："
         "<b>同一个 FlashAttention，在 GPU 上和在 TPU 上，"
         "每一站分别落在哪、由谁决定。</b>", "xs", 19)


# ══════════════════════════════════════════════════════════════════════
def _src(f):
    f.rect(20, SRC_Y, 1360, SRC_H, "#fff", GREY, 1.4, 10)
    f.t(38, SRC_Y + 26, "⚠️ 出处分层", "sec")
    y = para(f, 38, SRC_Y + 48, 1324,
             "<b>当场算的</b>：131,072² × 2 B ＝ 32 GiB；四趟 ＝ 128 GiB；"
             "Q/K/V/O ＝ 4 × 131,072 × 128 × 2 B ＝ 128 MiB。式子都写在图上，可以自己复核。",
             "xs", 17)
    para(f, 38, y + 2, 1324,
         "<b>公开结论</b>：FLOPs 不变、HBM 读写降到 O(n²/块大小)、显存 O(n²)→O(n) —— "
         "FlashAttention 原始结论。<g>「短序列可能不划算」也是公开说法，本图不给具体门槛。</g>",
         "xxs", 16)


if __name__ == "__main__":
    import io
    io.open("out/fig_p38_why_tiling.svg", "w", encoding="utf-8").write(build())
    print("ok fig_p38_why_tiling")
