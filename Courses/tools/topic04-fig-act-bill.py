# -*- coding: utf-8 -*-
r"""专题四 · §1.6「那一整条从头挂到尾的激活 ——&#160;以及峰值出现在哪一刻」

⭐⭐⭐ 2026-09-16 新画。这张图想送出的只有一个**画面**：
  前向一路往上堆，堆到 loss 那一刻最高，反向一路往下拆。
  **显存占用是一座山，而山顶在前向刚结束的时候。**

⭐⭐ 取舍：**横轴是时间，不是层号。**
  ⛔ 画成「第 1 层……第 61 层」会让人以为这是空间分布；
    画成时间轴，「什么时候最挤」这个问题才提得出来。
  ⭐ 判据：**想问「峰值在哪一刻」，横轴就必须是时刻。**

⭐ 这也是 §五那个「峰值出现在哪一刻」的提前埋点 ——&#160;
  到那一节只要把优化器状态那条水平带叠上来就行，山形不用重画。

⛔⛔ 刻意没画的：
  ① **优化器状态与权重。** 它们是**水平**的（不随时间变），
     画进来会把「山形」这个唯一要看的东西压扁。留给 §五。
  ② **具体每层多少 GiB。** 那是 §1.2 的表，图上只留形状和两个总数。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE, LINE2)

W = 1400

N_LAYER = 61                      # V3 的层数
# ⛔⛔ 2026-09-19 T04：峰值 ＝ 存档点 ＋ **当前正在重算的那一层**。
#   原来只算存档点（106.75），漏掉在算的那一层（71.14）——&#160;低估 67%。
#   ⭐ 这一讲自己在 §5.4 的小例子和 §2.2 图 Ⓒ 用的都是正确口径，只有这个头号数字没做。
ACT_RAW_TIB = 4.15                # 不开重算，一条 128K 序列
ACT_CKPT_GIB = 106.75             # 61 个存档点
ACT_INFLIGHT_GIB = 71.14          # 当前正在重算的那一层（MoE 块，最坏情况）
ACT_REMAT_GIB = ACT_CKPT_GIB + ACT_INFLIGHT_GIB       # ＝ 177.89
RATIO = ACT_RAW_TIB * 1024 / ACT_REMAT_GIB
assert abs(ACT_REMAT_GIB - 177.89) < 0.01
assert 23 < RATIO < 25            # 「约 24 倍」是算出来的，不是说顺口的


def main():
    f = Fig(W, "把显存占用按时间画出来，它是一座山："
               "前向从第一层到第六十一层一路堆高，"
               "堆到 loss 那一刻达到峰值，反向一层一层往回走才逐步释放。"
               "峰值不在训练的某个阶段，而在前向刚结束的那一瞬间")

    y0 = f.header(
        "激活是一座山　——　<tspan font-weight=\"700\">"
        "前向一路堆，山顶在 loss 那一刻</tspan>",
        "⭐ 横轴是<tspan font-weight=\"700\">时间</tspan>，不是层号"
        "　·　⛔ 权重和优化器状态没画 ——&#160;它们是<tspan font-weight=\"700\">"
        "水平的</tspan>，会把山形压扁",
        [(BL, "前向 · 堆"), (RD, "峰值"), (GR, "反向 · 拆")])

    # ══════════ Ⓐ 山形 ═══════════════════════════════════════════
    PH = 420
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 每前进一层，就多挂一份中间结果 ——　"
                 "<tspan font-weight=\"700\">而且一份都不能提前扔</tspan>", BL,
                 sub="⭐ 为什么不能扔：反向算权重梯度时"
                     "<tspan font-weight=\"700\">要用前向那一刻的输入</tspan>")

    X0, X1 = 130, 1320
    BOT, TOP = py + 320, py + 52
    MID = (X0 + X1) / 2.0

    f.line(X0, BOT, X1 + 10, BOT, GY2, 2.0, arrow=False)
    f.line(X0, BOT, X0, TOP - 14, GY2, 2.0, arrow=False)
    f.t(X0 - 14, TOP - 24, "显存里的激活", GY, True, 13, "end")

    # 台阶式的上升与下降 ——&#160;刻意画成阶梯，强调「一层加一份」
    N = 12
    for i in range(N):
        h = (BOT - TOP) * (i + 1) / float(N)
        w = (MID - X0) / N
        f.box(X0 + i * w, BOT - h, w, h, "#e8f0fe", BL, 0, 0.8)
    for i in range(N):
        h = (BOT - TOP) * (N - i) / float(N)
        w = (X1 - MID) / N
        f.box(MID + i * w, BOT - h, w, h, "#e6f4ea", GR, 0, 0.8)

    f.t((X0 + MID) / 2.0, BOT + 28, "前向：第 1 层 →　第 %d 层" % N_LAYER,
        BL, True, 15.5, "middle")
    f.t((MID + X1) / 2.0, BOT + 28, "反向：第 %d 层 →　第 1 层" % N_LAYER,
        GR, True, 15.5, "middle")
    f.t((X0 + MID) / 2.0, BOT + 52, "每过一层，多挂一份", GY, size=13, anchor="middle")
    f.t((MID + X1) / 2.0, BOT + 52, "每走回一层，释放一份", GY, size=13, anchor="middle")

    # 峰值
    f.line(MID, TOP - 8, MID, BOT, RD, 2.2, dash="5 4", arrow=False)
    f.box(MID - 168, TOP - 46, 336, 56, "#fce8e6", RD, 8)
    f.t(MID, TOP - 22, "⭐⭐ 峰值在这一刻", RD, True, 18, "middle")
    f.t(MID, TOP + 2, "前向刚算完、反向还没开始", GY, size=13, anchor="middle")
    f._pan = None

    # ══════════ Ⓑ 这座山有多高 ═══════════════════════════════════
    PH2 = 344
    py2 = f.panel(0, py + PH + 22, W, PH2,
                  "Ⓑ 这座山有多高 ——　<tspan font-weight=\"700\">"
                  "一条 128K 序列，V3 那个规模</tspan>", RD,
                  sub="⚠️ 自己按算子推的估算，<tspan font-weight=\"700\">"
                      "当量级看，别当准数</tspan>")

    CARDS = (
        (RD, "#fce8e6", "不开重算", "%.2f TiB" % ACT_RAW_TIB,
         "⛔ 一整条全挂着", "光这一项就已经装不下"),
        (GR, "#e6f4ea", "开了全量重算", "%.2f GiB" % ACT_REMAT_GIB,
         "⭐ 每层只留入口那一份", "约 %d 倍的差距" % round(RATIO)),
    )
    # ⛔ 逐图审抓到：原来两个框画成一样大，「约 40 倍」只活在文字里 ——
    #   那一格是表不是图。⭐ 改成**按 40:1 画高度**，不看数字也知道差多少。
    HI, LO = 172.0, 172.0 / RATIO
    for i, (col, fill, nm, num, a, b) in enumerate(CARDS):
        x = 120 + i * 620
        h = HI if i == 0 else LO
        top = py2 + 34 + (HI - h)
        f.box(x, top, 540, h, fill, col, 8)
        f.box(x, top, 540, 4, col, col, 2)
        f.t(x + 270, py2 + 18, nm, col, True, 18, "middle")
        if i == 0:
            f.t(x + 270, top + 62, num, col, True, 34, "middle")
            f.t(x + 270, top + 98, a, INK, True, 14.5, "middle")
            f.t(x + 270, top + 132, b, GY, size=13.5, anchor="middle")
        else:
            f.t(x + 270, top - 12, num, col, True, 26, "middle")
            f.t(x + 270, top + h + 30, a, INK, True, 14.5, "middle")
            f.t(x + 270, top + h + 58, b, GY, size=13.5, anchor="middle")
    f.t(700, py2 + 34 + HI + 96,
        "⭐ 两个框的<tspan font-weight=\"700\">高度是按真实比例画的</tspan> ——　"
        "右边那条薄片就是重算之后剩下的厚度", GY, size=14, anchor="middle")
    f.t(700, py2 + 230, "⭐ 下一节整节都在讲这两栏之间那个箭头",
        GY, True, 14.5, "middle")
    f._pan = None

    yy = f.band(py2 + PH2 + 22, "info", "这张图顺带把两件事一起讲了", [
        "⭐ <tspan font-weight=\"700\">「为什么训练比推理贵」</tspan>：算力只贵 3 倍，"
        "可推理<tspan font-weight=\"700\">根本没有这座山</tspan> ——&#160;"
        "它算完一层就把中间结果扔了，只留 KV cache。"
        "<tspan font-weight=\"700\">真正拉开差距的是显存，不是算力。</tspan>",
        "⭐⭐ <tspan font-weight=\"700\">「峰值出现在哪一刻」</tspan>："
        "就是山顶那一竖 ——&#160;前向刚结束、反向还没开始。"
        "⛔ 到<tspan font-weight=\"700\">第五节</tspan>会把权重和优化器状态那两条"
        "<tspan font-weight=\"700\">水平带</tspan>叠上来，山形不变，只是整体抬高。",
    ], keep=True)

    yy = f.src(yy + 24,
               "⚠️ 台阶画了 12 级只是为了看得清 ——&#160;"
               "<tspan font-weight=\"700\">真实是 %d 层</tspan>，"
               "而且每层内部还有若干个中间张量，山坡比图上细密得多" % N_LAYER,
               "⛔ 山形画成<tspan font-weight=\"700\">直上直下</tspan>是简化："
               "真实曲线会因为 MoE 派发、attention 那几个大中间量而有凸起，"
               "<tspan font-weight=\"700\">但「顶点在前向末尾」这个结论不受影响</tspan>",
               "⚠️ %.2f TiB 与 %.2f GiB 两个数是<tspan font-weight=\"700\">"
               "自己按算子推的</tspan>（输入：V3 的 config ＋ 官方参考实现的 MLA 前向），"
               "<tspan font-weight=\"700\">没有第三方背书</tspan>"
               % (ACT_RAW_TIB, ACT_REMAT_GIB))
    f.save("fig4-act-bill.svg", yy + 6)


main()
