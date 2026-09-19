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
    f = Fig(W, "把显存占用按时间画出来，它是一座山。横轴是一个 step 里的时间，"
               "纵轴是显存里挂着的激活，单位 GiB。"
               "蓝色那半边是前向：从第一层到第六十一层一路往上堆，"
               "曲线按 V3 真实的层构成算出来 ——&#160;前三层是 dense、后五十八层是 MoE。"
               "堆到 loss 那一刻达到峰值 132.7 GiB，"
               "绿色那半边是反向：一层一层往回走，逐步释放。"
               "峰值不在训练的某个阶段，而在前向刚结束的那一瞬间。"
               "图里还有一条贴着地板的橙色曲线，那是开了全量重算之后的同一笔账，"
               "峰值只有 5.56 GiB，省了二十四倍；"
               "左下角有一个放大插图把它单独画了一遍，纵轴换了一把尺 ——&#160;"
               "可以看到它是一级一级的台阶，而且它的峰值不在 loss 那一刻，"
               "而在反向途中，因为那时候要额外物化当前正在重算的那一层")

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

    # ⛔⛔ 2026-09-20 重画。原来这一格是 `N = 12` 根合成柱子拼的**对称三角形**
    #   ——&#160;一个真数据都没有，而这一讲其余每张图都是算出来的。现场：
    #   「这个图挺异类的…柱子搞那么粗，显得很没有技术、初级。」
    # ⭐⭐ 根因不是「柱子太粗」，是**图形语法用错了**：横轴是<时间>（连续量），
    #   而柱状图天然在说「每根是一个可数的东西」。连续量该用面积／折线。
    # ⭐⭐⭐ 换成真数据之后，白捡了两个三角形画不出来的事实：
    #   ① V3 **前 3 层是 dense、后 58 层是 MoE**（config 的 first_k_dense_replace=3），
    #      所以上坡**不是一条直线**：开头三段缓，之后一路陡。
    #   ② 开了重算之后，**峰值不在 loss 那一刻，而在反向途中** ——&#160;
    #      因为那时候要额外物化「当前正在重算的那一层」。
    #      （这正好是 T04 补回来的那一项，三角形根本表达不了。）
    S_BASE = 4096
    _MLA_W   = 7168 + 7168 + 3072 + 1088 + 24576 + 32768 + 16384
    _MOE_W   = 14336 + 256 + 64512 + 55296 + 64512
    _DENSE_W = 3 * 18432 + 14336
    GIB = 1024.0 ** 3
    _b = lambda w2, w4=0: (w2 * 2 + w4 * 4) * S_BASE / GIB
    L_MOE   = _b(_MLA_W + _MOE_W, 128)       # 一层 MoE 块
    L_DENSE = _b(_MLA_W + _DENSE_W, 128)     # 一层 dense 块
    L_ENTRY = _b(7168)                       # 入口那一份（重算模式下留的就是它）
    N_DENSE = 3                              # ⚠️ V3 config：前 3 层 dense
    LAYERS = [L_DENSE] * N_DENSE + [L_MOE] * (N_LAYER - N_DENSE)
    assert abs(sum(LAYERS) - 132.65) < 0.05, "全模型激活 %.2f GiB" % sum(LAYERS)
    assert abs(L_ENTRY * N_LAYER + L_MOE - 5.56) < 0.02, "重算后峰值对不上"

    # 两条曲线：y[i] ＝ 走到第 i 个时刻时，显存里挂着多少
    raw, keep = [0.0], [0.0]
    for h in LAYERS:                                   # 前向：一层一层堆
        raw.append(raw[-1] + h)
        keep.append(keep[-1] + L_ENTRY)
    _fwd = len(raw) - 1
    # ⛔ 顺序要对：重算第 k 层的那一刻，**61 个存档点还都在场**，
    #   额外多出来的是当前这一层被物化出来的完整激活；
    #   释放那个存档点是**算完之后**的事。
    #   ⭐ 先加后减 —— 峰值 ＝ 全部存档点 ＋ 一层，写反了会少算一个存档点。
    _ck = keep[-1]                                     # 61 个存档点
    for i, h in enumerate(reversed(LAYERS)):           # 反向：一层一层拆
        raw.append(raw[-1] - h)
        _ck -= L_ENTRY if i else 0.0
        keep.append(_ck + h)                           # 存档点 ＋ 正在重算的那一层
    keep[-1] = 0.0
    PEAK_RAW, PEAK_KEEP = max(raw), max(keep)
    K_PEAK = keep.index(PEAK_KEEP)
    assert abs(PEAK_RAW - 132.65) < 0.05 and abs(PEAK_KEEP - 5.56) < 0.02
    assert K_PEAK > _fwd, "重算那条的峰值应该落在**反向**途中，而不是 loss 那一刻"

    X0, X1 = 150, 1330
    BOT, TOP = py + 318, py + 58
    SX = lambda i: X0 + (X1 - X0) * i / float(len(raw) - 1)
    SY = lambda v: BOT - (BOT - TOP) * v / PEAK_RAW

    f.line(X0, BOT, X1 + 10, BOT, GY2, 2.0, arrow=False)
    f.line(X0, BOT, X0, TOP - 14, GY2, 2.0, arrow=False)
    f.t(X0 - 12, TOP - 24, "显存里的激活（GiB）", GY, True, 13, "end")
    for v in (0, 40, 80, 120):                          # 真刻度，不是示意
        f.line(X0 - 6, SY(v), X1, SY(v), LINE2, 0.8, dash="3 6", arrow=False)
        f.t(X0 - 12, SY(v) + 5, "%d" % v, GY2, size=12, anchor="end")

    # ── 不重算：一整座山（前向蓝、反向绿，按峰值切开）
    for seg, col, fill in ((range(0, _fwd + 1), BL, "#1a73e820"),
                           (range(_fwd, len(raw)), GR, "#18803420")):
        pts = list(seg)
        d = ("M %.1f %.1f " % (SX(pts[0]), BOT)
             + " ".join("L %.1f %.1f" % (SX(i), SY(raw[i])) for i in pts)
             + " L %.1f %.1f Z" % (SX(pts[-1]), BOT))
        f.poly(d, fill=fill, stroke=col, sw=2.0)

    # ── 开了全量重算：贴着地板的那条（同一把尺，所以差距是真的）
    d2 = ("M %.1f %.1f " % (X0, BOT)
          + " ".join("L %.1f %.1f" % (SX(i), SY(v)) for i, v in enumerate(keep))
          + " L %.1f %.1f Z" % (X1, BOT))
    f.poly(d2, fill="#5f636814", stroke=OR, sw=2.0)

    f.t((X0 + X1) / 2.0, BOT + 26, "前向：第 1 层 →　第 %d 层" % N_LAYER,
        BL, True, 15, "end")
    f.t((X0 + X1) / 2.0 + 24, BOT + 26, "反向：第 %d 层 →　第 1 层" % N_LAYER,
        GR, True, 15, "start")
    # ⛔ 原来这里写「开头三段缓」——&#160;61 层里的 3 层在这个尺度上**肉眼看不出来**。
    #   ⭐ 判据：**图注不许声称画面上看不见的东西**（这一讲自己反复在讲这条）。
    f.t(X0 + 6, BOT + 52,
        "⭐ 曲线是按 V3 真实层构成算的：前 3 层 dense（每层 %.2f GiB）"
        "＋ 58 层 MoE（每层 %.2f GiB）" % (L_DENSE, L_MOE), GY, size=12.5)

    # ── 两个峰，两个时刻
    f.line(SX(_fwd), SY(PEAK_RAW) - 6, SX(_fwd), BOT, RD, 2.0, dash="5 4", arrow=False)
    f.box(SX(_fwd) - 150, TOP - 48, 300, 54, "#fce8e6", RD, 8)
    f.t(SX(_fwd), TOP - 26, "⭐ 峰值 %.1f GiB" % PEAK_RAW, RD, True, 17, "middle")
    f.t(SX(_fwd), TOP - 6, "前向刚算完、反向还没开始", GY, size=12.5, anchor="middle")

    # ⛔⛔ 橙色那条在同一把尺上**必然贴着地板**（差 24 倍）——&#160;
    #   于是「省了 24 倍」这个结论看得见，可**它自己的形状看不见**，
    #   而那个形状才是这一格新加的信息（峰值不在 loss 那一刻）。
    # ⭐ 判据：**同一张图里差一个数量级以上的两条线，小的那条必须另给一把尺。**
    #   放大插图不是装饰，是让「看不见的那条」重新变成可读的。
    # ⛔ 第一版把插图放在右下 ——&#160;正好压住绿色那半座山，还跟落点文字撞了。
    #   ⭐ 挪到左下：蓝色上坡的**内侧**是一大片空白，插图放那儿谁也不挡。
    IX0, IX1 = X0 + 34, X0 + 470
    IBOT, ITOP = BOT - 26, BOT - 156
    f.box(IX0 - 12, ITOP - 34, (IX1 - IX0) + 34, (IBOT - ITOP) + 60,
          "#fffaf2", OR, 8, 1.2)
    f.t(IX0 - 4, ITOP - 14,
        "🔍 把橙色那条单独放大（<tspan font-weight=\"700\">纵轴换了一把尺</tspan>）",
        OR, True, 12.5)
    iSX = lambda i: IX0 + (IX1 - IX0) * i / float(len(keep) - 1)
    iSY = lambda v: IBOT - (IBOT - ITOP) * v / (PEAK_KEEP * 1.18)
    f.line(IX0, IBOT, IX1, IBOT, GY2, 1.2, arrow=False)
    d3 = ("M %.1f %.1f " % (IX0, IBOT)
          + " ".join("L %.1f %.1f" % (iSX(i), iSY(v)) for i, v in enumerate(keep))
          + " L %.1f %.1f Z" % (IX1, IBOT))
    f.poly(d3, fill="#f9ab0022", stroke=OR, sw=2.0)
    f.line(iSX(_fwd), IBOT, iSX(_fwd), ITOP + 6, GY2, 1.0, dash="3 4", arrow=False)
    f.t(iSX(_fwd) - 6, ITOP + 18, "loss", GY2, size=11, anchor="end")
    f.box(iSX(K_PEAK) - 3, iSY(PEAK_KEEP) - 3, 6, 6, OR, OR, 3)
    f.t(iSX(K_PEAK), iSY(PEAK_KEEP) - 10,
        "峰值 %.2f GiB ——&#160;<tspan font-weight=\"700\">在反向途中</tspan>"
        % PEAK_KEEP, OR, True, 12, "middle")
    f.t(IX0, IBOT + 18,
        "⭐ 台阶是每释放一个存档点掉一小格；<tspan font-weight=\"700\">"
        "那一跳是「当前正在重算的那一层」被物化出来</tspan>", GY, size=11.5)

    f.t(X1 - 10, SY(PEAK_KEEP) - 14,
        "↓ 开了全量重算，整条压到这儿 ——&#160;峰值 "
        "<tspan font-weight=\"700\">%.2f GiB</tspan>，省 "
        "<tspan font-weight=\"700\">%.0f 倍</tspan>"
        % (PEAK_KEEP, PEAK_RAW / PEAK_KEEP), OR, True, 13, "end")
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
