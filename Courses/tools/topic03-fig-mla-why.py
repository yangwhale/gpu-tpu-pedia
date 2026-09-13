# -*- coding: utf-8 -*-
r"""专题三 · §五「MLA 凭什么敢这么压」

⭐⭐⭐ 2026-09-13 **整张重画**。现场：「要跟现实生活中的世界结合起来，
   让普通人一眼看出原理，不要那么抽象 ——&nbsp;我们是**大众课程**。」

   这一版用一个人人都懂的画面：**一张原稿，复印了 128 份。**

   ① **白送的那一段（4.57×）** ——&nbsp;画一张原稿复印成一摞。
      摞起来很厚，可**信息还是那一张**。
      ⭐ 这就是「一次确定性映射不会凭空造出信息」的生活版：
      **复印件再多，也不会比原稿多出内容。**
      所以「存 128 份复印件」本来就是白占地方 ——&nbsp;
      改成「只存那一张原稿」，一个字的信息都没丢。
   ② **赌的那一段（12.4×）** ——&nbsp;画同一张原稿被**缩印**。
      缩印是**真会糊的** ——&nbsp;这一步不是白送，是赌。
   ③ **凭什么敢赌** ——&nbsp;因为在 MLA 之前，
      已经有人**拿训好的模型做过缩印实验**（Eigen Attention / Palu / LoRC）。
      ⭐ MLA 的不同是：**它从第一天就按缩印后的尺寸练字。**

📌 所有倍数当场算并断言，不写死。
⚠️ 「复印 / 缩印」是本课的比喻，论文那侧的说法是「低秩联合压缩」。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, GY, PU, INK, GY2, LINE, LINE2,
                          BG2)

W = 1400


def main():
    D_V3, NH, DH = 7168, 128, 128          # DeepSeek-V3 hidden / 头数 / 每头维
    KV = 2 * NH * DH                       # MHA 每 token 每层要存的数
    DC, DR = 512, 64                       # MLA 的隐向量 ＋ 解耦 RoPE
    LAT = DC + DR
    free = KV / float(D_V3)
    bet = D_V3 / float(LAT)
    tot = KV / float(LAT)
    assert KV == 32768 and LAT == 576
    assert abs(free - 4.571) < .01 and abs(tot - 56.9) < .05
    assert abs(free * bet - tot) < 1e-6

    f = Fig(W, "MLA 凭什么敢这么压：把它想成一张原稿复印 128 份 —— "
               "复印件再多也不会比原稿多出内容，所以扔掉复印件只存原稿是白送的；"
               "再把原稿缩印到 576 才是赌，而缩印实验在它之前就有人做过")
    f.marks = set()
    y0 = f.header(
        "MLA 凭什么敢这么压",
        "把它想成<tspan font-weight=\"700\">一张原稿，复印了 128 份</tspan>",
        [(GY, "复印件"), (BL, "原稿"), (PU, "缩印 ＝ 赌"), (GR, "白送的那段")])

    # ══════════ ① 白送 ══════════════════════════════════════════
    PH = 382
    py = f.panel(0, y0, W, PH, "① 第一步是白送的 ——　扔掉复印件，只留原稿",
                 BL, sub="复印 128 份，信息还是那一张")

    cy = py + 20
    f.t(56, cy + 26, "今天的存法：128 份复印件", GY, True, 20)
    for i in range(9):
        d = i * 5
        f.box(66 + d, cy + 48 + d, 146, 168, "#fff", LINE2, 8)
    f.box(111, cy + 93, 146, 168, "#fff", GY, 8)
    f.t(184, cy + 152, "K / V", GY, True, 22, "middle")
    f.t(184, cy + 186, "第 128 份", GY2, size=15, anchor="middle")
    f.t(56, cy + 296, "一共 %s 个数" % format(KV, ","), GY, True, 21)
    f.t(56, cy + 322, "每 token、每一层", GY2, size=14)

    f.line(296, cy + 176, 348, cy + 176, GY2, 1.6)
    f.t(322, cy + 158, "其实都是", GY2, size=14, anchor="middle")
    f.t(322, cy + 204, "从它算出来的", GY2, size=14, anchor="middle")

    f.box(366, cy + 86, 172, 180, "#e8f0fe", BL, 10)
    f.t(452, cy + 134, "原稿 h", BL, True, 24, "middle")
    f.t(452, cy + 174, "%s 个数" % format(D_V3, ","), BL, True, 20, "middle")
    f.t(452, cy + 212, "这一层的输入向量", GY, size=14, anchor="middle")

    f.box(596, cy + 48, 764, 258, "#fff", GR, 10)
    f.t(620, cy + 90, "⭐ 复印件再多，也不会比原稿多出内容", GR, True, 24)
    f.t(620, cy + 130, "所以「存 128 份」本来就是白占地方 ——", GY, size=18)
    f.t(620, cy + 162, "改成「只存那一张原稿」，一个字的信息都没丢。", GY, size=18)
    f.box(620, cy + 188, 300, 96, "#e6f4ea", GR, 8)
    f.t(770, cy + 226, "%s → %s" % (format(KV, ","), format(D_V3, ",")),
        GR, True, 24, "middle")
    f.t(770, cy + 262, "白送 %.2f 倍" % free, GR, True, 22, "middle")
    f.t(952, cy + 226, "这一段<tspan font-weight=\"700\">不用做实验</tspan>", GY, size=17)
    f.t(952, cy + 256, "算一下就是这样", GY2, size=15)

    # ══════════ ② 赌 ════════════════════════════════════════════
    y1 = y0 + PH + 18
    PH2 = 292
    py2 = f.panel(0, y1, W, PH2, "② 第二步才是赌 ——　把原稿再缩印一次",
                  PU, sub="缩印是真会糊的")

    dy = py2 + 20
    f.box(56, dy + 38, 168, 180, "#e8f0fe", BL, 10)
    f.t(140, dy + 112, "原稿", BL, True, 24, "middle")
    f.t(140, dy + 152, format(D_V3, ","), BL, True, 22, "middle")
    f.line(240, dy + 128, 296, dy + 128, PU, 1.8)
    f.t(268, dy + 110, "缩印", PU, True, 18, "middle")

    f.box(312, dy + 86, 122, 84, "#f3e8fd", PU, 10)
    f.t(373, dy + 126, "576", PU, True, 26, "middle")
    f.t(373, dy + 152, "个数", PU, size=15, anchor="middle")
    f.t(312, dy + 196, "＝ 512 ＋ 64", GY2, size=15)
    f.t(312, dy + 220, "后面那 64 是 RoPE，另走一路", GY2, size=13)

    f.box(478, dy + 38, 882, 204, "#fff", PU, 10)
    f.t(502, dy + 78, "⚠️ 这一步跟上一步，性质完全不同", PU, True, 22)
    f.t(502, dy + 114, "上一步是「扔掉复印件」——&#160;不丢信息，算出来的。", GY,
        size=18)
    f.t(502, dy + 146, "这一步是「把原稿缩小 12 倍」——&#160;一定会糊，", GY, size=18)
    f.t(502, dy + 176, "<tspan font-weight=\"700\">赌的是糊了也不影响用</tspan>。", GY,
        size=18)
    f.box(940, dy + 132, 200, 62, "#f3e8fd", PU, 8)
    f.t(1040, dy + 172, "赌 %.1f 倍" % bet, PU, True, 24, "middle")
    f.t(1168, dy + 172, "两段相乘 ＝ %.1f 倍" % tot, INK, True, 20)

    # ══════════ ③ 凭什么敢赌 ════════════════════════════════════
    # ⭐⭐⭐ 2026-09-14 R45 重画。原来这一格是**四张写着字的卡片**：三张橙的
    #   （Eigen Attention / Palu / LoRC）＋ 一张绿的写着「从第一天就按缩印后的
    #   尺寸练字」。那句绿卡上的话是这一格的全部价值，而它**只是一句话**。
    #
    # ⭐⭐ 装置：**两张稿纸，终点宽度像素级相同。**
    #   左边是一张写满大字的稿纸被缩印机压窄 ——&nbsp;笔画挤成一团；
    #   右边那张**生来就这么窄**，笔画少但根根分明。
    #   ⭐ 关键是**终点等宽**（代码里 assert 住）：终点一样、起点不一样，
    #     所以差别只能来自「什么时候变窄的」。这是 fig3-flip 面板①
    #     「两根柱子的权重段像素级相同」那一招的复用。
    # ⛔ 这一格**不主张「事后压掉点更多」** ——&nbsp;本课没有那个口径的实测。
    #   它只主张一件事：**两者性质不同**，一个是压，一个是生来如此。
    y2 = y1 + PH2 + 18
    PH3 = 348
    py3 = f.panel(0, y2, W, PH3, "③ 凭什么敢赌 ——　因为有人先拿训好的模型试过了",
                  GR, sub="事后缩印 vs 生来就这么窄 ——　终点一样宽，区别在什么时候变窄")

    ey = py3 + 12
    PAPH, NROW = 148, 4              # 稿纸高 / 行数

    def sheet(x, y0, w, label, col, tint):
        f.box(x, y0, w, PAPH, tint, col, 8, 1.6)
        f.t(x + w / 2, y0 + PAPH + 22, label, col, True, 17, "middle")

    def strokes(x, y0, w, n, col, sw=5.0):
        """⭐ 一行 n 笔，均匀铺满 w。**笔数是内容，宽度是容器** ——
        同样 n 笔塞进更窄的 w，笔就会挨上；这正是「缩印」和「生来就窄」的差别。"""
        for r in range(NROW):
            yr = y0 + 26 + r * 32
            for k in range(n):
                xk = x + 14 + (w - 28) * (k + 0.5) / n
                f.line(xk, yr, xk, yr + 18, col, sw, arrow=False)

    WIDE, NSTK = 300.0, 9
    # ⛔ 这两个必须相等，它就是整格的论点。分成两个名字而不是共用一个常数，
    #   是为了让「后来改了其中一个」当场炸掉 ——&#160;不等宽的话读者会读成
    #   「右边只是压得更狠」，而不是「它根本没被压过」。
    W_POST, W_MLA = 104.0, 104.0
    assert W_POST == W_MLA, "两张终点稿纸不等宽，这一格的论点就没了"
    NARROW = W_POST
    X_POST, X_MLA = 452.0, 940.0

    f.t(40, ey + 18, "事后缩印（MLA 之前就有人做）", OR, True, 20)
    sheet(40, ey + 34, WIDE, "训好的模型　每一笔都写开了", GY2, "#fff")
    strokes(40, ey + 34, WIDE, NSTK, GY2)
    f.line(360, ey + 108, 436, ey + 108, OR, 2.0)
    f.t(398, ey + 96, "缩印", OR, True, 16, "middle")
    f.t(398, ey + 130, "（事后低秩）", GY2, size=14, anchor="middle")
    sheet(X_POST, ey + 34, W_POST, "挤成一团", OR, "#fef7e0")
    strokes(X_POST, ey + 34, W_POST, NSTK, OR, sw=9.0)   # ⭐ 9 > 笔距 8.4，真的叠上
    f.t(40, ey + 232,
        "⛔ 同样 <tspan font-weight=\"700\">9 笔</tspan>硬塞进这么窄 ——&#160;"
        "挨上的那些就是<tspan font-weight=\"700\">被丢掉的方向</tspan>。"
        "写的时候没人知道以后要缩。", GY, size=17)

    f.line(700, ey + 10, 700, ey + 250, LINE, 1.2, dash="5 5", arrow=False)

    f.t(752, ey + 18, "MLA：生来就这么窄", GR, True, 20)
    sheet(X_MLA, ey + 34, W_MLA, "根根分明", GR, "#e6f4ea")
    strokes(X_MLA, ey + 34, W_MLA, 3, GR, sw=5.0)
    f.t(752, ey + 76, "没有「原稿」这一步。", GY, size=17)
    f.t(752, ey + 104, "训练时格子就这么大，", GY, size=17)
    f.t(752, ey + 132, "<tspan font-weight=\"700\">模型自己决定</tspan>", INK,
        size=17)
    f.t(752, ey + 160, "<tspan font-weight=\"700\">这三笔写什么</tspan>。", INK,
        size=17)
    f.t(752, ey + 232,
        "⭐ 两张终点稿纸<tspan font-weight=\"700\">一样宽</tspan> ——&#160;"
        "区别不在压得多狠，在<tspan font-weight=\"700\">什么时候变窄的</tspan>。",
        GR, True, 17)

    f.t(40, ey + 268,
        "📌 事后缩印这条路 MLA 之前早有人走，这也是本课说「赌得有依据」的依据："
        "<tspan font-weight=\"700\">Eigen Attention</tspan> 省 40%　·　"
        "<tspan font-weight=\"700\">Palu</tspan> 分组头低秩 ＋ 自动分配秩，省 50%　·　"
        "<tspan font-weight=\"700\">LoRC</tspan> 逐层分配不同的秩（未报统一比例）。",
        GY2, size=15)

    # ══════════ 落点 ════════════════════════════════════════════
    yy = y2 + PH3 + 20
    yy = f.band(yy, "info", "⭐ 带走一条判据：看任何一个压缩方案，先把它拆成两段", [
        "<tspan font-weight=\"700\">哪一段是「扔复印件」</tspan> ——&#160;"
        "信息本来就重复，算一下就知道，不用做实验。这一段是白送的。",
        "<tspan font-weight=\"700\">哪一段是「缩印」</tspan> ——&#160;"
        "赌它糊了也不影响用。<tspan font-weight=\"700\">这一段必须看掉点。</tspan>"
        "⛔ 一个方案只报总倍数、不拆这两段，多半是把赌的那部分算成了白送的。",
    ])

    yy = f.band(yy + 14, "warn", "两条口径，别讲过头", [
        "⚠️ 「白送 %.2f 倍」不是普适的：它等于 2·n_kv·d_h ÷ d_model，"
        "<tspan font-weight=\"700\">分子是真正被缓存的头数</tspan>。"
        "V3 是 MHA 所以 4.57；Llama-3-70B 是 GQA-8，算出来 <tspan "
        "font-weight=\"700\">0.25</tspan> ——&#160;那里根本没有「白送」这一段。" % free,
        "⚠️ 被缩印的是 <tspan font-weight=\"700\">K 和 V 合起来</tspan>那一份"
        "（两者共用同一张原稿），比「只压 K」狠得多；"
        "带 RoPE 的那 64 维<tspan font-weight=\"700\">另走一路，不在缩印范围里</tspan>。",
    ])

    yy = f.src(yy + 16,
               "维度出自 DeepSeek-V2 论文 §2.1（arXiv 2405.04434）与 V3 config："
               "d=7168, n_h=128, d_h=128, d_c=512, d_h^R=64；三个倍数由脚本当场算并断言",
               "事后低秩：Eigen Attention（EMNLP Findings 2024）、Palu、LoRC　"
               "「MLA 优于 MHA」的口径见 DeepSeek-V2 仓库 issue #26",
               "⚠️ 「复印 / 缩印」是<tspan font-weight=\"700\">本课的比喻</tspan> ——&#160;"
               "论文那一侧的说法是「低秩联合压缩」")
    f.save("fig3-mla-why.svg", yy + 6)


main()
