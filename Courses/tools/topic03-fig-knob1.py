# -*- coding: utf-8 -*-
r"""专题三 · §五「旋钮①：让每一份更小」的图（2026-09-12 加）。

⭐ 两件事，一张图：

  左 · **四种存法摆在同一个形状下** ——&nbsp;题眼是那个反直觉：
       **MQA 比 MLA 还小**。所以这一支比的从来不是「谁存得少」，
       是「同样一份字节换回多少能力」。

  右 · **为什么 RoPE 必须单独走一路** ——&nbsp;讲义写着这是
       「理解 MLA 的关键一步，也是最容易讲糊的一步」。
       它本质是**一个代数重写成不成立**的问题，
       ⭐ 而「成不成立」画出来比说出来清楚得多。

📌 数字与 §二 那张图、以及课前第一题同源，脚本里当场算并断言。
"""
from topic03_draw import (Fig, wpx, _sz,
                          BL, OR, GR, RD, GY, PU, CY, BR, INK,
                          GY2, LINE, LINE2, BG2)

W = 1400
GiB = 2 ** 30


def main():
    L, T, B = 61, 131072, 2
    def tot(per): return per * B * L * T / GiB
    VAR = [
        ("MHA", 2 * 128 * 128, "每个头各存一份 K/V", GY, "128 份"),
        ("GQA-8", 2 * 8 * 128, "分成 8 组，组内共用", OR, "8 份"),
        ("MLA", 512 + 64, "压成 512 维隐向量 ＋ 64 维 RoPE", GR, "不按头存"),
        ("MQA", 2 * 1 * 128, "所有头共用同一份", RD, "1 份"),
    ]
    assert abs(tot(VAR[0][1]) - 488) < 1 and abs(tot(VAR[2][1]) - 8.58) < .05
    assert tot(VAR[3][1]) < tot(VAR[2][1])          # ⭐ 题眼：MQA 比 MLA 还小

    f = Fig(W, "旋钮一：让每一份更小。四种存法在同一形状下的对照 —— "
               "MQA 比 MLA 存得还少，所以这一支比的不是谁存得少；"
               "右边讲为什么 MLA 的 RoPE 必须单独走一路")
    f.marks = set()
    y = f.header(
        '旋钮 ① 让每一份更小 ——&#160;'
        '<tspan font-weight="700">但比的从来不是「谁存得最少」</tspan>',
        "同一个形状（V3：61 层 · 128 头 · 每头 128 维 · 128K · bf16），换四种存法")

    # ⭐⭐⭐ 2026-09-13 重画 ①。审图原话：「左半张是一张**六列表格**，
    #   最后那列的柱子自己承认『后三根细到几乎看不见』。」
    # ⭐ 真正的题眼是**二维**的：MQA 比 MLA 还小，**但更差**。
    #   一维表格画不出「又小又差」——&nbsp;所以换成散点：
    #   横轴「一份占多少地方」，纵轴「换回多少能力」，四个点一摆自己就说话了。
    # ⛔⛔ 纵轴必须诚实：**四家没有同一份可比的实测**。
    #   唯一同基准的一对是 MQA 论文表 3（MHA 29.9 / MQA 30.2 / 真单头 31.2）；
    #   GQA 与 MLA 各自的论文只声称「接近／不弱于 MHA」，不是同一张表。
    #   所以纵轴画成**两档定性**，并在图上写死这一句 ——
    #   画成连续刻度等于编出一份不存在的实测。
    CW = W
    PH = 486

    # ⛔⛔⛔ 2026-09-14 R58 重排。现场原话：「这部分为什么这么乱呢？字也乱，
    #   图也乱。」实测量下来是**四处**，而且全是同一个病根：
    #   分区带只有 96／90px 高，每个点却压着 **3 行标签**（名字 21 ＋ 数值 17
    #   ＋ 小注 15，连行距共 78px）——&nbsp;装不下，于是**全往外漏**：
    #     · 三条小注（y=256）漏到绿带下沿（248）外，悬在两带中间的白缝里；
    #     · MQA 的小注（y=374）不但漏出红带（362），还**压过了横轴线**（368）；
    #     · 题眼那根红箭头的尾巴（y=284）悬在 MQA 名字（296）上方够不着点，
    #       箭头却**正扎在「8.58 GiB」这四个字上**（头 230 vs 字 234）；
    #     · 题眼那句（y=348）跟「3.81 GiB」（y=352）几乎同一条基线，挤成一坨。
    # ⭐⭐ 判据：**标签溢出不会报错，它只是把字画到别人的地盘上。**
    #   而这一版四处漏的方向各不相同（上/下/横轴/邻行），所以看着像「到处都乱」,
    #   其实只要一句话就说完：**带子不够高。**
    #
    # ⭐⭐⭐ 重排的关键一招不是「把带子加高」（那只治溢出，治不了箭头）——
    #   是**让两带的标签背对背**：绿带的字全在点的**上方**，红带的字全在点的
    #   **下方**。于是两个点隔着中缝面对面，中间 56px **一个字都没有** ——
    #   题眼那根箭头终于有地方走，不用再从谁的脸上碾过去。
    # ⛔ 另外砍掉了横轴上「3 / 30 / 488」三个刻度：**每个点底下已经印着自己的
    #   数**，刻度再印一遍就是同一个数出现两次，而 MQA 那处的撞车正是这么来的。
    #   只留 10 / 100 两个十进制刻度撑住「这是对数轴」。
    # ⛔ 「📐 横轴那几个数」整行搬进折叠的「出处与口径」——&nbsp;R56 建那个块
    #   就是给这种口径行用的，它不该占正文的三行之一。
    # ══════════ ① 四种存法：一张散点 ══════════════════════════
    ay = f.panel(0, y, CW, PH, "四种存法 ——　把它们摆到一张图上", BL,
                 sub="横轴：一份占多少地方（对数）　·　纵轴：换回多少能力（⚠️ 定性）")

    X0, X1 = 210, 940
    BX0, BW = 170, 850             # 分区带
    BH, MID = 130, 22              # 带高 / 中缝
    YT = ay + 44                   # 绿带上沿
    RT = YT + BH + MID             # 红带上沿
    YB = RT + BH                   # 两带下沿 ＝ 横轴所在
    import math as _m
    def sx(g):                     # 1 ～ 1000 GiB，三个数量级
        return X0 + (X1 - X0) * _m.log10(max(g, 1.0)) / 3.0
    # 两档定性分区
    f.spot(BX0, YT, BW, BH, "#e6f4ea")
    f.spot(BX0, RT, BW, BH, "#fce8e6")
    f.t(BX0 + 10, YT + 30, "跟 MHA 基本打平", GR, True, 18)
    f.t(BX0 + 10, RT + 60, "明显更差", RD, True, 18)

    # 横轴：只留十进制刻度，数据点自己的数印在点下面，不重复
    f.line(BX0, YB + 16, BX0 + BW, YB + 16, LINE, 1.4, arrow=False)
    for g, lab in ((10, "10"), (100, "100")):
        f.line(sx(g), YB + 16, sx(g), YB + 23, GY2, 1.2, arrow=False)
        f.t(sx(g), YB + 40, lab, GY2, size=15, anchor="middle")
    f.t(BX0 + BW + 16, YB + 40, "GiB　·　对数轴，每往右一格 ×10", GY2, size=15)

    # ⭐ hi=1 走绿带、字在点**上方**；hi=0 走红带、字在点**下方**。
    #   背对背排是为了把两点之间的中缝腾空（见上面那段）。
    PTS = [("MHA", 488.0, 1, GY, "每个头各存一份"),
           ("GQA-8", 30.50, 1, OR, "8 组，组内共用"),
           ("MLA", 8.58, 1, GR, "压成 512 ＋ 64"),
           ("MQA", 3.81, 0, RD, "所有头共用一份")]
    for nm, g, hi, col, desc in PTS:
        bt = YT if hi else RT
        ny, vy, dy, cy = ((bt + 30, bt + 56, bt + 78, bt + 104) if hi
                          else (bt + 60, bt + 86, bt + 108, bt + 26))
        f.box(sx(g) - 9, cy - 9, 18, 18, col, col, 9)
        f.t(sx(g), ny, nm, col, True, 21, "middle")
        f.t(sx(g), vy, "%.2f GiB" % g if g < 100 else "%.0f GiB" % g,
            col, True, 17, "middle")
        f.t(sx(g), dy, desc, GY2, size=15, anchor="middle")

    # 题眼：MLA → MQA，走中缝，两头都落在点的边上而不是字上
    assert abs(8.58 / 3.81 - 2.25) < .01
    f.line(sx(8.58), YT + 115, sx(3.81), RT + 15, RD, 2.0)
    f.t(sx(3.81) - 14, YT + BH + 18, "2.25×", RD, True, 15, "end")
    f.t(470, RT + 88, "⭐⭐ 题眼：MQA 比 MLA 还小 2.25 倍，"
        "<tspan font-weight=\"700\">却更差</tspan>", RD, True, 19, w=540)

    f.box(1076, YT, 308, BH * 2 + MID, "#fff", INK, 10)
    f.t(1096, YT + 36, "⭐ 所以这一支比的", INK, True, 19, w=268)
    f.t(1096, YT + 62, "不是「谁存得最少」", INK, True, 19, w=268)
    f.t(1096, YT + 100, "MQA 早在 2019 年就把", GY, size=16, w=268)
    f.t(1096, YT + 124, "体积压到头了。", GY, size=16, w=268)
    f.t(1096, YT + 162, "⭐ 要比的是：同样一份", BL, True, 17, w=268)
    f.t(1096, YT + 186, "字节，换回多少能力。", BL, True, 17, w=268)
    f.t(1096, YT + 224, "（这正是第五节的线）", GY2, size=14, w=268)

    # ⭐ 诚实口径单独装进一个警示框：它是「为什么纵轴只有两档」的理由，
    #   不是正文。摊成两行灰字混在图下面，读者只会当成又一段小字跳过。
    f.box(16, YB + 62, 1368, 58, "#fef7e0", OR, 8)
    f.t(36, YB + 88, "⚠️ <tspan font-weight=\"700\">纵轴是定性的</tspan>"
        " ——　四家<tspan font-weight=\"700\">没有同一份可比的实测</tspan>。"
        "唯一同基准的一对是 MQA 论文表 3："
        "MHA 29.9 ／ MQA 30.2 ／ 真单头 31.2（困惑度，越低越好）。",
        GY, size=15.5, w=1328)
    f.t(36, YB + 112, "GQA 与 MLA 的论文只声称「接近／不弱于 MHA」，"
        "不是同一张表 ——　所以这里只画两档，不画连续刻度。", GY, size=15.5,
        w=1328)

    # ══════════ 右：RoPE 为什么必须单独走一路 ══════════
    # ⭐⭐⭐ 2026-09-14 R44 再画。上一版（09-13）是**寄快递**：仓库、压缩包、
    #   「一道必须在仓库做的工序」。那个比喻本身不坏，坏在**它是这一讲的第二套语汇**
    #   ——&nbsp;隔壁 §5.4b 的 fig3-absorb 已经把同一件事画成了
    #   「**紫色的 W_UK 方块从 c 那侧搬到 q 那侧**」，而且画得更好。
    #   ⛔ 同一个动作用两套画法讲两遍，读者要在脑子里做一次翻译才能把它们对上，
    #     而那次翻译不产生任何理解。**两张图共用一套语汇，第二张只加新元素。**
    #
    # ⭐⭐ 所以这一格现在是 absorb 那套语汇 ＋ 一个新元素：**一道闸**。
    #   紫块搬家 → 撞上闸 → 搬不过去。三行就说完了。
    #
    # ⭐⭐⭐ 顺带把「为什么挪不走」讲准了一层。原来两处都写「R 夹在中间，分不开」，
    #   论文原话也是「matrix multiplication does not obey a commutative law」。
    #   **这是结论，不是原因** ——&nbsp;矩阵乘法本来就满足**结合律**，
    #   「吸收」要的是结合律不是交换律。推导链：
    #       score = (R_t W_UQ c_t)ᵀ (R_j W_UK c_j)
    #             = c_tᵀ · [ W_UQᵀ R_tᵀ R_j W_UK ] · c_j
    #             = c_tᵀ · [ W_UQᵀ R_(j−t) W_UK ] · c_j      ← RoPE 是相对的
    #   ⭐ 于是关键在**下标**：中间那块要是一个**固定**矩阵 M，
    #     W_UQᵀ M W_UK 照样能预乘成**一个**矩阵，吸收完全成立；
    #     卡住的是 R 带着 (j−t)，**每一对 query/key 都是一个不同的矩阵**，
    #     要预乘就得预乘出一整套（有多少种相对距离就有多少个）。
    #   ⛔ 这一条本讲以前没说过（grep 过「下标」「相对位置」「每一对」全无命中）。
    y = y + PH + 18
    RX = 0
    PH = 830
    by = f.panel(RX, y, W, PH, "为什么 MLA 的 RoPE 必须单独走一路", PU,
                 sub="沿用 §5.4b 那块紫方块 ——&#160;这一格只多一样东西：一道闸")

    CH, GAPC = 52, 30          # chip 高 / chip 间距

    def chip(x, y0, w, s, col, tint, sub2=None, h=CH):
        f.box(x, y0, w, h, tint, col, 7, 1.6)
        f.t(x + w / 2, y0 + h / 2 + 7, s, col, True, 18, "middle", mono=True)
        if sub2:
            f.t(x + w / 2, y0 + h + 18, sub2, GY2, size=14, anchor="middle")
        return x + w + GAPC

    ey = by + 6
    # ① 本来能做成什么 —— 紫块搬家（跟 §5.4b 同一个动作）
    f.t(RX + 16, ey + 20, "① 本来能做成什么 ——&#160;"
        "<tspan font-weight=\"700\">紫色那块搬得走</tspan>", GR, True, 17)
    # ⚠️ R44 实测：ty 原来是 ey+34，搬家那条弧（ty-10）正好画在标题上。
    #   弧线必须走在**轨道上方**（不然会压住 chip 的小注），所以只能把轨道推下去。
    ty = ey + 64
    x = chip(RX + 40, ty, 70, "q", BL, "#e8f0fe", "一步只有一个")
    x = chip(x, ty, 118, "W_UK", PU, "#f3e8fd", "上投影")
    chip(x, ty, 84, "c", GR, "#e6f4ea", "仓库里存的就是它")
    # ⭐ 搬家动作本身：一条从紫块出发、落到 q 上的弧
    f.elbow(RX + 40 + 70 + GAPC + 59, ty - 16, RX + 75, ty - 16, PU, 2.0, r=8)
    f.t(RX + 128, ty - 24, "搬过去", PU, True, 15, "middle")
    f.t(x + 130, ty + 34, "＝", GY2, size=24)
    x2 = chip(x + 170, ty, 176, "(W_UKᵀq)", PU, "#f3e8fd", "预乘好，一次算完")
    chip(x2, ty, 84, "c", GR, "#e6f4ea", "原封不动")
    f.t(x2 + 130, ty + 34, "✅ 仓库里那些 c，一个都不用拆",
        GR, True, 17)
    ey = ty + CH + 46

    # ② 插进 RoPE：路当中立了一道闸
    f.t(RX + 16, ey + 20, "② 插进 RoPE：<tspan font-weight=\"700\">"
        "路当中立了一道闸</tspan>", RD, True, 17)
    ty = ey + 64
    x = chip(RX + 40, ty, 70, "q", BL, "#e8f0fe")
    GX = x                                  # 闸的位置
    # ⭐ 闸后面再错开叠两层：**第一次见到它就该知道它不是一块，是一摞。**
    #   下面 ③ 会把这件事讲透，这里先埋个形状。
    for k in (2, 1):
        f.box(GX + k * 9, ty - k * 9, 132, CH, "#fff", RD, 7, 1.0)
    x = chip(x, ty, 132, "R(j-t)", RD, "#fce8e6")
    # ⭐ 把它画成闸而不是又一个方块：三道横杠 ——&#160;方块和方块之间没有阻挡关系，
    #   横杠有。读者不看字也知道「这儿过不去」。
    for k in range(3):
        f.line(GX + 10, ty + 12 + k * 14, GX + 122, ty + 12 + k * 14,
               RD, 1.1, dash="5 4", arrow=False)
    XU = x
    x = chip(x, ty, 118, "W_UK", PU, "#f3e8fd")
    chip(x, ty, 84, "c", GR, "#e6f4ea")
    # 紫块想往左搬，撞在闸上
    f.line(XU + 50, ty - 18, GX + 140, ty - 18, PU, 2.0)
    f.t(GX + 128, ty - 10, "✕", RD, True, 26, "middle")
    f.t(GX + 66, ty - 30, "想搬，搬不动", PU, True, 15, "middle")
    f.t(RX + 16, ty + CH + 30,
        "⛔ <tspan font-weight=\"700\">紫块过不去</tspan>，"
        "那 W_UK c 就只能<tspan font-weight=\"700\">在仓库里先算出来</tspan>"
        " ——&#160;那等于又存了一份<tspan font-weight=\"700\">没压缩的 K</tspan>，"
        "MLA 白压了。", GY, size=17)
    ey = ty + CH + 54

    # ③ 真正卡住的不是「中间有东西」，是那东西带下标 ——&#160;**一块 vs 一摞**
    # ⭐⭐⭐ 这一格是这一轮真正新的东西。调研把全网这个机制的解读翻了一遍，结论是：
    #   「中间那项不是一个矩阵，是一族（跟位置差 t−i 相关）」这个洞见**有人说过**
    #   （苏剑林 kexue.fm/archives/10091 原话：「这里的 W_q R_(t−i) W_kᵀ
    #   就无法合并为一个固定的投影矩阵了（跟位置差 t−i 相关）」，已核），
    #   **但没有人把它画出来** ——&#160;全都是两行公式加一句话。
    # ⭐⭐ 所以这里只做一件事：**把「一块」和「一摞」摆在一起。**
    #   一块能预乘掉，一摞不能。不需要任何代数，形状自己说完了。
    # ⛔ 别把这一格写成三行说明文字 ——&#160;上一版就是那样，
    #   而「一摞」这个词写出来读者脑子里不会真的出现一摞。
    f.box(RX + 16, ey, CW - 32, 214, "#fef7e0", OR, 8)
    f.t(RX + 32, ey + 28, "⭐⭐ 卡住的不是「中间有东西」，是那东西"
        "<tspan font-weight=\"700\">带下标</tspan> ——&#160;"
        "<tspan font-weight=\"700\">一块</tspan>预乘得掉，"
        "<tspan font-weight=\"700\">一摞</tspan>预乘不掉", BR, True, 17)

    def card(x, y0, w, h, s, col, tint, sw=1.8):
        f.box(x, y0, w, h, tint, col, 6, sw)
        f.t(x + w / 2, y0 + h / 2 + 6, s, col, True, 16, "middle", mono=True)

    cy0 = ey + 74
    # 左：一块固定的 M
    f.t(RX + 40, cy0 - 22, "假如闸是<tspan font-weight=\"700\">固定</tspan>的一块",
        GY, True, 17)
    card(RX + 40, cy0 + 18, 96, 50, "M", GY2, "#f1f3f4")
    f.t(RX + 152, cy0 + 50, "→", GY2, size=24)
    card(RX + 190, cy0 + 18, 230, 50, "W_UQᵀ M W_UK", PU, "#f3e8fd")
    f.t(RX + 40, cy0 + 96,
        "✅ 还是<tspan font-weight=\"700\">一个</tspan>矩阵，预乘好收工 ——&#160;"
        "跟 ① 没区别", GR, True, 17)
    f.line(RX + 470, cy0 - 34, RX + 470, cy0 + 108, LINE, 1.2, dash="4 4",
           arrow=False)

    # 右：一摞 —— 每一对 (query, key) 一块
    f.t(RX + 510, cy0 - 22, "可 RoPE 是<tspan font-weight=\"700\">相对</tspan>的："
        "R_tᵀ R_j ＝ R(j-t)", RD, True, 17)
    # ⭐ 往右后方错开叠四层。它只说一件事：**这不是一块。**
    for k in (3, 2, 1):
        f.box(RX + 510 + k * 9, cy0 + 18 - k * 9, 96, 50, "#fff", RD, 6, 1.1)
    card(RX + 510, cy0 + 18, 96, 50, "R(1)", RD, "#fce8e6")
    f.t(RX + 622, cy0 + 50, "→", RD, size=24)
    for k in (3, 2, 1):
        f.box(RX + 660 + k * 9, cy0 + 18 - k * 9, 230, 50, "#fff", RD, 6, 1.1)
    card(RX + 660, cy0 + 18, 230, 50, "W_UQᵀ R(j-t) W_UK", RD, "#fce8e6")
    f.t(RX + 936, cy0 + 44, "有多少种距离", GY2, size=15)
    f.t(RX + 936, cy0 + 66, "就有多少块", GY2, size=15)
    f.t(RX + 510, cy0 + 96,
        "⛔ 这是<tspan font-weight=\"700\">一摞</tspan>，不是一个 ——&#160;"
        "<tspan font-weight=\"700\">预乘不出「那一个」，因为根本不存在那一个</tspan>",
        RD, True, 17)

    # ⚠️ 这一行 R44 实测顶出过右边框（被切在「出处见」三个字上），所以带了断言。
    LAW = ("📌 论文原话是「matrix multiplication does not obey a commutative "
           "law」 ——　那是结论。吸收要的其实是结合律，而结合律一直成立。"
           "（说法取自苏剑林，出处见页脚）")
    assert RX + 32 + wpx(LAW, 15) < W - 16, RX + 32 + wpx(LAW, 15)
    f.t(RX + 32, ey + 196,
        "📌 论文原话是「matrix multiplication does not obey a commutative law」"
        " ——&#160;那是结论。<tspan font-weight=\"700\">吸收要的其实是结合律，"
        "而结合律一直成立</tspan>。（说法取自苏剑林，出处见页脚）", GY2, size=15)
    ey += 232

    # ④ 解法：一条轨劈成两条
    f.t(RX + 16, ey + 20, "③ 解法：<tspan font-weight=\"700\">"
        "把一条轨劈成两条</tspan> ——&#160;让闸只站在其中一条上", PU, True, 17)
    # ⚠️ R44：ty 原来是 ey+36，底板（ty-30）正好盖住上面这行标题。
    ty = ey + 66
    f.box(RX + 16, ty - 30, CW - 32, 156, BG2, LINE, 8)
    x = chip(RX + 48, ty, 70, "q", BL, "#e8f0fe", h=44)
    x = chip(x, ty, 118, "W_UK", PU, "#f3e8fd", h=44)
    chip(x, ty, 84, "c", GR, "#e6f4ea", h=44)
    # ⭐ 这条小弧是特意补的：光写「紫块照旧搬走」是一句话，画出来才是同一个动作。
    f.elbow(RX + 48 + 70 + GAPC + 59, ty - 12, RX + 83, ty - 12, PU, 1.8, r=7)
    f.t(RX + 48 + 372, ty + 29,
        "<tspan font-weight=\"700\">512 维</tspan>，不带位置 ——&#160;"
        "闸不在这条上，<tspan font-weight=\"700\">紫块照旧搬走</tspan>", GR,
        size=17)
    ty += 62
    x = chip(RX + 48, ty, 70, "qᴿ", BL, "#e8f0fe", h=44)
    GX = x
    x = chip(x, ty, 132, "R(j-t)", RD, "#fce8e6", h=44)
    for k in range(3):
        f.line(GX + 10, ty + 9 + k * 13, GX + 122, ty + 9 + k * 13, RD, 1.1,
               dash="5 4", arrow=False)
    chip(x, ty, 84, "kᴿ", RD, "#fce8e6", h=44)
    f.t(RX + 48 + 420, ty + 29,
        "<tspan font-weight=\"700\">64 维</tspan>，专扛位置 ——&#160;"
        "这条<tspan font-weight=\"700\">不吸收，老实存着</tspan>", RD, size=17)
    f.t(RX + 16, ty + 92,
        "⭐ 两条并起来 ＝ <tspan font-weight=\"700\">512 ＋ 64 ＝ 576</tspan>。"
        "⛔ 注意这不是「把 RoPE 关小了」——&#160;"
        "<tspan font-weight=\"700\">位置信息一点没少，只是被赶到了一条窄轨上</tspan>，"
        "好让宽的那条保住 ① 的搬家动作。", INK, size=17)

    yy = y + PH + 18
    yy = f.band(yy, "info", "⭐ 把 5.3 当成一个套路记住，别当成 MLA 的实现细节", [
        '<tspan font-weight="700">「为了保住某个代数变换，把功能拆成两路」</tspan>'
        '——&#160;这个动作后面还会换面貌出现：'
        '<tspan font-weight="700">V4 的部分 RoPE、K3 的 NoPE</tspan>。',
        '⛔ <tspan font-weight="700">MLA 的代价是用计算换显存</tspan>：多了一对降维／升维矩阵乘。'
        '而且<tspan font-weight="700">它省的是推理时的 KV cache，不是训练时的激活</tspan>'
        '——&#160;训练前向里 K/V 会被解压出来算。<tspan font-weight="700">这是个非常常见的误解。</tspan>',
        '⭐ <tspan font-weight="700">Gated MLA</tspan>（K3）在 MLA 输出端加一个全秩门控 —— '
        '⛔ <tspan font-weight="700">它一个字节的 KV 都不省</tspan>，'
        '它让模型能学会「这一层这个位置，注意力的输出干脆不要」。'])

    yy = f.src(yy + 16,
               "📐 横轴那几个数：61 层 · 128K · bf16 · batch 1，一个用户一份 ——&#160;"
               "MHA 488 GiB ／ GQA-8 30.50 ／ MLA 8.58 ／ MQA 3.81",
               "四个数由公式当场算出并断言（脚本内）；MLA 超参出自 V3 论文 §4.2："
               "n_h=128, d_h=128, d_c=512, d_h^R=64, 61 层",
               "MQA：Shazeer arXiv 1911.02150　GQA：Ainslie 等 arXiv 2305.13245　"
               "MLA：DeepSeek-V2/V3 arXiv 2412.19437　Gated MLA：Kimi K3 技术报告 §2.1.2",
               "⭐ 「中间那项跟位置差相关，所以合并不成一个固定矩阵」这个说法取自 苏剑林《缓存与效果的极限拉扯：从 MHA、MQA、GQA 到 MLA》（kexue.fm/archives/10091）——&#160;原话已核")
    f.save("fig3-knob1.svg", yy + 6)


main()
