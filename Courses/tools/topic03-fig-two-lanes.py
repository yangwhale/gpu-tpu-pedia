# -*- coding: utf-8 -*-
r"""专题三 · §5.4「那条看起来很别扭的窄轨 ——&nbsp;为什么 RoPE 必须单独走一路」

⭐⭐⭐ 2026-09-15 **从 `topic03-fig-knob1.py` 里拆出来。**

  原来那张图 1578 px，上下两格讲的是**两件不相干的事**：
    上 · 四种存法摆在同一个形状下 →&nbsp;回答「同样一份字节换回多少能力」（§5.1）
    下 · RoPE 为什么必须单独走一路 →&nbsp;回答「那条 64 维窄轨凭什么非有不可」（§5.4）

  ⛔⛔ **而且这不只是「图太高」，是放错了位置。**
  那张图挂在 §5.1，可下半格画的是 §5.4 的内容 ——&nbsp;
  于是读者走到 §5.4 的时候，看到的是**三段纯文字**在讲一件
  **三节之前已经画过**的事。
  ⭐ 判据：**一张图该待在「第一个需要它的段落」旁边**；
  它要是同时服务两个相隔很远的段落，那就是两张图。

  ⚠️ 顺带改掉一个 lint 抓不到的错：落点带原来写「把 **5.3** 当成一个套路记住」，
  而 5.3 是「凭什么敢压」，这个套路讲的是 5.4。
  **节号存在但指错 ——&nbsp;节号 lint 只查「存不存在」，查不了「对不对」。**
  所以干脆改成「这一节」，不再写死号码。

⭐⭐ 这一格跟 `fig3-absorb` **共用一套语汇**（紫色的 W_UK 方块从 c 那侧搬到 q 那侧），
   只多一样新东西：**一道闸**。
   ⛔ 同一个动作用两套画法讲两遍，读者要在脑子里做一次翻译，而那次翻译不产生理解。
"""
from topic03_draw import (Fig, wpx, _sz,
                          BL, OR, GR, RD, GY, PU, CY, BR, INK,
                          GY2, LINE, LINE2, BG2)

W = 1400


def main():
    f = Fig(W, "为什么 MLA 的 RoPE 必须单独走一路：紫色的上投影矩阵本来能从"
               "仓库那侧搬到提问那侧，可 RoPE 在中间立了一道闸；"
               "卡住的不是「中间有东西」，是那东西带下标 —— "
               "每一对谁问谁都是一个不同的矩阵，预乘不出「那一个」。"
               "解法是把一条轨劈成两条，让闸只站在其中一条上")
    f.marks = set()
    y = f.header(
        '那条看起来很别扭的窄轨 ——&#160;'
        '<tspan font-weight="700">512 ＋ 64 里的那个 64，是被逼出来的</tspan>',
        '⭐ 这一张只回答一个问题：<tspan font-weight="700">'
        'MLA 每个 token 存的那份，为什么是 512 ＋ 64，而不是干净的 512</tspan>',
        [(PU, "搬得走"), (RD, "被闸挡住"), (GR, "劈成两条轨之后")])

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
    RX, CW = 0, W
    PH = 830
    by = f.panel(RX, y, W, PH, "为什么 MLA 的 RoPE 必须单独走一路", PU,
                 sub="沿用「吸收」那张图里的紫方块 ——&#160;这一格只多一样东西：一道闸")

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
    yy = f.band(yy, "info", "⭐ 把这一节当成一个套路记住，别当成 MLA 的实现细节", [
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
               "MLA 超参出自 DeepSeek-V3 论文 §4.2：n_h=128, d_h=128, "
               "d_c=512, d_h^R=64, 61 层 ——&#160;512 ＋ 64 ＝ 576 由脚本断言",
               "⭐ 「中间那项跟位置差相关，所以合并不成一个固定矩阵」这个说法取自 苏剑林《缓存与效果的极限拉扯：从 MHA、MQA、GQA 到 MLA》（kexue.fm/archives/10091）——&#160;原话已核",
               "⛔ 论文原话「matrix multiplication does not obey a commutative law」"
               "<tspan font-weight=\"700\">是结论不是原因</tspan> ——&#160;"
               "矩阵乘法本来就满足结合律，而「吸收」要的正是结合律。"
               "真正卡住的是 R 带着 (j−t)：<tspan font-weight=\"700\">"
               "有多少种相对距离，就有多少个不同的中间矩阵</tspan>",
               "Gated MLA（K3 在 MLA 输出端加全秩门控）：Kimi K3 技术报告 §2.1.2")
    f.save("fig3-two-lanes.svg", yy + 6)


main()
