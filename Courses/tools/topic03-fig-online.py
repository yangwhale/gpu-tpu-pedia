# -*- coding: utf-8 -*-
r"""专题三 · §3.3「在线 softmax」—— 改到一半发现有人考得更高，怎么办

⭐⭐⭐ 2026-09-13 夜间 R16 新画。§三整节只有一张图，而**全节最像魔术的那一步
   一张图都没有**：softmax 明明要看完整行才能算，FlashAttention 凭什么
   一次只看一块也算得对？

⛔ 教材 3.3 用两句话带过了（「每来一块就更新一次 running max 和 running sum，
   并把已经累好的输出按比例重标定一次」）。⭐ 这句话**每个字都对**，
   但没学过的人读完不会有任何画面 —— 而它恰恰是后面「只跑到 35%」那个数的根。

⭐⭐ 本图的装置沿用 fig3-lowrank 那一招：**让读者自己核**。
   两条路（传统 / 在线）用同一组真数字各算一遍，最后一个数**必须一模一样**。
   脚本里断言差为 0 —— 不是「近似相等」，是精确相等。

🏠 生活版：**改到一半发现有人考得更高，不用把前面的卷子重翻一遍 ——
   把手里已经算出来的总分乘一个系数就行。**

⚠️ 这张图还要替 3.3 那句「不是免费的」出庭作证：
   那个系数每来一块就得乘一次，而它**不是 matmul**，还卡在归约链上。
"""
import math

from topic03_draw import (Fig, BL, OR, GR, RD, GY, PU, INK, GY2, LINE, LINE2,
                          BG2, wpx)

W = 1400
SC = [2.0, 3.0, 1.0, 5.0, 4.0]        # 五个 token 的注意力打分
VV = [1.0, 5.0, 2.0, 8.0, 3.0]        # 它们各自带来的内容（简化成一个数）
CUT = 3                                # 分两块：前三个 / 后两个


def main():
    # ══ 两条路各算一遍，最后一个数必须精确相等 ═══════════════════════
    m = max(SC)
    ex = [math.exp(x - m) for x in SC]
    S = sum(ex)
    OUT = sum(a * b for a, b in zip(ex, VV)) / S

    m1 = max(SC[:CUT])
    e1 = [math.exp(x - m1) for x in SC[:CUT]]
    s1 = sum(e1)
    n1 = sum(a * b for a, b in zip(e1, VV[:CUT]))
    m2 = max(m1, max(SC[CUT:]))
    fac = math.exp(m1 - m2)                       # ⭐ 这个系数就是「重标定」
    e2 = [math.exp(x - m2) for x in SC[CUT:]]
    s2 = s1 * fac + sum(e2)
    n2 = n1 * fac + sum(a * b for a, b in zip(e2, VV[CUT:]))
    # ⛔ 断言的是**精确相等**，不是近似 —— 在线 softmax 是恒等变换，不是近似算法。
    #   ⭐ 这一条要是写成「误差 < 1e-6」，读者就有理由怀疑它是个近似，而它不是。
    assert n2 / s2 == OUT, (n2 / s2, OUT)
    assert s2 == S, (s2, S)

    f = Fig(W, "在线 softmax：改到一半发现有人考得更高，怎么办")
    yy = f.header(
        "在线 softmax ——&#160;改到一半发现有人考得更高，不用把前面的卷子重翻一遍",
        "softmax 要先知道<tspan font-weight=\"700\">整行的最大值</tspan>才能算，"
        "而 FlashAttention 一次只看一小块。它凭什么算得对？"
        "这张图用五个真数字把两条路各走一遍 ——&#160;"
        "<tspan font-weight=\"700\">最后一个数你可以自己对。</tspan>",
        legend=[(RD, "传统：整行摊开"), (GR, "在线：一块一块来"),
                (OR, "重标定的那个系数")])

    # ⭐⭐⭐ 2026-09-14 R59 新增：旁白栏。
    #   ⛔ 这张图原本的毛病不是缺内容，是**缺一个连贯的声音**：
    #     三块面板各自都对，读者却要自己在心里把它们串成一个故事。
    #   ⭐ 现场给的三拍就是那条线，照抄不改：
    #     「先假装它就是最大的 → 来了更大的 → 把刚才假装留下的痕迹擦掉」。
    #   📌 刻意让算法**用第一人称说话**，而且只说动机、不说公式 ——
    #     公式一个都不动，旁白只负责回答「它这一步在想什么」。
    #     ⚠️ 所以旁白里不许出现任何这张图别处没有的数。
    def say(y, col, tint, lines, h=None, x0=30, x1=1370):
        h = h if h else 16 + 28 * len(lines)
        f.box(x0, y, x1 - x0, h, tint, "none", 8)
        f.box(x0, y, 5, h, col, col, 3)                 # 左侧引言竖条
        f.t(x0 + 22, y + 30, "💬", GY, size=18)
        for k, ln in enumerate(lines):
            f.t(x0 + 54, y + 30 + k * 28, ln, GY, size=16, w=x1 - x0 - 76)
        return y + h

    # 五张「卷子」的公共画法
    def papers(x0, y0, idx, col, tint):
        for k, i in enumerate(idx):
            f.icon("paper", x0 + k * 78, y0, 50, 60, col, tint)
            f.t(x0 + 25 + k * 78, y0 + 38, "%.0f" % SC[i], col, bold=True,
                size=20, anchor="middle")
            f.t(x0 + 25 + k * 78, y0 + 78, "内容 %.0f" % VV[i], GY2, size=14,
                anchor="middle")

    # ══ ① 传统：必须先看完整行 ═════════════════════════════════════
    PH1 = 268
    top = f.panel(0, yy, W, PH1,
                  "① 传统算法：<tspan font-weight=\"700\">得先把整行摊开</tspan>"
                  " ——&#160;因为要减最大值", RD, tag="片上要放得下 S 个数")
    ry = top + 46
    f.t(30, top + 32, "一整行的打分（这里只画 5 个，真实是 S 个）", GY,
        bold=True, size=16, cls="svglbl")
    papers(34, ry, range(5), RD, "#fce8e6")
    f.elbow(430, ry + 30, 500, ry + 30, RD, via="h")
    f.box(508, ry - 4, 210, 74, "none", RD, 8)
    f.t(613, ry + 26, "先找最大值", RD, bold=True, size=17, anchor="middle")
    f.t(613, ry + 52, "这里是 %.0f" % m, INK, bold=True, size=17,
        anchor="middle")
    f.t(742, ry + 22, "⛔ 这一步<tspan font-weight=\"700\">必须看完全部</tspan>"
        "才能动笔 ——", INK, size=16)
    f.t(742, ry + 46, "所以整行都得同时摆在片上暂存里。", INK, size=16)
    f.t(742, ry + 70, "128K 上下文，这一行就是 13 万个数。", RD, bold=True,
        size=16)
    say(top + 138, RD, "#fef3f2", [
        "「我得<tspan font-weight=\"700\">先把整行看完</tspan>，才知道该减掉多少 "
        "——　所以这一行你别想让我分块，它必须整个摊在我面前。」"])
    f.t(30, top + PH1 - 62,
        "⭐ <tspan font-weight=\"700\">注意省的是什么</tspan>："
        "减最大值是为了数值稳定（不减，exp 会溢出）。"
        "所以「必须先看完整行」不是实现懒，是这一步的定义就这样。", INK,
        size=16)
    f.t(30, top + PH1 - 36,
        "⛔ 而这正是那张 S×S 的大表非建不可的原因 ——&#160;"
        "FlashAttention 要拆掉的就是它。", GY, size=16)

    # ══ ② 在线：一块一块来，手里只攥三个数 ═══════════════════════════
    yy = top + PH1 + 26
    PH2 = 616                                  # ⭐ R59：430 → 616，让出三条旁白
    top = f.panel(0, yy, W, PH2,
                  "② 在线算法：一摞一摞地改，手里<tspan font-weight=\"700\">"
                  "只攥三个数</tspan>", GR, tag="片上只放一块")

    # — 第一摞
    ry = top + 54
    f.t(30, top + 34, "第一摞（前 3 个）", GR, bold=True, size=16, cls="svglbl")
    papers(34, ry, range(CUT), GR, "#e6f4ea")
    f.elbow(274, ry + 30, 344, ry + 30, GR, via="h")
    f.box(352, ry - 6, 300, 106, "none", GR, 8)
    f.t(368, ry + 20, "手里的三个数", GR, bold=True, size=16)
    f.t(368, ry + 46, "目前最高分 m ＝ %.0f" % m1, INK, size=16, mono=True)
    f.t(368, ry + 68, "目前总和   s ＝ %.4f" % s1, INK, size=16, mono=True)
    # ⛔ 这里原来写「已汇总内容 o」，而 o 是**还没除以 s** 的加权和。
    #   读者拿 9.8667 去对 ③ 里的 6.2793 会以为算错了 —— 把口径写进标签。
    f.t(368, ry + 90, "加权和（未除 s）o ＝ %.4f" % n1, INK, size=16, mono=True)
    f.t(676, ry + 34, "⭐ 就这三个标量。", GR, bold=True, size=17)
    f.t(676, ry + 58, "前三张卷子<tspan font-weight=\"700\">可以扔了</tspan>。",
        GY, size=16)

    say(top + 166, GR, "#eef7f0", [
        "「我<tspan font-weight=\"700\">先假装 3 就是全场最高分</tspan>，照这个口径"
        "把总和和加权和都算出来。反正真出现更高的，我到时候再改。」"])

    # — 第二摞：出现了更高分
    ry = top + 256
    f.t(30, ry - 18, "第二摞（后 2 个）——&#160;出事了", RD, bold=True, size=16,
        cls="svglbl")
    papers(34, ry, range(CUT, 5), RD, "#fce8e6")
    f.t(200, ry + 24, "⚠️ 这里有个 %.0f" % max(SC[CUT:]), RD, bold=True,
        size=18)
    f.t(200, ry + 48, "比手里的 %.0f 还高。" % m1, RD, size=16)
    f.t(200, ry + 72, "之前算的全都得改口径。", RD, size=16)

    # 重标定：一个系数
    f.elbow(404, ry + 30, 470, ry + 30, OR, via="h")
    f.box(478, ry - 6, 336, 106, "none", OR, 8)
    f.t(494, ry + 20, "重标定 ——&#160;只要乘一个数", OR, bold=True, size=16)
    f.t(494, ry + 48,
        "系数 ＝ exp(旧最高 − 新最高)", OR, size=16, mono=True)
    f.t(494, ry + 72,
        "     ＝ exp(%.0f − %.0f) ＝ %.4f" % (m1, m2, fac), INK, size=16,
        mono=True)
    f.t(494, ry + 94, "把 s 和 o 各乘一次，完事。", GY, size=15)

    f.elbow(822, ry + 30, 888, ry + 30, GR, via="h")
    f.box(896, ry - 6, 300, 106, "none", GR, 8)
    f.t(912, ry + 20, "更新后的三个数", GR, bold=True, size=16)
    f.t(912, ry + 46, "m ＝ %.0f" % m2, INK, size=16, mono=True)
    f.t(912, ry + 68, "s ＝ %.4f" % s2, INK, size=16, mono=True)
    f.t(912, ry + 90, "o ＝ %.4f（除以 s 才是输出）" % n2, INK, size=15,
        mono=True)

    say(top + 370, OR, "#fff8ec", [
        "「果然来了个 5。<tspan font-weight=\"700\">我不回头翻卷子</tspan> ——　"
        "只把刚才『假装 3 最高』在账上留下的痕迹擦掉：",
        "把 s 和 o <tspan font-weight=\"700\">各乘一次 exp(3−5)</tspan>，"
        "口径就换成『5 最高』了。」"])
    say(top + 456, BL, "#eef3fd", [
        "「擦完了。<tspan font-weight=\"700\">我手里这三个数，跟一开始就知道 "
        "5 最高、一次性算出来的完全一样</tspan> ——　下一格你可以自己核。」"])

    f.t(30, top + PH2 - 78,
        "🏠 <tspan font-weight=\"700\">这就是那句生活里的话</tspan>：",
        INK, size=17)
    f.t(30, top + PH2 - 50,
        "<tspan font-weight=\"700\">改到一半发现有人考得更高，"
        "不用把前面的卷子重翻一遍 ——&#160;把手里已经算出来的总分乘一个系数就行。"
        "</tspan>", INK, size=17)

    # ══ ③ 自己核 ══════════════════════════════════════════════════
    yy = top + PH2 + 26
    PH3 = 246
    top = f.panel(0, yy, W, PH3,
                  "③ 两条路，同一个数 ——&#160;这一格请<tspan "
                  "font-weight=\"700\">自己核</tspan>", BL,
                  tag="精确相等，不是近似")
    for i, (ttl, col, rows) in enumerate([
        ("传统：摊开整行", RD, [
            "减最大值 %.0f 之后：" % m,
            "  " + "  ".join("%.4f" % v for v in ex),
            "总和 s ＝ %.4f　加权和 o ＝ %.4f" % (S, sum(a * b for a, b
                                                      in zip(ex, VV))),
            "输出 ＝ o ÷ s ＝ %.4f" % OUT]),
        ("在线：两摞 ＋ 一次重标定", GR, [
            "第一摞：s ＝ %.4f　o ＝ %.4f" % (s1, n1),
            "乘系数 %.4f，再加第二摞：" % fac,
            "总和 s ＝ %.4f　加权和 o ＝ %.4f" % (s2, n2),
            "输出 ＝ o ÷ s ＝ %.4f" % (n2 / s2)]),
    ]):
        bx = 30 + i * 690
        f.box(bx, top + 30, 650, 150, "none", LINE, 8)
        f.t(bx + 18, top + 58, ttl, col, bold=True, size=17, cls="svglbl")
        for k, r in enumerate(rows):
            last = (k == len(rows) - 1)
            f.t(bx + 18, top + 88 + k * 26, r, INK if last else GY,
                last, 16, mono=True)
    f.t(700, top + 108, "＝", INK, bold=True, size=30, anchor="middle")
    f.t(700, top + 150, "✅", INK, size=22, anchor="middle")
    f.t(30, top + PH3 - 30,
        "⭐ 差是 <tspan font-weight=\"700\">0</tspan>，不是「误差很小」——&#160;"
        "在线 softmax 是<tspan font-weight=\"700\">恒等变换</tspan>，"
        "不是近似算法。<tspan font-weight=\"700\">它不掉点。</tspan>", INK,
        size=17)

    # ══ 落点 ══════════════════════════════════════════════════════
    yy = top + PH3 + 30
    yy = f.band(yy, "warn", "但它不是免费的 ——&#160;这就是 3.3 那句话的实体", [
        "每来一块，就多一次<tspan font-weight=\"700\">求 max</tspan>、"
        "一次 <tspan font-weight=\"700\">exp</tspan>、"
        "一次<tspan font-weight=\"700\">按系数重标定</tspan>。"
        "⛔ 这三样<tspan font-weight=\"700\">都不是矩阵乘</tspan>。",
        "⚠️ 更要命的是它们卡在<tspan font-weight=\"700\">归约链</tspan>上："
        "下一块要用上一块更新出来的 m 和 s，"
        "<tspan font-weight=\"700\">排不进矩阵乘的流水里一起跑</tspan>。",
        "⭐ 所以 3.6 那个「融合完还是只跑到 35%」，"
        "根不在配置上 ——&#160;<tspan font-weight=\"700\">有一部分就在这张图里</tspan>。",
    ])
    yy = f.band(yy + 14, "ok", "它省掉的到底是什么（这一条最容易说错）", [
        "⛔ <tspan font-weight=\"700\">不是省算力</tspan>："
        "该做的乘加一次不少，重标定还额外多了一些。",
        "⭐ 省的是<tspan font-weight=\"700\">片上暂存</tspan>："
        "传统要同时摆下整行 S 个数，在线一次只摆一块。"
        "<tspan font-weight=\"700\">于是那张 S×S 的大表根本不用建。</tspan>",
        "📌 这跟 §五 的账是<tspan font-weight=\"700\">两回事</tspan>："
        "§五 省的是要<tspan font-weight=\"700\">跨 token 留下来</tspan>的 KV cache，"
        "这里省的是<tspan font-weight=\"700\">算一步时中途摊开</tspan>的中间结果。"
        "⚠️ 两个都叫「省显存」，但省的不是同一样东西。",
    ])
    yy = f.src(yy + 16,
               "在线归约的做法出自 Milakov &amp; Gimelshein《Online normalizer "
               "calculation for softmax》（arXiv 1805.02867），"
               "FlashAttention（arXiv 2205.14135）把它用进了注意力",
               "⭐ 图里五个打分和五个「内容」是<tspan font-weight=\"700\">"
               "本课随手编的示例数</tspan>，但两条路的计算是脚本当场跑的，"
               "并且断言<tspan font-weight=\"700\">精确相等（差恒为 0）</tspan>",
               "⚠️ 「内容」这里简化成了一个数；真实的 V 是一个 d_h 维向量，"
               "重标定对整个向量同时做 ——&#160;道理一样，画成一个数只是为了能核")
    f.save("fig3-online.svg", yy + 6)


main()
