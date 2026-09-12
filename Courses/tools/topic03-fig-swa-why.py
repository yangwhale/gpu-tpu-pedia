# -*- coding: utf-8 -*-
r"""专题三 · §六「滑窗凭什么敢砍，砍了为什么会崩」（2026-09-13 夜间 · R6）。

⭐⭐ 三格，是一个完整的侦探故事，**按时间顺序讲最好听**：

  ① **凭什么敢砍** ——&nbsp;每层只看 W 个，但下一层看的是上一层的输出。
     递归下去，k 层之后信息能走 k×W ——&nbsp;**层数是免费的射程**。
     Mistral 7B：W=4096、32 层 → 理论射程 131,072（脚本当场乘出来断言）。

  ② **砍了为什么会崩** ——&nbsp;一个数字对比就够了：
     Llama-2-13B 在 PG19 上，纯窗口 0+1024 的困惑度是 **5158.07**；
     把**最前面四个 token** 留下来（4+1020），变成 **5.40**。
     ⭐⭐ 判决性实验：把那四个 token 换成**换行符**，5.60 ——&nbsp;几乎一样。
     **所以起作用的是位置，不是语义。**

  ③ **为什么会有这么个东西** ——&nbsp;softmax 要求一行加起来等于 1。
     当这一行「没什么特别想看的」，多出来的权重**总得放在某处**；
     而初始 token 因为自回归对**所有**后续位置可见，最容易被训成那个停车位。
     ⭐ 两个更彻底的解法：预训练时加一个可学的 sink token（一个就够），
     或者换成 softmax-off-by-one（分母 +1，**允许什么都不看**）。

⛔ 这一段是全讲「公式看不出来、画出来才看见」最好的例子 ——&nbsp;讲义里要点破。
"""
from topic03_draw import (Fig, wpx, BL, OR, GR, RD, GY, PU, CY, INK,
                          GY2, LINE, LINE2, BG2)

W = 1400
PX, PW = [0, 470, 940], [440, 440, 460]


def main():
    def fits(y, y0, ph, who):
        assert y <= y0 + ph - 6, "%s 到 %d，面板底边 %d" % (who, y, y0 + ph)

    WIN, LAY = 4096, 32
    span = WIN * LAY
    assert span == 131072                    # Mistral 说的「约 131K」就是这么来的
    assert 32768 // WIN == 8                 # 32K 序列下缓存省 8 倍
    PPL_WIN, PPL_SINK, PPL_NL = 5158.07, 5.40, 5.60
    assert PPL_WIN / PPL_SINK > 900          # 差了三个数量级
    SINKN = [(0, 3359.95), (1, 11.88), (2, 10.51), (4, 9.59), (8, 9.54)]
    assert SINKN[3][1] - SINKN[4][1] < 0.1   # 四个之后收益就没了

    f = Fig(W, "滑窗凭什么敢砍：层数是免费的射程；砍了为什么会崩：困惑度从 5.40 "
               "炸到 5158；为什么会有 attention sink：softmax 要求一行加起来等于一")
    f.marks = set()
    y0 = f.header(
        "滑窗　——　凭什么敢砍，砍了为什么会崩，那四个 token 到底是什么",
        "⭐ 这是一个侦探故事，<tspan font-weight=\"700\">按时间顺序讲最好听</tspan>："
        "先有办法，再出事故，最后才找到原因",
        [(GR, "能砍的理由"), (RD, "事故现场"), (PU, "真正的原因"),
         (OR, "更彻底的解法")])

    ph = 446

    # ══ ① 凭什么敢砍 ════════════════════════════════════════════
    x, pw = PX[0], PW[0]
    py = f.panel(x, y0, pw, ph, "① 凭什么敢砍", GR,
                 sub="层数是免费的射程")

    yy = py + 30
    # 画四层，每层窗口向左延伸，示意射程累加
    lh, lw = 34, pw - 106
    for k in range(4):
        ly = yy + k * lh
        f.box(x + 66, ly, lw, lh - 8, "#fff", LINE2, 4)
        reach = (k + 1) / 4.0
        f.box(x + 66 + lw * (1 - reach), ly + 2, lw * reach - 2, lh - 12,
              "#e6f4ea" if k < 3 else "#ceead6", "none", 3)
        f.t(x + 60, ly + 17, "第 %d 层" % (k + 1), GY2, size=11, anchor="end")
    f.t(x + 22, yy + 4 * lh + 12,
        "每层只看 W 个，但下一层看的是上一层的输出", GY, size=11.5,
        w=pw - 60)
    f.t(x + 22, yy + 4 * lh + 32,
        "递归下去 —— k 层之后，信息能走 k × W", GR, True, 12.5, w=pw - 60)

    yy = yy + 4 * lh + 48
    f.box(x + 22, yy, pw - 44, 76, "#fff", GR, 8)
    f.box(x + 22, yy, 4, 76, GR, GR, 2)
    f.box(x + 24, yy, 3, 76, "#fff", "#fff", 0)
    f.t(x + 40, yy + 24, "Mistral 7B：W = 4,096，32 层", GR, True, 12.5)
    f.t(x + 40, yy + 46, "→ 理论射程 %s 个 token" % format(span, ","),
        GY, size=12)
    f.t(x + 40, yy + 66, "缓存固定 W 个槽位，位置 i 存在 i mod W", GY2,
        size=11)
    yy += 88

    f.box(x + 22, yy, pw - 44, 64, "#fff", OR, 8)
    f.t(x + 38, yy + 24, "⚠️ 理论射程 ≠ 有效射程", OR, True, 12.5)
    f.t(x + 38, yy + 45, "信息每层只能挪一格窗口，而且一路被稀释", GY,
        size=11.5)
    fits(yy + 64, y0, ph, "①")

    # ══ ② 砍了为什么会崩 ════════════════════════════════════════
    x, pw = PX[1], PW[1]
    py = f.panel(x, y0, pw, ph, "② 事故现场", RD,
                 sub="Llama-2-13B，PG19 第一本书")

    yy = py + 26
    for cfg, ppl, note, col in [
        ("0 + 1024　纯窗口", PPL_WIN, "最前面那几个一被挤掉，模型就废了", RD),
        ("4 + 1020　留四个", PPL_SINK, "只是把最前面四个 token 留着", GR),
        ("4 个换行符 + 1020", PPL_NL, "把那四个换成「\\n」——&#160;几乎一样", PU),
    ]:
        f.box(x + 22, yy, pw - 44, 68, "#fff", col, 8)
        f.box(x + 22, yy, 4, 68, col, col, 2)
        f.box(x + 24, yy, 3, 68, "#fff", "#fff", 0)
        f.t(x + 40, yy + 26, cfg, col, True, 12.5)
        f.t(x + pw - 40, yy + 30, "%.2f" % ppl, col, True, 17, "end")
        f.t(x + 40, yy + 50, note, GY, size=11.5, w=pw - 150)
        yy += 78

    yy += 2
    f.box(x + 22, yy, pw - 44, 74, "#fff", PU, 8)
    f.box(x + 22, yy, 4, 74, PU, PU, 2)
    f.box(x + 24, yy, 3, 74, "#fff", "#fff", 0)
    f.t(x + 40, yy + 24, "⭐⭐ 第三行是判决性实验", PU, True, 13)
    f.t(x + 40, yy + 46, "换成换行符照样管用 ——", GY, size=11.5)
    f.t(x + 40, yy + 65, "起作用的是<tspan font-weight=\"700\">位置</tspan>，不是语义。", PU, True, 12.5)
    yy += 86

    f.t(x + 22, yy, "留几个够？（Llama-2-7B，4096 缓存）", GY, True, 12)
    yy += 14
    bx = x + 22
    for n_, v in SINKN:
        w_ = (pw - 44 - 4 * 6) / 5.0
        col = RD if n_ == 0 else (GR if n_ >= 4 else GY)
        f.box(bx, yy, w_, 42, "#fff", LINE, 6)
        f.t(bx + w_ / 2.0, yy + 17, "留 %d 个" % n_, GY2, size=11,
            anchor="middle")
        f.t(bx + w_ / 2.0, yy + 34, ("%.0f" if v > 100 else "%.2f") % v,
            col, True, 12, "middle")
        bx += w_ + 6
    fits(yy + 42, y0, ph, "②")

    # ══ ③ 真正的原因 ════════════════════════════════════════════
    x, pw = PX[2], PW[2]
    py = f.panel(x, y0, pw, ph, "③ 真正的原因：softmax 的一条硬约束", PU,
                 sub="它不是 bug，也不是特性")

    yy = py + 26
    f.box(x + 22, yy, pw - 44, 96, "#fff", PU, 8)
    f.box(x + 22, yy, 4, 96, PU, PU, 2)
    f.box(x + 24, yy, 3, 96, "#fff", "#fff", 0)
    f.t(x + 40, yy + 25, "softmax 要求<tspan font-weight=\"700\">一行加起来等于 1</tspan>", PU, True, 12.5)
    f.t(x + 40, yy + 48, "可这一行常常「没什么特别想看的」——", GY, size=11.5)
    f.t(x + 40, yy + 69, "那多出来的权重<tspan font-weight=\"700\">总得放在某处</tspan>。", GY, size=11.5)
    f.t(x + 40, yy + 88, "于是模型给自己找了个停车位。", GY2, size=11)
    yy += 110

    f.box(x + 22, yy, pw - 44, 74, "#fff", LINE, 8)
    f.t(x + 38, yy + 24, "为什么偏偏是最前面几个？", INK, True, 12.5)
    f.t(x + 38, yy + 46, "因为自回归 —— 它们对<tspan font-weight=\"700\">所有</tspan>后续位置都可见，",
        GY, size=11.5)
    f.t(x + 38, yy + 65, "全场只有它们人人都够得着。", GY, size=11.5)
    yy += 86

    f.t(x + 22, yy, "⭐ 两个更彻底的解法", OR, True, 13, cls="svglbl")
    yy += 22
    for lab, txt in [
        ("预训练时加一个可学的 sink token", "有了专用车位，<tspan font-weight=\"700\">一个就够</tspan>"),
        ("换 softmax-off-by-one", "分母 +1，<tspan font-weight=\"700\">允许这一行什么都不看</tspan>"),
    ]:
        f.box(x + 22, yy, pw - 44, 50, "#fff", OR, 8)
        f.t(x + 38, yy + 21, lab, OR, True, 12)
        f.t(x + 38, yy + 39, txt, GY, size=11.5)
        yy += 58
    fits(yy, y0, ph, "③")

    # ══ 落点带 ══════════════════════════════════════════════════
    yy = y0 + ph + 22
    yy = f.band(yy, "info", "⭐⭐ 这个故事真正的教益 —— 比 sink 本身值钱", [
        "attention sink 不是 bug，也不是谁设计的特性，"
        "它是<tspan font-weight=\"700\">归一化约束逼出来的副产品</tspan>："
        "你规定一行必须加起来等于 1，模型就一定会找个地方倒掉多余的那部分。",
        "⭐ 判据：<tspan font-weight=\"700\">看到模型里一个「毫无道理却极其稳定」的现象，"
        "先去找是不是某个守恒 / 归一化约束逼出来的。</tspan>"
        "量化里那批总也压不下去的 outlier，跟这是同一件事。",
        "⛔ 还有一条：<tspan font-weight=\"700\">这个 bug 从公式上完全看不出来</tspan> ——&#160;"
        "是把注意力矩阵画出来才发现的。这一讲所有的图，都是这个道理。",
    ])

    yy = f.band(yy + 14, "warn", "别把「理论射程」当「有效射程」", [
        "32 层 × 4,096 = <tspan font-weight=\"700\">131,072</tspan> 是个上界，"
        "说的是「信息最远能传到这儿」，不是「这么远还能用」——&#160;"
        "每跨一层只挪一格窗口，而且一路被后面的信息稀释。",
        "⭐ 稳妥说法：<tspan font-weight=\"700\">滑窗把「远处」从「看不见」"
        "变成了「看得见但很模糊」</tspan>，"
        "所以后面那些方案才要在滑窗之外再加一条「挑着看」的路。",
    ])

    yy = f.src(yy + 16,
               "① 出自 Mistral 7B arXiv 2310.06825 §2（k×W 射程、W=4096/32 层、"
               "rolling buffer cache）；131,072 与 8× 由脚本当场算并断言",
               "②③ 出自 StreamingLLM（Xiao 等 arXiv 2309.17453, ICLR 2024）"
               "表 1 / 表 2，以及原文 §3.1 与 §3.3：5158.07 → 5.40、换行符 5.60、"
               "留 1/2/4/8 个的对照、可学 sink token、softmax-off-by-one")
    f.save("fig3-swa-why.svg", yy + 6)


main()
