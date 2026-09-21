# -*- coding: utf-8 -*-
r"""专题四 · §1.0 插一格「nat / bit / 困惑度 —— 同一件事的三副面孔」

⭐⭐⭐ 2026-09-21 新画。现场（语音课）原话：
  「那个 9 次 per token 是什么鬼？一点都看不出来，这个得详细的讲一下。」
  ——&#160;他把 **nats** 听成了「9 次」。⛔ 但根子不在听错：
  **全讲从头到尾没有一处说过 nat 是什么**。听错只是把这个洞照出来。

⛔⛔ 第一版画成了**写字板**：40 段文字、0 个形状，构建时的图审计当场判
  「偏板子」。⭐ 判据（这一讲反复用的那条）：**图要画出关系，
  不是把结论写在板子上。** 所以重画，三块都用真几何：

  Ⓐ **双面尺** ——&#160;同一根轴，上沿刻 nat、下沿刻 bit。
     刻度疏密不同（1 nat ＝ 1.4427 bit），但**同一个横坐标是同一件事**。
     ⭐ 这就是厘米／英寸那把双面直尺 ——&#160;「换单位」这件事被画出来了。
  Ⓑ **压缩条** ——&#160;每个小格子＝1 bit，按真实比例排。
     瞎猜 17 格 vs 收敛 2.9 格，**长度差是量出来的，不是写上去的**。
  Ⓒ **困惑度** ——&#160;画成「在几个词之间犹豫」的词块。
     收敛 7.4 个画得下；瞎猜 129,280 个**画不下** ——&#160;
     ⭐ 这个「画不下」本身就是这一档最好的说明。

⭐ 所有数脚本当场算，四条 assert 钉着。
"""
import math

from topic03_draw import Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE

W = 1400
V = 129_280                      # V3 词表（专题一核过）

NAT2BIT = math.log2(math.e)      # 1 nat = 1.4427 bit
L_RAND = math.log(V)             # 瞎猜时的 loss（nat）
L_CONV = 2.00                    # 收敛附近，公开模型量级
B_RAND, B_CONV = L_RAND * NAT2BIT, L_CONV * NAT2BIT
RATIO = B_RAND / B_CONV

assert abs(NAT2BIT - 1.4426950) < 1e-6
assert abs(B_RAND - math.log2(V)) < 1e-6, "瞎猜那档换成 bit 必须正好＝log2(词表)"
assert abs(math.exp(L_RAND) - V) < 1.0, "瞎猜那档困惑度必须正好回到词表大小"
assert abs(RATIO - L_RAND / L_CONV) < 1e-9, "换底是等比例的，两套尺上压缩比必须相同"

STOPS = [(L_RAND, "刚初始化 · 瞎猜", RD),
         (7.00,   "训练早中期",      OR),
         (L_CONV, "收敛附近",        GR)]

X0, X1 = 300, 1210
NAT_HI = 12.5
PX_PER_NAT = (X1 - X0) / NAT_HI
PX_PER_BIT = PX_PER_NAT / NAT2BIT        # ⭐ 同一根轴，两种刻度密度


def main():
    f = Fig(W, "三块。第一块是一把双面尺：同一根横轴，上沿刻的是 nat，"
               "下沿刻的是 bit。两排刻度疏密不同，因为一个 nat 等于一点四四个 bit，"
               "但同一个横坐标代表的是同一件事 —— 就像厘米和英寸印在同一把直尺上。"
               "轴上标了三档：刚初始化瞎猜、训练早中期、收敛附近。"
               "第二块是压缩条，每个小格子代表一个 bit，按真实比例排开："
               "瞎猜时要十七格，收敛后只要不到三格，长度差将近六倍，"
               "这就是模型学到的东西换算成压缩的样子。"
               "第三块画困惑度，也就是模型在几个词之间犹豫："
               "收敛时是七点四个词，画得下；瞎猜时是十二万九千二百八十个词，"
               "整张纸都画不下，而这个画不下本身就是最好的说明")

    # ══ Ⓐ 双面尺 ══════════════════════════════════════════════════
    PH, py = 226, 60
    f.panel(60, py, W - 120, PH,
            "Ⓐ 一把双面尺 —— 上沿 nat，下沿 bit，同一个位置是同一件事", BL,
            sub="就像厘米和英寸印在同一把直尺上：刻度疏密不同，量的是同一段长度")

    ax = py + 116
    f.line(X0 - 26, ax, X1 + 36, ax, INK, 2.4)

    n = 0
    while n <= 12:
        x = X0 + n * PX_PER_NAT
        big = (n % 2 == 0)
        f.line(x, ax - (16 if big else 9), x, ax, BL, 1.6 if big else 1, arrow=False)
        if big:
            f.t(x, ax - 22, "%d" % n, BL, True, 12, "middle")
        n += 1
    f.t(X0 - 36, ax - 22, "nat", BL, True, 15, "end")
    f.t(X0 - 36, ax - 5, "loss 本身", GY2, size=11, anchor="end")

    b = 0
    while b <= 18:
        x = X0 + b * PX_PER_BIT
        if x > X1 + 30:
            break
        big = (b % 2 == 0)
        f.line(x, ax, x, ax + (16 if big else 9), PU, 1.6 if big else 1, arrow=False)
        if big:
            f.t(x, ax + 30, "%d" % b, PU, True, 12, "middle")
        b += 1
    f.t(X0 - 36, ax + 16, "bit", PU, True, 15, "end")
    f.t(X0 - 36, ax + 33, "× 1.4427", GY2, size=11, anchor="end")

    for nat, label, col in STOPS:
        x = X0 + nat * PX_PER_NAT
        f.line(x, ax - 58, x, ax + 46, col, 2.2, arrow=False)
        f.box(x - 78, ax - 86, 156, 26, "#fff", col, 6, sw=1.6)
        f.t(x, ax - 68, label, col, True, 12.5, "middle")
        f.t(x, ax + 62, "%.2f nat ＝ %.2f bit" % (nat, nat * NAT2BIT),
            col, True, 12.5, "middle")
    f._pan = None

    # ══ Ⓑ 压缩条：每格 1 bit ═════════════════════════════════════
    py2, PH2 = py + PH + 26, 190
    f.panel(60, py2, W - 120, PH2,
            "Ⓑ 一个格子 ＝ 一个 bit —— 长度是量出来的，不是写上去的", PU,
            sub="「模型平均要花多少个 bit，才能把下一个 token 记下来」")

    CW, CH, bx = 30, 26, 330
    for i, (lbl, bits, col) in enumerate(
            [("刚初始化 · 瞎猜", B_RAND, RD), ("收敛附近", B_CONV, GR)]):
        y = py2 + 58 + i * 56
        f.t(bx - 20, y + 18, lbl, col, True, 13.5, "end")
        full = int(bits)
        for k in range(full):
            f.box(bx + k * CW, y, CW - 3, CH, "#fff", col, 3, sw=1.4)
        frac = bits - full
        if frac > 0.02:
            f.box(bx + full * CW, y, max((CW - 3) * frac, 5), CH,
                  "#fff", col, 3, sw=1.4, dash="3 2")
        f.t(bx + bits * CW + 16, y + 18, "%.2f bit" % bits, col, True, 14)

    f.t(700, py2 + PH2 - 28,
        "⭐⭐⭐ 两条一比就完了：<tspan font-weight=\"700\">%.1f 倍</tspan>"
        "　——　<tspan font-weight=\"700\">这就是模型学到的全部东西，"
        "换算成压缩的样子</tspan>。" % RATIO, INK, size=15, anchor="middle")
    f._pan = None

    # ══ Ⓒ 困惑度：在几个词之间犹豫 ════════════════════════════════
    py3, PH3 = py2 + PH2 + 26, 188
    f.panel(60, py3, W - 120, PH3,
            "Ⓒ 困惑度 ＝ e^loss —— 「这一步在几个词之间犹豫」", OR,
            sub="同一个量的第三种读法，换算里同样不含任何实测值")

    f.t(310, py3 + 60, "收敛附近", GR, True, 13.5, "end")
    for k in range(7):
        f.box(330 + k * 46, py3 + 42, 40, 28, "#fff", GR, 4, sw=1.4)
    f.box(330 + 7 * 46, py3 + 42, 16, 28, "#fff", GR, 4, sw=1.4, dash="3 2")
    f.t(330 + 7 * 46 + 30, py3 + 60,
        "＝ 困惑度 %.1f —— 七个多词之间犹豫" % math.exp(L_CONV), GR, True, 13.5)

    f.t(310, py3 + 112, "刚初始化 · 瞎猜", RD, True, 13.5, "end")
    for k in range(14):
        f.box(330 + k * 46, py3 + 94, 40, 28, "#fff", RD, 4, sw=1.2)
    # ⛔ 这一句原来一行写完，结果它自己**顶出了面板** —— 图审计当场抓到。
    #   ⭐ 一句「画不下」自己画不下，很讽刺，但也说明：**溢出要靠工具查，
    #     不能靠眼睛扫**。拆成「块尾一句 ＋ 面板落点一句」。
    f.t(330 + 14 * 46 + 14, py3 + 112,
        "……　<tspan font-weight=\"700\">一共 %s 个</tspan>" % format(V, ","),
        RD, size=13.5)
    f.t(700, py3 + PH3 - 20,
        "⭐⭐ 瞎猜那一档<tspan font-weight=\"700\">画不下</tspan>　——　"
        "而它的困惑度正好<tspan font-weight=\"700\">就是整个词表</tspan>，"
        "「在十二万九千个词之间挑一个」<tspan font-weight=\"700\">就是瞎猜的定义</tspan>。",
        INK, size=14, anchor="middle")
    f._pan = None

    yb = f.band(py3 + PH3 + 20, "ok",
                "nat 和 bit 不是两种东西 —— 是同一把尺换了个底",
                ("⭐ <tspan font-weight=\"700\">bit</tspan> 是「一次二选一」的信息量，以 "
                 "<tspan font-weight=\"700\">2</tspan> 为底；"
                 "<tspan font-weight=\"700\">nat</tspan> 以 <tspan font-weight=\"700\">e</tspan> 为底 ——　"
                 "差一个固定常数 <tspan font-weight=\"700\">1.4427</tspan>。"
                 "我们这儿是 nat，<tspan font-weight=\"700\">纯粹因为 loss 的定义里写的是 ln</tspan>。",
                 "⭐⭐ <tspan font-weight=\"700\">per token</tspan> 那半截来自「一条序列几千个位置、"
                 "每个位置一个 loss、最后取平均」——　读作"
                 "<tspan font-weight=\"700\">「平均每个 token 多少 nat」</tspan>。"))

    yb = f.src(yb + 16,
               "⭐ Ⓐ 最左那档是<tspan font-weight=\"700\">自检</tspan>："
               "困惑度必须正好回到词表 129,280、bit 必须正好等于 log₂(词表)。"
               "脚本里 assert 钉着，对不上就是换算写错了。",
               "⚠️ 「训练早中期 7.00」是<tspan font-weight=\"700\">示意档</tspan>，"
               "「收敛附近 2.00」是公开模型的<tspan font-weight=\"700\">大致量级</tspan>　——　"
               "当参照看，别当某个具体模型的实测值。")

    f.save("fig4-natbit.svg", yb + 14)


if __name__ == "__main__":
    main()
