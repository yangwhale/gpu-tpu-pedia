# -*- coding: utf-8 -*-
r"""专题三 · §7.5「绕回原点：KV cache 才是状态的一种」

⭐⭐⭐ 2026-09-15 新画。起因是现场原话：
  「这一段讲得太好了，而且非常的重要。你在讲义和教材里边给我画图。」

⛔ **这一节是全讲的最高点，而它原来一张图都没有** ——&nbsp;
  §7 有 assoc / notepad / erase / chunkwise 四张，全部停在「机制」那一层；
  7.5 是**把前面七章全部重新安放一遍**的那一段，却只有台词。

⭐⭐ 这张图存在的理由，是**有一件事只有图做得到**：
  Ⓑ 那两个圈的**包含关系被翻过来**。
  「很多人说 Mamba 的状态是一种 KV cache；反过来说才更准确」——&nbsp;
  这句话用嘴说，台下要在脑子里自己转一次；**画出来是一眼的事**。
  判据①的正面用法：不是「图上没画的那句话」，是「这句话本来就该是画」。

⛔⛔ **刻意没画的东西**（免得以后有人「补全」）：
  ① Ⓐ 那条轴**没有刻度，也不会有** ——&nbsp;「压得有多狠」在四种存法之间
     不是一个可比的标量（GQA 砍的是头数，MLA 压的是维度，线性换的是整个
     数据结构）。画上刻度就是编。**它只承担「谁在左、谁在右、谁原地不动」。**
  ② Ⓒ 不画任何指标。数据库和大脑是**框架**不是 benchmark ——&nbsp;
     Gu 原文的落点恰恰是「这个比较本身不成立」。

📌 出处：Albert Gu《On the Tradeoffs of SSMs and Transformers》
  （goombalab.github.io/blog/2025/tradeoffs）。三段引文都在那里。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, GY, INK, GY2, LINE, LINE2)

W = 1400


def main():
    f = Fig(W, "把前面七章重新安放一遍：所有自回归模型都是状态模型，"
               "区别只在状态长什么样。全注意力把历史每一条原样留着，"
               "线性注意力压成一个固定大小的东西，而滑窗和 DSA 在这条轴上"
               "原地不动。于是 KV cache 不是状态的上位概念，"
               "它才是状态的一种，而且是最不压缩的那一种。"
               "Transformer 像一个数据库，这一支像一个大脑，而人是后者")

    y0 = f.header(
        "绕回原点：<tspan font-weight=\"700\">KV cache 才是状态的一种</tspan>，"
        "而不是反过来",
        "⭐ 这一格不讲新机制 ——&#160;它把<tspan font-weight=\"700\">"
        "前面走过的每一条路，重新摆到同一条轴上</tspan>",
        [(BL, "状态的两端"), (OR, "在这条轴上没动"), (GR, "落点")])

    # ══════════ Ⓐ 同一条轴 ══════════════════════════════════════
    PH = 430
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 每一个自回归模型都是「状态模型」——　区别只在状态长什么样",
                 BL,
                 sub="⚠️ 这条轴<tspan font-weight=\"700\">没有刻度</tspan>"
                     " ——　它只回答「谁在左、谁在右、谁原地不动」")

    AX = py + 96                      # 轴线的 y
    X0, X1 = 90, 1320
    f.line(X0, AX, X1, AX, GY2, 2.0)

    f.t(X0, AX - 46, "原样留着", BL, True, 17)
    f.t(X0, AX - 26, "每一条历史都单独存着，一个字不差", GY, size=13)
    f.t(X1, AX - 46, "压成一个固定大小的东西", BL, True, 17, "end")
    f.t(X1, AX - 26, "存多长的话，它都是这么大", GY, size=13, anchor="end")

    # 四站。x 是位置，不是刻度 —— 只表达先后。
    STOP = (
        (175, "全注意力", "把每一条 K/V 原样留着", BL, "①"),
        (470, "GQA", "八份合一份 ——　还是一条一条存", BL, "②"),
        (740, "MLA", "存压缩件 ——　小了很多，但仍随长度涨", BL, "③"),
        (1215, "线性注意力 / Mamba", "不存条目，只留一块板子", BL, "④"),
    )
    for cx, name, note, col, num in STOP:
        f.box(cx - 7, AX - 7, 14, 14, col, col, 7)
        f.t(cx, AX + 30, name, col, True, 16, "middle")
        f.t(cx, AX + 52, note, GY, size=12.5, anchor="middle")

    # ⭐ 视觉的全部重量在这里：左边三根越来越长，右边三个一样大。
    GY_ = AX + 108
    f.t(90, GY_, "句子越长，那个状态怎么变？", INK, True, 15)
    # ⛔ 渲染之后发现的：两组条形**没说清各自是谁的** ——&#160;
    #   读者会问「这三条是全注意力的，还是 GQA 的？」。答案是「左边那三站都是」。
    f.t(240, GY_ + 20, "左边那三站（全注意力 / GQA / MLA）", BL, True, 13.5)
    f.t(1120, GY_ + 20, "右端那一站（线性注意力）", GR, True, 13.5)
    LB = ("短句", "中等", "很长的一段")
    for i, wdt in enumerate((46, 104, 196)):
        yy = GY_ + 42 + i * 30
        f.box(240, yy, wdt, 19, "#e8f0fe", BL, 3)
        f.t(232, yy + 14, LB[i], GY, size=12, anchor="end")
    f.t(240, GY_ + 42 + 3 * 30 + 18, "⛔ 越说越长 ——　这三条都在这一端",
        RD, True, 14)

    for i in range(3):
        yy = GY_ + 42 + i * 30
        f.box(1120, yy, 96, 19, "#e6f4ea", GR, 3)
        f.t(1112, yy + 14, LB[i], GY, size=12, anchor="end")
    f.line(1216, GY_ + 36, 1216, GY_ + 42 + 3 * 30 - 4, GR, 1.4, "3 3",
           arrow=False)          # ⭐ 右边缘齐平线：把「一样大」变成看得见的
    f.t(1120, GY_ + 42 + 3 * 30 + 18, "⭐ 一样大 ——　它不认识「句子多长」",
        GR, True, 14)

    # ⛔⛔ 第六章那一支画在**轴上它该在的位置**，不是画在旁边再拉一条线过去。
    #   ⭐ 判据（渲染之后才看出来的）：**位置本身就是信息。**
    #     第一版把它放在中间下方、用一条长虚线指回最左边那个点 ——&#160;
    #     「原地不动」这件事于是只能靠读那行字，而不是靠看。
    #   现在：它就压在「全注意力」那一站的正上方，画一个回旋箭头。
    # ⛔⛔ 第一版把这两行放在 AX-116 / AX-98 ——&#160;那是 py-20，
    #   **压在面板标题栏上，糊成一团**。而自检没响：`_pan` 只追下沿不追上沿。
    #   ⭐ 判据（第二次踩）：**往上、往左跑出去的东西，探针是看不见的** ——&#160;
    #     只有真渲染出来看才抓得到。
    f.box(175 - 13, AX - 13, 26, 26, "none", OR, 13, 2.0)   # 同一站，第二个占位者
    f.t(175, AX + 74, "⛔ 滑窗 / DSA 也站在这一站", OR, True, 13.5, "middle")
    f.box(560, GY_ + 24, 470, 88, "#fef7e0", OR, 6)
    f.t(578, GY_ + 46, "⛔ 为什么它们不在这条轴上挪？", OR, True, 16)
    f.t(578, GY_ + 68, "因为它们改的是「每一步读进来几份」，不是「存几份」——", GY,
        size=13.5)
    f.t(578, GY_ + 88, "存的那一份一个字节都没少，所以站在原点一步没挪。", GY,
        size=13.5)

    yy = f.band(py + PH + 22, "info", "这张轴真正要说的一句话", [
        "前面<tspan font-weight=\"700\">六章</tspan>拧的旋钮，"
        "都在这条轴的<tspan font-weight=\"700\">左半边挪来挪去</tspan>；"
        "<tspan font-weight=\"700\">只有这一章，真的走到了右端。</tspan>",
        "⭐ 而走到右端的代价，也在图上：<tspan font-weight=\"700\">"
        "右边那三个方块一样大，意味着装不下的东西就是装不下</tspan>。",
    ])

    # ══════════ Ⓑ 包含关系被翻过来 ══════════════════════════════
    PH2 = 380
    py2 = f.panel(0, yy + 26, W, PH2,
                  "Ⓑ 反过来说才对 ——　这是这一段最狠的一句", BL,
                  sub="⭐ 同样两个词，<tspan font-weight=\"700\">"
                      "谁装着谁，换了个位置</tspan>")

    def pill(x, y, w, h, label, sub, col, fill="#fff", lsz=17):
        f.box(x, y, w, h, fill, col, h / 2.0, 1.6)
        f.t(x + w / 2.0, y + (28 if sub else h / 2.0 + 5), label, col, True,
            lsz, "middle")
        if sub:
            f.t(x + w / 2.0, y + 50, sub, GY, size=13, anchor="middle")

    # 左：大多数人的说法
    f.t(60, py2 + 28, "大多数人这么说", GY, True, 16)
    f.box(60, py2 + 44, 560, 196, "#fce8e6", RD, 24, 1.6)
    f.t(340, py2 + 76, "KV cache", RD, True, 20, "middle")
    f.t(340, py2 + 98, "（被当成那个大类）", GY, size=13, anchor="middle")
    pill(160, py2 + 126, 360, 86, "线性注意力的状态",
         "「它是一种 KV cache」", RD)
    f.t(60, py2 + 266, "⛔ 这么说的问题：", RD, True, 15)
    f.t(60, py2 + 288, "它把「最不压缩的那一种」当成了整个类别的名字。", GY,
        size=13.5)

    # 右：Gu 的说法
    f.t(780, py2 + 28, "⭐ Albert Gu 的说法", GR, True, 16)
    f.box(780, py2 + 44, 560, 196, "#e6f4ea", GR, 24, 1.6)
    f.t(1060, py2 + 76, "状态（state）", GR, True, 20, "middle")
    f.t(1060, py2 + 98, "每吐一个字就演化一次的那个东西", GY, size=13,
        anchor="middle")
    pill(806, py2 + 126, 246, 86, "KV cache", "最<tspan font-weight=\"700\">"
         "不</tspan>压缩的那一种", GR, "#fff", 16)
    pill(1068, py2 + 126, 246, 86, "一块固定的板子",
         "压得最狠的那一种", GR, "#fff", 16)
    f.t(780, py2 + 266, "⭐ 换过来之后多出来的东西：", GR, True, 15)
    f.t(780, py2 + 288,
        "它们是兄弟，不是父子 ——　于是「第一章那个盒子」不再是被淘汰的老办法。",
        GY, size=13.5)

    # 中间那个翻转箭头
    f.line(645, py2 + 142, 755, py2 + 142, GY2, 2.0)
    f.t(700, py2 + 128, "翻过来", GY, True, 14, "middle")

    yy = f.band(py2 + PH2 + 22, "ok", "翻过来之后，第一章那个盒子变成了什么", [
        "它不是一个<tspan font-weight=\"700\">被淘汰的老办法</tspan>，"
        "它是<tspan font-weight=\"700\">这一整个家族的原型</tspan>。",
        "⭐ 我们绕了三十年，不是绕回了一个旧东西 ——&#160;"
        "<tspan font-weight=\"700\">是绕回来之后，第一次知道自己为什么要它。</tspan>",
    ])

    # ══════════ Ⓒ 数据库 vs 大脑 ════════════════════════════════
    PH3 = 360
    py3 = f.panel(0, yy + 26, W, PH3,
                  "Ⓒ 一个比任何指标都好用的比喻", BL,
                  sub="⛔ 这一格<tspan font-weight=\"700\">刻意不放任何数字</tspan>"
                      " ——　它的落点恰恰是「这个比较本身不成立」")

    f.box(60, py3 + 24, 600, 210, "#fff", LINE, 8)
    f.t(360, py3 + 54, "Transformer　像一个数据库", BL, True, 19, "middle")
    for i, dx in enumerate((150, 250, 350, 450)):
        f.icon("drawer", dx, py3 + 76, 72, 66, BL, "#e8f0fe")
    f.t(360, py3 + 168, "每来一条新观察都当成重要资料归档", GY, size=14,
        anchor="middle")
    f.t(360, py3 + 190, "要用的时候翻出来 ——　一个字都不差", GY, size=14,
        anchor="middle")
    f.t(360, py3 + 216, "⛔ 代价：越存越多", RD, True, 15, "middle")

    f.box(740, py3 + 24, 600, 210, "#fff", LINE, 8)
    f.t(1040, py3 + 54, "这一支　像一个大脑", GR, True, 19, "middle")
    f.icon("board", 980, py3 + 76, 120, 66, GR, "#e6f4ea")
    f.t(1040, py3 + 168, "大小有限、一直在线、边听边处理", GY, size=14,
        anchor="middle")
    f.t(1040, py3 + 190, "不归档 ——　边听边把它揉进那块板子", GY, size=14,
        anchor="middle")
    f.t(1040, py3 + 216, "⛔ 代价：记不住一整本电话簿", RD, True, 15, "middle")

    f.icon("person", 672, py3 + 92, 56, 62, GR, "#e6f4ea")
    f.t(700, py3 + 176, "我们", GR, True, 15, "middle")
    f.line(700, py3 + 196, 700, py3 + 250, GR, 1.6, arrow=False)
    f.line(700, py3 + 250, 1040, py3 + 250, GR, 1.6, arrow=False)
    f.line(1040, py3 + 250, 1040, py3 + 240, GR, 1.6)
    f.t(700, py3 + 274,
        "而我们人，恰恰是<tspan font-weight=\"700\">后面这一种</tspan>　——　"
        "在「精确记忆」和「精确检索」上糟糕透顶", INK, size=15, anchor="middle")
    f.t(700, py3 + 300,
        "⭐ 而这好像<tspan font-weight=\"700\">并不妨碍智能出现</tspan>。",
        GR, True, 16, "middle")
    f._pan = None

    yy = f.band(py3 + PH3 + 22, "ok", "所以那个问法本身就不成立", [
        "很多人爱问「<tspan font-weight=\"700\">谁的长上下文更强</tspan>」。"
        "⭐ 可以反问一句：<tspan font-weight=\"700\">"
        "我自己的记忆，和我的研究笔记，哪个更好？</tspan>",
        "——&#160;<tspan font-weight=\"700\">它们只是不一样。</tspan>"
        "一个记得牢、查得准但越攒越厚；一个大小固定、一直在线但会记混。"
        "⛔ <tspan font-weight=\"700\">这正是下一章要「两个都要」的全部理由。</tspan>",
    ])

    yy = f.src(yy + 24,
               "三段引文均出自 Albert Gu《On the Tradeoffs of SSMs and "
               "Transformers》(goombalab.github.io/blog/2025/tradeoffs)："
               "①「每个自回归模型都持有一个状态」②「KV cache 才是状态的一种，"
               "而且是最不压缩的那一种」③「数据库 vs 大脑」",
               "⚠️ Ⓐ 那条轴<tspan font-weight=\"700\">没有刻度，而且不该有</tspan>"
               " ——&#160;GQA 砍的是头数、MLA 压的是维度、线性换的是整个数据结构，"
               "三者之间没有一个可比的标量。四个点的横坐标只表达先后，不表达倍数",
               "⚠️「滑窗 / DSA 原地不动」指的是<tspan font-weight=\"700\">"
               "这条轴（存多少）</tspan>——&#160;它们在另一条轴（每步读多少）上走得很远",
               "⛔ Ⓒ 刻意不放任何指标：数据库与大脑是<tspan font-weight=\"700\">"
               "框架不是 benchmark</tspan>，原文的落点就是「这个比较不成立」")
    f.save("fig3-gu.svg", yy + 6)


main()
