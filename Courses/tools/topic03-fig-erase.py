# -*- coding: utf-8 -*-
r"""专题三 · §7.2「查 / 擦 / 写」—— 三个动作，三代模型各缺一个

⭐⭐⭐ 2026-09-13 夜间 R19 新画。调研 agent 的结论很直接：
   **「先擦后写」这个动作，全网没有一张画好的图。**
   Songlin Yang 只给了 Householder 镜面（而且是 β=2 的完整反射，
   跟实际用的 β∈(0,1] 部分擦除对不上）；中文里讲得最透的那篇 KDA 长文
   **一张图都没有**。⭐ 所以这一张是本课最可能的差异点。

⭐ 装置：把状态画成**一排带地址的抽屉**，然后把 delta rule 拆成三个动作 ——
   **查（按地址找）· 擦（只掏空这一格）· 写（放新的进去）**。
   苏剑林给了这个动作最好的中文命名：**除旧迎新**。

⭐⭐ 然后同一排抽屉再用一次，讲遗忘门：
   · delta rule ＝ **定点擦一格**（手术刀橡皮）
   · 标量遗忘门 ＝ **所有格子一起变淡**（整屋一个调光开关）
   · KDA 的逐通道门 ＝ **每一格按自己的速度变淡**（每个灯泡一个调光器）
   收口是那句两行对仗：**decay 会忘但不会改，delta rule 会改但不会忘。**

⚠️ 一个必须写进图里的诚实修正：抽屉是**离散的**，而真实的「地址」是连续方向，
   擦也是**按比例擦**，而且**会顺带擦到相近的地址** ——
   ⭐⭐ 这正好回指 §1.3b：那里算过，128 维里塞一万个方向，
   最挤的一对还差 60 度 —— 「差不多不像」的代价，在这里就变成「擦串了」。

📌 比喻出处（都不是本课原创，逐条记明）：
   · 「除旧迎新」——&#160;苏剑林 kexue.fm/archives/11033
   · 「手术刀橡皮 ＋ 高压水枪」——&#160;Towards AI《Gated DeltaNet: The Surgical Eraser》
   · 「整屋一个调光开关 vs 每个灯泡一个调光器」——&#160;Amit Kapoor《Inside Kimi K3》
   · 「记忆的敌人不是时间，是别的记忆」——&#160;Eagleman《Livewired》，
     经 Songlin Yang 的 DeltaNet 博客引用
"""
from topic03_draw import (Fig, BL, OR, GR, RD, GY, PU, CY, INK, GY2, LINE,
                          LINE2, BG2, wpx, wrap_rich)

W = 1400
SLOT, SGAP = 108, 16              # 一格抽屉


def main():
    f = Fig(W, "delta rule 的三个动作：查、擦、写")
    yy = f.header(
        "查 · 擦 · 写 ——&#160;三个动作，三代模型各缺一个",
        "线性注意力把历史压进一块<tspan font-weight=\"700\">固定大小</tspan>的板子。"
        "板子会写满，这不奇怪 ——&#160;奇怪的是：<tspan font-weight=\"700\">"
        "为什么改进了这么多代，改的一直是「怎么擦」</tspan>？"
        "这张图把那个擦的动作拆开来看。",
        legend=[(BL, "查：按地址找"), (RD, "擦：只掏空这一格"),
                (GR, "写：放新的进去")])

    def shelf(x0, y0, fills, labels, dim=None, hi=None, empty=None,
              w=SLOT, h=78):
        """画一排抽屉。fills 是每格的颜色，dim 是变淡的比例（0=不淡）。"""
        for i, c in enumerate(fills):
            x = x0 + i * (w + SGAP)
            op = 1.0 - (dim[i] if dim else 0.0)
            f.box(x, y0, w, h, "none", RD if hi == i else LINE, 8,
                  sw=2 if hi == i else 1)
            if not (empty is not None and i == empty):
                f.p.append('<rect x="%d" y="%d" width="%d" height="%d" rx="5" '
                           'fill="%s" fill-opacity="%.2f"/>'
                           % (x + 10, y0 + 26, w - 20, h - 38, c, 0.85 * op))
            f.t(x + w / 2.0, y0 + 18, labels[i], GY, size=14, anchor="middle")
        return x0 + len(fills) * (w + SGAP) - SGAP

    COLS = [BL, GR, PU, OR, CY]
    ADDR = ["地址 k₁", "地址 k₂", "地址 k₃", "地址 k₄", "地址 k₅"]

    # ══ ① 三个动作 ════════════════════════════════════════════════
    PH1 = 492
    top = f.panel(0, yy, W, PH1,
                  "① 把 delta rule 拆成三个动作 ——&#160;"
                  "<tspan font-weight=\"700\">除旧迎新</tspan>", BL,
                  tag="S ← (I − βkkᵀ)·S ＋ βkvᵀ")
    for i, (nm, expl, col) in enumerate([
        ("① 查", "拿地址 k₃ 去找那一格，读出里面现在装的是什么", BL),
        ("② 擦", "只把这一格掏空。⭐ 旁边四格纹丝不动", RD),
        ("③ 写", "把新的内容放进同一格", GR),
    ]):
        ry = top + 40 + i * 140
        f.badge(24, ry + 18, i + 1, col)
        f.t(74, ry + 40, nm[1], col, bold=True, size=22, cls="svglbl")
        f.t(120, ry + 26, expl, GY, size=16)
        dim = [0, 0, 0.86, 0, 0] if i == 1 else None
        shelf(120, ry + 38, COLS, ADDR, dim=dim, hi=2,
              empty=None)
        if i == 0:
            f.t(120 + 2 * (SLOT + SGAP) + SLOT / 2.0, ry + 134,
                "读出 v_old ＝ S·k₃", BL, bold=True, size=16, anchor="middle")
        if i == 1:
            f.t(120 + 2 * (SLOT + SGAP) + SLOT / 2.0, ry + 134,
                "(I − βk₃k₃ᵀ)", RD, bold=True, size=16, anchor="middle")
        if i == 2:
            f.p.append('<rect x="%d" y="%d" width="%d" height="%d" rx="5" '
                       'fill="%s" fill-opacity="0.85"/>'
                       % (120 + 2 * (SLOT + SGAP) + 10, ry + 64, SLOT - 20, 40,
                          GR))
            f.t(120 + 2 * (SLOT + SGAP) + SLOT / 2.0, ry + 134,
                "＋ βk₃vᵀ（新内容）", GR, bold=True, size=16, anchor="middle")

    # ══ ② 三代模型，各缺一个动作 ═══════════════════════════════════
    yy = top + PH1 + 26
    PH2 = 446
    top = f.panel(0, yy, W, PH2,
                  "② 三代模型的区别，就是<tspan font-weight=\"700\">这三个动作"
                  "会几个</tspan>", GR, tag="改的一直是「怎么擦」")
    rows = [
        ("纯线性注意力", "只会「写」", RD,
         "只加不减。新的直接摞在旧的上面 ——&#160;"
         "<tspan font-weight=\"700\">板子越写越花</tspan>。",
         [0, 0, 0, 0, 0], None),
        ("遗忘门（标量 α）", "会「整体变淡」，不会定点擦", OR,
         "每来一步，<tspan font-weight=\"700\">所有格子一起按同一个比例变淡</tspan>。"
         "腾得出地方，但腾的是全部人的地方。",
         [0.45, 0.45, 0.45, 0.45, 0.45], None),
        ("delta rule", "会「定点擦」，不会整体淡", BL,
         "<tspan font-weight=\"700\">只擦你指名的那一格</tspan>，别的一点不碰。"
         "改得准，但没人主动腾地方。",
         [0, 0, 0.86, 0, 0], 2),
        ("门控 delta（GDN）", "两个都会", GR,
         "先整体淡一点，再定点擦一格。"
         "<tspan font-weight=\"700\">这就是 Gated DeltaNet。</tspan>",
         [0.35, 0.35, 0.88, 0.35, 0.35], 2),
        ("KDA（逐通道门）", "整体淡，但每格淡得不一样快", PU,
         "⭐ 调光器<tspan font-weight=\"700\">从一个总开关变成每格一个</tspan>。",
         [0.15, 0.62, 0.88, 0.30, 0.72], 2),
    ]
    SW, SH = 52, 40
    for i, (nm, cap, col, body, dim, hi) in enumerate(rows):
        ry = top + 40 + i * 68
        f.t(24, ry + 26, nm, col, bold=True, size=17, cls="svglbl")
        f.t(24, ry + 48, cap, GY2, size=14)
        for j in range(5):
            x = 250 + j * (SW + 8)
            f.box(x, ry + 8, SW, SH, "none", RD if hi == j else LINE, 5,
                  sw=2 if hi == j else 1)
            f.p.append('<rect x="%d" y="%d" width="%d" height="%d" rx="4" '
                       'fill="%s" fill-opacity="%.2f"/>'
                       % (x + 6, ry + 14, SW - 12, SH - 12, COLS[j],
                          0.85 * (1 - dim[j])))
        yy2 = ry + 26
        for r in wrap_rich(body, 812, 16 * 1.12):
            f.t(560, yy2, r, GY, size=16)
            yy2 += 22
    f.t(24, top + PH2 - 34,
        "⭐⭐ 这一支的改进史，一句话就能收住："
        "<tspan font-weight=\"700\">decay 会忘但不会改，delta rule 会改但不会忘</tspan>"
        " ——&#160;所以自然的下一步就是<tspan font-weight=\"700\">两个拼起来</tspan>。",
        INK, size=17)

    # ══ ③ 整屋一个调光开关，还是每个灯泡一个 ═══════════════════════
    yy = top + PH2 + 26
    PH3 = 296
    top = f.panel(0, yy, W, PH3,
                  "③ 「逐通道」到底是什么意思 ——&#160;"
                  "<tspan font-weight=\"700\">整屋一个调光开关，还是每个灯泡一个"
                  "</tspan>", PU, tag="KDA 相对 GDN 只改了这一处")
    for i, (ttl, sub_, dim, col) in enumerate([
        ("一个总开关", "标量 α ——&#160;Gated DeltaNet", [0.45] * 5, OR),
        ("每个灯泡一个旋钮", "向量 α ——&#160;KDA", [0.15, 0.62, 0.88, 0.30, 0.72],
         PU),
    ]):
        bx = 24 + i * 692
        f.box(bx, top + 34, 664, 156, "none", LINE, 9)
        f.t(bx + 18, top + 62, ttl, col, bold=True, size=19, cls="svglbl")
        f.t(bx + 18 + wpx(ttl, 19) + 16, top + 62, sub_, GY2, size=15)
        # ⛔⛔ 2026-09-15 R7：这几个百分比**必须当场标「示意」**。
        #   出处折叠里原来只写了「抽屉里的颜色深浅是示意」——&nbsp;
        #   ⭐ 可**数字比颜色像数据得多**：读者看见「85% 38% 12%」会当成
        #     实测的逐通道衰减率，而那是没有公开数据的。
        #   判据（R6 立的）：一条口径如果没了它结论就会被误读，它就是正文不是出处。
        # ⛔ 右对齐这行不能再长了：左边「向量 α ——&nbsp;KDA」那截就顶到这儿。
        #   2026-09-15 曾把「（示意值，不是实测）」塞进这一行，当场压穿两个框的
        #   小标题。⭐ 示意那句改成通栏单独一行，见下面 top+214。
        f.t(bx + 646, top + 62, "百分比 ＝ 一步之后还剩多少", GY2, size=14,
            anchor="end", w=248)
        for j in range(5):
            x = bx + 22 + j * 126
            f.icon("note", x, top + 84, 40, 46, GY2, "#fff")
            f.p.append('<rect x="%d" y="%d" width="26" height="30" rx="4" '
                       'fill="%s" fill-opacity="%.2f"/>'
                       % (x + 7, top + 94, COLS[j], 0.85 * (1 - dim[j])))
            f.t(x + 20, top + 150, "%d%%" % round(100 * (1 - dim[j])), GY2,
                size=14, anchor="middle")
        f.t(bx + 18, top + 178,
            "五格淡得一模一样" if i == 0 else "五格各淡各的 ——&#160;"
            "有的留得久，有的一步就没", GY, size=16)
    f.t(24, top + 214,
        '⚠️ <tspan font-weight="700">这十个百分比是示意值，不是实测</tspan>'
        ' ——&#160;逐通道的衰减率没有公开数据。'
        '这一格要画的只是<tspan font-weight="700">'
        '「五格一个样」和「五格各不一样」这个结构差别</tspan>。', GY, size=16)
    f.t(24, top + PH3 - 42,
        "🏠 为什么要分开调？<tspan font-weight=\"700\">"
        "「你现在在写哪门编程语言」这条该留很久；"
        "「刚离开的那个函数里的变量名」可以马上忘掉。</tspan>"
        "——&#160;一个总开关做不到这件事。", INK, size=17)

    # ══ 落点 ══════════════════════════════════════════════════════
    yy = top + PH3 + 30
    yy = f.band(yy, "warn", "⚠️ 抽屉这个画面有一处不诚实，必须说破", [
        "真实的「地址」<tspan font-weight=\"700\">不是一格一格的抽屉</tspan>，"
        "是连续的方向；擦也是<tspan font-weight=\"700\">按比例擦</tspan>"
        "（β 决定擦多干净），而且<tspan font-weight=\"700\">会顺带擦到相近的地址</tspan>。",
        "⭐⭐ 这正好回指讲<tspan font-weight=\"700\">「一个头装得下多少件事」</tspan>那张图：那里算过，"
        "128 维里塞一万个方向，最挤的一对还差 60 度 ——&#160;"
        "<tspan font-weight=\"700\">「差不多不像」的代价，在这里就变成「擦串了」</tspan>。",
        "📌 所以「定点擦」是个<tspan font-weight=\"700\">近似</tspan>，"
        "不是真的只动一格。板子越满，擦得越串。",
    ], fold=True)
    yy = f.band(yy + 14, "info", "为什么固定大小的板子一定会坏 ——&#160;一软一硬两句", [
        "<tspan font-weight=\"700\">硬的那句（可以验算）</tspan>："
        "d 维空间里最多只能有 d 个互相正交的方向，"
        "板子一满，新记录就只能<tspan font-weight=\"700\">挤在别人旁边</tspan>。"
        "　⭐ <tspan font-weight=\"700\">这一条上一张图已经画出来了</tspan> ——&#160;"
        "记事板那张的第 ③ 格：d 根方向、L 条要记的对应关系，"
        "<tspan font-weight=\"700\">L 一超过 d，就总有两条指到同一个方向上</tspan>。"
        "（把它放宽成「差不多不撞」之后还能装多少，在「主线 L300 · 完整版」。）",
        "<tspan font-weight=\"700\">软的那句（会被记住）</tspan>："
        "<tspan font-weight=\"700\">「记忆的敌人不是时间，是别的记忆。」</tspan>"
        "——&#160;你忘掉一个电话号码，不是因为时间久，是因为你又记了新的。",
        "⭐ 一软一硬配在一起，比任何一句单独说都有用："
        "诗给画面，数给它一个可以验算的身体。",
    ])
    yy = f.src(yy + 16,
               "delta rule 的 (I − βkkᵀ) 与「β∈(0,1] 是部分擦除、不是完整反射」"
               "出自 DeltaNet（arXiv 2406.06484）；逐通道遗忘门出自 Kimi Linear "
               "的 KDA（arXiv 2510.26692）",
               "⚠️ 图里的比喻<tspan font-weight=\"700\">都不是本课原创</tspan>，"
               "逐条记明：「除旧迎新」——&#160;苏剑林 kexue.fm/archives/11033；"
               "「手术刀橡皮 ＋ 高压水枪」——&#160;Towards AI《Gated DeltaNet: "
               "The Surgical Eraser》；「整屋一个调光开关 vs 每个灯泡一个调光器」"
               "——&#160;Amit Kapoor《Inside Kimi K3》；「记忆的敌人不是时间，"
               "是别的记忆」——&#160;Eagleman《Livewired》，"
               "经 Songlin Yang 的 DeltaNet 博客引用",
               "⭐ 抽屉里的颜色深浅是<tspan font-weight=\"700\">示意</tspan>，"
               "不对应任何模型的实测门控值")
    f.save("fig3-erase.svg", yy + 6)


main()
