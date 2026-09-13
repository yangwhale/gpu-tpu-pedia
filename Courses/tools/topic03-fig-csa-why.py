# -*- coding: utf-8 -*-
r"""专题三 · §6.4c「CSA 压什么、HCA 图什么、为什么交错」

⭐⭐⭐ 2026-09-14 **整张重画**，换成一个开过会的人都懂的画面：**记会议纪要。**

⛔ 第一件事仍然是把问题本身校正：CSA 压的**不是头，是 token**。
⭐⭐ 而这正是这一张最该留下的东西 ——&nbsp;**压缩有两个不同的方向**：

  · **每条记得更短** ——&nbsp;一条还是一条，只是字少了 →&nbsp;旋钮①（MQA/GQA/MLA）
  · **四条并成一条** ——&nbsp;条数变少了 →&nbsp;CSA / HCA
  · **只翻其中几条** ——&nbsp;那是旋钮②的稀疏

  ⭐ 画出来就一目了然：一摞便签，**一种是每张写得更短，一种是四张订成一张**。

⭐ HCA 的用意用「纪要的两个版本」讲：
  · CSA ＝ **详细版 ＋ 只翻几页** →&nbsp;看得细，但没翻到的完全不知道
  · HCA ＝ **极简版 ＋ 整本都看** →&nbsp;不会漏，但很粗
  ⚠️ 两种漏法正好相反，**交错摆也许正是为了互相兜底** ——&nbsp;
  但这只是一个讲得通的解释，论文只说了采用交错配置、没给理由。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, GY, PU, INK, GY2, LINE, LINE2,
                          BG2)

W = 1400


def main():
    M, MP = 4, 128                     # CSA 每 4 条压 1 条；HCA 每 128 条压 1 条
    assert MP // M == 32

    f = Fig(W, "CSA 压的是 token 不是头：把 KV 想成一摞会议便签 —— "
               "每张写得更短是旋钮一，四张订成一张才是 CSA；"
               "HCA 是极简版但整本都看，两种漏法相反，所以交错摆")
    f.marks = set()
    y0 = f.header(
        "CSA 压的是 token，不是头",
        "把 KV 想成<tspan font-weight=\"700\">一摞会议便签</tspan> ——&#160;"
        "压它有<tspan font-weight=\"700\">两个完全不同的方向</tspan>",
        [(BL, "每张写更短"), (GR, "四张订一张"), (OR, "只翻几页"),
         (PU, "两个版本交错")])

    # ══════════ ① 两个方向 ══════════════════════════════════════
    PH = 372
    py = f.panel(0, y0, W, PH, "① 一摞便签，两种压法 ——　它们是两件事",
                 BL, sub="这一格弄混了，后面全乱")

    ay = py + 24

    def note(x, y, w, h, lines, col, tint):
        f.box(x, y, w, h, tint, col, 6)
        for k, ln in enumerate(lines):
            f.box(x + 10, y + 12 + k * 12, ln, 5, col, "none", 2)

    # 原始：8 张便签，每张写得满
    f.t(56, ay + 22, "原来：8 张便签，每张写得满", GY, True, 20)
    for i in range(8):
        note(56 + i * 56, ay + 36, 46, 74, [26, 26, 26, 22, 26], GY2, "#fff")
    f.t(56, ay + 138, "8 条 × 每条 5 行", GY2, size=15)

    # 方向 A：每张写更短
    f.t(56, ay + 186, "方向 A：每张写得更短", BL, True, 22)
    for i in range(8):
        note(56 + i * 56, ay + 200, 46, 74, [26, 20], BL, "#e8f0fe")
    f.t(56, ay + 302, "还是 8 条，但每条只剩 2 行", BL, size=17)
    f.t(56, ay + 328, "→　<tspan font-weight=\"700\">这是旋钮①（MQA / GQA / MLA）"
        "</tspan>", BL, size=17)

    f.line(520, ay + 186, 520, ay + 300, LINE, 1.2, arrow=False)

    # 方向 B：四张订成一张
    f.t(560, ay + 186, "方向 B：四张订成一张", GR, True, 22)
    for i in range(2):
        note(560 + i * 120, ay + 200, 100, 74, [80, 80, 80, 70, 80], GR,
             "#e6f4ea")
        f.t(560 + i * 120 + 50, ay + 292, "第 %d 摞" % (i + 1), GR, size=14,
            anchor="middle")
    f.t(560, ay + 328, "条数 8 → 2　→　<tspan font-weight=\"700\">这才是 CSA"
        "（每 %d 条压 1 条）</tspan>" % M, GR, size=17)

    f.box(880, ay + 180, 480, 140, "#fff", INK, 10)
    f.t(904, ay + 220, "⭐ 两个方向互不相干", INK, True, 22)
    f.t(904, ay + 256, "「每条更短」和「条数更少」可以同时做 ——", GY, size=17)
    f.t(904, ay + 288, "DeepSeek-V4 就是<tspan font-weight=\"700\">两个一起，"
        "再加上「只翻几页」</tspan>。", GY, size=17)

    # ══════════ ② CSA vs HCA：纪要的两个版本 ════════════════════
    y1 = y0 + PH + 18
    PH2 = 318
    py2 = f.panel(0, y1, W, PH2, "② CSA 和 HCA ——　同一场会的两个版本",
                  PU, sub="一个细但会漏，一个粗但不漏")

    by = py2 + 22
    # CSA：详细版 + 只翻几页
    f.t(56, by + 20, "CSA ＝ 详细版　＋　只翻其中几页", GR, True, 22)
    for i in range(16):
        on = i in (2, 3, 10)
        f.box(56 + i * 40, by + 36, 32, 66, "#e6f4ea" if on else BG2,
              GR if on else LINE2, 4)
        if on:
            for k in range(4):
                f.box(62 + i * 40, by + 44 + k * 14, 20, 5, GR, "none", 2)
    f.t(56, by + 128, "✅ 翻到的那几页，内容很全", GR, True, 18)
    f.t(56, by + 158, "⛔ 没翻到的，<tspan font-weight=\"700\">等于完全不知道</tspan>", RD,
        size=18)

    # HCA：极简版 + 整本都看
    f.t(736, by + 20, "HCA ＝ 极简版　＋　整本都看", OR, True, 22)
    for i in range(16):
        f.box(736 + i * 40, by + 36, 32, 66, "#fef7e0", OR, 4)
        f.box(742 + i * 40, by + 62, 20, 5, OR, "none", 2)
    f.t(736, by + 128, "✅ 一页都不会漏", OR, True, 18)
    f.t(736, by + 158, "⛔ 但每页<tspan font-weight=\"700\">只剩一行，很粗</tspan>", RD,
        size=18)

    f.box(56, by + 188, 1304, 104, "#f3e8fd", PU, 10)
    f.t(80, by + 226, "⚠️ 两种漏法正好相反 ——　交错摆<tspan font-weight=\"700\">"
        "也许</tspan>正是为了让它们互相兜底", PU, True, 21)
    f.t(80, by + 264, "⛔ 但论文只说了采用交错配置，<tspan font-weight=\"700\">"
        "没给这个理由</tspan> ——&#160;这是一个讲得通的解释，不是它的设计意图。",
        GY, size=17)

    # ══════════ ③ 成绩 ══════════════════════════════════════════
    y2 = y1 + PH2 + 18
    PH3 = 196
    py3 = f.panel(0, y2, W, PH3, "③ 成绩：一百万上下文成为常规配置", GR,
                  sub="1M 下，V4-Pro 对 V3.2")

    ey = py3 + 24
    for i, (num, what, col, why) in enumerate([
        ("27%", "单 token 推理 FLOPs", GR, "「只翻几页」省的是算"),
        ("10%", "KV cache", BL, "「四条并一条」省的是存"),
    ]):
        bx = 120 + i * 420
        f.box(bx, ey + 18, 360, 112, "#fff", col, 10)
        f.t(bx + 28, ey + 74, num, col, True, 42)
        f.t(bx + 142, ey + 62, what, GY, True, 19)
        f.t(bx + 142, ey + 96, why, GY2, size=15)

    f.box(984, ey + 18, 376, 112, "#fff", PU, 10)
    f.t(1008, ey + 54, "⭐ 这两个数不一样，是有话说的", PU, True, 18)
    f.t(1008, ey + 88, "横着压主要省存储，", GY, size=16)
    f.t(1008, ey + 116, "稀疏主要省计算", GY, size=16)

    # ══════════ 落点 ════════════════════════════════════════════
    yy = y2 + PH3 + 20
    yy = f.band(yy, "info", "⭐⭐ 带走一条：拿到一个新方案，先问它动了哪几个方向", [
        "<tspan font-weight=\"700\">每条更短</tspan>（一个 token 存多少）· "
        "<tspan font-weight=\"700\">条数更少</tspan>（几个 token 合一条）· "
        "<tspan font-weight=\"700\">只翻几条</tspan>（这一步读哪些）——&#160;三个方向互不相干。",
        "⭐ 前两个都在回答「那份要留下来的有多大」，所以都算<tspan font-weight=\"700\">"
        "旋钮①</tspan>（只是下刀的维度不同，见 §4.2b）；第三个是<tspan "
        "font-weight=\"700\">旋钮②</tspan>。<tspan font-weight=\"700\">V4 三个一起拧。</tspan>",
    ])

    yy = f.band(yy + 14, "warn", "口径：这个 10% 跟别处那个 2% 不是一个基线", [
        "⚠️ 这里的 <tspan font-weight=\"700\">27% / 10%</tspan> 比的是 "
        "<tspan font-weight=\"700\">V3.2</tspan>；"
        "§6.1～6.6 那张五格 mask 图上写的「约 2%」比的是<tspan font-weight=\"700\">"
        "同形状的 GQA-8</tspan>，而且那 2% 还叠了一层跟注意力机制无关的 KV 混合精度。",
        "⭐ <tspan font-weight=\"700\">两个数都对 ——&#160;对的是各自的基线</tspan>，别并排比。",
    ])

    yy = f.src(yy + 16,
               "CSA / HCA 的机制出自 DeepSeek-V4 技术报告 arXiv 2606.19348 "
               "§2.3–2.3.1（每 m 个压一条 → DSA top-k → 并上滑窗；"
               "HCA 压 m′≫m 但保持密集，m=%d / m′=%d）" % (M, MP),
               "27% FLOPs / 10% KV cache 出自同一篇摘要与 §2.3.4（1M 上下文、对比 V3.2）",
               "⚠️ 「会议便签 / 两个版本」是<tspan font-weight=\"700\">本课的比喻</tspan>；"
               "⚠️ 「两种漏法互补所以交错」是从定义推出的解释，论文未给这个理由，"
               "也未在此给出层间配比")
    f.save("fig3-csa-why.svg", yy + 6)


main()
