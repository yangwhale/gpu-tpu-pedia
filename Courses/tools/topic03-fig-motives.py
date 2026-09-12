# -*- coding: utf-8 -*-
r"""专题三 · §二「两条独立的动机」的图（2026-09-12 加）。

⛔ 为什么这一节非画不可：**它原来是 887 汉字、0 张图** ——&nbsp;
   全讲最严重的一处「说话多、画图少」。而它的内容恰恰最适合画：
   一条推导阶梯、三个对照、三个观察、一个交汇处，全是结构。

⭐ 这张图替掉的正文（都已从 topic03-build.py 删掉）：
   · 488 GiB 那四行推导 →&nbsp;左栏的阶梯
   · 三个对照物那张表   →&nbsp;左栏的横条
   · 线索 B 的三个观察   →&nbsp;右栏三个小画面
   · 2.3 那段「只有 A / 只有 B / 两条合起来」→&nbsp;底部交汇带

📌 数字全部在 2026-09-11 当场复算过（见 wiki / 课前题）：
   488 ＝ 2×128×128 × 61 × 2 B × 131,072；GQA-8 ＝ 488/16 ＝ 30.5；
   MQA ＝ 488/128 ＝ 3.81；MLA ＝ (512+64)×2×61×131,072 ＝ 8.58 GiB。
⚠️ 488 是「**假如 V3 用 MHA**」的反事实值，不是 V3 实测 ——&nbsp;图上写死了。
"""
import math
from topic03_draw import (Fig, wpx, _sz,
                          BL, OR, GR, RD, GY, PU, CY, BR, INK,
                          GY2, LINE, LINE2, BG2)

W = 1400
GiB = 2 ** 30


def main():
    # ── 数字当场算，不写死 ─────────────────────────────────────
    L, T, B = 61, 131072, 2
    per_tok_layer = 2 * 128 * 128
    per_tok = per_tok_layer * L
    mha = per_tok * B * T / GiB
    gqa, mqa = mha / 16, mha / 128
    mla = (512 + 64) * B * L * T / GiB
    hbm, wgt = 94.74e9 / GiB, 671 * 2e9 / GiB
    assert abs(mha - 488) < 1 and abs(gqa - 30.5) < .1 and abs(mla - 8.58) < .05

    f = Fig(W, "长上下文的两条独立动机：左边是硬件账 —— 一个用户 128K 的 KV cache "
               "按最朴素的 MHA 算是 488 GiB，要 5.5 块 v7；右边是信息账 —— "
               "注意力矩阵极稀疏、有 attention sink、远近关注粒度不同；"
               "两条线交汇处才是所有变体的位置")
    f.marks = set()
    y = f.header(
        '为什么是现在 ——&#160;<tspan font-weight="700">两条完全独立的线，'
        '在同一个地方交汇</tspan>',
        "⛔ 这两条必须分开讲：它们指向同一批技术，但出发点完全不同 ——&#160;混在一起讲就成了名词罗列")

    CW, GAP = 672, 56
    RX = CW + GAP
    PH = 322

    # ══════════ 左：线索 A · 硬件账（必须省）══════════
    ay = f.panel(0, y, CW, PH, "线索 A · 硬件账算不过来", RD,
                 sub="结论：必须省。不解决它，长上下文根本上不了线")
    f.t(16, ay + 6, "拿 DeepSeek V3 砸体感（61 层 · 128 头 · 每头 128 维 · 128K · bf16）",
        GY, size=11)
    STEP = [
        ("每 token 每层", "2 × 128 × 128", "%s 个数" % format(per_tok_layer, ",")),
        ("× 61 层", "", "%s 个数" % format(per_tok, ",")),
        ("× 2 字节", "", "%.2f MiB / token" % (per_tok * B / 2 ** 20)),
        ("× 131,072 token", "", "%.0f GiB" % mha),
    ]
    sy = ay + 28
    for i, (a, b_, c) in enumerate(STEP):
        last = (i == len(STEP) - 1)
        f.t(20, sy + i * 24, a, INK if last else GY, bold=last, size=_sz(12))
        if b_:
            f.t(150, sy + i * 24, b_, GY2, size=11, mono=True)
        f.t(330, sy + i * 24, c, RD if last else GY, bold=last,
            size=_sz(14 if last else 12), mono=True)
    f.t(560, sy + 3 * 24, "← 一个用户", RD, bold=True, size=11)

    # 三个对照，用横条
    cy = sy + 4 * 24 + 14
    f.t(20, cy, "摆三个对照，让这个数站住", GY, bold=True, size=11)
    BAR0, BARW = 232, 190
    VALX, NOTEX = BAR0 + BARW + 74, BAR0 + BARW + 84
    REF = [("一块 v7 device 的 HBM", hbm, GY2, "一个用户就要 5.5 块"),
           ("V3 全部权重（671B × 2B）", wgt, BL, "单用户占 39%；三个并发就超过权重"),
           ("换成 GQA-8", gqa, OR, "省 16 倍 —— 还是装不进一块"),
           ("换成 MLA（V3 真实方案）", mla, GR, "省 56.9 倍")]
    mx = max(mha, wgt)
    for i, (nm, v, c, note) in enumerate(REF):
        yy = cy + 20 + i * 26
        f.t(20, yy + 4, nm, INK, size=11)
        # ⛔ 2026-09-12：第一版把数值写在「条子末端 ＋ 8px」，备注钉在固定列 ——
        #   条子一长，数值就冲进备注里。⭐ 判据（X-6 那次同一条）：
        #   **标签位置不要跟着条形长度走，钉在固定列上。**
        bw = max(3, BARW * math.log10(1 + v) / math.log10(1 + mx))
        f.box(BAR0, yy - 6, bw, 13, "#fff", c, 3, 1.4)
        f.t(VALX, yy + 4, "%.2f GiB" % v if v < 100 else "%.0f GiB" % v,
            c, bold=True, size=11, mono=True, anchor="end")
        f.t(NOTEX, yy + 4, note, GY2, size=11)
    f.t(20, cy + 20 + 4 * 26 + 12,
        '⛔ <tspan font-weight="700">权重是所有用户共享一份，KV cache 是每人一份</tspan>'
        '——&#160;所以它直接决定<tspan font-weight="700">你能同时服务多少人</tspan>。',
        RD, size=_sz(12))
    f.t(20, cy + 20 + 4 * 26 + 32,
        "⚠️ 488 是「假如 V3 用 MHA」的<tspan font-weight=\"700\">反事实</tspan>值，"
        "不是实测 ——&#160;V3 从第一天就是 MLA。", GY2, size=11)

    # ══════════ 右：线索 B · 信息账（可以省而不太亏）══════════
    by = f.panel(RX, y, CW, PH, "线索 B · 信息本身不需要那么多", GR,
                 sub="结论：可以省，而且不太亏")
    f.t(RX + 16, by + 6, "128K 的序列，真需要 128K 份独立的 KV 吗？三个观察 ——", GY, size=11)

    # ① 稀疏：画一个 n×n，只有少数格子深
    ox, oy = RX + 24, by + 26
    CELL, N = 9, 9
    for r in range(N):
        for c in range(N):
            if c > r:
                continue                      # 因果遮罩
            hot = (c == 0) or (c >= r - 1) or (r == 6 and c == 2)
            f.box(ox + c * CELL, oy + r * CELL, CELL - 1.5, CELL - 1.5,
                  "#1e8e3e" if hot else "#e6f4ea", "none", 1)
    f.t(ox + N * CELL + 16, oy + 18,
        '① <tspan font-weight="700">实测极其稀疏</tspan>', INK, size=_sz(12))
    f.t(ox + N * CELL + 16, oy + 36,
        "绝大部分权重集中在很少的位置，其余近乎为零。", GY, size=11)
    f.t(ox + N * CELL + 16, oy + 54,
        "那把近零的那些算出来，算的是什么？", GY2, size=11)

    # ② attention sink：第一列恒亮
    o2 = oy + N * CELL + 24
    for c in range(N):
        f.box(ox + c * CELL, o2, CELL - 1.5, CELL - 1.5,
              "#1e8e3e" if c == 0 else "#e6f4ea", "none", 1)
    f.t(ox + N * CELL + 16, o2 + 8,
        '② <tspan font-weight="700">Attention sink</tspan>', INK, size=_sz(12))
    f.t(ox + N * CELL + 16, o2 + 26,
        "注意力被大量「停放」在开头几个 token 上，<tspan font-weight=\"700\">跟内容无关</tspan>",
        GY, size=11)
    f.t(ox + N * CELL + 16, o2 + 44,
        "——&#160;说明有一部分权重根本不是在做检索。", GY2, size=11)

    # ③ 远近有别：近密远疏
    o3 = o2 + 66
    for c in range(18):
        dens = 1.0 if c > 13 else (0.45 if c > 7 else 0.18)
        f.box(ox + c * 7, o3, 5.5, 13, "#1e8e3e", "none", 1)
        f.box(ox + c * 7, o3, 5.5, 13 * (1 - dens), "#fff", "none", 1)
    f.t(ox, o3 + 26, "远 ←", GY2, size=11)
    f.t(ox + 18 * 7 - 22, o3 + 26, "→ 近", GY2, size=11)
    f.t(ox + 18 * 7 + 22, o3 + 6,
        '③ <tspan font-weight="700">远近有别</tspan>', INK, size=_sz(12))
    f.t(ox + 18 * 7 + 22, o3 + 24,
        "邻近几十个 token 密集细粒度；几万之外稀疏粗粒度。", GY, size=11)
    f.t(ox + 18 * 7 + 22, o3 + 42,
        "<tspan font-weight=\"700\">凭什么用同一套精度处理这两种？</tspan>", GY2, size=11)

    f.t(RX + 20, by + PH - 66,
        '⭐ 于是 <tspan font-weight="700">压缩</tspan>（远处多个合并成一个）、'
        '<tspan font-weight="700">稀疏</tspan>（只挑相关的看）、'
        '<tspan font-weight="700">分层</tspan>（近精细远粗糙）都变得合理。', GR, size=_sz(12))

    yy = y + PH + 18

    # ══════════ 底部：交汇处 ══════════
    yy = f.band(yy, "info", "两条缺一不可 ——&#160;所有变体都活在它们的交汇处", [
        '<tspan font-weight="700">只有 A</tspan>（必须省）：你得到的是一堆有损压缩的权宜之计，'
        '<tspan font-weight="700">效果掉了只能认</tspan>。',
        '<tspan font-weight="700">只有 B</tspan>（可以省）：你<tspan font-weight="700">'
        '没有动力</tspan>去付 kernel 那么难写的代价。',
        '⭐⭐ <tspan font-weight="700">两条合起来</tspan>，才解释了为什么这个方向三年里投了这么多人'
        '——&#160;它同时是一件<tspan font-weight="700">不得不做</tspan>和'
        '<tspan font-weight="700">做了不太亏</tspan>的事。',
        '⭐ 所以判一个变体好不好，就看它在'
        '<tspan font-weight="700">「省了多少（A）」和「亏了多少（B）」</tspan>之间落在哪。'])

    yy = f.src(yy + 16,
               "左栏四行推导与四个对照值均按公式当场算出（脚本里带断言）；"
               "口径沿用 V3 论文比较表（K、V 都按 d_h=128）——&#160;"
               "<tspan font-weight=\"700\">V3 真实的 K 每头是 128+64=192 维，严格算这个基线还会更大</tspan>",
               "⚠️ 右栏三个观察是<tspan font-weight=\"700\">现象</tspan>，不是本课实测；"
               "它们各自对应后面的一支方案（稀疏 →&#160;§六、分层 →&#160;§六、压缩 →&#160;§五）")
    f.save("fig3-motives.svg", yy + 6)


main()
