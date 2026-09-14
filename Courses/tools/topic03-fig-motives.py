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
    # ⛔⛔ 2026-09-13 学生审稿抓到两处单位/口径错，两处都很典型：
    #
    # ① **94.74 已经是 GiB，不是字节。** 原来写 `94.74e9 / GiB`，
    #    等于把一个 GiB 数当成 GB 又转了一次 → 88.23，比真值小 7%。
    #    ⭐ 专题二有一整张图在推这个数：192 GiB/chip ÷ 2 = 96.00 GiB/device，
    #      减去约 1.26 GiB 运行时预留 = **94.74 GiB 可分配**。全课统一用它。
    #
    # ② **V3 的权重原生是 FP8，不是 BF16。** 按 2 字节算出来的 1250 GiB
    #    是一个「从来不会被这么部署」的数。按 FP8 算是 625 GiB ——
    #    ⭐ 而且这个修正**对论点有利**：权重更小，KV 反而更快压过它。
    hbm = 94.74                       # GiB，可分配值（专题二 §附录 A 推导）
    wgt = 671 * 1e9 / GiB             # V3 原生 FP8：671B × 1 B/参数
    assert abs(mha - 488) < 1 and abs(gqa - 30.5) < .1 and abs(mla - 8.58) < .05

    f = Fig(W, "长上下文的两条独立动机：左边是硬件账 —— 一个用户 128K 的 KV cache "
               "按最朴素的 MHA 算是 488 GiB，要 5.2 块 v7；右边是信息账 —— "
               "注意力矩阵极稀疏、有 attention sink、远近关注粒度不同；"
               "两条线交汇处才是所有变体的位置")
    f.marks = set()
    y = f.header(
        '为什么是现在 ——&#160;<tspan font-weight="700">两条完全独立的线，'
        '在同一个地方交汇</tspan>',
        "⛔ 这两条必须分开讲：它们指向同一批技术，但出发点完全不同 ——&#160;混在一起讲就成了名词罗列")

    # ⭐⭐⭐ 2026-09-13 重排：原来是**左右两栏各 672px**，于是字只能给到 11–14px。
    # ⛔ 判据跟 arc 那张一样：**栏宽是可读性定的，不是内容条数定的。**
    #   两条线索没有必须并排的理由 —— 改成上下两整行，每行 1400px，字抬到 16–19。
    # ⛔ 这个高度要**盖得住实际画到哪儿**（阶梯 4 行 ＋ 对照 4 条 ＋ 一行提示）。
    #   声明的高度和真实内容对不上，下一个面板就会被压上来。
    PH = 400

    # ══════════ 上：线索 A · 硬件账（必须省）══════════
    ay = f.panel(0, y, W, PH, "线索 A · 硬件账算不过来", RD,
                 sub="结论：必须省。不解决它，长上下文根本上不了线")
    # ⭐⭐⭐ 2026-09-15 R5 重画这一格，两处都是「图在说另一件事」：
    #
    # ⛔ ① 原来上半是**一条四行乘法链**（32,768 → 1,998,848 → 3.81 MiB → 488 GiB）。
    #    那是算术，不是画。⭐ 但「小东西 × 很多个 ＝ 天文数字」这个**体感**要留住 ——
    #    所以只留最外面那一步画出来（一个小方块 × 131,072），
    #    中间三步整条折进「出处与口径」。
    #
    # ⛔⛔ ② 更要命的一条：原来那四根对照条里**没有 488 这一根**。
    #    标题写着「摆四个对照，让这个数站住」，可**被对照的那个数只是上方的一行字** ——
    #    读者得拿脑子里的一个数去跟四根条子比。
    #    ⭐ 判据：**参照系里必须包含被参照的那个东西**，否则它不叫对照，叫并列。
    #
    # ⛔ ③ 那四根条子还是 log10 画的、而且没有轴。可这一格要说的正是
    #    「488 装不进一块卡、8.58 装得进」——&nbsp;对数会把这件事压平。
    #    ⭐ 改成**换单位**：一格 ＝ 一块 v7 的 HBM。于是它天然是线性的，
    #      而且「小的那两个只占一格里的一小条」本身就是结论，不是缺陷。
    f.t(20, ay + 10, "拿 DeepSeek V3 砸体感（61 层 · 128 头 · 每头 128 维 · 128K · bf16）",
        GY, size=16)

    # ── 种子：一个 token 有多小，乘出来有多大 ──────────────────
    per_mib = per_tok * B / 2 ** 20
    sy = ay + 46
    f.box(24, sy, 14, 14, "#fce8e6", RD, 3, 1.4)      # 一个 token 那一点点
    f.t(48, sy + 12, "一个 token 要存 <tspan font-weight=\"700\">%.2f MiB</tspan>"
        % per_mib, GY, size=17)
    f.path([(300, sy + 7), (352, sy + 7)], RD, 2.0)
    f.t(364, sy + 12, "× <tspan font-weight=\"700\">131,072</tspan> 个 token",
        GY, size=17)
    f.path([(560, sy + 7), (612, sy + 7)], RD, 2.0)
    f.t(624, sy + 12, "<tspan font-weight=\"700\">%.0f GiB</tspan>　一个用户" % mha,
        RD, True, 19)
    f.t(880, sy + 12, "（中间那几步在「出处与口径」里）", GY2, size=14)

    # ── 换个单位：一格 ＝ 一块 v7 的 HBM ──────────────────────
    cy = sy + 40
    f.t(24, cy + 14, "换个单位就看得见了 ——　"
        "<tspan font-weight=\"700\">一格 ＝ 一块 v7 device 的 HBM"
        "（%.2f GiB 可分配）</tspan>" % hbm, GY, size=16)
    CW_, CHH, CG, SLOT = 84, 34, 5, 7
    X0 = 340
    REF = [("这一个用户的 KV cache", mha, RD, "⛔ 一个人就要 5.2 块"),
           ("V3 全部权重（671B，原生 FP8）", wgt, BL, "所有人共享这一份"),
           ("换成 GQA-8", gqa, OR, "省 16 倍 ——　还是装不进一块"),
           ("换成 MLA（V3 真实方案）", mla, GR, "省 56.9 倍 ——　这才塞得下")]
    assert max(v for _, v, _, _ in REF) / hbm < SLOT, "最长那条超出 7 格了"
    for i, (nm, v, c_, note) in enumerate(REF):
        yy_ = cy + 36 + i * (CHH + 10)
        f.t(24, yy_ + CHH / 2 + 6, nm, INK, size=17, w=300)
        n = v / hbm                                  # 占几块卡
        for k in range(SLOT):
            f.box(X0 + k * (CW_ + CG), yy_, CW_, CHH, "#fff", LINE2, 4)
            frac = min(1.0, max(0.0, n - k))         # 这一格填多少
            if frac > 0:
                f.box(X0 + k * (CW_ + CG), yy_, max(CW_ * frac, 2), CHH,
                      c_, "none", 4)
        VX = X0 + SLOT * (CW_ + CG) + 84
        f.t(VX, yy_ + CHH / 2 + 6,
            "%.2f GiB" % v if v < 100 else "%.0f GiB" % v,
            c_, bold=True, size=16, mono=True, anchor="end")
        # ⛔ 带上 w= ——&#160;不带的话越界不报，得靠肉眼在截图里发现
        f.t(VX + 12, yy_ + CHH / 2 + 6, note, GY2, size=16, w=1384 - VX - 12)
    f.t(24, cy + 36 + 4 * (CHH + 10) + 18,
        '⛔ <tspan font-weight="700">权重是所有用户共享一份，KV cache 是每人一份</tspan>'
        '——&#160;所以它直接决定<tspan font-weight="700">你能同时服务多少人</tspan>。'
        '　　⚠️ 488 是「假如 V3 用 MHA」的反事实值，不是实测。',
        RD, size=17)

    # ══════════ 下：线索 B · 信息账（可以省而不太亏）══════════
    y2 = y + PH + 18
    PH2 = 384
    by = f.panel(0, y2, W, PH2, "线索 B · 信息本身不需要那么多", GR,
                 sub="结论：可以省，而且不太亏")
    f.t(20, by + 10, "128K 的序列，真需要 128K 份独立的 KV 吗？三个观察 ——",
        GY, size=16)

    CW2 = 452
    oy = by + 40
    # ① 稀疏：一张因果三角，只有少数格子是深的
    ox = 24
    CELL, N = 15, 9
    for r in range(N):
        for c in range(N):
            if c > r:
                continue                      # 因果遮罩
            hot = (c == 0) or (c >= r - 1) or (r == 6 and c == 2)
            f.box(ox + c * CELL, oy + r * CELL, CELL - 2, CELL - 2,
                  "#1e8e3e" if hot else "#e6f4ea", "none", 1)
    f.t(ox, oy + N * CELL + 26, '① <tspan font-weight="700">实测极其稀疏</tspan>',
        INK, size=19)
    f.t(ox, oy + N * CELL + 52, "绝大部分权重集中在很少的位置，", GY, size=17)
    f.t(ox, oy + N * CELL + 76, "其余近乎为零。", GY, size=17)
    f.t(ox, oy + N * CELL + 104, "那把近零的那些算出来，算的是什么？",
        GY2, size=17)

    # ② attention sink：第一列恒亮
    ox2 = 24 + CW2 + 22
    for r in range(4):
        for c in range(N):
            f.box(ox2 + c * CELL, oy + r * CELL, CELL - 2, CELL - 2,
                  "#1e8e3e" if c == 0 else "#e6f4ea", "none", 1)
    f.t(ox2 + N * CELL + 18, oy + 30, "↑ 这一列", GR, bold=True, size=17)
    f.t(ox2 + N * CELL + 18, oy + 54, "总是亮的", GR, bold=True, size=17)
    f.t(ox2, oy + N * CELL + 26, '② <tspan font-weight="700">Attention sink</tspan>',
        INK, size=19)
    f.t(ox2, oy + N * CELL + 52, "注意力被大量「停放」在开头几个 token 上，",
        GY, size=17)
    f.t(ox2, oy + N * CELL + 76,
        "<tspan font-weight=\"700\">跟内容无关</tspan>。", GY, size=17)
    f.t(ox2, oy + N * CELL + 104, "有一部分权重根本不是在做检索。", GY2, size=17)

    # ③ 远近有别：近密远疏
    ox3 = 24 + 2 * (CW2 + 22)
    for c in range(18):
        dens = 1.0 if c > 13 else (0.45 if c > 7 else 0.18)
        f.box(ox3 + c * 13, oy + 20, 10, 40, "#1e8e3e", "none", 1)
        f.box(ox3 + c * 13, oy + 20, 10, 40 * (1 - dens), "#fff", "none", 1)
    f.t(ox3, oy + 78, "远 ←", GY2, size=17)
    f.t(ox3 + 18 * 13 - 34, oy + 78, "→ 近", GY2, size=17)
    f.t(ox3, oy + N * CELL + 26, '③ <tspan font-weight="700">远近有别</tspan>',
        INK, size=19)
    f.t(ox3, oy + N * CELL + 52, "邻近几十个 token 密集细粒度；", GY, size=17)
    f.t(ox3, oy + N * CELL + 76, "几万之外稀疏粗粒度。", GY, size=17)
    f.t(ox3, oy + N * CELL + 104,
        "<tspan font-weight=\"700\">凭什么用同一套精度处理这两种？</tspan>",
        GY2, size=17)

    f.t(24, y2 + PH2 - 22,
        '⭐ 于是这三样都变得合理：<tspan font-weight="700">压缩</tspan>'
        '（远处多个合并成一个）　·　<tspan font-weight="700">稀疏</tspan>'
        '（只挑相关的看）　·　<tspan font-weight="700">分层</tspan>（近精细远粗糙）',
        GR, size=17)

    yy = y2 + PH2 + 18

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
               "📐 488 GiB 那条乘法链（原来画在图上，2026-09-15 折到这里）："
               "每 token 每层 2 × 128 × 128 ＝ 32,768 个数 →&#160;× 61 层 ＝ "
               "1,998,848 个数 →&#160;× 2 字节 ＝ 3.81 MiB/token →&#160;"
               "× 131,072 token ＝ <tspan font-weight=\"700\">488 GiB</tspan>",
               "四个对照值均按公式当场算出（脚本里带断言）；"
               "口径沿用 V3 论文比较表（K、V 都按 d_h=128）——&#160;"
               "<tspan font-weight=\"700\">V3 真实的 K 每头是 128+64=192 维，严格算这个基线还会更大</tspan>",
               "⚠️ 右栏三个观察是<tspan font-weight=\"700\">现象</tspan>，不是本课实测；"
               "它们各自对应后面的一支方案（稀疏 →&#160;§六、分层 →&#160;§六、压缩 →&#160;§五）")
    f.save("fig3-motives.svg", yy + 6)


main()
