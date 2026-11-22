# -*- coding: utf-8 -*-
r"""专题九 · 第四节「找关键路径」的两张图。

fig8b-heavylight  回答：注意力省下来的时间去哪了？—— 同一配置，最忙的一颗芯片对轻的一颗：注意力、等专家权重通信、其余三段。
                  数字：topic08b_numbers（两份 profile 不是同一步，标推断）。
fig8b-maxsum      回答：两件事同时干，提速一件为什么只兑现一部分？—— 示意小账：师傅 10、小工 7，师傅提速到 6，一轮只到 7。
fig8b-cover       回答：为什么省了注意力，主核反而在等？—— 示意：一层里主核和小核两条线，注意力长时把专家权重那条链盖住，短了就露出来。
"""
import topic08_figfont  # noqa: F401
from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE, LINE2, PU
Fig.BAND_KEEP_LINES = 0
import topic08b_numbers as N

W = 1400


def fig_heavylight():
    a0, a1 = N.HL_ATTN
    w0, w1 = N.HL_WAIT
    r0, r1 = N.HL_HEAVY_STEP - a0 - w0, N.HL_LIGHT_STEP - a1 - w1
    f = Fig(W, "同一配置下两颗芯片的一步，按三段堆起来。注意力最重的那颗芯片：注意力 896 毫秒，等专家权重通信 518 毫秒。"
               "注意力很轻的那颗：注意力 223 毫秒，等专家权重通信 1059 毫秒。注意力少了约 672 毫秒，等待多了 541 毫秒，一步只短 195 毫秒。")
    y0 = f.header("注意力省下来的时间，大半变成了等专家权重的通信",
                  "条长 ＝ 一步里主核的时间（毫秒）；上一行是注意力最重的芯片，下一行是很轻的芯片，配置相同",
                  [(BL, "注意力"), (RD, "等专家权重的收集与梯度规约"), (LINE2, "其余")])
    PH = 280
    py = f.panel(0, y0, W, PH, "同一配置、原数据，两份 profile 里的一步（实测）", BL)
    X0, SC = 300, 1000 / 5600.0
    for i, (lab, a, w, r, step) in enumerate([("最忙的芯片（66 个子块）", a0, w0, r0, N.HL_HEAVY_STEP),
                                               ("轻的芯片（7 个子块）", a1, w1, r1, N.HL_LIGHT_STEP)]):
        yy = py + 60 + i * 90
        f.t(X0 - 14, yy + 25, lab, INK, True, 15, "end")
        x = X0
        f.box(x, yy, a * SC, 40, BL, BL, 3)
        f.t(x + a * SC / 2, yy + 26, "%d" % a, "#fff", True, 15, "middle")
        x += a * SC
        f.box(x, yy, w * SC, 40, RD, RD, 3)
        f.t(x + w * SC / 2, yy + 26, "%d" % w, "#fff", True, 15, "middle")
        x += w * SC
        f.box(x, yy, r * SC, 40, LINE2, GY2, 3)
        f.t(x + r * SC + 10, yy + 26, "一步 %.0f" % step, INK, True, 15)
    f.t(X0, py + 238, "注意力 −%d，等待 ＋%d，一步只 −%d" % (N.ATTN_SAVED_MS, w1 - w0, N.HL_STEP_DIFF), RD, True, 15)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "省下的注意力只有「超出它所盖住的通信」那一截能变成步时", [])
    yb = f.src(yb + 10,
               "📌 实测：同一配置、原数据。最忙的那份是第 11 步、这颗芯片正好是这一步注意力最重的设备（66 个子块）；轻的那份是另一个作业的第 20 步、这颗芯片只有 7 个子块。各段是主核叶子算子的自身时间按类别加总。",
               "⚠️ 推断：两份不是同一步，只当量级；一步 %.1f 与 %.1f ms 里「其余」是减出来的。小核的忙时两份一样（小核 0 约 26%%、小核 1 约 42%%）。多芯片同一步的 profile 已取回，待拆来坐实。" % (N.HL_HEAVY_STEP, N.HL_LIGHT_STEP))
    f.save("fig8b-heavylight.svg", yb + 14)


def fig_cover():
    f = Fig(W, "示意图，两种情形各两条线。上面：注意力很长，主核在算注意力的时候，小核上专家权重的收集和梯度规约同时在跑，做完时主核还没算完，主核不用等。"
               "下面：注意力变短了，主核很快算完，要用的专家权重还没收完，只能停下来等；小核上那条链一点没变短。")
    y0 = f.header("为什么省了注意力，主核反而在等：那条链本来就压在注意力底下",
                  "示意，一层；上面一组是注意力长的时候，下面一组是注意力短的时候；小核那条链两组一样长",
                  [(BL, "主核：注意力"), (LINE2, "主核：别的计算"), (RD, "主核：停着等"), (OR, "小核：专家权重收集与梯度规约")])
    PH = 330
    py = f.panel(0, y0, W, PH, "示意（长度不按比例）", BL)
    X0 = 260
    CH = 560   # 链长
    for k, (title, attn, wait) in enumerate([("注意力长", 640, 0), ("注意力短", 220, 340)]):
        yb_ = py + 40 + k * 140
        f.t(30, yb_ + 30, title, INK, True, 16)
        f.t(X0 - 14, yb_ + 24, "主核", INK, True, 14, "end")
        f.t(X0 - 14, yb_ + 74, "小核", INK, True, 14, "end")
        f.box(X0, yb_ + 6, attn, 30, BL, BL, 3)
        x = X0 + attn
        if wait:
            f.box(x, yb_ + 6, wait, 30, RD, RD, 3)
            f.t(x + wait / 2, yb_ + 27, "等", "#fff", True, 15, "middle")
            x += wait
        f.box(x, yb_ + 6, 260, 30, LINE2, GY2, 3)
        f.box(X0, yb_ + 56, CH, 30, OR, OR, 3)
        f.t(X0 + CH / 2, yb_ + 77, "专家权重的收集与梯度规约", "#fff", True, 14, "middle")
        f.line(X0 + CH, yb_ + 52, X0 + CH, yb_ + 2, GY2, 1.2, dash="4 4", arrow=False)
    f.t(X0 + CH + 12, py + 40 + 140 + 2, "主核要用它的结果，得等它做完", RD, True, 14)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "砍掉的计算如果原来盖着一条通信，砍完以后主核只是换个地方等", [])
    yb = f.src(yb + 10,
               "⚠️ 示意：长度不按比例；一层里实际还有很多段计算和通信，这里只画注意力和专家权重那条链。最重的芯片本来也有一截露在外面（fig-heavylight 里的 518 ms），图上只画原来被注意力盖住、后来露出来的那一截。",
               "📌 对上实测的量级：注意力省约 672 ms、等待多 541 ms，净约 130 ms；日志步时实际快 148 ms（本课推导：两个数不是同一个口径，只对量级）。")
    f.save("fig8b-cover.svg", yb + 14)


def fig_maxsum():
    f = Fig(W, "示意小账，三行。第一行：师傅一轮的活 10 分钟，小工同时跑腿 7 分钟，一轮 10 分钟。"
               "第二行是以为：师傅提速到 6 分钟，一轮就是 6 分钟。第三行是其实：师傅 6 分钟做完，小工 7 分钟才回来，师傅等 1 分钟，一轮 7 分钟。省了 4 分钟，只兑现 3 分钟。")
    y0 = f.header("同时干的两件事，一轮花的是长的那件：提速一件，只兑现到另一件为止",
                  "示意小账（数是编的，好算）；条长 ＝ 分钟",
                  [(BL, "师傅在干活"), (OR, "小工跑腿"), (RD, "师傅停着等")])
    PH = 400
    py = f.panel(0, y0, W, PH, "一轮：师傅干活和小工跑腿同时开始，师傅要等小工回来才能进下一轮", BL)
    X0, SC = 330, 70
    rows = [("原来", 10, 7, None, "一轮 10 分钟", INK),
            ("以为：师傅提速到 6", 6, None, None, "一轮 6 分钟？", GY),
            ("其实", 6, 7, 1, "一轮 7 分钟", RD)]
    for i, (lab, m, h, wait, res, col) in enumerate(rows):
        yy = py + 50 + i * 105
        f.t(X0 - 14, yy + 26, lab, col if i == 1 else INK, True, 16, "end")
        dash = i == 1
        f.box(X0, yy, m * SC, 34, LINE2 if dash else BL, GY2 if dash else BL, 3)
        f.t(X0 + m * SC / 2, yy + 23, "师傅 %d" % m, INK if dash else "#fff", True, 15, "middle")
        end = m
        if wait:
            f.box(X0 + m * SC, yy, wait * SC, 34, RD, RD, 3)
            f.t(X0 + m * SC + wait * SC / 2, yy + 23, "等 %d" % wait, "#fff", True, 15, "middle")
            end = m + wait
        if h:
            f.box(X0, yy + 42, h * SC, 26, OR, OR, 3)
            f.t(X0 + h * SC / 2, yy + 61, "小工跑腿 %d" % h, "#fff", True, 14, "middle")
        f.t(X0 + max(end, h or 0) * SC + 16, yy + 26, res, col, True, 16)
    f.t(X0, py + PH - 36, "省了 10 − 6 ＝ 4 分钟，只兑现 10 − 7 ＝ 3 分钟；兑现的上限是「原来的长度减去另一件的长度」", RD, True, 15)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "同时干，花的是长的那件；提速到比另一件还短，多省的都变成了等", [])
    yb = f.src(yb + 10,
               "⚠️ 示意：分钟数是为好算编的，不是本课的测量。",
               "📌 对应本课：师傅那条是注意力（省约 672 ms），小工那条是专家权重的收集与梯度规约（露出 541 ms），兑现约 130 ms（5.2）；轻的那颗一步短了 195 ms，比 130 多的那 60 来毫秒落在别处（两颗不是同一步，只对量级）。")
    f.save("fig8b-maxsum.svg", yb + 14)


if __name__ == "__main__":
    fig_heavylight()
    fig_maxsum()
    fig_cover()
