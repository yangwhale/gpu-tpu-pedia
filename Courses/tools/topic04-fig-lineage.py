# -*- coding: utf-8 -*-
r"""专题四 · §3.4「谱系」——&#160;AdaGrad → RMSProp → Adam，一条**因果链**

⭐⭐⭐ 2026-09-17 R06/20 新画。这是「二十轮重做」查出的第三个真空洞：
  这一讲原来只有一根「每参数几字节」的条，**没有因果链** ——&#160;
  读完只知道「有这么几个优化器」，不知道**谁解决了谁的什么问题、又留下了什么**。
  ⛔ 而没有因果链，这一串名字就只能靠背。

⭐⭐ 这一格的每一环都有出处，而且是 **Adam 原文自己写的**（已读原文）：
  · AdaGrad 的更新式，原文逐字：**θ_{t+1} ＝ θ_t −&#160;α·g_t / √(Σ_{i=1..t} g_i²)**
    ——&#160;分母是**累加的平方和**（不除以 t）。⭐ 所以它只增不减，
    有效学习率**单调衰减**。这个病是从式子里长出来的，不是我说的。
  · Adam 的定位，原文：**combine the advantages of AdaGrad**（擅长稀疏梯度）
    **and RMSProp**（擅长在线与非平稳设定）。
  · 偏差校正的理由，原文：滑动平均从 0 起步，于是估计**biased towards zero**，
    **especially during the initial timesteps**，而且 **β 越接近 1 越严重**。
  · ⭐⭐⭐ 原文还顺手证了一件事：**AdaGrad 是 Adam 的一个特例**
    （β₁ ＝ 0、(1−β₂) 取无穷小、α 按 t^(−1/2) 退火）。
    ——&#160;**一条链最强的证据，就是后面那个能把前面那个当特例含进去。**
  · 而同一段还点名了 RMSProp 的欠缺：**without bias correction, like in RMSProp**，
    β₂ 越接近 1 偏差越大。

⭐ 按这一轮的判据画：**把文字全删掉，还剩四条 + 两组真算出来的曲线**。
  Ⓐ 的四个节点不是四个方框，是**四条缩略曲线**；Ⓑ Ⓒ 是放大的两张。

⛔ 一条口径写在前面：Ⓑ Ⓒ **三条线用的是同一个 β₂**，
  为的是把「有没有遗忘」「有没有偏差校正」这两个变量**单独拎出来**。
  RMSProp 实际常用的衰减率没这么接近 1 ——&#160;这里不是在报它的真实配置。
"""
import math

from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE)

W = 1400

# ══════════════════════════════════════════════════════════════════
# ⭐ 全部用同一串**恒定梯度**喂进去，让四种做法的「有效学习率」自己跑出来。
#   恒定是故意的：梯度不变的时候，一个健康的优化器应当给出**平的**有效学习率。
#   凡是不平的，那就是它自己的毛病，赖不到数据头上。
# ══════════════════════════════════════════════════════════════════
B1, B2 = 0.9, 0.999          # Adam 的两个默认值
T = 600
G = 1.0                      # 恒定梯度


def eff_sgd(t):
    return 1.0


def eff_adagrad(t):
    """θ ←&#160;θ −&#160;α·g/√(Σg²)。恒定梯度下 Σ ＝ t·g²，所以 ∝ 1/√t。"""
    return 1.0 / math.sqrt(t)


def eff_rmsprop(t):
    """分母换成滑动平均，但**不做偏差校正**（原文点名 RMSProp 就是这样）。"""
    v = 1.0 - B2 ** t        # 恒定梯度下 EMA(g²) 的闭式
    return 1.0 / math.sqrt(v)


def eff_adam(t):
    """m 和 v 都做偏差校正 ——&#160;恒定梯度下应当**恒为 1**。"""
    m = (1.0 - B1 ** t) / (1.0 - B1 ** t)
    v = (1.0 - B2 ** t) / (1.0 - B2 ** t)
    return m / math.sqrt(v)


def eff_adam_nofix(t):
    """如果 Adam 不做偏差校正会怎样 ——&#160;m 的偏差会**部分抵消** v 的偏差。"""
    return (1.0 - B1 ** t) / math.sqrt(1.0 - B2 ** t)


# ⭐⭐ 图上那几个数，全是这儿算出来的
ADA_DECAY = eff_adagrad(T)                      # 跑到第 T 步只剩多少
RMS_SPIKE = eff_rmsprop(1)                      # 第一步被放大多少倍
NOFIX_PEAK_T = max(range(1, 400), key=eff_adam_nofix)
NOFIX_PEAK = eff_adam_nofix(NOFIX_PEAK_T)

assert ADA_DECAY < 0.05, "AdaGrad 跑 %d 步要掉到很小，现在还有 %.3f" % (T, ADA_DECAY)
assert all(eff_adagrad(t + 1) < eff_adagrad(t) for t in range(1, 200)), \
    "AdaGrad 的有效学习率必须**单调**衰减 ——　这就是它的病"
assert RMS_SPIKE > 25, "不做偏差校正，第一步该被放大很多倍，现在只有 %.1f" % RMS_SPIKE
assert abs(eff_adam(1) - 1.0) < 1e-12 and abs(eff_adam(T) - 1.0) < 1e-12, \
    "做了偏差校正，恒定梯度下必须从第一步起就恒为 1"
assert 1 < NOFIX_PEAK < RMS_SPIKE, \
    "Adam 不校正时 m 的偏差应当**部分抵消** v 的偏差，峰值要比 RMSProp 那条低"

# Ⓐ 的链：(名字, 年份, 它解决了上一环的什么, 它自己又留下什么, 曲线, 颜色)
CHAIN = (
    ("SGD", "", "", "⛔ 全局一个学习率\n尺度差大的参数伺候不了", eff_sgd, GY2),
    ("AdaGrad", "2011", "除以「它自己历史梯度的均方根」\n→ 每个参数一个尺度",
     "⛔ 分母只增不减\n→ 有效学习率单调掉到 0", eff_adagrad, RD),
    ("RMSProp", "2012", "累加换成滑动平均\n→ 分母会遗忘，不再单调掉",
     "⛔ 没动量，也没偏差校正\n→ 头几步冲得极高", eff_rmsprop, OR),
    ("Adam", "2014", "＋动量 ＋偏差校正\n→ 方向也平滑了，开头也稳了",
     "⭐ 恒定梯度下从第一步起就是平的", eff_adam, GR),
)


def main():
    f = Fig(W, "这一串优化器不是并列的四个选项，是一条因果链："
               "每一个都在补上一个留下的洞，同时又留下新的洞。"
               "喂同一串恒定的梯度进去，一个健康的优化器应该给出平的有效学习率；"
               "不平的那就是它自己的毛病。"
               "AdaGrad 的分母是累加的平方和，只增不减，"
               "所以有效学习率单调掉到接近零，后期走不动；"
               "RMSProp 把累加换成滑动平均，分母会遗忘了，可它没有偏差校正，"
               "头一步被放大三十多倍；"
               "Adam 把动量和偏差校正都加上，恒定梯度下从第一步起就是平的。"
               "而 Adam 论文自己证明了 AdaGrad 是它的一个特例")

    y0 = f.header(
        "从 AdaGrad 到 Adam　——　<tspan font-weight=\"700\">"
        "这不是四个选项，是一条因果链</tspan>",
        "⭐ 喂<tspan font-weight=\"700\">同一串恒定梯度</tspan>，"
        "看各自给出的「有效学习率」——&#160;"
        "<tspan font-weight=\"700\">健康的应该是一条平线</tspan>",
        [(RD, "AdaGrad：掉到 0"), (OR, "RMSProp：开头冲太高"), (GR, "Adam：平")])

    # ══════════ Ⓐ 四环的链 ═══════════════════════════════════════
    PH = 452
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 每一环<tspan font-weight=\"700\">补上一环的洞</tspan>，"
                 "同时<tspan font-weight=\"700\">留下一个新的</tspan>", INK,
                 sub="⭐ 每个节点里那条小曲线，"
                     "就是它给出的有效学习率（<tspan font-weight=\"700\">真跑的</tspan>）")

    CW, CGAP, CX0 = 306, 36, 42
    for i, (name, yr, fixes, leaves, fn, col) in enumerate(CHAIN):
        x = CX0 + i * (CW + CGAP)
        # 上一环留给它的洞 → 这一环怎么补：画在两节点之间
        if i:
            mx = x - CGAP / 2 - 4
            f.line(x - CGAP - 4, py + 150, x - 6, py + 150, INK, 2.2)
            for k, ln in enumerate(fixes.split("\n")):
                f.t(mx, py + 56 + k * 20, ln, GR, size=11.5, anchor="middle")

        f.box(x, py + 176, CW, 136, "#fff", col, 8, sw=1.8)
        f.box(x, py + 176, CW, 4, col, col, 2)
        f.t(x + 16, py + 206, name, col, True, 19)
        if yr:
            f.t(x + CW - 16, py + 206, yr, GY2, size=13, anchor="end")

        # ⭐ 节点里那条小曲线 ——&#160;这才是「不是写字板」的那部分
        gx, gy, gw, gh = x + 18, py + 296, CW - 36, 70
        f.line(gx, gy, gx + gw, gy, GY2, 1, arrow=False)
        f.line(gx, gy - gh - 4, gx, gy, GY2, 1, arrow=False)
        f.line(gx, gy - gh / 2.6, gx + gw, gy - gh / 2.6, GY2, 1,
               dash="2 3", arrow=False)
        d = None
        for k in range(121):
            tt = 1 + (T - 1) * (k / 120.0) ** 2.2        # 前段采密一点
            v = min(fn(tt), 2.6)
            px = gx + gw * k / 120.0
            pyv = gy - gh * (v / 2.6)
            d = ("M %.1f %.1f" % (px, pyv)) if d is None else d + " L %.1f %.1f" % (px, pyv)
        f.path(d, col, 2.4, arrow=False)

        for k, ln in enumerate(leaves.split("\n")):
            f.t(x + CW / 2, py + 392 + k * 20, ln,
                GR if leaves.startswith("⭐") else RD, True, 12.5, "middle")

    f._pan = None

    # ══════════ Ⓑ AdaGrad 的病：后期 ═══════════════════════════════
    PH2 = 360
    py2 = f.panel(0, py + PH + 20, W, PH2,
                  "Ⓑ ⭐⭐ AdaGrad 的病在<tspan font-weight=\"700\">后期</tspan>"
                  "　——　分母只增不减，学习率<tspan font-weight=\"700\">自己走向 0</tspan>", RD,
                  sub="⭐ 横轴是步数，纵轴是有效学习率（稳态 ＝ 1）")

    OX, OY, AW, AH = 150, py2 + 274, 1080, 200
    f.line(OX, OY, OX + AW + 30, OY, GY2, 1.2, arrow=False)
    f.line(OX, OY, OX, OY - AH - 20, GY2, 1.2, arrow=False)
    f.line(OX, OY - AH * 0.5, OX + AW, OY - AH * 0.5, GY2, 1, dash="3 4", arrow=False)
    f.t(OX - 12, OY - AH * 0.5 + 5, "1", GY2, size=12, anchor="end")
    f.t(OX - 12, OY + 5, "0", GY2, size=12, anchor="end")
    f.t(OX + AW, OY + 26, "第 %d 步" % T, GY2, size=12, anchor="end")

    # ⛔ 这里原来把 RMSProp 也画进来了 ——&#160;可它那条是**初期偏差的尾巴**，
    #   放在「后期」这一格里会被读成「RMSProp 也在后期衰减」，正好读反。
    #   ⭐ 判据：**一格只讲一件事；对照组要选「没有这个病」的那个，
    #     不要顺手放一个「有另一种病」的进来。**
    for fn, col, lab in ((eff_sgd, GY2, "SGD（对照）"), (eff_adagrad, RD, "AdaGrad")):
        d = None
        for k in range(241):
            tt = 1 + (T - 1) * k / 240.0
            v = min(fn(tt), 2.0)
            px = OX + AW * k / 240.0
            pyv = OY - AH * 0.5 * v
            d = ("M %.1f %.1f" % (px, pyv)) if d is None else d + " L %.1f %.1f" % (px, pyv)
        f.path(d, col, 2.6, arrow=False)
        f.t(OX + AW + 14, OY - AH * 0.5 * min(fn(T), 2.0) + 5, lab, col, True, 15)

    f.line(OX + AW * 0.62, OY - AH * 0.5 * eff_adagrad(T * 0.62) - 10,
           OX + AW * 0.62, OY - 92, RD, 1.6)
    f.t(OX + AW * 0.62, OY - 104,
        "⛔ 跑到第 %d 步只剩稳态的 <tspan font-weight=\"700\">%.1f%%</tspan>"
        % (T, ADA_DECAY * 100), RD, True, 14.5, "middle")

    f.t(700, py2 + 326,
        "⭐⭐ 注意这跟数据无关 ——&#160;<tspan font-weight=\"700\">"
        "喂进去的梯度自始至终是同一个数</tspan>，"
        "是那个<tspan font-weight=\"700\">只增不减的分母</tspan>把它自己压死的。",
        INK, size=15, anchor="middle")
    f._pan = None

    # ══════════ Ⓒ RMSProp 的病：初期 ═══════════════════════════════
    PH3 = 360
    py3 = f.panel(0, py2 + PH2 + 20, W, PH3,
                  "Ⓒ ⭐⭐⭐ 而 RMSProp 的病在<tspan font-weight=\"700\">初期</tspan>"
                  "　——　滑动平均从 0 起步，<tspan font-weight=\"700\">头几步严重低估</tspan>", OR,
                  sub="⛔ 分母被低估 → 更新被放大。"
                      "<tspan font-weight=\"700\">Adam 的偏差校正就是来修这一下的</tspan>")

    OX3, OY3, AW3, AH3 = 150, py3 + 272, 1080, 196
    NT = 400
    f.line(OX3, OY3, OX3 + AW3 + 30, OY3, GY2, 1.2, arrow=False)
    f.line(OX3, OY3, OX3, OY3 - AH3 - 20, GY2, 1.2, arrow=False)
    YMAX = 8.0
    for lv in (1.0, 4.0, 8.0):
        yy = OY3 - AH3 * lv / YMAX
        f.line(OX3 - 6, yy, OX3 + AW3, yy, GY2, 1, dash="3 4", arrow=False)
        f.t(OX3 - 12, yy + 5, "%d×" % lv, GY2, size=12, anchor="end")
    f.t(OX3 + AW3, OY3 + 26, "第 %d 步" % NT, GY2, size=12, anchor="end")

    # ⛔ 三条线在右端都收敛到 1 附近，标签叠在一起没法看。
    #   ⭐ 判据：**曲线的标签贴在「它跟别人最不一样」的那一段，不要一律贴右端。**
    for fn, col, lab, sw, at in ((eff_rmsprop, OR, "RMSProp（无校正）", 2.6, 0.06),
                                 (eff_adam_nofix, PU, "Adam 若不校正", 2.2, 0.20),
                                 (eff_adam, GR, "Adam（有校正）", 3.0, 0.72)):
        d = None
        for k in range(321):
            tt = 1 + (NT - 1) * (k / 320.0) ** 1.6
            v = min(fn(tt), YMAX)
            px = OX3 + AW3 * (tt - 1) / (NT - 1)
            pyv = OY3 - AH3 * v / YMAX
            d = ("M %.1f %.1f" % (px, pyv)) if d is None else d + " L %.1f %.1f" % (px, pyv)
        f.path(d, col, sw, arrow=False)
        _t = 1 + (NT - 1) * at
        _lx = OX3 + AW3 * (_t - 1) / (NT - 1)
        _ly = OY3 - AH3 * min(fn(_t), YMAX) / YMAX
        f.line(_lx, _ly, _lx + 30, _ly - 34, col, 1.4)
        f.t(_lx + 36, _ly - 38, lab, col, True, 14)

    f.t(OX3 + AW3 + 20, OY3 - AH3 - 6,
        "⛔ 第一步被放大 <tspan font-weight=\"700\">%.0f 倍</tspan>（冲出画面）"
        % RMS_SPIKE, OR, True, 15, "end")
    f.t(OX3 + AW3 + 20, OY3 - AH3 + 22,
        "⭐ 而 Adam 若不校正，峰值只有 <tspan font-weight=\"700\">%.1f 倍</tspan>"
        "　——　因为动量那一项的偏差<tspan font-weight=\"700\">部分抵消了它</tspan>"
        % NOFIX_PEAK, PU, True, 14, "end")

    f.t(700, py3 + 326,
        "⭐⭐⭐ 绿线<tspan font-weight=\"700\">从第一步起就是平的</tspan>"
        "　——　偏差校正不是小修小补，"
        "<tspan font-weight=\"700\">它把开头那几十步从「不可用」变成了「可用」。</tspan>",
        INK, size=15, anchor="middle")
    f._pan = None

    yb = f.band(py3 + PH3 + 20, "ok",
                "这条链最强的一个证据：后面那个能把前面那个当成特例含进去",
                ("⭐⭐⭐ Adam 原文自己证了一件事："
                 "<tspan font-weight=\"700\">AdaGrad 就是 Adam 的一个特例</tspan>"
                 "（β₁ 取 0、(1−β₂) 取无穷小、学习率按 t 的负二分之一次方退火）。"
                 "——&#160;<tspan font-weight=\"700\">所以这真的是一条链，不是四个并列的选项。</tspan>",
                 "⛔ 而每一环都<tspan font-weight=\"700\">还欠着下一环</tspan>："
                 "Adam 留下的那个洞是<tspan font-weight=\"700\">权重衰减被卷进了自适应的分母里</tspan>"
                 "（AdamW 来修），"
                 "以及<tspan font-weight=\"700\">每个参数要挂两份状态</tspan>（Muon 来省）。"
                 "这两环本讲后面都有。",
                 "⚠️ 口径：Ⓑ Ⓒ 三条线用的是<tspan font-weight=\"700\">同一个 β₂</tspan>，"
                 "为的是把「有没有遗忘」「有没有偏差校正」这两个变量单独拎出来 ——&#160;"
                 "<tspan font-weight=\"700\">这里不是在报 RMSProp 的真实常用配置。</tspan>"))

    yb = f.src(yb + 16,
               "📌 <tspan font-weight=\"700\">这一格每一环都出自 Adam 原文</tspan>"
               "（arXiv 1412.6980，已读原文）："
               "AdaGrad 的更新式原文逐字写作 "
               "<tspan font-weight=\"700\">θ_{t+1} ＝ θ_t −&#160;α·g_t / √(Σ g²)</tspan>"
               "　——　<tspan font-weight=\"700\">分母是累加的平方和，不除以 t</tspan>，"
               "所以「单调衰减」这个病是从式子里长出来的；"
               "Adam 的定位原文写作「combine the advantages of AdaGrad（稀疏梯度）"
               "and RMSProp（在线与非平稳）」。",
               "⭐ 偏差校正那一条，原文的说法是：滑动平均<tspan font-weight=\"700\">从 0 起步</tspan>，"
               "于是估计 biased towards zero，<tspan font-weight=\"700\">especially during "
               "the initial timesteps</tspan>，而且 <tspan font-weight=\"700\">β 越接近 1 越严重</tspan>。"
               "同一段还点名了 RMSProp：<tspan font-weight=\"700\">without bias correction, "
               "like in RMSProp</tspan> ——&#160;Ⓒ 那条橙线画的就是这句话。",
               "⚠️ <tspan font-weight=\"700\">李宏毅课上那页写的 Adagrad 是「均方根」形式</tspan>"
               "（分母里除了 t＋1），跟这里画的原版<tspan font-weight=\"700\">不一样</tspan>。"
               "⛔ 本讲按原版画，因为「有效学习率单调衰减」这个病"
               "<tspan font-weight=\"700\">只在原版（累加不除 t）上成立</tspan> ——&#160;"
               "这个区别值得知道，不然两边对不上会以为自己算错了。",
               "⭐ 图上的 %.1f%%、%.0f 倍、%.1f 倍三个数都是脚本算的，五条 assert 盯着："
               "AdaGrad 必须单调衰减、跑到第 %d 步必须掉到很小、"
               "不校正时第一步必须被放大很多倍、"
               "做了校正必须从第一步起恒为 1、"
               "以及「Adam 不校正的峰值必须低于 RMSProp 那条」"
               "（因为动量那一项的偏差会部分抵消）。"
               % (ADA_DECAY * 100, RMS_SPIKE, NOFIX_PEAK, T))

    f.save("fig4-lineage.svg", yb + 14)


if __name__ == "__main__":
    main()
