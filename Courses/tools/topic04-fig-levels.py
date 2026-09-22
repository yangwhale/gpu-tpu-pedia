# -*- coding: utf-8 -*-
r"""专题四 · §1.2e「那个平均，到底分几层？」

⭐⭐⭐ 2026-09-22 现场追问（原话）：
  「这个平均你说他到底是怎么个平均呢？是用 4095 这个序列级别的平均，
    然后在层结束的时候在搞这个 batch 级别的平均，还是说他搞这个
    per device batch 的平均，然后平均以后再 cross device 的平均？
    ……应该是分层次的。首先我这个序列内要平均一下，然后一个 device 上的
    所有的这个 batch 之间再平均一下，然后再 cross device 的再平均一下，
    然后再这个超过 micro batch 之类的。」

⭐ 「分层次」这个直觉**是对的**，但层数和动作都要改：

  ⛔ 他数的是**四层平均**。实际是 ——&#160;**三层加法 ＋ 一次除法**。

  · 层① **序列内 ＋ 同一张卡上的多条序列 ——&#160;这两层其实是同一层。**
    那块 δ 表的形状是「(本卡序列数 × 序列长度) × 宽度」，
    权重梯度那一次矩阵乘**一次就把这一整维收缩掉**。
    ⇒ 他以为是两次，实际是**一次**。
  · 层② **梯度累积（micro-batch）**：累加到同一个桶，**只加不除**。
  · 层③ **跨卡（DP）**：all-reduce **求和**。
  · 除法：**全程只有一次**，除以「全局有效 token 总数」。

⛔⛔ 为什么不能每层各除一次 ——&#160;**那就成了「平均的平均」**。
  各层的有效 token 数并不相等（padding、拼接边界、最后一个批次），
  一旦各自先除，token 少的那一批就被**当成跟多的那批一样重**。

📌 出处：MaxText 的梯度累积实现（`utils/gradient_accumulation.py`）逐行如此：
    acc["loss"]          += aux["xent_sum"]        # 累加 sum
    acc["total_weights"] += aux["total_weights"]   # 累加 token 数
    ...
    raw_grads = tree_map(lambda g: g / acc["total_weights"], raw_grads)   # 最后除一次
  ⭐ 注意它除的是**累加起来的 token 总数**，不是「micro-batch 的个数」——&#160;
    这正是躲开「平均的平均」的那一手。
  ⚠️ 辅助损失（MoE 负载均衡等）反而是除以 micro-batch 个数的，
    因为它们本来就是「每个 micro-batch 一个平均值」。两套口径，别混。
"""
from topic03_draw import Fig, BL, OR, GR, RD, PU, GY, INK, GY2

W = 1400

# ⭐ Ⓒ 那个反例的数：两个 micro-batch，有效 token 数**不相等**
MB = ((100, 200.0), (20, 60.0))        # (有效 token 数, 该批的 xent 之和)
_WRONG = sum(sm / n for n, sm in MB) / len(MB)          # 平均的平均
_RIGHT = sum(sm for _, sm in MB) / sum(n for n, _ in MB)
assert abs(_WRONG - 2.5) < 1e-9 and abs(_RIGHT - 13.0 / 6) < 1e-9
assert abs(_WRONG - _RIGHT) / _RIGHT > 0.1, \
    "反例必须偏得看得见 ——&#160;不然「不能先除」这条读起来像吹毛求疵"


def main():
    f = Fig(W, "那个平均分几层？分层次这个直觉是对的，但实际是三层加法加一次除法。"
               "序列内和同一张卡上的多条序列其实是同一层，"
               "因为权重梯度那一次矩阵乘一次就把它们收缩掉了；"
               "第二层是梯度累积，只加不除；第三层是跨卡求和。"
               "而除法全程只有一次，除以全局有效 token 总数。"
               "如果每层各除一次，就成了平均的平均，"
               "token 少的那一批会被当成跟多的那批一样重")

    y0 = f.header(
        "那个「平均」，到底<tspan font-weight=\"700\">分几层</tspan>",
        "⭐ 「分层次」这个直觉对了，但实际是　——　"
        "<tspan font-weight=\"700\">三层加法 ＋ 一次除法</tspan>",
        [(RD, "Ⓐ 以为的"), (GR, "Ⓑ 实际的"),
         (OR, "Ⓒ 为什么不能各除各的"), (PU, "Ⓓ 源码对照")])

    # ══════════ Ⓐ 以为的：四层各平均一次 ═══════════════════════════
    PH = 216
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 很自然会这么想：<tspan font-weight=\"700\">一层一层各平均一次</tspan>",
                 RD)
    GUESS = ("序列内\n平均一次", "本卡多条序列\n再平均一次",
             "micro-batch 之间\n再平均一次", "跨卡\n再平均一次")
    for k, g in enumerate(GUESS):
        x = 70 + k * 330
        f.box(x, py + 48, 270, 90, "#fce8e6", RD, 8, sw=1.4)
        a, b = g.split("\n")
        f.t(x + 135, py + 82, a, INK, True, 14.5, "middle")
        f.t(x + 135, py + 110, b, RD, True, 14, "middle")
        if k < 3:
            f.line(x + 274, py + 93, x + 326, py + 93, GY2, 1.6)
    f.t(700, py + 178,
        "⛔ <tspan font-weight=\"700\">四次除法。而实际上除法只有一次</tspan>"
        "　——　而且其中有一层，<tspan font-weight=\"700\">根本不存在</tspan>。",
        RD, True, 15, "middle")
    f._pan = None

    # ══════════ Ⓑ 实际：三层加法 ＋ 一次除法 ═══════════════════════
    PH2 = 330
    py2 = f.panel(0, py + PH + 20, W, PH2,
                  "Ⓑ 实际是这样：<tspan font-weight=\"700\">加法分三层，除法只有一次</tspan>",
                  GR)
    REAL = (
        (GR, "① 序列内 ＋ 本卡多条", "这两层是<tspan font-weight=\"700\">同一层</tspan>",
         "δ 表的形状是「(本卡条数 × 序列长度) × 宽」，\n"
         "权重梯度那一次矩阵乘<tspan font-weight=\"700\">一次收缩掉整维</tspan>"),
        (BL, "② micro-batch 之间", "<tspan font-weight=\"700\">累加</tspan>到同一个桶",
         "梯度累积：算一批加一批，\n<tspan font-weight=\"700\">只加不除</tspan>"),
        (PU, "③ 跨卡（DP）", "all-reduce <tspan font-weight=\"700\">求和</tspan>",
         "每张卡把自己那个桶交出来，\n加到一起"),
    )
    for k, (col, name, act, why) in enumerate(REAL):
        x = 40 + k * 340
        f.box(x, py2 + 48, 300, 168, "#fff", col, 8, sw=1.6)
        f.box(x, py2 + 48, 300, 4, col, col, 2)
        f.t(x + 150, py2 + 84, name, col, True, 15.5, "middle")
        f.t(x + 150, py2 + 116, act, INK, True, 15, "middle")
        for i, ln in enumerate(why.split("\n")):
            f.t(x + 150, py2 + 154 + i * 24, ln, GY, size=12.5, anchor="middle")
        if k < 2:
            f.line(x + 304, py2 + 132, x + 336, py2 + 132, GY2, 1.8)

    f.line(1096, py2 + 132, 1128, py2 + 132, OR, 2.4)
    f.box(1134, py2 + 48, 226, 168, "#fef7e0", OR, 8, sw=2.0)
    f.t(1247, py2 + 92, "④ 除一次", OR, True, 17, "middle")
    f.t(1247, py2 + 128, "÷ <tspan font-weight=\"700\">全局有效</tspan>",
        INK, True, 15, "middle")
    f.t(1247, py2 + 152, "<tspan font-weight=\"700\">token 总数</tspan>",
        INK, True, 15, "middle")
    f.t(1247, py2 + 192, "⭐ 全程<tspan font-weight=\"700\">就这一次</tspan>",
        OR, True, 13.5, "middle")

    f.t(700, py2 + 262,
        "⭐⭐⭐ 所以「加起来」这件事<tspan font-weight=\"700\">确实分层</tspan>，"
        "只不过每一层做的都是<tspan font-weight=\"700\">加</tspan>　——　"
        "<tspan font-weight=\"700\">除，从头到尾只有一下。</tspan>",
        INK, size=15, anchor="middle")
    f.t(700, py2 + 292,
        "⛔ 而 ① 那一层要特别说一句：你以为的「序列内平均」和「本卡跨序列平均」"
        "<tspan font-weight=\"700\">不是两步，是同一次矩阵乘</tspan>　——　"
        "那两维早就拼成一维了。",
        GY, size=13.5, anchor="middle")
    f._pan = None

    # ══════════ Ⓒ 为什么不能各除各的 ═══════════════════════════════
    PH3 = 330
    py3 = f.panel(0, py2 + PH2 + 20, W, PH3,
                  "Ⓒ 那<tspan font-weight=\"700\">为什么不能</tspan>每层各除各的？"
                  "　——　因为那是「平均的平均」", OR,
                  sub="⭐ 前提：各层的<tspan font-weight=\"700\">有效 token 数并不相等</tspan>"
                      "（padding、拼接边界、最后一批）")

    for k, (n, sm) in enumerate(MB):
        yy = py3 + 60 + k * 84
        f.box(60, yy, 470, 66, "#f1f3f4", GY2, 6)
        f.t(295, yy + 28, "micro-batch %d" % (k + 1), GY, True, 14, "middle")
        f.t(295, yy + 52,
            "有效 token <tspan font-weight=\"700\">%d</tspan> 个　·　"
            "这一批的 loss 加起来 <tspan font-weight=\"700\">%.0f</tspan>" % (n, sm),
            INK, size=13.5, anchor="middle")
        f.t(560, yy + 40, "→ 自己平均 %.1f" % (sm / n), GY2, size=13)

    f.box(760, py3 + 58, 290, 116, "#fce8e6", RD, 8)
    f.t(905, py3 + 92, "❌ 各除各的", RD, True, 16, "middle")
    f.t(905, py3 + 126, "(%.1f ＋ %.1f) ÷ 2" % (MB[0][1] / MB[0][0], MB[1][1] / MB[1][0]),
        INK, size=14, anchor="middle")
    f.t(905, py3 + 156, "＝ <tspan font-weight=\"700\">%.2f</tspan>" % _WRONG,
        RD, True, 19, "middle")

    f.box(1070, py3 + 58, 290, 116, "#e6f4ea", GR, 8)
    f.t(1215, py3 + 92, "✅ 先全加，最后除一次", GR, True, 15, "middle")
    f.t(1215, py3 + 126, "(%.0f ＋ %.0f) ÷ (%d ＋ %d)"
        % (MB[0][1], MB[1][1], MB[0][0], MB[1][0]), INK, size=14, anchor="middle")
    f.t(1215, py3 + 156, "＝ <tspan font-weight=\"700\">%.2f</tspan>" % _RIGHT,
        GR, True, 19, "middle")

    f.t(700, py3 + 232,
        "⭐⭐ 差 <tspan font-weight=\"700\">%.0f%%</tspan>。"
        "而错的那一边<tspan font-weight=\"700\">不报错、不发散、曲线看着完全正常</tspan>"
        "　——　它只是<tspan font-weight=\"700\">悄悄给每一批换了权重</tspan>。"
        % (abs(_WRONG - _RIGHT) / _RIGHT * 100), INK, size=15, anchor="middle")
    f.t(700, py3 + 262,
        "⛔ 病根一句话：<tspan font-weight=\"700\">先除，就等于宣布「每一批一样重」</tspan>"
        "　——　可它们的 token 数明明不一样。",
        RD, True, 14.5, "middle")
    f.t(700, py3 + 294,
        "⭐ 所以正确的做法是：<tspan font-weight=\"700\">一路只传两个数</tspan>"
        "　——　「loss 的和」和「token 的个数」，最后拿前者除后者。",
        GY, size=13.5, anchor="middle")
    f._pan = None

    # ══════════ Ⓓ 源码对照 ═════════════════════════════════════════
    PH4 = 270
    py4 = f.panel(0, py3 + PH3 + 20, W, PH4,
                  "Ⓓ 这不是我推的　——　<tspan font-weight=\"700\">梯度累积那段源码逐行就是这样</tspan>",
                  PU)
    LINES = (
        ("累加的是<tspan font-weight=\"700\">和</tspan>，不是平均",
         "acc[\"loss\"] += 这一批的 loss 之和"),
        ("<tspan font-weight=\"700\">token 个数</tspan>另外单独累加",
         "acc[\"total_weights\"] += 这一批的有效 token 数"),
        ("⭐ 最后<tspan font-weight=\"700\">除一次</tspan>，而且除的是"
         "<tspan font-weight=\"700\">累加起来的 token 总数</tspan>",
         "梯度 ÷ acc[\"total_weights\"]"),
    )
    for k, (say, code) in enumerate(LINES):
        yy = py4 + 54 + k * 58
        f.box(50, yy, 640, 46, "#f3e8fd", PU, 6)
        f.t(370, yy + 30, say, INK, size=14, anchor="middle")
        f.box(710, yy, 640, 46, "#f1f3f4", GY2, 6)
        f.t(1030, yy + 30, code, GY, True, 13.5, "middle")

    f.t(700, py4 + 236,
        "⛔ 注意它除的<tspan font-weight=\"700\">不是「micro-batch 的个数」</tspan>，"
        "是<tspan font-weight=\"700\">累加起来的 token 总数</tspan>"
        "　——　这一手就是用来躲开上面那个坑的。",
        INK, size=14.5, anchor="middle")
    f._pan = None

    yb = f.band(py4 + PH4 + 18, "ok", "⭐ 一句话记住", [
        "<tspan font-weight=\"700\">加法分三层（本卡一次矩阵乘 →&#160;梯度累积 →&#160;跨卡求和），"
        "除法只有一次</tspan>　——　除以全局有效 token 总数。",
        "<tspan font-weight=\"700\">一路只传两个数：loss 的和、token 的个数。</tspan>"
        "⛔ 任何一层提前做除法，都是「平均的平均」　——　"
        "它不报错，只是悄悄给每一批换了权重。",
    ])

    yb = f.src(yb + 10,
               "出处：MaxText 的梯度累积实现 —— 循环里 "
               "<tspan font-weight=\"700\">acc[\"loss\"] += xent_sum</tspan>、"
               "<tspan font-weight=\"700\">acc[\"total_weights\"] += total_weights</tspan>，"
               "循环外 <tspan font-weight=\"700\">grads ÷ acc[\"total_weights\"]</tspan>。",
               "⚠️ 一个例外要说清：<tspan font-weight=\"700\">辅助损失</tspan>"
               "（MoE 负载均衡这类）反而是<tspan font-weight=\"700\">除以 micro-batch 个数</tspan>的"
               "　——　因为它们本来就是「每个 micro-batch 一个平均值」。两套口径，别混。")

    f.save("fig4-levels.svg", yb + 14)


main()
