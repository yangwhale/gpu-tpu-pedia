# -*- coding: utf-8 -*-
r"""专题四 · §1.3b「前面猜错了，会不会影响后面」——&#160;teacher forcing

⭐⭐⭐ 2026-09-22 现场自己推出来的一问（原话）：
  「就属于那个错误不会向后传，不会因为前面的一系列 token 猜错了就影响我
    这一个步骤。因为我压根不管它前面猜的对不对，我就用你训练给我的那个
    前面的序列去算我自己的这个情况。」
  ——&#160;**完全正确，一个字不用改。** 这件事有名字：**teacher forcing**。

⛔ 但他的描述里有一处要说准：他说「每一个 token 的 Q 去查询前面的 K 和 V」，
  语义对，**操作上不是一个一个来的** ——&#160;
  训练里 Q/K/V 一把算完、注意力一把做完，**所有位置同时**。
  ⭐ 而这正是训练和推理**最大的形态差别**：推理才是真的一个一个来。

⭐⭐ 这一格真正值钱的是**代价**那一半：**exposure bias（曝光偏差）**。
  训练时它**从来没见过自己的错误**，一直活在「前文永远是对的」那个世界里；
  推理时前文**全是它自己生成的** ——&#160;一旦错一个，
  它就走进了一个**训练时从没见过的分布**。
  ⇒ 这解释了一个人人遇到过的现象：**长文本生成为什么会越跑越偏。**

📌 出处（2026-09-22 现查，不是凭记忆）：
  · **teacher forcing** 这个词由 **Williams & Zipser（1989）** 提出，
    在循环网络上；他们自己写的是 "We call this intuitively sensible
    technique teacher forcing"。
  · **scheduled sampling**（训练中按概率混入模型自己的预测，用来缓解
    曝光偏差）出自 **Bengio 等，arXiv 1506.03099（2015）**。
  ⚠️ 今天的大模型基本仍是**纯 teacher forcing** ——&#160;
    并行那个好处太大，大到愿意忍着曝光偏差。
"""
from topic03_draw import Fig, BL, OR, GR, RD, PU, GY, INK, GY2

W = 1400

TRUE = ("今", "天", "天", "气", "很", "好")
# 训练时每个位置的预测：第 3 个猜错了（猜成「空」）
PRED = ("天", "天", "空", "气", "很", "好")
BAD = 2
assert PRED[BAD] != TRUE[BAD], "示例里必须真有一个猜错的位置"
TGT = 4095


def main():
    f = Fig(W, "训练的时候每个位置的输入永远是真值，不是模型自己刚猜出来的那个字，"
               "所以前面猜错了也污染不到后面，这叫 teacher forcing；"
               "而且所有位置是一次矩阵乘同时算完的，不是一个一个来。"
               "推理时前文全是它自己生成的，必须一个一个来，"
               "一旦错一个，后面全建立在那个错的上面。"
               "代价叫曝光偏差：训练时它从来没见过自己的错误，"
               "一旦在推理时错一个，就走进了训练时从没见过的分布，"
               "这就是长文本生成越跑越偏的原因")

    y0 = f.header(
        "前面猜错了，会不会影响后面　——　<tspan font-weight=\"700\">训练时不会</tspan>",
        "⭐ 这件事有名字：<tspan font-weight=\"700\">teacher forcing</tspan>　——　"
        "每个位置的输入<tspan font-weight=\"700\">永远是真值</tspan>，"
        "不是模型自己刚猜的那个字",
        [(GR, "Ⓐ 训练"), (RD, "Ⓑ 推理"), (OR, "Ⓒ 代价与取舍")])

    # ══════════ Ⓐ 训练 ═════════════════════════════════════════════
    PH = 336
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 训练：<tspan font-weight=\"700\">输入那一排，从头到尾都是真值</tspan>", GR,
                 sub="⭐ 而且<tspan font-weight=\"700\">所有位置同时算完</tspan>"
                     "　——　一次矩阵乘，不是一个一个来")

    BX, BW, GAPX = 150, 92, 118
    f.t(96, py + 82, "真值", GR, True, 14, "end")
    for k, ch in enumerate(TRUE):
        x = BX + k * GAPX
        f.box(x, py + 58, BW, 48, "#e6f4ea", GR, 6, sw=1.4)
        f.t(x + BW / 2, py + 90, ch, INK, True, 20, "middle")
        f.line(x + BW / 2, py + 112, x + BW / 2, py + 142, GY2, 1.6)

    f.t(96, py + 182, "预测", GY, True, 14, "end")
    for k, ch in enumerate(PRED):
        x = BX + k * GAPX
        wrong = (k == BAD)
        f.box(x, py + 150, BW, 48, "#fce8e6" if wrong else "#f1f3f4",
              RD if wrong else GY2, 6, sw=1.4)
        f.t(x + BW / 2, py + 182, ch, RD if wrong else GY, True, 20, "middle")
        if wrong:
            f.t(x + BW / 2, py + 224, "✘ 猜错了", RD, True, 13, "middle")

    # 那个「错的不往下一格喂」的叉
    ax = BX + BAD * GAPX + BW / 2
    f.path("M%.0f,%.0f C%.0f,%.0f %.0f,%.0f %.0f,%.0f"
           % (ax, py + 200, ax + 30, py + 234, ax + 88, py + 234, ax + GAPX, py + 116),
           RD, 2.0, dash="5 4")
    f.t(ax + GAPX / 2 + 34, py + 258, "✘", RD, True, 22, "middle")
    f.t(ax + GAPX / 2 + 130, py + 258,
        "<tspan font-weight=\"700\">这条路不存在</tspan>", RD, True, 14)

    f.box(880, py + 58, 480, 186, "#e6f4ea", GR, 8)
    f.t(1120, py + 94, "所以「错误不会向后传」", GR, True, 17, "middle")
    f.t(1120, py + 130,
        "下一个位置拿到的输入，<tspan font-weight=\"700\">来自上面那一排真值</tspan>，",
        INK, size=14.5, anchor="middle")
    f.t(1120, py + 156,
        "<tspan font-weight=\"700\">不是来自刚才那个错的预测</tspan>。",
        INK, size=14.5, anchor="middle")
    f.t(1120, py + 196,
        "⭐ 每个位置<tspan font-weight=\"700\">各算各的</tspan>，互不污染　——",
        GR, True, 14, "middle")
    f.t(1120, py + 222,
        "所以 %s 个位置可以<tspan font-weight=\"700\">一次全算完</tspan>。"
        % format(TGT, ","), GY, size=13.5, anchor="middle")
    f._pan = None

    # ══════════ Ⓑ 推理 ═════════════════════════════════════════════
    PH2 = 300
    py2 = f.panel(0, py + PH + 20, W, PH2,
                  "Ⓑ 推理：<tspan font-weight=\"700\">前文全是它自己生成的</tspan>", RD,
                  sub="⛔ 而且必须<tspan font-weight=\"700\">一个一个来</tspan>"
                      "　——　下一个字得等上一个字出来")

    for k, ch in enumerate(PRED):
        x = BX + k * GAPX
        bad = (k >= BAD)
        f.box(x, py2 + 66, BW, 48, "#fce8e6" if bad else "#f1f3f4",
              RD if bad else GY2, 6, sw=1.4)
        f.t(x + BW / 2, py2 + 98, ch, RD if bad else GY, True, 20, "middle")
        if k < len(PRED) - 1:
            f.line(x + BW + 4, py2 + 90, x + GAPX - 4, py2 + 90,
                   RD if bad else GY2, 2.0)
    f.t(BX + BAD * GAPX + BW / 2, py2 + 140, "✘ 从这儿错", RD, True, 13, "middle")
    f.line(BX + BAD * GAPX, py2 + 158, BX + (len(PRED) - 1) * GAPX + BW,
           py2 + 158, RD, 2.0, arrow=False)
    f.t(BX + (BAD + len(PRED) - 1) * GAPX / 2 + BW / 2, py2 + 182,
        "<tspan font-weight=\"700\">后面每一个，都建立在那个错的上面</tspan>",
        RD, True, 14.5, "middle")

    f.box(880, py2 + 58, 480, 156, "#fce8e6", RD, 8)
    f.t(1120, py2 + 94, "⛔ 训练和推理，喂的东西不一样", RD, True, 16.5, "middle")
    f.t(1120, py2 + 130, "训练：<tspan font-weight=\"700\">真值</tspan>　·　"
        "推理：<tspan font-weight=\"700\">它自己的输出</tspan>",
        INK, True, 15, "middle")
    f.t(1120, py2 + 168,
        "训练：<tspan font-weight=\"700\">一次算完</tspan>　·　"
        "推理：<tspan font-weight=\"700\">串行 %s 次</tspan>" % format(TGT, ","),
        INK, True, 15, "middle")
    f.t(1120, py2 + 200,
        "⭐ 这是训练和推理<tspan font-weight=\"700\">最大的形态差别</tspan>。",
        GY, size=13.5, anchor="middle")
    f._pan = None

    # ══════════ Ⓒ 代价与取舍 ═══════════════════════════════════════
    PH3 = 322
    py3 = f.panel(0, py2 + PH2 + 20, W, PH3,
                  "Ⓒ 代价有个名字：<tspan font-weight=\"700\">曝光偏差</tspan>"
                  "（exposure bias）", OR,
                  sub="⭐ 训练时它<tspan font-weight=\"700\">从来没见过自己的错误</tspan>")

    f.box(50, py3 + 56, 630, 214, "#fef7e0", OR, 8)
    f.t(365, py3 + 92, "它一直活在一个「前文永远是对的」世界里", OR, True, 16.5, "middle")
    f.t(365, py3 + 132,
        "可推理时前文<tspan font-weight=\"700\">全是它自己生成的</tspan>。",
        INK, size=15, anchor="middle")
    f.t(365, py3 + 166,
        "⛔ 一旦错一个，它就走进了一个", INK, size=15, anchor="middle")
    f.t(365, py3 + 194,
        "<tspan font-weight=\"700\">训练时从没见过的分布</tspan>　——　"
        "而在那儿它没有任何经验。", INK, True, 15, "middle")
    f.t(365, py3 + 238,
        "⭐⭐ 这就解释了：<tspan font-weight=\"700\">长文本生成为什么会越跑越偏。</tspan>",
        OR, True, 15, "middle")
    f.t(365, py3 + 264,
        "不是它突然变笨了，是它走进了自己没被训练过的地方。",
        GY, size=13.5, anchor="middle")

    f.box(710, py3 + 56, 640, 214, "#e6f4ea", GR, 8)
    f.t(1030, py3 + 92, "那为什么还这么干？", GR, True, 16.5, "middle")
    f.t(1030, py3 + 132,
        "① <tspan font-weight=\"700\">并行</tspan>　——　"
        "用自己的预测喂自己的话，第 t 个位置", INK, size=14.5, anchor="middle")
    f.t(1030, py3 + 158,
        "得等第 t−1 个算完，<tspan font-weight=\"700\">%s 个位置就得串行 %s 次</tspan>。"
        % (format(TGT, ","), format(TGT, ",")), INK, size=14.5, anchor="middle")
    f.t(1030, py3 + 196,
        "② <tspan font-weight=\"700\">早期不发散</tspan>　——　"
        "刚开始模型猜得一塌糊涂，", INK, size=14.5, anchor="middle")
    f.t(1030, py3 + 222,
        "拿它自己的输出喂自己，<tspan font-weight=\"700\">一步就飞了</tspan>。",
        INK, size=14.5, anchor="middle")
    f.t(1030, py3 + 260,
        "⭐ 所以这是一笔<tspan font-weight=\"700\">明知有代价、仍然划算</tspan>的交易。",
        GR, True, 14, "middle")
    f._pan = None

    yb = f.band(py3 + PH3 + 18, "ok", "⭐ 两句话记住", [
        "<tspan font-weight=\"700\">训练喂真值、推理喂自己</tspan>　——　"
        "所以训练时「前面猜错了」污染不到后面，"
        "而且 %s 个位置能<tspan font-weight=\"700\">一次算完</tspan>；推理只能一个一个来。"
        % format(TGT, ","),
        "<tspan font-weight=\"700\">代价是曝光偏差</tspan>：它训练时没见过自己的错误，"
        "所以推理时一旦错一个，就进了没被训练过的分布　——　"
        "<tspan font-weight=\"700\">长文本越跑越偏，根子在这儿。</tspan>",
    ])

    yb = f.src(yb + 10,
               "出处（2026-09-22 现查，不是凭记忆）：<tspan font-weight=\"700\">"
               "teacher forcing</tspan> 这个词由 Williams &amp; Zipser（1989）在循环网络上提出，"
               "原文写的是 “We call this intuitively sensible technique teacher forcing”；",
               "<tspan font-weight=\"700\">scheduled sampling</tspan>"
               "（训练中按概率混入模型自己的预测，用来缓解曝光偏差）出自 Bengio 等，"
               "arXiv 1506.03099（2015）。⚠️ 今天的大模型基本仍是纯 teacher forcing　——　"
               "并行那个好处太大，大到愿意忍着曝光偏差。")

    f.save("fig4-teacher.svg", yb + 14)


main()
