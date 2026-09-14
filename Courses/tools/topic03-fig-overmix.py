# -*- coding: utf-8 -*-
r"""专题三 · §六「attention sink 有什么用」—— 一滴红墨水，和一个替它踩刹车的人

⭐⭐⭐ 2026-09-13 夜间 R18 新画。调研 agent 把这一处点成了第一名，理由很硬：
   **这是全课唯一一个能把「抽象危害」变成看得见的东西的装置。**

⛔ 本课到 R13 为止，只讲了 sink 的**成因**（softmax 不许弃权，
   模型就自己造一个弃权用的候选人）。**从没讲它有什么用。**
   于是读者会以为它是个纯粹的病灶 —— 而 StreamingLLM 保留它只是打补丁。
   ⭐ 真相反过来：**那张弃权票不是浪费掉的，它是刹车片。**

📌 装置偷自 Barbero 等《Why do LLMs attend to the first token?》
   （arXiv 2504.02732）的 Figure 1：**左右两张 token×层 的网格，
   唯一差别是右边多一个 sink**，然后看扰动的红色渗开多大面积。
   ⚠️ 但那张图自己也是**示意图**（实测在他们的 Figure 2，是折线，冲击力差得多）。

⭐ 本图在它之上补了两样，都是它没画的：
   ① **‖v‖ ≈ 0 这一笔**。光「吸走注意力」讲不通 ——
      敏锐的学生会问「那被吸走的信息不也进了输出吗」。
      必须是**吸得多、而且吐不出东西**，比喻才立得住。
   ② **数字**。原图是纯示意，本课当场跑一个线性简化模拟，把倍数算出来。

⚠️ 三条归属上的坑，调研 agent 逐条核过，这里写死免得以后引错：
   · 「pump the brakes」**不是 Barbero 的原话** —— v1/v4 全文零命中。
   · 「pressure valve / 泄压阀」也**不是论文的词**，是 MIT HAN Lab 博客的转述。
   · 「一滴墨水滴进水里」这个画面**本课原创** —— arXiv 2605.10828 确实用过
     墨水比喻，但那篇讲的是长上下文里的干扰信息，**跟 sink 无关**，别张冠李戴。
   能引的原话只有 Figure 1 caption 那两句，抄在出处行里。
"""
import numpy as np

from topic03_draw import (Fig, BL, OR, GR, RD, GY, PU, INK, GY2, LINE, LINE2,
                          BG2, wpx)

W = 1400
N, L, S, P = 6, 4, 0.80, 2        # 6 个 token · 4 层 · sink 吸 80% · 扰动第 2 个


def layers(sink):
    """逐层算：扰动第 P 个 token 之后，每个 token 被推动多少（相对它自己的量级）。

    ⭐ 这是一个**线性简化模拟**，不是实测：均匀注意力 ＋ 残差流
      h ← h ＋ Attn(h)，sink 吸走 S 的权重、而它的 value 记为 0。
    """
    A = np.zeros((N, N))
    for i in range(N):
        if sink:
            if i == 0:
                A[0, 0] = 1.0
            else:
                A[i, 0] = S
                A[i, 1:i + 1] = (1 - S) / i
        else:
            A[i, :i + 1] = 1.0 / (i + 1)
    if sink:
        A[:, 0] = 0.0                       # ⭐ 吸得多，但吐不出东西
    out, M = [], np.eye(N)
    for _ in range(L + 1):
        out.append([M[i, P] / M[i, i] for i in range(N)])
        M = (np.eye(N) + A) @ M
    return out


def main():
    NO, YES = layers(False), layers(True)
    # ⛔ 断言这张图的全部论点：没有 sink，四层之后旁观者被推动得跟当事人一样多。
    assert np.mean(NO[-1][3:]) > 0.9, NO[-1]
    assert np.mean(YES[-1][3:]) < 0.3, YES[-1]
    gain = np.mean(NO[-1][3:]) / np.mean(YES[-1][3:])
    assert 4 < gain < 5, gain

    f = Fig(W, "attention sink 有什么用：一滴红墨水，和一个替它踩刹车的人")
    yy = f.header(
        "attention sink 有什么用 ——&#160;一滴红墨水，和一个替它踩刹车的人",
        "前面只讲了 sink 是<tspan font-weight=\"700\">怎么来的</tspan>"
        "（softmax 不许弃权，模型就造一个弃权用的候选人）。"
        "那它<tspan font-weight=\"700\">有什么用</tspan>？"
        "——&#160;这张图说：那张弃权票<tspan font-weight=\"700\">不是浪费掉的，"
        "它是刹车片</tspan>。",
        legend=[(RD, "被扰动的那个词，以及它染红的部分"),
                (BL, "sink（第 0 个 token）"),
                (GY2, "没被影响到的词")])

    # ⛔ 原来 92/12/92 → 单张网格宽 716，右边那张从 724 起，直接冲出 1400。
    #   ⭐ 文字溢出会被裁掉且**不报错**，所以宽度要在这里算一次再画。
    CW, CH, GAP = 80, 52, 11           # 一个格子
    LW = 80                            # 左边「第 N 层」标签栏
    assert 694 + LW + N * (CW + GAP) <= W, "右边那张网格会冲出画布"

    def grid(x0, y0, data, sink, title, col, tag):
        f.t(x0, y0 - 12, title, col, bold=True, size=18, cls="svglbl")
        f.t(x0, y0 + 12, tag, GY, size=15)
        gy = y0 + 40
        # 列头：六个 token
        names = ["⟨起始⟩", "今天", "股价", "涨了", "很多", "。"]
        for j in range(N):
            f.t(x0 + LW + j * (CW + GAP) + CW / 2, gy, names[j], GY2, size=14,
                anchor="middle")
        gy += 12
        for l in range(L + 1):
            ry = gy + l * (CH + GAP)
            f.t(x0 + LW - 14, ry + CH / 2 + 6,
                "第 %d 层" % l if l else "输入", GY, size=15, anchor="end")
            for j in range(N):
                v = data[l][j]
                cx = x0 + LW + j * (CW + GAP)
                if sink and j == 0:
                    f.box(cx, ry, CW, CH, "#e8f0fe", BL, 6)
                    if l == 0:
                        f.t(cx + CW / 2, ry + CH / 2 + 5, "sink", BL, True, 15,
                            "middle")
                    else:
                        f.t(cx + CW / 2, ry + CH / 2 + 5, "吸 %d%%" % (S * 100),
                            BL, True, 15, "middle")
                    continue
                # ⭐ 红的深浅 ＝ 被推动的幅度。**不写数字的格子也能读**，
                #   但关键的几格还是把数字写上 —— 「面积」给直觉，数字给证据。
                a = min(1.0, v)
                f.p.append('<rect x="%d" y="%d" width="%d" height="%d" rx="6" '
                           'fill="%s" fill-opacity="%.3f" stroke="%s"/>'
                           % (cx, ry, CW, CH, RD, 0.06 + 0.80 * a,
                              RD if a > 0.02 else LINE))
                if v >= 0.03:
                    f.t(cx + CW / 2, ry + CH / 2 + 6, "%.0f%%" % (100 * v),
                        "#fff" if a > 0.45 else RD, True, 16, "middle")
        return gy + (L + 1) * (CH + GAP)

    # ══ ①② 左右对照 ═══════════════════════════════════════════════
    PH1 = 530
    top = f.panel(0, yy, W, PH1,
                  "同一句话、同一个扰动、同样四层 ——&#160;"
                  "<tspan font-weight=\"700\">右边只多了一个 sink</tspan>",
                  RD, tag="本课的线性简化模拟，不是实测")
    # ⛔ 这两行原来把 col 和 tag 传反了，于是副标题那一行**直接印出了
    #   「#d93025」这串颜色码**，而标题的 fill 收到一句中文 —— 浏览器当非法值忽略。
    #   ⭐ 判据：**位置参数超过五个就该改成关键字传参**，护栏一条都拦不住这种错。
    b1 = grid(24, top + 48, NO, False, title="① 没有 sink",
              col=RD, tag="红色像墨水一样铺开")
    f.line(670, top + 60, 670, top + PH1 - 46, LINE2, 1.2, arrow=False)
    grid(694, top + 48, YES, True, title="② 有 sink", col=GR,
         tag="红色基本困在原地")
    f.t(24, b1 + 34,
        "⭐ 只读一件事：<tspan font-weight=\"700\">最后一行有多红。</tspan>"
        "左边四层之后，旁观的三个词被推动了约 %.0f%% ——&#160;"
        "<tspan font-weight=\"700\">跟当事人一样多</tspan>，整句话都被污染了；"
        "右边只有约 %.0f%%，<tspan font-weight=\"700\">小 %.1f 倍</tspan>。"
        % (100 * np.mean(NO[-1][3:]), 100 * np.mean(YES[-1][3:]), gain),
        INK, size=17)

    # ══ ③ 机制是两半，缺一不可 ═════════════════════════════════════
    yy = top + PH1 + 26
    PH3 = 284
    top = f.panel(0, yy, W, PH3,
                  "③ 它凭什么能刹住？——&#160;机制是<tspan font-weight=\"700\">"
                  "两半</tspan>，少一半就不成立", BL, tag="第二半最常被漏掉")
    for i, (ttl, sub, body, col) in enumerate([
        ("吸得多", "attention 权重",
         "它把绝大部分注意力吸到自己身上。Llama 405B 里"
         "<tspan font-weight=\"700\">将近 80% 的注意力落在第一个 token 上</tspan>。"
         "——&#160;别人分到的就少了。", BL),
        ("吐得少", "‖v‖ ≈ 0",
         "⭐ <tspan font-weight=\"700\">而它的 value 几乎是零。</tspan>"
         "所以吸走的那一大块<tspan font-weight=\"700\">不带任何内容回来</tspan>"
         " ——&#160;等于凭空把混合强度按下去了。", GR),
    ]):
        bx = 30 + i * 684
        f.box(bx, top + 32, 656, 150, "none", LINE, 9)
        f.badge(bx + 18, top + 48, i + 1, col)
        f.t(bx + 64, top + 70, ttl, col, bold=True, size=19, cls="svglbl")
        f.t(bx + 64 + wpx(ttl, 19) + 16, top + 70, "（%s）" % sub, GY2, size=15)
        yy2 = top + 104
        from topic03_draw import wrap_rich
        for r in wrap_rich(body, 622, 16 * 1.12):
            f.t(bx + 18, yy2, r, GY, size=16)
            yy2 += 24
    f.t(30, top + PH3 - 58,
        "⛔ <tspan font-weight=\"700\">只讲第一半是讲不通的</tspan>："
        "如果它吸走的注意力照样带内容回来，那信息一样会混 ——&#160;"
        "只是换了条路。", RD, size=17)
    f.t(30, top + PH3 - 32,
        "⭐ 所以准确的说法是：<tspan font-weight=\"700\">它是一个「几乎什么都不做」"
        "的去处</tspan> ——&#160;论文管这个叫 approximate no-op。", INK, size=17)

    # ══ 落点 ══════════════════════════════════════════════════════
    yy = top + PH3 + 30
    yy = f.band(yy, "ok", "于是前面讲 sink 那一格要改一句口径", [
        "前面说「softmax 不许弃权，模型就自己造了一个弃权用的候选人」——&#160;"
        "那一句<tspan font-weight=\"700\">只说到成因</tspan>。",
        "⭐⭐ 这张图补上的是：<tspan font-weight=\"700\">那张弃权票不是浪费掉的，"
        "它是刹车片。</tspan>没有它，扰动一个词，四层之后整句话都跟着动。",
        "⛔ 所以 StreamingLLM 为什么砍掉开头几个 token 模型就崩 ——&#160;"
        "<tspan font-weight=\"700\">不是丢了信息，是刹车没了。</tspan>"
        "（那几个 token 本来就没什么内容，这正是它们能当 sink 的原因。）",
    ])
    yy = f.band(yy + 14, "bad", "⛔ 两个 80%，长得一样，意思完全不同", [
        "「Llama 405B 里将近 <tspan font-weight=\"700\">80% 的注意力</tspan>"
        "落在第一个 token 上」——&#160;这是<tspan font-weight=\"700\">权重占比</tspan>。",
        "「LLaMa 3.1 405B 里有 <tspan font-weight=\"700\">80% 的注意力头</tspan>"
        "形成了强 sink」——&#160;这是<tspan font-weight=\"700\">头的比例</tspan>"
        "（判据是阈值 ε=0.8）。",
        "⭐ 两句都出自同一篇论文，数字一样、含义毫不相干。"
        "<tspan font-weight=\"700\">讲的时候说串了，台下懂行的人一听就知道。</tspan>",
    ])
    yy = f.src(yy + 16,
               "装置与两句原话出自 Barbero 等《Why do LLMs attend to the first "
               "token?》（arXiv 2504.02732）Figure 1 caption："
               "「The presence of attention sinks slows down the mixing of "
               "information between tokens and hence makes Transformers more "
               "robust to perturbations of prompts」·「The presence of a sink "
               "draws attention away from the rest of the tokens, limiting the "
               "spread of perturbed information」",
               "⚠️ ①② 那两张网格是<tspan font-weight=\"700\">本课的线性简化模拟"
               "</tspan>（均匀注意力 ＋ 残差流 h ← h ＋ Attn(h)，sink 吸 80% 且 "
               "value 记 0），脚本当场跑并带断言 ——&#160;"
               "<tspan font-weight=\"700\">不是实测</tspan>。"
               "Barbero 的 Figure 1 本身也是示意图，他们的实测在 Figure 2",
               "⛔ 三条别引错：「pump the brakes」<tspan font-weight=\"700\">"
               "不是 Barbero 的原话</tspan>（全文零命中）；「泄压阀」是 MIT HAN "
               "Lab 博客的转述不是论文的词；<tspan font-weight=\"700\">"
               "「一滴墨水」这个画面是本课原创</tspan>，论文里没有任何生活比喻")
    f.save("fig3-overmix.svg", yy + 6)


main()
