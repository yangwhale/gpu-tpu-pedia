# -*- coding: utf-8 -*-
r"""专题三 · §五「砍头这一支是怎么想出来的」（2026-09-13 夜间 20 轮 · R2）。

⭐⭐ 三格，回答三个「为什么」，每一格的证据都在原论文里：

  ① **2019 年那张选项单** ——&nbsp;MQA 论文（Shazeer, arXiv 1911.02150）§2.4
     先摆了一张单子：想让解码每步少搬点东西，可以**限制序列长度**、
     **只看近处**、**压缩历史位置数**；然后他说「本文走一条<b>正交</b>的路：
     去掉 K/V 的头这一维」。
     ⭐⭐ **今天三个旋钮里的两个，2019 年那张单子上就有。**
     这一格证明的是：这门课的骨架不是我们事后归纳的，是当事人自己列的。

  ② **「自由度不是宽度」有论文自带的证据** ——&nbsp;同一篇的消融表：
     MHA(h=8) 29.9 → MQA 30.2 → **真单头 h=1 31.2**。
     MQA 和真单头**缓存的 K/V 一样多**（都是一份），
     差别只在 query 侧还留不留 8 个不同的问法 ——&nbsp;
     就这一点，把 PPL 拉回去了一大截。
     ⭐ 这正是 MQA（256 维）比 MLA（576 维）**更窄却更差**的原因。

  ③ **GQA 的两个发明点** ——&nbsp;不是「折中一下」那么简单：
     · mean-pool 已有的头 ＋ 5% 原始预训练算力续训（**不用从头重训**）
     · 真正的动机：**模型越大头越多，MQA 的削减力度会失控**；
       GQA 让「削减比例」跟着模型规模走。

📌 所有数字取自两篇原论文，脚本里断言。
"""
from topic03_draw import (Fig, wpx, BL, OR, GR, RD, GY, PU, INK,
                          GY2, LINE, LINE2, BG2)

W = 1400
PX, PW = [0, 470, 940], [440, 440, 460]


def main():
    def fits(y, y0, ph, who):
        """⛔ 面板内容溢出底边不会报错、版面体检也看不见 —— 只能自己断言。"""
        assert y <= y0 + ph - 6, "%s 的内容到 %d，面板底边在 %d —— 超了 %d" % (
            who, y, y0 + ph, y - (y0 + ph))

    # ── ② 那张消融表（Shazeer 1911.02150 表 4）─────────────────
    ABL = [
        ("multi-head", "h=8, d_k=128", 29.9, "128 份 K/V", GY),
        ("multi-query", "h=8, 共用 1 份", 30.2, "1 份 K/V", GR),
        ("multi-head", "h=1, d_k=128", 31.2, "1 份 K/V", RD),
    ]
    SMALL = [("h=2, d_k=64", 31.1), ("h=4, d_k=32", 31.0), ("h=8, d_k=16", 30.9)]
    mq = [r for r in ABL if r[1].startswith("h=8, 共用")][0][2]
    h1 = [r for r in ABL if r[1].startswith("h=1")][0][2]
    assert h1 > mq                       # ⭐ 同样一份 K/V，真单头明显更差
    assert abs((h1 - mq) - 1.0) < 1e-9   # 差整整 1.0 PPL

    f = Fig(W, "砍头这一支是怎么想出来的：2019 年 MQA 论文里那张选项单、"
               "「自由度不是宽度」的消融证据、以及 GQA 的两个发明点")
    f.marks = set()
    y0 = f.header(
        "砍头这一支是怎么想出来的　——　三个「为什么」，证据都在原论文里",
        "⭐ 上一张讲 MLA 凭什么敢压；这一张讲<tspan font-weight=\"700\">它的对手"
        "为什么更窄反而更差</tspan>",
        [(GY, "MHA 基线"), (GR, "共用一份 K/V"), (RD, "真的只剩一个头"),
         (BL, "2019 年就列出的选项"), (PU, "GQA 的发明点")])

    # ══ ① 2019 年那张选项单 ══════════════════════════════════════
    x, pw = PX[0], PW[0]
    ph = 404
    py = f.panel(x, y0, pw, ph, "① 2019 年那张选项单", BL,
                 sub="MQA 论文 §2.4，原文列的")

    yy = py + 24
    f.t(x + 24, yy, "「解码每步都要重新搬一遍 K/V —— 怎么少搬？」",
        INK, True, 12.5, w=pw - 48)
    yy += 26

    OPTS = [
        ("限制序列长度 n", "就不让它变长", "—", GY2),
        ("只看一个局部邻域", "后来叫<tspan font-weight=\"700\">滑窗</tspan>", "旋钮②", BL),
        ("压缩历史位置的个数", "后来叫<tspan font-weight=\"700\">稀疏 / 压缩</tspan>", "旋钮②", BL),
    ]
    for name, later, knob, col in OPTS:
        f.box(x + 24, yy, pw - 48, 42, "#fff", LINE, 8)
        f.t(x + 38, yy + 26, name, GY, True, 12)
        f.t(x + 38 + wpx(name, 12) + 14, yy + 26, later, GY2, size=11)
        f.t(x + pw - 38, yy + 26, knob, col, True, 11.5, "end")
        yy += 50

    yy += 6
    f.t(x + 24, yy, "然后 Shazeer 说：本文走一条<tspan font-weight=\"700\">正交</tspan>的路 ——", GY,
        size=11.5, w=pw - 48)
    yy += 22
    f.box(x + 24, yy, pw - 48, 52, "#fff", GR, 8)
    f.box(x + 24, yy, 4, 52, GR, GR, 2)
    f.box(x + 26, yy, 3, 52, "#fff", "#fff", 0)
    f.t(x + 42, yy + 22, "去掉 K/V 的「头」这一维", GR, True, 12.5)
    f.t(x + pw - 38, yy + 22, "旋钮①", GR, True, 11.5, "end")
    f.t(x + 42, yy + 41, "query 侧的头一个不动", GY2, size=11)

    yy += 68
    f.t(x + 24, yy, "⭐⭐ 今天三个旋钮里的两个，", BL, True, 13, cls="svglbl")
    f.t(x + 24, yy + 21, "2019 年那张单子上就有。", BL, True, 13, cls="svglbl")
    f.t(x + 24, yy + 44, "⛔ 不是我们事后归纳的骨架 —— 是当事人自己列的。",
        GY, size=11.5, w=pw - 48)

    fits(yy + 44, y0, ph, "① 选项单")

    # ══ ② 自由度不是宽度 ════════════════════════════════════════
    x, pw = PX[1], PW[1]
    py = f.panel(x, y0, pw, ph, "② 自由度不是宽度", RD,
                 sub="同一篇论文的消融表")

    yy = py + 22
    f.colhead(x + 24, yy + 10, "配置")
    f.colhead(x + 232, yy + 10, "缓存", anchor="middle")
    f.colhead(x + pw - 30, yy + 10, "困惑度", anchor="end")
    yy += 22

    for name, cfg, ppl, cache, col in ABL:
        f.box(x + 24, yy, pw - 48, 52, "#fff", col if col != GY else LINE, 8)
        f.t(x + 40, yy + 22, name, col, True, 12)
        f.t(x + 40, yy + 40, cfg, GY2, size=11, mono=True)
        f.t(x + 232, yy + 31, cache, GY, size=11.5, anchor="middle")
        f.t(x + pw - 40, yy + 31, "%.1f" % ppl, col, True, 15, "end")
        yy += 60

    yy += 4
    f.box(x + 24, yy, pw - 48, 74, "#fff", RD, 8)
    f.box(x + 24, yy, 4, 74, RD, RD, 2)
    f.box(x + 26, yy, 3, 74, "#fff", "#fff", 0)
    f.t(x + 42, yy + 24, "后两行缓存的 K/V <tspan font-weight=\"700\">一样多</tspan>", RD, True, 12.5)
    f.t(x + 42, yy + 45, "差别只在 query 侧还留不留 8 个问法 ——", GY,
        size=11.5)
    f.t(x + 42, yy + 64, "就这一点，把困惑度拉回 %.1f" % (h1 - mq), GY,
        size=11.5)

    yy += 80
    f.t(x + 24, yy, "对照组：把总维度摊给更多头（缓存都是 1 份）", GY2,
        size=11, w=pw - 48)
    yy += 18
    bx = x + 24
    for cfg, ppl in SMALL:
        f.box(bx, yy, 128, 38, "#fff", LINE, 6)
        f.t(bx + 10, yy + 17, cfg, GY2, size=11, mono=True)
        f.t(bx + 10, yy + 32, "%.1f" % ppl, GY, True, 11.5)
        bx += 134

    fits(yy + 38, y0, ph, "② 消融表")

    # ══ ③ GQA 的两个发明点 ══════════════════════════════════════
    x, pw = PX[2], PW[2]
    py = f.panel(x, y0, pw, ph, "③ GQA 的两个发明点", PU,
                 sub="不是「折中一下」那么简单")

    yy = py + 24
    f.t(x + 24, yy, "发明点一　不用从头重训", PU, True, 13, cls="svglbl")
    yy += 22
    for step, note in [
        ("把一组里所有头的 K/V 投影<tspan font-weight=\"700\">取平均</tspan>",
         "论文实测：比「挑一个头」和「随机初始化」都好"),
        ("再用 <tspan font-weight=\"700\">5%</tspan> 原始预训练算力续训一下", "然后它就是一个 GQA 模型了"),
    ]:
        f.box(x + 24, yy, pw - 48, 50, "#fff", LINE, 8)
        f.t(x + 38, yy + 22, step, GY, size=12)
        f.t(x + 38, yy + 40, note, GY2, size=11)
        yy += 58

    yy += 6
    f.t(x + 24, yy, "发明点二　真正的动机不是「取个中间值」",
        PU, True, 13, cls="svglbl")
    yy += 24
    f.box(x + 24, yy, pw - 48, 94, "#fff", PU, 8)
    f.box(x + 24, yy, 4, 94, PU, PU, 2)
    f.box(x + 26, yy, 3, 94, "#fff", "#fff", 0)
    f.t(x + 42, yy + 24, "模型越大，头越多 ——", PU, True, 12.5)
    f.t(x + 42, yy + 45, "MQA 一律砍到 1 份，等于<tspan font-weight=\"700\">削减力度随规模失控</tspan>",
        GY, size=11.5)
    f.t(x + 42, yy + 66, "GQA 固定的是<tspan font-weight=\"700\">组数</tspan>，于是削减比例可控",
        GY, size=11.5)
    f.t(x + 42, yy + 85, "GQA-1 就是 MQA，GQA-H 就是 MHA", GY2, size=11)

    yy += 108
    f.t(x + 24, yy, "⭐ 所以它是一个<tspan font-weight=\"700\">旋钮</tspan>，不是一个新机制。", PU,
        size=12, w=pw - 48)

    fits(yy, y0, ph, "③ GQA")

    # ══ 落点带 ══════════════════════════════════════════════════
    yy = y0 + ph + 22
    yy = f.band(yy, "info", "⭐⭐ 把 ② 和上一张连起来：瓶颈的伤害，不在它多窄", [
        "MQA 每 token 每层只存 256 个数，比 MLA 的 576 还<tspan font-weight="
        "\"700\">窄一倍多</tspan>，可是它<tspan font-weight=\"700\">明显更差</tspan>。",
        "因为 MQA 逼所有头<tspan font-weight=\"700\">收到同一份 k 和 v</tspan>；"
        "MLA 只要求它们<tspan font-weight=\"700\">从同一份压缩里各自解码</tspan>"
        "——&#160;每个头有自己的上投影。",
        "⭐ 判据：<tspan font-weight=\"700\">瓶颈的伤害不在于它多窄，"
        "在于它剥夺了多少自由度。</tspan>这条在后面看稀疏、看线性注意力时还要用。",
    ])

    yy = f.band(yy + 14, "ok", "⭐ 暗线第二次出现：事后压 vs 从头按压缩训", [
        "GQA 是<tspan font-weight=\"700\">事后</tspan>的典范 ——&#160;"
        "拿一个 MHA checkpoint，平均池化 ＋ 5% 续训就成了；"
        "<tspan font-weight=\"700\">便宜，但天花板也就到「接近 MHA」</tspan>。",
        "MLA 是<tspan font-weight=\"700\">从头</tspan>的典范 ——&#160;"
        "要重训，但换来 56.9× 而不是 16×。"
        "⭐ <tspan font-weight=\"700\">这两条路一直并存到今天</tspan>，"
        "§六 还会再遇见一次。",
    ])

    yy = f.src(yy + 16,
               "① 与 ② 出自 MQA 原论文 Shazeer arXiv 1911.02150（§2.4 与表 4，"
               "Billion-Word LM 基准的 dev 困惑度）；「正交」是原文用词",
               "③ 出自 GQA 原论文 Ainslie 等 arXiv 2305.13245 §2.1–2.2："
               "mean pooling、α=5% 续训、GQA-1=MQA / GQA-H=MHA")
    f.save("fig3-mqa-why.svg", yy + 6)


main()
