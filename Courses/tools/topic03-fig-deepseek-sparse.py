# -*- coding: utf-8 -*-
r"""专题三 · §6.6「DeepSeek 这一支的四步：NSA → DSA → CSA → HCA」

⭐⭐⭐ 2026-09-16 新画。现场原话：
  「NSA、CSA、HCA 全都给我画图详细讲解，因为这几个是 DeepSeek 的，
    所以大家都盯着呢。」

⛔ 本课原来把 NSA / CSA **明确标成「支线，当堂不讲」**，只讲 DSA。
  ⭐ 那个取舍在「一小时讲完」的前提下成立；现在前提没了，而且这四个名字是
    **同一条路上的四步**，拆开讲每一步都少一个对照组。

⭐⭐ 这张图存在的理由，是**那两条轴只有画出来才看得见**：
  横轴「压多狠」、纵轴「挑不挑」。四个名字摆上去，落点自己就浮出来了 ——
  **压得够狠的时候，就不需要挑了。**
  而这一句正好接住第九章那条「动态稀疏在 TPU 上每一层都难」：
  **HCA 不是把动态稀疏做快了，是把动态换成了静态。**

⛔⛔ **刻意没画的东西：**
  ① **一个性能数字都没有。** 本课没有这四者的对照实测，
     画柱子就是编。图上只画**结构**与**是否需要运行时决策**。
  ② Ⓑ 那条横轴**不是线性刻度**（1 / 4 / 128 等距摆）——
     它只表达「谁比谁压得狠」，不表达倍数关系。图上写明了。

📌 出处（全部公开）：
  · NSA ＝ Native Sparse Attention，arXiv 2502.11089（DeepSeek，2025-02）
  · CSA / HCA 的结构与参数取自 **MaxText 的公开实现**（Apache-2.0）：
    `AttentionType.COMPRESSED` 下 `compress_ratio == 4` 走 CSA（必须给
    indexer_mask），`compress_ratio > 4` 走 HCA（用编译期静态 mask）；
    HCA 的 docstring 自称 "DeepSeek-V4 Heavily Compressed Attention"，
    默认 compress_ratio=128、local_window=128。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE, LINE2)

W = 1400


def main():
    f = Fig(W, "DeepSeek 稀疏注意力的四步：NSA 用压缩、选择、滑窗三条支路加门控融合，"
               "块级选择且训练时就稀疏；DSA 换成一个轻量索引器挑单个 token；"
               "CSA 先把四个 token 压成一条再挑；HCA 压到一百二十八倍并且不再挑，"
               "于是 mask 可以在编译期算死。两条轴：压多狠，以及需不需要运行时挑")

    y0 = f.header(
        "DeepSeek 这一支的四步："
        "<tspan font-weight=\"700\">NSA → DSA → CSA → HCA</tspan>",
        "⭐ 四个名字看着各不相干，其实是<tspan font-weight=\"700\">同一条路上的四步</tspan>"
        " ——　而且最后一步<tspan font-weight=\"700\">把「挑」这件事整个取消了</tspan>",
        [(OR, "要运行时挑（动态）"), (GR, "编译期就定死（静态）")])

    # ══════════ Ⓐ NSA：三条支路 ═══════════════════════════════════
    PH = 400
    py = f.panel(0, y0, W, PH,
                 "Ⓐ NSA（2025-02）——　不是一条稀疏，是<tspan font-weight=\"700\">"
                 "三条支路并排跑，再用门控融合</tspan>", BL,
                 sub="⭐ 名字里的 Native 是重点：<tspan font-weight=\"700\">"
                     "训练的时候就稀疏</tspan>，不是训完再稀疏")

    # ⛔⛔ 渲染之后改的：三条支路原来用 蓝/橙/绿 当「区分色」，
    #   可 Ⓑ Ⓒ 里橙＝动态、绿＝静态是**语义色** ——&#160;同一张图里橙有两个意思。
    #   ⭐ 判据⑤：一张图里一个颜色只能有一个意思。
    #   ⭐⭐ 改成这里也用同一套语义之后，**反而多说出一件事**：
    #     NSA 三条支路里**只有「选择」那一条是动态的** ——&#160;
    #     而它正是后面 CSA 想缩小、HCA 想干掉的那一条。颜色自己把故事讲了。
    LANE = (
        ("① 压缩", "远处按块压成摘要", "静态　·　一个块一条", GR),
        ("② 选择", "挑出最相关的<tspan font-weight=\"700\">几块</tspan>", "⛔ 动态　·　跑起来才知道挑谁", OR),
        ("③ 滑窗", "身边那一段全留", "静态　·　窗口固定", GR),
    )
    LX, LW = 70, 380
    for i, (t1, t2, t3, col) in enumerate(LANE):
        x = LX + i * (LW + 30)
        f.box(x, py + 26, LW, 148, "#fff", LINE, 8)
        f.box(x, py + 26, LW, 4, col, col, 2)
        f.t(x + LW / 2.0, py + 60, t1, col, True, 19, "middle")
        f.t(x + LW / 2.0, py + 90, t2, INK, size=15, anchor="middle")
        f.t(x + LW / 2.0, py + 116, t3, GY, size=13.5, anchor="middle")
        f.line(x + LW / 2.0, py + 174, x + LW / 2.0, py + 208, GY2, 1.4)

    f.box(430, py + 208, 540, 54, "#f1f3f4", GY2, 27)
    f.t(700, py + 241, "门控融合　——　三条各自的结果按权重合起来", INK, True, 17,
        "middle")
    f.line(700, py + 262, 700, py + 292, GY2, 1.6)
    f.t(700, py + 318, "一个 token 的输出", INK, True, 17, "middle")

    f.box(70, py + 296, 300, 76, "#fef7e0", OR, 6)
    f.t(86, py + 320, "⭐ 三条里只有一条是动态的", OR, True, 16)
    f.t(86, py + 344, "后面三步，动的全是这一条", GY, size=13)

    f.box(1000, py + 296, 330, 76, "#e6f4ea", GR, 6)
    f.t(1016, py + 320, "⭐ 训练时就用", GR, True, 16)
    f.t(1016, py + 344, "不是训完再稀疏，所以模型是「长在稀疏上」的", GY, size=13)
    f._pan = None

    yy = f.band(py + PH + 22, "info", "NSA 为什么从第一天就好落到硬件上", [
        "因为它<tspan font-weight=\"700\">挑的是块，不是散落的单个 token</tspan> ——&#160;"
        "块是连续的一片，搬起来整齐。论文自己把这一点写进了标题："
        "<tspan font-weight=\"700\">hardware-aligned</tspan>。",
        "⛔ <tspan font-weight=\"700\">记住这个对照，下一格 DSA 正好反过来。</tspan>",
    ])

    # ══════════ Ⓑ 两条轴 ═════════════════════════════════════════
    PH2 = 470
    py2 = f.panel(0, yy + 26, W, PH2,
                  "Ⓑ 四个名字摆到同一张图上 ——　两条轴：<tspan font-weight=\"700\">"
                  "压多狠</tspan>　×　<tspan font-weight=\"700\">挑不挑</tspan>", BL,
                  sub="⚠️ 横轴<tspan font-weight=\"700\">不是线性刻度</tspan>"
                      "（1 / 4 / 128 等距摆）——　它只表达「谁比谁压得狠」")

    AX = py2 + 330
    f.line(110, AX, 1310, AX, GY2, 2.0)
    f.t(110, AX + 32, "不压", GY, True, 15)
    f.t(1310, AX + 32, "压得最狠", GY, True, 15, "end")
    f.t(700, AX + 60, "压　缩　比　（4 个 token 合 1 条 →　128 个合 1 条）", GY,
        size=14, anchor="middle")

    f.t(60, py2 + 40, "要运行时挑", OR, True, 16)
    f.t(60, py2 + 62, "（有 indexer）", GY, size=13)
    f.t(60, py2 + 240, "不用挑", GR, True, 16)
    f.t(60, py2 + 262, "（编译期定死）", GY, size=13)

    # (x, y, 名字, 副标题, 色, 两行说明)
    PTS = (
        (300, py2 + 84, "NSA", "块级选择 ＋ 压缩支路", OR,
         "挑「块」——　整齐", "2025-02　arXiv 2502.11089"),
        (560, py2 + 84, "DSA", "不压，纯挑 top-k 个 token", RD,
         "⛔ 挑散落的单 token ——　最碎", "本讲第六章的主角"),
        (860, py2 + 84, "CSA", "先压 4 倍，再挑", OR,
         # ⛔ 这里原来写「鸡小了 4 倍」，是对 fig3-chicken 那句
         #   「鸡生蛋没破 ——　但那只鸡小了 4 倍」的回指。
         #   ⭐ 但这张图里一个字都没提鸡生蛋，孤零零一个「鸡」被当成了错字（实测）。
         #   判据：**回指要么带上下文，要么就别回指** ——&#160;12.5px 的一行小字
         #   不是扛比喻的地方，直接说它是什么。
         "压过之后再挑 ——　候选少了 4 倍", "MaxText：compress_ratio ＝ 4"),
        (1200, py2 + 228, "HCA", "压 128 倍，<tspan font-weight=\"700\">不挑了</tspan>", GR,
         "⭐ 没有 indexer", "MaxText：compress_ratio ＞ 4"),
    )
    for x, y, name, sub_, col, l1, l2 in PTS:
        f.box(x - 108, y, 216, 96, "#fff", col, 8)
        f.box(x - 108, y, 216, 4, col, col, 2)
        f.t(x, y + 34, name, col, True, 22, "middle")
        f.t(x, y + 58, sub_, INK, size=13.5, anchor="middle")
        f.t(x, y + 80, l1, GY, size=12.5, anchor="middle")
        f.t(x, y + 118, l2, GY2, size=12, anchor="middle")
        f.line(x, y + 128, x, AX - 8, col, 1.2, "3 3")
        f.box(x - 6, AX - 6, 12, 12, col, col, 6)

    # ⛔ 这行字原来放在 (1030, py2+196)，**正好压在 CSA 那行出处上**（y 差 6px）。
    #   ⭐ 挪到 HCA 框的正上方 ——&#160;它标注的本来就是「到 HCA 这一步发生了什么」，
    #     位置对了，重叠也没了。判据：**标注要贴它说的那个东西。**
    f.elbow(860, py2 + 148, 1200, py2 + 222, GR, 2.2, 14)
    f.t(1200, py2 + 208, "⭐ 再压 32 倍　→　挑这一步整个没了", GR, True, 15,
        "middle")
    f._pan = None

    yy = f.band(py2 + PH2 + 22, "ok", "这张图真正要说的一句话", [
        "<tspan font-weight=\"700\">压得够狠的时候，就不需要挑了。</tspan>"
        "⭐ 挑，本来就是为了「在一大堆里只看几个」；"
        "可要是先把一大堆<tspan font-weight=\"700\">压成一小堆</tspan>，"
        "那就<tspan font-weight=\"700\">全看也无所谓了</tspan>。",
        "⛔ 代价写在横轴上：<tspan font-weight=\"700\">压 128 倍，分辨率就是没了</tspan> ——&#160;"
        "所以 HCA <tspan font-weight=\"700\">不能单独用</tspan>，"
        "它旁边还挂着一个<tspan font-weight=\"700\">局部滑窗</tspan>（默认 128）管近处。"
        "<tspan font-weight=\"700\">远处压得很糊，近处一个不落。</tspan>",
    ])

    # ══════════ Ⓒ 为什么 HCA 对 TPU 最友好 ═══════════════════════
    PH3 = 330
    py3 = f.panel(0, yy + 26, W, PH3,
                  "Ⓒ 落到 TPU 上：HCA 不是「把动态稀疏做快了」，是"
                  "<tspan font-weight=\"700\">把动态换成了静态</tspan>", BL,
                  # ⛔ 这里原来引了后面那份移植日志的原话，跟正文撞车（查重抓到）。
                  #   ⭐ 那句话是**别人日志里的原文**，该留在引用它的地方；
                  #     这张图只说它的推论。判据⑩ 的一个变体：**原话归引文，推论归图。**
                  sub="⭐ 这一格回答的是本讲后面那个问题："
                      "<tspan font-weight=\"700\">稀疏落到 TPU 上为什么难，"
                      "以及这一支是怎么绕开的</tspan>")

    f.box(60, py3 + 26, 620, 232, "#fff", OR, 8)
    f.box(60, py3 + 26, 620, 4, OR, OR, 2)
    f.t(370, py3 + 60, "动态（DSA / CSA）", OR, True, 19, "middle")
    for i, ln in enumerate((
            "跑起来才知道这一步该挑谁",
            "→　mask 只能在运行时现造",
            "→　要在 HBM 里放一张稠密的 mask",
            "→　还要占着算数部件去算它",
    )):
        f.t(96, py3 + 100 + i * 32, ln, GY, size=15)
    f.t(96, py3 + 234, "⛔ 每一层都在跟「运行时才知道」较劲", RD, True, 15)

    f.box(720, py3 + 26, 620, 232, "#fff", GR, 8)
    f.box(720, py3 + 26, 620, 4, GR, GR, 2)
    f.t(1030, py3 + 60, "静态（HCA）", GR, True, 19, "middle")
    for i, ln in enumerate((
            "窗口固定、压缩比固定",
            "→　mask 在<tspan font-weight=\"700\">编译期</tspan>就能算出来",
            "→　主机 CPU 上算好，打包成 bitmask",
            "→　直接塞进片上，HBM 里不存稠密 mask",
    )):
        f.t(756, py3 + 100 + i * 32, ln, GY, size=15)
    f.t(756, py3 + 234, "⭐ 运行时零决策，也零 mask 运算", GR, True, 15)
    f._pan = None

    yy = f.band(py3 + PH3 + 22, "ok", "把这四步连起来，是一条很干净的线", [
        "<tspan font-weight=\"700\">NSA 挑块</tspan>（整齐）→&#160;"
        "<tspan font-weight=\"700\">DSA 挑单 token</tspan>（更准，但最碎）→&#160;"
        "<tspan font-weight=\"700\">CSA 先压再挑</tspan>（把要挑的池子缩小）→&#160;"
        "<tspan font-weight=\"700\">HCA 压到不用挑</tspan>（决策整个消失）。",
        "⭐⭐ <tspan font-weight=\"700\">每一步都在把「运行时要做的决策」往编译期推。</tspan>"
        "⛔ 这跟本讲反复出现的那条判据是同一个形状："
        "<tspan font-weight=\"700\">一个麻烦最好的结局不是被解决，是不再存在。</tspan>",
    ])

    yy = f.src(yy + 24,
               "NSA ＝ Native Sparse Attention，arXiv <tspan font-weight=\"700\">"
               "2502.11089</tspan>（DeepSeek, 2025-02）：摘要原话是"
               "「coarse-grained token compression ＋ fine-grained token selection」，"
               "标题里写着 hardware-aligned 与 natively trainable",
               "CSA / HCA 的结构与默认参数取自 <tspan font-weight=\"700\">MaxText 的公开实现"
               "</tspan>（Apache-2.0）：两者同属 AttentionType.COMPRESSED；"
               "<tspan font-weight=\"700\">compress_ratio ＝ 4 走 CSA</tspan>（必须传 "
               "indexer_mask），<tspan font-weight=\"700\">compress_ratio ＞ 4 走 HCA"
               "</tspan>（用编译期静态 mask，默认 compress_ratio=128、local_window=128）。"
               "HCA 的 docstring 自称 “DeepSeek-V4 Heavily Compressed Attention”",
               "⛔ <tspan font-weight=\"700\">本图不含任何性能数字</tspan> ——&#160;"
               "本课没有这四者的对照实测，画柱子就是编。"
               "图上只画结构，以及「要不要运行时决策」",
               "⚠️ Ⓑ 的横轴<tspan font-weight=\"700\">不是线性刻度</tspan>，"
               "1 / 4 / 128 等距摆，只表达先后不表达倍数")
    f.save("fig3-deepseek-sparse.svg", yy + 6)


main()
