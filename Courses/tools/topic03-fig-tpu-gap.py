# -*- coding: utf-8 -*-
r"""专题三 · §十「这些注意力落到 TPU 上，难在哪」（2026-09-13 · TPU 轮 R27）。

⭐⭐⭐ 现场点的题：「注意力在 TPU 上跑起来有没有困难的地方？
   哪些本来是给 GPU 设计的，搬到 TPU 上有难度、需要克服？」

   这一张先把**结构性的错配**摆出来，因为不摆清楚，
   后面那些 kernel 技巧看起来就只是一堆技巧。

  ① **TPU 这一侧的三条硬约束**（不是缺点，是它快的原因）
     · XLA 是 **static-first**：形状要在编译期定死
     · 内存布局是 **tiled、粗粒度**的：细粒度切片本身就不便宜
     · 整条流水线为**规整访存**优化：一旦访问模式跟 layout 不对齐就要罚钱

  ② **现代注意力这一侧的三个动态性来源**（全是最近五年长出来的）
     · **ragged**：一个 batch 里各请求长度不同
     · **分页 KV**：一条序列的 KV 散在不连续的页上
     · **运行时 top-k**：这一步到底读哪 2048 条，**要跑起来才知道**

  ③ **两边一对上，就是三处具体的疼**

⛔ 最该引的一句话来自 Google 自己那篇 RPA 论文：
   「现有 LLM 推理 kernel 和服务系统**基本都是 GPU 中心的**，
     **还没有一套成熟的办法**把 LLM 负载高效地映射到 TPU 架构上。」
   ——&nbsp;这句话出自 2026 年 4 月，不是五年前。
"""
from topic03_draw import (Fig, wpx, BL, OR, GR, RD, GY, PU, CY, INK,
                          GY2, LINE, LINE2, BG2)

W = 1400
PX, PW = [0, 470, 940], [440, 440, 460]


def main():
    def fits(y, y0, ph, who):
        assert y <= y0 + ph - 6, "%s 到 %d，面板底边 %d" % (who, y, y0 + ph)

    f = Fig(W, "这些注意力落到 TPU 上难在哪：TPU 的三条硬约束（静态形状、"
               "tiled 粗粒度布局、偏好规整访存）对上现代注意力的三个动态性来源"
               "（ragged、分页 KV、运行时 top-k），一对上就是三处具体的疼")
    f.marks = set()
    y0 = f.header(
        "落到 TPU 上　——　先看清楚是<tspan font-weight=\"700\">哪两件事对不上</tspan>",
        "⛔ 不先摆清结构性的错配，后面那些 kernel 技巧看起来就只是一堆技巧",
        [(BL, "TPU 的硬约束"), (OR, "注意力的动态性"),
         (RD, "对上之后的疼"), (GR, "已经有的解法")])

    ph = 436

    # ══ ① TPU 侧 ════════════════════════════════════════════════
    x, pw = PX[0], PW[0]
    py = f.panel(x, y0, pw, ph, "① TPU 这一侧的三条硬约束", BL,
                 sub="⭐ 不是缺点 —— 是它快的原因")

    yy = py + 26
    for head, body, why in [
        ("XLA 是 static-first", "形状必须在<tspan font-weight=\"700\">编译期</tspan>定死",
         "换个形状就重编译；所以要 padding、要分桶"),
        ("内存布局 tiled、粗粒度", "最小 tile 是 <tspan font-weight=\"700\">(8, 128)</tspan> 这个量级",
         "想按 token 精细切一刀，本身就不便宜"),
        ("整条流水线为规整访存优化", "连续、可预测的搬运最划算",
         "访问模式一旦跟 layout 不对齐，就要罚钱"),
    ]:
        f.box(x + 22, yy, pw - 44, 100, "#fff", BL, 8)
        f.box(x + 22, yy, 4, 100, BL, BL, 2)
        f.box(x + 24, yy, 3, 100, "#fff", "#fff", 0)
        f.t(x + 40, yy + 26, head, BL, True, 12.5, w=pw - 76)
        f.t(x + 40, yy + 52, body, GY, size=11.5, w=pw - 76)
        f.t(x + 40, yy + 78, "→ " + why, GY2, size=11, w=pw - 76)
        yy += 110

    yy += 2
    f.t(x + 22, yy, "⭐ 这三条正是 TPU 在<tspan font-weight=\"700\">规整稠密</tspan>"
        "负载上", GY, size=11.5, w=pw - 44)
    f.t(x + 22, yy + 20, "效率那么高的原因。<tspan font-weight=\"700\">它们是一体两面。</tspan>",
        GY, size=11.5, w=pw - 44)
    fits(yy + 26, y0, ph, "①")

    # ══ ② 注意力侧 ══════════════════════════════════════════════
    x, pw = PX[1], PW[1]
    py = f.panel(x, y0, pw, ph, "② 现代注意力的三个动态性来源", OR,
                 sub="全是最近五年才长出来的")

    yy = py + 26
    for head, body, when in [
        ("ragged：一批里各人长度不同",
         "prefill 和 decode 还混在同一个 batch 里", "vLLM 那套调度带来的"),
        ("分页 KV：一条序列散在多页上",
         "页与页之间<tspan font-weight=\"700\">地址不连续</tspan>", "PagedAttention 带来的"),
        ("运行时 top-k：这一步读哪 2048 条",
         "<tspan font-weight=\"700\">要跑起来才知道</tspan> ——&#160;编译期算不出来", "DSA / NSA 这一支带来的"),
    ]:
        f.box(x + 22, yy, pw - 44, 100, "#fff", OR, 8)
        f.box(x + 22, yy, 4, 100, OR, OR, 2)
        f.box(x + 24, yy, 3, 100, "#fff", "#fff", 0)
        f.t(x + 40, yy + 26, head, OR, True, 12.5, w=pw - 76)
        f.t(x + 40, yy + 52, body, GY, size=11.5, w=pw - 76)
        f.t(x + 40, yy + 78, "来源：" + when, GY2, size=11, w=pw - 76)
        yy += 110

    yy += 2
    f.t(x + 22, yy, "⛔ 注意这三条<tspan font-weight=\"700\">全是在 GPU 上先长出来的</tspan>",
        OR, True, 12.5, w=pw - 44)
    f.t(x + 22, yy + 20, "——&#160;它们默认了一台「随手 gather 不太贵」的机器。",
        GY, size=11.5, w=pw - 44)
    fits(yy + 26, y0, ph, "②")

    # ══ ③ 对上之后的三处疼 ══════════════════════════════════════
    x, pw = PX[2], PW[2]
    py = f.panel(x, y0, pw, ph, "③ 一对上，就是三处具体的疼", RD,
                 sub="RPA 论文 §2 逐条点了名")

    yy = py + 26
    for head, body in [
        ("KV 要从<tspan font-weight=\"700\">动态算出来的、不连续的地址</tspan> gather",
         "原文：这让 <tspan font-weight=\"700\">DMA 调度</tspan>变得困难"),
        ("KV 更新要 <tspan font-weight=\"700\">scatter</tspan> 进只填了一半的页",
         "decode 时还是<tspan font-weight=\"700\">单 token 粒度</tspan>的写"),
        ("形状只有<tspan font-weight=\"700\">跑起来</tspan>才知道",
         "而 XLA 的整套优化都建立在「形状已知」上"),
    ]:
        f.box(x + 22, yy, pw - 44, 82, "#fff", RD, 8)
        f.box(x + 22, yy, 4, 82, RD, RD, 2)
        f.box(x + 24, yy, 3, 82, "#fff", "#fff", 0)
        f.t(x + 40, yy + 28, head, RD, True, 12.5, w=pw - 76)
        f.t(x + 40, yy + 56, body, GY, size=11.5, w=pw - 76)
        yy += 92

    yy += 4
    f.box(x + 22, yy, pw - 44, 92, "#fff", INK, 8)
    f.t(x + 38, yy + 26, "⭐⭐ 这不是「TPU 不行」", INK, True, 13,
        cls="svglbl")
    f.t(x + 38, yy + 50, "是<tspan font-weight=\"700\">这一批机制是在另一台机器上想出来的</tspan>，",
        GY, size=11.5, w=pw - 76)
    f.t(x + 38, yy + 72, "它们把「gather 不太贵」当成了背景假设。", GY,
        size=11.5, w=pw - 76)
    fits(yy + 92, y0, ph, "③")

    # ══ 落点带 ══════════════════════════════════════════════════
    yy = y0 + ph + 22
    yy = f.band(yy, "bad", "⛔ 这句话值得原样念一遍 —— 它出自 2026 年 4 月，不是五年前", [
        "「现有 LLM 推理 kernel 和服务系统<tspan font-weight=\"700\">基本都是 GPU 中心的</tspan>，"
        "而且<tspan font-weight=\"700\">还没有一套成熟的办法</tspan>"
        "把 LLM 负载高效地映射到 TPU 架构上。」",
        "——&#160;Google 自己那篇 Ragged Paged Attention 论文的摘要（arXiv 2604.15464）。",
        "⭐ 所以这一节讲的不是「怎么调参」，是<tspan font-weight=\"700\">一个还在打开的工程战场</tspan>。",
    ])

    yy = f.band(yy + 14, "info", "⭐ 一条可迁移的判据：看一个机制默认了什么样的机器", [
        "ragged 调度、分页 KV、运行时 top-k ——&#160;"
        "这三样<tspan font-weight=\"700\">都默认「随手 gather 不太贵」</tspan>。"
        "在一台为规整访存优化的机器上，这个假设<tspan font-weight=\"700\">不成立</tspan>。",
        "⭐ 所以移植的活儿不是「翻译代码」，是"
        "<tspan font-weight=\"700\">把那个隐含的硬件假设找出来，再换一个等价但规整的做法</tspan>。"
        "下面两张图讲的就是这个「换法」。",
    ])

    yy = f.src(yy + 16,
               "三处疼与那句摘要出自 Ragged Paged Attention（Jiang 等，"
               "arXiv 2604.15464，2026-04）§1 与 §2.4；TPU 的三条约束亦见该文 §1",
               "⚠️ 「最小 tile 是 (8,128) 这个量级」是该文举的例子（BF16(12,128) 被"
               "补齐到 BF16(16,128)），不同数据类型与配置下 tile 形状不同")
    f.save("fig3-tpu-gap.svg", yy + 6)


main()
