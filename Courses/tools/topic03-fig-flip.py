# -*- coding: utf-8 -*-
r"""专题三 · §2「为什么是现在」—— 同一个模型，两个场景，主角换人了

⭐⭐⭐ 2026-09-13 夜间 R20 新画。装置偷自知乎 姜富春《彻底理解 MLA》
   （zhuanlan.zhihu.com/p/16730036197）：**同一个模型、只把 batch 和上下文拧一下，
   瓶颈就从「参数」换成了「KV cache」。**

⭐ 为什么这个装置值钱：它回答的是一个**时间线上的怪事** ——
   这个形状 2017 年就造出来了，2019 年 MQA 那篇就指着它说是问题，
   可真正全行业动手改，是 2024 年以后。**中间那几年，技术一个字没变。**
   变的是**工作负载**。这张图把「变的是什么」画出来。

⛔ 画法上唯一的关键：**两根柱子里「权重」那一段必须像素级相同**
   ——&nbsp;同色、同高、同位置。读者要看到的是「什么都没改，主角却换了人」。
   如果两段高度稍有不同，这张图就废了。

📌 所有数都是本课前面已经核过的，这里只做除法：
   · 权重 625 GiB（671B 原生 FP8，1 B/参数）
   · MLA 的 KV：61 层 · 128K · bf16 · 一个用户 ＝ 8.58 GiB
   · 反事实对照：同一个模型假如用 MHA ＝ 488 GiB（同口径）
   ⚠️ 488 / 8.58 ＝ 56.9，正好对上 §五 那个 4.571 × 12.4 —— 互为交叉验证。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, GY, PU, INK, GY2, LINE, LINE2,
                          BG2, wpx, wrap_rich)

W = 1400
WEIGHT = 625.0        # GiB，671B 原生 FP8
KV128K = 8.58         # GiB，MLA · 61 层 · 128K · bf16 · 一个用户
KVMHA = 488.0         # GiB，同口径的反事实：假如它用 MHA
DEV = 94.74           # GiB / device，TPU v7


def main():
    A = KV128K * (4 / 128.0) * 1          # 一个人 · 4K
    B = KV128K * 1.0 * 64                 # 64 个人 · 128K
    # ⛔ 这个等式是这张图和 §五 之间的交叉验证，对不上就说明有一处口径错了。
    assert abs(KVMHA / KV128K - 56.9) < 0.1, KVMHA / KV128K
    nA = int(-(-(WEIGHT + A) // DEV))
    nB = int(-(-(WEIGHT + B) // DEV))
    assert nB > nA

    f = Fig(W, "同一个模型，两个场景：瓶颈从参数换成了 KV cache")
    yy = f.header(
        "为什么是现在 ——&#160;同一个模型，两个场景，<tspan font-weight=\"700\">"
        "主角换人了</tspan>",
        "这个形状 2017 年就造出来了，2019 年就有人指着它说是问题，"
        "可全行业真正动手改是 2024 年以后。"
        "<tspan font-weight=\"700\">中间那几年，技术一个字没变。</tspan>"
        "——&#160;这张图画的是那个变了的东西。",
        legend=[(GY2, "模型权重（两边完全一样）"), (RD, "KV cache"),
                (BL, "要几张卡")])

    # ══ ① 两根柱子 ════════════════════════════════════════════════
    PH1 = 470
    top = f.panel(0, yy, W, PH1,
                  "① 同一个模型（DeepSeek-V3）。"
                  "<tspan font-weight=\"700\">权重那一段一个字节没动。</tspan>",
                  RD, tag="61 层 · bf16 · MLA")
    BASE = top + 348                       # 地面
    SCALE = 300.0 / (WEIGHT + B)           # 最高那根柱占 300px
    BW = 190
    for i, (ttl, sub_, kv, nd) in enumerate([
        ("一个人，短对话", "上下文 4K", A, nA),
        ("64 个人，长文档", "上下文 128K", B, nB),
    ]):
        cx = 180 + i * 330
        hw = WEIGHT * SCALE
        hk = kv * SCALE
        # ⭐ 权重那一段：两根柱子里必须完全一样 —— 同一个表达式算出来的
        f.box(cx, BASE - hw, BW, hw, "#f1f3f4", GY2, 6)
        f.t(cx + BW / 2, BASE - hw / 2 - 6, "权重", GY, bold=True, size=18,
            anchor="middle")
        f.t(cx + BW / 2, BASE - hw / 2 + 18, "%.0f GiB" % WEIGHT, GY, size=16,
            anchor="middle")
        if hk >= 14:
            f.box(cx, BASE - hw - hk, BW, hk, "#fce8e6", RD, 6)
            f.t(cx + BW / 2, BASE - hw - hk / 2 - 6, "KV cache", RD, bold=True,
                size=18, anchor="middle")
            f.t(cx + BW / 2, BASE - hw - hk / 2 + 18, "%.0f GiB" % kv, RD,
                size=16, anchor="middle")
        else:
            f.box(cx, BASE - hw - 3, BW, 3, RD, RD, 1)
            f.t(cx + BW / 2, BASE - hw - 14,
                "KV cache　%.2f GiB（细到画不出来）" % kv, RD, bold=True,
                size=16, anchor="middle")
        f.line(cx, BASE + 1, cx + BW, BASE + 1, LINE, 1.2, arrow=False)
        f.t(cx + BW / 2, BASE + 30, ttl, INK, bold=True, size=19,
            anchor="middle", cls="svglbl")
        f.t(cx + BW / 2, BASE + 54, sub_, GY2, size=15, anchor="middle")
        f.t(cx + BW / 2, BASE + 82, "要 %d 张卡" % nd, BL, bold=True, size=18,
            anchor="middle")

    # 右边：三个数
    RX = 780
    f.box(RX, top + 28, 596, 300, "none", LINE, 9)
    f.t(RX + 20, top + 60, "把这两根柱子读成三个数", INK, bold=True, size=19,
        cls="svglbl")
    for j, (lab, val, col) in enumerate([
        ("权重", "%.0f GiB  →  %.0f GiB　（一个字节没动）"
         % (WEIGHT, WEIGHT), GY),
        ("KV cache", "%.2f GiB  →  %.0f GiB　（涨了 %.0f 倍）"
         % (A, B, B / A), RD),
        ("KV 占总量", "%.2f%%  →  %.1f%%" % (100 * A / (WEIGHT + A),
                                             100 * B / (WEIGHT + B)), RD),
        ("要几张卡", "%d  →  %d　（翻了将近一倍）" % (nA, nB), BL),
    ]):
        yj = top + 100 + j * 46
        f.t(RX + 20, yj, lab, col, bold=True, size=17)
        f.t(RX + 150, yj, val, INK if j else GY, size=17, mono=True)
    f.t(RX + 20, top + 300,
        "⭐⭐ <tspan font-weight=\"700\">什么都没改。</tspan>"
        "是<tspan font-weight=\"700\">工作负载</tspan>变了，不是技术变了。",
        INK, size=18)

    # ══ ② 而这已经是 MLA 之后的数字 ════════════════════════════════
    yy = top + PH1 + 26
    PH2 = 262
    top = f.panel(0, yy, W, PH2,
                  "② 别忘了：上面那根红柱子<tspan font-weight=\"700\">已经是 MLA "
                  "压过之后</tspan>的 ——&#160;如果它用 MHA 呢", BL,
                  tag="同口径的反事实")
    for i, (ttl, body, col) in enumerate([
        ("一个人，128K",
         "MLA：<tspan font-weight=\"700\">%.2f GiB</tspan>。"
         "MHA：<tspan font-weight=\"700\">%.0f GiB</tspan> ——&#160;"
         "比整个模型权重的四分之三还多，<tspan font-weight=\"700\">"
         "就为了一个人。</tspan>" % (KV128K, KVMHA), OR),
        ("64 个人，128K",
         "MLA：%.0f GiB，还算得出要几张卡。"
         "MHA：<tspan font-weight=\"700\">%.1f TiB</tspan> ——&#160;"
         "<tspan font-weight=\"700\">这时候问题已经不是「要几张卡」，是「做不了」。"
         "</tspan>" % (B, KVMHA * 64 / 1024.0), RD),
    ]):
        bx = 30 + i * 684
        f.box(bx, top + 32, 656, 132, "none", LINE, 9)
        f.badge(bx + 18, top + 48, i + 1, col)
        f.t(bx + 64, top + 70, ttl, col, bold=True, size=19, cls="svglbl")
        yj = top + 104
        for r in wrap_rich(body, 622, 16 * 1.12):
            f.t(bx + 18, yj, r, GY, size=16)
            yj += 24
    f.t(30, top + PH2 - 44,
        "⭐⭐ 所以 §五 那个 <tspan font-weight=\"700\">56.9×</tspan>"
        "（＝ %.0f ÷ %.2f）不是一次「优化」——&#160;"
        "<tspan font-weight=\"700\">它是把这件事从「做不了」变成「做得了」。</tspan>"
        % (KVMHA, KV128K), INK, size=17)

    # ══ 落点 ══════════════════════════════════════════════════════
    yy = top + PH2 + 30
    yy = f.band(yy, "info", "这张图解释了一条时间线上的怪事", [
        "<tspan font-weight=\"700\">2017</tspan> 年这个形状就造出来了；"
        "<tspan font-weight=\"700\">2019</tspan> 年 MQA 那篇论文的摘要里"
        "就写着「incremental inference is often slow, due to the "
        "memory-bandwidth cost of repeatedly loading the large keys and "
        "values tensors」——&#160;<tspan font-weight=\"700\">问题早就被指出来了</tspan>。",
        "⛔ 可全行业真正动手改，是 <tspan font-weight=\"700\">2024</tspan> 年以后。"
        "中间那几年，<tspan font-weight=\"700\">技术一个字没变</tspan>。",
        "⭐⭐ 变的是这张图右边那根柱子：<tspan font-weight=\"700\">"
        "上下文从几千涨到十几万，同时服务的人从一个涨到几十个</tspan>。"
        "两个都在乘，而它们乘的是<tspan font-weight=\"700\">同一项</tspan>。",
    ])
    yy = f.band(yy + 14, "ok", "顺手给一条提问顺序，本讲后面一直在用", [
        "<tspan font-weight=\"700\">问「省了多少」之前，先问「省的是哪一样」。</tspan>"
        "权重是<tspan font-weight=\"700\">所有人共享一份</tspan>，"
        "KV cache 是<tspan font-weight=\"700\">每人一份</tspan> ——&#160;"
        "它们根本不是同一类开销。",
        "⭐ 所以 KV cache 不是「显存里的一项」，它直接决定"
        "<tspan font-weight=\"700\">你能同时服务多少人</tspan>。"
        "<tspan font-weight=\"700\">这一条到专题六会变成 batch size 的硬上限。</tspan>",
    ])
    yy = f.src(yy + 16,
               "装置偷自知乎 姜富春《deepseek 技术解读(1)－彻底理解 MLA》"
               "（zhuanlan.zhihu.com/p/16730036197）——&#160;"
               "他用 Qwen-72B 做的这个对照，本图换成本讲一直在用的 V3 口径重算",
               "⚠️ 数全部来自本讲前面已核过的三个：权重 625 GiB（671B 原生 FP8）·"
               "MLA 的 KV 8.58 GiB（61 层 · 128K · bf16 · 一个用户）·"
               "反事实 MHA 488 GiB（同口径）。"
               "<tspan font-weight=\"700\">488 ÷ 8.58 ＝ 56.9</tspan>，"
               "正好对上 §五 那个 4.571 × 12.4 ——&#160;两条路算出同一个数，互为交叉验证",
               "📌 「要几张卡」按 <tspan font-weight=\"700\">TPU v7 每 device "
               "94.74 GiB</tspan> 算（这个数本课的 AOT 工具链里核过："
               "编译器自己报的 95.38G − 94.74G ＝ 656.93M，只有按 1024 才成立）。"
               "⚠️ 换别的硬件只是这两根柱子的刻度变，"
               "<tspan font-weight=\"700\">结论不变</tspan>；"
               "而且这笔账<tspan font-weight=\"700\">只算了装得下装不下，没算带宽</tspan>")
    f.save("fig3-flip.svg", yy + 6)


main()
