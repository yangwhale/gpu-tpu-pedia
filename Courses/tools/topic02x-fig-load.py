# -*- coding: utf-8 -*-
r"""外传 图 X-4 · **把活放回那两条线上** —— 谁喂得饱 560，谁喂不饱。

════════════════════════════════════════════════════════════════════
⭐⭐ 全讲的收口就是这一张
════════════════════════════════════════════════════════════════════
X-1 给了两条线（H100 295 / v6e 560），X-2、X-3 说清这两条线是从哪来的。
这一张把**各种活**摆到同一根轴上，于是结论自己会跳出来：

    · 落在两条线**左边**的活 ——&nbsp;两颗都在饿着，**v6e 饿得更狠**
    · 落在两条线**右边很远**的活 ——&nbsp;两颗都喂得饱，**此时比的是算力**
      而算力 918 对 989.5 ＝ **93%，基本打平**

⭐⭐ 于是那句「v6e 适合扩散」有了精确说法：
   **不是它跑扩散更快，是扩散把它的短板（带宽）挡在了瓶颈之外，
   只剩它不吃亏的那一项（算力）在起作用。**
   同一句话原样适用于 **prefill 重、decode 轻** 的任务 ——&nbsp;
   现场原话：「都是一个道理，因为 HBM 弱，但是算力强」。

════════════════════════════════════════════════════════════════════
📌 这根轴上的数是怎么来的（L100 只用一条规则）
════════════════════════════════════════════════════════════════════
**规则：算术强度 ≈ 一次前向里，同一份权重被多少个「位置」共用。**

  把一层的权重从 HBM 搬上来是固定开销；它服务的位置越多，
  这笔搬运就被摊得越薄。位置数 ≈ 强度。

  ⭐ 这不是新东西：专题二里 RNN 那一节算出「强度 ＝ batch size」，
    是同一条规则的特例；scaling-book 里 FSDP 那个
    「每卡 token 数要大于 C/W」的门槛，也是同一条。

| 活 | 位置数 ≈ 强度 | 怎么来的 |
|---|---|---|
| LLM decode，batch 1 | **1** | 一次只产 1 个 token |
| LLM decode，batch 64 | **64** | 64 条请求共用一次权重搬运 |
| LLM prefill，8K 上下文 | **8,192** | 整段一次过 |
| 文生图（1024×1024） | **≈ 16,000** | VAE 下采样 8 倍 → latent 128×128 ＝ 16,384 个位置 |
| 文生视频 | **再多一到两个数量级** | 多了时间维；具体数随模型而变，本图只标量级 |

⛔ **只标量级，不标精确值。** patchify 的块大小、模型层宽都会让这些数
   上下浮动几倍 ——&nbsp;⭐ 但**结论只依赖「在线的哪一边」，不依赖精确值**，
   而 16,000 与 560 差着一个半数量级，浮动几倍不改变边。
   这一条要写在图上，否则读者会拿它去做定量推算。
"""
from topic03_draw import (Fig, wpx, _sz, LINE, LINE2,
                          BL, OR, GR, RD, GY, GY2, PU, CY, INK)

import math

W = 1400

# (名字, 强度, 颜色, 说明)
LOADS = (
    ("LLM decode　batch 1", 1, RD, "一次只产 1 个 token"),
    ("LLM decode　batch 64", 64, RD, "64 条请求共用一次权重搬运"),
    ("LLM prefill　8K 上下文", 8192, GR, "整段一次过"),
    ("文生图　1024 × 1024", 16384, BL, "latent 128×128 ＝ 16,384 个位置"),
    ("文生视频", 300000, BL, "再多一到两个数量级（只标量级）"),
)
LINES = ((295, "H100", GY2), (560, "v6e", RD))


def main():
    f = Fig(W, "把各种负载按算术强度摆在一根对数轴上：LLM decode batch 1 是 1，"
               "batch 64 是 64，都远在 H100 的 295 和 v6e 的 560 这两条线左边；"
               "而 LLM prefill 8K 是 8192、文生图约 16000、文生视频更高，"
               "都远在两条线右边")
    f.marks = set()
    y = f.header(
        '把活放回那两条线上 ——&#160;'
        '<tspan font-weight="700">扩散模型落在很右边，而且是甩开两个数量级</tspan>',
        '⭐ 一条规则就够：<tspan font-weight="700">强度 ≈ 同一份权重被多少个「位置」共用</tspan>。'
        '权重从 HBM 搬上来是固定开销，服务的位置越多，这笔搬运摊得越薄。',
        [(RD, "喂不饱：卡在搬运上"), (GR, "喂得饱"), (BL, "扩散：甩开两个数量级")])

    AX0, AX1 = 300, W - 150
    LO, HI = 0.5, 1.0e6

    def xf(v):
        return AX0 + (AX1 - AX0) * (math.log10(v) - math.log10(LO)) / \
            (math.log10(HI) - math.log10(LO))

    ROW = 58
    top = y + 56

    # ⛔ 两条门槛线**先画**（背景层）——&nbsp;图 X-1 那一次就是把它们
    #   画在最后，把点和数字全盖住了。
    # ⛔ 295 和 560 在对数轴上只隔一点点，两个标签平放**必然重叠**
    #   （几何探针当场报了 H100 295 ⟂ v6e 560）。⭐ 错开成一高一低，
    #   并各自用一小段引线接回自己那条竖线 ——&nbsp;
    #   **不是把字缩小，而是换一层去放。**
    for k, (v, lab, col) in enumerate(LINES):
        f.line(xf(v), top - 22, xf(v), top + len(LOADS) * ROW - 10,
               col, 1.6 if col == RD else 1.2, dash="4 4", arrow=False)
        ly = top - 50 if k == 0 else top - 28
        f.line(xf(v), ly + 4, xf(v), top - 22, col, 1, arrow=False)
        f.t(xf(v) + (-8 if k == 0 else 8), ly,
            "%s　%d" % (lab, v), col, bold=True, size=_sz(12),
            anchor="end" if k == 0 else "start")
    # 两条线之间那一小段：标出「差 1.9 倍」
    f.poly("M %.1f %d L %.1f %d L %.1f %d L %.1f %d Z"
           % (xf(295), top - 18, xf(560), top - 18,
              xf(560), top + len(LOADS) * ROW - 10,
              xf(295), top + len(LOADS) * ROW - 10), "#fef7e0")

    for i, (name, v, col, note) in enumerate(LOADS):
        yy = top + i * ROW
        f.t(0, yy + 4, name, INK, bold=True, size=_sz(12))
        f.t(0, yy + 24, note, GY2, size=_sz(11), w=AX0 - 20)
        f.line(AX0, yy + 8, AX1, yy + 8, LINE2, 1, arrow=False)
        f.line(AX0, yy + 8, xf(v), yy + 8, col, 3.0, arrow=False)
        f.box(xf(v) - 7, yy + 1, 14, 14, col, col, 7)
        lab = ("%d" % v) if v < 1000 else ("%s" % format(v, ","))
        if v >= 100000:
            lab = "10 万 ～ 100 万"
        f.t(xf(v) + 16, yy + 13, lab, col, bold=True, size=_sz(13))

    ay = top + len(LOADS) * ROW - 2
    f.line(AX0, ay, AX1, ay, GY2, 1.2, arrow=False)
    for e in range(0, 7):
        v = 10 ** e
        if v < LO or v > HI:
            continue
        f.line(xf(v), ay, xf(v), ay + 5, GY2, 1, arrow=False)
        f.t(xf(v), ay + 20, format(v, ","), GY2, size=_sz(11), anchor="middle")
    f.t((AX0 + AX1) / 2.0, ay + 40,
        "算术强度（FLOP / byte，对数轴）——&#160;越往右，越是「算得多、搬得少」",
        GY, size=_sz(11), anchor="middle")

    # 左右两个区的标注
    f.t(xf(20), ay + 66, "← 这一边：卡在搬运上，算力再强也闲着",
        "#a50e0e", bold=True, size=_sz(12), anchor="middle")
    f.t(xf(30000), ay + 66, "这一边：喂得饱，此时比的是算力 →",
        "#0d652d", bold=True, size=_sz(12), anchor="middle")

    y = ay + 88

    y = f.band(y, "ok",
               "所以「v6e 适合扩散」的精确说法是这个",
               ['⛔ <tspan font-weight="700">不是它跑扩散更快。</tspan>'
                '是扩散把它的短板<tspan font-weight="700">挡在了瓶颈之外</tspan> ——&#160;'
                '强度上万，离 560 差着一个半数量级，那根窄管子根本没成为瓶颈。',
                '⭐ 于是只剩下没被挡住的那一项在起作用：'
                '<tspan font-weight="700">算力 918 对 989.5，是 H100 的 93%</tspan>。'
                '短板不参与，长板打平 ——&#160;这就是「合适」的全部含义。',
                '⭐⭐ 反过来也成立：'
                '<tspan font-weight="700">batch 小的 decode 落在最左边</tspan>，'
                '那里比的全是带宽，而 v6e 只有 H100 的 49% ——&#160;'
                '<tspan font-weight="700">同一颗芯片，在那种活上就是最吃亏的。</tspan>'])

    y = f.band(y + 14, "info",
               "同一条道理，原样适用于 prefill 重、decode 轻的任务",
               ['prefill 是<tspan font-weight="700">整段一次过</tspan>，'
                '几千上万个位置共用一次权重搬运 ——&#160;它跟扩散落在轴上的同一边。',
                '⭐ 所以选型判据可以压成一句：'
                '<tspan font-weight="700">先看你的活在这根轴上落在哪儿</tspan>，'
                '再看那一边比的是什么。落右边比算力，v6e 打平；落左边比带宽，v6e 吃亏。',
                '⛔ 这根轴<tspan font-weight="700">只标量级，不标精确值</tspan>：'
                'patchify 块大小、层宽都会让这些数上下浮动几倍。'
                '⭐ 但结论只依赖「在线的哪一边」——&#160;'
                '16,000 对 560 差一个半数量级，浮动几倍不改变边。'])

    y = f.src(y + 18,
              '规则「强度 ≈ 位置数」是专题二 RNN 一节那条「强度 ＝ batch size」的推广；'
              '门槛 295 / 560 见本讲图 X-1（官方规格表当场除出来的）',
              '⚠️ 位置数是量级估算：文生图按 1024×1024 经 VAE 下采样 8 倍得 latent 128×128；'
              '文生视频再加时间维，随模型而变，本图只给量级。⛔ 不要拿这根轴做定量推算。')
    f.save("figx-4.svg", y + 6)


main()
