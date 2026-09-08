# -*- coding: utf-8 -*-
r"""外传 图 X-6 · **一颗装得下吗** —— 为什么 v6e 的 Pod 只有 256 颗。

════════════════════════════════════════════════════════════════════
⭐ 这一张回答 X-2 里一个没解释的反常
════════════════════════════════════════════════════════════════════
X-2 列过三条对外规格，看着像是「减配」：

    ICI 端口 4 个（v7 是 6）· 2D 环面（v7 是 3D）· Pod 256 颗（v7 是 9,216）

⭐⭐ 但把扩散模型的**权重体积**摆出来，这三条立刻变得合理：
   **这一族模型基本在「一颗到几颗」的量级上**，
   而 v7 那套 3D 环面、九千多颗的规格，是为「一个模型摊在几千颗上」准备的。
   ——&nbsp;**v6e 不打那场仗，所以不用付那个成本。**

════════════════════════════════════════════════════════════════════
📌 权重体积怎么算的（bf16，每参数 2 字节；只算权重，不含激活与优化器）
════════════════════════════════════════════════════════════════════
| 模型 | 参数量 | 出处 | bf16 权重 |
|---|---|---|---|
| SDXL | 3.5B | 本仓库 SDXL/README | 7.0 GB |
| HunyuanVideo-1.5 | 8.3B | 腾讯官方 GitHub 原文「only 8.3B parameters」 | 16.6 GB |
| FLUX.1 [dev] | 12B | BFL 官方模型卡「12 billion parameter rectified flow transformer」 | 24 GB |
| Wan2.1-T2V-14B | 14B | Wan-AI 官方模型卡 | 28 GB |
| Wan2.2-T2V-A14B | 总 27B / 每步激活 14B | Wan-AI 官方模型卡原文 | 54 GB（总） |
| 对照：Qwen3.5-397B | 397B | 本课模型表 | 794 GB |
| 对照：DeepSeek-V3 | 671B | 本课模型表 | 1,342 GB |

⛔ **不要把这张图讲成「扩散都装得进一颗」**——&nbsp;它不是。
   Wan2.1-14B 的 28 GB 已经贴着 32 GB 的边（激活只剩 4 GB 余量），
   Wan2.2 那个 27B 总权重直接超了。
   ⭐ 图上要如实画出这两条**踩线和超线**的 ——&nbsp;
     一张「结论过于整齐」的图，台下第一时间就会怀疑它。

⚠️ 只算权重。真跑起来还要激活、KV（扩散没有）、中间 latent、编译缓存。
   这条要写在图上，否则「28 GB < 32 GB 所以能跑」会被当成结论。
"""
import math

from topic03_draw import (Fig, wpx, _sz, LINE, LINE2,
                          BL, OR, GR, RD, GY, GY2, PU, CY, INK)

W = 1400
CHIP_GB = 32.0        # 一颗 v6e 的 HBM（官方规格表）

# (名字, 参数量 B, 是不是扩散, 备注)
MODELS = (
    ("SDXL", 3.5, True, "文生图"),
    ("HunyuanVideo-1.5", 8.3, True, "文生视频"),
    ("FLUX.1 [dev]", 12.0, True, "文生图"),
    ("Wan2.1-T2V-14B", 14.0, True, "文生视频　⚠️ 贴着边"),
    ("Wan2.2-T2V-A14B", 27.0, True, "MoE：总 27B / 每步激活 14B　⛔ 总权重超了"),
    ("Qwen3.5-397B", 397.0, False, "对照：LLM"),
    ("DeepSeek-V3", 671.0, False, "对照：LLM"),
)


def main():
    f = Fig(W, "把扩散模型和大语言模型的 bf16 权重体积摆在同一根对数轴上，"
               "并画出一颗 v6e 的 32 GB 这条线：扩散这一族从 7 GB 到 54 GB，"
               "在一颗到两颗的量级；而 LLM 是 794 GB 和 1342 GB，要几十颗")
    f.marks = set()
    y = f.header(
        '一颗装得下吗 ——&#160;'
        '<tspan font-weight="700">这就是为什么 v6e 的 Pod 只有 256 颗</tspan>',
        '⭐ X-2 里那三条看着像减配的规格（4 个 ICI 口、2D 环面、Pod 256），'
        '把模型体积摆出来就合理了：<tspan font-weight="700">'
        '这一族模型在「一颗到几颗」的量级上，不需要摊到几千颗</tspan>。',
        [(BL, "扩散模型"), (GY, "对照：大语言模型"), (RD, "一颗 v6e ＝ 32 GB")])

    AX0, AX1 = 330, W - 290
    LO, HI = 4.0, 2000.0

    def xf(v):
        return AX0 + (AX1 - AX0) * (math.log10(v) - math.log10(LO)) / \
            (math.log10(HI) - math.log10(LO))

    ROW = 50
    top = y + 34

    # ⛔ 门槛线先画（背景层）——&nbsp;X-1 那次的教训。
    f.line(xf(CHIP_GB), top - 24, xf(CHIP_GB), top + len(MODELS) * ROW - 6,
           RD, 1.8, dash="5 4", arrow=False)
    f.t(xf(CHIP_GB), top - 30, "一颗 v6e　32 GB", RD, bold=True,
        size=_sz(12), anchor="middle")
    # 8 颗（一台主机）也标一下 —— 它是 v6e-8 那个单机全量配置
    f.line(xf(CHIP_GB * 8), top - 24, xf(CHIP_GB * 8),
           top + len(MODELS) * ROW - 6, GY2, 1.2, dash="3 4", arrow=False)
    f.t(xf(CHIP_GB * 8), top - 30, "一台主机 8 颗　256 GB", GY2,
        size=_sz(11), anchor="middle")

    for i, (name, pb, is_diff, note) in enumerate(MODELS):
        yy = top + i * ROW
        gb = pb * 2.0                      # bf16：每参数 2 字节
        col = BL if is_diff else GY
        f.t(0, yy + 4, name, INK if is_diff else GY, bold=True, size=_sz(12))
        f.t(0, yy + 24, note, GY2, size=_sz(11), w=AX0 - 20)
        f.line(AX0, yy + 8, AX1, yy + 8, LINE2, 1, arrow=False)
        f.line(AX0, yy + 8, xf(gb), yy + 8, col, 3.2, arrow=False)
        f.box(xf(gb) - 7, yy + 1, 14, 14, col, col, 7)
        # 右侧：体积 ＋ 要几颗
        need = int(math.ceil(gb / CHIP_GB))
        f.t(xf(gb) + 16, yy + 13,
            "%s GB" % (("%.1f" % gb).rstrip("0").rstrip(".")),
            col, bold=True, size=_sz(12))
        # ⛔ 光看「装不装得下权重」会给出误导性的绿灯：Wan2.1 的 28 GB
        #   确实 < 32，但激活只剩 4 GB 余量 ——&nbsp;判成「1 颗 ✅」等于替读者
        #   下了一个我们没验证过的结论。⭐ 把「余量不足两成」单列成一档黄灯。
        head = CHIP_GB * need - gb                    # 这么多颗之后剩下的余量
        if need == 1 and head < CHIP_GB * 0.2:
            lab, lc = "1 颗　⚠️ 余量只剩 %d GB" % round(head), "#b06000"
        elif need == 1:
            lab, lc = "1 颗", "#0d652d"
        else:
            lab, lc = "≥ %d 颗" % need, "#a50e0e"
        f.t(AX1 + 24, yy + 13, lab, lc, bold=True, size=_sz(12))

    ay = top + len(MODELS) * ROW + 2
    f.line(AX0, ay, AX1, ay, GY2, 1.2, arrow=False)
    for v in (10, 100, 1000):
        f.line(xf(v), ay, xf(v), ay + 5, GY2, 1, arrow=False)
        f.t(xf(v), ay + 20, "%s GB" % format(v, ","), GY2,
            size=_sz(11), anchor="middle")
    f.t((AX0 + AX1) / 2.0, ay + 40,
        "bf16 权重体积（对数轴）——&#160;⚠️ 只算权重，不含激活与中间结果",
        GY, size=_sz(11), anchor="middle")

    y = ay + 62

    y = f.band(y, "warn",
               "⛔ 别把这张图讲成「扩散都装得进一颗」——&#160;它不是",
               ['<tspan font-weight="700">Wan2.1-14B 的 28 GB 已经贴着 32 GB 的边</tspan>，'
                '激活只剩 4 GB 余量；'
                '<tspan font-weight="700">Wan2.2 那个总 27B 的 MoE 直接超线</tspan>，'
                '得两颗起。',
                '⭐ 图上如实画出这两条踩线和超线的 ——&#160;'
                '<tspan font-weight="700">一张结论过于整齐的图，台下第一时间就会怀疑它。</tspan>',
                '⚠️ 而且这根轴<tspan font-weight="700">只算权重</tspan>：真跑起来还有激活、'
                '中间 latent、编译缓存。'
                '<tspan font-weight="700">「28 &lt; 32 所以能跑」不是结论，只是必要条件。</tspan>'])

    y = f.band(y + 14, "ok",
               "但量级是清楚的：这一族在「一颗到几颗」，LLM 在「几十颗」",
               ['扩散这一族从 <tspan font-weight="700">7 GB 到 54 GB</tspan>；'
                '同一根轴上，Qwen3.5-397B 是 <tspan font-weight="700">794 GB</tspan>、'
                'DeepSeek-V3 是 <tspan font-weight="700">1,342 GB</tspan> ——&#160;'
                '<tspan font-weight="700">差了一个半到两个数量级</tspan>。',
                '⭐⭐ 于是 X-2 里那三条「减配」有了解释：'
                '<tspan font-weight="700">4 个 ICI 口、2D 环面、Pod 只有 256 颗</tspan>，'
                '是因为 v6e 不打「一个模型摊在几千颗上」那场仗 ——&#160;'
                '<tspan font-weight="700">不打，就不用付那个成本。</tspan>',
                '⭐ 反过来看也一样：模型一颗装得下，多卡就只是'
                '<tspan font-weight="700">各生成各的</tspan>，'
                '卡与卡之间几乎不用说话 ——&#160;这类扩展对互联的要求本来就低。'])

    y = f.src(y + 18,
              'SDXL 3.5B ——&#160;本仓库 SDXL/README；HunyuanVideo-1.5 8.3B ——&#160;'
              '腾讯官方 GitHub；FLUX.1 [dev] 12B ——&#160;Black Forest Labs 官方模型卡；'
              'Wan2.1-T2V-14B 与 Wan2.2-T2V-A14B（总 27B / 激活 14B）——&#160;Wan-AI 官方模型卡',
              '一颗 v6e HBM 32 GB、每 host 8 颗 ——&#160;官方 v6e 规格表。'
              'bf16 按每参数 2 字节换算。⛔ 只算权重，不含激活、中间 latent、编译缓存。')
    f.save("figx-6.svg", y + 6)


main()
