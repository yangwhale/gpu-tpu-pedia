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

# (名字, 参数量 B, 是不是扩散, 备注, ⭐ 我们实测跑在什么配置上, 颜色档)
# ⛔⛔ 2026-09-09 现场纠正：右边那一列原来是「1 颗 / ≥2 颗」——&nbsp;
#   **拿权重体积除以 32 GB 推出来的，不是实测。**
#   ⭐ 换成各模型 README「测试环境」段里真实记录的配置之后，
#     立刻看出两处「按体积猜会猜错」的地方（见图下第一条落点带）——&nbsp;
#     **「装得下」和「该用几颗」是两个不同的问题。**
MODELS = (
    ("S3Diff（SD-Turbo）", 3.3, True, "单步超分 4×",
     "单颗　⭐ 8 卡实测反而更慢", "ok"),
    ("SDXL", 3.5, True, "文生图",
     "v6e-1　（也测过 v6e-4 / v6e-8 数据并行）", "ok"),
    ("HunyuanVideo-1.5", 8.3, True, "文生视频",
     "v6e-8", "mid"),
    ("FLUX.1 [dev]", 12.0, True, "文生图",
     "⚠️ 未记录实测配置", "na"),
    ("Wan2.1-T2V-14B", 14.0, True, "文生视频　⚠️ 权重贴着 32 GB 的边",
     "v6e-8（dp=1, tp=8）", "mid"),
    ("Wan2.2-T2V-A14B", 27.0, True, "MoE：总 27B / 每步激活 14B",
     "v6e-16（dp=2, sp=1, tp=8）", "mid"),
    ("Qwen3.5-397B", 397.0, False, "对照：LLM",
     "—　我们没在 v6e 上跑过", "na"),
    ("DeepSeek-V3", 671.0, False, "对照：LLM",
     "—　我们没在 v6e 上跑过", "na"),
)
CFGCOL = {"ok": "#0d652d", "mid": "#174ea6", "na": GY2}


def main():
    f = Fig(W, "把扩散模型和大语言模型的 bf16 权重体积摆在同一根对数轴上，"
               "并画出一颗 v6e 的 32 GB 这条线：扩散这一族从 7 GB 到 54 GB，"
               "在一颗到两颗的量级；而 LLM 是 794 GB 和 1342 GB，要几十颗")
    f.marks = set()
    y = f.header(
        '装得下吗 ——&#160;<tspan font-weight="700">要算的是权重 ＋ 峰值激活，'
        '而扩散是激活说了算</tspan>',
        '⛔ 下面那根轴<tspan font-weight="700">只画权重</tspan>——&#160;'
        '而扩散模型真正吃显存的是<tspan font-weight="700">运行时的激活</tspan>，'
        '它比权重大、而且随分辨率和帧数涨。<tspan font-weight="700">'
        '图下半部分是我们自己撞 OOM 时的实测账。</tspan>',
        [(BL, "扩散模型"), (GY, "对照：大语言模型"), (RD, "一颗 v6e ＝ 32 GB"),
         (GR, "实测：单颗"), (BL, "实测：8 卡一台主机")])

    AX0, AX1 = 330, W - 470
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

    for i, (name, pb, is_diff, note, cfg, ck) in enumerate(MODELS):
        yy = top + i * ROW
        gb = pb * 2.0                      # bf16：每参数 2 字节
        col = BL if is_diff else GY
        f.t(0, yy + 4, name, INK if is_diff else GY, bold=True, size=_sz(12))
        f.t(0, yy + 24, note, GY2, size=_sz(11), w=AX0 - 20)
        f.line(AX0, yy + 8, AX1, yy + 8, LINE2, 1, arrow=False)
        f.line(AX0, yy + 8, xf(gb), yy + 8, col, 3.2, arrow=False)
        f.box(xf(gb) - 7, yy + 1, 14, 14, col, col, 7)
        # 右侧：体积 ＋ 要几颗
        f.t(xf(gb) + 16, yy + 13,
            "%s GB" % (("%.1f" % gb).rstrip("0").rstrip(".")),
            col, bold=True, size=_sz(12))
        # ⭐ 右列＝**我们真跑过的配置**，不是从体积推出来的颗数。
        #   ⛔ 原来这里写的是「1 颗 / ≥ N 颗」，那是 gb ÷ 32 算的 ——&nbsp;
        #     现场当场纠正：「我们不是全部都有实测吗，不要按模型大小去瞎猜。」
        f.t(AX1 + 24, yy + 13, cfg, CFGCOL[ck], bold=(ck != "na"),
            size=_sz(12), w=W - AX1 - 30)

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

    # ══════════════════ 同一颗芯片上的真实预算（全是实测）══════════════════
    # ⛔⛔ 2026-09-09 现场第二次纠正：「这个大小只是模型权重，你要考虑更深一层 ——
    #   运行过程中需要的激活。Diffusion model 权重占的不多，主要是激活占的多。」
    # ⭐ 他是对的，而且仓库里正好有一条**带 XLA 原始报错的 OOM 记录**可以当证据 ——
    #   比任何估算都硬。下面这一块全部取自那份案例研究。
    BUD = (
        ("芯片标称", "32 GB", "官方规格表上的 HBM 容量", GY2, GY),
        ("那次 OOM 时实际可用", "13.10 GB", "XLA 报错原文：There are 13.10G free ——&#160;"
         "其余被权重与运行时占着", "#b06000", "#b06000"),
        ("VAE 解码一步要多少", "19.00 GB", "XLA 报错原文：Attempting to reserve 19.00G "
         "——&#160;⛔ 19 &gt; 13.1，OOM", "#a50e0e", "#a50e0e"),
        ("对照：这个模型的权重", "10 GB", "CogVideoX-5B，bf16 ——&#160;"
         "⭐ <tspan font-weight=\"700\">激活是权重的近两倍</tspan>", BL, "#174ea6"),
        ("改实现之后", "&lt; 13 GB", "逐帧解码 ＋ 共享缓存 ——&#160;"
         "⭐ 解法不是换更大的卡，是改实现", GR, "#0d652d"),
    )
    hy = y
    TH = 34 + len(BUD) * 34
    f.box(0, hy, W, TH, "#fff", LINE, 8)
    f.colhead(14, hy + 22, "同一颗 v6e 上的真实预算（CogVideoX VAE，实测）")
    f.colhead(430, hy + 22, "多少")
    f.colhead(560, hy + 22, "出处 / 说明")
    f.line(0, hy + 34, W, hy + 34, LINE, 1, arrow=False)
    for i, (k, v, why, bar, tc) in enumerate(BUD):
        yy = hy + 34 + i * 34
        if i:
            f.line(0, yy, W, yy, LINE2, 1, arrow=False)
        f.box(0, yy + 6, 4, 22, bar, bar, 2)
        f.t(14, yy + 22, k, INK, bold=True, size=_sz(12), w=410)
        f.t(430, yy + 22, v, tc, bold=True, size=_sz(13))
        f.t(560, yy + 22, why, GY, size=_sz(11.5), w=W - 574)
    y = hy + TH + 22

    y = f.band(y, "bad",
               "⛔ 换成实测之后，立刻看出两处「按体积猜」会猜错的地方",
               ['① <tspan font-weight="700">S3Diff 只有 6.6 GB，按体积猜「一颗绰绰有余」——&#160;对，'
                '但那不是重点。</tspan>真正的发现是：'
                '<tspan font-weight="700">我们把它摊到 8 卡做张量并行，实测反而更慢</tspan>'
                '（5.46 秒 对 5.28 秒），而预热长了 15 倍。模型太小，通信开销盖过了收益。',
                '② <tspan font-weight="700">Wan2.1 的 28 GB 按体积猜「贴边能塞进一颗」</tspan>——&#160;'
                '而实测从来没人这么跑：它是在 <tspan font-weight="700">v6e-8 上 dp=1、tp=8 摊开</tspan>跑的。',
                '⭐⭐ <tspan font-weight="700">「装得下」和「该用几颗」是两个不同的问题</tspan>——&#160;'
                '前者看体积就能答，后者只能实测。'])

    y = f.band(y + 14, "bad",
               "⛔⛔ 所以「28 GB &lt; 32 GB 所以能跑」这句话是错的 ——&#160;两处都错",
               ['① <tspan font-weight="700">分母错了。</tspan>32 GB 是标称，不是预算 ——&#160;'
                '真跑起来权重和运行时先占掉一大块，'
                '那次 OOM 时 XLA 报的是<tspan font-weight="700">只剩 13.10 GB</tspan>。',
                '② <tspan font-weight="700">分子也错了。</tspan>要放进去的不只是权重，'
                '还有<tspan font-weight="700">峰值激活</tspan> ——&#160;'
                '而扩散这一族<tspan font-weight="700">激活比权重大</tspan>：'
                'CogVideoX-5B 权重 10 GB，它的 VAE 解码一步却要 19 GB。',
                '⭐⭐ 而且<tspan font-weight="700">激活随分辨率与帧数涨，权重一个字节不涨</tspan>——&#160;'
                'Wan2.1 的 480P 跑得动、720P OOM，用的是<tspan font-weight="700">同一份权重</tspan>。'
                '<tspan font-weight="700">权重决定装不装得进，激活决定跑不跑得动。</tspan>'])

    y = f.band(y + 14, "ok",
               "实测分布：小的单颗，主力清一色 8 卡 ——&#160;没有一个需要跨主机",
               ['<tspan font-weight="700">单颗</tspan>：S3Diff · SDXL（延迟最优）· Real-ESRGAN　'
                '<tspan font-weight="700">8 卡</tspan>：HunyuanVideo-1.5 / Wan2.1 / CogVideoX '
                '在 v6e-8 · Flux.2 在 v4-8 · Wan2.2 I2V 在 v6e-16',
                '⭐ 这正好解释 X-2 里那三条看着像减配的规格（4 个 ICI 口、二维环面、Pod 只有 256）：'
                '<tspan font-weight="700">v6e 不打「一个模型摊在几千颗上」那场仗，不打就不用付那个成本。</tspan>'])

    y = f.src(y + 18,
              '⭐ 右列「实测配置」全部取自各模型 README 的测试环境段：SDXL v6e-1/4/8 · '
              'HunyuanVideo-1.5 / Wan2.1 / CogVideoX 在 v6e-8 · Flux.2 在 v4-8 · '
              'Wan2.2 I2V 在 v6e-16（分片配置见该模型优化指南第三章）· '
              'S3Diff 与 Real-ESRGAN 单颗',
              '「8 卡反而更慢」出自 S3Diff README 的 Why Not Multi-Chip 段；'
              '权重体积按 bf16 每参数 2 字节换算，参数量出自各家官方模型卡',
              '⛔ 那根轴<tspan font-weight="700">只算权重</tspan>，不含激活与编译缓存 ——&#160;'
              '它能回答「装不装得下」，<tspan font-weight="700">回答不了「该用几颗」</tspan>。'
              '⚠️ FLUX.1 我们没有记录实测配置，图上如实留空。')
    f.save("figx-6.svg", y + 6)


main()
