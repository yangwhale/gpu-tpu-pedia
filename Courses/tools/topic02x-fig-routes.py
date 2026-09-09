# -*- coding: utf-8 -*-
r"""外传 图 X-13 · **三条移植路线，我们全走完了** ——&nbsp;以及为什么第二代的代码还留着。

════════════════════════════════════════════════════════════════════
⭐ 这张图回答的是选型问题，而选型问题最容易被讲成立场
════════════════════════════════════════════════════════════════════
「为什么不用纯 JAX 重写？」这个问题，**讲道理是讲不赢的** ——&nbsp;
两边都能说出一串听起来对的理由。

⭐⭐ 我们的答法是：**三条都走完，把对照版原样留在仓库里。**
   Wan 2.1 那个目录下现在同时躺着 torchax、Flax NNX、纯 JAX 三版 VAE 解码器，
   外加一份九百多行的三方逐项对比。
   ——&nbsp;**客户要判断，不需要听我们说，读那三份代码就行。**

════════════════════════════════════════════════════════════════════
📌 torchax 快在哪：它不是「又一个 tracing 框架」
════════════════════════════════════════════════════════════════════
Flax 与纯 JAX 都要**把整个函数 trace 一遍**再交给 XLA。
torchax 走的是 **PyTorch 自己的 C++ dispatcher**：
调用 `torch.nn.functional.conv3d` 时，dispatcher 认出算子类型，
路由到 torchax 注册的后端，**直接落到对应的 JAX 实现**。

  · 追踪开销：**无**（走 dispatcher）　vs　两边都要完整 tracing
  · 编译粒度：**增量**　　　　　　　　vs　两边都是完整函数
  · 可变状态：**直接支持**　　　　　　vs　Flax 要 `pytree=False`、纯 JAX 要显式传 cache
  · 生态兼容：**完整 PyTorch**　　　　vs　两边都要移植

⭐ 最后一条才是决定性的 ——&nbsp;前三条是快慢，**第四条是「新模型接得进来吗」**。

════════════════════════════════════════════════════════════════════
⛔ 别把这张图画成「torchax 全面胜出」
════════════════════════════════════════════════════════════════════
第二代不是走错了路：**没有它，我们不会知道该往哪个方向优化**，
而且那批 Flax / 纯 JAX 实现今天仍在当**数值对照**用。
⭐ 图上要如实写出「为什么留下」，而不只是「为什么放弃」。
"""
import re

from topic03_draw import (Fig, wpx, _sz, LINE, LINE2,
                          BL, OR, GR, RD, GY, GY2, PU, CY, INK)

W = 1400

GENS = (
    ("第一代", "官方 maxdiffusion", RD, "#a50e0e", "#fce8e6"),
    ("第二代", "手写 Flax NNX / 纯 JAX", OR, "#b06000", "#feefc3"),
    ("第三代", "torchax ＋ diffusers-tpu", GR, "#0d652d", "#e6f4ea"),
)

ROWS = (
    ("具体怎么做的",
     "直接用官方那套 JAX 原生的扩散库",
     "把 DiT 和 VAE 用 Flax NNX 或纯 JAX 重写一遍",
     "PyTorch 代码不改，靠 torchax 把 ATen 算子映射到 JAX"),
    ("当时为什么走这条",
     "现成、官方维护、TPU 原生 —— 理论上最省事",
     "能跑、能精细控制分片与内存、能做数值对齐验证",
     "★ 走 PyTorch 自己的 dispatcher，不需要重新 tracing"),
    ("撞到了什么",
     "★ Wan 2.1 的 720P 在 VAE 这一步直接 OOM，跑不起来；模型覆盖面也窄",
     "★ 每来一个新模型就得重写一遍 —— 权重、scheduler、pipeline 全要跟着移植",
     "—（没有换掉它的理由出现）"),
    ("今天仓库里还剩什么",
     "只作为参考实现被引用 —— Splash Attention 的几个技巧是从它那儿学的",
     "★ 刻意保留的对照版：Wan 2.1 的 Flax 版与纯 JAX 版 VAE、CogVideoX 的 dit_flax",
     "★ 十个模型的主线实现，全部在这条路上"),
)

# torchax 为什么留下：四项逐项对比（出自本仓库那份九百多行的分析）
CMP = (
    ("追踪开销", "需要完整 tracing", "需要完整 tracing", "无 —— 走 dispatcher"),
    ("编译粒度", "完整函数", "完整函数", "增量"),
    ("可变状态", "要 pytree=False", "要显式传 cache", "直接支持"),
    ("生态兼容", "需要移植", "需要移植", "★ 完整 PyTorch"),
)


def wrap(sfull, w, size=11.5):
    """⛔ Fig.t() 只画单行，超宽会被护栏拦下。这张图的格子是两行的正文，
    所以自己按像素宽折行 ——&nbsp;⭐ 折行判据用 wpx()，跟护栏同一把尺，
    不用「多少个字」这种在中英混排下必然失准的估法。"""
    # ⛔ 逐字符折会把英文词劈开 ——&nbsp;实测断出过「CogVideo / X 的 dit_flax」。
    #   ⭐ 先按「连续 ASCII 串 or 单个汉字」切成不可分的单元，再拼行。
    units = re.findall(r"[A-Za-z0-9_.+\-]+|\s+|.", sfull)
    out, cur = [], ""
    for u in units:
        if wpx(cur + u, size) > w and cur:
            out.append(cur.rstrip())
            cur = u.lstrip()
        else:
            cur += u
    if cur.strip():
        out.append(cur.rstrip())
    return out


def main():
    f = Fig(W, "三代移植方法的逐项对比：第一代用官方 maxdiffusion 撞上 OOM，"
               "第二代手写 Flax 与纯 JAX 能跑但每个新模型都要重写，"
               "第三代 torchax 走 PyTorch 自己的 dispatcher 不需要重新 tracing，"
               "成为今天十个模型的主线；第二代的代码刻意保留作数值对照")
    f.marks = set()
    y = f.header(
        '三条移植路线，<tspan font-weight="700">我们全走完了 ——&#160;'
        '而且第二代的代码还留在仓库里</tspan>',
        '⭐ 「为什么不用纯 JAX 重写」这种问题<tspan font-weight="700">讲道理是讲不赢的</tspan>。'
        '我们的答法是三条都走一遍，把对照版原样留着 ——&#160;'
        '<tspan font-weight="700">要判断的人读代码，不必听我们说。</tspan>',
        [(RD, "第一代：撞墙"), (OR, "第二代：能跑，但贵"), (GR, "第三代：留下了")])

    # ══════════════════ 上：三代逐项 ══════════════════
    C0 = 176
    CW = (W - C0) / 3.0
    hy = y + 6
    TH = 52 + len(ROWS) * 58
    f.box(0, hy, W, TH, "#fff", LINE, 8)
    for k, (g, sub, col, dark, fill) in enumerate(GENS):
        f.box(C0 + CW * k + 4, hy + 8, CW - 8, 36, fill, col, 5, 1.2)
        f.t(C0 + CW * k + 16, hy + 24, g, dark, bold=True, size=_sz(12.5))
        f.t(C0 + CW * k + 16, hy + 39, sub, dark, size=_sz(11), w=CW - 32)
    f.colhead(14, hy + 30, "问的是同一件事")
    f.line(0, hy + 52, W, hy + 52, LINE, 1, arrow=False)
    for i, row in enumerate(ROWS):
        yy = hy + 52 + i * 58
        if i:
            f.line(0, yy, W, yy, LINE2, 1, arrow=False)
        f.t(14, yy + 24, row[0], INK, bold=True, size=_sz(12), w=C0 - 22)
        for k in range(3):
            v = row[1 + k]
            star = v.startswith("★ ")
            f.box(C0 + CW * k + 4, yy + 10, 3, 38, GENS[k][2], GENS[k][2], 2)
            f.lines(C0 + CW * k + 16, yy + 22, CW - 34,
                    wrap(v.replace("★ ", ""), CW - 36),
                    size=11.5, lh=17, fill=GENS[k][3] if star else GY,
                    bold_first=star)
    y = hy + TH + 22

    # ══════════════════ 下：torchax 到底赢在哪 ══════════════════
    hy = y
    TH2 = 34 + len(CMP) * 34
    f.box(0, hy, W, TH2, "#fff", LINE, 8)
    f.colhead(14, hy + 22, "四项逐一比")
    f.colhead(C0 + CW * 0 + 16, hy + 22, "纯 JAX")
    f.colhead(C0 + CW * 1 + 16, hy + 22, "Flax NNX")
    f.colhead(C0 + CW * 2 + 16, hy + 22, "torchax")
    f.line(0, hy + 34, W, hy + 34, LINE, 1, arrow=False)
    for i, (k, a1, a2, a3) in enumerate(CMP):
        yy = hy + 34 + i * 34
        if i:
            f.line(0, yy, W, yy, LINE2, 1, arrow=False)
        f.t(14, yy + 22, k, INK, bold=True, size=_sz(12))
        for j, v in enumerate((a1, a2, a3)):
            star = v.startswith("★ ") or j == 2
            f.t(C0 + CW * j + 16, yy + 22, v.replace("★ ", ""),
                "#0d652d" if j == 2 else GY, bold=star, size=_sz(11.5), w=CW - 34)
    y = hy + TH2 + 22

    y = f.band(y, "ok",
               "⭐⭐ 四项里最后一项才是决定性的",
               ['前三项（追踪开销、编译粒度、可变状态）说的都是<tspan font-weight="700">快慢</tspan>。'
                '而第四项<tspan font-weight="700">生态兼容</tspan>说的是另一回事：'
                '<tspan font-weight="700">下一个模型接不接得进来。</tspan>',
                '⭐ 第二代那条路的真正代价<tspan font-weight="700">不是慢，是「每个新模型都要重写一遍」</tspan> ——&#160;'
                '这项成本不随熟练度下降，它随模型数量线性累加。',
                '⭐⭐ 所以选型的判据不是「哪条路这一次更快」，而是'
                '<tspan font-weight="700">「哪条路让第十一个模型变便宜」</tspan>。'
                '——&#160;这跟上一张图里提交数从 73 掉到 2，说的是同一件事。'])

    y = f.band(y + 14, "warn",
               "⛔ 但别把这张图读成「第二代是弯路」",
               ['<tspan font-weight="700">没有第二代，我们不会知道该往哪个方向优化。</tspan>'
                '分片策略、内存布局、数值对齐这些认识，都是在手写那一版里长出来的 ——&#160;'
                '换成一开始就用 torchax，那些东西会被框架挡在视野之外。',
                '⭐ 而且那批代码<tspan font-weight="700">今天仍在服役</tspan>：'
                '当 torchax 版算出可疑结果时，'
                '<tspan font-weight="700">拿纯 JAX 版跑同一个输入对一遍</tspan>，'
                '是最快的定位手段。',
                '⭐ 判据：<tspan font-weight="700">一条被换掉的路线，'
                '值不值得留下取决于它还能不能当参照物</tspan> ——&#160;'
                '而不是取决于它现在跑得快不快。'])

    y = f.src(y + 18,
              '四项逐项对比与三版实测 ——&#160;本仓库 Wan2.1 的 '
              'torchax_vs_flax_vs_jax_analysis.md（约 960 行），'
              '同目录另有 aten_ops_catalog.md 列明 torchax 实际映射了哪些算子',
              'maxdiffusion 版 VAE 在 Wan2.1 720P 上 OOM ——&#160;本仓库 Wan2.1/README 实测表；'
              '⚠️ 三代的时间界限见图 X-12，其中「第三代从 12-10 起算」的依据是'
              '那天手写 Flax 的文件被删除，不是一个事后追认的说法。')
    f.save("figx-13.svg", y + 6)


main()
