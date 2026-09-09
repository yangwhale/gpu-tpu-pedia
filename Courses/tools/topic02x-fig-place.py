# -*- coding: utf-8 -*-
r"""外传 图 X-11 · **谁吃算力、谁吃显存，于是该摆在哪** ——&nbsp;X-10 的下半场。

════════════════════════════════════════════════════════════════════
⭐ X-10 说了「能切」，这一张说「切完怎么摆」
════════════════════════════════════════════════════════════════════
三段的资源画像**完全不一样**，这才是分开部署真正的理由 ——&nbsp;
不是为了好看，是因为**把三种不同的胃口塞进同一台机器，一定有人吃不饱、有人撑着**。

  · Stage 1 文本编码：几乎不吃算力，就是把权重读一遍
  · Stage 2 DiT 去噪：**吃算力**，强度 75,600 是 v6e 门槛 560 的 135 倍
  · Stage 3 VAE 解码：**吃显存峰值**，把 4.8M 个数展开成 224M 个 ——&nbsp;
    当年 maxdiffusion 那版就是在这一步 OOM 的

════════════════════════════════════════════════════════════════════
⛔ 一个必须写进图里的反例 ——&nbsp;否则这张图会骗人
════════════════════════════════════════════════════════════════════
「text encoder 放 CPU」这条**不是普适结论**：

  · Wan2.1 用 T5，那一段只要 **3 秒**  → 放哪都行
  · Flux.2 用 Mistral3，放 CPU 要 **30 秒** ——&nbsp;
    而它的 DiT 在 TPU 上跑完 50 步只要 **13.5 秒**

⭐⭐ 也就是说 **Flux.2 的「轻量」那一段，反而是全程最慢的一段。**
   ⛔ 所以判据不是「text encoder 天生该放 CPU」，
     而是**先量一量这一段在 CPU 上要多久**，再决定。
   ⚠️ 一张只给顺口结论、不给反例的图，第一次遇到 Flux.2 就会失效。

════════════════════════════════════════════════════════════════════
📌 数从哪来
════════════════════════════════════════════════════════════════════
三段耗时（v6e-8，Wan2.1 720P 81 帧 50 步）——&nbsp;本仓库 Wan2.1/README 实测表：
  Stage1 ~3 s ／ Stage2 预热 110 s、正式 230 s ／ Stage3 预热 80 s、正式 1 s
Flux.2 三段（v4-8，1024×1024 50 步）——&nbsp;本仓库 Flux.2/README：
  Stage1 CPU ~30 s ／ Stage2 13.5 s ／ Stage3 1.5 s
分片配置（8 设备 dp=2 / tp=4，16 设备 dp=2 / tp=8；40 头 ÷ 8 = 5 头每设备）
  ——&nbsp;本仓库 Wan2.2/docs/wan_tpu_optimization_guide.md 第三章
SDXL v6e-8 数据并行 2.40 img/s ——&nbsp;本仓库 SDXL/README
"""
from topic03_draw import (Fig, wpx, _sz, LINE, LINE2,
                          BL, OR, GR, RD, GY, GY2, PU, CY, INK)

W = 1400

# (问的是同一件事, Stage1, Stage2, Stage3, 旁注)
ROWS = (
    ("它在干什么",
     "把 prompt 编成向量", "五十步去噪，全部算力在这", "把 latent 展开成像素",
     ""),
    ("吃的是哪种资源",
     "几乎不吃／读一遍权重", "★ 吃算力", "★ 吃显存峰值",
     "⭐ 三种胃口完全不同 ——&#160;这才是该分开的真正理由"),
    ("正式运行占多少时间",
     "3 秒 ／ 1.3%", "230 秒 ／ 98.3%", "1 秒 ／ 0.4%",
     "Wan2.1 720P，v6e-8 实测"),
    ("编译（预热）要多久",
     "—", "110 秒", "80 秒 → 之后每次 1 秒",
     "⭐ VAE 是 80 倍差 ——&#160;这个形状天生该做成常驻服务"),
    ("峰值内存卡在哪",
     "权重本身", "权重 28 GB ＋ 激活 774 MB", "展开成 448 MB 像素",
     "⛔ 当年 maxdiffusion 那版就是在第三段 OOM"),
    ("于是摆在哪",
     "CPU 或 TPU ——&#160;先量再定", "TPU 多芯片，TP／CP 分片", "TPU 单芯片，常驻",
     "⚠️ 第一格别写死，见下方 Flux.2 反例"),
)
CC = (GY, "#174ea6", "#0d652d")          # 三段各自的字色
CB = (GY2, BL, GR)


def main():
    f = Fig(W, "三段的资源画像对比：文本编码几乎不吃资源，DiT 去噪吃算力占九成八时间，"
               "VAE 解码吃显存峰值但只占零点四；下面画两种部署拓扑，"
               "一体化全塞一台机器，分开部署则 CPU 跑编码、八颗 TPU 做张量并行跑 DiT、"
               "一颗 TPU 常驻做 VAE 解码，跨机只传十九兆的 latent")
    f.marks = set()
    y = f.header(
        '谁吃算力、谁吃显存 ——&#160;'
        '<tspan font-weight="700">于是三段该摆在不同的地方</tspan>',
        '⭐ 上一张说明<tspan font-weight="700">切得动</tspan>（腰只有 19.4 MB）；'
        '这一张说明<tspan font-weight="700">为什么值得切</tspan> ——&#160;'
        '三段的胃口根本不是一回事，塞进同一台机器必然有人吃不饱、有人撑着。',
        [(GY2, "① 文本编码"), (BL, "② DiT 去噪（吃算力）"), (GR, "③ VAE 解码（吃显存）")])

    # ══════════════════ §A 资源画像 ══════════════════
    C0, CW = 178, 268
    NX = C0 + CW * 3 + 16
    hy = y + 6
    TH = 34 + len(ROWS) * 48
    f.box(0, hy, W, TH, "#fff", LINE, 8)
    f.colhead(14, hy + 22, "问的是同一件事")
    for k, nm in enumerate(("① 文本编码", "② DiT 去噪", "③ VAE 解码")):
        f.t(C0 + CW * k, hy + 22, nm, CC[k], bold=True, size=_sz(12.5))
    f.colhead(NX, hy + 22, "旁注")
    f.line(0, hy + 34, W, hy + 34, LINE, 1, arrow=False)
    for i, (k, a, b, c, note) in enumerate(ROWS):
        yy = hy + 34 + i * 48
        if i:
            f.line(0, yy, W, yy, LINE2, 1, arrow=False)
        f.t(14, yy + 22, k, INK, bold=True, size=_sz(12), w=C0 - 22)
        for j, v in enumerate((a, b, c)):
            star = v.startswith("★ ")
            f.box(C0 + CW * j - 12, yy + 9, 3, 30, CB[j], CB[j], 2)
            f.t(C0 + CW * j, yy + 22, v.replace("★ ", ""),
                CC[j], bold=star, size=_sz(12), w=CW - 20)
        if note:
            f.t(NX, yy + 22, note, GY2, size=_sz(11), w=W - NX - 14)
    y = hy + TH + 24

    # ══════════════════ §B 两种拓扑 ══════════════════
    PW = (W - 30) / 2.0
    # ⛔ 高度算出来，别猜 ——&nbsp;这条已经栽过四次，最后一次是**结论行压住了正文**。
    # ⭐ 关键在于**有两个坐标系**：f.panel(x, y, w, PH) 里的 PH 从 y 量起，
    #   而它返回的 ly 已经跳过了 30px 标题栏。心算时最容易把这 30 漏掉。
    #   以 ly 为原点写清楚：
    CARD0, CARD_GAP, CARD_H = 26, 88, 62      # 第一张卡的 y、卡间距、卡高
    CARD_END = CARD0 + 2 * CARD_GAP + CARD_H  # = 264，第三张卡的底
    CONC_Y = CARD_END + 24                    # = 288，结论行基线
    PH = 30 + CONC_Y + 14                     # 标题栏 ＋ 内容 ＋ 下沿

    def machine(x, yy, w, title, sub, body, col, fill):
        f.box(x, yy, w, 62, fill, col, 6, 1.6)
        f.t(x + 14, yy + 22, title, col, bold=True, size=_sz(12.5))
        f.t(x + 14, yy + 40, sub, GY, size=_sz(11), w=w - 28)
        f.t(x + 14, yy + 55, body, GY2, size=_sz(11), w=w - 28)

    def wire(x, yy, w, txt):
        f.line(x + w / 2.0, yy, x + w / 2.0, yy + 20, GR, 2)
        f.t(x + w / 2.0 + 12, yy + 15, txt, "#0d652d", bold=True, size=_sz(11.5))

    # ── 左：一体化 ────────────────────────────────────────────
    ly = f.panel(0, y, PW, PH, "方式一：端到端一体化", GY2,
                 sub="generate_torchax.py ·  一台 v6e-8，一个进程",
                 tag="验证 / 演示 / benchmark")
    mw = PW - 32
    f.box(16, ly + 14, mw, CARD_END - 2, "#fafbfc", LINE2, 8, 1, dash="4 4")
    for k, (t1, t2, t3) in enumerate((
            ("① 文本编码", "T5 权重驻留", "算完就闲着，但内存一直占着"),
            ("② DiT 去噪", "28 GB 权重 ＋ 激活", "★ 全程只有这一段真的在忙"),
            ("③ VAE 解码", "峰值展开 448 MB", "跑 1 秒，编译 80 秒每次重付"))):
        yy = ly + CARD0 + k * CARD_GAP
        machine(32, yy, mw - 32, t1, t2, t3.replace("★ ", ""),
                CB[k], "#fff")
        if k < 2:
            f.line(32 + (mw - 32) / 2.0, yy + 62, 32 + (mw - 32) / 2.0,
                   yy + CARD_GAP - 4, GY2, 1.6)
    f.t(16, ly + CONC_Y,
        "⛔ <tspan font-weight=\"700\">三种胃口共用一份资源</tspan>"
        "——&#160;按最馋的那一段配机器，另外两段的钱就白付了",
        "#a50e0e", size=_sz(11.5), w=mw)

    # ── 右：分开部署 ──────────────────────────────────────────
    RX = PW + 30
    ry = f.panel(RX, y, PW, PH, "方式二：三段分开部署", GR,
                 sub="stage1 / stage2 / stage3", tag="生产 / 多机 / 异构")
    mw2 = PW - 32
    for k, (t1, t2, t3) in enumerate((
            ("① CPU 节点（或小 TPU）", "算一次，换 50 个 seed 反复用",
             "⚠️ 前提是它在 CPU 上够快 ——&#160;见下方反例"),
            ("② TPU DiT 节点 · v6e-8", "8 颗做 TP：40 头 ÷ 8 ＝ 5 头每颗",
             "★ 算力全压这儿，扩容也只扩这一层"),
            ("③ TPU VAE 服务 · 1 颗常驻", "编译好放着，谁要解码谁来调",
             "80 秒编译摊到成千上万次调用 ≈ 0"))):
        yy = ry + CARD0 + k * CARD_GAP
        machine(RX + 32, yy, mw2 - 32, t1, t2, t3.replace("★ ", ""),
                CB[k], "#fff")
        if k < 2:
            wire(RX + 32, yy + 62, mw2 - 32,
                 "7.4 MB embedding ⏷" if k == 0 else "19.4 MB latent ⏷")
    f.t(RX + 16, ry + CONC_Y,
        "⭐ <tspan font-weight=\"700\">三段各配各的机器</tspan>"
        "——&#160;而把它们串起来的成本，是那两根绿线上的几十兆",
        "#0d652d", size=_sz(11.5), w=mw2)

    y = y + PH + 22

    # ══════════════════ §C 反例 ══════════════════
    y = f.band(y, "bad",
               "⛔ 「text encoder 放 CPU」不是普适结论 ——&#160;Flux.2 就是反例",
               ['Wan2.1 用 T5，那一段 <tspan font-weight="700">3 秒</tspan>，放哪都行。'
                '但 Flux.2 用 Mistral3，<tspan font-weight="700">放 CPU 要 30 秒</tspan> ——&#160;'
                '而它的 DiT 在 TPU 上跑完 50 步只要 <tspan font-weight="700">13.5 秒</tspan>。',
                '⭐⭐ 也就是说 <tspan font-weight="700">Flux.2 那个「轻量」的第一段，'
                '反而是全程最慢的一段</tspan>。照着「编码器放 CPU」的顺口结论摆，'
                '等于把整条链的瓶颈从 TPU 挪到了 CPU 上。',
                '⭐ 判据不是「哪一段天生该放哪」，而是'
                '<tspan font-weight="700">先量一量它在目标硬件上要多久</tspan>。'
                '分段的价值恰恰在这里 ——&#160;'
                '<tspan font-weight="700">拆开之后，每一段的账才第一次能单独算清楚。</tspan>'])

    # ══════════════════ §D 一个节点摆几路 ══════════════════
    y = f.band(y + 14, "info",
               "「一个节点能放几路」——&#160;先看权重塞不塞得下，再看它够不够忙",
               ['一台 v6e-8 是 <tspan font-weight="700">8 颗 × 32 GB ＝ 256 GB</tspan>。'
                'Wan2.1 的 28 GB 权重单颗只剩 4 GB 余量，'
                '所以走 <tspan font-weight="700">TP 摊到 8 颗</tspan>，每颗 3.5 GB，宽裕得多。',
                '⭐ 反过来，SDXL 只有 7 GB，<tspan font-weight="700">一颗就装得下</tspan> ——&#160;'
                '于是 v6e-8 上的正确用法不是切一个模型，而是'
                '<tspan font-weight="700">开 8 路各生成各的</tspan>（我们实测 2.40 img/s）。',
                '⭐⭐ 两条合起来是同一句话：'
                '<tspan font-weight="700">装不下就把一个模型摊开，装得下就多放几路。</tspan>'
                '而分段之后这个决定是<tspan font-weight="700">按段做的</tspan> ——&#160;'
                'DiT 那一层摊开，VAE 那一层多放几路，互不牵扯。'])

    y = f.src(y + 18,
              '三段耗时与预热 ——&#160;本仓库 Wan2.1/README（v6e-8）与 Flux.2/README（v4-8）实测表；'
              '分片配置 dp=2/tp=4、40 头 ÷ 8 设备 ——&#160;'
              '本仓库 Wan2.2/docs/wan_tpu_optimization_guide.md 第三章',
              'SDXL 一颗 7 GB、v6e-8 八路数据并行 2.40 img/s ——&#160;本仓库 SDXL/README；'
              'v6e 单颗 HBM 32 GB ——&#160;官方 v6e 规格表。'
              '⚠️ 三处实测硬件不同（v6e-8 / v4-8），本图只取各自的段间比例，不做横向快慢比较。')
    f.save("figx-11.svg", y + 6)


main()
