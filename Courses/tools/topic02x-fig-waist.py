# -*- coding: utf-8 -*-
r"""外传 图 X-10 · **又胖又瘦** ——&nbsp;三阶段流水线上，数据体积一路是怎么变的。

════════════════════════════════════════════════════════════════════
⭐ 这张图回答的问题
════════════════════════════════════════════════════════════════════
「三段拆开跑，跨机到底要传多少东西？」

答案出人意料地小：**19.4 MB**。而同一条链路上，DiT 内部随手一个激活张量
就是 774 MB，注意力分数矩阵要是显式展开更是 457 GB。

⭐⭐ 这就是**「腰」**的意思 ——&nbsp;
   一条又胖又瘦的管道，最细的那一处恰好就是**该切开的那一处**。

════════════════════════════════════════════════════════════════════
📌 每个数是怎么来的（全部以 Wan2.1-T2V-14B / 1280×720 / 81 帧为准）
════════════════════════════════════════════════════════════════════
⭐ 其中两个是**实测文件大小**，不是算出来的 ——&nbsp;
  它们躺在本仓库 tpu/Wan2.1/generate_diffusers_torchax_staged/stage_outputs/ 里：

  · stage1_embeddings.safetensors  = 7,406,544 B  ≈ 7.4 MB
  · stage2_latents.safetensors     = 19,353,872 B ≈ 19.4 MB
  · output_video.mp4               =   776,380 B  ≈ 0.76 MB

⭐⭐ 那个 latent 文件还顺手**验证了形状**：
   16 × 21 × 90 × 160 × 4 B(fp32) = 19,353,600 B，
   加 safetensors 头 272 B **正好等于 19,353,872** ——&nbsp;
   一个字节不差。**这比任何推导都硬。**
   ⚠️ 顺带说明它存的是 fp32；换 bf16 只要 9.7 MB。

算出来的三个（推导链写在这里，图上也标了）：
  · DiT 单个激活：75,600 token × 5,120 维 × 2 B = 774 MB
  · 注意力分数若显式展开：75,600² × 40 头 × 2 B = **457 GB**
    ⛔ 所以它**从不落地** ——&nbsp;Splash Attention 分块算，算完即弃
  · VAE 展开后的像素张量：81 × 720 × 1280 × 3 × 2 B = 448 MB

压缩比 = 223,948,800 个像素数 ÷ 4,838,400 个 latent 数 = **46.3 倍**
  （空间 8×8 = 64 倍，时间 81→21 = 3.86 倍，通道 3→16 反向摊薄 5.33 倍）

════════════════════════════════════════════════════════════════════
⛔ 这张图不说什么
════════════════════════════════════════════════════════════════════
它画的是**数据体积**，不是**时间**。腰细不等于那一段快 ——&nbsp;
恰恰相反，产出这个 19.4 MB 的 Stage 2 占了全程九成以上的时间。
时间的那张图是 X-11。
"""
import math

from topic03_draw import (Fig, wpx, _sz, LINE, LINE2,
                          BL, OR, GR, RD, GY, GY2, PU, CY, INK)

W = 1400

# (标签, 副标, 字节数, 类别, 是不是落盘点)
#   类别 col：blue = 在算的中间态；green = 落盘的产物；red = 从不落地
PTS = (
    ("文本 prompt",      "约 230 个字符",              230,          "in",   False),
    ("① 文本 embedding", "实测文件 7,406,544 B",       7406544,      "disk", True),
    ("DiT 单个激活",     "75,600 × 5,120 × 2 B",       774144000,    "hot",  False),
    ("注意力分数矩阵",   "75,600² × 40 头 × 2 B",      457200000000, "never", False),
    ("② latent",         "实测文件 19,353,872 B",      19353872,     "disk", True),
    ("VAE 展开的像素",   "81×720×1280×3 × 2 B",        448000000,    "hot",  False),
    ("③ 成片 mp4",       "实测文件 776,380 B",         776380,       "disk", True),
)

COL = {"in": GY2, "disk": GR, "hot": BL, "never": RD}
FILL = {"in": "#f1f3f4", "disk": "#e6f4ea", "hot": "#e8f0fe", "never": "#fce8e6"}
DARK = {"in": GY, "disk": "#0d652d", "hot": "#174ea6", "never": "#a50e0e"}


def _mult(r):
    """倍数标签：大数取整，小数留一位 ——&nbsp;「× 590」比「× 590.4」好读。"""
    return "%.0f" % r if r >= 10 else "%.1f" % r


def human(b):
    """⛔ 十进制，不是 1024 进制 ——&nbsp;见文件头「单位口径」那一段。"""
    for lim, div, unit, fmt in ((1e3, 1, "B", "%.0f"),
                                (1e6, 1e3, "KB", "%.0f"),
                                (1e9, 1e6, "MB", "%.3g"),
                                (1e12, 1e9, "GB", "%.0f"),
                                (1e99, 1e12, "TB", "%.0f")):
        if b < lim:
            return (fmt + " %s") % (b / div, unit)


def main():
    f = Fig(W, "三阶段流水线上数据体积的变化：文本 embedding 7.4 MB，"
               "DiT 内部单个激活 774 MB，注意力分数若展开 457 GB，"
               "而两段之间真正落盘、真正跨机传的 latent 只有 19.4 MB，"
               "最后成片 0.76 MB。管道又胖又瘦，最细的腰正好是该切开的地方")
    f.marks = set()
    y = f.header(
        '又胖又瘦 ——&#160;'
        '<tspan font-weight="700">跨机只传 19.4 MB，而管子里最粗处是 457 GB</tspan>',
        '⭐ 纵轴是<tspan font-weight="700">对数刻度</tspan>的数据体积。'
        '三个绿色的点是真正落盘的产物（本仓库里就有这三个文件，大小逐字节可核）；'
        '蓝色是算的时候在 HBM 里的中间态；红色那根<tspan font-weight="700">从不落地</tspan>。',
        [(GR, "落盘 · 可跨机"), (BL, "HBM 里的中间态"), (RD, "从不落地（分块算）")])

    # ══════════════════ 上半：对数柱 ＋ 沙漏轮廓 ══════════════════
    AX0, AX1 = 96, W - 40
    TOP = y + 30                      # 柱顶
    BASE = TOP + 268                  # 柱底基线
    LO, HI = 1e2, 1e12                # 对数轴范围

    def hpx(b):
        return (BASE - TOP) * (math.log10(b) - math.log10(LO)) / \
            (math.log10(HI) - math.log10(LO))

    # ⛔ 背景层先画 ——&nbsp;X-1 那次把参考线画在最后，压掉了三个点。
    for g in (1e3, 1e6, 1e9, 1e12):
        gy = BASE - hpx(g)
        f.line(AX0 - 46, gy, AX1, gy, LINE2, 1, dash="3 5", arrow=False)
        f.t(AX0 - 52, gy + 4, human(g).replace(".0", ""), GY2,
            size=_sz(11), anchor="end")
    f.line(AX0 - 46, BASE, AX1, BASE, LINE, 1.2, arrow=False)

    n = len(PTS)
    SLOT = (AX1 - AX0) / float(n)
    BW = 74.0

    def cx(i):
        return AX0 + SLOT * i + SLOT / 2.0

    # ── 沙漏轮廓：把七个柱顶连成一条带 ─────────────────────────
    #   ⭐ 这条带才是这张图真正要讲的东西 ——&nbsp;柱子只是刻度，
    #     「一会儿粗一会儿细」这个形状要一眼能看出来。
    mid = BASE - 130
    up, dn = [], []
    for i, (_, _, b, _, _) in enumerate(PTS):
        half = hpx(b) / 2.0
        up.append((cx(i), mid - half))
        dn.append((cx(i), mid + half))
    d = "M %.1f %.1f " % up[0]
    d += " ".join("L %.1f %.1f" % p for p in up[1:])
    d += " " + " ".join("L %.1f %.1f" % p for p in reversed(dn))
    d += " Z"
    f.poly(d, fill="#f1f3f4", stroke=LINE, sw=1)

    # ── 七根柱 ────────────────────────────────────────────────
    for i, (lab, sub, b, kind, disk) in enumerate(PTS):
        x = cx(i) - BW / 2.0
        h = hpx(b)
        f.box(x, BASE - h, BW, h, FILL[kind], COL[kind], 4,
              1.6 if kind != "never" else 1.2,
              dash="4 3" if kind == "never" else None)
        # 数值贴柱顶
        f.t(cx(i), BASE - h - 10, human(b), DARK[kind], bold=True,
            size=_sz(12.5), anchor="middle")
        # 标签在基线下
        f.t(cx(i), BASE + 20, lab, DARK[kind] if disk else GY,
            bold=disk, size=_sz(12), anchor="middle")
        f.t(cx(i), BASE + 37, sub, GY2, size=_sz(11), anchor="middle")
        if disk:
            f.t(cx(i), BASE + 55, "落盘 ⏷", "#0d652d", bold=True,
                size=_sz(11), anchor="middle")

    # ── 柱间倍数：把对数轴吃掉的落差用文字补回来 ──────────────
    for i in range(n - 1):
        b0, b1 = PTS[i][2], PTS[i + 1][2]
        r = b1 / float(b0)
        lab = ("× %s" % _mult(r)) if r >= 1 else ("÷ %s" % _mult(1.0 / r))
        col = "#a50e0e" if r >= 1 else "#0d652d"
        mx = (cx(i) + cx(i + 1)) / 2.0
        f.t(mx, BASE - 6, lab, col, bold=True, size=_sz(11.5), anchor="middle")

    # ── 把「腰」单独标出来 ────────────────────────────────────
    wi = 4                                     # latent 那一根
    f.line(cx(wi), BASE - hpx(PTS[wi][2]) - 40, cx(wi),
           BASE - hpx(PTS[wi][2]) - 14, GR, 2)
    f.t(cx(wi), BASE - hpx(PTS[wi][2]) - 48,
        "⭐ 最细的腰 ——&#160;跨机只传这一份",
        "#0d652d", bold=True, size=_sz(12.5), anchor="middle")

    y = BASE + 76

    # ══════════════════ 中：三条读数 ══════════════════
    ROWS = (
        ("压缩比", "46.3 ×",
         "223,948,800 个像素数 ÷ 4,838,400 个 latent 数",
         "空间 8×8 ＝ 64 倍、时间 81→21 ＝ 3.86 倍，通道 3→16 反向摊薄 5.33 倍"),
        ("腰有多细", "2.5 %",
         "19.4 MB ÷ 774 MB（DiT 单个激活）",
         "相对展开后的 448 MB 像素张量是 4.3%；相对 457 GB 那根是 0.0042%"),
        ("传它要多久", "1.5 ms",
         "19.4 MB 走 100 Gbps ＝ 1.5 ms；走 1 Gbps 也只要 155 ms",
         "⭐ 而 Stage 2 本身要跑 229 秒 ——&#160;传输占比 0.0007%，可以当成零"),
    )
    hy = y
    f.box(0, hy, W, 30 + len(ROWS) * 50, "#fff", LINE, 8)
    f.colhead(16, hy + 20, "读三个数")
    f.colhead(190, hy + 20, "多少")
    f.colhead(330, hy + 20, "怎么算的")
    f.colhead(770, hy + 20, "旁注")
    f.line(0, hy + 30, W, hy + 30, LINE, 1, arrow=False)
    for i, (k, v, how, note) in enumerate(ROWS):
        yy = hy + 30 + i * 50
        if i:
            f.line(0, yy, W, yy, LINE2, 1, arrow=False)
        f.t(16, yy + 22, k, INK, bold=True, size=_sz(12))
        f.t(190, yy + 22, v, "#0d652d", bold=True, size=_sz(14))
        f.t(330, yy + 22, how, GY, size=_sz(11.5), w=430)
        f.t(770, yy + 22, note, GY2, size=_sz(11.5), w=W - 786)
    y = hy + 30 + len(ROWS) * 50 + 22

    # ══════════════════ 落点 ══════════════════
    y = f.band(y, "ok",
               "为什么「最细的那一处」正好就是「该切开的那一处」",
               ['把流水线切开，代价是<tspan font-weight="700">切口上的数据要搬一趟</tspan>。'
                '所以切在哪，取决于<tspan font-weight="700">哪儿的数据最少</tspan>。',
                '⭐ 而扩散模型的形状<tspan font-weight="700">天然把这个位置摆在了明处</tspan>：'
                'DiT 吐出来的 latent 是全程最瘦的一处 ——&#160;'
                '往前是 774 MB 的激活，往后是 448 MB 的像素，它自己只有 19.4 MB。',
                '⭐⭐ 于是这一刀几乎<tspan font-weight="700">不要钱</tspan>：'
                '搬一趟 1.5 毫秒，而被切开的那一段要算 229 秒。'
                '<tspan font-weight="700">切口成本相对计算量是四个数量级以下的小数 ——'
                '「能不能切」这个问题在这里根本不成立，只剩「要不要切」。</tspan>'])

    y = f.band(y + 14, "bad",
               "⛔ 别把这张图读成「腰细所以那一段轻松」——&#160;正相反",
               ['<tspan font-weight="700">这张图画的是数据体积，不是时间。</tspan>'
                '产出那 19.4 MB 的 Stage 2，占了全程<tspan font-weight="700">九成以上的时间</tspan>。',
                '⛔ 那根 457 GB 的红柱子也要读对：'
                '<tspan font-weight="700">它从不真的存在。</tspan>'
                'Splash Attention 是分块算的，一块算完即弃 ——&#160;'
                '画它是为了说明<tspan font-weight="700">「不分块就没法跑」</tspan>，不是说 HBM 里真有这么多。',
                '⚠️ 口径：全部以 Wan2.1-T2V-14B、1280×720、81 帧、50 步为准。'
                '换模型换分辨率，绝对值全变 ——&#160;'
                '<tspan font-weight="700">但「中间那一处最瘦」这个形状不变，'
                '因为它是 VAE 压缩比决定的。</tspan>'])

    y = f.src(y + 18,
              '三个落盘点的字节数 ——&#160;本仓库 tpu/Wan2.1/generate_diffusers_torchax_staged/'
              'stage_outputs/ 下三个文件的实际大小，可逐字节复核；'
              '⭐ 其中 latent 那个：16×21×90×160×4 B ＋ 272 B 头 ＝ 19,353,872，一字不差',
              '75,600 token 与 5,120 维、40 头 ——&#160;Wan 官方 config wan_t2v_14B.py；'
              '229 秒 ——&#160;本仓库 Wan2.1/README 的 v6e-8 实测表。'
              '⛔ 457 GB 是「若显式展开」的假想值，实际从不落地。')
    f.save("figx-10.svg", y + 6)


main()
