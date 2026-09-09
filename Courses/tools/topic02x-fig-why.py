# -*- coding: utf-8 -*-
r"""外传 图 X-9 · **为什么扩散是计算密集型** —— 拿 Wan2.1 的真配置当场算一遍。

════════════════════════════════════════════════════════════════════
⭐⭐ 这一张回答现场那个问题
════════════════════════════════════════════════════════════════════
「像 Wan2.1 / 2.2、LongCat 这种视频生成和图片生成的，
  它为什么是计算密集型？把这个道理讲一下。」

⛔ **不讲原理讲不清楚，所以这一张破例算得比别的图细。**
  但每一步都用**官方仓库里的真配置**，不用「大概」「一般来说」。

════════════════════════════════════════════════════════════════════
📌 全部输入（Wan2.1-T2V-14B 官方仓库 wan/configs/wan_t2v_14B.py）
════════════════════════════════════════════════════════════════════
    vae_stride  = (4, 8, 8)      时间 4 倍、空间 8×8 下采样
    patch_size  = (1, 2, 2)      时间不切、空间 2×2 合成一个 token
    dim         = 5120
    ffn_dim     = 13824
    num_layers  = 40
    num_heads   = 40
    window_size = (-1, -1)       ⭐⭐ **不开窗口 ＝ 全局注意力**
    text_len    = 512            交叉注意力那一侧的文本长度

⭐ 最后那条 `window_size = (-1, -1)` 是这张图的关键：
  **视频里的注意力是全局的**，七万多个 token 两两之间都算。

════════════════════════════════════════════════════════════════════
🔢 三步推导，每一步都能当场复核
════════════════════════════════════════════════════════════════════
**① 一段 720×1280、81 帧的视频，变成多少个 token**

    latent：(81−1)/4+1 ＝ 21 帧 · 720/8 ＝ 90 · 1280/8 ＝ 160
    patch ：21 · 90/2 ＝ 45 · 160/2 ＝ 80
    **N ＝ 21 × 45 × 80 ＝ 75,600 个 token**

**② 每层每步要算多少（d ＝ 5120，ffn ＝ 13824）**

    自注意力投影  8·N·d²          ＝ 1.59e13
    **注意力本身  4·N²·d          ＝ 1.17e14  ← 占 72%**
    交叉注意力    ≈                 8.77e12
    FFN          4·N·d·ffn       ＝ 2.14e13
    ——&nbsp;每层合计 **1.63e14**，40 层一步 **6.52e15**，50 步 **3.26e17 FLOP**

**③ 外部锚点（⭐ 这一步不能省）**

    每层权重 4d² ＋ 4d² ＋ 2·d·ffn ＝ 351.3 M 参数
    × 40 层 ＝ **14.05 B** ——&nbsp;对上官方标称的 **14B**。
    ⭐ 参数量能对上，说明上面那套 FLOP 公式的形状是对的；
      **对不上任何锚点的孤立数字才该害怕。**

⚠️ 简化说明（图上也标了）：这套 FLOP 只数矩阵乘，没数 norm、激活、RoPE、
   softmax 那些向量运算；它给的是**量级和占比**，不是精确账。
"""
from topic03_draw import (Fig, wpx, _sz, LINE, LINE2,
                          BL, OR, GR, RD, GY, GY2, PU, CY, INK)

W = 1400

d, ffn, L = 5120, 13824, 40
N = 21 * 45 * 80
SELF = 8 * N * d * d
ATTN = 4 * N * N * d
CROSS = 2 * (2 * N + 2 * 512) * d * d + 4 * N * 512 * d
FFNF = 4 * N * d * ffn
PER = SELF + ATTN + CROSS + FFNF


def main():
    f = Fig(W, "以 Wan2.1-T2V-14B 的官方配置算一段 720×1280、81 帧的视频："
               "经 VAE 下采样与 patch 化后是 75600 个 token；每层每步 1.63e14 FLOP，"
               "其中全局注意力占 72%；40 层 50 步共 3.26e17 FLOP")
    f.marks = set()
    y = f.header(
        '为什么扩散是计算密集型 ——&#160;'
        '<tspan font-weight="700">拿 Wan2.1 的真配置当场算一遍</tspan>',
        '⭐ 结论先说：一段 720P、81 帧的视频 ＝ '
        '<tspan font-weight="700">75,600 个 token</tspan>，而注意力是'
        '<tspan font-weight="700">全局的</tspan>——&#160;'
        '于是算力的<tspan font-weight="700">七成花在注意力上，那是纯矩阵乘</tspan>。',
        [(BL, "① 视频怎么变成 token"), (RD, "② 算力花在哪"),
         (GR, "③ 为什么这配 v6e")])

    top = y + 8

    # ══ ① 三级放大 ══
    LW = 396
    PH = 300
    ly = f.panel(0, top, LW, PH, "① 一段视频 ＝ 多少个 token", BL,
                 tag="Wan2.1 官方配置")
    ST = (("像素", "720 × 1280 × 81 帧", "你要的那段视频", INK),
          ("↓ VAE  stride (4, 8, 8)", "", "时间 4 倍、空间 8×8 下采样", GY2),
          ("latent", "21 × 90 × 160", "(81−1)/4+1 ＝ 21", "#174ea6"),
          ("↓ patch (1, 2, 2)", "", "空间 2×2 合成一个 token", GY2),
          ("token", "21 × 45 × 80", "", "#174ea6"))
    yy = ly + 22
    for k, (a, b, c, col) in enumerate(ST):
        if b:
            f.box(20, yy, LW - 40, 46, "#fff", LINE, 6)
            f.t(32, yy + 20, a, col, bold=True, size=_sz(12))
            f.t(32, yy + 38, c, GY2, size=_sz(11), w=LW - 200)
            f.t(LW - 32, yy + 28, b, col, bold=True, size=_sz(13),
                anchor="end", mono=True)
            yy += 52
        else:
            f.t(32, yy + 14, a, GY2, size=_sz(11), mono=True)
            f.t(LW - 32, yy + 14, c, GY2, size=_sz(11), anchor="end")
            yy += 26
    f.box(20, yy + 4, LW - 40, 42, "#fff", BL, 6, 1.6)
    f.t(LW / 2.0, yy + 31, "N ＝ %s 个 token" % format(N, ","),
        BL, bold=True, size=16, anchor="middle")

    # ══ ② FLOP 拆解 ══
    MX = LW + 26
    MW = 470
    my = f.panel(MX, top, MW, PH, "② 每层每步的算力花在哪", RD,
                 tag="d ＝ 5120 · ffn ＝ 13824")
    PARTS = (("注意力本身", "4·N²·d", ATTN, RD),
             ("FFN", "4·N·d·ffn", FFNF, OR),
             ("自注意力投影", "8·N·d²", SELF, BL),
             ("交叉注意力（文本 512）", "≈", CROSS, GY2))
    BW = MW - 210
    yy = my + 22
    for nm, formula, v, col in PARTS:
        f.t(MX + 20, yy + 12, nm, INK, size=_sz(11))
        f.t(MX + 20, yy + 28, formula, GY2, size=_sz(11), mono=True)
        w = BW * v / float(ATTN)
        f.box(MX + 190, yy + 6, max(w, 3), 22, col, col, 3)
        f.t(MX + 190 + max(w, 3) + 8, yy + 22,
            "%.0f%%" % (v / float(PER) * 100), col, bold=True, size=_sz(12))
        yy += 44
    f.line(MX + 20, yy + 2, MX + MW - 20, yy + 2, LINE, 1, arrow=False)
    f.lines(MX + 20, yy + 24, MW - 40, [
        "每层合计 <tspan font-weight=\"700\">1.63e14</tspan> FLOP　→　"
        "40 层一步 <tspan font-weight=\"700\">6.52e15</tspan>",
        "50 步一段视频 <tspan font-weight=\"700\">3.26e17 FLOP</tspan>",
        "⭐⭐ <tspan font-weight=\"700\">注意力占七成</tspan>，而它是 N² 的"
        "<tspan font-weight=\"700\">纯矩阵乘</tspan>",
        "　 ——&#160;config 里 window_size ＝ (−1,−1)，<tspan font-weight=\"700\">"
        "不开窗口，全局都算</tspan>",
    ], size=11, lh=20, fill=GY)

    # ══ ③ 为什么配 v6e ══
    RX = LW + MW + 52
    RW = W - RX
    ry = f.panel(RX, top, RW, PH, "③ 于是它正好配 v6e", GR)
    f.t(RX + 16, ry + 26, "强度 ≈ 每份权重服务多少 token", GY, size=_sz(11))
    f.t(RX + 16, ry + 52, "75,600", "#0d652d", bold=True, size=22, mono=True)
    f.t(RX + 120, ry + 52, "／ 门槛 560", GY, size=_sz(12))
    f.t(RX + 16, ry + 76, "→ 高出 <tspan font-weight=\"700\">135 倍</tspan>"
        "，那根窄管子完全不是瓶颈", GY, size=_sz(11), w=RW - 32)
    f.line(RX + 16, ry + 92, RX + RW - 16, ry + 92, LINE, 1, arrow=False)
    f.lines(RX + 16, ry + 116, RW - 32, [
        "⭐ 而且七成算力是 <tspan font-weight=\"700\">N² 的注意力</tspan>——&#160;",
        "　 纯矩阵乘，<tspan font-weight=\"700\">MXU 的主场</tspan>，",
        "　 正是那两块 256×256 最擅长的形状",
        "",
        "⭐ 剩下要比的只有算力：",
        "　 <tspan font-weight=\"700\">918 对 989.5 ＝ 93%</tspan>，基本打平",
    ], size=11, lh=21, fill=GY)
    f.t(RX + 16, ry + 262,
        "⚠️ 一颗 v6e 的<tspan font-weight=\"700\">纯算力下界约 5.9 分钟</tspan>",
        "#b06000", size=_sz(11), w=RW - 32)
    f.t(RX + 16, ry + 280,
        "　 （3.26e17 ÷ 918 TFLOP/s，100% 利用率，实际做不到）",
        GY2, size=_sz(11), w=RW - 32)

    y = top + PH + 22

    y = f.band(y, "info",
               "把这一整套压成三句话",
               ['① <tspan font-weight="700">一段视频被压成七万五千个 token</tspan>，'
                '而它们共用同一份权重 ——&#160;'
                '每层那 702 MB 的权重搬一次，服务 75,600 个位置。',
                '② <tspan font-weight="700">注意力是全局的，所以算力随 token 数平方涨</tspan>。'
                '七万五千个 token 两两都算，光这一项就占了七成 FLOP ——&#160;'
                '<tspan font-weight="700">这就是「计算密集」四个字的全部来历</tspan>。',
                '③ 于是它落在强度轴最右边：'
                '<tspan font-weight="700">带宽不是瓶颈，比的只剩算力</tspan>；'
                '而这七成算力又恰好是<tspan font-weight="700">纯矩阵乘</tspan>——&#160;'
                'v6e 那两块 256×256 的大方阵，等的就是这种活。'])

    y = f.band(y + 14, "warn",
               "⚠️ 这张图刻意留白的地方",
               ['<tspan font-weight="700">FLOP 只数了矩阵乘</tspan>，没数 norm、'
                '激活、RoPE、softmax 那些向量运算。'
                '它给的是<tspan font-weight="700">量级和占比</tspan>，不是精确账。',
                '<tspan font-weight="700">那个 5.9 分钟是理论下界，不是性能预测</tspan> ——&#160;'
                '它假设 100% 利用率，而实际做不到。'
                '⛔ 本讲不给性能数，这一条也不例外，它只是让「3.26e17」这个数有个体感。',
                '⭐ 但有一条能对得上：<tspan font-weight="700">每层 351.3 M 参数 × 40 层 '
                '＝ 14.05 B，对上官方标称的 14B</tspan>。'
                '——&#160;参数量能对上，说明上面那套公式的形状是对的。'
                '<tspan font-weight="700">对不上任何锚点的孤立数字，才是该害怕的。</tspan>'])

    y = f.src(y + 18,
              'vae_stride (4,8,8)、patch_size (1,2,2)、dim 5120、ffn_dim 13824、'
              'num_layers 40、window_size (−1,−1)、text_len 512 ——&#160;'
              'Wan2.1 官方仓库 wan/configs/wan_t2v_14B.py',
              '同族同量级：Wan2.2-T2V-A14B（总 27B / 每步激活 14B，Wan-AI 官方模型卡）、'
              'LongCat-Video 13.6B（美团官方，DiT 架构）——&#160;'
              '形状不同，但「token 极多 ＋ 全局注意力」这两条一样成立。')
    f.save("figx-9.svg", y + 6)


main()
