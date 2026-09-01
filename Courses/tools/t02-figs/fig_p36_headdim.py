# -*- coding: utf-8 -*-
"""图 P-36 —— head_dim ＝ 128 这笔账：为什么它打 TPU 却不打 GPU。

**3.4 最能直接拿去用的一条，之前只活在图 T-4 的一张卡片里。** 这张图把它单独摆开，
因为它是整门课里<b>唯一一个「改模型配置就能换来两位数 MFU」</b>的例子。

**关键在于 head_dim 在两个矩阵乘里落在不同的维上。**
① <code>QKᵀ</code>：head_dim 是<b>收缩维</b> —— 撞的是 MXU 那 256 行；
② <code>PV</code>：head_dim 是<b>输出维</b> —— 撞的是 MXU 那 256 列。
<b>两处都只喂满一半，但堵的不是同一条边</b>。这一点 T-4 的卡片里没画出来，
只写了「两处各只喂满 256 的一半」—— 结论对，机制糊。

**GPU 那边同一个 128 一点事没有。** 收缩维 16 的 8 倍，切八条指令连发、
累加器一直待在 TMEM 里不落地；输出维 128 本来就在 wgmma／tcgen05 的合法 N 里。
<b>所以「head_dim 要对齐硬件」这句建议，换个平台就失效。</b>

**⚠️ 出处口径。** 「上限 50%」是<b>纯几何</b>推的，不需要任何内部信息：
128 ÷ 256。Qwen3-30B 那组 21% / 32% / 46% 是<b>实测</b>，
和 T-4 卡片 C 同源，这里不改数也不外推 —— <b>换个模型、换个序列长度都不该照抄。</b>
"""
from common import Fig, para, BL, GN, RD, YL, PU, TL, INK, SUB, GREY, FILL

W = 1400

PAN_Y, PAN_H = 84, 360
MEA_Y, MEA_H = PAN_Y + PAN_H + 22, 126
LAND_Y, LAND_H = MEA_Y + MEA_H + 22, 112
SRC_Y, SRC_H = LAND_Y + LAND_H + 22, 92
H = SRC_Y + SRC_H + 20

L_X, L_W = 20, 674
R_X, R_W = 706, 674

N = 16                      # 画 16×16 代表 256×256，一格 ＝ 16×16 个 cell
CS = 8.4                    # 格边长
CG = 1.2                    # 格间距
GRID = N * (CS + CG) - CG   # ≈ 152


def build():
    f = Fig(W, H, "head_dim 等于 128 时，TPU 的 MXU 在 QK 转置处只用到一半的行、"
                  "在 PV 处只用到一半的列，两处上限都是 50%；GPU 侧收缩维 16 与"
                  "输出维 128 都整除，没有浪费")
    f.title("<tspan font-weight=\"700\">head_dim ＝ 128</tspan> 这笔账"
            "　—— 同一个数字，打 TPU 不打 GPU", "3.4 最能直接用的一条")
    f.legend([(GN, "在动的行／列"), (GREY, "空转：没有第二个任务能填进来"),
              (BL, "GPU：整除，一点不浪费")])
    _tpu(f)
    _gpu(f)
    _measure(f)
    _land(f)
    _src(f)
    return f.out()


# ══════════════════════════════════════════════════════════════════════
def _grid(f, x, y, rows=None, cols=None, c=GN):
    """rows ＝ 前几行在动；cols ＝ 前几列在动。只给一个，另一个视为全开。"""
    for r in range(N):
        for cc in range(N):
            on = (rows is None or r < rows) and (cols is None or cc < cols)
            f.rect(x + cc * (CS + CG), y + r * (CS + CG), CS, CS,
                   FILL[c] if on else "#e0e3e7",
                   c if on else "#c4c8cc", 0.5, 1)
    # 光靠深浅还不够 —— 把空转的那一块直接框出来并署名，才是「一眼可见」。
    if rows is not None:
        f.rect(x, y + rows * (CS + CG), GRID, GRID - rows * (CS + CG) + CG,
               "none", RD, 1.6, 2, "4,3")
    if cols is not None:
        f.rect(x + cols * (CS + CG), y, GRID - cols * (CS + CG) + CG, GRID,
               "none", RD, 1.6, 2, "4,3")


# ── TPU 侧：两处各堵一半，但堵的边不同 ────────────────────────────────
def _tpu(f):
    f.rect(L_X, PAN_Y, L_W, PAN_H, FILL[RD], RD, 1.8, 10)
    f.t(L_X + 18, PAN_Y + 28, "TPU v7　·　MXU 是 256 × 256", "sec", RD)
    f.t(L_X + 18, PAN_Y + 50, "head_dim ＝ 128 在两个矩阵乘里各堵一次 —— "
                              "而且堵的不是同一条边", "xxs", SUB)

    for i, (tag, sub, rows, cols, note) in enumerate([
        ("① QKᵀ", "head_dim 是<b>收缩维</b>", 8, None,
         "喂进去 128 行，<b>下面 128 行没有数据</b>，整趟空转。"),
        ("② PV", "head_dim 是<b>输出维</b>", None, 8,
         "输出只要 128 列，<b>右边 128 列没有产出</b>，一样空转。"),
    ]):
        gx = L_X + 34 + i * 320
        f.t(gx, PAN_Y + 84, tag, "box", RD)
        para(f, gx + 60, PAN_Y + 84, 214, sub, "xs", 14, max_lines=2)
        _grid(f, gx, PAN_Y + 92, rows, cols)
        f.t(gx + GRID + 8, PAN_Y + 92 + GRID - 4, "50%", "numb", RD)
        para(f, gx, PAN_Y + 92 + GRID + 18, 268, note, "xs", 16, max_lines=2)

    f.rect(L_X + 18, PAN_Y + PAN_H - 70, L_W - 36, 56, "#fff", RD, 1.4, 6)
    para(f, L_X + 32, PAN_Y + PAN_H - 49, L_W - 64,
         "<r>灰的那一半没有办法拿去干别的</r> —— TPU 没有「换一个跑」这一层"
         "（§3 一路在说的那件事，在这儿又结了一次账）。"
         "<b>上限 50% 是纯几何推的：128 ÷ 256。</b>", "xs", 16, max_lines=3)


# ── GPU 侧：同一个 128，两处都整除 ────────────────────────────────────
def _gpu(f):
    f.rect(R_X, PAN_Y, R_W, PAN_H, FILL[BL], BL, 1.8, 10)
    f.t(R_X + 18, PAN_Y + 28, "GPU　·　收缩维 16、输出维步长 8", "sec", BL)
    f.t(R_X + 18, PAN_Y + 50, "同一个 head_dim ＝ 128，两处都整除 —— "
                              "改它的收益接近于零", "xxs", SUB)

    # ① QKᵀ：K = 128 切成 8 条 K=16 连发
    f.t(R_X + 34, PAN_Y + 84, "① QKᵀ", "box", BL)
    para(f, R_X + 94, PAN_Y + 84, 240, "K ＝ 128 ＝ <b>16 × 8</b>", "xs", 14,
         max_lines=2)
    for k in range(8):
        f.rect(R_X + 34 + k * 34, PAN_Y + 104, 28, 40, FILL[BL], BL, 1.3, 3)
        f.t(R_X + 48 + k * 34, PAN_Y + 128, "16", "xxs", BL, anchor="middle")
    f.line(R_X + 34, PAN_Y + 152, R_X + 34 + 7 * 34 + 28, PAN_Y + 152, BL, 1.6,
           marker="aB")
    para(f, R_X + 34, PAN_Y + 172, 290,
         "八条指令连发，<b>累加器一直待在 TMEM 里不落地</b>。"
         "<b>喂满率 100%。</b>", "xs", 16, max_lines=3)

    # ② PV：N = 128 本来就是合法宽度
    f.t(R_X + 360, PAN_Y + 84, "② PV", "box", BL)
    para(f, R_X + 412, PAN_Y + 84, 240, "N ＝ 128，<b>合法宽度</b>", "xs", 14,
         max_lines=2)
    f.rect(R_X + 360, PAN_Y + 104, 272, 40, FILL[BL], BL, 1.3, 3)
    f.t(R_X + 496, PAN_Y + 128, "N ＝ 128", "lbl", BL, anchor="middle")
    para(f, R_X + 360, PAN_Y + 172, 290,
         "wgmma 的 N 从 8 到 256 可选，tcgen05 单 SM 步长 8、配对步长 16 —— "
         "<b>128 三种都落在合法值里。</b>", "xs", 16, max_lines=4)

    f.rect(R_X + 34, PAN_Y + 224, R_W - 68, 58, "#fff", YL, 1.4, 6)
    para(f, R_X + 48, PAN_Y + 244, R_W - 96,
         "<b>但别读成「GPU 不挑食」。</b>它挑的是 16 的倍数 —— "
         "<code>head_dim ＝ 24</code> 在 GPU 上同样只有 75% "
         "（一条满 K=16 ＋ 一条只喂 8）。<g>这一条是从整除关系推的。</g>"
         "<b>两边都挑，只是那条边一个 16、一个 256。</b>", "xs", 16, max_lines=3)

    f.rect(R_X + 18, PAN_Y + PAN_H - 70, R_W - 36, 56, "#fff", BL, 1.4, 6)
    para(f, R_X + 32, PAN_Y + PAN_H - 49, R_W - 64,
         "<b>所以「head_dim 要对齐硬件」这条建议是有前提的。</b>"
         "在 GPU 上调它收益接近于零，在 TPU 上是实打实的 —— "
         "<r>同一个模型配置，两边的浪费根本不在一个位置。</r>", "xs", 16, max_lines=3)


# ══════════════════════════════════════════════════════════════════════
def _measure(f):
    f.rect(20, MEA_Y, 1360, MEA_H, FILL[GN], GN, 1.8, 10)
    f.t(38, MEA_Y + 28, "📐 实测佐证：参数量和 FLOP 一个都没变，只换了切法", "sec", GN)
    y = para(f, 38, MEA_Y + 54, 1324,
             "把 Qwen3-30B 的注意力从 <b>32 头 × 128</b> 改成 <b>16 头 × 256</b> —— "
             "<b>总的 head 维度不变，参数量不变，FLOP 不变</b>，"
             "变的只有「每一头撞不撞 MXU 那条 256 边」。", "xs", 19)
    y = para(f, 38, y + 2, 1324,
             "MFU 在 8K / 16K / 32K 三个序列长度上分别提升 "
             "<b>21% / 32% / 46%</b>。<g>序列越长，注意力占比越大，这条收益越明显 —— "
             "趋势本身就是这个机制的一个旁证。</g>", "xs", 19)
    para(f, 38, y + 2, 1324,
         "<r>但别把这三个数搬到别的模型上。</r>它们跟层数、序列长度、"
         "batch、有没有开 FlashAttention 都相关。<b>能搬的是那句话：先量 head_dim "
         "对不对得上收缩边，再去调别的。</b>", "xs", 19)


# ══════════════════════════════════════════════════════════════════════
def _land(f):
    f.rect(20, LAND_Y, 1360, LAND_H, FILL[BL], BL, 1.8, 10)
    f.t(38, LAND_Y + 28, "⭐ 落点：模型配置和硬件收缩边，要一起选", "sec", BL)
    y = para(f, 38, LAND_Y + 54, 1324,
             "<b>结论不是「TPU 不适合注意力」。</b>是这门课一路在说的那件事换个说法："
             "<b>静态的硬件把选择权交回给了你 —— 也就是说，选错了没人替你兜。</b>", "xs", 19)
    para(f, 38, y + 2, 1324,
         "<r>GPU 就算真撞上了（比如上面那个 head_dim ＝ 24），空出来的发射槽"
         "还能被别的 warp 顶上，你在 profile 上未必看得见；"
         "TPU 上没有第二个任务，空着就是真空着、真报在 MFU 上。</r>"
         "<b>这不是谁更好，是「谁替你收拾烂摊子」的差别 —— 而那份收拾是要付晶体管的。</b>",
         "xs", 19)


# ══════════════════════════════════════════════════════════════════════
def _src(f):
    f.rect(20, SRC_Y, 1360, SRC_H, "#fff", GREY, 1.4, 10)
    f.t(38, SRC_Y + 26, "⚠️ 出处分层", "sec")
    y = para(f, 38, SRC_Y + 50, 1324,
             "<b>纯几何</b>：上限 50% ＝ 128 ÷ 256，不需要任何内部信息。　"
             "<b>查到的</b>：MXU 256×256 出自公开工程博客；"
             "wgmma／tcgen05 的合法 N 出自 PTX ISA。", "xs", 18)
    para(f, 38, y + 2, 1324,
         "<b>实测</b>：21% / 32% / 46% 与图 T-4 卡片 C 同源，"
         "<r>是一个模型上的一组数，不是规律</r>。"
         "<g>「GPU 上收益接近于零」是从整除关系推的，没有配对实测。</g>", "xxs", 17)


if __name__ == "__main__":
    import io
    io.open("out/fig_p36_headdim.svg", "w", encoding="utf-8").write(build())
    print("ok fig_p36_headdim")
