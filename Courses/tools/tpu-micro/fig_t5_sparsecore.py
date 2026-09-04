# -*- coding: utf-8 -*-
"""图 T-5 —— SparseCore：第二种核，以及它在生产里到底在干什么。

这张图的重点**不是**「TPU 有个专门查表的核」。那是宣传语，也是这份材料上一版
犯的错。真正值得画出来的是三件事：

1. **两颗核的数据流是单向的。** SparseCore 能把结果直接推进 TensorCore 的
   暂存里，反过来不行。这不是软件约定，是硬件把生产者／消费者关系焊死了。
2. **差别在「一次搬多小」，不在「能搬多快」。** 两颗核共用 HBM 控制器，
   32 B 的通道宽度也一样 —— 差的是 tile 形状：(8,128) vs (8,)。
3. **它在我们的生产任务里一次 embedding 都没查。** 判据是 duplication factor，
   而语言模型的词表访问模式恰好落在这条判据的最不划算的一端。

单向那一条的完整依据只在内部资料里，公开版必须如实说「公开资料没有列出」，
不能把一个查不到出处的结论当结论讲 —— 所以这张图的箭头样式本身是随模式变的。
"""
from common import Fig, para, BL, RD, YL, GN, PU, TL, INK, SUB, GREY, FILL
import gate

W = 1400
TOP = 84

# ── 左栏：核间数据流有向图 ──────────────────────────────────────────────
LX, LW = 20, 620
GY = 176                       # 图区顶
ICI_Y = GY
CORE_Y = GY + 72
CORE_H = 142
HBM_Y = CORE_Y + CORE_H + 44
# 两个核之间要留 120px：核间那两条箭头的标签得写在缝里，
# 上一版缝只有 60px，标签直接压到 SparseCore 的标题上。
TCX, SCX, CW_ = 40, 360, 200
BARW = SCX + CW_ - TCX

NOTE_Y = HBM_Y + 56
GRAN_Y = NOTE_Y + 92           # 粒度对照块

# ── 右栏：逐项对照表 + 卡片 ────────────────────────────────────────────
RX, RW = 664, 716
TY = TOP + 36                  # 必须让开节标题（基线 TOP+26），不然表头压在标题上
THDR = 32
TRH = 36
NROW = 8
TBOT = TY + THDR + NROW * TRH

CA_Y = TBOT + 18               # 卡片 A：它能算，但算不了矩阵乘
CB_Y = CA_Y + 130              # 卡片 B：为什么生产里它没在查表

BAND_Y = max(GRAN_Y + 178, CB_Y + 174) + 20
H = BAND_Y + 152


def build():
    f = Fig(W, H, "TPU v7 SparseCore 与 TensorCore 的分工：核间数据流单向，"
                  "SparseCore 可直推 TensorCore 暂存，反向不存在")
    f.title("SparseCore　—— 第二种核，以及它在生产里到底在干什么", "第 5 / 8 张")
    # 图例只列这张图上真的画出来的东西 —— 挂一条没有指涉的图例，
    # 读者会满图去找那个颜色（T-1 已经踩过一次）。
    f.legend([(GN, "TensorCore 侧"), (PU, "SparseCore 侧"), (BL, "存在的数据通路"),
              (GREY, "公开资料没有列出") if gate.is_public()
              else (RD, "硬件上不存在的方向")])

    _graph(f)
    _granularity(f)
    _table(f)
    _cards(f)
    _band(f)
    return f.out()


# ══════════════════════════════════════════════════════════════════════
def _graph(f):
    f.t(LX, TOP + 26, "两颗核之间怎么传数　—— 注意它是单向的", "sec")
    para(f, LX, TOP + 46, LW,
         "一颗 device 里有 <b>1 个 TensorCore</b> 和 <b>2 个 SparseCore</b>。"
         "它们<r>不共享地址空间、也没有缓存一致性</r> —— 跨核只有 DMA 一条路，"
         "而 DMA 的落点可以直接是对方的私有 SRAM，不必绕 HBM。", "xs", 15)

    # ICI 出口
    f.rect(TCX, ICI_Y, BARW, 30, FILL[TL], TL, 1.4, 6)
    f.t(TCX + 12, ICI_Y + 20, "ICI　—— 出芯片，去邻居（见 §8）", "lbl", TL)

    # HBM
    f.rect(TCX, HBM_Y, BARW, 30, FILL[SUB], SUB, 1.4, 6)
    f.t(TCX + 12, HBM_Y + 20, "HBM　96 GiB / device　—— 两颗核共用同一个控制器", "lbl", INK)

    # TensorCore
    f.rect(TCX, CORE_Y, CW_, CORE_H, FILL[GN], GN, 2, 8)
    f.t(TCX + 12, CORE_Y + 24, "TensorCore ×1", "box", GN)
    f.rect(TCX + 12, CORE_Y + 34, CW_ - 24, 46, "#fff", GN, 1.2, 5)
    f.t(TCX + 22, CORE_Y + 52, "VMEM　64 MiB", "lbl", GN)
    f.t(TCX + 22, CORE_Y + 70, "最小一块 tile (8,128)", "xxs", GN)
    para(f, TCX + 12, CORE_Y + 96, CW_ - 24,
         "2 个 MXU ＋ 1 个 VPU。<b>矩阵乘只在这里发生。</b>", "xxs", 13)

    # SparseCore
    f.rect(SCX, CORE_Y, CW_, CORE_H, FILL[PU], PU, 2, 8)
    f.t(SCX + 12, CORE_Y + 24, "SparseCore ×2", "box", PU)
    f.rect(SCX + 12, CORE_Y + 34, CW_ - 24, 46, "#fff", PU, 1.2, 5)
    f.t(SCX + 22, CORE_Y + 52, "私有 SRAM　512 KiB", "lbl", PU)
    f.t(SCX + 22, CORE_Y + 70, "最小一块 (8,)", "xxs", PU)
    para(f, SCX + 12, CORE_Y + 96, CW_ - 24,
         "16 个向量子核 ＋ 1 个标量子核。<r>没有 MXU</r>。", "xxs", 13)

    # ── 通路 ───────────────────────────────────────────────────────────
    ch = (lambda n, s: s) if gate.is_public() else \
         (lambda n, s: gate.IP("通道 %d：%s" % (n, s), s, why="内部通道编号"))

    def up(x, y0, y1, c, lab, anchor="start"):
        f.line(x, y0, x, y1, c, 1.8, marker={BL: "aB", GN: "aG", PU: "aP"}[c])
        f.t(x + (7 if anchor == "start" else -7), (y0 + y1) / 2 + 4, lab,
            "xxs", c, "start" if anchor == "start" else "end")

    # HBM → 两颗核
    up(TCX + 50, HBM_Y - 2, CORE_Y + CORE_H + 6, BL, ch(2, "HBM → VMEM"))
    up(SCX + 50, HBM_Y - 2, CORE_Y + CORE_H + 6, BL, ch(6, "HBM → SC 私有 SRAM"))
    # 两颗核 → ICI
    up(TCX + 152, CORE_Y - 2, ICI_Y + 36, BL, ch(1, "VMEM → ICI"), "end")
    up(SCX + 152, CORE_Y - 2, ICI_Y + 36, BL, ch(8, "SC → ICI"), "end")

    # ── 核间那两条：缝宽 120px，标签只能写短的，长解释放到图下方 ────────
    gx0, gx1 = TCX + CW_, SCX               # 缝的左右边界
    gmid = (gx0 + gx1) / 2

    # ① SparseCore → TensorCore：唯一的核间直连
    my = CORE_Y + 44
    f.t(gmid, my - 26, "① 能：SC → TC", "box", PU, "middle")
    f.line(gx1 - 2, my, gx0 + 6, my, PU, 2.6, marker="aP")
    f.t(gmid, my + 18, ch(9, "直写 VMEM"), "xxs", PU, "middle")

    # ② 反方向：不存在（内部）／ 公开资料没列（公开）
    my2 = CORE_Y + CORE_H - 26
    c2 = GREY if gate.is_public() else RD
    f.t(gmid, my2 - 26, "② ？" if gate.is_public() else "② 不能", "box", c2, "middle")
    f.line(gx0 + 2, my2, gx1 - 6, my2, c2, 1.8, dash="5,4")
    f.t(gmid, my2 + 18, "公开未列" if gate.is_public() else "TC → SC", "xxs", c2, "middle")

    # 图下方的结论
    para(f, LX, NOTE_Y, LW,
         gate.IP(
             "端点一共 8 个，两两连通该有约 50 条有向路线，<b>实际只有 17 条</b> —— "
             "这不是一个全交叉开关，是<r>手挑出来的清单</r>。让 SparseCore 能写进 VMEM "
             "实打实占掉了其中一条编号通道，不是「顺便允许」。",
             "各条通路的<b>存在</b>可以从公开的 Pallas SparseCore 接口和它的内存空间约束看出来，"
             "但<g>完整的通道清单、以及反方向到底存不存在，公开资料没有列出</g> —— "
             "本文只画能站住的部分。",
             why="内部通道清单"), "xs", 16)
    para(f, LX, NOTE_Y + 52, LW,
         gate.I("<b>缺的那条（②）是 TensorCore → SparseCore</b>：SparseCore 能把结果推给 "
                "TensorCore，TensorCore 却推不回去，要给它送数据只能经 HBM 绕一圈。"
                "硬件把「谁是生产者、谁是消费者」直接焊进了连线里。", why="同上"), "xs", 16)


# ══════════════════════════════════════════════════════════════════════
def _granularity(f):
    """粒度对照：差的是 tile 形状，不是通道宽度。这条特别容易被讲错。"""
    f.rect(LX, GRAN_Y, LW, 168, "#fff", YL, 1.8, 10)
    f.t(LX + 14, GRAN_Y + 26, "「细 128 倍」指的是 tile 形状，不是 DMA 更快", "sec", YL)

    bx, by = LX + 16, GRAN_Y + 44
    # TensorCore 的一块
    f.t(bx, by + 12, "TensorCore 的最小一块", "lbl", GN)
    f.rect(bx, by + 20, 208, 52, FILL[GN], GN, 1.6, 4)
    f.t(bx + 104, by + 42, "(8, 128)", "numb", GN, "middle")
    f.t(bx + 104, by + 60, "= 4,096 B", "xxs", GN, "middle")

    # SparseCore 的一块
    f.t(bx + 250, by + 12, "SparseCore 的最小一块", "lbl", PU)
    f.rect(bx + 250, by + 20, 208, 52, "#fff", "#e8eaed", 1.0, 4)
    f.rect(bx + 250, by + 20, 208 / 128.0 * 1.0 + 12, 52, FILL[PU], PU, 1.6, 4)
    f.t(bx + 250 + 108, by + 42, "(8,) = 32 B", "numb", PU, "middle")
    f.t(bx + 250 + 108, by + 60, "同一张图上按比例画就是左边的 1/128", "xxs", PU, "middle")

    para(f, LX + 16, GRAN_Y + 130, LW - 32,
         "<b>32 B 是 HBM 通道宽度，两颗核完全一样。</b>SparseCore 并没有更快的搬运器，"
         "它只是允许你<r>按 32 B 为单位去要</r>；TensorCore 一开口就是一个 4 KB 的 tile。"
         "散落在词表里的几百行，用左边那种块去取，取回来的绝大部分都会被扔掉。", "xs", 16)


# ══════════════════════════════════════════════════════════════════════
ROWS = [
    ("每颗 chip 几个", "2 个 TensorCore", "4 个 SparseCore", None),
    ("每个 device 几个", "1", "2", None),
    ("一条向量指令多宽", "8 × 128 ＝ 1,024 格", "16 条 lane", None),
    ("私有 SRAM", "VMEM 64 MiB ＋ SMEM 1 MiB", "512 KiB / 子核", None),
    ("DMA 最小粒度", "32 B（通道宽度）", "32 B（同上）", "两边一样"),
    ("最小可寻址的一块", "tile (8, 128) ＝ 4,096 B", "(8,) ＝ 32 B", "差 128 倍"),
    ("有没有矩阵乘单元", "2 个 MXU，256×256", "没有", None),
    ("bf16 峰值", "≈1,155 TFLOP/s / device", "整颗 chip 4 个合计约 MXU 的 1%", "量级差两位"),
]


def _table(f):
    f.t(RX, TOP + 26, "两种核逐项对照", "sec")

    # 三列必须刚好加满 RW，且第三列要额外留 96px 给右上角的对比徽章
    c0, c1, c2 = 196, 232, 288
    f.rect(RX, TY, RW, THDR + NROW * TRH, "#fff", INK, 1.8, 10)
    f.rect(RX, TY, RW, THDR, FILL[SUB], rx=10)
    f.rect(RX, TY + THDR - 10, RW, 10, FILL[SUB], rx=0)
    f.t(RX + 14, TY + 21, "问的是同一件事", "box")
    f.t(RX + c0 + 14, TY + 21, "TensorCore", "box", GN)
    f.t(RX + c0 + c1 + 14, TY + 21, "SparseCore", "box", PU)
    f.line(RX + c0, TY, RX + c0, TBOT, "#e8eaed", 1)
    f.line(RX + c0 + c1, TY, RX + c0 + c1, TBOT, "#e8eaed", 1)

    for i, (q, a, b, tag) in enumerate(ROWS):
        y = TY + THDR + i * TRH
        if i:
            f.line(RX + 8, y, RX + RW - 8, y, "#e8eaed", 1)
        para(f, RX + 14, y + 16, c0 - 22, q, "xs", 13)
        para(f, RX + c0 + 14, y + 16, c1 - 22, a, "xxs", 13, GN)
        para(f, RX + c0 + c1 + 14, y + 16, c2 - 22 - (96 if tag else 0), b, "xxs", 13, PU)
        if tag:
            f.rect(RX + RW - 90, y + 8, 80, 20, FILL[YL], YL, 1.0, 10)
            f.t(RX + RW - 50, y + 22, tag, "xxs", "#9a6a00", "middle")


# ══════════════════════════════════════════════════════════════════════
def _cards(f):
    # 卡片 A —— 它能算，只是算不了矩阵乘
    f.rect(RX, CA_Y, RW, 118, FILL[PU], PU, 1.8, 10)
    f.t(RX + 16, CA_Y + 26, "它是一颗真的核，不是一台搬运机", "sec", PU)
    para(f, RX + 16, CA_Y + 48, RW - 32,
         "公开的 <code>pallas.tpu_sc</code> 里能直接看到它的指令面："
         "<code>cumsum</code>、<code>sort_key_val</code>、<code>fetch_and_add</code>、"
         "<code>addupdate_scatter</code>、<code>load_gather</code> —— "
         "全是<b>不规则访存 ＋ 归约</b>这一类活。", "xs", 16)
    para(f, RX + 16, CA_Y + 88, RW - 32,
         "<b>但整份接口里没有任何矩阵乘原语。</b>所以它不是「小一号的 TensorCore」，"
         "是另一种形状的核。", "xs", 16)

    # 卡片 B —— 落点：生产里它没在查表
    f.rect(RX, CB_Y, RW, 162, FILL[RD], RD, 1.8, 10)
    f.t(RX + 16, CB_Y + 26, "最反直觉的一条：我们的生产任务里，它一次表都没查", "sec", RD)
    # ⛔⛔ 2026-09-05 合规修正。这一格原来把那条判据的**具体算式**直接印了出来，
    #    而本课讲义里白纸黑字写过它不进对外材料。
    #    gate.py 的规则是「忘记标注的内容会默认出现在公开版里」—— 这里就是漏标，
    #    于是它跟着渲染进了公开仓库。**这不是排版问题，是闸门漏了一条。**
    #
    # ⚠️ 顺带：那条算式**照字面还算不出例子来**。分母若取整张表的行数，
    #    推荐系统是「几亿行里取几百个」＝ 极小，反而比语言模型那个比值还小得多，
    #    跟「推荐系统很高」正好相反。真正该做分母的是**这一批取到的不重复行数**。
    #    所以内部版也一并改口径，公开版只留白话。
    # ⚠️ 注释里不要复述被挡下的原文 —— 复述会自己撞上闸门（已撞过一次）。
    para(f, RX + 16, CB_Y + 48, RW - 32,
         "要不要把 embedding 卸载到 SparseCore，看的是<b>同一批里「重复取同一行」的程度</b>"
         " —— 重复得越厉害，专用通路省下的越多。" +
         gate.I("判据是 <b>duplication factor</b> ＝ 这一批取了多少个索引 ÷ "
                "<b>其中不重复的行数</b>。", why="判据公式出自内部资料"), "xs", 16)
    para(f, RX + 16, CB_Y + 100, RW - 32,
         "推荐系统那边少量热行被反复命中，重复度很高，这是 SparseCore 的主场。"
         "而我们这次的语言模型<b>几乎每个索引都取不同的行</b>（一批 131,072 个 token，"
         "词表 129,280 行，重合极少）—— 落在<r>最不划算的一端</r>，"
         "编译器于是根本没把它派过去。", "xs", 16)


# ══════════════════════════════════════════════════════════════════════
def _band(f):
    f.rect(20, BAND_Y, 1360, 132, FILL[GN], GN, 1.6, 10)
    f.t(34, BAND_Y + 26, "那生产里的 embedding 到底怎么跑的？—— 它根本不是查表", "sec", GN)
    para(f, 34, BAND_Y + 50, 1330,
         "MaxText 有一个开关 <code>use_iota_embed</code>。打开时，查表被写成"
         "<b>「把 token id 展成 one-hot，再和整张词表做一次矩阵乘」</b> —— "
         "一条 <code>dot</code>，跑在 MXU 上。关掉才是真的 <code>gather</code>。"
         "<r>上游默认是关的，但仓库里 34 份配置显式打开、0 份显式关闭</r>，"
         "包括那几份给 GPU 用的配置。", "xs", 18)
    para(f, 34, BAND_Y + 88, 1330,
         "代价算得清楚：Hunyuan3 那个尺寸下，这条 matmul 是 28.38 TFLOP/device，"
         "占一步的 <b>0.62%</b>，约 24.6 ms；同样的事用 gather 只要约 1.0 ms —— "
         "<r>matmul 慢约 24 倍</r>。所以不要说「反正 MXU 闲着，用算力换带宽很划算」，"
         "算术直接否掉了：它不省时间，它花时间。成立的唯一理由是<b>分母够大</b>，"
         "而真实动机（最可能是反向传播里 scatter-add 在分片下难做）"
         "<g>我没查实，不写成结论</g>。", "xs", 18)


if __name__ == "__main__":
    import io, sys
    io.open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/t5.svg", "w",
            encoding="utf-8").write(build())
    print("ok", H)
