# -*- coding: utf-8 -*-
"""图 T-3 —— 并行层级逐层对照：GPU 的每一层，在 TPU 上对应什么。

这张图的设计是「按问题对齐」而不是「按名词对齐」。左边一列写的是问题
（「硬件天然锁步的一组是多少」「运行时谁决定先跑哪个」），中右两列各自回答。

这么排的好处是：**有两行 TPU 那一栏是空的**，而且那两行恰好是 GPU 用来
藏延迟的两层。名词对名词地排是看不出这件事的 —— 会变成一张翻译表。
"""
from common import Fig, para, BL, RD, YL, GN, PU, TL, INK, SUB, GREY, FILL

W = 1400
TOP = 84

TX0, TW = 20, 1014
C0, C1, C2 = 250, 382, 382          # 列宽：问题 / GPU / TPU
HDR = 36
RH = 88
NROW = 7
TB = TOP + 34                       # 表头 y
TBODY = TB + HDR
TBOT = TBODY + NROW * RH

RX, RW = 1050, 330
BAND_Y = TBOT + 22
H = BAND_Y + 146


def _dots(f, x, y, n, cols, c, cell=5.0, gap=1.4, cap=None):
    """画 n 个小格子。cap 限制实际画出来的个数（画不下就截断并标注）。"""
    m = min(n, cap or n)
    for i in range(m):
        r, cc = divmod(i, cols)
        f.rect(x + cc * (cell + gap), y + r * (cell + gap), cell, cell, FILL[c], c, 0.4, 0.8)
    return y + ((m + cols - 1) // cols) * (cell + gap)


def build():
    f = Fig(W, H, "GPU 与 TPU 并行层级逐层对照：TPU 缺少「共享暂存的线程组」"
                  "与「运行时调度」这两层，而这两层正是 GPU 用来隐藏延迟的")
    f.title("并行层级逐层对照　—— 按问题对齐，不按名词对齐", "第 3 / 8 张")
    f.legend([(BL, "NVIDIA B200"), (GN, "TPU v7"), (RD, "TPU 上没有这一层")])

    _table(f)
    _side(f)
    _band(f)
    return f.out()


# ══════════════════════════════════════════════════════════════════════
def _table(f):
    f.rect(TX0, TB, TW, HDR + NROW * RH, "#fff", INK, 1.8, 10)
    f.rect(TX0, TB, TW, HDR, FILL[SUB], rx=10)
    f.rect(TX0, TB + HDR - 10, TW, 10, FILL[SUB], rx=0)
    f.t(TX0 + 16, TB + 23, "问的是同一个问题", "box")
    f.t(TX0 + C0 + 16, TB + 23, "NVIDIA B200", "box", BL)
    f.t(TX0 + C0 + C1 + 16, TB + 23, "TPU v7", "box", GN)
    f.line(TX0 + C0, TB, TX0 + C0, TBOT, "#e8eaed", 1)
    f.line(TX0 + C0 + C1, TB, TX0 + C0 + C1, TBOT, "#e8eaed", 1)

    for i, row in enumerate(ROWS):
        y = TBODY + i * RH
        if i:
            f.line(TX0 + 10, y, TX0 + TW - 10, y, "#e8eaed", 1)
        if row["absent"]:
            f.rect(TX0 + C0 + C1 + 6, y + 4, C2 - 12, RH - 8, FILL[RD], rx=6)
        f.t(TX0 + 16, y + 26, row["q"], "lbl", INK)
        para(f, TX0 + 16, y + 44, C0 - 30, row["qn"], "xxs", 13)
        row["gpu"](f, TX0 + C0 + 16, y)
        row["tpu"](f, TX0 + C0 + C1 + 16, y)


# ── 各单元格的画法 ─────────────────────────────────────────────────────
def _cell_text(title, note, c=INK, sub=None):
    def draw(f, x, y):
        f.t(x, y + 24, title, "lbl", c)
        if sub:
            f.t(x + 300, y + 24, sub, "xxs", c, "end")
        para(f, x, y + 42, 350, note, "xxs", 13)
    return draw


def _cell_dots(title, n, cols, c, note, cell=5.0, note_x=130, foot=None):
    """点阵 + 右侧说明。note_x 必须大于点阵实际宽度 ——
    上一版把两者都固定住，换了 cols 之后说明直接压在格子上。"""
    def draw(f, x, y):
        f.t(x, y + 22, title, "lbl", c)
        by = _dots(f, x, y + 30, n, cols, c, cell)
        if foot:
            f.t(x, by + 12, foot, "xxs", c)
        para(f, x + note_x, y + 42, 350 - note_x, note, "xxs", 13)
    return draw


def _cell_absent(what, why):
    def draw(f, x, y):
        f.t(x, y + 26, "✕　" + what, "box", RD)
        para(f, x, y + 46, 350, why, "xxs", 13)
    return draw


ROWS = [
    dict(q="最小的那个东西是什么", qn="一次能被单独指名道姓的最小执行体",
         absent=False,
         gpu=_cell_text("1 个 thread", "有自己的程序计数器、自己的寄存器。"
                        "<b>可以走自己的分支</b>。", BL),
         tpu=_cell_text("1 个元素", "只是向量里的一格。<b>没有程序计数器，"
                        "也不能走自己的分支</b> —— 它不是执行体，是数据。", GN)),

    dict(q="硬件天然锁步的一组是多少", qn="这一组必须一起执行同一条指令",
         absent=False,
         gpu=_cell_dots("1 个 warp ＝ 32 条 lane", 32, 16, BL,
                        "32 个 thread 锁步。分支不一致时<r>两边都要走一遍</r>。"),
         tpu=_cell_dots("1 条向量指令 ＝ 8 × 128", 128, 16, GN,
                        "8 个 sublane × 128 条 lane、共 1,024 个元素一起动。"
                        "<b>连「分支不一致」这个概念都没有</b> —— "
                        "没有分支可言。", 4.2, note_x=132,
                        foot="每格 ＝ 8 条 lane")),

    dict(q="共享一块暂存的是哪一组", qn="谁和谁能通过片上暂存互相看见数据",
         absent=True,
         gpu=_cell_text("1 个 thread block", "同一个 block 里的 warp 共享一块 shared memory，"
                        "由程序员在核函数里划定，<b>运行时才知道有几个</b>。", BL),
         tpu=_cell_absent("没有这一层",
                          "VMEM 属于<b>整个 TensorCore</b>，不属于某一组。"
                          "谁能看见什么，编译期就定死了 —— 没有「一组」这个中间概念。")),

    dict(q="运行时谁决定接下来跑哪个", qn="延迟出现时，硬件有没有别的活可切",
         absent=True,
         gpu=_cell_text("4 个 warp 调度器 / 64 个 warp 槽",
                        "某个 warp 卡在访存上，调度器立刻换一个能跑的。"
                        "<b>这就是 GPU 藏延迟的全部秘密</b>。", BL),
         tpu=_cell_absent("没有这一层",
                          "指令什么时候发、数据什么时候到，<b>编译期就排死了</b>。"
                          "卡住了就是真的空转 —— 没有别的活能顶上来（见 §7）。")),

    dict(q="一个物理核里有什么", qn="最小的、自带完整控制通路的硬件块",
         absent=False,
         gpu=_cell_text("1 个 SM", "128 个 CUDA Core ＋ 4 个 Tensor Core ＋ "
                        "228 KB L1／共享内存 <g>（容量为第三方）</g>", BL, "×148"),
         tpu=_cell_text("1 个 TensorCore", "2 个 MXU ＋ 1 个 VPU ＋ 64 MiB VMEM ＋ "
                        "1 MiB SMEM", GN, "×2")),

    dict(q="一颗芯片对软件是几个", qn="框架里 <code>devices()</code> 数出来是几",
         absent=False,
         gpu=_cell_text("1 个", "两个 die 由一致性总线缝成一个 GPU。"
                        "跨 die 更慢，<b>但 API 里看不出来</b>。", BL),
         tpu=_cell_text("2 个", "两个 chiplet 如实暴露成两个 device。"
                        "<b>缝在明处，由你的分片策略面对</b>。", GN)),

    dict(q="不换协议能连到多大", qn="超出这个规模就得换一套互联",
         absent=False,
         gpu=_cell_text("72 颗", "一个 NVLink 域。再往外换 InfiniBand／以太网，"
                        "<b>编程模型也跟着换</b>。", BL, "NVLink 5"),
         tpu=_cell_text("9,216 颗", "3D 环面一路铺到整个 pod，"
                        "<b>全程同一套 ICI</b>。这是 §8 的落点。", GN, "ICI 4.0")),
]


# ══════════════════════════════════════════════════════════════════════
def _side(f):
    f.rect(RX, TB, RW, HDR + NROW * RH, "#fff", RD, 1.8, 10)
    f.t(RX + 14, TB + 24, "看这张表要看的是空格", "sec", RD)
    para(f, RX + 14, TB + 46, RW - 28,
         "七行里有<b>两行</b> TPU 那一栏是红的。而这两行不是随便哪两行 —— "
         "它们恰好是 GPU 用来<r>在运行时藏住延迟</r>的那两层。", "xs", 15)

    blocks = [
        (RD, "空的第一格：没有「一组线程」",
         "GPU 的 block 是个<b>运行时</b>概念：有几个 block、落在哪个 SM 上，"
         "启动时才知道。TPU 没有这个中间层，所有归属在编译期就写死在指令里。"),
        (RD, "空的第二格：没有「换一个跑」",
         "GPU 一个 warp 卡住就换下一个，这需要同时驻留几十份上下文 —— "
         "那 256 KB 寄存器堆主要就是为它准备的。TPU 不留这些上下文，"
         "所以也省下了那片面积（见 §2）。"),
        (GN, "于是 TPU 的层级全在描述「形状」",
         "lane、sublane、tile、slice —— 每一层说的都是<b>数据长什么样</b>，"
         "而不是<b>谁在执行</b>。这就是为什么 TPU 编程里你调的是分片策略，"
         "而 CUDA 编程里你调的是线程组织。<br>"
         "顺带解释了一件常被问到的事：为什么 TPU 上「跑一个不规则的算法」这么别扭 —— "
         "不是编译器不肯，是<r>硬件层级里根本没有一个能承载「不规则」的单位</r>。"
         .replace("<br>", "")),
    ]
    y = TB + 92
    for c, ttl, body in blocks:
        f.rect(RX + 14, y, RW - 28, 172, FILL[c], c, 1.4, 8)
        f.t(RX + 26, y + 24, ttl, "lbl", c)
        para(f, RX + 26, y + 44, RW - 52, body, "xxs", 14)
        y += 182


def _band(f):
    f.rect(20, BAND_Y, 1360, 132, FILL[BL], BL, 1.6, 10)
    f.t(34, BAND_Y + 26, "一句话总结这张表", "sec", BL)
    para(f, 34, BAND_Y + 50, 1330,
         "<b>GPU 的层级是「执行体的层级」，TPU 的层级是「数据形状的层级」。</b>"
         "thread、warp、block 说的都是谁在跑；lane、sublane、tile、slice 说的都是数据被切成什么样。", "xs", 18)
    para(f, 34, BAND_Y + 88, 1330,
         "这个区别有个非常实际的后果：<b>GPU 的性能问题多半出在「占用率」上</b>"
         "（同时驻留的 warp 够不够多，能不能把延迟盖住）；"
         "<b>TPU 的性能问题多半出在「形状」上</b>（矩阵维度对不对齐、切片切得均不均匀）。"
         "两边的调优直觉<r>不能互相搬运</r> —— 这也是为什么 §4 那个 <code>head_dim=128</code> "
         "打 TPU 却不打 GPU。", "xs", 18)


if __name__ == "__main__":
    import io, sys
    io.open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/t3.svg", "w",
            encoding="utf-8").write(build())
    print("ok", H)
