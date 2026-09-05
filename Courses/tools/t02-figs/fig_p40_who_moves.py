# -*- coding: utf-8 -*-
"""图 P-40 —— 同一个 FlashAttention，两边是两种性质的工作：谁来安排搬运。

**这是 3.6 的收尾，也是第 5 节那条主线第一次落到一个具体 kernel 上。**
P-38 说清了要解决什么（128 GiB 全在路上），P-39 把两条路一站一站并排走完。
剩下最后一个问题：<b>同样是「把 tile 搬进片上暂存」，这件事是谁写的？</b>

**GPU 侧：kernel 作者亲手搬。** 块开多大、什么时候发 <code>cp.async</code>、
双缓冲怎么错开、哪一组 warp 专管搬运、屏障放在哪 —— 全在 kernel 源码里。
<b>硬件那层缓存在这个 kernel 里帮不上忙</b>：访问模式完全可预测，
「猜下次要什么」这件事没有价值，所以 FlashAttention 把复用显式搬到
软件管的那半边共享内存里。

**TPU 侧：声明块的形状，编译器排 DMA。** Pallas 里写 <code>BlockSpec</code>
说清每块多大、第 i 块取哪一段，剩下的双缓冲与 DMA 时序由编译器在编译期定死。
<b>没有缓存可以绕开 —— 因为本来就没有那一层。</b>

**⚠️ 这里最容易说过头，图上专门留了一格。** 「FlashAttention 绕开缓存」
这句话半对：<b>L2 绕不开</b>，所有 HBM 访问都过它。绕开的是<b>对缓存的依赖</b> ——
命中与否不再决定性能。能真正选的只有 L1（<code>cp.async</code> 的
<code>.ca</code> / <code>.cg</code>）。

**⚠️ 出处口径。** <code>cp.async</code> 的 <code>.ca</code>／<code>.cg</code>
出自 PTX ISA；<code>BlockSpec</code>／<code>index_map</code>／grid 出自 Pallas 文档；
Splash Attention 的块级掩码出自其公开实现。
<b>「共享内存 ≤ 227 KiB／线程块」沿用图 G-6 的标注。</b>
"""
from common import (Fig, para, wrap, plain, BL, GN, RD, YL, PU, TL, INK,
                    SUB, GREY, FILL)

W = 1400

HEAD_Y, HEAD_H = 84, 94
# PAN_H 的账：左栏 = 64（标题）+ 209（「你要亲手写的」七行）+ 14 + 74
# （「硬件替你兜的」）＝ 361，再留 15 底边。左栏多一行就要把这个数加 17。
PAN_Y, PAN_H = HEAD_Y + HEAD_H + 22, 376
OVER_Y, OVER_H = PAN_Y + PAN_H + 22, 104
LAND_Y, LAND_H = OVER_Y + OVER_H + 22, 118
SRC_Y, SRC_H = LAND_Y + LAND_H + 22, 92
H = SRC_Y + SRC_H + 20

L_X, L_W = 20, 674
R_X, R_W = 706, 674

# (文字, 占几行)。行数写死是为了能提前算出框高 —— 改文字记得同步改数字。
GPU_MINE = [
    ("<b>块开多大</b>：算到 Q / K / V 块加中间量刚好塞进 <b>≤ 227 KiB</b>", 1),
    ("<b>发搬运指令</b>：<code>cp.async</code> ／ TMA 把下一块从 HBM 拉进共享内存", 1),
    ("<b>双缓冲</b>：搬下一块的同时算这一块，两个缓冲区自己轮换", 1),
    # ⭐ 2026-09-03 补。原来只写了 __syncthreads() —— 那是 **Ampere 的写法**。
    #    Hopper 起真正在用的是 mbarrier ＋ warp 分工，而这两条恰恰**加强**本图论点：
    #    左栏又长了两行，右栏一个字没变。（缘由见 §3.3「三个名字的来历」那个折叠。）
    # ⚠️ 这一条实测 560 / 可用 598，只剩 38 px 余量 —— 再加两个字就翻行。
    #    行数写错不会报错、只会静默丢字，所以 _list 里加了逐条对账的断言。
    ("<b>warp 分工</b>：一组 warp 专发 TMA 当生产者，另几组专发 "
     "<code>wgmma</code> 当消费者 <g>—— 搬的和算的分开，流水线才叠得起来</g>", 1),
    ("<b>对齐</b>：Ampere 上是 <code>__syncthreads()</code>；"
     "<b>Hopper 起换成 <code>mbarrier</code> 异步屏障</b> "
     "<g>—— 搬完第 k 块就报到，算的那组等到就开工，同时搬 k＋1</g>", 2),
    ("<b>要不要占 L1</b>：<code>cp.async</code> 的 <code>.ca</code> 走 L1＋L2，"
     "<code>.cg</code> 只走 L2", 1),
]
GPU_FREE = [
    ("<b>warp 调度器每周期挑一个数据到位的 warp。</b>"
     "<g>你没搬完的那段延迟，有机会被别的 warp 的计算盖住 —— "
     "所以 GPU 上「搬慢一点」往往不是致命伤。</g>", 2),
]
TPU_MINE = [
    ("<b><code>BlockSpec</code></b>：每块多大，以及第 i 块取原数组的哪一段"
     "（<code>index_map</code>）", 1),
    ("<b>grid</b>：一共几块，循环怎么套", 1),
]
TPU_FREE = [
    ("<b>DMA 什么时候发、双缓冲怎么错开、VMEM 怎么分</b> —— "
     "全在<b>编译期</b>排好，源码里没有对应的那几行。", 1),
    ("<b>顺带白拿一样</b>：Splash Attention 支持<b>块级掩码</b>。"
     "<g>causal ／ sliding window 下整块用不上的，直接不算 —— "
     "省的不是搬运，是 FLOPs。</g>", 2),
]


def build():
    f = Fig(W, H, "同一个 FlashAttention：GPU 上由 kernel 作者亲手安排搬运与同步，"
                  "TPU 上由作者声明块形状、编译器在编译期排定 DMA。"
                  "数学与 FLOPs 完全相同，差别在于安排搬运的机制在运行时还是编译期")
    f.title("同一个算法，<tspan font-weight=\"700\">两种性质的工作</tspan>"
            "　—— 一边亲手搬，一边只声明形状", "3.6 的收尾")
    f.legend([(BL, "GPU：写在 kernel 里"), (GN, "TPU：写在声明里"),
              (RD, "容易说过头的地方")])
    _head(f)
    _panel(f, L_X, L_W, BL, "GPU　·　手写 CUDA kernel",
           "复用被显式搬进软件管的那半边共享内存",
           "你要亲手写的", GPU_MINE, "硬件替你兜的", GPU_FREE)
    _panel(f, R_X, R_W, GN, "TPU　·　Pallas ／ Splash Attention",
           "只声明块的形状，搬运的时序交给编译器",
           "你要写的", TPU_MINE, "编译器替你做的", TPU_FREE)
    _over(f)
    _land(f)
    _src(f)
    return f.out()


# ══════════════════════════════════════════════════════════════════════
# 先把「一样的部分」钉死，否则下面两栏的差别会被读成「算的东西不一样」。
def _head(f):
    f.rect(20, HEAD_Y, 1360, HEAD_H, FILL[PU], PU, 1.8, 10)
    f.t(38, HEAD_Y + 26, "先说两边完全一样的部分 —— 不一样的只有最后一行", "sec", PU)
    # ⛔ 2026-09-05：「FLOPs 一个都不差」原来是无条件说的 —— **只有前向成立**。
    #    反向要重算 S 和 P，是 7 次矩阵乘对 6 次（§2.3 折叠里有这笔账）。
    #    这门课讲的是训练，反向不能默认省略。
    same = [("数学", "online softmax：逐块更新最大值与分母，"
                     "新块来了先把旧结果按新最大值重标定一次"),
            ("FLOPs（前向）", "一个都不差，与朴素写法相同"
                              "<g>　·　反向要重算，7 次对 6 次（2.3）</g>"),
            ("结果", "与不分块的注意力<b>数值等价</b>")]
    for i, (k, v) in enumerate(same):
        x = 38 + i * 372
        f.t(x, HEAD_Y + 54, k, "lbl", SUB)
        para(f, x, HEAD_Y + 76, 352, v, "xs", 16, max_lines=2)
    f.rect(1160, HEAD_Y + 38, 202, 44, "#fff", RD, 1.5, 6)
    para(f, 1172, HEAD_Y + 58, 178,
         "<r>不一样的是：谁来安排搬运。</r>", "xs", 15, max_lines=2)


# ══════════════════════════════════════════════════════════════════════
def _list(f, x, y, w, c, head, items, headc=None):
    """画一个「谁负责」的小框，返回框底 y。高度按 items 里写死的行数算。"""
    h = 30 + sum(n * 17 + 10 for _, n in items)
    f.rect(x, y, w, h, "#fff", c, 1.6, 7)
    f.rect(x, y, w, 24, FILL[c], rx=7)
    f.rect(x, y + 16, w, 8, FILL[c], rx=0)
    f.t(x + 12, y + 17, head, "lbl", headc or c)
    yy = y + 40
    for txt, n in items:
        # ⛔ items 里那个行数是**手写**的，而 para 的 max_lines 是**静默截断**：
        #    写少了多出来的行直接丢（不报错、不留省略号），写多了则凭空留一道空行。
        #    两种都不会让脚本失败，所以这里逐条对账，要求分毫不差。
        #    ⚠️ 别放宽成 `<=`：写多了同样是错，而且更难看出来。
        need = len(wrap(txt, w - 40, 11))
        assert need == n, ("「%s…」实际 %d 行，items 里写的是 %d —— "
                           "改成 %d，并把 PAN_H 加减 %d"
                           % (plain(txt)[:16], need, n, need, abs(need - n) * 17))
        f.rect(x + 12, yy - 9, 4, n * 17 - 2, c, rx=2)
        para(f, x + 24, yy, w - 40, txt, "xs", 17, max_lines=n)
        yy += n * 17 + 10
    return y + h


def _panel(f, x, w, c, ttl, sub, h1, l1, h2, l2):
    f.rect(x, PAN_Y, w, PAN_H, FILL[c], c, 1.8, 10)
    f.t(x + 18, PAN_Y + 28, ttl, "sec", c)
    f.t(x + 18, PAN_Y + 50, sub, "xxs", SUB)
    y = _list(f, x + 18, PAN_Y + 64, w - 36, c, h1, l1, RD)
    y = _list(f, x + 18, y + 14, w - 36, GREY, h2, l2, SUB)
    # 右栏天然比左栏矮一截 —— 与其留白，不如把这个高度差本身讲出来。
    if c is GN:
        f.rect(x + 18, y + 14, w - 36, PAN_Y + PAN_H - 18 - (y + 14),
               FILL[YL], YL, 1.5, 7)
        para(f, x + 30, y + 36, w - 60,
             "⚖️ <b>右栏比左栏短，这件事本身就是结论</b>："
             "<r>少写的那四条没有消失，只是挪进了编译器。</r>", "xs", 16,
             max_lines=2)


# ══════════════════════════════════════════════════════════════════════
# ⚠️ 别删。「FlashAttention 绕开缓存」是这一节最容易讲过头的一句。
def _over(f):
    f.rect(20, OVER_Y, 1360, OVER_H, FILL[RD], RD, 1.8, 10)
    f.t(38, OVER_Y + 26, "⚠️ 这句话只对一半：「FlashAttention 绕开了缓存」", "sec", RD)
    y = para(f, 38, OVER_Y + 50, 1324,
             "<b>L2 绕不开</b> —— 所有 HBM 访问都要过它，没有哪条指令能跳过。"
             "<b>真正能选的只有 L1</b>：<code>cp.async</code> 的 <code>.ca</code> "
             "走 L1＋L2，<code>.cg</code> 只走 L2。", "xs", 18)
    para(f, 38, y + 2, 1324,
         "<r>绕开的不是缓存本身，是对缓存的依赖。</r>"
         "<b>复用被显式安排进共享内存之后，命中率高不高就不再决定性能了</b> —— "
         "这才是那句话想说的意思。", "xs", 18)


# ══════════════════════════════════════════════════════════════════════
def _land(f):
    f.rect(20, LAND_Y, 1360, LAND_H, FILL[BL], BL, 1.8, 10)
    f.t(38, LAND_Y + 28, "⭐ 落点：第 5 节那条主线，在一个具体 kernel 上长这样",
        "sec", BL)
    y = para(f, 38, LAND_Y + 54, 1324,
             "<b>GPU 上，FlashAttention 是一段和硬件的猜测协商的代码</b>："
             "硬件准备了缓存和 warp 调度器来兜住不确定性，而这个 kernel "
             "<b>不需要那种兜底</b>，于是自己接管了搬运。"
             "<b>TPU 上没有可绕开的东西</b>，于是只剩下声明形状。", "xs", 19)
    para(f, 38, y + 2, 1324,
         "<b>代价也对称</b>：<r>GPU 的手写 kernel 换一代硬件要重调；"
         "TPU 的声明写错了，编译器排出来就是真的停住，没有第二个任务顶班。</r>"
         "<g>同一个算法，两种工程性质 —— 这就是运行时与编译期的分工，落在一个 kernel 上。</g>",
         "xs", 19)


# ══════════════════════════════════════════════════════════════════════
def _src(f):
    f.rect(20, SRC_Y, 1360, SRC_H, "#fff", GREY, 1.4, 10)
    f.t(38, SRC_Y + 26, "⚠️ 出处分层", "sec")
    y = para(f, 38, SRC_Y + 50, 1324,
             "<b>查到的</b>：<code>cp.async</code> 的 <code>.ca</code> ／ "
             "<code>.cg</code> 缓存行为出自 PTX ISA；"
             "<code>BlockSpec</code> ／ <code>index_map</code> ／ grid 出自 Pallas 文档；"
             "Splash Attention 的块级掩码出自其公开实现。", "xs", 18)
    para(f, 38, y + 2, 1324,
         "<b>沿用本课已有标注</b>：共享内存 ≤ 227 KiB / 线程块出自图 G-6。"
         "<g>「FlashAttention 把复用显式放进共享内存」是公开实现的通行做法，"
         "不是某一份文档里的原话。</g>", "xxs", 17)


if __name__ == "__main__":
    import io
    io.open("out/fig_p40_who_moves.svg", "w", encoding="utf-8").write(build())
    print("ok fig_p40_who_moves")
