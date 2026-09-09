# -*- coding: utf-8 -*-
r"""外传 图 X-7 · **SparseCore 拆开看** —— 它有两个职责，我第一版只写了一个。

════════════════════════════════════════════════════════════════════
⛔⛔ 这张图我推翻重写过一次，而且是被现场当场纠正的
════════════════════════════════════════════════════════════════════
第一版我写的是「扩散模型用不到它，它是给推荐系统那类负载准备的」。
现场原话：

    「别瞎搞啊，那个 SparseCore 就是用于这个 communication offloading，
      只要有 all-reduce、all-gather 这些东西就都需要。
      那扩散模型只要是跨卡通信的话，也是有不少 all-gather 的。」

⭐ 去核了，**这条纠正是对的，而且官方三处都写得很明确**：

  · Cloud TPU 性能指南：「**Overlap communication with computation**：
    把 **all-reduce 这类集合通信卸载到 SparseCore**。这些操作**不占 MXU**，
    可以在 SparseCore 上执行，**同时 TensorCore 继续算**。」
  · 官方《TPU7x (Ironwood) 性能优化》：「在 TPU7x 上重叠通信与计算的
    **主要机制**叫 **SparseCore Collective Offloading**。」
  · Google Cloud 博客《Training large models on Ironwood TPUs》：
    「用特定的 **XLA flag**，可以把 **All-Gather 与 Reduce-Scatter
    直接卸载到 SparseCore**，让 TensorCore 专心做主计算，通信并行执行。」

════════════════════════════════════════════════════════════════════
⭐⭐ 所以 SparseCore 是**两个职责**，不是一个
════════════════════════════════════════════════════════════════════
① **稀疏 / 大表** ——&nbsp;架构文档里那个 "a primary use case"（推荐模型的 embedding）
② **集合通信卸载** ——&nbsp;all-reduce / all-gather / reduce-scatter 卸到它上面跑，
   **腾出 TensorCore 专心算**

⭐ 于是对扩散模型的结论**整个反过来**：
   **只要跨卡，它就在干活。** 扩散不是「用不到它」——&nbsp;
   是它用的是第 ② 条那条路（通信），不是第 ① 条（embedding）。

════════════════════════════════════════════════════════════════════
⛔ 我原来错在哪 ——&nbsp;这个错法本身值得记
════════════════════════════════════════════════════════════════════
我查了架构文档，看到 "a primary use case is ... recommendation models"，
**就把「主用途」当成了「用途的全集」**，然后顺着推出「扩散用不到」。
⭐ 判据：**"a primary use case" 这种措辞是在告诉你「还有别的」** ——&nbsp;
  它是一个明确的**不完全枚举**信号，而我把它读成了定义。
⚠️ 更该警惕的是：我当时**还给这个错误配了一条像模像样的推导链**
  （没有大表 → 没有稀疏聚合 → 基本闲着）。
  **推导链是对的，前提漏了一半，于是整条链推向了错的地方。**

⚠️ 口径：上面那批「集合通信卸载」的材料**主要是 v7x / 第四代 SparseCore 语境**
  （Trillium 的是**第三代**）。v6e 上这条路的成熟度我们没实测 ——&nbsp;
  图上如实标出来，不含糊过去。
"""
from topic03_draw import (Fig, wpx, _sz, LINE, LINE2,
                          BL, OR, GR, RD, GY, GY2, PU, CY, INK)

W = 1400


def main():
    f = Fig(W, "TPU v6e 上每颗芯片有 2 个 SparseCore，每个含 16 个子核、"
               "每个子核 8 条 lane 和 256 KiB 本地内存。它有两个职责："
               "一是加速稀疏运算与大表 embedding，二是集合通信卸载 ——"
               "把 all-reduce、all-gather、reduce-scatter 从 TensorCore 手里接过去，"
               "不占矩阵单元，与计算并行执行。所以只要跨卡，它就在干活")
    f.marks = set()
    y = f.header(
        'SparseCore 拆开看 ——&#160;'
        '<tspan font-weight="700">它有两个职责，跨卡的时候一直在干活</tspan>',
        '⭐ 一颗 v6e 上除了那个大 TensorCore，还蹲着两个小协处理器。'
        '它们不做矩阵乘 ——&#160;<tspan font-weight="700">一是在大表里到处乱查，'
        '二是把集合通信从 TensorCore 手里接过去</tspan>。',
        [(PU, "职责①：稀疏 / 大表"), (RD, "职责②：集合通信卸载"),
         (BL, "TensorCore：稠密矩阵乘")])

    top = y + 8
    # ⛔ 三栏排过一版，护栏一行一行地拦 ——&nbsp;那不是文案太长，是**版面本身
    #   放不下中文**（三栏各 430px，一行只够 18 个汉字）。⭐ 判据：护栏连续
    #   拦同一类内容时，该改的是布局，不是一句句去砍字。改成两栏。
    LW = 860
    RW = W - LW - 30
    RX = LW + 30
    # ⛔ 面板高度**按内容加出来**，第三次栽在这上面了（X-2 一次、这里一次）。
    #   左栏：标题栏 30 ＋ TensorCore 卡 72 ＋ 一行说明 20 ＋ SparseCore 块 96
    #        ＋ 间距 24 ＋ 四行注解 4×19 ＋ 下沿留白 10
    #   ⭐ 每次都想「大概这么高吧」，而每次都少 30–50px。
    #     判据：**只要面板里有变长的正文，高度就必须是算出来的。**
    PH = 30 + 18 + 54 + 20 + 12 + 96 + 24 + 4 * 19 + 12 + 24

    # ══ 左：它在哪儿 ＋ 里面是什么 ══
    ly = f.panel(0, top, LW, PH, "它在芯片的哪儿，里面是什么", PU,
                 tag="官方规格表 ＋ JAX 公开源码")
    f.cell(20, ly + 18, LW - 40, 54, "TensorCore × 1",
           "2 个 MXU（256×256）＋ 向量单元 ＋ 标量单元　——　稠密算力全在这儿",
           BL, "#fff", 13)
    f.t(20, ly + 92, "同一颗芯片上并排还蹲着两个：", GY, size=_sz(11), w=LW - 40)
    sw, sh, sp = 44.0, 28.0, 7.0
    for k in range(2):
        bx = 20 + k * ((LW - 52) / 2 + 12)
        bw = (LW - 52) / 2
        f.box(bx, ly + 104, bw, 96, "#fff", PU, 7)
        f.t(bx + 14, ly + 124, "SparseCore %d" % k, "#681da8",
            bold=True, size=_sz(12))
        for c in range(8):
            f.box(bx + 14 + c * (sw + sp), ly + 136, sw, sh, "#fff", PU, 3, 0.8)
            f.t(bx + 14 + c * (sw + sp) + sw / 2, ly + 154, "8 lane",
                "#681da8", size=_sz(11), anchor="middle")
        f.t(bx + 14, ly + 188, "…共 16 个子核，每个自带 256 KiB",
            GY2, size=_sz(11), w=bw - 28)
    f.lines(20, ly + 224, LW - 40, [
        "⭐ <tspan font-weight=\"700\">v6e 每 chip 2 个</tspan>；"
        "v5p 与 TPU7x 是 4 个 ——&#160;官方原文",
        "⭐ 跟 TensorCore 的<tspan font-weight=\"700\">粒度正相反</tspan>："
        "MXU 一次吞一整块 256×256 的方阵，",
        "　 SparseCore 是很多条窄 lane <tspan font-weight=\"700\">各查各的</tspan>"
        "——&#160;它是为「散着取」造的",
        "⚠️ 怎么用它由 <tspan font-weight=\"700\">XLA flag</tspan> 控制，"
        "不是写模型时决定的；内部数据流本讲不展开",
    ], size=11, lh=19, fill=GY)

    # ══ 右：它的第二个职责（第一版整个漏了） ══
    ry = f.panel(RX, top, RW, PH, "它的两个职责", RD, tag="官方原文")
    f.t(RX + 16, ry + 24, "① 稀疏 / 大表", "#681da8", bold=True, size=_sz(12))
    f.lines(RX + 16, ry + 46, RW - 32, [
        "「加速使用稀疏运算的模型；主用途是",
        "  加速重 embedding 的推荐模型」",
    ], size=11, lh=19, fill=GY)
    f.line(RX + 16, ry + 92, RX + RW - 16, ry + 92, LINE, 1, arrow=False)
    f.t(RX + 16, ry + 116, "② 集合通信卸载", "#a50e0e", bold=True, size=_sz(12))
    f.lines(RX + 16, ry + 138, RW - 32, [
        "「把 all-reduce 这类集合通信卸载到",
        "  SparseCore。这些操作<tspan font-weight=\"700\">不占 MXU</tspan>，",
        "  可以在它上面执行，<tspan font-weight=\"700\">同时 TensorCore</tspan>",
        "  <tspan font-weight=\"700\">继续算</tspan>」——&#160;官方性能指南",
        "",
        "All-Gather 与 Reduce-Scatter 也能卸，",
        "由 <tspan font-weight=\"700\">XLA flag</tspan> 开关",
    ], size=11, lh=19, fill=GY)
    f.t(RX + 16, ry + 292,
        "⭐ <tspan font-weight=\"700\">只要跨卡，它就在干活</tspan>",
        "#a50e0e", size=_sz(12), w=RW - 32)

    y = top + PH + 22

    y = f.band(y, "bad",
               "⛔ 这张图我推翻重写过一次 ——&#160;而这个错法值得原样讲给学员听",
               ['<tspan font-weight="700">第一版我写的是「扩散模型用不到它」。那是错的。</tspan>'
                '我查架构文档看到 <tspan font-weight="700">a primary use case</tspan> 是推荐模型，'
                '就<tspan font-weight="700">把「主用途」当成了「用途的全集」</tspan>。',
                '⭐ 判据：<tspan font-weight="700">「a primary use case」这种措辞'
                '本身就在告诉你「还有别的」</tspan> ——&#160;'
                '它是一个明确的<tspan font-weight="700">不完全枚举</tspan>信号，而我把它读成了定义。',
                '⚠️ 更该警惕的是：我当时<tspan font-weight="700">还给这个错误配了一条'
                '像模像样的推导链</tspan>（没有大表 → 没有稀疏聚合 → 基本闲着）。'
                '<tspan font-weight="700">链子是对的，前提漏了一半，于是整条链推向了错的地方。</tspan>'])

    y = f.band(y + 14, "info",
               "对扩散来说，结论整个反过来：只要跨卡，它就在干活",
               ['扩散<tspan font-weight="700">确实没有大 embedding 表</tspan>，'
                '所以它走的不是第 ① 条路。'
                '但只要模型要切开、要跨卡，<tspan font-weight="700">'
                'all-gather / reduce-scatter 就一大堆</tspan> ——&#160;第 ② 条路它走得很勤。',
                '⭐ 而这正是这颗协处理器最值钱的地方：'
                '<tspan font-weight="700">通信不占 MXU，可以跟计算真正并行</tspan>。'
                '——&#160;对一颗「算力强、显存弱」的芯片，'
                '<tspan font-weight="700">把通信从关键路径上挪开是格外划算的</tspan>。',
                '⚠️ 口径：上面那批集合通信卸载的材料'
                '<tspan font-weight="700">主要是 v7x / 第四代 SparseCore 语境</tspan>'
                '（Trillium 是第三代）。<tspan font-weight="700">v6e 上这条路的成熟度'
                '我们没有实测</tspan> ——&#160;这里如实标出，不含糊过去。'])

    y = f.src(y + 18,
              '职责① 与「v6e 每 chip 2 个」——&#160;官方《TPU architecture》；'
              '职责②「把 all-reduce 卸到 SparseCore，不占 MXU，与 TensorCore 并行」'
              '——&#160;官方《Cloud TPU 性能指南》',
              '16 subcore × 8 lane、每 subcore VMEM 256 KiB ——&#160;'
              'JAX 公开源码 tpu_info.py 的 TPU_V6E 分支。'
              '⚠️ 集合通信卸载那批材料以 v7x / 第四代 SparseCore 为主，'
              'Trillium 是第三代，v6e 上的成熟度本课未实测。')
    f.save("figx-7.svg", y + 6)


main()
