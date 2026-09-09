# -*- coding: utf-8 -*-
r"""外传 图 X-7 · **SparseCore 拆开看** —— 这一讲用不上它，但你该知道它在。

════════════════════════════════════════════════════════════════════
⛔⛔ 先修一句我自己写过头的话
════════════════════════════════════════════════════════════════════
X-2 的初版上我写了「**扩散模型用不到它**」。
**这句比出处强。** 官方原文只说到这里：

    「SparseCores are dataflow processors that accelerate models using
      sparse operations. **A primary use case** is accelerating
      recommendation models, which rely heavily on embeddings.」
      ——&nbsp;Cloud TPU 官方《TPU architecture》

⭐ 官方说的是 **primary use case**（主用途），不是 only use case。
  「扩散用不到」是**我的推论**，不是官方结论 ——&nbsp;
  所以图上必须把推导链摆出来，而不是当成事实陈述：

    扩散模型这条链路上没有大 embedding 表（文本编码器那点词表是小头），
    也没有 embedding 反向那种稀疏梯度聚合
        → 这颗为「大表随机查、稀疏聚合」造的协处理器，在这条链路上基本闲着

  ⚠️ **「基本闲着」和「用不到」不是一回事。** XLA 是否把某些通信卸到
    SparseCore 上，由 XLA flag 控制（官方同页），我们没有实测，不下断言。

════════════════════════════════════════════════════════════════════
📌 图上的数（全部有出处）
════════════════════════════════════════════════════════════════════
· **每 chip 2 个 SparseCore**（v6e）——&nbsp;官方《TPU architecture》原文；
  同页写明 v5p 和 TPU7x 是 4 个
· 每个 SparseCore：**16 个 subcore**、每 subcore **8 lane**、
  每 subcore **VMEM 256 KiB** ——&nbsp;JAX 公开源码 tpu_info.py 的 TPU_V6E 分支
· 「怎么用它」由 **XLA flag** 控制 ——&nbsp;官方同页

⛔ 这张图**不讲**它内部的数据流细节（gather / scatter 怎么排）。
  L100 只需要三件事：它在、它为什么存在、这一讲为什么不靠它。
"""
from topic03_draw import (Fig, wpx, _sz, LINE, LINE2,
                          BL, OR, GR, RD, GY, GY2, PU, CY, INK)

W = 1400


def main():
    f = Fig(W, "TPU v6e 上每颗芯片有 2 个 SparseCore，每个含 16 个子核、"
               "每个子核 8 条 lane 和 256 KiB 本地内存。它是为重 embedding 的"
               "推荐模型造的；扩散模型这条链路上没有大 embedding 表，"
               "所以它基本闲着")
    f.marks = set()
    y = f.header(
        'SparseCore 拆开看 ——&#160;'
        '<tspan font-weight="700">这一讲用不上它，但你该知道它在</tspan>',
        '⭐ 一颗 v6e 上除了那个大 TensorCore，还蹲着两个小协处理器。'
        '它们不做矩阵乘，<tspan font-weight="700">做的是「在一张很大的表里到处乱查」</tspan>。',
        [(PU, "SparseCore：稀疏 / 大表"), (BL, "TensorCore：稠密矩阵乘"),
         (GY2, "灰＝本讲不展开")])

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
    PH = 30 + 18 + 54 + 20 + 12 + 96 + 24 + 4 * 19 + 12

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

    # ══ 右：这一讲为什么不靠它 ══
    ry = f.panel(RX, top, RW, PH, "这一讲为什么不靠它", GY2,
                 tag="⚠️ 推论，非官方结论")
    f.t(RX + 16, ry + 24, "官方原话怎么说它的用途", INK, bold=True, size=_sz(12))
    f.lines(RX + 16, ry + 46, RW - 32, [
        "「加速使用<tspan font-weight=\"700\">稀疏运算</tspan>的模型；",
        "<tspan font-weight=\"700\">主用途</tspan>是加速重 embedding 的推荐模型」",
    ], size=11, lh=19, fill=GY)
    f.line(RX + 16, ry + 92, RX + RW - 16, ry + 92, LINE, 1, arrow=False)
    f.t(RX + 16, ry + 116, "那扩散这条链路上有什么", INK, bold=True, size=_sz(12))
    f.lines(RX + 16, ry + 138, RW - 32, [
        "· 没有大 embedding 表",
        "· 没有稀疏梯度聚合",
        "· 全是稠密的大矩阵乘",
    ], size=11, lh=20, fill=GY)
    f.t(RX + 16, ry + 212,
        "→ <tspan font-weight=\"700\">这颗协处理器基本闲着</tspan>",
        "#b06000", size=_sz(12), w=RW - 32)
    f.t(RX + 16, ry + 240, "⛔ 但「闲着」不等于「用不到」",
        "#a50e0e", size=_sz(11), w=RW - 32)
    f.t(RX + 16, ry + 262, "——&#160;我们没实测，所以不下断言",
        GY2, size=_sz(11), w=RW - 32)

    y = top + PH + 22

    y = f.band(y, "warn",
               "⛔ 这里我改过一次口 ——&#160;而这个改口本身值得讲给学员听",
               ['这张图的<tspan font-weight="700">初版</tspan>我写的是'
                '「扩散模型用不到它」。<tspan font-weight="700">那句比出处强。</tspan>'
                '官方原文说的是 <tspan font-weight="700">a primary use case</tspan>'
                '（主用途），不是 only use case。',
                '⭐ 所以现在图上摆的是<tspan font-weight="700">推导链</tspan>，'
                '不是结论：没有大 embedding 表 → 没有稀疏聚合 → '
                '这颗为「散着取」造的东西在这条链路上基本闲着。'
                '<tspan font-weight="700">读者可以顺着链子自己判断，也可以反驳。</tspan>',
                '⚠️ 而且 XLA 是否把某些通信卸到它上面，由 XLA flag 控制'
                '（官方同页）——&#160;'
                '<tspan font-weight="700">我们没有实测，所以不下断言。</tspan>'])

    y = f.src(y + 18,
              '「加速使用稀疏运算的模型；主用途是加速重 embedding 的推荐模型」、'
              '「v6e 每 chip 2 个，v5p 与 TPU7x 4 个」、「由 XLA flag 控制」'
              '——&#160;Cloud TPU 官方《TPU architecture》',
              '16 subcore × 8 lane、每 subcore VMEM 256 KiB ——&#160;'
              'JAX 公开源码 tpu_info.py 的 TPU_V6E 分支。'
              '⚠️ 右栏那条「基本闲着」是本讲的推论，图上已标明。')
    f.save("figx-7.svg", y + 6)


main()
