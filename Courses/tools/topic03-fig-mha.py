# -*- coding: utf-8 -*-
r"""专题三 · §一「MHA」的三张图。

⭐ 承接 §零：RNN 疼在三处（算不快、记不住、装不下）。
   这一节讲 2017 年那一刀 ——&nbsp;**把循环整个拿掉，只留下那个补丁**。

⭐⭐ **这三张图各答一个问题：**
     · `fig3-mha-swap`  →&nbsp;**换掉了什么**（一根横箭头 → 一张全连接图）
     · `fig3-mha-qkv`   →&nbsp;**一层里到底在算什么**（检索三件套 ＋ 为什么除 √d）
     · `fig3-mha-heads` →&nbsp;**多头在多什么**（不是堆算力，是不让平均把它糊掉）

════════════════════════════════════════════════════════════════════
📌 出处（2026-09-08 一手核过，全部来自论文原文）
════════════════════════════════════════════════════════════════════

· **Vaswani et al. 2017**《Attention Is All You Need》(arXiv 1706.03762)

  ① **注意力的定义就是检索**，§3.2 原话：「mapping a **query** and a set of
     **key-value pairs** to an output … The output is computed as a
     **weighted sum of the values**, where the weight assigned to each value is
     computed by a **compatibility function of the query with the corresponding
     key**.」⭐ 所以「query 问、key 挂牌、value 是货」不是我们编的比喻，
     是论文自己的措辞。

  ② **为什么要除 √d_k**，§3.2.1 原话：「for large values of d_k, the dot products
     **grow large in magnitude, pushing the softmax function into regions where
     it has extremely small gradients**. To counteract this effect, we scale the
     dot products by 1/√d_k.」
     ⭐⭐ 而**推导链在脚注 4 里**，可以完整讲给学生：
     假设 q、k 各维独立、均值 0、方差 1，则 q·k ＝ Σ q_i k_i **均值 0、方差 d_k**
     ——&nbsp;标准差就是 **√d_k**。除掉它，方差回到 1。
     ⛔ 别只说「防止 softmax 饱和」——&nbsp;那是结论，不是理由。

  ③ **多头在多什么**，§3.2.2 原话：「Multi-head attention allows the model to
     **jointly attend to information from different representation subspaces at
     different positions**. **With a single attention head, averaging inhibits
     this.**」⭐ 最后那半句才是重点：**单头不是「不够用」，是会把不同的关注平均掉。**
     基础模型的配置也在这一节：**h ＝ 8，d_k ＝ d_v ＝ d_model / h ＝ 64**
     ——&nbsp;⭐ 所以 8 × 64 ＝ 512 ＝ d_model，**总维度没变，只是切开了**。

  ④ **因果遮罩**，§3.2.3：「masking out (setting to **−∞**) all values in the
     input of the softmax which correspond to illegal connections」。

  ⑤ **表 1**（§零已经用过）：自注意力串行步数 O(1)、最长路径 O(1)；
     循环层两项都是 O(n)。⭐ 这一节就是那张表的「怎么做到的」。

· **Shazeer 2019**《Fast Transformer Decoding: One Write-Head is All You Need》
  (arXiv 1911.02150) ——&nbsp;⭐⭐ **KV cache 被点名成瓶颈的出处**，摘要原话：
  「training these layers is generally fast and simple, due to parallelizability
  across the length of the sequence, **incremental inference (where such
  parallelization is impossible) is often slow, due to the memory-bandwidth cost
  of repeatedly loading the large "keys" and "values" tensors**.」
  ⭐ 这句话把 §零 图三 Ⓒ 那一行**从我们的推论变成了原文**：
    2017 年造出这个形状，**2019 年就有人把它命名成问题了**，
    而那篇给出的解法（MQA）正是本课模型表里的第二行。


⛔⛔ **一个记号两种含义 ——&nbsp;2026-09-08 当场栽了一次。**
   本文件里的「§3.2」「§3.2.1」是 **Vaswani 论文自己的小节号**，
   而正文里还有「§X.Y」是 **本课的小节号**。
   插 §一 那次整体重编号，脚本按 `§(\d)\.(\d)` 机械 +1，
   **把论文的 §3.2 也改成了 §4.2** ——&nbsp;不报错、读起来还很像真的。
   ⭐ 判据：**同一种记号承载两种含义时，机械替换一定会串。**
     以后再重编号：本课节号请统一写成「本课 §X.Y」或只在正文字符串里出现，
     出处引文里的论文节号一律带上论文名前缀（本文件已经都带了「Vaswani … §3.2」）。

⛔ 本文件只画图。**画法基元在 `topic03_draw.py`，别在这里另起一套。**
"""
from topic03_draw import (Fig, wpx, _sz,
                          BL, OR, GR, RD, GY, PU, CY, BR, INK,
                          GY2, LINE, LINE2, BG2)   # ⭐ LINE2 2026-09-12 补：新图要用


# ══════════════════════════════════════════════════════════════════
# 图一：换掉了什么 —— 一根横箭头，换成一张全连接图
# ══════════════════════════════════════════════════════════════════
def fig_swap():
    W = 1400
    f = Fig(W, "RNN 与自注意力的信息通路对比：RNN 靠一条链一步一步传，任意两个位置"
               "之间要走 n 步；自注意力让每个位置直接连到所有位置，两点之间只隔一步，"
               "代价是连线数从 n 变成 n 的平方")
    f.marks = set()
    y = f.header(
        '2017 年那一刀 ——&#160;'
        '<tspan font-weight="700">把那根横箭头拿掉，换成「每个位置直接看所有位置」</tspan>',
        '⭐ §零 三个痛点里的第 ① 条，就是在这里被解掉的。'
        '<tspan font-weight="700">而解法本身，就是这一讲后面要还的那笔账。</tspan>',
        [(RD, "串行：只能一步一步走"), (BL, "要算的格子"),
         ("#f1f3f4", "被因果遮罩挡住的未来")])

    N, R, PY = 6, 19, y
    # ── 左：RNN ────────────────────────────────────────────────
    LT = f.panel(0, PY, 660, 272, "Ⓐ RNN：一条链", RD, "#fff",
                 sub="信息只能沿着链爬", tag="§零 讲过", tint="#fadad6")
    for i in range(N):
        cx = 70 + i * 96
        f.box(cx - R, LT + 42 - R, 2 * R, 2 * R, "#fff", RD, R)
        f.t(cx, LT + 47, "t%d" % (i + 1), RD, True, _sz(12), "middle")
        if i:
            f.line(cx - 96 + R + 5, LT + 42, cx - R - 5, LT + 42, RD, 1.8)
    f.t(14, LT + 96, '<tspan font-weight="700">t1 想影响 t6，得经过 5 跳</tspan>'
                     '——&#160;每一跳都是一次矩阵乘，而且必须排队。', GY, size=_sz(12))
    f.t(14, LT + 122, '⛔ 串行步数 <tspan font-weight="700">O(n)</tspan>'
                      '　·　最长路径 <tspan font-weight="700">O(n)</tspan>'
                      '　·　连线数 <tspan font-weight="700">n − 1</tspan>', RD, size=_sz(12))
    f.t(14, LT + 152, '⭐ 信息走得越远越容易被冲淡 ——&#160;'
                      '<tspan font-weight="700">这同时也是痛点 ② 梯度消失的几何解释</tspan>：'
                      '梯度也得沿着同一条链爬回去。', GY, size=_sz(12))
    f.t(14, LT + 178, '⭐ 而且这条链一次只能动一格，'
                      '<tspan font-weight="700">加速器上再多的并行单元也用不上</tspan>。',
        GY, size=_sz(12))

    # ── 右：自注意力 ────────────────────────────────────────────
    # ⛔ 初版画成「上下两排点 ＋ 贝塞尔连线」——&nbsp;n=6 就有 21 根线，
    #   全挤在右下角糊成一团，**「平方长大」这个要点反而看不见了**。
    # ⭐ 改成直接画 n×n 的格子：一格 ＝ 一个要算的数。
    #   这样三件事同时出来：① 总数是平方；② 因果遮罩遮掉的是哪一半；
    #   ③ **它就是后面反复出现的那个「注意力矩阵」本人**。
    #   判据：**要讲「有多少」，就别画「怎么连」。**
    QX = 676
    QT = f.panel(QX, PY, W - QX, 272, "Ⓑ 自注意力：一张 n × n 的表", BL, "#fff",
                 sub="每一格 ＝ 一个 query 对一个 key 的打分", tint="#d5e4fb")
    G, CELL = 8, 21
    gx, gy = QX + 96, QT + 26
    for r in range(G):
        for c in range(G):
            x, yv = gx + c * CELL, gy + r * CELL
            # ⛔ 原先每格都描一圈饱和蓝边 ——&nbsp;64 格就是 64 条彩线，
            #   那是「中性框线只占一半」的大头。⭐ 热力图本来就该只用填充。
            if c <= r:
                lit = (r == G - 2)
                f.box(x, yv, CELL - 3, CELL - 3,
                      "#669df6" if lit else "#c6dafc", "none", 2, 0)
            else:
                f.box(x, yv, CELL - 3, CELL - 3, "#f1f3f4", "none", 2, 0)
    f.t(gx - 10, gy + 10, "q1", GY2, size=_sz(11), anchor="end")
    f.t(gx - 10, gy + (G - 2) * CELL + 10, "q7", BL, True, _sz(11), anchor="end")
    f.t(gx - 10, gy + (G - 1) * CELL + 10, "qn", GY2, size=_sz(11), anchor="end")
    f.t(gx, gy - 8, "k1", GY2, size=_sz(11))
    f.t(gx + (G - 1) * CELL, gy - 8, "kn", GY2, size=_sz(11), anchor="middle")
    f.t(gx + G * CELL + 16, gy + (G - 2) * CELL + 10,
        "← q7 这一行：它看得到 k1…k7", BL, True, _sz(12))
    f.t(gx + G * CELL + 16, gy + 14, "灰格 ＝ 被因果遮罩挡住的未来", GY2, size=_sz(11))
    f.t(gx + G * CELL + 16, gy + 34, "（softmax 之前置成 −∞）", GY2, size=_sz(11))
    f.t(QX + 14, QT + 208, '⭐ 串行步数 <tspan font-weight="700">O(1)</tspan>'
                           '　·　最长路径 <tspan font-weight="700">O(1)</tspan>'
                           '　·　⛔ 格子数 <tspan font-weight="700">n² ——&#160;'
                           '这才是这一刀的账单</tspan>', BL, size=_sz(12))
    f.t(QX + 14, QT + 230, '⭐ 任意两个位置之间<tspan font-weight="700">只隔一格</tspan>'
                           '，而且<tspan font-weight="700">整张表可以一次算完</tspan>'
                           '——&#160;左边那条链两样都做不到。', GY, size=_sz(12))

    yy = PY + 272 + 16
    yy = f.band(yy, "warn", "这一刀换来了什么，又欠下了什么", [
        '<tspan font-weight="700">换来的</tspan>：串行步数从 O(n) 掉到 O(1)'
        '——&#160;整段序列一次算完，加速器终于喂得饱了（这正是 §零 图二那笔账的反面）。',
        '<tspan font-weight="700">欠下的</tspan>：连线数从 n 变成 n²。'
        '⛔ 而且注意 ——&#160;<tspan font-weight="700">状态没了</tspan>：'
        'RNN 那个固定大小的 h 被换成了「把所有历史原封不动留着」。',
        '⭐ 于是 §零 图三 Ⓒ 那一行的病根在这儿：'
        '<tspan font-weight="700">解码时每一步都得把「所有历史」重读一遍 ——&#160;那就是 KV cache。</tspan>'])
    yy = f.src(yy + 18,
               'Vaswani et al. 2017 (arXiv 1706.03762) 表 1：自注意力 串行步数 O(1)／最长路径 O(1)；'
               '循环层两项都是 O(n)。因果遮罩见 §3.2.3：把非法连接「setting to −∞」')
    f.save("fig3-mha-swap.svg", yy + 6)


# ══════════════════════════════════════════════════════════════════
# 图二：一层里到底在算什么 —— 检索三件套 ＋ 为什么除 √d
# ══════════════════════════════════════════════════════════════════
def fig_qkv():
    W = 1400
    f = Fig(W, "注意力的检索三件套：query 是要找什么，key 是每个 token 挂出来的牌子，"
               "value 是牌子后面的内容；打分之后除以根号 d 再做 softmax，"
               "最后按权重把 value 加权求和")
    f.marks = set()
    y = f.header(
        '一层里到底在算什么 ——&#160;'
        '<tspan font-weight="700">论文自己的说法就是「检索」，不是我们编的比喻</tspan>',
        '⭐ 原文：mapping a <tspan font-style="italic">query</tspan> and a set of '
        '<tspan font-style="italic">key-value pairs</tspan> to an output …&#160;'
        'the output is a <tspan font-weight="700">weighted sum of the values</tspan>。',
        [(PU, "Q 要找什么"), (CY, "K 挂出来的牌子"), (GR, "V 牌子后面的货"),
         (OR, "打分 ＋ 归一")])

    # ── 上：三件套 ───────────────────────────────────────────────
    TT = f.panel(0, y, W, 172, "① 三件套都是从同一个 x 投影出来的",
                 sub="同一个 token，用三套不同的权重看它三次")
    f.cell(30, TT + 30, 120, 56, "x", "这个 token", INK, "#fff")
    for i, (lab, sub, col, fill) in enumerate((
            ("Q = x·W_Q", "我要找什么", PU, "#f3e8fd"),
            ("K = x·W_K", "我是什么，挂个牌", CY, "#e0f7fa"),
            ("V = x·W_V", "我肚子里有什么货", GR, "#e6f4ea"))):
        f.cell(230, TT + 12 + i * 50, 230, 42, lab, sub, col, fill)
        f.line(152, TT + 58, 226, TT + 33 + i * 50, col, 1.3)
    f.t(500, TT + 40, '⭐ <tspan font-weight="700">为什么要投三次而不是直接拿 x 比</tspan>',
        INK, size=13, cls="svglbl")
    f.t(500, TT + 64, '因为「我想找什么」和「我能提供什么」<tspan font-weight="700">'
                      '本来就是两回事</tspan>。', GY, size=_sz(12))
    f.t(500, TT + 86, '同一个词当 query 时该问的问题，跟它当 key 时该挂的牌子，'
                      '不该是同一个向量。', GY, size=_sz(12))
    f.t(500, TT + 116, '⛔ 而 V 又跟 K 分开，是因为'
                       '<tspan font-weight="700">「凭什么被选中」和「被选中之后交出什么」</tspan>'
                       '也是两回事。', GY, size=_sz(12))
    f.t(500, TT + 138, '⭐ 三个投影 ＝ 三个可训练矩阵，'
                       '<tspan font-weight="700">这也是 KV cache 里存的那两样东西的出处</tspan>。',
        GY, size=_sz(12))

    # ── 下：一条流水线 ───────────────────────────────────────────
    BY = y + 172 + 14
    BT = f.panel(0, BY, W, 176, "② 一次注意力，四步", OR, "#fff",
                 sub="Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V", tint="#fbeecb")
    STEPS = (("Q · Kᵀ", "每个 query 跟每个 key 打分", PU, "#f3e8fd", 200),
             ("÷ √d_k", "把方差拉回 1", RD, "#fce8e6", 150),
             ("softmax", "归一成「注意力权重」", OR, "#fff3e0", 190),
             ("· V", "按权重把货加权求和", GR, "#e6f4ea", 180))
    x = 30
    for i, (main, sub, col, fill, w) in enumerate(STEPS):
        f.cell(x, BT + 22, w, 52, main, sub, col, fill)
        if i:
            f.line(x - 26, BT + 48, x - 8, BT + 48, GY2, 1.4)
        x += w + 34
    f.line(x - 26, BT + 48, x - 8, BT + 48, GY2, 1.4)
    f.box(x + 6, BT + 22, 210, 52, "#fff", OR, 6)
    f.t(x + 111, BT + 44, "输出 [n × d]", OR, True, 13, "middle")
    f.t(x + 111, BT + 62, "跟输入一样的形状", GY, size=_sz(11), anchor="middle")
    # ⛔ 2026-09-08：这里原先写「它是整层里唯一一个随长度平方长大的东西」——
    #   那是**存储口径**，而这一步根本不落地（FlashAttention 已是标配）。
    #   ⭐ 改成计算口径，并把「要留下来的是什么」这条主线接上去。
    f.t(30, BT + 104, '⛔ 第一步那个 Q·Kᵀ 是 <tspan font-weight="700">n × n 个数</tspan>'
                      '——&#160;但它<tspan font-weight="700">不落地</tspan>，'
                      '只在片上过。<tspan font-weight="700">贵在要算的次数，不在显存。</tspan>',
        BR, size=_sz(12))
    f.t(30, BT + 128, '⭐⭐ <tspan font-weight="700">真正要留下来的是 K 和 V</tspan>'
                      '——&#160;下一个 token 还要跟它们打分。'
                      '<tspan font-weight="700">这两份就是 KV cache，也是本讲唯一的账本。</tspan>',
        BR, size=_sz(12))

    yy = BY + 176 + 16
    yy = f.band(yy, "info", "那个 √d_k 不是玄学，推导链在论文脚注里，两行就能讲完", [
        '假设 q、k 各维<tspan font-weight="700">独立、均值 0、方差 1</tspan>，'
        '那么 q·k ＝ Σ q<tspan baseline-shift="sub" font-size="10">i</tspan>'
        'k<tspan baseline-shift="sub" font-size="10">i</tspan> 的'
        '<tspan font-weight="700">均值是 0，方差是 d_k</tspan>——&#160;标准差就是 √d_k。',
        '⭐ 所以除以 √d_k 只做一件事：<tspan font-weight="700">把打分的方差拉回 1</tspan>。'
        '维度越高分数摊得越开，不拉回来 softmax 会被推到「几乎全是 0 和 1」的角落，'
        '<tspan font-weight="700">梯度就没了</tspan>。',
        '⛔ 别只说「防止 softmax 饱和」——&#160;那是结论不是理由。'
        '<tspan font-weight="700">理由是方差随维度线性长大，而 softmax 只认绝对数值。</tspan>'])
    yy = f.src(yy + 18,
               'Vaswani et al. 2017 §3.2：「a weighted sum of the values … compatibility '
               'function of the query with the corresponding key」',
               '§3.2.1 与脚注 4：「the dot products grow large in magnitude, pushing the '
               'softmax function into regions where it has extremely small gradients」')
    f.save("fig3-mha-qkv.svg", yy + 6)


# ══════════════════════════════════════════════════════════════════
# 图三：多头在多什么
# ══════════════════════════════════════════════════════════════════
def fig_heads():
    W = 1400
    f = Fig(W, "多头注意力：把 d_model 切成 h 份，每份独立做一次注意力再拼回来。"
               "总维度不变，总计算量也基本不变；多头的价值是避免单头把不同的关注平均掉")
    f.marks = set()
    y = f.header(
        '多头在多什么 ——&#160;'
        '<tspan font-weight="700">不是堆算力，是不让「平均」把不同的关注糊成一团</tspan>',
        '⭐ 原文：「With a <tspan font-style="italic">single</tspan> attention head, '
        '<tspan font-weight="700">averaging inhibits this</tspan>.」'
        '——&#160;<tspan font-weight="700">单头不是不够用，是会把该分开的东西平均掉。</tspan>')

    AT = f.panel(0, y, 470, 214, "Ⓐ 单头", GY, BG2, sub="一个 512 维的注意力",
                 tint="#eceff1")
    f.box(24, AT + 24, 420, 40, "#fff", GY2, 6)
    f.t(234, AT + 49, "一次注意力，d = 512", GY, True, 13, "middle")
    f.t(24, AT + 92, '一个 query 同时要兼顾：', GY, size=_sz(12))
    for i, s_ in enumerate(("「上一个词是什么」", "「这句话的主语是谁」",
                            "「三段之前提到的那个人名」")):
        f.t(40, AT + 114 + i * 20, "· " + s_, GY2, size=_sz(12))
    f.t(24, AT + 182, '⛔ 只有<tspan font-weight="700">一组</tspan>权重可分配 ——&#160;'
                      '结果是<tspan font-weight="700">把三种关注平均了一下</tspan>。',
        RD, size=_sz(12))

    BX = 486
    BT = f.panel(BX, y, W - BX, 214, "Ⓑ 多头（h ＝ 8）", BL, "#fff",
                 sub="切成 8 份，每份 64 维，各看各的，最后拼回来", tint="#d5e4fb")
    COLS = (PU, CY, GR, OR, RD, "#00838f", "#7b1fa2", BL)
    for i in range(8):
        x = BX + 24 + i * 108
        f.box(x, BT + 24, 92, 40, "#fff", COLS[i], 6)
        f.t(x + 46, BT + 49, "head %d" % (i + 1), COLS[i], True, _sz(12), "middle")
        f.t(x + 46, BT + 80, "d = 64", GY2, size=_sz(11), anchor="middle")
        f.line(x + 46, BT + 92, x + 46, BT + 108, COLS[i], 1.2)
    f.box(BX + 24, BT + 112, 8 * 108 - 16, 34, "#fff", BL, 6)
    f.t(BX + 24 + (8 * 108 - 16) / 2.0, BT + 134,
        "Concat → 再过一个 W_O，拼回 512", BL, True, 13, "middle")
    f.t(BX + 24, BT + 182, '⭐ <tspan font-weight="700">8 × 64 ＝ 512</tspan>'
                           '——&#160;总维度没变、参数量没变、计算量也基本没变。'
                           '<tspan font-weight="700">多头是「切开」，不是「加倍」。</tspan>',
        BL, size=_sz(12))

    yy = y + 214 + 16
    yy = f.band(yy, "ok", "多头的代价，正好是本专题的题眼", [
        '<tspan font-weight="700">好处</tspan>：8 个头可以同时盯 8 种不同的关系，'
        '互不干扰 ——&#160;这是单头做不到的。',
        '⛔ <tspan font-weight="700">代价</tspan>：'
        '<tspan font-weight="700">每个头都要自己的 K 和 V</tspan>。'
        '于是要留下来的东西<tspan font-weight="700">乘以了 8</tspan>。',
        '⭐⭐ 所以后面第一个旋钮（每个 token 存多少）第一刀砍的就是这里：'
        '<tspan font-weight="700">MQA 让 8 个头共用 1 组 KV，GQA 折中成几组</tspan>'
        '——&#160;<tspan font-weight="700">砍的正是多头在这一步乘上去的那个 8。</tspan>'])
    yy = f.band(yy + 14, "bad", "而这个形状的代价，2019 年就被点名了", [
        'Shazeer 2019（MQA 那篇）摘要原话：训练很快，因为序列方向可以并行；但 ——',
        '<tspan font-style="italic">「incremental inference … is often slow, due to the '
        '<tspan font-weight="700">memory-bandwidth cost of repeatedly loading '
        'the large &quot;keys&quot; and &quot;values&quot; tensors</tspan>」</tspan>',
        '⭐⭐ <tspan font-weight="700">这句话把 §零 图三 Ⓒ 那一行从我们的推论变成了原文</tspan>'
        '——&#160;2017 年造出这个形状，<tspan font-weight="700">2019 年就有人把它命名成问题了</tspan>。'])
    yy = f.src(yy + 18,
               'Vaswani et al. 2017 §3.2.2：「jointly attend to information from different '
               'representation subspaces」、「With a single attention head, averaging '
               'inhibits this」；h = 8，d_k = d_v = d_model/h = 64',
               'Shazeer 2019 (arXiv 1911.02150)《Fast Transformer Decoding: One Write-Head '
               'is All You Need》摘要 ——&#160;本课模型表第二行 PaLM 用的就是它')
    f.save("fig3-mha-heads.svg", yy + 6)




# ══════════════════════════════════════════════════════════════════════
# 图 · 一次注意力到底把信息怎么搬过去的（2026-09-12 加）
# ══════════════════════════════════════════════════════════════════════
# ⭐ 现场要求原话：「借机先把 Transformer 讲了 —— 不用讲 MLP，就讲 attention；
#   讲 Attention is all you need 到底是为啥；讲一个大白话让大家理解，
#   为什么注意力就能把序列里所有信息互相传递；讲清楚到底怎么把所有信息
#   弄到最后一个 token 的 embedding 上去的。**主要以画图为主，别写太多字。**」
#
# ⛔ 为什么另起一张，而不是改 fig3-mha-qkv：
#   那张答的是「一层里在算什么」——&nbsp;四步、形状、√d，是**机械**的。
#   这张答的是「信息怎么流」——&nbsp;是**直觉**的。两个问题，两张图。
#   ⭐ 判据：**一张图只回答一个问题；「顺便也讲讲」就是它开始讲不清的时候。**
#
# ⚠️ 图里那组权重（5/30/8/45/7/5 ％）是**示意**，不是实测 —— 图上标了。
W3 = 1400


def fig_flow():
    f = Fig(W3, "跟着最后一个 token 走一遍注意力：每个位置长出 query、key、value 三样东西；"
                "最后那个 token 拿自己的 query 去跟每个 key 打分，softmax 成一组加起来等于一的权重；"
                "再按权重把所有位置的 value 加起来，得到它的新向量")
    y = f.header(
        "一次注意力，干的就是一件事 ——&#160;"
        '<tspan font-weight="700">每个位置都去全场取一次货</tspan>',
        "跟着最后那个 token 走一遍：它怎么提问、别人怎么报价、货怎么汇到它身上",
        [(BL, "query 我想找什么"), (OR, "key 我这儿有什么"), (GR, "value 被选中我就交这个")])

    TOK = ["t1", "t2", "t3", "t4", "t5", "t6"]
    WGT = [5, 30, 8, 45, 7, 5]              # ％，示意值，和 = 100
    assert sum(WGT) == 100
    ROW, TOP = 46, y + 62   # ⭐ colhead 带副标题，占到 y+16，内容要让开
    def ry(i): return TOP + i * ROW

    # ── ① 每个位置长出三样东西 ───────────────────────────────
    f.colhead(0, y + 16, "① 每个位置都长出三样东西", "同一个向量，乘三个不同的矩阵")
    for i, tk in enumerate(TOK):
        hot = (i == len(TOK) - 1)
        f.box(0, ry(i) - 14, 46, 26, "#e8f0fe" if hot else "#fff",
              BL if hot else LINE, 6, 1.6 if hot else 1)
        f.t(23, ry(i) + 4, tk, BL if hot else INK, bold=hot, size=12, anchor="middle")
        for j, c in enumerate((BL, OR, GR)):
            f.box(70 + j * 30, ry(i) - 11, 24, 20, "#fff", c, 4, 1.4)
            f.t(82 + j * 30, ry(i) + 4, "qkv"[j], c, bold=True, size=11, anchor="middle")
        f.line(48, ry(i), 66, ry(i), LINE2, 1.2, arrow=False)

    # ── ② t6 拿 q 去跟每块牌子打分 ───────────────────────────
    X2 = 215
    f.colhead(X2, y + 16, "② 最后那个 token 拿它的 q 去对每块牌子",
              "打分 → softmax → 一组加起来 ＝ 1 的权重")
    QX, QY = X2 + 6, ry(len(TOK) - 1)
    f.box(QX, QY - 15, 34, 28, "#e8f0fe", BL, 6, 1.6)
    f.t(QX + 17, QY + 4, "q", BL, bold=True, size=13, anchor="middle")
    f.t(QX + 17, QY + 30, "t6 的问题", BL, size=11, anchor="middle")
    KX = X2 + 120
    for i in range(len(TOK)):
        f.box(KX, ry(i) - 11, 26, 20, "#fff", OR, 4, 1.3)
        f.t(KX + 13, ry(i) + 4, "k", OR, bold=True, size=11, anchor="middle")
        f.path("M %d %d C %d %d %d %d %d %d" % (QX + 36, QY, KX - 34, QY,
                                                KX - 34, ry(i), KX - 3, ry(i)),
               GY2, 1.1, arrow=True)
    # 权重条
    BX = KX + 46
    f.t(BX, TOP - 16, "softmax 之后", GY, size=11)
    for i, w_ in enumerate(WGT):
        f.box(BX, ry(i) - 8, 2 + w_ * 1.9, 15, "#e6f4ea", GR, 3, 1)
        f.t(BX + 6 + w_ * 1.9 + 8, ry(i) + 4, "%d%%" % w_, GR, bold=(w_ >= 30), size=11)
    f.t(BX, ry(len(TOK) - 1) + 30, "加起来 ＝ 1", GR, bold=True, size=11)

    # ── ③ 按权重把货加起来 ───────────────────────────────────
    X3 = 700
    f.colhead(X3, y + 16, "③ 按权重把所有人的 value 加起来",
              "权重越大，交上来的那份占比越大")
    for i, w_ in enumerate(WGT):
        bw = 22 + w_ * 1.5
        f.box(X3, ry(i) - 11, bw, 20, "#e6f4ea", GR, 4, 1.3)
        f.t(X3 + bw / 2, ry(i) + 4, "v", GR, bold=True, size=11, anchor="middle")
    OUTX = X3 + 250
    for i in range(len(TOK)):
        f.path("M %d %d C %d %d %d %d %d %d"
               % (X3 + 22 + WGT[i] * 1.5 + 4, ry(i), OUTX - 60, ry(i),
                  OUTX - 60, ry(len(TOK) - 1), OUTX - 6, ry(len(TOK) - 1)),
               GR, 1.0 + WGT[i] * 0.055, arrow=True)
    f.box(OUTX, ry(len(TOK) - 1) - 18, 118, 34, "#e8f0fe", BL, 7, 1.8)
    f.t(OUTX + 59, ry(len(TOK) - 1) + 4, "t6 的新向量", BL, bold=True, size=12,
        anchor="middle")
    f.lines(OUTX - 30, ry(len(TOK) - 1) + 34, 210, [
        '这<tspan font-weight="700">一个</tspan>向量里，现在装着',
        '<tspan font-weight="700">全场按需加权</tspan>的内容。'], size=11, lh=16, fill=GY)

    yy = ry(len(TOK) - 1) + 76

    # ── 落点带 ───────────────────────────────────────────────
    yy = f.band(yy, "ok", "为什么标题敢叫「Attention Is All You Need」", [
        'RNN 要让 t1 影响 t6，得<tspan font-weight="700">一跳一跳传五次</tspan>；上面这一步 ——&#160;<tspan font-weight="700">一跳</tspan>。'
        '而且<tspan font-weight="700">六个位置是同时做的</tspan>，不是排队。',
        '整个过程<tspan font-weight="700">只有矩阵乘和一次 softmax</tspan>，没有任何循环 ——&#160;所以它能一次性并行算完整个序列。',
        '⭐ 所以那句标题说的<tspan font-weight="700">不是「注意力很强」，是「混合信息这件事，只要它就够了」</tspan>'
        '——&#160;<tspan font-weight="700">循环不需要，卷积也不需要</tspan>。'])

    yy = f.band(yy + 14, "info", "两个最常被跳过的「为什么」", [
        '<tspan font-weight="700">为什么要投三次，不能只用一个向量？</tspan>'
        "因为「我想找什么」和「我能提供什么」本来就是两回事（q ≠ k）；"
        "而「凭什么被选中」和「被选中之后交出什么」也是两回事（k ≠ v）。",
        '<tspan font-weight="700">为什么说「所有信息互相传递」，上面不是只画了 t6 吗？</tspan>'
        '因为<tspan font-weight="700">六个位置在同时做同样的事</tspan> ——&#160;'
        '一层过后，<tspan font-weight="700">每个位置的向量都变成了「全场的一个加权视角」</tspan>。',
        '⭐ 堆 L 层，就是把这件事<tspan font-weight="700">重复 L 次</tspan>，每一次都基于上一次的结果 ——&#160;'
        '<tspan font-weight="700">这就是「理解」在 Transformer 里的全部形式。</tspan>'])

    yy = f.src(yy + 16,
               "「query / key / value」与「输出是 value 的加权和」是 Vaswani 2017 §3.2 的原文措辞，"
               '不是本课编的比喻；<tspan font-weight="700">图中那组权重（5/30/8/45/7/5 ％）是示意值，不是实测</tspan>',
               '⚠️ 本图<tspan font-weight="700">只画注意力</tspan> ——&#160;一层 Transformer 里还有 FFN、残差、归一化，'
               "它们不在本专题这条轴上（本专题的账本只有 KV cache）")
    f.save("fig3-mha-flow.svg", yy + 6)


fig_swap()
fig_qkv()
fig_flow()      # ⭐ 2026-09-12 加：信息怎么流（直觉），跟 qkv 那张（机械）分工
fig_heads()
