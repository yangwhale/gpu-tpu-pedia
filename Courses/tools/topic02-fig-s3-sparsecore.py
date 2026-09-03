# -*- coding: utf-8 -*-
"""图 3-5 · SparseCore 拆开看 ——「处理不规则访存」到底是怎么处理的。

⭐ **为什么要有这一张。** 2026-09-03 现场追问：

    「这个 SparseCore 叫做处理不规则访存，到底是怎么处理的不规则访存？
      包括它到底操作的是 VMEM 还是 HBM，就是 DMA。然后它是能让 DMA
      一次不搬 8×128 吗？它的最小粒度是多少？还是说它既可以操作 VMEM
      又可以操作 HBM？」

三个问题都很具体，而课件里当时只有「处理不规则访存」这个名字，答不上来。

⛔ **回答里有一处必须纠正我自己之前的说法。** 我在讲的时候说
   「TensorCore 一次搬 (8,128) 一整块」——&nbsp;这句话是**含混的**：
   `(8,128)` 是**向量寄存器的形状**（8 sublane × 128 lane），
   是计算和 VMEM 布局的自然单位，**不是 DMA 的最小粒度**。
   把两件事混成一件，正好是这门课一直在批的那类错。

⭐ **真正能回答粒度的是一个可执行的官方口径**：JAX 的
   `pltpu.get_tpu_info().sparse_core` 在 TPU 7x 上直接报

       SparseCoreInfo(num_cores=2, num_subcores=16, num_lanes=16,
                      dma_granule_size_bytes=64)

   —— **64 字节**。这是这张图的锚点数。

📌 内存空间那一问的答案（JAX Pallas SparseCore 文档原话）：
   「Each vector subcore has its own VMEM and SMEM space. They also have
     access to a shared VMEM space. … All these spaces connect with the
     TPU's HBM.」而 OpenXLA 那篇明说 TensorCore 的 VMEM
   「is not directly managed by the SparseCore」。
   **所以「VMEM 还是 HBM」这个二选一本身是个陷阱：两块都碰，
     但它碰的那块 VMEM 不是 TensorCore 那块。**

📌 规格全部出自公开文档（本仓库是公开仓库，不写任何内部数字）：
   - docs.jax.dev/en/latest/pallas/tpu/sparsecore.html（含 Ironwood 一列）
   - openxla.org/xla/sparsecore（SPMEM / TileSPMEM 的命名对应关系）
"""
import io

BL, OR, GR, RD, GY = "#1a73e8", "#e8710a", "#1e8e3e", "#d93025", "#5f6368"
PU = "#8430ce"
W = 1400
p = []


def t(x, y, s, cls="svgsm", fill=None, bold=False, size=None):
    st = []
    if size:
        st.append("font-size:%dpx" % size)
    p.append('<text class="%s" x="%d" y="%d"%s%s>%s</text>' % (
        cls, x, y, ' fill="%s"' % fill if fill else '',
        ' style="%s"' % ';'.join(st) if st else '',
        '<tspan font-weight="700">%s</tspan>' % s if bold else s))


def box(x, y, w, h, fill="#fff", stroke="#dadce0", r=8):
    p.append('<rect x="%d" y="%d" width="%d" height="%d" rx="%d" fill="%s" stroke="%s"/>'
             % (x, y, w, h, r, fill, stroke))


p.append('<svg viewBox="0 0 %d 796" width="100%%" role="img" aria-label="'
         'SparseCore 拆开看：十六个向量子核各自带 VMEM，一个标量子核发 DMA，'
         '大表在 HBM；以及它与 TensorCore 在最小搬运粒度上的差别">' % W)
t(0, 17, 'SparseCore 拆开看 ——&#160;<tspan font-weight="700">'
         '「处理不规则访存」处理的是什么</tspan>', "svglbl", "#202124", size=14)
t(0, 37, '地址由数据算出来、编译期不知道 ——&#160;这件事难在两处，'
         '而这颗核对着这两处各配了一个解法')

# ══ 上半：两颗核 + 同一块 HBM ═══════════════════════════════════════
YD = 56

# 左：TensorCore
box(0, YD, 470, 214, "#f8f9fa")
t(14, YD + 24, 'TensorCore', "svglbl", "#202124", size=13)
t(14, YD + 42, '本课前六小节拆的都是它', fill=GY)
box(14, YD + 54, 200, 62, "#e8f0fe", BL)
t(26, YD + 76, 'MXU × 4', "svglbl", "#174ea6", size=12)
t(26, YD + 94, '256×256 脉动阵列', fill="#174ea6")
box(226, YD + 54, 230, 62, "#e8f0fe", BL)
t(238, YD + 76, 'VPU', "svglbl", "#174ea6", size=12)
t(238, YD + 94, '向量寄存器 8 sublane × 128 lane', fill="#174ea6")
box(14, YD + 126, 442, 48, "#fff", BL)
t(26, YD + 146, 'VMEM', "svglbl", "#174ea6", size=12)
t(88, YD + 146, '——&#160;<tspan font-weight="700">TensorCore 专用</tspan>。'
                'SparseCore 不直接管它', fill="#174ea6")
t(14, YD + 194, '⛔ 它是编译期就排死的机器：地址得提前知道', fill=RD)

# 右：SparseCore
box(486, YD, W - 486, 214, "#fef7e0", OR)
t(500, YD + 24, 'SparseCore', "svglbl", "#7a5000", size=13)
t(500, YD + 42, '按 device 看 2 颗（物理 4 颗／chip）·  占的面积很小', fill="#7a5000")

box(500, YD + 54, 214, 120, "#fff", OR)
t(512, YD + 76, '标量子核 × 1', "svglbl", "#7a5000", size=12)
for i, s in enumerate(('标量运算',
                       '<tspan font-weight="700">动态索引</tspan>',
                       '<tspan font-weight="700">发起 DMA 和 stream</tspan>',
                       '自带 SMEM')):
    t(512, YD + 96 + i * 17, '· ' + s, fill="#7a5000")

box(726, YD + 54, 480, 120, "#fff", OR)
t(738, YD + 76, '向量子核 × 16<tspan fill="%s">（文档里也叫 tile）</tspan>' % GY,
  "svglbl", "#7a5000", size=12)
for k in range(16):
    x = 738 + (k % 8) * 57
    y = YD + 88 + (k // 8) * 22
    box(x, y, 50, 17, "#fce8b2", OR, 3)
t(738, YD + 148, '<tspan font-weight="700">每个子核自带 VMEM ＋ SMEM，数据流各走各的</tspan>',
  fill="#7a5000")
t(738, YD + 165, 'SIMD 宽度 16（F32）／32（BF16）——&#160;一次动的是十几个数，不是一千个',
  fill="#7a5000")

box(1218, YD + 54, 168, 120, "#fff", OR)
t(1230, YD + 76, '共享 VMEM', "svglbl", "#7a5000", size=12)
t(1230, YD + 96, 'OpenXLA 文档里', fill="#7a5000")
t(1230, YD + 112, '叫它 <tspan font-weight="700">SPMEM</tspan>', fill="#7a5000")
t(1230, YD + 134, '很小、很快、', fill="#7a5000")
t(1230, YD + 150, '编译器显式管，不是缓存', fill="#7a5000")

t(500, YD + 194, '⭐ 它原生支持<tspan font-weight="700">数据相关的控制流和访存</tspan>'
                 '——&#160;地址可以是刚算出来的', fill="#7a5000")

# HBM 横条
YH = YD + 226
box(0, YH, W, 58, "#e6f4ea", GR)
t(16, YH + 24, 'HBM', "svglbl", "#0b6b30", size=13)
t(70, YH + 24, '——&#160;192 GiB（v7）。<tspan font-weight="700">大表、中间结果都在这儿。'
               '两颗核都直接连它</tspan>', fill="#0b6b30")
t(16, YH + 44, '⭐ 所以「操作 VMEM 还是 HBM」这个二选一本身是个陷阱：'
               '<tspan font-weight="700">两块都碰</tspan>——&#160;'
               '只是它碰的那块 VMEM 是<tspan font-weight="700">它自己的</tspan>，'
               '不是 TensorCore 那块。', fill="#0b6b30")

# ══ 中：粒度对比 ═══════════════════════════════════════════════════
YG = YH + 74
t(0, YG + 16, '那「最小粒度」到底是多少 ——&#160;'
              '<tspan font-weight="700">先把两件被混为一谈的事分开</tspan>',
  "svglbl", "#202124", size=13)

box(0, YG + 28, 690, 116, "#fff", BL)
t(14, YG + 50, 'TensorCore 那个 (8,128)', "svglbl", "#174ea6", size=12)
t(14, YG + 70, '<tspan font-weight="700">它是向量寄存器的形状</tspan>'
               '（8 sublane × 128 lane ＝ 1,024 格）,', fill="#174ea6")
t(14, YG + 87, '也是 VMEM 里的布局单位 ——&#160;fp32 下一块 4 KiB。', fill="#174ea6")
t(14, YG + 110, '⛔ <tspan font-weight="700">它不是「DMA 的最小粒度」</tspan>。'
                '把这两件事说成一件，是我讲这一段时犯过的错。', fill=RD)
t(14, YG + 130, '但方向是对的：你要的若只是散落的几行，'
                '按这个单位对齐取数就会付冤枉钱。', fill=GY)

box(710, YG + 28, W - 710, 116, "#fef7e0", OR)
t(724, YG + 50, 'SparseCore 的 DMA 粒度', "svglbl", "#7a5000", size=12)
t(724, YG + 74, '64', "svglbl", "#7a5000", size=30)
t(772, YG + 74, '字节', "svglbl", "#7a5000", size=14)
t(830, YG + 66, '——&#160;这是<tspan font-weight="700">可执行的官方口径</tspan>，不是文档里的形容词：',
  fill="#7a5000")
t(830, YG + 84, '<tspan font-family="ui-monospace,monospace">'
                'pltpu.get_tpu_info().sparse_core</tspan> 在 TPU 7x 上报', fill="#7a5000")
t(830, YG + 101, '<tspan font-family="ui-monospace,monospace">'
                 'num_cores=2, num_subcores=16, num_lanes=16, '
                 'dma_granule_size_bytes=<tspan font-weight="700">64</tspan></tspan>',
  fill="#7a5000")
t(724, YG + 130, '⭐ 所以回答「能不能不搬 (8,128)」：'
                 '<tspan font-weight="700">这个问题问的是两台机器的两套单位</tspan>'
                 '——&#160;它按 64 字节走。', fill="#7a5000")

# ══ 下：gather 长什么样 ════════════════════════════════════════════
YQ = YG + 160
box(0, YQ, W, 158, "#f8f9fa")
t(14, YQ + 24, '一次 gather 在这张图上怎么走', "svglbl", "#202124", size=13)
t(14, YQ + 46, '① <tspan font-weight="700">索引</tspan>放在 SparseCore 自己的 VMEM 里 '
               '→ ② <tspan font-weight="700">标量子核</tspan>照着索引发一堆 DMA '
               '→ ③ 十六个向量子核<tspan font-weight="700">各追各的地址</tspan>，'
               '从 <tspan font-weight="700">HBM</tspan> 里取回来', fill="#202124")
t(14, YQ + 66, 'Pallas 里就是一行：<tspan font-family="ui-monospace,monospace">'
               'sync_copy(data_ref.at[indices_ref], target_ref)</tspan>'
               '——&#160;<tspan fill="%s">data 在 HBM，indices 在 VMEM。'
               'scatter 是同一条路反着走。</tspan>' % GY, fill="#202124")
t(14, YQ + 86, '⭐ <tspan font-weight="700">这条路不只用来查 embedding</tspan>：'
               'Ironwood 上 Qwen 3.5 那篇公开调优记录里，'
               'SparseCore 干的是<tspan font-weight="700">按路由索引把 token '
               '从 HBM 间接 gather 出来</tspan>，', fill="#202124")
t(14, YQ + 104, '直接喂给 TensorCore 做 GMM ——&#160;'
                '<tspan font-weight="700">同一个形状，换到稀疏注意力上'
                '就是「把 top-k 那批零碎的 KV 取回来」</tspan>。', fill="#202124")
t(14, YQ + 124, '⚠️ <tspan font-weight="700">取回来之后落在哪，公开资料没有明写。</tspan>'
                '那篇只说「写进一块连续的虚拟缓冲，'
                '<tspan font-style="italic">绕开了在 HBM 里物化中间张量</tspan>」'
                '——&#160;绕开 HBM 是明写的，', fill=GY)
t(14, YQ + 141, '但<tspan font-weight="700">是不是直接落进 TensorCore 的 VMEM，'
                '没有一处公开文档这么说</tspan>；'
                '而 OpenXLA 那篇讲 embedding 的又说 SC 与 TC 之间'
                '「frequently involves HBM as an intermediary buffer」。'
                '<tspan font-weight="700">两处口径不一致，这里不下结论。</tspan>', fill=GY)

# ══ 落点 ══════════════════════════════════════════════════════════
YB = YQ + 172
box(0, YB, W, 96, "#e6f4ea", GR)
t(16, YB + 24, '⭐ 它扛延迟的方式，不是让每一次取数变快 ——&#160;'
               '<tspan font-weight="700">是同时欠着很多次取数</tspan>',
  "svglbl", "#0b6b30", size=13)
t(16, YB + 44, '散乱访存的痛点从来不是带宽，是<tspan font-weight="700">每一次都得等</tspan>。'
               '十六个子核各自独立的数据流，就是为了把一大把请求同时抛出去、'
               '各自回来。', fill="#0b6b30")
t(16, YB + 62, '<tspan font-weight="700">换个说法：TPU 把 TensorCore 上砍掉的'
               '「运行时才知道地址」那套能力，单独做了一颗小的放在旁边。</tspan>'
               '主核保持纯静态、拿满算力；需要动态的时候，交给它。', fill="#0b6b30")
t(16, YB + 82, '⚠️ 规格出自 JAX Pallas SparseCore 文档与 OpenXLA SparseCore 文档；'
               '「为什么这套本事也适合卸载 collective」<tspan font-weight="700">'
               '是推的，官方没有公开解释</tspan>。', fill=GY)

p.append('</svg>')
io.open('fig3-5.svg', 'w', encoding='utf-8').write('\n'.join(p))
print('fig3-5 ok')
