# -*- coding: utf-8 -*-
"""图 3-5b · 把 SparseCore 拆开 —— 里面有哪些子核、各干什么、彼此怎么连。

⭐ **为什么要有这一张。** 2026-09-06 现场要求：

    「Sparse Core 这一部分就跟 MXU 和 tensor core 一样重要，所以你在那个课件里边
      把这个图都给它拆开，画清楚：1. 里边都有什么 sub core 2. tile 3. 它们都是
      干什么的 4. 都讲清楚，然后彼此之间的关系，然后线都画上。」

前一张（fig3-5）回答的是「它跟 TensorCore 有什么不同、粒度多少」——&nbsp;
**它把 SparseCore 当成一个整体在用**。这一张才是真的把盖子掀开。

⭐ **画法上的一个决定：放大其中一个，而不是把细节铺满十六个。**
   十六个子核长得一模一样，逐个标注「本地 VMEM ＋ SMEM ＋ SIMD 宽度」
   会得到十六份同样的字，读者反而看不出「它们是同构的」这个要点。
   所以：**网格里只标编号，把 ① 号单独拉出来放大**——&nbsp;
   「同构 × 16」这件事由网格表达，「一个里面有什么」由放大图表达。
   （同样的理由，别再往网格的小格子里加字。）

⛔ **本仓库是公开仓库，这张图上每一个名字都必须是公开文档里的叫法。**
   JAX Pallas 官方文档用的词是 **vector subcore（也叫 tile）** 和
   **scalar subcore**，共享那块叫 **SPMEM**、每个 tile 自己那块叫
   **TileSPMEM / local SPMEM**。**照抄这套叫法，不要用别处看来的内部代号。**

📌 规格与描述全部出自公开文档：
   - docs.jax.dev/en/latest/pallas/tpu/sparsecore.html
     （子核构成、SIMD 宽度、`dma_granule_size_bytes=64`、擅长的操作清单、
       `ScalarSubcoreMesh` / `VectorSubcoreMesh`）
   - openxla.org/xla/sparsecore（SPMEM 命名）
"""
import io

BL, OR, GR, RD, GY = "#1a73e8", "#e8710a", "#1e8e3e", "#d93025", "#5f6368"
BR = "#7a5000"          # 橙系正文色，和 fig3-5 保持一致
W = 1400
p = []


def t(x, y, s, cls="svgsm", fill=None, bold=False, size=None, anchor=None):
    st = []
    if size:
        st.append("font-size:%dpx" % size)
    p.append('<text class="%s" x="%d" y="%d"%s%s%s>%s</text>' % (
        cls, x, y, ' fill="%s"' % fill if fill else '',
        ' text-anchor="%s"' % anchor if anchor else '',
        ' style="%s"' % ';'.join(st) if st else '',
        '<tspan font-weight="700">%s</tspan>' % s if bold else s))


def wpx(s, size=11.5):
    """粗估一段纯文本的像素宽度。

    ⛔ 2026-09-06 几何 lint 抓到「标量子核照着索引发 DMA」跟后半句撞车 ——&nbsp;
       原来用的是 `8 * len(s)`。**中日韩字符差不多是一个字号宽，ASCII 只有一半多**，
       用一个平均系数去乘字数，在中英混排的短语上必然偏小。
       ⭐ 规矩：**图上要把两段文字排在同一基线上时，别用 len() 估宽。**
    """
    n = 0.0
    for ch in s:
        n += 1.0 if ord(ch) > 0x2E80 else 0.55
    return int(n * size)


def box(x, y, w, h, fill="#fff", stroke="#dadce0", r=8, sw=1):
    p.append('<rect x="%d" y="%d" width="%d" height="%d" rx="%d" fill="%s" '
             'stroke="%s" stroke-width="%s"/>' % (x, y, w, h, r, fill, stroke, sw))


def arrow(x1, y1, x2, y2, color=GY, dash=None, both=False, sw=1.6):
    p.append('<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="%s"%s '
             'marker-end="url(#ah-%s)"%s/>' % (
                 x1, y1, x2, y2, color, sw,
                 ' stroke-dasharray="%s"' % dash if dash else '',
                 color.lstrip('#'),
                 ' marker-start="url(#ahr-%s)"' % color.lstrip('#') if both else ''))


p.append('<svg viewBox="0 0 %d 772" width="100%%" role="img" aria-label="'
         'SparseCore 内部拆解：一个标量子核派活、十六个同构的向量子核各自带本地内存、'
         '一块共享 SPMEM，全部直连 HBM">' % W)

# 箭头 marker（每种颜色一份）
p.append('<defs>')
for c in (GY, OR, BL, GR):
    cid = c.lstrip('#')
    p.append('<marker id="ah-%s" viewBox="0 0 10 10" refX="9" refY="5" '
             'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
             '<path d="M0,0 L10,5 L0,10 z" fill="%s"/></marker>' % (cid, c))
    p.append('<marker id="ahr-%s" viewBox="0 0 10 10" refX="1" refY="5" '
             'markerWidth="6" markerHeight="6" orient="auto">'
             '<path d="M10,0 L0,5 L10,10 z" fill="%s"/></marker>' % (cid, c))
p.append('</defs>')

t(0, 17, '把 SparseCore 拆开 ——&#160;<tspan font-weight="700">'
         '里面有哪些子核、各干什么、线怎么连</tspan>', "svglbl", "#202124", size=14)
t(0, 37, '「SparseCore 对比 TensorCore」那张把它当成一个整体在用；这一张掀开盖子。'
         '<tspan font-weight="700">形状上它更像一台十六路的搬运机，而不是一台计算器。</tspan>')

# ══ 主面板：一颗 SparseCore ════════════════════════════════════════
PY, PH = 50, 292
box(0, PY, W, PH, "#fef7e0", OR)
t(16, PY + 24, '一颗 SparseCore', "svglbl", BR, size=13)
t(150, PY + 24, '——&#160;v7 上每颗芯片 <tspan font-weight="700">4 个物理核</tspan>，'
                '按 device 算 <tspan font-weight="700">2 个</tspan>。'
                '整颗核占的面积很小，但它是这一节的第二个主角。', fill=BR)
t(16, PY + 42, '⭐ 读法：<tspan font-weight="700">橙色虚线＝派活（控制与 DMA 请求），'
               '蓝色实线＝真正的数据搬运</tspan>。'
               '两种线分开画，是因为这颗核最反直觉的一点就是'
               '——&#160;<tspan font-weight="700">发命令的人自己不搬数据</tspan>。', fill=GY)

BY, BH = PY + 54, 218          # 子面板 y = 104 .. 322

# ── 标量子核 ──────────────────────────────────────────────────────
SX, SW_ = 16, 222
box(SX, BY, SW_, BH, "#fff", OR)
t(SX + 12, BY + 22, '标量子核 × 1', "svglbl", BR, size=12)
t(SX + 12, BY + 38, 'scalar subcore', fill=GY)
for i, s in enumerate((
        '标量运算（一次一个数）',
        '<tspan font-weight="700">动态索引</tspan> ——&#160;地址可以是',
        '&#160;&#160;&#160;刚从数据里算出来的',
        '<tspan font-weight="700">发起 DMA 与 stream</tspan>',
        '自带 <tspan font-weight="700">SMEM</tspan>（它的私有便签）')):
    t(SX + 12, BY + 62 + i * 18, '· ' + s if not s.startswith('&#160;') else s, fill=BR)
box(SX + 12, BY + 158, SW_ - 24, 46, "#fce8b2", OR, 6)
t(SX + 22, BY + 178, '⭐ 它是调度中枢：', fill=BR, bold=True)
t(SX + 22, BY + 194, '不算数据，只决定谁去搬什么', fill=BR)

# ── 十六个向量子核 ────────────────────────────────────────────────
GX, GW = 270, 500
box(GX, BY, GW, BH, "#fff", OR)
t(GX + 12, BY + 22, '向量子核 × 16', "svglbl", BR, size=12)
t(GX + 118, BY + 22, '——&#160;公开文档里也直接叫 <tspan font-weight="700">tile</tspan>',
  fill=GY)
t(GX + 12, BY + 40, '<tspan font-weight="700">十六个完全同构</tspan>，'
                    '每个自带内存、数据流各走各的', fill=BR)
# ⛔ 2026-09-06 几何 lint 抓到：CELLH=34 / GAP=8 时第四行落在 BY+178..212，
#    正好压上 BY+206 那句 ⛔ 警示语。**16 个格子的网格高度是 4 行累加的，
#    调任何一个数都要重算最后一行的下沿**，不能只看单元格。
#    现在：4×30 + 3×5 = 135，网格 BY+50..BY+185，警示语 BY+207，留 22px。
CELLW, CELLH, GAP = 110, 30, 5
GX0, GY0 = GX + 12, BY + 50
for k in range(16):
    cx = GX0 + (k % 4) * (CELLW + GAP)
    cy = GY0 + (k // 4) * (CELLH + GAP)
    hot = (k == 0)
    box(cx, cy, CELLW, CELLH, "#fff5d6" if hot else "#fce8b2",
        RD if hot else OR, 5, 2 if hot else 1)
    t(cx + CELLW // 2, cy + 20, ('① ' if hot else '') + 'tile %d' % k,
      fill=RD if hot else BR, bold=hot, anchor="middle")
t(GX + 12, BY + 207, '⛔ 别把它当成「小一号的 TensorCore」：'
                     '<tspan font-weight="700">这里面没有 MXU</tspan>。', fill=RD)

# ── 共享 SPMEM ────────────────────────────────────────────────────
MX, MW = 802, 200
box(MX, BY, MW, BH, "#e8f0fe", BL)
t(MX + 12, BY + 22, '共享 VMEM', "svglbl", "#174ea6", size=12)
t(MX + 12, BY + 40, '文档里叫 <tspan font-weight="700">SPMEM</tspan>', fill="#174ea6")
for i, s in enumerate((
        '十六个 tile 都能访问',
        '<tspan font-weight="700">跨 tile 交换数据</tspan>',
        '&#160;&#160;走的就是这块',
        '很小、很快',
        '<tspan font-weight="700">编译器显式管</tspan>，',
        '&#160;&#160;不是缓存')):
    t(MX + 12, BY + 68 + i * 18, ('· ' + s) if not s.startswith('&#160;') else s,
      fill="#174ea6")
# ⛔ 单行「⛔ 不是 TensorCore 那块 VMEM」约 195px，盒子只有 176px —— 撑出去了。
#    中英混排的短语最容易在这里翻车，**放进定宽盒子的文字先用 wpx() 量一遍**。
box(MX + 12, BY + 166, MW - 24, 40, "#fff", BL, 5)
t(MX + 22, BY + 183, '⛔ 不是 TensorCore', fill="#174ea6")
t(MX + 22, BY + 199, '&#160;&#160;&#160;那块 VMEM', fill="#174ea6")

# ── 把 ① 号放大 ───────────────────────────────────────────────────
ZX, ZW = 1042, W - 1042 - 16
box(ZX, BY, ZW, BH, "#fff", RD, 8, 2)
t(ZX + 12, BY + 22, '① 把一个 tile 放大看', "svglbl", RD, size=12)
t(ZX + 12, BY + 40, '十六个长得一样，看懂一个就看懂全部', fill=GY)
zrows = (('向量 ALU',
          'SIMD <tspan font-weight="700">16 lane</tspan>（F32）'
          '／<tspan font-weight="700">32</tspan>（BF16）'),
         ('本地 VMEM',
          '文档里叫 <tspan font-weight="700">TileSPMEM</tspan> ／ local SPMEM'),
         ('本地 SMEM', '这个 tile 自己的标量便签'))
for i, (a_, b_) in enumerate(zrows):
    yy = BY + 56 + i * 44
    box(ZX + 12, yy, ZW - 24, 38, "#fef7e0", OR, 5)
    t(ZX + 22, yy + 16, a_, fill=BR, bold=True)
    t(ZX + 22, yy + 31, b_, fill=BR)
box(ZX + 12, BY + 190, ZW - 24, 20, "#fce8b2", OR, 5)
t(ZX + 22, BY + 205, '⭐ 一次动的是十几个数，不是一千个', fill=BR)

# ── 线：标量子核 →（派活）→ 十六个 tile ──────────────────────────
for r in range(4):
    yy = GY0 + r * (CELLH + GAP) + CELLH // 2
    arrow(SX + SW_ + 4, yy, GX0 - 6, yy, OR, dash="5 4")
# ⛔ 这两个标签原来放在 BY+44（容器标题那一行的高度），而那个 y 上
#    左右两个白盒子都还在 —— 于是标签压在边框上。**夹缝里的标签要放在
#    夹缝真正空着的那一段 y 上**：两行箭头之间。垫一块白底保证可读。
_LBLY = GY0 + CELLH + GAP // 2 + 4
box((SX + SW_ + GX) // 2 - 14, _LBLY - 11, 28, 15, "#fef7e0", "#fef7e0", 2)
t((SX + SW_ + GX) // 2, _LBLY, '派活', fill=OR, bold=True, anchor="middle")

# ── 线：十六个 tile ←→（数据）←→ SPMEM ────────────────────────────
for r in range(4):
    yy = GY0 + r * (CELLH + GAP) + CELLH // 2
    arrow(GX0 + 4 * CELLW + 3 * GAP + 6, yy, MX - 6, yy, BL, both=True)
box((GX + GW + MX) // 2 - 14, _LBLY - 11, 28, 15, "#fef7e0", "#fef7e0", 2)
t((GX + GW + MX) // 2, _LBLY, '数据', fill=BL, bold=True, anchor="middle")

# ══ HBM ═══════════════════════════════════════════════════════════
HY = PY + PH + 34                     # 376
for x in (127, 400, 640, 902):
    arrow(x, PY + PH + 2, x, HY - 6, GR, sw=2)
# ⛔ 这句话原来放在箭头那条夹缝里居中 —— 一行长文本必然横穿好几根竖箭头。
#    **竖箭头组成的那条带子上不能横着写字**，挪进绿条里当一行。
box(0, HY, W, 80, "#e6f4ea", GR)
t(16, HY + 24, 'HBM', "svglbl", "#0b6b30", size=13)
t(70, HY + 24, '——&#160;<tspan font-weight="700">大表、权重、中间结果都在这儿</tspan>。'
               '每次 DMA 的最小粒度 <tspan font-weight="700">64 字节</tspan>'
               '（<tspan font-family="ui-monospace,monospace">'
               'pltpu.get_tpu_info().sparse_core</tspan> 直接报得出来）',
  fill="#0b6b30")
t(16, HY + 44, '<tspan font-weight="700">三种角色都直连 HBM</tspan>'
               '——&#160;标量子核往那儿发 DMA，tile 往那儿取数、写回，'
               'SPMEM 里周转的东西也从那儿来。', fill="#0b6b30")
t(16, HY + 64, '⭐ 对照一下就知道这颗核为什么存在：'
               'TensorCore 那边按<tspan font-weight="700">向量寄存器的形状</tspan>'
               '（8 sublane × 128 lane）成块地取，'
               '要的若只是散落的几行，就得为整块付钱；'
               '这边按 <tspan font-weight="700">64 字节</tspan>走。', fill="#0b6b30")

# ══ 一次 gather 在这张图上怎么走 ═══════════════════════════════════
FY = HY + 98
box(0, FY, 690, 152, "#f8f9fa")
t(14, FY + 24, '一次 gather 沿着这些线怎么走', "svglbl", "#202124", size=13)
steps = (
    ('索引先到位', '要取哪些行，这份索引本身也是数据，放在 SparseCore 自己的内存里'),
    ('标量子核照着索引发 DMA', '<tspan font-weight="700">橙色虚线</tspan>'
                              '——&#160;它一次抛出一大把请求，不等任何一个回来'),
    ('十六个 tile 各追各的地址', '<tspan font-weight="700">蓝色实线</tspan>'
                               '——&#160;从 HBM 取回自己那份，互不排队'),
    ('要跨 tile 汇总时经 SPMEM', 'scatter 就是同一条路反着走'))
for i, (a_, b_) in enumerate(steps):
    yy = FY + 48 + i * 25
    t(14, yy, '%d.' % (i + 1), fill=GY, bold=True)
    t(34, yy, a_, fill="#202124", bold=True)
    t(34 + wpx(a_) + 14, yy, '——&#160;' + b_, fill=GY)
t(14, FY + 145, '⭐ <tspan font-weight="700">它扛延迟的方式不是让每一次取数变快，'
                '是同时欠着很多次取数。</tspan>', fill="#202124")

# ══ 它擅长的四类活 ═════════════════════════════════════════════════
box(710, FY, W - 710, 152, "#fff", OR)
t(724, FY + 24, '它擅长的四类活（官方文档原话，不是我归纳的）',
  "svglbl", BR, size=13)
ops = (('小向量算术', '一次十几个数的加减乘'),
       ('gather / scatter', '按索引取、按索引写'),
       ('排序 · 去重 · 计数 · 直方图', '全是「先看了数据才知道下一步」的活'),
       ('ragged 操作', '每一行长度都不一样的那种'))
for i, (a_, b_) in enumerate(ops):
    yy = FY + 48 + i * 25
    box(724, yy - 12, 8, 8, BR, BR, 2)
    t(742, yy, a_, fill=BR, bold=True)
    t(742 + wpx(a_) + 16, yy, '——&#160;' + b_, fill=GY)
t(724, FY + 145, '⭐ 四条共用一个形状：'
                 '<tspan font-weight="700">要么地址是算出来的，要么形状是不齐的</tspan>'
                 '——&#160;正好是静态编译的机器最难受的两件事。', fill=BR)

# ══ 落点 ══════════════════════════════════════════════════════════
LY = FY + 168                         # 622
box(0, LY, W, 118, "#e6f4ea", GR)
t(16, LY + 24, '⭐ 为什么这台「搬运机」后来被拿去扛集合通信',
  "svglbl", "#0b6b30", size=13)
t(16, LY + 46, 'All-Gather、Reduce-Scatter 这些活，拆开看'
               '<tspan font-weight="700">恰好也是「发一大把 DMA、各自回来、'
               '中间顺手加一下」</tspan>——&#160;'
               '跟 gather 是同一个形状，只是对面从 HBM 换成了别的芯片。',
  fill="#0b6b30")
t(16, LY + 66, '于是就有了这一节最后那件事：<tspan font-weight="700">'
               '把集合通信从 TensorCore 手里接过去，让计算和通信真正并行</tspan>。'
               'Pallas 里这条路是公开的 ——&#160;'
               '<tspan font-family="ui-monospace,monospace">VectorSubcoreMesh</tspan> '
               '让你', fill="#0b6b30")
t(16, LY + 84, '「用写 TensorCore collective 的同一套模型，在 SparseCore 上写 collective」'
               '（JAX Pallas 官方文档原话）。'
               '<tspan font-weight="700">这件事在 XLA 里就是一组 flag 开关。</tspan>',
  fill="#0b6b30")
t(16, LY + 104, '⚠️ 「为什么这套本事也适合卸载 collective」是<tspan font-weight="700">'
                '从形状推的</tspan>——&#160;官方公开了开关和收益，没有公开这段设计理由。',
  fill=GY)

p.append('</svg>')
io.open('fig3-5b.svg', 'w', encoding='utf-8').write('\n'.join(p))
print('fig3-5b ok')
