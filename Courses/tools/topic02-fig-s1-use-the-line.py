# -*- coding: utf-8 -*-
"""图 1-6 · 这条线怎么用 —— 同一道除法，当场量两个，再看融合动了哪一半。

⭐ **为什么要有这一张。** 2026-09-04 Chris 看完全文之后：

    「现在第一节里边为什么含字量那么高？这是我觉得唯一文字多的地方，
      你分析一下那些字有必要吗？还是可以折叠起来？还是可以变成图啊？」

量出来的账：**§1 净正文 1,388 汉字，却只有 1 张图** ——&nbsp;
1,388 字/图，是全篇最差的比例（§3 是 365，§6 是 591）。
而那 1,388 里有 **953 字挤在连排的四个提示框**里。

⛔ 最扎眼的一处：其中一个框的标题就叫
**「先拿这条线判两样东西 ——&nbsp;两行算术，不用图」**，
然后底下写了十五行算术。**它自己宣布不用图，接着把图用文字画了一遍。**
算式、比例、往哪边推 ——&nbsp;这三样全是天生该画的东西。

════════════════════════════════════════════════════════════════
⛔ 三条边界，加东西之前先读
════════════════════════════════════════════════════════════════
① **不画整根强度轴。** `fig2-8`（L300 §2 收尾）已经是「五个算子回到同一根轴」，
   这里再来一根整轴，第二次出现就是一次重复。
   ⭐ 只在第 ③ 栏画一小段轴，因为「融合＝把点往右推」这件事**只有轴画得出来**。
② **不重犯 `fig1-5` 的错。** 那张图被删是因为**提前剧透** ——&nbsp;
   它在结论出来之前就把 312.6 / 312.5 亮了。这一张排在 fig1-4 **之后**，
   312 已经是上一张图的产出，这里只是用它，不是揭它。
③ **不引第四个比方。** 本课比方总预算是三个本体（冷库＝HBM、灶台＝片上暂存、
   刀宽＝指令粒度），而且那套到 §3 才立。这张图**只写算式，不打比方**。

📌 图上每个数都能当场复核：
   · 向量加 bf16：读 a 2 B ＋ 读 b 2 B ＋ 写 y 2 B ＝ 6 B；**1 个 FLOP** → 1/6 ＝ 0.17
   · 312 ÷ 0.167 ＝ 1,872 倍
   · 跑满 8 TB/s → 8e12 ÷ 6 ＝ 1.33 TFLOPS；1.33 ÷ 2,500 ＝ 0.053%（万分之五）
   · 方阵乘 bf16：**2N³ 个 FLOP** ÷ 6N² B ＝ N/3；N/3 ＝ 312 → N ＝ 937；N ＝ 256 → 85.3
   · 片上那条线 64：分子 8,192 FLOP/周期/SM、分母 128 B/周期/SM（32 bank × 4 B），
     完整推导链在 fig1-4 下半段；312 ÷ 64 ＝ 4.9 倍

⛔⛔ 2026-09-05 口径修正：这两处原来都写的是「次乘加」。
   · 向量加 `y = a + b` **一个乘法都没有**，它是 1 次加法 ＝ 1 个 FLOP；
   · 方阵乘的乘加数是 N³，**2N³ 是 FLOP 数**（一次 MAC 记 2 个 FLOP）。
   两个数本身都对（强度 0.17 与 N/3 都算得出来），错的是那个词 ——
   而本课在 §1.1 折叠里自己立过「一次乘加 ＝ 2 FLOP」的规矩，
   §3.7 更是逐个数 MAC（B200 606,208 个/周期）。**这张图是全课唯一破口径的地方。**
⭐ 形状：**数对了不代表口径对了。** 单位词错了不会让任何一步算术露馅，
   它只在读者拿这个词去别处套的时候才爆 —— 而这门课通篇在教读者这么套。
"""
import io

BL, OR, GR, RD, GY, YL, PU = ("#1a73e8", "#e8710a", "#1e8e3e", "#d93025",
                              "#5f6368", "#f9ab00", "#8430ce")
GY2 = "#80868b"
W = 1400
p = []


def t(x, y, s, cls="svgsm", fill=None, size=None, anchor=None):
    p.append('<text class="%s" x="%s" y="%s"%s%s%s>%s</text>' % (
        cls, x, y, ' fill="%s"' % fill if fill else '',
        ' text-anchor="%s"' % anchor if anchor else '',
        ' style="font-size:%spx"' % size if size else '', s))


def box(x, y, w, h, fill="#fff", stroke="#dadce0", r=10, sw=1.0):
    p.append('<rect x="%s" y="%s" width="%s" height="%s" rx="%s" fill="%s"'
             ' stroke="%s" stroke-width="%s"/>' % (x, y, w, h, r, fill, stroke, sw))


TOP = 92
CH = 384                                  # 三栏卡片高
CY = TOP + 34
CW, GAP = 442, 17
CX = [0, CW + GAP, 2 * (CW + GAP)]
BY = CY + CH + 26                         # 底带
H = BY + 122

p.append('<svg viewBox="0 0 %d %d" width="100%%" role="img" aria-label="'
         '同一道除法用三次：向量加算出 0.17，方阵乘算出 N 除以 3，'
         '算子融合只把分母做小、把点往右推，而落点是片上那条 64 的线">' % (W, H))
p.append('<defs><marker id="aR" viewBox="0 0 10 10" refX="9" refY="5" '
         'markerWidth="6" markerHeight="6" orient="auto">'
         '<path d="M0 0 L10 5 L0 10 z" fill="%s"/></marker></defs>' % GR)

# ══════════ 顶部：那句最值钱的算法 ══════════
t(0, 16, '这条线怎么用 ——&#160;<tspan font-weight="700">同一道除法，当场量两个，'
         '再看融合动了哪一半</tspan>', "svglbl", "#202124", 14)

# ⛔ 顶带必须两行：第一版把「算法一句话」和右上那句注解排在同一行，
#    两串字直接叠上（本图第一次 build 就被 topic02-lint-layout.py 抓到）。
box(0, 30, W, 62, "#e8f0fe", BL, 8, 1.4)
t(16, 55, '算法只有一句话：', "svglbl", "#174ea6", 13)
t(126, 55, '数它搬几个字节　·　数它算几次　·　除一下', "svglbl", "#174ea6", 17)
t(506, 55, '——&#160;得到的就是这个算子的<tspan font-weight="700">算术强度</tspan>；'
           '跟 1.1 那张规格表算出来的 <tspan font-weight="700">312</tspan> 比大小，'
           '就知道它卡在哪一边。', None, "#174ea6")
t(16, 79, '⭐ 这句比 312 本身值钱 ——&#160;312 会随硬件换代变，这个算法不会。'
          '下面三栏就是它用三次的样子。', None, "#5f6368")

# ══════════ ① 向量加 ══════════
x = CX[0]
box(x, CY, CW, CH, "#fef7f6", RD, 10, 1.6)
t(x + 18, CY + 28, '① 向量加', "svglbl", RD, 16)
t(x + 118, CY + 28, '<tspan font-family="ui-monospace,monospace">y = a + b</tspan>'
                    '　bf16', "svglbl", RD, 13)

box(x + 18, CY + 44, CW - 36, 112, "#fff", "#f2c7c2", 8)
for i, (a, bb) in enumerate([('读 a', '2 字节'), ('读 b', '2 字节'), ('写 y', '2 字节')]):
    t(x + 34, CY + 68 + i * 22, a, None, GY)
    t(x + 148, CY + 68 + i * 22, bb, "svglbl", "#202124", 12, "end")
p.append('<path d="M%d %d h114" stroke="#f2c7c2"/>' % (x + 34, CY + 140))
t(x + 34, CY + 152, '搬', None, GY)
t(x + 148, CY + 152, '6 字节', "svglbl", RD, 13, "end")
t(x + 190, CY + 72, '算', None, GY)
t(x + 190, CY + 112, '1', "svglbl", RD, 30)
t(x + 218, CY + 112, '个 FLOP', "svglbl", RD, 13)
t(x + 190, CY + 142, '强度 ＝ 1 ÷ 6 ＝', None, GY)
t(x + 322, CY + 142, '0.17', "svglbl", RD, 17)

box(x + 18, CY + 168, CW - 36, 74, "#fff", "#f2c7c2", 8)
t(x + 34, CY + 192, '比 312 低了', None, GY)
t(x + 128, CY + 192, '一千八百多倍', "svglbl", RD, 14)
t(x + 34, CY + 214, '跑满 GB200 那 8 TB/s，也只有 <tspan font-weight="700">1.3 TFLOPS</tspan>', None, "#202124")
t(x + 34, CY + 232, '峰值 2,500 —— <tspan font-weight="700">用掉万分之五</tspan>', None, "#202124")

box(x + 18, CY + 254, CW - 36, 46, "#fce8e6", None, 8, 0)
t(x + 34, CY + 274, '买更贵的卡能让它变快吗？', "svglbl", "#a50e0e", 13)
t(x + 34, CY + 292, '不能。它压根不在算力那条路上。', "svglbl", "#a50e0e", 14)
t(x + 18, CY + 330, '⭐ 这就是「带宽受限」的样子：分母大得离谱，', None, GY)
t(x + 18, CY + 348, '　 分子小得可怜。<tspan font-weight="700">全世界的算力都救不了它。</tspan>', None, GY)

# ══════════ ② 方阵乘 ══════════
x = CX[1]
box(x, CY, CW, CH, "#f5faf6", GR, 10, 1.6)
t(x + 18, CY + 28, '② 方阵乘', "svglbl", GR, 16)
t(x + 118, CY + 28, 'N × N × N　bf16', "svglbl", GR, 13)

box(x + 18, CY + 44, CW - 36, 112, "#fff", "#c3e2cc", 8)
t(x + 34, CY + 70, '算', None, GY)
t(x + 70, CY + 74, '2N³', "svglbl", GR, 22)
t(x + 128, CY + 70, '个 FLOP', None, GY)
t(x + 34, CY + 108, '搬', None, GY)
t(x + 70, CY + 112, '6N²', "svglbl", GR, 22)
t(x + 128, CY + 108, '字节（三个矩阵进出）', None, GY)
t(x + 34, CY + 140, '约一下，强度 ＝', None, GY)
t(x + 176, CY + 140, 'N ÷ 3', "svglbl", GR, 17)
t(x + 252, CY + 140, '——&#160;只跟边长有关', None, GY)

box(x + 18, CY + 168, CW - 36, 74, "#fff", "#c3e2cc", 8)
for i, (n, v, c, note) in enumerate([('N ＝ 256', '85', RD, '还在带宽那一侧'),
                                     ('N ≈ 1,000', '312', GR, '这才刚够到线上')]):
    y = CY + 194 + i * 30
    t(x + 34, y, n, "svglbl", "#202124", 13)
    t(x + 140, y, '→', None, GY2)
    t(x + 168, y + 3, v, "svglbl", c, 17)
    t(x + 224, y, note, None, c)

box(x + 18, CY + 254, CW - 36, 46, "#e6f4ea", None, 8, 0)
t(x + 34, CY + 274, '「矩阵乘＝算力受限」是个错觉。', "svglbl", "#0d652d", 13)
t(x + 34, CY + 292, '它得够大才算数。', "svglbl", "#0d652d", 14)
t(x + 18, CY + 330, '⭐ 两栏合起来才是这条线的用法：', None, GY)
t(x + 18, CY + 348, '　 <tspan font-weight="700">不看算子叫什么名字，只看它的除法结果。</tspan>', None, GY)

# ══════════ ③ 融合：只动分母 ══════════
x = CX[2]
box(x, CY, CW, CH, "#fffdf5", YL, 10, 1.6)
t(x + 18, CY + 28, '③ 那怎么救', "svglbl", "#7a5000", 16)
t(x + 150, CY + 28, '——&#160;只有一条路：把分母做小', "svglbl", "#7a5000", 13)

box(x + 18, CY + 44, CW - 36, 92, "#fff", "#f0dfa8", 8)
t(x + 34, CY + 68, '把连着的几个算子合成一个 kernel，', None, "#202124")
t(x + 34, CY + 86, '让中间结果<tspan font-weight="700">不落 HBM</tspan> ——&#160;'
                   '这就是<tspan font-weight="700">算子融合</tspan>。', None, "#202124")
t(x + 34, CY + 112, '⭐ 它<tspan font-weight="700">一个 FLOP 都没省</tspan>：'
                    '分子分毫不动，', None, "#a50e0e")
t(x + 34, CY + 128, '　 变的只有分母。', None, "#a50e0e")

# 小段轴：融合＝把点往右推。**只画这一段，不画整根轴**（见文件头 ①）
ax, aw, ay = x + 40, CW - 96, CY + 176
p.append('<path d="M%d %d h%d" stroke="#9aa0a6" stroke-width="1.2"/>' % (ax, ay, aw))
x312 = ax + aw * 0.66
p.append('<path d="M%d %d v-30" stroke="%s" stroke-width="2.2" stroke-dasharray="5 4"/>'
         % (x312, ay + 6, YL))
t(x312, ay + 22, '312', "svglbl", "#7a5000", 12, "middle")
t(ax, ay + 22, '带宽受限', None, RD)
t(ax + aw, ay + 22, '算力受限', None, GR, None, "end")
p.append('<circle cx="%d" cy="%d" r="6" fill="%s"/>' % (ax + 26, ay - 14, RD))
p.append('<path d="M%d %d H%d" stroke="%s" stroke-width="2.4" marker-end="url(#aR)"/>'
         % (ax + 38, ay - 14, x312 + 46, GR))
t(ax + 150, ay - 24, '融合把它往右推', "svglbl", GR, 13, "middle")

box(x + 18, CY + 212, CW - 36, 88, "#fff", "#f0dfa8", 8)
t(x + 34, CY + 236, '⚠️ 但「不落 HBM」不等于「不落地」', "svglbl", "#a50e0e", 13)
t(x + 34, CY + 256, '它被塞进片上暂存，而片上<tspan font-weight="700">有自己的线：</tspan>', None, "#202124")
t(x + 34, CY + 282, '312', "svglbl", YL, 20)
t(x + 84, CY + 280, '→', "svglbl", GY2, 15)
t(x + 116, CY + 282, '64', "svglbl", PU, 20)
t(x + 158, CY + 278, '门槛只降 <tspan font-weight="700">4.9 倍</tspan>', None, "#202124")
t(x + 158, CY + 294, '片上流量却能涨<tspan font-weight="700">几十倍</tspan>', None, "#202124")

t(x + 18, CY + 330, '⭐ 所以融合做过头会在新的这条线上撞墙 ——&#160;', None, GY)
t(x + 18, CY + 348, '　 <tspan font-weight="700">不是门槛变高，是分母变大。</tspan>', None, GY)

# ══════════ 底带 ══════════
box(0, BY, W, 104, "#f8f9fa", "#dadce0", 10, 1.2)
t(20, BY + 26, '接下来每一节都在这条线上', "svglbl", "#202124", 14)
# ⛔ 三列各 466 宽，第一版的句子写到 500 多，直接压住下一列的 §。
#    ⭐ 定了列宽就得按列宽写句子 —— 这是本图第二处被 lint 抓到的撞车。
for i, seg in enumerate([
        '<tspan font-weight="700">§2</tspan>　融合前后差一千倍 ——&#160;这条线的主例',
        '<tspan font-weight="700">§3</tspan>　同一个融合，两块硬件上各是什么样',
        '<tspan font-weight="700">§5</tspan>　融合这件事，<tspan font-weight="700">该由谁来决定</tspan>']):
    t(20 + i * 452, BY + 54, seg, None, "#202124")
t(20, BY + 78, '⛔ 图上每个数都能当场复核：6 ＝ 2＋2＋2；312 ÷ 0.167 ＝ 1,872；'
               '8e12 ÷ 6 ＝ 1.33 TFLOPS，占 2,500 的 0.053%', None, GY2)
t(20, BY + 94, '　 2N³ ÷ 6N² ＝ N/3，N/3 ＝ 312 → N ＝ 937，N ＝ 256 → 85.3；'
               '312 ÷ 64 ＝ 4.9', None, GY2)

p.append('</svg>')
io.open('fig1-6.svg', 'w', encoding='utf-8').write('\n'.join(p))
print('fig1-6 ok  %d×%d' % (W, H))
