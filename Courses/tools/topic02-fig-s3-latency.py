# -*- coding: utf-8 -*-
"""图 3-7 · 延迟阶梯 ——&nbsp;同一条 load 指令，30 拍到 659 拍。

⭐ **为什么要有这一张。** 2026-09-06 现场追问：

    「我的 warp 需要一块数据的时候，L1 里没有，但是查询 lookup 是需要时间的呀，
      然后再去 L2 里 lookup，又需要时间，然后 L2 又去 HBM 里边搬，又需要时间，
      这一串下来要耗费多少时间啊？……那每一种情况耗费的时间还不一边长，
      这个东西就太不好规划了呀，这也不好像 TPU 那样事先编成 Graph 啊。」

⭐ **这一问问到了整门课的地基**，所以它必须是主线，不能折叠
（原话：「别都折起来，不怕时间长，我自己可以去跳着讲」）。

📌 **§3.2b 讲容量，§3.2c 讲带宽，这张图补的是第三个轴：延迟。**
   三个轴各自给出方向相反的结论，合起来才是完整的片上存储图景。

⛔ **一个必须写在图上的口径**：这组数出自 H100（GH100）与消费级 Blackwell
   （GB203）的公开微基准，**B200 没有同类公开数据**。量级与形状是这一代通用的，
   但不要把它当成 B200 的实测值。

⛔ **不要把三段延迟相加。** 659 是一次全 miss 的**端到端**耗时，
   L1 与 L2 那两次查找已经含在里面了 ——&nbsp;这正是提问里最容易走偏的一步。

📌 出处：arXiv 2507.10789《Dissecting the NVIDIA Blackwell Architecture with
   Microbenchmarks》——&nbsp;指针追逐微基准，L1 30–40 拍、GH100 L2 273 拍、
   两个分区打满 508 拍、全局内存 658.7 拍；GB203 L2 358 拍、全局 876.7 拍。
"""
import io

BL, OR, GR, RD, GY = "#1a73e8", "#e8710a", "#1e8e3e", "#d93025", "#5f6368"
BR = "#7a5000"
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
    """粗估纯文本像素宽 —— CJK 约一个字号宽，ASCII 约一半多。

    ⛔ 别用 `len() * 常数`：中英混排的短语上必然偏小，两段文字会撞在一起。
    """
    n = 0.0
    for ch in s:
        n += 1.0 if ord(ch) > 0x2E80 else 0.55
    return int(n * size)


def box(x, y, w, h, fill="#fff", stroke="#dadce0", r=8, sw=1):
    p.append('<rect x="%d" y="%d" width="%d" height="%d" rx="%d" fill="%s" '
             'stroke="%s" stroke-width="%s"/>' % (x, y, w, h, r, fill, stroke, sw))


p.append('<svg viewBox="0 0 %d 700" width="100%%" role="img" aria-label="'
         'GPU 访存延迟阶梯：L1 命中三四十拍、L2 命中 273 拍、一路到显存 659 拍；'
         '同一条 load 指令相差二十倍，而 TPU 那边这个轴上只有常数">' % W)

t(0, 17, '延迟阶梯 ——&#160;<tspan font-weight="700">同一条 load 指令，'
         '30 拍到 659 拍</tspan>', "svglbl", "#202124", size=14)
t(0, 37, '§3.2b 比的是<tspan font-weight="700">容量</tspan>，'
         '§3.2c 比的是<tspan font-weight="700">带宽</tspan>。'
         '这是第三个轴：<tspan font-weight="700">等多久</tspan>'
         '——&#160;而三个轴给出的结论方向并不一致。')

# ══ 柱状阶梯（占满整幅宽）═══════════════════════════════════════════
# ⛔ 2026-09-06 初版把柱状图和 TPU 那栏左右并排，结果最长那两根柱子的
#    说明文字直接顶进右栏（几何 lint 抓到「GB203……」跟右栏撞车），
#    面板底下两行还溢出了灰底。
#    ⭐ 教训：**带右侧标注的横向条形图不能跟别的东西并排** ——
#      条长 + 数值 + 说明是三段依次右移的东西，它天然要吃满整行宽度。
#      现在改成：柱状图独占一行，TPU 与对策两栏放到它下面。
PY, PH = 50, 358
box(0, PY, W, PH, "#f8f9fa")
t(16, PY + 24, 'NVIDIA GPU：一次取数可能落在哪一档', "svglbl", "#202124", size=13)
t(16, PY + 42, '指针追逐微基准实测，单位是<tspan font-weight="700">时钟周期（拍）</tspan>。'
               '前五行是 H100，最后一行是消费级 Blackwell 做对照。', fill=GY)

X0, BARMAX, SCALE = 250, 700, 900.0     # 900 拍 → 700 px
ROWS = (
    ('寄存器',              1,   '直接就在手里', GR, '≈ 1 拍'),
    ('共享内存 ／ L1 命中', 35,  'H100 与 Blackwell 都在这一档', GR, '30–40 拍'),
    ('L2 命中',            273, '<tspan font-weight="700">已经比 L1 慢了七八倍</tspan>',
                                OR, '273 拍'),
    ('L2 两个分区都打满',   508, '分区设计在拥挤时优势消失', OR, '508 拍'),
    ('一路到 HBM',         659, '<tspan font-weight="700">全 miss 的端到端耗时</tspan>',
                                RD, '659 拍'),
    ('（对照）消费级 GDDR7', 877, '换一种显存这个数就变了', GY, '877 拍'),
)
RY = PY + 58
for i, (name, cyc, note, col, lab) in enumerate(ROWS):
    y = RY + i * 40
    t(16, y + 17, name, fill="#202124", bold=True)
    w = max(3, int(cyc / SCALE * BARMAX))
    box(X0, y + 4, w, 20, col, col, 3)
    t(X0 + w + 10, y + 19, lab, fill=col, bold=True)
    t(X0 + w + 10 + wpx(lab) + 12, y + 19, '——&#160;' + note, fill=GY)

t(16, RY + 6 * 40 + 22, '⛔ <tspan font-weight="700">不要把这几段相加。</tspan>'
                        '659 拍是<tspan font-weight="700">一次全 miss 的端到端耗时</tspan>'
                        '——&#160;L1 和 L2 那两次扑空已经含在里面了。'
                        '<tspan font-weight="700">该记住的不是「加起来多少」，'
                        '是「同一条指令在 30 到 659 之间摆」。</tspan>', fill=RD)
t(16, RY + 6 * 40 + 42, '⚠️ B200 没有同类公开数据。'
                        '<tspan font-weight="700">量级与形状是这一代通用的，'
                        '但别把它当成 B200 的实测值。</tspan>'
                        '　出处：arXiv 2507.10789 的指针追逐微基准。', fill=GY)

# ══ 下半左：TPU 那一侧 ═════════════════════════════════════════════
BY2 = PY + PH + 16                    # 424
BH2 = 178
box(0, BY2, 680, BH2, "#e6f4ea", GR)
t(16, BY2 + 24, 'TPU：这个轴上只有常数', "svglbl", "#0b6b30", size=13)
t(16, BY2 + 44, '<tspan font-weight="700">不是数字保密 ——&#160;是没有'
                '「可能命中、可能没命中」这一档。</tspan>', fill="#0b6b30")
for i, (a_, b_) in enumerate((
        ('VMEM 访问',
         '编译器显式管的暂存。<tspan font-weight="700">数在那儿不是因为运气好，</tspan>'),
        ('', '<tspan font-weight="700">是因为编译器自己发的那条 DMA 把它放在那儿</tspan>'),
        ('HBM → VMEM 的 DMA',
         '这一段当然也有延迟，但它是<tspan font-weight="700">一次被安排好的搬运</tspan>：'),
        ('', '延迟已知，等待点也是编译器插的'))):
    y = BY2 + 70 + i * 19
    if a_:
        box(16, y - 8, 7, 7, GR, GR, 2)
        t(30, y, a_, fill="#0b6b30", bold=True)
        t(30 + wpx(a_) + 12, y, '——&#160;' + b_, fill="#0b6b30")
    else:
        t(30, y, b_, fill="#0b6b30")
box(16, BY2 + 152, 648, 1, GR, GR, 0)
t(16, BY2 + 170, '⭐ <tspan font-weight="700">所以「缓存缺失」在 TPU 上不存在'
                 '——&#160;不是缓存做得好，是根本没有硬件缓存。</tspan>'
                 '缓存是一台猜测机器；<tspan font-weight="700">编译期排好的机器不需要猜</tspan>。',
  fill="#0b6b30")

# ══ 下半右：GPU 的三条对策 ═════════════════════════════════════════
box(700, BY2, W - 700, BH2, "#fef7e0", OR)
t(716, BY2 + 24, '那怎么规划？——&#160;GPU 根本不规划', "svglbl", BR, size=13)
t(716, BY2 + 44, '<tspan font-weight="700">这二十倍的差距你在源码里看不出来</tspan>'
                 '——&#160;同一条 load，长得一模一样。'
                 '它的三条对策没有一条在试图算准：', fill=BR)
for i, (a_, b_) in enumerate((
        ('① 超量线程',
         '一个 SM 驻留几十个 warp，卡住就切下一个'),
        ('', '<tspan font-weight="700">延迟不是被消除，是被别人的活盖住的</tspan>'),
        ('② 记分板 ＋ 动态发射',
         '运行时盯着操作数到没到，到了才发'),
        ('③ 缓存本身',
         '它就是一台<tspan font-weight="700">猜测机器</tspan>，赌你还会再用一次'))):
    y = BY2 + 70 + i * 19
    if a_:
        t(716, y, a_, fill=BR, bold=True)
        t(716 + wpx(a_) + 12, y, '——&#160;' + b_, fill=GY)
    else:
        t(716 + wpx('① 超量线程') + 12, y, b_, fill=GY)
box(716, BY2 + 152, W - 700 - 32, 1, OR, OR, 0)
t(716, BY2 + 170, '⚠️ <tspan font-weight="700">调优在两边动的旋钮一样都不挨着</tspan>：'
                  'GPU 调「怎么让别的活足够多」，TPU 调「怎么让编译器排得开」。', fill=BR)

# ══ 落点 ══════════════════════════════════════════════════════════
LY = BY2 + BH2 + 16
box(0, LY, W, 62, "#e8f0fe", BL)
t(16, LY + 25, '⭐ 一句话对照', "svglbl", "#174ea6", size=13)
t(120, LY + 25, '<tspan font-weight="700">GPU 用「总有别的活可干」来盖住不确定的延迟；'
                'TPU 用「把不确定性删掉」来避免它。</tspan>'
                '——&#160;一个是统计学的答案，一个是确定性的答案。', fill="#174ea6")
t(16, LY + 47, '这也是这门课从第 1 节起一直在说的那件事，'
               '<tspan font-weight="700">第一次落到了「拍」这个单位上</tspan>。'
               '⚠️ 但别读成「两边越走越远」——&#160;'
               '现代追峰值的 CUDA kernel 用异步拷贝＋显式 barrier 自己管双缓冲，'
               '<tspan font-weight="700">那就是在 GPU 上手写一个静态调度</tspan>。', fill="#174ea6")

p.append('</svg>')
io.open('fig3-7.svg', 'w', encoding='utf-8').write('\n'.join(p))
print('fig3-7 ok')
