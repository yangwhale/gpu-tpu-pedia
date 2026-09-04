# -*- coding: utf-8 -*-
"""图 3-6 · 一块矩阵乘是怎么在 MXU 上跑完的 —— 三种驻留方式、切块、乒乓装载。

⭐ **为什么要有这一张。** 2026-09-04 Chris 的原话：

    「你把这一段给我画成图……把这个算术怎么切块，就用这个典型的场景切一遍。
      这样我们不就知道右边怎么流进去、然后怎么驻留、左边怎么流进去、
      然后左边怎么把这个 128K 切成 2048 一份的这种。然后因为这个份数还决定
      右矩阵重复传多少次。然后再把那个乒乓矩阵的那种搬运方式也说一下，
      **这个地方好好讲讲，因为从来没有人把这个地方讲明白过。**
      然后还有那种三种 MXU 的使用方式，左驻流、右驻流和中间累加驻流。」

⛔ 所以这张图有四条带，缺一条都不完整：
   ① 三种驻留方式（左驻 IS ／ 右驻 WS ／ 中间累加驻 OS）——&nbsp;
      **它们的区别只有一个：谁不动。** 而「谁不动」直接决定了哪两维是空间。
   ② 用真尺寸切块（Q 投影 M=128K, K=7168, N=1536）
   ③ **乒乓装载**的时间线 —— 这一条正是「从来没人讲明白」的那一段
   ④ 段数 ↔ 右矩阵重搬次数的账

📌 数字全部可当场复核：
   7168/256=28、1536/256=6、28×6=168；
   W bf16 ＝ 7168×1536×2B ＝ 21 MiB；
   一段 2048 行的累加器 ＝ 2048×256×4B ＝ 2 MiB；
   B 进阵列总次数 ＝ 168 × (131072/段长)。
   装一块权重 256 周期 ——&nbsp;出自 TPU 初代论文（双缓冲用来藏它）。

⚠️ 「藏得住的条件是段长 > 256」是把两个公开数字放一起推的，不是官方结论。
"""
import io

BL, OR, GR, RD, GY = "#1a73e8", "#e8710a", "#1e8e3e", "#d93025", "#5f6368"
PU = "#8430ce"
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


def box(x, y, w, h, fill="#fff", stroke="#dadce0", r=8):
    p.append('<rect x="%d" y="%d" width="%d" height="%d" rx="%d" fill="%s" stroke="%s"/>'
             % (x, y, w, h, r, fill, stroke))


def arrow(x1, y1, x2, y2, color=GY, dash=""):
    p.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" stroke-width="2"'
             ' marker-end="url(#aG)"%s/>'
             % (x1, y1, x2, y2, color, ' stroke-dasharray="%s"' % dash if dash else ''))


p.append('<svg viewBox="0 0 %d 1080" width="100%%" role="img" aria-label="'
         '三种脉动阵列驻留方式、Q 投影的切块、乒乓装载时间线，'
         '以及分段数与右矩阵重搬次数的关系">' % W)
p.append('<defs><marker id="aG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6"'
         ' markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="%s"/>'
         '</marker></defs>' % GY)

t(0, 17, '一块矩阵乘是怎么在 MXU 上跑完的 ——&#160;<tspan font-weight="700">'
         '谁驻留、谁流动、以及权重什么时候偷偷装进去</tspan>',
  "svglbl", "#202124", size=14)
t(0, 37, '用一个真尺寸走完：Q 投影 <tspan font-weight="700">M ＝ 128K　K ＝ 7168　'
         'N ＝ 1536</tspan>，阵列 256 × 256')

# ══ 带一：三种驻留方式 ═══════════════════════════════════════════════
YA = 58
t(0, YA + 14, '① 三种用法，区别只有一个：<tspan font-weight="700">谁不动</tspan>',
  "svglbl", "#202124", size=13)

CW = 448
CARDS = [
    ("左驻留（IS）", "格子里放 <tspan font-weight=\"700\">A</tspan> 的一块",
     "空间 ＝ M × K", "时间 ＝ N", BL, "A", "B 从上面流入", "C 流出"),
    ("右驻留（WS）　★ TPU 走这条", "格子里放 <tspan font-weight=\"700\">B</tspan> 的一块",
     "空间 ＝ K × N", "时间 ＝ M", OR, "B", "A 从左边流入", "C 向下累出"),
    ("中间累加驻留（OS）", "格子里放 <tspan font-weight=\"700\">C</tspan> 的一块",
     "空间 ＝ M × N", "时间 ＝ K", GR, "C", "A 左入 ＋ B 上入", "算完整块排出"),
]
for i, (name, hold, sp, tm, col, letter, flow, out) in enumerate(CARDS):
    x = i * (CW + 28)
    box(x, YA + 26, CW, 186, "#fff", col)
    t(x + 14, YA + 48, name, "svglbl", col, size=12)
    t(x + 14, YA + 66, hold, fill=col)
    # 小阵列
    ax, ay, aw = x + 14, YA + 78, 112
    box(ax, ay, aw, aw, "#f1f3f4", col, 4)
    for g in range(1, 4):
        p.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#dadce0"/>'
                 % (ax + g * aw // 4, ay, ax + g * aw // 4, ay + aw))
        p.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#dadce0"/>'
                 % (ax, ay + g * aw // 4, ax + aw, ay + g * aw // 4))
    t(ax + aw // 2, ay + aw // 2 + 8, letter, "svglbl", col, size=26, anchor="middle")
    t(ax + aw // 2, ay + aw + 15, '不动', fill=col, anchor="middle")
    # 右侧文字
    tx = ax + aw + 18
    t(tx, ay + 18, '流进来的：', fill=GY)
    t(tx, ay + 36, flow, fill="#202124", bold=True)
    t(tx, ay + 60, '出去的：', fill=GY)
    t(tx, ay + 78, out, fill="#202124")
    t(tx, ay + 106, sp, fill=col, bold=True)
    t(tx, ay + 124, tm, fill=col, bold=True)

box(0, YA + 224, W, 46, "#e6f4ea", GR)
t(16, YA + 246, '⭐ 三种做法只是在换「哪一维当时间」——&#160;'
                '<tspan font-weight="700">空间那两维不够大就有格子空转，'
                '时间那一维不够长就摊不动装权重的钱</tspan>。'
                '<tspan fill="%s">下面两条带讲的都是右驻留（TPU 的那条）。</tspan>' % GY,
  fill="#0b6b30")

# ══ 带二：切块 ══════════════════════════════════════════════════════
YB = YA + 290
t(0, YB + 14, '② 切块：<tspan font-weight="700">右矩阵切 168 块，左矩阵切 64 段</tspan>',
  "svglbl", "#202124", size=13)

# 右矩阵 W：28 行 × 6 列
box(0, YB + 26, 430, 250, "#fff", OR)
t(14, YB + 48, '右矩阵 W ＝ [7168, 1536]', "svglbl", "#7a5000", size=12)
t(14, YB + 66, 'bf16 一共 21 MiB ——&#160;<tspan font-weight="700">一次就全进 VMEM 了</tspan>',
  fill="#7a5000")
gx, gy, cw, ch = 20, YB + 78, 20, 5
for r in range(28):
    for c in range(6):
        box(gx + c * (cw + 3), gy + r * (ch + 2), cw, ch,
            "#fce8b2" if (r + c) % 2 == 0 else "#fdf1cf", OR, 1)
t(gx + 6 * (cw + 3) + 20, gy + 26, 'K 方向 7168 ÷ 256', fill="#7a5000")
t(gx + 6 * (cw + 3) + 20, gy + 44, '＝ 28 块', "svglbl", "#7a5000", size=13)
t(gx + 6 * (cw + 3) + 20, gy + 74, 'N 方向 1536 ÷ 256', fill="#7a5000")
t(gx + 6 * (cw + 3) + 20, gy + 92, '＝ 6 块', "svglbl", "#7a5000", size=13)
t(gx + 6 * (cw + 3) + 20, gy + 124, '一共 168 块', "svglbl", RD, size=14)
t(gx + 6 * (cw + 3) + 20, gy + 142, '每块 256×256', fill="#7a5000")

# 左矩阵 X
box(452, YB + 26, 430, 250, "#fff", BL)
t(466, YB + 48, '左矩阵 X ＝ [131072, 7168]', "svglbl", "#174ea6", size=12)
t(466, YB + 66, '<tspan font-weight="700">M 方向按 2048 行切一段</tspan>'
                '——&#160;切出 64 段', fill="#174ea6")
sx, sy = 472, YB + 82
for r in range(16):
    box(sx, sy + r * 11, 260, 8, "#e8f0fe" if r % 2 == 0 else "#d8e6fd", BL, 1)
t(sx, sy + 16 * 11 + 14, '……共 64 段', fill="#174ea6")
t(sx + 276, sy + 20, '一段 2048 行', "svglbl", "#174ea6", size=13)
t(sx + 276, sy + 40, '× 256 列（K 的一片）', fill="#174ea6")
t(sx + 276, sy + 68, '片上累加器', fill=GY)
t(sx + 276, sy + 86, '2048 × 256 × 4 B', fill="#174ea6")
t(sx + 276, sy + 104, '＝ 2 MiB', "svglbl", "#174ea6", size=13)

# 为什么不能一口气
box(904, YB + 26, W - 904, 250, "#fce8e6", RD)
t(918, YB + 48, '⛔ 为什么 M 不能一口气流完', "svglbl", "#a50e0e", size=12)
t(918, YB + 74, '那样每个 n 块的累加器要', fill="#a50e0e")
t(918, YB + 94, '131072 × 256 × 4 B', "svglbl", "#a50e0e", size=13)
t(918, YB + 114, '＝ 128 MiB', "svglbl", RD, size=18)
t(918, YB + 138, '——&#160;<tspan font-weight="700">比整块 VMEM（64 MB）还大</tspan>。',
  fill="#a50e0e")
t(918, YB + 166, '所以 M 必须切段。', "svglbl", "#a50e0e", size=13)
t(918, YB + 186, '而切段就<tspan font-weight="700">必然要重搬右矩阵</tspan>。',
  fill="#a50e0e")
t(918, YB + 214, '⚠️ 这不是实现不好，', fill=GY)
t(918, YB + 232, '是账本身如此。', fill=GY)

# ══ 带三：乒乓装载 ══════════════════════════════════════════════════
YC = YB + 296
t(0, YC + 14, '③ <tspan font-weight="700">乒乓装载</tspan>：每个格子有两套权重寄存器，'
              '<tspan font-weight="700">一套在算、一套在装</tspan>',
  "svglbl", "#202124", size=13)

box(0, YC + 26, W, 196, "#f8f9fa")
# 甘特：两行
g0, g1 = YC + 60, YC + 108
t(14, g0 + 14, '阵列在算', "svglbl", "#202124", size=12)
t(14, g1 + 14, '后台在装', "svglbl", "#202124", size=12)
# ⛔ 第一版 UNIT＝96 太窄，「算 B0 · 2048 周期」这行字比格子还长，
#    六个格子全压在一起。加宽到 150 并把标签缩短；装载块改成贴着
#    对应计算块的**起点**画，一眼能看出「装下一块」和「算这一块」是并行的。
X0, UNIT = 150, 190
t(X0, g0 - 22, '⛔ 只有第一块要干等 256 周期，之后全藏住', fill=RD)
# 第一块：阵列空转，后台在装 B0
box(X0, g1 - 4, 34, 22, "#f4c7c3", RD, 3)
t(X0 + 17, g1 + 12, 'B0', fill="#a50e0e", anchor="middle", cls="svgsm")
p.append('<rect x="%d" y="%d" width="34" height="22" rx="3" fill="none" stroke="%s"'
         ' stroke-dasharray="3 2"/>' % (X0, g0 - 4, RD))
t(X0 + 17, g0 + 12, '空', fill=RD, anchor="middle", cls="svgsm")

for k in range(6):
    x = X0 + 40 + k * UNIT
    box(x, g0 - 4, UNIT - 6, 22, "#e8f0fe", BL, 3)
    t(x + (UNIT - 6) // 2, g0 + 12, '算 B%d（2048 周期）' % k,
      fill="#174ea6", anchor="middle", cls="svgsm")
    if k < 5:
        box(x, g1 - 4, 34, 22, "#fce8b2", OR, 3)
        t(x + 17, g1 + 12, 'B%d' % (k + 1), fill="#7a5000", anchor="middle", cls="svgsm")
        t(x + 40, g1 + 12, '256 周期', fill=GY, cls="svgsm")
        p.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s"'
                 ' stroke-dasharray="2 3"/>' % (x, g0 + 20, x, g1 - 6, "#bdc1c6"))
t(X0 + 40 + 6 * UNIT - 40, g0 + 12, '……', fill=GY)

t(14, YC + 158, '⭐ <tspan font-weight="700">装一块权重要 256 个周期</tspan>'
                '（TPU 初代论文原话），而流一段 2048 行要 2048 个周期 ——&#160;'
                '<tspan font-weight="700">装的那 256 完全藏在算的那 2048 里面</tspan>。',
  fill="#202124")
t(14, YC + 178, '⚠️ 所以藏得住的条件很直白：<tspan font-weight="700">'
                '一段的行数要大于 256</tspan>。这里 2048 是它的 8 倍，稳。'
                '<tspan fill="%s">（这一条是把两个公开数字放一起推的，不是官方结论。）</tspan>' % GY,
  fill="#202124")
t(14, YC + 202, '⛔ 反例就在隔壁：注意力转置之后 M ＝ 128 ——&#160;'
                '<tspan font-weight="700">连 256 这个门槛都够不着，乒乓就藏不住了</tspan>。',
  fill=RD)

# ══ 带四：段数 ↔ 重搬次数 ════════════════════════════════════════════
YD = YC + 246
t(0, YD + 14, '④ 段切得越碎，右矩阵就要<tspan font-weight="700">重搬越多次</tspan>'
              '——&#160;而段能开多长，由片上累加器顶住',
  "svglbl", "#202124", size=13)

box(0, YD + 26, 820, 150, "#fff")
COLS = (16, 190, 380, 600)
for cx, h in zip(COLS, ('一段多少行', '片上累加器', 'B 进阵列的总次数', '乒乓藏得住吗')):
    t(cx, YD + 50, h, "svglbl", GY, size=12)
p.append('<line x1="0" y1="%d" x2="820" y2="%d" stroke="#dadce0"/>' % (YD + 58, YD + 58))
ROWS = (("256", "0.25 MiB", "86,016", "⛔ 藏不住（不大于 256）", RD),
        ("512", "0.5 MiB", "43,008", "能", GY),
        ("2048", "2 MiB", "10,752", "能", GR),
        ("8192", "8 MiB", "2,688", "能", GY))
for i, (a, b, c, d, col) in enumerate(ROWS):
    y = YD + 80 + i * 22
    bold = (a == "2048")
    for cx, v in zip(COLS, (a, b, c, d)):
        t(cx, y, v, "svglbl" if bold else "svgsm",
          "#202124" if bold else col, size=13 if bold else None)

box(840, YD + 26, W - 840, 150, "#e6f4ea", GR)
t(856, YD + 50, '⭐ 两头夹的又是同一件事', "svglbl", "#0b6b30", size=13)
t(856, YD + 76, '段开长 →&#160;右矩阵少搬 →&#160;省搬运', fill="#0b6b30")
t(856, YD + 96, '段开长 →&#160;累加器变大 →&#160;吃片上容量', fill="#0b6b30")
t(856, YD + 124, '<tspan font-weight="700">所以「一段多少行」这个旋钮，'
                 '拧到底还是被那口灶台顶住的。</tspan>', fill="#0b6b30")
t(856, YD + 148, '——&#160;跟 3.6 里 GPU 那堵墙是同一堵。', fill=GY)

p.append('</svg>')
io.open('fig3-6.svg', 'w', encoding='utf-8').write('\n'.join(p))
print('fig3-6 ok')
