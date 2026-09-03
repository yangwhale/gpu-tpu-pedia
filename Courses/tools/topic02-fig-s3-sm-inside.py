# -*- coding: utf-8 -*-
"""图 3-2 · 一个 SM 拆开看：四个格子，以及它养得起多少人。

⭐ **为什么要有这一张。** 2026-09-03 现场连着问出四个问题：

    「既然一个 warp 太小、非得 4 个一起动，干嘛不把 warp 做成 128？」
    「一个线程要多少资源？一个 SM 能跑多少线程？由什么决定？」
    「100 个线程把资源占没了，是不是就起不了 2,000 个？」
    「一个 SM 128 个 CUDA Core、一个 Tensor Core，是这样吧？」

⛔ 最后那个是**错的**（Tensor Core 是 4 个），而前三个**共用同一把钥匙**：
   **一个 SM 物理上不是一整块，是四个格子**。这个画面在文档里其实有 ——
   G-3 和《GPU 显微镜》都写了「四个处理块」—— 但它散在几个折叠里，
   **从来没有被单独立出来当成一张图讲过**，所以读者拿不到它。

⭐ 这张图把那个画面立起来，然后拿它一口气回答三件事：
   ① 32 是怎么来的（＝一个格子的宽度）、128 是怎么来的（＝四格各出一个）；
   ② 住得下多少人由三个闸门决定，谁先卡住算谁；
   ③ 换 warp 为什么是零成本 —— 以及那个零成本的账单长什么样。

📌 数都对得上文档：
   《GPU 显微镜》「每个处理块 16 个 warp 槽、64 KiB 寄存器、32 个 CUDA Core
   和 1 个 Tensor Core」「148 个 SM × 每 SM 4 个 Tensor Core ＝ 592」。
   寄存器上限 64 K 32-bit/SM、每线程 255、每 SM 最多 32 个 block ——
   NVIDIA Hopper Tuning Guide（cc 9.0）。
"""
import io

BL, OR, GR, RD, GY, YL = "#1a73e8", "#e8710a", "#1e8e3e", "#d93025", "#5f6368", "#f9ab00"
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


p.append('<svg viewBox="0 0 %d 690" width="100%%" role="img" aria-label="'
         '一个 SM 拆成四个处理块，以及决定它能驻留多少线程的三个闸门">' % W)
t(0, 17, '一个 SM 拆开看 ——&#160;<tspan font-weight="700">它不是一整块，是四个格子</tspan>',
  "svglbl", "#202124", size=14)
t(0, 37, 'warp 为什么是 32、住得下多少人、换 warp 为什么不要钱 ——&#160;'
         '这三个问题的答案都在这个画面里')

# ══ 上半：SM 拆开 ══════════════════════════════════════════════════
box(0, 56, 830, 196, "#f8f9fa")
t(14, 78, '1 个 SM<tspan fill="%s">（B200 全片 148 个）</tspan>' % GY, "svglbl", "#202124", size=13)

QW, QG = 196, 8
for k in range(4):
    x = 14 + k * (QW + QG)
    box(x, 90, QW, 116, "#fff", BL if k == 0 else "#dadce0")
    t(x + 12, 110, '处理块 %d' % k, "svglbl", BL, size=12)
    t(x + 12, 126, 'sub-core / quadrant', fill=GY)
    for i, line in enumerate(('1 个 warp scheduler',
                              '<tspan font-weight="700">32 个 CUDA Core</tspan>',
                              '<tspan font-weight="700">1 个 Tensor Core</tspan>',
                              '64 KiB 寄存器 · 16 个 warp 槽')):
        t(x + 12, 146 + i * 15, line, fill="#202124")
t(14, 226, '四个格子共用：<tspan font-weight="700">228 KB 的 L1 ＋ 共享内存</tspan>'
           '（单个线程块最多要走 227 KiB）· TMA 搬运引擎 · 通往 L2 的出口', fill="#202124")
t(14, 243, '⛔ 一个 warp 一旦被分到某个格子，就在那儿待到死 ——&#160;不迁走。'
           '这条「钉死」是后面所有「除以 4」的前提。', fill=RD)

box(850, 56, W - 850, 196, "#e8f0fe", BL)
t(866, 78, '四格加起来 ＝ 整个 SM', "svglbl", "#174ea6", size=13)
for i, (a, b) in enumerate(((
        '128 个', 'CUDA Core'), ('4 个', 'Tensor Core ——&#160;不是 1 个'),
        ('65,536 个', '32-bit 寄存器 ＝ 256 KiB'),
        ('64 个', 'warp 槽 ＝ 2,048 个线程'),
        ('最多 32 个', '线程块'))):
    t(866, 102 + i * 19, a, fill="#174ea6", bold=True)
    t(950, 102 + i * 19, b, fill="#174ea6")
t(866, 213, '⭐ 寄存器堆 256 KiB，<tspan font-weight="700">比它旁边那块 '
            'L1 ＋ 共享内存（228 KB）还大</tspan>。', fill="#174ea6")
t(866, 231, '一块芯片上最贵的 SRAM，最大的一份拿去当了「让人待命」的本钱。', fill="#174ea6")

# ══ 中：三个闸门 ══════════════════════════════════════════════════
Y = 274
t(0, Y - 8, '住得下多少人？<tspan font-weight="700">三个闸门，谁先卡住算谁</tspan>'
            '——&#160;这就是「占用率」这个词量的东西', "svglbl", "#202124", size=13)

CW = 452
box(0, Y + 6, CW, 158, "#fff", OR)
t(14, Y + 28, '① 寄存器 ——&#160;最常先卡住的那个', "svglbl", OR, size=12)
t(14, Y + 48, '<tspan font-weight="700">65,536 ÷ 2,048 ＝ 32</tspan>'
              '　每线程只能用 32 个，才住得满', fill="#202124")
for i, (r, th, pct) in enumerate((('32 个', '2,048 线程', '100%'),
                                  ('64 个', '1,024 线程', '50%'),
                                  ('128 个', '512 线程', '25%'),
                                  ('255 个（上限）', '256 线程', '12.5%'))):
    y = Y + 70 + i * 19
    t(24, y, '每线程 ' + r, fill=GY)
    t(190, y, '→　' + th, fill="#202124")
    t(330, y, '占用率 ' + pct, fill=OR if i else GR)
t(14, Y + 150, '⛔ 循环展开、把中间结果留在寄存器里 ——&#160;都在削减能待命的人数。', fill=RD)

box(CW + 12, Y + 6, CW, 158, "#fff", "#dadce0")
t(CW + 26, Y + 28, '② 共享内存', "svglbl", "#202124", size=12)
t(CW + 26, Y + 50, '一个 SM 一共 <tspan font-weight="700">228 KB</tspan>，'
                   '单个线程块最多要走 227 KiB。', fill="#202124")
t(CW + 26, Y + 70, '你的 block 要 100 KiB ——&#160;'
                   '<tspan font-weight="700">这个 SM 就只住得下 2 个 block</tspan>，', fill="#202124")
t(CW + 26, Y + 88, '寄存器还剩多少都没用了。', fill="#202124")
t(CW + 26, Y + 114, '⭐ 你问的「100 个线程把资源占没」——&#160;真会发生的是这一条。',
  fill=GR)
t(CW + 26, Y + 132, '一个 block 独吞 227 KiB，整个 SM 就只剩它一家，'
                    '最多 1,024 个线程。', fill=GR)

box(2 * CW + 24, Y + 6, W - 2 * CW - 24, 158, "#fff", "#dadce0")
t(2 * CW + 38, Y + 28, '③ 槽位本身', "svglbl", "#202124", size=12)
t(2 * CW + 38, Y + 50, '最多 <tspan font-weight="700">64 个 warp</tspan>、'
                       '最多 <tspan font-weight="700">32 个 block</tspan>。', fill="#202124")
t(2 * CW + 38, Y + 72, 'block 开得太小也吃亏：每块只有 32 个线程时，', fill="#202124")
t(2 * CW + 38, Y + 90, '32 块 × 1 warp ＝ 32 个 warp，', fill="#202124")
t(2 * CW + 38, Y + 108, '<tspan font-weight="700">64 个槽只填得满一半</tspan>。', fill="#202124")
t(2 * CW + 38, Y + 134, '资源一点没超，人就是招不满。', fill=GY)

# ══ 下：为什么是 32 ════════════════════════════════════════════════
Y2 = Y + 182
box(0, Y2, W, 122, "#fef7e0", YL)
t(14, Y2 + 22, '⚡ 那为什么不干脆把 warp 做成 128，省得四个凑一堆？',
  "svglbl", "#7a5000", size=13)
t(14, Y2 + 44, '<tspan font-weight="700">先把问题翻译一下</tspan>：32 就是一个格子的宽度，'
               '128 就是四个格子各出一个 warp。所以「把 warp 做成 128」'
               '＝「<tspan font-weight="700">把四个格子合成一个</tspan>」。', fill="#7a5000")
for i, s in enumerate((
        '① 合成一个就只剩一个 warp scheduler。'
        '现在四个格子能同时跑四条毫不相干的指令流 ——&#160;合并了这个能力就没了。',
        '② 绝大多数指令根本不需要 128 宽（访存、整数、分支、超越函数）。'
        '只有 wgmma 那一类要。为它加宽，等于让全部代码替矩阵乘付账。',
        '③ 32 × 4 B ＝ 128 B ＝ 正好一条 cache line；而分支发散的惩罚随宽度线性变重 ——&#160;'
        '一个 if 走岔，整组都得两边各走一遍。')):
    t(14, Y2 + 66 + i * 18, s, fill="#7a5000")

# ══ 落点 ══════════════════════════════════════════════════════════
Y3 = Y2 + 132
box(0, Y3, W, 70, "#e6f4ea", GR)
t(14, Y3 + 23, '⭐ 换 warp 是零成本的 ——&#160;而这张图就是那张账单',
  "svglbl", "#0b6b30", size=13)
t(14, Y3 + 43, 'CPU 换线程要把寄存器存下来再恢复；'
               'GPU <tspan font-weight="700">从来不搬</tspan> ——&#160;'
               '2,048 个线程的寄存器<tspan font-weight="700">一直物理占着</tspan>，'
               '谁的数据到了就发谁。', fill="#0b6b30")
t(14, Y3 + 61, '<tspan font-weight="700">代价就是那 256 KiB。</tspan>'
               '「运行时才知道数据什么时候到」这个赌注，最贵的一项不是调度器，'
               '是这堆养着待命者的寄存器。', fill="#0b6b30")

p.append('</svg>')
io.open('fig3-2.svg', 'w', encoding='utf-8').write('\n'.join(p))
print('fig3-2 ok')
