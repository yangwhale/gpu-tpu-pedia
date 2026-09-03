# -*- coding: utf-8 -*-
"""图 3-4 · 一块数据怎么进到共享内存 —— GPU 走了三代，TPU 出厂就在终点。

⭐ **为什么要有这一张。** 2026-09-03 现场连问四个问题：

    「1. 那个记分板到底是干什么的？
      2a. 一个 block 里边的同步机制是怎么样的？
      2b. cluster 之间可以访问别的 SM 里边的数据，硬件上怎么做到的？
      2c. 他们是通过哪个单元进行数据搬运，或者是直接内存访问？」

⛔ **这四个问题课程一个都没答，而其中一个词课程自己已经在用了。**
   L300 图 3-1 的图注原文就写着「代价才是那一整套 ——&nbsp;**记分板**、
   几十个 warp 槽、全部上下文常驻」。**用了这个词，却从没说过它是什么。**
   这是最坏的一种缺口：读者以为自己漏听了，其实是课程没讲。

⭐ **为什么这张图只画第 2c 个问题。** 四个答案里，只有搬运这一条
   **本身就是一条数据通路**，画出来是机器的图；记分板和 barrier 画出来
   只会是「一张我自己发明的分类账」——&nbsp;那正是 fig1-5 被删掉的原因
   （2026-09-03 现场判词：「特别单薄，而且也没有用」）。
   **所以它们走正文和折叠块，不占一张图。**

⭐ **而且搬运这条线是四个问题里唯一能落回本课主线的**：三代下来，
   搬运从「取数指令的副作用」变成「一台按描述符干活的引擎」——
   **终点正好是 TPU 第一天就有的形状**。这不是谁抄谁，是同一个物理约束
   （矩阵单元越快，喂料越不能占着算的人）逼出来的同一个答案。
   G-6 早就写过这半句（「GPU 侧搬运是取数指令的副作用；TPU 侧是一条独立的
   DMA」），这张图给的是**它怎么一步步变成现在这样**。

📌 **出处口径**（这一张全部是官方，没有第三方推测）：
   · `cp.async` / LDGSTS 建立 global→shared 直通、不落寄存器 —— CUDA 编程指南
     「Asynchronous Data Copies」章。
   · TMA「用拷贝描述符按张量维度和块坐标发起」「单线程编程模型，一个线程发起后
     地址生成与数据搬运全部由硬件完成」「完成后在共享内存 barrier 上signal」
     —— NVIDIA Hopper 架构官方博客原话。
   · 异步事务 barrier「不只数线程到达，还数事务（字节数）」—— 同上。

⛔ 图内**不写任何 TPU 侧带宽数字**（VMEM 带宽未公开），只写「编译期发的 DMA」
   这个机制事实 —— 与 fig_g6_memory.py 的既有口径一致。
"""
import io

BL, OR, GN, GY, PU = "#1a73e8", "#e8710a", "#1e8e3e", "#5f6368", "#8430ce"
DIM = "#9aa0a6"
W = 1400

# 列 x 起点与宽度：中转那一列（B）是主角，所以最宽
LBL_X, LBL_W = 0, 206
AX, AW = 218, 100          # 源
BX, BW = 352, 200          # 中转 ←—— 整张图的差别全在这一列
CX, CW = 584, 118          # 落点
DX, DW = 734, 100          # 消费者
T1X = 866                  # 地址、跨步、边界谁算
T2X = 1116                 # 发起它的线程接下来干什么

TOP, RH, BH = 104, 92, 48

# (代次, 指令, 源, 中转文字, 中转副文字, 中转色, 落点, 消费者, 谁算地址, 线程状态)
ROWS = [
    ("一直都有", "普通 <tspan font-weight=\"700\">ld / st</tspan>",
     "全局内存", "寄存器", "数据在这儿过一道", DIM, "共享内存", "Tensor Core",
     ["线程<tspan font-weight=\"700\">自己算</tspan>，自己跑循环"],
     ["<tspan font-weight=\"700\">被占住</tspan> ——&#160;搬完才轮到算"]),
    ("Ampere 起", "<tspan font-weight=\"700\">cp.async</tspan>（SASS：LDGSTS）",
     "全局内存", "直通", "不落寄存器", BL, "共享内存", "Tensor Core",
     ["<tspan font-weight=\"700\">还是线程自己算</tspan>"],
     ["省了寄存器这一道，", "地址和循环仍要自己跑"]),
    ("Hopper 起", "<tspan font-weight=\"700\">TMA</tspan> ＋ 一张拷贝描述符",
     "全局内存", "TMA 引擎", "按张量维度＋块坐标", GN, "共享内存", "Tensor Core",
     ["<tspan font-weight=\"700\">硬件算</tspan>：跨步、偏移、", "边界全由它接管"],
     ["<tspan font-weight=\"700\">一个线程发完就走</tspan>；", "搬完引擎自己去 barrier 报数"]),
    ("TPU：<tspan font-weight=\"700\">出厂就是这样</tspan>", "编译器显式发的 <tspan font-weight=\"700\">DMA</tspan>",
     "HBM", "DMA 引擎", "和上面那行是同一类东西", OR, "VMEM", "MXU",
     ["<tspan font-weight=\"700\">编译期就排好了</tspan>"],
     ["<tspan font-weight=\"700\">这一栏对 TPU 不成立</tspan> ——&#160;", "这边压根没有「线程」这个角色"]),
]

BAND_Y = TOP + len(ROWS) * RH + 14
BAND_H = 96
WARN_Y = BAND_Y + BAND_H + 16
WARN_H = 82
SRC_Y = WARN_Y + WARN_H + 16
H = SRC_Y + 40

p = ['<svg viewBox="0 0 %d %d" width="100%%" role="img" aria-label="'
     '一块数据从全局内存进到共享内存的四种路径：普通 load/store 经过寄存器、'
     'Ampere 的 cp.async 直通、Hopper 的 TMA 引擎按描述符搬运、'
     'TPU 从第一代起就用编译器显式发的 DMA">' % (W, H)]
p.append('<defs><marker id="a34" viewBox="0 0 8 8" refX="7" refY="4" '
         'markerWidth="5" markerHeight="5" orient="auto">'
         '<path d="M0 0 L8 4 L0 8 z" fill="#5f6368"/></marker></defs>')

p.append('<text class="svglbl" x="0" y="17" fill="#202124" style="font-size:14px">'
         '一块数据怎么进到共享内存 ——&#160;'
         '<tspan font-weight="700">GPU 走了三代，TPU 出厂就在终点</tspan></text>')
p.append('<text class="svgsm" x="0" y="37">'
         '只看两件事：<tspan font-weight="700">数据在半路要不要经过寄存器</tspan>，'
         '以及<tspan font-weight="700">地址、跨步、边界是谁算的</tspan>'
         '——&#160;这两件事一变，线程就从「搬运工」退成了「发号令的」</text>')

# 中转那一列整体加浅底：这张图的差别全长在这一列上
p.append('<rect x="%d" y="%d" width="%d" height="%d" rx="10" fill="#f1f3f4"/>'
         % (BX - 14, TOP - 30, BW + 28, len(ROWS) * RH + 24))
p.append('<text class="svgsm" x="%d" y="%d" fill="#5f6368" text-anchor="middle">'
         '中转这一格 ——&#160;<tspan font-weight="700">整张图的差别都在这儿</tspan></text>'
         % (BX + BW / 2, TOP - 12))

for x, s in ((T1X, "地址、跨步、边界谁算"), (T2X, "发起它的线程接下来干什么")):
    p.append('<text class="svglbl" x="%d" y="%d" fill="#5f6368" '
             'style="font-size:12px">%s</text>' % (x, TOP - 12, s))


def box(x, y, w, col, main, sub=None, strong=False):
    p.append('<rect x="%d" y="%d" width="%d" height="%d" rx="7" fill="%s" '
             'stroke="%s" stroke-width="%s"/>'
             % (x, y, w, BH, "#fff" if not strong else "#e6f4ea" if col == GN
                else "#fef7e0" if col == OR else "#fff", col,
                "2" if strong else "1.3"))
    p.append('<text class="svgsm" x="%d" y="%d" fill="%s" text-anchor="middle" '
             'font-weight="700">%s</text>' % (x + w / 2, y + (20 if sub else 28), col, main))
    if sub:
        p.append('<text class="svgsm" x="%d" y="%d" fill="#5f6368" '
                 'text-anchor="middle" style="font-size:10px">%s</text>'
                 % (x + w / 2, y + 36, sub))


def arrow(x0, x1, y):
    p.append('<path d="M%d %d H%d" stroke="#5f6368" stroke-width="1.4" '
             'marker-end="url(#a34)"/>' % (x0, y, x1 - 3))


for i, (gen, ins, src, mid, midsub, col, dst, cons, addr, thr) in enumerate(ROWS):
    y = TOP + i * RH
    by = y + (RH - BH) / 2 - 6
    cy = by + BH / 2
    if i:
        p.append('<line x1="0" y1="%d" x2="%d" y2="%d" stroke="#e8eaed"/>' % (y - 8, W, y - 8))

    p.append('<text class="svglbl" x="0" y="%d" fill="%s" style="font-size:12px">%s</text>'
             % (by + 18, col if col != DIM else "#202124", gen))
    p.append('<text class="svgsm" x="0" y="%d">%s</text>' % (by + 36, ins))

    box(AX, by, AW, DIM, src)
    arrow(AX + AW, BX, cy)
    box(BX, by, BW, col, mid, midsub, strong=(i >= 2))
    arrow(BX + BW, CX, cy)
    # ⛔ 这里**不能**用 col：落点和消费者四行都是同一个东西，一变色就等于
    #    在暗示「共享内存也不一样」，而这张图的论点正相反 ——&nbsp;只有中转那列在变。
    box(CX, by, CW, DIM, dst)
    arrow(CX + CW, DX, cy)
    box(DX, by, DW, DIM, cons)

    for j, s in enumerate(addr):
        p.append('<text class="svgsm" x="%d" y="%d" fill="#3c4043">%s</text>'
                 % (T1X, by + 20 + j * 17, s))
    for j, s in enumerate(thr):
        p.append('<text class="svgsm" x="%d" y="%d" fill="%s">%s</text>'
                 % (T2X, by + 20 + j * 17, "#3c4043" if i < 3 else "#7a3e00", s))

# ── 落点带 ────────────────────────────────────────────────────────────
p.append('<rect x="0" y="%d" width="%d" height="%d" rx="9" fill="#e6f4ea" stroke="%s"/>'
         % (BAND_Y, W, BAND_H, GN))
p.append('<text class="svglbl" x="16" y="%d" fill="#0b6b30" style="font-size:13px">'
         '⭐ 一条线看下来：搬运从「<tspan text-decoration="underline">取数指令的副作用</tspan>」，'
         '变成了「<tspan text-decoration="underline">一台按描述符干活的引擎</tspan>」</text>'
         % (BAND_Y + 25))
p.append('<text class="svgsm" x="16" y="%d" fill="#0b6b30">'
         'GPU 用了三代走到这儿，TPU 第一天就在这儿。'
         '<tspan font-weight="700">这不是谁抄谁</tspan> ——&#160;'
         '是同一个物理约束逼出来的同一个答案：</text>' % (BAND_Y + 46))
p.append('<text class="svgsm" x="16" y="%d" fill="#0b6b30">'
         '<tspan font-weight="700">矩阵单元越快，喂料这件事就越不能占着算的人</tspan>。'
         '两边的分歧不在终点，在<tspan font-weight="700">谁来发这条 DMA</tspan> ——&#160;'
         'GPU 是运行时某个线程，TPU 是编译期就排好的一步。</text>' % (BAND_Y + 65))
p.append('<text class="svgsm" x="16" y="%d" fill="%s">'
         '——&#160;「搬运是取数的副作用，还是一条独立的 DMA」这句话，'
         '在《GPU 显微镜》图 G-6 里已经出现过；这张图给的是它<tspan font-weight="700">怎么一步步变成现在这样</tspan>。</text>'
         % (BAND_Y + 84, GY))

# ── 连带的一条：生产者换了人，对齐的办法也得换 ────────────────────────
p.append('<rect x="0" y="%d" width="%d" height="%d" rx="9" fill="#fef7e0" stroke="%s"/>'
         % (WARN_Y, W, WARN_H, "#f9ab00"))
p.append('<text class="svglbl" x="16" y="%d" fill="#7a5000" style="font-size:13px">'
         '⚠️ 生产者一换人，<tspan font-weight="700">对齐的办法也得跟着换</tspan></text>'
         % (WARN_Y + 24))
p.append('<text class="svgsm" x="16" y="%d" fill="#7a5000">'
         '第三行那台引擎<tspan font-weight="700">不是线程</tspan>，它不会「到达」，'
         '只会「往里放了多少字节」。而 <tspan font-weight="700">'
         '<tspan font-family="monospace">__syncthreads()</tspan> 数的是人头</tspan>，对它没用。</text>'
         % (WARN_Y + 45))
p.append('<text class="svgsm" x="16" y="%d" fill="#7a5000">'
         '所以 Hopper 的 barrier 多了一项本事：<tspan font-weight="700">连字节数一起数</tspan>'
         '——&#160;人到齐<tspan font-weight="700">并且</tspan>字节够了，才放行。'
         '<tspan fill="%s">这就是「异步事务 barrier」这个名字的来历。</tspan></text>'
         % (WARN_Y + 64, GY))

p.append('<text class="svgsm" x="0" y="%d" fill="%s">'
         '📌 这一张全部出自官方：<tspan font-weight="700">cp.async</tspan> 的直通路径见 CUDA 编程指南；'
         '<tspan font-weight="700">TMA 的单线程发起、硬件接管地址生成</tspan>与'
         '<tspan font-weight="700">异步事务 barrier 数字节</tspan>，是 Hopper 架构官方博客的原话。</text>'
         % (SRC_Y + 12, DIM))
p.append('</svg>')
io.open('fig3-4.svg', 'w', encoding='utf-8').write('\n'.join(p))
print('fig3-4 ok', H)
