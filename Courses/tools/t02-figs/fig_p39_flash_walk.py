# -*- coding: utf-8 -*-
"""图 P-39 —— 同一个 FlashAttention，两条内存通路并排走完全程。

**2026-09-01 重画。** 上一版是一张三栏表格 —— 信息是齐的，但它不是图：
读者拿到的是「六行文字」，不是「两条路」。<b>表格能对齐信息，却画不出
「这条路上有一站是空的」，也画不出「两个暂存差 288 倍」。</b>
这一版改回真正的通路示意：

  · <b>两条竖直管道并排往下</b>，站与站之间是真箭头，走向一眼可见
  · <b>TPU 的第 ② 站画成虚线空位</b>，一根粗箭头从它中间直接跨过去 ——
    「没有这一层」不再是一句红字，是画面上一个看得见的窟窿
  · <b>第 ③ 站画出 tile 小方块和回旋箭头</b>，「反复搬进搬出」直接画出来
  · <b>第 ⑤ 站两个计算单元之间是双向箭头</b> —— 它每一拍都在两者之间跳
  · <b>底部那片 17×17 的格阵</b>把 288 倍从数字变成面积：
    GPU 的共享内存是其中一格，TPU 的 VMEM 是整片

**⚠️ 站数口径。** 本课此前两处数法不一致 —— T-6 沿 MXU 支线数得五站，
G-6 沿 VPU 主路数得四站。<b>两个都对，数的是不同的路。</b>
这张图不报总数，改成按功能对齐：谁在这一层、谁没有这一层。

**⚠️ 出处口径。** 容量与带宽沿用 G-6／T-6 的标注（含那里已写明的
「第三方」「官方未公开」）。<b>288 倍是本图当场算的</b>：64 MiB ÷ 227 KiB。
"""
from common import (Fig, para, grad, BL, GN, RD, YL, PU, TL,
                    INK, SUB, GREY, LINE, FILL)

W = 1400

# ── 两条管道的横向占位 ───────────────────────────────────────────
LX, LW = 40, 560          # GPU 管道
AX, AW = 620, 160         # 中间的对齐轴（站号 ＋ 这一站在干什么）
RX, RW = 800, 560         # TPU 管道

PIPE_Y = 114              # 管道顶（要给 title ＋ legend ＋ 两栏抬头让位）
GAP = 30                  # 站与站之间留给箭头的高度

# (站高, 站号, 轴上的站名)
STOPS = [(66, "①", "片外主存"),
         (78, "②", "硬件缓存"),
         (134, "③", "片上暂存"),
         (76, "④", "操作数缓冲"),
         (106, "⑤", "计算单元")]

# 每一站的顶 y，预先算好 —— 画箭头、画中轴都要用
SY, _y = [], PIPE_Y
for _h, _, _ in STOPS:
    SY.append(_y)
    _y += _h + GAP
PIPE_BOT = _y - GAP

SCH_Y, SCH_H = PIPE_BOT + 32, 112           # ⑥ 谁安排这个来回
RATIO_Y, RATIO_H = SCH_Y + SCH_H + 24, 216  # 288 倍那片格阵
LAND_Y, LAND_H = RATIO_Y + RATIO_H + 22, 116
SRC_Y, SRC_H = LAND_Y + LAND_H + 20, 88
H = SRC_Y + SRC_H + 20


def build():
    f = Fig(W, H, "FlashAttention 在 GPU 与 TPU 两条内存通路上的逐站对照。"
                  "两条竖直管道并排：片外主存、硬件缓存（TPU 这一站是空的）、"
                  "片上暂存、操作数缓冲、计算单元，"
                  "以及谁来安排矩阵单元与向量单元之间的切换")
    f.bg()
    f.title("同一个 FlashAttention，<tspan font-weight=\"700\">两条路并排走完</tspan>"
            "　—— 竖着看走向，横着看差别", "3.6 主图")
    f.legend([(BL, "GPU · B200"), (GN, "TPU v7"),
              (RD, "结构上真正不同的地方"), (YL, "FlashAttention 在这一站做的事")])

    _glows(f)
    _axis(f)
    _gpu_pipe(f)
    _tpu_pipe(f)
    _scheduler(f)
    _ratio(f)
    _land(f)
    _src(f)
    return f.out()


# ══════════════════════════════════════════════════════════════════════
# 雾化层：先铺光再画物件，否则光会把内容洗淡。
# 只给两处打光 —— 第 ③ 站（主战场）和第 ② 站（那个结构差异）。
def _glows(f):
    f.glow(LX + LW / 2, SY[2] + 67, 380, BL, 150)
    f.glow(RX + RW / 2, SY[2] + 67, 380, GN, 150)
    f.glow(LX + LW / 2, SY[1] + 39, 300, RD, 104)
    f.glow(RX + RW / 2, SY[1] + 39, 300, RD, 104)
    f.glow(LX + LW / 2, SY[4] + 53, 300, BL, 104)
    f.glow(RX + RW / 2, SY[4] + 53, 300, GN, 104)
    f.glow(700, SCH_Y + SCH_H / 2, 620, PU, 96)


# ══════════════════════════════════════════════════════════════════════
# 中轴：站号 ＋ 「FlashAttention 在这一站做什么」。
# 它同时是两条管道的对齐参照，所以要贯通到底。
def _axis(f):
    f.rect(AX + AW / 2 - 1, PIPE_Y - 16, 2, PIPE_BOT - PIPE_Y + 32,
           "#e3e5e8", rx=1)
    notes = [
        ["只有 Q / K / V / O 住这儿。", "<b>S 从来不出现。</b>"],
        ["<r>全图最大的结构差异</r>", "就在这一层。"],
        ["<b>主战场。</b>tile 住这儿，", "块能开多大看它的容量。"],
        ["进计算单元之前，", "要不要先落一次寄存器。"],
        ["<b>一直在两种单元间跳</b>：", "矩阵乘两次，夹一次 softmax。"],
    ]
    for i, (h, num, name) in enumerate(STOPS):
        y = SY[i]
        f.card(AX, y, AW, h, rx=9)
        f.rect(AX + AW / 2 - 15, y + 7, 30, 20, grad(SUB), GREY, 1.1, 10)
        f.t(AX + AW / 2, y + 21, num, "lbl", INK, "middle")
        f.t(AX + AW / 2, y + 43, name, "box", INK, "middle")
        for k, ln in enumerate(notes[i]):
            para(f, AX + AW / 2, y + 59 + k * 13, AW - 14, ln, "xxs", 13,
                 anchor="middle", max_lines=1)


def _arrow(f, cx, y, c, lab=None, w=14):
    """站与站之间那根下行箭头。w 是杆宽 —— 带宽越大画越粗。

    杆子刻意用半透明实色而不是 DEFS 里的浅渐变：渐变太淡，
    在白底上几乎看不见，走向就断了。"""
    f.rect(cx - w / 2, y, w, GAP - 11, c, rx=3, extra=' opacity=".42"')
    f.path(f"M{cx - w / 2 - 5},{y + GAP - 12} L{cx},{y + GAP - 2} "
           f"L{cx + w / 2 + 5},{y + GAP - 12} z", c, 0, c)
    if lab:
        f.t(cx + w / 2 + 11, y + GAP - 14, lab, "xxs", SUB)


# ══════════════════════════════════════════════════════════════════════
def _stop(f, x, w, y, h, c, ttl, spec):
    """一站的外壳：浮起的卡 ＋ 左侧色带 ＋ 标题行。返回正文可用的 y。"""
    f.card(x, y, w, h, rx=9, accent=c, elev=2)
    f.t(x + 20, y + 25, ttl, "box", c)
    f.t(x + w - 16, y + 25, spec, "xxs", SUB, anchor="end")
    return y + 42


# ── 左：GPU ───────────────────────────────────────────────────────
def _gpu_pipe(f):
    cx = LX + LW / 2
    f.t(LX + 6, PIPE_Y - 15, "GPU　·　B200", "sec", BL)

    yy = _stop(f, LX, LW, SY[0], STOPS[0][0], BL, "HBM3e", "192 GB · 8.0 TB/s")
    para(f, LX + 20, yy, LW - 40, "官方数字。<b>整颗芯片共用。</b>", "xs", 16,
         max_lines=1)
    _arrow(f, cx, SY[0] + STOPS[0][0], BL, "8.0 TB/s", 18)

    # ② L2 —— 右边那一格是空的，所以这一格要画得实
    yy = _stop(f, LX, LW, SY[1], STOPS[1][0], BL, "L2 缓存", "126 MB · 4 个分区")
    para(f, LX + 20, yy, LW - 40,
         "<b>硬件自动，有命中率。</b>程序管不着它留什么、赶走什么。",
         "xs", 16, max_lines=2)
    _arrow(f, cx, SY[1] + STOPS[1][0], BL, None, 18)

    # ③ 主战场：把「同一块硅两种身份」画成左右两半
    y = SY[2]
    _stop(f, LX, LW, y, STOPS[2][0], BL, "L1 ／ 共享内存",
          "256 KiB / SM · 共享 ≤ 227 KiB / 线程块")
    hx, hw = LX + 20, (LW - 52) / 2
    f.rect(hx, y + 46, hw, 28, grad(GREY), GREY, 1.2, 6)
    f.t(hx + hw / 2, y + 64, "左半：硬件管的 L1", "xxs", SUB, "middle")
    rx2 = hx + hw + 12
    f.rect(rx2, y + 46, hw, 28, grad(YL), YL, 1.6, 6)
    f.t(rx2 + hw / 2, y + 64, "右半：软件管的共享内存", "xxs", INK, "middle")
    _tiles(f, rx2 + 8, y + 82, YL)
    para(f, rx2 + 110, y + 90, hw - 118,
         "<r>tile 住这半边 —— 作者亲手搬进去。</r>", "xxs", 13, max_lines=2)
    _loop(f, hx + 62, y + 100, YL, "每块循环一次")
    _arrow(f, cx, y + STOPS[2][0], BL, None, 14)

    yy = _stop(f, LX, LW, SY[3], STOPS[3][0], BL, "寄存器堆 ＋ TMEM 支线",
               "各 256 KiB / SM")
    para(f, LX + 20, yy, LW - 40,
         "Blackwell 之前<b>矩阵操作数必须先落寄存器堆</b>；现在 TMA／"
         "<code>tcgen05.cp</code> 整块搬进 TMEM，<r>把寄存器绕过去</r>。",
         "xs", 15, max_lines=2)
    _arrow(f, cx, SY[3] + STOPS[3][0], BL, None, 14)

    _units(f, LX, LW, SY[4], STOPS[4][0], BL,
           ("Tensor Core", "4 / SM", "QKᵀ ／ PV"),
           ("CUDA Core", "128 / SM", "online softmax"))


# ── 右：TPU ───────────────────────────────────────────────────────
def _tpu_pipe(f):
    cx = RX + RW / 2
    f.t(RX + 6, PIPE_Y - 15, "TPU　·　v7", "sec", GN)

    yy = _stop(f, RX, RW, SY[0], STOPS[0][0], GN, "HBM3e",
               "192 GiB · 7.4 TB/s / chip")
    para(f, RX + 20, yy, RW - 40,
         "官方数字。<b>96 GiB / device</b>（v7 是 2 device / chip）。",
         "xs", 16, max_lines=1)

    # ② 这一站是空的 —— 整张图的重点，画成虚线窟窿 ＋ 一根跨过去的粗箭头
    y, h = SY[1], STOPS[1][0]
    f.rect(RX, y, RW, h, "none", RD, 2.0, 9, "7,5")
    f.t(RX + 20, y + 27, "没有这一站", "box", RD)
    para(f, RX + 20, y + 48, 210, "<b>HBM 直接进片上暂存。</b>", "xs", 16,
         max_lines=1)
    # 灰字必须让开中间那根粗箭头 —— 压上去两样都读不清
    para(f, cx + 28, y + 32, RX + RW - 20 - (cx + 28),
         "<g>省下的不只是面积，还有「不知道会不会命中」这件事本身。</g>",
         "xxs", 14, max_lines=2)
    # 一根从 ① 底穿过空位、直达 ③ 顶的箭头
    y0, y1 = SY[0] + STOPS[0][0], SY[2]
    f.rect(cx - 9, y0, 18, y1 - y0 - 10, GN, rx=3, extra=' opacity=".42"')
    f.path(f"M{cx - 14},{y1 - 11} L{cx},{y1 - 1} L{cx + 14},{y1 - 11} z",
           GN, 0, GN)
    f.t(cx + 22, y0 + 16, "一步到位", "xxs", GN)

    y = SY[2]
    _stop(f, RX, RW, y, STOPS[2][0], GN, "VMEM", "64 MiB / core")
    f.rect(RX + 20, y + 46, RW - 40, 28, grad(GN), GN, 1.2, 6)
    f.t(RX + 20 + (RW - 40) / 2, y + 64,
        "整块都是软件管的 —— 没有 tag，不会 miss", "xxs", INK, "middle")
    _tiles(f, RX + 28, y + 82, GN)
    para(f, RX + 138, y + 90, RW - 166,
         "Pallas 用 <code>BlockSpec</code> 声明每块搬多大，"
         "<r>剩下的编译器排 —— 不是作者亲手搬。</r>", "xxs", 13, max_lines=2)
    # 回旋箭头挪到 tile 阵下方 —— 放在同一行会直接压在方块上
    _loop(f, RX + 72, y + 124, GN, "每块循环一次")
    _arrow(f, cx, y + STOPS[2][0], GN, None, 14)

    yy = _stop(f, RX, RW, SY[3], STOPS[3][0], GN, "向量寄存器",
               "8 × 128 的二维块 · 个数未公开")
    para(f, RX + 20, yy, RW - 40,
         "<b>MXU 那条支线从来不经过它</b> —— 权重与数据直接推进阵列。"
         "<g>这点 GPU 到 Blackwell 才追上。</g>", "xs", 15, max_lines=2)
    _arrow(f, cx, SY[3] + STOPS[3][0], GN, None, 14)

    _units(f, RX, RW, SY[4], STOPS[4][0], GN,
           ("MXU", "256 × 256", "QKᵀ ／ PV"),
           ("VPU", "向量单元", "online softmax"))


# ══════════════════════════════════════════════════════════════════════
def _tiles(f, x, y, c):
    """一小片 tile 方块 —— 把「分块」画出来，而不是写出来。
    前 8 块填实 ＝ 已经算过的，后面空心 ＝ 还没轮到。"""
    for r in range(3):
        for col in range(6):
            done = (r * 6 + col) < 8
            f.rect(x + col * 15, y + r * 11, 12, 8,
                   c if done else "#fff", c, 0.9, 2)


def _loop(f, cx, y, c, lab):
    """一个回旋箭头：这一站要反复进出很多次。"""
    f.path(f"M{cx - 26},{y} a26,11 0 1 0 52,0", c, 1.6)
    f.path(f"M{cx + 21},{y - 7} L{cx + 28},{y + 1} L{cx + 18},{y + 4} z",
           c, 0, c)
    f.t(cx + 36, y + 4, lab, "xxs", c)


def _units(f, x, w, y, h, c, u1, u2):
    """⑤ 两个计算单元并排 ＋ 中间的双向箭头。"""
    f.card(x, y, w, h, rx=9, accent=c, elev=2)
    f.t(x + 20, y + 25, "算：两种单元轮流上", "box", c)
    bw = (w - 40 - 84) / 2
    for i, (nm, spec, job) in enumerate((u1, u2)):
        bx = x + 20 + i * (bw + 84)
        f.rect(bx, y + 38, bw, 54, grad(c if i == 0 else SUB),
               c if i == 0 else GREY, 1.4, 7)
        f.t(bx + bw / 2, y + 58, nm, "box", INK, "middle")
        f.t(bx + bw / 2, y + 74, spec, "xxs", SUB, "middle")
        f.t(bx + bw / 2, y + 88, job, "xxs", c if i == 0 else SUB, "middle")
    # 中间的来回 —— FlashAttention 每一拍都在这两个之间切
    mx = x + 20 + bw + 42
    f.path(f"M{mx - 26},{y + 58} L{mx + 22},{y + 58}", c, 1.8)
    f.path(f"M{mx + 27},{y + 58} L{mx + 17},{y + 53} L{mx + 17},{y + 63} z", c, 0, c)
    f.path(f"M{mx - 22},{y + 76} L{mx + 26},{y + 76}", c, 1.8)
    f.path(f"M{mx - 27},{y + 76} L{mx - 17},{y + 71} L{mx - 17},{y + 81} z", c, 0, c)
    f.t(mx, y + 46, "来回", "xxs", c, "middle")


# ══════════════════════════════════════════════════════════════════════
def _scheduler(f):
    """⑥ 谁安排这个来回 —— 横跨两栏，因为它是第 5 节那条主线。"""
    f.rect(40, SCH_Y, 1320, SCH_H, grad(PU), PU, 1.8, 12)
    f.t(58, SCH_Y + 26, "⑥　那个「来回」是谁安排的　—— "
                        "第 5 节那条主线，落在一个具体 kernel 上", "sec", PU)
    for i, (c, ttl, spec, body) in enumerate([
        (BL, "GPU：warp 调度器", "每个周期挑一次",
         "<b>运行时</b>从几十个 warp 里挑一个数据到位的发。"
         "算得慢就换一个上来 —— <b>延迟是被别人的工作盖住的</b>。"),
        (GN, "TPU：编译器 ＋ VLIW 槽", "编译期排死，精确到周期",
         "<b>编译期</b>定好哪条指令、第几周期、哪个发射槽。"
         "<r>没有第二个任务顶班 —— 排错了就是真的停住。</r>"),
    ]):
        x = 58 + i * 654
        f.card(x, SCH_Y + 38, 630, 60, rx=8, accent=c, aw=3)
        f.t(x + 16, SCH_Y + 58, ttl, "lbl", c)
        f.t(x + 618, SCH_Y + 58, spec, "xxs", SUB, anchor="end")
        para(f, x + 16, SCH_Y + 76, 596, body, "xxs", 14, max_lines=2)


# ══════════════════════════════════════════════════════════════════════
# 288 倍：与其写个数字，不如把它画成一片面积。
# 17 × 17 ＝ 289 格，GPU 的共享内存就是左上角那一格。
def _ratio(f):
    f.rect(40, RATIO_Y, 1320, RATIO_H, grad(YL), YL, 1.8, 12)
    f.t(58, RATIO_Y + 26, "📐 第 ③ 站两边差多少 —— 与其写个数字，不如画成面积",
        "sec")

    CS, N = 7.8, 17
    gx, gy = 68, RATIO_Y + 46
    for r in range(N):
        for c in range(N):
            first = not (r or c)
            f.rect(gx + c * CS, gy + r * CS, CS - 1.3, CS - 1.3,
                   BL if first else "#fff", BL, 0.5, 1)
    f.rect(gx - 4, gy - 4, N * CS + 5, N * CS + 5, "none", GN, 2.4, 3)
    # 那一格实在太小，套一圈描边把它标出来 —— 引线会穿出绿框，更乱
    f.rect(gx - 2.6, gy - 2.6, CS + 4, CS + 4, "none", BL, 1.6, 2)
    f.rect(gx + 4, gy + N * CS + 16, 9, 9, BL, BL, 0.8, 2)
    f.t(gx + 19, gy + N * CS + 24, "＝ GPU 共享内存 227 KiB（左上角那一格）",
        "xxs", BL)

    tx = gx + N * CS + 52
    f.card(tx, RATIO_Y + 44, 540, 150, rx=9, accent=GN)
    f.t(tx + 18, RATIO_Y + 68, "绿框整片 ＝ TPU VMEM 64 MiB", "box", GN)
    y = para(f, tx + 18, RATIO_Y + 92, 504,
             "64 MiB ÷ 227 KiB ≈ <b>288 倍</b>。左边那 <b>289</b> 个小格里，"
             "<b>GPU 只占蓝色那一格</b>。", "xs", 18, max_lines=2)
    para(f, tx + 18, y + 8, 504,
         "<b>这直接决定块能开多大</b> —— 块越大，K/V 要重复读的趟数越少。",
         "xs", 18, max_lines=2)

    ax = tx + 564
    f.card(ax, RATIO_Y + 44, 1342 - ax, 150, rx=9, accent=RD)
    f.t(ax + 18, RATIO_Y + 68, "⚠️ 顺带统一一个口径", "box", RD)
    para(f, ax + 18, RATIO_Y + 92, 1342 - ax - 36,
         "本课此前一处写 TPU「五站」、一处写「四站」——<b>两个都对</b>："
         "一处沿 MXU 支线数，一处沿 VPU 主路数。"
         "<r>数的是两条不同的路，所以这张图不报总数</r>，只问谁在这一层。",
         "xxs", 16, max_lines=6)


# ══════════════════════════════════════════════════════════════════════
def _land(f):
    f.rect(40, LAND_Y, 1320, LAND_H, grad(BL), BL, 1.8, 12)
    f.t(58, LAND_Y + 28, "⭐ 落点：这两条路上只有两处是结构性的，其余都是参数",
        "sec", BL)
    y = para(f, 58, LAND_Y + 54, 1284,
             "<b>第 ② 站</b>：GPU 有一整层硬件缓存，TPU 那里是个窟窿。　　"
             "<b>第 ⑥ 项</b>：那个来回由谁安排 —— 一边每周期现挑，一边编译期排死。"
             "<r>其余三站都是「同一件事，两边各有一个部件」，只是容量和名字不同。</r>",
             "xs", 19)
    para(f, 58, y + 2, 1284,
         "<b>而这两处恰好决定了 FlashAttention 在两边是两种性质的工作</b> —— "
         "<g>下一张图专门收这个口。</g>", "xs", 19)


def _src(f):
    f.card(40, SRC_Y, 1320, SRC_H, rx=12)
    f.t(58, SRC_Y + 26, "⚠️ 出处分层", "sec")
    y = para(f, 58, SRC_Y + 48, 1284,
             "<b>沿用图 G-6 / T-6 的标注</b>，含那里已写明的成色："
             "L2 带宽是第三方实测、TPU 片上带宽与向量寄存器个数<b>官方未公开</b>。",
             "xs", 17)
    para(f, 58, y + 2, 1284,
         "<b>本图当场算的</b>：64 MiB ÷ 227 KiB ≈ 288 倍。"
         "<g>格阵取 17×17 ＝ 289，是为了凑成方阵好数，差的那一格不是数据误差。</g>",
         "xxs", 16)


if __name__ == "__main__":
    import io
    io.open("out/fig_p39_flash_walk.svg", "w", encoding="utf-8").write(build())
    print("ok fig_p39_flash_walk")
