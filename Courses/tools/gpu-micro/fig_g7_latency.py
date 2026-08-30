# -*- coding: utf-8 -*-
"""图 G-7 —— 延迟怎么被藏起来：一边靠换人，一边靠排班。

两边都躲不掉「取一次数要几百个周期」。区别在于：
  · GPU 让很多 warp 轮流用同一个单元 —— 时间复用，代价是要养一堆现场；
  · TPU 让不同单元同时开工 —— 空间并行，代价是编译器必须算准每一拍。

TPU 侧的 VLIW 槽位构成出自 IEEE Micro 2021《The Design Process for Google's
Training Chips: TPUv2 and TPUv3》，是 v2/v3 的公开数字；v7 官方没有公布，
图上如实标注代次，不往 v7 上套。
"""
from common import Fig, para, BL, RD, YL, GN, PU, TL, INK, SUB, GREY, FILL

W = 1400
TOP = 84

PW = 676                       # 单侧面板宽
PX = [20, 20 + PW + 28]        # 左右面板 x
LANEL = 104                    # 泳道标签栏宽
NCYC = 12                      # 时间轴格数
CW = (PW - LANEL - 18) / NCYC  # 每格宽
LH = 25                        # 泳道行高

BAND = TOP
LANE_T = BAND + 86
NLANE = 7                      # 两侧统一 7 行（6 泳道 + 合成条）
LANE_B = LANE_T + NLANE * LH + 72

CAV_T = LANE_B + 8               # 通栏警示条：两条合成条的判据不一样
MID_T = CAV_T + 34 + 30
MID_H = 240
BOT_T = MID_T + MID_H + 38
H = BOT_T + 164

IDLE, HOLE = "#f1f3f4", RD


def build():
    f = Fig(W, H, "GPU 与 TPU 的延迟隐藏机制对照：GPU 靠多个 warp 轮流占用同一个单元，"
                  "TPU 靠编译期把不同功能单元的起止时刻排在一起")
    f.title("延迟怎么被藏起来　—— GPU 让<tspan fill=\"#1a73e8\">很多人轮流用同一个单元</tspan>，"
            "TPU 让<tspan fill=\"#1e8e3e\">不同单元同时开工</tspan>")
    _legend(f)

    _gpu(f)
    _tpu(f)
    _caveat(f)
    _bottom(f)
    return f.out()


def _legend(f):
    """图例直接调 _cell 画 —— 跟泳道用的是同一支笔。

    上一版是手写 f.legend([...])：四个色块里三个跟图上实际颜色对不上
    （图例给纯红，格子是浅粉描红边；图例给中灰，格子是近白）。
    这类「图例漂移」不是手滑，是两条独立的绘制路径必然的结局 ——
    所以修法不是把色号对齐一次，是让图例没有自己的画法。
    """
    x, y = 20, 48
    for cell, c, txt in ((("B", ""), BL, "这一拍在发指令 / 在干活"),
                         (("W", ""), BL, "在等数据回来（虚线框）"),
                         (("H", ""), BL, "气泡：没人能干活，这一拍白丢"),
                         (("-", ""), BL, "尚未参与 / 已结束")):
        _cell(f, x, y, 18, 12, cell, c)
        f.t(x + 24, y + 10, txt, "sm")
        x += 24 + 11.5 * sum(1.0 if ord(ch) > 0x2E80 else 0.55 for ch in txt) + 22


# ══════════════════════════════════════════════════════════════════════
def _panel(f, i, c, kicker, sub):
    x = PX[i]
    f.rect(x, BAND, PW, LANE_B - BAND, "#fff", c, 1.8, 10)
    f.rect(x, BAND, PW, 28, FILL[c], rx=10)
    f.rect(x, BAND + 19, PW, 9, FILL[c], rx=0)
    f.t(x + 14, BAND + 20, kicker, "box", c)
    para(f, x + 14, BAND + 44, PW - 28, sub, "xs", 14)
    return x


def _axis(f, x, c):
    """时间轴刻度 —— 横轴是周期，不是等比真实时间。"""
    ax = x + LANEL
    for k in range(NCYC + 1):
        f.line(ax + k * CW, LANE_T - 8, ax + k * CW, LANE_T - 4, LINE_C, 1)
    f.t(ax, LANE_T - 12, "周期 →", "xxs")
    f.t(ax + NCYC * CW, LANE_T - 12, "（示意，非等比）", "xxs", None, "end")


LINE_C = "#dadce0"


def _lane(f, x, row, label, note, cells, c):
    """一条泳道。cells 是 NCYC 个 (状态, 文字) —— 状态见下方 _cell。"""
    y = LANE_T + row * LH
    f.t(x + 12, y + 13 if note else y + 16, label, "lbl", c)
    if note:
        f.t(x + 12, y + 23, note, "xxs", None)
    ax = x + LANEL
    for k, cell in enumerate(cells):
        _cell(f, ax + k * CW, y + 3, CW - 2, LH - 7, cell, c)


def _cell(f, x, y, w, h, cell, c):
    st, txt = cell
    if st == "-":
        f.rect(x, y, w, h, IDLE, rx=2)
        return
    if st == "B":                       # 在干活
        f.rect(x, y, w, h, c, rx=2)
        fill = "#fff"
    elif st == "W":                     # 在等
        f.rect(x, y, w, h, FILL[c], c, 0.8, 2, "2,2")
        fill = None
    else:                               # 气泡
        f.rect(x, y, w, h, FILL[HOLE], HOLE, 1.4, 2)
        fill = HOLE
    if txt:
        f.t(x + w / 2, y + h / 2 + 3.5, txt, "xxs", fill, "middle")


# ══════════════════════════════════════════════════════════════════════
def _gpu(f):
    x = _panel(f, 0, BL, "NVIDIA B200　一个处理块（sub-core）",
               "调度器每一拍只能发<b>一条</b>指令。谁的数据回来了谁就能上 —— "
               "所以真正决定性能的不是单个 warp 多快，是<b>手上有多少个 warp 可以换</b>。")
    _axis(f, x, BL)

    L = 8                                # 取数延迟：8 个周期（示意）
    rows = []
    for w in range(6):
        cells = []
        for k in range(NCYC):
            if k < w:
                cells.append(("-", ""))
            elif k == w or k == w + L:
                cells.append(("B", "发"))
            elif w < k < w + L:
                cells.append(("W", ""))
            else:
                cells.append(("-", ""))
        rows.append((f"warp {w}", cells))

    for i, (lab, cells) in enumerate(rows):
        _lane(f, x, i, lab, None, cells, BL)

    # 合成条：这一拍这个处理块到底有没有活干
    comp = []
    for k in range(NCYC):
        busy = any(rows[w][1][k][0] == "B" for w in range(6))
        comp.append(("B", "") if busy else ("H", "气泡"))
    _lane(f, x, 6, "调度器发射槽", "有没有指令被发出", comp, BL)

    y = LANE_T + NLANE * LH + 24
    para(f, x + 14, y + 10, PW - 28,
         "第 6、7 拍两个<r>气泡</r>：六个 warp 全在等数，调度器无人可选。"
         "如果驻留的是 8 个 warp 而不是 6 个，这两拍就补上了 —— "
         "<b>能不能补上，取决于寄存器还剩多少。</b>", "xs", 14)


def _tpu(f):
    x = _panel(f, 1, GN, "TPU v7　一个 TensorCore",
               "没有「换一个 warp 上来」这回事。这里的四条泳道是<b>四个不同的物理单元</b>，"
               "本来就能同时动 —— 编译器要做的是让它们的起止时刻正好对上。")
    _axis(f, x, GN)

    B, Wt, N = ("B", ""), ("W", ""), ("-", "")
    lanes = [
        ("标量单元", "个位数周期",
         [("B", "跳"), N, N, N, N, N, N, ("B", "W"), N, N, N, N]),
        ("向量单元 VPU", "几十个周期",
         [B, B, B, N, N, B, B, B, N, B, B, B]),
        ("矩阵单元 MXU", "几百个周期",
         [B, B, B, B, B, B, B, B, B, B, B, B]),
        ("DMA 引擎", "几百个周期",
         [B, B, B, B, B, B, N, B, B, B, B, B]),
    ]
    for i, (lab, note, cells) in enumerate(lanes):
        _lane(f, x, i, lab, note, cells, GN)

    # 合成条 —— 跟 GPU 那条并排看：这里一个气泡都没有
    _lane(f, x, 4, "单元占用", "有没有单元在动", [("B", "")] * NCYC, GN)

    # 一条 VLIW bundle 的槽位构成
    by = LANE_T + 5 * LH + 10
    f.t(x + 12, by + 14, "一条 bundle", "lbl", INK)
    f.t(x + 12, by + 23, "322 位 · v2/v3", "xxs")
    ax = x + LANEL
    slots = [("标量", 2, PU), ("向量", 4, GN), ("矩阵 推/取", 2, RD),
             ("杂项", 1, YL), ("立即数", 6, SUB)]
    tot = sum(n for _, n, _ in slots)
    sw = (PW - LANEL - 18) / tot
    cx = ax
    for name, n, c in slots:
        f.rect(cx, by + 2, n * sw - 3, 26, FILL[c], c, 1.4, 4)
        f.t(cx + (n * sw - 3) / 2, by + 14, name, "xxs", c, "middle")
        f.t(cx + (n * sw - 3) / 2, by + 24, f"×{n}", "xxs", None, "middle")
        cx += n * sw
    # 括注：前四组才是「发射槽」，立即数是随指令带的常数字段，不占发射能力
    for x0, n, lab, c in ((ax, 9, "发射槽 ×9　—— 这 9 个是同一拍并发出去的", INK),
                          (ax + 9 * sw, 6, "立即数 ×6　不占发射槽", SUB)):
        f.path(f"M{x0},{by + 32} v5 h{n * sw - 3} v-5", c, 1.2)
        f.t(x0 + (n * sw - 3) / 2, by + 48, lab, "xxs", c, "middle")

    y = LANE_T + NLANE * LH + 24
    para(f, x + 14, y + 10, PW - 28,
         "<b>没有一个气泡。</b>「矩阵槽是一推一取」这件事本身就说明问题："
         "MXU 不是被调用的函数，是一条推进去、过一会儿取出来的流水线 —— "
         "中间那段时间，别的槽照常在发指令。", "xs", 14)


def _caveat(f):
    """把两条合成条的判据差异摊开说 —— 这是全图唯一一处「看起来能比、其实不能比」。

    左边那条问的是「调度器这一拍发出指令没有」，右边那条问的是
    「有没有任何一个单元在动」。后者天然更容易全绿：TPU 那四条泳道是四个
    物理单元，而 GPU 那六条 warp 抢的是同一个发射端口。
    一份反复强调「这不是分数表」的材料，不该在图上留一处默认读法是打分的地方。
    """
    f.rect(20, CAV_T, 1360, 34, FILL[YL], YL, 1.4, 8)
    f.rect(20, CAV_T, 6, 34, YL, rx=3)
    para(f, 38, CAV_T + 14, 1330,
         "<b>这两条合成条不能直接比高低 —— 它们问的不是同一个问题。</b>"
         "左边问「调度器这一拍发出指令了吗」（六个 warp 抢<b>同一个</b>发射端口，"
         "所以会空）；右边问「有没有任何一个单元在动」（四条泳道是<b>四个不同</b>的物理单元，"
         "天然更容易全绿）。图上还做了一处简化：假定每个 warp 手上只有一条<b>依赖取数结果</b>的指令 —— "
         "真实 kernel 里编译器会插入无关指令来填这些拍，气泡没有图上这么整齐。", "xs", 13)


# ══════════════════════════════════════════════════════════════════════
def _bottom(f):
    # ── 左：occupancy 账 ────────────────────────────────────────────
    x = PX[0]
    f.t(x, MID_T - 10, "那「有多少个 warp 可以换」由什么决定？　—— 寄存器", "sec")
    f.rect(x, MID_T, PW, MID_H, "#fff", BL, 1.8, 10)
    yy = para(f, x + 14, MID_T + 24, PW - 28,
              "一个 SM 有 <b>64K 个 32-bit 寄存器</b>，最多驻留 <b>64 个 warp</b>（2,048 线程）。"
              "warp 是<b>整个现场都留在寄存器堆里</b>才能随时切回来的 —— 切换零开销的代价，"
              "就是所有人的寄存器必须同时占着。所以：", "xs", 15)

    hdr = ["每个线程用的寄存器", "这个 SM 能同时驻留几个 warp", "占满 64 的比例", "够不够盖住上面那 8 拍"]
    cwid = [168, 208, 118, 154]
    rows = [
        ("32 个", "64（撞上限）", "100%", "够，而且有富余", GN),
        ("64 个", "32", "50%", "够", GN),
        ("128 个", "16", "25%", "开始吃紧", YL),
        ("255 个（上限）", "<b>8</b>", "<b>12.5%</b>", "<r>藏不住，气泡成片</r>", RD),
    ]
    ty = yy + 12
    cx0 = x + 16
    f.rect(cx0, ty, PW - 32, 22, FILL[SUB], rx=4)
    cx = cx0 + 8
    for i, hh in enumerate(hdr):
        f.t(cx, ty + 15, hh, "xxs", INK)
        cx += cwid[i]
    for r, (a, b, c_, d, col) in enumerate(rows):
        ry = ty + 24 + r * 21
        cx = cx0 + 8
        for i, v in enumerate([a, b, c_, d]):
            para(f, cx, ry + 13, cwid[i] - 10, v, "xs", 12,
                 fill=col if i == 3 else None)
            cx += cwid[i]

    para(f, x + 14, ty + 118, PW - 28,
         "<b>这就是 CUDA 调优里那条最基本的取舍。</b>循环展开、把中间结果存在寄存器里 —— "
         "这些让单个 warp 变快的手段，同时也在<r>减少能替它顶班的人</r>。", "xs", 14)

    # ── 右：编译期就知道每一拍 ──────────────────────────────────────
    x = PX[1]
    f.t(x, MID_T - 10, "那 TPU 凭什么能排得这么准？　—— 因为延迟是常数", "sec")
    f.rect(x, MID_T, PW, MID_H, "#fff", GN, 1.8, 10)
    yy = para(f, x + 14, MID_T + 24, PW - 28,
              "<b>没有 cache，就没有「命中没命中」这个变量。</b>取一个数要多少拍，"
              "编译期就是已知的：标量个位数、向量几十、矩阵几百。既然是常数，"
              "编译器就能把 WAIT 放在<b>正好那一拍</b>：", "xs", 15)

    steps = [
        (RD, "矩阵槽推进去", "MXU 开始算，接下来几百拍它自己忙自己的"),
        (GN, "向量槽同时在动", "VPU 把下一块权重准备进 Result FIFO"),
        (BL, "DMA 槽同时在搬", "把再下一块数据从 HBM 搬进 VMEM"),
        (YL, "杂项槽放一条 WAIT", "算好放在第几拍 —— 早了空转，晚了 MXU 干等"),
        (PU, "标量单元检查同步位", "MXU_BUSY 已清 → 直接跳下一条，不停顿"),
    ]
    sy = yy + 10
    for i, (c, ttl, note) in enumerate(steps):
        y = sy + i * 24
        f.rect(x + 16, y, 18, 18, FILL[c], c, 1.2, 4)
        f.t(x + 25, y + 13, str(i + 1), "xxs", c, "middle")
        f.t(x + 42, y + 13, ttl, "lbl", INK)
        f.t(x + 196, y + 13, note, "xs")

    para(f, x + 14, sy + 128, PW - 28,
         "<b>代价也很清楚：算错了没有兜底。</b>GPU 猜错只是慢一点，还有别的 warp 顶着；"
         "TPU 这边 WAIT 放早了就是纯空转，放晚了 MXU 就真的在干等。", "xs", 14)

    # ── 通栏结论 ────────────────────────────────────────────────────
    f.t(20, BOT_T, "所以这两句话是同一件事的两面", "sec")
    cards = [
        (BL, "GPU：把不确定性交给硬件",
         "有 cache，延迟就不是常数；不是常数，编译期就排不了班。"
         "于是只能准备一大批随时能上的替补 —— 这直接决定了 SM 长什么样："
         "<b>256 KiB 的寄存器堆</b>（装现场）、<b>64 个 warp 槽</b>（装人）、"
         "<b>4 个独立调度器</b>（挑人）。这些硅不算一次乘法，它们的全部工作是<r>盖住等待</r>。"),
        (GN, "TPU：把不确定性消灭在编译期",
         "没有 cache，延迟就是常数；是常数，就能在编译期把每一拍排满。"
         "于是不需要替补、不需要大寄存器堆、不需要调度器 —— "
         "省下的面积<b>全给了乘加阵列</b>。这也解释了图 G-8 那个结果："
         "同样一块芯片，TPU 能把更大比例的硅放在真正算数的地方。"),
        (PU, "为什么这条差异比带宽差异重要",
         "带宽差个百分之几十，是量的差别；<b>「谁负责知道数在哪」是质的差别</b>。"
         "它决定了两边的编译器长什么样、kernel 怎么写、"
         "调优时该盯哪个指标 —— GPU 盯 occupancy，TPU 盯的是编译器排出来的时间线有没有空隙。"),
    ]
    cw = (1360 - 2 * 16) / 3
    for i, (c, ttl, body) in enumerate(cards):
        bx = 20 + i * (cw + 16)
        f.rect(bx, BOT_T + 14, cw, 128, "#fff", c, 1.8, 10)
        f.rect(bx, BOT_T + 14, cw, 30, FILL[c], rx=10)
        f.rect(bx, BOT_T + 35, cw, 9, FILL[c], rx=0)
        f.t(bx + 14, BOT_T + 34, ttl, "box", c)
        para(f, bx + 14, BOT_T + 62, cw - 28, body, "xs", 15)


if __name__ == "__main__":
    import io, sys
    io.open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/g7.svg", "w",
            encoding="utf-8").write(build())
    print("ok", H)
