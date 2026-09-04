# -*- coding: utf-8 -*-
"""图 T-7 —— 延迟怎么被藏起来：VLIW 一拍多槽，班表在编译期就排好。

前面六张图反复出现同一句话：「这一步是编译期决定的」。这张图把那句话
摊开来看 —— 一拍到底发出去几条指令、谁往哪个槽里填、填不满会怎样。

三条纪律：

1. **bundle 的槽位构成是 v2/v3 论文里的公开数字**，不是 v7 的。v7 的
   bundle 宽度官方没有公开，图上必须写清「这是哪一代的」。
2. **甘特图是示意，不是实测 trace。** 它演示的是排班这件事的形状，
   不是某一次真实运行的时刻表 —— 图上直说，不让人误当数据引用。
3. 落点不是「VLIW 更先进」，而是**代价**：把调度搬到编译期，换来的是
   可预测，赔进去的是「排错了没有兜底」。
"""
from common import Fig, para, BL, RD, YL, GN, PU, TL, INK, SUB, GREY, FILL
import gate

W = 1400
TOP = 84

LX, LW = 20, 800               # 左栏
RX, RW = 848, 532              # 右栏

BUN_Y = TOP + 62               # bundle 示意
BUN_H = 74

# 甘特图占满整幅宽度 —— 上一版把它挤在左半边，一格只有 39px，
# 单拍的块塞不下「发 DMA」这种标签，而右下角同时空出一大片。
GY = BUN_Y + BUN_H + 78        # 甘特图顶
NCYC, CW_, RH = 18, 68.0, 34
GLAB = 96                      # 行标签列宽
GROWS = ["标量槽 ×2", "向量槽 ×4", "矩阵槽 ×2", "杂项槽 ×1"]
GH = len(GROWS) * RH

GBOT = GY + 40 + GH

CARD_Y = GBOT + 88
CARD_H = 158
BAND_Y = CARD_Y + CARD_H + 22
H = BAND_Y + 136


def build():
    f = Fig(W, H, "TPU VLIW 指令包与编译期排班：一拍发出多个槽位，"
                  "延迟必须由编译器提前用独立指令填上，填不上就是真空转")
    f.title("延迟怎么被藏起来　—— 班表在编译期就排好了，跑的时候改不了", "第 7 / 8 张")
    f.legend([(BL, "标量：发 DMA、算地址"), (TL, "向量：VPU"), (GN, "矩阵：MXU"),
              (YL, "杂项"), (GREY, "灰色 ＝ 这一拍这个槽是空的（NOP）")])

    _bundle(f)
    _gantt(f)
    _cards(f)
    _band(f)
    return f.out()


# ══════════════════════════════════════════════════════════════════════
SLOTS = [(BL, "标量", 2), (TL, "向量", 4), (GN, "矩阵", 2), (YL, "杂项", 1)]


def _bundle(f):
    f.t(LX, TOP + 26, "一个 VLIW 指令包 ＝ 一拍要干的所有事，打成一包", "sec")
    # ⛔ 2026-09-04 修一处**字压字**：原来「合计 322 bit」那个 176px 的框
    #    接在槽位条尾巴上，而槽位条本身已经 836px 宽、左栏只有 800px ——
    #    于是它整块伸进右栏，压在「GPU：运行时换一个跑」那段正文上。
    #    ⭐ 根因不是这个框画错了位置，是**一条横向排列的东西没人管它的总宽**。
    #    改法：出处这类元数据本来就不该跟槽位争同一条带，右对齐提到标题行。
    f.t(LX + LW, TOP + 26, "合计 322 bit　·　TPU v2 / v3，公开论文",
        "xxs", GREY, "end")
    para(f, LX, TOP + 46, LW,
         "没有乱序、没有记分板 —— <b>哪条指令在第几拍发、发给哪个单元，全部写死在指令流里</b>。", "xs", 15)

    x = LX
    for c, name, n in SLOTS:
        for k in range(n):
            f.rect(x, BUN_Y, 62, 42, FILL[c], c, 1.6, 5)
            f.t(x + 31, BUN_Y + 26, name, "lbl", c, "middle")
            x += 68
        x += 14
    # 6 个立即数 —— 宽度收到 130，让整条槽位带正好落在左栏 800px 之内
    f.rect(x, BUN_Y, 130, 42, FILL[SUB], SUB, 1.6, 5, "4,3")
    f.t(x + 65, BUN_Y + 26, "立即数 ×6", "lbl", SUB, "middle")

    para(f, LX, BUN_Y + 56, LW,
         "<g>这张槽位构成是 TPU v2 / v3 论文里的公开数字。</g>"
         "<b>v7 的 bundle 宽度和槽位构成，官方没有公开</b>"
         + gate.I("（内部 ISA 定义里见到过 64 B 和 74 B 两种，但我没能确认哪一份对应 v7，"
                  "所以不写进图里）", why="内部 ISA 定义")
         + "。这里画它，是因为<r>「一拍多槽、由编译器填」这个结构本身跨代没变</r>，"
           "变的只是每种槽有几个。", "xs", 15)


# ══════════════════════════════════════════════════════════════════════
# 甘特图的班表 —— 每个元组是 (起拍, 占几拍, 颜色, 标签)
# 演示的是「一次 tile 搬运 + 一次矩阵乘」这段最典型的依赖
SCHED = {
    0: [(0, 1, BL, "发 DMA"), (1, 2, BL, "算下一块地址"), (7, 1, BL, "发下一条")],
    1: [(1, 6, TL, "上一块的归一化 / 激活（独立的活）"), (7, 7, TL, "继续")],
    2: [(7, 7, GN, "矩阵乘：tile 到位才能开始")],
    3: [(0, 1, YL, "同步"), (7, 1, YL, "同步")],
}
WAIT = (0, 7)                   # DMA 在飞的那一段


def _idle(row):
    """没被排上的拍 —— 必须逐格画出来。

    上一版只把结尾那四拍标成灰的，中间同样空着的格子留白 ——
    于是图例说「灰色 ＝ 空」却有一半空拍不是灰的，读者会以为白色另有含义。
    """
    used = [False] * NCYC
    for c0, n, _c, _l in SCHED[row]:
        for k in range(c0, min(c0 + n, NCYC)):
            used[k] = True
    out, k = [], 0
    while k < NCYC:
        if used[k]:
            k += 1
            continue
        j = k
        while j < NCYC and not used[j]:
            j += 1
        out.append((k, j - k))
        k = j
    return out


def _gantt(f):
    f.t(LX, GY, "同一段依赖，编译器怎么排　—— 示意，不是实测 trace", "sec")

    gx = LX + GLAB
    gw = NCYC * CW_

    # DMA 在飞的那一段：只用两条竖虚线圈出来。
    # 不能再铺底纹了 —— 空拍本身就是灰的，底纹会跟它糊在一起。
    for c in WAIT:
        f.line(gx + c * CW_, GY + 22, gx + c * CW_, GY + 44 + GH, SUB, 1.2, dash="4,3")
    f.t(gx + (WAIT[0] + WAIT[1]) / 2 * CW_, GY + 16,
        "DMA 在飞的这几拍，矩阵槽只能空着 —— 编译器要么找到别的活填进来，要么认了", "xxs", SUB, "middle")

    # 拍号
    for c in range(NCYC):
        f.t(gx + c * CW_ + CW_ / 2, GY + 34, str(c), "xxs", GREY, "middle")

    for r, name in enumerate(GROWS):
        y = GY + 40 + r * RH
        f.t(LX, y + 20, name, "lbl", [BL, TL, GN, YL][r])
        f.rect(gx, y, gw, RH - 6, "#fff", "#e8eaed", 1.0, 3)
        for (c0, n) in _idle(r):                    # 先铺空拍，再画排上的活
            f.rect(gx + c0 * CW_ + 1.5, y + 2, n * CW_ - 3, RH - 10,
                   FILL[GREY], GREY, 1.2, 3, "3,3")
        for (c0, n, col, lab) in SCHED[r]:
            f.rect(gx + c0 * CW_ + 1.5, y + 2, n * CW_ - 3, RH - 10,
                   FILL[col], col, 1.2, 3, "3,3" if col is GREY else None)
            if lab:
                f.t(gx + c0 * CW_ + 8, y + 19, lab, "xxs", col)

    # 结尾那段真空转，单独点出来
    ex = gx + 14 * CW_
    f.line(ex, GBOT + 12, gx + NCYC * CW_, GBOT + 12, RD, 1.6)
    f.line(ex, GBOT + 8, ex, GBOT + 16, RD, 1.6)
    f.line(gx + NCYC * CW_, GBOT + 8, gx + NCYC * CW_, GBOT + 16, RD, 1.6)
    para(f, ex, GBOT + 32, 300,
         "<r>第 14 拍之后所有槽都空了</r> —— 独立的活用完了。"
         "这四拍<b>不会有任何东西自动顶上来</b>，因为没有第二个线程可切。", "xxs", 13)
    para(f, LX, GBOT + 32, 620,
         "第 1–6 拍的向量槽里塞的是<b>上一块</b>的后处理 —— 和这次搬运没有依赖关系，"
         "所以能提前挪过来。<b>这就是编译器「藏延迟」的全部手段：找独立的活。</b>", "xxs", 13)


# ══════════════════════════════════════════════════════════════════════
def _cards(f):
    f.t(RX, TOP + 26, "同一个延迟，两边怎么处理", "sec")

    # 对照
    f.rect(RX, TOP + 42, RW, 168, "#fff", INK, 1.8, 10)
    f.rect(RX, TOP + 42, RW / 2, 168, FILL[BL], rx=10)
    f.line(RX + RW / 2, TOP + 42, RX + RW / 2, TOP + 210, "#e8eaed", 1.2)

    f.t(RX + 16, TOP + 68, "GPU：运行时换一个跑", "box", BL)
    para(f, RX + 16, TOP + 88, RW / 2 - 32,
         "warp 卡住 → 调度器立刻挑一个能跑的。<b>代价是要同时驻留几十份上下文</b>，"
         "那 256 KB 寄存器堆就是为它准备的。<br>"
         "好处：形状不规则也能跑得不太难看。<br>"
         "坏处：同一份代码，两次跑的耗时可以差很多。"
         .replace("<br>", " "), "xxs", 14)

    f.t(RX + RW / 2 + 16, TOP + 68, "TPU：编译期提前挪", "box", GN)
    para(f, RX + RW / 2 + 16, TOP + 88, RW / 2 - 32,
         "编译器把独立的活挪进空拍。<b>不需要驻留上下文，那片硅省下来给了 MXU 和 VMEM</b>。 "
         "好处：耗时高度可预测，同样的形状每次都一样快。 "
         "坏处：<r>挪不动就是真空转</r>，运行时没有第二次机会。", "xxs", 14)

    # 三个实际后果 —— 铺满整幅宽度的三栏，别再竖着堆在右下角
    f.rect(20, CARD_Y, 1360, CARD_H, FILL[RD], RD, 1.8, 10)
    f.t(34, CARD_Y + 26, "这个取舍在日常里长什么样　—— 三件事其实是同一件事", "sec", RD)
    items = [
        ("① 形状一不规则，性能就掉得很难看",
         "不是编译器不肯优化，是<b>可挪的独立指令本来就少了</b>。"
         "batch 小、序列不齐、专家路由不均 —— 这几种情况的共同点都是"
         "「后一步紧跟着前一步」，空拍没东西填。"),
        ("② 编译很慢，而且慢得有道理",
         "v7 上一次 XLA 编译常见 <b>10–17 分钟</b>。"
         "它不是在「翻译」，是在<b>替硬件把整张班表排完</b> —— "
           "GPU 那边这件事是每次运行时由调度器现做的，成本摊在了跑的时候。"),
        ("③ 但跑起来非常稳",
         "同一个形状重复跑，步时几乎没有抖动 —— 因为根本没有运行时决策可抖。"
         "<b>可预测性是这个设计买来的东西，不是附带效果</b>：容量规划、"
         "性能回归检测在 TPU 上都因此简单不少。"),
    ]
    cw = (1360 - 28 - 2 * 24) / 3.0
    for i, (ttl, body) in enumerate(items):
        cx = 34 + i * (cw + 24)
        f.t(cx, CARD_Y + 56, ttl, "lbl", RD)
        para(f, cx, CARD_Y + 76, cw, body, "xxs", 14)


# ══════════════════════════════════════════════════════════════════════
def _band(f):
    f.rect(20, BAND_Y, 1360, 116, FILL[TL], TL, 1.6, 10)
    f.t(34, BAND_Y + 26, "把 §2 那句话补完整", "sec", TL)
    para(f, 34, BAND_Y + 50, 1330,
         "§2 说 TPU「省下了调度器、寄存器堆、记分板那片硅」。"
         "<b>这张图是那句话的另一半：省下来的东西并没有消失，它被搬到了编译期。</b>"
         "调度这件事总得有人做 —— 区别只在于是硬件每次运行时重做一遍，还是编译器一次做完。", "xs", 18)
    para(f, 34, BAND_Y + 80, 1330,
         "所以「TPU 更简单」是个误解。它<r>不是把复杂度删掉了，是把复杂度换了个地方放</r> —— "
         "从硅上换到了编译器里，也从运行时换到了你写模型配置的那一刻。", "xs", 18)


if __name__ == "__main__":
    import io, sys
    io.open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/t7.svg", "w",
            encoding="utf-8").write(build())
    print("ok", H)
