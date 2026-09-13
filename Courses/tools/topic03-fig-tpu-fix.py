# -*- coding: utf-8 -*-
r"""专题三 · §10.3「那怎么克服 —— 三招，以及两个一定会被问到的问题」

⭐⭐⭐ 2026-09-13 **整张重画**，接着上一张那家中央厨房往下讲。

  ① **三招的共同形状：把「一个临时改单」换成「一批预制套餐」。**
     来单的时候**挑一个**，而不是现开火。
  ② **那个「今天做哪几道菜」的决定，谁来算？要不要问前台？**
     ⭐⭐ 答案分两层，画成**前台 vs 后厨**：
     · **前台（host CPU）**：今天有几桌、每桌几个人 ——&nbsp;
       本来就在它手上，**一顿饭只报一次，可以忽略**。
     · **后厨（就在卡上）**：这道菜从哪个货架拿 ——&nbsp;
       **每道菜都要算，跑去问前台一次就废了。**
     ⭐ 而且它用的是**本来就闲着的那个人**：颠勺的时候（矩阵乘忙），
     算账那位（标量单元）正没事干，地址计算恰好是他的活。
  ③ **SparseCore 能不能干这个？** 画成一支**专门跑腿拣货的小队**：
     天生擅长散落取货 ——&nbsp;⚠️ 但要**先报一个数量上限**。
     对 DSA 恰好天然满足（k 就是 2048）。
     ⛔ 可公开的那套生产 kernel 走的是主厨这条线，**不是拣货小队**。
"""
from topic03_draw import (Fig, BL, OR, GR, RD, GY, PU, INK, GY2, LINE, LINE2,
                          BG2)

W = 1400


def main():
    f = Fig(W, "怎么克服：三招的共同形状是把一个临时改单换成一批预制套餐；"
               "那个运行时的决定分两层 —— 前台报几桌几人，后厨自己算货架地址，"
               "而且用的是颠勺时闲着的那个算账的人；SparseCore 像一支拣货小队")
    f.marks = set()
    y0 = f.header(
        "那怎么克服 ——　三招，和两个一定会被问到的问题",
        "共同形状：<tspan font-weight=\"700\">把「一个临时改单」换成「一批预制套餐」</tspan>",
        [(GR, "三招"), (BL, "谁来算"), (PU, "拣货小队"), (RD, "还没被验证的")])

    # ⭐⭐⭐ 2026-09-13 重画。审图原话：「**当前它就是一块写字的板子** ——
    #   七张要点卡，零个画面。中央厨房这个全课最好的比喻，一笔都没画。」
    # ⛔ 比喻写在脚本注释里、也写在卡片文字里 —— 唯独**没落到图上**。
    #   注释里的比喻救不了读者，他看到的只有卡片。

    # ══════════ ① 三招：保温台上的预制套餐 ══════════════════════
    PH = 372
    py = f.panel(0, y0, W, PH, "① 三招 ——　都是「不现开火，改成挑一个预制的」",
                 GR, sub="RPA 论文的三个做法")

    ay = py + 26
    # 保温台
    f.box(56, ay + 84, 800, 14, "#f1f3f4", GY2, 4)
    f.t(56, ay + 122, "保温台：开工前就把几套做好摆上", GY2, size=16)
    TRAYS = [("全长的", False), ("全短的", False), ("混着的", True), ("留一格", False)]
    for i, (lab, pick) in enumerate(TRAYS):
        tx = 76 + i * 196
        f.box(tx, ay + 24, 156, 58, "#e6f4ea" if pick else "#fff",
              GR if pick else LINE2, 8, 2.4 if pick else 1)
        f.t(tx + 78, ay + 58, lab, GR if pick else GY, pick, 19, "middle")
        if pick:
            f.line(tx + 78, ay + 6, tx + 78, ay + 20, GR, 2.2)
            f.t(tx + 78, ay - 4, "来单了，挑这套", GR, True, 17, "middle")

    f.box(896, ay + 12, 448, 116, "#fff", GR, 10)
    f.t(920, ay + 46, "⭐ 共同形状", GR, True, 20)
    f.t(920, ay + 76, "把「一个临时改单」", GY, size=17)
    f.t(920, ay + 102, "换成「一批预制套餐」", GY, size=17)

    FIX = [("1. 把盘子切小一点", "强制用最小的那种餐盒 ——　长短不一的那一维，别放在切盘子的方向上"),
           ("2. 上菜和收盘并成一趟", "decode 时那一下零碎的写，融进主菜一起做，用做菜的时间盖住它"),
           ("3. 按客流预制几套套餐", "⭐ 最像中央厨房：不做万能菜谱，做几套再挑 ——　就是上面这张图")]
    for i, (h_, d_) in enumerate(FIX):
        f.t(56, ay + 158 + i * 44, h_, GR, True, 19)
        f.t(330, ay + 158 + i * 44, d_, GY, size=17, w=1010)
    f.box(56, ay + 286, 1288, 34, "#e6f4ea", GR, 8)
    f.t(76, ay + 310, "⭐ 成绩：Llama 3 8B 在 TPU7x 上 ——　"
        "decode MBU 86%　·　prefill MFU 73%", GR, True, 20)

    # ══════════ ② 前台 vs 后厨：画成一张平面图 ══════════════════
    y1 = y0 + PH + 18
    PH2 = 326
    py2 = f.panel(0, y1, W, PH2,
                  "② 那个「今天做哪几道菜」的决定，谁来算　——　答案分两层",
                  BL, sub="别答成一个字")

    by = py2 + 26
    # 前台
    f.box(56, by + 16, 340, 176, "#f8f9fa", GY2, 10)
    f.t(226, by + 50, "前台", GY, True, 24, "middle")
    f.t(226, by + 74, "host CPU", GY2, size=16, anchor="middle")
    f.box(116, by + 92, 160, 62, "#fff", GY2, 6)          # 台卡
    f.t(196, by + 118, "今天 12 桌", INK, True, 20, "middle")
    f.t(196, by + 142, "每桌几个人", GY2, size=16, anchor="middle")
    f.t(226, by + 178, "一顿饭只报一次", GY, True, 18, "middle")

    # 中间那条「跑去问前台」的路 —— 打叉
    f.line(400, by + 104, 560, by + 104, RD, 2.0, dash="6,5")
    f.t(480, by + 90, "每道菜都跑去问？", RD, size=16, anchor="middle")
    for dx, dy in ((-14, -14), (-14, 14)):
        f.line(480 - dx, by + 104 - dy, 480 + dx, by + 104 + dy, RD, 3.0,
               arrow=False)
    f.t(480, by + 148, "一来一回是微秒级，", RD, size=16, anchor="middle")
    f.t(480, by + 170, "而这一步只有几十微秒", RD, size=16, anchor="middle")

    # 后厨
    f.box(564, by + 16, 420, 176, "#e8f0fe", BL, 10)
    f.t(774, by + 50, "后厨（就在卡上）", BL, True, 24, "middle")
    for k in range(5):                                     # 货架
        f.box(590 + k * 78, by + 70, 66, 54, "#fff", BL, 5)
        f.t(623 + k * 78, by + 102, "%d 号" % (k + 1), BL, size=16,
                anchor="middle")
    f.t(774, by + 150, "这道菜的料从哪个货架 ——　自己算", BL, True, 18, "middle")
    f.t(774, by + 178, "每道菜都要算一次", GY, size=17, anchor="middle")

    # 闲着的那个人
    f.box(1004, by + 16, 340, 176, "#e6f4ea", GR, 10)
    f.t(1024, by + 50, "⭐⭐ 这笔账是拿闲人付的", GR, True, 20)
    f.box(1024, by + 68, 140, 52, "#fff", GR, 6)
    f.t(1094, by + 100, "颠勺的", GY, True, 18, "middle")
    f.box(1180, by + 68, 140, 52, "#fff", GR, 6)
    f.t(1250, by + 92, "算账的", GR, True, 18, "middle")
    f.t(1250, by + 112, "正没事干", GR, size=15, anchor="middle")
    f.t(1024, by + 148, "矩阵乘忙得冒烟的时候，", GY, size=17, w=316)
    f.t(1024, by + 174, "标量单元正闲着 ——　地址计算是他的活", GY, size=16,
        w=316)

    f.t(56, by + 230, "⭐ 所以这一问的答案是两层："
        "<tspan font-weight=\"700\">「今天有几桌」前台报，一顿饭一次；"
        "「这道菜从哪个货架拿」后厨自己算，每道菜一次。</tspan>"
        "　⛔ 混成一句就必错。", INK, size=19, w=1300)

    # ══════════ ③ 拣货小队 ══════════════════════════════════════
    y2 = y1 + PH2 + 18
    PH3 = 312
    py3 = f.panel(0, y2, W, PH3, "③ 那 SparseCore 能不能干这个",
                  PU, sub="一支专门跑腿拣货的小队")

    ey = py3 + 26
    # 小队 ＋ 推车
    for k in range(3):
        f.box(76 + k * 62, ey + 30, 46, 56, "#f3e8fd", PU, 6)
        f.t(99 + k * 62, ey + 64, "拣", PU, True, 18, "middle")
    f.line(268, ey + 58, 300, ey + 58, PU, 2.0)
    f.box(308, ey + 26, 220, 64, "#fff", OR, 8)
    f.t(418, ey + 52, "推车", OR, True, 19, "middle")
    f.t(418, ey + 78, "最多 2048 件", OR, True, 20, "middle")
    f.t(76, ey + 116, "⚠️ 规矩：这一趟最多拿几件，<tspan font-weight=\"700\">"
        "必须开工前就报</tspan>　超了就分批，或者直接丢掉一部分", GY, size=17,
        w=820)
    f.t(76, ey + 146, "⭐ 对 DSA 反而天然满足 ——　k 就是 2048，定死的", OR,
        True, 19)

    f.box(872, ey + 20, 472, 148, "#f3e8fd", PU, 10)
    f.t(896, ey + 52, "✅ 架构上非常对口", PU, True, 20)
    for i, ln in enumerate(["天生干散落取货（不规则、稀疏访存）",
                            "能按条件决定去哪儿拿",
                            "跨通道排序、过滤、前缀和 ——　正是 top-k 要的"]):
        f.t(896, ey + 84 + i * 28, "· " + ln, GY, size=16, w=424)

    f.box(56, ey + 190, 1288, 76, "#fce8e6", RD, 8)
    f.t(76, ey + 220, "⛔ 但要诚实：公开的那套 TPU 生产注意力 kernel 走的是"
        "<tspan font-weight=\"700\">主厨</tspan>这条线"
        "（TensorCore ＋ Pallas/Mosaic），", RD, size=19, w=1248)
    f.t(76, ey + 250, "<tspan font-weight=\"700\">不是拣货小队</tspan>。"
        "没有公开材料说有人用它跑注意力的 top-k ——　台下如果有 TPU 的人，"
        "含糊一句就会被抓住。", RD, size=19, w=1248)

    # ══════════ 落点 ════════════════════════════════════════════
    yy = y2 + PH3 + 20
    yy = f.band(yy, "warn", "⛔ 第三格这条要留在「看起来很对、但还没被公开验证」上", [
        "SparseCore 的公开资料说它是「为<tspan font-weight=\"700\">不规则、稀疏访存"
        "</tspan>做的专用处理器」，并且<tspan font-weight=\"700\">原生支持"
        "数据相关的控制流与访存</tspan>、能做跨 lane 的排序 / 过滤 / 前缀和。",
        "⛔ 但<tspan font-weight=\"700\">没有公开材料</tspan>说有人用它跑注意力的 top-k。"
        "⭐ 台下如果有 TPU 的人，含糊一句就会被抓住 ——&#160;"
        "<tspan font-weight=\"700\">照着这行念。</tspan>",
    ])

    yy = f.band(yy + 14, "info", "⭐⭐ 跨层共享那一支，在这台机器上比在 GPU 上更值钱", [
        "<tspan font-weight=\"700\">GPU 上省的是</tspan>：索引器那部分算力"
        "（GLM-5.2 报 1M 下每 token 降 2.9×，见 §6.5b）。",
        "<tspan font-weight=\"700\">TPU 上还额外省三样</tspan>："
        "① 「这一趟拿哪几件」只算一次，后面几层直接复用；"
        "② 几层的取货路线<tspan font-weight=\"700\">完全一样</tspan>，推车的单子可以重用；"
        "③ <tspan font-weight=\"700\">临时改单的次数本身降了四倍</tspan>。",
        "⭐ 最后那条才是这一节真正想留下的判据："
        "<tspan font-weight=\"700\">在一家「按批预制」的厨房里，改单的次数本身就是成本"
        "</tspan> ——&#160;不只是每次改单有多贵。"
        "⚠️ 这是本课从 RPA 描述的机制推出来的，<tspan font-weight=\"700\">"
        "没有公开的对照实测</tspan>。",
    ])

    yy = f.src(yy + 16,
               "三招、「SREG 在计算密集阶段欠用」、以及 MBU 86% / MFU 73%，"
               "均出自 Ragged Paged Attention（Jiang 等 arXiv 2604.15464）§3–§5",
               "SparseCore 的定位与「必须声明静态上界、超了就 mini-batch 或丢 ID」"
               "出自 openxla.org 的 SparseCore 公开文档",
               "⚠️ 「中央厨房 / 前台后厨 / 拣货小队」是"
               "<tspan font-weight=\"700\">本课的比喻</tspan>；"
               "⚠️ 跨层共享在 TPU 上更值钱那一条是<tspan font-weight=\"700\">本课的推导"
               "</tspan>，无公开对照实测")
    f.save("fig3-tpu-fix.svg", yy + 6)


main()
