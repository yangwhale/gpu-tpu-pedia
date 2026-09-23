# -*- coding: utf-8 -*-
r"""专题四 · §1.2d「结账」——&#160;权重梯度到底在什么时候、在哪一步相加

⭐⭐⭐ 2026-09-22 现场原话（这是这一讲被追问得最深的一处，值得整张图）：
  「每一个函数就这么搞一回，搞一回再往前传，**就每一个函数这个位置都得做
    一次权重的平均，然后再往前搞，是这个意思吧？还是说把这一层算完了再平均？**」
  「这个篇幅太重要了，咱别怕内容多……甚至这课不干别的，
    就把这个地方讲明白就已经不错了。」

⭐ 他自己推到的那一步**是对的**，原话：
  「我当前的这个 token 得用生成我这个 token 的激活，
    也就是说我前面的所有的 token 都参与计算之后的激活来做偏导数。」
  ——&#160;完全正确，而且点到了要害：**线性层的权重梯度只用本位置的输入**，
  前面那些 token 是**在前向的 attention 里就已经被搅进 x_i 了**，
  不是在反向的时候才被拉进来。

⛔ 而他问的那个二选一，答案是**前者**：
  **每一个带权重的算子各自结一次账**，不是「一层算完再结」。
  而且更准确：**「结账」和「往前传」是同一步里的两件事，互不等待。**

⭐⭐⭐ 这张图真正的价值在于它把 §1.6 那句话接上了：
  **反向要付两笔乘法** ——&#160;现在可以说清那两笔各自在干嘛：
    · 一笔<b>沿位置维求和</b> →&#160;权重梯度（位置维被吃掉，账结在这儿）
    · 一笔<b>逐位置保留</b>   →&#160;传给上游的 δ（位置维一路留着）
  **同一张 δ 表，两种收缩方式。** 「1 ＋ 2 ＝ 3」那个 2，就是这两笔。

⛔⛔ 两个特别容易错的地方，图上都钉住了：
  ① **权重梯度的桶是固定大小的**，不随 token 数变大 ——&#160;
    每个位置只是往同一个桶里「加一笔」。这就是为什么整份梯度是
    「每个参数一个数」，而不是「每个 token 一套参数」。
  ② **每层只加不除。** 那个 ÷N 在最开头的种子里就做过一次了；
    每层再除一次的话，六十一层就除了六十一次。

📌 V3 口径：61 层（3 dense ＋ 58 MoE）、宽 7,168、专家腰 2,048、
  8 个路由专家 ＋ 1 个共享专家。⚠️ MoE 那一段有个额外的点：
  **一个 token 只往它选中的那几个专家的桶里加**，没被选中的专家这一步收不到。
"""
from topic03_draw import Fig, BL, OR, GR, RD, PU, GY, INK, GY2

W = 1400

DM = 7168           # 残差流宽度
# ⛔⛔ 2026-09-22 现场：「不要写 7168×7168，这个虽然是 Input 7168 和
#   Output 7168，但是里边**从来没有出现过这么样一个方阵**。」
#   ⭐ 核对了一遍：V3 一层里**没有任何一块权重是方阵** ——&#160;
#     专家那几块是 7,168 → 2,048、MLA 把 KV 压到 512 那一档，
#     而 o_proj 那一边反而更宽（16,384）。⛔ 注意别把它说成「全是瘦的」——&#160;
#     **准确的说法只有一条：没有一块两边都是 7,168。**
#     原来那么写是我顺手拿宽度平方了，
#     属于「听起来像常识的架构关系」——&#160;本仓库第一原则点名的那一类。
#   ⇒ 这张图改用一个**真实存在**的算子当例子：专家的 up 投影。
D_IN, D_OUT = 7168, 2048          # 专家 up：7,168 进、2,048 出
TGT = 4095
LAYERS = 61


def main():
    f = Fig(W, "一个带权重的算子在反向时做两件事，"
               "而这两件事就是第一节说的那两笔乘法："
               "一笔是把这一层的责任和这一层的输入相乘、沿位置求和，"
               "得到对权重的改动，位置这一维在这一笔里被吃掉，账就结在这儿；"
               "另一笔是把责任乘上权重本身，得到该传给上游的新责任，"
               "位置这一维在这一笔里完整保留。"
               "所以结账是每一个带权重的算子各结一次，不是一层算完再结，"
               "而且结账和往前传互不等待。"
               "权重梯度的桶是固定大小的，每个位置只是往同一个桶里加一笔；"
               "而且每层只加不除，那个除以总数在最开头的种子里就做过一次了")

    y0 = f.header(
        "「结账」——&#160;<tspan font-weight=\"700\">权重梯度是在哪一步相加的</tspan>",
        "⭐ 现场问：「每一个函数都做一次，还是把这一层算完再做？」——　"
        "<tspan font-weight=\"700\">每一个带权重的算子各结一次，而且不等待</tspan>",
        [(GR, "Ⓐ 两笔乘法"), (BL, "Ⓑ 一层里谁结账"),
         (OR, "Ⓒ 桶是固定大小的")])

    # ══════════ Ⓐ 同一张 δ 表，两种收缩 ═════════════════════════════
    PH = 400
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 一个带权重的算子，反向时做<tspan font-weight=\"700\">两件事</tspan>"
                 "　——　这就是第一节说的那两笔乘法", GR,
                 sub="⭐ <tspan font-weight=\"700\">同一张 δ 表，两种收缩方式</tspan>："
                     "一笔把位置维吃掉，一笔把位置维留着")

    # 中间：这一层的两张表
    f.box(560, py + 52, 280, 86, "#e8f0fe", BL, 6)
    f.t(700, py + 82, "δ　（出口那一侧的责任）", BL, True, 15, "middle")
    f.t(700, py + 112, "%s 行 × %s 列" % (format(TGT, ","), format(D_OUT, ",")),
        INK, True, 15, "middle")
    f.box(560, py + 150, 280, 76, "#f1f3f4", GY2, 6)
    f.t(700, py + 178, "x　（入口那一侧的输入）", GY, True, 14, "middle")
    f.t(700, py + 206, "%s 行 × %s 列" % (format(TGT, ","), format(D_IN, ",")),
        GY, size=14, anchor="middle")

    # 左：结账
    f.box(40, py + 52, 460, 296, "#e6f4ea", GR, 8)
    f.t(270, py + 88, "① 结账　——　<tspan font-weight=\"700\">位置维被吃掉</tspan>",
        GR, True, 17, "middle")
    f.t(270, py + 128, "<tspan font-weight=\"700\">δ　与　x　相乘，沿着「位置」求和</tspan>",
        INK, True, 15.5, "middle")
    f.t(270, py + 162, "得到一张 %s × %s 的改动表　——　跟这块权重一样大"
        % (format(D_OUT, ","), format(D_IN, ",")), INK, size=14.5, anchor="middle")
    f.t(270, py + 196, "⭐ <tspan font-weight=\"700\">%s 这个数，在这一笔里消失了</tspan>"
        % format(TGT, ","), GR, True, 14.5, "middle")
    f.t(270, py + 228, "——　所有位置对这块权重的意见，", GY, size=13.5, anchor="middle")
    f.t(270, py + 252, "<tspan font-weight=\"700\">在这里合成了一个数</tspan>。",
        GY, True, 13.5, "middle")
    f.t(270, py + 292, "⛔ 这一笔<tspan font-weight=\"700\">不往下传</tspan>　——",
        RD, True, 14, "middle")
    f.t(270, py + 318, "它就留在这块权重的账上，等更新那一刻。",
        GY2, size=13, anchor="middle")
    f.line(552, py + 96, 508, py + 96, GR, 2.4)

    # 右：往前传
    f.box(900, py + 52, 460, 296, "#e8f0fe", BL, 8)
    f.t(1130, py + 88, "② 往前传　——　<tspan font-weight=\"700\">位置维留着</tspan>",
        BL, True, 17, "middle")
    f.t(1130, py + 128, "<tspan font-weight=\"700\">δ　乘上权重本身</tspan>",
        INK, True, 15.5, "middle")
    f.t(1130, py + 162, "得到一张<tspan font-weight=\"700\">还是 %s 行</tspan>的新表"
        % format(TGT, ","), INK, size=14.5, anchor="middle")
    f.t(1130, py + 196, "⭐ <tspan font-weight=\"700\">每个位置仍然各是各的</tspan>",
        BL, True, 14.5, "middle")
    f.t(1130, py + 228, "——　它就是上一层要收到的那份责任。",
        GY, size=13.5, anchor="middle")
    f.t(1130, py + 292, "⭐ 位置这一维，<tspan font-weight=\"700\">一路留到最前面</tspan>。",
        BL, True, 14, "middle")
    f.line(848, py + 96, 892, py + 96, BL, 2.4)

    f.t(700, py + 368,
        "⭐⭐⭐ 所以第一节那句「反向要付<tspan font-weight=\"700\">两笔</tspan>乘法，"
        "所以训练比推理贵三倍」——　<tspan font-weight=\"700\">那两笔，就是这两笔。</tspan>",
        INK, size=15, anchor="middle")
    f._pan = None

    # ══════════ Ⓑ 一层里谁结账、谁只是路过 ═════════════════════════
    PH2 = 350
    py2 = f.panel(0, py + PH + 20, W, PH2,
                  "Ⓑ 一层里有十几个算子　——　"
                  "<tspan font-weight=\"700\">带权重的各结各的账，不带权重的只是路过</tspan>",
                  BL,
                  sub="⛔ 回答现场那个二选一：<tspan font-weight=\"700\">是「每个算子一次」，"
                      "不是「一层算完再一次」</tspan>")

    OPS = (
        ("归一化", 1), ("Q/K/V 投影", 1), ("注意力打分", 0), ("softmax", 0),
        ("加权求和", 0), ("输出投影", 1), ("残差相加", 0), ("归一化", 1),
        ("路由打分", 1), ("专家 gate/up", 1), ("SwiGLU 相乘", 0),
        ("专家 down", 1), ("残差相加", 0),
    )
    bw = (W - 80) / float(len(OPS))
    for k, (name, has_w) in enumerate(OPS):
        x = 40 + k * bw
        col = OR if has_w else GY2
        fill = "#fef7e0" if has_w else "#f1f3f4"
        f.box(x + 3, py2 + 56, bw - 6, 78, fill, col, 5, sw=1.3)
        f.t(x + bw / 2, py2 + 90, name, INK if has_w else GY, has_w, 11.5, "middle")
        f.t(x + bw / 2, py2 + 118, "结账" if has_w else "只路过",
            col, True, 11, "middle")
        if k < len(OPS) - 1:
            f.line(x + bw - 2, py2 + 95, x + bw + 2, py2 + 95, GY2, 1.0, arrow=False)
    f.line(W - 40, py2 + 152, 40, py2 + 152, PU, 2.0)
    f.t(W / 2, py2 + 178, "δ 从右往左走，<tspan font-weight=\"700\">"
        "每碰到一个橙色的就顺手结一次账</tspan>", PU, True, 14.5, "middle")

    f.t(W / 2, py2 + 222,
        "⭐ <tspan font-weight=\"700\">橙色的：自己有参数，所以有账要结。</tspan>"
        "　　灰色的：自己没有参数（打分、softmax、逐元素相乘、加法），"
        "<tspan font-weight=\"700\">只把责任改个形状递过去</tspan>。",
        INK, size=14, anchor="middle")
    f.t(W / 2, py2 + 254,
        "⛔ 所以<tspan font-weight=\"700\">「一层算完再平均」是没有的</tspan>　——　"
        "根本没有任何东西需要等一层走完。",
        RD, True, 14.5, "middle")
    f.t(W / 2, py2 + 292,
        "📌 MoE 那几格还有一条：<tspan font-weight=\"700\">"
        "一个 token 只往它选中的那几个专家的账上加</tspan>，"
        "没被选中的专家这一步收不到任何东西。",
        GY, size=13.5, anchor="middle")
    f.t(W / 2, py2 + 320,
        "⚠️ 上面这一排是<tspan font-weight=\"700\">示意</tspan>："
        "真实的 MLA ＋ MoE 一层比这还多几格，但「谁结账」这件事一模一样。",
        GY2, size=12.5, anchor="middle")
    f._pan = None

    # ══════════ Ⓒ 桶 ═══════════════════════════════════════════════
    PH3 = 320
    py3 = f.panel(0, py2 + PH2 + 20, W, PH3,
                  "Ⓒ 每块权重有一个<tspan font-weight=\"700\">固定大小的桶</tspan>"
                  "　——　所有位置往同一个桶里加", OR,
                  sub="⭐ 这就是为什么整份梯度是「每个参数一个数」，"
                      "而不是「每个 token 一套参数」")

    # 多个位置 → 一个桶
    for k in range(7):
        yy = py3 + 58 + k * 26
        f.box(90, yy, 150, 20, "#e8f0fe", BL, 3)
        f.t(165, yy + 15, "位置 %d 的那一笔" % (k + 1), BL, size=11, anchor="middle")
        f.line(248, yy + 10, 356, py3 + 150, OR, 1.2, arrow=False)
    f.t(165, py3 + 254, "…… 共 %s 笔（batch＝2 就是 %s 笔）"
        % (format(TGT, ","), format(TGT * 2, ",")), GY2, size=12.5, anchor="middle")

    f.box(370, py3 + 96, 250, 110, "#fef7e0", OR, 8, sw=2.0)
    f.t(495, py3 + 134, "这块权重的桶", OR, True, 16, "middle")
    f.t(495, py3 + 166, "%s × %s 个数" % (format(D_OUT, ","), format(D_IN, ",")),
        INK, True, 15, "middle")
    f.t(495, py3 + 192, "⛔ 大小固定，永远不变", RD, True, 12.5, "middle")

    f.box(670, py3 + 56, 690, 224, "#fef7e0", OR, 8)
    f.t(1015, py3 + 92, "两条最容易搞错的：", OR, True, 17, "middle")
    f.t(1015, py3 + 132,
        "① <tspan font-weight=\"700\">桶不会因为 token 多就变大</tspan>　——　"
        "四千个位置只是", INK, size=14.5, anchor="middle")
    f.t(1015, py3 + 158,
        "往同一个桶里<tspan font-weight=\"700\">加了四千笔</tspan>，桶还是那么大。",
        INK, size=14.5, anchor="middle")
    f.t(1015, py3 + 200,
        "② <tspan font-weight=\"700\">每层只加，不除。</tspan>"
        "那个「÷ 总数」在最开头的种子里", INK, size=14.5, anchor="middle")
    f.t(1015, py3 + 226,
        "就做过<tspan font-weight=\"700\">一次</tspan>了　——　"
        "每层再除一遍的话，%d 层就除了 %d 次。" % (LAYERS, LAYERS),
        INK, size=14.5, anchor="middle")
    f.t(1015, py3 + 260,
        "⭐ 所以严格说这一步叫<tspan font-weight=\"700\">求和</tspan>，不叫求平均　——　"
        "平均早就摊进每一笔里了。", GY, size=13.5, anchor="middle")
    f._pan = None

    yb = f.band(py3 + PH3 + 18, "ok", "⭐ 这张图的两句话", [
        "<tspan font-weight=\"700\">一个带权重的算子，反向时做两件事</tspan>　——　"
        "「δ 配本位置的输入、沿位置求和」结出权重梯度（位置维被吃掉）；"
        "「δ 乘权重」结出传给上游的责任（位置维留着）。<tspan font-weight=\"700\">"
        "那就是第一节那两笔乘法。</tspan>",
        "<tspan font-weight=\"700\">所以结账是「每个带权重的算子各一次」，"
        "不是「一层算完再一次」</tspan>　——　而且桶的大小固定、每层只加不除。",
    ])

    yb = f.src(yb + 10,
               "⭐ 现场自己推到的那一步是对的：「我当前这个 token 得用生成我这个 token "
               "的激活，也就是前面所有 token 都参与计算之后的激活」——&#160;"
               "线性层的权重梯度<tspan font-weight=\"700\">只用本位置的输入</tspan>，",
               "而前面那些 token 是<tspan font-weight=\"700\">在前向的 attention 里"
               "就已经被搅进这个输入了</tspan>，不是反向时才拉进来的。"
               "⛔ 口径：V3 一层里<tspan font-weight=\"700\">没有任何一块 "
               "%s × %s 的方阵</tspan> ——&#160;残差流是 %s 宽，"
               "但<tspan font-weight=\"700\">每一块的另一边都是别的数</tspan>"
               "（专家 %s → %s、MLA 把 KV 压到 512 那一档，而 o_proj 那一边是 16,384）。"
               "本图用专家 up 当例子。共 %d 层。"
               % (format(DM, ","), format(DM, ","), format(DM, ","),
                  format(D_IN, ","), format(D_OUT, ","), LAYERS))

    f.save("fig4-settle.svg", yb + 14)


# ══════════════════════════════════════════════════════════════════
# 第二张：只有「顺序」那一格。⭐ 课件用 <details> 收着（__FIG_SETTLE_ORDER__）。
# ══════════════════════════════════════════════════════════════════
def order():
    g = Fig(W, "结账这件事不需要排队等：哪一层收到上游传来的责任，哪一层就能立刻把"
               "自己那笔权重梯度算掉，谁也不用等谁。"
               "而这正是多卡训练能把通信藏进计算里的原因 —— "
               "最后几层的账早就结完了，前面几层还在算，"
               "那段时间正好拿来把已经结完的那部分传出去")

    y0d = g.header(
        "补一格：<tspan font-weight=\"700\">结账不用排队</tspan>",
        "⭐ 这一格在<tspan font-weight=\"700\">这一讲</tspan>里不承重；"
        "它真正派上用场是在<tspan font-weight=\"700\">专题五 · 并行策略</tspan>",
        [(PU, "顺序：谁也不用等谁")])

    # ══════════ 单独一张：顺序 —— 一收到责任就能结账 ══════════
    # ⛔ 2026-09-23 现场：「这个地方有点讲不明白，把它折起来。
    #   因为它对全局重要性也不高。」——&#160;它原来是 fig-settle 的第四格。
    #   ⭐ 面板折不起来（一张 SVG 要么整张在要么整张不在），所以先拆成独立一张。
    #   ⚠️ 没删：这一格的落点（多卡训练能把通信藏进计算里）在专题五还要用；
    #     只是它在**这一讲**里确实不承重 ——&#160;这一讲要的只是「结账不往前传」。
    PH4 = 268
    py4 = g.panel(0, y0d, W, PH4,
                  "顺序：<tspan font-weight=\"700\">一收到责任就能结账，"
                  "谁也不用等谁</tspan>", PU)

    g.line(1320, py4 + 92, 80, py4 + 92, PU, 2.6)
    g.t(1330, py4 + 66, "loss 那头", PU, True, 13, "end")
    g.t(90, py4 + 66, "输入那头", PU, True, 13)
    for k in range(6):
        x = 1200 - k * 216
        g.box(x - 54, py4 + 74, 108, 36, "#fff", OR, 5, sw=1.5)
        g.t(x, py4 + 98, "算子 %d" % (k + 1), OR, True, 12.5, "middle")
        g.line(x, py4 + 114, x, py4 + 146, OR, 1.8)
        g.t(x, py4 + 168, "结账", OR, True, 12, "middle")

    g.t(700, py4 + 208,
        "⭐⭐ 而这正是<tspan font-weight=\"700\">多卡训练能把通信藏进计算里</tspan>的原因："
        "最后几层的账早就结完了，前面几层还在算　——　",
        INK, size=14.5, anchor="middle")
    g.t(700, py4 + 234,
        "<tspan font-weight=\"700\">那段时间正好拿来把已经结完的那部分传出去。</tspan>",
        PU, True, 14.5, "middle")
    g._pan = None


    yb = g.src(y0d + PH4 + 26,
               "⭐ 这一格原本是上一张图的第四格，2026-09-23 拆成独立一张"
               "——&#160;<tspan font-weight=\"700\">为的是让课件那边能把它折叠起来</tspan>。")

    g.save("fig4-settle-order.svg", yb + 14)


main()
order()
