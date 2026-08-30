# -*- coding: utf-8 -*-
"""图 P-7 —— 推理天生是动态的，而两边对「不静态」的处置完全不同。

这一节最容易讲砸的方式，是讲成「TPU 要静态形状，GPU 不用」。
**那是错的**，而且错得很有迷惑性 —— 因为它听起来像常识。

我们自己两份 runlog 摆在一起就能戳破它：

  GPU 侧（`gpu/inference/a4x-max/deepseek-v3/` 那份 GB300 runlog）
      decode 一样在分桶：capture 52 个 batch size，bs 1 → 512。
  TPU 侧（`tpu/vllm-torchtpu/README.md`）
      token 数补齐到预设桶，指数分桶。

**两边都在分桶。**

⚠️ 这张图第一版还多推了一步，而那一步是错的 —— 我写的是
「GPU 桶没命中可以退回去照跑，**TPU 没有这个选项**」。
审计当场用**我们自己的仓库**把它推翻了：`tpu-inference` 那一整条路线
默认就带着 `--enforce-eager` 在跑，也就是**关掉全图编译、照跑**，
和 GPU 侧关掉 CUDA graph 是同一个开关。

**「听起来像常识的架构断言」又一次栽了** —— 而且和第 4 节那次
（拿一个走错路的对照项去证明链路够用）是同一类错。

改对之后论点反而更锋利：**两边都有退路，差的是退路的标价。**

  GPU：prefill 的 graph 关了就**永久关着**，没人回头调它 —— 便宜到可以当默认配置。
  TPU：关掉全图编译的标价，我们自己文档里写着**「预期开了能再快 2-3x」**，
       而且被列进了待办项 —— **那是技术债，不是默认配置。**

所以静态在 TPU 上**仍然是一条约束**，只是这条约束的形状不是「不许动」，
而是**「你要么预付编译，要么长期付吞吐」**。

⚠️ 图里那两个「先付多少」**故意不放在一起做减法**。模型、硬件、引擎、**规模**
四样都不同 —— 最后那样最容易漏：同一份 GB300 runlog 里写着，那个 2–3 分钟
在更大的拓扑上会涨到十分钟以上。**能比的是取舍的形状，不是快多少这个数量。**
这条限制写进图里，不写进讲稿里 —— 因为讲稿会漏，图不会。
"""
from common import Fig, para, BL, RD, GN, YL, PU, INK, SUB, GREY, FILL

W = 1400
TOP = 84

SRC_Y = TOP + 40
CARD_H = 170
SRC_H = 44 + CARD_H + 16

TB_Y = SRC_Y + SRC_H + 42
HDR_H = 40
ROW_H = 96
LBL_X, LBL_W = 20, 168
GX, GW = 200, 580
TX, TW = 796, 584
TB_H = HDR_H + 2 * ROW_H

PREP_Y = TB_Y + TB_H + 28
PREP_H = 158

BAND_Y = PREP_Y + PREP_H + 26
H = BAND_Y + 112

# ── 动态性的四个来源 ────────────────────────────────────────────
# 排序不是随便的：从「用户给的」到「数据决定的」，一层比一层更晚才知道。
# 讲的时候顺着念下来，听众会自己得出「这些没有一个能在编译期知道」。
SRC = [
    ("① prompt 有多长，是用户给的", GREY, "len",
     "同一个服务，这一秒来 200 个 token，下一秒来一万两千个。"
     "<b>prefill 这一路的形状，每个请求都不一样。</b>"),
    ("② batch 的成员每一步都在换", GREY, "batch",
     "连续批处理：谁生成完了就退出，队列里的新请求立刻补上。"
     "<b>batch 这一维每一步都在动。</b>"),
    ("③ KV cache 一格一格地长", GREY, "kv",
     "每生成一个 token，它就长一格。"
     "<b>这个形状是时间的函数</b> —— 而时间只有跑起来才存在。"),
    ("④ 哪个专家收几个 token，数据说了算", GREY, "moe",
     "同一批 token，这一次和下一次分给各个专家的分法完全不同。"
     "<b>要算到路由那一层才知道。</b>"),
]

# ── 两行对照 ────────────────────────────────────────────────────
# 左右两栏必须**逐行对齐同一个问题**，不能各写各的强项 ——
# 各写各的会变成「两份特性清单」，听众只能比长短；
# 同一个问题两种答案，听众才看得见那是**同一个分岔口的两种走法**。
ROWS = [
    ("形状变了，<br>第一反应是什么<br><g>（这一行是平台性质）</g>",
     "<r>能不静态，就不静态。</r>prompt 每次长度不同，CUDA graph 复用不了 —— "
     "SGLang <b>直接把 prefill 的 graph 关掉</b>，照跑。<br>"
     "<g>日志里就是那一行 <code>Disable prefill CUDA graph</code>。</g>",
     "<b>同样的开关，这边也有</b>：<code>--enforce-eager</code> 关掉全图编译，照跑。<br>"
     "<g>我们那一整批 TPU 推理验证，<b>默认就是开着它跑出来的</b>。</g>"),
    ("关掉之后，<br>代价是什么<br><g>（这一行是各自的取舍）</g>",
     "<b>关了就永久关着，没人回头调它。</b>"
     "少掉的只是「消 CPU 启动开销」那点收益 —— "
     "<b>便宜到可以直接写进默认配置。</b>",
     "标价写在我们自己文档里：关着全图编译，<b>「预期开了能再快 2-3x」</b>。<br>"
     "而它被列进了<b>待办项</b> —— <b>这是技术债，不是默认配置。</b>"),
]

# 「准备一次要多久」原本是这张表的第三行。抽出来单独成带，是因为
# **表格版式本身在邀请横向对比**，而这两个数恰恰不能横着读
# （模型／硬件／引擎／规模四样都不同）。与其在表下再加一句警告，
# 不如把「并排」这个视觉暗示直接拆掉 —— 版式和警告打架时，听众信版式。
PREP = [
    (RD, "GPU 侧",
     "capture <b>52 个 batch size，从 1 到 512</b>，graph capture ＋ warmup "
     "<b>约 2–3 分钟</b>。<br><g>这是 1P1D 8 卡那一档。同一份 runlog 写着，"
     "换到更大的拓扑会涨到十分钟以上 —— </g><b>规模是第四个变量。</b>"),
    (GN, "TPU 侧",
     "token 数走<b>指数分桶</b>，每个桶各编一张图。<b>编译到 server ready："
     "约 7 ／ 约 15 ／ 19 分钟</b>（0.6B ／ 35B ／ 397B）。<br>"
     "<g>行名是「到 server ready」，不是「纯编译」。</g>"
     "<b>不按参数量线性外推</b> —— 397B 只比 35B 多 4 分钟。"),
]



def build():
    f = Fig(W, H, "推理动态性的四个来源，以及 GPU 与 TPU 两侧对形状变化的不同处置："
                  "两边都做分桶，区别在于桶未命中时一边退回原路执行、一边触发重新编译")
    f.title("推理天生是动态的　—— <tspan font-weight=\"700\" fill=\"#d93025\">两边都在分桶</tspan>，"
            "也都有关掉它的开关，区别在<tspan font-weight=\"700\" fill=\"#d93025\">退路的标价</tspan>")
    # legend 里只留真正的颜色键。原来第三项「灰＝小图」在图上**没有对应色块** ——
    # 那是脚注冒充色卡，而这一节的主要毛病恰恰就是颜色语义不干净。
    f.legend([(RD, "GPU 侧　·　SGLang on GB300"), (GN, "TPU 侧　·　vLLM-TorchTPU on v7")])

    _sources(f)
    _table(f)
    _prep(f)
    _band(f)
    return f.out()


# ══════════════════════════════════════════════════════════════════════
def _sources(f):
    f.rect(20, SRC_Y, 1360, SRC_H, "#fff", "#dadce0", 1.2, 10)
    f.t(38, SRC_Y + 27,
        "动态性从哪来　—— 四个来源，<tspan font-weight=\"700\" fill=\"#d93025\">"
        "没有一个能在编译期知道</tspan>", "sec")

    cw = (1360 - 36 - 3 * 16) / 4
    for i, (title, c, kind, body) in enumerate(SRC):
        x = 38 + i * (cw + 16)
        y = SRC_Y + 44
        f.rect(x, y, cw, CARD_H, FILL[c], c, 1.2, 8)
        para(f, x + 14, y + 22, cw - 28, "<b>%s</b>" % title, "lbl", 16)
        para(f, x + 14, y + 60, cw - 28, body, "xs", 17)
        _sketch(f, kind, x + 14, y + 112, cw - 28, c)


# ══════════════════════════════════════════════════════════════════════
def _sketch(f, kind, x, y, w, c):
    """每张小图只画一件事：**这个量在变**。

    刻意不标数值 —— 标了数值听众会去读数，而这里要他们读的是「不齐」。
    """
    if kind == "len":                                  # 三条不等长的横条
        for i, frac in enumerate((0.72, 0.30, 0.95)):
            f.rect(x, y + i * 14, w * frac, 9, c, None, 0, 4)

    elif kind == "batch":                              # 两步之间成分换了
        # 三种状态：实心＝还在跑，虚线幽灵＝这一步退出了，空心＝新补进来的。
        # 「谁退出」和「谁进来」是连续批处理的两半，原来只画了后者。
        # 原来这四格还用了 BL/GN/RD/YL 四色 —— 而图顶 legend 刚把红绿指派给
        # 「GPU 侧／TPU 侧」，同一块画布上给同两个颜色安了两种含义。
        cell, gap = 20, 5
        for gi, states in enumerate((("f", "f", "f", "f"), ("f", "f", "x", "n"))):
            gx = x + gi * (4 * cell + 3 * gap + 34)
            for j, st in enumerate(states):
                cx = gx + j * (cell + gap)
                if st == "f":
                    f.rect(cx, y, cell, cell, GREY, None, 0, 4)
                elif st == "x":
                    f.rect(cx, y, cell, cell, "#fff", GREY, 1.2, 4, dash="3 3")
                else:
                    f.rect(cx, y, cell, cell, "#fff", INK, 1.6, 4)
            f.t(gx, y + cell + 13,
                "第 t 步" if gi == 0 else "第 t＋1 步：虚线退出，空心补上", "xxs")
        ax = x + 4 * cell + 3 * gap + 8
        f.line(ax, y + cell / 2, ax + 20, y + cell / 2, SUB, 1.4, "aK")

    elif kind == "kv":                                 # 阶梯，只增不减
        bw = (w - 30) / 6
        for i in range(6):
            hh = 8 + i * 5
            f.rect(x + i * (bw + 6), y + 34 - hh, bw, hh, c, None, 0, 3)

    elif kind == "moe":                                # 高低不齐的柱子
        hs = (11, 32, 6, 24, 17, 3, 29, 13)
        bw = (w - 7 * 5) / 8
        for i, hh in enumerate(hs):
            f.rect(x + i * (bw + 5), y + 34 - hh, bw, hh, c, None, 0, 3)


# ══════════════════════════════════════════════════════════════════════
def _table(f):
    f.rect(20, TB_Y, 1360, HDR_H, "#f1f3f4", None, 0, 8)
    for x, h, cc in [(LBL_X, "同一个问题", INK),
                     (GX, "GPU 侧　·　SGLang on GB300", RD),
                     (TX, "TPU 侧　·　vLLM-TorchTPU on v7", GN)]:
        f.t(x + 14, TB_Y + 25, h, "lbl", cc)

    for r, (lbl, gtxt, ttxt) in enumerate(ROWS):
        y = TB_Y + HDR_H + r * ROW_H
        if r % 2:
            f.rect(20, y, 1360, ROW_H, "#fafafa", None, 0, 0)
        f.line(20, y, 1380, y, "#e8eaed", 1)

        yy = y + 30
        for seg in lbl.split("<br>"):
            yy = para(f, LBL_X + 14, yy, LBL_W - 24, "<b>%s</b>" % seg, "lbl", 17)

        for x, w, c, txt in [(GX, GW, RD, gtxt), (TX, TW, GN, ttxt)]:
            f.rect(x, y + 14, 3, ROW_H - 28, c, None, 0, 2)
            yy = y + 32
            for seg in txt.split("<br>"):
                yy = para(f, x + 16, yy, w - 32, seg, "xs", 17) + 3

    f.rect(20, TB_Y, 1360, TB_H, "none", "#dadce0", 1.2, 8)


# ══════════════════════════════════════════════════════════════════════
def _prep(f):
    """两侧各自的准备成本 —— 有意**不做成表格**。

    表格有对齐的分栏线，那条线本身就是在说「这两个数可以横着读」。
    这里两块之间只留空白，标题也直接把话说死。
    """
    f.rect(20, PREP_Y, 1360, PREP_H, "#fff", "#dadce0", 1.2, 10)
    f.t(38, PREP_Y + 26,
        "那不关呢：<tspan font-weight=\"700\" fill=\"#d93025\">两侧各自的准备成本</tspan>"
        "　—— 各看各的，<tspan font-weight=\"700\" fill=\"#d93025\">不要并排读</tspan>", "sec")
    para(f, 38, PREP_Y + 46, 1324,
         "<g>两块之间没有分栏线，是故意的。</g>"
         "<b>能比的是「取舍的形状」，不是「快多少」这个数量。</b>", "xs")

    bw = 640
    for i, (c, who, body) in enumerate(PREP):
        x = 38 + i * (bw + 44)
        f.rect(x, PREP_Y + 62, bw, 82, FILL[c], c, 1.2, 8)
        f.t(x + 14, PREP_Y + 82, who, "lbl", c)
        yy = PREP_Y + 82
        # `para()` 不认 `<br>` —— 那是 `_table()` 自己 split 出来的约定，
        # 忘了这一条就会把标签原样画进 SVG（第一版这里就是这么漏出来的）。
        for seg in body.split("<br>"):
            yy = para(f, x + 82, yy, bw - 98, seg, "xs", 17) + 2


# ══════════════════════════════════════════════════════════════════════
def _band(f):
    """落点带只留「新东西」。

    原来这里有三段，前两段分别是**图标题的复述**和 **legend ＋ 第一行的复述** ——
    同一句话在一屏之内出现三遍。删掉之后反而看得见真正的落点。
    """
    f.rect(20, BAND_Y, 1360, 112, FILL[BL], BL, 1.4, 10)
    f.t(38, BAND_Y + 28, "落点 —— 以及一笔这里没算的账", "sec")
    y = BAND_Y + 54
    for seg in [
            "<b>「TPU 要静态形状，GPU 不用」—— 前半句对，后半句不对。</b>"
            "两边都在分桶，两边也都有关掉它的开关。<r>真正的区别是退路的标价：</r>"
            "GPU 那条便宜到可以永久走着；TPU 那条标价<b>吞吐 2–3 倍</b>，"
            "所以它在待办列表里，不在默认配置里。"
            "<b>约束没消失，它换了个形状 —— 要么预付编译，要么长期付吞吐。</b>",
            "<r>⚠️ 还有一笔这里没算：</r>「补齐到桶」意味着<b>补上去的那部分 token 是白算的</b>。"
            "<b>这一笔我们没量，先记在账上</b> —— 留白会被读成「不存在」，明写「没量」不会。"]:
        y = para(f, 38, y, 1324, seg, "xs", 19) + 4



if __name__ == "__main__":
    import sys
    open(sys.argv[1] if len(sys.argv) > 1 else "out/fig_p7_dynshape.svg",
         "w", encoding="utf-8").write(build())
