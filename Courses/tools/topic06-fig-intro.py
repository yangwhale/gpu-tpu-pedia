# -*- coding: utf-8 -*-
r"""专题六 · 第零节「一次请求走一遍」的两张静态图。

fig6-life.svg —— 回答：一个请求从进门到结束，时间花在哪两段，用户感到的 TTFT、TPOT 落在哪？
  · 数字：没有实测数。这是示意时间线，格子长度只表达「prefill 一大块、decode 一步一小格」。
  · 按定义必然成立的部分是真算的：KV 的格数 ＝ prompt 长度 ＋ 已出的字数；
    第 k 步 decode 要读的 KV 格数就是那一刻的 KV 长度（脚本里 assert）。
  · 刻意没画：排队里的调度细节、batch 里别的请求（第四节的事）、prefill 内部的分块（chunked prefill）。
    也没画时间刻度：画上刻度就是编。

fig6-map.svg —— 回答：后面十二节各自在主线的哪一格？
  · 节名取自 topic06-build.py 的 SECTIONS（脚本里 import 同一张表并 assert 一一对上）。
  · 归类是本课的分法：按这一节主要贡献归一格；有的节两头都沾（例如 DEP 既让专家分到更多行，
    也让 attention 各管各的请求），只放它最主要的那一格。
  · 刻意没画：节与节之间的依赖箭头（会把路线图画成一团线）。
"""

import os

from topic03_draw import Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE

W = 1400
HERE = os.path.dirname(os.path.abspath(__file__))

# ── 示意时间线的几何（不是实测）──────────────────────────────────────
N_PROMPT = 6          # prompt 画成 6 格 KV（示意）
N_STEPS = 8           # decode 画 8 步（示意）
KV_AFTER = [N_PROMPT + k for k in range(1, N_STEPS + 1)]   # 第 k 步 decode 结束后 KV 的格数
assert KV_AFTER[0] == N_PROMPT + 1 and KV_AFTER[-1] == N_PROMPT + N_STEPS
# 第 k 步 decode 读的 KV ＝ 进这一步之前已有的格数（prompt ＋ 前面出的字，含 prefill 出的第 1 个字）
KV_READ = [N_PROMPT + k for k in range(N_STEPS)]
assert KV_READ[0] == N_PROMPT and all(b - a == 1 for a, b in zip(KV_READ, KV_READ[1:]))


def fig_life():
    f = Fig(W, "一个请求的一生，示意时间线。请求进门后可能先排队；然后 prefill 一次把整段 prompt 算完，"
               "同时把 prompt 每个 token 的 KV 写进显存，算完吐出第一个字。从进门到第一个字的时间叫 TTFT。"
               "之后进入 decode：每一步只出一个字，每一步都要把全部权重和到目前为止的全部 KV 从显存读一遍，"
               "KV 每一步长一格。相邻两个字之间的时间叫 TPOT。出到结束符为止。"
               "prefill 一次读权重分给整段 prompt，吃的是算力；decode 一次读权重只出一个字，吃的是带宽")
    y0 = f.header("一个请求的一生　——　<tspan font-weight=\"700\">一大块 prefill，接一长串一步一字的 decode</tspan>",
                  "示意时间线，格子长短不按比例。上：在做什么、何时出字；中：显存里的 KV 有多长；下：decode 每一步要从显存读什么",
                  [(OR, "prefill：整段 prompt 一次算完"), (GR, "decode：一步出一个字"),
                   (BL, "权重"), (PU, "KV cache")])
    PH = 550
    py = f.panel(0, y0, W, PH, "从进门到结束", INK)
    LX = 24                                # 左侧行标签
    T0, QW, PW_ = 170, 70, 290             # 进门时刻、排队宽、prefill 宽
    SW, BW = 100, 84                       # decode 每步的间距、块宽
    XP = T0 + QW                           # prefill 起点
    XD = XP + PW_                          # 第一个字（prefill 结束）
    xs = [XD + 10 + i * SW for i in range(N_STEPS)]
    assert xs[-1] + BW < W - 40

    # ① 在做什么
    RY, RH = py + 70, 54
    f.t(LX, RY + 26, "在做什么", INK, True, 15)
    f.t(LX, RY + 46, "何时出字", GY, size=13)
    f.box(T0, RY, QW - 8, RH, "none", GY2, 4, dash="5,4")
    f.t(T0 + (QW - 8) / 2, RY + 33, "排队", GY, True, 14, "middle")
    f.box(XP, RY, PW_ - 8, RH, OR, OR, 4)
    f.t(XP + (PW_ - 8) / 2, RY + 24, "prefill", "#ffffff", True, 16, "middle")
    f.t(XP + (PW_ - 8) / 2, RY + 44, "整段 prompt 一次算完", "#ffffff", False, 13, "middle")
    for i, x in enumerate(xs):
        f.box(x, RY, BW, RH, GR, GR, 4)
        f.t(x + BW / 2, RY + 33, "第 %d 步" % (i + 1), "#ffffff", True, 13.5, "middle")
    # 出字：prefill 出第 1 个字，之后每步一个，最后一个是结束符
    toks = [XD - 8] + [x + BW for x in xs]
    for i, tx in enumerate(toks):
        lab = "结束" if i == len(toks) - 1 else "字 %d" % (i + 1)
        f.p.append('<circle cx="%.1f" cy="%.1f" r="6" fill="%s"/>' % (tx, RY - 14, RD if i == 0 else INK))
        f.t(tx, RY - 26, lab, RD if i == 0 else INK, i == 0, 13, "middle")
    # 进门
    f.line(T0, RY - 40, T0, RY - 4, GY2, 1.6)
    f.t(T0, RY - 46, "请求进门", GY, True, 13, "middle")
    # TTFT、TPOT 尺子
    BY = RY + RH + 26
    f.line(T0, BY, XD - 8, BY, RD, 2, arrow=False)
    f.line(T0, BY - 7, T0, BY + 7, RD, 2, arrow=False)
    f.line(XD - 8, BY - 7, XD - 8, BY + 7, RD, 2, arrow=False)
    f.t((T0 + XD - 8) / 2, BY + 24, "TTFT：进门到第一个字（含排队）", RD, True, 14, "middle")
    a, b = toks[3], toks[4]
    f.line(a, BY, b, BY, BL, 2, arrow=False)
    f.line(a, BY - 7, a, BY + 7, BL, 2, arrow=False)
    f.line(b, BY - 7, b, BY + 7, BL, 2, arrow=False)
    f.t((a + b) / 2, BY + 24, "TPOT：相邻两个字的间隔", BL, True, 14, "middle")

    # ② KV 有多长：阶梯
    KY0 = py + 330                         # 阶梯底线
    CU = 13                                # 每格 KV 的高度（示意）
    f.t(LX, KY0 - 70, "显存里的", INK, True, 15)
    f.t(LX, KY0 - 48, "KV 有多长", INK, True, 15)
    f.line(T0, KY0, W - 40, KY0, LINE, 1, arrow=False)
    pts = [(XP, KY0), (XD - 8, KY0 - N_PROMPT * CU)]
    h = N_PROMPT
    for i, x in enumerate(xs):
        pts.append((x + BW, KY0 - h * CU))
        h = KV_AFTER[i]
        pts.append((x + BW, KY0 - h * CU))
    pts.append((xs[-1] + BW + 30, KY0 - h * CU))
    f.poly(pts + [(xs[-1] + BW + 30, KY0)], "#f3e8fd")
    f.path(pts, PU, 2.4, arrow=False)
    f.t(XP + 4, KY0 - N_PROMPT * CU - 14, "prefill 一次写满整段 prompt 的 KV", PU, True, 13.5)
    f.t(xs[4], KY0 - 22, "decode 每出一个字，KV 长一格", PU, True, 13.5)

    # ③ decode 每步读什么：权重（固定）＋ KV（越来越长）
    RB = py + 520                          # 条形底线
    WH = 56                                # 权重条高度（示意，每步一样）
    f.t(LX, RB - 100, "每一步", INK, True, 15)
    f.t(LX, RB - 78, "从显存读什么", INK, True, 15)
    for i, x in enumerate(xs):
        kv_h = KV_READ[i] * 7
        f.box(x + 22, RB - WH, BW - 44, WH, BL, BL, 2)
        f.box(x + 22, RB - WH - kv_h - 2, BW - 44, kv_h, PU, PU, 2)
    f.t(xs[0] - 14, RB - 20, "权重：每步整份", BL, True, 13, "end")
    f.t(xs[0] - 14, RB - WH - 18, "KV：每步全部", PU, True, 13, "end")
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "两段的账正好相反", [
        "prefill 读一遍权重，分给整段 prompt 的几千个 token 一起算：<tspan font-weight=\"700\">卡在算力上</tspan>。",
        "decode 读一遍权重和全部 KV，只出一个字：<tspan font-weight=\"700\">卡在显存带宽上</tspan>，而且字越往后 KV 越长、这一趟越重。",
    ])
    yb = f.src(yb + 10,
               "⚠️ 示意时间线：格子长短不按比例，prompt 画成 6 格 KV、decode 画 8 步，都是为了看得清。",
               "📌 TTFT、TPOT 的定义按 vLLM 的 benchmark_serving：TTFT 从请求发出算到第一个输出 token，"
               "TPOT 是除第一个 token 外平均每个输出 token 的时间。第二节会细讲它们的口径（含不含排队、平均还是 p99）。",
               "⚠️ 「第一个字由 prefill 产出」是标准实现：prefill 的最后一个位置直接给出第一个输出 token 的分布。")
    f.save("fig6-life.svg", yb + 14)


# ── 路线图：三类归位 ──────────────────────────────────────────────
_src = open(os.path.join(HERE, "topic06-build.py"), encoding="utf-8").read()
_secblock = _src[_src.index("SECTIONS = ["):_src.index("]\n", _src.index("SECTIONS = [")) + 1]
_ns = {}
exec(_secblock, _ns)
SEC = {num: title for _, num, title in _ns["SECTIONS"]}

BASE = [("一", "吞 prompt 卡算力，吐字卡带宽"), ("二", "指标和口径对齐了再比快慢")]
GROUPS = [
    ("让这一趟分给更多 token", BL, [
        ("四", "一趟分给一整批请求"),
        ("五", "单卡上把这一趟喂饱"),
        ("六", "专家收齐所有卡的 token"),
        ("十一", "一趟出好几个字"),
    ]),
    ("少搬", GR, [
        ("三", "算过的前缀不再算、不重复存"),
        ("七", "KV 不再每张卡各存一份"),
        ("十二", "每个数少几位，中间结果不落显存"),
    ]),
    ("把活拆开，放对地方", OR, [
        ("八", "吞 prompt 和吐字分两批机器"),
        ("九", "按负载配比，再往细里拆"),
        ("十", "同样的卡，摆法决定快慢"),
        ("十三", "在 TPU 上跑成服务"),
    ]),
]
_placed = [n for n, _ in BASE] + [n for _, _, items in GROUPS for n, _ in items]
_want = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "十一", "十二", "十三"]
assert sorted(_placed, key=_want.index) == _want, "第一到十三节必须各归一格、不重不漏"
assert all(n in SEC for n in _want)


def _short(title):
    """节名里冒号后面的副题太长时，只留冒号前面那一截。"""
    return title.split("：")[0]


def fig_map():
    f = Fig(W, "这一课的路线图。第一、二节是地基：吞 prompt 卡算力、吐字卡带宽，指标和口径对齐了再比快慢。"
               "后面十一节分三类。让 decode 那一趟分给更多 token：第四节批处理与调度、第五节单卡里的 decode 注意力与 MoE、"
               "第六节 TEP 和 DEP、第十一节投机解码。少搬：第三节 KV cache、第七节切序列、第十二节量化与 kernel 融合。"
               "把活拆开放对地方：第八节 PD 分离、第九节配比与 AFD、第十节先选拓扑、第十三节在 TPU 上落地")
    y0 = f.header("后面每一节都在这一趟上做文章　——　<tspan font-weight=\"700\">分摊它、削减它，或者换个地方搬</tspan>",
                  "每格是一节：上面一行节名，下面一行它对这一趟做了什么。有的节两头都沾，只放进它最主要的那一格",
                  [(INK, "地基"), (BL, "分给更多 token"), (GR, "少搬"), (OR, "拆开放对地方")])
    # 地基条
    BH = 74
    py = f.panel(0, y0, W, BH + 46, "地基：先认清这一趟", INK)
    for i, (n, hook) in enumerate(BASE):
        x = 24 + i * 691
        f.box(x, py + 16, 661, BH - 10, "none", GY2, 8, sw=1.4)
        f.t(x + 18, py + 44, "第%s节　%s" % (n, _short(SEC[n])), INK, True, 15)
        f.t(x + 18, py + 68, hook, GY, size=13.5)
    # 三栏
    CW, GAP = 453, 20
    top = py + BH + 16 + 20
    CH = 4 * 92 + 120
    for gi, (gname, col, items) in enumerate(GROUPS):
        x = gi * (CW + GAP)
        cy = f.panel(x, top, CW, CH, gname, col)
        # 小图标：一趟（方块）→ 怎么处理
        ix, iy = x + 24, cy + 22
        if gi == 0:          # 一个箱子，箭头分给四个点
            f.box(ix, iy, 40, 30, col, col, 4)
            for k in range(4):
                f.line(ix + 44, iy + 15, ix + 96, iy - 6 + k * 14, col, 1.4, arrow=False)
                f.p.append('<circle cx="%d" cy="%d" r="5" fill="%s"/>' % (ix + 102, iy - 6 + k * 14, col))
            f.t(ix + 124, iy + 21, "搬一趟，更多 token 分", GY, size=13.5)
        elif gi == 1:        # 大箱子 → 小箱子
            f.box(ix, iy - 4, 46, 38, "none", col, 4, sw=1.6)
            f.line(ix + 54, iy + 15, ix + 84, iy + 15, col, 1.6)
            f.box(ix + 92, iy + 6, 22, 18, col, col, 3)
            f.t(ix + 130, iy + 21, "每趟的字节变少", GY, size=13.5)
        else:                # 两台机器各干各的
            f.box(ix, iy, 36, 30, OR, OR, 4)
            f.box(ix + 56, iy, 36, 30, GR, GR, 4)
            f.line(ix + 38, iy + 15, ix + 52, iy + 15, GY2, 1.4)
            f.t(ix + 110, iy + 21, "两种活各放各的机器", GY, size=13.5)
        for k, (n, hook) in enumerate(items):
            yy = cy + 76 + k * 92
            f.box(x + 16, yy, CW - 32, 78, "none", col, 8, sw=1.4)
            f.box(x + 16, yy, 5, 78, col, col, 2)
            f.t(x + 34, yy + 30, "第%s节　%s" % (n, _short(SEC[n])), INK, True, 15, w=CW - 60)
            f.t(x + 34, yy + 56, hook, col, False, 13.5, w=CW - 60)
    f._pan = None
    yb = f.band(top + CH + 20, "ok", "三类招数，对着同一趟搬运", [
        "前两类直接改这一趟：要么让同一趟服务更多 token，要么让每一趟的字节变少。",
        "第三类不改这一趟本身，而是让吞 prompt 和吐字各自在合适的机器上跑，互不拖累。",
    ])
    yb = f.src(yb + 10,
               "⚠️ 本课的分法：每节按它主要的贡献归一格。DEP 也让 attention 各管各的请求，KV 复用也省了 prefill 的计算，这类兼顾的地方只放一格。",
               "📌 节名取自 topic06-build.py 的 SECTIONS，脚本里断言第一到十三节各归一格、不重不漏。")
    f.save("fig6-map.svg", yb + 14)


fig_life()
fig_map()
