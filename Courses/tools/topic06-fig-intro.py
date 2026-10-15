# -*- coding: utf-8 -*-
r"""专题六 · 第零节「一次请求走一遍」的两张静态图。

fig6-life.svg —— 回答：一个请求从进门到结束，时间花在哪两段，用户感到的 TTFT、TPOT 落在哪？
  · 数字：没有实测数。这是示意时间线，格子长度只表达「prefill 一大块、decode 一步一小格」。
  · 按定义必然成立的部分是真算的：KV 的格数 ＝ prompt 长度 ＋ 已出的字数；
    第 k 步 decode 要读的 KV 格数就是那一刻的 KV 长度（脚本里 assert）。
  · 刻意没画：排队里的调度细节、batch 里别的请求（第四节的事）、prefill 内部的分块（chunked prefill）。
    也没画时间刻度：画上刻度就是编。

fig6-map.svg —— 回答：后面十二节各自在主线的哪一处？
  · 画成「搬一趟」场景：仓库（显存）里放着权重和 KV，车把它们拉到工位（算力），decode 每出一个字搬一趟。
    三根杠杆：车上多坐人（一趟分给更多 token，蓝）、货变轻（每趟字节变少，绿）、开两个仓库（吞 prompt 与吐字分开放，橙）。
    节号做成小标签贴在杠杆上；第一节贴在路上（这一趟本身），第二节贴在尺子上（量这一趟）。
  · 节名取自 topic06-build.py 的 SECTIONS（脚本里 import 同一张表并 assert 一一对上）。
  · 归类是本课的分法：按这一节主要贡献归一处；有的节两头都沾（例如 DEP 既让专家分到更多行，
    也让 attention 各管各的请求），只放它最主要的那一处。投机解码按第十一节自己的说法「搬一趟，出好几个字」归蓝。
  · scene() 是 map 和 fig6-eight（第十四节）共用的底图，eight 用 importlib 引进来，往同一个场景上钉图钉。
  · 刻意没画：节与节之间的依赖箭头；每节的说明文字（节名在正文目录里）。
"""

import os

from topic03_draw import Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE, wpx

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




# ── 路线图：「搬一趟」场景 ＋ 三根杠杆 ──────────────────────────────
_src = open(os.path.join(HERE, "topic06-build.py"), encoding="utf-8").read()
_secblock = _src[_src.index("SECTIONS = ["):_src.index("]\n", _src.index("SECTIONS = [")) + 1]
_ns = {}
exec(_secblock, _ns)
SEC = {num: title for _, num, title in _ns["SECTIONS"]}

# 杠杆：(名字, 颜色, 节号, 作用在场景的哪个部件)
LEVERS = [
    ("开两个仓库", OR, ["八", "九", "十", "十三"], "wh2"),
    ("货变轻", GR, ["三", "七", "十二"], "cargo"),
    ("车上多坐人", BL, ["四", "五", "六", "十一"], "seats"),
]
BASE = [("一", "road"), ("二", "ruler")]           # 地基：这一趟本身、量这一趟的尺子
_placed = [n for n, _ in BASE] + [n for _, _, ns, _ in LEVERS for n in ns]
_want = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "十一", "十二", "十三"]
assert sorted(_placed, key=_want.index) == _want, "第一到十三节必须各归一格、不重不漏"
assert all(n in SEC for n in _want)


def _house(f, x, y_gnd, w, h, roof, col, dash=None, sw=2.0):
    """一栋房子：墙 ＋ 不填实的屋顶。(x, y_gnd) 是左下角。"""
    f.box(x, y_gnd - h, w, h, "none", col, 3, sw=sw, dash=dash)
    f.p.append('<path d="M %.1f %.1f L %.1f %.1f L %.1f %.1f" fill="none" stroke="%s" '
               'stroke-width="%s" stroke-linejoin="round"%s/>'
               % (x - 10, y_gnd - h, x + w / 2.0, y_gnd - h - roof, x + w + 10, y_gnd - h, col, sw,
                  ' stroke-dasharray="%s"' % dash if dash else ''))


def scene(f, top):
    """「搬一趟」底图：两个仓库 → 一辆车拉货 → 工位。map 和 eight 共用。
    返回各部件的锚点（杠杆标签 / 图钉往这儿指）和场景下沿。"""
    GND = top + 230                         # 地面线
    A = {}
    # 第二个仓库（橙，虚线）：专门吞 prompt
    _house(f, 34, GND, 150, 100, 36, OR, dash="7,5")
    f.t(109, GND - 60, "吞 prompt", OR, True, 15, "middle")
    for k in range(3):
        f.box(70 + k * 26, GND - 40, 22, 22, PU, PU, 3)
    A["wh2"] = (109, GND - 140)
    f.line(192, GND - 52, 238, GND - 52, PU, 2)
    f.t(215, GND - 62, "KV", PU, True, 14, "middle")
    # 主仓库 ＝ 显存
    MX, MW = 250, 230
    _house(f, MX, GND, MW, 160, 50, INK)
    f.t(MX + MW / 2, GND - 175, "仓库 ＝ 显存", INK, True, 16, "middle")
    f.box(MX + 18, GND - 100, 112, 76, BL, BL, 4)
    f.t(MX + 74, GND - 56, "权重", "#ffffff", True, 16, "middle")
    for k in range(3):
        f.box(MX + 146, GND - 42 - k * 20, 66, 17, PU, PU, 3)
    f.t(MX + 179, GND - 69, "KV", "#ffffff", True, 13, "middle")
    A["kv"] = (MX + 179, GND - 84)
    A["wh"] = (MX + MW / 2, GND - 210)
    # 路
    RX0, RX1 = MX + MW + 14, 1100
    f.box(RX0, GND - 6, RX1 - RX0, 12, "#e8eaed", "#e8eaed", 2)
    f.line(RX0 + 10, GND, RX1 - 10, GND, "#ffffff", 2, dash="14,10", arrow=False)
    A["road"] = ((RX0 + RX1) / 2, GND + 6)
    # 车：左边货斗，右边带四个窗的长驾驶室（座位）
    TX = 560
    BY0, BY1 = GND - 104, GND - 26          # 货斗上下沿
    f.box(TX, BY0, 210, BY1 - BY0, "none", GY, 4, sw=2)
    f.box(TX + 10, BY0 + 10, 108, BY1 - BY0 - 18, BL, BL, 3)
    f.t(TX + 64, BY0 + 46, "权重", "#ffffff", True, 15, "middle")
    for k in range(3):
        f.box(TX + 128, BY1 - 26 - k * 22, 72, 18, PU, PU, 3)
    A["cargo"] = (TX + 105, BY0)
    CX0, CX1 = TX + 218, TX + 400
    f.p.append('<path d="M %.1f %.1f V %.1f H %.1f Q %.1f %.1f %.1f %.1f V %.1f Z" fill="none" '
               'stroke="%s" stroke-width="2"/>' % (CX0, BY1, BY0 - 18, CX1 - 26, CX1, BY0 - 18, CX1, BY0 + 20, BY1, GY))
    for k in range(4):                      # 四个窗，每个窗里坐一个人
        wx = CX0 + 12 + k * 40
        f.box(wx, BY0 - 6, 32, 38, "none", GY2, 4)
        f.p.append('<circle cx="%.1f" cy="%.1f" r="7" fill="%s"/>' % (wx + 16, BY0 + 8, BL))
        f.p.append('<path d="M %.1f %.1f a 11 9 0 0 1 22 0 z" fill="%s"/>' % (wx + 5, BY0 + 30, BL))
    A["seats"] = ((CX0 + CX1) / 2, BY0 - 18)
    A["cab"] = (CX1 - 22, BY0 - 14)
    for wxc in (TX + 50, TX + 170, CX1 - 50):
        f.p.append('<circle cx="%.1f" cy="%.1f" r="14" fill="#ffffff" stroke="%s" stroke-width="2.4"/>'
                   % (wxc, GND - 18, INK))
    for k in range(3):                      # 速度线
        f.line(TX - 70 + k * 10, BY0 + 18 + k * 22, TX - 16, BY0 + 18 + k * 22, GY2, 1.6, arrow=False)
    f.line(CX1 + 12, GND - 60, RX1 - 6, GND - 60, GY2, 2)
    # 工位 ＝ 算力
    WX, WW = 1120, 250
    f.box(WX, GND - 96, WW, 14, INK, INK, 2)
    f.line(WX + 20, GND - 82, WX + 20, GND, INK, 2.4, arrow=False)
    f.line(WX + WW - 20, GND - 82, WX + WW - 20, GND, INK, 2.4, arrow=False)
    for r in range(3):
        for c in range(6):
            f.box(WX + 44 + c * 28, GND - 176 + r * 26, 22, 22, "none", INK, 3, sw=1.6)
    f.t(WX + WW / 2, GND - 190, "工位 ＝ 算力", INK, True, 16, "middle")
    A["desk"] = (WX + WW / 2, GND - 212)
    # 尺子：量这一趟要多久
    RY = GND + 46
    f.line(RX0, RY, RX1, RY, RD, 2, arrow=False)
    f.line(RX0, RY - 8, RX0, RY + 8, RD, 2, arrow=False)
    f.line(RX1, RY - 8, RX1, RY + 8, RD, 2, arrow=False)
    for k in range(1, 12):
        xx = RX0 + (RX1 - RX0) * k / 12.0
        f.line(xx, RY - 4, xx, RY, RD, 1.6, arrow=False)
    A["ruler"] = (RX0, RY)
    A["ruler_mid"] = ((RX0 + RX1) / 2, RY)
    return A, GND, RY + 46


def _chips(f, x, y, nums, col):
    """节号小标签。返回右沿。"""
    for n in nums:
        lab = "第%s节" % n
        w = wpx(lab, 14) + 16
        f.box(x, y, w, 26, "#ffffff", col, 13, sw=1.6)
        f.t(x + w / 2, y + 18, lab, col, True, 14, "middle")
        x += w + 6
    return x - 6


def _chips_w(nums):
    return sum(wpx("第%s节" % n, 14) + 16 for n in nums) + 6 * (len(nums) - 1)


def fig_map():
    f = Fig(W, "这一课的路线图，画成一个搬货场景。仓库是显存，里面放着权重和 KV；工位是算力。"
               "decode 每出一个字，就要用一辆车把权重和 KV 从仓库拉到工位一趟，这是第一节；第二节是量这一趟的尺子。"
               "后面的招数是三根杠杆。车上多坐人：一趟分给更多请求，第四、五、六、十一节。"
               "货变轻：每一趟的字节变少，第三、七、十二节。开两个仓库：吞 prompt 和吐字分开放，第八、九、十、十三节")
    y0 = f.header("后面每一节都在这一趟上做文章　——　<tspan font-weight=\"700\">三根杠杆</tspan>",
                  "decode 每出一个字，就要把权重和 KV 从仓库拉到工位一趟",
                  [(BL, "车上多坐人"), (GR, "货变轻"), (OR, "开两个仓库")])
    TOP = 120
    PH = 30 + TOP + 230 + 46 + 46 + 18
    py = f.panel(0, y0, W, PH, "搬一趟", INK)
    A, GND, bot = scene(f, py + TOP)
    assert bot + 18 == py - 30 + PH          # panel() 返回的是标题栏下沿，比面板顶低 30
    # 三根杠杆：标签 ＋ 节号，一根引线指到它动的那个部件
    LX = {"wh2": 24, "cargo": 470, "seats": 830}
    for name, col, nums, key in LEVERS:
        x = LX[key]
        ly = py + 22
        f.t(x, ly + 22, name, col, True, 19, big=True)
        cw = _chips_w(nums)
        _chips(f, x, ly + 36, nums, col)
        tx, ty = A[key]
        sx = min(max(tx, x + 20), x + cw - 20)
        f.line(sx, ly + 66, tx, ty - 4, col, 2)
    # 地基：路（第一节）和尺子（第二节）
    rx, ry = A["road"]
    f.t(rx + 60, GND + 30, "每出一个字，搬一趟", INK, True, 15, "middle")
    _chips(f, rx + 160, GND + 12, ["一"], INK)
    mx, my = A["ruler_mid"]
    f.t(mx + 60, my + 34, "量这一趟", RD, True, 15, "middle")
    _chips(f, mx + 120, my + 16, ["二"], RD)
    f._pan = None
    yb = f.src(py + PH + 4,
               "⚠️ 本课的分法：每节按它主要的贡献归一根杠杆。DEP 也让 attention 各管各的请求，KV 复用也省了 prefill 的计算，这类兼顾的地方只放一处。",
               "⚠️ 投机解码放在「车上多坐人」：它让搬一趟换来好几个字，和批处理一样是把一趟分给更多 token。",
               "📌 节名取自 topic06-build.py 的 SECTIONS，脚本里断言第一到十三节各归一处、不重不漏。")
    f.save("fig6-map.svg", yb + 6)


if __name__ == "__main__":
    fig_life()
    fig_map()
