# -*- coding: utf-8 -*-
r"""专题四 · 一个学习率伺候不了所有参数 ——&#160;`fig-onelr` 配的动画（候选第 4）

⭐⭐⭐ 这一段刻意**不用前四段那个套路**（「两个东西同时跑，比谁快」）。
  静态图已经把 η 大和 η 小各画了一条轨迹，再让它们动一遍，
  增量只有「谁先谁后」——&#160;上一段 `vanish` 刚在这儿栽过。

⭐⭐⭐ 这一段的增量来自**第三个维度：学习率自己**。
  把 η 做成一根可以走的轴，从小连续扫到大，于是观众看到的不是两条轨迹，
  是**一条轨迹的形态在连续演变**：
      贴着谷底慢慢蹭 → 走得刚好 → 开始左右横跳 → 炸出画面
  ⛔ 静态图给不了这个 ——&#160;它只能挑两三个 η 各画一条，
    而**「中间发生了什么」「拐点在哪」正是这一节要讲的东西**。

⭐⭐ 落点跟静态图同一个，但这里是**看**出来的不是读出来的：
  **η 只要变动一成，就从收敛翻成发散。**
  ⛔ 这句话第一版我写成「能用的区间窄得吓人 ——&#160;从开始横跳到炸只有一条线宽」，
    **然后被自己的数打了脸**：从「开始横跳」到「炸」实测占标尺 **57%**，
    一点都不窄。我把两件事混成了一件。
  ⭐⭐ 判据：**叙事跟着数走，不是调参数去迁就叙事。**
    真正窄的是**门槛那一带** ——&#160;±5% 两条线在标尺上只隔 ~10%，
    而跨过去就是收敛与发散的分界。所以标尺上标的是那两条，不是横跳起点。

⭐ 还有一条静态图说不清的：**过了门槛不是「差一点」，是性质变了。**
  门槛以下每步都在缩，以上每步都在放大 ——&#160;滑块越线那一瞬间画面会翻脸。

════════════════════════════════════════════════════════════════════
⭐⭐⭐ S4（2026-09-18）：房规①松绑之后回来给这一段**加字**。

⭐ 先判它缺什么 ——&#160;**缺的是「解释」，不是「铺陈」。**
  理由一句话：**画面本身已经演完了，观众却不知道自己在看什么量。**
  扫描、翻脸、炸出画面这些动作一个都不缺、也不快（15 秒里各占一段），
  可下面那根横线**没有名字**，中间那条红线**没有身份** ——&#160;
  于是「滑块往右走」读不成「学习率在变大」，「越过红线」读不成「越过 2/A₁」。
  ⛔ 所以**时长和 `loop` 一个都不动**，只往画面里加名字、公式和分段字幕。
  判据（skill 房规）：**「不够 fancy」的解药是信息密度，不是时长。**

⭐⭐ 字幕随 η 分三段换词，而**它必须也是 η 的纯函数** ——&#160;
  这样三角波扫回来的时候字幕自动对上，首末同帧仍然是**波形自己保证的**，
  不用在结尾补任何复位。实现上：把三个相位的边界**换算成时刻**切成 5 段
  `play`（绿→蓝→红→蓝→绿），边界关于 T_UP 对称，脚本里有断言钉着。
  ⛔ 没有给 `Text` 挂每帧改 opacity 的 updater ——&#160;`descend` 那一轮实测
    -qh 会从 28 it/s 掉到 0.47 it/s。**幕与幕之间是离散事件，不是连续量。**

⭐⭐ 文字的位置**也当场验**：把所有会画出来的东西（等高线、全扫描区间的
  轨迹点、标尺与三根刻度）采样成一堆点，每一块字先声明自己的地盘，
  再断言 ① 那块地盘里一个点都没有、② 排出来的字确实装得进那块地盘。
  ⛔ 判据：**「我看草稿觉得没压住」不是证据** ——&#160;压没压住是可以算的，
    而且 η 扫描的轨迹**每一帧形状都不一样**，肉眼只能看到其中一帧。

════════════════════════════════════════════════════════════════════
⛔ 房规（见 skill `manim-teaching-figures`，2026-09-18 松绑过一次）：
  ① **文字为讲解服务，不为装饰** ——&#160;可以放 MathTex 与中文 Text；
     ⛔ 但多一条硬义务：页面的 `aria-label` 必须把画面里每一句话都复述一遍
     （读屏读不到视频里的字）。**画面上的每一句都定义在下面
     「画面上的字」那一段里**，改字先改那儿，再回去改 `topic-04.html`。
  ② 静态图排在它上面；③ 首帧 ≡ 末帧；④ 数据当场算。

⭐⭐ 复位这次不用造：**参数扫描天然首尾闭合** ——&#160;η 扫上去再扫回来
  就是一个三角波，末帧＝首帧是**波形自己保证的**，不需要 `clock()` 那套补丁。
  （`memtime` / `vanish` 那两段是累积式的，才必须另造复位段。）

📌 渲染：
    ~/.claude/skills/manim-teaching-figures/scripts/render.sh \
        tools/manim/topic04-anim-onelr.py OneLR WebPages/media/topic04-onelr.mp4

📌 实测成本（cc-tw，2026-09-18，给下次「是不是变慢了」一个可比的数）：
    加字前 `--draft` 7.3 秒；加字后 `--draft` 13.1 秒、`-qh` 26 秒 / 188 KB。
    片长 14.60 秒（加字前 14.57）——&#160;**时长没动**，图注那句「15 秒」仍然对。

📌 首末一致度 16.0% → **26.4%，这是变好了不是变糟**。判据不是感觉，是分布：
    加字后画面墨水 5172 → 28460 px（多了一堆抗锯齿的字边），
    而**每个差异像素的幅度反而更小**（中位 23→17、90 分位 65→30、最大 196→130），
    把判据阈值从 12 抬到 30，旧版 6.4%、新版 2.6%。
    ⭐ 也就是说：这个指标的分母涨了 5.5 倍，分子里新增的全是**擦着阈值**的字边。
      ⛔ 它又一次印证了 traps §1.1 那句 ——&#160;**一个标量分不开两类差异**，
      拼接图上首末两帧连字带线完全重合。
"""
import math

import numpy as np
from manim import (Scene, VGroup, Dot, Line, Text, MathTex, ValueTracker,
                   always_redraw, WHITE, DOWN, RIGHT, LEFT, linear)

RD_, BL_, GR_, GY_, GY2_, INK_ = ("#d93025", "#4285f4", "#1e8e3e",
                                  "#c3c7cb", "#80868b", "#202124")

# ── 谷的形状：跟静态图同一套口径 ──────────────────────────────────
A1, A2 = 25.0, 1.0                # A1 ＝ 陡方向，A2 ＝ 平方向
ETA_MAX = 2.0 / A1                # ⭐ 发散门槛，闭式：沿陡方向每步放大 |1 − ηA₁|
W0 = (1.5, 8.0)                   # 起点 (w1 陡, w2 平)
NSTEP = 26

# ⭐ 下端取得够小，平方向才「蹭」得出来（26 步只走掉三分之一）
ETA_LO, ETA_HI = ETA_MAX * 0.20, ETA_MAX * 1.16


def run(eta, n=NSTEP):
    out, w1, w2 = [W0], W0[0], W0[1]
    for _ in range(n):
        w1 -= eta * A1 * w1
        w2 -= eta * A2 * w2
        out.append((w1, w2))
    return out


# ⭐⭐⭐ 这段动画的三个论点，全部当场跑出来钉住 ——&#160;不是我写上去的
_lo = run(ETA_LO)
_hi = run(ETA_HI)
assert abs(_lo[-1][0]) < abs(W0[0]) * 0.5, "扫描下端必须是收敛的，不然没有对照"
assert abs(_hi[-1][0]) > abs(W0[0]) * 5, "扫描上端必须真的炸，不然「越线翻脸」是编的"
assert _lo[-1][1] > W0[1] * 0.5, \
    "⭐ 下端那条在**平方向**必须还没走到一半 ——&#160;「两头都不满意」的另一头"
assert ETA_LO < ETA_MAX < ETA_HI, "扫描区间必须跨过门槛，否则看不到翻脸那一下"

# ⭐⭐ 字幕里那句「26 步只走掉三分之一」也是一条断言 ——&#160;它是**画面上的字**，
#   而画面上的字过期起来一声不响（traps §3.3）。这里把它钉死在 ±0.04 里。
_WALKED = (W0[1] - _lo[-1][1]) / W0[1]
assert 0.30 <= _WALKED <= 0.38, \
    "字幕写的是「只走掉三分之一」，实测 %.1f%% ——&#160;不对就改字，别改断言" \
    % (_WALKED * 100)

# ⭐ 颜色分三段用的分界（**不是**「窄」的证据，见文件头那条教训）：
#   η 超过 ETA_MAX/2 ＝ 1/A₁ 时 1−ηA₁ 变负，陡方向从「单调缩」变成「左右横跳」。
_ETA_OSC = ETA_MAX * 0.5
assert abs(_ETA_OSC - 1.0 / A1) < 1e-12, "横跳起点的闭式就是 1/A₁"

# ⭐⭐ 真正窄的那一带：门槛 ±5%。一边收敛、一边发散，而它们几乎贴在一起。
ETA_OK, ETA_BAD = ETA_MAX * 0.95, ETA_MAX * 1.05
BAND = (ETA_BAD - ETA_OK) / (ETA_HI - ETA_LO)
assert BAND < 0.15, "门槛那一带要在标尺上挤成窄窄一条（现在 %.0f%%）" % (BAND * 100)
assert abs(run(ETA_OK)[-1][0]) < W0[0] and abs(run(ETA_BAD)[-1][0]) > W0[0], \
    "⭐ ±5% 必须真的一个收敛一个发散 ——&#160;「差一成就翻脸」全靠这条"

# ⭐⭐ 蓝段字幕说「还在收敛」，而**这一段里就藏着整条标尺上最好的那个 η**。
#   扫一遍 26 步之后离谷底的距离，最优点实测落在 0.963×门槛 ——&#160;
#   也就是说**最好的学习率就贴在悬崖边上**（还在 ±5% 那条细线以内）。
#   ⛔ 这句结论没进画面（字会太多），但它必须是算出来的，不是印象。
_dist = lambda e: math.hypot(*run(e)[-1])                          # noqa: E731
_ETA_BEST = min((ETA_LO + (ETA_HI - ETA_LO) * i / 2000
                 for i in range(2001)), key=_dist)
assert _ETA_OSC < _ETA_BEST < ETA_MAX, \
    "最好的 η 得落在「已经横跳但还收敛」这一段里，否则蓝段字幕站不住"
assert ETA_OK < _ETA_BEST, "最优点还贴在 ±5% 带子里面（实测 %.3f×门槛）" \
    % (_ETA_BEST / ETA_MAX)

# ── 画面 ──────────────────────────────────────────────────────────
SX, SY = 0.62, 0.62               # w2 → 横，w1 → 纵（平方向铺宽，陡方向在纵向跳）
CY = 0.75                         # 等高线区域的中心高度
BOX_X, BOX_Y = 6.0, 2.5           # 画到这儿就算飞出去了
DOT_R = 0.055
RULE_Y = -2.85                    # η 标尺的高度
RX0, RX1 = -5.0, 5.0
LEVELS = (4.0, 16.0, 36.0, 64.0)  # 等高线取的几个值

T_UP = 5.6
T_END = T_UP * 2                  # ⭐ 三角波：上去再回来，首末天然同帧
SPEED = 1.3                       # 播放放慢的倍数（⛔ 动它就改了片长，图注会过期）


def pt(w1, w2):
    return np.array([w2 * SX, CY + w1 * SY, 0])


def rule_x(eta):
    return RX0 + (RX1 - RX0) * (eta - ETA_LO) / (ETA_HI - ETA_LO)


def eta_at(t):
    """三角波：0→T_UP 扫上去，T_UP→T_END 原路扫回来。**首末天然同帧的根。**"""
    p = t / T_UP if t <= T_UP else (T_END - t) / T_UP
    return ETA_LO + (ETA_HI - ETA_LO) * max(0.0, min(1.0, p))


def phase_of(eta):
    """0 ＝ 单调缩（蹭），1 ＝ 已横跳但还收敛，2 ＝ 越过门槛、发散。

    ⭐ 轨迹颜色、滑块颜色、字幕三处共用这一个函数 ——&#160;
      **同一条分段规则只定义一处**，不然改了阈值总有一处忘了跟。
    """
    return 2 if eta > ETA_MAX else (1 if eta > _ETA_OSC else 0)


PH_COL = (GR_, BL_, RD_)

# ══════════════════════════════════════════════════════════════════
# ⭐⭐ 文字的地盘 ——&#160;先声明，再当场验
#
# 把所有会被画出来的东西采样成点集，然后对每一块字断言「这块地盘是空的」。
# ⛔ 为什么不靠看草稿：η 扫描的轨迹**每一帧形状都不一样**，
#   肉眼一次只能看到一帧，而压住与否要对**整个扫描区间**成立。
# ══════════════════════════════════════════════════════════════════
def _sample_seg(a, b, n=8):
    return [(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n)
            for i in range(n + 1)]


def _ring_pts(c):
    return [pt(math.sqrt(c / A1) * math.sin(math.radians(a)),
               math.sqrt(c / A2) * math.cos(math.radians(a)))
            for a in range(0, 361, 3)]


def _traj_pts(eta):
    out = []
    for w1, w2 in run(eta):
        if abs(w1) > BOX_Y / SY or abs(w2) > BOX_X / SX:
            break
        out.append(pt(w1, w2))
    return out


_ink = []
for _c in LEVELS:                                   # 等高线
    _r = _ring_pts(_c)
    for _u, _v in zip(_r, _r[1:]):
        _ink += _sample_seg(_u, _v, 2)
for _i in range(601):                               # 全扫描区间的轨迹
    _p = _traj_pts(ETA_LO + (ETA_HI - ETA_LO) * _i / 600)
    for _u, _v in zip(_p, _p[1:]):
        _ink += _sample_seg(_u, _v, 6)
_ink += _sample_seg((RX0 - 0.2, RULE_Y), (RX1 + 0.2, RULE_Y), 220)   # 标尺
for _e, _h in ((ETA_MAX, 0.30), (ETA_OK, 0.19), (ETA_BAD, 0.19)):    # 三根刻度
    _ink += _sample_seg((rule_x(_e), RULE_Y - _h), (rule_x(_e), RULE_Y + _h), 12)
_ink += _sample_seg((rule_x(ETA_LO), RULE_Y), (rule_x(ETA_HI), RULE_Y), 220)  # 滑块轨道
INK_PTS = np.array(_ink)

# 每一块字的地盘：(x0, x1, y0, y1)
TTL_BOX = (-6.40, 6.40, 3.40, 3.96)     # 顶部标题
LGD_BOX = (-6.95, -2.60, 1.86, 3.24)    # 左上角：更新式 ＋ 两个方向的曲率
CAP_BOX = (-6.95, 0.30, -2.42, -0.80)   # 分段字幕（左侧那块空档）
THR_BOX = (1.45, 5.25, -2.44, -1.96)    # 门槛标签（红线正上方，贴着刻度顶端）
RUL_BOX = (-5.40, -1.40, -3.66, -3.12)  # 标尺的名字（左端下方）
PM5_BOX = (0.95, 5.75, -3.97, -3.24)    # ±5% 那两条细线的说明（右下）


def _assert_empty(box, what):
    x0, x1, y0, y1 = box
    hit = int(np.count_nonzero((INK_PTS[:, 0] >= x0) & (INK_PTS[:, 0] <= x1)
                               & (INK_PTS[:, 1] >= y0 - DOT_R)
                               & (INK_PTS[:, 1] <= y1 + DOT_R)))
    assert hit == 0, "%s 的地盘压住了 %d 个画出来的点 ——&#160;挪字或挪图，别删断言" \
        % (what, hit)


for _box, _what in ((TTL_BOX, "标题"), (LGD_BOX, "左上角图例"),
                    (CAP_BOX, "分段字幕"), (THR_BOX, "门槛标签"),
                    (RUL_BOX, "标尺名字"), (PM5_BOX, "±5% 说明")):
    _assert_empty(_box, _what)


def _place(m, box, what):
    """把一块字摆进它的地盘中央，并断言**排出来真的装得下**。

    ⭐ 中文按 Pango 排版，宽度事先只能估 ——&#160;估错了要当场炸，
      不能等到渲完看图才发现字戳出去了。
    """
    x0, x1, y0, y1 = box
    m.move_to(np.array([(x0 + x1) / 2, (y0 + y1) / 2, 0]))
    assert m.width <= x1 - x0 + 1e-6 and m.height <= y1 - y0 + 1e-6, \
        "%s 装不进它的地盘：排出来 %.2f×%.2f，地盘 %.2f×%.2f" \
        % (what, m.width, m.height, x1 - x0, y1 - y0)
    return m


# ══════════════════════════════════════════════════════════════════
# 画面上的字 ——&#160;**全部在这一段里定义，改字只改这儿**
#
# ⛔ 房规①的硬义务：视频里的字选不中、读屏读不到 ——&#160;
#   所以页面 `<video>` 的 aria-label 必须把下面每一句都复述一遍。
#   动了这一段，就得回去动 `topic-04.html` 那条 aria-label。
# ══════════════════════════════════════════════════════════════════
TITLE = "同一个谷、同一个起点 —— 只把学习率 η 从小扫到大"

# 左上角：读懂门槛公式所需的全部前提（中文 ＋ 跟它并排的公式）
RULE_TEX = r"w \leftarrow w - \eta\,A\,w"
LGD_ROWS = (("陡方向（纵）", r"A_1=25"), ("平方向（横）", r"A_2=1"))

# 分段字幕：**按 η 分段，所以回扫时自动对上**（见文件头）
CAPS = (("η 太小：贴着谷底慢慢蹭", "平方向 26 步只走掉三分之一"),
        ("η 够大了：陡方向左右横跳", "可幅度一跳比一跳小，还在收敛"),
        ("η 越过门槛：每步在放大", "几步就冲出画面 —— 是性质变了"))

RULE_NAME = "学习率 η"                      # 标尺的名字（＋两端取值，下面生成）
THR_NAME = "发散门槛"                       # 红线（＋ η = 2/A₁ = 0.08）
PM5_L1 = "两侧细线 ＝ 门槛 ±5%"
PM5_L2 = "左边收敛 / 右边发散 —— 只差一成"

# ⭐ 图例里那两个数必须跟真在跑的常数是同一个来源 ——&#160;
#   写死「A₁=25」而回头改了 A1，画面会一声不响地撒谎（traps §3.3）。
assert LGD_ROWS[0][1] == r"A_1=%g" % A1 and LGD_ROWS[1][1] == r"A_2=%g" % A2, \
    "左上角图例上的曲率跟脚本里的 A1/A2 对不上了"


class OneLR(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        tt = ValueTracker(0.0)

        def eta_now():
            return eta_at(tt.get_value())

        # ── 底稿：几条真椭圆等高线 ＋ 一根 η 标尺 ────────────────
        base = VGroup()
        for c in LEVELS:
            pts = _ring_pts(c)          # A1·w1² + A2·w2² = c 的参数式
            for u, v in zip(pts, pts[1:]):
                base.add(Line(u, v, stroke_color=GY_, stroke_width=1.2))
        base.add(Dot(pt(0, 0), radius=0.06, color=GY2_))        # 谷底
        base.add(Line(np.array([RX0 - 0.2, RULE_Y, 0]),
                      np.array([RX1 + 0.2, RULE_Y, 0]),
                      stroke_color=GY2_, stroke_width=2.2))
        # ⭐⭐ 门槛那条红线是全片唯一的「阈值」视觉 ——&#160;它是算出来的（2/A₁），
        #   不是我挑的位置。滑块越过它，上面的轨迹当场翻脸。
        base.add(Line(np.array([rule_x(ETA_MAX), RULE_Y - 0.30, 0]),
                      np.array([rule_x(ETA_MAX), RULE_Y + 0.30, 0]),
                      stroke_color=RD_, stroke_width=3.0))
        # ⭐⭐ 门槛 ±5% 两条细线：它们之间只有标尺的一成宽，
        #   而滑块从左边那条走到右边那条，上面的轨迹就从收敛变成发散。
        #   **「差一成就翻脸」这句话的全部画面就是这两条线有多近。**
        for e in (ETA_OK, ETA_BAD):
            base.add(Line(np.array([rule_x(e), RULE_Y - 0.19, 0]),
                          np.array([rule_x(e), RULE_Y + 0.19, 0]),
                          stroke_color=GY2_, stroke_width=1.6))
        self.add(base)

        # ── 常驻的字：标题 ＋ 左上角图例 ＋ 标尺名字 ＋ 两处刻度说明 ──
        self.add(_place(Text(TITLE, font_size=26, color=INK_), TTL_BOX, "标题"))

        # ⭐ 左上角这三行是**读懂门槛公式所需的全部前提**：
        #   更新式给出「每步 w₁ 乘 (1 − ηA₁)」，A₁ 给出它的数值，
        #   于是红线那条 η = 2/A₁ 不是天上掉下来的，是 |1 − ηA₁| = 1 解出来的。
        def _row(zh, tex):
            return VGroup(Text(zh, font_size=19, color=GY2_),
                          MathTex(tex, color=GY2_).scale(0.52)
                          ).arrange(RIGHT, buff=0.13)

        legend = VGroup(
            MathTex(RULE_TEX, color=GY2_).scale(0.62),
            *[_row(zh, tex) for zh, tex in LGD_ROWS],
        ).arrange(DOWN, buff=0.17, aligned_edge=LEFT)
        self.add(_place(legend, LGD_BOX, "左上角图例"))

        # 标尺的名字 ＋ 它两端的实际取值 ——&#160;数字从扫描区间直接生成
        ruler_name = VGroup(
            Text(RULE_NAME, font_size=22, color=INK_),
            MathTex(r"%.3f \rightarrow %.3f" % (ETA_LO, ETA_HI),
                    color=GY2_).scale(0.52),
        ).arrange(RIGHT, buff=0.22)
        self.add(_place(ruler_name, RUL_BOX, "标尺名字"))

        # 门槛标签：**把 2/A₁ 写出来**，这是 S4 这一轮的正主
        thr = VGroup(
            Text(THR_NAME, font_size=21, color=RD_),
            MathTex(r"\eta = 2/A_1 = %.2f" % ETA_MAX, color=RD_).scale(0.58),
        ).arrange(RIGHT, buff=0.16)
        self.add(_place(thr, THR_BOX, "门槛标签"))

        pm5 = VGroup(
            Text(PM5_L1, font_size=18, color=GY2_),
            Text(PM5_L2, font_size=18, color=GY2_,
                 t2c={"左边收敛": GR_, "右边发散": RD_}),
        ).arrange(DOWN, buff=0.12)
        self.add(_place(pm5, PM5_BOX, "±5% 说明"))

        # ── 分段字幕：三块，按相位 add/remove，**不挂 updater** ──────
        caps = []
        for i, (l1, l2) in enumerate(CAPS):
            g = VGroup(Text(l1, font_size=22, color=PH_COL[i]),
                       Text(l2, font_size=19, color=GY2_)
                       ).arrange(DOWN, buff=0.15)
            caps.append(_place(g, CAP_BOX, "字幕（第 %d 段）" % (i + 1)))

        # ── 当前 η 那条轨迹 ──────────────────────────────────────
        def traj():
            eta = eta_now()
            col = PH_COL[phase_of(eta)]
            g, prev = VGroup(), None
            for q in _traj_pts(eta):           # ⭐ 飞出画面就断掉，读成「炸了」
                if prev is not None:
                    g.add(Line(prev, q, stroke_color=col, stroke_width=3.2))
                g.add(Dot(q, radius=DOT_R, color=col))
                prev = q
            return g

        self.add(always_redraw(traj))

        # ── 标尺上的滑块 ─────────────────────────────────────────
        def knob():
            eta = eta_now()
            return Dot(np.array([rule_x(eta), RULE_Y, 0]), radius=0.12,
                       color=PH_COL[phase_of(eta)])

        self.add(always_redraw(knob))

        # ── 时间轴：按**相位边界**切成 5 段 ───────────────────────
        # ⭐⭐ 字幕换词的时刻不是我挑的，是从 η 的两个阈值反解出来的；
        #   而三角波让回扫段的边界关于 T_UP 对称 ——&#160;于是字幕跟 η 一样
        #   是**时间的偶对称函数**，末帧自动回到第一段绿字。
        def t_of(eta):
            return T_UP * (eta - ETA_LO) / (ETA_HI - ETA_LO)

        _up = [t_of(_ETA_OSC), t_of(ETA_MAX)]
        MARKS = [0.0] + _up + [T_END - m for m in reversed(_up)] + [T_END]
        assert all(abs(MARKS[i] + MARKS[-1 - i] - T_END) < 1e-9
                   for i in range(len(MARKS))), \
            "⭐ 相位边界必须关于 T_UP 对称，否则回扫时字幕对不上，末帧≠首帧"
        assert all(b > a for a, b in zip(MARKS, MARKS[1:])), "相位边界得递增"
        assert abs(sum(b - a for a, b in zip(MARKS, MARKS[1:])) - T_END) < 1e-9, \
            "⛔ 切段不许改总时长 ——&#160;图注里那句「15 秒无声循环」是条 lint"

        cur = None
        for a, b in zip(MARKS, MARKS[1:]):
            ph = phase_of(eta_at((a + b) / 2))
            if ph != cur:
                if cur is not None:
                    self.remove(caps[cur])
                self.add(caps[ph])
                cur = ph
            self.play(tt.animate.set_value(b), run_time=(b - a) * SPEED,
                      rate_func=linear)
        assert cur == 0, "⭐ 末段必须回到第一段字幕 ——&#160;这就是「首末同帧」本身"
