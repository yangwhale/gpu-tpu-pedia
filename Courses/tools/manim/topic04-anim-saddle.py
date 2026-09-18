# -*- coding: utf-8 -*-
r"""专题四 · 鞍点的 3D 曲面 ——&#160;`fig-saddle` Ⓐ 配的那段 8 秒动画

⭐⭐⭐ 2026-09-18 现场：「你看看三蓝一棕的绘图工具好不好用，
  要是好用的话你就多用用。」——&#160;实测下来结论分两半：

  ⛔ **manim 不能替换我们的静态图。** 它的 `--format` 只接受
    png / gif / mp4 / webm / mov ——&#160;**没有 SVG**。
    而我们那套图是内联 SVG，靠它换来的东西不只是好看：
    文字可选可搜、屏幕阅读器能读（每张都有 aria）、投屏无损缩放、
    进 git 能 diff，**而且全部 lint 都建立在「它是 SVG」上**
    （字号、撞车、图内节号、墨水体检）。换成 PNG 这些全丢。

  ⭐⭐ **但它在另一头无可替代：它本来就是拍动画的。**
    而 3B1B 的杀手锏从来就是「**动起来**」，不是静态构图。

⭐⭐⭐ 这一段是**增量最大**的那个用例：
  `fig-saddle` Ⓐ 只能画**两个剖面**，
  而「一个方向上翘、另一个方向下沉」这件事 ——&#160;**天生是三维的**。
  二维的图讲得出结论，讲不出那个形状。

═══════════════════════════════════════════════════════════════════
⭐⭐⭐ 2026-09-18 第二轮 · 加解释性文字。动手前先判房规那一句：

  **它缺的是「解释」还是「铺陈」？——&#160;缺的是「解释」。**

  理由：画面本身**已经讲完了**。曲面转一圈，鞍的形状、两条剖面一升一降、
  中间那个平点，全都看得见 ——&#160;这些是**形状**，而形状不缺时间。
  缺的是**命名和归因**：观众看得见「两条彩线不一样」，
  但不知道那个词叫**鞍点**，也不知道两条线**为什么**颜色不同。

  ⭐⭐ 于是按房规那张表走「解释」那一行：**加字幕和公式，
    时长和 `loop` 都不动** ——&#160;仍然是 8.0 秒、仍然转满整圈、仍然首尾同帧。
  ⛔ 反过来说：这一段**不该**改成长叙事片。它没有起承转合可讲，
    拉长只会让同一圈转得更慢。**「不够 fancy」的解药是信息密度，不是时长。**

⛔ 旧 docstring 里那条「一个字都不放」已经作废 ——&#160;房规 2026-09-18 松绑，
  现在的①是「**文字为讲解服务，不为装饰**」。这一段加的七行字全部在解释画面，
  没有一行是装饰。
═══════════════════════════════════════════════════════════════════

⭐⭐ `add_fixed_in_frame_mobjects` 在本机 manim CE 0.21.0 上**实测过才敢用**
  （探针：同样的 ThreeDScene ＋ 同样的 `begin_ambient_camera_rotation`，
  抽 t=0 / 1 / 2 / 3 / 3.9 五帧比对）。三条结论：
    ① 文字**不跟相机转**：同一块文字在五帧里的像素差只有个位数，
       而且 t0-vs-t1 和 t0-vs-t3.9 的差**完全相同** ——&#160;说明那点差是
       h.264 的块噪声，不是位移。
    ② 文字**画在 3D 内容之上，不会被曲面遮住**：探针故意把一行红字
       压在曲面正中央，五帧里都清清楚楚盖在棋盘格上面。
    ③ 它**自己会把 mobject 加进 scene**，不要再 `self.add` 一遍。
  ⛔ 唯一的坑是「加了不遮挡 ≠ 不碍事」：字压在曲面上虽然看得见，但花。
    所以下面那组 SAFE_* 是**量出来的**，不是估的 ——&#160;见它们的注释。

⛔ 三条刻意的取舍：
  ① **配色跟 `fig-saddle` 对齐**：红 ＝ 往上（走不了），绿 ＝ 往下（能走出去）。
     ⭐ 两条剖面曲线是**真画在曲面上**的，不是贴上去的；曲面本身是
       `z(u, v)` 当场算出来的参数曲面。
  ② **图例用色块，不用指向箭头。** 相机转满一圈，红线一会儿在左、一会儿在右 ——
     ⭐ 判据（同 `anim-momentum` 那条）：**钉在位置上的标注，寿命不能超过
       它指的那个东西停在那儿的时间。** 这里那个时间是 0，所以只能做图例。
     文字里也直接写「沿红线走 / 沿绿线走」，跟它此刻转到哪边无关。
  ③ **整圈旋转，无缝循环** ——&#160;转满 360° 正好回到起点。
     ⭐ 加的字全是 `add_fixed_in_frame_mobjects` 的**静态**文字：
       它不随相机变，也不随时间变，所以首帧末帧上它逐像素相同 ——
       **首尾同帧这条不变量没有被削弱，反而多了一块恒定的锚。**

⛔⛔ 文字进画面之后多出一条**新义务**：视频里的字选不中、读屏读不到。
  ⭐ `topic-04.html` 那个 `<video>` 的 `aria-label` 必须把画面上出现的
    **每一句话都复述一遍**。本脚本不改 HTML（并行改动中），
    新 aria-label 的建议文本见 `ARIA_HINT`。

📌 渲染：
    ~/.claude/skills/manim-teaching-figures/scripts/render.sh \
        tools/manim/topic04-anim-saddle.py Saddle WebPages/media/topic04-saddle.mp4
"""
import inspect

import numpy as np
from manim import (ThreeDScene, Surface, Dot3D, ParametricFunction, VGroup,
                   Line, Text, MathTex, WHITE, DEGREES, UP, DOWN, LEFT, RIGHT,
                   ORIGIN, config)

# ⭐ 跟 topic03_draw 的主色对齐（那边 RD / GR 就是这两个值）
RD_ = "#d93025"          # 往上 ——&#160;走不了
GR_ = "#1e8e3e"          # 往下 ——&#160;还能走
BL_ = "#4285f4"
INK_ = "#202124"
GY_ = "#5f6368"

SPAN = 2.0               # 曲面在 u / v 上的范围
K = 0.35                 # z ＝ K(u² − v²)
TURN = 8.0               # 转满一圈用几秒（＝ 视频时长）


def z(u, v):
    return K * (u * u - v * v)


# ⭐ 这一条是整段动画的全部内容，写成 assert 钉住：
#   沿 u 走 loss **上升**，沿 v 走 loss **下降** ——&#160;这才叫鞍点。
assert z(SPAN, 0) > 0 > z(0, SPAN), "不是鞍面：两个方向得一升一降"
assert abs(z(0, 0)) < 1e-12, "鞍点本身的高度应当是 0"


# ══════════════════════════════════════════════════════════════════
# 文字往哪儿放 ——&#160;**量出来的空地**，不是估的
# ══════════════════════════════════════════════════════════════════
# ⭐⭐ 做法：先拿同参数的探针片渲一版，把 60 帧（转满一圈）逐帧按**颜色**
#   （蓝 − 红 > 12 且 蓝 > 120，只认棋盘格曲面，白底和黑红文字都不算）
#   取曲面的屏幕包围盒，再取一圈里的最大范围。实测 854×480 上：
#       x ∈ [229, 623]    y ∈ [104, 395]
#   → 四边的空白占比 L 26.8% / R 27.1% / T 21.7% / B 17.7%（与分辨率无关）
# ⛔ 第一版是按「非白像素」量的，那会把探针自己的文字一起算进曲面里，
#   得靠手动屏蔽几个矩形 ——&#160;而屏蔽框本身又可能切掉真正的曲面边缘。
#   ⭐ 判据：**要量 A 就按 A 的固有属性选，别按「不是 B」选。**
#     曲面的固有属性是「它是蓝的」，按颜色一句话就分干净了。
_FW, _FH = config.frame_width, config.frame_height
SAFE_L = -_FW / 2 + 0.2681 * _FW      # 文字右缘不得越过这条线
SAFE_R = _FW / 2 - 0.2705 * _FW       # 文字左缘不得越过这条线
SAFE_T = _FH / 2 - 0.2167 * _FH       # 文字下缘不得低于这条线
SAFE_B = -_FH / 2 + 0.1771 * _FH      # 文字上缘不得高于这条线
assert SAFE_L < SAFE_R and SAFE_B < SAFE_T, "空地算反了"


def _clear(m, name):
    """这块文字整个落在曲面**一圈都扫不到**的四条边之一里。"""
    l, r = m.get_left()[0], m.get_right()[0]
    b, t = m.get_bottom()[1], m.get_top()[1]
    assert (r <= SAFE_L or l >= SAFE_R or b >= SAFE_T or t <= SAFE_B), (
        "「%s」会跟转动中的曲面撞上（x %.2f→%.2f, y %.2f→%.2f；"
        "空地 L≤%.2f R≥%.2f T≥%.2f B≤%.2f）"
        % (name, l, r, b, t, SAFE_L, SAFE_R, SAFE_T, SAFE_B))
    return m


def _apart(items):
    """⛔ 这一条是补上来的：第一版只查了「文字 vs 曲面」，于是右下角那条方程
    **跟底下那句话叠在了一起** ——&#160;两块都在「底边空地」里，各自都合规，
    合起来就糊成一团。
    ⭐ 判据：**「每个都在空地里」推不出「它们互不相撞」。**
      约束是成对的，就得成对地查。"""
    for i, (ma, na) in enumerate(items):
        for mb, nb in items[i + 1:]:
            gap_x = max(ma.get_left()[0] - mb.get_right()[0],
                        mb.get_left()[0] - ma.get_right()[0])
            gap_y = max(ma.get_bottom()[1] - mb.get_top()[1],
                        mb.get_bottom()[1] - ma.get_top()[1])
            assert max(gap_x, gap_y) >= 0.12, (
                "「%s」和「%s」挨得太近或叠上了（横向间隙 %.2f，纵向 %.2f）"
                % (na, nb, gap_x, gap_y))


# ⛔ 画面上出现的每一句话，`<video aria-label>` 都得复述 ——&#160;抄这段过去。
ARIA_HINT = (
    "一个鞍面在缓慢自转，画面上打着四组说明文字。顶上一行大字写着"
    "「同一个点，一个方向是谷，另一个方向是峰」，下一行小字写着"
    "「这样的点叫鞍点 —— 翻山的那个垭口」。左边是红色图例："
    "「沿红线走：往上 ↑」「这条路是上坡，loss 变大」。右边是绿色图例："
    "「沿绿线走：往下 ↓」「顺着它接着走，loss 变小」。底下一行写着"
    "「黑点：这里梯度为零 —— 可它既不是谷底，也不是山顶」，"
    "左下角标着曲面的方程 z(u,v)=%.2f(u²−v²)。" % K +
    "画面中央那张蓝色棋盘格曲面上，红色曲线沿一个方向往上翘，"
    "绿色曲线沿另一个方向往下沉，两条在中间那个黑点处交叉。"
)


class Saddle(ThreeDScene):
    def construct(self):
        self.camera.background_color = WHITE

        surf = Surface(
            lambda u, v: np.array([u, v, z(u, v)]),
            u_range=[-SPAN, SPAN], v_range=[-SPAN, SPAN],
            resolution=(28, 28), fill_opacity=0.62,
            checkerboard_colors=[BL_, "#a8c7fa"],
            stroke_width=0.35, stroke_color="#5f6368",
        )

        # ⭐ 两条**真画在曲面上**的剖面：红的往上翘，绿的往下沉
        up = ParametricFunction(
            lambda t: np.array([t, 0.0, z(t, 0.0) + 0.02]),
            t_range=[-SPAN, SPAN], color=RD_, stroke_width=7)
        down = ParametricFunction(
            lambda t: np.array([0.0, t, z(0.0, t) + 0.02]),
            t_range=[-SPAN, SPAN], color=GR_, stroke_width=7)

        # ⛔ 原来 radius=0.085、z 跟两条剖面一样高 ——&#160;它**被两条线压在底下**，
        #   草稿上放大看才找得到一小粒黑。以前没人在意，因为画面上没提过它；
        #   ⭐ 现在底下那行字点名说「黑点」，**它就必须一眼看得见**。
        #   判据：**加了一句指认某个元素的话，就等于给那个元素追加了可见性要求。**
        #   抬高一点 z，让它压在两条剖面之上（Cairo 的 3D 是按深度排序画的）。
        ball = Dot3D(np.array([0.0, 0.0, 0.12]), color=INK_, radius=0.15)

        # ── 解释性文字（全部固定在画面上，不随相机也不随时间变）────────
        # ⭐⭐ 这一句是整段的论点，所以给它最大的字号，并且让「谷」「峰」
        #   **直接染成两条剖面的颜色** ——&#160;句子和画面的绑定不靠观众自己连线。
        title = Text("同一个点，一个方向是谷，另一个方向是峰",
                     font_size=34, color=INK_,
                     t2c={"谷": GR_, "峰": RD_}).to_edge(UP, buff=0.28)
        # ⭐ 命名单独占一行：上一行讲的是**现象**，这一行才给它**名字**。
        #   「垭口」是本讲自己的主比喻（见 fig-saddle 的 Ⓐ），口径对齐。
        name = Text("这样的点叫鞍点 —— 翻山的那个垭口",
                    font_size=24, color=GY_).next_to(title, DOWN, buff=0.16)

        def legend(col, head, sub, edge):
            """一条剖面的图例：色块 ＋ 一句结论 ＋ 一句为什么。"""
            swatch = Line(ORIGIN, RIGHT * 0.52, color=col, stroke_width=7)
            h = Text(head, font_size=24, color=col)
            s = Text(sub, font_size=19, color=GY_)
            row = VGroup(swatch, h).arrange(RIGHT, buff=0.16)
            g = VGroup(row, s).arrange(DOWN, buff=0.13, aligned_edge=LEFT)
            return g.to_edge(edge, buff=0.34).shift(UP * 0.55)

        lg_up = legend(RD_, "沿红线走：往上 ↑", "这条路是上坡，loss 变大", LEFT)
        lg_dn = legend(GR_, "沿绿线走：往下 ↓", "顺着它接着走，loss 变小", RIGHT)

        # ⭐ 黑点是画面上唯一没被颜色解释过的东西 ——&#160;它需要自己那一行。
        #   用「谷底 / 山顶」而不是「极小 / 极大」，跟 fig-saddle Ⓐ 的三分法同词。
        dot_note = Text("黑点：这里梯度为零 —— 可它既不是谷底，也不是山顶",
                        font_size=25, color=INK_).to_edge(DOWN, buff=0.80)
        # ⭐ 方程里的系数**从 K 插值进去**，不是手打的 ——&#160;改了 K 它跟着变，
        #   不会像注释里的数字那样悄悄过期（traps.md §3.3）。
        # ⛔ 放右下角会跟上面那句话的尾巴叠上（`_apart` 就是那次加的）。
        #   挪到左下角**还不够** ——&#160;那句话是居中的、六个单位宽，
        #   左下角跟它仍然有横向重叠。⭐ 最后靠的是**纵向**分开：
        #   底边空地有 1.4 个单位高，够码两行 ——&#160;方程当脚注压在最底下一行。
        eq = MathTex(r"z(u,v)=%.2f\,(u^{2}-v^{2})" % K,
                     color=GY_).scale(0.66).to_corner(DOWN + LEFT, buff=0.22)

        blocks = ((title, "标题"), (name, "命名"), (lg_up, "红图例"),
                  (lg_dn, "绿图例"), (dot_note, "黑点说明"), (eq, "方程"))
        for m, n in blocks:
            _clear(m, n)
        _apart(blocks)

        self.set_camera_orientation(phi=66 * DEGREES, theta=-55 * DEGREES, zoom=1.15)
        self.add(surf, VGroup(up, down), ball)
        # ⛔ 这一句自己会把它们加进 scene，别再 self.add 一遍（实测确认）。
        self.add_fixed_in_frame_mobjects(title, name, lg_up, lg_dn, dot_note, eq)
        # ⭐ 转满一整圈 → 首尾同帧 → 页面里 loop 起来看不出接缝
        self.begin_ambient_camera_rotation(rate=2 * np.pi / TURN)
        self.wait(TURN)


# ⭐⭐ 首尾同帧在这一段靠的是一条很强的性质：**除了相机，画面上没有任何
#   东西随时间变**。哪天有人往里加一个 tracker，这条就得重新论证 ——
#   所以把它写成断言，钉在 construct 的源码上（traps.md §1.3 的同款做法）。
_BODY = inspect.getsource(Saddle.construct)
assert "ValueTracker" not in _BODY and "always_redraw" not in _BODY, (
    "这一段的「首尾同帧」是因为只有相机在动。加了随时间变的东西，"
    "就得自己重新保证末帧回到首帧，别指望转满一圈还管用")
