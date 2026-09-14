# -*- coding: utf-8 -*-
r"""专题三 · §5.4c「MLA 好在哪里 —— 功劳到底该记给谁」

⭐⭐⭐ 2026-09-14 夜间 R46 新画。这张图**反转本讲自己的一处定性**。

═══ 起因：课程把那条 64 维窄轨讲成「妥协」═══
§5.3（fig3-knob1）和 §5.4b 讲的是：RoPE 带下标 → 吸收卡住 → 只好把带位置的
那一小块单独拎出来走 64 维一路。R44 的收尾原话是「被赶到了一条窄轨上」。
⛔ **这个定性可能是反的。** 苏剑林 2026 年那组受控消融的原话是：
  「MLA 的设计中，RoPE 和 NoPE 拼接这部分**看似无奈的设计，
    极有可能是它效果优异的关键原因**！」

═══ 数据（全部一手核过 kexue.fm/archives/10907，不是转述）═══
公共设置：类 LLAMA3 Dense，hidden 2048 / 12 层 / 16 头，优化器 Muon，
训练长度 4096，**总 16B tokens / 16k 步**。所有实验只改 Attention，
所以**参数量不严格对齐**（这一点见面板③，苏剑林自己补了对齐实验）。

  方案             Params   Loss    Cache
  MHA              931 M    2.721   4096
  MLA              894 M    2.721    576
  MLA-256          989 M    2.705    576
  GQA2-128         842 M    2.750    512
  GQA1-256         943 M    2.720    512
  GQA1-256-PR      943 M    2.711    512

⭐ Cache 那一列的单位是**每 token 每层的 KV 维度数**，不是字节。自己验一遍：
     MHA       = 16 heads × 128 × (K+V 两份) = 4096  ✅
     GQA2-128  =  2 groups × 128 × 2         =  512  ✅
     GQA1-256  =  1 group  × 256 × 2         =  512  ✅
     MLA       = 512 (压缩 KV) + 64 (解耦 RoPE)  =  576  ✅
   ⛔ 四个都对得上，才敢把它们画在同一根轴上。口径对不上就不是同一根轴。

═══ 本图自己推出来的那一句（推导链，别当成原文）═══
512 那一列是一组**每步只改一件事**的受控实验：
  GQA2-128 → GQA1-256      只改 head_dims（128→256）   降 0.030
  GQA1-256 → GQA1-256-PR   只改 Partial RoPE           降 0.009
  两级合计 **0.039**；而 MLA 比这一列的起点只好 2.750 − 2.721 = **0.029**。
⭐⭐ 所以：**head_dims ＋ Partial RoPE 这两样，在 GQA 上已经走过了 MLA**
    （2.711 < 2.721），而这两样**一样都不是低秩**。
⛔ 这句话是我从表里减出来的，苏剑林原文没有这么说。他的原文结论是
   「1、增大 head_dims 收益最大；2、Partial RoPE 对 Loss 也有一定帮助；
     3、KV-Shared 应该也有一定作用」——&nbsp;方向一致，但这个减法是本课的。

═══ ⛔ 这张图**不**主张的三件事（防止读反）═══
1. **不主张「MLA 不行」。** 同一张表里 MLA-256 是 2.705，仍然最好。
   苏剑林自己的落点是「GQA2-(192+64)-S2 比不上 MLA-256」。
2. **不主张低秩没用。** 它主张的是**功劳的排序** —— 低秩解决的是
   「存多少」，head_dims 和 Partial RoPE 解决的是「同样存这么多，学得多好」。
   §5.4a 那笔 56.9× 的账一个字都不用改。
3. **不能外推到 671B。** 这是 ~900M dense、16B tokens 的小规模消融。
   DeepSeek-V3 是 671B MoE、十几 T tokens。⭐ 结论的**方向**值得信，
   **幅度**不要搬。

═══ 为什么值得单开一张图 ═══
本讲整个 §五全是「**存多少**」的算账（56.9×、576 vs 4096、一张卡换 64 个人），
**质量那一维一次实测都没有**。而学生一定会问「压这么狠，模型不会变笨吗」。
这张图是那个问题唯一的实测回答，而且回答得反直觉。
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from topic03_draw import (Fig, wpx, BL, OR, GR, RD, GY, GY2, PU, CY, INK,
                          LINE, LINE2, BG2)

W = 1400

# ── 数据（上面已逐条核过出处）────────────────────────────────────
MLA_LOSS, MLA_CACHE = 2.721, 576
MHA_LOSS, MHA_CACHE = 2.721, 4096
# 512 那一列的三级台阶：(名字, 副标, loss, 这一步改了什么)
STEPS = [
    ("GQA2-128", "2 组 × 128 维", 2.750, None),
    ("GQA1-256", "1 组 × 256 维", 2.720, "head_dims 128 → 256"),
    ("GQA1-256-PR", "256 拆成 192 ＋ 64", 2.711, "只给那 64 维加 RoPE"),
]
# ⛔ 这三个 cache 必须全等于 512 —— 整格的前提是「代价没变，只换形状」。
#   写成断言而不是注释，是因为将来只要有人改了其中一档的配置，这一格的
#   论点当场就不成立了，应该炸掉而不是继续画。
CACHE512 = 512
assert 2 * 128 * 2 == CACHE512 and 1 * 256 * 2 == CACHE512
assert 16 * 128 * 2 == MHA_CACHE and 512 + 64 == MLA_CACHE

D_HEAD = STEPS[0][2] - STEPS[1][2]        # head_dims 那一步
D_PR = STEPS[1][2] - STEPS[2][2]          # Partial RoPE 那一步
D_SUM = STEPS[0][2] - STEPS[2][2]         # 两级合计
D_MLA = STEPS[0][2] - MLA_LOSS            # MLA 比这一列起点好多少
assert abs(D_HEAD - 0.030) < 1e-9, D_HEAD
assert abs(D_PR - 0.009) < 1e-9, D_PR
assert abs(D_SUM - 0.039) < 1e-9, D_SUM
assert abs(D_MLA - 0.029) < 1e-9, D_MLA
# ⭐ 这一条才是整张图的题眼：终点低于 MLA，也就是「不用低秩也走到了」。
assert STEPS[2][2] < MLA_LOSS, (STEPS[2][2], MLA_LOSS)
RATIO = MHA_CACHE / MLA_CACHE
assert 7.0 < RATIO < 7.2, RATIO       # 7.111…，苏剑林原文写「7倍」

PH1, PH2, PH3 = 596, 248, 258
f = Fig(W, "MLA 好在哪里：同等 KV Cache 下的受控消融 —— 功劳主要记给 head_dims "
           "和 Partial RoPE，不是低秩本身")

top = f.header(
    "§5.4c　那条「无奈」的窄轨，可能才是 MLA 好的原因",
    "同一组受控实验，每一步只改一件事 ——　而 KV Cache 全程钉死不动")

# ══════════════════════════════════════════════════════════════
# ① 楼梯：同样 512，每步只改一件事
# ══════════════════════════════════════════════════════════════
py1 = f.panel(0, top, W, PH1,
              "① 把 KV Cache 钉死在 512，只改注意力的形状 ——　两级台阶",
              BL, sub="纵轴是训练 loss，越低越好。三档的 KV Cache 一模一样，"
                      "所以台阶量的是「同样的代价换回多少」",
              tag="~900M dense · 16B tokens · seq 4096")

# 坐标系。⛔ 纵轴范围是**写死的**，不是从数据自动算的 —— 自动算会让
#   「2.720 和 2.721 几乎重合」这件事被拉开，而那个重合正是「追平」的意思。
LO, HI = 2.700, 2.760
PX, PY, PW, PHT = 150, py1 + 34, 880, 360


def sy(v):
    return PY + (HI - v) / (HI - LO) * PHT


# 背景横线 + 刻度。⚠️ R46 版面体检抓到：顶格刻度「2.760」跟轴标题的
#   「越低越好」同高同列撞在一起。⭐ 顶格本来就没有数据（最高的点是 2.750），
#   所以**横线留着、数字不标** —— 留白就是轴标题的位置，不用挪任何东西。
for v in (2.700, 2.710, 2.720, 2.730, 2.740, 2.750, 2.760):
    f.line(PX, sy(v), PX + PW, sy(v), "#eceff1", 1, arrow=False)
    if v < HI:
        f.t(PX - 12, sy(v) + 5, "%.3f" % v, GY2, False, 13, "end")
f.t(PX - 100, PY - 18, "训练 loss", INK, True, 15)
f.t(PX - 100, PY + 2, "越低越好", GY2, False, 13)

# ⭐ MLA 的水平参考线：它是这一格的「水位」，三级台阶是在跟它比高低。
yM = sy(MLA_LOSS)
f.line(PX, yM, PX + PW + 96, yM, PU, 2.0, dash="7 5", arrow=False)
f.t(PX + PW + 104, yM - 6, "MLA", PU, True, 17)
f.t(PX + PW + 104, yM + 13, "%.3f" % MLA_LOSS, PU, True, 15)
f.t(PX + PW + 104, yM + 31, "cache 576", GY2, False, 13)

# ⚠️⚠️ R46 实测撞车，改法记在这里免得下次又踩：台面标签原来画在台面**上方**
#   （y−34 / y−15），而台面上方正是**落差段所在的区域** —— 第三级落差只有
#   0.009，在这个尺度上 54 px，根本塞不下「注解三行 ＋ 下一级标题两行」。
#   渲染出来 GQA1-256-PR 那一团整个糊住了。
# ⭐ 结构性改法，不是挪几个像素：**上方永远归落差注解，标签一律下移。**
#   这样「往下走一级」和「走到哪儿了」在视觉上就是上下两层，不会再互相侵占。
SW = PW / 3.0                       # 每级台面的宽度
for i, (nm, sub, ls, what) in enumerate(STEPS):
    x0 = PX + i * SW
    y = sy(ls)
    col = GR if ls < MLA_LOSS else BL   # 掉到 MLA 线以下就转绿
    f.line(x0 + 10, y, x0 + SW - 10, y, col, 5.0, arrow=False)
    # ── 上层：竖直落差段 ＋「这一步改了什么」
    if i:
        yp = sy(STEPS[i - 1][2])
        f.line(x0 + 10, yp, x0 + 10, y, OR, 2.4, arrow=True)
        d = STEPS[i - 1][2] - ls
        f.t(x0 + 24, yp + 26, "−%.3f" % d, OR, True, 19)
        f.t(x0 + 24, yp + 48, what, OR, False, 15)
        # ⭐ 「没改的那些」只在第一次落差时写全 —— 受控这件事说一遍就够，
        #   第二级落差只有 54 px，再塞第三行必然溢出到下一级的标签区。
        if i == 1:
            f.t(x0 + 24, yp + 69, "（cache、层数、训练 tokens 全没动）",
                GY2, False, 13)
    # ── 下层：这一级是谁、落在哪
    f.t(x0 + SW / 2, y + 26, "%s　%s" % (nm, sub), col, True, 17, "middle")
    f.t(x0 + SW / 2, y + 54, "%.3f" % ls, col, True, 22, "middle")

# ⭐ 终点那一级已经在 MLA 线下方 —— 用一个竖直的双箭头把差值量出来，
#   而不是只写一句「超过了」。差 0.010，在这个尺度上是 50 px，量得出来。
#   ⚠️ R46 实测：括号原来放在 xE+150、文字横写在中点，正好跟右侧 MLA 标签的
#     「cache 576」那一行同高同列。改成夹在台面右端(1020)与 MLA 标签(1134)之间，
#     文字压到 yE 下方 —— 那一带只有台面自己的 2.711，而它在 x 轴上离得很远。
XG = 1055
yE = sy(STEPS[2][2])
_gap = MLA_LOSS - STEPS[2][2]
assert abs(_gap - 0.010) < 1e-9, _gap
f.line(XG, yM, XG, yE, GR, 1.6, arrow=False)
for yy in (yM, yE):
    f.line(XG - 7, yy, XG + 7, yy, GR, 1.6, arrow=False)
_GT = "低于 MLA %.3f" % _gap
# ⚠️ 第二次撞：放 yE+22 居中时，右端吃进了第三级台面标签的尾巴（"…192 ＋ 64"）。
#   ⭐ 现在贴着括号**往左**写，纵向卡在「MLA 虚线」与「落差注解」之间那条空档里。
#     三个邻居的坐标都在下面这组断言里钉死了，谁挪谁炸。
_GX, _GY = XG - 12, yM + 18
assert _GY > yM and _GY + 16 < sy(STEPS[1][2]) + 48, _GY      # 不碰虚线、不碰注解
assert _GX - wpx(_GT, 15) > PX + 2 * SW + 24 + wpx(STEPS[2][3], 15), "撞落差注解"
f.t(_GX, _GY, _GT, GR, True, 15, "end")

# 控制变量牌。⚠️ R46 实测：原来放在左上角 x=28，正好压住 y 轴刻度
#   （刻度 anchor=end 落在 x=138）和「越低越好」那行。
#   ⭐ 挪到图表**右上角** —— 那一块是真空的：MLA 的标签在 y 轴中段偏下，
#     第一级台面在左端，右上角没有任何东西经过。
f.box(1156, PY - 6, 212, 84, "#e8f0fe", BL, 6, 1.4)
f.t(1262, PY + 18, "KV Cache", BL, True, 15, "middle")
f.t(1262, PY + 48, "512", BL, True, 30, "middle", mono=True)
f.t(1262, PY + 68, "三档一模一样", GY2, False, 13, "middle")

# 底部落点。⛔ 三行都过宽度断言 —— R46 第一版把整段写成一行，实测 2174 px，
#   超出图宽 794 px（会被静默切在半句话上）。
BOT1 = top + PH1
LAND = [
    ("⭐⭐ 两级台阶合计 <tspan font-weight=\"700\">−%.3f</tspan>，"
     "而 MLA 比这一列的起点只好 <tspan font-weight=\"700\">−%.3f</tspan>。"
     % (D_SUM, D_MLA), INK, 17),
    ("也就是说：<tspan font-weight=\"700\">把 head_dims 放宽、再拆出一小段给 RoPE，"
     "在普通 GQA 上就已经走过了 MLA</tspan>", INK, 17),
    ("——&#160;<tspan font-weight=\"700\">而这两样，一样都不是低秩。</tspan>", GR, 17),
    ("⛔ 这个减法是本课做的，不是原文的话 ——&#160;原文的结论是"
     "「增大 head_dims 收益最大，Partial RoPE 也有一定帮助」。", GY, 15),
]
# ⭐ 落点行的 y **从图表底边算起**，不从面板底边倒着减 —— 图表一改高
#   （这一轮 PHT 就从 300 改成了 360），倒减法会让文字压进图里，而且不报错。
#   面板高度反过来由内容决定，写成断言。
LY0 = PY + PHT + 44
for k, (s, c, z) in enumerate(LAND):
    assert 28 + wpx(s, z) < W - 20, (k, 28 + wpx(s, z))
    f.t(28, LY0 + k * 26, s, c, False, z)
assert LY0 + (len(LAND) - 1) * 26 + 16 < BOT1, (LY0, BOT1)

# ══════════════════════════════════════════════════════════════
# ② MHA 那一头：7.1 倍的 cache，换回同一个 loss
# ══════════════════════════════════════════════════════════════
y2 = top + PH1 + 16
py2 = f.panel(0, y2, W, PH2,
              "② 反过来看另一头：把 cache 放大 7 倍，一分没赚",
              OR, sub="同一张表里的 MHA ——　两根条按真实维度数等比画")

# 每 1 个维度 = 多少 px。⛔ 不要给短的那根加最小宽度 —— 它短得刺眼才对。
PXC = 930.0 / MHA_CACHE
BX = 230
# ⚠️ R46 实测：loss 标签原来跟在**各自条尾**，于是 MLA 的落在 x≈390、
#   MHA 的落在 x≈1284，那条「把两个数串起来」的竖线只挨得着右边那个 ——
#   ⛔ 装置整个失效了，而图看上去还挺正常。
# ⭐ 判据：**凡是要读者「比两个数」的，那两个数必须对齐到同一条线上**，
#   否则比的是位置不是数值。所以 loss 一律钉在 XL，跟条尾脱钩。
XL = BX + MHA_CACHE * PXC + 30
assert XL + wpx("loss 2.721", 18) < W - 16, XL + wpx("loss 2.721", 18)
for j, (nm, ca, ls, col, tint) in enumerate([
        ("MLA", MLA_CACHE, MLA_LOSS, PU, "#f3e8fd"),
        ("MHA", MHA_CACHE, MHA_LOSS, RD, "#fce8e6")]):
    yb = py2 + 26 + j * 62
    f.t(BX - 16, yb + 28, nm, col, True, 19, "end", mono=True)
    f.box(BX, yb, ca * PXC, 42, tint, col, 5, 1.6)
    f.t(BX + 12, yb + 27, "%d" % ca, col, True, 18, mono=True)
    # ⭐ loss 钉在 XL 之后，短的那根（MLA）离自己的标签有 800 px 的空白 ——
    #   补一条极淡的引导点线，否则读者不确定那个数字是谁的。
    f.line(BX + ca * PXC + 10, yb + 21, XL - 18, yb + 21,
           "#e0e0e0", 1.0, dash="2 4", arrow=False)
    f.t(XL, yb + 27, "loss %.3f" % ls, col, True, 18)

f.line(XL - 12, py2 + 26, XL - 12, py2 + 26 + 62 + 42, GY2, 1.4, arrow=False)
f.t(28, py2 + 26 + 128,
    "⭐ 两个 loss <tspan font-weight=\"700\">一模一样</tspan>（都是 %.3f）"
    "，而上面那根的 KV Cache 是下面的 <tspan font-weight=\"700\">%.1f 倍</tspan>"
    " ——&#160;<tspan font-weight=\"700\">多存的那 %d 个维度，一分没换回来。</tspan>"
    % (MLA_LOSS, RATIO, MHA_CACHE - MLA_CACHE), INK, False, 17)
f.t(28, py2 + 26 + 152,
    "📌 这跟 DeepSeek-V2 论文里「MLA 甚至优于 MHA」是同一件事的两种说法。"
    "原文对这个现象的猜测是：被比的那个 MHA，head_dims 只有 128。",
    GY, False, 15)

# ══════════════════════════════════════════════════════════════
# ③ 会不会只是参数更多？—— 对齐了再比一次
# ══════════════════════════════════════════════════════════════
y3 = y2 + PH2 + 16
py3 = f.panel(0, y3, W, PH3,
              "③ 先堵一个洞：宽的那档参数量本来就多，会不会赢在参数上",
              GR, sub="原作者补了三种对齐方式，这里画其中最干净的一种："
                      "把窄的那档 num_heads 翻倍，两边都是 943 M")

# ⭐ 装置：同样长的一条「料」，切成 32 条窄的 vs 16 条宽的。
#   ⛔ 两条料必须等长 —— 等长就是「参数量对齐」这件事本身。
BW, BXX = 560.0, 300
ALIGN = [("GQA2-128", 32, 2.723, BL, "#e8f0fe", "32 条窄的"),
         ("GQA1-256", 16, 2.720, GR, "#e6f4ea", "16 条宽的")]
for j, (nm, ncut, ls, col, tint, cap) in enumerate(ALIGN):
    yb = py3 + 30 + j * 74
    f.t(BXX - 16, yb + 26, nm, col, True, 18, "end", mono=True)
    f.box(BXX, yb, BW, 44, tint, col, 5, 1.6)
    for k in range(1, ncut):
        xk = BXX + BW * k / ncut
        f.line(xk, yb + 3, xk, yb + 41, col, 0.9, arrow=False)
    f.t(BXX + BW + 16, yb + 18, cap, col, True, 16)
    f.t(BXX + BW + 16, yb + 38, "loss %.3f" % ls, col, True, 16)

_da = ALIGN[0][2] - ALIGN[1][2]
assert abs(_da - 0.003) < 1e-9, _da
f.t(28, py3 + 30 + 168,
    "⭐ 同样 <tspan font-weight=\"700\">943 M</tspan> 参数、同样 "
    "<tspan font-weight=\"700\">512</tspan> 的 cache，只差「切成几条」"
    " ——&#160;<tspan font-weight=\"700\">还是宽的那边赢 %.3f。</tspan>"
    "所以刚才那两级台阶不是参数量堆出来的。" % _da, INK, False, 17)
f.t(28, py3 + 30 + 192,
    "📌 另外两种对齐方式（缩 MLP、给 Q/O 上 LoRA）结论同向，"
    "原文给的幅度是「heads 翻倍相比 head_dims 翻倍，loss 稳定差 0.003 左右」。",
    GY, False, 15)

# ══════════════════════════════════════════════════════════════
yb = y3 + PH3 + 16
yb = f.band(yb, "ok", "带得走的那一条",
            # ⚠️ R46 跨节指针体检抓到：这里原来写「§5.4a」——&#160;
            #   本讲**根本没有 5.4a 这个小节**，是我凭印象编的号。
            #   ⭐ 56.9× 那笔账（4.571 白送 × 12.4 赌出来）在 **§5.0c**。
            #   ⛔ 教训：引别的小节前去 grep 一下标题，别照着记忆写节号。
            ["压得少和学得好，是<tspan font-weight=\"700\">两件事</tspan>。"
             "§5.0c 那笔 56.9× 的账管的是「存多少」，一个字都不用改；"
             "这一格管的是「同样存这么多，学得多好」。",
             "⭐ 所以 §5.3 里那条被 RoPE 逼出来的 64 维窄轨，"
             "<tspan font-weight=\"700\">大概率不是妥协，是这个设计顺手做对的一件事</tspan>"
             " ——&#160;原文的说法是「看似无奈的设计，极有可能是它效果优异的关键原因」。"])
yb = f.band(yb + 10, "warn", "别把这一格读过头",
            ["⛔ <tspan font-weight=\"700\">不是说 MLA 不行</tspan>："
             "同一张表里 MLA 把 head_dims 升到 192+64 就是 2.705，仍然是最好的几档之一。"
             "原作者自己的落点也是「换掉 MLA 的替代品还比不上它」。",
             "⛔ <tspan font-weight=\"700\">不要外推规模</tspan>："
             "这是 ~900M dense、16B tokens 的消融，"
             "DeepSeek-V3 是 671B MoE、十几 T tokens。"
             "结论的<tspan font-weight=\"700\">方向</tspan>值得信，"
             "<tspan font-weight=\"700\">幅度</tspan>不要搬。"])

yb = f.src(yb + 12,
           "📌 全部数字一手核自　苏剑林《Transformer升级之路：20、MLA好在哪里?（上）》"
           "kexue.fm/archives/10907　——　Part I / II / VI 三张表。",
           "📌 公共设置：类 LLAMA3 Dense，hidden 2048 / 12 层 / 16 头，"
           "优化器 Muon，训练长度 4096，总 16B tokens / 16k 步；"
           "除面板③ 外参数量不严格对齐（原文说明）。",
           "📌 「两级台阶 0.039 vs MLA 0.029」这个减法是本课做的，不是原文结论；"
           "原文结论为「增大 head_dims 收益最大，Partial RoPE 也有一定帮助」。")

f.save("fig3-mla-credit.svg", yb + 14)
