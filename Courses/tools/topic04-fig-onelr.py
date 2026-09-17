# -*- coding: utf-8 -*-
r"""专题四 · §3.3「一个学习率伺候不了所有参数」——&#160;两头都不满意的那个折中

⭐⭐⭐ 2026-09-17 新画。这一节原来只有一句纯文字的断言：
  「不同参数的梯度尺度可能差好几个数量级，结果就是梯度大的走过头、
   梯度小的几乎不动 ——&#160;你调学习率其实是在两头之间找一个都不太满意的折中。」
  ⛔ 这句话**对，但没有画面**。台下点头，可他没看见「走过头」长什么样。

⭐⭐ 取法来自**李宏毅**。2025 年那份《Training Tip》上把这个两难
  钉成了两句极具体的话（原句，繁体）：
    · η ＝ 0.001　——&#160;**Update 第二次就飛出了地圖之外**
    · η ＝ 0.0001 ——&#160;**Update 一百次都還走不到谷底**
    · 结论：**不同參數應該要有不同的 Learning Rate**
  ⭐ 他那两个数是课堂示意，**我们不照抄** ——&#160;抄一个别人挑出来的数
    等于把别人的地形当成自己的。这里自己搭谷、自己跑，让轨迹自己长出来。

⭐⭐⭐ 而本讲自己的落点有两条，李宏毅那两句里都没有：
  ① **两条轨迹的学习率只差一成** ——&#160;一条炸、一条慢。能下手的区间就这么窄。
  ② **中间也没有好的**：在不发散的前提下能用的**最大**学习率，
     让平方向走完一半仍然要 `0.37 × 陡峭比` 步 ——&#160;
     **这不是调参没调好，是谷的形状决定的，而形状是模型给你的。**

⚠️ 图上是一个**二维的二次谷**，不是真实 loss 曲面。
  它只说明「尺度差」这一件事；真实曲面还有别的花样（见 `fig-saddle`）。
"""
import math

from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE)

W = 1400

# ══════════════════════════════════════════════════════════════════
# 谷的形状：L ＝ ½(A1·w1² ＋ A2·w2²)，梯度 ＝ (A1·w1, A2·w2)
#
# ⛔⛔ 这里踩过一个**画图特有**的坑，值得记下来：
#   第一版把陡峭比直接设成 1,000（对应正文那句「差好几个数量级」）——&#160;
#   **物理完全正确，画面彻底废了**：平方向 44 步只挪了 40 像素，
#   于是那条「锯齿」退化成一根竖线，看不出任何锯齿。
#   ⭐⭐ 判据：**一张要「让读者看见某个动态」的图，
#     参数按「看得见」选，不按「最真实」选** ——&#160;
#     真实那个量级用**公式外推**交代，不要硬塞进画面。
#   ⭐ 所以现在是：**画面用陡峭比 25（锯齿清清楚楚），
#     Ⓑ 给闭式公式，再用它把真实量级算出来。** 两头都不含糊。
# ══════════════════════════════════════════════════════════════════
A1, A2 = 25.0, 1.0
KAPPA = A1 / A2                   # ＝ 25，这是**画面**用的
KAPPA_REAL = 1000.0               # 正文那句「差好几个数量级」的量级

# 沿陡方向每步的放大倍数是 |1 − η·A1|，所以**发散门槛就是 η ＝ 2 / A1**。
ETA_MAX = 2.0 / A1
ETA_BIG = ETA_MAX * 1.05          # 只大 5% → 炸
ETA_OK = ETA_MAX * 0.95           # 只小 5% → 已是不炸的前提下最大的那个
W0 = (1.5, 8.0)


def run(eta, n, w=W0):
    out, w1, w2 = [w], w[0], w[1]
    for _ in range(n):
        w1 -= eta * A1 * w1
        w2 -= eta * A2 * w2
        out.append((w1, w2))
    return out


TRAJ_BIG = run(ETA_BIG, 16)
TRAJ_OK = run(ETA_OK, 22)

# ⭐ 图上那几句话，必须是跑出来的，不是写上去的
assert abs(TRAJ_BIG[-1][0]) > abs(TRAJ_BIG[0][0]) * 1.5, \
    "大学习率那条必须真的发散，不然「飞出画面」是我编的"
assert all(TRAJ_BIG[i][0] * TRAJ_BIG[i + 1][0] < 0
           for i in range(len(TRAJ_BIG) - 1)), \
    "而且它必须是**左右横跳**着发散的 ——　锯齿是这张图的全部画面"
assert all(abs(TRAJ_OK[i + 1][0]) < abs(TRAJ_OK[i][0])
           for i in range(len(TRAJ_OK) - 1)), \
    "小一点那条在陡方向上必须收敛"


def half_steps(kappa):
    """平的那个方向走完一半要几步 ——&#160;闭式解，不是数出来的。

    每步收缩 (1 −&#160;η·A2)，而 η ＝ 0.95 × 2/A1，
    于是收缩率只跟**陡峭比**有关：1 −&#160;1.9 / kappa。
    """
    return math.log(0.5) / math.log(1.0 - 1.9 / kappa)


HALF_PIC = int(round(half_steps(KAPPA)))            # 画面里这个谷
HALF_REAL = int(round(half_steps(KAPPA_REAL)))      # 真实量级

# ⭐⭐ 闭式解要跟模拟对得上，否则那条公式是我瞎写的
_sim = next(i for i, (w1, w2) in enumerate(run(ETA_OK, 400))
            if w2 <= W0[1] / 2.0)
assert abs(_sim - HALF_PIC) <= 1, \
    "闭式解 %d 跟真跑出来的 %d 对不上" % (HALF_PIC, _sim)
assert HALF_REAL > 300, "外推到真实量级要足够难看，现在是 %d" % HALF_REAL


def main():
    f = Fig(W, "把两个参数放在一起看：一个方向陡、一个方向平。"
               "用同一个学习率，大 5% 那条在陡的方向上来回弹、越弹越大，"
               "直接冲出画面；小 5% 那条弹幅在收，可它也是一路锯齿着蹭过去的，"
               "每一步大半的位移花在左右横跳上，真正朝谷底去的只有一点点。"
               "两条轨迹的学习率只差一成，你能下手的区间就这么窄。"
               "更要紧的是中间也没有好的：在不发散的前提下能用的最大学习率，"
               "让平方向走完一半要零点三七乘以陡峭比那么多步 —— "
               "图上这个谷是九步，真实模型上是三百多步。"
               "这不是调参没调好，是谷的形状决定的")

    y0 = f.header(
        "一个学习率<tspan font-weight=\"700\">伺候不了所有参数</tspan>"
        "　——　而中间那个「折中」也不存在",
        "⭐ 同一个谷、同一套规则，只改学习率 ——&#160;"
        "<tspan font-weight=\"700\">两条轨迹都是真跑出来的</tspan>",
        [(RD, "大 5%：炸"), (BL, "小 5%：慢"), (PU, "中间：也不行")])

    # ══════════ Ⓐ 一个细长的谷，两条轨迹 ═══════════════════════════
    PH = 470
    py = f.panel(0, y0, W, PH,
                 "Ⓐ 这个谷<tspan font-weight=\"700\">一个方向陡、一个方向平</tspan>"
                 "　——　陡峭程度差 <tspan font-weight=\"700\">%d 倍</tspan>"
                 % int(KAPPA), PU,
                 sub="⭐ 竖着弹的是陡的那个参数，横着爬的是平的那个")

    # ⛔⛔ 这里踩过第二个坑：发散那条 |w1| 会一路涨出面板，
    #   而我只给**圆点**加了越界判断、**路径本身没裁** ——&#160;
    #   渲染出来是一根贯穿全图的竖线，把另一条轨迹整个盖住。
    #   SVG 合法、脚本不报错、面板越界自检也不管（它只看下沿，不看上方）。
    #   ⭐ 判据：**凡是「让它自己跑出来」的轨迹，都要假设它会跑出画布；
    #     裁剪做在<生成路径>的地方，不是做在<画点>的地方。**
    CX, CY = 1080, py + 212
    SX, SY = 46.0, 66.0
    YLIM = 168.0

    def px(w2):
        return CX - w2 * SX

    def pyv(w1):
        # ⚠️ SVG 的 y 向下长：w1 为正 ＝ 屏幕上**偏上** ＝ y 更小
        return CY - w1 * SY

    for c in (6.0, 20.0, 45.0, 80.0):
        r1, r2 = math.sqrt(2 * c / A1), math.sqrt(2 * c / A2)
        f.p.append('<ellipse cx="%.1f" cy="%.1f" rx="%.1f" ry="%.1f" '
                   'fill="none" stroke="%s" stroke-width="1" '
                   'stroke-dasharray="3 4"/>'
                   % (CX, CY, r2 * SX, r1 * SY, GY2))
    f.line(180, CY, 1352, CY, GY2, 1, arrow=False)
    f.box(CX - 5, CY - 5, 10, 10, INK, INK, 5)
    f.t(CX + 14, CY + 22, "谷底", INK, True, 13)

    def draw(traj, col, nmax, sw=2.0):
        """⭐ 裁剪在这儿做：越出可视半幅就停。"""
        pts = []
        for (w1, w2) in traj[:nmax + 1]:
            if abs(w1) * SY > YLIM:
                break
            pts.append((px(w2), pyv(w1)))
        d = "M %.1f %.1f" % pts[0]
        for xx, yy in pts[1:]:
            d += " L %.1f %.1f" % (xx, yy)
        f.path(d, col, sw, arrow=False)
        for xx, yy in pts:
            f.box(xx - 3.5, yy - 3.5, 7, 7, col, col, 4)
        return pts

    draw(TRAJ_OK, BL, 22)
    red = draw(TRAJ_BIG, RD, 16, sw=2.4)
    # 红色冲出去的那一段：顺着它最后的方向补一根出画面的箭头
    _up = red[-1][1] < CY
    f.line(red[-1][0], red[-1][1], red[-1][0] - 26,
           (CY - YLIM - 20) if _up else (CY + YLIM + 20), RD, 2.4)
    f.t(red[-1][0] - 36, (CY - YLIM - 26) if _up else (CY + YLIM + 16),
        "⛔ 出画面了", RD, True, 14, "end")

    f.box(px(W0[1]) - 7, pyv(W0[0]) - 7, 14, 14, "#fff", INK, 7, sw=1.8)
    f.t(px(W0[1]) + 16, pyv(W0[0]) - 10, "起点", INK, True, 13)

    f.t(80, CY - 150, "⛔ 红：学习率只<tspan font-weight=\"700\">大 5%</tspan>",
        RD, True, 15)
    f.t(80, CY - 128, "每弹一次幅度更大 ——　收不住", GY, size=12.5)
    f.t(80, CY - 16, "↑ 竖直 ＝ 陡的那个参数", GY2, size=12)
    f.t(80, CY + 14, "→ 横向 ＝ 平的那个参数", GY2, size=12)
    f.t(80, CY + 118, "✅ 蓝：学习率只<tspan font-weight=\"700\">小 5%</tspan>",
        BL, True, 15)
    f.t(80, CY + 140, "不炸了 ——　可你看那串锯齿：", GY, size=12.5)
    f.t(80, CY + 160, "<tspan font-weight=\"700\">每一步大半的力气花在左右横跳上</tspan>",
        GY, size=12.5)
    f.t(80, CY + 180, "真正朝谷底去的只有那一丁点横向位移", GY, size=12.5)

    f.t(700, py + 432,
        "⭐⭐⭐ 两条轨迹的学习率<tspan font-weight=\"700\">只差一成</tspan>"
        "　——　一条炸了，一条慢得让人着急。"
        "<tspan font-weight=\"700\">你能下手的区间就这么窄。</tspan>",
        INK, size=15, anchor="middle")
    f._pan = None

    # ══════════ Ⓑ 中间也没有好的 ═══════════════════════════════════
    PH2 = 310
    py2 = f.panel(0, py + PH + 20, W, PH2,
                  "Ⓑ ⭐⭐⭐ 那<tspan font-weight=\"700\">取中间那个值</tspan>呢"
                  "　——　问题就在这儿：<tspan font-weight=\"700\">中间也没有好的</tspan>", RD,
                  sub="⭐ 下面三行不是估的，是<tspan font-weight=\"700\">算出来的</tspan>")

    FACTS = (
        (RD, "再大 5%", "陡的方向直接发散",
         "门槛卡死在 <tspan font-weight=\"700\">2 ÷ 陡峭程度</tspan>"
         "　——　跟你想不想快无关"),
        (BL, "就取最大的安全值", "平的方向走完<tspan font-weight=\"700\">一半</tspan>",
         "要 <tspan font-weight=\"700\">0.37 × 陡峭比</tspan> 步　——　"
         "图上这个谷（比 %d）＝ <tspan font-weight=\"700\">%d 步</tspan>"
         % (int(KAPPA), HALF_PIC)),
        (PU, "⭐ 而真实模型呢", "梯度尺度差几个数量级",
         "比值按 <tspan font-weight=\"700\">%d</tspan> 算，同一条公式给出 "
         "<tspan font-weight=\"700\">%d 步</tspan>　——　"
         "而这<tspan font-weight=\"700\">已经是最快的了</tspan>"
         % (int(KAPPA_REAL), HALF_REAL)),
    )
    for i, (col, q, a, why) in enumerate(FACTS):
        ry = py2 + 44 + i * 68
        f.box(50, ry, 1300, 56, "#fff", col, 8, sw=1.8 if i == 2 else 1.0)
        f.box(50, ry, 5, 56, col, col, 2)
        f.t(86, ry + 35, q, col, True, 15.5)
        f.t(340, ry + 35, a, INK, True, 15)
        f.t(700, ry + 35, why, GY, size=13)

    f.t(700, py2 + 272,
        "⭐⭐⭐ 所以这不是<tspan font-weight=\"700\">「参数没调好」</tspan>"
        "　——　<tspan font-weight=\"700\">是这个谷的形状决定的，"
        "而形状是模型给你的，不是你能选的。</tspan>",
        INK, size=15, anchor="middle")
    f._pan = None

    yb = f.band(py2 + PH2 + 20, "warn",
                "出路只有一条：别再找那个「最好的全局学习率」了，它不存在",
                ("⭐⭐⭐ <tspan font-weight=\"700\">给每个参数配它自己的那一个。</tspan>"
                 "——&#160;而「它自己的」该是多少？"
                 "下一格给标准答案（<tspan font-weight=\"700\">一阶导 ÷ 二阶导</tspan>），"
                 "再往后讲为什么真实训练里只能<tspan font-weight=\"700\">估</tspan>它。",
                 "⛔ 代价也在这儿："
                 "<tspan font-weight=\"700\">「每个参数一个」意味着要为每个参数存东西</tspan>"
                 "　——&#160;这一讲开头那笔 12 字节的优化器状态，"
                 "根子就是这张图逼出来的。",
                 "⚠️ 图上是一个<tspan font-weight=\"700\">二维的二次谷</tspan>，不是真实 loss 曲面。"
                 "它只用来说明「尺度差」这一件事。"))

    yb = f.src(yb + 16,
               "📌 「同一个学习率，大了第二次更新就飞出地图之外、"
               "小了更新一百次还走不到谷底，所以不同参数应该有不同的 learning rate」"
               "这个两难取自 <tspan font-weight=\"700\">李宏毅</tspan>"
               "《Training Tip》投影片（2025 秋 GenAI-ML 课程）。",
               "⛔ 但<tspan font-weight=\"700\">他那两个具体数字我们没照抄</tspan> ——&#160;"
               "抄一个别人挑出来的学习率，等于把别人的地形当成自己的。"
               "这里自己搭了谷、自己跑了两遍，每条轨迹都有 assert 盯着。",
               "⚠️ <tspan font-weight=\"700\">画面用的陡峭比是 %d，不是正文说的「几个数量级」。</tspan>"
               "第一版按 1,000 画，物理没错但画面废了：平方向几十步只挪几十像素，"
               "锯齿退化成一根竖线。⭐ 判据：<tspan font-weight=\"700\">"
               "要让人「看见」某个动态，参数按「看得见」选，真实量级交给公式外推。"
               "</tspan>Ⓑ 第三行就是那条外推。" % int(KAPPA),
               "⭐ Ⓑ 那两个步数是<tspan font-weight=\"700\">闭式算的</tspan>："
               "平方向每步收缩 (1 −&#160;1.9 ÷ 陡峭比)，走到一半就是 "
               "log½ ÷ log(那个收缩率)。<tspan font-weight=\"700\">"
               "脚本里拿真跑一遍的结果对过账</tspan>，差不超过 1 步。")

    f.save("fig4-onelr.svg", yb + 14)


if __name__ == "__main__":
    main()
