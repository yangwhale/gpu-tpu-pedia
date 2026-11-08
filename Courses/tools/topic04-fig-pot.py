# -*- coding: utf-8 -*-
r"""专题四 · §3.3b「Muon 从零讲」第六张：大锅饭 —— 单步平均分，长期论功行赏

2026-11-08 补。现场最后讲通的比方：Muon 一步就是「大锅饭」。
  · 有编制才有份：动量里有这个方向（强度不为 0），才分得到；完全没有的，一粒米都没有。
  · 编制每步按数据重发：方向和谁有份，每一步都由这一步的动量重新决定。
  · 每人约一碗：强度 10 和强度 1 的都分到约 0.7 碗；
    临时工（极弱的方向，0.01）只分到半碗（0.47）。总量最后由学习率定。
  · 单步是大锅饭，长期靠「一直被同向推」论功行赏：每一步都朝同一边的方向越走越远，
    今天朝东明天朝西的方向原地打转 —— 强弱从「单步大小」换成「多步一致性」来表达。

两格：
  Ⓐ 单步分饭：四个方向，强度 10 ／ 1 ／ 0.01 ／ 0 → 0.70 ／ 0.71 ／ 0.47 ／ 0 碗（topic04_muon_nums 现算）。
  Ⓑ 五十步之后：两个方向每步都分到 0.7 碗，一个每步同向，一个每步随机朝东或朝西。
     同向那个走 35；随机那个平均只离起点约 4（理论值 4.4，200 个随机种子平均，断言两者接近）。

刻意简化：Ⓑ 假设每步分到的碗数不变，只看「方向是否一致」这一个变量 —— 真实训练里两样都在变。
"""
import math
import random

from topic03_draw import (Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE)
import topic04_muon_nums as N

W = 1400
SHARE = list(N.FINAL) + [0.0]
STR = ["10", "1", "0.01", "0"]
assert abs(SHARE[2] - 0.47) < 0.005 and SHARE[3] == 0.0 and N.p5(0.0) == 0.0

STEPS, BOWL = 50, 0.7
CONSIST = [BOWL * (k + 1) for k in range(STEPS)]
assert abs(CONSIST[-1] - 35) < 1e-9
random.seed(3)
_ends = []
for _s in range(200):
    pos = 0.0
    for _ in range(STEPS):
        pos += BOWL * random.choice((-1, 1))
    _ends.append(abs(pos))
RAND_AVG = sum(_ends) / len(_ends)
_theory = BOWL * math.sqrt(2 * STEPS / math.pi)
assert abs(RAND_AVG - _theory) < 0.6 and abs(RAND_AVG - 4.4) < 0.6, (RAND_AVG, _theory)
random.seed(5)
WALK = [0.0]
for _ in range(STEPS):
    WALK.append(WALK[-1] + BOWL * random.choice((-1, 1)))
assert max(abs(v) for v in WALK) < 10


def bowl(f, cx, by, level, col):
    """一只碗：上口 84、下底 52、高 46；level ∈ [0,1] 盛到多高。"""
    TW, BWD, H = 84.0, 52.0, 46.0
    if level > 0:
        lh = H * min(level, 1.0)
        tw = BWD + (TW - BWD) * min(level, 1.0)
        f.path("M %.1f %.1f L %.1f %.1f L %.1f %.1f L %.1f %.1f Z"
               % (cx - BWD / 2, by, cx + BWD / 2, by, cx + tw / 2, by - lh, cx - tw / 2, by - lh),
               col, 0.6, arrow=False, fill=col)
    f.path("M %.1f %.1f L %.1f %.1f L %.1f %.1f L %.1f %.1f"
           % (cx - TW / 2, by - H, cx - BWD / 2, by, cx + BWD / 2, by, cx + TW / 2, by - H),
           INK, 2.0, arrow=False)


def main():
    f = Fig(W, "大锅饭。第一格：四个方向，动量里的强度是 10、1、0.01、0，Muon 一步分到的分别是 0.70、0.71、0.47、0 碗："
               "有编制就约一碗，临时工半碗，没编制没有。第二格：五十步后，每步都同向的方向走了 35，"
               "每步随机朝东朝西的方向平均只离起点约 4，强弱从单步大小换成了多步一致性")
    y0 = f.header("一个比方：大锅饭　——　<tspan font-weight=\"700\">单步人人约一碗，长期看谁一直被同向推</tspan>",
                  "碗里盛多少 ＝ 这一步在这个方向上迈多大（拉平之后的强度）",
                  [(GR, "有编制"), (OR, "临时工（极弱的方向）"), (GY2, "没编制")])

    PH = 320
    py = f.panel(0, y0, W, PH, "Ⓐ 这一步怎么分饭", GR, sub="编制（谁有份）每一步都按这一步的数据重新发")
    for j, (sh, st) in enumerate(zip(SHARE, STR)):
        cx = 140 + j * 230
        col = GR if j < 2 else (OR if j == 2 else GY2)
        f.icon("person", cx - 20, py + 30, 40, 44, col=col)
        f.t(cx, py + 100, "动量里强度 %s" % st, INK, True, 14, "middle")
        bowl(f, cx, py + 196, sh, col)
        f.t(cx, py + 226, ("%.2f 碗" % sh) if sh else "0", col, True, 16, "middle")
        f.t(cx, py + 252, ("有编制：约一碗", "有编制：约一碗", "临时工：半碗", "没编制：没有")[j], col, True, 14, "middle")
    tx = 1000
    f.t(tx, py + 70, "强度差一千倍，饭量差不到两倍", INK, True, 15)
    f.t(tx, py + 104, "完全没有的方向，一粒米也不给", INK, True, 15)
    f.t(tx, py + 138, "（不凭空造方向）", GY, size=13)
    f.t(tx, py + 180, "一锅总共多少饭：", INK, True, 14)
    f.t(tx, py + 204, "随人数（频率个数）变，最后再乘学习率", GY, size=13)
    f._pan = None

    PH2 = 350
    py2 = f.panel(0, py + PH + 22, W, PH2, "Ⓑ 五十步之后：论功行赏", BL,
                  sub="两个方向每一步都分到 %.1f 碗；差别只在方向一致不一致" % BOWL)
    X0, X1, YM = 110, 900, py2 + 175
    SC = 3.6

    def xs(k):
        return X0 + k / STEPS * (X1 - X0)

    f.line(X0, YM, X1 + 10, YM, GY2, 1.2, arrow=False)
    f.line(X0, py2 + 40, X0, py2 + 280, GY2, 1.2, arrow=False)
    f.t(X0 - 8, YM + 4, "起点", GY, size=12.5, anchor="end")
    f.t(X1, YM + 22, "第 %d 步" % STEPS, GY, size=12.5, anchor="middle")
    # 同向：截到图框内（35×3.6 ＝ 126 px）
    f.path("M " + " L ".join("%.1f %.1f" % (xs(k), YM - v * SC) for k, v in enumerate([0.0] + CONSIST)),
           GR, 2.8, arrow=False)
    f.path("M " + " L ".join("%.1f %.1f" % (xs(k), YM - v * SC) for k, v in enumerate(WALK)),
           OR, 2.4, arrow=False)
    f.t(xs(STEPS) + 12, YM - CONSIST[-1] * SC + 5, "每步同向：走了 %.0f" % CONSIST[-1], GR, True, 15)
    f.t(xs(STEPS) + 12, YM - WALK[-1] * SC + 20, "今天东明天西：原地打转", OR, True, 15)
    f.t(xs(STEPS) + 12, YM - WALK[-1] * SC + 44, "（平均离起点约 %.1f）" % RAND_AVG, OR, size=13)
    f.t(X0 + 10, py2 + 306, "单步里人人平等；拉开差距的，是「一直被同一个方向推」", INK, True, 14.5)
    f._pan = None

    yb = f.band(py2 + PH2 + 22, "ok", "带走一句：强弱没有消失，是从「单步多大」搬到了「多步多一致」", [
        "Adam 是每个参数自己按历史定步子；Muon 是这一步每个频率约一碗，谁重要靠一步一步攒出来。",
    ])
    yb = f.src(yb + 10,
               "注：本课比方：「大锅饭」「编制」「临时工」是讲课用的说法，不出自论文。"
               "Ⓐ 的碗数就是三个频率例子五轮后的强度（topic04_muon_nums.py 现算并断言），第四个方向强度 0 → 0。",
               "本课推导：Ⓑ 每步 ±%.1f、%d 步随机游走，|终点| 的期望 ＝ %.1f×√(2×%d／π) ≈ %.1f；"
               "200 个随机种子实测平均 %.1f（断言两者接近）。这一格只演示「一致性」这一个变量。"
               % (BOWL, STEPS, BOWL, STEPS, _theory, RAND_AVG),
               "出处：「矩阵版取符号」这一说法：苏剑林《Muon 优化器赏析》把 Muon 写成 msign（矩阵符号函数），"
               "是逐元素 sign 的矩阵推广（kexue.fm/archives/10592）。")
    f.save("fig4-pot.svg", yb + 14)


main()
