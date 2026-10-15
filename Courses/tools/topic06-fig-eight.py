# -*- coding: utf-8 -*-
r"""专题六 · 第十四节「第一天的八问」：接手一个推理服务，第一天按顺序问的八件事。

⭐ 回答的问题：一个陌生模型、一批陌生的卡、一份 SLO 摆在面前，先问什么、后问什么？
   复用第零节路线图的「搬一趟」底图（topic06-fig-intro.py 的 scene()），八问做成编号图钉，
   钉在它问的那个部件上：尺子钉起点、字节账钉车上的货、batch 上限钉座位、PD 钉第二个仓库……
   7 号问拆成两颗钉：7a 量化（绿，少搬）钉货，7b 投机（蓝，一趟分给更多 token）钉座位，与第零节路线图的归类一致。
   图钉颜色对主线三类：灰 ＝ 校准尺子／上线；蓝 ＝ 让一趟分给更多 token；绿 ＝ 少搬；橙 ＝ 把活拆开放对地方。
⛔ 数字：这张图没有承重数字。顺序是本课归纳（先定口径和负载，再定放不放得下，最后才调招数）。
⛔ 刻意没画：每一问拿什么回答、具体阈值（各节里有，正文第十四节逐条写了）。

fig6-s14-day.svg —— 回答：DeepSeek 公开的那一天，token 都花在哪？（第十四节 14.3 末尾）
  · 数从哪来：github.com/deepseek-ai/open-infra-index 202502OpenSourceWeek/day_6（统计 2025-02-27～28）：
    226.75 个 H800 节点；输入 6,080 亿（其中 3,420 亿命中磁盘 KV，56.3%）、输出 1,680 亿。
  · 本课推导：未命中 ＝ 6,080 − 3,420 ＝ 2,660 亿；三段占全天 7,760 亿的 44.1% ／ 34.3% ／ 21.6%；
    输入占 78.4%；1,814 卡 ＝ 226.75 × 8；每卡每秒约 4,950 ＝ 7,760 亿 ÷ 86,400 ÷ 1,814（同台账第 30、144 行）。
  · 颜色照本课约定：命中缓存 ＝ KV 紫，未命中要现算 ＝ prefill 橙，输出 ＝ decode 绿。
  · 刻意没画：545% 利润率、每节点 decode 吞吐（口径另说，正文 14.3 不用）。
"""
import importlib.util
import os

from topic03_draw import Fig, BL, OR, GR, PU, GY, INK, GY2, LINE

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("topic06_fig_intro", os.path.join(HERE, "topic06-fig-intro.py"))
_intro = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_intro)          # 模块末尾有 __main__ 护栏，import 不会顺手重画 life / map

W = 1400
# (颜色, 问题, 钉在场景的哪个部件, 图钉相对锚点的偏移)
Q = [
    (GY, "负载长什么样？", "desk", (0, 0)),
    (GY, "用哪把尺子？", "ruler", (0, 0)),
    (GR, "一步至少搬多少字节？", "cargo", (0, 0)),
    (BL, "放得下几条序列？", "seats", (0, 0)),
    (BL, "attention 怎么切？", "kv", (0, 0)),
    (OR, "拆不拆 PD，配几台？", "wh2", (0, 0)),
    (GR, "量化、投机开不开？", "cargo", (64, 0)),
    (GY, "上线前跑通了吗？", "cab", (0, 0)),
]
assert len(Q) == 8
_CAT = {GY: "校准尺子 ／ 上线", BL: "分给更多 token", GR: "少搬", OR: "拆开放对地方"}
assert set(c for c, *_ in Q) == set(_CAT)


def pin(f, x, y, n, col, down=False):
    """图钉：针尖落在 (x, y)，钉头是带编号的圆；down=True 时钉头朝下。"""
    R, L = 17, 26
    s = 1 if down else -1
    f.line(x, y, x, y + s * L, col, 2.4, arrow=False)
    cy = y + s * (L + R)
    f.p.append('<circle cx="%.1f" cy="%.1f" r="%d" fill="%s" stroke="#ffffff" stroke-width="2.5"/>'
               % (x, cy, R, col))
    lab = str(n)
    f.t(x, cy + 6 if len(lab) == 1 else cy + 5, lab, "#ffffff", True, 17 if len(lab) == 1 else 14, "middle")


def fig_eight():
    f = Fig(W, "第一天的八问，钉在第零节那张搬货图上。仓库是显存，车把权重和 KV 拉到工位。"
               "一，负载长什么样，钉在工位；二，用哪把尺子，钉在尺子起点；三，一步至少搬多少字节，钉在车上的货；"
               "四，放得下几条序列，钉在座位；五，attention 怎么切，钉在仓库里的 KV；六，拆不拆 PD，钉在第二个仓库；"
               "七 a，量化开不开，钉在货上；七 b，投机开不开，钉在座位上；八，上线前跑通了吗，钉在车头")
    y0 = f.header("第一天的八问　——　<tspan font-weight=\"700\">先定尺子和账，再挑招数</tspan>",
                  "还是第零节那一趟：每一问钉在它问的那个部件上",
                  [(c, _CAT[c]) for c in (GY, BL, GR, OR)])
    TOP = 70
    PH = 30 + TOP + 230 + 46 + 26 + 34 + 16      # 尺子下面要给朝下的 2 号钉留位
    py = f.panel(0, y0, W, PH, "八个图钉", INK)
    A, GND, bot = _intro.scene(f, py + TOP)
    for i, (col, q, key, (dx, dy)) in enumerate(Q):
        x, y = A[key]
        pin(f, x + dx, y + dy, "7a" if i == 6 else i + 1, col, down=(key == "ruler"))
    # 7 号问两件事，归两根杠杆：7a 量化少搬（绿，钉货），7b 投机把一趟分给更多 token（蓝，钉座位）
    sx, sy = A["seats"]
    pin(f, sx - 64, sy, "7b", BL)
    # 编号对照：两行，每行四问
    f._pan = None
    ky = py - 30 + PH + 34
    CW = (W - 90) / 4.0                          # 最左 90px 放行名：前四问算账、后四问挑招数（第十四节）
    for i, (col, q, key, _) in enumerate(Q):
        r, c = divmod(i, 4)
        x, y = 90 + c * CW, ky + r * 44
        if c == 0:
            f.t(10, y - 1, "算账" if r == 0 else "挑招数", GY, True, 15)
        if i == 6:   # 7a 量化（绿）／7b 投机（蓝），与图上两颗钉一一对应
            f.badge(x, y - 20, "7a", GR, r=14)
            f.t(x + 38, y - 1, "量化、", INK, True, 17)
            xb = x + 38 + 56
            f.badge(xb, y - 20, "7b", BL, r=14)
            f.t(xb + 38, y - 1, "投机开不开？", INK, True, 17)
            continue
        f.badge(x, y - 20, i + 1, col, r=14)
        f.t(x + 38, y - 1, q, INK, True, 17)
    yb = f.src(ky + 44 + 18, "⚠️ 本课归纳：顺序按「先校准、再算账、后调招数」排，各问拿什么回答、判据见第十四节正文和对应节。")
    f.save("fig6-eight.svg", yb + 6)


# ── DeepSeek 一天的账（单位：亿 token）
D_HIT, D_IN, D_OUT = 3420, 6080, 1680
D_MISS, D_ALL = D_IN - D_HIT, D_IN + D_OUT
D_NODES, D_GPU = 226.75, round(226.75 * 8)
D_PER = D_ALL * 1e8 / 86400 / D_GPU
assert D_MISS == 2660 and D_ALL == 7760 and D_GPU == 1814
assert abs(D_HIT / D_IN * 100 - 56.3) < 0.06          # 342 ÷ 608 ＝ 56.25%，原文写 56.3%
assert round(D_IN / D_ALL * 100, 1) == 78.4
assert round(D_PER, -1) == 4950                        # 4,951 → 台账写「约 4,950」
D_SEG = [(PU, "输入 · 命中缓存", D_HIT), (OR, "输入 · 未命中", D_MISS), (GR, "输出", D_OUT)]
assert [round(v / D_ALL * 100, 1) for _, _, v in D_SEG] == [44.1, 34.3, 21.6]


def fig_day():
    f = Fig(W, "DeepSeek 2025-03 公开的一天。一根横条是全天 7,760 亿 token：输入命中缓存 3,420 亿，占 44.1%；"
               "输入未命中 2,660 亿，占 34.3%；输出 1,680 亿，占 21.6%。输入合计 6,080 亿，占 78.4%，其中 56.3% 命中缓存。"
               "右边：平均 226.75 个 H800 节点，合 1,814 张卡，全天平均每卡每秒约 4,950 token")
    y0 = f.header("近八成 token 是输入，其中一半多直接命中 KV 缓存",
                  "DeepSeek 2025-03 公开的一天（2025-02-27～28），V3 ／ R1 线上服务；条长按 token 数",
                  [(PU, "命中缓存（KV 复用）"), (OR, "未命中（要现算 prefill）"), (GR, "输出（decode）")])
    PH = 250
    py = f.panel(0, y0, W, PH, "一天 7,760 亿 token 怎么分", INK)
    BX, BW, BH = 30, 960, 54
    by = py + 78
    # 输入两段上面一道括号
    xin = BX + BW * D_IN / D_ALL
    f.line(BX, by - 14, xin, by - 14, GY2, 1.4, arrow=False)
    f.line(BX, by - 14, BX, by - 6, GY2, 1.4, arrow=False)
    f.line(xin, by - 14, xin, by - 6, GY2, 1.4, arrow=False)
    f.t((BX + xin) / 2, by - 24, "输入 6,080 亿（78.4%），其中 56.3% 命中缓存", INK, True, 15, "middle")
    x = BX
    for col, lab, v in D_SEG:
        w = BW * v / D_ALL
        f.box(x, by, w - 2, BH, col, col, 3)
        f.t(x + w / 2, by + BH / 2 + 7, "{:,} 亿".format(v), "#ffffff", True, 19, "middle")
        f.t(x + w / 2, by + BH + 28, lab, col, True, 15, "middle")
        f.t(x + w / 2, by + BH + 52, "占全天 %.1f%%" % (v / D_ALL * 100), GY, False, 14, "middle")
        x += w
    # 右边：机器
    RX = BX + BW + 40
    f.line(RX - 20, py + 24, RX - 20, py + PH - 30 - 24, LINE, 1, arrow=False)
    f.t(RX, by - 4, "平均 226.75 个 H800 节点", INK, True, 15)
    f.t(RX, by + 20, "× 8 卡 ＝ 1,814 张卡", GY, False, 14)
    f.t(RX, by + 72, "每卡每秒约 4,950 token", GR, True, 17)
    f.t(RX, by + 96, "全天平均，输入输出都算", GY, False, 14)
    f._pan = None
    yb = f.src(py - 30 + PH + 10,
               "📌 github.com/deepseek-ai/open-infra-index 202502OpenSourceWeek/day_6（2025-03-01 公布，统计 2025-02-27～28）："
               "226.75 节点、输入 6,080 亿（其中 3,420 亿命中磁盘 KV，56.3%）、输出 1,680 亿。",
               "⚠️ 本课推导：未命中 ＝ 6,080 − 3,420 ＝ 2,660 亿；各段占比 ＝ 段 ÷ 7,760 亿；1,814 ＝ 226.75 × 8；"
               "每卡每秒 ＝ 7,760 亿 ÷ 86,400 s ÷ 1,814 卡。分母是全部卡、全天平均，不是 decode 卡的峰值。")
    f.save("fig6-s14-day.svg", yb + 6)


fig_eight()
fig_day()
