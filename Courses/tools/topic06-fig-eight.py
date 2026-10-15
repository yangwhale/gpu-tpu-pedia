# -*- coding: utf-8 -*-
r"""专题六 · 第十四节「第一天的八问」：接手一个推理服务，第一天按顺序问的八件事。

⭐ 回答的问题：一个陌生模型、一批陌生的卡、一份 SLO 摆在面前，先问什么、后问什么？
   复用第零节路线图的「搬一趟」底图（topic06-fig-intro.py 的 scene()），八问做成编号图钉，
   钉在它问的那个部件上：尺子钉起点、字节账钉车上的货、batch 上限钉座位、PD 钉第二个仓库……
   图钉颜色对主线三类：灰 ＝ 校准尺子／上线；蓝 ＝ 让一趟分给更多 token；绿 ＝ 少搬；橙 ＝ 把活拆开放对地方。
⛔ 数字：这张图没有承重数字。顺序是本课归纳（先定口径和负载，再定放不放得下，最后才调招数）。
⛔ 刻意没画：每一问拿什么回答、具体阈值（各节里有，正文第十四节逐条写了）。
"""
import importlib.util
import os

from topic03_draw import Fig, BL, OR, GR, GY, INK, GY2

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
    (GR, "投机和量化开不开？", "cargo", (64, 0)),
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
    f.t(x, cy + 6, str(n), "#ffffff", True, 17, "middle")


def fig_eight():
    f = Fig(W, "第一天的八问，钉在第零节那张搬货图上。仓库是显存，车把权重和 KV 拉到工位。"
               "一，负载长什么样，钉在工位；二，用哪把尺子，钉在尺子起点；三，一步至少搬多少字节，钉在车上的货；"
               "四，放得下几条序列，钉在座位；五，attention 怎么切，钉在仓库里的 KV；六，拆不拆 PD，钉在第二个仓库；"
               "七，投机和量化开不开，也钉在货上；八，上线前跑通了吗，钉在车头")
    y0 = f.header("第一天的八问　——　<tspan font-weight=\"700\">先定尺子和账，再挑招数</tspan>",
                  "还是第零节那一趟：每一问钉在它问的那个部件上",
                  [(c, _CAT[c]) for c in (GY, BL, GR, OR)])
    TOP = 70
    PH = 30 + TOP + 230 + 46 + 26 + 34 + 16      # 尺子下面要给朝下的 2 号钉留位
    py = f.panel(0, y0, W, PH, "八个图钉", INK)
    A, GND, bot = _intro.scene(f, py + TOP)
    for i, (col, q, key, (dx, dy)) in enumerate(Q):
        x, y = A[key]
        pin(f, x + dx, y + dy, i + 1, col, down=(key == "ruler"))
    # 编号对照：两行，每行四问
    f._pan = None
    ky = py - 30 + PH + 34
    CW = W / 4.0
    for i, (col, q, key, _) in enumerate(Q):
        r, c = divmod(i, 4)
        x, y = c * CW + 10, ky + r * 44
        f.badge(x, y - 20, i + 1, col, r=14)
        f.t(x + 38, y - 1, q, INK, True, 17)
    yb = f.src(ky + 44 + 18, "⚠️ 本课归纳：顺序按「先校准、再算账、后调招数」排，各问拿什么回答、判据见第十四节正文和对应节。")
    f.save("fig6-eight.svg", yb + 6)


fig_eight()
