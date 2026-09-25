# -*- coding: utf-8 -*-
r"""静态图最小模板 —— 复制成 tools/topic0N-fig-<小节>.py 再改。

用法（在 Courses/ 目录下）：
    python3 构建手册/模板/fig-模板.py
产物写到 tools/（基元库 save() 固定写在它自己的目录）：
    tools/fig-kit-demo.svg ＋ tools/fig-kit-demo.src.html（出处片段）
试完删掉这两个文件；正式的图由 topic0N-build.py 的 FIGS 表登记后内联进页面。

⭐ 文件头写三样：这张图回答哪一个问题；承重的数字从哪来；刻意没画什么（免得以后有人「补全」）。
   这张示例图回答：「AllReduce 为什么正好等于 ReduceScatter ＋ AllGather？」
   数字：NCCL nccl-tests PERFORMANCE.md 的 busbw 修正系数 —— AllReduce 2(n−1)/n，另两个各 (n−1)/n。
   刻意没画：延迟项。图只讲带宽，延迟要另开一张。
"""
import os
import sys

# 模板放在 构建手册/模板/ 下，所以要把 tools/ 加进路径；复制进 tools/ 之后这三行可以删掉
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))

from topic03_draw import Fig, BL, OR, GR, GY, INK   # noqa: E402

W = 1400

# ── ① 数字在最上面现算，每个承重的数一条窄断言 ─────────────────────────
N = 4                                       # 例子里的卡数
RS = (N - 1) / N                            # ReduceScatter：每卡发出整份的 (n−1)/n
AG = (N - 1) / N                            # AllGather：同上
AR = 2 * (N - 1) / N                        # AllReduce：NCCL 给的系数
assert abs(RS + AG - AR) < 1e-12            # 这张图的论点本身，必须是算出来成立的
assert abs(AR - 1.5) < 1e-12, AR            # n=4 时每卡发出 1.5 份 —— 图上写的就是这个数


def fig_demo():
    # ── ② aria-label 必填：一两句话完整描述这张图，给读屏用 ─────────────
    f = Fig(W, "四张卡做一次 AllReduce，每张卡要发出 1.5 份数据；"
               "拆成先 ReduceScatter 再 AllGather，两步各发出 0.75 份，加起来还是 1.5 份，一个字节不多")

    # ── ③ 标题一句话说结论；副标题是读图钥匙（放上面，不放图底）；图例 ────
    y0 = f.header("AllReduce ＝ ReduceScatter ＋ AllGather　——　"
                  "<tspan font-weight=\"700\">拆开做，一个字节不多</tspan>",
                  "条长 ＝ 每张卡要发出多少份数据（整份算 1）。四张卡",
                  [(GR, "ReduceScatter"), (BL, "AllGather"), (OR, "一步到位的 AllReduce")])

    # ── ④ 面板里画原理：长度对长度，删掉字也看得出「两段拼起来一样长」 ──
    PH, BX, SC = 200, 260, 600              # 面板高、条形起点、每份多少像素
    py = f.panel(0, y0, W, PH, "每张卡发出的量", GR)
    f.t(40, py + 70, "一步到位", INK, True, 16)
    f.box(BX, py + 48, AR * SC, 36, OR, OR, 4)
    f.t(BX + AR * SC + 14, py + 72, "%.2f 份" % AR, INK, True, 15)
    f.t(40, py + 140, "拆成两步", INK, True, 16)
    f.box(BX, py + 118, RS * SC, 36, GR, GR, 4)
    f.box(BX + RS * SC, py + 118, AG * SC, 36, BL, BL, 4)
    f.t(BX + AR * SC + 14, py + 142, "%.2f ＋ %.2f 份" % (RS, AG), INK, True, 15)

    # ── ⑤ 落点带最多两行：换来了什么 ／ 代价是什么 ────────────────────
    yb = f.band(py + PH + 20, "ok", "所以后面好几种并行能「白捡」", [
        "拆开之后两半可以放在不同的时间点做，中间还能插进别的事（比如只更新自己那一份）。",
        "代价是多一次同步点；数据很小时比的是步数，不是带宽。",
    ])

    # ── ⑥ 出处与口径：折进图下；推导标「本课推导」 ─────────────────────
    yb = f.src(yb + 10,
               "📌 系数取自 NCCL nccl-tests PERFORMANCE.md（busbw 修正）：AllReduce 2(n−1)/n，AllGather ／ ReduceScatter (n−1)/n。",
               "⚠️ 本课推导：「拆开一个字节不多」＝ (n−1)/n × 2 ＝ 2(n−1)/n。")

    # ── ⑦ 完整文件名，带 .svg ─────────────────────────────────────────
    f.save("fig-kit-demo.svg", yb + 14)


fig_demo()
