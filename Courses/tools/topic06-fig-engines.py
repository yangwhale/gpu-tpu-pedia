# -*- coding: utf-8 -*-
r"""专题六 · 第零节 0.1「两大引擎你方唱罢我登场」（fig6-engines）。R85 首版，R88 重画，L80（2026-10-16）精简。

⭐ 回答的问题：vLLM 和 SGLang 你追我赶，最关键的是哪几下？
   上面一条蓝线是 vLLM，下面一条红线是 SGLang，中间是月份轴，轴上两个灰色菱形是外部锚点
   （Mooncake 开源、DeepSeek 开源周）。每个时刻一个点，标签只写「年月 ＋ 一件事」，
   跟进的一方多一行带底色的小条（颜色 ＝ 先做的那一家）。
⭐ L80 为什么精简：R90 版三条泳道 33 张卡，现场评语「跟一大篇一样贴在上面」。
   只留 8 个时刻（9 个点：接 Mooncake 两家各一个），其余三十多张卡的内容收进图下的折叠表
   （topic06_s0.py 的 ENGINE_TABLE），一条不丢。
⛔ 8 个时刻与日期都取自第十六节台账「R85 补充」：
   ① 2023-06-20 vLLM PagedAttention 博客　② 2024-01-17 SGLang 首发（原文写明复用 vLLM 代码）
   ③ 2024-07-25 SGLang v0.2（3.1×，对 vLLM 0.5.2）　④ 2024-09-05 vLLM v0.6.0（自承 CPU 开销）
   ⑤ 2025-01-27 vLLM V1（原文：SGLang 等显著影响）　⑥ 接 Mooncake：vLLM 2024-12-15、SGLang 2025-04-10
   ⑦ 2025-05-05 SGLang 大 EP＋PD 复现　⑧ 2026-06-23 SGLang V4-Pro on GB300 5.1×。
   外部锚点：Mooncake 传输引擎 2024-11 开源（README 时间线）；DeepSeek 开源周 2025-02。
⛔ 跟进小条只标原文明说的（复用代码、显著影响）和有日期的先后（接 Mooncake 晚 3 个多月，
   2024-12-15 → 2025-04-10，本课推导，往下取整）。v0.6.0 是不是因为被 v0.2 点名，原文没说，不标。
⛔ 不画引擎之间的连线：先后关系靠横轴和小条已经说清，连线只会横穿中间的轴。
"""
from topic03_draw import Fig, BL, RD, GY, INK, LINE, wpx, MONO_K

W = 1400
XL = 80                      # 2023-06 的 x
PM = 29.5                    # 每月多少 px
M_END = 40                   # 轴画到 2026-10
TINT = {BL: "#e8f0fe", RD: "#fce8e6"}
T_SZ, D_SZ, B_SZ = 14, 12.5, 12
ROW_H = 64                   # 标签行高
GAP = 10                     # 同一行两个标签之间至少留多少 px


def mon(ym):
    y, m = map(int, ym.split("-"))
    return (y - 2023) * 12 + (m - 6)


def xm(m):
    return XL + m * PM


# (年月, 行（1 ＝ 贴着引擎线）, 左对齐/右对齐, [条目])；条目 ("t", 标题) ／ ("d", 细节) ／ ("b", 跟进小条, 先做那家的颜色)
# ⛔ 行号和左右是手排的：远一行的引线不能穿过近一行的标签（下面有断言）。
VLLM = [
    ("2023-06", 1, "L", [("t", "2023-06 PagedAttention"), ("d", "vLLM 首发：KV 按页存")]),
    ("2024-09", 2, "R", [("t", "2024-09 v0.6.0"), ("d", "博客自承：CPU 开销拖累 GPU")]),
    ("2024-12", 2, "L", [("t", "2024-12 接 Mooncake"), ("d", "同月合入首批 PD 代码")]),
    ("2025-01", 1, "L", [("t", "2025-01 V1 重写"), ("b", "原文：受 SGLang 等影响显著", RD)]),
]
SGL = [
    ("2024-01", 2, "L", [("t", "2024-01 SGLang 首发"), ("d", "RadixAttention"), ("b", "原文：复用了 vLLM 的代码", BL)]),
    ("2024-07", 1, "L", [("t", "2024-07 v0.2"), ("d", "纯 Python 调度"), ("d", "最多是 vLLM 0.5.2 的 3.1×")]),
    ("2025-04", 2, "L", [("t", "2025-04 接 Mooncake"), ("b", "晚 vLLM 3 个多月", BL)]),
    ("2025-05", 1, "L", [("t", "2025-05 大 EP＋PD 复现"), ("d", "最多是同卡数纯 TP 的 5×")]),
    ("2026-06", 1, "L", [("t", "2026-06 GB300 × V4-Pro"), ("d", "5.1×（同交互档）"), ("d", "对自家首发日")]),
]
EXT = [("2024-11", "R", "2024-11 Mooncake 开源"), ("2025-02", "L", "2025-02 DeepSeek 开源周")]


def item_w(it):
    k, s = it[0], it[1]
    if k == "t":
        return wpx(s, T_SZ) * MONO_K * 1.08
    if k == "d":
        return wpx(s, D_SZ) * MONO_K
    return wpx(s, B_SZ) * MONO_K + 12


def lane(f, pre, col, name, events, yline, up):
    """画一条引擎线和它的点、标签。up＝标签在线上方。返回 [(x0, x1, y0, y1, 标题)] 供撞车检查。"""
    f.line(XL - 6, yline, xm(M_END), yline, col, 2.4, arrow=False)
    f.t(4, yline - 8 if up else yline + 20, name, col, True, 15)
    boxes, leads = [], []
    for ym, row, side, items in events:
        x = xm(mon(ym))
        w = max(item_w(it) for it in items)
        n = len(items)
        lh = [{"t": 18, "d": 17, "b": 21}[it[0]] for it in items]
        hgt = sum(lh)
        if up:                                      # 标签底边离线 (row-1)*ROW_H + 18
            yb = yline - 16 - (row - 1) * ROW_H
            ytop = yb - hgt
        else:
            ytop = yline + 16 + (row - 1) * ROW_H
            yb = ytop + hgt
        x0 = x + 8 if side == "L" else x - 8 - w
        boxes.append((x0, x0 + w, ytop, yb, items[0][1], row))
        leads.append((x, row, items[0][1]))
        f.p.append('<circle cx="%.1f" cy="%.1f" r="6" fill="%s" stroke="#fff" stroke-width="1.5"/>' % (x, yline, col))
        # 引线：从点到标签那一行
        ye = ytop if not up else yb
        f.line(x, yline + (-7 if up else 7), x, ye, LINE, 1, arrow=False)
        f.line(x, ytop + 2, x, yb - 2, col, 2, arrow=False)        # 标签左（右）边的一根色竖线
        yy = ytop
        anchor = None if side == "L" else "end"
        tx = x0 if side == "L" else x0 + w
        for it, h in zip(items, lh):
            yy += h
            kind, s = it[0], it[1]
            if kind == "t":
                f.t(tx, yy - 4, s, INK, True, T_SZ, anchor)
            elif kind == "d":
                f.t(tx, yy - 4, s, col, False, D_SZ, anchor)
            else:
                bc = it[2]
                bw = wpx(s, B_SZ) * MONO_K + 12
                bx = tx - 2 if side == "L" else tx - bw + 2
                f.box(bx, yy - 18, int(bw + 0.5), 17, TINT[bc], "none", 3, 0)
                f.t(bx + 6, yy - 5, s, bc, False, B_SZ)
    # 同一行不撞、不出界；远行的引线不穿近行的标签
    for a in boxes:
        assert a[0] >= 0 and a[1] <= W - 2, "%s 出界：%s" % (name, a[4])
        for b in boxes:
            if a is b or a[5] != b[5]:
                continue
            if a[0] < b[0]:
                assert a[1] + GAP <= b[0], "%s 第 %d 行撞车：%s / %s" % (name, a[5], a[4], b[4])
    for x, row, title in leads:
        for b in boxes:
            if b[5] < row and b[0] - 4 <= x <= b[1] + 4:
                raise AssertionError("%s：「%s」的引线穿过了「%s」" % (name, title, b[4]))
    return boxes


def fig_engines():
    f = Fig(W, "vLLM 和 SGLang 你追我赶的八个时刻，横轴按月从 2023 年 6 月到 2026 年 10 月。"
               "上面蓝线是 vLLM：2023 年 6 月首发 PagedAttention；2024 年 9 月 v0.6.0，博客自己承认 CPU 开销拖累了 GPU；"
               "2024 年 12 月接上 Mooncake；2025 年 1 月重写成 V1，原文说受 SGLang 等影响显著。"
               "下面红线是 SGLang：2024 年 1 月首发 RadixAttention，原文写着复用了 vLLM 的代码；"
               "2024 年 7 月 v0.2 用纯 Python 调度，吞吐最多是 vLLM 0.5.2 的 3.1 倍；2025 年 4 月接 Mooncake，比 vLLM 晚三个多月；"
               "2025 年 5 月复现大 EP 加 PD 分离；2026 年 6 月在 GB300 上跑 V4-Pro，同交互档每卡是自家首发日的 5.1 倍。"
               "中间轴上两个灰色菱形是外部锚点：2024 年 11 月 Mooncake 开源，2025 年 2 月 DeepSeek 开源周。")
    y0 = f.header("两大引擎　——　<tspan font-weight=\"700\">你方唱罢我登场</tspan>",
                  "只留八个关键时刻，横轴按月等距；带底色的小条是跟进的一方，颜色是先做的那一家",
                  [(BL, "vLLM"), (RD, "SGLang"), (GY, "外部锚点")])
    yv = y0 + 2 * ROW_H + 26            # vLLM 线
    ya = yv + 46                        # 月份轴
    ys = ya + 50                        # SGLang 线
    bv = lane(f, "v", BL, "vLLM", VLLM, yv, True)
    # ── 月份轴 ──
    for m in range(M_END + 1):
        x = xm(m)
        if (m + 5) % 12 == 0 or m == 0:
            f.line(x, ya - 6, x, ya + 6, GY, 1.2, arrow=False)
            f.t(x + 4, ya - 9, "%d" % (2023 + (m + 5) // 12) if m else "2023-06", GY, True, 13)
        elif m % 3 == 0:
            f.line(x, ya - 3, x, ya + 3, GY, 1, arrow=False)
    f.line(XL, ya, xm(M_END), ya, GY, 1.4, arrow=False)
    for ym, side, lab in EXT:
        x = xm(mon(ym))
        d = 7
        f.poly([(x, ya - d), (x + d, ya), (x, ya + d), (x - d, ya)], GY, "#fff", 1.2)
        wl = wpx(lab, 12.5) * MONO_K
        if side == "R":
            f.t(x - 10, ya + 21, lab, GY, True, 12.5, "end")
        else:
            f.t(x + 10, ya + 21, lab, GY, True, 12.5)
    # 两个外部锚点的字不能撞
    (xa, _, la), (xb, _, lb) = [(xm(mon(e[0])), e[1], e[2]) for e in EXT]
    assert xa - 10 <= xb + 10, "外部锚点顺序反了"
    bs = lane(f, "s", RD, "SGLang", SGL, ys, False)
    ybot = max(b[3] for b in bs) + 10
    f.src(ybot, "全部日期见第十六节台账「R85 补充」：vLLM 博客 2023-06-20、2024-09-05（v0.6.0）、2025-01-27（V1，「These engines have significantly influenced our work」）；"
                "vLLM 合入 Mooncake 2024-12-15、首批 PD 代码 2024-12-01；"
                "SGLang lmsys 博客 2024-01-17（「reused some code from … vLLM」）、2024-07-25（v0.2，3.1× 对 vLLM 0.5.2）、"
                "SGLang 支持 Mooncake 2025-04-10、大 EP 博客 2025-05-05（96×H100，5× 对同卡数纯 TP）；"
                "PyTorch 博客 2026-06-23（V4 on GB300，5.1× 对自家 Day-0，同交互档）；"
                "Mooncake GitHub README 时间线（2024-11 开源传输引擎）；DeepSeek open-infra-index（2025-02 开源周）。"
                "晚 3 个多月按两边日期本课推导，往下取整。其余三十多个时间点见图下折叠表。")
    f.save("fig6-engines.svg", ybot + 4)


fig_engines()
