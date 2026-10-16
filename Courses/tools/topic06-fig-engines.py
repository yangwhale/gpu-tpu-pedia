# -*- coding: utf-8 -*-
r"""专题六 · 第零节 0.1「两大引擎你方唱罢我登场」（fig6-engines）。R85 首版，R88（2026-10-16）重画。

⭐ 回答的问题：vLLM 和 SGLang 这三年多谁先做了什么、谁跟进了谁？外面的论文和新玩家在哪儿插进来？
   上中下三条轨：vLLM（蓝）／外部玩家（灰，细）／SGLang（红）。
   横轴按月等距（2023-06 → 2026-09），每张卡的**左边缘就是它的月份**。
⭐ R88 为什么重画：R85 版是「一盘面条」——卡片按宽度自动阶梯堆叠（SGLang 道 7 层），
   十根贝塞尔长箭头从顶到底横穿外部玩家道，蓝红交叉成一团；年刻度只有三根，卡对不准日期。
   ⭐ 改法：
     ① 行号手工指定，每道最多 3 行；挤不下的**合并成一张两段卡**（MTP＋EAGLE-3、接 Mooncake＋大 EP），
       不再靠自动堆叠。
     ② 「谁跟进谁」不画线，改成跟进方卡片里的一条**小色条**（颜色 ＝ 先做的那一家）。
       两条引擎道中间隔着外部玩家道，任何引擎↔引擎的线都必然横穿它 —— 所以干脆不画。
     ③ 只剩一种线：外部玩家 → 引擎的**竖直短虚线**，落在目标卡的左边缘内侧；
       画之前断言它不穿过任何别的卡（见 fig_engines 里「虚线 … 穿过了」那道断言）。
⛔ 跟进关系只标两类，都有原文或日期：
     ① 原文明说的复用／致谢：SGLang 首发博客「reused some code from … vLLM」；
        vLLM V1 博客「SGLang … significantly influenced our work」。
     ② 有日期的先后：分块 prefill、接 Mooncake 传输 vLLM 先；MTP、EAGLE-3、大 EP SGLang 先。
   推测类的借鉴（APC 是否回应 RadixAttention、v0.6.0 是否因被点名、0.11 KV 下放是否跟 HiCache）一律不标。
⛔ 不写任何融资、估值、钱。
⛔ 出处：/tmp/t6/engines-timeline.md 的逐条原文 URL，已抄进第十五节台账「R85 补充」（0.11 KV 下放见「R25 补充」）。
   「晚 3 个多月」「晚 5 周」这类间隔是按两边日期算的（本课推导），口语一律往下取整。
R88 删掉的卡（当时为了腾版面）：vLLM「前缀缓存 APC」「KV Connector」「TPU megakernel」；
   外部「Orca」「Sarathi-Serve」「Mooncake 论文」「DeepSeek-V3」「Dynamo」。
⭐ R90（2026-10-16）找回：创作阶段不卡图面，正文讲到的都该在图上有落点。
   ① vLLM：APC 放回 V1 同一行（V0 默认关 → V1 默认开，一眼对上）；KV Connector 单占顶行
      （LMCache、NIXL 正文都讲）；TPU megakernel 并进 MRV2 卡第二段（2026-09 在最右，单放出右边界）。
   ② 外部：Orca 放进横轴左边的「2022」轴外格（轴断开，不按月）；Sarathi-Serve、Dynamo 各回一张卡；
      Mooncake 论文并进 Mooncake 开源卡的一行细节。
   ③ 为了腾出 2022 格：vLLM／SGLang 道名仍在左栏（左栏里没有卡），外部道名挪到左栏顶上，
      Orca 卡放在它下面；2023-06 从 x=80 挪到 118，每月从 31.4px 收到 30.6px。vLLM 道放宽到 4 行。
   ⛔ 仍不放「DeepSeek-V3」：排在 2024-12 必然挡住 Mooncake 开源 → vLLM 那根虚线；
      它进 SGLang 的虚线要落到 v0.4 卡（SGLang 第 3 行），得穿两行卡。V3 Day-0 已写在 SGLang v0.4 卡上。
"""
from topic03_draw import Fig, BL, RD, GY, INK, LINE, wpx, MONO_K

W = 1400
XL = 118                     # 2023-06 的 x；左边是泳道名栏，外部道在这一栏里放「2022」轴外格（Orca）
XSTUB = 12                   # 轴外格卡片的左边缘
PM = 30.6                    # 每个月多少 px（再小 SGLang v0.2 ↔ MTP 就撞车；再大 V4 on GB300 出右边界）
M_END = 39                   # 2026-09 ＝ 第 39 个月（2023-06 ＝ 0）
TINT = {BL: "#e8f0fe", RD: "#fce8e6"}

T_SZ, D_SZ, B_SZ = 12.5, 11.5, 11
T_LH, D_LH, B_LH = 17, 16, 20
PADX, PADT, PADB = 8, 15, 6


def mon(ym):
    y, m = map(int, ym.split("-"))
    return (y - 2023) * 12 + (m - 6)


def xm(m):
    return XL + m * PM


def xpos(ym):
    """卡片左边缘。'2022' 是轴外格，不按月。"""
    return XSTUB if ym == "2022" else xm(mon(ym))


# (key, 年月, 行, [条目...])；条目：("t", 标题) ／ ("d", 细节) ／ ("b", 跟进色条, 先做那家的颜色)
VLLM = [
    ("pa", "2023-06", 3, [("t", "2023-06 PagedAttention"), ("d", "吞吐最多是 HF 的 24×")]),
    ("apc", "2024-03", 1, [("t", "2024-03 前缀缓存 APC"), ("d", "V0 里默认关")]),
    ("chunk", "2024-04", 3, [("t", "2024-04 分块 prefill"), ("d", "引擎里先落地")]),
    ("v060", "2024-09", 2, [("t", "2024-09 v0.6.0"), ("d", "自承 CPU 开销"), ("d", "拖累 GPU"), ("d", "2.7×（对 v0.5.3）")]),
    ("mc", "2024-12", 3, [("t", "2024-12 接 Mooncake"), ("d", "同月合入首批 PD 代码")]),
    ("v1", "2025-01", 1, [("t", "2025-01 V1 重写"), ("b", "原文：受 SGLang 等影响显著", RD), ("d", "前缀缓存改成默认开")]),
    ("mtp", "2025-02", 2, [("t", "2025-02 MTP"), ("b", "晚 SGLang 5 天", RD),
                           ("t", "2025-04 EAGLE-3"), ("b", "晚 SGLang 5 周", RD)]),
    ("conn", "2025-04", 0, [("t", "2025-04 KV Connector"), ("d", "可插拔的 KV 传输接口"), ("d", "LMCache、NIXL 等接进来")]),
    ("deepep", "2025-06", 3, [("t", "2025-06 接 DeepEP"), ("d", "做大 EP"), ("b", "晚 SGLang 1 个多月", RD)]),
    ("v011", "2025-10", 2, [("t", "2025-10 v0.11 删掉 V0"), ("d", "自带 KV 下放到 CPU"), ("d", "TPU 新后端 tpu-inference")]),
    ("wide", "2025-12", 3, [("t", "2025-12 Wide-EP"), ("d", "每 H200 1.5k → 2.2k tok/s")]),
    ("mrv2", "2026-03", 1, [("t", "2026-03 MRV2"), ("d", "吞吐 +56%（Qwen3-0.6B）"),
                            ("t", "2026-09 TPU megakernel"), ("d", "Inferact")]),
]
EXT = [
    ("orca", "2022", 1, [("t", "2022 Orca"), ("d", "连续批")]),
    ("fg", "2023-11", 0, [("t", "2023-11 DS-FastGen"), ("d", "分块思路")]),
    ("dist", "2024-01", 1, [("t", "2024-01 DistServe"), ("d", "PD 分离")]),
    ("sar", "2024-03", 2, [("t", "2024-03 Sarathi-Serve"), ("d", "分块 prefill")]),
    ("v2", "2024-05", 0, [("t", "2024-05 DeepSeek-V2"), ("d", "MLA")]),
    ("mco", "2024-11", 2, [("t", "2024-11 Mooncake 开源"), ("d", "传输引擎"), ("d", "论文 2024-06：Kimi 的 KV 池")]),
    ("osw", "2025-02", 0, [("t", "2025-02 DeepSeek 开源周"), ("d", "FlashMLA、DeepEP")]),
    ("dyn", "2025-03", 1, [("t", "2025-03 Dynamo 开源"), ("d", "首发即支持两家")]),
    ("v32", "2025-09", 0, [("t", "2025-09 DeepSeek-V3.2"), ("d", "两家同日 Day-0")]),
    ("v4", "2026-04", 0, [("t", "2026-04 DeepSeek-V4"), ("d", "两家都 Day-0")]),
]
SGL = [
    ("radix", "2024-01", 0, [("t", "2024-01 SGLang 首发"), ("d", "RadixAttention"), ("d", "吞吐最多 5×（对 vLLM 等）"),
                             ("b", "原文：复用了 vLLM 的代码", BL)]),
    ("v02", "2024-07", 1, [("t", "2024-07 v0.2"), ("d", "纯 Python 调度"), ("d", "点名 vLLM 的 CPU 开销"),
                           ("d", "3.1×（对 vLLM 0.5.2）"), ("b", "分块 prefill 晚 vLLM 3 个多月", BL)]),
    ("mla", "2024-09", 0, [("t", "2024-09 v0.3 MLA 优化"), ("d", "先手；3–7×（对基线）")]),
    ("v04", "2024-12", 2, [("t", "2024-12 v0.4"), ("d", "零开销调度、DP attention"), ("d", "V3 Day-0，早 vLLM 一两天")]),
    ("mtp", "2025-02", 1, [("t", "2025-02 MTP"), ("d", "先手"), ("t", "2025-03 EAGLE-3"), ("d", "自称首个支持")]),
    ("mc", "2025-04", 0, [("t", "2025-04 接 Mooncake"), ("b", "晚 vLLM 3 个多月", BL),
                          ("t", "2025-05 大 EP＋PD 复现"), ("d", "最多纯 TP 的 5×")]),
    ("hic", "2025-09", 1, [("t", "2025-09 HiCache 分层 KV"), ("d", "KV 下放到 CPU 和远端存储")]),
    ("jax", "2025-10", 2, [("t", "2025-10 SGLang-JAX"), ("d", "上 TPU")]),
    ("gb", "2026-06", 0, [("t", "2026-06 V4 on GB300"), ("d", "5.1×（同交互档）"), ("d", "对自家 Day-0")]),
]
# 外部玩家进入哪个引擎：(外部卡, 引擎卡)。竖虚线落在引擎卡左边缘内侧 10px。
LINKS = [
    ("e.fg", "v.chunk"),    # vLLM 原文：This optimization is on vLLM's roadmap
    ("e.v2", "s.mla"),      # SGLang v0.3：DeepSeek MLA 优化
    ("e.mco", "v.mc"),      # Mooncake README：vLLM officially supports Mooncake Transfer Engine
    ("e.mco", "s.mc"),
    ("e.osw", "v.deepep"),  # vLLM PR #18434 接 DeepEP
]
# 泳道：(前缀, 颜色, 名字, 卡片)。vLLM 第 3 行、SGLang 第 0 行贴着外部道，虚线的目标卡都放在这两行。
# ⛔ 外部道的行号不是随便排的：往上连的卡（DS-FastGen、开源周、Mooncake 开源）上方不能有卡，
#    往下连的卡（DeepSeek-V2、Mooncake 开源）下方不能有卡 —— 断言会拦，但先想清楚省得试。
LANES = [("v", BL, "vLLM", VLLM), ("e", GY, "外部玩家", EXT), ("s", RD, "SGLang", SGL)]


def layout(items):
    """每个条目的基线相对卡顶的偏移 ＋ 卡高。画和量走同一个函数，免得两边算法漂移。"""
    ys, yy = [], PADT
    for k, it in enumerate(items):
        if it[0] == "t" and k:
            yy += 5                   # 两段卡：第二段标题前多空一点
        if it[0] == "b":
            yy += 2
        ys.append(yy)
        yy += {"t": T_LH, "d": D_LH, "b": B_LH}[it[0]]
    last = items[-1][0]
    h = ys[-1] + (8 if last == "b" else 4) + PADB
    return ys, int(h)


def item_w(it, ext):
    k, s = it[0], it[1]
    if k == "t":
        return wpx(s, 12 if ext else T_SZ) * MONO_K * 1.08
    if k == "d":
        return wpx(s, 11 if ext else D_SZ) * MONO_K
    return wpx(s, B_SZ) * MONO_K + 12


def card_wh(items, ext):
    w = max(item_w(it, ext) for it in items) + 2 * PADX + 2
    return int(w + 0.5), layout(items)[1]


def fig_engines():
    f = Fig(W, "vLLM 和 SGLang 的三泳道时间线，横轴按月从 2023 年 6 月到 2026 年 9 月，每张卡的左边缘对准它的月份；"
               "最左边断开的一格是 2022 年，放 Orca 论文的连续批。"
               "上面蓝色是 vLLM，下面红色是 SGLang，中间灰色细道是外部玩家：Orca、DeepSpeed-FastGen、DistServe、Sarathi-Serve、"
               "DeepSeek-V2、Mooncake（2024 年 6 月论文、11 月开源）、DeepSeek 开源周、NVIDIA Dynamo、DeepSeek-V3.2 和 V4，"
               "灰色竖虚线标出它们进了哪个引擎。"
               "vLLM 2024 年 3 月加前缀缓存 APC，V0 里默认关，V1 改成默认开；2025 年 4 月加可插拔的 KV Connector，"
               "LMCache、NIXL 等从这里接进来；2026 年 9 月 Inferact 发 TPU megakernel。"
               "跟进的一方卡片里有一条色条，颜色是先做的那一家。vLLM 2023 年 6 月首发 PagedAttention；"
               "SGLang 2024 年 1 月首发 RadixAttention，原文写着复用了 vLLM 的代码。分块 prefill 和接 Mooncake 都是 vLLM 先、"
               "SGLang 晚三个多月。SGLang 先做 MLA 优化，MTP 早 5 天、EAGLE-3 早 5 周、大 EP 早一个多月。"
               "vLLM 2025 年 1 月重写成 V1，原文说受 SGLang 等影响显著。2025 年 9 月 SGLang 推 HiCache，"
               "10 月 vLLM 0.11 自带 KV 下放到 CPU。DeepSeek-V3.2、V4 两家都在首日支持")
    y0 = f.header("两大引擎　——　<tspan font-weight=\"700\">你方唱罢我登场</tspan>",
                  "横轴按月等距（最左 2022 一格在轴外），卡片左边缘就是首次公开的月份；卡里带底色的小条写的是跟进了谁、晚多久",
                  [(BL, "vLLM"), (RD, "SGLang"), (GY, "外部玩家（竖虚线：进了哪个引擎）")])
    # ── 月份刻度 ──
    ya = y0 + 6
    for m in range(M_END + 1):
        x = xm(m)
        if m % 3 == 0:
            f.t(x, ya + 10, "%d-%02d" % (2023 + (m + 5) // 12, (m + 5) % 12 + 1), GY, False, 11, "middle")
            f.line(x, ya + 15, x, ya + 22, GY, 1, arrow=False)
        else:
            f.line(x, ya + 18, x, ya + 22, LINE, 1, arrow=False)
    f.line(XL, ya + 22, xm(M_END), ya + 22, LINE, 1, arrow=False)
    # 轴外格「2022」：和主轴之间断开（两道斜杠），提示这一格不按月
    xs1 = XL - 16
    f.t((XSTUB + xs1) / 2, ya + 10, "2022", GY, False, 11, "middle")
    f.line(XSTUB, ya + 22, xs1 - 4, ya + 22, LINE, 1, arrow=False)
    for dx in (-3, 3):
        f.line(XL - 8 + dx - 3, ya + 27, XL - 8 + dx + 3, ya + 17, GY, 1, arrow=False)
    y = ya + 30
    ytop = y
    geo = {}                      # key -> (x, y, w, h)
    lanebox = []
    for pre, col, name, cards in LANES:
        ext = pre == "e"
        nrow = max(c[2] for c in cards) + 1
        assert nrow <= 4, "%s 道超过 4 行" % name
        sz = {c[0]: card_wh(c[3], ext) for c in cards}
        rh = [max(sz[c[0]][1] for c in cards if c[2] == r) for r in range(nrow)]
        gap = 6 if ext else 8
        ry = [y + gap + sum(rh[:r]) + gap * r for r in range(nrow)]
        h = sum(rh) + gap * (nrow + 1)
        lanebox.append((y, h, col, name))
        for key, ym, r, items in cards:
            w, hh = sz[key]
            geo[pre + "." + key] = [xpos(ym), ry[r], w, hh, items, col, ext, r]
        # 同一行不重叠、不出右边界
        for r in range(nrow):
            row = sorted((g for k, g in geo.items() if k[0] == pre and g[7] == r), key=lambda g: g[0])
            for a, b in zip(row, row[1:]):
                assert a[0] + a[2] + 6 <= b[0], "%s 道第 %d 行撞车：%s / %s" % (name, r, a[4][0][1], b[4][0][1])
            for g in row:
                assert g[0] + g[2] <= W - 2, "出右边界：%s" % g[4][0][1]
                if g[0] < XL:     # 轴外格的卡不能伸进 2023-06
                    assert g[0] + g[2] + 6 <= XL, "轴外格的卡太宽：%s" % g[4][0][1]
        y += h + 10
    ybot = y - 10
    # 外部卡要盖住它的虚线落点：不够宽就加宽（只加宽，不挪位置）
    for a, b in LINKS:
        xb = geo[b][0] + 10
        need = xb + 10 - geo[a][0]
        if geo[a][2] < need:
            geo[a][2] = int(need)
    # ── 轨道底框 → 网格 → 虚线 → 卡片 ──
    for ly, h, col, name in lanebox:
        f.box(0, ly, W, h, "none", LINE, 6)
        f.box(0, ly, 4, h, col, col, 0)
        if col == GY:     # 外部道：左栏下面放 Orca，道名挪到顶上
            f.t(14, ly + 22, name, col, True, 13)
        else:
            f.t(14, ly + h / 2 + 6, name, col, True, 15)
    for m in range(0, M_END + 1, 3):
        f.line(xm(m), ytop, xm(m), ybot, "#e8eaed", 1, "3 4", arrow=False)
    for a, b in LINKS:
        xa_, ya_, wa, ha = geo[a][:4]
        xb_, yb_, wb, hb = geo[b][:4]
        x = xb_ + 10
        assert xa_ + 6 <= x <= xa_ + wa - 6, "虚线没落在外部卡上：%s" % a
        if yb_ < ya_:
            y1, y2 = ya_, yb_ + hb + 3
        else:
            y1, y2 = ya_ + ha, yb_ - 3
        lo, hi = min(y1, y2), max(y1, y2)
        for k, g in geo.items():
            if k in (a, b):
                continue
            if g[0] - 2 <= x <= g[0] + g[2] + 2 and g[1] < hi and g[1] + g[3] > lo:
                raise AssertionError("虚线 %s→%s 穿过了 %s" % (a, b, g[4][0][1]))
        f.line(x, y1, x, y2, GY, 1.2, "3 3")
    for key, (x, cy, w, hh, items, col, ext, r) in geo.items():
        f.box(x, cy, w, hh, "#fff", col, 5, 1 if ext else 1.3)
        for it, oy in zip(items, layout(items)[0]):
            kind, s = it[0], it[1]
            yy = cy + oy
            if kind == "t":
                f.t(x + PADX, yy, s, INK if not ext else GY, True, 12 if ext else T_SZ)
            elif kind == "d":
                f.t(x + PADX, yy, s, col, False, 11 if ext else D_SZ)
            else:
                bc = it[2]
                bw = wpx(s, B_SZ) * MONO_K + 12
                f.box(x + PADX - 2, yy - 12.5, int(bw + 0.5), 17, TINT[bc], "none", 3, 0)
                f.t(x + PADX + 4, yy, s, bc, False, B_SZ)
    f.src(y, "vLLM：blog.vllm.ai 2023-06-20（对 HF 24×）、2024-09-05（v0.6.0，2.7× 对 v0.5.3）、2025-01-27（V1）；"
             "PR #2762（APC，2024-03-02）、#3884（分块 prefill，2024-04-11）、#10884（Mooncake，2024-12-15）、#10502（PD，2024-12-01）、"
             "#12755（MTP，2025-02-19）、#16937（EAGLE-3，2025-04-25）、#15960（KV Connector，2025-04-17）、#18434（DeepEP，2025-06-10）；"
             "v0.11.0 release（删 V0）与 vllm.ai/blog 2026-01-08（OffloadingConnector「introduced in vLLM 0.11.0」）；"
             "blog.vllm.ai 2025-10-16（tpu-inference）；vllm.ai/blog 2025-12-17（Wide-EP）、2026-03-24（MRV2）；"
             "inferact.ai/blog/tpu-megakernels（TPU megakernel）。"
             "SGLang：lmsys.org/blog 2024-01-17（5× 对 Guidance、vLLM；复用 vLLM 代码）、2024-07-25（v0.2，3.1× 对 vLLM 0.5.2；"
             "原文评 vLLM 调度 CPU 开销高）、2024-09-04（v0.3 MLA 3–7×）、2024-12-04（v0.4）、2025-05-05（大 EP，96×H100）、"
             "2025-09-10（HiCache）、2025-10-29（SGLang-JAX）；"
             "PR #800（分块 prefill，2024-07-29）、#3582（MTP，2025-02-14）、#4247（EAGLE-3，2025-03-18）；v0.4.1 release（V3 Day-0）；"
             "PyTorch 博客 2026-06-23（V4 on GB300）。"
             "外部：Orca，OSDI'22；DeepSpeed-FastGen 博客与 vLLM 回应（2023-11-14）；DistServe arXiv 2401.09670；"
             "Sarathi-Serve arXiv 2403.02310；DeepSeek-V2 arXiv 2405.04434；Mooncake arXiv 2407.00079 与 GitHub README；"
             "DeepSeek open-infra-index；NVIDIA Dynamo 新闻稿 2025-03-18；"
             "DeepSeek-V3.2 两家 Day-0 博客 2025-09-29；V4 两家博客 2026-04-24／25。"
             "间隔（晚 5 天、5 周、1 个多月、3 个多月）按两边日期本课推导，往下取整。推测的借鉴不标。")
    f.save("fig6-engines.svg", ybot + 6)


fig_engines()
