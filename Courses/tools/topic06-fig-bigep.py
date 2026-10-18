# -*- coding: utf-8 -*-
r"""专题六 · 第六节「大 EP」的两张静态图（动画 BigEP 在 manim/topic06-anim-serve.py）。

回答的问题
  · fig6-bigep-what  EP 和大 EP 差在哪？同一个 V3，一台机器里铺开 vs DeepSeek 线上 decode 铺到 18 台。
  · fig6-bigep-net   大 EP 对网络要什么？卡越多每趟传送多慢？是不是非 NVL72／ICI 不可？

数从哪来
  · DeepSeek open-infra-index Day 6（Inference System Overview）：decode 路由专家 EP144、MLA／共享专家 DP144，
    每个部署单元 18 个节点、32 个冗余路由专家，每卡 2 个路由专家 ＋ 1 个共享专家。
  · DeepSeek-V3 技术报告 arXiv 2412.19437 §3.4：机内 NVLink、全集群 IB 全互连；decode 走 IB 点对点并用 IBGDA。
  · DeepEP v1.2.1 README：测试机 H800（NVLink 约 160 GB/s）、每卡一张 CX7 400 Gb/s IB 网卡（约 50 GB/s）；
    低延迟 kernel 表（每批 128 token、hidden 7168、top-8、FP8 分发 ／ BF16 汇回）。
  · TensorRT-LLM 博客「Scaling Expert Parallelism in TensorRT-LLM」：大规模 EP 指 EP > 8。
  · LMSYS 2025-06-16 GB200 Part I：纯 NVLink 延迟低，可以关掉 two-batch overlap。
  · Google Cloud 博客「Inside the Ironwood TPU codesigned AI stack」：64 颗一个 cube、3D torus、每颗连 6 个，
    OCS 连到 9,216 颗（144 个 cube），再往外走 DCN。
  · V3 config.json：256 路由专家、moe_intermediate_size 2048、hidden 7168、61 层中前 3 层稠密（MoE 58 层）。

刻意没画
  · NVL72 和 ICI 的带宽数字：这一轮没逐条核对原文，不写。
  · 低延迟 kernel 里 EP8 那一档走不走机内 NVLink：README 说低延迟 kernel 是纯 RDMA，测试拓扑没细写，不标。
  · 「58 层 × 一去一回」是不重叠时的上限，按 EP256 那一档算；线上开了双微批，实际露出来的远小于它。
"""
from topic03_draw import Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE

W = 1400

# ── 专家怎么摆 ───────────────────────────────────────────────────────────
N_EXP, TOPK = 256, 8
EP_SMALL = 8
EP_BIG, NODES, REDUN = 144, 18, 32
PER_SMALL = N_EXP // EP_SMALL
PER_BIG = (N_EXP + REDUN) // EP_BIG
assert PER_SMALL == 32 and PER_BIG == 2 and NODES * 8 == EP_BIG
V3_PARAMS_B, H800_GB = 671, 80
assert V3_PARAMS_B > EP_SMALL * H800_GB             # FP8 每参数 1 字节：671 GB > 8 × 80 GB（本课推导）

# ── DeepEP v1.2.1 低延迟 kernel（µs）────────────────────────────────────
LL = [(8, 77, 114), (16, 118, 195), (32, 155, 273), (64, 173, 314), (128, 192, 369), (256, 194, 360)]
MOE_LAYERS = 61 - 3
assert MOE_LAYERS == 58
NAKED_MS = (LL[-1][1] + LL[-1][2]) * MOE_LAYERS / 1000
assert round(NAKED_MS) == 32, NAKED_MS
assert round(LL[-1][1] / LL[0][1], 1) == 2.5 and round(LL[-1][2] / LL[0][2], 1) == 3.2


def fig_what():
    f = Fig(W, "EP 和大 EP 对照，同一个 DeepSeek-V3，256 个路由专家。左边 EP8：一台机器 8 张卡，每张卡放 32 个专家，"
               "专家之间传 token 走机内 NVLink；按每参数 1 字节，V3 约 671 GB，8 张 80 GB 的卡放不下。"
               "右边是 DeepSeek 线上 decode 的大 EP：18 台机器共 144 张卡，256 个专家加 32 个冗余副本共 288 份，"
               "每张卡只放 2 个路由专家加 1 个共享专家，跨机器走 InfiniBand，每张卡一张网卡。")
    y0 = f.header("EP 和大 EP：同样 256 个专家，铺多宽",
                  "同一个 DeepSeek-V3。左：一台 8 卡机器；右：DeepSeek 线上 decode，18 台机器、144 张卡",
                  [(BL, "路由专家"), (GR, "共享专家"), (OR, "跨机器的 RDMA 网络")])
    PH = 470
    # ── 左：EP8 ──
    LW = 540
    py = f.panel(0, y0, LW, PH, "EP8：一台机器里铺开", BL)
    f.box(18, py + 14, LW - 36, 300, "none", BL, 10, 1.2, dash="6 4")
    f.t(30, py + 38, "一台机器 · 机内 NVLink", BL, True, 15)
    CW, CHH = 112, 118
    for k in range(8):
        cx = 34 + (k % 4) * (CW + 12)
        cy = py + 54 + (k // 4) * (CHH + 12)
        f.box(cx, cy, CW, CHH, "none", LINE, 6, 1)
        f.t(cx + 10, cy + 20, "卡 %d" % k, INK, True, 14)
        for e in range(PER_SMALL):
            f.box(cx + 10 + (e % 8) * 11.5, cy + 32 + (e // 8) * 13, 9, 10, "#e8f0fe", BL, 2, 0.8)
        f.t(cx + 10, cy + CHH - 12, "32 个专家", BL, True, 13)
    f.t(24, py + 344, "每张卡 32 个专家，每步都要把它们读一遍", INK, True, 16)
    f.t(24, py + 372, "按每参数 1 字节，V3 约 671 GB；", GY, size=15)
    f.t(24, py + 396, "8 张 80 GB 的卡加起来 640 GB，放不下", RD, True, 15)
    f._pan = None
    # ── 右：EP144 ──
    RX, RW = LW + 20, W - LW - 20
    py = f.panel(RX, y0, RW, PH, "大 EP（EP144）：铺到 18 台机器", OR)
    NW, NH, GX = 118, 52, 14
    for n in range(6):
        nx = RX + 22 + n * (NW + GX)
        f.line(nx + NW / 2, py + 18 + NH / 2, nx + NW / 2, py + 236, OR, 1.0, arrow=False)
    for n in range(NODES):
        nx = RX + 22 + (n % 6) * (NW + GX)
        ny = py + 18 + (n // 6) * (NH + 16)
        f.box(nx, ny, NW, NH, "#fff", LINE, 6, 1)
        for g in range(8):
            gx_, gy_ = nx + 8 + (g % 4) * 26, ny + 8 + (g // 4) * 19
            f.box(gx_, gy_, 22, 15, "none", GY2, 2, 0.8)
            f.box(gx_ + 3, gy_ + 4, 5, 7, "#e8f0fe", BL, 1, 0.6)
            f.box(gx_ + 9, gy_ + 4, 5, 7, "#e8f0fe", BL, 1, 0.6)
            f.box(gx_ + 15, gy_ + 4, 4, 7, "#e6f4ea", GR, 1, 0.6)
    f.box(RX + 22, py + 236, 6 * NW + 5 * GX, 26, "#fef3e0", OR, 6, 1.2)
    f.t(RX + 22 + (6 * NW + 5 * GX) / 2, py + 254, "InfiniBand 全互连 · 每张卡一张网卡 · GPU 直接发起 RDMA", OR, True, 14, anchor="middle")
    # 放大一张卡
    zx, zy = RX + 30, py + 286
    f.box(zx, zy, 230, 92, "none", INK, 8, 1.4)
    f.t(zx + 14, zy + 24, "放大一张卡", INK, True, 15)
    f.cell(zx + 14, zy + 38, 58, 40, "路由", col=BL, fill="#e8f0fe", size=13)
    f.cell(zx + 80, zy + 38, 58, 40, "路由", col=BL, fill="#e8f0fe", size=13)
    f.cell(zx + 146, zy + 38, 70, 40, "共享", col=GR, fill="#e6f4ea", size=13)
    f.t(zx + 254, zy + 26, "每张卡只放 2 个路由专家 ＋ 1 个共享专家", INK, True, 16)
    f.t(zx + 254, zy + 52, "256 个专家 ＋ 32 份热门专家的副本 ＝ 288 份，", GY, size=15)
    f.t(zx + 254, zy + 74, "288 ÷ 144 ＝ 每卡 2 份", GY, size=15)
    f.t(RX + 22, py + PH - 70, "注意力不跟着切专家：144 张卡各管自己那批请求（数据并行）", PU, True, 15)
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "专家铺出一台机器，就叫大 EP", [
        "DeepSeek 线上：prefill 用 EP32（4 台），decode 用 EP144（18 台）；注意力各卡各管自己的请求。"])
    yb = f.src(yb + 10,
               "DeepSeek open-infra-index Day 6「Inference System Overview」：decode EP144／DP144，每单元 18 个节点、32 个冗余专家，每卡 2 路由 ＋ 1 共享；prefill EP32、4 个节点。DeepSeek-V3 技术报告 §3.4：机内 NVLink，全集群 IB 全互连，decode 用 IBGDA。TensorRT-LLM 博客「Scaling Expert Parallelism」：large-scale EP 指 EP > 8。",
               "本课推导：671B 参数按 FP8 每个 1 字节约 671 GB（不含缩放因子），8 × 80 GB ＝ 640 GB；288 ÷ 144 ＝ 2。卡内专家格子数照实，机器内 8 张卡的摆法是示意。")
    f.save("fig6-bigep-what.svg", yb + 14)


def fig_net():
    f = Fig(W, "左边：DeepEP 低延迟 kernel 的实测，H800、每卡一张 400 Gb InfiniBand 网卡，每批 128 个 token。"
               "卡数从 8 涨到 256，分发从 77 微秒涨到 194 微秒，汇回从 114 微秒涨到 360 微秒；卡数涨了 32 倍，延迟涨两到三倍。"
               "按 EP256 算，每层一去一回、58 个 MoE 层，不藏起来约 32 毫秒。"
               "右边：三种互连域。8 卡机器加 InfiniBand，DeepSeek 的 EP144 就跑在这上面；GB200 NVL72 是 72 张卡一个 NVLink 域，"
               "SGLang 因此能关掉双微批；TPU Ironwood 64 颗一组用 ICI 连成 3D 环面，光交换再连到 9216 颗。")
    y0 = f.header("大 EP 吃网络：卡越多，每趟越慢",
                  "左：DeepEP 低延迟 kernel 实测（H800，每卡一张 400 Gb InfiniBand 网卡；每批 128 个 token）。右：三种互连域",
                  [(OR, "分发：token 送到专家所在的卡"), (RD, "汇回：专家算完送回来")])
    PH = 500
    LW = 760
    py = f.panel(0, y0, LW, PH, "每趟传送多久（微秒）", OR)
    X0, X1, Yb, Yt, YMAX = 90, LW - 60, py + 330, py + 40, 400
    def xp(i):
        return X0 + (X1 - X0) * i / (len(LL) - 1)
    def yp(v):
        return Yb - (Yb - Yt) * v / YMAX
    for v in (0, 100, 200, 300, 400):
        f.line(X0, yp(v), X1, yp(v), "#e8eaed", 0.8, arrow=False)
        f.t(X0 - 12, yp(v) + 5, str(v), GY, size=13, anchor="end")
    for i, (ep, d, c) in enumerate(LL):
        f.t(xp(i), Yb + 24, "EP%d" % ep, INK, True, 14, anchor="middle")
    for col, idx in ((OR, 1), (RD, 2)):
        for i in range(len(LL) - 1):
            f.line(xp(i), yp(LL[i][idx]), xp(i + 1), yp(LL[i + 1][idx]), col, 2.2, arrow=False)
        for i, row in enumerate(LL):
            f.box(xp(i) - 5, yp(row[idx]) - 5, 10, 10, "#fff", col, 5, 2)
            f.t(xp(i), yp(row[idx]) + (-12 if idx == 2 else 24), str(row[idx]), col, True, 13, anchor="middle")
    f.t(X0, Yb + 60, "卡数 ×32：分发 77 → 194（×2.5），汇回 114 → 360（×3.2）", INK, True, 16)
    f.t(X0, Yb + 88, "按 EP256：每层一去一回 × 58 个 MoE 层 ≈ %d 毫秒，不藏起来就全露在每一步里" % round(NAKED_MS), RD, True, 15)
    f.t(X0, Yb + 112, "（本课推导；线上把它藏进计算，靠的是双微批）", GY, size=14)
    f._pan = None
    RX, RW = LW + 20, W - LW - 20
    py = f.panel(RX, y0, RW, PH, "三种互连域", BL)
    cards = [("8 卡机器 ＋ InfiniBand", OR, ["机内 NVLink 约 160 GB/s，机间每卡约 50 GB/s",
                                            "DeepSeek 的 EP144 就跑在这上面：跨 18 台"]),
             ("GB200 NVL72", GR, ["72 张卡在同一个 NVLink 域里",
                                  "SGLang：延迟低到可以关掉双微批"]),
             ("TPU Ironwood", BL, ["64 颗一组，ICI 直连成 3D 环面，每颗连 6 个邻居",
                                   "光交换连到 9,216 颗，再往外走数据中心网络"])]
    cy = py + 16
    for title, col, ls in cards:
        f.box(RX + 18, cy, RW - 36, 128, "none", col, 8, 1.4)
        f.t(RX + 34, cy + 30, title, col, True, 17)
        for j, s in enumerate(ls):
            f.t(RX + 34, cy + 62 + j * 26, s, INK if j else GY, j == 1, 15)
        cy += 152
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "不是非 NVL72 不可", [
        "普通 8 卡机器，只要每张卡一张 RDMA 网卡、全互连、GPU 能直接发起传送，就跑得起；域越大，每趟越快，越不用藏。"])
    yb = f.src(yb + 10,
               "左：DeepEP v1.2.1 README 低延迟 kernel 表（H800 NVLink 约 160 GB/s；CX7 400 Gb/s IB 约 50 GB/s；128 token/批、hidden 7168、top-8、FP8 分发、BF16 汇回）。右：DeepSeek-V3 技术报告 §3.4；LMSYS 2025-06-16「GB200 NVL72 Part I」（pure-NVLink … allows two-batch overlap to be disabled）；Google Cloud「Inside the Ironwood TPU codesigned AI stack」（64-chip cube、3D torus、9,216 颗、DCN）。",
               "本课推导：(194 ＋ 360) µs × 58 层 ≈ 32 ms，是不重叠时的上限；V3 61 层中前 3 层稠密，MoE 58 层（config.json first_k_dense_replace 3）。NVL72、ICI 的带宽没写数字。")
    f.save("fig6-bigep-net.svg", yb + 14)


fig_what()
fig_net()
