# -*- coding: utf-8 -*-
r"""专题六 · 第四节「批处理与调度：一趟分给更多人」的四张静态图。

fig6-gantt.svg —— 回答：静态 batch 和逐步调度（continuous batching）差在哪？
  · 示意：4 个槽位、8 个请求，各要 3/8/2/5/6/2/7/4 步。两种调度都由下面的代码模拟出来，
    步数和空转格数是算出来的（assert），不是手画的。
  · 静态 batch：凑满 4 个一起开跑，最长的那个跑完才整批返回、才放下一批进来。
  · 逐步调度：每一步结束都检查一次，有槽位空出来就立刻塞进排队的下一个请求（Orca，OSDI'22 的
    iteration-level scheduling）。
  · 刻意没画：新请求进来那一步的 prefill 有多重（下一张图的事）；请求到达时间（假设一开始全在排队）。

fig6-chunk.svg —— 回答：长 prompt 插进来那一步，正在出字的人为什么会卡一下，chunked prefill 怎么摊平？
  · 示意单位：decode 一步计 1 格，一段长 prompt 的 prefill 计 8 格，切成 4 块每块 2 格。
    混合步按相加计时（偏保守：Sarathi 的观察是 decode 搭在 prefill 上几乎不额外花时间）。
  · 算出来的：最长一次字间隔 9 格 → 3 格；12 格里出的字数一样；新请求的首字从第 9 格推迟到第 12 格。
  · 对照：vLLM 文档（V1 默认开 chunked prefill，decode 优先；max_num_batched_tokens 小则 ITL 好、大则 TTFT 好）。
    实例：tpu/tpu-inference/Gemma4-31B/README.md:225-226（--max-num-batched-tokens 16384，256K 切 16 块）。

fig6-conc.svg —— 回答：槽位满了以后再加并发，换来的是什么？
  · gpu/inference/a4x-max/deepseek-v4/SGLANG-V4PRO-RUNBOOK.md:593-605（GB300，SGLang dep8，ISL 8192，
    output ÷ decode GPU；TTFT 只在 600 / 1200 / 1800 三档有记录）。
  · Little 定律核对：同文件 794-801（dep8 9,032 × 8 × 0.060 s ≈ 4,335 个请求在批里；dep16 84,321 × 0.033 ≈ 2,783）。
  · 槽位不白给：同文件 849-858、885-894（ISL 4096：batch 886 → 999，峰值 12,063 → 12,070，TPOT 77.7 → 85.3 ms）。
  · 刻意没画：并发 400 / 900 的 TTFT（runbook 没记）。

fig6-overlap.svg —— 回答：CPU 排下一批的那几毫秒，为什么也要藏起来？
  · SGLang-JAX 博客（lmsys.org/blog/2025-10-29-sglang-jax）：Qwen3-32B 上 prefill 与 decode 之间的空档
    约 12 ms → 38 µs，另一处约 7 ms → 24 µs；做法是 TPU 跑第 N 批时 CPU 先排第 N+1 批。
  · 时间线长短是示意；只有 12 ms 与 38 µs 两个数是原文的。
"""

from topic03_draw import Fig, wpx, MONO_K, BL, OR, GR, RD, PU, GY, INK, GY2, LINE

W = 1400

# ── 甘特图：两种调度现算 ────────────────────────────────────────
REQ = [("A", 3), ("B", 8), ("C", 2), ("D", 5), ("E", 6), ("F", 2), ("G", 7), ("H", 4)]
SLOTS = 4
WORK = sum(n for _, n in REQ)


def static_sched():
    grid, t = [], 0
    for b in range(0, len(REQ), SLOTS):
        batch = REQ[b:b + SLOTS]
        T = max(n for _, n in batch)
        for k in range(T):
            grid.append([(r, k) if k < n else None for r, n in batch])
    return grid


def continuous_sched():
    q, slots, left, grid = list(REQ), [None] * SLOTS, {}, []
    while q or any(slots):
        for i in range(SLOTS):
            if slots[i] is None and q:
                r, n = q.pop(0)
                slots[i], left[r] = r, n
        grid.append([(s, dict(REQ)[s] - left[s]) if s else None for s in slots])
        for i in range(SLOTS):
            if slots[i]:
                left[slots[i]] -= 1
                if left[slots[i]] == 0:
                    slots[i] = None
    return grid


G_ST, G_CB = static_sched(), continuous_sched()
IDLE_ST = sum(c is None for row in G_ST for c in row)
IDLE_CB = sum(c is None for row in G_CB for c in row)
assert WORK == 37 and len(G_ST) == 15 and IDLE_ST == 23
assert len(G_CB) == 12 and IDLE_CB == 11
# 逐步调度的空转全在尾巴上（队列已空），中间没有一格是「有人排队却空着」
_first_idle = min(t for t, row in enumerate(G_CB) if None in row)
assert _first_idle == 8 and all(None not in row for row in G_CB[:8])

# ── chunked prefill（示意单位）─────────────────────────────────
D, P, K = 1, 8, 4
NOCHUNK = [D + P] + [D] * 3          # 长 prompt 一步吞下，之后 3 步纯 decode
CHUNK = [D + P // K] * K              # 切 4 块，每步带一块
assert sum(NOCHUNK) == sum(CHUNK) == 12 and len(NOCHUNK) == len(CHUNK) == 4
assert max(NOCHUNK) == 9 and max(CHUNK) == 3
TTFT_NO, TTFT_CH = NOCHUNK[0], sum(CHUNK)
assert (TTFT_NO, TTFT_CH) == (9, 12)

# ── 并发扫描（V4S:593-605）──────────────────────────────────────
CONC = [(400, 5600, 8777, "58–59", None), (600, 8400, 9168, "59–61", 45),
        (900, 12600, 9280, "62–63", None), (1200, 16800, 9442, "62", 150),
        (1800, 25200, 9745, "60", 253)]
assert all(c * 14 == o for c, o, *_ in CONC)
assert round(CONC[-1][1] / CONC[0][1], 1) == 4.5 and round(CONC[-1][2] / CONC[0][2] - 1, 2) == 0.11
assert round(CONC[-1][4] / CONC[1][4], 1) == 5.6
LITTLE8 = 9032 * 8 * 0.060
LITTLE16 = 84321 * 0.033
assert round(LITTLE8) == 4335 and round(LITTLE16) == 2783
# 槽位不白给（ISL 4096，mem-frac 0.85 → 0.88）
SLOT = dict(batch=(886, 999), peak=(12063, 12070), tpot=(77.7, 85.3))
assert round(SLOT["batch"][1] / SLOT["batch"][0] - 1, 2) == 0.13
assert round((SLOT["peak"][1] / SLOT["peak"][0] - 1) * 100, 2) == 0.06
assert round((SLOT["tpot"][1] / SLOT["tpot"][0] - 1) * 100) == 10

# ── overlap（SGLang-JAX 博客）────────────────────────────────────
GAP_BEFORE_MS, GAP_AFTER_US = 12, 38
assert round(GAP_BEFORE_MS * 1000 / GAP_AFTER_US) == 316

# 八个请求八种颜色，不重复（评审：A 和 E 同色会让人以为是同一个请求）
COLS = {"A": BL, "B": OR, "C": GR, "D": PU, "E": "#00838f", "F": "#d01884", "G": "#795548", "H": "#9e9d24"}
assert len(set(COLS.values())) == len(REQ)


def _gantt(f, x0, y0, grid, CW, RH):
    for t, row in enumerate(grid):
        for s, c in enumerate(row):
            x, y = x0 + t * CW, y0 + s * (RH + 8)
            if c is None:
                f.box(x + 2, y, CW - 4, RH, "#f1f3f4", GY2, 4, dash="4,3")
            else:
                r, k = c
                col = COLS[r]
                f.box(x + 2, y, CW - 4, RH, col, col, 4)
                f.t(x + CW / 2, y + RH / 2 + 5, r if k == 0 else "", "#ffffff", True, 14, "middle")


def fig_gantt():
    f = Fig(W, "四个槽位、八个请求，各要 3、8、2、5、6、2、7、4 步。上：静态 batch，凑满四个一起跑，"
               "最长的 B 跑 8 步，其余三个早就跑完了，槽位空等；整批返回后才放下一批进来，总共 15 步，空转 23 格。"
               "下：逐步调度，每一步结束都看一次，哪个槽位空了就立刻塞进下一个排队的请求，总共 12 步，"
               "空转 11 格全在最后队列空了之后")
    y0 = f.header("每一步都重新排一次　——　<tspan font-weight=\"700\">谁跑完谁让位，槽位不空等</tspan>",
                  "示意：4 个槽位 × 8 个请求，每个请求一种颜色、字母标在它的第一格。每一列是一步 decode",
                  None)
    # 图例手画：实心格有 8 种颜色，单色色块会被读成「只有蓝色是」；虚线格画成虚线框
    ly = y0 - 8
    for i, c in enumerate((BL, OR, GR)):
        f.box(i * 17, ly - 11, 14, 14, c, c, 3)
    f.t(3 * 17 + 4, ly, "彩色实心格 ＝ 这个请求这一步出一个字", GY, size=14)
    lx2 = 3 * 17 + 4 + wpx("彩色实心格 ＝ 这个请求这一步出一个字", 14) * MONO_K + 26
    f.box(lx2, ly - 11, 14, 14, "#fff", GY2, 3, 1.2, dash="3 2")
    f.t(lx2 + 21, ly, "虚线格 ＝ 槽位空转，这一趟白搬", GY, size=14)
    y0 += 20
    CW, RH = 56, 30
    X0 = 220
    PH = 4 * (RH + 8) + 100
    py = f.panel(0, y0, W, PH, "静态 batch：凑满一批，等最慢的跑完", GY2)
    _gantt(f, X0, py + 22, G_ST, CW, RH)
    for s in range(SLOTS):
        f.t(X0 - 16, py + 22 + s * (RH + 8) + 21, "槽位 %d" % (s + 1), GY, size=13.5, anchor="end")
    f.t(X0 + 8 * CW, py + PH - 38, "↑ 第二批到这里才能进来", GY, size=13)
    f.t(X0 + len(G_ST) * CW + 24, py + 50, "%d 步做完" % len(G_ST), INK, True, 17)
    f.t(X0 + len(G_ST) * CW + 24, py + 80, "空转 %d 格" % IDLE_ST, RD, True, 17)
    f.t(X0 + len(G_ST) * CW + 24, py + 106, "C 第 2 步就写完，", GY, size=13)
    f.t(X0 + len(G_ST) * CW + 24, py + 126, "要陪 B 到第 8 步才返回", GY, size=13)
    y1 = py + PH - 30 + 16
    py2 = f.panel(0, y1, W, PH, "逐步调度（Orca）：每一步都看一次，空出来就补人", GR)
    _gantt(f, X0, py2 + 22, G_CB, CW, RH)
    for s in range(SLOTS):
        f.t(X0 - 16, py2 + 22 + s * (RH + 8) + 21, "槽位 %d" % (s + 1), GY, size=13.5, anchor="end")
    f.t(X0 + _first_idle * CW, py2 + PH - 38, "↑ 从这里起队列空了，才开始有空格", GY, size=13)
    f.t(X0 + len(G_ST) * CW + 24, py2 + 50, "%d 步做完" % len(G_CB), INK, True, 17)
    f.t(X0 + len(G_ST) * CW + 24, py2 + 80, "空转 %d 格" % IDLE_CB, GR, True, 17)
    f.t(X0 + len(G_ST) * CW + 24, py2 + 106, "排队的人一来，", GY, size=13)
    f.t(X0 + len(G_ST) * CW + 24, py2 + 126, "下一步就有位子", GY, size=13)
    f._pan = None
    yb = f.src(y1 + PH + 4,
               "逐步调度：Orca（Yu et al., OSDI'22）的 iteration-level scheduling；论文在 GPT-3 175B 上报告"
               "同等延迟下吞吐是 FasterTransformer 的 36.9 倍。",
               "⚠️ 示意：请求长度是编排的，假设一开始全在排队；两种调度的步数、空转格数由本脚本模拟得出。")
    f.save("fig6-gantt.svg", yb + 6)


def fig_chunk():
    f = Fig(W, "示意单位：decode 一步算 1 格，一段长 prompt 的 prefill 算 8 格。上：不切，长 prompt 在某一步里一口吞下，"
               "这一步 9 格长，正在出字的所有人这一下都等了 9 格。下：切成 4 块，每步带一块，每步 3 格，"
               "12 格里出的字一样多，最长字间隔从 9 格降到 3 格；代价是新请求的首字从第 9 格推迟到第 12 格")
    y0 = f.header("长 prompt 切开喂　——　<tspan font-weight=\"700\">别人的字间隔摊平，新人的首字晚一点</tspan>",
                  "示意单位：decode 一步 1 格，长 prompt 的 prefill 8 格。混合步按两者相加计时",
                  [(GR, "decode：老请求各出一个字"), (OR, "prefill：新来的长 prompt")])
    U = 70
    X0 = 260
    PH = 300
    py = f.panel(0, y0, W, PH, "正在出字的人，每个字隔多久", INK)
    for row, (lab, steps, prechunks) in enumerate((("不切：一步吞下", NOCHUNK, [P, 0, 0, 0]),
                                                   ("切成 4 块", CHUNK, [P // K] * K))):
        yy = py + 50 + row * 120
        f.t(X0 - 20, yy + 30, lab, INK, True, 15, "end")
        x = X0
        for k, (st, pc) in enumerate(zip(steps, prechunks)):
            f.box(x + 2, yy, D * U - 4, 46, GR, GR, 4)
            if pc:
                f.box(x + D * U + 2, yy, pc * U - 4, 46, OR, OR, 4)
                f.t(x + D * U + pc * U / 2, yy + 29, "prefill %d 格" % pc, "#ffffff", True, 14, "middle")
            f.line(x + st * U, yy - 10, x + st * U, yy + 56, INK, 1.6, arrow=False)
            x += st * U
        f.t(x + 16, yy + 22, "最长字间隔 %d 格" % max(steps), RD if row == 0 else GR, True, 16)
        f.t(x + 16, yy + 46, "新请求首字在第 %d 格" % (TTFT_NO if row == 0 else TTFT_CH), GY, size=13.5)
    f.t(X0, py + PH - 40, "竖线 ＝ 老请求出一个字。两行都在 12 格里出 4 个字", GY, size=13.5)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "token 预算是一个旋钮，两头各拧一边", [
        "每步塞多少 prefill token（vLLM 的 max_num_batched_tokens）：调小，字间隔稳；调大，首字快。",
        "这只解决一半：prefill 和 decode 还挤在同一批卡上抢算力。彻底分开要靠 PD 分离：两种活放到两批机器上。",
    ])
    yb = f.src(yb + 10,
               "chunked prefill：Sarathi（arXiv 2308.16369）、Sarathi-Serve（arXiv 2403.02310，stall-free 调度）。",
               "vLLM 文档 Optimization 页：V1 默认开 chunked prefill、decode 优先；max_num_batched_tokens 小则 ITL 好、大则 TTFT 好。",
               "本课程作者实测：gpu-tpu-pedia tpu/tpu-inference/Gemma4-31B/README.md:225-226"
               "（TPU v7x 上 --max-num-batched-tokens 16384，256K prompt 切成 16 块 16K）。",
               "⚠️ 示意：格子数是编排的。混合步按相加计时偏保守，Sarathi 的观察是 decode 搭在 prefill 上几乎不额外花时间。")
    f.save("fig6-chunk.svg", yb + 14)


QUEUE = [round(c[4] / 23) for c in CONC if c[4] is not None]   # 门口排队的人数，示意，与 TTFT 成比例
assert QUEUE == [2, 7, 11]


def fig_conc():
    f = Fig(W, "像一家座位坐满的餐厅：座位就是批里的槽位，已经坐满；翻台速度不变，每张卡的吞吐就几乎不动；"
               "再来的客人只能在门口排队，队越排越长。实测：GB300 上 SGLang 跑 DeepSeek-V4-Pro，8K 进 1K 出，decode 8 张卡。"
               "每路并发从 400 拉到 1,800，总负载涨 4.5 倍，每张 decode 卡的吞吐只从 8,777 涨到 9,745，涨 11%；"
               "TPOT 一直钉在 58 到 63 毫秒。首字等待却从 45 秒涨到 150 秒、253 秒")
    y0 = f.header("槽位满了再加并发　——　<tspan font-weight=\"700\">吞吐几乎不动，多出来的人全在排队</tspan>",
                  "GB300 · SGLang · DeepSeek-V4-Pro · dep8 decode · 8K 进 1K 出 · 吞吐按 output ÷ decode GPU",
                  [(GR, "每张 decode 卡的吞吐"), (RD, "TTFT 中位数（含排队）")])
    # ① 餐厅：座位满了，翻台速度不变，门口排队变长
    PH0 = 30 + 190
    p0 = f.panel(0, y0, W, PH0, "像一家坐满的餐厅", INK)
    RX, RW = 30, 470
    f.box(RX, p0 + 24, RW, 120, "none", INK, 8, sw=2)
    for r in range(2):
        for c in range(7):
            tx, ty = RX + 22 + c * 64, p0 + 40 + r * 52
            f.box(tx + 6, ty + 30, 42, 8, GY2, GY2, 2)
            f.icon("person", tx + 13, ty - 2, 28, 32, col=GR, tint=GR)
    f.t(RX + RW / 2, p0 + 172, "座位 ＝ 槽位：全坐满了，翻台速度不变", GR, True, 16, "middle")
    DX = RX + RW
    f.box(DX - 3, p0 + 84, 6, 60, "#ffffff", "#ffffff", 0)       # 门口
    QX0 = DX + 150
    ROWS = [c for c in CONC if c[4] is not None]
    for qi, (n, row) in enumerate(zip(QUEUE, ROWS)):
        qy = p0 + 22 + qi * 42
        f.t(QX0 - 16, qy + 26, "并发 %s" % format(row[0], ","), INK, True, 15, "end")
        for k in range(n):
            f.icon("person", QX0 + k * 40, qy, 28, 32, col=RD, tint="#fce8e6")
    f.line(QX0 - 110, p0 + 106, DX + 10, p0 + 106, GY2, 1.6)
    f.t(QX0, p0 + 172, "门口的队越排越长 ＝ 首字等待", RD, True, 16)
    # ② 实测：吞吐 vs 首字等待
    HW = (W - 20) // 2
    PH = 30 + 340
    y1 = y0 + PH0 + 16
    py = f.panel(0, y1, HW, PH, "翻台速度：总负载 ×4.5，吞吐 +11%", GR)
    BX0, BW_, BH = 40, 110, 230
    base = py + 40 + BH
    for i, (c, o, thr, tpot, _) in enumerate(CONC):
        x = BX0 + i * (BW_ + 22)
        h = BH * thr / 10000
        f.box(x, base - h, BW_, h, GR, GR, 4)
        f.t(x + BW_ / 2, base - h - 10, format(thr, ","), GR, True, 16, "middle")
        f.t(x + BW_ / 2, base + 24, "并发 %s" % format(c, ","), INK, True, 15, "middle")
        f.t(x + BW_ / 2, base + 48, "TPOT %s ms" % tpot, GY, size=14, anchor="middle")
    f.panel(HW + 20, y1, HW, PH, "门口排队：首字等待 45 秒 → 253 秒", RD)
    X2 = HW + 60
    for i, (c, o, thr, tpot, ttft) in enumerate(CONC):
        x = X2 + i * (BW_ + 22)
        if ttft is None:
            f.t(x + BW_ / 2, base - 10, "未记录", GY2, size=14, anchor="middle")
        else:
            h = BH * ttft / 270
            f.box(x, base - h, BW_, h, RD, RD, 4)
            f.t(x + BW_ / 2, base - h - 10, "%d 秒" % ttft, RD, True, 16, "middle")
        f.t(x + BW_ / 2, base + 24, "并发 %s" % format(c, ","), INK, True, 15, "middle")
    f._pan = None
    yb = f.src(y1 + PH + 4,
               "本课程作者实测：gpu-tpu-pedia gpu/inference/a4x-max/deepseek-v4/SGLANG-V4PRO-RUNBOOK.md:593-609"
               "（并发 ＝ 每路 conc × 14 个 frontend；ISL 8192、OSL 1024；--max-running-requests 8192；TTFT 只记了三档）。",
               "⚠️ 餐厅里的人数是示意：14 个座位、门口排的人数按 TTFT 的比例画（45 : 150 : 253 秒 → 2 : 7 : 11 人，三档各画一排）。",
               "Little 定律核对与「槽位不白给」实验（batch 886 → 999、TPOT 77.7 → 85.3 ms）见正文；出处同文件 794-801、849-858、885-894。")
    f.save("fig6-conc.svg", yb + 6)


def fig_overlap():
    f = Fig(W, "上：串行。CPU 排好第 N 批交给 TPU，TPU 算完再回来等 CPU 排第 N+1 批，每一步中间都夹着一段 TPU 空闲。"
               "下：重叠。TPU 算第 N 批的同时，CPU 已经在排第 N+1 批，TPU 一步接一步几乎不停。"
               "SGLang-JAX 在 Qwen3-32B 上把 prefill 和 decode 之间的空档从约 12 毫秒压到 38 微秒")
    y0 = f.header("CPU 排下一批，别让加速器干等　——　<tspan font-weight=\"700\">12 ms → 38 µs</tspan>",
                  "示意时间线。只有 12 ms、38 µs 两个数是原文的（SGLang-JAX，Qwen3-32B，prefill 与 decode 之间的空档）",
                  [(BL, "CPU：调度、拼 batch、准备输入"), (OR, "TPU：跑这一批")])
    PH = 330
    py = f.panel(0, y0, W, PH, "两条泳道：CPU 和 TPU", INK)
    X0, U = 200, 40
    CPU, TPU = 2, 6
    for row, lab in enumerate(("串行", "重叠")):
        yc = py + 40 + row * 140
        f.t(30, yc + 50, lab, INK, True, 16)
        f.t(X0 - 14, yc + 22, "CPU", BL, True, 13.5, "end")
        f.t(X0 - 14, yc + 72, "TPU", OR, True, 13.5, "end")
        if row == 0:
            x = X0
            for n in range(3):
                f.box(x, yc, CPU * U - 4, 34, BL, BL, 4)
                f.t(x + CPU * U / 2, yc + 23, "排 N+%d" % n if n else "排 N", "#ffffff", True, 13, "middle")
                x += CPU * U
                f.box(x, yc + 50, TPU * U - 4, 34, OR, OR, 4)
                f.t(x + TPU * U / 2, yc + 73, "算 N+%d" % n if n else "算 N", "#ffffff", True, 13, "middle")
                if n < 2:
                    f.box(x + TPU * U, yc + 50, CPU * U - 4, 34, "#fce8e6", RD, 4, dash="4,3")
                    f.t(x + TPU * U + CPU * U / 2, yc + 73, "空等", RD, True, 13, "middle")
                x += TPU * U
        else:
            f.box(X0, yc, CPU * U - 4, 34, BL, BL, 4)
            f.t(X0 + CPU * U / 2, yc + 23, "排 N", "#ffffff", True, 13, "middle")
            for n in range(3):
                xt = X0 + CPU * U + n * TPU * U
                f.box(xt, yc + 50, TPU * U - 6, 34, OR, OR, 4)
                f.t(xt + TPU * U / 2, yc + 73, "算 N+%d" % n if n else "算 N", "#ffffff", True, 13, "middle")
                if n < 2:
                    f.box(xt, yc, CPU * U - 4, 34, BL, BL, 4)
                    f.t(xt + CPU * U / 2, yc + 23, "排 N+%d" % (n + 1), "#ffffff", True, 13, "middle")
    f.t(X0 + 3 * (CPU + TPU) * U + 20, py + 112, "空档 ≈ 12 ms", RD, True, 16)
    f.t(X0 + 3 * (CPU + TPU) * U + 20, py + 252, "空档 ≈ 38 µs", GR, True, 16)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "decode 一步越短，CPU 那几毫秒占得越多", [
        "批开得越大，CPU 要拼的东西越多；TPU 一步本来才几十毫秒，串着走就白白空掉一截。",
        "SGLang-JAX 的做法是 TPU 跑第 N 批时 CPU 先排第 N+1 批，并提前为不同 batch 大小编译好计算图。",
    ])
    yb = f.src(yb + 10,
               "SGLang-JAX 博客（lmsys.org/blog/2025-10-29-sglang-jax）原文：For Qwen/Qwen3-32B, we reduced the time gap "
               "between prefilling and decoding from approximately 12ms to 38us, and from approximately 7ms to 24us。"
               "同文：The scheduler pre-compiles Jax computation graphs for different batch sizes。",
               "⚠️ 示意：格子长短是编排的，CPU 与 TPU 段的比例不对应任何实测。")
    f.save("fig6-overlap.svg", yb + 14)


fig_gantt()
fig_chunk()
fig_conc()
fig_overlap()
