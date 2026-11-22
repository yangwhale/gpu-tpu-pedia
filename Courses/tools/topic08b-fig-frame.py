# -*- coding: utf-8 -*-
r"""专题九 · 第一节「一步的时间去哪了」的三张图。

fig8b-budget  回答：一步 5.31 秒里，有用的计算按峰值只要多久？其余的是在算还是在等？
              数字：topic08b_numbers（主机日志第 13–20 步；profile 第 11 步按叶子算子自身时间分两类）。
              刻意没画：「在算」里面再拆（重算、填充、效率），第二节以后讲。
fig8b-hide    回答：一条通信什么时候不花时间？—— 同一层里挑三条：全藏住、一半、大半露出。
              数字：同一份 profile 的一层反向，开始、时长、主核开始等的时刻。
fig8b-lanes   回答：一层反向里三条车道各在干什么？主核在等谁？
              数字：同一层的全部 ≥ 0.3 ms 事件。读图钥匙：主核上的红段 ＝ 停着等，红段结束的时刻 ＝ 它等的那条通信做完。
              刻意没画：事件的张量形状（不需要，也不公开）。
"""
import topic08_figfont  # noqa: F401  字号地板 13.1（须在 topic03_draw 的 wpx 之前；沿用专题八的局部地板）
from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE, LINE2, PU, CY
Fig.BAND_KEEP_LINES = 0   # 图上落点带只留一行粗体结论，其余进图下折叠（同专题八）
import topic08b_numbers as N

W = 1400
KCOL = {"egrad": RD, "ew": OR, "tok": PU, "dense": CY, "small": GY2}
KNAME = {"egrad": "专家梯度 reduce-scatter", "ew": "专家权重 all-gather", "tok": "token 收发",
         "dense": "稠密层权重 all-gather", "small": "小集合、数据重排"}


# ── fig8b-budget ────────────────────────────────────────────────────
def fig_budget():
    f = Fig(W, "两根横条，同一把时间尺。上面一根是理想步时：每个设备每步有用的计算 %s TFLOP，全按峰值 1155 TFLOP/s 跑，"
               "要 %.2f 秒。下面一根是实测：profile 里的一步，设备上 %.2f 秒，其中主核在算 %.2f 秒、停着等通信 %.2f 秒、空闲不到 2 毫秒；"
               "主机日志的平均步时 %.2f 秒，多出来的约 %.0f 毫秒在设备一步之外。实测是理想的 %.1f 倍。"
               % (N.EFF_TF_T, N.IDEAL_S, N.PROF_DEVSTEP_MS / 1000, N.PROF_COMPUTE_MS / 1000, N.PROF_WAIT_MS / 1000,
                  N.CTL_STEP, N.HOST_MINUS_DEV_MS, N.GAP_X))
    y0 = f.header("一步 %s 秒，有用的计算按峰值只要 %s 秒" % (N.CTL_T, N.IDEAL_T),
                  "条长 ＝ 时间（秒），两根同一把尺；理想 ＝ 每设备每步有用的计算 ÷ 每设备峰值",
                  [(GR, "理想：有用的计算全按峰值跑"), (BL, "主核在算"), (RD, "主核停着等通信"), (LINE2, "设备一步之外")])
    PH = 300
    py = f.panel(0, y0, W, PH, "站 13 · 半层底座短文，64 片 v7x（实测）", BL)
    X0, XW, TMAX = 170, 1170, 5.5

    def X(t):
        return X0 + t / TMAX * XW
    # 刻度
    yax = py + PH - 52
    f.line(X0, yax, X(TMAX), yax, GY2, 1.2, arrow=False)
    for i in range(12):
        t = i * 0.5
        f.line(X(t), yax - 5, X(t), yax + 5, GY2, 1.1, arrow=False)
        if i % 2 == 0:
            f.t(X(t), yax + 24, "%g 秒" % t, GY, size=13, anchor="middle")
    # 理想
    y1 = py + 46
    f.t(X0 - 14, y1 + 25, "理想", INK, True, 15, "end")
    f.box(X0, y1, X(N.IDEAL_S) - X0, 40, GR, GR, 4)
    f.t(X(N.IDEAL_S) + 12, y1 + 27, "%.2f 秒 ＝ %s TFLOP ÷ 1155 TFLOP/s" % (N.IDEAL_S, N.EFF_TF_T), INK, True, 15)
    # 实测
    y2 = py + 130
    f.t(X0 - 14, y2 + 25, "实测一步", INK, True, 15, "end")
    c, w = N.PROF_COMPUTE_MS / 1000, N.PROF_WAIT_MS / 1000
    f.box(X0, y2, X(c) - X0, 40, BL, BL, 4)
    f.t((X0 + X(c)) / 2, y2 + 27, "在算 %.2f 秒" % c, "#fff", True, 15, "middle")
    f.box(X(c), y2, X(c + w) - X(c), 40, RD, RD, 4)
    f.t((X(c) + X(c + w)) / 2, y2 + 27, "在等 %.2f 秒" % w, "#fff", True, 15, "middle")
    f.box(X(c + w), y2, X(N.CTL_STEP) - X(c + w), 40, LINE2, GY2, 2)
    f.t(X(N.CTL_STEP), y2 - 10, "主机均值 %.2f 秒" % N.CTL_STEP, GY, size=13, anchor="middle")
    f.t(X0, y2 + 66, "设备上这一步 %.2f 秒（主机均值多出的在设备之外）；空着不到 2 毫秒，四成时间在等通信" % (N.PROF_DEVSTEP_MS / 1000), GY, size=14)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "实测是理想的 %.1f 倍：四成在等；在算的那部分，也是理想的 %.1f 倍" % (N.GAP_X, N.COMPUTE_OVER_IDEAL), [])
    yb = f.src(yb + 10,
               "📌 实测（主机日志）：这一格跑 22 步，取第 13–20 步；步时 %.3f 秒（逐步 5.301–5.318），每设备每步有效运算量 %.1f TFLOP（第二节讲怎么数）。" % (N.CTL_STEP, N.CTL_EFF_TF),
               "⚠️ 本课推导：每设备峰值取 1155 TFLOP/s（MaxText 日志与项目报告的常数；v7x 每芯片公开值 2,307 ÷ 2 ＝ 1153.5，差 0.13%%）；理想 %.3f 秒 ＝ %.1f ÷ 1155；有效 MFU %.2f%% ＝ %.1f ÷ %.3f ÷ 1155。" % (N.IDEAL_S, N.CTL_EFF_TF, N.CTL_MFU * 100, N.CTL_EFF_TF, N.CTL_STEP),
               "📌 实测（profile，第 11 步，一颗芯片上的设备 0）：设备一步 %.1f ms；每个叶子算子的自身时间按两类加总：在算 %.1f、停着等集合通信（含主核自己跑的同步集合）%.1f，剩下 %.1f ms 是空闲。" % (N.PROF_DEVSTEP_MS, N.PROF_COMPUTE_MS, N.PROF_WAIT_MS, N.PROF_IDLE_MS),
               "⚠️ 本课推导：主机均值比设备一步长约 %.0f ms，两个数取的不是同一步，这里只当量级；长出来的部分在设备一步之外（主机派发、步与步之间）。" % N.HOST_MINUS_DEV_MS,
               "⚠️ 这颗芯片这一步注意力负载不算重（滤掉长文档以后各设备接近），所以用它的账代表一步；哪颗芯片代表关键路径，第四节专门讲。")
    f.save("fig8b-budget.svg", yb + 14)


# ── fig8b-hide ──────────────────────────────────────────────────────
def fig_hide():
    f = Fig(W, "同一层反向里挑三条通信，每条一行：灰底是通信在小核上从开始到做完，蓝条是这段时间里主核在算别的，红条是主核停下来等它。"
               "第一行 token 收发 1.6 毫秒，主核一直在算，全部藏住。"
               "第二行专家权重 all-gather 8.1 毫秒，前 2.5 毫秒主核在算，后 5.6 毫秒主核停着等。"
               "第三行专家梯度 reduce-scatter 16.2 毫秒，前 6.1 毫秒藏住，后 10.1 毫秒露出。")
    y0 = f.header("一条通信花不花时间，看主核那时候有没有别的活干",
                  "灰底 ＝ 这条通信在小核上从开始到做完；蓝 ＝ 同一段时间主核在算别的（藏住）；红 ＝ 主核停着等它做完（露出）",
                  [(GY2, "通信在小核上"), (BL, "藏住"), (RD, "露出")])
    PH = 300
    py = f.panel(0, y0, W, PH, "同一层反向里的三条（实测，毫秒）", BL)
    X0, SC = 400, 42.0
    for i, (name, s, d, w) in enumerate(N.HIDE):
        yy = py + 40 + i * 80
        hid = d if w is None else w - s
        exp = 0.0 if w is None else d - hid
        f.t(X0 - 16, yy + 22, name, INK, True, 15, "end")
        f.box(X0, yy, d * SC, 36, LINE2, GY2, 4)
        f.box(X0, yy + 40, hid * SC, 10, BL, BL, 2)
        if exp:
            f.box(X0 + hid * SC, yy + 40, exp * SC, 10, RD, RD, 2)
        tag = ("%.1f 毫秒，全藏住" % d) if not exp else ("%.1f 毫秒：藏住 %.1f，露出 %.1f" % (d, hid, exp))
        f.t(X0 + d * SC + 14, yy + 24, tag, RD if exp > hid else (BL if not exp else INK), True, 15)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "通信本身多长不要紧，要紧的是露出来的那一截：它才算进步时", [])
    yb = f.src(yb + 10,
               "📌 实测（profile，第 11 步，设备 0，反向某一层）：三条通信在小核 1 上的开始时刻与时长、主核停在对应 done 上的时刻与时长，逐条读出；时刻从这一层专家梯度 reduce-scatter 开始算起。",
               "📌 读法同专题七 §2.5：主核走到 done 时通信已做完 ＝ 全藏住；没做完，主核停在 done 上，停的这一段 ＝ 露出。")
    f.save("fig8b-hide.svg", yb + 14)


# ── fig8b-lanes ─────────────────────────────────────────────────────
def fig_lanes():
    f = Fig(W, "一层反向 79.6 毫秒的时间线，三行：主核、小核 0、小核 1。主核浅蓝是在算，色段是停着等，颜色表示等的是哪类通信；"
               "小核上的色块是各类通信。小核 1 一开始就在做这一层的专家梯度 reduce-scatter，要 16.2 毫秒；"
               "主核在第 6.1 毫秒就走到它的完成点，于是停下来等了 10.1 毫秒。之后主核又等了专家权重 all-gather、token 收发和另一条专家梯度 reduce-scatter，"
               "这一层一共等了 27.7 毫秒，在算 51.9 毫秒。小核 1 这一层忙约 49 毫秒，小核 0 只忙约 8 毫秒。")
    y0 = f.header("一层反向，主核和两颗小核：主核在等，可小工并没有忙不过来",
                  "横轴 ＝ 这一层里的时间（毫秒）；主核上的色段 ＝ 停着等，颜色 ＝ 等的是哪类通信，色段右端 ＝ 那条通信刚做完",
                  [(BL, "主核在算"), (RD, "专家梯度 reduce-scatter"), (OR, "专家权重 all-gather"), (PU, "token 收发"),
                   (CY, "稠密层权重 all-gather"), (GY2, "小集合、数据重排")])
    PH = 350
    py = f.panel(0, y0, W, PH, "反向某一层，%.1f 毫秒（实测）" % N.LAYER_MS, BL)
    X0, XW = 150, 1200
    SC = XW / 80.0

    def X(t):
        return X0 + t * SC
    rows = [("主核", py + 50), ("小核 0", py + 150), ("小核 1", py + 220)]
    for lab, yy in rows:
        f.t(X0 - 16, yy + 24, lab, INK, True, 15, "end")
        f.line(X0, yy + 36, X(N.LAYER_MS), yy + 36, LINE, 1, arrow=False)
    # 主核：先整条蓝，再盖红
    yt = rows[0][1]
    f.box(X0, yt, X(N.LAYER_MS) - X0, 36, "#d2e3fc", BL, 3)
    for s, d, k in N.TC_WAITS:
        f.box(X(s), yt, d * SC, 36, KCOL[k], KCOL[k], 2)
    f.t(X(6.1) + 4, yt - 10, "等 10.1", RD, True, 14)
    f.t(X(40.1), yt - 10, "等 5.6", OR, True, 14)
    f.t(X(66.3), yt - 10, "等 3.6", RD, True, 14)
    for lane, yy in ((N.SC0, rows[1][1]), (N.SC1, rows[2][1])):
        for s, d, k, lab in lane:
            f.box(X(s), yy, max(d * SC, 2.5), 36, KCOL[k], KCOL[k], 2)
    # 两处标注
    y1 = rows[2][1]
    f.t(X(0) + 4, y1 + 60, "专家梯度 RS 16.2", RD, True, 14)
    f.t(X(37.6), y1 + 60, "专家权重 AG 8.1 ＋ 4.1", OR, True, 14)
    f.t(X(61.5), y1 + 60, "专家梯度 RS 8.4", RD, True, 14)
    f.line(X(16.2), y1 - 2, X(16.2), yt + 40, RD, 1.4, dash="5 4", arrow=False)
    f.t(X(16.2) + 8, rows[1][1] - 18, "它做完，主核才接着算", RD, True, 14)
    f.t(X(80), rows[1][1] - 18, "这一层：主核在算 %.1f、在等 %.1f；小核 1 忙约 %.0f、小核 0 忙约 %.0f" % (N.LAYER_COMPUTE, N.LAYER_WAIT, N.SC1_BUSY, N.SC0_BUSY), GY, size=14, anchor="end")
    # 刻度
    yax = py + PH - 36
    for t in range(0, 81, 10):
        f.t(X(t), yax, ("%d" % t) if t < 80 else "80 毫秒", GY, size=13, anchor="middle")
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "小核 1 一层只忙六成，主核照样等了 %.1f 毫秒：卡住它的是先后，不是忙闲" % N.LAYER_WAIT, [])
    yb = f.src(yb + 10,
               "📌 实测（profile，第 11 步，设备 0）：窗口从反向某一层专家梯度 reduce-scatter（gate/up 那一组）在小核 1 上开始，到下一层同一事件开始，%.1f ms；只画 ≥ 0.3 ms 的事件，小核忙时因此是下限。" % N.LAYER_MS,
               "📌 实测：反向各层逐层切开取中位数：每层 %.1f ms，主核在算 %.1f、在等 %.1f；小核 0 忙 %.1f、小核 1 忙 %.1f。这一层跟中位数接近。" % (N.MED_LAYER, N.MED_COMPUTE, N.MED_WAIT, N.MED_SC0, N.MED_SC1),
               "📌 专家梯度 reduce-scatter 在这一配置下按 FP32 规约、固定放在小核 1 上；专家权重按 BF16 收集。小集合是注意力、路由这些小参数的梯度规约，每条约 0.3–1 ms，两颗小核一起跑。",
               "⚠️ 本课推导：「卡住它的是先后」指主核在第 6.1 ms 就走到这条 reduce-scatter 的完成点（结果真正要到优化器才用，完成点排在这里是调度器定的），而它在第 16.2 ms 才做完；为什么调度成这样，这一层的数据说明不了，第七节回来讲。")
    f.save("fig8b-lanes.svg", yb + 14)


if __name__ == "__main__":
    fig_budget()
    fig_hide()
    fig_lanes()
