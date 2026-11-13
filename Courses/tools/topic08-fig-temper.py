# -*- coding: utf-8 -*-
r"""专题八 · 第十节的「TPU 脾气」图（第二次迁移 ＝ 混元 4 缩小版，TPU v5p 自比）。

fig8-temper  回答：在 TPU 上改了一样东西，结果会不会变？变了算不算 bug？哪些要处理？
             同一个小尺寸模型、同一份权重和数据，每次只改一样东西，跟一次确认过的 TPU 运行（golden）比，尺子是 GPU 同设置重跑的噪声上限。
             Ⓐ 120 种改动（绝大多数在运行库 0.0.46 上）分六堆（一格一种）：逐位相同／变了但跟 GPU 重跑一样大／真有影响（归成 6 件）／量不出来／故意改坏／没出结果。
             Ⓑ 后两堆以外、有曲线的那些：横轴 300 步曲线最差 r（对数）；蓝点 ＝ 跟重跑一样大；红点 ＝ 真有影响的那 6 件里的实验，
                标出能不能忍；曲线分不开、靠第 1 步对 FP64 抓到的画空心。
             数据：topic08_tpuself.json 的 piles（导出自对齐记录 TPU 自比一节汇总页的六张自动生成的表，实测）。
             2026-11-11 重画：原图按 8 类画 89 个改动（旧快照），对齐记录改成六堆、115 种以后整张换掉。 11-11 晚：对齐记录 119 种（另有 0.0.52 上 18 次重测，画在 fig8-libtpu）。
⛔ 刻意没画：故意改坏的 15 种（另一张图 fig8-sens）；1000 步那一列（正文和出处里写）；多花的算力。
"""
import math

from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE, PU
Fig.BAND_KEEP_LINES = 0   # 锤炼阶段（2026-11-14）：图上落点带只留一行粗体结论，其余进图下折叠
import topic08_numbers as N

W = 1400
P = N.tpuself()["piles"]
OV = P["overview"]                      # [120, 33, 34, 46, 14, 5, 15, 6]（运行库 0.0.46 上的改动；0.0.52 上的重测另算，见 fig8-libtpu）
TOTAL, LONG, N_SAME, N_NOISE, N_ISSUE, N_BLIND, N_BROKEN, N_PEND = OV
assert TOTAL == N_SAME + N_NOISE + N_ISSUE + N_BLIND + N_BROKEN + N_PEND == 120
V52 = "TPU 运行库 0.0.52 上重测"
NOISE46 = [a for a in P["noise"] if a["cat"] != V52]
BLIND46 = [b for b in P["blind"] if "0.0.52" not in b["change"]]
assert (len(NOISE46), len(BLIND46), len(P["broken"])) == (N_NOISE, N_BLIND, N_BROKEN)
ISSUES = [dict(i, exps=[e for e in i["exps"] if "0.0.52" not in e[0] and "只让最后一层" not in e[0]]) for i in P["issues"]]   # 这张图只画 0.0.46；「只让最后一层真舍两次」是查机理用的诊断臂，画在 fig8-rms2r
assert len(ISSUES) == 6 and sum(len(i["exps"]) for i in ISSUES) >= N_ISSUE - 1
NOT_OK = [i for i in ISSUES if i["tolerable"] == "不能忍"]   # 0.0.46 上
assert len(NOT_OK) == 2
PILES = [("逐位相同", N_SAME, GY2), ("变了，跟 GPU 重跑一样大", N_NOISE, BL), ("真有影响（归成 6 件）", N_ISSUE, RD),
         ("动了精度、量不出来", N_BLIND, OR), ("故意改坏（另一张图）", N_BROKEN, PU), ("没出结果", N_PEND, LINE)]
NAME = {"纯数据并行": "纯数据并行", "关掉编译器的省舍入（excess precision）": "关省舍入", "关小矩阵乘改写": "关小矩阵乘改写",
        "RMSNorm 多舍一次": "RMSNorm 多舍一次", "上下文并行": "上下文并行"}


def _short(n):
    return NAME.get(n, n.replace("多路残差混合 不强制舍成 BF16", "多路残差不强制舍 BF16"))


def _tol(i):
    return "能忍" if i["tolerable"].startswith("TPU 运行库 0.0.46 上能忍") else i["tolerable"]


def fig_temper():
    f = Fig(W, "两块，绝大多数是运行库 0.0.46 上的结果（含几次换版本本身）。上：120 种改动一格一种，分六堆：逐位相同 34、变了但跟 GPU 重跑一样大 46、真有影响 14（归成 6 件，其中一个是查机理的诊断臂）、"
               "动了精度但量不出来 5、故意改坏 15、没出结果 6。下：有曲线的那些按 300 步曲线最差 r 画点，对数轴；"
               "蓝点挤在 1 附近。真有影响的 6 件里，不能忍的两件是纯数据并行（8 卡 2.62、16 卡 2.65、不累积 3.69、v7x 不累积 4.06、FP32 开关撞上编译器 bug 那次 611；v7x 2.21 没超过纯噪声最高分，画空心）和关省舍入（13，v7x 上 3.4）；"
               "另外四件曲线都在 1 附近，是第 1 步对 FP64 抓到的：关小矩阵乘改写、RMSNorm 多舍一次、多路残差不强制舍 BF16、上下文并行，都能忍。"
               "RMSNorm 那一行旁注：换到运行库 0.0.52 上不能忍，见下一张图。")
    y0 = f.header("TPU 的脾气：改一样东西，结果变不变、要不要管",
                  "第二次迁移（混元 4 缩小版），绝大多数在运行库 0.0.46 上；每次只改一样，对 TPU 自己的 golden；r 的分母取 GPU 同设置重跑",
                  [(c, "%s %d" % (n, k)) for n, k, c in PILES[:4]] + [(PU, "故意改坏 %d" % N_BROKEN), (GY2, "没出结果 %d" % N_PEND)])
    # Ⓐ 六堆
    PH1 = 150
    py = f.panel(0, y0, W, PH1, "Ⓐ %d 种改动，一格一种" % TOTAL, GY)
    y, cs = py + 22, 22
    k = 0
    for name, n, col in PILES:
        for j in range(n):
            fill = col if col != LINE else "#f1f3f4"
            f.box(24 + (k % 50) * (cs + 3), y + (k // 50) * (cs + 3), cs, cs, fill, col if col != LINE else GY2, 3)
            k += 1
    lab_y = y + 3 * (cs + 3) + 18
    f.t(24, lab_y, "灰 逐位 %d ｜ 蓝 跟重跑一样大 %d ｜ 红 真有影响 %d ｜ 橙 量不出来 %d ｜ 紫 故意改坏 %d ｜ 空 没出结果 %d"
        % (N_SAME, N_NOISE, N_ISSUE, N_BLIND, N_BROKEN, N_PEND), INK, True, 14)
    f._pan = None
    # Ⓑ 曲线 r
    y1 = py + PH1 + 16
    PH2 = 460
    py2 = f.panel(0, y1, W, PH2, "Ⓑ 300 步曲线最差 r（对数）：蓝 ＝ 跟重跑一样大，红 ＝ 真有影响", RD)
    X0, X1 = 230, W - 330
    RL, RHI = 0.7, 20.0

    def X(r):
        r = max(min(r, RHI), RL)
        return X0 + (math.log10(r) - math.log10(RL)) / (math.log10(RHI) - math.log10(RL)) * (X1 - X0)
    ytop, ybot = py2 + 20, py2 + PH2 - 70
    f.box(X(RL), ytop, X(1) - X(RL), ybot - ytop, "#e6f4ea", "#e6f4ea", 0)
    f.box(X(1), ytop, X(N.LOO8_300_MAX) - X(1), ybot - ytop, "#fef7e0", "#fef7e0", 0)
    f.box(X(N.LOO8_300_MAX), ytop, X(RHI) - X(N.LOO8_300_MAX), ybot - ytop, "#fce8e6", "#fce8e6", 0)
    for v in (1, 2, 5, 10, 20):
        f.line(X(v), ytop, X(v), ybot, LINE if v != 1 else GY2, 0.8, arrow=False)
        f.t(X(v), ybot + 20, "%g" % v, GY, size=13, anchor="middle")
    f.line(X(N.LOO8_300_MAX), ytop, X(N.LOO8_300_MAX), ybot, INK, 1.4, dash="4 3", arrow=False)
    f.t(X(N.LOO8_300_MAX) + 6, ytop + 16, "纯噪声最高 %s" % N.LOO8_300_MAX_T, INK, True, 13)
    # 蓝：45 种
    yb_ = ytop + 74
    f.t(20, yb_ + 5, "跟重跑一样大 %d" % N_NOISE, BL, True, 14)
    used = {}
    for a in sorted(NOISE46, key=lambda a: a["r300"]):
        xx = X(a["r300"])
        slot = int(xx // 8)
        kk = used.get(slot, 0)
        used[slot] = kk + 1
        off = ((kk + 1) // 2) * 8 * (1 if kk % 2 else -1)
        f.box(xx - 4, yb_ + off - 4, 8, 8, BL, BL, 4)
    # 红：6 件，一行一件
    for i, iss in enumerate(ISSUES):
        yy = ytop + 160 + i * 34
        bad = _tol(iss) == "不能忍"
        f.t(20, yy + 5, _short(iss["name"]), RD, bad, 14)
        f.t(X0 - 12, yy + 5, "不能忍" if bad else "能忍", RD, bad, 12.5, "end")
        NX = X1 + 16                       # 右侧说明列
        if _tol(iss) != iss["tolerable"]:
            f.t(NX, yy + 5, "运行库 0.0.52 上不能忍（下一张图）", RD, True, 12.5)
        elif all(r <= N.LOO8_300_MAX for _l, r in iss["exps"]):
            f.t(NX, yy + 5, "曲线分不开，第 1 步对 FP64 抓到", GY, size=12.5)
        solid = [r for _l, r in iss["exps"] if r > N.LOO8_300_MAX]
        for (lab, r) in sorted(iss["exps"], key=lambda e: e[1]):
            xx = X(r)
            if r > RHI:
                continue
            if r > N.LOO8_300_MAX:
                f.box(xx - 5.5, yy - 5.5, 11, 11, RD, INK, 5.5)
            else:
                f.box(xx - 5.5, yy - 5.5, 11, 11, "#fff", RD, 5.5, sw=2)
        inside = [r for r in solid if r <= RHI]
        if inside:
            f.t(X(max(inside)) + 9, yy + 5, "%.1f" % max(inside) if max(inside) < 10 else "%.0f" % max(inside), RD, True, 12.5)
        big = [r for r in solid if r > RHI]
        if big:
            f.t(X1 - 4, yy - 9, "→ %.0f（FP32 开关撞上编译器 bug）" % max(big), RD, True, 11.5, "end")
    f.t(X(RHI), ybot + 34, "300 步曲线最差 r（对数）", GY, size=13, anchor="end")
    f._pan = None
    yb = f.band(py2 + PH2 + 20, "ok", "不逐位不等于有 bug：%d 种逐位、%d 种跟重跑一样大；真有影响的 6 件里，不能忍的只有两件（运行库 0.0.46）" % (N_SAME, N_NOISE), [
        "不能忍：纯数据并行（词嵌入梯度在 BF16 里累加，第八节）和关掉省舍入；能忍的四件曲线都在 1 附近，靠第 1 步对 FP64 才抓到，各有开关。",
        "代价：还有 %d 种动了精度、跑到 1000 步也分不开，第 1 步对 FP64 也没抓到 —— 分不开不等于没影响，只是这两把尺子都量不出来。" % N_BLIND,
    ])
    yb = f.src(yb + 10,
               "📌 实测（第二次迁移，小尺寸模型，TPU v5p 8 设备，FSDP 8、每设备微批 1 × 累积 2）：每个改动从 golden 的命令出发只改一样，跑 300 步，对 golden 比 8 项检查（7 条曲线，主 loss 两个窗口）的三个统计；"
               "尺子是 GPU 同设置重跑的噪声上限（300 步 8 份两两 28 对，留一法最大 %s）；第 1 步 205 个梯度张量另对 CPU FP64 定责（离 FP64 比 golden 与 GPU 里远的那个还远 1.2 倍以上，记丢了精度）。"
               % N.LOO8_300_MAX_T,
               "📌 分堆由对齐记录的生成器按每次实验的结果文件自动归类（共 %d 种，其中 %d 种另外跑到 1000 步，尺子 GPU 16 份两两 120 对、留一法最大 %s）。真有影响的 6 件：%s。"
               % (TOTAL, LONG, N.LOO16_1K_MAX_T, "；".join("%s（%s）" % (_short(i["name"]), i["tolerable"].replace("TPU 运行库 ", "运行库 ")) for i in ISSUES)),
               "📌 量不出来的 %d 种：%s。" % (N_BLIND, "、".join("%s（300 步 %.2f，1000 步 %.2f）" % (b["change"], b["r300"], b["r1000"]) for b in BLIND46)),
               "⚠️ 蓝点竖着错开只为不重叠；红点一行一件，同一件的几次实验画在一行。")
    f.save("fig8-temper.svg", yb + 14)


def fig_libtpu():
    """fig8-libtpu 回答：换 TPU 运行库版本，是不是只换了求和顺序？
    Ⓐ 基准运行本身 ＋ 六件，0.0.46 → 0.0.52 各一个 r（哑铃）；Ⓑ 词嵌入 scatter 的排序门槛，芯片 × 版本四格；
    Ⓒ 一条离线预测、上卡核对（v7x 纯数据并行不累积）。数据：topic08_numbers 的 LIB_*（实测）。"""
    f = Fig(W, "两块。左：换 TPU 运行库 0.0.46 到 0.0.52，基准运行自己在噪声内（0.99）；六件每件各对同版本的基准重测，"
               "纯数据并行 2.62 到 2.82、关省舍入 13.3 到 12.6 仍超，关小矩阵乘改写、多路残差两件仍在 1 附近；"
               "RMSNorm 多舍一次从 1.06 跳到 5.50（1000 步 3.68），由能忍变成不能忍；上下文并行从 1.27 降到 0.95。"
               "右：两根数轴。v7x 纯数据并行不累积，每卡 8192 个下标；0.0.46 的门槛 15105 在它右边，不排序；0.0.52 的门槛 7181 在它左边，排序。据此预测 0.0.52 上不再丢精度，上卡核对成立，r 4.06 降到 0.89。")
    y0 = f.header("换运行库，不一定只换了求和顺序",
                  "第二次迁移，TPU 运行库 0.0.46 → 0.0.52：基准运行重建，六件逐件重测，各对同版本的基准比",
                  [(GY2, "空心 ＝ 0.0.46"), (RD, "0.0.52，超出纯噪声最高分"), (BL, "0.0.52，在线内")])
    PH = 470
    LW = 800
    py = f.panel(0, y0, LW, PH, "Ⓐ 300 步曲线最差 r（对数）：结论变了的只有 RMSNorm 一件", RD)
    X0, X1 = 250, LW - 70
    RL, RHI = 0.7, 20.0

    def X(r):
        r = max(min(r, RHI), RL)
        return X0 + (math.log10(r) - math.log10(RL)) / (math.log10(RHI) - math.log10(RL)) * (X1 - X0)
    ytop, ybot = py + 18, py + PH - 62
    f.box(X(RL), ytop, X(1) - X(RL), ybot - ytop, "#e6f4ea", "#e6f4ea", 0)
    f.box(X(1), ytop, X(N.LOO8_300_MAX) - X(1), ybot - ytop, "#fef7e0", "#fef7e0", 0)
    f.box(X(N.LOO8_300_MAX), ytop, X(RHI) - X(N.LOO8_300_MAX), ybot - ytop, "#fce8e6", "#fce8e6", 0)
    for v in (1, 2, 5, 10, 20):
        f.line(X(v), ytop, X(v), ybot, LINE if v != 1 else GY2, 0.8, arrow=False)
        f.t(X(v), ybot + 20, "%g" % v, GY, size=13, anchor="middle")
    f.line(X(N.LOO8_300_MAX), ytop, X(N.LOO8_300_MAX), ybot, INK, 1.4, dash="4 3", arrow=False)
    f.t(X(N.LOO8_300_MAX) + 6, ytop + 14, "纯噪声最高 %s" % N.LOO8_300_MAX_T, INK, True, 12.5)
    rows = [("基准运行自己", N.LIB_GOLDEN_R[0], N.LIB_GOLDEN_R[0], "", "")] + list(N.LIB_SIX)
    note = {"RMSNorm 多舍一次": "主 loss 系统偏高；1000 步 %s；最后一层真舍了两次" % N.LIB_RMS_R1K_T,
            "上下文并行（不开 FP32 logits）": "两次都在线内；那个张量随顺序跳一倍，降了不说明变好",
            "基准运行自己": "0.0.52 对 0.0.46，在噪声内"}
    for i, (name, a, b, va, vb) in enumerate(rows):
        yy = ytop + 34 + i * 52
        worse = b > a * 1.5
        better = b < a / 1.25
        col = RD if worse else (BL if (better or i == 0) else INK)
        f.t(20, yy + 5, name, col if (worse or better) else INK, worse or better or i == 0, 13.5)
        if i == 0:
            f.box(X(b) - 6, yy - 6, 12, 12, BL, INK, 6)
        else:
            f.line(X(a), yy, X(b), yy, col if col != INK else GY2, 2.4 if worse else 1.6, arrow=abs(X(b) - X(a)) > 18)
            f.box(X(a) - 6, yy - 6, 12, 12, "#fff", GY2, 6, sw=2)
            f.box(X(b) - 6, yy - 6, 12, 12, RD if b > N.LOO8_300_MAX else BL, INK, 6)
            lo, hi = (a, b) if a < b else (b, a)
            f.t(X(lo) - 10, yy + 5, "%.2f" % lo, GY if lo == a else (RD if lo > N.LOO8_300_MAX else BL), lo == b, 12, "end")
            f.t(X(hi) + 10, yy + 5, "%.2f" % hi, GY if hi == a else (RD if hi > N.LOO8_300_MAX else BL), hi == b, 12)
        if name in note:
            f.t(X(1.0) + 4 if name != "RMSNorm 多舍一次" else X(0.72), yy + 24, note[name], RD if worse else GY, worse, 12)
    f._pan = None
    RX = LW + 20
    RW = W - RX
    py2 = f.panel(RX, y0, RW, PH, "Ⓑ 一次离线预测：v7x 纯数据并行、不累积", BL)
    t = N.LIB_V7X_GA1
    th46, th52 = N.LIB_THRESH["v7x_046"][(120832, 1024)], N.LIB_THRESH["v7x_052"][(120832, 1024)]
    NIDX = 8192
    f.t(RX + 16, py2 + 30, "每卡一次 scatter 加 8,192 个下标；", INK, True, 14)
    f.t(RX + 16, py2 + 52, "下标数过了门槛（蓝色阴影），后端才自己插排序", INK, True, 14)
    LX0, LX1 = RX + 40, RX + RW - 40
    VMAXL = 18000.0

    def XL(v):
        return LX0 + v / VMAXL * (LX1 - LX0)
    for row, (lab, th, col, verdict) in enumerate([("运行库 0.0.46", th46, GY, "8,192 在门槛左边 → 不排序"),
                                                  ("运行库 0.0.52", th52, BL, "8,192 过了门槛 → 排序")]):
        yy = py2 + 120 + row * 120
        f.t(RX + 16, yy - 32, lab, col, True, 14)
        f.box(XL(th), yy - 10, XL(VMAXL) - XL(th), 20, "#d2e3fc", "none", 0)    # 阴影 ＝ 过了门槛、会排序的范围
        f.line(LX0, yy, LX1, yy, INK, 1.4, arrow=False)
        for v in (0, 5000, 10000, 15000):
            f.line(XL(v), yy - 4, XL(v), yy + 4, INK, 1, arrow=False)
            f.t(XL(v), yy + 20, format(v, ","), GY, size=12, anchor="middle")
        f.line(XL(th), yy - 18, XL(th), yy + 18, col, 2.4, arrow=False)
        f.t(XL(th), yy - 24, "门槛 %s" % format(th, ","), col, True, 13, "middle")
        f.box(XL(NIDX) - 7, yy - 7, 14, 14, RD, RD, 7)
        f.t(XL(NIDX), yy + 42, verdict, RD if col == GY else BL, True, 13, "middle")
    f.t(RX + 16, py2 + PH - 70, "上卡核对：r %.2f → %.2f，词嵌入梯度 R %.2f → %.2f" % (t["r"] + t["R"]), BL, True, 14)
    f.t(RX + 16, py2 + PH - 46, "预测成立：先离线编译看排没排序，再上卡", INK, True, 14)
    f._pan = None
    yb = f.band(y0 + PH + 20, "warn", "换运行库：先建新基准，再把六件重测；结论只对测过的版本对和配置成立", [
        "优化后的程序和后端生成的 kernel 都跟着变：RMSNorm 那件改动带来的程序差别两个版本是同一类，结果却从能忍变成不能忍。离线筛判不了换版本。",
        "做法：开着 RMSNorm 单次舍入（对齐、生产都开）；换版本后看一眼那次 scatter 有没有 sort 和 indices_are_sorted=true。",
    ])
    yb = f.src(yb + 10,
               "📌 实测（第二次迁移，小尺寸，TPU v5p 8 设备，FSDP 8；JAX 0.11.1 不变、只换运行库，这不是官方配套的组合，换官方组合要重测）：新基准换一组机器重跑逐位相同；新基准对旧基准 300 ／ 1000 步 %s，步时 %.1f%%。"
               "六件各按基准的命令只改那一样，对同版本的基准比 8 项检查的三个统计（上下文并行对同版本的普通注意力底），尺子 GPU 同设置重跑 8 份（留一法最大 %s）。"
               % (N.LIB_GOLDEN_R_T, N.LIB_GOLDEN_STEP * 100, N.LOO8_300_MAX_T),
               "📌 第 1 步：纯数据并行词嵌入梯度 R %s → %s，关省舍入 %s → %s，两个版本几乎一样；RMSNorm 多舍一次第 0 层归一化权重的 R %s → %s（第 1 步看不出，偏差是跑起来以后攒的），"
               "前 300 步主 loss 对基准的带符号均值 %s → %s，换一组机器重跑逐位相同。三种修法 0.0.52 上（300 ／ 1000 步）：%s，都跟 GPU 重跑分不开。"
               % (N.LIB_SIX_R["纯数据并行"] + N.LIB_SIX_R["关省舍入"] + N.LIB_RMS_R_STEP1 + N.LIB_RMS_SM_T + (N.LIB_FIX_T,)),
               "📌 门槛表（单算子离线编译二分出的最小插排序下标数）：表 120,832 × 1024 —— v5p 0.0.46 %s、0.0.52 %s，v7x 0.0.46 %s、0.0.52 %s；120,832 × 128 —— v5p %s → %s；480,000 × 128 —— v5p %s → %s；生产宽度 120,832 × 6144 —— v7x 0.0.52 %s，v5p 0.0.52 查到 65,536 都不插。0.0.46 上都是行数 ÷ 8 ＋ 1，0.0.52 上随列宽和芯片变。"
               % tuple(format(x, ",") for x in (N.LIB_THRESH["v5p_046"][(120832, 1024)], N.LIB_THRESH["v5p_052"][(120832, 1024)], N.LIB_THRESH["v7x_046"][(120832, 1024)], N.LIB_THRESH["v7x_052"][(120832, 1024)],
                                                N.LIB_THRESH["v5p_046"][(120832, 128)], N.LIB_THRESH["v5p_052"][(120832, 128)], N.LIB_THRESH["v5p_046"][(480000, 128)], N.LIB_THRESH["v5p_052"][(480000, 128)], N.PROD_TH["v7x_052"])),
               "📌 门槛的测法：单算子离线编译（CPU 上为目标芯片编译，看优化后 HLO 里有没有 sort），单设备、BF16 表，二分最小插排序的下标数，查到 65536。"
               "整张表按 FP32 放得进片上内存的小表（1000 × 1024；v5p 上的 30000 × 128）两个版本都不插。v7x 上部分形状走另一条执行路径（HLO 可见），另算；表里 v7x 只列 1024 以上的列宽。最后一行是生产宽度（隐藏维 6144），0.0.52 上 v5p 查到 65536 都不插。",
               "📌 Ⓑ 上卡：v7x 4 颗芯片（8 个设备），各对同版本的 FSDP 8 比：%s。" % N.LIB_V7X_GA1_T,
               "⚠️ RMSNorm 那件为什么变强，对齐记录已查清，机理在本节折叠里。上下文并行那件降到 0.95，只能说这次没超。")
    f.save("fig8-libtpu.svg", yb + 14)


def fig_rms2r():
    """fig8-rms2r 回答：RMSNorm 多舍一次，为什么换到运行库 0.0.52 就从能忍变成不能忍？
    Ⓐ 最后一层 RMSNorm：0.0.46 融在一颗 kernel 里只舍一次；0.0.52 插了一次布局转换，归一化值写回内存，两次舍入都真发生。
    Ⓑ 为什么两次舍入有偏：γ 离 1 越近（又不等于 1），|h| 被系统地缩小得越多（FP64 里扫 γ）。
    Ⓒ 只动最后一层就能开、关这件事（300 步曲线最差 r）。数据：topic08_numbers 的 RMS_*（实测）。"""
    f = Fig(W, "三块。左上：最后一层 RMSNorm 在两个运行库版本里的样子。0.0.46：归一化、乘 γ、舍到 BF16 在一颗 kernel 里，只舍一次；"
               "0.0.52：编译器给归一化值插了一次布局转换，挡住融合，归一化值先舍成 BF16 写回内存，下一颗 kernel 读回来乘 γ 再舍一次。"
               "右上：FP64 里扫 γ，两次舍入让输出的绝对值平均缩小多少；γ 等于 1 时为 0，γ 是 0.996 时缩小 1.7e-3，离 1 远了就小；混元 4 最后一层 γ 均值 1.001。"
               "下：只动最后一层的五个运行，300 步曲线最差 r：0.0.46 默认 1.06；0.0.46 最后一层强制两次 6.83；0.0.52 默认 5.50；0.0.52 只把最后一层改回一次 0.84；0.0.52 最后一层强制两次 6.97。")
    y0 = f.header("RMSNorm 多舍一次：0.0.52 让最后一层真的舍了两次",
                  "第二次迁移，小尺寸，v5p；默认写法 ＝ 归一化值先舍成 BF16、乘 γ 再舍一次",
                  [(GR, "只舍一次"), (RD, "真舍两次")])
    PH1 = 300
    LW = 760
    py = f.panel(0, y0, LW, PH1, "Ⓐ 最后一层 RMSNorm，两个版本编出来不一样", GR)
    rows = [("运行库 0.0.46", GR, [("归一化", GR), ("× γ", GR), ("舍 BF16", GR)], "一颗 kernel，中间那次被省掉", False),
            ("运行库 0.0.52", RD, [("归一化", RD), ("舍 BF16，写回内存", RD), ("读回 × γ", RD), ("再舍 BF16", RD)], "布局转换挡住了融合，两次都真舍", True)]
    for i, (lab, col, steps, note, split) in enumerate(rows):
        y = py + 34 + i * 128
        f.t(20, y + 30, lab, col, True, 15)
        x = 150
        boxes = []
        for j, (t, c) in enumerate(steps):
            w = 150 if len(t) > 6 else 104
            boxes.append((x, w))
            f.box(x, y + 6, w, 40, "#fff", c, 8, sw=1.6)
            f.t(x + w / 2, y + 31, t, c, True, 13, "middle")
            if j < len(steps) - 1:
                gap = 30 if (split and j == 1) else 14
                f.line(x + w, y + 26, x + w + gap, y + 26, GY, 1.4)
                x += w + gap + 4
        if split:
            x1, w1 = boxes[1]
            x2, _ = boxes[2]
            f.box(x1 - 6, y - 4, w1 + 12, 60, "none", RD, 10, dash="5 4")
            f.box(x2 - 6, y - 4, boxes[3][0] + boxes[3][1] - x2 + 12, 60, "none", RD, 10, dash="5 4")
        else:
            f.box(144, y - 4, x + boxes[-1][1] - 144 + 6 - (x - boxes[-1][0]), 60, "none", GR, 10, dash="5 4")
        f.t(150, y + 76, note, col, True, 14)
    f._pan = None
    RX = LW + 20
    py2 = f.panel(RX, y0, W - RX, PH1, "Ⓑ 为什么两次舍入有偏：γ 贴近 1", OR)
    X0, X1 = RX + 60, W - 30
    YT, YB = py2 + 40, py2 + PH1 - 74
    gmin, gmax, vmin = 0.85, 1.20, -1.8e-3

    def GX(g):
        return X0 + (g - gmin) / (gmax - gmin) * (X1 - X0)

    def VY(v):
        return YT + (v / vmin) * (YB - YT) if v < 0 else YT
    f.line(X0, YT, X1, YT, GY2, 1, arrow=False)
    f.t(X0 - 6, YT + 4, "0", GY, size=11.5, anchor="end")
    f.t(X0 - 6, VY(-1.5e-3) + 4, "−1.5e-3", GY, size=11.5, anchor="end")
    f.line(X0, VY(-1.5e-3), X1, VY(-1.5e-3), LINE, 0.8, arrow=False)
    pts = [(GX(g), VY(c)) for g, c in N.RMS_SWEEP]
    f.path(pts, OR, 2.2, arrow=False)
    for (g, c), (x, y) in zip(N.RMS_SWEEP, pts):
        f.box(x - 4, y - 4, 8, 8, OR, OR, 4)
    f.line(GX(1.0), YT - 6, GX(1.0), YB, INK, 1, dash="4 3", arrow=False)
    f.t(GX(1.0) + 6, YT - 10, "γ ＝ 1：第二次舍入精确，没偏差", INK, True, 12)
    f.t(GX(0.996) - 10, VY(-1.71e-3) + 4, "γ ＝ 0.996：缩小 1.7e-3", OR, True, 12.5, "end")
    f.t(GX(0.86), VY(-0.9e-3), "离 1 远处只剩几e-5，有正有负", GY, size=12)
    for g in (0.9, 1.0, 1.1, 1.2):
        f.t(GX(g), YB + 18, "%g" % g, GY, size=12, anchor="middle")
    f.t(X0, YB + 40, "最后一层的 γ：均值 %.3f，范围 %.2f–%.2f → 输出头的输入整体缩小 %s" % (N.RMS_GAMMA + (N.RMS_SHRINK_T,)), INK, True, 12.5)
    f._pan = None
    y2 = y0 + PH1 + 16
    PH2 = 270
    py3 = f.panel(0, y2, W, PH2, "Ⓒ 只动最后一层，就能打开、关掉这件事（300 步曲线最差 r）", RD)
    BX0, BX1 = 520, W - 180
    rmax = 8.0

    def BXr(v):
        return BX0 + v / rmax * (BX1 - BX0)
    for i, (tag, ver, rest, last, r) in enumerate(N.RMS_AB):
        y = py3 + 22 + i * 40
        bad = r > N.LOO8_300_MAX
        f.t(20, y + 21, "%s　其余各层 %s，最后一层 %s" % (ver, rest, last), RD if bad else GR, bad, 14)
        f.box(BX0, y + 6, max(BXr(r) - BX0, 3), 22, RD if bad else GR, RD if bad else GR, 4)
        f.t(BXr(r) + 8, y + 23, "%.2f" % r, RD if bad else GR, True, 15)
    f.line(BXr(N.LOO8_300_MAX), py3 + 14, BXr(N.LOO8_300_MAX), py3 + 22 + 5 * 40, INK, 1.3, dash="4 3", arrow=False)
    f.t(BXr(N.LOO8_300_MAX) + 6, py3 + 22 + 5 * 40 + 14, "纯噪声最高 %s" % N.LOO8_300_MAX_T, INK, True, 12.5)
    f._pan = None
    yb = f.band(y2 + PH2 + 20, "warn", "机理跟版本无关，版本只决定编译器有没有把中间那次舍入省掉", [
        "同一份权重上前向就高 %s（16 批平均；FP64 模拟第 301 步那批 %s）；第 1 步几乎没有，学会预测后才显（推断）。" % (N.RMS_FROZEN_T, N.RMS_FP64_T),
        "做法：开 RMSNorm 单次舍入（对齐、生产都开）；换版本后先用同一份权重比一次前向。",
    ])
    yb = f.src(yb + 10,
               "📌 实测（第二次迁移，小尺寸，TPU v5p 8 设备，FSDP 8）：整步优化后的程序里，「融合出口就是归一化值转 BF16」的融合只在 0.0.52 默认写法里有 1 处（最后一层）；0.0.46、0.0.52 只把最后一层改回一次、v7x 0.0.52 都是 0 处。"
               "v7x 0.0.52 默认写法没有单独写回，300 步 %.2f（跟 GPU 重跑分不开）。" % N.RMS_V7X_R,
               "📌 同一份权重（0.0.52 基准第 300 步更新后）各算 16 批前向：默认写法 − 单次舍入 %s；0.0.46 上同样的比较 16 批平均 −8e-11、单批都在 ±1.1e-7 内。CPU FP64 模拟第 301 步这一批 %s（TPU 同一批 ＋3.13e-5），第 1 步 ＋2.5e-7（在误差内）。"
               "把两次舍入的变化拆成「沿输出方向的整体缩放」和其余部分，整体缩放解释了全部增量的 %.1f%%。" % (N.RMS_FROZEN_T, N.RMS_FP64_T, N.RMS_SCALE_SHARE * 100),
               "📌 Ⓑ 是推导配实测：FP64 里 400 万个标准正态数，单个 γ，先舍 BF16 再乘 γ 再舍，对比一次舍入，|h| 平均的相对变化。",
               "⚠️ 为什么只有最后一层要紧是推断：它的输出直接进输出头；层内的输出还要过投影、加回残差、再归一化，整体缩放被冲淡，没有单独量。")
    f.save("fig8-rms2r.svg", yb + 14)


fig_temper()
fig_libtpu()
fig_rms2r()
