# -*- coding: utf-8 -*-
r"""专题八 · 第十节的「TPU 脾气」图（第二次迁移 ＝ 混元 4 缩小版，TPU v5p 自比）。

fig8-temper  回答：在 TPU 上改了一样东西，结果会不会变？变了算不算 bug？哪些要处理？
             同一个小尺寸模型、同一份权重和数据，每次只改一样东西，跟一次确认过的 TPU 运行（golden）比，尺子是 GPU 同设置重跑的噪声上限。
             Ⓐ 115 种改动分六堆（一格一种）：逐位相同／变了但跟 GPU 重跑一样大／真有影响（归成 6 件）／量不出来／故意改坏／没出结果。
             Ⓑ 后两堆以外、有曲线的那些：横轴 300 步曲线最差 r（对数）；蓝点 ＝ 跟重跑一样大；红点 ＝ 真有影响的那 6 件里的实验，
                标出能不能忍；曲线分不开、靠第 1 步对 FP64 抓到的画空心。
             数据：topic08_tpuself.json 的 piles（导出自对齐记录 TPU 自比一节汇总页的六张自动生成的表，实测）。
             2026-11-11 重画：原图按 8 类画 89 个改动（旧快照），研究线改成六堆、115 种以后整张换掉。
⛔ 刻意没画：故意改坏的 15 种（另一张图 fig8-sens）；1000 步那一列（正文和出处里写）；多花的算力。
"""
import math

from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE, PU
import topic08_numbers as N

W = 1400
P = N.tpuself()["piles"]
OV = P["overview"]                      # [119, 33, 34, 46, 13, 5, 15, 6]（运行库 0.0.46 上的改动；0.0.52 上的重测另算，见 fig8-libtpu）
TOTAL, LONG, N_SAME, N_NOISE, N_ISSUE, N_BLIND, N_BROKEN, N_PEND = OV
assert TOTAL == N_SAME + N_NOISE + N_ISSUE + N_BLIND + N_BROKEN + N_PEND == 119
V52 = "TPU 运行库 0.0.52 上重测"
NOISE46 = [a for a in P["noise"] if a["cat"] != V52]
assert (len(NOISE46), len(P["blind"]), len(P["broken"])) == (N_NOISE, N_BLIND, N_BROKEN)
ISSUES = [dict(i, exps=[e for e in i["exps"] if "0.0.52" not in e[0]]) for i in P["issues"]]   # 这张图只画 0.0.46
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
    f = Fig(W, "两块，都是运行库 0.0.46 上的结果。上：119 种改动一格一种，分六堆：逐位相同 34、变了但跟 GPU 重跑一样大 46、真有影响 13（归成 6 件）、"
               "动了精度但量不出来 5、故意改坏 15、没出结果 6。下：有曲线的那些按 300 步曲线最差 r 画点，对数轴；"
               "蓝点挤在 1 附近。真有影响的 6 件里，不能忍的两件是纯数据并行（8 卡 2.62、16 卡 2.65、不累积 3.69、FP32 开关撞上编译器 bug 那次 611；v7x 2.21 没超过纯噪声最高分，画空心）和关省舍入（13，v7x 上 3.4）；"
               "另外四件曲线都在 1 附近，是第 1 步对 FP64 抓到的：关小矩阵乘改写、RMSNorm 多舍一次、多路残差不强制舍 BF16、上下文并行，都能忍。"
               "RMSNorm 那一行旁注：换到运行库 0.0.52 上不能忍，见下一张图。")
    y0 = f.header("TPU 的脾气：改一样东西，结果变不变、要不要管",
                  "第二次迁移（混元 4 缩小版），TPU 运行库 0.0.46；每次只改一样，对 TPU 自己的 golden；r 的分母取 GPU 同设置重跑",
                  [(c, "%s %d" % (n, k)) for n, k, c in PILES[:4]])
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
    X0, X1 = 230, W - 40
    RL, RHI = 0.7, 1000.0

    def X(r):
        r = max(min(r, RHI), RL)
        return X0 + (math.log10(r) - math.log10(RL)) / (math.log10(RHI) - math.log10(RL)) * (X1 - X0)
    ytop, ybot = py2 + 20, py2 + PH2 - 70
    f.box(X(RL), ytop, X(1) - X(RL), ybot - ytop, "#e6f4ea", "#e6f4ea", 0)
    f.box(X(1), ytop, X(N.LOO8_300_MAX) - X(1), ybot - ytop, "#fef7e0", "#fef7e0", 0)
    f.box(X(N.LOO8_300_MAX), ytop, X(RHI) - X(N.LOO8_300_MAX), ybot - ytop, "#fce8e6", "#fce8e6", 0)
    for v in (1, 2, 10, 100, 1000):
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
        f.t(20, yy + 5, _short(iss["name"]), RD if bad else OR, True, 14)
        f.t(X0 - 12, yy + 5, "不能忍" if bad else "能忍", RD if bad else OR, bad, 12.5, "end")
        if _tol(iss) != iss["tolerable"]:
            f.t(X(30), yy + 5, "换到运行库 0.0.52：不能忍（下一张图）", RD, True, 12.5)
        last = None
        for (lab, r) in sorted(iss["exps"], key=lambda e: e[1]):
            xx = X(r)
            if r > N.LOO8_300_MAX:
                f.box(xx - 5.5, yy - 5.5, 11, 11, RD, INK, 5.5)
                if last is None or xx - last > 40:
                    f.t(xx + 9, yy + 5, "%g" % (round(r, 1) if r < 100 else round(r)), RD, True, 12.5)
                    last = xx
            else:
                f.box(xx - 5.5, yy - 5.5, 11, 11, "#fff", RD if bad else OR, 5.5, sw=2)
        if all(r <= N.LOO8_300_MAX for _l, r in iss["exps"]):
            f.t(X(3.2), yy + 5, "曲线分不开，第 1 步对 FP64 抓到", OR, size=12.5)
    f.t(X(RHI), ybot + 40, "300 步曲线最差 r（对数）", GY, size=13, anchor="end")
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
               "📌 量不出来的 %d 种：%s。" % (N_BLIND, "、".join("%s（300 步 %.2f，1000 步 %.2f）" % (b["change"], b["r300"], b["r1000"]) for b in P["blind"])),
               "⚠️ 蓝点竖着错开只为不重叠；红点一行一件，同一件的几次实验画在一行。")
    f.save("fig8-temper.svg", yb + 14)


def fig_libtpu():
    """fig8-libtpu 回答：换 TPU 运行库版本，是不是只换了求和顺序？
    Ⓐ 基准运行本身 ＋ 六件，0.0.46 → 0.0.52 各一个 r（哑铃）；Ⓑ 词嵌入 scatter 的排序门槛，芯片 × 版本四格；
    Ⓒ 一条离线预测、上卡核对（v7x 纯数据并行不累积）。数据：topic08_numbers 的 LIB_*（实测）。"""
    f = Fig(W, "三块。左：换 TPU 运行库 0.0.46 到 0.0.52，基准运行自己在噪声内（0.99）；六件每件各对同版本的基准重测，"
               "纯数据并行 2.62 到 2.82、关省舍入 13.3 到 12.6 仍超，关小矩阵乘改写、多路残差两件仍在 1 附近；"
               "RMSNorm 多舍一次从 1.06 跳到 5.50（1000 步 3.68），由能忍变成不能忍；上下文并行从 1.27 降到 0.95。"
               "右上：词嵌入 scatter 插排序的门槛，0.0.46 上是表行数的八分之一，0.0.52 上随列宽和芯片变，v7x 列数 1024 的表降到 7181。"
               "右下：据此预测 v7x 纯数据并行不累积（每卡 8192 个下标）在 0.0.52 上会排序、不再丢精度；上卡核对成立，r 4.06 降到 0.89。")
    y0 = f.header("换运行库，不一定只换了求和顺序",
                  "第二次迁移，TPU 运行库 0.0.46 → 0.0.52：基准运行重建，六件逐件重测，各对同版本的基准比",
                  [(GY2, "空心 ＝ 0.0.46"), (RD, "0.0.52，超出纯噪声最高分"), (BL, "0.0.52，在线内")])
    PH = 470
    LW = 800
    py = f.panel(0, y0, LW, PH, "Ⓐ 300 步曲线最差 r（对数）：一件变强、一件变弱", RD)
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
    note = {"RMSNorm 多舍一次": "主 loss 系统偏高；1000 步 %s；机理在查" % N.LIB_RMS_R1K_T,
            "上下文并行（不开 FP32 logits）": "这次没超；那个张量本来就随顺序跳",
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
    PH1 = 270
    py2 = f.panel(RX, y0, RW, PH1, "Ⓑ 词嵌入那次 scatter：几个下标起插排序", OR)
    cols = [("v5p_046", "v5p 0.0.46"), ("v5p_052", "v5p 0.0.52"), ("v7x_046", "v7x 0.0.46"), ("v7x_052", "v7x 0.0.52")]
    shapes = [(120832, 1024), (120832, 128), (30000, 1024), (480000, 128)]
    cx0, cw, rh = RX + 150, 102, 38
    for j, (_k, lab) in enumerate(cols):
        f.t(cx0 + j * cw + cw / 2, py2 + 26, lab, INK, True, 12.5, "middle")
    for i, (v, d) in enumerate(shapes):
        yy = py2 + 42 + i * rh
        f.t(RX + 14, yy + 24, "%s × %d" % (format(v, ","), d), INK, True, 12.5)
        for j, (k, _lab) in enumerate(cols):
            n = N.LIB_THRESH[k][(v, d)]
            if n is None:
                txt, fill, c = "不插", "#f1f3f4", GY
            elif d == 128 and k.startswith("v7x"):
                txt, fill, c = "SparseCore", "#e8f0fe", BL
            elif n == v // 8 + 1:
                txt, fill, c = format(n, ","), "#fff", INK
            else:
                txt, fill, c = format(n, ","), "#fef7e0", OR
            f.box(cx0 + j * cw + 3, yy + 4, cw - 6, rh - 8, fill, LINE, 4)
            f.t(cx0 + j * cw + cw / 2, yy + 24, txt, c, c != INK, 12, "middle", w=cw - 10)
    f.t(RX + 14, py2 + 42 + 4 * rh + 22, "白：行数 ÷ 8 ＋ 1　黄：不再是 1/8　蓝：v7x 上交给 SparseCore，下标到 2 就排", GY, size=11.5)
    f._pan = None
    y3 = y0 + PH1 + 16
    PH2 = PH - PH1 - 16
    py3 = f.panel(RX, y3, RW, PH2, "Ⓒ 先离线预测，再上卡核对", BL)
    t = N.LIB_V7X_GA1
    f.t(RX + 14, py3 + 26, "v7x 纯数据并行、不累积：每卡 8,192 个下标", INK, True, 13)
    f.t(RX + 14, py3 + 50, "0.0.46：门槛 15,105，不过线 → 不排序", GY, size=12.5)
    f.t(RX + 14, py3 + 72, "0.0.52：门槛 %s，过线 → 排序（离线编译看到）" % format(N.LIB_THRESH["v7x_052"][(120832, 1024)], ","), BL, True, 12.5)
    f.t(RX + 14, py3 + 100, "上卡：r %.2f → %.2f　词嵌入梯度 R %.2f → %.2f" % (t["r"] + t["R"]), BL, True, 13)
    f.t(RX + 14, py3 + 124, "预测成立：问题随版本消失，机理不变", INK, True, 12.5)
    f._pan = None
    yb = f.band(y0 + PH + 20, "warn", "换运行库：先建新基准，再把六件重测；结论只对测过的版本对和配置成立", [
        "同一份 HLO，后端怎么生成 kernel 也会变：RMSNorm 那件在优化后程序里的差别两个版本是同一类，结果却从能忍变成不能忍。离线筛判不了换版本。",
        "做法：开着 RMSNorm 单次舍入（对齐、生产都开）；换版本后看一眼那次 scatter 有没有 sort 和 indices_are_sorted=true。",
    ])
    yb = f.src(yb + 10,
               "📌 实测（第二次迁移，小尺寸，TPU v5p 8 设备，FSDP 8，JAX 不变、只换运行库）：新基准换一组机器重跑逐位相同；新基准对旧基准 300 ／ 1000 步 %s，步时 %.1f%%。"
               "六件各按基准的命令只改那一样，对同版本的基准比 8 项检查的三个统计，尺子 GPU 同设置重跑 8 份（留一法最大 %s）。"
               % (N.LIB_GOLDEN_R_T, N.LIB_GOLDEN_STEP * 100, N.LOO8_300_MAX_T),
               "📌 第 1 步：纯数据并行词嵌入梯度 R %s → %s，关省舍入 %s → %s，两个版本几乎一样；RMSNorm 多舍一次第 0 层归一化权重的 R %s → %s（第 1 步看不出，偏差是跑起来以后攒的），"
               "前 300 步主 loss 对基准的带符号均值 %s → %s，换一组机器重跑逐位相同。三种修法 0.0.52 上（300 ／ 1000 步）：%s，都跟 GPU 重跑分不开。"
               % (N.LIB_SIX_R["纯数据并行"] + N.LIB_SIX_R["关省舍入"] + N.LIB_RMS_R_STEP1 + N.LIB_RMS_SM_T + (N.LIB_FIX_T,)),
               "📌 Ⓑ 单算子离线编译（CPU 上为目标芯片编译，看优化后 HLO 里有没有 sort），单设备、BF16 表，二分最小插排序的下标数，查到 65536。"
               "整张表按 FP32 放得进片上内存的小表（1000 × 1024；v5p 上的 30000 × 128）两个版本都不插。v7x 上列数 128 的表交给 SparseCore 执行（HLO 可见）。",
               "📌 Ⓒ v7x 4 颗芯片（8 个设备），各对同版本的 FSDP 8 比：%s。" % N.LIB_V7X_GA1_T,
               "⚠️ RMSNorm 那件为什么在 0.0.52 上变强，研究线还在查；这里只写实测现象。上下文并行那件变弱，只能说这次没超。")
    f.save("fig8-libtpu.svg", yb + 14)


fig_temper()
fig_libtpu()
