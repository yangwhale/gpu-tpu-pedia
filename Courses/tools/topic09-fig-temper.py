# -*- coding: utf-8 -*-
r"""专题九 · 第十节的「TPU 脾气」图（第二次迁移 ＝ 混元 4 缩小版，TPU v5p 自比）。

fig9-temper  回答：在 TPU 上改了一样东西，结果会不会变？变了算不算 bug？哪些要处理？
             同一个小尺寸模型、同一份权重和数据，每次只改一样东西，跟一次确认过的 TPU 运行（golden）比，尺子是 GPU 同设置重跑的噪声上限。
             Ⓐ 115 种改动分六堆（一格一种）：逐位相同／变了但跟 GPU 重跑一样大／真有影响（归成 6 件）／量不出来／故意改坏／没出结果。
             Ⓑ 后两堆以外、有曲线的那些：横轴 300 步曲线最差 r（对数）；蓝点 ＝ 跟重跑一样大；红点 ＝ 真有影响的那 6 件里的实验，
                标出能不能忍；曲线分不开、靠第 1 步对 FP64 抓到的画空心。
             数据：topic09_tpuself.json 的 piles（导出自对齐记录 TPU 自比一节 README 的六张自动生成的表，实测）。
             2026-11-11 重画：原图按 8 类画 89 个改动（旧快照），研究线改成六堆、115 种以后整张换掉。
⛔ 刻意没画：故意改坏的 15 种（另一张图 fig9-sens）；1000 步那一列（正文和出处里写）；多花的算力。
"""
import math

from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE, PU
import topic09_numbers as N

W = 1400
P = N.tpuself()["piles"]
OV = P["overview"]                      # [115, 33, 32, 45, 12, 5, 15, 6]
TOTAL, LONG, N_SAME, N_NOISE, N_ISSUE, N_BLIND, N_BROKEN, N_PEND = OV
assert TOTAL == N_SAME + N_NOISE + N_ISSUE + N_BLIND + N_BROKEN + N_PEND == 115
assert (len(P["same"]), len(P["noise"]), len(P["blind"]), len(P["broken"])) == (N_SAME, N_NOISE, N_BLIND, N_BROKEN)
ISSUES = P["issues"]
assert len(ISSUES) == 6 and sum(len(i["exps"]) for i in ISSUES) >= N_ISSUE - 1
NOT_OK = [i for i in ISSUES if i["tolerable"] == "不能忍"]
assert len(NOT_OK) == 2
PILES = [("逐位相同", N_SAME, GY2), ("变了，跟 GPU 重跑一样大", N_NOISE, BL), ("真有影响（归成 6 件）", N_ISSUE, RD),
         ("动了精度、量不出来", N_BLIND, OR), ("故意改坏（另一张图）", N_BROKEN, PU), ("没出结果", N_PEND, LINE)]
NAME = {"纯数据并行": "纯数据并行", "关掉编译器的省舍入（excess precision）": "关省舍入", "关小矩阵乘改写": "关小矩阵乘改写",
        "RMSNorm 多舍一次": "RMSNorm 多舍一次", "上下文并行": "上下文并行"}


def _short(n):
    return NAME.get(n, n.replace("多路残差混合 不强制舍成 BF16", "多路残差不强制舍 BF16"))


def fig_temper():
    f = Fig(W, "两块。上：115 种改动一格一种，分六堆：逐位相同 32、变了但跟 GPU 重跑一样大 45、真有影响 12（归成 6 件）、"
               "动了精度但量不出来 5、故意改坏 15、没出结果 6。下：有曲线的那些按 300 步曲线最差 r 画点，对数轴；"
               "蓝点挤在 1 附近。真有影响的 6 件里，不能忍的两件是纯数据并行（2.6，16 卡 2.7、不累积 3.7）和关省舍入（13，v7x 上 3.4）；"
               "另外四件曲线都在 1 附近，是第 1 步对 FP64 抓到的：关小矩阵乘改写、RMSNorm 多舍一次、多路残差不强制舍 BF16、上下文并行，都能忍。")
    y0 = f.header("TPU 的脾气：改一样东西，结果变不变、要不要管",
                  "第二次迁移（混元 4 缩小版），每次只改一样，对 TPU 自己的 golden；r 的分母取 GPU 同设置重跑",
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
    for a in sorted(P["noise"], key=lambda a: a["r300"]):
        xx = X(a["r300"])
        slot = int(xx // 8)
        kk = used.get(slot, 0)
        used[slot] = kk + 1
        off = ((kk + 1) // 2) * 8 * (1 if kk % 2 else -1)
        f.box(xx - 4, yb_ + off - 4, 8, 8, BL, BL, 4)
    # 红：6 件，一行一件
    for i, iss in enumerate(ISSUES):
        yy = ytop + 160 + i * 34
        bad = iss["tolerable"] == "不能忍"
        f.t(20, yy + 5, _short(iss["name"]), RD if bad else OR, True, 14)
        f.t(X0 - 12, yy + 5, "不能忍" if bad else "能忍", RD if bad else OR, bad, 12.5, "end")
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
    yb = f.band(py2 + PH2 + 20, "ok", "不逐位不等于有 bug：%d 种逐位、%d 种跟重跑一样大；真有影响的 6 件里，不能忍的只有两件" % (N_SAME, N_NOISE), [
        "不能忍：纯数据并行（词嵌入梯度在 BF16 里累加，第八节）和关掉省舍入；能忍的四件曲线都在 1 附近，靠第 1 步对 FP64 才抓到，各有开关。",
        "代价：还有 %d 种动了精度、跑到 1000 步也分不开，第 1 步对 FP64 也没抓到 —— 分不开不等于没影响，只是这两把尺子都量不出来。" % N_BLIND,
    ])
    yb = f.src(yb + 10,
               "📌 实测（第二次迁移，小尺寸模型，TPU v5p 8 设备，FSDP 8、每设备微批 1 × 累积 2）：每个改动从 golden 的命令出发只改一样，跑 300 步，对 golden 比 8 条曲线的三个统计；"
               "尺子是 GPU 同设置重跑的噪声上限（300 步 8 份两两 28 对，留一法最大 %s）；第 1 步 205 个梯度张量另对 CPU FP64 定责（离 FP64 比 golden 与 GPU 里远的那个还远 1.2 倍以上，记丢了精度）。"
               % N.LOO8_300_MAX_T,
               "📌 分堆由对齐记录的生成器按每次实验的结果文件自动归类（共 %d 种，其中 %d 种另外跑到 1000 步，尺子 GPU 16 份两两 120 对、留一法最大 %s）。真有影响的 6 件：%s。"
               % (TOTAL, LONG, N.LOO16_1K_MAX_T, "；".join("%s（%s）" % (_short(i["name"]), i["tolerable"]) for i in ISSUES)),
               "📌 量不出来的 %d 种：%s。" % (N_BLIND, "、".join("%s（300 步 %.2f，1000 步 %.2f）" % (b["change"], b["r300"], b["r1000"]) for b in P["blind"])),
               "⚠️ 蓝点竖着错开只为不重叠；红点一行一件，同一件的几次实验画在一行。")
    f.save("fig9-temper.svg", yb + 14)


fig_temper()
