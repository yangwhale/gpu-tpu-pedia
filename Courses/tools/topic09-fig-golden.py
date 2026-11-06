# -*- coding: utf-8 -*-
r"""专题九 · 第九、十节的四张图。

fig9-golden   回答：最终那份 golden 是怎么一步步来的？每一步加了什么、换来了什么？
              数字：同一个被测（v5p，FP32 累加开关打开前的配置）对三份 golden 的参数范数带符号均值 ÷ 同一把尺子 N（实测）。
fig9-decomp   回答：TPU 对 GPU 生产 golden 的参数范数差是谁的？—— 两个生产程序只差 GPU 输出头 dgrad 的长累加；对 ④ 的那截是三笔相加（实测，11-06 版）。
fig9-twolayer 回答：golden 自己有系统偏差时怎么验收？—— 分两关：第一关对精确轨迹必须过，第二关对生产 golden 写成已解释差异。
fig9-regress  回答：验收过了以后，每天合进来的代码怎么守？—— 先要求逐位；不逐位按 PR 的数值声明分流。
"""
from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE, LINE2, PU
import topic09_numbers as N
from topic09_numbers import tx

W = 1400
NOISE = N.GPU_RERUN_PN
# R3：颜色 ＝ 全课约定（GPU 生产 golden 深灰、精确轨迹紫）；卡片上不提前用「第一关／第二关」，关卡名留到 fig9-twolayer 再出
STEPS = [("① 起点", "框架默认配置", ["BF16，不重算", "TF32 开着"], None, GY2, "归因参照"),
         ("② 生产近似", "＋ 整层重算 ＋ 路由回放", ["模仿生产开重算", "跟①比：各项都在 GPU 噪声内"], 2.93e-6, GY, "生产 golden：差要写清来由"),
         ("③ 只修一处", "＋ 输出头 dgrad 换成 FP32", ["解释 loss 偏移的 73%", "参数范数 ＋0.0124", "定案前的尺子取自这组"], 1.25e-6, OR, "归因证据"),
         ("④ 精确轨迹", "＋ BF16 权重、全程 FP32", ["真关 TF32；可输出头", "长求和一次累加，有偏"], 4.36e-7, PU, "定案前的基线"),
         ("⑤ 最终 golden", "② ＋ dgrad 分段累加", ["只改这一处", "GPU 步时多六成", "尺子：自己 10 份 45 对"], None, GR, "第一关的基线（定案）")]
R = [None if st[3] is None else st[3] / NOISE for st in STEPS]
R[4] = N.FINAL[0][4]                     # ⑤ 用它自己的 45 对尺子，v5p 当前配置
assert abs(R[1] - 23.6) < 0.1 and abs(R[2] - 10.1) < 0.1 and abs(R[3] - 3.52) < 0.01, R


def fig_golden():
    f = Fig(W, "四张卡片从左到右：第一份是起点，框架默认配置；第二份加上整层重算和路由回放，跟生产一样；第三份再把输出头 dgrad 换成 FP32 乘法（一次累加）；"
               "第四份再改成 BF16 权重、全程 FP32 计算，叫精确轨迹，是定案前的基线；后来发现它输出头那次长求和仍有偏。第五份是最终 golden：回到第二份，只把输出头 dgrad 改成分段累加。"
               "卡片下面是同一个 TPU 被测对它的参数范数差，按同一把尺子 N 算倍数：对第二份 24 倍，对第三份 10 倍，对第四份 3.5 倍；第五份换成它自己 45 对的尺子，当前 TPU 是 0.50 倍，过。")
    y0 = f.header("golden 是一步步改出来的　——　<tspan font-weight=\"700\">先对齐超参和数据，再每一步只加一样东西</tspan>",
                  "卡片 ＝ 一份 golden（GPU 上跑的标准答案）；下面的柱子 ＝ 同一个 TPU 被测对它的参数范数差是噪声的几倍（同一把尺子）",
                  [(GY2, "起点"), (GY, "生产近似"), (OR, "只修一处"), (PU, "精确轨迹"), (GR, "最终 golden"), (RD, "不过：超过 2×N")])
    PH = 420
    py = f.panel(0, y0, W, PH, "混元 3 的五份 golden（实测）", BL)
    CW, GAP = 252, 20
    for i, ((name, add, lines, v, col, role), r) in enumerate(zip(STEPS, R)):
        x = 30 + i * (CW + GAP)
        f.box(x, py + 24, CW, 192, "#fff", col, 10, sw=2)
        f.box(x, py + 24, CW, 6, col, col, 3)
        f.t(x + 18, py + 66, name, col, True, 20, big=True)
        f.t(x + 18, py + 96, add, INK, True, 14)
        for k, ln in enumerate(lines):
            f.t(x + 18, py + 124 + k * 22, ln, OR if k == 2 else GY, k == 2, 13)
        f.t(x + 18, py + 204, role, col, True, 13)
        if i < 4:
            f.line(x + CW + 6, py + 108, x + CW + GAP - 6, py + 108, GY2, 2)
        if r is not None:
            L = min(r / 24.0 * 216, 216)
            f.box(x + 18, py + 250, L, 30, RD if r > 2 else GR, RD if r > 2 else GR, 4)
            # ⭐ R3：每根柱子补一条 2×N 的参考短线，「3.5 倍还不过」一眼能看出来
            x2 = x + 18 + 2 / 24.0 * 216
            f.line(x2, py + 242, x2, py + 288, INK, 2, arrow=False)
            f.t(x2, py + 236, "2×N", INK, True, 12, "middle")
            f.t(x + 18, py + 306, ("TPU 对它：%.2f × N" if r < 1 else "TPU 对它：%.1f × N") % r, RD if r > 2 else GR, True, 15)
            if i == 3:
                f.t(x + 18, py + 328, "被测：开关打开前那一版 TPU", GY, size=12)
            if i == 4:
                f.t(x + 18, py + 328, "被测：当前 TPU；尺子换成它自己的", GY, size=12)
        else:
            f.t(x + 18, py + 270, "（不重算，跟生产配置不一样，不拿来验收）", GY, size=13)
    f.t(30, py + 350, "柱子：TPU 被测的参数范数带符号均值 ÷ N（②–④ 的 N ＝ ③ 那组 5 份两两 10 对的最大值；⑤ 用它自己 10 份 45 对）；竖线 ＝ 2×N", GY, size=13)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "每往精确值靠一步，TPU 的「偏差」就缩一截：24 倍 → 10 倍 → 3.5 倍；最终 golden 上 0.50 倍", [
        "缩掉的部分都查清了是 GPU golden 自己的偏差；④ 上剩的那截也拆清了：④ 自己少漂一笔、TPU 的 dlogits 舍入少漂一笔，两笔相减。最后只修 GPU 那一次长求和，两边就落进同一团噪声。",
        "代价：每一份 golden 都要单独跑满 1000 步、存好数据；改了什么、为什么改、证据在哪，都要进一张出处总表。",
    ])
    yb = f.src(yb + 10,
               "📌 实测：被测是 v5p 上生产配置 1000 步（FP32 累加开关打开之前的那一版），param_norm 带符号均值对 ② 2.93e-6、对 ③ 1.25e-6、对 ④ 4.36e-7；N ＝ 1.24e-7。开关打开后的当前配置对 ④ 是 4.27e-7（3.45 倍）。",
               "📌 实测：③ − ② 在第 301–1000 步 loss 上是 −9.8e-5，占 TPU 对 ② 偏移（−1.34e-4）的 73%；第 1000 步参数范数 ＋1.24e-2（③ 的 dgrad 用的是 cuBLAS FP32 一次累加，本身也偏，所以这只是 BF16 那一截）。")
    f.save("fig9-golden.svg", yb + 14)


def fig_decomp():
    f = Fig(W, "上面一条数轴是第 1000 步参数范数对 GPU 生产 golden 的差：GPU 精确轨迹 ＋0.0186，GPU 生产只把输出头 dgrad 改成分段累加的那份 ＋0.0207，TPU 生产 ＋0.0209，真精确 ＋0.0274。"
               "从 0 到 ＋0.0207 是 GPU 输出头 dgrad 长累加的偏差；从 ＋0.0207 到 ＋0.0274 是两边都有的 dlogits 舍入。"
               "下面是 TPU 生产对精确轨迹那 ＋0.00235 的三笔：精确轨迹自己少漂 ＋0.0087，TPU 的 dlogits 舍入 −0.0068，其余 ＋0.0005。")
    y0 = f.header("新的账：两个生产程序只差 GPU 那一次长累加",
                  "上：各条运行第 1000 步落在哪（实测）；下：TPU 对精确轨迹那一小截的三笔（实测，按定义加起来正好闭合）",
                  [(GY, "GPU 生产"), (PU, "精确轨迹 ／ 真精确"), (OR, "GPU 生产＋分段累加"), (BL, "TPU 生产"), (RD, "GPU 长累加偏差"), (LINE2, "两边共有")])
    PH = 300
    py = f.panel(0, y0, W, PH, "第 1000 步参数范数，对 GPU 生产 golden 5 份均值的差（生产近似配置）", BL)
    X0, BW, VMAX = 60, 1260, 0.029
    X = lambda v: X0 + v / VMAX * BW
    ya = py + 120
    f.line(X0, ya, X0 + BW, ya, INK, 1.5, arrow=False)
    for v in (0, 0.01, 0.02):
        f.line(X(v), ya - 5, X(v), ya + 5, INK, 1.2, arrow=False)
        f.t(X(v), ya + 22, "0" if v == 0 else "＋%.2f" % v, GY, size=12, anchor="middle")
    pts = [(0.0, "GPU 生产", GY, -1), (N.PN_FIX1, "④ 精确轨迹 ＋%.4f" % N.PN_FIX1, PU, -1),
           (N.PN_PRODCHUNK, "⑤ 生产＋分段累加 ＋%.4f" % N.PN_PRODCHUNK, OR, -2), (N.PN_TPU_PROD, "TPU 生产 ＋%.4f" % N.PN_TPU_PROD, BL, -3),
           (N.PN_TRUE, "⑥ 真精确 ＋%.4f" % N.PN_TRUE, PU, -1)]
    for v, lab, c, row in pts:
        x = X(v)
        f.box(x - 7, ya - 7, 14, 14, c, c, 7)
        yl = ya + 18 * row - 14
        f.line(x, ya - 9, x, yl + 4, c, 1, arrow=False)
        anc = "end" if v > 0.024 else ("middle" if v < 0.001 else "end")
        if v < 0.001:
            anc = "start"
        f.t(x + (4 if anc == "start" else -4), yl, lab, c, True, 13, anchor=anc)
    yb1 = ya + 50
    f.box(X(0), yb1, X(N.PN_PRODCHUNK) - X(0), 22, RD, RD, 3)
    f.t(X(0) + 8, yb1 + 16, "GPU 输出头 dgrad 的长累加偏差 ＋%.4f（BF16 张量核 ＋ cuBLAS FP32 一次累加）" % N.LADDER_DGRAD, "#fff", True, 13)
    f.box(X(N.PN_PRODCHUNK), yb1, X(N.PN_TRUE) - X(N.PN_PRODCHUNK), 22, LINE2, GY2, 3)
    f.t(X(N.PN_TRUE), yb1 + 46, "两边共有：输出头反向的 dlogits 舍成 BF16 ＋%.4f" % N.LADDER_SHARED, GY, True, 13, anchor="end")
    f.t(X0, py + PH - 40, "TPU 生产和 ⑤ 几乎落在同一点；真精确 ＝ ⑥，TPU 自己的精确轨迹也落在这里", GY, size=12)
    f._pan = None
    PH2 = 232
    py2 = f.panel(0, py + PH + 16, W, PH2, "TPU 生产对 ④ 的 ＋%.5f 是三笔相加（第 1000 步）" % N.TPU_VS_FIX1, BL)
    K = 760 / 0.0095
    xs = X0 + 120
    ytop = py2 + 50
    steps = [(N.B_GPUEXACT, RD, "① ④ 自己少漂：cuBLAS FP32 一次累加整个词表（GPU 一侧）", "%.2e"),
             (N.B_DLOGITS, BL, "② TPU 输出头反向的 dlogits 舍成 BF16（TPU 一侧）", "%.2e"),
             (N.B_REST, LINE2, "③ 其余 BF16 存储与逐元素", "%.1e")]
    cur = 0.0
    for k, (v, c, lab, sp) in enumerate(steps):
        a, b = cur, cur + v
        x1, x2 = xs + min(a, b) * K, xs + max(a, b) * K
        yy = ytop + k * 40
        f.box(x1, yy, max(x2 - x1, 3), 24, c, GY2 if c == LINE2 else c, 3)
        if k < 2:
            f.t(x1 + 10, yy + 17, "%s　%s" % (lab, tx(v, sp, plus=True)), "#fff", True, 13)
        else:
            f.t(x2 + 10, yy + 17, "%s　%s" % (lab, tx(v, sp, plus=True)), GY, True, 13)
        cur = b
    f.line(xs + cur * K, ytop + 84, xs + cur * K, ytop + 128, INK, 1.5, dash="4,3", arrow=False)
    f.t(xs + cur * K + 8, ytop + 140, "合计 ％s，正好等于实测" .replace("％s", tx(N.TPU_VS_FIX1)), INK, True, 13)
    f._pan = None
    yb = f.band(py2 + PH2 + 20, "ok", "两个生产程序之间，唯一测得出的差是 GPU 输出头 dgrad 的长累加", [
        "GPU 生产只修这一处，TPU 生产就跟它几乎重合（按 ③ 那把尺子 v5p 五项全过）；剩下离真精确的那截是两边都有的 dlogits 舍入，不算两边的差。",
        "第一关对 ④ 的那点差，是 ① 和 ② 两笔方向相反的偏差相减剩下的：基线自己偏一笔，TPU 自己舍入一笔。",
        "代价：要两边各补跑一批只改一处的 1000 步运行；最后第一关的基线就定在 ⑤。",
    ])
    yb = f.src(yb + 10,
               "📌 实测：第 1000 步参数范数对 GPU 生产 golden 5 份均值：④ ＋1.857e-2、⑤ ＋2.071e-2、TPU 生产 ＋2.092e-2（v5p、v7x 当前配置同值）、⑥ ＋2.744e-2；TPU 精确轨迹 ＋2.73e-2。TPU 生产对 TPU 精确少 6.3e-3，对 ⑥ 少 6.5e-3，差不到一个记录刻度。",
               "📌 实测：① ＋8.697e-3（GPU 精确改分段累加 ＋8.55e-3）；② −6.836e-3（TPU 只把 dlogits 舍成 BF16、其余全精确）；GPU 真精确只多舍这一次 dlogits 少漂 6.74e-3；③ 由差额得出，约两个 FP32 记录刻度，单独分辨不清。")
    f.save("fig9-decomp.svg", yb + 14)


GATE1_PN = N.ACC[0][3]                 # 第一关正式判定：param_norm 带符号均值 ÷ ③ 那组 GPU 重跑的 N
assert N.ACC[0][0].startswith("param_norm") and GATE1_PN == 3.45


def fig_twolayer():
    FV5, FV7 = N.FINAL[0], N.FINAL[1]
    f = Fig(W, "定案后的验收协议。上一层：第一关，必须过。TPU 被测对最终 golden ⑤ 的一份比，⑤ 是 GPU 生产程序只把输出头 dgrad 改成分段累加；尺子是 ⑤ 自己同设置重跑 10 份、两两 45 对。"
               "v5p 和 v7x 当前配置每一项都在噪声内，最差 0.85 倍。旁边只报不判：TPU 只改执行参数的 7 份扰动运行，每份也都不超过 0.85 倍。"
               "下一层：第二关，已解释差异。TPU 对 GPU 生产 golden ② 仍差参数范数 +0.0209、loss −1.37e-4，全部是 GPU 输出头 dgrad 长求和的偏差。")
    y0 = f.header("golden 自己有偏差时：分两关验收（定案版）",
                  "上 ＝ 必须过的那一关；下 ＝ 只写清楚、不判过不过的那一关；箭头 ＝ 谁对谁比",
                  [(OR, "第一关：对最终 golden ⑤"), (GY, "第二关：对 GPU 生产 golden ②"), (BL, "TPU 被测"), (GR, "过")])
    PH = 400
    py = f.panel(0, y0, W, PH, "混元 3 的验收协议（第一关基线已定）", OR)

    def node(x, y, w, h, title, sub, col, fill="#fff"):
        f.box(x, y, w, h, fill, col, 10, sw=2)
        f.t(x + w / 2, y + 30, title, col, True, 16, "middle")
        if sub:
            f.t(x + w / 2, y + 54, sub, GY, size=13, anchor="middle")
    node(560, py + 150, 280, 76, "TPU 被测", "当前对齐配置，1000 步", BL)
    node(60, py + 30, 400, 76, "⑤ 最终 golden（一份当基线）", "GPU 生产 ＋ 输出头 dgrad 分段累加", OR)
    node(960, py + 30, 380, 76, "尺子：⑤ 自己重跑", "%d 份两两 %d 对" % (N.FINAL_RUNS, N.FINAL_PAIRS), OR)
    f.box(922, py + 140, 456, 54, "#fff", GY2, 10, sw=1.6, dash="5 4")
    f.t(1150, py + 162, "只报：TPU 扰动组 %d 份（只改执行参数）" % N.ENSEMBLE_RUNS, GY, True, 14, "middle")
    f.t(1150, py + 183, "每份对 ⑤ 的四项 r 都 ≤ %.2f" % N.FINAL_WORST, GY, size=12, anchor="middle")
    f.line(922, py + 167, 842, py + 185, GY2, 1.4, dash="5 4")
    node(60, py + 270, 400, 76, "② GPU 生产 golden", "整层重算 ＋ 路由回放", GY)
    node(960, py + 270, 380, 76, "已解释差异", "全部是 GPU 输出头 dgrad 的长求和", GY)
    f.elbow(560, py + 175, 460, py + 68, OR, 2.2)
    f.t(470, py + 124, "第一关：必须过", OR, True, 15)
    f.t(452, py + 148, "v5p 最差 %.2f×N · v7x 最差 %.2f×N，全过" % (max(FV5[1:5]), max(FV7[1:5])), GR, True, 13, "end")
    f.line(960, py + 68, 842, py + 165, OR, 1.6)
    f.t(850, py + 92, "量出 N", OR, True, 13)
    f.t(1150, py + 238, "param_norm：v5p %.2f×N、v7x %.2f×N" % (FV5[4], FV7[4]), GY, size=13, anchor="middle")
    f.elbow(560, py + 205, 460, py + 308, GY, 2.2)
    f.t(470, py + 262, "第二关：不判", GY, True, 15)
    f.line(462, py + 308, 958, py + 308, GY, 1.6)
    f.t(710, py + 330, "参数范数 ＋%.4f、loss %s（v5p）" % (N.VS_RC_PN, N.VS_RC_LOSS_V5P_T), GY, size=13, anchor="middle")
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "噪声尺子量的是「重跑会差多少」，量不出 golden 自己的系统偏差 —— 所以要分两关", [
        "定案前，第一关拿精确轨迹当基线、借 ③ 那组的尺子，判出 3.45 倍不过；那份基线自己的长求和也偏，这个判定是两笔相反偏差相减的结果。",
        "代价：最终 golden 每一步都要多做一次分段累加，GPU 步时多六成；第二关要把差的每一部分都对上一条查清的机制。",
    ])
    yb = f.src(yb + 10,
               "📌 实测：第一关对 ⑤ 的一份、尺子 ⑤ 自己 45 对。v5p loss 全程 %.2f、尾段 %.2f、梯度范数 %.2f、参数范数 %.2f；v7x %.2f / %.2f / %.2f / %.2f；EP＝8 两平台也全过；学习率逐位。"
               % (FV5[1], FV5[2], FV5[3], FV5[4], FV7[1], FV7[2], FV7[3], FV7[4]),
               "📌 实测：第二关对 ②：参数范数 ＋2.09e-2，loss −1.37e-4（v5p）／ −1.07e-4（v7x）。定案前的旧协议：对精确轨迹、尺子 ③ 10 对，v5p 参数范数 3.45 倍。")
    f.save("fig9-twolayer.svg", yb + 14)


def fig_regress():
    f = Fig(W, "每天的流程。训练代码的每个改动都要写一栏数值影响声明，然后跑一轮回归，对上一次全部通过的 TPU 运行比。"
               "逐位相同就通过，指针前移。不逐位就看声明：写了数学不变的，退回；执行级的改动或升级依赖，用 TPU 自己的执行噪声（TPU 28 对）判；"
               "数学级的改动，要对起点那份 golden（①）不变差（这是历史门，待换成第一关定下的基线），必要时回到验收的某一级重做。")
    y0 = f.header("日常回归：先要求逐位，不逐位再按声明分流",
                  "方块 ＝ 一个环节；菱形 ＝ 一个判断；参照不是 GPU，是上一次全部通过的 TPU 运行（LKG）",
                  [(GR, "通过"), (RD, "退回"), (OR, "用 TPU 噪声判"), (PU, "回到对齐验收")])
    PH = 360
    py = f.panel(0, y0, W, PH, "TPU 同一程序重跑逐位相同，所以「数学不变」的改动应当一位不变", BL)

    def node(x, y, w, h, t1, t2, col, fill="#fff"):
        f.box(x, y, w, h, fill, col, 10, sw=2)
        f.t(x + w / 2, y + (h / 2 + 5 if not t2 else 28), t1, col, True, 15, "middle")
        if t2:
            f.t(x + w / 2, y + 52, t2, GY, size=12, anchor="middle")
    node(20, py + 130, 210, 76, "训练代码改动", "写「数值影响」声明", INK)
    node(270, py + 130, 220, 76, "跑一轮回归", "一次约 10–25 分钟", INK)
    f.poly([(560, py + 168), (640, py + 120), (720, py + 168), (640, py + 216)], "#fff", BL, 2)
    f.t(640, py + 164, "对 LKG", BL, True, 14, "middle")
    f.t(640, py + 184, "逐位？", BL, True, 14, "middle")
    node(780, py + 30, 240, 60, "通过，LKG 前移", None, GR, "#e6f4ea")
    node(780, py + 140, 240, 56, "看声明", None, INK)
    node(1080, py + 40, 300, 66, "写了数学不变却不逐位 → 退回", "没写声明的也按这条处理", RD, "#fce8e6")
    node(1080, py + 130, 300, 66, "执行级 / 升级依赖", "容差门：TPU 执行噪声（TPU 28 对）", OR, "#fef7e0")
    node(1080, py + 220, 300, 76, "数学级", "对齐门：对起点那份 golden（①）不变差", PU, "#f3e8fd")
    f.t(1230, py + 290, "（历史门，待换成定下的基线）", PU, size=12, anchor="middle")
    f.line(230, py + 168, 268, py + 168, GY2, 1.8)
    f.line(490, py + 168, 558, py + 168, GY2, 1.8)
    f.elbow(640, py + 120, 778, py + 60, GR, 1.8)
    f.t(660, py + 92, "是", GR, True, 14)
    f.line(722, py + 168, 778, py + 168, RD, 1.8)
    f.t(734, py + 190, "否", RD, True, 14)
    for yy, c in ((py + 73, RD), (py + 163, OR), (py + 258, PU)):
        f.elbow(1020, py + 168, 1078, yy, c, 1.6)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "迁移验收证明「TPU 和 GPU 训的是同一个模型」；日常回归证明「今天的 TPU 和上次验过的 TPU 是同一个」", [
        "首个 LKG 要同一个提交跑两次逐位才能当；数值没变时指针自动前移，数值变了的每次换代都要负责人批准。",
        "代价：TPU 的执行噪声要自己造 —— 只改执行方式、不改数学的变体（叫扰动臂），旧配置试了 14 个、三项检查（第三项目前只判 loss）都过的 7 个，加基线 8 份凑出 TPU 28 对；配置一换，这 7 个重跑、重标。",
    ])
    yb = f.src(yb + 10,
               "📌 实测（v5p，8 颗芯片）：20 步那档一次约 10 分钟（提交到取回）；1000 步那档约 25 分钟。首个 LKG 同一提交双跑逐位；「数学不变」复测逐位通过，故意改一个开关被拦下。",
               "📌 首次标定在 FP32 累加开关打开之前的配置上做；开关打开后，上一次全过的运行同一提交双跑逐位、负责人批准后重建，7 个扰动臂在新配置下重跑全过，TPU 28 对重新标定（正式）；第三项检查目前只判了 loss。扰动臂要过三项检查：臂自己重跑逐位；配置、版本、编译出的程序除了改的那一处都不变；第 1 步梯度和 1000 步 loss 对基线的差不大于 GPU 自噪声的上限。标定绑死平台、档位、训练配置、镜像和依赖版本，任何一样变了就重标。")
    f.save("fig9-regress.svg", yb + 14)


fig_golden()
fig_decomp()
fig_twolayer()
fig_regress()
