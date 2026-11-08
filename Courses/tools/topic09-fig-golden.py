# -*- coding: utf-8 -*-
r"""专题九 · 第九、十节的四张图。

fig9-golden   回答：最终那份 golden 是怎么一步步来的？每一步加了什么、换来了什么？
              数字：同一个被测（v5p，FP32 累加开关打开前的配置）对三份 golden 的参数范数带符号均值 ÷ 同一把尺子 N（实测）。
fig9-decomp   回答：TPU 对 GPU 生产 golden 的参数范数差是谁的？—— 两个生产程序只差 GPU 输出头 dgrad 的长累加；对 ④ 的那截是三笔相加（实测，11-06 版）。
fig9-twolayer 回答：golden 自己有系统偏差时怎么验收？—— 分两关：第一关对精确轨迹必须过，第二关对生产 golden 写成已解释差异。
fig9-regress  回答：验收过了以后，每天合进来的代码怎么守？—— 先要求逐位；不逐位按 PR 的数值声明分流。
fig9-switch    回答：一个对齐开关，生产要不要开？—— 三问：修的是正确性吗？让 TPU 离真值更近吗？代价过 1% 吗？分成对齐专用、修错（生产也开）、取舍三类。
               1% 的界是本课取的口径；例子都是两次迁移的实测或读代码所得。
fig9-ladder    回答：规模怎么一级级往上，才知道新冒出来的差是谁带来的？—— 属性 × 级的格子：差冒出来的那一级翻色的几行就是嫌疑。
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
R[4] = N.FINAL_PN_SM_R                     # ⑤ 用它自己的 45 对尺子，v5p 当前配置；带符号均值，跟 ②–④ 同口径
assert abs(R[1] - 23.6) < 0.1 and abs(R[2] - 10.1) < 0.1 and abs(R[3] - 3.52) < 0.01, R


def fig_golden():
    f = Fig(W, "四张卡片从左到右：第一份是起点，框架默认配置；第二份加上整层重算和路由回放，跟生产一样；第三份再把输出头 dgrad 换成 FP32 乘法（一次累加）；"
               "第四份再改成 BF16 权重、全程 FP32 计算，叫精确轨迹，是定案前的基线；后来发现它输出头那次长求和仍有偏。第五份是最终 golden：回到第二份，只把输出头 dgrad 改成分段累加。"
               "卡片下面是同一个 TPU 被测对它的参数范数差，按同一把尺子 N 算倍数：对第二份 24 倍，对第三份 10 倍，对第四份 3.5 倍；第五份换成当前 TPU 和它自己 10 份两两 45 对的尺子，带符号均值只有 0.07 倍，过。")
    y0 = f.header("golden 是一步步改出来的　——　<tspan font-weight=\"700\">先对齐超参和数据，再每一步只加一样东西</tspan>",
                  "卡片 ＝ 一份 golden（GPU 上跑的标准答案）；柱子 ＝ TPU 被测对它的参数范数差是噪声的几倍（②–④ 同被测同尺子；⑤ 换成当前 TPU 和它自己的尺子）",
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
            f.t(x + 18, py + 270, "（不重算，跟生产不一样；1000 步验收不用它）", GY, size=13)
    f.t(30, py + 350, "柱子：TPU 被测的参数范数带符号均值 ÷ N（②–④ 的 N ＝ ③ 那组 5 份两两 10 对的最大值；⑤ 用它自己 10 份 45 对）；竖线 ＝ 2×N", GY, size=13)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "每往精确值靠一步，TPU 的「偏差」就缩一截：24 倍 → 10 倍 → 3.5 倍；最终 golden 上 0.07 倍", [
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
    py2 = f.panel(0, py + PH + 16, W, PH2, "TPU 生产对 ④ 的 ＋%.5f 是三笔相加（第 1000 步，v7x 那组补跑；v5p 那组见出处）" % N.TPU_VS_FIX1, BL)
    K = 760 / 0.0095
    xs = X0 + 120
    ytop = py2 + 50
    steps = [(N.B_GPUEXACT, RD, "第一笔　④ 自己少漂：cuBLAS FP32 一次累加整个词表（GPU 一侧）", "%.2e"),
             (N.B_DLOGITS, BL, "第二笔　TPU 输出头反向的 dlogits 舍成 BF16（TPU 一侧）", "%.2e"),
             (N.B_REST, LINE2, "第三笔　其余 BF16 存储与逐元素", "%.1e")]
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
        "GPU 生产只修这一处，TPU 生产就跟它几乎重合（按 ⑤ 自己 45 对的尺子，v5p、v7x 全过）；剩下离真精确的那截是两边都有的 dlogits 舍入，不算两边的差。",
        "定案前对 ④ 的那点差，是头两笔方向相反的偏差相减剩下的：基线自己偏一笔，TPU 自己舍入一笔。",
        "代价：要两边各补跑一批只改一处的 1000 步运行；最后第一关的基线就定在 ⑤。",
    ])
    yb = f.src(yb + 10,
               "📌 实测：第 1000 步参数范数对 GPU 生产 golden 5 份均值：④ ＋1.857e-2、⑤ ＋2.071e-2、TPU 生产 ＋2.092e-2（v5p、v7x 当前配置同值）、⑥ ＋2.744e-2；TPU 精确轨迹 ＋2.73e-2。TPU 生产对 TPU 精确少 6.3e-3，对 ⑥ 少 6.5e-3，差不到一个记录刻度。",
               "📌 实测（v7x）：第一笔 ＋8.697e-3（GPU 精确改分段累加 ＋8.55e-3）；第二笔 −6.836e-3（TPU 只把 dlogits 舍成 BF16、其余全精确）；第三笔由差额得出，约两个 FP32 记录刻度。v5p 那组：＋7.97e-3、−6.10e-3、＋4.9e-4，同样合到 ＋2.35e-3。GPU 真精确只多舍这一次 dlogits 少漂 6.74e-3。")
    f.save("fig9-decomp.svg", yb + 14)


GATE1_PN = N.ACC[0][3]                 # 第一关正式判定：param_norm 带符号均值 ÷ ③ 那组 GPU 重跑的 N
assert N.ACC[0][0].startswith("param_norm") and GATE1_PN == 3.45


def fig_twolayer():
    FV5, FV7 = N.FINAL[0], N.FINAL[1]
    f = Fig(W, "定案后的验收协议。上一层：第一关，必须过。TPU 被测对最终 golden ⑤ 的一份比，⑤ 是 GPU 生产程序只把输出头 dgrad 改成分段累加；尺子是 ⑤ 自己同设置重跑 10 份、两两 45 对。"
               "v5p 和 v7x 当前配置每一项都在噪声内，最差 0.85 倍。旁边只报不判：TPU 只改执行参数的 9 份扰动运行，8 份不超过 0.85 倍，1 份梯度范数待定，落在 TPU 自己的执行噪声里。"
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
    f.t(1150, py + 183, "%d 份四项 r ≤ %.2f；1 份梯度范数待定（TPU 自己的噪声内）" % (N.ENSEMBLE_OK, N.FINAL_WORST), GY, size=12, anchor="middle")
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
               "📌 实测：第二关对 ②：参数范数 ＋2.09e-2，loss −1.37e-4（v5p）；v7x 约 −1.08e-4（推断，v7x 逐步序列没入库）。定案前的旧协议：对精确轨迹、尺子 ③ 10 对，v5p 参数范数 3.45 倍。")
    f.save("fig9-twolayer.svg", yb + 14)


def fig_regress():
    f = Fig(W, "每天的流程。训练代码的每个改动都要写一栏数值影响声明，然后跑一轮回归，对上一次全部通过的 TPU 运行比。"
               "逐位相同就通过，指针前移。不逐位就看声明：写了数学不变的，退回；执行级的改动或升级依赖，用 TPU 自己的执行噪声（TPU 28 对）判；"
               "数学级的改动，要对起点那份 golden（①）不变差（这是历史门，本课建议换成最终 golden ⑤），必要时回到验收的某一级重做。")
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
    f.t(1230, py + 290, "（历史门，建议换成最终 golden ⑤）", PU, size=12, anchor="middle")
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
        "代价：TPU 的执行噪声要自己造 —— 只改执行方式、不改数学的变体（叫扰动臂），旧配置试了 14 个、三项检查都过的 7 个，加基线 8 份凑出 TPU 28 对；配置一换，这 7 个重跑、重标。",
    ])
    yb = f.src(yb + 10,
               "📌 实测（v5p，8 颗芯片）：20 步那档一次约 10 分钟（提交到取回）；1000 步那档约 25 分钟。首个 LKG 同一提交双跑逐位；「数学不变」复测逐位通过，故意改一个开关被拦下。",
               "📌 首次标定在 FP32 累加开关打开之前的配置上做；开关打开后，上一次全过的运行同一提交双跑逐位、负责人批准后重建，7 个扰动臂在新配置下重跑全过，TPU 28 对重新标定（正式）。扰动臂要过三项检查：臂自己重跑逐位；配置、版本、编译出的程序除了改的那一处都不变；1000 步 loss 对基线不超过 GPU 自噪声的上限，第 1 步梯度离 FP64 不比基线远（当前配置 6 个臂 R：v5p 1.001–1.008、v7x 1.003–1.039，全过）。标定绑死平台、档位、训练配置、镜像和依赖版本，任何一样变了就重标。")
    f.save("fig9-regress.svg", yb + 14)



def fig_journey():
    rows = [
        (True,  "地基铺平", "数据原样回放、权重逐张量逐位、按实际生效的权重衰减", "第 1 步 loss 只差 %s" % N.D1_T),
        (True,  "换成噪声尺子", "同设置 GPU 跑 10 份，看 TPU 落在哪", "偏移在参考档里，却离 GPU 均值 %s 个标准差：真偏差" % N.SIGMA_RANGE),
        (True,  "按参数分组定位", "参数范数多涨的那截拆到组、拆到行", "%s 在输出层，集中在没出现过的词" % N.GROUP_HEAD_SHARE_T),
        (True,  "双向干预", "两边都去掉「所有 logits 一起升降」那股力", "lse 的差 %s → %s" % (N.ORIG_LSE_T, tx(N.REMOVE_LSE, "%.3f"))),
        (False, "走错：「FP32 dgrad」其实是 TF32", "偏 %s，比 BF16 的 %s 还大，一度判成换 dgrad 也不动" % (tx(abs(N.HEAD_TF32_S1)), tx(abs(N.HEAD_BF16_S1))), "教训：开关以实测为准，读回值不作数"),
        (True,  "找到源头", "拿训练里的真张量，对 FP64 拆输出头反向", "GPU dgrad 朝零 %s，TPU %s（第 1 步）" % (tx(N.HEAD_BF16_S1), N.HEAD_TPU_S1_T)),
        (False, "走错：全程 FP32 下路由也翻，当成两边实现不同", "撤回：TPU 那份 FP32 参照里路由专家其实按 BF16 算，翻转是它的下游", "教训：参照本身先对 FP64 验一遍"),
        (True,  "golden 一步步改", "② 生产近似 → ③ dgrad 换 FP32 → ④ 精确轨迹", "TPU 对它 %s → %s → %s 倍噪声" % (N.GOLDEN_R2_T, N.GOLDEN_R3_T, N.GOLDEN_R4_1_T)),
        (True,  "TPU 开关：部分和留 FP32", "窗口化 collective matmul 不再在 BF16 里累加", "qkv 投影 R %.3f → %s；对 ④ %s → %s 倍" % (N.R_QKV_BEFORE, N.R_QKV_AFTER_T, N.GOLDEN_R4_T, "3.45")),
        (False, "走错：精确轨迹的噪声只看了一对", "像是比借来的尺子窄 %d 倍；五份 10 对一量，同一量级" % N.EXACT_PAIR1_X, "教训：一对样本量不出散布"),
        (False, "走错：把 ④ 当成精确值", "FP32 一次累加 12 万也偏，④ 自己少漂 %s；分段累加后两边精确轨迹重合" % N.B_GPUEXACT_T, "教训：裁判也要被核对"),
        (True,  "定案 ⑤：只修那一次长求和", "10 对时几项 %s 倍待定 → 两处测量口径 → 补到 45 对" % N.PEND10_T, "v5p、v7x 全部在噪声内，最差 %s 倍" % N.FINAL_WORST_T),
    ]
    f = Fig(W, "从上往下是这一次排查大致的先后：绿色是留下来的一步，写着改了什么、前后的数；红色虚框是走错的四步，写着错在哪、留下什么教训。"
               "最后一步定案：GPU 生产只修输出头 dgrad 那一次长求和，当第一关的基线，TPU 在 v5p、v7x 上全部落进噪声，最差 0.85 倍。")
    y0 = f.header("一路走来：十二步，其中四步走错",
                  "大致按时间从上往下；绿 ＝ 留下来的一步（改了什么 → 前后的数），红虚框 ＝ 走错的一步（错在哪 → 教训）",
                  [(GR, "留下来的一步"), (RD, "走错的一步")])
    RH, GAP = 50, 8
    PH = len(rows) * (RH + GAP) + 44
    py = f.panel(0, y0, W, PH, "混元 3 截 4 层，GPU → TPU（实测）", BL)
    k = 0
    for i, (ok, title, what, res) in enumerate(rows):
        y = py + 18 + i * (RH + GAP)
        col = GR if ok else RD
        f.box(30, y, W - 60, RH, "#fff" if ok else "#fce8e6", col, 8, sw=1.6, dash=None if ok else "6 4")
        if ok:
            k += 1
            f.t(52, y + 31, str(k), col, True, 18, "middle")
        else:
            f.t(52, y + 31, "✗", col, True, 18, "middle")
        f.t(78, y + 22, title, col, True, 15)
        f.t(78, y + 42, what, GY, size=12.5)
        f.t(W - 50, y + 31, res, INK if ok else RD, True, 13.5, "end")
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "四步走错，三步错在「参照」：开关、FP32 参照、精确轨迹都被当成了它名字说的那样", [
        "最终方案里改数值的主要是两处：TPU 的部分和留 FP32、GPU golden 的分段累加（另有几个只做显式保证的 TPU 开关）；其余几步都是把量法和参照弄对。",
        "代价：每一步都要一条只改一处的运行做证据；走错的几步也写进记录，不悄悄删掉。",
    ])
    yb = f.src(yb + 10, "📌 实测：各行的数都在出处台账里；「走错」那几步的原结论和更正都留在排查记录里。")
    f.save("fig9-journey.svg", yb + 14)

def fig_switch():
    f = Fig(W, "一棵三问的决策树。第一问：这个开关修的是正确性吗，比如 NaN、语义写错？是，归修错类，生产也开。"
               "不是，第二问：打开以后 TPU 离 FP64 真值更近吗？不是，它只是照 GPU 多舍一次、换个舍入位置，归对齐专用，对齐时开、生产关。"
               "是，第三问：步时代价小于 1%% 吗？小于，生产也开；不小于，归取舍。"
               "叶子下面各举实例：对齐专用有强制舍入照 GPU、优化器迭代改到 BF16；修错有 NaN 的修复；更准且便宜有部分和留 FP32、逐项 FP32 乘加、精确的除法开方；"
               "取舍：同一个部分和 FP32 开关，在 v5p 上多 %s，在 v7x 上多 %s。" % (N.DFA_COST_V5P_T, N.DFA_COST_V7X_T))
    y0 = f.header("一个开关上不上生产：问三件事",
                  "蓝色圆角框 ＝ 一个问题；彩框 ＝ 结论和例子；1% 是本课取的步时界线",
                  [(RD, "修错：生产也开"), (GR, "更准且便宜：生产也开"), (GY, "对齐专用：生产关"), (OR, "取舍：看代价")])
    PH = 470
    py = f.panel(0, y0, W, PH, "决策树（例子取自两次迁移）", BL)

    def ask(x, y, txt):
        f.box(x, y, 300, 52, "#e8f0fe", BL, 26)
        f.t(x + 150, y + 31, txt, BL, True, 14, "middle")

    def leaf(x, y, w, col, head, ex):
        f.box(x, y, w, 40 + 22 * len(ex), "#fff", col, 6, sw=2)
        f.t(x + 14, y + 26, head, col, True, 15)
        for k, e in enumerate(ex):
            f.t(x + 14, y + 50 + 22 * k, "· " + e, INK, size=12.5)
    ask(80, py + 30, "修的是正确性吗？")
    leaf(460, py + 26, 400, RD, "修错 → 生产也开", ["负数开根号出 NaN 的修复"])
    f.line(380, py + 56, 458, py + 56, GY2, 1.6); f.t(400, py + 48, "是", GY, size=12)
    f.line(230, py + 82, 230, py + 150, GY2, 1.6); f.t(240, py + 120, "不是", GY, size=12)
    ask(80, py + 152, "打开后 TPU 离 FP64 更近吗？")
    leaf(460, py + 140, 470, GY, "对齐专用 → 对齐时开、生产关", ["照 GPU 强制舍成 BF16", "优化器一段迭代改到 BF16", "多舍一次 BF16、照 GPU 的舍入位置"])
    f.line(380, py + 178, 458, py + 178, GY2, 1.6); f.t(390, py + 170, "不是", GY, size=12)
    f.line(230, py + 204, 230, py + 290, GY2, 1.6); f.t(240, py + 254, "是", GY, size=12)
    ask(80, py + 292, "步时代价 ＜ %s？" % N.SW_COST_GATE)
    leaf(460, py + 280, 430, GR, "更准且便宜 → 生产也开", ["部分和留 FP32（v5p ＋%s）" % N.DFA_COST_V5P_T, "逐项 FP32 乘加（＋%s）" % N.SW_MIX_COST, "精确的除法、开方（＋%s）" % N.SW_ADAM_COST])
    f.line(380, py + 318, 458, py + 318, GY2, 1.6); f.t(400, py + 310, "是", GY, size=12)
    leaf(960, py + 280, 410, OR, "取舍 → 看代价", ["同一个部分和开关，v7x ＋%s" % N.DFA_COST_V7X_T, "平台不同，类别可能不同"])
    f.path("M230 %d L230 %d L1165 %d L1165 %d" % (py + 344, py + 430, py + 430, py + 392), GY2, 1.6)
    f.t(250, py + 422, "不是", GY, size=12)
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "对齐要开的开关，不等于生产要开的开关：先问它让 TPU 更准还是只是更像 GPU", [
        "只为照 GPU 的舍入位置而开的（多舍一次 BF16），生产关掉同样准，甚至更准；修正确性的、让 TPU 更准又便宜的，生产也开。",
        "代价：步时百分比在小模型上量，固定开销占比高，不能直接外推到生产规模；按目标规模重量一次。",
    ])
    yb = f.src(yb + 10,
               "📌 步时：部分和 FP32（混元 3，100 步对比）v5p ＋%s、v7x ＋%s；逐项 FP32 乘加 ＋%s（第二次迁移，只差这一个开关的两次运行）；强制舍入 ＋%s（跨代对比）。"
               % (N.DFA_COST_V5P_T, N.DFA_COST_V7X_T, N.SW_MIX_COST, N.SW_ELIDE_COST),
               "⚠️ 1% 的界是本课取的口径（照同一实践的分类）；v7x 那一项按这个口径归「取舍」是本课推导（同一实践在小模型上把这个开关记作「更准且便宜」）；精确除法开方的 ＋" + N.SW_ADAM_COST + " 是跨代对比。另有不进这棵树的两类：按训练配方定的设置（例如路由偏置做不做零均值更新），和判法、工具、量法。")
    f.save("fig9-switch.svg", yb + 14)


def fig_ladder():
    f = Fig(W, "一张格子：行是模型的几样属性（宽度和注意力形状、专家个数、专家宽度、层数、权重），列是三级：小模型、中等尺寸、真实权重截层。"
               "每格涂色：浅灰是跟小模型一样，蓝是跟真实模型一样，橙是改了、但两边都不像。"
               "小模型全是浅灰。中等尺寸只有宽度和注意力形状变蓝、专家宽度变橙，其余仍是浅灰，权重仍随机。真实权重截层：专家个数、专家宽度、权重变蓝，层数变橙（只截一到两层）。"
               "最右一列是结果：小模型上查根源、验证开关；中等尺寸 300 步过 5、不过 0、待定 4，真实权重那一级的不过没有出现；真实权重截层第 1 步前向就差 8.5e-5。"
               "差冒出来的那一级翻色的是专家个数、专家宽度、层数、权重，嫌疑就在这几行；宽度和注意力形状在中等尺寸已经变蓝、没出这个差，排除。")
    y0 = f.header("差冒出来的那一级翻了哪几行，嫌疑就在那几行",
                  "行 ＝ 模型的一样属性；列 ＝ 一级；颜色 ＝ 这一级的这一样跟谁一样；最右 ＝ 这一级的结果",
                  [(GY, "同小模型"), (BL, "同真实模型"), (OR, "改了，两边都不像")])
    ROWS = ["宽度、注意力形状", "专家个数", "专家宽度", "层数", "权重"]
    COLS = [("① 小模型", ["s", "s", "s", "s", "s"], "查根源、验证开关", BL),
            ("② 中等尺寸", ["r", "s", "m", "s", "s"], "300 步：真实权重那一级的不过没出现（%d 过、%d 不过、%d 待定，最差 %.2f N）" % (N.LADDER_MID_PASS + (N.LADDER_MID_WORST,)), OR),
            ("③ 真实权重截层", ["r", "r", "r", "m", "r"], "第 1 步前向就差 %s（小模型 ≤ 3.6e-6）" % N.LADDER_REAL_STEP1_T, RD)]
    FILL = {"s": ("#f1f3f4", GY, "同小模型"), "r": ("#e8f0fe", BL, "同真实模型"), "m": ("#fef7e0", OR, "改了")}
    RH, X0, CW = 44, 220, 170
    PH = 70 + RH * len(ROWS) + 50
    py = f.panel(0, y0, W, PH, "三级，每级换了哪几样（第二次迁移）", BL)
    for j, (h, _, _, col) in enumerate(COLS):
        f.t(X0 + j * (CW + 10) + CW / 2.0, py + 28, h, col, True, 15, "middle")
    for i, rname in enumerate(ROWS):
        y = py + 44 + i * RH
        f.t(24, y + 27, rname, INK, True, 14)
        for j, (_, cells, _, _) in enumerate(COLS):
            fill, col, lab = FILL[cells[i]]
            changed = j > 0 and cells[i] != COLS[j - 1][1][i]
            f.box(X0 + j * (CW + 10), y + 4, CW, RH - 8, fill, RD if (changed and j == 2) else col, 4, sw=2.4 if (changed and j == 2) else 1)
            f.t(X0 + j * (CW + 10) + CW / 2.0, y + 27, lab, col, size=12.5, anchor="middle")
    RX = X0 + 3 * (CW + 10) + 20
    import math as _m
    f.t(RX, py + 28, "第 1 步主 loss 的差（对数轴）", INK, True, 15)
    AX0, AX1 = RX + 130, W - 40
    def LX(v):
        return AX0 + (_m.log10(v) - (-6.3)) / (-3.7 - (-6.3)) * (AX1 - AX0)
    vals = [(max(N.LADDER_MINI_STEP1), N.LADDER_MINI_T), (None, "第 1 步诊断待出"), (abs(N.LADDER_REAL_STEP1), N.LADDER_REAL_STEP1_T)]
    for j, ((h, _, _, col), (v, lab)) in enumerate(zip(COLS, vals)):
        y = py + 56 + j * 54
        f.t(RX, y + 14, h.split(" ")[0] + " " + h.split(" ")[1], col, True, 13)
        if v is None:
            f.box(AX0, y, (AX1 - AX0) * 0.5, 20, "none", col, 3, dash="5 4")
            f.t(AX0 + 8, y + 15, lab, col, size=12)
        else:
            f.box(AX0, y, LX(v) - AX0, 20, col, col, 3)
            f.t(LX(v) + 6, y + 15, lab, col, True, 12)
    for e in (1e-6, 1e-5, 1e-4):
        f.line(LX(e), py + 44, LX(e), py + 56 + 3 * 54 - 10, LINE2, 1, arrow=False)
        f.t(LX(e), py + 56 + 3 * 54 + 6, "%.0e" % e, GY, size=11, anchor="middle")
    f.t(RX, py + 56 + 3 * 54 + 30, "② 只有 300 步的判定（所以排除是推断）：%d 过、%d 不过、%d 待定（最差 %.2f N）" % (N.LADDER_MID_PASS + (N.LADDER_MID_WORST,)), OR, True, 12.5)
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "每一级尽量只换一类；差冒出来时，看这一级翻色的那几行（红框）", [
        "宽度和注意力形状在第二级就换成真的、没出那一级的不过，排除（推断）；第三级一起换了专家个数、专家宽度、层数和权重，还拆不开。",
        "代价：每多一级要多一套 GPU 重跑当尺子；这一次要再拆，得再补一级（比如真实结构配随机权重）。",
    ])
    yb = f.src(yb + 10,
               "📌 实测：中等尺寸 300 步对 GPU 8 份两两 28 对，四个待定都只略超噪声；它的第 1 步逐张量诊断还没出，所以「没出这个差」说的是 300 步那一级。真实权重截层第 1 步主 loss 相对差 %s，GPU 8 份第 1 步逐位相同；小模型各代第 1 步 %s，在 GPU 份间散布以内。" % (N.LADDER_REAL_STEP1_T, N.LADDER_MINI_T),
               "⚠️ 推断：排除宽度和注意力形状，靠的是中等尺寸跟真实模型这几样相同、又没出那一级的不过。格子只写同谁一样，不写具体尺寸。")
    f.save("fig9-ladder.svg", yb + 14)


fig_golden()
fig_decomp()
fig_journey()
fig_twolayer()
fig_regress()
fig_switch()
fig_ladder()
