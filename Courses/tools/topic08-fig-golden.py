# -*- coding: utf-8 -*-
r"""专题八 · 第九、十节的四张图。

fig8-golden   回答：最终那份 golden 是怎么一步步来的？每一步加了什么、换来了什么？
              数字：同一个被测（v5p，FP32 累加开关打开前的配置）对三份 golden 的参数范数带符号均值 ÷ 同一把尺子 N（实测）。
fig8-decomp   回答：TPU 对 GPU 生产 golden 的参数范数差是谁的？—— 两个生产程序只差 GPU 输出头 dgrad 的长累加；对 ④ 的那截是三笔相加（实测，11-06 版）。
fig8-twolayer 回答：golden 自己有系统偏差时怎么验收？—— 分两关：第一关对精确轨迹必须过，第二关对生产 golden 写成已解释差异。
fig8-regress  回答：验收过了以后，每天合进来的代码怎么守？—— 先要求逐位；不逐位按 PR 的数值声明分流。
fig8-switch    回答：一个对齐开关，生产要不要开？—— 三问：修的是正确性吗？让 TPU 离真值更近吗？代价过 1% 吗？分成对齐专用、修错（生产也开）、取舍三类。
               1% 的界是本课取的口径；例子都是两次迁移的实测或读代码所得。
fig8-ladder    回答：规模怎么一级级往上，才知道新冒出来的差是谁带来的？—— 属性 × 级的格子：差冒出来的那一级翻色的几行就是嫌疑。
"""
from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE, LINE2, PU
Fig.BAND_KEEP_LINES = 0   # 锤炼阶段（2026-11-14）：图上落点带只留一行粗体结论，其余进图下折叠
import topic08_numbers as N
from topic08_numbers import tx

W = 1400
NOISE = N.GPU_RERUN_PN
# R3：颜色 ＝ 全课约定（GPU 生产 golden 深灰、精确轨迹紫）；卡片上不提前用「第一关／第二关」，关卡名留到 fig8-twolayer 再出
STEPS = [("① 起点", "框架默认配置", ["BF16，不重算", "TF32 开着"], None, GY2, "归因参照"),
         ("② 生产近似", "＋ 整层重算 ＋ 路由回放", ["模仿生产开重算", "跟①比：各项都在 GPU 噪声内"], 2.93e-6, GY, "生产 golden：差要写清来由"),
         ("③ 只修一处", "＋ 输出头 dgrad 换成 FP32", ["解释 loss 偏移的 73%", "参数范数 ＋0.0124", "定案前的尺子取自这组"], 1.25e-6, OR, "归因证据"),
         ("④ 精确轨迹", "＋ BF16 权重、全程 FP32", ["真关 TF32；可输出头", "长求和一次累加，有偏"], 4.36e-7, PU, "定案前的基线"),
         ("⑤ 修好的 golden", "② ＋ dgrad 升 FP32 分块", ["只改这一处；一步 0.38 s（生产 0.236 s）", "只分块不升 FP32 也够：0.25 s", "尺子：自己 10 份 45 对"], None, GR, "第一关的基线（定案）")]
R = [None if st[3] is None else st[3] / NOISE for st in STEPS]
R[4] = N.FINAL_PN_SM_R                     # ⑤ 用它自己的 45 对尺子，v5p 当前配置；带符号均值，跟 ②–④ 同口径
assert abs(R[1] - 23.6) < 0.1 and abs(R[2] - 10.1) < 0.1 and abs(R[3] - 3.52) < 0.01, R


def fig_golden():
    f = Fig(W, "五张卡片从左到右：第一份是起点，框架默认配置；第二份加上整层重算和路由回放，跟生产一样；第三份再把输出头 dgrad 换成 FP32 乘法（一次累加）；"
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
                f.t(x + 18, py + 346, "（带符号均值；三个统计取最差是 %s）" % N.FINAL_PN_V5P_T, GY, size=12)
        else:
            f.t(x + 18, py + 270, "（不重算；1000 步", GY, size=13)
            f.t(x + 18, py + 290, "验收不用它）", GY, size=13)
    f.t(30, py + 372, "柱子：TPU 被测的参数范数带符号均值 ÷ N（②–④ 的 N ＝ ③ 那组 5 份两两 10 对的最大值；⑤ 用它自己 10 份 45 对）；竖线 ＝ 2×N", GY, size=13)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "每往精确值靠一步，TPU 的「偏差」就缩一截：24 倍 → 10 倍 → 3.5 倍；最终 golden 上 0.07 倍", [
        "缩掉的部分都查清了是 GPU golden 自己的偏差；④ 上剩的那截也拆清了：④ 自己少漂一笔、TPU 的 dlogits 舍入少漂一笔，两笔相减。最后只修 GPU 那一次长求和，两边就落进同一团噪声。",
        "代价：每一份 golden 都要单独跑满 1000 步、存好数据；改了什么、为什么改、证据在哪，都要进一张出处总表。",
    ])
    yb = f.src(yb + 10,
               "📌 实测：被测是 v5p 上生产配置 1000 步（FP32 累加开关打开之前的那一版），param_norm 带符号均值对 ② 2.93e-6、对 ③ 1.25e-6、对 ④ 4.36e-7；N ＝ 1.24e-7。开关打开后的当前配置对 ④ 是 4.27e-7（3.45 倍）。",
               "📌 实测：③ − ② 在第 301–1000 步 loss 上是 −9.8e-5，占 TPU 对 ② 偏移（−1.34e-4）的 73%；第 1000 步参数范数 ＋1.24e-2（③ 的 dgrad 用的是 cuBLAS FP32 一次累加，本身也偏，所以这只是 BF16 那一截）。")
    f.save("fig8-golden.svg", yb + 14)


def fig_decomp():
    f = Fig(W, "上面一条数轴是第 1000 步参数范数对 GPU 生产 golden 的差：GPU 精确轨迹 ＋0.0186，GPU 生产只把输出头 dgrad 改成分段累加的那份 ＋0.0207，TPU 生产 ＋0.0209，真精确 ＋0.0274。"
               "从 0 到 ＋0.0207 是 GPU 输出头 dgrad 长累加的偏差；从 ＋0.0207 到 ＋0.0274 是两边都有的 dlogits 舍入。"
               "下面是 TPU 生产对精确轨迹那 ＋0.00235 的三笔：精确轨迹自己少漂 ＋0.0087，TPU 的 dlogits 舍入 −0.0068，其余 ＋0.0005。")
    y0 = f.header("新的账：两个生产程序只差 GPU 那一次长累加",
                  "上：各条运行第 1000 步落在哪（实测）；下：TPU 对精确轨迹那一小截的三笔（实测，按定义加起来正好闭合）",
                  [(GY, "GPU 生产"), (PU, "精确轨迹 ／ 真精确"), (OR, "GPU 生产＋升 FP32 分段"), (BL, "TPU 生产"), (RD, "GPU 长累加偏差"), (LINE2, "两个生产程序都有")])
    PH = 300
    py = f.panel(0, y0, W, PH, "第 1000 步参数范数，对 GPU 生产 golden 5 份均值的差（生产近似配置，跟上一张图的不重算配置不同）", BL)
    X0, BW, VMAX = 60, 1260, 0.029
    X = lambda v: X0 + v / VMAX * BW
    ya = py + 120
    f.line(X0, ya, X0 + BW, ya, INK, 1.5, arrow=False)
    for v in (0, 0.01, 0.02):
        f.line(X(v), ya - 5, X(v), ya + 5, INK, 1.2, arrow=False)
        f.t(X(v), ya + 22, "0" if v == 0 else "＋%.2f" % v, GY, size=12, anchor="middle")
    pts = [(0.0, "GPU 生产", GY, -1), (N.PN_FIX1, "④ 精确轨迹 ＋%.4f" % N.PN_FIX1, PU, -1),
           (N.PN_PRODCHUNK, "⑤ 生产＋升 FP32 分段 ＋%.4f" % N.PN_PRODCHUNK, OR, -2), (N.PN_TPU_PROD, "TPU 生产 ＋%.4f" % N.PN_TPU_PROD, BL, -3),
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
    f.box(X(0), yb1, X(N.PN_RCX5) - X(0), 22, RD, RD, 3)
    f.t(X(0) + 8, yb1 + 16, "换成 FP32（去掉截断那条路）＋%.4f" % N.PN_RCX5, "#fff", True, 13)
    f.box(X(N.PN_RCX5), yb1, X(N.PN_PRODCHUNK) - X(N.PN_RCX5), 22, "#f28b82", "#f28b82", 3)
    f.t(X(N.PN_RCX5) + 8, yb1 + 16, "再分块（去掉吞小项，推断）＋%.4f" % (N.PN_PRODCHUNK - N.PN_RCX5), INK, True, 13)
    f.box(X(N.PN_PRODCHUNK), yb1, X(N.PN_TRUE) - X(N.PN_PRODCHUNK), 22, LINE2, GY2, 3)
    f.t(X(N.PN_TRUE), yb1 + 46, "两个生产程序都有（对真精确比）：输出头反向的 dlogits 舍成 BF16 ＋%.4f" % N.LADDER_SHARED, GY, True, 13, anchor="end")
    f.t(X0, py + PH - 40, "TPU 生产和 ⑤ 几乎落在同一点；⑥ 真精确 ＝ ④ 再把那次长求和分段（只当参照），TPU 自己的精确轨迹也落在这里", GY, size=13)
    f._pan = None
    PH2 = 250
    py2 = f.panel(0, py + PH + 16, W, PH2, "TPU 生产对 ④ 的 ＋%.5f 是三笔相加（第 1000 步，v7x 那组补跑；v5p 那组见出处）" % N.TPU_VS_FIX1, BL)
    K = 760 / 0.0095
    xs = X0 + 120
    ytop = py2 + 50
    steps = [(N.B_GPUEXACT, RD, "第一笔 →　④ 自己少漂：cuBLAS FP32 一次累加整个词表（GPU 一侧）", "%.2e"),
             (N.B_DLOGITS, BL, "← 第二笔　TPU 的 dlogits 舍成 BF16（④ 没有这一舍）", "%.2e"),
             (N.B_REST, LINE2, "第三笔　其余 BF16 存储与逐元素", "%.1e")]
    f.line(xs, ytop - 14, xs, ytop + 128, INK, 1.4, arrow=False)
    f.t(xs - 6, ytop - 18, "0", INK, True, 13, "end")
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
        ac = GY2 if c == LINE2 else c
        f.line(xs + a * K, yy + 31, xs + b * K, yy + 31, ac, 2, arrow=True)     # 方向：这一笔把读数往哪推
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
    f.save("fig8-decomp.svg", yb + 14)


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
    node(60, py + 30, 400, 76, "⑤ 最终 golden（一份当基线）", "GPU 生产 ＋ 输出头 dgrad 升 FP32 分段累加", OR)
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
    f.t(452, py + 168, "其中参数范数：v5p %.2f×N、v7x %.2f×N" % (FV5[4], FV7[4]), GR, size=12, anchor="end")
    f.elbow(560, py + 205, 460, py + 308, GY, 2.2)
    f.t(470, py + 262, "第二关：不判", GY, True, 15)
    f.line(462, py + 308, 958, py + 308, GY, 1.6)
    f.t(710, py + 330, "参数范数 ＋%.4f、loss %s（v5p）" % (N.VS_RC_PN, N.VS_RC_LOSS_V5P_T), GY, size=13, anchor="middle")
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "噪声尺子量的是「重跑会差多少」，量不出 golden 自己的系统偏差 —— 所以要分两关", [
        "定案前，第一关拿精确轨迹当基线、借 ③ 那组的尺子，判出 3.45 倍不过；那份基线自己的长求和也偏，这个判定是两笔相反偏差相减的结果。",
        "代价：最终 golden 每一步都要多做一次分段累加，GPU 一步从 0.236 s 变 0.38 s；第二关要把差的每一部分都对上一条查清的机制。",
    ])
    yb = f.src(yb + 10,
               "📌 实测：第一关对 ⑤ 的一份、尺子 ⑤ 自己 45 对。v5p loss 全程 %.2f、尾段 %.2f、梯度范数 %.2f、参数范数 %.2f；v7x %.2f / %.2f / %.2f / %.2f；EP＝8 两平台也全过；学习率逐位。"
               % (FV5[1], FV5[2], FV5[3], FV5[4], FV7[1], FV7[2], FV7[3], FV7[4]),
               "📌 实测：第二关对 ②：参数范数 ＋2.09e-2，loss −1.37e-4（v5p）；v7x −1.07e-4（直接算）。定案前的旧协议：对精确轨迹、尺子 ③ 10 对，v5p 参数范数 3.45 倍。")
    f.save("fig8-twolayer.svg", yb + 14)


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
    f.save("fig8-regress.svg", yb + 14)



def fig_journey():
    G = [("地基铺平", "第 1 步 loss 只差 %s" % N.D1_T),
         ("换成噪声尺子", "离 GPU 均值 %s 个标准差" % N.SIGMA_RANGE),
         ("按参数分组定位", "%s 在输出层的冷门词" % N.GROUP_HEAD_SHARE_T),
         ("双向干预", "lse 的差 %s → %s" % (N.ORIG_LSE_T, tx(N.REMOVE_LSE, "%.3f"))),
         ("找到源头", "GPU dgrad 朝零 %s" % tx(N.HEAD_BF16_S1)),
         ("golden 一步步改", "%s → %s → %s 倍噪声" % (N.GOLDEN_R2_T, N.GOLDEN_R3_T, N.GOLDEN_R4_1_T)),
         ("部分和留 FP32", "qkv R %.3f → %s" % (N.R_QKV_BEFORE, N.R_QKV_AFTER_T)),
         ("定案 ⑤", "最差 %s 倍，全过" % N.FINAL_WORST_T)]
    X = [("「FP32 dgrad」其实是 TF32", "教训：开关以实测为准"),
         ("路由翻转当成两边实现不同", "教训：参照先对 FP64 验"),
         ("精确轨迹的噪声只看一对", "教训：一对量不出散布"),
         ("把 ④ 当成了精确值", "教训：裁判也要被核对")]
    f = Fig(W, "一条路：八个绿色站点是留下来的八步，按先后排成两行，每站写一个数；四个红色虚框是走错的四步，从主线岔出去、又折回来，各写一条教训。"
               "第 4 步后岔出去一次：以为 FP32 dgrad，其实是 TF32；第 5 步后一次：路由翻转当成两边实现不同；第 7 步后两次：精确轨迹的噪声只看一对、把 ④ 当成了精确值。"
               "最后一站定案 ⑤，v5p、v7x 全部在噪声内，最差 %s 倍。" % N.FINAL_WORST_T)
    y0 = f.header("一路走来：八步留下，四步走错又折回",
                  "混元 3 截 4 层；绿站 ＝ 留下来的一步（只写一个数），红虚框 ＝ 走错的一步（只写教训），细节在图下",
                  [(GR, "留下来的一步"), (RD, "走错、折回")])
    PH = 560
    py = f.panel(0, y0, W, PH, "这一次排查的路线（实测）", BL)
    y1, y2 = py + 100, py + 340
    xs1 = [140, 440, 740, 1040]
    xs2 = [1040, 740, 440, 140]
    pos = [(x, y1) for x in xs1] + [(x, y2) for x in xs2]
    # 主线
    f.line(xs1[0], y1, xs1[-1], y1, GR, 4, arrow=False)
    f.path("M%d %d L1250 %d L1250 %d L%d %d" % (xs1[-1], y1, y1, y2, xs2[0], y2), GR, 4, arrow=False)
    f.line(xs2[0], y2, xs2[-1], y2, GR, 4, arrow=False)
    for k, ((x, y), (t1, t2)) in enumerate(zip(pos, G)):
        f.box(x - 18, y - 18, 36, 36, GR, GR, 18)
        f.t(x, y + 6, str(k + 1), "#fff", True, 16, "middle", big=True)
        if k < 4:                              # 第一行字在线上方，第二行字在线下方：两行之间留给岔路
            f.t(x, y - 50, t1, GR, True, 15, "middle")
            f.t(x, y - 28, t2, INK, True, 13, "middle")
        else:
            f.t(x, y + 42, t1, GR, True, 15, "middle")
            f.t(x, y + 64, t2, INK, True, 13, "middle")

    BWd, BHd = 290, 62

    def detour(bx, by, side, t1, t2, out_x=None, back_x=None):
        """第二行从右往左走：岔出去的那根在右（离上一站近），折回的那根在左、带箭头。"""
        L, R = bx - BWd / 2.0, bx + BWd / 2.0
        f.box(L, by, BWd, BHd, "#fce8e6", RD, 8, sw=1.6, dash="6 4")
        f.t(bx, by + 24, "✗ " + t1, RD, True, 14, "middle")
        f.t(bx, by + 47, t2, RD, size=13, anchor="middle")
        if side == "above":                   # 从第二行往上岔
            ox, bk = out_x or R - 40, back_x or L + 40
            f.line(ox, y2 - 4, ox, by + BHd, RD, 1.6, dash="5,4", arrow=False)
            f.line(bk, by + BHd, bk, y2 - 8, RD, 1.6, dash="5,4")
        elif side == "below":                 # 从第二行往下岔（从两站的字之间穿过）
            ox, bk = out_x, back_x
            f.line(ox, y2 + 4, ox, by, RD, 1.6, dash="5,4", arrow=False)
            f.line(bk, by, bk, y2 + 8, RD, 1.6, dash="5,4")
        else:                                 # 从右侧竖线往左岔
            f.line(1250, by + 14, R, by + 14, RD, 1.6, dash="5,4", arrow=False)
            f.line(R, by + BHd - 14, 1246, by + BHd - 14, RD, 1.6, dash="5,4")
    detour(1080, py + 120, "left", *X[0])
    detour(890, py + 222, "above", *X[1])
    detour(290, py + 222, "above", *X[2])
    detour(290, y2 + 110, "below", *X[3], out_x=305, back_x=275)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "四步走错，三步错在「参照」：开关、FP32 参照、精确轨迹都被当成了它名字说的那样", [])
    yb = f.src(yb + 10,
               "📌 每站的细节：① 数据原样回放、权重逐张量逐位、按实际生效的权重衰减；② 同设置 GPU 跑 10 份，偏移在参考档里，却是真偏差；③ 参数范数多涨的那截拆到组、拆到行；④ 两边都去掉「所有 logits 一起升降」那股力；⑤ 拿训练里的真张量对 FP64 拆输出头反向，TPU 第 1 步 %s；⑥ ② 生产近似 → ③ dgrad 换 FP32 → ④ 精确轨迹；⑦ 边传边乘的部分和不再在 BF16 里累加，对 ④ %s → 3.45 倍；⑧ 只修那一次长求和，10 对时几项 %s 倍待定，补到 45 对。" % (N.HEAD_TPU_S1_T, N.GOLDEN_R4_T, N.PEND10_T),
               "📌 走错的四步：「FP32 dgrad」偏 %s，比 BF16 的 %s 还大，一度判成换 dgrad 也不动；TPU 那份 FP32 参照里路由专家其实按 BF16 算，翻转是它的下游，撤回；精确轨迹只看一对扰动，像是比借来的尺子窄 %d 倍，五份 10 对一量同一量级；FP32 一次累加 12 万也偏，④ 自己少漂 %s，分段累加后两边精确轨迹重合。"
               % (tx(abs(N.HEAD_TF32_S1)), tx(abs(N.HEAD_BF16_S1)), N.EXACT_PAIR1_X, N.B_GPUEXACT_T),
               "📌 最终方案里改数值的主要是两处：TPU 的部分和留 FP32、GPU golden 的分段累加；其余几步都是把量法和参照弄对。各数都在出处台账里。")
    f.save("fig8-journey.svg", yb + 14)

def fig_switch():
    f = Fig(W, "一棵三问的决策树。第一问：这个开关修的是正确性吗，比如 NaN、语义写错？是，归修错类，生产也开。"
               "不是，第二问：打开以后 TPU 离 FP64 真值更近吗？不是，它只是照 GPU 多舍一次、换个舍入位置，归对齐专用，对齐时开、生产关。"
               "是，第三问：步时代价小于 1%% 吗？小于，生产也开；不小于，归取舍。"
               "叶子下面各举实例：对齐专用有强制舍入照 GPU、优化器迭代改到 BF16、四路相加照 GPU 的顺序；修错有 NaN 的修复；更准且便宜有部分和留 FP32、精确的除法开方；"
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
    leaf(460, py + 140, 470, GY, "对齐专用 → 对齐时开、生产关", ["照 GPU 强制舍成 BF16", "优化器一段迭代改到 BF16", "多舍一次 BF16、照 GPU 的舍入位置", "四路相加照 GPU 的顺序（＋%s，更准未证实）" % N.SW_MIX_COST])
    f.line(380, py + 178, 458, py + 178, GY2, 1.6); f.t(390, py + 170, "不是", GY, size=12)
    f.line(230, py + 204, 230, py + 290, GY2, 1.6); f.t(240, py + 254, "是", GY, size=12)
    ask(80, py + 292, "步时代价 ＜ %s？" % N.SW_COST_GATE)
    leaf(460, py + 280, 430, GR, "更准且便宜 → 生产也开", ["部分和留 FP32（v5p ＋%s）" % N.DFA_COST_V5P_T, "精确的除法、开方（＋%s）" % N.SW_ADAM_COST])
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
               "📌 步时：部分和 FP32（混元 3，100 步对比）v5p ＋%s、v7x ＋%s；四路相加改依次 ＋%s（第二次迁移；对 FP64 是否更准没证实，暂归对齐专用）；强制舍入 ＋%s（跨代对比）。"
               % (N.DFA_COST_V5P_T, N.DFA_COST_V7X_T, N.SW_MIX_COST, N.SW_ELIDE_COST),
               "⚠️ 1% 的界是本课取的口径（照同一实践的分类）；v7x 那一项按这个口径归「取舍」是本课推导（同一实践在小模型上把这个开关记作「更准且便宜」）；精确除法开方的 ＋" + N.SW_ADAM_COST + " 是跨代对比。另有不进这棵树的两类：按训练配方定的设置（例如路由偏置做不做零均值更新），和判法、工具、量法。")
    f.save("fig8-switch.svg", yb + 14)


def fig_ladder():
    f = Fig(W, "左右两块。左：一张格子，行是模型的几样属性（宽度和注意力形状、专家个数、专家宽度、层数、权重），列是三级：小模型、中等尺寸、真实权重截层；"
               "浅灰是跟小模型一样，蓝是跟真实模型一样，橙是改了、但两边都不像。中等尺寸换了宽度和注意力形状、专家宽度；真实权重截层又换了专家个数、专家宽度、权重，层数只截前几层。"
               "右：每一级第 1 步主 loss 两边各自对 CPU FP64 的差，对数轴。小模型两边都在 3.7e-6 以内；中等尺寸 GPU 偏高 1.42e-5、TPU 只有 2.4e-7，差在 GPU 一侧；"
               "真实权重截层 GPU 仍偏高 5.3e-5，TPU 偏低 9.9e-5，只关 base2 以后 TPU 到 1.2e-6。")
    y0 = f.header("每一级两边各自对 FP64：差冒出来的那一级翻了哪几行，嫌疑就在那几行",
                  "行 ＝ 模型的一样属性；列 ＝ 一级；颜色 ＝ 这一级的这一样跟谁一样；右 ＝ 第 1 步主 loss 两边各自对 FP64",
                  [(GY, "同小模型"), (BL, "同真实模型"), (OR, "改了，两边都不像")])
    ROWS = ["宽度、注意力形状", "专家个数", "专家宽度", "层数", "权重"]
    COLS = [("① 小模型", ["s", "s", "s", "s", "s"], BL),
            ("② 中等尺寸", ["r", "s", "m", "s", "s"], OR),
            ("③ 真实权重截层", ["r", "r", "r", "m", "r"], RD)]
    FILL = {"s": ("#f1f3f4", GY, "同小模型"), "r": ("#e8f0fe", BL, "同真实模型"), "m": ("#fef7e0", OR, "改了")}
    RH, X0, CW = 44, 200, 150
    PH = 70 + RH * len(ROWS) + 50
    py = f.panel(0, y0, W, PH, "三级，每级换了哪几样（第二次迁移）", BL)
    for j, (h, _, col) in enumerate(COLS):
        f.t(X0 + j * (CW + 10) + CW / 2.0, py + 28, h, col, True, 14, "middle")
    for i, rname in enumerate(ROWS):
        y = py + 44 + i * RH
        f.t(20, y + 27, rname, INK, True, 13.5)
        for j, (_, cells, _) in enumerate(COLS):
            fill, col, lab = FILL[cells[i]]
            changed = j > 0 and cells[i] != COLS[j - 1][1][i]
            edge = INK if changed else col
            f.box(X0 + j * (CW + 10), y + 4, CW, RH - 8, fill, edge, 4, sw=2.6 if changed else 1)
            f.t(X0 + j * (CW + 10) + CW / 2.0, y + 27, lab, col, size=12.5, anchor="middle")
    f.t(X0, py + 44 + len(ROWS) * RH + 22, "黑粗框 ＝ 这一级新换的", GY, True, 12.5)
    import math as _m
    RX = X0 + 3 * (CW + 10) + 30
    f.t(RX, py + 28, "第 1 步主 loss 对 CPU FP64（对数轴，标正负）", INK, True, 14)
    AX0, AX1 = RX + 90, W - 110

    def LX(v):
        return AX0 + (_m.log10(abs(v)) - (-7.0)) / (-4.0 - (-7.0)) * (AX1 - AX0)
    CG, CT = INK, PU
    rows = [("①", [("GPU", max(abs(x) for x in N.LAD_MINI_GPU_RANGE), "8 份都在 ±2.5e-6 内", CG),
                   ("TPU", N.LAD_MINI_TPU, "−3.7e-6（同一量级）", CT)]),
            ("②", [("GPU", N.LAD_MID[0], N.LAD_MID_T[0], CG), ("TPU", N.LAD_MID[1], N.LAD_MID_T[1], CT)]),
            ("③", [("GPU", N.LAD_REAL_GPU, N.LAD_REAL_GPU_T, CG), ("TPU", N.LAD_REAL_TPU[0], "−9.9e-5", CT)])]
    for k, (lev, bars) in enumerate(rows):
        yb0 = py + 48 + k * 74
        f.t(RX, yb0 + 30, lev, COLS[k][2], True, 15)
        for m, (side, v, lab, col) in enumerate(bars):
            y = yb0 + m * 28
            f.t(RX + 30, y + 15, side, col, True, 12.5)
            if k == 0 and side == "GPU":
                f.box(AX0, y + 2, max(LX(v) - AX0, 3), 18, "#e8eaed", GY2, 3, dash="4 3")
            else:
                f.box(AX0, y + 2, max(LX(v) - AX0, 3), 18, col, col, 3)
            f.t(LX(v) + 6, y + 16, lab, col, True, 12.5)
        if k == 2:
            f.box(LX(N.LAD_REAL_TPU[1]) - 1.5, yb0 + 28 + 2, 3, 18, "#fff", "#fff", 0)
    for e in (1e-7, 1e-6, 1e-5, 1e-4):
        f.line(LX(e), py + 44, LX(e), py + 48 + 3 * 74 - 8, LINE2, 1, arrow=False)
        f.t(LX(e), py + 48 + 3 * 74 + 8, "%.0e" % e, GY, size=11, anchor="middle")
    f.t(RX, py + 48 + 3 * 74 + 30, "深灰 GPU、紫 TPU；③ 紫条里的白线 ＝ 只关 base2 以后：−1.2e-6", INK, True, 12.5)
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "每一级尽量只换一类，而且两边各自对 FP64 看：差是谁的、从哪一级冒出来，一眼分开", [
        "GPU 的差第二级冒出来：宽度和注意力一换成真的就有（来源在查）。",
        "TPU 的差第三级才冒出来：定到权重那一行，真实权重的 logits 大，放大了 base2 常数舍成 BF16 的误差。",
    ])
    yb = f.src(yb + 10,
               "📌 实测，第 1 步主 loss 对 CPU 上 GPU 同一套代码的 FP64 整网前向：小模型 GPU 8 份 −2.46e-6 到 ＋4.7e-7，TPU 关 base2 −3.68e-6（开 −2.14e-6）；中等尺寸（真实宽度与注意力、随机权重）GPU ＋1.42e-5、TPU ＋2.4e-7，"
               "比 GPU 8 份第 1 步的极差（9.4e-6）还大；真实权重截前三层（真实宽度）GPU 4 份 ＋5.1e-5 到 ＋5.8e-5（条上标的是中位数 ＋5.3e-5；第二次迁移成绩单那张图用的是其中一份 ＋5.76e-5），TPU 开 base2 −9.87e-5、只关 base2 −1.24e-6。",
               "📌 中等尺寸 300 步对 GPU 8 份两两 28 对：%d 过、%d 不过、%d 待定（最差 %.2f N）；它的 1000 步和单层、单算子两级都没做。格子只写同谁一样，不写具体尺寸。" % (N.LADDER_MID_PASS + (N.LADDER_MID_WORST,)),
               "⚠️ 「GPU 一侧的偏高跟宽度有关」是从「第二级才出现」推的；GPU 那份偏高从哪一层起还没查。")
    f.save("fig8-ladder.svg", yb + 14)

fig_golden()
fig_decomp()
fig_journey()
fig_twolayer()
fig_regress()
fig_switch()
fig_ladder()
