# -*- coding: utf-8 -*-
r"""专题九 · 第九、十节的四张图。

fig9-golden   回答：最终那份 golden 是怎么一步步来的？每一步加了什么、换来了什么？
              数字：同一个被测（v5p，FP32 累加开关打开前的配置）对三份 golden 的参数范数带符号均值 ÷ 同一把尺子 N（实测）。
fig9-decomp   回答：TPU 对 GPU 生产 golden 的参数范数差，各有几成是谁的？—— 59% / 29% / 12%（实测拆账）。
fig9-twolayer 回答：golden 自己有系统偏差时怎么验收？—— 分两关：第一关对精确轨迹必须过，第二关对生产 golden 写成已解释差异。
fig9-regress  回答：验收过了以后，每天合进来的代码怎么守？—— 先要求逐位；不逐位按 PR 的数值声明分流。
"""
from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE, LINE2, PU
import topic09_numbers as N

W = 1400
NOISE = N.GPU_RERUN_PN
# R3：颜色 ＝ 全课约定（GPU 生产 golden 深灰、精确轨迹紫）；卡片上不提前用「第一关／第二关」，关卡名留到 fig9-twolayer 再出
STEPS = [("① 起点", "框架默认配置", ["BF16，不重算", "TF32 开着"], None, GY2, "归因参照"),
         ("② 生产近似", "＋ 整层重算 ＋ 路由回放", ["模仿生产开重算", "跟①比：各项都在 GPU 噪声内"], 2.93e-6, GY, "生产 golden：差要写清来由"),
         ("③ 只修一处", "＋ 输出头 dgrad 换精确", ["解释了 loss 偏移的 73%", "参数范数差的 59%", "第一关的尺子取自这组重跑"], 1.25e-6, OR, "归因证据"),
         ("④ 精确轨迹", "＋ BF16 权重、全程 FP32", ["真关 TF32；这就是", "BF16 训练的精确轨迹"], 4.36e-7, PU, "最终 golden")]
R = [None if st[3] is None else st[3] / NOISE for st in STEPS]
assert abs(R[1] - 23.6) < 0.1 and abs(R[2] - 10.1) < 0.1 and abs(R[3] - 3.52) < 0.01, R


def fig_golden():
    f = Fig(W, "四张卡片从左到右：第一份是起点，框架默认配置；第二份加上整层重算和路由回放，跟生产一样；第三份再把输出头 dgrad 换成精确乘法；"
               "第四份再改成 BF16 权重、全程 FP32 计算，就是精确轨迹，最终 golden。"
               "卡片下面是同一个 TPU 被测对它的参数范数差，按同一把尺子 N 算倍数：对第二份 24 倍，对第三份 10 倍，对第四份 3.5 倍。")
    y0 = f.header("最终 golden 是一步步改出来的　——　<tspan font-weight=\"700\">每一步只加一样东西，各自留下证据</tspan>",
                  "卡片 ＝ 一份 golden（GPU 上跑的标准答案）；下面的柱子 ＝ 同一个 TPU 被测对它的参数范数差是噪声的几倍（同一把尺子）",
                  [(GY2, "起点"), (GY, "生产近似"), (OR, "只修一处"), (PU, "精确轨迹"), (RD, "不过：超过 2×N")])
    PH = 420
    py = f.panel(0, y0, W, PH, "混元 3 的四份 golden（实测）", BL)
    CW, GAP = 300, 46
    for i, ((name, add, lines, v, col, role), r) in enumerate(zip(STEPS, R)):
        x = 30 + i * (CW + GAP)
        f.box(x, py + 24, CW, 192, "#fff", col, 10, sw=2)
        f.box(x, py + 24, CW, 6, col, col, 3)
        f.t(x + 18, py + 66, name, col, True, 20, big=True)
        f.t(x + 18, py + 96, add, INK, True, 14)
        for k, ln in enumerate(lines):
            f.t(x + 18, py + 124 + k * 22, ln, OR if k == 2 else GY, k == 2, 13)
        f.t(x + 18, py + 204, role, col, True, 13)
        if i < 3:
            f.line(x + CW + 6, py + 108, x + CW + GAP - 6, py + 108, GY2, 2)
        if r is not None:
            L = min(r / 24.0 * 250, 250)
            f.box(x + 18, py + 250, L, 30, RD if r > 2 else GR, RD if r > 2 else GR, 4)
            # ⭐ R3：每根柱子补一条 2×N 的参考短线，「3.5 倍还不过」一眼能看出来
            x2 = x + 18 + 2 / 24.0 * 250
            f.line(x2, py + 242, x2, py + 288, INK, 2, arrow=False)
            f.t(x2, py + 236, "2×N", INK, True, 12, "middle")
            f.t(x + 18, py + 306, "TPU 对它：%.1f × N" % r, RD, True, 15)
            if i == 3:
                f.t(x + 18, py + 328, "被测：FP32 累加开关打开前的那一版 TPU", GY, size=12)
        else:
            f.t(x + 18, py + 270, "（不重算，跟生产配置不一样，不拿来验收）", GY, size=13)
    f.t(30, py + 350, "柱子：同一个 TPU 被测的参数范数带符号均值 ÷ N（N ＝ ③ 那组 GPU 重跑 5 份两两 10 对的最大值）；竖线 ＝ 2×N，超过就判不过", GY, size=13)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "每往精确值靠一步，TPU 的「偏差」就缩一截：24 倍 → 10 倍 → 3.5 倍", [
        "缩掉的部分都查清了是 GPU golden 自己的偏差，所以可以写成「已解释差异」；剩下的那截（开关前 3.52 倍，当前 3.45 倍）在 TPU 一侧（还含两边切法不同），还在查。",
        "代价：每一份 golden 都要单独跑满 1000 步、存好数据；改了什么、为什么改、证据在哪，都要进一张出处总表。",
    ])
    yb = f.src(yb + 10,
               "📌 实测：被测是 v5p 上生产配置 1000 步（FP32 累加开关打开之前的那一版），param_norm 带符号均值对 ② 2.93e-6、对 ③ 1.25e-6、对 ④ 4.36e-7；N ＝ 1.24e-7。开关打开后的当前配置对 ④ 是 4.27e-7（3.45 倍）。",
               "📌 实测：③ − ② 在第 301–1000 步 loss 上是 −9.8e-5，占 TPU 对 ② 偏移（−1.34e-4）的 73%；第 1000 步参数范数 ＋1.24e-2，占 ＋2.12e-2 的 59%。")
    f.save("fig9-golden.svg", yb + 14)


def fig_decomp():
    f = Fig(W, "一根长条代表 TPU 对 GPU 生产 golden 第 1000 步的参数范数差 +0.0212，拆成三段："
               "输出头 dgrad 的截断 +0.0124，59%；GPU 其余的 BF16 误差 +0.0062，29%；TPU 一侧对精确轨迹 +0.0026，12%，里面含两边切分方式不同。"
               "下面一根是第 301 到 1000 步的 loss：TPU 对生产 golden 低 1.34e-4，其中 73% 是 dgrad 截断，其余没有拆；精确轨迹比 TPU 还低 7.6e-5。")
    y0 = f.header("拆账：参数范数的差里，近九成是 GPU golden 自己的",
                  "长条全长 ＝ TPU 对 GPU 生产 golden 的差；每一段 ＝ 由哪一处造成（实测拆分）",
                  [(RD, "输出头 dgrad 截断（GPU）"), (OR, "GPU 其余 BF16 误差"), (BL, "TPU 一侧"), (LINE2, "其余（没拆）")])
    PH = 300
    py = f.panel(0, y0, W, PH, "第 1000 步参数范数差 ＋%.4f（实测；对 GPU 生产 golden 5 份的均值，生产近似配置）" % N.DECOMP_TOTAL, BL)
    X0, BW = 40, 1300
    x = X0
    cols = [RD, OR, BL]
    for k, ((lab, v), c, sh) in enumerate(zip(N.DECOMP, cols, N.DECOMP_SHARE)):
        L = v / N.DECOMP_TOTAL * BW
        f.box(x, py + 40, L, 56, c, c, 4)
        f.t(x + L / 2, py + 76, "%.0f%%" % (sh * 100), "#fff", True, 18, "middle", big=True)
        if k == 2:      # 最后一段窄，标签靠右对齐，别顶出面板
            f.t(X0 + BW, py + 124, "TPU 一侧（含两边切分方式不同）", c, True, 14, "end")
            f.t(X0 + BW, py + 146, "＋%.4f" % v, c, size=13, anchor="end")
        else:
            f.t(x + 6, py + 124, lab, c, True, 14)
            f.t(x + 6, py + 146, "＋%.4f" % v, c, size=13)
        x += L
    # loss
    y = py + 190
    f.t(X0, y, "第 301–1000 步 loss：TPU 比生产 golden 低 %.2e，其中 %.0f%% 是 dgrad 截断；精确轨迹比 TPU 还低 7.6e-5" %
        (abs(N.LOSS_TPU_VS_PROD_TAIL), N.LOSS_FIXHEAD_MINUS_PROD / N.LOSS_TPU_VS_PROD_TAIL * 100), INK, True, 14)
    L1 = abs(N.LOSS_TPU_VS_PROD_TAIL) / abs(N.LOSS_EXACT_MINUS_PROD) * BW * 0.85
    L2 = abs(N.LOSS_FIXHEAD_MINUS_PROD) / abs(N.LOSS_EXACT_MINUS_PROD) * BW * 0.85
    L3 = BW * 0.85
    f.box(X0, y + 16, L3, 14, "#f3e8fd", PU, 3)
    # ⭐ R3：TPU 细条在红段之外剩的 27% 原来是浅蓝，会被读成「loss 上 TPU 自己占 27%」→ 改中性灰，标「其余（没拆）」
    f.box(X0, y + 34, L1, 14, LINE2, GY2, 3)
    f.box(X0, y + 34, L2, 14, RD, RD, 3)
    f.t(X0 + L2 + (L1 - L2) / 2, y + 45, "其余（没拆）", GY, True, 11, "middle")
    f.t(X0 + L3 + 8, y + 28, "精确轨迹 −2.09e-4", PU, True, 12)
    f.t(X0 + L1 + 8, y + 46, "TPU −1.34e-4：红 ＝ dgrad 截断 −9.8e-5，灰 ＝ 其余", BL, True, 12)
    f.t(X0, y + 70, "细条长度 ＝ 比 GPU 生产 golden 低多少；loss 上精确轨迹走得比 TPU 还远", GY, size=12)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "第二关写成「已解释差异」：量级、由哪一处造成、证据在哪，一样不少", [
        "loss 上 TPU 落在 GPU 生产 golden 和精确值之间 —— 不是 TPU 偏离了 GPU，是 TPU 更接近精确值。",
        "代价：剩下 12%（＋0.0026）在 TPU 一侧（含两边切分方式不同），第一关还没过，要接着查是哪一处 BF16 误差。",
    ])
    yb = f.src(yb + 10,
               "📌 实测：TPU 对生产 golden（整层重算 ＋ 路由回放，5 份均值）第 1000 步 param_norm ＋2.12e-2；只修 dgrad 的那份对生产 golden ＋1.24e-2；精确轨迹对只修 dgrad 的 ＋0.62e-2；TPU 对精确轨迹 ＋0.26e-2。",
               "📌 实测：第 301–1000 步 loss 相对生产 golden 均值：只修 dgrad 的 −9.8e-5，精确轨迹 −2.09e-4，TPU −1.34e-4；TPU 对精确轨迹 ＋7.6e-5（生产 golden 各份标准差 2.6e-5）。")
    f.save("fig9-decomp.svg", yb + 14)


GATE1_PN = N.ACC[0][3]                 # 第一关正式判定：param_norm 带符号均值 ÷ ③ 那组 GPU 重跑的 N
assert N.ACC[0][0].startswith("param_norm") and GATE1_PN == 3.45


def fig_twolayer():
    f = Fig(W, "上一层：第一关，必须过。TPU 对精确轨迹那一份比；尺子例外地借 ③ 那组 GPU 重跑（生产近似＋输出头 dgrad 换精确）5 份两两 10 对的散布，因为精确轨迹自己是确定性的，量不出噪声。"
               + ("旁边另做两份交叉核对，都只报不判：一份尺子换成精确轨迹 5 份（基线＋4 份扰动版）两两 10 对，参数范数 %.2f 倍、loss 全程 %.2f 倍、梯度范数 %.2f 倍超过 2，loss 尾段 %.2f 倍待定；"
               "另一份换成 TPU 自己的执行噪声 28 对，只有参数范数 %.2f 倍超过 2，loss 全程 %.2f 倍、尾段 %.2f 倍待定，梯度范数 %.2f 倍过。"
               % (N.XCHK_SELF["param_norm"], N.XCHK_SELF["loss 全程"], N.XCHK_SELF["grad_norm"], N.XCHK_SELF["loss 尾段"],
                  N.XCHK_TPU_PN, N.XCHK_TPU_LOSS, N.XCHK_TPU_TAIL, N.XCHK_TPU_GN))
               + "下一层：第二关，已解释差异。TPU 对 GPU 生产 golden 的差不判过不过，写清量级、哪一部分由哪个已查清的 GPU 偏差解释、证据在哪。")
    y0 = f.header("golden 自己有偏差时：分两关验收",
                  "上 ＝ 必须过的那一关；下 ＝ 只写清楚、不判过不过的那一关；箭头 ＝ 谁对谁比",
                  [(PU, "第一关：对精确轨迹"), (GY, "第二关：对 GPU 生产 golden"), (BL, "TPU 被测"), (OR, "尺子：③ 那组 GPU 重跑")])
    PH = 400
    py = f.panel(0, y0, W, PH, "混元 3 的验收协议", PU)

    def node(x, y, w, h, title, sub, col, fill="#fff"):
        f.box(x, y, w, h, fill, col, 10, sw=2)
        f.t(x + w / 2, y + 30, title, col, True, 16, "middle")
        if sub:
            f.t(x + w / 2, y + 54, sub, GY, size=13, anchor="middle")
    node(560, py + 150, 280, 76, "TPU 被测", "生产配置，1000 步", BL)
    node(80, py + 30, 360, 76, "精确轨迹（一份）", "BF16 权重、全程 FP32", PU)
    node(960, py + 30, 360, 76, "尺子：③ 那组 GPU 重跑", "生产近似＋精确 dgrad · 5 份两两 10 对", OR)
    # R4 一致性第 15 条：讲义和课件都说「两份交叉核对」，原图只画了 TPU 执行噪声那一份 → 两份都画，共一句「只报不判」
    XS = N.XCHK_SELF
    for k, (yx, t1, t2) in enumerate([
            (py + 112, "交叉核对一：精确轨迹 5 份（基线＋4 份扰动版）· 10 对",
             "param_norm %.2f× · loss %.2f× / 尾段 %.2f× · grad_norm %.2f×" % (XS["param_norm"], XS["loss 全程"], XS["loss 尾段"], XS["grad_norm"])),
            (py + 174, "交叉核对二：TPU 执行噪声 28 对", "param_norm %.2f× · loss %.2f× / 尾段 %.2f× · grad_norm %.2f×" % (N.XCHK_TPU_PN, N.XCHK_TPU_LOSS, N.XCHK_TPU_TAIL, N.XCHK_TPU_GN))]):
        # R18：橙色「量出 N」箭头原来切过核对一的左上角 → 两个虚线框整体右移 30 px
        f.box(922, yx, 456, 54, "#fff", GY2, 10, sw=1.6, dash="5 4")
        f.t(1150, yx + 22, t1, GY, True, 14, "middle")
        f.t(1150, yx + 43, t2, GY, size=12, anchor="middle")
        f.line(922, yx + 27, 842, py + 176 + 22 * k, GY2, 1.4, dash="5 4")
    node(80, py + 270, 360, 76, "② GPU 生产 golden", "整层重算 ＋ 路由回放（不是尺子）", GY)
    node(960, py + 270, 360, 76, "已解释差异", "量级 ＋ 由谁造成 ＋ 证据", GY)
    f.elbow(560, py + 175, 440, py + 68, PU, 2.2)
    f.t(470, py + 132, "第一关：必须过", PU, True, 15)
    f.t(430, py + 154, "正式判定：param_norm %.2f×N，不过" % GATE1_PN, RD, True, 13, "end")
    f.line(960, py + 68, 842, py + 165, OR, 1.6)
    f.t(850, py + 92, "量出 N", OR, True, 13)      # R3：原来被斜箭头穿过 → 挪到箭头左上方
    f.t(1150, py + 250, "× ＝ 被测 ÷ 该尺子的 N；超过 2：核对一 3 项，核对二仅 param_norm", GY, size=13, anchor="middle")
    f.elbow(560, py + 205, 440, py + 308, GY, 2.2)
    f.t(470, py + 268, "第二关：不判", GY, True, 15)
    f.line(442, py + 308, 958, py + 308, GY, 1.6)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "噪声尺子量的是「重跑会差多少」，量不出 golden 自己的系统偏差 —— 所以要分两关", [
        "精确轨迹重跑逐位相同，自己量不出噪声；正式判定事先定好借 ③ 那组 GPU 重跑的散布。精确轨迹扰动出来的五份散布（%s）跟它同一量级。" % ("%.2e" % N.EXACT_SELF_PN).replace("e-0", "e-"),
        "代价：例外要写在检查目录的文件头；两份交叉核对换别的尺子并排报，正式判定以 GPU 那把为准。",
    ])
    yb = f.src(yb + 10,
               "📌 实测：param_norm 带符号均值的噪声上限，精确轨迹扰动出来的五份两两 10 对 %s，③ 那组 GPU 重跑 10 对 %s，同一量级；正式判定按事先写定的协议借后者，验收中途不换尺子。"
               % (("%.2e" % N.EXACT_SELF_PN).replace("e-0", "e-"), ("%.2e" % N.GPU_RERUN_PN).replace("e-0", "e-")),
               "📌 实测：v5p 当前配置第一关 param_norm %.2f 倍不过、loss 两项和梯度范数待定；交叉核对一（噪声换成精确轨迹 5 份（基线＋4 份扰动版）两两 10 对）param_norm %.2f 倍不过、loss 全程 %.2f 倍不过、尾段 %.2f 倍待定、梯度范数 %.2f 倍不过；交叉核对二（TPU 执行噪声 28 对）param_norm %.2f 倍不过、loss 全程 %.2f 倍和尾段 %.2f 倍待定、梯度范数 %.2f 倍过；v7x param_norm 3.53 倍不过。"
               % (GATE1_PN, N.XCHK_SELF["param_norm"], N.XCHK_SELF["loss 全程"], N.XCHK_SELF["loss 尾段"], N.XCHK_SELF["grad_norm"], N.XCHK_TPU_PN, N.XCHK_TPU_LOSS, N.XCHK_TPU_TAIL, N.XCHK_TPU_GN))
    f.save("fig9-twolayer.svg", yb + 14)


def fig_regress():
    f = Fig(W, "每天的流程。训练代码的每个改动都要写一栏数值影响声明，然后跑一轮回归，对上一次全部通过的 TPU 运行比。"
               "逐位相同就通过，指针前移。不逐位就看声明：写了数学不变的，退回；执行级的改动或升级依赖，用 TPU 自己的执行噪声（TPU 28 对）判；"
               "数学级的改动，要对起点那份 golden（①）不变差（这是历史门，待换成 ④ 精确轨迹），必要时回到验收的某一级重做。")
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
    node(1080, py + 40, 300, 66, "数学不变 → 退回", "没写声明的也按这条处理", RD, "#fce8e6")
    node(1080, py + 130, 300, 66, "执行级 / 升级依赖", "容差门：TPU 执行噪声（TPU 28 对）", OR, "#fef7e0")
    node(1080, py + 220, 300, 76, "数学级", "对齐门：对起点那份 golden（①）不变差", PU, "#f3e8fd")
    f.t(1230, py + 290, "（历史门，待换成 ④）", PU, size=12, anchor="middle")
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
               "📌 实测（v5p，8 颗芯片）：T0 档 20 步，一次约 10 分钟（提交到取回）；T1 档 1000 步，约 25 分钟。首个 LKG 同一提交双跑逐位；「数学不变」复测逐位通过，故意改一个开关被拦下。",
               "📌 首次标定在 FP32 累加开关打开之前的配置上做；开关打开后，上一次全过的运行同一提交双跑逐位、负责人批准后重建，7 个扰动臂在新配置下重跑全过，TPU 28 对重新标定（正式）；第三项检查目前只判了 loss。扰动臂要过三项检查：臂自己重跑逐位；配置、版本、编译出的 HLO 计数等不变；第 1 步梯度和 1000 步 loss 对基线的差不大于 GPU 自噪声的上限。标定绑死平台、档位、训练配置、镜像和依赖版本，任何一样变了就重标。")
    f.save("fig9-regress.svg", yb + 14)


fig_golden()
fig_decomp()
fig_twolayer()
fig_regress()
