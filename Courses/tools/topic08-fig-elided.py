# -*- coding: utf-8 -*-
r"""专题八 · §9.8 的重点展示：先编译、再扫描。

fig8-elided  回答：「该舍 BF16 没舍」的地方，整个训练步里到底有几处、在哪？
          Ⓐ 一处往返怎么认：多路残差分支输入那一处，代码写了 FP32 → BF16 → FP32（＝声明这里该舍），优化后 HLO 里两条 convert 都还在，
             可算均方那颗 kernel 的 LLO 里打包成 BF16 的指令 0 条 —— 声明了，但被省掉。同一处补上 reduce_precision：256 条，真舍。
             ⛔ 别换成收拢处那颗的 688 条：那 688 条来自两路相加之后补的另一次舍入，原来那两条 convert 补完照样被省。
          Ⓑ 把现行对齐配置的整个训练步离线编译一次，按「同一颗融合里 f32→bf16 又转回 f32」扫：往返 272 处（使用次数，不是 kernel 数），
             按位置归类；其中整颗只含往返的 kernel 41 颗，LLO 打包全 0 条；关省舍入的负对照里，同类 kernel 35 颗，没有一颗是 0 条。
             iHC、流梯度、hc_head 这几处已经用 reduce_precision 补上，不在 272 里。
          数据：topic08_numbers 的 EL_* 块（实测；整步扫描的按类数字取自扫描结果文件，合入前口径，见 numbers 注释）。
          ⛔ 刻意没画：41 颗 kernel「分三类」—— 注意力输入梯度合流、前向几处都落在同时含真舍入的混合 kernel 里，
             LLO 条数没法单独归给它们；它们的证据是 HLO 规则加最小复现确立的机制。别把右块的 41 个方块按三类重新上色。
"""
from topic03_draw import Fig, BL, OR, GR, GY, INK, GY2, LINE
import topic08_numbers as N

W = 1400
P = N.EL_PROBE
U = N.EL_USES
assert N.EL_TOTAL == 272 and N.EL_K_TOTAL == 41 and P["packs_base"] == 0


def fig_elided():
    f = Fig(W, "两块。左：多路残差分支输入那一处的三层。代码把 4 路流的加权和从 FP32 转成 BF16、再按 FP32 进归一化，这是在声明这里该舍一次；"
               "编译器优化后的 HLO 里那对 convert 都还在；可到后端指令 LLO，算均方那颗 kernel 里打包成 BF16 的指令一条都没有，平方用的还是没舍的 FP32。"
               "同一处补上 reduce_precision 以后，这颗 kernel 有 256 条打包，真舍了。"
               "右：把现行对齐配置的整个训练步离线编译一次，扫所有这样的往返，一共 272 处：每个微批的权重梯度 75，"
               "分支输入的梯度合流 70，前向几处 98，其余 20 处优先级低，索引器掩码 9 处无害。"
               "整颗只含往返的 kernel 有 41 颗，后端打包指令全是 0；关掉省舍入再编，同类 kernel 35 颗，没有一颗是 0。"
               "已经补上的 55 处不算在里面。")
    y0 = f.header("先编译、再扫描：代码里的往返 convert，就是「这里该舍」的声明",
                  "第二次迁移，小尺寸，TPU v5p 离线编译（省舍入默认开）；左：一处怎么认，右：整个训练步扫一遍",
                  [(OR, "写了舍、被省掉"), (GR, "已补上 / 真舍"), (GY2, "优先级低"), ("#bdc1c6", "无害")])
    PH = 470
    LW = 560
    # ── Ⓐ ───────────────────────────────────────────────
    py = f.panel(0, y0, LW, PH, "Ⓐ 一处往返：声明了，但被省掉", OR)
    LX = 20
    BX = 128
    rows_y = [py + 26, py + 140, py + 262]
    labs = [("代码", "写了往返"), ("优化后 HLO", "convert 都在"), ("LLO 指令", "打包 0 条")]
    for (a, b), y in zip(labs, rows_y):
        f.t(LX, y + 26, a, INK, True, 14)
        f.t(LX, y + 48, b, GY, size=12)
    y = rows_y[0]
    xs = [BX, BX + 140, BX + 280]
    for i, (lab, col) in enumerate([("4 路加权和", BL), ("转 BF16", OR), ("转回 FP32", BL)]):
        f.box(xs[i], y + 10, 110, 38, "#fff", col, 8, sw=1.6)
        f.t(xs[i] + 55, y + 34, lab, col, True, 13.5, "middle")
        if i < 2:
            f.line(xs[i] + 110, y + 29, xs[i] + 138, y + 29, GY, 1.4)
    f.t(BX, y + 74, "多路残差的分支输入。往返不改数学，写它就是说「这里要舍」", INK, size=12)
    y = rows_y[1]
    f.box(BX - 10, y + 2, 400, 70, "none", GY2, 8, sw=1.2, dash="5 4")
    f.t(BX + 384, y + 18, "算均方那颗融合", GY, size=11.5, anchor="end")
    f.box(BX + 10, y + 26, 160, 34, "#fff", OR, 6, sw=1.4)
    f.t(BX + 90, y + 48, "convert → bf16", OR, True, 12.5, "middle", mono=True)
    f.line(BX + 170, y + 43, BX + 200, y + 43, GY, 1.4)
    f.box(BX + 202, y + 26, 160, 34, "#fff", BL, 6, sw=1.4)
    f.t(BX + 282, y + 48, "convert → f32", BL, True, 12.5, "middle", mono=True)
    f.t(BX, y + 96, "这一层看不出毛病：两条都在，HLO 读着是舍了", INK, size=12.5)
    y = rows_y[2]
    f.box(BX, y + 8, 190, 92, "#fff", OR, 8, sw=1.8)
    f.t(BX + 95, y + 30, "照原样", OR, True, 13, "middle")
    f.t(BX + 95, y + 74, "%d" % N.EL_IHC_PACKS[0], OR, True, 38, "middle", big=True)
    f.t(BX + 95, y + 94, "条打包", GY, size=12, anchor="middle")
    f.box(BX + 210, y + 8, 190, 92, "#fff", GR, 8, sw=1.4)
    f.t(BX + 305, y + 30, "补上 reduce_precision", GR, True, 12.5, "middle")
    f.t(BX + 305, y + 74, "%d" % N.EL_IHC_FIXED, GR, True, 38, "middle", big=True)
    f.t(BX + 305, y + 94, "条打包", GY, size=12, anchor="middle")
    f.t(LX, y0 + PH - 40, "同一个值：均方那颗用没舍的，归一化那颗从内存读舍过的。", INK, size=12.5)
    f.t(LX, y0 + PH - 16, "规律：往返留在一颗融合里就不舍；写回内存交给下一颗的才真舍。", GY, size=12)
    f._pan = None
    # ── Ⓑ ───────────────────────────────────────────────
    RX = LW + 20
    RW = W - RX
    py2 = f.panel(RX, y0, RW, PH, "Ⓑ 整个训练步扫一遍：往返 %d 处，按位置分" % N.EL_TOTAL, OR)
    bars = [("① 分支输入的梯度合流", U["branchin"], OR, "反向 · 几份梯度在 FP32 里相加，典型是注意力的输入"),
            ("② 每个微批的权重梯度 dW", U["wgrad"], OR, "反向"),
            ("③ 前向几处", N.EL_FWD, OR, "线性层输出 %d · 输出门 %d · 专家 %d" % (U["fwd_linear"], U["fwd_attn_gate"], U["fwd_expert"])),
            ("其余（专家反向、norm）", N.EL_MINOR, GY2, "专家反向 %d · norm 输入梯度 %d · norm 输出 %d" % (U["expert_bwd"], U["norm_dgrad"], U["fwd_norm"])),
            ("索引器掩码（0 和 −∞，转了也精确）", U["idx_mask"], "#bdc1c6", "无害")]
    LB = RX + 16
    B0 = RX + 300
    B1 = W - 70
    vmax = 110.0

    def BXv(v):
        return B0 + v / vmax * (B1 - B0)
    for i, (lab, v, col, note) in enumerate(bars):
        yy = py2 + 22 + i * 44
        f.t(LB, yy + 19, lab, col if col in (OR, GR) else GY, True, 13)
        f.box(B0, yy + 4, BXv(v) - B0, 22, col, col, 4)
        f.t(BXv(v) + 8, yy + 20, "%d" % v, col if col in (OR, GR) else GY, True, 13)
        if note:
            f.t(B0, yy + 40, note, GY, size=11.5)
    # 剩下的范围：三根橙条括起来
    f.path("M %d %d h -6 v %d h 6" % (LB - 4, py2 + 24, 3 * 44 - 6), OR, 2, arrow=False)
    f.t(W - 24, py2 + 22 + 44 + 20, "先查的范围：%d 处" % N.EL_LEFT, OR, True, 13.5, "end")
    # 已补上的 chips
    yc = py2 + 22 + 5 * 44 + 4
    f.t(LB, yc + 18, "已补上 %d 处，不计入：" % N.EL_COVERED, GR, True, 13)
    cx = LB + 170
    for lab in ("多路残差、流梯度、收拢处 %d" % N.EL_COVERED_MHC, "norm、线性层 %d" % (N.EL_COVERED - N.EL_COVERED_MHC)):
        w = 26 + 13.0 * len(lab)
        f.box(cx, yc + 2, w, 24, "#fff", GR, 12, sw=1.4)
        f.t(cx + w / 2, yc + 19, lab, GR, True, 12.5, "middle")
        cx += w + 10
    # 41 颗只含往返的 kernel
    yk = yc + 46
    f.line(RX + 12, yk - 6, W - 12, yk - 6, LINE, 1, arrow=False)
    f.t(LB, yk + 16, "整颗只含往返的 kernel %d 颗，LLO 打包指令：全是 0" % N.EL_K_TOTAL, INK, True, 13.5)
    sx = LB
    for key, col, lab in (("wgrad", OR, "dW"), ("norm", GY2, "norm"), ("idx_mask", "#bdc1c6", "掩码")):
        n = N.EL_K[key]
        for j in range(n):
            f.box(sx + j * 14, yk + 28, 10, 10, col, col, 2)
        f.t(sx, yk + 58, "%s %d" % (lab, n), col if col in (OR, GR) else GY, True, 12)
        sx += max(n * 14, 13 * len(lab) + 34) + 16
    f.t(LB, yk + 86, "负对照：关掉省舍入再编，同类 kernel %d 颗，打包 0 条的 %d 颗" % (N.EL_K_NOEXC, N.EL_K_NOEXC_ZERO), GY, size=12)
    f.t(LB, y0 + PH - 14, "三类里 LLO 能逐颗核的只有 dW；合流、前向几处跟真舍混在一颗 kernel 里，按同一规则推断", GY, size=11.5)
    f._pan = None
    yb = f.band(y0 + PH + 20, "warn", "不用一处处猜：编一次、扫一遍，同类位置全列出来，问题的范围就划出来了", [
        "先查三根橙条：① 分支输入的梯度合流、② 每个微批的 dW、③ 前向几处，共 %d 处；灰条优先级低，没排除。" % N.EL_LEFT,
        "只找得到「写了往返却被省掉」这一类；GPU 那边舍不舍要对照 GPU 代码；列出来的是候选，哪类真造成尾巴，还在上卡验证。",
    ])
    yb = f.src(yb + 10,
               "📌 Ⓐ 实测：多路残差「4 路加权和 → 转 BF16 → 归一化」的最小复现（MaxText 真模块），CPU 上为 TPU v5p 离线编译（运行库 0.0.46，省舍入默认开），读优化后 HLO 与后端指令（LLO）；训练步的优化后 HLO 里同一对 convert 也都在。"
               "打包指令指后端指令里把 FP32 打包成 BF16 的那一类。关掉省舍入，两种写法都真舍。收拢处那一段（另一份最小复现）两路梯度各转 BF16 的两条 convert 也是 0 条打包。",
               "📌 Ⓑ 实测：现行对齐配置（多路残差分支输入与流、流梯度、收拢处都照 GPU 舍）的训练步，v5p-16 离线编译。"
               "272 是 convert(f32→bf16) 在同一计算里又被转回 f32 的使用次数（同一行代码在 scan 外的层、MTP 块、前向、重算、反向里各记一次），按源码行与形状归类（经验规则）；用 reduce_precision 补上、或输入本来就在 BF16 网格上的往返不计。"
               "「只含往返」＝ 这颗 kernel 有往返、没有真舍、没有 reduce_precision。",
               "⚠️ 少舍一次，单看这一处离精确值更近（推断，没量过）；可省不省取决于融合怎么切，同一个值可能这边舍了、那边没舍。小尺寸、单一配置；别的模型要重编重扫。")
    f.save("fig8-elided.svg", yb + 14)


fig_elided()
