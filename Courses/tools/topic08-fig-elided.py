# -*- coding: utf-8 -*-
r"""专题八 · §9.7 的重点展示：先编译、再扫描。

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
Fig.BAND_KEEP_LINES = 0   # 锤炼阶段（2026-11-14）：图上落点带只留一行粗体结论，其余进图下折叠
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
    py = f.panel(0, y0, LW, PH, "左：一处往返（多路残差分支输入，退回没补时的写法）：声明了，但被省掉", OR)
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
    f.t(LX, y0 + PH - 16, "规律：往返留在一颗融合里就不舍；写回内存、喂给矩阵乘的才真舍。", GY, size=12)
    f._pan = None
    # ── Ⓑ ───────────────────────────────────────────────
    RX = LW + 20
    RW = W - RX
    py2 = f.panel(RX, y0, RW, PH, "右：整个训练步扫一遍，往返 %d 处，按位置分" % N.EL_TOTAL, OR)
    bars = [("A 分支输入的梯度合流", U["branchin"], OR, "反向 · 几份梯度在 FP32 里相加，典型是注意力的输入"),
            ("B 每个微批的权重梯度 dW", U["wgrad"], OR, "反向"),
            ("C 前向几处", N.EL_FWD, OR, "线性层输出 %d · 输出门 %d · 专家 %d" % (U["fwd_linear"], U["fwd_attn_gate"], U["fwd_expert"])),
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
    yb = f.band(y0 + PH + 20, "warn", "不用一处处猜：编一次、扫一遍，同类位置全列出来，就有了先查哪儿的清单", [
        "先查三根橙条 A、B、C，共 %d 处；灰条优先级低，没排除。" % N.EL_LEFT,
        "只找得到「写了往返却被省掉」这一类；GPU 那边舍不舍要对照 GPU 代码。",
    ])
    yb = f.src(yb + 10,
               "📌 左块实测：多路残差「4 路加权和 → 转 BF16 → 归一化」的最小复现（MaxText 真模块），CPU 上为 TPU v5p 离线编译（运行库 0.0.46，省舍入默认开），读优化后 HLO 与后端指令（LLO）；训练步的优化后 HLO 里同一对 convert 也都在。"
               "打包指令指后端指令里把 FP32 打包成 BF16 的那一类。关掉省舍入，两种写法都真舍。收拢处那一段（另一份最小复现）两路梯度各转 BF16 的两条 convert 也是 0 条打包。",
               "📌 右块实测：现行对齐配置（多路残差分支输入与流、流梯度、收拢处都照 GPU 舍）的训练步，v5p-16 离线编译。"
               "272 是 convert(f32→bf16) 在同一计算里又被转回 f32 的使用次数（同一行代码在 scan 外的层、MTP 块、前向、重算、反向里各记一次），按源码行与形状归类（经验规则）；用 reduce_precision 补上、或输入本来就在 BF16 网格上的往返不计。"
               "「只含往返」＝ 这颗 kernel 有往返、没有真舍、没有 reduce_precision。",
               "⚠️ 省不省取决于融合怎么切，同一个值可能这边舍了、那边没舍。小尺寸、单一配置；别的模型要重编重扫。")
    f.save("fig8-elided.svg", yb + 14)


def fig_cancel():
    """fig8-cancel 回答：为什么两处都照 GPU 补上，后段反而比只补流梯度高？—— 研究线的比方：一台秤三个毛病，甲（流梯度少舍）偏重、
    乙（收拢处那次梯度少舍）偏轻、丙偏重还没找到。只修甲，乙丙大致抵掉；甲乙都修，丙露出来。
    数据：topic08_numbers 的 CX、TW_P_T、EL_GPU_HEADGRAD（实测）；「三个毛病、抵掉、丙」是推断。
    ⛔ 刻意没画：甲乙丙各自的大小（没有单独测出）；箭头只表示方向。"""
    C = N.CX
    f = Fig(W, "像一台秤有三个毛病：甲让读数偏重，是流梯度少舍，主因，已找到；乙让读数偏轻，是收拢处那次梯度少舍；丙也偏重，还没找到。"
               "四行是四种配置，右边是第 551 到 1000 步主 loss 的偏差，单位十的负六次方。都没修 ＋9.4；只修乙 ＋12.5，单看分不开；只修甲 ＋1.9，乙和丙大致抵掉，看着准了；"
               "甲乙都修 ＋6.9，比只修甲高 5.0，p 0.024，丙露出来了。最下面：GPU 上拿掉乙那次舍入，后段压低 2.9，方向一致，是那 5.0 的一部分。这几步的解释是推断。")
    y0 = f.header("修好一个毛病，读数反而偏了：另一个毛病一直在帮着抵（推断）",
                  "第二次迁移，小尺寸，v5p；读数 ＝ 第 551–1000 步主 loss 偏差（TPU − GPU，×1e-6），每格 4–6 份运行的平均；甲乙丙的解释是推断",
                  [(OR, "偏重：甲 流梯度少舍、丙 还没找到"), (BL, "偏轻：乙 收拢处梯度少舍"), (GR, "已修")])
    PH = 400
    py = f.panel(0, y0, W, PH, "一台秤，三个毛病：四种修法，读数是多少", OR)
    cols = {"a": 290, "b": 410, "c": 530}
    for k, lab, col in (("a", "甲", OR), ("b", "乙", BL), ("c", "丙", OR)):
        f.t(cols[k], py + 26, lab, col, True, 14, "middle")
    B0, B1, vmax = 640, W - 230, 13.5e-6

    def BX(v):
        return B0 + v / vmax * (B1 - B0)
    rows = [("都没修（修之前）", False, False, C["none"], "三股相抵后的读数"),
            ("只修乙（收拢处）", False, True, C["head"], "均值往上走 %s，4 份单看分不开（p %.2f）" % (N.CX_HEAD_UP_T, N.CX_HEAD_P)),
            ("只修甲（流梯度）", True, False, C["stream"], "乙和丙大致抵掉，看着准了"),
            ("甲乙都修（对齐配置）", True, True, C["both"], "比只修甲高 %s（p %s）：丙露出来了" % (N.CX_UP_T, N.TW_P_T["c_m58"]))]
    for i, (lab, a_ok, b_ok, v, note) in enumerate(rows):
        y = py + 50 + i * 68
        f.t(20, y + 24, lab, INK, True, 13.5)
        for key, ok, col, sign in (("a", a_ok, OR, 1), ("b", b_ok, BL, -1), ("c", False, OR, 1)):
            x = cols[key]
            if ok:
                f.box(x - 30, y + 8, 60, 26, "#fff", GR, 13, sw=1.4)
                f.t(x, y + 26, "已修", GR, True, 12.5, "middle")
            elif key == "c":
                f.line(x, y + 21 + 14 * sign, x, y + 21 - 14 * sign, col, 3, dash="4 3")
                f.t(x + 14, y + 26, "？", col, True, 13)
            else:
                f.line(x, y + 21 + 14 * sign, x, y + 21 - 14 * sign, col, 3)
        hl = (i == 3)
        f.box(B0, y + 8, max(BX(v) - B0, 3), 24, OR, OR, 4)
        f.t(BX(v) + 8, y + 26, "＋%s" % N.CX_T[["none", "head", "stream", "both"][i]], INK, True, 14)
        f.t(B0, y + 50, note, OR if hl else GY, hl, 12.5 if hl else 12)
    f.line(B0, py + 40, B0, py + 50 + 4 * 68 - 20, GY, 1.2, arrow=False)
    yg = py + 50 + 4 * 68 + 6
    f.line(14, yg - 8, W - 14, yg - 8, LINE, 1, arrow=False)
    f.t(20, yg + 18, "反过来在 GPU 上验乙：", BL, True, 13.5)
    f.t(210, yg + 18, "只拿掉收拢处那次梯度舍入（%d 份），后段压低 %s：方向一致，推断是那 %s 的一部分。"
        % (N.EL_GPU_HEADGRAD_N, "%.1f" % (-N.EL_GPU_HEADGRAD["late"] * 1e6), N.CX_UP_T), INK, size=13)
    f._pan = None
    yb = f.band(y0 + PH + 20, "warn", "甲乙都修以后剩下的 ＋6.9，不是修坏了，是被乙遮住的丙露了出来（推断）", [
        "丙在哪？下一步去找它。",
    ])
    yb = f.src(yb + 10,
               "📌 实测：四格都是第 551–1000 步主 loss 对 GPU 16 份逐步均值的带符号均值，单位 1e-6；都没修 4 份、只修乙 4 份、只修甲 5 份、甲乙都修 6 份；甲乙都修对只修甲是置换检验。GPU 那一行对 GPU 16 份留一法，4 份；它只拿掉收拢处反向那一次梯度舍入，跟 fig-twoway 里整段换收拢处写法的那格不是同一个改法。",
               "⚠️ 推断：三个毛病、乙丙抵掉、丙在别处，是对齐记录的解释；GPU 一侧方向一致，但大小只占 5.0 的一部分。箭头只表示方向，甲乙丙各自多大没有单独测出。")
    f.save("fig8-cancel.svg", yb + 14)

def fig_sites():
    """fig8-sites 回答：多路残差这一段，GPU 舍、TPU 可能被省的舍入点在哪？—— 三处：⓪ 分支输入（修之前就照 GPU 舍了）、
    ① 流梯度（反向三份梯度合流）、② 收拢处（四路收成一路）。只是地图，不给结论；哪处是主因留给后面两头验证。
    ⛔ 刻意没画：层数、门控的公式、MTP；每层只画一次分支。"""
    f = Fig(W, "一张地图。残差流复制成 4 路，横着走过一层又一层。每层里，4 路按权混成一份送进分支（注意力或 MLP），这是 ⓪ 分支输入，"
               "修之前就已经照 GPU 舍了；反向时，流在一层里被用了三次，回来三份梯度要加成一份，这是 ① 流梯度；"
               "最后 4 路收成一路、进末层归一化，这是 ② 收拢处。三处在代码里都写了「转 BF16」。")
    y0 = f.header("多路残差这一段：三处该舍 BF16 的地方",
                  "地图，不是结论：哪一处让尾巴长出来，后面一处处验",
                  [(BL, "前向：流"), (OR, "反向：梯度"), (GR, "修之前就照 GPU 舍了")])
    PH = 330
    py = f.panel(0, y0, W, PH, "4 路流走过一层，最后收成一路", BL)
    lanes = [py + 60 + k * 34 for k in range(4)]
    X0, X1 = 40, 1000
    for y in lanes:
        f.line(X0, y, X1, y, BL, 2.2, arrow=False)
    f.t(X0, lanes[0] - 14, "残差流 × 4 路", BL, True, 13)
    # 一层：混合 → 分支 → 加回
    mx = 300
    f.box(mx - 10, lanes[0] - 12, 20, lanes[3] - lanes[0] + 24, "#fff", BL, 6, sw=1.4)
    f.elbow(mx + 10, (lanes[1] + lanes[2]) / 2, mx + 120, py + 250, BL, 1.6)
    f.box(mx + 120, py + 230, 170, 40, "#fff", BL, 8, sw=1.6)
    f.t(mx + 205, py + 255, "分支（注意力／MLP）", BL, True, 13, "middle")
    f.elbow(mx + 290, py + 250, 620, lanes[3] + 4, BL, 1.6)
    f.badge(mx + 62, py + 201, "0", GR)
    f.t(mx + 92, py + 205, "分支输入", GR, True, 13)
    f.t(mx + 92, py + 222, "修之前就照 GPU 舍了", GR, size=12)
    # 反向：三份梯度合流（三处用到流的地方各回来一份梯度，虚线往回指）
    gx = 700
    for xx in (760, 860, 960):
        f.path([(xx, lanes[3] - 2), (xx, lanes[3] + 40), (gx + 28, lanes[3] + 40)], OR, 1.6, dash="5 4")
    f.badge(gx, lanes[3] + 27, "1", OR)
    f.t(gx + 36, lanes[3] + 80, "① 流梯度：流在一层里被用了三次，", OR, True, 13)
    f.t(gx + 36, lanes[3] + 100, "反向回来三份梯度，要加成一份", OR, size=12.5)
    # 收拢处
    cx = 1040
    for y in lanes:
        f.line(X1, y, cx, (lanes[1] + lanes[2]) / 2, BL, 1.8, arrow=False)
    f.line(cx, (lanes[1] + lanes[2]) / 2, cx + 90, (lanes[1] + lanes[2]) / 2, BL, 2.4)
    f.box(cx + 92, (lanes[1] + lanes[2]) / 2 - 20, 160, 40, "#fff", BL, 8, sw=1.6)
    f.t(cx + 172, (lanes[1] + lanes[2]) / 2 + 5, "末层归一化", BL, True, 13, "middle")
    f.badge(cx + 30, lanes[3] + 30, "2", INK)
    f.t(cx + 62, lanes[3] + 46, "② 收拢处：4 路收成一路", INK, True, 13.5)
    f.t(cx + 62, lanes[3] + 66, "前向舍一次，梯度传回来也舍", GY, size=12.5)
    f._pan = None
    yb = f.band(y0 + PH + 20, "info", "三处在代码里都写了「转 BF16」；GPU 都真舍了，TPU 可能被编译器省掉", [
        "⓪ 修之前就补上了；① 和 ② 是这一段要查的两处。",
    ])
    f.save("fig8-sites.svg", yb + 14)


def fig_verify():
    """fig8-verify 回答：扫出来的候选，上卡两头验了以后怎样？—— 左：GPU 上照 TPU 改（dW、注意力输入、MoE 输入、两处一起）；
    右：TPU 上在对齐配置之上补注意力输入、注意力加 MoE 输入。两头都没把残余造出来或消下去：清单前三类排除，残余还在查。
    数据：topic08_numbers 的 GV、GV_RUNS、TV、TV_RUNS（研究线已合入的两份结果文件；实测）；判读来自对齐记录（注意力输入单开那次是偶然：推断）。
    ⛔ 刻意没画：前向几处（还没验）。"""
    R, T = N.GV_RUNS, N.TV_RUNS
    f = Fig(W, "上卡两头验证。左块 GPU 一头：在 GPU 上把候选照 TPU 改，看 GPU 自己长不长出尾巴。注意力输入那处单开 ＋3.0e-6，p 0.021；"
               "跟 MoE 输入那处一起开只剩 ＋1.9e-7，p 0.87；每个微批的 dW −3.8e-7，p 0.73。右块 TPU 一头：在对齐配置之上补上，看尾巴下不下去。"
               "对齐配置 6 份 ＋6.9e-6；补注意力输入那处 ＋6.1e-6，p 0.67；注意力加 MoE 输入 ＋5.3e-6，p 0.37，都分不开。两头都排除了清单前三类，残余还在查。")
    y0 = f.header("两头验证：清单前三类都排除了，残余还在查",
                  "第二次迁移，小尺寸，第 551–1000 步主 loss 偏差，每点一份运行，横线是均值",
                  [(GY2, "参照格"), (GR, "改了一处，结论：排除")])
    PH, GAP = 420, 22
    LW = 820
    vmin, vmax = -5e-6, 1.2e-5

    def panel(x0, w, title, col, cells, data, side):
        py = f.panel(x0, y0, w, PH, title, col)
        X0, X1 = x0 + 80, x0 + w - 16
        YT, YB = py + 40, py + PH - 100

        def Y(v):
            return YT + (vmax - v) / (vmax - vmin) * (YB - YT)
        for v in (-4e-6, 4e-6, 8e-6, 1.2e-5):
            f.line(X0, Y(v), X1, Y(v), LINE, 0.6, arrow=False)
            f.t(X0 - 6, Y(v) + 4, {-4e-6: "−4e-6", 4e-6: "＋4e-6", 8e-6: "＋8e-6", 1.2e-5: "＋1.2e-5"}[v], GY, size=11.5, anchor="end")
        f.line(X0, Y(0), X1, Y(0), GY, 1.2, arrow=False)
        f.t(X0 - 6, Y(0) + 4, "0", GY, size=11.5, anchor="end")
        cw = (X1 - X0) / float(len(cells))
        for k, (key, lab, c, mean_t, ptxt) in enumerate(cells):
            cx = X0 + cw * (k + 0.5)
            vals = data[key]
            n = len(vals)
            for m, v in enumerate(vals):
                dx = (m - (n - 1) / 2.0) * min(11, 120.0 / n)
                f.box(cx + dx - 5, Y(v) - 5, 10, 10, c, "#fff", 5, sw=1)
            mean = sum(vals) / n
            f.line(cx - 40, Y(mean), cx + 40, Y(mean), INK, 2.4, arrow=False)
            f.t(cx, YB + 24, lab, c if c != GY2 else GY, True, 13, "middle")
            if mean_t:
                f.t(cx, YB + 44, mean_t, INK, False, 12, "middle")
            if ptxt:
                f.t(cx, YT - 10, ptxt, GY, True, 12, "middle")
        f._pan = None
        return py
    gcells = [("loo", "GPU 自己", GY2, "16 份", ""), ("attn", "注意力输入", GR, "均值 %s" % N.GV_T["attn"][0], "p %s" % N.GV_ATTN_P),
              ("moe", "MoE 输入", GR, "均值 %s" % N.GV_T["moe"][0], "p %s" % N.GV_T["moe"][1]),
              ("both", "两处一起", GR, "均值 %s" % N.GV_BOTH_T, "p %s" % N.GV_BOTH_P),
              ("wgrad", "每微批 dW", GR, "均值 %s" % N.GV_WGRAD_T, "p %s" % N.GV_WGRAD_P)]
    panel(0, LW, "GPU 一头：照 TPU 改，长不长出尾巴（对 GPU 16 份均值）", OR, gcells, R, "gpu")
    tcells = [("C", "对齐配置", GY2, "6 份 %s" % N.TV_T["C"], ""), ("attn", "＋补注意力输入", GR, "均值 %s" % N.TV_ATTN_T, "p %s" % N.TV_ATTN_P),
              ("all", "＋注意力和 MoE 输入", GR, "均值 %s" % N.TV_ALL_T, "p %s" % N.TV_ALL_P)]
    panel(LW + GAP, W - LW - GAP, "TPU 一头：补上，尾巴下不下去", BL, tcells, T, "tpu")
    yb = f.band(y0 + PH + 20, "warn", "清单是候选，不是结论：前三类两头都排除，残余还在查", [
        "注意力输入那处 GPU 单开像有一点，可合开就没了，按格数校正 p 0.084，对齐记录推断是偶然；TPU 补上也跟对齐配置分不开。下一批：前向几处，和扫描看不到的差。",
    ])
    yb = f.src(yb + 10,
               "📌 实测（第二次迁移，小尺寸，1000 步，每格 4 份）：第 551–1000 步主 loss 的带符号均值。左块 B200×8，改了一处的格子对 GPU 16 份均值，GPU 自己那格是每份对其余 15 份，p 对 GPU 自己；"
               "右块 TPU v5p 8 设备，对 GPU 16 份均值，p 对对齐配置 6 份。「注意力输入」：q、kv 下投影和输出门三份梯度合流；「MoE 输入」：路由、共享专家、路由专家三份梯度合流；「每微批 dW」：每个微批的权重梯度。",
               "⚠️ 判读来自对齐记录。还没验：前向几处，以及扫描找不到的差（代码没写舍入、GPU 却舍了的地方；累加顺序）。")
    f.save("fig8-verify.svg", yb + 14)

fig_elided()
fig_cancel()
fig_sites()
fig_verify()
