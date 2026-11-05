# -*- coding: utf-8 -*-
r"""专题九 · 第四节「张量级指标」的两张图。

fig9-cos     回答：rel_L2 和 cos 各看见什么、看不见什么？
             现算：被测 ＝ 参照 × 2 时 cos ＝ 1、rel_L2 ＝ 1；同长度转 2.56° 时 cos ＝ 0.999、rel_L2 ≈ 4.5%（1 − cos ≈ ½·rel_L2²）。
fig9-rratio  回答：TPU 离 GPU 远了，是谁的错？—— 拿 FP64 真值当靶心，比两边各离靶心多远：R ＝ 被测距离 ÷ GPU 距离。
             数字：门槛 1.2（推断）；√2 ≈ 1.41 ＝ 多一次同量级独立舍入（本课推导）；qkv 投影 1.235 → 1.000（实测）。
fig9-flip    回答：第 1 步一个专家的梯度差 0.102、全体只有 1.28e-3，差从哪来？—— 路由第 8、9 名对调，专家换了 token。
             数字全取 topic09_numbers（实测）：STEP1_EXPERT / STEP1_ALL / STEP1_TENSORS、E0_*、FLIP_MARGIN、TOPK、EXPERTS。
             刻意没画：名次条的长度和第 1–10 名的分数（示意，画上刻度就是编）；点阵里哪几个 token 被换（位置示意，个数实测）。
"""
import math

from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE
import topic09_numbers as N

W = 1400
TH = math.degrees(math.acos(0.999))
REL_ROT = 2 * math.sin(math.radians(TH) / 2)          # 同长度、夹角 θ：‖a − b‖ ÷ ‖b‖ ＝ 2 sin(θ/2)
assert abs(TH - 2.56) < 0.01 and abs(REL_ROT - N.COS_0999_RELL2) < 1e-4


def fig_cos():
    f = Fig(W, "三格。左格：被测是参照的两倍长、方向一样，cos 是 1，看不出问题；rel_L2 是 100%，范数比是 2。"
               "中格：被测和参照一样长、转了 2.56 度，cos 是 0.999，看着很接近；rel_L2 已经是 4.5%。"
               "右格一张小表：rel_L2 主判，cos 补方向，范数比补 cos 看不见的缩放。")
    y0 = f.header("一个张量出一个数：rel_L2 主判，cos 和范数比各补一个盲点",
                  "灰箭头 ＝ 参照（GPU）；彩色箭头 ＝ 被测（TPU）；红色虚线 ＝ 两者之差 a − b，rel_L2 就是它的长度 ÷ 灰箭头的长度",
                  [(GY, "参照 b"), (BL, "被测 a"), (RD, "差 a − b")])
    PH = 330
    for k, (title, scale, ang, txt) in enumerate([
            ("被测 ＝ 参照 × 2", 2.0, 0.0, ["cos ＝ 1（看不出）", "rel_L2 ＝ 100%", "范数比 ＝ 2"]),
            ("一样长，转了 %.2f°" % TH, 1.0, TH * 6, ["cos ＝ 0.999（看着很近）", "rel_L2 ＝ %.1f%%" % (REL_ROT * 100), "范数比 ＝ 1"])]):
        px = k * 470
        py = f.panel(px, y0, 450, PH, title, BL)
        ox, oy, L = px + 50, py + 230, 150
        f.line(ox, oy, ox + L, oy, GY, 4)
        a = math.radians(ang)
        ax, ay = ox + L * scale * math.cos(a), oy - L * scale * math.sin(a)
        if ang == 0:
            f.line(ox, oy - 26, ox + L * scale, oy - 26, BL, 4)
            f.line(ox + L, oy - 40, ox + L * scale, oy - 40, RD, 2, dash="6 5")
        else:
            f.line(ox, oy, ax, ay, BL, 4)
            f.line(ox + L, oy, ax, ay, RD, 2, dash="6 5", arrow=False)
        for i, t in enumerate(txt):
            f.t(px + 24, py + 40 + i * 24, t, RD if i == 1 else INK, i == 1, 15)
        f._pan = None
    px = 940
    py = f.panel(px, y0, W - px, PH, "谁看得见什么", GR)
    rows = [("", "整体 × 2", "方向偏", "个别大错"),
            ("rel_L2", "✓", "✓", "✓（分不清哪种）"),
            ("cos", "✕", "✓", "✓"),
            ("范数比", "✓", "✕", "✕")]
    for i, r in enumerate(rows):
        for j, c in enumerate(r):
            x = px + 18 + [0, 110, 210, 300][j]
            col = GR if c.startswith("✓") else (RD if c == "✕" else INK)
            f.t(x, py + 44 + i * 44, c, col, i == 0 or j == 0, 14)
    f.t(px + 18, py + 44 + 4 * 44 + 10, "长度相同、差很小时 1 − cos ≈ ½·rel_L2²：", GY, size=13)
    f.t(px + 18, py + 44 + 4 * 44 + 32, "cos 0.999 ≈ rel_L2 4.5%", GY, True, 13)
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "主判看 rel_L2：差占信号几成；梯度一定要配范数比", [
        "梯度整体大了几倍时 cos 仍是 1，靠范数比才抓得到：累加了 2 个 microbatch 忘了除以 2，范数比就是 2。",
        "代价：一个张量压成一个数，分不清均匀小错还是几个大错 —— 超线以后再按行、按专家拆开定位。",
    ])
    yb = f.src(yb + 10,
               "⚠️ 本课推导：同长度、夹角 θ 时 rel_L2 ＝ 2·sin(θ/2)；cos θ ＝ 0.999 → θ ≈ 2.56°，rel_L2 ≈ 4.47%。图里中格的角度放大了 6 倍画，方便看见。",
               "📌 指标一律在 FP64 里算：两个相近的数相减本身是精确的，丢精度的是后面的平方和、点积和除法。")
    f.save("fig9-cos.svg", yb + 14)


def fig_rratio():
    f = Fig(W, "一个靶子，靶心是同一份输入在 FP64 里算出的真值。GPU 离靶心的距离定为 1，画成灰色圆。"
               "四种读法直接画在靶子上：灰圆里面淡蓝，R 小于 1，TPU 比 GPU 更近；灰圆到 1.2 倍的绿环，平台固有；1.2 倍以外淡红，去查 TPU；"
               "橙色虚圆是 1.41 倍，相当于比 GPU 多舍入了一次同量级的数。"
               "TPU 的 qkv 投影原来在 1.235 倍，红点，落在淡红区；改成 FP32 累加以后回到 1.000，绿点，和 GPU 一样远。")
    y0 = f.header("TPU 离 GPU 远了，先问：两边各离真值多远",
                  "靶心 ＝ 同一份输入在 FP64 里算的真值；R ＝ TPU 离靶心的距离 ÷ GPU 离靶心的距离；底色 ＝ 落在这一圈该怎么读",
                  [(GY, "GPU 的距离（R ＝ 1）"), (GR, "门槛 R ＝ 1.2"), (OR, "多舍一次 R ≈ 1.41")])
    PH = 420
    py = f.panel(0, y0, W, PH, "单个算子：注意力的 qkv 投影（实测）", BL)
    cx, cy, U = 700, py + 200, 112

    def disk(r, fill):
        f.p.append('<circle cx="%d" cy="%d" r="%.1f" fill="%s" stroke="none"/>' % (cx, cy, r * U, fill))

    def circ(r, col, dash=None, sw=2):
        f.p.append('<circle cx="%d" cy="%d" r="%.1f" fill="none" stroke="%s" stroke-width="%s"%s/>'
                   % (cx, cy, r * U, col, sw, ' stroke-dasharray="%s"' % dash if dash else ""))
    # ⭐ R3：右格「四种读法」原来是一张文字清单 → 直接画成靶子上的底色（淡蓝 ＜ 1、淡绿 1–1.2、淡红 ＞ 1.2）
    RMAX = 1.68
    disk(RMAX, "#fce8e6")
    disk(N.R_GATE, "#e6f4ea")
    disk(1.0, "#e8f0fe")
    circ(N.R_ONE_EXTRA, OR, "7 6")
    circ(N.R_GATE, GR, None, 2.5)
    circ(1.0, GY, None, 2)
    f.box(cx - 6, cy - 6, 12, 12, INK, INK, 6)
    f.t(cx, cy + 26, "FP64 真值", INK, True, 13, "middle")
    # 四种读法：标签放在靶子两侧，引线指到对应的那一圈
    LX, RX = cx - RMAX * U - 40, cx + RMAX * U + 40
    for (lx, ly, anchor, col, a, b, tx, ty) in [
            (LX, cy - 110, "end", BL, "R ＜ 1", "TPU 更近、GPU 远", cx - 0.62 * U, cy - 0.55 * U),
            (LX, cy + 40, "end", GR, "R ≈ 1–1.2", "平台固有：只是加法顺序不同", cx - 1.1 * U, cy + 0.1 * U),
            (RX, cy - 110, "start", RD, "R ＞ 1.2", "TPU 多舍了：去查 TPU", cx + 1.15 * U, cy - 1.1 * U),
            (RX, cy + 40, "start", OR, "R ≈ 1.41", "像是多舍入了一次同量级的数", cx + N.R_ONE_EXTRA * U * 0.97, cy + 0.36 * U)]:
        f.t(lx, ly, a, col, True, 17, anchor)
        f.t(lx, ly + 24, b, INK, size=14, anchor=anchor)
        x0 = lx + (8 if anchor == "end" else -8)
        f.line(x0, ly - 6, tx, ty, col, 1.0, arrow=False)
        f.box(tx - 3, ty - 3, 6, 6, col, col, 3)
    # 两个实测点：标签放在靶子外面，用短引线连回点
    for r, ang, col, lab, ldx, ldy in [(N.R_QKV_BEFORE, 62, RD, "改之前 R ＝ %.3f" % N.R_QKV_BEFORE, 70, -48),
                                       (N.R_QKV_AFTER, 235, GR, "改之后 R ＝ %.3f" % N.R_QKV_AFTER, -40, 70)]:
        x, y = cx + r * U * math.cos(math.radians(ang)), cy - r * U * math.sin(math.radians(ang))
        f.line(x, y, x + ldx, y + ldy, col, 1.0, arrow=False)
        f.box(x - 9, y - 9, 18, 18, col, INK, 9)
        f.t(x + ldx + (6 if ldx > 0 else -6), y + ldy + (0 if ldy < 0 else 14), lab, col, True, 15,
            "start" if ldx > 0 else "end")
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "FP64 不是标准答案，是定责的工具：它告诉你差是谁带来的", [
        "qkv 投影超线的根因在 TPU：分片部分积在 BF16 里累加（下一节讲），改成 FP32 累加后 R 从 1.235 回到 1.000。",
        "代价：要在 CPU 上用同一份真实输入重算一遍 FP64；只适合单个算子和单层，整网多步靠噪声尺子。",
    ])
    yb = f.src(yb + 10,
               "📌 距离都用 rel_L2 量；靶子上两个点画在不同角度只为分开，只有离圆心多远有意义。",
               "⚠️ 推断（门槛）：只差舍入顺序时 R ≈ 1；多一次同量级的独立舍入，误差按平方相加，R ≈ √2 ≈ 1.41；取 1.2 当线。",
               "📌 实测：v5p 上单算子 FP64 对比，qkv 投影 R 1.235 → 1.000，第 0 层稠密 MLP 的输入投影 1.294 → 1.117（打开 FP32 累加的开关后，9 条单算子检查全过）。")
    f.save("fig9-rratio.svg", yb + 14)


def fig_flip():
    ratio = N.STEP1_EXPERT / N.STEP1_ALL
    assert 70 < ratio < 90, ratio
    f = Fig(W, "两格。左格：同一个 token 在 GPU 和 TPU 上给专家打分的名次表，前 %d 名入选；"
               "第 8 名和第 9 名分数几乎一样，两边正好对调，GPU 选了专家 A，TPU 选了专家 B。翻了的 token，两名之间的分差中位数是 %s。"
               "右格：第 3 层 0 号专家分到 %d 个 token，TPU 上少了 %d 个、多了 %d 个；它的梯度 rel_L2 是 %s，一根长红柱，"
               "而比到的 %d 个张量合起来只有 %s，一根很短的蓝柱，约差 %d 倍；GPU 自己重跑路由不变，宽度为 0。"
               % (N.TOPK, N.FLIP_MARGIN, N.E0_TOKENS, N.E0_OUT, N.E0_IN, N.STEP1_EXPERT,
                  N.STEP1_TENSORS, N.STEP1_ALL, round(ratio)))
    y0 = f.header("第 8、9 名一对调，一个专家就换了一批 token",
                  "左格：同一个 token 的专家名次，上面 %d 名入选（共 %d 个专家）；右格：这个专家换了几个 token、梯度差多大（柱长按 rel_L2 线性画）"
                  % (N.TOPK, N.EXPERTS),
                  [(GY, "GPU"), (BL, "TPU"), (RD, "对调的两名 ／ 偏差来源")])
    PH = 420
    # ── 左格：名次表 ──
    LW = 640
    py = f.panel(0, y0, LW, PH, "名次表：同一个 token，两边的第 8、9 名对调", BL)
    scores = [1.00, 0.93, 0.87, 0.82, 0.78, 0.74, 0.71, 0.685, 0.684, 0.62]   # 示意
    RH, top = 25, py + 52
    cols = [(70, "GPU", GY, ["", "", "", "", "", "", "", "A", "B", ""]),
            (355, "TPU", BL, ["", "", "", "", "", "", "", "B", "A", ""])]
    BLEN = 200
    for cx, name, col, labs in cols:
        f.t(cx, top - 16, name, col, True, 15)
    for i in range(10):
        y = top + i * RH
        f.t(56, y + 15, "%d" % (i + 1), GY, i in (7, 8), 13, "end")
    for cx, name, col, labs in cols:
        for i, sc in enumerate(scores):
            y = top + i * RH
            hot = i in (7, 8)
            L = BLEN * sc
            if hot:
                f.box(cx, y + 3, L, RH - 8, col, RD, 3, sw=2)
                f.t(cx + L + 8, y + 16, "专家 " + labs[i], RD, True, 14)
            else:
                f.box(cx, y + 3, L, RH - 8, col, col, 3)
    # 入选线：第 8 名下面
    yc = top + 8 * RH - 2
    f.line(30, yc, LW - 24, yc, INK, 1.4, dash="6 4", arrow=False)
    f.t(LW - 24, yc - 6, "以上入选", INK, True, 13, "end")
    f.t(LW - 24, yc + 16, "以下落选", GY, False, 13, "end")
    # 对调的交叉线：GPU 的 A（第 8）→ TPU 的 A（第 9），B 反之
    xg, xt = 70 + BLEN * scores[7] + 70, 355 - 12
    f.line(xg, top + 7 * RH + 13, xt, top + 8 * RH + 13, RD, 1.4, dash="4 3", arrow=True)
    f.line(xg, top + 8 * RH + 13, xt, top + 7 * RH + 13, RD, 1.4, dash="4 3", arrow=True)
    f.t(36, top + 10 * RH + 20, "⋮ 后面还有 %d 名" % (N.EXPERTS - 10), GY, size=13)
    f.t(24, py + PH - 76, "第 8、9 名分差：中位数 %s" % N.FLIP_MARGIN.replace(" 到 ", "–"), RD, True, 15)
    f.t(24, py + PH - 52, "（条长是示意；真实差小到画不出来）", GY, size=13)
    f._pan = None
    # ── 右格：后果 ──
    px = LW + 20
    RW = W - px
    py = f.panel(px, y0, RW, PH, "后果：这个专家的梯度差", RD)
    f.t(px + 20, py + 26, "第 3 层 0 号专家分到 %d 个 token" % N.E0_TOKENS, INK, True, 14)
    per, dx = 28, 15
    gx, gy = px + 24, py + 44
    out_idx = {23, 71, 118}                     # 位置示意
    assert len(out_idx) == N.E0_OUT
    for k in range(N.E0_TOKENS + N.E0_IN):
        x, y = gx + (k % per) * dx, gy + (k // per) * dx
        if k >= N.E0_TOKENS:
            f.box(x, y, 10, 10, RD, RD, 5)                       # 换进
        elif k in out_idx:
            f.box(x, y, 10, 10, "#fff", RD, 5, sw=2)             # 换出
        else:
            f.box(x, y, 10, 10, GY2, GY2, 5)
    lx = gx + per * dx + 20
    f.box(lx, gy + 4, 10, 10, "#fff", RD, 5, sw=2)
    f.t(lx + 18, gy + 14, "TPU 换出 %d 个" % N.E0_OUT, RD, True, 14)
    f.box(lx, gy + 30, 10, 10, RD, RD, 5)
    f.t(lx + 18, gy + 40, "换进 %d 个" % N.E0_IN, RD, True, 14)
    f.t(lx, gy + 66, "一个点 ＝ 一个 token", GY, size=13)
    # 柱子：rel_L2 线性
    BX, BMAX = px + 24, 440
    by = gy + 5 * dx + 50
    bars = [("这个专家的梯度（rel_L2）", N.STEP1_EXPERT, RD, "%g" % N.STEP1_EXPERT),
            ("%d 个张量合起来" % N.STEP1_TENSORS, N.STEP1_ALL, BL, ("%.2e" % N.STEP1_ALL).replace("e-0", "e-")),
            ("GPU 重跑", 0.0, GY, "0")]
    for i, (lab, v, col, vs) in enumerate(bars):
        y = by + i * 60
        f.t(BX, y, lab, INK, i == 0, 14)
        L = BMAX * v / N.STEP1_EXPERT
        if L > 0:
            f.box(BX, y + 8, max(L, 3), 20, col, col, 3)
        else:
            f.line(BX, y + 6, BX, y + 30, GY, 2, arrow=False)
        f.t(BX + max(L, 3) + 10, y + 24, vs, col, True, 15)
        if i == 2:
            f.t(BX + 40, y + 24, "路由不变，尺子宽度 0", GY, size=14)
    f.t(BX + BMAX + 80, by + 24, "约 %d 倍" % round(ratio), RD, True, 17)
    f._pan = None
    yb = f.band(y0 + PH + 20, "warn", "差出在路由上，不在专家的算术上", [
        "近平局 ＋ 上游一丝输入差 ＝ 选了另一个专家；同输入则 0 翻转。",
        "所以第 1 步张量比较先看路由槽位，翻了就按两边选中的同一批 token 重比。",
    ])
    yb = f.src(yb + 10,
               "📌 实测：第 1 步梯度对 GPU 的 rel_L2，第 3 层 0 号专家的权重梯度 %g，比到的 %d 个张量合起来 %g；"
               "隔离回放的那批 token 里这个专家分到 %d 个，TPU 侧少 %d 多 %d；发生翻转的 token，第 8、9 名「分数＋偏置」之差的中位数按层在 %s 之间。"
               % (N.STEP1_EXPERT, N.STEP1_TENSORS, N.STEP1_ALL, N.E0_TOKENS, N.E0_OUT, N.E0_IN, N.FLIP_MARGIN),
               "⚠️ 示意：左格名次条的长度和第 1–10 名的分数是示意，只有「第 8、9 名对调」和分差中位数是实测；右格点阵的个数是实测，哪几个点被换出是随手标的位置。")
    f.save("fig9-flip.svg", yb + 14)


fig_cos()
fig_rratio()
fig_flip()
