# -*- coding: utf-8 -*-
r"""专题九 · 第十一节「稀疏注意力核：把挑中的 key 当成 MoE 来算」的图（2026-11-24 补）。

fig8b-dsa-moe     回答：每个 query 只挑 2048 个 key，现成的块稀疏核为什么一点不省？怎么改成能省的？
                  —— Ⓐ 玩具账：4 个 query × 16 个 key，每人挑 4 个，4 块里 4 块都要算；Ⓑ 把这一单要的 key 收拢成一摞（并集 8 行），
                  整单一起乘：这一单就是 MoE 的一个专家，收拢来的 key 行就是分给它的 token。
fig8b-dsa-arch    回答：这颗核的软件架构长什么样？—— 五步（indexer 分块 top-k → 账本 → 小核收行 → 主核流式计算 → 小核散射累加 dKV），
                  标出每步在主核 Pallas／小核 Pallas／XLA 上跑，下半是前向、反向两条时间线上谁跟谁重叠。
fig8b-dsa-perrow  回答：照 MoE 的分工，为什么搬行要交给小核？—— Ⓐ 每搬一行后面跟多少活：MoE 约 3800 万次乘加、DSA 约 111 万；
                  这一行在主核上只算约 2 纳秒，主核搬它要 13–16 纳秒；Ⓑ 交给小核以后切段重叠：13.9 → 8.8 ms。
fig8b-dsa-ladder  回答：从慢 9 倍到快 1.3–1.6 倍，一步一步怎么来的？—— 三段（口径各不相同，段内才能比），两种宽度；下面是账本那条线 71 → 3.5 ms。
fig8b-dsa-result  回答：最后快多少？—— 32K 快 1.29 / 1.61 倍（含账本 1.19 / 1.52），256K 快 5.9 / 8.2 倍（含账本 5.6 / 7.8）。
fig8b-dsa-layer   回答：单测快了，放进整层呢？—— 一台主机上的整层代理：稠密层 43.3 ms（MoE 的通信藏在注意力底下 14.9 ms），
                  稀疏层 52.0 ms（一点没藏住）；两层串起来 99.3 对 99.0。

⛔ 数字全部 import 自 topic08b_numbers.py（DSA_*），断言在那边。不写内部代码名、运行库版本、开关名、机器名、桶路径。
"""
import topic08_figfont  # noqa: F401
from topic03_draw import Fig, BL, OR, GR, RD, GY, INK, GY2, LINE, LINE2, PU
Fig.BAND_KEEP_LINES = 0
import topic08b_numbers as N

W = 1400
C32, C64 = OR, BL           # 32 头（较小的模型）／ 64 头（公开的 Hy4）

# 玩具挑选：query 12–15，各挑 4 个 key（含自己），因果
TOY = {12: (0, 5, 11, 12), 13: (0, 5, 12, 13), 14: (0, 5, 13, 14), 15: (0, 9, 14, 15)}
TOY_UNION = sorted(set(k for v in TOY.values() for k in v))
assert TOY_UNION == [0, 5, 9, 11, 12, 13, 14, 15]
TOY_TILES_ACTIVE = sum(1 for b in range(4) if any(b * 4 <= k < b * 4 + 4 for v in TOY.values() for k in v))
assert TOY_TILES_ACTIVE == 4

# 每搬一行，后面跟多少乘加（本课推导）
MOE_MAC_ROW = N.MC_HIDDEN * 2 * N.MC_EXPERT_FF + N.MC_EXPERT_FF * N.MC_HIDDEN     # wi（gate ＋ up）＋ wo
DSA_MAC_ROW = N.DSA_MAC_PER_ROW[1024]
assert MOE_MAC_ROW == 37748736 and DSA_MAC_ROW == 1114112
ROW_NS = lambda mac: 2.0 * mac / (N.PEAK_DEV * 1e12) * 1e9                       # 这一行的乘法按峰值要几纳秒
assert abs(ROW_NS(DSA_MAC_ROW) - 1.93) < 0.01 and abs(ROW_NS(MOE_MAC_ROW) - 65.4) < 0.1


def fig_moe():
    f = Fig(W, "左：一个 query 块的 4 个 query（位置 12 到 15）各从 16 个 key 里挑 4 个，蓝格是挑中的；key 按 4 个一块切，"
               "4 块里每块都至少有一个蓝格，所以 4 块都要整块算，跟稠密一样。右：把这 4 个 query 挑中的 key 收拢成一摞，并集 8 行（0、5、9、11、12、13、14、15），"
               "整单对着这 8 行一起乘，没挑的格子用成员位掩掉。这一单就像 MoE 的一个专家，收拢来的 key 行就像分给它的 token。")
    y0 = f.header("每个 query 只挑 2048 个，块稀疏却一块都跳不掉；拼单收拢再乘，就是一次 MoE",
                  "玩具账：一单 4 个 query（位置 12–15），每人从 16 个 key 里挑 4 个（真实是从 32768 里挑 2048）；蓝格 ＝ 挑中，灰格 ＝ 因果看不见",
                  [(BL, "挑中"), (LINE2, "因果看不见"), (RD, "这一块要整块算"), (GR, "收拢后的一摞")])
    PH = 402
    LW = 670
    py = f.panel(0, y0, LW, PH, "Ⓐ 按块跳：块里有一个挑中，整块都得算", RD)
    CS, gx, gy = 34, 84, py + 64
    for c in range(16):
        f.t(gx + c * CS + CS / 2.0, gy - 10, str(c), GY, size=12, anchor="middle")
    f.t(gx - 12, gy - 10, "key", GY, size=12, anchor="end")
    for r, q in enumerate(sorted(TOY)):
        yy = gy + r * CS
        f.t(gx - 12, yy + 23, "q%d" % q, INK, True, 14, "end")
        for c in range(16):
            x = gx + c * CS
            if c > q:
                f.box(x + 1, yy + 1, CS - 2, CS - 2, LINE2, LINE2, 2)
            elif c in TOY[q]:
                f.box(x + 2, yy + 2, CS - 4, CS - 4, BL, BL, 3)
            else:
                f.box(x + 1, yy + 1, CS - 2, CS - 2, "#fff", LINE2, 2)
    for b in range(4):
        f.box(gx + b * 4 * CS, gy, 4 * CS, 4 * CS, "none", RD, 4, sw=2.4)
        f.t(gx + b * 4 * CS + 2 * CS, gy + 4 * CS + 24, "要算", RD, True, 14, "middle")
    yt = gy + 4 * CS + 64
    f.t(30, yt, "4 块里 4 块要算：块激活 100%，算量跟稠密一样", RD, True, 16)
    f.lines(30, yt + 34, LW - 50, [
        "真实情况（512 × 512 的块、每人散着挑 6%）也是 100%"], size=14.5, lh=24, fill=GY)

    RX = LW + 30
    RWd = W - RX
    py2 = f.panel(RX, y0, RWd, PH, "Ⓑ 拼一单、收拢再乘：这一单就是一个专家", GR)
    gx2, gy2 = RX + 96, py2 + 64
    for c, k in enumerate(TOY_UNION):
        f.t(gx2 + c * CS + CS / 2.0, gy2 - 10, str(k), GR, True, 12, "middle")
    f.t(gx2 - 12, gy2 - 10, "key", GY, size=12, anchor="end")
    for r, q in enumerate(sorted(TOY)):
        yy = gy2 + r * CS
        f.t(gx2 - 12, yy + 23, "q%d" % q, INK, True, 14, "end")
        for c, k in enumerate(TOY_UNION):
            x = gx2 + c * CS
            if k in TOY[q]:
                f.box(x + 2, yy + 2, CS - 4, CS - 4, BL, BL, 3)
            else:
                f.box(x + 1, yy + 1, CS - 2, CS - 2, "#fff", LINE2, 2)
    f.box(gx2, gy2, len(TOY_UNION) * CS, 4 * CS, "none", GR, 4, sw=2.4)
    f.t(gx2 + len(TOY_UNION) * CS + 16, gy2 + 2 * CS - 6, "并集 8 行", GR, True, 15)
    f.t(gx2 + len(TOY_UNION) * CS + 16, gy2 + 2 * CS + 18, "ρ ＝ 8 ÷ 4 ＝ 2", GY, True, 14)
    f.t(gx2, gy2 + 4 * CS + 26, "白格：这一行没挑这个 key，乘完用成员位掩掉", GY, size=13.5)
    yt2 = gy2 + 4 * CS + 64
    rows = [("MoE", "一个专家", "路由到它的 token 行", "按组矩阵乘"),
            ("DSA", "一个 query 块", "它的并集 key 行", "按组矩阵乘 ＋ 成员位")]
    cx = [RX + 30, RX + 110, RX + 260, RX + 470]
    for j, h in enumerate(("", "谁是专家", "分给它的行", "怎么算")):
        f.t(cx[j], yt2, h, GY, True, 13.5)
    for i, row in enumerate(rows):
        for j, s in enumerate(row):
            f.t(cx[j], yt2 + 32 + i * 30, s, (GR if i else INK) if j else (GR if i else GY2), True if j == 0 else (i == 1), 15)
    f.t(RX + 30, yt2 + 106, "真实的 ρ：相邻 4 个 query 只有 1.31–1.49，相邻 8 个 1.50–1.87", GY, size=13.5)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "训练里省的是乘法：先把一单要的 key 收拢成一摞，再整块乘", [])
    yb = f.src(yb + 10,
               "玩具挑选是编的（4 个 query、16 个 key、每人挑 4 个）；真实挑选：32K 长文，每个 query 挑 2048 个，「最近 512 个几乎全挑、其余散在整个历史上」。",
               "块激活率：32K、k ＝ 2048／512／128、块 512 × 512 和 1024 × 2048、均匀挑与偏向最近，12 种组合全部 100%（模拟）；k 从 2048 压到 128 时现成的动态块稀疏核 987.8 → 987.6 ms（另一代芯片上的实测）。",
               "ρ（并集膨胀比）＝ 一单的并集行数 ÷ k；真实值是公开的 Hy4 预览版权重在 CPU 上抓的 32K 长文 top-k 统计（4 个 F 层、两份数据的均值）：ρ(4) %.2f–%.2f、ρ(8) %.2f–%.2f、ρ(16) %.2f–%.2f。" % (N.DSA_RHO[4] + N.DSA_RHO[8] + N.DSA_RHO[16]),
               "MoE 的对应关系：专家 ↔ query 块，token ↔ 并集里的 key 行；前向是按组矩阵乘 ＋ 分段 softmax，反向 dKV ＝ 按 key 把部分行加回去，对拍到 1e-6。")
    f.save("fig8b-dsa-moe.svg", yb + 14)


def fig_perrow():
    f = Fig(W, "上：每搬一行，后面跟多少乘加。MoE 一行 token 进专家矩阵乘，约 3800 万次；DSA 一行 key 只跟约 111 万次（M ＝ 1024）。"
               "按峰值算，DSA 这一行的乘法在主核上只要约 1.9 纳秒，主核每行一条 DMA 却要 13 到 16 纳秒，所以搬行要交给小核。"
               "下：交给小核以后把一层切成段，小核收下一段时主核算这一段；切 1、4、8、16 段，前向 13.9、10.9、9.5、8.8 毫秒。")
    y0 = f.header("为什么搬行要交给小工：DSA 每一行后面跟的活太少",
                  "Ⓐ 每搬一行，后面跟多少乘加（条长按对数画）；Ⓑ 交给小工以后切段重叠，前向耗时随段数下降",
                  [(GY2, "MoE"), (RD, "DSA"), (GR, "切段以后")])
    PA = 300
    py = f.panel(0, y0, W, PA, "Ⓐ 每搬一行，后面跟多少乘加（本课推导）", PU)
    X0 = 360
    import math
    SC = 150.0
    for i, (lab, mac, col, note) in enumerate((
            ("MoE：一行 token（12 KB）", MOE_MAC_ROW, GY2, "约 3800 万次 · 主核算它约 65 纳秒"),
            ("DSA：一行 key（1.1 KB）", DSA_MAC_ROW, RD, "约 111 万次 · 主核算它约 1.9 纳秒"))):
        yy = py + 50 + i * 64
        L = (math.log10(mac) - 5) * SC
        f.t(X0 - 16, yy + 22, lab, INK, True, 15, "end")
        f.box(X0, yy + 4, L, 28, col, col, 3)
        f.t(X0 + L + 12, yy + 23, note, col, True, 15)
    yy = py + 50 + 2 * 64 + 8
    f.t(40, yy + 20, "搬一行的固定开销：主核每行一条 DMA 13–16 纳秒（1 KB 到 12 KB 几乎一样）；主核按行取，每行约 11 条标量指令", INK, True, 15)
    f.t(40, yy + 50, "→ MoE 里固定开销被后面的乘法摊没了；DSA 里它是主角，搬一行比算一行还贵。交给小工：每秒收 4.2 亿行，跟师傅并行", RD, True, 15)
    f._pan = None
    PB = 60 + 46 * len(N.DSA_V4_KSWEEP_32H)
    pyb = f.panel(0, py + PA + 24, W, PB, "Ⓑ 交给小工以后，一层切成几段重叠（32 头，合成挑法，只看前向，实测；稠密 4.85 ms）", BL)
    SCk = 50.0
    for i, (k, v) in enumerate(N.DSA_V4_KSWEEP_32H):
        y2 = pyb + 40 + i * 46
        f.t(X0 - 16, y2 + 20, "切 %d 段" % k if k > 1 else "不切段（小工收完、师傅再算）", INK, True, 15, "end")
        c = GR if k == 16 else "#a8dab5"
        f.box(X0, y2 + 4, v * SCk, 24, c, c, 3)
        f.t(X0 + v * SCk + 10, y2 + 21, "%.1f ms" % v, c, True, 15)
    f._pan = None
    yb = f.band(pyb + PB + 20, "ok", "固定开销按行算的活，交给专门搬行的小核；主核只看到连续的大块", [])
    yb = f.src(yb + 10,
               "本课推导（Ⓐ）：MoE 一行 token 进本讲模型的专家（隐藏 6144、专家中间维 2048）：gate ＋ up 6144 × 4096 ＋ down 2048 × 6144 ≈ 3775 万次乘加；"
               "DSA 一行 key 跟 M × (576 ＋ 512) 次，M ＝ 一单 query 数 × 头数 ＝ 1024 时约 111 万；按每设备峰值 1155 TFLOP/s 折成纳秒，是理想值。",
               "实测（单设备微基准）：主核每行一条 DMA 13–16 ns／行，1 KB 到 12 KB 行几乎不变；潜变量整份放进片上暂存、主核按行取，离线编译产物里每行约 11 条标量指令；"
               "小核间接收集两颗每秒 4.2 亿行（含写回），带加法的间接写每秒 3.9 亿行。",
               "实测（Ⓑ）：32 头、合成挑法、rank 0；不切段时小核收 6.0 ms、主核算 6.6 ms，串在一个程序里约 14 ms；切段以后编译器把小核的收集和主核的核交错排。")
    f.save("fig8b-dsa-perrow.svg", yb + 14)


def fig_arch():
    f = Fig(W, "稀疏核的软件架构，五步。第一步 indexer 按 query 块打分（主核 Pallas 核），再在小核上做基数选择，每个 query 出 2048 个下标。第二步账本：主核上一颗 Pallas 核做合并、标记、压紧，"
               "再用 XLA 出段表，每组层算一次、前向反向共用。第三步小核按段把并集 key 行收进暂存。第四步主核的流式核每段连续读一块，做矩阵乘和在线 softmax；"
               "反向算 dQ 和每行的部分梯度。第五步小核把部分梯度按 key 加回 dKV。下半是时间线：前向小核收第 i ＋ 1 段时主核算第 i 段；"
               "反向倒序算，部分梯度分 8 组交给小核加，第一组跟后面几段的计算重叠。")
    y0 = f.header("稀疏核的软件架构：五步，三种执行单元，两两重叠",
                  "框的颜色 ＝ 谁来跑：蓝 主核（TensorCore）上的 Pallas 核；紫 小核（SparseCore）上的 Pallas 核；灰 编译器（XLA）自己生成的计算",
                  [(BL, "主核 Pallas"), (PU, "小核 Pallas"), (GY2, "XLA")])
    PA = 330
    py = f.panel(0, y0, W, PA, "Ⓐ 一层稀疏注意力的数据流（定稿配置）", BL)
    steps = [
        ("① 分块选 top-k", "indexer 按 query 块打分", "小核基数选择 · 2048 个下标", BL, False),
        ("② 账本", "合并 → 标记 → 压紧（一个核）", "段表 · 成员位 · 散射范围", BL, False),
        ("③ 收行", "按段收并集 key 行进暂存", "每秒 4.2 亿行", PU, False),
        ("④ 流式计算", "QK → 掩码 → softmax → PV", "反向：dQ、部分梯度行", BL, False),
        ("⑤ 散射累加", "部分梯度按 key 加回 dKV", "每秒 3.9 亿行，按固定顺序", PU, False),
    ]
    BW, GAP, BX = 236, 37, 26
    BY, BH = py + 52, 150
    for i, (t1, t2, t3, col, dash) in enumerate(steps):
        x = BX + i * (BW + GAP)
        f.box(x, BY, BW, BH, "#fff", col, 8, sw=2.0, dash="6,4" if dash else None)
        f.t(x + 14, BY + 30, t1, col, True, 16)
        f.t(x + 14, BY + 62, t2, INK, False, 13.5, w=BW - 20)
        f.t(x + 14, BY + 90, t3, GY, False, 13.5, w=BW - 20)
        unit = {BL: "主核 Pallas", PU: "小核 Pallas"}[col]
        if i == 0:
            unit = "主核 Pallas ＋ 小核选择"
        if i == 1:
            unit = "主核 Pallas ＋ XLA 段表"
        f.t(x + 14, BY + BH - 16, unit, col, True, 13, w=BW - 20)
        if i:
            f.line(x - GAP + 4, BY + BH / 2.0, x - 6, BY + BH / 2.0, GY2, 1.8)
    yy = BY + BH + 30
    f.t(BX, yy, "账本只跟下标有关：几层共用一份挑选时共用一份账本（按 4 层一组），前向反向共用；32K 每组 3.1–3.5 ms", GY, size=13.5)
    f.t(BX, yy + 26, "XLA 另做几样零碎：delta ＝ Σ dO · O、q 和 dQ 换布局；整模型里反向再收一遍暂存（几十层的暂存留不住，每层多 3.6–4.2 ms）", GY, size=13.5)
    f.t(BX, yy + 52, "④ 每段连续读一块 1024 行，主核看不到一行一行的下标；⑤ 每个小核子核独占 128 个 key、按块顺序加，所以重跑逐位", GY, size=13.5)
    f._pan = None

    PB = 260
    pyb = f.panel(0, py + PA + 24, W, PB, "Ⓑ 时间线：谁跟谁重叠（示意，切 8 段、规约分 8 组）", PU)
    LX = 150
    U = 50.0

    def lane(y, lab, col):
        f.t(LX - 14, y + 19, lab, col, True, 14, "end")
    yf = pyb + 50
    f.t(LX, yf - 10, "前向", INK, True, 15)
    lane(yf + 4, "小核", PU)
    lane(yf + 36, "主核", BL)
    f.box(LX, yf + 36, U * 1.2, 26, BL, BL, 3)
    f.t(LX + U * 0.6, yf + 54, "账本", "#fff", True, 12.5, "middle")
    x0 = LX + U * 1.3
    for i in range(8):
        f.box(x0 + i * U, yf + 4, U - 4, 26, PU, PU, 3)
        f.t(x0 + i * U + (U - 4) / 2.0, yf + 22, "收%d" % (i + 1), "#fff", True, 12, "middle")
        f.box(x0 + (i + 1) * U, yf + 36, U - 4, 26, BL, BL, 3)
        f.t(x0 + (i + 1) * U + (U - 4) / 2.0, yf + 54, "算%d" % (i + 1), "#fff", True, 12, "middle")
    f.box(x0 + 9 * U + 2, yf + 36, 24, 26, GY2, GY2, 3)
    xb = x0 + 9 * U + 30
    f.t(xb, yf - 10, "反向（倒序；重收暂存，图里省略）", INK, True, 15)
    for i in range(8):
        f.box(xb + i * U, yf + 36, U - 4, 26, BL, BL, 3)
        f.t(xb + i * U + (U - 4) / 2.0, yf + 54, "反%d" % (8 - i), "#fff", True, 12, "middle")
        f.box(xb + (i + 1) * U, yf + 4, U - 4, 26, PU, PU, 3)
        f.t(xb + (i + 1) * U + (U - 4) / 2.0, yf + 22, "加%d" % (8 - i), "#fff", True, 12, "middle")
    ytx = yf + 100
    f.t(LX, ytx, "前向：小核收第 i ＋ 1 段的时候，主核在算第 i 段；profile 里主核等收集只露 0.85 ms（合成挑法那格）", GY, size=13.5)
    f.t(LX, ytx + 26, "反向：大段先算；部分梯度分 8 组，每组一次小核调用接着上一组加，第一组跟后面几段的计算重叠", GY, size=13.5)
    f.t(LX, ytx + 52, "灰块：XLA 零碎（delta、换布局）。一层 11.1 ms 的 profile 里主核没有空闲：反向核 5.83、前向核 3.37 ms，其余是 XLA 零碎（后来又去掉一截）", GY, size=13.5)
    f._pan = None
    yb = f.band(pyb + PB + 20, "ok", "照 MoE 的分工：账本在外面算一次，小核搬，主核只做连续的大块乘法，两种核交错排", [])
    yb = f.src(yb + 10,
               "①：indexer 按 1024 个 query 一块打分（融合的 Pallas 核，不落整行分数），选 top-k 用小核上的基数选择（借自 tokamax），第 k 名有并列时按框架原来的规则重选；把账本的并集那一步接到 indexer 每块最后一步，是下一步。",
               "②：账本是整数计算，放在求导之外（照 MoE 的做法：整数映射只依赖专家号，只有收集本身求导）；合并、标记同一个 key 的成员位、把并集压到前面，三步在一颗 Pallas 核里做完，段表和逐段数组是 XLA。",
               "时间线是示意：段的个数照定稿配置（8 段、8 组），块长不按比例；0.85 ms 来自合成挑法、切 8 段那一格的 profile，11.1 ms 的拆账来自真实挑选、一单 1024 行那一版的 profile（32 头）。")
    f.save("fig8b-dsa-arch.svg", yb + 14)


def fig_ladder():
    syn, real, prod = N.DSA_LAD_SYN, N.DSA_LAD_REAL, N.DSA_LAD_PROD
    segs = [
        ("① 只看前向 · 合成挑法 · 默认设置", N.DSA_FWD_DENSE, 1, [
            ("第 1 轮　主核每行一条 DMA 去搬", N.DSA_FWD_V1),
            ("第 3 轮　潜变量放进片上暂存，主核按行取", N.DSA_FWD_V2),
            ("第 7–8 轮　交给小工搬，切 16 段", N.DSA_FWD_V4_K16)]),
        ("② 前向 ＋ 反向 · 换成真实挑选 · 默认设置", real["dense"], 1, [
            ("第 12 轮　换成真实挑选", real["r12"]),
            ("第 13 轮　一单的 query 加倍、段长 1024", real["qb"]),
            ("第 14 轮　层里的整数活挪进账本", real["r14"])]),
        ("③ 前向 ＋ 反向 · 生产设置 · 设备时间", prod["dense"], 2, [
            ("第 20 轮　换口径复测", prod["r20"]),
            ("第 21 轮　一单再加倍（M ＝ 1024）", prod["r21"]),
            ("第 24 轮　去掉核外零碎", prod["r24"]),
            ("第 28 轮　前向两组行交错", prod["r28"])]),
    ]
    nrow = sum(len(s[3]) for s in segs)
    f = Fig(W, "稀疏核一轮一轮的耗时，分三段，每段口径不同，只在段内跟各自的稠密基线比。每行两根条：32 头和 64 头。"
               "第一段只看前向：主核每行搬 43.4 / 53.9 毫秒，放进片上暂存 12.5 / 16.2，交给小工搬并切段 8.8 / 11.6（稠密 4.87 / 9.72）。"
               "第二段换成真实挑选、前向加反向：18.6 / 29.9 → 15.3 / 22.7 → 14.6 / 21.7（稠密 15.7 / 29.8）。"
               "第三段生产设置、设备时间：13.20 / 21.86 → 11.24 / 18.29 → 10.60 / 17.09 → 10.04 / 16.02（稠密 12.98 / 25.73）。"
               "下面一条是账本，每组从 71 毫秒降到约 3.5 毫秒。")
    y0 = f.header("29 轮：从第一版慢 9 倍，到快 1.3–1.6 倍",
                  "每行两根条：上 ＝ 32 头（较小的模型），下 ＝ 64 头（公开的 Hy4）；虚线 ＝ 这一段的稠密基线；三段口径不同，只在段内比；单位毫秒，最慢的设备",
                  [(C32, "32 头"), (C64, "64 头"), (GR, "✓ 比稠密快")])
    RH = 64
    PH = 40 + nrow * RH + len(segs) * 46
    py = f.panel(0, y0, W, PH, "一层注意力、单个设备的实测（32K，每个 query 挑 2048 个）", BL)
    X0, SC = 470, 13.5
    yy = py + 16
    for title, dense, nd, rows in segs:
        yy += 12
        f.box(16, yy, W - 32, 28, "#f1f3f4", "#f1f3f4", 4)
        f.t(30, yy + 20, title, GY, True, 14)
        top = yy + 34
        yy += 34
        for j, (d, col) in enumerate(((dense[0], C32), (dense[1], C64))):
            x = X0 + d * SC
            f.line(x, top, x, top + len(rows) * RH - 6, col, 1.6, dash="5,4", arrow=False)
        for name, (a, b) in rows:
            f.t(X0 - 16, yy + 30, name, INK, True, 14.5, "end")
            for j, (v, col, d) in enumerate(((a, C32, dense[0]), (b, C64, dense[1]))):
                y2 = yy + 6 + j * 26
                c = col
                f.box(X0, y2, v * SC, 20, c, c, 3)
                tag = "%.*f" % (nd, v) + ("  ✓ 快" if v < d else "")
                f.box(X0 + v * SC + 4, y2 + 1, 9 * len(tag) + 8, 19, "#fff", "#fff", 2)
                f.t(X0 + v * SC + 8, y2 + 16, tag, GR if v < d else c, True, 13.5)
            yy += RH
        f.t(W - 30, top + 16, "稠密 %.2f ／ %.2f" % dense if dense[0] < 13 else "稠密 %.1f ／ %.1f" % dense, GY, True, 13.5, "end")
    f._pan = None
    # 账本
    ms = N.DSA_META_STEPS
    PM = 60 + 40 * len(ms)
    pym = f.panel(0, py + PH + 24, W, PM, "账本（元数据）：每组 4 层共用一份，前向反向共用；不含在上面的层耗时里（32 头为主，毫秒）", GY2)
    SCm = 9.0
    for i, (name, v) in enumerate(ms):
        y2 = pym + 40 + i * 40
        f.t(X0 - 16, y2 + 18, name, INK, True, 14.5, "end")
        c = GY2
        f.box(X0, y2 + 3, v * SCm, 22, c, c, 3)
        f.t(X0 + v * SCm + 8, y2 + 19, ("%.0f" % v) if v > 20 else ("%.1f" % v), c, True, 13.5)
    f._pan = None
    yb = f.band(pym + PM + 20, "ok", "每一轮先量清楚卡在哪：先是搬一行太贵，再是数据，再是核外的零碎和账本", [])
    yb = f.src(yb + 10,
               "实测（单设备、同一进程里稀疏和稠密各测；稠密是本讲模型调优用的块稀疏配置，按块掩码）。① 段的稠密虚线取第 3 轮那次读数 4.87 ／ 9.72（第 7–8 轮同批是 4.85 ／ 9.79）。合成挑法「一半挑最近、一半在更早的里随机」，32 头一单 8 个 query（ρ 2.28）、64 头一单 4 个（ρ 1.45），rank 0，只看前向。"
               "第 3 轮 12.5 对稠密 4.87，还慢 2.6 倍；第 7 轮不切段 14.0 / 17.4。合成挑法上前向 ＋ 反向：第 5 轮 60.6 / 79.4，第 9–11 轮小工散射累加、规约分组倒序 27.1 → 21.8 / 34.1 → 30.6（稠密 15.0 / 29.7）。",
               "② 真实挑选：公开的 Hy4 预览版权重在 CPU 上抓的一篇 32K 长文，按上下文并行 8 的负载均衡切法取每个设备的本地 query；8 个设备取最慢的那个；默认运行库设置、每次调用后同步（含主机派发）。",
               "③ 换成生产配置用的运行库设置，连续派发 10 次取平均（设备时间口径，训练里整步异步派发就是这样）；稠密在这个设置下也快了一截，所以第 20 轮看着比第 14 轮快、其实是换了尺子。M ＝ 一单 query 数 × 头数：32 头一单 32 个、64 头一单 16 个。",
               "账本：第 1 轮 71 ms（两次 840 万元素的 scatter ＋ 逐段切片）→ 排序替掉 scatter 7.8–9.9 → 全局一张表 8.9 → 两次排序写成 Pallas 双调网络 5.6 → 压紧写成 Pallas 移位网络 3.7–4.1 → 合并、标记、压紧一个核 3.1–3.6（一单加倍那一轮一度回到 5 ms 多）。")
    f.save("fig8b-dsa-ladder.svg", yb + 14)


def fig_result():
    f = Fig(W, "定稿：一层注意力前向加反向，最慢的设备。32K：32 头稠密 12.98 毫秒、稀疏 10.04，快 1.29 倍，摊进每层的账本后 10.9，快 1.19 倍；"
               "64 头稠密 25.73、稀疏 16.02，快 1.61 倍，含账本 16.9，快 1.52 倍。256K：32 头稠密 186.0、稀疏 31.3，快 5.9 倍，含账本 5.6 倍；"
               "64 头稠密 377.6、稀疏 46.3，快 8.2 倍，含账本 7.8 倍。全部零溢出、重跑逐位。")
    y0 = f.header("定稿：单个分片的注意力计算本身，32K 快 1.3–1.6 倍，256K 快 5.9–8.2 倍",
                  "毫秒；口径：单设备 ＝ 上下文并行里的一个分片（全长 KV 当作已在本地），不含跨卡收集 KV、跨主机通信和小核上的通信争用，稠密同口径；灰 ＝ 稠密，绿 ＝ 稀疏核，浅绿 ＝ 摊进每层的账本（推断）；条越短越快；取最慢的分片",
                  [(GY2, "稠密"), (GR, "稀疏核"), ("#a8dab5", "账本（每层）")])
    PH = 300
    half = (W - 30) / 2
    for k, (title, key, sc, fmt) in enumerate((("32K · 上下文并行 8（每设备 4096 个 query）", N.DSA_FIN_32K, 17.0, "%.2f"),
                                               ("256K · 上下文并行 32（每设备 8192 个 query）", N.DSA_FIN_256K, 1.18, "%.1f"))):
        px = k * (half + 30)
        py = f.panel(px, y0, half, PH, title, GR)
        X0 = px + 110
        for i, (h, lab) in enumerate((("32h", "32 头"), ("64h", "64 头"))):
            sp, de, spm = key[h]
            yy = py + 40 + i * 124
            f.t(px + 20, yy + 40, lab, C32 if i == 0 else C64, True, 16)
            f.box(X0, yy + 8, de * sc, 26, GY2, GY2, 3)
            f.t(X0 + de * sc + 8, yy + 27, fmt % de, GY, True, 14)
            f.box(X0, yy + 44, sp * sc, 26, GR, GR, 3)
            f.box(X0 + sp * sc, yy + 44, (spm - sp) * sc, 26, "#a8dab5", "#a8dab5", 3)
            f.t(X0 + spm * sc + 8, yy + 63, fmt % sp, GR, True, 14)
            f.t(X0, yy + 98, "快 %.2f 倍（含账本 %.2f）" % (de / sp, de / spm) if k == 0 else "快 %.1f 倍（含账本 %.1f）" % (de / sp, de / spm),
                GR, True, 15)
        f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "长度越长，稠密按长度的平方涨、稀疏只按长度涨：32K 小胜，256K 大胜；零溢出、重跑逐位", [])
    yb = f.src(yb + 10,
               "实测（定稿代码：一单 32 头 32 个 query、64 头 16 个，即 M ＝ 1024；段长 1024；生产配置的运行库设置；连续派发 10 次取平均）：32K 8 个设备都测，取最慢的那个比最慢的那个；账本每组 3.1–3.5 ms。",
               "256K：一本 PG-19 书前 262144 个 token、一层的真实挑选，测 5 个设备（第 0、8、15、23、31 个），稠密取同设置的读数；账本每组 6.9–8.4 ms；一本书一层，ρ 会随文本变。",
               "数值：核内改精确算术时，输出、lse、dQ、dKV、d sink 对规格最大相对差 ≤ 3e-6；dKV 原来多个小核子核往同一个 key 加、顺序不定，改成每个子核独占一段 key 以后，全部输出重跑逐位。",
               "一行两篇文档拼满（稠密按文档跳块）：快 1.19 / 1.36 倍，含账本 1.07 / 1.28 倍。")
    f.save("fig8b-dsa-result.svg", yb + 14)


def fig_layer():
    L = N.DSA_LAYER
    f = Fig(W, "一台主机 8 个设备上的整层代理，注意力加 MoE 一个程序前向加反向，64 头。只跑 MoE 30.2 毫秒；只跑注意力，稠密 28.0、稀疏 21.6。"
               "整层：稠密 43.3，比两部分相加少 14.9，MoE 的通信藏在注意力底下；稀疏 52.0，比两部分相加还多一点，一点没藏住；稀疏整层比稠密慢 8.7。"
               "主核等通信：稠密层 2.2 毫秒，稀疏层 14.3。两层串起来：稠密 99.3、稀疏 99.0，持平。")
    y0 = f.header("单测快了，放进整层：省下的师傅时间，变成了等通信",
                  "每组上一行 ＝ 注意力和 MoE 各自单跑、首尾相接；下一行 ＝ 放进一个程序的实测，红块 ＝ 其中主核等通信的合计（位置示意）；整层代理，不是真实的层",
                  [(GY2, "稠密注意力"), (GR, "稀疏注意力"), (OR, "MoE"), (RD, "主核等通信")])
    PH = 350
    py = f.panel(0, y0, W, PH, "整层代理实测（毫秒）", OR)
    X0, SC = 300, 14.0
    rows = [("稠密层", L["att_dense"], L["layer_dense"], L["wait_dense"], GY2),
            ("稀疏层", L["att_sparse"], L["layer_sparse"], L["wait_sparse"], GR)]
    for i, (lab, att, layer, wait, col) in enumerate(rows):
        yy = py + 50 + i * 130
        f.t(X0 - 16, yy + 22, lab, INK, True, 16, "end")
        f.t(X0 - 16, yy + 62, "放进一个程序", GY, True, 13.5, "end")
        f.box(X0, yy + 4, att * SC, 28, col, col, 3)
        f.box(X0 + att * SC, yy + 4, L["moe"] * SC, 28, OR, OR, 3)
        f.t(X0 + att * SC / 2.0, yy + 23, "注意力 %.1f" % att, "#fff", True, 13.5, "middle")
        f.t(X0 + (att + L["moe"] / 2.0) * SC, yy + 23, "MoE %.1f" % L["moe"], "#fff", True, 13.5, "middle")
        f.t(X0 + (att + L["moe"]) * SC + 10, yy + 23, "相加 %.1f" % (att + L["moe"]), GY, True, 14)
        f.box(X0, yy + 44, layer * SC, 28, "#fff", INK, 3, sw=1.6)
        f.box(X0 + (layer - wait) * SC, yy + 44, wait * SC, 28, RD, RD, 3)
        diff = att + L["moe"] - layer
        note = ("实测 %.1f：藏住 %.1f" % (layer, diff)) if diff > 1 else ("实测 %.1f：一点没藏住" % layer)
        f.t(X0 + layer * SC + 10, yy + 63, note, GR if diff > 1 else RD, True, 15)
        f.t(X0 + (layer - wait) * SC - 8, yy + 63, "等通信 %.1f" % wait, RD, True, 13.5, "end")
    f.t(40, py + 50 + 2 * 130 - 4, "两层串起来（后一层的收集能藏到前一层底下）：稠密 %.1f、稀疏 %.1f，持平" % (L["two_dense"], L["two_sparse"]), INK, True, 15)
    f._pan = None
    yb = f.band(py + PH + 20, "warn", "单测快不等于整层快：注意力短了，原来盖在它底下的通信露了出来（第五节那条规矩，又一次）", [])
    yb = f.src(yb + 10,
               "实测（一台主机 8 个设备、专家并行 8；注意力用真实挑选、32K 上下文并行 8 的一个设备那一份；MoE 256 专家 ＋ 1 共享、每 token 挑 8 个，专家权重收集、token 收发、权重梯度规约都交给小核跑，跟稀疏核自己的收集、散射累加共用小核）。",
               "profile（设备 0 一步）：露得最多的是专家 wi 权重的收集（6.3 ms），稀疏层里它的放行到完成点之间只夹了约 1 ms 主核活，稠密层里约 11 ms。",
               "这是代理，不是真实的层：没有真实路由、没有按真实配置给小核分工，通信只在一台主机里。它只说明风险是真的；整层收益要在真实的层上量。")
    f.save("fig8b-dsa-layer.svg", yb + 14)


# ── 2026-11-25 补：拼单两笔账、整模型、上限与口径、indexer 与蒸馏 ─────────────────────────────

def fig_qb():
    import math
    f = Fig(W, "拼单的两笔账。相邻几个 query 拼一单，把各自挑的 key 合成一张单子，一起收、一起算。横轴是一单几个 query：8、16、32、64。"
               "每档两根柱：蓝 ＝ 算的时间，随 ρ 涨（每个 query 要跟整张单子算）；紫 ＝ 收行的时间，单子越大、要收的行越少（197 万、125 万、79 万、50 万行）。"
               "一层的时间取两者中较大的那根。32 头：较大值 4.68、2.97、1.88、2.27 毫秒，一单 32 个最省；64 头每个 query 的活是两倍，较大值 4.68、2.97、3.59、4.54，一单 16 个最省。两种都是一单 1024 行。")
    y0 = f.header("拼单的两笔账：单子越大，收得越少、白算越多；取两者较大的那根最小",
                  "每档两根柱（毫秒，每设备每层前向）：蓝 ＝ 算，随 ρ 涨；紫 ＝ 收 key 行，行数 ÷ 小工每秒 4.2 亿行；绿框 ＝ 两根里较高的那根最矮的一档（本课推导）",
                  [(BL, "算"), (PU, "收行"), (GR, "最优档")])
    PH = 420
    half = (W - 30) / 2.0
    for k, (title, base, hcol, heads) in enumerate((("32 头：一单 32 个 query 最省（一次乘 32 × 32 ＝ 1024 行）", 0.595, C32, 32),
                                                    ("64 头：一单 16 个 query 最省（一次乘 16 × 64 ＝ 1024 行）", 1.19, C64, 64))):
        px = k * (half + 30)
        py = f.panel(px, y0, half, PH, title, hcol)
        bx0, by = px + 70, py + 320
        SC = 52.0
        f.line(bx0 - 10, by, px + half - 20, by, LINE, 1.2, arrow=False)
        vals = []
        for i, (qb, rho, rows) in enumerate(N.DSA_QB_ACCT):
            comp = base * rho
            gat = rows * 1e4 / (N.DSA_SC_GATHER_MROWS * 1e6) * 1e3
            vals.append(max(comp, gat))
        best = vals.index(min(vals))
        for i, (qb, rho, rows) in enumerate(N.DSA_QB_ACCT):
            comp = base * rho
            gat = rows * 1e4 / (N.DSA_SC_GATHER_MROWS * 1e6) * 1e3
            gx = bx0 + i * 145
            f.box(gx, by - comp * SC, 44, comp * SC, BL, BL, 3)
            f.box(gx + 50, by - gat * SC, 44, gat * SC, PU, PU, 3)
            f.t(gx + 22, by - comp * SC - 8, "%.2f" % comp, BL, True, 13, "middle")
            f.t(gx + 72, by - gat * SC - 8, "%.2f" % gat, PU, True, 13, "middle")
            if i == best:
                top = max(comp, gat) * SC
                f.box(gx - 8, by - top - 30, 110, top + 30, "none", GR, 6, sw=2.4)
            f.t(gx + 47, by + 24, "一单 %d 个" % qb, INK, True, 14, "middle")
            f.t(gx + 47, by + 46, "ρ %.2f" % rho, GY, False, 12.5, "middle")
            f.t(gx + 47, by + 66, "收 %d 万行" % rows, GY, False, 12.5, "middle")
        f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "收行受限时，单子往大走：两种模型都停在一单 1024 行，上卡扫出来的也是这一档", [])
    yb = f.src(yb + 10,
               "本课推导：算 ＝ 每设备每层前向的理想时间（32 头 0.595 ms、64 头 1.19 ms，QK 维按矩阵单元补齐到 768）× ρ；这几档一单的行数（query 数 × 头数）都已经够多，矩阵单元喂得饱，计算账只随 ρ 涨。"
               "收行 ＝ 行数 ÷ 小核每秒 4.2 亿行。32K、上下文并行 8 的一个设备（4096 个 query）。",
               "ρ（并集膨胀比）＝ 一单的并集行数 ÷ 2048，取真实挑选 32K 各层的上沿；一单翻倍，ρ 只涨 1.17–1.27 倍，所以行数降 37–41%。",
               "上卡：一单 512 行 → 1024 行，32 头 13.20 → 11.24 ms、64 头 21.86 → 18.29 ms（一层前向 ＋ 反向，最慢的设备）。")
    f.save("fig8b-dsa-qb.svg", yb + 14)


def fig_share():
    import math
    f = Fig(W, "为什么长序列才快。左：每设备 16K 个 token 不变，稠密一步要算的量里注意力占多少：32K 是 923.6 对其余 551.6 万亿次，占 63%；256K 是 7388.7 对 523.6，占 93%；"
               "注意力核自己的效率两种长度都在 52% 到 54%。右：横轴序列长度，纵轴每设备每步万亿次运算，按对数画。稠密注意力随长度成正比涨，其余矩阵乘不变，两条线在约 17K 相交；"
               "稀疏注意力几乎不随长度变，约 115；indexer 打分随长度涨，但只有稠密注意力的 2.6%。")
    y0 = f.header("每设备的 token 不变，变的是每个 query 要看多少 key：注意力从 63% 涨到 93%",
                  "Ⓐ 稠密一步要算的量里各占多少（柱高按占比画，两根总量不同）；Ⓑ 四类计算随序列长度怎么涨（万亿次／设备／步，纵轴按对数画，本课推导，32 头的模型，每设备 16K 个 query）",
                  [(GY2, "稠密注意力"), (OR, "其余矩阵乘"), (GR, "稀疏注意力"), (RD, "indexer 打分")])
    PH = 484
    LW = 430
    py = f.panel(0, y0, LW, PH, "Ⓐ 稠密一步的构成（实测分子）", GY2)
    bx, by, BH = 90, py + 400, 330
    for i, k in enumerate(("32k", "256k")):
        att = N.DSA_ATT_DENSE[k]; tot = N.DSA_NUM_DENSE[k]; rest = tot - att
        x = bx + i * 170
        ha = BH * att / tot
        f.box(x, by - BH, 90, BH - ha, OR, OR, 3)
        f.box(x, by - ha, 90, ha, GY2, GY2, 3)
        f.t(x + 45, by - ha / 2.0 + 6, "%d%%" % round(N.DSA_ATT_SHARE[k]), "#fff", True, 17, "middle")
        f.t(x + 45, by + 24, "32K" if i == 0 else "256K", INK, True, 15, "middle")
        f.t(x + 45, by + 46, "注意力 %.0f ／ 共 %.0f" % (att, tot), GY, False, 12.5, "middle")
    f._pan = None
    RX = LW + 30
    RWd = W - RX
    py2 = f.panel(RX, y0, RWd, PH, "Ⓑ 四类计算随长度怎么涨（本课推导）", BL)
    ax0, ax1 = RX + 90, RX + RWd - 150
    ay0, ay1 = py2 + 400, py2 + 60            # 下 ＝ 1，上 ＝ 10000
    Ts = [8192, 16384, 32768, 65536, 131072, 262144]
    X = lambda T: ax0 + (math.log2(T) - 13) / 5.0 * (ax1 - ax0)
    Y = lambda v: ay0 - math.log10(v) / 4.0 * (ay0 - ay1)
    for v in (1, 10, 100, 1000, 10000):
        f.line(ax0, Y(v), ax1, Y(v), LINE2, 1, arrow=False)
        f.t(ax0 - 10, Y(v) + 5, "%d" % v, GY, False, 12.5, "end")
    for T in Ts:
        f.t(X(T), ay0 + 24, "%dK" % (T // 1024), INK, True, 13.5, "middle")
    series = [(GY2, "稠密注意力", lambda T: N.DSA_ATT_DENSE["32k"] * T / 32768),
              (OR, "其余矩阵乘", lambda T: N.DSA_LIN),
              (GR, "稀疏注意力", lambda T: N.dsa_sparse_att(T)),
              (RD, "indexer 打分", lambda T: N.dsa_indexer_score(T))]
    for col, lab, fn in series:
        pts = [(X(T), Y(fn(T))) for T in Ts]
        f.path(pts, col, 3.0, arrow=False)
        for (x, y) in pts:
            f.box(x - 4, y - 4, 8, 8, col, col, 4)
        f.t(ax1 + 12, pts[-1][1] + 5, "%s %.0f" % (lab, fn(262144)), col, True, 13.5)
    xc = X(N.DSA_CROSS_T)
    f.line(xc, ay1 + 10, xc, ay0, GY, 1.4, dash="5,4", arrow=False)
    f.t(xc + 8, ay1 + 24, "约 17K：注意力 ＝ 其余矩阵乘", INK, True, 14)
    f.t(X(32768) + 8, Y(N.DSA_ATT_DENSE["32k"]) - 12, "%.0f" % N.DSA_ATT_DENSE["32k"], GY, True, 12.5)
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "序列越长，稠密里注意力越占大头，稀疏能省的越多；短序列上注意力本来就不大，省不出多少", [])
    yb = f.src(yb + 10,
               "实测分子（每设备每步，前向 ＋ 反向）：其余矩阵乘 ＝ 投影、稠密与专家 FFN、输出头（含 MTP 层）484.5，加 MTP 输出头 35.0、视觉塔（按实际图像，32K 32.1、256K 4.1）；"
               "稠密注意力 ＝ 2 · t · T · 32 头 · 512 ÷ 2 × 3 × 35 层，32K 923.6、256K 7388.7；注意力核效率 ＝ 923.6 ÷ 1.55 s、7388.7 ÷ 11.86 s，再 ÷ 1155。",
               "本课推导：交点 ＝ 32768 × 484.5 ÷ 923.6 ≈ 17,200（只跟其余矩阵乘比；加上 MTP 头约 18K，再加视觉塔约 20K）。稀疏注意力照框架的写法（每个 query 2048 个 key、不计并集白算）；"
               "indexer 打分 ＝ 每个 query 对全部可见 key 算 32 头 × 128 维、只在 11 层、只有前向。其余矩阵乘那条只画线性层 484.5。")
    f.save("fig8b-dsa-share.svg", yb + 14)


def fig_step():
    f = Fig(W, "整模型一步的拆账，32 头的模型，64 片，每设备 16K 个 token，同一种切法放大。32K：稠密 4.49 秒，其中注意力核 1.55；稀疏 5.11 秒，注意力核 1.09，"
               "但多了 indexer、账本、等小核 0.49，露出的通信从 0.35 涨到 0.72，比稠密慢。256K：稠密 15.27 秒，注意力核 11.86；稀疏 7.82 秒，注意力核 1.48，"
               "多了 indexer 1.80、等小核 0.68、账本 0.09，露出的通信 1.14，快约 1.95 倍。虚线是阿姆达尔下限约 3.9 秒：只有注意力核那段会缩，其余不变。")
    y0 = f.header("一夜优化之前那一版的拆账：32K 慢 15%，256K 快 1.95 倍",
                  "每行一步（秒，profile 中位步，同一把尺子）；灰 ＝ 注意力核，浅灰 ＝ 注意力的投影等，红 ＝ 稀疏新增（indexer、账本、等小核），橙 ＝ MoE 与其余计算，紫 ＝ 露出的通信",
                  [(GY2, "注意力核"), ("#bdc1c6", "投影等"), (RD, "稀疏新增"), (OR, "MoE 等"), (PU, "露出的通信"), (LINE, "其余")])
    PH = 440
    py = f.panel(0, y0, W, PH, "一步的时间去哪了（32 头的模型 · 64 片 · 每设备 16K token · 合成挑选）", BL)
    X0, SC = 250, 0.066
    cols = (GY2, "#bdc1c6", RD, OR, PU, LINE)
    rows = (("32K 稠密", "32k_dense"), ("32K 稀疏", "32k_dsa"), ("256K 稠密", "256k_dense"), ("256K 稀疏", "256k_dsa"))
    for i, (lab, key) in enumerate(rows):
        yy = py + 46 + i * 80 + (14 if i >= 2 else 0)
        f.t(X0 - 16, yy + 24, lab, INK, True, 16, "end")
        x = X0
        segs = N.DSA_STEP[key]
        for j, v in enumerate(segs):
            if v <= 0:
                continue
            f.box(x, yy + 4, v * SC, 32, cols[j], "#fff" if j < 5 else LINE, 2)
            if v * SC > 64 and j in (0, 2, 3):
                f.t(x + v * SC / 2.0, yy + 26, "%.2f" % (v / 1000), "#fff", True, 13, "middle")
            x += v * SC
        tot = sum(segs)
        tag = "%.2f 秒" % (tot / 1000)
        if key == "32k_dsa":
            tag += " · 日志步时慢 %d%%" % N.DSA_WM_SLOW_32K_PCT
        if key == "256k_dsa":
            tag += " · 日志步时快 %.2f 倍" % N.DSA_WM_X[0]
        f.t(x + 10, yy + 26, tag, GR if key == "256k_dsa" else (RD if key == "32k_dsa" else GY), True, 15)
    yA = py + 46 + 3 * 80 + 14
    xa = X0 + N.DSA_AMDAHL_FLOOR * 1000 * SC
    f.line(xa, yA - 8, xa, yA + 50, INK, 1.8, dash="5,4", arrow=False)
    f.t(xa - 6, yA + 66, "阿姆达尔下限 ≈ %.1f 秒（只有注意力核会缩）" % N.DSA_AMDAHL_FLOOR, INK, True, 14, "end")
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "注意力占得越多，稀疏省得越多；256K 离下限还差三块：挑下标、小核搬运的等待、多露出的通信", [])
    yb = f.src(yb + 10,
               "实测（64 片 v7x，每设备 16K 个 token：32K 上下文并行 2 ＋ FSDP 64，256K 上下文并行 16 ＋ FSDP 8，参数都切 128 份；稠密与稀疏同配置）。日志步时 32K 4.503 对 5.167 秒，256K 15.337 对 7.880 秒（同一批）。",
               "这个模型没有训过的 indexer：挑选用按真实挑选统计拟合的合成选择（32K ρ(32) 2.85，比真实偏高 12–15%；256K 3.71，偏低约 10%，按真实挑选约多 0.2 秒，推算），indexer 照常算。",
               "本课推导（下限）：稠密 15.27 秒里只有注意力核 11.86 秒会随稀疏缩，缩到稀疏这一侧的计算量（稀疏注意力 115.0 ＋ indexer 197.4 万亿次）÷ 稠密注意力 7388.7 ＝ 4.2%，约 0.50 秒；投影等 1.40 秒和其余 2.01 秒不动：下限约 3.9 秒，上限约 3.9 倍。稀疏这边的投影实测只要 0.70 秒（吸收式不用为核构完整的 K/V），按它算上限还能再高一些（约 4.8 倍），这里取保守的 3.9。")
    f.save("fig8b-dsa-step.svg", yb + 14)


def fig_mfu():
    f = Fig(W, "同一步 6.51 秒（256K，不开蒸馏，一夜优化以后），三种分子。稠密等价：按稠密那一步该算的 7912.3 万亿次，有效 MFU 105.2%，超过 100%，说明比稠密跑满峰值还快，不是利用率。"
               "框架自带计数 2523.4，33.6%，其中 74% 是按 35 层、前向加反向数的 indexer。实际执行 836.1，11.1%。下：indexer 的三层折扣，每对的运算是主干注意力的四分之一，"
               "不开蒸馏时只有前向再剩三分之一，只在 35 层里的 11 层算再剩 11/35，合起来是稠密注意力的 2.6%。")
    y0 = f.header("稀疏以后的 MFU：同一步，三种分子差近十倍；实际只有一成左右，却快了一倍多",
                  "Ⓐ 256K、不开蒸馏的那一步（6.51 秒）按三种分子算的 MFU（分母相同：步时 × 1155）；Ⓑ indexer 为什么只占一点：三层折扣（本课推导）",
                  [(GY2, "稠密等价"), (OR, "框架自带计数"), (RD, "其中按 35 层 × 前反向数的 indexer"), (GR, "实际执行")])
    PA = 312
    py = f.panel(0, y0, W, PA, "Ⓐ 三种分子（256K，同一步）", BL)
    X0, SC = 330, 6.4
    m = N.DSA_MFU3_FIN
    rows = (("稠密等价：照稠密该算的数", m[0], N.DSA_NUM_DENSE["256k"], GY2, "超过 100%：比稠密跑满峰值还快"),
            ("框架自带计数", m[1], N.DSA_NUM_FRAME["256k"], OR, "比实际执行高估 3 倍"),
            ("实际执行：这一步真做了的", m[2], N.DSA_NUM_EXEC["256k"][0], GR, "衡量硬件用得好不好，看这一个"))
    for i, (lab, v, num, col, note) in enumerate(rows):
        yy = py + 46 + i * 74
        f.t(X0 - 16, yy + 22, lab, INK, True, 15, "end")
        f.t(X0 - 16, yy + 44, "分子 %.1f 万亿次" % num, GY, False, 13, "end")
        f.box(X0, yy + 4, v * SC, 30, col, col, 3)
        if i == 1:
            fr = N.DSA_FRAME_IDX_256K / num
            f.box(X0 + v * SC * (1 - fr), yy + 4, v * SC * fr, 30, RD, RD, 3)
        f.t(X0 + v * SC + 10, yy + 25, "%.1f%%" % v, col, True, 16)
        f.t(max(X0 + v * SC + 80, X0 + N.DSA_MFU_DENSE_LOCK * SC + 16) if i else X0 + v * SC + 80, yy + 25, note, GY, False, 13.5)
    xd = X0 + N.DSA_MFU_DENSE_SAMEJOB * SC
    f.line(xd, py + 38, xd, py + 46 + 3 * 74 - 10, INK, 1.4, dash="5,4", arrow=False)
    f.t(xd + 6, py + 46 + 3 * 74 + 6, "同作业稠密（15.33 秒）的有效 MFU %.1f%%" % N.DSA_MFU_DENSE_SAMEJOB, INK, True, 13.5)
    f._pan = None
    PB = 250
    pyb = f.panel(0, py + PA + 24, W, PB, "Ⓑ indexer 的三层折扣：它的量是稠密注意力的几分之一（256K）", RD)
    steps = (("稠密注意力（每对 32 头 × 512 维，前向 ＋ 反向，35 层）", 1.0),
             ("每对只算 32 头 × 128 维：× 1/4", 1 / 4.0),
             ("不开蒸馏时只有前向（挑下标不求导）：× 1/3", 1 / 12.0),
             ("只在 11 层算（其余 24 层借下标）：× 11/35", N.DSA_IDX_DISCOUNT))
    SC2 = 800.0
    for i, (lab, v) in enumerate(steps):
        yy = pyb + 44 + i * 46
        f.t(X0 - 16, yy + 20, lab, INK, True, 13.5, "end") if False else None
        f.t(30, yy + 20, lab, INK, True, 14)
        x1 = 520
        f.box(x1, yy + 4, max(v * SC2, 3), 24, RD if i == 3 else "#f6aea9", RD if i == 3 else "#f6aea9", 3)
        f.t(x1 + max(v * SC2, 3) + 10, yy + 21, "%.1f%%" % (v * 100) if v < 1 else "100%", RD, True, 14)
    f._pan = None
    yb = f.band(pyb + PB + 20, "ok", "实际执行那根短，是两笔都不记账：拼单多做的乘法，和等待的时间", [])
    yb = f.src(yb + 10,
               "本课推导：MFU ＝ 分子 ÷（6.510 秒 × 1155）。稠密等价分子 ＝ 同长度稠密格的有效分子 7912.3；实际执行 ＝ 线性层 484.5 ＋ MTP 头 35.0 ＋ 视觉 4.1 ＋ 稀疏注意力 115.0 ＋ indexer 打分 195.0 ＋ indexer 投影 2.4 ＝ 836.1；"
               "框架自带计数 2523.4，其中 indexer 打分 1861.6（按 35 层 × 3 数）。",
               "实际执行里稀疏注意力照框架按 256 ＋ 256 维、每个 query 2048 个 key 数，不计并集的白算；核在矩阵单元上实际做的是吸收式 576 ＋ 512 维、ρ 倍的行，比分子多。所以实际执行 MFU 低有两个原因：一是分子扣得干净，二是 indexer、等小核、露出的通信这些不产生分子的时间。",
               "Ⓑ：7388.7 × 1/4 × 1/3 × 11/35 ≈ 193.5，加上按头加权那一项是 195.0；对稠密注意力是 2.6%。")
    f.save("fig8b-dsa-mfu.svg", yb + 14)


def fig_aux():
    f = Fig(W, "上：indexer 只在 35 层里的 11 层算：第 0 层、每 4 层一组的首层、最后一层和 MTP 层，蓝色；其余 24 层灰色，借本组首层挑出的下标。"
               "借来的下标拿到它自己最优挑法 94% 到 99% 的概率质量。下：热身阶段只训 indexer，256K 一步从 84.2 秒到 70.5 秒、24.3 秒，再到 14.3 秒，"
               "24.3 秒里 20.4 秒是覆盖全部可见 key 的教师；稀疏阶段的蒸馏只在挑中的 2048 个 key 上，32K 多 0.85 秒、256K 多 0.96 秒，不随长度涨。")
    y0 = f.header("indexer 少挑几层，蒸馏少看几个 key",
                  "Ⓐ 哪几层跑 indexer（蓝 ＝ 自己打分挑下标的 F 层，灰 ＝ 借本组下标的 S 层）；Ⓑ 两个训练阶段里蒸馏的代价（256K，一步的秒数）",
                  [(BL, "F 层"), (GY2, "S 层"), (RD, "教师"), ("#f6aea9", "热身一步的其余")])
    PA = 188
    py = f.panel(0, y0, W, PA, "Ⓐ 35 层里只有 11 层跑 indexer（32 头的模型：34 层 ＋ 1 层 MTP，每 4 层一组）", BL)
    BW, GAP, X0 = 33, 4, 30
    yb0 = py + 50
    for i in range(35):
        isF = (i == 0) or (i == 33) or (i == 34) or ((i - 1) % 4 == 0 and 1 <= i <= 32)
        x = X0 + i * (BW + GAP)
        col = BL if isF else GY2
        f.box(x, yb0, BW, 46, col if isF else "#fff", col, 4, sw=1.6)
        f.t(x + BW / 2.0, yb0 + 29, "F" if isF else "S", "#fff" if isF else GY, True, 14, "middle")
        lab = "MTP" if i == 34 else str(i)
        f.t(x + BW / 2.0, yb0 + 68, lab, GY, False, 11.5, "middle")
        if not isF and 1 <= i <= 32:
            f0 = X0 + (1 + 4 * ((i - 1) // 4)) * (BW + GAP) + BW / 2.0
            if (i - 1) % 4 == 1:
                f.path([(f0, yb0 - 4), (f0, yb0 - 14), (x + BW * 3 + GAP * 2 - 4, yb0 - 14)], BL, 1.4, arrow=False)
    f.t(X0, yb0 + 104, "S 层用本组 F 层的挑选：挑中的注意力权重，是它自己最优挑法能挑中的 94–99%", INK, True, 14)
    f._pan = None
    PB = 300
    pyb = f.panel(0, py + PA + 24, W, PB, "Ⓑ 蒸馏的代价：热身阶段的教师按长度的平方涨，稀疏阶段的不涨", RD)
    XB, SCb = 470, 9.6
    rows = (("热身 · 旧版：主干稠密、对全部参数求导", N.DSA_WARM[0], 0),
            ("热身 · 主干稀疏、只对 indexer 求导", N.DSA_WARM[1], 0),
            ("热身 · 再把教师写进 Pallas 核", N.DSA_WARM[2], N.DSA_WARM_TEACHER[0]),
            ("热身 · 前向统计跨重算存下、教师核分块调小", N.DSA_WARM[3], 0))
    for i, (lab, v, tch) in enumerate(rows):
        yy = pyb + 44 + i * 50
        f.t(XB - 16, yy + 22, lab, INK, True, 14.5, "end")
        f.box(XB, yy + 4, v * SCb, 28, "#f6aea9" if i < 3 else GR, "#f6aea9" if i < 3 else GR, 3)
        if tch:
            f.box(XB, yy + 4, tch * SCb, 28, RD, RD, 3)
            f.t(XB + tch * SCb / 2.0, yy + 23, "教师 %.1f（84%%）" % tch, "#fff", True, 13, "middle")
        f.t(XB + v * SCb + 10, yy + 23, "%.1f 秒" % v, RD if i < 2 else (GR if i == 3 else INK), True, 15)
    yy = pyb + 44 + 4 * 50 + 6
    f.t(30, yy + 20, "稀疏阶段的蒸馏：只在每个 query 挑中的 2048 个 key 上算，开蒸馏多出的时间 32K、256K 都约 0.58 秒，不随长度涨", GR, True, 14.5)
    f._pan = None
    yb = f.band(pyb + PB + 20, "ok", "热身的教师看全部可见 key，按长度的平方涨；稀疏阶段只看挑中的 2048 个，不涨", [])
    yb = f.src(yb + 10,
               "实测（同上整模型配置，256K，合成挑选）：热身旧版 84.2 秒（主干稠密、所有参数求导、教师用 XLA 流式算）→ 70.5 秒（主干走稀疏核、只对 indexer 求导、主干不做反向也不为反向存激活；估计显存 91.9 → 23.6 GiB）→ 24.3 秒（教师四个核写成 Pallas）。"
               "→ 14.3 秒（前向统计跨重算存下，教师核的 query 块调到 32）。按 1000 步算约 23.4 → 4 小时（推算，按这一格的批量）。",
               "热身的教师覆盖每个 query 的全部可见 key（O(T²)），256K 上比稀疏阶段多约 T ÷ 2k ≈ 64 倍的对数；稀疏阶段只在挑中集合上算，跟框架原来先算稠密再掩码的写法数学等价、只差求和顺序。",
               "S 层蒸馏：KL 里跟学生有关的项对教师是线性的，所以逐层 KL 取平均与「组内平均教师算一个 KL」梯度相同、loss 差一个常数（CPU 核对梯度 1e-7 量级）。32K 实测：只蒸馏 F 层 6.03 秒；全层蒸馏、每层重算学生 8.00 秒，学生只算一次 6.84 秒（后两格同作业，CPU 上梯度相对差 ≤ 2e-7）。")
    f.save("fig8b-dsa-aux.svg", yb + 14)


def fig_len():
    import math
    L = N.DSA_FIN_LEN
    ks = ("32K", "64K", "128K", "256K")
    SHOWN = {"32K": ("0.90", "0.76"), "64K": ("1.13", "0.97"), "128K": ("1.57", "1.35"), "256K": ("2.36", "2.06")}   # 照记录的写法（2.355、0.755 进位）
    for k, (a, b) in SHOWN.items():
        assert abs(float(a) - L[k][0] / L[k][1]) < 0.006 and abs(float(b) - L[k][0] / L[k][2]) < 0.006, k
    f = Fig(W, "一夜优化以后，四个长度的整模型读数，每设备都是 16K 个 token，同作业对稠密。不开蒸馏：32K 0.90 倍，64K 1.13 倍，128K 1.57 倍，256K 2.36 倍。"
               "全层蒸馏：0.76、0.97、1.35、2.06 倍。开蒸馏比不开多 0.89 到 1.01 秒，四个长度差不多。稠密步时 4.59、6.01、8.99、15.33 秒。")
    y0 = f.header("四个长度：越长越快，32K 和 64K 之间打平",
                  "左：每个长度一组三根柱，一步几秒（同作业）；右：对稠密快几倍，虚线 ＝ 打平；每设备都是 16K 个 token，32 头的模型，64 片，合成挑选",
                  [(GY2, "稠密"), (GR, "稀疏，不开蒸馏"), (BL, "稀疏，全层蒸馏")])
    PH = 380
    LW = 700
    py = f.panel(0, y0, LW, PH, "Ⓐ 一步几秒", GY2)
    bx0, by, SC = 70, py + 300, 16.0
    for i, k in enumerate(ks):
        d, y, kd = L[k]
        gx = bx0 + i * 155
        for j, (v, col) in enumerate(((d, GY2), (y, GR), (kd, BL))):
            f.box(gx + j * 40, by - v * SC, 34, v * SC, col, col, 3)
            f.t(gx + j * 40 + 17, by - v * SC - 8, "%.1f" % v, col, True, 12.5, "middle")
        f.t(gx + 57, by + 24, k, INK, True, 15, "middle")
    f._pan = None
    RX = LW + 30
    RWd = W - RX
    py2 = f.panel(RX, y0, RWd, PH, "Ⓑ 对稠密快几倍", GR)
    ax0, ax1 = RX + 70, RX + RWd - 60
    ay0, ay1 = py2 + 300, py2 + 50
    Y = lambda v: ay0 - (v - 0.5) / 2.0 * (ay0 - ay1)
    X = lambda i: ax0 + i * (ax1 - ax0) / 3.0
    for v in (0.5, 1.0, 1.5, 2.0, 2.5):
        f.line(ax0, Y(v), ax1, Y(v), LINE2 if v != 1.0 else GY, 1.4 if v == 1.0 else 1, dash="5,4" if v == 1.0 else None, arrow=False)
        f.t(ax0 - 10, Y(v) + 5, "%.1f×" % v, GY, False, 12.5, "end")
    for i, k in enumerate(ks):
        f.t(X(i), ay0 + 24, k, INK, True, 14, "middle")
    for col, idx, dy in ((GR, 1, -12), (BL, 2, 22)):
        pts = [(X(i), Y(L[k][0] / L[k][idx])) for i, k in enumerate(ks)]
        f.path(pts, col, 3.0, arrow=False)
        for i, (x, yv) in enumerate(pts):
            f.box(x - 5, yv - 5, 10, 10, col, col, 5)
            f.t(x, yv + dy, SHOWN[ks[i]][idx - 1], col, True, 13, "middle")
    f.t(ax1 - 4, Y(1.0) + 20, "打平", GY, True, 13, "end")
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "开蒸馏比不开多 0.89–1.01 秒，四个长度差不多：蒸馏只在挑中的 key 上算，不随长度涨", [])
    yb = f.src(yb + 10,
               "实测（同一份代码；每个长度稠密与稀疏同一作业）：64 片 v7x，上下文并行 2／4／8／16，每设备 16K 个 token；256K 第 12–21 步中位数，64K／128K 第 20–35 步，32K 常驻作业第 12–19 步。"
               "32K 没开 indexer 流水和学生散射那两项（在 32K 上更慢）。挑选是按真实挑选统计拟合的合成选择；256K 合成的 ρ 比真实偏低约 10%。",
               "全层蒸馏 ＝ 组内 S 层也用自己的注意力当教师各算一个 KL（主报口径）。")
    f.save("fig8b-dsa-len.svg", yb + 14)


def fig_kd():
    st = N.DSA_KD_STAGES
    f = Fig(W, "256K 全层蒸馏一夜的阶段，每一阶段对同作业稠密。修前每个 S 层重算一遍学生 11.57 秒；S 层不重算学生 9.94 秒；再去三处重复 9.13 秒；"
               "加 indexer 有序 top-k 与流水、散射原地累加 8.07 秒；小核收集 4 路、F 层学生改写 7.49 秒；最终 7.46 秒，是稠密 15.33 秒的 2.06 倍。"
               "红色是去掉的重复计算。开蒸馏比不开多出的时间从约 2.1 秒降到 0.95 秒。")
    y0 = f.header("全层蒸馏 11.57 → 7.09 秒：头两刀和最后一刀都是去掉重复计算",
                  "每行一步（秒，256K，第 12–21 步中位）；虚框 ＝ 这一阶段砍掉的部分：红 ＝ 去掉重复计算，绿 ＝ indexer 和小核搬运",
                  [(BL, "全层蒸馏一步"), (RD, "去掉的重复计算"), (GR, "其余提速")])
    PH = 60 + 52 * len(st)
    py = f.panel(0, y0, W, PH, "256K，全层蒸馏（主报口径），64 片 v7x", BL)
    X0, SC = 470, 62.0
    for i, (lab, v) in enumerate(st):
        yy = py + 40 + i * 52
        f.t(X0 - 16, yy + 22, lab, INK, True, 14.5, "end")
        f.box(X0, yy + 4, v * SC, 28, BL, BL, 3)
        if i:
            prev = st[i - 1][1]
            col = RD if i in (1, 2, len(st) - 1) else GR
            f.box(X0 + v * SC, yy + 4, (prev - v) * SC, 28, "none", col, 3, sw=2, dash="5,3")
            f.t(X0 + prev * SC + 10, yy + 23, "%.2f 秒（−%.2f）" % (v, prev - v), col, True, 14)
        else:
            f.t(X0 + v * SC + 10, yy + 23, "%.2f 秒" % v, BL, True, 14)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "蒸馏贵在重复：同一个学生每层算一遍、教师单独再收再算、前向核重跑；红色几刀合计约 2.8 秒", [])
    yb = f.src(yb + 10,
               "实测（同作业对照；阶段按当晚报告的组合顺序，⑤–⑦ 那几项各自在修后底上测过，单项收益不能直接相加）：S 层不重算学生 −1.31 秒；无 sink 的 lse 存成残差、重算时不再整遍重跑注意力前向核 −0.36；"
               "S 层教师与 KL 进注意力核、复用已收好的段 −0.35；反向的教师由注意力反向核顺手出 −0.11；indexer 有序 top-k ＋ 打分与选择流水 −1.12；小核收集 4 路 −0.31。",
               "最后一行是上午另一个作业：同作业里改前 7.465 秒、改后 7.092 秒（稠密 15.357 秒，2.17 倍）；上一行 7.457 秒是夜里那个作业。"
               "修前那一格不分输出头，之后各格分 4 份（共享学生要把缓存留到反向，不分放不下）；同设置下修前是 11.25 秒。数值：CPU 上梯度相对差 ≤ 2e-7；卡上融合改了求和顺序，24 步末 loss 相对差 ≤ 1.2e-5。")
    f.save("fig8b-dsa-kd.svg", yb + 14)


def fig_teacher():
    f = Fig(W, "教师放在前向还是反向。上：放在前向。前向核扫完所有段，每个头的归一化总和才定下来，所以教师要等扫完再从头扫一遍：再收一遍 key 行、再乘一遍；"
               "也不能边扫边先按头加起来，因为每个头的修正系数不一样；把每个头每个 key 的值存下来又太大。拆账里这一遍每步约 0.24 秒，再多占几 GB 显存。"
               "下：放在反向。反向核本来就要用最终的归一化总和重算每个头的概率，教师只是顺手多一次乘和一次按头求和，每步约多 37 毫秒。"
               "反向从后往前走，组里的 S 层先跑、教师先累加，最后 F 层的学生做一次反向。loss 的数值通过一个零值的探针输入带回日志。")
    y0 = f.header("教师放进反向：省掉前向的第二遍，反向只多一次乘",
                  "两条时间线各画一层注意力的前向和反向（示意，块长不按比例）；红 ＝ 只为教师多出来的活；数字是 256K 整模型的实测",
                  [(BL, "注意力核本来就要做的"), (RD, "只为教师多出来的")])
    PA = 250
    py = f.panel(0, y0, W, PA, "Ⓐ 教师在前向：前向要扫两遍", RD)
    LX, U = 200, 54.0
    def lane(y, lab, col):
        f.t(LX - 14, y + 20, lab, col, True, 14.5, "end")
    yf = py + 50
    lane(yf, "前向", BL)
    for i in range(6):
        f.box(LX + i * U, yf + 2, U - 4, 28, BL, BL, 3)
    f.t(LX + 3 * U, yf - 8, "第一遍：在线 softmax，扫到最后一段归一化总和才定", INK, True, 13.5, "middle")
    x2 = LX + 6 * U + 16
    for i in range(6):
        f.box(x2 + i * U, yf + 2, U - 4, 28, RD, RD, 3)
    f.t(x2 + 3 * U, yf - 8, "第二遍：再收、再乘，按头归一后求和 ＝ 教师", RD, True, 13.5, "middle")
    yb = yf + 64
    lane(yb, "反向", BL)
    for i in range(6):
        f.box(LX + i * U, yb + 2, U - 4, 28, BL, BL, 3)
    f.t(LX, yb + 70, "为什么不能一遍做完：每个头的修正系数不同，不能边扫边按头加；存每头每 key 的中间值，是教师本身的 32 倍大", GY, True, 13.5)
    f.t(LX, yb + 96, "代价：每步约 0.24 秒（拆账里的教师核），外加几 GB 显存", RD, True, 14)
    f._pan = None
    PB = 300
    pyb = f.panel(0, py + PA + 24, W, PB, "Ⓑ 教师在反向：反向核本来就重算概率，顺手出教师", GR)
    yf = pyb + 50
    lane(yf, "前向", BL)
    for i in range(6):
        f.box(LX + i * U, yf + 2, U - 4, 28, BL, BL, 3)
    f.t(LX + 6 * U + 16, yf + 21, "只扫一遍；教师不在这里", GR, True, 14)
    yb = yf + 56
    lane(yb, "反向", BL)
    for i in range(6):
        f.box(LX + i * U, yb + 2, U - 4, 28, BL, BL, 3)
        f.box(LX + i * U + U - 14, yb + 2, 10, 28, RD, RD, 2)
    f.t(LX + 6 * U + 16, yb + 21, "用最终的归一化总和重算概率 → 多一次乘、一次按头求和（每步约多 37 毫秒）", RD, True, 14)
    yo = yb + 66
    f.t(LX - 14, yo + 18, "一组 4 层", INK, True, 14.5, "end")
    order = [("S 层", GY2), ("S 层", GY2), ("S 层", GY2), ("F 层", BL)]
    for i, (lab, col) in enumerate(order):
        x = LX + i * 150
        f.box(x, yo, 130, 30, "#fff", col, 4, sw=1.6)
        f.t(x + 65, yo + 20, lab + "：教师累加" if i < 3 else "F 层：学生反向一次", col, True, 13.5, "middle")
        if i < 3:
            f.line(x + 132, yo + 15, x + 148, yo + 15, GY2, 1.6)
    f.t(LX + 4 * 150 + 10, yo + 20, "反向从后往前：S 层先跑，教师先加好，F 层的学生只反向一次", GY, True, 13.5)
    f.t(LX, yo + 66, "loss 的数值：零值探针输入 → 反向把 KL 当它的梯度带回日志", GY, True, 13.5)
    f._pan = None
    yb2 = f.band(pyb + PB + 20, "ok", "放在前向：多一整遍、多几 GB；放在反向：每段末尾顺手一乘", [])
    yb2 = f.src(yb2 + 10,
               "实测（同作业）：256K 全层蒸馏 7.465 → 7.092 秒，对稠密 15.357 秒从 2.06 倍到 2.17 倍；32K 6.097 → 5.845 秒。离线显存峰值 90.82 → 91.01 GiB。",
               "拆账：上一版开蒸馏比不开多出的约 0.95 秒里，教师核 243 毫秒（S 层 24 层前向 ＋ F 层）、注意力反向核出教师多 37 毫秒；改成只在反向出以后，开蒸馏多出约 0.58 秒（对不开蒸馏 6.51 秒那一格）。",
               "数值：CPU 上核的梯度逐位、探针的值等于 KL 之和；整模型小配置 3 步 loss 与梯度范数逐位。卡上第 0 步 loss 与梯度范数相同；24 步内融合改了求和顺序，loss 相对差 ≤ 1.7e-5。")
    f.save("fig8b-dsa-teacher.svg", yb2 + 14)


def fig_scq():
    f = Fig(W, "小核上的排队，三种情形。上：以前，主核的计算还长，小核上散射慢一点，排在后面的 MoE 梯度规约照样藏在主核计算底下，关掉原地散射只慢约 0.15 秒。"
               "中：现在主核的时间被其他优化压短了，非原地散射在小核上占得久，梯度规约排在它后面，主核做完只能等，256K 慢约 2.2 秒。下：原地散射，小核上占得短，规约及时做完。")
    y0 = f.header("瓶颈会搬家：主核快了以后，小核上的排队整段露出来",
                  "每组两条车道（示意，块长不按比例）：上 ＝ 主核，下 ＝ 小核；紫 ＝ 稀疏核的收行与散射，橙 ＝ MoE 的梯度规约，红 ＝ 主核停着等",
                  [(BL, "主核计算"), (PU, "小核：收行／散射"), (OR, "小核：MoE 梯度规约"), (RD, "主核等")])
    rows = [("以前：主核计算长", 11.0, 4.5, 2.8, "关掉原地散射只慢约 0.15 秒：规约还藏得住", GR),
            ("现在，非原地散射", 6.5, 4.5, 2.8, "散射占得久，规约排在后面：主核等，256K 慢约 2.2 秒", RD),
            ("现在，原地散射", 6.5, 2.2, 2.8, "散射占得短，规约跟着做完", GR)]
    PH = 70 + 112 * len(rows)
    py = f.panel(0, y0, W, PH, "一段反向里的两条车道（示意）", PU)
    X0, U = 330, 60.0
    for i, (lab, tc, sc, rs, note, col) in enumerate(rows):
        yy = py + 40 + i * 112
        f.t(X0 - 16, yy + 30, lab, INK, True, 15, "end")
        f.box(X0, yy, tc * U, 26, BL, BL, 3)
        f.box(X0, yy + 34, sc * U, 26, PU, PU, 3)
        f.box(X0 + sc * U, yy + 34, rs * U, 26, OR, OR, 3)
        end_sc = sc + rs
        if end_sc > tc:
            f.box(X0 + tc * U, yy, (end_sc - tc) * U, 26, RD, RD, 3)
        f.t(X0 + max(tc, end_sc) * U + 14, yy + 34, note, col, True, 14.5)
    f._pan = None
    yb = f.band(py + PH + 20, "warn", "同一项优化，收益要在当前配置下重测：别处把主核压短以后，小核上的排队才露出来，原地散射和多路收集要一起开", [])
    yb = f.src(yb + 10,
               "实测：256K 最终配置里关掉原地散射与预取，步时 7.465 → 9.69 秒（同作业两次读数逐位相同）；夜里单独测这一项时只差 0.15 秒（9.948 → 9.800 秒，那时主核上的活还多）。",
               "机理是推断：非原地散射在小核上占得久，排在它后面的 MoE 权重梯度规约（同样跑在小核上）晚完成，主核在完成点上等；示意图的块长不按比例。")
    f.save("fig8b-dsa-scq.svg", yb + 14)


def fig_cap():
    C = N.DSA_CAP
    f = Fig(W, "容量是编译时定死的段数上限。一层里每个 query 块的并集切成 1024 行一段，段数随挑选变；程序按容量开暂存和循环上限。"
               "256K 用真实挑选回放，某些层每块最多要 726 段，原来的容量 720 段（1.4 倍）不够，改成 768 段（1.5 倍）；32K 最多 525 段，624 段（1.2 倍）够用。"
               "溢出有两种后果：不开回放时，容量以外的段不算，这些 key 被静默丢掉；开回放时，这一步换大一档容量的程序重跑，不丢。")
    y0 = f.header("容量：编译时定死的段数上限；真实挑选比合成的尾巴更重，256K 要放大",
                  "条 ＝ 每块最多要几段（真实挑选回放）；竖线 ＝ 容量（编译时定死）；超过竖线的红段就是溢出",
                  [(BL, "要的段数"), (RD, "旧容量"), (GR, "新容量")])
    PH = 250
    py = f.panel(0, y0, W, PH, "Ⓐ 要多少段、给了多少段", BL)
    X0, SC, B0 = 300, 2.4, 400          # 横轴从 400 段画起，放大差别
    for i, (lab, need, caps) in enumerate((("256K", C["256K"]["need_max"], [(RD, C["256K"]["cap_old"]), (GR, C["256K"]["cap_new"])]),
                                            ("32K", C["32K"]["need_max"], [(GR, C["32K"]["cap"])]))):
        yy = py + 60 + i * 90
        f.t(X0 - 16, yy + 22, lab, INK, True, 16, "end")
        f.box(X0, yy + 4, (need - B0) * SC, 28, BL, BL, 3)
        if lab == "256K":
            f.box(X0 + (720 - B0) * SC, yy + 4, (need - 720) * SC, 28, RD, RD, 2)
            f.t(X0 + (need - B0) * SC + 8, yy + 24, "溢出 6 段", RD, True, 13.5)
        f.t(X0 + 10, yy + 24, "最多 %d 段" % need, "#fff", True, 14)
        for j, (col, (frac, segs)) in enumerate(caps):
            x = X0 + (segs - B0) * SC
            f.line(x, yy - 6, x, yy + 42, col, 2.4, arrow=False)
            f.t(x + (-6 if (lab == "256K" and j == 0) else 6), yy - 10 + (j * 64 if lab == "256K" else 0), "%d 段（%.1f 倍）" % (segs, frac), col, True, 13.5, "end" if (lab == "256K" and j == 0) else None)
    f.t(X0, py + PH - 30, "横轴从 400 段画起", GY, False, 12.5)
    f._pan = None
    PB = 170
    pyb = f.panel(0, py + PH + 24, W, PB, "Ⓑ 溢出以后两种结果", RD)
    f.t(40, pyb + 56, "✕ 不开回放：多出来的段丢掉（每块至少留一段，输出不是脏内存，但少了 key）", RD, True, 15)
    f.t(40, pyb + 96, "✓ 开回放：记下溢出 → 换大一档的程序重跑这一步；不溢出时零代价", GR, True, 15)
    f._pan = None
    yb = f.band(pyb + PB + 20, "ok", "容量按真实挑选定，留余量；同时开溢出回放、盯溢出计数", [])
    yb = f.src(yb + 10,
               "真实挑选回放：把公开的 Hy4 预览版在真实长文上的 top-k 直接喂给 32 头模型的稀疏核，替换合成选择。32K 真实 5.967 秒对合成 6.062 秒（真实 ρ(32) 2.52、合成 2.85，合成偏保守）；"
               "256K 真实并集略大（ρ 3.86 对 3.71），尾巴更重，才有 726 段那几层。768 段那一档的步时代价待卡上确认。",
               "容量只决定暂存显存和循环上限，网格长度是实际段数；回放在 CPU 上强制溢出测过，单测里上卡也测过一次（小档溢出、换大档重跑，结果与直接跑大档逐位相同）。")
    f.save("fig8b-dsa-cap.svg", yb + 14)


def fig_64k():
    f = Fig(W, "64K 的两种口径。上：整模型。稠密一步 6.01 秒，注意力约 3.11 秒，其余 2.90 秒不随稀疏变；就算注意力变成零，整步也只能快约 2.07 倍，所以整模型口径到不了 2 倍以上多少。"
               "下：单层核心注意力算子组（单层、不含投影、生产层均含蒸馏）。稠密核每层 359 毫秒；稀疏一组修前 278 毫秒（1.29 倍），上午 237 毫秒（1.52 倍）；到 2 倍要 180 毫秒，还差约 57 毫秒，空间在 F 层。")
    y0 = f.header("对标 GPU 的 2 倍，要看单层算子组口径：64K 整模型的天花板只有约 2 倍",
                  "上：64K 整模型一步（秒）；下：单层核心注意力算子组，每层毫秒（只量注意力，含蒸馏）",
                  [(GY2, "注意力（会缩）"), ("#bdc1c6", "其余（不缩）"), (GR, "稀疏算子组")])
    PA = 200
    py = f.panel(0, y0, W, PA, "Ⓐ 整模型 64K：天花板", GY2)
    X0, SC = 300, 150.0
    yy = py + 50
    f.t(X0 - 16, yy + 22, "稠密一步 6.01 秒", INK, True, 15, "end")
    f.box(X0, yy + 4, 3.11 * SC, 30, GY2, GY2, 3)
    f.box(X0 + 3.11 * SC, yy + 4, 2.90 * SC, 30, "#bdc1c6", "#bdc1c6", 3)
    f.t(X0 + 3.11 * SC / 2, yy + 24, "注意力 3.11", "#fff", True, 14, "middle")
    f.t(X0 + 3.11 * SC + 2.90 * SC / 2, yy + 24, "其余 2.90", INK, True, 14, "middle")
    yy2 = yy + 60
    f.t(X0 - 16, yy2 + 22, "注意力全省掉", INK, True, 15, "end")
    f.box(X0, yy2 + 4, 2.90 * SC, 30, "#bdc1c6", "#bdc1c6", 3)
    f.t(X0 + 2.90 * SC + 12, yy2 + 24, "6.01 ÷ 2.90 ≈ 2.07 倍：整模型口径的极限", RD, True, 14.5)
    f._pan = None
    PB = 250
    pyb = f.panel(0, py + PA + 24, W, PB, "Ⓑ 单层核心注意力算子组 64K：每层几毫秒", GR)
    SC2 = 1.9
    rows = [("稠密核（开 sink）", 359.2, GY2, ""), ("稀疏，修前", 278, "#a8dab5", "1.29 倍"), ("稀疏，上午", 237, GR, "1.52 倍"), ("2 倍的目标", 359.2 / 2, "none", "")]
    for i, (lab, v, col, tag) in enumerate(rows):
        yy = pyb + 44 + i * 46
        f.t(X0 - 16, yy + 22, lab, INK, True, 14.5, "end")
        if col == "none":
            f.box(X0, yy + 4, v * SC2, 28, "none", RD, 3, sw=2, dash="5,3")
            f.t(X0 + v * SC2 + 10, yy + 24, "%.0f 毫秒：还差每层约 57 毫秒" % v, RD, True, 14)
        else:
            f.box(X0, yy + 4, v * SC2, 28, col, col, 3)
            f.t(X0 + v * SC2 + 10, yy + 24, "%.0f 毫秒%s" % (v, ("　" + tag) if tag else ""), GR if tag else GY, True, 14)
    f._pan = None
    yb = f.band(pyb + PB + 20, "ok", "整模型倍数里有不随稀疏缩的部分；跟只量注意力的参照比，要用同样只量注意力的口径", [])
    yb = f.src(yb + 10,
               "整模型（64 片、上下文并行 4、每设备 16K token）：稠密 6.014 秒，profile 里注意力以外 2.90 秒；稀疏全层蒸馏 6.229 秒，比稠密的注意力以外多出 3.27 秒：稀疏核 1.10、露出的通信 0.68、蒸馏 0.65、indexer 与账本 0.35、核外粘合 0.35、等待 0.12。",
               "单层算子组（单层、上下文并行 1、不含任何投影，前向 ＋ 反向；稠密基线是开 sink 的生产态稠密核）：64K 修前 278 → 上午 237 毫秒；32K 约 0.95 倍。拆账显示 S 层的主核已满载，剩下的空间在 F 层：indexer 打分与 top-k 之间的粘合、账本、F 层稀疏反向里等小核约 30 毫秒。")
    f.save("fig8b-dsa-64k.svg", yb + 14)


if __name__ == "__main__":
    fig_moe()
    fig_arch()
    fig_perrow()
    fig_ladder()
    fig_result()
    fig_layer()
    fig_qb()
    fig_share()
    fig_step()
    fig_mfu()
    fig_aux()
    fig_len()
    fig_kd()
    fig_teacher()
    fig_scq()
    fig_cap()
    fig_64k()
