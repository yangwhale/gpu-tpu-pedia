# -*- coding: utf-8 -*-
r"""专题九 · 第十一节「稀疏注意力核：把挑中的 key 当成 MoE 来算」的图（2026-11-24 补）。

fig8b-dsa-moe     回答：每个 query 只挑 2048 个 key，现成的块稀疏核为什么一点不省？怎么改成能省的？
                  —— Ⓐ 玩具账：4 个 query × 16 个 key，每人挑 4 个，4 块里 4 块都要算；Ⓑ 把这一桌要的 key 收拢成一摞（并集 8 行），
                  整桌一起乘：这一桌就是 MoE 的一个专家，收拢来的 key 行就是分给它的 token。
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
               "整桌对着这 8 行一起乘，没挑的格子用成员位掩掉。这一桌就像 MoE 的一个专家，收拢来的 key 行就像分给它的 token。")
    y0 = f.header("每个 query 只挑 2048 个，块稀疏却一块都跳不掉；先收拢再乘，就是一次 MoE",
                  "玩具账：一桌 4 个 query（位置 12–15），每人从 16 个 key 里挑 4 个（真实是从 32768 里挑 2048）；蓝格 ＝ 挑中，灰格 ＝ 因果看不见",
                  [(BL, "挑中"), (RD, "这一块要整块算"), (GR, "收拢后的一摞")])
    PH = 420
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
        "真实情况：一块 512 个 query × 512 个 key，每个 query 挑 2048 ÷ 32768 ≈ 6%，",
        "块里一个都没挑中几乎不可能。模拟 12 种组合（3 种 k × 2 种块 × 2 种挑法），",
        "块激活率全是 100%；把 k 从 2048 压到 128，现成的核耗时一点没变。"], size=14.5, lh=24, fill=GY)

    RX = LW + 30
    RWd = W - RX
    py2 = f.panel(RX, y0, RWd, PH, "Ⓑ 先收拢再乘：这一桌就是一个专家", GR)
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
    yb = f.band(py + PH + 20, "ok", "训练里省的是乘法：先把一桌要的 key 收拢成一摞，再整块乘", [])
    yb = f.src(yb + 10,
               "玩具挑选是编的（4 个 query、16 个 key、每人挑 4 个）；真实挑选：32K 长文，每个 query 挑 2048 个，「最近 512 个几乎全挑、其余散在整个历史上」。",
               "块激活率：32K、k ＝ 2048／512／128、块 512 × 512 和 1024 × 2048、均匀挑与偏向最近，12 种组合全部 100%（模拟）；k 从 2048 压到 128 时现成的动态块稀疏核 987.8 → 987.6 ms（另一代芯片上的实测）。",
               "ρ（并集膨胀比）＝ 一桌的并集行数 ÷ k；真实值是公开的 Hy4 预览版权重在 CPU 上抓的 32K 长文 top-k 统计（4 个 F 层、两份数据的均值）：ρ(4) %.2f–%.2f、ρ(8) %.2f–%.2f、ρ(16) %.2f–%.2f。" % (N.DSA_RHO[4] + N.DSA_RHO[8] + N.DSA_RHO[16]),
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
        c = GR if k == 16 else GY2
        f.box(X0, y2 + 4, v * SCk, 24, c, c, 3)
        f.t(X0 + v * SCk + 10, y2 + 21, "%.1f ms" % v, c, True, 15)
    f._pan = None
    yb = f.band(pyb + PB + 20, "ok", "固定开销按行算的活，交给专门搬行的小核；主核只看到连续的大块", [])
    yb = f.src(yb + 10,
               "本课推导（Ⓐ）：MoE 一行 token 进本讲模型的专家（隐藏 6144、专家中间维 2048）：gate ＋ up 6144 × 4096 ＋ down 2048 × 6144 ≈ 3775 万次乘加；"
               "DSA 一行 key 跟 M × (576 ＋ 512) 次，M ＝ 一桌 query 数 × 头数 ＝ 1024 时约 111 万；按每设备峰值 1155 TFLOP/s 折成纳秒，是理想值。",
               "实测（单设备微基准）：主核每行一条 DMA 13–16 ns／行，1 KB 到 12 KB 行几乎不变；潜变量整份放进片上暂存、主核按行取，离线编译产物里每行约 11 条标量指令；"
               "小核间接收集两颗每秒 4.2 亿行（含写回），带加法的间接写每秒 3.9 亿行。",
               "实测（Ⓑ）：32 头、合成挑法、rank 0；不切段时小核收 6.0 ms、主核算 6.6 ms，串在一个程序里约 14 ms；切段以后编译器把小核的收集和主核的核交错排。")
    f.save("fig8b-dsa-perrow.svg", yb + 14)


def fig_arch():
    f = Fig(W, "稀疏核的软件架构，五步。第一步 indexer 分块选 top-k（设计中，还没实现，现在用现成的整行 top-k）。第二步账本：主核上一颗 Pallas 核做合并、标记、压紧，"
               "再用 XLA 出段表，每组层算一次、前向反向共用。第三步小核按段把并集 key 行收进暂存。第四步主核的流式核每段连续读一块，做矩阵乘和在线 softmax；"
               "反向算 dQ 和每行的部分梯度。第五步小核把部分梯度按 key 加回 dKV。下半是时间线：前向小核收第 i ＋ 1 段时主核算第 i 段；"
               "反向倒序算，部分梯度分 8 组交给小核加，第一组跟后面几段的计算重叠。")
    y0 = f.header("稀疏核的软件架构：五步，三种执行单元，两两重叠",
                  "框的颜色 ＝ 谁来跑：蓝 主核（TensorCore）上的 Pallas 核；紫 小核（SparseCore）上的 Pallas 核；灰 编译器（XLA）自己生成的计算；虚框 ＝ 设计中、还没实现",
                  [(BL, "主核 Pallas"), (PU, "小核 Pallas"), (GY2, "XLA")])
    PA = 330
    py = f.panel(0, y0, W, PA, "Ⓐ 一层稀疏注意力的数据流（定稿配置）", BL)
    steps = [
        ("① 分块选 top-k", "indexer 按 query 块打分", "每个 query 2048 个下标", BL, True),
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
            unit = "主核 · 设计中，未实现"
        if i == 1:
            unit = "主核 Pallas ＋ XLA 段表"
        f.t(x + 14, BY + BH - 16, unit, col, True, 13, w=BW - 20)
        if i:
            f.line(x - GAP + 4, BY + BH / 2.0, x - 6, BY + BH / 2.0, GY2, 1.8)
    yy = BY + BH + 30
    f.t(BX, yy, "账本只跟下标有关：几层共用一份挑选时共用一份账本（按 4 层一组），前向反向共用；32K 每组 3.1–3.5 ms", GY, size=13.5)
    f.t(BX, yy + 26, "XLA 另做几样零碎：delta ＝ Σ dO · O、q 和 dQ 换布局；前向收来的暂存留给反向，反向不再收（每层省 3.6–4.2 ms，多占 2–4 GB）", GY, size=13.5)
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
    f.t(xb, yf - 10, "反向（倒序；暂存留着，不再收）", INK, True, 15)
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
               "①：现在用的是现成的整行 top-k；按 query 块分块出下标、并把账本的并集那一步接到 indexer 每块最后一步，是设计里的下一步，还没上卡。",
               "②：账本是整数计算，放在求导之外（照 MoE 的做法：整数映射只依赖专家号，只有收集本身求导）；合并、标记同一个 key 的成员位、把并集压到前面，三步在一颗 Pallas 核里做完，段表和逐段数组是 XLA。",
               "时间线是示意：段的个数照定稿配置（8 段、8 组），块长不按比例；0.85 ms 来自合成挑法、切 8 段那一格的 profile，11.1 ms 的拆账来自真实挑选、一桌 1024 行那一版的 profile（32 头）。")
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
            ("第 13 轮　一桌的 query 加倍、段长 1024", real["qb"]),
            ("第 14 轮　层里的整数活挪进账本", real["r14"])]),
        ("③ 前向 ＋ 反向 · 生产设置 · 设备时间", prod["dense"], 2, [
            ("第 20 轮　换口径复测", prod["r20"]),
            ("第 21 轮　一桌再加倍（M ＝ 1024）", prod["r21"]),
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
                  [(C32, "32 头"), (C64, "64 头"), (GR, "比稠密快")])
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
                c = GR if v < d else col
                f.box(X0, y2, v * SC, 20, c, c, 3)
                tag = "%.*f" % (nd, v)
                f.box(X0 + v * SC + 4, y2 + 1, 9 * len(tag) + 8, 19, "#fff", "#fff", 2)
                f.t(X0 + v * SC + 8, y2 + 16, tag, c, True, 13.5)
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
        c = GR if v < 6 else GY2
        f.box(X0, y2 + 3, v * SCm, 22, c, c, 3)
        f.t(X0 + v * SCm + 8, y2 + 19, ("%.0f" % v) if v > 20 else ("%.1f" % v), c, True, 13.5)
    f._pan = None
    yb = f.band(pym + PM + 20, "ok", "每一轮先量清楚卡在哪：先是搬一行太贵，再是数据，再是核外的零碎和账本", [])
    yb = f.src(yb + 10,
               "实测（单设备、同一进程里稀疏和稠密各测；稠密是本讲模型调优用的块稀疏配置，按块掩码）。① 合成挑法「一半挑最近、一半在更早的里随机」，32 头一桌 8 个 query（ρ 2.28）、64 头一桌 4 个（ρ 1.45），rank 0，只看前向。"
               "第 3 轮 12.5 对稠密 4.87，还慢 2.6 倍；第 7 轮不切段 14.0 / 17.4。合成挑法上前向 ＋ 反向：第 5 轮 60.6 / 79.4，第 9–11 轮小工散射累加、规约分组倒序 27.1 → 21.8 / 34.1 → 30.6（稠密 15.0 / 29.7）。",
               "② 真实挑选：公开的 Hy4 预览版权重在 CPU 上抓的一篇 32K 长文，按上下文并行 8 的负载均衡切法取每个设备的本地 query；8 个设备取最慢的那个；默认运行库设置、每次调用后同步（含主机派发）。",
               "③ 换成生产配置用的运行库设置，连续派发 10 次取平均（设备时间口径，训练里整步异步派发就是这样）；稠密在这个设置下也快了一截，所以第 20 轮看着比第 14 轮快、其实是换了尺子。M ＝ 一桌 query 数 × 头数：32 头一桌 32 个、64 头一桌 16 个。",
               "账本：第 1 轮 71 ms（两次 840 万元素的 scatter ＋ 逐段切片）→ 排序替掉 scatter 7.8–9.9 → 全局一张表 8.9 → 两次排序写成 Pallas 双调网络 5.6 → 压紧写成 Pallas 移位网络 3.7–4.1 → 合并、标记、压紧一个核 3.1–3.6（一桌加倍那一轮一度回到 5 ms 多）。")
    f.save("fig8b-dsa-ladder.svg", yb + 14)


def fig_result():
    f = Fig(W, "定稿：一层注意力前向加反向，最慢的设备。32K：32 头稠密 12.98 毫秒、稀疏 10.04，快 1.29 倍，摊进每层的账本后 10.9，快 1.19 倍；"
               "64 头稠密 25.73、稀疏 16.02，快 1.61 倍，含账本 16.9，快 1.52 倍。256K：32 头稠密 186.0、稀疏 31.3，快 5.9 倍，含账本 5.6 倍；"
               "64 头稠密 377.6、稀疏 46.3，快 8.2 倍，含账本 7.8 倍。全部零溢出、重跑逐位。")
    y0 = f.header("定稿：32K 快 1.2–1.6 倍，256K 快 5.9–8.2 倍（单层、单测）",
                  "灰 ＝ 稠密（块稀疏核，按块掩码）；绿 ＝ 我们的稀疏核；浅绿 ＝ 摊进每层的账本（每组 ÷ 4，推断）；条越短越快；最慢的设备",
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
               "实测（定稿代码：一桌 32 头 32 个 query、64 头 16 个，即 M ＝ 1024；段长 1024；生产配置的运行库设置；连续派发 10 次取平均）：32K 8 个设备都测，取最慢的那个比最慢的那个；账本每组 3.1–3.5 ms。",
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


if __name__ == "__main__":
    fig_moe()
    fig_arch()
    fig_perrow()
    fig_ladder()
    fig_result()
    fig_layer()
