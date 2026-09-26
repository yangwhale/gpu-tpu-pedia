# -*- coding: utf-8 -*-
r"""专题五 · 第二节「第一刀：切数据」的两张图。

⭐ 这一节的承重结论（ZeRO 原论文 arXiv 1910.02054 sec. 7 原话核过）：
   · ZeRO-1 / ZeRO-2 的通信量跟数据并行**一模一样**（2Ψ）——&#160;白送
   · ZeRO-3（＝ FSDP）要 **3Ψ，是数据并行的 1.5 倍** ——&#160;最后那 2 个字节是要付钱的
⛔ 大纲原来写「FSDP 白送」，**说宽了**：白送的是前两级，而前两级已经削掉 16 字节里的 14 个。

⛔ 每卡字节数全部现算：每参数 16 字节 ＝ 权重 2（bf16）＋ 梯度 2（bf16）＋ 优化器状态 12
   （fp32 主权重 4 ＋ m 4 ＋ v 4），口径跟专题四那张 16 字节表一致。

⛔ 刻意没画：激活（它随 batch 和序列长度变，专题四讲过）；ZeRO++ 这类通信优化。每卡字节只算模型状态。
"""
from topic03_draw import Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE

from topic05_numbers import PSI, N_DP as N, TIB, GIB, W_B, G_B, O_B   # 16 字节的账只有一份

W = 1400
STAGES = [                       # (名字, 权重份额, 梯度份额, 状态份额, 通信 Ψ 倍数)
    ("数据并行", 1, 1, 1, 2),
    ("ZeRO-1", 1, 1, 1 / N, 2),
    ("ZeRO-2", 1, 1 / N, 1 / N, 2),
    ("ZeRO-3 ＝ FSDP", 1 / N, 1 / N, 1 / N, 3),
]


def per_param(ws, gs, os_):
    return W_B * ws + G_B * gs + O_B * os_


def per_dev_bytes(st):
    return PSI * per_param(*st[1:4])


_b = [per_dev_bytes(s) for s in STAGES]
assert abs(_b[0] / TIB - 9.76) < 0.01, _b[0] / TIB          # 跟专题四 5.1 的 9.76 TiB 对齐
assert abs(_b[1] / TIB - 2.50) < 0.01, _b[1] / TIB
assert abs(_b[2] / TIB - 1.29) < 0.01, _b[2] / TIB
assert abs(_b[3] / GIB - 78.11) < 0.01, _b[3] / GIB         # 9.76 TiB ÷ 128 ≈ 78 GiB（约 84 GB）


def fmt(b):
    return "%.2f TiB" % (b / TIB) if b >= TIB else "%.2f GiB" % (b / GIB)


def fig_mem():
    f = Fig(W, "数据并行和 ZeRO 三级，每张卡的常驻显存。每个参数 16 字节：权重 2、梯度 2、优化器状态 12。"
               "数据并行每张卡都存一整份，16 字节。ZeRO-1 把优化器状态切成 128 份，每卡剩约 4 字节；"
               "ZeRO-2 再切梯度，剩约 2 字节；ZeRO-3 连权重也切，只剩 0.125 字节。"
               "换成 DeepSeek-V3 的 6710 亿参数、128 路：每卡 9.76 TiB、2.50 TiB、1.29 TiB、78.11 GiB。"
               "通信量：前三种都是 2 Ψ，跟数据并行一模一样；ZeRO-3 是 3 Ψ，多一半")
    y0 = f.header("ZeRO：把重复的削掉，一次削一样"
                  "　——　<tspan font-weight=\"700\">前两级白送，最后一级要付 50% 的通信</tspan>",
                  "每个参数 16 字节，按专题四那张表的口径拆成三块。例子：DeepSeek-V3，6,710 亿参数，128 路数据并行（2,048 卡 ÷ 16 段流水线）",
                  [(BL, "权重 2 字节（bf16）"), (OR, "梯度 2 字节（bf16）"), (GR, "优化器状态 12 字节（fp32）")])
    RH, BX, SCALE = 64, 250, 38.0          # 每字节 38 px，16 字节 ＝ 608 px
    PH = 30 + 30 + len(STAGES) * RH + 44
    py = f.panel(0, y0, W, PH, "每张卡要常驻多少字节 / 参数", GR, tag="条长 ∝ 每参数字节数")
    f.t(BX, py + 22, "每参数字节（切成 128 份之后）", GY, True, 13)
    f.t(1040, py + 22, "V3 每卡常驻", GY, True, 13)
    f.t(1220, py + 22, "每步通信", GY, True, 13)
    for i, st in enumerate(STAGES):
        name, ws, gs, os_, comm = st
        yy = py + 34 + i * RH
        f.t(18, yy + 28, name, INK, True, 16)
        x = BX
        for col, bytes_, share in ((BL, W_B, ws), (OR, G_B, gs), (GR, O_B, os_)):
            w = bytes_ * share * SCALE
            if w >= 1:
                f.box(x, yy + 8, w, 32, col, col, 3)
            x += w
        pp = per_param(ws, gs, os_)
        if x - BX < 8:
            # ⭐ 2026-09-25 逐图审：ZeRO-3 的条细到看不清。画一根细线 ＋ 放大后的样子（128 路时放大 20 倍）。
            f.box(BX, yy + 8, max(2, x - BX), 32, INK, INK, 0)
            zx, ZOOM = BX + 130, 20
            f.line(BX + 92, yy + 24, zx - 6, yy + 24, GY2, 1.2, dash="3,3")
            for col, bytes_, share in ((BL, W_B, ws), (OR, G_B, gs), (GR, O_B, os_)):
                w = bytes_ * share * SCALE * ZOOM
                f.box(zx, yy + 8, w, 32, col, col, 3)
                zx += w
            f.t(zx + 10, yy + 30, "← 放大 %d 倍才看得见" % ZOOM, GY, size=13)
        f.t(x + 12, yy + 30, ("%.2f 字节" % pp) if pp >= 0.2 else ("%.3f 字节" % pp), INK, True, 14)
        f.t(1040, yy + 30, fmt(per_dev_bytes(st)), RD if i < 3 else GR, True, 16)
        f.t(1220, yy + 30, "%dΨ%s" % (comm, "（1.5×）" if comm == 3 else "（＝数据并行）" if i else ""),
            RD if comm == 3 else INK, comm == 3, 15)
    f.t(18, py + PH - 46, "被切掉的那几块没有消失：每张卡只剩自己那 1/128。", GY, size=13)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "前两级白送，最后一级要付钱", [
        "ZeRO-1、ZeRO-2 通信量跟数据并行<tspan font-weight=\"700\">一个字节都不多</tspan>（一步只同步一次时），却已经把 16 字节削到约 2 字节。"
        "　可 V3 按这个算每卡还要 1.29 TiB，一张卡照样装不下。",
        "只有 ZeRO-3 能降到每卡 78.11 GiB，代价是通信从 2Ψ 变成 <tspan font-weight=\"700\">3Ψ，多一半</tspan>。"
        "　可一张 H800 才 80 GB，放进去就满了，激活没地方放：光切数据还不够。",
    ])
    yb = f.src(yb + 10,
               "📌 出处：Rajbhandari 等，ZeRO，arXiv 1910.02054 sec. 5、sec. 7。原文：Pos、Pos+g 通信量与数据并行相同（2Ψ），"
               "Pos+g+p 最多 1.5 倍。Ψ 为参数个数，此处按元素数计。",
               "⚠️ 每卡字节数本脚本现算并 assert；9.76 TiB 与专题四 5.1 同一口径（671e9 × 16 ÷ 1024⁴）。128 路 ＝ V3 的 2,048 卡 ÷ 16 段流水线（技术报告 arXiv 2412.19437 sec. 3.2 给出 PP16，除法为本课推导）。"
               "只算常驻，不含激活。")
    f.save("fig5-zero-mem.svg", yb + 14)


def fig_step():
    f = Fig(W, "一层在一步训练里要做哪些通信。数据并行：前向和反向都不通信，反向算完梯度后做一次全归约，"
               "也就是一次归约分散加一次全收集，一共两份。FSDP：前向之前先全收集把这一层的权重拼回来，算完就扔；"
               "反向之前再全收集一次，算完梯度做一次归约分散，一共三份，是数据并行的一点五倍。"
               "多出来的那一份，就是反向时把扔掉的权重再拼一次")
    y0 = f.header("FSDP 为什么多 50%　——　<tspan font-weight=\"700\">反向时要把扔掉的权重再拼一次</tspan>",
                  "同一层、同一步。每个色块是一次通信，量都是这一层权重的 (n−1)/n 份",
                  [(BL, "AllGather（拼回整份）"), (GR, "ReduceScatter（加起来再分）"), (GY2, "计算")])
    PH = 30 + 2 * 92 + 30
    py = f.panel(0, y0, W, PH, "一层 · 一步", BL)
    rows = [
        ("数据并行", [("前向", GY2, 170), ("反向", GY2, 300), ("RS", GR, 110), ("AG", BL, 110)],
         "2 份（RS ＋ AG ＝ 一次 AllReduce）"),
        ("FSDP", [("AG", BL, 110), ("前向", GY2, 170), ("扔掉权重", None, 90), ("AG", BL, 110),
                  ("反向", GY2, 300), ("RS", GR, 110)], "3 份 ＝ 1.5×"),
    ]
    for i, (name, segs, tail) in enumerate(rows):
        yy = py + 34 + i * 92
        f.t(18, yy + 30, name, INK, True, 16)
        x = 160
        for lab, col, w in segs:
            if col is None:
                f.box(x, yy + 8, w - 8, 36, "none", LINE, 4, dash="4,3")
                f.t(x + (w - 8) / 2.0, yy + 31, lab, GY, size=12.5, anchor="middle")
            else:
                f.box(x, yy + 8, w - 8, 36, col, col, 4)
                f.t(x + (w - 8) / 2.0, yy + 31, lab, "#ffffff", True, 14, "middle")
            x += w
        f.t(x + 16, yy + 31, tail, RD if i else INK, True, 15)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "拼一次、扔掉、再拼一次", [
        "FSDP 每层前向前拼一次权重，算完就扔，所以显存里永远只有一层是完整的；"
        "反向要用时，<tspan font-weight=\"700\">只好再拼一次</tspan>　——　多出来的正是这一份。",
        "反过来也成立：前向后不扔（PyTorch 叫 reshard_after_forward=False），通信就回到 2 份，"
        "代价是整个反向期间都攥着完整权重。又是一次拿通信换显存。",
    ])
    f.save("fig5-fsdp-step.svg", yb + 14)


# ════════════════════════════════════════════════════════════════
# 图三：ZeRO 的一步 —— 拆开的 AllReduce，中间夹一次只在本卡做的更新
# ⭐ 2026-09-25 现场「多画图少说话」：原来 §2.2 用两段话讲「前两级为什么白送、就地更新为什么不用通信」，改成这张图。
#   画法沿用 §1.1 那套（topic05_blocks）：四张卡四种颜色，条纹＝加过，虚线框＝不存。
# ⛔ 刻意没画：主权重、m、v 各自的块（它们跟着「第 k 段」走，画出来只会更挤）；梯度裁剪那次标量 AllReduce 写在落点带里。
# ════════════════════════════════════════════════════════════════
import topic05_blocks as B                                            # noqa: E402


def fig_zero_step():
    f = Fig(W, "ZeRO 的一步，四张卡。第一格：反向算完，每张卡手里一整份梯度，但只是自己那批样本的。"
               "第二格：做一次 ReduceScatter，卡 k 只拿到第 k 段梯度的总和。"
               "第三格：每张卡只用这一段梯度，更新自己负责的那一段参数，这一步不通信。"
               "第四格：做一次 AllGather，把各自更新好的那一段拼回来，每张卡又有了完整的新权重。"
               "中间那步不用通信，是因为 Adam 逐个元素算，第 i 个参数只看它自己的梯度、动量和主权重。"
               "ReduceScatter 加 AllGather 正好等于原来那一次 AllReduce，通信一个字节不多")
    y0 = f.header("ZeRO 的一步：中间那次更新只在自己卡上做"
                  "　——　<tspan font-weight=\"700\">拆开的 AllReduce，正好把它夹在中间</tspan>",
                  "四张卡；梯度按参数切成四段，卡 k 负责第 k 段。条纹＝几张卡的加在一起，虚线框＝这张卡不存",
                  [(BL, "卡 0"), (OR, "卡 1"), (GR, "卡 2"), (PU, "卡 3")])
    N4 = B.N
    PH = 30 + 34 + N4 * B.RH + 36
    py = f.panel(0, y0, W, PH, "一步里发生的三件事", GR)
    g_full = [[([k], "g%d" % j) for j in range(N4)] for k in range(N4)]
    g_rs = [[(list(range(N4)), "Σ%d" % j) if j == k else ([], None) for j in range(N4)] for k in range(N4)]
    w_new = [[(list(range(N4)), "W%d" % j) if j == k else ([], None) for j in range(N4)] for k in range(N4)]
    w_all = [[(list(range(N4)), "W%d" % j) for j in range(N4)] for _ in range(N4)]
    states = [("反向算完：各有一整份梯度", g_full, False),
              ("ReduceScatter：只拿第 k 段的总和", g_rs, False),
              ("本地更新第 k 段（不通信）", w_new, True),
              ("AllGather：拼回整份新权重", w_all, False)]
    SX = [66, 400, 734, 1068]
    for i, (lab, st, hot) in enumerate(states):
        f.t(SX[i], py + 26, lab, INK, True, 13.5)
        for k in range(N4):
            yy = py + 40 + k * B.RH
            if i == 0:
                B.rowlab(f, 14, yy, k)
            B.row(f, SX[i], yy, st[k], hot=(k,) if hot else ())
    mid = py + 40 + N4 * B.RH / 2.0 - 4
    for i, lab in enumerate(["①", "②", "③"]):
        x1 = SX[i] + N4 * (B.CW + B.GAP) + 10
        x2 = SX[i + 1] - 14
        f.line(x1, mid, x2, mid, GR, 2)
        f.t((x1 + x2) / 2.0, mid - 10, lab, GR, True, 15, "middle")
    yb = f.band(py + PH + 20, "ok", "中间那步不用通信：Adam 是逐个元素算的", [
        "第 i 个参数的新值只看它自己的梯度、两个动量和主权重。四张卡各更新一段，跟一张卡全部更新一遍，结果一模一样。",
        "① ＋ ③ 正好是原来那一次 AllReduce，<tspan font-weight=\"700\">一个字节不多</tspan>；唯一例外是梯度裁剪：各卡算自己那段的平方和，再 AllReduce 一个数。",
    ])
    yb = f.src(yb + 10,
               "📌 ZeRO 原论文 arXiv 1910.02054 sec. 7：Pos、Pos+g 的通信量与数据并行相同（2Ψ）。Adam 逐元素更新：Kingma 与 Ba，arXiv 1412.6980 算法 1。",
               "⚠️ 一步切成几个 micro-batch 时，只剩 ZeRO-1 严格白送：ZeRO-2 不留整份梯度，没法把几份梯度先攒起来，只好每份算完就 ReduceScatter 一次；"
               "所以配流水线（必须切 micro-batch）时，V3、Megatron 都选 ZeRO-1。")
    f.save("fig5-zero-step.svg", yb + 14)


# ════════════════════════════════════════════════════════════════
# 图四：动量能不能存成 bf16 —— 每一步挪不过半格，就被舍回原处
# ⭐ 开场第三问的答案（§2.2）画成一把刻度尺。舍入是真按 bf16 算的，不是画示意。
# ⛔ 刻意没画：随机舍入（它能救回被舍掉的部分，专题四 §3.2 讲过）；一阶动量（β₁＝0.9，挪得更远，更不成问题）。
# ════════════════════════════════════════════════════════════════
import struct                                                         # noqa: E402
from topic05_numbers import V3_BETA2, TORCH_BETA2                     # noqa: E402


def fig_zero_busy():
    """⭐ 2026-09-25 夜 · 蒸馏 R3：ZeRO 三级的顺序，按「闲忙」讲比按「大小」讲更像故事。
    越闲的越先削（削了几乎不用付钱），越忙的越后削（削了就得每次用之前把别人那份复制过来）。
    时间轴是示意：前向 1 份、反向 2 份、更新一小段；只画「这块东西在哪段时间被读写」。"""
    f = Fig(W, "一步训练里，三块东西各在什么时候被用到。权重从前向到反向每一层都要用，最忙；梯度到反向才一层层算出来、攒到更新；"
               "优化器状态最大，占 12 字节，却只在最后更新那一下才用，整整一步都在显存里干放着。所以 ZeRO 先削最闲的优化器状态，"
               "再削梯度，最后才削最忙的权重，削了权重就得每层用之前把别人那几段复制过来，要多付一半通信")
    y0 = f.header("越闲的越先削　——　<tspan font-weight=\"700\">ZeRO 三级的顺序，就是闲忙的顺序</tspan>",
                  "一步训练里，每块东西在什么时候被读写（时间轴示意：前向 1 份、反向 2 份、最后一小段更新）",
                  [(BL, "权重 2 字节"), (OR, "梯度 2 字节"), (GR, "优化器状态 12 字节")])
    PH = 300
    py = f.panel(0, y0, W, PH, "一步：前向 → 反向 → 更新", INK)
    TX, TW = 220, 640
    SEG = [("前向", 0, 1), ("反向", 1, 3), ("更新", 3, 3.35)]
    U = TW / 3.35
    for lab, a, b in SEG:
        f.t(TX + (a + b) / 2 * U, py + 34, lab, GY, True, 14, "middle")
        f.line(TX + a * U, py + 44, TX + a * U, py + 250, LINE, 1, arrow=False)
    f.line(TX + TW, py + 44, TX + TW, py + 250, LINE, 1, arrow=False)
    rows = [("权重", BL, [(0, 3.35, True)], "ZeRO-3 最后削：最忙，削了每层都得先复制回来", "多付一半通信"),
            ("梯度", OR, [(1, 3.35, True)], "ZeRO-2 再削：反向时才一层层出现", "白送"),
            ("优化器状态", GR, [(0, 3, False), (3, 3.35, True)], "ZeRO-1 先削：最大又最闲，只在更新那一下用", "白送")]
    for r, (lab, col, spans, why, cost) in enumerate(rows):
        yy = py + 62 + r * 64
        f.t(24, yy + 24, lab, col, True, 15)
        for a, b, busy in spans:
            if busy:
                f.box(TX + a * U, yy, (b - a) * U, 36, col, col, 3)
            else:
                f.box(TX + a * U, yy, (b - a) * U - 2, 36, "none", col, 3, sw=1.6, dash="6,4")
                f.t(TX + (a + b) / 2 * U, yy + 23, "整整一步都在显存里干放着", col, size=13, anchor="middle")
        f.t(TX + TW + 24, yy + 16, why, col, True, 14)
        f.t(TX + TW + 24, yy + 36, cost, RD if cost != "白送" else GY, True, 13)
    f.t(24, py + PH - 40, "实心 ＝ 这段时间在读写它；虚线 ＝ 占着显存，但没人碰", GY, size=13)
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "最大的恰好也最闲，所以先削它", [
        "优化器状态占 16 字节里的 12 个，一步里只在最后更新那一下用：每张卡只管自己那 1/n，谁也不耽误。",
        "越往上越忙：权重每一层都要用，削了它，每层算之前就得先把别人那几段复制过来，这就是 FSDP 多付的那一半通信。",
    ])
    f.save("fig5-zero-busy.svg", yb + 14)


def bf16(x):
    """把一个数按 bf16（round-to-nearest-even）存一次，再读回来。"""
    u = struct.unpack(">I", struct.pack(">f", x))[0]
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return struct.unpack(">f", struct.pack(">I", u))[0]


V0, G2 = 1.0, 2.0                        # 当前 v ＝ 1，新来的 g² 是它的 2 倍
STEP = 2 ** -7                           # [1, 2) 里相邻两个 bf16 的间隔
V_A = V3_BETA2 * V0 + (1 - V3_BETA2) * G2        # 1.05
V_B = TORCH_BETA2 * V0 + (1 - TORCH_BETA2) * G2  # 1.001
S_A, S_B = bf16(V_A), bf16(V_B)
assert abs(V_A - 1.05) < 1e-12 and abs(V_B - 1.001) < 1e-12
assert S_A > V0 and abs(S_A - 1.046875) < 1e-9, S_A     # 留下来了（6 格）
assert S_B == V0, S_B                                    # 被舍回 1.0
assert bf16(1 + STEP) == 1 + STEP and bf16(1 + STEP / 2 - 1e-6) == 1.0


def fig_bf16_beta():
    f = Fig(W, "二阶动量每一步按 v 等于 β₂ 乘旧 v 加上 1 减 β₂ 乘新梯度平方来更新。"
               "图里是一把刻度尺，刻度是 bf16 在 1 附近能表示的数，相邻两个相差 128 分之一。"
               "假设当前 v 等于 1，新来的梯度平方是它的 2 倍。"
               "β₂ 等于 0.95 时，新值占 5%，v 挪到 1.05，存成 bf16 是 1.047，这一步留下来了。"
               "β₂ 等于 0.999 时，新值只占 0.1%，v 挪到 1.001，还不到半格，存成 bf16 又回到 1，这一步被舍掉了。"
               "V3 用 β₂ 等于 0.95，所以它敢把动量存成 bf16；主权重每步只加一点点、要一直累加，所以必须留 fp32")
    y0 = f.header("动量能不能存成 bf16：看一步挪多远"
                  "　——　<tspan font-weight=\"700\">挪不过半格，就被舍回原处</tspan>",
                  "刻度＝bf16 在 1 附近能表示的数（相邻差 1/128）。v ← β₂·v ＋ (1−β₂)·g²；假设当前 v＝1，新来的 g² 是它的 2 倍",
                  [(GR, "β₂＝0.95（V3）"), (RD, "β₂＝0.999（PyTorch 默认）")])
    PH = 272
    py = f.panel(0, y0, W, PH, "一把 bf16 的刻度尺", GR)
    X0, SC = 120, 1100 / (8 * STEP)             # 画 1.0 到 1+8 格
    ay = py + 150
    f.line(X0 - 20, ay, X0 + 8 * STEP * SC + 20, ay, GY2, 1.5, arrow=False)
    for k in range(9):
        x = X0 + k * STEP * SC
        f.line(x, ay - 12, x, ay + 12, INK if k == 0 else GY, 2 if k == 0 else 1.2, arrow=False)
        f.t(x, ay + 32, "%.4f" % (1 + k * STEP), GY, size=12, anchor="middle")
    xa, xb = X0 + (V_A - 1) * SC, X0 + (V_B - 1) * SC
    xsa = X0 + (S_A - 1) * SC
    # β₂ = 0.95：挪到 1.05，落在第 6 格附近 → 存成 1.047
    f.path("M%.1f,%.1f Q%.1f,%.1f %.1f,%.1f" % (X0, ay - 16, (X0 + xa) / 2, ay - 110, xa, ay - 18), GR, 2.4)
    f.box(xsa - 6, ay - 6, 12, 12, GR, GR, 6)
    f.t((X0 + xa) / 2, ay - 96, "新值占 5%%：挪到 %.3f，存成 %.4f —— 留下来了" % (V_A, S_A), GR, True, 14, "middle")
    # β₂ = 0.999：挪到 1.001，不到半格 → 舍回 1.000
    f.path("M%.1f,%.1f L%.1f,%.1f" % (X0, ay + 50, xb + 14, ay + 50), RD, 2.4)
    f.box(X0 - 6, ay - 6, 12, 12, "none", RD, 6, sw=2.4)            # 舍回原处：存下来的还是 1.000
    f.t(X0 + 30, ay + 80, "新值占 0.1%%：挪到 %.3f，不到半格（%.4f），存成 %.3f —— 被舍掉了" % (V_B, STEP / 2, S_B), RD, True, 14)
    yb = f.band(py + PH + 20, "ok", "所以 V3 敢把两个动量存成 bf16，主权重却不行", [
        "动量是滑动平均，β₂ 不太接近 1 时每步挪得够远，bf16 存得住；V3 用的是 0.95。",
        "主权重每一步只加一点点、而且要一直累加下去：小于半格的更新全被舍掉，所以留在 fp32。",
    ])
    yb = f.src(yb + 10,
               "📌 DeepSeek-V3 技术报告 arXiv 2412.19437 sec. 3.3.3：用 BF16 代替 FP32 追踪 AdamW 一、二阶矩，「未观察到性能退化」；主权重与用于 batch 累积的梯度仍保留 FP32。sec. 4.2：β₁＝0.9、β₂＝0.95。",
               "📌 这不是默认做法：常规做法里优化器状态与参数同精度；Megatron Core 要显式开启 --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16。PyTorch AdamW 默认 β₂＝0.999。",
               "⚠️ 本课推导：刻度与舍入按 bf16（8 位有效位、就近舍入）真算。V3 报告的验证是整套 FP8 方案（含 bf16 动量）对 BF16 基线，16B 与 230B 两个规模损失相对误差低于 0.25%；未单独消融 bf16 动量。")
    f.save("fig5-bf16-beta.svg", yb + 14)


fig_mem()
fig_step()
fig_zero_step()
fig_bf16_beta()
fig_zero_busy()


# ════════════════════════════════════════════════════════════════
# 图六：四库藏书楼 —— 同一套书，三种分法
# ⭐ 2026-09-26 现场定稿的比方（三轮打磨）：
#   · 全部权重 ＝ 一整套四库全书；61 间书房 ＝ 61 层（前 3 间 Dense，后 58 间 MoE，大多数长得一样）。
#     ⛔ 第一版按「卷」分层 —— 被指出「又把内容变小了」：一层是一整间书房，不是某本书的一卷。
#   · 阁的个数用 V3 的真实数：数据并行 128、EP 64、PP 16。⛔ 不用乾隆的「七阁」，那个数跟模型对不上。
#   · ⭐⭐ FSDP 是**无脑切碎**：一间书房的书一页页排成一长条，按长度等分，切口落在书中间也不管。
#     单独一段谁也读不了，拼回整间才能读 ——&#160;这正是 AllGather 之前那份分片「没用」的意思。
#     ⛔ 不能画成「每阁分到几本完整的书」：那是 EP 的分法（一架 ＝ 一个完整专家，拿着就能用）。
# ⛔ 刻意没画：TP（它切的是一本书内部，第二刀自己有图）；数据并行 128 个阁里的每一个。
# ════════════════════════════════════════════════════════════════
import random                                                         # noqa: E402

N_LAYER, N_DENSE, N_EXP, N_PP, N_EP = 61, 3, 256, 16, 64           # V3 config.json ＋ 技术报告 sec. 3.2
assert N_EXP % N_EP == 0 and N_EXP // N_EP == 4
BOOK_TINT = ["#fef7e0", "#e8f0fe", "#e6f4ea", "#f3e8fd", "#fce8e6"]


def fig_siku():
    f = Fig(W, "把 DeepSeek-V3 的全部权重想成一整套四库全书，存进 61 间书房，一间就是一层："
               "前 3 间是 Dense，后 58 间是 MoE，每间有一张注意力目录台、1 个共享书架和 256 个专家书架。"
               "按 V3 的配置一共 128 个阁。数据并行：每阁一整套。"
               "FSDP：把每间书房的书一页页排成一长条，按长度等分成 128 段，切口落在书中间也不管，每阁拿一段；"
               "单独一段读不了，要用这间时把 128 段各抄一份拼回整间，读完扔掉抄本，只留自己那段。"
               "EP：256 个专家书架整架分给 64 个阁，每阁 4 整架，拿着就能用；PP：61 间书房按顺序分成 16 段")
    y0 = f.header("一座藏书楼，三种分法　——　<tspan font-weight=\"700\">FSDP 切碎了、拼回来才能读；EP 整架分、拿着就能用</tspan>",
                  "把 V3 的全部权重想成一整套四库全书：一间书房 ＝ 一层，一个阁 ＝ 一张卡。阁数按 V3 的真实配置",
                  [(GY2, "Dense 书房"), (BL, "注意力目录台"), (GR, "共享书架"), (OR, "专家书架")])

    # ── ① 藏书楼 ＝ 模型 ─────────────────────────────────────────
    PH1 = 222
    py = f.panel(0, y0, W, PH1, "① 藏书楼 ＝ 模型：61 间书房 ＝ 61 层", INK)
    RX, RW, RG = 24, 11, 3
    for i in range(N_LAYER):
        x = RX + i * (RW + RG)
        col = GY2 if i < N_DENSE else OR
        f.box(x, py + 40, RW, 56, "#f1f3f4" if i < N_DENSE else "#fef7e0", col, 2, sw=1.2)
    xe = RX + N_LAYER * (RW + RG) - RG
    f.t(RX, py + 118, "前 3 间：Dense", GY, True, 13)
    f.t(RX + N_DENSE * (RW + RG) + 60, py + 118, "后 58 间：MoE，大多数长得一模一样", OR, True, 13)
    f.t(RX, py + 142, "第 1 间在最左，数据从左往右一间一间过", GY, size=13)
    f.t(RX, py + 164, "整套约 6,710 亿个参数；这是全部的书，不是其中一本", GY, size=13)
    # 右边：放大一间 MoE 书房
    ZX = xe + 70
    f.line(xe + 8, py + 68, ZX - 10, py + 68, GY2, 1.2, dash="3,3")
    f.box(ZX, py + 40, W - ZX - 20, 140, "none", OR, 6, sw=1.4)
    f.t(ZX + 12, py + 60, "放大一间 MoE 书房", OR, True, 13.5)
    f.box(ZX + 12, py + 74, 70, 40, BL, BL, 3)
    f.t(ZX + 47, py + 99, "目录台", "#ffffff", True, 13, "middle")
    f.box(ZX + 12, py + 122, 70, 40, GR, GR, 3)
    f.t(ZX + 47, py + 147, "共享架", "#ffffff", True, 13, "middle")
    GX, CELL = ZX + 100, 5
    for k in range(N_EXP):
        r, c = divmod(k, 32)
        f.box(GX + c * (CELL + 2), py + 76 + r * (CELL + 5), CELL, CELL + 3, OR, OR, 1)
    f.t(GX, py + 172, "256 个专家书架，每架是一个完整的专家", OR, size=13)

    # ── ② 数据并行 ───────────────────────────────────────────────
    y2 = py + PH1 - 30 + 20
    PH2 = 118
    py = f.panel(0, y2, W, PH2, "② 数据并行：128 个阁，每阁一整套", GY)
    for j, lab in enumerate(["阁 1", "阁 2", "阁 3", "阁 4", "…", "阁 128"]):
        x = 24 + j * 150
        if lab == "…":
            f.t(x + 50, py + 50, "…", GY, True, 20, "middle")
            continue
        f.box(x, py + 18, 120, 48, "none", GY2, 4)
        for i in range(0, N_LAYER, 2):
            f.box(x + 6 + i * 1.75, py + 26, 2.6, 22, GY2 if i < N_DENSE else OR, "none", 0)
        f.t(x + 60, py + 62, lab + "：整套", INK, True, 12.5, "middle")
    f.t(930, py + 38, "谁也不缺，可每个阁都被同一套书占满", RD, True, 14)
    f.t(930, py + 62, "128 个阁存了 128 份一模一样的书", GY, size=13)

    # ── ③ FSDP ───────────────────────────────────────────────────
    y3 = py + PH2 - 30 + 20
    PH3 = 272
    py = f.panel(0, y3, W, PH3, "③ FSDP：每间书房拍平、按长度等分，每阁只拿一段", BL,
                 tag="示意画 8 段，实际 128 段")
    SX, SW, SY = 150, 1100, py + 46
    f.t(24, SY + 22, "第 k 间的书，", INK, True, 13)
    f.t(24, SY + 40, "一页页排成一条", GY, size=12.5)
    rnd = random.Random(7)
    x, b = SX, 0
    while x < SX + SW - 1:
        # ⭐ 每本书都比一段长：没有哪一段里装着一整本 —— 每阁手里都只是碎片
        w = min(rnd.choice([150, 172, 196, 214]), SX + SW - x)
        f.box(x, SY, w, 44, BOOK_TINT[b % len(BOOK_TINT)], GY2, 2, sw=1)
        if w > 50:
            f.t(x + w / 2.0, SY + 27, "书 %d" % (b + 1), GY, size=12, anchor="middle")
        x, b = x + w, b + 1
    SEG = 8
    segw = SW / SEG
    for k in range(SEG + 1):
        cx = SX + k * segw
        f.line(cx, SY - 12, cx, SY + 58, RD, 2, dash="5,3", arrow=False)
    names = ["阁 1", "阁 2", "阁 3", "阁 4", "…", "阁 126", "阁 127", "阁 128"]
    for k, nm in enumerate(names):
        f.t(SX + (k + 0.5) * segw, SY + 76, nm, RD if k == 2 else INK, k == 2, 13, "middle")
    f.t(SX + SW + 14, SY + 20, "红线等距切，", RD, True, 13)
    f.t(SX + SW + 14, SY + 38, "不看书的边界", RD, True, 13)
    # 阁 3 那一段放大：半本 ＋ 另一本的开头
    hx = SX + 2 * segw
    f.box(hx, SY - 4, segw, 52, "none", RD, 3, sw=2.4)
    f.t(hx + segw / 2.0, SY + 100, "阁 3 手里：一本书的后半截 ＋ 另一本的开头", INK, size=13, anchor="middle")
    f.t(hx + segw / 2.0, SY + 120, "单拿这一段，读不了", RD, True, 14, "middle")
    # 用之前：抄齐、读、扔
    FY = SY + 148
    steps = [("要用第 k 间", GY2, 150), ("128 段各抄一份，拼回整间（AllGather）", BL, 330),
             ("读：这一层算完", GY2, 170), ("扔掉抄本，只留自己那段", GY2, 230)]
    x = 150
    for i, (lab, col, w) in enumerate(steps):
        filled = col == BL
        f.box(x, FY, w, 34, col if filled else "none", col, 4, sw=1.4)
        f.t(x + w / 2.0, FY + 22, lab, "#ffffff" if filled else INK, True, 13, "middle")
        if i < len(steps) - 1:
            f.line(x + w + 4, FY + 17, x + w + 30, FY + 17, GY2, 1.6)
        x += w + 34
    f.t(x + 6, FY + 22, "反向时再抄一次", BL, True, 13)

    # ── ④ EP 与 PP ───────────────────────────────────────────────
    y4 = py + PH3 - 30 + 20
    PH4 = 200
    py = f.panel(0, y4, W, PH4, "④ 对比：EP、PP 按整件分，拿到手就能用", OR)
    f.t(24, py + 40, "EP", OR, True, 15)
    f.t(24, py + 58, "256 架分 64 阁", GY, size=12.5)
    labs = [("阁 1", "1–4"), ("阁 2", "5–8"), ("阁 3", "9–12"), ("…", ""), ("阁 64", "253–256")]
    for j, (nm, rng) in enumerate(labs):
        x = 150 + j * 170
        if nm == "…":
            f.t(x + 60, py + 46, "…", GY, True, 20, "middle")
            continue
        f.box(x, py + 16, 140, 50, "none", OR, 4)
        for q in range(4):
            f.box(x + 10 + q * 31, py + 24, 24, 26, OR, OR, 2)
        f.t(x + 70, py + 62, "%s：书架 %s" % (nm, rng), INK, True, 12, "middle")
    f.t(1000, py + 38, "每阁 4 整架，一架就是一个完整专家", OR, True, 14)
    f.t(1000, py + 60, "拿着就能用；要查，就跑去那一阁", GY, size=13)
    f.t(24, py + 118, "PP", PU, True, 15)
    f.t(24, py + 136, "61 间分 16 段", GY, size=12.5)
    x = 150
    for g in range(N_PP):
        n = 4 if g < N_LAYER - 3 * N_PP else 3            # 示意：13 段 4 间、3 段 3 间，凑满 61
        gw = 6 + n * 7 + (n - 1) * 2
        f.box(x, py + 96, gw, 40, "none", PU, 3, sw=1.4)
        for i in range(n):
            f.box(x + 3 + i * 9, py + 102, 7, 28, "#f3e8fd", PU, 1)
        x += gw + 8
    f.t(1000, py + 116, "每段按顺序拿几间完整的书房", PU, True, 14)
    f.t(1000, py + 138, "一间就是一整层，拿着就能算", GY, size=13)
    f.t(150, py + 166, "每段约 4 间（示意；V3 实际每段几层，报告没写）", GY, size=12.5)

    f._pan = None
    yb = f.band(py + PH4 + 20, "ok", "区别在拿到手的那一份能不能单独用", [
        "FSDP 不管书的边界，只按长度等分，为的是通信好切块；所以单独一段没用，每次用之前都得 AllGather 拼回整间。",
        "EP、PP 按整件分：一架是一个完整专家，一间是一整层，拿着就能算。它们付的是别的钱：token 得跑去找专家，层与层得排队递活。",
    ])
    yb = f.src(yb + 10,
               "📌 V3 配置：技术报告 arXiv 2412.19437 sec. 3.1–3.2（2,048 块 H800；PP 16、EP 64 跨 8 节点、ZeRO-1 数据并行）；"
               "config.json（61 层，前 3 层 dense，256 个路由专家 ＋ 1 个共享专家）。",
               "⚠️ 本课推导：数据并行 128 ＝ 2,048 ÷ 16。V3 实际用 ZeRO-1 不是 FSDP，这里借它的路数讲 FSDP 怎么切。"
               "PyTorch FSDP 第一版把一层的参数拍平成一条再等分；FSDP2 按每个参数的第 0 维切，单独一份同样不能直接拿来算。")
    f.save("fig5-siku.svg", yb + 14)


fig_siku()
