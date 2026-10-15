# -*- coding: utf-8 -*-
r"""专题六 · 第十节「先选拓扑，再调参数」的五张静态图。

回答的问题
  · fig6-s10-lens   「调参 +45%，换拓扑 +347%」到底比的是什么？换成每卡口径，结论变成什么样？
  · fig6-s10-wire   拓扑约束有多硬？同一个模型，同一份 TP，走 ICI 和走 DCN 差多少？
  · fig6-s10-rank   TPU 上 device 顺序为什么就是拓扑？顺序错了会怎样？（只画机制）
  · fig6-s10-slice  DeepSeek-V3 在 TPU v7 上最少要几颗芯片才起得来一个引擎？跟上下文和并发什么关系？
  · fig6-s10-conc   容量估算为什么不能拿 conc=8 的数？并发扫一遍长什么样？

数从哪来
  · V4-Pro / vLLM / GB300 拓扑扫描（总吞吐 ＝ prompt ＋ 输出 token，4k1k）：
    gpu/inference/a4x-max/deepseek-v4/VLLM-V4PRO-RUNBOOK.md:446、448、449、465、468；
    「+45% / +347%」原话 :489；每卡 2.47× 原话 :476；256→512 涨 28% 原话 :487。
    GPU 数：1p1d = 4 + 4；3p1d = 3×4 + 4；3p + dep8 = 3×4 + 8（:476 给出 16 与 20）。每卡数本脚本现算。
  · Qwen3-Coder-480B 单机 v7x-8 对多机 v7x-16（TP16）输出 tok/s：
    tpu/tpu-inference/Qwen3-Coder-480B/README-multinode.md:441-445（跨 DCN）、
    README-TPU-VM.md:1000-1009（ICI slice，Workload Policy）。
  · V4-Pro SGLang 必须同一 NVL72 域：gpu/inference/a4x-max/deepseek-v4/SGLANG-V4PRO-RUNBOOK.md:34、63。
  · DeepSeek-V3 结构与 TPU v7 规格：topic06_numbers.py（HF config.json ＋ 官方规格）；
    每 device 可用 HBM 94.75 GB、FP8 MoE 实测约 101.8 GB/device：tpu/tpu-inference/DeepSeek-R1-671B-FP4/README.md:59、73。
  · R1 / SGLang / GB300 3P2D（20 GPU）并发扫描（总吞吐，1024 进 512 出）与「~288 并发」估算：
    gpu/inference/a4x-max/deepseek-v3/sglang-r1-nvfp4-gb300-3p2d-DEPLOY-GUIDE.md:401、405-413。

刻意没画
  · fig6-s10-lens 不画延迟。dep8 同时把出字间隔从 46.8 降到 11.8 ms，那是第六节的事，这里只讲口径。
  · fig6-s10-rank 不画真实 v7 的 3D torus 拓扑和任何事故记录，只画「顺序决定邻居」的机制。
  · fig6-s10-slice 不算激活的精确峰值，统一按可用 HBM 的 10% 预留（本课假设）；
    也不画 DP 注意力把约 17B 非专家权重在每个 device 上各放一份的额外开销。
"""
import math
from topic03_draw import Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE
import topic06_numbers as N

W = 1400

# ══ ① V4-Pro：总吞吐 vs 每卡 ═══════════════════════════════════════════
#   (名字, 总吞吐, GPU 数, 并发)
CFG = [("1p1d", 14563, 8, 256),
       ("3p1d", 16459, 16, 256),
       ("3p1d", 21100, 16, 512),
       ("3p＋dep8", 55153, 20, 512),
       ("3p＋dep8", 65132, 20, 1536)]
PER = [round(t / g) for _, t, g, _ in CFG]
assert PER == [1820, 1029, 1319, 2758, 3257], PER
TUNE = round((21100 / 14563 - 1) * 100)            # 「调参」：1p1d → 3p1d 最佳，总吞吐
TOPO = round((65132 / 14563 - 1) * 100)            # 「换拓扑」：1p1d → dep8 最佳，总吞吐
assert (TUNE, TOPO) == (45, 347)
TUNE_PER = round((1319 / 1820 - 1) * 100)          # 同一步换成每卡
assert TUNE_PER == -28
PURE_CONC = round((21100 / 16459 - 1) * 100)       # 同为 3p1d，只调并发 256 → 512
assert PURE_CONC == 28
X_VS_3P = round(3257 / 1319, 2)                    # 原文的每卡 2.47×（对 3p1d 最佳）
X_VS_1P = round(3257 / 1820, 2)                    # 对 1p1d 起点
assert (X_VS_3P, X_VS_1P) == (2.47, 1.79)

# ══ ② Qwen3-Coder：单机 vs 多机（输出 tok/s）════════════════════════════
#   (场景, 单机 v7x-8 TP8, 多机 DCN TP16, 多机 ICI slice TP16 或 None)
WIRE = [("1K/1K c=1", 48, 37.5, 37.7),
        ("1K/1K c=4", 177, 98.4, 143.2),
        ("1K/1K c=16", 602, 220.0, 513.8),
        ("8K/1K c=4", 162, 134.3, 136.0),
        ("8K/1K c=16", 483, 223.1, None)]
DCN = [round((d / s - 1) * 100) for _, s, d, _ in WIRE]
ICI = [round((i / s - 1) * 100) if i else None for _, s, _, i in WIRE]
assert DCN == [-22, -44, -63, -17, -54], DCN
assert ICI == [-21, -19, -15, -16, None], ICI
assert (min(DCN), max(DCN)) == (-63, -17) and (min(i for i in ICI if i), max(i for i in ICI if i)) == (-21, -15)

# ══ ④ DeepSeek-V3 在 TPU v7 上的最小切片（本课推导）════════════════════
GB = 1e9
W_FP8 = N.V3_TOTAL * 1 / GB                         # FP8：每参数 1 字节（忽略 block scale）
assert round(W_FP8) == 671
KVT = N.V3_KV_PER_TOKEN                              # 70,272 B / token（bf16 MLA，61 层）
assert KVT == 70272
DEV_USABLE = 94.75                                   # GB / device，实测可用（R1 README:59）
CHIP_USABLE = DEV_USABLE * N.V7_DEVICES_PER_CHIP     # 189.5 GB / chip
RESERVE = 0.10                                       # 本课假设：激活 ＋ 编译临时区
CHIP_FOR_DATA = CHIP_USABLE * (1 - RESERVE)
assert round(CHIP_USABLE, 1) == 189.5 and round(CHIP_FOR_DATA, 2) == 170.55
SLICES = [4, 8, 16, 32]                              # v7x-8 / -16 / -32 / -64（名字按 device 数）
CTX = [("4K", 4096), ("32K", 32768), ("128K", 131072)]
CONCS = [64, 256]


def need(ctx, conc):
    kv = ctx * conc * KVT / GB
    tot = W_FP8 + kv
    raw = tot / CHIP_FOR_DATA
    chips = next(s for s in SLICES if s >= raw)
    return kv, tot, raw, chips


GRID = {(c, n): need(t, n) for c, t in CTX for n in CONCS}
TOTS = {k: round(v[1]) for k, v in GRID.items()}
CHIPS = {k: v[3] for k, v in GRID.items()}
assert TOTS == {("4K", 64): 689, ("4K", 256): 745, ("32K", 64): 818, ("32K", 256): 1260,
                ("128K", 64): 1260, ("128K", 256): 3029}, TOTS
assert CHIPS == {("4K", 64): 8, ("4K", 256): 8, ("32K", 64): 8, ("32K", 256): 8,
                 ("128K", 64): 8, ("128K", 256): 32}, CHIPS
LEFT4 = 4 * CHIP_FOR_DATA - W_FP8                     # 4 颗芯片放完权重还剩多少给 KV
LEFT4_TOK = LEFT4 * GB / KVT
assert round(LEFT4) == 11 and round(LEFT4_TOK / 1000) == 159
MAX4_AT_4K = int(LEFT4_TOK // 4096)
assert MAX4_AT_4K == 38
# 纸面的 FP8 MoE 每 device：路由专家 256 × 58 层 × 3 个矩阵 × 7168 × 2048
ROUTED = N.V3_EXPERTS * 58 * 3 * N.V3_HIDDEN * 2048
assert round(ROUTED / 1e9, 1) == 653.9
PAPER_MOE_DEV = ROUTED / GB / 8
assert round(PAPER_MOE_DEV, 1) == 81.7
REAL_MOE_DEV = 101.8                                  # 实测记录（R1 README:73）

# ══ ⑤ R1 3P2D 并发扫描（总吞吐 tok/s，20 GPU）══════════════════════════
SWEEP = [(8, 854), (32, 4635), (64, 5265), (128, 6878), (192, 9975),
         (256, 10390), (384, 10438), (512, 10715)]
CAP_EST = 288
USE8 = round(8 / CAP_EST * 100, 1)
assert USE8 == 2.8                                    # 原文写「~3%」
PEAK_X = round(10715 / 854, 1)
assert PEAK_X == 12.5


def _axes(f, x0, y0, w, h, ymax, yticks, fmt):
    for v in yticks:
        yy = y0 + h - h * v / ymax
        f.line(x0, yy, x0 + w, yy, "#eceff1", 1, arrow=False)
        f.t(x0 - 10, yy + 4, fmt(v), GY, size=12, anchor="end")
    f.line(x0, y0 + h, x0 + w, y0 + h, GY2, 1.2, arrow=False)


# ─────────────────────────────────────────────────────────────────────────
def fig_lens():
    f = Fig(W, "同一组 V4-Pro 实测数，左边画总吞吐，右边画每张卡的吞吐。总吞吐看：1p1d 14563，"
               "3p1d 调到最好 21100 是加 45%，换 dep8 到 65132 是加 347%。但这三档分别用了 8、16、20 张卡。"
               "换成每卡：1p1d 1820，3p1d 最好 1319，反而少 28%；dep8 最好 3257，是 3p1d 的 2.47 倍、1p1d 的 1.79 倍。"
               "同为 3p1d，只把并发从 256 调到 512，总吞吐加 28%")
    y0 = f.header("同一组数，换个分母结论就变　——　<tspan font-weight=\"700\">「调参 +45%」按每卡算是 -28%</tspan>",
                  "GB300 · DeepSeek-V4-Pro · vLLM · 4K 进 1K 出。吞吐 ＝ prompt ＋ 输出 token 每秒；柱下是 GPU 数和并发",
                  [(GY2, "1p1d 起点"), (OR, "3p1d：加 prefill、调并发"), (BL, "换 decode 拓扑：dep8")])
    PH = 460
    PW = (W - 20) // 2
    cols = [GY2, OR, OR, BL, BL]

    def draw(px, title, vals, ymax, ticks, fmt, notes):
        py = f.panel(px, y0, PW, PH, title, INK)
        X0, AW, AY, AH = px + 80, PW - 110, py + 40, PH - 200
        _axes(f, X0, AY, AW, AH, ymax, ticks, fmt)
        bw = AW / len(vals)
        for i, v in enumerate(vals):
            bx = X0 + i * bw + bw * 0.18
            bh = AH * v / ymax
            f.box(bx, AY + AH - bh, bw * 0.64, bh, cols[i], cols[i], 3)
            f.t(bx + bw * 0.32, AY + AH - bh - 8, format(v, ","), cols[i], True, 13.5, "middle")
            nm, _, g, c = CFG[i]
            f.t(bx + bw * 0.32, AY + AH + 20, nm, INK, True, 12.5, "middle")
            f.t(bx + bw * 0.32, AY + AH + 38, "%d 卡" % g, GY, size=12, anchor="middle")
            f.t(bx + bw * 0.32, AY + AH + 55, "并发 %d" % c, GY, size=11.5, anchor="middle")
        for k, (s, col) in enumerate(notes):
            f.t(px + 20, py + PH - 70 + k * 24, s, col, True, 14)
        f._pan = None
        return X0, AY, AW, AH, bw

    draw(0, "总吞吐（tok/s）", [t for _, t, _, _ in CFG], 70000, [0, 20000, 40000, 60000],
         lambda v: "%dK" % (v // 1000) if v else "0",
         [("调参：14,563 → 21,100，+45%（8 卡 → 16 卡）", OR),
          ("换拓扑：14,563 → 65,132，+347%（8 卡 → 20 卡）", BL)])
    draw(PW + 20, "每卡吞吐（tok/s/GPU，本课现算 ＝ 总吞吐 ÷ 卡数）", PER, 3600, [0, 1000, 2000, 3000],
         lambda v: format(v, ","),
         [("调参那一步：1,820 → 1,319，每卡 -28%", RD),
          ("换拓扑：3,257 ＝ 3p1d 最好的 2.47×，1p1d 起点的 1.79×", BL)])
    yb = f.band(y0 + PH + 20, "ok", "比较两个配置之前，先问分母是不是同一个", [
        "+45% 里大半是加卡：卡数翻倍，总量只涨一半。同为 3p1d 只调并发（256 → 512），总吞吐才是 +28%。",
        "换拓扑在两种口径下都赢：不是加了卡才赢，每张卡搬一趟能分给的 token 变多了。",
    ])
    yb = f.src(yb + 10,
               "📌 gpu/inference/a4x-max/deepseek-v4/VLLM-V4PRO-RUNBOOK.md:446（1p1d 并发 256，14,563）、:448（3p1d 并发 256，16,459）、"
               ":449（3p1d 并发 512，21,100）、:465（3p ＋ dep8 并发 512，55,153）、:468（并发 1,536，65,132）。"
               "「+45% / +347%」原话见 :489；「每卡 2.47×」见 :476；「256→512 涨 28%」见 :487。",
               "⚠️ 本课推导：每卡数 ＝ 总吞吐 ÷ GPU 数（1p1d 8 卡、3p1d 16 卡、3p ＋ dep8 20 卡，后两个数见 :476）；"
               "-28% 与 1.79× 是现算，原文没有。吞吐口径是 prompt 加输出的总 token，sa-bench、temperature=0。")
    f.save("fig6-s10-lens.svg", yb + 14)


# ─────────────────────────────────────────────────────────────────────────
def fig_wire():
    f = Fig(W, "Qwen3-Coder-480B 在 TPU v7 上，单机 v7x-8 跑 TP8 作对照。换成两台机器拼成 v7x-16 跑 TP16："
               "如果两台之间走数据中心网络 DCN，输出吞吐低 17% 到 63%，并发越高越惨；"
               "如果两台被编进同一个 ICI 切片，只低 15% 到 21%。GPU 这边，V4-Pro 的 SGLang 部署要求所有节点在同一个 NVL72 域里，"
               "跨域时 PD 之间传 KV 会从 NVLink 退化到 RDMA")
    y0 = f.header("卡数一样，线不一样　——　<tspan font-weight=\"700\">TP16 跨 DCN 最多慢 63%，同一 ICI 切片只慢 15–21%</tspan>",
                  "Qwen3-Coder-480B · TPU v7 · vLLM（tpu-inference）。纵轴：多机 TP16 的输出吞吐相对单机 v7x-8 TP8 差多少",
                  [(RD, "两台 v7x-8 走 DCN"), (GR, "两台编进同一个 ICI 切片")])
    PH = 420
    LW = 900
    py = f.panel(0, y0, LW, PH, "多机 TP16 对单机 TP8（输出 tok/s）", INK)
    X0, AW, AY, AH = 110, LW - 150, py + 40, PH - 130
    for v in (0, 20, 40, 60):
        yy = AY + AH * v / 70
        f.line(X0, yy, X0 + AW, yy, "#eceff1" if v else GY2, 1.2 if not v else 1, arrow=False)
        f.t(X0 - 10, yy + 4, "-%d%%" % v if v else "打平", GY, size=12, anchor="end")
    gw = AW / len(WIRE)
    for i, (nm, s, d, ic) in enumerate(WIRE):
        gx = X0 + i * gw
        for j, (val, col) in enumerate(((-DCN[i], RD), (-ICI[i] if ICI[i] is not None else None, GR))):
            bx = gx + gw * 0.14 + j * gw * 0.38
            if val is None:
                f.t(bx + gw * 0.17, AY + 24, "未测", GY, size=12, anchor="middle")
                continue
            bh = AH * val / 70
            f.box(bx, AY, gw * 0.34, bh, col, col, 3)
            f.t(bx + gw * 0.17, AY + bh + 18, "-%d%%" % val, col, True, 13.5, "middle")
        f.t(gx + gw / 2, AY + AH + 24, nm, INK, True, 12.5, "middle")
    f.t(X0 + AW / 2, AY + AH + 50, "柱子往下长：越长越慢", GY, size=12.5, anchor="middle")
    f._pan = None
    RX, RW = LW + 20, W - LW - 20
    py2 = f.panel(RX, y0, RW, PH, "同样的规矩在 GPU 上", BL)
    rows = [("TPU v7", "芯片之间：ICI", "切片之间：DCN"),
            ("GB300", "NVL72 域内：NVLink", "域之间：RDMA")]
    for k, (hw, fast, slow) in enumerate(rows):
        yy = py2 + 30 + k * 100
        f.t(RX + 18, yy + 14, hw, INK, True, 15)
        f.box(RX + 18, yy + 28, 190, 36, "#e6f4ea", GR, 5)
        f.t(RX + 113, yy + 51, fast, GR, True, 13, "middle")
        f.box(RX + 226, yy + 28, 200, 36, "#fce8e6", RD, 5)
        f.t(RX + 326, yy + 51, slow, RD, True, 13, "middle")
    f.t(RX + 18, py2 + 248, "V4-Pro SGLang 部署：16 个节点", INK, True, 14)
    f.t(RX + 18, py2 + 272, "必须同一 NVL72 域", RD, True, 14)
    f.t(RX + 18, py2 + 300, "跨域后 PD 之间传 KV 退化到 RDMA，", GY, size=13)
    f.t(RX + 18, py2 + 322, "原文说「显著变慢」，没给倍数", GY, size=13)
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "快线多大，决定一个引擎最多铺多大", [
        "TP 每出一个 token 都要做一次 all-reduce；这一步跨慢线，并发越高越吃亏（1K/1K 从 -22% 一路掉到 -63%）。",
        "原记录的建议是：装得下就别跨，要扩容就开两份单机做 DP，让每份 TP 都留在快线里。",
    ])
    yb = f.src(yb + 10,
               "📌 跨 DCN：tpu/tpu-inference/Qwen3-Coder-480B/README-multinode.md:441-445（单机 v7x-8 TP8 对多机 v7x-16 TP16，输出 tok/s）；"
               "原因与 DP 建议见同文件 :457 起。",
               "📌 ICI 切片：tpu/tpu-inference/Qwen3-Coder-480B/README-TPU-VM.md:1000-1009（Workload Policy 把两台编进 2x2x2 切片，"
               "表中 Aggregate tok/s；8K/1K c=16 没测，记作「未测」）。",
               "📌 NVL72：gpu/inference/a4x-max/deepseek-v4/SGLANG-V4PRO-RUNBOOK.md:34（必须同一 NVL72 域）、:63（跨域退化到 RDMA）。")
    f.save("fig6-s10-wire.svg", yb + 14)


# ─────────────────────────────────────────────────────────────────────────
def fig_rank():
    f = Fig(W, "四颗芯片连成一个环，每颗芯片有两个 device，共 8 个 device。左：按物理相邻排的顺序，"
               "环形 all-reduce 时每一步都只和隔壁说话，一跳就到。中：同样 8 个 device 但顺序打乱，"
               "程序以为的「下一个」在物理上可能隔着两三跳，数据绕远路，还会和别人抢同一条线。"
               "右：如果不同进程算出来的顺序不一致，A 在等 B，B 在等 C，C 又在等 A，第一次集合通信就卡死，不报错")
    y0 = f.header("TPU 上的 device 顺序就是拓扑　——　<tspan font-weight=\"700\">顺序错了，数据发给错的对端</tspan>",
                  "示意：4 颗芯片连成环，每颗 2 个 device（chip ≠ device）。方块里的数字是 mesh 里的编号，箭头是环形 all-reduce 的「下一个」",
                  [(GR, "一跳到邻居"), (OR, "绕远路"), (RD, "互相等，卡死")])
    PH = 400
    PW = (W - 40) // 3

    def ring(px, py, order, col_of):
        # 4 颗芯片放在正方形四角，物理环：左上 → 右上 → 右下 → 左下
        cx = [px + 70, px + PW - 190, px + PW - 190, px + 70]
        cy = [py + 40, py + 40, py + 210, py + 210]
        for k in range(4):
            f.box(cx[k], cy[k], 120, 70, "none", GY2, 8, 1.4)
            f.t(cx[k] + 4, cy[k] + 88, "芯片 %d" % k, GY, size=11.5)
        pos = []
        for k in range(4):
            for d in range(2):
                pos.append((cx[k] + 10 + d * 56, cy[k] + 12))
        # 物理链路（芯片之间）
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
            ax, ay = cx[a] + 60, cy[a] + 35
            bx, by = cx[b] + 60, cy[b] + 35
            f.line(ax, ay, bx, by, LINE, 6, arrow=False)
        for phys, (x, y) in enumerate(pos):
            f.box(x, y, 44, 44, "#fff", INK, 6, 1.2)
            f.t(x + 22, y + 29, str(order[phys]), INK, True, 16, "middle")
        return pos

    # 左：正确顺序
    py = f.panel(0, y0, PW, PH, "按物理相邻编号", GR)
    good = [0, 1, 2, 3, 4, 5, 6, 7]
    pos = ring(0, py, good, None)
    f.t(20, py + PH - 64, "mesh 里的「下一个」＝ 物理隔壁", GR, True, 14)
    f.t(20, py + PH - 40, "环形 all-reduce 每步一跳", GY, size=13)
    f._pan = None
    # 中：打乱
    X2 = PW + 20
    py = f.panel(X2, y0, PW, PH, "顺序打乱（同样 8 个 device）", OR)
    bad = [0, 4, 2, 6, 1, 5, 3, 7]
    pos = ring(X2, py, bad, None)
    inv = {v: i for i, v in enumerate(bad)}
    a, b = pos[inv[0]], pos[inv[1]]
    f.path([(a[0] + 22, a[1] + 44), (a[0] + 22, a[1] + 90), (b[0] + 22, b[1] + 90), (b[0] + 22, b[1] + 44)],
           OR, 2.2)
    f.t(X2 + 20, py + PH - 64, "0 的「下一个」1 远在对角", OR, True, 14)
    f.t(X2 + 20, py + PH - 40, "数据绕两跳，还和别人抢同一条线", GY, size=13)
    f._pan = None
    # 右：各进程顺序不一致 → 循环等待
    X3 = 2 * (PW + 20)
    py = f.panel(X3, y0, PW, PH, "各进程算出的顺序不一致", RD)
    nodes = [("进程 A", X3 + PW / 2 - 60, py + 30), ("进程 B", X3 + PW - 160, py + 180), ("进程 C", X3 + 40, py + 180)]
    for nm, x, y in nodes:
        f.box(x, y, 120, 50, "#fce8e6", RD, 8)
        f.t(x + 60, y + 31, nm, RD, True, 14, "middle")
    f.path([(nodes[0][1] + 120, nodes[0][2] + 40), (nodes[1][1] + 60, nodes[1][2])], RD, 2)
    f.path([(nodes[1][1], nodes[1][2] + 30), (nodes[2][1] + 120, nodes[2][2] + 30)], RD, 2)
    f.path([(nodes[2][1] + 60, nodes[2][2]), (nodes[0][1], nodes[0][2] + 40)], RD, 2)
    f.t(X3 + PW / 2, py + 150, "都在等对方先发", RD, size=13, anchor="middle")
    f.t(X3 + 20, py + PH - 64, "第一次集合通信就卡住", RD, True, 14)
    f.t(X3 + 20, py + PH - 40, "不报错，只是不动", GY, size=13)
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "mesh 别手排：让框架按物理拓扑生成", [
        "JAX 的 mesh_utils.create_device_mesh 按物理位置排 device，跨切片用 create_hybrid_device_mesh 分开 ICI 和 DCN 两层。",
        "名字也别数错：v7x-8 是 4 颗芯片、8 个 device。按芯片数去配 mesh 形状，形状就对不上。",
    ])
    yb = f.src(yb + 10,
               "📌 JAX 文档 jax.experimental.mesh_utils：create_device_mesh「creates a performant device mesh」，"
               "create_hybrid_device_mesh 分 ICI 与 DCN 两层。",
               "📌 v7x-8 ＝ 4 chips ／ 8 devices：tpu/tpu-inference/README.md:41。",
               "⚠️ 本课示意：只画「顺序决定邻居」的机制；真实 v7 是 3D torus，环的画法和 device 编号都是简化。卡死那一栏是机制推演，不对应某次公开事故。")
    f.save("fig6-s10-rank.svg", yb + 14)


# ─────────────────────────────────────────────────────────────────────────
def fig_slice():
    f = Fig(W, "DeepSeek-V3 用 FP8 权重 671 GB，每个 token 的 KV 70272 字节。TPU v7 每颗芯片可用约 189.5 GB，"
               "预留一成给激活后按 170.55 GB 算。4K 上下文 64 并发要 689 GB，4.04 颗芯片，切片取 8 颗；"
               "4K 256 并发 745 GB，32K 64 并发 818 GB，32K 256 并发和 128K 64 并发都是 1260 GB，都是 8 颗；"
               "128K 256 并发要 3029 GB，要 32 颗。4 颗芯片放完权重只剩约 11 GB 给 KV，约 15.9 万个 token。"
               "而实测记录里 FP8 的 MoE 每个 device 就要 101.8 GB，超过 94.75 GB，所以 v7x-8 上实际用 FP4")
    y0 = f.header("DeepSeek-V3 在 TPU v7 上最少几颗芯片　——　<tspan font-weight=\"700\">权重是地板，KV 决定往上加多少</tspan>",
                  "本课推导：FP8 权重 671 GB ＋ KV（每 token 70,272 B）÷ 每芯片 170.55 GB（可用 189.5 GB，预留一成给激活），向上取到切片档位",
                  [(BL, "权重 671 GB"), (OR, "KV ＝ 上下文 × 并发 × 70,272 B"), (GY2, "切片档位")])
    PH = 470
    LW = 930
    py = f.panel(0, y0, LW, PH, "要多少 HBM（GB），取几颗芯片", INK)
    X0, AW, AY, AH = 130, LW - 330, py + 30, PH - 90
    YMAX = 3600
    rows = [(c, n) for c, _ in CTX for n in CONCS]
    bh = AH / len(rows) * 0.62
    sx = AW / YMAX
    for s in SLICES[:3]:
        gx = X0 + s * CHIP_FOR_DATA * sx
        f.line(gx, AY - 6, gx, AY + AH, GY2, 1.1, "4 3", arrow=False)
        f.t(gx, AY + AH + 18, "%d 颗" % s, GY, size=12, anchor="middle")
    f.t(X0 + 16 * CHIP_FOR_DATA * sx + 40, AY + AH + 18, "虚线 ＝ 该档能放多少（32 颗在图外）", GY, size=11.5)
    for i, key in enumerate(rows):
        kv, tot, raw, chips = GRID[key]
        yy = AY + i * AH / len(rows) + 6
        f.t(X0 - 12, yy + bh / 2 + 5, "%s × %d" % key, INK, True, 13.5, "end")
        f.box(X0, yy, W_FP8 * sx, bh, BL, BL, 3)
        f.box(X0 + W_FP8 * sx, yy, max(kv * sx, 2), bh, OR, OR, 3)
        f.t(X0 + tot * sx + 8, yy + bh / 2 + 5, format(round(tot), ","), INK, True, 13)
        f.t(LW - 170, yy + bh / 2 + 5, "%.2f 颗 → %d 颗" % (raw, chips), RD if chips > 8 else INK, True, 14)
    f.t(LW - 170, AY - 8, "纸面 → 取档", GY, True, 12.5)
    f._pan = None
    RX, RW = LW + 20, W - LW - 20
    py2 = f.panel(RX, y0, RW, PH, "4 颗芯片（v7x-8）为什么不够", RD)
    lines = [("4 颗能放的", "%.0f GB" % (4 * CHIP_FOR_DATA), INK),
             ("减去 FP8 权重", "%d GB" % round(W_FP8), BL),
             ("剩给 KV", "约 %d GB" % round(LEFT4), OR),
             ("折合 token", "约 %.1f 万" % (LEFT4_TOK / 1e4), OR),
             ("4K 请求最多", "%d 个" % MAX4_AT_4K, OR)]
    for k, (a, b, col) in enumerate(lines):
        yy = py2 + 36 + k * 36
        f.t(RX + 18, yy, a, GY, size=13.5)
        f.t(RX + RW - 18, yy, b, col, True, 15, "end")
    yy = py2 + 36 + 5 * 36
    f.line(RX + 16, yy - 14, RX + RW - 16, yy - 14, LINE, 1, arrow=False)
    f.t(RX + 18, yy + 10, "实测比纸面更紧", RD, True, 14.5)
    f.t(RX + 18, yy + 38, "纸面：FP8 MoE 每 device", GY, size=13)
    f.t(RX + RW - 18, yy + 38, "约 %.1f GB" % PAPER_MOE_DEV, INK, True, 13.5, "end")
    f.t(RX + 18, yy + 64, "实测记录", GY, size=13)
    f.t(RX + RW - 18, yy + 64, "约 %.1f GB" % REAL_MOE_DEV, RD, True, 13.5, "end")
    f.t(RX + 18, yy + 90, "每 device 可用", GY, size=13)
    f.t(RX + RW - 18, yy + 90, "%.2f GB" % DEV_USABLE, INK, True, 13.5, "end")
    f.t(RX + 18, yy + 124, "装不下 → v7x-8 上实际用 FP4", RD, True, 14)
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "先用纸面账定下限，再用一次真实启动定上限", [
        "短上下文时权重占九成，几档并发都落在 8 颗；128K × 256 时 KV 是权重的 3.5 倍，一下跳到 32 颗。",
        "纸面账只是下限：同一个 FP8 MoE，实测每 device 比纸面多约 20 GB，原记录没拆来源。",
    ])
    yb = f.src(yb + 10,
               "📌 DeepSeek-V3 结构（671B、61 层、MLA kv_lora_rank 512 ＋ rope 64、256 专家）：topic06_numbers.py，来自 HF config.json。"
               "每 token KV 70,272 B ＝ (512 ＋ 64) × 2 B × 61 层。",
               "📌 每 device 可用 94.75 GB、FP8 MoE 约 101.8 GB/device 超 HBM 所以用 FP4：tpu/tpu-inference/DeepSeek-R1-671B-FP4/README.md:59、:72-73（R1 与 V3 同架构）。",
               "⚠️ 本课推导与假设：FP8 每参数 1 字节、不计 block scale；激活预留 10% 是假设；KV 不重复（DP 注意力或 EP），"
               "没算 DP 注意力在每个 device 上各放一份约 17B 非专家权重的开销；切片档位取 4／8／16／32 颗（v7x-8／16／32／64）。"
               "纸面 MoE 81.7 GB/device ＝ 256 × 58 层 × 3 × 7168 × 2048 字节 ÷ 8 device。")
    f.save("fig6-s10-slice.svg", yb + 14)


# ─────────────────────────────────────────────────────────────────────────
def fig_conc():
    f = Fig(W, "DeepSeek-R1 在 GB300 上用 SGLang 跑 3P2D，共 20 张卡，1024 进 512 出。并发 8 时总吞吐 854，"
               "32 时 4635，64 时 5265，128 时 6878，192 时 9975，256 时 10390，384 时 10438，512 时 10715。"
               "文档估算这套 DEP8 decode 能扛约 288 个并发，并发 8 只用到约 3%。峰值是并发 8 的 12.5 倍，256 之后基本见顶")
    y0 = f.header("拿 conc=8 的数估容量，只看到了 3%　——　<tspan font-weight=\"700\">峰值是它的 12.5 倍</tspan>",
                  "GB300 · DeepSeek-R1 · SGLang · 3P2D（20 卡）· 1024 进 512 出。纵轴：总吞吐（prompt ＋ 输出 token 每秒）",
                  [(RD, "conc=8：只用到估算容量的约 3%"), (BL, "往上扫，直到见顶"), (GY2, "估算容量约 288 并发")])
    PH = 420
    py = f.panel(0, y0, W, PH, "并发 → 总吞吐", INK)
    X0, AW, AY, AH = 120, W - 200, py + 40, PH - 120
    YMAX = 12000
    XMAX = 540
    _axes(f, X0, AY, AW, AH, YMAX, [0, 4000, 8000, 12000], lambda v: format(v, ","))

    def px(c):
        return X0 + AW * c / XMAX

    def pyv(v):
        return AY + AH - AH * v / YMAX
    gx = px(CAP_EST)
    f.line(gx, AY - 4, gx, AY + AH, GY2, 1.4, "6 4", arrow=False)
    f.t(gx + 8, AY + 12, "估算容量 ~288", GY, True, 13)
    f.path([(px(c), pyv(v)) for c, v in SWEEP], BL, 2.6, arrow=False)
    for c, v in SWEEP:
        col = RD if c == 8 else BL
        f.box(px(c) - 5, pyv(v) - 5, 10, 10, col, col, 5)
        f.t(px(c), AY + AH + 22, str(c), GY, size=12, anchor="middle")
    f.t(X0 + AW / 2, AY + AH + 48, "并发（压测端发的）", GY, size=13, anchor="middle")
    f.t(px(8) + 14, pyv(854) - 6, "854", RD, True, 15)
    f.t(px(256), pyv(10390) - 16, "10,390（甜点，TTFT 12 s）", BL, True, 13.5, "middle")
    f.t(px(512) - 6, pyv(10715) + 28, "10,715（峰值，TTFT 30 s）", BL, True, 13.5, "end")
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "容量要扫出来：一直加并发，直到吞吐不涨、延迟开始爆", [
        "conc=8 时 decode 每一趟只分给 8 个请求，测到的是「这套机器闲着时多快」，不是它能扛多少。",
        "过了 256 吞吐只再涨 3%，首字延迟从 12 s 涨到 30 s。甜点由延迟目标定，不由吞吐峰值定。",
    ])
    yb = f.src(yb + 10,
               "📌 gpu/inference/a4x-max/deepseek-v3/sglang-r1-nvfp4-gb300-3p2d-DEPLOY-GUIDE.md:401（「conc=8 只用到 DEP8 decode 容量的 ~3%，"
               "GB300 DEP8 理论可扛 ~288 并发」，是估算不是实测）、:405-413（并发扫描表：总吞吐、TTFT median）、:415（峰值是 conc8 的 12.5×）。",
               "⚠️ 横轴是压测端设的并发；表中「实际并发」更低（512 时实测 416），原因是开环压测里有请求在排队。8 ÷ 288 ＝ 2.8%，原文写约 3%。")
    f.save("fig6-s10-conc.svg", yb + 14)


if __name__ == "__main__":
    fig_lens()
    fig_wire()
    fig_rank()
    fig_slice()
    fig_conc()
