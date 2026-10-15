# -*- coding: utf-8 -*-
r"""专题六 · 第十三节「落地：TPU 服务化与编排」的三张静态图。

回答的问题
  · fig6-s13-bucket  服务侧怎么选桶？一步里整批的 token 总数补到桶的上沿（RPA 的 ragged 布局下单条请求不再补齐）；桶越多 padding 越少，但每个桶都要编译一次。
  · fig6-s13-cold    TPU 上一个引擎从启动到能接请求，时间花在哪？为什么「权重加载」那一段常常其实是编译？
  · fig6-s13-route   路由层能做什么？KV 感知路由把首字延迟从多少降到多少？坏副本不摘掉会怎样？
  · fig6-mem-leak    为什么强杀一个 decode 之后下次必然 OOM，而报错给的建议方向正好反了？

数从哪来
  · fig6-s13-bucket：本课推导示意。每步一批的 token 总数取对数正态（中位 700 token、σ ＝ 0.9、截到 8,192），
    用 2,000 个分位点确定性生成，不是任何真实流量。
    Qwen3.5-397B 新形状首个请求要编译 60–90 s：tpu/tpu-inference/Qwen3.5-397B-A17B-FP8/README.md:794。
  · fig6-s13-cold：
    GLM-5.1（v7x-8，未优化）启动拆账：tpu/tpu-inference/GLM-5.1-754B-FP4/README.md:470-476；
    Qwen3-Coder-480B 两机 ICI 切片首次启动 ~40 min ＝ 加载 5 ＋ XLA 编译 29 ＋ 初始化 6：
      tpu/tpu-inference/Qwen3-Coder-480B/README-TPU-VM.md:1011；
    MiMo-V2.5-Pro（2× v7x-8）~50 min ＝ 权重 25 ＋ FP8 QKV 反量化 5 ＋ XLA 编译 15 ＋ decode 预编译 7：
      tpu/tpu-inference/MiMo-V2.5-Pro-BF16/README.md:184-188；
    R1 的 clear_caches：1,367 个 tensor 只有 25 种形状，non-MoE 加载 215.7 s → 21.4 s；
      重启后进程内 transfer cache 丢失，1:48 → 3:17–3:51：tpu/tpu-inference/DeepSeek-R1-671B-FP4/README.md:1202-1223、1238-1259；
    Kimi K2.6 无缓存 57 min → /dev/shm 缓存 6 min 8 s（9.3×）：tpu/tpu-inference/Kimi-K2.6-1T-A32B-INT4/README.md:39-40。
  · fig6-s13-route：Red Hat Developer 2025-10-07「Master KV cache aware routing with llm-d」：
    命中缓存 TTFT 340 ms，对照（无路由的最坏情况）2,850 ms，命中率 87.4%（4,776 次中 4,176 次）。
    坏副本那一栏是本课推导示意（8 个副本坏 1 个，轮询时 1/8 的请求落空）。

刻意没画
  · fig6-s13-bucket 不画 batch 维的桶。真实服务的编译次数是 batch 档数 × token 档数，这里只画一维。
  · fig6-s13-cold 不把三个模型横向比快慢：拓扑、精度、优化程度都不同，只看每根条的构成。
  · fig6-s13-route 不画 llm-d / Dynamo / GKE Inference Gateway 各自的打分公式，只画「同前缀送同一个副本」这一个动作。
"""
import math
from statistics import NormalDist
from topic03_draw import Fig, wpx, BL, OR, GR, RD, PU, GY, INK, GY2, LINE

W = 1400

# ══ ① 分桶：padding 浪费 vs 编译次数（本课推导示意）══════════════════════
NREQ, MED, SIG, CAP = 2000, 700, 0.9, 8192
LENS = [min(CAP, max(16, round(MED * math.exp(SIG * NormalDist().inv_cdf((i + 0.5) / NREQ)))))
        for i in range(NREQ)]
assert sorted(LENS)[NREQ // 2] == 700


def waste(bk):
    tot = pad = 0
    for ln in LENS:
        b = next(x for x in bk if x >= ln)
        tot += b
        pad += b - ln
    return round(pad / tot * 100, 1)


SCHEMES = [("只有 8K 一个桶", [8192], GY2),
           ("4 个桶（1K 起翻倍）", [1024, 2048, 4096, 8192], GY2),
           ("2 的幂（128 起）", [128 * 2 ** k for k in range(7)], BL),
           ("√2 倍一档", sorted(set(int(round(128 * 2 ** (k / 2) / 16) * 16) for k in range(13))), GR),
           ("每 512 一档", list(range(512, 8193, 512)), OR),
           ("每 256 一档", list(range(256, 8193, 256)), OR)]
WASTE = [(nm, len(b), waste(b), col) for nm, b, col in SCHEMES]
assert [(n, w) for _, n, w, _ in WASTE] == [(1, 87.3), (4, 39.1), (7, 29.9), (13, 16.0), (16, 19.7), (32, 10.8)], WASTE
COMPILE_S = (60, 90)                     # Qwen3.5-397B 新形状首个请求的编译时间（秒）

# ══ ② 冷启动拆账（分钟）══════════════════════════════════════════════
#   类别：init 初始化 / load 权重加载 / fake 名义加载、实为重编译 / conv 反量化 / comp 编译 / warm 预编译与 warmup / gap 未拆
GLM = [("init", 1 + 44 / 60), ("fake", 6 + 29 / 60), ("load", 1 + 51 / 60), ("warm", 25 / 60)]
GLM_STATED = 10 + 44 / 60
GLM_GAP = GLM_STATED - sum(v for _, v in GLM)
assert round(GLM_GAP * 60) == 15          # 原表分项加起来 10m29s，总数写 ~10m44s
GLM.append(("gap", GLM_GAP))
CODER = [("load", 5), ("comp", 29), ("init", 6)]
assert sum(v for _, v in CODER) == 40
MIMO = [("load", 25), ("conv", 5), ("comp", 15), ("warm", 7)]
assert sum(v for _, v in MIMO) == 52      # 原表总计写 ~50 min
BARS = [("GLM-5.1 · v7x-8 · 未优化", "原表 ~10 分 44 秒", GLM, GLM_STATED),
        ("Qwen3-Coder · 两机 ICI 切片 · 首次", "原表 ~40 分钟", CODER, 40),
        ("MiMo-V2.5-Pro · 2× v7x-8", "原表 ~50 分钟（分项和 52）", MIMO, 50)]
CAT = {"init": ("初始化", GY2), "load": ("权重加载", BL), "fake": ("名义加载，实为重编译", PU),
       "conv": ("反量化", OR), "comp": ("XLA 编译", RD), "warm": ("预编译 ／ warmup", OR), "gap": ("未拆", LINE)}
R1_FIX = (215.7, 21.4)
assert round(R1_FIX[0] / R1_FIX[1], 1) == 10.1
KIMI = (57 * 60, 6 * 60 + 8)
assert round(KIMI[0] / KIMI[1], 1) == 9.3

# ══ ③ 路由 ═══════════════════════════════════════════════════════════
TTFT_COLD, TTFT_HIT = 2850, 340
assert round((1 - TTFT_HIT / TTFT_COLD) * 100) == 88
HIT = round(4176 / 4776 * 100, 1)
assert HIT == 87.4
NREP, NBAD = 8, 1
LOST = NBAD / NREP * 100
assert LOST == 12.5


def _axes(f, x0, y0, w, h, ymax, yticks, fmt):
    for v in yticks:
        yy = y0 + h - h * v / ymax
        f.line(x0, yy, x0 + w, yy, "#eceff1", 1, arrow=False)
        f.t(x0 - 10, yy + 4, fmt(v), GY, size=12, anchor="end")
    f.line(x0, y0 + h, x0 + w, y0 + h, GY2, 1.2, arrow=False)


# ─────────────────────────────────────────────────────────────────────────
HATCH = ('<defs><pattern id="hatchR" width="8" height="8" patternUnits="userSpaceOnUse" '
         'patternTransform="rotate(45)"><rect width="8" height="8" fill="#fce8e6"/>'
         '<line x1="0" y1="0" x2="0" y2="8" stroke="#d93025" stroke-width="2.4"/></pattern></defs>')

# 衣服尺码那个小例子：落在 512–1K 这个桶里的三批（示意），都补到 1,024
SIZE_EX = [560, 700, 900]
SIZE_TOP = 1024
assert all(512 < v <= SIZE_TOP for v in SIZE_EX) and SIZE_TOP - 700 == 324


def fig_bucket():
    f = Fig(W, "左边是一份示意的分布：每一步整批请求拼起来的 token 总数，中位数 700，长尾拖到 8K，竖线是 2 的幂的桶边界。"
               "左下角用衣服尺码打比方：总数落在 512 到 1K 的那几步都穿 1K 的码，560、700、900 都补零补到 1,024，斜线部分是白算的零。"
               "右边比较六种分桶：只有一个 8K 桶时，算的 token 里 87.3% 是补出来的零；"
               "4 个桶 39.1%；2 的幂 7 个桶 29.9%；根号 2 倍一档 13 个桶 16%；每 512 一档 16 个桶反而有 19.7%；每 256 一档 32 个桶 10.8%。"
               "每多一个桶就多一次编译")
    f.p.append(HATCH)
    y0 = f.header("桶越细补零越少，编译越多　——　<tspan font-weight=\"700\">按分布的形状切，不按等距切</tspan>",
                  "本课推导示意：一步里整批的 token 总数按对数正态生成（中位 700，长尾到 8K），不是真实流量。桶 ＝ 衣服尺码，总数补零到所在桶的上沿",
                  [(BL, "2 的幂"), (GR, "√2 倍一档"), (OR, "等距"), (GY2, "太粗"), (RD, "补出来的零")])
    PH = 490
    LW = 700
    py = f.panel(0, y0, LW, PH, "每步 token 总数的分布和 2 的幂桶边界", BL)
    X0, AW, AY, AH = 70, LW - 110, py + 40, 250
    lo, hi = math.log2(64), math.log2(8192)

    def px(v):
        return X0 + AW * (math.log2(v) - lo) / (hi - lo)
    NB = 28
    edges = [2 ** (lo + (hi - lo) * k / NB) for k in range(NB + 1)]
    cnt = [sum(1 for ln in LENS if edges[k] <= ln < edges[k + 1] or (k == NB - 1 and ln == 8192)) for k in range(NB)]
    cmax = max(cnt)
    for k, c in enumerate(cnt):
        x1, x2 = px(edges[k]), px(edges[k + 1])
        bh = AH * c / cmax
        f.box(x1 + 1, AY + AH - bh, x2 - x1 - 2, bh, "#d2e3fc", "#d2e3fc", 1)
    f.line(X0, AY + AH, X0 + AW, AY + AH, GY2, 1.2, arrow=False)
    for b in [128 * 2 ** k for k in range(7)]:
        f.line(px(b), AY - 4, px(b), AY + AH, BL, 1.4, "5 4", arrow=False)
        f.t(px(b), AY + AH + 22, "%dK" % (b // 1024) if b >= 1024 else str(b), BL, True, 15, "middle")
    # 中位数：标在 512 和 1K 两条虚线之间的空当上方，不压线
    xm = px(700)
    f.line(xm, AY - 14, xm, AY + 6, INK, 1.6, arrow=False)
    f.t(xm + 6, AY - 18, "中位 700", INK, True, 15)
    # 衣服尺码：512–1K 这个桶里的三个请求，补到 1K
    sy = AY + AH + 50
    f.t(24, sy + 18, "一个桶 ＝ 一个尺码：总数落在 512–1K 的都穿 1K", INK, True, 15)
    SX, SW = 24, 560
    ss = SW / SIZE_TOP
    for k, v in enumerate(SIZE_EX):
        yy = sy + 32 + k * 30
        f.box(SX, yy, v * ss, 22, "#d2e3fc", BL, 3)
        f.t(SX + 8, yy + 17, str(v), BL, True, 14)
        f.box(SX + v * ss, yy, (SIZE_TOP - v) * ss, 22, "url(#hatchR)", RD, 3)
    f.line(SX + SW, sy + 26, SX + SW, sy + 32 + 3 * 30, RD, 1.6, arrow=False)
    f.t(SX + SW + 8, sy + 70, "1,024", RD, True, 15)
    f._pan = None

    RX, RW = LW + 20, W - LW - 20
    py2 = f.panel(RX, y0, RW, PH, "六种分法：桶数（＝ 编译次数）对补零浪费", INK)
    for i, (nm, n, w, col) in enumerate(WASTE):
        yy = py2 + 30 + i * 66
        f.t(RX + 16, yy + 18, nm, INK, True, 16)
        f.t(RX + 16, yy + 42, "%d 个桶" % n, GY, size=15)
        bx = RX + 220
        bw = (RW - 320) * w / 90
        f.box(bx, yy + 8, bw, 30, col, col, 3)
        f.t(bx + bw + 10, yy + 30, "%.1f%%" % w, col, True, 17)
    f._pan = None
    yb = f.src(y0 + PH + 6,
               "⚠️ 本课推导示意：2,000 步的整批 token 总数取对数正态分位点（中位 700、σ ＝ 0.9、截到 8,192），确定性生成；√2 档取 128 × 2^(k/2) 并取整到 16 的倍数。"
               "左下角 560／700／900 三批是随手挑的例子。补齐按整批总数（RPA ragged 布局），不是每条请求各补一次。",
               "📌 新形状首个请求编译 60–90 s、之后 <1 s：tpu/tpu-inference/Qwen3.5-397B-A17B-FP8/README.md:794。",
               "📌 预编译参数 --precompile-bs-paddings 1 4 8 16 32 64 128 256、--precompile-token-paddings 4096：tpu/sglang-jax/mimo/README.md:269-270。")
    f.save("fig6-s13-bucket.svg", yb + 14)


# ─────────────────────────────────────────────────────────────────────────
GLM_LAB = {"init": "初始化 1:44", "fake": "账面：加载非 MoE 权重 6:29", "load": "MoE 权重 1:51",
           "warm": "warmup 0:25", "gap": "未拆 0:15"}


def fig_cold():
    f = Fig(W, "上面把 GLM-5.1 在 v7x-8 上未优化时的冷启动放大，约 10 分 44 秒：初始化 1 分 44 秒，"
               "紫色那段 6 分 29 秒账面上是加载非 MoE 权重，实际是每加载一个 tensor 就清一次编译缓存、下一个又从头编；"
               "真正的 MoE 权重加载只有 1 分 51 秒。下面是另外两次冷启动的构成："
               "Qwen3-Coder 两台机器编进一个 ICI 切片第一次启动约 40 分钟，其中 XLA 编译 29 分钟；"
               "MiMo-V2.5-Pro 在两台 v7x-8 上约 50 分钟，权重加载 25、反量化 5、编译 15、decode 预编译 7")
    y0 = f.header("TPU 冷启动的大头常常是编译　——　<tspan font-weight=\"700\">连「权重加载」那一段也可能是</tspan>",
                  "tpu-inference 公开记录。上：GLM-5.1 一次冷启动放大看；下：另外两次，只看构成，不比快慢",
                  [(BL, "权重加载"), (PU, "账面加载，实为重编译"), (RD, "XLA 编译"), (OR, "反量化 ／ 预编译 ／ warmup"), (GY2, "初始化"), (LINE, "未拆：总时长减分项和")])
    PH = 250
    py = f.panel(0, y0, W, PH, "GLM-5.1 · v7x-8 · 未优化 · 约 10 分 44 秒", PU)
    X0, AW = 40, W - 80
    sx = AW / GLM_STATED
    yy = py + 80
    x = X0
    for cat, v in GLM:
        lab, col = CAT[cat]
        w = v * sx
        f.box(x, yy, w, 54, col, "#fff", 3, 1.5)
        txt = GLM_LAB[cat]
        if cat == "fake":
            f.t(x + w / 2, yy + 34, txt, "#fff", True, 18, "middle")
        elif wpx(txt, 15) + 12 < w:
            f.t(x + w / 2, yy + 33, txt, "#fff", True, 15, "middle")
        x += w
    # 小段标签：放在条下方，避免挤
    x = X0
    under = []
    for cat, v in GLM:
        w = v * sx
        if cat != "fake" and wpx(GLM_LAB[cat], 15) + 12 >= w:
            under.append((x + w / 2, GLM_LAB[cat]))
        x += w
    for k, (cx, txt) in enumerate(under):
        f.line(cx, yy + 58, cx, yy + 72 + k * 24, GY2, 1, arrow=False)
        f.t(cx + 6, yy + 84 + k * 24, txt, GY, True, 15, "end" if cx > W - 200 else None)
    xf0 = X0 + GLM[0][1] * sx
    xf1 = xf0 + GLM[1][1] * sx
    f.path("M%d,%d L%d,%d L%d,%d L%d,%d" % (xf0, yy - 8, xf0, yy - 16, xf1, yy - 16, xf1, yy - 8), PU, 1.6, arrow=False)
    f.t((xf0 + xf1) / 2, yy - 26, "实际在干：每加载一个 tensor 清一次编译缓存，下一个从头编", PU, True, 17, "middle")
    f._pan = None
    # 下：另外两次
    py2 = y0 + PH + 14
    PH2 = 230
    pb = f.panel(0, py2, W, PH2, "另外两次冷启动（分钟）", INK)
    XMAX = 55
    s2 = AW / XMAX
    for k, (nm, note, parts, stated) in enumerate(BARS[1:]):
        yy = pb + 26 + k * 92
        f.t(X0, yy + 10, nm, INK, True, 16)
        f.t(X0 + AW, yy + 10, note, GY, size=15, anchor="end")
        x = X0
        for cat, v in parts:
            lab, col = CAT[cat]
            w = v * s2
            f.box(x, yy + 22, w, 40, col, "#fff", 2, 1)
            f.t(x + w / 2, yy + 48, "%d" % round(v), "#fff", True, 16, "middle")
            x += w
    yb = f.src(py2 + PH2 + 6,
               "📌 GLM-5.1：tpu/tpu-inference/GLM-5.1-754B-FP4/README.md:470-476（未优化列：初始化 1m44s、non-MoE 6m29s、MoE cache 1m51s、server init ＋ warmup 25s，"
               "总计 ~10m44s；分项和是 10m29s，差的 15 s 画成「未拆」）。",
               "📌 Qwen3-Coder：tpu/tpu-inference/Qwen3-Coder-480B/README-TPU-VM.md:1011（加载 ~5 ＋ XLA 编译 ~29 ＋ 初始化 ~6 ≈ 40 min，首次无 XLA cache）。"
               "MiMo-V2.5-Pro：tpu/tpu-inference/MiMo-V2.5-Pro-BF16/README.md:184-188（分项和 52，原表写 ~50）。",
               "📌 正文里缓存那几个数 —— R1：tpu/tpu-inference/DeepSeek-R1-671B-FP4/README.md:1202-1223（clear_caches，1,367 个 tensor 只有 25 种形状，215.7 s → 21.4 s）、"
               ":1238-1259（transfer cache 不持久化，1:48 → 3:17–3:51）。"
               "Kimi K2.6：tpu/tpu-inference/Kimi-K2.6-1T-A32B-INT4/README.md:23、39-40（57 min → 6 min 8 s；原表无缓存那行是 60 层，有缓存是 61 层）。",
               "📌 编译缓存的键：JAX 文档「Persistent compilation cache」。")
    f.save("fig6-s13-cold.svg", yb + 14)


# ─────────────────────────────────────────────────────────────────────────
def fig_route():
    f = Fig(W, "四个副本，三个请求共用同一段长前缀、各带一小截新问题。上面是轮询：三个请求撒到三个副本，每个副本都要把前缀重算一遍。"
               "下面是 KV 感知路由：三个请求都送到已经存着这段前缀的副本 2，只算各自新的那一截。"
               "右边是 Red Hat 用 llm-d 的实测：对照的最坏情况首字延迟 2,850 毫秒，命中缓存时 340 毫秒，命中率 87.4%；原文只给最坏情况、没给平均，不能读成路由快八倍")
    y0 = f.header("路由把「少算」搬到集群层　——　<tspan font-weight=\"700\">同前缀送同一个副本，前缀只算一次</tspan>",
                  "左是示意；右是 Red Hat 用 llm-d 的公开实测（TTFT 含排队与 prefill）",
                  [(GY2, "共用的长前缀"), (BL, "各自的新问题"), (RD, "前缀要重算"), (GR, "前缀已在 KV 里")])
    PH = 470
    PW1 = 920
    py = f.panel(0, y0, PW1, PH, "轮询 vs KV 感知（示意）", INK)
    RX0, RS, RWD = 330, 140, 120
    for row, (title, smart) in enumerate((("轮询：撒开，三个副本各算一遍前缀", False),
                                          ("KV 感知：都送到存着前缀的副本 2", True))):
        yy = py + 30 + row * 214
        f.t(20, yy, title, RD if not smart else GR, True, 17)
        for r in range(4):
            rx = RX0 + r * RS
            has = (r == 1) if smart else (r < 3)
            col = GR if smart else RD
            f.box(rx, yy + 20, RWD, 130, "none", GY2, 8)
            f.t(rx + RWD / 2, yy + 172, "副本 %d" % (r + 1), GY, True, 15, "middle")
            if has:
                f.box(rx + 10, yy + 34, RWD - 20, 38, "#e6f4ea" if smart else "#fce8e6", col, 5)
                f.t(rx + RWD / 2, yy + 59, "前缀已存" if smart else "重算前缀", col, True, 15, "middle")
        for q in range(3):
            qy = yy + 34 + q * 42
            f.box(20, qy, 150, 32, "#e8eaed", GY2, 5)
            f.t(85, qy + 22, "长前缀", GY, True, 15, "middle")
            f.box(170, qy, 60, 32, BL, BL, 5)
            f.t(200, qy + 22, "问 %d" % (q + 1), "#fff", True, 15, "middle")
            tx = RX0 + (1 if smart else q) * RS
            f.line(234, qy + 16, tx - 4, yy + 92 + (q - 1) * 14, BL, 1.5)
    f._pan = None
    X2 = PW1 + 20
    PW2 = W - X2
    py2 = f.panel(X2, y0, PW2, PH, "Red Hat · llm-d 实测首字延迟", GR)
    AX, AY, AH = X2 + 70, py2 + 60, 260
    for k, (v, col, lab, sub) in enumerate(((TTFT_COLD, RD, "最坏情况", "不路由"), (TTFT_HIT, GR, "命中缓存", "路由后"))):
        bx = AX + k * 170
        bh = AH * v / 3000
        f.box(bx, AY + AH - bh, 110, bh, col, col, 3)
        f.t(bx + 55, AY + AH - bh - 10, format(v, ",") + " ms", col, True, 18, "middle")
        f.t(bx + 55, AY + AH + 26, lab, INK, True, 16, "middle")
        f.t(bx + 55, AY + AH + 46, sub, GY, False, 13, "middle")
    f.line(AX - 10, AY + AH, AX + 300, AY + AH, GY2, 1.2, arrow=False)
    f.t(X2 + 20, py2 + PH - 68, "命中率 %.1f%%" % HIT, GR, True, 17)
    f.t(X2 + 20, py2 + PH - 40, "红柱是最坏情况：原文没给平均，别读成快八倍", RD, True, 14)
    f._pan = None
    yb = f.src(y0 + PH + 6,
               "📌 Red Hat Developer，2025-10-07，「Master KV cache aware routing with llm-d」：warm cache hit TTFT 340 ms，"
               "对照为无 KV 感知路由的最坏情况 2,850 ms；命中率 87.4%（4,776 次查询中 4,176 次）。⚠️ 2,850 是最坏情况，不是轮询的平均值。",
               "⚠️ 本课示意：左栏的副本与请求是示意。",
               "正文里的健康检查那句（8 个副本坏 1 个，1/8 的请求白等）是本课推导示意：12.5% ＝ 1 ÷ 8，只在纯轮询、坏副本照常收请求时成立。")
    f.save("fig6-s13-route.svg", yb + 14)


# ══ ④ pkill -9 漏显存（SGLANG-V4PRO-RUNBOOK.md:960-978，单位照 runbook 写 GB ＝ nvidia-smi MiB ÷ 1000）══
LEAK_TOT, LEAK_LOST, LEAK_NEED = 284, 97, 241     # 每卡总量、强杀后残留、mem-fraction-static 0.85 的静态预留
LEAK_FREE = LEAK_TOT - LEAK_LOST
assert LEAK_FREE == 187                           # runbook 原文「只剩 187 GB」
assert round(LEAK_TOT * 0.85) == LEAK_NEED        # runbook：284 × 0.85 ≈ 241
LEAK_GAP = LEAK_NEED - LEAK_FREE                  # 本课推导
assert LEAK_GAP == 54


def fig_leak():
    f = Fig(W, "GB300 每卡显存 284 GB。强杀一个满载的 decode 之后，每卡还有 97 GB 没释放，runbook 推测是跨机器 NVLink 那部分；"
               "能用的只剩 187 GB。SGLang 的 mem-fraction-static 0.85 要静态占 241 GB，差 54 GB，下次启动必然 OOM。"
               "报错却劝你调大这个比例，那等于把要占的线往右挪，差得更多。删 pod 重建，56 秒后 284 GB 全都能用，241 落在里面，起得来")
    y0 = f.header("pkill -9 漏掉的显存　——　<tspan font-weight=\"700\">97 GB 没还，241 GB 的预留永远够不着</tspan>",
                  "GB300 每卡显存（nvidia-smi，单位 GB，照 runbook 写）；虚线 ＝ mem-fraction-static 0.85 要静态占的 241",
                  [(GR, "还能用"), (RD, "没释放（runbook 推测是跨机 NVLink，即 MNNVL）"), (INK, "静态预留要占到这里")])
    PH = 378
    py = f.panel(0, y0, W, PH, "DeepSeek-V4-Pro · SGLang · GB300 · 每卡", INK)
    X0, AW = 250, 1000
    sc = AW / LEAK_TOT

    def X(v):
        return X0 + v * sc
    BH = 46
    y1 = py + 70
    y2 = y1 + BH + 124
    # 上条：pkill -9 之后
    f.t(24, y1 + 20, "pkill -9 之后", INK, True, 17)
    f.t(24, y1 + 42, "强杀一个满载的 decode", GY, size=14)
    f.box(X0, y1, LEAK_FREE * sc, BH, GR, GR, 3)
    f.t(X0 + 16, y1 + 30, "%d　还能用" % LEAK_FREE, "#fff", True, 16)
    f.box(X(LEAK_FREE), y1, LEAK_LOST * sc, BH, "#fce8e6", RD, 3, 1.5)
    f.t(X(LEAK_TOT) - 12, y1 + 30, "%d　没释放" % LEAK_LOST, RD, True, 16, "end")
    # 差 54 的括号：187 → 241
    xa, xb = X(LEAK_FREE), X(LEAK_NEED)
    yk = y1 + BH + 14
    f.path("M%.1f,%d L%.1f,%d L%.1f,%d L%.1f,%d" % (xa, yk - 6, xa, yk, xb, yk, xb, yk - 6), RD, 2, arrow=False)
    f.t(xb - 10, yk + 24, "差 %d → 必然显存不足（OOM）" % LEAK_GAP, RD, True, 16, "end")
    f.t(xb - 10, yk + 44, "推导：%d − %d" % (LEAK_NEED, LEAK_FREE), RD, size=13, anchor="end")
    # 报错气泡：劝你调大 ＝ 线往右挪
    bx, bw, by_, bh = X0, 520, yk + 4, 62
    f.box(bx, by_, bw, bh, "#fff", RD, 8, 1.4)
    f.t(bx + 14, by_ + 25, "报错：请调大 mem-fraction-static", RD, True, 16)
    f.t(bx + 14, by_ + 49, "调大 ＝ 虚线往右挪 →，差得更多；方向正好反了", GY, size=14)
    f.line(xb + 8, y1 - 14, xb + 70, y1 - 14, RD, 2)
    f.t(xb + 76, y1 - 9, "调大", RD, True, 14)
    # 下条：删 pod 重建
    f.t(24, y2 + 20, "删 pod 重建", INK, True, 17)
    f.t(24, y2 + 42, "56 秒后显存归零", GY, size=14)
    f.box(X0, y2, LEAK_TOT * sc, BH, GR, GR, 3)
    f.t(X0 + 16, y2 + 30, "%d　都能用" % LEAK_TOT, "#fff", True, 16)
    f.t(X(LEAK_TOT) + 12, y2 + 30, "起得来 ✓", GR, True, 16)
    # 241 虚线：贯穿两条
    f.line(xb, y1 - 34, xb, y2 + BH + 10, INK, 1.8, dash="6,4", arrow=False)
    f.t(xb - 8, y1 - 22, "mem-fraction-static 0.85 要静态占 %d" % LEAK_NEED, INK, True, 15, "end")
    # 横轴
    ya = y2 + BH + 34
    f.line(X0, ya - 12, X(LEAK_TOT), ya - 12, GY2, 1.2, arrow=False)
    for v in (0, 50, 100, 150, 200, 250, LEAK_TOT):
        f.line(X(v), ya - 16, X(v), ya - 8, GY2, 1.2, arrow=False)
        f.t(X(v), ya + 8, str(v), GY, size=14, anchor="middle")
    f.t(X(LEAK_TOT) + 30, ya + 8, "GB", GY, size=14)
    f._pan = None
    yb = f.src(py + PH + 10,
               "📌 gpu/inference/a4x-max/deepseek-v4/SGLANG-V4PRO-RUNBOOK.md:960-978：pkill -9 后 nvidia-smi 每卡 97,486 MiB／284,208 MiB；"
               "「284 GB 减掉 97 GB 只剩 187 GB，而 mem-fraction-static 0.85 要 241 GB」；报错原文 Not enough memory. Please try to increase --mem-fraction-static.；"
               "推测原因 MNNVL/IMEX 导出的显存在 SIGKILL 时不走正常回收、容器 PID 1 不 reap；删 pod 到显存归零实测 56 秒。",
               "单位照 runbook 写 GB（MiB ÷ 1000，三个数同一把尺）。差 54 ＝ 241 − 187，本课推导。")
    f.save("fig6-mem-leak.svg", yb + 14)


if __name__ == "__main__":
    fig_bucket()
    fig_cold()
    fig_route()
    fig_leak()
