# -*- coding: utf-8 -*-
r"""专题六 · 第十三节「落地：TPU 服务化与编排」的三张静态图。

回答的问题
  · fig6-s13-bucket  服务侧怎么选桶？桶越多 padding 越少，但每个桶都要编译一次，这笔账怎么权衡？
  · fig6-s13-cold    TPU 上一个引擎从启动到能接请求，时间花在哪？为什么「权重加载」那一段常常其实是编译？
  · fig6-s13-route   路由层能做什么？KV 感知路由把首字延迟从多少降到多少？坏副本不摘掉会怎样？

数从哪来
  · fig6-s13-bucket：本课推导示意。请求长度取对数正态（中位 700 token、σ ＝ 0.9、截到 8,192），
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
from topic03_draw import Fig, BL, OR, GR, RD, PU, GY, INK, GY2, LINE

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
def fig_bucket():
    f = Fig(W, "左边是一份示意的请求长度分布，中位数 700 个 token，长尾拖到 8K，竖线是 2 的幂的桶边界，"
               "每个请求要补零补到它所在桶的上沿。右边比较六种分桶：只有一个 8K 桶时，算的 token 里 87.3% 是补出来的零；"
               "4 个桶 39.1%；2 的幂 7 个桶 29.9%；根号 2 倍一档 13 个桶 16%；每 512 一档 16 个桶反而有 19.7%；每 256 一档 32 个桶 10.8%。"
               "每多一个桶就多一次编译，Qwen3.5-397B 上一个新形状要编译 60 到 90 秒")
    y0 = f.header("桶越细补零越少，编译越多　——　<tspan font-weight=\"700\">按长度分布的形状切，不按等距切</tspan>",
                  "本课推导示意：请求长度按对数正态生成（中位 700 token，长尾到 8K），不是任何真实流量。补零浪费 ＝ 补出来的 token ÷ 实际算的 token",
                  [(BL, "2 的幂"), (GR, "√2 倍一档"), (OR, "等距"), (GY2, "太粗")])
    PH = 440
    LW = 700
    py = f.panel(0, y0, LW, PH, "请求长度分布和 2 的幂桶边界", BL)
    X0, AW, AY, AH = 70, LW - 110, py + 30, PH - 120
    # 对数横轴 64 → 8192
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
        f.t(px(b), AY + AH + 20, "%dK" % (b // 1024) if b >= 1024 else str(b), BL, True, 12, "middle")
    f.t(px(700), AY + 14, "中位 700", INK, True, 13, "middle")
    f.t(X0 + AW / 2, AY + AH + 46, "请求长度（token，对数刻度）；虚线 ＝ 桶上沿，请求补零到上沿", GY, size=12.5, anchor="middle")
    f._pan = None

    RX, RW = LW + 20, W - LW - 20
    py2 = f.panel(RX, y0, RW, PH, "六种分法：桶数（＝ 编译次数）对补零浪费", INK)
    for i, (nm, n, w, col) in enumerate(WASTE):
        yy = py2 + 24 + i * 52
        f.t(RX + 16, yy + 16, nm, INK, True, 13.5)
        f.t(RX + 16, yy + 36, "%d 个桶" % n, GY, size=12)
        bx = RX + 200
        bw = (RW - 300) * w / 90
        f.box(bx, yy + 6, bw, 26, col, col, 3)
        f.t(bx + bw + 8, yy + 25, "%.1f%%" % w, col, True, 14)
    f.t(RX + 16, py2 + PH - 66, "每 512 一档用了 16 个桶，还不如 √2 的 13 个：", INK, True, 13)
    f.t(RX + 16, py2 + PH - 44, "长度分布是对数形状，等距的桶在短请求那头太粗", GY, size=13)
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "桶数就是编译次数，所以桶要在开服前编好", [
        "没预编译的桶，第一个撞上它的请求要等编译：Qwen3.5-397B 上是 60–90 s，之后同形状不到 1 s。",
        "SGLang-JAX 把 batch 档和 token 档写进启动参数预编译；两维的档数相乘才是总编译次数。",
    ])
    yb = f.src(yb + 10,
               "⚠️ 本课推导示意：2,000 个请求长度取对数正态分位点（中位 700、σ ＝ 0.9、截到 8,192），确定性生成；√2 档取 128 × 2^(k/2) 并取整到 16 的倍数。",
               "📌 新形状首个请求编译 60–90 s、之后 <1 s：tpu/tpu-inference/Qwen3.5-397B-A17B-FP8/README.md:794。",
               "📌 预编译参数 --precompile-bs-paddings 1 4 8 16 32 64 128 256、--precompile-token-paddings 4096：tpu/sglang-jax/mimo/README.md:269-270。")
    f.save("fig6-s13-bucket.svg", yb + 14)


# ─────────────────────────────────────────────────────────────────────────
def fig_cold():
    f = Fig(W, "三条 TPU 上的冷启动拆账。GLM-5.1 在 v7x-8 上未优化时约 10 分 44 秒，其中 6 分 29 秒名义上是加载非 MoE 权重，"
               "实际是每个 tensor 都在重编译。Qwen3-Coder 两台机器编进一个 ICI 切片第一次启动约 40 分钟，其中 XLA 编译 29 分钟。"
               "MiMo-V2.5-Pro 在两台 v7x-8 上约 50 分钟，权重加载 25、反量化 5、编译 15、decode 预编译 7。"
               "右边是缓存的作用：R1 去掉一行清缓存的代码，非 MoE 加载从 215.7 秒到 21.4 秒；Kimi K2.6 有本地内存盘缓存时从 57 分钟到 6 分 8 秒")
    y0 = f.header("TPU 冷启动的大头常常是编译　——　<tspan font-weight=\"700\">连「权重加载」那一段也可能是</tspan>",
                  "tpu-inference 公开记录。横轴：分钟。三根条的拓扑、精度、优化程度都不同，只看构成，不比快慢",
                  [(BL, "权重加载"), (PU, "名义加载，实为重编译"), (RD, "XLA 编译"), (OR, "反量化 ／ 预编译 ／ warmup"), (GY2, "初始化")])
    PH = 350
    LW = 900
    py = f.panel(0, y0, LW, PH, "从启动到能接请求，分钟都去哪了", INK)
    X0, AW = 40, LW - 80
    XMAX = 55
    sx = AW / XMAX
    for k, (nm, note, parts, stated) in enumerate(BARS):
        yy = py + 30 + k * 92
        f.t(X0, yy + 4, nm, INK, True, 14)
        f.t(X0 + AW, yy + 4, note, GY, size=12.5, anchor="end")
        x = X0
        for cat, v in parts:
            lab, col = CAT[cat]
            w = v * sx
            f.box(x, yy + 16, w, 34, col, "#fff", 2, 1)
            if w > 34:
                f.t(x + w / 2, yy + 38, ("%.1f" % v).rstrip("0").rstrip(".") if v < 10 else "%d" % round(v),
                    "#fff" if col not in (LINE,) else GY, True, 13, "middle")
            x += w
    for m in (0, 10, 20, 30, 40, 50):
        f.t(X0 + m * sx, py + PH - 50, str(m), GY, size=12, anchor="middle")
    f.t(X0 + GLM_STATED * sx + 14, py + 30 + 38, "← 紫色那 6.5 分钟：每个 tensor 都在重编译传输程序", PU, True, 13)
    f._pan = None
    RX, RW = LW + 20, W - LW - 20
    py2 = f.panel(RX, y0, RW, PH, "缓存在不在，差一个数量级", GR)
    items = [("R1：权重加载时每个 tensor 清一次缓存", "1,367 个 tensor 只有 25 种形状", "215.7 s → 21.4 s", "10.1×"),
             ("R1：进程重启，内存里的编译缓存丢了", "同一份权重，换个新进程", "1:48 → 3:17–3:51", "冷的才是常态"),
             ("Kimi K2.6：权重缓存放本地内存盘", "放共享存储慢 38 倍", "57 分 → 6 分 8 秒", "9.3×")]
    for k, (a, b, c, d) in enumerate(items):
        yy = py2 + 30 + k * 90
        f.t(RX + 16, yy, a, INK, True, 13.5)
        f.t(RX + 16, yy + 22, b, GY, size=12.5)
        f.t(RX + 16, yy + 50, c, GR, True, 15)
        f.t(RX + RW - 16, yy + 50, d, GR, True, 15, "end")
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "冷启动先拆账：哪段是搬字节，哪段是在编译", [
        "编译那段靠持久化缓存省，搬字节那段靠本地缓存省；两种都省不掉时，就要提前起好备用副本。",
        "JAX 的编译缓存按计算图、jaxlib 版本、XLA 参数、device 数和拓扑做键，换任何一样都要重编。",
    ])
    yb = f.src(yb + 10,
               "📌 GLM-5.1：tpu/tpu-inference/GLM-5.1-754B-FP4/README.md:470-476（未优化列：初始化 1m44s、non-MoE 6m29s、MoE cache 1m51s、server init ＋ warmup 25s，"
               "总计 ~10m44s；分项和是 10m29s，差的 15 s 画成「未拆」）。",
               "📌 Qwen3-Coder：tpu/tpu-inference/Qwen3-Coder-480B/README-TPU-VM.md:1011（加载 ~5 ＋ XLA 编译 ~29 ＋ 初始化 ~6 ≈ 40 min，首次无 XLA cache）。"
               "MiMo-V2.5-Pro：tpu/tpu-inference/MiMo-V2.5-Pro-BF16/README.md:184-188（分项和 52，原表写 ~50）。",
               "📌 R1：tpu/tpu-inference/DeepSeek-R1-671B-FP4/README.md:1202-1223（clear_caches）、:1238-1259（transfer cache 不持久化）。"
               "Kimi K2.6：tpu/tpu-inference/Kimi-K2.6-1T-A32B-INT4/README.md:23、39-40（原表无缓存那行是 60 层，有缓存是 61 层）。",
               "📌 编译缓存的键：JAX 文档「Persistent compilation cache」。")
    f.save("fig6-s13-cold.svg", yb + 14)


# ─────────────────────────────────────────────────────────────────────────
def fig_route():
    f = Fig(W, "左：四个副本，三个请求共用同一段长前缀。轮询把它们撒到不同副本，每个副本都要把前缀重算一遍；"
               "KV 感知路由把它们送到已经存着这段前缀的副本，只算新的那一截。中：Red Hat 用 llm-d 实测，命中缓存时首字延迟 340 毫秒，"
               "对照的最坏情况 2850 毫秒，快 88%，命中率 87.4%。右：8 个副本坏了 1 个，路由不做健康检查的话，轮询时八分之一的请求被送进坏副本白等")
    y0 = f.header("路由把「少算」搬到集群层　——　<tspan font-weight=\"700\">同前缀送同一个副本，首字 2,850 → 340 ms</tspan>",
                  "左、右是示意；中间是 Red Hat 用 llm-d 的公开实测（TTFT 含排队与 prefill）",
                  [(GR, "前缀已在 KV 里"), (RD, "要从头算 ／ 坏副本"), (BL, "请求")])
    PH = 400
    PW1, PW2 = 560, 360
    PW3 = W - PW1 - PW2 - 40
    py = f.panel(0, y0, PW1, PH, "轮询 vs KV 感知（示意）", INK)
    for row, (title, smart) in enumerate((("轮询：撒开，三个副本各算一遍前缀", False),
                                          ("KV 感知：送到存着前缀的那个", True))):
        yy = py + 24 + row * 180
        f.t(20, yy, title, INK, True, 13.5)
        for r in range(4):
            rx = 190 + r * 90
            has = (r == 1) if smart else (r < 3)
            col = GR if (smart and r == 1) else (RD if has else LINE)
            f.box(rx, yy + 26, 74, 110, "none", GY2, 6)
            f.t(rx + 37, yy + 154, "副本 %d" % (r + 1), GY, size=11.5, anchor="middle")
            if has:
                f.box(rx + 8, yy + 36, 58, 30, "#e6f4ea" if smart else "#fce8e6", col, 4)
                f.t(rx + 37, yy + 56, "前缀" if smart else "重算", col, True, 12, "middle")
        for q in range(3):
            qy = yy + 40 + q * 32
            f.box(20, qy, 90, 24, "#e8f0fe", BL, 4)
            f.t(65, qy + 17, "请求 %d" % (q + 1), BL, True, 12, "middle")
            tx = 190 + (1 if smart else q) * 90
            f.line(112, qy + 12, tx - 2, yy + 80 + (q - 1) * 10, BL, 1.3)
    f._pan = None
    X2 = PW1 + 20
    py2 = f.panel(X2, y0, PW2, PH, "Red Hat · llm-d 实测", GR)
    AX, AY, AH = X2 + 70, py2 + 40, PH - 140
    for k, (v, col, lab) in enumerate(((TTFT_COLD, RD, "最坏情况"), (TTFT_HIT, GR, "命中缓存"))):
        bx = AX + k * 120
        bh = AH * v / 3000
        f.box(bx, AY + AH - bh, 80, bh, col, col, 3)
        f.t(bx + 40, AY + AH - bh - 8, format(v, ",") + " ms", col, True, 14, "middle")
        f.t(bx + 40, AY + AH + 20, lab, INK, True, 12.5, "middle")
    f.line(AX - 10, AY + AH, AX + 230, AY + AH, GY2, 1.2, arrow=False)
    f.t(X2 + 20, py2 + PH - 46, "首字快 88%，命中率 87.4%", GR, True, 14)
    f._pan = None
    X3 = X2 + PW2 + 20
    py3 = f.panel(X3, y0, PW3, PH, "不做健康检查（示意）", RD)
    for r in range(NREP):
        cx = X3 + 30 + (r % 4) * 105
        cy = py3 + 30 + (r // 4) * 110
        bad = r == 5
        f.box(cx, cy, 80, 70, "#fce8e6" if bad else "#e6f4ea", RD if bad else GR, 6)
        f.t(cx + 40, cy + 30, "副本 %d" % (r + 1), RD if bad else GR, True, 12.5, "middle")
        f.t(cx + 40, cy + 52, "引擎已崩" if bad else "正常", RD if bad else GY, size=12, anchor="middle")
    f.t(X3 + 20, py3 + 270, "进程还在、端口还通，引擎已经崩了", INK, True, 13.5)
    f.t(X3 + 20, py3 + 296, "轮询照样往里送：1/8 ＝ %.1f%% 的请求白等" % LOST, RD, True, 13.5)
    f.t(X3 + 20, py3 + 322, "健康检查要真发一个推理请求，不只看端口", GY, size=13)
    f._pan = None
    yb = f.band(y0 + PH + 20, "ok", "一个副本内靠前缀缓存少算，一群副本靠路由少算", [
        "前缀缓存只在一台机器里有效；路由不知道谁存着什么，命中率就随副本数摊薄。",
        "llm-d、NVIDIA Dynamo、Gateway API Inference Extension、GKE Inference Gateway 做的都是这件事，外加摘掉坏副本。",
    ])
    yb = f.src(yb + 10,
               "📌 Red Hat Developer，2025-10-07，「Master KV cache aware routing with llm-d」：warm cache hit TTFT 340 ms，"
               "对照为无 KV 感知路由的最坏情况 2,850 ms（88% faster）；命中率 87.4%（4,776 次查询中 4,176 次）。",
               "⚠️ 本课示意：左栏的副本与请求、右栏的 8 坏 1 都是示意；12.5% ＝ 1 ÷ 8，只在纯轮询、坏副本照常收请求时成立。")
    f.save("fig6-s13-route.svg", yb + 14)


if __name__ == "__main__":
    fig_bucket()
    fig_cold()
    fig_route()
