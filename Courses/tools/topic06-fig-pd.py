# -*- coding: utf-8 -*-
r"""专题六 · 第八节「拆工作：PD 分离与那一趟 KV」的四张静态图。

fig6-kv-stack.svg —— 回答：PD 之间那一趟 KV，是哪几层软件在管？GPU 和 TPU 各用什么？
  · 数从哪来：没有数，是分层归类。每个组件的定位取自各自公开文档：
    NIXL（github.com/ai-dynamo/nixl，插件含 UCX、GDS、Mooncake、POSIX……）、
    Mooncake（github.com/kvcache-ai/Mooncake：Transfer Engine 走 RDMA ／ NVLink ／ TCP 等，Store 是多层 KV 存储）、
    LMCache（arXiv 2510.09665：GPU 外的 KV 层，接 vLLM 和 SGLang）、
    Dynamo KVBM（docs.nvidia.com/dynamo 的 KVBM 页：GPU、主机、SSD、远端存储，经 NIXL，接 TensorRT-LLM 和 vLLM）、
    TPU Sync（github.com/google/tpu-sync README：HBM ↔ 主机内存 ↔ 远端 TPU，零拷贝，主机内存当第二层前缀缓存）、
    tpu-inference 的 TPUConnector（用 jax.experimental.transfer 的 transfer server，decode 端去拉）。
  · 刻意没画：各组件之间谁调用谁的箭头。同一个组件能被好几层用（NIXL 也能当 KVBM 的搬运层），
    画箭头会画成一团线；这张图只回答「这一层是谁」。

fig6-kv-bw.svg —— 回答：同一个库配错一个开关，那一趟 KV 会慢到什么程度？
  · 数从哪来：vLLM DeepSeek-V4-Pro runbook（gpu/inference/a4x-max/deepseek-v4/VLLM-V4PRO-RUNBOOK.md:194）实测：
    KV 传输 200 MB/s → 7–167 GB/s，单次传输 200–800 ms → 2–10 ms。
  · 本课推导：拿专题五那份 1.04 GB（Qwen3-Coder-480B，8K prompt，FP8）分别按这几档带宽算一趟要多久，
    跟专题五的 100 Gbps 数据中心网络（83 ms）和 1–2 s 的 prefill 放在同一根对数轴上。
    ⚠️ 1.04 GB 不是 V4-Pro 的 KV，只是一把统一的尺子；图上写明。
  · 刻意没画：带宽为什么在 7 到 167 GB/s 之间浮动（runbook 没给分解，画了就是编）。

fig6-hybrid-kv.svg —— 回答：混合模型（GDN ／ Mamba 层 ＋ 少量注意力层）的 KV 为什么不能照老办法传？
  · 数从哪来：tpu-inference 的 tpu_connector_hma.py 原文注释（D 端拉全部 block、本地零重算，Mamba 正确性要求）；
    非 HMA 版把 prompt 长度按 block 向下取整、尾巴由 D 端重算；
    Qwen3.5-397B 45 层 GDN ＋ 15 层注意力（tpu/tpu-inference/Qwen3.5-397B-A17B-FP8/README.md:375–389），
    与 PrFaaS 论文写的 3:1 线性对全注意力一致（脚本里 assert）。
  · 刻意没画：block 的真实大小和层内布局（示意格子）。

fig6-prfaas.svg —— 回答：PD 能不能跨数据中心？卡在哪个数上？
  · 数从哪来：PrFaaS（arXiv 2604.15039）Table 3：8×H200、SGLang v0.5.9、32K 输入时一台 prefill 每秒要外送的 KV（Φ_kv，Gbps）。
    案例：内部 1T 混合模型，吞吐 +54%、P90 TTFT −64%，跨集群平均出口 13 Gbps。
  · 刻意没画：PrFaaS 的调度细节（按长度阈值外送、带宽感知），正文用一句话交代。
"""
import math

from topic03_draw import Fig, BL, OR, GR, RD, PU, CY, GY, INK, GY2, LINE

W = 1400

# ── 统一的尺子：专题五那份 KV（Qwen3-Coder-480B config：62 层、8 个 KV 头、head_dim 128；8K prompt；FP8 每元素 1 字节）
KV_8K = 2 * 62 * 8 * 128 * 8192
assert abs(KV_8K / 1e9 - 1.04) < 0.01

# ── vLLM V4-Pro runbook 实测带宽（MB/s）
BW_BAD = 200e6                      # 走了 cuda_copy
BW_LO, BW_HI = 7e9, 167e9           # 三件套之后
BW_DCN = 100e9 / 8                  # 专题五：100 Gbps 数据中心网络 ＝ 12.5 GB/s
T = {k: KV_8K / v * 1e3 for k, v in (("bad", BW_BAD), ("lo", BW_LO), ("hi", BW_HI), ("dcn", BW_DCN))}
assert round(T["bad"]) == 5201 and round(T["lo"]) == 149 and round(T["hi"], 1) == 6.2 and round(T["dcn"]) == 83
PREFILL_MS = (1000, 2000)           # 专题五：8K prompt 的 prefill 1–2 s
assert T["bad"] > PREFILL_MS[1] * 2.5          # 配错时，搬一趟比 prefill 本身还长两倍半以上
XFER_BEFORE, XFER_AFTER = (200, 800), (2, 10)  # runbook 实测单次传输（ms）
assert XFER_BEFORE[0] / XFER_AFTER[0] == 100 and XFER_BEFORE[1] / XFER_AFTER[1] == 80   # 单次传输快约两个数量级

# ── Qwen3.5-397B：45 GDN ＋ 15 注意力；PrFaaS 写 3:1
GDN, ATT = 45, 15
assert GDN + ATT == 60 and GDN / ATT == 3

# ── PrFaaS Table 3（32K，Gbps，8×H200，SGLang v0.5.9）
PHI = [("Qwen3-235B", "全注意力", 33.35, OR), ("Qwen3.5-397B", "混合 3:1", 8.25, GR),
       ("MiniMax-M2.5", "全注意力", 59.93, OR), ("MiMo-V2-Flash", "混合 5:1（SWA）", 4.66, GR)]
assert round(PHI[0][2] / PHI[1][2]) == 4 and round(PHI[2][2] / PHI[3][2]) == 13
LINK = 100                          # PrFaaS 案例的跨集群带宽约 100 Gbps
EGRESS = 13                         # 案例里 PrFaaS 集群平均出口
assert EGRESS / LINK == 0.13


def fig_kv_stack():
    f = Fig(W, "PD 之间那一趟 KV，分三层软件在管。左列 GPU，右列 TPU。"
               "第一层管搬运：GPU 上是 NIXL 和 Mooncake Transfer Engine，TPU 上是 TPU Sync 和 JAX 的 transfer server。"
               "第二层管调度：谁决定哪个请求去哪台 prefill、算完什么时候搬；GPU 上是 Dynamo、vllm-router 和引擎里的 connector，"
               "TPU 上是 tpu-inference 的代理和 TPUConnector。第三层管存储：KV 不只在显存里，还能落到主机内存、SSD、远端，"
               "GPU 上是 Mooncake Store、LMCache、Dynamo KVBM，TPU 上是 TPU Sync 的主机内存层")
    y0 = f.header("那一趟 KV 背后有三层软件　——　<tspan font-weight=\"700\">搬运、调度、存储各是一摊</tspan>",
                  "每一格写的是这一层里常见的公开组件。同一个组件可能横跨两层（例如 NIXL 也给 KVBM 当搬运层），这里按它最主要的角色放",
                  [(BL, "GPU 侧"), (CY, "TPU 侧")])
    ROWS = [
        ("搬运", "字节从哪块显存到哪块显存", OR,
         [("NIXL", "NVIDIA 的传输库，插件式：UCX、GDS、Mooncake……"),
          ("Mooncake Transfer Engine", "RDMA、NVLink、TCP 等多种通道，多网卡聚合")],
         [("TPU Sync", "HBM、主机内存、远端 TPU 之间零拷贝搬 buffer"),
          ("JAX transfer server", "tpu-inference 的 TPUConnector 用它：decode 端去拉")]),
        ("调度", "哪个请求去哪台、什么时候搬", PU,
         [("Dynamo ／ vllm-router", "前端路由：请求先进 prefill，再交给 decode"),
          ("引擎里的 connector", "vLLM 的 NixlConnector；SGLang 的 transfer backend")],
         [("tpu-inference 的代理", "先把请求以 max_tokens=1 发给 prefill"),
          ("TPUConnector ／ TPUConnectorHMA", "后者给混合模型用，按层组分开传")]),
        ("存储", "KV 不只放显存：主机内存、SSD、远端", GR,
         [("Mooncake Store", "跨实例复用前缀，多层缓存"),
          ("LMCache ／ Dynamo KVBM", "GPU 外的 KV 层；KVBM 管到 SSD 和远端存储")],
         [("TPU Sync 的主机内存层", "主机内存当第二层前缀缓存，放 /dev/shm"),
          ("", "服务重启后 KV 还在")]),
    ]
    LX, LW = 0, 250
    GX, CW = 270, 555
    TX = GX + CW + 20
    f.t(GX + CW / 2, y0 + 16, "GPU", BL, True, 17, "middle")
    f.t(TX + CW / 2, y0 + 16, "TPU", CY, True, 17, "middle")
    y = y0 + 34
    RH = 150
    for lab, sub, col, g, t in ROWS:
        f.box(LX, y, LW, RH - 14, "none", LINE, 9)
        f.box(LX, y, 4, RH - 14, col, col, 2)
        f.t(LX + 20, y + 44, lab, col, True, 22)
        f.t(LX + 20, y + 74, sub, GY, size=13)
        for x0, items, c2 in ((GX, g, BL), (TX, t, CY)):
            f.box(x0, y, CW, RH - 14, "none", LINE, 9)
            yy = y + 30
            for name, desc in items:
                if name:
                    f.t(x0 + 18, yy, name, c2, True, 16)
                    yy += 24
                f.t(x0 + 18, yy, desc, INK, size=13.5)
                yy += 34
        y += RH
    f._pan = None
    yb = f.band(y + 6, "ok", "先认清一个故障出在哪一层", [
        "带宽掉到几百 MB/s 是搬运层没走对通道；请求接不上、握手失败多半在调度层；前缀命中率低看存储层。",
        "TPU 这一列组件少得多：TPU Sync 的 README 自己写着还不建议通用场景使用。",
    ])
    yb = f.src(yb + 10,
               "📌 NIXL：github.com/ai-dynamo/nixl（插件 UCX、GDS、Mooncake、POSIX 等）。Mooncake：github.com/kvcache-ai/Mooncake（Transfer Engine ＋ Store）。",
               "📌 LMCache：arXiv 2510.09665。Dynamo KVBM：docs.nvidia.com/dynamo 的 KVBM 页（GPU、主机、SSD、远端存储，经 NIXL）。",
               "📌 TPU Sync：github.com/google/tpu-sync（C++，零拷贝 PJRT buffer，约 200 GB/s 双向 PCIe，支持 v7x ／ v6e，not yet recommended for general use）。",
               "📌 tpu-inference：vllm-project/tpu-inference 的 kv_transfer 目录（TPUConnector 用 jax.experimental.transfer；HMA 版按 MambaSpec 分组）。")
    f.save("fig6-kv-stack.svg", yb + 14)


def fig_kv_bw():
    f = Fig(W, "同一份 1.04 GB 的 KV，按四档带宽各搬一趟要多久，画在对数时间轴上。"
               "200 MB/s 是 vLLM 在 GB300 上配错时的实测带宽，搬一趟约 5.2 秒，比 1 到 2 秒的 prefill 还长。"
               "专题五那条 100 Gbps 数据中心网络约 83 毫秒。按对三个开关之后实测 7 到 167 GB/s，一趟只要 149 到 6 毫秒。"
               "runbook 里单次传输从 200 到 800 毫秒降到 2 到 10 毫秒")
    y0 = f.header("同一个库，配错一个开关，搬一趟比 prefill 还慢　——　<tspan font-weight=\"700\">慢了两个数量级</tspan>",
                  "横轴是对数时间。每根条是同一份 1.04 GB 的 KV（专题五那个 8K prompt）按这一档带宽搬一趟；灰带是这个 prompt 的 prefill 本身",
                  [(RD, "配错：实测 200 MB/s"), (GY2, "专题五：100 Gbps 网络"), (GR, "配对：实测 7–167 GB/s")])
    PH = 400
    py = f.panel(0, y0, W, PH, "搬一趟 1.04 GB 要多久（本课推导，带宽是实测）", OR)
    BX, BW = 330, 960
    LO, HI = 1, 10000                 # 1 ms … 10 s

    def X(ms):
        return BX + BW * (math.log10(ms) - math.log10(LO)) / (math.log10(HI) - math.log10(LO))
    AX = py + PH - 78
    for v, s in ((1, "1 ms"), (10, "10 ms"), (100, "100 ms"), (1000, "1 s"), (10000, "10 s")):
        f.line(X(v), py + 40, X(v), AX, LINE, 1, arrow=False)
        f.t(X(v), AX + 22, s, GY, size=13, anchor="middle")
    f.box(X(PREFILL_MS[0]), py + 40, X(PREFILL_MS[1]) - X(PREFILL_MS[0]), AX - py - 40, "#e8eaed", "none", 0)
    f.t(X(1414), py + 60, "prefill 1–2 s", GY, True, 13.5, "middle")
    bars = [("配错（cuda_copy）", "200 MB/s", T["bad"], RD, "%s ms" % "{:,}".format(round(T["bad"]))),
            ("专题五：数据中心网络", "12.5 GB/s", T["dcn"], GY2, "%.0f ms" % T["dcn"]),
            ("配对，低端", "7 GB/s", T["lo"], GR, "%.0f ms" % T["lo"]),
            ("配对，高端", "167 GB/s", T["hi"], GR, "%.1f ms" % T["hi"])]
    by = py + 84
    for lab, bw, ms, col, txt in bars:
        f.t(BX - 16, by + 4, lab, INK, True, 15, "end")
        f.t(BX - 16, by + 24, bw, GY, size=13, anchor="end")
        f.box(BX, by - 12, X(ms) - BX, 30, col, col, 3)
        f.t(X(ms) + 10, by + 9, txt, col, True, 15)
        by += 62
    f._pan = None
    yb = f.band(py + PH + 20, "bad", "专题五说「网络不是瓶颈」，前提是真跑在设计带宽上", [
        "runbook 实测：单次传输 200–800 ms 降到 2–10 ms。差的只是三样配置：只留 NVLink 通道、允许跨节点 IPC、换 cuMem 分配器。",
        "配错时什么都不报错，只是 TTFT 慢得说不清原因。上线前先量一次 KV 传输带宽。",
    ])
    yb = f.src(yb + 10,
               "📌 带宽和单次传输时间：gpu/inference/a4x-max/deepseek-v4/VLLM-V4PRO-RUNBOOK.md:194（GB300，vLLM，NixlConnector）。",
               "⚠️ 本课推导：1.04 GB ＝ 2 × 62 × 8 × 128 × 8,192 × 1 字节（Qwen3-Coder-480B，FP8），只当一把统一的尺子，不是 V4-Pro 的 KV。"
               "时间 ＝ 1.04 GB ÷ 带宽；没算握手和排队。prefill 1–2 s 与 100 Gbps 取自专题五 §6.2。")
    f.save("fig6-kv-bw.svg", yb + 14)


def fig_hybrid_kv():
    f = Fig(W, "混合模型的 KV 有两种。注意力层的 KV 按 token 一格一格长，切成固定大小的 block，最后一块常常没填满。"
               "GDN 或 Mamba 层只有一份固定大小的状态，是整段 prompt 的递归摘要，每个请求一块。"
               "老办法把 prompt 长度按 block 向下取整，只传整块，尾巴让 decode 端自己重算：对注意力层没问题，"
               "对递归状态就错了，尾巴那几个 token 会被喂进递归两次。混合模型版的 connector 每一块都传，包括没填满的最后一块和那份状态，decode 端一个 token 都不重算")
    y0 = f.header("混合模型的 KV 不能「传整块、尾巴自己算」　——　<tspan font-weight=\"700\">递归状态里尾巴会被算两遍</tspan>",
                  "示意格子，不按真实 block 大小。以 Qwen3.5-397B 为例：60 层里 45 层 GDN（递归状态）、15 层注意力（按 token 长的 KV）",
                  [(BL, "注意力层的 KV block"), (PU, "GDN 层的递归状态"), (RD, "decode 端重算的尾巴")])
    PH = 380
    py = f.panel(0, y0, W, PH, "同一个请求，两种传法", PU)
    CW, X0 = 62, 330
    NB = 7                                      # 6 个满 block ＋ 1 个半满

    def row(y, label, sub, mode):
        f.t(24, y + 24, label, INK, True, 16)
        f.t(24, y + 48, sub, GY, size=13)
        # 注意力
        f.t(X0, y - 6, "注意力层（15 层）", BL, True, 13.5)
        for i in range(NB):
            x = X0 + i * CW
            if i < NB - 1:
                f.box(x, y + 4, CW - 6, 40, BL, BL, 4)
            else:
                f.box(x, y + 4, CW - 6, 40, "none", BL, 4, sw=1.6)
                f.box(x, y + 4, (CW - 6) * 0.45, 40, RD if mode == "old" else BL, RD if mode == "old" else BL, 4)
        # GDN
        gx = X0 + NB * CW + 70
        f.t(gx, y - 6, "GDN 层（45 层）", PU, True, 13.5)
        f.box(gx, y + 4, 120, 40, PU, PU, 4)
        f.t(gx + 60, y + 30, "整段摘要", "#ffffff", True, 13.5, "middle")
        nx = gx + 150
        if mode == "old":
            f.t(nx, y + 18, "只传 6 个整块；红色尾巴在 decode 端重算", RD, True, 14.5)
            f.t(nx, y + 42, "注意力层：重算没问题", GR, size=14)
            f.t(nx, y + 64, "GDN 层：尾巴已经在摘要里，再喂一遍就错了", RD, True, 14.5)
        else:
            f.t(nx, y + 18, "7 块全传，包括没填满的最后一块", GR, True, 14.5)
            f.t(nx, y + 42, "GDN 状态每个请求一块，跟着一起传", GR, size=14)
            f.t(nx, y + 64, "decode 端零重算", GR, True, 14.5)
    row(py + 60, "老办法", "非 HMA 的 connector", "old")
    f.line(20, py + 200, W - 20, py + 200, LINE, 1, arrow=False)
    row(py + 250, "混合模型版", "TPUConnectorHMA", "new")
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "混合模型要换 connector，还要打开一个默认关着的开关", [
        "tpu-inference 跑 Qwen3.5 PD 要用 TPUConnectorHMA，并加 --no-disable-hybrid-kv-cache-manager；漏了它，两种层合不到一个池里，启动就报错。",
        "GPU 上同理：SGLang 跑 Kimi K3 时，PD 的 decode 端每个请求只有一个状态槽，原来的前缀缓存策略在 PD 下不起作用。",
    ])
    yb = f.src(yb + 10,
               "📌 tpu-inference tpu_connector_hma.py 注释原文：No trim, no block-alignment rounding … Required for Mamba correctness. "
               "P's Mamba state is a recurrent summary of all prompt tokens. If D re-prefilled any tail tokens locally, they would be fed through the Mamba recurrence a second time on D.",
               "📌 45 GDN ＋ 15 注意力与开关：tpu/tpu-inference/Qwen3.5-397B-A17B-FP8/README.md:375–389；PrFaaS（arXiv 2604.15039）写 Qwen3.5-397B 线性对全注意力 3:1，一致。",
               "📌 Kimi K3（GPU）：gpu/inference/kimi-k3/SGLANG-K3-RUNBOOK.md:1319（PD decode 的状态池一请求一 slot，--mamba-radix-cache-strategy 在 PD 下失效）。")
    f.save("fig6-hybrid-kv.svg", yb + 14)


def fig_prfaas():
    f = Fig(W, "四个模型在 32K 输入时，一台 8 卡 H200 做 prefill，每秒要往外送多少 KV。"
               "全注意力的 Qwen3-235B 要 33.35 Gbps，混合的 Qwen3.5-397B 只要 8.25，差 4 倍。"
               "全注意力的 MiniMax-M2.5 要 59.93，混合的 MiMo-V2-Flash 只要 4.66，差 13 倍。"
               "跨数据中心的专线按 100 Gbps 画一条线。PrFaaS 的案例里，prefill 集群平均出口只有 13 Gbps")
    y0 = f.header("KV 变薄，PD 才敢跨数据中心　——　<tspan font-weight=\"700\">混合模型要送的字节少 4 到 13 倍</tspan>",
                  "PrFaaS 论文 Table 3：32K 输入，一台 8×H200（SGLang v0.5.9）做 prefill 时每秒要外送的 KV，单位 Gbps",
                  [(OR, "全注意力"), (GR, "混合（线性或滑窗 ＋ 少量全注意力）")])
    PH = 390
    py = f.panel(0, y0, W, PH, "每台 prefill 每秒要外送的 KV（Gbps，32K 输入）", GR)
    BX, BW = 330, 900
    VMAX = 110.0

    def X(v):
        return BX + BW * v / VMAX
    AX = py + PH - 72
    for v in (0, 20, 40, 60, 80, 100):
        f.line(X(v), py + 40, X(v), AX, LINE, 1, arrow=False)
        f.t(X(v), AX + 22, str(v), GY, size=13, anchor="middle")
    by = py + 70
    for i, (name, kind, v, col) in enumerate(PHI):
        f.t(BX - 16, by + 4, name, INK, True, 15, "end")
        f.t(BX - 16, by + 24, kind, GY, size=13, anchor="end")
        f.box(BX, by - 12, X(v) - BX, 30, col, col, 3)
        f.t(X(v) + 10, by + 9, "%.2f" % v, col, True, 15)
        by += 64 if i % 2 else 56
    f.path("M%.1f,%d L%.1f,%d" % (X(LINK), py + 40, X(LINK), AX), RD, 2, dash="5,4", arrow=False)
    f.t(X(LINK) - 8, py + 58, "跨数据中心专线约 %d Gbps" % LINK, RD, True, 14, "end")
    f._pan = None
    yb = f.band(py + PH + 20, "ok", "PrFaaS：只把长 prompt 送去远处的 prefill 集群", [
        "内部 1T 混合模型：32 张 H200 做 prefill ＋ 64 张 H20 做本地 PD，对比 96 张 H20 同构 PD，吞吐 +54%，P90 TTFT −64%。",
        "平均出口只有 %d Gbps，约占专线的 %d%%。光靠 KV 变薄不够，还要按长度和带宽挑哪些请求外送。" % (EGRESS, round(EGRESS / LINK * 100)),
    ])
    yb = f.src(yb + 10,
               "📌 PrFaaS，arXiv 2604.15039（Moonshot AI ＋ 清华）：Table 3 的 Φ_kv；案例吞吐 +54%、P90 TTFT −64%、同成本约 +15%、比不调度的朴素异构 +32%；"
               "长度阈值 19.4K，约一半请求外送；平均出口 13 Gbps。",
               "⚠️ 那 4 倍、13 倍是两两同代对比（论文原文），不是同一参数量下的严格对照。")
    f.save("fig6-prfaas.svg", yb + 14)


if __name__ == "__main__":
    fig_kv_stack()
    fig_kv_bw()
    fig_hybrid_kv()
    fig_prfaas()
