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

from topic03_draw import Fig, wpx, BL, OR, GR, RD, PU, CY, GY, INK, GY2, LINE

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
# 标题里的倍数：对 prefill 2.6–5.2 倍；对配对两档 35–835 倍（＝ 带宽之比 7/0.2、167/0.2）
assert "%.1f" % (T["bad"] / PREFILL_MS[1]) == "2.6" and "%.1f" % (T["bad"] / PREFILL_MS[0]) == "5.2"
assert round(T["bad"] / T["lo"]) == 35 and round(T["bad"] / T["hi"]) == 835
XFER_BEFORE, XFER_AFTER = (200, 800), (2, 10)  # runbook 实测单次传输（ms）
assert XFER_BEFORE[0] / XFER_AFTER[0] == 100 and XFER_BEFORE[1] / XFER_AFTER[1] == 80   # 单次传输快约两个数量级

# ── Qwen3.5-397B：45 GDN ＋ 15 注意力；PrFaaS 写 3:1
GDN, ATT = 45, 15
assert GDN + ATT == 60 and GDN / ATT == 3

# ── PrFaaS Table 3（32K，Gbps，8×H200，SGLang v0.5.9）
PHI = [("Qwen3-235B", "全注意力", 33.35, OR), ("Qwen3.5-397B", "每 4 层里 3 层线性", 8.25, GR),
       ("MiniMax-M2.5", "全注意力", 59.93, OR), ("MiMo-V2-Flash", "每 6 层里 5 层只看近处", 4.66, GR)]
assert round(PHI[0][2] / PHI[1][2]) == 4 and round(PHI[2][2] / PHI[3][2]) == 13
LINK = 100                          # PrFaaS 案例整条跨集群以太网（13 Gbps ＝ 13%，论文 §4.3.1）；整个 prefill 集群共用，不是单机网口
EGRESS = 13                         # 案例里 PrFaaS 集群平均出口
assert EGRESS / LINK == 0.13


def fig_kv_stack():
    f = Fig(W, "把那一趟 KV 画成一次搬家：左边 prefill 机，右边 decode 机，中间一条路。"
               "路和车是搬运层，GPU 上是 NIXL、Mooncake Transfer Engine，TPU 上是 TPU Sync 和 JAX 的 transfer server。"
               "路口的调度员是调度层，决定请求先去哪台 prefill、算完交给哪台 decode：GPU 上是 Dynamo、vllm-router 和引擎里的 connector，"
               "TPU 上是 tpu-inference 的代理和 TPUConnector。路边的仓库是存储层，KV 可以先存到主机内存、SSD 或远端："
               "GPU 上是 Mooncake Store、LMCache、Dynamo KVBM，TPU 上是 TPU Sync 的主机内存层")
    y0 = f.header("那一趟 KV 像一次搬家　——　<tspan font-weight=\"700\">路和车、路口调度、路边仓库各是一层</tspan>",
                  "小字是各层常见的公开组件：蓝字 GPU，青字 TPU",
                  [(OR, "搬运层"), (GY, "调度层"), (GR, "存储层"), (PU, "车上运的 KV")])
    PH = 440
    py = f.panel(0, y0, W, PH, "prefill 机 → decode 机", OR)
    RD_W = 48                                   # 路宽
    ry = py + 190                               # 路的上沿
    RX0, RX1 = 50 + 170 * 0.84, W - 50 - 170 * 0.84
    JX = 520                                    # 路口（调度员）
    WX = 930                                    # 去仓库的岔路
    ROAD = "#e8eaed"
    # 路：主路 ＋ 路口往上的进口 ＋ 往下去仓库的岔路
    f.box(RX0, ry, RX1 - RX0, RD_W, ROAD, "none", 0)
    f.box(JX - RD_W / 2, py + 20, RD_W, ry - py - 20, ROAD, "none", 0)
    f.box(WX - RD_W / 2, ry + RD_W, RD_W, 40, ROAD, "none", 0)
    f.path("M%d,%d L%d,%d" % (RX0 + 20, ry + RD_W / 2, RX1 - 20, ry + RD_W / 2), "#ffffff", 2.5, dash="14,10", arrow=False)
    # 两台机器
    f.icon("house", 50, ry - 110, 170, 170, INK, label="prefill 机", lsize=17)
    f.icon("house", W - 50 - 170, ry - 110, 170, 170, INK, label="decode 机", lsize=17)
    # 车：两辆，往右开
    for cx in (300, 700):
        f.icon("cart", cx, ry - 4, 76, 52, OR, "#fef7e0")
        f.box(cx + 14, ry + 6, 30, 18, PU, PU, 2)
    f.line(800, ry + RD_W / 2, 880, ry + RD_W / 2, OR, 2.5)
    # 搬运层标签：路下方左段
    f.t(260, ry + RD_W + 44, "搬运层 ＝ 路和车", OR, True, 18)
    f.t(260, ry + RD_W + 74, "NIXL · Mooncake Transfer Engine", BL, True, 15)
    f.t(260, ry + RD_W + 100, "TPU Sync · JAX transfer server", CY, True, 15)
    # 调度员：路口旁
    f.t(JX - RD_W / 2 - 12, py + 50, "新请求", GY, True, 15, "end")
    f.line(JX, py + 26, JX, py + 92, GY2, 2)
    f.icon("person", JX + RD_W / 2 + 14, ry - 74, 56, 64, GY, "#f1f3f4")
    f.t(JX + 110, py + 40, "调度层 ＝ 路口调度员", GY, True, 18)
    f.t(JX + 110, py + 70, "Dynamo · vllm-router · 引擎里的 connector", BL, True, 15)
    f.t(JX + 110, py + 96, "tpu-inference 的代理 · TPUConnector", CY, True, 15)
    # 仓库：岔路尽头
    f.icon("shelf", WX - 50, ry + RD_W + 40, 100, 100, GR, "#e6f4ea")
    f.t(WX + 74, ry + RD_W + 74, "存储层 ＝ 路边仓库", GR, True, 18)
    f.t(WX + 74, ry + RD_W + 104, "Mooncake Store · LMCache · Dynamo KVBM", BL, True, 15)
    f.t(WX + 74, ry + RD_W + 130, "TPU Sync 的主机内存层", CY, True, 15)
    f.t(WX + 74, ry + RD_W + 156, "放主机内存、SSD 或远端", GY, size=15)
    f._pan = None
    yb = f.src(py + PH + 16,
               "📌 NIXL：github.com/ai-dynamo/nixl（插件 UCX、GDS、Mooncake、POSIX 等）。Mooncake：github.com/kvcache-ai/Mooncake（Transfer Engine ＋ Store）。",
               "📌 LMCache：arXiv 2510.09665。Dynamo KVBM：docs.nvidia.com/dynamo 的 KVBM 页（GPU、主机、SSD、远端存储，经 NIXL）。",
               "📌 TPU Sync：github.com/google/tpu-sync（C++，零拷贝 PJRT buffer，约 200 GB/s 双向 PCIe，支持 v7x ／ v6e，not yet recommended for general use）。",
               "📌 tpu-inference：vllm-project/tpu-inference 的 kv_transfer 目录（TPUConnector 用 jax.experimental.transfer，decode 端去拉；HMA 版按 MambaSpec 分组）。",
               "⚠️ 同一个组件可能横跨两层（例如 NIXL 也给 KVBM 当搬运层），图上按它最主要的角色放；路和车只是比方，不代表真实拓扑。")
    f.save("fig6-kv-stack.svg", yb + 14)


def fig_kv_bw():
    f = Fig(W, "同一份 1.04 GB 的 KV，按四档带宽各搬一趟要多久，画在对数时间轴上。"
               "200 MB/s 是 vLLM 在 GB300 上配错时的实测带宽，搬一趟约 5.2 秒，比 1 到 2 秒的 prefill 还长。"
               "专题五那条 100 Gbps 数据中心网络约 83 毫秒。按对三个开关之后实测 7 到 167 GB/s，一趟只要 149 到 6 毫秒。"
               "runbook 里单次传输从 200 到 800 毫秒降到 2 到 10 毫秒")
    y0 = f.header("配错一个开关，搬一趟比 prefill 本身还慢 2.6–5.2 倍　——　<tspan font-weight=\"700\">比配对慢 35–835 倍</tspan>",
                  "横轴是对数时间。每根条是同一份 1.04 GB 的 KV（专题五那个 8K prompt）按这一档带宽搬一趟；灰带是这个 prompt 的 prefill 本身",
                  [(RD, "配错：实测 200 MB/s"), (GY2, "专题五：100 Gbps ＝ 12.5 GB/s"), (GR, "配对：实测 7–167 GB/s")])
    PH = 400
    py = f.panel(0, y0, W, PH, "搬一趟 1.04 GB 要多久（本课推导，带宽是实测）", OR)
    BX, BW = 330, 960
    LO, HI = 1, 10000                 # 1 ms … 10 s

    def X(ms):
        return BX + BW * (math.log10(ms) - math.log10(LO)) / (math.log10(HI) - math.log10(LO))
    AX = py + PH - 78
    for v, s in ((1, "1 ms"), (10, "10 ms"), (100, "100 ms"), (1000, "1 s"), (10000, "10 s")):
        f.line(X(v), py + 40, X(v), AX, LINE, 1, arrow=False)
        f.t(X(v), AX + 24, s, GY, size=15, anchor="middle")
    f.box(X(PREFILL_MS[0]), py + 40, X(PREFILL_MS[1]) - X(PREFILL_MS[0]), AX - py - 40, "#e8eaed", "none", 0)
    f.t(X(1414), py + 62, "prefill 1–2 s", GY, True, 15, "middle")
    bars = [("配错（cuda_copy）", "200 MB/s", T["bad"], RD, "%s ms" % "{:,}".format(round(T["bad"]))),
            ("专题五：数据中心网络", "100 Gbps ＝ 12.5 GB/s", T["dcn"], GY2, "%.0f ms" % T["dcn"]),
            ("配对，低端", "7 GB/s", T["lo"], GR, "%.0f ms" % T["lo"]),
            ("配对，高端", "167 GB/s", T["hi"], GR, "%.1f ms" % T["hi"])]
    by = py + 84
    for lab, bw, ms, col, txt in bars:
        f.t(BX - 16, by + 4, lab, INK, True, 15, "end")
        f.t(BX - 16, by + 25, bw, GY, size=15, anchor="end")
        f.box(BX, by - 12, X(ms) - BX, 30, col, col, 3)
        f.t(X(ms) + 10, by + 9, txt, col, True, 15)
        by += 62
    f._pan = None
    yb = py + PH + 6
    yb = f.src(yb + 10,
               "📌 带宽和单次传输时间：gpu/inference/a4x-max/deepseek-v4/VLLM-V4PRO-RUNBOOK.md:194（GB300，vLLM，NixlConnector）：单次传输 200–800 ms 降到 2–10 ms。",
               "⚠️ 本课推导：1.04 GB ＝ 2 × 62 × 8 × 128 × 8,192 × 1 字节（Qwen3-Coder-480B，FP8），只当一把统一的尺子，不是 V4-Pro 的 KV。"
               "时间 ＝ 1.04 GB ÷ 带宽；没算握手和排队。prefill 1–2 s 与 100 Gbps 取自专题五 §6.2。")
    f.save("fig6-kv-bw.svg", yb + 14)


def fig_hybrid_kv():
    f = Fig(W, "老办法只传填满的整块，最后那截尾巴让 decode 端自己补算。"
               "左边是注意力层：KV 像一本逐格记的账，尾巴那格本来空着，decode 端补算只是把它填上，每格只记一次，结果不变。"
               "右边是 GDN 层：状态像存折余额，prefill 端算完时尾巴已经记进余额里了；decode 端再补算尾巴，"
               "等于同一笔钱又入账一次，余额多出一截，红色标着重复入账。"
               "修法是全部格子连同余额原样传过去，decode 端一个 token 都不补算")
    y0 = f.header("混合模型不能「传整块、尾巴自己算」　——　<tspan font-weight=\"700\">GDN 的状态会把尾巴记两遍</tspan>",
                  "示意，不按真实大小。Qwen3.5-397B 的 60 层里 45 层 GDN、15 层注意力",
                  [(PU, "注意力层的 KV"), ("#d7aefb", "GDN 层的状态"), (RD, "decode 端补算的尾巴")])
    PH = 312
    py = f.panel(0, y0, W, PH, "老办法：只传整块，尾巴让 decode 端补算", PU)
    NB, CW, CH = 7, 62, 40
    TAIL = 0.45                                 # 最后一块填了多少（示意）
    LX, RX = 40, 730
    BW_, TW = 400, round(CW * TAIL) + 30        # 余额条长度、尾巴那一截
    f.t(LX, py + 22, "注意力层（%d 层）＝ 逐格账本" % ATT, PU, True, 18)
    f.t(RX, py + 22, "GDN 层（%d 层）＝ 存折余额" % GDN, PU, True, 18)
    f.line(RX - 40, py + 4, RX - 40, py + 210, LINE, 1, arrow=False)

    def cells(y, tail):
        for i in range(NB):
            x = LX + i * CW
            if i < NB - 1:
                f.box(x, y, CW - 6, CH, PU, PU, 4)
            else:
                f.box(x, y, CW - 6, CH, "none", PU, 4, sw=1.6, dash=None if tail else "4,3")
                if tail:
                    f.box(x, y, (CW - 6) * TAIL, CH, RD, RD, 4)
    for k, (lab, tail) in enumerate((("prefill 端传来", False), ("decode 端补算尾巴后", True))):
        y = py + 68 + k * 92
        f.t(LX, y - 10, lab, INK, True, 15)
        cells(y, tail)
        f.t(RX, y - 10, lab, INK, True, 15)
        f.box(RX, y, BW_, CH, "#d7aefb", PU, 4)
        f.line(RX + BW_ - TW, y + 6, RX + BW_ - TW, y + CH - 6, "#681da8", 1.6, arrow=False)
        f.t(RX + BW_ - TW / 2, y + 26, "尾巴", "#681da8", True, 15, "middle")
        if tail:
            f.box(RX + BW_ + 4, y, TW, CH, RD, RD, 4)
            f.t(RX + BW_ + TW + 16, y + 27, "重复入账", RD, True, 17)
    f.t(LX + NB * CW + 10, py + 68 + 92 + 27, "空格填上，账不变", GR, True, 16)
    f.t(LX + NB * CW + 10, py + 68 + 27, "尾巴那格空着", GY, size=15)
    f.line(20, py + 222, W - 20, py + 222, LINE, 1, arrow=False)
    f.t(LX, py + 256, "混合模型版（TPUConnectorHMA）：%d 格连同余额全传，decode 端一个 token 都不补算" % NB, GR, True, 17)
    f._pan = None
    yb = f.src(py + PH + 16,
               "📌 tpu-inference tpu_connector_hma.py 注释原文：No trim, no block-alignment rounding … Required for Mamba correctness. "
               "P's Mamba state is a recurrent summary of all prompt tokens. If D re-prefilled any tail tokens locally, they would be fed through the Mamba recurrence a second time on D.",
               "📌 非 HMA 版把 prompt 长度按 block 向下取整，尾巴由 D 端重算。存折、账本只是比方：GDN 状态是固定大小的递归摘要，「多一截」指多喂了一遍，不是状态变大。",
               "📌 45 GDN ＋ 15 注意力与开关：tpu/tpu-inference/Qwen3.5-397B-A17B-FP8/README.md:375–389；PrFaaS（arXiv 2604.15039）写 Qwen3.5-397B 线性对全注意力 3:1，一致。",
               "📌 Kimi K3（GPU）：gpu/inference/kimi-k3/SGLANG-K3-RUNBOOK.md:1319（PD decode 的状态池一请求一 slot，--mamba-radix-cache-strategy 在 PD 下失效）。")
    f.save("fig6-hybrid-kv.svg", yb + 14)


def fig_prfaas():
    f = Fig(W, "四个模型在 32K 输入时，一台 8 卡 H200 做 prefill，每秒要往外送多少 KV。"
               "全注意力的 Qwen3-235B 要 33.35 Gbps，混合的 Qwen3.5-397B 只要 8.25，差 4 倍。"
               "全注意力的 MiniMax-M2.5 要 59.93，混合的 MiMo-V2-Flash 只要 4.66，差 13 倍。"
               "红虚线是案例里整条跨集群专线约 100 Gbps，整个 prefill 集群共用：Qwen3-235B 只够并 2 台，MiniMax 只够 1 台，Qwen3.5 能并 12 台，MiMo 21 台。案例里整个 prefill 集群平均出口只有 13 Gbps")
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
        f.t(X(v), AX + 24, str(v), GY, size=15, anchor="middle")
    f.path("M%.1f,%d L%.1f,%d" % (X(EGRESS), py + 40, X(EGRESS), AX), GR, 2, dash="5,4", arrow=False)
    f.t(X(EGRESS) + 8, AX - 12, "案例：整个 prefill 集群平均出口 %d Gbps ＝ 专线的 13%%" % EGRESS, GR, True, 15)
    by = py + 70
    for i, (name, kind, v, col) in enumerate(PHI):
        f.t(BX - 16, by + 4, name, INK, True, 15, "end")
        f.t(BX - 16, by + 25, kind, GY, size=15, anchor="end")
        f.box(BX, by - 12, X(v) - BX, 30, col, col, 3)
        f.box(X(v) + 4, by - 8, wpx("%.2f" % v, 15) + 12, 24, "#ffffff", "none", 0)   # 白底：挡住穿过的虚线
        f.t(X(v) + 10, by + 9, "%.2f" % v, col, True, 15)
        f.t(X(LINK) + 16, by + 9, "够并 %d 台" % int(LINK // v), GY, size=14)        # 红线右边单列：台数 ＝ 100 ÷ 柱长
        by += 64 if i % 2 else 56
    f.path("M%.1f,%d L%.1f,%d" % (X(LINK), py + 40, X(LINK), AX), RD, 2, dash="5,4", arrow=False)
    f.t(X(LINK) - 8, py + 58, "整条跨集群专线约 %d Gbps（集群共用）" % LINK, RD, True, 15, "end")
    f._pan = None
    yb = py + PH + 6
    yb = f.src(yb + 10,
               "📌 PrFaaS，arXiv 2604.15039（Moonshot AI ＋ 清华）：Table 3 的 Φ_kv；案例吞吐 +54%、P90 TTFT −64%、同成本约 +15%、比不调度的朴素异构 +32%；"
               "长度阈值 19.4K，约一半请求外送；平均出口 13 Gbps，占整条以太网的 13%（§4.3.1，故专线约 100 Gbps）。「够并几台」＝ ⌊100 ÷ 每台外送⌋，本课推导。",
               "📌 案例：内部 1T 混合模型，32 张 H200 做 prefill ＋ 64 张 H20 做本地 PD，对比 96 张 H20 同构 PD。「每 4 层里 3 层线性」即论文的混合 3:1；「每 6 层里 5 层只看近处」即 5:1 滑窗注意力（SWA）。",
               "⚠️ 那 4 倍、13 倍是两两同代对比（论文原文），不是同一参数量下的严格对照。")
    f.save("fig6-prfaas.svg", yb + 14)


if __name__ == "__main__":
    fig_kv_stack()
    fig_kv_bw()
    fig_hybrid_kv()
    fig_prfaas()
