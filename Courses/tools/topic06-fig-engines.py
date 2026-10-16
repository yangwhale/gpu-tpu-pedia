# -*- coding: utf-8 -*-
r"""专题六 · 第零节 0.1「两大引擎你方唱罢我登场」（fig6-engines）。R85（2026-10-16）。

⭐ 回答的问题：vLLM 和 SGLang 这三年谁先做了什么、谁跟进了谁？外面的论文和新玩家在哪儿插进来？
   上中下三条轨：vLLM（蓝）／外部玩家（灰：论文、DeepSeek、Mooncake、Dynamo）／SGLang（红）。
   卡片落在首次公开的年月上，卡上写日期和原文给的数（带对照对象）。
⛔ 箭头只画两类，都有原文或日期：
     ① 原文明说的复用／致谢：SGLang 首发博客「reused some code from … vLLM」；
        vLLM V1 博客「SGLang … significantly influenced our work」。
     ② 有日期的先后：分块 prefill、接 Mooncake 传输 vLLM 先；MTP、EAGLE-3、大 EP SGLang 先。
   推测类的借鉴（APC 是否回应 RadixAttention、V1 默认开前缀缓存是否被倒逼、v0.6.0 是否因被点名）一律不画。
⛔ 不写任何融资、估值、钱。
⛔ 出处：/tmp/t6/engines-timeline.md 的逐条原文 URL，已抄进第十五节台账「R85 补充」。
   「晚 3 个多月」「晚 5 周」这类间隔是按两边日期算的（本课推导），口语一律往下取整。
"""
from topic03_draw import Fig, BL, RD, GY, INK, LINE, wpx, MONO_K

W = 1400
Y0, Y1 = 2023.30, 2026.78
XL, XR = 118, W - 2
CWMAX = 216
CH1 = 42                     # 一行细节的卡高
LH = 17

# (key, 年份小数, 标题, [细节行...])
VLLM = [
    ("pa", 2023.46, "2023-06 PagedAttention", ["吞吐最多是 HF 的 24×"]),
    ("apc", 2024.21, "2024-03 前缀缓存 APC", ["V0 里默认关"]),
    ("chunk", 2024.29, "2024-04 分块 prefill", ["引擎里先落地"]),
    ("v060", 2024.68, "2024-09 v0.6.0", ["自承 CPU 开销拖累 GPU", "吞吐 2.7×（对 v0.5.3）"]),
    ("mc", 2024.96, "2024-12 接 Mooncake", ["传输引擎", "同月合入首批 PD 代码"]),
    ("v1", 2025.07, "2025-01 V1 重写", ["原文：SGLang 等对它影响显著", "前缀缓存改成默认开"]),
    ("mtp", 2025.13, "2025-02 MTP", ["晚 SGLang 5 天"]),
    ("conn", 2025.29, "2025-04 KV Connector", ["可插拔的 KV 传输接口", "LMCache、NIXL 等接进来"]),
    ("eagle", 2025.32, "2025-04 EAGLE-3", ["晚 SGLang 5 周"]),
    ("deepep", 2025.44, "2025-06 接 DeepEP", ["大 EP 晚 SGLang 1 个多月"]),
    ("tpu", 2025.79, "2025-10 删掉 V0", ["TPU 新后端 tpu-inference"]),
    ("wide", 2025.96, "2025-12 Wide-EP", ["每 H200 1.5k → 2.2k tok/s"]),
    ("mrv2", 2026.23, "2026-03 MRV2", ["吞吐 +56%（Qwen3-0.6B）"]),
    ("mk", 2026.71, "2026-09 TPU megakernel", ["Inferact"]),
]
EXT = [
    ("orca", 2023.30, "2022 Orca 论文", ["连续批"]),
    ("fg", 2023.87, "2023-11 DS-FastGen", ["分块思路", "vLLM 回「已在路线图」"]),
    ("dist", 2024.04, "2024-01 DistServe 论文", ["PD 分离"]),
    ("sar", 2024.21, "2024-03 Sarathi 论文", ["分块 prefill"]),
    ("v2", 2024.37, "2024-05 DeepSeek-V2", ["MLA"]),
    ("mcp", 2024.46, "2024-06 Mooncake 论文", ["Kimi 的 KV 池"]),
    ("mco", 2024.90, "2024-11 Mooncake 开源", ["传输引擎"]),
    ("v3", 2024.98, "2024-12 DeepSeek-V3", [""]),
    ("osw", 2025.15, "2025-02 DeepSeek 开源周", ["FlashMLA、DeepEP、", "DeepGEMM"]),
    ("dyn", 2025.21, "2025-03 Dynamo 开源", ["首发即支持两家"]),
    ("v32", 2025.74, "2025-09 DeepSeek-V3.2", ["DSA；两家同日 Day-0"]),
    ("v4", 2026.29, "2026-04 DeepSeek-V4", ["两家都 Day-0"]),
]
SGL = [
    ("radix", 2024.04, "2024-01 SGLang 首发", ["RadixAttention", "吞吐最多 5×（对 vLLM 等）", "原文：复用了 vLLM 的代码"]),
    ("chunk", 2024.54, "2024-07 v0.2", ["纯 Python 调度", "3.1×（对 vLLM 0.5.2）", "分块 prefill：晚 vLLM", "3 个多月"]),
    ("mla", 2024.68, "2024-09 v0.3 MLA 优化", ["先手；3–7×（对基线）"]),
    ("v04", 2024.92, "2024-12 v0.4", ["零开销调度、DP attention", "V3 Day-0，早 vLLM 一两天"]),
    ("mtp", 2025.12, "2025-02 MTP", ["先手"]),
    ("eagle", 2025.21, "2025-03 EAGLE-3", ["自称首个支持"]),
    ("mc", 2025.27, "2025-04 接 Mooncake", ["晚 vLLM 3 个多月"]),
    ("ep", 2025.34, "2025-05 大 EP＋PD 复现", ["最多纯 TP 的 5×"]),
    ("hic", 2025.69, "2025-09 HiCache 分层 KV", ["KV 下放到 CPU 和远端存储"]),
    ("jax", 2025.82, "2025-10 SGLang-JAX", ["上 TPU"]),
    ("gb", 2026.46, "2026-06 V4 on GB300", ["5.1×（同交互档）", "对自家 Day-0"]),
]
# 跟进箭头：(从, 到, 颜色)。颜色 ＝ 先做的那一家
ARROWS = [
    ("v.pa", "s.radix", BL),       # 原文：reused some code from vLLM
    ("s.v04", "v.v1", RD),         # 原文：SGLang 等 significantly influenced
    ("v.chunk", "s.chunk", BL),
    ("v.mc", "s.mc", BL),
    ("s.mtp", "v.mtp", RD),
    ("s.eagle", "v.eagle", RD),
    ("s.ep", "v.deepep", RD),
    ("e.fg", "v.chunk", GY),       # vLLM 原文：This optimization is on vLLM's roadmap
    ("e.v2", "s.mla", GY),         # SGLang v0.3：DeepSeek MLA 优化
    ("e.mco", "v.mc", GY),         # Mooncake README：vLLM officially supports Mooncake Transfer Engine
]
LANES = [("v", BL, "vLLM", VLLM), ("e", GY, "外部玩家", EXT), ("s", RD, "SGLang", SGL)]


def xof(yr):
    return XL + (yr - Y0) / (Y1 - Y0) * (XR - XL - CWMAX)


def cw(title, det):
    w = max([wpx(title, 13) * MONO_K * 1.08] + [wpx(d, 12) * MONO_K for d in det]) + 22
    assert w <= CWMAX + 2, "卡太宽：%s（%d px）" % (title, w)
    return int(w)


def fig_engines():
    f = Fig(W, "vLLM 和 SGLang 的双泳道时间线，2023 年 6 月到 2026 年。上面蓝色是 vLLM，下面红色是 SGLang，中间灰色是外部玩家："
               "Orca、DistServe、Sarathi-Serve 这些论文，DeepSeek 的 V2（MLA）、V3、开源周、V3.2、V4，Mooncake，NVIDIA Dynamo。"
               "箭头从先做的指向跟进的。2023 年 6 月 vLLM 首发 PagedAttention；2024 年 1 月 SGLang 首发 RadixAttention，原文写着复用了 vLLM 的代码。"
               "分块 prefill 和接 Mooncake 传输都是 vLLM 先、SGLang 晚三个多月。SGLang 2024 年 9 月先做 MLA 优化，12 月 DeepSeek-V3 首日可用，"
               "2025 年 2 月 MTP 比 vLLM 早 5 天，EAGLE-3 早 5 周，大 EP 也先做。vLLM 2025 年 1 月重写成 V1，原文说受 SGLang 等显著影响。"
               "DeepSeek-V3.2、V4 两家都在首日支持")
    y0 = f.header("两大引擎　——　<tspan font-weight=\"700\">你方唱罢我登场</tspan>",
                  "卡片落在首次公开的年月上，数字带对照对象；箭头从先做的指向跟进的，只画原文明说的复用、致谢和有日期的先后",
                  [(BL, "vLLM 先"), (RD, "SGLang 先"), (GY, "外部玩家")])
    ya = y0 + 4
    ticks = [(xof(yr), str(yr)) for yr in (2024, 2025, 2026)]
    for x, lab in ticks:
        f.t(x, ya + 12, lab, GY, True, 15, "middle")
    y = ya + 26
    ytop = y
    geo = {}                      # key -> (x, y, w, h)
    lanebox = []
    for pre, col, name, cards in LANES:
        rows, place = [], []
        for key, yr, title, det in sorted(cards, key=lambda c: c[1]):
            x = xof(yr)
            w = cw(title, det)
            for ri, right in enumerate(rows):
                if x >= right + 8:
                    rows[ri] = x + w
                    place.append((ri, x, w, key, title, det))
                    break
            else:
                rows.append(x + w)
                place.append((len(rows) - 1, x, w, key, title, det))
        hof = lambda det: 26 + LH * len([d for d in det if d]) + 4
        rh = [0] * len(rows)          # 每个子行高 ＝ 该行最高的卡 ＋ 间距
        for ri, x, w, key, title, det in place:
            rh[ri] = max(rh[ri], hof(det) + 12)
        ry = [y + 8 + sum(rh[:i]) for i in range(len(rows))]
        h = sum(rh) + 4
        lanebox.append((y, h, col, name))
        for ri, x, w, key, title, det in place:
            geo[pre + "." + key] = (x, ry[ri], w, hof(det), title, det, col)
        y += h + 10
    # 轨道底框（不填）→ 网格 → 箭头 → 卡片，后画的盖前画的
    for ly, h, col, name in lanebox:
        f.box(0, ly, W, h, "none", LINE, 6)
        f.box(0, ly, 4, h, col, col, 0)
        f.t(16, ly + h / 2 + 6, name, col, True, 16)
    for x, _ in ticks:
        f.line(x, ytop - 4, x, y - 10, LINE, 1, "4 4", arrow=False)
    for a, b, col in ARROWS:
        xa, ya_, wa, ha = geo[a][:4]
        xb, yb, wb, hb = geo[b][:4]
        x1 = xa + min(wa / 2, 40)
        x2 = xb + min(wb / 2, 40)
        if yb > ya_:
            y1, y2 = ya_ + ha + 2, yb - 4
        else:
            y1, y2 = ya_ - 2, yb + hb + 4
        ym = (y1 + y2) / 2
        f.path("M %.1f %.1f C %.1f %.1f, %.1f %.1f, %.1f %.1f" % (x1, y1, x1, ym, x2, ym, x2, y2),
               col, 1.8, None if col != GY else "5 3")
    for key, (x, cy, w, hh, title, det, col) in geo.items():
        f.box(x, cy, w, hh, "#fff", col, 6, 1.3)
        f.t(x + 9, cy + 18, title, INK, True, 13)
        for k, d in enumerate(dd for dd in det if dd):
            f.t(x + 9, cy + 18 + LH * (k + 1), d, col if col != GY else GY, False, 12)
    f.src(y, "vLLM：blog.vllm.ai 2023-06-20（对 HF 24×）、2024-09-05（v0.6.0，2.7× 对 v0.5.3）、2025-01-27（V1）；"
             "PR #2762（APC）、#3884（分块 prefill，2024-04-11）、#10884（Mooncake，2024-12-15）、#10502（PD，2024-12-01）、"
             "#12755（MTP，2025-02-19）、#16937（EAGLE-3，2025-04-25）、#18434（DeepEP，2025-06-10）、#15960（KV Connector）；"
             "v0.11.0 release（删 V0）；blog.vllm.ai 2025-10-16（tpu-inference）；vllm.ai/blog 2025-12-17（Wide-EP）、2026-03-24（MRV2）；"
             "inferact.ai/blog/tpu-megakernels。"
             "SGLang：lmsys.org/blog 2024-01-17（5× 对 Guidance、vLLM；复用 vLLM 代码）、2024-07-25（v0.2，3.1× 对 vLLM 0.5.2）、"
             "2024-09-04（v0.3 MLA 3–7×）、2024-12-04（v0.4）、2025-05-05（大 EP，96×H100）、2025-09-10（HiCache）、2025-10-29（SGLang-JAX）；"
             "PR #800（分块 prefill，2024-07-29）、#3582（MTP，2025-02-14）、#4247（EAGLE-3，2025-03-18）；v0.4.1 release（V3 Day-0）；"
             "PyTorch 博客 2026-06-23（V4 on GB300）。"
             "外部：DeepSpeed-FastGen 博客与 vLLM 回应（2023-11-14）；DistServe arXiv 2401.09670；Sarathi-Serve arXiv 2403.02310；"
             "DeepSeek-V2 arXiv 2405.04434；Mooncake arXiv 2407.00079 与 GitHub README；DeepSeek open-infra-index；"
             "NVIDIA Dynamo 新闻稿 2025-03-18；DeepSeek-V3.2 两家 Day-0 博客 2025-09-29；V4 两家博客 2026-04-24／25。"
             "间隔（晚 5 天、5 周、1 个多月、3 个多月）按两边日期本课推导，往下取整。推测的借鉴不画箭头。")
    f.save("fig6-engines.svg", y + 4)


fig_engines()
