#!/usr/bin/env python3
"""Build the customer-facing HTML report from results/ (single source of truth for every number).

Usage: python3 scripts/report/build_report.py [--out report.html]
Reads only files under results/; every table and chart in the HTML is computed here.
"""
import argparse
import glob
import json
import math
import os
import re
import statistics

import goodput

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "results")

# Validated categorical palette (dataviz skill reference palette, light mode), fixed order.
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4"}
INK, INK2, GRID = "#202124", "#5f6368", "#e8eaed"


def metric(text, key):
    m = re.search(re.escape(key) + r"[^:\n]*:\s+([\d.]+)", text)
    return float(m.group(1)) if m else float("nan")


def bench(path):
    t = open(path).read()
    keys = {
        "ok": "Successful requests", "fail": "Failed requests", "out": "Output token throughput",
        "total": "Total token throughput", "rps": "Request throughput", "ttft_mean": "Mean TTFT",
        "ttft_med": "Median TTFT", "ttft_p99": "P99 TTFT", "tpot": "Mean TPOT", "itl_p99": "P99 ITL",
        "e2e": "Mean E2EL",
    }
    d = {k: metric(t, v) for k, v in keys.items()}
    if math.isnan(d["fail"]):
        d["fail"] = 0.0
    return d


def sweep(dirpath, levels=(1, 2, 4, 8, 16, 32, 64, 128, 256)):
    rows = {}
    for c in levels:
        f = os.path.join(dirpath, f"bench-in8192-out1024-c{c}.txt")
        if os.path.exists(f):
            rows[c] = bench(f)
    return rows


def dp4_sweep():
    rows = {}
    for c in (4, 8, 16, 32, 64):
        fs = glob.glob(os.path.join(RES, "colocated-4xtp4-nvfp4", "*", "inst-*", f"bench-in8192-out1024-c{c}.txt"))
        bs = [bench(f) for f in fs]
        rows[4 * c] = {
            "out": sum(b["out"] for b in bs), "total": sum(b["total"] for b in bs),
            "ttft_med": statistics.mean(b["ttft_med"] for b in bs), "ttft_p99": max(b["ttft_p99"] for b in bs),
            "tpot": statistics.mean(b["tpot"] for b in bs), "itl_p99": max(b["itl_p99"] for b in bs),
            "fail": sum(b["fail"] for b in bs), "n_inst": len(bs),
        }
    return rows


def rate_rows(dirpath):
    rows = {}
    for f in glob.glob(os.path.join(dirpath, "rate-*.txt")):
        r = float(os.path.basename(f)[5:-4])
        rows[r] = bench(f)
    return dict(sorted(rows.items()))


def multi_rate(dirpath, scale):
    """Rate sweep split over several clients (inst-*): sum throughput, worst-case latency."""
    per = {}
    for f in glob.glob(os.path.join(dirpath, "inst-*", "rate-*.txt")):
        r = round(float(os.path.basename(f)[5:-4]) * scale, 4)
        per.setdefault(r, []).append(bench(f))
    return {r: {"rps": sum(b["rps"] for b in bs), "out": sum(b["out"] for b in bs), "total": sum(b["total"] for b in bs),
                "ttft_med": max(b["ttft_med"] for b in bs), "ttft_p99": max(b["ttft_p99"] for b in bs),
                "tpot": max(b["tpot"] for b in bs), "itl_p99": max(b["itl_p99"] for b in bs),
                "fail": sum(b["fail"] for b in bs)} for r, bs in sorted(per.items())}


def capacity():
    out = {}
    for tag in ("cap-nvfp4-tp8", "cap-nvfp4-tp4-8001", "cap-nvfp4-tp4-8002"):
        d = os.path.join(RES, "capacity", tag)
        if not os.path.isdir(d):
            continue
        pre = {int(re.search(r"c(\d+)", f).group(1)): bench(f) for f in glob.glob(os.path.join(d, "prefill-c*.txt"))}
        dec = {int(re.search(r"c(\d+)", f).group(1)): bench(f) for f in glob.glob(os.path.join(d, "decode-c*.txt"))}
        out[tag] = {"prefill": dict(sorted(pre.items())), "decode": dict(sorted(dec.items()))}
    return out


def gsm8k():
    res = {}
    for tag in ("fp8", "nvfp4"):
        f = glob.glob(os.path.join(RES, "eval", f"eval-{tag}-pd", "glm", "results_*.json"))
        if not f:
            continue
        r = json.load(open(f[0]))["results"]["gsm8k"]
        res[tag] = {k: r[k] for k in r if k.startswith("exact_match")}
    pa = os.path.join(RES, "eval", "paired-analysis.txt")
    res["paired"] = open(pa).read() if os.path.exists(pa) else ""
    return res


def kv_xfer(path):
    """Weighted average of vLLM 'KV Transfer metrics' lines in a decode log."""
    if not os.path.exists(path):
        return None
    rows = [l for l in open(path, errors="ignore") if "KV Transfer metrics" in l]
    g = lambda k, l: float(re.search(k + r"=([\d.]+)", l).group(1))
    n = sum(g("Num successful transfers", l) for l in rows)
    if not n:
        return None
    w = lambda k: sum(g(k, l) * g("Num successful transfers", l) for l in rows) / n
    return {"n": n, "ms": w(r"Avg xfer time \(ms\)"), "post": w(r"Avg post time \(ms\)"),
            "mb": w("Avg MB per transfer"), "mbps": w(r"Throughput \(MB/s\)")}


def nixl_rows():
    f = os.path.join(RES, "nixl-bench", "results.txt")
    if not os.path.exists(f):
        return []
    out, head = [], None
    for l in open(f):
        l = l.strip()
        if l.startswith("###"):
            head = l[3:].strip()
        elif l.startswith("{") and head:
            d = json.loads(l)
            out.append((head, d))
        elif l.startswith("8 pairs") or l.startswith("4 pairs"):
            out.append((head, {"pairs_line": l}))
    return out


# ---------------------------------------------------------------- SVG helpers
def nice_ticks(lo, hi, n=5):
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    t = math.floor(lo / step) * step
    ticks = []
    while True:
        ticks.append(round(t, 6))
        if t >= hi - 1e-9:
            break
        t += step
    return ticks


def line_chart(series, xlabel, ylabel, w=760, h=380, logx=False, xticks=None, ymax=None, xmax=None, label_last=True):
    """series: list of (name, color, [(x, y, tooltip)], dashed)"""
    L, R, T, B = 64, 150, 16, 48
    pw, ph = w - L - R, h - T - B
    xs = [p[0] for s in series for p in s[2]]
    ys = [p[1] for s in series for p in s[2]]
    x0, x1 = (min(xs), xmax or max(xs))
    y1 = ymax or max(ys) * 1.1
    yt = nice_ticks(0, y1)
    y1 = yt[-1]

    def fx(x):
        if logx:
            return L + (math.log2(x) - math.log2(x0)) / (math.log2(x1) - math.log2(x0)) * pw
        return L + (x - x0) / (x1 - x0) * pw

    def fy(y):
        return T + ph - y / y1 * ph

    o = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    for t in yt:
        o.append(f'<line x1="{L}" x2="{L+pw}" y1="{fy(t):.1f}" y2="{fy(t):.1f}" stroke="{GRID}"/>')
        o.append(f'<text x="{L-8}" y="{fy(t)+4:.1f}" text-anchor="end" class="tick">{t:g}</text>')
    for t in (xticks or nice_ticks(x0, x1)):
        o.append(f'<text x="{fx(t):.1f}" y="{T+ph+18}" text-anchor="middle" class="tick">{t:g}</text>')
    o.append(f'<line x1="{L}" x2="{L+pw}" y1="{T+ph}" y2="{T+ph}" stroke="#9aa0a6"/>')
    o.append(f'<text x="{L+pw/2}" y="{h-8}" text-anchor="middle" class="axis">{xlabel}</text>')
    o.append(f'<text x="14" y="{T+ph/2}" text-anchor="middle" class="axis" transform="rotate(-90 14 {T+ph/2})">{ylabel}</text>')
    labels = []
    for name, color, pts, dashed in series:
        pts = sorted(pts)
        d = " ".join(f"{'M' if i == 0 else 'L'}{fx(x):.1f},{fy(y):.1f}" for i, (x, y, _) in enumerate(pts))
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        o.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"{dash}/>')
        for x, y, tip in pts:
            o.append(f'<circle cx="{fx(x):.1f}" cy="{fy(y):.1f}" r="4.5" fill="{color}" stroke="#fff" stroke-width="2"><title>{name}: {tip}</title></circle>')
        if label_last and pts:
            labels.append([fx(pts[-1][0]) + 8, fy(pts[-1][1]) + 4, name])
    # direct labels at line ends, de-overlapped vertically (min 14px apart)
    labels.sort(key=lambda t: t[1])
    for i in range(1, len(labels)):
        if abs(labels[i][0] - labels[i-1][0]) < 160 and labels[i][1] - labels[i-1][1] < 14:
            labels[i][1] = labels[i-1][1] + 14
    for lx, ly, name in labels:
        o.append(f'<text x="{lx:.1f}" y="{ly:.1f}" class="dlabel">{name}</text>')
    o.append("</svg>")
    legend = '<div class="legend">' + "".join(
        f'<span><i style="background:{c}"></i>{n}</span>' for n, c, _, _ in series) + "</div>"
    return legend + "\n".join(o)


def bar_chart(groups, series, ylabel, w=760, h=320, fmt="{:.0f}", ymax=None):
    """groups: list of labels; series: list of (name, color, [values])"""
    L, R, T, B = 64, 16, 24, 44
    pw, ph = w - L - R, h - T - B
    y1 = ymax or max(v for s in series for v in s[2] if v == v) * 1.15
    yt = nice_ticks(0, y1)
    y1 = yt[-1]
    gw = pw / len(groups)
    bw = min(34, (gw - 16) / len(series) - 2)
    o = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">']
    for t in yt:
        y = T + ph - t / y1 * ph
        o.append(f'<line x1="{L}" x2="{L+pw}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}"/>')
        o.append(f'<text x="{L-8}" y="{y+4:.1f}" text-anchor="end" class="tick">{t:g}</text>')
    for gi, g in enumerate(groups):
        cx = L + gw * gi + gw / 2
        o.append(f'<text x="{cx:.1f}" y="{T+ph+18}" text-anchor="middle" class="tick">{g}</text>')
        tot = len(series) * (bw + 2)
        for si, (name, color, vals) in enumerate(series):
            v = vals[gi]
            if v != v:
                continue
            x = cx - tot / 2 + si * (bw + 2)
            bh = v / y1 * ph
            y = T + ph - bh
            o.append(f'<path d="M{x:.1f},{T+ph} V{y+4:.1f} Q{x:.1f},{y:.1f} {x+4:.1f},{y:.1f} H{x+bw-4:.1f} Q{x+bw:.1f},{y:.1f} {x+bw:.1f},{y+4:.1f} V{T+ph} Z" fill="{color}"><title>{name} · {g}: {fmt.format(v)}</title></path>')
            o.append(f'<text x="{x+bw/2:.1f}" y="{y-5:.1f}" text-anchor="middle" class="vlabel">{fmt.format(v)}</text>')
    o.append(f'<line x1="{L}" x2="{L+pw}" y1="{T+ph}" y2="{T+ph}" stroke="#9aa0a6"/>')
    o.append(f'<text x="14" y="{T+ph/2}" text-anchor="middle" class="axis" transform="rotate(-90 14 {T+ph/2})">{ylabel}</text>')
    o.append("</svg>")
    legend = '<div class="legend">' + "".join(f'<span><i style="background:{c}"></i>{n}</span>' for n, c, _ in series) + "</div>"
    return legend + "\n".join(o)


def table(head, rows, cls=""):
    h = "".join(f"<th>{x}</th>" for x in head)
    b = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def f0(x):
    return "—" if x != x else f"{x:,.0f}"


def f1(x):
    return "—" if x != x else f"{x:,.1f}"


def s2(ms):
    return "—" if ms != ms else (f"{ms/1000:.2f} s" if ms >= 1000 else f"{ms:.0f} ms")


# ---------------------------------------------------------------- build
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "report.html"))
    a = ap.parse_args()

    J = lambda *p: os.path.join(RES, *p)
    single = sweep(J("fp8-tp8-single", "8k1k"))
    base = {os.path.basename(f)[6:-4]: bench(f) for f in glob.glob(J("fp8-tp8-single", "bench-in*.txt"))}
    col4 = dp4_sweep()
    p2 = sweep(J("pd-2p2d-nvfp4"))
    p4 = sweep(J("pd-4p2d-nvfp4"))
    r_fp8 = rate_rows(J("rate-colocated-fp8-tp8"))
    r_nv8 = rate_rows(J("rate-colocated-nvfp4-tp8"))
    r_nv44 = multi_rate(J("rate-colocated-nvfp4-2xtp4"), 2)
    r_p2 = rate_rows(J("rate-2p2d-nvfp4"))
    r_p4 = rate_rows(J("rate-4p2d-nvfp4"))
    cap = capacity()
    ev = gsm8k()

    SWH = ["并发", "输出 tok/s", "每卡 tok/s", "每路 tok/s", "TTFT 中位数", "TTFT P99", "平均 TPOT", "P99 ITL", "失败"]
    def sw_rows(rows, gpus):
        return [[c, f1(r["out"]), f1(r["out"] / gpus), f"{1000/r['tpot']:.0f}" if r["tpot"] else "—", s2(r["ttft_med"]), s2(r["ttft_p99"]),
                 f"{r['tpot']:.1f} ms", f"{r['itl_p99']:.0f} ms", f0(r["fail"])] for c, r in rows.items()]
    RTH = ["到达速率", "实际完成 req/s", "输出 tok/s", "每卡 tok/s", "TTFT 中位数", "TTFT P99", "平均 TPOT", "P99 ITL", "失败"]
    def rate_table(rows, gpus):
        return table(RTH, [[f"{r:g} req/s", f"{v['rps']:.2f}", f0(v["out"]), f1(v["out"] / gpus), s2(v["ttft_med"]), s2(v["ttft_p99"]),
                            f"{v['tpot']:.1f} ms", f"{v['itl_p99']:.0f} ms", f0(v["fail"])] for r, v in rows.items()])

    # chart: per-GPU throughput vs TPOT (concurrency sweeps)
    def pts(rows, gpus, skip=()):
        return [(r["tpot"], r["out"] / gpus, f"并发 {c}：整体 {r['out']:.0f} tok/s，TPOT {r['tpot']:.1f} ms") for c, r in rows.items() if c not in skip]
    frontier = line_chart([
        ("FP8 单机 TP8", C["blue"], pts(single, 8), False),
        ("NVFP4 不分离 4×TP4", C["aqua"], pts(col4, 16), False),
        ("NVFP4 2P2D", C["yellow"], pts(p2, 16), False),
        ("NVFP4 4P2D", C["orange"], pts(p4, 24), False),
    ], "平均 TPOT（ms，越左越快）", "每卡输出 tok/s", xmax=140)
    def itl(rows):
        return [(c, r["itl_p99"], f"并发 {c}：P99 ITL {r['itl_p99']:.0f} ms") for c, r in rows.items() if c >= 4]
    itlc = line_chart([
        ("不分离", C["aqua"], itl(col4), False),
        ("PD 分离 2P2D", C["yellow"], itl(p2), False),
    ], "总并发（对数刻度，16 卡）", "生成卡顿 P99 ITL（ms）", logx=True, xticks=[4, 8, 16, 32, 64, 128, 256])
    # chart: TTFT median vs load per 8 GPUs
    def norm(rows, machines):
        return [(r / machines, v["ttft_med"] / 1000, f"每台机器 {r/machines:.2f} req/s：TTFT 中位数 {v['ttft_med']/1000:.2f} s") for r, v in rows.items()]
    ttftc = line_chart([
        ("NVFP4 不分离 2×TP4", C["aqua"], norm(r_nv44, 1), False),
        ("NVFP4 2P2D", C["yellow"], norm(r_p2, 2), False),
        ("NVFP4 4P2D", C["orange"], norm(r_p4, 3), False),
    ], "每台机器（8 卡）的到达速率，req/s", "TTFT 中位数（s）")
    tpotc = line_chart([
        ("FP8 TP8", C["blue"], [(r, v["tpot"], f"{r} req/s：TPOT {v['tpot']:.1f} ms") for r, v in r_fp8.items()], False),
        ("NVFP4 TP8", C["orange"], [(r, v["tpot"], f"{r} req/s：TPOT {v['tpot']:.1f} ms") for r, v in r_nv8.items()], False),
        ("NVFP4 2×TP4", C["aqua"], [(r, v["tpot"], f"{r} req/s：TPOT {v['tpot']:.1f} ms") for r, v in r_nv44.items()], False),
    ], "单台机器的到达速率，req/s", "平均 TPOT（ms）", ymax=160)

    # capacity
    pre = lambda t: max(v["total"] for v in cap[t]["prefill"].values())
    pre8, pre4, pre4b = pre("cap-nvfp4-tp8"), pre("cap-nvfp4-tp4-8001"), pre("cap-nvfp4-tp4-8002")
    capbars = bar_chart(["每个实例", "每台机器（8 卡）"], [("1 × TP8", C["blue"], [pre8, pre8]), ("TP4（每台 2 个）", C["orange"], [pre4, pre4 + pre4b])],
                        "prefill 吞吐（输入 tok/s）", h=280)
    cap_table = table(["实例", "prefill（8K 输入）", "decode 并发 32", "decode 并发 64", "decode 并发 128", "decode 并发 256"],
        [[lab, f0(pre(t)) + " tok/s"] + [f"{cap[t]['decode'][c]['out']:,.0f} · {cap[t]['decode'][c]['tpot']:.0f} ms" for c in (32, 64, 128, 256)]
         for t, lab in (("cap-nvfp4-tp8", "NVFP4 TP8（8 卡）"), ("cap-nvfp4-tp4-8001", "NVFP4 TP4 #1（4 卡）"), ("cap-nvfp4-tp4-8002", "NVFP4 TP4 #2（同机并行）"))])

    # GSM8K
    g8, g4 = ev.get("fp8", {}), ev.get("nvfp4", {})
    gsm_rows = []
    for k, lab in (("exact_match,strict-match", "strict-match"), ("exact_match,flexible-extract", "flexible-extract")):
        se = lambda g: g[k.replace("exact_match", "exact_match_stderr")]
        gsm_rows.append([lab, f"{g8[k]*100:.2f}% ± {se(g8)*100:.2f}", f"{g4[k]*100:.2f}% ± {se(g4)*100:.2f}", f"{(g4[k]-g8[k])*100:+.2f} 个百分点"])
    pm = re.search(r"\[strict-match\].*?both right=(\d+)\s+only FP8 right=(\d+)\s+only NVFP4 right=(\d+)\s+both wrong=(\d+).*?identical extracted answer: (\d+)/(\d+).*?p = ([\d.]+)", ev.get("paired", ""), re.S)

    # network
    kv_old = kv_xfer(J("_archive", "pd-2p2d-nvfp4-ucxdefault", "pd-decode-d1.log"))
    kv_new = kv_xfer(J("pd-2p2d-nvfp4", "pd-decode-d1.log"))
    nrows = []
    for head, d in nixl_rows():
        env = head.split("|")[0].strip() if "|" in head else head
        envs = "默认" if not env else env.replace("-e ", "").replace("UCX_TCP_TX_SEG_SIZE=", "TX=").replace("UCX_TCP_RX_SEG_SIZE=", "RX=")
        if "pairs_line" in d:
            m = re.search(r"(\d+) pairs, sum aggregate Gbps = ([\d.]+)", d["pairs_line"])
            nrows.append([head.replace("pairs=", "并行 agent 对数 "), "GPU 显存", "—", "—", f"{m.group(2)} Gbps（{m.group(1)} 对合计）"])
        else:
            nrows.append([envs, "GPU 显存" if d["mem"] == "cuda" else "CPU 内存", f"{d['descs']} 块 × {d['streams']} 路", f"{d['median_ms']} ms", f"{d['aggregate_Gbps']} Gbps"])
    kv_html = table(["NVFP4 2P2D", "每个 rank 每次传输", "平均传输耗时", "传完后处理", "单次传输速度"],
                    [[lab, f"{k['mb']:.0f} MB", f"{k['ms']:.0f} ms", f"{k['post']:.0f} ms", f"{k['mbps']:.0f} MB/s"] for lab, k in (("UCX 默认", kv_old), ("分段 1m", kv_new)) if k])

    # goodput
    gp = goodput.table()
    gp_rows, kbest = [], None
    for name, gpus, g in gp:
        cells = [name, gpus]
        for sla in goodput.SLAS:
            x = g[sla["key"]] if g else None
            cells.append("不达标" if x is None else f"<b>{x[0]/gpus:.0f}</b> / {x[1]/gpus:,.0f}" + ("*" if x[3].startswith("≥") else ""))
        gp_rows.append(cells)
        if g and g["S2"] and (not kbest or g["S2"][0] / gpus > kbest[0]):
            kbest = (g["S2"][0] / gpus, name)
    gp_html = table(["部署", "卡数"] + [f"{s['key']} {s['name']}<br><span class='note'>{s['desc']}</span>" for s in goodput.SLAS], gp_rows)
    pick = {"FP8 单机 TP8（不分离）": ("FP8，8 卡实例，不分离", C["blue"]), "NVFP4 单机 2×TP4（不分离）": ("NVFP4，4 卡实例，不分离", C["aqua"]),
            "NVFP4 2P2D（两机，全 TP4）": ("NVFP4，PD 分离 2P2D", C["yellow"])}
    gpbars = bar_chart([s_["key"] + " " + s_["name"] for s_ in goodput.SLAS],
        [(pick[name][0], pick[name][1], [(g[s_["key"]][0] / gpus if g and g[s_["key"]] else float("nan")) for s_ in goodput.SLAS])
         for name, gpus, g in gp if name in pick], "每卡输出 tok/s", h=320)
    peak = []
    for lab, rows, gpus in (("FP8 单机 TP8", single, 8), ("NVFP4 不分离 4×TP4", col4, 16), ("NVFP4 2P2D", p2, 16), ("NVFP4 4P2D", p4, 24)):
        c, r = max(rows.items(), key=lambda kv: kv[1]["out"])
        peak.append([lab, gpus, c, f"{r['out']:,.0f}", f"{r['out']/gpus:.0f}", f"{r['total']/gpus:,.0f}", f"{r['tpot']:.0f} ms", s2(r["ttft_med"])])

    c128 = lambda rows: rows[128]
    D3 = ((col4, 16), (p2, 16), (p4, 24))
    cmp = table(["总并发", "每卡输出 tok/s<br>不分离 / 2P2D / 4P2D", "平均 TPOT（ms）<br>不分离 / 2P2D / 4P2D", "P99 ITL（ms）<br>不分离 / 2P2D / 4P2D", "TTFT 中位数（s）<br>不分离 / 2P2D / 4P2D"],
        [[c, " / ".join(f"{r[c]['out']/g:.0f}" for r, g in D3), " / ".join(f"{r[c]['tpot']:.0f}" for r, g in D3),
          " / ".join(f"{r[c]['itl_p99']:.0f}" for r, g in D3), " / ".join(f"{r[c]['ttft_med']/1000:.1f}" for r, g in D3)] for c in (16, 32, 64, 128, 256)])

    rep = {
        "%%BASE_TABLE%%": table(["场景（输入 / 输出 / 并发）", "输出 tok/s", "TTFT 中位数", "平均 TPOT", "P99 ITL"],
            [[k.replace("in", "").replace("-out", " / ").replace("-c", " / "), f1(v["out"]), s2(v["ttft_med"]), f"{v['tpot']:.2f} ms", f"{v['itl_p99']:.1f} ms"]
             for k, v in sorted(base.items(), key=lambda kv: [int(x) for x in re.findall(r"\d+", kv[0])])]),
        "%%SINGLE_TABLE%%": table(SWH, sw_rows(single, 8)),
        "%%COL4_TABLE%%": table(["总并发"] + SWH[1:], sw_rows(col4, 16)),
        "%%P2_TABLE%%": table(SWH, sw_rows(p2, 16)), "%%P4_TABLE%%": table(SWH, sw_rows(p4, 24)),
        "%%R_FP8%%": rate_table(r_fp8, 8), "%%R_NV8%%": rate_table(r_nv8, 8), "%%R_NV44%%": rate_table(r_nv44, 8),
        "%%R_P2%%": rate_table(r_p2, 16), "%%R_P4%%": rate_table(r_p4, 24),
        "%%FRONTIER%%": frontier, "%%ITL%%": itlc, "%%TTFT%%": ttftc, "%%TPOTC%%": tpotc, "%%CAPBARS%%": capbars, "%%CAP_TABLE%%": cap_table,
        "%%CMP%%": cmp, "%%GSM_TABLE%%": table(["指标", "FP8", "NVFP4", "差值"], gsm_rows),
        "%%PAIRED%%": table(["", "题数"], [["两边都对", pm.group(1)], ["只有 FP8 对", pm.group(2)], ["只有 NVFP4 对", pm.group(3)], ["两边都错", pm.group(4)], ["抽取出的答案完全相同", f"{pm.group(5)} / {pm.group(6)}"]]),
        "%%PVAL%%": f"{float(pm.group(7)):.2f}", "%%NET_NIXL%%": table(["UCX 设置", "数据位置", "拆分 / 并发", "80 MB 一次耗时", "吞吐"], nrows), "%%NET_KV%%": kv_html,
        "%%GOODPUT%%": gp_html, "%%GPBARS%%": gpbars, "%%PEAK%%": table(["部署", "卡数", "最高并发", "输出 tok/s", "每卡输出 tok/s", "每卡总 tok/s", "TPOT", "TTFT 中位数"], peak),
        "%%KPI_S2%%": f"{kbest[0]:.0f}", "%%KPI_S2_WHO%%": kbest[1],
        "%%PRE8%%": f0(pre8), "%%PRE4%%": f0(pre4), "%%PRE4X2%%": f0(pre4 + pre4b),
        "%%P2_128_ITL%%": f"{p2[128]['itl_p99']:.0f}", "%%C4_128_ITL%%": f"{col4[128]['itl_p99']:.0f}",
        "%%P2_128_OUT%%": f0(p2[128]["out"]), "%%C4_128_OUT%%": f0(col4[128]["out"]),
    }
    html = TEMPLATE
    for k, v in rep.items():
        html = html.replace(k, str(v))
    left = sorted(set(re.findall(r"%%[A-Z0-9_]+%%", html)))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    open(a.out, "w").write(html)
    print("wrote", a.out)


TEMPLATE = open(os.path.join(HERE, "template.html")).read()

if __name__ == "__main__":
    main()
