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
        fs = glob.glob(os.path.join(RES, "dp4-nvfp4-colocated", "*", "inst-*", f"bench-in8192-out1024-c{c}.txt"))
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


def dp4_rate():
    rows = {}
    for d in glob.glob(os.path.join(RES, "rate-dp4-nvfp4", "total-*")):
        r = float(os.path.basename(d)[6:])
        bs = [bench(f) for f in glob.glob(os.path.join(d, "inst-*", "rate-*.txt"))]
        if not bs:
            continue
        rows[r] = {
            "rps": sum(b["rps"] for b in bs), "out": sum(b["out"] for b in bs),
            "ttft_med": statistics.mean(b["ttft_med"] for b in bs), "ttft_p99": max(b["ttft_p99"] for b in bs),
            "tpot": statistics.mean(b["tpot"] for b in bs), "itl_p99": max(b["itl_p99"] for b in bs),
            "fail": sum(b["fail"] for b in bs),
        }
    return dict(sorted(rows.items()))


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

    single = sweep(os.path.join(RES, "8k1k"))
    fp8_1p1d = sweep(os.path.join(RES, "pd-1p1d"))
    nv_1p1d = sweep(os.path.join(RES, "pd-1p1d-nvfp4"))
    nv_2p2d = sweep(os.path.join(RES, "pd-2p2d-nvfp4"))
    dp4 = dp4_sweep()
    r2p2d = rate_rows(os.path.join(RES, "rate-2p2d-nvfp4"))
    rdp4 = dp4_rate()
    cap = capacity()
    ev = gsm8k()
    base = {os.path.basename(f)[6:-4]: bench(f) for f in glob.glob(os.path.join(RES, "bench-in*.txt"))}

    summary = {"single": single, "fp8_1p1d": fp8_1p1d, "nv_1p1d": nv_1p1d, "nv_2p2d": nv_2p2d, "dp4": dp4,
               "rate_2p2d": r2p2d, "rate_dp4": rdp4, "capacity": cap, "gsm8k": {k: v for k, v in ev.items() if k != "paired"}}
    json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=1, default=str)

    # Chart 1: throughput per GPU vs TPOT (the latency/throughput frontier)
    def pts(rows, gpus, skip=()):
        return [(r["tpot"], r["out"] / gpus, f"并发 {c}，{r['out']:.0f} tok/s 整体，TPOT {r['tpot']:.1f} ms")
                for c, r in rows.items() if c not in skip and r["out"] == r["out"]]
    frontier = line_chart([
        ("FP8 单机 TP8（8 卡）", C["magenta"], pts(single, 8), True),
        ("FP8 1P1D（16 卡）", C["blue"], pts(fp8_1p1d, 16), False),
        ("NVFP4 1P1D（16 卡）", C["orange"], pts(nv_1p1d, 16), False),
        ("NVFP4 不分离 4×TP4（16 卡）", C["aqua"], pts(dp4, 16), False),
        ("NVFP4 2P2D 全 TP4（16 卡）", C["yellow"], pts(nv_2p2d, 16, skip=(1,)), False),
    ], "平均 TPOT（ms，越左越快）", "每卡输出 tok/s", xmax=140)

    # Chart 2: P99 ITL vs concurrency (stream smoothness)
    def itl(rows, skip=()):
        return [(c, r["itl_p99"], f"并发 {c}：P99 ITL {r['itl_p99']:.0f} ms") for c, r in rows.items() if c not in skip and c >= 4]
    itlc = line_chart([
        ("NVFP4 不分离 4×TP4", C["aqua"], itl(dp4), False),
        ("NVFP4 2P2D", C["yellow"], itl(nv_2p2d, skip=(1,)), False),
        ("NVFP4 1P1D", C["orange"], itl(nv_1p1d), False),
    ], "总并发（对数刻度）", "P99 ITL（ms）", logx=True, xticks=[4, 8, 16, 32, 64, 128, 256])

    # Chart 3: TTFT median vs request rate
    if r2p2d and rdp4:
        ttftc = line_chart([
            ("NVFP4 不分离 4×TP4", C["aqua"], [(r, v["ttft_med"] / 1000, f"{r} req/s：TTFT 中位数 {v['ttft_med']/1000:.2f} s") for r, v in rdp4.items()], False),
            ("NVFP4 2P2D", C["yellow"], [(r, v["ttft_med"] / 1000, f"{r} req/s：TTFT 中位数 {v['ttft_med']/1000:.2f} s") for r, v in r2p2d.items()], False),
        ], "到达速率（请求/秒，泊松分布）", "TTFT 中位数（s）", xticks=[0.5, 1, 1.5, 2, 2.5])
    else:
        ttftc = '<p class="note">速率压测进行中。</p>'

    # Chart 4: FP8 vs NVFP4 1P1D throughput by concurrency
    lv = [1, 4, 8, 16, 32, 64, 128]
    fpbars = bar_chart([str(c) for c in lv], [
        ("FP8 1P1D", C["blue"], [fp8_1p1d[c]["out"] for c in lv]),
        ("NVFP4 1P1D", C["orange"], [nv_1p1d[c]["out"] for c in lv]),
    ], "输出 tok/s（16 卡）")

    # Chart 5: capacity
    tp8 = cap.get("cap-nvfp4-tp8", {}); tp4 = cap.get("cap-nvfp4-tp4-8001", {})
    pre8 = max(v["total"] for v in tp8["prefill"].values()) if tp8 else float("nan")
    pre4 = max(v["total"] for v in tp4["prefill"].values()) if tp4 else float("nan")
    pre4b = max(v["total"] for v in cap["cap-nvfp4-tp4-8002"]["prefill"].values()) if "cap-nvfp4-tp4-8002" in cap else float("nan")
    capbars = bar_chart(["每个实例", "每台机器（8 卡）"], [
        ("1 × TP8", C["blue"], [pre8, pre8]),
        ("TP4（每台 2 个）", C["orange"], [pre4, pre4 + pre4b]),
    ], "prefill 吞吐（输入 tok/s）", h=280)

    # GSM8K
    g8, g4 = ev.get("fp8", {}), ev.get("nvfp4", {})
    gsm_rows = []
    for k, lab in (("exact_match,strict-match", "strict-match"), ("exact_match,flexible-extract", "flexible-extract")):
        if k in g8 and k in g4:
            se8 = g8[k.replace("exact_match", "exact_match_stderr")]
            se4 = g4[k.replace("exact_match", "exact_match_stderr")]
            gsm_rows.append([lab, f"{g8[k]*100:.2f}% ± {se8*100:.2f}", f"{g4[k]*100:.2f}% ± {se4*100:.2f}", f"{(g4[k]-g8[k])*100:+.2f} 个百分点"])
    paired = ev.get("paired", "")
    pm = re.search(r"\[strict-match\].*?both right=(\d+)\s+only FP8 right=(\d+)\s+only NVFP4 right=(\d+)\s+both wrong=(\d+).*?identical extracted answer: (\d+)/(\d+).*?p = ([\d.]+)", paired, re.S)

    def sw_rows(rows, levels):
        return [[c, f1(r["out"]), f0(r["total"]), f"{1000/r['tpot']:.0f}" if r["tpot"] else "—", s2(r["ttft_med"]), s2(r["ttft_p99"]),
                 f"{r['tpot']:.1f} ms", f"{r['itl_p99']:.0f} ms", f0(r["fail"])] for c, r in rows.items() if c in levels]
    SWH = ["并发", "输出 tok/s", "总 tok/s", "每路 tok/s", "TTFT 中位数", "TTFT P99", "平均 TPOT", "P99 ITL", "失败"]

    # headline numbers
    h_dp4 = dp4[128]; h_2p2d = nv_2p2d[128]; h_1p1d = nv_1p1d[128]
    imp_2p2d = (h_2p2d["out"] / h_1p1d["out"] - 1) * 100
    gap_dp4 = (h_2p2d["out"] / h_dp4["out"] - 1) * 100

    def rate_table(rows):
        return table(["到达速率", "实际完成 req/s", "输出 tok/s", "TTFT 中位数", "TTFT P99", "平均 TPOT", "P99 ITL", "失败"],
                     [[f"{r} req/s", f"{v['rps']:.2f}", f0(v["out"]), s2(v["ttft_med"]), s2(v["ttft_p99"]), f"{v['tpot']:.1f} ms",
                       f"{v['itl_p99']:.0f} ms", f0(v["fail"])] for r, v in rows.items()])

    html = TEMPLATE
    rep = {
        "%%FRONTIER%%": frontier, "%%ITL%%": itlc, "%%TTFT%%": ttftc, "%%FPBARS%%": fpbars, "%%CAPBARS%%": capbars,
        "%%BASE_TABLE%%": table(["场景（输入 / 输出 / 并发）", "输出 tok/s", "TTFT 中位数", "平均 TPOT", "P99 ITL"],
            [[k.replace("in", "").replace("-out", " / ").replace("-c", " / "), f1(v["out"]), s2(v["ttft_med"]), f"{v['tpot']:.2f} ms", f"{v['itl_p99']:.1f} ms"]
             for k, v in sorted(base.items(), key=lambda kv: [int(x) for x in re.findall(r"\d+", kv[0])])]),
        "%%SINGLE_TABLE%%": table(SWH, sw_rows(single, [1, 2, 4, 8, 16, 32, 64, 128])),
        "%%FP8_1P1D_TABLE%%": table(SWH, sw_rows(fp8_1p1d, [1, 2, 4, 8, 16, 32, 64, 128])),
        "%%NV_1P1D_TABLE%%": table(SWH, sw_rows(nv_1p1d, [1, 2, 4, 8, 16, 32, 64, 128])),
        "%%DP4_TABLE%%": table(["总并发"] + SWH[1:3] + SWH[4:], [[c, f1(r["out"]), f0(r["total"]), s2(r["ttft_med"]), s2(r["ttft_p99"]), f"{r['tpot']:.1f} ms", f"{r['itl_p99']:.0f} ms", f0(r["fail"])] for c, r in dp4.items()]),
        "%%2P2D_TABLE%%": table(SWH, sw_rows(nv_2p2d, [1, 2, 4, 8, 16, 32, 64, 128, 256])),
        "%%FP_TABLE%%": table(["并发", "FP8 tok/s", "NVFP4 tok/s", "变化", "FP8 TPOT", "NVFP4 TPOT", "FP8 TTFT 中位数", "NVFP4 TTFT 中位数"],
            [[c, f1(fp8_1p1d[c]["out"]), f1(nv_1p1d[c]["out"]), (lambda v: "0%" if abs(v) < 0.5 else f"{v:+.0f}%")((nv_1p1d[c]['out']/fp8_1p1d[c]['out']-1)*100), f"{fp8_1p1d[c]['tpot']:.1f} ms", f"{nv_1p1d[c]['tpot']:.1f} ms", s2(fp8_1p1d[c]["ttft_med"]), s2(nv_1p1d[c]["ttft_med"])] for c in [1, 2, 4, 8, 16, 32, 64, 128]]),
        "%%CMP_TABLE%%": table(["总并发", "1P1D 输出 tok/s", "2P2D 输出 tok/s", "不分离 4×TP4 输出 tok/s", "2P2D TTFT 中位数", "不分离 TTFT 中位数", "2P2D P99 ITL", "不分离 P99 ITL"],
            [[c, f1(nv_1p1d.get(c, {}).get("out", float("nan"))), f1(nv_2p2d[c]["out"]), f1(dp4[c]["out"]), s2(nv_2p2d[c]["ttft_med"]), s2(dp4[c]["ttft_med"]), f"{nv_2p2d[c]['itl_p99']:.0f} ms", f"{dp4[c]['itl_p99']:.0f} ms"] for c in (16, 32, 64, 128, 256)]),
        "%%CAP_TABLE%%": table(["实例", "prefill 吞吐（8K 输入）", "decode：并发 32", "decode：并发 64", "decode：并发 128", "decode：并发 256"],
            [[lab, f0(max(v["total"] for v in cap[t]["prefill"].values())) + " tok/s"] + [f"{cap[t]['decode'][c]['out']:,.0f} tok/s · {cap[t]['decode'][c]['tpot']:.0f} ms" if c in cap[t]["decode"] else "—" for c in (32, 64, 128, 256)]
             for t, lab in (("cap-nvfp4-tp8", "NVFP4 TP8（8 卡）"), ("cap-nvfp4-tp4-8001", "NVFP4 TP4 #1（4 卡）"), ("cap-nvfp4-tp4-8002", "NVFP4 TP4 #2（4 卡，同机并行）")) if t in cap]),
        "%%TTFT_BLOCK%%": (f'''<div class="fig">{ttftc}<div class="cap">图 5　TTFT 中位数对到达速率（8K 输入 / 1K 输出，16 卡）</div></div>
<h3>2P2D</h3>{rate_table(r2p2d)}<h3>不分离 4×TP4</h3>{rate_table(rdp4)}
<p class="note">不分离的速率测试由 4 个客户端各按总速率的四分之一独立发送；每个速率档使用不同的随机种子，避免重复 prompt 命中前缀缓存。</p>''' if (r2p2d and rdp4) else
            '<div class="callout">按到达速率的 TTFT 测试正在重跑（第一次每档用了同一个随机种子，后面的档命中前缀缓存，数据作废），完成后本页在同一地址更新。在此之前请以并发扫描的 <b>TTFT 中位数</b>为参考：16–128 并发时 2P2D 为 1.4–1.7 s，不分离 4×TP4 为 2.4–2.6 s（第六节表格）。</div>'),
        "%%GSM_TABLE%%": table(["指标", "FP8", "NVFP4", "差值"], gsm_rows),
        "%%PAIRED%%": table(["", "题数"], [["两边都对", pm.group(1)], ["只有 FP8 对", pm.group(2)], ["只有 NVFP4 对", pm.group(3)], ["两边都错", pm.group(4)], ["抽取出的答案完全相同", f"{pm.group(5)} / {pm.group(6)}"]]) if pm else "",
        "%%PVAL%%": pm.group(7) if pm else "—",
        "%%IMP_2P2D%%": f"{imp_2p2d:+.0f}%", "%%GAP_DP4%%": f"{gap_dp4:+.0f}%",
        "%%H2P2D_OUT%%": f0(h_2p2d["out"]), "%%HDP4_OUT%%": f0(h_dp4["out"]), "%%H1P1D_OUT%%": f0(h_1p1d["out"]),
        "%%H2P2D_ITL%%": f"{h_2p2d['itl_p99']:.0f}", "%%HDP4_ITL%%": f"{h_dp4['itl_p99']:.0f}",
        "%%H2P2D_TTFT%%": s2(h_2p2d["ttft_med"]), "%%HDP4_TTFT%%": s2(h_dp4["ttft_med"]),
        "%%PRE8%%": f0(pre8), "%%PRE4%%": f0(pre4), "%%PRE4X2%%": f0(pre4 + pre4b),
    }
    for k, v in rep.items():
        html = html.replace(k, str(v))
    open(a.out, "w").write(html)
    print("wrote", a.out)


TEMPLATE = open(os.path.join(HERE, "template.html")).read()

if __name__ == "__main__":
    main()
