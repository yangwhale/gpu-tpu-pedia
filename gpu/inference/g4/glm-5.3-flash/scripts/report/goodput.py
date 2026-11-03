#!/usr/bin/env python3
"""SLA goodput: the most tokens/s per GPU a deployment sustains while every SLA is still met.

Input: Poisson request-rate sweeps (vllm bench serve --request-rate ...) under results/.
For each deployment we walk up the rate sweep, keep the highest rate whose metrics meet the SLA,
and linearly interpolate to the crossing point with the first failing rate.

Usage: python3 scripts/report/goodput.py   (prints tables; build_report.py imports it)
"""
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(os.path.dirname(HERE)), "results")

# 8K input / 1K output.  TTFT is judged at P90 (what most users see in the worst minute),
# TPOT at the mean (it is a per-request average already), stutter at P99 ITL.
SLAS = [
    {"key": "S1", "name": "严格流式", "ttft_p90": 2000, "tpot": 25, "itl_p99": 200,
     "desc": "TTFT P90 ≤ 2 s，TPOT ≤ 25 ms（每路 ≥ 40 tok/s），P99 ITL ≤ 200 ms"},
    {"key": "S2", "name": "标准对话", "ttft_p90": 3000, "tpot": 40, "itl_p99": None,
     "desc": "TTFT P90 ≤ 3 s，TPOT ≤ 40 ms（每路 ≥ 25 tok/s）"},
    {"key": "S3", "name": "宽松", "ttft_p90": 5000, "tpot": 60, "itl_p99": None,
     "desc": "TTFT P90 ≤ 5 s，TPOT ≤ 60 ms（每路 ≥ 17 tok/s）"},
]

# name, result glob (one or more client dirs whose numbers are summed per rate), GPUs, rate scale
# (per-client rate x scale = deployment rate)
DEPLOYMENTS = [
    ("FP8 单机 TP8（不分离）", ["rate-col-fp8-tp8"], 8, 1),
    ("NVFP4 单机 TP8（不分离）", ["rate-col-nvfp4-tp8"], 8, 1),
    ("NVFP4 单机 2×TP4（不分离）", ["rate-col-nvfp4-2xtp4/inst-*"], 8, 2),
    ("NVFP4 两机 4×TP4（不分离）", ["rate-dp4-nvfp4/*/total-*/inst-*"], 16, None),
    ("NVFP4 2P2D（两机，TP4，UCX 默认）", ["rate-2p2d-nvfp4"], 16, 1),
    ("NVFP4 2P2D（两机，TP4，UCX 调优）", ["rate-2p2d-nvfp4-ucx"], 16, 1),
    ("FP8 2P1D（三机，TP8，UCX 默认）", ["rate-2p1d-fp8"], 24, 1),
    ("NVFP4 4P2D（三机，TP4，UCX 调优）", ["rate-4p2d-nvfp4-ucx"], 24, 1),
]


def metric(t, key):
    m = re.search(re.escape(key) + r"[^:\n]*:\s+([\d.]+)", t)
    return float(m.group(1)) if m else float("nan")


def read(f):
    t = open(f).read()
    return {"rps": metric(t, "Request throughput"), "out": metric(t, "Output token throughput"),
            "total": metric(t, "Total token throughput"), "ttft_med": metric(t, "Median TTFT"),
            "ttft_p90": metric(t, "P90 TTFT"), "tpot": metric(t, "Mean TPOT"), "itl_p99": metric(t, "P99 ITL"),
            "fail": metric(t, "Failed requests")}


def sweep(globs, scale):
    """rate -> aggregated metrics (sums for throughput, worst case for latency)."""
    per_rate = {}
    for g in globs:
        if scale is None:  # dp4 layout: .../total-<deployment rate>/inst-*/rate-<client rate>.txt
            for f in glob.glob(os.path.join(RES, g, "rate-*.txt")):
                r = float(re.search(r"total-([\d.]+)", f).group(1))
                per_rate.setdefault(r, []).append(read(f))
        else:
            for f in glob.glob(os.path.join(RES, g, "rate-*.txt")):
                r = round(float(os.path.basename(f)[5:-4]) * scale, 4)
                per_rate.setdefault(r, []).append(read(f))
    out = {}
    for r, xs in sorted(per_rate.items()):
        out[r] = {"rps": sum(x["rps"] for x in xs), "out": sum(x["out"] for x in xs),
                  "total": sum(x["total"] for x in xs), "ttft_med": max(x["ttft_med"] for x in xs),
                  "ttft_p90": max(x["ttft_p90"] for x in xs), "tpot": max(x["tpot"] for x in xs),
                  "itl_p99": max(x["itl_p99"] for x in xs), "fail": sum(x["fail"] if x["fail"] == x["fail"] else 0 for x in xs),
                  "clients": len(xs)}
    return out


def ok(m, sla):
    if m["fail"] > 0:
        return False
    if m["ttft_p90"] > sla["ttft_p90"] or m["tpot"] > sla["tpot"]:
        return False
    if sla["itl_p99"] is not None and m["itl_p99"] > sla["itl_p99"]:
        return False
    return True


def slack(m, sla):
    """>=0 when the SLA holds; the most violated constraint, as a ratio margin."""
    v = [1 - m["ttft_p90"] / sla["ttft_p90"], 1 - m["tpot"] / sla["tpot"]]
    if sla["itl_p99"] is not None:
        v.append(1 - m["itl_p99"] / sla["itl_p99"])
    return min(v)


def goodput(sw, sla):
    """(out tok/s, total tok/s, rate, note) at the SLA edge; None if even the lowest rate fails."""
    rates = sorted(sw)
    best = None
    for i, r in enumerate(rates):
        m = sw[r]
        if ok(m, sla):
            best = (m["out"], m["total"], r, "")
            continue
        if best is not None:  # interpolate between the last passing and this failing rate
            p = sw[rates[i - 1]]
            s0, s1 = slack(p, sla), slack(m, sla)
            f = s0 / (s0 - s1) if s0 != s1 else 0
            best = (p["out"] + f * (m["out"] - p["out"]), p["total"] + f * (m["total"] - p["total"]),
                    rates[i - 1] + f * (r - rates[i - 1]), "插值")
        return best
    if best is not None:
        best = best[:3] + ("≥（最高速率仍满足）",)
    return best


def table():
    rows = []
    for name, globs, gpus, scale in DEPLOYMENTS:
        sw = sweep(globs, scale)
        if not sw:
            rows.append((name, gpus, None))
            continue
        rows.append((name, gpus, {s["key"]: goodput(sw, s) for s in SLAS}))
    return rows


if __name__ == "__main__":
    for name, gpus, g in table():
        if g is None:
            print(f"{name}: no data")
            continue
        cells = []
        for s in SLAS:
            x = g[s["key"]]
            cells.append("fail" if x is None else f"{x[0]/gpus:6.1f} out/GPU {x[1]/gpus:7.0f} tot/GPU @{x[2]:.2f}rps {x[3]}")
        print(f"{name} ({gpus} GPU): " + " | ".join(cells))
