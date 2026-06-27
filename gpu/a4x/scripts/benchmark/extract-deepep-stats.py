#!/usr/bin/env python3
"""Extract DeepEP test_ep / intranode / low_latency stats from a rank-0 log.
Output: per-metric { min / max / avg / count } GB/s + latency us."""
import re, sys, os

def stat(values):
    if not values: return None
    return {"min": min(values), "max": max(values), "avg": sum(values)/len(values), "n": len(values)}

def fmt(s):
    return f"{s['min']}-{s['max']} avg={s['avg']:.0f} (n={s['n']})" if s else "-"

PATTERNS = {
    # raw dispatch / combine: e.g. "dispatch: 77 GB/s (SO), 172 GB/s (SU), 246.773 us"
    "dispatch":           r'(?<!expanded )(?<!cached )(?<!reduced )dispatch:\s+(\d+)\s+GB/s\s+\(SO\),\s+(\d+)\s+GB/s\s+\(SU\),\s+([\d.]+)\s+us',
    "expanded dispatch":  r'expanded dispatch:\s+(\d+)\s+GB/s\s+\(SO\),\s+(\d+)\s+GB/s\s+\(SU\),\s+([\d.]+)\s+us',
    "cached dispatch":    r'cached dispatch:\s+(\d+)\s+GB/s\s+\(SO\),\s+(\d+)\s+GB/s\s+\(SU\),\s+([\d.]+)\s+us',
    "combine":            r'(?<!reduced )combine:\s+(\d+)\s+GB/s\s+\(SO\),\s+(\d+)\s+GB/s\s+\(SU\),\s+([\d.]+)\s+us',
    "reduced combine":    r'reduced combine:\s+(\d+)\s+GB/s\s+\(SO\),\s+(\d+)\s+GB/s\s+\(SU\),\s+([\d.]+)\s+us',
}
SIMPLE_PATTERNS = {
    # reduce: 1648 GB/s, 26.724 us | copy: 3745 GB/s, 25.194 us
    "reduce": r'reduce:\s+(\d+)\s+GB/s,\s+([\d.]+)\s+us',
    "copy":   r'copy:\s+(\d+)\s+GB/s,\s+([\d.]+)\s+us',
}
# Intranode pattern: "[tuning] Best dispatch (FP8): SMs 24, NVL chunk 32: 308.11 GB/s, t: 354.04 us"
INTRANODE = re.compile(r'Best (\w+(?:\s+\(FP8\)|\s+\(BF16\))?):\s+SMs\s+\d+,\s+NVL chunk\s+\d+:\s+([\d.]+)\s+GB/s,\s+t:\s+([\d.]+)\s+us')
# Low-latency pattern: "[tuning] Dispatch:  ... US/SU/...  XXX GB/s, t: XX.X us"
LOWLAT = re.compile(r'(Dispatch|Combine):\s+.+?(\d+(?:\.\d+)?)\s+GB/s.+?([\d.]+)\s+us')

def parse(logpath):
    if not os.path.exists(logpath): return None, None
    text = open(logpath, errors='ignore').read()

    pair = {}  # SO/SU/lat triples
    for name, pat in PATTERNS.items():
        matches = re.findall(pat, text)
        if matches:
            so = [int(m[0]) for m in matches]
            su = [int(m[1]) for m in matches]
            lat = [float(m[2]) for m in matches]
            pair[name] = {"SO": stat(so), "SU": stat(su), "us": stat(lat)}

    simple = {}
    for name, pat in SIMPLE_PATTERNS.items():
        matches = re.findall(pat, text)
        if matches:
            bw = [int(m[0]) for m in matches]
            lat = [float(m[1]) for m in matches]
            simple[name] = {"bw": stat(bw), "us": stat(lat)}

    # intranode
    intra = {}
    for m in INTRANODE.finditer(text):
        kind, bw, lat = m.group(1), float(m.group(2)), float(m.group(3))
        intra.setdefault(kind, {"bw": [], "us": []})
        intra[kind]["bw"].append(bw)
        intra[kind]["us"].append(lat)
    intra = {k: {"bw": stat(v["bw"]), "us": stat(v["us"])} for k, v in intra.items()}

    return pair, simple, intra, len(text)

def main():
    files = sys.argv[1:]
    if not files:
        print("usage: extract-deepep-stats.py <log1> [log2 ...]")
        sys.exit(1)
    for f in files:
        result = parse(f)
        if result is None or result[0] is None:
            print(f"\n=== {f} ===\nNOT FOUND")
            continue
        pair, simple, intra, sz = result
        print(f"\n=== {f} (size={sz//1024} KB) ===")
        if pair:
            print("\nTest_ep / Combine metrics:")
            print(f"  {'Metric':22} {'SO GB/s (min-max avg)':28} {'SU GB/s (min-max avg)':28} {'Latency us (min-max avg)':28}")
            for name, d in pair.items():
                so = f"{d['SO']['min']}-{d['SO']['max']} avg={d['SO']['avg']:.0f} (n={d['SO']['n']})"
                su = f"{d['SU']['min']}-{d['SU']['max']} avg={d['SU']['avg']:.0f}"
                lat = f"{d['us']['min']:.1f}-{d['us']['max']:.1f} avg={d['us']['avg']:.1f}"
                print(f"  {name:22} {so:28} {su:28} {lat:28}")
        if simple:
            print("\nReduce / Copy (single metric):")
            for name, d in simple.items():
                bw = f"{d['bw']['min']}-{d['bw']['max']} avg={d['bw']['avg']:.0f} (n={d['bw']['n']})"
                lat = f"{d['us']['min']:.1f}-{d['us']['max']:.1f} avg={d['us']['avg']:.1f}"
                print(f"  {name:22} BW: {bw:35} Latency us: {lat}")
        if intra:
            print("\nIntranode tuning best:")
            for k, d in intra.items():
                if not d['bw']: continue
                bw = f"{d['bw']['min']:.1f}-{d['bw']['max']:.1f} avg={d['bw']['avg']:.1f} (n={d['bw']['n']})"
                lat = f"{d['us']['min']:.1f}-{d['us']['max']:.1f} avg={d['us']['avg']:.1f}"
                print(f"  Best {k:30} BW: {bw:35} Latency us: {lat}")

if __name__ == '__main__':
    main()
