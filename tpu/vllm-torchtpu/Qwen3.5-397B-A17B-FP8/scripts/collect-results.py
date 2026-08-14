#!/usr/bin/env python3
"""从 4 个 pod 收 benchmark 结果，与 baseline 对比。

用法: python3 collect-results.py --context CTX --ns default [--hours 4] [--out results]

⚠️ 必须按时间过滤。结果目录里会混进上一轮的 run 目录 ——
run-config.sh 的后台 copier 是 `cp -r benchmark_runs/*`，会把历史全拷进去，
不过滤就会把旧数当新数用。
"""
import argparse
import json
import os
import re
import subprocess
import sys

BASE = "scripts/vllm/benchmarking/baselines/perf/qwen3.5-397b-fp8-{}-ep.baseline.json"
CFGS = ["tp1-dp8", "tp2-dp4", "tp8-dp1"]


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", required=True)
    ap.add_argument("--ns", default="default")
    ap.add_argument("--hours", type=int, default=4)
    ap.add_argument("--out", default="results")
    ap.add_argument("--repo", default="vllm-torchtpu", help="vllm-torchtpu 源码目录（取 baseline 用）")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    kube = ["kubectl", "--context", a.context, "-n", a.ns]

    pods = [p.split("/")[1] for p in sh(*kube, "get", "pods", "-o", "name").split()
            if "ttpu16" in p]
    if not pods:
        print("✗ 没找到 ttpu16 的 pod")
        sys.exit(1)

    # 归因必须读同目录的 config.json，不能猜目录名。
    # run 目录名形如 _work_models_qwen3.5-397b_tp8_20260814-024321 —— 只有 tp8 没有 dp，
    # 靠正则从名字里抠 config 会全部落空（实测整列 config 变成 "?"）。
    # config.json 里有 tensor_parallelism / data_parallelism / enable_ep / timestamp，那才是真值。
    seen, n, dup = {}, 0, 0
    for p in pods:
        files = sh(*kube, "exec", p, "--", "bash", "-c",
                   f'find /work/results -name "isl*.json" -newermt "-{a.hours} hours" 2>/dev/null')
        for f in files.split():
            rundir = os.path.dirname(f)
            try:
                cj = json.loads(sh(*kube, "exec", p, "--", "cat", f"{rundir}/config.json"))
            except Exception:
                print(f"  ⚠ 跳过（读不到 config.json）: {f}")
                continue
            cfg = (f"tp{cj['tensor_parallelism']}-dp{cj['data_parallelism']}"
                   + ("-ep" if cj.get("enable_ep") else ""))
            cell = os.path.basename(f)[:-5]
            # 同一份结果会被拷进多个目录，用 (config, cell, timestamp) 去重
            key = (cfg, cell, cj.get("timestamp"))
            if key in seen:
                dup += 1
                continue
            body = sh(*kube, "exec", p, "--", "cat", f)
            if not body:
                continue
            seen[key] = True
            open(f"{a.out}/{cfg}__{cell}__{cj.get('timestamp','na')}.json", "w").write(body)
            n += 1
    print(f"收到 {n} 个结果（去重丢弃 {dup} 个重复）→ {a.out}/")

    base = {}
    for c in CFGS:
        try:
            base[c] = json.load(open(os.path.join(a.repo, BASE.format(c))))
        except Exception:
            pass

    print(f"\n{'config':12s} {'cell':22s} {'完成':>9s} {'out tok/s':>11s} {'vs base':>8s} "
          f"{'TTFT中位':>9s} {'TPOT中位':>9s}")
    rows = incomplete = 0
    for fn in sorted(os.listdir(a.out)):
        if not fn.endswith(".json"):
            continue
        cfg, cell, _ = fn[:-5].split("__")
        d = json.load(open(f"{a.out}/{fn}"))
        b = base.get(cfg.replace("-ep", ""), {}).get(cell)
        delta = ""
        if b:
            v = ((d["output_throughput"] - b["output_token_throughput"])
                 / b["output_token_throughput"] * 100)
            delta = f"{v:+.1f}%"
        ok = d["completed"] == d["num_prompts"]
        if not ok:
            incomplete += 1
        print(f"{cfg:12s} {cell:22s} {d['completed']:4d}/{d['num_prompts']:<4d}{'' if ok else '✗'} "
              f"{d['output_throughput']:11.0f} {delta:>8s} "
              f"{d['median_ttft_ms']:9.0f} {d['median_tpot_ms']:9.2f}")
        rows += 1
    print(f"\n共 {rows} 格，未跑满 {incomplete} 格。"
          f"{'✓ 全部跑满' if incomplete == 0 else '✗ 有 cell 没跑满，结果不可用'}")

    # 同一个 (config, cell) 出现多次时必须告警。
    # config.json 只记录 21 个字段，**不含 EXTRA_SERVE_ARGS**，所以开了投机解码的 run
    # 和没开的在归因上完全一样。不报警的话，使用者会把两者的数混在一起比。
    from collections import defaultdict
    g = defaultdict(list)
    for fn in os.listdir(a.out):
        if fn.endswith(".json"):
            cfg, cell, ts = fn[:-5].split("__")
            g[(cfg, cell)].append((ts, json.load(open(f"{a.out}/{fn}"))))
    amb = {k: v for k, v in g.items() if len(v) > 1}
    if amb:
        print(f"\n⚠ {len(amb)} 个 (config, cell) 有多份结果 —— config.json 不记录 "
              f"EXTRA_SERVE_ARGS，投机解码等差异在归因上是隐形的。")
        print("  这些格必须靠 timestamp 自行区分，不要直接混在一起比：")
        for (cfg, cell), v in sorted(amb.items()):
            for ts, d in sorted(v):
                print(f"    {cfg:12s} {cell:22s} ts={ts}  "
                      f"{d['completed']}/{d['num_prompts']}  {d['output_throughput']:.0f} tok/s")


if __name__ == "__main__":
    main()
