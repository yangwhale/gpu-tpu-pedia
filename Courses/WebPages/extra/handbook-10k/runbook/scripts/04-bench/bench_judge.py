#!/usr/bin/env python3
"""bench_judge.py —— 04 组件验收：把 c1 / c2 / c3 / gcsfuse 的原始日志汇总成表，并按 thresholds.csv 判 PASS / WARN / FAIL。

用法：
  bench_judge.py <目录或日志>... [--shape 4x8x8] [--th thresholds.csv]
    目录 = 04-bench.sh collect 的输出（里面是 pod-*.log）；也可以直接给日志文件。
    自动识别日志里的 C1_ROW / C2_ROW / C3_ROW / GCSFUSE_READ 行，有哪种判哪种。
    --shape 只在 ici1d 模式（c2_ici_bench.py，一维 mesh，info 行没有形状）时需要；c3 的形状从 info 行的 dims 读。
    注意：日志行名 C2_ROW 是 c2_ici_bench.py 的一维 ICI，不是 04 卡片 C2（HBM，在 C1_ROW 的 hbm 行里）。
退出码：0 = 全部 PASS 或 WARN；1 = 有 FAIL；2 = 一行结果都没解析到。

由本书实测时的汇总脚本合并改写。
口径：
  - C1 matmul：all_local = 一台 host 的 8 个 device 同时算，报「每芯片」= 总 FLOP/s ÷ 4；判最慢那台 host。
    同时给两种「占峰值」：① 官方峰值 bf16 2,307 / fp8 4,614 TFLOP/s/芯片（= P7 频率下的数）；
    ② 当前档天花板：default 档 = 官方 × 0.455/0.526 ≈ ×0.865 → 1,996 / 3,991 [推断：按 P3/P7 周期比，
    P7 档单 device 实测 +14.7% 与之吻合]；P7 档 = 官方峰值本身。
  - HBM：y = x*c + 1，字节 = 2 × 数组；官方 7.38 TB/s/芯片（与功耗档无关，P7 档实测不变）。
  - ICI：nccl-tests 口径 busbw；AR = S/t × 2(k−1)/k。reduce_scatter 行不判（XLA 把它改写成了 all-reduce，见 c2/c3 文件头）。
"""
import csv
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PEAK_CHIP = {"bf16": 2307.0, "fp8": 4614.0}
DEFAULT_CEIL = 0.455 / 0.526      # [推断] P3(默认)/P7 周期比 ≈ 0.865
HBM_PEAK = 7.38

args = sys.argv[1:]
shape_arg = None
th_path = os.path.join(HERE, "thresholds.csv")
paths = []
i = 0
while i < len(args):
    if args[i] == "--shape":
        shape_arg = args[i + 1]; i += 2
    elif args[i] == "--th":
        th_path = args[i + 1]; i += 2
    else:
        paths.append(args[i]); i += 1
if not paths:
    sys.exit(__doc__)

files = []
for p in paths:
    files += sorted(glob.glob(os.path.join(p, "pod-*.log"))) if os.path.isdir(p) else [p]

rows = defaultdict(list)     # tag -> [(file, dict)]
gcs = []
for f in files:
    for line in open(f, errors="replace"):
        for tag in ("C1_ROW", "C2_ROW", "C3_ROW"):
            j = line.find(tag + " ")
            if j >= 0:
                try:
                    rows[tag].append((f, json.loads(line[j + len(tag) + 1:])))
                except Exception:
                    pass
        j = line.find("GCSFUSE_READ ")
        if j >= 0:
            kv = dict(x.split("=", 1) for x in line[j + 13:].split() if "=" in x)
            gcs.append(kv)

TH = [t for t in csv.DictReader(l for l in open(th_path) if not l.startswith("#"))]
LV = ["PASS", "WARN", "FAIL"]
worst = 0
report = []


def judge(test, metric, value, shape="*", p_state="*"):
    """返回 (等级, 阈值行)；找不到阈值行 → (None, None)"""
    for t in TH:
        if t["test"] != test or t["metric"] != metric:
            continue
        if t["shape"] not in ("*", shape) or t["p_state"] not in ("*", p_state):
            continue
        ps, wn = float(t["pass"]), float(t["warn"])
        if t["dir"] == "ge":
            lv = 0 if value >= ps else (1 if value >= wn else 2)
        else:
            lv = 0 if value <= ps else (1 if value <= wn else 2)
        return lv, t
    return None, None


def line(name, value_txt, lv, t, extra=""):
    global worst
    if lv is None:
        verdict = "-（无阈值，只记录）"
    else:
        worst = max(worst, lv)
        cmp = "≥" if t["dir"] == "ge" else "≤"
        verdict = f"{LV[lv]}（基线 {t['baseline_lo']}–{t['baseline_hi']}，PASS {cmp} {t['pass']}，WARN {cmp} {t['warn']}）"
    print(f"  {name:38} {value_txt:>28}  {verdict}{extra}")


# ---------------- C1 / C2(HBM) / C5(H2D)：每 host 一份 ----------------
if rows["C1_ROW"]:
    info = [r for _, r in rows["C1_ROW"] if r.get("kind") == "info"]
    pstates = sorted({r.get("p_state", "?") for r in info})
    p_state = pstates[0] if len(pstates) == 1 else "mixed"
    if p_state not in ("default", "7"):
        print(f"注意：功耗档 {pstates}（结果行没打印档位时为 '?'，按 default 判）")
        p_state = "default" if p_state in ("?",) else p_state
    hosts = {r.get("host") for _, r in rows["C1_ROW"]}
    libtpu = sorted({r.get("libtpu", "?") for r in info})
    print(f"== C1 单芯片算力 / C2 HBM / C5 H2D（{len(hosts)} 台不同 host，功耗档 {p_state}，libtpu {libtpu}）")
    agg = defaultdict(list)
    for _, r in rows["C1_ROW"]:
        k = r.get("kind")
        if "error" in r:
            print("  ERROR", r); worst = max(worst, 2); continue
        if k == "matmul" and r.get("scope") == "all_local":
            agg[f"matmul/all_local/{r['dtype']}/{r['n']}"].append(r["tflops_chip"])
        elif k == "matmul" and r.get("scope") == "single_device":
            agg[f"matmul/single_device/{r['dtype']}/{r['n']}"].append(r["tflops_dev"])
        elif k == "hbm" and r.get("scope") == "all_local":
            agg[f"hbm/all_local/{r['gib']}"].append(r["tbps_chip"])
        elif k in ("h2d", "d2h") and "gbps_host" in r:
            agg[f"{k}/{r['scope']}"].append(r["gbps_host"])
    def _k(s):
        p = s.split("/")
        return (p[0], p[1], p[2] if len(p) > 2 else "", int(p[-1]) if p[-1].isdigit() else 0)
    for key in sorted(agg, key=_k):
        v = agg[key]; lo, med, hi = min(v), statistics.median(v), max(v)
        if key.startswith("matmul"):
            dt = key.split("/")[2]
            per = 2 if "single" in key else 1           # single_device 是半个芯片
            peak = PEAK_CHIP[dt] / per
            ceil = peak * (DEFAULT_CEIL if p_state == "default" else 1.0)
            txt = f"{lo:,.0f} / {med:,.0f} / {hi:,.0f}"
            extra = f"  中位占官方峰值 {100*med/peak:.1f}%，占当前档天花板 {100*med/ceil:.1f}%"
        elif key.startswith("hbm"):
            txt = f"{lo:.2f} / {med:.2f} / {hi:.2f}"; extra = f"  中位占 7.38 TB/s 的 {100*med/HBM_PEAK:.1f}%"
        else:
            txt = f"{lo:.1f} / {med:.1f} / {hi:.1f}"; extra = ""
        test = "C1" if key.startswith("matmul") else ("C2" if key.startswith("hbm") else "C5")
        lv, t = judge(test, key, lo, p_state=p_state)       # 判最慢那台
        line(key + "（min/中位/max）", txt, lv, t, extra)
    v = agg.get("matmul/all_local/bf16/32768")
    if v:
        sp = max(v) / min(v)
        lv, t = judge("C1", "spread/all_local/bf16/32768", sp, p_state=p_state)
        line("host 间离散 bf16 N=32768（max/min）", f"{sp:.3f}", lv, t,
             "" if p_state == "default" else "  P7 下离散 5–7% 是正常的（功耗顶住，P7 档实测），不判")
    print("  （matmul 单位 TFLOP/s：all_local = 每芯片，single_device = 每 device；HBM TB/s/芯片；H2D/D2H GB/s/host）")

# ---------------- C3：片内 ICI（process 0 一份） ----------------
for tag in ("C3_ROW", "C2_ROW"):
    if not rows[tag]:
        continue
    by_file = defaultdict(list)
    for f, r in rows[tag]:
        by_file[f].append(r)
    for f, rs in by_file.items():
        info = next((r for r in rs if r.get("kind") == "info"), {})
        shape = "x".join(map(str, info["dims"])) if info.get("dims") else shape_arg
        print(f"== C3 片内 ICI（{os.path.basename(os.path.dirname(f)) or f}：{tag[:2]}，形状 {shape or '未知（ici1d / C2_ROW 请加 --shape）'}，"
              f"{info.get('devices', '?')} device）")
        for r in rs:
            if "error" in r:
                print("  ERROR", r); worst = max(worst, 2)
        if tag == "C3_ROW":
            ar = {(r["group"], r["mib"]): r for r in rs if r.get("kind") == "all_reduce" and "busbw_GBps" in r}
            for g in ("all_1d", "all_nd", "combo_xyz", "axis_x", "axis_y", "axis_z", "axis_c"):
                r = ar.get((g, 1024))
                if not r:
                    continue
                metric = "all_reduce/all_1d/1024" if g == "all_1d" else ("all_reduce/axis/1024" if g in ("axis_x", "axis_y", "axis_z") else None)
                lv, t = judge("C3", metric, r["busbw_GBps"], shape=shape or "?") if metric else (None, None)
                line(f"all-reduce 1 GiB {g}（k={r['k']}）", f"{r['busbw_GBps']:.1f} GB/s", lv, t)
            ag = {(r["group"], r["mib"]): r for r in rs if r.get("kind") == "all_gather" and "busbw_GBps" in r}
            if ("all_1d", 1024) in ag:
                line("all-gather 1 GiB all_1d", f"{ag[('all_1d', 1024)]['busbw_GBps']:.1f} GB/s", None, None,
                     "  （同模式两次可差 20%，只记录）")
            hlo = [r for r in rs if r.get("kind") == "hlo"]
            if hlo:   # C3_HLO=1 时：编译后 HLO 里的集合通信指令。reduce_scatter 用例里出现 all-reduce 就说明被改写了
                rs_bad = sum(1 for r in hlo if r.get("op") == "reduce_scatter" and "reduce-scatter" not in (r.get("collectives") or {}))
                print(f"  HLO：{len(hlo)} 个用例；reduce_scatter 用例里没有 reduce-scatter 指令的 {rs_bad} 个（>0 = 被改写成 all-reduce，RS 行作废）")
        else:
            r = next((r for r in rs if r.get("kind") == "all_reduce" and r.get("mib") == 1024 and "busbw_GBps" in r), None)
            if r:
                lv, t = judge("C3", "all_reduce/all_1d/1024", r["busbw_GBps"], shape=shape or "?")
                line("all-reduce 1 GiB（一维 mesh）", f"{r['busbw_GBps']:.1f} GB/s", lv, t)

# ---------------- C6：gcsfuse ----------------
if gcs:
    byr = defaultdict(list)
    for g in gcs:
        byr[int(g.get("readers", 1))].append(float(g["GBps"]))   # 没有 readers 字段 = 1 路
    print(f"== C6 gcsfuse 读（{len(gcs)} 个 pod）")
    for R, v in sorted(byr.items()):
        metric = f"gcsfuse/readers={R}"
        lv, t = judge("C6", metric, min(v))
        line(f"每 host {R} 路（min/中位/max，合计）", f"{min(v):.2f} / {statistics.median(v):.2f} / {max(v):.2f}，Σ {sum(v):.1f}", lv, t)

if not (rows["C1_ROW"] or rows["C2_ROW"] or rows["C3_ROW"] or gcs):
    print("没有解析到任何 C1_ROW / C2_ROW / C3_ROW / GCSFUSE_READ 行（还没跑完？看的是不是 process 0 那份？）")
    sys.exit(2)
print("总判定：", LV[worst])
sys.exit(1 if worst == 2 else 0)
