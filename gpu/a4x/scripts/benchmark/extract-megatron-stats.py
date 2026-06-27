#!/usr/bin/env python3
"""Parse Megatron-LM training log for iteration timing, throughput, loss.

Megatron core_v0.16 iteration log format:
  iteration       40/      50 | consumed samples:    256 | elapsed time per iteration (ms): 6045.3 | learning rate: 1.234E-04 | global batch size:    256 | lm loss: 11.123 | grad norm: 1.234 | number of skipped iterations:   0 | number of nan iterations:   0 |
  [maybe also]  throughput per GPU (TFLOP/s/GPU): xxx.x | tokens-per-second-per-GPU: yyy.y

Usage: extract-megatron-stats.py <log_file>
       Prints: TSV header + per-iter row + summary (skip warmup first 5 iters)
"""
import re
import sys
from pathlib import Path

if len(sys.argv) != 2:
    print(__doc__)
    sys.exit(1)

log = Path(sys.argv[1]).read_text(errors="replace")

iter_pat = re.compile(
    r'iteration\s+(\d+)/\s*(\d+)\s*\|'
    r'.*?consumed samples:\s*(\d+)\s*\|'
    r'.*?elapsed time per iteration \(ms\):\s*([\d.]+)\s*\|'
    r'(?:.*?learning rate.*?\|)?'
    r'.*?global batch size:\s*(\d+)\s*\|'
    r'.*?lm loss:\s*([\d.E+\-nan]+)',
    re.IGNORECASE
)
tflops_pat = re.compile(
    r'(?:throughput per GPU|TFLOPs/GPU|tflops/sec/gpu|tflops_per_gpu)[^0-9]*([\d.]+)',
    re.IGNORECASE
)
tokens_pat = re.compile(
    r'tokens(?:[-_ ])?per(?:[-_ ])?second(?:[-_ ])?per(?:[-_ ])?GPU[^0-9]*([\d.]+)',
    re.IGNORECASE
)

iters = []
for line in log.splitlines():
    m = iter_pat.search(line)
    if not m:
        continue
    i, total, samples, elapsed_ms, gbs, loss = m.groups()
    row = dict(
        i=int(i), total=int(total), samples=int(samples),
        elapsed_ms=float(elapsed_ms), gbs=int(gbs), loss=loss,
    )
    tf = tflops_pat.search(line)
    tk = tokens_pat.search(line)
    if tf:
        row['tflops_per_gpu'] = float(tf.group(1))
    if tk:
        row['tokens_per_sec_per_gpu'] = float(tk.group(1))
    iters.append(row)

if not iters:
    print("NO_ITER_FOUND")
    # debug: print last 10 non-empty lines
    print("--- last 10 log lines ---")
    for ln in [l for l in log.splitlines() if l.strip()][-10:]:
        print(ln[:200])
    sys.exit(2)

# warmup skip — first 5 iters
warm = 5 if len(iters) > 10 else 0
steady = iters[warm:]

print(f"{'iter':>6} {'ms':>10} {'TFLOPs/GPU':>12} {'tok/s/GPU':>12} {'loss':>10}")
for r in iters:
    print(f"{r['i']:>6} {r['elapsed_ms']:>10.1f} "
          f"{r.get('tflops_per_gpu', float('nan')):>12.2f} "
          f"{r.get('tokens_per_sec_per_gpu', float('nan')):>12.2f} "
          f"{str(r['loss']):>10}")

if steady:
    avg_ms = sum(r['elapsed_ms'] for r in steady) / len(steady)
    tflops_list = [r['tflops_per_gpu'] for r in steady if 'tflops_per_gpu' in r]
    tokens_list = [r['tokens_per_sec_per_gpu'] for r in steady if 'tokens_per_sec_per_gpu' in r]
    print()
    print(f"=== Summary (skip first {warm}/{len(iters)} iters) ===")
    print(f"steady iter count: {len(steady)}")
    print(f"avg ms/iter      : {avg_ms:.1f}")
    if tflops_list:
        print(f"avg TFLOPs/GPU   : {sum(tflops_list)/len(tflops_list):.2f}")
        print(f"max TFLOPs/GPU   : {max(tflops_list):.2f}")
    if tokens_list:
        print(f"avg tokens/s/GPU : {sum(tokens_list)/len(tokens_list):.2f}")
        print(f"max tokens/s/GPU : {max(tokens_list):.2f}")
    print(f"last loss        : {steady[-1]['loss']}")
