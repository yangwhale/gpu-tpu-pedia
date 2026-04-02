#!/usr/bin/env python3
"""RepVGG CPU inference benchmark.

Measures CPU inference latency for comparison with TPU results.

Usage:
    python repvgg_cpu_benchmark.py [--variant RepVGG-A0] [--input-size 128] [--runs 100]
"""

import argparse
import json
import sys
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="RepVGG-A0")
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    from repvgg import func_dict, repvgg_model_convert

    # Create deploy model
    print(f"Creating {args.variant} deploy model on CPU...")
    train_model = func_dict[args.variant](deploy=False)
    train_model.eval()
    deploy_model = repvgg_model_convert(train_model, do_copy=True)
    deploy_model.eval()

    param_count = sum(p.numel() for p in deploy_model.parameters())
    print(f"  Params: {param_count:,}")

    # Input
    img = torch.randn(1, 3, args.input_size, args.input_size, dtype=torch.float32)

    # Warmup
    print("Warmup...")
    for _ in range(10):
        with torch.no_grad():
            _ = deploy_model(img)

    # Benchmark
    print(f"Running {args.runs} iterations on CPU...")
    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = deploy_model(img)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

    times_sorted = sorted(times)
    avg = sum(times) / len(times)
    p50 = times_sorted[len(times) // 2]
    p99 = times_sorted[int(len(times) * 0.99)]

    print(f"\n{'─' * 50}")
    print(f"CPU Results ({args.variant}, input={args.input_size}x{args.input_size}, float32)")
    print(f"{'─' * 50}")
    print(f"  Average:  {avg:.2f} ms")
    print(f"  Median:   {p50:.2f} ms")
    print(f"  P99:      {p99:.2f} ms")
    print(f"  Min:      {min(times):.2f} ms")
    print(f"  Max:      {max(times):.2f} ms")
    print(f"  Output:   shape={tuple(out.shape)}, dtype={out.dtype}")
    print(f"{'─' * 50}")

    results = {
        "method": "cpu",
        "variant": args.variant,
        "input_size": args.input_size,
        "dtype": "float32",
        "params": param_count,
        "num_runs": args.runs,
        "avg_ms": round(avg, 3),
        "median_ms": round(p50, 3),
        "p99_ms": round(p99, 3),
        "min_ms": round(min(times), 3),
        "max_ms": round(max(times), 3),
        "output_shape": list(out.shape),
    }
    print(f"\n{json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
