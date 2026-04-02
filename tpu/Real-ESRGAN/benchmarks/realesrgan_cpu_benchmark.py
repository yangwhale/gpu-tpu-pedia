#!/usr/bin/env python3
"""Real-ESRGAN CPU inference benchmark."""

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
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--num-block", type=int, default=12)
    parser.add_argument("--input-h", type=int, default=2048)
    parser.add_argument("--input-w", type=int, default=1536)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    from rrdbnet import RRDBNet

    print(f"Creating RRDBNet (scale={args.scale}, blocks={args.num_block}) on CPU...")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=args.scale, num_feat=64,
                    num_block=args.num_block, num_grow_ch=32)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Params: {param_count:,}")

    img = torch.randn(1, 3, args.input_h, args.input_w, dtype=torch.float32)

    print("Warmup...")
    for _ in range(2):
        with torch.no_grad():
            _ = model(img)

    print(f"Running {args.runs} iterations on CPU ({args.input_w}x{args.input_h})...")
    times = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(img)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        print(f"  Run {i+1:>2}: {elapsed:.1f}ms")

    times_sorted = sorted(times)
    avg = sum(times) / len(times)
    p50 = times_sorted[len(times) // 2]

    out_h, out_w = out.shape[2], out.shape[3]

    print(f"\n{'─' * 60}")
    print(f"CPU Results (RRDBNet scale={args.scale} blocks={args.num_block})")
    print(f"Input: {args.input_w}x{args.input_h} → Output: {out_w}x{out_h}")
    print(f"{'─' * 60}")
    print(f"  Average:  {avg:.1f} ms")
    print(f"  Median:   {p50:.1f} ms")
    print(f"  Min:      {min(times):.1f} ms")
    print(f"  Max:      {max(times):.1f} ms")
    print(f"  Output:   shape={tuple(out.shape)}, dtype={out.dtype}")
    print(f"{'─' * 60}")

    results = {
        "method": "cpu", "scale": args.scale, "num_block": args.num_block,
        "input_size": f"{args.input_w}x{args.input_h}",
        "output_size": f"{out_w}x{out_h}",
        "params": param_count, "dtype": "float32",
        "avg_ms": round(avg, 1), "median_ms": round(p50, 1),
        "min_ms": round(min(times), 1), "max_ms": round(max(times), 1),
    }
    print(f"\n{json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
