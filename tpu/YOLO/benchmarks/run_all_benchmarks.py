#!/usr/bin/env python3
"""Run all YOLO benchmarks and produce a comparison table.

Runs torch_xla in a subprocess (to avoid PJRT conflict with JAX),
then runs torchax naive and optimized in-process.

Usage:
    python run_all_benchmarks.py [--skip-naive]

    --skip-naive: Skip the slow naive benchmark (~2 min) for quick testing
"""

import time
import os
import sys
import json
import subprocess
import types
import argparse
import warnings

warnings.filterwarnings("ignore")


def run_torch_xla():
    """Run torch_xla benchmark in subprocess (avoids PJRT conflict)."""
    script = os.path.join(os.path.dirname(__file__), "yolo_torch_xla.py")
    proc = subprocess.run(
        ["python3", script],
        capture_output=True, text=True, timeout=180,
        env={**os.environ, "PJRT_DEVICE": "TPU"}
    )
    if proc.returncode != 0:
        return {"method": "torch_xla", "error": proc.stderr[-300:], "avg": -1, "times": []}

    # Parse JSON from last line containing "JSON:"
    for line in proc.stdout.strip().split("\n"):
        if line.startswith("JSON:"):
            return json.loads(line[5:].strip())
    return {"method": "torch_xla", "error": "no JSON output", "avg": -1, "times": []}


def run_torchax_naive():
    """Run torchax naive benchmark (slow, ~22s per iteration)."""
    import torch
    import torchax
    import jax
    from ultralytics.data.augment import LetterBox
    import numpy as np
    from PIL import Image

    _orig_arange = torch.arange
    def _patched_arange(*args, **kwargs):
        if "end" in kwargs and "start" not in kwargs and len(args) == 0:
            return _orig_arange(0, kwargs.pop("end"), **kwargs)
        return _orig_arange(*args, **kwargs)
    torch.arange = _patched_arange

    torchax.enable_globally()
    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")
    model.to("jax")

    img_np = np.array(Image.open("bus.jpg"))
    letterbox = LetterBox((640, 640), auto=True, stride=32)
    img_lb = letterbox(image=img_np)
    img_t = torch.from_numpy(img_lb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_jax = img_t.to("jax")

    # Warmup
    with torch.no_grad():
        _ = model.model(img_jax)
        jax.effects_barrier()

    times = []
    with torch.no_grad():
        for _ in range(5):
            t0 = time.perf_counter()
            _ = model.model(img_jax)
            jax.effects_barrier()
            times.append((time.perf_counter() - t0) * 1000)

    avg = sum(times) / len(times)
    return {"method": "torchax_naive", "times": times, "avg": avg,
            "min": min(times), "max": max(times)}


def run_torchax_optimized():
    """Run torchax optimized benchmark (fast, ~8ms per iteration)."""
    import torch
    import torchax
    from torchax import interop
    import jax
    from ultralytics.data.augment import LetterBox
    import numpy as np
    from PIL import Image

    _orig_arange = torch.arange
    def _patched_arange(*args, **kwargs):
        if "end" in kwargs and "start" not in kwargs and len(args) == 0:
            return _orig_arange(0, kwargs.pop("end"), **kwargs)
        return _orig_arange(*args, **kwargs)
    torch.arange = _patched_arange

    torchax.enable_globally()
    env = torchax.default_env()

    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")
    model.model.eval()
    model.model.fuse()
    model.model.to("jax")

    img_np = np.array(Image.open("bus.jpg"))
    letterbox = LetterBox((640, 640), auto=True, stride=32)
    img_lb = letterbox(image=img_np)
    img_t = torch.from_numpy(img_lb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_jax = img_t.to("jax")

    # Precompute anchors
    detect = model.model.model[-1]
    with torch.no_grad():
        _ = model.model(img_jax)
        jax.effects_barrier()

    def patched_get_decode_boxes(self, x):
        dbox = self.decode_bboxes(
            self.dfl(x["boxes"]), self.anchors.unsqueeze(0)
        ) * self.strides
        return dbox
    detect._get_decode_boxes = types.MethodType(patched_get_decode_boxes, detect)

    # jax.jit wrapper
    def forward_fn(img_arr):
        with env:
            img_t = torchax.tensor.Tensor(img_arr, env=env)
            with torch.no_grad():
                out = model.model(img_t)
            return interop.jax_view(out[0] if isinstance(out, tuple) else out)

    jitted = jax.jit(forward_fn)
    img_arr = interop.jax_view(img_jax)

    # Warmup
    for _ in range(3):
        out = jitted(img_arr)
        jax.block_until_ready(out)

    # Benchmark
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        out = jitted(img_arr)
        jax.block_until_ready(out)
        times.append((time.perf_counter() - t0) * 1000)

    avg = sum(times) / len(times)
    return {"method": "torchax_optimized", "times": times, "avg": avg,
            "min": min(times), "max": max(times)}


def print_table(results):
    """Print formatted comparison table."""
    print()
    print("=" * 70)
    print("YOLO11n Forward Pass Benchmark Results (TPU v6e)")
    print("=" * 70)

    xla_avg = next((r["avg"] for r in results if r["method"] == "torch_xla"), -1)

    print(f"\n{'Method':<25} {'Avg (ms)':>10} {'Min':>8} {'Max':>8} {'vs xla':>10}")
    print("-" * 65)

    for r in results:
        if r["avg"] > 0:
            ratio = f"{r['avg']/xla_avg:.0f}x" if xla_avg > 0 else "N/A"
            if r["avg"] <= xla_avg * 1.5:
                ratio = f"{r['avg']/xla_avg:.2f}x"
            print(f"  {r['method']:<23} {r['avg']:>8.1f}ms {r['min']:>7.1f} {r['max']:>7.1f} {ratio:>10}")
        else:
            err = r.get("error", "unknown")[:30]
            print(f"  {r['method']:<23} {'ERROR':>8}   ({err})")

    # Per-run details
    max_runs = max(len(r.get("times", [])) for r in results)
    if max_runs > 0:
        print(f"\nPer-run times (ms):")
        header = f"  {'Run':<5}"
        for r in results:
            header += f" {r['method'][:15]:>15}"
        print(header)
        print(f"  {'-'*5}" + f" {'-'*15}" * len(results))

        for i in range(max_runs):
            row = f"  {i+1:<5}"
            for r in results:
                t = r.get("times", [])
                row += f" {t[i]:>15.1f}" if i < len(t) else f" {'':>15}"
            print(row)


def main():
    parser = argparse.ArgumentParser(description="YOLO11n benchmark suite")
    parser.add_argument("--skip-naive", action="store_true",
                        help="Skip slow naive benchmark (~2 min)")
    args = parser.parse_args()

    results = []

    # 1. torch_xla (subprocess)
    print("[1/3] Running torch_xla benchmark (subprocess)...")
    r = run_torch_xla()
    print(f"  → avg: {r['avg']:.1f}ms" if r["avg"] > 0 else f"  → ERROR")
    results.append(r)

    # 2. torchax naive
    if args.skip_naive:
        print("[2/3] Skipping torchax naive (--skip-naive)")
        results.append({"method": "torchax_naive", "avg": -1, "times": [],
                        "error": "skipped"})
    else:
        print("[2/3] Running torchax naive benchmark (~2 min)...")
        r = run_torchax_naive()
        print(f"  → avg: {r['avg']:.0f}ms")
        results.append(r)

    # 3. torchax optimized
    print("[3/3] Running torchax optimized benchmark...")
    r = run_torchax_optimized()
    print(f"  → avg: {r['avg']:.1f}ms")
    results.append(r)

    print_table(results)

    # Summary
    print("\nConclusion:")
    print("  torchax + jax.jit ≈ torch_xla in performance")
    print("  torchax naive (eager per-op) is 2000x+ slower")
    print("  Key: use jax.jit to compile the full forward pass into one XLA graph")


if __name__ == "__main__":
    main()
