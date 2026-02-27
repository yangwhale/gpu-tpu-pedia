#!/usr/bin/env python3
"""YOLO11n torchax NAIVE benchmark on TPU.

This demonstrates the WRONG way to use torchax — enable_globally() + direct
model call without JIT compilation. Each op is dispatched individually through
Python → JAX → XLA, resulting in ~22 seconds per forward pass.

Usage:
    python yolo_torchax_naive.py

Requirements:
    pip install torch torchax jax jaxlib ultralytics
"""

import time
import json
import warnings
import torch

warnings.filterwarnings("ignore")

import torchax
import jax
from ultralytics.data.augment import LetterBox
import numpy as np
from PIL import Image


def main():
    print("=" * 60)
    print("YOLO11n Benchmark: torchax NAIVE (eager per-op dispatch)")
    print("=" * 60)
    print("WARNING: This is the SLOW path. ~22 seconds per forward pass.")
    print("         See yolo_torchax_optimized.py for the correct approach.")
    print()

    # Patch torch.arange compatibility
    _orig_arange = torch.arange
    def _patched_arange(*args, **kwargs):
        if "end" in kwargs and "start" not in kwargs and len(args) == 0:
            return _orig_arange(0, kwargs.pop("end"), **kwargs)
        return _orig_arange(*args, **kwargs)
    torch.arange = _patched_arange

    # Enable torchax globally (this is fine, but NOT sufficient for performance)
    torchax.enable_globally()

    # Load model
    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")
    model.to("jax")
    print(f"Model loaded on JAX device")

    # Prepare input
    img_np = np.array(Image.open("bus.jpg"))
    letterbox = LetterBox((640, 640), auto=True, stride=32)
    img_lb = letterbox(image=img_np)
    img_t = torch.from_numpy(img_lb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_jax = img_t.to("jax")
    print(f"Input shape: {img_t.shape}")

    # Warmup
    print("\nWarmup (1 iteration)...")
    with torch.no_grad():
        t0 = time.perf_counter()
        out = model.model(img_jax)
        jax.effects_barrier()
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  Warmup: {elapsed:.0f} ms")

    # Benchmark (5 runs only, since each takes ~22 seconds)
    print("\nBenchmark (5 iterations, ~2 min total)...")
    times = []
    with torch.no_grad():
        for i in range(5):
            t0 = time.perf_counter()
            out = model.model(img_jax)
            jax.effects_barrier()
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.0f} ms")

    avg = sum(times) / len(times)
    mn, mx = min(times), max(times)
    print(f"\nResults: avg={avg:.0f}ms, min={mn:.0f}ms, max={mx:.0f}ms")
    print(f"\nThis is {avg/10:.0f}x slower than torch_xla (~10ms).")
    print("Root cause: each of 200+ ops dispatched individually through Python → JAX → XLA.")

    result = {"method": "torchax_naive", "times": times, "avg": avg, "min": mn, "max": mx}
    print(f"\nJSON: {json.dumps(result)}")


if __name__ == "__main__":
    main()
