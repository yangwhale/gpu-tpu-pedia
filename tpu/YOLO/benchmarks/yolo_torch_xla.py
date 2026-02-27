#!/usr/bin/env python3
"""YOLO11n torch_xla baseline benchmark on TPU.

This script measures YOLO forward pass performance using torch_xla's
lazy tensor mode. It serves as the baseline for comparison with torchax.

Usage:
    python yolo_torch_xla.py

Requirements:
    pip install torch torch_xla ultralytics
    pip install 'torch_xla[tpu]' -f https://storage.googleapis.com/libtpu-releases/index.html
"""

import time
import json
import warnings
import torch

warnings.filterwarnings("ignore")

import torch_xla
import torch_xla.core.xla_model as xm
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
import numpy as np
from PIL import Image


def main():
    print("=" * 60)
    print("YOLO11n Benchmark: torch_xla on TPU")
    print("=" * 60)

    # Setup device
    device = torch_xla.device()
    print(f"Device: {device}")

    # Load model
    model = YOLO("yolo11n.pt")
    model.model.eval()
    model.model.fuse()
    model.model.to(device)
    print(f"Model loaded and fused: {sum(p.numel() for p in model.model.parameters())} params")

    # Prepare input
    img_np = np.array(Image.open("bus.jpg"))
    letterbox = LetterBox((640, 640), auto=True, stride=32)
    img_lb = letterbox(image=img_np)
    img_t = torch.from_numpy(img_lb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_xla = img_t.to(device)
    xm.mark_step()
    print(f"Input shape: {img_t.shape}")

    # Warmup (includes XLA graph compilation)
    print("\nWarmup (3 iterations, includes graph compilation)...")
    with torch.no_grad():
        for i in range(3):
            t0 = time.perf_counter()
            out = model.model(img_xla)
            xm.mark_step()
            if isinstance(out, (list, tuple)):
                _ = out[0].cpu()
            else:
                _ = out.cpu()
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"  Warmup {i+1}: {elapsed:.1f} ms")

    # Benchmark
    print("\nBenchmark (10 iterations)...")
    times = []
    with torch.no_grad():
        for i in range(10):
            xm.mark_step()
            t0 = time.perf_counter()
            out = model.model(img_xla)
            xm.mark_step()
            if isinstance(out, (list, tuple)):
                _ = out[0].cpu()
            else:
                _ = out.cpu()
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.1f} ms")

    avg = sum(times) / len(times)
    mn, mx = min(times), max(times)
    print(f"\nResults: avg={avg:.1f}ms, min={mn:.1f}ms, max={mx:.1f}ms")

    # Output JSON for automated comparison
    result = {"method": "torch_xla", "times": times, "avg": avg, "min": mn, "max": mx}
    print(f"\nJSON: {json.dumps(result)}")


if __name__ == "__main__":
    main()
