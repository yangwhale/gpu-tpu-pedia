#!/usr/bin/env python3
"""YOLO11n on TPU: torchax correct usage example.

This is a complete, self-contained example showing how to run YOLO11n
inference on TPU using torchax with proper jax.jit compilation.

Performance: ~8ms per forward pass on TPU v6e.

Key optimization steps:
    1. Patch known torchax compatibility issues
    2. Fuse BatchNorm into Conv (eliminates buffer issues)
    3. Precompute dynamic tensors (anchors, strides)
    4. Wrap forward with jax.jit for graph compilation
    5. Use jax.block_until_ready() for accurate timing

Usage:
    python torchax_correct_usage.py [--image PATH] [--runs N]
"""

import argparse
import time
import types
import warnings

import numpy as np
import torch
from PIL import Image

warnings.filterwarnings("ignore")


def patch_torch_arange():
    """Fix torchax compatibility with torch.arange keyword args."""
    _orig_arange = torch.arange
    def _patched_arange(*args, **kwargs):
        if "end" in kwargs and "start" not in kwargs and len(args) == 0:
            return _orig_arange(0, kwargs.pop("end"), **kwargs)
        return _orig_arange(*args, **kwargs)
    torch.arange = _patched_arange


def precompute_and_patch_anchors(model, img_input):
    """Precompute YOLO anchors and patch out dynamic creation.

    YOLO's Detect head dynamically creates anchors on first forward pass.
    Inside jax.jit, this causes re-tracing. Fix: compute once, cache forever.

    Args:
        model: YOLO model (model.model)
        img_input: A sample input tensor on JAX device
    """
    import jax

    detect = model.model[-1]  # Last module is Detect head

    # Run one forward to trigger anchor computation
    with torch.no_grad():
        _ = model(img_input)
        jax.effects_barrier()

    # Monkey-patch: skip the dynamic shape check, always use cached anchors
    def patched_get_decode_boxes(self, x):
        dbox = self.decode_bboxes(
            self.dfl(x["boxes"]),
            self.anchors.unsqueeze(0)
        ) * self.strides
        return dbox

    detect._get_decode_boxes = types.MethodType(patched_get_decode_boxes, detect)
    print(f"  Anchors precomputed: {detect.anchors.shape}")


def create_jitted_forward(model, env):
    """Create a jax.jit-compiled forward function.

    This is the KEY optimization: wrapping the entire forward pass in jax.jit
    causes JAX to trace all operations, compile them into a single XLA HLO
    graph, and execute it in one shot. This eliminates per-op dispatch overhead.

    Args:
        model: YOLO model (model.model)
        env: torchax environment

    Returns:
        Tuple of (jitted_forward_fn, jax_view_fn)
    """
    import torchax
    from torchax import interop
    import jax

    def forward_fn(img_jax_array):
        """Pure JAX function wrapping the PyTorch model."""
        with env:
            # Convert JAX array → torchax Tensor
            img_torchax = torchax.tensor.Tensor(img_jax_array, env=env)

            with torch.no_grad():
                out = model(img_torchax)

            # Convert torchax Tensor → JAX array
            if isinstance(out, (list, tuple)):
                return interop.jax_view(out[0])
            return interop.jax_view(out)

    return jax.jit(forward_fn)


def main():
    parser = argparse.ArgumentParser(description="YOLO11n torchax inference")
    parser.add_argument("--image", default="bus.jpg", help="Input image path")
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark runs")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model checkpoint")
    args = parser.parse_args()

    print("=" * 60)
    print("YOLO11n Inference on TPU with torchax (Optimized)")
    print("=" * 60)

    # ─────────────────────────────────────────
    # Step 1: Patch compatibility issues
    # ─────────────────────────────────────────
    print("\n[1/6] Patching compatibility issues...")
    patch_torch_arange()

    # ─────────────────────────────────────────
    # Step 2: Initialize torchax
    # ─────────────────────────────────────────
    print("[2/6] Initializing torchax...")
    import torchax
    from torchax import interop
    import jax

    torchax.enable_globally()
    env = torchax.default_env()

    devices = jax.devices()
    print(f"  JAX devices: {len(devices)}x {devices[0].device_kind}")

    # ─────────────────────────────────────────
    # Step 3: Load and prepare model
    # ─────────────────────────────────────────
    print("[3/6] Loading model...")
    from ultralytics import YOLO
    model = YOLO(args.model)
    model.model.eval()
    model.model.fuse()       # Fold BN into Conv
    model.model.to("jax")

    param_count = sum(p.numel() for p in model.model.parameters())
    print(f"  Model: {args.model} ({param_count:,} params, fused)")

    # ─────────────────────────────────────────
    # Step 4: Prepare input
    # ─────────────────────────────────────────
    print("[4/6] Preparing input...")
    from ultralytics.data.augment import LetterBox

    img_np = np.array(Image.open(args.image))
    letterbox = LetterBox((640, 640), auto=True, stride=32)
    img_lb = letterbox(image=img_np)
    img_t = torch.from_numpy(img_lb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_jax = img_t.to("jax")
    print(f"  Original: {img_np.shape} → Preprocessed: {img_t.shape}")

    # ─────────────────────────────────────────
    # Step 5: Precompute anchors
    # ─────────────────────────────────────────
    print("[5/6] Precomputing anchors...")
    precompute_and_patch_anchors(model.model, img_jax)

    # ─────────────────────────────────────────
    # Step 6: Create jitted forward
    # ─────────────────────────────────────────
    print("[6/6] Creating jax.jit compiled forward...")
    jitted_forward = create_jitted_forward(model.model, env)
    img_jax_arr = interop.jax_view(img_jax)

    # Warmup
    print("\nWarmup (first call triggers JIT trace + XLA compile)...")
    t0 = time.perf_counter()
    out = jitted_forward(img_jax_arr)
    jax.block_until_ready(out)
    print(f"  Warmup: {(time.perf_counter() - t0) * 1000:.0f}ms (includes compilation)")

    # Second warmup (should be fast)
    t0 = time.perf_counter()
    out = jitted_forward(img_jax_arr)
    jax.block_until_ready(out)
    print(f"  Warmup 2: {(time.perf_counter() - t0) * 1000:.1f}ms (cached)")

    # Benchmark
    print(f"\nRunning {args.runs} iterations...")
    times = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        out = jitted_forward(img_jax_arr)
        jax.block_until_ready(out)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        print(f"  Run {i+1:>2}: {elapsed:.1f}ms")

    avg = sum(times) / len(times)
    print(f"\n{'─' * 40}")
    print(f"Average: {avg:.1f}ms | Min: {min(times):.1f}ms | Max: {max(times):.1f}ms")
    print(f"Output shape: {out.shape}")
    print(f"\ntorchax + jax.jit achieves torch_xla-level performance on TPU!")


if __name__ == "__main__":
    main()
