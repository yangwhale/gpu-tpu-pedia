#!/usr/bin/env python3
"""Real-ESRGAN RRDBNet on TPU: torchax inference example.

Ports Real-ESRGAN (RRDBNet, scale=2, num_block=12) from GPU L40S to TPU v6e.

Key challenges vs RepVGG:
  - Dense Block has 5-layer concat chains (XLA compiler needs to handle)
  - F.interpolate(mode='nearest') for upsampling
  - pixel_unshuffle (view + permute + reshape)
  - Much larger compute (~42 GFLOPS for 1536x2048 input)
  - inplace LeakyReLU must be converted to non-inplace

Usage:
    python torchax_realesrgan_inference.py [--input-h 2048] [--input-w 1536] [--runs 20]
"""

import argparse
import sys
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))


def patch_conv2d_defaults():
    """Patch torchax conv2d to add default arguments if needed."""
    try:
        import torchax as _tx
        import os
        torchax_path = os.path.dirname(_tx.__file__)
        jaten_file = os.path.join(torchax_path, "ops", "jaten.py")
        with open(jaten_file, "r") as f:
            content = f.read()
        old_sig = "def _aten_conv2d(\n  input,\n  weight,\n  bias,\n  stride,\n  padding,\n  dilation,\n  groups,\n):"
        new_sig = "def _aten_conv2d(\n  input,\n  weight,\n  bias=None,\n  stride=(1, 1),\n  padding=(0, 0),\n  dilation=(1, 1),\n  groups=1,\n):"
        if old_sig in content:
            content = content.replace(old_sig, new_sig)
            with open(jaten_file, "w") as f:
                f.write(content)
            print("  [patch] conv2d defaults added")
        else:
            print("  [patch] conv2d already patched")
    except Exception as e:
        print(f"  [patch] skipped: {e}")


def create_model(scale=2, num_block=12, num_feat=64, num_grow_ch=32):
    """Create RRDBNet model for Real-ESRGAN inference.

    Args:
        scale: Upsampling factor (2 for 2x super-resolution)
        num_block: Number of RRDB blocks (12 = lightweight variant)
        num_feat: Feature channels
        num_grow_ch: Growth channels per dense layer

    Returns:
        RRDBNet model in eval mode
    """
    from rrdbnet import RRDBNet
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        scale=scale, num_feat=num_feat,
        num_block=num_block, num_grow_ch=num_grow_ch
    )
    model.eval()
    return model


def create_jitted_forward(model, env):
    """Create a jax.jit-compiled forward function."""
    import torchax
    from torchax import interop
    import jax

    def forward_fn(img_jax_array):
        with env:
            img_tx = torchax.tensor.Tensor(img_jax_array, env=env)
            with torch.no_grad():
                out = model(img_tx)
            return interop.jax_view(out)

    return jax.jit(forward_fn)


def main():
    parser = argparse.ArgumentParser(description="Real-ESRGAN torchax inference on TPU")
    parser.add_argument("--scale", type=int, default=2, help="Upsampling factor")
    parser.add_argument("--num-block", type=int, default=12, help="Number of RRDB blocks")
    parser.add_argument("--input-h", type=int, default=2048, help="Input image height")
    parser.add_argument("--input-w", type=int, default=1536, help="Input image width")
    parser.add_argument("--runs", type=int, default=20, help="Number of benchmark runs")
    parser.add_argument("--weights", default=None, help="Path to pretrained weights")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Real-ESRGAN (RRDBNet, scale={args.scale}, blocks={args.num_block})")
    print(f"Input: {args.input_w}x{args.input_h} → Output: {args.input_w*args.scale}x{args.input_h*args.scale}")
    print("=" * 60)

    # ─── Step 1: Create model BEFORE torchax ───
    print(f"\n[1/5] Creating RRDBNet (scale={args.scale}, num_block={args.num_block})...")
    model = create_model(scale=args.scale, num_block=args.num_block)

    if args.weights:
        state_dict = torch.load(args.weights, map_location='cpu')
        if 'params_ema' in state_dict:
            state_dict = state_dict['params_ema']
        elif 'params' in state_dict:
            state_dict = state_dict['params']
        model.load_state_dict(state_dict, strict=True)
        print(f"  Loaded weights: {args.weights}")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Params: {param_count:,}")

    # ─── Step 2: Initialize torchax ───
    print("[2/5] Initializing torchax...")
    patch_conv2d_defaults()

    import torchax
    from torchax import interop
    import jax

    jax.config.update("jax_default_matmul_precision", "highest")
    torchax.enable_globally()
    env = torchax.default_env()

    devices = jax.devices()
    print(f"  JAX devices: {len(devices)}x {devices[0].device_kind}")
    print(f"  Precision: float32 (highest)")

    # ─── Step 3: Move to TPU ───
    print("[3/5] Moving model to JAX device...")
    model.to("jax")

    # ─── Step 4: JIT compile ───
    print(f"[4/5] Creating jax.jit forward ({args.input_w}x{args.input_h})...")
    jitted_forward = create_jitted_forward(model, env)

    img_jax = jax.numpy.ones((1, 3, args.input_h, args.input_w), dtype=jax.numpy.float32)

    print("\n  Warmup (JIT trace + XLA compile)...")
    t0 = time.perf_counter()
    out = jitted_forward(img_jax)
    jax.block_until_ready(out)
    compile_time = (time.perf_counter() - t0) * 1000
    print(f"  Warmup 1: {compile_time:.0f}ms (includes compilation)")

    t0 = time.perf_counter()
    out = jitted_forward(img_jax)
    jax.block_until_ready(out)
    print(f"  Warmup 2: {(time.perf_counter() - t0) * 1000:.2f}ms (cached)")

    # ─── Step 5: Benchmark ───
    print(f"\n[5/5] Running {args.runs} iterations...")
    times = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        out = jitted_forward(img_jax)
        jax.block_until_ready(out)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        print(f"  Run {i+1:>2}: {elapsed:.2f}ms")

    avg = sum(times) / len(times)
    p50 = sorted(times)[len(times) // 2]
    p99 = sorted(times)[int(len(times) * 0.99)]

    out_h, out_w = out.shape[2], out.shape[3]

    print(f"\n{'─' * 60}")
    print(f"Results (RRDBNet scale={args.scale} blocks={args.num_block}, "
          f"input={args.input_w}x{args.input_h}, float32)")
    print(f"{'─' * 60}")
    print(f"  Average:  {avg:.2f} ms")
    print(f"  Median:   {p50:.2f} ms")
    print(f"  P99:      {p99:.2f} ms")
    print(f"  Min:      {min(times):.2f} ms")
    print(f"  Max:      {max(times):.2f} ms")
    print(f"  Compile:  {compile_time:.0f} ms")
    print(f"  Output:   shape=(1, 3, {out_h}, {out_w}), dtype={out.dtype}")
    print(f"  Scale:    {args.input_w}x{args.input_h} → {out_w}x{out_h}")
    print(f"{'─' * 60}")


if __name__ == "__main__":
    main()
