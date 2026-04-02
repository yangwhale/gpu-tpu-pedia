#!/usr/bin/env python3
"""RepVGG on TPU: torchax inference example.

This is a complete, self-contained example showing how to run RepVGG
inference on TPU using torchax with proper jax.jit compilation.

RepVGG is simpler than YOLO for TPU porting because:
  - Deploy mode = pure 3x3 Conv2d + ReLU (no branches, no BN, no residuals)
  - No dynamic anchors or data-dependent control flow
  - No need for complex patches

Key steps:
  1. Create RepVGG model in deploy mode (or convert from train mode)
  2. Move to JAX device via torchax
  3. Wrap forward with jax.jit
  4. Enforce float32 precision throughout

Usage:
    python torchax_repvgg_inference.py [--variant RepVGG-A0] [--runs 100]
"""

import argparse
import sys
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

# Add parent dir to path for repvgg.py import
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))


def patch_conv2d_defaults():
    """Patch torchax conv2d to add default arguments if needed.

    torchax v0.0.11's _aten_conv2d may require all arguments positionally.
    This patches the source to add defaults for stride, padding, dilation, groups.
    """
    try:
        import torchax as _tx
        import os
        torchax_path = os.path.dirname(_tx.__file__)
        jaten_file = os.path.join(torchax_path, "ops", "jaten.py")

        with open(jaten_file, "r") as f:
            content = f.read()

        old_sig = """def _aten_conv2d(
  input,
  weight,
  bias,
  stride,
  padding,
  dilation,
  groups,
):"""
        new_sig = """def _aten_conv2d(
  input,
  weight,
  bias=None,
  stride=(1, 1),
  padding=(0, 0),
  dilation=(1, 1),
  groups=1,
):"""
        if old_sig in content:
            content = content.replace(old_sig, new_sig)
            with open(jaten_file, "w") as f:
                f.write(content)
            print("  [patch] conv2d defaults added to torchax")
        else:
            print("  [patch] conv2d already patched or different version")
    except Exception as e:
        print(f"  [patch] conv2d patch skipped: {e}")


def ensure_float32_conv(model):
    """Ensure all Conv2d layers use float32 weights.

    torchax may auto-cast conv inputs to bfloat16 for performance.
    For precision-sensitive applications, we force float32 throughout.
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            if module.weight.dtype != torch.float32:
                module.weight.data = module.weight.data.float()
                if module.bias is not None:
                    module.bias.data = module.bias.data.float()
    return model


def create_deploy_model(variant='RepVGG-A0', num_classes=1000):
    """Create a RepVGG model in deploy mode (inference-ready).

    Deploy mode fuses the multi-branch training architecture
    (3x3 conv + 1x1 conv + identity BN) into single 3x3 conv + bias.
    The result is a pure VGG-style sequential model.

    Args:
        variant: Model variant name (e.g., 'RepVGG-A0', 'RepVGG-B0')
        num_classes: Number of output classes

    Returns:
        Deploy-mode model with fused weights (random init)
    """
    from repvgg import func_dict, repvgg_model_convert

    # Create training model, then convert to deploy
    create_fn = func_dict[variant]
    train_model = create_fn(deploy=False)
    train_model.eval()

    # Convert: fuse all branches into single 3x3 convs
    deploy_model = repvgg_model_convert(train_model, do_copy=True)
    deploy_model.eval()

    return deploy_model


def create_jitted_forward(model, env):
    """Create a jax.jit-compiled forward function.

    Wraps the entire forward pass in jax.jit for XLA graph compilation.
    This eliminates per-op dispatch overhead (critical for torchax performance).

    Args:
        model: RepVGG deploy-mode model on JAX device
        env: torchax environment

    Returns:
        jax.jit-compiled forward function (JAX array in → JAX array out)
    """
    import torchax
    from torchax import interop
    import jax

    def forward_fn(img_jax_array):
        with env:
            img_torchax = torchax.tensor.Tensor(img_jax_array, env=env)
            with torch.no_grad():
                out = model(img_torchax)
            return interop.jax_view(out)

    return jax.jit(forward_fn)


def main():
    parser = argparse.ArgumentParser(description="RepVGG torchax inference on TPU")
    parser.add_argument("--variant", default="RepVGG-A0", help="Model variant")
    parser.add_argument("--input-size", type=int, default=128, help="Input image size")
    parser.add_argument("--runs", type=int, default=100, help="Number of benchmark runs")
    parser.add_argument("--weights", default=None, help="Path to pretrained weights (.pth)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"RepVGG ({args.variant}) Inference on TPU with torchax")
    print("=" * 60)

    # ─────────────────────────────────────────
    # Step 1: Create deploy-mode model (BEFORE torchax)
    # ─────────────────────────────────────────
    # Must create and convert model before torchax.enable_globally()
    # because deepcopy in repvgg_model_convert fails with torchax dispatch
    print(f"\n[1/5] Creating {args.variant} in deploy mode...")
    model = create_deploy_model(args.variant)

    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location='cpu'))
        print(f"  Loaded weights: {args.weights}")

    # Ensure float32 precision
    model = ensure_float32_conv(model)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model: {args.variant} deploy mode ({param_count:,} params)")

    # ─────────────────────────────────────────
    # Step 2: Patch and initialize torchax
    # ─────────────────────────────────────────
    print("[2/5] Patching and initializing torchax...")
    patch_conv2d_defaults()

    import torchax
    from torchax import interop
    import jax

    # Use highest matmul precision: multiple bf16 passes to simulate fp32
    # Critical for precision-sensitive applications
    jax.config.update("jax_default_matmul_precision", "highest")

    torchax.enable_globally()
    env = torchax.default_env()

    devices = jax.devices()
    print(f"  JAX devices: {len(devices)}x {devices[0].device_kind}")

    # ─────────────────────────────────────────
    # Step 3: Move model to JAX device
    # ─────────────────────────────────────────
    print("[3/5] Moving model to JAX device...")
    model.to("jax")

    print(f"  Precision: float32 (enforced)")

    # ─────────────────────────────────────────
    # Step 4: Create jitted forward
    # ─────────────────────────────────────────
    print(f"[4/5] Creating jax.jit compiled forward (input: {args.input_size}x{args.input_size})...")
    jitted_forward = create_jitted_forward(model, env)

    # Create test input (float32)
    img_np = np.random.rand(1, 3, args.input_size, args.input_size).astype(np.float32)
    img_jax = jax.numpy.array(img_np)

    # Warmup (triggers JIT trace + XLA compile)
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

    # ─────────────────────────────────────────
    # Step 5: Benchmark
    # ─────────────────────────────────────────
    print(f"\n[5/5] Running {args.runs} iterations...")
    times = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        out = jitted_forward(img_jax)
        jax.block_until_ready(out)
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

    avg = sum(times) / len(times)
    p50 = sorted(times)[len(times) // 2]
    p99 = sorted(times)[int(len(times) * 0.99)]

    print(f"\n{'─' * 50}")
    print(f"Results ({args.variant}, input={args.input_size}x{args.input_size}, dtype=float32)")
    print(f"{'─' * 50}")
    print(f"  Average:  {avg:.2f} ms")
    print(f"  Median:   {p50:.2f} ms")
    print(f"  P99:      {p99:.2f} ms")
    print(f"  Min:      {min(times):.2f} ms")
    print(f"  Max:      {max(times):.2f} ms")
    print(f"  Compile:  {compile_time:.0f} ms")
    print(f"  Output:   shape={tuple(out.shape)}, dtype={out.dtype}")
    print(f"{'─' * 50}")


if __name__ == "__main__":
    main()
