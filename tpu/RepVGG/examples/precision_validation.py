#!/usr/bin/env python3
"""RepVGG precision validation: CPU (float32 ground truth) vs TPU (torchax float32).

Validates that TPU inference produces numerically close results to CPU,
ensuring float32 precision is maintained through the torchax pipeline.

This is critical for precision-sensitive applications where bfloat16
degradation is not acceptable.

Usage:
    python precision_validation.py [--variant RepVGG-A0] [--input-size 128] [--runs 10]
"""

import argparse
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))


def validate_precision(variant='RepVGG-A0', input_size=128, num_runs=10):
    """Compare CPU vs TPU outputs for numerical precision.

    Args:
        variant: RepVGG model variant
        input_size: Input image dimension
        num_runs: Number of random inputs to test

    Returns:
        dict with precision metrics
    """
    from repvgg import func_dict, repvgg_model_convert

    # ── Step 1: Create deploy model on CPU ──
    print(f"Creating {variant} deploy model...")
    train_model = func_dict[variant](deploy=False)
    train_model.eval()
    deploy_model = repvgg_model_convert(train_model, do_copy=True)
    deploy_model.eval()

    # Save state dict for identical weights
    state_dict = deploy_model.state_dict()

    # ── Step 2: CPU reference inference ──
    print("Running CPU reference inference (float32)...")
    cpu_model = func_dict[variant](deploy=True)
    cpu_model.load_state_dict(state_dict)
    cpu_model.eval()

    # Generate random test inputs
    test_inputs = [
        np.random.rand(1, 3, input_size, input_size).astype(np.float32)
        for _ in range(num_runs)
    ]

    cpu_outputs = []
    for inp in test_inputs:
        with torch.no_grad():
            out = cpu_model(torch.from_numpy(inp))
        cpu_outputs.append(out.numpy())

    # ── Step 3: TPU inference with torchax ──
    print("Running TPU inference (torchax float32)...")

    # Create TPU model BEFORE torchax.enable_globally() to avoid deepcopy issues
    tpu_model = func_dict[variant](deploy=True)
    tpu_model.load_state_dict(state_dict)
    tpu_model.eval()

    # Patch conv2d defaults
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
    except Exception:
        pass

    import torchax
    from torchax import interop
    import jax

    # Use highest matmul precision: multiple bf16 passes to simulate fp32
    # This is the "arithmetic simulation" mode on TPU
    jax.config.update("jax_default_matmul_precision", "highest")

    torchax.enable_globally()
    env = torchax.default_env()

    # Now move to JAX device
    tpu_model.to("jax")

    # Create jitted forward
    def forward_fn(img_jax):
        with env:
            img_tx = torchax.tensor.Tensor(img_jax, env=env)
            with torch.no_grad():
                out = tpu_model(img_tx)
            return interop.jax_view(out)

    jitted_forward = jax.jit(forward_fn)

    # Warmup
    warmup_jax = jax.numpy.array(test_inputs[0])
    out = jitted_forward(warmup_jax)
    jax.block_until_ready(out)

    tpu_outputs = []
    for inp in test_inputs:
        inp_jax = jax.numpy.array(inp)
        out = jitted_forward(inp_jax)
        jax.block_until_ready(out)
        tpu_outputs.append(np.array(out))

    # ── Step 4: Compare precision ──
    print(f"\n{'═' * 60}")
    print(f"Precision Report: {variant} (input={input_size}x{input_size})")
    print(f"{'═' * 60}")

    all_max_diffs = []
    all_mean_diffs = []
    all_median_diffs = []

    for i, (cpu_out, tpu_out) in enumerate(zip(cpu_outputs, tpu_outputs)):
        diff = np.abs(cpu_out - tpu_out)
        max_diff = diff.max()
        mean_diff = diff.mean()
        median_diff = np.median(diff)
        all_max_diffs.append(max_diff)
        all_mean_diffs.append(mean_diff)
        all_median_diffs.append(median_diff)

        if i < 3:  # Show first 3 detailed
            print(f"\n  Run {i+1}:")
            print(f"    Max diff:    {max_diff:.6f}")
            print(f"    Mean diff:   {mean_diff:.6f}")
            print(f"    Median diff: {median_diff:.6f}")
            print(f"    CPU range:   [{cpu_out.min():.4f}, {cpu_out.max():.4f}]")
            print(f"    TPU range:   [{tpu_out.min():.4f}, {tpu_out.max():.4f}]")

    print(f"\n{'─' * 60}")
    print(f"Summary across {num_runs} runs:")
    print(f"{'─' * 60}")
    print(f"  Max diff (worst case):  {max(all_max_diffs):.6f}")
    print(f"  Mean diff (average):    {np.mean(all_mean_diffs):.6f}")
    print(f"  Median diff (typical):  {np.median(all_median_diffs):.6f}")

    max_threshold = 0.01
    median_threshold = 0.001
    max_ok = max(all_max_diffs) < max_threshold
    median_ok = np.median(all_median_diffs) < median_threshold

    print(f"\n  Precision check:")
    print(f"    Max diff < {max_threshold}:    {'PASS ✓' if max_ok else 'FAIL ✗'} ({max(all_max_diffs):.6f})")
    print(f"    Median diff < {median_threshold}: {'PASS ✓' if median_ok else 'FAIL ✗'} ({np.median(all_median_diffs):.6f})")
    print(f"{'═' * 60}")

    return {
        'variant': variant,
        'input_size': input_size,
        'max_diff': max(all_max_diffs),
        'mean_diff': float(np.mean(all_mean_diffs)),
        'median_diff': float(np.median(all_median_diffs)),
        'pass': max_ok and median_ok,
    }


def main():
    parser = argparse.ArgumentParser(description="RepVGG precision validation")
    parser.add_argument("--variant", default="RepVGG-A0")
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    result = validate_precision(args.variant, args.input_size, args.runs)

    if result['pass']:
        print("\n✓ Precision validation PASSED")
    else:
        print("\n✗ Precision validation FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
