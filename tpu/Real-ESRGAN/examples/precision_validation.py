#!/usr/bin/env python3
"""Real-ESRGAN precision validation: CPU (float32) vs TPU (torchax float32 highest).

Uses smaller input size for faster validation. Same model architecture as production.

Usage:
    python precision_validation.py [--input-h 256] [--input-w 192] [--runs 10]
"""

import argparse
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))


def validate_precision(scale=2, num_block=12, input_h=256, input_w=192, num_runs=10):
    from rrdbnet import RRDBNet

    # ── Create model on CPU ──
    print(f"Creating RRDBNet (scale={scale}, blocks={num_block})...")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=scale, num_feat=64,
                    num_block=num_block, num_grow_ch=32)
    model.eval()
    state_dict = model.state_dict()

    # ── CPU inference ──
    print(f"Running CPU reference ({input_w}x{input_h}, float32)...")
    cpu_model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=scale, num_feat=64,
                        num_block=num_block, num_grow_ch=32)
    cpu_model.load_state_dict(state_dict)
    cpu_model.eval()

    test_inputs = [np.random.rand(1, 3, input_h, input_w).astype(np.float32) for _ in range(num_runs)]

    cpu_outputs = []
    for inp in test_inputs:
        with torch.no_grad():
            out = cpu_model(torch.from_numpy(inp))
        cpu_outputs.append(out.numpy())

    # ── TPU inference ──
    print("Running TPU inference (torchax float32 highest)...")
    tpu_model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=scale, num_feat=64,
                        num_block=num_block, num_grow_ch=32)
    tpu_model.load_state_dict(state_dict)
    tpu_model.eval()

    # Patch conv2d
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

    jax.config.update("jax_default_matmul_precision", "highest")
    torchax.enable_globally()
    env = torchax.default_env()

    tpu_model.to("jax")

    def forward_fn(img_jax):
        with env:
            img_tx = torchax.tensor.Tensor(img_jax, env=env)
            with torch.no_grad():
                out = tpu_model(img_tx)
            return interop.jax_view(out)

    jitted_forward = jax.jit(forward_fn)

    # Warmup
    warmup = jax.numpy.array(test_inputs[0])
    out = jitted_forward(warmup)
    jax.block_until_ready(out)

    tpu_outputs = []
    for inp in test_inputs:
        inp_jax = jax.numpy.array(inp)
        out = jitted_forward(inp_jax)
        jax.block_until_ready(out)
        tpu_outputs.append(np.array(out))

    # ── Compare ──
    print(f"\n{'=' * 60}")
    print(f"Precision Report: RRDBNet (scale={scale}, blocks={num_block})")
    print(f"Input: {input_w}x{input_h} → Output: {input_w*scale}x{input_h*scale}")
    print(f"{'=' * 60}")

    all_max = []
    all_mean = []
    all_median = []

    for i, (cpu_out, tpu_out) in enumerate(zip(cpu_outputs, tpu_outputs)):
        diff = np.abs(cpu_out - tpu_out)
        mx = diff.max()
        mn = diff.mean()
        md = np.median(diff)
        all_max.append(mx)
        all_mean.append(mn)
        all_median.append(md)

        if i < 3:
            print(f"\n  Run {i+1}:")
            print(f"    Max diff:    {mx:.6f}")
            print(f"    Mean diff:   {mn:.6f}")
            print(f"    Median diff: {md:.6f}")
            print(f"    CPU range:   [{cpu_out.min():.4f}, {cpu_out.max():.4f}]")
            print(f"    TPU range:   [{tpu_out.min():.4f}, {tpu_out.max():.4f}]")

    print(f"\n{'─' * 60}")
    print(f"Summary across {num_runs} runs:")
    print(f"{'─' * 60}")
    print(f"  Max diff (worst case):  {max(all_max):.6f}")
    print(f"  Mean diff (average):    {np.mean(all_mean):.6f}")
    print(f"  Median diff (typical):  {np.median(all_median):.6f}")

    max_ok = max(all_max) < 0.01
    median_ok = np.median(all_median) < 0.001

    print(f"\n  Precision check:")
    print(f"    Max diff < 0.01:    {'PASS' if max_ok else 'FAIL'} ({max(all_max):.6f})")
    print(f"    Median diff < 0.001: {'PASS' if median_ok else 'FAIL'} ({np.median(all_median):.6f})")
    print(f"{'=' * 60}")

    return {'max_diff': max(all_max), 'mean_diff': float(np.mean(all_mean)),
            'median_diff': float(np.median(all_median)), 'pass': max_ok and median_ok}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--num-block", type=int, default=12)
    parser.add_argument("--input-h", type=int, default=256)
    parser.add_argument("--input-w", type=int, default=192)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()

    result = validate_precision(args.scale, args.num_block, args.input_h, args.input_w, args.runs)
    if result['pass']:
        print("\nPrecision validation PASSED")
    else:
        print("\nPrecision validation FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
