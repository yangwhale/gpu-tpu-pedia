#!/usr/bin/env python3
"""RepVGG torchax benchmark on TPU.

Benchmarks RepVGG inference latency on TPU with torchax + jax.jit.
Outputs results as JSON for automated comparison.

Usage:
    python repvgg_torchax_benchmark.py [--variant RepVGG-A0] [--input-size 128] [--runs 100]
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


def benchmark_torchax(variant, input_size, num_runs, warmup_runs=10):
    """Benchmark RepVGG on TPU with torchax.

    Returns:
        dict with benchmark results
    """
    from repvgg import func_dict, repvgg_model_convert

    # Create deploy model BEFORE torchax to avoid deepcopy issues
    train_model = func_dict[variant](deploy=False)
    train_model.eval()
    deploy_model = repvgg_model_convert(train_model, do_copy=True)
    deploy_model.eval()

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

    # Use highest matmul precision for fp32 accuracy
    jax.config.update("jax_default_matmul_precision", "highest")

    torchax.enable_globally()
    env = torchax.default_env()

    # Move to JAX device
    deploy_model.to("jax")

    param_count = sum(p.numel() for p in deploy_model.parameters())

    # Create jitted forward
    def forward_fn(img_jax):
        with env:
            img_tx = torchax.tensor.Tensor(img_jax, env=env)
            with torch.no_grad():
                out = deploy_model(img_tx)
            return interop.jax_view(out)

    jitted_forward = jax.jit(forward_fn)

    # Input
    img_jax = jax.numpy.ones((1, 3, input_size, input_size), dtype=jax.numpy.float32)

    # Compilation warmup
    t0 = time.perf_counter()
    out = jitted_forward(img_jax)
    jax.block_until_ready(out)
    compile_time_ms = (time.perf_counter() - t0) * 1000

    # Warmup runs
    for _ in range(warmup_runs):
        out = jitted_forward(img_jax)
        jax.block_until_ready(out)

    # Benchmark
    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        out = jitted_forward(img_jax)
        jax.block_until_ready(out)
        times.append((time.perf_counter() - t0) * 1000)

    times_sorted = sorted(times)
    results = {
        "method": "torchax_jit",
        "variant": variant,
        "input_size": input_size,
        "dtype": "float32",
        "params": param_count,
        "device": str(jax.devices()[0].device_kind),
        "num_devices": len(jax.devices()),
        "compile_time_ms": round(compile_time_ms, 1),
        "num_runs": num_runs,
        "avg_ms": round(sum(times) / len(times), 3),
        "median_ms": round(times_sorted[len(times) // 2], 3),
        "p99_ms": round(times_sorted[int(len(times) * 0.99)], 3),
        "min_ms": round(min(times), 3),
        "max_ms": round(max(times), 3),
        "output_shape": list(out.shape),
        "output_dtype": str(out.dtype),
    }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="RepVGG-A0")
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--json-output", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    print(f"Benchmarking {args.variant} (input={args.input_size}x{args.input_size}, runs={args.runs})...")
    results = benchmark_torchax(args.variant, args.input_size, args.runs)

    print(f"\n{'─' * 50}")
    print(f"RepVGG torchax Benchmark Results")
    print(f"{'─' * 50}")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print(f"{'─' * 50}")

    if args.json_output:
        with open(args.json_output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.json_output}")

    # Also print JSON to stdout for piping
    print(f"\n{json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
