#!/usr/bin/env python3
# main_tpu.py
"""
TPU GEMM Performance Benchmark - Entry point for TPU backends.

This is a separate entry point from main.py to avoid mixing JAX and PyTorch imports.
TPU benchmarking uses JAX/XLA for native TPU support.

Usage:
    python main_tpu.py --config config/tpu_simple.json
    python main_tpu.py --config config/tpu_gemm.json --output results.csv
"""

import json
import argparse
import time
from datetime import datetime
from typing import Optional
import csv
import os

import jax
import jax.numpy as jnp

from backends.tpu import (
    detect_tpu_backend,
    is_tpu_available,
    parse_jax_dtype,
    get_jax_output_dtype,
    TpuBackendBase,
)


def calculate_performance(m: int, n: int, k: int,
                         input_dtype: jnp.dtype, output_dtype: jnp.dtype,
                         avg_time_us: float) -> tuple:
    """
    Calculate TFLOPS and bandwidth from benchmark results.

    Args:
        m, n, k: Matrix dimensions (C = A @ B, where A is MxK, B is KxN)
        input_dtype: Input data type (for A and B matrices)
        output_dtype: Output data type (for C matrix)
        avg_time_us: Average execution time in microseconds

    Returns:
        Tuple of (tflops, bandwidth_gbps)
    """
    if avg_time_us <= 0:
        return 0.0, 0.0

    avg_time_s = avg_time_us / 1_000_000

    # FLOPS calculation: 2 * M * N * K (multiply-add for each element)
    flops = 2 * m * n * k
    tflops = (flops / 1e12) / avg_time_s

    # Bandwidth calculation: read A, read B, write C
    # Get itemsize for JAX dtypes
    input_itemsize = jnp.dtype(input_dtype).itemsize
    output_itemsize = jnp.dtype(output_dtype).itemsize

    bytes_a = m * k * input_itemsize
    bytes_b = k * n * input_itemsize
    bytes_c = m * n * output_itemsize

    total_data_bytes = bytes_a + bytes_b + bytes_c
    bandwidth_gbps = (total_data_bytes / 1e9) / avg_time_s

    return tflops, bandwidth_gbps


def run_benchmark_from_config(config: dict, backend: TpuBackendBase,
                               output_file: Optional[str] = None) -> list:
    """
    Run GEMM benchmarks based on configuration.

    Args:
        config: Benchmark configuration dictionary
        backend: TPU backend instance
        output_file: Optional CSV file path for results

    Returns:
        List of result dictionaries
    """
    print(f"使用设备: {backend.device_name} (后端: {backend.__class__.__name__})")
    print(f"TPU Generation: {backend.get_tpu_generation()}")
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print()

    results = []

    for case_generator in config["cases"]:
        m_values = case_generator["M"]
        kn_pairs = case_generator["K.N"]
        dtype_strs = case_generator["dtype"]

        for dtype_str in dtype_strs:
            try:
                input_dtype = parse_jax_dtype(dtype_str)
            except ValueError as e:
                print(f"[Warning] 跳过不支持的数据类型 '{dtype_str}': {e}")
                continue

            output_dtype = get_jax_output_dtype(input_dtype)

            print(f"\n测试数据类型: input:{dtype_str} ----> output:{output_dtype}")
            print(f"{'M':>8s} | {'N':>8s} | {'K':>8s} | {'Time (us)':>10s} | {'TFLOPS':>10s} | {'GB/s':>8s} | {'MFU':>6s}")
            print("-" * 80)

            # Get theoretical peak for MFU calculation
            peak_tflops = backend.get_theoretical_peak(dtype_str)

            for k, n in kn_pairs:
                for m in m_values:
                    avg_time_us = backend.run(m, n, k, input_dtype)

                    if avg_time_us > 0:
                        tflops, bandwidth_gbps = calculate_performance(
                            m, n, k, input_dtype, output_dtype, avg_time_us
                        )
                        mfu = (tflops / peak_tflops) * 100 if peak_tflops > 0 else 0

                        print(f"{m:8d} | {n:8d} | {k:8d} | {avg_time_us:10.2f} | {tflops:10.2f} | {bandwidth_gbps:8.2f} | {mfu:5.1f}%")

                        results.append({
                            "timestamp": datetime.now().isoformat(),
                            "device": backend.device_name,
                            "tpu_gen": backend.get_tpu_generation(),
                            "timing_mode": "trace" if backend.use_trace else "legacy",
                            "dtype": dtype_str,
                            "m": m,
                            "n": n,
                            "k": k,
                            "time_us": avg_time_us,
                            "tflops": tflops,
                            "bandwidth_gbps": bandwidth_gbps,
                            "mfu_percent": mfu,
                            "peak_tflops": peak_tflops,
                        })
                    else:
                        print(f"{m:8d} | {n:8d} | {k:8d} | {'Failed':>10s} | {'N/A':>10s} | {'N/A':>8s} | {'N/A':>6s}")

    # Write results to CSV if output file specified
    if output_file and results:
        write_results_csv(results, output_file)
        print(f"\n结果已保存到: {output_file}")

    return results


def write_results_csv(results: list, output_file: str):
    """Write benchmark results to CSV file."""
    if not results:
        return

    fieldnames = results[0].keys()
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_summary(results: list, backend: TpuBackendBase):
    """Print benchmark summary statistics."""
    if not results:
        print("\n没有有效的测试结果。")
        return

    print("\n" + "=" * 80)
    print("测试结果摘要")
    print("=" * 80)

    # Group by dtype
    by_dtype = {}
    for r in results:
        dtype = r["dtype"]
        if dtype not in by_dtype:
            by_dtype[dtype] = []
        by_dtype[dtype].append(r)

    for dtype, dtype_results in by_dtype.items():
        tflops_values = [r["tflops"] for r in dtype_results]
        mfu_values = [r["mfu_percent"] for r in dtype_results]
        bw_values = [r["bandwidth_gbps"] for r in dtype_results]

        peak = dtype_results[0]["peak_tflops"]

        print(f"\n{dtype}:")
        print(f"  理论峰值: {peak:.1f} TFLOPS")
        print(f"  实测 TFLOPS: min={min(tflops_values):.2f}, max={max(tflops_values):.2f}, avg={sum(tflops_values)/len(tflops_values):.2f}")
        print(f"  MFU: min={min(mfu_values):.1f}%, max={max(mfu_values):.1f}%, avg={sum(mfu_values)/len(mfu_values):.1f}%")
        print(f"  带宽: min={min(bw_values):.2f}, max={max(bw_values):.2f} GB/s")


def main():
    parser = argparse.ArgumentParser(
        description="TPU GEMM Performance Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main_tpu.py --config config/tpu_simple.json
    python main_tpu.py --config config/tpu_gemm.json --output results.csv
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/tpu_simple.json",
        help="测试配置 JSON 文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 CSV 文件路径（可选）"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="覆盖配置中的 warmup 迭代次数"
    )
    parser.add_argument(
        "--prof-iter",
        type=int,
        default=None,
        help="覆盖配置中的 profiling 迭代次数"
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        default=False,
        help="禁用 trace 模式，使用传统 time.perf_counter() 计时 (包含 Python overhead, MFU 约 65-75%%)"
    )
    args = parser.parse_args()

    # Trace mode is enabled by default, --no-trace disables it
    args.use_trace = not args.no_trace

    # Check TPU availability first
    if not is_tpu_available():
        print("错误: 未检测到 TPU 设备。")
        print("请确保在 TPU VM 上运行此脚本。")
        print(f"当前 JAX 检测到的设备: {jax.devices()}")
        return 1

    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"错误: 配置文件 '{args.config}' 未找到。")
        return 1
    except json.JSONDecodeError:
        print(f"错误: 配置文件 '{args.config}' 格式不正确。")
        return 1

    # Get benchmark settings
    settings = config.get("benchmark_settings", {"warmup_iter": 10, "prof_iter": 100})
    warmup_iter = args.warmup if args.warmup is not None else settings.get("warmup_iter", 10)
    prof_iter = args.prof_iter if args.prof_iter is not None else settings.get("prof_iter", 100)

    # Detect and initialize TPU backend
    use_trace = args.use_trace
    backend = detect_tpu_backend(warmup_iter, prof_iter, use_trace)
    if backend is None:
        print("错误: TPU 后端初始化失败。")
        return 1

    print("=" * 80)
    print("TPU GEMM Performance Benchmark")
    print("=" * 80)
    print(f"配置文件: {args.config}")
    print(f"Warmup 迭代: {warmup_iter}")
    print(f"Profiling 迭代: {prof_iter}")
    print(f"Trace 模式: {'启用 (纯设备执行时间)' if use_trace else '禁用 (包含 Python overhead)'}")
    print()

    # Run benchmarks
    try:
        start_time = time.time()
        results = run_benchmark_from_config(config, backend, args.output)
        elapsed = time.time() - start_time

        # Print summary
        print_summary(results, backend)
        print(f"\n总耗时: {elapsed:.1f} 秒")

    except KeyboardInterrupt:
        print("\n\n测试被用户中断。")
        return 1
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
