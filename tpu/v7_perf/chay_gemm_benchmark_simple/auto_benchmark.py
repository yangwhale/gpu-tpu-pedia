#!/usr/bin/env python3
"""
auto_benchmark.py - Automatic Chip Performance Test

This script automatically detects hardware (GPU/TPU) and runs appropriate benchmarks.
It then generates a performance report in Markdown format.

Usage:
    python auto_benchmark.py                    # Auto-detect and run
    python auto_benchmark.py --quick            # Quick test (fewer configurations)
    python auto_benchmark.py --output report.md # Save report to file
    python auto_benchmark.py --csv results.csv  # Also save raw data to CSV

Example:
    python auto_benchmark.py --quick --output benchmark_report.md
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
import json


# ============================================================================
# Hardware Detection
# ============================================================================

def detect_hardware() -> Dict[str, Any]:
    """Detect available hardware and return info dict."""
    result = {
        "type": None,
        "name": None,
        "count": 0,
        "details": {}
    }

    # Try TPU first (JAX)
    try:
        import jax
        devices = jax.devices('tpu')
        if devices:
            result["type"] = "tpu"
            result["count"] = len(devices)
            result["name"] = "Google TPU"
            result["details"]["jax_version"] = jax.__version__

            # Try to detect TPU version
            device_str = str(devices[0]).lower()
            if 'v7' in device_str or 'ironwood' in device_str:
                result["name"] = "Google TPU v7 (Ironwood)"
                result["details"]["generation"] = "v7"
            elif 'v6' in device_str or 'trillium' in device_str:
                result["name"] = "Google TPU v6e (Trillium)"
                result["details"]["generation"] = "v6e"
            else:
                result["name"] = "Google TPU v6e (Trillium)"  # Default
                result["details"]["generation"] = "v6e"
            return result
    except:
        pass

    # Try GPU (PyTorch CUDA)
    try:
        import torch
        if torch.cuda.is_available():
            result["type"] = "gpu"
            result["count"] = torch.cuda.device_count()
            result["name"] = torch.cuda.get_device_name(0)
            result["details"]["cuda_version"] = torch.version.cuda
            result["details"]["pytorch_version"] = torch.__version__
            return result
    except:
        pass

    # Try nvidia-smi as fallback
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        if output:
            result["type"] = "gpu"
            result["count"] = len(output.split('\n'))
            result["name"] = output.split('\n')[0]
            return result
    except:
        pass

    return result


def get_hardware_specs(hw_info: Dict[str, Any]) -> Dict[str, Any]:
    """Get theoretical peak specs for detected hardware."""
    specs = {
        # TPU v6e (Trillium)
        "Google TPU v6e (Trillium)": {
            "bandwidth_gbs": 1600,
            "bfloat16_tflops": 918,
            "float16_tflops": 918,
            "float32_tflops": 918,  # Uses bf16 compute
            "int8_tops": 1836,
        },
        # TPU v7 (Ironwood) - Per chiplet (JAX device) specs
        # Source: https://docs.cloud.google.com/tpu/docs/tpu7x (2026-02-09)
        # Per chip: BF16 2307 TFLOPS, HBM 7380 GB/s, 192 GiB
        # Per chiplet: BF16 1153.5 TFLOPS, HBM 3690 GB/s, 96 GiB
        "Google TPU v7 (Ironwood)": {
            "bandwidth_gbs": 3690,
            "bfloat16_tflops": 1153.5,
            "float16_tflops": 1153.5,
            "float32_tflops": 1153.5,
            "int8_tops": 2307,
        },
        # NVIDIA GPUs
        "NVIDIA H100": {
            "bandwidth_gbs": 3350,
            "bfloat16_tflops": 990,
            "float16_tflops": 990,
            "float32_tflops": 67,
            "int8_tops": 1980,
        },
        "NVIDIA H20": {
            "bandwidth_gbs": 4000,
            "bfloat16_tflops": 147,
            "float16_tflops": 147,
            "float32_tflops": 40,
            "int8_tops": 293,
        },
        "NVIDIA A100": {
            "bandwidth_gbs": 2039,
            "bfloat16_tflops": 312,
            "float16_tflops": 312,
            "float32_tflops": 19.5,
            "int8_tops": 624,
        },
    }

    hw_name = hw_info.get("name", "")

    # Try exact match first
    if hw_name in specs:
        return specs[hw_name]

    # Try partial match
    for key in specs:
        if key.lower() in hw_name.lower() or hw_name.lower() in key.lower():
            return specs[key]

    # Return unknown specs
    return {
        "bandwidth_gbs": 0,
        "bfloat16_tflops": 0,
        "float16_tflops": 0,
        "float32_tflops": 0,
        "int8_tops": 0,
    }


# ============================================================================
# Benchmark Execution
# ============================================================================

def run_tpu_benchmark(config_file: str, output_csv: Optional[str] = None) -> str:
    """Run TPU benchmark and return output."""
    cmd = ["python3", "main_tpu.py", "--config", config_file]
    if output_csv:
        cmd.extend(["--output", output_csv])

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout + result.stderr


def run_gpu_benchmark(config_file: str) -> str:
    """Run GPU benchmark and return output."""
    cmd = ["python3", "main.py", "--config", config_file]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout + result.stderr


def parse_benchmark_output(output: str) -> Dict[str, List[Dict]]:
    """Parse benchmark output into structured data."""
    results = {}
    current_dtype = None

    for line in output.split('\n'):
        # Detect dtype header
        if 'input:' in line and '---->' in line:
            # Extract dtype from "input:bfloat16 ----> output:..."
            parts = line.split('input:')
            if len(parts) > 1:
                dtype_part = parts[1].split()[0]
                current_dtype = dtype_part
                results[current_dtype] = []

        # Parse data rows (look for numeric patterns)
        elif current_dtype and '|' in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 6:
                try:
                    m = int(parts[0])
                    n = int(parts[1])
                    k = int(parts[2])
                    time_us = float(parts[3])
                    tflops = float(parts[4])
                    gbps = float(parts[5])
                    mfu = float(parts[6].replace('%', '')) if len(parts) > 6 else 0

                    results[current_dtype].append({
                        "m": m, "n": n, "k": k,
                        "time_us": time_us,
                        "tflops": tflops,
                        "bandwidth_gbps": gbps,
                        "mfu_percent": mfu,
                    })
                except (ValueError, IndexError):
                    continue

    return results


# ============================================================================
# Report Generation
# ============================================================================

def generate_report(hw_info: Dict, hw_specs: Dict, results: Dict,
                   output: str) -> str:
    """Generate Markdown performance report."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# Chip Performance Test Report

> Generated: {timestamp}
> Tool: auto_benchmark.py

---

## 1. Hardware Information

| Property | Value |
|----------|-------|
| **Device Type** | {hw_info.get('type', 'Unknown').upper()} |
| **Device Name** | {hw_info.get('name', 'Unknown')} |
| **Device Count** | {hw_info.get('count', 0)} |
"""

    # Add details
    for key, value in hw_info.get('details', {}).items():
        report += f"| {key} | {value} |\n"

    # Add specs
    report += f"""
## 2. Theoretical Peak Performance

| Metric | Value |
|--------|-------|
| HBM Bandwidth | {hw_specs.get('bandwidth_gbs', 'N/A')} GB/s |
| BF16 Peak | {hw_specs.get('bfloat16_tflops', 'N/A')} TFLOPS |
| FP16 Peak | {hw_specs.get('float16_tflops', 'N/A')} TFLOPS |
| FP32 Peak | {hw_specs.get('float32_tflops', 'N/A')} TFLOPS |
| INT8 Peak | {hw_specs.get('int8_tops', 'N/A')} TOPS |

---

## 3. Benchmark Results

"""

    # Generate results table for each dtype
    for dtype, data in results.items():
        if not data:
            continue

        tflops_values = [d['tflops'] for d in data]
        mfu_values = [d['mfu_percent'] for d in data if d['mfu_percent'] > 0]
        bw_values = [d['bandwidth_gbps'] for d in data]

        # Determine peak name based on dtype
        if 'int8' in dtype:
            peak_name = 'int8_tops'
        elif 'bfloat16' in dtype or 'bf16' in dtype:
            peak_name = 'bfloat16_tflops'
        elif 'float16' in dtype or 'fp16' in dtype:
            peak_name = 'float16_tflops'
        else:
            peak_name = 'float32_tflops'

        peak = hw_specs.get(peak_name, 0)

        report += f"""### {dtype.upper()}

| Metric | Min | Max | Average |
|--------|-----|-----|---------|
| TFLOPS | {min(tflops_values):.2f} | {max(tflops_values):.2f} | {sum(tflops_values)/len(tflops_values):.2f} |
| MFU | {min(mfu_values):.1f}% | {max(mfu_values):.1f}% | {sum(mfu_values)/len(mfu_values):.1f}% |
| Bandwidth (GB/s) | {min(bw_values):.2f} | {max(bw_values):.2f} | {sum(bw_values)/len(bw_values):.2f} |

**Peak Theoretical**: {peak} {'TOPS' if 'int8' in dtype else 'TFLOPS'}

<details>
<summary>Detailed Results</summary>

| M | N | K | Time (μs) | TFLOPS | GB/s | MFU |
|---|---|---|-----------|--------|------|-----|
"""
        for d in data:
            report += f"| {d['m']} | {d['n']} | {d['k']} | {d['time_us']:.2f} | {d['tflops']:.2f} | {d['bandwidth_gbps']:.2f} | {d['mfu_percent']:.1f}% |\n"

        report += """
</details>

"""

    # Analysis section
    report += """---

## 4. Analysis

### Performance Observations

"""

    # Find best performing configs
    all_results = []
    for dtype, data in results.items():
        for d in data:
            d['dtype'] = dtype
            all_results.append(d)

    if all_results:
        # Best TFLOPS
        best_tflops = max(all_results, key=lambda x: x['tflops'])
        report += f"- **Highest TFLOPS**: {best_tflops['tflops']:.2f} TFLOPS ({best_tflops['dtype']}, M={best_tflops['m']}, K={best_tflops['k']}, N={best_tflops['n']})\n"

        # Best MFU
        best_mfu = max(all_results, key=lambda x: x['mfu_percent'])
        report += f"- **Highest MFU**: {best_mfu['mfu_percent']:.1f}% ({best_mfu['dtype']}, M={best_mfu['m']})\n"

        # Observations based on data patterns
        if best_mfu['m'] >= 2048:
            report += "- **Large batch sizes recommended**: Best MFU achieved with M ≥ 2048\n"
        if best_mfu['m'] <= 512:
            report += "- **Small batches efficient**: Good MFU achieved even with small M values\n"

    report += f"""
### Recommendations

1. **Optimal Batch Size**: Use M ≥ 512 for reasonable MFU (>30%)
2. **Best Data Type**: Use bfloat16 for optimal compute/memory balance
3. **Memory Bound**: For large matrices, bandwidth may become the bottleneck

---

## 5. Raw Output

<details>
<summary>Click to expand</summary>

```
{output}
```

</details>

---

*Report generated by auto_benchmark.py*
*GEMM Benchmark Tool: chay_gemm_benchmark_simple*
"""

    return report


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Automatic Chip Performance Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--quick", action="store_true",
                       help="Run quick test with fewer configurations")
    parser.add_argument("--output", type=str, default=None,
                       help="Save report to markdown file")
    parser.add_argument("--csv", type=str, default=None,
                       help="Save raw data to CSV file")
    parser.add_argument("--skip-benchmark", action="store_true",
                       help="Skip benchmark, only detect hardware")
    args = parser.parse_args()

    print("=" * 70)
    print("Chip Performance Test - Auto Benchmark")
    print("=" * 70)
    print()

    # Step 1: Detect hardware
    print("Step 1: Detecting hardware...")
    hw_info = detect_hardware()

    if not hw_info["type"]:
        print("ERROR: No supported hardware detected (GPU or TPU)")
        print("Please ensure you have CUDA/PyTorch or JAX with TPU access.")
        return 1

    print(f"  Detected: {hw_info['name']} ({hw_info['type'].upper()})")
    print(f"  Device count: {hw_info['count']}")
    print()

    # Step 2: Get hardware specs
    print("Step 2: Looking up hardware specifications...")
    hw_specs = get_hardware_specs(hw_info)
    if hw_specs['bfloat16_tflops'] > 0:
        print(f"  BF16 Peak: {hw_specs['bfloat16_tflops']} TFLOPS")
        print(f"  HBM Bandwidth: {hw_specs['bandwidth_gbs']} GB/s")
    else:
        print("  WARNING: Hardware specs not found, MFU calculation may be inaccurate")
    print()

    if args.skip_benchmark:
        print("Skipping benchmark (--skip-benchmark)")
        return 0

    # Step 3: Run benchmark
    print("Step 3: Running benchmark...")

    if hw_info["type"] == "tpu":
        config = "config/tpu_simple.json" if args.quick else "config/tpu_full.json"
        output = run_tpu_benchmark(config, args.csv)
    else:
        config = "config/simple.json" if args.quick else "config/gemm.json"
        output = run_gpu_benchmark(config)

    print(output)

    # Step 4: Parse results
    print("\nStep 4: Parsing results...")
    results = parse_benchmark_output(output)
    total_tests = sum(len(v) for v in results.values())
    print(f"  Parsed {total_tests} test results across {len(results)} data types")

    # Step 5: Generate report
    print("\nStep 5: Generating report...")
    report = generate_report(hw_info, hw_specs, results, output)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"  Report saved to: {args.output}")
    else:
        print("\n" + "=" * 70)
        print("REPORT")
        print("=" * 70)
        print(report)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
