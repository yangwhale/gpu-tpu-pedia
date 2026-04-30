#!/usr/bin/env python3
"""Convert native_fp4 cache files to float8_e4m3fn format in-place.

Usage:
    python3 convert_fp4_to_fp8_cache.py --cache-dir /path/to/ep8_tp1_gmm_ep_fp4e2m1_bsNone
    python3 convert_fp4_to_fp8_cache.py --cache-dir /dev/shm/ep8_tp1_gmm_ep_fp4e2m1_bsNone --workers 12

Background:
    gen_fp4_cache_cpu_parallel.py saves native float4_e2m1fn bytes, but the
    vLLM loader (fp8.py _load_moe_cache_npy_v1) expects float8_e4m3fn bytes.
    A raw .view() reinterprets the bits incorrectly (FP4(1.0) = 0x02 becomes
    FP8(0.003906), shrinking all MoE weights ~256x and producing garbage output).

    This script uses a 16-entry LUT to correctly convert FP4 byte values to
    their FP8 equivalents, updating meta.json to mark the conversion.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import ml_dtypes
import numpy as np

# 16-entry lookup table: FP4 byte (0-15) → FP8 byte
LUT = (np.arange(16, dtype=np.uint8)
       .view(ml_dtypes.float4_e2m1fn)
       .astype(np.float32)
       .astype(ml_dtypes.float8_e4m3fn)
       .view(np.uint8))


def convert_layer(layer_dir: str) -> str:
    meta_path = os.path.join(layer_dir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    if meta.get("_storage_format") != "native_fp4":
        return f"{os.path.basename(layer_dir)}: already converted, skipped"

    t0 = time.time()
    for name in ["w13_weight", "w2_weight"]:
        npy_path = os.path.join(layer_dir, f"{name}.npy")
        if os.path.exists(npy_path):
            arr = np.load(npy_path, mmap_mode="r")
            converted = LUT[arr.view(np.uint8)]
            np.save(npy_path, converted.view(ml_dtypes.float8_e4m3fn))

    meta["_storage_format"] = "fp8_reinterpreted"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return f"{os.path.basename(layer_dir)}: converted in {time.time() - t0:.1f}s"


def main():
    parser = argparse.ArgumentParser(description="Convert native FP4 cache to FP8 format")
    parser.add_argument("--cache-dir", required=True, help="Path to cache directory (e.g. ep8_tp1_gmm_ep_fp4e2m1_bsNone)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers (default: 8)")
    args = parser.parse_args()

    layer_dirs = sorted([
        os.path.join(args.cache_dir, d)
        for d in os.listdir(args.cache_dir)
        if d.startswith("model_layers_") and os.path.isdir(os.path.join(args.cache_dir, d))
    ])
    print(f"Found {len(layer_dirs)} layer dirs in {args.cache_dir}")

    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(convert_layer, layer_dirs)):
            print(f"[{i + 1}/{len(layer_dirs)}] {result}")

    print(f"\nTotal: {time.time() - t_start:.1f}s ({args.workers} workers)")


if __name__ == "__main__":
    main()
