"""Extract non-MoE weights from safetensors into a single consolidated file.

Scans all safetensors shards, skips mlp.experts keys, saves everything else
into one safetensors file (~23 GB for DeepSeek R1 671B).

This file can be placed in /dev/shm alongside the FP4 MoE cache for fast
vLLM startup (~15s vs ~4:30 from 70 shards on disk).

Usage:
  python3 extract_non_moe_weights.py --model-dir /data/models/DeepSeek-R1
  python3 extract_non_moe_weights.py --model-dir /data/models/DeepSeek-R1 --output /data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone/non_moe_weights.safetensors
"""
import argparse
import glob
import os
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def extract_non_moe(model_dir: str, output_path: str):
    sf_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    print(f"Scanning {len(sf_files)} safetensors files...")

    tensors = {}
    t0 = time.time()
    skipped_moe = 0

    for i, sf_file in enumerate(sf_files):
        with safe_open(sf_file, framework="pt") as f:
            for key in f.keys():
                if "mlp.experts" in key:
                    skipped_moe += 1
                    continue
                tensors[key] = f.get_tensor(key)

        if (i + 1) % 20 == 0 or i == len(sf_files) - 1:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(sf_files)}] {len(tensors)} non-MoE keys, "
                  f"{skipped_moe} MoE keys skipped, {elapsed:.1f}s")

    # Compute total size
    total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"\nExtracted {len(tensors)} non-MoE tensors, "
          f"total {total_bytes / 1e9:.2f} GB")

    # Save
    print(f"Saving to {output_path} ...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_file(tensors, output_path)

    file_size = os.path.getsize(output_path)
    total_elapsed = time.time() - t0
    print(f"Done: {file_size / 1e9:.2f} GB, {total_elapsed:.1f}s total")


def main():
    parser = argparse.ArgumentParser(
        description="Extract non-MoE weights into a single safetensors file")
    parser.add_argument("--model-dir", required=True,
                        help="Path to DeepSeek-R1 model directory")
    parser.add_argument("--output", default=None,
                        help="Output path (default: <model-dir>/../non_moe_weights.safetensors)")
    args = parser.parse_args()

    output = args.output or os.path.join(
        os.path.dirname(args.model_dir), "non_moe_weights.safetensors")
    extract_non_moe(args.model_dir, output)


if __name__ == "__main__":
    main()
