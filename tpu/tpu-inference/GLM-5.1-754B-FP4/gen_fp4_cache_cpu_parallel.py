"""Generate FP4 cache for MoE layers — CPU-only parallel processing.

Direct safetensors → FP4 conversion without FP8 intermediate or TPU.
Uses ProcessPoolExecutor to process multiple layers in parallel.

Adapted from DeepSeek R1 version for GLM-5.1 (754B MoE).
Key differences: 78 layers (75 MoE, first 3 dense), hidden_size=6144.

Pipeline per layer (all numpy, no JAX/TPU):
  1. Load 256 experts from safetensors → w13 (256, 4096, 6144) FP8 + scale
  2. Dequant: FP8 × scale → FP32
  3. Per-channel quantize FP32 → FP4 along axis=2 (matching quantize_moe_weights)
  4. GMM_EP layout: swapaxes(1, 2) on weights, swapaxes + expand_dims on scales
  5. Save as npy + meta.json

Output shapes match process_fp8_moe_weights + process_moe_weights(GMM_EP):
  w13_weight:       (256, 6144, 4096) float4_e2m1fn
  w13_weight_scale: (256, 1, 1, 4096) float32
  w2_weight:        (256, 2048, 6144) float4_e2m1fn
  w2_weight_scale:  (256, 1, 1, 6144) float32

Usage:
  python3 gen_fp4_cache_cpu_parallel.py --model-dir /data/models/GLM-5.1 --workers 8
  python3 gen_fp4_cache_cpu_parallel.py --model-dir /data/models/GLM-5.1 --cache-dir /dev/shm/fp4-cache --workers 4
"""
import argparse
import gc
import json
import os
import time
import glob
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import ml_dtypes
import numpy as np
import torch
from safetensors import safe_open

E = 256
ALL_MOE_LAYERS = list(range(3, 79))  # layers 3-78, 76 total (first_k_dense=3, layer 78=MTP also has MoE)

fp4_max = float(ml_dtypes.finfo(ml_dtypes.float4_e2m1fn).max)
fp4_min = float(ml_dtypes.finfo(ml_dtypes.float4_e2m1fn).min)


def build_safetensors_index(model_dir: str, layers: list[int]) -> dict:
    """Build layer → [(file_path, [keys])] index in one pass."""
    t0 = time.time()
    sf_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    layer_set = set(layers)

    file_layer_keys = defaultdict(lambda: defaultdict(list))

    for sf_file in sf_files:
        with safe_open(sf_file, framework="pt") as f:
            for key in f.keys():
                if "mlp.experts" not in key:
                    continue
                parts = key.split(".")
                layer_idx = int(parts[2])
                if layer_idx in layer_set:
                    file_layer_keys[sf_file][layer_idx].append(key)

    index = defaultdict(list)
    for sf_file, layer_keys in file_layer_keys.items():
        for layer_idx, keys in layer_keys.items():
            index[layer_idx].append((sf_file, keys))

    elapsed = time.time() - t0
    total_keys = sum(len(keys) for entries in index.values() for _, keys in entries)
    print(f"[Index] Scanned {len(sf_files)} files in {elapsed:.1f}s, "
          f"{total_keys} expert keys across {len(index)} layers")
    return dict(index)


def load_layer_experts(layer_idx: int, index: dict) -> tuple:
    """Load all 256 experts for a layer from safetensors → numpy arrays."""
    gate_w, up_w, down_w = {}, {}, {}
    gate_s, up_s, down_s = {}, {}, {}

    for sf_file, keys in index[layer_idx]:
        with safe_open(sf_file, framework="pt") as f:
            for key in keys:
                prefix = f"model.layers.{layer_idx}.mlp.experts."
                parts = key[len(prefix):].split(".")
                eid = int(parts[0])
                ptype = parts[1]
                is_scale = "scale" in parts[-1]
                t = f.get_tensor(key).cpu()
                if is_scale:
                    if "gate" in ptype: gate_s[eid] = t
                    elif "up" in ptype: up_s[eid] = t
                    elif "down" in ptype: down_s[eid] = t
                else:
                    if "gate" in ptype: gate_w[eid] = t
                    elif "up" in ptype: up_w[eid] = t
                    elif "down" in ptype: down_w[eid] = t

    assert len(gate_w) == E, f"Layer {layer_idx}: expected {E} experts, got {len(gate_w)}"

    # Stack and concat: w13 = [gate; up], shape (256, 4096, 7168) FP8
    gate_stack = torch.stack([gate_w[i] for i in range(E)])
    up_stack = torch.stack([up_w[i] for i in range(E)])
    w13_weight = torch.cat([gate_stack, up_stack], dim=1)
    w2_weight = torch.stack([down_w[i] for i in range(E)])

    gate_ss = torch.stack([gate_s[i] for i in range(E)])
    up_ss = torch.stack([up_s[i] for i in range(E)])
    w13_scale = torch.cat([gate_ss, up_ss], dim=1)
    w2_scale = torch.stack([down_s[i] for i in range(E)])

    # Convert to numpy, handling FP8 dtype
    def to_numpy(t):
        if t.dtype == torch.float8_e4m3fn:
            return t.view(torch.uint8).numpy().view(ml_dtypes.float8_e4m3fn)
        return t.numpy()

    return (to_numpy(w13_weight), w13_scale.numpy(),
            to_numpy(w2_weight), w2_scale.numpy())


def dequant_fp8_blocked(w_fp8: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Dequantize FP8 with 2D block scaling → FP32. Memory-optimized.

    DeepSeek R1 uses 128×128 block quantization:
      weight (E, 4096, 7168) + scale (E, 32, 56)

    Uses in-place multiply to avoid allocating a separate result array.
    Peak: 1× tensor size in FP32 (the blocked array).
    """
    E_dim, d1, d2 = w_fp8.shape
    _, s1, s2 = scale.shape
    bs1 = d1 // s1
    bs2 = d2 // s2

    # reshape is a view (no copy), astype creates one FP32 copy
    w_blocked = w_fp8.reshape(E_dim, s1, bs1, s2, bs2).astype(np.float32)
    scale_expanded = scale.astype(np.float32)[:, :, np.newaxis, :, np.newaxis]

    # In-place multiply: avoids allocating a second FP32 copy
    w_blocked *= scale_expanded
    return w_blocked.reshape(E_dim, d1, d2)


def quantize_to_fp4(w_fp32: np.ndarray) -> tuple:
    """Per-channel FP32 → FP4 quantization along axis=2. Memory-optimized.

    Uses in-place operations to minimize peak memory.
    Modifies w_fp32 in place (caller should not reuse it).
    Peak: 1× input size (the abs buffer, freed before quantization).
    """
    # Compute abs_max with temporary buffer (freed immediately)
    abs_buf = np.abs(w_fp32)
    abs_max = np.max(abs_buf, axis=2, keepdims=True)  # (E, dim1, 1)
    del abs_buf  # free full-size temporary

    scale = np.where(abs_max == 0, 1.0, abs_max / fp4_max).astype(np.float32)
    del abs_max

    scale_inv = np.where(scale == 0, np.inf, 1.0 / scale)

    # In-place: w_fp32 *= scale_inv, then clip in-place
    w_fp32 *= scale_inv
    del scale_inv
    np.clip(w_fp32, fp4_min, fp4_max, out=w_fp32)
    fp4 = w_fp32.astype(ml_dtypes.float4_e2m1fn)

    return fp4, scale


def convert_one_tensor(w_fp8, scale_orig, out_path_w, out_path_s):
    """Full pipeline for one weight tensor: dequant → quantize → layout → save.

    Processes and saves immediately, freeing memory before returning.
    Peak memory: ~2× tensor size in FP32 (dequant intermediate + result).
    """
    # Dequant FP8 → FP32 (block quantization)
    w_fp32 = dequant_fp8_blocked(w_fp8, scale_orig)
    del w_fp8, scale_orig

    # Quantize FP32 → FP4 (per-channel, axis=2)
    fp4, scale = quantize_to_fp4(w_fp32)
    del w_fp32

    # GMM_EP layout: swapaxes(1,2)
    fp4 = np.ascontiguousarray(np.swapaxes(fp4, 1, 2))
    # Scale: swapaxes(1,2) + expand_dims(2) → 4D
    scale = np.expand_dims(np.swapaxes(scale, 1, 2), 2)

    # Save
    np.save(out_path_w, fp4)
    np.save(out_path_s, scale)
    del fp4, scale
    gc.collect()


def process_layer(args):
    """Process a single MoE layer: load → convert w13 → convert w2 → save meta.

    Memory-optimized: processes w13 and w2 sequentially, freeing each before
    starting the next. Peak ~70 GB instead of ~130 GB.
    """
    layer_idx, index, cache_dir = args
    t0 = time.perf_counter()

    layer_name = f"model_layers_{layer_idx}_mlp_experts"
    out_dir = os.path.join(cache_dir, layer_name)
    os.makedirs(out_dir, exist_ok=True)

    # Load all experts (loads w13 + w2 together from safetensors)
    t_load = time.perf_counter()
    w13_fp8, w13_scale, w2_fp8, w2_scale = load_layer_experts(layer_idx, index)
    t_load = time.perf_counter() - t_load

    # Process w13 first (bigger: 256×4096×6144), free before w2
    t_w13 = time.perf_counter()
    convert_one_tensor(
        w13_fp8, w13_scale,
        os.path.join(out_dir, "w13_weight.npy"),
        os.path.join(out_dir, "w13_weight_scale.npy"),
    )
    del w13_fp8, w13_scale
    t_w13 = time.perf_counter() - t_w13

    # Process w2 (smaller: 256×6144×2048)
    t_w2 = time.perf_counter()
    convert_one_tensor(
        w2_fp8, w2_scale,
        os.path.join(out_dir, "w2_weight.npy"),
        os.path.join(out_dir, "w2_weight_scale.npy"),
    )
    del w2_fp8, w2_scale
    t_w2 = time.perf_counter() - t_w2

    # Save meta.json
    meta = {
        "_cache_format": "npy_v1",
        "_storage_format": "native_fp4",
        "w13_weight_dtype": "float4_e2m1fn",
        "w13_weight_scale_dtype": "float32",
        "w2_weight_dtype": "float4_e2m1fn",
        "w2_weight_scale_dtype": "float32",
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    gc.collect()

    total = time.perf_counter() - t0
    print(f"  Layer {layer_idx}: {total:.1f}s "
          f"(load={t_load:.1f}s w13={t_w13:.1f}s w2={t_w2:.1f}s)",
          flush=True)
    return layer_idx, total


def get_missing_layers(cache_dir: str) -> list[int]:
    """Detect which MoE layers are missing from cache."""
    existing = set()
    if os.path.exists(cache_dir):
        for d in os.listdir(cache_dir):
            if d.startswith("model_layers_"):
                layer_dir = os.path.join(cache_dir, d)
                if os.path.exists(os.path.join(layer_dir, "meta.json")):
                    try:
                        layer_num = int(d.split("_")[2])
                        existing.add(layer_num)
                    except (ValueError, IndexError):
                        pass
    return sorted(set(ALL_MOE_LAYERS) - existing)


def main():
    parser = argparse.ArgumentParser(
        description="Generate FP4 MoE cache — CPU-only parallel")
    parser.add_argument("--model-dir", default="/data/models/GLM-5.1",
                        help="Path to GLM-5.1 safetensors")
    parser.add_argument("--cache-dir", default=None,
                        help="Cache output dir (default: /data/moe-cache-fp4-cpu/...)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel workers (default: 8)")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate all 76 layers")
    args = parser.parse_args()

    if args.cache_dir is None:
        args.cache_dir = "/data/moe-cache-fp4-cpu/ep8_tp1_gmm_ep_fp4e2m1_bsNone"

    # Detect missing layers
    if args.force:
        missing = ALL_MOE_LAYERS[:]
    else:
        missing = get_missing_layers(args.cache_dir)

    if not missing:
        print(f"All 76 layers already cached in {args.cache_dir}")
        return

    print(f"=== CPU-only Parallel FP4 Cache Generation ===")
    print(f"  Model:   {args.model_dir}")
    print(f"  Output:  {args.cache_dir}")
    print(f"  Workers: {args.workers}")
    print(f"  Layers:  {len(missing)} ({missing[0]}-{missing[-1]})")
    print(f"  FP4 range: [{fp4_min}, {fp4_max}]")
    print()

    # Phase 1: Build safetensors index (one-time scan)
    index = build_safetensors_index(args.model_dir, missing)

    # Phase 2: Process layers in parallel
    os.makedirs(args.cache_dir, exist_ok=True)
    t_start = time.perf_counter()

    work_items = [(layer_idx, index, args.cache_dir) for layer_idx in missing]

    # max_tasks_per_child=1: kill worker after each layer to fully release memory.
    # Without this, glibc malloc arena fragmentation causes OOM after ~9 layers.
    with ProcessPoolExecutor(max_workers=args.workers,
                             max_tasks_per_child=1) as pool:
        results = list(pool.map(process_layer, work_items))

    t_total = time.perf_counter() - t_start
    layer_times = [t for _, t in results]

    # Summary
    n_out = len([d for d in os.listdir(args.cache_dir)
                 if d.startswith("model_layers_")])
    print(f"\n{'='*60}")
    print(f"Done: {n_out}/76 layers cached")
    print(f"Processed {len(missing)} layers in {t_total:.1f}s ({t_total/60:.1f} min)")
    print(f"Average: {sum(layer_times)/len(layer_times):.1f}s/layer")
    print(f"Min/Max: {min(layer_times):.1f}s / {max(layer_times):.1f}s")
    print(f"Cache dir: {args.cache_dir}")


if __name__ == "__main__":
    main()
