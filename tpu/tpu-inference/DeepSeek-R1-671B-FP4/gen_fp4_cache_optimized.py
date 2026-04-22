"""Generate FP4 cache for MoE layers — optimized with parallel I/O.

Optimizations over gen_missing_fp4_cache.py:
1. Build safetensors index once (avoid scanning 163 files per layer)
2. Prefetch next layer's experts while TPU processes current layer
3. Async npy saves (overlap I/O with computation)

Usage:
  python3 gen_fp4_cache_optimized.py [--model-dir PATH] [--cache-dir PATH] [--force]
  python3 gen_fp4_cache_optimized.py --backend gmm_tp  # For NEW_MODEL_DESIGN=1 + DP attention
"""
import argparse
import json
import os
import time
import glob
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional

import numpy as np
import ml_dtypes
import torch
from safetensors import safe_open

# Set env BEFORE importing tpu_inference (it reads env at import time)
os.environ["MOE_REQUANTIZE_WEIGHT_DTYPE"] = "float4_e2m1fn"

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, process_fp8_moe_weights
)

E = 256
ALL_MOE_LAYERS = list(range(3, 61))  # layers 3-60, 58 total


def t2j(t: torch.Tensor) -> jnp.ndarray:
    """Convert torch tensor to JAX array, handling FP8 dtype."""
    if t.dtype == torch.float8_e4m3fn:
        return jnp.array(t.view(torch.uint8).numpy()).view(jnp.float8_e4m3fn)
    return jnp.array(t.numpy())


def build_safetensors_index(model_dir: str, layers: list[int]) -> dict[int, list[tuple[str, list[str]]]]:
    """Build layer → [(file_path, [keys])] index in one pass.

    Instead of scanning all 163 files per layer, scan once and group by layer.
    """
    t0 = time.time()
    sf_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    layer_set = set(layers)

    # file → {layer → [keys]}
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

    # Reshape to layer → [(file, [keys])]
    index = defaultdict(list)
    for sf_file, layer_keys in file_layer_keys.items():
        for layer_idx, keys in layer_keys.items():
            index[layer_idx].append((sf_file, keys))

    elapsed = time.time() - t0
    total_keys = sum(len(keys) for entries in index.values() for _, keys in entries)
    print(f"[Index] Scanned {len(sf_files)} safetensors files in {elapsed:.1f}s")
    print(f"[Index] Found {total_keys} expert keys across {len(index)} layers")
    return dict(index)


def load_layer_experts(layer_idx: int, index: dict) -> tuple:
    """Load all 256 experts for a layer from safetensors. CPU-bound."""
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

    # Stack and concat — NO transpose (safetensors format matches convention)
    gate_stack = torch.stack([gate_w[i] for i in range(E)])
    up_stack = torch.stack([up_w[i] for i in range(E)])
    w13_weight = torch.cat([gate_stack, up_stack], dim=1)
    w2_weight = torch.stack([down_w[i] for i in range(E)])

    gate_ss = torch.stack([gate_s[i] for i in range(E)])
    up_ss = torch.stack([up_s[i] for i in range(E)])
    w13_scale = torch.cat([gate_ss, up_ss], dim=1)
    w2_scale = torch.stack([down_s[i] for i in range(E)])

    return w13_weight, w13_scale, w2_weight, w2_scale


def weights_to_numpy(output_weights: FusedMoEWeights) -> dict:
    """Convert JAX device arrays to numpy on host. Frees device memory."""
    result = {}
    meta = {"_cache_format": "npy_v1"}

    for name in ["w13_weight", "w13_weight_scale", "w13_bias",
                  "w2_weight", "w2_weight_scale", "w2_bias"]:
        val = getattr(output_weights, name)
        if val is not None:
            np_val = np.asarray(val)
            dtype_str = str(val.dtype)
            if 'float4' in dtype_str:
                np_val = np_val.astype(ml_dtypes.float8_e4m3fn)
                meta["_storage_format"] = "native_fp4"
            result[name] = np_val
            meta[f"{name}_dtype"] = dtype_str

    result["_meta"] = meta
    return result


def save_layer_numpy(out_dir: str, np_data: dict):
    """Save numpy arrays to disk. CPU-bound I/O only (no device refs)."""
    os.makedirs(out_dir, exist_ok=True)
    meta = np_data.pop("_meta")

    for name, np_val in np_data.items():
        np.save(os.path.join(out_dir, f"{name}.npy"), np_val)

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)


def get_missing_layers(cache_dir: str) -> list[int]:
    """Detect which MoE layers are missing from cache."""
    existing = set()
    if os.path.exists(cache_dir):
        for d in os.listdir(cache_dir):
            if d.startswith("model_layers_"):
                try:
                    layer_num = int(d.split("_")[2])
                    # Verify it has actual files (not empty dir from interrupted run)
                    layer_dir = os.path.join(cache_dir, d)
                    if os.path.exists(os.path.join(layer_dir, "meta.json")):
                        existing.add(layer_num)
                except (ValueError, IndexError):
                    pass
    missing = sorted(set(ALL_MOE_LAYERS) - existing)
    return missing


def main():
    parser = argparse.ArgumentParser(description="Generate FP4 MoE cache (optimized)")
    parser.add_argument("--model-dir", default="/data/models/DeepSeek-R1")
    parser.add_argument("--cache-dir", default=None,
                        help="Cache output dir. Default auto-selects based on backend.")
    parser.add_argument("--backend", default="gmm_tp", choices=["gmm_ep", "gmm_tp"],
                        help="MoE backend (default: gmm_tp for NEW_MODEL_DESIGN=1)")
    parser.add_argument("--force", action="store_true", help="Regenerate all 58 layers")
    args = parser.parse_args()

    # Auto-select cache dir based on backend
    if args.cache_dir is None:
        backend_suffix = args.backend.replace("_", "_")
        args.cache_dir = f"/dev/shm/ep8_tp1_{backend_suffix}_fp4e2m1_bsNone"
        print(f"[Config] Auto cache dir: {args.cache_dir}")

    # Select MoE backend
    if args.backend == "gmm_tp":
        moe_backend = MoEBackend.GMM_TP
    else:
        moe_backend = MoEBackend.GMM_EP
    print(f"[Config] MoE backend: {moe_backend}")

    # Setup JAX mesh
    devices = jax.devices()
    mesh = Mesh(
        np.array(devices).reshape(1, 8, 1, 1, 1),
        axis_names=('data', 'attn_dp', 'attn_dp_expert', 'expert', 'model')
    )

    # Detect missing layers
    if args.force:
        missing = ALL_MOE_LAYERS[:]
    else:
        missing = get_missing_layers(args.cache_dir)

    if not missing:
        print(f"All 58 layers already cached in {args.cache_dir}")
        return

    print(f"Processing {len(missing)} layers: {missing}")

    # Phase 1: Build safetensors index (one-time ~30s scan)
    index = build_safetensors_index(args.model_dir, missing)

    # Phase 2: Process layers with I/O pipelining
    prefetch_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch")
    save_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="save")
    save_futures: list[Future] = []

    # Kick off prefetch for first layer
    next_future: Optional[Future] = prefetch_pool.submit(load_layer_experts, missing[0], index)

    total_t0 = time.time()
    layer_times = []

    for i, layer_idx in enumerate(missing):
        t0 = time.time()

        # Wait for current layer's data (already prefetching)
        t_io_start = time.time()
        w13_weight, w13_scale, w2_weight, w2_scale = next_future.result()
        t_io = time.time() - t_io_start

        # Start prefetching next layer (overlaps with TPU processing)
        if i + 1 < len(missing):
            next_future = prefetch_pool.submit(load_layer_experts, missing[i + 1], index)

        # Convert to JAX arrays
        input_weights = FusedMoEWeights(
            w13_weight=t2j(w13_weight),
            w13_weight_scale=t2j(w13_scale),
            w13_bias=None,
            w2_weight=t2j(w2_weight),
            w2_weight_scale=t2j(w2_scale),
            w2_bias=None,
        )

        # FP4 requantization on TPU
        t_tpu_start = time.time()
        output_weights = process_fp8_moe_weights(
            input_weights,
            moe_backend=moe_backend,
            mesh=mesh,
            activation="silu",
            weight_block_size=None,
        )
        t_tpu = time.time() - t_tpu_start

        # Copy device arrays to host numpy BEFORE async save
        # This releases HBM immediately, preventing OOM on next layer
        t_copy_start = time.time()
        np_data = weights_to_numpy(output_weights)
        del output_weights, input_weights
        t_copy = time.time() - t_copy_start

        # Async save numpy arrays (no device refs held)
        out_dir = os.path.join(args.cache_dir, f"model_layers_{layer_idx}_mlp_experts")
        save_futures.append(save_pool.submit(save_layer_numpy, out_dir, np_data))

        # Free JIT compilation cache to prevent HBM OOM on next layer
        jax.clear_caches()

        elapsed = time.time() - t0
        layer_times.append(elapsed)
        print(f"  [{i+1}/{len(missing)}] Layer {layer_idx}: "
              f"{elapsed:.1f}s (io={t_io:.1f}s tpu={t_tpu:.1f}s d2h={t_copy:.1f}s) "
              f"w13={w13_weight.shape} w2={w2_weight.shape}")

    # Wait for all async saves to complete
    print(f"\nWaiting for {len(save_futures)} async saves to complete...")
    for f in save_futures:
        f.result()  # Raises if any save failed

    total_elapsed = time.time() - total_t0

    # Summary
    prefetch_pool.shutdown(wait=False)
    save_pool.shutdown(wait=False)

    total_cached = len([d for d in os.listdir(args.cache_dir)
                        if d.startswith("model_layers")])
    avg_time = sum(layer_times) / len(layer_times) if layer_times else 0

    print(f"\n{'='*60}")
    print(f"Total: {total_cached}/58 layers cached")
    print(f"Processed {len(missing)} layers in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"Average: {avg_time:.1f}s/layer")
    print(f"Min/Max: {min(layer_times):.1f}s / {max(layer_times):.1f}s")
    print(f"Cache dir: {args.cache_dir}")


if __name__ == "__main__":
    main()
