"""Generate FP4 cache for missing MoE layers (direct FP4, no FP8 intermediate)."""
import json, os, time, glob
import numpy as np
import ml_dtypes
import torch
from safetensors import safe_open

# Set env BEFORE importing tpu_inference (it reads env at import time)
os.environ["MOE_REQUANTIZE_WEIGHT_DTYPE"] = "float4_e2m1fn"

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

MODEL_DIR = "/data/models/DeepSeek-R1"
CACHE_DIR = "/data/moe-cache-fp4-direct/ep8_tp1_gmm_ep_fp4e2m1_bsNone"
E = 256

from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, process_fp8_moe_weights
)

# Create mesh matching the vLLM config
devices = jax.devices()
mesh = Mesh(
    np.array(devices).reshape(1, 1, 8, 1, 1),
    axis_names=('data', 'attn_dp', 'attn_dp_expert', 'expert', 'model')
)

sf_files = sorted(glob.glob(os.path.join(MODEL_DIR, "*.safetensors")))

def t2j(t):
    if t.dtype == torch.float8_e4m3fn:
        return jnp.array(t.view(torch.uint8).numpy()).view(jnp.float8_e4m3fn)
    return jnp.array(t.numpy())

# Detect which layers are missing
existing = set()
if os.path.exists(CACHE_DIR):
    for d in os.listdir(CACHE_DIR):
        if d.startswith("model_layers_"):
            layer_num = int(d.split("_")[2])
            existing.add(layer_num)

all_moe_layers = list(range(3, 61))  # layers 3-60
missing = sorted(set(all_moe_layers) - existing)

if not missing:
    print(f"All 58 layers already cached in {CACHE_DIR}")
else:
    print(f"Missing {len(missing)} layers: {missing}")
    print(f"Existing: {len(existing)}/58")

for layer_idx in missing:
    t0 = time.time()
    print(f"\n=== Layer {layer_idx} ===")

    gate_w, up_w, down_w = {}, {}, {}
    gate_s, up_s, down_s = {}, {}, {}

    for sf_file in sf_files:
        with safe_open(sf_file, framework="pt") as f:
            for key in f.keys():
                prefix = f"model.layers.{layer_idx}.mlp.experts."
                if not key.startswith(prefix):
                    continue
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

    assert len(gate_w) == E
    print(f"  All {E} experts loaded, gate[0]={gate_w[0].shape} {gate_w[0].dtype}")

    # NO transpose — safetensors format matches process_fp8_moe_weights convention:
    #   w13 = (E, 2*intermediate, hidden) — gate+up concat on axis=1
    #   w2  = (E, hidden, intermediate)
    gate_stack = torch.stack([gate_w[i] for i in range(E)])
    up_stack = torch.stack([up_w[i] for i in range(E)])
    w13_weight = torch.cat([gate_stack, up_stack], dim=1)

    w2_weight = torch.stack([down_w[i] for i in range(E)])

    gate_ss = torch.stack([gate_s[i] for i in range(E)])
    up_ss = torch.stack([up_s[i] for i in range(E)])
    w13_scale = torch.cat([gate_ss, up_ss], dim=1)
    w2_scale = torch.stack([down_s[i] for i in range(E)])

    print(f"  w13={w13_weight.shape} scale={w13_scale.shape}")
    print(f"  w2={w2_weight.shape} scale={w2_scale.shape}")

    input_weights = FusedMoEWeights(
        w13_weight=t2j(w13_weight),
        w13_weight_scale=t2j(w13_scale),
        w13_bias=None,
        w2_weight=t2j(w2_weight),
        w2_weight_scale=t2j(w2_scale),
        w2_bias=None,
    )

    print(f"  Running FP4 requantization...")
    output_weights = process_fp8_moe_weights(
        input_weights,
        moe_backend=MoEBackend.GMM_EP,
        mesh=mesh,
        activation="silu",
        weight_block_size=None,
    )

    out_dir = os.path.join(CACHE_DIR, f"model_layers_{layer_idx}_mlp_experts")
    os.makedirs(out_dir, exist_ok=True)

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
            np.save(os.path.join(out_dir, f"{name}.npy"), np_val)
            meta[f"{name}_dtype"] = dtype_str

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

total = len([d for d in os.listdir(CACHE_DIR) if d.startswith("model_layers")])
print(f"\nTotal: {total}/58 layers cached")
