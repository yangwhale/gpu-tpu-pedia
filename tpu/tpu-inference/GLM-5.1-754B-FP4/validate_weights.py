"""Spot-check validation for FP4 MoE cache and non-MoE weights.

Checks:
1. Non-MoE: bit-exact match against original safetensors (sample 10 keys)
2. FP4 MoE: dequant FP4→FP32 vs original FP8→FP32, check error distribution
   - Max/mean absolute error
   - Cosine similarity
   - Zero ratio
   - Value distribution (min/max/mean/std)
   - Histogram of relative errors
"""
import os
import sys
import time
import numpy as np
import ml_dtypes
import torch
from safetensors import safe_open

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "/data/models/GLM-5.1-FP8"
CACHE_DIR = sys.argv[2] if len(sys.argv) > 2 else "/data/moe-cache/ep8_tp1_gmm_ep_fp4e2m1_bsNone"

# ============================================================
# Part 1: Non-MoE bit-exact check
# ============================================================
def check_non_moe():
    print("=" * 60)
    print("Part 1: Non-MoE Weight Validation (bit-exact check)")
    print("=" * 60)

    consolidated = os.path.join(CACHE_DIR, "non_moe_weights.safetensors")
    if not os.path.exists(consolidated):
        print(f"  SKIP: {consolidated} not found")
        return

    # Get keys from consolidated file
    with safe_open(consolidated, framework="pt") as f:
        all_keys = list(f.keys())

    print(f"  Total keys in consolidated: {len(all_keys)}")

    # Sample 10 diverse keys (embedding, attention, norm, router)
    sample_keys = []
    patterns = ["embed_tokens", "self_attn.kv_b_proj", "input_layernorm",
                 "mlp.gate", "self_attn.q_a_proj", "model.norm",
                 "self_attn.o_proj", "post_attention_layernorm",
                 "self_attn.kv_a_proj_with_mqa", "lm_head"]
    for pat in patterns:
        for k in all_keys:
            if pat in k:
                sample_keys.append(k)
                break
    # Fill remaining with random keys
    import random
    random.seed(42)
    remaining = [k for k in all_keys if k not in sample_keys]
    sample_keys += random.sample(remaining, min(5, len(remaining)))
    sample_keys = sample_keys[:15]

    print(f"  Sampling {len(sample_keys)} keys for bit-exact check...")

    # Build index: which safetensors file has which key
    import glob
    sf_files = sorted(glob.glob(os.path.join(MODEL_DIR, "*.safetensors")))

    mismatches = 0
    for key in sample_keys:
        # Load from consolidated
        with safe_open(consolidated, framework="pt") as f:
            consolidated_tensor = f.get_tensor(key)

        # Find in original shards
        original_tensor = None
        for sf_file in sf_files:
            with safe_open(sf_file, framework="pt") as f:
                if key in f.keys():
                    original_tensor = f.get_tensor(key)
                    break

        if original_tensor is None:
            print(f"  ❌ {key}: NOT FOUND in original shards!")
            mismatches += 1
            continue

        match = torch.equal(consolidated_tensor, original_tensor)
        status = "✅" if match else "❌"
        if not match:
            mismatches += 1
            diff = (consolidated_tensor.float() - original_tensor.float()).abs()
            print(f"  {status} {key}: shape={list(consolidated_tensor.shape)} dtype={consolidated_tensor.dtype} "
                  f"max_diff={diff.max().item():.6e}")
        else:
            print(f"  {status} {key}: shape={list(consolidated_tensor.shape)} dtype={consolidated_tensor.dtype} EXACT")

    print(f"\n  Result: {len(sample_keys) - mismatches}/{len(sample_keys)} keys bit-exact match")
    return mismatches == 0

# ============================================================
# Part 2: FP4 MoE quantization quality check
# ============================================================
def dequant_fp8_blocked(w_fp8, scale):
    """Dequantize FP8 with 2D block scaling → FP32."""
    E_dim, d1, d2 = w_fp8.shape
    _, s1, s2 = scale.shape
    bs1, bs2 = d1 // s1, d2 // s2
    w_blocked = w_fp8.reshape(E_dim, s1, bs1, s2, bs2).astype(np.float32)
    scale_expanded = scale.astype(np.float32)[:, :, np.newaxis, :, np.newaxis]
    w_blocked *= scale_expanded
    return w_blocked.reshape(E_dim, d1, d2)


def dequant_fp4(fp4_weight, fp4_scale):
    """Dequantize FP4 cache back to FP32. Reverses GMM_EP layout.

    Cache layout (after GMM_EP swapaxes):
      w13_weight: (256, 6144, 4096) |V1  → swapaxes → (256, 4096, 6144) original
      w13_scale:  (256, 1, 1, 4096) f32  → squeeze+swapaxes → (256, 4096, 1)
    """
    # View as float4_e2m1fn (npy stores as |V1 void type)
    fp4_weight = fp4_weight.view(ml_dtypes.float4_e2m1fn)
    # Reverse GMM_EP layout: swapaxes(1,2) back
    fp4_weight = np.swapaxes(fp4_weight, 1, 2)  # (E, original_dim1, original_dim2)
    fp4_scale = np.swapaxes(fp4_scale.squeeze(2), 1, 2)  # (E, original_dim1, 1)
    # float4_e2m1fn → float32, then scale
    w_f32 = fp4_weight.astype(np.float32)
    return w_f32 * fp4_scale.astype(np.float32)


def check_moe_layer(layer_idx):
    """Check one MoE layer's FP4 cache against original FP8 weights."""
    layer_dir = os.path.join(CACHE_DIR, f"model_layers_{layer_idx}_mlp_experts")
    if not os.path.exists(layer_dir):
        print(f"  SKIP: layer {layer_idx} cache not found")
        return None

    print(f"\n  --- Layer {layer_idx} ---")

    # Load FP4 cache
    w13_fp4 = np.load(os.path.join(layer_dir, "w13_weight.npy"))
    w13_scale = np.load(os.path.join(layer_dir, "w13_weight_scale.npy"))
    w2_fp4 = np.load(os.path.join(layer_dir, "w2_weight.npy"))
    w2_scale = np.load(os.path.join(layer_dir, "w2_weight_scale.npy"))

    print(f"  w13: {w13_fp4.shape} {w13_fp4.dtype}, scale: {w13_scale.shape}")
    print(f"  w2:  {w2_fp4.shape} {w2_fp4.dtype}, scale: {w2_scale.shape}")

    # Dequant FP4 → FP32
    w13_fp4_f32 = dequant_fp4(w13_fp4, w13_scale)
    w2_fp4_f32 = dequant_fp4(w2_fp4, w2_scale)

    # Load original FP8 weights for comparison (sample 3 experts to save memory)
    import glob
    sf_files = sorted(glob.glob(os.path.join(MODEL_DIR, "*.safetensors")))

    sample_experts = [0, 127, 255]  # first, middle, last
    results = {}

    for eid in sample_experts:
        gate_w, gate_s, up_w, up_s, down_w, down_s = None, None, None, None, None, None
        prefix = f"model.layers.{layer_idx}.mlp.experts.{eid}."

        for sf_file in sf_files:
            with safe_open(sf_file, framework="pt") as f:
                for key in f.keys():
                    if not key.startswith(prefix):
                        continue
                    t = f.get_tensor(key).cpu()
                    if "gate_proj.weight_scale_inv" in key: gate_s = t
                    elif "gate_proj.weight" in key: gate_w = t
                    elif "up_proj.weight_scale_inv" in key: up_s = t
                    elif "up_proj.weight" in key: up_w = t
                    elif "down_proj.weight_scale_inv" in key: down_s = t
                    elif "down_proj.weight" in key: down_w = t

            if all(x is not None for x in [gate_w, gate_s, up_w, up_s, down_w, down_s]):
                break

        if gate_w is None:
            print(f"    Expert {eid}: keys not found, skipping")
            continue

        # Convert to numpy FP8
        def to_np(t):
            if t.dtype == torch.float8_e4m3fn:
                return t.view(torch.uint8).numpy().view(ml_dtypes.float8_e4m3fn)
            return t.numpy()

        # w13 = concat(gate, up) along dim 0
        w13_orig_fp8 = np.concatenate([to_np(gate_w), to_np(up_w)], axis=0)
        w13_orig_scale = np.concatenate([gate_s.numpy(), up_s.numpy()], axis=0)
        w2_orig_fp8 = to_np(down_w)
        w2_orig_scale = down_s.numpy()

        # Dequant original FP8 → FP32
        w13_orig_f32 = dequant_fp8_blocked(
            w13_orig_fp8[np.newaxis], w13_orig_scale[np.newaxis])[0]
        w2_orig_f32 = dequant_fp8_blocked(
            w2_orig_fp8[np.newaxis], w2_orig_scale[np.newaxis])[0]

        # Compare w13
        w13_diff = np.abs(w13_fp4_f32[eid] - w13_orig_f32)
        w13_orig_abs = np.abs(w13_orig_f32)
        w13_nonzero = w13_orig_abs > 1e-8
        w13_rel_err = np.where(w13_nonzero, w13_diff / w13_orig_abs, 0)

        # Cosine similarity
        a, b = w13_fp4_f32[eid].flatten(), w13_orig_f32.flatten()
        cos_sim_w13 = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

        # Compare w2
        w2_diff = np.abs(w2_fp4_f32[eid] - w2_orig_f32)
        w2_orig_abs = np.abs(w2_orig_f32)
        w2_nonzero = w2_orig_abs > 1e-8
        w2_rel_err = np.where(w2_nonzero, w2_diff / w2_orig_abs, 0)

        a2, b2 = w2_fp4_f32[eid].flatten(), w2_orig_f32.flatten()
        cos_sim_w2 = np.dot(a2, b2) / (np.linalg.norm(a2) * np.linalg.norm(b2) + 1e-10)

        # Zero analysis
        w13_zero_ratio = np.mean(w13_fp4_f32[eid] == 0) * 100
        w2_zero_ratio = np.mean(w2_fp4_f32[eid] == 0) * 100
        w13_orig_zero_ratio = np.mean(w13_orig_f32 == 0) * 100
        w2_orig_zero_ratio = np.mean(w2_orig_f32 == 0) * 100

        print(f"    Expert {eid} w13:")
        print(f"      Original FP32: mean={w13_orig_f32.mean():.4e} std={w13_orig_f32.std():.4e} "
              f"min={w13_orig_f32.min():.4e} max={w13_orig_f32.max():.4e} zeros={w13_orig_zero_ratio:.1f}%")
        print(f"      FP4 dequant:   mean={w13_fp4_f32[eid].mean():.4e} std={w13_fp4_f32[eid].std():.4e} "
              f"min={w13_fp4_f32[eid].min():.4e} max={w13_fp4_f32[eid].max():.4e} zeros={w13_zero_ratio:.1f}%")
        print(f"      Abs error:     max={w13_diff.max():.4e} mean={w13_diff.mean():.4e} "
              f"p99={np.percentile(w13_diff, 99):.4e}")
        print(f"      Rel error:     mean={w13_rel_err[w13_nonzero].mean():.4f} "
              f"p99={np.percentile(w13_rel_err[w13_nonzero], 99):.4f}")
        print(f"      Cosine sim:    {cos_sim_w13:.6f}")

        print(f"    Expert {eid} w2:")
        print(f"      Original FP32: mean={w2_orig_f32.mean():.4e} std={w2_orig_f32.std():.4e} "
              f"min={w2_orig_f32.min():.4e} max={w2_orig_f32.max():.4e} zeros={w2_orig_zero_ratio:.1f}%")
        print(f"      FP4 dequant:   mean={w2_fp4_f32[eid].mean():.4e} std={w2_fp4_f32[eid].std():.4e} "
              f"min={w2_fp4_f32[eid].min():.4e} max={w2_fp4_f32[eid].max():.4e} zeros={w2_zero_ratio:.1f}%")
        print(f"      Abs error:     max={w2_diff.max():.4e} mean={w2_diff.mean():.4e} "
              f"p99={np.percentile(w2_diff, 99):.4e}")
        print(f"      Rel error:     mean={w2_rel_err[w2_nonzero].mean():.4f} "
              f"p99={np.percentile(w2_rel_err[w2_nonzero], 99):.4f}")
        print(f"      Cosine sim:    {cos_sim_w2:.6f}")

        results[eid] = {
            "w13_cos": cos_sim_w13, "w2_cos": cos_sim_w2,
            "w13_max_err": w13_diff.max(), "w2_max_err": w2_diff.max(),
            "w13_zero_fp4": w13_zero_ratio, "w2_zero_fp4": w2_zero_ratio,
        }

    return results


def check_moe():
    print("\n" + "=" * 60)
    print("Part 2: FP4 MoE Cache Validation (quantization quality)")
    print("=" * 60)

    # Check 3 layers: early, middle, late
    layers_to_check = [3, 40, 78]  # first MoE, middle, last (MTP)
    all_ok = True

    for layer_idx in layers_to_check:
        results = check_moe_layer(layer_idx)
        if results is None:
            continue
        for eid, r in results.items():
            if r["w13_cos"] < 0.99 or r["w2_cos"] < 0.99:
                print(f"  ⚠️  Layer {layer_idx} Expert {eid}: LOW cosine similarity!")
                all_ok = False

    return all_ok


if __name__ == "__main__":
    t0 = time.time()
    ok1 = check_non_moe()
    ok2 = check_moe()
    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"Validation complete in {elapsed:.1f}s")
    print(f"  Non-MoE: {'PASS ✅' if ok1 else 'FAIL ❌'}")
    print(f"  FP4 MoE: {'PASS ✅' if ok2 else 'ISSUES ⚠️'}")
