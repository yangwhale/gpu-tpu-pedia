#!/usr/bin/env python3
"""Test: compile UNet with de_mod and measure performance difference."""
import os, sys, warnings, logging, time, gc

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTHONUNBUFFERED'] = '1'
warnings.filterwarnings('ignore')
for n in ['root', '', 'diffusers', 'transformers']:
    logging.getLogger(n).setLevel(logging.ERROR)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import functools
from pathlib import Path
from PIL import Image
from torchvision import transforms

# Import from generate_torchax.py
from generate_torchax import (
    load_s3diff_model, preprocess_image, encode_prompts,
    compute_degradation_modulation, setup_pytree_registrations,
    move_module_to_xla, override_op, get_hbm_usage, print_hbm,
    DEFAULT_POS_PROMPT, DEFAULT_NEG_PROMPT, GUIDANCE_SCALE,
    my_lora_fwd, ResidualBlockNoBN, DEResNet, get_layer_number,
)

def main():
    import jax
    import jax.numpy as jnp
    from jax.experimental import mesh_utils
    from jax.sharding import Mesh

    print("=== Loading model on CPU ===", flush=True)
    from huggingface_hub import snapshot_download, hf_hub_download
    sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")
    pretrained_path = hf_hub_download(repo_id="zhangap/S3Diff", filename="s3diff.pkl")
    de_net_path = "assets/mm-realsr/de_net.pth"

    components = load_s3diff_model(sd_path, pretrained_path, de_net_path)

    # Preprocess
    im_lr, im_lr_resize, im_lr_resize_norm, (resize_h, resize_w) = preprocess_image("test_images/real_photo_128.png")
    print(f"  Input: {im_lr.shape[2]}x{im_lr.shape[3]} -> {resize_h}x{resize_w}", flush=True)

    # Encode prompts on CPU
    pos_enc, neg_enc = encode_prompts(
        components['tokenizer'], components['text_encoder'],
        DEFAULT_POS_PROMPT, DEFAULT_NEG_PROMPT
    )
    del components['tokenizer'], components['text_encoder']; gc.collect()

    # Degradation estimation + modulation on CPU
    with torch.no_grad():
        deg_score = components['de_net'](im_lr.float())
        compute_degradation_modulation(components, deg_score)
    print(f"  Degradation: blur={deg_score[0,0]:.4f}, noise={deg_score[0,1]:.4f}", flush=True)
    del components['de_net']; gc.collect()

    # Enable torchax
    torch.set_default_dtype(torch.bfloat16)
    setup_pytree_registrations()

    import torchax
    from torchax.ops import jaten
    from splash_attention_utils import sdpa_reference

    torchax.enable_globally()
    env = torchax.default_env()
    mesh = Mesh(mesh_utils.create_device_mesh((1,), allow_split_physical_axes=True), ("x",))

    # Override ops
    def sdpa_impl(query, key, value, attn_mask=None, dropout_p=0.0,
                  is_causal=False, scale=None, enable_gqa=False):
        return sdpa_reference(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)

    def conv2d_impl(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, *, env=env):
        jinput, jweight, jbias = env.t2j_iso((input, weight, bias))
        if jinput.dtype != jweight.dtype:
            jinput = jinput.astype(jweight.dtype)
        if jbias is not None and jbias.dtype != jweight.dtype:
            jbias = jbias.astype(jweight.dtype)
        res = jaten._aten_conv2d(jinput, jweight, jbias, stride, padding, dilation, groups)
        return env.j2t_iso(res)

    override_op(env, F.conv2d, functools.partial(conv2d_impl, env=env))
    override_op(env, F.scaled_dot_product_attention, sdpa_impl)

    # Move to XLA
    print("\n=== Moving to XLA ===", flush=True)
    unet = components['unet']
    vae = components['vae']

    for p in vae.parameters(): p.data = p.data.to(torch.bfloat16)
    for b in vae.buffers(): b.data = b.data.to(torch.bfloat16)
    for p in unet.parameters(): p.data = p.data.to(torch.bfloat16)
    for b in unet.buffers(): b.data = b.data.to(torch.bfloat16)

    move_module_to_xla(env, vae)
    move_module_to_xla(env, unet)

    for name in ['vae_de_mlp', 'unet_de_mlp', 'vae_block_mlp', 'unet_block_mlp',
                 'vae_fuse_mlp', 'unet_fuse_mlp', 'vae_block_embeddings', 'unet_block_embeddings']:
        move_module_to_xla(env, components[name])

    for _, module in vae.named_modules():
        if hasattr(module, 'de_mod'):
            module.de_mod = env.to_xla(module.de_mod.to(torch.bfloat16))
    for _, module in unet.named_modules():
        if hasattr(module, 'de_mod'):
            module.de_mod = env.to_xla(module.de_mod.to(torch.bfloat16))

    # Move inputs to XLA
    with env:
        im_lr_resize_norm = im_lr_resize_norm.to('jax').to(torch.bfloat16)
        pos_enc = pos_enc.to('jax').to(torch.bfloat16)
        neg_enc = neg_enc.to('jax').to(torch.bfloat16)
        components['scheduler'].alphas_cumprod = components['scheduler'].alphas_cumprod.to('jax')

    vae.decoder = torchax.compile(vae.decoder)
    timestep_xla = env.j2t_iso(jnp.array([999], dtype=jnp.int32))

    # ============================================================
    # TEST 1: UNet WITHOUT compile (current behavior)
    # ============================================================
    print(f"\n{'='*60}", flush=True)
    print("TEST 1: UNet WITHOUT compile (enable_globally only)", flush=True)
    print(f"{'='*60}", flush=True)

    with mesh:
        lq_latent = vae.encode(im_lr_resize_norm).latent_dist.sample() * vae.config.scaling_factor
        jax.effects_barrier()
        print("  VAE encode done", flush=True)

        # Warmup UNet
        t0 = time.perf_counter()
        pred = unet(lq_latent, timestep_xla, encoder_hidden_states=pos_enc, return_dict=False)[0]
        jax.effects_barrier()
        print(f"  UNet warmup (uncompiled): {time.perf_counter()-t0:.2f}s", flush=True)

        # Benchmark 3 iters
        times_uncompiled = []
        for i in range(3):
            t0 = time.perf_counter()
            pred = unet(lq_latent, timestep_xla, encoder_hidden_states=pos_enc, return_dict=False)[0]
            jax.effects_barrier()
            t1 = time.perf_counter() - t0
            times_uncompiled.append(t1)
            print(f"  UNet iter {i+1}: {t1:.3f}s", flush=True)

    avg_uncompiled = sum(times_uncompiled) / len(times_uncompiled)
    print(f"  AVG uncompiled: {avg_uncompiled:.3f}s", flush=True)

    # ============================================================
    # TEST 2: UNet WITH compile
    # ============================================================
    print(f"\n{'='*60}", flush=True)
    print("TEST 2: UNet WITH torchax.compile()", flush=True)
    print(f"{'='*60}", flush=True)

    print("  Creating JittableModule...", flush=True)
    compiled_unet = torchax.compile(unet)

    # Check de_mod in buffers
    de_mod_keys = [k for k in compiled_unet.buffers if 'de_mod' in k]
    print(f"  de_mod keys in buffers: {len(de_mod_keys)}", flush=True)
    if de_mod_keys:
        print(f"  Example key: {de_mod_keys[0]}", flush=True)
        print(f"  Example shape: {list(compiled_unet.buffers[de_mod_keys[0]].shape)}", flush=True)

    with mesh:
        # Warmup (first call = JIT trace + compile)
        print("  Running JIT warmup (this may take a while)...", flush=True)
        t0 = time.perf_counter()
        pred_compiled = compiled_unet(lq_latent, timestep_xla, encoder_hidden_states=pos_enc, return_dict=False)[0]
        jax.effects_barrier()
        jit_time = time.perf_counter() - t0
        print(f"  UNet JIT warmup: {jit_time:.2f}s", flush=True)

        # Benchmark 3 iters
        times_compiled = []
        for i in range(3):
            t0 = time.perf_counter()
            pred_compiled = compiled_unet(lq_latent, timestep_xla, encoder_hidden_states=pos_enc, return_dict=False)[0]
            jax.effects_barrier()
            t1 = time.perf_counter() - t0
            times_compiled.append(t1)
            print(f"  UNet iter {i+1}: {t1:.3f}s", flush=True)

    avg_compiled = sum(times_compiled) / len(times_compiled)
    speedup = avg_uncompiled / avg_compiled if avg_compiled > 0 else float('inf')

    # Correctness check
    pred_np = np.array(pred._elem.astype(jnp.float32))
    pred_c_np = np.array(pred_compiled._elem.astype(jnp.float32))
    max_diff = np.max(np.abs(pred_np - pred_c_np))

    print(f"\n{'='*60}", flush=True)
    print("RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Uncompiled avg: {avg_uncompiled:.3f}s", flush=True)
    print(f"  Compiled avg:   {avg_compiled:.3f}s", flush=True)
    print(f"  Speedup:        {speedup:.1f}x", flush=True)
    print(f"  JIT overhead:   {jit_time:.1f}s (one-time)", flush=True)
    print(f"  de_mod buffers: {len(de_mod_keys)} keys", flush=True)
    print(f"  Max output diff: {max_diff:.6f}", flush=True)
    print(f"  Correct:        {'YES' if max_diff < 0.1 else 'NO'}", flush=True)

    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
