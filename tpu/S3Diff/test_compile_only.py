#!/usr/bin/env python3
"""Test: ONLY compile UNet and benchmark. Skip uncompiled test."""
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

from generate_torchax import (
    load_s3diff_model, preprocess_image, encode_prompts,
    compute_degradation_modulation, setup_pytree_registrations,
    move_module_to_xla, override_op,
    DEFAULT_POS_PROMPT, DEFAULT_NEG_PROMPT,
    my_lora_fwd, ResidualBlockNoBN, DEResNet, get_layer_number,
)

def main():
    import jax
    import jax.numpy as jnp
    from jax.experimental import mesh_utils
    from jax.sharding import Mesh

    jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax_cache"))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    print("=== Loading model ===", flush=True)
    from huggingface_hub import snapshot_download, hf_hub_download
    sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")
    pretrained_path = hf_hub_download(repo_id="zhangap/S3Diff", filename="s3diff.pkl")

    components = load_s3diff_model(sd_path, pretrained_path, "assets/mm-realsr/de_net.pth")
    im_lr, im_lr_resize, im_lr_resize_norm, (resize_h, resize_w) = preprocess_image("test_images/real_photo_128.png")
    print(f"  Input: {im_lr.shape[2]}x{im_lr.shape[3]}", flush=True)

    pos_enc, neg_enc = encode_prompts(
        components['tokenizer'], components['text_encoder'],
        DEFAULT_POS_PROMPT, DEFAULT_NEG_PROMPT
    )
    del components['tokenizer'], components['text_encoder']; gc.collect()

    with torch.no_grad():
        deg_score = components['de_net'](im_lr.float())
        compute_degradation_modulation(components, deg_score)
    del components['de_net']; gc.collect()

    torch.set_default_dtype(torch.bfloat16)
    setup_pytree_registrations()

    import torchax
    from torchax.ops import jaten
    from splash_attention_utils import sdpa_reference

    torchax.enable_globally()
    env = torchax.default_env()
    mesh = Mesh(mesh_utils.create_device_mesh((1,), allow_split_physical_axes=True), ("x",))

    def sdpa_impl(query, key, value, attn_mask=None, dropout_p=0.0,
                  is_causal=False, scale=None, enable_gqa=False):
        return sdpa_reference(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)

    def conv2d_impl(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, *, env=env):
        jinput, jweight, jbias = env.t2j_iso((input, weight, bias))
        if jinput.dtype != jweight.dtype: jinput = jinput.astype(jweight.dtype)
        if jbias is not None and jbias.dtype != jweight.dtype: jbias = jbias.astype(jweight.dtype)
        res = jaten._aten_conv2d(jinput, jweight, jbias, stride, padding, dilation, groups)
        return env.j2t_iso(res)

    override_op(env, F.conv2d, functools.partial(conv2d_impl, env=env))
    override_op(env, F.scaled_dot_product_attention, sdpa_impl)

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

    with env:
        im_lr_resize_norm = im_lr_resize_norm.to('jax').to(torch.bfloat16)
        pos_enc = pos_enc.to('jax').to(torch.bfloat16)
        neg_enc = neg_enc.to('jax').to(torch.bfloat16)
        components['scheduler'].alphas_cumprod = components['scheduler'].alphas_cumprod.to('jax')

    vae.decoder = torchax.compile(vae.decoder)
    timestep_xla = env.j2t_iso(jnp.array([999], dtype=jnp.int32))

    # === Step 1: Quick verify extract_all_buffers sees de_mod ===
    print("\n=== Checking extract_all_buffers ===", flush=True)
    from torchax.interop import extract_all_buffers
    params, buffers = extract_all_buffers(unet)
    de_mod_keys = [k for k in buffers if 'de_mod' in k]
    bias_keys = [k for k in buffers if k.endswith('.bias')]
    print(f"  Total params: {len(params)}", flush=True)
    print(f"  Total buffers: {len(buffers)}", flush=True)
    print(f"  de_mod keys: {len(de_mod_keys)}", flush=True)
    print(f"  .bias keys: {len(bias_keys)} (should be 0 after fix)", flush=True)
    if de_mod_keys:
        print(f"  Example de_mod: {de_mod_keys[0]} shape={list(buffers[de_mod_keys[0]].shape)}", flush=True)
    if bias_keys:
        print(f"  WARNING: bias keys found: {bias_keys[:3]}", flush=True)

    # === Step 2: Compile UNet ===
    print("\n=== Compiling UNet ===", flush=True)
    compiled_unet = torchax.compile(unet)
    de_mod_in_compiled = [k for k in compiled_unet.buffers if 'de_mod' in k]
    print(f"  JittableModule created", flush=True)
    print(f"  de_mod in compiled buffers: {len(de_mod_in_compiled)}", flush=True)

    # === Step 3: VAE encode (uncompiled, keep as-is) ===
    with mesh:
        lq_latent = vae.encode(im_lr_resize_norm).latent_dist.sample() * vae.config.scaling_factor
        jax.effects_barrier()
        print(f"  VAE encode done, latent: {list(lq_latent.shape)}", flush=True)

    # === Step 4: Compiled UNet warmup (JIT trace + XLA compile) ===
    print("\n=== UNet JIT Warmup ===", flush=True)
    with mesh:
        t0 = time.perf_counter()
        pred = compiled_unet(lq_latent, timestep_xla, encoder_hidden_states=pos_enc).sample
        jax.effects_barrier()
        jit_time = time.perf_counter() - t0
        print(f"  JIT warmup: {jit_time:.1f}s", flush=True)

    # === Step 5: Benchmark ===
    print("\n=== Benchmark (compiled) ===", flush=True)
    times = []
    with mesh:
        for i in range(5):
            t0 = time.perf_counter()
            pred = compiled_unet(lq_latent, timestep_xla, encoder_hidden_states=pos_enc).sample
            jax.effects_barrier()
            t1 = time.perf_counter() - t0
            times.append(t1)
            print(f"  Iter {i+1}: {t1:.3f}s", flush=True)

    avg = sum(times) / len(times)
    min_t = min(times)
    print(f"\n{'='*60}", flush=True)
    print(f"RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Uncompiled (previous test): ~11.0s", flush=True)
    print(f"  Compiled avg: {avg:.3f}s", flush=True)
    print(f"  Compiled min: {min_t:.3f}s", flush=True)
    print(f"  Speedup: {11.0/avg:.1f}x (vs 11.0s uncompiled)", flush=True)
    print(f"  JIT overhead: {jit_time:.1f}s (one-time)", flush=True)

    sys.stdout.flush()
    os._exit(0)

if __name__ == '__main__':
    main()
