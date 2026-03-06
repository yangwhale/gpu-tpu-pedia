#!/usr/bin/env python3
"""Multi-image, multi-size benchmark for S3Diff on TPU v6e.

Tests different images and sizes, runs 3 iterations each to show
JIT compile / warmup / steady-state performance.
Saves results as JSON + SR output images.
"""
import os, sys, warnings, logging, time, gc, json

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
    move_module_to_xla, override_op, postprocess_output,
    DEFAULT_POS_PROMPT, DEFAULT_NEG_PROMPT, GUIDANCE_SCALE,
    my_lora_fwd, ResidualBlockNoBN, DEResNet, get_layer_number,
)


# Test matrix: (name, path, description)
TEST_IMAGES = [
    ("photo_64",      "test_images/real_photo_64.png",   "Real Photo 64x64"),
    ("photo_128",     "test_images/real_photo_128.png",  "Real Photo 128x128"),
    ("photo_192",     "test_images/real_photo_192.png",  "Real Photo 192x192"),
    ("photo_256",     "test_images/real_photo_256.jpg",  "Real Photo 256x256"),
    ("dog_64",        "test_images/real_dog_64.png",     "Dog 64x64"),
    ("dog_128",       "test_images/real_dog_128.png",    "Dog 128x128"),
    ("dog_256",       "test_images/real_dog_256.png",    "Dog 256x256"),
    ("city_128",      "test_images/cityscape_128.png",   "Cityscape 128x128"),
    ("city_256",      "test_images/cityscape_256.png",   "Cityscape 256x256"),
]

NUM_ITERS = 3


def run_single_inference(vae, unet, scheduler, im_lr_resize_norm, pos_enc, neg_enc, mesh, env):
    """Run one full inference pass, return per-stage times."""
    import jax
    import jax.numpy as jnp

    with mesh:
        # VAE Encode
        t0 = time.perf_counter()
        lq_latent = vae.encode(im_lr_resize_norm).latent_dist.sample() * vae.config.scaling_factor
        jax.effects_barrier()
        vae_enc = time.perf_counter() - t0

        # UNet
        t1 = time.perf_counter()
        timestep_xla = env.j2t_iso(jnp.array([999], dtype=jnp.int32))
        pos_pred = unet(lq_latent, timestep_xla, encoder_hidden_states=pos_enc, return_dict=False)[0]
        neg_pred = unet(lq_latent, timestep_xla, encoder_hidden_states=neg_enc, return_dict=False)[0]
        model_pred = neg_pred + GUIDANCE_SCALE * (pos_pred - neg_pred)
        jax.effects_barrier()
        unet_t = time.perf_counter() - t1

        # Scheduler
        x_denoised = scheduler.step(model_pred, 999, lq_latent, return_dict=True).prev_sample

        # VAE Decode
        t2 = time.perf_counter()
        output = vae.decode(x_denoised / vae.config.scaling_factor).sample
        jax.effects_barrier()
        vae_dec = time.perf_counter() - t2

    total = vae_enc + unet_t + vae_dec
    return output, {
        'vae_encode': round(vae_enc, 3),
        'unet': round(unet_t, 3),
        'vae_decode': round(vae_dec, 3),
        'total': round(total, 3),
    }


def main():
    import jax
    import jax.numpy as jnp
    from jax.experimental import mesh_utils
    from jax.sharding import Mesh

    jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax_cache"))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    print("=" * 70, flush=True)
    print("S3Diff Multi-Image Benchmark — TPU v6e", flush=True)
    print("=" * 70, flush=True)

    # Load model
    print("\n[1/4] Loading model...", flush=True)
    from huggingface_hub import snapshot_download, hf_hub_download
    sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")
    pretrained_path = hf_hub_download(repo_id="zhangap/S3Diff", filename="s3diff.pkl")

    components = load_s3diff_model(sd_path, pretrained_path, "assets/mm-realsr/de_net.pth")

    # Encode prompts
    pos_enc_cpu, neg_enc_cpu = encode_prompts(
        components['tokenizer'], components['text_encoder'],
        DEFAULT_POS_PROMPT, DEFAULT_NEG_PROMPT
    )
    del components['tokenizer'], components['text_encoder']; gc.collect()

    # Pre-compute degradation scores for ALL images on CPU (before torchax)
    print("\n[1.5/4] Pre-computing degradation for all images...", flush=True)
    de_net = components['de_net']
    precomputed = {}
    for name, path, desc in TEST_IMAGES:
        im_lr, im_lr_resize, im_lr_resize_norm, (resize_h, resize_w) = preprocess_image(path)
        with torch.no_grad():
            deg_score = de_net(im_lr.float())
            compute_degradation_modulation(components, deg_score)
        # Save the de_mod values (CPU tensors) for each LoRA layer
        vae_de_mods = {}
        for layer_name, module in components['vae'].named_modules():
            if hasattr(module, 'de_mod'):
                vae_de_mods[layer_name] = module.de_mod.clone()
        unet_de_mods = {}
        for layer_name, module in components['unet'].named_modules():
            if hasattr(module, 'de_mod'):
                unet_de_mods[layer_name] = module.de_mod.clone()
        precomputed[name] = {
            'deg_score': deg_score,
            'vae_de_mods': vae_de_mods,
            'unet_de_mods': unet_de_mods,
            'preprocess': (im_lr, im_lr_resize, im_lr_resize_norm, resize_h, resize_w),
        }
        print(f"  {name}: blur={deg_score[0,0]:.4f} noise={deg_score[0,1]:.4f}", flush=True)
    del de_net, components['de_net']; gc.collect()

    # Setup torchax
    print("\n[2/4] Setting up torchax...", flush=True)
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

    # Move models to XLA (once)
    print("\n[3/4] Moving models to XLA...", flush=True)
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
    components['W'] = env.to_xla(components['W'])

    # Move prompts to XLA
    with env:
        pos_enc = pos_enc_cpu.to('jax').to(torch.bfloat16)
        neg_enc = neg_enc_cpu.to('jax').to(torch.bfloat16)
        components['scheduler'].alphas_cumprod = components['scheduler'].alphas_cumprod.to('jax')

    # Compile VAE decoder (no LoRA)
    vae.decoder = torchax.compile(vae.decoder)

    print("  Models on XLA", flush=True)

    # Run benchmarks
    print(f"\n[4/4] Running benchmarks ({len(TEST_IMAGES)} images x {NUM_ITERS} iters)...", flush=True)
    print("=" * 70, flush=True)

    all_results = []
    output_dir = Path("test_images/benchmark_outputs")
    output_dir.mkdir(exist_ok=True)

    # Track which shapes we've already compiled for (to identify recompilation)
    seen_shapes = set()

    for img_idx, (name, path, desc) in enumerate(TEST_IMAGES):
        print(f"\n--- [{img_idx+1}/{len(TEST_IMAGES)}] {desc} ({path}) ---", flush=True)

        # Use precomputed data
        pre = precomputed[name]
        im_lr, im_lr_resize, im_lr_resize_norm, resize_h, resize_w = pre['preprocess']
        input_h, input_w = im_lr.shape[2], im_lr.shape[3]
        output_h, output_w = resize_h, resize_w
        padded_shape = list(im_lr_resize_norm.shape)
        latent_shape_str = f"[1,4,{padded_shape[2]//8},{padded_shape[3]//8}]"

        is_new_shape = (padded_shape[2], padded_shape[3]) not in seen_shapes
        seen_shapes.add((padded_shape[2], padded_shape[3]))

        print(f"  Input: {input_h}x{input_w} -> padded {padded_shape[2]}x{padded_shape[3]} -> output {output_h}x{output_w}", flush=True)
        print(f"  Latent: {latent_shape_str} | New shape: {'YES (will recompile)' if is_new_shape else 'NO (cached)'}", flush=True)

        deg_score = pre['deg_score']
        blur_val = deg_score[0, 0].item()
        noise_val = deg_score[0, 1].item()
        print(f"  Degradation: blur={blur_val:.4f}, noise={noise_val:.4f}", flush=True)

        # Restore precomputed de_mod values and convert to XLA bf16
        for layer_name, module in vae.named_modules():
            if layer_name in pre['vae_de_mods']:
                module.de_mod = env.to_xla(pre['vae_de_mods'][layer_name].to(torch.bfloat16))
        for layer_name, module in unet.named_modules():
            if layer_name in pre['unet_de_mods']:
                module.de_mod = env.to_xla(pre['unet_de_mods'][layer_name].to(torch.bfloat16))

        # Re-compile UNet with updated de_mod buffers
        compiled_unet = torchax.compile(
            unet, torchax.CompileOptions(jax_jit_kwargs={'static_argnames': ('return_dict',)})
        )

        # Move input to XLA
        with env:
            input_xla = im_lr_resize_norm.to('jax').to(torch.bfloat16)

        img_result = {
            'name': name,
            'description': desc,
            'path': path,
            'input_size': f"{input_h}x{input_w}",
            'output_size': f"{output_h}x{output_w}",
            'padded_size': f"{padded_shape[2]}x{padded_shape[3]}",
            'latent_shape': latent_shape_str,
            'is_new_shape': is_new_shape,
            'degradation': {'blur': round(blur_val, 4), 'noise': round(noise_val, 4)},
            'iterations': [],
        }

        for i in range(NUM_ITERS):
            output, times = run_single_inference(
                vae, compiled_unet, components['scheduler'],
                input_xla, pos_enc, neg_enc, mesh, env
            )
            label = "compile" if i == 0 and is_new_shape else ("warmup" if i == 0 else "steady")
            times['label'] = label
            img_result['iterations'].append(times)
            print(f"  Iter {i+1} ({label}): VAE-enc={times['vae_encode']:.3f}s  UNet={times['unet']:.3f}s  VAE-dec={times['vae_decode']:.3f}s  Total={times['total']:.3f}s", flush=True)

        # Save SR output from last iteration
        sr_img = postprocess_output(output, resize_h, resize_w, im_lr_resize, align_method='wavelet')
        out_path = output_dir / f"{name}_sr.png"
        sr_img.save(str(out_path))
        img_result['output_path'] = str(out_path)
        print(f"  Saved: {out_path}", flush=True)

        all_results.append(img_result)

    # Save JSON results
    results_path = "test_images/benchmark_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}", flush=True)

    # Print summary
    print(f"\n{'='*70}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Image':<20} {'Size':<10} {'Iter1':>8} {'Iter2':>8} {'Iter3':>8} {'NewShape':>10}", flush=True)
    print("-" * 70, flush=True)
    for r in all_results:
        iters = r['iterations']
        t1 = iters[0]['total']
        t2 = iters[1]['total']
        t3 = iters[2]['total']
        ns = "YES" if r['is_new_shape'] else "no"
        print(f"{r['name']:<20} {r['input_size']:<10} {t1:>7.3f}s {t2:>7.3f}s {t3:>7.3f}s {ns:>10}", flush=True)

    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
