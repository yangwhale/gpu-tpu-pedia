#!/usr/bin/env python3
"""Bus.jpg multi-size benchmark for S3Diff on TPU v6e, target_size=2048."""
import os, sys, warnings, logging, time, gc, json, math

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTHONUNBUFFERED'] = '1'
warnings.filterwarnings('ignore')
for n in ['root', '', 'diffusers', 'transformers']:
    logging.getLogger(n).setLevel(logging.ERROR)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from pathlib import Path
from PIL import Image
from torchvision import transforms

from generate_torchax import (
    load_s3diff_model, preprocess_image, encode_prompts,
    compute_degradation_modulation, setup_pytree_registrations,
    move_module_to_xla, override_op, postprocess_output,
    DEFAULT_POS_PROMPT, DEFAULT_NEG_PROMPT, GUIDANCE_SCALE,
)

TARGET_SIZE = int(os.environ.get('TARGET_SIZE', '2048'))
NUM_ITERS = int(os.environ.get('NUM_ITERS', '3'))
SYNC_STAGES = not os.environ.get('NOSYNC')
PROFILE_DIR = os.environ.get('PROFILE')  # Set to a dir path to enable JAX profiler trace

TEST_IMAGES_2048 = [
    ("bus1_maxsize1080", "test_images/bus_benchmark/bus_maxsize1080.jpg"),
    ("bus2_maxsize1280", "test_images/bus_benchmark/bus_maxsize1280.jpg"),
    ("bus3_maxsize1440", "test_images/bus_benchmark/bus_maxsize1440.jpg"),
    ("bus4_maxsize900",  "test_images/bus_benchmark/bus_maxsize900.jpg"),
    ("bus5_maxsize512",  "test_images/bus_benchmark/bus_maxsize512.jpg"),
    ("bus6_maxsize1621", "test_images/bus_benchmark/bus_maxsize1621.jpg"),
    ("bus7_maxsize1707", "test_images/bus_benchmark/bus_maxsize1707.jpg"),
]

TEST_IMAGES_1800 = [
    ("bus4_maxsize900",  "test_images/bus_benchmark/bus_maxsize900.jpg"),
    ("bus5_maxsize512",  "test_images/bus_benchmark/bus_maxsize512.jpg"),
]

TEST_IMAGES_1080 = [
    ("bus1_maxsize1080", "test_images/bus_benchmark/bus_maxsize1080.jpg"),
]

if os.environ.get('SINGLE_1080'):
    TEST_IMAGES = TEST_IMAGES_1080
elif TARGET_SIZE == 1800:
    TEST_IMAGES = TEST_IMAGES_1800
else:
    TEST_IMAGES = TEST_IMAGES_2048


def run_single(vae, unet, scheduler, im_lr_resize_norm, pos_enc, neg_enc, mesh, env, sync_stages=True):
    import jax
    import jax.numpy as jnp

    with mesh:
        t0 = time.perf_counter()
        with jax.profiler.TraceAnnotation("stage/vae_encode"):
            lq_latent = vae.encode(im_lr_resize_norm).latent_dist.sample() * vae.config.scaling_factor
            if sync_stages and hasattr(lq_latent, '_elem'):
                lq_latent._elem.block_until_ready()
        vae_enc = time.perf_counter() - t0

        t1 = time.perf_counter()
        with jax.profiler.TraceAnnotation("stage/unet"):
            timestep_xla = env.j2t_iso(jnp.array([999], dtype=jnp.int32))
            pos_pred = unet(lq_latent, timestep_xla, encoder_hidden_states=pos_enc, return_dict=False)[0]
            neg_pred = unet(lq_latent, timestep_xla, encoder_hidden_states=neg_enc, return_dict=False)[0]
            model_pred = neg_pred + GUIDANCE_SCALE * (pos_pred - neg_pred)
            if sync_stages and hasattr(model_pred, '_elem'):
                model_pred._elem.block_until_ready()
        unet_t = time.perf_counter() - t1

        with jax.profiler.TraceAnnotation("stage/scheduler_step"):
            x_denoised = scheduler.step(model_pred, 999, lq_latent, return_dict=True).prev_sample

        t2 = time.perf_counter()
        with jax.profiler.TraceAnnotation("stage/vae_decode"):
            output = vae.decode(x_denoised / vae.config.scaling_factor).sample
            # Always sync at the end to get total wall time
            if hasattr(output, '_elem'):
                output._elem.block_until_ready()
        vae_dec = time.perf_counter() - t2

    return output, {
        'vae_encode': round(vae_enc, 3),
        'unet': round(unet_t, 3),
        'vae_decode': round(vae_dec, 3),
        'total': round(vae_enc + unet_t + vae_dec, 3),
    }


def main():
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh

    jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax_cache"))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    print("=" * 70)
    print(f"S3Diff Bus.jpg Benchmark — target_size={TARGET_SIZE}")
    print("=" * 70)

    # Load model
    print("\n[1/4] Loading model...")
    from huggingface_hub import snapshot_download, hf_hub_download
    sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")
    pretrained_path = hf_hub_download(repo_id="zhangap/S3Diff", filename="s3diff.pkl")
    components = load_s3diff_model(sd_path, pretrained_path, "assets/mm-realsr/de_net.pth")

    pos_enc_cpu, neg_enc_cpu = encode_prompts(
        components['tokenizer'], components['text_encoder'],
        DEFAULT_POS_PROMPT, DEFAULT_NEG_PROMPT
    )
    del components['tokenizer'], components['text_encoder']; gc.collect()

    # Precompute degradation on CPU
    print("\n[1.5/4] Precompute degradation...")
    de_net = components['de_net']
    precomputed = {}
    for name, path in TEST_IMAGES:
        im_lr, im_lr_resize, im_lr_resize_norm, (resize_h, resize_w) = preprocess_image(path, target_size=TARGET_SIZE)
        with torch.no_grad():
            deg_score = de_net(im_lr.float())
            compute_degradation_modulation(components, deg_score)
        vae_de_mods = {}
        for ln, mod in components['vae'].named_modules():
            if hasattr(mod, 'de_mod'):
                vae_de_mods[ln] = mod.de_mod.clone()
        unet_de_mods = {}
        for ln, mod in components['unet'].named_modules():
            if hasattr(mod, 'de_mod'):
                unet_de_mods[ln] = mod.de_mod.clone()
        precomputed[name] = {
            'vae_de_mods': vae_de_mods,
            'unet_de_mods': unet_de_mods,
            'preprocess': (im_lr, im_lr_resize, im_lr_resize_norm, resize_h, resize_w),
        }
        ori_h, ori_w = im_lr.shape[2], im_lr.shape[3]
        padded_h, padded_w = im_lr_resize_norm.shape[2], im_lr_resize_norm.shape[3]
        lat_h, lat_w = padded_h // 8, padded_w // 8
        print(f"  {name}: {ori_w}x{ori_h} -> latent {lat_w}x{lat_h}")
    del de_net, components['de_net']; gc.collect()

    # Setup torchax
    print("\n[2/4] Setup torchax...")
    torch.set_default_dtype(torch.bfloat16)
    setup_pytree_registrations()

    import torchax
    from torchax.ops import jaten
    from splash_attention_utils import sdpa_reference

    torchax.enable_globally()
    env = torchax.default_env()
    mesh = Mesh(np.array(jax.devices()[:1]), ("x",))

    def sdpa_impl(query, key, value, attn_mask=None, dropout_p=0.0,
                  is_causal=False, scale=None, enable_gqa=False):
        return sdpa_reference(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)

    def conv2d_impl(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, *, env=env):
        jinput, jweight, jbias = env.t2j_iso((input, weight, bias))
        if jinput.dtype != jweight.dtype:
            target_dtype = jnp.promote_types(jinput.dtype, jweight.dtype)
            jinput = jinput.astype(target_dtype)
            jweight = jweight.astype(target_dtype)
            if jbias is not None:
                jbias = jbias.astype(target_dtype)
        elif jbias is not None and jbias.dtype != jweight.dtype:
            jbias = jbias.astype(jweight.dtype)
        res = jaten._aten_conv2d(jinput, jweight, jbias, stride, padding, dilation, groups)
        return env.j2t_iso(res)

    override_op(env, F.conv2d, functools.partial(conv2d_impl, env=env))
    override_op(env, F.scaled_dot_product_attention, sdpa_impl)

    # Move to XLA
    print("\n[3/4] Move to XLA...")
    unet = components['unet']
    vae = components['vae']

    for p in vae.parameters(): p.data = p.data.to(torch.bfloat16)
    for b in vae.buffers(): b.data = b.data.to(torch.bfloat16)
    for p in unet.parameters(): p.data = p.data.to(torch.bfloat16)
    for b in unet.buffers(): b.data = b.data.to(torch.bfloat16)

    move_module_to_xla(env, vae)
    move_module_to_xla(env, unet)

    for n in ['vae_de_mlp', 'unet_de_mlp', 'vae_block_mlp', 'unet_block_mlp',
              'vae_fuse_mlp', 'unet_fuse_mlp', 'vae_block_embeddings', 'unet_block_embeddings']:
        move_module_to_xla(env, components[n])
    components['W'] = env.to_xla(components['W'])

    with env:
        pos_enc = pos_enc_cpu.to('jax').to(torch.bfloat16)
        neg_enc = neg_enc_cpu.to('jax').to(torch.bfloat16)
        components['scheduler'].alphas_cumprod = components['scheduler'].alphas_cumprod.to('jax')

    # de_mod must be XLA tensors BEFORE compile (otherwise JIT tracing hits CPU tensors)
    first_pre = list(precomputed.values())[0]
    for ln, mod in vae.named_modules():
        if ln in first_pre['vae_de_mods']:
            mod.de_mod = env.to_xla(first_pre['vae_de_mods'][ln].to(torch.bfloat16))
    for ln, mod in unet.named_modules():
        if ln in first_pre['unet_de_mods']:
            mod.de_mod = env.to_xla(first_pre['unet_de_mods'][ln].to(torch.bfloat16))

    vae.encoder = torchax.compile(vae.encoder)
    vae.decoder = torchax.compile(vae.decoder)
    compiled_unet = torchax.compile(
        unet, torchax.CompileOptions(jax_jit_kwargs={'static_argnames': ('return_dict',)})
    )

    # Run benchmark
    print(f"\n[4/4] Benchmark ({len(TEST_IMAGES)} images x {NUM_ITERS} iters)...")
    print("=" * 70)

    results = []
    for idx, (name, path) in enumerate(TEST_IMAGES):
        pre = precomputed[name]
        im_lr, im_lr_resize, im_lr_resize_norm, resize_h, resize_w = pre['preprocess']
        ori_h, ori_w = im_lr.shape[2], im_lr.shape[3]
        padded_h, padded_w = im_lr_resize_norm.shape[2], im_lr_resize_norm.shape[3]

        print(f"\n--- [{idx+1}/{len(TEST_IMAGES)}] {name}: {ori_w}x{ori_h} ---")

        # Restore de_mod (JittableModule tracks these as dynamic buffers)
        for ln, mod in vae.named_modules():
            if ln in pre['vae_de_mods']:
                mod.de_mod = env.to_xla(pre['vae_de_mods'][ln].to(torch.bfloat16))
        for ln, mod in unet.named_modules():
            if ln in pre['unet_de_mods']:
                mod.de_mod = env.to_xla(pre['unet_de_mods'][ln].to(torch.bfloat16))

        with env:
            input_xla = im_lr_resize_norm.to('jax').to(torch.bfloat16)

        best_total = None
        best_unet = None
        for i in range(NUM_ITERS):
            # Profile the last iteration if PROFILE is set
            use_profile = PROFILE_DIR and (i == NUM_ITERS - 1)
            if use_profile:
                profile_path = os.path.join(PROFILE_DIR, f"trace_{name}")
                os.makedirs(profile_path, exist_ok=True)
                print(f"  [Profile] Tracing iter {i+1} -> {profile_path}")
                jax.profiler.start_trace(profile_path)

            output, times = run_single(vae, compiled_unet, components['scheduler'],
                                        input_xla, pos_enc, neg_enc, mesh, env, sync_stages=SYNC_STAGES)

            if use_profile:
                jax.profiler.stop_trace()
                print(f"  [Profile] Trace saved to {profile_path}")

            label = "compile" if i == 0 else ("warmup" if i == 1 else "steady")
            print(f"  Iter {i+1} ({label}): vae_enc={times['vae_encode']:.3f}s  unet={times['unet']:.3f}s  vae_dec={times['vae_decode']:.3f}s  total={times['total']:.3f}s")
            if i == NUM_ITERS - 1:
                best_total = times['total']
                best_unet = times['unet']

        # Save SR output from last iteration
        sr_img = postprocess_output(output, resize_h, resize_w, im_lr_resize, align_method='wavelet')
        out_dir = Path("test_images/bus_benchmark/sr_outputs")
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"{name}_sr.png"
        sr_img.save(str(out_path))
        print(f"  Saved: {out_path}")

        results.append({
            'name': name,
            'size': f"{ori_w} x {ori_h}",
            'latent': f"{padded_w//8}x{padded_h//8}",
            'total': best_total,
            'unet': best_unet,
        })

    # Print table
    print(f"\n{'='*70}")
    print(f"{'图片名称':<24} {'图片尺寸':<16} {'latent 尺寸':<14} {'最后一轮总耗时(s)':<18} {'最后一轮 Unet 耗时(s)'}")
    print("-" * 90)
    for r in results:
        print(f"{r['name']:<24} {r['size']:<16} {r['latent']:<14} {r['total']:<18.3f} {r['unet']:.3f}")

    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
