#!/usr/bin/env python3
"""
Flux.1 Text-to-Image (TPU via Torchax + Splash Attention)

Uses torchax + JAX to run Flux.1 on TPU.
Text encoding (CLIP + T5) runs on CPU, diffusion runs on TPU.

Usage:
  python generate_torchax.py --prompt "a cat" --num_inference_steps 28
  python generate_torchax.py --num_inference_steps 3 --warmup  # benchmark mode
"""

import os
import warnings
import logging

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')
for logger_name in ['root', '', 'diffusers', 'transformers']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

import argparse
import functools
import gc
import math
import re
import time
from contextlib import nullcontext
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torchax
from diffusers import AutoencoderKL, FluxPipeline, FluxTransformer2DModel
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from torchax.ops import jaten, ops_registry

from splash_attention_utils import sdpa_reference, tpu_splash_attention


# ============================================================================
# Config
# ============================================================================

MODEL_ID = "Freepik/flux.1-lite-8B"
WIDTH, HEIGHT = 1024, 1024
NUM_STEPS = 28
GUIDANCE_SCALE = 3.5
SEED = 11
USE_K_SMOOTH = True
PROFILE_OUT_PATH = "/dev/shm/jax_trace"

DEFAULT_PROMPT = (
    "A close-up image of a green alien with fluorescent skin in the middle "
    "of a dark forest, illuminated by bioluminescent plants"
)


# ============================================================================
# Transformer Sharding (1D mesh: tp)
# ============================================================================

TRANSFORMER_SHARDINGS = {
    # Double-stream Blocks - Attention
    r'transformer_blocks.*.attn.to_q.weight': ('tp', None),
    r'transformer_blocks.*.attn.to_k.weight': ('tp', None),
    r'transformer_blocks.*.attn.to_v.weight': ('tp', None),
    r'transformer_blocks.*.attn.to_out.0.weight': (None, 'tp'),
    r'transformer_blocks.*.attn.add_q_proj.weight': ('tp', None),
    r'transformer_blocks.*.attn.add_k_proj.weight': ('tp', None),
    r'transformer_blocks.*.attn.add_v_proj.weight': ('tp', None),
    r'transformer_blocks.*.attn.to_add_out.weight': (None, 'tp'),
    # Double-stream Blocks - FeedForward (GEGLU)
    r'transformer_blocks.*.ff.net.0.proj.weight': ('tp', None),
    r'transformer_blocks.*.ff.net.2.weight': (None, 'tp'),
    r'transformer_blocks.*.ff_context.net.0.proj.weight': ('tp', None),
    r'transformer_blocks.*.ff_context.net.2.weight': (None, 'tp'),
    # Single-stream Blocks
    r'single_transformer_blocks.*.attn.to_q.weight': ('tp', None),
    r'single_transformer_blocks.*.attn.to_k.weight': ('tp', None),
    r'single_transformer_blocks.*.attn.to_v.weight': ('tp', None),
    r'single_transformer_blocks.*.proj_mlp.weight': ('tp', None),
    r'single_transformer_blocks.*.proj_out.weight': (None, 'tp'),
    # Modulation
    r'transformer_blocks.*.norm1.linear.weight': ('tp', None),
    r'transformer_blocks.*.norm1_context.linear.weight': ('tp', None),
    r'single_transformer_blocks.*.norm.linear.weight': ('tp', None),
    # Embedders & Projections
    r'x_embedder.weight': ('tp', None),
    r'context_embedder.weight': ('tp', None),
    r'proj_out.weight': (None, 'tp'),
    # Time + Text + Guidance Embedding
    r'time_text_embed.timestep_embedder.linear_1.weight': ('tp', None),
    r'time_text_embed.timestep_embedder.linear_2.weight': (None, 'tp'),
    r'time_text_embed.guidance_embedder.linear_1.weight': ('tp', None),
    r'time_text_embed.guidance_embedder.linear_2.weight': (None, 'tp'),
    r'time_text_embed.text_embedder.linear_1.weight': ('tp', None),
    r'time_text_embed.text_embedder.linear_2.weight': (None, 'tp'),
}


# ============================================================================
# Helpers
# ============================================================================

def shard_weight_dict(weight_dict, sharding_dict, mesh):
    result = {}
    for k, v in weight_dict.items():
        matched = False
        for pattern, sharding in sharding_dict.items():
            if re.fullmatch(pattern, k) is not None:
                v.apply_jax_(jax.device_put, NamedSharding(mesh, P(*sharding)))
                matched = True
                break
        if not matched:
            v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
        result[k] = v
    return result


def override_op(env, op, impl):
    env._ops[op] = ops_registry.Operator(
        op, impl, is_jax_function=False, is_user_defined=True,
        needs_env=False, is_view_op=False,
    )


def move_to_xla(env, module):
    with jax.default_device("cpu"):
        state_dict = module.state_dict()
        state_dict = env.to_xla(state_dict)
        module.load_state_dict(state_dict, assign=True)


def torch_conv2d_jax(input, weight, bias=None, stride=1, padding=0,
                     dilation=1, groups=1, *, env):
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


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                  is_causal=False, scale=None, enable_gqa=False,
                                  env=None, mesh=None):
    if key.shape[2] > 20000:
        assert attn_mask is None and dropout_p == 0.0 and not is_causal
        assert not enable_gqa and scale is None
        jquery, jkey, jvalue = env.t2j_iso((query, key, value))
        if USE_K_SMOOTH:
            jkey = jkey - jnp.mean(jkey, axis=2, keepdims=True)
        res = tpu_splash_attention(jquery, jkey, jvalue, mesh, scale=scale)
        return env.j2t_iso(res)
    return sdpa_reference(query, key, value, attn_mask, dropout_p, is_causal,
                           scale, enable_gqa)


# ============================================================================
# Latent helpers (from FluxPipeline)
# ============================================================================

def prepare_latent_image_ids(height, width, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
    return latent_image_ids.reshape(height * width, 3).to(dtype=dtype)  # 2D, no batch


def pack_latents(latents, batch_size, num_channels, height, width):
    latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)


def unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))
    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(batch_size, channels // 4, height, width)


def calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096,
                    base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


# ============================================================================
# Text Encoding (CPU)
# ============================================================================

def encode_prompt_cpu(model_id, prompt, max_seq_len=512):
    """Encode prompt using CLIP + T5 on CPU, return embeddings."""
    from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

    print("\n=== Text Encoding (CPU) ===")
    t0 = time.perf_counter()

    # CLIP (pooled_prompt_embeds)
    print("  Loading CLIP...")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    text_encoder.eval()

    text_input = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        clip_output = text_encoder(text_input.input_ids, output_hidden_states=False)
    pooled_prompt_embeds = clip_output.pooler_output.to(torch.bfloat16)
    print(f"  CLIP pooled: {pooled_prompt_embeds.shape}")

    del text_encoder, tokenizer
    gc.collect()

    # T5 (prompt_embeds)
    print("  Loading T5...")
    tokenizer_2 = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer_2")
    text_encoder_2 = T5EncoderModel.from_pretrained(
        model_id, subfolder="text_encoder_2", torch_dtype=torch.bfloat16
    )
    text_encoder_2.eval()

    text_input_2 = tokenizer_2(
        prompt, padding="max_length", max_length=max_seq_len,
        truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        t5_output = text_encoder_2(text_input_2.input_ids)
    prompt_embeds = t5_output[0].to(torch.bfloat16)
    print(f"  T5 embeds: {prompt_embeds.shape}")

    del text_encoder_2, tokenizer_2
    gc.collect()

    t_encode = time.perf_counter() - t0
    print(f"  Prompt编码: {t_encode:.2f}s")

    return prompt_embeds, pooled_prompt_embeds


# ============================================================================
# Inference
# ============================================================================

def run_inference(transformer, vae, scheduler, prompt_embeds, pooled_prompt_embeds,
                  height, width, num_steps, guidance_scale, seed, mesh, env,
                  warmup=False, profile=False):
    """Run Flux.1 denoising + VAE decode on TPU."""

    vae_scale_factor = 8  # standard for Flux VAE
    latent_h = 2 * (height // (vae_scale_factor * 2))
    latent_w = 2 * (width // (vae_scale_factor * 2))
    num_channels = transformer.config.in_channels // 4  # 16

    # Prepare latents on CPU, then move to XLA
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latents = torch.randn(1, num_channels, latent_h, latent_w,
                          generator=generator, dtype=torch.bfloat16)
    latents = pack_latents(latents, 1, num_channels, latent_h, latent_w)
    latent_image_ids = prepare_latent_image_ids(latent_h // 2, latent_w // 2, torch.bfloat16)

    # Text IDs
    text_ids = torch.zeros(prompt_embeds.shape[1], 3, dtype=torch.bfloat16)

    # Guidance
    guidance = torch.full([1], guidance_scale, dtype=torch.float32)

    # Move to XLA
    latents = env.to_xla(latents)
    latent_image_ids = env.to_xla(latent_image_ids)
    text_ids = env.to_xla(text_ids)
    prompt_embeds_xla = env.to_xla(prompt_embeds)
    pooled_prompt_embeds_xla = env.to_xla(pooled_prompt_embeds)
    guidance_xla = env.to_xla(guidance)

    # Scheduler setup
    image_seq_len = latents.shape[1]
    mu = calculate_shift(image_seq_len)
    sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
    scheduler.set_timesteps(sigmas=sigmas, device="cpu", mu=mu)
    timesteps = scheduler.timesteps

    # Tracking
    transformer_times = []
    vae_times = []

    print(f"\n{'='*60}")
    print(f"推理: steps={num_steps}, guidance={guidance_scale}, seed={seed}, {width}x{height}")
    print(f"{'='*60}")

    def run_loop(steps_to_run, label, track=False):
        nonlocal latents
        # Reset latents and scheduler for each run
        generator_reset = torch.Generator(device="cpu").manual_seed(seed)
        latents_reset = torch.randn(1, num_channels, latent_h, latent_w,
                                    generator=generator_reset, dtype=torch.bfloat16)
        latents_reset = pack_latents(latents_reset, 1, num_channels, latent_h, latent_w)
        latents = env.to_xla(latents_reset)

        sigmas_run = np.linspace(1.0, 1 / steps_to_run, steps_to_run)
        scheduler.set_timesteps(sigmas=sigmas_run, device="cpu", mu=mu)
        ts = scheduler.timesteps

        print(f"\n{label}...")
        from tqdm import tqdm
        t_start = time.perf_counter()

        for i, t in enumerate(tqdm(range(steps_to_run), total=steps_to_run)):
            timestep_val = ts[i]
            timestep = torch.tensor([timestep_val], dtype=torch.bfloat16)
            timestep = env.to_xla(timestep)

            t_trans = time.perf_counter()
            noise_pred = transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance_xla,
                pooled_projections=pooled_prompt_embeds_xla,
                encoder_hidden_states=prompt_embeds_xla,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]
            if hasattr(noise_pred, '_elem'):
                noise_pred._elem.block_until_ready()
            t_trans_end = time.perf_counter()

            if track:
                transformer_times.append((t_trans_end - t_trans) * 1000)

            latents = scheduler.step(noise_pred, timestep_val, latents, return_dict=False)[0]

        t_loop = time.perf_counter() - t_start
        print(f"✓ {label}耗时: {t_loop:.2f}s")
        return t_loop

    # Warmup (transformer + VAE)
    if warmup:
        warmup_steps = 2
        run_loop(warmup_steps, f"预热（触发 JIT 编译，{warmup_steps} 步）")
        # VAE warmup
        print("\n  VAE 预热...")
        t_vw = time.perf_counter()
        warmup_latents = unpack_latents(latents, height, width, vae_scale_factor)
        warmup_for_vae = (warmup_latents / vae.config.scaling_factor) + vae.config.shift_factor
        warmup_img = vae.decode(warmup_for_vae, return_dict=False)[0]
        if hasattr(warmup_img, '_elem'):
            warmup_img._elem.block_until_ready()
        print(f"  VAE 预热: {(time.perf_counter()-t_vw)*1000:.0f}ms")
        del warmup_img, warmup_latents, warmup_for_vae

    # Benchmark
    profiler_ctx = (jax.profiler.trace(PROFILE_OUT_PATH, create_perfetto_link=False)
                    if profile else nullcontext())
    with profiler_ctx:
        infer_time = run_loop(num_steps, f"正式推理", track=True)

    # VAE decode
    print(f"\n--- VAE Decode ---")
    latents_unpacked = unpack_latents(latents, height, width, vae_scale_factor)
    latents_for_vae = (latents_unpacked / vae.config.scaling_factor) + vae.config.shift_factor

    t_vae = time.perf_counter()
    image_tensor = vae.decode(latents_for_vae, return_dict=False)[0]
    # Normalize on XLA
    image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
    image_uint8 = (image_tensor * 255.0).to(torch.uint8)
    if hasattr(image_uint8, '_elem'):
        image_uint8._elem.block_until_ready()
    t_vae_end = time.perf_counter()
    vae_time = (t_vae_end - t_vae) * 1000
    vae_times.append(vae_time)
    print(f"  VAE Decoder: {vae_time:.2f}ms")

    # D2H
    if hasattr(image_uint8, '_elem'):
        image_np = np.array(image_uint8._elem)
    else:
        image_np = image_uint8.cpu().numpy()
    image_np = np.transpose(image_np[0], (1, 2, 0))
    from PIL import Image
    image = Image.fromarray(image_np)

    # Save
    file_name = f"output_tpu.png"
    image.save(file_name)
    print(f"\n图像已保存: {file_name}")

    # Stats
    print(f"\n{'='*60}")
    print("耗时统计")
    print(f"{'='*60}")
    if warmup:
        print(f"  预热({2}步):      已完成")
    print(f"  推理({num_steps}步):     {infer_time:.2f}s")
    print(f"{'='*60}")

    print(f"\n正式推理（{num_steps}步）各模块详细耗时")
    print(f"{'='*60}")
    print(f"Transformer: 共调用 {len(transformer_times)} 次, 总耗时 {sum(transformer_times):.2f}ms ({sum(transformer_times)/1000:.3f}s)")
    for idx, tt in enumerate(transformer_times):
        print(f"  - 第 {idx+1} 次: {tt:.2f}ms")
    if transformer_times:
        print(f"  - 平均: {sum(transformer_times)/len(transformer_times):.2f}ms/次")

    print(f"VAE Encoder: 未被调用")
    print(f"VAE Decoder: 共调用 {len(vae_times)} 次, 总耗时 {sum(vae_times):.2f}ms ({sum(vae_times)/1000:.3f}s)")

    tracked_total = sum(transformer_times) + sum(vae_times)
    other_ms = infer_time * 1000 - sum(transformer_times)
    print(f"\n  已追踪模块 TPU 耗时合计: {tracked_total:.2f}ms ({tracked_total/1000:.3f}s)")
    print(f"  其他开销（scheduler, 数据搬运等）: {other_ms:.2f}ms ({other_ms/1000:.3f}s)")
    wall_ms = infer_time * 1000 + sum(vae_times)
    print(f"  推理总耗时 (wall clock): {wall_ms:.2f}ms ({wall_ms/1000:.3f}s)")
    print(f"{'='*60}")

    return image


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Flux.1 TPU 图像生成")
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--num_inference_steps", type=int, default=NUM_STEPS)
    parser.add_argument("--guidance_scale", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--warmup", action="store_true", help="先预热再跑基准")
    parser.add_argument("--profile", action="store_true", help="启用 JAX profiler")
    parser.add_argument("--max_sequence_length", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print("Flux.1 Text-to-Image（TPU Splash Attention）")
    print(f"{'='*60}")
    print(f"  模型: {args.model_id}")
    print(f"  分辨率: {args.width}x{args.height}")
    print(f"  步数: {args.num_inference_steps}, 引导: {args.guidance_scale}, 种子: {args.seed}")

    # JAX config
    jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax_cache"))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    torch.set_default_dtype(torch.bfloat16)

    # 1. Text encoding on CPU
    t_total_start = time.perf_counter()
    prompt_embeds, pooled_prompt_embeds = encode_prompt_cpu(
        args.model_id, args.prompt, args.max_sequence_length
    )

    # 2. Load transformer + VAE on CPU
    print("\n=== 加载模型 ===")
    t_load = time.perf_counter()
    transformer = FluxTransformer2DModel.from_pretrained(
        args.model_id, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    vae = AutoencoderKL.from_pretrained(
        args.model_id, subfolder="vae", torch_dtype=torch.bfloat16
    )
    # VAE decoder stays float32 to avoid NaN (bf16 precision insufficient for decoder conv chain)
    for p in vae.decoder.parameters():
        p.data = p.data.to(torch.float32)
    for b in vae.decoder.buffers():
        b.data = b.data.to(torch.float32)
    scheduler = FlowMatchEulerDiscreteScheduler()
    print(f"  模型加载: {time.perf_counter() - t_load:.2f}s")

    # 3. Enable torchax
    print("\n=== TPU准备 ===")
    t_tpu = time.perf_counter()
    torchax.enable_globally()
    env = torchax.default_env()

    tp_dim = len(jax.devices())
    mesh_devices = mesh_utils.create_device_mesh((tp_dim,), allow_split_physical_axes=True)
    mesh = Mesh(mesh_devices, ("tp",))
    print(f"  Mesh: tp={tp_dim}, 设备数={len(jax.devices())}")

    # Register custom ops
    override_op(env, torch.nn.functional.conv2d,
                functools.partial(torch_conv2d_jax, env=env))
    override_op(env, torch.nn.functional.scaled_dot_product_attention,
                functools.partial(scaled_dot_product_attention, env=env, mesh=mesh))

    # Move to XLA
    print("  Moving Transformer to XLA...")
    move_to_xla(env, transformer)
    print("  Moving VAE to XLA...")
    move_to_xla(env, vae)

    # Compile
    with mesh:
        transformer = torchax.compile(transformer, torchax.CompileOptions(
            jax_jit_kwargs={'static_argnames': ('return_dict',)}))

        # Shard transformer weights
        transformer.params = shard_weight_dict(transformer.params, TRANSFORMER_SHARDINGS, mesh)
        transformer.buffers = shard_weight_dict(transformer.buffers, TRANSFORMER_SHARDINGS, mesh)

        vae.decoder = torchax.compile(vae.decoder)

    t_tpu_done = time.perf_counter()
    print(f"  TPU准备: {t_tpu_done - t_tpu:.2f}s")

    # 4. Run inference
    with mesh:
        image = run_inference(
            transformer, vae, scheduler,
            prompt_embeds, pooled_prompt_embeds,
            args.height, args.width,
            args.num_inference_steps, args.guidance_scale, args.seed,
            mesh, env,
            warmup=args.warmup, profile=args.profile,
        )

    t_total = time.perf_counter() - t_total_start
    print(f"\n总耗时: {t_total:.2f}s")
    print(f"\n{'='*60}")
    print("✓ 生成完成！")
    print(f"{'='*60}")
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
