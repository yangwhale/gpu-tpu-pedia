#!/usr/bin/env python3
"""
SDXL 三阶段生成 - 阶段2：UNet Denoising (TPU)

加载 stage1 的 prompt embeddings，在 TPU 上运行 denoising loop 生成 latents。
SDXL 使用传统 CFG，需要对 prompt 和 negative prompt 分别进行前向传播。

输入：stage1_embeddings.safetensors
输出：stage2_latents.safetensors
"""

import argparse
import functools
import logging
import math
import os
import sys
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torchax
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import EulerDiscreteScheduler
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from torchax.ops import jaten, ops_registry
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from splash_attention_utils import sdpa_reference

from utils import (
    GUIDANCE_SCALE,
    HEIGHT,
    MODEL_NAME,
    NUM_STEPS,
    UNET_SHARDINGS,
    USE_K_SMOOTH,
    WIDTH,
    get_default_paths,
    load_embeddings_from_safetensors,
    load_generation_config,
    move_module_to_xla,
    save_generation_config,
    save_latents_to_safetensors,
    setup_jax_cache,
    setup_pytree_registrations,
    shard_weight_dict,
)


# ============================================================================
# Torchax 算子覆盖
# ============================================================================

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                  is_causal=False, scale=None, enable_gqa=False,
                                  env=None, mesh=None):
    """SDPA：SDXL 序列较短，使用参考实现即可。"""
    # SDXL 的 attention 序列长度较短（最大 4096 在最低分辨率层），
    # 不需要 Splash Attention
    return sdpa_reference(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)


def torch_conv2d_jax(input, weight, bias=None, stride=1, padding=0,
                     dilation=1, groups=1, *, env):
    """JAX 兼容的 conv2d。"""
    jinput, jweight, jbias = env.t2j_iso((input, weight, bias))
    res = jaten._aten_conv2d(jinput, jweight, jbias, stride, padding, dilation, groups)
    return env.j2t_iso(res)


def override_op(env, op, impl):
    """覆盖 torchax 算子。"""
    env._ops[op] = ops_registry.Operator(
        op, impl, is_jax_function=False, is_user_defined=True,
        needs_env=False, is_view_op=False,
    )


# ============================================================================
# Pipeline 配置
# ============================================================================

def setup_unet(unet, mesh, env):
    """配置 UNet 用于 TPU 推理。"""
    print("\n=== 配置 UNet (TPU) ===")

    override_op(env, torch.nn.functional.conv2d, functools.partial(torch_conv2d_jax, env=env))
    override_op(env, torch.nn.functional.scaled_dot_product_attention,
                functools.partial(scaled_dot_product_attention, env=env, mesh=mesh))

    move_module_to_xla(env, unet)
    unet = torchax.compile(unet, torchax.CompileOptions(
        jax_jit_kwargs={'static_argnames': ('return_dict',)}))
    unet.params = shard_weight_dict(unet.params, UNET_SHARDINGS, mesh)
    unet.buffers = shard_weight_dict(unet.buffers, UNET_SHARDINGS, mesh)
    torchax.interop.call_jax(jax.block_until_ready, unet.params)

    print("✓ UNet 配置完成")
    return unet


def get_add_time_ids(original_size, crops_coords_top_left, target_size, dtype):
    """生成 SDXL 的 add_time_ids。"""
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    return add_time_ids


def run_denoising(
    unet,
    scheduler,
    prompt_embeds,
    pooled_prompt_embeds,
    negative_prompt_embeds,
    negative_pooled_prompt_embeds,
    config,
    mesh,
    env,
    num_steps,
    desc="Denoising",
):
    """运行 denoising loop。"""
    height = config['height']
    width = config['width']
    guidance_scale = config['guidance_scale']
    seed = config['seed']

    # 初始化 latents
    generator = torch.Generator()
    generator.manual_seed(seed)

    # VAE 的 scaling factor
    vae_scale_factor = 8
    latent_height = height // vae_scale_factor
    latent_width = width // vae_scale_factor

    # 初始化随机 latents
    latents_shape = (1, 4, latent_height, latent_width)
    latents = torch.randn(latents_shape, generator=generator, dtype=torch.float32)

    # 设置 scheduler
    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps

    # 初始化 latents 到 scheduler 的初始噪声级别
    latents = latents * scheduler.init_noise_sigma

    # 准备 add_time_ids (SDXL 特有)
    original_size = (height, width)
    target_size = (height, width)
    crops_coords_top_left = (0, 0)
    add_time_ids = get_add_time_ids(original_size, crops_coords_top_left, target_size, prompt_embeds.dtype)
    negative_add_time_ids = add_time_ids  # 使用相同的 time_ids

    # CFG: 拼接 prompt_embeds
    # (2, seq_len, 2048)
    prompt_embeds_cfg = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    # (2, 1280)
    pooled_embeds_cfg = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    # (2, 6)
    add_time_ids_cfg = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

    # 转换到 XLA
    with env:
        latents = latents.to('jax').to(torch.bfloat16)
        prompt_embeds_cfg = prompt_embeds_cfg.to('jax').to(torch.bfloat16)
        pooled_embeds_cfg = pooled_embeds_cfg.to('jax').to(torch.bfloat16)
        add_time_ids_cfg = add_time_ids_cfg.to('jax').to(torch.bfloat16)

    with mesh:
        progress = tqdm(total=num_steps, desc=desc, ncols=100)
        step_start = [time.perf_counter()]

        for i, t in enumerate(timesteps):
            # 扩展 latents 用于 CFG
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # timestep 处理 - t 可能是 tensor 或 float
            if isinstance(t, torch.Tensor):
                t_val = t.item()
            else:
                t_val = float(t)
            # 在 torchax 环境中创建 timestep tensor
            timestep = env.j2t_iso(jnp.array([t_val], dtype=jnp.int32))

            # UNet 前向传播
            added_cond_kwargs = {
                "text_embeds": pooled_embeds_cfg,
                "time_ids": add_time_ids_cfg,
            }

            noise_pred = unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=prompt_embeds_cfg,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # CFG guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Scheduler step
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            jax.effects_barrier()
            step_time = time.perf_counter() - step_start[0]
            progress.set_postfix({'step': f'{step_time:.2f}s'})
            progress.update(1)
            step_start[0] = time.perf_counter()

        progress.close()

    return latents


def main():
    parser = argparse.ArgumentParser(description='SDXL 阶段2：UNet Denoising (TPU)')
    parser.add_argument('--input_dir', type=str, default='./stage_outputs')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--num_inference_steps', type=int, default=NUM_STEPS)
    parser.add_argument('--guidance_scale', type=float, default=GUIDANCE_SCALE)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--height', type=int, default=HEIGHT)
    parser.add_argument('--width', type=int, default=WIDTH)
    parser.add_argument('--warmup_steps', type=int, default=2)
    parser.add_argument('--model_id', type=str, default=None)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    input_paths = get_default_paths(args.input_dir)
    output_paths = get_default_paths(output_dir)

    setup_jax_cache()
    warnings.filterwarnings('ignore')
    logging.getLogger().setLevel(logging.ERROR)
    torch.set_default_dtype(torch.bfloat16)
    setup_pytree_registrations()

    print(f"\n{'='*60}")
    print("SDXL 阶段2：UNet Denoising (TPU)")
    print(f"{'='*60}")

    # 加载配置和 embeddings
    config = load_generation_config(input_paths['config'])
    config.update({
        'num_inference_steps': args.num_inference_steps,
        'guidance_scale': args.guidance_scale,
        'seed': args.seed,
        'height': args.height,
        'width': args.width,
    })
    if args.model_id:
        config['model_id'] = args.model_id

    embeddings, _ = load_embeddings_from_safetensors(input_paths['embeddings'], restore_dtype=True)
    prompt_embeds = embeddings['prompt_embeds']
    pooled_prompt_embeds = embeddings['pooled_prompt_embeds']
    negative_prompt_embeds = embeddings['negative_prompt_embeds']
    negative_pooled_prompt_embeds = embeddings['negative_pooled_prompt_embeds']

    # 加载 UNet
    model_id = config.get('model_id', MODEL_NAME)
    print(f"\n加载模型: {model_id}")

    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=torch.bfloat16
    )
    unet.eval()

    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

    # 启用 torchax
    torchax.enable_globally()
    env = torchax.default_env()

    # 创建 mesh
    tp_dim = len(jax.devices())
    mesh = Mesh(mesh_utils.create_device_mesh((tp_dim,), allow_split_physical_axes=True), ("tp",))
    print(f"\nMesh: tp={tp_dim}")

    unet = setup_unet(unet, mesh, env)

    # 预热
    if args.warmup_steps > 0:
        print(f"\n预热 ({args.warmup_steps} 步)...")
        run_denoising(
            unet, scheduler,
            prompt_embeds, pooled_prompt_embeds,
            negative_prompt_embeds, negative_pooled_prompt_embeds,
            config, mesh, env, args.warmup_steps, "Warmup"
        )

    # 推理
    print(f"\n推理 ({args.num_inference_steps} 步)...")
    start = time.perf_counter()
    latents = run_denoising(
        unet, scheduler,
        prompt_embeds, pooled_prompt_embeds,
        negative_prompt_embeds, negative_pooled_prompt_embeds,
        config, mesh, env, args.num_inference_steps, "Denoising"
    )
    elapsed = time.perf_counter() - start

    print(f"\n✓ 完成: {elapsed:.2f}s ({elapsed/args.num_inference_steps:.2f}s/step)")

    # 转换并保存 latents
    if hasattr(latents, '_elem'):
        jax_latents = latents._elem
        if jax_latents.dtype == jnp.bfloat16:
            torch_latents = torch.from_numpy(np.array(jax_latents.astype(jnp.float32))).to(torch.bfloat16)
        else:
            torch_latents = torch.from_numpy(np.array(jax_latents))
    else:
        torch_latents = latents.cpu()

    save_latents_to_safetensors(torch_latents, output_paths['latents'], {
        'num_inference_steps': str(config['num_inference_steps']),
        'seed': str(config['seed']),
    })
    save_generation_config(config, output_paths['config'])

    print(f"\n{'='*60}")
    print("✓ 阶段2 完成！下一步：运行 stage3_vae_decoder.py")
    print(f"{'='*60}")

    os._exit(0)


if __name__ == "__main__":
    main()
