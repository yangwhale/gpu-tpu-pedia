#!/usr/bin/env python3
"""
Flux.2 三阶段生成 - 阶段2：Transformer (TPU)

加载 stage1 的 prompt embeddings，在 TPU 上运行 denoising loop 生成 latents。
Flux.2 使用 Embedded CFG，只需 1D mesh (tp)。

输入：stage1_embeddings.safetensors
输出：stage2_latents.safetensors
"""

import argparse
import functools
import logging
import os
import sys
import time
import warnings
from contextlib import nullcontext
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torchax
from diffusers.models.autoencoders.autoencoder_kl_flux2_torchax import AutoencoderKLFlux2
from diffusers.models.transformers.transformer_flux2_torchax import Flux2Transformer2DModel
from diffusers.pipelines.flux2.pipeline_flux2_torchax import Flux2Pipeline
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from torchax.ops import jaten, ops_registry
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from splash_attention_utils import sdpa_reference, tpu_splash_attention

from utils import (
    GUIDANCE_SCALE,
    HEIGHT,
    MODEL_NAME,
    NUM_STEPS,
    TRANSFORMER_SHARDINGS,
    USE_K_SMOOTH,
    WIDTH,
    get_default_paths,
    load_embeddings_from_safetensors,
    load_generation_config,
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
    """SDPA：长序列用 Splash Attention，短序列用参考实现。"""
    if key.shape[2] > 20000:
        jquery, jkey, jvalue = env.t2j_iso((query, key, value))
        if USE_K_SMOOTH:
            jkey = jkey - jnp.mean(jkey, axis=2, keepdims=True)
        res = tpu_splash_attention(jquery, jkey, jvalue, mesh, scale=scale)
        return env.j2t_iso(res)
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


def move_module_to_xla(env, module):
    """将模块移动到 XLA。"""
    with jax.default_device("cpu"):
        state_dict = env.to_xla(module.state_dict())
        module.load_state_dict(state_dict, assign=True)


# ============================================================================
# Pipeline 配置
# ============================================================================

def setup_pipeline(pipe, mesh, env):
    """配置 Pipeline 用于 Transformer 推理。"""
    print("\n=== 配置 Transformer (TPU) ===")
    
    override_op(env, torch.nn.functional.conv2d, functools.partial(torch_conv2d_jax, env=env))
    override_op(env, torch.nn.functional.scaled_dot_product_attention,
                functools.partial(scaled_dot_product_attention, env=env, mesh=mesh))
    
    move_module_to_xla(env, pipe.transformer)
    pipe.transformer = torchax.compile(pipe.transformer, torchax.CompileOptions(
        jax_jit_kwargs={'static_argnames': ('return_dict',)}))
    pipe.transformer.params = shard_weight_dict(pipe.transformer.params, TRANSFORMER_SHARDINGS, mesh)
    pipe.transformer.buffers = shard_weight_dict(pipe.transformer.buffers, TRANSFORMER_SHARDINGS, mesh)
    torchax.interop.call_jax(jax.block_until_ready, pipe.transformer.params)
    
    # 删除 VAE 节省内存
    if hasattr(pipe, 'vae') and pipe.vae is not None:
        del pipe.vae
        pipe.vae = None
    
    print("✓ Transformer 配置完成")
    return pipe


def run_denoising(pipe, prompt_embeds, config, mesh, num_steps, desc="Denoising"):
    """运行 denoising loop。"""
    generator = torch.Generator()
    generator.manual_seed(config['seed'])
    
    with mesh:
        progress = tqdm(total=num_steps, desc=desc, ncols=100)
        step_start = [time.perf_counter()]
        
        def callback(pipe, step, timestep, kwargs):
            jax.effects_barrier()
            step_time = time.perf_counter() - step_start[0]
            progress.set_postfix({'step': f'{step_time:.2f}s'})
            progress.update(1)
            step_start[0] = time.perf_counter()
            return kwargs
        
        result = pipe(
            prompt=None,
            prompt_embeds=prompt_embeds,
            height=config['height'],
            width=config['width'],
            num_inference_steps=num_steps,
            guidance_scale=config['guidance_scale'],
            generator=generator,
            output_type='latent',
            callback_on_step_end=callback,
        )
        jax.effects_barrier()
        progress.update(1)
        progress.close()
    
    return result.images


def main():
    parser = argparse.ArgumentParser(description='Flux.2 阶段2：Transformer (TPU)')
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
    print("Flux.2 阶段2：Transformer (TPU)")
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
    
    prompt_embeds_dict, _ = load_embeddings_from_safetensors(input_paths['embeddings'], restore_dtype=True)
    prompt_embeds = prompt_embeds_dict['prompt_embeds']
    
    # 加载 Pipeline
    model_id = config.get('model_id', MODEL_NAME)
    print(f"\n加载模型: {model_id}")
    
    pipe = Flux2Pipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        text_encoder=None,
        vae=AutoencoderKLFlux2.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.bfloat16),
        transformer=Flux2Transformer2DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16),
        scheduler=FlowMatchEulerDiscreteScheduler(),
    )
    
    # 启用 torchax
    torchax.enable_globally()
    env = torchax.default_env()
    
    # 创建 mesh
    tp_dim = len(jax.devices())
    mesh = Mesh(mesh_utils.create_device_mesh((tp_dim,), allow_split_physical_axes=True), ("tp",))
    print(f"\nMesh: tp={tp_dim}")
    
    pipe = setup_pipeline(pipe, mesh, env)
    
    # 转换 embeddings 到 XLA
    with env:
        prompt_embeds = prompt_embeds.to('jax')
    
    # 预热
    if args.warmup_steps > 0:
        print(f"\n预热 ({args.warmup_steps} 步)...")
        run_denoising(pipe, prompt_embeds, config, mesh, args.warmup_steps, "Warmup")
    
    # 推理
    print(f"\n推理 ({args.num_inference_steps} 步)...")
    start = time.perf_counter()
    latents = run_denoising(pipe, prompt_embeds, config, mesh, args.num_inference_steps, "Denoising")
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
