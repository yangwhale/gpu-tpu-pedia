#!/usr/bin/env python3
"""
Wan 2.2 I2V 三阶段生成 - 阶段2：Transformer (Denoising)

本阶段负责：
1. 加载阶段1生成的 embeddings
2. 设置 JAX/TPU 环境和 Splash Attention
3. 加载双 Transformer 模型并进行权重分片
4. 运行 denoising loop（双模型切换）
5. 将 latents 保存为 SafeTensors 格式

输入文件：
- stage1_embeddings.safetensors: prompt embeddings 和 condition
- generation_config.json: 生成配置

输出文件：
- stage2_latents.safetensors: 生成的 latents

注意：本实现仅支持 A14B 模式 (expand_timesteps=False)
"""

import os
import sys
import time
import functools
import argparse
import warnings
import logging
import numpy as np
from contextlib import nullcontext
from pathlib import Path
from tqdm import tqdm

import jax
import jax.numpy as jnp
import torch
import torchax
from torchax.ops import ops_registry, jaten
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
from jax.experimental import mesh_utils

# 导入共用的 Splash Attention 模块
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'kernels'))
from splash_attention_utils import tpu_splash_attention, sdpa_reference

from diffusers.models.transformers.transformer_wan_torchax import WanTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from utils import (
    MODEL_ID,
    NUM_STEPS,
    GUIDANCE_SCALE,
    BOUNDARY_RATIO,
    SHIFT,
    BQSIZE,
    BKVSIZE,
    BKVCOMPUTESIZE,
    USE_K_SMOOTH,
    DEFAULT_DP,
    TRANSFORMER_SHARDINGS,
    shard_weight_dict,
    setup_pytree_registrations,
    setup_jax_cache,
    load_embeddings_from_safetensors,
    save_latents_to_safetensors,
    save_generation_config,
    load_generation_config,
    get_default_paths,
)


# ============================================================================
# Scaled Dot-Product Attention (使用共用模块)
# ============================================================================

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                  is_causal=False, scale=None, enable_gqa=False,
                                  env=None, mesh=None):
    """封装 SDPA，长序列使用 TPU Splash Attention。"""
    # 仅对长序列（self-attention）使用 TPU Splash Attention
    if key.shape[2] > 20000:
        assert attn_mask is None
        assert dropout_p == 0.0
        assert is_causal is False
        assert enable_gqa is False
        assert scale is None
        
        jquery, jkey, jvalue = env.t2j_iso((query, key, value))
        
        if USE_K_SMOOTH:
            key_mean = jnp.mean(jkey, axis=2, keepdims=True)
            jkey = jkey - key_mean
        
        res = tpu_splash_attention(jquery, jkey, jvalue, mesh, scale=scale)
        return env.j2t_iso(res)

    return sdpa_reference(query, key, value, attn_mask, dropout_p,
                          is_causal, scale, enable_gqa)


# ============================================================================
# Pipeline 设置
# ============================================================================

def torch_conv2d_jax(input, weight, bias=None, stride=1, padding=0,
                     dilation=1, groups=1, *, env):
    """JAX-compatible conv2d override."""
    jinput, jweight, jbias = env.t2j_iso((input, weight, bias))
    res = jaten._aten_conv2d(jinput, jweight, jbias, stride, padding,
                             dilation, groups)
    return env.j2t_iso(res)


def override_op_definition(env, op_to_override, op_impl):
    """Override operator definition in torchax environment."""
    env._ops[op_to_override] = ops_registry.Operator(
        op_to_override,
        op_impl,
        is_jax_function=False,
        is_user_defined=True,
        needs_env=False,
        is_view_op=False,
    )


def move_module_to_xla(env, module):
    """Move module weights to XLA devices."""
    with jax.default_device("cpu"):
        state_dict = module.state_dict()
        state_dict = env.to_xla(state_dict)
        module.load_state_dict(state_dict, assign=True)


def setup_transformers_for_jax(transformer, transformer_2, mesh, env):
    """Setup dual Transformers for JAX/TPU execution."""
    print("\n=== 配置 Transformer 模型 ===")
    
    # Register custom operators
    print("- 注册自定义 JAX 算子...")
    override_op_definition(
        env,
        torch.nn.functional.conv2d,
        functools.partial(torch_conv2d_jax, env=env)
    )
    override_op_definition(
        env,
        torch.nn.functional.scaled_dot_product_attention,
        functools.partial(scaled_dot_product_attention, env=env, mesh=mesh),
    )
    
    transformer_options = torchax.CompileOptions(
        jax_jit_kwargs={"static_argnames": ("return_dict",)}
    )
    
    # Setup Transformer 1
    print("- 配置 Transformer 1 (高噪声阶段)...")
    move_module_to_xla(env, transformer)
    transformer = torchax.compile(transformer, transformer_options)
    transformer.params = shard_weight_dict(
        transformer.params, TRANSFORMER_SHARDINGS, mesh
    )
    transformer.buffers = shard_weight_dict(
        transformer.buffers, TRANSFORMER_SHARDINGS, mesh
    )
    
    # Setup Transformer 2
    print("- 配置 Transformer 2 (低噪声阶段)...")
    move_module_to_xla(env, transformer_2)
    transformer_2 = torchax.compile(transformer_2, transformer_options)
    transformer_2.params = shard_weight_dict(
        transformer_2.params, TRANSFORMER_SHARDINGS, mesh
    )
    transformer_2.buffers = shard_weight_dict(
        transformer_2.buffers, TRANSFORMER_SHARDINGS, mesh
    )
    
    # Wait for sharding to complete
    torchax.interop.call_jax(jax.block_until_ready, transformer.params)
    torchax.interop.call_jax(jax.block_until_ready, transformer_2.params)
    
    print("✓ Transformer 配置完成")
    return transformer, transformer_2


# ============================================================================
# Denoising Loop
# ============================================================================

def run_denoising_loop(
    transformer,
    transformer_2,
    scheduler,
    prompt_embeds,
    negative_prompt_embeds,
    condition,
    config,
    mesh,
    num_steps,
    desc="Denoising",
    is_warmup=False,
):
    """
    运行 denoising loop（A14B 模式）
    
    A14B 模式特点：
    - latent_model_input = concat([latents, condition], dim=1)
    - 使用统一 timestep（标量）
    - 双模型切换（boundary_ratio=0.9）
    """
    step_times = []
    start_time = time.perf_counter()
    
    # 初始化噪声 latents
    batch_size = 1
    num_latent_frames = config['num_latent_frames']
    latent_height = config['latent_height']
    latent_width = config['latent_width']
    
    # latents shape: [B, 16, T', H', W']
    latent_shape = (batch_size, 16, num_latent_frames, latent_height, latent_width)
    
    generator = torch.Generator()
    generator.manual_seed(config['seed'])
    
    latents = torch.randn(latent_shape, generator=generator, dtype=torch.bfloat16)
    latents = latents.to('jax')
    
    # 设置 shift（Wan 2.2 I2V 模型默认 shift=5.0）
    # shift 参数调整采样的时间步长分布，较高的 shift 值会将更多步数分配给低噪声阶段
    shift_value = config.get('shift', SHIFT)
    scheduler.set_shift(shift_value)
    
    # 设置 timesteps
    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps
    
    # 计算 boundary timestep
    boundary_timestep = BOUNDARY_RATIO * scheduler.config.num_train_timesteps
    
    # tqdm 进度条
    progress_bar = tqdm(
        total=num_steps,
        desc=desc,
        ncols=130,
    )
    
    with mesh:
        for i, t in enumerate(timesteps):
            step_start_time = time.perf_counter()
            
            # 选择模型
            if t >= boundary_timestep:
                current_model = transformer
            else:
                current_model = transformer_2
            
            # 构建 latent_model_input (A14B 模式)
            # [B, 16, T', H', W'] + [B, 20, T', H', W'] = [B, 36, T', H', W']
            latent_model_input = torch.cat([latents, condition], dim=1)
            
            # CFG: 复制输入
            batch_input = torch.cat([latent_model_input, latent_model_input])
            batch_embeds = torch.cat([prompt_embeds, negative_prompt_embeds])
            
            # Timestep
            timestep = t.expand(batch_size * 2).to('jax')
            
            # Transformer forward
            noise = current_model(
                hidden_states=batch_input,
                timestep=timestep,
                encoder_hidden_states=batch_embeds,
                return_dict=False
            )[0]
            
            # CFG 计算
            noise_pred = noise[0:1]
            noise_uncond = noise[1:2]
            noise_pred = noise_uncond + config['guidance_scale'] * (noise_pred - noise_uncond)
            
            # Scheduler step
            latents = scheduler.step(noise_pred, t, latents).prev_sample
            
            # 等待计算完成
            jax.effects_barrier()
            
            step_time = time.perf_counter() - step_start_time
            step_times.append(step_time)
            
            if is_warmup:
                progress_bar.set_postfix({
                    'step': f'{step_time:.2f}s',
                    'model': 'T1' if t >= boundary_timestep else 'T2',
                })
            else:
                avg_time = sum(step_times) / len(step_times)
                remaining_steps = num_steps - i - 1
                progress_bar.set_postfix({
                    'step': f'{step_time:.2f}s',
                    'avg': f'{avg_time:.2f}s',
                    'eta': f'{avg_time * remaining_steps:.1f}s',
                    'model': 'T1' if t >= boundary_timestep else 'T2',
                })
            
            progress_bar.update(1)
    
    progress_bar.close()
    
    elapsed = time.perf_counter() - start_time
    return latents, step_times, elapsed


def run_transformer_inference(
    transformer,
    transformer_2,
    scheduler,
    embeddings_dict,
    config,
    mesh,
    warmup_steps=0,
):
    """
    运行 Transformer 推理生成 latents
    """
    print(f"\n=== 阶段2：Transformer 推理 ===")
    print(f"推理步数: {config['num_inference_steps']}")
    print(f"帧数: {config['num_frames']}")
    print(f"引导尺度: {config['guidance_scale']}")
    print(f"随机种子: {config['seed']}")
    print(f"分辨率: {config['height']}x{config['width']}")
    print(f"Boundary ratio: {BOUNDARY_RATIO}")
    if warmup_steps > 0:
        print(f"预热步数: {warmup_steps}")
    
    # 获取 embeddings
    prompt_embeds = embeddings_dict['prompt_embeds']
    negative_prompt_embeds = embeddings_dict['negative_prompt_embeds']
    condition = embeddings_dict['condition']
    
    # === Warmup (可选) ===
    if warmup_steps > 0:
        print(f"\n预热中（{warmup_steps}步，触发 JIT 编译）...")
        
        _, warmup_times, warmup_elapsed = run_denoising_loop(
            transformer=transformer,
            transformer_2=transformer_2,
            scheduler=scheduler,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            condition=condition,
            config=config,
            mesh=mesh,
            num_steps=warmup_steps,
            desc="Warmup (JIT)",
            is_warmup=True,
        )
        
        print(f"  ✓ 预热完成，耗时: {warmup_elapsed:.2f}秒")
    
    # === 正式推理 ===
    print("\n开始 Transformer 推理...")
    
    latents, step_times, elapsed = run_denoising_loop(
        transformer=transformer,
        transformer_2=transformer_2,
        scheduler=scheduler,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        condition=condition,
        config=config,
        mesh=mesh,
        num_steps=config['num_inference_steps'],
        desc="Denoising (TPU)",
        is_warmup=False,
    )
    
    print(f"\n✓ Transformer 推理完成，耗时: {elapsed:.2f} 秒")
    
    # 打印性能统计
    if len(step_times) > 1:
        avg_time = sum(step_times) / len(step_times)
        print(f"  平均每步时间: {avg_time:.2f}s")
        avg_time_ex_first = sum(step_times[1:]) / len(step_times[1:])
        print(f"  平均每步时间（排除首步）: {avg_time_ex_first:.2f}s")
    
    # 转换为 PyTorch tensor
    if hasattr(latents, '_elem'):
        jax_latents = latents._elem
        is_bf16 = (jax_latents.dtype == jnp.bfloat16)
        if is_bf16:
            np_latents = np.array(jax_latents.astype(jnp.float32))
            torch_latents = torch.from_numpy(np_latents).to(torch.bfloat16)
        else:
            np_latents = np.array(jax_latents)
            torch_latents = torch.from_numpy(np_latents)
    else:
        torch_latents = latents.cpu()
    
    print(f"  Latents shape: {torch_latents.shape}")
    print(f"  Latents dtype: {torch_latents.dtype}")
    print(f"  Latents range: [{torch_latents.float().min():.4f}, {torch_latents.float().max():.4f}]")
    
    return torch_latents, elapsed


def main():
    parser = argparse.ArgumentParser(
        description='Wan 2.2 I2V 阶段2：Transformer (Denoising)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 基本用法（使用阶段1的输出）
  python stage2_transformer.py
  
  # 指定输入目录
  python stage2_transformer.py --input_dir ./my_outputs
  
  # 覆盖配置参数
  python stage2_transformer.py --num_steps 20 --guidance_scale 4.0
        """
    )
    
    parser.add_argument(
        '--input_dir', type=str, default='./stage_outputs',
        help='输入目录（包含阶段1输出）'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='输出目录（默认与输入目录相同）'
    )
    
    # 推理参数
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS, help='推理步数')
    parser.add_argument('--guidance_scale', type=float, default=GUIDANCE_SCALE, help='CFG 引导尺度')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--warmup_steps', type=int, default=2, help='预热步数（0=不预热）')
    
    # Sharding 参数
    parser.add_argument('--dp', type=int, default=DEFAULT_DP, help='数据并行维度')
    
    # 模型参数
    parser.add_argument('--model_id', type=str, default=None, help='覆盖模型 ID')
    
    # 其他参数
    parser.add_argument('--profile', action='store_true', default=False, help='运行 profiler')
    parser.add_argument('--profile_output_path', type=str, default='/tmp/wan_prof', help='Profiler 输出路径')
    
    args = parser.parse_args()
    
    output_dir = args.output_dir or args.input_dir
    input_paths = get_default_paths(args.input_dir)
    output_paths = get_default_paths(output_dir)
    
    # 设置 JAX 配置
    setup_jax_cache()
    
    warnings.filterwarnings('ignore')
    logging.getLogger().setLevel(logging.ERROR)
    
    torch.set_default_dtype(torch.bfloat16)
    setup_pytree_registrations()
    
    print(f"\n{'='*60}")
    print("Wan 2.2 I2V 阶段2：Transformer (Denoising)")
    print(f"{'='*60}")
    
    # 加载阶段1配置
    print(f"\n加载阶段1配置: {input_paths['config']}")
    config = load_generation_config(input_paths['config'])
    
    # 设置推理参数
    config['num_inference_steps'] = args.num_steps
    config['guidance_scale'] = args.guidance_scale
    config['seed'] = args.seed
    if args.model_id is not None:
        config['model_id'] = args.model_id
    
    model_id = config.get('model_id', MODEL_ID)
    
    # 设置随机种子
    import random
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    # 加载 embeddings
    print(f"\n加载 embeddings: {input_paths['embeddings']}")
    embeddings_dict, embed_metadata = load_embeddings_from_safetensors(
        input_paths['embeddings'],
        device='cpu',
        restore_dtype=True
    )
    
    print(f"\n加载的 tensor shapes：")
    for key, value in embeddings_dict.items():
        print(f"  - {key}: {value.shape}")
    
    # 加载双 Transformer（在启用 torchax 之前）
    print(f"\n加载模型: {model_id}")
    print("（加载双 Transformer 模型）")
    
    transformer = WanTransformer3DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    transformer_2 = WanTransformer3DModel.from_pretrained(
        model_id, subfolder="transformer_2", torch_dtype=torch.bfloat16
    )
    
    # 加载 scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_id, subfolder="scheduler"
    )
    
    print("✓ 模型加载完成")
    
    # 启用 torchax
    torchax.enable_globally()
    env = torchax.default_env()
    
    # 创建 mesh
    assert len(jax.devices()) % args.dp == 0
    tp_dim = len(jax.devices()) // args.dp
    mesh_devices = mesh_utils.create_device_mesh(
        (args.dp, tp_dim), allow_split_physical_axes=True
    )
    mesh = Mesh(mesh_devices, ("dp", "tp"))
    print(f"\nMesh: {mesh}")
    print(f"  dp_dim={args.dp}, tp_dim={tp_dim}")
    print(f"  总设备数: {len(jax.devices())}")
    
    # 配置 Transformers
    transformer, transformer_2 = setup_transformers_for_jax(
        transformer, transformer_2, mesh, env
    )
    
    # 将 embeddings 转换为 XLA tensor
    print("\n- 将 embeddings 转换为 XLA tensor...")
    with env:
        for key in embeddings_dict:
            embeddings_dict[key] = embeddings_dict[key].to('jax')
    print("  ✓ 所有 embeddings 已转换")
    
    # 将 scheduler 参数移到 JAX
    print("- 将 Scheduler 参数移到 JAX...")
    for k, v in scheduler.__dict__.items():
        if isinstance(v, torch.Tensor):
            setattr(scheduler, k, v.to('jax'))
    
    # 运行推理
    if args.profile:
        print(f"\n[Profiler] 启用 JAX Profiler，输出目录: {args.profile_output_path}")
        profiler_context = jax.profiler.trace(
            args.profile_output_path,
            create_perfetto_link=False
        )
    else:
        profiler_context = nullcontext()
    
    with profiler_context:
        latents, elapsed = run_transformer_inference(
            transformer, transformer_2, scheduler,
            embeddings_dict, config, mesh,
            warmup_steps=args.warmup_steps
        )
    
    # 保存 latents
    print(f"\n保存 latents 到: {output_paths['latents']}")
    metadata = {
        'num_frames': str(config['num_frames']),
        'num_inference_steps': str(config['num_inference_steps']),
        'guidance_scale': str(config['guidance_scale']),
        'seed': str(config['seed']),
        'height': str(config['height']),
        'width': str(config['width']),
    }
    save_latents_to_safetensors(latents, output_paths['latents'], metadata)
    
    # 更新配置
    save_generation_config(config, output_paths['config'])
    
    # 打印性能统计
    print(f"\n=== 性能统计 ===")
    print(f"推理时间: {elapsed:.4f} 秒")
    print(f"Block sizes: BQSIZE={BQSIZE}, BKVSIZE={BKVSIZE}, BKVCOMPUTESIZE={BKVCOMPUTESIZE}")
    
    print(f"\n{'='*60}")
    print("阶段2 完成！")
    print(f"{'='*60}")
    print(f"\n输出文件：")
    print(f"  - Latents: {output_paths['latents']}")
    print(f"\n下一步：运行 stage3_vae_decoder.py 进行 VAE 解码")
    
    print("\n✓ 阶段2 执行完成")
    
    # 强制退出以避免 torchax 后台线程阻塞
    os._exit(0)


if __name__ == "__main__":
    main()