#!/usr/bin/env python3
"""
Flux.2 三阶段生成 - 阶段2：Transformer (Denoising)

本阶段负责：
1. 加载阶段1生成的 prompt embeddings
2. 设置 JAX/TPU 环境和 Splash Attention
3. 加载 Transformer 模型并进行权重分片
4. 运行 denoising loop 生成 latents
5. 将 latents 保存为 SafeTensors 格式

输入文件：
- stage1_embeddings.safetensors: prompt embeddings
- generation_config.json: 生成配置

输出文件：
- stage2_latents.safetensors: 生成的 latents

注意：Flux.2 使用 Embedded CFG，guidance_scale 嵌入到 timestep embedding 中，
      不需要 negative_prompt_embeds，只使用 1D mesh (tp)。
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

import jax
import jax.numpy as jnp
import torch
import torchax
from torchax.ops import ops_registry, jaten
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
from jax.experimental import mesh_utils
from tqdm import tqdm

# 使用本地 splash_attention_utils（父目录）
sys.path.insert(0, str(Path(__file__).parent.parent))
from splash_attention_utils import tpu_splash_attention, sdpa_reference

from diffusers.pipelines.flux2.pipeline_flux2_torchax import Flux2Pipeline
from diffusers.models.autoencoders.autoencoder_kl_flux2_torchax import AutoencoderKLFlux2
from diffusers.models.transformers.transformer_flux2_torchax import Flux2Transformer2DModel
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from utils import (
    MODEL_NAME,
    WIDTH, HEIGHT, NUM_STEPS, GUIDANCE_SCALE,
    USE_K_SMOOTH,
    TRANSFORMER_SHARDINGS,
    shard_weight_dict,
    setup_jax_cache,
    setup_pytree_registrations,
    load_embeddings_from_safetensors,
    save_latents_to_safetensors,
    load_generation_config,
    save_generation_config,
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


def torch_conv2d_jax(input, weight, bias=None, stride=1, padding=0,
                     dilation=1, groups=1, *, env):
    """JAX 兼容的 conv2d 覆盖实现。"""
    jinput, jweight, jbias = env.t2j_iso((input, weight, bias))
    res = jaten._aten_conv2d(jinput, jweight, jbias, stride, padding, dilation, groups)
    return env.j2t_iso(res)


# ============================================================================
# Pipeline 设置
# ============================================================================

def move_module_to_xla(env, module):
    """Move module weights to XLA devices."""
    with jax.default_device("cpu"):
        state_dict = module.state_dict()
        state_dict = env.to_xla(state_dict)
        module.load_state_dict(state_dict, assign=True)


def override_op_definition(env, op, impl):
    """在 torchax 环境中覆盖算子定义。"""
    env._ops[op] = ops_registry.Operator(
        op, impl, is_jax_function=False, is_user_defined=True,
        needs_env=False, is_view_op=False,
    )


def setup_pipeline_for_transformer_only(pipe, mesh, env):
    """
    设置 Pipeline 仅用于 Transformer 推理（不包含 VAE）
    """
    print("\n=== 配置 Transformer (TPU) ===")

    # Register custom ops
    print("- 注册自定义 JAX 算子...")
    override_op_definition(env, torch.nn.functional.conv2d,
                           functools.partial(torch_conv2d_jax, env=env))
    override_op_definition(env, torch.nn.functional.scaled_dot_product_attention,
                           functools.partial(scaled_dot_product_attention, env=env, mesh=mesh))

    # Move Transformer to XLA
    print("- 将 Transformer 移到 TPU...")
    move_module_to_xla(env, pipe.transformer)

    # Compile Transformer
    print("- 编译 Transformer...")
    options = torchax.CompileOptions(
        jax_jit_kwargs={'static_argnames': ('return_dict',)}
    )
    pipe.transformer = torchax.compile(pipe.transformer, options)

    # Apply sharding
    print("- 对 Transformer 进行权重分片...")
    pipe.transformer.params = shard_weight_dict(
        pipe.transformer.params, TRANSFORMER_SHARDINGS, mesh
    )
    pipe.transformer.buffers = shard_weight_dict(
        pipe.transformer.buffers, TRANSFORMER_SHARDINGS, mesh
    )
    
    # Wait for sharding to complete
    torchax.interop.call_jax(jax.block_until_ready, pipe.transformer.params)

    # Delete VAE to save memory (not needed in stage 2)
    print("- 删除 VAE 以节省内存（阶段2不需要）...")
    if hasattr(pipe, 'vae') and pipe.vae is not None:
        del pipe.vae
        pipe.vae = None

    # Text Encoder is already None (loaded with text_encoder=None)
    print("✓ Transformer 配置完成")
    return pipe


def run_denoising_loop(
    pipe,
    prompt_embeds,
    config,
    mesh,
    env,
    num_steps,
    desc="Denoising",
    is_warmup=False,
):
    """
    Flux.2 的 Denoising 循环。
    
    注意：Flux.2 使用 Embedded CFG，只需要一个 prompt_embeds，
    guidance_scale 嵌入到 timestep embedding 中。
    """
    step_times = []
    start_time = time.perf_counter()
    
    generator = torch.Generator()
    generator.manual_seed(config['seed'])
    
    # Flux.2 pipeline 参数
    gen_kwargs = {
        'prompt': None,  # Already encoded
        'prompt_embeds': prompt_embeds,
        'height': config['height'],
        'width': config['width'],
        'num_inference_steps': num_steps,
        'guidance_scale': config['guidance_scale'],
        'generator': generator,
        'output_type': 'latent',  # 关键：只返回 latents，不解码
    }
    
    with mesh:
        # 使用自定义的 callback 来追踪每一步的时间
        step_start_time = [None]  # 用列表包装以在闭包中修改
        
        # tqdm 进度条
        progress_bar = tqdm(
            total=num_steps,
            desc=desc,
            ncols=130,
        )
        
        def step_callback(pipe, step_index, timestep, callback_kwargs):
            """每一步结束后的回调，用于更新进度条"""
            nonlocal step_times
            
            # 等待计算完成（JAX/XLA 是惰性执行的）
            jax.effects_barrier()
            
            if step_start_time[0] is not None:
                step_time = time.perf_counter() - step_start_time[0]
                step_times.append(step_time)
                avg_time = sum(step_times) / len(step_times)
                
                if is_warmup:
                    progress_bar.set_postfix({
                        'step': f'{step_time:.2f}s',
                    })
                else:
                    remaining_steps = num_steps - step_index - 1
                    progress_bar.set_postfix({
                        'step': f'{step_time:.2f}s',
                        'avg': f'{avg_time:.2f}s',
                        'eta': f'{avg_time * remaining_steps:.1f}s',
                    })
                
                progress_bar.update(1)
            
            # 记录下一步的开始时间
            step_start_time[0] = time.perf_counter()
            
            return callback_kwargs
        
        # 记录第一步的开始时间
        step_start_time[0] = time.perf_counter()
        
        # 添加 callback
        gen_kwargs['callback_on_step_end'] = step_callback
        
        result = pipe(**gen_kwargs)
        jax.effects_barrier()
        latents = result.images  # output_type='latent' 时，images 就是 latents
        
        # 处理最后一步
        if step_start_time[0] is not None:
            step_time = time.perf_counter() - step_start_time[0]
            step_times.append(step_time)
            avg_time = sum(step_times) / len(step_times)
            
            if is_warmup:
                progress_bar.set_postfix({
                    'step': f'{step_time:.2f}s',
                })
            else:
                progress_bar.set_postfix({
                    'step': f'{step_time:.2f}s',
                    'avg': f'{avg_time:.2f}s',
                    'eta': '0.0s',
                })
            progress_bar.update(1)
        
        progress_bar.close()
    
    elapsed = time.perf_counter() - start_time
    return latents, step_times, elapsed


def run_transformer_inference(pipe, prompt_embeds, config, mesh, env,
                               warmup_steps=0):
    """
    运行 Transformer 推理生成 latents
    
    使用 output_type='latent' 让 pipeline 跳过 VAE 解码
    """
    print(f"\n=== 阶段2：Transformer 推理 ===")
    print(f"推理步数: {config['num_inference_steps']}")
    print(f"引导尺度: {config['guidance_scale']}")
    print(f"随机种子: {config['seed']}")
    print(f"分辨率: {config['height']}x{config['width']}")
    if warmup_steps > 0:
        print(f"预热步数: {warmup_steps}")

    # === Warmup (可选) ===
    if warmup_steps > 0:
        print(f"\n预热中（{warmup_steps}步，触发 JIT 编译）...")
        
        _, warmup_times, warmup_elapsed = run_denoising_loop(
            pipe=pipe,
            prompt_embeds=prompt_embeds,
            config=config,
            mesh=mesh,
            env=env,
            num_steps=warmup_steps,
            desc="Warmup (JIT)",
            is_warmup=True,
        )
        
        print(f"  ✓ 预热完成，耗时: {warmup_elapsed:.2f}秒")

    # === 正式推理 ===
    print("\n开始 Transformer 推理...")
    
    latents, step_times, elapsed = run_denoising_loop(
        pipe=pipe,
        prompt_embeds=prompt_embeds,
        config=config,
        mesh=mesh,
        env=env,
        num_steps=config['num_inference_steps'],
        desc="Denoising (TPU)",
        is_warmup=False,
    )
    
    print(f"\n✓ Transformer 推理完成，耗时: {elapsed:.2f} 秒")
    
    # 打印性能统计
    if len(step_times) > 1:
        avg_time = sum(step_times) / len(step_times)
        print(f"  平均每步时间: {avg_time:.2f}s")
        # 排除第一步（可能包含额外编译时间）
        avg_time_ex_first = sum(step_times[1:]) / len(step_times[1:])
        print(f"  平均每步时间（排除首步）: {avg_time_ex_first:.2f}s")

    # 转换为可保存的格式
    if hasattr(latents, '_elem'):
        # torchax tensor -> JAX array -> numpy (float32) -> torch -> bfloat16
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
    print(f"  Latents range: [{torch_latents.min():.4f}, {torch_latents.max():.4f}]")

    return torch_latents, elapsed


def main():
    parser = argparse.ArgumentParser(
        description='Flux.2 阶段2：Transformer (Denoising)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 基本用法（使用阶段1的输出）
  python stage2_transformer.py
  
  # 指定输入目录
  python stage2_transformer.py --input_dir ./my_outputs
  
  # 覆盖配置参数
  python stage2_transformer.py --num_inference_steps 50
        """
    )

    parser.add_argument(
        '--input_dir', type=str, default='./stage_outputs',
        help='Input directory containing stage1 outputs (default: ./stage_outputs)'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory (default: same as input_dir)'
    )

    # 可覆盖的配置参数
    parser.add_argument('--num_inference_steps', type=int, default=NUM_STEPS, help=f'推理步数（默认{NUM_STEPS}）')
    parser.add_argument('--guidance_scale', type=float, default=GUIDANCE_SCALE, help=f'引导尺度（默认{GUIDANCE_SCALE}）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子（默认42）')
    parser.add_argument('--height', type=int, default=HEIGHT, help=f'图像高度（默认{HEIGHT}）')
    parser.add_argument('--width', type=int, default=WIDTH, help=f'图像宽度（默认{WIDTH}）')
    parser.add_argument('--warmup_steps', type=int, default=2,
                        help='预热步数（0=不预热，1=一次，2=两次，用于触发 JIT 编译）')

    # 其他参数
    parser.add_argument('--model_id', type=str, default=None, help='Override model ID from stage1')
    parser.add_argument('--num_iterations', type=int, default=1, help='Number of benchmark iterations')
    parser.add_argument('--profile', action='store_true', default=False, help='Run profiler')
    parser.add_argument('--profiler_output_dir', type=str, default='/dev/shm/jax-trace',
                        help='Profiler 输出目录')

    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    input_paths = get_default_paths(args.input_dir)
    output_paths = get_default_paths(output_dir)

    # 设置 JAX 配置
    setup_jax_cache()

    warnings.filterwarnings('ignore', message='.*dtype.*int64.*truncated to dtype int32.*')
    warnings.filterwarnings('ignore')
    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger('diffusers').setLevel(logging.ERROR)
    logging.getLogger('transformers').setLevel(logging.ERROR)

    torch.set_default_dtype(torch.bfloat16)
    setup_pytree_registrations()

    print(f"\n{'='*60}")
    print("Flux.2 阶段2：Transformer (Denoising)")
    print(f"{'='*60}")

    # 加载阶段1配置
    print(f"\n加载阶段1配置: {input_paths['config']}")
    config = load_generation_config(input_paths['config'])

    # Stage2 专用参数（使用命令行参数或默认值）
    config['num_inference_steps'] = args.num_inference_steps
    config['guidance_scale'] = args.guidance_scale
    config['seed'] = args.seed
    config['height'] = args.height
    config['width'] = args.width
    if args.model_id is not None:
        config['model_id'] = args.model_id

    # 设置随机种子
    import random
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    # 加载 embeddings
    print(f"\n加载 embeddings: {input_paths['embeddings']}")
    prompt_embeds_dict, embed_metadata = load_embeddings_from_safetensors(
        input_paths['embeddings'],
        device='cpu',
        restore_dtype=True
    )
    prompt_embeds = prompt_embeds_dict['prompt_embeds']
    print(f"  prompt_embeds shape: {prompt_embeds.shape}")

    # 加载 Pipeline（仅 Transformer，在启用 torchax 之前）
    model_id = config.get('model_id', MODEL_NAME)
    print(f"\n加载模型: {model_id}")
    print("（注意：仅加载 Transformer 组件，不加载 VAE 和 Text Encoder）")

    scheduler = FlowMatchEulerDiscreteScheduler()
    
    # 显式加载 torchax 版本的模型
    transformer = Flux2Transformer2DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    
    # 创建一个最小的 VAE 用于 pipeline 初始化（稍后会删除）
    vae = AutoencoderKLFlux2.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.bfloat16
    )
    
    # 使用 text_encoder=None
    pipe = Flux2Pipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        text_encoder=None,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
    )
    print("✓ Pipeline 加载完成")

    # 初始化 torchax 和创建环境
    torchax.enable_globally()
    env = torchax.default_env()

    # 创建 mesh (只用 tp，1D mesh)
    tp_dim = len(jax.devices())
    mesh_devices = mesh_utils.create_device_mesh(
        (tp_dim,), allow_split_physical_axes=True
    )
    mesh = Mesh(mesh_devices, ("tp",))
    print(f"\nMesh: tp={tp_dim}, 总设备数={len(jax.devices())}")

    # 配置 Pipeline
    pipe = setup_pipeline_for_transformer_only(pipe, mesh, env)

    # 将 embeddings 转换为 XLA tensor
    print("\n- 将 embeddings 转换为 XLA tensor...")
    with env:
        prompt_embeds = prompt_embeds.to('jax')
    print("  ✓ embeddings 已转换")

    # 运行推理
    times = []
    latents = None

    # 第一次迭代（包含预热）
    if args.num_iterations >= 1:
        print(f"\n--- 迭代 1/{args.num_iterations} ---")
        latents, elapsed = run_transformer_inference(
            pipe, prompt_embeds, config, mesh, env,
            warmup_steps=args.warmup_steps
        )
        times.append(elapsed)
        print(f"  耗时: {elapsed:.2f} 秒" + (" (不含预热)" if args.warmup_steps > 0 else ""))

    # 后续迭代（可选的 profiler 包裹）
    if args.num_iterations > 1:
        # 创建 profiler context
        if args.profile:
            print(f"\n[Profiler] 启用 JAX Profiler，输出目录: {args.profiler_output_dir}")
            print(f"[Profiler] 将 profile 后续 {args.num_iterations - 1} 次迭代")
            profiler_context = jax.profiler.trace(
                args.profiler_output_dir,
                create_perfetto_link=False
            )
        else:
            profiler_context = nullcontext()
        
        with profiler_context:
            for i in range(1, args.num_iterations):
                print(f"\n--- 迭代 {i+1}/{args.num_iterations} ---")
                latents, elapsed = run_transformer_inference(
                    pipe, prompt_embeds, config, mesh, env,
                    warmup_steps=0  # 后续迭代不需要预热
                )
                times.append(elapsed)
                print(f"  耗时: {elapsed:.2f} 秒")

    # 保存 latents
    print(f"\n保存 latents 到: {output_paths['latents']}")
    metadata = {
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
    print(f"总迭代次数: {len(times)}")
    print(f"第一次运行（含编译）: {times[0]:.4f} 秒")
    if len(times) > 1:
        avg_time = sum(times[1:]) / len(times[1:])
        print(f"后续运行平均时间: {avg_time:.4f} 秒")

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
