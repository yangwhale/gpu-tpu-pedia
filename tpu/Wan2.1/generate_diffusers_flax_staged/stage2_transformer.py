#!/usr/bin/env python3
"""
Wan 2.1 三阶段生成 - 阶段2：Transformer (Denoising)

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
"""

import os
import sys
import time
import re
import math
import functools
import argparse
import warnings
import logging
import numpy as np
from contextlib import nullcontext
import jax
import jax.numpy as jnp
import torch
import torchax
from torchax.ops import ops_registry
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
from jax.experimental.pallas.ops.tpu import splash_attention
from jax.experimental.shard_map import shard_map
from jax.experimental import mesh_utils
from tqdm import tqdm

# Add parent directory to path for custom_splash_attention
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import custom_splash_attention

from diffusers.pipelines.wan.pipeline_wan_flax import WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

from utils import (
    MODEL_NAME,
    FLOW_SHIFT,
    BQSIZE, BKVSIZE, BKVCOMPUTESIZE, BKVCOMPUTEINSIZE,
    USE_K_SMOOTH, USE_CUSTOM_ATTENTION, LOG2_E,
    DEFAULT_DP,
    TRANSFORMER_SHARDINGS,
    shard_weight_dict,
    setup_jax_cache,
    pad_to_multiple,
    setup_pytree_registrations,
    load_embeddings_from_safetensors,
    save_latents_to_safetensors,
    load_generation_config,
    get_default_paths,
)


# === Splash Attention 实现 ===

def _sdpa_reference(query, key, value, attn_mask=None, dropout_p=0.0,
                    is_causal=False, scale=None, enable_gqa=False):
    """Reference implementation of Scaled Dot-Product Attention."""
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    
    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    
    if dropout_p > 0:
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    
    return attn_weight @ value


def _tpu_splash_attention(query, key, value, env, scale=None, is_causal=False,
                           window_size=None, bqsize=BQSIZE, bkvsize=BKVSIZE,
                           bkvcomputesize=BKVCOMPUTESIZE):
    """TPU Splash Attention implementation with sharding support."""
    mesh = getattr(env, '_mesh', None) or env.param.mesh
    num_heads = query.shape[1]

    def _attention_on_slices(q, k, v):
        scale_factor = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
        q = q * scale_factor

        def pad_to_multiple(x, multiple, axis):
            seq_len = x.shape[axis]
            pad_len = (multiple - seq_len % multiple) % multiple
            if pad_len == 0:
                return x, seq_len
            pad_width = [(0, 0)] * x.ndim
            pad_width[axis] = (0, pad_len)
            return jnp.pad(x, pad_width), seq_len

        def kernel_3d(q_3d, k_3d, v_3d):
            num_heads_on_device = q_3d.shape[0]
            
            # block_kv_compute must be a multiple of 128 (NUM_LANES in splash attention)
            MIN_KV_MULTIPLE = 128
            
            # Self attention (long KV sequence)
            if k_3d.shape[1] > 10000:
                q_3d_padded, q_orig_len = pad_to_multiple(q_3d, bqsize, axis=1)
                k_3d_padded, k_orig_len = pad_to_multiple(k_3d, bkvsize, axis=1)
                v_3d_padded, v_orig_len = pad_to_multiple(v_3d, bkvsize, axis=1)
            else:
                # Cross attention (short KV sequence)
                # Need to pad KV to at least 128 to satisfy block_kv_compute constraint
                q_3d_padded, q_orig_len = pad_to_multiple(q_3d, bqsize, axis=1)
                k_3d_padded, k_orig_len = pad_to_multiple(k_3d, MIN_KV_MULTIPLE, axis=1)
                v_3d_padded, v_orig_len = pad_to_multiple(v_3d, MIN_KV_MULTIPLE, axis=1)

            padded_q_seq_len = q_3d_padded.shape[1]
            padded_kv_seq_len = k_3d_padded.shape[1]

            # Create attention mask
            if window_size is not None:
                mask_class = functools.partial(
                    splash_attention.LocalMask, window_size=window_size, offset=0
                )
            else:
                mask_class = splash_attention.FullMask

            mask = splash_attention.MultiHeadMask([
                mask_class((padded_q_seq_len, padded_kv_seq_len))
                for _ in range(num_heads_on_device)
            ])

            # For cross attention, use smaller block sizes
            if k_3d.shape[1] <= 10000:
                actual_bkvsize = min(256, padded_kv_seq_len)
                actual_bkvcomputesize = min(128, padded_kv_seq_len)
            else:
                actual_bkvsize = min(bkvsize, padded_kv_seq_len)
                actual_bkvcomputesize = min(bkvcomputesize, padded_kv_seq_len)

            block_sizes = splash_attention.BlockSizes(
                block_q=min(bqsize, padded_q_seq_len),
                block_kv=actual_bkvsize,
                block_kv_compute=actual_bkvcomputesize,
            )
            
            splash_kernel = splash_attention.make_splash_mha(
                mask=mask, block_sizes=block_sizes, head_shards=1, q_seq_shards=1
            )
            out = splash_kernel(q_3d_padded, k_3d_padded, v_3d_padded)
            return out[:, :q_orig_len, ...]

        vmapped_kernel = jax.vmap(kernel_3d, in_axes=(0, 0, 0), out_axes=0)
        return vmapped_kernel(q, k, v)

    # Determine partition specs based on attention type
    if num_heads < mesh.size:
        # Replicated for VAE
        q_partition_spec = P()
        kv_partition_spec = P()
    else:
        # Self attention (long KV)
        if key.shape[2] > 10000:
            q_partition_spec = P('dp', 'tp', 'sp', None)
            kv_partition_spec = P('dp', 'tp', None, None)
        else:
            # Cross attention (short KV)
            q_partition_spec = P('dp', None, ('tp', 'sp'), None)
            kv_partition_spec = P('dp', None, None, None)

    sharded_fn = shard_map(
        _attention_on_slices,
        mesh=mesh,
        in_specs=(q_partition_spec, kv_partition_spec, kv_partition_spec),
        out_specs=q_partition_spec,
        check_rep=False,
    )
    out = sharded_fn(query, key, value)
    out = jax.lax.with_sharding_constraint(out, P('dp', None, ('tp', 'sp'), None))
    return out


def _tpu_custom_attention(query, key, value, mesh, scale=None):
    """TPU Custom Splash Attention with exp2 optimization."""
    def _attention_on_slices(q, k, v):
        scale_factor = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
        # Fuse the ops of exp in softmax - multiply by log2(e)
        q = q * scale_factor * LOG2_E

        def kernel_3d(q_3d, k_3d, v_3d):
            # Pad to block size multiple to avoid NaN in incomplete blocks
            def pad_to_block_multiple(x, block_size, axis):
                seq_len = x.shape[axis]
                pad_len = (block_size - seq_len % block_size) % block_size
                if pad_len == 0:
                    return x, seq_len
                pad_width = [(0, 0)] * x.ndim
                pad_width[axis] = (0, pad_len)
                return jnp.pad(x, pad_width), seq_len
            
            q_3d_padded, q_orig_len = pad_to_block_multiple(q_3d, BQSIZE, axis=1)
            k_3d_padded, k_orig_len = pad_to_block_multiple(k_3d, BKVSIZE, axis=1)
            v_3d_padded, v_orig_len = pad_to_block_multiple(v_3d, BKVSIZE, axis=1)

            padded_q_seq_len = q_3d_padded.shape[1]
            padded_kv_seq_len = k_3d_padded.shape[1]

            block_sizes = splash_attention.BlockSizes(
                block_q=min(BQSIZE, padded_q_seq_len),
                block_kv=min(BKVSIZE, padded_kv_seq_len),
                block_kv_compute=min(BKVCOMPUTESIZE, padded_kv_seq_len),
            )
            splash_kernel = custom_splash_attention.make_splash_mha(
                block_sizes=block_sizes, bkv_compute_in=BKVCOMPUTEINSIZE
            )
            out = splash_kernel(q_3d_padded, k_3d_padded, v_3d_padded).astype(q_3d.dtype)
            out = jnp.swapaxes(out, 1, 2)
            # Remove padding
            return out[:, :q_orig_len, :]

        vmapped_kernel = jax.vmap(kernel_3d, in_axes=(0, 0, 0), out_axes=0)
        return vmapped_kernel(q, k, v)

    # Determine sharding strategy
    if key.shape[0] > 1:
        dp_mesh_key = "dp"
        remain_mesh_key = ("tp",)
    else:
        dp_mesh_key = None
        remain_mesh_key = ("dp", "tp")
    
    remain_devices_prod = 1
    for d in remain_mesh_key:
        remain_devices_prod *= mesh.axis_sizes[mesh.axis_names.index(d)]

    q_num_head = query.shape[1]
    q_seq_len = query.shape[2]
    kv_num_head = key.shape[1]
    kv_seq_len = key.shape[2]
    
    # Attn1 self attention (long KV sequence) - use context parallel
    if (kv_seq_len > 10000 and
        kv_num_head % remain_devices_prod == 0 and
        q_num_head % remain_devices_prod == 0):
        q_partition_spec = P(dp_mesh_key, remain_mesh_key, None, None)
        kv_partition_spec = P(dp_mesh_key, remain_mesh_key, None, None)
    else:
        # Attn2 cross attention (short KV) - use sequence parallel
        if q_seq_len % remain_devices_prod != 0:
            query, _ = pad_to_multiple(query, remain_devices_prod, axis=2)
        
        q_partition_spec = P(dp_mesh_key, None, remain_mesh_key, None)
        kv_partition_spec = P(dp_mesh_key, None, None, None)

    # Apply sharding constraints
    sharded_fn = shard_map(
        _attention_on_slices,
        mesh=mesh,
        in_specs=(q_partition_spec, kv_partition_spec, kv_partition_spec),
        out_specs=q_partition_spec,
        check_rep=False,
    )
    
    query = jax.lax.with_sharding_constraint(
        query, P(dp_mesh_key, None, remain_mesh_key, None)
    )
    key = jax.lax.with_sharding_constraint(
        key, P(dp_mesh_key, None, remain_mesh_key, None)
    )
    value = jax.lax.with_sharding_constraint(
        value, P(dp_mesh_key, None, remain_mesh_key, None)
    )
    
    out = sharded_fn(query, key, value)
    
    # Remove potential padding for sp
    out = out[:, :, :q_seq_len, :]
    out = jax.lax.with_sharding_constraint(
        out, P(dp_mesh_key, None, remain_mesh_key, None)
    )
    return out


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                  is_causal=False, scale=None, enable_gqa=False,
                                  env=None, mesh=None):
    """Wrapper for scaled dot-product attention with TPU Splash support."""
    # Only use custom attention for long sequences (self-attention)
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
        
        res = _tpu_custom_attention(jquery, jkey, jvalue, mesh, scale=scale)
        return env.j2t_iso(res)

    return _sdpa_reference(query, key, value, attn_mask, dropout_p,
                          is_causal, scale, enable_gqa)




# ============================================================================
# Pipeline 设置
# ============================================================================

def move_module_to_xla(env, module):
    """Move module weights to XLA devices."""
    with jax.default_device("cpu"):
        state_dict = module.state_dict()
        state_dict = env.to_xla(state_dict)
        module.load_state_dict(state_dict, assign=True)


def setup_pipeline_for_transformer_only(pipe, mesh, env):
    """
    设置 Pipeline 仅用于 Transformer 推理（不包含 VAE）
    """
    print("\n=== 配置 Transformer (TPU) ===")

    # Register custom attention
    print("- 注册自定义 JAX 算子...")
    custom_attention = functools.partial(
        scaled_dot_product_attention,
        env=env,
        mesh=mesh,
    )
    op_to_override = torch.nn.functional.scaled_dot_product_attention
    env._ops[op_to_override] = ops_registry.Operator(
        op_to_override,
        custom_attention,
        is_jax_function=False,
        is_user_defined=True,
        needs_env=False,
        is_view_op=False,
    )

    # Move Transformer to XLA
    print("- 将 Transformer 移到 TPU...")
    move_module_to_xla(env, pipe.transformer)
    
    # Move rope embeddings to JAX
    if hasattr(pipe.transformer.rope, 'freqs'):
        pipe.transformer.rope.freqs = pipe.transformer.rope.freqs.to('jax')
    else:
        pipe.transformer.rope.freqs_cos = pipe.transformer.rope.freqs_cos.to('jax')
        pipe.transformer.rope.freqs_sin = pipe.transformer.rope.freqs_sin.to('jax')

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

    # Delete Text Encoder (already used in stage 1)
    print("- 删除 Text Encoder（阶段1已完成编码）...")
    if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
        del pipe.text_encoder
        pipe.text_encoder = None

    print("✓ Transformer 配置完成")
    return pipe


def run_denoising_loop(
    pipe,
    prompt_embeds_dict,
    config,
    mesh,
    env,
    num_steps,
    desc="Denoising",
    is_warmup=False,
    profiler_ctx=None,
):
    """
    统一的 Denoising 循环，预热和正式推理共用同一套代码。
    
    参照 HunyuanVideo-1.5 的 run_denoising_loop 实现，使用 tqdm 显示时间信息。
    
    Args:
        pipe: WanPipeline 实例
        prompt_embeds_dict: prompt embeddings 字典
        config: 生成配置
        mesh: JAX mesh
        env: torchax 环境
        num_steps: 运行步数
        desc: 进度条描述
        is_warmup: 是否是预热模式（影响进度条显示）
        profiler_ctx: Profiler context（可选）
    
    Returns:
        (latents, step_times, elapsed_time)
    """
    step_times = []
    start_time = time.perf_counter()
    
    # 使用 profiler context（如果提供）
    ctx = profiler_ctx if profiler_ctx else nullcontext()
    
    generator = torch.Generator()
    generator.manual_seed(config['seed'])
    
    gen_kwargs = {
        'prompt': None,  # Already encoded
        'negative_prompt': None,  # Already encoded
        'prompt_embeds': prompt_embeds_dict['prompt_embeds'],
        'negative_prompt_embeds': prompt_embeds_dict['negative_prompt_embeds'],
        'height': config['height'],
        'width': config['width'],
        'num_frames': config['frames'],
        'num_inference_steps': num_steps,
        'guidance_scale': config['guidance_scale'],
        'generator': generator,
        'output_type': 'latent',  # 关键：只返回 latents，不解码
        'use_dp': config.get('use_dp', True),
    }
    
    with ctx:
        with mesh:
            # 使用自定义的 callback 来追踪每一步的时间
            step_start_time = [None]  # 用列表包装以在闭包中修改
            current_step = [0]
            
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
                current_step[0] = step_index + 1
                
                return callback_kwargs
            
            # 记录第一步的开始时间
            step_start_time[0] = time.perf_counter()
            
            # 添加 callback
            gen_kwargs['callback_on_step_end'] = step_callback
            
            result = pipe(**gen_kwargs)
            jax.effects_barrier()
            latents = result.frames  # output_type='latent' 时，frames 就是 latents
            
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


def run_transformer_inference(pipe, prompt_embeds_dict, config, mesh, env,
                               warmup_steps=0):
    """
    运行 Transformer 推理生成 latents
    
    使用 output_type='latent' 让 pipeline 跳过 VAE 解码
    
    参照 HunyuanVideo-1.5 的实现，支持 warmup 预热和正式推理分离。
    
    Args:
        pipe: WanPipeline 实例
        prompt_embeds_dict: prompt embeddings 字典
        config: 生成配置
        mesh: JAX mesh
        env: torchax 环境
        warmup_steps: 预热步数（0=不预热）
    
    Returns:
        (torch_latents, elapsed_time)
    """
    print(f"\n=== 阶段2：Transformer 推理 ===")
    print(f"推理步数: {config['num_inference_steps']}")
    print(f"帧数: {config['frames']}")
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
            prompt_embeds_dict=prompt_embeds_dict,
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
        prompt_embeds_dict=prompt_embeds_dict,
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
        description='Wan 2.1 阶段2：Transformer (Denoising)',
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

    # 可覆盖的配置参数（这些参数有默认值，适用于 T2V 480P）
    parser.add_argument('--num_inference_steps', type=int, default=50, help='推理步数（默认50）')
    parser.add_argument('--guidance_scale', type=float, default=5.0, help='引导尺度（默认5.0）')
    parser.add_argument('--seed', type=int, default=2025, help='随机种子（默认2025）')
    parser.add_argument('--height', type=int, default=480, help='视频高度（默认480）')
    parser.add_argument('--width', type=int, default=848, help='视频宽度（默认848）')
    parser.add_argument('--frames', type=int, default=81, help='视频帧数（默认81，约5秒）')
    parser.add_argument('--warmup_steps', type=int, default=2,
                        help='预热步数（0=不预热，1=一次，2=两次，用于触发 JIT 编译）')

    # Sharding 参数
    parser.add_argument('--dp', type=int, default=DEFAULT_DP, help='数据并行维度')

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
    logging.getLogger().setLevel(logging.ERROR)

    torch.set_default_dtype(torch.bfloat16)
    setup_pytree_registrations()

    print(f"\n{'='*60}")
    print("Wan 2.1 阶段2：Transformer (Denoising)")
    print(f"{'='*60}")

    # 加载阶段1配置
    print(f"\n加载阶段1配置: {input_paths['config']}")
    config = load_generation_config(input_paths['config'])

    # Stage2 专用参数（使用命令行参数或默认值）
    # Stage1 只保存 prompt 相关信息，视频参数由 stage2 控制
    config['num_inference_steps'] = args.num_inference_steps
    config['guidance_scale'] = args.guidance_scale
    config['seed'] = args.seed
    config['height'] = args.height
    config['width'] = args.width
    config['frames'] = args.frames
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

    # 初始化 torchax 和创建环境
    torchax.enable_globally()
    env = torchax.default_env()

    # 创建 mesh (简化为 dp, tp 两维)
    assert len(jax.devices()) % args.dp == 0
    tp_dim = len(jax.devices()) // args.dp
    mesh_devices = mesh_utils.create_device_mesh(
        (args.dp, tp_dim), allow_split_physical_axes=True
    )
    mesh = Mesh(mesh_devices, ("dp", "tp"))
    print(f"\nMesh: {mesh}")
    print(f"  dp_dim={args.dp}, tp_dim={tp_dim}")
    print(f"  总设备数: {len(jax.devices())}")

    # 加载 Pipeline（仅 Transformer）
    model_id = config.get('model_id', MODEL_NAME)
    print(f"\n加载模型: {model_id}")
    print("（注意：仅加载 Transformer 组件）")

    # 临时禁用 torchax 加载 PyTorch 组件
    torchax.disable_globally()
    try:
        scheduler = UniPCMultistepScheduler(
            prediction_type='flow_prediction',
            use_flow_sigmas=True,
            num_train_timesteps=1000,
            flow_shift=config.get('flow_shift', FLOW_SHIFT)
        )
        pipe = WanPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        )
        pipe.scheduler = scheduler
    finally:
        torchax.enable_globally()

    # 配置 Pipeline
    pipe = setup_pipeline_for_transformer_only(pipe, mesh, env)

    # 将 embeddings 转换为 XLA tensor
    print("\n- 将 embeddings 转换为 XLA tensor...")
    with env:
        for key in prompt_embeds_dict:
            prompt_embeds_dict[key] = prompt_embeds_dict[key].to('jax')
    print("  ✓ 所有 embeddings 已转换")

    # 运行推理（profiler 包裹整个正式运行阶段，不包括预热）
    times = []
    latents = None

    # 第一次迭代（包含预热）
    if args.num_iterations >= 1:
        print(f"\n--- 迭代 1/{args.num_iterations} ---")
        latents, elapsed = run_transformer_inference(
            pipe, prompt_embeds_dict, config, mesh, env,
            warmup_steps=args.warmup_steps
        )
        times.append(elapsed)
        print(f"  耗时: {elapsed:.2f} 秒" + (" (不含预热)" if args.warmup_steps > 0 else ""))

    # 后续迭代（可选的 profiler 包裹）
    if args.num_iterations > 1:
        # 创建 profiler context（仅包裹正式推理，不包括预热）
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
                    pipe, prompt_embeds_dict, config, mesh, env,
                    warmup_steps=0  # 后续迭代不需要预热
                )
                times.append(elapsed)
                print(f"  耗时: {elapsed:.2f} 秒")

    # 保存 latents
    print(f"\n保存 latents 到: {output_paths['latents']}")
    metadata = {
        'num_frames': str(config['frames']),
        'num_inference_steps': str(config['num_inference_steps']),
        'guidance_scale': str(config['guidance_scale']),
        'seed': str(config['seed']),
        'height': str(config['height']),
        'width': str(config['width']),
    }
    save_latents_to_safetensors(latents, output_paths['latents'], metadata)

    # 更新配置
    from utils import save_generation_config
    save_generation_config(config, output_paths['config'])

    # 打印性能统计
    print(f"\n=== 性能统计 ===")
    print(f"总迭代次数: {len(times)}")
    print(f"第一次运行（含编译）: {times[0]:.4f} 秒")
    if len(times) > 1:
        avg_time = sum(times[1:]) / len(times[1:])
        print(f"后续运行平均时间: {avg_time:.4f} 秒")
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