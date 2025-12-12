#!/usr/bin/env python3
"""
CogVideoX 三阶段生成 - 阶段2：Transformer (Denoising)

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

from diffusers import CogVideoXPipeline

from utils import (
    MODEL_NAME,
    WIDTH, HEIGHT, FRAMES, FPS, NUM_STEPS, GUIDANCE_SCALE,
    BQSIZE, BKVSIZE, BKVCOMPUTESIZE, BKVCOMPUTEINSIZE,
    USE_K_SMOOTH, USE_CUSTOM_ATTENTION, LOG2_E,
    USE_DP, SP_NUM, USE_FSDP,
    setup_pytree_registrations,
    load_embeddings_from_safetensors,
    save_latents_to_safetensors,
    save_generation_config,
    load_generation_config,
    get_default_paths,
)


# === Splash Attention 实现 ===

def _sdpa_reference(query, key, value, attn_mask=None, dropout_p=0.0,
                    is_causal=False, scale=None, enable_gqa=False):
    """
    Scaled Dot-Product Attention 参考实现
    
    用于在不支持 Splash attention 时作为回退方案
    """
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
    """
    TPU Splash Attention 实现
    
    使用 JAX 的 Splash Attention 在 TPU 上高效计算注意力
    支持可选的局部窗口注意力和 K-smooth
    """
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

            # 填充到块大小的倍数
            q_3d_padded, q_orig_len = pad_to_multiple(q_3d, bqsize, axis=1)
            k_3d_padded, k_orig_len = pad_to_multiple(k_3d, bkvsize, axis=1)
            v_3d_padded, v_orig_len = pad_to_multiple(v_3d, bkvsize, axis=1)

            padded_q_seq_len = q_3d_padded.shape[1]
            padded_kv_seq_len = k_3d_padded.shape[1]

            # 创建注意力掩码
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

            block_sizes = splash_attention.BlockSizes(
                block_q=min(bqsize, padded_q_seq_len),
                block_kv=min(bkvsize, padded_kv_seq_len),
                block_kv_compute=min(bkvcomputesize, padded_kv_seq_len),
            )
            
            splash_kernel = splash_attention.make_splash_mha(
                mask=mask, block_sizes=block_sizes, head_shards=1, q_seq_shards=1
            )
            out = splash_kernel(q_3d_padded, k_3d_padded, v_3d_padded)
            return out[:, :q_orig_len, ...]

        vmapped_kernel = jax.vmap(kernel_3d, in_axes=(0, 0, 0), out_axes=0)
        return vmapped_kernel(q, k, v)

    # 根据设备数量和头数确定分片策略
    if num_heads < mesh.size:
        q_partition_spec = P()
        kv_partition_spec = P()
    else:
        # 根据 query 和 key 的序列长度判断是自注意力还是交叉注意力
        if query.shape[2] == key.shape[2]:  # 自注意力
            q_partition_spec = P('dp', 'tp', 'sp', None)
            kv_partition_spec = P('dp', 'tp', None, None)
        else:  # 交叉注意力
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
    
    # 添加输出 sharding constraint（与 Wan2.1 一致）
    out = jax.lax.with_sharding_constraint(out, P('dp', None, ('tp', 'sp'), None))
    
    return out


def _tpu_custom_attention(query, key, value, env, scale=None, is_causal=False,
                          window_size=None, bqsize=BQSIZE, bkvsize=BKVSIZE,
                          bkvcomputesize=BKVCOMPUTESIZE, bkvcomputeinsize=BKVCOMPUTEINSIZE):
    """
    TPU Custom Splash Attention 实现（使用 exp2 优化）
    
    使用自定义的 Pallas kernel，用 exp2 替代 exp 以获得更好的 TPU VPU 硬件利用率。
    关键优化：Query 预乘 LOG2_E，使 exp(x) 变为 exp2(x * LOG2_E)
    """
    mesh = getattr(env, '_mesh', None) or env.param.mesh
    num_heads = query.shape[1]

    def _attention_on_slices(q, k, v):
        # 计算缩放因子并应用 LOG2_E 预乘
        scale_factor = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
        q = q * (scale_factor * LOG2_E)

        def pad_to_multiple(x, multiple, axis):
            seq_len = x.shape[axis]
            pad_len = (multiple - seq_len % multiple) % multiple
            if pad_len == 0:
                return x, seq_len
            pad_width = [(0, 0)] * x.ndim
            pad_width[axis] = (0, pad_len)
            return jnp.pad(x, pad_width), seq_len

        def kernel_3d(q_3d, k_3d, v_3d):
            q_orig_len = q_3d.shape[1]

            # 填充到块大小的倍数
            q_3d_padded, _ = pad_to_multiple(q_3d, bqsize, axis=1)
            k_3d_padded, _ = pad_to_multiple(k_3d, bkvsize, axis=1)
            v_3d_padded, _ = pad_to_multiple(v_3d, bkvsize, axis=1)

            padded_q_seq_len = q_3d_padded.shape[1]
            padded_kv_seq_len = k_3d_padded.shape[1]

            block_sizes = custom_splash_attention._BlockSizes(
                block_q=min(bqsize, padded_q_seq_len),
                block_kv=min(bkvsize, padded_kv_seq_len),
                block_kv_compute=min(bkvcomputesize, padded_kv_seq_len),
            )
            
            splash_kernel = custom_splash_attention.make_splash_mha(
                block_sizes=block_sizes,
                bkv_compute_in=bkvcomputeinsize
            )
            out = splash_kernel(
                q_3d_padded.astype(jnp.float32),
                k_3d_padded.astype(jnp.float32),
                v_3d_padded.astype(jnp.float32)
            ).astype(q_3d_padded.dtype)
            
            # 调整输出维度
            out = jnp.swapaxes(out, 1, 2)
            return out[:, :q_orig_len, ...]

        vmapped_kernel = jax.vmap(kernel_3d, in_axes=(0, 0, 0), out_axes=0)
        return vmapped_kernel(q, k, v)

    # 根据设备数量和头数确定分片策略
    if num_heads < mesh.size:
        q_partition_spec = P()
        kv_partition_spec = P()
    else:
        if query.shape[2] == key.shape[2]:  # 自注意力
            q_partition_spec = P('dp', 'tp', 'sp', None)
            kv_partition_spec = P('dp', 'tp', None, None)
        else:  # 交叉注意力
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
    
    # 添加输出 sharding constraint（与 Wan2.1 一致）
    out = jax.lax.with_sharding_constraint(out, P('dp', None, ('tp', 'sp'), None))
    
    return out


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                  is_causal=False, scale=None, enable_gqa=False,
                                  env=None, window_size=None, use_k_smooth=USE_K_SMOOTH,
                                  use_custom_attention=USE_CUSTOM_ATTENTION):
    """
    Scaled Dot-Product Attention 封装函数
    
    根据环境配置选择使用 TPU Splash Attention 或参考实现
    """
    use_splash = getattr(env.config, 'use_tpu_splash_attention', False) if env else False
    
    if use_splash and env is not None:
        jquery, jkey, jvalue = env.t2j_iso((query, key, value))
        
        # 可选的 K-smooth 处理
        if use_k_smooth:
            key_mean = jnp.mean(jkey, axis=2, keepdims=True)
            jkey = jkey - key_mean
        
        # 选择使用标准 splash attention 还是 custom splash attention
        if use_custom_attention:
            res = _tpu_custom_attention(
                jquery, jkey, jvalue, env,
                scale=scale, is_causal=is_causal, window_size=window_size
            )
        else:
            res = _tpu_splash_attention(
                jquery, jkey, jvalue, env,
                scale=scale, is_causal=is_causal, window_size=window_size
            )
        return env.j2t_iso(res)

    return _sdpa_reference(query, key, value, attn_mask, dropout_p, is_causal,
                           scale, enable_gqa)


# === 权重分片策略 ===

# Transformer sharding策略 - FSDP模式（默认）
# 注意：所有模式都以 .weight$ 结尾，这样不会匹配到 bias 等1维参数
transformer_shardings_fsdp = {
    r'.*\.to_q\.weight$': (None, ('tp', 'sp')),
    r'.*\.to_k\.weight$': (None, ('tp', 'sp')),
    r'.*\.to_v\.weight$': (None, ('tp', 'sp')),
    r'.*\.to_out.*\.weight$': (('tp', 'sp'), None),
    r'.*\.ff\.net\.0\.weight$': (None, ('tp', 'sp')),
    r'.*\.ff\.net\.2\.weight$': (('tp', 'sp'), None),
}

# Transformer sharding策略 - Tensor Parallel模式
transformer_shardings_tp = {
    r'.*\.to_q\.weight$': (('tp', 'sp'), None),
    r'.*\.to_k\.weight$': (('tp', 'sp'), None),
    r'.*\.to_v\.weight$': (('tp', 'sp'), None),
    r'.*\.to_out.*\.weight$': (None, ('tp', 'sp')),
    r'.*\.ff\.net\.0\.weight$': (('tp', 'sp'), None),
    r'.*\.ff\.net\.2\.weight$': (None, ('tp', 'sp')),
}

# Text Encoder (T5) sharding策略
text_encoder_shardings = {
    r'shared\.weight$': (('tp', 'dp', 'sp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.SelfAttention\.q\.weight$': (('tp', 'dp', 'sp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.SelfAttention\.k\.weight$': (('tp', 'dp', 'sp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.SelfAttention\.v\.weight$': (('tp', 'dp', 'sp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.SelfAttention\.o\.weight$': (None, ('tp', 'dp', 'sp')),
    r'encoder\.block\.\d+\.layer\.\d+\.DenseReluDense\.wi_0\.weight$': (('tp', 'dp', 'sp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.DenseReluDense\.wi_1\.weight$': (('tp', 'dp', 'sp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.DenseReluDense\.wo\.weight$': (None, ('tp', 'dp', 'sp')),
}


def shard_weight_dict(weight_dict, sharding_dict, mesh):
    """Apply sharding to weights based on pattern matching."""
    result = {}
    for k, v in weight_dict.items():
        matched = False
        for target, sharding in sharding_dict.items():
            if re.fullmatch(target, k) is not None:
                v.apply_jax_(jax.device_put, NamedSharding(mesh, P(*sharding)))
                matched = True
                break
        
        if not matched:
            v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
        
        result[k] = v
    return result


# === Pipeline 设置 ===

def setup_pipeline_for_transformer_only(pipe, mesh, env, window_size=None,
                                         use_k_smooth=USE_K_SMOOTH,
                                         use_custom_attention=USE_CUSTOM_ATTENTION,
                                         use_fsdp=USE_FSDP):
    """
    设置 Pipeline 仅用于 Transformer 推理（不包含 VAE）
    """
    print("\n配置 Pipeline 以使用 JAX 和 Splash Attention（仅 Transformer）...")

    # 注册自定义的 Scaled Dot-Product Attention
    attention_type = "Custom Splash Attention (exp2)" if use_custom_attention else "Standard Splash Attention"
    print(f"- 注册 {attention_type}（窗口大小: {window_size}）...")
    custom_attention = functools.partial(
        scaled_dot_product_attention,
        env=env,
        window_size=window_size,
        use_k_smooth=use_k_smooth,
        use_custom_attention=use_custom_attention,
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

    def _move_module(module):
        with jax.default_device('cpu'):
            state_dict = module.state_dict()
            state_dict = env.to_xla(state_dict)
            module.load_state_dict(state_dict, assign=True)

    # Move Transformer to XLA
    print("- 将 Transformer 移到 XLA...")
    _move_module(pipe.transformer)

    # Move scheduler parameters to JAX
    print("- 将 Scheduler 参数移到 JAX...")
    for k, v in pipe.scheduler.__dict__.items():
        if isinstance(v, torch.Tensor):
            setattr(pipe.scheduler, k, v.to('jax'))

    # Compile Transformer
    print("- 编译 Transformer...")
    options = torchax.CompileOptions(
        jax_jit_kwargs={'static_argnames': ('return_dict',)}
    )
    pipe.transformer = torchax.compile(pipe.transformer, options)

    # Apply sharding
    print("- 对 Transformer 进行权重分片...")
    transformer_shardings = transformer_shardings_fsdp if use_fsdp else transformer_shardings_tp
    pipe.transformer.params = shard_weight_dict(
        pipe.transformer.params, transformer_shardings, mesh
    )
    pipe.transformer.buffers = shard_weight_dict(
        pipe.transformer.buffers, transformer_shardings, mesh
    )

    # 确保所有权重已分片完成
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

    print("Pipeline 配置完成（仅 Transformer）")
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
    }
    
    with ctx:
        with mesh, env:
            # 使用自定义的 callback 来追踪每一步的时间
            step_start_time = [None]
            
            # tqdm 进度条
            progress_bar = tqdm(
                total=num_steps,
                desc=desc,
                ncols=130,
            )
            
            def step_callback(pipe, step_index, timestep, callback_kwargs):
                """每一步结束后的回调，用于更新进度条"""
                nonlocal step_times
                
                # 等待计算完成
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
                
                step_start_time[0] = time.perf_counter()
                
                return callback_kwargs
            
            # 记录第一步的开始时间
            step_start_time[0] = time.perf_counter()
            
            # 添加 callback
            gen_kwargs['callback_on_step_end'] = step_callback
            
            result = pipe(**gen_kwargs)
            jax.effects_barrier()
            latents = result.frames  # output_type='latent' 时返回 latents
            
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
        avg_time_ex_first = sum(step_times[1:]) / len(step_times[1:])
        print(f"  平均每步时间（排除首步）: {avg_time_ex_first:.2f}s")

    # 转换为可保存的格式
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
    print(f"  Latents range: [{torch_latents.min():.4f}, {torch_latents.max():.4f}]")

    return torch_latents, elapsed


def main():
    parser = argparse.ArgumentParser(
        description='CogVideoX 阶段2：Transformer (Denoising)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 基本用法（使用阶段1的输出）
  python stage2_transformer.py
  
  # 指定输入目录
  python stage2_transformer.py --input_dir ./my_outputs
  
  # 覆盖配置参数
  python stage2_transformer.py --num_inference_steps 20
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

    # 视频生成参数（阶段2的主要参数）
    parser.add_argument('--num_inference_steps', type=int, default=NUM_STEPS, help='推理步数')
    parser.add_argument('--guidance_scale', type=float, default=GUIDANCE_SCALE, help='CFG 引导尺度')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--height', type=int, default=HEIGHT, help='视频高度')
    parser.add_argument('--width', type=int, default=WIDTH, help='视频宽度')
    parser.add_argument('--frames', type=int, default=FRAMES, help='视频帧数')
    parser.add_argument('--fps', type=int, default=FPS, help='视频帧率（保存到配置供阶段3使用）')
    parser.add_argument('--warmup_steps', type=int, default=2,
                        help='预热步数（0=不预热）')

    # Attention 参数
    parser.add_argument('--window_size', type=int, nargs=2, default=None, help='Attention window size')
    parser.add_argument('--bqsize', type=int, default=BQSIZE, help='Query block size')
    parser.add_argument('--bkvsize', type=int, default=BKVSIZE, help='KV block size')
    parser.add_argument('--bkvcomputesize', type=int, default=BKVCOMPUTESIZE, help='KV compute block size')
    parser.add_argument('--use_k_smooth', action='store_true', default=USE_K_SMOOTH, help='Use K smoothing')
    parser.add_argument('--use_custom_attention', action='store_true', default=USE_CUSTOM_ATTENTION, help='Use custom exp2 attention')

    # Sharding 参数
    parser.add_argument('--use_dp', action='store_true', default=USE_DP, help='Use data parallelism')
    parser.add_argument('--sp_num', type=int, default=SP_NUM, help='Sequence parallelism number')
    parser.add_argument('--use_fsdp', action='store_true', default=USE_FSDP, help='Use FSDP mode')

    # 其他参数
    parser.add_argument('--model_id', type=str, default=None, help='Override model ID from stage1')
    parser.add_argument('--profile', action='store_true', default=False, help='Run profiler')
    parser.add_argument('--profiler_output_dir', type=str, default='/dev/shm/jax-trace',
                        help='Profiler 输出目录')

    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    input_paths = get_default_paths(args.input_dir)
    output_paths = get_default_paths(output_dir)

    # 设置 JAX 配置
    jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

    warnings.filterwarnings('ignore', message='.*dtype.*int64.*truncated to dtype int32.*')
    logging.getLogger().setLevel(logging.ERROR)

    torch.set_default_dtype(torch.bfloat16)
    setup_pytree_registrations()

    print(f"\n{'='*60}")
    print("CogVideoX 阶段2：Transformer (Denoising)")
    print(f"{'='*60}")

    # 加载阶段1配置
    print(f"\n加载阶段1配置: {input_paths['config']}")
    config = load_generation_config(input_paths['config'])

    # 设置视频生成参数（从命令行参数获取，使用默认值）
    config['num_inference_steps'] = args.num_inference_steps
    config['guidance_scale'] = args.guidance_scale
    config['seed'] = args.seed
    config['height'] = args.height
    config['width'] = args.width
    config['frames'] = args.frames
    config['fps'] = args.fps
    if args.model_id is not None:
        config['model_id'] = args.model_id
    
    config['window_size'] = args.window_size

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

    # 创建 mesh
    tp_dim, dp_dim, sp_dim = len(jax.devices()), 1, 1
    if args.use_dp:
        tp_dim //= 2
        dp_dim = 2
    if args.sp_num > 1:
        tp_dim //= args.sp_num
        sp_dim = args.sp_num

    print(f"\nMesh 维度: dp_dim={dp_dim}, sp_dim={sp_dim}, tp_dim={tp_dim}")
    print(f"总设备数: {len(jax.devices())}")

    # mesh 顺序：('dp', 'sp', 'tp') - 与 Wan2.1 一致
    mesh_devices = mesh_utils.create_device_mesh(
        (dp_dim, sp_dim, tp_dim), allow_split_physical_axes=True
    )
    mesh = Mesh(mesh_devices, ('dp', 'sp', 'tp'))

    # 配置 env - 直接设置属性（不使用 hasattr 检查，因为属性可能不存在需要创建）
    env._mesh = mesh
    env._initial_content.mesh = mesh
    env.config.use_tpu_splash_attention = True
    
    print(f"✓ env.config.use_tpu_splash_attention = {env.config.use_tpu_splash_attention}")

    # 加载 Pipeline（仅 Transformer）
    model_id = config.get('model_id', MODEL_NAME)
    print(f"\n加载模型: {model_id}")
    print("（注意：仅加载 Transformer 组件）")

    # 临时禁用 torchax 加载 PyTorch 组件
    torchax.disable_globally()
    try:
        pipe = CogVideoXPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )
    finally:
        torchax.enable_globally()

    # 配置 Pipeline
    pipe = setup_pipeline_for_transformer_only(
        pipe, mesh, env,
        window_size=config.get('window_size'),
        use_k_smooth=args.use_k_smooth,
        use_custom_attention=args.use_custom_attention,
        use_fsdp=args.use_fsdp
    )

    # 将 embeddings 转换为 XLA tensor
    print("\n- 将 embeddings 转换为 XLA tensor...")
    with env:
        for key in prompt_embeds_dict:
            prompt_embeds_dict[key] = prompt_embeds_dict[key].to('jax')
    print("  ✓ 所有 embeddings 已转换")

    # 运行推理
    if args.profile:
        print(f"\n[Profiler] 启用 JAX Profiler，输出目录: {args.profiler_output_dir}")
        profiler_context = jax.profiler.trace(
            args.profiler_output_dir,
            create_perfetto_link=False
        )
    else:
        profiler_context = nullcontext()
    
    with profiler_context:
        latents, elapsed = run_transformer_inference(
            pipe, prompt_embeds_dict, config, mesh, env,
            warmup_steps=args.warmup_steps
        )

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

    # 更新配置（添加阶段2参数）
    config['bqsize'] = args.bqsize
    config['bkvsize'] = args.bkvsize
    config['bkvcomputesize'] = args.bkvcomputesize
    save_generation_config(config, output_paths['config'])

    # 打印性能统计
    print(f"\n=== 性能统计 ===")
    print(f"推理时间: {elapsed:.4f} 秒")
    print(f"Block sizes: BQSIZE={args.bqsize}, BKVSIZE={args.bkvsize}, BKVCOMPUTESIZE={args.bkvcomputesize}")

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