#!/usr/bin/env python3
"""
HunyuanVideo-1.5 三阶段生成 - 阶段2：Transformer (DiT)

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
import time
import re
import math
import functools
import argparse
import warnings
import logging
import numpy as np
import jax
import jax.numpy as jnp
import torch
import torchax
from torchax.ops import ops_registry
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
from jax.experimental.pallas.ops.tpu import splash_attention
from jax.experimental.shard_map import shard_map
from jax.experimental import mesh_utils
from diffusers import HunyuanVideo15Pipeline

from utils import (
    MODEL_NAME,
    BQSIZE, BKVSIZE, BKVCOMPUTESIZE,
    USE_DP, SP_NUM, USE_TP,
    setup_pytree_registrations,
    load_embeddings_from_safetensors,
    save_latents_to_safetensors,
    load_generation_config,
    get_default_paths,
)


# === Splash Attention 实现 ===

# 保存原始的 SDPA 实现
_ORIGINAL_SDPA = None


def _is_xla_tensor(tensor):
    """检测 tensor 是否是 XLA/torchax tensor"""
    if tensor is None:
        return False
    if hasattr(tensor, '_elem'):
        return True
    if hasattr(tensor, 'device'):
        device_str = str(tensor.device)
        if 'jax' in device_str or 'xla' in device_str:
            return True
    return False


def _sdpa_reference(
    query, key, value,
    attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False,
) -> torch.Tensor:
    """Scaled Dot-Product Attention 参考实现"""
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    
    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)
    
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    
    if is_causal:
        assert attn_mask is None
        causal_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_weight = attn_weight.masked_fill(causal_mask.logical_not(), float("-inf"))
    
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_weight = attn_weight.masked_fill(attn_mask.logical_not(), float("-inf"))
        else:
            attn_weight = attn_weight + attn_mask
    
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = attn_weight.masked_fill(torch.isnan(attn_weight), 0.0)
    
    if dropout_p > 0:
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    
    return attn_weight @ value


def _tpu_splash_attention(query, key, value, mesh, scale=None, is_causal=False, window_size=None):
    """TPU Splash Attention 实现"""
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

            q_3d_padded, q_orig_len = pad_to_multiple(q_3d, BQSIZE, axis=1)
            k_3d_padded, k_orig_len = pad_to_multiple(k_3d, BKVSIZE, axis=1)
            v_3d_padded, v_orig_len = pad_to_multiple(v_3d, BKVSIZE, axis=1)

            padded_q_seq_len = q_3d_padded.shape[1]
            padded_kv_seq_len = k_3d_padded.shape[1]

            if window_size is not None:
                mask_class = functools.partial(splash_attention.LocalMask, window_size=window_size, offset=0)
            else:
                mask_class = splash_attention.FullMask

            mask = splash_attention.MultiHeadMask(
                [mask_class((padded_q_seq_len, padded_kv_seq_len)) for _ in range(num_heads_on_device)]
            )

            block_sizes = splash_attention.BlockSizes(
                block_q=min(BQSIZE, padded_q_seq_len),
                block_kv=min(BKVSIZE, padded_kv_seq_len),
                block_kv_compute=min(BKVCOMPUTESIZE, padded_kv_seq_len),
            )
            
            splash_kernel = splash_attention.make_splash_mha(
                mask=mask, block_sizes=block_sizes, head_shards=1, q_seq_shards=1
            )
            out = splash_kernel(q_3d_padded, k_3d_padded, v_3d_padded)
            
            return out[:, :q_orig_len, ...]

        vmapped_kernel = jax.vmap(kernel_3d, in_axes=(0, 0, 0), out_axes=0)
        return vmapped_kernel(q, k, v)

    if num_heads < mesh.size:
        q_partition_spec = P()
        kv_partition_spec = P()
    else:
        if query.shape[2] == key.shape[2]:
            q_partition_spec = P('dp', 'tp', 'sp', None)
            kv_partition_spec = P('dp', 'tp', None, None)
        else:
            q_partition_spec = P('dp', None, ('tp', 'sp'), None)
            kv_partition_spec = P('dp', None, None, None)

    sharded_fn = shard_map(
        _attention_on_slices,
        mesh=mesh,
        in_specs=(q_partition_spec, kv_partition_spec, kv_partition_spec),
        out_specs=q_partition_spec,
        check_rep=False,
    )
    return sharded_fn(query, key, value)


def scaled_dot_product_attention(
    query, key, value,
    attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False,
    env=None, window_size=None,
) -> torch.Tensor:
    """Scaled Dot-Product Attention 封装函数"""
    global _ORIGINAL_SDPA
    
    if not _is_xla_tensor(query):
        if _ORIGINAL_SDPA is not None:
            return _ORIGINAL_SDPA(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)
        return _sdpa_reference(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)
    
    if attn_mask is not None:
        return _sdpa_reference(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)
    
    if env is not None and hasattr(env.config, 'use_tpu_splash_attention') and env.config.use_tpu_splash_attention:
        jquery, jkey, jvalue = env.t2j_iso((query, key, value))
        res = _tpu_splash_attention(jquery, jkey, jvalue, env._mesh, scale=scale, is_causal=is_causal, window_size=window_size)
        return env.j2t_iso(res)
    
    return _sdpa_reference(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)


# === 权重分片策略 ===

transformer_shardings_tp = {
    r'.*\.img_attn_q\.weight$': (('tp', 'sp'), None),
    r'.*\.img_attn_k\.weight$': (('tp', 'sp'), None),
    r'.*\.img_attn_v\.weight$': (('tp', 'sp'), None),
    r'.*\.img_attn_proj\.weight$': (None, ('tp', 'sp')),
    r'.*\.txt_attn_q\.weight$': (('tp', 'sp'), None),
    r'.*\.txt_attn_k\.weight$': (('tp', 'sp'), None),
    r'.*\.txt_attn_v\.weight$': (('tp', 'sp'), None),
    r'.*\.txt_attn_proj\.weight$': (None, ('tp', 'sp')),
    r'.*\.img_mlp\.fc1\.weight$': (('tp', 'sp'), None),
    r'.*\.img_mlp\.fc2\.weight$': (None, ('tp', 'sp')),
    r'.*\.txt_mlp\.fc1\.weight$': (('tp', 'sp'), None),
    r'.*\.txt_mlp\.fc2\.weight$': (None, ('tp', 'sp')),
    r'.*\.linear1_q\.weight$': (('tp', 'sp'), None),
    r'.*\.linear1_k\.weight$': (('tp', 'sp'), None),
    r'.*\.linear1_v\.weight$': (('tp', 'sp'), None),
    r'.*\.linear1_mlp\.weight$': (('tp', 'sp'), None),
    r'.*\.linear2\.linear\.weight$': (None, ('tp', 'sp')),
    r'.*\.txt_in\..*\.to_q\.weight$': (('tp', 'sp'), None),
    r'.*\.txt_in\..*\.to_k\.weight$': (('tp', 'sp'), None),
    r'.*\.txt_in\..*\.to_v\.weight$': (('tp', 'sp'), None),
    r'.*\.txt_in\..*\.to_out\.0\.weight$': (None, ('tp', 'sp')),
    r'.*\.byt5_in\.fc1\.weight$': (('tp', 'sp'), None),
    r'.*\.byt5_in\.fc3\.weight$': (('tp', 'sp'), None),
    r'.*\.byt5_in\.fc2\.weight$': (None, ('tp', 'sp')),
    r'.*\.byt5_in\.fc4\.weight$': (None, ('tp', 'sp')),
    r'.*\.to_q\.weight$': (('tp', 'sp'), None),
    r'.*\.to_k\.weight$': (('tp', 'sp'), None),
    r'.*\.to_v\.weight$': (('tp', 'sp'), None),
    r'.*\.to_out\.0\.weight$': (None, ('tp', 'sp')),
    r'.*\.ff\.net\.0\.proj\.weight$': (('tp', 'sp'), None),
    r'.*\.ff\.net\.2\.weight$': (None, ('tp', 'sp')),
    r'.*\.proj_in\.weight$': (('tp', 'sp'), None),
    r'.*\.proj_out\.weight$': (None, ('tp', 'sp')),
}

transformer_shardings_fsdp = {
    r'.*\.img_attn_q\.weight$': (None, ('tp', 'sp')),
    r'.*\.img_attn_k\.weight$': (None, ('tp', 'sp')),
    r'.*\.img_attn_v\.weight$': (None, ('tp', 'sp')),
    r'.*\.img_attn_proj\.weight$': (('tp', 'sp'), None),
    r'.*\.txt_attn_q\.weight$': (None, ('tp', 'sp')),
    r'.*\.txt_attn_k\.weight$': (None, ('tp', 'sp')),
    r'.*\.txt_attn_v\.weight$': (None, ('tp', 'sp')),
    r'.*\.txt_attn_proj\.weight$': (('tp', 'sp'), None),
    r'.*\.img_mlp\.fc1\.weight$': (None, ('tp', 'sp')),
    r'.*\.img_mlp\.fc2\.weight$': (('tp', 'sp'), None),
    r'.*\.txt_mlp\.fc1\.weight$': (None, ('tp', 'sp')),
    r'.*\.txt_mlp\.fc2\.weight$': (('tp', 'sp'), None),
    r'.*\.linear1_q\.weight$': (None, ('tp', 'sp')),
    r'.*\.linear1_k\.weight$': (None, ('tp', 'sp')),
    r'.*\.linear1_v\.weight$': (None, ('tp', 'sp')),
    r'.*\.linear1_mlp\.weight$': (None, ('tp', 'sp')),
    r'.*\.linear2\.linear\.weight$': (('tp', 'sp'), None),
    r'.*\.txt_in\..*\.to_q\.weight$': (None, ('tp', 'sp')),
    r'.*\.txt_in\..*\.to_k\.weight$': (None, ('tp', 'sp')),
    r'.*\.txt_in\..*\.to_v\.weight$': (None, ('tp', 'sp')),
    r'.*\.txt_in\..*\.to_out\.0\.weight$': (('tp', 'sp'), None),
    r'.*\.byt5_in\.fc1\.weight$': (None, ('tp', 'sp')),
    r'.*\.byt5_in\.fc2\.weight$': (('tp', 'sp'), None),
    r'.*\.byt5_in\.fc3\.weight$': (None, ('tp', 'sp')),
    r'.*\.byt5_in\.fc4\.weight$': (('tp', 'sp'), None),
    r'.*\.to_q\.weight$': (None, ('tp', 'sp')),
    r'.*\.to_k\.weight$': (None, ('tp', 'sp')),
    r'.*\.to_v\.weight$': (None, ('tp', 'sp')),
    r'.*\.to_out\.0\.weight$': (('tp', 'sp'), None),
    r'.*\.ff\.net\.0\.proj\.weight$': (None, ('tp', 'sp')),
    r'.*\.ff\.net\.2\.weight$': (('tp', 'sp'), None),
    r'.*\.proj_in\.weight$': (None, ('tp', 'sp')),
    r'.*\.proj_out\.weight$': (('tp', 'sp'), None),
}


def shard_weights(mesh, weights, sharding_dict):
    """对模型权重进行分片"""
    result = {}
    matched_count = 0
    unmatched_count = 0
    
    for k, v in weights.items():
        matched = False
        for target, sharding in sharding_dict.items():
            if re.fullmatch(target, k) is not None:
                v.apply_jax_(jax.device_put, NamedSharding(mesh, P(*sharding)))
                matched = True
                matched_count += 1
                break
        
        if not matched:
            v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
            unmatched_count += 1
        
        result[k] = v
    
    print(f"  权重分片完成: {matched_count} 个匹配规则, {unmatched_count} 个复制到所有设备")
    return result


def shard_weights_transformer(mesh, weights, use_tp=True):
    """对 Transformer 模型的权重进行分片"""
    sharding_dict = transformer_shardings_tp if use_tp else transformer_shardings_fsdp
    return shard_weights(mesh, weights, sharding_dict)


# === Pipeline 设置 ===

def setup_pipeline_for_transformer_only(pipe, window_size=None):
    """
    设置 Pipeline 仅用于 Transformer 推理（不包含 VAE）
    """
    print("\n配置 Pipeline 以使用 JAX 和 Splash Attention（仅 Transformer）...")

    tp_dim, dp_dim, sp_dim = jax.device_count(), 1, 1
    if USE_DP:
        tp_dim //= 2
        dp_dim = 2
    
    if SP_NUM > 1:
        tp_dim //= SP_NUM
        sp_dim = SP_NUM
    
    print(f"  Mesh 维度: tp_dim={tp_dim}, dp_dim={dp_dim}, sp_dim={sp_dim}")
    print(f"  总设备数: {jax.device_count()}")
    
    mesh_devices = mesh_utils.create_device_mesh((tp_dim, dp_dim, sp_dim), allow_split_physical_axes=True)
    mesh = Mesh(mesh_devices, ('tp', 'dp', 'sp'))
    
    env = torchax.default_env()
    env._mesh = mesh
    env.config.use_tpu_splash_attention = True

    global _ORIGINAL_SDPA
    _ORIGINAL_SDPA = torch.nn.functional.scaled_dot_product_attention
    print(f"- 保存原始 SDPA 实现")
    
    custom_attention = functools.partial(
        scaled_dot_product_attention,
        env=env,
        window_size=window_size
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
    
    def _move_scheduler_to_jax(scheduler):
        print("- 将 scheduler 参数移动到 JAX 设备...")
        for k, v in scheduler.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(scheduler, k, v.to('jax'))

    def _move_module_to_xla(module, name="module"):
        print(f"- 将 {name} 移到 XLA...")
        with jax.default_device('cpu'):
            state_dict = module.state_dict()
            state_dict = env.to_xla(state_dict)
            module.load_state_dict(state_dict, assign=True)
    
    with env:
        _move_scheduler_to_jax(pipe.scheduler)
        
        _move_module_to_xla(pipe.transformer, "Transformer")
        print("- 对 Transformer 进行权重分片...")
        transformer_weights = shard_weights_transformer(mesh, pipe.transformer.state_dict(), use_tp=USE_TP)
        pipe.transformer.load_state_dict(transformer_weights, assign=True, strict=False)
        torchax.interop.call_jax(jax.block_until_ready, transformer_weights)
        
        # 删除不需要的组件以节省内存
        print("- 删除 VAE 以节省内存（阶段2不需要）...")
        if hasattr(pipe, 'vae') and pipe.vae is not None:
            del pipe.vae
            pipe.vae = None
        
        print("- Text Encoder 不需要在阶段2使用")
        
        print("- 编译 Transformer...")
        pipe.transformer = torchax.compile(
            pipe.transformer,
            torchax.CompileOptions(
                jax_jit_kwargs={'static_argnames': ('return_dict', 'is_t2v')}
            )
        )
        
        print("- Monkeypatch _execution_device...")
        def _jax_execution_device(self):
            return torch.device('jax')
        type(pipe)._execution_device = property(_jax_execution_device)
    
    print("Pipeline 配置完成（仅 Transformer）")
    return pipe, env, mesh


def run_transformer_inference(pipe, prompt_embeds_dict, config, env):
    """
    运行 Transformer 推理生成 latents
    
    使用 output_type='latent' 让 pipeline 跳过 VAE 解码
    """
    print(f"\n=== 阶段2：Transformer 推理 ===")
    print(f"推理步数: {config['num_inference_steps']}")
    print(f"帧数: {config['num_frames']}")
    print(f"引导尺度: {config['guidance_scale']}")
    print(f"随机种子: {config['seed']}")
    if config.get('height'):
        print(f"分辨率: {config['height']}x{config['width']}")
    
    # 注意：HunyuanVideo15Pipeline 不直接接受 guidance_scale 参数
    # CFG 通过 guider 组件控制，后面会处理
    gen_kwargs = {
        'prompt': None,
        'negative_prompt': None,
        'prompt_embeds': prompt_embeds_dict['prompt_embeds'],
        'prompt_embeds_mask': prompt_embeds_dict['prompt_embeds_mask'],
        'prompt_embeds_2': prompt_embeds_dict['prompt_embeds_2'],
        'prompt_embeds_mask_2': prompt_embeds_dict['prompt_embeds_mask_2'],
        'negative_prompt_embeds': prompt_embeds_dict['negative_prompt_embeds'],
        'negative_prompt_embeds_mask': prompt_embeds_dict['negative_prompt_embeds_mask'],
        'negative_prompt_embeds_2': prompt_embeds_dict['negative_prompt_embeds_2'],
        'negative_prompt_embeds_mask_2': prompt_embeds_dict['negative_prompt_embeds_mask_2'],
        'num_frames': config['num_frames'],
        'num_inference_steps': config['num_inference_steps'],
        'output_type': 'latent',  # 关键：只返回 latents，不解码
    }
    
    if config.get('height'):
        gen_kwargs['height'] = config['height']
    if config.get('width'):
        gen_kwargs['width'] = config['width']
    
    generator = torch.Generator(device='cpu').manual_seed(config['seed'])
    gen_kwargs['generator'] = generator
    
    original_default_device = None
    try:
        original_default_device = torch.get_default_device()
    except Exception:
        pass
    
    torch.set_default_device('jax')
    
    print("\n开始 Transformer 推理...")
    start_time = time.perf_counter()
    
    result = pipe(**gen_kwargs)
    latents = result.frames  # output_type='latent' 时，frames 就是 latents
    
    elapsed = time.perf_counter() - start_time
    print(f"✓ Transformer 推理完成，耗时: {elapsed:.2f} 秒")
    
    # 恢复默认设备
    if original_default_device is not None:
        torch.set_default_device(original_default_device)
    else:
        torch.set_default_device('cpu')
    
    # 转换为可保存的格式
    if hasattr(latents, '_elem'):
        # torchax tensor -> JAX array -> numpy (float32) -> torch -> bfloat16
        # numpy 不支持 bfloat16，所以需要经过 float32 中转
        jax_latents = latents._elem
        is_bf16 = (jax_latents.dtype == jnp.bfloat16)
        if is_bf16:
            # 先转为 float32 再转 numpy
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
        description='HunyuanVideo-1.5 阶段2：Transformer (DiT)',
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
    
    # 阶段2 必需参数
    parser.add_argument(
        '--num_frames', type=int, default=49,
        help='Number of frames to generate (default: 49, ~2 seconds at 24fps). For longer videos, use --window_size.'
    )
    parser.add_argument(
        '--num_inference_steps', type=int, default=50,
        help='Number of inference steps (default: 50)'
    )
    parser.add_argument(
        '--guidance_scale', type=float, default=6.0,
        help='Guidance scale for CFG (default: 6.0)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--height', type=int, default=720,
        help='Video height (default: 720)'
    )
    parser.add_argument(
        '--width', type=int, default=1280,
        help='Video width (default: 1280)'
    )
    
    # 可选参数
    parser.add_argument('--model_id', type=str, default=None, help='Override model ID from stage1')
    parser.add_argument('--window_size', type=int, default=None, help='Attention window size (default: None for full attention)')
    parser.add_argument('--num_iterations', type=int, default=1, help='Number of benchmark iterations')
    
    args = parser.parse_args()
    
    output_dir = args.output_dir or args.input_dir
    input_paths = get_default_paths(args.input_dir)
    output_paths = get_default_paths(output_dir)
    
    # 设置 JAX 配置
    jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax_cache"))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

    warnings.filterwarnings('ignore', message='.*dtype.*int64.*truncated to dtype int32.*')
    logging.getLogger().setLevel(logging.ERROR)
    
    torch.set_default_dtype(torch.bfloat16)
    setup_pytree_registrations()
    
    print(f"\n{'='*60}")
    print("HunyuanVideo-1.5 阶段2：Transformer (DiT)")
    print(f"{'='*60}")
    
    # 加载阶段1配置（仅用于获取 model_id）
    print(f"\n加载阶段1配置: {input_paths['config']}")
    stage1_config = load_generation_config(input_paths['config'])
    
    # 构建阶段2配置（使用命令行参数）
    config = {
        'model_id': args.model_id or stage1_config.get('model_id', MODEL_NAME),
        'num_frames': args.num_frames,
        'num_inference_steps': args.num_inference_steps,
        'guidance_scale': args.guidance_scale,
        'seed': args.seed,
        'height': args.height,
        'width': args.width,
        'window_size': args.window_size,
    }
    
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
    
    # 加载 Pipeline（仅 Transformer）
    model_id = config.get('model_id', MODEL_NAME)
    print(f"\n加载模型: {model_id}")
    print("（注意：仅加载 Transformer 组件）")
    
    pipe = HunyuanVideo15Pipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )
    
    # 配置 JAX 环境
    pipe, env, mesh = setup_pipeline_for_transformer_only(
        pipe,
        window_size=config.get('window_size')
    )
    
    # 将 embeddings 转换为 XLA tensor
    print("\n- 将 embeddings 转换为 XLA tensor...")
    with env:
        for key in prompt_embeds_dict:
            prompt_embeds_dict[key] = prompt_embeds_dict[key].to('jax')
    print("  ✓ 所有 embeddings 已转换")
    
    # 运行推理
    times = []
    latents = None
    
    with mesh, env:
        for i in range(args.num_iterations):
            print(f"\n--- 迭代 {i+1}/{args.num_iterations} ---")
            latents, elapsed = run_transformer_inference(pipe, prompt_embeds_dict, config, env)
            times.append(elapsed)
            print(f"  耗时: {elapsed:.2f} 秒" + (" (包含 JIT 编译)" if i == 0 else ""))
    
    # 保存 latents
    print(f"\n保存 latents 到: {output_paths['latents']}")
    metadata = {
        'num_frames': str(config['num_frames']),
        'num_inference_steps': str(config['num_inference_steps']),
        'guidance_scale': str(config['guidance_scale']),
        'seed': str(config['seed']),
    }
    save_latents_to_safetensors(latents, output_paths['latents'], metadata)
    
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