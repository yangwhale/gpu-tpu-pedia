#!/usr/bin/env python3
"""
CogVideoX Text-to-Video Generation with TPU Splash Attention

本脚本提供在 TPU 上运行 CogVideoX 1.5B Text-to-Video 生成的完整实现，
使用 JAX/TorchAx 和优化的 Splash Attention。

结构:
1. Imports and Configuration
2. Helper Functions
3. Splash Attention Implementation
4. Sharding Strategies
5. Weight Sharding Functions
6. Pipeline Setup
7. Main Function

Optimizations:
- TorchAx VAE with compiled decoder for better memory efficiency
- TorchAx Transformer with FSDP sharding
- Custom Splash Attention with exp2 optimization
- 2D Mesh (dp, tp) for optimal performance
"""

import os
import warnings
import logging

# ============================================================================
# 环境配置和 Warning 过滤（必须在其他 import 之前）
# ============================================================================

os.environ.setdefault('JAX_MEMORY_DEBUG', '0')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 全局过滤 warnings
warnings.filterwarnings('ignore')

# 配置 logging
logging.getLogger('root').setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger('diffusers').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)

import time
import re
import math
import functools
import argparse
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import jax
import jax.numpy as jnp
import torch
from jax.tree_util import register_pytree_node
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
from jax.experimental.pallas.ops.tpu import splash_attention
from jax.experimental.shard_map import shard_map
from jax.experimental import mesh_utils

import torchax
from torchax.ops import ops_registry
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutputWithPastAndCrossAttentions
from transformers import T5EncoderModel, T5Tokenizer

# TorchAx components
from diffusers.pipelines.cogvideo.pipeline_cogvideox_torchax import CogVideoXPipeline
from diffusers.models.autoencoders.autoencoder_kl_cogvideox_torchax import AutoencoderKLCogVideoX
from diffusers.models.transformers.cogvideox_transformer_3d_torchax import CogVideoXTransformer3DModel
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers import CogVideoXDPMScheduler
from diffusers.utils import export_to_video

# Custom splash attention
import custom_splash_attention


# ============================================================================
# Configuration Constants
# ============================================================================

# Model Configuration
MODEL_ID = "zai-org/CogVideoX1.5-5B"

# Video Generation Settings
WIDTH = 1280
HEIGHT = 768
FRAMES = 49
FPS = 8
NUM_STEPS = 50

# Splash Attention Block Sizes
BQSIZE = 3328
BKVSIZE = 2816
BKVCOMPUTESIZE = 256
BKVCOMPUTEINSIZE = 256

# Attention Settings
USE_K_SMOOTH = True  # K smoothing for better numerical stability
USE_CUSTOM_ATTENTION = True  # Use exp2 optimized attention
LOG2_E = 1.44269504

# Mesh Sharding Configuration
USE_DP = True
USE_FSDP = True

# Profiler Output Path
PROFILE_OUT_PATH = "/dev/shm/jax-trace"


# ============================================================================
# Helper Functions
# ============================================================================

def setup_pytree_registrations():
    """Register PyTree nodes for JAX transformations."""
    def model_output_flatten(obj):
        return obj.to_tuple(), type(obj)

    def model_output_unflatten(aux, children):
        return aux(*children)

    register_pytree_node(
        BaseModelOutputWithPooling,
        model_output_flatten,
        model_output_unflatten
    )
    
    register_pytree_node(
        BaseModelOutputWithPastAndCrossAttentions,
        model_output_flatten,
        model_output_unflatten
    )
    
    register_pytree_node(
        DecoderOutput,
        model_output_flatten,
        model_output_unflatten
    )


def prepare_video_for_export(video, target_frames):
    """Prepare video tensor for export to file."""
    if isinstance(video, (list, tuple)):
        return [prepare_video_for_export(v, target_frames) for v in video]
    
    if isinstance(video, torch.Tensor):
        if video.dim() == 5:  # (B, C, T, H, W)
            video = video[0]
        if video.dim() == 4 and video.shape[0] != target_frames:  # (C, T, H, W)
            video = video.permute(1, 0, 2, 3)
        if video.shape[-1] != 3:
            video = video.permute(0, 2, 3, 1)
        if video.shape[-1] > 3:
            video = video[..., :3]
        
        video = video.cpu().numpy()
        video = np.clip(video, 0, 255).astype(np.uint8)
        
        if video.shape[-1] == 1:
            video = np.repeat(video, 3, axis=-1)
        return video
    
    if isinstance(video, np.ndarray):
        if video.shape[-1] == 1:
            video = np.repeat(video, 3, axis=-1)
        return video
    
    return video


# ============================================================================
# Splash Attention Implementation
# ============================================================================

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


def _tpu_custom_attention(query, key, value, mesh, scale=None):
    """TPU Custom Splash Attention with exp2 optimization."""
    def _attention_on_slices(q, k, v):
        scale_factor = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
        q = q * scale_factor * LOG2_E

        def pad_to_block_multiple(x, block_size, axis):
            seq_len = x.shape[axis]
            pad_len = (block_size - seq_len % block_size) % block_size
            if pad_len == 0:
                return x, seq_len
            pad_width = [(0, 0)] * x.ndim
            pad_width[axis] = (0, pad_len)
            return jnp.pad(x, pad_width), seq_len

        def kernel_3d(q_3d, k_3d, v_3d):
            q_3d_padded, q_orig_len = pad_to_block_multiple(q_3d, BQSIZE, axis=1)
            k_3d_padded, _ = pad_to_block_multiple(k_3d, BKVSIZE, axis=1)
            v_3d_padded, _ = pad_to_block_multiple(v_3d, BKVSIZE, axis=1)
            
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
    
    q_seq_len = query.shape[2]
    kv_seq_len = key.shape[2]
    
    # Self attention vs cross attention
    if q_seq_len == kv_seq_len:
        q_partition_spec = P(dp_mesh_key, remain_mesh_key, None, None)
        kv_partition_spec = P(dp_mesh_key, remain_mesh_key, None, None)
    else:
        remain_devices_prod = 1
        for d in remain_mesh_key:
            remain_devices_prod *= mesh.axis_sizes[mesh.axis_names.index(d)]
        
        def pad_to_multiple(x, multiple, axis):
            seq_len = x.shape[axis]
            pad_len = (multiple - seq_len % multiple) % multiple
            if pad_len == 0:
                return x, seq_len
            pad_width = [(0, 0)] * x.ndim
            pad_width[axis] = (0, pad_len)
            return jnp.pad(x, pad_width), seq_len
        
        if q_seq_len % remain_devices_prod != 0:
            query, _ = pad_to_multiple(query, remain_devices_prod, axis=2)
        
        q_partition_spec = P(dp_mesh_key, None, remain_mesh_key, None)
        kv_partition_spec = P(dp_mesh_key, None, None, None)

    sharded_fn = shard_map(
        _attention_on_slices,
        mesh=mesh,
        in_specs=(q_partition_spec, kv_partition_spec, kv_partition_spec),
        out_specs=q_partition_spec,
        check_rep=False,
    )
    
    query = jax.lax.with_sharding_constraint(query, P(dp_mesh_key, None, remain_mesh_key, None))
    key = jax.lax.with_sharding_constraint(key, P(dp_mesh_key, None, remain_mesh_key, None))
    value = jax.lax.with_sharding_constraint(value, P(dp_mesh_key, None, remain_mesh_key, None))
    
    out = sharded_fn(query, key, value)
    out = out[:, :, :q_seq_len, :]
    out = jax.lax.with_sharding_constraint(out, P(dp_mesh_key, None, remain_mesh_key, None))
    return out


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                  is_causal=False, scale=None, enable_gqa=False,
                                  env=None, mesh=None):
    """Wrapper for scaled dot-product attention with TPU Splash support."""
    if key.shape[2] > 20000:
        # TPU Splash Attention 的限制条件：
        # 这些特性在高性能 Splash Attention 路径中不支持
        assert attn_mask is None, "Splash Attention 不支持 attn_mask"
        assert dropout_p == 0.0, "Splash Attention 不支持 dropout"
        assert is_causal is False, "Splash Attention 不支持 causal mask（CogVideoX 使用双向注意力）"
        assert enable_gqa is False, "Splash Attention 不支持 GQA（分组查询注意力）"
        
        jquery, jkey, jvalue = env.t2j_iso((query, key, value))
        
        if USE_K_SMOOTH:
            key_mean = jnp.mean(jkey, axis=2, keepdims=True)
            jkey = jkey - key_mean
        
        res = _tpu_custom_attention(jquery, jkey, jvalue, mesh, scale=scale)
        return env.j2t_iso(res)

    return _sdpa_reference(query, key, value, attn_mask, dropout_p, is_causal,
                           scale, enable_gqa)


# ============================================================================
# Sharding Strategies
# ============================================================================

# Transformer sharding - 2D mesh (dp, tp)
transformer_shardings_fsdp = {
    r'.*\.to_q\.weight$': (None, 'tp'),
    r'.*\.to_k\.weight$': (None, 'tp'),
    r'.*\.to_v\.weight$': (None, 'tp'),
    r'.*\.to_out.*\.weight$': ('tp', None),
    r'.*\.ff\.net\.0\.weight$': (None, 'tp'),
    r'.*\.ff\.net\.2\.weight$': ('tp', None),
}

# Text Encoder (T5) sharding - 2D mesh (dp, tp)
text_encoder_shardings = {
    r'shared\.weight$': (('dp', 'tp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.SelfAttention\.q\.weight$': (('dp', 'tp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.SelfAttention\.k\.weight$': (('dp', 'tp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.SelfAttention\.v\.weight$': (('dp', 'tp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.SelfAttention\.o\.weight$': (None, ('dp', 'tp')),
    r'encoder\.block\.\d+\.layer\.\d+\.DenseReluDense\.wi_0\.weight$': (('dp', 'tp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.DenseReluDense\.wi_1\.weight$': (('dp', 'tp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.DenseReluDense\.wo\.weight$': (None, ('dp', 'tp')),
}


# ============================================================================
# Weight Sharding Functions
# ============================================================================

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


def shard_weights_vae(mesh, weights):
    """Shard VAE weights (replicate to all devices)."""
    result = {}
    for k, v in weights.items():
        v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
        result[k] = v
    return result


# ============================================================================
# Pipeline Setup
# ============================================================================

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


def load_pipeline(args):
    """Load pipeline before enabling torchax."""
    print("\n=== Loading CogVideoX Pipeline ===")
    print(f"Model: {args.model_id}")
    
    print("- Loading Tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    
    print("- Loading Text Encoder...")
    text_encoder = T5EncoderModel.from_pretrained(args.model_id, subfolder="text_encoder")
    
    print("- Loading TorchAx VAE...")
    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.model_id, subfolder="vae", torch_dtype=torch.bfloat16
    )
    
    print("- Loading TorchAx Transformer...")
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        args.model_id, subfolder="transformer"
    )
    
    print("- Loading Scheduler...")
    scheduler = CogVideoXDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    
    print("- Creating Pipeline...")
    pipe = CogVideoXPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
    )
    
    print("✓ Models loaded successfully\n")
    return pipe


def setup_pipeline_for_jax(pipe, args, mesh, env):
    """Setup CogVideoX pipeline for JAX/TPU execution."""
    print("=== Moving Models to TPU ===")
    
    # Register custom operators
    print("- Registering custom JAX operators...")
    # 注意：测试验证 conv2d 替换不必要，且会拖慢 JIT 编译（266s → 60s）
    # 因此不再使用 torch_conv2d_jax 替换
    override_op_definition(
        env,
        torch.nn.functional.scaled_dot_product_attention,
        functools.partial(scaled_dot_product_attention, env=env, mesh=mesh),
    )
    
    # Move scheduler to JAX
    print("- Moving Scheduler to JAX...")
    for k, v in pipe.scheduler.__dict__.items():
        if isinstance(v, torch.Tensor):
            setattr(pipe.scheduler, k, v.to('jax'))
    
    # Text Encoder
    print("- Moving Text Encoder to XLA and sharding...")
    move_module_to_xla(env, pipe.text_encoder)
    text_encoder_weights = shard_weight_dict(
        pipe.text_encoder.state_dict(), text_encoder_shardings, mesh
    )
    pipe.text_encoder.load_state_dict(text_encoder_weights, assign=True, strict=False)
    torchax.interop.call_jax(jax.block_until_ready, text_encoder_weights)
    
    # Transformer
    print("- Moving Transformer to XLA and sharding...")
    move_module_to_xla(env, pipe.transformer)
    transformer_weights = shard_weight_dict(
        pipe.transformer.state_dict(), transformer_shardings_fsdp, mesh
    )
    pipe.transformer.load_state_dict(transformer_weights, assign=True, strict=False)
    torchax.interop.call_jax(jax.block_until_ready, transformer_weights)
    
    # VAE
    print("- Moving VAE to XLA and sharding...")
    move_module_to_xla(env, pipe.vae)
    vae_weights = shard_weights_vae(mesh, pipe.vae.state_dict())
    pipe.vae.load_state_dict(vae_weights, assign=True, strict=False)
    torchax.interop.call_jax(jax.block_until_ready, vae_weights)
    
    # Compile components
    print("- Compiling Transformer...")
    pipe.transformer = torchax.compile(
        pipe.transformer,
        torchax.CompileOptions(jax_jit_kwargs={'static_argnames': ('return_dict',)})
    )
    
    print("- Compiling Text Encoder...")
    pipe.text_encoder = torchax.compile(pipe.text_encoder)
    
    print("- Compiling VAE Decoder...")
    pipe.vae.decoder = torchax.compile(pipe.vae.decoder)
    
    print("=== Pipeline Setup Complete ===\n")
    return pipe


def run_generation(pipe, args, mesh, env):
    """Run video generation with warmup and benchmark."""
    prompt = ("A panda, dressed in a small, red jacket and a tiny hat, sits on a "
              "wooden stool in a serene bamboo forest. The panda's fluffy paws strum "
              "a miniature acoustic guitar, producing soft, melodic tunes. Nearby, "
              "a few other pandas gather, watching curiously and some clapping in rhythm. "
              "Sunlight filters through the tall bamboo, casting a gentle glow on the scene.")
    
    generator = torch.Generator()
    generator.manual_seed(42)
    
    pipe_kwargs = {
        'prompt': prompt,
        'height': args.height,
        'width': args.width,
        'num_inference_steps': args.num_inference_steps,
        'num_frames': args.frames,
        'guidance_scale': 6.0,
        'generator': generator,
    }
    
    # 打印生成配置
    print(f"\n{'='*60}")
    print("生成配置")
    print(f"{'='*60}")
    print(f"  分辨率: {args.width}x{args.height}")
    print(f"  帧数: {args.frames}")
    print(f"  FPS: {args.fps}")
    print(f"  推理步数: {args.num_inference_steps}")
    print(f"  引导尺度: 6.0")
    print(f"  随机种子: 42")
    print(f"  Block sizes: BQSIZE={BQSIZE}, BKVSIZE={BKVSIZE}, BKVCOMPUTESIZE={BKVCOMPUTESIZE}, BKVCOMPUTEINSIZE={BKVCOMPUTEINSIZE}")
    
    # Profile context
    if args.profile:
        print(f"\n[Profiler] 启用 JAX Profiler，输出目录: {PROFILE_OUT_PATH}")
        profiler_context = jax.profiler.trace(
            PROFILE_OUT_PATH,
            create_perfetto_link=False
        )
    else:
        profiler_context = nullcontext()
    
    with mesh, profiler_context, env:
        # === Warmup Run ===
        print(f"\n{'='*60}")
        print("预热运行（触发 JIT 编译）")
        print(f"{'='*60}")
        warmup_start = time.perf_counter()
        output = pipe(**pipe_kwargs).frames[0]
        jax.effects_barrier()
        warmup_time = time.perf_counter() - warmup_start
        
        warmup_time_per_step = warmup_time / args.num_inference_steps
        print(f"\n✓ 预热完成")
        print(f"  总耗时: {warmup_time:.2f}s")
        print(f"  平均每步: {warmup_time_per_step:.2f}s")
        
        # Save warmup output
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"cogvideox_{current_datetime}.mp4"
        export_to_video(output, file_name, fps=args.fps)
        print(f"  视频保存至: {file_name}")
        
        # === Benchmark Run ===
        print(f"\n{'='*60}")
        print("基准测试运行")
        print(f"{'='*60}")
        benchmark_start = time.perf_counter()
        output = pipe(**pipe_kwargs)
        jax.effects_barrier()
        benchmark_time = time.perf_counter() - benchmark_start
        
        benchmark_time_per_step = benchmark_time / args.num_inference_steps
        
        print(f"\n✓ 基准测试完成")
        print(f"  总耗时: {benchmark_time:.2f}s")
        print(f"  平均每步: {benchmark_time_per_step:.2f}s")
    
    # === 性能统计 ===
    print(f"\n{'='*60}")
    print("性能统计")
    print(f"{'='*60}")
    print(f"  预热时间: {warmup_time:.2f}s ({warmup_time_per_step:.2f}s/step)")
    print(f"  基准时间: {benchmark_time:.2f}s ({benchmark_time_per_step:.2f}s/step)")
    print(f"  加速比: {warmup_time / benchmark_time:.2f}x")


# ============================================================================
# Main Function
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CogVideoX Video Generation on TPU")
    
    # Model settings
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    
    # Video settings
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--num_inference_steps", type=int, default=NUM_STEPS)
    
    # Sharding settings
    parser.add_argument("--use_dp", action="store_true", default=USE_DP)
    parser.add_argument("--use_fsdp", action="store_true", default=USE_FSDP)
    
    # Other settings
    parser.add_argument("--profile", action="store_true", default=False,
                        help="Run profiler")
    
    return parser.parse_args()


def main():
    """Main entry point for CogVideoX video generation."""
    args = parse_args()
    
    print(f"\n{'='*60}")
    print("CogVideoX Text-to-Video 生成（TPU Splash Attention）")
    print(f"{'='*60}")
    print(f"Configuration: {args}")
    
    # Configure JAX
    jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    
    torch.set_default_dtype(torch.bfloat16)
    
    # Setup PyTree registrations
    setup_pytree_registrations()
    
    # Load pipeline BEFORE enabling torchax
    pipe = load_pipeline(args)
    
    # Now enable torchax
    torchax.enable_globally()
    env = torchax.default_env()
    
    # Create mesh - 2D mesh (dp, tp) like Wan2.1
    assert len(jax.devices()) % 2 == 0
    dp_dim = 2 if args.use_dp else 1
    tp_dim = len(jax.devices()) // dp_dim
    
    mesh_devices = mesh_utils.create_device_mesh(
        (dp_dim, tp_dim), allow_split_physical_axes=True
    )
    mesh = Mesh(mesh_devices, ("dp", "tp"))
    
    print(f"\nMesh 配置:")
    print(f"  dp_dim={dp_dim}, tp_dim={tp_dim}")
    print(f"  总设备数: {len(jax.devices())}")
    print(f"  Mesh: {mesh}")
    
    # Setup pipeline for JAX (move to TPU and shard)
    pipe = setup_pipeline_for_jax(pipe, args, mesh, env)
    
    # Run generation
    run_generation(pipe, args, mesh, env)
    
    print(f"\n{'='*60}")
    print("✓ 生成完成！")
    print(f"{'='*60}")
    
    # Force exit to avoid torchax background thread blocking
    os._exit(0)


if __name__ == '__main__':
    main()
