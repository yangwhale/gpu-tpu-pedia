#!/usr/bin/env python3
"""
Wan 2.1 Text-to-Video Generation with TPU Splash Attention

本脚本提供在 TPU 上运行 Wan 2.1 Text-to-Video 生成的完整实现，
使用 JAX/Flax 和优化的 Splash Attention。

结构:
1. Imports and Configuration
2. Helper Functions
3. Splash Attention Implementation
4. Sharding Strategies
5. Weight Sharding Functions
6. Pipeline Setup
7. Main Function

Optimizations from Wan2.2:
- Using diffusers' AutoencoderKLWan (Flax version) for better compatibility
- PyTree registrations for VAE outputs
- torch_conv2d_jax op override for correct conv2d behavior
- Pipeline loading before torchax.enable_globally() to avoid safetensors issues
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
from flax import nnx

import torchax
from torchax.ops import ops_registry, jaten
from transformers import modeling_outputs

# VAE outputs for PyTree registration
from diffusers.models.autoencoders import vae as diffusers_vae
from diffusers.models import modeling_outputs as diffusers_modeling_outputs

# Wan-specific imports - use diffusers' torchax VAE
from diffusers.pipelines.wan.pipeline_wan_torchax import WanPipeline
from diffusers.models.autoencoders.autoencoder_kl_wan_torchax import AutoencoderKLWan
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

# Custom splash attention
import custom_splash_attention


# ============================================================================
# Configuration Constants
# ============================================================================

# Model Configuration
MODEL_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"

# Video Generation Settings (720P)
FLOW_SHIFT = 5.0  # 5.0 for 720P, 3.0 for 480P
WIDTH = 1280
HEIGHT = 720
FRAMES = 81
FPS = 16
NUM_STEPS = 50

# Splash Attention Block Sizes
BQSIZE = 3328
BKVSIZE = 2816
BKVCOMPUTESIZE = 256
BKVCOMPUTEINSIZE = 256

# Attention Settings
USE_K_SMOOTH = False  # K smoothing for better numerical stability

# Mesh Sharding Configuration
USE_DP = True
USE_FSDP = True

# Profiler Output Path
PROFILE_OUT_PATH = "/dev/shm/tensorboard"


# ============================================================================
# Helper Functions
# ============================================================================

def to_torch_recursive(x):
    """Recursively convert JAX arrays to PyTorch tensors."""
    if 'ArrayImpl' in str(type(x)) or isinstance(x, jnp.ndarray):
        np_array = np.array(x)
        if hasattr(x, 'dtype') and x.dtype == jnp.bfloat16:
            return torch.from_numpy(np_array.astype(np.float32)).to(torch.bfloat16)
        return torch.from_numpy(np_array)
    elif isinstance(x, (list, tuple)):
        return type(x)(to_torch_recursive(xx) for xx in x)
    elif isinstance(x, dict):
        return {k: to_torch_recursive(v) for k, v in x.items()}
    elif hasattr(x, 'sample'):
        sample = to_torch_recursive(x.sample)
        return x.replace(sample=sample) if hasattr(x, 'replace') else sample
    return x


def setup_pytree_registrations():
    """Register PyTree nodes for JAX transformations."""
    def model_output_flatten(obj):
        return obj.to_tuple(), type(obj)

    def model_output_unflatten(aux, children):
        return aux(*children)

    # Text encoder output
    register_pytree_node(
        modeling_outputs.BaseModelOutputWithPastAndCrossAttentions,
        model_output_flatten,
        model_output_unflatten
    )
    
    # VAE decode output
    register_pytree_node(
        diffusers_vae.DecoderOutput,
        model_output_flatten,
        model_output_unflatten
    )
    
    # VAE encode output
    register_pytree_node(
        diffusers_modeling_outputs.AutoencoderKLOutput,
        model_output_flatten,
        model_output_unflatten
    )
    
    # DiagonalGaussianDistribution
    def flatten_gaussian(obj):
        return (obj.parameters, obj.mean, obj.logvar, obj.deterministic,
                obj.std, obj.var), None
    
    def unflatten_gaussian(aux, children):
        obj = object.__new__(diffusers_vae.DiagonalGaussianDistribution)
        obj.parameters = children[0]
        obj.mean = children[1]
        obj.logvar = children[2]
        obj.deterministic = children[3]
        obj.std = children[4]
        obj.var = children[5]
        return obj
    
    register_pytree_node(
        diffusers_vae.DiagonalGaussianDistribution,
        flatten_gaussian,
        unflatten_gaussian
    )


def sharded_device_put(tensor, sharding):
    """Put tensor on devices with proper sharding for multi-host setups."""
    if isinstance(tensor, tuple):
        return tuple(sharded_device_put(t, sharding) for t in tensor)
    
    num_global_devices = jax.device_count()
    num_local_devices = jax.local_device_count()

    if num_global_devices == num_local_devices:
        return jax.device_put(tensor, sharding)

    shape = tensor.shape
    x_split = [
        jax.device_put(tensor[i], device)
        for device, i in sharding.addressable_devices_indices_map(shape).items()
    ]
    return jax.make_array_from_single_device_arrays(shape, sharding, x_split)


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
        # Fuse the ops of exp in softmax - multiply by log2(e)
        _LOG2_E = 1.44269504
        q = q * scale_factor * _LOG2_E

        def pad_to_block_multiple(x, block_size, axis):
            seq_len = x.shape[axis]
            pad_len = (block_size - seq_len % block_size) % block_size
            if pad_len == 0:
                return x, seq_len
            pad_width = [(0, 0)] * x.ndim
            pad_width[axis] = (0, pad_len)
            return jnp.pad(x, pad_width), seq_len

        def kernel_3d(q_3d, k_3d, v_3d):
            # Pad to block size multiple to avoid NaN in incomplete blocks
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

    return _sdpa_reference(query, key, value, attn_mask, dropout_p, is_causal,
                           scale, enable_gqa)


# ============================================================================
# Sharding Strategies
# ============================================================================

# Transformer sharding - using 2D mesh (dp, tp) like Wan2.2
transformer_shardings_fsdp = {
    r'condition_embedder.time_embedder.linear_1.weight': ('tp',),
    r'condition_embedder.time_embedder.linear_1.bias': ('tp',),
    r'condition_embedder.time_embedder.linear_2.weight': (None, 'tp',),
    r'condition_embedder.text_embedder.linear_1.weight': ('tp',),
    r'condition_embedder.text_embedder.linear_1.bias': ('tp',),
    r'condition_embedder.text_embedder.linear_2.weight': (None, 'tp',),
    r'blocks.*.attn1.to_q.weight': ('tp',),
    r'blocks.*.attn1.to_q.bias': ('tp',),
    r'blocks.*.attn1.to_k.weight': ('tp',),
    r'blocks.*.attn1.to_k.bias': ('tp',),
    r'blocks.*.attn1.to_v.weight': ('tp',),
    r'blocks.*.attn1.to_v.bias': ('tp',),
    r'blocks.*.attn1.to_out.*.weight': (None, 'tp',),
    r'blocks.*.attn2.to_q.weight': ('tp',),
    r'blocks.*.attn2.to_q.bias': ('tp',),
    r'blocks.*.attn2.to_k.weight': ('tp',),
    r'blocks.*.attn2.to_k.bias': ('tp',),
    r'blocks.*.attn2.to_v.weight': ('tp',),
    r'blocks.*.attn2.to_v.bias': ('tp',),
    r'blocks.*.attn2.to_out.*.weight': (None, 'tp',),
    r'blocks.*.ffn.net.*.proj.weight': ('tp',),
    r'blocks.*.ffn.net.*.proj.bias': ('tp',),
    r'blocks.*.ffn.net.*.weight': (None, 'tp',),
}

# Same sharding for TP mode
transformer_shardings_tp = transformer_shardings_fsdp

# Text Encoder (T5) sharding - using 2D mesh (dp, tp)
text_encoder_shardings = {
    r'shared.weight': (('dp', 'tp'),),
    r'encoder.block.*.layer.*.SelfAttention.q.weight': (('dp', 'tp'),),
    r'encoder.block.*.layer.*.SelfAttention.k.weight': (('dp', 'tp'),),
    r'encoder.block.*.layer.*.SelfAttention.v.weight': (('dp', 'tp'),),
    r'encoder.block.*.layer.*.SelfAttention.o.weight': (None, ('dp', 'tp')),
    r'encoder.block.*.layer.*.DenseReluDense.wi_0.weight': (('dp', 'tp'),),
    r'encoder.block.*.layer.*.DenseReluDense.wi_1.weight': (('dp', 'tp'),),
    r'encoder.block.*.layer.*.DenseReluDense.wo.weight': (None, ('dp', 'tp')),
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


# ============================================================================
# Pipeline Setup
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


def load_pipeline(args):
    """Load pipeline before enabling torchax to avoid safetensors issues."""
    print("\n=== Loading Wan 2.1 T2V Pipeline ===")
    print("Loading models from HuggingFace...")
    
    scheduler = UniPCMultistepScheduler(
        prediction_type='flow_prediction',
        use_flow_sigmas=True,
        num_train_timesteps=1000,
        flow_shift=args.flow_shift
    )
    
    # Load pipeline with diffusers' Flax VAE
    pipe = WanPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        vae=AutoencoderKLWan.from_pretrained(
            args.model_id, subfolder="vae", torch_dtype=torch.bfloat16
        ),
    )
    pipe.scheduler = scheduler
    
    print("✓ Models loaded successfully\n")
    return pipe


def setup_pipeline_for_jax(pipe, args, mesh, env):
    """Setup Wan pipeline for JAX/TPU execution."""
    print("=== Moving Models to TPU ===")
    
    # Register custom operators
    print("- Registering custom JAX operators...")
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
    
    # Text Encoder
    if args.t5_cpu:
        print("- Keeping Text Encoder on CPU...")
        pipe.text_encoder.to("cpu")
    else:
        print("- Moving Text Encoder to XLA and sharding...")
        move_module_to_xla(env, pipe.text_encoder)
        pipe.text_encoder = torchax.compile(pipe.text_encoder)
        pipe.text_encoder.params = shard_weight_dict(
            pipe.text_encoder.params, text_encoder_shardings, mesh
        )
        pipe.text_encoder.buffers = shard_weight_dict(
            pipe.text_encoder.buffers, text_encoder_shardings, mesh
        )
    
    # Transformer
    print("- Moving Transformer to XLA and sharding...")
    move_module_to_xla(env, pipe.transformer)
    # Move rope embeddings to JAX
    if hasattr(pipe.transformer.rope, 'freqs'):
        pipe.transformer.rope.freqs = pipe.transformer.rope.freqs.to('jax')
    else:
        pipe.transformer.rope.freqs_cos = pipe.transformer.rope.freqs_cos.to('jax')
        pipe.transformer.rope.freqs_sin = pipe.transformer.rope.freqs_sin.to('jax')
    
    transformer_options = torchax.CompileOptions(
        jax_jit_kwargs={'static_argnames': ('return_dict',)}
    )
    pipe.transformer = torchax.compile(pipe.transformer, transformer_options)
    
    transformer_shardings = transformer_shardings_fsdp if args.use_fsdp else transformer_shardings_tp
    pipe.transformer.params = shard_weight_dict(
        pipe.transformer.params, transformer_shardings, mesh
    )
    pipe.transformer.buffers = shard_weight_dict(
        pipe.transformer.buffers, transformer_shardings, mesh
    )
    
    # VAE - using diffusers' Flax VAE
    print("- Moving VAE to XLA...")
    move_module_to_xla(env, pipe.vae)
    pipe.vae.encoder = torchax.compile(pipe.vae.encoder)
    pipe.vae.decoder = torchax.compile(pipe.vae.decoder)
    
    # Wrap VAE decode to ensure correct dtype
    original_decode = pipe.vae.decode
    def decode_wrapper(z, *args, **kwargs):
        # Handle both JAX arrays and PyTorch tensors
        if isinstance(z, jnp.ndarray) or 'ArrayImpl' in str(type(z)):
            z = z.astype(jnp.bfloat16)
        elif hasattr(z, 'dtype') and z.dtype != torch.bfloat16:
            z = z.to(torch.bfloat16)
        return original_decode(z, *args, **kwargs)
    pipe.vae.decode = decode_wrapper
    
    print("=== Pipeline Setup Complete ===\n")
    return pipe


def run_generation(pipe, args, mesh):
    """Run video generation with optional profiling."""
    prompt = ("A cat and a dog baking a cake together in a kitchen. "
              "The cat is carefully measuring flour, while the dog is "
              "stirring the batter with a wooden spoon. The kitchen is cozy, "
              "with sunlight streaming through the window.")
    
    negative_prompt = ("Bright tones, overexposed, static, blurred details, "
                       "subtitles, style, works, paintings, images, static, "
                       "overall gray, worst quality, low quality, JPEG compression residue, "
                       "ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, "
                       "deformed, disfigured, misshapen limbs, fused fingers, still picture, "
                       "messy background, three legs, many people in the background, walking backwards")
    
    generator = torch.Generator()
    generator.manual_seed(42)
    
    pipe_kwargs = {
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'height': args.height,
        'width': args.width,
        'num_inference_steps': args.num_inference_steps,
        'num_frames': args.frames,
        'guidance_scale': 5.0,
        'generator': generator,
        'use_dp': args.use_dp,
    }
    
    # 打印生成配置
    print(f"\n{'='*60}")
    print("生成配置")
    print(f"{'='*60}")
    print(f"  分辨率: {args.width}x{args.height}")
    print(f"  帧数: {args.frames}")
    print(f"  FPS: {args.fps}")
    print(f"  推理步数: {args.num_inference_steps}")
    print(f"  引导尺度: 5.0")
    print(f"  随机种子: 42")
    print(f"  Block sizes: BQSIZE={BQSIZE}, BKVSIZE={BKVSIZE}, BKVCOMPUTESIZE={BKVCOMPUTESIZE}")
    
    # Profile context
    if args.profile:
        print(f"\n[Profiler] 启用 JAX Profiler，输出目录: {PROFILE_OUT_PATH}")
        profiler_context = jax.profiler.trace(
            PROFILE_OUT_PATH,
            create_perfetto_link=False
        )
    else:
        profiler_context = nullcontext()
    
    with mesh, profiler_context:
        # === Warmup Run ===
        print(f"\n{'='*60}")
        print("预热运行（触发 JIT 编译）")
        print(f"{'='*60}")
        warmup_start = time.perf_counter()
        output = pipe(**pipe_kwargs).frames[0]
        jax.effects_barrier()
        warmup_time = time.perf_counter() - warmup_start
        
        # 计算每步平均时间（预热）
        warmup_time_per_step = warmup_time / args.num_inference_steps
        print(f"\n✓ 预热完成")
        print(f"  总耗时: {warmup_time:.2f}s")
        print(f"  平均每步: {warmup_time_per_step:.2f}s")
        
        # Save output
        output = prepare_video_for_export(output, args.frames)
        if isinstance(output, np.ndarray) and output.ndim == 4 and output.shape[-2] == 3:
            output = output.transpose(3, 0, 1, 2)
        
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{current_datetime}.mp4"
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
        
        # 计算每步平均时间（基准测试）
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
    parser = argparse.ArgumentParser(description="Wan 2.1 Video Generation on TPU")
    
    # Model settings
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--flow_shift", type=float, default=FLOW_SHIFT)
    
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
    parser.add_argument("--t5_cpu", action="store_true", default=False,
                        help="Offload T5 text encoder to CPU")
    parser.add_argument("--profile", action="store_true", default=False,
                        help="Run profiler")
    
    return parser.parse_args()


def main():
    """Main entry point for Wan video generation."""
    # Parse arguments
    args = parse_args()
    
    print(f"\n{'='*60}")
    print("Wan 2.1 Text-to-Video 生成（TPU Splash Attention）")
    print(f"{'='*60}")
    print(f"Configuration: {args}")
    
    # Configure JAX
    jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    
    torch.set_default_dtype(torch.bfloat16)
    
    # Setup PyTree registrations
    setup_pytree_registrations()
    
    # Load pipeline BEFORE enabling torchax to avoid safetensors loading issues
    pipe = load_pipeline(args)
    
    # Now enable torchax
    torchax.enable_globally()
    env = torchax.default_env()
    
    # Create mesh - use 2D mesh (dp, tp) like Wan2.2
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
    run_generation(pipe, args, mesh)
    
    print(f"\n{'='*60}")
    print("✓ 生成完成！")
    print(f"{'='*60}")
    
    # Force exit to avoid torchax background thread blocking
    os._exit(0)


if __name__ == '__main__':
    main()
