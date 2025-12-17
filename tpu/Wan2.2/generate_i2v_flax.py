#!/usr/bin/env python3
"""
Wan 2.2 Image-to-Video Generation with TPU Splash Attention

本脚本提供在 TPU 上运行 Wan 2.2 Image-to-Video 生成的完整实现，
使用 JAX/Flax 和优化的 Splash Attention。

结构:
1. Imports and Configuration
2. Helper Functions
3. Splash Attention Implementation
4. Sharding Strategies
5. Weight Sharding Functions
6. Pipeline Setup
7. Main Function
"""

import os
import sys
import warnings
import logging

# ============================================================================
# 环境配置和 Warning 过滤（必须在其他 import 之前）
# ============================================================================

os.environ.setdefault('JAX_MEMORY_DEBUG', '0')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 全局过滤 warnings
warnings.filterwarnings('ignore')  # 过滤所有 warnings

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
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
from jax.experimental.pallas.ops.tpu import splash_attention
from jax.experimental.shard_map import shard_map
from jax.experimental import mesh_utils

import torchax
from torchax.ops import ops_registry, jaten, jtorch

# Wan 2.2 I2V pipeline with _flax versions
from diffusers.pipelines.wan.pipeline_wan_i2v_flax import WanImageToVideoPipeline
from diffusers.models.autoencoders.autoencoder_kl_wan_flax import AutoencoderKLWan
from diffusers.models.transformers.transformer_wan_flax import WanTransformer3DModel
from diffusers.utils import export_to_video, load_image
from diffusers.models.autoencoders import vae as diffusers_vae
from diffusers.models import modeling_outputs as diffusers_modeling_outputs
from transformers import modeling_outputs

# Custom splash attention
import custom_splash_attention


# ============================================================================
# Configuration Constants
# ============================================================================

# Model Configuration
MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

# Size Configuration
SIZE_CONFIGS = {
    "720*1280": (720, 1280),
    "1280*720": (1280, 720),
    "480*832": (480, 832),
    "832*480": (832, 480),
}

MAX_AREA_CONFIGS = {
    "720*1280": 720 * 1280,
    "1280*720": 1280 * 720,
    "480*832": 480 * 832,
    "832*480": 832 * 480,
}

# Default Prompts and Image
DEFAULT_PROMPT = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
DEFAULT_NEG_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
DEFAULT_IMAGE_PATH = "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG"

# Video Generation Settings
FRAMES = 81
FPS = 16
NUM_STEPS = 40
GUIDANCE_SCALE = 3.5

# Splash Attention Block Sizes
BQSIZE = 3328
BKVSIZE = 2816
BKVCOMPUTESIZE = 256
BKVCOMPUTEINSIZE = 256

# Attention Settings
USE_K_SMOOTH = False
USE_CUSTOM_ATTENTION = True

# Mesh Sharding Configuration
DEFAULT_DP = 2

# Profiler Output Path
PROFILE_OUT_PATH = "/tmp/wan_prof"


# ============================================================================
# Helper Functions
# ============================================================================

def pad_to_multiple(x, multiple, axis):
    """Pad array to next multiple along axis."""
    seq_len = x.shape[axis]
    pad_len = (multiple - seq_len % multiple) % multiple
    if pad_len == 0:
        return x, seq_len
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, pad_len)
    return jnp.pad(x, pad_width), seq_len


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
# Sharding Strategies
# ============================================================================

TEXT_ENCODER_SHARDINGS = {
    r'shared.weight': (('dp','tp'),),
    r'encoder.block.*.layer.*.SelfAttention.q.weight': (('dp','tp'),),
    r'encoder.block.*.layer.*.SelfAttention.k.weight': (('dp','tp'),),
    r'encoder.block.*.layer.*.SelfAttention.v.weight': (('dp','tp'),),
    r'encoder.block.*.layer.*.SelfAttention.o.weight': (None, ('dp','tp'),),
    r'encoder.block.*.layer.*.DenseReluDense.wi_0.weight': (('dp','tp'),),
    r'encoder.block.*.layer.*.DenseReluDense.wi_1.weight': (('dp','tp'),),
    r'encoder.block.*.layer.*.DenseReluDense.wo.weight': (None, ('dp','tp'),),
}

TRANSFORMER_SHARDINGS = {
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

VAE_ENCODER_SHARDINGS = {}
VAE_DECODER_SHARDINGS = {}


# ============================================================================
# Weight Sharding Functions
# ============================================================================

def shard_weight_dict(weight_dict, sharding_dict, mesh):
    """Apply sharding to weights based on pattern matching."""
    result = {}
    for k, v in weight_dict.items():
        if isinstance(v, torch.Tensor):
            v = v.to("jax")
        
        matched = False
        for target, sharding in sharding_dict.items():
            if re.fullmatch(target, k) is not None:
                v.apply_jax_(jax.device_put, NamedSharding(mesh, P(*sharding)))
                matched = True
                break
        
        if not matched:
            # Replicate
            v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
        
        result[k] = v
    return result


# ============================================================================
# PyTree Registrations
# ============================================================================

def setup_pytree_registrations():
    """Register PyTree nodes for JAX transformations."""
    def flatten_model_output(obj):
        return obj.to_tuple(), type(obj)
    
    def unflatten_model_output(aux, children):
        return aux(*children)
    
    # Text encoder output
    jax.tree_util.register_pytree_node(
        modeling_outputs.BaseModelOutputWithPastAndCrossAttentions,
        flatten_model_output,
        unflatten_model_output
    )
    
    # VAE decode output
    jax.tree_util.register_pytree_node(
        diffusers_vae.DecoderOutput,
        flatten_model_output,
        unflatten_model_output
    )
    
    # VAE encode output
    jax.tree_util.register_pytree_node(
        diffusers_modeling_outputs.AutoencoderKLOutput,
        flatten_model_output,
        unflatten_model_output
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
    
    jax.tree_util.register_pytree_node(
        diffusers_vae.DiagonalGaussianDistribution,
        flatten_gaussian,
        unflatten_gaussian
    )


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
    print("\n=== Loading Wan 2.2 I2V Pipeline ===")
    print("Loading models from HuggingFace...")
    
    pipe = WanImageToVideoPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        vae=AutoencoderKLWan.from_pretrained(
            args.model_id, subfolder="vae", torch_dtype=torch.bfloat16
        ),
        transformer=WanTransformer3DModel.from_pretrained(
            args.model_id, subfolder="transformer", torch_dtype=torch.bfloat16
        ),
        transformer_2=WanTransformer3DModel.from_pretrained(
            args.model_id, subfolder="transformer_2", torch_dtype=torch.bfloat16
        ),
    )
    print("✓ Models loaded successfully\n")
    return pipe


def setup_pipeline_for_jax(pipe, mesh, env):
    """Setup Wan I2V pipeline for JAX/TPU execution."""
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
    
    print("- Text Encoder...")
    move_module_to_xla(env, pipe.text_encoder)
    pipe.text_encoder = torchax.compile(pipe.text_encoder)
    pipe.text_encoder.params = shard_weight_dict(
        pipe.text_encoder.params, TEXT_ENCODER_SHARDINGS, mesh
    )
    pipe.text_encoder.buffers = shard_weight_dict(
        pipe.text_encoder.buffers, TEXT_ENCODER_SHARDINGS, mesh
    )
    
    transformer_options = torchax.CompileOptions(
        jax_jit_kwargs={"static_argnames": ("return_dict",)}
    )
    
    print("- Transformer...")
    move_module_to_xla(env, pipe.transformer)
    pipe.transformer = torchax.compile(pipe.transformer, transformer_options)
    pipe.transformer.params = shard_weight_dict(
        pipe.transformer.params, TRANSFORMER_SHARDINGS, mesh
    )
    pipe.transformer.buffers = shard_weight_dict(
        pipe.transformer.buffers, TRANSFORMER_SHARDINGS, mesh
    )
    
    print("- Transformer 2...")
    move_module_to_xla(env, pipe.transformer_2)
    pipe.transformer_2 = torchax.compile(pipe.transformer_2, transformer_options)
    pipe.transformer_2.params = shard_weight_dict(
        pipe.transformer_2.params, TRANSFORMER_SHARDINGS, mesh
    )
    pipe.transformer_2.buffers = shard_weight_dict(
        pipe.transformer_2.buffers, TRANSFORMER_SHARDINGS, mesh
    )
    
    print("- VAE...")
    move_module_to_xla(env, pipe.vae)
    pipe.vae.encoder = torchax.compile(pipe.vae.encoder)
    pipe.vae.encoder.params = shard_weight_dict(
        pipe.vae.encoder.params, VAE_ENCODER_SHARDINGS, mesh
    )
    pipe.vae.encoder.buffers = shard_weight_dict(
        pipe.vae.encoder.buffers, VAE_ENCODER_SHARDINGS, mesh
    )
    
    pipe.vae.decoder = torchax.compile(pipe.vae.decoder)
    pipe.vae.decoder.params = shard_weight_dict(
        pipe.vae.decoder.params, VAE_DECODER_SHARDINGS, mesh
    )
    pipe.vae.decoder.buffers = shard_weight_dict(
        pipe.vae.decoder.buffers, VAE_DECODER_SHARDINGS, mesh
    )
    
    print("=== Pipeline Setup Complete ===\n")
    return pipe


def run_generation(pipe, args, mesh):
    """Run I2V generation with optional profiling."""
    # Load and resize image
    image = load_image(args.image)
    max_area = MAX_AREA_CONFIGS[args.size]
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    image = image.resize((width, height))
    
    print(f"\n{'='*60}")
    print("生成配置")
    print(f"{'='*60}")
    print(f"  图像尺寸: {width}x{height}")
    print(f"  帧数: {args.frames}")
    print(f"  FPS: {args.fps}")
    print(f"  推理步数: {args.num_steps}")
    print(f"  引导尺度: {args.guidance_scale}")
    print(f"  随机种子: {args.seed}")
    print(f"  Block sizes: BQSIZE={BQSIZE}, BKVSIZE={BKVSIZE}, BKVCOMPUTESIZE={BKVCOMPUTESIZE}")
    
    generator = torch.Generator().manual_seed(args.seed)
    
    pipe_kwargs = {
        'image': image,
        'prompt': args.prompt,
        'negative_prompt': args.negative_prompt,
        'height': height,
        'width': width,
        'num_frames': args.frames,
        'guidance_scale': args.guidance_scale,
        'num_inference_steps': args.num_steps,
        'generator': generator,
    }
    
    # Profile context
    if args.profile:
        print(f"\n[Profiler] 启用 JAX Profiler，输出目录: {args.profile_output_path}")
        profiler_context = jax.profiler.trace(
            args.profile_output_path,
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
        warmup_time_per_step = warmup_time / args.num_steps
        print(f"\n✓ 预热完成")
        print(f"  总耗时: {warmup_time:.2f}s")
        print(f"  平均每步: {warmup_time_per_step:.2f}s")
        
        # Save output
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{current_datetime}.mp4"
        export_to_video(output, file_name, fps=args.fps)
        print(f"  视频保存至: {file_name}")
        
        # === Benchmark Run ===
        print(f"\n{'='*60}")
        print("基准测试运行")
        print(f"{'='*60}")
        benchmark_start = time.perf_counter()
        output = pipe(**pipe_kwargs).frames[0]
        jax.effects_barrier()
        benchmark_time = time.perf_counter() - benchmark_start
        
        # 计算每步平均时间（基准测试）
        benchmark_time_per_step = benchmark_time / args.num_steps
        
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
    parser = argparse.ArgumentParser(
        description="Wan 2.2 Image-to-Video Generation on TPU"
    )
    
    # Model settings
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    
    # Video settings
    parser.add_argument("--size", type=str, default="720*1280",
                       choices=list(SIZE_CONFIGS.keys()))
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS)
    parser.add_argument("--guidance_scale", type=float, default=GUIDANCE_SCALE)
    
    # Input settings
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEG_PROMPT)
    parser.add_argument("--seed", type=int, default=0)
    
    # Sharding settings
    parser.add_argument("--dp", type=int, default=DEFAULT_DP,
                       help="Data parallelism dimension")
    
    # Other settings
    parser.add_argument("--profile", action="store_true", default=False,
                       help="Run profiler")
    parser.add_argument("--profile_output_path", type=str,
                       default=PROFILE_OUT_PATH)
    
    return parser.parse_args()


def main():
    """Main entry point for Wan I2V generation."""
    args = parse_args()
    
    print(f"\n{'='*60}")
    print("Wan 2.2 Image-to-Video 生成（TPU Splash Attention）")
    print(f"{'='*60}")
    
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
    
    # Create mesh
    assert len(jax.devices()) % args.dp == 0
    tp_dim = len(jax.devices()) // args.dp
    mesh_devices = mesh_utils.create_device_mesh(
        (args.dp, tp_dim), allow_split_physical_axes=True
    )
    mesh = Mesh(mesh_devices, ("dp", "tp"))
    
    print(f"\nMesh 配置:")
    print(f"  dp_dim={args.dp}, tp_dim={tp_dim}")
    print(f"  总设备数: {len(jax.devices())}")
    print(f"  Mesh: {mesh}")
    
    # Setup pipeline for JAX (move to TPU and shard)
    pipe = setup_pipeline_for_jax(pipe, mesh, env)
    
    # Run generation
    run_generation(pipe, args, mesh)
    
    print(f"\n{'='*60}")
    print("✓ 生成完成！")
    print(f"{'='*60}")
    
    # 强制退出以避免 torchax 后台线程阻塞
    os._exit(0)


if __name__ == '__main__':
    main()