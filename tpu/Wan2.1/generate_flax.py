"""
Wan 2.1 Video Generation with TPU Splash Attention

This file provides a clean implementation for running Wan 2.1 text-to-video
generation on TPU using JAX/Flax, with optimized Splash Attention.

Structure:
1. Imports and Configuration
2. Helper Functions
3. Splash Attention Implementation
4. Sharding Strategies
5. Weight Sharding Functions
6. VAE Components
7. Pipeline Setup
8. Main Function
"""

import os
os.environ.setdefault('JAX_MEMORY_DEBUG', '0')

import time
import re
import math
import functools
import argparse
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
from flax.linen import partitioning as nn_partitioning

import torchax
from torchax.ops import ops_registry
from transformers import modeling_outputs

# Wan-specific imports
from maxdiffusion.models.wan.autoencoder_kl_wan import (
    AutoencoderKLWan,
    AutoencoderKLWanCache,
)
from maxdiffusion.models.wan.wan_utils import load_wan_vae
from diffusers import WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video


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
BQSIZE = 3024
BKVSIZE = 2048
BKVCOMPUTESIZE = 1024

# Attention Settings
WINDOW_SIZE = None  # None for full attention, tuple (left, right) for local
USE_K_SMOOTH = True

# Mesh Sharding Configuration
USE_DP = True
SP_NUM = 1
USE_FSDP = True

# VAE Sharding Rules
LOGICAL_AXIS_RULES = (
    ('conv_out', ('tp', 'dp', 'sp')),
    ('conv_in', ('tp', 'dp', 'sp'))
)

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

    register_pytree_node(
        modeling_outputs.BaseModelOutputWithPastAndCrossAttentions,
        model_output_flatten,
        model_output_unflatten
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


def _tpu_splash_attention(query, key, value, env, scale=None, is_causal=False, 
                          window_size=None, bqsize=BQSIZE, bkvsize=BKVSIZE, 
                          bkvcomputesize=BKVCOMPUTESIZE):
    """TPU Splash Attention implementation with sharding support."""
    mesh = env._mesh
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
            
            # Self attention (long KV sequence)
            if k_3d.shape[1] > 10000:
                q_3d_padded, q_orig_len = pad_to_multiple(q_3d, bqsize, axis=1)
                k_3d_padded, k_orig_len = pad_to_multiple(k_3d, bkvsize, axis=1)
                v_3d_padded, v_orig_len = pad_to_multiple(v_3d, bkvsize, axis=1)
            else:
                # Cross attention (short KV sequence, no padding)
                q_3d_padded, q_orig_len = pad_to_multiple(q_3d, bqsize, axis=1)
                k_3d_padded, k_orig_len = k_3d, k_3d.shape[1]
                v_3d_padded, v_orig_len = v_3d, v_3d.shape[1]

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


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                  is_causal=False, scale=None, enable_gqa=False,
                                  env=None, window_size=None):
    """Wrapper for scaled dot-product attention with TPU Splash support."""
    if env.config.use_tpu_splash_attention:
        jquery, jkey, jvalue = env.t2j_iso((query, key, value))
        
        if USE_K_SMOOTH:
            key_mean = jnp.mean(jkey, axis=2, keepdims=True)
            jkey = jkey - key_mean
        
        res = _tpu_splash_attention(
            jquery, jkey, jvalue, env, 
            scale=scale, is_causal=is_causal, window_size=window_size
        )
        return env.j2t_iso(res)

    return _sdpa_reference(query, key, value, attn_mask, dropout_p, is_causal,
                           scale, enable_gqa)


# ============================================================================
# Sharding Strategies
# ============================================================================

# Transformer sharding for FSDP mode
transformer_shardings_fsdp = {
    r'condition_embedder.time_embedder.linear_1.weight': (None, ('tp', 'sp')),
    r'condition_embedder.time_embedder.linear_2.weight': (('tp', 'sp'), None),
    r'condition_embedder.time_proj.weight': (('tp', 'sp'), None),
    r'condition_embedder.text_embedder.linear_1.weight': (None, ('tp', 'sp')),
    r'condition_embedder.text_embedder.linear_2.weight': (('tp', 'sp'), None),
    r'blocks.\d+.attn1.to_q.weight': (None, ('tp', 'sp')),
    r'blocks.\d+.attn1.to_k.weight': (None, ('tp', 'sp')),
    r'blocks.\d+.attn1.to_v.weight': (None, ('tp', 'sp')),
    r'blocks.\d+.attn1.to_out.0.weight': (('tp', 'sp'), None),
    r'blocks.\d+.attn2.to_q.weight': (None, ('tp', 'sp')),
    r'blocks.\d+.attn2.to_k.weight': (None, ('tp', 'sp')),
    r'blocks.\d+.attn2.to_v.weight': (None, ('tp', 'sp')),
    r'blocks.\d+.attn2.to_out.0.weight': (('tp', 'sp'), None),
    r'blocks.\d+.ffn.net.0.proj.weight': (None, ('tp', 'sp')),
    r'blocks.\d+.ffn.net.2.weight': (('tp', 'sp'), None),
    r'proj_out.weight': (None, ('tp', 'sp')),
}

# Transformer sharding for Tensor Parallel mode
transformer_shardings_tp = {
    r'condition_embedder.time_embedder.linear_1.weight': (('tp', 'sp'), None),
    r'condition_embedder.time_embedder.linear_1.bias': (('tp', 'sp'),),
    r'condition_embedder.time_embedder.linear_2.weight': (None, ('tp', 'sp')),
    r'condition_embedder.text_embedder.linear_1.weight': (('tp', 'sp'), None),
    r'condition_embedder.text_embedder.linear_1.bias': (('tp', 'sp'),),
    r'condition_embedder.text_embedder.linear_2.weight': (None, ('tp', 'sp')),
    r'blocks.\d+.attn1.to_q.weight': (('tp', 'sp'), None),
    r'blocks.\d+.attn1.to_q.bias': (('tp', 'sp'),),
    r'blocks.\d+.attn1.to_k.weight': (('tp', 'sp'),),
    r'blocks.\d+.attn1.to_k.bias': (('tp', 'sp'),),
    r'blocks.\d+.attn1.to_v.weight': (('tp', 'sp'),),
    r'blocks.\d+.attn1.to_v.bias': (('tp', 'sp'),),
    r'blocks.\d+.attn1.to_out.0.weight': (None, ('tp', 'sp')),
    r'blocks.\d+.attn2.to_q.weight': (('tp', 'sp'),),
    r'blocks.\d+.attn2.to_q.bias': (('tp', 'sp'),),
    r'blocks.\d+.attn2.to_k.weight': (('tp', 'sp'),),
    r'blocks.\d+.attn2.to_k.bias': (('tp', 'sp'),),
    r'blocks.\d+.attn2.to_v.weight': (('tp', 'sp'),),
    r'blocks.\d+.attn2.to_v.bias': (('tp', 'sp'),),
    r'blocks.\d+.attn2.to_out.0.weight': (None, ('tp', 'sp')),
    r'blocks.\d+.ffn.net.0.proj.weight': (('tp', 'sp'),),
    r'blocks.\d+.ffn.net.0.proj.bias': (('tp', 'sp'),),
    r'blocks.\d+.ffn.net.2.weight': (None, ('tp', 'sp')),
}

# Text Encoder (T5) sharding
text_encoder_shardings = {
    r'shared.weight': (('tp', 'dp', 'sp'),),
    r'encoder.block.*.layer.*.SelfAttention.q.weight': (('tp', 'dp', 'sp'),),
    r'encoder.block.*.layer.*.SelfAttention.k.weight': (('tp', 'dp', 'sp'),),
    r'encoder.block.*.layer.*.SelfAttention.v.weight': (('tp', 'dp', 'sp'),),
    r'encoder.block.*.layer.*.SelfAttention.o.weight': (None, ('tp', 'dp', 'sp')),
    r'encoder.block.*.layer.*.DenseReluDense.wi_0.weight': (('tp', 'dp', 'sp'),),
    r'encoder.block.*.layer.*.DenseReluDense.wi_1.weight': (('tp', 'dp', 'sp'),),
    r'encoder.block.*.layer.*.DenseReluDense.wo.weight': (None, ('tp', 'dp', 'sp')),
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


def _add_sharding_rule(vs, logical_axis_rules):
    """Add sharding rules to variable state."""
    vs.sharding_rules = logical_axis_rules
    return vs


@nnx.jit(static_argnums=(1,), donate_argnums=(0,))
def create_sharded_logical_model(model, logical_axis_rules):
    """Create a sharded model with logical axis rules."""
    graphdef, state, rest_of_state = nnx.split(model, nnx.Param, ...)
    p_add_sharding_rule = functools.partial(
        _add_sharding_rule, logical_axis_rules=logical_axis_rules
    )
    state = jax.tree.map(
        p_add_sharding_rule, state, 
        is_leaf=lambda x: isinstance(x, nnx.VariableState)
    )
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    model = nnx.merge(graphdef, sharded_state, rest_of_state)
    return model


# ============================================================================
# VAE Components
# ============================================================================

class ConfigWrapper:
    """Wrapper to make VAE config accessible as both dict and attributes."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)


class VAEProxy:
    """Proxy class for JAX VAE to work with PyTorch pipeline."""
    def __init__(self, vae, vae_cache, dtype, config):
        self._vae = vae
        self.vae_cache = vae_cache
        self.dtype = dtype
        self.config = config
    
    def __getattr__(self, name):
        return getattr(self._vae, name)
    
    def decode(self, *args, **kwargs):
        if 'feat_cache' not in kwargs:
            kwargs['feat_cache'] = self.vae_cache
        out = self._vae.decode(*args, **kwargs)
        return to_torch_recursive(out)


def load_wan_vae_fixed(pretrained_model_name_or_path, eval_shapes, device, hf_download=True):
    """Load Wan VAE weights with proper type handling to avoid torchax issues."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from flax.traverse_util import unflatten_dict
    from maxdiffusion.models.modeling_flax_pytorch_utils import (
        rename_key, rename_key_and_reshape_tensor, validate_flax_state_dict
    )
    
    device_obj = jax.local_devices(backend=device)[0]
    with jax.default_device(device_obj):
        if hf_download:
            ckpt_path = hf_hub_download(
                pretrained_model_name_or_path, 
                subfolder="vae", 
                filename="diffusion_pytorch_model.safetensors"
            )
        
        print(f"Loading Wan 2.1 VAE on {device}")
        
        tensors = {}
        with safe_open(ckpt_path, framework="np") as f:
            for k in f.keys():
                tensors[k] = jnp.array(f.get_tensor(k))
        
        flax_state_dict = {}
        cpu = jax.local_devices(backend="cpu")[0]
        
        for pt_key, tensor in tensors.items():
            renamed_pt_key = rename_key(pt_key)
            # Apply Wan-specific key transformations
            for old, new in [
                ("up_blocks_", "up_blocks."),
                ("mid_block_", "mid_block."),
                ("down_blocks_", "down_blocks."),
                ("conv_in.bias", "conv_in.conv.bias"),
                ("conv_in.weight", "conv_in.conv.weight"),
                ("conv_out.bias", "conv_out.conv.bias"),
                ("conv_out.weight", "conv_out.conv.weight"),
                ("attentions_", "attentions."),
                ("resnets_", "resnets."),
                ("upsamplers_", "upsamplers."),
                ("resample_", "resample."),
                ("conv1.bias", "conv1.conv.bias"),
                ("conv1.weight", "conv1.conv.weight"),
                ("conv2.bias", "conv2.conv.bias"),
                ("conv2.weight", "conv2.conv.weight"),
                ("time_conv.bias", "time_conv.conv.bias"),
                ("time_conv.weight", "time_conv.conv.weight"),
                ("quant_conv", "quant_conv.conv"),
                ("conv_shortcut", "conv_shortcut.conv"),
            ]:
                renamed_pt_key = renamed_pt_key.replace(old, new)
            
            if "decoder" in renamed_pt_key:
                renamed_pt_key = renamed_pt_key.replace("resample.1.bias", "resample.layers.1.bias")
                renamed_pt_key = renamed_pt_key.replace("resample.1.weight", "resample.layers.1.weight")
            if "encoder" in renamed_pt_key:
                renamed_pt_key = renamed_pt_key.replace("resample.1", "resample.conv")
            
            pt_tuple_key = tuple(renamed_pt_key.split("."))
            flax_key, flax_tensor = rename_key_and_reshape_tensor(pt_tuple_key, tensor, eval_shapes)
            flax_key = tuple(
                int(item) if isinstance(item, str) and item.isdigit() else item 
                for item in flax_key
            )
            flax_state_dict[flax_key] = jax.device_put(jnp.asarray(flax_tensor), device=cpu)
        
        validate_flax_state_dict(eval_shapes, flax_state_dict)
        flax_state_dict = unflatten_dict(flax_state_dict)
        del tensors
        jax.clear_caches()
    
    return flax_state_dict


# ============================================================================
# Pipeline Setup
# ============================================================================

def setup_wan_vae(model_id, mesh, vae_mesh):
    """Initialize and load Wan VAE with proper sharding."""
    with vae_mesh:
        key = jax.random.key(0)
        rngs = nnx.Rngs(key)
        
        wan_vae = AutoencoderKLWan(
            rngs=rngs,
            base_dim=96,
            z_dim=16,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            attn_scales=[],
            temperal_downsample=[False, True, True],
            mesh=vae_mesh
        )
    
    with mesh:
        vae_cache = AutoencoderKLWanCache(wan_vae)
        
        graphdef, state = nnx.split(wan_vae)
        params = state.to_pure_dict()
        params = load_wan_vae_fixed(model_id, params, "tpu")
        
        # Replicate to all devices
        sharding = NamedSharding(mesh, P())
        params = jax.tree_util.tree_map(
            lambda x: sharded_device_put(x, sharding), params
        )
        params = jax.tree_util.tree_map(
            lambda x: x.astype(jnp.bfloat16), params
        )
        wan_vae = nnx.merge(graphdef, params)
        
        # Apply logical sharding
        wan_vae = create_sharded_logical_model(
            model=wan_vae, 
            logical_axis_rules=LOGICAL_AXIS_RULES
        )
    
    return wan_vae, vae_cache


def setup_pipeline_for_jax(args, mesh):
    """Setup Wan pipeline for JAX/TPU execution."""
    print("\nConfiguring Wan Pipeline for JAX...")
    
    # Create VAE mesh
    vae_mesh = jax.make_mesh((1, len(jax.devices())), ('conv_in', 'conv_out'))
    
    # Initialize VAE
    print("- Loading JAX VAE...")
    wan_vae, vae_cache = setup_wan_vae(args.model_id, mesh, vae_mesh)
    
    # Temporarily disable torchax to load PyTorch components
    torchax.disable_globally()
    
    try:
        scheduler = UniPCMultistepScheduler(
            prediction_type='flow_prediction',
            use_flow_sigmas=True,
            num_train_timesteps=1000,
            flow_shift=args.flow_shift
        )
        pipe = WanPipeline.from_pretrained(
            args.model_id, 
            torch_dtype=torch.bfloat16, 
            use_safetensors=True
        )
        pipe.scheduler = scheduler
    finally:
        torchax.enable_globally()
    
    # Create torchax environment
    env = torchax.default_env()
    env.default_device_or_sharding = NamedSharding(mesh, P())
    env._mesh = mesh
    env.config.use_tpu_splash_attention = True
    
    # Replace VAE with JAX version
    vae_config = ConfigWrapper(
        latents_mean=np.array(wan_vae.latents_mean),
        latents_std=np.array(wan_vae.latents_std),
        z_dim=wan_vae.z_dim
    )
    pipe.vae = VAEProxy(wan_vae, vae_cache, torch.bfloat16, vae_config)
    pipe.vae_cache = vae_cache
    
    # Register custom attention
    print(f"- Registering Splash Attention (window_size: {args.window_size})...")
    custom_attention = functools.partial(
        scaled_dot_product_attention,
        env=env,
        window_size=args.window_size
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
    
    # Move and compile modules
    def _move_module(module):
        with jax.default_device('cpu'):
            state_dict = module.state_dict()
            state_dict = env.to_xla(state_dict)
            module.load_state_dict(state_dict, assign=True)
    
    # Text Encoder
    if args.t5_cpu:
        print("- Keeping Text Encoder on CPU...")
        pipe.text_encoder.to("cpu")
    else:
        print("- Moving Text Encoder to XLA and sharding...")
        _move_module(pipe.text_encoder)
        pipe.text_encoder = torchax.compile(pipe.text_encoder)
        pipe.text_encoder.params = shard_weight_dict(
            pipe.text_encoder.params, text_encoder_shardings, mesh
        )
        pipe.text_encoder.buffers = shard_weight_dict(
            pipe.text_encoder.buffers, text_encoder_shardings, mesh
        )
    
    # Transformer
    print("- Moving Transformer to XLA and sharding...")
    _move_module(pipe.transformer)
    pipe.transformer.rope.freqs = pipe.transformer.rope.freqs.to('jax')
    
    options = torchax.CompileOptions(
        jax_jit_kwargs={'static_argnames': ('return_dict',)}
    )
    pipe.transformer = torchax.compile(pipe.transformer, options)
    
    transformer_shardings = transformer_shardings_fsdp if args.use_fsdp else transformer_shardings_tp
    pipe.transformer.params = shard_weight_dict(
        pipe.transformer.params, transformer_shardings, mesh
    )
    pipe.transformer.buffers = shard_weight_dict(
        pipe.transformer.buffers, transformer_shardings, mesh
    )
    
    print("Pipeline configuration complete")
    return pipe, env


def run_generation(pipe, args, mesh):
    """Run video generation with optional profiling."""
    prompt = ("A cat and a dog baking a cake together in a kitchen. "
              "The cat is carefully measuring flour, while the dog is "
              "stirring the batter with a wooden spoon. The kitchen is cozy, "
              "with sunlight streaming through the window.")
    
    negative_prompt = ("Bright tones, overexposed, static, blurred details, "
                       "subtitles, style, works, paintings, images, static, "
                       "overall gray, worst quality, low quality")
    
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
    
    with mesh, nn_partitioning.axis_rules(LOGICAL_AXIS_RULES):
        # Warmup and save video
        print("\nGenerating video (warmup run)...")
        start = time.perf_counter()
        output = pipe(**pipe_kwargs).frames[0]
        jax.effects_barrier()
        warmup_time = time.perf_counter() - start
        print(f"Warmup completed in {warmup_time:.2f}s")
        
        # Save output
        output = prepare_video_for_export(output, args.frames)
        if isinstance(output, np.ndarray) and output.ndim == 4 and output.shape[-2] == 3:
            output = output.transpose(3, 0, 1, 2)
        
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{current_datetime}.mp4"
        export_to_video(output, file_name, fps=args.fps)
        print(f"Video saved to: {file_name}")
        
        # Profile if requested
        if args.profile:
            print("\nRunning profiler...")
            jax.profiler.start_trace(PROFILE_OUT_PATH)
            _ = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=3,
                num_frames=args.frames,
                guidance_scale=5.0,
                output_type="latent",
                generator=generator,
                use_dp=args.use_dp,
            )
            jax.effects_barrier()
            jax.profiler.stop_trace()
            print(f"Profile saved to: {PROFILE_OUT_PATH}")
        
        # Benchmark
        print("\nBenchmark run...")
        start = time.perf_counter()
        output = pipe(**pipe_kwargs)
        jax.effects_barrier()
        benchmark_time = time.perf_counter() - start
        print(f"Benchmark completed in {benchmark_time:.2f}s")
        print(f"Block sizes: BQSIZE={args.bqsize}, BKVSIZE={args.bkvsize}, "
              f"BKVCOMPUTESIZE={args.bkvcomputesize}")


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
    
    # Attention settings
    parser.add_argument("--window_size", type=int, nargs=2, default=None)
    parser.add_argument("--bqsize", type=int, default=BQSIZE)
    parser.add_argument("--bkvsize", type=int, default=BKVSIZE)
    parser.add_argument("--bkvcomputesize", type=int, default=BKVCOMPUTESIZE)
    parser.add_argument("--use_k_smooth", type=bool, default=USE_K_SMOOTH)
    
    # Sharding settings
    parser.add_argument("--use_dp", action="store_true", default=USE_DP)
    parser.add_argument("--sp_num", type=int, default=SP_NUM)
    parser.add_argument("--use_fsdp", type=bool, default=USE_FSDP)
    
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
    print(f"Configuration: {args}")
    
    # Update global settings from args
    global BQSIZE, BKVSIZE, BKVCOMPUTESIZE, USE_K_SMOOTH
    BQSIZE = args.bqsize
    BKVSIZE = args.bkvsize
    BKVCOMPUTESIZE = args.bkvcomputesize
    USE_K_SMOOTH = args.use_k_smooth
    
    # Configure JAX
    jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", 
                      "xla_gpu_per_fusion_autotune_cache_dir")
    
    torch.set_default_dtype(torch.bfloat16)
    
    # Setup PyTree registrations
    setup_pytree_registrations()
    
    # Initialize torchax
    torchax.enable_globally()
    
    # Create mesh
    tp_dim, dp_dim, sp_dim = len(jax.devices()), 1, 1
    if args.use_dp:
        tp_dim //= 2
        dp_dim = 2
    if args.sp_num > 1:
        tp_dim //= args.sp_num
        sp_dim = args.sp_num
    
    print(f"Mesh dimensions: tp_dim={tp_dim}, dp_dim={dp_dim}, sp_dim={sp_dim}")
    print(f"Number of devices: {len(jax.devices())}")
    
    mesh_devices = mesh_utils.create_device_mesh(
        (tp_dim, dp_dim, sp_dim), allow_split_physical_axes=True
    )
    mesh = Mesh(mesh_devices, ('tp', 'dp', 'sp'))
    
    # Setup pipeline
    pipe, env = setup_pipeline_for_jax(args, mesh)
    
    # Run generation
    with env:
        run_generation(pipe, args, mesh)
    
    print("\n✓ Generation complete")


if __name__ == '__main__':
    main()
