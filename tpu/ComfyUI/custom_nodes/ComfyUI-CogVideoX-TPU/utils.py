"""
CogVideoX TPU ComfyUI Node - Utility Functions

Based on gpu-tpu-pedia/tpu/CogVideoX/generate_diffusers_torchax_staged/utils.py
"""

import os
import re
import json
import numpy as np

import jax
import jax.numpy as jnp
import torch
from jax.tree_util import register_pytree_node
from jax.sharding import PartitionSpec as P, NamedSharding


# ============================================================================
# Model Configuration
# ============================================================================
MODEL_NAME = "zai-org/CogVideoX1.5-5B"

# Video Generation Settings
# 720P = 1280x720, 81 frames, 16fps (~5s video)
# Note: frames must satisfy (FRAMES-1)/4+1 is odd, otherwise VAE decoding adds extra frames
# Valid frames: 41, 49, 57, 65, 73, 81, 89, 97...
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FRAMES = 81
DEFAULT_FPS = 16
DEFAULT_NUM_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 6.0

# Splash Attention Configuration
BQSIZE = 3328
BKVSIZE = 2816
BKVCOMPUTESIZE = 256
BKVCOMPUTEINSIZE = 256

# K-smooth and custom attention settings
USE_K_SMOOTH = True
USE_CUSTOM_ATTENTION = True
LOG2_E = 1.44269504


# ============================================================================
# Mesh Configuration
# ============================================================================
DEFAULT_DP = 2  # Recommended DP=2 (faster for CFG positive+negative prompt parallelism)
USE_DP = True
USE_TP = True


# ============================================================================
# Text Encoder Sharding (T5 Model)
# ============================================================================
TEXT_ENCODER_SHARDINGS = {
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
# Transformer Sharding (CogVideoX-1.5 Model)
# ============================================================================
# Tensor Parallel mode (recommended) - 2D mesh (dp, tp)
TRANSFORMER_SHARDINGS_TP = {
    r'.*\.to_q\.weight$': (None, 'tp'),
    r'.*\.to_k\.weight$': (None, 'tp'),
    r'.*\.to_v\.weight$': (None, 'tp'),
    r'.*\.to_out.*\.weight$': ('tp', None),
    r'.*\.ff\.net\.0\.weight$': (None, 'tp'),
    r'.*\.ff\.net\.2\.weight$': ('tp', None),
}

# FSDP mode - 2D mesh (dp, tp)
TRANSFORMER_SHARDINGS_FSDP = {
    r'.*\.to_q\.weight$': ('tp', None),
    r'.*\.to_k\.weight$': ('tp', None),
    r'.*\.to_v\.weight$': ('tp', None),
    r'.*\.to_out.*\.weight$': (None, 'tp'),
    r'.*\.ff\.net\.0\.weight$': ('tp', None),
    r'.*\.ff\.net\.2\.weight$': (None, 'tp'),
}


# ============================================================================
# PyTree Registrations
# ============================================================================

def setup_pytree_registrations():
    """
    Register necessary PyTree nodes to support JAX transformations.
    """
    from transformers import modeling_outputs
    from diffusers.models.autoencoders import vae as diffusers_vae
    from diffusers.models import modeling_outputs as diffusers_modeling_outputs
    
    print("[CogVideoX] Registering PyTree nodes...")
    
    def flatten_model_output(obj):
        return obj.to_tuple(), type(obj)
    
    def unflatten_model_output(aux, children):
        return aux(*children)
    
    # Text encoder output
    try:
        register_pytree_node(
            modeling_outputs.BaseModelOutputWithPastAndCrossAttentions,
            flatten_model_output,
            unflatten_model_output
        )
    except ValueError:
        pass  # Already registered
    
    # VAE decode output
    try:
        register_pytree_node(
            diffusers_vae.DecoderOutput,
            flatten_model_output,
            unflatten_model_output
        )
    except ValueError:
        pass
    
    # VAE encode output
    try:
        register_pytree_node(
            diffusers_modeling_outputs.AutoencoderKLOutput,
            flatten_model_output,
            unflatten_model_output
        )
    except ValueError:
        pass
    
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
    
    try:
        register_pytree_node(
            diffusers_vae.DiagonalGaussianDistribution,
            flatten_gaussian,
            unflatten_gaussian
        )
    except ValueError:
        pass


# ============================================================================
# Sharding Utilities
# ============================================================================

def shard_weight_dict(weight_dict, sharding_dict, mesh):
    """Apply sharding to weights based on pattern matching.
    
    Args:
        weight_dict: Weight dictionary
        sharding_dict: Sharding strategy dictionary (regex -> PartitionSpec)
        mesh: JAX Mesh
        
    Returns:
        Sharded weight dictionary
    """
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


def move_module_to_xla(env, module):
    """Move module weights to XLA devices."""
    with jax.default_device("cpu"):
        state_dict = module.state_dict()
        state_dict = env.to_xla(state_dict)
        module.load_state_dict(state_dict, assign=True)


# ============================================================================
# Data Conversion
# ============================================================================

def to_torch_recursive(x):
    """Recursively convert JAX arrays to PyTorch tensors."""
    if 'ArrayImpl' in str(type(x)) or isinstance(x, jnp.ndarray):
        np_array = np.array(x)
        if hasattr(x, 'dtype') and x.dtype == jnp.bfloat16:
            return torch.from_numpy(np_array.astype(np.float32)).to(torch.bfloat16)
        else:
            return torch.from_numpy(np_array)
    elif hasattr(x, 'sample'):
        sample = to_torch_recursive(x.sample)
        if hasattr(x, 'replace'):
            return x.replace(sample=sample)
        else:
            return sample
    elif isinstance(x, (list, tuple)):
        return type(x)(to_torch_recursive(xx) for xx in x)
    elif isinstance(x, dict):
        return {k: to_torch_recursive(v) for k, v in x.items()}
    else:
        return x


def to_jax_recursive(x):
    """Recursively convert PyTorch tensors to JAX arrays."""
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.bfloat16:
            return jnp.array(x.detach().to(torch.float32).cpu().numpy()).astype(jnp.bfloat16)
        else:
            return jnp.array(x.detach().cpu().numpy())
    elif isinstance(x, (list, tuple)):
        return type(x)(to_jax_recursive(xx) for xx in x)
    elif isinstance(x, dict):
        return {k: to_jax_recursive(v) for k, v in x.items()}
    else:
        return x


# ============================================================================
# Video Processing
# ============================================================================

def prepare_video_for_export(video, target_frames):
    """Prepare video tensor for export to file.
    
    Handles CogVideoX VAE output format and converts to saveable video frames.
    
    Args:
        video: Video tensor or numpy array
            - Expected input format: [B, C, T, H, W] (PyTorch format)
            - Or [B, T, H, W, C] (JAX format)
        target_frames: Target number of frames (for validation)
        
    Returns:
        list: float32 numpy array list (0-1 range), suitable for export_to_video
    """
    if isinstance(video, (list, tuple)):
        return [prepare_video_for_export(v, target_frames) for v in video]
    
    if isinstance(video, torch.Tensor):
        video = video.cpu()
        
        # Handle different input formats
        if video.dim() == 5:
            if video.shape[-1] == 3:  # [B, T, H, W, C]
                video = video.permute(0, 4, 1, 2, 3)  # -> [B, C, T, H, W]
            batch_vid = video[0]  # [C, T, H, W]
        elif video.dim() == 4:
            if video.shape[0] == 3:  # [C, T, H, W]
                batch_vid = video
            elif video.shape[-1] == 3:  # [T, H, W, C]
                batch_vid = video.permute(3, 0, 1, 2)  # -> [C, T, H, W]
            else:
                raise ValueError(f"Unexpected 4D video shape: {video.shape}")
        else:
            raise ValueError(f"Unexpected video dimensions: {video.dim()}")
        
        # batch_vid: [C, T, H, W] -> [T, C, H, W]
        batch_vid = batch_vid.permute(1, 0, 2, 3)
        
        # Denormalize: [-1, 1] -> [0, 1]
        batch_vid = (batch_vid * 0.5 + 0.5).clamp(0, 1)
        
        # Convert to [T, H, W, C] for video export
        batch_vid = batch_vid.permute(0, 2, 3, 1)
        
        # Convert to float32 numpy (0-1 range)
        video = batch_vid.float().numpy()
        
        if video.shape[-1] == 1:
            video = np.repeat(video, 3, axis=-1)
            
        return [video[i] for i in range(video.shape[0])]
    
    if isinstance(video, np.ndarray):
        if video.ndim == 5:
            if video.shape[-1] == 3:  # [B, T, H, W, C]
                video = np.transpose(video, (0, 4, 1, 2, 3))
            batch_vid = video[0]  # [C, T, H, W]
        elif video.ndim == 4:
            if video.shape[0] == 3:  # [C, T, H, W]
                batch_vid = video
            elif video.shape[-1] == 3:  # [T, H, W, C]
                batch_vid = np.transpose(video, (3, 0, 1, 2))
            else:
                raise ValueError(f"Unexpected 4D video shape: {video.shape}")
        else:
            raise ValueError(f"Unexpected video dimensions: {video.ndim}")
        
        batch_vid = np.transpose(batch_vid, (1, 0, 2, 3))
        
        if batch_vid.min() < 0:
            batch_vid = np.clip(batch_vid * 0.5 + 0.5, 0, 1)
        
        batch_vid = np.transpose(batch_vid, (0, 2, 3, 1))
        video = batch_vid.astype(np.float32)
        
        if video.shape[-1] == 1:
            video = np.repeat(video, 3, axis=-1)
        
        return [video[i] for i in range(video.shape[0])]
    
    return video


# ============================================================================
# JAX Configuration
# ============================================================================

def setup_jax_cache():
    """Setup JAX compilation cache (important: avoid repeated compilation)."""
    jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax_cache"))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
