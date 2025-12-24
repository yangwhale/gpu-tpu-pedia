#!/usr/bin/env python3
"""
Wan 2.1 阶段3：VAE Decoder (Pure JAX)

加载 stage2 生成的 latents，使用 VAE 解码为视频。
纯 JAX 版本 - 使用 autoencoder_kl_wan_jax.py (无 NNX 依赖)

性能优势（相比 NNX 版本）:
- 首次编译时间减少 ~55%
- 缓存加载时间减少 ~64%
- 无 pytree=False hack，完整 JAX 追踪优化
"""

# Suppress warnings FIRST before any imports
import warnings
import logging
import os
warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import argparse
import functools
import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
import torch

from diffusers.utils import export_to_video
from diffusers.models.autoencoders.vae import DecoderOutput

# Import pure JAX VAE
from diffusers.models.autoencoders.autoencoder_kl_wan_jax import (
    AutoencoderKLWan,
    AutoencoderKLWanConfig,
    load_vae_params,
    vae_decode,
)

from utils import (
    MODEL_NAME,
    FPS,
    prepare_video_for_export,
    load_latents_from_safetensors,
    load_generation_config,
    get_default_paths,
)


# ============================================================================
# Pure JAX VAE Proxy
# ============================================================================

class PureJAXVAEProxy:
    """Pure JAX VAE wrapper with PyTorch-compatible interface."""
    
    def __init__(self, params, config, mesh=None, enable_jit=True):
        self.params = params
        self.config = config
        self.mesh = mesh
        self.dtype = torch.bfloat16
        
        # Create JIT-compiled decode function
        if enable_jit:
            self._decode_fn = self._create_jit_decode()
        else:
            self._decode_fn = None
    
    def _create_jit_decode(self):
        """Create JIT-compiled decode function with sharding."""
        
        @functools.partial(jax.jit)
        def decode_fn(params, z):
            return vae_decode(params, z, self.config)
        
        return decode_fn
    
    def decode(self, latents, return_dict=True):
        """Decode: PyTorch -> JAX -> decode -> PyTorch"""
        # PyTorch (B, C, T, H, W) -> JAX (B, T, H, W, C)
        if latents.dtype == torch.bfloat16:
            latents_np = latents.to(torch.float32).cpu().numpy()
        else:
            latents_np = latents.cpu().numpy()
        
        latents_jax = jnp.array(np.transpose(latents_np, (0, 2, 3, 4, 1)), dtype=jnp.bfloat16)
        
        # Decode using JIT-compiled function or direct call
        if self._decode_fn is not None:
            frames_jax = self._decode_fn(self.params, latents_jax)
        else:
            frames_jax = vae_decode(self.params, latents_jax, self.config)
        
        # JAX (B, T, H, W, C) -> PyTorch (B, C, T, H, W)
        frames_np = np.asarray(frames_jax.transpose(0, 4, 1, 2, 3))
        frames_torch = torch.from_numpy(frames_np.astype(np.float32)).to(torch.bfloat16)
        
        if return_dict:
            return DecoderOutput(sample=frames_torch)
        return frames_torch


# ============================================================================
# VAE Functions
# ============================================================================

def load_vae(model_id, mesh, enable_jit=True):
    """Load Pure JAX VAE."""
    print(f"加载 VAE (Pure JAX): {model_id}")
    
    # Load parameters and config
    params, config = load_vae_params(
        model_id, subfolder="vae", dtype=jnp.bfloat16
    )
    print("✓ VAE 参数加载完成")
    
    # Create proxy
    vae = PureJAXVAEProxy(params, config, mesh, enable_jit=enable_jit)
    
    if enable_jit:
        print("✓ VAE Decoder JIT 编译已启用")
    
    return vae


def run_vae_decode(vae, latents, desc="VAE Decode"):
    """Run VAE decode once."""
    start = time.perf_counter()
    print(f"{desc}...")
    
    output = vae.decode(latents, return_dict=True)
    jax.effects_barrier()
    
    elapsed = time.perf_counter() - start
    print(f"✓ {desc}: {elapsed:.2f}s")
    return output.sample, elapsed


def decode_latents(vae, latents, warmup=True):
    """Decode latents to video."""
    print(f"\n=== VAE 解码 (Pure JAX) ===")
    print(f"latents: {latents.shape}, {latents.dtype}")
    
    # Handle nan
    nan_count = torch.isnan(latents).sum().item()
    if nan_count > 0:
        print(f"警告: 发现 {nan_count} 个 nan，替换为 0")
        latents = torch.nan_to_num(latents, nan=0.0)
    
    # Denormalize: x * std + mean
    latents_mean = torch.tensor(list(vae.config.latents_mean), dtype=latents.dtype).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(list(vae.config.latents_std), dtype=latents.dtype).view(1, -1, 1, 1, 1)
    latents = latents * latents_std + latents_mean
    latents = latents.to(torch.bfloat16)
    
    # Warmup
    if warmup:
        run_vae_decode(vae, latents, "Warmup (JIT)")
    
    # Decode
    video, elapsed = run_vae_decode(vae, latents, "VAE Decode")
    print(f"video: {video.shape}")
    
    return video, elapsed


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Wan 2.1 阶段3：VAE Decoder (Pure JAX)')
    parser.add_argument('--input_dir', type=str, default='./stage_outputs')
    parser.add_argument('--output_video', type=str, default=None)
    parser.add_argument('--model_id', type=str, default=None)
    parser.add_argument('--fps', type=int, default=None)
    parser.add_argument('--dp', type=int, default=1)
    parser.add_argument('--no_warmup', action='store_true')
    parser.add_argument('--no_jit', action='store_true')
    args = parser.parse_args()
    
    # Setup JAX cache
    jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    
    paths = get_default_paths(args.input_dir)
    
    print(f"\n{'='*50}")
    print("阶段3：VAE Decoder (Pure JAX)")
    print(f"{'='*50}")
    
    # Load config
    config = load_generation_config(paths['config'])
    model_id = args.model_id or config.get('model_id', MODEL_NAME)
    fps = args.fps or config.get('fps', FPS)
    target_frames = config.get('frames', 81)
    output_video = args.output_video or paths['video']
    
    print(f"\n模型: {model_id}")
    print(f"FPS: {fps}, 帧数: {target_frames}")
    
    # Load latents
    print(f"\n加载 latents: {paths['latents']}")
    latents, _ = load_latents_from_safetensors(paths['latents'], device='cpu', restore_dtype=True)
    
    # Create mesh
    print(f"\nJAX 设备: {len(jax.devices())}")
    tp_dim = len(jax.devices()) // args.dp
    mesh_devices = mesh_utils.create_device_mesh((args.dp, tp_dim), allow_split_physical_axes=True)
    mesh = Mesh(mesh_devices, ("dp", "tp"))
    print(f"Mesh: dp={args.dp}, tp={tp_dim}")
    
    # Load VAE
    print()
    vae = load_vae(model_id, mesh, enable_jit=not args.no_jit)
    
    # Decode
    with mesh:
        video, decode_time = decode_latents(vae, latents, warmup=not args.no_warmup)
    
    # Export
    frames = prepare_video_for_export(video, target_frames)
    
    print(f"\n导出视频: {output_video}")
    export_to_video(frames, output_video, fps=fps)
    
    # Summary
    print(f"\n{'='*50}")
    print(f"✓ 完成! {frames.shape[0]}帧, {frames.shape[2]}x{frames.shape[1]}, {fps}fps")
    print(f"  解码耗时: {decode_time:.2f}s")
    print(f"  输出: {output_video}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
