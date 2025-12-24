#!/usr/bin/env python3
"""
Wan 2.1 阶段3：VAE Decoder (Pure JAX)

加载 stage2 生成的 latents，使用 VAE 解码为视频。
Pure JAX 版本 - 使用 autoencoder_kl_wan_jax.py

=============================================================================
FLAX NNX → PURE JAX 改造指南 (测试脚本版)
=============================================================================

本文件从 stage3_vae_decoder_flax.py 改造而来。
改动点用 # PURE_JAX: 标记。

主要改动：
1. 导入 autoencoder_kl_wan_jax 替代 autoencoder_kl_wan_flax
2. 移除 flax.nnx 依赖
3. VAEProxy 适配新的纯函数式 API

=============================================================================
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
import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.experimental import mesh_utils
# PURE_JAX: 移除 from flax import nnx
import torch

from diffusers.utils import export_to_video
# PURE_JAX: 导入纯 JAX 版本
from diffusers.models.autoencoders.autoencoder_kl_wan_jax import AutoencoderKLWan
from diffusers.models.autoencoders.vae import DecoderOutput

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
# PURE_JAX: 原 FlaxVAEProxy 类，主要改动：
# 1. 移除 nnx.jit，使用 jax.jit
# 2. 适配新的 vae.decode API

class PureJAXVAEProxy:
    """
    Pure JAX VAE wrapper with PyTorch interface.
    
    PURE_JAX: 与 FlaxVAEProxy 的区别：
    - self._flax_vae → self._vae（不再是 Flax NNX 模块）
    - 使用 jax.jit 包装 decode 函数
    """
    
    def __init__(self, vae, config, enable_jit=True):
        # PURE_JAX: vae 是 AutoencoderKLWan 实例（普通类，非 nnx.Module）
        self._vae = vae
        self.config = config
        self.dtype = torch.bfloat16
        
        # PURE_JAX: 使用 jax.jit 替代 nnx.jit
        if enable_jit:
            self._decode_fn = jax.jit(self._decode_impl)
        else:
            self._decode_fn = self._decode_impl
    
    def _decode_impl(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        PURE_JAX: 纯 JAX decode 实现
        直接调用 vae.decode，无需 nnx.split/merge
        """
        return self._vae.decode(z)
    
    def decode(self, latents, return_dict=True):
        """
        Decode: PyTorch -> JAX -> decode -> PyTorch
        
        PURE_JAX: 与 Flax 版本相同的接口
        """
        # PyTorch (B, C, T, H, W) -> JAX (B, T, H, W, C)
        if latents.dtype == torch.bfloat16:
            latents_np = latents.to(torch.float32).cpu().numpy()
        else:
            latents_np = latents.cpu().numpy()
        
        latents_jax = jnp.array(np.transpose(latents_np, (0, 2, 3, 4, 1)), dtype=jnp.bfloat16)
        
        # PURE_JAX: 直接调用 jitted 函数
        frames_jax = self._decode_fn(latents_jax)
        
        # JAX (B, T, H, W, C) -> PyTorch (B, C, T, H, W)
        frames_np = np.asarray(frames_jax.transpose(0, 4, 1, 2, 3))
        frames_torch = torch.from_numpy(frames_np.astype(np.float32)).to(torch.bfloat16)
        
        if return_dict:
            return DecoderOutput(sample=frames_torch)
        return frames_torch


# ============================================================================
# VAE Functions
# ============================================================================
# PURE_JAX: load_vae 函数改动

def load_vae(model_id, mesh, enable_jit=True):
    """
    Load Pure JAX VAE.
    
    PURE_JAX: 与 Flax 版本的区别：
    1. AutoencoderKLWan.from_pretrained 直接返回可用实例（无需 nnx.jit 包装）
    2. 使用 PureJAXVAEProxy 替代 FlaxVAEProxy
    """
    print(f"加载 VAE (Pure JAX): {model_id}")
    
    # PURE_JAX: from_pretrained 返回的是普通类实例，不是 nnx.Module
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", dtype=jnp.bfloat16
    )
    print("✓ VAE 参数加载完成")
    
    # PURE_JAX: JIT 编译在 Proxy 中完成，不需要 nnx.jit(vae.decoder)
    if enable_jit:
        print("✓ VAE Decoder JIT 编译已启用")
    
    return PureJAXVAEProxy(vae, vae.config, enable_jit=enable_jit)


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
    latents_mean = torch.tensor(vae.config.latents_mean, dtype=latents.dtype).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std, dtype=latents.dtype).view(1, -1, 1, 1, 1)
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
    # PURE_JAX: 描述改为 Pure JAX
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
    print("阶段3：VAE Decoder (Pure JAX)")  # PURE_JAX: 标题改变
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
