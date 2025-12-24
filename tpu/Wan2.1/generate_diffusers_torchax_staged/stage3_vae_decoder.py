#!/usr/bin/env python3
"""
Wan 2.1 阶段3：VAE Decoder (TorchAx)

加载 stage2 生成的 latents，使用 VAE 解码为视频。
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

import jax
import torch
from jax.sharding import Mesh
from jax.experimental import mesh_utils

import torchax
from torchax.ops import ops_registry, jaten

from diffusers.utils import export_to_video
from diffusers.models.autoencoders.autoencoder_kl_wan_torchax import AutoencoderKLWan

from utils import (
    MODEL_NAME,
    FPS,
    DEFAULT_DP,
    VAE_DECODER_SHARDINGS,
    shard_weight_dict,
    prepare_video_for_export,
    load_latents_from_safetensors,
    load_generation_config,
    get_default_paths,
    setup_jax_cache,
    setup_pytree_registrations,
)


# ============================================================================
# TorchAx Helpers
# ============================================================================

def torch_conv2d_jax(input, weight, bias=None, stride=1, padding=0,
                     dilation=1, groups=1, *, env):
    """JAX-compatible conv2d."""
    jinput, jweight, jbias = env.t2j_iso((input, weight, bias))
    res = jaten._aten_conv2d(jinput, jweight, jbias, stride, padding, dilation, groups)
    return env.j2t_iso(res)


def override_op_definition(env, op_to_override, op_impl):
    """Override operator definition."""
    env._ops[op_to_override] = ops_registry.Operator(
        op_to_override, op_impl,
        is_jax_function=False, is_user_defined=True,
        needs_env=False, is_view_op=False,
    )


def move_module_to_xla(env, module):
    """Move module weights to XLA."""
    with jax.default_device("cpu"):
        state_dict = module.state_dict()
        state_dict = env.to_xla(state_dict)
        module.load_state_dict(state_dict, assign=True)


# ============================================================================
# VAE Functions
# ============================================================================

def load_vae(model_id):
    """Load VAE (before enabling torchax)."""
    print(f"加载 VAE: {model_id}")
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.bfloat16
    )
    print("✓ VAE 加载完成")
    return vae


def setup_vae_for_tpu(vae, mesh, env):
    """Setup VAE for TPU execution."""
    print("配置 VAE Decoder...")
    
    # Register conv2d
    override_op_definition(
        env, torch.nn.functional.conv2d,
        functools.partial(torch_conv2d_jax, env=env)
    )
    
    # Move to XLA and compile
    move_module_to_xla(env, vae)
    vae.decoder = torchax.compile(vae.decoder)
    vae.decoder.params = shard_weight_dict(vae.decoder.params, VAE_DECODER_SHARDINGS, mesh)
    vae.decoder.buffers = shard_weight_dict(vae.decoder.buffers, VAE_DECODER_SHARDINGS, mesh)
    
    print("✓ VAE Decoder JIT 编译完成")
    return vae


def denormalize_latents(latents, vae):
    """Denormalize latents: x * std + mean"""
    latents_mean = getattr(vae.config, 'latents_mean', None)
    latents_std = getattr(vae.config, 'latents_std', None)
    
    if latents_mean is None or latents_std is None:
        return latents
    
    mean = torch.tensor(latents_mean).view(1, 16, 1, 1, 1).to(latents.device, latents.dtype)
    std = torch.tensor(latents_std).view(1, 16, 1, 1, 1).to(latents.device, latents.dtype)
    return latents * std + mean


def run_vae_decode(vae, latents, env, desc="VAE Decode"):
    """Run VAE decode once."""
    start = time.perf_counter()
    print(f"{desc}...")
    
    with torch.no_grad():
        video = vae.decode(latents).sample
    jax.effects_barrier()
    
    elapsed = time.perf_counter() - start
    print(f"✓ {desc}: {elapsed:.2f}s")
    return video, elapsed


def decode_latents(vae, latents, env, warmup=True):
    """Decode latents to video."""
    print(f"\n=== VAE 解码 ===")
    print(f"latents: {latents.shape}, {latents.dtype}")
    
    # Handle nan
    nan_count = torch.isnan(latents).sum().item()
    if nan_count > 0:
        print(f"警告: 发现 {nan_count} 个 nan，替换为 0")
        latents = torch.nan_to_num(latents, nan=0.0)
    
    # Convert and denormalize
    latents = latents.to(vae.dtype)
    latents = env.to_xla(latents)
    latents = denormalize_latents(latents, vae)
    
    # Warmup
    if warmup:
        run_vae_decode(vae, latents, env, "Warmup (JIT)")
    
    # Decode
    video, elapsed = run_vae_decode(vae, latents, env, "VAE Decode")
    print(f"video: {video.shape}")
    
    return video, elapsed


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Wan 2.1 阶段3：VAE Decoder (TorchAx)')
    parser.add_argument('--input_dir', type=str, default='./stage_outputs')
    parser.add_argument('--output_video', type=str, default=None)
    parser.add_argument('--model_id', type=str, default=None)
    parser.add_argument('--fps', type=int, default=None)
    parser.add_argument('--dp', type=int, default=DEFAULT_DP)
    parser.add_argument('--no_warmup', action='store_true')
    args = parser.parse_args()
    
    # Setup
    setup_jax_cache()
    setup_pytree_registrations()
    torch.set_default_dtype(torch.bfloat16)
    
    paths = get_default_paths(args.input_dir)
    
    print(f"\n{'='*50}")
    print("阶段3：VAE Decoder (TorchAx)")
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
    
    # Load VAE (before torchax)
    print()
    vae = load_vae(model_id)
    
    # Enable torchax
    print(f"\nJAX 设备: {len(jax.devices())}")
    torchax.enable_globally()
    env = torchax.default_env()
    
    # Create mesh
    tp_dim = len(jax.devices()) // args.dp
    mesh_devices = mesh_utils.create_device_mesh((args.dp, tp_dim), allow_split_physical_axes=True)
    mesh = Mesh(mesh_devices, ("dp", "tp"))
    print(f"Mesh: dp={args.dp}, tp={tp_dim}")
    
    # Setup and decode
    with mesh:
        vae = setup_vae_for_tpu(vae, mesh, env)
        video, decode_time = decode_latents(vae, latents, env, warmup=not args.no_warmup)
    
    # Export
    video = video.to('cpu')
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
