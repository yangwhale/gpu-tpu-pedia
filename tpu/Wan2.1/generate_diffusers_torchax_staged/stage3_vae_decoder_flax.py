#!/usr/bin/env python3
"""
Wan 2.1 三阶段生成 - 阶段3：VAE Decoder (Flax)

本阶段负责：
1. 加载阶段2生成的 latents
2. 加载 Flax VAE 模型（FlaxAutoencoderKLWan）
3. 应用缩放因子（使用 latents_mean/latents_std 反归一化）
4. 使用 VAE 解码为视频帧
5. 后处理并导出最终视频

输入文件：
- stage2_latents.safetensors: 生成的 latents
- generation_config.json: 生成配置

输出文件：
- output_video.mp4: 最终生成的视频

注意：这个版本使用 Flax VAE (autoencoder_kl_wan_flax.py)
     如需使用 TorchAx VAE，请使用 stage3_vae_decoder.py
"""

import time
import argparse
import warnings
import logging
import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.experimental import mesh_utils
from flax import nnx
import torch
from diffusers.utils import export_to_video
from diffusers.models.autoencoders.autoencoder_kl_wan_flax import FlaxAutoencoderKLWan
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
# Flax VAE Proxy
# ============================================================================

class FlaxVAEProxy:
    """
    Flax VAE 的 PyTorch 接口代理
    
    将 PyTorch latents 转换为 JAX，使用 Flax VAE 解码，
    然后将结果转回 PyTorch 格式。
    """
    def __init__(self, flax_vae):
        self._flax_vae = flax_vae
        self.config = flax_vae.config
        self.dtype = torch.bfloat16
    
    def __getattr__(self, name):
        return getattr(self._flax_vae, name)
    
    def decode(self, latents, return_dict=True):
        """解码：PyTorch -> JAX -> 解码 -> PyTorch
        
        Wan VAE 使用 latents_mean 和 latents_std 进行归一化。
        输入的 latents 需要先反归一化再送入 decoder。
        
        Args:
            latents: PyTorch tensor [B, C, T, H, W]
                     其中 C=16(latent_channels)
            return_dict: 是否返回 DecoderOutput
            
        Returns:
            DecoderOutput 或 torch.Tensor
        """
        # 转换 latents: PyTorch (B, C, T, H, W) -> JAX (B, T, H, W, C)
        if latents.dtype == torch.bfloat16:
            latents_np = latents.to(torch.float32).cpu().numpy()
        else:
            latents_np = latents.cpu().numpy()
        
        # (B, C, T, H, W) -> (B, T, H, W, C)
        latents_transposed = np.transpose(latents_np, (0, 2, 3, 4, 1))
        latents_jax = jnp.array(latents_transposed, dtype=jnp.bfloat16)
        
        # Flax VAE 解码
        frames_jax = self._flax_vae.decode(latents_jax)
        
        # 转换输出: JAX (B, T, H, W, C) -> PyTorch (B, C, T, H, W)
        frames_jax_transposed = frames_jax.transpose(0, 4, 1, 2, 3)
        frames_np = np.asarray(frames_jax_transposed)
        frames_torch = torch.from_numpy(frames_np.astype(np.float32)).to(torch.bfloat16)
        
        if return_dict:
            return DecoderOutput(sample=frames_torch)
        return frames_torch


# ============================================================================
# VAE 加载与配置
# ============================================================================

def load_flax_vae(model_id, mesh=None, dtype=jnp.bfloat16, enable_jit=True):
    """
    加载 Flax VAE 模型，支持 TPU 分片和 JIT 编译
    
    Args:
        model_id: 模型 ID 或路径
        mesh: JAX Mesh 对象，用于 TPU 分片
        dtype: 模型数据类型
        enable_jit: 是否启用 JIT 编译
        
    Returns:
        FlaxVAEProxy: 包装后的 Flax VAE
    """
    print(f"\n加载 Flax VAE: {model_id}")
    
    flax_vae = FlaxAutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        dtype=dtype
    )
    
    print("✓ Flax VAE 加载完成")
    print(f"  z_dim: {flax_vae.config.z_dim}")
    print(f"  base_dim: {flax_vae.config.base_dim}")
    print(f"  dim_mult: {flax_vae.config.dim_mult}")
    
    # 如果提供了 mesh 且启用 JIT，使用 nnx.jit 编译 decoder
    if mesh is not None and enable_jit:
        print(f"  配置 TPU 分片和 JIT 编译...")
        print(f"  Mesh: {mesh}")
        flax_vae.decoder = nnx.jit(flax_vae.decoder)
        print("  ✓ Decoder JIT 编译完成")
    else:
        print("  注意：未启用 JIT 编译（无 mesh 或 enable_jit=False）")
    
    return FlaxVAEProxy(flax_vae)


# ============================================================================
# 解码函数
# ============================================================================

def run_vae_decode(vae_proxy, latents, desc="VAE Decode"):
    """
    运行一次 VAE 解码
    
    Args:
        vae_proxy: FlaxVAEProxy 实例
        latents: latent tensor [B, C, T, H, W]
        desc: 描述信息
    
    Returns:
        (video, elapsed_time)
    """
    start_time = time.perf_counter()
    
    print(f"\n{desc}...")
    output = vae_proxy.decode(latents, return_dict=True)
    jax.effects_barrier()
    
    elapsed = time.perf_counter() - start_time
    print(f"✓ {desc} 完成，耗时: {elapsed:.2f} 秒")
    
    return output.sample, elapsed


def decode_latents_to_video(vae_proxy, latents, config, warmup=True):
    """
    使用 VAE 解码 latents 为视频帧
    
    Args:
        vae_proxy: FlaxVAEProxy 实例
        latents: latents tensor [B, C, T, H, W]
        config: 生成配置
        warmup: 是否运行预热解码
        
    Returns:
        video: 解码后的视频
        elapsed: 解码耗时（不含预热）
    """
    print(f"\n=== 阶段3：VAE 解码 [Flax 版本] ===")
    print(f"输入 latents shape: {latents.shape}, dtype: {latents.dtype}")
    
    # 检查并处理 nan 值
    nan_count = torch.isnan(latents).sum().item()
    if nan_count > 0:
        print(f"警告：发现 {nan_count} 个 nan 值，将替换为 0")
        latents = torch.nan_to_num(latents, nan=0.0)
    
    # Wan VAE 使用 latents_mean 和 latents_std 进行归一化
    # 解码前需要反归一化: latents = latents * latents_std + latents_mean
    latents_mean = torch.tensor(vae_proxy.config.latents_mean, dtype=latents.dtype)
    latents_std = torch.tensor(vae_proxy.config.latents_std, dtype=latents.dtype)
    
    # 调整形状以便广播 [1, C, 1, 1, 1]
    latents_mean = latents_mean.view(1, -1, 1, 1, 1)
    latents_std = latents_std.view(1, -1, 1, 1, 1)
    
    latents = latents * latents_std + latents_mean
    
    # 转换为 bfloat16
    latents = latents.to(torch.bfloat16)
    
    # VAE 解码
    
    # 预热运行 (触发 JIT 编译)
    if warmup:
        _, _ = run_vae_decode(vae_proxy, latents, desc="Warmup VAE (JIT)")
    
    # 正式解码
    video, decode_time = run_vae_decode(vae_proxy, latents, desc="VAE Decode")
    
    print(f"输出 video shape: {video.shape}")
    
    return video, decode_time


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Wan 2.1 阶段3：VAE Decoder (Flax)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 基本用法（使用阶段2的输出）
  python stage3_vae_decoder_flax.py
  
  # 指定输入目录
  python stage3_vae_decoder_flax.py --input_dir ./my_outputs
  
  # 指定输出视频路径
  python stage3_vae_decoder_flax.py --output_video my_video.mp4
        """
    )
    
    parser.add_argument('--input_dir', type=str, default='./stage_outputs')
    parser.add_argument('--output_video', type=str, default=None)
    parser.add_argument('--model_id', type=str, default=None)
    parser.add_argument('--fps', type=int, default=None)
    parser.add_argument('--dp', type=int, default=1, help='Data parallelism dimension')
    parser.add_argument('--warmup', action='store_true', default=True,
                        help='运行预热解码触发 JIT 编译（默认启用）')
    parser.add_argument('--no_warmup', action='store_false', dest='warmup',
                        help='禁用预热解码')
    parser.add_argument('--no_jit', action='store_true', default=False,
                        help='禁用 JIT 编译（调试用）')
    
    args = parser.parse_args()
    
    # 设置 JAX 编译缓存
    jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
    
    paths = get_default_paths(args.input_dir)
    
    warnings.filterwarnings('ignore')
    logging.getLogger().setLevel(logging.ERROR)
    
    print(f"\n{'='*60}")
    print("Wan 2.1 阶段3：VAE Decoder (Flax)")
    print(f"{'='*60}")
    
    # 加载配置
    print(f"\n加载配置: {paths['config']}")
    config = load_generation_config(paths['config'])
    
    # 应用命令行覆盖
    model_id = args.model_id or config.get('model_id', MODEL_NAME)
    fps = args.fps or config.get('fps', FPS)
    target_frames = config.get('frames', 81)
    
    output_video = args.output_video or paths['video']
    
    print(f"\n配置参数：")
    print(f"  模型: {model_id}")
    print(f"  FPS: {fps}")
    print(f"  目标帧数: {target_frames}")
    
    # 加载 latents
    print(f"\n加载 latents: {paths['latents']}")
    latents, latents_metadata = load_latents_from_safetensors(
        paths['latents'],
        device='cpu',
        restore_dtype=True
    )
    
    print(f"\n设备信息：")
    print(f"  JAX 设备数: {len(jax.devices())}")
    
    # 创建 mesh
    assert len(jax.devices()) % args.dp == 0, f"设备数 {len(jax.devices())} 必须能被 dp={args.dp} 整除"
    tp_dim = len(jax.devices()) // args.dp
    mesh_devices = mesh_utils.create_device_mesh(
        (args.dp, tp_dim), allow_split_physical_axes=True
    )
    mesh = Mesh(mesh_devices, ("dp", "tp"))
    print(f"  Mesh: {mesh}")
    
    # 加载 VAE
    vae_proxy = load_flax_vae(
        model_id,
        mesh=mesh,
        dtype=jnp.bfloat16,
        enable_jit=not args.no_jit
    )
    
    # 在 mesh 上下文中解码
    with mesh:
        video, decode_time = decode_latents_to_video(
            vae_proxy,
            latents,
            config,
            warmup=args.warmup
        )
    
    # 准备视频导出
    print(f"\n准备视频导出...")
    frames = prepare_video_for_export(video, target_frames)
    print(f"后处理后 video shape: {frames.shape}, dtype: {frames.dtype}")
    
    # 导出视频
    print(f"\n导出视频到: {output_video}")
    print(f"FPS: {fps}")
    
    export_to_video(frames, output_video, fps=fps)
    
    print(f"✓ 视频已保存!")
    
    # 统计信息
    print(f"\n=== 生成统计 ===")
    print(f"帧数: {frames.shape[0]}")
    print(f"分辨率: {frames.shape[2]}x{frames.shape[1]}")
    print(f"FPS: {fps}")
    print(f"VAE 解码耗时: {decode_time:.2f} 秒（不含预热）")
    
    print(f"\n{'='*60}")
    print("阶段3 完成！(Flax 版本)")
    print(f"{'='*60}")
    print(f"\n输出视频: {output_video}")
    
    print("\n✓ 视频生成完成！")


if __name__ == "__main__":
    main()
