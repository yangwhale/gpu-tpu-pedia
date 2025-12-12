#!/usr/bin/env python3
"""
CogVideoX 三阶段生成 - 阶段3：VAE Decoder

本阶段负责：
1. 加载阶段2生成的 latents
2. 加载 Flax VAE 模型（FlaxAutoencoderKLCogVideoX）
3. 解码 latents 为视频帧
4. 导出最终视频

输入文件：
- stage2_latents.safetensors: 生成的 latents
- generation_config.json: 生成配置

输出文件：
- output_video.mp4: 最终生成的视频
"""

import os
import sys
import time
import argparse
import warnings
import logging
import numpy as np

import jax
import jax.numpy as jnp
import torch
from diffusers.utils import export_to_video

# Add parent directory to path for FlaxAutoencoderKLCogVideoX
sys.path.insert(0, '/home/chrisya/diffusers-tpu-chris/src')
from diffusers.models.autoencoders.autoencoder_kl_cogvideox_flax import FlaxAutoencoderKLCogVideoX
from diffusers.models.autoencoders.vae import DecoderOutput

from utils import (
    MODEL_NAME,
    FPS,
    to_torch_recursive,
    prepare_video_for_export,
    load_latents_from_safetensors,
    load_generation_config,
    get_default_paths,
)


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
        
        关键优化：全程使用 BF16，避免 FP32 中间数组占用 2倍内存
        
        Args:
            latents: PyTorch tensor - CogVideoX 格式为 (B, T, C, H, W)
                     其中 T=latent_frames, C=16(latent_channels)
            return_dict: 是否返回 DecoderOutput
            
        Returns:
            DecoderOutput 或 torch.Tensor
        """
        # 应用缩放因子
        scaling_factor = self.config.scaling_factor if hasattr(self.config, 'scaling_factor') else 1.15258426
        latents = latents / scaling_factor
        
        # 转换 latents: PyTorch (B, T, C, H, W) -> numpy (B, T, H, W, C)
        # CogVideoX latents 格式: [B, T_latent, C=16, H, W]
        if latents.dtype == torch.bfloat16:
            # BF16: 通过 FP32 中转（JAX 的 bfloat16 限制）
            latents_np = latents.to(torch.float32).cpu().numpy()
        else:
            latents_np = latents.cpu().numpy()
        
        # Transpose: (B, T, C, H, W) -> (B, T, H, W, C)
        # 注意：CogVideoX latents 是 [B, T, C, H, W]，不是 [B, C, T, H, W]
        latents_jax = jnp.array(
            np.transpose(latents_np, (0, 1, 3, 4, 2)),  # (B, T, C, H, W) -> (B, T, H, W, C)
            dtype=jnp.bfloat16
        )
        
        print(f"  VAE 输入 latents shape: {latents_jax.shape}")
        
        # Flax VAE 解码（BF16 -> BF16）
        frames_jax = self._flax_vae.decode(latents_jax)
        
        print(f"  VAE 输出 frames shape: {frames_jax.shape}")
        
        # 转换输出: JAX (B, T, H, W, C) -> PyTorch (B, C, T, H, W)
        frames_jax_transposed = frames_jax.transpose(0, 4, 1, 2, 3)  # (B, C, T, H, W)
        
        # 转为 numpy
        frames_np = np.asarray(frames_jax_transposed)
        
        # 转为 PyTorch（通过 FP32 中转）
        frames_torch = torch.from_numpy(frames_np.astype(np.float32)).to(torch.bfloat16)
        
        # 返回 DecoderOutput
        if return_dict:
            return DecoderOutput(sample=frames_torch)
        return frames_torch


def load_flax_vae(model_id, dtype=jnp.bfloat16):
    """
    加载 Flax VAE 模型
    
    Args:
        model_id: 模型 ID 或路径
        dtype: 模型数据类型
        
    Returns:
        FlaxVAEProxy: 包装后的 Flax VAE
    """
    print(f"\n加载 Flax VAE: {model_id}")
    
    flax_vae = FlaxAutoencoderKLCogVideoX.from_pretrained(
        model_id,
        subfolder="vae",
        dtype=dtype
    )
    
    print(f"  ✓ Flax VAE 已加载")
    
    return FlaxVAEProxy(flax_vae)


def decode_latents_to_video(vae_proxy, latents, config):
    """
    使用 VAE 解码 latents 为视频帧
    
    Args:
        vae_proxy: FlaxVAEProxy 实例
        latents: latents tensor [B, C, T, H, W]
        config: 生成配置
        
    Returns:
        frames: 视频帧列表
        elapsed: 解码耗时
    """
    print(f"\n=== 阶段3：VAE 解码 ===")
    print(f"输入 latents shape: {latents.shape}")
    print(f"输入 latents dtype: {latents.dtype}")
    
    # VAE decode
    print("\n开始 VAE 解码...")
    start_time = time.perf_counter()
    
    output = vae_proxy.decode(latents, return_dict=True)
    jax.effects_barrier()
    
    elapsed = time.perf_counter() - start_time
    print(f"✓ VAE 解码完成，耗时: {elapsed:.2f} 秒")
    
    # 提取 sample
    frames = output.sample
    
    print(f"输出 frames shape: {frames.shape}")
    print(f"输出 frames dtype: {frames.dtype}")
    
    return frames, elapsed


def main():
    parser = argparse.ArgumentParser(
        description='CogVideoX 阶段3：VAE Decoder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 基本用法（使用阶段2的输出）
  python stage3_vae_decoder.py
  
  # 指定输入目录
  python stage3_vae_decoder.py --input_dir ./my_outputs
  
  # 指定输出视频路径
  python stage3_vae_decoder.py --output_video my_video.mp4
        """
    )
    
    parser.add_argument(
        '--input_dir', type=str, default='./stage_outputs',
        help='Input directory containing stage2 outputs (default: ./stage_outputs)'
    )
    parser.add_argument(
        '--output_video', type=str, default=None,
        help='Output video path (default: stage_outputs/output_video.mp4)'
    )
    
    # VAE 配置
    parser.add_argument(
        '--model_id', type=str, default=None,
        help='Override model ID for VAE'
    )
    
    # 视频输出配置
    parser.add_argument(
        '--fps', type=int, default=None,
        help='Output video FPS (default: from config)'
    )
    
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
    print("CogVideoX 阶段3：VAE Decoder")
    print(f"{'='*60}")
    
    # 加载配置
    print(f"\n加载配置: {paths['config']}")
    config = load_generation_config(paths['config'])
    
    # 应用命令行覆盖
    model_id = args.model_id or config.get('model_id', MODEL_NAME)
    fps = args.fps or config.get('fps', FPS)
    target_frames = config.get('frames', 81)
    
    output_video = args.output_video or paths['video']
    
    # 加载 latents
    print(f"\n加载 latents: {paths['latents']}")
    latents, latents_metadata = load_latents_from_safetensors(
        paths['latents'],
        device='cpu',
        restore_dtype=True
    )
    
    print(f"总设备数: {len(jax.devices())}")
    
    # 加载 VAE
    vae_proxy = load_flax_vae(model_id, dtype=jnp.bfloat16)
    
    # 解码
    video, decode_time = decode_latents_to_video(
        vae_proxy,
        latents,
        config
    )
    
    # 准备视频导出
    print(f"\n准备视频导出...")
    frames = prepare_video_for_export(video, target_frames)
    
    if isinstance(frames, list):
        print(f"后处理后帧数: {len(frames)}")
        if len(frames) > 0:
            print(f"每帧 shape: {frames[0].shape}")
    else:
        print(f"后处理后 video shape: {frames.shape}")
    
    # 导出视频
    print(f"\n导出视频到: {output_video}")
    print(f"FPS: {fps}")
    
    export_to_video(frames, output_video, fps=fps)
    
    print(f"✓ 视频已保存!")
    
    # 统计信息
    print(f"\n=== 生成统计 ===")
    if isinstance(frames, list):
        print(f"帧数: {len(frames)}")
        if len(frames) > 0:
            print(f"分辨率: {frames[0].shape[1]}x{frames[0].shape[0]}")
    print(f"FPS: {fps}")
    print(f"VAE 解码耗时: {decode_time:.2f} 秒")
    
    print(f"\n{'='*60}")
    print("阶段3 完成！")
    print(f"{'='*60}")
    print(f"\n输出视频: {output_video}")
    
    print("\n✓ 视频生成完成！")


if __name__ == "__main__":
    main()