#!/usr/bin/env python3
"""
CogVideoX 三阶段生成 - 阶段3：VAE Decoder (Flax)

本阶段负责：
1. 加载阶段2生成的 latents
2. 加载 Flax VAE 模型（FlaxAutoencoderKLCogVideoX）
3. 应用缩放因子
4. 使用 VAE 解码为视频帧
5. 后处理并导出最终视频

输入文件：
- stage2_latents.safetensors: 生成的 latents
- generation_config.json: 生成配置

输出文件：
- output_video.mp4: 最终生成的视频

注意：这个版本使用 Flax VAE (autoencoder_kl_cogvideox_flax.py)
     如需使用 TorchAx VAE，请使用 stage3_vae_decoder.py
"""

import time
import argparse
import warnings
import logging
import numpy as np

import jax
import jax.numpy as jnp
import torch
from diffusers.utils import export_to_video
from diffusers.models.autoencoders.autoencoder_kl_cogvideox_flax import FlaxAutoencoderKLCogVideoX
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
        
        完全复制原始 generate_flax.py 的 FlaxVAEProxy.decode() 实现。
        关键优化：全程使用 BF16，避免 FP32 中间数组占用 2倍内存
        
        Args:
            latents: PyTorch tensor [B, C, T, H, W]
                     其中 C=16(latent_channels)
            return_dict: 是否返回 DecoderOutput
            
        Returns:
            DecoderOutput 或 torch.Tensor
        """
        # 转换 latents: PyTorch (B, C, T, H, W) -> numpy (B, T, H, W, C)
        # 关键：保持 BF16 dtype 以节省内存
        if latents.dtype == torch.bfloat16:
            # BF16: 通过 FP32 中转（JAX 的 bfloat16 限制）
            latents_np = latents.to(torch.float32).cpu().numpy()
        else:
            latents_np = latents.cpu().numpy()
        
        # Transpose 并**直接创建 BF16 数组**（避免 FP32 中间数组）
        # 关键修复：使用 (0, 2, 3, 4, 1) 而不是 (0, 1, 3, 4, 2)
        # (B, C, T, H, W) -> (B, T, H, W, C)
        latents_jax = jnp.array(
            np.transpose(latents_np, (0, 2, 3, 4, 1)),
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


# ============================================================================
# VAE 加载与配置
# ============================================================================

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
    
    print("✓ Flax VAE 加载完成")
    
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
    
    注意：stage2 已经裁剪了 additional_frames，这里直接解码即可
    
    Args:
        vae_proxy: FlaxVAEProxy 实例
        latents: latents tensor [B, C, T, H, W]（已裁剪 additional_frames）
        config: 生成配置
        warmup: 是否运行预热解码
        
    Returns:
        video: 解码后的视频
        elapsed: 解码耗时（不含预热）
    """
    print(f"\n=== 阶段3：VAE 解码 ===")
    print(f"输入 latents shape: {latents.shape}")
    print(f"输入 latents dtype: {latents.dtype}")
    
    # 检查 nan 值
    latents_float = latents.float()
    nan_count = torch.isnan(latents_float).sum().item()
    total = latents_float.numel()
    print(f"输入 latents nan 统计: {nan_count}/{total} ({nan_count/total*100:.2f}%)")
    
    # 处理 nan 值 - 替换为 0
    if nan_count > 0:
        print(f"警告：发现 {nan_count} 个 nan 值，将替换为 0")
        latents = torch.nan_to_num(latents, nan=0.0)
    
    # 1. 应用缩放因子（CogVideoX 使用 scaling_factor）
    scaling_factor = getattr(vae_proxy.config, 'scaling_factor', 1.15258426)
    print(f"应用缩放因子: 1/{scaling_factor}")
    latents = latents / scaling_factor
    
    # 2. 转换为 VAE dtype
    print("\n转换 latents 到 bfloat16...")
    latents = latents.to(torch.bfloat16)
    
    # 3. VAE 解码
    
    # 预热运行 (触发 JIT 编译)
    warmup_time = 0
    if warmup:
        _, warmup_time = run_vae_decode(vae_proxy, latents, desc="Warmup VAE (JIT)")
    
    # 正式解码
    video, decode_time = run_vae_decode(vae_proxy, latents, desc="VAE Decode")
    elapsed = decode_time  # 只记录正式解码时间
    
    print(f"输出 video shape: {video.shape}")
    print(f"输出 video dtype: {video.dtype}")
    
    # 检查 video 范围
    video_float = video.float()
    print(f"输出 video 范围: min={video_float.min().item():.4f}, max={video_float.max().item():.4f}")
    
    return video, elapsed


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CogVideoX 阶段3：VAE Decoder (Flax)',
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
    parser.add_argument('--warmup', action='store_true', default=True,
                        help='运行预热解码触发 JIT 编译（默认启用）')
    parser.add_argument('--no_warmup', action='store_false', dest='warmup',
                        help='禁用预热解码')
    
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
    print("CogVideoX 阶段3：VAE Decoder (Flax)")
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
    
    # 加载 VAE
    vae_proxy = load_flax_vae(model_id, dtype=jnp.bfloat16)
    
    # 解码
    video, decode_time = decode_latents_to_video(
        vae_proxy,
        latents,
        config,
        warmup=args.warmup
    )
    
    # 准备视频导出
    print(f"\n准备视频导出...")
    frames = prepare_video_for_export(video, target_frames)
    print(f"帧数: {len(frames)}, 每帧 shape: {frames[0].shape if frames else 'N/A'}")
    
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
    print(f"VAE 解码耗时: {decode_time:.2f} 秒（不含预热）")
    
    print(f"\n{'='*60}")
    print("阶段3 完成！(Flax 版本)")
    print(f"{'='*60}")
    print(f"\n输出视频: {output_video}")
    
    print("\n✓ 视频生成完成！")


if __name__ == "__main__":
    main()
