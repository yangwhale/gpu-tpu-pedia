#!/usr/bin/env python3
"""
HunyuanVideo-1.5 三阶段生成 - 阶段3：VAE Decoder

本阶段负责：
1. 加载阶段2生成的 latents
2. 加载 Flax VAE 模型
3. 解码 latents 为视频帧
4. 导出最终视频

输入文件：
- stage2_latents.safetensors: 生成的 latents
- generation_config.json: 生成配置

输出文件：
- output_video.mp4: 最终生成的视频
"""

import os
import time
import argparse
import warnings
import logging
import numpy as np
import jax
import jax.numpy as jnp
import torch
from PIL import Image

from diffusers.utils import export_to_video
from diffusers.models.autoencoders.autoencoder_kl_hunyuanvideo15_flax import (
    FlaxAutoencoderKLHunyuanVideo15,
)

from safetensors.torch import save_file as save_safetensors
from utils import (
    MODEL_NAME,
    load_latents_from_safetensors,
    load_generation_config,
    get_default_paths,
)


def load_flax_vae(model_id=MODEL_NAME, dtype=jnp.bfloat16):
    """
    加载 Flax VAE 模型
    
    Args:
        model_id: HuggingFace 模型 ID
        dtype: 数据类型
        
    Returns:
        Flax VAE 模型实例
    """
    print("\n加载 Flax VAE 模型...")
    
    flax_vae = FlaxAutoencoderKLHunyuanVideo15.from_pretrained(
        model_id,
        subfolder="vae",
        dtype=dtype,
    )
    
    print(f"  ✓ Flax VAE 已加载")
    print(f"  数据类型: {dtype}")
    
    return flax_vae


def decode_latents_to_video(vae, latents, config, enable_tiling=True):
    """
    使用 VAE 解码 latents 为视频帧
    
    Args:
        vae: Flax VAE 模型
        latents: latents tensor [B, C, T, H, W]
        config: 生成配置
        enable_tiling: 是否启用 tiling
        
    Returns:
        frames: PIL Image 列表
        raw_output: 原始 VAE 输出 (JAX array, [-1, 1] 范围)
        elapsed: 解码耗时
    """
    print(f"\n=== 阶段3：VAE 解码 ===")
    print(f"输入 latents shape: {latents.shape}")
    print(f"输入 latents dtype: {latents.dtype}")
    print(f"Tiling: {'启用' if enable_tiling else '禁用'}")
    
    # 转换为 JAX array
    if isinstance(latents, torch.Tensor):
        if latents.dtype == torch.bfloat16:
            jax_latents = jnp.array(latents.to(torch.float32).cpu().numpy()).astype(jnp.bfloat16)
        else:
            jax_latents = jnp.array(latents.cpu().numpy())
    else:
        jax_latents = latents
    
    print(f"JAX latents shape: {jax_latents.shape}, dtype: {jax_latents.dtype}")
    
    # 从 VAE config 读取 scaling factor
    # 注意：Pipeline 在返回 latents 时已经应用了 scaling，我们需要 unscale
    scaling_factor = vae.config.scaling_factor
    print(f"VAE scaling_factor: {scaling_factor}")
    
    # unscale latents (Pipeline 在输出 latents 前乘以了 1/scaling_factor)
    # 所以这里需要乘以 scaling_factor 来还原
    # 实际上不需要手动处理，因为 Flax VAE 的 decode 内部会处理
    # 这里不做任何 scaling 处理，让 Flax VAE 自己处理
    print(f"Latents range (无 scaling 修改): [{float(jnp.min(jax_latents)):.4f}, {float(jnp.max(jax_latents)):.4f}]")
    
    # 转换格式: [B, C, T, H, W] -> [B, T, H, W, C]
    if jax_latents.ndim == 5 and jax_latents.shape[1] == 32:  # C=32 是 latent channels
        print("转换 BCTHW -> BTHWC...")
        jax_latents = jnp.transpose(jax_latents, (0, 2, 3, 4, 1))
        print(f"转换后 shape: {jax_latents.shape}")
    
    # 启用 tiling
    if enable_tiling:
        print("启用 VAE Tiling...")
        vae.enable_tiling(
            tile_sample_min_height=256,
            tile_sample_min_width=256,
            tile_latent_min_height=16,
            tile_latent_min_width=16,
            tile_overlap_factor=0.25,
        )
    
    # 解码
    print("\n开始 VAE 解码...")
    start_time = time.perf_counter()
    
    output = vae.decode(jax_latents)
    
    elapsed = time.perf_counter() - start_time
    print(f"✓ VAE 解码完成，耗时: {elapsed:.2f} 秒")
    
    print(f"输出 shape: {output.shape}, dtype: {output.dtype}")
    print(f"输出 range: [{float(jnp.min(output)):.4f}, {float(jnp.max(output)):.4f}]")
    
    # 转换格式: [B, T, H, W, C] -> frames list
    # output 是 [B, T, H, W, C]，需要取 batch=0，然后逐帧处理
    output_np = np.array(output[0])  # [T, H, W, C]
    
    # 将值范围从 [-1, 1] 转换到 [0, 255]
    output_np = (output_np + 1.0) / 2.0  # [-1,1] -> [0,1]
    output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)
    
    # 转换为 PIL Image 列表
    frames = []
    for i in range(output_np.shape[0]):
        frame = Image.fromarray(output_np[i])
        frames.append(frame)
    
    print(f"生成 {len(frames)} 帧，分辨率: {frames[0].size}")
    
    # 返回原始 output 用于调试（output 已经是 [B, T, H, W, C]）
    return frames, output, elapsed


def main():
    parser = argparse.ArgumentParser(
        description='HunyuanVideo-1.5 阶段3：VAE Decoder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 基本用法（使用阶段2的输出，默认禁用 tiling 以获得最佳性能）
  python stage3_vae_decoder.py
  
  # 指定输入目录
  python stage3_vae_decoder.py --input_dir ./my_outputs
  
  # 启用 tiling（用于内存受限的情况）
  python stage3_vae_decoder.py --enable_tiling
  
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
        '--enable_tiling', action='store_true',
        help='Enable VAE tiling (disabled by default for best performance on TPU)'
    )
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
    jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax_cache"))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
    
    paths = get_default_paths(args.input_dir)
    
    warnings.filterwarnings('ignore', message='.*dtype.*int64.*truncated to dtype int32.*')
    warnings.filterwarnings('ignore')
    logging.getLogger().setLevel(logging.ERROR)
    
    print(f"\n{'='*60}")
    print("HunyuanVideo-1.5 阶段3：VAE Decoder")
    print(f"{'='*60}")
    
    # 加载配置
    print(f"\n加载配置: {paths['config']}")
    config = load_generation_config(paths['config'])
    
    # 应用命令行覆盖
    model_id = args.model_id or config.get('model_id', MODEL_NAME)
    fps = args.fps or config.get('fps', 24)
    # 默认禁用 tiling（TPU 上性能更好），使用 --enable_tiling 开启
    enable_tiling = args.enable_tiling or config.get('enable_vae_tiling', False)
    
    output_video = args.output_video or paths['video']
    
    # 加载 latents
    print(f"\n加载 latents: {paths['latents']}")
    latents, latents_metadata = load_latents_from_safetensors(
        paths['latents'],
        device='cpu',
        restore_dtype=True
    )
    
    # 加载 VAE
    vae = load_flax_vae(model_id=model_id, dtype=jnp.bfloat16)
    
    # 解码
    frames, raw_output, decode_time = decode_latents_to_video(
        vae,
        latents,
        config,
        enable_tiling=enable_tiling
    )
    
    # 保存原始 frames tensor 用于调试
    frames_tensor_path = os.path.join(args.input_dir, 'stage3_frames.safetensors')
    print(f"\n保存原始 frames tensor 到: {frames_tensor_path}")
    
    # 转换 JAX array 到 torch tensor (bfloat16 -> float32 -> bfloat16)
    if raw_output.dtype == jnp.bfloat16:
        # JAX bfloat16 -> float32 -> numpy -> torch bfloat16
        raw_output_np = np.array(raw_output.astype(jnp.float32))
        raw_output_torch = torch.from_numpy(raw_output_np).to(torch.bfloat16)
    else:
        raw_output_np = np.array(raw_output)
        raw_output_torch = torch.from_numpy(raw_output_np)
    
    # 确保 tensor 是 contiguous 的
    raw_output_torch = raw_output_torch.contiguous()
    
    # 保存为 safetensors
    save_safetensors({'frames': raw_output_torch}, frames_tensor_path)
    print(f"  Frames tensor shape: {raw_output_torch.shape}")
    print(f"  Frames tensor dtype: {raw_output_torch.dtype}")
    print(f"  Frames tensor range: [{raw_output_torch.min():.4f}, {raw_output_torch.max():.4f}]")
    print(f"  ✓ 原始 frames tensor 已保存")
    
    # 导出视频
    print(f"\n导出视频到: {output_video}")
    print(f"FPS: {fps}")
    
    export_to_video(frames, output_video, fps=fps)
    
    print(f"✓ 视频已保存!")
    
    # 统计信息
    print(f"\n=== 生成统计 ===")
    print(f"帧数: {len(frames)}")
    print(f"分辨率: {frames[0].size[0]}x{frames[0].size[1]}")
    print(f"FPS: {fps}")
    print(f"时长: {len(frames)/fps:.2f} 秒")
    print(f"VAE 解码耗时: {decode_time:.2f} 秒")
    
    print(f"\n{'='*60}")
    print("阶段3 完成！")
    print(f"{'='*60}")
    print(f"\n输出视频: {output_video}")
    
    print("\n✓ 视频生成完成！")


if __name__ == "__main__":
    main()