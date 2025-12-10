#!/usr/bin/env python3
"""
HunyuanVideo-1.5 三阶段生成 - 阶段3：VAE Decoder (GPU 版本)

本阶段负责：
1. 加载阶段2生成的 latents
2. 加载 VAE 模型
3. 解码 latents 为视频帧
4. 导出最终视频

关键特性：
- 支持多卡并行 VAE tiling (enable_tile_parallelism)
- 使用 fp16 以节省显存
- 需要使用 torchrun 启动以利用多卡

输入文件：
- stage2_latents.safetensors: 生成的 latents
- generation_config.json: 生成配置

输出文件：
- output_video.mp4: 最终生成的视频

用于 GPU H100 8卡环境
"""

import os
import sys
import time
import argparse
import torch

# 添加 HunyuanVideo-1.5-TPU 到路径
HUNYUAN_ROOT = os.path.expanduser("~/HunyuanVideo-1.5-TPU")
if HUNYUAN_ROOT not in sys.path:
    sys.path.insert(0, HUNYUAN_ROOT)

from hyvideo.models.autoencoders import hunyuanvideo_15_vae
from hyvideo.commons.parallel_states import initialize_parallel_state, get_parallel_state

from utils import (
    load_latents_from_safetensors,
    load_generation_config,
    save_generation_config,
    get_default_paths,
    print_rank0,
    is_main_process,
    save_video,
)


def decode_latents_to_video(vae, latents, config, enable_tiling=True, enable_tile_parallelism=True):
    """
    使用 VAE 解码 latents 为视频帧
    
    Args:
        vae: VAE 模型
        latents: latents tensor [B, C, T, H, W]
        config: 生成配置
        enable_tiling: 是否启用 tiling
        enable_tile_parallelism: 是否启用多卡并行 tiling
        
    Returns:
        video_frames: 解码后的视频帧 tensor [B, C, T, H, W]，值范围 [0, 1]
                     注意：如果启用了 tile_parallelism，只有 rank 0 返回完整结果
        elapsed: 解码耗时
    """
    print_rank0(f"\n=== 阶段3：VAE 解码 ===")
    print_rank0(f"输入 latents shape: {latents.shape}")
    print_rank0(f"输入 latents dtype: {latents.dtype}")
    print_rank0(f"Tiling: {'启用' if enable_tiling else '禁用'}")
    print_rank0(f"Tile Parallelism: {'启用' if enable_tile_parallelism else '禁用'}")
    
    # 确保 latents 在正确的设备和类型上
    device = next(vae.parameters()).device if hasattr(vae, 'parameters') else 'cuda'
    
    # 检查并调整维度
    if latents.ndim == 4:
        latents = latents.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]
    elif latents.ndim != 5:
        raise ValueError(f"Latents must have 4 or 5 dimensions, got {latents.ndim}")
    
    # 应用 VAE scaling factor
    scaling_factor = vae.config.scaling_factor if hasattr(vae.config, 'scaling_factor') else 0.476986
    shift_factor = vae.config.shift_factor if hasattr(vae.config, 'shift_factor') else None
    
    print_rank0(f"VAE scaling_factor: {scaling_factor}")
    if shift_factor is not None:
        print_rank0(f"VAE shift_factor: {shift_factor}")
    
    # Unscale latents
    if shift_factor is not None:
        latents = latents / scaling_factor + shift_factor
    else:
        latents = latents / scaling_factor
    
    print_rank0(f"Latents range (after unscaling): [{latents.min():.4f}, {latents.max():.4f}]")
    
    # 移动到设备，使用 fp16
    latents = latents.to(device=device, dtype=torch.float16)
    
    # 启用 tiling
    if enable_tiling:
        print_rank0("启用 VAE Tiling...")
        vae.enable_tiling()
    
    # 启用多卡并行 tiling（关键：使用 SP 分布式解码）
    if enable_tile_parallelism and hasattr(vae, 'enable_tile_parallelism'):
        sp_state = get_parallel_state()
        if sp_state.sp_enabled:
            print_rank0(f"启用 Tile Parallelism (SP size: {sp_state.sp})")
            vae.enable_tile_parallelism()
        else:
            print_rank0("SP 未启用，跳过 Tile Parallelism")
    
    # 解码
    print_rank0("\n开始 VAE 解码...")
    start_time = time.perf_counter()
    
    # 关键：使用 no_grad 避免梯度累积导致 OOM
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        video_frames = vae.decode(latents, return_dict=False, generator=None)[0]
    
    elapsed = time.perf_counter() - start_time
    print_rank0(f"✓ VAE 解码完成，耗时: {elapsed:.2f} 秒")
    
    # 禁用 tiling
    if enable_tiling:
        vae.disable_tiling()
    
    # 禁用 tile parallelism
    if enable_tile_parallelism and hasattr(vae, 'disable_tile_parallelism'):
        vae.disable_tile_parallelism()
    
    # 检查是否有有效输出（tile_parallelism 模式下只有 rank 0 有完整结果）
    if video_frames.numel() == 0:
        print_rank0("注意：当前 rank 无输出（tile parallelism 模式下正常）")
        return None, elapsed
    
    print_rank0(f"输出 shape: {video_frames.shape}, dtype: {video_frames.dtype}")
    print_rank0(f"输出 range: [{video_frames.min():.4f}, {video_frames.max():.4f}]")
    
    # 归一化到 [0, 1]
    video_frames = (video_frames / 2 + 0.5).clamp(0, 1)
    
    print_rank0(f"归一化后 range: [{video_frames.min():.4f}, {video_frames.max():.4f}]")
    
    return video_frames, elapsed


def main():
    parser = argparse.ArgumentParser(
        description='HunyuanVideo-1.5 阶段3：VAE Decoder (GPU)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 基本用法（使用阶段2的输出）
  python stage3_vae_decoder.py
  
  # 指定输入目录
  python stage3_vae_decoder.py --input_dir ./my_outputs
  
  # 禁用 tiling（需要更多 GPU 内存）
  python stage3_vae_decoder.py --disable_tiling
  
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
        '--disable_tiling', action='store_true',
        help='Disable VAE tiling (requires more GPU memory)'
    )
    parser.add_argument(
        '--save_frames', action='store_true',
        help='Save raw frames tensor for debugging'
    )
    
    # 视频输出配置
    parser.add_argument(
        '--fps', type=int, default=24,
        help='Output video FPS (default: 24)'
    )
    parser.add_argument(
        '--disable_tile_parallelism', action='store_true',
        help='Disable multi-GPU tile parallelism'
    )
    
    args = parser.parse_args()
    
    # 初始化并行状态（使用所有可用 GPU 进行并行 tiling）
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
    
    parallel_dims = initialize_parallel_state(sp=world_size)
    print_rank0(f"并行状态: sp_enabled={parallel_dims.sp_enabled}, sp_size={parallel_dims.sp}")
    
    paths = get_default_paths(args.input_dir)
    
    print_rank0(f"\n{'='*60}")
    print_rank0("HunyuanVideo-1.5 阶段3：VAE Decoder (GPU)")
    print_rank0(f"{'='*60}")
    
    # 加载配置
    print_rank0(f"\n加载配置: {paths['config']}")
    config = load_generation_config(paths['config'])
    
    # 获取模型路径
    model_path = config['model_path']
    
    # 加载 latents
    print_rank0(f"\n加载 latents: {paths['latents']}")
    latents, latents_metadata = load_latents_from_safetensors(
        paths['latents'],
        device='cpu'
    )
    
    # 加载 VAE
    print_rank0(f"\n加载 VAE: {model_path}")
    
    # 获取 VAE 配置（参考官方 pipeline 的 get_vae_inference_config）
    # 使用 fp16 和 no_grad 后可以使用正常的 tile size
    vae_config = {
        'sample_size': 256,  # 标准 tile size
        'tile_overlap_factor': 0.25,
        'dtype': torch.float16,  # 使用 fp16 节省显存
    }
    
    vae = hunyuanvideo_15_vae.AutoencoderKLConv3D.from_pretrained(
        os.path.join(model_path, "vae"),
        torch_dtype=vae_config['dtype'],
    ).to('cuda')
    
    vae.set_tile_sample_min_size(
        vae_config['sample_size'],
        vae_config['tile_overlap_factor']
    )
    vae.eval()
    
    print_rank0(f"  ✓ VAE 已加载")
    print_rank0(f"  数据类型: {vae_config['dtype']}")
    print_rank0(f"  Tile size: {vae_config['sample_size']}")
    
    # 解码
    enable_tiling = not args.disable_tiling
    enable_tile_parallelism = not args.disable_tile_parallelism
    
    video_frames, decode_time = decode_latents_to_video(
        vae,
        latents,
        config,
        enable_tiling=enable_tiling,
        enable_tile_parallelism=enable_tile_parallelism,
    )
    
    # 保存结果（只有 rank 0 有完整结果）
    if is_main_process() and video_frames is not None:
        # 保存原始 frames tensor（可选）
        if args.save_frames:
            from safetensors.torch import save_file as save_safetensors
            frames_path = paths['frames']
            print_rank0(f"\n保存原始 frames tensor 到: {frames_path}")
            save_safetensors({'frames': video_frames.cpu().contiguous()}, frames_path)
            print_rank0(f"  Frames tensor shape: {video_frames.shape}")
            print_rank0(f"  Frames tensor dtype: {video_frames.dtype}")
        
        # 导出视频
        output_video = args.output_video or paths['video']
        print_rank0(f"\n导出视频到: {output_video}")
        print_rank0(f"FPS: {args.fps}")
        
        # video_frames 是 [B, C, T, H, W]，需要取 batch=0
        video_tensor = video_frames[0].cpu().float()  # [C, T, H, W]
        save_video(video_tensor, output_video, fps=args.fps)
        
        # 统计信息
        num_frames = video_tensor.shape[1]
        height = video_tensor.shape[2]
        width = video_tensor.shape[3]
        
        print_rank0(f"\n=== 生成统计 ===")
        print_rank0(f"帧数: {num_frames}")
        print_rank0(f"分辨率: {width}x{height}")
        print_rank0(f"FPS: {args.fps}")
        print_rank0(f"时长: {num_frames/args.fps:.2f} 秒")
        print_rank0(f"VAE 解码耗时: {decode_time:.2f} 秒")
        
        # 更新配置
        config['stage3_decode_time'] = decode_time
        config['output_video'] = output_video
        save_generation_config(config, paths['config'])
        
        print_rank0(f"\n{'='*60}")
        print_rank0("阶段3 完成！")
        print_rank0(f"{'='*60}")
        print_rank0(f"\n输出视频: {output_video}")
    
    # 清理
    del vae
    del video_frames
    del latents
    torch.cuda.empty_cache()
    
    print_rank0("\n✓ 视频生成完成！")


if __name__ == "__main__":
    main()