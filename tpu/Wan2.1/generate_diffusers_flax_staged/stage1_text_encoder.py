#!/usr/bin/env python3
"""
Wan 2.1 三阶段生成 - 阶段1：Text Encoder

本阶段负责：
1. 加载 WanPipeline（仅用于 text encoding）
2. 使用 T5 Text Encoder 编码 prompt
3. 将 prompt embeddings 保存为 SafeTensors 格式

输出文件：
- stage1_embeddings.safetensors: 包含所有 prompt embeddings
- generation_config.json: 生成配置参数
"""

import os
import argparse
import warnings
import logging
import torch

from diffusers import WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

from utils import (
    MODEL_NAME,
    FLOW_SHIFT,
    WIDTH, HEIGHT, FRAMES, FPS, NUM_STEPS,
    save_embeddings_to_safetensors,
    save_generation_config,
    get_default_paths,
)


def encode_prompts(pipe, prompt, negative_prompt="", device='cpu'):
    """
    使用 Text Encoder 编码 prompt
    
    Args:
        pipe: WanPipeline
        prompt: 正面提示词
        negative_prompt: 负面提示词
        device: 计算设备
        
    Returns:
        dict: 包含所有 prompt embeddings 的字典
    """
    print(f"\n=== 阶段1：Text Encoder ===")
    print(f"正面 prompt: {prompt}")
    print(f"负面 prompt: {negative_prompt if negative_prompt else '(空)'}")
    print(f"设备: {device}")
    
    # 计算正面 prompt embeddings
    print("\n编码正面 prompt...")
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        do_classifier_free_guidance=True,
        device=device,
        num_videos_per_prompt=1,
    )
    print(f"  prompt_embeds shape: {prompt_embeds.shape}, dtype: {prompt_embeds.dtype}")
    print(f"  negative_prompt_embeds shape: {negative_prompt_embeds.shape}")
    
    embeddings_dict = {
        'prompt_embeds': prompt_embeds,
        'negative_prompt_embeds': negative_prompt_embeds,
    }
    
    print("\n✓ 所有 prompt embeddings 已计算完成")
    return embeddings_dict


def main():
    parser = argparse.ArgumentParser(
        description='Wan 2.1 阶段1：Text Encoder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 基本用法
  python stage1_text_encoder.py --prompt "A beautiful sunset over the ocean"
  
  # 指定输出目录
  python stage1_text_encoder.py --prompt "A cat playing piano" --output_dir ./my_outputs
  
  # 使用负面提示词
  python stage1_text_encoder.py --prompt "A serene lake" --negative_prompt "blurry, low quality"
        """
    )
    
    # Prompt 参数（阶段1必需）
    parser.add_argument(
        '--prompt', type=str,
        default=('A cat and a dog baking a cake together in a kitchen. '
                 'The cat is carefully measuring flour, while the dog is '
                 'stirring the batter with a wooden spoon. The kitchen is cozy, '
                 'with sunlight streaming through the window.'),
        help='Text prompt for video generation'
    )
    parser.add_argument(
        '--negative_prompt', type=str,
        default=('Bright tones, overexposed, static, blurred details, '
                 'subtitles, style, works, paintings, images, static, '
                 'overall gray, worst quality, low quality, JPEG compression residue, '
                 'ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, '
                 'deformed, disfigured, misshapen limbs, fused fingers, still picture, '
                 'messy background, three legs, many people in the background, walking backwards'),
        help='Negative prompt for CFG'
    )
    
    # 模型参数
    parser.add_argument(
        '--model_id', type=str,
        default=MODEL_NAME,
        help='Model ID or path'
    )
    parser.add_argument(
        '--flow_shift', type=float,
        default=FLOW_SHIFT,
        help='Flow shift for scheduler (5.0 for 720P, 3.0 for 480P)'
    )
    
    # 视频参数（保存到配置中供后续阶段使用）
    parser.add_argument('--width', type=int, default=WIDTH, help='Video width')
    parser.add_argument('--height', type=int, default=HEIGHT, help='Video height')
    parser.add_argument('--frames', type=int, default=FRAMES, help='Number of frames')
    parser.add_argument('--fps', type=int, default=FPS, help='Video FPS')
    parser.add_argument('--num_inference_steps', type=int, default=NUM_STEPS, help='Inference steps')
    parser.add_argument('--guidance_scale', type=float, default=5.0, help='CFG scale')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # 输出参数
    parser.add_argument(
        '--output_dir', type=str, default='./stage_outputs',
        help='Output directory for intermediate files (default: ./stage_outputs)'
    )
    
    args = parser.parse_args()
    
    # 设置输出路径
    paths = get_default_paths(args.output_dir)
    
    # 配置日志
    warnings.filterwarnings('ignore', message='.*dtype.*int64.*truncated to dtype int32.*')
    logging.getLogger().setLevel(logging.ERROR)
    
    torch.set_default_dtype(torch.bfloat16)
    
    print(f"\n{'='*60}")
    print("Wan 2.1 阶段1：Text Encoder")
    print(f"{'='*60}")
    
    # 加载 Pipeline（仅需要 text encoder 部分）
    print(f"\n加载模型: {args.model_id}")
    print("（注意：仅使用 Text Encoder 组件）")
    
    # Create scheduler with flow matching settings
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
    
    # 编码 prompt
    embeddings_dict = encode_prompts(
        pipe,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        device='cpu'
    )
    
    # 保存 embeddings
    print(f"\n保存 embeddings 到: {paths['embeddings']}")
    metadata = {
        'prompt': args.prompt,
        'negative_prompt': args.negative_prompt,
        'model_id': args.model_id,
    }
    save_embeddings_to_safetensors(embeddings_dict, paths['embeddings'], metadata)
    
    # 保存完整配置（包含阶段1的参数和视频参数）
    config = {
        'prompt': args.prompt,
        'negative_prompt': args.negative_prompt,
        'model_id': args.model_id,
        'flow_shift': args.flow_shift,
        'width': args.width,
        'height': args.height,
        'frames': args.frames,
        'fps': args.fps,
        'num_inference_steps': args.num_inference_steps,
        'guidance_scale': args.guidance_scale,
        'seed': args.seed,
    }
    save_generation_config(config, paths['config'])
    
    print(f"\n{'='*60}")
    print("阶段1 完成！")
    print(f"{'='*60}")
    print(f"\n输出文件：")
    print(f"  - Embeddings: {paths['embeddings']}")
    print(f"  - 配置文件:   {paths['config']}")
    print(f"\n下一步：运行 stage2_transformer.py 进行 Transformer 推理")
    
    # 清理内存
    del pipe
    del embeddings_dict
    
    print("\n✓ 阶段1 执行完成")


if __name__ == "__main__":
    main()