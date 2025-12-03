#!/usr/bin/env python3
"""
HunyuanVideo-1.5 Generation using Diffusers
Uses the official Diffusers pipeline for easier deployment
"""

import argparse
import torch
from diffusers import HunyuanVideo15Pipeline
from diffusers.utils import export_to_video


def main():
    parser = argparse.ArgumentParser(description='Generate video using HunyuanVideo-1.5 Diffusers')
    
    # Required parameters
    parser.add_argument(
        '--prompt', type=str, required=True,
        help='Text prompt for video generation'
    )
    
    # Model configuration
    parser.add_argument(
        '--model_id', type=str, 
        default='hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v',
        help='Model ID or path (default: 720p t2v model from HuggingFace)'
    )
    
    # Generation parameters
    parser.add_argument(
        '--num_frames', type=int, default=121,
        help='Number of frames to generate (default: 121)'
    )
    parser.add_argument(
        '--num_inference_steps', type=int, default=50,
        help='Number of inference steps (default: 50)'
    )
    parser.add_argument(
        '--guidance_scale', type=float, default=6.0,
        help='Guidance scale for classifier-free guidance (default: 6.0)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--height', type=int, default=None,
        help='Video height (default: None, uses model default)'
    )
    parser.add_argument(
        '--width', type=int, default=None,
        help='Video width (default: None, uses model default)'
    )
    
    # Performance options
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to use (default: cuda)'
    )
    parser.add_argument(
        '--dtype', type=str, default='bf16', choices=['bf16', 'fp16', 'fp32'],
        help='Data type (default: bf16)'
    )
    parser.add_argument(
        '--enable_cpu_offload', action='store_true',
        help='Enable CPU offload to save GPU memory'
    )
    parser.add_argument(
        '--enable_vae_tiling', action='store_true', default=False,
        help='Enable VAE tiling to reduce memory usage (default: False)'
    )
    parser.add_argument(
        '--enable_sequential_cpu_offload', action='store_true',
        help='Enable sequential CPU offload (slower but uses less memory)'
    )
    
    # Output options
    parser.add_argument(
        '--output_path', type=str, default='output.mp4',
        help='Output video file path (default: output.mp4)'
    )
    parser.add_argument(
        '--fps', type=int, default=24,
        help='Output video FPS (default: 24)'
    )
    
    args = parser.parse_args()
    
    # Setup dtype
    dtype_map = {
        'bf16': torch.bfloat16,
        'fp16': torch.float16,
        'fp32': torch.float32,
    }
    dtype = dtype_map[args.dtype]
    
    # Check device availability
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'
        dtype = torch.float32  # CPU doesn't support bf16/fp16
    
    print(f"Loading model: {args.model_id}")
    print(f"Device: {args.device}, dtype: {args.dtype}")
    
    # Load pipeline
    pipe = HunyuanVideo15Pipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
    )
    
    # Apply memory optimizations
    if args.enable_sequential_cpu_offload:
        print("Enabling sequential CPU offload...")
        pipe.enable_sequential_cpu_offload()
    elif args.enable_cpu_offload:
        print("Enabling model CPU offload...")
        pipe.enable_model_cpu_offload()
    else:
        print(f"Moving pipeline to {args.device}...")
        pipe = pipe.to(args.device)
    
    if args.enable_vae_tiling:
        print("Enabling VAE tiling...")
        pipe.vae.enable_tiling()
    
    # Setup generator
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    
    # Prepare generation kwargs
    gen_kwargs = {
        'prompt': args.prompt,
        'num_frames': args.num_frames,
        'num_inference_steps': args.num_inference_steps,
        'guidance_scale': args.guidance_scale,
        'generator': generator,
    }
    
    # Add optional parameters
    if args.height is not None:
        gen_kwargs['height'] = args.height
    if args.width is not None:
        gen_kwargs['width'] = args.width
    
    # Generate video
    print(f"\nGenerating video with prompt: '{args.prompt}'")
    print(f"Parameters: {args.num_frames} frames, {args.num_inference_steps} steps, guidance_scale={args.guidance_scale}, seed={args.seed}")
    
    output = pipe(**gen_kwargs)
    video = output.frames[0]
    
    # Export video
    print(f"\nExporting video to: {args.output_path}")
    export_to_video(video, args.output_path, fps=args.fps)
    print(f"✓ Video saved successfully!")


if __name__ == '__main__':
    main()