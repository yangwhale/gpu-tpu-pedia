#!/usr/bin/env python3
"""
Flux.2 GPU 三阶段生成 - 阶段2：Transformer (Denoising)

本阶段负责：
1. 加载阶段1生成的 prompt embeddings
2. 加载 Transformer 模型
3. 运行 denoising loop 生成 latents
4. 将 latents 保存为 SafeTensors 格式

输入文件：
- stage1_embeddings.safetensors: prompt embeddings
- generation_config.json: 生成配置

输出文件：
- stage2_latents.safetensors: 生成的 latents

注意：Flux.2 使用 Embedded CFG，guidance_scale 嵌入到 timestep embedding 中。
"""

import os
import time
import argparse
import warnings
import logging
import numpy as np
import random

import torch
from tqdm import tqdm

from diffusers import Flux2Pipeline

from utils import (
    MODEL_NAME,
    WIDTH, HEIGHT, NUM_STEPS, GUIDANCE_SCALE,
    load_embeddings_from_safetensors,
    save_latents_to_safetensors,
    load_generation_config,
    save_generation_config,
    get_default_paths,
)


# ============================================================================
# Transformer 推理
# ============================================================================

def run_transformer_inference(
    pipe,
    prompt_embeds,
    config,
    device,
    warmup_steps=0,
):
    """
    运行 Transformer 推理生成 latents
    
    使用 output_type='latent' 让 pipeline 跳过 VAE 解码
    """
    print(f"\n=== 阶段2：Transformer 推理 ===")
    print(f"推理步数: {config['num_inference_steps']}")
    print(f"引导尺度: {config['guidance_scale']}")
    print(f"随机种子: {config['seed']}")
    print(f"分辨率: {config['height']}x{config['width']}")
    
    generator = torch.Generator(device=device)
    generator.manual_seed(config['seed'])
    
    # Flux.2 pipeline 参数
    gen_kwargs = {
        'prompt': None,  # Already encoded
        'prompt_embeds': prompt_embeds.to(device),
        'height': config['height'],
        'width': config['width'],
        'num_inference_steps': config['num_inference_steps'],
        'guidance_scale': config['guidance_scale'],
        'generator': generator,
        'output_type': 'latent',  # 关键：只返回 latents，不解码
    }
    
    # === Warmup (可选) ===
    if warmup_steps > 0:
        print(f"\n预热中（{warmup_steps}步）...")
        warmup_kwargs = gen_kwargs.copy()
        warmup_kwargs['num_inference_steps'] = warmup_steps
        warmup_generator = torch.Generator(device=device)
        warmup_generator.manual_seed(config['seed'])
        warmup_kwargs['generator'] = warmup_generator
        
        start_warmup = time.perf_counter()
        with torch.no_grad():
            _ = pipe(**warmup_kwargs)
        torch.cuda.synchronize()
        warmup_elapsed = time.perf_counter() - start_warmup
        print(f"  ✓ 预热完成，耗时: {warmup_elapsed:.2f}秒")
    
    # === 正式推理 ===
    print("\n开始 Transformer 推理...")
    
    start_time = time.perf_counter()
    
    # Re-create generator for actual run
    generator = torch.Generator(device=device)
    generator.manual_seed(config['seed'])
    gen_kwargs['generator'] = generator
    
    with torch.no_grad():
        result = pipe(**gen_kwargs)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    
    latents = result.images  # output_type='latent' 时，images 就是 latents
    
    print(f"\n✓ Transformer 推理完成，耗时: {elapsed:.2f} 秒")
    print(f"  平均每步时间: {elapsed / config['num_inference_steps']:.2f}s")

    # 转换为 CPU tensor
    torch_latents = latents.cpu()

    print(f"  Latents shape: {torch_latents.shape}")
    print(f"  Latents dtype: {torch_latents.dtype}")
    print(f"  Latents range: [{torch_latents.min():.4f}, {torch_latents.max():.4f}]")

    return torch_latents, elapsed


def main():
    parser = argparse.ArgumentParser(
        description='Flux.2 GPU 阶段2：Transformer (Denoising)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 基本用法（使用阶段1的输出）
  python stage2_transformer.py
  
  # 指定输入目录
  python stage2_transformer.py --input_dir ./my_outputs
  
  # 覆盖配置参数
  python stage2_transformer.py --num_inference_steps 50
        """
    )

    parser.add_argument(
        '--input_dir', type=str, default='./stage_outputs',
        help='Input directory containing stage1 outputs (default: ./stage_outputs)'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory (default: same as input_dir)'
    )

    # 可覆盖的配置参数
    parser.add_argument('--num_inference_steps', type=int, default=NUM_STEPS, help=f'推理步数（默认{NUM_STEPS}）')
    parser.add_argument('--guidance_scale', type=float, default=GUIDANCE_SCALE, help=f'引导尺度（默认{GUIDANCE_SCALE}）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子（默认42）')
    parser.add_argument('--height', type=int, default=HEIGHT, help=f'图像高度（默认{HEIGHT}）')
    parser.add_argument('--width', type=int, default=WIDTH, help=f'图像宽度（默认{WIDTH}）')
    parser.add_argument('--warmup_steps', type=int, default=2,
                        help='预热步数（0=不预热）')

    # 其他参数
    parser.add_argument('--model_id', type=str, default=None, help='Override model ID from stage1')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for inference')
    parser.add_argument('--num_iterations', type=int, default=1, help='Number of benchmark iterations')

    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    input_paths = get_default_paths(args.input_dir)
    output_paths = get_default_paths(output_dir)

    warnings.filterwarnings('ignore')
    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger('diffusers').setLevel(logging.ERROR)
    logging.getLogger('transformers').setLevel(logging.ERROR)

    torch.set_default_dtype(torch.bfloat16)

    print(f"\n{'='*60}")
    print("Flux.2 GPU 阶段2：Transformer (Denoising)")
    print(f"{'='*60}")

    # 加载阶段1配置
    print(f"\n加载阶段1配置: {input_paths['config']}")
    config = load_generation_config(input_paths['config'])

    # Stage2 专用参数（使用命令行参数或默认值）
    config['num_inference_steps'] = args.num_inference_steps
    config['guidance_scale'] = args.guidance_scale
    config['seed'] = args.seed
    config['height'] = args.height
    config['width'] = args.width
    if args.model_id is not None:
        config['model_id'] = args.model_id

    # 设置随机种子
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    # 加载 embeddings
    print(f"\n加载 embeddings: {input_paths['embeddings']}")
    prompt_embeds_dict, embed_metadata = load_embeddings_from_safetensors(
        input_paths['embeddings'],
        device='cpu',
        restore_dtype=True
    )
    prompt_embeds = prompt_embeds_dict['prompt_embeds']
    print(f"  prompt_embeds shape: {prompt_embeds.shape}")

    # 加载 Pipeline
    model_id = config.get('model_id', MODEL_NAME)
    print(f"\n加载模型: {model_id}")
    print(f"设备: {args.device}")

    pipe = Flux2Pipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        text_encoder=None,  # 不需要 text encoder
    )
    pipe.enable_model_cpu_offload()
    
    print("✓ Pipeline 加载完成")

    # 运行推理
    times = []
    latents = None

    for i in range(args.num_iterations):
        print(f"\n--- 迭代 {i+1}/{args.num_iterations} ---")
        latents, elapsed = run_transformer_inference(
            pipe, prompt_embeds, config, args.device,
            warmup_steps=args.warmup_steps if i == 0 else 0
        )
        times.append(elapsed)
        print(f"  耗时: {elapsed:.2f} 秒")

    # 保存 latents
    print(f"\n保存 latents 到: {output_paths['latents']}")
    metadata = {
        'num_inference_steps': str(config['num_inference_steps']),
        'guidance_scale': str(config['guidance_scale']),
        'seed': str(config['seed']),
        'height': str(config['height']),
        'width': str(config['width']),
    }
    save_latents_to_safetensors(latents, output_paths['latents'], metadata)

    # 更新配置
    save_generation_config(config, output_paths['config'])

    # 打印性能统计
    print(f"\n=== 性能统计 ===")
    print(f"总迭代次数: {len(times)}")
    print(f"第一次运行: {times[0]:.4f} 秒")
    if len(times) > 1:
        avg_time = sum(times[1:]) / len(times[1:])
        print(f"后续运行平均时间: {avg_time:.4f} 秒")

    print(f"\n{'='*60}")
    print("阶段2 完成！")
    print(f"{'='*60}")
    print(f"\n输出文件：")
    print(f"  - Latents: {output_paths['latents']}")
    print(f"\n下一步：运行 stage3_vae_decoder.py 进行 VAE 解码")

    print("\n✓ 阶段2 执行完成")


if __name__ == "__main__":
    main()
