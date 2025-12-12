#!/usr/bin/env python3
"""
CogVideoX 三阶段生成 - 阶段1：Text Encoder

本阶段负责：
1. 加载 CogVideoXPipeline（仅用于 text encoding）
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

from diffusers import CogVideoXPipeline

from utils import (
    MODEL_NAME,
    save_embeddings_to_safetensors,
    save_generation_config,
    get_default_paths,
)


def encode_prompts(pipe, prompt, negative_prompt="", device='cpu'):
    """
    使用 Text Encoder 编码 prompt
    
    CogVideoX 使用 T5 Text Encoder，需要调用 pipeline 的内部方法
    
    Args:
        pipe: CogVideoXPipeline
        prompt: 正面提示词
        negative_prompt: 负面提示词
        device: 计算设备
        
    Returns:
        dict: 包含所有 prompt embeddings 的字典
    """
    print(f"\n=== 阶段1：Text Encoder ===")
    print(f"正面 prompt: {prompt[:100]}..." if len(prompt) > 100 else f"正面 prompt: {prompt}")
    print(f"负面 prompt: {negative_prompt if negative_prompt else '(空)'}")
    print(f"设备: {device}")
    
    # CogVideoX 使用 _get_t5_prompt_embeds 方法
    # 参考 CogVideoXPipeline 的 encode_prompt 方法
    print("\n编码正面 prompt...")
    
    # 获取 prompt embeddings
    # CogVideoX pipeline 有 encode_prompt 方法
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        max_sequence_length=226,  # CogVideoX 默认值
        device=device,
        dtype=torch.bfloat16,
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
        description='CogVideoX 阶段1：Text Encoder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 基本用法
  python stage1_text_encoder.py --prompt "A panda playing guitar"
  
  # 指定输出目录
  python stage1_text_encoder.py --prompt "A cat playing piano" --output_dir ./my_outputs
  
  # 使用负面提示词
  python stage1_text_encoder.py --prompt "A serene lake" --negative_prompt "blurry, low quality"
        """
    )
    
    # Prompt 参数
    parser.add_argument(
        '--prompt', type=str,
        default=("A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool "
                 "in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, "
                 "producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously "
                 "and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle "
                 "glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. "
                 "The background includes a small, flowing stream and vibrant green foliage, enhancing the "
                 "peaceful and magical atmosphere of this unique musical performance."),
        help='Text prompt for video generation'
    )
    parser.add_argument(
        '--negative_prompt', type=str,
        default='',
        help='Negative prompt for CFG (optional)'
    )
    
    # 模型参数
    parser.add_argument(
        '--model_id', type=str,
        default=MODEL_NAME,
        help='Model ID or path'
    )
    
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
    print("CogVideoX 阶段1：Text Encoder")
    print(f"{'='*60}")
    
    # 加载 Pipeline（仅需要 text encoder 部分）
    print(f"\n加载模型: {args.model_id}")
    print("（注意：仅使用 Text Encoder 组件）")
    
    pipe = CogVideoXPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
    )
    
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
    
    # 保存阶段1配置（只包含文本编码相关参数）
    config = {
        'prompt': args.prompt,
        'negative_prompt': args.negative_prompt,
        'model_id': args.model_id,
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