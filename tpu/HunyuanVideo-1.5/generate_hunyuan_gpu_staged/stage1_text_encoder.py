#!/usr/bin/env python3
"""
HunyuanVideo-1.5 三阶段生成 - 阶段1：Text Encoder (GPU 版本)

本阶段负责：
1. 加载 Text Encoder (LLM + ByT5)
2. 编码 prompt 为 embeddings
3. 将 prompt embeddings 保存为 SafeTensors 格式

输出文件：
- stage1_embeddings.safetensors: 包含所有 prompt embeddings
- generation_config.json: 生成配置参数

用于 GPU H100 8卡环境
"""

import os
import sys
import argparse
from types import SimpleNamespace
import torch

# 添加 HunyuanVideo-1.5-TPU 到路径
HUNYUAN_ROOT = os.path.expanduser("~/HunyuanVideo-1.5-TPU")
if HUNYUAN_ROOT not in sys.path:
    sys.path.insert(0, HUNYUAN_ROOT)

from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.commons.parallel_states import initialize_parallel_state
from hyvideo.commons.infer_state import initialize_infer_state

from utils import (
    save_embeddings_to_safetensors,
    save_generation_config,
    get_default_paths,
    print_rank0,
    is_main_process,
    DEFAULT_MODEL_PATH,
    DEFAULT_RESOLUTION,
    DEFAULT_ASPECT_RATIO,
    DEFAULT_VIDEO_LENGTH,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_SEED,
)


def encode_prompts(pipe, prompt, negative_prompt="", device='cuda'):
    """
    使用 Text Encoder 编码 prompt
    
    Args:
        pipe: HunyuanVideo_1_5_Pipeline
        prompt: 正面提示词
        negative_prompt: 负面提示词
        device: 计算设备
        
    Returns:
        dict: 包含所有 prompt embeddings 的字典
    """
    print_rank0(f"\n=== 阶段1：Text Encoder ===")
    print_rank0(f"正面 prompt: {prompt}")
    print_rank0(f"负面 prompt: {negative_prompt if negative_prompt else '(空)'}")
    print_rank0(f"设备: {device}")
    
    # 使用 pipeline 的 encode_prompt 方法
    # 计算正面 prompt embeddings
    print_rank0("\n编码正面 prompt...")
    
    prompt_embeds, negative_prompt_embeds, prompt_mask, negative_prompt_mask = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_videos_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt if negative_prompt else "",
        data_type="video",
    )
    
    print_rank0(f"  prompt_embeds shape: {prompt_embeds.shape}, dtype: {prompt_embeds.dtype}")
    if prompt_mask is not None:
        print_rank0(f"  prompt_mask shape: {prompt_mask.shape}")
    
    # 处理 text_encoder_2 的输出（如果存在）
    prompt_embeds_2 = None
    negative_prompt_embeds_2 = None
    prompt_mask_2 = None
    negative_prompt_mask_2 = None
    
    if pipe.text_encoder_2 is not None:
        print_rank0("\n编码 text_encoder_2...")
        (
            prompt_embeds_2,
            negative_prompt_embeds_2,
            prompt_mask_2,
            negative_prompt_mask_2,
        ) = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt if negative_prompt else "",
            text_encoder=pipe.text_encoder_2,
            data_type="video",
        )
        print_rank0(f"  prompt_embeds_2 shape: {prompt_embeds_2.shape}")
    
    # 处理 ByT5 embeddings
    # 注意：_prepare_byt5_embeddings 返回的是已经 concat 的 [negative, positive]
    # 我们需要拆分为与 TPU 版本一致的格式
    prompt_embeds_2 = None  # 重置为 ByT5
    negative_prompt_embeds_2 = None
    prompt_embeds_mask_2 = None
    negative_prompt_embeds_mask_2 = None
    
    if pipe.config.glyph_byT5_v2 and pipe.byt5_model is not None:
        print_rank0("\n编码 ByT5 embeddings...")
        # 设置 _guidance_scale 以便 do_classifier_free_guidance 属性可用
        pipe._guidance_scale = pipe.config.guidance_scale
        byt5_kwargs = pipe._prepare_byt5_embeddings(prompt, device)
        if byt5_kwargs:
            byt5_text_states = byt5_kwargs.get("byt5_text_states")  # [2, 256, 1472] concat
            byt5_text_mask = byt5_kwargs.get("byt5_text_mask")      # [2, 256] concat
            if byt5_text_states is not None:
                # 拆分为 [negative, positive]，与 TPU 版本格式一致
                negative_prompt_embeds_2 = byt5_text_states[0:1]  # [1, 256, 1472]
                prompt_embeds_2 = byt5_text_states[1:2]           # [1, 256, 1472]
                negative_prompt_embeds_mask_2 = byt5_text_mask[0:1]  # [1, 256]
                prompt_embeds_mask_2 = byt5_text_mask[1:2]           # [1, 256]
                print_rank0(f"  prompt_embeds_2 shape: {prompt_embeds_2.shape}")
                print_rank0(f"  negative_prompt_embeds_2 shape: {negative_prompt_embeds_2.shape}")
    
    # 使用与 TPU 版本一致的 key 名称
    embeddings_dict = {
        'prompt_embeds': prompt_embeds,
        'negative_prompt_embeds': negative_prompt_embeds,
        'prompt_embeds_mask': prompt_mask,
        'negative_prompt_embeds_mask': negative_prompt_mask,
    }
    
    if prompt_embeds_2 is not None:
        embeddings_dict['prompt_embeds_2'] = prompt_embeds_2
        embeddings_dict['negative_prompt_embeds_2'] = negative_prompt_embeds_2
        embeddings_dict['prompt_embeds_mask_2'] = prompt_embeds_mask_2
        embeddings_dict['negative_prompt_embeds_mask_2'] = negative_prompt_embeds_mask_2
    
    print_rank0("\n✓ 所有 prompt embeddings 已计算完成")
    return embeddings_dict


def main():
    parser = argparse.ArgumentParser(
        description='HunyuanVideo-1.5 阶段1：Text Encoder (GPU)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 基本用法
  python stage1_text_encoder.py --model_path /path/to/model --prompt "A beautiful sunset"
  
  # 指定输出目录
  python stage1_text_encoder.py --model_path /path/to/model --prompt "A cat playing piano" --output_dir ./my_outputs
  
  # 使用负面提示词
  python stage1_text_encoder.py --model_path /path/to/model --prompt "A serene lake" --negative_prompt "blurry, low quality"
        """
    )
    
    # === Stage 1 专属参数 ===
    
    # Prompt 参数（Stage 1 核心功能）
    parser.add_argument(
        '--prompt', type=str, required=True,
        help='Text prompt for video generation'
    )
    parser.add_argument(
        '--negative_prompt', type=str, default='',
        help='Negative prompt for CFG (default: empty)'
    )
    
    # 模型参数
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to pretrained model'
    )
    parser.add_argument(
        '--resolution', type=str, default=DEFAULT_RESOLUTION,
        choices=['480p', '720p'],
        help=f'Video resolution, used to determine transformer version (default: {DEFAULT_RESOLUTION})'
    )
    
    # 任务类型（影响 transformer 版本选择）
    parser.add_argument(
        '--image_path', type=str, default=None,
        help='Path to reference image for i2v (if provided, uses i2v mode)'
    )
    
    # 输出参数
    parser.add_argument(
        '--output_dir', type=str, default='./stage_outputs',
        help='Output directory for intermediate files (default: ./stage_outputs)'
    )
    
    # 其他参数
    parser.add_argument(
        '--dtype', type=str, default='bf16', choices=['bf16', 'fp32'],
        help='Data type for text encoder (default: bf16)'
    )
    
    args = parser.parse_args()
    
    # 初始化并行状态（单 GPU 模式）
    initialize_parallel_state(sp=1)
    
    # 初始化 infer_state（pipeline 需要，使用默认值）
    infer_args = SimpleNamespace(
        use_sageattn=False,
        sage_blocks_range="0-53",
        enable_torch_compile=False,
        enable_cache=False,
        cache_type="deepcache",
        no_cache_block_id="53",
        cache_start_step=11,
        cache_end_step=45,
        total_steps=50,
        cache_step_interval=4,
    )
    initialize_infer_state(infer_args)
    
    # 设置输出路径
    paths = get_default_paths(args.output_dir)
    
    # 设置 dtype
    if args.dtype == 'bf16':
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32
    
    print_rank0(f"\n{'='*60}")
    print_rank0("HunyuanVideo-1.5 阶段1：Text Encoder (GPU)")
    print_rank0(f"{'='*60}")
    
    # 确定任务类型
    task_type = 'i2v' if args.image_path else 't2v'
    
    # 获取 transformer 版本
    transformer_version = HunyuanVideo_1_5_Pipeline.get_transformer_version(
        args.resolution, task_type, cfg_distilled=False, step_distilled=False, sparse_attn=False
    )
    
    print_rank0(f"\n加载模型: {args.model_path}")
    print_rank0(f"Transformer 版本: {transformer_version}")
    print_rank0("（注意：仅使用 Text Encoder 组件）")
    
    # 创建 pipeline（仅加载 text encoder 相关组件）
    # 禁用 offloading 以避免设备不匹配问题
    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=args.model_path,
        transformer_version=transformer_version,
        enable_offloading=False,  # 禁用 offloading 避免设备不匹配
        enable_group_offloading=False,
        create_sr_pipeline=False,
        transformer_dtype=torch_dtype,
        device=torch.device('cuda'),
    )
    
    # 编码 prompt
    embeddings_dict = encode_prompts(
        pipe,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        device='cuda'
    )
    
    # 仅在主进程保存
    if is_main_process():
        # 保存 embeddings
        print_rank0(f"\n保存 embeddings 到: {paths['embeddings']}")
        metadata = {
            'prompt': args.prompt,
            'negative_prompt': args.negative_prompt,
            'model_path': args.model_path,
            'resolution': args.resolution,
            'transformer_version': transformer_version,
            'task_type': task_type,
        }
        save_embeddings_to_safetensors(embeddings_dict, paths['embeddings'], metadata)
        
        # 保存基本配置（Stage 1 的参数，Stage 2/3 的参数由各自阶段提供）
        config = {
            # Stage 1 参数
            'prompt': args.prompt,
            'negative_prompt': args.negative_prompt,
            'model_path': args.model_path,
            'resolution': args.resolution,
            'transformer_version': transformer_version,
            'task_type': task_type,
            'image_path': args.image_path,
            'dtype': args.dtype,
        }
        save_generation_config(config, paths['config'])
        
        print_rank0(f"\n{'='*60}")
        print_rank0("阶段1 完成！")
        print_rank0(f"{'='*60}")
        print_rank0(f"\n输出文件：")
        print_rank0(f"  - Embeddings: {paths['embeddings']}")
        print_rank0(f"  - 配置文件:   {paths['config']}")
        print_rank0(f"\n下一步：运行 stage2_transformer.py 进行 DiT 推理")
    
    # 清理内存
    del pipe
    del embeddings_dict
    torch.cuda.empty_cache()
    
    print_rank0("\n✓ 阶段1 执行完成")


if __name__ == "__main__":
    main()