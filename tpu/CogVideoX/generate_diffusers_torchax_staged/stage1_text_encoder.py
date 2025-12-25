#!/usr/bin/env python3
"""
CogVideoX 三阶段生成 - 阶段1：Text Encoder (TPU)

本阶段负责：
1. 加载 CogVideoXPipeline
2. 启用 torchax 并移动 Text Encoder 到 TPU
3. 使用 T5 Text Encoder 编码 prompt
4. 将 prompt embeddings 保存为 SafeTensors 格式

输出文件：
- stage1_embeddings.safetensors: 包含所有 prompt embeddings
- generation_config.json: 生成配置参数
"""

import argparse
import warnings
import logging

import jax
import torch
from jax.sharding import Mesh
from jax.experimental import mesh_utils

import torchax

from diffusers import CogVideoXPipeline

from utils import (
    MODEL_NAME,
    DEFAULT_DP,
    TEXT_ENCODER_SHARDINGS,
    shard_weight_dict,
    setup_jax_cache,
    setup_pytree_registrations,
    save_embeddings_to_safetensors,
    save_generation_config,
    get_default_paths,
)


# ============================================================================
# Torchax 帮助函数
# ============================================================================

def move_module_to_xla(env, module):
    """Move module weights to XLA devices."""
    with jax.default_device("cpu"):
        state_dict = module.state_dict()
        state_dict = env.to_xla(state_dict)
        module.load_state_dict(state_dict, assign=True)


# ============================================================================
# Pipeline 加载与设置
# ============================================================================

def load_pipeline(model_id):
    """Load pipeline BEFORE enabling torchax to avoid safetensors issues."""
    print("\n=== 加载 CogVideoX Pipeline ===")
    print("加载模型中（在启用 torchax 之前）...")
    
    pipe = CogVideoXPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    )
    print("✓ 模型加载完成\n")
    return pipe


def setup_text_encoder_for_jax(pipe, mesh, env):
    """Setup Text Encoder for JAX/TPU execution."""
    print("=== 设置 Text Encoder (TPU) ===")
    
    # Move Text Encoder to XLA
    print("- 移动 Text Encoder 到 TPU...")
    move_module_to_xla(env, pipe.text_encoder)
    pipe.text_encoder = torchax.compile(pipe.text_encoder)
    pipe.text_encoder.params = shard_weight_dict(
        pipe.text_encoder.params, TEXT_ENCODER_SHARDINGS, mesh
    )
    pipe.text_encoder.buffers = shard_weight_dict(
        pipe.text_encoder.buffers, TEXT_ENCODER_SHARDINGS, mesh
    )
    
    # Wait for sharding to complete
    torchax.interop.call_jax(jax.block_until_ready, pipe.text_encoder.params)
    
    print("✓ Text Encoder 设置完成\n")
    return pipe


# ============================================================================
# 编码函数
# ============================================================================

def encode_prompts(pipe, prompt, negative_prompt, device='jax', dtype=torch.bfloat16):
    """
    使用 Text Encoder 编码 prompt
    """
    print(f"\n=== 编码文本提示词 ===")
    print(f"正面 prompt: {prompt[:100]}..." if len(prompt) > 100 else f"正面 prompt: {prompt}")
    print(f"负面 prompt: {negative_prompt[:50]}..." if len(negative_prompt) > 50 else f"负面 prompt: {negative_prompt if negative_prompt else '(空)'}")
    
    # 编码 prompts
    print("\n编码 prompts...")
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        max_sequence_length=226,  # CogVideoX 默认值
        device=device,
        dtype=dtype,
    )
    print(f"  prompt_embeds shape: {prompt_embeds.shape}, dtype: {prompt_embeds.dtype}")
    print(f"  negative_prompt_embeds shape: {negative_prompt_embeds.shape}")
    
    return {
        'prompt_embeds': prompt_embeds,
        'negative_prompt_embeds': negative_prompt_embeds,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='CogVideoX 阶段1：Text Encoder (TPU)',
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
    
    # 模型和 Mesh 参数
    parser.add_argument('--model_id', type=str, default=MODEL_NAME)
    parser.add_argument('--dp', type=int, default=DEFAULT_DP, help='Data parallelism dimension')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./stage_outputs')
    
    args = parser.parse_args()
    
    # 设置输出路径
    paths = get_default_paths(args.output_dir)
    
    # 配置日志
    warnings.filterwarnings('ignore')
    logging.getLogger().setLevel(logging.ERROR)
    
    # 设置 JAX
    setup_jax_cache()
    
    torch.set_default_dtype(torch.bfloat16)
    
    # Setup PyTree registrations
    setup_pytree_registrations()
    
    print(f"\n{'='*60}")
    print("CogVideoX 阶段1：Text Encoder (TPU)")
    print(f"{'='*60}")
    
    print(f"\n配置参数：")
    print(f"  模型: {args.model_id}")
    print(f"  DP: {args.dp}")
    
    # 加载 Pipeline（在启用 torchax 之前）
    pipe = load_pipeline(args.model_id)
    
    # 启用 torchax
    print("启用 torchax...")
    torchax.enable_globally()
    env = torchax.default_env()
    
    # 创建 mesh
    assert len(jax.devices()) % args.dp == 0
    tp_dim = len(jax.devices()) // args.dp
    mesh_devices = mesh_utils.create_device_mesh(
        (args.dp, tp_dim), allow_split_physical_axes=True
    )
    mesh = Mesh(mesh_devices, ("dp", "tp"))
    print(f"Mesh: {mesh}\n")
    
    # 设置 Text Encoder
    with mesh:
        pipe = setup_text_encoder_for_jax(pipe, mesh, env)
        
        # 编码 prompt
        embeddings_dict = encode_prompts(
            pipe,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            device='jax',
            dtype=torch.bfloat16,
        )
    
    # 转换回 CPU tensor 以保存
    print("\n转换 tensor 到 CPU...")
    cpu_embeddings = {}
    for key, tensor in embeddings_dict.items():
        if hasattr(tensor, 'to'):
            cpu_embeddings[key] = tensor.to('cpu')
        else:
            cpu_embeddings[key] = tensor
    
    # 保存 embeddings
    print(f"\n保存 embeddings 到: {paths['embeddings']}")
    metadata = {
        'prompt': args.prompt,
        'negative_prompt': args.negative_prompt,
        'model_id': args.model_id,
    }
    save_embeddings_to_safetensors(cpu_embeddings, paths['embeddings'], metadata)
    
    # 保存配置（仅保存 stage1 相关参数，视频参数在 stage2 设置）
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
    print(f"\n保存的 tensor shapes：")
    for key, value in cpu_embeddings.items():
        print(f"  - {key}: {value.shape}")
    print(f"\n下一步：运行 stage2_transformer.py 进行 Transformer 推理")
    
    # 清理内存
    del pipe
    del embeddings_dict
    del cpu_embeddings
    
    print("\n✓ 阶段1 执行完成")


if __name__ == "__main__":
    main()
