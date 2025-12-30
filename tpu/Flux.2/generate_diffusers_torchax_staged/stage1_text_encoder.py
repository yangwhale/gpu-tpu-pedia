#!/usr/bin/env python3
# pyright: reportArgumentType=false, reportCallIssue=false
"""
Flux.2 三阶段生成 - 阶段1：Text Encoder (CPU)

本阶段负责：
1. 在 CPU 上加载 Mistral3 Text Encoder
2. 在 CPU 上运行 prompt 编码
3. 将 prompt embeddings 保存为 SafeTensors 格式

注意：Mistral3 使用动态控制流（如 attention_mask.all()），
无法直接在 TPU 上用 JAX JIT 编译。因此在 CPU 上编码。
这与 Flux.2 单文件版本的 --mistral_cpu 模式一致。

输出文件：
- stage1_embeddings.safetensors: 包含 prompt embeddings
- generation_config.json: 生成配置参数
"""

import argparse
import warnings
import logging
import gc
import os

import torch

from utils import (
    MODEL_NAME,
    DEFAULT_PROMPT,
    setup_jax_cache,
    save_embeddings_to_safetensors,
    save_generation_config,
    get_default_paths,
)


# ============================================================================
# Text Encoder 加载和编码
# ============================================================================

def encode_prompt_on_cpu(model_id, prompt, max_sequence_length=512):
    """
    在 CPU 上使用 Mistral3 编码 prompt。
    
    参考 GPU 版本的 encode_prompt 函数，但强制在 CPU 上运行。
    """
    from transformers import PixtralProcessor, Mistral3ForConditionalGeneration
    
    print("\n=== 在 CPU 上加载 Mistral3 Text Encoder ===")
    
    # 直接从 text_encoder 目录加载
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    text_encoder.eval()
    
    # 使用 PixtralProcessor 从 tokenizer 目录加载
    tokenizer = PixtralProcessor.from_pretrained(
        model_id,
        subfolder="tokenizer"
    )
    
    print("✓ Text Encoder 加载成功")
    print(f"- 编码 prompt: {prompt[:80]}...")
    
    # 格式化输入 - 使用 Flux.2 pipeline 中的 SYSTEM_MESSAGE
    # 来源: diffusers/pipelines/flux2/system_messages.py
    SYSTEM_MESSAGE = """You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object
attribution and actions without speculation."""
    
    # 使用与 pipeline 相同的 format_input 逻辑
    # 移除 [IMG] tokens 以避免 Pixtral 验证问题
    cleaned_prompt = prompt.replace("[IMG]", "")
    
    messages = [[
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
        {"role": "user", "content": [{"type": "text", "text": cleaned_prompt}]},
    ]]
    
    # Tokenize
    inputs = tokenizer.apply_chat_template(  # type: ignore[call-arg]
        messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_sequence_length,
    )
    
    # 前向传播
    print("\n编码 prompt (CPU)...")
    with torch.no_grad():
        output = text_encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
        )
    
    # 提取隐藏状态 - 使用层 10, 20, 30
    hidden_states_layers = (10, 20, 30)
    out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
    out = out.to(dtype=torch.bfloat16)
    
    # 重塑为 [batch, seq_len, hidden_dim * num_layers]
    batch_size, num_channels, seq_len, hidden_dim = out.shape
    prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)
    
    print(f"✓ Prompt embeddings shape: {prompt_embeds.shape}")
    print(f"  - batch_size: {batch_size}")
    print(f"  - seq_len: {seq_len}")
    print(f"  - hidden_dim: {num_channels * hidden_dim} (= {num_channels} layers × {hidden_dim})")
    
    # 释放内存
    del text_encoder, tokenizer
    gc.collect()
    
    return prompt_embeds


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Flux.2 阶段1：Text Encoder (CPU)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 基本用法
  python stage1_text_encoder.py --prompt "A beautiful sunset over the ocean"
  
  # 指定输出目录
  python stage1_text_encoder.py --prompt "A cat playing piano" --output_dir ./my_outputs
        """
    )
    
    # Prompt 参数
    parser.add_argument(
        '--prompt', type=str,
        default=DEFAULT_PROMPT,
        help='Text prompt for image generation'
    )
    
    # 模型参数
    parser.add_argument('--model_id', type=str, default=MODEL_NAME)
    parser.add_argument('--max_sequence_length', type=int, default=512,
                        help='Maximum sequence length for text encoder')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./stage_outputs')
    
    args = parser.parse_args()
    
    # 设置输出路径
    paths = get_default_paths(args.output_dir)
    
    # 配置日志
    warnings.filterwarnings('ignore')
    logging.getLogger().setLevel(logging.ERROR)
    
    # 设置 JAX（为后续阶段准备）
    setup_jax_cache()
    
    torch.set_default_dtype(torch.bfloat16)
    
    print(f"\n{'='*60}")
    print("Flux.2 阶段1：Text Encoder (CPU)")
    print(f"{'='*60}")
    
    print(f"\n配置参数：")
    print(f"  模型: {args.model_id}")
    print(f"  最大序列长度: {args.max_sequence_length}")
    
    # 编码 prompt
    prompt_embeds = encode_prompt_on_cpu(
        model_id=args.model_id,
        prompt=args.prompt,
        max_sequence_length=args.max_sequence_length,
    )
    
    # 保存 embeddings
    cpu_embeddings = {
        'prompt_embeds': prompt_embeds,
    }
    
    print(f"\n保存 embeddings 到: {paths['embeddings']}")
    metadata = {
        'prompt': args.prompt,
        'model_id': args.model_id,
        'max_sequence_length': str(args.max_sequence_length),
    }
    save_embeddings_to_safetensors(cpu_embeddings, paths['embeddings'], metadata)
    
    # 保存配置
    config = {
        'prompt': args.prompt,
        'model_id': args.model_id,
        'max_sequence_length': args.max_sequence_length,
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
        print(f"  - {key}: {value.shape}, dtype: {value.dtype}")
    print(f"\n下一步：运行 stage2_transformer.py 进行 Transformer 推理")
    
    # 清理内存
    del prompt_embeds
    del cpu_embeddings
    gc.collect()
    
    print("\n✓ 阶段1 执行完成")


if __name__ == "__main__":
    main()
