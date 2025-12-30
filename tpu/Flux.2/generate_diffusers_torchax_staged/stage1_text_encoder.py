#!/usr/bin/env python3
# pyright: reportArgumentType=false, reportCallIssue=false
"""
Flux.2 三阶段生成 - 阶段1：Text Encoder (CPU)

在 CPU 上使用 Mistral3 编码 prompt 并保存 embeddings。
Mistral3 使用动态控制流，无法在 TPU 上 JIT 编译。

输出：stage1_embeddings.safetensors, generation_config.json
"""

import argparse
import gc
import logging
import warnings

import torch

from utils import (
    DEFAULT_PROMPT,
    MODEL_NAME,
    SYSTEM_MESSAGE,
    get_default_paths,
    save_embeddings_to_safetensors,
    save_generation_config,
    setup_jax_cache,
)


def encode_prompt_on_cpu(model_id: str, prompt: str, max_sequence_length: int = 512) -> torch.Tensor:
    """在 CPU 上使用 Mistral3 编码 prompt。"""
    from transformers import Mistral3ForConditionalGeneration, PixtralProcessor
    
    print("\n=== 在 CPU 上加载 Mistral3 Text Encoder ===")
    
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    text_encoder.eval()
    
    tokenizer = PixtralProcessor.from_pretrained(model_id, subfolder="tokenizer")
    
    print("✓ Text Encoder 加载成功")
    print(f"- 编码 prompt: {prompt[:80]}...")
    
    # 移除 [IMG] token 以避免 Pixtral 验证问题
    cleaned_prompt = prompt.replace("[IMG]", "")
    
    messages = [[
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
        {"role": "user", "content": [{"type": "text", "text": cleaned_prompt}]},
    ]]
    
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
    
    print("\n编码 prompt (CPU)...")
    with torch.no_grad():
        output = text_encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
        )
    
    # 提取 layers 10, 20, 30 的隐藏状态并拼接
    hidden_states_layers = (10, 20, 30)
    out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
    out = out.to(dtype=torch.bfloat16)
    
    batch_size, num_channels, seq_len, hidden_dim = out.shape
    prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)
    
    print(f"✓ Prompt embeddings shape: {prompt_embeds.shape}")
    
    del text_encoder, tokenizer
    gc.collect()
    
    return prompt_embeds


def main():
    parser = argparse.ArgumentParser(description='Flux.2 阶段1：Text Encoder (CPU)')
    parser.add_argument('--prompt', type=str, default=DEFAULT_PROMPT)
    parser.add_argument('--model_id', type=str, default=MODEL_NAME)
    parser.add_argument('--max_sequence_length', type=int, default=512)
    parser.add_argument('--output_dir', type=str, default='./stage_outputs')
    args = parser.parse_args()
    
    paths = get_default_paths(args.output_dir)
    
    warnings.filterwarnings('ignore')
    logging.getLogger().setLevel(logging.ERROR)
    setup_jax_cache()
    torch.set_default_dtype(torch.bfloat16)
    
    print(f"\n{'='*60}")
    print("Flux.2 阶段1：Text Encoder (CPU)")
    print(f"{'='*60}")
    
    # 编码 prompt
    prompt_embeds = encode_prompt_on_cpu(
        model_id=args.model_id,
        prompt=args.prompt,
        max_sequence_length=args.max_sequence_length,
    )
    
    # 保存 embeddings
    save_embeddings_to_safetensors(
        {'prompt_embeds': prompt_embeds},
        paths['embeddings'],
        {'prompt': args.prompt, 'model_id': args.model_id}
    )
    
    # 保存配置
    save_generation_config({
        'prompt': args.prompt,
        'model_id': args.model_id,
        'max_sequence_length': args.max_sequence_length,
    }, paths['config'])
    
    print(f"\n{'='*60}")
    print("✓ 阶段1 完成！下一步：运行 stage2_transformer.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
