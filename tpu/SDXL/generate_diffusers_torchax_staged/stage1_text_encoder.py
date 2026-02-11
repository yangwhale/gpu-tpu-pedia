#!/usr/bin/env python3
"""
SDXL 三阶段生成 - 阶段1：Text Encoder (CPU)

使用两个 CLIP Text Encoder 编码 prompt：
- text_encoder: CLIPTextModel (clip-vit-large-patch14)
- text_encoder_2: CLIPTextModelWithProjection (laion/CLIP-ViT-bigG-14)

输出：
- prompt_embeds: (batch, seq_len, 2048) - 两个 encoder 隐藏状态拼接
- pooled_prompt_embeds: (batch, 1280) - text_encoder_2 的 pooled output
- negative_prompt_embeds: (batch, seq_len, 2048)
- negative_pooled_prompt_embeds: (batch, 1280)

保存到 ./stage_outputs/stage1_embeddings.safetensors
"""

import argparse
import gc
import os
import warnings

import torch
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from utils import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PROMPT,
    GUIDANCE_SCALE,
    HEIGHT,
    MODEL_NAME,
    NUM_STEPS,
    WIDTH,
    get_default_paths,
    save_embeddings_to_safetensors,
    save_generation_config,
    setup_jax_cache,
)


def encode_prompt(
    prompt: str,
    negative_prompt: str,
    tokenizer: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    text_encoder_2: CLIPTextModelWithProjection,
    device: str = "cpu",
) -> dict:
    """编码 prompt 和 negative prompt。

    Returns:
        dict: 包含 prompt_embeds, pooled_prompt_embeds,
              negative_prompt_embeds, negative_pooled_prompt_embeds
    """
    # Tokenize prompts
    text_inputs_1 = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_inputs_2 = tokenizer_2(
        prompt,
        padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    # Encode with text_encoder (CLIPTextModel)
    with torch.no_grad():
        text_output_1 = text_encoder(
            text_inputs_1.input_ids.to(device),
            output_hidden_states=True,
        )
        # 使用倒数第二层的隐藏状态
        prompt_embeds_1 = text_output_1.hidden_states[-2]

    # Encode with text_encoder_2 (CLIPTextModelWithProjection)
    with torch.no_grad():
        text_output_2 = text_encoder_2(
            text_inputs_2.input_ids.to(device),
            output_hidden_states=True,
        )
        # pooled output 来自 text_encoder_2
        pooled_prompt_embeds = text_output_2[0]
        # 使用倒数第二层的隐藏状态
        prompt_embeds_2 = text_output_2.hidden_states[-2]

    # 拼接两个 encoder 的输出: (batch, seq_len, 768 + 1280) = (batch, seq_len, 2048)
    prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)

    # 对 negative prompt 做同样的处理
    neg_text_inputs_1 = tokenizer(
        negative_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    neg_text_inputs_2 = tokenizer_2(
        negative_prompt,
        padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        neg_output_1 = text_encoder(
            neg_text_inputs_1.input_ids.to(device),
            output_hidden_states=True,
        )
        negative_prompt_embeds_1 = neg_output_1.hidden_states[-2]

        neg_output_2 = text_encoder_2(
            neg_text_inputs_2.input_ids.to(device),
            output_hidden_states=True,
        )
        negative_pooled_prompt_embeds = neg_output_2[0]
        negative_prompt_embeds_2 = neg_output_2.hidden_states[-2]

    negative_prompt_embeds = torch.cat([negative_prompt_embeds_1, negative_prompt_embeds_2], dim=-1)

    return {
        'prompt_embeds': prompt_embeds,
        'pooled_prompt_embeds': pooled_prompt_embeds,
        'negative_prompt_embeds': negative_prompt_embeds,
        'negative_pooled_prompt_embeds': negative_pooled_prompt_embeds,
    }


def main():
    parser = argparse.ArgumentParser(description='SDXL 阶段1：Text Encoder (CPU)')
    parser.add_argument('--prompt', type=str, default=DEFAULT_PROMPT)
    parser.add_argument('--negative_prompt', type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument('--output_dir', type=str, default='./stage_outputs')
    parser.add_argument('--model_id', type=str, default=MODEL_NAME)
    parser.add_argument('--height', type=int, default=HEIGHT)
    parser.add_argument('--width', type=int, default=WIDTH)
    parser.add_argument('--num_inference_steps', type=int, default=NUM_STEPS)
    parser.add_argument('--guidance_scale', type=float, default=GUIDANCE_SCALE)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    warnings.filterwarnings('ignore')
    setup_jax_cache()
    paths = get_default_paths(args.output_dir)

    print(f"\n{'='*60}")
    print("SDXL 阶段1：Text Encoder (CPU)")
    print(f"{'='*60}")

    # 加载 tokenizers
    print("\n=== 加载 CLIP Tokenizers ===")
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer_2")
    print("✓ Tokenizers 加载成功")

    # 加载 text encoders
    print("\n=== 加载 CLIP Text Encoders (CPU) ===")
    text_encoder = CLIPTextModel.from_pretrained(
        args.model_id, subfolder="text_encoder", torch_dtype=torch.float16
    )
    text_encoder.eval()
    print("✓ text_encoder (CLIPTextModel) 加载成功")

    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.model_id, subfolder="text_encoder_2", torch_dtype=torch.float16
    )
    text_encoder_2.eval()
    print("✓ text_encoder_2 (CLIPTextModelWithProjection) 加载成功")

    # 编码 prompt
    print(f"\n编码 prompt (CPU)...")
    print(f"  Prompt: {args.prompt[:60]}...")
    print(f"  Negative: {args.negative_prompt[:60]}...")

    embeddings = encode_prompt(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        device="cpu",
    )

    print(f"✓ prompt_embeds shape: {embeddings['prompt_embeds'].shape}")
    print(f"✓ pooled_prompt_embeds shape: {embeddings['pooled_prompt_embeds'].shape}")
    print(f"✓ negative_prompt_embeds shape: {embeddings['negative_prompt_embeds'].shape}")
    print(f"✓ negative_pooled_prompt_embeds shape: {embeddings['negative_pooled_prompt_embeds'].shape}")

    # 保存 embeddings
    save_embeddings_to_safetensors(embeddings, paths['embeddings'])

    # 保存配置
    config = {
        'model_id': args.model_id,
        'prompt': args.prompt,
        'negative_prompt': args.negative_prompt,
        'height': args.height,
        'width': args.width,
        'num_inference_steps': args.num_inference_steps,
        'guidance_scale': args.guidance_scale,
        'seed': args.seed,
    }
    save_generation_config(config, paths['config'])

    # 清理内存
    del text_encoder, text_encoder_2, tokenizer, tokenizer_2
    gc.collect()

    print(f"\n{'='*60}")
    print("✓ 阶段1 完成！下一步：运行 stage2_unet.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
