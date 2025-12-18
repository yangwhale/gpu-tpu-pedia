#!/usr/bin/env python3
"""
Wan 2.2 I2V 三阶段生成 - 阶段1：Text & Image Encoder (TPU)

本阶段负责：
1. 加载 WanImageToVideoPipeline
2. 启用 torchax 并移动模型到 TPU
3. 使用 UMT5-XXL Text Encoder 编码 prompt
4. 预处理图像并使用 VAE 编码为 latent_condition
5. 构建 condition 和 mask（A14B 模式）
6. 将所有 embeddings 保存为 SafeTensors 格式

输出文件：
- stage1_embeddings.safetensors: 包含所有 embeddings
- generation_config.json: 生成配置参数

注意：本实现仅支持 A14B 模式 (expand_timesteps=False)
"""

import os
import argparse
import warnings
import logging
import functools
import re

import numpy as np
import torch
import jax
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
from jax.experimental import mesh_utils

import torchax
from torchax.ops import ops_registry, jaten

from diffusers.pipelines.wan.pipeline_wan_i2v_torchax import WanImageToVideoPipeline
from diffusers.models.autoencoders.autoencoder_kl_wan_torchax import AutoencoderKLWan
from diffusers.utils import load_image

from utils import (
    MODEL_ID,
    SIZE_CONFIGS,
    MAX_AREA_CONFIGS,
    FRAMES,
    FPS,
    DEFAULT_PROMPT,
    DEFAULT_NEG_PROMPT,
    DEFAULT_IMAGE_PATH,
    TEXT_ENCODER_SHARDINGS,
    VAE_ENCODER_SHARDINGS,
    normalize_latents,
    save_embeddings_to_safetensors,
    save_generation_config,
    get_default_paths,
    setup_pytree_registrations,
)


# ============================================================================
# Torchax 帮助函数 (从 generate_i2v_flax.py 复制)
# ============================================================================

def torch_conv2d_jax(input, weight, bias=None, stride=1, padding=0,
                     dilation=1, groups=1, *, env):
    """JAX-compatible conv2d override."""
    jinput, jweight, jbias = env.t2j_iso((input, weight, bias))
    res = jaten._aten_conv2d(jinput, jweight, jbias, stride, padding,
                             dilation, groups)
    return env.j2t_iso(res)


def override_op_definition(env, op_to_override, op_impl):
    """Override operator definition in torchax environment."""
    env._ops[op_to_override] = ops_registry.Operator(
        op_to_override,
        op_impl,
        is_jax_function=False,
        is_user_defined=True,
        needs_env=False,
        is_view_op=False,
    )


def move_module_to_xla(env, module):
    """Move module weights to XLA devices."""
    with jax.default_device("cpu"):
        state_dict = module.state_dict()
        state_dict = env.to_xla(state_dict)
        module.load_state_dict(state_dict, assign=True)


def shard_weight_dict(weight_dict, sharding_dict, mesh):
    """Apply sharding to weights based on pattern matching."""
    result = {}
    for k, v in weight_dict.items():
        if isinstance(v, torch.Tensor):
            v = v.to("jax")
        
        matched = False
        for target, sharding in sharding_dict.items():
            if re.fullmatch(target, k) is not None:
                v.apply_jax_(jax.device_put, NamedSharding(mesh, P(*sharding)))
                matched = True
                break
        
        if not matched:
            # Replicate
            v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
        
        result[k] = v
    return result


# ============================================================================
# Pipeline 加载与设置
# ============================================================================

def load_pipeline(model_id):
    """Load pipeline BEFORE enabling torchax to avoid safetensors issues."""
    print("\n=== 加载 Wan 2.2 I2V Pipeline ===")
    print("加载模型中（在启用 torchax 之前）...")
    
    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        vae=AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.bfloat16
        ),
    )
    print("✓ 模型加载完成\n")
    return pipe


def setup_text_encoder_and_vae(pipe, mesh, env):
    """Setup Text Encoder and VAE for JAX/TPU execution."""
    print("=== 设置 Text Encoder 和 VAE (TPU) ===")
    
    # Register custom operators
    print("- 注册 JAX conv2d 操作...")
    override_op_definition(
        env,
        torch.nn.functional.conv2d,
        functools.partial(torch_conv2d_jax, env=env)
    )
    
    # Text Encoder
    print("- 移动 Text Encoder 到 TPU...")
    move_module_to_xla(env, pipe.text_encoder)
    pipe.text_encoder = torchax.compile(pipe.text_encoder)
    pipe.text_encoder.params = shard_weight_dict(
        pipe.text_encoder.params, TEXT_ENCODER_SHARDINGS, mesh
    )
    pipe.text_encoder.buffers = shard_weight_dict(
        pipe.text_encoder.buffers, TEXT_ENCODER_SHARDINGS, mesh
    )
    
    # VAE Encoder
    print("- 移动 VAE Encoder 到 TPU...")
    move_module_to_xla(env, pipe.vae)
    pipe.vae.encoder = torchax.compile(pipe.vae.encoder)
    pipe.vae.encoder.params = shard_weight_dict(
        pipe.vae.encoder.params, VAE_ENCODER_SHARDINGS, mesh
    )
    pipe.vae.encoder.buffers = shard_weight_dict(
        pipe.vae.encoder.buffers, VAE_ENCODER_SHARDINGS, mesh
    )
    
    print("✓ Text Encoder 和 VAE 设置完成\n")
    return pipe


# ============================================================================
# 编码函数
# ============================================================================

def encode_prompts(pipe, prompt, negative_prompt, device='jax', dtype=torch.bfloat16):
    """
    使用 Text Encoder 编码 prompt
    
    Wan 2.2 使用 UMT5-XXL Text Encoder
    """
    print(f"\n=== 编码文本提示词 ===")
    print(f"正面 prompt: {prompt[:100]}..." if len(prompt) > 100 else f"正面 prompt: {prompt}")
    print(f"负面 prompt: {negative_prompt[:50]}..." if len(negative_prompt) > 50 else f"负面 prompt: {negative_prompt}")
    
    # 编码正面 prompt
    print("\n编码正面 prompt...")
    prompt_embeds, _ = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=None,
        do_classifier_free_guidance=False,
        num_videos_per_prompt=1,
        device=device,
        dtype=dtype,
    )
    print(f"  prompt_embeds shape: {prompt_embeds.shape}, dtype: {prompt_embeds.dtype}")
    
    # 编码负面 prompt
    print("编码负面 prompt...")
    negative_prompt_embeds, _ = pipe.encode_prompt(
        prompt=negative_prompt,
        negative_prompt=None,
        do_classifier_free_guidance=False,
        num_videos_per_prompt=1,
        device=device,
        dtype=dtype,
    )
    print(f"  negative_prompt_embeds shape: {negative_prompt_embeds.shape}")
    
    return {
        'prompt_embeds': prompt_embeds,
        'negative_prompt_embeds': negative_prompt_embeds,
    }


def encode_image_condition(pipe, image, height, width, num_frames, device='jax', dtype=torch.bfloat16):
    """
    预处理图像并编码为 latent_condition（A14B 模式）
    
    A14B 模式 (expand_timesteps=False):
    1. 在像素空间构建完整 video_condition [image, zeros, zeros, ...]
    2. VAE 编码整个 video_condition
    3. 构建 mask 并与 latent_condition 拼接
    """
    print(f"\n=== 编码图像条件 ===")
    print(f"分辨率: {width}x{height}")
    print(f"帧数: {num_frames}")
    
    # 1. 加载和调整图像大小
    if isinstance(image, str):
        print(f"加载图像: {image}")
        image = load_image(image)
    
    # 调整图像大小以匹配目标分辨率
    image = image.resize((width, height))
    print(f"  图像调整为: {image.size}")
    
    # 2. 预处理图像
    print("预处理图像...")
    image_tensor = pipe.video_processor.preprocess(image, height=height, width=width)
    # image_tensor: [1, 3, H, W]
    print(f"  预处理后 shape: {image_tensor.shape}")
    
    # 3. 构建 video_condition（像素空间）
    # A14B 模式：[image, zeros, zeros, ...]
    print("构建 video_condition（A14B 模式）...")
    batch_size = 1
    
    # 将图像扩展为 [B, 3, 1, H, W]
    image_tensor = image_tensor.unsqueeze(2).to(device, dtype=dtype)
    
    # 创建零填充帧
    zeros = torch.zeros(
        batch_size, 3, num_frames - 1, height, width,
        device=device, dtype=dtype
    )
    
    # 拼接：[B, 3, num_frames, H, W]
    video_condition = torch.cat([image_tensor, zeros], dim=2)
    print(f"  video_condition shape: {video_condition.shape}")
    
    # 4. VAE 编码
    print("VAE 编码 video_condition...")
    # Wan VAE 期望输入格式: [B, C, T, H, W]
    # video_condition 已经是 [B, C, T, H, W] = [1, 3, 21, H, W]
    
    with torch.no_grad():
        latent_dist = pipe.vae.encode(video_condition)
        latent_condition = latent_dist.latent_dist.mode()
    
    # latent_condition 输出格式: [B, C, T', H', W'] 其中 C=16
    print(f"  latent_condition shape: {latent_condition.shape}")
    
    # 5. 归一化 latents（使用 VAE 的 config 中的参数）
    print("归一化 latent_condition...")
    latent_condition = normalize_latents(latent_condition, vae=pipe.vae)
    
    # 6. 构建 mask（A14B 模式）
    # 与原始 pipeline 完全一致的实现
    print("构建 mask...")
    
    vae_scale_factor_temporal = 4
    vae_scale_factor_spatial = 8
    
    num_latent_frames = (num_frames - 1) // vae_scale_factor_temporal + 1
    latent_height = height // vae_scale_factor_spatial
    latent_width = width // vae_scale_factor_spatial
    
    # 构建 mask（与原始 pipeline_wan_i2v_torchax.py 第469-482行完全一致）
    mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width,
                               device=device, dtype=dtype)
    mask_lat_size[:, :, list(range(1, num_frames))] = 0  # 第一帧=1, 其他帧=0
    
    # 处理第一帧的时间扩展
    first_frame_mask = mask_lat_size[:, :, 0:1]
    first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=vae_scale_factor_temporal)
    
    # 拼接并 reshape
    mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
    mask_lat_size = mask_lat_size.view(batch_size, -1, vae_scale_factor_temporal, latent_height, latent_width)
    mask_lat_size = mask_lat_size.transpose(1, 2)
    mask_lat_size = mask_lat_size.to(latent_condition.device)
    
    print(f"  mask shape: {mask_lat_size.shape}")
    
    # 7. 拼接 mask 和 latent_condition
    condition = torch.cat([mask_lat_size, latent_condition], dim=1)
    print(f"  condition shape: {condition.shape}")
    
    return {
        'condition': condition,
        'num_latent_frames': num_latent_frames,
        'latent_height': latent_height,
        'latent_width': latent_width,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Wan 2.2 I2V 阶段1：Text & Image Encoder (TPU)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 基本用法
  python stage1_encoder.py --image "path/to/image.jpg" --prompt "描述文本..."
  
  # 指定输出目录和尺寸
  python stage1_encoder.py --image input.jpg --size "720*1280" --output_dir ./my_outputs
        """
    )
    
    # 图像和提示词参数
    parser.add_argument('--image', type=str, default=DEFAULT_IMAGE_PATH)
    parser.add_argument('--prompt', type=str, default=DEFAULT_PROMPT)
    parser.add_argument('--negative_prompt', type=str, default=DEFAULT_NEG_PROMPT)
    
    # 视频参数
    parser.add_argument('--size', type=str, default='720*1280', choices=list(SIZE_CONFIGS.keys()))
    parser.add_argument('--frames', type=int, default=FRAMES)
    parser.add_argument('--fps', type=int, default=FPS)
    
    # 模型参数
    parser.add_argument('--model_id', type=str, default=MODEL_ID)
    
    # Mesh 参数
    parser.add_argument('--dp', type=int, default=2, help='Data parallelism dimension')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./stage_outputs')
    
    args = parser.parse_args()
    
    # 设置输出路径
    paths = get_default_paths(args.output_dir)
    
    # 配置日志
    warnings.filterwarnings('ignore')
    logging.getLogger().setLevel(logging.ERROR)
    
    # 配置 JAX
    jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    
    torch.set_default_dtype(torch.bfloat16)
    
    print(f"\n{'='*60}")
    print("Wan 2.2 I2V 阶段1：Text & Image Encoder (TPU)")
    print(f"{'='*60}")
    
    # Setup PyTree registrations
    setup_pytree_registrations()
    
    # 解析尺寸
    if args.size in SIZE_CONFIGS:
        height, width = SIZE_CONFIGS[args.size]
    else:
        height, width = map(int, args.size.split('*'))
    
    # 根据图像宽高比调整尺寸
    image = load_image(args.image)
    max_area = MAX_AREA_CONFIGS.get(args.size, height * width)
    aspect_ratio = image.height / image.width
    
    # 计算合适的尺寸
    mod_value = 8 * 2
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    
    print(f"\n配置参数：")
    print(f"  模型: {args.model_id}")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧数: {args.frames}")
    print(f"  FPS: {args.fps}")
    
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
    
    # 设置 Text Encoder 和 VAE
    with mesh:
        pipe = setup_text_encoder_and_vae(pipe, mesh, env)
        
        # 编码 prompt
        prompt_dict = encode_prompts(
            pipe,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            device='jax',
            dtype=torch.bfloat16,
        )
        
        # 编码图像条件
        condition_dict = encode_image_condition(
            pipe,
            image=args.image,
            height=height,
            width=width,
            num_frames=args.frames,
            device='jax',
            dtype=torch.bfloat16,
        )
    
    # 转换回 CPU tensor 以保存
    print("\n转换 tensor 到 CPU...")
    embeddings_dict = {}
    for key in ['prompt_embeds', 'negative_prompt_embeds']:
        tensor = prompt_dict[key]
        if hasattr(tensor, 'to'):
            embeddings_dict[key] = tensor.to('cpu')
        else:
            embeddings_dict[key] = tensor
    
    condition_tensor = condition_dict['condition']
    if hasattr(condition_tensor, 'to'):
        embeddings_dict['condition'] = condition_tensor.to('cpu')
    else:
        embeddings_dict['condition'] = condition_tensor
    
    # 保存 embeddings
    print(f"\n保存 embeddings 到: {paths['embeddings']}")
    metadata = {
        'prompt': args.prompt,
        'negative_prompt': args.negative_prompt,
        'model_id': args.model_id,
        'height': str(height),
        'width': str(width),
        'num_frames': str(args.frames),
        'num_latent_frames': str(condition_dict['num_latent_frames']),
        'latent_height': str(condition_dict['latent_height']),
        'latent_width': str(condition_dict['latent_width']),
        'expand_timesteps': 'False',
    }
    save_embeddings_to_safetensors(embeddings_dict, paths['embeddings'], metadata)
    
    # 保存配置
    config = {
        'prompt': args.prompt,
        'negative_prompt': args.negative_prompt,
        'model_id': args.model_id,
        'height': height,
        'width': width,
        'num_frames': args.frames,
        'fps': args.fps,
        'num_latent_frames': condition_dict['num_latent_frames'],
        'latent_height': condition_dict['latent_height'],
        'latent_width': condition_dict['latent_width'],
        'expand_timesteps': False,
    }
    save_generation_config(config, paths['config'])
    
    print(f"\n{'='*60}")
    print("阶段1 完成！")
    print(f"{'='*60}")
    print(f"\n输出文件：")
    print(f"  - Embeddings: {paths['embeddings']}")
    print(f"  - 配置文件:   {paths['config']}")
    print(f"\n保存的 tensor shapes：")
    for key, value in embeddings_dict.items():
        print(f"  - {key}: {value.shape}")
    print(f"\n下一步：运行 stage2_transformer.py 进行 Transformer 推理")
    
    # 清理内存
    del pipe
    del embeddings_dict
    
    print("\n✓ 阶段1 执行完成")


if __name__ == "__main__":
    main()