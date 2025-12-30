#!/usr/bin/env python3
"""
Flux.2 GPU 阶段3：VAE Decoder

加载 stage2 生成的 latents，使用 VAE 解码为图像。

注意：Flux.2 的 latents 是 packed 格式 (B, H*W, C)，需要：
1. Unpack latents (使用 latent_ids)
2. 应用 batch normalization 反标准化
3. Unpatchify latents
4. VAE 解码
5. 后处理为 PIL 图像
"""

import warnings
import logging
import os
warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

import time
import argparse

import torch
from PIL import Image

from diffusers import AutoencoderKLFlux2
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor

from utils import (
    MODEL_NAME,
    WIDTH, HEIGHT,
    load_latents_from_safetensors,
    load_generation_config,
    get_default_paths,
)


# ============================================================================
# Latent Processing (from Flux2Pipeline)
# ============================================================================

def prepare_latent_ids(height, width, device=None):
    """
    Generates 4D position coordinates (T, H, W, L) for latent tensors.
    """
    t = torch.arange(1, device=device)  # [0] - time dimension
    h = torch.arange(height, device=device)
    w = torch.arange(width, device=device)
    l = torch.arange(1, device=device)  # [0] - layer dimension

    # Create position IDs: (H*W, 4)
    latent_ids = torch.cartesian_prod(t, h, w, l)

    # Expand to batch: (1, H*W, 4)
    latent_ids = latent_ids.unsqueeze(0)

    return latent_ids


def unpack_latents_with_ids(x, x_ids):
    """
    Using position ids to scatter tokens into place.
    x: (B, seq_len, C)
    x_ids: (B, seq_len, 4) - position coordinates (T, H, W, L)
    """
    x_list = []
    for data, pos in zip(x, x_ids):
        _, ch = data.shape
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)

        h = torch.max(h_ids) + 1
        w = torch.max(w_ids) + 1

        flat_ids = h_ids * w + w_ids

        out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

        # reshape from (H * W, C) to (H, W, C) and permute to (C, H, W)
        out = out.view(h, w, ch).permute(2, 0, 1)
        x_list.append(out)

    return torch.stack(x_list, dim=0)


def unpatchify_latents(latents):
    """
    Unpatchify: (B, C*4, H/2, W/2) -> (B, C, H, W)
    """
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)
    return latents


# ============================================================================
# VAE Functions
# ============================================================================

def load_vae(model_id, device):
    """Load VAE."""
    print(f"加载 VAE: {model_id}")
    vae = AutoencoderKLFlux2.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.bfloat16
    )
    vae.to(device)
    vae.eval()
    print("✓ VAE 加载完成")
    return vae


def process_latents_for_vae(latents, config, vae, device):
    """
    Process packed latents for VAE decoding.
    
    Steps:
    1. Create latent_ids based on config dimensions
    2. Unpack latents using position IDs
    3. Apply batch normalization denormalization
    4. Unpatchify latents
    """
    print(f"\n=== 处理 Latents ===")
    print(f"输入 latents: {latents.shape}, {latents.dtype}")
    
    # Move to device
    latents = latents.to(device)
    
    # Get dimensions
    height = config['height']
    width = config['width']
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    
    # Calculate latent dimensions (accounting for packing)
    latent_height = 2 * (int(height) // (vae_scale_factor * 2))
    latent_width = 2 * (int(width) // (vae_scale_factor * 2))
    
    print(f"  图像尺寸: {height}x{width}")
    print(f"  Latent 尺寸: {latent_height}x{latent_width} (packed: {latent_height//2}x{latent_width//2})")
    
    # Create latent_ids
    latent_ids = prepare_latent_ids(latent_height // 2, latent_width // 2, device=device)
    print(f"  Latent IDs: {latent_ids.shape}")
    
    # Unpack latents: (B, H*W, C) -> (B, C, H/2, W/2)
    latents = unpack_latents_with_ids(latents, latent_ids)
    print(f"  Unpacked latents: {latents.shape}")
    
    # Apply batch normalization denormalization
    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device, latents.dtype)
    bn_var = vae.bn.running_var.view(1, -1, 1, 1).to(device, latents.dtype)
    bn_std = torch.sqrt(bn_var + vae.config.batch_norm_eps)
    latents = latents * bn_std + bn_mean
    print(f"  After BN denorm: range [{latents.min():.4f}, {latents.max():.4f}]")
    
    # Unpatchify: (B, C*4, H/2, W/2) -> (B, C, H, W)
    latents = unpatchify_latents(latents)
    print(f"  Unpatchified latents: {latents.shape}")
    
    return latents


def decode_latents(vae, latents, config, device, warmup=True):
    """Decode latents to image."""
    print(f"\n=== VAE 解码 ===")
    
    # Handle nan
    nan_count = torch.isnan(latents).sum().item()
    if nan_count > 0:
        print(f"警告: 发现 {nan_count} 个 nan，替换为 0")
        latents = torch.nan_to_num(latents, nan=0.0)
    
    # Process latents (unpack, denorm, unpatchify)
    latents = process_latents_for_vae(latents, config, vae, device)
    
    # Convert to VAE dtype
    latents = latents.to(vae.dtype)
    
    # Warmup
    if warmup:
        print("Warmup...")
        start = time.perf_counter()
        with torch.no_grad():
            _ = vae.decode(latents, return_dict=False)[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(f"✓ Warmup: {elapsed:.2f}s")
    
    # Decode
    print("VAE Decode...")
    start = time.perf_counter()
    with torch.no_grad():
        image = vae.decode(latents, return_dict=False)[0]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    print(f"✓ VAE Decode: {elapsed:.2f}s")
    print(f"  Output image: {image.shape}")
    
    return image, elapsed


def postprocess_image(image):
    """Post-process decoded image to PIL."""
    print("\n=== 后处理图像 ===")
    
    # Move to CPU
    image = image.cpu()
    
    print(f"  Image tensor: {image.shape}, range [{image.min():.4f}, {image.max():.4f}]")
    
    # Use Flux2ImageProcessor for proper postprocessing
    image_processor = Flux2ImageProcessor(vae_scale_factor=16)
    pil_images = image_processor.postprocess(image, output_type="pil")
    
    print(f"  Output: {len(pil_images)} PIL image(s)")
    return pil_images


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Flux.2 GPU 阶段3：VAE Decoder')
    parser.add_argument('--input_dir', type=str, default='./stage_outputs')
    parser.add_argument('--output_image', type=str, default=None)
    parser.add_argument('--model_id', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--no_warmup', action='store_true')
    args = parser.parse_args()
    
    # Setup
    torch.set_default_dtype(torch.bfloat16)
    
    paths = get_default_paths(args.input_dir)
    
    print(f"\n{'='*50}")
    print("阶段3：VAE Decoder (GPU)")
    print(f"{'='*50}")
    
    # Load config
    config = load_generation_config(paths['config'])
    model_id = args.model_id or config.get('model_id', MODEL_NAME)
    output_image = args.output_image or paths['image']
    
    print(f"\n模型: {model_id}")
    print(f"设备: {args.device}")
    print(f"分辨率: {config.get('height', HEIGHT)}x{config.get('width', WIDTH)}")
    
    # Load latents
    print(f"\n加载 latents: {paths['latents']}")
    latents, metadata = load_latents_from_safetensors(paths['latents'], device='cpu', restore_dtype=True)
    print(f"  Latents shape: {latents.shape}")
    
    # Load VAE
    print()
    vae = load_vae(model_id, args.device)
    
    # Decode
    image, decode_time = decode_latents(vae, latents, config, args.device, warmup=not args.no_warmup)
    
    # Post-process
    pil_images = postprocess_image(image)
    
    # Save
    pil_images[0].save(output_image)
    
    # Summary
    print(f"\n{'='*50}")
    print(f"✓ 完成!")
    print(f"  解码耗时: {decode_time:.2f}s")
    print(f"  分辨率: {pil_images[0].size[0]}x{pil_images[0].size[1]}")
    print(f"  输出: {output_image}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
