#!/usr/bin/env python3
"""
Flux.2 三阶段生成 - 阶段3：VAE Decoder (TPU)

加载 stage2 的 latents，使用 VAE 解码生成最终图像。
Flux.2 latents 是 packed 格式，需要 unpack -> denorm -> unpatchify -> decode。

输入：stage2_latents.safetensors
输出：output_image.png
"""

import argparse
import functools
import logging
import os
import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torchax
from diffusers.models.autoencoders.autoencoder_kl_flux2_torchax import AutoencoderKLFlux2
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from torchax.ops import jaten, ops_registry

from utils import (
    HEIGHT,
    MODEL_NAME,
    VAE_DECODER_SHARDINGS,
    WIDTH,
    get_default_paths,
    load_generation_config,
    load_latents_from_safetensors,
    setup_jax_cache,
    setup_pytree_registrations,
    shard_weight_dict,
)

warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)


# ============================================================================
# Torchax 辅助函数
# ============================================================================

def torch_conv2d_jax(input, weight, bias=None, stride=1, padding=0,
                     dilation=1, groups=1, *, env):
    """JAX 兼容的 conv2d。"""
    jinput, jweight, jbias = env.t2j_iso((input, weight, bias))
    res = jaten._aten_conv2d(jinput, jweight, jbias, stride, padding, dilation, groups)
    return env.j2t_iso(res)


def override_op(env, op, impl):
    """覆盖 torchax 算子。"""
    env._ops[op] = ops_registry.Operator(
        op, impl, is_jax_function=False, is_user_defined=True,
        needs_env=False, is_view_op=False,
    )


def move_module_to_xla(env, module):
    """将模块移动到 XLA。"""
    with jax.default_device("cpu"):
        state_dict = env.to_xla(module.state_dict())
        module.load_state_dict(state_dict, assign=True)


# ============================================================================
# Latent 处理
# ============================================================================

def prepare_latent_ids(height, width, device=None):
    """生成 latent 位置坐标 (T, H, W, L)。"""
    t = torch.arange(1, device=device)
    h = torch.arange(height, device=device)
    w = torch.arange(width, device=device)
    l = torch.arange(1, device=device)
    latent_ids = torch.cartesian_prod(t, h, w, l)
    return latent_ids.unsqueeze(0)


def unpack_latents(x, x_ids):
    """使用位置 ID 展开 latents。x: (B, seq_len, C), x_ids: (B, seq_len, 4)。"""
    x_list = []
    for data, pos in zip(x, x_ids):
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)
        h, w = torch.max(h_ids) + 1, torch.max(w_ids) + 1
        flat_ids = h_ids * w + w_ids
        out = torch.zeros((h * w, data.shape[1]), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, data.shape[1]), data)
        out = out.view(h, w, data.shape[1]).permute(2, 0, 1)
        x_list.append(out)
    return torch.stack(x_list)


def unpatchify_latents(latents):
    """Unpatchify: (B, C*4, H/2, W/2) -> (B, C, H, W)。"""
    b, c, h, w = latents.shape
    latents = latents.reshape(b, c // 4, 2, 2, h, w)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(b, c // 4, h * 2, w * 2)


def process_latents(latents, config, vae):
    """处理 packed latents 用于 VAE 解码。"""
    print(f"\n=== 处理 Latents ===")
    print(f"输入: {latents.shape}")
    
    height, width = config['height'], config['width']
    vae_scale = 2 ** (len(vae.config.block_out_channels) - 1)
    latent_h = 2 * (height // (vae_scale * 2))
    latent_w = 2 * (width // (vae_scale * 2))
    
    # Unpack
    latent_ids = prepare_latent_ids(latent_h // 2, latent_w // 2, device=latents.device)
    latents = unpack_latents(latents, latent_ids)
    print(f"Unpacked: {latents.shape}")
    
    # Denormalize
    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    bn_var = vae.bn.running_var.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    latents = latents * torch.sqrt(bn_var + vae.config.batch_norm_eps) + bn_mean
    
    # Unpatchify
    latents = unpatchify_latents(latents)
    print(f"Unpatchified: {latents.shape}")
    
    return latents


# ============================================================================
# VAE 解码
# ============================================================================

def setup_vae(vae, mesh, env):
    """配置 VAE 用于 TPU。"""
    print("\n配置 VAE Decoder...")
    override_op(env, torch.nn.functional.conv2d, functools.partial(torch_conv2d_jax, env=env))
    move_module_to_xla(env, vae)
    vae.decoder = torchax.compile(vae.decoder)
    vae.decoder.params = shard_weight_dict(vae.decoder.params, VAE_DECODER_SHARDINGS, mesh)
    vae.decoder.buffers = shard_weight_dict(vae.decoder.buffers, VAE_DECODER_SHARDINGS, mesh)
    print("✓ VAE Decoder 配置完成")
    return vae


def decode_latents(vae, latents, config, env, warmup=True):
    """解码 latents 生成图像。"""
    # 处理 nan
    nan_count = torch.isnan(latents).sum().item()
    if nan_count > 0:
        print(f"警告: {nan_count} 个 nan，替换为 0")
        latents = torch.nan_to_num(latents, nan=0.0)
    
    # 处理并转换到 XLA
    latents = process_latents(latents, config, vae)
    latents = env.to_xla(latents.to(vae.dtype))
    
    # 预热
    if warmup:
        print("\nWarmup...")
        start = time.perf_counter()
        with torch.no_grad():
            vae.decode(latents, return_dict=False)
        jax.effects_barrier()
        print(f"✓ Warmup: {time.perf_counter() - start:.2f}s")
    
    # 解码
    print("\nVAE Decode...")
    start = time.perf_counter()
    with torch.no_grad():
        image = vae.decode(latents, return_dict=False)[0]
    jax.effects_barrier()
    elapsed = time.perf_counter() - start
    print(f"✓ VAE Decode: {elapsed:.2f}s")
    
    return image, elapsed


def postprocess_image(image):
    """后处理图像为 PIL 格式。"""
    if hasattr(image, '_elem'):
        jax_image = image._elem
        if jax_image.dtype == jnp.bfloat16:
            np_image = np.array(jax_image.astype(jnp.float32))
        else:
            np_image = np.array(jax_image)
        image = torch.from_numpy(np_image)
    else:
        image = image.cpu()
    
    processor = Flux2ImageProcessor(vae_scale_factor=16)
    return processor.postprocess(image, output_type="pil")


def main():
    parser = argparse.ArgumentParser(description='Flux.2 阶段3：VAE Decoder (TPU)')
    parser.add_argument('--input_dir', type=str, default='./stage_outputs')
    parser.add_argument('--output_image', type=str, default=None)
    parser.add_argument('--model_id', type=str, default=None)
    parser.add_argument('--no_warmup', action='store_true')
    args = parser.parse_args()
    
    setup_jax_cache()
    setup_pytree_registrations()
    torch.set_default_dtype(torch.bfloat16)
    
    paths = get_default_paths(args.input_dir)
    
    print(f"\n{'='*50}")
    print("Flux.2 阶段3：VAE Decoder (TPU)")
    print(f"{'='*50}")
    
    # 加载配置和 latents
    config = load_generation_config(paths['config'])
    model_id = args.model_id or config.get('model_id', MODEL_NAME)
    output_image = args.output_image or paths['image']
    
    print(f"\n模型: {model_id}")
    print(f"分辨率: {config.get('height', HEIGHT)}x{config.get('width', WIDTH)}")
    
    latents, _ = load_latents_from_safetensors(paths['latents'], restore_dtype=True)
    
    # 加载 VAE
    print(f"\n加载 VAE...")
    vae = AutoencoderKLFlux2.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.bfloat16)
    
    # 启用 torchax
    torchax.enable_globally()
    env = torchax.default_env()
    
    # 创建 mesh
    tp_dim = len(jax.devices())
    mesh = Mesh(mesh_utils.create_device_mesh((tp_dim,), allow_split_physical_axes=True), ("tp",))
    print(f"Mesh: tp={tp_dim}")
    
    # 解码
    with mesh:
        vae = setup_vae(vae, mesh, env)
        image, decode_time = decode_latents(vae, latents, config, env, warmup=not args.no_warmup)
    
    # 后处理并保存
    pil_images = postprocess_image(image)
    pil_images[0].save(output_image)
    
    print(f"\n{'='*50}")
    print(f"✓ 完成!")
    print(f"  解码耗时: {decode_time:.2f}s")
    print(f"  分辨率: {pil_images[0].size[0]}x{pil_images[0].size[1]}")
    print(f"  输出: {output_image}")
    print(f"{'='*50}")
    
    os._exit(0)


if __name__ == "__main__":
    main()
