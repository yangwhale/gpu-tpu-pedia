#!/usr/bin/env python3
"""
SDXL 三阶段生成 - 阶段3：VAE Decoder (TPU)

加载 stage2 的 latents，在 TPU 上使用 VAE 解码为图像。

输入：stage2_latents.safetensors
输出：output_image.png
"""

import argparse
import functools
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torchax
from diffusers import AutoencoderKL
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from PIL import Image
from torchax.ops import jaten, ops_registry

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    MODEL_NAME,
    get_default_paths,
    load_generation_config,
    load_latents_from_safetensors,
    move_module_to_xla,
    setup_jax_cache,
    setup_pytree_registrations,
)


# ============================================================================
# Torchax 算子覆盖
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


# ============================================================================
# VAE 配置
# ============================================================================

def setup_vae(vae, mesh, env):
    """配置 VAE 用于 TPU 推理。"""
    print("\n=== 配置 VAE (TPU) ===")

    override_op(env, torch.nn.functional.conv2d, functools.partial(torch_conv2d_jax, env=env))

    move_module_to_xla(env, vae)
    vae.decoder = torchax.compile(vae.decoder)
    torchax.interop.call_jax(jax.block_until_ready, vae.decoder.params)

    print("✓ VAE 配置完成")
    return vae


def decode_latents(vae, latents, env):
    """将 latents 解码为图像。"""
    # SDXL VAE scaling factor
    scaling_factor = 0.13025

    # Scale latents
    latents = latents / scaling_factor

    # Decode
    with torch.no_grad():
        image = vae.decode(latents, return_dict=False)[0]

    # 后处理
    image = (image / 2 + 0.5).clamp(0, 1)

    return image


def tensor_to_pil(image_tensor):
    """将 tensor 转换为 PIL Image。"""
    # 转换为 numpy
    if hasattr(image_tensor, '_elem'):
        jax_array = image_tensor._elem
        if jax_array.dtype == jnp.bfloat16:
            np_array = np.array(jax_array.astype(jnp.float32))
        else:
            np_array = np.array(jax_array)
    else:
        np_array = image_tensor.cpu().float().numpy()

    # (batch, channels, height, width) -> (batch, height, width, channels)
    np_array = np.transpose(np_array, (0, 2, 3, 1))

    # Scale to 0-255
    np_array = (np_array * 255).round().astype(np.uint8)

    # 返回第一张图像
    return Image.fromarray(np_array[0])


def main():
    parser = argparse.ArgumentParser(description='SDXL 阶段3：VAE Decoder (TPU)')
    parser.add_argument('--input_dir', type=str, default='./stage_outputs')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--model_id', type=str, default=None)
    parser.add_argument('--warmup', action='store_true')
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    input_paths = get_default_paths(args.input_dir)
    output_paths = get_default_paths(output_dir)

    setup_jax_cache()
    warnings.filterwarnings('ignore')
    logging.getLogger().setLevel(logging.ERROR)
    torch.set_default_dtype(torch.bfloat16)
    setup_pytree_registrations()

    print(f"\n{'='*60}")
    print("SDXL 阶段3：VAE Decoder (TPU)")
    print(f"{'='*60}")

    # 加载配置和 latents
    config = load_generation_config(input_paths['config'])
    latents, _ = load_latents_from_safetensors(input_paths['latents'], restore_dtype=True)

    # 加载 VAE
    model_id = args.model_id or config.get('model_id', MODEL_NAME)
    print(f"\n加载 VAE: {model_id}")

    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.bfloat16
    )
    vae.eval()

    # 启用 torchax
    torchax.enable_globally()
    env = torchax.default_env()

    # 创建 mesh
    tp_dim = len(jax.devices())
    mesh = Mesh(mesh_utils.create_device_mesh((tp_dim,), allow_split_physical_axes=True), ("tp",))
    print(f"\nMesh: tp={tp_dim}")

    vae = setup_vae(vae, mesh, env)

    # 转换 latents 到 XLA
    with env:
        latents_xla = latents.to('jax')

    with mesh:
        # 预热
        if args.warmup:
            print("\n预热 VAE decoder...")
            warmup_start = time.perf_counter()
            _ = decode_latents(vae, latents_xla, env)
            jax.effects_barrier()
            print(f"✓ 预热完成: {time.perf_counter() - warmup_start:.2f}s")

        # 解码
        print("\n解码 latents...")
        decode_start = time.perf_counter()
        image_tensor = decode_latents(vae, latents_xla, env)
        jax.effects_barrier()
        decode_time = time.perf_counter() - decode_start
        print(f"✓ 解码完成: {decode_time:.2f}s")

    # 转换为 PIL 并保存
    image = tensor_to_pil(image_tensor)
    image.save(output_paths['image'])
    print(f"✓ 图像已保存到: {output_paths['image']}")

    print(f"\n{'='*60}")
    print("✓ 阶段3 完成！图像生成完毕。")
    print(f"{'='*60}")

    os._exit(0)


if __name__ == "__main__":
    main()
