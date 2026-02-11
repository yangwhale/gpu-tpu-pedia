#!/usr/bin/env python3
"""
SDXL Text-to-Image 生成脚本 (TPU)

使用 Torchax + JAX 在 TPU 上运行 SDXL 图像生成。
包含完整的 Text Encoder + UNet + VAE 流程。

使用方法：
    python generate_torchax.py --prompt "Your prompt here" --output output.png
"""

import os
import warnings
import logging

# 环境配置（必须在其他 import 之前）
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')
for logger_name in ['root', '', 'diffusers', 'transformers']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

import argparse
import functools
import gc
import time
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torchax
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import EulerDiscreteScheduler
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.tree_util import register_pytree_node
from PIL import Image
from torchax.ops import jaten, ops_registry
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from splash_attention_utils import sdpa_reference


# ============================================================================
# 配置常量
# ============================================================================

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
WIDTH, HEIGHT = 1024, 1024
NUM_STEPS = 25
GUIDANCE_SCALE = 7.5

DEFAULT_PROMPT = (
    "A majestic lion with a flowing golden mane, standing on a rocky cliff "
    "overlooking a vast savanna at sunset, dramatic lighting, photorealistic, "
    "8k ultra detailed, cinematic composition"
)
DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, distorted, deformed"


# ============================================================================
# UNet 分片策略
# ============================================================================

UNET_SHARDINGS = {
    # Down blocks - Attention
    r'down_blocks.*.attentions.*.transformer_blocks.*.attn1.to_q.weight': ('tp', None),
    r'down_blocks.*.attentions.*.transformer_blocks.*.attn1.to_k.weight': ('tp', None),
    r'down_blocks.*.attentions.*.transformer_blocks.*.attn1.to_v.weight': ('tp', None),
    r'down_blocks.*.attentions.*.transformer_blocks.*.attn1.to_out.0.weight': (None, 'tp'),
    r'down_blocks.*.attentions.*.transformer_blocks.*.attn2.to_q.weight': ('tp', None),
    r'down_blocks.*.attentions.*.transformer_blocks.*.attn2.to_k.weight': ('tp', None),
    r'down_blocks.*.attentions.*.transformer_blocks.*.attn2.to_v.weight': ('tp', None),
    r'down_blocks.*.attentions.*.transformer_blocks.*.attn2.to_out.0.weight': (None, 'tp'),
    r'down_blocks.*.attentions.*.transformer_blocks.*.ff.net.0.proj.weight': ('tp', None),
    r'down_blocks.*.attentions.*.transformer_blocks.*.ff.net.2.weight': (None, 'tp'),
    # Mid block - Attention
    r'mid_block.attentions.*.transformer_blocks.*.attn1.to_q.weight': ('tp', None),
    r'mid_block.attentions.*.transformer_blocks.*.attn1.to_k.weight': ('tp', None),
    r'mid_block.attentions.*.transformer_blocks.*.attn1.to_v.weight': ('tp', None),
    r'mid_block.attentions.*.transformer_blocks.*.attn1.to_out.0.weight': (None, 'tp'),
    r'mid_block.attentions.*.transformer_blocks.*.attn2.to_q.weight': ('tp', None),
    r'mid_block.attentions.*.transformer_blocks.*.attn2.to_k.weight': ('tp', None),
    r'mid_block.attentions.*.transformer_blocks.*.attn2.to_v.weight': ('tp', None),
    r'mid_block.attentions.*.transformer_blocks.*.attn2.to_out.0.weight': (None, 'tp'),
    r'mid_block.attentions.*.transformer_blocks.*.ff.net.0.proj.weight': ('tp', None),
    r'mid_block.attentions.*.transformer_blocks.*.ff.net.2.weight': (None, 'tp'),
    # Up blocks - Attention
    r'up_blocks.*.attentions.*.transformer_blocks.*.attn1.to_q.weight': ('tp', None),
    r'up_blocks.*.attentions.*.transformer_blocks.*.attn1.to_k.weight': ('tp', None),
    r'up_blocks.*.attentions.*.transformer_blocks.*.attn1.to_v.weight': ('tp', None),
    r'up_blocks.*.attentions.*.transformer_blocks.*.attn1.to_out.0.weight': (None, 'tp'),
    r'up_blocks.*.attentions.*.transformer_blocks.*.attn2.to_q.weight': ('tp', None),
    r'up_blocks.*.attentions.*.transformer_blocks.*.attn2.to_k.weight': ('tp', None),
    r'up_blocks.*.attentions.*.transformer_blocks.*.attn2.to_v.weight': ('tp', None),
    r'up_blocks.*.attentions.*.transformer_blocks.*.attn2.to_out.0.weight': (None, 'tp'),
    r'up_blocks.*.attentions.*.transformer_blocks.*.ff.net.0.proj.weight': ('tp', None),
    r'up_blocks.*.attentions.*.transformer_blocks.*.ff.net.2.weight': (None, 'tp'),
    # Time embedding
    r'time_embedding.linear_1.weight': ('tp', None),
    r'time_embedding.linear_2.weight': (None, 'tp'),
    # Add embedding (SDXL 特有)
    r'add_embedding.linear_1.weight': ('tp', None),
    r'add_embedding.linear_2.weight': (None, 'tp'),
}


# ============================================================================
# 辅助函数
# ============================================================================

def setup_pytree_registrations():
    """注册必要的 PyTree 节点以支持 JAX 转换。"""
    from diffusers.models.autoencoders import vae as diffusers_vae
    from diffusers.models import modeling_outputs as diffusers_modeling_outputs

    print("注册 PyTree 节点...")

    def flatten(obj):
        return obj.to_tuple(), type(obj)

    def unflatten(aux, children):
        return aux(*children)

    classes = [
        (diffusers_vae.DecoderOutput, "DecoderOutput"),
        (diffusers_modeling_outputs.AutoencoderKLOutput, "AutoencoderKLOutput"),
    ]

    try:
        from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
        classes.append((UNet2DConditionOutput, "UNet2DConditionOutput"))
    except ImportError:
        pass

    for cls, name in classes:
        try:
            register_pytree_node(cls, flatten, unflatten)
            print(f"  - {name} 已注册")
        except ValueError:
            print(f"  - {name} 已存在")


import re
from jax.sharding import NamedSharding, PartitionSpec as P


def shard_weight_dict(weight_dict, sharding_dict, mesh):
    """按模式匹配应用权重分片。"""
    result = {}
    sharded_count = replicated_count = 0
    sharded_bytes = replicated_bytes = 0

    for k, v in weight_dict.items():
        tensor_bytes = np.prod(v.shape) * 2 if hasattr(v, 'shape') else 0

        if isinstance(v, torch.Tensor):
            with jax.default_device("cpu"):
                v = v.to("jax")

        matched = False
        for target, sharding in sharding_dict.items():
            if re.fullmatch(target, k) is not None:
                v.apply_jax_(jax.device_put, NamedSharding(mesh, P(*sharding)))
                matched = True
                sharded_count += 1
                sharded_bytes += tensor_bytes
                break

        if not matched:
            v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
            replicated_count += 1
            replicated_bytes += tensor_bytes

        result[k] = v

    print(f"  分片统计: {sharded_count} 个分片 ({sharded_bytes/1e9:.2f}GB), "
          f"{replicated_count} 个复制 ({replicated_bytes/1e9:.2f}GB)")
    return result


def move_module_to_xla(env, module):
    """将模块权重移动到 XLA 设备。"""
    with jax.default_device("cpu"):
        state_dict = module.state_dict()
        state_dict = env.to_xla(state_dict)
        module.load_state_dict(state_dict, assign=True)


def torch_conv2d_jax(input, weight, bias=None, stride=1, padding=0,
                     dilation=1, groups=1, *, env):
    """JAX 兼容的 conv2d 覆盖实现。"""
    jinput, jweight, jbias = env.t2j_iso((input, weight, bias))
    res = jaten._aten_conv2d(jinput, jweight, jbias, stride, padding, dilation, groups)
    return env.j2t_iso(res)


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                  is_causal=False, scale=None, enable_gqa=False,
                                  env=None, mesh=None):
    """SDPA：SDXL 序列较短，使用参考实现。"""
    return sdpa_reference(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)


def override_op(env, op, impl):
    """覆盖 torchax 算子。"""
    env._ops[op] = ops_registry.Operator(
        op, impl, is_jax_function=False, is_user_defined=True,
        needs_env=False, is_view_op=False,
    )


# ============================================================================
# Text Encoding
# ============================================================================

def encode_prompt(
    prompt: str,
    negative_prompt: str,
    tokenizer: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    text_encoder_2: CLIPTextModelWithProjection,
) -> dict:
    """编码 prompt 和 negative prompt。"""
    # Tokenize prompts
    text_inputs_1 = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    text_inputs_2 = tokenizer_2(
        prompt, padding="max_length", max_length=tokenizer_2.model_max_length,
        truncation=True, return_tensors="pt",
    )

    with torch.no_grad():
        text_output_1 = text_encoder(text_inputs_1.input_ids, output_hidden_states=True)
        prompt_embeds_1 = text_output_1.hidden_states[-2]

        text_output_2 = text_encoder_2(text_inputs_2.input_ids, output_hidden_states=True)
        pooled_prompt_embeds = text_output_2[0]
        prompt_embeds_2 = text_output_2.hidden_states[-2]

    prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)

    # Negative prompt
    neg_text_inputs_1 = tokenizer(
        negative_prompt, padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    )
    neg_text_inputs_2 = tokenizer_2(
        negative_prompt, padding="max_length", max_length=tokenizer_2.model_max_length,
        truncation=True, return_tensors="pt",
    )

    with torch.no_grad():
        neg_output_1 = text_encoder(neg_text_inputs_1.input_ids, output_hidden_states=True)
        negative_prompt_embeds_1 = neg_output_1.hidden_states[-2]

        neg_output_2 = text_encoder_2(neg_text_inputs_2.input_ids, output_hidden_states=True)
        negative_pooled_prompt_embeds = neg_output_2[0]
        negative_prompt_embeds_2 = neg_output_2.hidden_states[-2]

    negative_prompt_embeds = torch.cat([negative_prompt_embeds_1, negative_prompt_embeds_2], dim=-1)

    return {
        'prompt_embeds': prompt_embeds,
        'pooled_prompt_embeds': pooled_prompt_embeds,
        'negative_prompt_embeds': negative_prompt_embeds,
        'negative_pooled_prompt_embeds': negative_pooled_prompt_embeds,
    }


# ============================================================================
# UNet Denoising
# ============================================================================

def get_add_time_ids(original_size, crops_coords_top_left, target_size, dtype):
    """生成 SDXL 的 add_time_ids。"""
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    return torch.tensor([add_time_ids], dtype=dtype)


def run_denoising(unet, scheduler, embeddings, config, mesh, env):
    """运行 denoising loop。"""
    height, width = config['height'], config['width']
    guidance_scale = config['guidance_scale']
    num_steps = config['num_inference_steps']
    seed = config['seed']

    prompt_embeds = embeddings['prompt_embeds']
    pooled_prompt_embeds = embeddings['pooled_prompt_embeds']
    negative_prompt_embeds = embeddings['negative_prompt_embeds']
    negative_pooled_prompt_embeds = embeddings['negative_pooled_prompt_embeds']

    # 初始化 latents
    generator = torch.Generator()
    generator.manual_seed(seed)

    vae_scale_factor = 8
    latent_height = height // vae_scale_factor
    latent_width = width // vae_scale_factor
    latents_shape = (1, 4, latent_height, latent_width)
    latents = torch.randn(latents_shape, generator=generator, dtype=torch.float32)

    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps
    latents = latents * scheduler.init_noise_sigma

    # 准备 add_time_ids
    original_size = (height, width)
    target_size = (height, width)
    crops_coords_top_left = (0, 0)
    add_time_ids = get_add_time_ids(original_size, crops_coords_top_left, target_size, prompt_embeds.dtype)

    # CFG: 拼接
    prompt_embeds_cfg = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    pooled_embeds_cfg = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    add_time_ids_cfg = torch.cat([add_time_ids, add_time_ids], dim=0)

    # 转换到 XLA
    with env:
        latents = latents.to('jax').to(torch.bfloat16)
        prompt_embeds_cfg = prompt_embeds_cfg.to('jax').to(torch.bfloat16)
        pooled_embeds_cfg = pooled_embeds_cfg.to('jax').to(torch.bfloat16)
        add_time_ids_cfg = add_time_ids_cfg.to('jax').to(torch.bfloat16)

    with mesh:
        progress = tqdm(total=num_steps, desc="Denoising", ncols=100)
        step_start = [time.perf_counter()]

        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # timestep
            if isinstance(t, torch.Tensor):
                t_val = t.item()
            else:
                t_val = float(t)
            timestep = env.j2t_iso(jnp.array([t_val], dtype=jnp.int32))

            added_cond_kwargs = {
                "text_embeds": pooled_embeds_cfg,
                "time_ids": add_time_ids_cfg,
            }

            noise_pred = unet(
                latent_model_input, timestep,
                encoder_hidden_states=prompt_embeds_cfg,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            jax.effects_barrier()
            step_time = time.perf_counter() - step_start[0]
            progress.set_postfix({'step': f'{step_time:.2f}s'})
            progress.update(1)
            step_start[0] = time.perf_counter()

        progress.close()

    return latents


# ============================================================================
# VAE Decode
# ============================================================================

def decode_latents(vae, latents, env):
    """将 latents 解码为图像。"""
    scaling_factor = 0.13025
    latents = latents / scaling_factor

    with torch.no_grad():
        image = vae.decode(latents, return_dict=False)[0]

    image = (image / 2 + 0.5).clamp(0, 1)
    return image


def tensor_to_pil(image_tensor):
    """将 tensor 转换为 PIL Image。"""
    if hasattr(image_tensor, '_elem'):
        jax_array = image_tensor._elem
        if jax_array.dtype == jnp.bfloat16:
            np_array = np.array(jax_array.astype(jnp.float32))
        else:
            np_array = np.array(jax_array)
    else:
        np_array = image_tensor.cpu().float().numpy()

    np_array = np.transpose(np_array, (0, 2, 3, 1))
    np_array = (np_array * 255).round().astype(np.uint8)
    return Image.fromarray(np_array[0])


# ============================================================================
# 主函数
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="SDXL TPU 图像生成")
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--num_inference_steps", type=int, default=NUM_STEPS)
    parser.add_argument("--guidance_scale", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print("SDXL Text-to-Image 生成 (TPU)")
    print(f"{'='*60}")

    # 配置 JAX
    jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax_cache"))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    torch.set_default_dtype(torch.bfloat16)
    setup_pytree_registrations()

    # 加载 text encoders (CPU)
    print("\n=== 加载 Text Encoders (CPU) ===")
    tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder", torch_dtype=torch.float16)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.model_id, subfolder="text_encoder_2", torch_dtype=torch.float16)
    text_encoder.eval()
    text_encoder_2.eval()
    print("✓ Text Encoders 加载成功")

    # 编码 prompt
    print(f"\n编码 prompt...")
    print(f"  Prompt: {args.prompt[:60]}...")
    embeddings = encode_prompt(
        args.prompt, args.negative_prompt,
        tokenizer, tokenizer_2, text_encoder, text_encoder_2,
    )
    print(f"✓ prompt_embeds shape: {embeddings['prompt_embeds'].shape}")

    # 清理 text encoders
    del text_encoder, text_encoder_2, tokenizer, tokenizer_2
    gc.collect()

    # 加载 UNet 和 VAE
    print("\n=== 加载 UNet 和 VAE ===")
    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet", torch_dtype=torch.bfloat16)
    unet.eval()
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae", torch_dtype=torch.bfloat16)
    vae.eval()
    scheduler = EulerDiscreteScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    print("✓ UNet 和 VAE 加载成功")

    # 启用 torchax
    torchax.enable_globally()
    env = torchax.default_env()

    # 创建 mesh
    tp_dim = len(jax.devices())
    mesh = Mesh(mesh_utils.create_device_mesh((tp_dim,), allow_split_physical_axes=True), ("tp",))
    print(f"\nMesh: tp={tp_dim}, 总设备数={len(jax.devices())}")

    # 配置 UNet
    print("\n=== 配置 UNet (TPU) ===")
    override_op(env, torch.nn.functional.conv2d, functools.partial(torch_conv2d_jax, env=env))
    override_op(env, torch.nn.functional.scaled_dot_product_attention,
                functools.partial(scaled_dot_product_attention, env=env, mesh=mesh))

    move_module_to_xla(env, unet)
    unet = torchax.compile(unet, torchax.CompileOptions(jax_jit_kwargs={'static_argnames': ('return_dict',)}))
    unet.params = shard_weight_dict(unet.params, UNET_SHARDINGS, mesh)
    unet.buffers = shard_weight_dict(unet.buffers, UNET_SHARDINGS, mesh)
    torchax.interop.call_jax(jax.block_until_ready, unet.params)
    print("✓ UNet 配置完成")

    # 配置 VAE
    print("\n=== 配置 VAE (TPU) ===")
    move_module_to_xla(env, vae)
    vae.decoder = torchax.compile(vae.decoder)
    print("✓ VAE 配置完成")

    config = {
        'height': args.height,
        'width': args.width,
        'num_inference_steps': args.num_inference_steps,
        'guidance_scale': args.guidance_scale,
        'seed': args.seed,
    }

    # 预热
    if args.warmup_steps > 0:
        print(f"\n{'='*60}")
        print(f"预热运行 ({args.warmup_steps} 步)")
        print(f"{'='*60}")
        warmup_config = {**config, 'num_inference_steps': args.warmup_steps}
        warmup_start = time.perf_counter()
        _ = run_denoising(unet, scheduler, embeddings, warmup_config, mesh, env)
        jax.effects_barrier()
        print(f"✓ 预热完成: {time.perf_counter() - warmup_start:.2f}s")

    # 推理
    print(f"\n{'='*60}")
    print(f"基准测试运行 ({args.num_inference_steps} 步)")
    print(f"{'='*60}")
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    benchmark_start = time.perf_counter()
    latents = run_denoising(unet, scheduler, embeddings, config, mesh, env)
    jax.effects_barrier()
    unet_time = time.perf_counter() - benchmark_start

    # VAE decode
    print("\n解码 latents...")
    decode_start = time.perf_counter()
    with mesh:
        image_tensor = decode_latents(vae, latents, env)
        jax.effects_barrier()
    decode_time = time.perf_counter() - decode_start
    print(f"✓ VAE 解码完成: {decode_time:.2f}s")

    # 保存图像
    image = tensor_to_pil(image_tensor)
    output_path = args.output or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    image.save(output_path)

    total_time = unet_time + decode_time
    print(f"\n{'='*60}")
    print("性能统计")
    print(f"{'='*60}")
    print(f"  UNet ({args.num_inference_steps} 步): {unet_time:.2f}s ({unet_time/args.num_inference_steps:.3f}s/step)")
    print(f"  VAE Decode: {decode_time:.2f}s")
    print(f"  总时间: {total_time:.2f}s")
    print(f"  图像保存至: {output_path}")

    print(f"\n{'='*60}")
    print("✓ 生成完成！")
    print(f"{'='*60}")

    os._exit(0)


if __name__ == '__main__':
    main()
