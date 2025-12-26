#!/usr/bin/env python3
"""
Wan 2.2 Image-to-Video 生成脚本 (TPU Splash Attention)

使用 Torchax + JAX 在 TPU 上运行 Wan 2.2 I2V 生成。
"""

import os
import warnings
import logging

# ============================================================================
# 环境配置和 Warning 过滤（必须在其他 import 之前）
# ============================================================================

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

warnings.filterwarnings('ignore')

logging.getLogger('root').setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger('diffusers').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)

import time
import re
import functools
import argparse
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
from jax.experimental import mesh_utils

import torchax
from torchax.ops import ops_registry, jaten

from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.pipelines.wan.pipeline_wan_i2v_torchax import WanImageToVideoPipeline
from diffusers.models.autoencoders.autoencoder_kl_wan_torchax import AutoencoderKLWan
from diffusers.models.transformers.transformer_wan_torchax import WanTransformer3DModel
from diffusers.utils import export_to_video, load_image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'kernels'))
from splash_attention_utils import tpu_splash_attention, sdpa_reference


# ============================================================================
# 配置常量
# ============================================================================

MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

SIZE_CONFIGS = {
    "720*1280": (720, 1280),
    "1280*720": (1280, 720),
    "480*832": (480, 832),
    "832*480": (832, 480),
}

MAX_AREA_CONFIGS = {
    "720*1280": 720 * 1280,
    "1280*720": 1280 * 720,
    "480*832": 480 * 832,
    "832*480": 832 * 480,
}

DEFAULT_PROMPT = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
DEFAULT_NEG_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
DEFAULT_IMAGE_PATH = "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG"

FRAMES = 81
FPS = 16
NUM_STEPS = 40
GUIDANCE_SCALE = 3.5
DEFAULT_DP = 2

# K 平滑以提高数值稳定性
USE_K_SMOOTH = False

PROFILE_OUT_PATH = "/tmp/wan_prof"


# ============================================================================
# 分片策略
# ============================================================================

TEXT_ENCODER_SHARDINGS = {
    r'shared.weight': (('dp', 'tp'), None),
    r'encoder.block.*.layer.*.SelfAttention.[qkv].weight': (('dp', 'tp'), None),
    r'encoder.block.*.layer.*.SelfAttention.o.weight': (None, ('dp', 'tp')),
    r'encoder.block.*.layer.*.DenseReluDense.wi_[01].weight': (('dp', 'tp'), None),
    r'encoder.block.*.layer.*.DenseReluDense.wo.weight': (None, ('dp', 'tp')),
}

TRANSFORMER_SHARDINGS = {
    r'condition_embedder.*.linear_1.weight': ('tp', None),
    r'condition_embedder.*.linear_1.bias': ('tp',),
    r'condition_embedder.*.linear_2.weight': (None, 'tp'),
    r'blocks.*.attn[12].to_[qkv].weight': ('tp', None),
    r'blocks.*.attn[12].to_[qkv].bias': ('tp',),
    r'blocks.*.attn[12].to_out.*.weight': (None, 'tp'),
    r'blocks.*.ffn.net.*.proj.weight': ('tp', None),
    r'blocks.*.ffn.net.*.proj.bias': ('tp',),
    r'blocks.*.ffn.net.*.weight': (None, 'tp'),
}


# ============================================================================
# 辅助函数
# ============================================================================

def setup_pytree_registrations():
    """注册 PyTree 节点以支持 JAX 转换。"""
    print("注册 PyTree 节点...")
    
    def flatten(obj):
        return obj.to_tuple(), type(obj)
    
    def unflatten(aux, children):
        return aux(*children)
    
    for cls in [BaseModelOutputWithPastAndCrossAttentions, DecoderOutput, AutoencoderKLOutput]:
        jax.tree_util.register_pytree_node(cls, flatten, unflatten)
        print(f"  - {cls.__name__} 已注册")
    
    # DiagonalGaussianDistribution（I2V 需要 VAE encoder 输出）
    def flatten_gaussian(obj):
        return (obj.parameters, obj.mean, obj.logvar, obj.deterministic,
                obj.std, obj.var), None
    
    def unflatten_gaussian(aux, children):
        obj = object.__new__(DiagonalGaussianDistribution)
        obj.parameters, obj.mean, obj.logvar, obj.deterministic, obj.std, obj.var = children
        return obj
    
    jax.tree_util.register_pytree_node(DiagonalGaussianDistribution, flatten_gaussian, unflatten_gaussian)
    print("  - DiagonalGaussianDistribution 已注册")


def shard_weight_dict(weight_dict, sharding_dict, mesh):
    """按模式匹配应用权重分片。"""
    result = {}
    for k, v in weight_dict.items():
        if isinstance(v, torch.Tensor):
            v = v.to("jax")
        matched = False
        for pattern, sharding in sharding_dict.items():
            if re.fullmatch(pattern, k) is not None:
                v.apply_jax_(jax.device_put, NamedSharding(mesh, P(*sharding)))
                matched = True
                break
        if not matched:
            v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
        result[k] = v
    return result


def override_op_definition(env, op, impl):
    """在 torchax 环境中覆盖算子定义。"""
    env._ops[op] = ops_registry.Operator(
        op, impl, is_jax_function=False, is_user_defined=True,
        needs_env=False, is_view_op=False,
    )


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
    """SDPA 封装：长序列用 Splash Attention，短序列用参考实现。"""
    if key.shape[2] > 20000:
        assert attn_mask is None and dropout_p == 0.0 and not is_causal
        assert not enable_gqa and scale is None
        
        jquery, jkey, jvalue = env.t2j_iso((query, key, value))
        if USE_K_SMOOTH:
            jkey = jkey - jnp.mean(jkey, axis=2, keepdims=True)
        res = tpu_splash_attention(jquery, jkey, jvalue, mesh, scale=scale)
        return env.j2t_iso(res)

    return sdpa_reference(query, key, value, attn_mask, dropout_p, is_causal,
                          scale, enable_gqa)


# ============================================================================
# Pipeline 设置
# ============================================================================

def load_pipeline(args):
    """加载 Pipeline（在启用 torchax 之前加载以避免 safetensors 问题）。"""
    print("\n=== 加载 Wan 2.2 I2V Pipeline ===")
    
    pipe = WanImageToVideoPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        vae=AutoencoderKLWan.from_pretrained(
            args.model_id, subfolder="vae", torch_dtype=torch.bfloat16
        ),
        transformer=WanTransformer3DModel.from_pretrained(
            args.model_id, subfolder="transformer", torch_dtype=torch.bfloat16
        ),
        transformer_2=WanTransformer3DModel.from_pretrained(
            args.model_id, subfolder="transformer_2", torch_dtype=torch.bfloat16
        ),
    )
    print("✓ 模型加载成功\n")
    return pipe


def setup_pipeline_for_jax(pipe, mesh, env):
    """设置 Wan I2V Pipeline 以在 JAX/TPU 上执行。"""
    print("=== 将模型移动到 TPU ===")
    
    print("- 注册自定义 JAX 算子...")
    override_op_definition(env, torch.nn.functional.conv2d,
                           functools.partial(torch_conv2d_jax, env=env))
    override_op_definition(env, torch.nn.functional.scaled_dot_product_attention,
                           functools.partial(scaled_dot_product_attention, env=env, mesh=mesh))
    
    print("- 将 Text Encoder 移动到 XLA 并分片...")
    move_module_to_xla(env, pipe.text_encoder)
    pipe.text_encoder = torchax.compile(pipe.text_encoder)
    pipe.text_encoder.params = shard_weight_dict(
        pipe.text_encoder.params, TEXT_ENCODER_SHARDINGS, mesh)
    pipe.text_encoder.buffers = shard_weight_dict(
        pipe.text_encoder.buffers, TEXT_ENCODER_SHARDINGS, mesh)
    
    transformer_options = torchax.CompileOptions(
        jax_jit_kwargs={"static_argnames": ("return_dict",)})
    
    print("- 将 Transformer 移动到 XLA 并分片...")
    move_module_to_xla(env, pipe.transformer)
    pipe.transformer = torchax.compile(pipe.transformer, transformer_options)
    pipe.transformer.params = shard_weight_dict(
        pipe.transformer.params, TRANSFORMER_SHARDINGS, mesh)
    pipe.transformer.buffers = shard_weight_dict(
        pipe.transformer.buffers, TRANSFORMER_SHARDINGS, mesh)
    
    print("- 将 Transformer 2 移动到 XLA 并分片...")
    move_module_to_xla(env, pipe.transformer_2)
    pipe.transformer_2 = torchax.compile(pipe.transformer_2, transformer_options)
    pipe.transformer_2.params = shard_weight_dict(
        pipe.transformer_2.params, TRANSFORMER_SHARDINGS, mesh)
    pipe.transformer_2.buffers = shard_weight_dict(
        pipe.transformer_2.buffers, TRANSFORMER_SHARDINGS, mesh)
    
    print("- 将 VAE 移动到 XLA...")
    move_module_to_xla(env, pipe.vae)
    pipe.vae.encoder = torchax.compile(pipe.vae.encoder)
    pipe.vae.decoder = torchax.compile(pipe.vae.decoder)
    
    print("=== Pipeline 设置完成 ===\n")
    return pipe


def run_generation(pipe, args, mesh):
    """运行 I2V 生成（支持可选的性能分析）。"""
    image = load_image(args.image)
    max_area = MAX_AREA_CONFIGS[args.size]
    aspect_ratio = image.height / image.width
    mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    image = image.resize((width, height))
    
    generator = torch.Generator().manual_seed(args.seed)
    
    pipe_kwargs = {
        'image': image,
        'prompt': args.prompt,
        'negative_prompt': args.negative_prompt,
        'height': height,
        'width': width,
        'num_frames': args.frames,
        'guidance_scale': args.guidance_scale,
        'num_inference_steps': args.num_steps,
        'generator': generator,
    }
    
    print(f"\n{'='*60}")
    print("生成配置")
    print(f"{'='*60}")
    print(f"  图像尺寸: {width}x{height}, 帧数: {args.frames}, FPS: {args.fps}")
    print(f"  推理步数: {args.num_steps}, 引导尺度: {args.guidance_scale}, 种子: {args.seed}")
    
    with mesh:
        # Warmup（只跑 2 步）
        print(f"\n{'='*60}\n预热运行（触发 JIT 编译，2 步）\n{'='*60}")
        warmup_kwargs = {**pipe_kwargs, 'num_inference_steps': 2}
        warmup_start = time.perf_counter()
        pipe(**warmup_kwargs)
        jax.effects_barrier()
        warmup_time = time.perf_counter() - warmup_start
        print(f"\n✓ 预热完成: {warmup_time:.2f}s")
        
        # Benchmark - 重新设置 seed 保证可复现性
        generator.manual_seed(args.seed)
        profiler_context = (jax.profiler.trace(args.profile_output_path, create_perfetto_link=False)
                            if args.profile else nullcontext())
        
        print(f"\n{'='*60}\n基准测试运行\n{'='*60}")
        with profiler_context:
            benchmark_start = time.perf_counter()
            output = pipe(**pipe_kwargs).frames[0]
            jax.effects_barrier()
            benchmark_time = time.perf_counter() - benchmark_start
        
        print(f"\n✓ 基准测试完成: {benchmark_time:.2f}s ({benchmark_time/args.num_steps:.2f}s/step)")
        
        file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        export_to_video(output, file_name, fps=args.fps)
        print(f"  视频保存至: {file_name}")
    
    print(f"\n{'='*60}\n性能统计\n{'='*60}")
    print(f"  基准时间: {benchmark_time:.2f}s ({benchmark_time/args.num_steps:.2f}s/step)")


# ============================================================================
# 主函数
# ============================================================================

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Wan 2.2 I2V TPU 视频生成")
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--size", type=str, default="720*1280", choices=list(SIZE_CONFIGS.keys()))
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS)
    parser.add_argument("--guidance_scale", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEG_PROMPT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dp", type=int, default=DEFAULT_DP, help="Data parallelism dimension")
    parser.add_argument("--profile", action="store_true", default=False, help="启用性能分析")
    parser.add_argument("--profile_output_path", type=str, default=PROFILE_OUT_PATH)
    return parser.parse_args()


def main():
    """Wan I2V 视频生成主入口。"""
    args = parse_args()
    
    print(f"\n{'='*60}")
    print("Wan 2.2 Image-to-Video 生成（TPU Splash Attention）")
    print(f"{'='*60}")
    
    jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    
    torch.set_default_dtype(torch.bfloat16)
    setup_pytree_registrations()
    
    pipe = load_pipeline(args)
    
    torchax.enable_globally()
    env = torchax.default_env()
    
    assert len(jax.devices()) % args.dp == 0
    tp_dim = len(jax.devices()) // args.dp
    mesh_devices = mesh_utils.create_device_mesh((args.dp, tp_dim), allow_split_physical_axes=True)
    mesh = Mesh(mesh_devices, ("dp", "tp"))
    
    print(f"\nMesh: dp={args.dp}, tp={tp_dim}, 总设备数={len(jax.devices())}")
    
    pipe = setup_pipeline_for_jax(pipe, mesh, env)
    run_generation(pipe, args, mesh)
    
    print(f"\n{'='*60}\n✓ 生成完成！\n{'='*60}")
    os._exit(0)


if __name__ == '__main__':
    main()
