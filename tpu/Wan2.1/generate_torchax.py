#!/usr/bin/env python3
"""
Wan 2.1 Text-to-Video 生成脚本 (TPU Splash Attention)

使用 Torchax + JAX 在 TPU 上运行 Wan 2.1 视频生成。
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

import jax
import jax.numpy as jnp
import torch
from jax.tree_util import register_pytree_node
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
from jax.experimental import mesh_utils

import torchax
from torchax.ops import ops_registry, jaten

from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.pipelines.wan.pipeline_wan_torchax import WanPipeline
from diffusers.models.autoencoders.autoencoder_kl_wan_torchax import AutoencoderKLWan
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'kernels'))
from splash_attention_utils import tpu_splash_attention, sdpa_reference


# ============================================================================
# 配置常量
# ============================================================================

MODEL_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"

# 视频生成参数 (720P)
FLOW_SHIFT = 5.0  # 720P 用 5.0，480P 用 3.0
WIDTH = 1280
HEIGHT = 720
FRAMES = 81
FPS = 16
NUM_STEPS = 50

# Mesh 分片配置
USE_DP = True
USE_TP = True

# K 平滑以提高数值稳定性
USE_K_SMOOTH = True

# Profiler 输出路径
PROFILE_OUT_PATH = "/dev/shm/jax_trace"


# ============================================================================
# 分片策略
# ============================================================================

# Transformer 分片（2D mesh: dp, tp）
# 规则：输出投影用 ('tp', None)，输入投影用 (None, 'tp')，1D bias 用 ('tp',)
TRANSFORMER_SHARDINGS = {
    # Condition Embedder
    r'condition_embedder.*.linear_1.weight': ('tp', None),
    r'condition_embedder.*.linear_1.bias': ('tp',),
    r'condition_embedder.*.linear_2.weight': (None, 'tp'),
    # Attention (attn1 + attn2)
    r'blocks.*.attn[12].to_[qkv].weight': ('tp', None),
    r'blocks.*.attn[12].to_[qkv].bias': ('tp',),
    r'blocks.*.attn[12].to_out.*.weight': (None, 'tp'),
    # FFN
    r'blocks.*.ffn.net.*.proj.weight': ('tp', None),
    r'blocks.*.ffn.net.*.proj.bias': ('tp',),
    r'blocks.*.ffn.net.*.weight': (None, 'tp'),
}

# Text Encoder (T5) 分片 - 使用所有设备联合分片
TEXT_ENCODER_SHARDINGS = {
    r'shared.weight': (('dp', 'tp'), None),
    r'encoder.block.*.layer.*.SelfAttention.[qkv].weight': (('dp', 'tp'), None),
    r'encoder.block.*.layer.*.SelfAttention.o.weight': (None, ('dp', 'tp')),
    r'encoder.block.*.layer.*.DenseReluDense.wi_[01].weight': (('dp', 'tp'), None),
    r'encoder.block.*.layer.*.DenseReluDense.wo.weight': (None, ('dp', 'tp')),
}


# ============================================================================
# 辅助函数
# ============================================================================

def setup_pytree_registrations():
    """注册必要的 PyTree 节点以支持 JAX 转换。"""
    print("注册 PyTree 节点...")
    
    def flatten(obj):
        return obj.to_tuple(), type(obj)

    def unflatten(aux, children):
        return aux(*children)
    
    for cls in [BaseModelOutputWithPastAndCrossAttentions, DecoderOutput, AutoencoderKLOutput]:
        register_pytree_node(cls, flatten, unflatten)
        print(f"  - {cls.__name__} 已注册")


def shard_weight_dict(weight_dict, sharding_dict, mesh):
    """按模式匹配应用权重分片。"""
    result = {}
    for k, v in weight_dict.items():
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
    print("\n=== 加载 Wan 2.1 T2V Pipeline ===")
    
    scheduler = UniPCMultistepScheduler(
        prediction_type='flow_prediction',
        use_flow_sigmas=True,
        num_train_timesteps=1000,
        flow_shift=args.flow_shift
    )
    
    pipe = WanPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        vae=AutoencoderKLWan.from_pretrained(
            args.model_id, subfolder="vae", torch_dtype=torch.bfloat16
        ),
        scheduler=scheduler,
    )
    
    print("✓ 模型加载成功\n")
    return pipe


def setup_pipeline_for_jax(pipe, args, mesh, env):
    """设置 Wan Pipeline 以在 JAX/TPU 上执行。"""
    print("=== 将模型移动到 TPU ===")
    
    # 注册自定义算子
    print("- 注册自定义 JAX 算子...")
    override_op_definition(env, torch.nn.functional.conv2d,
                           functools.partial(torch_conv2d_jax, env=env))
    override_op_definition(env, torch.nn.functional.scaled_dot_product_attention,
                           functools.partial(scaled_dot_product_attention, env=env, mesh=mesh))
    
    # Text Encoder 处理
    if args.t5_cpu:
        print("- 保持 Text Encoder 在 CPU...")
        pipe.text_encoder.to("cpu")
    else:
        print("- 将 Text Encoder 移动到 XLA 并分片...")
        move_module_to_xla(env, pipe.text_encoder)
        pipe.text_encoder = torchax.compile(pipe.text_encoder)
        pipe.text_encoder.params = shard_weight_dict(
            pipe.text_encoder.params, TEXT_ENCODER_SHARDINGS, mesh)
        pipe.text_encoder.buffers = shard_weight_dict(
            pipe.text_encoder.buffers, TEXT_ENCODER_SHARDINGS, mesh)
    
    # Transformer 处理
    print("- 将 Transformer 移动到 XLA 并分片...")
    move_module_to_xla(env, pipe.transformer)
    
    if hasattr(pipe.transformer.rope, 'freqs'):
        pipe.transformer.rope.freqs = pipe.transformer.rope.freqs.to('jax')
    else:
        pipe.transformer.rope.freqs_cos = pipe.transformer.rope.freqs_cos.to('jax')
        pipe.transformer.rope.freqs_sin = pipe.transformer.rope.freqs_sin.to('jax')
    
    pipe.transformer = torchax.compile(pipe.transformer, torchax.CompileOptions(
        jax_jit_kwargs={'static_argnames': ('return_dict',)}))
    pipe.transformer.params = shard_weight_dict(
        pipe.transformer.params, TRANSFORMER_SHARDINGS, mesh)
    pipe.transformer.buffers = shard_weight_dict(
        pipe.transformer.buffers, TRANSFORMER_SHARDINGS, mesh)
    
    # VAE 处理
    print("- 将 VAE 移动到 XLA...")
    move_module_to_xla(env, pipe.vae)
    pipe.vae.encoder = torchax.compile(pipe.vae.encoder)
    pipe.vae.decoder = torchax.compile(pipe.vae.decoder)
    
    print("=== Pipeline 设置完成 ===\n")
    return pipe


def run_generation(pipe, args, mesh):
    """运行视频生成（支持可选的性能分析）。"""
    prompt = ("A cat and a dog baking a cake together in a kitchen. "
              "The cat is carefully measuring flour, while the dog is "
              "stirring the batter with a wooden spoon. The kitchen is cozy, "
              "with sunlight streaming through the window.")
    
    negative_prompt = ("Bright tones, overexposed, static, blurred details, "
                       "subtitles, style, works, paintings, images, static, "
                       "overall gray, worst quality, low quality, JPEG compression residue, "
                       "ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, "
                       "deformed, disfigured, misshapen limbs, fused fingers, still picture, "
                       "messy background, three legs, many people in the background, walking backwards")
    
    generator = torch.Generator()
    generator.manual_seed(42)
    
    pipe_kwargs = {
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'height': args.height,
        'width': args.width,
        'num_inference_steps': args.num_inference_steps,
        'num_frames': args.frames,
        'guidance_scale': 5.0,
        'generator': generator,
        'use_dp': args.use_dp,
    }
    
    print(f"\n{'='*60}")
    print("生成配置")
    print(f"{'='*60}")
    print(f"  分辨率: {args.width}x{args.height}, 帧数: {args.frames}, FPS: {args.fps}")
    print(f"  推理步数: {args.num_inference_steps}, 引导尺度: 5.0, 种子: 42")
    
    with mesh:
        # Warmup（只跑 2 步，不包含在 profiler 中）
        print(f"\n{'='*60}\n预热运行（触发 JIT 编译，2 步）\n{'='*60}")
        warmup_kwargs = {**pipe_kwargs, 'num_inference_steps': 2}
        warmup_start = time.perf_counter()
        pipe(**warmup_kwargs)
        jax.effects_barrier()
        warmup_time = time.perf_counter() - warmup_start
        print(f"\n✓ 预热完成: {warmup_time:.2f}s")
        
        # Benchmark（可选 profiler）- 重新设置 seed 以保证可复现性
        generator.manual_seed(42)
        profiler_context = (jax.profiler.trace(PROFILE_OUT_PATH, create_perfetto_link=False)
                            if args.profile else nullcontext())
        
        print(f"\n{'='*60}\n基准测试运行\n{'='*60}")
        with profiler_context:
            benchmark_start = time.perf_counter()
            output = pipe(**pipe_kwargs).frames[0]
            jax.effects_barrier()
            benchmark_time = time.perf_counter() - benchmark_start
        
        print(f"\n✓ 基准测试完成: {benchmark_time:.2f}s ({benchmark_time/args.num_inference_steps:.2f}s/step)")
        
        # 保存视频
        file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        export_to_video(output, file_name, fps=args.fps)
        print(f"  视频保存至: {file_name}")
    
    print(f"\n{'='*60}\n性能统计\n{'='*60}")
    print(f"  基准时间: {benchmark_time:.2f}s ({benchmark_time/args.num_inference_steps:.2f}s/step)")


# ============================================================================
# 主函数
# ============================================================================

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Wan 2.1 TPU 视频生成")
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--flow_shift", type=float, default=FLOW_SHIFT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--num_inference_steps", type=int, default=NUM_STEPS)
    parser.add_argument("--use_dp", action="store_true", default=USE_DP)
    parser.add_argument("--use_tp", action="store_true", default=USE_TP)
    parser.add_argument("--t5_cpu", action="store_true", default=False,
                        help="将 T5 text encoder 放在 CPU 上")
    parser.add_argument("--profile", action="store_true", default=False,
                        help="启用性能分析")
    return parser.parse_args()


def main():
    """Wan 视频生成主入口。"""
    args = parse_args()
    
    print(f"\n{'='*60}")
    print("Wan 2.1 Text-to-Video 生成（TPU Splash Attention）")
    print(f"{'='*60}")
    
    # 配置 JAX
    jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    
    torch.set_default_dtype(torch.bfloat16)
    setup_pytree_registrations()
    
    # 加载 pipeline（在启用 torchax 之前）
    pipe = load_pipeline(args)
    
    # 启用 torchax
    torchax.enable_globally()
    env = torchax.default_env()
    
    # 创建 mesh
    assert len(jax.devices()) % 2 == 0
    dp_dim = 2 if args.use_dp else 1
    tp_dim = len(jax.devices()) // dp_dim
    
    mesh_devices = mesh_utils.create_device_mesh((dp_dim, tp_dim), allow_split_physical_axes=True)
    mesh = Mesh(mesh_devices, ("dp", "tp"))
    
    print(f"\nMesh: dp={dp_dim}, tp={tp_dim}, 总设备数={len(jax.devices())}")
    
    # 设置并运行
    pipe = setup_pipeline_for_jax(pipe, args, mesh, env)
    run_generation(pipe, args, mesh)
    
    print(f"\n{'='*60}\n✓ 生成完成！\n{'='*60}")
    os._exit(0)


if __name__ == '__main__':
    main()
