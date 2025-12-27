#!/usr/bin/env python3
"""
CogVideoX Text-to-Video 生成脚本 (TPU Splash Attention)

使用 Torchax + JAX 在 TPU 上运行 CogVideoX 视频生成。
"""

import os
import sys
import warnings
import logging
from pathlib import Path

# ============================================================================
# 环境配置和 Warning 过滤（必须在其他 import 之前）
# ============================================================================

os.environ.setdefault('JAX_MEMORY_DEBUG', '0')
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
from torchax.ops import ops_registry

from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutputWithPastAndCrossAttentions
from diffusers import CogVideoXPipeline
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.utils import export_to_video

# 使用 TorchAx 版本的 VAE（优化版，使用 feat_cache 和 Welford GroupNorm）
from diffusers.models.autoencoders.autoencoder_kl_cogvideox_torchax import AutoencoderKLCogVideoX as TorchAxVAE

# 使用共享的 Splash Attention 模块
# 使用本地 splash_attention_utils
from splash_attention_utils import tpu_splash_attention, sdpa_reference


# ============================================================================
# 配置常量
# ============================================================================

MODEL_ID = "zai-org/CogVideoX1.5-5B"

# 视频生成参数 (720P)
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
TRANSFORMER_SHARDINGS = {
    r'.*\.to_q\.weight$': (None, 'tp'),
    r'.*\.to_k\.weight$': (None, 'tp'),
    r'.*\.to_v\.weight$': (None, 'tp'),
    r'.*\.to_out.*\.weight$': ('tp', None),
    r'.*\.ff\.net\.0\.weight$': (None, 'tp'),
    r'.*\.ff\.net\.2\.weight$': ('tp', None),
}

# Text Encoder (T5) 分片 - 使用所有设备联合分片
TEXT_ENCODER_SHARDINGS = {
    r'shared\.weight$': (('dp', 'tp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.SelfAttention\.q\.weight$': (('dp', 'tp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.SelfAttention\.k\.weight$': (('dp', 'tp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.SelfAttention\.v\.weight$': (('dp', 'tp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.SelfAttention\.o\.weight$': (None, ('dp', 'tp')),
    r'encoder\.block\.\d+\.layer\.\d+\.DenseReluDense\.wi_0\.weight$': (('dp', 'tp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.DenseReluDense\.wi_1\.weight$': (('dp', 'tp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.DenseReluDense\.wo\.weight$': (None, ('dp', 'tp')),
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
    
    for cls in [BaseModelOutputWithPooling, BaseModelOutputWithPastAndCrossAttentions, DecoderOutput]:
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
    print("\n=== 加载 CogVideoX Pipeline ===")
    
    pipe = CogVideoXPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
    )
    
    # 替换 VAE 为 TorchAx 优化版本（使用 feat_cache 和 Welford GroupNorm）
    print("- 加载 TorchAx VAE（优化版本）...")
    del pipe.vae
    
    torchax_vae = TorchAxVAE.from_pretrained(
        args.model_id,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    )
    pipe.vae = torchax_vae
    
    print("✓ 模型加载成功\n")
    return pipe


def setup_pipeline_for_jax(pipe, args, mesh, env):
    """设置 CogVideoX Pipeline 以在 JAX/TPU 上执行。"""
    print("=== 将模型移动到 TPU ===")
    
    # 注册自定义算子
    print("- 注册自定义 JAX 算子...")
    override_op_definition(env, torch.nn.functional.scaled_dot_product_attention,
                           functools.partial(scaled_dot_product_attention, env=env, mesh=mesh))
    
    # 移动 Scheduler 到 JAX
    print("- 将 Scheduler 移动到 JAX...")
    for k, v in pipe.scheduler.__dict__.items():
        if isinstance(v, torch.Tensor):
            setattr(pipe.scheduler, k, v.to('jax'))
    
    # Text Encoder 处理
    print("- 将 Text Encoder 移动到 XLA 并分片...")
    move_module_to_xla(env, pipe.text_encoder)
    pipe.text_encoder = torchax.compile(pipe.text_encoder)
    pipe.text_encoder.params = shard_weight_dict(
        pipe.text_encoder.params, TEXT_ENCODER_SHARDINGS, mesh)
    pipe.text_encoder.buffers = shard_weight_dict(
        pipe.text_encoder.buffers, TEXT_ENCODER_SHARDINGS, mesh)
    torchax.interop.call_jax(jax.block_until_ready, pipe.text_encoder.params)
    
    # Transformer 处理
    print("- 将 Transformer 移动到 XLA 并分片...")
    move_module_to_xla(env, pipe.transformer)
    pipe.transformer = torchax.compile(pipe.transformer, torchax.CompileOptions(
        jax_jit_kwargs={'static_argnames': ('return_dict',)}))
    pipe.transformer.params = shard_weight_dict(
        pipe.transformer.params, TRANSFORMER_SHARDINGS, mesh)
    pipe.transformer.buffers = shard_weight_dict(
        pipe.transformer.buffers, TRANSFORMER_SHARDINGS, mesh)
    torchax.interop.call_jax(jax.block_until_ready, pipe.transformer.params)
    
    # VAE 处理
    print("- 将 VAE 移动到 XLA...")
    move_module_to_xla(env, pipe.vae)
    pipe.vae.decoder = torchax.compile(pipe.vae.decoder)
    
    print("=== Pipeline 设置完成 ===\n")
    return pipe


def run_generation(pipe, args, mesh):
    """运行视频生成（支持可选的性能分析）。"""
    prompt = ("A panda, dressed in a small, red jacket and a tiny hat, sits on a "
              "wooden stool in a serene bamboo forest. The panda's fluffy paws strum "
              "a miniature acoustic guitar, producing soft, melodic tunes. Nearby, "
              "a few other pandas gather, watching curiously and some clapping in rhythm. "
              "Sunlight filters through the tall bamboo, casting a gentle glow on the scene.")
    
    generator = torch.Generator()
    generator.manual_seed(42)
    
    pipe_kwargs = {
        'prompt': prompt,
        'height': args.height,
        'width': args.width,
        'num_inference_steps': args.num_inference_steps,
        'num_frames': args.frames,
        'guidance_scale': 6.0,
        'generator': generator,
    }
    
    print(f"\n{'='*60}")
    print("生成配置")
    print(f"{'='*60}")
    print(f"  分辨率: {args.width}x{args.height}, 帧数: {args.frames}, FPS: {args.fps}")
    print(f"  推理步数: {args.num_inference_steps}, 引导尺度: 6.0, 种子: 42")
    
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
    parser = argparse.ArgumentParser(description="CogVideoX TPU 视频生成")
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--num_inference_steps", type=int, default=NUM_STEPS)
    parser.add_argument("--use_dp", action="store_true", default=USE_DP)
    parser.add_argument("--use_tp", action="store_true", default=USE_TP)
    parser.add_argument("--profile", action="store_true", default=False,
                        help="启用性能分析")
    return parser.parse_args()


def main():
    """CogVideoX 视频生成主入口。"""
    args = parse_args()
    
    print(f"\n{'='*60}")
    print("CogVideoX Text-to-Video 生成（TPU Splash Attention）")
    print(f"{'='*60}")
    
    # 配置 JAX
    jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax_cache"))
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
