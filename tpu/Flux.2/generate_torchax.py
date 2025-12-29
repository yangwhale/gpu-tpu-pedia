#!/usr/bin/env python3
"""
Flux.2 Text-to-Image 生成脚本 (TPU Splash Attention)

使用 Torchax + JAX 在 TPU 上运行 Flux.2 图像生成。
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
from diffusers.pipelines.flux2.pipeline_flux2_torchax import Flux2Pipeline
from diffusers.models.autoencoders.autoencoder_kl_flux2_torchax import AutoencoderKLFlux2
from diffusers.models.transformers.transformer_flux2_torchax import Flux2Transformer2DModel
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from PIL import Image

from huggingface_hub import get_token
import requests
import io

import sys
from pathlib import Path
# 使用本地 splash_attention_utils
from splash_attention_utils import tpu_splash_attention, sdpa_reference


# ============================================================================
# 配置常量
# ============================================================================

MODEL_ID = "black-forest-labs/FLUX.2-dev"

# 图像生成参数
WIDTH = 1024
HEIGHT = 1024
NUM_STEPS = 50
GUIDANCE_SCALE = 4.0

# Mesh 分片配置（只用 TP，不用 DP）
USE_TP = True

# K 平滑以提高数值稳定性
USE_K_SMOOTH = True

# Profiler 输出路径
PROFILE_OUT_PATH = "/dev/shm/jax_trace"


# ============================================================================
# 分片策略 (只用 tp，不用 dp)
# ============================================================================

# Flux2Transformer 分片（1D mesh: tp）
# 规则：输出投影用 ('tp', None)，输入投影用 (None, 'tp')
TRANSFORMER_SHARDINGS = {
    # Double-stream Transformer Blocks (transformer_blocks.*)
    # Attention: Flux2Attention
    r'transformer_blocks.*.attn.to_q.weight': ('tp', None),
    r'transformer_blocks.*.attn.to_k.weight': ('tp', None),
    r'transformer_blocks.*.attn.to_v.weight': ('tp', None),
    r'transformer_blocks.*.attn.to_out.0.weight': (None, 'tp'),
    r'transformer_blocks.*.attn.add_q_proj.weight': ('tp', None),
    r'transformer_blocks.*.attn.add_k_proj.weight': ('tp', None),
    r'transformer_blocks.*.attn.add_v_proj.weight': ('tp', None),
    r'transformer_blocks.*.attn.to_add_out.weight': (None, 'tp'),
    # FeedForward: Flux2FeedForward
    r'transformer_blocks.*.ff.linear_in.weight': ('tp', None),
    r'transformer_blocks.*.ff.linear_out.weight': (None, 'tp'),
    r'transformer_blocks.*.ff_context.linear_in.weight': ('tp', None),
    r'transformer_blocks.*.ff_context.linear_out.weight': (None, 'tp'),
    
    # Single-stream Transformer Blocks (single_transformer_blocks.*)
    # Parallel Self-Attention: Flux2ParallelSelfAttention
    r'single_transformer_blocks.*.attn.to_qkv_mlp_proj.weight': ('tp', None),
    r'single_transformer_blocks.*.attn.to_out.weight': (None, 'tp'),
    
    # Input/Output Projections
    r'x_embedder.weight': ('tp', None),
    r'context_embedder.weight': ('tp', None),
    r'proj_out.weight': (None, 'tp'),
    
    # Modulation
    r'double_stream_modulation_img.linear.weight': ('tp', None),
    r'double_stream_modulation_txt.linear.weight': ('tp', None),
    r'single_stream_modulation.linear.weight': ('tp', None),
    
    # Time + Guidance Embedding
    r'time_guidance_embed.timestep_embedder.linear_1.weight': ('tp', None),
    r'time_guidance_embed.timestep_embedder.linear_2.weight': (None, 'tp'),
    r'time_guidance_embed.guidance_embedder.linear_1.weight': ('tp', None),
    r'time_guidance_embed.guidance_embedder.linear_2.weight': (None, 'tp'),
}

# Text Encoder (Mistral3) 分片 - 使用 tp 维度
TEXT_ENCODER_SHARDINGS = {
    # Embedding
    r'model.embed_tokens.weight': ('tp', None),
    # Attention projections
    r'model.layers.*.self_attn.q_proj.weight': ('tp', None),
    r'model.layers.*.self_attn.k_proj.weight': ('tp', None),
    r'model.layers.*.self_attn.v_proj.weight': ('tp', None),
    r'model.layers.*.self_attn.o_proj.weight': (None, 'tp'),
    # MLP
    r'model.layers.*.mlp.gate_proj.weight': ('tp', None),
    r'model.layers.*.mlp.up_proj.weight': ('tp', None),
    r'model.layers.*.mlp.down_proj.weight': (None, 'tp'),
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

def remote_text_encoder(prompts, max_retries=3, retry_delay=5):
    """使用 HuggingFace 远程 text encoder 服务获取 prompt embeddings。"""
    print(f"=== 调用远程 Text Encoder ===")
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://remote-text-encoder-flux-2.huggingface.co/predict",
                json={"prompt": prompts},
                headers={
                    "Authorization": f"Bearer {get_token()}",
                    "Content-Type": "application/json"
                },
                timeout=60
            )
            if response.status_code == 200:
                prompt_embeds = torch.load(io.BytesIO(response.content))
                print(f"✓ Prompt embeddings shape: {prompt_embeds.shape}")
                return prompt_embeds
            elif response.status_code == 503:
                print(f"  服务暂不可用 (503)，重试 {attempt + 1}/{max_retries}...")
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"Remote text encoder failed: {response.status_code=}")
        except requests.exceptions.RequestException as e:
            print(f"  请求失败: {e}，重试 {attempt + 1}/{max_retries}...")
            time.sleep(retry_delay)
    
    raise RuntimeError(f"Remote text encoder failed after {max_retries} retries")


def encode_prompt_on_cpu(args, prompt):
    """在 CPU 上使用 Mistral3 编码 prompt（在启用 torchax 之前）。"""
    from transformers import PixtralProcessor, Mistral3ForConditionalGeneration
    
    print("\n=== 在 CPU 上加载 Mistral3 Text Encoder ===")
    
    # 直接从 text_encoder 目录加载
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        args.model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    text_encoder.eval()
    
    # 使用 PixtralProcessor 从 tokenizer 目录加载
    tokenizer = PixtralProcessor.from_pretrained(
        args.model_id,
        subfolder="tokenizer"
    )
    
    print("✓ Text Encoder 加载成功")
    print(f"- 编码 prompt: {prompt[:50]}...")
    
    # 格式化输入 - 使用 Flux.2 pipeline 中的格式
    SYSTEM_MESSAGE = "You are an image generation assistant. Convert user text to detailed image descriptions."
    messages = [[
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]]
    
    # Tokenize
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    
    # 前向传播
    with torch.no_grad():
        output = text_encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
        )
    
    # 提取隐藏状态
    hidden_states_layers = (10, 20, 30)
    out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
    out = out.to(dtype=torch.bfloat16)
    
    batch_size, num_channels, seq_len, hidden_dim = out.shape
    prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)
    
    print(f"✓ Prompt embeddings shape: {prompt_embeds.shape}")
    
    # 释放内存
    del text_encoder, tokenizer
    import gc
    gc.collect()
    
    return prompt_embeds


def load_pipeline(args):
    """加载 Pipeline（在启用 torchax 之前加载以避免 safetensors 问题）。"""
    print("\n=== 加载 Flux.2 Pipeline ===")
    
    scheduler = FlowMatchEulerDiscreteScheduler()
    
    # 显式加载 torchax 版本的模型
    vae = AutoencoderKLFlux2.from_pretrained(
        args.model_id, subfolder="vae", torch_dtype=torch.bfloat16
    )
    transformer = Flux2Transformer2DModel.from_pretrained(
        args.model_id, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    
    # 使用 text_encoder=None，使用外部编码的 prompt_embeds
    pipe = Flux2Pipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        text_encoder=None,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
    )
    
    print("✓ 模型加载成功（不加载 text encoder）\n")
    return pipe


def setup_pipeline_for_jax(pipe, args, mesh, env):
    """设置 Flux.2 Pipeline 以在 JAX/TPU 上执行。"""
    print("=== 将模型移动到 TPU ===")
    
    # 注册自定义算子
    print("- 注册自定义 JAX 算子...")
    override_op_definition(env, torch.nn.functional.conv2d,
                           functools.partial(torch_conv2d_jax, env=env))
    override_op_definition(env, torch.nn.functional.scaled_dot_product_attention,
                           functools.partial(scaled_dot_product_attention, env=env, mesh=mesh))
    
    # Text Encoder 使用远程服务，无需处理
    print("- 使用远程 Text Encoder，跳过本地处理...")
    
    # Transformer 处理
    print("- 将 Transformer 移动到 XLA 并分片...")
    move_module_to_xla(env, pipe.transformer)
    
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


# 默认 prompt
DEFAULT_PROMPT = ("Realistic macro photograph of a hermit crab using a soda can as its shell, "
                  "partially emerging from the can, captured with sharp detail and natural colors, "
                  "on a sunlit beach with soft shadows and a shallow depth of field, "
                  "with blurred ocean waves in the background. "
                  "The can has the text `BFL Diffusers` on it and it has a color gradient "
                  "that start with #FF5733 at the top and transitions to #33FF57 at the bottom.")


def run_generation(pipe, args, mesh, prompt_embeds, env):
    """运行图像生成（支持可选的性能分析）。"""
    generator = torch.Generator()
    generator.manual_seed(42)
    
    # 将 prompt_embeds 转换为 XLA tensor
    prompt_embeds_xla = env.to_xla(prompt_embeds)
    
    pipe_kwargs = {
        'prompt_embeds': prompt_embeds_xla,
        'height': args.height,
        'width': args.width,
        'num_inference_steps': args.num_inference_steps,
        'guidance_scale': args.guidance_scale,
        'generator': generator,
    }
    
    print(f"\n{'='*60}")
    print("生成配置")
    print(f"{'='*60}")
    print(f"  分辨率: {args.width}x{args.height}")
    print(f"  推理步数: {args.num_inference_steps}, 引导尺度: {args.guidance_scale}, 种子: 42")
    
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
            output = pipe(**pipe_kwargs).images[0]
            jax.effects_barrier()
            benchmark_time = time.perf_counter() - benchmark_start
        
        print(f"\n✓ 基准测试完成: {benchmark_time:.2f}s ({benchmark_time/args.num_inference_steps:.2f}s/step)")
        
        # 保存图像
        file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        output.save(file_name)
        print(f"  图像保存至: {file_name}")
    
    print(f"\n{'='*60}\n性能统计\n{'='*60}")
    print(f"  基准时间: {benchmark_time:.2f}s ({benchmark_time/args.num_inference_steps:.2f}s/step)")


# ============================================================================
# 主函数
# ============================================================================

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Flux.2 TPU 图像生成")
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--num_inference_steps", type=int, default=NUM_STEPS)
    parser.add_argument("--guidance_scale", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--use_tp", action="store_true", default=USE_TP)
    parser.add_argument("--mistral_cpu", action="store_true", default=False,
                        help="将 Mistral3 text encoder 放在 CPU 上")
    parser.add_argument("--profile", action="store_true", default=False,
                        help="启用性能分析")
    return parser.parse_args()


def main():
    """Flux.2 图像生成主入口。"""
    args = parse_args()
    
    print(f"\n{'='*60}")
    print("Flux.2 Text-to-Image 生成（TPU Splash Attention）")
    print(f"{'='*60}")
    
    # 配置 JAX
    jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax_cache"))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    
    torch.set_default_dtype(torch.bfloat16)
    setup_pytree_registrations()
    
    # 获取 prompt embeddings（在启用 torchax 之前）
    if args.mistral_cpu:
        # 在 CPU 上使用 Mistral3 编码
        prompt_embeds = encode_prompt_on_cpu(args, DEFAULT_PROMPT)
    else:
        # 使用远程 text encoder
        prompt_embeds = remote_text_encoder(DEFAULT_PROMPT)
    
    # 加载 pipeline（在启用 torchax 之前）
    pipe = load_pipeline(args)
    
    # 启用 torchax
    torchax.enable_globally()
    env = torchax.default_env()
    
    # 创建 mesh（只用 tp，不用 dp）
    tp_dim = len(jax.devices())
    
    mesh_devices = mesh_utils.create_device_mesh((tp_dim,), allow_split_physical_axes=True)
    mesh = Mesh(mesh_devices, ("tp",))
    
    print(f"\nMesh: tp={tp_dim}, 总设备数={len(jax.devices())}")
    
    # 设置并运行
    pipe = setup_pipeline_for_jax(pipe, args, mesh, env)
    run_generation(pipe, args, mesh, prompt_embeds=prompt_embeds, env=env)
    
    print(f"\n{'='*60}\n✓ 生成完成！\n{'='*60}")
    os._exit(0)


if __name__ == '__main__':
    main()
