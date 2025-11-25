# 导入必要的库
import os
os.environ.setdefault('JAX_MEMORY_DEBUG', '0')  # 默认关闭内存调试

import sys
import time
import math
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from tqdm import tqdm
import warnings
import logging
from contextlib import nullcontext

# 添加 diffusers-tpu 路径
sys.path.insert(0, '/home/chrisya/diffusers-tpu/src')

from diffusers.models.transformers.cogvideox_transformer_3d_flax import (
    FlaxCogVideoXTransformer3DModel,
    FlaxCogVideoXTransformer3DConfig,
)

# --- 全局配置 ---
MODEL_NAME = "zai-org/CogVideoX1.5-5B"

# --- 性能测试工具函数 ---

def record_time_tpu(call_method):
    """
    记录一个函数调用的执行时间（TPU版本）
    使用jax.block_until_ready确保计算完成
    
    参数:
    call_method (function): 需要被测量时间的函数。
    
    返回:
    tuple: 一个元组，包含被调用函数的输出和以毫秒为单位的执行时间。
    """
    start = time.time()
    output = call_method()
    
    # 确保JAX计算完成
    jax.block_until_ready(output)
    
    end = time.time()
    return output, (end - start) * 1000  # s -> ms


def record_peak_memory_tpu(call_method):
    """
    记录一个函数调用期间的 TPU 峰值显存使用量和执行时间
    
    注意: TPU的内存监控与GPU不同，这里使用JAX的内存统计
    
    参数:
    call_method (function): 需要被测量的函数。
    
    返回:
    tuple: 一个元组，包含被调用函数的输出、以 MB 为单位的峰值显存和以毫秒为单位的执行时间。
    """
    # 调用 record_time_tpu 来执行函数并获取其输出和执行时间
    output, time_cost = record_time_tpu(call_method)
    
    # TPU 内存统计（占位值）
    peak_memory_mb = 0.0
    
    return output, peak_memory_mb, time_cost


# --- 结果打印 ---

def print_results(results, frames):
    """
    打印测试结果的统计信息。
    
    参数:
    results (list of dict): 测试结果的列表。
    frames (int): 测试的帧数。
    """
    if not results:
        print("没有测试结果")
        return
    
    times = [r['time'] for r in results]
    
    print(f"\n=== DiT Pure Flax 测试结果 (帧数: {frames}) ===")
    print(f"运行次数: {len(results)}")
    print(f"\n执行时间 (ms):")
    print(f"  平均值: {sum(times)/len(times):.2f}")
    print(f"  最小值: {min(times):.2f}")
    print(f"  最大值: {max(times):.2f}")
    
    if len(times) > 1:
        print(f"\n首次运行（含JIT编译）: {times[0]:.2f} ms")
        avg_time = sum(times[1:]) / len(times[1:])
        print(f"后续运行平均时间: {avg_time:.2f} ms")
        if times[0] > 0:
            print(f"加速比: {times[0] / avg_time:.2f}x")


# --- DiT 模型性能测试核心函数 ---

def dit_test(transformer, frames=64, num_runs=1, warmup_runs=1, profiler_context=None):
    """
    测试 DiT (Diffusion Transformer) 模型在TPU上的性能（纯Flax版本）。
    先进行预热运行，然后对指定帧数重复运行多次以获取稳定的性能数据。
    
    参数:
    transformer: FlaxCogVideoXTransformer3DModel 实例
    frames (int): 测试的视频帧数，默认64帧（必须能被4整除）。
    num_runs (int): 重复运行的次数，默认10次。
    warmup_runs (int): 预热运行次数，默认3次（不计入统计）。
    profiler_context: Profiler上下文管理器，可选。
    """
    # DiT 模型期望的输入维度（使用更小的尺寸）
    batch = 1
    channel = 16
    height = 60  # 对应 sample_height
    width = 90   # 对应 sample_width
    
    # Transformer 的输入帧数，考虑 temporal_compression_ratio
    latent_frames = frames // 4
    
    # 定义运行单次测试的函数
    def run_single_test():
        # --- 准备模型输入 (JAX arrays, channel-last format) ---
        # 1. 创建主要的输入张量 (hidden_states) - (B, T, H, W, C)
        key = jax.random.key(42)
        input_tensor = jax.random.normal(key, (batch, latent_frames, height, width, channel), dtype=jnp.bfloat16)
        
        # 2. 创建文本嵌入 (encoder_hidden_states) - (B, text_seq_len, text_embed_dim)
        text_seq_len = 226  # max_text_seq_length from config
        text_embed_dim = 4096
        key, subkey = jax.random.split(key)
        input_embd = jax.random.normal(subkey, (batch, text_seq_len, text_embed_dim), dtype=jnp.bfloat16)
        
        # 3. 创建时间步 (timestep)
        timestep = jnp.array([999], dtype=jnp.float32)
        
        # 4. 创建旋转位置编码 (image_rotary_emb) - optional
        # 暂时设为 None，因为默认配置不使用 rotary embeddings
        image_rotary_emb = None
        
        # 定义调用 Transformer 模型的函数
        def dit_call():
            return transformer(
                hidden_states=input_tensor,
                encoder_hidden_states=input_embd,
                timestep=timestep,
                image_rotary_emb=image_rotary_emb,
                deterministic=True,
                return_dict=False,
            )
        
        # 记录执行时间
        output, peak_memory_mb, time_cost = record_peak_memory_tpu(dit_call)
        del output  # 释放内存
        
        return peak_memory_mb, time_cost
    
    # 预热运行
    print(f"开始预热运行 (预热次数: {warmup_runs})")
    for run in tqdm(range(warmup_runs), desc="Warmup DiT on TPU"):
        try:
            run_single_test()
        except Exception as e:
            print(f"预热第 {run + 1} 次运行出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    # 正式测试运行
    results = []
    print(f"\n开始正式测试 DiT Pure Flax 性能 (帧数: {frames}, 运行次数: {num_runs})")
    
    context = profiler_context if profiler_context else nullcontext()
    
    with context:
        for run in tqdm(range(num_runs), desc="Testing DiT on TPU"):
            try:
                peak_memory_mb, time_cost = run_single_test()
                
                results.append({
                    'run': run + 1,
                    'peak_memory_mb': peak_memory_mb,
                    'time': time_cost
                })

            except Exception as e:
                print(f"第 {run + 1} 次运行出错: {str(e)}")
                import traceback
                traceback.print_exc()
                break

    return results


# --- 主测试流程 ---

def load_transformer(model_name=MODEL_NAME, dtype=jnp.bfloat16, use_pretrained=True):
    """
    加载或创建纯 Flax Transformer模型
    
    Args:
        model_name: 预训练模型名称（仅当 use_pretrained=True 时使用）
        dtype: 数据类型
        use_pretrained: 是否加载预训练权重
        
    Returns:
        transformer: FlaxCogVideoXTransformer3DModel 实例
    """
    if use_pretrained:
        print(f"正在加载预训练模型: {model_name}")
        print("使用 FlaxCogVideoXTransformer3DModel.from_pretrained()")
        
        transformer = FlaxCogVideoXTransformer3DModel.from_pretrained(
            model_name,
            subfolder="transformer",
            dtype=dtype,
        )
    else:
        print("创建随机初始化的 Transformer 模型（用于测试结构）")
        
        # 创建配置（使用更小的配置以避免OOM）
        # 基于 CogVideoX-2B 的配置
        config = FlaxCogVideoXTransformer3DConfig(
            num_attention_heads=30,  # 减小到30个头
            attention_head_dim=64,
            in_channels=16,
            out_channels=16,
            time_embed_dim=512,
            text_embed_dim=4096,
            num_layers=30,  # 减小到30层
            sample_width=90,  # 减小分辨率
            sample_height=60,
            patch_size=2,
            temporal_compression_ratio=4,
            max_text_seq_length=226,
        )
        
        # 创建模型
        key = jax.random.key(0)
        rngs = nnx.Rngs(key)
        transformer = FlaxCogVideoXTransformer3DModel(
            config=config,
            rngs=rngs,
            dtype=dtype,
        )
    
    print("模型加载完成")
    return transformer


def compile_transformer(transformer):
    """
    使用 JAX JIT 编译 Transformer
    
    Args:
        transformer: FlaxCogVideoXTransformer3DModel 实例
        
    Returns:
        compiled_fn: JIT 编译后的函数
    """
    print("\n编译 Transformer...")
    
    # 创建一个包装函数用于 JIT
    def forward_fn(hidden_states, encoder_hidden_states, timestep):
        return transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            image_rotary_emb=None,
            deterministic=True,
            return_dict=False,
        )
    
    # JIT 编译
    compiled_fn = jax.jit(forward_fn)
    
    print("编译完成")
    return compiled_fn


def dit(frames=64, num_runs=10):
    """
    执行 DiT 模型在TPU上的性能测试（纯Flax版本）。
    
    参数:
    frames (int): 测试的视频帧数，默认64帧（必须能被4整除）。
    num_runs (int): 重复运行的次数，默认10次。
    """
    print("--- 开始 DiT Pure Flax TPU 性能测试 ---")
    
    # 设置JAX配置
    warnings.filterwarnings('ignore')
    logging.getLogger().setLevel(logging.ERROR)
    
    # 设置随机种子
    import random
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    
    # 加载 Transformer 模型（使用随机初始化权重进行测试）
    # 设置 use_pretrained=True 来加载预训练权重
    transformer = load_transformer(dtype=jnp.bfloat16, use_pretrained=True)
    
    # Profiler 配置
    profiler_context = None
    if False:  # 设为 True 启用 profiling
        print("\n启用 JAX Profiler...")
        profiler_context = jax.profiler.trace(
            "/dev/shm/jax-trace",
            create_perfetto_link=False
        )
    
    # 执行 DiT 测试
    results = dit_test(
        transformer,
        frames=frames,
        num_runs=num_runs,
        profiler_context=profiler_context,
    )
    
    # 打印统计结果
    print_results(results, frames)


# --- 脚本执行入口 ---
if __name__ == "__main__":
    # Set JAX config to enable compilation cache
    jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    
    warnings.filterwarnings('ignore')
    logging.getLogger().setLevel(logging.ERROR)
    
    # 执行 DiT 的TPU性能测试
    dit(frames=64, num_runs=3)