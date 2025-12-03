# 导入必要的库
import os
os.environ.setdefault('JAX_MEMORY_DEBUG', '0')  # 默认关闭内存调试

import time
import re
import math
import functools
import numpy as np
import jax
import jax.numpy as jnp
import torch
import torchax
from torchax.ops import ops_registry
from jax.tree_util import register_pytree_node
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
from jax.experimental.pallas.ops.tpu import splash_attention
from jax.experimental.shard_map import shard_map
from jax.experimental import mesh_utils
from tqdm import tqdm
import warnings
import logging
from contextlib import nullcontext
import sys

# 添加 HunyuanVideo-1.5-TPU 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..', 'HunyuanVideo-1.5-TPU'))

# --- TPU 专用: Mock 并行状态以禁用 CUDA 初始化 ---
# 在导入 HunyuanVideo 模型之前，需要设置一个 TPU 兼容的并行状态
from dataclasses import dataclass

@dataclass
class TPUParallelDims:
    """TPU 环境的并行状态 Mock，禁用 CUDA mesh 初始化"""
    sp: int = 1
    world_size: int = 1
    
    def __post_init__(self):
        # TPU 环境不需要 CUDA mesh
        pass
    
    @property
    def sp_enabled(self):
        return False
    
    @property
    def sp_group(self):
        return None
    
    @property
    def sp_mesh(self):
        return None
    
    @property
    def sp_rank(self):
        return 0
    
    @property
    def dp_enabled(self):
        return False

# 预先设置 mock 并行状态，阻止 CUDA 初始化
import hyvideo.commons.parallel_states as parallel_states_module
parallel_states_module.__parallel_dims = TPUParallelDims()
# 重写 initialize_parallel_state 和 get_parallel_state 函数
parallel_states_module.initialize_parallel_state = lambda sp=1: TPUParallelDims(sp=sp)
parallel_states_module.get_parallel_state = lambda: parallel_states_module.__parallel_dims

from hyvideo.models.transformers.hunyuanvideo_1_5_transformer import HunyuanVideo_1_5_DiffusionTransformer

# --- TPU 专用: 替换 sequence_parallel_attention 以使用标准 SDPA ---
# HunyuanVideo-1.5 的 torch 模式使用 flex_attention，不支持 JAX
# 我们需要替换为标准的 F.scaled_dot_product_attention
import hyvideo.models.transformers.modules.attention as attention_module

def _tpu_sequence_parallel_attention(q, k, v,
                                     img_q_len, img_kv_len,
                                     attn_mode=None, text_mask=None,
                                     attn_param=None,
                                     block_idx=None):
    """
    TPU 兼容版本的 sequence_parallel_attention
    使用标准 F.scaled_dot_product_attention 替代 flex_attention
    """
    assert attn_mode is not None
    query, encoder_query = q
    key, encoder_key = k
    value, encoder_value = v
    
    # 不使用 SP (sequence parallel)
    sequence_length = query.size(1)
    encoder_sequence_length = encoder_query.size(1)
    
    # 拼接 image 和 text tokens
    query = torch.cat([query, encoder_query], dim=1)
    key = torch.cat([key, encoder_key], dim=1)
    value = torch.cat([value, encoder_value], dim=1)
    
    # 转置为 (B, H, L, D) 格式
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    
    # 使用标准 SDPA (会被我们注册的 Splash Attention 替换)
    hidden_states = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
    )
    
    # 转置回 (B, L, H, D)
    hidden_states = hidden_states.transpose(1, 2)
    
    b, s, a, d = hidden_states.shape
    hidden_states = hidden_states.reshape(b, s, -1)
    
    return hidden_states

# 替换原始函数
attention_module.sequence_parallel_attention = _tpu_sequence_parallel_attention
attention_module.parallel_attention = lambda q, k, v, img_q_len, img_kv_len, attn_mode=None, text_mask=None, attn_param=None, block_idx=None: _tpu_sequence_parallel_attention(q, k, v, img_q_len, img_kv_len, attn_mode, text_mask, attn_param, block_idx)
print("已替换 sequence_parallel_attention 为 TPU 兼容版本")

# --- 全局配置 ---
MODEL_NAME = "tencent/HunyuanVideo-1.5"

#### Splash Attention 配置参数 ####
# Splash attention 块大小配置
# HunyuanVideo-1.5 序列长度较大（720p: ~108000 tokens），需要较大的块大小
BQSIZE = 2048           # Query 块大小
BKVSIZE = 2048          # Key/Value 块大小
BKVCOMPUTESIZE = 1024   # Key/Value 计算块大小

# 窗口大小（None 表示使用完整注意力）
WINDOW_SIZE = None

# 是否使用 K-smooth（对 key 进行平滑处理）
USE_K_SMOOTH = True

# Mesh 分片配置
USE_DP = False          # 是否使用 data parallelism
SP_NUM = 1              # Spatial parallelism 数量
USE_TP = True           # 是否使用 Tensor Parallel 模式（Megatron Column-Row风格）


# --- PyTree 注册 ---

def setup_pytree_registrations():
    """
    注册必要的pytree节点以支持JAX转换
    HunyuanVideo-1.5 返回 tuple，不需要注册特殊输出类
    """
    print("注册PyTree节点...")
    # HunyuanVideo-1.5 的 forward 返回 tuple (img, features_list)
    # 不需要注册额外的输出类
    print("  - HunyuanVideo-1.5 使用 tuple 输出，无需额外注册")


# --- Splash Attention 实现 ---

def _tpu_splash_attention(query, key, value, mesh, scale=None, is_causal=False, window_size=None):
    """
    TPU Splash Attention 实现（纯 JAX 版本）
    
    使用 JAX 的 Splash Attention 在 TPU 上高效计算注意力
    针对 HunyuanVideo-1.5 的大序列长度（720p 约 108,000 tokens）进行优化
    
    Args:
        query: Query 张量 (batch, heads, seq_len, head_dim)
        key: Key 张量 (batch, heads, seq_len, head_dim)
        value: Value 张量 (batch, heads, seq_len, head_dim)
        mesh: JAX 设备网格
        scale: 缩放因子（默认为 1/sqrt(head_dim)）
        is_causal: 是否使用因果掩码
        window_size: 局部注意力窗口大小
    """
    num_heads = query.shape[1]

    def _attention_on_slices(q, k, v):
        # 缩放 query 张量
        scale_factor = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
        q = q * scale_factor

        def pad_to_multiple(x, multiple, axis):
            seq_len = x.shape[axis]
            pad_len = (multiple - seq_len % multiple) % multiple
            if pad_len == 0:
                return x, seq_len
            pad_width = [(0, 0)] * x.ndim
            pad_width[axis] = (0, pad_len)
            return jnp.pad(x, pad_width), seq_len

        def kernel_3d(q_3d, k_3d, v_3d):
            q_seq_len = q_3d.shape[1]
            kv_seq_len = k_3d.shape[1]
            num_heads_on_device = q_3d.shape[0]

            # 填充到块大小的倍数
            q_3d_padded, q_orig_len = pad_to_multiple(q_3d, BQSIZE, axis=1)
            k_3d_padded, k_orig_len = pad_to_multiple(k_3d, BKVSIZE, axis=1)
            v_3d_padded, v_orig_len = pad_to_multiple(v_3d, BKVSIZE, axis=1)

            padded_q_seq_len = q_3d_padded.shape[1]
            padded_kv_seq_len = k_3d_padded.shape[1]

            # 创建注意力掩码
            if window_size is not None:
                mask_class = functools.partial(splash_attention.LocalMask, window_size=window_size, offset=0)
            else:
                mask_class = splash_attention.FullMask

            mask = splash_attention.MultiHeadMask(
                [mask_class((padded_q_seq_len, padded_kv_seq_len)) for _ in range(num_heads_on_device)]
            )

            # 配置块大小
            block_sizes = splash_attention.BlockSizes(
                block_q=min(BQSIZE, padded_q_seq_len),
                block_kv=min(BKVSIZE, padded_kv_seq_len),
                block_kv_compute=min(BKVCOMPUTESIZE, padded_kv_seq_len),
            )
            
            # 创建并执行 Splash attention kernel
            splash_kernel = splash_attention.make_splash_mha(
                mask=mask, block_sizes=block_sizes, head_shards=1, q_seq_shards=1
            )
            out = splash_kernel(q_3d_padded, k_3d_padded, v_3d_padded)
            
            # 移除填充
            return out[:, :q_orig_len, ...]

        # 在批次维度上映射 kernel
        vmapped_kernel = jax.vmap(kernel_3d, in_axes=(0, 0, 0), out_axes=0)
        return vmapped_kernel(q, k, v)

    # 根据设备数量和头数确定分片策略
    if num_heads < mesh.size:
        q_partition_spec = P()
        kv_partition_spec = P()
    else:
        if query.shape[2] == key.shape[2]:  # 自注意力
            q_partition_spec = P('dp', 'tp', 'sp', None)
            kv_partition_spec = P('dp', 'tp', None, None)
        else:  # 交叉注意力
            q_partition_spec = P('dp', None, ('tp', 'sp'), None)
            kv_partition_spec = P('dp', None, None, None)

    # 使用 shard_map 在设备间分片执行
    sharded_fn = shard_map(
        _attention_on_slices,
        mesh=mesh,
        in_specs=(q_partition_spec, kv_partition_spec, kv_partition_spec),
        out_specs=q_partition_spec,
        check_rep=False,
    )
    out = sharded_fn(query, key, value)
    
    return out


def splash_attention_fn(query, key, value, mesh, scale=None, is_causal=False, window_size=None):
    """
    Splash Attention 封装函数
    
    Args:
        query: Query 张量 (batch, heads, seq_len, head_dim)
        key: Key 张量 (batch, heads, seq_len, head_dim)
        value: Value 张量 (batch, heads, seq_len, head_dim)
        mesh: JAX 设备网格
        scale: 缩放因子
        is_causal: 是否使用因果掩码
        window_size: 局部注意力窗口大小
    
    Returns:
        attention 输出
    """
    # 可选的 K-smooth 处理
    if USE_K_SMOOTH:
        key_mean = jnp.mean(key, axis=2, keepdims=True)
        key = key - key_mean
    
    return _tpu_splash_attention(query, key, value, mesh, scale=scale, is_causal=is_causal, window_size=window_size)


def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
    env=None,
    window_size=None,
) -> torch.Tensor:
    """
    Scaled Dot-Product Attention 封装函数（torchax 兼容版本）
    将 PyTorch 张量转换为 JAX，执行 Splash Attention，再转回
    """
    if env is not None and hasattr(env.config, 'use_tpu_splash_attention') and env.config.use_tpu_splash_attention:
        # 使用 TPU Splash Attention
        jquery, jkey, jvalue = env.t2j_iso((query, key, value))
        
        res = splash_attention_fn(jquery, jkey, jvalue, env._mesh, scale=scale, is_causal=is_causal, window_size=window_size)
        return env.j2t_iso(res)
    
    # 回退到标准 PyTorch SDPA
    return torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
        is_causal=is_causal, scale=scale
    )


# --- HunyuanVideo-1.5 Transformer 权重分片策略 ---

# HunyuanVideo-1.5 权重分片策略 - Tensor Parallel模式（Megatron Column-Row风格）
# PyTorch weight shape: (out_features, in_features)
# Column Parallel (Q/K/V, FF1): 在 out_features 维度分片 -> (('tp', 'sp'), None)
# Row Parallel (Proj, FF2): 在 in_features 维度分片 -> (None, ('tp', 'sp'))
transformer_shardings_tp = {
    # === MMDoubleStreamBlock 层 ===
    # Image attention - Column Parallel: Q/K/V 在 out_features 分片（按heads切分）
    r'.*\.img_attn_q\.weight$': (('tp', 'sp'), None),
    r'.*\.img_attn_k\.weight$': (('tp', 'sp'), None),
    r'.*\.img_attn_v\.weight$': (('tp', 'sp'), None),
    # Image attention - Row Parallel: Proj 在 in_features 分片（聚合）
    r'.*\.img_attn_proj\.weight$': (None, ('tp', 'sp')),
    
    # Text attention - Column Parallel
    r'.*\.txt_attn_q\.weight$': (('tp', 'sp'), None),
    r'.*\.txt_attn_k\.weight$': (('tp', 'sp'), None),
    r'.*\.txt_attn_v\.weight$': (('tp', 'sp'), None),
    # Text attention - Row Parallel
    r'.*\.txt_attn_proj\.weight$': (None, ('tp', 'sp')),
    
    # Image MLP - Column Parallel: fc1 在 out_features 分片
    r'.*\.img_mlp\.fc1\.weight$': (('tp', 'sp'), None),
    # Image MLP - Row Parallel: fc2 在 in_features 分片
    r'.*\.img_mlp\.fc2\.weight$': (None, ('tp', 'sp')),
    
    # Text MLP - Column Parallel
    r'.*\.txt_mlp\.fc1\.weight$': (('tp', 'sp'), None),
    # Text MLP - Row Parallel
    r'.*\.txt_mlp\.fc2\.weight$': (None, ('tp', 'sp')),
    
    # === MMSingleStreamBlock 层 ===
    # Single stream - Column Parallel: Q/K/V/MLP 在 out_features 分片
    r'.*\.linear1_q\.weight$': (('tp', 'sp'), None),
    r'.*\.linear1_k\.weight$': (('tp', 'sp'), None),
    r'.*\.linear1_v\.weight$': (('tp', 'sp'), None),
    r'.*\.linear1_mlp\.weight$': (('tp', 'sp'), None),
    # Single stream - Row Parallel: linear2 在 in_features 分片（聚合）
    r'.*\.linear2\.linear\.weight$': (None, ('tp', 'sp')),
    
    # === 其他层 ===
    # txt_in (SingleTokenRefiner) - Column Parallel
    r'.*\.txt_in\..*\.to_q\.weight$': (('tp', 'sp'), None),
    r'.*\.txt_in\..*\.to_k\.weight$': (('tp', 'sp'), None),
    r'.*\.txt_in\..*\.to_v\.weight$': (('tp', 'sp'), None),
    # txt_in - Row Parallel
    r'.*\.txt_in\..*\.to_out\.0\.weight$': (None, ('tp', 'sp')),
    
    # byt5_in (ByT5Mapper) - Column Parallel
    r'.*\.byt5_in\.fc1\.weight$': (('tp', 'sp'), None),
    r'.*\.byt5_in\.fc3\.weight$': (('tp', 'sp'), None),
    # byt5_in - Row Parallel
    r'.*\.byt5_in\.fc2\.weight$': (None, ('tp', 'sp')),
    r'.*\.byt5_in\.fc4\.weight$': (None, ('tp', 'sp')),
}

# HunyuanVideo-1.5 权重分片策略 - FSDP模式（均匀分片，与TP相反）
transformer_shardings_fsdp = {
    # === MMDoubleStreamBlock 层 ===
    r'.*\.img_attn_q\.weight$': (None, ('tp', 'sp')),
    r'.*\.img_attn_k\.weight$': (None, ('tp', 'sp')),
    r'.*\.img_attn_v\.weight$': (None, ('tp', 'sp')),
    r'.*\.img_attn_proj\.weight$': (('tp', 'sp'), None),
    
    r'.*\.txt_attn_q\.weight$': (None, ('tp', 'sp')),
    r'.*\.txt_attn_k\.weight$': (None, ('tp', 'sp')),
    r'.*\.txt_attn_v\.weight$': (None, ('tp', 'sp')),
    r'.*\.txt_attn_proj\.weight$': (('tp', 'sp'), None),
    
    r'.*\.img_mlp\.fc1\.weight$': (None, ('tp', 'sp')),
    r'.*\.img_mlp\.fc2\.weight$': (('tp', 'sp'), None),
    
    r'.*\.txt_mlp\.fc1\.weight$': (None, ('tp', 'sp')),
    r'.*\.txt_mlp\.fc2\.weight$': (('tp', 'sp'), None),
    
    # === MMSingleStreamBlock 层 ===
    r'.*\.linear1_q\.weight$': (None, ('tp', 'sp')),
    r'.*\.linear1_k\.weight$': (None, ('tp', 'sp')),
    r'.*\.linear1_v\.weight$': (None, ('tp', 'sp')),
    r'.*\.linear1_mlp\.weight$': (None, ('tp', 'sp')),
    r'.*\.linear2\.linear\.weight$': (('tp', 'sp'), None),
    
    # txt_in
    r'.*\.txt_in\..*\.to_q\.weight$': (None, ('tp', 'sp')),
    r'.*\.txt_in\..*\.to_k\.weight$': (None, ('tp', 'sp')),
    r'.*\.txt_in\..*\.to_v\.weight$': (None, ('tp', 'sp')),
    r'.*\.txt_in\..*\.to_out\.0\.weight$': (('tp', 'sp'), None),
    
    # byt5_in
    r'.*\.byt5_in\.fc1\.weight$': (None, ('tp', 'sp')),
    r'.*\.byt5_in\.fc2\.weight$': (('tp', 'sp'), None),
    r'.*\.byt5_in\.fc3\.weight$': (None, ('tp', 'sp')),
    r'.*\.byt5_in\.fc4\.weight$': (('tp', 'sp'), None),
}


def shard_weights_transformer(mesh, weights, use_tp=True):
    """
    对 HunyuanVideo-1.5 Transformer 模型的权重进行分片
    
    Args:
        mesh: JAX设备网格
        weights: 模型权重字典
        use_tp: 是否使用Tensor Parallel模式（默认True），否则使用FSDP模式
        
    Returns:
        分片后的权重字典
    """
    # 选择分片策略
    sharding_dict = transformer_shardings_tp if use_tp else transformer_shardings_fsdp
    
    result = {}
    matched_count = 0
    unmatched_count = 0
    
    for k, v in weights.items():
        # 尝试匹配分片规则
        matched = False
        for target, sharding in sharding_dict.items():
            if re.fullmatch(target, k) is not None:
                # 找到匹配的模式，应用分片
                v.apply_jax_(jax.device_put, NamedSharding(mesh, P(*sharding)))
                matched = True
                matched_count += 1
                break
        
        if not matched:
            # 没有匹配到任何模式，复制到所有设备
            v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
            unmatched_count += 1
        
        result[k] = v
    
    print(f"  权重分片完成: {matched_count} 个匹配分片规则, {unmatched_count} 个复制到所有设备")
    return result


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
    # HunyuanVideo-1.5 返回 tuple (img, features_list)
    target = output
    if isinstance(output, tuple):
        target = output[0]  # 取第一个元素（img）

    if hasattr(target, '_elem'):
        jax.block_until_ready(target._elem)
    else:
        jax.block_until_ready(target)
    
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
    
    # TPU 内存统计（可能需要根据实际环境调整）
    # JAX没有直接的峰值内存API，这里返回一个占位值
    # 实际使用时可以通过TPU profiler或其他工具获取
    peak_memory_mb = 0.0  # 占位值
    
    return output, peak_memory_mb, time_cost


# --- 结果打印 ---

def print_results(results, frames, resolution):
    """
    打印测试结果的统计信息。
    
    参数:
    results (list of dict): 测试结果的列表。
    frames (int): 测试的帧数。
    resolution (str): 视频分辨率。
    """
    if not results:
        print("没有测试结果")
        return
    
    times = [r['time'] for r in results]
    
    print(f"\n=== HunyuanVideo-1.5 DiT TPU 测试结果 (帧数: {frames}, 分辨率: {resolution}) ===")
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

def dit_test(transformer, frames=121, resolution='720p', num_runs=1, warmup_runs=1, 
             profiler_context=None, enable_cfg=False):
    """
    测试 HunyuanVideo-1.5 DiT (Diffusion Transformer) 模型在TPU上的性能。
    先进行预热运行，然后对指定帧数重复运行多次以获取稳定的性能数据。
    
    参数:
    transformer (torch.nn.Module): 已加载的 Transformer 模型。
    frames (int): 测试的视频帧数，默认121帧。
    resolution (str): 视频分辨率 ('480p' 或 '720p')。
    num_runs (int): 重复运行的次数，默认1次。
    warmup_runs (int): 预热运行次数，默认1次（不计入统计）。
    profiler_context: Profiler上下文管理器，可选。
    enable_cfg (bool): 是否启用 CFG（默认False）。
    """
    # HunyuanVideo-1.5 的输入维度
    batch = 2 if enable_cfg else 1
    
    # 重要：HunyuanVideo-1.5 的 VAE 输出 32 通道 latents
    channel = 32  # VAE latent channels
    
    # 根据分辨率设置高度和宽度
    # VAE 空间压缩比 = 16 倍, 时间压缩比 = 4 倍
    if resolution == '720p':
        height = 45   # 720 / 16 = 45
        width = 80    # 1280 / 16 = 80
    else:  # 480p
        height = 30   # 480 / 16 = 30
        width = 53    # 848 / 16 = 53
    
    # Transformer 的输入帧数
    latent_frames = (frames - 1) // 4 + 1
    
    # 输入格式: latents (32ch) + cond_latents (33ch) = 65 channels
    total_input_channels = channel + (channel + 1)  # 32 + 33 = 65
    
    # 计算 patch 后的 token 数量
    patch_t, patch_h, patch_w = 1, 1, 1
    token_t = latent_frames // patch_t
    token_h = height // patch_h
    token_w = width // patch_w
    total_tokens = token_t * token_h * token_w
    
    cfg_status = "启用" if enable_cfg else "禁用 (guidance_scale=1.0)"
    print(f"输入配置:")
    print(f"  帧数: {frames}, 分辨率: {resolution}, CFG: {cfg_status}")
    print(f"  Latent shape (单个): [{batch}, {channel}, {latent_frames}, {height}, {width}]")
    print(f"  Input shape (拼接后): [{batch}, {total_input_channels}, {latent_frames}, {height}, {width}]")
    print(f"  Token shape (after patch [{patch_t},{patch_h},{patch_w}]): {token_t} x {token_h} x {token_w} = {total_tokens} tokens")
    
    # 定义运行单次测试的函数
    def run_single_test():
        # --- 准备模型输入 ---
        # 1. 创建主要的 latents
        latents = torch.randn((batch, channel, latent_frames, height, width),
                             dtype=torch.bfloat16).to('jax')
        
        # 2. 创建条件 latents (用于 multitask，t2v 时为零)
        cond_latents_only = torch.zeros((batch, channel, latent_frames, height, width),
                                        dtype=torch.bfloat16).to('jax')
        
        # 3. 创建 mask (用于 multitask，t2v 时全为零)
        mask = torch.zeros((batch, 1, latent_frames, height, width),
                          dtype=torch.bfloat16).to('jax')
        
        # 4. 合并 condition 和 mask 成 cond_latents (33 通道)
        cond_latents = torch.cat([cond_latents_only, mask], dim=1)
        
        # 5. 拼接 latents 和 cond_latents 成完整输入 (65 通道)
        hidden_states = torch.cat([latents, cond_latents], dim=1)

        # 6. 创建时间步 (timestep)
        timestep = torch.tensor([999], dtype=torch.long, device='jax')

        # 7. 创建文本嵌入 (text_states)
        text_seq_len = 1000
        text_states = torch.randn((batch, text_seq_len, 3584),
                                 dtype=torch.bfloat16).to('jax')
        
        # 8. text_states_2 为 None（720p_i2v 配置）
        text_states_2 = None

        # 9. 创建注意力掩码 (encoder_attention_mask)
        # 使用 bool 类型避免 parallel_attention 中的 assert attn_mask.max() 检查
        encoder_attention_mask = torch.ones((batch, text_seq_len),
                                           dtype=torch.bool, device='jax')
        
        # 10. extra_kwargs - glyph_byT5_v2 已禁用，不需要传递
        extra_kwargs = None
        
        # 11. 创建旋转位置编码 (freqs_cos, freqs_sin)
        # 类似于 cogvideo 的方式，预先生成随机 rotary embedding 避免混合 Tensor 问题
        # head_dim = hidden_size / heads_num = 2048 / 16 = 128
        head_dim = 128
        freqs_cos = torch.randn((total_tokens, head_dim), dtype=torch.float32).to('jax')
        freqs_sin = torch.randn((total_tokens, head_dim), dtype=torch.float32).to('jax')

        # 定义调用 Transformer 模型的函数
        dit_call = lambda: transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            text_states=text_states,
            text_states_2=text_states_2,
            encoder_attention_mask=encoder_attention_mask,
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            return_dict=False,
            extra_kwargs=extra_kwargs,
        )

        # 记录执行时间
        output, peak_memory_mb, time_cost = record_peak_memory_tpu(dit_call)
        del output  # 释放内存
        
        return peak_memory_mb, time_cost
    
    # 预热运行
    print(f"\n开始预热运行 (预热次数: {warmup_runs})")
    for run in tqdm(range(warmup_runs), desc="Warmup HunyuanVideo-1.5 DiT on TPU"):
        try:
            run_single_test()
        except Exception as e:
            print(f"预热第 {run + 1} 次运行出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    # 正式测试运行
    results = []
    print(f"\n开始正式测试 HunyuanVideo-1.5 DiT TPU 性能 (运行次数: {num_runs})")
    
    context = profiler_context if profiler_context else nullcontext()
    
    with context:
        for run in tqdm(range(num_runs), desc="Testing HunyuanVideo-1.5 DiT on TPU"):
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

def load_transformer(model_path, resolution='720p'):
    """
    加载 HunyuanVideo-1.5 Transformer 模型（在普通PyTorch环境中）
    
    Args:
        model_path: 模型检查点目录路径
        resolution: 视频分辨率 ('480p' 或 '720p')
        
    Returns:
        transformer: Transformer模型
    """
    # 构建正确的模型目录路径
    if resolution == '720p':
        model_dir = os.path.join(model_path, 'transformer', '720p_i2v')
    else:  # 480p
        model_dir = os.path.join(model_path, 'transformer', '480p_i2v')
    
    print(f"正在加载模型: {model_dir}")
    
    # 使用 torch 模式加载（不是 flash，因为 TPU 不支持）
    transformer = HunyuanVideo_1_5_DiffusionTransformer.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_mode="torch",  # TPU 使用 torch 模式，SDPA 会被替换为 Splash Attention
    )
    
    print("模型加载完成")
    print(f"模型参数量: {sum(p.numel() for p in transformer.parameters()) / 1e9:.2f}B")
    
    # 禁用 glyph_byT5_v2，因为 reorder_txt_token 使用布尔索引，torchax 不支持
    # 对于纯 DiT 性能测试，不需要 ByT5 功能
    if hasattr(transformer, 'glyph_byT5_v2') and transformer.glyph_byT5_v2:
        print("注意: 禁用 glyph_byT5_v2 (DiT 性能测试不需要)")
        transformer.glyph_byT5_v2 = False
    
    # 禁用 cond_type_embedding，避免依赖 text_mask.device
    if hasattr(transformer, 'cond_type_embedding') and transformer.cond_type_embedding is not None:
        print("注意: 禁用 cond_type_embedding (避免 TPU 兼容性问题)")
        transformer.cond_type_embedding = None
    
    return transformer


def setup_transformer_for_tpu(transformer):
    """
    设置 HunyuanVideo-1.5 Transformer 以在 TPU 上运行
    
    Args:
        transformer: 已加载的Transformer模型
        
    Returns:
        transformer: 编译后的Transformer模型
        env: torchax环境
        mesh: JAX mesh
    """
    print("\n配置 HunyuanVideo-1.5 Transformer 以使用 JAX 和 Splash Attention...")
    
    # 计算mesh维度
    tp_dim, dp_dim, sp_dim = jax.device_count(), 1, 1
    if USE_DP:
        tp_dim //= 2
        dp_dim = 2
    
    if SP_NUM > 1:
        tp_dim //= SP_NUM
        sp_dim = SP_NUM
    
    print(f"  Mesh 维度: tp_dim={tp_dim}, dp_dim={dp_dim}, sp_dim={sp_dim}")
    print(f"  总设备数: {jax.device_count()}")
    
    # 创建三维 mesh (tp, dp, sp)
    mesh_devices = mesh_utils.create_device_mesh((tp_dim, dp_dim, sp_dim), allow_split_physical_axes=True)
    mesh = Mesh(mesh_devices, ('tp', 'dp', 'sp'))
    
    # 创建 torchax 环境
    env = torchax.default_env()
    
    # 配置环境以启用 TPU Splash Attention
    env._mesh = mesh
    env.config.use_tpu_splash_attention = True

    # 注册自定义的 Scaled Dot-Product Attention
    print(f"- 注册 Splash Attention（窗口大小: {WINDOW_SIZE}）...")
    custom_attention = functools.partial(
        scaled_dot_product_attention,
        env=env,
        window_size=WINDOW_SIZE
    )
    
    # 覆盖 PyTorch 的 scaled_dot_product_attention
    op_to_override = torch.nn.functional.scaled_dot_product_attention
    env._ops[op_to_override] = ops_registry.Operator(
        op_to_override,
        custom_attention,
        is_jax_function=False,
        is_user_defined=True,
        needs_env=False,
        is_view_op=False,
    )
    
    # 辅助函数：将模块权重移动到 XLA
    def _move_module_to_xla(module):
        """将模块的权重转换为 JAX Array，但先在 CPU 上操作"""
        with jax.default_device('cpu'):
            state_dict = module.state_dict()
            state_dict = env.to_xla(state_dict)
            module.load_state_dict(state_dict, assign=True)
    
    with env:
        # 对 Transformer 进行处理：先移到 XLA，再分片
        print("- 将 HunyuanVideo-1.5 Transformer 移到 XLA 并进行分片...")
        _move_module_to_xla(transformer)
        transformer_weights = shard_weights_transformer(mesh, transformer.state_dict(), use_tp=USE_TP)
        transformer.load_state_dict(transformer_weights, assign=True, strict=False)
        
        # 确保所有权重已分片完成
        torchax.interop.call_jax(jax.block_until_ready, transformer_weights)
        
        # 编译transformer
        print("- 编译 HunyuanVideo-1.5 Transformer...")
        transformer = torchax.compile(
            transformer,
            torchax.CompileOptions(
                jax_jit_kwargs={'static_argnames': ('return_dict', 'output_features', 'mask_type')}
            )
        )
    
    print("HunyuanVideo-1.5 Transformer 配置完成")
    return transformer, env, mesh


def dit(frames=121, resolution='720p', num_runs=3, model_path='/dev/shm/HunyuanVideo-1.5/ckpts', 
        enable_cfg=False):
    """
    执行 HunyuanVideo-1.5 DiT 模型在 TPU 上的性能测试。
    
    参数:
    frames (int): 测试的视频帧数，默认121帧。
    resolution (str): 视频分辨率 ('480p' 或 '720p')。
    num_runs (int): 重复运行的次数，默认3次。
    model_path (str): 模型检查点目录路径。
    enable_cfg (bool): 是否启用 CFG（默认False）。
    """
    print("--- 开始 HunyuanVideo-1.5 DiT TPU 性能测试 ---")
    
    # 设置JAX配置
    warnings.filterwarnings('ignore')
    logging.getLogger().setLevel(logging.ERROR)
    
    # 设置随机种子
    import random
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    torch.set_default_dtype(torch.bfloat16)
    
    # 注册PyTree节点
    setup_pytree_registrations()
    
    # 加载Transformer模型
    transformer = load_transformer(model_path, resolution)
    
    # 设置Transformer在TPU上运行
    transformer, env, mesh = setup_transformer_for_tpu(transformer)
    
    # Profiler 配置（可选）
    profiler_context = None
    if False:  # 设为 True 启用 profiling
        print("\n启用 JAX Profiler...")
        profiler_context = jax.profiler.trace(
            "/dev/shm/jax-trace",
            create_perfetto_link=False
        )
    
    # 在mesh和env上下文中执行测试
    with mesh, env:
        # 执行 DiT 测试
        results = dit_test(
            transformer,
            frames=frames,
            resolution=resolution,
            num_runs=num_runs,
            enable_cfg=enable_cfg,
            profiler_context=profiler_context,
        )
    
    # 打印统计结果
    print_results(results, frames, resolution)


# --- 脚本执行入口 ---
if __name__ == "__main__":
    import argparse
    
    # Set JAX config to enable compilation cache
    jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

    warnings.filterwarnings('ignore', message='.*dtype.*int64.*truncated to dtype int32.*')
    logging.getLogger().setLevel(logging.ERROR)
    
    parser = argparse.ArgumentParser(description='HunyuanVideo-1.5 DiT TPU Performance Test')
    parser.add_argument('--frames', type=int, default=121,
                       help='Number of frames to test (default: 121)')
    parser.add_argument('--resolution', type=str, default='720p',
                       choices=['480p', '720p'],
                       help='Video resolution (default: 720p)')
    parser.add_argument('--num_runs', type=int, default=3,
                       help='Number of test runs (default: 3)')
    parser.add_argument('--model_path', type=str, default='/dev/shm/HunyuanVideo-1.5/ckpts',
                       help='Path to model checkpoints directory (default: /dev/shm/HunyuanVideo-1.5/ckpts)')
    parser.add_argument('--enable_cfg', action='store_true',
                       help='Enable Classifier-Free Guidance (default: False)')
    
    args = parser.parse_args()
    
    # 执行 DiT 的TPU性能测试
    dit(
        frames=args.frames,
        resolution=args.resolution,
        num_runs=args.num_runs,
        model_path=args.model_path,
        enable_cfg=args.enable_cfg
    )