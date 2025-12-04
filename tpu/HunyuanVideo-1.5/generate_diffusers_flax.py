#!/usr/bin/env python3
"""
HunyuanVideo-1.5 Generation using Diffusers with TPU/JAX (torchax)
基于 generate_flax.py 的模式，使用 Splash Attention 在 TPU 上运行

VAE 使用纯 Flax 版本，Transformer 使用 torchax + Splash Attention
"""

import sys
sys.path.insert(0, "/home/chrisya/diffusers-tpu/src")

import os
import time
import re
import math
import functools
import argparse
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
from diffusers import HunyuanVideo15Pipeline
from diffusers.utils import export_to_video
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.autoencoders.autoencoder_kl_hunyuanvideo15_flax import (
    FlaxAutoencoderKLHunyuanVideo15,
    FlaxAutoencoderKLHunyuanVideo15Config,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutputWithPastAndCrossAttentions
import warnings
import logging
from flax import nnx


# === 模型配置 ===
MODEL_NAME = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v"

# === Splash Attention 配置参数 ===
BQSIZE = 2048           # Query 块大小
BKVSIZE = 2048          # Key/Value 块大小
BKVCOMPUTESIZE = 1024   # Key/Value 计算块大小
WINDOW_SIZE = None      # 窗口大小（None 表示使用完整注意力）

# === Mesh 分片配置 ===
USE_DP = False          # 是否使用 data parallelism
SP_NUM = 1              # Spatial parallelism 数量
USE_TP = True           # 是否使用 Tensor Parallel 模式


# === 工具函数 ===

def to_torch_recursive(x):
    """递归地将 JAX 数组转换为 PyTorch 张量"""
    if 'ArrayImpl' in str(type(x)) or isinstance(x, jnp.ndarray):
        np_array = np.array(x)
        if hasattr(x, 'dtype') and x.dtype == jnp.bfloat16:
            return torch.from_numpy(np_array.astype(np.float32)).to(torch.bfloat16)
        else:
            return torch.from_numpy(np_array)
    elif isinstance(x, (list, tuple)):
        return type(x)(to_torch_recursive(xx) for xx in x)
    elif isinstance(x, dict):
        return {k: to_torch_recursive(v) for k, v in x.items()}
    elif hasattr(x, 'sample'):
        sample = to_torch_recursive(x.sample)
        if hasattr(x, 'replace'):
            return x.replace(sample=sample)
        else:
            return sample
    else:
        return x


def to_jax_recursive(x):
    """递归地将 PyTorch 张量转换为 JAX 数组"""
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.bfloat16:
            return jnp.array(x.detach().to(torch.float32).cpu().numpy()).astype(jnp.bfloat16)
        else:
            return jnp.array(x.detach().cpu().numpy())
    elif isinstance(x, (list, tuple)):
        return type(x)(to_jax_recursive(xx) for xx in x)
    elif isinstance(x, dict):
        return {k: to_jax_recursive(v) for k, v in x.items()}
    else:
        return x


def setup_pytree_registrations():
    """
    注册必要的pytree节点以支持JAX转换
    """
    print("注册PyTree节点...")
    
    def model_output_flatten(obj):
        return obj.to_tuple(), type(obj)

    def model_output_unflatten(aux, children):
        return aux(*children)
    
    OUTPUT_CLASSES = [
        BaseModelOutputWithPooling,
        BaseModelOutputWithPastAndCrossAttentions,
        DecoderOutput,
    ]

    for cls in OUTPUT_CLASSES:
        register_pytree_node(cls, model_output_flatten, model_output_unflatten)
        print(f"  - {cls.__name__} 已注册")


# === Splash Attention 实现 ===
# 参考 dit_flax.py 的实现

# 保存原始的 SDPA 实现，用于非 XLA tensor 的情况
_ORIGINAL_SDPA = None


def _is_xla_tensor(tensor):
    """检测 tensor 是否是 XLA/torchax tensor"""
    if tensor is None:
        return False
    # 检查 torchax tensor 的特征
    if hasattr(tensor, '_elem'):
        return True
    # 检查设备类型
    if hasattr(tensor, 'device'):
        device_str = str(tensor.device)
        if 'jax' in device_str or 'xla' in device_str:
            return True
    return False


def _sdpa_reference(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
) -> torch.Tensor:
    """
    Scaled Dot-Product Attention 参考实现（用于有 mask 的情况）
    
    关键修复：当 attention mask 导致某行全为 -inf 时，softmax 会产生 NaN。
    使用 masked_fill 将 NaN 替换为 0（这些 padding 位置的输出不会被使用）。
    """
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    
    # 处理 GQA
    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)
    
    # 计算 attention weights
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    
    # 应用 mask
    if is_causal:
        assert attn_mask is None
        causal_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_weight = attn_weight.masked_fill(causal_mask.logical_not(), float("-inf"))
    
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_weight = attn_weight.masked_fill(attn_mask.logical_not(), float("-inf"))
        else:
            attn_weight = attn_weight + attn_mask
    
    # Softmax + NaN 修复
    # 当一行全是 -inf 时，softmax 产生 NaN，用 0 替换（padding 位置的输出不会被使用）
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = attn_weight.masked_fill(torch.isnan(attn_weight), 0.0)
    
    if dropout_p > 0:
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    
    return attn_weight @ value


def _tpu_splash_attention(query, key, value, mesh, scale=None, is_causal=False, window_size=None):
    """
    TPU Splash Attention 实现（纯 JAX 版本）
    参考 dit_flax.py 的实现
    """
    num_heads = query.shape[1]

    def _attention_on_slices(q, k, v):
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
            num_heads_on_device = q_3d.shape[0]

            q_3d_padded, q_orig_len = pad_to_multiple(q_3d, BQSIZE, axis=1)
            k_3d_padded, k_orig_len = pad_to_multiple(k_3d, BKVSIZE, axis=1)
            v_3d_padded, v_orig_len = pad_to_multiple(v_3d, BKVSIZE, axis=1)

            padded_q_seq_len = q_3d_padded.shape[1]
            padded_kv_seq_len = k_3d_padded.shape[1]

            if window_size is not None:
                mask_class = functools.partial(splash_attention.LocalMask, window_size=window_size, offset=0)
            else:
                mask_class = splash_attention.FullMask

            mask = splash_attention.MultiHeadMask(
                [mask_class((padded_q_seq_len, padded_kv_seq_len)) for _ in range(num_heads_on_device)]
            )

            block_sizes = splash_attention.BlockSizes(
                block_q=min(BQSIZE, padded_q_seq_len),
                block_kv=min(BKVSIZE, padded_kv_seq_len),
                block_kv_compute=min(BKVCOMPUTESIZE, padded_kv_seq_len),
            )
            
            splash_kernel = splash_attention.make_splash_mha(
                mask=mask, block_sizes=block_sizes, head_shards=1, q_seq_shards=1
            )
            out = splash_kernel(q_3d_padded, k_3d_padded, v_3d_padded)
            
            return out[:, :q_orig_len, ...]

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

    sharded_fn = shard_map(
        _attention_on_slices,
        mesh=mesh,
        in_specs=(q_partition_spec, kv_partition_spec, kv_partition_spec),
        out_specs=q_partition_spec,
        check_rep=False,
    )
    out = sharded_fn(query, key, value)
    
    return out


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
    Scaled Dot-Product Attention 封装函数
    
    路由策略：
    - 普通 torch.Tensor（text encoder）-> 原始 SDPA
    - XLA tensor + attn_mask（VAE causal attention）-> 参考实现
    - XLA tensor + 无 mask -> TPU Splash Attention
    """
    global _ORIGINAL_SDPA
    
    # 非 XLA tensor 使用原始 SDPA（text encoder 等）
    if not _is_xla_tensor(query):
        if _ORIGINAL_SDPA is not None:
            return _ORIGINAL_SDPA(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)
        return _sdpa_reference(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)
    
    # 有 attn_mask 时使用参考实现（Splash Attention 不支持任意 mask）
    if attn_mask is not None:
        return _sdpa_reference(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)
    
    # 使用 TPU Splash Attention
    if env is not None and hasattr(env.config, 'use_tpu_splash_attention') and env.config.use_tpu_splash_attention:
        jquery, jkey, jvalue = env.t2j_iso((query, key, value))
        res = _tpu_splash_attention(jquery, jkey, jvalue, env._mesh, scale=scale, is_causal=is_causal, window_size=window_size)
        return env.j2t_iso(res)
    
    return _sdpa_reference(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)


# === HunyuanVideo-1.5 Transformer 权重分片策略 ===
# 参考 dit_flax.py 的实现

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
    
    # === Diffusers 版本的通用命名 ===
    # 兼容 Diffusers 的标准命名约定
    r'.*\.to_q\.weight$': (('tp', 'sp'), None),
    r'.*\.to_k\.weight$': (('tp', 'sp'), None),
    r'.*\.to_v\.weight$': (('tp', 'sp'), None),
    r'.*\.to_out\.0\.weight$': (None, ('tp', 'sp')),
    
    # Feedforward layers
    r'.*\.ff\.net\.0\.proj\.weight$': (('tp', 'sp'), None),
    r'.*\.ff\.net\.2\.weight$': (None, ('tp', 'sp')),
    
    # Linear projections
    r'.*\.proj_in\.weight$': (('tp', 'sp'), None),
    r'.*\.proj_out\.weight$': (None, ('tp', 'sp')),
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
    
    # === Diffusers 版本的通用命名 ===
    r'.*\.to_q\.weight$': (None, ('tp', 'sp')),
    r'.*\.to_k\.weight$': (None, ('tp', 'sp')),
    r'.*\.to_v\.weight$': (None, ('tp', 'sp')),
    r'.*\.to_out\.0\.weight$': (('tp', 'sp'), None),
    
    r'.*\.ff\.net\.0\.proj\.weight$': (None, ('tp', 'sp')),
    r'.*\.ff\.net\.2\.weight$': (('tp', 'sp'), None),
    
    r'.*\.proj_in\.weight$': (None, ('tp', 'sp')),
    r'.*\.proj_out\.weight$': (('tp', 'sp'), None),
}

# Text Encoder sharding 策略 (Llama 风格)
text_encoder_shardings = {
    r'embed_tokens\.weight$': (('tp', 'dp', 'sp'),),
    # Self Attention
    r'layers\.\d+\.self_attn\.q_proj\.weight$': (('tp', 'dp', 'sp'),),
    r'layers\.\d+\.self_attn\.k_proj\.weight$': (('tp', 'dp', 'sp'),),
    r'layers\.\d+\.self_attn\.v_proj\.weight$': (('tp', 'dp', 'sp'),),
    r'layers\.\d+\.self_attn\.o_proj\.weight$': (None, ('tp', 'dp', 'sp')),
    # MLP
    r'layers\.\d+\.mlp\.gate_proj\.weight$': (('tp', 'dp', 'sp'),),
    r'layers\.\d+\.mlp\.up_proj\.weight$': (('tp', 'dp', 'sp'),),
    r'layers\.\d+\.mlp\.down_proj\.weight$': (None, ('tp', 'dp', 'sp')),
    # Layer Norms
    r'layers\.\d+\.input_layernorm\.weight$': (None,),
    r'layers\.\d+\.post_attention_layernorm\.weight$': (None,),
    r'norm\.weight$': (None,),
}


def shard_weights(mesh, weights, sharding_dict):
    """
    对模型权重进行分片
    
    Args:
        mesh: JAX设备网格
        weights: 模型权重字典
        sharding_dict: 分片规则字典
        
    Returns:
        分片后的权重字典
    """
    result = {}
    matched_count = 0
    unmatched_count = 0
    
    for k, v in weights.items():
        matched = False
        for target, sharding in sharding_dict.items():
            if re.fullmatch(target, k) is not None:
                v.apply_jax_(jax.device_put, NamedSharding(mesh, P(*sharding)))
                matched = True
                matched_count += 1
                break
        
        if not matched:
            v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
            unmatched_count += 1
        
        result[k] = v
    
    print(f"  权重分片完成: {matched_count} 个匹配规则, {unmatched_count} 个复制到所有设备")
    return result


def shard_weights_transformer(mesh, weights, use_tp=True):
    """对 Transformer 模型的权重进行分片"""
    sharding_dict = transformer_shardings_tp if use_tp else transformer_shardings_fsdp
    return shard_weights(mesh, weights, sharding_dict)


def shard_weights_text_encoder(mesh, weights):
    """对 Text Encoder 的权重进行分片"""
    return shard_weights(mesh, weights, text_encoder_shardings)


def shard_weights_vae(mesh, weights):
    """对 VAE 的权重进行分片（全部复制）"""
    result = {}
    for k, v in weights.items():
        v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
        result[k] = v
    return result


# === VAE 代理 ===

class FlaxVAEProxy:
    """Flax VAE 的代理类，包装纯 Flax VAE 使其与 diffusers Pipeline 兼容
    
    主要功能：
    1. 将 torchax tensor 转换为 JAX array 进行解码
    2. 将 JAX array 输出转换回 torchax tensor
    3. 保持与原始 VAE 相同的接口（config, scaling_factor 等）
    
    注意：scaling_factor 由 Pipeline 处理，这里不再重复处理
    """
    def __init__(self, flax_vae, original_config, env):
        self._vae = flax_vae
        self.config = original_config  # 保持原始 PyTorch VAE 的 config
        self._env = env
        self.dtype = torch.bfloat16
        
        # 复制 tiling 相关属性
        self.use_tiling = False
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_latent_min_height = 16
        self.tile_latent_min_width = 16
        self.tile_overlap_factor = 0.25
    
    def decode(self, latents, return_dict=True):
        """解码 latents 到视频帧
        
        Args:
            latents: torchax tensor [B, C, T, H, W] 或 [B, T, H, W, C]
            return_dict: 是否返回 DecoderOutput
            
        Returns:
            DecoderOutput 或 tensor
        """
        # 将 torchax tensor 转换为 JAX array
        if hasattr(latents, '_elem'):
            # torchax tensor -> JAX array
            jax_latents = latents._elem
        else:
            # 普通 torch tensor -> JAX array
            jax_latents = jnp.array(latents.detach().cpu().numpy())
        
        # HunyuanVideo 的 latents 是 BCTHW 格式，需要转换为 BTHWC
        # PyTorch VAE 使用 BCTHW，Flax VAE 使用 BTHWC
        if jax_latents.ndim == 5 and jax_latents.shape[1] == 32:
            # [B, C, T, H, W] -> [B, T, H, W, C]
            jax_latents = jnp.transpose(jax_latents, (0, 2, 3, 4, 1))
        
        # 调用 Flax VAE 解码
        # 注意：Flax VAE 的 decode 输出已经是 [B, T, H, W, C] 格式
        output = self._vae.decode(jax_latents)
        
        # 输出从 BTHWC 转换为 BCTHW（diffusers Pipeline 期望的格式）
        # [B, T, H, W, C] -> [B, C, T, H, W]
        output = jnp.transpose(output, (0, 4, 1, 2, 3))
        
        # 将 JAX array 转换回 torchax tensor
        output_tensor = self._env.j2t_iso(output)
        
        if return_dict:
            return DecoderOutput(sample=output_tensor)
        return output_tensor
    
    def enable_tiling(self, tile_sample_min_height=256, tile_sample_min_width=256, tile_overlap_factor=0.25):
        """启用 VAE tiling 以减少内存使用"""
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width
        self.tile_overlap_factor = tile_overlap_factor
        # 计算 latent tile 尺寸（16x 空间压缩）
        self.tile_latent_min_height = tile_sample_min_height // 16
        self.tile_latent_min_width = tile_sample_min_width // 16
        # 同时在 Flax VAE 上启用
        self._vae.enable_tiling(tile_sample_min_height, tile_sample_min_width, tile_overlap_factor)
        return self
    
    def disable_tiling(self):
        """禁用 VAE tiling"""
        self.use_tiling = False
        self._vae.disable_tiling()
        return self


def load_flax_vae(mesh, original_vae_config, dtype=jnp.bfloat16):
    """加载 Flax VAE 模型
    
    Args:
        mesh: JAX device mesh
        original_vae_config: 原始 PyTorch VAE 的 config
        dtype: 数据类型
        
    Returns:
        Flax VAE 模型实例
    """
    print("\n加载 Flax VAE 模型...")
    
    # 创建 Flax VAE config
    config = FlaxAutoencoderKLHunyuanVideo15Config()
    
    # 创建随机数生成器
    key = jax.random.key(0)
    rngs = nnx.Rngs(key)
    
    # 创建 Flax VAE
    flax_vae = FlaxAutoencoderKLHunyuanVideo15(
        config=config,
        rngs=rngs,
        dtype=dtype,
    )
    
    print(f"  Flax VAE 已创建（使用随机初始化权重）")
    print(f"  数据类型: {dtype}")
    
    # TODO: 从 PyTorch 权重加载（需要实现权重转换）
    # 目前使用随机初始化，可以后续添加 from_pretrained 支持
    
    return flax_vae


# 保留旧的 VAEProxy 以便兼容（但不再使用）
class VAEProxy:
    """VAE 的 torchax 兼容代理（已废弃，使用 FlaxVAEProxy）
    
    注意：不要在这里处理 scaling_factor，因为 Pipeline 内部已经处理了。
    这个代理主要用于确保 VAE 与 torchax 环境兼容。
    """
    def __init__(self, vae, env):
        self._vae = vae
        self._env = env
        self.config = vae.config
        self.dtype = torch.bfloat16
    
    def __getattr__(self, name):
        return getattr(self._vae, name)
    
    def decode(self, latents, return_dict=True):
        """解码 latents 到视频帧
        
        注意：scaling_factor 已经在 Pipeline 中处理，这里不再重复处理
        """
        # 直接执行解码，不处理 scaling（Pipeline 内部已处理）
        output = self._vae.decode(latents, return_dict=return_dict)
        return output
    
    def enable_tiling(self, *args, **kwargs):
        """启用 VAE tiling 以减少内存使用"""
        return self._vae.enable_tiling(*args, **kwargs)
    
    def disable_tiling(self):
        """禁用 VAE tiling"""
        return self._vae.disable_tiling()


# === Pipeline 设置 ===

def setup_pipeline_for_jax(pipe, model_id=MODEL_NAME):
    """
    设置 Pipeline 以在 JAX 环境中运行
    
    将所有模型权重移动到 JAX 设备并编译关键组件
    """
    print("\n配置 Pipeline 以使用 JAX 和 Splash Attention...")

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

    # 保存原始 SDPA 实现
    global _ORIGINAL_SDPA
    _ORIGINAL_SDPA = torch.nn.functional.scaled_dot_product_attention
    print(f"- 保存原始 SDPA 实现: {_ORIGINAL_SDPA}")
    
    # 注册自定义的 Scaled Dot-Product Attention
    print(f"- 注册 Splash Attention（窗口大小: {WINDOW_SIZE}）...")
    print(f"  注意：非 XLA tensor 会自动回退到原始 SDPA（用于 text encoder）")
    custom_attention = functools.partial(
        scaled_dot_product_attention,
        env=env,
        window_size=WINDOW_SIZE
    )
    
    op_to_override = torch.nn.functional.scaled_dot_product_attention
    env._ops[op_to_override] = ops_registry.Operator(
        op_to_override,
        custom_attention,
        is_jax_function=False,
        is_user_defined=True,
        needs_env=False,
        is_view_op=False,
    )
    
    def _move_scheduler_to_jax(scheduler):
        print("- 将 scheduler 参数移动到 JAX 设备...")
        for k, v in scheduler.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(scheduler, k, v.to('jax'))

    def _move_module_to_xla(module, name="module"):
        """将模块的权重转换为 JAX Array"""
        print(f"- 将 {name} 移到 XLA...")
        with jax.default_device('cpu'):
            state_dict = module.state_dict()
            state_dict = env.to_xla(state_dict)
            module.load_state_dict(state_dict, assign=True)
    
    with env:
        # 移动 scheduler 参数
        _move_scheduler_to_jax(pipe.scheduler)
        
        # 对 Transformer 进行处理：先移到 XLA，再分片
        _move_module_to_xla(pipe.transformer, "Transformer")
        print("- 对 Transformer 进行权重分片...")
        transformer_weights = shard_weights_transformer(mesh, pipe.transformer.state_dict(), use_tp=USE_TP)
        pipe.transformer.load_state_dict(transformer_weights, assign=True, strict=False)
        torchax.interop.call_jax(jax.block_until_ready, transformer_weights)
        
        # Text Encoder (Qwen2.5-VL 和 T5) 说明：
        # HunyuanVideo-1.5 的 text encoder 内部会创建普通 torch.Tensor（position embeddings 等）。
        # 我们的 SDPA override 会自动检测输入类型：
        # - XLA tensor -> 使用 Splash Attention
        # - 普通 torch.Tensor -> 使用原始 SDPA
        # 这样 text encoder 可以正常工作，而 transformer 使用 Splash Attention。
        print("- Text Encoder (Qwen2.5-VL) 保持在 CPU 上（SDPA 自动回退）")
        print("- Text Encoder 2 (T5) 保持在 CPU 上（SDPA 自动回退）")
        
        # 对 VAE 进行处理 - 使用纯 Flax VAE 替代 PyTorch VAE
        print("- 创建 Flax VAE 替代 PyTorch VAE...")
        original_vae_config = pipe.vae.config  # 保存原始 config
        
        # 加载 Flax VAE
        flax_vae = load_flax_vae(mesh, original_vae_config, dtype=jnp.bfloat16)
        
        # 使用 FlaxVAEProxy 包装
        pipe.vae = FlaxVAEProxy(flax_vae, original_vae_config, env)
        print("  ✓ VAE 已替换为 Flax VAE（包装在 FlaxVAEProxy 中）")
        
        # 编译 Transformer（is_t2v 必须是静态参数）
        print("- 编译 Transformer...")
        pipe.transformer = torchax.compile(
            pipe.transformer,
            torchax.CompileOptions(
                jax_jit_kwargs={'static_argnames': ('return_dict', 'is_t2v')}
            )
        )
        
        # 不编译 Text Encoder - 它们在 CPU 上运行
        print("- Text Encoder 不编译（CPU 模式）")
        
        # Monkeypatch _execution_device 属性，确保 pipeline 内部创建的 tensor 在 JAX 设备上
        # 这是必要的，因为 text encoder 保留在 CPU 上，而 pipeline 默认使用第一个模块的设备
        print("- Monkeypatch _execution_device 返回 'jax' 设备...")
        def _jax_execution_device(self):
            return torch.device('jax')
        type(pipe)._execution_device = property(_jax_execution_device)
    
    print("Pipeline 配置完成")
    return pipe, env, mesh


# === 视频生成 ===

def precompute_all_prompt_embeds(pipe, prompt, negative_prompt="", device='cpu'):
    """
    预计算所有 prompt embeddings（正面和负面）
    
    必须在 setup_pipeline_for_jax 之前调用，因为 torchax 环境会覆盖 SDPA，
    导致 text encoder 中的普通 torch.Tensor 与 XLA tensor 混合错误。
    
    Args:
        pipe: HunyuanVideo15Pipeline (未经 JAX 配置)
        prompt: 正面提示词
        negative_prompt: 负面提示词（默认空字符串）
        device: 设备（默认 'cpu'）
        
    Returns:
        dict: 包含所有 prompt embeddings 的字典
    """
    print(f"\n预计算 prompt embeddings（{device} 模式，在 JAX 配置之前）...")
    
    # 计算正面 prompt embeddings
    print("  编码正面 prompt...")
    prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2 = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_videos_per_prompt=1,
    )
    print(f"    prompt_embeds shape: {prompt_embeds.shape}, dtype: {prompt_embeds.dtype}")
    print(f"    prompt_embeds_2 shape: {prompt_embeds_2.shape}")
    
    # 计算负面 prompt embeddings（用于 CFG）
    print("  编码负面 prompt...")
    neg_prompt_embeds, neg_prompt_embeds_mask, neg_prompt_embeds_2, neg_prompt_embeds_mask_2 = pipe.encode_prompt(
        prompt=negative_prompt if negative_prompt else "",
        device=device,
        num_videos_per_prompt=1,
    )
    print(f"    negative_prompt_embeds shape: {neg_prompt_embeds.shape}")
    print(f"    negative_prompt_embeds_2 shape: {neg_prompt_embeds_2.shape}")
    
    print("  ✓ 所有 prompt embeddings 已预计算")
    
    return {
        'prompt_embeds': prompt_embeds,
        'prompt_embeds_mask': prompt_embeds_mask,
        'prompt_embeds_2': prompt_embeds_2,
        'prompt_embeds_mask_2': prompt_embeds_mask_2,
        'negative_prompt_embeds': neg_prompt_embeds,
        'negative_prompt_embeds_mask': neg_prompt_embeds_mask,
        'negative_prompt_embeds_2': neg_prompt_embeds_2,
        'negative_prompt_embeds_mask_2': neg_prompt_embeds_mask_2,
    }


def run_generation_benchmark(pipe, prompt_embeds_dict, num_inference_steps=50, num_frames=121,
                             height=None, width=None, guidance_scale=6.0,
                             seed=42, num_iterations=2):
    """
    运行视频生成基准测试
    
    Args:
        pipe: HunyuanVideo15Pipeline
        prompt_embeds_dict: 预计算的 prompt embeddings 字典（已转换为 XLA tensor）
        num_inference_steps: 推理步数
        num_frames: 视频帧数
        height: 视频高度
        width: 视频宽度
        guidance_scale: 引导尺度
        seed: 随机种子
        num_iterations: 迭代次数
        debug: 是否启用调试输出
        
    Returns:
        frames: 最后生成的视频帧
        times: 各次迭代的时间列表
    """
    print(f"\n运行 {num_iterations} 次视频生成基准测试...")
    print(f"推理步数: {num_inference_steps}")
    print(f"视频帧数: {num_frames}")
    if height and width:
        print(f"分辨率: {height}x{width}")
    print(f"引导尺度: {guidance_scale}")
    print(f"随机种子: {seed}")
    
    times = []
    frames = None
    
    # 准备生成参数
    # 注意：传入所有预计算的 embeddings（正面和负面），pipeline 会跳过 text encoder
    gen_kwargs = {
        'prompt': None,  # 使用 None，让 pipeline 使用 prompt_embeds
        'negative_prompt': None,  # 使用 None，让 pipeline 使用 negative_prompt_embeds
        'prompt_embeds': prompt_embeds_dict['prompt_embeds'],
        'prompt_embeds_mask': prompt_embeds_dict['prompt_embeds_mask'],
        'prompt_embeds_2': prompt_embeds_dict['prompt_embeds_2'],
        'prompt_embeds_mask_2': prompt_embeds_dict['prompt_embeds_mask_2'],
        'negative_prompt_embeds': prompt_embeds_dict['negative_prompt_embeds'],
        'negative_prompt_embeds_mask': prompt_embeds_dict['negative_prompt_embeds_mask'],
        'negative_prompt_embeds_2': prompt_embeds_dict['negative_prompt_embeds_2'],
        'negative_prompt_embeds_mask_2': prompt_embeds_dict['negative_prompt_embeds_mask_2'],
        'num_frames': num_frames,
        'num_inference_steps': num_inference_steps,
        'output_type': 'pil',  # 始终输出 PIL 以避免格式问题
    }
    
    if height is not None:
        gen_kwargs['height'] = height
    if width is not None:
        gen_kwargs['width'] = width
    
    # 设置默认设备为 'jax'，这样 pipeline 内部创建的 tensor 也会在 JAX 设备上
    # 这对于 image_embeds, cond_latents_concat, mask_concat 等内部创建的 tensor 很重要
    original_default_device = None
    try:
        original_default_device = torch.get_default_device()
    except Exception:
        pass  # 可能没有设置默认设备
    
    torch.set_default_device('jax')
    
    for i in range(num_iterations):
        print(f"\n迭代 {i}:")
        
        generator = torch.Generator(device='cpu').manual_seed(seed)
        gen_kwargs['generator'] = generator
        
        start = time.perf_counter()
        result = pipe(**gen_kwargs)
        frames = result.frames[0]
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        
        print(f"  完成时间: {elapsed:.2f} 秒" + (" (包含 JIT 编译)" if i == 0 else ""))
        del result
    
    # 恢复原始默认设备
    if original_default_device is not None:
        torch.set_default_device(original_default_device)
    else:
        torch.set_default_device('cpu')
    
    return frames, times


def print_performance_summary(times):
    """打印性能统计"""
    if len(times) > 0:
        print(f"\n=== 性能统计 ===")
        print(f"总迭代次数: {len(times)}")
        print(f"第一次运行（含编译）: {times[0]:.4f} 秒")
        
        if len(times) > 1:
            avg_time = sum(times[1:]) / len(times[1:])
            print(f"后续运行平均时间: {avg_time:.4f} 秒")
            print(f"加速比: {times[0] / avg_time:.2f}x")
        
        print(f"\n各次迭代详细时间:")
        for i, t in enumerate(times):
            print(f"  迭代 {i}: {t:.4f} 秒")


# === 主函数 ===

def main():
    parser = argparse.ArgumentParser(description='Generate video using HunyuanVideo-1.5 with TPU/JAX')
    
    # 必需参数
    parser.add_argument(
        '--prompt', type=str, 
        default='A girl holding a paper with words "Hello, world!"',
        help='Text prompt for video generation'
    )
    
    # 模型配置
    parser.add_argument(
        '--model_id', type=str, 
        default=MODEL_NAME,
        help='Model ID or path'
    )
    
    # 生成参数
    parser.add_argument(
        '--num_frames', type=int, default=61,
        help='Number of frames to generate (default: 61)'
    )
    parser.add_argument(
        '--num_inference_steps', type=int, default=30,
        help='Number of inference steps (default: 30)'
    )
    parser.add_argument(
        '--guidance_scale', type=float, default=6.0,
        help='Guidance scale for classifier-free guidance (default: 6.0)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--height', type=int, default=None,
        help='Video height (default: None, uses model default)'
    )
    parser.add_argument(
        '--width', type=int, default=None,
        help='Video width (default: None, uses model default)'
    )
    
    parser.add_argument(
        '--disable_vae_tiling', action='store_true',
        help='Disable VAE tiling (tiling is enabled by default to save memory)'
    )

    # 基准测试
    parser.add_argument(
        '--num_iterations', type=int, default=2,
        help='Number of benchmark iterations (default: 2)'
    )
    
    # 输出选项
    parser.add_argument(
        '--output_path', type=str, default='output_flax.mp4',
        help='Output video file path (default: output_flax.mp4)'
    )
    parser.add_argument(
        '--fps', type=int, default=24,
        help='Output video FPS (default: 24)'
    )
    
    args = parser.parse_args()
    
    # 设置 JAX 配置
    jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

    warnings.filterwarnings('ignore', message='.*dtype.*int64.*truncated to dtype int32.*')
    logging.getLogger().setLevel(logging.ERROR)

    # 设置随机数种子
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    torch.set_default_dtype(torch.bfloat16)
 
    setup_pytree_registrations()
    
    # 加载 Pipeline
    print(f"\n加载模型: {args.model_id}")
    pipe = HunyuanVideo15Pipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
    )
    
    # 默认启用 VAE Tiling 以节省内存（除非显式禁用）
    if not args.disable_vae_tiling:
        print("启用 VAE Tiling（默认开启以节省 VMEM）...")
        pipe.vae.enable_tiling()
    else:
        print("VAE Tiling 已禁用")

    # 重要：在 setup_pipeline_for_jax 之前预计算所有 prompt embeddings！
    # 因为 setup_pipeline_for_jax 会覆盖 SDPA，导致 text encoder 中的
    # 普通 torch.Tensor 与 XLA tensor 混合错误。
    prompt_embeds_dict = precompute_all_prompt_embeds(pipe, args.prompt, negative_prompt="")
    
    # 配置 Pipeline 使用 JAX（包括 SDPA override）
    pipe, env, mesh = setup_pipeline_for_jax(pipe, args.model_id)
    
    # 将所有 prompt embeddings 转换为 XLA tensor
    print("\n- 将所有 prompt embeddings 转换为 XLA tensor...")
    with env:
        for key in prompt_embeds_dict:
            prompt_embeds_dict[key] = prompt_embeds_dict[key].to('jax')
    print("  ✓ 所有 prompt embeddings 已转换为 XLA tensor")
    
    # 禁用 CFG guider（如果需要）
    # HunyuanVideo15Pipeline 使用 guider 来控制 CFG
    if args.guidance_scale <= 1.0:
        if hasattr(pipe, 'guider') and pipe.guider is not None:
            print("禁用 CFG guider (guidance_scale <= 1.0)...")
            pipe.guider._enabled = False
    
    # 运行生成
    with mesh, env:
        frames, times = run_generation_benchmark(
            pipe,
            prompt_embeds_dict,
            num_inference_steps=args.num_inference_steps,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            num_iterations=args.num_iterations
        )
    
    # 保存视频
    print(f"\n保存生成的视频到: {args.output_path}")
    export_to_video(frames, args.output_path, fps=args.fps)
    print(f"✓ 视频已保存!")
    
    # 打印性能统计
    print_performance_summary(times)
    
    print("\n✓ 测试完成")


if __name__ == "__main__":
    main()