#!/usr/bin/env python3
"""
HunyuanVideo-1.5 三阶段生成 - 阶段2：Transformer (DiT) (TPU 版本)

基于 generate_diffusers_flax.py 的模式，使用 Splash Attention 在 TPU 上运行
使用 HunyuanVideo-1.5-TPU 原版 Transformer

用于 TPU v4/v5 环境
"""

import os
import sys
import time
import re
import math
import functools
import argparse
import warnings
import numpy as np
from types import SimpleNamespace

# 过滤掉各种无害警告
warnings.filterwarnings('ignore', message='.*jax.experimental.shard_map is deprecated.*')
warnings.filterwarnings('ignore', message='.*NumPy array is not writable.*')
# int64 截断警告来自 HunyuanVideo-1.5-TPU 库代码，无法修改
warnings.filterwarnings('ignore', message='.*int64.*is not available.*')
# 过滤 flash attention fallback 警告（我们用 Splash Attention 替代）
warnings.filterwarnings('ignore', message='.*Falling back from.*')

# JAX 和 torchax 导入
import jax
import jax.numpy as jnp
import torch
import torchax
from torchax.ops import ops_registry
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
from jax.experimental.pallas.ops.tpu import splash_attention
from jax.experimental.shard_map import shard_map
from jax.experimental import mesh_utils
from tqdm import tqdm

# 添加 diffusers-tpu 到路径（editable 安装可能在不同 Python 版本下）
DIFFUSERS_TPU_ROOT = os.path.expanduser("~/diffusers-tpu")
if os.path.exists(DIFFUSERS_TPU_ROOT) and DIFFUSERS_TPU_ROOT not in sys.path:
    sys.path.insert(0, DIFFUSERS_TPU_ROOT)

# 添加 HunyuanVideo-1.5-TPU 到路径
HUNYUAN_ROOT = os.path.expanduser("~/HunyuanVideo-1.5-TPU")
if HUNYUAN_ROOT not in sys.path:
    sys.path.insert(0, HUNYUAN_ROOT)

# Mock parallel state（TPU 使用 GSPMD，不需要 GPU 风格的 SP）
# 必须在导入 transformer 之前 mock
import hyvideo.commons.parallel_states as parallel_states_module
from types import SimpleNamespace

_mock_parallel_state = SimpleNamespace(
    sp=1,
    sp_enabled=False,
    sp_group=None,
)
parallel_states_module.get_parallel_state = lambda: _mock_parallel_state

# 导入 HunyuanVideo-1.5-TPU 的组件
from hyvideo.models.transformers.hunyuanvideo_1_5_transformer import HunyuanVideo_1_5_DiffusionTransformer
from hyvideo.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from hyvideo.commons import PIPELINE_CONFIGS
from hyvideo.utils.multitask_utils import merge_tensor_by_mask
import hyvideo.models.transformers.modules.attention as attention_module


# Monkey-patch reorder_txt_token 以避免布尔索引（torchax 不支持）
def _reorder_txt_token_tpu_compatible(self, byt5_txt, txt, byt5_text_mask, text_mask, zero_feat=False, is_reorder=True):
    """
    TPU 兼容版本的 reorder_txt_token
    
    使用 argsort + gather 替代布尔索引，实现相同的重排序逻辑：
    - 有效 byt5 tokens -> 有效 text tokens -> padding byt5 -> padding text
    
    Args:
        byt5_txt: [B, byt5_len, D] ByT5 text embeddings
        txt: [B, text_len, D] Text embeddings
        byt5_text_mask: [B, byt5_len] ByT5 mask (1=有效, 0=padding)
        text_mask: [B, text_len] Text mask (1=有效, 0=padding)
        zero_feat: 是否将 padding 位置的 token 置零
        is_reorder: 是否执行重排序（False 时只做简单拼接）
    
    Returns:
        reorder_txt: [B, byt5_len + text_len, D] 重排后的 text embeddings
        reorder_mask: [B, byt5_len + text_len] 重排后的 mask
    """
    B = byt5_txt.shape[0]
    byt5_len = byt5_txt.shape[1]
    text_len = txt.shape[1]
    
    # 拼接 byt5 和 text
    combined_txt = torch.concat([byt5_txt, txt], dim=1)
    combined_mask = torch.concat([byt5_text_mask, text_mask], dim=1)
    
    if not is_reorder:
        # 简单拼接，不重排序
        return combined_txt, combined_mask.to(dtype=torch.int32)
    
    # 创建分组标识：byt5=0, text=1
    group = torch.cat([
        torch.zeros(B, byt5_len, device=combined_mask.device, dtype=combined_mask.dtype),
        torch.ones(B, text_len, device=combined_mask.device, dtype=combined_mask.dtype)
    ], dim=1)
    
    # 计算排序优先级：
    # - priority = 2*(1-mask) + group
    # - 有效 byt5: 2*0 + 0 = 0
    # - 有效 text: 2*0 + 1 = 1
    # - padding byt5: 2*1 + 0 = 2
    # - padding text: 2*1 + 1 = 3
    # 排序后顺序：有效byt5 -> 有效text -> padding byt5 -> padding text
    priority = 2 * (1 - combined_mask) + group
    
    # 计算排序索引
    sort_indices = torch.argsort(priority, dim=1, stable=True)
    
    # 使用 gather 重排 tokens
    sort_indices_expanded = sort_indices.unsqueeze(-1).expand_as(combined_txt).to(torch.int32)
    reorder_txt = torch.gather(combined_txt, dim=1, index=sort_indices_expanded)
    
    # 重排 mask
    reorder_mask = torch.gather(combined_mask, dim=1, index=sort_indices.to(torch.int32))
    reorder_mask = reorder_mask.to(dtype=torch.int32)
    
    # 如果 zero_feat=True，把 padding 位置的 token 置零
    if zero_feat:
        reorder_txt = reorder_txt * reorder_mask.unsqueeze(-1).to(dtype=reorder_txt.dtype)
    
    return reorder_txt, reorder_mask

# 替换原始方法
HunyuanVideo_1_5_DiffusionTransformer.reorder_txt_token = _reorder_txt_token_tpu_compatible


# 全局引用：我们的自定义 SDPA 实现（在 main 中设置）
_CUSTOM_SDPA = None


# Monkey-patch parallel_attention 以避免 JIT 不兼容的断言和运行时检查
def _parallel_attention_tpu(q, k, v, img_q_len, img_kv_len,
                             attn_mode=None, text_mask=None,
                             attn_param=None,
                             block_idx=None):
    """
    TPU 兼容版本的 parallel_attention
    - 使用 Splash Attention + segment_ids 处理 padding
    - 移除断言检查（避免 JIT concretization 问题）
    
    Splash Attention 的 segment_ids 机制：
    - valid tokens (img + valid txt): segment_id = 0
    - padding tokens: segment_id = 1
    只有相同 segment_id 的 tokens 才能互相 attend
    """
    query, encoder_query = q
    key, encoder_key = k
    value, encoder_value = v
    
    # 合并 image 和 text tokens
    query = torch.cat([query, encoder_query], dim=1)
    key = torch.cat([key, encoder_key], dim=1)
    value = torch.cat([value, encoder_value], dim=1)
    
    # transpose q,k,v dim to fit scaled_dot_product_attention
    # 原始: B * L * H * D -> 目标: B * H * L * D
    query = query.transpose(1, 2)  # B * Head_num * length * dim
    key = key.transpose(1, 2)      # B * Head_num * length * dim
    value = value.transpose(1, 2)  # B * Head_num * length * dim
    
    # 计算 segment_ids
    # img tokens: segment_id = 0 (always valid)
    # valid txt tokens: segment_id = 0
    # padding txt tokens: segment_id = 1
    #
    # 重要：segment_ids 需要有 batch 维度 [B, total_len]
    # 因为 CFG 模式下 batch[0] (unconditional) 和 batch[1] (conditional)
    # 可能有不同的 padding pattern
    total_len = query.shape[2]
    batch_size = query.shape[0]
    txt_len = total_len - img_q_len  # = 1256 (固定 padding 长度)
    
    # 构造 segment_ids: [B, total_len]
    segment_ids = None
    if text_mask is not None:
        # text_mask: [B, txt_len], 1=valid, 0=padding
        # img 部分全为 0，形状 [B, img_q_len]
        img_segment = torch.zeros(batch_size, img_q_len, dtype=torch.int32, device=query.device)
        # txt 部分根据 text_mask 设置，padding 位置设为 1
        txt_segment = (1 - text_mask).to(torch.int32)
        # segment_ids: [B, total_len]
        segment_ids = torch.cat([img_segment, txt_segment], dim=1)
        
    
    # 直接调用我们的自定义 SDPA（而不是通过 F.scaled_dot_product_attention）
    hidden_states = _CUSTOM_SDPA(
        query, key, value, attn_mask=None, _segment_ids=segment_ids
    )
    
    # transpose back
    hidden_states = hidden_states.transpose(1, 2)  # B * L * H * D
    
    b, s, a, d = hidden_states.shape
    hidden_states = hidden_states.reshape(b, s, -1)
    
    return hidden_states


# 替换 parallel_attention 和 sequence_parallel_attention
attention_module.parallel_attention = _parallel_attention_tpu
attention_module.sequence_parallel_attention = _parallel_attention_tpu

# 本地工具模块
from utils import (
    load_embeddings_from_safetensors,
    save_latents_to_safetensors,
    load_generation_config,
    save_generation_config,
    get_default_paths,
)


# === Splash Attention 配置参数 ===
BQSIZE = 2048           # Query 块大小
BKVSIZE = 2048          # Key/Value 块大小
BKVCOMPUTESIZE = 1024   # Key/Value 计算块大小
WINDOW_SIZE = None      # 窗口大小（None 表示使用完整注意力）

# === Mesh 分片配置 ===
USE_DP = False          # 是否使用 data parallelism
SP_NUM = 1              # Spatial parallelism 数量
USE_TP = True           # 是否使用 Tensor Parallel 模式


# ============================================================================
# Splash Attention 实现
# 参考 generate_diffusers_flax.py
# ============================================================================

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
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = attn_weight.masked_fill(torch.isnan(attn_weight), 0.0)
    
    if dropout_p > 0:
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    
    return attn_weight @ value


def _tpu_splash_attention(query, key, value, mesh, scale=None, is_causal=False, window_size=None, segment_ids=None):
    """
    TPU Splash Attention 实现（纯 JAX 版本）
    参考 generate_diffusers_flax.py
    
    Args:
        query: [B, H, Q_len, D]
        key: [B, H, KV_len, D]
        value: [B, H, KV_len, D]
        mesh: JAX mesh
        scale: attention scale factor
        is_causal: 是否 causal mask
        window_size: local attention 窗口大小
        segment_ids: [B, seq_len] 用于区分 valid (0) 和 padding (1) tokens
                     现在支持 per-batch 的 padding pattern
    """
    num_heads = query.shape[1]
    seq_len = query.shape[2]

    def _attention_on_slices(q, k, v, seg_ids):
        """
        Args:
            q, k, v: [B, H, L, D]
            seg_ids: [B, L] per-batch segment_ids，或 None
        """
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

        def pad_segment_ids_1d(seg, multiple):
            """Padding 1D segment_ids，padding 部分设为 1（隔离）"""
            if seg is None:
                return None, 0
            length = seg.shape[0]
            pad_len = (multiple - length % multiple) % multiple
            if pad_len == 0:
                return seg, length
            # padding 部分设为 1，与 valid tokens (0) 隔离
            return jnp.pad(seg, (0, pad_len), constant_values=1), length

        def kernel_3d(q_3d, k_3d, v_3d, seg_1d):
            """
            处理单个 batch 的 attention
            
            Args:
                q_3d, k_3d, v_3d: [H, L, D]
                seg_1d: [L] 单个 batch 的 segment_ids，或 None
            """
            num_heads_on_device = q_3d.shape[0]

            q_3d_padded, q_orig_len = pad_to_multiple(q_3d, BQSIZE, axis=1)
            k_3d_padded, k_orig_len = pad_to_multiple(k_3d, BKVSIZE, axis=1)
            v_3d_padded, v_orig_len = pad_to_multiple(v_3d, BKVSIZE, axis=1)

            padded_q_seq_len = q_3d_padded.shape[1]
            padded_kv_seq_len = k_3d_padded.shape[1]

            # 处理 segment_ids padding
            splash_segment_ids = None
            if seg_1d is not None:
                seg_padded, _ = pad_segment_ids_1d(seg_1d, max(BQSIZE, BKVSIZE))
                # 创建 Splash Attention 的 SegmentIds
                splash_segment_ids = splash_attention.SegmentIds(
                    q=seg_padded[:padded_q_seq_len],
                    kv=seg_padded[:padded_kv_seq_len]
                )

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
            out = splash_kernel(q_3d_padded, k_3d_padded, v_3d_padded, segment_ids=splash_segment_ids)
            
            return out[:, :q_orig_len, ...]

        # segment_ids 现在是 [B, L]，需要在 vmap 中也包含它
        if seg_ids is not None:
            # seg_ids: [B, L]，每个 batch 有独立的 segment_ids
            vmapped_kernel = jax.vmap(
                kernel_3d,
                in_axes=(0, 0, 0, 0), out_axes=0
            )
            return vmapped_kernel(q, k, v, seg_ids)
        else:
            # 没有 segment_ids
            vmapped_kernel = jax.vmap(
                lambda q, k, v: kernel_3d(q, k, v, None),
                in_axes=(0, 0, 0), out_axes=0
            )
            return vmapped_kernel(q, k, v)

    # 根据设备数量和头数确定分片策略
    if num_heads < mesh.size:
        q_partition_spec = P()
        kv_partition_spec = P()
        seg_partition_spec = P()
    else:
        if query.shape[2] == key.shape[2]:  # 自注意力
            q_partition_spec = P('dp', 'tp', 'sp', None)
            kv_partition_spec = P('dp', 'tp', None, None)
            # segment_ids: [B, L]，B 维度按 dp 分片
            seg_partition_spec = P('dp', None)
        else:  # 交叉注意力
            q_partition_spec = P('dp', None, ('tp', 'sp'), None)
            kv_partition_spec = P('dp', None, None, None)
            seg_partition_spec = P('dp', None)

    sharded_fn = shard_map(
        _attention_on_slices,
        mesh=mesh,
        in_specs=(q_partition_spec, kv_partition_spec, kv_partition_spec, seg_partition_spec),
        out_specs=q_partition_spec,
        check_rep=False,
    )
    out = sharded_fn(query, key, value, segment_ids)
    
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
    _segment_ids=None,  # 新增：用于 Splash Attention 的 segment_ids
) -> torch.Tensor:
    """
    Scaled Dot-Product Attention 封装函数
    
    路由策略：
    - 普通 torch.Tensor -> 原始 SDPA
    - XLA tensor + attn_mask -> 参考实现
    - XLA tensor + 无 mask -> TPU Splash Attention（支持 segment_ids）
    
    Args:
        _segment_ids: [seq_len] tensor，用于区分 valid (0) 和 padding (1) tokens
                      只有相同 segment_id 的 tokens 才能互相 attend
    """
    global _ORIGINAL_SDPA
    
    # 非 XLA tensor 使用原始 SDPA
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
        # 转换 segment_ids 到 JAX
        jsegment_ids = None
        if _segment_ids is not None:
            jsegment_ids = env.t2j_iso(_segment_ids)
        res = _tpu_splash_attention(
            jquery, jkey, jvalue, env._mesh,
            scale=scale, is_causal=is_causal, window_size=window_size,
            segment_ids=jsegment_ids
        )
        return env.j2t_iso(res)
    
    return _sdpa_reference(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)


# ============================================================================
# HunyuanVideo-1.5 Transformer 权重分片策略
# 适配 HunyuanVideo-1.5-TPU 原版 Transformer 的权重命名
# ============================================================================

# Tensor Parallel 模式（Megatron Column-Row 风格）
# PyTorch weight shape: (out_features, in_features)
# Column Parallel (Q/K/V, FF1): 在 out_features 维度分片 -> (('tp', 'sp'), None)
# Row Parallel (Proj, FF2): 在 in_features 维度分片 -> (None, ('tp', 'sp'))
transformer_shardings_tp = {
    # === MMDoubleStreamBlock 层 ===
    # Image attention - Column Parallel
    r'.*\.img_attn_q\.weight$': (('tp', 'sp'), None),
    r'.*\.img_attn_k\.weight$': (('tp', 'sp'), None),
    r'.*\.img_attn_v\.weight$': (('tp', 'sp'), None),
    # Image attention - Row Parallel
    r'.*\.img_attn_proj\.weight$': (None, ('tp', 'sp')),
    
    # Text attention - Column Parallel
    r'.*\.txt_attn_q\.weight$': (('tp', 'sp'), None),
    r'.*\.txt_attn_k\.weight$': (('tp', 'sp'), None),
    r'.*\.txt_attn_v\.weight$': (('tp', 'sp'), None),
    # Text attention - Row Parallel
    r'.*\.txt_attn_proj\.weight$': (None, ('tp', 'sp')),
    
    # Image MLP - Column Parallel
    r'.*\.img_mlp\.fc1\.weight$': (('tp', 'sp'), None),
    # Image MLP - Row Parallel
    r'.*\.img_mlp\.fc2\.weight$': (None, ('tp', 'sp')),
    
    # Text MLP - Column Parallel
    r'.*\.txt_mlp\.fc1\.weight$': (('tp', 'sp'), None),
    # Text MLP - Row Parallel
    r'.*\.txt_mlp\.fc2\.weight$': (None, ('tp', 'sp')),
    
    # === MMSingleStreamBlock 层 ===
    # Single stream - Column Parallel
    r'.*\.linear1_q\.weight$': (('tp', 'sp'), None),
    r'.*\.linear1_k\.weight$': (('tp', 'sp'), None),
    r'.*\.linear1_v\.weight$': (('tp', 'sp'), None),
    r'.*\.linear1_mlp\.weight$': (('tp', 'sp'), None),
    # Single stream - Row Parallel
    r'.*\.linear2\.linear\.weight$': (None, ('tp', 'sp')),
    
    # === 其他层 ===
    # txt_in (SingleTokenRefiner)
    r'.*\.txt_in\..*\.to_q\.weight$': (('tp', 'sp'), None),
    r'.*\.txt_in\..*\.to_k\.weight$': (('tp', 'sp'), None),
    r'.*\.txt_in\..*\.to_v\.weight$': (('tp', 'sp'), None),
    r'.*\.txt_in\..*\.to_out\.0\.weight$': (None, ('tp', 'sp')),
    
    # byt5_in (ByT5Mapper)
    r'.*\.byt5_in\.fc1\.weight$': (('tp', 'sp'), None),
    r'.*\.byt5_in\.fc3\.weight$': (('tp', 'sp'), None),
    r'.*\.byt5_in\.fc2\.weight$': (None, ('tp', 'sp')),
    r'.*\.byt5_in\.fc4\.weight$': (None, ('tp', 'sp')),
}

# FSDP 模式（均匀分片，与 TP 相反）
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


def shard_weights(mesh, weights, sharding_dict):
    """
    对模型权重进行分片
    
    Args:
        mesh: JAX 设备网格
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


# ============================================================================
# 辅助函数
# ============================================================================

def get_latent_size(video_length, height, width, vae_temporal_ratio=4, vae_spatial_ratio=16):
    """计算 latent 尺寸"""
    video_length = (video_length - 1) // vae_temporal_ratio + 1
    height = height // vae_spatial_ratio
    width = width // vae_spatial_ratio
    return video_length, height, width


def get_task_mask(task_type, latent_target_length):
    """获取任务 mask"""
    if task_type == "t2v":
        return torch.zeros(latent_target_length)
    elif task_type == "i2v":
        mask = torch.zeros(latent_target_length)
        mask[0] = 1.0
        return mask
    else:
        raise ValueError(f"{task_type} is not supported!")


def prepare_latents(batch_size, num_channels, latent_height, latent_width, video_length,
                   dtype, device, generator):
    """准备随机 latents"""
    shape = (batch_size, num_channels, video_length, latent_height, latent_width)
    latents = torch.randn(shape, generator=generator, device=torch.device('cpu'), dtype=dtype).to(device)
    return latents


def prepare_cond_latents(task_type, image_cond, latents, multitask_mask):
    """准备条件 latents"""
    if image_cond is not None and task_type == 'i2v':
        latents_concat = image_cond.repeat(1, 1, latents.shape[2], 1, 1)
        latents_concat[:, :, 1:, :, :] = 0.0
    else:
        latents_concat = torch.zeros_like(latents)
    
    mask_zeros = torch.zeros(latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4])
    mask_ones = torch.ones(latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4])
    mask_concat = merge_tensor_by_mask(mask_zeros.cpu(), mask_ones.cpu(), mask=multitask_mask.cpu(), dim=2).to(device=latents.device)
    
    return torch.concat([latents_concat, mask_concat], dim=1)


def get_closest_resolution(aspect_ratio, target_resolution):
    """根据宽高比获取最接近的分辨率"""
    from hyvideo.utils.data_utils import generate_crop_size_list, get_closest_ratio
    
    target_size_config = {
        "360p": {"bucket_hw_base_size": 480, "bucket_hw_bucket_stride": 16},
        "480p": {"bucket_hw_base_size": 640, "bucket_hw_bucket_stride": 16},
        "720p": {"bucket_hw_base_size": 960, "bucket_hw_bucket_stride": 16},
        "1080p": {"bucket_hw_base_size": 1440, "bucket_hw_bucket_stride": 16},
    }
    
    bucket_hw_base_size = target_size_config[target_resolution]["bucket_hw_base_size"]
    bucket_hw_bucket_stride = target_size_config[target_resolution]["bucket_hw_bucket_stride"]
    
    if ":" in aspect_ratio:
        w_ratio, h_ratio = map(int, aspect_ratio.split(":"))
    else:
        w_ratio, h_ratio = 16, 9
    
    crop_size_list = generate_crop_size_list(bucket_hw_base_size, bucket_hw_bucket_stride)
    aspect_ratios = np.array([round(float(h) / float(w), 5) for h, w in crop_size_list])
    closest_size, _ = get_closest_ratio(h_ratio, w_ratio, aspect_ratios, crop_size_list)
    
    return closest_size[0], closest_size[1]  # height, width


def print_rank0(msg):
    """只在 rank 0 上打印（TPU 版本简化为直接打印）"""
    print(msg)


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='HunyuanVideo-1.5 Stage 2: Transformer (TPU)')
    
    parser.add_argument('--input_dir', type=str, default='./stage_outputs')
    parser.add_argument('--output_dir', type=str, default=None)
    
    # 视频生成参数
    parser.add_argument('--aspect_ratio', type=str, default='16:9')
    parser.add_argument('--video_length', type=int, default=49)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=6.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warmup_steps', type=int, default=2,
                        help='预热步数（0=不预热，1=一次，2=两次，用于触发 JIT 编译）')
    
    args = parser.parse_args()
    
    # === 设置 JAX 配置 ===
    jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
    
    torch.set_default_dtype(torch.bfloat16)
    
    output_dir = args.output_dir or args.input_dir
    input_paths = get_default_paths(args.input_dir)
    output_paths = get_default_paths(output_dir)
    
    print_rank0(f"\n{'='*60}")
    print_rank0("HunyuanVideo-1.5 Stage 2: Transformer (TPU + Splash Attention)")
    print_rank0(f"{'='*60}")
    
    # === 加载 Stage 1 配置 ===
    print_rank0(f"\n加载 Stage 1 配置: {input_paths['config']}")
    config = load_generation_config(input_paths['config'])
    
    # 加载 Stage 1 embeddings
    print_rank0(f"加载 Stage 1 embeddings: {input_paths['embeddings']}")
    embeddings_dict, _ = load_embeddings_from_safetensors(input_paths['embeddings'], device='cpu')
    
    # 更新配置
    config['aspect_ratio'] = args.aspect_ratio
    config['video_length'] = args.video_length
    config['num_inference_steps'] = args.num_inference_steps
    config['guidance_scale'] = args.guidance_scale
    config['seed'] = args.seed
    
    model_path = config['model_path']
    transformer_version = config['transformer_version']
    resolution = config.get('resolution', '720p')
    task_type = config.get('task_type', 't2v')
    
    print_rank0(f"\n配置:")
    print_rank0(f"  model_path: {model_path}")
    print_rank0(f"  transformer_version: {transformer_version}")
    print_rank0(f"  resolution: {resolution}")
    print_rank0(f"  task_type: {task_type}")
    print_rank0(f"  aspect_ratio: {args.aspect_ratio}")
    print_rank0(f"  video_length: {args.video_length}")
    print_rank0(f"  num_inference_steps: {args.num_inference_steps}")
    print_rank0(f"  guidance_scale: {args.guidance_scale}")
    print_rank0(f"  seed: {args.seed}")
    print_rank0(f"  JAX 设备数: {jax.device_count()}")
    
    # === 创建 JAX Mesh ===
    print_rank0(f"\n创建 JAX Mesh...")
    tp_dim, dp_dim, sp_dim = jax.device_count(), 1, 1
    if USE_DP:
        tp_dim //= 2
        dp_dim = 2
    
    if SP_NUM > 1:
        tp_dim //= SP_NUM
        sp_dim = SP_NUM
    
    print_rank0(f"  Mesh 维度: tp_dim={tp_dim}, dp_dim={dp_dim}, sp_dim={sp_dim}")
    
    mesh_devices = mesh_utils.create_device_mesh((tp_dim, dp_dim, sp_dim), allow_split_physical_axes=True)
    mesh = Mesh(mesh_devices, ('tp', 'dp', 'sp'))
    
    # === 创建 torchax 环境 ===
    print_rank0(f"\n创建 torchax 环境...")
    env = torchax.default_env()
    env._mesh = mesh
    env.config.use_tpu_splash_attention = True
    
    # 保存原始 SDPA 实现
    global _ORIGINAL_SDPA
    _ORIGINAL_SDPA = torch.nn.functional.scaled_dot_product_attention
    
    # 注册自定义的 Scaled Dot-Product Attention（替代 flash/torch SDPA）
    print_rank0(f"- 注册 TPU Splash Attention（替代 flash/torch，窗口大小: {'全局' if WINDOW_SIZE is None else WINDOW_SIZE}）...")
    custom_attention = functools.partial(
        scaled_dot_product_attention,
        env=env,
        window_size=WINDOW_SIZE
    )
    
    # 设置全局变量，供 _parallel_attention_tpu 直接调用
    global _CUSTOM_SDPA
    _CUSTOM_SDPA = custom_attention
    
    op_to_override = torch.nn.functional.scaled_dot_product_attention
    env._ops[op_to_override] = ops_registry.Operator(
        op_to_override,
        custom_attention,
        is_jax_function=False,
        is_user_defined=True,
        needs_env=False,
        is_view_op=False,
    )
    
    # === 加载 Transformer ===
    print_rank0(f"\n加载 Transformer...")
    
    dtype = config.get('dtype', 'bf16')
    if dtype == 'bf16':
        transformer_dtype = torch.bfloat16
    else:
        transformer_dtype = torch.float32
    
    transformer_path = os.path.join(model_path, "transformer", transformer_version)
    print_rank0(f"  加载路径: {transformer_path}")
    
    transformer = HunyuanVideo_1_5_DiffusionTransformer.from_pretrained(
        transformer_path,
        torch_dtype=transformer_dtype,
        low_cpu_mem_usage=True,
        attn_mode='torch',  # 使用 torch SDPA，会被我们的 override 拦截
    )
    
    print_rank0(f"  ✓ Transformer 加载完成")
    print_rank0(f"  use_meanflow: {transformer.config.use_meanflow}")
    
    # === 转换权重到 XLA 并分片 ===
    print_rank0(f"\n转换权重到 XLA...")
    
    with env:
        with jax.default_device('cpu'):
            state_dict = transformer.state_dict()
            state_dict = env.to_xla(state_dict)
            transformer.load_state_dict(state_dict, assign=True)
        
        print_rank0("- 对 Transformer 进行权重分片...")
        transformer_weights = shard_weights_transformer(mesh, transformer.state_dict(), use_tp=USE_TP)
        transformer.load_state_dict(transformer_weights, assign=True, strict=False)
        torchax.interop.call_jax(jax.block_until_ready, transformer_weights)
    
    # 设置为评估模式
    transformer.eval()
    
    # === 预计算 Rotary Position Embeddings ===
    # get_rotary_pos_embed 包含动态 tensor 创建（torch.arange），不适合放在 JIT 里
    # 对于固定尺寸的视频，可以预计算并缓存
    print_rank0("\n预计算 Rotary Position Embeddings...")
    
    # 获取 latent 尺寸（需要先计算分辨率）
    height_temp, width_temp = get_closest_resolution(args.aspect_ratio, resolution)
    video_length_temp = args.video_length
    latent_target_length_temp, latent_height_temp, latent_width_temp = get_latent_size(video_length_temp, height_temp, width_temp)
    
    # 预计算 rotary embeddings（在 CPU 上计算，避免 torchax 问题）
    with torch.no_grad():
        freqs_cos, freqs_sin = transformer.get_rotary_pos_embed((latent_target_length_temp, latent_height_temp, latent_width_temp))
        # 转换到 XLA 设备并缓存
        with env:
            transformer._cached_freqs_cos = freqs_cos.to('jax')
            transformer._cached_freqs_sin = freqs_sin.to('jax')
    
    # Monkey-patch get_rotary_pos_embed 使用缓存
    original_get_rotary_pos_embed = transformer.get_rotary_pos_embed
    def cached_get_rotary_pos_embed(self, latent_size):
        if hasattr(self, '_cached_freqs_cos') and hasattr(self, '_cached_freqs_sin'):
            return self._cached_freqs_cos, self._cached_freqs_sin
        return original_get_rotary_pos_embed(latent_size)
    
    import types
    transformer.get_rotary_pos_embed = types.MethodType(cached_get_rotary_pos_embed, transformer)
    print_rank0(f"  ✓ Rotary Embeddings 已预计算并缓存 (latent size: {latent_target_length_temp}x{latent_height_temp}x{latent_width_temp})")
    
    # === 编译 Transformer ===
    print_rank0("\n编译 Transformer...")
    with env:
        transformer = torchax.compile(
            transformer,
            torchax.CompileOptions(
                jax_jit_kwargs={'static_argnames': ('return_dict', 'mask_type')}
            )
        )
    print_rank0("  ✓ Transformer 编译完成")
    
    # === 加载 Scheduler ===
    print_rank0(f"\n加载 Scheduler...")
    
    pipeline_config = PIPELINE_CONFIGS.get(transformer_version, PIPELINE_CONFIGS['720p_t2v'])
    flow_shift = pipeline_config['flow_shift']
    
    scheduler = FlowMatchDiscreteScheduler(
        shift=flow_shift,
        reverse=True,
        solver="euler",
    )
    print_rank0(f"  flow_shift: {flow_shift}")
    
    # === 设置参数 ===
    guidance_scale = args.guidance_scale
    seed = args.seed
    do_classifier_free_guidance = guidance_scale > 1.0
    use_meanflow = transformer.config.use_meanflow
    target_dtype = transformer_dtype
    
    # 计算分辨率
    height, width = get_closest_resolution(args.aspect_ratio, resolution)
    print_rank0(f"\n分辨率: {width}x{height}")
    
    # 设置随机种子
    generator = torch.Generator(device=torch.device('cpu')).manual_seed(seed)
    
    # 获取 latent 尺寸
    video_length = args.video_length
    latent_target_length, latent_height, latent_width = get_latent_size(video_length, height, width)
    n_tokens = latent_target_length * latent_height * latent_width
    
    print_rank0(f"Latent 尺寸: {latent_target_length}x{latent_height}x{latent_width}")
    print_rank0(f"Token 数量: {n_tokens}")
    
    # === 准备 embeddings ===
    print_rank0(f"\n准备 embeddings...")
    
    with env:
        # 将 embeddings 移动到 JAX 设备
        prompt_embeds = embeddings_dict['prompt_embeds'].to(dtype=transformer_dtype).to('jax')
        negative_prompt_embeds = embeddings_dict['negative_prompt_embeds'].to(dtype=transformer_dtype).to('jax')
        prompt_mask = embeddings_dict['prompt_embeds_mask'].to('jax')
        negative_prompt_mask = embeddings_dict['negative_prompt_embeds_mask'].to('jax')
        
        # ByT5 embeddings
        prompt_embeds_2 = embeddings_dict.get('prompt_embeds_2')
        negative_prompt_embeds_2 = embeddings_dict.get('negative_prompt_embeds_2')
        prompt_embeds_mask_2 = embeddings_dict.get('prompt_embeds_mask_2')
        negative_prompt_embeds_mask_2 = embeddings_dict.get('negative_prompt_embeds_mask_2')
        
        # 合并 CFG embeddings（LLM）
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            prompt_mask = torch.cat([negative_prompt_mask, prompt_mask])
        
        # 准备 byt5 embeddings（合并 CFG）
        # 使用 bf16 以获得最佳 TPU 性能（TPU bf16 原生优化，累加器为 float32）
        # 之前的质量问题是由 attention mask 导致的，现在已通过 K/V 置零方案修复
        extra_kwargs = {}
        if prompt_embeds_2 is not None:
            prompt_embeds_2 = prompt_embeds_2.to(dtype=transformer_dtype).to('jax')
            prompt_embeds_mask_2 = prompt_embeds_mask_2.to('jax')
            if do_classifier_free_guidance:
                negative_prompt_embeds_2 = negative_prompt_embeds_2.to(dtype=transformer_dtype).to('jax')
                negative_prompt_embeds_mask_2 = negative_prompt_embeds_mask_2.to('jax')
                byt5_text_states = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
                byt5_text_mask = torch.cat([negative_prompt_embeds_mask_2, prompt_embeds_mask_2])
            else:
                byt5_text_states = prompt_embeds_2
                byt5_text_mask = prompt_embeds_mask_2
            extra_kwargs = {
                "byt5_text_states": byt5_text_states,
                "byt5_text_mask": byt5_text_mask,
            }
        
        print_rank0(f"  prompt_embeds shape: {prompt_embeds.shape}")
        print_rank0(f"  prompt_mask shape: {prompt_mask.shape}")
        if extra_kwargs:
            print_rank0(f"  byt5_text_states shape: {extra_kwargs['byt5_text_states'].shape}")
        
        # prompt_embeds_2 设为 None（720p_t2v 不使用）
        prompt_embeds_2_for_transformer = None
        
        # 获取 multitask mask
        multitask_mask = get_task_mask(task_type, latent_target_length)
        
        # 准备 latents
        num_channels_latents = transformer.config.in_channels
        latents = prepare_latents(
            1,  # batch_size
            num_channels_latents,
            latent_height,
            latent_width,
            latent_target_length,
            target_dtype,
            'jax',  # device
            generator,
        )
        
        # 准备 cond_latents
        cond_latents = prepare_cond_latents(task_type, None, latents, multitask_mask)
        
        # 准备 vision_states
        # t2v 模式：设为 None 以跳过 vision_in 处理
        # 这避免了 torch.all(vision_states == 0) 在 JIT 中的 concretization 问题
        #
        # 注意：从 transformer 代码看，两种情况效果相同：
        # - vision_states = None → 跳过 vision_in 分支
        # - vision_states = 零向量 → extra_encoder_hidden_states *= 0，extra_attention_mask = 0
        # 两者都导致 vision tokens 不参与注意力计算
        if task_type == 't2v':
            vision_states = None
        else:
            # i2v 或其他模式需要实际的 vision_states
            vision_num_tokens = 729
            vision_dim = 1152
            
            vision_states = torch.zeros(
                latents.shape[0],
                vision_num_tokens,
                vision_dim,
                device='jax',
                dtype=target_dtype,
            )
            
            if do_classifier_free_guidance:
                vision_states = vision_states.repeat(2, 1, 1)
        
        print_rank0(f"  latents shape: {latents.shape}")
        print_rank0(f"  cond_latents shape: {cond_latents.shape}")
        print_rank0(f"  vision_states: {vision_states.shape if vision_states is not None else 'None (t2v mode)'}")
        
        # 设置 timesteps
        # 注意：scheduler 需要在 JAX 设备上
        # 先在 CPU 上设置，然后移动
        scheduler.set_timesteps(args.num_inference_steps, device='cpu', n_tokens=n_tokens)
        timesteps = scheduler.timesteps.to('jax')
    
    # === 定义统一的 Denoising 循环函数 ===
    def run_denoising_loop(
        latents_input,
        timesteps_input,
        num_steps,
        desc="Denoising",
        is_warmup=False,
    ):
        """
        统一的 Denoising 循环，预热和正式推理共用同一套代码。
        
        Args:
            latents_input: 输入 latents
            timesteps_input: timesteps 列表
            num_steps: 运行步数
            desc: 进度条描述
            is_warmup: 是否是预热模式（影响进度条显示）
        
        Returns:
            (latents, step_times, elapsed_time)
        """
        step_times = []
        start_time = time.perf_counter()
        
        with mesh, env:
            # clone 必须在 torchax 环境内执行
            loop_latents = latents_input.clone() if is_warmup else latents_input
            with torch.no_grad():
                progress_bar = tqdm(
                    range(num_steps),
                    total=num_steps,
                    desc=desc,
                    ncols=130,
                )
                
                for i in progress_bar:
                    step_start = time.perf_counter()
                    
                    # 获取当前 timestep
                    t = timesteps_input[i % len(timesteps_input)]
                    
                    # 准备输入
                    latents_concat = torch.concat([loop_latents, cond_latents], dim=1)
                    latent_model_input = torch.cat([latents_concat] * 2) if do_classifier_free_guidance else latents_concat
                    latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                    
                    t_expand = t.repeat(latent_model_input.shape[0])
                    
                    # Meanflow timestep_r
                    if use_meanflow:
                        if i == len(timesteps_input) - 1:
                            timesteps_r = torch.tensor([0.0], device='jax')
                        else:
                            next_idx = (i + 1) % len(timesteps_input)
                            timesteps_r = timesteps_input[next_idx]
                        timesteps_r = timesteps_r.repeat(latent_model_input.shape[0])
                    else:
                        timesteps_r = None
                    
                    # guidance（embedded guidance scale 为 None）
                    guidance_expand = None
                    
                    # Transformer forward
                    output = transformer(
                        latent_model_input,
                        t_expand,
                        prompt_embeds,
                        prompt_embeds_2_for_transformer,
                        prompt_mask,
                        timestep_r=timesteps_r,
                        vision_states=vision_states,
                        mask_type=task_type,
                        guidance=guidance_expand,
                        return_dict=False,
                        extra_kwargs=extra_kwargs,
                    )
                    noise_pred = output[0]
                    
                    # CFG
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                    # Scheduler step
                    loop_latents = scheduler.step(noise_pred, t, loop_latents, generator=generator, return_dict=False)[0]
                    loop_latents = loop_latents.to(target_dtype)
                    
                    # 等待计算完成（JAX/XLA 是惰性执行的，必须显式等待才能准确计时）
                    torchax.interop.call_jax(jax.block_until_ready, loop_latents._elem)
                    
                    # 更新进度条
                    step_time = time.perf_counter() - step_start
                    step_times.append(step_time)
                    avg_time = sum(step_times) / len(step_times)
                    
                    if is_warmup:
                        progress_bar.set_postfix({
                            'step': f'{step_time:.2f}s',
                        })
                    else:
                        remaining_steps = num_steps - i - 1
                        progress_bar.set_postfix({
                            'step': f'{step_time:.2f}s',
                            'avg': f'{avg_time:.2f}s',
                            'eta': f'{avg_time * remaining_steps:.1f}s'
                        })
        
        elapsed = time.perf_counter() - start_time
        return loop_latents, step_times, elapsed
    
    # === Warmup (可选) ===
    num_inference_steps = len(timesteps)
    
    if args.warmup_steps > 0:
        print_rank0(f"\n预热中（{args.warmup_steps}步，触发 JIT 编译）...")
        
        _, warmup_times, warmup_elapsed = run_denoising_loop(
            latents_input=latents,
            timesteps_input=timesteps,
            num_steps=args.warmup_steps,
            desc="Warmup (JIT)",
            is_warmup=True,
        )
        
        print_rank0(f"  ✓ 预热完成，耗时: {warmup_elapsed:.2f}秒")
    
    # === Denoising Loop ===
    print_rank0(f"\n开始 Transformer 推理...")
    print_rank0(f"  使用 Meanflow: {use_meanflow}")
    print_rank0(f"  使用 CFG: {do_classifier_free_guidance}")
    
    latents, step_times, elapsed = run_denoising_loop(
        latents_input=latents,
        timesteps_input=timesteps,
        num_steps=num_inference_steps,
        desc="Denoising (TPU)",
        is_warmup=False,
    )
    
    print_rank0(f"\n✓ Transformer 推理完成，耗时: {elapsed:.2f} 秒")
    print_rank0(f"  Latents shape: {latents.shape}")
    
    # === 保存 latents ===
    print_rank0(f"\n保存 latents 到: {output_paths['latents']}")
    
    # 将 latents 从 JAX 设备移动到 CPU
    with env:
        latents_cpu = latents.cpu()
    
    metadata = {
        'height': str(height),
        'width': str(width),
        'video_length': str(video_length),
        'num_inference_steps': str(args.num_inference_steps),
        'guidance_scale': str(guidance_scale),
        'seed': str(seed),
        'elapsed_time': str(elapsed),
        'device': 'tpu',
    }
    save_latents_to_safetensors(latents_cpu, output_paths['latents'], metadata)
    
    # 更新配置
    config['height'] = height
    config['width'] = width
    config['stage2_elapsed_time'] = elapsed
    save_generation_config(config, output_paths['config'])
    
    print_rank0(f"\n{'='*60}")
    print_rank0("Stage 2 完成！")
    print_rank0(f"{'='*60}")
    print_rank0(f"输出: {output_paths['latents']}")
    print_rank0(f"下一步: 运行 stage3_vae_decoder.py")
    
    print_rank0("\n✓ Stage 2 执行完成")
    
    # 强制退出（避免 torchax/JAX 后台线程阻塞）
    # sys.exit(0) 可能会被阻塞，使用 os._exit(0) 强制退出
    os._exit(0)


if __name__ == "__main__":
    main()