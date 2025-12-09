#!/usr/bin/env python3
"""
HunyuanVideo-1.5 三阶段生成 - 阶段2：Transformer (DiT) (TPU 版本 + DeepCache)

基于 stage2_transformer_flax.py，添加 DeepCache 加速支持。

DeepCache 原理：
- HunyuanVideo 有 20 个 double_blocks + 40 个 single_blocks
- DeepCache 缓存 double_blocks 的输出，在某些步骤跳过 double_blocks 计算
- 理论加速比: 61/41 ≈ 1.49x

与 stage2_transformer_flax.py 的主要差异：
1. 新增 TPUDeepCache 类和 FullForwardModule/CachedForwardModule
2. 新增 --enable_cache 等命令行参数
3. run_denoising_loop 支持 DeepCache 分支

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
warnings.filterwarnings('ignore', message='.*int64.*is not available.*')
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

# 添加依赖库到路径
DIFFUSERS_TPU_ROOT = os.path.expanduser("~/diffusers-tpu")
if os.path.exists(DIFFUSERS_TPU_ROOT) and DIFFUSERS_TPU_ROOT not in sys.path:
    sys.path.insert(0, DIFFUSERS_TPU_ROOT)

HUNYUAN_ROOT = os.path.expanduser("~/HunyuanVideo-1.5-TPU")
if HUNYUAN_ROOT not in sys.path:
    sys.path.insert(0, HUNYUAN_ROOT)

# Mock parallel state（TPU 使用 GSPMD）
import hyvideo.commons.parallel_states as parallel_states_module
_mock_parallel_state = SimpleNamespace(sp=1, sp_enabled=False, sp_group=None)
parallel_states_module.get_parallel_state = lambda: _mock_parallel_state

# 导入 HunyuanVideo-1.5-TPU 的组件
from hyvideo.models.transformers.hunyuanvideo_1_5_transformer import HunyuanVideo_1_5_DiffusionTransformer
from hyvideo.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from hyvideo.commons import PIPELINE_CONFIGS
from hyvideo.utils.multitask_utils import merge_tensor_by_mask
import hyvideo.models.transformers.modules.attention as attention_module

# 本地工具模块
from utils import (
    load_embeddings_from_safetensors,
    save_latents_to_safetensors,
    load_generation_config,
    save_generation_config,
    get_default_paths,
)


# ============================================================================
# DeepCache 实现 (TPU 友好版本)
# ============================================================================

class TPUDeepCache:
    """
    TPU 友好的 DeepCache 实现
    
    原理：
    1. 缓存 double_blocks 的输出 (img, txt)
    2. 在 "填充缓存" 步骤：执行完整 transformer，保存中间状态
    3. 在 "使用缓存" 步骤：跳过 double_blocks，复用缓存
    
    参数说明：
    - cache_start_step: 开始使用缓存的步数（前期不缓存保证质量）
    - cache_end_step: 停止使用缓存的步数（后期不缓存保证收敛）
    - cache_step_interval: 每隔几步刷新一次缓存
    """
    
    def __init__(self, cache_start_step, cache_end_step, cache_step_interval, total_steps):
        self.cache_start_step = cache_start_step
        self.cache_end_step = cache_end_step
        self.cache_step_interval = cache_step_interval
        self.total_steps = total_steps
        
        # 计算需要完整计算的步骤（no_cache_steps）
        self.no_cache_steps = set(
            list(range(0, cache_start_step)) +  # 前期
            list(range(cache_start_step, cache_end_step, cache_step_interval)) +  # 刷新缓存
            list(range(cache_end_step, total_steps))  # 后期
        )
        
        # 缓存状态
        self.cached_img = None
        self.cached_txt = None
        self._cached_vec = None
        self._cached_text_mask = None
        
        # 统计
        self.cache_hit_count = 0
        self.cache_miss_count = 0
    
    def should_use_cache(self, step):
        """判断是否使用缓存"""
        return step not in self.no_cache_steps and self.cached_img is not None
    
    def update_cache(self, img, txt, vec, text_mask):
        """更新缓存"""
        self.cached_img = img
        self.cached_txt = txt
        self._cached_vec = vec
        self._cached_text_mask = text_mask
        self.cache_miss_count += 1
    
    def get_cache(self):
        """获取缓存"""
        self.cache_hit_count += 1
        return self.cached_img, self.cached_txt, self._cached_vec, self._cached_text_mask
    
    def clear(self):
        """清除缓存"""
        self.cached_img = None
        self.cached_txt = None
        self._cached_vec = None
        self._cached_text_mask = None
        self.cache_hit_count = 0
        self.cache_miss_count = 0
    
    def get_stats(self):
        """获取统计"""
        total = max(1, self.cache_hit_count + self.cache_miss_count)
        return {
            'cache_hit': self.cache_hit_count,
            'cache_miss': self.cache_miss_count,
            'hit_rate': self.cache_hit_count / total,
        }


class FullForwardModule(torch.nn.Module):
    """
    完整 transformer forward 的封装
    
    返回: (output, img_after_double, txt_after_double, vec, text_mask)
    额外返回 double_blocks 后的中间状态用于缓存
    """
    
    def __init__(self, transformer, mask_type, extra_kwargs):
        super().__init__()
        self.transformer = transformer
        self.mask_type = mask_type
        self.extra_kwargs = extra_kwargs
    
    def forward(self, hidden_states, timestep, text_states, text_states_2,
                encoder_attention_mask, timestep_r, vision_states, guidance,
                freqs_cos, freqs_sin):
        transformer = self.transformer
        extra_kwargs = self.extra_kwargs
        
        if guidance is None:
            guidance = torch.tensor([6016.0], device=hidden_states.device, dtype=torch.bfloat16)
        
        # === 输入处理 ===
        img = x = hidden_states
        text_mask = encoder_attention_mask
        txt = text_states
        
        bs, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // transformer.patch_size[0],
            oh // transformer.patch_size[1],
            ow // transformer.patch_size[2],
        )
        transformer.attn_param['thw'] = [tt, th, tw]
        
        if freqs_cos is None and freqs_sin is None:
            freqs_cos, freqs_sin = transformer.get_rotary_pos_embed((tt, th, tw))
        
        img = transformer.img_in(img)
        
        # === Modulation vectors ===
        vec = transformer.time_in(timestep)
        if text_states_2 is not None:
            vec = vec + transformer.vector_in(text_states_2)
        if transformer.guidance_embed:
            vec = vec + transformer.guidance_in(guidance)
        if timestep_r is not None:
            vec = vec + transformer.time_r_in(timestep_r)
        
        # === Text embedding ===
        if transformer.text_projection == "linear":
            txt = transformer.txt_in(txt)
        elif transformer.text_projection == "single_refiner":
            txt = transformer.txt_in(txt, timestep, text_mask if transformer.use_attention_mask else None)
        else:
            raise NotImplementedError(f"Unsupported text_projection: {transformer.text_projection}")
        
        if transformer.cond_type_embedding is not None:
            cond_emb = transformer.cond_type_embedding(
                torch.zeros_like(txt[:, :, 0], device=text_mask.device, dtype=torch.long)
            )
            txt = txt + cond_emb
        
        # === ByT5 处理 ===
        if transformer.glyph_byT5_v2:
            byt5_txt = transformer.byt5_in(extra_kwargs["byt5_text_states"])
            if transformer.cond_type_embedding is not None:
                cond_emb = transformer.cond_type_embedding(
                    torch.ones_like(byt5_txt[:, :, 0], device=byt5_txt.device, dtype=torch.long)
                )
                byt5_txt = byt5_txt + cond_emb
            txt, text_mask = transformer.reorder_txt_token(
                byt5_txt, txt, extra_kwargs["byt5_text_mask"], text_mask, zero_feat=True
            )
        
        # === Vision 处理 ===
        if transformer.vision_in is not None and vision_states is not None:
            extra_encoder_hidden_states = transformer.vision_in(vision_states)
            if self.mask_type == "t2v" and torch.all(vision_states == 0):
                extra_attention_mask = torch.zeros((bs, extra_encoder_hidden_states.shape[1]),
                                                   dtype=text_mask.dtype, device=text_mask.device)
                extra_encoder_hidden_states = extra_encoder_hidden_states * 0.0
            else:
                extra_attention_mask = torch.ones((bs, extra_encoder_hidden_states.shape[1]),
                                                  dtype=text_mask.dtype, device=text_mask.device)
            if transformer.cond_type_embedding is not None:
                cond_emb = transformer.cond_type_embedding(
                    2 * torch.ones_like(extra_encoder_hidden_states[:, :, 0], dtype=torch.long,
                                        device=extra_encoder_hidden_states.device)
                )
                extra_encoder_hidden_states = extra_encoder_hidden_states + cond_emb
            txt, text_mask = transformer.reorder_txt_token(
                extra_encoder_hidden_states, txt, extra_attention_mask, text_mask
            )
        
        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        
        # === Double blocks ===
        for index, block in enumerate(transformer.double_blocks):
            force_full_attn = (
                transformer.attn_mode in ["flex-block-attn"]
                and transformer.attn_param["win_type"] == "hybrid"
                and transformer.attn_param["win_ratio"] > 0
                and ((index + 1) % transformer.attn_param["win_ratio"] == 0
                     or (index + 1) == len(transformer.double_blocks))
            )
            transformer.attn_param["layer-name"] = f"double_block_{index+1}"
            img, txt = block(img=img, txt=txt, vec=vec, freqs_cis=freqs_cis,
                            text_mask=text_mask, attn_param=transformer.attn_param,
                            is_flash=force_full_attn, block_idx=index)
        
        # 保存中间状态（缓存点）
        img_after_double = img
        txt_after_double = txt
        
        # === Single blocks ===
        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]
        x = torch.cat((img, txt), 1)
        
        for index, block in enumerate(transformer.single_blocks):
            force_full_attn = (
                transformer.attn_mode in ["flex-block-attn"]
                and transformer.attn_param["win_type"] == "hybrid"
                and transformer.attn_param["win_ratio"] > 0
                and ((index + 1) % transformer.attn_param["win_ratio"] == 0
                     or (index + 1) == len(transformer.single_blocks))
            )
            transformer.attn_param["layer-name"] = f"single_block_{index+1}"
            x = block(x=x, vec=vec, txt_len=txt_seq_len, freqs_cis=(freqs_cos, freqs_sin),
                     text_mask=text_mask, attn_param=transformer.attn_param, is_flash=force_full_attn)
        
        img = x[:, :img_seq_len, ...]
        
        # === Final layer ===
        img = transformer.final_layer(img, vec)
        img = transformer.unpatchify(img, tt, th, tw)
        
        return (img, img_after_double, txt_after_double, vec, text_mask)


class CachedForwardModule(torch.nn.Module):
    """
    使用缓存的 forward 封装
    跳过 double_blocks，只执行 single_blocks + final_layer
    """
    
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
    
    def forward(self, cached_img, cached_txt, vec, freqs_cos, freqs_sin, text_mask):
        transformer = self.transformer
        
        tt, th, tw = transformer.attn_param['thw']
        txt_seq_len = cached_txt.shape[1]
        img_seq_len = cached_img.shape[1]
        
        x = torch.cat((cached_img, cached_txt), 1)
        
        # === Single blocks ===
        for index, block in enumerate(transformer.single_blocks):
            force_full_attn = (
                transformer.attn_mode in ["flex-block-attn"]
                and transformer.attn_param["win_type"] == "hybrid"
                and transformer.attn_param["win_ratio"] > 0
                and ((index + 1) % transformer.attn_param["win_ratio"] == 0
                     or (index + 1) == len(transformer.single_blocks))
            )
            transformer.attn_param["layer-name"] = f"single_block_{index+1}"
            x = block(x=x, vec=vec, txt_len=txt_seq_len, freqs_cis=(freqs_cos, freqs_sin),
                     text_mask=text_mask, attn_param=transformer.attn_param, is_flash=force_full_attn)
        
        img = x[:, :img_seq_len, ...]
        
        # === Final layer ===
        img = transformer.final_layer(img, vec)
        img = transformer.unpatchify(img, tt, th, tw)
        
        return img


def create_deepcache_modules(transformer, mask_type, extra_kwargs):
    """创建 DeepCache 所需的两个模块"""
    full_module = FullForwardModule(transformer, mask_type, extra_kwargs)
    cached_module = CachedForwardModule(transformer)
    return full_module, cached_module


# ============================================================================
# TPU 兼容性 Monkey-patches
# ============================================================================

def _reorder_txt_token_tpu_compatible(self, byt5_txt, txt, byt5_text_mask, text_mask, zero_feat=False, is_reorder=True):
    """TPU 兼容版本，避免布尔索引"""
    reorder_txt = torch.concat([byt5_txt, txt], dim=1)
    reorder_mask = torch.concat([byt5_text_mask, text_mask], dim=1).to(dtype=torch.int32)
    return reorder_txt, reorder_mask

HunyuanVideo_1_5_DiffusionTransformer.reorder_txt_token = _reorder_txt_token_tpu_compatible


def _parallel_attention_tpu(q, k, v, img_q_len, img_kv_len, attn_mode=None, text_mask=None, attn_param=None, block_idx=None):
    """TPU 兼容版本的 parallel_attention"""
    import torch.nn.functional as F
    
    query, encoder_query = q
    key, encoder_key = k
    value, encoder_value = v
    
    # 将 padding 位置的 K/V 设为零
    if text_mask is not None:
        text_mask_expanded = text_mask.unsqueeze(-1).unsqueeze(-1).to(encoder_key.dtype)
        encoder_key = encoder_key * text_mask_expanded
        encoder_value = encoder_value * text_mask_expanded
    
    query = torch.cat([query, encoder_query], dim=1).transpose(1, 2)
    key = torch.cat([key, encoder_key], dim=1).transpose(1, 2)
    value = torch.cat([value, encoder_value], dim=1).transpose(1, 2)
    
    hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=None)
    hidden_states = hidden_states.transpose(1, 2)
    
    b, s, a, d = hidden_states.shape
    return hidden_states.reshape(b, s, -1)

attention_module.parallel_attention = _parallel_attention_tpu
attention_module.sequence_parallel_attention = _parallel_attention_tpu


# ============================================================================
# Splash Attention 实现
# ============================================================================

BQSIZE = 2048
BKVSIZE = 2048
BKVCOMPUTESIZE = 1024
WINDOW_SIZE = None
USE_DP = False
SP_NUM = 1
USE_TP = True

_ORIGINAL_SDPA = None


def _is_xla_tensor(tensor):
    if tensor is None:
        return False
    if hasattr(tensor, '_elem'):
        return True
    if hasattr(tensor, 'device'):
        device_str = str(tensor.device)
        if 'jax' in device_str or 'xla' in device_str:
            return True
    return False


def _sdpa_reference(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    
    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)
    
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    
    if is_causal:
        causal_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_weight = attn_weight.masked_fill(causal_mask.logical_not(), float("-inf"))
    
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_weight = attn_weight.masked_fill(attn_mask.logical_not(), float("-inf"))
        else:
            attn_weight = attn_weight + attn_mask
    
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = attn_weight.masked_fill(torch.isnan(attn_weight), 0.0)
    
    if dropout_p > 0:
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    
    return attn_weight @ value


def _tpu_splash_attention(query, key, value, mesh, scale=None, is_causal=False, window_size=None):
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
            k_3d_padded, _ = pad_to_multiple(k_3d, BKVSIZE, axis=1)
            v_3d_padded, _ = pad_to_multiple(v_3d, BKVSIZE, axis=1)

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

    if num_heads < mesh.size:
        q_partition_spec = P()
        kv_partition_spec = P()
    else:
        if query.shape[2] == key.shape[2]:
            q_partition_spec = P('dp', 'tp', 'sp', None)
            kv_partition_spec = P('dp', 'tp', None, None)
        else:
            q_partition_spec = P('dp', None, ('tp', 'sp'), None)
            kv_partition_spec = P('dp', None, None, None)

    sharded_fn = shard_map(
        _attention_on_slices,
        mesh=mesh,
        in_specs=(q_partition_spec, kv_partition_spec, kv_partition_spec),
        out_specs=q_partition_spec,
        check_rep=False,
    )
    return sharded_fn(query, key, value)


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                                  scale=None, enable_gqa=False, env=None, window_size=None):
    global _ORIGINAL_SDPA
    
    if not _is_xla_tensor(query):
        if _ORIGINAL_SDPA is not None:
            return _ORIGINAL_SDPA(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)
        return _sdpa_reference(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)
    
    if attn_mask is not None:
        return _sdpa_reference(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)
    
    if env is not None and hasattr(env.config, 'use_tpu_splash_attention') and env.config.use_tpu_splash_attention:
        jquery, jkey, jvalue = env.t2j_iso((query, key, value))
        res = _tpu_splash_attention(jquery, jkey, jvalue, env._mesh, scale=scale, is_causal=is_causal, window_size=window_size)
        return env.j2t_iso(res)
    
    return _sdpa_reference(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)


# ============================================================================
# 权重分片策略
# ============================================================================

transformer_shardings_tp = {
    r'.*\.img_attn_q\.weight$': (('tp', 'sp'), None),
    r'.*\.img_attn_k\.weight$': (('tp', 'sp'), None),
    r'.*\.img_attn_v\.weight$': (('tp', 'sp'), None),
    r'.*\.img_attn_proj\.weight$': (None, ('tp', 'sp')),
    r'.*\.txt_attn_q\.weight$': (('tp', 'sp'), None),
    r'.*\.txt_attn_k\.weight$': (('tp', 'sp'), None),
    r'.*\.txt_attn_v\.weight$': (('tp', 'sp'), None),
    r'.*\.txt_attn_proj\.weight$': (None, ('tp', 'sp')),
    r'.*\.img_mlp\.fc1\.weight$': (('tp', 'sp'), None),
    r'.*\.img_mlp\.fc2\.weight$': (None, ('tp', 'sp')),
    r'.*\.txt_mlp\.fc1\.weight$': (('tp', 'sp'), None),
    r'.*\.txt_mlp\.fc2\.weight$': (None, ('tp', 'sp')),
    r'.*\.linear1_q\.weight$': (('tp', 'sp'), None),
    r'.*\.linear1_k\.weight$': (('tp', 'sp'), None),
    r'.*\.linear1_v\.weight$': (('tp', 'sp'), None),
    r'.*\.linear1_mlp\.weight$': (('tp', 'sp'), None),
    r'.*\.linear2\.linear\.weight$': (None, ('tp', 'sp')),
    r'.*\.txt_in\..*\.to_q\.weight$': (('tp', 'sp'), None),
    r'.*\.txt_in\..*\.to_k\.weight$': (('tp', 'sp'), None),
    r'.*\.txt_in\..*\.to_v\.weight$': (('tp', 'sp'), None),
    r'.*\.txt_in\..*\.to_out\.0\.weight$': (None, ('tp', 'sp')),
    r'.*\.byt5_in\.fc1\.weight$': (('tp', 'sp'), None),
    r'.*\.byt5_in\.fc3\.weight$': (('tp', 'sp'), None),
    r'.*\.byt5_in\.fc2\.weight$': (None, ('tp', 'sp')),
    r'.*\.byt5_in\.fc4\.weight$': (None, ('tp', 'sp')),
}


def shard_weights(mesh, weights, sharding_dict):
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
    
    print(f"  权重分片完成: {matched_count} 个匹配, {unmatched_count} 个复制")
    return result


# ============================================================================
# 辅助函数
# ============================================================================

def get_latent_size(video_length, height, width, vae_temporal_ratio=4, vae_spatial_ratio=16):
    video_length = (video_length - 1) // vae_temporal_ratio + 1
    height = height // vae_spatial_ratio
    width = width // vae_spatial_ratio
    return video_length, height, width


def get_task_mask(task_type, latent_target_length):
    if task_type == "t2v":
        return torch.zeros(latent_target_length)
    elif task_type == "i2v":
        mask = torch.zeros(latent_target_length)
        mask[0] = 1.0
        return mask
    raise ValueError(f"{task_type} is not supported!")


def prepare_latents(batch_size, num_channels, latent_height, latent_width, video_length, dtype, device, generator):
    shape = (batch_size, num_channels, video_length, latent_height, latent_width)
    return torch.randn(shape, generator=generator, device=torch.device('cpu'), dtype=dtype).to(device)


def prepare_cond_latents(task_type, image_cond, latents, multitask_mask):
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
    from hyvideo.utils.data_utils import generate_crop_size_list, get_closest_ratio
    
    config = {
        "360p": {"bucket_hw_base_size": 480, "bucket_hw_bucket_stride": 16},
        "480p": {"bucket_hw_base_size": 640, "bucket_hw_bucket_stride": 16},
        "720p": {"bucket_hw_base_size": 960, "bucket_hw_bucket_stride": 16},
        "1080p": {"bucket_hw_base_size": 1440, "bucket_hw_bucket_stride": 16},
    }
    
    base_size = config[target_resolution]["bucket_hw_base_size"]
    stride = config[target_resolution]["bucket_hw_bucket_stride"]
    
    if ":" in aspect_ratio:
        w_ratio, h_ratio = map(int, aspect_ratio.split(":"))
    else:
        w_ratio, h_ratio = 16, 9
    
    crop_size_list = generate_crop_size_list(base_size, stride)
    aspect_ratios = np.array([round(float(h) / float(w), 5) for h, w in crop_size_list])
    closest_size, _ = get_closest_ratio(h_ratio, w_ratio, aspect_ratios, crop_size_list)
    return closest_size[0], closest_size[1]


def print_rank0(msg):
    print(msg)


def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='HunyuanVideo-1.5 Stage 2: Transformer (TPU + DeepCache)')
    
    # 基本参数
    parser.add_argument('--input_dir', type=str, default='./stage_outputs')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--aspect_ratio', type=str, default='16:9')
    parser.add_argument('--video_length', type=int, default=49)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=6.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warmup_steps', type=int, default=2)
    
    # DeepCache 参数
    parser.add_argument('--enable_cache', type=str_to_bool, nargs='?', const=True, default=False,
                        help='启用 DeepCache 加速')
    parser.add_argument('--cache_start_step', type=int, default=11,
                        help='开始使用缓存的步数')
    parser.add_argument('--cache_end_step', type=int, default=45,
                        help='停止使用缓存的步数')
    parser.add_argument('--cache_step_interval', type=int, default=4,
                        help='缓存刷新间隔')
    
    args = parser.parse_args()
    
    # === 打印配置 ===
    if args.enable_cache:
        print_rank0(f"\n[DeepCache 配置]")
        print_rank0(f"  cache_start_step: {args.cache_start_step}")
        print_rank0(f"  cache_end_step: {args.cache_end_step}")
        print_rank0(f"  cache_step_interval: {args.cache_step_interval}")
    
    # === JAX 配置 ===
    jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
    
    torch.set_default_dtype(torch.bfloat16)
    
    output_dir = args.output_dir or args.input_dir
    input_paths = get_default_paths(args.input_dir)
    output_paths = get_default_paths(output_dir)
    
    print_rank0(f"\n{'='*60}")
    print_rank0("HunyuanVideo-1.5 Stage 2: Transformer (TPU + DeepCache)")
    print_rank0(f"{'='*60}")
    
    # === 加载配置和 embeddings ===
    print_rank0(f"\n加载配置: {input_paths['config']}")
    config = load_generation_config(input_paths['config'])
    
    print_rank0(f"加载 embeddings: {input_paths['embeddings']}")
    embeddings_dict, _ = load_embeddings_from_safetensors(input_paths['embeddings'], device='cpu')
    
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
    print_rank0(f"  video_length: {args.video_length}")
    print_rank0(f"  num_inference_steps: {args.num_inference_steps}")
    print_rank0(f"  JAX 设备数: {jax.device_count()}")
    
    # === 创建 Mesh 和环境 ===
    tp_dim = jax.device_count()
    mesh_devices = mesh_utils.create_device_mesh((tp_dim, 1, 1), allow_split_physical_axes=True)
    mesh = Mesh(mesh_devices, ('tp', 'dp', 'sp'))
    
    env = torchax.default_env()
    env._mesh = mesh
    env.config.use_tpu_splash_attention = True
    
    global _ORIGINAL_SDPA
    _ORIGINAL_SDPA = torch.nn.functional.scaled_dot_product_attention
    
    custom_attention = functools.partial(scaled_dot_product_attention, env=env, window_size=WINDOW_SIZE)
    op_to_override = torch.nn.functional.scaled_dot_product_attention
    env._ops[op_to_override] = ops_registry.Operator(
        op_to_override, custom_attention, is_jax_function=False, is_user_defined=True, needs_env=False, is_view_op=False
    )
    
    # === 加载 Transformer ===
    print_rank0(f"\n加载 Transformer...")
    transformer_dtype = torch.bfloat16
    transformer_path = os.path.join(model_path, "transformer", transformer_version)
    
    transformer = HunyuanVideo_1_5_DiffusionTransformer.from_pretrained(
        transformer_path, torch_dtype=transformer_dtype, low_cpu_mem_usage=True, attn_mode='torch'
    )
    print_rank0(f"  ✓ Transformer 加载完成")
    
    # === 初始化 DeepCache ===
    deep_cache = None
    full_forward_fn = None
    cached_forward_fn = None
    
    if args.enable_cache:
        deep_cache = TPUDeepCache(
            cache_start_step=args.cache_start_step,
            cache_end_step=args.cache_end_step,
            cache_step_interval=args.cache_step_interval,
            total_steps=args.num_inference_steps,
        )
        print_rank0(f"\n[DeepCache 初始化]")
        print_rank0(f"  no_cache_steps: {len(deep_cache.no_cache_steps)} / {args.num_inference_steps}")
        print_rank0(f"  预计 cache 步数: {args.num_inference_steps - len(deep_cache.no_cache_steps)}")
    
    # === 转换权重 ===
    print_rank0(f"\n转换权重到 XLA...")
    with env:
        with jax.default_device('cpu'):
            state_dict = transformer.state_dict()
            state_dict = env.to_xla(state_dict)
            transformer.load_state_dict(state_dict, assign=True)
        
        print_rank0("- 权重分片...")
        transformer_weights = shard_weights(mesh, transformer.state_dict(), transformer_shardings_tp)
        transformer.load_state_dict(transformer_weights, assign=True, strict=False)
        torchax.interop.call_jax(jax.block_until_ready, transformer_weights)
    
    transformer.eval()
    
    # === 预计算 Rotary Embeddings ===
    print_rank0("\n预计算 Rotary Embeddings...")
    height, width = get_closest_resolution(args.aspect_ratio, resolution)
    latent_t, latent_h, latent_w = get_latent_size(args.video_length, height, width)
    
    with torch.no_grad():
        freqs_cos, freqs_sin = transformer.get_rotary_pos_embed((latent_t, latent_h, latent_w))
        with env:
            transformer._cached_freqs_cos = freqs_cos.to('jax')
            transformer._cached_freqs_sin = freqs_sin.to('jax')
    
    original_get_rotary = transformer.get_rotary_pos_embed
    def cached_get_rotary(self, latent_size):
        if hasattr(self, '_cached_freqs_cos'):
            return self._cached_freqs_cos, self._cached_freqs_sin
        return original_get_rotary(latent_size)
    
    import types
    transformer.get_rotary_pos_embed = types.MethodType(cached_get_rotary, transformer)
    
    # === 编译 Transformer ===
    print_rank0("\n编译 Transformer...")
    with env:
        if not args.enable_cache:
            transformer = torchax.compile(
                transformer,
                torchax.CompileOptions(jax_jit_kwargs={'static_argnames': ('return_dict', 'mask_type')})
            )
            print_rank0("  ✓ Transformer 编译完成")
        else:
            print_rank0("  DeepCache 模式：延迟编译")
    
    # === 加载 Scheduler ===
    pipeline_config = PIPELINE_CONFIGS.get(transformer_version, PIPELINE_CONFIGS['720p_t2v'])
    scheduler = FlowMatchDiscreteScheduler(shift=pipeline_config['flow_shift'], reverse=True, solver="euler")
    
    # === 设置参数 ===
    guidance_scale = args.guidance_scale
    seed = args.seed
    do_cfg = guidance_scale > 1.0
    use_meanflow = transformer.config.use_meanflow
    
    generator = torch.Generator(device=torch.device('cpu')).manual_seed(seed)
    latent_target_length, latent_height, latent_width = get_latent_size(args.video_length, height, width)
    n_tokens = latent_target_length * latent_height * latent_width
    
    print_rank0(f"\n分辨率: {width}x{height}, Latent: {latent_target_length}x{latent_height}x{latent_width}")
    
    # === 准备 embeddings ===
    with env:
        prompt_embeds = embeddings_dict['prompt_embeds'].to(dtype=transformer_dtype).to('jax')
        negative_prompt_embeds = embeddings_dict['negative_prompt_embeds'].to(dtype=transformer_dtype).to('jax')
        prompt_mask = embeddings_dict['prompt_embeds_mask'].to('jax')
        negative_prompt_mask = embeddings_dict['negative_prompt_embeds_mask'].to('jax')
        
        if do_cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            prompt_mask = torch.cat([negative_prompt_mask, prompt_mask])
        
        extra_kwargs = {}
        prompt_embeds_2 = embeddings_dict.get('prompt_embeds_2')
        if prompt_embeds_2 is not None:
            prompt_embeds_2 = prompt_embeds_2.to(dtype=transformer_dtype).to('jax')
            prompt_embeds_mask_2 = embeddings_dict.get('prompt_embeds_mask_2').to('jax')
            if do_cfg:
                negative_prompt_embeds_2 = embeddings_dict.get('negative_prompt_embeds_2').to(dtype=transformer_dtype).to('jax')
                negative_prompt_embeds_mask_2 = embeddings_dict.get('negative_prompt_embeds_mask_2').to('jax')
                byt5_text_states = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
                byt5_text_mask = torch.cat([negative_prompt_embeds_mask_2, prompt_embeds_mask_2])
            else:
                byt5_text_states = prompt_embeds_2
                byt5_text_mask = prompt_embeds_mask_2
            extra_kwargs = {"byt5_text_states": byt5_text_states, "byt5_text_mask": byt5_text_mask}
        
        prompt_embeds_2_for_transformer = None
        multitask_mask = get_task_mask(task_type, latent_target_length)
        
        latents = prepare_latents(1, transformer.config.in_channels, latent_height, latent_width,
                                   latent_target_length, transformer_dtype, 'jax', generator)
        cond_latents = prepare_cond_latents(task_type, None, latents, multitask_mask)
        vision_states = None if task_type == 't2v' else torch.zeros(latents.shape[0], 729, 1152, device='jax', dtype=transformer_dtype)
        
        if vision_states is not None and do_cfg:
            vision_states = vision_states.repeat(2, 1, 1)
        
        scheduler.set_timesteps(args.num_inference_steps, device='cpu', n_tokens=n_tokens)
        timesteps = scheduler.timesteps.to('jax')
        
        # === 创建 DeepCache 模块 ===
        if args.enable_cache:
            print_rank0(f"\n创建 DeepCache 模块...")
            full_forward_module, cached_forward_module = create_deepcache_modules(transformer, task_type, extra_kwargs)
            full_forward_module.eval()
            cached_forward_module.eval()
            
            print_rank0("- 编译 full_forward...")
            full_forward_fn = torchax.compile(full_forward_module)
            print_rank0("- 编译 cached_forward...")
            cached_forward_fn = torchax.compile(cached_forward_module)
            
            # 预热
            print_rank0("\n预热 DeepCache...")
            t_warmup = timesteps[0]
            latents_warmup = latents.clone()
            latents_concat_warmup = torch.concat([latents_warmup, cond_latents], dim=1)
            latent_model_input_warmup = torch.cat([latents_concat_warmup] * 2) if do_cfg else latents_concat_warmup
            latent_model_input_warmup = scheduler.scale_model_input(latent_model_input_warmup, t_warmup)
            t_expand_warmup = t_warmup.repeat(latent_model_input_warmup.shape[0])
            
            with torch.no_grad():
                result_warmup = full_forward_fn(
                    latent_model_input_warmup, t_expand_warmup, prompt_embeds,
                    prompt_embeds_2_for_transformer, prompt_mask, None, vision_states, None,
                    transformer._cached_freqs_cos, transformer._cached_freqs_sin
                )
                torchax.interop.call_jax(jax.block_until_ready, result_warmup[0]._elem)
                
                cached_result_warmup = cached_forward_fn(
                    result_warmup[1], result_warmup[2], result_warmup[3],
                    transformer._cached_freqs_cos, transformer._cached_freqs_sin, result_warmup[4]
                )
                torchax.interop.call_jax(jax.block_until_ready, cached_result_warmup._elem)
            
            print_rank0("  ✓ DeepCache 预热完成")
    
    # === Denoising 循环 ===
    def run_denoising_loop(latents_input, timesteps_input, num_steps, desc="Denoising", is_warmup=False):
        step_times = []
        start_time = time.perf_counter()
        cache_hits = 0
        cache_misses = 0
        
        with mesh, env:
            loop_latents = latents_input.clone() if is_warmup else latents_input
            
            if deep_cache is not None:
                deep_cache.clear()
            
            with torch.no_grad():
                progress_bar = tqdm(range(num_steps), total=num_steps, desc=desc, ncols=130)
                
                for i in progress_bar:
                    step_start = time.perf_counter()
                    t = timesteps_input[i % len(timesteps_input)]
                    
                    latents_concat = torch.concat([loop_latents, cond_latents], dim=1)
                    latent_model_input = torch.cat([latents_concat] * 2) if do_cfg else latents_concat
                    latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                    t_expand = t.repeat(latent_model_input.shape[0])
                    
                    timesteps_r = None
                    if use_meanflow:
                        if i == len(timesteps_input) - 1:
                            timesteps_r = torch.tensor([0.0], device='jax')
                        else:
                            timesteps_r = timesteps_input[(i + 1) % len(timesteps_input)]
                        timesteps_r = timesteps_r.repeat(latent_model_input.shape[0])
                    
                    # === Forward（DeepCache 分支）===
                    if deep_cache is not None and deep_cache.should_use_cache(i):
                        cache_hits += 1
                        cached_img, cached_txt, cached_vec, cached_mask = deep_cache.get_cache()
                        noise_pred = cached_forward_fn(
                            cached_img, cached_txt, cached_vec,
                            transformer._cached_freqs_cos, transformer._cached_freqs_sin, cached_mask
                        )
                    elif deep_cache is not None:
                        cache_misses += 1
                        result = full_forward_fn(
                            latent_model_input, t_expand, prompt_embeds,
                            prompt_embeds_2_for_transformer, prompt_mask, timesteps_r,
                            vision_states, None,
                            transformer._cached_freqs_cos, transformer._cached_freqs_sin
                        )
                        noise_pred = result[0]
                        deep_cache.update_cache(result[1], result[2], result[3], result[4])
                    else:
                        output = transformer(
                            latent_model_input, t_expand, prompt_embeds,
                            prompt_embeds_2_for_transformer, prompt_mask,
                            timestep_r=timesteps_r, vision_states=vision_states,
                            mask_type=task_type, guidance=None, return_dict=False,
                            extra_kwargs=extra_kwargs
                        )
                        noise_pred = output[0]
                    
                    # CFG
                    if do_cfg:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                    # Scheduler step
                    loop_latents = scheduler.step(noise_pred, t, loop_latents, generator=generator, return_dict=False)[0]
                    loop_latents = loop_latents.to(transformer_dtype)
                    
                    torchax.interop.call_jax(jax.block_until_ready, loop_latents._elem)
                    
                    step_time = time.perf_counter() - step_start
                    step_times.append(step_time)
                    avg_time = sum(step_times) / len(step_times)
                    
                    if not is_warmup:
                        cache_info = f'H:{cache_hits}/M:{cache_misses}' if deep_cache else ''
                        progress_bar.set_postfix({
                            'step': f'{step_time:.2f}s', 'avg': f'{avg_time:.2f}s',
                            'eta': f'{avg_time * (num_steps - i - 1):.1f}s', 'cache': cache_info
                        })
        
        elapsed = time.perf_counter() - start_time
        
        if deep_cache is not None and not is_warmup:
            stats = deep_cache.get_stats()
            print_rank0(f"\n[DeepCache 统计] 命中: {stats['cache_hit']}, 未命中: {stats['cache_miss']}, 命中率: {stats['hit_rate']:.1%}")
        
        return loop_latents, step_times, elapsed
    
    # === 执行 ===
    num_inference_steps = len(timesteps)
    
    if args.warmup_steps > 0:
        print_rank0(f"\n预热中（{args.warmup_steps}步）...")
        _, _, warmup_elapsed = run_denoising_loop(latents, timesteps, args.warmup_steps, "Warmup", True)
        print_rank0(f"  ✓ 预热完成: {warmup_elapsed:.2f}秒")
    
    print_rank0(f"\n开始推理...")
    latents, step_times, elapsed = run_denoising_loop(latents, timesteps, num_inference_steps, "Denoising (TPU)", False)
    
    print_rank0(f"\n✓ 推理完成: {elapsed:.2f} 秒")
    
    # === 保存 ===
    with env:
        latents_cpu = latents.cpu()
    
    metadata = {
        'height': str(height), 'width': str(width), 'video_length': str(args.video_length),
        'num_inference_steps': str(args.num_inference_steps), 'guidance_scale': str(guidance_scale),
        'seed': str(seed), 'elapsed_time': str(elapsed), 'device': 'tpu',
    }
    save_latents_to_safetensors(latents_cpu, output_paths['latents'], metadata)
    
    config['height'] = height
    config['width'] = width
    config['stage2_elapsed_time'] = elapsed
    save_generation_config(config, output_paths['config'])
    
    print_rank0(f"\n输出: {output_paths['latents']}")
    print_rank0("✓ Stage 2 完成")
    
    os._exit(0)


if __name__ == "__main__":
    main()