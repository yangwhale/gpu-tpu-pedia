"""
TPU Splash Attention 工具模块

为 Torchax 提供 TPU 优化的 Scaled Dot-Product Attention 实现。

关键优化：使用 exp2 代替 exp，更好利用 TPU VPU 硬件。
调用者需要将 query 预乘以 LOG2_E = 1.44269504。
"""

import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

# ============================================================================
# 常量
# ============================================================================

# Splash Attention 块大小
BQSIZE = 3328
BKVSIZE = 2816
BKVCOMPUTESIZE = 256
BKVCOMPUTEINSIZE = 256

# Pallas kernel 常量
DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)
NUM_SUBLANES = 8
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))

# LOG2_E 用于 exp2 优化
LOG2_E = 1.44269504


# ============================================================================
# 辅助类和函数
# ============================================================================

@dataclasses.dataclass(frozen=True, slots=True)
class _BlockSizes:
    block_q: int
    block_kv: int
    block_kv_compute: int | None = None

    def __post_init__(self):
        if self.block_kv_compute is None:
            object.__setattr__(self, "block_kv_compute", self.block_kv)


def _pad_to_multiple(x, multiple, axis):
    """将张量 padding 到指定倍数。"""
    seq_len = x.shape[axis]
    pad_len = (multiple - seq_len % multiple) % multiple
    if pad_len == 0:
        return x, seq_len
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, pad_len)
    return jnp.pad(x, pad_width), seq_len


# ============================================================================
# Pallas Flash Attention Kernel (exp2 优化)
# ============================================================================

def _flash_attention_kernel(
    q_ref, k_ref, v_ref,
    m_scratch_ref, l_scratch_ref, o_scratch_ref, o_ref,
    *, mask_value: float, grid_width: int,
    bq: int, bkv: int, bkv_compute: int, bkv_compute_in: int, head_dim_v: int,
):
    """Flash attention kernel with exp2 optimization.
    
    Note: Query should be pre-multiplied by LOG2_E in the caller.
    """
    float32 = jnp.float32
    head_dim_v_repeats, rem = divmod(head_dim_v, NUM_SUBLANES)
    if rem != 0:
        raise NotImplementedError(f"{head_dim_v=} should be a multiple of {NUM_SUBLANES}")

    h, i, j = pl.program_id(0), pl.program_id(1), pl.program_id(2)

    @pl.when(j == 0)
    def init():
        o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)
        m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)
        l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)

    def body(kv_compute_index, _):
        slice_k = pl.ds(kv_compute_index * bkv_compute, bkv_compute)
        m_prev, l_prev = m_scratch_ref[...], l_scratch_ref[...]

        q = q_ref[...]
        k = k_ref[slice_k, :]
        qk = lax.dot_general(k, q, NT_DIM_NUMBERS, preferred_element_type=float32)

        o_prev = o_scratch_ref[:]
        v = v_ref[slice_k, :].astype(float32)
        step = bkv_compute_in
        
        for idx in range(0, qk.shape[0], step):
            m_curr = qk[idx:idx+step].max(axis=0)[None, :]
            m_next = jnp.maximum(m_prev, m_curr)

            # Use exp2 for TPU VPU optimization (query pre-multiplied by LOG2_E)
            s_curr = jnp.exp2(qk[idx:idx+step] - m_next[0:1])

            l_curr = s_curr.sum(axis=0, keepdims=True)
            alpha = jnp.exp2(m_prev - m_next)
            l_next = l_curr + alpha * l_prev

            sv_dims = (((0,), (0,)), ((), ()))
            o_curr = lax.dot_general(v[idx:idx+step], s_curr, sv_dims)
            alpha_o = alpha[0:1, ...]
            o_prev = alpha_o * o_prev + o_curr

            m_prev = m_next
            l_prev = l_next

        m_scratch_ref[...], l_scratch_ref[...] = m_next, l_next
        o_scratch_ref[:] = o_prev

    lax.fori_loop(0, (bkv // bkv_compute), body, None, unroll=True)

    @pl.when(j == grid_width - 1)
    def end():
        l = l_scratch_ref[...]
        l_inv = pltpu.repeat(1.0 / l, head_dim_v_repeats, axis=0)
        o_ref[...] = (o_scratch_ref[...] * l_inv).astype(o_ref.dtype)


def _splash_attention_forward(
    q: jax.Array, k: jax.Array, v: jax.Array,
    block_sizes: _BlockSizes, bkv_compute_in: int, interpret: bool = False,
):
    """Forward pass of splash attention with exp2 optimization."""
    num_q_heads, q_seq_len, head_dim_qk = q.shape
    head_dim_v = v.shape[-1]
    bq, bkv = block_sizes.block_q, block_sizes.block_kv
    bkv_compute = block_sizes.block_kv_compute
    num_kv_heads = k.shape[0]
    kv_seq_len = k.shape[1]
    q_heads_per_kv_head = num_q_heads // num_kv_heads

    def q_index_map(h, i, j, *_):
        return (h, i, 0)

    def out_index_map(h, i, j, *_):
        return h, 0, i

    def k_index_map(h, i, j, *_):
        return (h // q_heads_per_kv_head, j, 0)

    def v_index_map(h, i, j, *_):
        return (h // q_heads_per_kv_head, j, 0)

    in_specs = [
        pl.BlockSpec((None, bq, head_dim_qk), q_index_map),
        pl.BlockSpec((None, bkv, head_dim_qk), k_index_map),
        pl.BlockSpec((None, bkv, head_dim_v), v_index_map),
    ]
    out_shapes = [
        jax.ShapeDtypeStruct((NUM_SUBLANES, bq), jnp.float32),
        jax.ShapeDtypeStruct((NUM_SUBLANES, bq), jnp.float32),
        jax.ShapeDtypeStruct((head_dim_v, bq), jnp.float32),
        jax.ShapeDtypeStruct((num_q_heads, head_dim_v, q_seq_len), q.dtype),
    ]
    out_specs = [
        pl.BlockSpec((NUM_SUBLANES, bq), lambda *_: (0, 0)),
        pl.BlockSpec((NUM_SUBLANES, bq), lambda *_: (0, 0)),
        pl.BlockSpec((head_dim_v, bq), lambda *_: (0, 0)),
        pl.BlockSpec((None, head_dim_v, bq), out_index_map),
    ]
    
    grid_width = kv_seq_len // bkv
    grid = (num_q_heads, q_seq_len // bq, grid_width)

    all_out = pl.pallas_call(
        functools.partial(
            _flash_attention_kernel,
            mask_value=DEFAULT_MASK_VALUE,
            grid_width=grid_width,
            bq=bq, bkv=bkv, bkv_compute=bkv_compute,
            bkv_compute_in=bkv_compute_in, head_dim_v=head_dim_v,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=in_specs,
            out_specs=out_specs,
            grid=grid,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary"),
            flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": True}
        ),
        out_shape=out_shapes,
        interpret=interpret,
    )(q, k, v)
    return all_out[-1]


def _make_splash_mha(block_sizes: _BlockSizes, bkv_compute_in: int, interpret: bool = False):
    """Create a splash attention function with given block sizes.
    
    Note: Query should be pre-multiplied by LOG2_E (1.44269504) for exp2 optimization.
    """
    def _splash_attention(q: jax.Array, k: jax.Array, v: jax.Array):
        return _splash_attention_forward(q, k, v, block_sizes, bkv_compute_in, interpret)
    return _splash_attention


# ============================================================================
# SDPA 参考实现
# ============================================================================

def sdpa_reference(query, key, value, attn_mask=None, dropout_p=0.0,
                   is_causal=False, scale=None, enable_gqa=False):
    """SDPA 参考实现（用于短序列 cross-attention）。"""
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    
    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    
    if dropout_p > 0:
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    
    return attn_weight @ value


# ============================================================================
# TPU Splash Attention
# ============================================================================

def tpu_splash_attention(query, key, value, mesh, scale=None):
    """TPU Splash Attention（使用 exp2 优化）。
    
    Args:
        query: JAX array of shape (batch, heads, seq_len, head_dim)
        key: JAX array of shape (batch, heads, seq_len, head_dim)
        value: JAX array of shape (batch, heads, seq_len, head_dim)
        mesh: JAX mesh for sharding
        scale: Optional attention scale factor
        
    Returns:
        Output tensor of same shape as query
    """
    def _attention_kernel(q, k, v):
        scale_factor = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
        # 融合 softmax 中的 exp 操作 - 乘以 log2(e)
        q = q * scale_factor * LOG2_E

        def kernel_3d(q_3d, k_3d, v_3d):
            # Padding 到块大小的倍数
            q_3d_padded, q_orig_len = _pad_to_multiple(q_3d, BQSIZE, axis=1)
            k_3d_padded, _ = _pad_to_multiple(k_3d, BKVSIZE, axis=1)
            v_3d_padded, _ = _pad_to_multiple(v_3d, BKVSIZE, axis=1)
            
            block_sizes = _BlockSizes(
                block_q=min(BQSIZE, q_3d_padded.shape[1]),
                block_kv=min(BKVSIZE, k_3d_padded.shape[1]),
                block_kv_compute=min(BKVCOMPUTESIZE, k_3d_padded.shape[1]),
            )
            splash_kernel = _make_splash_mha(block_sizes=block_sizes, bkv_compute_in=BKVCOMPUTEINSIZE)
            out = splash_kernel(q_3d_padded, k_3d_padded, v_3d_padded).astype(q_3d.dtype)
            out = jnp.swapaxes(out, 1, 2)
            return out[:, :q_orig_len, :]

        return jax.vmap(kernel_3d, in_axes=(0, 0, 0), out_axes=0)(q, k, v)

    # 确定分片策略
    dp_mesh_key = "dp" if key.shape[0] > 1 else None
    remain_mesh_key = ("tp",) if key.shape[0] > 1 else ("dp", "tp")
    
    remain_devices = 1
    for d in remain_mesh_key:
        remain_devices *= mesh.axis_sizes[mesh.axis_names.index(d)]

    q_seq_len = query.shape[2]
    kv_seq_len = key.shape[2]
    
    # 长 KV 序列（self-attention）使用 head parallel
    if (kv_seq_len > 10000 and 
        key.shape[1] % remain_devices == 0 and 
        query.shape[1] % remain_devices == 0):
        q_spec = P(dp_mesh_key, remain_mesh_key, None, None)
        kv_spec = P(dp_mesh_key, remain_mesh_key, None, None)
    else:
        # 短 KV 序列（cross-attention）使用 sequence parallel
        if q_seq_len % remain_devices != 0:
            query, _ = _pad_to_multiple(query, remain_devices, axis=2)
        q_spec = P(dp_mesh_key, None, remain_mesh_key, None)
        kv_spec = P(dp_mesh_key, None, None, None)

    # 应用分片
    sharded_fn = shard_map(
        _attention_kernel, mesh=mesh,
        in_specs=(q_spec, kv_spec, kv_spec), out_specs=q_spec, check_rep=False,
    )
    
    constraint = P(dp_mesh_key, None, remain_mesh_key, None)
    query = jax.lax.with_sharding_constraint(query, constraint)
    key = jax.lax.with_sharding_constraint(key, constraint)
    value = jax.lax.with_sharding_constraint(value, constraint)
    
    out = sharded_fn(query, key, value)
    out = out[:, :, :q_seq_len, :]  # 移除 padding
    return jax.lax.with_sharding_constraint(out, constraint)
