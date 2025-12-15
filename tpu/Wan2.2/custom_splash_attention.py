"""
Custom Splash Attention for TPU

This module provides an optimized attention implementation using exp2 instead of exp
for better TPU VPU hardware utilization.

Key optimization: The caller must pre-multiply query by LOG2_E = 1.44269504
before calling these kernels, allowing direct use of exp2 instructions.
"""

import functools
import math
import dataclasses
import enum
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding, PartitionSpec as P

partial = functools.partial
DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)
NUM_LANES = 128
NUM_SUBLANES = 8
NN_DIM_NUMBERS = (((1,), (0,)), ((), ()))
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))


class _QKVLayout(enum.IntEnum):
    HEAD_DIM_MINOR = enum.auto()
    SEQ_MINOR = enum.auto()


def _from_head_minor(vals: tuple[Any, ...], layout: _QKVLayout):
    if layout == _QKVLayout.HEAD_DIM_MINOR:
        return vals
    return (*vals[:-2], vals[-1], vals[-2])


@dataclasses.dataclass(frozen=True, slots=True)
class _BlockSizes:
    block_q: int
    block_kv: int
    block_kv_compute: int | None = None
    q_layout: _QKVLayout = _QKVLayout.HEAD_DIM_MINOR
    k_layout: _QKVLayout = _QKVLayout.HEAD_DIM_MINOR
    v_layout: _QKVLayout = _QKVLayout.HEAD_DIM_MINOR

    def __post_init__(self):
        if self.block_kv_compute is None:
            object.__setattr__(self, "block_kv_compute", self.block_kv)


def _flash_attention_kernel(
    q_ref,
    k_ref,
    v_ref,
    m_scratch_ref,
    l_scratch_ref,
    o_scratch_ref,
    o_ref,
    *,
    mask_value: float,
    grid_width: int,
    bq: int,
    bkv: int,
    bkv_compute: int,
    bkv_compute_in: int,
    head_dim_v: int,
):
    """Flash attention kernel with exp2 optimization.
    
    Note: Query should be pre-multiplied by LOG2_E in the caller for exp2 optimization.
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
        assert m_prev.shape == (NUM_SUBLANES, bq)
        assert l_prev.shape == (NUM_SUBLANES, bq)

        q = q_ref[...]
        k = k_ref[slice_k, :]
        qk = lax.dot_general(k, q, NT_DIM_NUMBERS, preferred_element_type=float32)
        assert qk.shape == (bkv_compute, bq)

        o_prev = o_scratch_ref[:]
        v = v_ref[slice_k, :].astype(float32)
        step = bkv_compute_in
        assert qk.shape[0] % step == 0
        
        for i in range(0, qk.shape[0], step):
            m_curr = qk[i:i+step].max(axis=0)[None, :]
            assert m_curr.shape == (1, bq)
            
            m_next = jnp.maximum(m_prev, m_curr)
            assert m_next.shape == (NUM_SUBLANES, bq)

            # Use exp2 for TPU VPU optimization (query pre-multiplied by LOG2_E)
            s_curr = jnp.exp2(qk[i:i+step] - m_next[0:1])

            l_curr = s_curr.sum(axis=0, keepdims=True)
            assert l_curr.shape == (1, bq)

            alpha = jnp.exp2(m_prev - m_next)
            l_next = l_curr + alpha * l_prev

            sv_dims = (((0,), (0,)), ((), ()))
            o_curr = lax.dot_general(v[i:i+step], s_curr, sv_dims)
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
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    block_sizes: _BlockSizes,
    bkv_compute_in: int,
    interpret: bool = False,
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
        partial(
            _flash_attention_kernel,
            mask_value=DEFAULT_MASK_VALUE,
            grid_width=grid_width,
            bq=bq,
            bkv=bkv,
            bkv_compute=bkv_compute,
            bkv_compute_in=bkv_compute_in,
            head_dim_v=head_dim_v,
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


def make_splash_mha(
    block_sizes: _BlockSizes,
    bkv_compute_in: int,
    interpret: bool = False,
):
    """Create a splash attention function with given block sizes.
    
    Args:
        block_sizes: Block size configuration
        bkv_compute_in: Inner block size for KV computation
        interpret: Whether to use interpret mode for debugging
    
    Returns:
        A function that computes attention: (q, k, v) -> output
        
    Note:
        Query should be pre-multiplied by LOG2_E (1.44269504) for exp2 optimization.
    """
    def _splash_attention(q: jax.Array, k: jax.Array, v: jax.Array):
        return _splash_attention_forward(q, k, v, block_sizes, bkv_compute_in, interpret)
    return _splash_attention


# Default block sizes for standalone testing
BQSIZE = 1024
BKVSIZE = 1024
BKVCOMPUTESIZE = 256

sharded_fn = None


def _tpu_splash_attention(query, key, value, mesh):
    """Sharded TPU splash attention for standalone testing."""
    global sharded_fn
    num_heads = query.shape[1]

    def _attention_on_slices(q, k, v):
        scale_factor = 1.0 / math.sqrt(q.shape[-1])
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
            q_3d_padded, q_orig_len = pad_to_multiple(q_3d, BQSIZE, axis=1)
            k_3d_padded, k_orig_len = pad_to_multiple(k_3d, BKVSIZE, axis=1)
            v_3d_padded, v_orig_len = pad_to_multiple(v_3d, BKVSIZE, axis=1)
            padded_q_seq_len = q_3d_padded.shape[1]
            padded_kv_seq_len = k_3d_padded.shape[1]

            block_sizes = _BlockSizes(
                block_q=min(BQSIZE, padded_q_seq_len),
                block_kv=min(BKVSIZE, padded_kv_seq_len),
                block_kv_compute=min(BKVCOMPUTESIZE, padded_kv_seq_len),
            )
            splash_kernel = make_splash_mha(block_sizes=block_sizes, bkv_compute_in=BKVCOMPUTESIZE)
            out = splash_kernel(q_3d_padded, k_3d_padded, v_3d_padded)
            out = jnp.swapaxes(out, 1, 2)
            return out[:, :q_orig_len, ...]

        vmapped_kernel = jax.vmap(kernel_3d, in_axes=(0, 0, 0), out_axes=0)
        return vmapped_kernel(q, k, v)

    if sharded_fn is None:
        q_partition_spec = P('dp', 'axis', None, None)
        kv_partition_spec = P('dp', 'axis', None, None)
        sharded_fn = jax.jit(shard_map(
            _attention_on_slices,
            mesh=mesh,
            in_specs=(q_partition_spec, kv_partition_spec, kv_partition_spec),
            out_specs=q_partition_spec,
            check_rep=False,
        ))
    out = sharded_fn(query, key, value)
    return jax.lax.with_sharding_constraint(out, P('dp', None, 'axis', None))


if __name__ == "__main__":
    import time

    shape = (1, 40, 75600, 128)
    q = jnp.arange(np.prod(shape), dtype=jnp.bfloat16).reshape(*shape)
    k = jnp.arange(np.prod(shape), dtype=jnp.bfloat16).reshape(*shape)
    v = jnp.arange(np.prod(shape), dtype=jnp.bfloat16).reshape(*shape)

    mesh = jax.make_mesh((len(jax.devices()), 1, 1), ('axis', 'dp', 'sp'))
    q = jax.device_put(q, NamedSharding(mesh, P('dp', None, ('axis', 'sp'), None)))
    k = jax.device_put(k, NamedSharding(mesh, P('dp', None, ('axis', 'sp'), None)))
    v = jax.device_put(v, NamedSharding(mesh, P('dp', None, ('axis', 'sp'), None)))

    # Warmup
    with mesh:
        output = _tpu_splash_attention(q, k, v, mesh)
    output.block_until_ready()

    # Profile
    with mesh:
        with jax.profiler.trace("/dev/shm/tensorboard"):
            output = _tpu_splash_attention(q, k, v, mesh)
            output.block_until_ready()

    # Benchmark
    with mesh:
        num_time = 50
        start_time = time.time()
        for _ in range(num_time):
            output = _tpu_splash_attention(q, k, v, mesh)
        output.block_until_ready()
        end_time = time.time()
        print(f"Average time: {(end_time-start_time)/num_time:.4f}s")
