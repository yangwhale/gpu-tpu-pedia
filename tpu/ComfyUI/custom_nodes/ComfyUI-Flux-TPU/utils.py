"""
ComfyUI Flux.2 TPU - 工具模块
============================

包含:
  - 分片策略配置
  - 权重分片函数
  - PyTree 注册
  - JAX 配置
"""

import re

import jax
import numpy as np
import torch
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.tree_util import register_pytree_node


# ============================================================================
# Transformer 分片策略
# ============================================================================
# 规则：输出投影 ('tp', None)，输入投影 (None, 'tp')

TRANSFORMER_SHARDINGS = {
    # Double-stream Blocks - Attention
    r'transformer_blocks.*.attn.to_q.weight': ('tp', None),
    r'transformer_blocks.*.attn.to_k.weight': ('tp', None),
    r'transformer_blocks.*.attn.to_v.weight': ('tp', None),
    r'transformer_blocks.*.attn.to_out.0.weight': (None, 'tp'),
    r'transformer_blocks.*.attn.add_q_proj.weight': ('tp', None),
    r'transformer_blocks.*.attn.add_k_proj.weight': ('tp', None),
    r'transformer_blocks.*.attn.add_v_proj.weight': ('tp', None),
    r'transformer_blocks.*.attn.to_add_out.weight': (None, 'tp'),
    # Double-stream Blocks - FeedForward
    r'transformer_blocks.*.ff.linear_in.weight': ('tp', None),
    r'transformer_blocks.*.ff.linear_out.weight': (None, 'tp'),
    r'transformer_blocks.*.ff_context.linear_in.weight': ('tp', None),
    r'transformer_blocks.*.ff_context.linear_out.weight': (None, 'tp'),
    # Single-stream Blocks
    r'single_transformer_blocks.*.attn.to_qkv_mlp_proj.weight': ('tp', None),
    r'single_transformer_blocks.*.attn.to_out.weight': (None, 'tp'),
    # Embedders & Projections
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

# VAE 不分片（使用 replicate）
VAE_ENCODER_SHARDINGS = {}
VAE_DECODER_SHARDINGS = {}


# ============================================================================
# 权重分片函数
# ============================================================================

def shard_weight_dict(weight_dict, sharding_dict, mesh, debug=False):
    """
    按模式匹配应用权重分片。
    
    Args:
        weight_dict: 权重字典 {name: tensor}
        sharding_dict: 分片规则 {pattern: (axis0, axis1)}
        mesh: JAX Mesh
        debug: 是否打印详细信息
    
    Returns:
        分片后的权重字典
    """
    result = {}
    sharded_count = replicated_count = 0
    sharded_bytes = replicated_bytes = 0
    
    for k, v in weight_dict.items():
        tensor_bytes = np.prod(v.shape) * 2 if hasattr(v, 'shape') else 0
            
        if isinstance(v, torch.Tensor):
            with jax.default_device("cpu"):
                v = v.to("jax")
        
        matched = False
        for pattern, sharding in sharding_dict.items():
            if re.fullmatch(pattern, k) is not None:
                v.apply_jax_(jax.device_put, NamedSharding(mesh, P(*sharding)))
                matched = True
                sharded_count += 1
                sharded_bytes += tensor_bytes
                if debug:
                    print(f"  ✓ SHARDED: {k} -> {sharding}")
                break
        
        if not matched:
            v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
            replicated_count += 1
            replicated_bytes += tensor_bytes
        
        result[k] = v
    
    print(f"  分片统计: {sharded_count} 个分片 ({sharded_bytes/1e9:.2f}GB), "
          f"{replicated_count} 个复制 ({replicated_bytes/1e9:.2f}GB)")
    return result


def move_module_to_xla(env, module):
    """
    将 PyTorch 模块权重转换为 torchax tensor。
    
    Args:
        env: torchax 环境
        module: PyTorch 模块
    """
    with jax.default_device("cpu"):
        state_dict = module.state_dict()
        state_dict = env.to_xla(state_dict)
        module.load_state_dict(state_dict, assign=True)


# ============================================================================
# PyTree 注册
# ============================================================================

_pytree_registered = False


def setup_pytree_registrations():
    """
    注册必要的 PyTree 节点以支持 JAX 转换。
    
    注册的类型:
      - BaseModelOutputWithPastAndCrossAttentions (transformers)
      - DecoderOutput (diffusers VAE)
      - AutoencoderKLOutput (diffusers VAE)
    """
    global _pytree_registered
    if _pytree_registered:
        return
    
    from diffusers.models import modeling_outputs as diffusers_modeling_outputs
    from diffusers.models.autoencoders import vae as diffusers_vae
    from transformers import modeling_outputs
    
    print("注册 PyTree 节点...")
    
    def flatten(obj):
        return obj.to_tuple(), type(obj)
    
    def unflatten(aux, children):
        return aux(*children)
    
    classes = [
        (modeling_outputs.BaseModelOutputWithPastAndCrossAttentions, "BaseModelOutputWithPastAndCrossAttentions"),
        (diffusers_vae.DecoderOutput, "DecoderOutput"),
        (diffusers_modeling_outputs.AutoencoderKLOutput, "AutoencoderKLOutput"),
    ]
    
    for cls, name in classes:
        try:
            register_pytree_node(cls, flatten, unflatten)
            print(f"  - {name} 已注册")
        except ValueError:
            print(f"  - {name} 已存在")
    
    _pytree_registered = True


# ============================================================================
# JAX 配置
# ============================================================================

def setup_jax_cache():
    """设置 JAX 编译缓存以加速后续编译。"""
    import os
    cache_dir = os.path.expanduser("~/.cache/jax_cache")
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    print(f"✓ JAX 编译缓存: {cache_dir}")


# ============================================================================
# 自定义算子注册（备用）
# ============================================================================

def register_operators_on_env(env, mesh_obj):
    """
    在 torchax 环境上注册 TPU 所需的自定义算子。
    
    注册的算子:
      - conv2d: 2D 卷积
      - cartesian_prod: 笛卡尔积
      - chunk: 张量分块
      - layer_norm / native_layer_norm: 层归一化
      - unflatten: 维度展开
      - rms_norm: RMS 归一化
      - dropout / native_dropout: Dropout（推理时直接返回）
      - group_norm / native_group_norm: 组归一化
      - scaled_dot_product_attention: Splash Attention（可选）
    """
    import functools
    import jax.numpy as jnp
    from torchax.ops import jaten, ops_registry
    
    def override_op(op, impl):
        """注册或覆盖一个算子"""
        env._ops[op] = ops_registry.Operator(
            op, impl, is_jax_function=False, is_user_defined=True,
            needs_env=False, is_view_op=False,
        )
    
    # ---- conv2d ----
    def conv2d_impl(input, weight, bias=None, stride=1, padding=0,
                    dilation=1, groups=1, *, env=env):
        jinput, jweight, jbias = env.t2j_iso((input, weight, bias))
        res = jaten._aten_conv2d(jinput, jweight, jbias, stride, padding, dilation, groups)
        return env.j2t_iso(res)
    
    override_op(torch.nn.functional.conv2d, functools.partial(conv2d_impl, env=env))
    
    # ---- cartesian_prod ----
    def cartesian_prod_impl(tensors, env=env):
        if len(tensors) == 0:
            return env.j2t_iso(jnp.empty((0, 0)))
        if len(tensors) == 1:
            jt = env.t2j_iso(tensors[0])
            return env.j2t_iso(jnp.expand_dims(jt, axis=1))
        jarrays = [env.t2j_iso(t) for t in tensors]
        grids = jnp.meshgrid(*jarrays, indexing='ij')
        result = jnp.stack([g.ravel() for g in grids], axis=-1)
        return env.j2t_iso(result)
    
    try:
        override_op(torch.ops.aten.cartesian_prod.default, functools.partial(cartesian_prod_impl, env=env))
    except Exception:
        pass
    
    # ---- chunk ----
    def chunk_impl(input, chunks, dim=0, env=env):
        jinput = env.t2j_iso(input)
        if dim < 0:
            dim = len(jinput.shape) + dim
        size = jinput.shape[dim]
        chunk_size = (size + chunks - 1) // chunks
        splits = []
        for i in range(chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, size)
            if start >= size:
                break
            slices = [slice(None)] * len(jinput.shape)
            slices[dim] = slice(start, end)
            splits.append(env.j2t_iso(jinput[tuple(slices)]))
        return splits
    
    try:
        override_op(torch.ops.aten.chunk.default, functools.partial(chunk_impl, env=env))
    except Exception:
        pass
    
    # ---- layer_norm ----
    def layer_norm_impl(input, normalized_shape, weight=None, bias=None, eps=1e-5, env=env):
        jinput = env.t2j_iso(input)
        jweight = env.t2j_iso(weight) if weight is not None else None
        jbias = env.t2j_iso(bias) if bias is not None else None
        
        axis = tuple(range(-len(normalized_shape), 0))
        mean = jnp.mean(jinput, axis=axis, keepdims=True)
        var = jnp.var(jinput, axis=axis, keepdims=True)
        result = (jinput - mean) / jnp.sqrt(var + eps)
        
        if jweight is not None:
            result = result * jweight
        if jbias is not None:
            result = result + jbias
        return env.j2t_iso(result)
    
    def native_layer_norm_impl(input, normalized_shape, weight, bias, eps, env=env):
        jinput = env.t2j_iso(input)
        jweight = env.t2j_iso(weight) if weight is not None else None
        jbias = env.t2j_iso(bias) if bias is not None else None
        
        axis = tuple(range(-len(normalized_shape), 0))
        mean = jnp.mean(jinput, axis=axis, keepdims=True)
        var = jnp.var(jinput, axis=axis, keepdims=True)
        rstd = 1.0 / jnp.sqrt(var + eps)
        result = (jinput - mean) * rstd
        
        if jweight is not None:
            result = result * jweight
        if jbias is not None:
            result = result + jbias
        return env.j2t_iso(result), env.j2t_iso(mean.squeeze(axis)), env.j2t_iso(rstd.squeeze(axis))
    
    try:
        override_op(torch.ops.aten.layer_norm.default, functools.partial(layer_norm_impl, env=env))
        override_op(torch.ops.aten.native_layer_norm.default, functools.partial(native_layer_norm_impl, env=env))
    except Exception:
        pass
    
    # ---- unflatten ----
    def unflatten_impl(input, dim, sizes, env=env):
        jinput = env.t2j_iso(input)
        shape = list(jinput.shape)
        if dim < 0:
            dim = len(shape) + dim
        
        sizes = list(sizes)
        if -1 in sizes:
            neg_idx = sizes.index(-1)
            known_prod = 1
            for i, s in enumerate(sizes):
                if i != neg_idx:
                    known_prod *= s
            sizes[neg_idx] = shape[dim] // known_prod
        
        new_shape = shape[:dim] + sizes + shape[dim+1:]
        return env.j2t_iso(jnp.reshape(jinput, new_shape))
    
    try:
        override_op(torch.ops.aten.unflatten.int, functools.partial(unflatten_impl, env=env))
    except Exception:
        pass
    
    # ---- rms_norm ----
    def rms_norm_impl(input, normalized_shape, weight=None, eps=1e-6, env=env):
        jinput = env.t2j_iso(input)
        jweight = env.t2j_iso(weight) if weight is not None else None
        
        axis = tuple(range(-len(normalized_shape), 0))
        rms = jnp.sqrt(jnp.mean(jinput ** 2, axis=axis, keepdims=True) + eps)
        result = jinput / rms
        
        if jweight is not None:
            result = result * jweight
        return env.j2t_iso(result)
    
    try:
        override_op(torch.ops.aten.rms_norm.default, functools.partial(rms_norm_impl, env=env))
        override_op(torch.rms_norm, functools.partial(rms_norm_impl, env=env))
    except Exception:
        pass
    
    # ---- dropout ----
    def dropout_impl(input, p=0.5, training=False, inplace=False, env=env):
        if not training or p == 0:
            return input
        jinput = env.t2j_iso(input)
        key = jax.random.PRNGKey(42)
        mask = jax.random.bernoulli(key, 1 - p, shape=jinput.shape)
        return env.j2t_iso(jinput * mask / (1 - p))
    
    def native_dropout_impl(input, p, train, env=env):
        if not train or p == 0:
            return input, torch.ones_like(input, dtype=torch.bool)
        jinput = env.t2j_iso(input)
        key = jax.random.PRNGKey(42)
        mask = jax.random.bernoulli(key, 1 - p, shape=jinput.shape)
        return env.j2t_iso(jinput * mask / (1 - p)), env.j2t_iso(mask.astype(jnp.bool_))
    
    try:
        override_op(torch.ops.aten.dropout.default, functools.partial(dropout_impl, env=env))
        override_op(torch.ops.aten.native_dropout.default, functools.partial(native_dropout_impl, env=env))
    except Exception:
        pass
    
    # ---- group_norm ----
    def group_norm_impl(input, num_groups, weight=None, bias=None, eps=1e-5, env=env):
        jinput = env.t2j_iso(input)
        jweight = env.t2j_iso(weight) if weight is not None else None
        jbias = env.t2j_iso(bias) if bias is not None else None
        
        shape = jinput.shape
        N, C = shape[0], shape[1]
        spatial_dims = shape[2:]
        group_size = C // num_groups
        
        x = jnp.reshape(jinput, (N, num_groups, group_size) + spatial_dims)
        reduce_axes = tuple(range(2, len(x.shape)))
        mean = jnp.mean(x, axis=reduce_axes, keepdims=True)
        var = jnp.var(x, axis=reduce_axes, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + eps)
        result = jnp.reshape(x, shape)
        
        if jweight is not None:
            weight_shape = (1, C) + (1,) * len(spatial_dims)
            result = result * jnp.reshape(jweight, weight_shape)
        if jbias is not None:
            bias_shape = (1, C) + (1,) * len(spatial_dims)
            result = result + jnp.reshape(jbias, bias_shape)
        return env.j2t_iso(result)
    
    def native_group_norm_impl(input, weight, bias, N, C, HxW, group, eps, env=env):
        jinput = env.t2j_iso(input)
        jweight = env.t2j_iso(weight) if weight is not None else None
        jbias = env.t2j_iso(bias) if bias is not None else None
        
        shape = jinput.shape
        spatial_dims = shape[2:]
        group_size = C // group
        
        x = jnp.reshape(jinput, (N, group, group_size) + spatial_dims)
        reduce_axes = tuple(range(2, len(x.shape)))
        mean = jnp.mean(x, axis=reduce_axes, keepdims=True)
        var = jnp.var(x, axis=reduce_axes, keepdims=True)
        rstd = 1.0 / jnp.sqrt(var + eps)
        x = (x - mean) * rstd
        result = jnp.reshape(x, shape)
        
        if jweight is not None:
            weight_shape = (1, C) + (1,) * len(spatial_dims)
            result = result * jnp.reshape(jweight, weight_shape)
        if jbias is not None:
            bias_shape = (1, C) + (1,) * len(spatial_dims)
            result = result + jnp.reshape(jbias, bias_shape)
        
        mean_out = jnp.mean(x, axis=reduce_axes).reshape(N, group)
        rstd_out = jnp.mean(rstd, axis=reduce_axes).reshape(N, group)
        return env.j2t_iso(result), env.j2t_iso(mean_out), env.j2t_iso(rstd_out)
    
    try:
        override_op(torch.ops.aten.group_norm.default, functools.partial(group_norm_impl, env=env))
        override_op(torch.ops.aten.native_group_norm.default, functools.partial(native_group_norm_impl, env=env))
    except Exception:
        pass
    
    # ---- Splash Attention (可选) ----
    try:
        from .splash_attention import sdpa_reference, tpu_splash_attention
        
        def sdpa_tpu(query, key, value, attn_mask=None, dropout_p=0.0,
                     is_causal=False, scale=None, enable_gqa=False, env=env, mesh=mesh_obj):
            if key.shape[2] > 20000:
                jquery, jkey, jvalue = env.t2j_iso((query, key, value))
                jkey = jkey - jnp.mean(jkey, axis=2, keepdims=True)  # K-smooth
                res = tpu_splash_attention(jquery, jkey, jvalue, mesh, scale=scale)
                return env.j2t_iso(res)
            return sdpa_reference(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)
        
        override_op(torch.nn.functional.scaled_dot_product_attention,
                    functools.partial(sdpa_tpu, env=env, mesh=mesh_obj))
    except ImportError:
        pass
