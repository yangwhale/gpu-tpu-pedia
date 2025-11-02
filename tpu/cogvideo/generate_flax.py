import time
import re
import math
import functools
import numpy as np
import jax
import jax.numpy as jnp
import torch
import imageio
from diffusers import CogVideoXPipeline
from diffusers.models.autoencoders.vae import DecoderOutput
import torchax
from torchax.ops import ops_registry
from jax.tree_util import register_pytree_node
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
from jax.experimental.pallas.ops.tpu import splash_attention
from jax.experimental.shard_map import shard_map
from jax.experimental import mesh_utils
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutputWithPastAndCrossAttentions
import warnings
import logging
from flax.linen import partitioning as nn_partitioning
import sys
sys.path.insert(0, '/home/chrisya/diffusers-tpu-chris/src')
from diffusers.models.autoencoders.autoencoder_kl_cogvideox_flax import FlaxAutoencoderKLCogVideoX

MODEL_NAME = "zai-org/CogVideoX1.5-5B"
#### Splash Attention 配置参数 ####
# Splash attention 块大小配置
# 注意：这些值需要根据 TPU vmem 限制调整
# 减小块大小以避免 vmem 超限（当前 vmem 限制约 32MB）
BQSIZE = 2048           # Query 块大小（从 3024 减小）
BKVSIZE = 1024          # Key/Value 块大小（从 2048 减小）
BKVCOMPUTESIZE = 512    # Key/Value 计算块大小（从 1024 减小）

# 窗口大小（None 表示使用完整注意力，否则使用局部窗口注意力）
WINDOW_SIZE = None

# 是否使用 K-smooth（对 key 进行平滑处理）
USE_K_SMOOTH = True

# Mesh 分片配置
USE_DP = False          # 是否使用 data parallelism
SP_NUM = 1             # Spatial parallelism 数量
USE_FSDP = True        # 是否使用 FSDP 模式（vs Tensor Parallel）

# VAE sharding 配置
LOGICAL_AXIS_RULES = (
    ('conv_out', ('tp','dp','sp')),
    ('conv_in', ('tp','dp','sp'))
)

def to_torch_recursive(x):
    """递归地将 JAX 数组转换为 PyTorch 张量"""
    if 'ArrayImpl' in str(type(x)) or isinstance(x, jnp.ndarray):
        # 处理 JAX 数组
        np_array = np.array(x)
        # 如果是 bfloat16，通过 float32 转换
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
        # 特别处理 BFloat16
        if x.dtype == torch.bfloat16:
            # 先转换为 float32，再转为 JAX 数组
            return jnp.array(x.detach().to(torch.float32).cpu().numpy()).astype(jnp.bfloat16)
        else:
            return jnp.array(x.detach().cpu().numpy())
    elif isinstance(x, (list, tuple)):
        return type(x)(to_jax_recursive(xx) for xx in x)
    elif isinstance(x, dict):
        return {k: to_jax_recursive(v) for k, v in x.items()}
    else:
        return x

def setup_jax_config():
    """配置JAX环境参数"""
    # jax.config.update('jax_default_matmul_precision', 'high')
    # print("JAX配置: 使用高精度矩阵乘法")


def setup_pytree_registrations():
    """
    注册必要的pytree节点以支持JAX转换
    使其可以在JAX的函数转换中正常使用
    """
    print("注册PyTree节点...")
    
    # 注册 PyTree 节点的通用 flatten 和 unflatten 方法
    def model_output_flatten(obj):
        """将模型输出对象展平为元组"""
        return obj.to_tuple(), type(obj)

    def model_output_unflatten(aux, children):
        """从元组重建模型输出对象"""
        return aux(*children)
    
    # 定义需要注册的所有类型
    OUTPUT_CLASSES = [
        BaseModelOutputWithPooling,
        BaseModelOutputWithPastAndCrossAttentions,
        DecoderOutput,
    ]

    # 批量注册
    for cls in OUTPUT_CLASSES:
        register_pytree_node(cls, model_output_flatten, model_output_unflatten)
        print(f"  - {cls.__name__} 已注册")

#### Splash Attention 实现 ####

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
    Scaled Dot-Product Attention 参考实现
    
    用于在不支持 Splash attention 时作为回退方案
    """
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(
            L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
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


def _tpu_splash_attention(query, key, value, env, scale=None, is_causal=False, window_size=None):
    """
    TPU Splash Attention 实现
    
    使用 JAX 的 Splash Attention 在 TPU 上高效计算注意力
    支持可选的局部窗口注意力和 K-smooth
    
    Args:
        query: Query 张量 [batch, num_heads, seq_len, head_dim]
        key: Key 张量 [batch, num_heads, seq_len, head_dim]
        value: Value 张量 [batch, num_heads, seq_len, head_dim]
        env: torchax 环境
        scale: 缩放因子（默认为 1/sqrt(head_dim)）
        is_causal: 是否使用因果掩码
        window_size: 局部窗口大小（None 表示全局注意力）
        
    Returns:
        注意力输出张量
    """
    mesh = env._mesh
    num_heads = query.shape[1]

    # 在设备切片上执行的注意力函数
    def _attention_on_slices(q, k, v):
        # 缩放 query 张量
        scale_factor = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
        q = q * scale_factor

        # 辅助函数：填充到指定倍数
        def pad_to_multiple(x, multiple, axis):
            seq_len = x.shape[axis]
            pad_len = (multiple - seq_len % multiple) % multiple
            if pad_len == 0:
                return x, seq_len
            pad_width = [(0, 0)] * x.ndim
            pad_width[axis] = (0, pad_len)
            return jnp.pad(x, pad_width), seq_len

        # 在批次维度上操作的核函数
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
    # 参考: diffusers/cog_tx_splash_attn.py 第 287-301 行
    if num_heads < mesh.size:
        # 头数太少，复制到所有设备
        q_partition_spec = P()
        kv_partition_spec = P()
    else:
        # 根据 query 和 key 的序列长度判断是自注意力还是交叉注意力
        # 自注意力：q 和 k 序列长度相同
        # 交叉注意力：q 和 k 序列长度不同
        if query.shape[2] == key.shape[2]:  # 自注意力
            # 在三维 mesh 上分片 (dp, tp, sp, None)
            q_partition_spec = P('dp', 'tp', 'sp', None)
            kv_partition_spec = P('dp', 'tp', None, None)
        else:  # 交叉注意力
            # 交叉注意力的 sharding 策略不同
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
    
    # 应用输出 sharding constraint
    # 使用 NamedSharding 而不是 PartitionSpec，避免需要 mesh context
    # output_sharding = NamedSharding(mesh, P('dp', None, ('tp', 'sp'), None))
    # out = jax.lax.with_sharding_constraint(out, output_sharding)
    
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
    
    根据环境配置选择使用 TPU Splash Attention 或参考实现
    
    Args:
        query: Query 张量
        key: Key 张量
        value: Value 张量
        attn_mask: 注意力掩码（可选）
        dropout_p: Dropout 概率
        is_causal: 是否使用因果掩码
        scale: 缩放因子
        enable_gqa: 是否启用 GQA (Grouped Query Attention)
        env: torchax 环境
        window_size: 局部窗口大小（用于 Splash Attention）
        
    Returns:
        注意力输出张量
    """
    if env is not None and hasattr(env.config, 'use_tpu_splash_attention') and env.config.use_tpu_splash_attention:
        # 使用 TPU Splash Attention
        jquery, jkey, jvalue = env.t2j_iso((query, key, value))
        
        # 可选的 K-smooth 处理
        if USE_K_SMOOTH:
            key_mean = jnp.mean(jkey, axis=2, keepdims=True)
            jkey = jkey - key_mean
        
        res = _tpu_splash_attention(jquery, jkey, jvalue, env, scale=scale, is_causal=is_causal, window_size=window_size)
        return env.j2t_iso(res)
    
    # 回退到参考实现
    return _sdpa_reference(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)

########################################


def load_cogvideo_pipeline(model_name=MODEL_NAME):
    """
    加载CogVideoX Pipeline
    
    Args:
        model_name: 预训练模型名称
        
    Returns:
        pipe: CogVideoX Pipeline实例
    """
    print(f"正在加载模型: {model_name}")
    pipe = CogVideoXPipeline.from_pretrained(model_name)
    print("模型加载完成")
    return pipe


# Transformer sharding策略 - FSDP模式（默认）
# 参考: diffusers/cog_tx_splash_attn.py 第 82-91 行
# 注意：所有模式都以 .weight$ 结尾，这样不会匹配到 bias 等1维参数
# 使用三维 mesh (tp, dp, sp)
transformer_shardings_fsdp = {
    # Attention layers - 在输出维度分片
    r'.*\.to_q\.weight$': (None, ('tp', 'sp')),
    r'.*\.to_k\.weight$': (None, ('tp', 'sp')),
    r'.*\.to_v\.weight$': (None, ('tp', 'sp')),
    r'.*\.to_out.*\.weight$': (('tp', 'sp'), None),
    # Feedforward layers
    r'.*\.ff\.net\.0\.weight$': (None, ('tp', 'sp')),
    r'.*\.ff\.net\.2\.weight$': (('tp', 'sp'), None),
}

# Transformer sharding策略 - Tensor Parallel模式
# 参考: diffusers/cog_tx_splash_attn.py 第 93-102 行
transformer_shardings_tp = {
    # Attention layers - 在输入维度分片
    r'.*\.to_q\.weight$': (('tp', 'sp'), None),
    r'.*\.to_k\.weight$': (('tp', 'sp'), None),
    r'.*\.to_v\.weight$': (('tp', 'sp'), None),
    r'.*\.to_out.*\.weight$': (None, ('tp', 'sp')),
    # Feedforward layers
    r'.*\.ff\.net\.0\.weight$': (('tp', 'sp'), None),
    r'.*\.ff\.net\.2\.weight$': (None, ('tp', 'sp')),
}

# Text Encoder (T5) sharding策略
# 参考: diffusers/cog_tx_splash_attn.py 第 104-116 行
# 使用三维 mesh (tp, dp, sp)
text_encoder_shardings = {
    r'shared\.weight$': (('tp', 'dp', 'sp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.SelfAttention\.q\.weight$': (('tp', 'dp', 'sp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.SelfAttention\.k\.weight$': (('tp', 'dp', 'sp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.SelfAttention\.v\.weight$': (('tp', 'dp', 'sp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.SelfAttention\.o\.weight$': (None, ('tp', 'dp', 'sp')),
    r'encoder\.block\.\d+\.layer\.\d+\.DenseReluDense\.wi_0\.weight$': (('tp', 'dp', 'sp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.DenseReluDense\.wi_1\.weight$': (('tp', 'dp', 'sp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.DenseReluDense\.wo\.weight$': (None, ('tp', 'dp', 'sp')),
}

# VAE sharding策略
# 对于卷积层，在输出通道维度分片（第0维）
# 参考: diffusers/cog_tx_splash_attn.py 中的 LOGICAL_AXIS_RULES
vae_shardings = {
    # Encoder 卷积层 - 在输出通道分片
    r'encoder\..*\.conv\.weight$': ('tp', None, None, None),
    r'encoder\..*\.conv_in\.weight$': ('tp', None, None, None),
    r'encoder\..*\.conv_out\.weight$': ('tp', None, None, None),
    # Decoder 卷积层 - 在输出通道分片
    r'decoder\..*\.conv\.weight$': ('tp', None, None, None),
    r'decoder\..*\.conv_in\.weight$': ('tp', None, None, None),
    r'decoder\..*\.conv_out\.weight$': ('tp', None, None, None),
    # 其他卷积层
    r'.*\.conv_shortcut\.weight$': ('tp', None, None, None),
}


def shard_weights_transformer(mesh, weights, use_fsdp=True):
    """
    对CogVideoX Transformer模型的权重进行分片
    关键点：
    - 只匹配以 .weight 结尾的参数，避免对1维的 bias 等参数进行错误分片
    - 使用 apply_jax_ (in-place) 而不是 apply_jax
    - 未匹配的参数会被复制到所有设备
    
    Args:
        mesh: JAX设备网格
        weights: 模型权重字典
        use_fsdp: 是否使用FSDP模式（默认True），否则使用Tensor Parallel模式
        
    Returns:
        分片后的权重字典
    """
    # 选择分片策略
    sharding_dict = transformer_shardings_fsdp if use_fsdp else transformer_shardings_tp
    
    result = {}
    for k, v in weights.items():
        # 尝试匹配分片规则
        matched = False
        for target, sharding in sharding_dict.items():
            if re.fullmatch(target, k) is not None:
                # 找到匹配的模式，应用分片（使用 apply_jax_ 进行 in-place 操作）
                v.apply_jax_(jax.device_put, NamedSharding(mesh, P(*sharding)))
                matched = True
                break
        
        if not matched:
            # 没有匹配到任何模式，复制到所有设备
            v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
        
        result[k] = v
    return result


def shard_weights_text_encoder(mesh, weights):
    """
    对Text Encoder (T5)模型的权重进行分片
        
    Args:
        mesh: JAX设备网格
        weights: Text Encoder权重字典
        
    Returns:
        分片后的权重字典
    """
    result = {}
    for k, v in weights.items():
        # 尝试匹配分片规则
        matched = False
        for target, sharding in text_encoder_shardings.items():
            if re.fullmatch(target, k) is not None:
                # 找到匹配的模式，应用分片
                v.apply_jax_(jax.device_put, NamedSharding(mesh, P(*sharding)))
                matched = True
                break
        
        if not matched:
            # 没有匹配到任何模式，复制到所有设备
            v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
        
        result[k] = v
    return result


def shard_weights_vae(mesh, weights):
    """
    对VAE模型的权重进行分片
    
    注意：VAE的卷积层通道数可能不规则（如3、128等），
    在不同设备数（4、8、16等）下可能无法整除。
    因此当前策略是将所有VAE权重复制到所有设备，不进行分片。
    
    Args:
        mesh: JAX设备网格
        weights: VAE权重字典
        
    Returns:
        分片后的权重字典（实际上是复制到所有设备）
    """
    result = {}
    for k, v in weights.items():
        # VAE权重全部复制到所有设备，不进行分片
        # 这样可以在任意数量的设备上工作（4、8、16等）
        v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
        result[k] = v
    return result


def setup_pipeline_for_jax(pipe, model_id=MODEL_NAME):
    """
    设置Pipeline以在JAX环境中运行
    
    将所有模型权重移动到JAX设备并编译关键组件:
    - Transformer: DiT模型的核心网络
    - VAE: 使用 Flax 原生实现（支持长视频）
    - Text Encoder: 文本编码器
    
    同时注册自定义的 Splash Attention 实现
    """
    print("\n配置Pipeline以使用JAX...")

    tp_dim, dp_dim, sp_dim = jax.device_count(), 1, 1
    # 根据配置调整维度
    if USE_DP:  # 默认 False
        tp_dim //= 2
        dp_dim = 2
    
    if SP_NUM > 1:  # 默认 1
        tp_dim //= SP_NUM
        sp_dim = SP_NUM
    
    print(f"  Mesh 维度: tp_dim={tp_dim}, dp_dim={dp_dim}, sp_dim={sp_dim}")
    
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
    
    # 辅助函数：将scheduler模块权重移动到 XLA
    def _move_scheduler_to_jax(scheduler):
        print("将scheduler参数移动到JAX设备...")
        for k, v in scheduler.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(scheduler, k, v.to('jax'))

    # 辅助函数：将模块权重移动到 XLA
    def _move_module_to_xla(module):
        """将模块的权重转换为 JAX Array，但先在 CPU 上操作"""
        with jax.default_device('cpu'):
            state_dict=module.state_dict()
            state_dict = env.to_xla(state_dict)
            module.load_state_dict(state_dict, assign=True)
    
    with env:
        # 移动scheduler参数
        _move_scheduler_to_jax(pipe.scheduler)
        
        # 对 Transformer 进行处理：先移到 XLA，再分片
        _move_module_to_xla(pipe.transformer)
        transformer_weights = shard_weights_transformer(mesh, pipe.transformer.state_dict())
        pipe.transformer.load_state_dict(transformer_weights, assign=True, strict=False)
        # 确保所有权重已分片完成
        torchax.interop.call_jax(jax.block_until_ready, transformer_weights)
        
        # 对 Text Encoder 进行处理：先移到 XLA，再分片
        print("- 将Text Encoder移到XLA并进行分片...")
        _move_module_to_xla(pipe.text_encoder)
        text_encoder_weights = shard_weights_text_encoder(mesh, pipe.text_encoder.state_dict())
        pipe.text_encoder.load_state_dict(text_encoder_weights, assign=True, strict=False)
        # 确保所有权重已分片完成
        torchax.interop.call_jax(jax.block_until_ready, text_encoder_weights)
        
        # 替换 PyTorch VAE 为 Flax VAE
        print("- 加载 Flax VAE (原生 JAX 实现)...")
        flax_vae = FlaxAutoencoderKLCogVideoX.from_pretrained(
            model_id,
            subfolder="vae",
            dtype=jnp.bfloat16
        )
        
        # 禁用 Tiling（用于测试）
        print("- 跳过 VAE Tiling 配置（测试完整解码）...")
        # flax_vae.enable_tiling(
        #     tile_sample_min_height=192,
        #     tile_sample_min_width=340,
        #     tile_overlap_factor_height=1/6,
        #     tile_overlap_factor_width=1/5,
        # )
        
        # 使用 FlaxVAEProxy 包装并替换 pipeline 的 VAE
        pipe.vae = FlaxVAEProxy(flax_vae)
        print("  ✓ Flax VAE 已替换 Pipeline 的 VAE")
        
        # 编译transformer（DiT的核心网络）
        pipe.transformer = torchax.compile(
            pipe.transformer,
            torchax.CompileOptions(
                jax_jit_kwargs={'static_argnames': ('return_dict', )}
            )
        )
        
        # 编译文本编码器
        pipe.text_encoder = torchax.compile(pipe.text_encoder)
    
    print("Pipeline配置完成")
    return pipe, env, mesh


class FlaxVAEProxy:
    """Flax VAE 的 PyTorch 接口代理"""
    def __init__(self, flax_vae):
        self._flax_vae = flax_vae
        self.config = flax_vae.config
        self.dtype = torch.bfloat16
    
    def __getattr__(self, name):
        return getattr(self._flax_vae, name)
    
    def decode(self, latents, return_dict=True):
        """解码：PyTorch -> JAX -> 解码 -> PyTorch
        
        关键优化：全程使用 BF16，避免 FP32 中间数组占用 2倍内存
        """
        # 应用缩放因子
        scaling_factor = self.config.scaling_factor if hasattr(self.config, 'scaling_factor') else 1.15258426
        latents = latents / scaling_factor
        
        # 转换 latents: PyTorch (B, C, T, H, W) -> numpy (B, T, H, W, C)
        # 关键：保持 BF16 dtype 以节省内存
        if latents.dtype == torch.bfloat16:
            # BF16: 通过 FP32 中转（JAX 的 bfloat16 限制）
            latents_np = latents.to(torch.float32).cpu().numpy()
        else:
            latents_np = latents.cpu().numpy()
        
        # Transpose 并**直接创建 BF16 数组**（避免 FP32 中间数组）
        latents_jax = jnp.array(
            np.transpose(latents_np, (0, 2, 3, 4, 1)),
            dtype=jnp.bfloat16  # 关键修复：直接创建 BF16，不走 FP32
        )
        
        # Flax VAE 解码（BF16 -> BF16）
        frames_jax = self._flax_vae.decode(latents_jax)
        
        # 转换输出: JAX (B, T, H, W, C) -> PyTorch (B, C, T, H, W)
        # 保持 BF16 以节省内存
        # 先在 JAX 上 transpose (避免 numpy 上的大数组操作)
        frames_jax_transposed = frames_jax.transpose(0, 4, 1, 2, 3)  # (B, C, T, H, W) in BF16
        
        # 转为 numpy（保持 BF16 view，不创建副本）
        # 注意：np.asarray 创建 view，不分配新内存
        frames_np = np.asarray(frames_jax_transposed)
        
        # 转为 PyTorch（通过 FP32 中转，因为 torch.from_numpy 不支持 bfloat16）
        # 这里的 FP32 转换是必须的，但数据已经 transpose 过，所以内存占用最小
        frames_torch = torch.from_numpy(frames_np.astype(np.float32)).to(torch.bfloat16)
        
        # 返回 DecoderOutput
        if return_dict:
            return DecoderOutput(sample=frames_torch)
        return frames_torch


def run_generation_benchmark(pipe, prompt, num_inference_steps=20, num_frames=49, height=56, width=104, num_iterations=2):
    """
    运行视频生成基准测试
    
    Args:
        pipe: CogVideoX Pipeline
        prompt: 提示词
        num_inference_steps: 推理步数
        num_frames: 视频帧数
        height: 视频高度
        width: 视频宽度
        num_iterations: 迭代次数
        
    Returns:
        frames: 最后生成的视频帧
        times: 各次迭代的时间列表
    """
    print(f"\n运行{num_iterations}次视频生成基准测试...")
    print(f"提示词: '{prompt}'")
    print(f"推理步数: {num_inference_steps}")
    print(f"视频帧数: {num_frames}")
    print(f"分辨率: {height}x{width}")
    
    times = []
    frames = None
    
    for i in range(num_iterations):
        if i == 0:
            print(f"\n迭代 {i} (包含 JIT 编译，会比较慢):")
        else:
            print(f"\n迭代 {i} (使用已编译代码):")
        
        start = time.perf_counter()
        result = pipe(prompt, num_inference_steps=num_inference_steps, num_frames=num_frames, height=height, width=width)
        frames = result.frames[0]  # CogVideoX 返回 frames 而不是 images
        end = time.perf_counter()
        elapsed = end - start
        times.append(elapsed)
        
        if i == 0:
            print(f"  完成时间: {elapsed:.2f} 秒 (包含 Transformer + Text Encoder 的真正 JIT 编译)")
        else:
            print(f"  完成时间: {elapsed:.2f} 秒")
        
        # 显式删除中间结果以释放内存
        del result
    
    return frames, times

def print_performance_summary(times):
    """
    Args:
        times: 时间列表
    """
    if len(times) > 0:
        print(f"总迭代次数: {len(times)}")
        print(f"第一次运行（含编译）: {times[0]:.4f} 秒")
        
        if len(times) > 1:
            avg_time = sum(times[1:]) / len(times[1:])
            print(f"后续运行平均时间: {avg_time:.4f} 秒")
            print(f"加速比: {times[0] / avg_time:.2f}x")
            print(f"\n说明:")
            print(f"- 第一次运行包含JIT编译时间, 因此较慢")
            print(f"- 后续运行使用编译后的代码, 速度显著提升")
        
        print(f"\n各次迭代详细时间:")
        for i, t in enumerate(times):
            print(f"  迭代 {i}: {t:.4f} 秒")

def main():
    # Set JAX config to enable compilation cache
    jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

    warnings.filterwarnings('ignore', message='.*dtype.*int64.*truncated to dtype int32.*')
    logging.getLogger().setLevel(logging.ERROR)

    torch.set_default_dtype(torch.bfloat16)
 
    setup_pytree_registrations()
    
    pipe = CogVideoXPipeline.from_pretrained(MODEL_NAME)
    print("\n配置Pipeline以使用JAX、Splash Attention 和 Flax VAE...")
    pipe, env, mesh = setup_pipeline_for_jax(pipe)
    
    prompt = "A cat walks on the grass, realistic style."
    # prompt = "A dog cooking cake in the kithen, realistic style."
    
    with mesh, nn_partitioning.axis_rules(LOGICAL_AXIS_RULES), env:
        frames, times = run_generation_benchmark(
            pipe,
            prompt,
            num_inference_steps=50,  # 增加推理步数
            num_frames=16,           # 16 帧（测试 Tiling）
            height=768,              # 480x720 标准分辨率
            width=1360,
            num_iterations=2
        )
    
    print("\n保存生成的视频...")
    output_path = 'output_video_flax_vae.mp4'
    imageio.mimsave(output_path, frames, fps=8)
    print(f"视频已保存到: {output_path}")
    
    print_performance_summary(times)
    
    # 显式退出程序
    print("\n✓ 测试完成，程序退出")
    import sys
    sys.exit(0)

if __name__ == "__main__":
    main()
