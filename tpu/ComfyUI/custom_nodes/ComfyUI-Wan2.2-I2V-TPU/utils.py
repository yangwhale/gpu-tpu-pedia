"""
ComfyUI Wan 2.2 I2V TPU - 工具模块
==================================

包含:
  - 配置常量（模型、视频生成、VAE归一化参数）
  - 分片策略配置
  - 权重分片函数
  - PyTree 注册
  - 视频处理工具
  - JAX 配置
  - 算子注册函数
"""

import functools
import os
import re

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.tree_util import register_pytree_node


# ============================================================================
# 模型配置
# ============================================================================

MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"


# ============================================================================
# 视频生成默认参数
# ============================================================================

# 默认参数 (竖屏 832x1104 适合移动端)
DEFAULT_WIDTH = 832
DEFAULT_HEIGHT = 1104
DEFAULT_FRAMES = 81
DEFAULT_FPS = 16

# 尺寸配置
SIZE_CONFIGS = {
    "720*1280": (720, 1280),
    "1280*720": (1280, 720),
    "480*832": (480, 832),
    "832*480": (832, 480),
}

MAX_AREA_CONFIGS = {
    "720*1280": 720 * 1280,
    "1280*720": 1280 * 720,
    "480*832": 480 * 832,
    "832*480": 832 * 480,
}


# ============================================================================
# 推理配置
# ============================================================================

NUM_STEPS = 40
GUIDANCE_SCALE = 3.5
BOUNDARY_RATIO = 0.9  # 双 Transformer 切换阈值
SHIFT = 5.0  # Wan 2.2 I2V 默认 shift 值


# ============================================================================
# VAE 归一化参数 (Wan-AI/Wan2.2-I2V-A14B-Diffusers 默认值)
# ============================================================================

DEFAULT_LATENTS_MEAN = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
]
DEFAULT_LATENTS_STD = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916
]

# VAE 时空压缩因子
VAE_SCALE_FACTOR_TEMPORAL = 4
VAE_SCALE_FACTOR_SPATIAL = 8


# ============================================================================
# Splash Attention 配置
# ============================================================================

BQSIZE = 3328
BKVSIZE = 2816
BKVCOMPUTESIZE = 256
BKVCOMPUTEINSIZE = 256

USE_K_SMOOTH = False
LOG2_E = 1.44269504


# ============================================================================
# Mesh 配置
# ============================================================================

DEFAULT_DP = 2


# ============================================================================
# Text Encoder 分片策略 (UMT5-XXL)
# ============================================================================

TEXT_ENCODER_SHARDINGS = {
    r'shared\.weight': (('dp', 'tp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.SelfAttention\.q\.weight': (('dp', 'tp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.SelfAttention\.k\.weight': (('dp', 'tp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.SelfAttention\.v\.weight': (('dp', 'tp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.SelfAttention\.o\.weight': (None, ('dp', 'tp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.DenseReluDense\.wi_0\.weight': (('dp', 'tp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.DenseReluDense\.wi_1\.weight': (('dp', 'tp'),),
    r'encoder\.block\.\d+\.layer\.\d+\.DenseReluDense\.wo\.weight': (None, ('dp', 'tp'),),
}


# ============================================================================
# Transformer 分片策略 (WanTransformer3DModel)
# ============================================================================

TRANSFORMER_SHARDINGS = {
    # Condition Embedder
    r'condition_embedder\.time_embedder\.linear_1\.weight': ('tp',),
    r'condition_embedder\.time_embedder\.linear_1\.bias': ('tp',),
    r'condition_embedder\.time_embedder\.linear_2\.weight': (None, 'tp',),
    r'condition_embedder\.text_embedder\.linear_1\.weight': ('tp',),
    r'condition_embedder\.text_embedder\.linear_1\.bias': ('tp',),
    r'condition_embedder\.text_embedder\.linear_2\.weight': (None, 'tp',),
    # Self Attention
    r'blocks\.\d+\.attn1\.to_q\.weight': ('tp',),
    r'blocks\.\d+\.attn1\.to_q\.bias': ('tp',),
    r'blocks\.\d+\.attn1\.to_k\.weight': ('tp',),
    r'blocks\.\d+\.attn1\.to_k\.bias': ('tp',),
    r'blocks\.\d+\.attn1\.to_v\.weight': ('tp',),
    r'blocks\.\d+\.attn1\.to_v\.bias': ('tp',),
    r'blocks\.\d+\.attn1\.to_out\.\d+\.weight': (None, 'tp',),
    # Cross Attention
    r'blocks\.\d+\.attn2\.to_q\.weight': ('tp',),
    r'blocks\.\d+\.attn2\.to_q\.bias': ('tp',),
    r'blocks\.\d+\.attn2\.to_k\.weight': ('tp',),
    r'blocks\.\d+\.attn2\.to_k\.bias': ('tp',),
    r'blocks\.\d+\.attn2\.to_v\.weight': ('tp',),
    r'blocks\.\d+\.attn2\.to_v\.bias': ('tp',),
    r'blocks\.\d+\.attn2\.to_out\.\d+\.weight': (None, 'tp',),
    # FFN
    r'blocks\.\d+\.ffn\.net\.\d+\.proj\.weight': ('tp',),
    r'blocks\.\d+\.ffn\.net\.\d+\.proj\.bias': ('tp',),
    r'blocks\.\d+\.ffn\.net\.\d+\.weight': (None, 'tp',),
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
        sharding_dict: 分片规则 {pattern: (axis0, axis1, ...)}
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
# Latent 归一化函数
# ============================================================================

def get_latents_params(vae=None, mean=None, std=None):
    """
    获取 latent 归一化参数
    
    优先级：传入参数 > VAE config > 默认值
    """
    if mean is not None and std is not None:
        return mean, std
    
    if vae is not None and hasattr(vae, 'config'):
        config = vae.config
        latents_mean = getattr(config, 'latents_mean', DEFAULT_LATENTS_MEAN)
        latents_std = getattr(config, 'latents_std', DEFAULT_LATENTS_STD)
        return latents_mean, latents_std
    
    return DEFAULT_LATENTS_MEAN, DEFAULT_LATENTS_STD


def normalize_latents(latents, vae=None, mean=None, std=None):
    """
    归一化 latents: (x - mean) / std
    
    Args:
        latents: [B, C, T, H, W] 格式的 latent tensor
        vae: VAE 模型实例（用于获取 config 中的参数）
        mean: 直接传入的 mean 值（可选）
        std: 直接传入的 std 值（可选）
        
    Returns:
        归一化后的 latents
    """
    latents_mean, latents_std = get_latents_params(vae, mean, std)
    mean_tensor = torch.tensor(latents_mean).view(1, 16, 1, 1, 1).to(latents.device, latents.dtype)
    std_inv = 1.0 / torch.tensor(latents_std).view(1, 16, 1, 1, 1).to(latents.device, latents.dtype)
    return (latents - mean_tensor) * std_inv


def denormalize_latents(latents, vae=None, mean=None, std=None):
    """
    反归一化 latents: x * std + mean
    
    Args:
        latents: [B, C, T, H, W] 格式的归一化 latent tensor
        vae: VAE 模型实例（用于获取 config 中的参数）
        mean: 直接传入的 mean 值（可选）
        std: 直接传入的 std 值（可选）
        
    Returns:
        反归一化后的 latents
    """
    latents_mean, latents_std = get_latents_params(vae, mean, std)
    mean_tensor = torch.tensor(latents_mean).view(1, 16, 1, 1, 1).to(latents.device, latents.dtype)
    std_tensor = torch.tensor(latents_std).view(1, 16, 1, 1, 1).to(latents.device, latents.dtype)
    return latents * std_tensor + mean_tensor


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
      - DiagonalGaussianDistribution (diffusers VAE)
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
    
    # 标准模型输出
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
    
    # DiagonalGaussianDistribution 需要特殊处理
    def flatten_gaussian(obj):
        return (obj.parameters, obj.mean, obj.logvar, obj.deterministic,
                obj.std, obj.var), None
    
    def unflatten_gaussian(aux, children):
        obj = object.__new__(diffusers_vae.DiagonalGaussianDistribution)
        obj.parameters = children[0]
        obj.mean = children[1]
        obj.logvar = children[2]
        obj.deterministic = children[3]
        obj.std = children[4]
        obj.var = children[5]
        return obj
    
    try:
        register_pytree_node(
            diffusers_vae.DiagonalGaussianDistribution,
            flatten_gaussian,
            unflatten_gaussian
        )
        print("  - DiagonalGaussianDistribution 已注册")
    except ValueError:
        print("  - DiagonalGaussianDistribution 已存在")
    
    _pytree_registered = True


# ============================================================================
# 视频处理工具
# ============================================================================

def prepare_video_for_export(video, target_frames):
    """
    准备视频 tensor 用于导出。
    
    输入: JAX VAE 输出格式 [B, T, H, W, C]
    输出: numpy array [T, H, W, C] (float32, 范围 [0, 1])
    
    Args:
        video: 视频 tensor 或 numpy array
        target_frames: 目标帧数（用于验证）
        
    Returns:
        numpy array: [T, H, W, C] 格式的 float32 视频
    """
    if isinstance(video, (list, tuple)):
        return [prepare_video_for_export(v, target_frames) for v in video]
    
    if isinstance(video, torch.Tensor):
        if video.dim() == 5:
            if video.shape[-1] == 3:  # [B, T, H, W, C]
                video = video.permute(0, 4, 1, 2, 3)  # -> [B, C, T, H, W]
            
            batch_vid = video[0]  # [C, T, H, W]
            batch_vid = batch_vid.permute(1, 0, 2, 3)  # -> [T, C, H, W]
            batch_vid = (batch_vid * 0.5 + 0.5).clamp(0, 1)  # denormalize
            video = batch_vid.cpu().permute(0, 2, 3, 1).float().numpy()  # -> [T, H, W, C]
            
        elif video.dim() == 4:
            if video.shape[0] == 3:  # [C, T, H, W]
                batch_vid = video.permute(1, 0, 2, 3)
                batch_vid = (batch_vid * 0.5 + 0.5).clamp(0, 1)
                video = batch_vid.cpu().permute(0, 2, 3, 1).float().numpy()
            elif video.shape[-1] == 3:  # [T, H, W, C]
                video = (video * 0.5 + 0.5).clamp(0, 1)
                video = video.cpu().float().numpy()
        
        if video.shape[-1] == 1:
            video = np.repeat(video, 3, axis=-1)
        return video
    
    if isinstance(video, np.ndarray):
        if video.ndim == 5:
            if video.shape[-1] == 3:  # [B, T, H, W, C]
                video = np.transpose(video, (0, 4, 1, 2, 3))
            
            batch_vid = video[0]  # [C, T, H, W]
            batch_vid = np.transpose(batch_vid, (1, 0, 2, 3))  # -> [T, C, H, W]
            
            if batch_vid.min() < 0:
                batch_vid = np.clip(batch_vid * 0.5 + 0.5, 0, 1)
            
            video = np.transpose(batch_vid, (0, 2, 3, 1))  # -> [T, H, W, C]
            
        elif video.ndim == 4:
            if video.shape[0] == 3:  # [C, T, H, W]
                video = np.transpose(video, (1, 0, 2, 3))
                if video.min() < 0:
                    video = np.clip(video * 0.5 + 0.5, 0, 1)
                video = np.transpose(video, (0, 2, 3, 1))
            elif video.shape[-1] == 3:  # [T, H, W, C]
                if video.min() < 0:
                    video = np.clip(video * 0.5 + 0.5, 0, 1)
        
        video = video.astype(np.float32)
        
        if video.shape[-1] == 1:
            video = np.repeat(video, 3, axis=-1)
        return video
    
    return video


# ============================================================================
# JAX 配置
# ============================================================================

def setup_jax_cache():
    """设置 JAX 编译缓存以加速后续编译。"""
    cache_dir = os.path.expanduser("~/.cache/jax_cache")
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    print(f"✓ JAX 编译缓存: {cache_dir}")


# ============================================================================
# 算子注册
# ============================================================================

_text_encoder_ops_registered = False


def register_text_encoder_ops(env):
    """
    注册 UMT5 Text Encoder 需要的算子。
    """
    global _text_encoder_ops_registered
    if _text_encoder_ops_registered:
        return
    
    from torchax.ops import ops_registry
    
    def override_op(op, impl):
        """注册或覆盖一个算子"""
        env._ops[op] = ops_registry.Operator(
            op, impl, is_jax_function=False, is_user_defined=True,
            needs_env=False, is_view_op=False,
        )
    
    # ---- dropout ----
    def dropout_impl(input, p=0.5, training=False, inplace=False, env=env):
        return input
    
    def native_dropout_impl(input, p, train, env=env):
        if hasattr(input, '_elem'):
            mask = torch.ones(input.shape, dtype=torch.bool)
            return input, mask.to('jax')
        return input, torch.ones_like(input, dtype=torch.bool)
    
    try:
        override_op(torch.ops.aten.dropout.default, functools.partial(dropout_impl, env=env))
        override_op(torch.ops.aten.native_dropout.default, functools.partial(native_dropout_impl, env=env))
        print("  - Registered dropout operators")
    except Exception as e:
        print(f"  - Warning: Failed to register dropout: {e}")
    
    # ---- minimum/maximum ----
    def minimum_impl(input, other, env=env):
        jinput = env.t2j_iso(input)
        jother = env.t2j_iso(other)
        return env.j2t_iso(jnp.minimum(jinput, jother))
    
    def maximum_impl(input, other, env=env):
        jinput = env.t2j_iso(input)
        jother = env.t2j_iso(other)
        return env.j2t_iso(jnp.maximum(jinput, jother))
    
    try:
        override_op(torch.ops.aten.minimum.default, functools.partial(minimum_impl, env=env))
        override_op(torch.ops.aten.maximum.default, functools.partial(maximum_impl, env=env))
        override_op(torch.ops.aten.min.other, functools.partial(minimum_impl, env=env))
        override_op(torch.ops.aten.max.other, functools.partial(maximum_impl, env=env))
        print("  - Registered minimum/maximum operators")
    except Exception as e:
        print(f"  - Warning: Failed to register minimum/maximum: {e}")
    
    # ---- clamp ----
    def clamp_impl(input, min_val=None, max_val=None, env=env):
        jinput = env.t2j_iso(input)
        if min_val is not None:
            jinput = jnp.maximum(jinput, min_val)
        if max_val is not None:
            jinput = jnp.minimum(jinput, max_val)
        return env.j2t_iso(jinput)
    
    try:
        override_op(torch.ops.aten.clamp.default, functools.partial(clamp_impl, env=env))
        print("  - Registered clamp operator")
    except Exception as e:
        print(f"  - Warning: Failed to register clamp: {e}")
    
    # ---- abs ----
    def abs_impl(input, env=env):
        jinput = env.t2j_iso(input)
        return env.j2t_iso(jnp.abs(jinput))
    
    try:
        override_op(torch.ops.aten.abs.default, functools.partial(abs_impl, env=env))
        print("  - Registered abs operator")
    except Exception as e:
        print(f"  - Warning: Failed to register abs: {e}")
    
    # ---- log ----
    def log_impl(input, env=env):
        jinput = env.t2j_iso(input)
        return env.j2t_iso(jnp.log(jinput))
    
    try:
        override_op(torch.ops.aten.log.default, functools.partial(log_impl, env=env))
        print("  - Registered log operator")
    except Exception as e:
        print(f"  - Warning: Failed to register log: {e}")
    
    # ---- floor ----
    def floor_impl(input, env=env):
        jinput = env.t2j_iso(input)
        return env.j2t_iso(jnp.floor(jinput))
    
    try:
        override_op(torch.ops.aten.floor.default, functools.partial(floor_impl, env=env))
        print("  - Registered floor operator")
    except Exception as e:
        print(f"  - Warning: Failed to register floor: {e}")
    
    # ---- item ----
    def item_impl(input, env=env):
        jinput = env.t2j_iso(input)
        scalar_val = np.array(jinput).item()
        if jinput.dtype in (jnp.int32, jnp.int64, jnp.int16, jnp.int8, jnp.uint32, jnp.uint64):
            return int(scalar_val)
        return scalar_val
    
    try:
        override_op(torch.ops.aten.item.default, functools.partial(item_impl, env=env))
        override_op(torch.ops.aten._local_scalar_dense.default, functools.partial(item_impl, env=env))
        print("  - Registered item operator")
    except Exception as e:
        print(f"  - Warning: Failed to register item: {e}")
    
    _text_encoder_ops_registered = True


def register_conv_ops(env):
    """
    注册卷积算子 (conv2d, conv3d)
    """
    from torchax.ops import jaten, ops_registry
    
    def override_op(op, impl):
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
    print("  - Registered conv2d operator")
    
    # ---- conv3d ----
    def conv3d_impl(input, weight, bias=None, stride=1, padding=0,
                    dilation=1, groups=1, *, env=env):
        jinput, jweight, jbias = env.t2j_iso((input, weight, bias))
        res = jaten._aten_convolution(
            jinput, jweight, jbias,
            stride, padding, dilation,
            transposed=False,
            output_padding=1,
            groups=groups
        )
        return env.j2t_iso(res)
    
    conv3d_fn = functools.partial(conv3d_impl, env=env)
    override_op(torch.nn.functional.conv3d, conv3d_fn)
    try:
        override_op(torch.ops.aten.conv3d, conv3d_fn)
        override_op(torch.ops.aten.conv3d.default, conv3d_fn)
    except Exception:
        pass
    print("  - Registered conv3d operator")


def register_expand_as_op(env):
    """
    注册 expand_as 算子（用于 F.normalize）
    """
    from torchax.ops import ops_registry
    
    def expand_as_impl(input, other, env=env):
        jinput = env.t2j_iso(input)
        jother = env.t2j_iso(other)
        target_shape = jother.shape
        result = jnp.broadcast_to(jinput, target_shape)
        return env.j2t_iso(result)
    
    env._ops[torch.ops.aten.expand_as.default] = ops_registry.Operator(
        torch.ops.aten.expand_as.default, functools.partial(expand_as_impl, env=env),
        is_jax_function=False, is_user_defined=True,
        needs_env=False, is_view_op=False,
    )
    print("  - Registered expand_as operator")
