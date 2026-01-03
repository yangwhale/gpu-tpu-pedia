"""
ComfyUI Wan 2.1 TPU - 工具模块
==============================

包含:
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
# 视频生成默认参数
# ============================================================================

# 720P 默认参数
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FRAMES = 81
DEFAULT_FPS = 16
DEFAULT_FLOW_SHIFT = 5.0  # 5.0 for 720P, 3.0 for 480P


# ============================================================================
# Text Encoder 分片策略 (T5-XXL)
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
# 算子注册（仅在 USE_FULL_OPS_REGISTRATION=True 时使用）
# ============================================================================

_text_encoder_ops_registered = False
_operators_registered = False


def register_text_encoder_ops(env):
    """
    注册 T5/UMT5 Text Encoder 需要的算子。
    
    UMT5 模型使用以下特殊操作：
    - dropout: 推理时直接返回输入
    - minimum/maximum: 元素级 min/max（用于 relative position bucket）
    - min.other: torch.min(a, b) 元素级比较
    - clamp: 张量裁剪
    - abs: 绝对值
    - log: 对数
    - floor: 向下取整
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
    # 推理时直接返回输入（不做 dropout）
    def dropout_impl(input, p=0.5, training=False, inplace=False, env=env):
        # 推理模式：直接返回输入
        return input
    
    def native_dropout_impl(input, p, train, env=env):
        # 推理模式：返回 (input, 全1 mask)
        if hasattr(input, '_elem'):
            # XLA tensor
            mask = torch.ones(input.shape, dtype=torch.bool)
            return input, mask.to('jax')
        return input, torch.ones_like(input, dtype=torch.bool)
    
    try:
        override_op(torch.ops.aten.dropout.default, functools.partial(dropout_impl, env=env))
        override_op(torch.ops.aten.native_dropout.default, functools.partial(native_dropout_impl, env=env))
        print("  - Registered dropout operators")
    except Exception as e:
        print(f"  - Warning: Failed to register dropout: {e}")
    
    # ---- minimum/maximum (元素级) ----
    # UMT5 的 _relative_position_bucket 使用 torch.min(a, b) 进行元素级比较
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
        print("  - Registered minimum/maximum operators")
    except Exception as e:
        print(f"  - Warning: Failed to register minimum/maximum: {e}")
    
    # ---- min.other / max.other (两个 tensor 的元素级比较) ----
    # 当调用 torch.min(a, b) 且 b 是 tensor 时，PyTorch 使用 aten.min.other
    # torchax 默认的 aten.min 把第二个参数当作 dim，导致错误
    def min_other_impl(input, other, env=env):
        jinput = env.t2j_iso(input)
        jother = env.t2j_iso(other)
        return env.j2t_iso(jnp.minimum(jinput, jother))
    
    def max_other_impl(input, other, env=env):
        jinput = env.t2j_iso(input)
        jother = env.t2j_iso(other)
        return env.j2t_iso(jnp.maximum(jinput, jother))
    
    try:
        override_op(torch.ops.aten.min.other, functools.partial(min_other_impl, env=env))
        override_op(torch.ops.aten.max.other, functools.partial(max_other_impl, env=env))
        print("  - Registered min.other/max.other operators")
    except Exception as e:
        print(f"  - Warning: Failed to register min.other/max.other: {e}")
    
    # ---- clamp ----
    def clamp_impl(input, min_val=None, max_val=None, env=env):
        jinput = env.t2j_iso(input)
        if min_val is not None:
            jinput = jnp.maximum(jinput, min_val)
        if max_val is not None:
            jinput = jnp.minimum(jinput, max_val)
        return env.j2t_iso(jinput)
    
    def clamp_tensor_impl(input, min_tensor=None, max_tensor=None, env=env):
        jinput = env.t2j_iso(input)
        if min_tensor is not None:
            jmin = env.t2j_iso(min_tensor)
            jinput = jnp.maximum(jinput, jmin)
        if max_tensor is not None:
            jmax = env.t2j_iso(max_tensor)
            jinput = jnp.minimum(jinput, jmax)
        return env.j2t_iso(jinput)
    
    try:
        override_op(torch.ops.aten.clamp.default, functools.partial(clamp_impl, env=env))
        override_op(torch.ops.aten.clamp.Tensor, functools.partial(clamp_tensor_impl, env=env))
        print("  - Registered clamp operators")
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
    
    # ---- item (tensor -> scalar) ----
    # scheduler 使用 .item() 将单元素 tensor 转换为 Python 标量
    def item_impl(input, env=env):
        jinput = env.t2j_iso(input)
        # 获取标量值，保持正确的类型（int 或 float）
        scalar_val = np.array(jinput).item()
        # 保持原始类型：如果是整数类型，返回 int
        if jinput.dtype in (jnp.int32, jnp.int64, jnp.int16, jnp.int8, jnp.uint32, jnp.uint64):
            return int(scalar_val)
        return scalar_val  # 对于 float 类型，numpy.item() 返回正确类型
    
    try:
        override_op(torch.ops.aten.item.default, functools.partial(item_impl, env=env))
        override_op(torch.ops.aten._local_scalar_dense.default, functools.partial(item_impl, env=env))
        print("  - Registered item operator")
    except Exception as e:
        print(f"  - Warning: Failed to register item: {e}")
    
    _text_encoder_ops_registered = True


def register_operators_on_env(env, mesh_obj):
    """
    在 torchax 环境上注册 TPU 所需的自定义算子。
    
    注册的算子:
      - conv2d: 2D 卷积
      - conv3d: 3D 卷积
      - cartesian_prod: 笛卡尔积
      - chunk: 张量分块
      - layer_norm / native_layer_norm: 层归一化
      - unflatten: 维度展开
      - rms_norm: RMS 归一化
      - dropout / native_dropout: Dropout（推理时直接返回）
      - group_norm / native_group_norm: 组归一化
      - expand_as: 张量扩展（用于 F.normalize）
      - scaled_dot_product_attention: Splash Attention（可选）
    """
    global _operators_registered
    if _operators_registered:
        return
    
    # 延迟导入 torchax 组件
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
    
    # ---- conv3d ----
    # 使用 torchax 的通用 _aten_convolution 实现，支持 3D 卷积
    def conv3d_impl(input, weight, bias=None, stride=1, padding=0,
                    dilation=1, groups=1, *, env=env):
        """
        3D 卷积实现，用于 WanTransformer3DModel 的 patch_embedding。
        
        Args:
            input: [N, C, D, H, W] - 输入张量
            weight: [out_channels, in_channels/groups, kD, kH, kW] - 卷积核
            bias: [out_channels] - 偏置（可选）
            stride: 步长
            padding: 填充
            dilation: 膨胀
            groups: 组数
        
        Returns:
            [N, out_channels, D', H', W'] - 输出张量
        """
        jinput, jweight, jbias = env.t2j_iso((input, weight, bias))
        # 使用 torchax 的通用卷积实现
        res = jaten._aten_convolution(
            jinput, jweight, jbias,
            stride, padding, dilation,
            transposed=False,
            output_padding=1,  # 非转置卷积忽略此参数
            groups=groups
        )
        return env.j2t_iso(res)
    
    # 注册所有可能的 conv3d 变体（重要：必须全部注册！）
    print("  - Registering conv3d operator variants...")
    conv3d_fn = functools.partial(conv3d_impl, env=env)
    
    # 1. torch.nn.functional.conv3d
    override_op(torch.nn.functional.conv3d, conv3d_fn)
    print("    ✓ torch.nn.functional.conv3d")
    
    # 2. torch.ops.aten.conv3d (OpOverloadPacket) - 关键！
    try:
        override_op(torch.ops.aten.conv3d, conv3d_fn)
        print("    ✓ torch.ops.aten.conv3d")
    except Exception as e:
        print(f"    ✗ torch.ops.aten.conv3d: {e}")
    
    # 3. torch.ops.aten.conv3d.default (OpOverload)
    try:
        override_op(torch.ops.aten.conv3d.default, conv3d_fn)
        print("    ✓ torch.ops.aten.conv3d.default")
    except Exception as e:
        print(f"    ✗ torch.ops.aten.conv3d.default: {e}")
    
    # 4. torch.ops.aten.convolution (通用卷积接口)
    def convolution_impl(input, weight, bias=None, stride=1, padding=0, dilation=1,
                         transposed=False, output_padding=0, groups=1, *, env=env):
        """通用卷积实现，支持 2D 和 3D"""
        jinput, jweight, jbias = env.t2j_iso((input, weight, bias))
        res = jaten._aten_convolution(
            jinput, jweight, jbias,
            stride, padding, dilation,
            transposed, output_padding, groups
        )
        return env.j2t_iso(res)
    
    try:
        override_op(torch.ops.aten.convolution, functools.partial(convolution_impl, env=env))
        override_op(torch.ops.aten.convolution.default, functools.partial(convolution_impl, env=env))
        print("    ✓ torch.ops.aten.convolution")
    except Exception as e:
        print(f"    ✗ torch.ops.aten.convolution: {e}")
    
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
    
    # ---- expand_as (用于 F.normalize) ----
    def expand_as_impl(input, other, env=env):
        """
        将 input 扩展到与 other 相同的形状。
        用于 torch.nn.functional.normalize 中的 denom.expand_as(input)
        """
        jinput = env.t2j_iso(input)
        jother = env.t2j_iso(other)
        target_shape = jother.shape
        # 使用 JAX broadcast 扩展
        result = jnp.broadcast_to(jinput, target_shape)
        return env.j2t_iso(result)
    
    try:
        override_op(torch.ops.aten.expand_as.default, functools.partial(expand_as_impl, env=env))
        print("  - Registered expand_as operator")
    except Exception as e:
        print(f"  - Warning: Failed to register expand_as: {e}")
    
    # ---- Splash Attention ----
    try:
        try:
            from .splash_attention import sdpa_reference, tpu_splash_attention
        except ImportError:
            from splash_attention import sdpa_reference, tpu_splash_attention
        USE_K_SMOOTH = True
        
        def sdpa_tpu(query, key, value, attn_mask=None, dropout_p=0.0,
                     is_causal=False, scale=None, enable_gqa=False, env=env, mesh=mesh_obj):
            # 仅对长序列使用 Splash Attention
            if key.shape[2] > 20000:
                jquery, jkey, jvalue = env.t2j_iso((query, key, value))
                if USE_K_SMOOTH:
                    jkey = jkey - jnp.mean(jkey, axis=2, keepdims=True)
                res = tpu_splash_attention(jquery, jkey, jvalue, mesh, scale=scale)
                return env.j2t_iso(res)
            return sdpa_reference(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)
        
        override_op(torch.nn.functional.scaled_dot_product_attention,
                    functools.partial(sdpa_tpu, env=env, mesh=mesh_obj))
    except ImportError:
        pass
    
    _operators_registered = True
