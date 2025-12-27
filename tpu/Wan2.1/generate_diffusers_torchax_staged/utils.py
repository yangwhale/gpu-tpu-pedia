#!/usr/bin/env python3
"""
Wan 2.1 三阶段生成 - 共享工具模块

包含:
- 配置常量（模型、视频生成）
- Splash Attention 配置
- Sharding 策略
- SafeTensors 数据存储
- 配置管理
- PyTree 注册
- 辅助工具函数
"""

import os
import re
import json
import numpy as np
import jax
import jax.numpy as jnp
import torch
from safetensors import safe_open
from safetensors.torch import save_file as torch_save_file, load_file as torch_load_file
from jax.tree_util import register_pytree_node
from jax.sharding import PartitionSpec as P, NamedSharding


# ============================================================================
# 模型配置
# ============================================================================
MODEL_NAME = "Wan-AI/Wan2.1-T2V-14B-Diffusers"

# === Video Generation Settings (720P) ===
FLOW_SHIFT = 5.0  # 5.0 for 720P, 3.0 for 480P
WIDTH = 1280
HEIGHT = 720
FRAMES = 81
FPS = 16
NUM_STEPS = 50


# ============================================================================
# Splash Attention 配置
# ============================================================================
BQSIZE = 3328           # Query 块大小
BKVSIZE = 2816          # Key/Value 块大小
BKVCOMPUTESIZE = 256    # Key/Value 计算块大小
BKVCOMPUTEINSIZE = 256  # Key/Value 内部计算块大小

# 是否使用 K-smooth
USE_K_SMOOTH = True

# 是否使用 custom splash attention (exp2 优化)
USE_CUSTOM_ATTENTION = True

# LOG2_E 常量，用于 exp2 优化
LOG2_E = 1.44269504


# ============================================================================
# Mesh 配置
# ============================================================================
DEFAULT_DP = 2


# ============================================================================
# Sharding 策略
# ============================================================================

# Text Encoder sharding (T5-XXL)
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

# Transformer sharding (WanTransformer3DModel)
TRANSFORMER_SHARDINGS = {
    r'condition_embedder\.time_embedder\.linear_1\.weight': ('tp',),
    r'condition_embedder\.time_embedder\.linear_1\.bias': ('tp',),
    r'condition_embedder\.time_embedder\.linear_2\.weight': (None, 'tp',),
    r'condition_embedder\.text_embedder\.linear_1\.weight': ('tp',),
    r'condition_embedder\.text_embedder\.linear_1\.bias': ('tp',),
    r'condition_embedder\.text_embedder\.linear_2\.weight': (None, 'tp',),
    r'blocks\.\d+\.attn1\.to_q\.weight': ('tp',),
    r'blocks\.\d+\.attn1\.to_q\.bias': ('tp',),
    r'blocks\.\d+\.attn1\.to_k\.weight': ('tp',),
    r'blocks\.\d+\.attn1\.to_k\.bias': ('tp',),
    r'blocks\.\d+\.attn1\.to_v\.weight': ('tp',),
    r'blocks\.\d+\.attn1\.to_v\.bias': ('tp',),
    r'blocks\.\d+\.attn1\.to_out\.\d+\.weight': (None, 'tp',),
    r'blocks\.\d+\.attn2\.to_q\.weight': ('tp',),
    r'blocks\.\d+\.attn2\.to_q\.bias': ('tp',),
    r'blocks\.\d+\.attn2\.to_k\.weight': ('tp',),
    r'blocks\.\d+\.attn2\.to_k\.bias': ('tp',),
    r'blocks\.\d+\.attn2\.to_v\.weight': ('tp',),
    r'blocks\.\d+\.attn2\.to_v\.bias': ('tp',),
    r'blocks\.\d+\.attn2\.to_out\.\d+\.weight': (None, 'tp',),
    r'blocks\.\d+\.ffn\.net\.\d+\.proj\.weight': ('tp',),
    r'blocks\.\d+\.ffn\.net\.\d+\.proj\.bias': ('tp',),
    r'blocks\.\d+\.ffn\.net\.\d+\.weight': (None, 'tp',),
}

# VAE sharding (空字典 - 不分片，使用 replicate)
VAE_ENCODER_SHARDINGS = {}
VAE_DECODER_SHARDINGS = {}


# ============================================================================
# 权重分片函数
# ============================================================================

def shard_weight_dict(weight_dict, sharding_dict, mesh):
    """Apply sharding to weights based on pattern matching."""
    result = {}
    for k, v in weight_dict.items():
        if isinstance(v, torch.Tensor):
            v = v.to("jax")
        
        matched = False
        for target, sharding in sharding_dict.items():
            if re.fullmatch(target, k) is not None:
                v.apply_jax_(jax.device_put, NamedSharding(mesh, P(*sharding)))
                matched = True
                break
        
        if not matched:
            # Replicate
            v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
        
        result[k] = v
    return result


# ============================================================================
# 数据转换工具
# ============================================================================

def to_torch_recursive(x):
    """递归地将 JAX 数组转换为 PyTorch 张量
    
    Note: 检查顺序很重要：
    1. 首先检查 JAX array
    2. 然后检查 .sample 属性（优先于 dict，因为 dataclass 可能也是 dict-like）
    3. 最后检查 list/tuple/dict
    """
    if 'ArrayImpl' in str(type(x)) or isinstance(x, jnp.ndarray):
        np_array = np.array(x)
        if hasattr(x, 'dtype') and x.dtype == jnp.bfloat16:
            return torch.from_numpy(np_array.astype(np.float32)).to(torch.bfloat16)
        else:
            return torch.from_numpy(np_array)
    elif hasattr(x, 'sample'):
        # Handle objects with .sample attribute (like DecoderOutput)
        sample = to_torch_recursive(x.sample)
        if hasattr(x, 'replace'):
            return x.replace(sample=sample)
        else:
            return sample
    elif isinstance(x, (list, tuple)):
        return type(x)(to_torch_recursive(xx) for xx in x)
    elif isinstance(x, dict):
        return {k: to_torch_recursive(v) for k, v in x.items()}
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
    注册必要的 PyTree 节点以支持 JAX 转换
    """
    from transformers import modeling_outputs
    from diffusers.models.autoencoders import vae as diffusers_vae
    from diffusers.models import modeling_outputs as diffusers_modeling_outputs
    
    print("注册 PyTree 节点...")
    
    def flatten_model_output(obj):
        return obj.to_tuple(), type(obj)
    
    def unflatten_model_output(aux, children):
        return aux(*children)
    
    # Text encoder output
    try:
        register_pytree_node(
            modeling_outputs.BaseModelOutputWithPastAndCrossAttentions,
            flatten_model_output,
            unflatten_model_output
        )
        print("  - BaseModelOutputWithPastAndCrossAttentions 已注册")
    except ValueError:
        print("  - BaseModelOutputWithPastAndCrossAttentions 已存在")
    
    # VAE decode output
    try:
        register_pytree_node(
            diffusers_vae.DecoderOutput,
            flatten_model_output,
            unflatten_model_output
        )
        print("  - DecoderOutput 已注册")
    except ValueError:
        print("  - DecoderOutput 已存在")
    
    # VAE encode output
    try:
        register_pytree_node(
            diffusers_modeling_outputs.AutoencoderKLOutput,
            flatten_model_output,
            unflatten_model_output
        )
        print("  - AutoencoderKLOutput 已注册")
    except ValueError:
        print("  - AutoencoderKLOutput 已存在")
    
    # DiagonalGaussianDistribution
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


def sharded_device_put(tensor, sharding):
    """Put tensor on devices with proper sharding for multi-host setups."""
    import jax
    
    if isinstance(tensor, tuple):
        return tuple(sharded_device_put(t, sharding) for t in tensor)
    
    num_global_devices = jax.device_count()
    num_local_devices = jax.local_device_count()

    if num_global_devices == num_local_devices:
        return jax.device_put(tensor, sharding)

    shape = tensor.shape
    x_split = [
        jax.device_put(tensor[i], device)
        for device, i in sharding.addressable_devices_indices_map(shape).items()
    ]
    return jax.make_array_from_single_device_arrays(shape, sharding, x_split)


def prepare_video_for_export(video, target_frames):
    """Prepare video tensor for export to file.
    
    完全模仿 VideoProcessor.postprocess_video 的处理方式：
    1. 输入: [B, T, H, W, C] (JAX VAE 输出格式)
    2. 转换为 [B, C, T, H, W]
    3. VideoProcessor 处理:
       - [B, C, T, H, W] -> 取 batch[0] -> [C, T, H, W]
       - permute(1, 0, 2, 3) -> [T, C, H, W]
       - denormalize: (x * 0.5 + 0.5).clamp(0, 1)
       - pt_to_numpy: permute(0, 2, 3, 1) -> [T, H, W, C]
    4. 返回 float32 [0, 1] 范围 (与 VideoProcessor 一致)
    
    Args:
        video: 视频 tensor 或 numpy array
            - 期望输入格式: [B, T, H, W, C] (JAX VAE 输出)
        target_frames: 目标帧数（用于验证）
        
    Returns:
        numpy array: [T, H, W, C] 格式的 float32 视频 (范围 [0, 1])
    """
    if isinstance(video, (list, tuple)):
        return [prepare_video_for_export(v, target_frames) for v in video]
    
    if isinstance(video, torch.Tensor):
        # JAX VAE 输出格式是 [B, T, H, W, C]
        # 模仿 VideoProcessor 的处理方式
        
        if video.dim() == 5:
            if video.shape[-1] == 3:  # [B, T, H, W, C]
                # 转换为 [B, C, T, H, W] (VideoProcessor 期望的格式)
                video = video.permute(0, 4, 1, 2, 3)
            # else: 已经是 [B, C, T, H, W]
            
            # 取第一个 batch: [C, T, H, W]
            batch_vid = video[0]
            
            # permute(1, 0, 2, 3): [C, T, H, W] -> [T, C, H, W]
            batch_vid = batch_vid.permute(1, 0, 2, 3)
            
            # denormalize: [-1, 1] -> [0, 1]
            batch_vid = (batch_vid * 0.5 + 0.5).clamp(0, 1)
            
            # pt_to_numpy with permute(0, 2, 3, 1): [T, C, H, W] -> [T, H, W, C]
            video = batch_vid.cpu().permute(0, 2, 3, 1).float().numpy()
            
        elif video.dim() == 4:
            # 可能是 [C, T, H, W] 或 [T, H, W, C]
            if video.shape[0] == 3:  # [C, T, H, W]
                batch_vid = video.permute(1, 0, 2, 3)  # -> [T, C, H, W]
                batch_vid = (batch_vid * 0.5 + 0.5).clamp(0, 1)
                video = batch_vid.cpu().permute(0, 2, 3, 1).float().numpy()
            elif video.shape[-1] == 3:  # [T, H, W, C]
                # 已经是正确格式，直接 denormalize
                video = (video * 0.5 + 0.5).clamp(0, 1)
                video = video.cpu().float().numpy()
        
        if video.shape[-1] == 1:
            video = np.repeat(video, 3, axis=-1)
        return video
    
    if isinstance(video, np.ndarray):
        # numpy array 处理
        if video.ndim == 5:
            if video.shape[-1] == 3:  # [B, T, H, W, C]
                video = np.transpose(video, (0, 4, 1, 2, 3))  # -> [B, C, T, H, W]
            
            batch_vid = video[0]  # [C, T, H, W]
            batch_vid = np.transpose(batch_vid, (1, 0, 2, 3))  # -> [T, C, H, W]
            
            if batch_vid.min() < 0:  # [-1, 1] 范围
                batch_vid = np.clip(batch_vid * 0.5 + 0.5, 0, 1)
            
            video = np.transpose(batch_vid, (0, 2, 3, 1))  # -> [T, H, W, C]
            
        elif video.ndim == 4:
            if video.shape[0] == 3:  # [C, T, H, W]
                video = np.transpose(video, (1, 0, 2, 3))  # -> [T, C, H, W]
                if video.min() < 0:
                    video = np.clip(video * 0.5 + 0.5, 0, 1)
                video = np.transpose(video, (0, 2, 3, 1))  # -> [T, H, W, C]
            elif video.shape[-1] == 3:  # [T, H, W, C]
                if video.min() < 0:
                    video = np.clip(video * 0.5 + 0.5, 0, 1)
        
        video = video.astype(np.float32)
        
        if video.shape[-1] == 1:
            video = np.repeat(video, 3, axis=-1)
        return video
    
    return video


# === SafeTensors 数据存储 ===

def save_embeddings_to_safetensors(embeddings_dict, output_path, metadata=None):
    """
    将 embeddings 字典保存为 SafeTensors 格式
    
    Args:
        embeddings_dict: 包含 tensor 的字典
        output_path: 输出文件路径
        metadata: 可选的元数据字典
    """
    # 转换为可保存的格式
    tensors_to_save = {}
    dtype_info = {}
    
    for key, value in embeddings_dict.items():
        if isinstance(value, torch.Tensor):
            # 将 bfloat16 转换为 float32 以便保存（safetensors 对 bf16 支持有限）
            if value.dtype == torch.bfloat16:
                tensors_to_save[key] = value.to(torch.float32).cpu()
                dtype_info[key] = 'bfloat16'
            else:
                tensors_to_save[key] = value.cpu()
                dtype_info[key] = str(value.dtype)
        elif isinstance(value, (np.ndarray, jnp.ndarray)):
            np_array = np.array(value)
            if np_array.dtype == np.float16 or 'bfloat16' in str(value.dtype):
                tensors_to_save[key] = torch.from_numpy(np_array.astype(np.float32))
                dtype_info[key] = 'bfloat16'
            else:
                tensors_to_save[key] = torch.from_numpy(np_array)
                dtype_info[key] = str(np_array.dtype)
    
    # 添加元数据
    final_metadata = metadata or {}
    final_metadata['dtype_info'] = json.dumps(dtype_info)
    
    # 保存
    torch_save_file(tensors_to_save, output_path, metadata=final_metadata)
    print(f"✓ 已保存 {len(tensors_to_save)} 个 tensors 到 {output_path}")
    
    return output_path


def load_embeddings_from_safetensors(input_path, device='cpu', restore_dtype=True):
    """
    从 SafeTensors 文件加载 embeddings
    
    Args:
        input_path: 输入文件路径
        device: 目标设备
        restore_dtype: 是否恢复原始 dtype（如 bfloat16）
        
    Returns:
        embeddings_dict: 包含 tensor 的字典
        metadata: 元数据字典
    """
    # 加载 tensors
    tensors = torch_load_file(input_path, device=device)
    
    # 加载元数据
    metadata = {}
    dtype_info = {}
    with safe_open(input_path, framework="pt") as f:
        raw_metadata = f.metadata()
        if raw_metadata:
            metadata = dict(raw_metadata)
            if 'dtype_info' in metadata:
                dtype_info = json.loads(metadata['dtype_info'])
    
    # 恢复原始 dtype
    if restore_dtype:
        for key in tensors:
            if key in dtype_info and dtype_info[key] == 'bfloat16':
                tensors[key] = tensors[key].to(torch.bfloat16)
    
    print(f"✓ 已加载 {len(tensors)} 个 tensors 从 {input_path}")
    
    return tensors, metadata


def save_latents_to_safetensors(latents, output_path, metadata=None):
    """
    将 latents 保存为 SafeTensors 格式
    
    Args:
        latents: latents tensor (torch.Tensor 或 JAX array)
        output_path: 输出文件路径
        metadata: 可选的元数据字典
        
    注意：safetensors 原生支持 bfloat16，直接保存无需转换
    """
    tensors_to_save = {}
    dtype_info = {}
    
    # 处理不同类型的输入
    if isinstance(latents, torch.Tensor):
        # safetensors 原生支持 bfloat16，直接保存
        # 需要 .contiguous() 确保内存连续
        tensors_to_save['latents'] = latents.cpu().contiguous()
        dtype_info['latents'] = str(latents.dtype)
    elif 'ArrayImpl' in str(type(latents)) or isinstance(latents, jnp.ndarray):
        # JAX array 需要经过 numpy，但 numpy 不支持 bfloat16
        if latents.dtype == jnp.bfloat16:
            # 先转 float32 再转 torch，最后转回 bfloat16
            np_array = np.array(latents.astype(jnp.float32))
            tensors_to_save['latents'] = torch.from_numpy(np_array).to(torch.bfloat16)
            dtype_info['latents'] = 'torch.bfloat16'
        else:
            np_array = np.array(latents)
            tensors_to_save['latents'] = torch.from_numpy(np_array).contiguous()
            dtype_info['latents'] = str(np_array.dtype)
    else:
        raise TypeError(f"Unsupported latents type: {type(latents)}")
    
    # 添加元数据
    final_metadata = metadata or {}
    final_metadata['dtype_info'] = json.dumps(dtype_info)
    
    # 保存
    torch_save_file(tensors_to_save, output_path, metadata=final_metadata)
    print(f"✓ 已保存 latents 到 {output_path}")
    print(f"  shape: {tensors_to_save['latents'].shape}")
    print(f"  dtype: {tensors_to_save['latents'].dtype}")
    
    return output_path


def load_latents_from_safetensors(input_path, device='cpu', restore_dtype=True):
    """
    从 SafeTensors 文件加载 latents
    
    Args:
        input_path: 输入文件路径
        device: 目标设备
        restore_dtype: 是否恢复原始 dtype
        
    Returns:
        latents: latents tensor
        metadata: 元数据字典
    """
    tensors, metadata = load_embeddings_from_safetensors(input_path, device, restore_dtype)
    latents = tensors['latents']
    print(f"  latents shape: {latents.shape}, dtype: {latents.dtype}")
    return latents, metadata


# ============================================================================
# 配置管理
# ============================================================================

def save_generation_config(config_dict, output_path):
    """保存生成配置到 JSON 文件"""
    with open(output_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"✓ 已保存配置到 {output_path}")


def load_generation_config(input_path):
    """从 JSON 文件加载生成配置"""
    with open(input_path, 'r') as f:
        config = json.load(f)
    print(f"✓ 已加载配置从 {input_path}")
    return config


# ============================================================================
# 默认输出路径
# ============================================================================

def get_default_paths(output_dir="./stage_outputs"):
    """获取默认的中间文件路径"""
    os.makedirs(output_dir, exist_ok=True)
    return {
        'embeddings': os.path.join(output_dir, 'stage1_embeddings.safetensors'),
        'latents': os.path.join(output_dir, 'stage2_latents.safetensors'),
        'config': os.path.join(output_dir, 'generation_config.json'),
        'video': os.path.join(output_dir, 'output_video.mp4'),
    }


# ============================================================================
# JAX 配置辅助
# ============================================================================

def setup_jax_cache():
    """设置 JAX 编译缓存（重要：避免重复编译）"""
    jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax_cache"))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
    print("✓ JAX 编译缓存已启用:", os.path.expanduser("~/.cache/jax_cache"))


def pad_to_multiple(x, multiple, axis):
    """Pad array to next multiple along axis."""
    seq_len = x.shape[axis]
    pad_len = (multiple - seq_len % multiple) % multiple
    if pad_len == 0:
        return x, seq_len
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, pad_len)
    return jnp.pad(x, pad_width), seq_len