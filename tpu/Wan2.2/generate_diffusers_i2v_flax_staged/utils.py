#!/usr/bin/env python3
"""
Wan 2.2 I2V 三阶段生成 - 共享工具模块

包含:
- 配置常量（模型、视频生成、VAE归一化参数）
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
MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

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
# 视频生成配置
# ============================================================================
FRAMES = 81
FPS = 16
NUM_STEPS = 40
GUIDANCE_SCALE = 3.5
BOUNDARY_RATIO = 0.9
SHIFT = 5.0  # Wan 2.2 I2V 默认 shift 值，影响时间步长分布

# 默认 Prompts
DEFAULT_PROMPT = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
DEFAULT_NEG_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
# 使用本地图像文件
DEFAULT_IMAGE_PATH = "/home/chrisya/gpu-tpu-pedia/tpu/Wan2.2/wan_i2v_input.JPG"


# ============================================================================
# VAE 归一化参数 - 动态从 VAE config 加载
# 以下是 Wan-AI/Wan2.2-I2V-A14B-Diffusers 的默认值，用于无 VAE 对象时
# ============================================================================
DEFAULT_LATENTS_MEAN = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
]
DEFAULT_LATENTS_STD = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916
]


# ============================================================================
# Splash Attention 配置
# ============================================================================
BQSIZE = 3328
BKVSIZE = 2816
BKVCOMPUTESIZE = 256
BKVCOMPUTEINSIZE = 256

# 是否使用 K-smooth
USE_K_SMOOTH = False

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

# Text Encoder sharding (UMT5-XXL)
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
# Latent 归一化函数 - 优先从 VAE config 读取参数
# ============================================================================

def get_latents_params(vae=None, mean=None, std=None):
    """
    获取 latent 归一化参数
    
    优先级：传入参数 > VAE config > 默认值
    
    Args:
        vae: VAE 模型实例（可选）
        mean: 直接传入的 mean 值（可选）
        std: 直接传入的 std 值（可选）
        
    Returns:
        (latents_mean, latents_std) 元组
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
    归一化 latents: (x - mean) * (1/std)
    
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
# SafeTensors 数据存储
# 设计决策: 保存时转为 float32，加载时转回 bfloat16（跨框架兼容性）
# ============================================================================

def save_embeddings_to_safetensors(embeddings_dict, output_path, metadata=None):
    """
    保存 embeddings 到 SafeTensors 格式
    
    自动处理 bfloat16 -> float32 转换以确保跨框架兼容性
    
    Args:
        embeddings_dict: 包含 tensor 的字典
        output_path: 输出文件路径
        metadata: 可选的元数据字典
    """
    tensors_to_save = {}
    dtype_info = {}
    
    for key, value in embeddings_dict.items():
        if isinstance(value, torch.Tensor):
            # 将 bfloat16 转换为 float32 以确保兼容性
            if value.dtype == torch.bfloat16:
                tensors_to_save[key] = value.to(torch.float32).cpu().contiguous()
                dtype_info[key] = 'bfloat16'
            else:
                tensors_to_save[key] = value.cpu().contiguous()
                dtype_info[key] = str(value.dtype)
        elif isinstance(value, (np.ndarray, jnp.ndarray)):
            np_array = np.array(value)
            if 'bfloat16' in str(value.dtype):
                tensors_to_save[key] = torch.from_numpy(np_array.astype(np.float32))
                dtype_info[key] = 'bfloat16'
            else:
                tensors_to_save[key] = torch.from_numpy(np_array)
                dtype_info[key] = str(np_array.dtype)
    
    # 添加 dtype 信息到 metadata
    final_metadata = metadata or {}
    final_metadata['dtype_info'] = json.dumps(dtype_info)
    
    # 保存
    torch_save_file(tensors_to_save, output_path, metadata=final_metadata)
    print(f"✓ 已保存 {len(tensors_to_save)} 个 tensors 到 {output_path}")
    
    return output_path


def load_embeddings_from_safetensors(input_path, device='cpu', restore_dtype=True):
    """
    从 SafeTensors 文件加载 embeddings
    
    自动恢复原始 dtype（如 bfloat16）
    
    Args:
        input_path: 输入文件路径
        device: 目标设备
        restore_dtype: 是否恢复原始 dtype
        
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
    """
    tensors_to_save = {}
    dtype_info = {}
    
    # 处理不同类型的输入
    if isinstance(latents, torch.Tensor):
        if latents.dtype == torch.bfloat16:
            tensors_to_save['latents'] = latents.to(torch.float32).cpu().contiguous()
            dtype_info['latents'] = 'bfloat16'
        else:
            tensors_to_save['latents'] = latents.cpu().contiguous()
            dtype_info['latents'] = str(latents.dtype)
    elif 'ArrayImpl' in str(type(latents)) or isinstance(latents, jnp.ndarray):
        # JAX array
        if latents.dtype == jnp.bfloat16:
            np_array = np.array(latents.astype(jnp.float32))
            tensors_to_save['latents'] = torch.from_numpy(np_array)
            dtype_info['latents'] = 'bfloat16'
        else:
            np_array = np.array(latents)
            tensors_to_save['latents'] = torch.from_numpy(np_array).contiguous()
            dtype_info['latents'] = str(np_array.dtype)
    else:
        raise TypeError(f"Unsupported latents type: {type(latents)}")
    
    # 添加 dtype 信息到 metadata
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
# PyTree 注册
# ============================================================================

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


# ============================================================================
# 数据转换工具
# ============================================================================

def to_torch_recursive(x):
    """
    递归地将 JAX 数组转换为 PyTorch 张量
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
    """
    递归地将 PyTorch 张量转换为 JAX 数组
    """
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


# ============================================================================
# 视频处理工具
# ============================================================================

def prepare_video_for_export(video, target_frames=None):
    """
    准备视频 tensor 用于导出
    
    Args:
        video: 视频 tensor [B, C, T, H, W] 或 [B, T, H, W, C]
        target_frames: 目标帧数（用于验证）
        
    Returns:
        list: float32 numpy array 列表 (0-1 范围)，适用于 export_to_video
    """
    if isinstance(video, (list, tuple)):
        return [prepare_video_for_export(v, target_frames) for v in video]
    
    if isinstance(video, torch.Tensor):
        video = video.cpu()
        
        # 处理不同的输入格式
        if video.dim() == 5:
            if video.shape[-1] == 3:  # [B, T, H, W, C]
                video = video.permute(0, 4, 1, 2, 3)
            # 现在是 [B, C, T, H, W]
            batch_vid = video[0]  # [C, T, H, W]
        elif video.dim() == 4:
            if video.shape[0] == 3:  # [C, T, H, W]
                batch_vid = video
            elif video.shape[-1] == 3:  # [T, H, W, C]
                batch_vid = video.permute(3, 0, 1, 2)
            else:
                raise ValueError(f"Unexpected 4D video shape: {video.shape}")
        else:
            raise ValueError(f"Unexpected video dimensions: {video.dim()}")
        
        # batch_vid: [C, T, H, W] -> [T, C, H, W]
        batch_vid = batch_vid.permute(1, 0, 2, 3)
        
        # Denormalize: [-1, 1] -> [0, 1]
        batch_vid = (batch_vid * 0.5 + 0.5).clamp(0, 1)
        
        # [T, C, H, W] -> [T, H, W, C]
        batch_vid = batch_vid.permute(0, 2, 3, 1)
        
        # Convert to float32 numpy
        video = batch_vid.float().numpy()
        
        if video.shape[-1] == 1:
            video = np.repeat(video, 3, axis=-1)
        
        return [video[i] for i in range(video.shape[0])]
    
    if isinstance(video, np.ndarray):
        # 处理 numpy array
        if video.ndim == 5:
            if video.shape[-1] == 3:  # [B, T, H, W, C]
                video = np.transpose(video, (0, 4, 1, 2, 3))
            batch_vid = video[0]  # [C, T, H, W]
        elif video.ndim == 4:
            if video.shape[0] == 3:  # [C, T, H, W]
                batch_vid = video
            elif video.shape[-1] == 3:  # [T, H, W, C]
                batch_vid = np.transpose(video, (3, 0, 1, 2))
            else:
                raise ValueError(f"Unexpected 4D video shape: {video.shape}")
        else:
            raise ValueError(f"Unexpected video dimensions: {video.ndim}")
        
        # [C, T, H, W] -> [T, C, H, W]
        batch_vid = np.transpose(batch_vid, (1, 0, 2, 3))
        
        # Denormalize if needed
        if batch_vid.min() < 0:
            batch_vid = np.clip(batch_vid * 0.5 + 0.5, 0, 1)
        
        # [T, C, H, W] -> [T, H, W, C]
        batch_vid = np.transpose(batch_vid, (0, 2, 3, 1))
        
        video = batch_vid.astype(np.float32)
        
        if video.shape[-1] == 1:
            video = np.repeat(video, 3, axis=-1)
        
        return [video[i] for i in range(video.shape[0])]
    
    return video


# ============================================================================
# JAX 配置辅助
# ============================================================================

def setup_jax_cache():
    """设置 JAX 编译缓存（重要：避免重复编译）"""
    jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
    print("✓ JAX 编译缓存已启用: /dev/shm/jax_cache")


def pad_to_multiple(x, multiple, axis):
    """Pad array to next multiple along axis."""
    seq_len = x.shape[axis]
    pad_len = (multiple - seq_len % multiple) % multiple
    if pad_len == 0:
        return x, seq_len
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, pad_len)
    return jnp.pad(x, pad_width), seq_len