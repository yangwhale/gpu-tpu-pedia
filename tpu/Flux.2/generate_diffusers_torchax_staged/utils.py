#!/usr/bin/env python3
"""
Flux.2 三阶段生成 - 共享工具模块

包含:
- 配置常量（模型、图像生成）
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
MODEL_NAME = "black-forest-labs/FLUX.2-dev"

# === Image Generation Settings ===
WIDTH = 1024
HEIGHT = 1024
NUM_STEPS = 50
GUIDANCE_SCALE = 4.0  # Embedded CFG

# === Default Prompt ===
DEFAULT_PROMPT = ("Realistic macro photograph of a hermit crab using a soda can as its shell, "
                  "partially emerging from the can, captured with sharp detail and natural colors, "
                  "on a sunlit beach with soft shadows and a shallow depth of field, "
                  "with blurred ocean waves in the background. "
                  "The can has the text `BFL Diffusers` on it and it has a color gradient "
                  "that start with #FF5733 at the top and transitions to #33FF57 at the bottom.")


# ============================================================================
# Splash Attention 配置
# ============================================================================

# 是否使用 K-smooth
USE_K_SMOOTH = True


# ============================================================================
# Sharding 策略 (只用 tp，不用 dp)
# ============================================================================

# Flux2Transformer 分片（1D mesh: tp）
# 规则：输出投影用 ('tp', None)，输入投影用 (None, 'tp')
TRANSFORMER_SHARDINGS = {
    # Double-stream Transformer Blocks (transformer_blocks.*)
    # Attention: Flux2Attention
    r'transformer_blocks.*.attn.to_q.weight': ('tp', None),
    r'transformer_blocks.*.attn.to_k.weight': ('tp', None),
    r'transformer_blocks.*.attn.to_v.weight': ('tp', None),
    r'transformer_blocks.*.attn.to_out.0.weight': (None, 'tp'),
    r'transformer_blocks.*.attn.add_q_proj.weight': ('tp', None),
    r'transformer_blocks.*.attn.add_k_proj.weight': ('tp', None),
    r'transformer_blocks.*.attn.add_v_proj.weight': ('tp', None),
    r'transformer_blocks.*.attn.to_add_out.weight': (None, 'tp'),
    # FeedForward: Flux2FeedForward
    r'transformer_blocks.*.ff.linear_in.weight': ('tp', None),
    r'transformer_blocks.*.ff.linear_out.weight': (None, 'tp'),
    r'transformer_blocks.*.ff_context.linear_in.weight': ('tp', None),
    r'transformer_blocks.*.ff_context.linear_out.weight': (None, 'tp'),
    
    # Single-stream Transformer Blocks (single_transformer_blocks.*)
    # Parallel Self-Attention: Flux2ParallelSelfAttention
    r'single_transformer_blocks.*.attn.to_qkv_mlp_proj.weight': ('tp', None),
    r'single_transformer_blocks.*.attn.to_out.weight': (None, 'tp'),
    
    # Input/Output Projections
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

# Text Encoder (Mistral3ForConditionalGeneration) 分片 - 使用 tp 维度
# Mistral3 有 language_model 和 vision_tower 两部分
# 大权重需要分片：embed_tokens (1.3GB), lm_head (1.3GB), mlp layers (335MB each)
TEXT_ENCODER_SHARDINGS = {
    # Language Model - Embedding
    r'model\.language_model\.embed_tokens\.weight': ('tp', None),
    # LM Head
    r'lm_head\.weight': ('tp', None),
    # Language Model - Attention projections
    r'model\.language_model\.layers\.\d+\.self_attn\.q_proj\.weight': ('tp', None),
    r'model\.language_model\.layers\.\d+\.self_attn\.k_proj\.weight': ('tp', None),
    r'model\.language_model\.layers\.\d+\.self_attn\.v_proj\.weight': ('tp', None),
    r'model\.language_model\.layers\.\d+\.self_attn\.o_proj\.weight': (None, 'tp'),
    # Language Model - MLP
    r'model\.language_model\.layers\.\d+\.mlp\.gate_proj\.weight': ('tp', None),
    r'model\.language_model\.layers\.\d+\.mlp\.up_proj\.weight': ('tp', None),
    r'model\.language_model\.layers\.\d+\.mlp\.down_proj\.weight': (None, 'tp'),
    # Vision Tower - Attention projections
    r'model\.vision_tower\.transformer\.layers\.\d+\.attention\.q_proj\.weight': ('tp', None),
    r'model\.vision_tower\.transformer\.layers\.\d+\.attention\.k_proj\.weight': ('tp', None),
    r'model\.vision_tower\.transformer\.layers\.\d+\.attention\.v_proj\.weight': ('tp', None),
    r'model\.vision_tower\.transformer\.layers\.\d+\.attention\.o_proj\.weight': (None, 'tp'),
    # Vision Tower - MLP
    r'model\.vision_tower\.transformer\.layers\.\d+\.feed_forward\.gate_proj\.weight': ('tp', None),
    r'model\.vision_tower\.transformer\.layers\.\d+\.feed_forward\.up_proj\.weight': ('tp', None),
    r'model\.vision_tower\.transformer\.layers\.\d+\.feed_forward\.down_proj\.weight': (None, 'tp'),
}

# VAE sharding (空字典 - 不分片，使用 replicate)
VAE_ENCODER_SHARDINGS = {}
VAE_DECODER_SHARDINGS = {}


# ============================================================================
# 权重分片函数
# ============================================================================

def shard_weight_dict(weight_dict, sharding_dict, mesh, debug=False):
    """
    Apply sharding to weights based on pattern matching.
    
    对于 torchax tensor（已经是 XLA 格式）：直接 device_put 分片
    对于 PyTorch tensor（仍在 CPU）：在 CPU 上转换后再 device_put 分片
    """
    result = {}
    sharded_count = 0
    replicated_count = 0
    sharded_bytes = 0
    replicated_bytes = 0
    
    for k, v in weight_dict.items():
        # 获取 tensor 大小（用于统计）
        if hasattr(v, 'shape'):
            tensor_bytes = np.prod(v.shape) * 2  # 假设 bf16 = 2 bytes
        else:
            tensor_bytes = 0
            
        if isinstance(v, torch.Tensor):
            # 在 CPU 上转换为 JAX，然后 device_put 到 TPU
            with jax.default_device("cpu"):
                v = v.to("jax")
        
        matched = False
        for target, sharding in sharding_dict.items():
            if re.fullmatch(target, k) is not None:
                v.apply_jax_(jax.device_put, NamedSharding(mesh, P(*sharding)))
                matched = True
                sharded_count += 1
                sharded_bytes += tensor_bytes
                if debug:
                    print(f"  ✓ SHARDED: {k} -> {sharding}")
                break
        
        if not matched:
            # Replicate
            v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
            replicated_count += 1
            replicated_bytes += tensor_bytes
            if debug:
                print(f"  ○ REPLICATE: {k}")
        
        result[k] = v
    
    print(f"  分片统计: {sharded_count} 个分片 ({sharded_bytes/1e9:.2f}GB), "
          f"{replicated_count} 个复制 ({replicated_bytes/1e9:.2f}GB)")
    
    return result


def move_module_to_xla(env, module):
    """
    将模块权重转换为 torchax tensor 格式（仍在 CPU 内存中）。
    
    注意：这个函数只是转换格式，不会 device_put 到 TPU。
    真正的 device_put 发生在 shard_weight_dict 调用时。
    """
    with jax.default_device("cpu"):
        state_dict = module.state_dict()
        state_dict = env.to_xla(state_dict)
        module.load_state_dict(state_dict, assign=True)


# ============================================================================
# 数据转换工具
# ============================================================================

def to_torch_recursive(x):
    """递归地将 JAX 数组转换为 PyTorch 张量"""
    if 'ArrayImpl' in str(type(x)) or isinstance(x, jnp.ndarray):
        np_array = np.array(x)
        if hasattr(x, 'dtype') and x.dtype == jnp.bfloat16:
            return torch.from_numpy(np_array.astype(np.float32)).to(torch.bfloat16)
        else:
            return torch.from_numpy(np_array)
    elif hasattr(x, 'sample'):
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


# ============================================================================
# SafeTensors 数据存储
# ============================================================================

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
            # 将 bfloat16 转换为 float32 以便保存
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
    """
    tensors_to_save = {}
    dtype_info = {}
    
    # 处理不同类型的输入
    if isinstance(latents, torch.Tensor):
        tensors_to_save['latents'] = latents.cpu().contiguous()
        dtype_info['latents'] = str(latents.dtype)
    elif 'ArrayImpl' in str(type(latents)) or isinstance(latents, jnp.ndarray):
        if latents.dtype == jnp.bfloat16:
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
        'image': os.path.join(output_dir, 'output_image.png'),
    }


# ============================================================================
# JAX 配置辅助
# ============================================================================

def setup_jax_cache():
    """设置 JAX 编译缓存（重要：避免重复编译）"""
    jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax_cache"))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
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
