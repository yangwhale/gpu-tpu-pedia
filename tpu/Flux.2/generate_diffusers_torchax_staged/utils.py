#!/usr/bin/env python3
"""
Flux.2 三阶段生成 - 共享工具模块

包含配置常量、分片策略、数据存储和辅助函数。
"""

import json
import os
import re

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.tree_util import register_pytree_node
from safetensors import safe_open
from safetensors.torch import load_file as torch_load_file
from safetensors.torch import save_file as torch_save_file

# ============================================================================
# 配置常量
# ============================================================================

MODEL_NAME = "black-forest-labs/FLUX.2-dev"
WIDTH, HEIGHT = 1024, 1024
NUM_STEPS = 50
GUIDANCE_SCALE = 4.0  # Embedded CFG
USE_K_SMOOTH = True

DEFAULT_PROMPT = (
    "Realistic macro photograph of a hermit crab using a soda can as its shell, "
    "partially emerging from the can, captured with sharp detail and natural colors, "
    "on a sunlit beach with soft shadows and a shallow depth of field, "
    "with blurred ocean waves in the background. "
    "The can has the text `BFL Diffusers` on it and it has a color gradient "
    "that start with #FF5733 at the top and transitions to #33FF57 at the bottom."
)

# Flux.2 pipeline 使用的 system message（来自 diffusers/pipelines/flux2/system_messages.py）
SYSTEM_MESSAGE = """You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object
attribution and actions without speculation."""


# ============================================================================
# Transformer 分片策略 (1D mesh: tp)
# 规则：输出投影 ('tp', None)，输入投影 (None, 'tp')
# ============================================================================

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
    """按模式匹配应用权重分片。"""
    result = {}
    sharded_count = replicated_count = 0
    sharded_bytes = replicated_bytes = 0
    
    for k, v in weight_dict.items():
        tensor_bytes = np.prod(v.shape) * 2 if hasattr(v, 'shape') else 0
            
        if isinstance(v, torch.Tensor):
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
            v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
            replicated_count += 1
            replicated_bytes += tensor_bytes
        
        result[k] = v
    
    print(f"  分片统计: {sharded_count} 个分片 ({sharded_bytes/1e9:.2f}GB), "
          f"{replicated_count} 个复制 ({replicated_bytes/1e9:.2f}GB)")
    return result


def move_module_to_xla(env, module):
    """将模块权重转换为 torchax tensor 格式。"""
    with jax.default_device("cpu"):
        state_dict = module.state_dict()
        state_dict = env.to_xla(state_dict)
        module.load_state_dict(state_dict, assign=True)


# ============================================================================
# PyTree 注册
# ============================================================================

def setup_pytree_registrations():
    """注册必要的 PyTree 节点以支持 JAX 转换。"""
    from diffusers.models.autoencoders import vae as diffusers_vae
    from diffusers.models import modeling_outputs as diffusers_modeling_outputs
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


# ============================================================================
# SafeTensors 数据存储
# ============================================================================

def save_embeddings_to_safetensors(embeddings_dict, output_path, metadata=None):
    """将 embeddings 字典保存为 SafeTensors 格式。"""
    tensors_to_save = {}
    dtype_info = {}
    
    for key, value in embeddings_dict.items():
        if isinstance(value, torch.Tensor):
            if value.dtype == torch.bfloat16:
                tensors_to_save[key] = value.to(torch.float32).cpu()
                dtype_info[key] = 'bfloat16'
            else:
                tensors_to_save[key] = value.cpu()
                dtype_info[key] = str(value.dtype)
        elif isinstance(value, (np.ndarray, jnp.ndarray)):
            np_array = np.array(value)
            if 'bfloat16' in str(value.dtype):
                tensors_to_save[key] = torch.from_numpy(np_array.astype(np.float32))
                dtype_info[key] = 'bfloat16'
            else:
                tensors_to_save[key] = torch.from_numpy(np_array)
                dtype_info[key] = str(np_array.dtype)
    
    final_metadata = metadata or {}
    final_metadata['dtype_info'] = json.dumps(dtype_info)
    torch_save_file(tensors_to_save, output_path, metadata=final_metadata)
    print(f"✓ 已保存 {len(tensors_to_save)} 个 tensors 到 {output_path}")
    return output_path


def load_embeddings_from_safetensors(input_path, device='cpu', restore_dtype=True):
    """从 SafeTensors 文件加载 embeddings。"""
    tensors = torch_load_file(input_path, device=device)
    
    metadata = {}
    dtype_info = {}
    with safe_open(input_path, framework="pt") as f:
        raw_metadata = f.metadata()
        if raw_metadata:
            metadata = dict(raw_metadata)
            if 'dtype_info' in metadata:
                dtype_info = json.loads(metadata['dtype_info'])
    
    if restore_dtype:
        for key in tensors:
            if key in dtype_info and dtype_info[key] == 'bfloat16':
                tensors[key] = tensors[key].to(torch.bfloat16)
    
    print(f"✓ 已加载 {len(tensors)} 个 tensors 从 {input_path}")
    return tensors, metadata


def save_latents_to_safetensors(latents, output_path, metadata=None):
    """将 latents 保存为 SafeTensors 格式。"""
    tensors_to_save = {}
    dtype_info = {}
    
    if isinstance(latents, torch.Tensor):
        tensors_to_save['latents'] = latents.cpu().contiguous()
        dtype_info['latents'] = str(latents.dtype)
    elif isinstance(latents, jnp.ndarray) or 'ArrayImpl' in str(type(latents)):
        if latents.dtype == jnp.bfloat16:
            np_array = np.array(latents.astype(jnp.float32))
            tensors_to_save['latents'] = torch.from_numpy(np_array).to(torch.bfloat16)
        else:
            np_array = np.array(latents)
            tensors_to_save['latents'] = torch.from_numpy(np_array).contiguous()
        dtype_info['latents'] = str(latents.dtype)
    else:
        raise TypeError(f"Unsupported latents type: {type(latents)}")
    
    final_metadata = metadata or {}
    final_metadata['dtype_info'] = json.dumps(dtype_info)
    torch_save_file(tensors_to_save, output_path, metadata=final_metadata)
    print(f"✓ 已保存 latents 到 {output_path}, shape: {tensors_to_save['latents'].shape}")
    return output_path


def load_latents_from_safetensors(input_path, device='cpu', restore_dtype=True):
    """从 SafeTensors 文件加载 latents。"""
    tensors, metadata = load_embeddings_from_safetensors(input_path, device, restore_dtype)
    latents = tensors['latents']
    print(f"  latents shape: {latents.shape}, dtype: {latents.dtype}")
    return latents, metadata


# ============================================================================
# 配置管理
# ============================================================================

def save_generation_config(config_dict, output_path):
    """保存生成配置到 JSON 文件。"""
    with open(output_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"✓ 已保存配置到 {output_path}")


def load_generation_config(input_path):
    """从 JSON 文件加载生成配置。"""
    with open(input_path, 'r') as f:
        config = json.load(f)
    print(f"✓ 已加载配置从 {input_path}")
    return config


def get_default_paths(output_dir="./stage_outputs"):
    """获取默认的中间文件路径。"""
    os.makedirs(output_dir, exist_ok=True)
    return {
        'embeddings': os.path.join(output_dir, 'stage1_embeddings.safetensors'),
        'latents': os.path.join(output_dir, 'stage2_latents.safetensors'),
        'config': os.path.join(output_dir, 'generation_config.json'),
        'image': os.path.join(output_dir, 'output_image.png'),
    }


# ============================================================================
# JAX 配置
# ============================================================================

def setup_jax_cache():
    """设置 JAX 编译缓存。"""
    cache_dir = os.path.expanduser("~/.cache/jax_cache")
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    print(f"✓ JAX 编译缓存: {cache_dir}")
