#!/usr/bin/env python3
"""
HunyuanVideo-1.5 三阶段生成 - 共享工具模块

包含数据序列化、转换和共享配置
"""

import os
import json
import numpy as np
import jax.numpy as jnp
import torch
from safetensors import safe_open
from safetensors.numpy import save_file as np_save_file
from safetensors.torch import save_file as torch_save_file, load_file as torch_load_file
from jax.tree_util import register_pytree_node
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutputWithPastAndCrossAttentions
from diffusers.models.autoencoders.vae import DecoderOutput


# === 模型配置 ===
MODEL_NAME = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v"

# === Splash Attention 配置参数 ===
BQSIZE = 2048           # Query 块大小
BKVSIZE = 2048          # Key/Value 块大小
BKVCOMPUTESIZE = 1024   # Key/Value 计算块大小

# === Mesh 分片配置 ===
USE_DP = False          # 是否使用 data parallelism
SP_NUM = 1              # Spatial parallelism 数量
USE_TP = True           # 是否使用 Tensor Parallel 模式


# === 数据转换工具 ===

def to_torch_recursive(x):
    """递归地将 JAX 数组转换为 PyTorch 张量"""
    if 'ArrayImpl' in str(type(x)) or isinstance(x, jnp.ndarray):
        np_array = np.array(x)
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
    注册必要的pytree节点以支持JAX转换
    """
    print("注册PyTree节点...")
    
    def model_output_flatten(obj):
        return obj.to_tuple(), type(obj)

    def model_output_unflatten(aux, children):
        return aux(*children)
    
    OUTPUT_CLASSES = [
        BaseModelOutputWithPooling,
        BaseModelOutputWithPastAndCrossAttentions,
        DecoderOutput,
    ]

    for cls in OUTPUT_CLASSES:
        try:
            register_pytree_node(cls, model_output_flatten, model_output_unflatten)
            print(f"  - {cls.__name__} 已注册")
        except ValueError:
            # 已经注册过
            print(f"  - {cls.__name__} 已存在")


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


# === 配置管理 ===

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


# === 默认输出路径 ===

def get_default_paths(output_dir="./stage_outputs"):
    """获取默认的中间文件路径"""
    os.makedirs(output_dir, exist_ok=True)
    return {
        'embeddings': os.path.join(output_dir, 'stage1_embeddings.safetensors'),
        'latents': os.path.join(output_dir, 'stage2_latents.safetensors'),
        'config': os.path.join(output_dir, 'generation_config.json'),
        'video': os.path.join(output_dir, 'output_video.mp4'),
    }