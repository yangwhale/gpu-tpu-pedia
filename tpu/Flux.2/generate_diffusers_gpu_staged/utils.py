#!/usr/bin/env python3
"""
Flux.2 GPU 三阶段生成 - 共享工具模块

包含:
- 配置常量（模型、图像生成）
- SafeTensors 数据存储
- 配置管理
- 辅助工具函数
"""

import os
import json
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file as torch_save_file, load_file as torch_load_file


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
        elif isinstance(value, np.ndarray):
            tensors_to_save[key] = torch.from_numpy(value)
            dtype_info[key] = str(value.dtype)
    
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
        latents: latents tensor (torch.Tensor)
        output_path: 输出文件路径
        metadata: 可选的元数据字典
    """
    tensors_to_save = {}
    dtype_info = {}
    
    if isinstance(latents, torch.Tensor):
        if latents.dtype == torch.bfloat16:
            tensors_to_save['latents'] = latents.to(torch.float32).cpu().contiguous()
            dtype_info['latents'] = 'bfloat16'
        else:
            tensors_to_save['latents'] = latents.cpu().contiguous()
            dtype_info['latents'] = str(latents.dtype)
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
