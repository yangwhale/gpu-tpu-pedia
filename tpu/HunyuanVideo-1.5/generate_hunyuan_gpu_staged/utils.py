#!/usr/bin/env python3
"""
HunyuanVideo-1.5 三阶段生成 - 共享工具模块 (GPU 版本)

包含数据序列化、转换和共享配置
用于 H100 8卡 GPU 环境
"""

import os
import json
import numpy as np
import torch
from safetensors.torch import save_file as torch_save_file, load_file as torch_load_file
from safetensors import safe_open


# === 默认配置 ===
DEFAULT_MODEL_PATH = "/path/to/HunyuanVideo-1.5"  # 用户需要修改为实际路径
DEFAULT_RESOLUTION = "720p"
DEFAULT_ASPECT_RATIO = "16:9"
DEFAULT_VIDEO_LENGTH = 121
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_SEED = 42


# === 数据转换工具 ===

def tensor_to_safetensors_dict(tensor, key='data'):
    """将 tensor 转换为可保存的字典格式"""
    if tensor.dtype == torch.bfloat16:
        # safetensors 支持 bfloat16，但需要确保 contiguous
        return {key: tensor.cpu().contiguous()}
    return {key: tensor.cpu().contiguous()}


def save_embeddings_to_safetensors(embeddings_dict, output_path, metadata=None):
    """
    将 embeddings 字典保存为 SafeTensors 格式
    
    Args:
        embeddings_dict: 包含 tensor 的字典
        output_path: 输出文件路径
        metadata: 可选的元数据字典
    """
    tensors_to_save = {}
    dtype_info = {}
    
    for key, value in embeddings_dict.items():
        if isinstance(value, torch.Tensor):
            # safetensors 原生支持 bfloat16
            tensors_to_save[key] = value.cpu().contiguous()
            dtype_info[key] = str(value.dtype)
        elif value is None:
            # 跳过 None 值
            continue
    
    # 添加元数据
    final_metadata = metadata or {}
    final_metadata['dtype_info'] = json.dumps(dtype_info)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存
    torch_save_file(tensors_to_save, output_path, metadata=final_metadata)
    print(f"✓ 已保存 {len(tensors_to_save)} 个 tensors 到 {output_path}")
    
    return output_path


def load_embeddings_from_safetensors(input_path, device='cpu'):
    """
    从 SafeTensors 文件加载 embeddings
    
    Args:
        input_path: 输入文件路径
        device: 目标设备
        
    Returns:
        embeddings_dict: 包含 tensor 的字典
        metadata: 元数据字典
    """
    # 加载 tensors
    tensors = torch_load_file(input_path, device=device)
    
    # 加载元数据
    metadata = {}
    with safe_open(input_path, framework="pt") as f:
        raw_metadata = f.metadata()
        if raw_metadata:
            metadata = dict(raw_metadata)
    
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
        tensors_to_save['latents'] = latents.cpu().contiguous()
        dtype_info['latents'] = str(latents.dtype)
    else:
        raise TypeError(f"Unsupported latents type: {type(latents)}")
    
    # 添加元数据
    final_metadata = metadata or {}
    final_metadata['dtype_info'] = json.dumps(dtype_info)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存
    torch_save_file(tensors_to_save, output_path, metadata=final_metadata)
    print(f"✓ 已保存 latents 到 {output_path}")
    print(f"  shape: {tensors_to_save['latents'].shape}")
    print(f"  dtype: {tensors_to_save['latents'].dtype}")
    
    return output_path


def load_latents_from_safetensors(input_path, device='cpu'):
    """
    从 SafeTensors 文件加载 latents
    
    Args:
        input_path: 输入文件路径
        device: 目标设备
        
    Returns:
        latents: latents tensor
        metadata: 元数据字典
    """
    tensors, metadata = load_embeddings_from_safetensors(input_path, device)
    latents = tensors['latents']
    print(f"  latents shape: {latents.shape}, dtype: {latents.dtype}")
    return latents, metadata


# === 配置管理 ===

def save_generation_config(config_dict, output_path):
    """保存生成配置到 JSON 文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
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
        'frames': os.path.join(output_dir, 'stage3_frames.safetensors'),
    }


# === GPU/分布式工具 ===

def get_world_size():
    """获取分布式进程数"""
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return int(os.environ.get('WORLD_SIZE', '1'))


def get_rank():
    """获取当前进程的 rank"""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get('RANK', '0'))


def get_local_rank():
    """获取当前进程的 local rank"""
    return int(os.environ.get('LOCAL_RANK', '0'))


def is_main_process():
    """判断是否为主进程"""
    return get_rank() == 0


def print_rank0(*args, **kwargs):
    """仅在 rank 0 上打印"""
    if is_main_process():
        print(*args, **kwargs)


def setup_distributed():
    """设置分布式环境"""
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl')
        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)
        return True
    return False


def cleanup_distributed():
    """清理分布式环境"""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# === 视频保存工具 ===

def save_video(video_tensor, output_path, fps=24):
    """
    保存视频 tensor 为 mp4 文件
    
    Args:
        video_tensor: shape [C, T, H, W] 或 [B, C, T, H, W]，值范围 [0, 1]
        output_path: 输出路径
        fps: 帧率
    """
    import imageio
    from einops import rearrange
    
    if video_tensor.ndim == 5:
        assert video_tensor.shape[0] == 1
        video_tensor = video_tensor[0]
    
    # [C, T, H, W] -> [T, H, W, C]
    vid = (video_tensor * 255).clamp(0, 255).to(torch.uint8)
    vid = rearrange(vid, 'c f h w -> f h w c')
    vid = vid.cpu().numpy()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimwrite(output_path, vid, fps=fps)
    print(f"✓ 视频已保存到 {output_path}")