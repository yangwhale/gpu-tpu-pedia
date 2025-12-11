#!/usr/bin/env python3
"""
Wan 2.1 三阶段生成 - 阶段3：VAE Decoder

本阶段负责：
1. 加载阶段2生成的 latents
2. 加载 JAX Wan VAE 模型（NNX 实现）
3. 解码 latents 为视频帧
4. 导出最终视频

输入文件：
- stage2_latents.safetensors: 生成的 latents
- generation_config.json: 生成配置

输出文件：
- output_video.mp4: 最终生成的视频
"""

import os
import sys
import time
import functools
import argparse
import warnings
import logging
import numpy as np
import jax
import jax.numpy as jnp
import torch

from flax import nnx
from flax.linen import partitioning as nn_partitioning
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
from jax.experimental import mesh_utils

from diffusers.utils import export_to_video

# Add parent directory to path for maxdiffusion imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from maxdiffusion.models.wan.autoencoder_kl_wan import (
    AutoencoderKLWan,
    AutoencoderKLWanCache,
)

from utils import (
    MODEL_NAME,
    FPS,
    LOGICAL_AXIS_RULES,
    to_torch_recursive,
    sharded_device_put,
    prepare_video_for_export,
    load_latents_from_safetensors,
    load_generation_config,
    get_default_paths,
)


# === VAE 加载函数 ===

def load_wan_vae_weights(pretrained_model_name_or_path, eval_shapes, device, hf_download=True):
    """Load Wan VAE weights with proper type handling."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from flax.traverse_util import unflatten_dict
    from maxdiffusion.models.modeling_flax_pytorch_utils import (
        rename_key, rename_key_and_reshape_tensor, validate_flax_state_dict
    )
    
    device_obj = jax.local_devices(backend=device)[0]
    with jax.default_device(device_obj):
        if hf_download:
            ckpt_path = hf_hub_download(
                pretrained_model_name_or_path,
                subfolder="vae",
                filename="diffusion_pytorch_model.safetensors"
            )
        
        print(f"加载 Wan 2.1 VAE 权重 (设备: {device})")
        
        tensors = {}
        with safe_open(ckpt_path, framework="np") as f:
            for k in f.keys():
                tensors[k] = jnp.array(f.get_tensor(k))
        
        flax_state_dict = {}
        cpu = jax.local_devices(backend="cpu")[0]
        
        for pt_key, tensor in tensors.items():
            renamed_pt_key = rename_key(pt_key)
            # Apply Wan-specific key transformations
            for old, new in [
                ("up_blocks_", "up_blocks."),
                ("mid_block_", "mid_block."),
                ("down_blocks_", "down_blocks."),
                ("conv_in.bias", "conv_in.conv.bias"),
                ("conv_in.weight", "conv_in.conv.weight"),
                ("conv_out.bias", "conv_out.conv.bias"),
                ("conv_out.weight", "conv_out.conv.weight"),
                ("attentions_", "attentions."),
                ("resnets_", "resnets."),
                ("upsamplers_", "upsamplers."),
                ("resample_", "resample."),
                ("conv1.bias", "conv1.conv.bias"),
                ("conv1.weight", "conv1.conv.weight"),
                ("conv2.bias", "conv2.conv.bias"),
                ("conv2.weight", "conv2.conv.weight"),
                ("time_conv.bias", "time_conv.conv.bias"),
                ("time_conv.weight", "time_conv.conv.weight"),
                ("quant_conv", "quant_conv.conv"),
                ("conv_shortcut", "conv_shortcut.conv"),
            ]:
                renamed_pt_key = renamed_pt_key.replace(old, new)
            
            if "decoder" in renamed_pt_key:
                renamed_pt_key = renamed_pt_key.replace("resample.1.bias", "resample.layers.1.bias")
                renamed_pt_key = renamed_pt_key.replace("resample.1.weight", "resample.layers.1.weight")
            if "encoder" in renamed_pt_key:
                renamed_pt_key = renamed_pt_key.replace("resample.1", "resample.conv")
            
            pt_tuple_key = tuple(renamed_pt_key.split("."))
            flax_key, flax_tensor = rename_key_and_reshape_tensor(pt_tuple_key, tensor, eval_shapes)
            flax_key = tuple(
                int(item) if isinstance(item, str) and item.isdigit() else item
                for item in flax_key
            )
            flax_state_dict[flax_key] = jax.device_put(jnp.asarray(flax_tensor), device=cpu)
        
        validate_flax_state_dict(eval_shapes, flax_state_dict)
        flax_state_dict = unflatten_dict(flax_state_dict)
        del tensors
        jax.clear_caches()
    
    return flax_state_dict


def _add_sharding_rule(vs, logical_axis_rules):
    """Add sharding rules to variable state."""
    vs.sharding_rules = logical_axis_rules
    return vs


@nnx.jit(static_argnums=(1,), donate_argnums=(0,))
def create_sharded_logical_model(model, logical_axis_rules):
    """Create a sharded model with logical axis rules."""
    graphdef, state, rest_of_state = nnx.split(model, nnx.Param, ...)
    p_add_sharding_rule = functools.partial(
        _add_sharding_rule, logical_axis_rules=logical_axis_rules
    )
    state = jax.tree.map(
        p_add_sharding_rule, state,
        is_leaf=lambda x: isinstance(x, nnx.VariableState)
    )
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    model = nnx.merge(graphdef, sharded_state, rest_of_state)
    return model


def setup_wan_vae(model_id, mesh, vae_mesh):
    """Initialize and load Wan VAE with proper sharding."""
    print("\n加载 JAX Wan VAE 模型...")
    
    with vae_mesh:
        key = jax.random.key(0)
        rngs = nnx.Rngs(key)
        
        wan_vae = AutoencoderKLWan(
            rngs=rngs,
            base_dim=96,
            z_dim=16,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            attn_scales=[],
            temperal_downsample=[False, True, True],
            mesh=vae_mesh
        )
    
    with mesh:
        vae_cache = AutoencoderKLWanCache(wan_vae)
        
        graphdef, state = nnx.split(wan_vae)
        params = state.to_pure_dict()
        params = load_wan_vae_weights(model_id, params, "tpu")
        
        # Replicate to all devices
        sharding = NamedSharding(mesh, P())
        params = jax.tree_util.tree_map(
            lambda x: sharded_device_put(x, sharding), params
        )
        params = jax.tree_util.tree_map(
            lambda x: x.astype(jnp.bfloat16), params
        )
        wan_vae = nnx.merge(graphdef, params)
        
        # Apply logical sharding
        wan_vae = create_sharded_logical_model(
            model=wan_vae,
            logical_axis_rules=LOGICAL_AXIS_RULES
        )
    
    print(f"  ✓ Wan VAE 已加载")
    return wan_vae, vae_cache


def decode_latents_to_video(vae, vae_cache, latents, config, mesh):
    """
    使用 VAE 解码 latents 为视频帧
    
    Args:
        vae: JAX Wan VAE 模型
        vae_cache: VAE 缓存
        latents: latents tensor [B, C, T, H, W]
        config: 生成配置
        mesh: JAX mesh
        
    Returns:
        frames: 视频帧 numpy array
        elapsed: 解码耗时
    """
    print(f"\n=== 阶段3：VAE 解码 ===")
    print(f"输入 latents shape: {latents.shape}")
    print(f"输入 latents dtype: {latents.dtype}")
    
    # 转换为 JAX array
    if isinstance(latents, torch.Tensor):
        if latents.dtype == torch.bfloat16:
            jax_latents = jnp.array(latents.to(torch.float32).cpu().numpy()).astype(jnp.bfloat16)
        else:
            jax_latents = jnp.array(latents.cpu().numpy())
    else:
        jax_latents = latents
    
    # 反归一化 latents（关键步骤！）
    # 这是 pipeline 中 VAE decode 前的预处理
    latents_mean = jnp.array(vae.latents_mean).reshape(1, vae.z_dim, 1, 1, 1)
    latents_std = 1.0 / jnp.array(vae.latents_std).reshape(1, vae.z_dim, 1, 1, 1)
    jax_latents = jax_latents / latents_std + latents_mean
    print(f"反归一化后 latents range: [{float(jnp.min(jax_latents)):.4f}, {float(jnp.max(jax_latents)):.4f}]")
    
    print(f"JAX latents shape: {jax_latents.shape}, dtype: {jax_latents.dtype}")
    
    # VAE decode
    print("\n开始 VAE 解码...")
    start_time = time.perf_counter()
    
    with mesh, nn_partitioning.axis_rules(LOGICAL_AXIS_RULES):
        output = vae.decode(jax_latents, feat_cache=vae_cache)
        jax.effects_barrier()
    
    elapsed = time.perf_counter() - start_time
    print(f"✓ VAE 解码完成，耗时: {elapsed:.2f} 秒")
    
    # 转换为 PyTorch tensor
    output = to_torch_recursive(output)
    
    # 如果输出是 DecoderOutput 类型，提取 sample
    if hasattr(output, 'sample'):
        output = output.sample
    
    print(f"输出 shape: {output.shape}, dtype: {output.dtype}")
    
    return output, elapsed


def main():
    parser = argparse.ArgumentParser(
        description='Wan 2.1 阶段3：VAE Decoder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 基本用法（使用阶段2的输出）
  python stage3_vae_decoder.py
  
  # 指定输入目录
  python stage3_vae_decoder.py --input_dir ./my_outputs
  
  # 指定输出视频路径
  python stage3_vae_decoder.py --output_video my_video.mp4
        """
    )
    
    parser.add_argument(
        '--input_dir', type=str, default='./stage_outputs',
        help='Input directory containing stage2 outputs (default: ./stage_outputs)'
    )
    parser.add_argument(
        '--output_video', type=str, default=None,
        help='Output video path (default: stage_outputs/output_video.mp4)'
    )
    
    # VAE 配置
    parser.add_argument(
        '--model_id', type=str, default=None,
        help='Override model ID for VAE'
    )
    
    # 视频输出配置
    parser.add_argument(
        '--fps', type=int, default=None,
        help='Output video FPS (default: from config)'
    )
    
    args = parser.parse_args()
    
    # 设置 JAX 编译缓存
    jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
    
    paths = get_default_paths(args.input_dir)
    
    warnings.filterwarnings('ignore', message='.*dtype.*int64.*truncated to dtype int32.*')
    warnings.filterwarnings('ignore')
    logging.getLogger().setLevel(logging.ERROR)
    
    print(f"\n{'='*60}")
    print("Wan 2.1 阶段3：VAE Decoder")
    print(f"{'='*60}")
    
    # 加载配置
    print(f"\n加载配置: {paths['config']}")
    config = load_generation_config(paths['config'])
    
    # 应用命令行覆盖
    model_id = args.model_id or config.get('model_id', MODEL_NAME)
    fps = args.fps or config.get('fps', FPS)
    target_frames = config.get('frames', 81)
    
    output_video = args.output_video or paths['video']
    
    # 加载 latents
    print(f"\n加载 latents: {paths['latents']}")
    latents, latents_metadata = load_latents_from_safetensors(
        paths['latents'],
        device='cpu',
        restore_dtype=True
    )
    
    # 创建 mesh
    print(f"\n设置 JAX Mesh...")
    print(f"总设备数: {len(jax.devices())}")
    
    # Main mesh for VAE
    mesh_devices = mesh_utils.create_device_mesh(
        (2, 1, len(jax.devices()) // 2), allow_split_physical_axes=True
    )
    mesh = Mesh(mesh_devices, ('dp', 'sp', 'tp'))
    
    # VAE mesh (different layout for conv operations)
    vae_mesh = jax.make_mesh((1, len(jax.devices())), ('conv_in', 'conv_out'))
    
    # 加载 VAE
    vae, vae_cache = setup_wan_vae(model_id, mesh, vae_mesh)
    
    # 解码
    video, decode_time = decode_latents_to_video(
        vae, vae_cache,
        latents,
        config,
        mesh
    )
    
    # 准备视频导出（使用与 VideoProcessor 一致的后处理方式）
    print(f"\n准备视频导出...")
    video = prepare_video_for_export(video, target_frames)
    
    print(f"后处理后 video shape: {video.shape}")
    
    # 导出视频
    print(f"\n导出视频到: {output_video}")
    print(f"FPS: {fps}")
    
    export_to_video(video, output_video, fps=fps)
    
    print(f"✓ 视频已保存!")
    
    # 统计信息
    print(f"\n=== 生成统计 ===")
    if isinstance(video, np.ndarray):
        print(f"帧数: {video.shape[0]}")
        print(f"分辨率: {video.shape[2]}x{video.shape[1]}")
    print(f"FPS: {fps}")
    print(f"VAE 解码耗时: {decode_time:.2f} 秒")
    
    print(f"\n{'='*60}")
    print("阶段3 完成！")
    print(f"{'='*60}")
    print(f"\n输出视频: {output_video}")
    
    print("\n✓ 视频生成完成！")


if __name__ == "__main__":
    main()