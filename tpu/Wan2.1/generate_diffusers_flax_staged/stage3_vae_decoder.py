#!/usr/bin/env python3
"""
Wan 2.1 三阶段生成 - 阶段3：VAE Decoder (TPU)

本阶段负责：
1. 加载阶段2生成的 latents
2. 启用 torchax 并移动 VAE 到 TPU
3. 反归一化 latents
4. 使用 VAE 解码为视频帧
5. 后处理并导出最终视频

输入文件：
- stage2_latents.safetensors: 生成的 latents
- generation_config.json: 生成配置

输出文件：
- output_video.mp4: 最终生成的视频
"""

import time
import argparse
import warnings
import logging
import functools

import jax
import torch
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
from jax.experimental import mesh_utils

import torchax
from torchax.ops import ops_registry, jaten

from diffusers.utils import export_to_video
from diffusers.models.autoencoders.autoencoder_kl_wan_flax import AutoencoderKLWan

from utils import (
    MODEL_NAME,
    FPS,
    DEFAULT_DP,
    VAE_DECODER_SHARDINGS,
    shard_weight_dict,
    prepare_video_for_export,
    load_latents_from_safetensors,
    load_generation_config,
    get_default_paths,
    setup_jax_cache,
    setup_pytree_registrations,
)


# ============================================================================
# Torchax 帮助函数
# ============================================================================

def torch_conv2d_jax(input, weight, bias=None, stride=1, padding=0,
                     dilation=1, groups=1, *, env):
    """JAX-compatible conv2d override."""
    jinput, jweight, jbias = env.t2j_iso((input, weight, bias))
    res = jaten._aten_conv2d(jinput, jweight, jbias, stride, padding,
                             dilation, groups)
    return env.j2t_iso(res)


def override_op_definition(env, op_to_override, op_impl):
    """Override operator definition in torchax environment."""
    env._ops[op_to_override] = ops_registry.Operator(
        op_to_override,
        op_impl,
        is_jax_function=False,
        is_user_defined=True,
        needs_env=False,
        is_view_op=False,
    )


def move_module_to_xla(env, module):
    """Move module weights to XLA devices."""
    with jax.default_device("cpu"):
        state_dict = module.state_dict()
        state_dict = env.to_xla(state_dict)
        module.load_state_dict(state_dict, assign=True)


# ============================================================================
# Latent 归一化
# ============================================================================

def denormalize_latents(latents, vae):
    """
    反归一化 latents: x * std + mean
    
    Args:
        latents: [B, C, T, H, W] 格式的归一化 latent tensor
        vae: VAE 模型实例（用于获取 config 中的参数）
        
    Returns:
        反归一化后的 latents
    """
    latents_mean = getattr(vae.config, 'latents_mean', None)
    latents_std = getattr(vae.config, 'latents_std', None)
    
    if latents_mean is None or latents_std is None:
        print("警告：VAE config 中没有 latents_mean/latents_std，跳过反归一化")
        return latents
    
    mean_tensor = torch.tensor(latents_mean).view(1, 16, 1, 1, 1).to(latents.device, latents.dtype)
    std_tensor = torch.tensor(latents_std).view(1, 16, 1, 1, 1).to(latents.device, latents.dtype)
    return latents * std_tensor + mean_tensor


# ============================================================================
# VAE 加载与配置
# ============================================================================

def load_vae(model_id):
    """Load VAE BEFORE enabling torchax."""
    print(f"\n加载 VAE: {model_id}")
    vae = AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.bfloat16
    )
    print("✓ VAE 加载完成")
    return vae


def setup_vae_for_jax(vae, mesh, env):
    """Setup VAE decoder for JAX/TPU execution."""
    print("\n=== 配置 VAE Decoder (TPU) ===")
    
    # Register custom operators
    print("- 注册 JAX conv2d 操作...")
    override_op_definition(
        env,
        torch.nn.functional.conv2d,
        functools.partial(torch_conv2d_jax, env=env)
    )
    
    # Move VAE to XLA
    print("- 移动 VAE Decoder 到 TPU...")
    move_module_to_xla(env, vae)
    vae.decoder = torchax.compile(vae.decoder)
    vae.decoder.params = shard_weight_dict(
        vae.decoder.params, VAE_DECODER_SHARDINGS, mesh
    )
    vae.decoder.buffers = shard_weight_dict(
        vae.decoder.buffers, VAE_DECODER_SHARDINGS, mesh
    )
    
    print("✓ VAE Decoder 配置完成")
    return vae


# ============================================================================
# 解码函数
# ============================================================================

def run_vae_decode(vae, latents, env, desc="VAE Decode"):
    """
    运行一次 VAE 解码
    
    Args:
        vae: VAE 模型
        latents: latent tensor (已经在 XLA 上)
        env: torchax 环境
        desc: 描述信息
    
    Returns:
        (video, elapsed_time)
    """
    start_time = time.perf_counter()
    
    print(f"\n{desc}...")
    with torch.no_grad():
        video = vae.decode(latents).sample
    jax.effects_barrier()
    
    elapsed = time.perf_counter() - start_time
    print(f"✓ {desc} 完成，耗时: {elapsed:.2f} 秒")
    
    return video, elapsed


def decode_latents_to_video(vae, latents, config, env, warmup=True):
    """
    使用 VAE 解码 latents 为视频帧
    """
    print(f"\n=== 阶段3：VAE 解码 ===")
    print(f"输入 latents shape: {latents.shape}")
    print(f"输入 latents dtype: {latents.dtype}")
    
    # 检查 nan 值
    latents_float = latents.float()
    nan_count = torch.isnan(latents_float).sum().item()
    total = latents_float.numel()
    print(f"输入 latents nan 统计: {nan_count}/{total} ({nan_count/total*100:.2f}%)")
    
    # 处理 nan 值 - 替换为 0
    if nan_count > 0:
        print(f"警告：发现 {nan_count} 个 nan 值，将替换为 0")
        latents = torch.nan_to_num(latents, nan=0.0)
    
    # 检查每帧的统计信息
    print("\n每帧统计（反归一化前）:")
    for t in range(min(5, latents.shape[2])):
        frame = latents_float[0, :, t]
        valid = frame[~torch.isnan(frame)]
        if len(valid) > 0:
            print(f"  Frame {t}: mean={valid.mean():.4f}, std={valid.std():.4f}")
    
    # 1. 转换为 VAE dtype 并转换为 XLA tensor
    print("\n转换 latents 到 XLA...")
    latents = latents.to(vae.dtype)  # 转换为 bfloat16
    latents = env.to_xla(latents)
    
    # 2. 反归一化 latents（使用 VAE 的 config 中的参数）
    print("反归一化 latents...")
    latents = denormalize_latents(latents, vae=vae)
    
    # 3. VAE 解码 (VAE 期望 [B, C, T, H, W] 格式)
    
    # 预热运行 (触发 JIT 编译)
    warmup_time = 0
    if warmup:
        _, warmup_time = run_vae_decode(vae, latents, env, desc="Warmup VAE (JIT)")
    
    # 正式解码
    video, decode_time = run_vae_decode(vae, latents, env, desc="VAE Decode")
    elapsed = decode_time  # 只记录正式解码时间
    
    print(f"输出 video shape: {video.shape}")
    print(f"输出 video dtype: {video.dtype}")
    
    # 检查 video 范围
    video_cpu = video.to('cpu').float()
    print(f"输出 video 范围: min={video_cpu.min():.4f}, max={video_cpu.max():.4f}")
    
    # 检查每帧
    print("\n每帧统计（VAE 解码后）:")
    for t in range(min(5, video_cpu.shape[2])):
        frame = video_cpu[0, :, t]
        print(f"  Frame {t}: mean={frame.mean():.4f}, std={frame.std():.4f}, min={frame.min():.4f}, max={frame.max():.4f}")
    
    return video, elapsed


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Wan 2.1 阶段3：VAE Decoder (TPU)',
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
    
    parser.add_argument('--input_dir', type=str, default='./stage_outputs')
    parser.add_argument('--output_video', type=str, default=None)
    parser.add_argument('--model_id', type=str, default=None)
    parser.add_argument('--fps', type=int, default=None)
    parser.add_argument('--dp', type=int, default=DEFAULT_DP, help='Data parallelism dimension')
    parser.add_argument('--warmup', action='store_true', default=True,
                        help='运行预热解码触发 JIT 编译（默认启用）')
    parser.add_argument('--no_warmup', action='store_false', dest='warmup',
                        help='禁用预热解码')
    
    args = parser.parse_args()
    
    # 设置 JAX 编译缓存
    setup_jax_cache()
    
    paths = get_default_paths(args.input_dir)
    
    warnings.filterwarnings('ignore')
    logging.getLogger().setLevel(logging.ERROR)
    
    torch.set_default_dtype(torch.bfloat16)
    
    # Setup PyTree registrations
    setup_pytree_registrations()
    
    print(f"\n{'='*60}")
    print("Wan 2.1 阶段3：VAE Decoder (TPU)")
    print(f"{'='*60}")
    
    # 加载配置
    print(f"\n加载配置: {paths['config']}")
    config = load_generation_config(paths['config'])
    
    # 应用命令行覆盖
    model_id = args.model_id or config.get('model_id', MODEL_NAME)
    fps = args.fps or config.get('fps', FPS)
    target_frames = config.get('frames', 81)
    
    output_video = args.output_video or paths['video']
    
    print(f"\n配置参数：")
    print(f"  模型: {model_id}")
    print(f"  FPS: {fps}")
    print(f"  目标帧数: {target_frames}")
    
    # 加载 latents
    print(f"\n加载 latents: {paths['latents']}")
    latents, latents_metadata = load_latents_from_safetensors(
        paths['latents'],
        device='cpu',
        restore_dtype=True
    )
    
    print(f"\n设备信息：")
    print(f"  JAX 设备数: {len(jax.devices())}")
    
    # 加载 VAE（在启用 torchax 之前）
    vae = load_vae(model_id)
    
    # 启用 torchax
    print("\n启用 torchax...")
    torchax.enable_globally()
    env = torchax.default_env()
    
    # 创建 mesh
    assert len(jax.devices()) % args.dp == 0
    tp_dim = len(jax.devices()) // args.dp
    mesh_devices = mesh_utils.create_device_mesh(
        (args.dp, tp_dim), allow_split_physical_axes=True
    )
    mesh = Mesh(mesh_devices, ("dp", "tp"))
    print(f"Mesh: {mesh}\n")
    
    # 配置 VAE
    with mesh:
        vae = setup_vae_for_jax(vae, mesh, env)
        
        # 解码
        video, decode_time = decode_latents_to_video(
            vae,
            latents,
            config,
            env,
            warmup=args.warmup
        )
    
    # 转换回 CPU
    print("\n转换视频到 CPU...")
    if hasattr(video, 'to'):
        video = video.to('cpu')
    
    # 准备视频导出
    print(f"\n准备视频导出...")
    frames = prepare_video_for_export(video, target_frames)
    print(f"后处理后 video shape: {frames.shape}")
    
    # 导出视频
    print(f"\n导出视频到: {output_video}")
    print(f"FPS: {fps}")
    
    export_to_video(frames, output_video, fps=fps)
    
    print(f"✓ 视频已保存!")
    
    # 统计信息
    print(f"\n=== 生成统计 ===")
    if hasattr(frames, 'shape'):
        print(f"帧数: {frames.shape[0]}")
        print(f"分辨率: {frames.shape[2]}x{frames.shape[1]}")
    print(f"FPS: {fps}")
    print(f"VAE 解码耗时: {decode_time:.2f} 秒（不含预热）")
    
    print(f"\n{'='*60}")
    print("阶段3 完成！")
    print(f"{'='*60}")
    print(f"\n输出视频: {output_video}")
    
    print("\n✓ 视频生成完成！")


if __name__ == "__main__":
    main()