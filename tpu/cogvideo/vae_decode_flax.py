"""
纯 Flax 版本的 CogVideoX VAE Decode 性能测试

与 vae_decode.py (torchax版本) 的主要区别：
- 使用原生 Flax VAE 实现，无需 PyTorch/torchax
- 使用 JAX 原生分片和 JIT 编译
- 更简洁的代码，更好的性能
"""
import sys
sys.path.insert(0, "/home/chrisya/diffusers-tpu-chris/src")

import time
import numpy as np
import jax
import jax.numpy as jnp
import warnings
import logging
from contextlib import nullcontext
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
from jax.experimental import mesh_utils

from diffusers.models.autoencoders.autoencoder_kl_cogvideox_flax import (
    FlaxAutoencoderKLCogVideoX,
)

MODEL_NAME = "zai-org/CogVideoX1.5-5B"
VAE_SUBFOLDER = "vae"

def record_time(call_method):
    """记录函数执行时间"""
    start = time.time()
    output = call_method()
    jax.block_until_ready(output)
    return output, (time.time() - start) * 1000

def create_mesh(num_devices=None):
    """创建设备 Mesh 用于分片"""
    if num_devices is None:
        num_devices = jax.device_count()
    
    print(f"\n创建设备 Mesh...")
    print(f"  可用设备数: {num_devices}")
    
    # 简单的1D分片：所有设备在 tp 维度
    devices = mesh_utils.create_device_mesh((num_devices,))
    mesh = Mesh(devices, ('tp',))
    
    print(f"  Mesh 形状: {mesh.shape}")
    print(f"  Mesh 轴名: {mesh.axis_names}")
    
    return mesh

def load_vae_flax(dtype=jnp.bfloat16):
    """加载 Flax VAE 模型"""
    print(f"\n加载 Flax VAE 模型: {MODEL_NAME}")
    print(f"  目标 dtype: {dtype}")
    
    vae = FlaxAutoencoderKLCogVideoX.from_pretrained(
        MODEL_NAME,
        subfolder=VAE_SUBFOLDER,
        dtype=dtype,
    )
    
    print("✓ VAE 模型加载完成")
    return vae

def setup_tiling(vae):
    """配置 VAE Tiling 以节省显存"""
    print("\n配置 VAE Tiling...")
    
    vae.enable_tiling(
        tile_sample_min_height=192,  # 默认 480//2
        tile_sample_min_width=340,   # 默认 720//2
        tile_overlap_factor_height=1/6,  # 16.7% 垂直重叠
        tile_overlap_factor_width=1/5,   # 20% 水平重叠
    )
    
    print(f"  ✓ Tile 最小尺寸: {vae.tile_sample_min_height}x{vae.tile_sample_min_width}")
    print(f"  ✓ 重叠因子: 垂直={vae.tile_overlap_factor_height:.2%}, 水平={vae.tile_overlap_factor_width:.2%}")
    print("Tiling 配置完成")

def create_decode_fn(vae, mesh):
    """创建 decode 函数（不JIT，避免循环展开）
    
    关键：不对外层 decode 使用 JIT，保持 Python 循环
    这样可以避免 XLA 展开循环导致的巨大内存分配
    """
    print("\n准备 VAE decode 函数...")
    
    # 不使用 JIT！让 Python for 循环在运行时逐帧执行
    # 这样 XLA 不会展开循环
    def decode_fn(latents):
        return vae.decode(latents, deterministic=True)
    
    print("✓ Decode 函数准备完成（无JIT，避免循环展开）")
    
    return decode_fn

def prepare_latents(mesh, frames, height, width, dtype=jnp.bfloat16):
    """准备测试用的 latents
    
    关键优化：根据帧数自动选择分片策略
    - 帧数 < mesh.size：使用复制模式（避免分片维度不能整除）
    - 帧数 >= mesh.size：使用时间维度分片
    """
    latent_channels = 16
    latent_frames = max(1, frames // 4)  # CogVideoX 时间压缩比 4x
    latent_height = height // 8  # 空间压缩比 8x
    latent_width = width // 8
    
    # JAX 格式: (B, T, H, W, C)
    global_shape = (1, latent_frames, latent_height, latent_width, latent_channels)
    
    print(f"\n创建测试 Latents...")
    print(f"  全局形状: {global_shape}")
    print(f"  数据类型: {dtype}")
    
    # 根据 latent 帧数选择分片策略
    if latent_frames < mesh.size:
        sharding = NamedSharding(mesh, P())  # 复制到所有设备
        print(f"  分片策略: 复制模式 (latent帧数 {latent_frames} < 设备数 {mesh.size})")
    else:
        sharding = NamedSharding(mesh, P(None, 'tp', None, None, None))
        print(f"  分片策略: 时间维度分片")
    
    # 在 CPU 上创建随机数据
    latents_np = np.random.randn(*global_shape).astype(np.float32)
    
    # 分片并上传到设备
    latents = jax.device_put(latents_np, sharding).astype(dtype)
    jax.block_until_ready(latents)
    
    # 计算内存占用
    num_elements = np.prod(global_shape)
    bytes_per_element = 2 if dtype == jnp.bfloat16 else 4
    memory_mb = (num_elements * bytes_per_element) / (1024 ** 2)
    
    print(f"  ✓ Latents 已创建并分片到设备")
    print(f"  内存占用: {memory_mb:.2f} MB")
    
    return latents

def test_vae_decode(
    decode_fn,
    latents,
    num_runs=5,
    profiler_context=None,
):
    """测试 VAE decode 性能"""
    print(f"\n{'='*60}")
    print(f"开始性能测试 - 运行 {num_runs} 次")
    print(f"{'='*60}\n")
    
    # 多轮预热以触发所有 JIT 编译变体
    # 原因：nnx.jit 对不同的 conv_cache 状态会编译不同版本
    # - 第1帧：conv_cache=None → 编译版本A
    # - 第2-8帧：conv_cache=Dict → 编译版本B
    print("预热运行（3轮，确保所有 JIT 变体都被编译）...")
    for i in range(3):
        output, time_cost = record_time(lambda: decode_fn(latents))
        jax.block_until_ready(output)  # 确保计算完成
        print(f"  第 {i+1} 轮预热: {time_cost:.2f} ms")
        if i == 2:  # 最后一轮显示输出信息
            print(f"  输出形状: {output.shape}")
            print(f"  输出 dtype: {output.dtype}")
        
        # 清理内存
        del output
        jax.clear_caches()
        import gc
        gc.collect()
    
    # 性能测试
    print(f"\n正式测试 ({num_runs} 次运行):")
    times = []
    
    context = profiler_context if profiler_context else nullcontext()
    
    with context:
        for run_idx in range(num_runs):
            output, time_cost = record_time(lambda: decode_fn(latents))
            times.append(time_cost)
            print(f"  第 {run_idx + 1} 次: {time_cost:.2f} ms")
            
            # 关键：立即删除输出并清理缓存
            del output
            jax.clear_caches()  # 清理 JAX 编译缓存
            
            # 强制垃圾回收
            import gc
            gc.collect()
    
    # 统计结果
    print(f"\n{'='*60}")
    print("性能统计:")
    print(f"{'='*60}")
    print(f"  平均耗时: {np.mean(times):.2f} ms")
    print(f"  最小耗时: {np.min(times):.2f} ms")
    print(f"  最大耗时: {np.max(times):.2f} ms")
    print(f"  标准差:   {np.std(times):.2f} ms")
    print(f"{'='*60}\n")

def main():
    """主函数"""
    # 过滤警告
    warnings.filterwarnings('ignore', message='.*dtype.*int64.*truncated to dtype int32.*')
    logging.getLogger().setLevel(logging.ERROR)
    
    # JAX 配置
    print("配置 JAX 环境...")
    jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    
    # 基础内存优化：禁用预分配
    import os
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    
    print("✓ JAX 配置完成")
    
    # 创建设备 Mesh
    mesh = create_mesh()
    
    # 加载模型
    vae = load_vae_flax(dtype=jnp.bfloat16)
    
    # 配置 Tiling（暂时禁用用于调试）
    # setup_tiling(vae)
    
    # 创建 decode 函数（不JIT）
    decode_fn = create_decode_fn(vae, mesh)
    
    # 准备测试数据
    # 测试配置：32 帧 @ 768x1360 分辨率（之前会 OOM 的配置）
    # 关键修复：自动选择合适的分片策略
    latents = prepare_latents(
        mesh=mesh,
        frames=32,
        height=768,
        width=1360,
        dtype=jnp.bfloat16,
    )
    
    # Profiler 配置（可选）
    profiler_context = None
    if False:  # 设为 True 启用 profiling
        print("\n启用 JAX Profiler...")
        profiler_context = jax.profiler.trace(
            "/dev/shm/jax-trace",
            create_perfetto_link=False
        )
    
    # 运行性能测试
    test_vae_decode(
        decode_fn=decode_fn,
        latents=latents,
        num_runs=5,
        profiler_context=profiler_context,
    )
    
    print("✅ 所有测试完成！")

if __name__ == "__main__":
    main()