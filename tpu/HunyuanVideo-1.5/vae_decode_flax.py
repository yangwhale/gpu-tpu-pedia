"""
Flax version of HunyuanVideo-1.5 VAE Decode performance test

Based on cogvideo/vae_decode_flax.py, adapted for HunyuanVideo-1.5.

Key differences from CogVideoX:
- Spatial compression: 16x (vs 8x)
- Temporal compression: 4x (same)
- Latent channels: 32 (vs 16)
- Block architecture: DCAE-style with channel rearrangement
- Normalization: RMS normalization (vs GroupNorm)
- Attention: Causal attention with masking

Reference: gpu-tpu-pedia/tpu/HunyuanVideo-1.5/generate_diffusers_flax.py
- VAEProxy class for torchax compatibility
- Default frames: 61 (for 720p t2v)
- VAE Tiling: enabled by default for memory efficiency
- scaling_factor: 1.03682 (handled by Pipeline, not VAE directly)
"""
import sys
sys.path.insert(0, "/home/chrisya/diffusers-tpu/src")

import time
import numpy as np
import jax
import jax.numpy as jnp
import warnings
import logging
from contextlib import nullcontext
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
from jax.experimental import mesh_utils

from diffusers.models.autoencoders.autoencoder_kl_hunyuanvideo15_flax import (
    FlaxAutoencoderKLHunyuanVideo15,
    FlaxAutoencoderKLHunyuanVideo15Config,
)

# === 模型配置 ===
# HunyuanVideo-1.5 Diffusers 720p t2v 模型
# 参考: generate_diffusers_flax.py 第 33 行
MODEL_NAME = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v"
VAE_SUBFOLDER = "vae"

# === VAE 参数 ===
# 参考: autoencoder_kl_hunyuanvideo15.py
# - spatial_compression_ratio: 16
# - temporal_compression_ratio: 4
# - latent_channels: 32
# - scaling_factor: 1.03682
SPATIAL_COMPRESSION = 16
TEMPORAL_COMPRESSION = 4
LATENT_CHANNELS = 32
SCALING_FACTOR = 1.03682

# === 默认生成参数 ===
# 参考: generate_diffusers_flax.py 第 840-856 行
# 注意: 原始 61 帧 → 15 latent frames，不能被 8 TPU 整除
# 使用 64 帧 → 16 latent frames，能被 8 整除
DEFAULT_NUM_FRAMES = 64  # 调整为 64 帧以便 latent 维度能被 8 整除
DEFAULT_HEIGHT = 544     # 720p 默认高度 (720/16*12=544)
DEFAULT_WIDTH = 960      # 720p 默认宽度 (1280/16*12=960)

def record_time(call_method):
    """Record function execution time"""
    start = time.time()
    output = call_method()
    jax.block_until_ready(output)
    return output, (time.time() - start) * 1000

def create_mesh(num_devices=None):
    """Create device Mesh for sharding"""
    if num_devices is None:
        num_devices = jax.device_count()
    
    print(f"\nCreating device Mesh...")
    print(f"  Available devices: {num_devices}")
    
    devices = mesh_utils.create_device_mesh((num_devices,))
    mesh = Mesh(devices, ('tp',))
    
    print(f"  Mesh shape: {mesh.shape}")
    print(f"  Mesh axis names: {mesh.axis_names}")
    
    return mesh

def create_vae_flax(dtype=jnp.bfloat16):
    """Create Flax VAE model with default config (for testing without pretrained weights)"""
    print(f"\nCreating Flax VAE model with default config")
    print(f"  Target dtype: {dtype}")
    
    config = FlaxAutoencoderKLHunyuanVideo15Config()
    key = jax.random.key(0)
    rngs = jax.lax.stop_gradient(jax.random.split(key, 1))[0]
    from flax import nnx
    rngs = nnx.Rngs(rngs)
    
    vae = FlaxAutoencoderKLHunyuanVideo15(
        config=config,
        rngs=rngs,
        dtype=dtype,
    )
    
    print("VAE model created successfully")
    return vae

def load_vae_flax(dtype=jnp.bfloat16):
    """Load Flax VAE model from pretrained weights"""
    print(f"\nLoading Flax VAE model: {MODEL_NAME}")
    print(f"  Target dtype: {dtype}")
    
    try:
        vae = FlaxAutoencoderKLHunyuanVideo15.from_pretrained(
            MODEL_NAME,
            subfolder=VAE_SUBFOLDER,
            dtype=dtype,
        )
        print("VAE model loaded successfully")
    except Exception as e:
        print(f"Warning: Failed to load pretrained weights: {e}")
        print("Falling back to random initialization...")
        vae = create_vae_flax(dtype)
    
    return vae

def setup_tiling(vae):
    """
    Configure VAE Tiling for memory efficiency
    
    参考: generate_diffusers_flax.py 第 914-919 行
    默认启用 VAE Tiling 以节省 VMEM
    """
    print("\nConfiguring VAE Tiling...")
    
    # HunyuanVideo-1.5 默认 tile 尺寸 (参考 autoencoder_kl_hunyuanvideo15.py 第 688-694 行)
    vae.enable_tiling(
        tile_sample_min_height=256,   # 默认值
        tile_sample_min_width=256,    # 默认值
        tile_overlap_factor=0.25,     # 默认值
    )
    
    print(f"  Tile min size: {vae.tile_sample_min_height}x{vae.tile_sample_min_width}")
    print(f"  Tile latent min size: {vae.tile_latent_min_height}x{vae.tile_latent_min_width}")
    print(f"  Overlap factor: {vae.tile_overlap_factor:.2%}")
    print("Tiling configuration complete")

def create_decode_fn(vae, mesh):
    """
    Create decode function
    
    Note: No JIT on outer function to avoid loop unrolling
    
    关于 scaling_factor 的处理:
    - 参考: generate_diffusers_flax.py 第 516-522 行
    - VAEProxy.decode 中提到: "scaling_factor 已经在 Pipeline 中处理，这里不再重复处理"
    - 但在独立测试 VAE 时，我们需要手动处理 scaling_factor
    """
    print("\nPreparing VAE decode function...")
    
    scaling_factor = getattr(vae.config, 'scaling_factor', SCALING_FACTOR)
    print(f"  Scaling factor: {scaling_factor}")
    print(f"  Note: In pipeline, scaling is handled by Pipeline, not VAE")
    
    def decode_fn(latents):
        # 独立测试时需要手动应用 scaling_factor
        # Pipeline 中这个操作在 latents 传入 VAE 之前已经完成
        scaled_latents = latents / scaling_factor
        return vae.decode(scaled_latents)
    
    print("Decode function prepared (no JIT to avoid loop unrolling)")
    
    return decode_fn

def prepare_latents(mesh, frames, height, width, dtype=jnp.bfloat16):
    """
    Prepare test latents
    
    HunyuanVideo-1.5 compression (参考 autoencoder_kl_hunyuanvideo15.py):
    - Spatial: 16x (height/16, width/16)
    - Temporal: 4x (frames/4)
    - Latent channels: 32
    
    默认参数 (参考 generate_diffusers_flax.py 第 840-856 行):
    - frames: 61
    - height: 544 (720p)
    - width: 960 (720p)
    """
    latent_channels = LATENT_CHANNELS
    latent_frames = max(1, frames // TEMPORAL_COMPRESSION)
    latent_height = height // SPATIAL_COMPRESSION
    latent_width = width // SPATIAL_COMPRESSION
    
    global_shape = (1, latent_frames, latent_height, latent_width, latent_channels)
    
    print(f"\nCreating test Latents...")
    print(f"  Global shape: {global_shape}")
    print(f"  Data type: {dtype}")
    
    if latent_frames < mesh.size:
        sharding = NamedSharding(mesh, P())
        print(f"  Sharding strategy: Replicate (latent frames {latent_frames} < devices {mesh.size})")
    else:
        sharding = NamedSharding(mesh, P(None, 'tp', None, None, None))
        print(f"  Sharding strategy: Time dimension sharding")
    
    latents_np = np.random.randn(*global_shape).astype(np.float32)
    
    latents = jax.device_put(latents_np, sharding).astype(dtype)
    jax.block_until_ready(latents)
    
    num_elements = np.prod(global_shape)
    bytes_per_element = 2 if dtype == jnp.bfloat16 else 4
    memory_mb = (num_elements * bytes_per_element) / (1024 ** 2)
    
    print(f"  Latents created and sharded to devices")
    print(f"  Memory usage: {memory_mb:.2f} MB")
    
    return latents

def test_vae_decode(
    decode_fn,
    latents,
    num_runs=5,
    profiler_context=None,
):
    """Test VAE decode performance"""
    print(f"\n{'='*60}")
    print(f"Starting performance test - {num_runs} runs")
    print(f"{'='*60}\n")
    
    print("Warmup runs (3 rounds to compile all JIT variants)...")
    for i in range(3):
        output, time_cost = record_time(lambda: decode_fn(latents))
        jax.block_until_ready(output)
        print(f"  Warmup {i+1}: {time_cost:.2f} ms")
        if i == 2:
            print(f"  Output shape: {output.shape}")
            print(f"  Output dtype: {output.dtype}")
        
        del output
        jax.clear_caches()
        import gc
        gc.collect()
    
    print(f"\nFormal test ({num_runs} runs):")
    times = []
    
    context = profiler_context if profiler_context else nullcontext()
    
    with context:
        for run_idx in range(num_runs):
            output, time_cost = record_time(lambda: decode_fn(latents))
            times.append(time_cost)
            print(f"  Run {run_idx + 1}: {time_cost:.2f} ms")
            
            del output
            jax.clear_caches()
            
            import gc
            gc.collect()
    
    print(f"\n{'='*60}")
    print("Performance Statistics:")
    print(f"{'='*60}")
    print(f"  Mean: {np.mean(times):.2f} ms")
    print(f"  Min:  {np.min(times):.2f} ms")
    print(f"  Max:  {np.max(times):.2f} ms")
    print(f"  Std:  {np.std(times):.2f} ms")
    print(f"{'='*60}\n")

def main():
    """Main function"""
    warnings.filterwarnings('ignore', message='.*dtype.*int64.*truncated to dtype int32.*')
    logging.getLogger().setLevel(logging.ERROR)
    
    print("Configuring JAX environment...")
    jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    
    import os
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    
    print("JAX configuration complete")
    
    mesh = create_mesh()
    
    # 创建/加载 VAE 模型
    vae = create_vae_flax(dtype=jnp.bfloat16)
    
    # 启用 VAE Tiling（参考 generate_diffusers_flax.py 第 914-919 行）
    # 默认启用以节省 VMEM
    # setup_tiling(vae)  # 取消注释以启用 tiling
    
    # 创建 decode 函数
    decode_fn = create_decode_fn(vae, mesh)
    
    # 准备测试 latents
    # 使用默认的 61 帧和 720p 分辨率（参考 generate_diffusers_flax.py）
    latents = prepare_latents(
        mesh=mesh,
        frames=DEFAULT_NUM_FRAMES,  # 61 帧
        height=DEFAULT_HEIGHT,       # 544
        width=DEFAULT_WIDTH,         # 960
        dtype=jnp.bfloat16,
    )
    
    profiler_context = None
    if False:
        print("\nEnabling JAX Profiler...")
        profiler_context = jax.profiler.trace(
            "/dev/shm/jax-trace",
            create_perfetto_link=False
        )
    
    test_vae_decode(
        decode_fn=decode_fn,
        latents=latents,
        num_runs=5,
        profiler_context=profiler_context,
    )
    
    print("All tests completed!")

if __name__ == "__main__":
    main()