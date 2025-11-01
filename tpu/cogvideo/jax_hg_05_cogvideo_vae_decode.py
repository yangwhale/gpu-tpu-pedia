import time
import numpy as np
import jax
import jax.numpy as jnp
import torch
import warnings
from diffusers import AutoencoderKLCogVideoX
from diffusers.models.autoencoders.vae import DecoderOutput
import torchax
from jax.tree_util import register_pytree_node
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
from jax.experimental import mesh_utils
from contextlib import nullcontext
import logging

MODEL_NAME = "zai-org/CogVideoX1.5-5B"
VAE_SUBFOLDER = "vae"

def setup_pytree_registrations():
    print("注册PyTree节点...")
    def model_output_flatten(obj):
        return obj.to_tuple(), type(obj)
    def model_output_unflatten(aux, children):
        return aux(*children)
    for cls in [DecoderOutput]:
        register_pytree_node(cls, model_output_flatten, model_output_unflatten)
        print(f"  - {cls.__name__} 已注册")

def record_time(call_method):
    start = time.time()
    output = call_method()
    if hasattr(output, 'sample'):
        jax.block_until_ready(output.sample)
    else:
        jax.block_until_ready(output)
    return output, (time.time() - start) * 1000

def shard_weights_vae(mesh, weights):
    result = {}
    for k, v in weights.items():
        v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
        result[k] = v
    return result

def setup_vae_for_jax(vae):
    print("\n配置VAE以使用JAX...")
    tp_dim, dp_dim, sp_dim = jax.device_count(), 1, 1
    
    print(f"  Mesh 维度: tp_dim={tp_dim}, dp_dim={dp_dim}, sp_dim={sp_dim}")
    
    mesh_devices = mesh_utils.create_device_mesh((tp_dim, dp_dim, sp_dim), allow_split_physical_axes=True)
    mesh = Mesh(mesh_devices, ('tp', 'dp', 'sp'))
    env = torchax.default_env()
    env._mesh = mesh

    def _move_module_to_xla(module):
        with jax.default_device('cpu'):
            state_dict = module.state_dict()
            for k, v in state_dict.items():
                if hasattr(v, 'dtype') and v.dtype == torch.float32:
                    state_dict[k] = v.to(torch.bfloat16)
            state_dict = env.to_xla(state_dict)
            module.load_state_dict(state_dict, assign=True)
    
    with env:
        print("- 将VAE移到XLA并进行分片...")
        _move_module_to_xla(vae)
        vae_weights = shard_weights_vae(mesh, vae.state_dict())
        vae.load_state_dict(vae_weights, assign=True, strict=False)
        torchax.interop.call_jax(jax.block_until_ready, vae_weights)
        
        # 只编译 decoder.forward，不编译 tiling 循环
        # 优势：编译快（~2分钟 vs ~10分钟）
        # 代价：推理慢（~3.5秒 vs ~1毫秒）
        # 适合：开发调试阶段
        print("- 编译 Decoder (只编译核心计算，tiling循环在Python执行)...")
        vae.decoder = torchax.compile(
            vae.decoder,
            torchax.CompileOptions(
                methods_to_compile=['forward'],
                jax_jit_kwargs={'static_argnames': ('temb',)}
            )
        )
        
        # 如需生产环境的极速推理（1毫秒），改用以下配置：
        # vae = torchax.compile(
        #     vae,
        #     torchax.CompileOptions(
        #         methods_to_compile=['decode'],
        #         jax_jit_kwargs={'static_argnames': ('return_dict',)}
        #     )
        # )
    print("VAE配置完成")
    return vae, env, mesh

def test_vae_decode(vae, env, mesh, profiler_context, num_runs, frames, height, width):

    print(f"\n开始测试 - 运行 {num_runs} 次\n")
    latent_channels = 16
    latent_frames = max(1, frames // 4)
    latent_height = height // 8
    latent_width = width // 8
    # (B, C, T, H, W)
    global_shape = (
        1, latent_channels, latent_frames,
        latent_height, latent_width
    )
    print(f"  全局 Latent 形状: {global_shape}")

    sharding = NamedSharding(mesh, P(None, None, 'tp', None, None))
    print(f"  分片规则: {sharding}")

    # 在 CPU 上创建全局数据 (用 numpy)
    # 将 numpy 数组 JIT 地加载到 TPU 上，并应用分片
    # JAX 会自动处理，只在每张卡上创建它需要的那"一片"
    latents_np = np.random.randn(*global_shape).astype(jnp.float32)
    latents_jax = jax.device_put(latents_np, sharding).astype(jnp.bfloat16)
    jax.block_until_ready(latents_jax)
    
    # 将分片的 JAX 数组转换为 torchax 张量
    latents = env.j2t_iso(latents_jax)
    # with env:
    #     latents = torch.randn(1, latent_channels, latent_frames, latent_height, latent_width, dtype=torch.bfloat16).to('jax')
    
    with env:
        # 预热运行
        output, time_cost = record_time(lambda: vae.decode(latents).sample)                
        print(f"预热运行: {time_cost:.2f} ms")
        del output
    with env, profiler_context:
        for run_idx in range(num_runs):
            output, time_cost = record_time(lambda: vae.decode(latents).sample)                
            print(f"第 {run_idx + 1} 次运行: {time_cost:.2f} ms")
            # 清理输出，保留 latents 以便下次循环使用
            del output

def main():
    # 过滤TPU不支持64位数据类型的警告（TPU会自动将int64截断为int32）
    warnings.filterwarnings('ignore', message='.*dtype.*int64.*truncated to dtype int32.*')
    logging.getLogger().setLevel(logging.ERROR)
    
    # JAX配置
    jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    # jax.config.update("jax_log_compiles", True)
    torch.set_default_dtype(torch.bfloat16)

    # 加载VAE模型
    print(f"正在加载VAE模型: {MODEL_NAME}")
    vae = AutoencoderKLCogVideoX.from_pretrained(MODEL_NAME, subfolder=VAE_SUBFOLDER).to(dtype=torch.bfloat16)
    print("VAE模型加载完成")
    
    # 启用 Tiling 以节省显存
    print("\n配置 VAE Tiling...")
    vae.enable_tiling(
        tile_sample_min_height=192,  # 默认 480//2
        tile_sample_min_width=340,   # 默认 720//2
        tile_overlap_factor_height=1/6,  # 16.7% 垂直重叠
        tile_overlap_factor_width=1/5,   # 20% 水平重叠
    )
    print(f"  - Tile 最小尺寸: {vae.tile_sample_min_height}x{vae.tile_sample_min_width}")
    print(f"  - 重叠因子: 垂直={vae.tile_overlap_factor_height:.2%}, 水平={vae.tile_overlap_factor_width:.2%}")
    print("Tiling 配置完成")
    
    setup_pytree_registrations()
    vae, env, mesh = setup_vae_for_jax(vae)
    
    # 创建一个空的上下文管理器
    profiler_context = nullcontext()
    if False:
        print("启用JAX profiler...")
        profiler_context = jax.profiler.trace("/dev/shm/jax-trace", create_perfetto_link=False)
    
    # 运行测试
    test_vae_decode(vae, env, mesh, profiler_context, num_runs=5, frames=32, height=768, width=1360)
    print("✅ 测试完成！")

if __name__ == "__main__":
    main()