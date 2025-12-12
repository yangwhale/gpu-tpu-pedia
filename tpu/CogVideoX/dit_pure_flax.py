# 导入必要的库
import os
os.environ.setdefault('JAX_MEMORY_DEBUG', '0')  # 默认关闭内存调试

import sys
import time
import re
import math
import functools
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
from jax.experimental.pallas.ops.tpu import splash_attention
from jax.experimental.shard_map import shard_map
try:
    from jax import shard_map
except ImportError:
    from jax.experimental.shard_map import shard_map
from jax.experimental import mesh_utils
from tqdm import tqdm
import warnings
import logging
from contextlib import nullcontext

# 添加 diffusers-tpu 路径
sys.path.insert(0, '/home/chrisya/diffusers-tpu/src')

from diffusers.models.transformers.cogvideox_transformer_3d_flax import (
    FlaxCogVideoXTransformer3DModel,
    FlaxCogVideoXTransformer3DConfig,
    set_global_mesh,
)

# --- 全局配置 ---
MODEL_NAME = "zai-org/CogVideoX1.5-5B"

#### Splash Attention 配置参数 ####
# Splash attention 块大小配置
BQSIZE = 2048           # Query 块大小
BKVSIZE = 2048          # Key/Value 块大小
BKVCOMPUTESIZE = 1024   # Key/Value 计算块大小

# 窗口大小（None 表示使用完整注意力）
WINDOW_SIZE = None

# 是否使用 K-smooth（对 key 进行平滑处理）
USE_K_SMOOTH = True

# Mesh 分片配置
USE_DP = False          # 是否使用 data parallelism
SP_NUM = 1              # Spatial parallelism 数量
USE_TP = True           # 是否使用 Tensor Parallel 模式（Megatron Column-Row风格）


# --- Splash Attention 实现 ---

def _tpu_splash_attention(query, key, value, mesh, scale=None, is_causal=False, window_size=None):
    """
    TPU Splash Attention 实现（纯 JAX 版本）
    
    使用 JAX 的 Splash Attention 在 TPU 上高效计算注意力
    
    Args:
        query: Query 张量 (batch, heads, seq_len, head_dim)
        key: Key 张量 (batch, heads, seq_len, head_dim)
        value: Value 张量 (batch, heads, seq_len, head_dim)
        mesh: JAX 设备网格
        scale: 缩放因子（默认为 1/sqrt(head_dim)）
        is_causal: 是否使用因果掩码
        window_size: 局部注意力窗口大小
    """
    num_heads = query.shape[1]

    def _attention_on_slices(q, k, v):
        # 缩放 query 张量
        scale_factor = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
        q = q * scale_factor

        def pad_to_multiple(x, multiple, axis):
            seq_len = x.shape[axis]
            pad_len = (multiple - seq_len % multiple) % multiple
            if pad_len == 0:
                return x, seq_len
            pad_width = [(0, 0)] * x.ndim
            pad_width[axis] = (0, pad_len)
            return jnp.pad(x, pad_width), seq_len

        def kernel_3d(q_3d, k_3d, v_3d):
            q_seq_len = q_3d.shape[1]
            kv_seq_len = k_3d.shape[1]
            num_heads_on_device = q_3d.shape[0]

            # 填充到块大小的倍数
            q_3d_padded, q_orig_len = pad_to_multiple(q_3d, BQSIZE, axis=1)
            k_3d_padded, k_orig_len = pad_to_multiple(k_3d, BKVSIZE, axis=1)
            v_3d_padded, v_orig_len = pad_to_multiple(v_3d, BKVSIZE, axis=1)

            padded_q_seq_len = q_3d_padded.shape[1]
            padded_kv_seq_len = k_3d_padded.shape[1]

            # 创建注意力掩码
            if window_size is not None:
                mask_class = functools.partial(splash_attention.LocalMask, window_size=window_size, offset=0)
            else:
                mask_class = splash_attention.FullMask

            mask = splash_attention.MultiHeadMask(
                [mask_class((padded_q_seq_len, padded_kv_seq_len)) for _ in range(num_heads_on_device)]
            )

            # 配置块大小
            block_sizes = splash_attention.BlockSizes(
                block_q=min(BQSIZE, padded_q_seq_len),
                block_kv=min(BKVSIZE, padded_kv_seq_len),
                block_kv_compute=min(BKVCOMPUTESIZE, padded_kv_seq_len),
            )
            
            # 创建并执行 Splash attention kernel
            splash_kernel = splash_attention.make_splash_mha(
                mask=mask, block_sizes=block_sizes, head_shards=1, q_seq_shards=1
            )
            out = splash_kernel(q_3d_padded, k_3d_padded, v_3d_padded)
            
            # 移除填充
            return out[:, :q_orig_len, ...]

        # 在批次维度上映射 kernel
        vmapped_kernel = jax.vmap(kernel_3d, in_axes=(0, 0, 0), out_axes=0)
        return vmapped_kernel(q, k, v)

    # 根据设备数量和头数确定分片策略
    if num_heads < mesh.size:
        q_partition_spec = P()
        kv_partition_spec = P()
    else:
        if query.shape[2] == key.shape[2]:  # 自注意力
            q_partition_spec = P('dp', 'tp', 'sp', None)
            kv_partition_spec = P('dp', 'tp', None, None)
        else:  # 交叉注意力
            q_partition_spec = P('dp', None, ('tp', 'sp'), None)
            kv_partition_spec = P('dp', None, None, None)

    # 使用 shard_map 在设备间分片执行
    sharded_fn = shard_map(
        _attention_on_slices,
        mesh=mesh,
        in_specs=(q_partition_spec, kv_partition_spec, kv_partition_spec),
        out_specs=q_partition_spec,
        check_rep=False,
    )
    out = sharded_fn(query, key, value)
    
    return out


def splash_attention_fn(query, key, value, mesh, scale=None, is_causal=False, window_size=None):
    """
    Splash Attention 封装函数
    
    Args:
        query: Query 张量 (batch, heads, seq_len, head_dim)
        key: Key 张量 (batch, heads, seq_len, head_dim)
        value: Value 张量 (batch, heads, seq_len, head_dim)
        mesh: JAX 设备网格
        scale: 缩放因子
        is_causal: 是否使用因果掩码
        window_size: 局部注意力窗口大小
    
    Returns:
        attention 输出
    """
    # 可选的 K-smooth 处理
    if USE_K_SMOOTH:
        key_mean = jnp.mean(key, axis=2, keepdims=True)
        key = key - key_mean
    
    return _tpu_splash_attention(query, key, value, mesh, scale=scale, is_causal=is_causal, window_size=window_size)


# --- Transformer 权重分片策略 ---

# Transformer sharding策略 - Tensor Parallel模式（默认，Megatron Column-Row风格）
# Flax NNX 路径格式: transformer_blocks.0.attn1.to_q.kernel
transformer_shardings_tp = {
    # Attention layers - 在输出维度分片（按heads切分）
    r'.*\.to_q\.kernel$': (None, ('tp', 'sp')),
    r'.*\.to_k\.kernel$': (None, ('tp', 'sp')),
    r'.*\.to_v\.kernel$': (None, ('tp', 'sp')),
    r'.*\.to_out\.kernel$': (('tp', 'sp'), None),
    # Feedforward layers - Flax NNX 使用 linear1/linear2
    r'.*\.ff\.linear1\.kernel$': (None, ('tp', 'sp')),
    r'.*\.ff\.linear2\.kernel$': (('tp', 'sp'), None),
}

# Transformer sharding策略 - FSDP模式（在输入维度均匀分片）
transformer_shardings_fsdp = {
    # Attention layers - 在输入维度分片
    r'.*\.to_q\.kernel$': (('tp', 'sp'), None),
    r'.*\.to_k\.kernel$': (('tp', 'sp'), None),
    r'.*\.to_v\.kernel$': (('tp', 'sp'), None),
    r'.*\.to_out\.kernel$': (None, ('tp', 'sp')),
    # Feedforward layers
    r'.*\.ff\.linear1\.kernel$': (('tp', 'sp'), None),
    r'.*\.ff\.linear2\.kernel$': (None, ('tp', 'sp')),
}


def shard_weights_transformer(mesh, model, use_tp=True):
    """
    对纯 Flax Transformer 模型的权重进行分片
    
    Args:
        mesh: JAX设备网格
        model: Flax NNX 模型
        use_tp: 是否使用Tensor Parallel模式（默认True），否则使用FSDP模式
        
    Returns:
        分片后的模型
    """
    # 选择分片策略
    sharding_dict = transformer_shardings_tp if use_tp else transformer_shardings_fsdp
    
    # 提取模型状态
    graphdef, state = nnx.split(model)
    
    # 使用 flat_state() 获取扁平化状态
    flat = state.flat_state()
    keys = flat._keys  # 元组列表，如 [('norm_final', 'bias'), ...]
    values = flat._values  # Param 对象列表
    
    sharded_count = 0
    replicated_count = 0
    
    # 对每个参数应用分片
    new_values = []
    for key_tuple, param in zip(keys, values):
        # 将元组路径转换为点分隔字符串
        path_str = '.'.join(str(k) for k in key_tuple)
        
        # 获取参数值
        value = param.value
        
        # 尝试匹配分片规则
        matched = False
        for pattern, sharding in sharding_dict.items():
            if re.search(pattern, path_str) is not None:
                # 找到匹配的模式，应用分片
                sharded_value = jax.device_put(value, NamedSharding(mesh, P(*sharding)))
                # 创建新的 Param 对象
                new_param = type(param)(sharded_value)
                new_values.append(new_param)
                matched = True
                sharded_count += 1
                break
        
        if not matched:
            # 没有匹配到任何模式，复制到所有设备
            sharded_value = jax.device_put(value, NamedSharding(mesh, P()))
            new_param = type(param)(sharded_value)
            new_values.append(new_param)
            replicated_count += 1
    
    print(f"  分片: {sharded_count} 个参数, 复制: {replicated_count} 个参数")
    
    # 重建状态 - 使用 FlatState 和 to_nested_state
    from flax.nnx.statelib import FlatState
    # FlatState 需要 items: Iterable[tuple[PathParts, V]]
    items = list(zip(keys, new_values))
    new_flat = FlatState(items, sort=False)
    new_state = new_flat.to_nested_state()
    
    # 重新合并
    model = nnx.merge(graphdef, new_state)
    
    return model


def create_mesh():
    """
    创建 JAX 设备网格
    
    Returns:
        mesh: JAX Mesh 对象
    """
    # 计算mesh维度
    tp_dim, dp_dim, sp_dim = jax.device_count(), 1, 1
    if USE_DP:
        tp_dim //= 2
        dp_dim = 2
    
    if SP_NUM > 1:
        tp_dim //= SP_NUM
        sp_dim = SP_NUM
    
    print(f"  Mesh 维度: tp_dim={tp_dim}, dp_dim={dp_dim}, sp_dim={sp_dim}")
    print(f"  总设备数: {jax.device_count()}")
    
    # 创建三维 mesh (tp, dp, sp)
    mesh_devices = mesh_utils.create_device_mesh((tp_dim, dp_dim, sp_dim), allow_split_physical_axes=True)
    mesh = Mesh(mesh_devices, ('tp', 'dp', 'sp'))
    
    return mesh

# --- 性能测试工具函数 ---

def record_time_tpu(call_method):
    """
    记录一个函数调用的执行时间（TPU版本）
    使用jax.block_until_ready确保计算完成
    
    参数:
    call_method (function): 需要被测量时间的函数。
    
    返回:
    tuple: 一个元组，包含被调用函数的输出和以毫秒为单位的执行时间。
    """
    start = time.time()
    output = call_method()
    
    # 确保JAX计算完成
    jax.block_until_ready(output)
    
    end = time.time()
    return output, (end - start) * 1000  # s -> ms


def record_peak_memory_tpu(call_method):
    """
    记录一个函数调用期间的 TPU 峰值显存使用量和执行时间
    
    注意: TPU的内存监控与GPU不同，这里使用JAX的内存统计
    
    参数:
    call_method (function): 需要被测量的函数。
    
    返回:
    tuple: 一个元组，包含被调用函数的输出、以 MB 为单位的峰值显存和以毫秒为单位的执行时间。
    """
    # 调用 record_time_tpu 来执行函数并获取其输出和执行时间
    output, time_cost = record_time_tpu(call_method)
    
    # TPU 内存统计（占位值）
    peak_memory_mb = 0.0
    
    return output, peak_memory_mb, time_cost


# --- 结果打印 ---

def print_results(results, frames):
    """
    打印测试结果的统计信息。
    
    参数:
    results (list of dict): 测试结果的列表。
    frames (int): 测试的帧数。
    """
    if not results:
        print("没有测试结果")
        return
    
    times = [r['time'] for r in results]
    
    print(f"\n=== DiT Pure Flax 测试结果 (帧数: {frames}) ===")
    print(f"运行次数: {len(results)}")
    print(f"\n执行时间 (ms):")
    print(f"  平均值: {sum(times)/len(times):.2f}")
    print(f"  最小值: {min(times):.2f}")
    print(f"  最大值: {max(times):.2f}")
    
    if len(times) > 1:
        print(f"\n首次运行（含JIT编译）: {times[0]:.2f} ms")
        avg_time = sum(times[1:]) / len(times[1:])
        print(f"后续运行平均时间: {avg_time:.2f} ms")
        if times[0] > 0:
            print(f"加速比: {times[0] / avg_time:.2f}x")


# --- DiT 模型性能测试核心函数 ---

def dit_test(transformer_fn, frames=64, num_runs=1, warmup_runs=1, profiler_context=None):
    """
    测试 DiT (Diffusion Transformer) 模型在TPU上的性能（纯Flax版本）。
    先进行预热运行，然后对指定帧数重复运行多次以获取稳定的性能数据。
    
    参数:
    transformer_fn: JIT编译后的模型函数
    frames (int): 测试的视频帧数，默认64帧（必须能被4整除）。
    num_runs (int): 重复运行的次数，默认10次。
    warmup_runs (int): 预热运行次数，默认3次（不计入统计）。
    profiler_context: Profiler上下文管理器，可选。
    """
    # DiT 模型期望的输入维度（与 dit_flax.py 保持一致）
    batch = 1
    channel = 16
    height = 80   # 对应 sample_height
    width = 160   # 对应 sample_width
    
    # Transformer 的输入帧数，考虑 temporal_compression_ratio
    latent_frames = frames // 4
    
    # rotary embedding 维度（与模型的 attention_head_dim 相关）
    rotary_dim = 64  # attention_head_dim
    
    # 定义运行单次测试的函数
    def run_single_test():
        # --- 准备模型输入 (JAX arrays, channel-last format) ---
        # 1. 创建主要的输入张量 (hidden_states) - (B, T, H, W, C)
        key = jax.random.key(42)
        input_tensor = jax.random.normal(key, (batch, latent_frames, height, width, channel), dtype=jnp.bfloat16)
        
        # 2. 创建文本嵌入 (encoder_hidden_states) - (B, text_seq_len, text_embed_dim)
        text_seq_len = 10  # max_text_seq_length from config
        text_embed_dim = 4096
        key, subkey = jax.random.split(key)
        input_embd = jax.random.normal(subkey, (batch, text_seq_len, text_embed_dim), dtype=jnp.bfloat16)
        
        # 3. 创建时间步 (timestep)
        timestep = jnp.array([999], dtype=jnp.float32)
        
        # 4. 创建旋转位置编码 (image_rotary_emb)
        # 计算 token 数量：latent_frames * (height // patch_size) * (width // patch_size) // 2
        # patch_size = 2, 根据 CogVideoX 的结构
        patch_h, patch_w = height // 2, width // 2
        tokens = latent_frames * patch_h * patch_w // 2  # temporal compression factor
        key, subkey = jax.random.split(subkey)
        rotary_emb_cos = jax.random.normal(subkey, (tokens, rotary_dim), dtype=jnp.float32)
        key, subkey = jax.random.split(key)
        rotary_emb_sin = jax.random.normal(subkey, (tokens, rotary_dim), dtype=jnp.float32)
        image_rotary_emb = (rotary_emb_cos, rotary_emb_sin)
        
        # 定义调用 Transformer 模型的函数
        def dit_call():
            return transformer_fn(
                input_tensor,
                input_embd,
                timestep,
                image_rotary_emb
            )
        
        # 记录执行时间
        output, peak_memory_mb, time_cost = record_peak_memory_tpu(dit_call)
        del output  # 释放内存
        
        return peak_memory_mb, time_cost
    
    # 预热运行
    print(f"开始预热运行 (预热次数: {warmup_runs})")
    for run in tqdm(range(warmup_runs), desc="Warmup DiT on TPU"):
        try:
            run_single_test()
        except Exception as e:
            print(f"预热第 {run + 1} 次运行出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    # 正式测试运行
    results = []
    print(f"\n开始正式测试 DiT Pure Flax 性能 (帧数: {frames}, 运行次数: {num_runs})")
    
    context = profiler_context if profiler_context else nullcontext()
    
    with context:
        for run in tqdm(range(num_runs), desc="Testing DiT on TPU"):
            try:
                peak_memory_mb, time_cost = run_single_test()
                
                results.append({
                    'run': run + 1,
                    'peak_memory_mb': peak_memory_mb,
                    'time': time_cost
                })

            except Exception as e:
                print(f"第 {run + 1} 次运行出错: {str(e)}")
                import traceback
                traceback.print_exc()
                break

    return results


# --- 主测试流程 ---

def load_transformer(model_name=MODEL_NAME, dtype=jnp.bfloat16, use_pretrained=True):
    """
    加载或创建纯 Flax Transformer模型
    
    Args:
        model_name: 预训练模型名称（仅当 use_pretrained=True 时使用）
        dtype: 数据类型
        use_pretrained: 是否加载预训练权重
        
    Returns:
        transformer: FlaxCogVideoXTransformer3DModel 实例
    """
    if use_pretrained:
        print(f"正在加载预训练模型: {model_name}")
        print("使用 FlaxCogVideoXTransformer3DModel.from_pretrained()")
        
        transformer = FlaxCogVideoXTransformer3DModel.from_pretrained(
            model_name,
            subfolder="transformer",
            dtype=dtype,
        )
    else:
        print("创建随机初始化的 Transformer 模型（用于测试结构）")
        
        # 创建配置（与 dit_flax.py 保持一致的分辨率）
        # 基于 CogVideoX-2B 的配置
        config = FlaxCogVideoXTransformer3DConfig(
            num_attention_heads=30,  # 减小到30个头
            attention_head_dim=64,
            in_channels=16,
            out_channels=16,
            time_embed_dim=512,
            text_embed_dim=4096,
            num_layers=30,  # 减小到30层
            sample_width=160,  # 与 dit_flax.py 保持一致
            sample_height=80,   # 与 dit_flax.py 保持一致
            patch_size=2,
            temporal_compression_ratio=4,
            max_text_seq_length=226,
        )
        
        # 创建模型
        key = jax.random.key(0)
        rngs = nnx.Rngs(key)
        transformer = FlaxCogVideoXTransformer3DModel(
            config=config,
            rngs=rngs,
            dtype=dtype,
        )
    
    print("模型加载完成")
    return transformer


def compile_transformer(transformer):
    """
    使用 nnx.jit 编译 Transformer
    
    Args:
        transformer: FlaxCogVideoXTransformer3DModel 实例
        
    Returns:
        compiled_fn: JIT 编译后的函数
    """
    print("\n编译 Transformer...")
    
    # 使用 nnx.jit 装饰器为 NNX 模型创建 JIT 编译函数
    # 这样可以正确处理模型状态，并支持编译缓存
    @nnx.jit
    def forward_fn(model, hidden_states, encoder_hidden_states, timestep, image_rotary_emb):
        return model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            image_rotary_emb=image_rotary_emb,
            deterministic=True,
            return_dict=False,
        )
    
    # 返回一个包装函数，将 transformer 作为第一个参数传入
    def compiled_fn(hidden_states, encoder_hidden_states, timestep, image_rotary_emb):
        return forward_fn(transformer, hidden_states, encoder_hidden_states, timestep, image_rotary_emb)
    
    print("编译完成")
    return compiled_fn


def setup_transformer_for_tpu(transformer, mesh):
    """
    设置 Transformer 以在 TPU 上运行（分片权重）
    
    Args:
        transformer: 已加载的 Transformer 模型
        mesh: JAX mesh
        
    Returns:
        transformer: 分片后的 Transformer 模型
    """
    print("\n配置 Transformer 以使用分片...")
    
    # 设置全局 mesh，让 Splash Attention 使用多设备分片
    print(f"- 设置全局 Mesh (size={mesh.size})...")
    set_global_mesh(mesh)
    
    # 对权重进行分片
    print(f"- 对 Transformer 权重进行分片 (TP={USE_TP})...")
    transformer = shard_weights_transformer(mesh, transformer, use_tp=USE_TP)
    
    # 确保所有权重已分片完成
    graphdef, state = nnx.split(transformer)
    jax.block_until_ready(state)
    
    print("Transformer 配置完成")
    return transformer


def dit(frames=64, num_runs=10):
    """
    执行 DiT 模型在TPU上的性能测试（纯Flax版本）。
    
    参数:
    frames (int): 测试的视频帧数，默认64帧（必须能被4整除）。
    num_runs (int): 重复运行的次数，默认10次。
    """
    print("--- 开始 DiT Pure Flax TPU 性能测试 ---")
    
    # 设置JAX配置
    warnings.filterwarnings('ignore')
    logging.getLogger().setLevel(logging.ERROR)
    
    # 设置随机种子
    import random
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    
    # 创建 mesh
    print("\n配置 JAX Mesh...")
    mesh = create_mesh()
    
    # 加载 Transformer 模型
    # 设置 use_pretrained=True 来加载预训练权重
    transformer = load_transformer(dtype=jnp.bfloat16, use_pretrained=True)
    
    # 设置 Transformer 在 TPU 上运行（分片权重）
    transformer = setup_transformer_for_tpu(transformer, mesh)
    
    # Profiler 配置
    profiler_context = None
    if False:  # 设为 True 启用 profiling
        print("\n启用 JAX Profiler...")
        profiler_context = jax.profiler.trace(
            "/dev/shm/jax-trace",
            create_perfetto_link=False
        )
    
    # 在 mesh 上下文中执行测试
    with mesh:
        # 编译 Transformer (启用 GSPMD)
        transformer_fn = compile_transformer(transformer)

        # 执行 DiT 测试
        results = dit_test(
            transformer_fn,
            frames=frames,
            num_runs=num_runs,
            profiler_context=profiler_context,
        )
    
    # 打印统计结果
    print_results(results, frames)


# --- 脚本执行入口 ---
if __name__ == "__main__":
    # Set JAX config to enable compilation cache
    jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    
    warnings.filterwarnings('ignore')
    logging.getLogger().setLevel(logging.ERROR)
    
    # 执行 DiT 的TPU性能测试
    dit(frames=64, num_runs=10)