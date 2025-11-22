# 导入必要的库
import os
os.environ.setdefault('JAX_MEMORY_DEBUG', '0')  # 默认关闭内存调试

import time
import re
import math
import functools
import numpy as np
import jax
import jax.numpy as jnp
import torch
import torchax
from torchax.ops import ops_registry
from jax.tree_util import register_pytree_node
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
from jax.experimental.pallas.ops.tpu import splash_attention
from jax.experimental.shard_map import shard_map
from jax.experimental import mesh_utils
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutputWithPastAndCrossAttentions
from diffusers import CogVideoXTransformer3DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from tqdm import tqdm
import warnings
import logging

# --- 全局配置 ---
MODEL_NAME = "zai-org/CogVideoX1.5-5B"

#### Splash Attention 配置参数 ####
# Splash attention 块大小配置
BQSIZE = 2048           # Query 块大小
BKVSIZE = 1024          # Key/Value 块大小
BKVCOMPUTESIZE = 512    # Key/Value 计算块大小

# 窗口大小（None 表示使用完整注意力）
WINDOW_SIZE = None

# 是否使用 K-smooth（对 key 进行平滑处理）
USE_K_SMOOTH = True

# Mesh 分片配置
USE_DP = False          # 是否使用 data parallelism
SP_NUM = 1             # Spatial parallelism 数量
USE_FSDP = True        # 是否使用 FSDP 模式（vs Tensor Parallel）


# --- PyTree 注册 ---

def setup_pytree_registrations():
    """
    注册必要的pytree节点以支持JAX转换
    """
    print("注册PyTree节点...")
    
    def model_output_flatten(obj):
        """将模型输出对象展平为元组"""
        return obj.to_tuple(), type(obj)

    def model_output_unflatten(aux, children):
        """从元组重建模型输出对象"""
        return aux(*children)
    
    OUTPUT_CLASSES = [
        BaseModelOutputWithPooling,
        BaseModelOutputWithPastAndCrossAttentions,
        Transformer2DModelOutput,
    ]

    for cls in OUTPUT_CLASSES:
        register_pytree_node(cls, model_output_flatten, model_output_unflatten)
        print(f"  - {cls.__name__} 已注册")


# --- Splash Attention 实现 ---

def _sdpa_reference(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
) -> torch.Tensor:
    """
    Scaled Dot-Product Attention 参考实现
    用于在不支持 Splash attention 时作为回退方案
    """
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(
            L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    if dropout_p > 0:
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


def _tpu_splash_attention(query, key, value, env, scale=None, is_causal=False, window_size=None):
    """
    TPU Splash Attention 实现
    
    使用 JAX 的 Splash Attention 在 TPU 上高效计算注意力
    """
    mesh = env._mesh
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


def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
    env=None,
    window_size=None,
) -> torch.Tensor:
    """
    Scaled Dot-Product Attention 封装函数
    根据环境配置选择使用 TPU Splash Attention 或参考实现
    """
    if env is not None and hasattr(env.config, 'use_tpu_splash_attention') and env.config.use_tpu_splash_attention:
        # 使用 TPU Splash Attention
        jquery, jkey, jvalue = env.t2j_iso((query, key, value))
        
        # 可选的 K-smooth 处理
        if USE_K_SMOOTH:
            key_mean = jnp.mean(jkey, axis=2, keepdims=True)
            jkey = jkey - key_mean
        
        res = _tpu_splash_attention(jquery, jkey, jvalue, env, scale=scale, is_causal=is_causal, window_size=window_size)
        return env.j2t_iso(res)
    
    # 回退到参考实现
    return _sdpa_reference(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)


# --- Transformer 权重分片策略 ---

# Transformer sharding策略 - FSDP模式（默认）
transformer_shardings_fsdp = {
    # Attention layers - 在输出维度分片
    r'.*\.to_q\.weight$': (None, ('tp', 'sp')),
    r'.*\.to_k\.weight$': (None, ('tp', 'sp')),
    r'.*\.to_v\.weight$': (None, ('tp', 'sp')),
    r'.*\.to_out.*\.weight$': (('tp', 'sp'), None),
    # Feedforward layers
    r'.*\.ff\.net\.0\.weight$': (None, ('tp', 'sp')),
    r'.*\.ff\.net\.2\.weight$': (('tp', 'sp'), None),
}

# Transformer sharding策略 - Tensor Parallel模式
transformer_shardings_tp = {
    # Attention layers - 在输入维度分片
    r'.*\.to_q\.weight$': (('tp', 'sp'), None),
    r'.*\.to_k\.weight$': (('tp', 'sp'), None),
    r'.*\.to_v\.weight$': (('tp', 'sp'), None),
    r'.*\.to_out.*\.weight$': (None, ('tp', 'sp')),
    # Feedforward layers
    r'.*\.ff\.net\.0\.weight$': (('tp', 'sp'), None),
    r'.*\.ff\.net\.2\.weight$': (None, ('tp', 'sp')),
}


def shard_weights_transformer(mesh, weights, use_fsdp=True):
    """
    对CogVideoX Transformer模型的权重进行分片
    
    Args:
        mesh: JAX设备网格
        weights: 模型权重字典
        use_fsdp: 是否使用FSDP模式（默认True），否则使用Tensor Parallel模式
        
    Returns:
        分片后的权重字典
    """
    # 选择分片策略
    sharding_dict = transformer_shardings_fsdp if use_fsdp else transformer_shardings_tp
    
    result = {}
    for k, v in weights.items():
        # 尝试匹配分片规则
        matched = False
        for target, sharding in sharding_dict.items():
            if re.fullmatch(target, k) is not None:
                # 找到匹配的模式，应用分片
                v.apply_jax_(jax.device_put, NamedSharding(mesh, P(*sharding)))
                matched = True
                break
        
        if not matched:
            # 没有匹配到任何模式，复制到所有设备
            v.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
        
        result[k] = v
    return result


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
    # 注意：torchax返回的是包装后的Tensor，需要访问底层的JAX array (_elem)
    # 或者依赖 torchax 的自动解包机制，但显式访问更安全
    target = output
    if hasattr(output, 'sample'):
        target = output.sample

    if hasattr(target, '_elem'):
        jax.block_until_ready(target._elem)
    else:
        jax.block_until_ready(target)
    
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
    
    # TPU 内存统计（可能需要根据实际环境调整）
    # JAX没有直接的峰值内存API，这里返回一个占位值
    # 实际使用时可以通过TPU profiler或其他工具获取
    peak_memory_mb = 0.0  # 占位值
    
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
    
    print(f"\n=== DiT TPU 测试结果 (帧数: {frames}) ===")
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

def dit_test(transformer, frames=64, num_runs=1, warmup_runs=1):
    """
    测试 DiT (Diffusion Transformer) 模型在TPU上的性能。
    先进行预热运行，然后对指定帧数重复运行多次以获取稳定的性能数据。
    
    参数:
    transformer (torch.nn.Module): 已加载的 Transformer 模型。
    frames (int): 测试的视频帧数，默认64帧（必须能被4整除）。
    num_runs (int): 重复运行的次数，默认10次。
    warmup_runs (int): 预热运行次数，默认3次（不计入统计）。
    """
    # DiT 模型期望的输入维度
    batch = 1  # 使用 batch=1 模拟 guidance_scale=1.0 的情况（不做 CFG）
    channel = 16
    height = 80
    width = 160
    
    # Transformer 的输入帧数，考虑 temporal_compression_ratio
    latent_frames = frames // 4
    
    # 定义运行单次测试的函数
    def run_single_test():
        # --- 准备模型输入 ---
        # 1. 创建主要的输入张量 (hidden_states)
        input_tensor = torch.randn((batch, latent_frames, channel, height, width),
                                   dtype=torch.bfloat16).to('jax')

        # 2. 创建随机的文本嵌入 (encoder_hidden_states)
        embedding = torch.nn.Embedding(10, 4096).to(dtype=torch.bfloat16, device='jax')
        text_guide_ids = torch.arange(10, dtype=torch.int64, device='jax')
        text_guide_embd = embedding(text_guide_ids)
        input_embd = text_guide_embd.unsqueeze(0)  # shape: [1, 10, 4096]

        # 3. 创建时间步 (timestep)
        timestep = torch.full((batch,), 999, dtype=torch.int64, device='jax')

        # 4. 创建旋转位置编码 (image_rotary_emb)
        patch_h, patch_w = height // 2, width // 2
        tokens = latent_frames * patch_h * patch_w // 2
        rotary_emb = torch.randn((tokens, 64), dtype=torch.float32).to('jax')
        rotary_emb = (rotary_emb, rotary_emb)

        # 定义调用 Transformer 模型的函数
        dit_call = lambda: transformer(
            hidden_states=input_tensor,
            encoder_hidden_states=input_embd,
            timestep=timestep,
            image_rotary_emb=rotary_emb
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
    print(f"\n开始正式测试 DiT TPU 性能 (帧数: {frames}, 运行次数: {num_runs})")
    
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

def load_transformer(model_name=MODEL_NAME):
    """
    加载Transformer模型（在普通PyTorch环境中）
    
    Args:
        model_name: 预训练模型名称
        
    Returns:
        transformer: Transformer模型
    """
    print(f"正在加载模型: {model_name}")
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        model_name,
        subfolder="transformer"
    ).to(dtype=torch.bfloat16)
    print("模型加载完成")
    return transformer


def setup_transformer_for_tpu(transformer):
    """
    设置Transformer以在TPU上运行
    
    Args:
        transformer: 已加载的Transformer模型
        
    Returns:
        transformer: 编译后的Transformer模型
        env: torchax环境
        mesh: JAX mesh
    """
    print("\n配置Transformer以使用JAX和Splash Attention...")
    
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
    
    # 创建 torchax 环境
    env = torchax.default_env()
    
    # 配置环境以启用 TPU Splash Attention
    env._mesh = mesh
    env.config.use_tpu_splash_attention = True

    # 注册自定义的 Scaled Dot-Product Attention
    print(f"- 注册 Splash Attention（窗口大小: {WINDOW_SIZE}）...")
    custom_attention = functools.partial(
        scaled_dot_product_attention,
        env=env,
        window_size=WINDOW_SIZE
    )
    
    # 覆盖 PyTorch 的 scaled_dot_product_attention
    op_to_override = torch.nn.functional.scaled_dot_product_attention
    env._ops[op_to_override] = ops_registry.Operator(
        op_to_override,
        custom_attention,
        is_jax_function=False,
        is_user_defined=True,
        needs_env=False,
        is_view_op=False,
    )
    
    # 辅助函数：将模块权重移动到 XLA
    def _move_module_to_xla(module):
        """将模块的权重转换为 JAX Array，但先在 CPU 上操作"""
        with jax.default_device('cpu'):
            state_dict = module.state_dict()
            state_dict = env.to_xla(state_dict)
            module.load_state_dict(state_dict, assign=True)
    
    with env:
        # 对 Transformer 进行处理：先移到 XLA，再分片
        print("- 将Transformer移到XLA并进行分片...")
        _move_module_to_xla(transformer)
        transformer_weights = shard_weights_transformer(mesh, transformer.state_dict(), use_fsdp=USE_FSDP)
        transformer.load_state_dict(transformer_weights, assign=True, strict=False)
        
        # 确保所有权重已分片完成
        torchax.interop.call_jax(jax.block_until_ready, transformer_weights)
        
        # 编译transformer（DiT的核心网络）
        print("- 编译Transformer...")
        transformer = torchax.compile(
            transformer,
            torchax.CompileOptions(
                jax_jit_kwargs={'static_argnames': ('return_dict', )}
            )
        )
    
    print("Transformer配置完成")
    return transformer, env, mesh


def dit(frames=64, num_runs=10):
    # Set JAX config to enable compilation cache
    jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

    """
    执行 DiT 模型在TPU上的性能测试。
    
    参数:
    frames (int): 测试的视频帧数，默认64帧（必须能被4整除）。
    num_runs (int): 重复运行的次数，默认10次。
    """
    print("--- 开始 DiT TPU 性能测试 ---")
    
    # 设置JAX配置
    warnings.filterwarnings('ignore')
    logging.getLogger().setLevel(logging.ERROR)
    
    # 设置随机种子
    import random
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    torch.set_default_dtype(torch.bfloat16)
    
    # 注册PyTree节点
    setup_pytree_registrations()
    
    # 加载Transformer模型
    transformer = load_transformer()
    
    # 设置Transformer在TPU上运行
    transformer, env, mesh = setup_transformer_for_tpu(transformer)
    
    # 在mesh和env上下文中执行测试
    with mesh, env:
        # 执行 DiT 测试
        results = dit_test(transformer, frames=frames, num_runs=num_runs)
    
    # 打印统计结果
    print_results(results, frames)


# --- 脚本执行入口 ---
if __name__ == "__main__":
    # Set JAX config to enable compilation cache
    jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

    warnings.filterwarnings('ignore', message='.*dtype.*int64.*truncated to dtype int32.*')
    logging.getLogger().setLevel(logging.ERROR)
    # 执行 DiT 的TPU性能测试
    dit(num_runs=20)