# 导入必要的库
import time  # 用于测量时间
import torch  # PyTorch 库，用于深度学习和张量操作
from tqdm import tqdm  # 用于显示进度条
import sys
import os

# 添加 HunyuanVideo-1.5-TPU 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..', 'HunyuanVideo-1.5-TPU'))

from hyvideo.models.transformers.hunyuanvideo_1_5_transformer import HunyuanVideo_1_5_DiffusionTransformer

# --- 全局配置 ---
# 设置默认的计算设备为第一个 CUDA 设备 (GPU)
DEVICE = 'cuda:0'

# --- 性能评测工具函数 ---

def record_time(call_method):
    """
    记录一个函数调用的执行时间。
    为了精确测量 GPU 操作的时间，此函数在操作前后都进行了 torch.cuda.synchronize() 调用。
    这确保了所有在 CUDA 流中排队的任务都已完成，从而得到准确的耗时。

    参数:
    call_method (function): 需要被测量时间的函数。

    返回:
    tuple: 一个元组，包含被调用函数的输出和以毫秒为单位的执行时间。
    """
    # 确保在该时间点之前的所有 CUDA 内核都已经完成
    torch.cuda.synchronize()
    start = time.time()  # 记录开始时间

    # 执行需要被测量的函数
    output = call_method()

    # 再次同步，确保 call_method 中的所有 CUDA 操作都已完成
    torch.cuda.synchronize()
    end = time.time()  # 记录结束时间

    # 返回函数输出和执行时间（从秒转换为毫秒）
    return output, (end - start) * 1000  # s -> ms

def record_peak_memory(call_method):
    """
    记录一个函数调用期间的 GPU 峰值显存使用量和执行时间。

    参数:
    call_method (function): 需要被测量的函数。

    返回:
    tuple: 一个元组，包含被调用函数的输出、以 MB 为单位的峰值显存和以毫秒为单位的执行时间。
    """
    # 在测量前，清空 CUDA 缓存，以减少之前操作的干扰
    torch.cuda.empty_cache()
    # 重置 PyTorch 的峰值显存统计数据，确保从零开始记录
    torch.cuda.reset_peak_memory_stats()

    # 调用 record_time 来执行函数并获取其输出和执行时间
    output, time_cost = record_time(call_method)

    # 获取 PyTorch 记录的"所有已分配字节"的峰值
    peak_memory = torch.cuda.memory_stats()["allocated_bytes.all.peak"]

    # --- 修正非 PyTorch 分配的显存 ---
    # 有时，除了 PyTorch，其他库（如 CUDA 驱动本身）也可能分配显存。
    # 为了得到更全面的峰值显存，我们尝试估算这部分。
    torch_allocated_bytes = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    total_allocated_bytes = torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]
    non_torch_allocations = total_allocated_bytes - torch_allocated_bytes
    if non_torch_allocations > 0:
        peak_memory += non_torch_allocations

    # 将峰值显存从字节（bytes）转换为兆字节（MB）
    peak_memory_mb = peak_memory / (1024 ** 2)

    # 再次清空缓存
    torch.cuda.empty_cache()
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
    
    peak_memories = [r['peak_memory_mb'] for r in results]
    times = [r['time'] for r in results]
    
    print(f"\n=== DiT 测试结果 (帧数: {frames}) ===")
    print(f"运行次数: {len(results)}")
    print(f"\n峰值显存 (MB):")
    print(f"  平均值: {sum(peak_memories)/len(peak_memories):.2f}")
    print(f"  最小值: {min(peak_memories):.2f}")
    print(f"  最大值: {max(peak_memories):.2f}")
    print(f"\n执行时间 (ms):")
    print(f"  平均值: {sum(times)/len(times):.2f}")
    print(f"  最小值: {min(times):.2f}")
    print(f"  最大值: {max(times):.2f}")


# --- DiT 模型性能测试核心函数 ---

def dit_test(transformer, frames=129, resolution='720p', num_runs=10, enable_cfg=False):
    """
    测试 DiT (Diffusion Transformer) 模型的性能。
    对指定帧数重复运行多次以获取稳定的性能数据。

    参数:
    transformer (torch.nn.Module): 已加载的 Transformer 模型。
    frames (int): 测试的视频帧数，默认129帧。
    resolution (str): 视频分辨率 ('480p' 或 '720p')。
    num_runs (int): 重复运行的次数，默认10次。
    enable_cfg (bool): 是否启用 CFG（Classifier-Free Guidance），默认False。
    """
    # HunyuanVideo-1.5 的输入维度
    # 当 CFG 启用时，batch 会翻倍（unconditional + conditional）
    batch = 2 if enable_cfg else 1
    
    # 重要：HunyuanVideo-1.5 的 VAE 输出 32 通道 latents（不是标准的 4 通道！）
    channel = 32  # VAE latent channels (32 channels, not 4!)
    
    # 根据分辨率设置高度和宽度
    # VAE 空间压缩比 = 16 倍 (ffactor_spatial = 16)
    # VAE 时间压缩比 = 4 倍 (ffactor_temporal = 4)
    # 720p: 1280x720 -> latent: 80x45
    # 480p: 848x480 -> latent: 53x30
    if resolution == '720p':
        height = 45   # 720 / 16 = 45
        width = 80    # 1280 / 16 = 80
    else:  # 480p
        height = 30   # 480 / 16 = 30
        width = 53    # 848 / 16 = 53
    
    # Transformer 的输入帧数
    # VAE 时间压缩公式: latent_frames = (frames - 1) // 4 + 1
    latent_frames = (frames - 1) // 4 + 1
    
    # 由于模型使用 concat_condition=True，需要准备额外的条件通道
    # 输入格式: latents (32ch) + cond_latents (33ch, 其中 32ch condition + 1ch mask) = 65 channels
    # 注意：mask 已经在 _prepare_cond_latents 中与 condition 合并
    total_input_channels = channel + (channel + 1)  # 32 + 33 = 65
    
    # 计算 patch 后的 token 数量
    # 注意：720p_i2v 使用 patch_size=[1, 1, 1]，不是 [1,2,2]
    patch_t, patch_h, patch_w = 1, 1, 1
    token_t = latent_frames // patch_t
    token_h = height // patch_h
    token_w = width // patch_w
    total_tokens = token_t * token_h * token_w
    
    results = []
    cfg_status = "启用" if enable_cfg else "禁用 (guidance_scale=1.0)"
    print(f"开始测试 DiT 性能 (帧数: {frames}, 分辨率: {resolution}, 运行次数: {num_runs})")
    print(f"CFG 状态: {cfg_status}")
    print(f"Latent shape (单个): [{batch}, {channel}, {latent_frames}, {height}, {width}]")
    print(f"Input shape (拼接后): [{batch}, {total_input_channels}, {latent_frames}, {height}, {width}]")
    print(f"Token shape (after patch [{patch_t},{patch_h},{patch_w}]): {token_t} x {token_h} x {token_w} = {total_tokens} tokens")
    
    for run in tqdm(range(num_runs), desc="Testing DiT"):
        try:
            # --- 准备模型输入 ---
            # 1. 创建主要的 latents
            # 格式: [batch, channel, frames, height, width]
            latents = torch.randn((batch, channel, latent_frames, height, width),
                                 dtype=torch.bfloat16).to(DEVICE)
            
            # 2. 创建条件 latents (用于 multitask，t2v 时为零，32 通道)
            cond_latents_only = torch.zeros((batch, channel, latent_frames, height, width),
                                            dtype=torch.bfloat16).to(DEVICE)
            
            # 3. 创建 mask (用于 multitask，t2v 时全为零，1 通道)
            mask = torch.zeros((batch, 1, latent_frames, height, width),
                              dtype=torch.bfloat16).to(DEVICE)
            
            # 4. 先合并 condition 和 mask 成 cond_latents (33 通道)
            cond_latents = torch.cat([cond_latents_only, mask], dim=1)  # 32 + 1 = 33
            
            # 5. 再拼接 latents 和 cond_latents 成完整输入 (65 通道)
            hidden_states = torch.cat([latents, cond_latents], dim=1)  # 32 + 33 = 65

            # 2. 创建时间步 (timestep)
            timestep = torch.tensor([999], dtype=torch.long, device=DEVICE)

            # 3. 创建文本嵌入 (text_states)
            # 720p_i2v 使用 text_states_dim = 3584
            text_seq_len = 1000  # 文本序列长度（从 config 和实际运行看到的）
            text_states = torch.randn((batch, text_seq_len, 3584),
                                     dtype=torch.bfloat16).to(DEVICE)
            
            # 4. 创建第二个文本嵌入 (text_states_2)
            # 720p_i2v 配置下 text_pool_type=None, 所以 text_states_2 应该是 None
            text_states_2 = None

            # 5. 创建注意力掩码 (encoder_attention_mask)
            encoder_attention_mask = torch.ones((batch, text_seq_len),
                                               dtype=torch.int64, device=DEVICE)
            
            # 6. 创建 extra_kwargs（用于 glyph_byT5_v2）
            # 由于模型有 glyph_byT5_v2=True，需要提供这些参数
            # 对于性能测试，我们使用零张量（模拟无 glyph 文本的情况）
            byt5_max_length = 256  # 从 config 获取的默认值
            extra_kwargs = {
                "byt5_text_states": torch.zeros((batch, byt5_max_length, 1472),
                                               dtype=torch.bfloat16, device=DEVICE),
                "byt5_text_mask": torch.zeros((batch, byt5_max_length),
                                             dtype=torch.int64, device=DEVICE)
            }

            # 定义调用 Transformer 模型的函数
            # 使用 autocast 确保 dtype 正确处理（与 pipeline 保持一致）
            def dit_call():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    return transformer(
                        hidden_states=hidden_states,
                        timestep=timestep,
                        text_states=text_states,
                        text_states_2=text_states_2,
                        encoder_attention_mask=encoder_attention_mask,
                        freqs_cos=None,  # 让模型自己生成
                        freqs_sin=None,
                        return_dict=False,
                        extra_kwargs=extra_kwargs,
                    )

            # 记录峰值显存和执行时间
            output, peak_memory_mb, time_cost = record_peak_memory(dit_call)
            del output  # 释放显存

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

@torch.inference_mode()
def dit(frames=121, resolution='720p', num_runs=10, model_path=None, enable_cfg=False):
    """
    执行 DiT 模型的性能测试。
    
    参数:
    frames (int): 测试的视频帧数，默认121帧（必须能被4整除）。
    resolution (str): 视频分辨率 ('480p' 或 '720p')。
    num_runs (int): 重复运行的次数，默认10次。
    model_path (str): 模型路径（可选，如果提供则从该路径加载权重）。
    enable_cfg (bool): 是否启用 CFG（默认False，对应 guidance_scale=1.0）。
    """
    print("--- 开始 DiT 性能测试 ---")
    
    # 使用 from_pretrained 加载模型，这会自动从 config.json 加载正确的配置
    if model_path is not None:
        # 构建正确的模型目录路径
        if resolution == '720p':
            model_dir = os.path.join(model_path, 'transformer', '720p_i2v')
        else:  # 480p
            model_dir = os.path.join(model_path, 'transformer', '480p_i2v')
        
        print(f"从 {model_dir} 加载模型（使用 from_pretrained）...")
        transformer = HunyuanVideo_1_5_DiffusionTransformer.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(DEVICE)
    else:
        # 如果没有提供路径，使用默认配置创建模型（仅用于测试形状）
        print("警告: 未提供模型路径，使用随机权重初始化模型")
        if resolution == '720p':
            transformer = HunyuanVideo_1_5_DiffusionTransformer(
                patch_size=[1, 1, 1],
                in_channels=32,
                out_channels=32,
                hidden_size=2048,
                heads_num=16,
                mlp_width_ratio=4.0,
                mlp_act_type="gelu_tanh",
                mm_double_blocks_depth=54,
                mm_single_blocks_depth=0,
                rope_dim_list=[16, 56, 56],
                qkv_bias=True,
                qk_norm=True,
                qk_norm_type="rms",
                guidance_embed=False,
                text_projection="single_refiner",
                use_attention_mask=True,
                text_states_dim=3584,
                text_states_dim_2=None,
                attn_mode="flash",
                concat_condition=True,
            )
        else:  # 480p
            transformer = HunyuanVideo_1_5_DiffusionTransformer(
                patch_size=[1, 1, 1],
                in_channels=32,
                out_channels=32,
                hidden_size=2048,
                heads_num=16,
                mlp_width_ratio=4.0,
                mlp_act_type="gelu_tanh",
                mm_double_blocks_depth=54,
                mm_single_blocks_depth=0,
                rope_dim_list=[16, 56, 56],
                qkv_bias=True,
                qk_norm=True,
                qk_norm_type="rms",
                guidance_embed=False,
                text_projection="single_refiner",
                use_attention_mask=True,
                text_states_dim=3584,
                text_states_dim_2=None,
                attn_mode="flash",
                concat_condition=True,
            )
        transformer = transformer.to(dtype=torch.bfloat16, device=DEVICE)
    
    transformer.eval()
    
    print(f"模型参数量: {sum(p.numel() for p in transformer.parameters()) / 1e9:.2f}B")
    
    # 执行 DiT 测试
    results = dit_test(transformer, frames=frames, resolution=resolution, num_runs=num_runs, enable_cfg=enable_cfg)
    
    # 打印统计结果
    print_results(results, frames)


# --- 脚本执行入口 ---
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='HunyuanVideo-1.5 DiT Performance Test')
    parser.add_argument('--frames', type=int, default=121,
                       help='Number of frames to test (default: 121)')
    parser.add_argument('--resolution', type=str, default='720p',
                       choices=['480p', '720p'],
                       help='Video resolution (default: 720p)')
    parser.add_argument('--num_runs', type=int, default=3,
                       help='Number of test runs (default: 3)')
    parser.add_argument('--model_path', type=str, default='/dev/shm/HunyuanVideo-1.5/ckpts',
                       help='Path to model checkpoints directory (default: /dev/shm/HunyuanVideo-1.5/ckpts)')
    parser.add_argument('--enable_cfg', action='store_true',
                       help='Enable Classifier-Free Guidance (default: False, equivalent to guidance_scale=1.0)')
    
    args = parser.parse_args()
    
    # 不需要调整帧数，VAE 的公式是 (frames-1)//4+1，可以处理任意帧数
    # if args.frames % 4 != 0:
    #     print(f"警告: 帧数 {args.frames} 不能被4整除，将调整为 {(args.frames // 4) * 4}")
    #     args.frames = (args.frames // 4) * 4
    
    # 执行 DiT 的性能测试
    dit(
        frames=args.frames,
        resolution=args.resolution,
        num_runs=args.num_runs,
        model_path=args.model_path,
        enable_cfg=args.enable_cfg
    )