# 导入必要的库
import time  # 用于测量时间
import torch  # PyTorch 库，用于深度学习和张量操作
from diffusers import CogVideoXTransformer3DModel  # 从 diffusers 库导入 Transformer 模型类
from tqdm import tqdm  # 用于显示进度条

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

def dit_test(transformer, frames=64, num_runs=10):
    """
    测试 DiT (Diffusion Transformer) 模型的性能。
    对指定帧数重复运行多次以获取稳定的性能数据。

    参数:
    transformer (torch.nn.Module): 已加载的 Transformer 模型。
    frames (int): 测试的视频帧数，默认64帧（必须能被4整除）。
    num_runs (int): 重复运行的次数，默认10次。
    """
    # DiT 模型期望的输入维度
    batch = 1  # 使用 batch=1 模拟 guidance_scale=1.0 的情况（不做 CFG）
    channel = 16
    height = 80
    width = 160
    
    # Transformer 的输入帧数，考虑 temporal_compression_ratio
    latent_frames = frames // 4
    
    results = []
    print(f"开始测试 DiT 性能 (帧数: {frames}, 运行次数: {num_runs})")
    
    for run in tqdm(range(num_runs), desc="Testing DiT"):
        try:
            # --- 准备模型输入 ---
            # 1. 创建主要的输入张量 (hidden_states)
            input_tensor = torch.randn((batch, latent_frames, channel, height, width),
                                       dtype=torch.bfloat16).to(DEVICE)

            # 2. 创建随机的文本嵌入 (encoder_hidden_states)
            embedding = torch.nn.Embedding(10, 4096).to(dtype=torch.bfloat16, device=DEVICE)
            text_guide_ids = torch.arange(10, dtype=torch.int64, device=DEVICE)
            text_guide_embd = embedding(text_guide_ids)
            input_embd = text_guide_embd.unsqueeze(0)  # shape: [1, 10, 4096]

            # 3. 创建时间步 (timestep)
            timestep = torch.full((batch,), 999, dtype=torch.int64, device=DEVICE)

            # 4. 创建旋转位置编码 (image_rotary_emb)
            patch_h, patch_w = height // 2, width // 2
            tokens = latent_frames * patch_h * patch_w // 2
            rotary_emb = torch.randn((tokens, 64), dtype=torch.float32).to(DEVICE)
            rotary_emb = (rotary_emb, rotary_emb)

            # 定义调用 Transformer 模型的函数
            dit_call = lambda: transformer(
                hidden_states=input_tensor,
                encoder_hidden_states=input_embd,
                timestep=timestep,
                image_rotary_emb=rotary_emb
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
            break

    return results


# --- 主测试流程 ---

@torch.inference_mode()
def dit(frames=64, num_runs=10):
    """
    执行 DiT 模型的性能测试。
    
    参数:
    frames (int): 测试的视频帧数，默认64帧（必须能被4整除）。
    num_runs (int): 重复运行的次数，默认10次。
    """
    print("--- 开始 DiT 性能测试 ---")
    # 从配置文件初始化 Transformer 模型
    transformer = CogVideoXTransformer3DModel.from_config('/home/chrisya/.cache/huggingface/hub/models--THUDM--CogVideoX1.5-5B/snapshots/fdc5267c90b5c06492985b966e43aae984e189e0/transformer/').to(dtype=torch.bfloat16, device=DEVICE)
    
    # 执行 DiT 测试
    results = dit_test(transformer, frames=frames, num_runs=num_runs)
    
    # 打印统计结果
    print_results(results, frames)


# --- 脚本执行入口 ---
if __name__ == "__main__":
    # 执行 DiT 的性能测试
    dit()