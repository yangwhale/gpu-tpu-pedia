import os
import time
import warnings
from typing import Tuple

import torch
import torchvision
from PIL import Image
import numpy as np
from torchvision.transforms import InterpolationMode

# JAX 和 torchax 相关导入
import jax
import jax.numpy as jnp
import torchax
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
from jax.experimental import mesh_utils

# 过滤警告
warnings.filterwarnings('ignore', message='.*Explicitly requested dtype int64.*')
warnings.filterwarnings('ignore', message='.*NumPy array is not writable.*')


def setup_jax_config():
    """配置JAX环境参数"""
    # 启用编译缓存以提高性能
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    
    # 可选：启用64位支持以避免警告（但会增加内存使用）
    # jax.config.update("jax_enable_x64", True)
    
    print("JAX配置完成")


def create_mesh():
    """创建JAX设备网格"""
    num_devices = jax.device_count()
    print(f"检测到 {num_devices} 个TPU设备")
    
    # 创建一维mesh，将所有设备用于并行处理
    mesh_devices = mesh_utils.create_device_mesh((num_devices,))
    mesh = Mesh(mesh_devices, ('devices',))
    return mesh


# opencv的resize没有antialias，等价与torch vison的antialias=False
def resize_video(video_frames: torch.Tensor,  # batch, frame,channel,height,width 或者 frame,channel,height,width
                 resize_height: int,
                 resize_width: int,
                 interpolation: InterpolationMode = InterpolationMode.BILINEAR,
                 antialias: bool = False) -> torch.Tensor:
    '''
        torchvision resize 不支持5D的tensor，需要合并一下维度
    '''
    resize = torchvision.transforms.Resize((resize_height, resize_width), interpolation=interpolation, antialias=antialias)
    origin_shape = video_frames.shape
    frames, channels, origi_height, origi_width = origin_shape[-4:]
    if len(origin_shape) == 5:
        video_frames = video_frames.reshape(-1, channels, origi_height, origi_width)

    frames_mem = video_frames.numel() * video_frames.element_size() / (1024**2)  # 计算batch的大小，过大的话resize可能会oom，拆小
    frames_mem = int(frames_mem)
    max_mem = 200  # TODO 可以放入配置中
    if frames_mem > max_mem:
        chunks = (frames_mem + max_mem - 1) // max_mem
        video_chunks = video_frames.chunk(chunks, dim=0)
        result = []
        for chunk in video_chunks:
            result.append(resize(chunk))
        result = torch.cat(result)
    else:
        result = resize(video_frames)

    if len(origin_shape) == 5:
        result = result.reshape(origin_shape[0], frames, channels, resize_height, resize_width)
    return result


def resize_frames(video_frames: torch.Tensor,
                  mask_frames: torch.Tensor,
                  image_size: int = 640,
                  max_length: int = 1280) -> Tuple[torch.Tensor, torch.Tensor]:
    # 入参已经校验video_frames和mask_frames是否匹配，这里就不再校验
    original_height, original_width = video_frames.shape[-2:]

    if min(original_width, original_height) > image_size:
        # First calculate the scale to resize the short side to image_size
        if original_width < original_height:
            scale = image_size / original_width
            w_resize, h_resize = image_size, int(original_height * scale)
        else:
            scale = image_size / original_height
            w_resize, h_resize = int(original_width * scale), image_size

        # Then check if the long side exceeds max_length
        if max(w_resize, h_resize) > max_length:
            if w_resize > h_resize:
                # Width is the longer side
                new_scale = max_length / w_resize
                w_resize, h_resize = max_length, int(h_resize * new_scale)
            else:
                # Height is the longer side
                new_scale = max_length / h_resize
                w_resize, h_resize = int(w_resize * new_scale), max_length
    else:
        # 短边小于640
        w_resize = original_width
        h_resize = original_height

    # 确保输入的长宽能够被对应的数整除，推理过程中算子对形状有要求(ring attention要求height被64整除)
    w_resize, h_resize = w_resize // 16 * 16, h_resize // 16 * 16

    if w_resize == original_width and h_resize == original_height:
        return video_frames, mask_frames
    else:
        return (resize_video(video_frames, h_resize, w_resize),
                resize_video(mask_frames, h_resize, w_resize))


def crop_based_on_mask_torch(video_frames, mask_frames, padding_ratio=0.5, min_size=(512, 512), critical_ratio=2.0):
    """
    Crop video frames and mask frames based on the mask's valid region using PyTorch.

    Args:
        video_frames: torch.Tensor - video frame tensor (N, C, H, W)
        mask_frames: torch.Tensor - mask frame tensor (N, C, H, W) C = 1
        padding_ratio: float - padding ratio to add around valid region (0-1)
        min_size: tuple (w, h) - minimum output size, if None no restriction
        critical_ratio: float - maximum allowed aspect ratio (height/width)

    Returns:
        tuple: (coordinates (x1, y1, x2, y2), cropped_video_frames, cropped_mask_frames)
    """
    # Combine all masks to find the bounding box (max projection across frames and channels)
    combined_mask = torch.any(mask_frames > 0.5, dim=0)  # Combine across frames
    combined_mask = torch.any(combined_mask, dim=0)  # Combine across channels

    # Find non-zero pixels
    rows = torch.any(combined_mask, dim=1)
    cols = torch.any(combined_mask, dim=0)

    if not torch.any(rows) or not torch.any(cols):
        # No valid region, return original
        return None, video_frames, mask_frames

    # Calculate bounding box
    nonzero_rows = torch.where(rows)[0]
    nonzero_cols = torch.where(cols)[0]
    rmin, rmax = nonzero_rows[[0, -1]].tolist()  # 转成int标量
    cmin, cmax = nonzero_cols[[0, -1]].tolist()

    # Add padding
    height = rmax - rmin + 1
    width = cmax - cmin + 1

    pad_h = int(height * padding_ratio)
    pad_w = int(width * padding_ratio)

    # Calculate final crop coordinates, ensuring they don't exceed image boundaries
    h, w = combined_mask.shape
    y1 = max(0, rmin - pad_h)
    y2 = min(h, rmax + pad_h + 1)
    x1 = max(0, cmin - pad_w)
    x2 = min(w, cmax + pad_w + 1)

    # Ensure minimum size requirements
    if min_size is not None:
        min_w, min_h = min_size
        crop_w = x2 - x1
        crop_h = y2 - y1

        if crop_w < min_w:
            diff = min_w - crop_w
            x1 -= diff // 2
            x2 += diff - (diff // 2)
            if x1 < 0:
                x2 = min(w, x2 - x1)
                x1 = 0
            elif x2 > w:
                x1 = max(0, x1 - (x2 - w))
                x2 = w

        if crop_h < min_h:
            diff = min_h - crop_h
            y1 -= diff // 2
            y2 += diff - (diff // 2)
            if y1 < 0:
                y2 = min(y2 - y1, h)
                y1 = 0
            elif y2 > h:
                y1 = max(y1 - (y2 - h), 0)
                y2 = h

    # Ensure aspect ratio is within critical bounds
    crop_w, crop_h = x2 - x1, y2 - y1
    aspect_ratio = crop_h / crop_w

    # Adjust width if height is too large
    if aspect_ratio > critical_ratio:
        target_w = int(crop_h / critical_ratio)
        delta = target_w - crop_w

        # Symmetrically expand width
        x1 -= delta // 2
        x2 += delta - (delta // 2)

        # Handle out-of-bounds
        if x1 < 0:
            x2 = min(w, x2 - x1)
            x1 = 0
        elif x2 > w:
            x1 = max(0, x1 - (x2 - w))
            x2 = w

    # Adjust height if width is too large
    elif aspect_ratio < (1 / critical_ratio):
        target_h = int(crop_w / critical_ratio)
        delta = target_h - crop_h

        # Symmetrically expand height
        y1 -= delta // 2
        y2 += delta - (delta // 2)

        # Handle out-of-bounds
        if y1 < 0:
            y2 = min(y2 - y1, h)
            y1 = 0
        elif y2 > h:
            y1 = max(y1 - (y2 - h), 0)
            y2 = h

    # Ensure even dimensions
    if (y2 - y1) % 2 != 0:
        if y2 < h:
            y2 += 1
        elif y1 > 0:
            y1 -= 1
    if (x2 - x1) % 2 != 0:
        if x2 < w:
            x2 += 1
        elif x1 > 0:
            x1 -= 1

    # Apply the same crop to all frames
    cropped_video_frames = video_frames[:, :, y1:y2, x1:x2]
    cropped_mask_frames = mask_frames[:, :, y1:y2, x1:x2]

    return (x1, y1, x2, y2), cropped_video_frames, cropped_mask_frames


# 工具方法
def np_to_torch(frames_np):
    return torch.stack([torch.from_numpy(frame) for frame in frames_np])


def normalize_video(input: torch.Tensor):
    return (input / 255.0 - 0.5) * 2


def denormalize_video(input: torch.Tensor):
    return ((input * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8)


def preprocess(video_path, mask_path):
    """
    预处理函数 - 在TPU上执行
    """
    video_files = [os.path.join(video_path, f) for f in os.listdir(video_path) if f.lower().endswith('.png')]
    mask_files = [os.path.join(mask_path, f) for f in os.listdir(mask_path) if f.lower().endswith('.png')]

    if not video_files or not mask_files or len(video_files) != len(mask_files):
        print("frame 文件数据不正确")
        return None, None

    video_frames_np = [np.array(Image.open(frame)) for frame in video_files]
    mask_frames_np = [np.array(Image.open(frame)) for frame in mask_files]
    mask_frames_np = [(mask > 0.5).astype(np.uint8) for mask in mask_frames_np]

    # 将数据转换为torch tensor并移动到JAX设备
    video_frames_np_to_torch = np_to_torch(video_frames_np).permute(0, 3, 1, 2).to('jax')
    mask_frames_np_to_torch = np_to_torch(mask_frames_np).permute(0, 3, 1, 2).to('jax')

    print("开始在TPU上执行crop操作...")
    start_time = time.time()
    coordinates_torch, cropped_video_torch, cropped_mask_torch = crop_based_on_mask_torch(
        video_frames_np_to_torch, mask_frames_np_to_torch, 0.5, (512, 512))
    # 等待TPU计算完成
    torchax.interop.call_jax(jax.block_until_ready, cropped_video_torch)
    print(f'TPU crop time: {time.time() - start_time:.4f} 秒')

    print("开始在TPU上执行resize操作...")
    start_time = time.time()
    video_resized_torch, mask_resized_torch = resize_frames(cropped_video_torch, cropped_mask_torch)
    mask_resized_torch = torch.where(mask_resized_torch > 0.5, 1, 0).to(dtype=torch.uint8)
    # 等待TPU计算完成
    torchax.interop.call_jax(jax.block_until_ready, video_resized_torch)
    print(f'TPU resize time: {time.time() - start_time:.4f} 秒')

    return cropped_video_torch, cropped_mask_torch


def postprocess(input_video_torch, input_mask_torch):
    """
    后处理函数 - 在TPU上执行高斯模糊
    """
    # 确保数据在JAX设备上
    input_video_torch = input_video_torch.to('jax')
    input_mask_torch = input_mask_torch.to('jax')

    import torchvision.transforms.functional as TF
    kernel_size = int(3 * 2 + 1)
    
    print("开始在TPU上执行gaussian_blur操作...")
    start_time = time.time()
    blur_mask_torch = TF.gaussian_blur(input_mask_torch * 255, kernel_size=[kernel_size, kernel_size], sigma=[1.5, 1.5]) / 255  # [N,C,H,W]
    # 等待TPU计算完成
    torchax.interop.call_jax(jax.block_until_ready, blur_mask_torch)
    print(f'TPU gaussian_blur time: {time.time() - start_time:.4f} 秒')
    
    return blur_mask_torch


def main():
    """主函数 - 设置TPU环境并运行测试"""
    print("=" * 60)
    print("TPU版本图像处理测试 - 运行5次以观察tracing和缓存效果")
    print("=" * 60)
    
    # 设置JAX配置
    setup_jax_config()
    
    # 使用float32以获得更好的精度（与GPU版本保持一致）
    # 注意：使用bfloat16可以获得更好的TPU性能，但精度会降低
    torch.set_default_dtype(torch.float32)
    
    # 创建mesh
    mesh = create_mesh()
    
    # 创建torchax环境
    env = torchax.default_env()
    env._mesh = mesh
    
    # 设置测试路径
    video_path = './test_set/video'
    mask_path = './test_set/mask'
    
    print("\n目的：")
    print("1. 测试torch下常规的tensor数学和逻辑计算（crop_based_on_mask_torch方法中体现）")
    print("2. 测试TorchVision的图像处理算法（resize和gaussian_blur方法中体现）")
    print("3. 所有计算在TPU上执行")
    print("4. 第1次运行包含代码tracing时间，后续运行使用缓存\n")
    
    # 记录每次运行的时间
    all_times = []
    
    # 在torchax环境中执行 - 运行5次
    with env, mesh:
        for iteration in range(5):
            print(f"\n{'='*60}")
            print(f"第 {iteration + 1} 次运行")
            print(f"{'='*60}")
            
            iteration_start = time.time()
            
            # 预处理 - crop和resize
            print("\n--- 预处理阶段 ---")
            result = preprocess(video_path, mask_path)
            if result is None or result[0] is None:
                print("预处理失败，请检查输入路径")
                return
            
            cropped_video_torch, cropped_mask_torch = result
            
            # 后处理 - gaussian blur
            print("\n--- 后处理阶段 ---")
            blur_result = postprocess(cropped_video_torch, cropped_mask_torch)
            
            iteration_time = time.time() - iteration_start
            all_times.append(iteration_time)
            
            print(f"\n第 {iteration + 1} 次运行总耗时: {iteration_time:.4f} 秒")
            if iteration == 0:
                print("(包含代码tracing和编译缓存扫描时间)")
            else:
                print("(使用缓存，无需重新tracing)")
    
    # 打印总结
    print("\n" + "=" * 60)
    print("性能总结")
    print("=" * 60)
    print(f"\n各次运行时间：")
    for i, t in enumerate(all_times, 1):
        print(f"  第 {i} 次: {t:.4f} 秒")
    
    if len(all_times) > 1:
        print(f"\n第1次运行（含tracing）: {all_times[0]:.4f} 秒")
        avg_cached = sum(all_times[1:]) / len(all_times[1:])
        print(f"后续运行平均时间: {avg_cached:.4f} 秒")
        speedup = all_times[0] / avg_cached
        print(f"加速比: {speedup:.2f}x")
        
        print(f"\n说明：")
        print(f"- 第1次运行包含PyTorch代码tracing和编译缓存扫描")
        print(f"- 后续运行直接使用缓存的编译结果，速度更快")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()