import os
import time
import warnings
import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
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
    print("JAX配置完成")


def create_mesh():
    """创建JAX设备网格"""
    num_devices = jax.device_count()
    print(f"检测到 {num_devices} 个TPU设备")
    
    # 创建一维mesh，将所有设备用于并行处理
    mesh_devices = mesh_utils.create_device_mesh((num_devices,))
    mesh = Mesh(mesh_devices, ('devices',))
    return mesh


def normalize_video(input: torch.Tensor):
    return (input / 255.0 - 0.5) * 2


def compare_resize(directory_path):
    """
    比较 torchvision 和 OpenCV 的 resize 差异（TPU版本）
    
    参数:
        directory_path: 包含PNG文件的目录路径
    """
    # 获取目录下所有PNG文件
    png_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.lower().endswith('.png')]

    if not png_files:
        print("目录中没有找到PNG文件")
        return

    size = (1080, 640)
    
    # 使用PIL加载图片（避免torchvision.io.decode_image在torchax环境中的兼容性问题）
    print("使用PIL在CPU上加载图片...")
    frames_np = [np.array(Image.open(frame)) for frame in png_files]
    frames_np_to_torch = [torch.from_numpy(frame) for frame in frames_np]
    frames_np_to_torch = torch.stack(frames_np_to_torch).permute(0, 3, 1, 2)
    
    # 移动到TPU设备
    print("将数据移动到TPU设备...")
    frames_np_to_torch = frames_np_to_torch.to('jax')
    
    print('-------------跳过torchvision.io.decode_image测试（与torchax不兼容）-------------')
    print('直接使用PIL加载的图片进行后续测试...\n')
    
    # 使用torch api resize
    resize = torchvision.transforms.Resize(size, interpolation=InterpolationMode.BILINEAR, antialias=False)

    # 使用opencv resize（在CPU上）
    opencv_interpolation = cv2.INTER_LINEAR
    start_time = time.time()
    frames_np_resized = [cv2.resize(frame, (size[1], size[0]), interpolation=opencv_interpolation) for frame in frames_np]
    print(f'Opencv resize: {len(frames_np)} frames time: {time.time() - start_time:.4f} 秒')
    frames_np_resized_to_torch = [torch.from_numpy(frame) for frame in frames_np_resized]
    frames_np_resized_to_torch = torch.stack(frames_np_resized_to_torch).permute(0, 3, 1, 2)
    frames_np_resized_to_torch = normalize_video(frames_np_resized_to_torch)

    # 使用torch api resize（在TPU上）,输入为pil解码的结果
    print(f'\n在TPU上执行torch resize...')
    start_time = time.time()
    frames_np_to_torch_resized = resize(frames_np_to_torch)
    # 等待TPU计算完成
    torchax.interop.call_jax(jax.block_until_ready, frames_np_to_torch_resized)
    resize_time = time.time() - start_time
    print(f'TPU resize: {len(frames_np)} frames time: {resize_time:.4f} 秒')
    frames_np_to_torch_resized = normalize_video(frames_np_to_torch_resized).cpu()

    diff = torch.abs(frames_np_resized_to_torch - frames_np_to_torch_resized).to(torch.float)
    print('-------------比较解码使用pil，resize使用torch与opencv之间的差异-------------')
    print(f"Maximum difference across all images: {diff.max().item()}")
    print(f"Minimum difference across all images: {diff.min().item()}")
    print(f"Median difference: {diff.mean().item()}")
    non_zero_pixels = torch.sum(diff > 0).item()
    print(f"Number of pixels with non-zero difference: {non_zero_pixels}")
    zero_pixels = torch.sum(diff == 0).item()
    print(f"Number of pixels with zero difference: {zero_pixels}")
    lt_mean_pixels = torch.sum(diff > diff.mean().item()).item()
    print(f"Number of pixels with lt_mean difference: {lt_mean_pixels}")
    
    return resize_time


def main():
    """主函数 - 设置TPU环境并运行测试"""
    print("=" * 60)
    print("TPU版本resize测试 - 运行5次以观察tracing和缓存效果")
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
    path = './test_set/video'
    
    print("\n目的：比较torchvision的resize和opencv的resize之间的精度差异")
    print("所有计算在TPU上执行")
    print("第1次运行包含代码tracing时间，后续运行使用缓存\n")
    
    # 记录每次运行的resize时间
    resize_times = []
    
    # 在torchax环境中执行 - 运行5次
    with env, mesh:
        for iteration in range(5):
            print(f"\n{'='*60}")
            print(f"第 {iteration + 1} 次运行")
            print(f"{'='*60}\n")
            
            resize_time = compare_resize(path)
            resize_times.append(resize_time)
            
            if iteration == 0:
                print(f"\n(第1次运行包含代码tracing和编译缓存扫描时间)")
            else:
                print(f"\n(使用缓存，无需重新tracing)")
    
    # 打印总结
    print("\n" + "=" * 60)
    print("性能总结 - TPU Resize时间")
    print("=" * 60)
    print(f"\n各次运行的resize时间：")
    for i, t in enumerate(resize_times, 1):
        print(f"  第 {i} 次: {t:.4f} 秒")
    
    if len(resize_times) > 1:
        print(f"\n第1次运行（含tracing）: {resize_times[0]:.4f} 秒")
        avg_cached = sum(resize_times[1:]) / len(resize_times[1:])
        print(f"后续运行平均时间: {avg_cached:.4f} 秒")
        speedup = resize_times[0] / avg_cached
        print(f"加速比: {speedup:.2f}x")
        
        print(f"\n说明：")
        print(f"- 第1次运行包含PyTorch代码tracing和编译缓存扫描")
        print(f"- 后续运行直接使用缓存的编译结果，速度更快")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()