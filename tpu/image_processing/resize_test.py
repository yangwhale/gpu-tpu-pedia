import os
import time
import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import InterpolationMode


def normalize_video(input: torch.Tensor):
    return (input / 255.0 - 0.5) * 2


def compare_resize(directory_path):
    """
    比较 torchvision 和 OpenCV 的 resize 差异
    
    参数:
        directory_path: 包含PNG文件的目录路径
        output_size: 目标resize尺寸 (height, width)
    """
    # 获取目录下所有PNG文件
    png_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.lower().endswith('.png')]

    if not png_files:
        print("目录中没有找到PNG文件")
        return
  

    size = (1080,640)
    # 使用torch的解码api读图片
    frames_torch = [torchvision.io.decode_image(frame) for frame in png_files]  # torch 读取出来的形状是channel,height,width
    frames_torch = torch.stack(frames_torch).cuda()
    
    
    frames_np = [np.array(Image.open(frame)) for frame in png_files]
    #frames_np = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_np]  #opencv的cv2.imdecode和imencode方法要求图像的通道排布是BGR，因此在使用opencv decode出来的结果，或者要写回为图片格式文件之前，要做转换
    frames_np_to_torch = [torch.from_numpy(frame) for frame in frames_np]
    frames_np_to_torch = torch.stack(frames_np_to_torch).permute(0,3, 1, 2).cuda()
    diff = torch.abs(frames_np_to_torch.cpu() - frames_torch.cpu()).to(torch.float)
    print('-------------比较torch和pil解码api结果的差异-------------')  # 对比结果显示两个不同的解码api存在差异
    print(f"Maximum origin difference across all images: {diff.max().item()}")
    print(f"Minimum origin difference across all images: {diff.min().item()}")
    print(f"Median origin difference: {diff.mean().item()}")
    

    # 实验结果：
    # 1 当torch和opencv都设置为InterpolationMode.NEAREST、cv2.INTER_NEAREST时，resize结果能完全对齐，max diff为0
    # 2 当opencv不指定interpolation，即使用默认的cv2.INTER_LINEAR，torch指定与其对应的InterpolationMode.BILINEAR 时，有0.007的max diff
    # 3 torch有antialias(抗锯齿)参数，并且默认是true，这会造成和opencv的差异加大

    # 使用torch api resize
    resize = torchvision.transforms.Resize(size,interpolation=InterpolationMode.BILINEAR,antialias=False)
    frames_torch_resized = resize(frames_torch)
    frames_torch_resized = normalize_video(frames_torch_resized).cpu()

    # 使用opencv resize
    opencv_interpolation = cv2.INTER_LINEAR #cv2.INTER_LINEAR
    strat_time = time.time()
    frames_np_resized = [cv2.resize(frame, (size[1],size[0]),interpolation=opencv_interpolation) for frame in frames_np]
    print(f'Opencv resize:{len(frames_np)} frames time:{time.time() - strat_time}')
    frames_np_resized_to_torch = [torch.from_numpy(frame) for frame in frames_np_resized]
    frames_np_resized_to_torch = torch.stack(frames_np_resized_to_torch).permute(0,3, 1, 2)
    frames_np_resized_to_torch = normalize_video(frames_np_resized_to_torch)

    # opencv resize默认使用 INTER_LINEAR，torch中对应的为InterpolationMode.BILINEAR   
    diff = torch.abs(frames_np_resized_to_torch - frames_torch_resized).to(torch.float)
    print('-------------比较解码和resize都使用torch与opencv之间的差异-------------')
    print(f"Maximum difference across all images: {diff.max().item()}")
    print(f"Minimum difference across all images: {diff.min().item()}")
    print(f"Median difference: {diff.mean().item()}")
    non_zero_pixels = torch.sum(diff > 0).item()
    print(f"Number of pixels with non-zero difference: {non_zero_pixels}")
    zero_pixels = torch.sum(diff == 0).item()
    print(f"Number of pixels with zero difference: {zero_pixels}")
    lt_mean_pixels = torch.sum(diff > diff.mean().item()).item()
    print(f"Number of  pixels with lt_mean difference: {lt_mean_pixels}")


    # 使用torch api resize,输入为pil解码的结果
    strat_time = time.time()
    frames_np_to_torch_resized = resize(frames_np_to_torch)
    torch.cuda.synchronize()
    print(f'Torch resize:{len(frames_np)} frames time:{time.time() - strat_time}')
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
    print(f"Number of  pixels with lt_mean difference: {lt_mean_pixels}")


if __name__ == "__main__":
    path = './test_set/video'

    #目的：比较torchvision的resize和opencv的resize之间的精度差异
    compare_resize(path)