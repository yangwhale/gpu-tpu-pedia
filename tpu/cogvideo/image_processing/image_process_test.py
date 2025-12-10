
import os

import time
from typing import Tuple

import torch
import torchvision
from PIL import Image

import numpy as np
from torchvision.transforms import InterpolationMode


# opencv的resize没有antialias，等价与torch vison的antialias=False
def resize_video(video_frames:torch.Tensor, #batch, frame,channel,height,width 或者 frame,channel,height,width
                 resize_height:int,
                 resize_width:int,
                 interpolation:InterpolationMode = InterpolationMode.BILINEAR,
                 antialias:bool = False) -> torch.Tensor:
    '''
        torchvision resize 不支持5D的tensor，需要合并一下维度
    '''
    resize = torchvision.transforms.Resize((resize_height, resize_width),interpolation=interpolation,antialias=antialias)
    origin_shape = video_frames.shape
    frames, channels, origi_height, origi_width = origin_shape[-4:]
    if len(origin_shape) == 5:
        video_frames = video_frames.reshape(-1, channels, origi_height, origi_width)

    frames_mem = video_frames.numel() * video_frames.element_size() / (1024**2) #计算batch的大小，过大的话resize可能会oom，拆小
    frames_mem = int(frames_mem)
    max_mem = 200  #TODO 可以放入配置中
    if frames_mem > max_mem:
        chunks =  (frames_mem + max_mem - 1) // max_mem
        video_frames = video_frames.chunk(chunks,dim=0)
        result = []
        for chunk in video_frames:
            result.append(resize(chunk))
        result = torch.cat(result)
    else:
        result = resize(video_frames)

    if len(origin_shape) == 5:
        result = result.reshape(origin_shape[0], frames, channels, resize_height, resize_width)
    return result

def resize_frames(video_frames: torch.Tensor,
                  mask_frames: torch.Tensor,
                  image_size:int=640,
                  max_length:int=1280) -> Tuple[torch.Tensor, torch.Tensor]:
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
        return (resize_video(video_frames,h_resize,w_resize),
                resize_video(mask_frames,h_resize,w_resize))


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
        return None,video_frames,mask_frames

    # Calculate bounding box
    nonzero_rows = torch.where(rows)[0]
    nonzero_cols = torch.where(cols)[0]
    rmin, rmax = nonzero_rows[[0, -1]].tolist() # 转成int标量
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


def preprocess(video_path,mask_path):
    video_files = [os.path.join(video_path, f) for f in os.listdir(video_path) if f.lower().endswith('.png')]
    mask_files = [os.path.join(mask_path, f) for f in os.listdir(mask_path) if f.lower().endswith('.png')]

    if not video_files or not mask_files or len(video_files) != len(mask_files):
        print("frame 文件数据不正确")
        return
    
    video_frames_np = [np.array(Image.open(frame)) for frame in video_files]
    #video_frames_np = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in video_frames_np]

    mask_frames_np = [np.array(Image.open(frame)) for frame in mask_files]
    mask_frames_np = [(mask > 0.5).astype(np.uint8) for mask in mask_frames_np]

    video_frames_np_to_torch = np_to_torch(video_frames_np).permute(0,3, 1, 2).cuda()
    mask_frames_np_to_torch = np_to_torch(mask_frames_np).permute(0,3, 1, 2).cuda()


    start_time = time.time()
    coordinates_torch, cropped_video_torch, cropped_mask_torch = crop_based_on_mask_torch(video_frames_np_to_torch,mask_frames_np_to_torch,0.5,(512,512))
    torch.cuda.synchronize()
    print(f'Torch crop time:{time.time() - start_time}')

    start_time = time.time()
    video_resized_torch, mask_resized_torch = resize_frames(cropped_video_torch,cropped_mask_torch)
    torch.cuda.synchronize()
    mask_resized_torch = torch.where(mask_resized_torch > 0.5, 1, 0).to(dtype=torch.uint8)
    print(f'Torch resize time:{time.time() - start_time}')


    return cropped_video_torch,cropped_mask_torch,

    


def postprocess(input_video_torch,input_mask_torch):
    input_video_torch = input_video_torch.cuda()
    input_mask_torch = input_mask_torch.cuda()

    import torchvision.transforms.functional as TF
    kernel_size = int(3 * 2 + 1)
    start_time = time.time()
    blur_mask_torch = TF.gaussian_blur(input_mask_torch * 255,kernel_size=kernel_size,sigma=1.5) / 255  # [N,C,H,W]
    torch.cuda.synchronize()
    print(f'Torch gaussian_blur time:{time.time() - start_time}')


            
if __name__ == "__main__":
    video_path = './test_set/video'
    mask_path = './test_set/mask'

    # 目的：
    # 1.测试torch下常规的tensor数学和逻辑计算（crop_based_on_mask_torch方法中体现）
    # 2.测试TorchVision的图像处理算法(resize和gaussian_blur方法中体现)
    cropped_video_torch,cropped_mask_torch = preprocess(video_path,mask_path)
    postprocess(cropped_video_torch,cropped_video_torch)