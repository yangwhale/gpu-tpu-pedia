#!/usr/bin/env python3
"""
Flux.2 Text-to-Image 生成脚本 (GPU)

在 GPU 上使用 Diffusers 运行 Flux.2 图像生成。
"""

import torch
from diffusers import Flux2Pipeline

# 配置
MODEL_ID = "black-forest-labs/FLUX.2-dev"
DEVICE = "cuda:0"
DTYPE = torch.bfloat16

# 默认 prompt
DEFAULT_PROMPT = (
    "Realistic macro photograph of a hermit crab using a soda can as its shell, "
    "partially emerging from the can, captured with sharp detail and natural colors, "
    "on a sunlit beach with soft shadows and a shallow depth of field, "
    "with blurred ocean waves in the background. "
    "The can has the text `BFL Diffusers` on it and it has a color gradient "
    "that start with #FF5733 at the top and transitions to #33FF57 at the bottom."
)


def main():
    """运行 Flux.2 图像生成。"""
    pipe = Flux2Pipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    # 对于 >80G VRAM 的卡（H200, B200 等），可以用 pipe.to(device) 代替
    pipe.enable_model_cpu_offload()
    
    image = pipe(
        prompt=DEFAULT_PROMPT,
        generator=torch.Generator(device=DEVICE).manual_seed(42),
        num_inference_steps=50,
        guidance_scale=4,
    ).images[0]
    
    image.save("flux2_output_gpu.png")
    print("✓ 图像已保存至 flux2_output_gpu.png")


if __name__ == "__main__":
    main()
