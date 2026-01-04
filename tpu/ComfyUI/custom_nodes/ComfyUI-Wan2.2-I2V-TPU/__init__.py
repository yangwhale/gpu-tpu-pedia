"""
ComfyUI Wan 2.2 I2V TPU Custom Nodes
====================================

在 TPU 上运行 Wan 2.2 Image-to-Video 生成。

Nodes:
  - Wan22I2VImageEncoder: 图像条件编码
  - Wan22I2VTextEncoder: 文本编码 (UMT5-XXL)
  - Wan22I2VTPUSampler: 双 Transformer 去噪
  - Wan22I2VTPUVAEDecoder: VAE 解码

使用方法:
  1. 加载输入图像
  2. 使用 Wan22I2VImageEncoder 编码图像条件
  3. 使用 Wan22I2VTextEncoder 编码文本提示
  4. 使用 Wan22I2VTPUSampler 运行去噪
  5. 使用 Wan22I2VTPUVAEDecoder 解码视频

基于 gpu-tpu-pedia/tpu/Wan2.2/generate_diffusers_i2v_torchax_staged 实现。
"""

from .nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
