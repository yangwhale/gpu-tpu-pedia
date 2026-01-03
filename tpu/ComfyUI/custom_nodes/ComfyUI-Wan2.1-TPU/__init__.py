"""
ComfyUI-Wan-TPU
===============

使用 diffusers torchax 优化模型在 TPU 上运行 Wan 2.1 视频生成。

注意: JAX/TPU 初始化和 torchax.enable_globally() 都延迟到节点执行时才进行。
这样可以确保模型在 torchax 环境外加载，避免 JIT 追踪问题。

Nodes:
  - Wan21TextEncoder: TPU 上运行 T5-XXL 编码 prompt
  - Wan21TPUSampler: TPU 上运行 Transformer 生成 latents  
  - Wan21TPUVAEDecoder: TPU 上运行 VAE 解码 latents 为视频
  - Wan21TPUPipeline: 端到端 Pipeline
"""

import logging
import os
import warnings

# ============================================================================
# 基础环境配置（不涉及 TPU）
# ============================================================================

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')
logging.getLogger('diffusers').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)


# ============================================================================
# 导出 Nodes（不导入任何 TPU 相关代码）
# ============================================================================

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = [
    'NODE_CLASS_MAPPINGS',
    'NODE_DISPLAY_NAME_MAPPINGS',
]

print("[ComfyUI-Wan-TPU] Custom nodes registered (TPU init deferred)")
