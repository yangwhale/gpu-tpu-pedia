"""
TPU Kernels 模块

提供 TPU 优化的 Pallas kernel 实现，可供多个项目共用。
"""

from .splash_attention_utils import (
    tpu_splash_attention,
    sdpa_reference,
)

__all__ = [
    'tpu_splash_attention',
    'sdpa_reference',
]
