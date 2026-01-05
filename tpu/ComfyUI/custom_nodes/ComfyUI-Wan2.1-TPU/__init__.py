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
# 注册模型卸载钩子
# ============================================================================

def _register_tpu_cleanup_hook():
    """
    注册 TPU 模型清理钩子到 comfy.model_management.unload_all_models()
    
    当用户点击 ComfyUI Manager 的 "Unload Models" 或 "Free Models and Node Cache" 按钮时，
    会调用 comfy.model_management.unload_all_models()，我们通过 monkey-patch 让它
    同时调用我们的 cleanup_wan21_tpu_models() 函数来清理 TPU 缓存。
    """
    try:
        import comfy.model_management as mm
        
        # 避免重复注册
        if hasattr(mm, '_wan21_tpu_cleanup_registered') and mm._wan21_tpu_cleanup_registered:
            return
        
        # 保存原始函数
        _original_unload_all_models = mm.unload_all_models
        
        def _patched_unload_all_models():
            """带有 TPU 清理的 unload_all_models"""
            # 先调用原始函数
            _original_unload_all_models()
            
            # 然后清理 TPU 缓存
            try:
                from .nodes import cleanup_wan21_tpu_models
                cleanup_wan21_tpu_models()
            except Exception as e:
                print(f"[Wan21-TPU] Warning: Failed to cleanup TPU models: {e}")
        
        # 替换原始函数
        mm.unload_all_models = _patched_unload_all_models
        mm._wan21_tpu_cleanup_registered = True
        print("[ComfyUI-Wan-TPU] Registered TPU cleanup hook for unload_all_models()")
        
    except ImportError:
        print("[ComfyUI-Wan-TPU] Warning: Could not import comfy.model_management, cleanup hook not registered")
    except Exception as e:
        print(f"[ComfyUI-Wan-TPU] Warning: Failed to register cleanup hook: {e}")


# 注册钩子
_register_tpu_cleanup_hook()


# ============================================================================
# 导出 Nodes（不导入任何 TPU 相关代码）
# ============================================================================

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = [
    'NODE_CLASS_MAPPINGS',
    'NODE_DISPLAY_NAME_MAPPINGS',
]

print("[ComfyUI-Wan-TPU] Custom nodes registered (TPU init deferred)")
