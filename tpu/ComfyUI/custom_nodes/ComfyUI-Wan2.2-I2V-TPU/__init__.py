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


# ============================================================================
# 注册模型卸载钩子
# ============================================================================

def _register_tpu_cleanup_hook():
    """
    注册 TPU 模型清理钩子到 comfy.model_management.unload_all_models()
    
    当用户点击 ComfyUI Manager 的 "Unload Models" 或 "Free Models and Node Cache" 按钮时，
    会调用 comfy.model_management.unload_all_models()，我们通过 monkey-patch 让它
    同时调用我们的 cleanup_wan22_i2v_tpu_models() 函数来清理 TPU 缓存。
    """
    try:
        import comfy.model_management as mm
        
        # 避免重复注册
        if hasattr(mm, '_wan22_i2v_tpu_cleanup_registered') and mm._wan22_i2v_tpu_cleanup_registered:
            return
        
        # 保存原始函数
        _original_unload_all_models = mm.unload_all_models
        
        def _patched_unload_all_models():
            """带有 TPU 清理的 unload_all_models"""
            # 先调用原始函数
            _original_unload_all_models()
            
            # 然后清理 TPU 缓存
            try:
                from .nodes import cleanup_wan22_i2v_tpu_models
                cleanup_wan22_i2v_tpu_models()
            except Exception as e:
                print(f"[Wan22I2V-TPU] Warning: Failed to cleanup TPU models: {e}")
        
        # 替换原始函数
        mm.unload_all_models = _patched_unload_all_models
        mm._wan22_i2v_tpu_cleanup_registered = True
        print("[ComfyUI-Wan2.2-I2V-TPU] Registered TPU cleanup hook for unload_all_models()")
        
    except ImportError:
        print("[ComfyUI-Wan2.2-I2V-TPU] Warning: Could not import comfy.model_management, cleanup hook not registered")
    except Exception as e:
        print(f"[ComfyUI-Wan2.2-I2V-TPU] Warning: Failed to register cleanup hook: {e}")


# 注册钩子
_register_tpu_cleanup_hook()


# ============================================================================
# 导出 Nodes
# ============================================================================

from .nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print("[ComfyUI-Wan2.2-I2V-TPU] Custom nodes registered (TPU init deferred)")
