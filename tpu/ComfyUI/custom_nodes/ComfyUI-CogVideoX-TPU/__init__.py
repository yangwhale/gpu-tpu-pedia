"""
ComfyUI-CogVideoX-TPU

CogVideoX video generation on Google Cloud TPU with Splash Attention optimization.

Three-stage pipeline:
1. CogVideoXTextEncoder - T5 text encoding on TPU
2. CogVideoXTPUSampler - Transformer denoising with custom exp2-optimized Splash Attention
3. CogVideoXTPUVAEDecoder - VAE decoding on TPU
"""


# ============================================================================
# 注册模型卸载钩子
# ============================================================================

def _register_tpu_cleanup_hook():
    """
    注册 TPU 模型清理钩子到 comfy.model_management.unload_all_models()
    
    当用户点击 ComfyUI Manager 的 "Unload Models" 或 "Free Models and Node Cache" 按钮时，
    会调用 comfy.model_management.unload_all_models()，我们通过 monkey-patch 让它
    同时调用我们的 cleanup_cogvideox_tpu_models() 函数来清理 TPU 缓存。
    """
    try:
        import comfy.model_management as mm
        
        # 避免重复注册
        if hasattr(mm, '_cogvideox_tpu_cleanup_registered') and mm._cogvideox_tpu_cleanup_registered:
            return
        
        # 保存原始函数
        _original_unload_all_models = mm.unload_all_models
        
        def _patched_unload_all_models():
            """带有 TPU 清理的 unload_all_models"""
            # 先调用原始函数
            _original_unload_all_models()
            
            # 然后清理 TPU 缓存
            try:
                from .nodes import cleanup_cogvideox_tpu_models
                cleanup_cogvideox_tpu_models()
            except Exception as e:
                print(f"[CogVideoX-TPU] Warning: Failed to cleanup TPU models: {e}")
        
        # 替换原始函数
        mm.unload_all_models = _patched_unload_all_models
        mm._cogvideox_tpu_cleanup_registered = True
        print("[ComfyUI-CogVideoX-TPU] Registered TPU cleanup hook for unload_all_models()")
        
    except ImportError:
        print("[ComfyUI-CogVideoX-TPU] Warning: Could not import comfy.model_management, cleanup hook not registered")
    except Exception as e:
        print(f"[ComfyUI-CogVideoX-TPU] Warning: Failed to register cleanup hook: {e}")


# 注册钩子
_register_tpu_cleanup_hook()


# ============================================================================
# 导出 Nodes
# ============================================================================

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("[ComfyUI-CogVideoX-TPU] Custom nodes registered (TPU init deferred)")
