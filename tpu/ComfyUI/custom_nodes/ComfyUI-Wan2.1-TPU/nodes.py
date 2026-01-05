"""
ComfyUI Wan 2.1 TPU Nodes
=========================

使用 diffusers 的 torchax 优化模型在 TPU 上运行 Wan 2.1 视频生成。
基于 gpu-tpu-pedia/tpu/Wan2.1/generate_diffusers_torchax_staged 实现。

Nodes:
  - Wan21TextEncoder: TPU 上运行 T5-XXL 编码 prompt
  - Wan21TPUSampler: TPU 上运行 Transformer 生成 latents
  - Wan21TPUVAEDecoder: TPU 上运行 VAE 解码 latents 为视频
  - Wan21TPUPipeline: 端到端视频生成 Pipeline

核心设计（Hybrid 方案）：
  - 使用 `enable_globally()` 保持 Mode 栈激活（解决 XLA tensor 逃逸问题）
  - 模型缓存后权重保持 XLA 状态
  - 节点返回值必须转为 CPU tensor（确保与 ComfyUI 兼容）
  - 参考：docs/torchax_comfyui_integration.md
"""

import functools
import gc
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.experimental import mesh_utils
from jax.sharding import Mesh

# ComfyUI progress bar and server
try:
    from comfy.utils import ProgressBar
    from server import PromptServer
    _HAS_SERVER = True
except ImportError:
    # 独立运行时的 fallback
    class ProgressBar:
        def __init__(self, total, node_id=None):
            self.total = total
            self.current = 0
            self.node_id = node_id
        def update(self, n=1):
            self.current += n
        def update_absolute(self, value, total=None, preview=None):
            self.current = value
    PromptServer = None
    _HAS_SERVER = False

# Hybrid 方案：使用 enable_globally() 保持 Mode 栈激活
# 这样缓存的 XLA 模型可以在后续调用中正常工作

# 从 utils 导入所有必要的函数和常量
from .utils import (
    TEXT_ENCODER_SHARDINGS,
    TRANSFORMER_SHARDINGS,
    VAE_DECODER_SHARDINGS,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_FRAMES,
    DEFAULT_FPS,
    DEFAULT_FLOW_SHIFT,
    move_module_to_xla,
    prepare_video_for_export,
    shard_weight_dict,
    setup_jax_cache,
    setup_pytree_registrations,
    register_text_encoder_ops,
    register_operators_on_env,
)

# 全局 mesh（延迟创建）
_mesh = None


# ============================================================================
# 模型缓存清理函数
# ============================================================================

def cleanup_wan21_tpu_models():
    """
    清理所有 Wan 2.1 TPU 模型缓存。
    
    当用户点击 ComfyUI Manager 的 "Unload Models" 或 "Free Models and Node Cache" 按钮时，
    ComfyUI 会调用 comfy.model_management.unload_all_models()，
    我们通过 monkey-patch 让它同时调用此函数来清理 TPU 缓存。
    
    清理内容：
    - Wan21TextEncoder: Text Encoder Pipeline
    - Wan21TPUSampler: Transformer Pipeline
    - Wan21TPUVAEDecoder: VAE Decoder
    - 全局 Mesh
    - Torchax 全局状态
    - JAX 编译缓存
    """
    global _mesh, _torchax_env, _ops_registered, _globally_enabled
    
    print("\n[Wan21-TPU] Cleaning up cached models...")
    
    # 清理 TextEncoder 缓存
    if Wan21TextEncoder._cached_pipe is not None:
        print("  - Clearing Text Encoder cache")
        del Wan21TextEncoder._cached_pipe
        Wan21TextEncoder._cached_pipe = None
        Wan21TextEncoder._cached_model_id = None
        Wan21TextEncoder._is_compiled = False
        Wan21TextEncoder._env = None
    
    # 清理 Sampler 缓存
    if Wan21TPUSampler._cached_pipe is not None:
        print("  - Clearing Transformer Pipeline cache")
        del Wan21TPUSampler._cached_pipe
        Wan21TPUSampler._cached_pipe = None
        Wan21TPUSampler._cached_model_id = None
        Wan21TPUSampler._env = None
        Wan21TPUSampler._mesh = None
    
    # 清理 VAE 缓存
    if Wan21TPUVAEDecoder._cached_vae is not None:
        print("  - Clearing VAE Decoder cache")
        del Wan21TPUVAEDecoder._cached_vae
        Wan21TPUVAEDecoder._cached_vae = None
        Wan21TPUVAEDecoder._cached_model_id = None
        Wan21TPUVAEDecoder._env = None
    
    # 清理全局 mesh
    if _mesh is not None:
        print("  - Clearing global Mesh")
        _mesh = None
    
    # 重置 torchax 全局状态
    _torchax_env = None
    _ops_registered = False
    
    # 禁用 torchax 全局模式
    if _globally_enabled:
        try:
            import torchax
            torchax.disable_globally()
            print("  - Disabled torchax globally")
        except Exception as e:
            print(f"  - Warning: Could not disable torchax: {e}")
        _globally_enabled = False
    
    # 清理 JAX 缓存
    try:
        jax.clear_caches()
        print("  - Cleared JAX caches")
    except Exception as e:
        print(f"  - Warning: Could not clear JAX caches: {e}")
    
    # 强制垃圾回收
    gc.collect()
    
    print("[Wan21-TPU] Cleanup complete!\n")


def get_mesh():
    """获取全局 mesh，如果不存在则创建"""
    global _mesh
    if _mesh is None:
        print("[Wan21] Creating 2D Mesh for TPU...")
        devices = jax.devices('tpu')
        dp_dim = min(2, len(devices))
        tp_dim = len(devices) // dp_dim
        mesh_devices = mesh_utils.create_device_mesh(
            (dp_dim, tp_dim), allow_split_physical_axes=True
        )
        _mesh = Mesh(mesh_devices, ("dp", "tp"))
        print(f"[Wan21] Created Mesh: dp={dp_dim}, tp={tp_dim}")
    return _mesh


# ============================================================================
# 性能配置开关
# ============================================================================

# 设置为 True 启用完整算子注册 + inference_mode(True)
# 设置为 False 使用 VAE conv2d only + inference_mode(False) fallback（更快）
USE_FULL_OPS_REGISTRATION = False


# ============================================================================
# 公共工具函数
# ============================================================================

def to_cpu_tensor(tensor):
    """
    将 XLA tensor 安全转换为 CPU tensor。
    
    这是一个公共函数，供所有节点类使用，避免重复代码。
    
    处理逻辑：
    1. XLA tensor（有 _elem 属性）：转换 JAX array -> numpy -> torch
    2. bfloat16：先转为 float32 再转为 numpy（numpy 不支持 bfloat16）
    3. 普通 torch tensor：调用 .cpu()
    4. 其他类型：直接返回
    
    Args:
        tensor: XLA tensor, torch tensor, 或其他类型
        
    Returns:
        CPU 上的 torch tensor
    """
    if hasattr(tensor, '_elem'):
        # XLA tensor: 转换为 numpy 再转为 torch
        jax_arr = tensor._elem
        if jax_arr.dtype == jnp.bfloat16:
            np_arr = np.array(jax_arr.astype(jnp.float32))
            return torch.from_numpy(np_arr).to(torch.bfloat16)
        else:
            return torch.from_numpy(np.array(jax_arr))
    elif hasattr(tensor, 'cpu'):
        return tensor.cpu()
    else:
        return tensor


# ============================================================================
# Torchax 环境管理（Hybrid 方案：enable_globally）
# ============================================================================

# 全局状态
_torchax_env = None
_ops_registered = False
_globally_enabled = False
_jax_cache_initialized = False


def ensure_torchax_enabled(mesh_obj=None):
    """
    确保 torchax 全局启用，返回 env。
    
    Hybrid 方案：
    - 首次调用时 enable_globally()，之后保持启用
    - 这样缓存的 XLA 模型权重可以在后续调用中正常工作
    - 节点返回值必须转为 CPU tensor（由各节点负责）
    - 初始化 JAX 编译缓存以加速后续编译
    
    算子注册策略：
    - USE_FULL_OPS_REGISTRATION=True: 注册所有算子，使用 inference_mode(True)
    - USE_FULL_OPS_REGISTRATION=False: 只注册必要算子，使用 inference_mode(False) fallback
    """
    global _torchax_env, _ops_registered, _globally_enabled, _jax_cache_initialized
    import torchax
    
    # 首次调用时设置 JAX 编译缓存
    if not _jax_cache_initialized:
        setup_jax_cache()
        _jax_cache_initialized = True
    
    # 首次调用时全局启用
    if not _globally_enabled:
        print("[Torchax] Enabling globally (Hybrid mode)...")
        torchax.enable_globally()
        _globally_enabled = True
    
    # 获取 env
    if _torchax_env is None:
        _torchax_env = torchax.default_env()
    
    # 只在 USE_FULL_OPS_REGISTRATION=True 时注册所有算子
    if USE_FULL_OPS_REGISTRATION and not _ops_registered and mesh_obj is not None:
        print("[Torchax] Registering ALL operators (full mode)...")
        register_operators_on_env(_torchax_env, mesh_obj)
        _ops_registered = True
    
    return _torchax_env


# 保留旧函数名以兼容
def get_torchax_env(mesh_obj=None):
    """兼容旧代码，内部调用 ensure_torchax_enabled"""
    return ensure_torchax_enabled(mesh_obj)


# ============================================================================
# Wan 2.1 Text Encoder (TPU) - Hybrid 方案
# ============================================================================

class Wan21TextEncoder:
    """
    Wan 2.1 Text Encoder - 在 TPU 上运行 T5-XXL 编码 prompt。
    
    Hybrid 方案：
    - 使用 ensure_torchax_enabled() 保持 Mode 栈激活
    - 模型缓存后权重保持 XLA 状态
    - 返回值转为 CPU tensor（确保与 ComfyUI 兼容）
    
    输入: prompt, negative_prompt 文本
    输出: prompt_embeds, negative_prompt_embeds tensor (CPU)
    """
    
    _cached_pipe = None
    _cached_model_id = None
    _is_compiled = False
    _env = None  # 缓存 env 对象
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True,
                    "default": "A cat and a dog baking a cake together in a kitchen."}),
                "negative_prompt": ("STRING", {"multiline": True,
                    "default": "Bright tones, overexposed, static, blurred details, low quality"}),
            },
            "optional": {
                "model_id": ("STRING", {"default": "Wan-AI/Wan2.1-T2V-14B-Diffusers"}),
            }
        }
    
    RETURN_TYPES = ("TENSOR", "TENSOR")
    RETURN_NAMES = ("prompt_embeds", "negative_prompt_embeds")
    FUNCTION = "encode"
    CATEGORY = "TPU/Wan2.1"
    
    def encode(self, prompt, negative_prompt, model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers"):
        print(f"\n[Wan21TextEncoder] Encoding prompt on TPU...")
        print(f"  Prompt: {prompt[:50]}...")
        
        real_mesh = get_mesh()
        pipe, env = self._get_or_create_pipeline(model_id, real_mesh)
        
        print("  Encoding prompts...")
        # Hybrid 方案：enable_globally() 已激活，只需 with mesh: 用于 sharding context
        with real_mesh:
            # 根据配置决定 inference_mode
            # - USE_FULL_OPS_REGISTRATION=True: 所有算子已注册，可以使用 inference_mode(True)
            # - USE_FULL_OPS_REGISTRATION=False: 需要 fallback，必须用 inference_mode(False)
            use_inference_mode = USE_FULL_OPS_REGISTRATION
            with torch.inference_mode(use_inference_mode), torch.no_grad():
                prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    do_classifier_free_guidance=True,
                    num_videos_per_prompt=1,
                    device='jax',
                )
            
            # 转换回 CPU（返回给 ComfyUI）
            prompt_embeds_cpu = to_cpu_tensor(prompt_embeds)
            negative_prompt_embeds_cpu = to_cpu_tensor(negative_prompt_embeds)
        
        print(f"  prompt_embeds shape: {prompt_embeds_cpu.shape}")
        return (prompt_embeds_cpu, negative_prompt_embeds_cpu)
    
    @classmethod
    def _get_or_create_pipeline(cls, model_id, mesh):
        """
        加载和配置 Pipeline（Hybrid 方案）
        
        流程：
        1. 注册 PyTree
        2. 临时禁用 torchax 加载模型（避免拦截 transformers 加载逻辑）
        3. 启用 torchax 并注册算子
        4. 在 with mesh: 块内：move_to_xla, compile, shard
        """
        import torchax
        
        if (cls._cached_pipe is not None and
            cls._cached_model_id == model_id and
            cls._is_compiled):
            print("  Using cached pipeline")
            return cls._cached_pipe, cls._env
        
        print(f"  Loading Wan 2.1 Pipeline from {model_id}...")
        
        # ===== 步骤 1：注册 PyTree =====
        setup_pytree_registrations()
        
        # ===== 步骤 2：加载模型（禁用 torchax 避免拦截 transformers 加载逻辑）=====
        # 参考：stage2_transformer.py:514-530
        global _globally_enabled
        if _globally_enabled:
            torchax.disable_globally()
            _globally_enabled = False
        
        from diffusers import WanPipeline
        torch.set_default_dtype(torch.bfloat16)
        
        pipe = WanPipeline.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, use_safetensors=True
        )
        print("  ✓ 模型加载完成")
        
        # ===== 步骤 3：启用 torchax 并注册算子 =====
        env = ensure_torchax_enabled(mesh)
        
        # 只在 USE_FULL_OPS_REGISTRATION=True 时注册 Text Encoder 专用算子
        if USE_FULL_OPS_REGISTRATION:
            print("  注册 Text Encoder 算子...")
            register_text_encoder_ops(env)
        
        # ===== 步骤 4：在 with mesh: 块内设置 Text Encoder =====
        print(f"  Mesh: {mesh}")
        with mesh:
            print("  - 移动 Text Encoder 到 TPU...")
            move_module_to_xla(env, pipe.text_encoder)
            pipe.text_encoder = torchax.compile(pipe.text_encoder)
            
            print(f"  - Sharding Text Encoder weights...")
            pipe.text_encoder.params = shard_weight_dict(
                pipe.text_encoder.params, TEXT_ENCODER_SHARDINGS, mesh
            )
            pipe.text_encoder.buffers = shard_weight_dict(
                pipe.text_encoder.buffers, TEXT_ENCODER_SHARDINGS, mesh
            )
            
            # 等待分片完成
            torchax.interop.call_jax(jax.block_until_ready, pipe.text_encoder.params)
        
        print("  ✓ Text Encoder 设置完成")
        
        # 删除不需要的组件
        if hasattr(pipe, 'transformer') and pipe.transformer is not None:
            del pipe.transformer
            pipe.transformer = None
        if hasattr(pipe, 'vae') and pipe.vae is not None:
            del pipe.vae
            pipe.vae = None
        
        gc.collect()
        cls._cached_pipe = pipe
        cls._cached_model_id = model_id
        cls._is_compiled = True
        cls._env = env
        print("  Text Encoder ready!")
        return pipe, env


# ============================================================================
# Wan 2.1 TPU Sampler - Hybrid 方案
# ============================================================================

def _scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                   is_causal=False, scale=None, enable_gqa=False,
                                   env=None, mesh=None):
    """封装 SDPA，长序列使用 TPU Splash Attention。"""
    try:
        from .splash_attention import sdpa_reference, tpu_splash_attention
    except ImportError:
        from splash_attention import sdpa_reference, tpu_splash_attention
    
    USE_K_SMOOTH = True
    
    # 仅对长序列（self-attention）使用 TPU Splash Attention
    if key.shape[2] > 20000:
        assert attn_mask is None
        assert dropout_p == 0.0
        assert is_causal is False
        assert enable_gqa is False
        assert scale is None
        
        jquery, jkey, jvalue = env.t2j_iso((query, key, value))
        
        if USE_K_SMOOTH:
            key_mean = jnp.mean(jkey, axis=2, keepdims=True)
            jkey = jkey - key_mean
        
        res = tpu_splash_attention(jquery, jkey, jvalue, mesh, scale=scale)
        return env.j2t_iso(res)

    return sdpa_reference(query, key, value, attn_mask, dropout_p,
                          is_causal, scale, enable_gqa)


def _setup_pipeline_for_transformer_only(pipe, mesh, env):
    """
    设置 Pipeline 仅用于 Transformer 推理（不包含 VAE）
    
    注意：此函数应该在 `with mesh, env:` 块内调用！
    """
    from torchax.ops import ops_registry, jaten
    import torchax
    
    print("\n=== 配置 Transformer (TPU) ===")

    def override_op(op, impl):
        """注册或覆盖一个算子"""
        env._ops[op] = ops_registry.Operator(
            op, impl, is_jax_function=False, is_user_defined=True,
            needs_env=False, is_view_op=False,
        )

    # Register conv3d for WanTransformer3DModel.patch_embedding
    print("- 注册 conv3d 算子...")
    def conv3d_impl(input, weight, bias=None, stride=1, padding=0,
                    dilation=1, groups=1, *, env=env):
        """
        3D 卷积实现，用于 WanTransformer3DModel 的 patch_embedding。
        """
        jinput, jweight, jbias = env.t2j_iso((input, weight, bias))
        res = jaten._aten_convolution(
            jinput, jweight, jbias,
            stride, padding, dilation,
            transposed=False,
            output_padding=1,
            groups=groups
        )
        return env.j2t_iso(res)
    
    # 注册所有可能的 conv3d 变体
    override_op(torch.nn.functional.conv3d, functools.partial(conv3d_impl, env=env))
    try:
        override_op(torch.ops.aten.conv3d, functools.partial(conv3d_impl, env=env))
        override_op(torch.ops.aten.conv3d.default, functools.partial(conv3d_impl, env=env))
    except Exception as e:
        print(f"  Warning: Failed to register aten.conv3d variants: {e}")

    # Register custom attention
    print("- 注册自定义 JAX 算子...")
    custom_attention = functools.partial(
        _scaled_dot_product_attention,
        env=env,
        mesh=mesh,
    )
    op_to_override = torch.nn.functional.scaled_dot_product_attention
    override_op(op_to_override, custom_attention)

    # Move Transformer to XLA
    print("- 将 Transformer 移到 TPU...")
    move_module_to_xla(env, pipe.transformer)
    
    # Move rope embeddings to JAX
    if hasattr(pipe.transformer.rope, 'freqs'):
        pipe.transformer.rope.freqs = pipe.transformer.rope.freqs.to('jax')
    else:
        pipe.transformer.rope.freqs_cos = pipe.transformer.rope.freqs_cos.to('jax')
        pipe.transformer.rope.freqs_sin = pipe.transformer.rope.freqs_sin.to('jax')

    # Compile Transformer
    print("- 编译 Transformer...")
    options = torchax.CompileOptions(
        jax_jit_kwargs={'static_argnames': ('return_dict',)}
    )
    pipe.transformer = torchax.compile(pipe.transformer, options)

    # Apply sharding
    print("- 对 Transformer 进行权重分片...")
    pipe.transformer.params = shard_weight_dict(
        pipe.transformer.params, TRANSFORMER_SHARDINGS, mesh
    )
    pipe.transformer.buffers = shard_weight_dict(
        pipe.transformer.buffers, TRANSFORMER_SHARDINGS, mesh
    )
    
    # Wait for sharding to complete
    torchax.interop.call_jax(jax.block_until_ready, pipe.transformer.params)

    # Delete VAE to save memory (not needed in stage 2)
    print("- 删除 VAE 以节省内存...")
    if hasattr(pipe, 'vae') and pipe.vae is not None:
        del pipe.vae
        pipe.vae = None

    # Delete Text Encoder (already used in stage 1)
    print("- 删除 Text Encoder...")
    if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
        del pipe.text_encoder
        pipe.text_encoder = None

    print("✓ Transformer 配置完成")
    return pipe


class Wan21TPUSampler:
    """
    Wan 2.1 TPU Sampler - 在 TPU 上运行 Transformer 去噪。
    
    Hybrid 方案：
    - 使用 ensure_torchax_enabled() 保持 Mode 栈激活
    - 模型缓存后权重保持 XLA 状态
    - 返回值转为 CPU tensor（确保与 ComfyUI 兼容）
    
    输入: prompt_embeds, negative_prompt_embeds
    输出: latents (用于 VAE Decoder)
    """
    
    _cached_pipe = None
    _cached_model_id = None
    _env = None  # 缓存 env 对象
    _mesh = None  # 缓存 mesh 对象
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_embeds": ("TENSOR",),
                "negative_prompt_embeds": ("TENSOR",),
                "height": ("INT", {"default": DEFAULT_HEIGHT, "min": 256, "max": 1280, "step": 16}),
                "width": ("INT", {"default": DEFAULT_WIDTH, "min": 256, "max": 1280, "step": 16}),
                "num_frames": ("INT", {"default": DEFAULT_FRAMES, "min": 17, "max": 121, "step": 4}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 2025, "min": 0, "max": 2**32 - 1}),
            },
            "optional": {
                "model_id": ("STRING", {"default": "Wan-AI/Wan2.1-T2V-14B-Diffusers"}),
                "flow_shift": ("FLOAT", {"default": DEFAULT_FLOW_SHIFT, "min": 1.0, "max": 10.0, "step": 0.5}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }
    
    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("latents", "num_frames")
    FUNCTION = "sample"
    CATEGORY = "TPU/Wan2.1"
    
    def sample(self, prompt_embeds, negative_prompt_embeds, height, width, num_frames,
               num_inference_steps, guidance_scale, seed,
               model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers", flow_shift=DEFAULT_FLOW_SHIFT,
               unique_id=None):
        """
        运行 Transformer 推理生成 latents（Hybrid 方案）
        """
        print(f"\n[Wan21TPUSampler] Starting TPU inference...")
        print(f"  Resolution: {width}x{height}, Frames: {num_frames}")
        print(f"  Steps: {num_inference_steps}, Guidance: {guidance_scale}, Seed: {seed}")
        
        # 注册 PyTree
        setup_pytree_registrations()
        
        # 加载 Pipeline（如果需要）
        pipe, mesh, env = self._get_or_create_pipeline(model_id, flow_shift)
        
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        # 运行推理
        print(f"\n=== 阶段2：Transformer 推理 ===")
        print(f"推理步数: {num_inference_steps}")
        print(f"帧数: {num_frames}")
        print(f"引导尺度: {guidance_scale}")
        
        # 创建 ComfyUI 进度条
        pbar = ProgressBar(num_inference_steps)
        
        # 用于计算每步时间和 ETA 的状态
        step_times = []
        loop_start_time = [None]  # 使用列表以便在闭包中修改
        
        # 进度回调函数
        def progress_callback(pipe, step_index, timestep, callback_kwargs):
            """每一步更新进度条和显示时间信息"""
            current_time = time.perf_counter()
            
            # 计算当前步耗时
            if loop_start_time[0] is not None:
                step_time = current_time - loop_start_time[0]
                step_times.append(step_time)
            loop_start_time[0] = current_time
            
            # 更新进度条
            pbar.update(1)
            
            # 如果有 PromptServer 且有 unique_id，发送进度文本
            if _HAS_SERVER and unique_id is not None and len(step_times) > 0:
                # 计算平均每步时间
                avg_step_time = sum(step_times) / len(step_times)
                # 已完成步数（step_index 是 0-based，callback 在步完成后调用）
                completed_steps = step_index + 1
                remaining_steps = num_inference_steps - completed_steps
                eta_seconds = remaining_steps * avg_step_time
                
                # 格式化 ETA
                if eta_seconds >= 60:
                    eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
                else:
                    eta_str = f"{eta_seconds:.1f}s"
                
                # 发送进度文本到 UI
                progress_text = f"Step {completed_steps}/{num_inference_steps} | {step_times[-1]:.2f}s/step | ETA: {eta_str}"
                try:
                    PromptServer.instance.send_progress_text(progress_text, unique_id)
                except Exception:
                    pass  # 忽略发送失败
            
            return callback_kwargs
        
        start_time = time.perf_counter()
        loop_start_time[0] = start_time  # 初始化循环开始时间
        
        # Hybrid 方案：enable_globally() 已激活，只需 with mesh: 用于 sharding context
        with mesh:
            # 将 embeddings 转换为 XLA tensor
            prompt_embeds_xla = prompt_embeds.to('jax')
            negative_prompt_embeds_xla = negative_prompt_embeds.to('jax')
            
            # 根据配置决定 inference_mode
            # - USE_FULL_OPS_REGISTRATION=True: 所有算子已注册，可以使用 inference_mode(True)
            # - USE_FULL_OPS_REGISTRATION=False: 需要 fallback，必须用 inference_mode(False)
            use_inference_mode = USE_FULL_OPS_REGISTRATION
            with torch.inference_mode(use_inference_mode), torch.no_grad():
                result = pipe(
                    prompt=None,
                    negative_prompt=None,
                    prompt_embeds=prompt_embeds_xla,
                    negative_prompt_embeds=negative_prompt_embeds_xla,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    output_type='latent',
                    use_dp=True,
                    callback_on_step_end=progress_callback,
                )
            jax.effects_barrier()
            
            # 转换 latents 为 CPU tensor（返回给 ComfyUI）
            torch_latents = to_cpu_tensor(result.frames)
        
        elapsed = time.perf_counter() - start_time
        print(f"\n✓ Transformer 推理完成，耗时: {elapsed:.2f} 秒")
        print(f"  平均每步时间: {elapsed/num_inference_steps:.2f}s")
        print(f"  Latents shape: {torch_latents.shape}")
        print(f"  Latents dtype: {torch_latents.dtype}")
        
        # 发送完成消息
        if _HAS_SERVER and unique_id is not None:
            try:
                PromptServer.instance.send_progress_text(
                    f"Complete! Total: {elapsed:.2f}s ({elapsed/num_inference_steps:.2f}s/step)",
                    unique_id
                )
            except Exception:
                pass
        
        return ({"samples": torch_latents, "num_frames": num_frames}, num_frames)
    
    def _get_or_create_pipeline(self, model_id, flow_shift):
        """
        加载和配置 Pipeline（Hybrid 方案）
        
        流程：
        1. 创建 mesh
        2. 临时禁用 torchax 加载模型
        3. 启用 torchax 并注册算子
        4. 在 with mesh: 块内配置 Pipeline
        """
        import torchax
        
        if (Wan21TPUSampler._cached_pipe is not None and
            Wan21TPUSampler._cached_model_id == model_id):
            print("  Using cached pipeline")
            return Wan21TPUSampler._cached_pipe, Wan21TPUSampler._mesh, Wan21TPUSampler._env
        
        print(f"  Loading Wan 2.1 Pipeline from {model_id}...")
        
        # ===== 步骤 1：创建 mesh =====
        dp_dim = 2
        tp_dim = len(jax.devices()) // dp_dim
        mesh_devices = mesh_utils.create_device_mesh(
            (dp_dim, tp_dim), allow_split_physical_axes=True
        )
        mesh = Mesh(mesh_devices, ("dp", "tp"))
        print(f"Mesh: {mesh}")
        print(f"  dp_dim={dp_dim}, tp_dim={tp_dim}")
        print(f"  总设备数: {len(jax.devices())}")
        
        # ===== 步骤 2：禁用 torchax 加载模型（避免拦截 transformers 加载逻辑）=====
        global _globally_enabled
        if _globally_enabled:
            torchax.disable_globally()
            _globally_enabled = False
        
        # 设置 default dtype
        torch.set_default_dtype(torch.bfloat16)
        
        from diffusers.pipelines.wan.pipeline_wan_torchax import WanPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        
        scheduler = UniPCMultistepScheduler(
            prediction_type='flow_prediction',
            use_flow_sigmas=True,
            num_train_timesteps=1000,
            flow_shift=flow_shift
        )
        
        pipe = WanPipeline.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, use_safetensors=True
        )
        pipe.scheduler = scheduler
        print("  ✓ 模型加载完成")
        
        # ===== 步骤 3：启用 torchax 并注册算子 =====
        env = ensure_torchax_enabled(mesh)
        
        # ===== 步骤 4：在 with mesh: 块内配置 Pipeline =====
        with mesh:
            pipe = _setup_pipeline_for_transformer_only(pipe, mesh, env)
        
        gc.collect()
        Wan21TPUSampler._cached_pipe = pipe
        Wan21TPUSampler._cached_model_id = model_id
        Wan21TPUSampler._env = env
        Wan21TPUSampler._mesh = mesh
        print("  Pipeline ready!")
        return pipe, mesh, env


# ============================================================================
# Wan 2.1 TPU VAE Decoder - Hybrid 方案
# ============================================================================

class Wan21TPUVAEDecoder:
    """
    Wan 2.1 VAE Decoder - 在 TPU 上解码 latents 为视频。
    
    Hybrid 方案：
    - 使用 ensure_torchax_enabled() 保持 Mode 栈激活
    - 模型缓存后权重保持 XLA 状态
    - 返回值转为 CPU tensor（确保与 ComfyUI 兼容）
    
    输入: latents (来自 Sampler)
    输出: video frames (ComfyUI IMAGE 格式)
    """
    
    _cached_vae = None
    _cached_model_id = None
    _env = None  # 缓存 env 对象
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latents": ("LATENT",),
            },
            "optional": {
                "model_id": ("STRING", {"default": "Wan-AI/Wan2.1-T2V-14B-Diffusers"}),
                "fps": ("INT", {"default": DEFAULT_FPS, "min": 1, "max": 60}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("frames", "fps")
    FUNCTION = "decode"
    CATEGORY = "TPU/Wan2.1"
    
    def decode(self, latents, model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers", fps=DEFAULT_FPS):
        print(f"\n[Wan21TPUVAEDecoder] Starting VAE decode...")
        
        if isinstance(latents, dict):
            latent_tensor = latents["samples"]
            num_frames = latents.get("num_frames", DEFAULT_FRAMES)
        else:
            latent_tensor = latents
            num_frames = DEFAULT_FRAMES
        
        real_mesh = get_mesh()
        vae, env = self._get_or_create_vae(model_id, real_mesh)
        
        start_time = time.perf_counter()
        
        # Hybrid 方案：enable_globally() 已激活，只需 with mesh: 用于 sharding context
        with real_mesh:
            # 处理 latents
            processed_latents = latent_tensor.to(vae.dtype)  # 转为 bfloat16
            processed_latents = env.to_xla(processed_latents)  # 转为 XLA
            processed_latents = self._denormalize_latents(processed_latents, vae, env)
            
            print("  Decoding...")
            # 根据配置决定 inference_mode
            # - USE_FULL_OPS_REGISTRATION=True: 所有算子已注册，可以使用 inference_mode(True)
            # - USE_FULL_OPS_REGISTRATION=False: 需要 fallback，必须用 inference_mode(False)
            use_inference_mode = USE_FULL_OPS_REGISTRATION
            with torch.inference_mode(use_inference_mode), torch.no_grad():
                video = vae.decode(processed_latents).sample
            jax.effects_barrier()
            
            # 转换回 CPU（返回给 ComfyUI）
            video_cpu = to_cpu_tensor(video)
        
        print(f"  VAE decode: {time.perf_counter() - start_time:.2f}s")
        
        # 后处理（在 CPU 上）
        frames = prepare_video_for_export(video_cpu, num_frames)
        frames_tensor = torch.from_numpy(frames)
        
        print(f"  Output: {frames_tensor.shape}")
        return (frames_tensor, fps)
    
    def _denormalize_latents(self, latents, vae, env):
        """
        Denormalize latents: x * std + mean
        
        注意：此时 latents 已经是 XLA tensor (bfloat16)
        必须在 with env: 块内调用！
        """
        latents_mean = getattr(vae.config, 'latents_mean', None)
        latents_std = getattr(vae.config, 'latents_std', None)
        
        if latents_mean is None or latents_std is None:
            return latents
        
        # 创建 mean 和 std tensor，转换为 XLA
        mean = torch.tensor(latents_mean, dtype=torch.bfloat16).view(1, 16, 1, 1, 1).to('jax')
        std = torch.tensor(latents_std, dtype=torch.bfloat16).view(1, 16, 1, 1, 1).to('jax')
        return latents * std + mean
    
    def _get_or_create_vae(self, model_id, mesh):
        """
        加载和配置 VAE（Hybrid 方案）
        
        流程：
        1. 临时禁用 torchax 加载模型
        2. 启用 torchax 并注册算子
        3. 在 with mesh: 块内：move_to_xla, compile, shard
        """
        import torchax
        
        if (Wan21TPUVAEDecoder._cached_vae is not None and
            Wan21TPUVAEDecoder._cached_model_id == model_id):
            print("  Using cached VAE")
            return Wan21TPUVAEDecoder._cached_vae, Wan21TPUVAEDecoder._env
        
        print(f"  Loading VAE from {model_id}...")
        
        # ===== 步骤 1：禁用 torchax 加载模型 =====
        global _globally_enabled
        if _globally_enabled:
            torchax.disable_globally()
            _globally_enabled = False
        
        from diffusers.models.autoencoders.autoencoder_kl_wan_torchax import AutoencoderKLWan
        vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.bfloat16
        )
        print("  ✓ VAE 加载完成")
        
        # ===== 步骤 2：启用 torchax 并注册算子 =====
        env = ensure_torchax_enabled(mesh)
        
        # 注册 conv2d 算子
        print("  - 注册 conv2d 算子...")
        from torchax.ops import jaten, ops_registry
        
        def conv2d_impl(input, weight, bias=None, stride=1, padding=0,
                        dilation=1, groups=1):
            jinput, jweight, jbias = env.t2j_iso((input, weight, bias))
            res = jaten._aten_conv2d(jinput, jweight, jbias, stride, padding, dilation, groups)
            return env.j2t_iso(res)
        
        env._ops[torch.nn.functional.conv2d] = ops_registry.Operator(
            torch.nn.functional.conv2d, conv2d_impl,
            is_jax_function=False, is_user_defined=True,
            needs_env=False, is_view_op=False,
        )
        
        # 注册 expand_as 算子（用于 F.normalize）
        print("  - 注册 expand_as 算子...")
        def expand_as_impl(input, other):
            jinput = env.t2j_iso(input)
            jother = env.t2j_iso(other)
            target_shape = jother.shape
            result = jnp.broadcast_to(jinput, target_shape)
            return env.j2t_iso(result)
        
        env._ops[torch.ops.aten.expand_as.default] = ops_registry.Operator(
            torch.ops.aten.expand_as.default, expand_as_impl,
            is_jax_function=False, is_user_defined=True,
            needs_env=False, is_view_op=False,
        )
        
        # ===== 步骤 3：在 with mesh: 块内设置 VAE Decoder =====
        with mesh:
            print("  - 将 VAE 移到 TPU...")
            move_module_to_xla(env, vae)
            
            print("  - 编译 VAE Decoder...")
            vae.decoder = torchax.compile(vae.decoder)
            
            num_devices = mesh.devices.size
            print(f"  - 复制权重到 {num_devices} TPU cores...")
            vae.decoder.params = shard_weight_dict(
                vae.decoder.params, VAE_DECODER_SHARDINGS, mesh
            )
            vae.decoder.buffers = shard_weight_dict(
                vae.decoder.buffers, VAE_DECODER_SHARDINGS, mesh
            )
        
        print("  ✓ VAE Decoder JIT 编译完成")
        
        gc.collect()
        Wan21TPUVAEDecoder._cached_vae = vae
        Wan21TPUVAEDecoder._cached_model_id = model_id
        Wan21TPUVAEDecoder._env = env
        print("  VAE ready!")
        return vae, env


# ============================================================================
# Wan 2.1 Full Pipeline
# ============================================================================

class Wan21TPUPipeline:
    """
    Wan 2.1 TPU Full Pipeline - 端到端视频生成。
    
    组合 TextEncoder -> Sampler -> VAEDecoder 三个阶段。
    
    重要：按照参考实现的三阶段设计：
    - 每个阶段只加载需要的组件
    - 阶段间清理内存
    - 这样可以在有限的 HBM 上运行 14B 模型
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True,
                    "default": "A cat and a dog baking a cake together in a kitchen."}),
                "negative_prompt": ("STRING", {"multiline": True,
                    "default": "Bright tones, overexposed, static, blurred details, low quality"}),
                "height": ("INT", {"default": DEFAULT_HEIGHT, "min": 256, "max": 1280, "step": 16}),
                "width": ("INT", {"default": DEFAULT_WIDTH, "min": 256, "max": 1280, "step": 16}),
                "num_frames": ("INT", {"default": DEFAULT_FRAMES, "min": 17, "max": 121, "step": 4}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 2025, "min": 0, "max": 2**32 - 1}),
            },
            "optional": {
                "model_id": ("STRING", {"default": "Wan-AI/Wan2.1-T2V-14B-Diffusers"}),
                "fps": ("INT", {"default": DEFAULT_FPS, "min": 1, "max": 60}),
                "flow_shift": ("FLOAT", {"default": DEFAULT_FLOW_SHIFT, "min": 1.0, "max": 10.0, "step": 0.5}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("frames", "fps")
    FUNCTION = "generate"
    CATEGORY = "TPU/Wan2.1"
    
    def generate(self, prompt, negative_prompt, height, width, num_frames,
                 num_inference_steps, guidance_scale, seed,
                 model_id="Wan-AI/Wan2.1-T2V-14B-Diffusers",
                 fps=DEFAULT_FPS, flow_shift=DEFAULT_FLOW_SHIFT):
        
        print(f"\n{'='*60}")
        print("Wan 2.1 TPU Full Pipeline")
        print(f"{'='*60}")
        
        # Stage 1: Text Encoding (TPU)
        prompt_embeds, negative_prompt_embeds = Wan21TextEncoder().encode(
            prompt, negative_prompt, model_id
        )
        
        # 清理 Stage 1 的缓存，释放 HBM
        print("\n[Pipeline] 清理 Text Encoder 以释放 HBM...")
        Wan21TextEncoder._cached_pipe = None
        Wan21TextEncoder._cached_model_id = None
        Wan21TextEncoder._is_compiled = False
        gc.collect()
        
        # Stage 2: Denoising (TPU)
        latents, _ = Wan21TPUSampler().sample(
            prompt_embeds, negative_prompt_embeds,
            height, width, num_frames,
            num_inference_steps, guidance_scale, seed,
            model_id, flow_shift
        )
        
        # 清理 Stage 2 的缓存，释放 HBM
        print("\n[Pipeline] 清理 Transformer 以释放 HBM...")
        Wan21TPUSampler._cached_pipe = None
        Wan21TPUSampler._cached_model_id = None
        gc.collect()
        
        # Stage 3: VAE Decoding (TPU)
        frames, fps_out = Wan21TPUVAEDecoder().decode(latents, model_id, fps)
        
        print(f"\n{'='*60}")
        print("Generation complete!")
        print(f"{'='*60}")
        
        return (frames, fps_out)


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "Wan21TextEncoder": Wan21TextEncoder,
    "Wan21TPUSampler": Wan21TPUSampler,
    "Wan21TPUVAEDecoder": Wan21TPUVAEDecoder,
    "Wan21TPUPipeline": Wan21TPUPipeline,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan21TextEncoder": "Wan 2.1 Text Encoder (TPU)",
    "Wan21TPUSampler": "Wan 2.1 TPU Sampler",
    "Wan21TPUVAEDecoder": "Wan 2.1 TPU VAE Decoder",
    "Wan21TPUPipeline": "Wan 2.1 TPU Full Pipeline",
}
