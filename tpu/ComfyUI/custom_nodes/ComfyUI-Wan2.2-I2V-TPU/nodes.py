"""
ComfyUI Wan 2.2 I2V TPU Nodes
=============================

使用 diffusers 的 torchax 优化模型在 TPU 上运行 Wan 2.2 I2V 视频生成。
基于 gpu-tpu-pedia/tpu/Wan2.2/generate_diffusers_i2v_torchax_staged 实现。

Nodes:
  - Wan22I2VImageEncoder: TPU 上编码图像条件
  - Wan22I2VTextEncoder: TPU 上运行 UMT5-XXL 编码 prompt
  - Wan22I2VTPUSampler: TPU 上运行双 Transformer 去噪
  - Wan22I2VTPUVAEDecoder: TPU 上运行 VAE 解码 latents 为视频

核心设计（Hybrid 方案）：
  - 使用 `enable_globally()` 保持 Mode 栈激活
  - 模型缓存后权重保持 XLA 状态
  - 节点返回值必须转为 CPU tensor
  - 双 Transformer 架构：boundary_ratio=0.9 切换模型
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
    class ProgressBar:
        def __init__(self, total, node_id=None):
            self.total = total
            self.current = 0
        def update(self, n=1):
            self.current += n
        def update_absolute(self, value, total=None, preview=None):
            self.current = value
    PromptServer = None
    _HAS_SERVER = False

from .utils import (
    MODEL_ID,
    TEXT_ENCODER_SHARDINGS,
    TRANSFORMER_SHARDINGS,
    VAE_ENCODER_SHARDINGS,
    VAE_DECODER_SHARDINGS,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_FRAMES,
    DEFAULT_FPS,
    NUM_STEPS,
    GUIDANCE_SCALE,
    BOUNDARY_RATIO,
    SHIFT,
    VAE_SCALE_FACTOR_TEMPORAL,
    VAE_SCALE_FACTOR_SPATIAL,
    move_module_to_xla,
    prepare_video_for_export,
    shard_weight_dict,
    setup_jax_cache,
    setup_pytree_registrations,
    register_text_encoder_ops,
    register_conv_ops,
    register_expand_as_op,
    normalize_latents,
    denormalize_latents,
)

# 全局 mesh（延迟创建）
_mesh = None


def get_mesh():
    """获取全局 mesh，如果不存在则创建"""
    global _mesh
    if _mesh is None:
        print("[Wan22I2V] Creating 2D Mesh for TPU...")
        devices = jax.devices('tpu')
        dp_dim = min(2, len(devices))
        tp_dim = len(devices) // dp_dim
        mesh_devices = mesh_utils.create_device_mesh(
            (dp_dim, tp_dim), allow_split_physical_axes=True
        )
        _mesh = Mesh(mesh_devices, ("dp", "tp"))
        print(f"[Wan22I2V] Created Mesh: dp={dp_dim}, tp={tp_dim}")
    return _mesh


# ============================================================================
# 公共工具函数
# ============================================================================

def to_cpu_tensor(tensor):
    """
    将 XLA tensor 安全转换为 CPU tensor。
    """
    if hasattr(tensor, '_elem'):
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
# Torchax 环境管理
# ============================================================================

_torchax_env = None
_globally_enabled = False
_jax_cache_initialized = False


def ensure_torchax_enabled(mesh_obj=None):
    """确保 torchax 全局启用，返回 env。"""
    global _torchax_env, _globally_enabled, _jax_cache_initialized
    import torchax
    
    if not _jax_cache_initialized:
        setup_jax_cache()
        _jax_cache_initialized = True
    
    if not _globally_enabled:
        print("[Torchax] Enabling globally (Hybrid mode)...")
        torchax.enable_globally()
        _globally_enabled = True
    
    if _torchax_env is None:
        _torchax_env = torchax.default_env()
    
    return _torchax_env


def disable_torchax_temporarily():
    """临时禁用 torchax（用于加载模型）"""
    global _globally_enabled
    import torchax
    if _globally_enabled:
        torchax.disable_globally()
        _globally_enabled = False


# ============================================================================
# TPU 模型清理函数
# ============================================================================

def cleanup_wan22_i2v_tpu_models():
    """
    清理所有 Wan 2.2 I2V TPU 缓存的模型和资源。
    
    当用户点击 ComfyUI Manager 的 "Unload Models" 按钮时调用此函数，
    释放 TPU 内存以便加载其他模型。
    """
    global _mesh, _torchax_env, _globally_enabled
    
    print("[Wan22I2V-TPU] Cleaning up cached models...")
    
    cleaned = []
    
    # 清理 Image Encoder
    if Wan22I2VImageEncoder._cached_pipe is not None:
        Wan22I2VImageEncoder._cached_pipe = None
        Wan22I2VImageEncoder._cached_model_id = None
        Wan22I2VImageEncoder._env = None
        cleaned.append("ImageEncoder")
    
    # 清理 Text Encoder
    if Wan22I2VTextEncoder._cached_pipe is not None:
        Wan22I2VTextEncoder._cached_pipe = None
        Wan22I2VTextEncoder._cached_model_id = None
        Wan22I2VTextEncoder._is_compiled = False
        Wan22I2VTextEncoder._env = None
        cleaned.append("TextEncoder")
    
    # 清理 Sampler (双 Transformer)
    if Wan22I2VTPUSampler._cached_transformers is not None:
        Wan22I2VTPUSampler._cached_transformers = None
        Wan22I2VTPUSampler._cached_model_id = None
        Wan22I2VTPUSampler._scheduler = None
        Wan22I2VTPUSampler._env = None
        Wan22I2VTPUSampler._mesh = None
        cleaned.append("Sampler (Transformers)")
    
    # 清理 VAE Decoder
    if Wan22I2VTPUVAEDecoder._cached_vae is not None:
        Wan22I2VTPUVAEDecoder._cached_vae = None
        Wan22I2VTPUVAEDecoder._cached_model_id = None
        Wan22I2VTPUVAEDecoder._env = None
        cleaned.append("VAEDecoder")
    
    # 清理全局 mesh
    if _mesh is not None:
        _mesh = None
        cleaned.append("Mesh")
    
    # 清理 torchax 状态
    if _torchax_env is not None:
        _torchax_env = None
    
    if _globally_enabled:
        try:
            import torchax
            torchax.disable_globally()
            _globally_enabled = False
            cleaned.append("Torchax")
        except Exception:
            pass
    
    # 强制垃圾回收和 JAX 缓存清理
    gc.collect()
    try:
        jax.clear_caches()
        cleaned.append("JAX caches")
    except Exception:
        pass
    
    if cleaned:
        print(f"[Wan22I2V-TPU] Cleaned: {', '.join(cleaned)}")
    print("[Wan22I2V-TPU] Cleanup complete!")


# ============================================================================
# Wan22I2VImageEncoder - 图像条件编码
# ============================================================================

class Wan22I2VImageEncoder:
    """
    Wan 2.2 I2V Image Encoder - 在 TPU 上编码图像为 latent condition。
    
    实现 A14B 模式：
    1. 预处理图像
    2. 构建 video_condition [image, zeros...]
    3. VAE encode
    4. 归一化 latents
    5. 构建 mask
    6. 拼接 condition = [mask, latent_condition]
    """
    
    _cached_pipe = None
    _cached_model_id = None
    _env = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "height": ("INT", {"default": DEFAULT_HEIGHT, "min": 256, "max": 1280, "step": 16}),
                "width": ("INT", {"default": DEFAULT_WIDTH, "min": 256, "max": 1280, "step": 16}),
                "num_frames": ("INT", {"default": DEFAULT_FRAMES, "min": 17, "max": 121, "step": 4}),
            },
            "optional": {
                "model_id": ("STRING", {"default": MODEL_ID}),
            }
        }
    
    RETURN_TYPES = ("CONDITION", "LATENT_INFO")
    RETURN_NAMES = ("condition", "latent_info")
    FUNCTION = "encode"
    CATEGORY = "TPU/Wan2.2-I2V"
    
    def encode(self, image, height, width, num_frames, model_id=MODEL_ID):
        print(f"\n[Wan22I2VImageEncoder] Encoding image condition...")
        print(f"  Target resolution: {width}x{height}, Frames: {num_frames}")
        
        real_mesh = get_mesh()
        pipe, env = self._get_or_create_pipeline(model_id, real_mesh)
        
        # ComfyUI IMAGE 格式: [T, H, W, C] float32 范围 [0, 1]
        # 取第一帧作为输入图像
        if image.dim() == 4:
            input_image = image[0]  # [H, W, C]
        else:
            input_image = image
        
        # 转换为 PIL Image
        from PIL import Image
        img_np = (input_image.cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)
        
        # 调整大小
        pil_image = pil_image.resize((width, height))
        print(f"  Image resized to: {pil_image.size}")
        
        with real_mesh:
            # 所有 TPU 操作都需要在 inference_mode(False) + no_grad 内
            with torch.inference_mode(False), torch.no_grad():
                # 预处理图像
                print("  Preprocessing image...")
                image_tensor = pipe.video_processor.preprocess(pil_image, height=height, width=width)
                # image_tensor: [1, 3, H, W]
                
                # 构建 video_condition [B, 3, num_frames, H, W]
                print("  Building video_condition (A14B mode)...")
                batch_size = 1
                image_tensor = image_tensor.unsqueeze(2).to('jax', dtype=torch.bfloat16)
                zeros = torch.zeros(
                    batch_size, 3, num_frames - 1, height, width,
                    device='jax', dtype=torch.bfloat16
                )
                video_condition = torch.cat([image_tensor, zeros], dim=2)
                print(f"  video_condition shape: {video_condition.shape}")
                
                # VAE 编码
                print("  VAE encoding...")
                latent_dist = pipe.vae.encode(video_condition)
                latent_condition = latent_dist.latent_dist.mode()
                
                print(f"  latent_condition shape: {latent_condition.shape}")
                
                # 归一化
                print("  Normalizing latents...")
                latent_condition = normalize_latents(latent_condition, vae=pipe.vae)
                
                # 构建 mask
                print("  Building mask...")
                num_latent_frames = (num_frames - 1) // VAE_SCALE_FACTOR_TEMPORAL + 1
                latent_height = height // VAE_SCALE_FACTOR_SPATIAL
                latent_width = width // VAE_SCALE_FACTOR_SPATIAL
                
                mask_lat_size = torch.ones(
                    batch_size, 1, num_frames, latent_height, latent_width,
                    device='jax', dtype=torch.bfloat16
                )
                mask_lat_size[:, :, list(range(1, num_frames))] = 0
                
                first_frame_mask = mask_lat_size[:, :, 0:1]
                first_frame_mask = torch.repeat_interleave(
                    first_frame_mask, dim=2, repeats=VAE_SCALE_FACTOR_TEMPORAL
                )
                mask_lat_size = torch.cat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
                mask_lat_size = mask_lat_size.view(
                    batch_size, -1, VAE_SCALE_FACTOR_TEMPORAL, latent_height, latent_width
                )
                mask_lat_size = mask_lat_size.transpose(1, 2)
                
                print(f"  mask shape: {mask_lat_size.shape}")
                
                # 拼接 condition [B, 20, T_latent, H_latent, W_latent]
                condition = torch.cat([mask_lat_size, latent_condition], dim=1)
                print(f"  condition shape: {condition.shape}")
            
            # 转换回 CPU
            condition_cpu = to_cpu_tensor(condition)
        
        latent_info = {
            "num_latent_frames": num_latent_frames,
            "latent_height": latent_height,
            "latent_width": latent_width,
            "num_frames": num_frames,
            "height": height,
            "width": width,
        }
        
        print(f"  ✓ Image encoding complete")
        return (condition_cpu, latent_info)
    
    @classmethod
    def _get_or_create_pipeline(cls, model_id, mesh):
        """加载和配置 Pipeline"""
        import torchax
        
        if (cls._cached_pipe is not None and
            cls._cached_model_id == model_id):
            print("  Using cached pipeline")
            return cls._cached_pipe, cls._env
        
        print(f"  Loading Wan 2.2 I2V Pipeline from {model_id}...")
        
        setup_pytree_registrations()
        
        # 确保 torchax 完全禁用，避免干扰 safetensors 加载
        disable_torchax_temporarily()
        
        # 保存当前默认设备和类型，设置为 CPU 加载
        original_device = torch.get_default_device() if hasattr(torch, 'get_default_device') else None
        try:
            torch.set_default_device('cpu')
        except Exception:
            pass  # 老版本 PyTorch 不支持
        
        from diffusers.pipelines.wan.pipeline_wan_i2v_torchax import WanImageToVideoPipeline
        from diffusers.models.autoencoders.autoencoder_kl_wan_torchax import AutoencoderKLWan
        
        torch.set_default_dtype(torch.bfloat16)
        
        # 显式指定 device_map 为 CPU 以避免 torchax 干扰
        print("  Loading VAE on CPU (avoiding torchax interference)...")
        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        
        print("  Loading Pipeline on CPU...")
        pipe = WanImageToVideoPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            vae=vae,
            device_map="cpu",
        )
        print("  ✓ Pipeline loaded on CPU")
        
        # 恢复默认设备
        if original_device is not None:
            try:
                torch.set_default_device(original_device)
            except Exception:
                pass
        
        env = ensure_torchax_enabled(mesh)
        register_conv_ops(env)
        register_expand_as_op(env)
        
        with mesh:
            print("  - Moving VAE Encoder to TPU...")
            move_module_to_xla(env, pipe.vae)
            pipe.vae.encoder = torchax.compile(pipe.vae.encoder)
            pipe.vae.encoder.params = shard_weight_dict(
                pipe.vae.encoder.params, VAE_ENCODER_SHARDINGS, mesh
            )
            pipe.vae.encoder.buffers = shard_weight_dict(
                pipe.vae.encoder.buffers, VAE_ENCODER_SHARDINGS, mesh
            )
        
        # 删除不需要的组件
        if hasattr(pipe, 'transformer') and pipe.transformer is not None:
            del pipe.transformer
            pipe.transformer = None
        if hasattr(pipe, 'transformer_2') and pipe.transformer_2 is not None:
            del pipe.transformer_2
            pipe.transformer_2 = None
        if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
            del pipe.text_encoder
            pipe.text_encoder = None
        
        gc.collect()
        cls._cached_pipe = pipe
        cls._cached_model_id = model_id
        cls._env = env
        print("  ✓ Image Encoder ready!")
        return pipe, env


# ============================================================================
# Wan22I2VTextEncoder - 文本编码
# ============================================================================

class Wan22I2VTextEncoder:
    """
    Wan 2.2 I2V Text Encoder - 在 TPU 上运行 UMT5-XXL 编码 prompt。
    """
    
    _cached_pipe = None
    _cached_model_id = None
    _is_compiled = False
    _env = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True,
                    "default": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."}),
                "negative_prompt": ("STRING", {"multiline": True,
                    "default": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作"}),
            },
            "optional": {
                "model_id": ("STRING", {"default": MODEL_ID}),
            }
        }
    
    RETURN_TYPES = ("TENSOR", "TENSOR")
    RETURN_NAMES = ("prompt_embeds", "negative_prompt_embeds")
    FUNCTION = "encode"
    CATEGORY = "TPU/Wan2.2-I2V"
    
    def encode(self, prompt, negative_prompt, model_id=MODEL_ID):
        print(f"\n[Wan22I2VTextEncoder] Encoding prompt on TPU...")
        print(f"  Prompt: {prompt[:80]}...")
        
        real_mesh = get_mesh()
        pipe, env = self._get_or_create_pipeline(model_id, real_mesh)
        
        print("  Encoding prompts...")
        with real_mesh:
            with torch.inference_mode(False), torch.no_grad():
                prompt_embeds, _ = pipe.encode_prompt(
                    prompt=prompt,
                    negative_prompt=None,
                    do_classifier_free_guidance=False,
                    num_videos_per_prompt=1,
                    device='jax',
                    dtype=torch.bfloat16,
                )
                
                negative_prompt_embeds, _ = pipe.encode_prompt(
                    prompt=negative_prompt,
                    negative_prompt=None,
                    do_classifier_free_guidance=False,
                    num_videos_per_prompt=1,
                    device='jax',
                    dtype=torch.bfloat16,
                )
            
            prompt_embeds_cpu = to_cpu_tensor(prompt_embeds)
            negative_prompt_embeds_cpu = to_cpu_tensor(negative_prompt_embeds)
        
        print(f"  prompt_embeds shape: {prompt_embeds_cpu.shape}")
        return (prompt_embeds_cpu, negative_prompt_embeds_cpu)
    
    @classmethod
    def _get_or_create_pipeline(cls, model_id, mesh):
        """加载和配置 Text Encoder Pipeline"""
        import torchax
        
        if (cls._cached_pipe is not None and
            cls._cached_model_id == model_id and
            cls._is_compiled):
            print("  Using cached pipeline")
            return cls._cached_pipe, cls._env
        
        print(f"  Loading Wan 2.2 I2V Pipeline from {model_id}...")
        
        setup_pytree_registrations()
        
        # 确保 torchax 完全禁用，避免干扰 safetensors 加载
        disable_torchax_temporarily()
        
        # 保存当前默认设备，设置为 CPU 加载
        original_device = torch.get_default_device() if hasattr(torch, 'get_default_device') else None
        try:
            torch.set_default_device('cpu')
        except Exception:
            pass
        
        from diffusers.pipelines.wan.pipeline_wan_i2v_torchax import WanImageToVideoPipeline
        from diffusers.models.autoencoders.autoencoder_kl_wan_torchax import AutoencoderKLWan
        
        torch.set_default_dtype(torch.bfloat16)
        
        # 显式指定 device_map 为 CPU 以避免 torchax 干扰
        print("  Loading VAE on CPU (avoiding torchax interference)...")
        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        
        print("  Loading Pipeline on CPU...")
        pipe = WanImageToVideoPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            vae=vae,
            device_map="cpu",
        )
        print("  ✓ Pipeline loaded on CPU")
        
        # 恢复默认设备
        if original_device is not None:
            try:
                torch.set_default_device(original_device)
            except Exception:
                pass
        
        env = ensure_torchax_enabled(mesh)
        register_text_encoder_ops(env)
        register_conv_ops(env)
        
        with mesh:
            print("  - Moving Text Encoder to TPU...")
            move_module_to_xla(env, pipe.text_encoder)
            pipe.text_encoder = torchax.compile(pipe.text_encoder)
            pipe.text_encoder.params = shard_weight_dict(
                pipe.text_encoder.params, TEXT_ENCODER_SHARDINGS, mesh
            )
            pipe.text_encoder.buffers = shard_weight_dict(
                pipe.text_encoder.buffers, TEXT_ENCODER_SHARDINGS, mesh
            )
            
            torchax.interop.call_jax(jax.block_until_ready, pipe.text_encoder.params)
        
        print("  ✓ Text Encoder setup complete")
        
        # 删除不需要的组件
        if hasattr(pipe, 'transformer') and pipe.transformer is not None:
            del pipe.transformer
            pipe.transformer = None
        if hasattr(pipe, 'transformer_2') and pipe.transformer_2 is not None:
            del pipe.transformer_2
            pipe.transformer_2 = None
        if hasattr(pipe, 'vae') and pipe.vae is not None:
            del pipe.vae
            pipe.vae = None
        
        gc.collect()
        cls._cached_pipe = pipe
        cls._cached_model_id = model_id
        cls._is_compiled = True
        cls._env = env
        print("  ✓ Text Encoder ready!")
        return pipe, env


# ============================================================================
# Wan22I2VTPUSampler - 双 Transformer 去噪
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


class Wan22I2VTPUSampler:
    """
    Wan 2.2 I2V TPU Sampler - 使用双 Transformer 运行去噪循环。
    
    核心逻辑：
    - boundary_ratio = 0.9
    - t >= 900: 使用 transformer
    - t < 900: 使用 transformer_2
    - latent_model_input = concat(latents, condition)
    """
    
    _cached_transformers = None
    _cached_model_id = None
    _scheduler = None
    _env = None
    _mesh = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_embeds": ("TENSOR",),
                "negative_prompt_embeds": ("TENSOR",),
                "condition": ("CONDITION",),
                "latent_info": ("LATENT_INFO",),
                "num_inference_steps": ("INT", {"default": NUM_STEPS, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": GUIDANCE_SCALE, "min": 0.0, "max": 20.0, "step": 0.1}),
                "shift": ("FLOAT", {"default": SHIFT, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
            },
            "optional": {
                "model_id": ("STRING", {"default": MODEL_ID}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }
    
    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("latents", "num_frames")
    FUNCTION = "sample"
    CATEGORY = "TPU/Wan2.2-I2V"
    
    def sample(self, prompt_embeds, negative_prompt_embeds, condition, latent_info,
               num_inference_steps, guidance_scale, shift, seed,
               model_id=MODEL_ID, unique_id=None):
        print(f"\n[Wan22I2VTPUSampler] Starting TPU inference...")
        print(f"  Steps: {num_inference_steps}, Guidance: {guidance_scale}, Shift: {shift}, Seed: {seed}")
        print(f"  Latent info: {latent_info}")
        
        setup_pytree_registrations()
        
        transformers, scheduler, mesh, env = self._get_or_create_transformers(model_id)
        
        num_latent_frames = latent_info['num_latent_frames']
        latent_height = latent_info['latent_height']
        latent_width = latent_info['latent_width']
        num_frames = latent_info['num_frames']
        
        # 设置随机种子
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        # 初始化噪声 latents
        latent_shape = (1, 16, num_latent_frames, latent_height, latent_width)
        latents = torch.randn(latent_shape, generator=generator, dtype=torch.bfloat16)
        
        # 创建进度条
        pbar = ProgressBar(num_inference_steps)
        step_times = []
        
        print(f"\n=== 阶段2：Transformer 推理 ===")
        print(f"  Boundary ratio: {BOUNDARY_RATIO}")
        
        start_time = time.perf_counter()
        
        with mesh:
            # 转换为 XLA tensor
            latents = latents.to('jax')
            condition_xla = condition.to('jax')
            prompt_embeds_xla = prompt_embeds.to('jax')
            negative_prompt_embeds_xla = negative_prompt_embeds.to('jax')
            
            # 设置 scheduler（shift 参数调整采样时间步长分布）
            scheduler.set_shift(shift)
            scheduler.set_timesteps(num_inference_steps)
            timesteps = scheduler.timesteps
            
            boundary_timestep = BOUNDARY_RATIO * scheduler.config.num_train_timesteps
            
            transformer, transformer_2 = transformers
            
            with torch.inference_mode(False), torch.no_grad():
                for i, t in enumerate(timesteps):
                    step_start = time.perf_counter()
                    
                    # 选择模型
                    if t >= boundary_timestep:
                        current_model = transformer
                        model_name = 'T1'
                    else:
                        current_model = transformer_2
                        model_name = 'T2'
                    
                    # 构建 latent_model_input [B, 36, T_latent, H_latent, W_latent]
                    latent_model_input = torch.cat([latents, condition_xla], dim=1)
                    
                    # CFG: 复制输入
                    batch_input = torch.cat([latent_model_input, latent_model_input])
                    batch_embeds = torch.cat([prompt_embeds_xla, negative_prompt_embeds_xla])
                    
                    # Timestep
                    timestep = t.expand(2).to('jax')
                    
                    # Transformer forward
                    noise = current_model(
                        hidden_states=batch_input,
                        timestep=timestep,
                        encoder_hidden_states=batch_embeds,
                        return_dict=False
                    )[0]
                    
                    # CFG 计算
                    noise_pred = noise[0:1]
                    noise_uncond = noise[1:2]
                    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
                    
                    # Scheduler step
                    latents = scheduler.step(noise_pred, t, latents).prev_sample
                    
                    jax.effects_barrier()
                
                    step_time = time.perf_counter() - step_start
                    step_times.append(step_time)
                    pbar.update(1)
                    
                    if _HAS_SERVER and unique_id is not None and len(step_times) > 0:
                        avg_time = sum(step_times) / len(step_times)
                        remaining = num_inference_steps - i - 1
                        eta = avg_time * remaining
                        progress_text = f"Step {i+1}/{num_inference_steps} | {step_time:.2f}s | {model_name} | ETA: {eta:.1f}s"
                        try:
                            PromptServer.instance.send_progress_text(progress_text, unique_id)
                        except Exception:
                            pass
            
            # 转换回 CPU
            torch_latents = to_cpu_tensor(latents)
        
        elapsed = time.perf_counter() - start_time
        print(f"\n✓ Transformer inference complete: {elapsed:.2f}s")
        print(f"  Average step time: {elapsed/num_inference_steps:.2f}s")
        print(f"  Latents shape: {torch_latents.shape}")
        
        return ({"samples": torch_latents, "num_frames": num_frames}, num_frames)
    
    @classmethod
    def _get_or_create_transformers(cls, model_id):
        """加载和配置双 Transformer"""
        import torchax
        from torchax.ops import ops_registry
        
        if (cls._cached_transformers is not None and
            cls._cached_model_id == model_id):
            print("  Using cached transformers")
            return cls._cached_transformers, cls._scheduler, cls._mesh, cls._env
        
        print(f"  Loading Wan 2.2 I2V Transformers from {model_id}...")
        
        # 创建 mesh
        dp_dim = 2
        tp_dim = len(jax.devices()) // dp_dim
        mesh_devices = mesh_utils.create_device_mesh(
            (dp_dim, tp_dim), allow_split_physical_axes=True
        )
        mesh = Mesh(mesh_devices, ("dp", "tp"))
        print(f"  Mesh: dp={dp_dim}, tp={tp_dim}")
        
        disable_torchax_temporarily()
        
        from diffusers.models.transformers.transformer_wan_torchax import WanTransformer3DModel
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        
        torch.set_default_dtype(torch.bfloat16)
        
        transformer = WanTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        transformer_2 = WanTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer_2", torch_dtype=torch.bfloat16
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        print("  ✓ Transformers loaded")
        
        env = ensure_torchax_enabled(mesh)
        register_conv_ops(env)
        
        # 注册 Attention
        def override_op(op, impl):
            env._ops[op] = ops_registry.Operator(
                op, impl, is_jax_function=False, is_user_defined=True,
                needs_env=False, is_view_op=False,
            )
        
        override_op(
            torch.nn.functional.scaled_dot_product_attention,
            functools.partial(_scaled_dot_product_attention, env=env, mesh=mesh)
        )
        
        transformer_options = torchax.CompileOptions(
            jax_jit_kwargs={"static_argnames": ("return_dict",)}
        )
        
        with mesh:
            # Setup Transformer 1
            print("  - Setting up Transformer 1 (high noise)...")
            move_module_to_xla(env, transformer)
            transformer = torchax.compile(transformer, transformer_options)
            transformer.params = shard_weight_dict(
                transformer.params, TRANSFORMER_SHARDINGS, mesh
            )
            transformer.buffers = shard_weight_dict(
                transformer.buffers, TRANSFORMER_SHARDINGS, mesh
            )
            
            # Setup Transformer 2
            print("  - Setting up Transformer 2 (low noise)...")
            move_module_to_xla(env, transformer_2)
            transformer_2 = torchax.compile(transformer_2, transformer_options)
            transformer_2.params = shard_weight_dict(
                transformer_2.params, TRANSFORMER_SHARDINGS, mesh
            )
            transformer_2.buffers = shard_weight_dict(
                transformer_2.buffers, TRANSFORMER_SHARDINGS, mesh
            )
            
            torchax.interop.call_jax(jax.block_until_ready, transformer.params)
            torchax.interop.call_jax(jax.block_until_ready, transformer_2.params)
        
        # 移动 scheduler 参数
        for k, v in scheduler.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(scheduler, k, v.to('jax'))
        
        print("  ✓ Transformers ready!")
        
        gc.collect()
        cls._cached_transformers = (transformer, transformer_2)
        cls._cached_model_id = model_id
        cls._scheduler = scheduler
        cls._env = env
        cls._mesh = mesh
        return (transformer, transformer_2), scheduler, mesh, env


# ============================================================================
# Wan22I2VTPUVAEDecoder - VAE 解码
# ============================================================================

class Wan22I2VTPUVAEDecoder:
    """
    Wan 2.2 I2V VAE Decoder - 在 TPU 上解码 latents 为视频。
    """
    
    _cached_vae = None
    _cached_model_id = None
    _env = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latents": ("LATENT",),
            },
            "optional": {
                "model_id": ("STRING", {"default": MODEL_ID}),
                "fps": ("INT", {"default": DEFAULT_FPS, "min": 1, "max": 60}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("frames", "fps")
    FUNCTION = "decode"
    CATEGORY = "TPU/Wan2.2-I2V"
    
    def decode(self, latents, model_id=MODEL_ID, fps=DEFAULT_FPS):
        print(f"\n[Wan22I2VTPUVAEDecoder] Starting VAE decode...")
        
        if isinstance(latents, dict):
            latent_tensor = latents["samples"]
            num_frames = latents.get("num_frames", DEFAULT_FRAMES)
        else:
            latent_tensor = latents
            num_frames = DEFAULT_FRAMES
        
        print(f"  Input latents shape: {latent_tensor.shape}")
        
        real_mesh = get_mesh()
        vae, env = self._get_or_create_vae(model_id, real_mesh)
        
        start_time = time.perf_counter()
        
        with real_mesh:
            # 处理 latents
            processed_latents = latent_tensor.to(vae.dtype)
            processed_latents = env.to_xla(processed_latents)
            processed_latents = denormalize_latents(processed_latents, vae=vae)
            
            print("  Decoding...")
            with torch.inference_mode(False), torch.no_grad():
                video = vae.decode(processed_latents).sample
            jax.effects_barrier()
            
            video_cpu = to_cpu_tensor(video)
        
        decode_time = time.perf_counter() - start_time
        print(f"  ✓ VAE decode complete: {decode_time:.2f}s")
        
        # 后处理
        frames = prepare_video_for_export(video_cpu, num_frames)
        frames_tensor = torch.from_numpy(frames)
        
        print(f"  Output frames shape: {frames_tensor.shape}")
        return (frames_tensor, fps)
    
    @classmethod
    def _get_or_create_vae(cls, model_id, mesh):
        """加载和配置 VAE Decoder"""
        import torchax
        
        if (cls._cached_vae is not None and
            cls._cached_model_id == model_id):
            print("  Using cached VAE")
            return cls._cached_vae, cls._env
        
        print(f"  Loading VAE from {model_id}...")
        
        # 确保 torchax 完全禁用，避免干扰 safetensors 加载
        disable_torchax_temporarily()
        
        # 保存当前默认设备，设置为 CPU 加载
        original_device = torch.get_default_device() if hasattr(torch, 'get_default_device') else None
        try:
            torch.set_default_device('cpu')
        except Exception:
            pass
        
        from diffusers.models.autoencoders.autoencoder_kl_wan_torchax import AutoencoderKLWan
        
        # 显式指定 device_map 为 CPU 以避免 torchax 干扰
        print("  Loading VAE on CPU (avoiding torchax interference)...")
        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
        print("  ✓ VAE loaded on CPU")
        
        # 恢复默认设备
        if original_device is not None:
            try:
                torch.set_default_device(original_device)
            except Exception:
                pass
        
        env = ensure_torchax_enabled(mesh)
        register_conv_ops(env)
        register_expand_as_op(env)
        
        with mesh:
            print("  - Moving VAE Decoder to TPU...")
            move_module_to_xla(env, vae)
            vae.decoder = torchax.compile(vae.decoder)
            vae.decoder.params = shard_weight_dict(
                vae.decoder.params, VAE_DECODER_SHARDINGS, mesh
            )
            vae.decoder.buffers = shard_weight_dict(
                vae.decoder.buffers, VAE_DECODER_SHARDINGS, mesh
            )
        
        print("  ✓ VAE Decoder ready!")
        
        gc.collect()
        cls._cached_vae = vae
        cls._cached_model_id = model_id
        cls._env = env
        return vae, env


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "Wan22I2VImageEncoder": Wan22I2VImageEncoder,
    "Wan22I2VTextEncoder": Wan22I2VTextEncoder,
    "Wan22I2VTPUSampler": Wan22I2VTPUSampler,
    "Wan22I2VTPUVAEDecoder": Wan22I2VTPUVAEDecoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Wan22I2VImageEncoder": "Wan 2.2 I2V Image Encoder (TPU)",
    "Wan22I2VTextEncoder": "Wan 2.2 I2V Text Encoder (TPU)",
    "Wan22I2VTPUSampler": "Wan 2.2 I2V TPU Sampler",
    "Wan22I2VTPUVAEDecoder": "Wan 2.2 I2V TPU VAE Decoder",
}
