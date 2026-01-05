"""
CogVideoX TPU ComfyUI Nodes

Three-stage pipeline for CogVideoX video generation on TPU:
1. CogVideoXTextEncoder - Encode text prompt using T5
2. CogVideoXTPUSampler - Run transformer denoising on TPU with Splash Attention
3. CogVideoXTPUVAEDecoder - Decode latents to video using VAE

Based on gpu-tpu-pedia/tpu/CogVideoX/generate_diffusers_torchax_staged/

Hybrid 方案（完全按照 Wan2.1 验证的架构）:
  - 使用 enable_globally() 保持 Mode 栈激活
  - 模型缓存后权重保持 XLA 状态
  - 节点返回值必须转为 CPU tensor（确保与 ComfyUI 兼容）
  - inference_mode(False) + torch.no_grad() 组合
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

# 从 utils 导入所有必要的函数和常量
from .utils import (
    TEXT_ENCODER_SHARDINGS,
    TRANSFORMER_SHARDINGS_TP,
    DEFAULT_DP,
    move_module_to_xla,
    prepare_video_for_export,
    shard_weight_dict,
    setup_jax_cache,
    setup_pytree_registrations,
    USE_K_SMOOTH,
)

# ============================================================================
# 全局 mesh（延迟创建）- 与 Wan2.1 一致
# ============================================================================

_mesh = None


def get_mesh():
    """获取全局 mesh，如果不存在则创建"""
    global _mesh
    if _mesh is None:
        print("[CogVideoX] Creating 2D Mesh for TPU...")
        devices = jax.devices('tpu')
        dp_dim = min(DEFAULT_DP, len(devices))
        tp_dim = len(devices) // dp_dim
        mesh_devices = mesh_utils.create_device_mesh(
            (dp_dim, tp_dim), allow_split_physical_axes=True
        )
        _mesh = Mesh(mesh_devices, ("dp", "tp"))
        print(f"[CogVideoX] Created Mesh: dp={dp_dim}, tp={tp_dim}")
    return _mesh


# ============================================================================
# 公共工具函数 - 与 Wan2.1 一致
# ============================================================================

def to_cpu_tensor(tensor):
    """
    将 XLA tensor 安全转换为 CPU tensor。
    
    处理逻辑：
    1. XLA tensor（有 _elem 属性）：转换 JAX array -> numpy -> torch
    2. bfloat16：先转为 float32 再转为 numpy（numpy 不支持 bfloat16）
    3. 普通 torch tensor：调用 .cpu()
    4. 其他类型：直接返回
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
# Torchax 环境管理（Hybrid 方案：enable_globally）- 与 Wan2.1 完全一致
# ============================================================================

_torchax_env = None
_globally_enabled = False
_jax_cache_initialized = False


def ensure_torchax_enabled(mesh_obj=None):
    """
    确保 torchax 全局启用，返回 env。
    
    Hybrid 方案（与 Wan2.1 完全一致）：
    - 首次调用时 enable_globally()，之后保持启用
    - 这样缓存的 XLA 模型权重可以在后续调用中正常工作
    - 节点返回值必须转为 CPU tensor（由各节点负责）
    - 初始化 JAX 编译缓存以加速后续编译
    """
    global _torchax_env, _globally_enabled, _jax_cache_initialized
    import torchax
    
    # 首次调用时设置 JAX 编译缓存
    if not _jax_cache_initialized:
        setup_jax_cache()
        _jax_cache_initialized = True
    
    # 首次调用时全局启用
    if not _globally_enabled:
        print("[CogVideoX Torchax] Enabling globally (Hybrid mode)...")
        torchax.enable_globally()
        _globally_enabled = True
    
    # 获取 env
    if _torchax_env is None:
        _torchax_env = torchax.default_env()
        # Configure env for TPU splash attention
        if mesh_obj is not None:
            _torchax_env._mesh = mesh_obj
            _torchax_env._initial_content.mesh = mesh_obj
        _torchax_env.config.use_tpu_splash_attention = True
    
    return _torchax_env


# ============================================================================
# TPU 模型清理函数
# ============================================================================

def cleanup_cogvideox_tpu_models():
    """
    清理所有 CogVideoX TPU 缓存的模型和资源。
    
    当用户点击 ComfyUI Manager 的 "Unload Models" 按钮时调用此函数，
    释放 TPU 内存以便加载其他模型。
    """
    global _mesh, _torchax_env, _globally_enabled
    
    print("[CogVideoX-TPU] Cleaning up cached models...")
    
    cleaned = []
    
    # 清理 Text Encoder
    if CogVideoXTextEncoder._cached_pipe is not None:
        CogVideoXTextEncoder._cached_pipe = None
        CogVideoXTextEncoder._cached_model_id = None
        CogVideoXTextEncoder._is_compiled = False
        CogVideoXTextEncoder._env = None
        cleaned.append("TextEncoder")
    
    # 清理 Sampler (Transformer)
    if CogVideoXTPUSampler._cached_pipe is not None:
        CogVideoXTPUSampler._cached_pipe = None
        CogVideoXTPUSampler._cached_model_id = None
        CogVideoXTPUSampler._env = None
        CogVideoXTPUSampler._mesh = None
        cleaned.append("Sampler (Transformer)")
    
    # 清理 VAE Decoder
    if CogVideoXTPUVAEDecoder._cached_vae is not None:
        CogVideoXTPUVAEDecoder._cached_vae = None
        CogVideoXTPUVAEDecoder._cached_model_id = None
        CogVideoXTPUVAEDecoder._env = None
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
        print(f"[CogVideoX-TPU] Cleaned: {', '.join(cleaned)}")
    print("[CogVideoX-TPU] Cleanup complete!")


# ============================================================================
# CogVideoXTextEncoder Node - 按照 Wan2.1 架构
# ============================================================================

class CogVideoXTextEncoder:
    """
    CogVideoX Text Encoder - 在 TPU 上运行 T5 编码 prompt。
    
    Hybrid 方案（与 Wan2.1 完全一致）：
    - 使用 ensure_torchax_enabled() 保持 Mode 栈激活
    - 模型缓存后权重保持 XLA 状态
    - 返回值转为 CPU tensor（确保与 ComfyUI 兼容）
    """
    
    _cached_pipe = None
    _cached_model_id = None
    _is_compiled = False
    _env = None  # 缓存 env 对象
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A panda, dressed in a small red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes."
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "model_id": ("STRING", {
                    "default": "zai-org/CogVideoX1.5-5B"
                }),
            },
        }
    
    RETURN_TYPES = ("COGVIDEOX_EMBEDS",)
    RETURN_NAMES = ("embeddings",)
    FUNCTION = "encode"
    CATEGORY = "CogVideoX-TPU"
    
    def encode(self, prompt, negative_prompt, model_id):
        print(f"\n[CogVideoX TextEncoder] Encoding prompt on TPU...")
        print(f"  Prompt: {prompt[:50]}...")
        
        real_mesh = get_mesh()
        pipe, env = self._get_or_create_pipeline(model_id, real_mesh)
        
        print("  Encoding prompts...")
        # Hybrid 方案：enable_globally() 已激活，只需 with mesh: 用于 sharding context
        with real_mesh:
            # 使用 inference_mode(False) + torch.no_grad() 组合（与 Wan2.1 一致）
            with torch.inference_mode(False), torch.no_grad():
                prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    do_classifier_free_guidance=True,
                    num_videos_per_prompt=1,
                    max_sequence_length=226,  # CogVideoX default
                    device='jax',
                    dtype=torch.bfloat16,
                )
            
            # 转换回 CPU（返回给 ComfyUI）
            prompt_embeds_cpu = to_cpu_tensor(prompt_embeds)
            negative_prompt_embeds_cpu = to_cpu_tensor(negative_prompt_embeds)
        
        print(f"  prompt_embeds shape: {prompt_embeds_cpu.shape}")
        
        embeddings = {
            'prompt_embeds': prompt_embeds_cpu,
            'negative_prompt_embeds': negative_prompt_embeds_cpu,
            'model_id': model_id,
        }
        
        return (embeddings,)
    
    @classmethod
    def _get_or_create_pipeline(cls, model_id, mesh):
        """
        加载和配置 Pipeline（Hybrid 方案 - 与 Wan2.1 完全一致）
        
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
        
        print(f"  Loading CogVideoX Pipeline from {model_id}...")
        
        # ===== 步骤 1：注册 PyTree =====
        print("  [DEBUG] Step 1: Registering PyTree...")
        setup_pytree_registrations()
        
        # ===== 步骤 2：加载模型（禁用 torchax 避免拦截 transformers 加载逻辑）=====
        global _globally_enabled
        if _globally_enabled:
            print("  [DEBUG] Step 2: Disabling torchax before model loading...")
            torchax.disable_globally()
            _globally_enabled = False
        
        from diffusers import CogVideoXPipeline
        torch.set_default_dtype(torch.bfloat16)
        
        print("  [DEBUG] Loading CogVideoXPipeline.from_pretrained...")
        pipe = CogVideoXPipeline.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, use_safetensors=True
        )
        print("  ✓ 模型加载完成")
        
        # ===== 步骤 3：启用 torchax =====
        print("  [DEBUG] Step 3: Enabling torchax...")
        env = ensure_torchax_enabled(mesh)
        
        # ===== 步骤 4：在 with mesh: 块内设置 Text Encoder =====
        print(f"  [DEBUG] Step 4: Setting up Text Encoder in mesh context...")
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
            print("  - Waiting for sharding to complete...")
            torchax.interop.call_jax(jax.block_until_ready, pipe.text_encoder.params)
        
        print("  ✓ Text Encoder 设置完成")
        
        # 删除不需要的组件
        print("  - Deleting transformer and VAE to save memory...")
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
# CogVideoX Splash Attention - CogVideoX 特有的 exp2 优化
# ============================================================================

def _scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                   is_causal=False, scale=None, enable_gqa=False,
                                   env=None, mesh=None):
    """封装 SDPA，使用 CogVideoX 优化的 Splash Attention。"""
    from .splash_attention import tpu_splash_attention_cogvideox
    
    # CogVideoX 的 attention 都使用 Splash Attention
    jquery, jkey, jvalue = env.t2j_iso((query, key, value))
    
    if USE_K_SMOOTH:
        key_mean = jnp.mean(jkey, axis=2, keepdims=True)
        jkey = jkey - key_mean
    
    res = tpu_splash_attention_cogvideox(jquery, jkey, jvalue, mesh, scale=scale)
    return env.j2t_iso(res)


# ============================================================================
# CogVideoXTPUSampler Node - 按照 Wan2.1 架构
# ============================================================================

class CogVideoXTPUSampler:
    """
    CogVideoX TPU Sampler - 在 TPU 上运行 Transformer 去噪。
    
    Hybrid 方案（与 Wan2.1 完全一致）：
    - 使用 ensure_torchax_enabled() 保持 Mode 栈激活
    - 模型缓存后权重保持 XLA 状态
    - 返回值转为 CPU tensor（确保与 ComfyUI 兼容）
    """
    
    _cached_pipe = None
    _cached_model_id = None
    _env = None  # 缓存 env 对象
    _mesh = None  # 缓存 mesh 对象
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "embeddings": ("COGVIDEOX_EMBEDS",),
                "height": ("INT", {"default": 720, "min": 256, "max": 1080, "step": 8}),
                "width": ("INT", {"default": 1280, "min": 256, "max": 1920, "step": 8}),
                "num_frames": ("INT", {"default": 81, "min": 17, "max": 161, "step": 8}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**32 - 1}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            }
        }
    
    RETURN_TYPES = ("COGVIDEOX_LATENTS", "STRING")
    RETURN_NAMES = ("latents", "info")
    FUNCTION = "sample"
    CATEGORY = "CogVideoX-TPU"
    
    def sample(self, embeddings, height, width, num_frames, num_inference_steps,
               guidance_scale, seed, unique_id=None, **kwargs):
        """
        运行 Transformer 推理生成 latents（Hybrid 方案 - 与 Wan2.1 完全一致）
        """
        model_id = embeddings['model_id']
        
        print(f"\n[CogVideoX TPUSampler] Starting TPU inference...")
        print(f"  Resolution: {width}x{height}, Frames: {num_frames}")
        print(f"  Steps: {num_inference_steps}, CFG: {guidance_scale}, Seed: {seed}")
        
        # 注册 PyTree
        setup_pytree_registrations()
        
        # 加载 Pipeline（如果需要）
        pipe, mesh, env = self._get_or_create_pipeline(model_id)
        
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        # 创建进度条和计时器
        pbar = ProgressBar(num_inference_steps)
        step_times = []
        last_step_time = [time.perf_counter()]  # 使用列表以便在闭包中修改
        
        # 定义 callback 函数
        def progress_callback(pipe, step_index, timestep, callback_kwargs):
            nonlocal step_times, last_step_time
            
            current_time = time.perf_counter()
            step_time = current_time - last_step_time[0]
            last_step_time[0] = current_time
            
            # 跳过第一步的时间（包含编译时间）
            if step_index > 0:
                step_times.append(step_time)
            
            pbar.update(1)
            
            # 发送进度信息到 ComfyUI
            if _HAS_SERVER and unique_id is not None and len(step_times) > 0:
                avg_time = sum(step_times) / len(step_times)
                remaining = num_inference_steps - step_index - 1
                eta = avg_time * remaining
                progress_text = f"Step {step_index + 1}/{num_inference_steps} | {step_time:.2f}s | ETA: {eta:.1f}s"
                print(f"  {progress_text}")
                try:
                    PromptServer.instance.send_progress_text(progress_text, unique_id)
                except Exception:
                    pass
            elif step_index > 0:
                print(f"  Step {step_index + 1}/{num_inference_steps} | {step_time:.2f}s")
            
            return callback_kwargs
        
        # 运行推理
        print(f"\n=== Transformer 推理 ===")
        start_time = time.perf_counter()
        
        # Hybrid 方案：enable_globally() 已激活，只需 with mesh: 用于 sharding context
        with mesh:
            # 将 embeddings 转换为 XLA tensor
            prompt_embeds_xla = embeddings['prompt_embeds'].to('jax')
            negative_prompt_embeds_xla = embeddings['negative_prompt_embeds'].to('jax')
            
            # 使用 inference_mode(False) + torch.no_grad() 组合（与 Wan2.1 一致）
            with torch.inference_mode(False), torch.no_grad():
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
                    callback_on_step_end=progress_callback,
                )
            jax.effects_barrier()
            
            # 转换 latents 为 CPU tensor（返回给 ComfyUI）
            torch_latents = to_cpu_tensor(result.frames)
        
        elapsed = time.perf_counter() - start_time
        avg_step_time = sum(step_times) / len(step_times) if step_times else elapsed / num_inference_steps
        print(f"\n✓ Transformer 推理完成，耗时: {elapsed:.2f} 秒")
        print(f"  平均每步时间: {avg_step_time:.2f}s (不含首步编译)")
        print(f"  Raw latents shape: {torch_latents.shape}")
        
        # Permute to [B, C, T, H, W] format (expected by decode_latents)
        torch_latents = torch_latents.permute(0, 2, 1, 3, 4)
        print(f"  Permuted latents shape: {torch_latents.shape} (format: [B, C, T, H, W])")
        
        # Trim CogVideoX-1.5 additional_frames (padding frames)
        vae_scale_factor_temporal = 4
        patch_size_t = 2
        
        latent_frames = (num_frames - 1) // vae_scale_factor_temporal + 1
        additional_frames = 0
        if latent_frames % patch_size_t != 0:
            additional_frames = patch_size_t - latent_frames % patch_size_t
        
        if additional_frames > 0:
            print(f"  Trimming additional_frames: {additional_frames}")
            torch_latents = torch_latents[:, :, additional_frames:, :, :]
            print(f"  Trimmed latents shape: {torch_latents.shape}")
        
        latent_data = {
            'latents': torch_latents,
            'model_id': model_id,
            'num_frames': num_frames,
        }
        
        # 构建 info 字符串
        info_text = f"Total: {elapsed:.2f}s | Step: {avg_step_time:.2f}s"
        
        return (latent_data, info_text)
    
    @classmethod
    def _get_or_create_pipeline(cls, model_id):
        """
        加载和配置 Pipeline（Hybrid 方案 - 与 Wan2.1 完全一致）
        
        流程：
        1. 创建 mesh
        2. 临时禁用 torchax 加载模型
        3. 启用 torchax 并注册算子
        4. 在 with mesh: 块内配置 Pipeline
        """
        import torchax
        from torchax.ops import ops_registry
        
        if (cls._cached_pipe is not None and
            cls._cached_model_id == model_id):
            print("  Using cached pipeline")
            return cls._cached_pipe, cls._mesh, cls._env
        
        print(f"  Loading CogVideoX Pipeline from {model_id}...")
        
        # ===== 步骤 1：创建 mesh =====
        dp_dim = DEFAULT_DP
        tp_dim = len(jax.devices()) // dp_dim
        mesh_devices = mesh_utils.create_device_mesh(
            (dp_dim, tp_dim), allow_split_physical_axes=True
        )
        mesh = Mesh(mesh_devices, ("dp", "tp"))
        print(f"  [DEBUG] Mesh: dp_dim={dp_dim}, tp_dim={tp_dim}")
        
        # ===== 步骤 2：禁用 torchax 加载模型（避免拦截 transformers 加载逻辑）=====
        global _globally_enabled
        if _globally_enabled:
            print("  [DEBUG] Disabling torchax before model loading...")
            torchax.disable_globally()
            _globally_enabled = False
        
        # 设置 default dtype
        torch.set_default_dtype(torch.bfloat16)
        
        from diffusers import CogVideoXPipeline
        
        print("  [DEBUG] Loading CogVideoXPipeline.from_pretrained...")
        pipe = CogVideoXPipeline.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, use_safetensors=True
        )
        print("  ✓ 模型加载完成")
        
        # ===== 步骤 3：启用 torchax 并注册算子 =====
        print("  [DEBUG] Enabling torchax...")
        env = ensure_torchax_enabled(mesh)
        
        # ===== 步骤 4：在 with mesh: 块内配置 Pipeline =====
        print("  [DEBUG] Setting up Transformer in mesh context...")
        with mesh:
            # Register custom attention
            print("  - 注册 Splash Attention...")
            custom_attention = functools.partial(
                _scaled_dot_product_attention,
                env=env,
                mesh=mesh,
            )
            op_to_override = torch.nn.functional.scaled_dot_product_attention
            env._ops[op_to_override] = ops_registry.Operator(
                op_to_override,
                custom_attention,
                is_jax_function=False,
                is_user_defined=True,
                needs_env=False,
                is_view_op=False,
            )
            
            # Move Transformer to XLA
            print("  - 将 Transformer 移到 TPU...")
            move_module_to_xla(env, pipe.transformer)
            
            # Move scheduler parameters to JAX
            for k, v in pipe.scheduler.__dict__.items():
                if isinstance(v, torch.Tensor):
                    setattr(pipe.scheduler, k, v.to('jax'))
            
            # Compile Transformer
            print("  - 编译 Transformer...")
            options = torchax.CompileOptions(
                jax_jit_kwargs={'static_argnames': ('return_dict',)}
            )
            pipe.transformer = torchax.compile(pipe.transformer, options)
            
            # Apply sharding
            print("  - 对 Transformer 进行权重分片...")
            pipe.transformer.params = shard_weight_dict(
                pipe.transformer.params, TRANSFORMER_SHARDINGS_TP, mesh
            )
            pipe.transformer.buffers = shard_weight_dict(
                pipe.transformer.buffers, TRANSFORMER_SHARDINGS_TP, mesh
            )
            
            # Wait for sharding to complete
            print("  - Waiting for sharding to complete...")
            torchax.interop.call_jax(jax.block_until_ready, pipe.transformer.params)
        
        print("  ✓ Transformer 配置完成")
        
        # 删除不需要的组件
        print("  - Deleting VAE and Text Encoder to save memory...")
        if hasattr(pipe, 'vae') and pipe.vae is not None:
            del pipe.vae
            pipe.vae = None
        if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
            del pipe.text_encoder
            pipe.text_encoder = None
        
        gc.collect()
        cls._cached_pipe = pipe
        cls._cached_model_id = model_id
        cls._env = env
        cls._mesh = mesh
        print("  Transformer ready!")
        return pipe, mesh, env


# ============================================================================
# CogVideoXTPUVAEDecoder Node - 按照 Wan2.1 架构
# ============================================================================

class CogVideoXTPUVAEDecoder:
    """
    CogVideoX VAE Decoder - 在 TPU 上解码 latents 为视频。
    
    Hybrid 方案（与 Wan2.1 完全一致）：
    - 使用 ensure_torchax_enabled() 保持 Mode 栈激活
    - 模型缓存后权重保持 XLA 状态
    - 返回值转为 CPU tensor（确保与 ComfyUI 兼容）
    """
    
    _cached_vae = None
    _cached_model_id = None
    _env = None  # 缓存 env 对象
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latents": ("COGVIDEOX_LATENTS",),
                "fps": ("INT", {"default": 16, "min": 1, "max": 60}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("frames", "fps")
    FUNCTION = "decode"
    CATEGORY = "CogVideoX-TPU"
    
    def decode(self, latents, fps):
        print(f"\n[CogVideoX VAEDecoder] Starting VAE decode...")
        
        model_id = latents['model_id']
        torch_latents = latents['latents']
        num_frames = latents['num_frames']
        
        print(f"  Latents shape: {torch_latents.shape}")
        print(f"  Model: {model_id}")
        
        real_mesh = get_mesh()
        vae, env = self._get_or_create_vae(model_id, real_mesh)
        
        start_time = time.perf_counter()
        
        # Hybrid 方案：enable_globally() 已激活，只需 with mesh: 用于 sharding context
        with real_mesh:
            # 处理 latents
            # Handle nan values
            nan_count = torch.isnan(torch_latents).sum().item()
            if nan_count > 0:
                print(f"  Warning: Found {nan_count} nan values, replacing with 0")
                torch_latents = torch.nan_to_num(torch_latents, nan=0.0)
            
            # Apply scaling factor
            scaling_factor = getattr(vae.config, 'scaling_factor', 1.15258426)
            processed_latents = torch_latents.to(vae.dtype) / scaling_factor
            processed_latents = processed_latents.to('jax')  # 转为 XLA
            
            print("  Decoding...")
            # 使用 inference_mode(False) + torch.no_grad() 组合（与 Wan2.1 一致）
            with torch.inference_mode(False), torch.no_grad():
                vae.clear_cache()
                video = vae.decode(processed_latents).sample
            jax.effects_barrier()
            
            # 转换回 CPU（返回给 ComfyUI）
            video_cpu = to_cpu_tensor(video)
        
        print(f"  VAE decode: {time.perf_counter() - start_time:.2f}s")
        
        # 后处理（在 CPU 上）
        frames_list = prepare_video_for_export(video_cpu, num_frames)
        
        # prepare_video_for_export 返回 list of [H, W, C] numpy arrays
        # 需要 stack 成 [T, H, W, C] 格式
        if isinstance(frames_list, list):
            frames = np.stack(frames_list, axis=0)
        else:
            frames = frames_list
        
        frames_tensor = torch.from_numpy(frames)
        
        print(f"  Output: {frames_tensor.shape}")
        return (frames_tensor, fps)
    
    @classmethod
    def _get_or_create_vae(cls, model_id, mesh):
        """
        加载和配置 VAE（Hybrid 方案 - 与 Wan2.1 完全一致）
        
        流程：
        1. 临时禁用 torchax 加载模型
        2. 启用 torchax 并注册算子
        3. 在 with mesh: 块内：move_to_xla, compile, shard
        """
        import torchax
        from torchax.ops import jaten, ops_registry
        
        if (cls._cached_vae is not None and
            cls._cached_model_id == model_id):
            print("  Using cached VAE")
            return cls._cached_vae, cls._env
        
        print(f"  Loading VAE from {model_id}...")
        
        # ===== 步骤 1：禁用 torchax 加载模型 =====
        global _globally_enabled
        if _globally_enabled:
            print("  [DEBUG] Disabling torchax before model loading...")
            torchax.disable_globally()
            _globally_enabled = False
        
        from diffusers.models.autoencoders.autoencoder_kl_cogvideox_torchax import AutoencoderKLCogVideoX
        
        print("  [DEBUG] Loading AutoencoderKLCogVideoX.from_pretrained...")
        vae = AutoencoderKLCogVideoX.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.bfloat16, use_safetensors=True
        )
        print("  ✓ VAE 加载完成")
        
        # ===== 步骤 2：启用 torchax 并注册算子 =====
        print("  [DEBUG] Enabling torchax...")
        env = ensure_torchax_enabled(mesh)
        
        # 注册 conv2d 算子
        print("  - 注册 conv2d 算子...")
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
        print("  [DEBUG] Setting up VAE in mesh context...")
        with mesh:
            print("  - 将 VAE 移到 TPU...")
            move_module_to_xla(env, vae)
            
            print("  - 编译 VAE Decoder...")
            vae.decoder = torchax.compile(vae.decoder)
        
        print("  ✓ VAE Decoder JIT 编译完成")
        
        gc.collect()
        cls._cached_vae = vae
        cls._cached_model_id = model_id
        cls._env = env
        print("  VAE ready!")
        return vae, env


# ============================================================================
# Node Mappings
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "CogVideoXTextEncoder": CogVideoXTextEncoder,
    "CogVideoXTPUSampler": CogVideoXTPUSampler,
    "CogVideoXTPUVAEDecoder": CogVideoXTPUVAEDecoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CogVideoXTextEncoder": "CogVideoX Text Encoder (TPU)",
    "CogVideoXTPUSampler": "CogVideoX TPU Sampler",
    "CogVideoXTPUVAEDecoder": "CogVideoX TPU VAE Decoder",
}
