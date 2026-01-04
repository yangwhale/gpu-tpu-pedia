"""
CogVideoX TPU ComfyUI Nodes

Three-stage pipeline for CogVideoX video generation on TPU:
1. CogVideoXTextEncoder - Encode text prompt using T5
2. CogVideoXTPUSampler - Run transformer denoising on TPU with Splash Attention
3. CogVideoXTPUVAEDecoder - Decode latents to video using VAE

Based on gpu-tpu-pedia/tpu/CogVideoX/generate_diffusers_torchax_staged/
"""

import os
import sys
import time
import functools
import numpy as np

import torch

# ============================================================================
# Global State Management
# ============================================================================

_initialized = False
_torchax_env = None
_mesh = None
_pipeline = None
_vae = None
_pipeline_model_id = None
_vae_model_id = None


def _get_or_init_jax():
    """Initialize JAX and torchax (lazy loading)."""
    global _initialized, _torchax_env, _mesh
    
    if _initialized:
        return _torchax_env, _mesh
    
    print("[CogVideoX] Initializing JAX and torchax...")
    
    import jax
    from jax.sharding import Mesh
    from jax.experimental import mesh_utils
    import torchax
    
    from .utils import setup_jax_cache, setup_pytree_registrations, DEFAULT_DP
    
    # Setup JAX cache
    setup_jax_cache()
    
    # Setup PyTree registrations
    setup_pytree_registrations()
    
    # Set default dtype
    torch.set_default_dtype(torch.bfloat16)
    
    # Enable torchax globally
    torchax.enable_globally()
    _torchax_env = torchax.default_env()
    
    # Create mesh with DP=2 (recommended for CFG)
    num_devices = len(jax.devices())
    dp_dim = min(DEFAULT_DP, num_devices)
    tp_dim = num_devices // dp_dim
    
    print(f"[CogVideoX] JAX devices: {num_devices}, Mesh: dp={dp_dim}, tp={tp_dim}")
    
    mesh_devices = mesh_utils.create_device_mesh(
        (dp_dim, tp_dim), allow_split_physical_axes=True
    )
    _mesh = Mesh(mesh_devices, ("dp", "tp"))
    
    # Configure env
    _torchax_env._mesh = _mesh
    _torchax_env._initial_content.mesh = _mesh
    _torchax_env.config.use_tpu_splash_attention = True
    
    _initialized = True
    print("[CogVideoX] JAX/torchax initialized successfully")
    
    return _torchax_env, _mesh


# ============================================================================
# CogVideoXTextEncoder Node
# ============================================================================

class CogVideoXTextEncoder:
    """
    CogVideoX Text Encoder Node
    
    Encodes text prompts using T5 text encoder on TPU.
    """
    
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
        global _pipeline, _pipeline_model_id
        
        import jax
        import torchax
        from diffusers import CogVideoXPipeline
        
        from .utils import (
            TEXT_ENCODER_SHARDINGS,
            shard_weight_dict,
            move_module_to_xla,
        )
        
        env, mesh = _get_or_init_jax()
        
        print(f"\n[CogVideoX TextEncoder] Encoding prompt...")
        print(f"  Model: {model_id}")
        print(f"  Prompt: {prompt[:80]}..." if len(prompt) > 80 else f"  Prompt: {prompt}")
        
        # Load pipeline if needed (before torchax for model loading)
        if _pipeline is None or _pipeline_model_id != model_id:
            print("[CogVideoX] Loading CogVideoX pipeline...")
            
            # Temporarily disable torchax for model loading
            torchax.disable_globally()
            try:
                _pipeline = CogVideoXPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                )
            finally:
                torchax.enable_globally()
            
            _pipeline_model_id = model_id
            
            # Move Text Encoder to XLA and shard
            print("[CogVideoX] Moving Text Encoder to TPU...")
            move_module_to_xla(env, _pipeline.text_encoder)
            _pipeline.text_encoder = torchax.compile(_pipeline.text_encoder)
            
            with mesh:
                _pipeline.text_encoder.params = shard_weight_dict(
                    _pipeline.text_encoder.params, TEXT_ENCODER_SHARDINGS, mesh
                )
                _pipeline.text_encoder.buffers = shard_weight_dict(
                    _pipeline.text_encoder.buffers, TEXT_ENCODER_SHARDINGS, mesh
                )
            
            # Wait for sharding to complete
            torchax.interop.call_jax(jax.block_until_ready, _pipeline.text_encoder.params)
            print("[CogVideoX] Text Encoder ready")
        
        # Encode prompts
        print("[CogVideoX] Encoding text...")
        start_time = time.time()
        
        with mesh, env:
            prompt_embeds, negative_prompt_embeds = _pipeline.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                do_classifier_free_guidance=True,
                num_videos_per_prompt=1,
                max_sequence_length=226,  # CogVideoX default
                device='jax',
                dtype=torch.bfloat16,
            )
        
        elapsed = time.time() - start_time
        print(f"[CogVideoX] Text encoding completed in {elapsed:.2f}s")
        print(f"  prompt_embeds: {prompt_embeds.shape}")
        print(f"  negative_prompt_embeds: {negative_prompt_embeds.shape}")
        
        embeddings = {
            'prompt_embeds': prompt_embeds,
            'negative_prompt_embeds': negative_prompt_embeds,
            'model_id': model_id,
        }
        
        return (embeddings,)


# ============================================================================
# CogVideoXTPUSampler Node
# ============================================================================

class CogVideoXTPUSampler:
    """
    CogVideoX TPU Sampler Node
    
    Runs transformer denoising loop on TPU with Splash Attention.
    """
    
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
                "num_devices": ("INT", {"default": 8, "min": 1, "max": 8}),
            },
        }
    
    RETURN_TYPES = ("COGVIDEOX_LATENTS",)
    RETURN_NAMES = ("latents",)
    FUNCTION = "sample"
    CATEGORY = "CogVideoX-TPU"
    
    def sample(self, embeddings, height, width, num_frames, num_inference_steps,
               guidance_scale, seed, num_devices):
        
        import jax
        import jax.numpy as jnp
        import torchax
        from torchax.ops import ops_registry
        from diffusers import CogVideoXPipeline
        
        from .utils import (
            TRANSFORMER_SHARDINGS_TP,
            shard_weight_dict,
            move_module_to_xla,
            USE_K_SMOOTH,
        )
        from .splash_attention import scaled_dot_product_attention
        
        global _pipeline, _pipeline_model_id
        
        env, mesh = _get_or_init_jax()
        model_id = embeddings['model_id']
        
        print(f"\n[CogVideoX TPUSampler] Starting sampling...")
        print(f"  Resolution: {width}x{height}, Frames: {num_frames}")
        print(f"  Steps: {num_inference_steps}, CFG: {guidance_scale}, Seed: {seed}")
        
        # Load pipeline if needed
        if _pipeline is None or _pipeline_model_id != model_id:
            print("[CogVideoX] Loading pipeline...")
            torchax.disable_globally()
            try:
                _pipeline = CogVideoXPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                )
            finally:
                torchax.enable_globally()
            _pipeline_model_id = model_id
        
        # Register custom attention
        print("[CogVideoX] Registering Splash Attention...")
        custom_attention = functools.partial(
            scaled_dot_product_attention,
            env=env,
            use_k_smooth=USE_K_SMOOTH,
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
        
        # Setup Transformer for TPU
        print("[CogVideoX] Setting up Transformer for TPU...")
        move_module_to_xla(env, _pipeline.transformer)
        
        # Move scheduler parameters to JAX
        for k, v in _pipeline.scheduler.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(_pipeline.scheduler, k, v.to('jax'))
        
        # Compile Transformer
        options = torchax.CompileOptions(
            jax_jit_kwargs={'static_argnames': ('return_dict',)}
        )
        _pipeline.transformer = torchax.compile(_pipeline.transformer, options)
        
        # Shard Transformer weights
        with mesh:
            _pipeline.transformer.params = shard_weight_dict(
                _pipeline.transformer.params, TRANSFORMER_SHARDINGS_TP, mesh
            )
            _pipeline.transformer.buffers = shard_weight_dict(
                _pipeline.transformer.buffers, TRANSFORMER_SHARDINGS_TP, mesh
            )
        
        torchax.interop.call_jax(jax.block_until_ready, _pipeline.transformer.params)
        
        # Delete VAE and Text Encoder to save memory
        if hasattr(_pipeline, 'vae') and _pipeline.vae is not None:
            del _pipeline.vae
            _pipeline.vae = None
        if hasattr(_pipeline, 'text_encoder') and _pipeline.text_encoder is not None:
            del _pipeline.text_encoder
            _pipeline.text_encoder = None
        
        # Convert embeddings to XLA
        prompt_embeds = embeddings['prompt_embeds']
        negative_prompt_embeds = embeddings['negative_prompt_embeds']
        
        if not hasattr(prompt_embeds, '_elem'):  # Not already on XLA
            with env:
                prompt_embeds = prompt_embeds.to('jax')
                negative_prompt_embeds = negative_prompt_embeds.to('jax')
        
        # Setup generator
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        # Run denoising
        print("[CogVideoX] Running denoising loop...")
        start_time = time.time()
        
        with mesh, env:
            result = _pipeline(
                prompt=None,
                negative_prompt=None,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type='latent',
            )
            jax.effects_barrier()
        
        elapsed = time.time() - start_time
        print(f"[CogVideoX] Denoising completed in {elapsed:.2f}s ({elapsed/num_inference_steps:.2f}s/step)")
        
        latents = result.frames  # output_type='latent' returns latents
        
        # Convert to torch tensor
        if hasattr(latents, '_elem'):
            jax_latents = latents._elem
            is_bf16 = (jax_latents.dtype == jnp.bfloat16)
            if is_bf16:
                np_latents = np.array(jax_latents.astype(jnp.float32))
                torch_latents = torch.from_numpy(np_latents).to(torch.bfloat16)
            else:
                np_latents = np.array(jax_latents)
                torch_latents = torch.from_numpy(np_latents)
        else:
            torch_latents = latents.cpu()
        
        print(f"  Raw latents shape: {torch_latents.shape} (format: [B, T, C, H, W])")
        
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
        
        return (latent_data,)


# ============================================================================
# CogVideoXTPUVAEDecoder Node
# ============================================================================

class CogVideoXTPUVAEDecoder:
    """
    CogVideoX TPU VAE Decoder Node
    
    Decodes latents to video frames using VAE on TPU.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latents": ("COGVIDEOX_LATENTS",),
                "fps": ("INT", {"default": 16, "min": 1, "max": 60}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "decode"
    CATEGORY = "CogVideoX-TPU"
    OUTPUT_IS_LIST = (True,)
    
    def decode(self, latents, fps):
        global _vae, _vae_model_id
        
        import jax
        import torchax
        from torchax.ops import ops_registry, jaten
        
        from diffusers.models.autoencoders.autoencoder_kl_cogvideox_torchax import AutoencoderKLCogVideoX
        
        from .utils import move_module_to_xla, prepare_video_for_export
        
        env, mesh = _get_or_init_jax()
        
        model_id = latents['model_id']
        torch_latents = latents['latents']
        num_frames = latents['num_frames']
        
        print(f"\n[CogVideoX VAEDecoder] Decoding latents...")
        print(f"  Latents shape: {torch_latents.shape}")
        print(f"  Model: {model_id}")
        print(f"  FPS: {fps}")
        
        # Load VAE if needed
        if _vae is None or _vae_model_id != model_id:
            print("[CogVideoX] Loading VAE...")
            
            torchax.disable_globally()
            try:
                _vae = AutoencoderKLCogVideoX.from_pretrained(
                    model_id, subfolder="vae", torch_dtype=torch.bfloat16
                )
            finally:
                torchax.enable_globally()
            
            _vae_model_id = model_id
            
            # Register conv2d for JAX
            def torch_conv2d_jax(input, weight, bias=None, stride=1, padding=0,
                                 dilation=1, groups=1, *, env=env):
                jinput, jweight, jbias = env.t2j_iso((input, weight, bias))
                res = jaten._aten_conv2d(jinput, jweight, jbias, stride, padding, dilation, groups)
                return env.j2t_iso(res)
            
            env._ops[torch.nn.functional.conv2d] = ops_registry.Operator(
                torch.nn.functional.conv2d,
                functools.partial(torch_conv2d_jax, env=env),
                is_jax_function=False,
                is_user_defined=True,
                needs_env=False,
                is_view_op=False,
            )
            
            # Move to XLA and compile
            move_module_to_xla(env, _vae)
            _vae.decoder = torchax.compile(_vae.decoder)
            print("[CogVideoX] VAE ready")
        
        # Handle nan values
        nan_count = torch.isnan(torch_latents).sum().item()
        if nan_count > 0:
            print(f"[CogVideoX] Warning: Found {nan_count} nan values, replacing with 0")
            torch_latents = torch.nan_to_num(torch_latents, nan=0.0)
        
        # Apply scaling factor
        scaling_factor = getattr(_vae.config, 'scaling_factor', 1.15258426)
        torch_latents = torch_latents.to(_vae.dtype) / scaling_factor
        
        # Move to XLA
        xla_latents = env.to_xla(torch_latents)
        
        # Warmup decode
        print("[CogVideoX] VAE warmup (JIT compile)...")
        with torch.no_grad():
            _vae.clear_cache()
            _ = _vae.decode(xla_latents).sample
        jax.effects_barrier()
        
        # Actual decode
        print("[CogVideoX] VAE decoding...")
        start_time = time.time()
        
        with torch.no_grad():
            _vae.clear_cache()
            video = _vae.decode(xla_latents).sample
        jax.effects_barrier()
        
        elapsed = time.time() - start_time
        print(f"[CogVideoX] VAE decoding completed in {elapsed:.2f}s")
        print(f"  Video shape: {video.shape}")
        
        # Convert to CPU and prepare for export
        video = video.to('cpu')
        frames = prepare_video_for_export(video, num_frames)
        
        print(f"[CogVideoX] Generated {len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]}")
        
        # Convert frames to ComfyUI IMAGE format
        # ComfyUI expects [B, H, W, C] tensors with values in [0, 1]
        comfy_frames = []
        for frame in frames:
            # frame is already [H, W, C] numpy array in [0, 1]
            frame_tensor = torch.from_numpy(frame).float()
            comfy_frames.append(frame_tensor.unsqueeze(0))  # Add batch dim
        
        return (comfy_frames,)


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
