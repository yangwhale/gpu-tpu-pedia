#!/usr/bin/env python3
"""
S3Diff Image Super-Resolution on TPU v6e (torchax)

Single-step diffusion-based 4x image upscaling using S3Diff model.
Runs on a single TPU chip (SD-Turbo 3.3B fits entirely on one chip).

Based on: https://github.com/ArcticHare105/S3Diff
Reference: SDXL torchax implementation in gpu-tpu-pedia/tpu/SDXL/

Usage:
    TPU_VISIBLE_DEVICES=0 python generate_torchax.py --input test_images/test_lr.png --output output_sr.png --warmup
"""

import os
import sys
import warnings
import logging

# Environment setup (must be before other imports)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')
for logger_name in ['root', '', 'diffusers', 'transformers']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

import argparse
import copy
import functools
import gc
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ============================================================================
# Configuration
# ============================================================================

SD_TURBO_ID = "stabilityai/sd-turbo"
S3DIFF_REPO = "zhangap/S3Diff"
S3DIFF_WEIGHTS = "s3diff.pkl"

DEFAULT_POS_PROMPT = "A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting."
DEFAULT_NEG_PROMPT = "oil painting, cartoon, blur, dirty, messy, low quality, deformation, low resolution, oversmooth"
GUIDANCE_SCALE = 1.07
SCALE_FACTOR = 4

# ============================================================================
# DEResNet (Degradation Estimation Network)
# ============================================================================

class ResidualBlockNoBN(nn.Module):
    """Residual block without BN (from basicsr)."""
    def __init__(self, num_feat=64):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out


class DEResNet(nn.Module):
    """Degradation Estimation Network."""
    def __init__(self, num_in_ch=3, num_degradation=2, degradation_degree_actv='sigmoid',
                 num_feats=None, num_blocks=None, downscales=None):
        super().__init__()
        if num_feats is None:
            num_feats = [64, 64, 64, 128]
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]
        if downscales is None:
            downscales = [1, 1, 2, 1]

        num_stage = len(num_feats)
        self.conv_first = nn.ModuleList()
        for _ in range(num_degradation):
            self.conv_first.append(nn.Conv2d(num_in_ch, num_feats[0], 3, 1, 1))

        self.body = nn.ModuleList()
        for _ in range(num_degradation):
            body = []
            for stage in range(num_stage):
                for _ in range(num_blocks[stage]):
                    body.append(ResidualBlockNoBN(num_feats[stage]))
                if downscales[stage] == 1:
                    if stage < num_stage - 1 and num_feats[stage] != num_feats[stage + 1]:
                        body.append(nn.Conv2d(num_feats[stage], num_feats[stage + 1], 3, 1, 1))
                elif downscales[stage] == 2:
                    body.append(nn.Conv2d(num_feats[stage], num_feats[min(stage + 1, num_stage - 1)], 3, 2, 1))
            self.body.append(nn.Sequential(*body))

        self.num_degradation = num_degradation
        self.fc_degree = nn.ModuleList()
        actv = nn.Sigmoid if degradation_degree_actv == 'sigmoid' else nn.Tanh
        for _ in range(num_degradation):
            self.fc_degree.append(nn.Sequential(
                nn.Linear(num_feats[-1], 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1),
                actv(),
            ))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        degrees = []
        for i in range(self.num_degradation):
            x_out = self.conv_first[i](x)
            feat = self.body[i](x_out)
            feat = self.avg_pool(feat)
            feat = feat.squeeze(-1).squeeze(-1)
            degrees.append(self.fc_degree[i](feat).squeeze(-1))
        return torch.stack(degrees, dim=1)


# ============================================================================
# Custom LoRA Forward (supports degradation modulation)
# ============================================================================

def my_lora_fwd(self, x, *args, **kwargs):
    """Custom LoRA forward with degradation modulation via de_mod."""
    kwargs.pop("adapter_names", None)
    result = self.base_layer(x, *args, **kwargs)

    if self.disable_adapters or self.merged:
        return result

    torch_result_dtype = result.dtype
    for active_adapter in self.active_adapters:
        if active_adapter not in self.lora_A.keys():
            continue
        lora_A = self.lora_A[active_adapter]
        lora_B = self.lora_B[active_adapter]
        dropout = self.lora_dropout[active_adapter]
        scaling = self.scaling[active_adapter]
        x = x.to(lora_A.weight.dtype)

        _tmp = lora_A(dropout(x))
        if isinstance(lora_A, nn.Conv2d):
            _tmp = torch.einsum('...khw,...kr->...rhw', _tmp, self.de_mod)
        elif isinstance(lora_A, nn.Linear):
            _tmp = torch.einsum('...lk,...kr->...lr', _tmp, self.de_mod)
        result = result + lora_B(_tmp) * scaling

    return result.to(torch_result_dtype)


# ============================================================================
# UNet layer numbering (for degradation embedding mapping)
# ============================================================================

def get_layer_number(module_name):
    base_layers = {'down_blocks': 0, 'mid_block': 4, 'up_blocks': 5}
    if module_name == 'conv_out':
        return 9
    base_layer = None
    for key in base_layers:
        if key in module_name:
            base_layer = base_layers[key]
            break
    if base_layer is None:
        return None
    additional_layers = int(re.findall(r'\.(\d+)', module_name)[0])
    return base_layer + additional_layers


# ============================================================================
# Wavelet Color Fix (post-processing)
# ============================================================================

def wavelet_blur(image, radius):
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    kernel = kernel[None, None].repeat(3, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode='replicate')
    return F.conv2d(image, kernel, groups=3, dilation=radius)


def wavelet_color_fix(target_pil, source_pil):
    to_tensor = transforms.ToTensor()
    target = to_tensor(target_pil).unsqueeze(0).float()
    source = to_tensor(source_pil).unsqueeze(0).float()

    # Wavelet decomposition
    target_high = torch.zeros_like(target)
    t_img = target.clone()
    for i in range(5):
        low = wavelet_blur(t_img, 2 ** i)
        target_high += (t_img - low)
        t_img = low

    s_img = source.clone()
    for i in range(5):
        low = wavelet_blur(s_img, 2 ** i)
        s_img = low
    source_low = s_img

    result = (target_high + source_low).squeeze(0).clamp_(0.0, 1.0)
    return transforms.ToPILImage()(result)


# ============================================================================
# Model Loading (CPU, before torchax)
# ============================================================================

def load_s3diff_model(sd_path, pretrained_path, de_net_path, lora_rank_unet=32, lora_rank_vae=16):
    """Load all S3Diff components on CPU."""
    from diffusers import AutoencoderKL, DDPMScheduler
    from diffusers import UNet2DConditionModel
    from transformers import AutoTokenizer, CLIPTextModel
    from peft import LoraConfig

    print("\n=== Loading S3Diff Components (CPU) ===")

    # 1. Load base SD-Turbo components
    print("  Loading tokenizer & text encoder...")
    tokenizer = AutoTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder")
    text_encoder.eval()

    print("  Loading VAE...")
    vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae")

    print("  Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")

    print("  Loading scheduler...")
    scheduler = DDPMScheduler.from_pretrained(sd_path, subfolder="scheduler")
    scheduler.set_timesteps(1, device="cpu")

    # 2. Load S3Diff weights and apply LoRA
    print(f"  Loading S3Diff weights from {pretrained_path}...")
    sd = torch.load(pretrained_path, map_location="cpu")

    # Apply VAE LoRA
    vae_lora_config = LoraConfig(
        r=sd["rank_vae"],
        init_lora_weights="gaussian",
        target_modules=sd["vae_lora_target_modules"]
    )
    vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
    _sd_vae = vae.state_dict()
    for k in sd["state_dict_vae"]:
        _sd_vae[k] = sd["state_dict_vae"][k]
    vae.load_state_dict(_sd_vae)

    # Apply UNet LoRA
    unet_lora_config = LoraConfig(
        r=sd["rank_unet"],
        init_lora_weights="gaussian",
        target_modules=[
            "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2",
            "conv_shortcut", "conv_out", "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
        ]
    )
    unet.add_adapter(unet_lora_config)
    _sd_unet = unet.state_dict()
    for k in sd["state_dict_unet"]:
        _sd_unet[k] = sd["state_dict_unet"][k]
    unet.load_state_dict(_sd_unet)

    # 3. Build degradation modulation MLPs
    num_embeddings = 64
    block_embedding_dim = 64
    W = nn.Parameter(sd["w"], requires_grad=False)

    vae_de_mlp = nn.Sequential(nn.Linear(num_embeddings * 4, 256), nn.ReLU(True))
    unet_de_mlp = nn.Sequential(nn.Linear(num_embeddings * 4, 256), nn.ReLU(True))
    vae_block_mlp = nn.Sequential(nn.Linear(block_embedding_dim, 64), nn.ReLU(True))
    unet_block_mlp = nn.Sequential(nn.Linear(block_embedding_dim, 64), nn.ReLU(True))
    vae_fuse_mlp = nn.Linear(256 + 64, lora_rank_vae ** 2)
    unet_fuse_mlp = nn.Linear(256 + 64, lora_rank_unet ** 2)
    vae_block_embeddings = nn.Embedding(6, block_embedding_dim)
    unet_block_embeddings = nn.Embedding(10, block_embedding_dim)

    # Load MLP weights
    for name, module in [
        ("vae_de_mlp", vae_de_mlp), ("unet_de_mlp", unet_de_mlp),
        ("vae_block_mlp", vae_block_mlp), ("unet_block_mlp", unet_block_mlp),
        ("vae_fuse_mlp", vae_fuse_mlp), ("unet_fuse_mlp", unet_fuse_mlp),
    ]:
        _sd_mlp = module.state_dict()
        for k in sd[f"state_dict_{name}"]:
            _sd_mlp[k] = sd[f"state_dict_{name}"][k]
        module.load_state_dict(_sd_mlp)

    vae_block_embeddings.load_state_dict(sd["state_embeddings"]["state_dict_vae_block"])
    unet_block_embeddings.load_state_dict(sd["state_embeddings"]["state_dict_unet_block"])

    # 4. Identify LoRA layers and override forward
    vae_lora_layers = []
    for name, module in vae.named_modules():
        if 'base_layer' in name:
            vae_lora_layers.append(name[:-len(".base_layer")])
    for name, module in vae.named_modules():
        if name in vae_lora_layers:
            module.forward = my_lora_fwd.__get__(module, module.__class__)

    unet_lora_layers = []
    for name, module in unet.named_modules():
        if 'base_layer' in name:
            unet_lora_layers.append(name[:-len(".base_layer")])
    for name, module in unet.named_modules():
        if name in unet_lora_layers:
            module.forward = my_lora_fwd.__get__(module, module.__class__)

    unet_layer_dict = {name: get_layer_number(name) for name in unet_lora_layers}

    # 5. Load DEResNet
    print(f"  Loading DEResNet from {de_net_path}...")
    de_net = DEResNet(num_in_ch=3, num_degradation=2)
    de_net_sd = torch.load(de_net_path, map_location="cpu")
    de_net.load_state_dict(de_net_sd, strict=True)
    de_net.eval()

    vae.eval()
    unet.eval()

    print("  All components loaded successfully")

    return {
        'tokenizer': tokenizer,
        'text_encoder': text_encoder,
        'vae': vae,
        'unet': unet,
        'scheduler': scheduler,
        'de_net': de_net,
        'W': W,
        'vae_de_mlp': vae_de_mlp,
        'unet_de_mlp': unet_de_mlp,
        'vae_block_mlp': vae_block_mlp,
        'unet_block_mlp': unet_block_mlp,
        'vae_fuse_mlp': vae_fuse_mlp,
        'unet_fuse_mlp': unet_fuse_mlp,
        'vae_block_embeddings': vae_block_embeddings,
        'unet_block_embeddings': unet_block_embeddings,
        'vae_lora_layers': vae_lora_layers,
        'unet_lora_layers': unet_lora_layers,
        'unet_layer_dict': unet_layer_dict,
        'lora_rank_unet': lora_rank_unet,
        'lora_rank_vae': lora_rank_vae,
    }


# ============================================================================
# torchax Utilities (from SDXL reference)
# ============================================================================

def get_hbm_usage(device=None):
    """Get current and peak HBM usage on TPU in GB."""
    import jax
    if device is None:
        device = jax.devices()[0]
    stats = device.memory_stats()
    return {
        'used_gb': stats['bytes_in_use'] / 1024**3,
        'peak_gb': stats['peak_bytes_in_use'] / 1024**3,
        'total_gb': stats['bytes_limit'] / 1024**3,
    }


def print_hbm(label, device=None):
    """Print HBM usage with a label."""
    hbm = get_hbm_usage(device)
    print(f"  [HBM] {label}: {hbm['used_gb']:.2f} GB / {hbm['total_gb']:.1f} GB (peak: {hbm['peak_gb']:.2f} GB)")
    return hbm


def setup_pytree_registrations():
    """Register necessary PyTree nodes for JAX transforms."""
    from jax.tree_util import register_pytree_node
    from diffusers.models.autoencoders import vae as diffusers_vae

    def flatten(obj):
        return obj.to_tuple(), type(obj)

    def unflatten(aux, children):
        return aux(*children)

    classes_to_register = [
        (diffusers_vae.DecoderOutput, "DecoderOutput"),
    ]

    try:
        from diffusers.models.autoencoders.vae import AutoencoderKLOutput
        classes_to_register.append((AutoencoderKLOutput, "AutoencoderKLOutput"))
    except ImportError:
        from diffusers.models.modeling_outputs import AutoencoderKLOutput
        classes_to_register.append((AutoencoderKLOutput, "AutoencoderKLOutput"))

    try:
        from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
        classes_to_register.append((UNet2DConditionOutput, "UNet2DConditionOutput"))
    except ImportError:
        pass

    for cls, name in classes_to_register:
        try:
            register_pytree_node(cls, flatten, unflatten)
            print(f"  PyTree registered: {name}")
        except ValueError:
            print(f"  PyTree exists: {name}")


def move_module_to_xla(env, module):
    """Move module weights to XLA device."""
    import jax
    with jax.default_device("cpu"):
        state_dict = module.state_dict()
        state_dict = env.to_xla(state_dict)
        module.load_state_dict(state_dict, assign=True)


def override_op(env, op, impl):
    """Override a torchax op."""
    from torchax.ops import ops_registry
    env._ops[op] = ops_registry.Operator(
        op, impl, is_jax_function=False, is_user_defined=True,
        needs_env=False, is_view_op=False,
    )


# ============================================================================
# Inference Pipeline
# ============================================================================

def preprocess_image(image_path, scale_factor=4):
    """Load and preprocess LR image for S3Diff."""
    img = Image.open(image_path).convert('RGB')
    to_tensor = transforms.ToTensor()
    im_lr = to_tensor(img).unsqueeze(0)  # (1, 3, H, W) in [0, 1]

    ori_h, ori_w = im_lr.shape[2:]
    # 4x bilinear upscale
    im_lr_resize = F.interpolate(
        im_lr,
        size=(ori_h * scale_factor, ori_w * scale_factor),
        mode='bilinear',
        align_corners=False,
    )

    # Normalize to [-1, 1]
    im_lr_resize_norm = (im_lr_resize * 2 - 1.0).clamp(-1.0, 1.0)
    resize_h, resize_w = im_lr_resize_norm.shape[2:]

    # Pad to multiple of 64
    pad_h = (math.ceil(resize_h / 64)) * 64 - resize_h
    pad_w = (math.ceil(resize_w / 64)) * 64 - resize_w
    if pad_h > 0 or pad_w > 0:
        im_lr_resize_norm = F.pad(im_lr_resize_norm, (0, pad_w, 0, pad_h), mode='reflect')

    return im_lr, im_lr_resize, im_lr_resize_norm, (resize_h, resize_w)


def encode_prompts(tokenizer, text_encoder, pos_prompt, neg_prompt):
    """Encode positive and negative prompts on CPU."""
    pos_tokens = tokenizer(
        pos_prompt, max_length=tokenizer.model_max_length,
        padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids
    neg_tokens = tokenizer(
        neg_prompt, max_length=tokenizer.model_max_length,
        padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids

    with torch.no_grad():
        pos_enc = text_encoder(pos_tokens)[0]
        neg_enc = text_encoder(neg_tokens)[0]

    return pos_enc, neg_enc


def compute_degradation_modulation(components, deg_score):
    """Compute degradation-guided LoRA modulation and set de_mod on modules."""
    W = components['W']
    vae_de_mlp = components['vae_de_mlp']
    unet_de_mlp = components['unet_de_mlp']
    vae_block_mlp = components['vae_block_mlp']
    unet_block_mlp = components['unet_block_mlp']
    vae_fuse_mlp = components['vae_fuse_mlp']
    unet_fuse_mlp = components['unet_fuse_mlp']
    vae_block_embeddings = components['vae_block_embeddings']
    unet_block_embeddings = components['unet_block_embeddings']
    lora_rank_vae = components['lora_rank_vae']
    lora_rank_unet = components['lora_rank_unet']

    # Fourier embedding
    deg_proj = deg_score[..., None] * W[None, None, :] * 2 * np.pi
    deg_proj = torch.cat([torch.sin(deg_proj), torch.cos(deg_proj)], dim=-1)
    deg_proj = torch.cat([deg_proj[:, 0], deg_proj[:, 1]], dim=-1)

    # MLP forward
    vae_de_c_embed = vae_de_mlp(deg_proj)
    unet_de_c_embed = unet_de_mlp(deg_proj)
    vae_block_c_embeds = vae_block_mlp(vae_block_embeddings.weight)
    unet_block_c_embeds = unet_block_mlp(unet_block_embeddings.weight)

    vae_embeds = vae_fuse_mlp(torch.cat([
        vae_de_c_embed.unsqueeze(1).repeat(1, vae_block_c_embeds.shape[0], 1),
        vae_block_c_embeds.unsqueeze(0).repeat(vae_de_c_embed.shape[0], 1, 1)
    ], -1))
    unet_embeds = unet_fuse_mlp(torch.cat([
        unet_de_c_embed.unsqueeze(1).repeat(1, unet_block_c_embeds.shape[0], 1),
        unet_block_c_embeds.unsqueeze(0).repeat(unet_de_c_embed.shape[0], 1, 1)
    ], -1))

    # Set de_mod on VAE LoRA layers
    vae = components['vae']
    for layer_name, module in vae.named_modules():
        if layer_name in components['vae_lora_layers']:
            split_name = layer_name.split(".")
            if split_name[1] == 'down_blocks':
                block_id = int(split_name[2])
                vae_embed = vae_embeds[:, block_id]
            elif split_name[1] == 'mid_block':
                vae_embed = vae_embeds[:, -2]
            else:
                vae_embed = vae_embeds[:, -1]
            module.de_mod = vae_embed.reshape(-1, lora_rank_vae, lora_rank_vae)

    # Set de_mod on UNet LoRA layers
    unet = components['unet']
    for layer_name, module in unet.named_modules():
        if layer_name in components['unet_lora_layers']:
            split_name = layer_name.split(".")
            if split_name[0] == 'down_blocks':
                block_id = int(split_name[1])
                unet_embed = unet_embeds[:, block_id]
            elif split_name[0] == 'mid_block':
                unet_embed = unet_embeds[:, 4]
            elif split_name[0] == 'up_blocks':
                block_id = int(split_name[1]) + 5
                unet_embed = unet_embeds[:, block_id]
            else:
                unet_embed = unet_embeds[:, -1]
            module.de_mod = unet_embed.reshape(-1, lora_rank_unet, lora_rank_unet)


def run_s3diff_inference(components, im_lr_resize_norm, pos_enc, neg_enc, mesh, env):
    """Run the S3Diff inference pipeline on TPU."""
    import jax
    import jax.numpy as jnp

    vae = components['vae']
    unet = components['unet']
    scheduler = components['scheduler']

    timesteps = torch.tensor([999]).long()

    with mesh:
        print_hbm("Before inference")

        # VAE Encode
        print("  VAE Encoding...")
        t0 = time.perf_counter()
        lq_latent = vae.encode(im_lr_resize_norm).latent_dist.sample() * vae.config.scaling_factor
        jax.effects_barrier()
        vae_enc_time = time.perf_counter() - t0
        print(f"    VAE Encode: {vae_enc_time:.2f}s, latent shape: {list(lq_latent.shape)}")
        print_hbm("After VAE Encode")

        # UNet single-step denoising (no tiling for small images)
        print("  UNet Denoising (1 step)...")
        t1 = time.perf_counter()

        # Convert timestep to XLA
        timestep_xla = env.j2t_iso(jnp.array([999], dtype=jnp.int32))

        # return_dict is declared as static_argname in CompileOptions so JAX
        # treats it as a compile-time constant instead of tracing it.
        pos_model_pred = unet(lq_latent, timestep_xla, encoder_hidden_states=pos_enc, return_dict=False)[0]
        neg_model_pred = unet(lq_latent, timestep_xla, encoder_hidden_states=neg_enc, return_dict=False)[0]
        model_pred = neg_model_pred + GUIDANCE_SCALE * (pos_model_pred - neg_model_pred)

        jax.effects_barrier()
        unet_time = time.perf_counter() - t1
        print(f"    UNet Denoise: {unet_time:.2f}s")
        print_hbm("After UNet")

        # Scheduler step
        # DDPM scheduler step on CPU (small tensor operation)
        x_denoised = scheduler.step(model_pred, 999, lq_latent, return_dict=True).prev_sample

        # VAE Decode
        print("  VAE Decoding...")
        t2 = time.perf_counter()
        output = vae.decode(x_denoised / vae.config.scaling_factor).sample
        jax.effects_barrier()
        vae_dec_time = time.perf_counter() - t2
        print(f"    VAE Decode: {vae_dec_time:.2f}s")
        print_hbm("After VAE Decode")

    return output, {
        'vae_encode': vae_enc_time,
        'unet_denoise': unet_time,
        'vae_decode': vae_dec_time,
    }


def postprocess_output(output_tensor, resize_h, resize_w, im_lr_resize, align_method='wavelet'):
    """Post-process the output tensor to PIL image."""
    import jax.numpy as jnp

    # Crop padding
    output_tensor = output_tensor[:, :, :resize_h, :resize_w]

    # Convert to [0, 1]
    output_tensor = output_tensor * 0.5 + 0.5

    # Convert to numpy
    if hasattr(output_tensor, '_elem'):
        jax_array = output_tensor._elem
        if jax_array.dtype == jnp.bfloat16:
            np_array = np.array(jax_array.astype(jnp.float32))
        else:
            np_array = np.array(jax_array)
    else:
        np_array = output_tensor.cpu().float().numpy()

    np_array = np.transpose(np_array[0], (1, 2, 0))
    np_array = (np_array * 255).clip(0, 255).astype(np.uint8)
    output_pil = Image.fromarray(np_array)

    # Color correction
    if align_method != 'nofix':
        im_lr_np = im_lr_resize[0].cpu().numpy()
        im_lr_np = np.transpose(im_lr_np, (1, 2, 0))
        im_lr_np = (im_lr_np * 255).clip(0, 255).astype(np.uint8)
        source_pil = Image.fromarray(im_lr_np)

        if align_method == 'wavelet':
            output_pil = wavelet_color_fix(output_pil, source_pil)

    return output_pil


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="S3Diff 4x Super-Resolution on TPU")
    parser.add_argument("--input", type=str, required=True, help="Input LR image path")
    parser.add_argument("--output", type=str, default=None, help="Output SR image path")
    parser.add_argument("--sd_path", type=str, default=None, help="SD-Turbo model path")
    parser.add_argument("--pretrained_path", type=str, default=None, help="S3Diff weights path")
    parser.add_argument("--de_net_path", type=str, default=None, help="DEResNet weights path")
    parser.add_argument("--pos_prompt", type=str, default=DEFAULT_POS_PROMPT)
    parser.add_argument("--neg_prompt", type=str, default=DEFAULT_NEG_PROMPT)
    parser.add_argument("--align_method", type=str, default="wavelet", choices=["wavelet", "adain", "nofix"])
    parser.add_argument("--warmup", action="store_true", help="Run warmup pass before benchmark")
    parser.add_argument("--benchmark_iters", type=int, default=1, help="Number of benchmark iterations")
    return parser.parse_args()


def main():
    args = parse_args()

    import jax
    import jax.numpy as jnp
    from jax.experimental import mesh_utils
    from jax.sharding import Mesh

    n_devices = len(jax.devices())
    print(f"\n{'='*60}")
    print("S3Diff 4x Super-Resolution (TPU)")
    print(f"{'='*60}")
    print(f"  Devices: {n_devices} TPU chip(s)")
    print(f"  Input: {args.input}")

    # Configure JAX cache
    jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax_cache"))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    # Resolve model paths
    if args.sd_path is None:
        from huggingface_hub import snapshot_download
        args.sd_path = snapshot_download(repo_id=SD_TURBO_ID)
    if args.pretrained_path is None:
        from huggingface_hub import hf_hub_download
        args.pretrained_path = hf_hub_download(repo_id=S3DIFF_REPO, filename=S3DIFF_WEIGHTS)
    if args.de_net_path is None:
        script_dir = Path(__file__).parent
        # Try upstream repo first, then local
        local_path = script_dir / "assets" / "mm-realsr" / "de_net.pth"
        alt_path = script_dir / "assets" / "de_net.pth"
        if local_path.exists():
            args.de_net_path = str(local_path)
        elif alt_path.exists():
            args.de_net_path = str(alt_path)
        else:
            from huggingface_hub import hf_hub_download
            args.de_net_path = hf_hub_download(repo_id=S3DIFF_REPO, filename="assets/mm-realsr/de_net.pth")

    # 1. Preprocess image (CPU)
    print(f"\n=== Preprocessing ===")
    im_lr, im_lr_resize, im_lr_resize_norm, (resize_h, resize_w) = preprocess_image(args.input)
    print(f"  Input: {im_lr.shape[2]}x{im_lr.shape[3]}")
    print(f"  After 4x upscale: {resize_h}x{resize_w}")
    print(f"  After padding: {im_lr_resize_norm.shape[2]}x{im_lr_resize_norm.shape[3]}")

    # 2. Load model (CPU, before torchax)
    # DEResNet needs float32, set bfloat16 default after loading
    components = load_s3diff_model(args.sd_path, args.pretrained_path, args.de_net_path)

    # 3. Encode prompts (CPU)
    print("\n=== Encoding Prompts (CPU) ===")
    pos_enc, neg_enc = encode_prompts(
        components['tokenizer'], components['text_encoder'],
        args.pos_prompt, args.neg_prompt
    )
    print(f"  Prompt embeddings: {list(pos_enc.shape)}")

    # Free text encoder
    del components['tokenizer'], components['text_encoder']
    gc.collect()

    # 4. Degradation estimation (CPU)
    print("\n=== Degradation Estimation (CPU) ===")
    with torch.no_grad():
        deg_score = components['de_net'](im_lr.float())
    print(f"  Degradation scores: blur={deg_score[0,0].item():.4f}, noise={deg_score[0,1].item():.4f}")

    # 5. Compute degradation modulation (CPU)
    print("  Computing LoRA modulation...")
    with torch.no_grad():
        compute_degradation_modulation(components, deg_score)
    print("  de_mod set on all LoRA layers")

    # Free DEResNet
    del components['de_net']
    gc.collect()

    # Set bfloat16 default for TPU operations
    torch.set_default_dtype(torch.bfloat16)

    # 6. Enable torchax
    print("\n=== Enabling torchax ===")
    setup_pytree_registrations()

    import torchax
    from torchax.ops import jaten, ops_registry

    torchax.enable_globally()
    env = torchax.default_env()

    # Create mesh (single device)
    mesh = Mesh(
        mesh_utils.create_device_mesh((1,), allow_split_physical_axes=True),
        ("x",)
    )

    # Override ops
    from splash_attention_utils import sdpa_reference

    def scaled_dot_product_attention_impl(query, key, value, attn_mask=None, dropout_p=0.0,
                                           is_causal=False, scale=None, enable_gqa=False):
        return sdpa_reference(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)

    def torch_conv2d_jax(input, weight, bias=None, stride=1, padding=0,
                          dilation=1, groups=1, *, env=env):
        jinput, jweight, jbias = env.t2j_iso((input, weight, bias))
        # Ensure dtype consistency (scheduler may output float32)
        if jinput.dtype != jweight.dtype:
            jinput = jinput.astype(jweight.dtype)
        if jbias is not None and jbias.dtype != jweight.dtype:
            jbias = jbias.astype(jweight.dtype)
        res = jaten._aten_conv2d(jinput, jweight, jbias, stride, padding, dilation, groups)
        return env.j2t_iso(res)

    override_op(env, torch.nn.functional.conv2d, functools.partial(torch_conv2d_jax, env=env))
    override_op(env, torch.nn.functional.scaled_dot_product_attention, scaled_dot_product_attention_impl)

    # 7. Move models to XLA
    print("\n=== Moving Models to XLA ===")
    vae = components['vae']
    unet = components['unet']

    # Convert ALL parameters to bfloat16 before moving to XLA
    # Use state_dict approach to catch everything (including PEFT-wrapped layers)
    for p in vae.parameters():
        p.data = p.data.to(torch.bfloat16)
    for b in vae.buffers():
        b.data = b.data.to(torch.bfloat16)
    for p in unet.parameters():
        p.data = p.data.to(torch.bfloat16)
    for b in unet.buffers():
        b.data = b.data.to(torch.bfloat16)

    move_module_to_xla(env, vae)
    move_module_to_xla(env, unet)

    # Move MLP modules to XLA
    for name in ['vae_de_mlp', 'unet_de_mlp', 'vae_block_mlp', 'unet_block_mlp',
                 'vae_fuse_mlp', 'unet_fuse_mlp', 'vae_block_embeddings', 'unet_block_embeddings']:
        move_module_to_xla(env, components[name])
    components['W'] = env.to_xla(components['W'])

    print_hbm("After loading weights to XLA")

    # Convert de_mod to XLA torchax tensors (MUST be before compile)
    print("  Converting de_mod to XLA...")
    for layer_name, module in vae.named_modules():
        if hasattr(module, 'de_mod'):
            module.de_mod = env.to_xla(module.de_mod.to(torch.bfloat16))
    for layer_name, module in unet.named_modules():
        if hasattr(module, 'de_mod'):
            module.de_mod = env.to_xla(module.de_mod.to(torch.bfloat16))

    # Compile VAE decoder and UNet with torchax.compile() for JIT optimization.
    # de_mod attributes are captured as dynamic buffers by JittableModule and
    # properly restored via _reparametrize_module during each forward call.
    # Note: requires torchax patch to skip properties in extract_all_buffers.
    vae.decoder = torchax.compile(vae.decoder)
    components['unet'] = torchax.compile(
        unet, torchax.CompileOptions(jax_jit_kwargs={'static_argnames': ('return_dict',)})
    )
    print("  Models on XLA (VAE decoder + UNet compiled)")

    # Move input tensors to XLA
    with env:
        im_lr_resize_norm = im_lr_resize_norm.to('jax').to(torch.bfloat16)
        pos_enc = pos_enc.to('jax').to(torch.bfloat16)
        neg_enc = neg_enc.to('jax').to(torch.bfloat16)

    # Update scheduler alphas_cumprod to XLA
    with env:
        components['scheduler'].alphas_cumprod = components['scheduler'].alphas_cumprod.to('jax')

    # 8. Warmup run
    if args.warmup:
        print(f"\n{'='*60}")
        print("Warmup Run (JIT compilation)")
        print(f"{'='*60}")
        warmup_start = time.perf_counter()
        _, _ = run_s3diff_inference(components, im_lr_resize_norm, pos_enc, neg_enc, mesh, env)
        jax.effects_barrier()
        warmup_time = time.perf_counter() - warmup_start
        print(f"  Warmup: {warmup_time:.2f}s")

    # 9. Benchmark run(s)
    n_iters = args.benchmark_iters
    print(f"\n{'='*60}")
    print(f"Benchmark Run ({n_iters} iteration{'s' if n_iters > 1 else ''})")
    print(f"{'='*60}")

    all_timings = []
    for i in range(n_iters):
        if n_iters > 1:
            print(f"\n  --- Iteration {i+1}/{n_iters} ---")
        iter_start = time.perf_counter()
        output_tensor, timings = run_s3diff_inference(components, im_lr_resize_norm, pos_enc, neg_enc, mesh, env)
        jax.effects_barrier()
        timings['total'] = time.perf_counter() - iter_start
        all_timings.append(timings)

    # Use last iteration for output
    total_time = all_timings[-1]['total']

    # 10. Post-process
    print(f"\n=== Post-processing ===")
    output_pil = postprocess_output(output_tensor, resize_h, resize_w, im_lr_resize, args.align_method)

    # Save output
    if args.output is None:
        stem = Path(args.input).stem
        args.output = f"{stem}_sr_4x.png"
    output_pil.save(args.output)
    print(f"  Output saved: {args.output} ({output_pil.size[0]}x{output_pil.size[1]})")

    # Performance summary
    print(f"\n{'='*60}")
    print("Performance Summary")
    print(f"{'='*60}")
    print(f"  Input:       {im_lr.shape[3]}x{im_lr.shape[2]}")
    print(f"  Output:      {output_pil.size[0]}x{output_pil.size[1]} (4x upscale)")

    if n_iters > 1:
        avg_t = {k: sum(t[k] for t in all_timings) / n_iters for k in all_timings[0]}
        min_t = {k: min(t[k] for t in all_timings) for k in all_timings[0]}
        print(f"  --- Average over {n_iters} iterations ---")
        print(f"  VAE Encode:  {avg_t['vae_encode']:.3f}s (min {min_t['vae_encode']:.3f}s)")
        print(f"  UNet (1 step): {avg_t['unet_denoise']:.3f}s (min {min_t['unet_denoise']:.3f}s)")
        print(f"  VAE Decode:  {avg_t['vae_decode']:.3f}s (min {min_t['vae_decode']:.3f}s)")
        print(f"  Total:       {avg_t['total']:.3f}s (min {min_t['total']:.3f}s)")
    else:
        print(f"  VAE Encode:  {timings['vae_encode']:.2f}s")
        print(f"  UNet (1 step): {timings['unet_denoise']:.2f}s")
        print(f"  VAE Decode:  {timings['vae_decode']:.2f}s")
        print(f"  Total:       {total_time:.2f}s")

    print(f"  Devices:     {n_devices} TPU chip(s)")

    # HBM summary
    hbm = get_hbm_usage()
    print(f"\n  --- HBM Usage ---")
    print(f"  Current:     {hbm['used_gb']:.2f} GB")
    print(f"  Peak:        {hbm['peak_gb']:.2f} GB")
    print(f"  Total:       {hbm['total_gb']:.1f} GB")
    print(f"  Utilization: {hbm['peak_gb'] / hbm['total_gb'] * 100:.1f}%")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
