#!/usr/bin/env python3
"""
HunyuanVideo-1.5 三阶段生成 - 阶段2：Transformer (DiT) (GPU 版本)

关键修复：直接加载 transformer，不使用 create_pipeline
避免加载不需要的组件（text encoder 14GB）导致 OOM

用于 GPU H100 8卡环境
"""

import os
import sys

# ============================================================================
# 关键：和官方 generate.py 一样，在模块级别初始化并行状态
# generate.py 第 37-38 行:
#   parallel_dims = initialize_parallel_state(sp=int(os.environ.get('WORLD_SIZE', '1')))
#   torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
# ============================================================================

if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 添加 HunyuanVideo-1.5-TPU 到路径
HUNYUAN_ROOT = os.path.expanduser("~/HunyuanVideo-1.5-TPU")
if HUNYUAN_ROOT not in sys.path:
    sys.path.insert(0, HUNYUAN_ROOT)

import torch
from hyvideo.commons.parallel_states import initialize_parallel_state

# 模块级别初始化（和官方一致）
parallel_dims = initialize_parallel_state(sp=int(os.environ.get('WORLD_SIZE', '1')))
torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))

# 现在导入其他模块
import time
import random
import argparse
import atexit
from types import SimpleNamespace
import numpy as np
from PIL import Image
from torch import distributed as dist
from tqdm import tqdm

# 直接导入需要的组件，不使用 create_pipeline
from hyvideo.models.transformers.hunyuanvideo_1_5_transformer import HunyuanVideo_1_5_DiffusionTransformer
from hyvideo.schedulers.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from hyvideo.commons.infer_state import initialize_infer_state, get_infer_state
from hyvideo.commons import auto_offload_model, PIPELINE_CONFIGS, is_sparse_attn_available
from hyvideo.commons.parallel_states import get_parallel_state
from hyvideo.utils.multitask_utils import merge_tensor_by_mask

# 检查 angelslim 是否可用（用于 cache 加速）
def is_angelslim_available():
    try:
        import angelslim
        return True
    except ImportError:
        return False

from utils import (
    load_embeddings_from_safetensors,
    save_latents_to_safetensors,
    load_generation_config,
    save_generation_config,
    get_default_paths,
)

# Register cleanup function
def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

atexit.register(cleanup_distributed)


def get_rank():
    return int(os.environ.get('RANK', '0'))

def print_rank0(msg):
    if get_rank() == 0:
        print(msg)


# ============================================================================
# 辅助函数：复制自 pipeline 的必要功能
# ============================================================================

def get_latent_size(video_length, height, width, vae_temporal_ratio=4, vae_spatial_ratio=16):
    """计算 latent 尺寸"""
    video_length = (video_length - 1) // vae_temporal_ratio + 1
    height = height // vae_spatial_ratio
    width = width // vae_spatial_ratio
    return video_length, height, width


def get_task_mask(task_type, latent_target_length):
    """获取任务 mask"""
    if task_type == "t2v":
        return torch.zeros(latent_target_length)
    elif task_type == "i2v":
        mask = torch.zeros(latent_target_length)
        mask[0] = 1.0
        return mask
    else:
        raise ValueError(f"{task_type} is not supported!")


def prepare_latents(batch_size, num_channels, latent_height, latent_width, video_length,
                   dtype, device, generator):
    """准备随机 latents"""
    shape = (batch_size, num_channels, video_length, latent_height, latent_width)
    latents = torch.randn(shape, generator=generator, device=torch.device('cpu'), dtype=dtype).to(device)
    return latents


def prepare_cond_latents(task_type, image_cond, latents, multitask_mask):
    """准备条件 latents"""
    if image_cond is not None and task_type == 'i2v':
        latents_concat = image_cond.repeat(1, 1, latents.shape[2], 1, 1)
        latents_concat[:, :, 1:, :, :] = 0.0
    else:
        latents_concat = torch.zeros_like(latents)
    
    mask_zeros = torch.zeros(latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4])
    mask_ones = torch.ones(latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4])
    mask_concat = merge_tensor_by_mask(mask_zeros.cpu(), mask_ones.cpu(), mask=multitask_mask.cpu(), dim=2).to(device=latents.device)
    
    return torch.concat([latents_concat, mask_concat], dim=1)


def get_closest_resolution(aspect_ratio, target_resolution):
    """根据宽高比获取最接近的分辨率"""
    from hyvideo.utils.data_utils import generate_crop_size_list, get_closest_ratio
    
    target_size_config = {
        "360p": {"bucket_hw_base_size": 480, "bucket_hw_bucket_stride": 16},
        "480p": {"bucket_hw_base_size": 640, "bucket_hw_bucket_stride": 16},
        "720p": {"bucket_hw_base_size": 960, "bucket_hw_bucket_stride": 16},
        "1080p": {"bucket_hw_base_size": 1440, "bucket_hw_bucket_stride": 16},
    }
    
    bucket_hw_base_size = target_size_config[target_resolution]["bucket_hw_base_size"]
    bucket_hw_bucket_stride = target_size_config[target_resolution]["bucket_hw_bucket_stride"]
    
    if ":" in aspect_ratio:
        w_ratio, h_ratio = map(int, aspect_ratio.split(":"))
    else:
        w_ratio, h_ratio = 16, 9
    
    crop_size_list = generate_crop_size_list(bucket_hw_base_size, bucket_hw_bucket_stride)
    aspect_ratios = np.array([round(float(h) / float(w), 5) for h, w in crop_size_list])
    closest_size, _ = get_closest_ratio(h_ratio, w_ratio, aspect_ratios, crop_size_list)
    
    return closest_size[0], closest_size[1]  # height, width


def str_to_bool(v):
    """将字符串转换为布尔值（用于命令行参数）"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser(description='HunyuanVideo-1.5 Stage 2: Transformer')
    
    parser.add_argument('--input_dir', type=str, default='./stage_outputs')
    parser.add_argument('--output_dir', type=str, default=None)
    
    # 视频生成参数（和 generate.py 一致）
    parser.add_argument('--aspect_ratio', type=str, default='16:9')
    parser.add_argument('--video_length', type=int, default=49)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=6.0)
    parser.add_argument('--seed', type=int, default=42)
    
    # ========================================================================
    # Attention 模式参数
    # ========================================================================
    parser.add_argument(
        '--attn_mode', type=str, default='flash',
        choices=['flash', 'flash2', 'flash3', 'torch', 'sageattn', 'flex-block-attn'],
        help='Attention 实现模式:\n'
             '  flash: 自动选择 flash2/flash3 (默认)\n'
             '  flash2: Flash Attention 2\n'
             '  flash3: Flash Attention 3 (Hopper GPU)\n'
             '  torch: PyTorch 原生 SDPA\n'
             '  sageattn: SageAttention (需要安装)\n'
             '  flex-block-attn: Sparse Attention (SSTA, 仅 H100)'
    )
    parser.add_argument(
        '--sparse_attn', type=str_to_bool, nargs='?', const=True, default=False,
        help='启用 Sparse Attention (SSTA)。等效于 --attn_mode flex-block-attn\n'
             '注意: 需要 distilled 版本的模型权重'
    )
    parser.add_argument(
        '--use_sageattn', type=str_to_bool, nargs='?', const=True, default=False,
        help='使用 SageAttention 加速 (~1.2x)。等效于 --attn_mode sageattn\n'
             '注意: 需要安装 sageattention 包'
    )
    
    # ========================================================================
    # Cache 加速参数 (DeepCache / TeaCache)
    # ========================================================================
    parser.add_argument(
        '--enable_cache', type=str_to_bool, nargs='?', const=True, default=False,
        help='启用 Cache 加速 (DeepCache/TeaCache)。\n'
             '原理: 在某些步骤跳过部分 transformer block 的计算，复用之前的输出。\n'
             '效果: 可加速 1.3x-2x，但可能略微影响质量。'
    )
    parser.add_argument(
        '--cache_type', type=str, default='deepcache',
        choices=['deepcache', 'teacache'],
        help='Cache 类型:\n'
             '  deepcache: DeepCache 策略 (默认)\n'
             '  teacache: TeaCache 策略'
    )
    parser.add_argument(
        '--no_cache_block_id', type=str, default='53',
        help='不使用 cache 的 block ID (默认: 53, 即最后一层不 cache)'
    )
    parser.add_argument(
        '--cache_start_step', type=int, default=11,
        help='开始使用 cache 的步数 (默认: 11, 前期不 cache 保证质量)'
    )
    parser.add_argument(
        '--cache_end_step', type=int, default=45,
        help='停止使用 cache 的步数 (默认: 45, 后期不 cache 保证收敛)'
    )
    parser.add_argument(
        '--cache_step_interval', type=int, default=4,
        help='Cache 步长间隔 (默认: 4, 每4步复用一次缓存)'
    )
    
    args = parser.parse_args()
    
    # 检查互斥参数
    if args.sparse_attn and args.use_sageattn:
        raise ValueError("sparse_attn 和 use_sageattn 不能同时启用，请只选择一个")
    
    # 根据便捷参数设置 attn_mode
    if args.sparse_attn:
        args.attn_mode = 'flex-block-attn'
    elif args.use_sageattn:
        args.attn_mode = 'sageattn'
    
    # 初始化 infer_state（和 generate.py 一致）
    # 根据参数决定是否启用 SageAttention 和 Cache
    use_sageattn_for_infer = (args.attn_mode == 'sageattn')
    
    infer_args = SimpleNamespace(
        use_sageattn=use_sageattn_for_infer,
        sage_blocks_range="0-53",
        enable_torch_compile=False,
        enable_cache=args.enable_cache,
        cache_type=args.cache_type,
        no_cache_block_id=args.no_cache_block_id,
        cache_start_step=args.cache_start_step,
        cache_end_step=args.cache_end_step,
        total_steps=args.num_inference_steps,
        cache_step_interval=args.cache_step_interval,
    )
    initialize_infer_state(infer_args)
    
    # 打印 cache 配置
    if args.enable_cache:
        print_rank0(f"\n[Cache 配置]")
        print_rank0(f"  cache_type: {args.cache_type}")
        print_rank0(f"  no_cache_block_id: {args.no_cache_block_id}")
        print_rank0(f"  cache_start_step: {args.cache_start_step}")
        print_rank0(f"  cache_end_step: {args.cache_end_step}")
        print_rank0(f"  cache_step_interval: {args.cache_step_interval}")
    
    output_dir = args.output_dir or args.input_dir
    input_paths = get_default_paths(args.input_dir)
    output_paths = get_default_paths(output_dir)
    
    print_rank0(f"\n{'='*60}")
    print_rank0("HunyuanVideo-1.5 Stage 2: Transformer (直接加载，无 text encoder)")
    print_rank0(f"{'='*60}")
    
    # 加载 Stage 1 配置
    print_rank0(f"\n加载 Stage 1 配置: {input_paths['config']}")
    config = load_generation_config(input_paths['config'])
    
    # 加载 Stage 1 embeddings
    print_rank0(f"加载 Stage 1 embeddings: {input_paths['embeddings']}")
    embeddings_dict, _ = load_embeddings_from_safetensors(input_paths['embeddings'], device='cpu')
    
    # 更新配置
    config['aspect_ratio'] = args.aspect_ratio
    config['video_length'] = args.video_length
    config['num_inference_steps'] = args.num_inference_steps
    config['guidance_scale'] = args.guidance_scale
    config['seed'] = args.seed
    
    model_path = config['model_path']
    transformer_version = config['transformer_version']
    resolution = config.get('resolution', '720p')
    task_type = config.get('task_type', 't2v')
    
    # ========================================================================
    # 处理 Sparse Attention 的 transformer_version
    # ========================================================================
    # Sparse Attention 需要使用 distilled_sparse 版本的模型权重
    # 例如: 720p_t2v -> 720p_t2v_distilled_sparse
    if args.sparse_attn:
        if '_distilled_sparse' not in transformer_version:
            # 检查是否需要添加 _distilled 前缀
            if '_distilled' not in transformer_version:
                transformer_version = f"{transformer_version}_distilled_sparse"
            else:
                transformer_version = f"{transformer_version}_sparse"
            print_rank0(f"  [Sparse Attention] 切换到 distilled_sparse 版本: {transformer_version}")
        
        # 检查 sparse attention 是否可用
        if not is_sparse_attn_available():
            raise RuntimeError(
                f"Sparse Attention (flex-block-attn) 在当前 GPU 上不可用。\n"
                f"该功能仅支持 NVIDIA H100 GPU。\n"
                f"当前 GPU: {torch.cuda.get_device_properties(0).name}"
            )
    
    print_rank0(f"\n配置:")
    print_rank0(f"  model_path: {model_path}")
    print_rank0(f"  transformer_version: {transformer_version}")
    print_rank0(f"  resolution: {resolution}")
    print_rank0(f"  task_type: {task_type}")
    print_rank0(f"  aspect_ratio: {args.aspect_ratio}")
    print_rank0(f"  video_length: {args.video_length}")
    print_rank0(f"  num_inference_steps: {args.num_inference_steps}")
    print_rank0(f"  guidance_scale: {args.guidance_scale}")
    print_rank0(f"  seed: {args.seed}")
    print_rank0(f"  attn_mode: {args.attn_mode}")
    print_rank0(f"  enable_cache: {args.enable_cache}")
    if args.enable_cache:
        print_rank0(f"  cache_type: {args.cache_type}")
    
    # ========================================================================
    # 直接加载 Transformer（不使用 create_pipeline，避免加载 text encoder）
    # ========================================================================
    print_rank0(f"\n直接加载 Transformer（不加载其他组件）...")
    
    dtype = config.get('dtype', 'bf16')
    if dtype == 'bf16':
        transformer_dtype = torch.bfloat16
    else:
        transformer_dtype = torch.float32
    
    # 只加载 transformer
    transformer_path = os.path.join(model_path, "transformer", transformer_version)
    print_rank0(f"  加载路径: {transformer_path}")
    
    transformer = HunyuanVideo_1_5_DiffusionTransformer.from_pretrained(
        transformer_path,
        torch_dtype=transformer_dtype,
        low_cpu_mem_usage=True,
        attn_mode=args.attn_mode,  # 传入 attention 模式
    )
    
    # 移动到 GPU
    device = torch.device('cuda')
    transformer = transformer.to(device)
    transformer.eval()
    
    # 设置 attention 模式（确保所有 block 都使用相同的模式）
    transformer.set_attn_mode(args.attn_mode)
    
    # ========================================================================
    # 设置 Cache Helper（如果启用）
    # Cache 机制通过 angelslim 库实现，hook transformer 的 double_blocks
    # ========================================================================
    cache_helper = None
    if args.enable_cache:
        if not is_angelslim_available():
            raise RuntimeError(
                "请安装 angelslim==0.2.1 以启用 cache 加速:\n"
                "  pip install angelslim==0.2.1"
            )
        
        from angelslim.compressor.diffusion import DeepCacheHelper, TeaCacheHelper
        
        infer_state = get_infer_state()
        
        # 计算不使用 cache 的步骤列表
        # - 前期 (0 ~ cache_start_step): 不 cache，保证质量
        # - 中期 (cache_start_step ~ cache_end_step): 每隔 cache_step_interval 步 cache
        # - 后期 (cache_end_step ~ total_steps): 不 cache，保证收敛
        no_cache_steps = (
            list(range(0, infer_state.cache_start_step)) +  # 前期不 cache
            list(range(infer_state.cache_start_step, infer_state.cache_end_step, infer_state.cache_step_interval)) +  # 中期间隔 cache
            list(range(infer_state.cache_end_step, infer_state.total_steps))  # 后期不 cache
        )
        
        print_rank0(f"\n[Cache Helper 初始化]")
        print_rank0(f"  cache_type: {args.cache_type}")
        print_rank0(f"  no_cache_steps 数量: {len(no_cache_steps)} / {infer_state.total_steps}")
        print_rank0(f"  实际 cache 步数: {infer_state.total_steps - len(no_cache_steps)}")
        
        if args.cache_type == 'deepcache':
            # DeepCache: 指定不 cache 的 block ID
            no_cache_block_id = {"double_blocks": infer_state.no_cache_block_id}
            cache_helper = DeepCacheHelper(
                double_blocks=transformer.double_blocks,
                no_cache_steps=no_cache_steps,
                no_cache_block_id=no_cache_block_id,
            )
            print_rank0(f"  no_cache_block_id: {infer_state.no_cache_block_id}")
        elif args.cache_type == 'teacache':
            cache_helper = TeaCacheHelper(
                double_blocks=transformer.double_blocks,
                no_cache_steps=no_cache_steps,
            )
        else:
            raise ValueError(f"未知的 cache 类型: {args.cache_type}")
        
        # 启用 cache helper（注册 hooks）
        cache_helper.enable()
        print_rank0(f"  ✓ Cache Helper 已启用")
    
    print_rank0(f"  ✓ Transformer 加载完成")
    print_rank0(f"  attn_mode: {transformer.attn_mode}")
    if args.attn_mode == 'flex-block-attn':
        print_rank0(f"  attn_param: {transformer.attn_param}")
    print_rank0(f"  use_meanflow: {transformer.config.use_meanflow}")
    print_rank0(f"  dtype: {transformer.dtype}")
    
    # 加载 scheduler
    scheduler_path = os.path.join(model_path, "scheduler")
    scheduler = FlowMatchDiscreteScheduler.from_pretrained(scheduler_path)
    
    # 获取 pipeline 配置
    pipeline_config = PIPELINE_CONFIGS.get(transformer_version, PIPELINE_CONFIGS['720p_t2v'])
    flow_shift = pipeline_config['flow_shift']
    default_guidance_scale = pipeline_config['guidance_scale']
    
    print_rank0(f"  flow_shift: {flow_shift}")
    print_rank0(f"  default_guidance_scale: {default_guidance_scale}")
    
    # 重建 scheduler with flow_shift
    scheduler = FlowMatchDiscreteScheduler(
        shift=flow_shift,
        reverse=True,
        solver="euler",
    )
    
    # ========================================================================
    # 设置参数
    # ========================================================================
    guidance_scale = args.guidance_scale
    seed = args.seed
    do_classifier_free_guidance = guidance_scale > 1.0
    use_meanflow = transformer.config.use_meanflow
    target_dtype = transformer_dtype
    
    # 计算分辨率
    height, width = get_closest_resolution(args.aspect_ratio, resolution)
    print_rank0(f"\n分辨率: {width}x{height}")
    
    # 设置随机种子（和 pipeline 一致）
    if get_parallel_state().sp_enabled:
        if dist.is_initialized():
            obj_list = [seed]
            group_src_rank = dist.get_global_rank(get_parallel_state().sp_group, 0)
            dist.broadcast_object_list(obj_list, src=group_src_rank, group=get_parallel_state().sp_group)
            seed = obj_list[0]
    
    generator = torch.Generator(device=torch.device('cpu')).manual_seed(seed)
    
    # 获取 latent 尺寸
    video_length = args.video_length
    latent_target_length, latent_height, latent_width = get_latent_size(video_length, height, width)
    n_tokens = latent_target_length * latent_height * latent_width
    
    print_rank0(f"Latent 尺寸: {latent_target_length}x{latent_height}x{latent_width}")
    print_rank0(f"Token 数量: {n_tokens}")
    
    # 设置 timesteps
    scheduler.set_timesteps(args.num_inference_steps, device=device, n_tokens=n_tokens)
    timesteps = scheduler.timesteps
    
    # 获取 multitask mask
    multitask_mask = get_task_mask(task_type, latent_target_length)
    
    # ========================================================================
    # 准备 embeddings（使用 Stage 1 的输出）
    # 格式与 TPU 版本一致：分开存储 positive/negative
    # ========================================================================
    print_rank0(f"\n准备 embeddings...")
    
    # 从 stage1 加载的 embeddings（使用与 TPU 一致的 key 名称）
    prompt_embeds = embeddings_dict['prompt_embeds'].to(device=device, dtype=transformer_dtype)
    negative_prompt_embeds = embeddings_dict['negative_prompt_embeds'].to(device=device, dtype=transformer_dtype)
    prompt_mask = embeddings_dict['prompt_embeds_mask'].to(device=device)
    negative_prompt_mask = embeddings_dict['negative_prompt_embeds_mask'].to(device=device)
    
    # ByT5 embeddings（与 TPU 版本一致的 key 名称）
    prompt_embeds_2 = embeddings_dict.get('prompt_embeds_2')
    negative_prompt_embeds_2 = embeddings_dict.get('negative_prompt_embeds_2')
    prompt_embeds_mask_2 = embeddings_dict.get('prompt_embeds_mask_2')
    negative_prompt_embeds_mask_2 = embeddings_dict.get('negative_prompt_embeds_mask_2')
    
    # 合并 CFG embeddings（LLM）
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        prompt_mask = torch.cat([negative_prompt_mask, prompt_mask])
    
    # 准备 byt5 embeddings（合并 CFG）
    extra_kwargs = {}
    if prompt_embeds_2 is not None:
        prompt_embeds_2 = prompt_embeds_2.to(device=device, dtype=torch.float32)
        prompt_embeds_mask_2 = prompt_embeds_mask_2.to(device=device)
        if do_classifier_free_guidance:
            negative_prompt_embeds_2 = negative_prompt_embeds_2.to(device=device, dtype=torch.float32)
            negative_prompt_embeds_mask_2 = negative_prompt_embeds_mask_2.to(device=device)
            byt5_text_states = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
            byt5_text_mask = torch.cat([negative_prompt_embeds_mask_2, prompt_embeds_mask_2])
        else:
            byt5_text_states = prompt_embeds_2
            byt5_text_mask = prompt_embeds_mask_2
        extra_kwargs = {
            "byt5_text_states": byt5_text_states,
            "byt5_text_mask": byt5_text_mask,
        }
    
    print_rank0(f"  prompt_embeds shape: {prompt_embeds.shape}")
    print_rank0(f"  prompt_mask shape: {prompt_mask.shape}")
    if byt5_text_states is not None:
        print_rank0(f"  byt5_text_states shape: {byt5_text_states.shape}")
    
    # prompt_embeds_2 = None（720p_t2v 不使用）
    prompt_embeds_2 = None
    
    # 准备 latents
    num_channels_latents = transformer.config.in_channels
    latents = prepare_latents(
        1,  # batch_size
        num_channels_latents,
        latent_height,
        latent_width,
        latent_target_length,
        target_dtype,
        device,
        generator,
    )
    
    # 准备 cond_latents
    cond_latents = prepare_cond_latents(task_type, None, latents, multitask_mask)
    
    # 准备 vision_states（t2v 模式使用零向量）
    # 从 transformer config 获取参数
    vision_num_tokens = 729  # 默认值
    vision_dim = 1152  # 默认值
    
    vision_states = torch.zeros(
        latents.shape[0],
        vision_num_tokens,
        vision_dim
    ).to(device=device, dtype=target_dtype)
    
    if do_classifier_free_guidance:
        vision_states = vision_states.repeat(2, 1, 1)
    
    print_rank0(f"  latents shape: {latents.shape}")
    print_rank0(f"  cond_latents shape: {cond_latents.shape}")
    print_rank0(f"  vision_states shape: {vision_states.shape}")
    
    # ========================================================================
    # Denoising Loop
    # ========================================================================
    
    print_rank0(f"\n开始 Transformer 推理...")
    print_rank0(f"  使用 Meanflow: {use_meanflow}")
    print_rank0(f"  使用 CFG: {do_classifier_free_guidance}")
    print_rank0(f"  SP 状态: sp_enabled={get_parallel_state().sp_enabled}, sp_size={get_parallel_state().sp}")
    
    # 打印 GPU 内存使用
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        print_rank0(f"  GPU 内存: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")
    
    start_time = time.perf_counter()
    num_inference_steps = len(timesteps)
    
    # 只在 rank 0 显示进度条，其他 rank 用 disable=True 禁用
    is_main_process = get_rank() == 0
    
    # 如果启用 cache，在 denoising 开始前清除缓存状态
    if cache_helper is not None:
        cache_helper.clear_states()
        print_rank0(f"  ✓ Cache 状态已清除")
    
    with torch.no_grad():
        # 使用 tqdm 包装 timesteps
        progress_bar = tqdm(
            enumerate(timesteps),
            total=num_inference_steps,
            desc="Denoising",
            disable=not is_main_process,  # 只在 rank 0 显示
            ncols=100,  # 固定宽度
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        for i, t in progress_bar:
            # ============================================================
            # 关键：设置 cache helper 的当前 timestep
            # cache helper 根据 cur_timestep 决定是否跳过某些 block 的计算
            # ============================================================
            if cache_helper is not None:
                cache_helper.cur_timestep = i
            # 更新进度条描述（显示当前 GPU 内存）
            if is_main_process and torch.cuda.is_available() and i % 10 == 0:
                allocated = torch.cuda.memory_allocated() / (1024**3)
                progress_bar.set_postfix({'GPU': f'{allocated:.1f}GB'})
            
            # 准备输入
            latents_concat = torch.concat([latents, cond_latents], dim=1)
            latent_model_input = torch.cat([latents_concat] * 2) if do_classifier_free_guidance else latents_concat
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            
            t_expand = t.repeat(latent_model_input.shape[0])
            
            # Meanflow timestep_r
            if use_meanflow:
                if i == len(timesteps) - 1:
                    timesteps_r = torch.tensor([0.0], device=device)
                else:
                    timesteps_r = timesteps[i + 1]
                timesteps_r = timesteps_r.repeat(latent_model_input.shape[0])
            else:
                timesteps_r = None
            
            # guidance（embedded guidance scale 为 None）
            guidance_expand = None
            
            # Transformer forward
            with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=True):
                output = transformer(
                    latent_model_input,
                    t_expand,
                    prompt_embeds,
                    prompt_embeds_2,
                    prompt_mask,
                    timestep_r=timesteps_r,
                    vision_states=vision_states,
                    mask_type=task_type,
                    guidance=guidance_expand,
                    return_dict=False,
                    extra_kwargs=extra_kwargs,
                )
                noise_pred = output[0]
            
            # CFG
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Scheduler step
            latents = scheduler.step(noise_pred, t, latents, generator=generator, return_dict=False)[0]
    
    # 禁用 cache helper（清理 hooks）
    if cache_helper is not None:
        cache_helper.disable()
        print_rank0(f"\n✓ Cache Helper 已禁用")
    
    elapsed = time.perf_counter() - start_time
    print_rank0(f"\n✓ Transformer 推理完成，耗时: {elapsed:.2f} 秒")
    print_rank0(f"  Latents shape: {latents.shape}")
    print_rank0(f"  Latents dtype: {latents.dtype}")
    
    # 保存 latents
    if get_rank() == 0:
        print_rank0(f"\n保存 latents 到: {output_paths['latents']}")
        metadata = {
            'height': str(height),
            'width': str(width),
            'video_length': str(video_length),
            'num_inference_steps': str(args.num_inference_steps),
            'guidance_scale': str(guidance_scale),
            'seed': str(seed),
            'elapsed_time': str(elapsed),
        }
        save_latents_to_safetensors(latents.cpu(), output_paths['latents'], metadata)
        
        # 更新配置
        config['height'] = height
        config['width'] = width
        config['stage2_elapsed_time'] = elapsed
        save_generation_config(config, output_paths['config'])
        
        print_rank0(f"\n{'='*60}")
        print_rank0("Stage 2 完成！")
        print_rank0(f"{'='*60}")
        print_rank0(f"输出: {output_paths['latents']}")
        print_rank0(f"下一步: 运行 stage3_vae_decoder.py")
    
    # 清理
    del transformer
    torch.cuda.empty_cache()
    
    print_rank0("\n✓ Stage 2 执行完成")


if __name__ == "__main__":
    main()