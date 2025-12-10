# HunyuanVideo-1.5 TPU 推理（原生版本）

在 TPU v6e-8 上使用原生 HunyuanVideo-1.5-TPU 代码库运行视频生成。

## 🔄 完整工作流（TPU + GPU 协作）

本目录只包含 **Stage 2: Transformer 推理**，需要与其他阶段配合使用：

```
┌─────────────────────────────────────────────────────────────────────┐
│                        完整 Pipeline 流程                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────── TPU 机器 ───────────────────┐                 │
│  │                                                │                 │
│  │  Stage 1: Text Encoder (CPU)                   │                 │
│  │  ├─ 目录: ../generate_diffusers_flax_staged/   │                 │
│  │  ├─ 脚本: stage1_text_encoder.py               │                 │
│  │  └─ 输出: stage_outputs/                       │                 │
│  │           ├─ stage1_embeddings.safetensors     │                 │
│  │           └─ generation_config.json            │                 │
│  │                         ↓                      │                 │
│  │            复制到本目录（同一机器）              │                 │
│  │                         ↓                      │                 │
│  │  Stage 2: Transformer (TPU) ← 本目录           │                 │
│  │  ├─ 目录: ./generate_hunyuan_flax_staged/      │                 │
│  │  ├─ 脚本: stage2_transformer.py                │                 │
│  │  └─ 输出: stage_outputs/stage2_latents.safetensors               │
│  │                                                │                 │
│  └────────────────────────────────────────────────┘                 │
│                              ↓                                      │
│                     传输到 GPU 机器                                  │
│                              ↓                                      │
│  ┌─────────────────── GPU 机器 ───────────────────┐                 │
│  │                                                │                 │
│  │  Stage 3: VAE Decoder (GPU)                    │                 │
│  │  ├─ 目录: ../generate_hunyuan_gpu_staged/      │                 │
│  │  ├─ 脚本: run_stage3.sh                        │                 │
│  │  └─ 输出: stage_outputs/output_video.mp4       │                 │
│  │                                                │                 │
│  └────────────────────────────────────────────────┘                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 完整操作步骤

**1. 在 TPU 机器上运行 Stage 1（CPU 执行）**

```bash
# TPU 机器上（使用 CPU 运行 Text Encoder）
cd ~/gpu-tpu-pedia/tpu/HunyuanVideo-1.5/generate_diffusers_flax_staged

# 运行 Text Encoder
python stage1_text_encoder.py --prompt "A beautiful sunset over the ocean"

# 检查输出
ls stage_outputs/
# → stage1_embeddings.safetensors, generation_config.json
```

**2. 复制 Stage 1 输出到本目录（同一机器）**

```bash
# 在 TPU 机器上
cp -r ~/gpu-tpu-pedia/tpu/HunyuanVideo-1.5/generate_diffusers_flax_staged/stage_outputs \
      ~/gpu-tpu-pedia/tpu/HunyuanVideo-1.5/generate_hunyuan_flax_staged/
```

**3. 在 TPU 机器上运行 Stage 2（TPU 执行）**

```bash
# TPU 机器上
cd ~/gpu-tpu-pedia/tpu/HunyuanVideo-1.5/generate_hunyuan_flax_staged

# 运行 Transformer 推理
python stage2_transformer.py \
    --input_dir ./stage_outputs \
    --video_length 121 \
    --num_inference_steps 50 \
    --warmup_steps 2

# 检查输出
ls stage_outputs/
# → stage2_latents.safetensors
```

**4. 将 Stage 2 输出传到 GPU 机器**

```bash
# 从 TPU 机器传输到 GPU 机器
scp stage_outputs/stage2_latents.safetensors gpu-machine:~/gpu-tpu-pedia/tpu/HunyuanVideo-1.5/generate_hunyuan_gpu_staged/stage_outputs/

# 同时传输 generation_config.json（Stage 3 需要）
scp stage_outputs/generation_config.json gpu-machine:~/gpu-tpu-pedia/tpu/HunyuanVideo-1.5/generate_hunyuan_gpu_staged/stage_outputs/
```

**5. 在 GPU 机器上运行 Stage 3**

```bash
# GPU 机器上
cd ~/gpu-tpu-pedia/tpu/HunyuanVideo-1.5/generate_hunyuan_gpu_staged

# 运行 VAE Decoder
bash run_stage3.sh

# 查看生成的视频
ls stage_outputs/output_video.mp4
```

---

## 🚀 快速开始（本目录 Stage 2）

```bash
# 前提：stage_outputs/ 目录已包含 Stage 1 的输出文件

# 运行 Transformer 推理
python stage2_transformer.py \
    --input_dir ./stage_outputs \
    --video_length 121 \
    --num_inference_steps 50 \
    --warmup_steps 2
```

## 📊 性能数据

### TPU 性能（本项目）

**环境**：TPU v6e-8，121帧 720p，50步

| 模式 | 每步时间 | 总时间 | 加速比 |
|------|----------|--------|--------|
| 标准 TP | 8.12s | 6.8 分钟 | 1.0x |
| **TP + fc2 Replicated (默认)** | **7.29s** | **6.1 分钟** | **1.11x** |
| TP + DeepCache | ~4s | ~3.5 分钟 | ~2x |

### GPU 性能对比（Baseline）

**环境**：NVIDIA H100 × 8，121帧 720p，50步

| 日期 | 分辨率 | 帧数 | Step Time | CFG_DISTILLED | SAGE_ATTN | ENABLE_CACHE | 备注 |
|------|--------|------|-----------|---------------|-----------|--------------|------|
| 2025-12-03 | 720p | 121 | 5.10-5.11s | false | false | false | 基础配置 |
| 2025-12-03 | 720p | 121 | 5.14-5.15s | false | false | true | ENABLE_CACHE 开启 |
| 2025-12-03 | 480p | 121 | 1.47-1.48s | false | false | false | 480p 基础配置 |
| 2025-12-03 | 480p | 121 | 0.877-0.878s | true | false | false | CFG_DISTILLED 开启 |
| 2025-12-03 | 720p | 121 | ~2.74s | false | false | false | guidance_scale=1.0 |
| 2025-12-03 | 720p | 121 | **1.67s** | false | true | false | **SageAttention，1.31x 加速** ⚡ |

### TPU vs GPU 对比

| 平台 | 配置 | 720p 121帧 Step Time | 备注 |
|------|------|---------------------|------|
| GPU H100 × 8 | Flash Attention 2 | 5.10s | GPU 基线 |
| GPU H100 × 8 | SageAttention | 1.67s | GPU 最快（有损） |
| **TPU v6e-8** | **TP + fc2 Replicated** | **7.29s** | TPU 默认配置 |
| TPU v6e-8 | TP + DeepCache | ~4s | TPU + 缓存加速 |

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `stage2_transformer.py` | 主推理脚本（支持 DeepCache） |
| `utils.py` | 工具函数（加载/保存 safetensors） |
| `run_stage2.sh` | 运行脚本 |
| `TORCHAX_MIGRATION_GUIDE.md` | ⭐ GPU→TPU 迁移完整指南 |
| `GPU_TPU_COMPARISON.md` | GPU/TPU 代码对比 |
| `DEEPCACHE_EXPLAINED.md` | DeepCache 原理说明 |

## ⚙️ 参数说明

### 基本参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input_dir` | `./stage_outputs` | Stage 1 输出目录 |
| `--output_dir` | 同 input_dir | 输出目录 |
| `--video_length` | 49 | 视频帧数（49≈2秒, 121≈5秒） |
| `--num_inference_steps` | 50 | 推理步数 |
| `--guidance_scale` | 6.0 | CFG 引导尺度 |
| `--seed` | 42 | 随机种子 |
| `--warmup_steps` | 2 | 预热步数（触发 JIT 编译） |

### DeepCache 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable_cache` | False | 启用 DeepCache |
| `--cache_start_step` | 11 | 开始使用缓存的步数 |
| `--cache_end_step` | 45 | 停止使用缓存的步数 |
| `--cache_step_interval` | 4 | 缓存刷新间隔 |

### Profiler 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable_profiler` | False | 启用 JAX Profiler |
| `--profiler_output_dir` | `/dev/shm/jax-trace` | Profiler 输出目录 |

## 📖 使用示例

### 基本使用

```bash
# 49帧视频（约2秒）
python stage2_transformer.py --video_length 49 --num_inference_steps 50

# 121帧视频（约5秒）
python stage2_transformer.py --video_length 121 --num_inference_steps 50
```

### 使用 DeepCache 加速

```bash
# 启用 DeepCache（+70% 速度，质量稍降）
python stage2_transformer.py \
    --enable_cache \
    --video_length 121 \
    --num_inference_steps 50

# 自定义 cache 参数
python stage2_transformer.py \
    --enable_cache \
    --cache_start_step 15 \
    --cache_end_step 40 \
    --cache_step_interval 3
```

### 性能分析

```bash
# 抓取 3 步的 profiler
python stage2_transformer.py \
    --enable_profiler \
    --num_inference_steps 3 \
    --warmup_steps 2
```

## 🔧 技术架构

### 权重分片策略

默认使用 **TP + fc2/proj Replicated** 策略：

```python
# Column Parallel（Q/K/V, fc1）- 输出维度分片
r'.*\.img_attn_q\.weight$': (('tp', 'sp'), None)

# REPLICATED（fc2, proj）- 完全复制，无 all-reduce
r'.*\.img_attn_proj\.weight$': (None, None)
r'.*\.img_mlp\.fc2\.weight$': (None, None)
```

**为什么这样设计**：
- Row Parallel 层（fc2, proj）原本需要 all-reduce
- 复制这些层可消除 all-reduce，提升 10.2% 性能
- 额外 HBM 开销：~12 GB

### DeepCache 原理

跳过 transformer 中间层的计算，复用上一步的输出：

```
完整 forward: 54 double_blocks → 7.29s
缓存 forward: 1 double_block → ~0.5s

加速比: 50% cache hit → ~1.8x
```

详见 [DEEPCACHE_EXPLAINED.md](DEEPCACHE_EXPLAINED.md)

## 📚 技术文档

| 文档 | 内容 |
|------|------|
| [TORCHAX_MIGRATION_GUIDE.md](TORCHAX_MIGRATION_GUIDE.md) | GPU→TPU 迁移完整指南 |
| [GPU_TPU_COMPARISON.md](GPU_TPU_COMPARISON.md) | GPU/TPU 代码对比 |
| [DEEPCACHE_EXPLAINED.md](DEEPCACHE_EXPLAINED.md) | DeepCache 原理说明 |

## ❓ 常见问题

### 1. 首次运行很慢（60s+）

这是 XLA/JAX 编译造成的，正常现象。使用 `--warmup_steps 2` 进行预热。

```bash
# 清除编译缓存（如需重新编译）
rm -rf /dev/shm/jax_cache
```

### 2. OOM 内存不足

- 减少 `--video_length`（建议 49）
- 启用 DeepCache 减少峰值内存

### 3. 缺少 Stage 1 embeddings

需要先运行 Stage 1 生成 embeddings：

```bash
cd ../generate_diffusers_flax_staged
python stage1_text_encoder.py --prompt "Your prompt"
cp -r stage_outputs ../generate_hunyuan_flax_staged/
```

## 🔗 依赖

- JAX (TPU)
- torchax
- HunyuanVideo-1.5-TPU（`~/HunyuanVideo-1.5-TPU`）
- safetensors

## 📝 License

与 HunyuanVideo 项目保持一致。