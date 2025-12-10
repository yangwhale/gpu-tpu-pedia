# HunyuanVideo-1.5 TPU 推理（原生版本）

在 TPU v6e-8 上使用原生 HunyuanVideo-1.5-TPU 代码库运行视频生成。

## 🚀 快速开始

```bash
# 1. 确保已运行 Stage 1 生成 embeddings
#    使用 ../generate_diffusers_flax_staged/stage1_text_encoder.py

# 2. 运行 Transformer 推理
python stage2_transformer.py \
    --input_dir ./stage_outputs \
    --video_length 121 \
    --num_inference_steps 50 \
    --warmup_steps 2
```

## 📊 性能数据

**环境**：TPU v6e-8，121帧 720p，50步

| 模式 | 每步时间 | 总时间 | 加速比 |
|------|----------|--------|--------|
| 标准 TP | 8.12s | 6.8 分钟 | 1.0x |
| **TP + fc2 Replicated (默认)** | **7.29s** | **6.1 分钟** | **1.11x** |
| TP + DeepCache | ~4s | ~3.5 分钟 | ~2x |

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