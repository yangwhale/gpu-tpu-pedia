# CogVideoX 三阶段生成流程

本目录包含将 `generate_flax.py` 拆分后的三阶段生成流程，用于在 TPU 上运行 CogVideoX 视频生成。

## 目录结构

```
generate_diffusers_flax_staged/
├── utils.py                    # 共享工具模块
├── stage1_text_encoder.py      # 阶段1：文本编码
├── stage2_transformer.py       # 阶段2：Transformer 推理
├── stage3_vae_decoder.py       # 阶段3：VAE 解码
├── stage_outputs/              # 中间文件存储目录
│   ├── generation_config.json  # 生成配置
│   ├── stage1_embeddings.safetensors  # 文本 embeddings
│   ├── stage2_latents.safetensors     # 生成的 latents
│   └── output_video.mp4               # 最终视频
└── README.md                   # 本文档
```

## 使用方法

### 阶段1：文本编码

```bash
python stage1_text_encoder.py \
  --prompt "A panda playing guitar in a bamboo forest" \
  --output_dir ./stage_outputs
```

**参数：**
- `--prompt`: 正面提示词
- `--negative_prompt`: 负面提示词（可选）
- `--model_id`: 模型 ID
- `--output_dir`: 输出目录

**输出文件：**
- `stage1_embeddings.safetensors`: prompt embeddings
- `generation_config.json`: 生成配置（仅包含文本相关参数）

### 阶段2：Transformer 推理 (TPU)

```bash
python stage2_transformer.py \
  --input_dir ./stage_outputs \
  --num_inference_steps 10 \
  --height 768 \
  --width 1360 \
  --frames 81 \
  --warmup_steps 2
```

**参数：**
- `--num_inference_steps`: 推理步数（默认10）
- `--height`: 视频高度（默认768）
- `--width`: 视频宽度（默认1360）
- `--frames`: 视频帧数（默认81）
- `--guidance_scale`: CFG 引导尺度（默认6.0）
- `--seed`: 随机种子（默认42）
- `--fps`: 视频帧率（默认8，保存到配置供阶段3使用）
- `--warmup_steps`: 预热步数（默认2）

**输出文件：**
- `stage2_latents.safetensors`: 生成的 latents

### 阶段3：VAE 解码 (TPU)

```bash
python stage3_vae_decoder.py \
  --input_dir ./stage_outputs \
  --output_video ./stage_outputs/output_video.mp4
```

**输出文件：**
- `output_video.mp4`: 最终生成的视频

## 配置参数

### 默认模型配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| MODEL_NAME | `zai-org/CogVideoX1.5-5B` | 模型 ID |
| WIDTH | 1360 | 视频宽度 |
| HEIGHT | 768 | 视频高度 |
| FRAMES | 81 | 视频帧数 |
| FPS | 8 | 输出视频帧率 |
| NUM_STEPS | 10 | 推理步数 |
| GUIDANCE_SCALE | 6.0 | CFG 引导尺度 |

### Splash Attention 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| BQSIZE | 3328 | Query 块大小（与 Wan2.1 一致） |
| BKVSIZE | 2816 | Key/Value 块大小（与 Wan2.1 一致） |
| BKVCOMPUTESIZE | 256 | KV 计算块大小（与 Wan2.1 一致） |
| BKVCOMPUTEINSIZE | 256 | KV 内层计算块大小 |
| USE_K_SMOOTH | True | 是否使用 K 平滑 |
| USE_CUSTOM_ATTENTION | True | 是否使用 exp2 优化版 |

### Sharding 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| USE_DP | True | 是否使用数据并行（默认开启，性能提升 35%） |
| SP_NUM | 1 | 序列并行数量 |
| USE_FSDP | True | 是否使用 FSDP 模式 |

## 数据流

```
prompt (文本)
    ↓
[Stage 1: Text Encoder]
    ↓
embeddings.safetensors
    ↓
[Stage 2: Transformer + Splash Attention]
    ↓
latents.safetensors
    ↓
[Stage 3: Flax VAE Decoder]
    ↓
output_video.mp4
```

## 依赖项

- JAX/Flax (TPU 支持)
- PyTorch (模型加载)
- torchax (PyTorch-JAX 互操作)
- diffusers (CogVideoX Pipeline)
- transformers (T5 Text Encoder)
- safetensors (数据存储)
- imageio (视频导出)

## 与原版 generate_flax.py 的对比

| 特性 | generate_flax.py | 三阶段版本 |
|------|-----------------|------------|
| 内存使用 | 全部加载 | 分阶段加载 |
| 调试便利性 | 困难 | 可单独调试各阶段 |
| 中间结果 | 不保存 | 保存为 SafeTensors |
| 重试灵活性 | 需从头开始 | 可从任意阶段重试 |

## 性能基准测试

### 测试环境
- **硬件**: TPU v4-8 (8 chips)
- **模型**: CogVideoX1.5-5B
- **分辨率**: 768 × 1360
- **帧数**: 81
- **推理步数**: 10
- **测试日期**: 2024-12-12

### Stage 2 优化总表（CFG 模式，guidance_scale=6.0）

| # | 优化项 | DP | Mesh 顺序 | Block Size | sharding constraint | 每步时间 | 10步总时间 | 相对基线 |
|---|--------|-----|-----------|------------|---------------------|----------|-----------|---------|
| 1 | 基线版本 | ✗ | `('tp', 'dp', 'sp')` | 原始 | ✗ | 4.04s | 40.54s | - |
| 2 | +sharding constraint | ✗ | `('tp', 'dp', 'sp')` | 原始 | ✓ | 3.93s | 39.43s | +2.7% |
| 3 | +DP | ✓ | `('tp', 'sp', 'dp')` | 原始 | ✓ | 3.08s | 33.90s | +16.4% |
| 4 | +DP + mesh优化 | ✓ | `('dp', 'sp', 'tp')` | 原始 | ✓ | 2.75s | 30.26s | +25.4% |
| 5 | **最优配置** | ✓ | `('dp', 'sp', 'tp')` | Wan2.1 | ✓ | **2.31s** | **25.36s** | **+37.4%** |

### No-CFG 模式（guidance_scale=1.0）

| # | 分辨率 | 帧数 | DP | Block Size | 每步时间 | 10步总时间 | 说明 |
|---|--------|------|-----|------------|----------|-----------|------|
| 6 | 768×1360 | 81 | ✗ | Wan2.1 | 1.50s | 16.55s | 单 batch，无需 DP |
| 7 | **640×1280** | **61** | ✗ | Wan2.1 | **0.71s** | **7.86s** | 低分辨率更快 |

> **注意**: No-CFG 模式下 batch_size=1（无 negative prompt 分支），无法使用 DP 分片。
> 相比 CFG 模式（25.36s），No-CFG 768×1360×81 快约 **35%**，因为计算量减半。

> **Block Size 配置对比**:
> - 原始: BQSIZE=2048, BKVSIZE=1024, BKVCOMPUTESIZE=512
> - Wan2.1: BQSIZE=3328, BKVSIZE=2816, BKVCOMPUTESIZE=256

### 各优化项增量效果

| 优化项 | 效果 | 说明 |
|--------|------|------|
| sharding constraint | +2.7% | 在 attention 输出后添加 `jax.lax.with_sharding_constraint()` |
| Data Parallelism (DP) | +14.0% | 使用 `--use_dp` 参数，dp_dim=2, tp_dim=4 |
| Mesh 顺序优化 | +10.7% | 从 `('tp', 'sp', 'dp')` 改为 `('dp', 'sp', 'tp')` |
| Wan2.1 Block Size | +16.2% | 更大的 Q/KV 块 + 更小的计算块 |
| **累计优化** | **+37.4%** | 从 40.54s 降至 25.36s |

### 技术要点

1. **sharding constraint**
   ```python
   out = jax.lax.with_sharding_constraint(out, P('dp', None, ('tp', 'sp'), None))
   ```

2. **env.config 配置**
   - 必须直接设置 `env.config.use_tpu_splash_attention = True`
   - 使用 `hasattr` 检查会导致 Splash Attention 不生效

3. **Block Size 选择原则**
   - 更大的 BQSIZE/BKVSIZE 提高并行度
   - 更小的 BKVCOMPUTESIZE 优化内存访问

### 预热与推理时间

| 配置 | Warmup (2步) | 每步时间 |
|------|-------------|---------|
| 无 DP | ~120s | ~3.5-4.0s |
| 有 DP（原始 block size） | ~130s | ~2.7-3.1s |
| 有 DP（Wan2.1 block size） | ~137s | ~2.3-2.7s |

### 全流程时间参考

| 阶段 | 基线配置 | 最优配置 | 说明 |
|------|---------|---------|------|
| Stage 1 (Text Encoder) | ~2s | ~2s | CPU 上运行 |
| Stage 2 (Transformer) | ~40s | **~25s** | 不含预热，10步 |
| Stage 2 预热 | ~120s | ~137s | 含 JIT 编译 |
| Stage 3 (VAE Decoder) | ~90s | ~90s | Flax VAE |
| **总计（首次运行）** | ~252s | ~254s | 含编译 |
| **总计（后续运行）** | ~132s | **~117s** | 无需编译 |

## 注意事项

1. **阶段2 依赖 custom_splash_attention.py**：确保父目录中存在此文件
2. **阶段3 依赖 FlaxAutoencoderKLCogVideoX**：需要安装修改版 diffusers
3. **TPU 内存**：各阶段分开运行可降低峰值内存使用