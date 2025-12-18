# CogVideoX 三阶段生成流程

本目录包含将 `generate_flax.py` 拆分后的三阶段生成流程，用于在 TPU 上运行 CogVideoX 视频生成。

## 目录结构

```
generate_diffusers_torchax_staged/
├── utils.py                    # 共享工具模块
├── stage1_text_encoder.py      # 阶段1：文本编码
├── stage2_transformer.py       # 阶段2：Transformer 推理
├── stage3_vae_decoder.py       # 阶段3：VAE 解码 (TorchAx 版本，推荐)
├── stage3_vae_decoder_flax.py  # 阶段3：VAE 解码 (Flax 版本，备用)
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

**推荐使用 TorchAx VAE（速度更快）：**

```bash
python stage3_vae_decoder.py \
  --input_dir ./stage_outputs \
  --output_video ./stage_outputs/output_video.mp4
```

> **性能对比**：TorchAx VAE 解码仅需 **~2.4秒**（不含 JIT 预热），比 Flax VAE 的 ~90秒快 **37倍**！

**备用 Flax VAE：**

```bash
python stage3_vae_decoder_flax.py \
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
[Stage 3: TorchAx VAE Decoder] ← 推荐！快 37x
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
- **TPU**: v6e-8 / v6e-4 / v6e-1
- **GPU**: NVIDIA H200
- **模型**: CogVideoX1.5-5B
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

| # | 硬件 | 分辨率 | 帧数 | DP | Block Size | 每步时间 | 10步总时间 | vs 1卡加速 |
|---|------|--------|------|-----|------------|----------|-----------|-----------|
| 6 | v6e-8 | 768×1360 | 81 | ✗ | Wan2.1 | 1.50s | 16.55s | - |
| 7 | v6e-8 | 640×1280 | 61 | ✗ | Wan2.1 | 0.71s | 7.86s | 3.83x |
| 8 | v6e-4 | 640×1280 | 61 | ✗ | Wan2.1 | 1.07s | 11.73s | 2.56x |
| 9 | **v6e-1** | **640×1280** | **61** | ✗ | Wan2.1 | **2.73s** | **30.07s** | **1.00x** |

> **注意**: No-CFG 模式下 batch_size=1（无 negative prompt 分支），无法使用 DP 分片。
> 相比 CFG 模式（25.36s），No-CFG 768×1360×81 快约 **35%**，因为计算量减半。

**Scaling 分析 (640×1280×61):**
- 8 卡 vs 1 卡: 3.83x 加速（理论 8x，效率 48%）
- 4 卡 vs 1 卡: 2.56x 加速（理论 4x，效率 64%）
- 8 卡 vs 4 卡: 1.49x 加速（理论 2x，效率 75%）

> Scaling 效率低于理论值是预期的，因为 Transformer 是 memory-bound 工作负载，通信开销随芯片数增加。

### 单芯片 Baseline 对比（640×1280×61, No-CFG）

| 硬件 | Attention 实现 | 每步时间 | 相对速度 | 相对价格* | 性价比* |
|------|---------------|----------|----------|----------|---------|
| **H200** | CUDA FlashAttention | **2.024s** | 1.00x | 2.00x | 1.00x |
| v6e-1 | 原始 Splash Attention | 3.38s | 0.60x | 1.00x | 0.60x |
| v6e-1 | Custom Splash (exp2) | 2.73s | 0.74x | 1.00x | **1.48x** |

*\*价格假设：H200 = 2 × v6e-1（仅供参考）*

**TPU vs GPU 分析:**
- v6e-1 (exp2 优化) vs H200: 2.73s vs 2.024s，TPU 慢 35%
- v6e-1 (原始 Splash) vs H200: 3.38s vs 2.024s，TPU 慢 67%
- Custom Splash Attention (exp2) 对 TPU 性能至关重要，提升 **19%** (3.38s → 2.73s)

**性价比计算（假设 H200 价格 = 2 × v6e-1）:**

| 硬件 | 相对速度 | 相对价格 | 性价比 (速度/价格) | vs H200 |
|------|----------|----------|-------------------|---------|
| H200 | 1.00 | 2.00 | 0.50 | 1.00x |
| **v6e-1 (exp2)** | 0.74 | 1.00 | **0.74** | **1.48x** |

> 结论：v6e-1 性价比比 H200 高 **48%**，即使绝对性能较慢，同等成本下能处理更多任务。

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
| Stage 3 (TorchAx VAE) | - | **~2.4s** | 推荐！ |
| Stage 3 (Flax VAE) | ~90s | ~90s | 备用 |
| Stage 3 预热 (TorchAx) | - | ~126s | 首次 JIT 编译 |
| **总计（首次运行）** | ~252s | **~166s** | 含编译，使用 TorchAx VAE |
| **总计（后续运行）** | ~132s | **~29s** | 无需编译，使用 TorchAx VAE |

> **2024-12-18 更新**: Stage 3 新增 TorchAx VAE 实现，解码速度从 ~90s 降至 **~2.4s**！
> 总流程时间（后续运行）从 ~117s 降至 **~29s**，整体提速 **4倍**！

## 技术细节

### Latents 维度格式

| 阶段 | 格式 | 说明 |
|------|------|------|
| Pipeline 输出 | `[B, T, C, H, W]` | `output_type='latent'` 返回的原始格式 |
| **stage2 保存** | `[B, C, T, H, W]` | **PyTorch 标准格式** |
| Flax VAE 输入 | `[B, T, H, W, C]` | JAX channel-last 格式 |

stage2 在保存前会：
1. `permute(0, 2, 1, 3, 4)` - 转换为标准格式
2. 裁剪 `additional_frames` - CogVideoX-1.5 的 patch_size_t 填充

### prepare_video_for_export 输出格式

- 返回 `List[np.ndarray]`，每帧为 `float32` 数组，范围 `[0, 1]`
- 与 `diffusers.utils.export_to_video` 兼容
- `export_to_video` 会自动 `* 255` 并转换为 `uint8`

## 注意事项

1. **阶段2 依赖 custom_splash_attention.py**：确保父目录中存在此文件
2. **阶段3 推荐使用 TorchAx VAE**：`stage3_vae_decoder.py`，比 Flax 快 37 倍
3. **阶段3 备用 Flax VAE**：`stage3_vae_decoder_flax.py`，依赖 `FlaxAutoencoderKLCogVideoX`
4. **TorchAx VAE 依赖**：需要安装修改版 diffusers (`diffusers-tpu`)，导入 `autoencoder_kl_cogvideox_torchax`
5. **TPU 内存**：各阶段分开运行可降低峰值内存使用
6. **视频导出**：使用 `export_to_video` 而非 `imageio.mimsave`，配合 `prepare_video_for_export` 使用

## VAE 版本对比

| 特性 | TorchAx VAE | Flax VAE |
|------|-------------|----------|
| 脚本 | `stage3_vae_decoder.py` | `stage3_vae_decoder_flax.py` |
| 解码时间 | **~2.4s** | ~90s |
| JIT 预热 | ~126s | - |
| 速度提升 | **37x** | 基线 |
| 内存使用 | 相当 | 相当 |
| 推荐使用 | ✅ **推荐** | 备用 |