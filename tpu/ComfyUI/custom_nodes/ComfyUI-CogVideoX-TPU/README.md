# ComfyUI-CogVideoX-TPU

CogVideoX 文本到视频 (T2V) 生成的 ComfyUI 节点，专为 Google Cloud TPU 优化。

## 特性

- **TPU 原生加速**: 使用 JAX/torchax 在 TPU 上运行 CogVideoX 推理
- **Splash Attention**: 使用 exp2 优化的自定义 Pallas kernel，充分利用 TPU VPU 硬件
- **三阶段 Pipeline**: 文本编码、Transformer 去噪、VAE 解码分离，内存效率高
- **K-Smooth 优化**: 可选的 Key 平滑处理，提升注意力稳定性
- **CFG 并行**: DP=2 支持 CFG 正负 prompt 并行处理

## 节点列表

| 节点名称 | 功能 |
|---------|------|
| `CogVideoXTextEncoder` | 使用 T5 编码文本 prompt |
| `CogVideoXTPUSampler` | 在 TPU 上运行 Transformer 去噪 |
| `CogVideoXTPUVAEDecoder` | 解码 latents 为视频帧 |

## 工作流

```
TextEncoder → TPUSampler → TPUVAEDecoder → CreateVideo → SaveVideo
```

## 参数说明

### CogVideoXTextEncoder

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt` | STRING | - | 正面提示词 |
| `negative_prompt` | STRING | "" | 负面提示词 |
| `model_id` | STRING | "zai-org/CogVideoX1.5-5B" | HuggingFace 模型 ID |

### CogVideoXTPUSampler

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `embeddings` | COGVIDEOX_EMBEDS | - | TextEncoder 输出 |
| `height` | INT | 720 | 视频高度 |
| `width` | INT | 1280 | 视频宽度 |
| `num_frames` | INT | 81 | 视频帧数 (81 = ~5秒 @ 16fps) |
| `num_inference_steps` | INT | 50 | 采样步数 |
| `guidance_scale` | FLOAT | 6.0 | CFG 引导强度 |
| `seed` | INT | 42 | 随机种子 |
| `num_devices` | INT | 8 | TPU 设备数量 |

**注意**: `num_frames` 应满足 `(num_frames-1)/4+1` 为奇数，否则 VAE 解码会多出帧。
有效帧数: 41, 49, 57, 65, 73, 81, 89, 97...

### CogVideoXTPUVAEDecoder

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `latents` | COGVIDEOX_LATENTS | - | Sampler 输出 |
| `fps` | INT | 16 | 视频帧率 |

## 技术实现

### Splash Attention 优化

CogVideoX 使用自定义的 Splash Attention 实现，关键优化包括：

1. **exp2 替代 exp**: Query 预乘 `LOG2_E = 1.44269504`，使 `exp(x)` 变为 `exp2(x * LOG2_E)`，更好利用 TPU VPU 硬件
2. **Block 化计算**: 使用 Pallas kernel 进行高效的分块注意力计算
3. **K-Smooth**: 可选的 Key 平均值减法，提升数值稳定性

```python
# 默认 block sizes
BQSIZE = 3328      # Query 块大小
BKVSIZE = 2816     # Key/Value 块大小
BKVCOMPUTESIZE = 256  # KV 计算块大小
```

### 权重分片策略

Transformer 使用 Tensor Parallel (TP) 模式分片：

```python
TRANSFORMER_SHARDINGS_TP = {
    r'.*\.to_q\.weight$': (None, 'tp'),
    r'.*\.to_k\.weight$': (None, 'tp'),
    r'.*\.to_v\.weight$': (None, 'tp'),
    r'.*\.to_out.*\.weight$': ('tp', None),
    r'.*\.ff\.net\.0\.weight$': (None, 'tp'),
    r'.*\.ff\.net\.2\.weight$': ('tp', None),
}
```

## 依赖

- Python 3.10+
- JAX (TPU 版本)
- torchax
- diffusers (TPU 优化版)
- transformers

## 安装

```bash
# 复制到 ComfyUI custom_nodes 目录
cp -r ComfyUI-CogVideoX-TPU ~/ComfyUI/custom_nodes/

# 确保已安装 TPU 依赖
pip install jax[tpu] torchax

# 安装 diffusers-tpu（包含 TorchAx VAE）
git clone https://github.com/yangwhale/diffusers-tpu.git
cd diffusers-tpu && pip install -e .
```

## 使用示例

1. 启动 ComfyUI: `python main.py --cpu --listen 0.0.0.0`
2. 加载工作流 `examples/cogvideox_t2v_720p.json`
3. 输入提示词并执行

## 性能参考

在 TPU v6e-8 上的性能参考：

| 配置 | 分辨率 | 帧数 | 步数 | 时间 |
|------|--------|------|------|------|
| CogVideoX-1.5-5B | 1280x720 | 81 | 50 | ~3 min |

## 相关链接

- [CogVideoX 官方](https://github.com/THUDM/CogVideo)
- [diffusers-tpu](https://github.com/yangwhale/diffusers-tpu)
- [gpu-tpu-pedia](https://github.com/yangwhale/gpu-tpu-pedia)

## 许可证

Apache License 2.0
