# CogVideoX TPU 加速项目

本项目实现了 CogVideoX 视频生成模型在 Google Cloud TPU 上的高性能推理，通过 JAX + torchax 实现了显著的性能提升和内存优化。

## 📋 项目概述

CogVideoX 是一个强大的文本到视频生成模型，本项目将其迁移到 TPU 平台，利用以下技术实现高效推理：

- **JAX/torchax 框架**：替换 PyTorch，充分利用 TPU 的 XLA 编译优化
- **Splash Attention**：TPU 原生的高效注意力机制，支持长序列处理
- **Flax VAE**：原生 JAX 实现的 VAE 解码器，解决 OOM 问题并支持长视频生成
- **模型分片**：智能的权重分片策略（FSDP/Tensor Parallel），支持多 TPU 并行
- **BFloat16 优化**：全流程 BF16 计算，减少内存占用并提升性能

## 🚀 Quick Start

### 1. 环境安装

```bash
# 安装核心依赖
pip install huggingface-hub
pip install -U transformers datasets evaluate accelerate timm flax numpy
pip install torchax
pip install jax[tpu]
pip install tensorflow-cpu

# 安装辅助工具
pip install sentencepiece
sudo apt install ffmpeg -y
pip install imageio[ffmpeg]
pip install tpu-info
pip install matplotlib
```

### 2. 配置环境变量

```bash
# 设置 Hugging Face 缓存目录（使用共享内存加速）
export HF_HOME=/dev/shm

# 设置 Hugging Face Token（用于下载模型）
export HF_TOKEN=<your HF_TOKEN>
```

> **提示**：从 [Hugging Face Settings](https://huggingface.co/settings/tokens) 获取你的 API Token

### 3. 克隆并安装项目

```bash
# 克隆 diffusers-tpu 项目（包含 Flax VAE 实现）
git clone https://github.com/yangwhale/diffusers-tpu.git
cd diffusers-tpu
pip install -e .

# 克隆本项目
git clone https://github.com/yangwhale/gpu-tpu-pedia.git
cd gpu-tpu-pedia/tpu/cogvideo/
```

### 4. 运行视频生成

```bash
python generate_torchax.py
```

生成的视频将保存为 `output_video_torchax_vae.mp4`。

## 📁 项目结构

```
cogvideo/
├── README.md                     # 本文档
├── generate_torchax.py           # ⭐ 主程序：完整的 TPU 视频生成流程
├── generate_gpu.py               # GPU PyTorch 版本（参考）
├── vae_decode_flax.py            # Flax VAE 解码测试
├── vae_decode_gpu.py             # GPU PyTorch VAE 解码测试（参考）
└── output_video_torchax_vae.mp4  # 生成的视频示例
```

## 🎯 核心功能

### 1. Splash Attention 优化

[`generate_torchax.py`](generate_torchax.py:170-286) 实现了 TPU 专用的 Splash Attention：

```python
# 配置参数（可根据需求调整）
BQSIZE = 2048           # Query 块大小
BKVSIZE = 1024          # Key/Value 块大小
BKVCOMPUTESIZE = 512    # Key/Value 计算块大小
WINDOW_SIZE = None      # 窗口大小（None = 全局注意力）
USE_K_SMOOTH = True     # K-smooth 优化
```

**特性**：
- 块状计算，避免 VMEM 溢出
- 支持局部窗口注意力（减少计算量）
- K-smooth 技术提升数值稳定性
- 自动处理序列填充

### 2. 智能权重分片

支持两种分片模式：

#### FSDP 模式（推荐，默认）
```python
USE_FSDP = True

# Attention 层在输出维度分片
r'.*\.to_out.*\.weight$': (('tp', 'sp'), None)
```

#### Tensor Parallel 模式
```python
USE_FSDP = False

# Attention 层在输入维度分片
r'.*\.to_q\.weight$': (('tp', 'sp'), None)
```

**权重分片函数**：
- [`shard_weights_transformer()`](generate_torchax.py:414-449)：Transformer 模型分片
- [`shard_weights_text_encoder()`](generate_torchax.py:452-479)：T5 文本编码器分片
- [`shard_weights_vae()`](generate_torchax.py:482-503)：VAE 权重分片（当前复制模式）

### 3. Flax VAE 集成

[`FlaxVAEProxy`](generate_torchax.py:637-689) 类实现了 PyTorch 到 Flax VAE 的无缝切换：

**关键优化**：
- 全流程 BF16 计算，避免中间 FP32 数组
- 逐帧解码，支持长视频（避免 OOM）
- 内存高效的数据转换（使用 numpy view）
- 可选的 Tiling 解码（处理超高分辨率）

```python
# 在 Pipeline 中替换 VAE
flax_vae = FlaxAutoencoderKLCogVideoX.from_pretrained(
    model_id, subfolder="vae", dtype=jnp.bfloat16
)
pipe.vae = FlaxVAEProxy(flax_vae)
```

### 4. 完整的 Pipeline 设置

[`setup_pipeline_for_jax()`](generate_torchax.py:506-634) 函数执行完整的 TPU 配置：

1. **创建设备网格**：支持 TP/DP/SP 三维并行
2. **注册自定义算子**：Splash Attention 替换标准 SDPA
3. **权重迁移和分片**：
   - Transformer → XLA + 分片
   - Text Encoder → XLA + 分片
   - VAE → Flax 原生实现
4. **JIT 编译**：编译 Transformer 和 Text Encoder

## ⚙️ 配置参数

### Mesh 分片配置

```python
USE_DP = False          # Data Parallelism（多 batch 并行）
SP_NUM = 1              # Spatial Parallelism（空间维度并行）
USE_FSDP = True         # FSDP vs Tensor Parallel 模式
```

**设备分配示例**（8 TPU cores）：
- 默认：`(tp=8, dp=1, sp=1)` - 纯 Tensor Parallel
- `USE_DP=True`：`(tp=4, dp=2, sp=1)` - TP + DP 混合
- `SP_NUM=2`：`(tp=4, dp=1, sp=2)` - TP + SP 混合

### VAE Tiling 配置

处理超大分辨率视频时可启用（当前禁用用于测试）：

```python
flax_vae.enable_tiling(
    tile_sample_min_height=192,      # 瓦片最小高度
    tile_sample_min_width=340,       # 瓦片最小宽度
    tile_overlap_factor_height=1/6,  # 高度重叠因子
    tile_overlap_factor_width=1/5,   # 宽度重叠因子
)
```

## 📊 性能特性

### 1. JIT 编译加速

- **第一次运行**：包含 JIT 编译（较慢）
- **后续运行**：使用缓存的编译结果（快速）
- **典型加速比**：2-5x（取决于模型大小）

### 2. 内存优化

- **BF16 计算**：相比 FP32 减少 50% 内存
- **逐帧 VAE 解码**：避免大视频的 OOM
- **高效数据转换**：使用 numpy view，避免拷贝

### 3. 并行策略

支持三种并行维度的灵活组合：
- **Tensor Parallel (TP)**：跨设备分片模型权重
- **Data Parallel (DP)**：并行处理多个 batch
- **Spatial Parallel (SP)**：空间维度并行（高分辨率）

## 🎬 使用示例

### 基础视频生成

```python
from diffusers import CogVideoXPipeline

# 加载模型
pipe = CogVideoXPipeline.from_pretrained("zai-org/CogVideoX1.5-5B")

# 配置为 TPU
pipe, env, mesh = setup_pipeline_for_jax(pipe)

# 生成视频
prompt = "A cat walks on the grass, realistic style."
with mesh, env:
    result = pipe(
        prompt,
        num_inference_steps=50,
        num_frames=64,
        height=768,
        width=1360
    )
    frames = result.frames[0]

# 保存视频
import imageio
imageio.mimsave('output.mp4', frames, fps=8)
```

### 性能基准测试

使用 [`run_generation_benchmark()`](generate_torchax.py:692-739) 函数：

```python
frames, times = run_generation_benchmark(
    pipe,
    prompt="A dog cooking cake in the kitchen",
    num_inference_steps=50,
    num_frames=64,
    height=768,
    width=1360,
    num_iterations=2
)

# 打印性能摘要
print_performance_summary(times)
```

输出示例：
```
总迭代次数: 2
第一次运行（含编译）: 45.2341 秒
后续运行平均时间: 18.5672 秒
加速比: 2.44x
```

## 🔧 故障排查

### 1. VMEM 溢出

**症状**：`RESOURCE_EXHAUSTED: Ran out of memory in memory space vmem`

**解决方案**：减小 Splash Attention 块大小
```python
BQSIZE = 1024        # 从 2048 减小
BKVSIZE = 512        # 从 1024 减小
BKVCOMPUTESIZE = 256 # 从 512 减小
```

### 2. OOM 错误

**症状**：内存不足错误

**解决方案**：
1. 启用 VAE Tiling（处理大视频）
2. 减少视频帧数或分辨率
3. 使用 Data Parallel 模式分散内存

### 3. 编译缓存

启用 JAX 编译缓存以加速重复运行：

```python
jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
```

## 📚 相关资源

- [CogVideoX 官方仓库](https://github.com/THUDM/CogVideo)
- [Diffusers TPU 版本](https://github.com/yangwhale/diffusers-tpu)
- [JAX 文档](https://jax.readthedocs.io/)
- [Flax 文档](https://flax.readthedocs.io/)
- [TPU 开发指南](https://cloud.google.com/tpu/docs)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目遵循原始项目的许可证。

## 🙏 致谢

- CogVideoX 团队
- Hugging Face Diffusers 团队
- Google JAX/Flax 团队
- TPU 社区

---

**最后更新**：2025-11-04