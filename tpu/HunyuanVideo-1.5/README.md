# HunyuanVideo-1.5 TPU/GPU 运行指南

在 TPU v6e-8 或 GPU H100 上运行 HunyuanVideo-1.5 文本到视频生成。

---

## 🛠 环境配置（必读）

### 前置条件

1. **Hugging Face Token**：从 [Hugging Face Settings](https://huggingface.co/settings/tokens) 获取 Access Token

2. **设置环境变量**：
   ```bash
   # 设置 Hugging Face Token（必需）
   export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
   
   # 设置 Hugging Face 缓存目录（推荐使用 /dev/shm 加速）
   export HF_HOME=/dev/shm
   ```

3. **将环境变量添加到 ~/.bashrc（持久化）**：
   ```bash
   echo 'export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"' >> ~/.bashrc
   echo 'export HF_HOME=/dev/shm' >> ~/.bashrc
   source ~/.bashrc
   ```

---

### TPU 环境安装

在 Google Cloud TPU v6e-8 上运行：

```bash
# 1. 安装基础依赖
pip install --upgrade pip
pip install numpy scipy pillow imageio loguru einops safetensors

# 2. 安装 JAX（TPU 版本）
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# 3. 安装 PyTorch + torchax
pip install torch torchvision
pip install torchax

# 4. 安装 transformers 和 diffusers
pip install transformers accelerate
pip install diffusers

# 5. 安装 ffmpeg（视频编码）
sudo apt update && sudo apt install -y ffmpeg

# 6. 克隆 HunyuanVideo-1.5-TPU 代码库（包含模型定义）
git clone https://github.com/yangwhale/HunyuanVideo-1.5-TPU.git ~/HunyuanVideo-1.5-TPU

# 7. 克隆本项目
git clone https://github.com/yangwhale/gpu-tpu-pedia.git ~/gpu-tpu-pedia
```

---

### 下载模型权重

> 📖 **官方文档**：[checkpoints-download.md](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/checkpoints-download.md)

推荐下载到 `/dev/shm`（内存文件系统，读取更快）。

#### 1. 安装下载工具

```bash
pip install -U "huggingface_hub[cli]"
pip install modelscope
```

#### 2. 下载 DiT + VAE 权重（约 25GB）

```bash
# 创建目标目录
mkdir -p /dev/shm/ckpts

# 下载 DiT 和 VAE
hf download tencent/HunyuanVideo-1.5 --local-dir /dev/shm/ckpts

# 中国用户使用镜像加速
HF_ENDPOINT=https://hf-mirror.com hf download tencent/HunyuanVideo-1.5 --local-dir /dev/shm/ckpts
```

#### 3. 下载 Text Encoder

**MLLM（推荐使用 Qwen2.5-VL-7B）**：
```bash
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir /dev/shm/ckpts/text_encoder/llm
```

**ByT5 Encoder**：
```bash
# 下载 byt5-small
hf download google/byt5-small --local-dir /dev/shm/ckpts/text_encoder/byt5-small

# 下载 Glyph-SDXL-v2（从 ModelScope）
modelscope download --model AI-ModelScope/Glyph-SDXL-v2 --local_dir /dev/shm/ckpts/text_encoder/Glyph-SDXL-v2
```

#### 4. 下载 Vision Encoder（可选，用于 I2V）

需要先在 [Hugging Face FLUX.1-Redux-dev](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev) 申请访问权限，获批后：

```bash
hf download black-forest-labs/FLUX.1-Redux-dev \
    --local-dir /dev/shm/ckpts/vision_encoder/siglip \
    --token $HF_TOKEN
```

#### 5. 最终目录结构

```
/dev/shm/ckpts/
├── hunyuan-video-t2v-720p/            # Transformer 权重
│   └── transformers/
│       ├── mp_rank_00_model_states.pt
│       └── ...
├── vae/                               # VAE 权重
├── text_encoder/                      # Text Encoder
│   ├── llm/                           # Qwen2.5-VL-7B
│   ├── byt5-small/                    # ByT5
│   └── Glyph-SDXL-v2/                 # Glyph 模型
│       ├── assets/
│       │   ├── color_idx.json
│       │   └── ...
│       └── checkpoints/
│           └── byt5_model.pt
└── vision_encoder/                    # Vision Encoder（可选）
    └── siglip/
```

#### 6. 验证下载

```bash
# 检查 Transformer 权重
ls -la /dev/shm/ckpts/hunyuan-video-t2v-720p/transformers/
# 应看到 mp_rank_00_model_states.pt

# 检查 Text Encoder
ls -la /dev/shm/ckpts/text_encoder/llm/
ls -la /dev/shm/ckpts/text_encoder/Glyph-SDXL-v2/checkpoints/
```

> 💡 **断点续传**：如果下载中断，直接重新运行下载命令即可自动续传。

---

### GPU 环境安装

在 NVIDIA H100 8卡上运行：

```bash
# 1. 安装基础依赖
pip install --upgrade pip
pip install numpy scipy pillow imageio loguru einops safetensors

# 2. 安装 PyTorch + CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. 安装 Flash Attention 2（H100 推荐）
pip install flash-attn --no-build-isolation

# 4. 安装 transformers 和 diffusers
pip install transformers accelerate
pip install diffusers

# 5. 安装 ffmpeg
sudo apt update && sudo apt install -y ffmpeg

# 6. 克隆 HunyuanVideo-1.5-TPU 代码库
git clone https://github.com/yangwhale/HunyuanVideo-1.5-TPU.git ~/HunyuanVideo-1.5-TPU

# 7. 克隆本项目
git clone https://github.com/yangwhale/gpu-tpu-pedia.git ~/gpu-tpu-pedia

# 8. 下载模型权重（同 TPU 环境）
mkdir -p /dev/shm/HunyuanVideo1.5
huggingface-cli download tencent/HunyuanVideo-1.5 \
    --local-dir /dev/shm/HunyuanVideo1.5 \
    --local-dir-use-symlinks False
```

---

## 🚀 快速开始

### 方案 A：TPU 运行（推荐）

```bash
cd ~/gpu-tpu-pedia/tpu/HunyuanVideo-1.5/generate_hunyuan_flax_staged

# 运行 121帧 720p 视频生成（约 6 分钟）
python stage2_transformer.py \
    --video_length 121 \
    --num_inference_steps 50 \
    --warmup_steps 2
```

### 方案 B：GPU 运行

```bash
cd ~/gpu-tpu-pedia/tpu/HunyuanVideo-1.5/generate_hunyuan_gpu_staged

# 三阶段运行
bash run_stage1.sh  # Text Encoder（单卡）
bash run_stage2.sh  # Transformer（8卡）
bash run_stage3.sh  # VAE Decoder（8卡）
```

---

## 📦 目录结构

```
HunyuanVideo-1.5/
├── 📁 generate_hunyuan_flax_staged/   # ⭐ TPU 推荐版本
├── 📁 generate_hunyuan_gpu_staged/    # GPU H100 版本
├── 📁 generate_diffusers_flax_staged/ # TPU + Diffusers 版本
├── 📁 docs/                           # 技术文档
├── generate_diffusers_flax.py         # TPU 单文件版本
├── generate_diffusers_gpu.py          # GPU 单文件版本
└── run_diffusers_gpu.sh               # GPU 运行脚本
```

### 各目录说明

| 目录 | 平台 | 说明 | 推荐度 |
|------|------|------|--------|
| `generate_hunyuan_flax_staged/` | TPU | 使用原生 HunyuanVideo-1.5-TPU，Splash Attention | ⭐⭐⭐ |
| `generate_hunyuan_gpu_staged/` | GPU | 使用原生 HunyuanVideo-1.5-TPU，Flash Attention | ⭐⭐⭐ |
| `generate_diffusers_flax_staged/` | TPU | 使用 diffusers-tpu 库 | ⭐⭐ |
| `docs/` | - | 技术分析文档 | - |

---

## 📂 目录详解

### 1. `generate_hunyuan_flax_staged/` — TPU 推荐版本

**使用场景**：在 TPU v6e-8 上运行 HunyuanVideo-1.5

**技术特点**：
- 使用原生 HunyuanVideo-1.5-TPU 代码库
- Splash Attention（TPU 优化）
- TP + fc2/proj Replicated 分片策略（+10.2% 性能）
- 支持 DeepCache 加速

**文件说明**：
```
generate_hunyuan_flax_staged/
├── stage2_transformer.py              # 主推理脚本
├── utils.py                           # 工具函数
├── TORCHAX_MIGRATION_GUIDE.md         # ⭐ GPU→TPU 迁移完整指南
├── GPU_TPU_COMPARISON.md              # GPU/TPU 代码对比
└── DEEPCACHE_EXPLAINED.md             # DeepCache 原理说明
```

**使用方法**：
```bash
# 前提：需要先在其他地方运行 Stage 1 生成 embeddings
# 或者使用 generate_diffusers_flax_staged/ 的 stage1

# 运行 Transformer 推理
python stage2_transformer.py \
    --input_dir ./stage_outputs \
    --video_length 121 \
    --num_inference_steps 50 \
    --seed 42

# 启用 DeepCache 加速（+70% 速度，质量稍降）
python stage2_transformer.py \
    --enable_cache \
    --cache_start_step 11 \
    --cache_end_step 45 \
    --cache_step_interval 4
```

**性能数据（TPU v6e-8）**：
| 配置 | 每步时间 | 50步总时间 |
|------|----------|------------|
| 121帧 720p | 7.29s | 6.1 分钟 |
| 121帧 720p + DeepCache | ~4s | 3.5 分钟 |

---

### 2. `generate_hunyuan_gpu_staged/` — GPU H100 版本

**使用场景**：在 NVIDIA H100 8卡上运行

**技术特点**：
- 使用原生 HunyuanVideo-1.5-TPU 代码库
- 支持 Flash Attention 2/3、SageAttention
- Sequence Parallelism 多卡并行
- 支持 DeepCache 加速

**文件说明**：
```
generate_hunyuan_gpu_staged/
├── README.md                          # ⭐ 完整使用指南
├── stage1_text_encoder.py             # Stage 1: 文本编码
├── stage2_transformer.py              # Stage 2: Transformer
├── stage2_transformer_explained.py    # 带详细注释的版本
├── stage3_vae_decoder.py              # Stage 3: VAE 解码
├── run_stage1.sh                      # 运行脚本
├── run_stage2.sh
├── run_stage3.sh
└── utils.py
```

**使用方法**：
```bash
# 完整三阶段 Pipeline
bash run_stage1.sh  # 单卡运行 Text Encoder
bash run_stage2.sh  # 8卡运行 Transformer
bash run_stage3.sh  # 8卡运行 VAE Decoder

# 或直接运行
python stage1_text_encoder.py --model_path /dev/shm/HunyuanVideo1.5 --prompt "Your prompt"
torchrun --nproc_per_node=8 stage2_transformer.py --input_dir ./stage_outputs
torchrun --nproc_per_node=8 stage3_vae_decoder.py --input_dir ./stage_outputs
```

**性能数据（H100 8卡）**：
| 方案 | 加速比 | 每步时间 |
|------|--------|----------|
| Flash Attention 2 | 1.0x | 5.2s |
| DeepCache | 1.83x | 2.84s |
| SageAttention | 1.6x | 3.25s |

详见 [`generate_hunyuan_gpu_staged/README.md`](generate_hunyuan_gpu_staged/README.md)

---

### 3. `generate_diffusers_flax_staged/` — TPU + Diffusers 版本

**使用场景**：使用 diffusers-tpu 库在 TPU 上运行

**技术特点**：
- 基于 Hugging Face diffusers 库
- 需要修改版 diffusers-tpu 库
- Splash Attention + SDPA

**文件说明**：
```
generate_diffusers_flax_staged/
├── README.md                          # 使用指南
├── stage1_text_encoder.py             # Stage 1: Text Encoder
├── stage2_transformer.py              # Stage 2: Transformer
├── stage3_vae_decoder.py              # Stage 3: VAE Decoder
└── utils.py
```

**使用方法**：
```bash
cd generate_diffusers_flax_staged

python stage1_text_encoder.py  # CPU 运行
python stage2_transformer.py   # TPU 运行
python stage3_vae_decoder.py   # TPU 运行
```

详见 [`generate_diffusers_flax_staged/README.md`](generate_diffusers_flax_staged/README.md)

---

### 4. `docs/` — 技术文档

深入分析 HunyuanVideo-1.5 的实现细节。

| 文档 | 内容 |
|------|------|
| `deepcache_explained.md` | DeepCache 加速原理与实现 |
| `dit_implementation_analysis.md` | DiT Transformer 架构分析 |
| `scheduler_explained.md` | Flow Matching Scheduler 详解 |
| `splash_attention_kernel_analysis.md` | Splash Attention Kernel 分析 |

---

## 📊 性能对比

### TPU vs GPU

| 平台 | 配置 | 121帧 720p 50步 | 每步时间 |
|------|------|-----------------|----------|
| TPU v6e-8 | TP + fc2 Replicated | 6.1 分钟 | 7.29s |
| GPU H100 8卡 | Flash Attention 2 | 4.3 分钟 | 5.2s |
| GPU H100 8卡 | DeepCache | 2.4 分钟 | 2.84s |

### 加速方案对比

| 方案 | 加速比 | 质量 | 推荐场景 |
|------|--------|------|----------|
| 标准（无加速） | 1.0x | ✅ 最优 | 生产环境 |
| DeepCache | 1.8x | ✅ 良好 | 日常使用 |
| SageAttention | 1.6x | ⚠️ 有损 | 快速预览 |

---

## ❓ 常见问题

### 1. 权重下载失败

```bash
# 检查 HF_TOKEN 是否设置
echo $HF_TOKEN

# 手动登录
huggingface-cli login

# 使用代理（如需）
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
```

### 2. OOM（内存不足）

**TPU**：
- 使用 `--video_length 49` 减少帧数
- 启用 DeepCache 减少峰值内存

**GPU**：
- Stage 2：不要使用 `create_pipeline()`，直接加载 Transformer
- Stage 3：必须使用 `torch.no_grad()`

### 3. 首次运行很慢

这是 XLA/JAX 编译造成的，正常现象。后续运行会使用缓存。

```bash
# 清除编译缓存（如需重新编译）
rm -rf /dev/shm/jax_cache
```

### 4. 视频质量问题

- 确保使用 bf16 精度
- 检查 Attention Mask 是否正确处理（K/V 置零方案）
- 参考 `TORCHAX_MIGRATION_GUIDE.md` 的修复说明

### 5. 找不到模型文件

```bash
# 检查模型路径
ls -la /dev/shm/HunyuanVideo1.5/

# 检查 transformer 权重
ls -la /dev/shm/HunyuanVideo1.5/ckpt/hunyuan-video-t2v-720p/transformers/
```

---

## 📚 参考资料

- [HunyuanVideo-1.5-TPU](https://github.com/yangwhale/HunyuanVideo-1.5-TPU) - 原生代码库
- [torchax](https://github.com/pytorch/xla) - PyTorch → JAX 桥接
- [JAX Splash Attention](https://github.com/jax-ml/jax) - TPU 优化注意力
- [diffusers-tpu](https://github.com/yangwhale/diffusers-tpu) - 修改版 diffusers

---

## 📝 License

与 HunyuanVideo 项目保持一致。