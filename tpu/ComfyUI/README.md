# ComfyUI on TPU

**中文** | **[English](README_EN.md)**

在 Google Cloud TPU 上运行 ComfyUI，支持 Flux.2 图像生成和 CogVideoX、Wan2.1、Wan2.2 视频生成模型。

**作者**: Chris Yang

---

## 快速开始

使用一键安装脚本快速配置 TPU 环境（推荐）：

```bash
# 1. 克隆 gpu-tpu-pedia 仓库
git clone https://github.com/yangwhale/gpu-tpu-pedia.git
cd gpu-tpu-pedia/tpu/ComfyUI

# 2. 运行安装脚本（需要 sudo 权限）
python3 setup.py

# 3. 安装完成后，重新加载环境变量
source ~/.bashrc

# 4. 设置 HuggingFace Token（访问 gated 模型需要）
export HF_TOKEN=<your_huggingface_token>

# 5. 启动 ComfyUI
cd ~/ComfyUI && python main.py --cpu --listen 0.0.0.0
```

访问 ComfyUI: `http://<TPU_VM_IP>:8188`

---

## 目录

- [快速开始](#快速开始)
- [环境要求](#环境要求)
- [手动安装](#手动安装)
  - [安装 Python 3.12](#安装-python-312ubuntu-2204)
  - [安装 ComfyUI](#安装-comfyui)
  - [安装 ComfyUI Manager](#安装-comfyui-manager)
  - [安装 TPU Custom Nodes](#安装-tpu-custom-nodes)
  - [安装 TPU 核心依赖](#安装-tpu-核心依赖)
- [启动 ComfyUI](#启动-comfyui)
- [切换模型前清理 HBM](#切换模型前清理-hbm)
- [支持的模型](#支持的模型)
  - [Flux.2-TPU（图像生成）](#comfyui-flux2-tpu)
  - [CogVideoX-TPU（文本到视频）](#comfyui-cogvideox-tpu)
  - [Wan2.1-TPU（文本到视频）](#comfyui-wan21-tpu)
  - [Wan2.2-I2V-TPU（图像到视频）](#comfyui-wan22-i2v-tpu)
  - [Crystools（硬件监控）](#comfyui-crystools)
- [性能数据](#性能数据)
- [故障排除](#故障排除)
- [相关链接](#相关链接)

---

## 环境要求

| 项目 | 要求 |
|------|------|
| **硬件** | Google Cloud TPU v4, v5, v6e（推荐 v6e-8） |
| **操作系统** | Ubuntu 22.04 |
| **Python** | 3.10+（推荐 3.12） |
| **JAX** | 0.8.1 + libtpu 0.0.30 |
| **存储** | 100GB+（模型缓存建议使用 /dev/shm） |

### 为什么使用 JAX 0.8.1？

JAX 0.8.2 的 CPU AOT 编译器在某些 CPU 架构（如 AMD EPYC）上存在兼容性问题，会出现 `prefer-no-scatter` 特性不匹配的警告。使用 JAX 0.8.1 + libtpu 0.0.30 可以避免这些问题。

---

## 手动安装

如果不想使用一键安装脚本，可以按以下步骤手动安装。

### 安装 Python 3.12（Ubuntu 22.04）

TPU VM 默认使用 Python 3.10，建议升级到 Python 3.12：

```bash
# 1. 停止 unattended-upgrades（避免 apt lock 冲突）
sudo systemctl stop unattended-upgrades

# 2. 添加 deadsnakes PPA 并安装 Python 3.12
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev

# 3. 初始化 pip（Python 3.12 移除了 distutils）
python3.12 -m ensurepip --upgrade

# 4. 设置为默认 python
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# 5. 验证
python --version  # 应显示 Python 3.12.x
```

### 配置 pip

Python 3.12 默认禁止系统级安装（PEP 668），需要配置：

```bash
mkdir -p ~/.config/pip
cat > ~/.config/pip/pip.conf << 'EOF'
[global]
break-system-packages = true
EOF
```

### 安装 ComfyUI

```bash
cd ~
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt
```

### 安装 ComfyUI Manager

```bash
cd ~/ComfyUI/custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git

# 配置使用 pip（避免 uv 权限问题）
mkdir -p ~/ComfyUI/user/__manager
cat > ~/ComfyUI/user/__manager/config.ini << 'EOF'
[default]
use_uv = False
EOF
```

### 安装 TPU Custom Nodes

```bash
# 从 gpu-tpu-pedia 复制 TPU 优化的节点
cd ~/gpu-tpu-pedia/tpu/ComfyUI/custom_nodes
cp -r ComfyUI-Flux.2-TPU ~/ComfyUI/custom_nodes/
cp -r ComfyUI-CogVideoX-TPU ~/ComfyUI/custom_nodes/
cp -r ComfyUI-Wan2.1-TPU ~/ComfyUI/custom_nodes/
cp -r ComfyUI-Wan2.2-I2V-TPU ~/ComfyUI/custom_nodes/
cp -r ComfyUI-Crystools ~/ComfyUI/custom_nodes/

# 安装 Crystools 依赖
pip install -r ~/ComfyUI/custom_nodes/ComfyUI-Crystools/requirements.txt
```

### 安装 TPU 核心依赖

```bash
# 核心 ML 库
pip install huggingface-hub transformers datasets evaluate accelerate timm flax numpy

# JAX with TPU（使用 0.8.1 避免 CPU AOT 兼容性问题）
pip install 'jax[tpu]==0.8.1' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install torchax tensorflow-cpu

# 辅助工具
pip install sentencepiece imageio[ffmpeg] tpu-info matplotlib
pip install 'jinja2>=3.1.0'  # Flux.2 需要
pip install ftfy             # Wan2.1 需要

# 安装 ffmpeg
sudo apt-get install -y ffmpeg

# 安装 diffusers-tpu（TPU 优化版 Diffusers）
cd ~
git clone https://github.com/yangwhale/diffusers-tpu.git
cd diffusers-tpu && pip install -e . && cd ..
```

### 配置环境变量

```bash
cat >> ~/.bashrc << 'EOF'

# === ComfyUI TPU Environment ===
export PATH=$HOME/.local/bin:$PATH
export HF_HOME=/dev/shm
export HF_TOKEN=<your_huggingface_token>
export JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax_cache
# === End ComfyUI TPU Environment ===
EOF

source ~/.bashrc
```

> **注意**: 请将 `<your_huggingface_token>` 替换为你的 [HuggingFace Token](https://huggingface.co/settings/tokens)。访问 gated 模型（如 Flux.2）需要此 token。

---

## 启动 ComfyUI

### 基本启动

```bash
cd ~/ComfyUI
python main.py --cpu --listen 0.0.0.0
```

**参数说明**:
- `--cpu`: 禁用 CUDA，使用 CPU 作为默认设备（TPU 节点会自动使用 JAX/TPU）
- `--listen 0.0.0.0`: 允许外部访问
- `--port 8188`: 指定端口（默认 8188）

### 后台运行

```bash
# 使用 nohup
nohup python main.py --cpu --listen 0.0.0.0 > comfyui.log 2>&1 &

# 查看日志
tail -f comfyui.log

# 使用 screen
screen -S comfyui
python main.py --cpu --listen 0.0.0.0
# Ctrl+A, D 分离; screen -r comfyui 恢复
```

---

## 切换模型前清理 HBM

⚠️ **重要**: TPU 的 HBM（高带宽内存）有限。在切换到不同模型之前，**必须先清理 HBM**，否则会 OOM。

### 清理方法

1. **使用 ComfyUI Manager**: 点击界面右上角 **Manager** → **🧹 Unload Models**
2. **重启 ComfyUI**: `pkill -f "python main.py" && cd ~/ComfyUI && python main.py --cpu --listen 0.0.0.0`

### 何时需要清理

| 场景 | 需要清理 |
|------|---------|
| Flux.2 → Wan2.1 | ✅ 是 |
| Wan2.1 → CogVideoX | ✅ 是 |
| CogVideoX → Wan2.2-I2V | ✅ 是 |
| 同一模型多次生成 | ❌ 否 |
| 修改 seed/prompt | ❌ 否 |

---

## 支持的模型

### ComfyUI-Flux.2-TPU

**用途**: 在 TPU 上运行 Flux.2 图像生成模型（黑森林实验室）。

**功能特性**:
- 🚀 **TPU 加速**: 使用 torchax 在 TPU 上运行 Flux.2 Transformer 和 VAE
- 🔧 **模块化设计**: 分离的 Text Encoder、Sampler 和 VAE Decoder 节点
- ⚡ **Splash Attention**: 针对长序列的 TPU 优化 attention 实现（使用 exp2 优化）
- 🔄 **自动分片**: 自动将模型权重分布到 8 个 TPU 核心

**节点列表**:

| 节点名称 | 运行位置 | 功能 |
|---------|----------|------|
| **Flux.2 Text Encoder (CPU)** | CPU | 使用 Mistral3 编码文本 prompt |
| **Flux.2 TPU Sampler** | TPU | 运行 Transformer 去噪，生成 latents |
| **Flux.2 TPU VAE Decoder** | TPU | 解码 latents 为最终图像 |
| **Flux.2 TPU Full Pipeline** | TPU | 端到端图像生成（组合以上三个） |

**工作流程**:

```
┌─────────────────────┐     ┌──────────────────┐     ┌─────────────────────┐     ┌───────────────┐
│ Flux.2 Text Encoder │────▶│ Flux.2 TPU      │────▶│ Flux.2 TPU VAE      │────▶│ Preview Image │
│ (CPU)               │     │ Sampler         │     │ Decoder             │     │               │
│                     │     │                 │     │                     │     │               │
│ prompt: "..."       │     │ height: 1024    │     │ height: 1024        │     │               │
│ model_id: ...       │     │ width: 1024     │     │ width: 1024         │     │               │
│                     │     │ steps: 50       │     │ model_id: ...       │     │               │
│                     │     │ guidance: 4.0   │     │                     │     │               │
│                     │     │ seed: ...       │     │                     │     │               │
└─────────────────────┘     └──────────────────┘     └─────────────────────┘     └───────────────┘
       prompt_embeds ────────────▶ LATENT ─────────────────▶ IMAGE ─────────────────▶
```

**参数说明**:

#### Text Encoder
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `prompt` | - | 图像描述文本 |
| `model_id` | `black-forest-labs/FLUX.2-dev` | HuggingFace 模型 ID |

#### TPU Sampler
| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `height` | 1024 | 256-2048 | 输出图像高度 |
| `width` | 1024 | 256-2048 | 输出图像宽度 |
| `num_inference_steps` | 50 | 1-100 | 去噪步数 |
| `guidance_scale` | 4.0 | 0-20 | Embedded CFG 引导强度 |
| `seed` | 42 | - | 随机种子 |

#### VAE Decoder
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `height` | 1024 | 必须与 Sampler 一致 |
| `width` | 1024 | 必须与 Sampler 一致 |
| `model_id` | `black-forest-labs/FLUX.2-dev` | HuggingFace 模型 ID |

**架构说明**:

- **Text Encoder (Mistral3)**: Flux.2 使用 Pixtral 7B 变体 (Mistral3) 作为文本编码器。由于包含动态控制流，目前在 CPU 上运行。
- **Transformer (TPU)**: 24 层 MMDiT 架构，Attention heads: 24，Hidden size: 3072，使用 Splash Attention 优化（exp2 替代 exp），权重自动分片到 8 个 TPU 核心。
- **VAE Decoder (TPU)**: Flux.2 专用 VAE，与 SDXL VAE 不兼容，在 TPU 上运行以加速解码。

**示例 Workflow**: [`examples/flux2_tpu_basic.json`](custom_nodes/ComfyUI-Flux.2-TPU/examples/flux2_tpu_basic.json)

---

### ComfyUI-CogVideoX-TPU

**用途**: 在 TPU 上运行 CogVideoX 1.5-5B 文本到视频模型（智源研究院）。

![CogVideoX T2V ComfyUI 工作流](custom_nodes/ComfyUI-CogVideoX-TPU/examples/cogvideox_t2v_720p_demo.png)

**功能特性**:
- **TPU 原生加速**: 使用 JAX/torchax 在 TPU 上运行 CogVideoX 推理
- **Splash Attention**: 使用 exp2 优化的自定义 Pallas kernel，充分利用 TPU VPU 硬件
- **三阶段 Pipeline**: 文本编码、Transformer 去噪、VAE 解码分离，内存效率高
- **K-Smooth 优化**: 可选的 Key 平滑处理，提升注意力稳定性
- **CFG 并行**: DP=2 支持 CFG 正负 prompt 并行处理

**节点列表**:

| 节点名称 | 功能 |
|---------|------|
| `CogVideoXTextEncoder` | 使用 T5 编码文本 prompt |
| `CogVideoXTPUSampler` | 在 TPU 上运行 Transformer 去噪 |
| `CogVideoXTPUVAEDecoder` | 解码 latents 为视频帧 |

**工作流程**: `TextEncoder → TPUSampler → TPUVAEDecoder → CreateVideo → SaveVideo`

**参数说明**:

#### CogVideoXTextEncoder
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt` | STRING | - | 正面提示词 |
| `negative_prompt` | STRING | "" | 负面提示词 |
| `model_id` | STRING | `zai-org/CogVideoX1.5-5B` | HuggingFace 模型 ID |

#### CogVideoXTPUSampler
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

> **注意**: `num_frames` 应满足 `(num_frames-1)/4+1` 为奇数，否则 VAE 解码会多出帧。有效帧数: 41, 49, 57, 65, 73, 81, 89, 97...

#### CogVideoXTPUVAEDecoder
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `latents` | COGVIDEOX_LATENTS | - | Sampler 输出 |
| `fps` | INT | 16 | 视频帧率 |

**技术实现**:

- **Splash Attention 优化**: Query 预乘 `LOG2_E = 1.44269504`，使 `exp(x)` 变为 `exp2(x * LOG2_E)`，更好利用 TPU VPU 硬件；使用 Pallas kernel 进行高效的分块注意力计算；K-Smooth 技术减少数值溢出。
- **权重分片策略 (Tensor Parallel)**:
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

**性能数据（8x TPU v6e）**:

| 指标 | 首次运行 | 缓存后 |
|------|---------|--------|
| Transformer (50步) | 126s | 104s |
| 每步推理时间 | 2.28s | 2.08s |
| VAE 解码 | 6.24s | 1.78s |
| 总时间 | 152s | 108s |

**示例 Workflow**: [`examples/cogvideox_t2v_720p.json`](custom_nodes/ComfyUI-CogVideoX-TPU/examples/cogvideox_t2v_720p.json)

---

### ComfyUI-Wan2.1-TPU

**用途**: 在 TPU 上运行 Wan2.1-T2V-14B 文本到视频模型（阿里巴巴）。

![Wan2.1 T2V ComfyUI 工作流](custom_nodes/ComfyUI-Wan2.1-TPU/examples/wan21_t2v_720p_demo.png)

**功能特性**:
- 🚀 **TPU 加速**: 使用 torchax 在 TPU 上运行 Wan 2.1 全部组件
- 🎬 **视频生成**: 支持 720P (1280x720) 和 480P (848x480) 分辨率
- 🔧 **模块化设计**: 分离的 Text Encoder、Sampler 和 VAE Decoder 节点
- ⚡ **Splash Attention**: 针对长序列的 TPU 优化 attention 实现（exp2 优化）
- 🔄 **2D Mesh 分片**: 自动将模型权重分布到 8 个 TPU 核心 (dp=2, tp=4)

**节点列表**:

| 节点名称 | 运行位置 | 功能 |
|---------|----------|------|
| **Wan 2.1 Text Encoder (TPU)** | TPU | 使用 T5-XXL 编码 prompt |
| **Wan 2.1 TPU Sampler** | TPU | 运行 Transformer 去噪，生成 latents |
| **Wan 2.1 TPU VAE Decoder** | TPU | 解码 latents 为视频帧 |
| **Wan 2.1 TPU Full Pipeline** | TPU | 端到端视频生成（组合以上三个） |

**工作流程**:

```
┌───────────────────────┐     ┌──────────────────┐     ┌───────────────────────┐
│ Wan 2.1 Text Encoder  │────▶│ Wan 2.1 TPU     │────▶│ Wan 2.1 TPU VAE       │
│ (TPU)                 │     │ Sampler         │     │ Decoder               │
│                       │     │                 │     │                       │
│ prompt: "..."         │     │ height: 720     │     │                       │
│ negative_prompt: "..."│     │ width: 1280     │     │                       │
│                       │     │ num_frames: 81  │     │                       │
│                       │     │ steps: 50       │     │                       │
│                       │     │ guidance: 5.0   │     │                       │
└───────────────────────┘     └──────────────────┘     └───────────────────────┘
  prompt_embeds ──────────────────▶ latents ─────────────────▶ frames
  negative_prompt_embeds ─────────┘
```

**参数说明**:

#### Text Encoder
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `prompt` | - | 视频描述文本 |
| `negative_prompt` | - | 负面提示词 |
| `model_id` | `Wan-AI/Wan2.1-T2V-14B-Diffusers` | HuggingFace 模型 ID |

#### TPU Sampler
| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `height` | 720 | 256-1280 | 视频高度 |
| `width` | 1280 | 256-1280 | 视频宽度 |
| `num_frames` | 81 | 17-121 | 视频帧数（需为 4n+1） |
| `num_inference_steps` | 50 | 1-100 | 去噪步数 |
| `guidance_scale` | 5.0 | 0-20 | CFG 引导强度 |
| `seed` | 2025 | - | 随机种子 |
| `flow_shift` | 5.0 | 1-10 | Flow Matching 位移（720P=5.0，480P=3.0） |

#### VAE Decoder
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `fps` | 16 | 输出视频帧率 |
| `model_id` | `Wan-AI/Wan2.1-T2V-14B-Diffusers` | HuggingFace 模型 ID |

**分辨率推荐**:

| 分辨率 | height | width | flow_shift | 说明 |
|--------|--------|-------|------------|------|
| 720P | 720 | 1280 | 5.0 | 高质量，推荐 |
| 480P | 480 | 848 | 3.0 | 快速测试 |

**技术细节**:

- **2D Mesh 配置**: 使用 `(dp=2, tp=4)` 配置 8 个 TPU chips，dp: Data Parallel (batch sharding)，tp: Tensor Parallel (weight sharding)
- **Splash Attention**: 使用 exp2 代替 exp，利用 TPU VPU 硬件指令；K-Smooth 技术减少数值溢出；长序列 (>20000) 使用 Splash Attention，短序列使用标准实现

**示例 Workflow**: [`examples/wan21_tpu_basic.json`](custom_nodes/ComfyUI-Wan2.1-TPU/examples/wan21_tpu_basic.json)

---

### ComfyUI-Wan2.2-I2V-TPU

**用途**: 在 TPU 上运行 Wan2.2 图像到视频模型，使用双 Transformer A14B 架构。

![Wan 2.2 I2V ComfyUI 工作流](custom_nodes/ComfyUI-Wan2.2-I2V-TPU/examples/wan22_i2v_full_view.png)

**节点列表**:

| 节点名称 | 功能 | 输入 | 输出 |
|---------|------|------|------|
| **Wan22I2VImageEncoder** | 图像条件编码 | IMAGE | CONDITION, LATENT_INFO |
| **Wan22I2VTextEncoder** | 文本编码 (UMT5-XXL) | prompt, negative_prompt | prompt_embeds, negative_prompt_embeds |
| **Wan22I2VTPUSampler** | 双 Transformer 去噪 | embeds, condition, latent_info | LATENT |
| **Wan22I2VTPUVAEDecoder** | VAE 解码 | LATENT | IMAGE |

**工作流程**:

```
┌─────────────┐     ┌─────────────────────┐
│ Load Image  │────▶│ Wan22I2VImageEncoder │──▶ condition
└─────────────┘     └─────────────────────┘     │
                                                 │
┌─────────────┐     ┌────────────────────┐      │
│ Prompt Text │────▶│ Wan22I2VTextEncoder │──┬──│
└─────────────┘     └────────────────────┘  │  │
                                             │  │
                    ┌────────────────────┐◀─┴──┘
                    │ Wan22I2VTPUSampler │
                    └────────────────────┘
                             │
                             ▼
                    ┌──────────────────────┐
                    │ Wan22I2VTPUVAEDecoder │──▶ VIDEO FRAMES
                    └──────────────────────┘
```

**参数说明**:

#### Wan22I2VImageEncoder
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image` | IMAGE | - | ComfyUI IMAGE 格式 |
| `height` | INT | 720 | 目标高度 |
| `width` | INT | 1280 | 目标宽度 |
| `num_frames` | INT | 81 | 视频帧数 |
| `model_id` | STRING | - | 模型路径 (可选) |

**输出**:
- `condition`: 图像条件 tensor `[B, 20, T_latent, H_latent, W_latent]`
- `latent_info`: 尺寸信息字典

#### Wan22I2VTextEncoder
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt` | STRING | - | 正面提示词 |
| `negative_prompt` | STRING | - | 负面提示词 |
| `model_id` | STRING | - | 模型路径 (可选) |

#### Wan22I2VTPUSampler
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt_embeds` | EMBEDS | - | 文本 embeddings |
| `negative_prompt_embeds` | EMBEDS | - | 负面 embeddings |
| `condition` | CONDITION | - | 图像条件 |
| `latent_info` | DICT | - | 尺寸信息 |
| `num_inference_steps` | INT | 40 | 推理步数 |
| `guidance_scale` | FLOAT | 3.5 | CFG 引导尺度 |
| `shift` | FLOAT | 5.0 | 时间步长分布偏移 |
| `seed` | INT | - | 随机种子 |

> **注意**: `shift` 参数较高值将更多步数分配给低噪声阶段

#### Wan22I2VTPUVAEDecoder
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `latents` | LATENT | - | LATENT dict |
| `model_id` | STRING | - | 模型路径 (可选) |
| `fps` | INT | 16 | 帧率 |

**技术细节**:

- **双 Transformer 架构**: Wan 2.2 I2V 使用双 Transformer 架构
  - **Transformer 1**: 处理高噪声阶段 (t >= 900)
  - **Transformer 2**: 处理低噪声阶段 (t < 900)
  - 切换阈值由 `BOUNDARY_RATIO = 0.9` 控制

- **A14B 模式**: 图像条件编码采用 A14B 模式
  1. 输入图像 resize 到目标分辨率
  2. 构建 video_condition: `[image, zeros, zeros, ...]`
  3. VAE 编码得到 latent_condition
  4. 归一化: `(x - mean) / std`
  5. 构建 mask (第一帧=1, 其他帧=0)
  6. 拼接 condition = `concat(mask, latent_condition)`

- **分片策略**: 使用 2D Mesh (dp=2, tp=N/2) 进行模型并行
  - Text Encoder: 词嵌入和 FFN 分片
  - Transformer: Attention 和 FFN 分片
  - VAE: 复制 (不分片)

**示例 Workflow**: 参见 [`examples/`](custom_nodes/ComfyUI-Wan2.2-I2V-TPU/examples/) 目录

---

### ComfyUI-Crystools

**用途**: 实时监控 TPU/GPU 硬件状态，提供资源监控、进度条、元数据查看等功能。

![Crystools TPU 监控器](custom_nodes/ComfyUI-Crystools/ComfyUI_Crystools_demo.png)

**功能特性**:
- 🎉 **资源监控**: 实时显示 CPU、GPU、RAM、VRAM、GPU 温度和存储空间
- 📊 **进度条**: 在菜单栏显示工作流执行进度和耗时
- 📝 **元数据**: 提取、比较和显示图像/工作流元数据
- 🔧 **调试工具**: 显示任意值到控制台/显示
- 🔗 **管道工具**: 更好地组织工作流连接

**监控指标**:

| 指标 | 说明 |
|------|------|
| **CPU** | CPU 使用率百分比 |
| **RAM** | 内存使用量和百分比 |
| **GPU/TPU** | VRAM/HBM 使用量 |
| **GPU Temp** | GPU 温度（仅 NVIDIA） |
| **HDD** | 磁盘空间使用情况 |

**主要节点**:

| 节点名称 | 功能 |
|---------|------|
| **Load image with metadata** | 加载图像并提取元数据 |
| **Save image with extra metadata** | 保存图像并附加自定义元数据 |
| **Preview from image** | 预览图像并显示当前 prompt |
| **Metadata extractor** | 提取图像的完整元数据 |
| **Metadata comparator** | 比较两个图像的元数据差异 |
| **Show any** | 在控制台/显示中查看任意值 |
| **JSON comparator** | 比较两个 JSON 的差异 |

**配置方法**: 监控器显示在 ComfyUI 界面顶部菜单栏，可在 **Settings → Crystools** 中配置刷新率和显示项目。

> **注意**: 将刷新率设置为 `0` 可禁用监控以降低系统开销。

---

## 性能数据

测试环境: **TPU v6e-8**（8 芯片，每芯片 32 GiB HBM）

### Flux.2（图像生成，1024x1024）

| 阶段 | 首次运行 | 缓存后 |
|------|---------|--------|
| Transformer（50步） | 190s | ~120s |
| 每步推理 | 3.81s | ~2.4s |
| VAE 解码 | 19s | ~5s |
| **总计** | **292s** | **~150s** |

### CogVideoX（视频生成，720p, 81帧）

| 阶段 | 首次运行 | 缓存后 |
|------|---------|--------|
| Transformer（50步） | 231s | ~105s |
| 每步推理 | 3.35s | 2.08s |
| VAE 解码 | 79s | ~6s |
| **总计** | **355s** | **~130s** |

### Wan2.1（视频生成，720p, 81帧）

| 阶段 | 数值 |
|------|------|
| Transformer（50步） | ~227s |
| 每步推理 | ~4.54s |
| VAE 解码 | ~1.16s |
| **总计** | **~230s** |

> **注意**: 首次运行包含 JAX JIT 编译时间。后续运行会使用缓存，速度显著提升。

---

## 故障排除

### 1. "No module named 'tpu_info'"

```bash
pip install tpu-info
```

### 2. "Could not find TPU devices"

确保在 TPU VM 上运行：

```bash
python -c "import jax; print(jax.devices())"
# 应显示 [TpuDevice(...), ...]
```

### 3. "JAX TPU init failed" / libtpu 版本不匹配

使用推荐版本：

```bash
pip install 'jax[tpu]==0.8.1' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### 4. "prefer-no-scatter" CPU AOT 兼容性警告

这是 JAX 0.8.2 的已知问题，降级到 0.8.1 可解决：

```bash
pip install 'jax[tpu]==0.8.1' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### 5. "401 Client Error: Unauthorized"（访问 HuggingFace 模型）

设置 HuggingFace Token：

```bash
export HF_TOKEN=<your_token>
# 或添加到 ~/.bashrc
```

### 6. "name 'ftfy' is not defined"（Wan2.1）

```bash
pip install ftfy
```

### 7. "jinja2.exceptions.TemplateNotFound"（Flux.2）

```bash
pip install 'jinja2>=3.1.0'
```

### 8. 内存不足 (OOM)

- 切换模型前先 Unload Models
- 减少 `num_frames`、`height`/`width`
- 减少 batch size

### 9. 视频保存失败

```bash
sudo apt-get install -y ffmpeg
pip install imageio[ffmpeg]
```

---

## 相关链接

- [ComfyUI 官方仓库](https://github.com/comfyanonymous/ComfyUI)
- [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager)
- [diffusers-tpu](https://github.com/yangwhale/diffusers-tpu)
- [JAX 官方文档](https://jax.readthedocs.io/)
- [tpu-info](https://github.com/google/tpu_info)
- [HuggingFace Hub](https://huggingface.co/)

---

## 许可证

MIT License
