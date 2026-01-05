# ComfyUI on TPU

**中文** | **[English](README_EN.md)**

本指南介绍如何在 Google Cloud TPU 上运行 ComfyUI，包括 TPU 专用 Custom Nodes 的安装和使用。

## 目录

- [环境要求](#环境要求)
- [安装 ComfyUI](#安装-comfyui)
- [安装 Custom Nodes](#安装-custom-nodes)
- [启动 ComfyUI](#启动-comfyui)
- [Custom Nodes 介绍](#custom-nodes-介绍)
  - [ComfyUI-CogVideoX-TPU](#comfyui-cogvideox-tpu)
  - [ComfyUI-Wan2.1-TPU](#comfyui-wan21-tpu)
  - [ComfyUI-Wan2.2-I2V-TPU](#comfyui-wan22-i2v-tpu)
  - [ComfyUI-Flux.2-TPU](#comfyui-flux2-tpu)
  - [ComfyUI-Crystools](#comfyui-crystools)
- [TPU 环境配置](#tpu-环境配置)
- [故障排除](#故障排除)

---

## 环境要求

- **硬件**: Google Cloud TPU v4, v5, v6e 或更高版本
- **操作系统**: Ubuntu 20.04+ / Debian 11+
- **Python**: 3.10+ (推荐 3.12)
- **依赖库**: JAX, PyTorch/XLA, tpu_info

## 安装 ComfyUI

### 1. 克隆 ComfyUI 仓库

```bash
cd ~
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
```

### 2. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 3. 安装视频处理依赖（可选）

如果需要生成视频，还需要安装 ffmpeg：

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# 或使用 conda
conda install ffmpeg
```

---

## 安装 Custom Nodes

Custom Nodes 需要放置在 `ComfyUI/custom_nodes/` 目录下。

### 方法一：使用 ComfyUI Manager（推荐）

```bash
cd ~/ComfyUI/custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
```

启动 ComfyUI 后，通过 Manager 界面搜索并安装 Custom Nodes。

### 方法二：从 gpu-tpu-pedia 安装（推荐 TPU 用户）

```bash
# 克隆 gpu-tpu-pedia 仓库
git clone https://github.com/yangwhale/gpu-tpu-pedia.git
cd gpu-tpu-pedia/tpu/ComfyUI/custom_nodes

# 复制 TPU Custom Nodes 到 ComfyUI
cp -r ComfyUI-CogVideoX-TPU ~/ComfyUI/custom_nodes/
cp -r ComfyUI-Wan2.1-TPU ~/ComfyUI/custom_nodes/
cp -r ComfyUI-Wan2.2-I2V-TPU ~/ComfyUI/custom_nodes/
cp -r ComfyUI-Flux.2-TPU ~/ComfyUI/custom_nodes/
cp -r ComfyUI-Crystools ~/ComfyUI/custom_nodes/

# 安装依赖
pip install -r ~/ComfyUI/custom_nodes/ComfyUI-Crystools/requirements.txt
```

---

## 启动 ComfyUI

### 在 TPU 机器上启动

由于 ComfyUI 默认使用 CUDA，在 TPU 机器上需要使用 `--cpu` 参数启动：

```bash
cd ~/ComfyUI
python main.py --cpu --listen 0.0.0.0
```

**参数说明：**
- `--cpu`: 禁用 CUDA，使用 CPU 作为默认设备（TPU 节点会自动使用 JAX/TPU）
- `--listen 0.0.0.0`: 允许外部访问（用于 SSH 端口转发或直接访问）
- `--port 8188`: 指定端口（默认 8188）

### 后台运行

```bash
nohup python main.py --cpu --listen 0.0.0.0 > comfyui.log 2>&1 &
```

### 使用 Screen/Tmux

```bash
screen -S comfyui
python main.py --cpu --listen 0.0.0.0
# Ctrl+A, D 分离
```

---

## Custom Nodes 介绍

### ComfyUI-CogVideoX-TPU

**用途**：在 TPU 上运行 CogVideoX 1.5-5B 文本到视频 (T2V) 模型，使用 Splash Attention 加速生成高质量视频。

![CogVideoX T2V ComfyUI 工作流](custom_nodes/ComfyUI-CogVideoX-TPU/examples/cogvideox_t2v_720p_demo.png)

**节点列表：**

| 节点名称 | 功能 |
|---------|------|
| `CogVideoXTextEncoder` | 编码文本提示词，使用 T5 生成 prompt embeddings |
| `CogVideoXTPUSampler` | 在 TPU 上运行 Transformer 扩散采样，生成 latents |
| `CogVideoXTPUVAEDecoder` | 解码 latents 为视频帧 |

**工作流程：**

```
TextEncoder → TPUSampler → TPUVAEDecoder → CreateVideo → SaveVideo
```

**使用示例工作流：**

在 ComfyUI 界面中，点击左侧的 **Templates** 标签页，找到 **CogVideoX T2V 720p** 模板，点击即可加载完整工作流。

**参数说明：**

- **CogVideoXTextEncoder**
  - `prompt`: 正面提示词
  - `negative_prompt`: 负面提示词
  - `model_id`: 模型路径 (默认 `zai-org/CogVideoX1.5-5B`)

- **CogVideoXTPUSampler**
  - `height`: 视频高度 (720)
  - `width`: 视频宽度 (1280)
  - `num_frames`: 帧数 (81 = 5秒 @ 16fps)
  - `num_inference_steps`: 采样步数 (50)
  - `guidance_scale`: CFG 强度 (6.0)
  - `seed`: 随机种子

- **CogVideoXTPUVAEDecoder**
  - `fps`: 视频帧率 (16)

**性能数据（8x TPU v6e）：**

| 指标 | 首次运行 | 缓存后 |
|------|---------|--------|
| Transformer (50步) | 126s | 104s |
| 每步推理时间 | 2.28s | 2.08s |
| VAE 解码 | 6.24s | 1.78s |
| 总时间 | 152s | 108s |

**技术特点：**

- **Splash Attention**：TPU 优化的注意力实现，使用 exp2 代替 exp 以提升 TPU 性能
- **Tensor Parallelism**：支持跨 TPU 设备的权重分片 (dp=2, tp=4)
- **SafeTensors 加载**：使用 `use_safetensors=True` 确保安全加载
- **Protobuf 冲突修复**：预加载 Tokenizer 避免与 JAX 的 protobuf 版本冲突

---

### ComfyUI-Wan2.1-TPU

**用途**：在 TPU 上运行 Wan2.1 文本到视频 (T2V) 模型，生成高质量视频。

![Wan2.1 T2V ComfyUI 工作流](custom_nodes/ComfyUI-Wan2.1-TPU/examples/wan21_t2v_720p_demo.png)

**节点列表：**

| 节点名称 | 功能 |
|---------|------|
| `Wan21TextEncoder` | 编码文本提示词，生成 prompt embeddings |
| `Wan21TPUSampler` | 在 TPU 上运行扩散采样，生成 latents |
| `Wan21TPUVAEDecoder` | 解码 latents 为视频帧 |

**工作流程：**

```
TextEncoder → TPUSampler → TPUVAEDecoder → CreateVideo → SaveVideo
```

**示例工作流：**

加载 `custom_nodes/ComfyUI-Wan2.1-TPU/examples/wan21_t2v_720p.json`

**参数说明：**

- **Wan21TextEncoder**
  - `prompt`: 正面提示词
  - `negative_prompt`: 负面提示词
  - `model_id`: 模型路径 (如 `Wan-AI/Wan2.1-T2V-14B-Diffusers`)

- **Wan21TPUSampler**
  - `height`: 视频高度 (720)
  - `width`: 视频宽度 (1280)
  - `num_frames`: 帧数 (81 = 5秒 @ 16fps)
  - `num_inference_steps`: 采样步数 (50)
  - `guidance_scale`: CFG 强度 (5.0)
  - `seed`: 随机种子
  - `num_devices`: 使用的 TPU 设备数量 (1-8)

- **Wan21TPUVAEDecoder**
  - `fps`: 视频帧率 (16)

**性能数据（8x TPU v6e）：**

| 指标 | 数值 |
|------|------|
| Transformer (50步) | 227s |
| 每步推理时间 | 4.54s |
| VAE 解码 | 1.16s |
| 总时间 | 230s |

**技术特点：**

- **14B 参数模型**：Wan2.1-T2V-14B 是大规模视频生成模型
- **Splash Attention**：TPU 优化的注意力实现
- **WanVAE**：使用视频编解码器，避免与 JAX 的 protobuf 冲突

---

### ComfyUI-Wan2.2-I2V-TPU

**用途**：在 TPU 上运行 Wan2.2 图像到视频 (I2V) 模型，使用双 Transformer A14B 架构生成高质量视频。

![Wan 2.2 I2V ComfyUI 工作流](custom_nodes/ComfyUI-Wan2.2-I2V-TPU/examples/wan22_i2v_full_view.png)

**节点列表：**

| 节点名称 | 功能 |
|---------|------|
| `Wan22I2VImageEncoder` | 编码输入图像，生成 CLIP 和 VAE 条件 |
| `Wan22I2VTextEncoder` | 编码文本提示词，生成 prompt embeddings |
| `Wan22I2VTPUSampler` | 在 TPU 上运行双 Transformer 扩散采样 |
| `Wan22I2VTPUVAEDecoder` | 解码 latents 为视频帧 |

**工作流程：**

```
Image → ImageEncoder ─┬→ TPUSampler → TPUVAEDecoder → CreateVideo → SaveVideo
                      │
TextEncoder ──────────┘
```

**示例工作流：**

加载 `custom_nodes/ComfyUI-Wan2.2-I2V-TPU/examples/wan22_i2v_720p.json`

**参数说明：**

- **Wan22I2VImageEncoder**
  - `image`: 输入图像 (首帧)
  - `model_id`: 模型路径 (如 `Wan-AI/Wan2.2-I2V-14B-720P-Diffusers`)

- **Wan22I2VTextEncoder**
  - `prompt`: 正面提示词
  - `negative_prompt`: 负面提示词
  - `model_id`: 模型路径

- **Wan22I2VTPUSampler**
  - `height`: 视频高度 (720)
  - `width`: 视频宽度 (1280)
  - `num_frames`: 帧数 (81 = 5秒 @ 16fps)
  - `num_inference_steps`: 采样步数 (50)
  - `guidance_scale`: CFG 强度 (5.0)
  - `shift`: 时间步长分布偏移 (5.0)
  - `seed`: 随机种子
  - `num_devices`: 使用的 TPU 设备数量 (1-8)
  - `boundary_ratio`: A14B 模型切换比例 (0.9)

- **Wan22I2VTPUVAEDecoder**
  - `fps`: 视频帧率 (16)

**技术特点：**

- **双 Transformer 架构 (A14B)**：使用 `boundary_ratio=0.9` 在两个模型之间切换，前 90% 步数使用主模型，后 10% 使用辅助模型
- **Splash Attention**：TPU 优化的注意力实现，大幅提升推理速度
- **图像条件**：支持输入首帧图像作为视频生成的条件

---

### ComfyUI-Flux.2-TPU

**用途**：在 TPU 上运行 Flux.2 图像生成模型。

**节点列表：**

| 节点名称 | 功能 |
|---------|------|
| `FluxTPUTextEncoder` | 编码文本提示词 |
| `FluxTPUSampler` | 在 TPU 上运行扩散采样 |
| `FluxTPUVAEDecoder` | 解码 latents 为图像 |

**示例工作流：**

加载 `custom_nodes/ComfyUI-Flux.2-TPU/examples/flux2_tpu_basic.json`

---

### ComfyUI-Crystools

**用途**：实时监控硬件状态。在 TPU 环境下，自动检测并显示 TPU 设备信息。

![Crystools TPU 监控器](custom_nodes/ComfyUI-Crystools/ComfyUI_Crystools_demo.png)

**功能特性：**

- **CPU 监控**：显示 CPU 使用率
- **RAM 监控**：显示系统内存使用量和百分比
- **TPU 监控**（每个设备）：
  - **HBM**：高带宽内存使用量和百分比
  - **Busy**：TPU 忙碌状态百分比
  - **MFU**：Model FLOPS Utilization（模型算力利用率）

**配置：**

监控器显示在 ComfyUI 界面顶部菜单栏。可在 Settings → Crystools 中配置：

- 各个监控项的显示/隐藏
- 刷新频率（默认 0.5 秒）
- 监控器尺寸（宽度/高度）

---

## TPU 环境配置

### 0. 配置模型存储路径（推荐）

TPU VM 的本地磁盘容量有限（通常 100GB），而大型模型（如 Wan2.1-14B）需要大量存储空间。建议将模型目录软链接到共享内存 `/dev/shm`：

```bash
# 创建共享内存中的模型目录
mkdir -p /dev/shm/comfyui_models

# 先将原有 models 内容复制到共享内存
cp -r ~/ComfyUI/models/* /dev/shm/comfyui_models/

# 删除原有 models 目录并创建软链接
rm -rf ~/ComfyUI/models
ln -s /dev/shm/comfyui_models ~/ComfyUI/models

# 验证软链接
ls -la ~/ComfyUI/models
```

**注意**：`/dev/shm` 使用内存作为存储，重启后数据会丢失。如果需要持久化存储，可以考虑：
- 挂载 GCS bucket
- 使用持久化磁盘

### 1. 安装核心依赖

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

# 设置 Hugging Face Token
export HF_TOKEN=<your HF_TOKEN>

# JAX 编译缓存（加速重复运行）
export JAX_COMPILATION_CACHE_DIR=/dev/shm/jax_cache
```

### 3. 安装 diffusers-tpu（TPU 优化版 Diffusers）

```bash
# 克隆 diffusers-tpu 项目（包含 TorchAx/Flax VAE 实现）
git clone https://github.com/yangwhale/diffusers-tpu.git
cd diffusers-tpu
pip install -e .
cd ..
```

### 4. 验证 TPU 可用性

```python
import jax
print(jax.devices())
# 应该显示 [TpuDevice(...), TpuDevice(...), ...]
```

### 5. 检查 TPU 状态

```bash
# 使用 tpu_info CLI
tpu-info

# 或 Python
python -c "from tpu_info import device; print(device.get_local_chips())"
```

---

## 故障排除

### 1. "No module named 'tpu_info'"

```bash
pip install tpu_info
```

### 2. "Could not find TPU devices"

确保在 TPU VM 上运行，或检查 TPU 环境变量：

```bash
# 检查 TPU 名称
echo $TPU_NAME
echo $TPU_LOAD_LIBRARY
```

### 3. "JAX TPU init failed"

可能是 libtpu 版本不匹配：

```bash
pip install --upgrade jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### 4. ComfyUI 显示 GPU 而不是 TPU

确保：
1. 使用 `--cpu` 参数启动
2. ComfyUI-Crystools-TPU 正确安装（不是原版 ComfyUI-Crystools）

### 5. 视频保存失败

安装 ffmpeg：

```bash
sudo apt-get install ffmpeg
```

### 6. 内存不足 (OOM)

- 减少 `num_frames`
- 减少 `height`/`width`
- 减少 batch size

---

## 相关链接

- [ComfyUI 官方仓库](https://github.com/comfyanonymous/ComfyUI)
- [ComfyUI-TPU (Wan + Flux)](https://github.com/yangwhale/ComfyUI-TPU)
- [ComfyUI-Crystools-TPU](https://github.com/yangwhale/ComfyUI-Crystools-TPU)
- [JAX 官方文档](https://jax.readthedocs.io/)
- [tpu_info](https://github.com/google/tpu_info)

---

## 许可证

MIT License
