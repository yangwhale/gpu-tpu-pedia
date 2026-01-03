# ComfyUI on TPU

本指南介绍如何在 Google Cloud TPU 上运行 ComfyUI，包括 TPU 专用 Custom Nodes 的安装和使用。

## 目录

- [环境要求](#环境要求)
- [安装 ComfyUI](#安装-comfyui)
- [安装 Custom Nodes](#安装-custom-nodes)
- [启动 ComfyUI](#启动-comfyui)
- [Custom Nodes 介绍](#custom-nodes-介绍)
  - [ComfyUI-Wan-TPU](#comfyui-wan-tpu)
  - [ComfyUI-Flux-TPU](#comfyui-flux-tpu)
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
cp -r ComfyUI-Wan-TPU ~/ComfyUI/custom_nodes/
cp -r ComfyUI-Flux-TPU ~/ComfyUI/custom_nodes/
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

### ComfyUI-Wan-TPU

**用途**：在 TPU 上运行 Wan2.1 文本到视频 (T2V) 模型，生成高质量视频。

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

加载 `custom_nodes/ComfyUI-Wan-TPU/examples/wan21_t2v_720p.json`

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

---

### ComfyUI-Flux-TPU

**用途**：在 TPU 上运行 Flux.1/Flux.2 图像生成模型。

**节点列表：**

| 节点名称 | 功能 |
|---------|------|
| `FluxTPUTextEncoder` | 编码文本提示词 |
| `FluxTPUSampler` | 在 TPU 上运行扩散采样 |
| `FluxTPUVAEDecoder` | 解码 latents 为图像 |

**示例工作流：**

加载 `custom_nodes/ComfyUI-Flux-TPU/examples/flux2_tpu_basic.json`

---

### ComfyUI-Crystools

**用途**：实时监控 TPU 硬件状态，包括 HBM 内存使用、Duty Cycle 和 TensorCore 利用率。

**功能特性：**

- **HBM 监控**：显示每个 TPU 芯片的 HBM 内存使用量和百分比
- **Duty Cycle**：显示 TPU 的工作负载百分比
- **TensorCore Util**：显示 TensorCore 的利用率

**配置：**

监控器默认显示在 ComfyUI 界面右上角。可在 Settings → Crystools 中配置：

- 显示的 TPU 数量（默认显示前 2 个）
- 刷新频率
- 监控器大小

**显示布局：**

```
| CPU | RAM | HBM 0 | Duty 0 | TC 0 |
|     |     | HBM 1 | Duty 1 | TC 1 |
```

---

## TPU 环境配置

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
