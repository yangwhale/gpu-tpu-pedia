# ComfyUI on TPU

This guide covers running ComfyUI on Google Cloud TPU, including installation and usage of TPU-optimized Custom Nodes.

**[中文文档](README.md)** | **English**

## Table of Contents

- [Requirements](#requirements)
- [Installing ComfyUI](#installing-comfyui)
- [Installing Custom Nodes](#installing-custom-nodes)
- [Starting ComfyUI](#starting-comfyui)
- [Custom Nodes Reference](#custom-nodes-reference)
  - [ComfyUI-CogVideoX-TPU](#comfyui-cogvideox-tpu)
  - [ComfyUI-Wan2.1-TPU](#comfyui-wan21-tpu)
  - [ComfyUI-Wan2.2-I2V-TPU](#comfyui-wan22-i2v-tpu)
  - [ComfyUI-Flux.2-TPU](#comfyui-flux2-tpu)
  - [ComfyUI-Crystools](#comfyui-crystools)
- [TPU Environment Setup](#tpu-environment-setup)
- [Troubleshooting](#troubleshooting)

---

## Requirements

- **Hardware**: Google Cloud TPU v4, v5, v6e or later
- **Operating System**: Ubuntu 20.04+ / Debian 11+
- **Python**: 3.10+ (3.12 recommended)
- **Dependencies**: JAX, PyTorch/XLA, tpu_info

## Installing ComfyUI

### 1. Clone the ComfyUI Repository

```bash
cd ~
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Video Processing Dependencies (Optional)

For video generation, ffmpeg is required:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# Or via conda
conda install ffmpeg
```

---

## Installing Custom Nodes

Custom Nodes must be placed in the `ComfyUI/custom_nodes/` directory.

### Method 1: Using ComfyUI Manager (Recommended for General Use)

```bash
cd ~/ComfyUI/custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
```

After launching ComfyUI, search and install Custom Nodes via the Manager interface.

### Method 2: From gpu-tpu-pedia (Recommended for TPU Users)

```bash
# Clone the gpu-tpu-pedia repository
git clone https://github.com/yangwhale/gpu-tpu-pedia.git
cd gpu-tpu-pedia/tpu/ComfyUI/custom_nodes

# Copy TPU Custom Nodes to ComfyUI
cp -r ComfyUI-CogVideoX-TPU ~/ComfyUI/custom_nodes/
cp -r ComfyUI-Wan2.1-TPU ~/ComfyUI/custom_nodes/
cp -r ComfyUI-Wan2.2-I2V-TPU ~/ComfyUI/custom_nodes/
cp -r ComfyUI-Flux.2-TPU ~/ComfyUI/custom_nodes/
cp -r ComfyUI-Crystools ~/ComfyUI/custom_nodes/

# Install dependencies
pip install -r ~/ComfyUI/custom_nodes/ComfyUI-Crystools/requirements.txt
```

---

## Starting ComfyUI

### Launching on TPU Machines

Since ComfyUI defaults to CUDA, you must use the `--cpu` flag on TPU machines. The TPU-specific nodes will automatically detect and utilize JAX/TPU:

```bash
cd ~/ComfyUI
python main.py --cpu --listen 0.0.0.0
```

**Command-line Arguments:**
- `--cpu`: Disables CUDA, uses CPU as the default device (TPU nodes automatically leverage JAX/TPU)
- `--listen 0.0.0.0`: Enables external access (for SSH port forwarding or direct access)
- `--port 8188`: Specifies the port (default: 8188)

### Running in Background

```bash
nohup python main.py --cpu --listen 0.0.0.0 > comfyui.log 2>&1 &
```

### Using Screen/Tmux

```bash
screen -S comfyui
python main.py --cpu --listen 0.0.0.0
# Ctrl+A, D to detach
```

---

## Custom Nodes Reference

### ComfyUI-CogVideoX-TPU

**Purpose**: Run CogVideoX 1.5-5B text-to-video (T2V) model on TPU with Splash Attention acceleration for high-quality video generation.

![CogVideoX T2V ComfyUI Workflow](custom_nodes/ComfyUI-CogVideoX-TPU/examples/cogvideox_t2v_720p_demo.png)

**Available Nodes:**

| Node Name | Function |
|-----------|----------|
| `CogVideoXTextEncoder` | Encodes text prompts using T5 to generate prompt embeddings |
| `CogVideoXTPUSampler` | Runs Transformer diffusion sampling on TPU to generate latents |
| `CogVideoXTPUVAEDecoder` | Decodes latents into video frames |

**Pipeline Flow:**

```
TextEncoder → TPUSampler → TPUVAEDecoder → CreateVideo → SaveVideo
```

**Using the Example Workflow:**

In the ComfyUI interface, click the **Templates** tab on the left, find **CogVideoX T2V 720p**, and click to load the complete workflow.

**Parameter Reference:**

- **CogVideoXTextEncoder**
  - `prompt`: Positive prompt describing the desired video
  - `negative_prompt`: Negative prompt for elements to avoid
  - `model_id`: Model path (default: `zai-org/CogVideoX1.5-5B`)

- **CogVideoXTPUSampler**
  - `height`: Video height (720)
  - `width`: Video width (1280)
  - `num_frames`: Frame count (81 = 5 seconds @ 16fps)
  - `num_inference_steps`: Sampling steps (50)
  - `guidance_scale`: CFG strength (6.0)
  - `seed`: Random seed for reproducibility

- **CogVideoXTPUVAEDecoder**
  - `fps`: Video frame rate (16)

**Performance Benchmarks (8x TPU v6e):**

| Metric | First Run (JIT Compile) | Cached (Subsequent) |
|--------|-------------------------|---------------------|
| Transformer (50 steps) | 126s | 104s |
| Per-step inference | 2.28s | 2.08s |
| VAE decode | 6.24s | 1.78s |
| Total time | 152s | 108s |

**Technical Highlights:**

- **Splash Attention**: TPU-optimized attention implementation using exp2 instead of exp for better TPU performance
- **Tensor Parallelism**: Supports weight sharding across TPU devices (dp=2, tp=4)
- **SafeTensors Loading**: Uses `use_safetensors=True` for secure model loading
- **Protobuf Conflict Resolution**: Pre-loads Tokenizer to avoid protobuf version conflicts with JAX

---

### ComfyUI-Wan2.1-TPU

**Purpose**: Run Wan2.1 text-to-video (T2V) model on TPU for high-quality video generation.

![Wan2.1 T2V ComfyUI Workflow](custom_nodes/ComfyUI-Wan2.1-TPU/examples/wan21_t2v_720p_demo.png)

**Available Nodes:**

| Node Name | Function |
|-----------|----------|
| `Wan21TextEncoder` | Encodes text prompts to generate prompt embeddings |
| `Wan21TPUSampler` | Runs diffusion sampling on TPU to generate latents |
| `Wan21TPUVAEDecoder` | Decodes latents into video frames |

**Pipeline Flow:**

```
TextEncoder → TPUSampler → TPUVAEDecoder → CreateVideo → SaveVideo
```

**Example Workflow:**

Load `custom_nodes/ComfyUI-Wan2.1-TPU/examples/wan21_t2v_720p.json`

**Parameter Reference:**

- **Wan21TextEncoder**
  - `prompt`: Positive prompt
  - `negative_prompt`: Negative prompt
  - `model_id`: Model path (e.g., `Wan-AI/Wan2.1-T2V-14B-Diffusers`)

- **Wan21TPUSampler**
  - `height`: Video height (720)
  - `width`: Video width (1280)
  - `num_frames`: Frame count (81 = 5 seconds @ 16fps)
  - `num_inference_steps`: Sampling steps (50)
  - `guidance_scale`: CFG strength (5.0)
  - `seed`: Random seed
  - `num_devices`: Number of TPU devices to use (1-8)

- **Wan21TPUVAEDecoder**
  - `fps`: Video frame rate (16)

**Performance Benchmarks (8x TPU v6e):**

| Metric | Value |
|--------|-------|
| Transformer (50 steps) | 227s |
| Per-step inference | 4.54s |
| VAE decode | 1.16s |
| Total time | 230s |

**Technical Highlights:**

- **14B Parameter Model**: Wan2.1-T2V-14B is a large-scale video generation model
- **Splash Attention**: TPU-optimized attention implementation
- **WanVAE**: Uses a dedicated video codec, avoiding protobuf conflicts with JAX

---

### ComfyUI-Wan2.2-I2V-TPU

**Purpose**: Run Wan2.2 image-to-video (I2V) model on TPU with dual Transformer A14B architecture for high-quality video generation from a reference image.

![Wan 2.2 I2V ComfyUI Workflow](custom_nodes/ComfyUI-Wan2.2-I2V-TPU/examples/wan22_i2v_full_view.png)

**Available Nodes:**

| Node Name | Function |
|-----------|----------|
| `Wan22I2VImageEncoder` | Encodes input image to generate CLIP and VAE conditions |
| `Wan22I2VTextEncoder` | Encodes text prompts to generate prompt embeddings |
| `Wan22I2VTPUSampler` | Runs dual Transformer diffusion sampling on TPU |
| `Wan22I2VTPUVAEDecoder` | Decodes latents into video frames |

**Pipeline Flow:**

```
Image → ImageEncoder ─┬→ TPUSampler → TPUVAEDecoder → CreateVideo → SaveVideo
                      │
TextEncoder ──────────┘
```

**Example Workflow:**

Load `custom_nodes/ComfyUI-Wan2.2-I2V-TPU/examples/wan22_i2v_720p.json`

**Parameter Reference:**

- **Wan22I2VImageEncoder**
  - `image`: Input image (first frame)
  - `model_id`: Model path (e.g., `Wan-AI/Wan2.2-I2V-14B-720P-Diffusers`)

- **Wan22I2VTextEncoder**
  - `prompt`: Positive prompt
  - `negative_prompt`: Negative prompt
  - `model_id`: Model path

- **Wan22I2VTPUSampler**
  - `height`: Video height (720)
  - `width`: Video width (1280)
  - `num_frames`: Frame count (81 = 5 seconds @ 16fps)
  - `num_inference_steps`: Sampling steps (50)
  - `guidance_scale`: CFG strength (5.0)
  - `shift`: Timestep distribution shift (5.0)
  - `seed`: Random seed
  - `num_devices`: Number of TPU devices to use (1-8)
  - `boundary_ratio`: A14B model switching ratio (0.9)

- **Wan22I2VTPUVAEDecoder**
  - `fps`: Video frame rate (16)

**Technical Highlights:**

- **Dual Transformer Architecture (A14B)**: Uses `boundary_ratio=0.9` to switch between two models—the primary model handles the first 90% of denoising steps, then the auxiliary model refines the final 10%
- **Splash Attention**: TPU-optimized attention implementation, significantly accelerating inference
- **Image Conditioning**: Supports input image as a conditioning signal for video generation

---

### ComfyUI-Flux.2-TPU

**Purpose**: Run Flux.2 image generation model on TPU.

**Available Nodes:**

| Node Name | Function |
|-----------|----------|
| `FluxTPUTextEncoder` | Encodes text prompts |
| `FluxTPUSampler` | Runs diffusion sampling on TPU |
| `FluxTPUVAEDecoder` | Decodes latents into images |

**Example Workflow:**

Load `custom_nodes/ComfyUI-Flux.2-TPU/examples/flux2_tpu_basic.json`

---

### ComfyUI-Crystools

**Purpose**: Real-time hardware monitoring. On TPU environments, automatically detects and displays TPU device information.

![Crystools TPU Monitor](custom_nodes/ComfyUI-Crystools/ComfyUI_Crystools_demo.png)

**Features:**

- **CPU Monitoring**: Displays CPU utilization
- **RAM Monitoring**: Shows system memory usage and percentage
- **TPU Monitoring** (per device):
  - **HBM**: High Bandwidth Memory usage and percentage
  - **Busy**: TPU busy state percentage
  - **MFU**: Model FLOPS Utilization (computational efficiency)

**Configuration:**

The monitor appears in the top menu bar of the ComfyUI interface. Configure via Settings → Crystools:

- Show/hide individual monitoring metrics
- Refresh rate (default: 0.5 seconds)
- Monitor dimensions (width/height)

---

## TPU Environment Setup

### 0. Configure Model Storage Path (Recommended)

TPU VM local disks have limited capacity (typically 100GB), while large models (e.g., Wan2.1-14B) require substantial storage. Symlinking the model directory to shared memory `/dev/shm` is recommended:

```bash
# Create model directory in shared memory
mkdir -p /dev/shm/comfyui_models

# Copy existing models to shared memory
cp -r ~/ComfyUI/models/* /dev/shm/comfyui_models/

# Remove original models directory and create symlink
rm -rf ~/ComfyUI/models
ln -s /dev/shm/comfyui_models ~/ComfyUI/models

# Verify symlink
ls -la ~/ComfyUI/models
```

**Note**: `/dev/shm` uses RAM as storage and data will be lost on reboot. For persistent storage, consider:
- Mounting a GCS bucket
- Using a persistent disk

### 1. Install Core Dependencies

```bash
# Install core packages
pip install huggingface-hub
pip install -U transformers datasets evaluate accelerate timm flax numpy
pip install torchax
pip install jax[tpu]
pip install tensorflow-cpu

# Install utilities
pip install sentencepiece
sudo apt install ffmpeg -y
pip install imageio[ffmpeg]
pip install tpu-info
pip install matplotlib
```

### 2. Configure Environment Variables

```bash
# Set Hugging Face cache directory (use shared memory for speed)
export HF_HOME=/dev/shm

# Set Hugging Face Token
export HF_TOKEN=<your HF_TOKEN>

# JAX compilation cache (speeds up repeated runs)
export JAX_COMPILATION_CACHE_DIR=/dev/shm/jax_cache
```

### 3. Install diffusers-tpu (TPU-Optimized Diffusers)

```bash
# Clone diffusers-tpu project (includes TorchAx/Flax VAE implementation)
git clone https://github.com/yangwhale/diffusers-tpu.git
cd diffusers-tpu
pip install -e .
cd ..
```

### 4. Verify TPU Availability

```python
import jax
print(jax.devices())
# Should display [TpuDevice(...), TpuDevice(...), ...]
```

### 5. Check TPU Status

```bash
# Using tpu_info CLI
tpu-info

# Or via Python
python -c "from tpu_info import device; print(device.get_local_chips())"
```

---

## Troubleshooting

### 1. "No module named 'tpu_info'"

```bash
pip install tpu_info
```

### 2. "Could not find TPU devices"

Ensure you're running on a TPU VM, or check TPU environment variables:

```bash
# Check TPU name
echo $TPU_NAME
echo $TPU_LOAD_LIBRARY
```

### 3. "JAX TPU init failed"

Likely a libtpu version mismatch:

```bash
pip install --upgrade jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### 4. ComfyUI Shows GPU Instead of TPU

Ensure:
1. You're using the `--cpu` flag at startup
2. ComfyUI-Crystools-TPU is correctly installed (not the original ComfyUI-Crystools)

### 5. Video Save Fails

Install ffmpeg:

```bash
sudo apt-get install ffmpeg
```

### 6. Out of Memory (OOM)

- Reduce `num_frames`
- Reduce `height`/`width`
- Reduce batch size

---

## Related Links

- [ComfyUI Official Repository](https://github.com/comfyanonymous/ComfyUI)
- [ComfyUI-TPU (Wan + Flux)](https://github.com/yangwhale/ComfyUI-TPU)
- [ComfyUI-Crystools-TPU](https://github.com/yangwhale/ComfyUI-Crystools-TPU)
- [JAX Official Documentation](https://jax.readthedocs.io/)
- [tpu_info](https://github.com/google/tpu_info)

---

## License

MIT License
