#!/bin/bash
# =============================================================================
# vLLM 安装脚本
# 适用于 NVIDIA B200/H100/A100 GPU + CUDA 12.9
# 版本: v0.15.0rc2
# 更新日期: 2026-01-28
# 参考: https://github.com/vllm-project/vllm/blob/v0.15.0rc2/docker/Dockerfile
# =============================================================================

set -e

# =============================================================================
# 版本配置 (集中管理，便于更新)
# =============================================================================
VLLM_VERSION="0.14.1"
FLASHINFER_VERSION="0.5.3"
BITSANDBYTES_VERSION="0.46.1"
NCCL_VERSION="2.28.3"
CUDNN_VERSION="9.16.0.29"

# GPU 架构配置
# 10.0=Blackwell, 9.0=Hopper, 8.0=Ampere, 7.5=Turing, 7.0=Volta
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0 7.5 8.0 8.9 9.0 10.0 12.0}"

# 安装选项
INSTALL_FLASHINFER="${INSTALL_FLASHINFER:-true}"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

echo "=============================================="
echo "vLLM v${VLLM_VERSION} 安装脚本"
echo "=============================================="
echo ""
info "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"
info "INSTALL_FLASHINFER: ${INSTALL_FLASHINFER}"
echo ""

# =============================================================================
# 1. LSSD 挂载 (可选，检测 NVMe SSD 是否存在)
# =============================================================================
info "检查 LSSD 配置..."

LSSD_MOUNTED=false
if mountpoint -q /lssd 2>/dev/null; then
    success "LSSD 已挂载: $(df -h /lssd | tail -1 | awk '{print $2}')"
    LSSD_MOUNTED=true
elif ls /dev/disk/by-id/google-local-nvme-ssd-0 &>/dev/null; then
    NVME_COUNT=$(ls /dev/disk/by-id/ 2>/dev/null | grep -c "google-local-nvme-ssd" || echo 0)
    info "检测到 ${NVME_COUNT} 块 NVMe SSD"

    if [ "$NVME_COUNT" -ge 1 ]; then
        info "创建 RAID0 阵列 (${NVME_COUNT} 块 NVMe SSD)..."

        # 检查 md0 是否已存在
        if [ -e /dev/md0 ]; then
            warn "RAID 设备 /dev/md0 已存在，跳过创建"
        else
            # 动态获取所有 NVMe 设备
            NVME_DEVICES=$(ls /dev/disk/by-id/google-local-nvme-ssd-* 2>/dev/null | sort -V | tr '\n' ' ')
            sudo mdadm --create /dev/md0 --level=0 --raid-devices=$NVME_COUNT $NVME_DEVICES
            sudo mkfs.ext4 -F /dev/md0
        fi

        sudo mkdir -p /lssd
        sudo mount /dev/md0 /lssd || warn "挂载失败，可能已挂载"
        sudo chmod a+w /lssd
        sudo mkdir -p /lssd/huggingface
        sudo chmod a+w /lssd/huggingface/
        success "LSSD 挂载完成: $(df -h /lssd | tail -1 | awk '{print $2}')"
        LSSD_MOUNTED=true
    fi
else
    warn "未检测到 NVMe SSD，跳过 LSSD 配置"
fi

# 设置 HuggingFace 缓存目录
if [ "$LSSD_MOUNTED" = true ]; then
    export HF_HOME=/lssd/huggingface
    info "HuggingFace 缓存目录: ${HF_HOME}"

    # 将 HF_HOME 写入 bashrc (持久化)
    BASHRC_FILE="$HOME/.bashrc"
    if ! grep -q 'export HF_HOME=/lssd/huggingface' "$BASHRC_FILE" 2>/dev/null; then
        echo '' >> "$BASHRC_FILE"
        echo '# HuggingFace cache on LSSD (high-speed local SSD)' >> "$BASHRC_FILE"
        echo 'export HF_HOME=/lssd/huggingface' >> "$BASHRC_FILE"
        success "已添加 HF_HOME 到 $BASHRC_FILE"
    else
        info "HF_HOME 已存在于 $BASHRC_FILE"
    fi
fi

# =============================================================================
# 2. 环境变量设置
# =============================================================================
info "设置环境变量..."

export DEBIAN_FRONTEND=noninteractive
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
export VLLM_USAGE_SOURCE="production-script"

# DeepEP 相关路径 (如果已安装)
if [ -d /opt/deepep/gdrcopy ]; then
    export GDRCOPY_HOME=/opt/deepep/gdrcopy
    info "GDRCopy: ${GDRCOPY_HOME}"
fi
if [ -d /opt/deepep/nvshmem ]; then
    export NVSHMEM_DIR=/opt/deepep/nvshmem
    info "NVSHMEM: ${NVSHMEM_DIR}"
fi

# =============================================================================
# 3. 系统依赖安装
# =============================================================================
info "安装系统依赖..."

# 设置时区
echo 'tzdata tzdata/Areas select America' | sudo debconf-set-selections
echo 'tzdata tzdata/Zones/America select Los_Angeles' | sudo debconf-set-selections

sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    tzdata \
    software-properties-common \
    curl wget git \
    build-essential \
    ccache \
    libnuma-dev libibverbs-dev \
    ffmpeg libsm6 libxext6 libgl1 \
    python3-dev python3-pip \
    gdb vim locales less unzip

# 创建 python 软链接
if [ ! -L /usr/bin/python ]; then
    sudo ln -sf /usr/bin/python3 /usr/bin/python
fi

# 清理 apt 缓存
sudo rm -rf /var/lib/apt/lists/*
sudo apt-get clean

success "系统依赖安装完成"

# =============================================================================
# 4. Python 环境准备
# =============================================================================
info "准备 Python 环境..."

# Ubuntu 24.04 不自带 pip，需要先安装
if ! python3 -m pip --version &>/dev/null; then
    info "安装 python3-pip..."
    sudo apt-get install -y -qq python3-pip
fi

python3 -m pip install --break-system-packages --upgrade pip setuptools wheel

success "Python 环境准备完成"

# =============================================================================
# 5. 安装 PyTorch
# =============================================================================
info "安装 PyTorch (CUDA 12.9)..."

python3 -m pip install --break-system-packages \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu129

# 验证 PyTorch
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" \
    && success "PyTorch 安装成功" \
    || error "PyTorch 安装失败"

# =============================================================================
# 6. 安装 vLLM
# =============================================================================
info "安装 vLLM v${VLLM_VERSION}..."

# 创建工作目录
sudo mkdir -p /vllm-workspace
sudo chmod a+w /vllm-workspace
cd /vllm-workspace

# 方式1: 从 PyPI 安装 (推荐，更快)
python3 -m pip install --break-system-packages \
    vllm==${VLLM_VERSION} \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    && VLLM_INSTALLED=true \
    || VLLM_INSTALLED=false

# 方式2: 如果 PyPI 安装失败，从源码安装
if [ "$VLLM_INSTALLED" = false ]; then
    warn "PyPI 安装失败，尝试从源码安装..."

    if [ ! -d "vllm" ]; then
        git clone -b v${VLLM_VERSION} --depth 1 https://github.com/vllm-project/vllm.git
    else
        cd vllm
        git fetch --tags
        git checkout v${VLLM_VERSION}
        cd ..
    fi

    cd vllm
    python3 -m pip install --break-system-packages -e . --extra-index-url https://download.pytorch.org/whl/cu129
    cd ..
fi

success "vLLM 安装完成"

# =============================================================================
# 7. 安装 NVIDIA 库 (关键依赖)
# =============================================================================
info "安装 NVIDIA 库..."

# ★ 这些库必须用 --force-reinstall --no-deps 安装
# 否则可能会因为版本冲突导致 libcudnn.so.9 等找不到
python3 -m pip install --no-cache-dir --break-system-packages \
    nvidia-nccl-cu12==${NCCL_VERSION} --force-reinstall --no-deps

python3 -m pip install --no-cache-dir --break-system-packages \
    nvidia-cudnn-cu12==${CUDNN_VERSION} --force-reinstall --no-deps

python3 -m pip install --no-cache-dir --break-system-packages \
    nvidia-cusparselt-cu12 --force-reinstall --no-deps

success "NVIDIA 库安装完成"

# =============================================================================
# 8. 安装 FlashInfer (可选但推荐)
# =============================================================================
if [ "$INSTALL_FLASHINFER" = true ]; then
    info "安装 FlashInfer v${FLASHINFER_VERSION}..."

    # FlashInfer 使用新的包名 (flashinfer-python + flashinfer-cubin)
    python3 -m pip install --no-cache-dir --break-system-packages \
        flashinfer-python==${FLASHINFER_VERSION} \
        flashinfer-cubin==${FLASHINFER_VERSION} \
        && success "FlashInfer 安装成功" \
        || warn "FlashInfer 安装失败，将使用默认 attention backend"
fi

# =============================================================================
# 9. 安装额外依赖
# =============================================================================
info "安装额外依赖..."

python3 -m pip install --no-cache-dir --break-system-packages \
    bitsandbytes==${BITSANDBYTES_VERSION} \
    "timm>=1.0.17" \
    openai httpx \
    pandas matplotlib \
    ray \
    outlines \
    lm-format-enforcer \
    compressed-tensors

success "额外依赖安装完成"

# =============================================================================
# 10. 生成环境配置脚本
# =============================================================================
info "生成环境配置脚本..."

cat > /vllm-workspace/vllm-env.sh << 'VLLM_ENV_EOF'
#!/bin/bash
# =============================================================================
# vLLM 环境变量配置
# 使用: source /vllm-workspace/vllm-env.sh
# =============================================================================

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

# ★ 关键：设置 LD_LIBRARY_PATH 以包含所有 nvidia pip 包的库
NVIDIA_LIB_PATHS=""

# 系统级 pip 包
for d in /usr/local/lib/python3.*/dist-packages/nvidia/*/lib; do
    [ -d "$d" ] && NVIDIA_LIB_PATHS="${d}:${NVIDIA_LIB_PATHS}"
done

# 用户级 pip 包
for d in $HOME/.local/lib/python3.*/site-packages/nvidia/*/lib; do
    [ -d "$d" ] && NVIDIA_LIB_PATHS="${d}:${NVIDIA_LIB_PATHS}"
done

export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${NVIDIA_LIB_PATHS}${LD_LIBRARY_PATH}

# HuggingFace 缓存 (如果 LSSD 可用)
if [ -d /lssd/huggingface ]; then
    export HF_HOME=/lssd/huggingface
fi

# vLLM 配置
export VLLM_USAGE_SOURCE=production-script

# DeepEP 相关
if [ -d /opt/deepep/gdrcopy ]; then
    export GDRCOPY_HOME=/opt/deepep/gdrcopy
fi
if [ -d /opt/deepep/nvshmem ]; then
    export NVSHMEM_DIR=/opt/deepep/nvshmem
fi

echo "=============================================="
echo "vLLM 环境已加载"
echo "=============================================="
echo "CUDA_HOME: $CUDA_HOME"
echo "HF_HOME: ${HF_HOME:-~/.cache/huggingface}"
echo "LD_LIBRARY_PATH 已配置"
echo ""
echo "启动 OpenAI API 服务器示例:"
echo "  vllm serve Qwen/Qwen2.5-7B-Instruct \\"
echo "    --tensor-parallel-size 4 \\"
echo "    --port 8000"
echo ""
echo "或使用传统方式:"
echo "  python3 -m vllm.entrypoints.openai.api_server \\"
echo "    --model Qwen/Qwen2.5-7B-Instruct \\"
echo "    --tensor-parallel-size 4 \\"
echo "    --port 8000"
echo ""
echo "测试服务器:"
echo "  curl http://localhost:8000/v1/models"
echo "=============================================="
VLLM_ENV_EOF
chmod +x /vllm-workspace/vllm-env.sh

success "环境配置脚本生成完成: /vllm-workspace/vllm-env.sh"

# 将 vllm-env.sh source 添加到 bashrc (持久化)
BASHRC_FILE="$HOME/.bashrc"
if ! grep -q 'source /vllm-workspace/vllm-env.sh' "$BASHRC_FILE" 2>/dev/null; then
    echo '' >> "$BASHRC_FILE"
    echo '# vLLM environment' >> "$BASHRC_FILE"
    echo 'if [ -f /vllm-workspace/vllm-env.sh ]; then' >> "$BASHRC_FILE"
    echo '    source /vllm-workspace/vllm-env.sh' >> "$BASHRC_FILE"
    echo 'fi' >> "$BASHRC_FILE"
    success "已添加 vllm-env.sh 到 $BASHRC_FILE"
else
    info "vllm-env.sh 已存在于 $BASHRC_FILE"
fi

# =============================================================================
# 11. 验证安装
# =============================================================================
info "验证安装..."

echo ""
echo "=============================================="
echo "版本验证"
echo "=============================================="

# 加载环境
source /vllm-workspace/vllm-env.sh 2>/dev/null || true

# PyTorch
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || warn "PyTorch 导入失败"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || true
python3 -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')" 2>/dev/null || true

# vLLM
python3 -c "import vllm; print(f'vLLM: {vllm.__version__}')" 2>/dev/null || warn "vLLM 导入失败"

# FlashInfer
if [ "$INSTALL_FLASHINFER" = true ]; then
    python3 -c "import flashinfer; print('FlashInfer: OK')" 2>/dev/null || warn "FlashInfer 导入失败"
fi

# DeepEP (可选)
python3 -c "import deep_ep; print('DeepEP: OK')" 2>/dev/null || \
    python3 -c "import deepep; print('DeepEP: OK')" 2>/dev/null || \
    info "DeepEP 未安装 (MoE 模型可选)"

echo ""
echo "=============================================="
echo -e "${GREEN}vLLM v${VLLM_VERSION} 安装完成！${NC}"
echo "=============================================="
echo ""
echo "已安装组件:"
echo "  - vLLM: v${VLLM_VERSION}"
echo "  - FlashInfer: v${FLASHINFER_VERSION}"
echo "  - bitsandbytes: v${BITSANDBYTES_VERSION}"
echo "  - nvidia-nccl-cu12: v${NCCL_VERSION}"
echo "  - nvidia-cudnn-cu12: v${CUDNN_VERSION}"
echo ""
echo "下一步操作:"
echo ""
echo "1. 加载环境变量:"
echo "   source /vllm-workspace/vllm-env.sh"
echo ""
echo "2. 启动 OpenAI API 服务器:"
echo "   vllm serve Qwen/Qwen2.5-7B-Instruct \\"
echo "     --tensor-parallel-size 4 \\"
echo "     --port 8000 \\"
echo "     --host 0.0.0.0"
echo ""
echo "3. 测试服务器:"
echo "   curl http://localhost:8000/v1/models"
echo ""
echo "4. 发送请求:"
echo "   curl http://localhost:8000/v1/chat/completions \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"model\": \"Qwen/Qwen2.5-7B-Instruct\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'"
echo ""
echo "=============================================="
