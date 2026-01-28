#!/bin/bash
# =============================================================================
# SGLang 安装脚本
# 适用于 NVIDIA B200/H100/A100 GPU + CUDA 12.9
# 版本: v0.5.6.post2
# 更新日期: 2026-01-28
# =============================================================================

set -e

# =============================================================================
# 版本配置 (集中管理，便于更新)
# =============================================================================
SGLANG_VERSION="0.5.6.post2"
SGL_KERNEL_VERSION="0.3.21"
MOONCAKE_VERSION="0.3.8.post1"
NCCL_VERSION="2.28.3"
CUDNN_VERSION="9.16.0.29"
FLASHINFER_VERSION="0.5.3"
CMAKE_VERSION="3.31.1"

# GPU 架构配置
BUILD_TYPE="${BUILD_TYPE:-blackwell}"  # blackwell, hopper, ampere, all
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"  # 10.0=Blackwell, 9.0=Hopper, 8.0=Ampere

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
echo "SGLang v${SGLANG_VERSION} 安装脚本"
echo "=============================================="
echo ""
info "BUILD_TYPE: ${BUILD_TYPE}"
info "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"
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

    if [ "$NVME_COUNT" -ge 32 ]; then
        info "创建 RAID0 阵列 (32 块 NVMe SSD)..."

        # 检查 md0 是否已存在
        if [ -e /dev/md0 ]; then
            warn "RAID 设备 /dev/md0 已存在，跳过创建"
        else
            sudo mdadm --create /dev/md0 --level=0 --raid-devices=32 \
                /dev/disk/by-id/google-local-nvme-ssd-{28,18,31,10,25,15,26,17,16,29,30,20,19,14,12,7,6,21,11,24,23,13,27,22,3,5,8,9,4,1,2,0}
            sudo mkfs.ext4 -F /dev/md0
        fi

        sudo mkdir -p /lssd
        sudo mount /dev/md0 /lssd || warn "挂载失败，可能已挂载"
        sudo chmod a+w /lssd
        sudo mkdir -p /lssd/huggingface
        sudo chmod a+w /lssd/huggingface/
        success "LSSD 挂载完成: $(df -h /lssd | tail -1 | awk '{print $2}')"
        LSSD_MOUNTED=true
    else
        warn "NVMe SSD 数量不足 (需要 32 块，检测到 ${NVME_COUNT} 块)，跳过 LSSD 配置"
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
export BUILD_TYPE=${BUILD_TYPE}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
export CUDA_VERSION=12.9.1
export CMAKE_BUILD_PARALLEL_LEVEL=2

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
    software-properties-common netcat-openbsd \
    lsof zsh ccache tmux htop git-lfs tree \
    libopenmpi-dev libnuma1 libnuma-dev \
    libnl-3-200 libnl-route-3-200 libnl-route-3-dev libnl-3-dev \
    libgoogle-glog-dev libgtest-dev libjsoncpp-dev libunwind-dev \
    libboost-all-dev libssl-dev \
    libgrpc-dev libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc \
    pybind11-dev \
    libhiredis-dev libcurl4-openssl-dev \
    libczmq4 libczmq-dev \
    libfabric-dev \
    patchelf \
    libsubunit0 libsubunit-dev \
    gdb vim locales silversearcher-ag cloc bear less unzip gnupg

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

python3 -m pip install --break-system-packages --upgrade pip setuptools wheel

# 修复 DeepEP IBGDA symlink (如果需要)
if [ -f /usr/lib/$(uname -m)-linux-gnu/libmlx5.so.1 ]; then
    sudo ln -sf /usr/lib/$(uname -m)-linux-gnu/libmlx5.so.1 /usr/lib/$(uname -m)-linux-gnu/libmlx5.so 2>/dev/null || true
fi

success "Python 环境准备完成"

# =============================================================================
# 5. 安装 SGLang
# =============================================================================
info "安装 SGLang v${SGLANG_VERSION}..."

# 创建工作目录
sudo mkdir -p /sgl-workspace
sudo chmod a+w /sgl-workspace
cd /sgl-workspace

# 克隆 SGLang
if [ ! -d "sglang" ]; then
    info "克隆 SGLang 仓库..."
    git clone -b v${SGLANG_VERSION} --depth 1 https://github.com/sgl-project/sglang.git
else
    info "SGLang 目录已存在，更新到 v${SGLANG_VERSION}..."
    cd sglang
    git fetch --tags
    git checkout v${SGLANG_VERSION}
    cd ..
fi
cd sglang

# ★ 关键：先安装 sgl-kernel，再安装 SGLang
# 这样可以确保 sgl-kernel 版本正确，避免 SGLang 自动安装旧版本
info "安装 sgl-kernel v${SGL_KERNEL_VERSION}..."
python3 -m pip install --no-cache-dir --break-system-packages sgl-kernel==${SGL_KERNEL_VERSION}

info "安装 SGLang..."
python3 -m pip install --no-cache-dir --break-system-packages \
    -e "python[${BUILD_TYPE}]" \
    --extra-index-url https://download.pytorch.org/whl/cu129

success "SGLang 安装完成"

# =============================================================================
# 6. 安装 NVIDIA 库 (关键依赖)
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
# 7. 安装额外依赖
# =============================================================================
info "安装额外依赖..."

# Flashinfer cubin 缓存
FLASHINFER_CUBIN_DOWNLOAD_THREADS=8 FLASHINFER_LOGGING_LEVEL=warning \
    python3 -m flashinfer --download-cubin || warn "flashinfer cubin 下载跳过"

# Python 工具包
python3 -m pip install --no-cache-dir --break-system-packages \
    datamodel_code_generator \
    mooncake-transfer-engine==${MOONCAKE_VERSION} \
    pre-commit pytest black isort icdiff uv wheel \
    scikit-build-core py-spy cubloaty google-cloud-storage \
    pandas matplotlib tabulate termplotlib \
    nvidia-cudnn-frontend \
    nixl \
    openai httpx typing-extensions

success "额外依赖安装完成"

# =============================================================================
# 8. 开发工具安装
# =============================================================================
info "安装开发工具..."

# nsight-systems-cli
echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu2004/$(if [ "$(uname -m)" = "aarch64" ]; then echo "arm64"; else echo "amd64"; fi) /" | sudo tee /etc/apt/sources.list.d/nvidia-devtools.list
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/$(if [ "$(uname -m)" = "aarch64" ]; then echo "arm64"; else echo "x86_64"; fi)/7fa2af80.pub 2>/dev/null || true
sudo apt update -y && sudo apt install -y nsight-systems-cli || warn "nsight-systems-cli 安装跳过"

# 设置 locale
sudo locale-gen en_US.UTF-8 || true
export LANG=en_US.UTF-8
export LANGUAGE=en_US:en
export LC_ALL=en_US.UTF-8

# diff-so-fancy
curl -LSso /usr/local/bin/diff-so-fancy https://github.com/so-fancy/diff-so-fancy/releases/download/v1.4.4/diff-so-fancy 2>/dev/null || true
sudo chmod +x /usr/local/bin/diff-so-fancy 2>/dev/null || true

# clang-format
curl -LSso /usr/local/bin/clang-format https://github.com/muttleyxd/clang-tools-static-binaries/releases/download/master-32d3ac78/clang-format-16_linux-amd64 2>/dev/null || true
sudo chmod +x /usr/local/bin/clang-format 2>/dev/null || true

# CMake
ARCH=$(uname -m)
CMAKE_INSTALLER="cmake-${CMAKE_VERSION}-linux-${ARCH}"
wget -q "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_INSTALLER}.tar.gz" -O /tmp/${CMAKE_INSTALLER}.tar.gz || true
if [ -f /tmp/${CMAKE_INSTALLER}.tar.gz ]; then
    tar -xzf /tmp/${CMAKE_INSTALLER}.tar.gz -C /tmp
    sudo cp -r /tmp/${CMAKE_INSTALLER}/bin/* /usr/local/bin/
    sudo cp -r /tmp/${CMAKE_INSTALLER}/share/* /usr/local/share/
    rm -rf /tmp/${CMAKE_INSTALLER}*
fi

# just
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | sudo bash -s -- --to /usr/local/bin 2>/dev/null || true

success "开发工具安装完成"

# =============================================================================
# 9. 生成环境配置脚本
# =============================================================================
info "生成环境配置脚本..."

cat > /sgl-workspace/sglang-env.sh << 'SGLANG_ENV_EOF'
#!/bin/bash
# =============================================================================
# SGLang 环境变量配置
# 使用: source /sgl-workspace/sglang-env.sh
# =============================================================================

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

# ★ 关键：设置 LD_LIBRARY_PATH 以包含所有 nvidia pip 包的库
# 这是解决 libcudnn.so.9, libcusparseLt.so.0 等找不到问题的关键
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

# DeepEP 相关
if [ -d /opt/deepep/gdrcopy ]; then
    export GDRCOPY_HOME=/opt/deepep/gdrcopy
fi
if [ -d /opt/deepep/nvshmem ]; then
    export NVSHMEM_DIR=/opt/deepep/nvshmem
fi

echo "=============================================="
echo "SGLang 环境已加载"
echo "=============================================="
echo "CUDA_HOME: $CUDA_HOME"
echo "HF_HOME: ${HF_HOME:-~/.cache/huggingface}"
echo "LD_LIBRARY_PATH 已配置"
echo ""
echo "启动服务器示例:"
echo "  python3 -m sglang.launch_server \\"
echo "    --model-path Qwen/Qwen2.5-7B-Instruct \\"
echo "    --tp 4 \\"
echo "    --port 30000"
echo ""
echo "注意 TP 值必须能整除模型的 attention heads 数量:"
echo "  Qwen2.5-7B:  28 heads -> tp=1,2,4,7,14"
echo "  Qwen2.5-72B: 64 heads -> tp=1,2,4,8,16,32"
echo "  DeepSeek-R1: 128 heads -> tp=1,2,4,8,16,32,64"
echo "=============================================="
SGLANG_ENV_EOF
chmod +x /sgl-workspace/sglang-env.sh

success "环境配置脚本生成完成: /sgl-workspace/sglang-env.sh"

# 将 sglang-env.sh source 添加到 bashrc (持久化)
BASHRC_FILE="$HOME/.bashrc"
if ! grep -q 'source /sgl-workspace/sglang-env.sh' "$BASHRC_FILE" 2>/dev/null; then
    echo '' >> "$BASHRC_FILE"
    echo '# SGLang environment' >> "$BASHRC_FILE"
    echo 'if [ -f /sgl-workspace/sglang-env.sh ]; then' >> "$BASHRC_FILE"
    echo '    source /sgl-workspace/sglang-env.sh' >> "$BASHRC_FILE"
    echo 'fi' >> "$BASHRC_FILE"
    success "已添加 sglang-env.sh 到 $BASHRC_FILE"
else
    info "sglang-env.sh 已存在于 $BASHRC_FILE"
fi

# =============================================================================
# 10. Mooncake nvlink_allocator 修复
# =============================================================================
info "修复 Mooncake nvlink_allocator..."

cd /sgl-workspace
if [ ! -d "Mooncake" ]; then
    git clone -b v0.3.8 https://github.com/kvcache-ai/Mooncake.git || warn "Mooncake 克隆失败"
fi

if [ -d "Mooncake" ]; then
    MOONCAKE_SO_PATH=$(python3 -c "import mooncake; import os; print(os.path.dirname(mooncake.__file__))" 2>/dev/null || echo "")
    if [ -n "$MOONCAKE_SO_PATH" ]; then
        PATH=/usr/local/cuda/bin:$PATH nvcc \
            "Mooncake/mooncake-transfer-engine/nvlink-allocator/nvlink_allocator.cpp" \
            -o "${MOONCAKE_SO_PATH}/nvlink_allocator.so" \
            -shared -Xcompiler -fPIC -lcuda -I/usr/local/cuda/include 2>/dev/null \
            && success "Mooncake nvlink_allocator 修复完成" \
            || warn "Mooncake nvlink_allocator 编译跳过"
    fi
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
source /sgl-workspace/sglang-env.sh 2>/dev/null || true

# PyTorch
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || warn "PyTorch 导入失败"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || true
python3 -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')" 2>/dev/null || true

# SGLang
python3 -c "import sglang; print(f'SGLang: {sglang.__version__}')" 2>/dev/null || warn "SGLang 导入失败"

# sgl-kernel
python3 -c "import sgl_kernel; print('sgl_kernel: OK')" 2>/dev/null || warn "sgl_kernel 导入失败"

# flashinfer
python3 -c "import flashinfer; print('flashinfer: OK')" 2>/dev/null || warn "flashinfer 导入失败"

# DeepEP (可选)
python3 -c "import deep_ep; print('DeepEP: OK')" 2>/dev/null || \
    python3 -c "import deepep; print('DeepEP: OK')" 2>/dev/null || \
    warn "DeepEP 未安装 (MoE 模型如 DeepSeek-R1 需要)"

echo ""
echo "=============================================="
echo -e "${GREEN}SGLang v${SGLANG_VERSION} 安装完成！${NC}"
echo "=============================================="
echo ""
echo "已安装组件:"
echo "  - SGLang: v${SGLANG_VERSION}"
echo "  - sgl-kernel: v${SGL_KERNEL_VERSION}"
echo "  - mooncake: v${MOONCAKE_VERSION}"
echo "  - nvidia-nccl-cu12: v${NCCL_VERSION}"
echo "  - nvidia-cudnn-cu12: v${CUDNN_VERSION}"
echo ""
echo "下一步操作:"
echo ""
echo "1. 加载环境变量:"
echo "   source /sgl-workspace/sglang-env.sh"
echo ""
echo "2. 启动服务器 (以 Qwen2.5-7B 为例):"
echo "   python3 -m sglang.launch_server \\"
echo "     --model-path Qwen/Qwen2.5-7B-Instruct \\"
echo "     --tp 4 \\"
echo "     --port 30000 \\"
echo "     --host 0.0.0.0"
echo ""
echo "3. 测试服务器:"
echo "   curl http://localhost:30000/health"
echo ""
echo "4. 运行诊断 (如遇问题):"
echo "   python3 /path/to/sglang-installer/scripts/diagnose.py"
echo ""
echo "=============================================="
