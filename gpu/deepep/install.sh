#!/bin/bash

set -ex
export NVSHMEM_IBGDA_SUPPORT=1
export NVSHMEM_USE_GDRCOPY=1
export USE_NVPEERMEM=1
export GDRCOPY_HOME=/opt/deepep/gdrcopy
export NVSHMEM_HOME=/opt/deepep/nvshmem
export CUDA_HOME=/usr/local/cuda


# 切换到用户主目录
pushd ~

# 设置非交互式安装
export DEBIAN_FRONTEND=noninteractive

# 更新包管理器
apt-get update -y -qq

#==============================================================================
# 安装 DOCA OFED
#==============================================================================
if ! command -v ofed_info &> /dev/null; then
    echo "DOCA OFED not found, installing now..."
    wget https://www.mellanox.com/downloads/DOCA/DOCA_v3.0.0/host/doca-host_3.0.0-058000-25.04-ubuntu2404_amd64.deb
    dpkg -i doca-host_3.0.0-058000-25.04-ubuntu2404_amd64.deb
    apt-get update -y -qq && apt-get -y install doca-ofed
    ofed_info -s
else
    echo "DOCA OFED is already installed"
    ofed_info -s
fi

#==============================================================================
# 安装 GPU 驱动和 CUDA
#==============================================================================
if ! lsmod | grep -q nvidia; then
    echo "GPU driver module not found, installing now..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt-get update -y && apt install nvidia-open-575 -y
    apt-get -y install cuda-toolkit-12.9
    nvidia-smi
else
    echo "GPU driver is already installed"
    nvidia-smi
fi

#==============================================================================
# 构建 gdrcopy
#==============================================================================
if [ "$NVSHMEM_USE_GDRCOPY" == "1" ]; then
    echo "NVSHMEM_USE_GDRCOPY=1, checking if gdrcopy is installed..."
    
    if ! ls /usr/lib/x86_64-linux-gnu/libgdr* >/dev/null 2>&1; then
        echo "gdrcopy library not found, installing now..."
        
        # 安装依赖
        apt-get install -y -qq --no-install-recommends \
            build-essential devscripts debhelper fakeroot \
            pkg-config dkms -y
        
        # 构建 gdrcopy
        rm -rf /tmp/gdrcopy
        git clone https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy
        cd /tmp/gdrcopy && git checkout tags/v2.5.1
        
        make -j$(nproc)
        make prefix=/opt/deepep/gdrcopy CUDA=/usr/local/cuda install
        
        cd packages
        CUDA=/usr/local/cuda ./build-deb-packages.sh
        dpkg -i *.deb
        
        echo "gdrcopy installation complete"
        
        # 清理
        pushd ~
        rm -rf /tmp/gdrcopy
        
        # 验证安装 (可选)
        # /opt/deepep/gdrcopy/bin/gdrcopy_copybw
    else
        echo "gdrcopy is already installed"
    fi
# SGLang need this
    if ! grep -q "PeerMappingOverride=1" "/etc/modprobe.d/nvidia-graphics-drivers-kms.conf" 2>/dev/null; then
        echo 'options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"' | \
            sudo tee -a /etc/modprobe.d/nvidia-graphics-drivers-kms.conf
    else
        cat /etc/modprobe.d/nvidia-graphics-drivers-kms.conf
    fi

else
    echo "NVSHMEM_USE_GDRCOPY=0, setting driver config..."
    
fi

#==============================================================================
# 重启节点（如有必要）
#==============================================================================
REBOOT_NEEDED=false

# 检查是否需要重启的条件
if ! lsmod | grep -q nvidia; then
    REBOOT_NEEDED=true
elif [ "$USE_NVPEERMEM" == "1" ] && ! lsmod | grep -q nvidia_peermem;
 then
    REBOOT_NEEDED=true
elif [ "$NVSHMEM_USE_GDRCOPY" != "1" ] && ! grep -q "PeerMappingOverride=1" /proc/driver/nvidia/params 2>/dev/null; then
    REBOOT_NEEDED=true
elif [ "$NVSHMEM_USE_GDRCOPY" == "1" ] && ! lsmod | grep -q gdrdrv; then
    REBOOT_NEEDED=true
fi

if [ "$REBOOT_NEEDED" == "true" ]; then
    echo "Rebooting the node to load driver and modules..."
    reboot
else
    echo "Modules are loaded correctly"
fi

#==============================================================================
# 加载 nvpeermem
#==============================================================================
if [ "$USE_NVPEERMEM" == "1" ]; then
    echo "USE_NVPEERMEM=1, loading nvidia_peermem..."
    modprobe nvidia_peermem
    
    if lsmod | grep -q nvidia_peermem; then
        echo "nvidia_peermem loaded"
    else
        echo "Failed to load nvidia_peermem module, exiting..."
        exit 1
    fi
fi

#==============================================================================
# 构建 NVSHMEM
#==============================================================================
NVSHMEM_INFO_CMD="${NVSHMEM_HOME}/bin/nvshmem-info"

if [ ! -x "$NVSHMEM_INFO_CMD" ]; then
    echo "NVSHMEM not found, installing now..."
    
    # 安装依赖
    apt-get install -y -qq --no-install-recommends \
        python3-venv python3-pip ninja-build cmake \
        python3.12-dev python3.12 \
        build-essential devscripts debhelper dkms git
    
    # 构建 NVSHMEM
    BUILD_DIR="/tmp/nvshmem_build_src"
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    
    wget https://developer.nvidia.com/downloads/assets/secure/nvshmem/nvshmem_src_cuda12-all \
        -O "${BUILD_DIR}/nvshmem_src_cuda12-all.tar.gz"
    tar -xvf "${BUILD_DIR}/nvshmem_src_cuda12-all.tar.gz" -C "$BUILD_DIR"
    cd "${BUILD_DIR}/nvshmem_src"
    
    # 配置构建选项
    CUDA_HOME="$CUDA_HOME" \
    GDRCOPY_HOME="$GDRCOPY_HOME" \
    NVSHMEM_SHMEM_SUPPORT=0 \
    NVSHMEM_UCX_SUPPORT=0 \
    NVSHMEM_USE_NCCL=0 \
    NVSHMEM_MPI_SUPPORT=0 \
    NVSHMEM_PMIX_SUPPORT=0 \
    NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
    NVSHMEM_USE_GDRCOPY="$NVSHMEM_USE_GDRCOPY" \
    NVSHMEM_IBGDA_SUPPORT="$NVSHMEM_IBGDA_SUPPORT" \
    cmake -GNinja -S . -B build/ -DCMAKE_INSTALL_PREFIX="$NVSHMEM_HOME"
    
    cmake --build build/ --target install
    echo "NVSHMEM installation complete."
    
    # 清理
    pushd ~
    rm -rf "$BUILD_DIR"
else
    echo "NVSHMEM is already installed"
fi

#==============================================================================
# 验证 NVSHMEM 配置
#==============================================================================
echo "Verifying NVSHMEM configuration..."

if [ -x "$NVSHMEM_INFO_CMD" ]; then
    if "$NVSHMEM_INFO_CMD" -a | grep -q "NVSHMEM_IBGDA_SUPPORT=ON"; then
        echo "NVSHMEM_IBGDA_SUPPORT is enabled correctly."
    else
        echo "NVSHMEM_IBGDA_SUPPORT is not enabled, exiting..."
        exit 1
    fi
else
    echo "nvshmem-info command not found after installation, exiting..."
    exit 1
fi

#==============================================================================
# 构建 DeepEP
#==============================================================================
if ! python3 -c "import deep_ep" &> /dev/null; then
    echo "deepep not found, installing now..."
    
    # 安装 PyTorch
    pip3 install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu129 \
        --break-system-packages
    
    # 构建 DeepEP
    BUILD_DIR="/opt/deepep_build"
    rm -rf "$BUILD_DIR"
    git clone https://github.com/deepseek-ai/DeepEP.git "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    # 设置环境变量
    export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/opt/deepep/nvshmem/lib:/opt/deepep/gdrcopy/lib:$LD_LIBRARY_PATH
    
    # 构建和安装（保持架构一致性）
    TORCH_CUDA_ARCH_LIST=10.0 NVSHMEM_DIR=/opt/deepep/nvshmem python3 setup.py build
    TORCH_CUDA_ARCH_LIST=10.0 NVSHMEM_DIR=/opt/deepep/nvshmem python3 setup.py install
    
    # 验证安装
    if ! python3 -c "import deep_ep" &> /dev/null; then
        echo "Failed to install DeepEP, exiting..."
        exit 1
    else
        echo "DeepEP installation complete."
    fi
else
    echo "deepep is already installed"
fi

echo "All installations completed successfully!"

#sudo python3 /opt/deepep_build/tests/test_low_latency.py

# export RANK=0
# export WORLD_SIZE=2
# export MASTER_ADDR=10.4.0.14
# echo "Starting test with RANK=${RANK}, WORLD_SIZE=${WORLD_SIZE}, MASTER_ADDR=${MASTER_ADDR}"
# sudo -E python3 /opt/deepep_build/tests/test_internode.py

