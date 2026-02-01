#!/bin/bash
# DeepEP Unified Installation Script for GCP B200 with RoCE
# Incorporates all lessons learned from debugging IBGDA initialization
#
# Key discoveries:
# 1. PyTorch ships nvidia-nvshmem-cu13 v3.4.5, causing version conflict
#    Solution: LD_PRELOAD=/opt/deepep/nvshmem/lib/libnvshmem_host.so.3
# 2. NVSHMEM defaults to IBRC transport, not IBGDA
#    Solution: NVSHMEM_REMOTE_TRANSPORT=ibgda
# 3. IBGDA with cpu handler requires GDRCopy
#    Solution: Build NVSHMEM with NVSHMEM_USE_GDRCOPY=ON
# 4. nvidia_peermem may fail to load, but dma-buf works via ibv_reg_dmabuf_mr
# 5. GCP RoCE uses rocep* device names
#    Solution: NVSHMEM_HCA_PREFIX=rocep
# 6. Ubuntu 24.04 requires DOCA-OFED for MLX5DV IBGDA extensions
#    The bundled rdma-core 50.0 lacks MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT
#    Solution: Install doca-ofed-userspace from NVIDIA repository
# 7. NVIDIA driver needs PeerMappingOverride=1 for dma-buf GPU memory registration
#    Solution: Add modprobe config and reboot
# 8. GDRCopy library needs LD_PRELOAD for runtime loading
#    Solution: Add libgdrapi.so to LD_PRELOAD alongside libnvshmem_host.so

set -ex

export DEBIAN_FRONTEND=noninteractive

# Configuration
CUDA_VERSION="12.9"
NVSHMEM_VERSION="3.5.19"
GDRCOPY_VERSION="2.5.1"
DOCA_OFED_VERSION="2.9.0"
DEEPEP_REPO="https://github.com/deepseek-ai/DeepEP.git"
INSTALL_PREFIX="/opt/deepep"

# Paths
GDRCOPY_HOME="${INSTALL_PREFIX}/gdrcopy"
NVSHMEM_HOME="${INSTALL_PREFIX}/nvshmem"
CUDA_HOME="/usr/local/cuda"
BUILD_DIR="/tmp/deepep_build"

echo "=========================================="
echo "DeepEP Unified Installation Script"
echo "=========================================="
echo "CUDA: ${CUDA_VERSION}"
echo "NVSHMEM: ${NVSHMEM_VERSION}"
echo "GDRCopy: ${GDRCOPY_VERSION}"
echo "Install prefix: ${INSTALL_PREFIX}"
echo "=========================================="

# Step 1: Install CUDA if not present
install_cuda() {
    echo "[1/6] Checking CUDA installation..."
    if command -v nvcc &> /dev/null; then
        echo "CUDA already installed: $(nvcc --version | grep release)"
        return 0
    fi

    echo "Installing CUDA ${CUDA_VERSION}..."
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt-get update -qq
    apt-get install -y cuda-toolkit-12-9
    rm cuda-keyring_1.1-1_all.deb

    # Verify
    export PATH=${CUDA_HOME}/bin:$PATH
    nvcc --version
}

# Step 2: Install GDRCopy
install_gdrcopy() {
    echo "[2/6] Checking GDRCopy installation..."

    if lsmod | grep -q gdrdrv && [ -f "${GDRCOPY_HOME}/lib/libgdrapi.so" ]; then
        echo "GDRCopy already installed and loaded"
        return 0
    fi

    echo "Installing GDRCopy ${GDRCOPY_VERSION}..."
    apt-get install -y -qq build-essential devscripts debhelper fakeroot pkg-config dkms

    rm -rf /tmp/gdrcopy
    git clone https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy
    cd /tmp/gdrcopy && git checkout tags/v${GDRCOPY_VERSION}

    # Build and install library
    make -j$(nproc)
    make prefix=${GDRCOPY_HOME} CUDA=${CUDA_HOME} install

    # Build and install kernel module via deb packages
    cd packages
    CUDA=${CUDA_HOME} ./build-deb-packages.sh
    dpkg -i gdrdrv-dkms_*.deb libgdrapi_*.deb || true

    # Configure ldconfig
    echo "${GDRCOPY_HOME}/lib" > /etc/ld.so.conf.d/gdrcopy.conf
    ldconfig

    # Load module
    modprobe gdrdrv || true

    # Verify
    if lsmod | grep -q gdrdrv; then
        echo "GDRCopy installed successfully, gdrdrv loaded"
    else
        echo "WARNING: gdrdrv module not loaded, may need reboot"
    fi

    cd ~
    rm -rf /tmp/gdrcopy
}

# Step 3: Install DOCA-OFED userspace (Ubuntu 24.04 requirement)
install_doca_ofed() {
    echo "[3/7] Checking DOCA-OFED installation..."

    # Check if already installed
    if dpkg -l doca-ofed-userspace 2>/dev/null | grep -q '^ii'; then
        echo "DOCA-OFED userspace already installed"
        return 0
    fi

    echo "Installing DOCA-OFED ${DOCA_OFED_VERSION} userspace..."

    # Add DOCA repository
    curl -fsSL https://linux.mellanox.com/public/repo/doca/GPG-KEY-Mellanox.pub | \
        gpg --dearmor -o /usr/share/keyrings/doca.gpg
    echo "deb [signed-by=/usr/share/keyrings/doca.gpg] https://linux.mellanox.com/public/repo/doca/${DOCA_OFED_VERSION}/ubuntu24.04/x86_64/ ./" \
        > /etc/apt/sources.list.d/doca.list
    apt-get update -qq

    # Install specific mft version required by DOCA-OFED
    apt-get install -y mft=4.30.0-139 || true
    apt-get install -y doca-ofed-userspace

    # Verify MLX5DV extensions are available
    if grep -q "MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT" /usr/include/infiniband/mlx5_api.h 2>/dev/null; then
        echo "DOCA-OFED installed successfully with MLX5DV IBGDA extensions"
    else
        echo "WARNING: MLX5DV IBGDA extensions not found after DOCA-OFED installation"
    fi
}

# Step 4: Install NVSHMEM with IBGDA and GDRCopy support
install_nvshmem() {
    echo "[4/7] Checking NVSHMEM installation..."

    if [ -x "${NVSHMEM_HOME}/bin/nvshmem-info" ]; then
        local installed_version=$(${NVSHMEM_HOME}/bin/nvshmem-info -v 2>/dev/null | head -1)
        local has_ibgda=$(${NVSHMEM_HOME}/bin/nvshmem-info -a 2>/dev/null | grep "NVSHMEM_IBGDA_SUPPORT=ON" || true)
        local has_gdrcopy=$(${NVSHMEM_HOME}/bin/nvshmem-info -a 2>/dev/null | grep "NVSHMEM_USE_GDRCOPY=ON" || true)

        if [ -n "$has_ibgda" ] && [ -n "$has_gdrcopy" ]; then
            echo "NVSHMEM ${installed_version} already installed with IBGDA and GDRCopy support"
            return 0
        else
            echo "NVSHMEM installed but missing IBGDA or GDRCopy, reinstalling..."
        fi
    fi

    echo "Installing NVSHMEM ${NVSHMEM_VERSION} with IBGDA and GDRCopy..."
    apt-get install -y -qq cmake ninja-build python3-pip

    local nvshmem_build="/tmp/nvshmem_build"
    rm -rf ${nvshmem_build}

    # Clone NVSHMEM from GitHub (NVIDIA download URLs often 404)
    git clone --depth 1 --branch v${NVSHMEM_VERSION}-1 https://github.com/NVIDIA/nvshmem.git ${nvshmem_build}
    cd ${nvshmem_build}

    # Build with IBGDA and GDRCopy support
    # Critical: Must enable NVSHMEM_IBGDA_SUPPORT and NVSHMEM_USE_GDRCOPY
    CUDA_HOME=${CUDA_HOME} \
    GDRCOPY_HOME=${GDRCOPY_HOME} \
    NVSHMEM_IBGDA_SUPPORT=1 \
    NVSHMEM_USE_GDRCOPY=1 \
    NVSHMEM_SHMEM_SUPPORT=0 \
    NVSHMEM_UCX_SUPPORT=0 \
    NVSHMEM_USE_NCCL=0 \
    NVSHMEM_MPI_SUPPORT=0 \
    NVSHMEM_PMIX_SUPPORT=0 \
    NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
    cmake -GNinja -S . -B build/ \
        -DCMAKE_INSTALL_PREFIX=${NVSHMEM_HOME} \
        -DCMAKE_CUDA_ARCHITECTURES=100 \
        -DNVSHMEM_BUILD_EXAMPLES=OFF \
        -DNVSHMEM_BUILD_PERFTEST=OFF

    cmake --build build/ --target install

    # Configure ldconfig
    echo "${NVSHMEM_HOME}/lib" > /etc/ld.so.conf.d/nvshmem.conf
    ldconfig

    # Verify IBGDA and GDRCopy support
    echo "Verifying NVSHMEM configuration..."
    ${NVSHMEM_HOME}/bin/nvshmem-info -a | grep -E "IBGDA|GDRCOPY"

    cd ~
    rm -rf ${nvshmem_build}
}

# Step 5: Configure NVIDIA driver options for peer memory
configure_nvidia_driver() {
    echo "[5/7] Configuring NVIDIA driver options..."

    local conf_file="/etc/modprobe.d/nvidia-peermem.conf"

    if grep -q "PeerMappingOverride=1" "$conf_file" 2>/dev/null; then
        echo "NVIDIA driver options already configured"
        cat "$conf_file"
        return 0
    fi

    echo 'options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"' > "$conf_file"
    echo "NVIDIA driver options configured"
    echo "WARNING: A reboot is required for PeerMappingOverride to take effect!"
}

# Step 6: Install PyTorch and DeepEP
install_deepep() {
    echo "[6/7] Installing PyTorch and DeepEP..."

    # Install PyTorch if not present
    if ! python3 -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
        echo "Installing PyTorch with CUDA ${CUDA_VERSION}..."
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129 --break-system-packages
    else
        echo "PyTorch already installed"
    fi

    # Clone and build DeepEP
    rm -rf ${BUILD_DIR}
    git clone ${DEEPEP_REPO} ${BUILD_DIR}
    cd ${BUILD_DIR}

    # Set up environment for build
    export LD_LIBRARY_PATH=${NVSHMEM_HOME}/lib:${GDRCOPY_HOME}/lib:${LD_LIBRARY_PATH}

    # Build DeepEP with correct NVSHMEM
    # B200 uses compute capability 10.0
    TORCH_CUDA_ARCH_LIST="10.0" \
    NVSHMEM_DIR=${NVSHMEM_HOME} \
    python3 setup.py build

    # Create symlink for the built module
    ln -sf build/lib.linux-x86_64-cpython-*/deep_ep_cpp.*.so .

    # Install to system
    TORCH_CUDA_ARCH_LIST="10.0" \
    NVSHMEM_DIR=${NVSHMEM_HOME} \
    python3 setup.py install --user

    # Verify installation
    if python3 -c "import deep_ep; print('DeepEP imported successfully')" 2>/dev/null; then
        echo "DeepEP installed successfully"
    else
        echo "WARNING: DeepEP import failed, may need environment setup"
    fi
}

# Step 7: Create unified environment script
create_env_script() {
    echo "[7/7] Creating unified environment script..."

    local env_script="${INSTALL_PREFIX}/unified-env.sh"

    cat > ${env_script} << 'ENVEOF'
#!/bin/bash
# DeepEP Unified Environment Script for GCP B200 with RoCE
# Configured for IBGDA transport mode with custom NVSHMEM 3.5.19

export NVSHMEM_HOME=/opt/deepep/nvshmem
export GDRCOPY_HOME=/opt/deepep/gdrcopy
export CUDA_HOME=/usr/local/cuda

# Library paths
export LD_LIBRARY_PATH=${NVSHMEM_HOME}/lib:${GDRCOPY_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}

# CRITICAL: Force our NVSHMEM and GDRCopy over PyTorch bundled versions
# PyTorch ships nvidia-nvshmem-cu12 v3.4.5, we need our 3.5.19 with IBGDA
# GDRCopy needs to be preloaded for NVSHMEM_IBGDA_NIC_HANDLER=cpu
export LD_PRELOAD="${NVSHMEM_HOME}/lib/libnvshmem_host.so.3:${GDRCOPY_HOME}/lib/libgdrapi.so.2"

# NVSHMEM IBGDA Configuration for RoCE
export NVSHMEM_REMOTE_TRANSPORT=ibgda
export NVSHMEM_IBGDA_NIC_HANDLER=cpu
export NVSHMEM_HCA_PREFIX=rocep
export NVSHMEM_IB_GID_INDEX=3
export NVSHMEM_DISABLE_CUDA_VMM=1

# DeepEP source path (for tests)
export PYTHONPATH=/tmp/deepep_build:${PYTHONPATH:-}

echo "DeepEP IBGDA environment configured:"
echo "  NVSHMEM_HOME=${NVSHMEM_HOME}"
echo "  GDRCOPY_HOME=${GDRCOPY_HOME}"
echo "  LD_PRELOAD=${LD_PRELOAD}"
echo "  NVSHMEM_REMOTE_TRANSPORT=${NVSHMEM_REMOTE_TRANSPORT}"
echo "  Using custom NVSHMEM 3.5.19 with DOCA-OFED and GDRCopy"
ENVEOF

    chmod +x ${env_script}
    echo "Environment script created: ${env_script}"
    echo "Usage: source ${env_script}"
}

# Main installation flow
main() {
    mkdir -p ${INSTALL_PREFIX}

    install_cuda
    install_gdrcopy
    install_doca_ofed   # Ubuntu 24.04 requirement for MLX5DV IBGDA extensions
    install_nvshmem
    configure_nvidia_driver
    install_deepep
    create_env_script

    echo ""
    echo "=========================================="
    echo "Installation Complete!"
    echo "=========================================="
    echo ""
    echo "IMPORTANT: A REBOOT IS REQUIRED for internode communication!"
    echo "The PeerMappingOverride driver option needs to be loaded."
    echo ""
    echo "After reboot, to use DeepEP, run:"
    echo "  source /opt/deepep/unified-env.sh"
    echo ""
    echo "For intranode testing (works without reboot):"
    echo "  source /opt/deepep/unified-env.sh"
    echo "  cd /tmp/deepep_build/tests"
    echo "  python3 test_intranode.py --num-processes 2"
    echo ""
    echo "For internode testing (requires reboot), on each node run:"
    echo "  export RANK=<node_rank>  # 0 for master, 1+ for workers"
    echo "  export WORLD_SIZE=<total_nodes>"
    echo "  export MASTER_ADDR=<master_ip>"
    echo "  export MASTER_PORT=29500"
    echo "  python3 /tmp/deepep_build/tests/test_internode.py"
    echo ""
    echo "=========================================="
}

# Run main
main "$@"
