#!/bin/bash
# DeepEP Installation Script for NVIDIA GPUs (B200/H100/A100)
# Extracted from deepep-installer.yaml DaemonSet
#
# This script installs the complete DeepEP stack:
#   1. DOCA OFED (Mellanox network drivers)
#   2. NVIDIA GPU Driver + CUDA Toolkit
#   3. NVSHMEM with IBGDA support (traditional mode, no GDRCopy)
#   4. DeepEP (DeepSeek Expert Parallelism)
#
# Usage: sudo ./setup-deepep.sh [--skip-reboot]
#
# Requirements:
#   - Ubuntu 24.04
#   - NVIDIA GPU (B200/H100/A100)
#   - Root privileges

set -e

# ============================================================================
# Environment Variables
# ============================================================================
export NVSHMEM_IBGDA_SUPPORT=1
export NVSHMEM_USE_GDRCOPY=0  # Disabled: using traditional IBGDA with PeerMappingOverride
export NVSHMEM_HOME=/opt/deepep/nvshmem
export USE_NVPEERMEM=0  # Disabled: using PeerMappingOverride instead of nvidia_peermem
export CUDA_HOME=/usr/local/cuda

# GPU Architecture (change based on your GPU)
# B200: 10.0, H100: 9.0, A100: 8.0
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"10.0"}

export DEBIAN_FRONTEND=noninteractive

# ============================================================================
# Parse Arguments
# ============================================================================
SKIP_REBOOT=false
for arg in "$@"; do
    case $arg in
        --skip-reboot)
            SKIP_REBOOT=true
            shift
            ;;
    esac
done

# ============================================================================
# Helper Functions
# ============================================================================
log_info() {
    echo "[INFO] $1"
}

log_warn() {
    echo "[WARN] $1"
}

log_error() {
    echo "[ERROR] $1"
    exit 1
}

check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "This script must be run as root. Use: sudo $0"
    fi
}

# ============================================================================
# Step 1: Install DOCA OFED
# ============================================================================
install_doca_ofed() {
    log_info "Checking RDMA/OFED installation..."

    # Check if mlx5 kernel modules are already loaded (cloud providers pre-install RDMA)
    if lsmod | grep -q mlx5_ib; then
        log_info "mlx5_ib kernel module already loaded (pre-installed by cloud provider or system)"
        if command -v ibv_devinfo &> /dev/null && ibv_devinfo 2>/dev/null | grep -q "PORT_ACTIVE"; then
            log_info "RDMA devices are active:"
            ibv_devinfo 2>/dev/null | grep -E "hca_id|link_layer|state" | head -12
        fi
        if command -v ofed_info &> /dev/null; then
            log_info "OFED version: $(ofed_info -s)"
        fi
        return 0
    fi

    # Check if RDMA is already available via ibverbs
    if command -v ibv_devinfo &> /dev/null && ibv_devinfo 2>/dev/null | grep -q "PORT_ACTIVE"; then
        log_info "RDMA is already available"
        ibv_devinfo 2>/dev/null | grep -E "hca_id|link_layer|state" | head -12
        return 0
    fi

    if command -v ofed_info &> /dev/null; then
        log_info "DOCA OFED is already installed: $(ofed_info -s)"
        return 0
    fi

    log_info "Installing DOCA OFED 3.2.1 LTS..."
    pushd /tmp > /dev/null

    # Add DOCA 3.2.1 LTS apt repository
    wget -qO - https://linux.mellanox.com/public/keys/GPG-KEY-Mellanox.pub | \
        gpg --dearmor -o /usr/share/keyrings/GPG-KEY-Mellanox.gpg

    cat > /etc/apt/sources.list.d/doca.list << EOF
deb [signed-by=/usr/share/keyrings/GPG-KEY-Mellanox.gpg] https://linux.mellanox.com/public/repo/doca/3.2.1/ubuntu24.04/x86_64/ ./
EOF

    apt-get update -y -qq
    apt-get -y install doca-ofed

    popd > /dev/null

    log_info "DOCA OFED installed: $(ofed_info -s)"
}

# ============================================================================
# Step 2: Install GPU Driver and CUDA
# ============================================================================
install_gpu_driver_cuda() {
    log_info "Checking GPU driver installation..."

    if lsmod | grep -q nvidia; then
        log_info "GPU driver is already installed"
        nvidia-smi
        return 0
    fi

    log_info "Installing NVIDIA GPU driver and CUDA toolkit..."
    pushd /tmp > /dev/null

    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt-get update -y
    # Use driver branch 580 for reproducibility (580.126.09)
    apt-get install -y nvidia-open-580
    apt-get install -y cuda-toolkit-13-0

    rm -f cuda-keyring_1.1-1_all.deb
    popd > /dev/null

    log_info "GPU driver and CUDA installed"
}

# ============================================================================
# Step 3: Build and Install GDRCopy
# ============================================================================
install_gdrcopy() {
    if [ "$NVSHMEM_USE_GDRCOPY" != "1" ]; then
        log_info "NVSHMEM_USE_GDRCOPY is not set to 1, skipping gdrcopy installation"
        return 0
    fi

    log_info "Checking gdrcopy installation..."

    if ls /usr/lib/x86_64-linux-gnu/libgdr* >/dev/null 2>&1; then
        log_info "gdrcopy is already installed"
        return 0
    fi

    log_info "Installing gdrcopy..."

    # Install dependencies
    apt-get install -y -qq --no-install-recommends \
        build-essential devscripts debhelper fakeroot pkg-config dkms

    # Build gdrcopy
    rm -rf /tmp/gdrcopy
    git clone https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy
    cd /tmp/gdrcopy && git checkout tags/v2.5.1

    make -j$(nproc)
    make prefix="$GDRCOPY_HOME" CUDA="$CUDA_HOME" install

    cd packages
    CUDA="$CUDA_HOME" ./build-deb-packages.sh
    dpkg -i *.deb

    # Cleanup
    rm -rf /tmp/gdrcopy

    log_info "gdrcopy installation complete"
}

# ============================================================================
# Step 4: Configure NVIDIA Kernel Options for SGLang
# ============================================================================
configure_nvidia_options() {
    local CONF_FILE="/etc/modprobe.d/nvidia-graphics-drivers-kms.conf"

    if grep -q "PeerMappingOverride=1" "$CONF_FILE" 2>/dev/null; then
        log_info "NVIDIA kernel options already configured"
        cat "$CONF_FILE"
        return 0
    fi

    log_info "Configuring NVIDIA kernel options for SGLang..."
    echo 'options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"' | \
        tee -a "$CONF_FILE"

    log_info "NVIDIA kernel options configured"
}

# ============================================================================
# Step 5: Load nvidia_peermem Module
# ============================================================================
load_nvpeermem() {
    if [ "$USE_NVPEERMEM" != "1" ]; then
        log_info "USE_NVPEERMEM is not set to 1, skipping nvidia_peermem loading"
        return 0
    fi

    log_info "Loading nvidia_peermem module..."

    modprobe nvidia_peermem || log_warn "Failed to load nvidia_peermem module, continuing anyway..."

    if lsmod | grep -q nvidia_peermem; then
        log_info "nvidia_peermem loaded successfully"
    else
        log_warn "nvidia_peermem module not loaded, but continuing installation..."
    fi
}

# ============================================================================
# Step 6: Check if Reboot is Needed
# ============================================================================
check_reboot_needed() {
    local REBOOT_NEEDED=false

    if ! lsmod | grep -q nvidia; then
        REBOOT_NEEDED=true
        log_warn "nvidia module not loaded, reboot required"
    elif [ "$USE_NVPEERMEM" == "1" ] && ! lsmod | grep -q nvidia_peermem; then
        REBOOT_NEEDED=true
        log_warn "nvidia_peermem module not loaded, reboot required"
    elif [ "$NVSHMEM_USE_GDRCOPY" != "1" ] && ! grep -q "PeerMappingOverride=1" /proc/driver/nvidia/params 2>/dev/null; then
        REBOOT_NEEDED=true
        log_warn "PeerMappingOverride not set, reboot required"
    elif [ "$NVSHMEM_USE_GDRCOPY" == "1" ] && ! lsmod | grep -q gdrdrv; then
        REBOOT_NEEDED=true
        log_warn "gdrdrv module not loaded, reboot required"
    fi

    if [ "$REBOOT_NEEDED" == "true" ]; then
        if [ "$SKIP_REBOOT" == "true" ]; then
            log_warn "Reboot is required but --skip-reboot was specified"
            log_warn "Please reboot manually and re-run this script to continue installation"
            exit 0
        else
            log_info "Rebooting the system to load driver and modules..."
            log_info "Please re-run this script after reboot to continue installation"
            reboot
        fi
    fi

    log_info "All required modules are loaded correctly"
}

# ============================================================================
# Step 7: Build and Install NVSHMEM
# ============================================================================
install_nvshmem() {
    local NVSHMEM_INFO_CMD="${NVSHMEM_HOME}/bin/nvshmem-info"

    log_info "Checking NVSHMEM installation..."

    if [ -x "$NVSHMEM_INFO_CMD" ]; then
        log_info "NVSHMEM is already installed"
        return 0
    fi

    log_info "Installing NVSHMEM..."

    # Install dependencies
    apt-get install -y -qq --no-install-recommends \
        python3-venv python3-pip ninja-build cmake \
        python3.12-dev python3.12 \
        build-essential devscripts debhelper dkms git \
        rdma-core libibverbs-dev librdmacm-dev  # Required for IBGDA MLX5 support

    # Build NVSHMEM
    local BUILD_DIR="/tmp/nvshmem_build_src"
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"

    # Download NVSHMEM 3.4.5 from GitHub (more stable than 3.5.x for DeepEP)
    wget -q https://github.com/NVIDIA/nvshmem/archive/refs/tags/v3.4.5-0.tar.gz \
        -O "${BUILD_DIR}/nvshmem_src.tar.gz"
    tar -xf "${BUILD_DIR}/nvshmem_src.tar.gz" -C "$BUILD_DIR"
    cd "${BUILD_DIR}/nvshmem-3.4.5-0"

    # Apply fix for GitHub Issue #21: Uninitialized ah_attr in RoCE environments
    # https://github.com/NVIDIA/nvshmem/issues/21
    # This fix initializes ah_attr to zero before use, fixing EINVAL on RoCE
    # The fix targets ibgda_setup_cq_and_qp() function in ibgda.cpp
    log_info "Applying RoCE fix (Issue #21) to NVSHMEM..."
    local IBGDA_FILE="src/modules/transport/ibgda/ibgda.cpp"
    if grep -q "memset.*ah_attr.*sizeof" "$IBGDA_FILE" 2>/dev/null; then
        log_info "RoCE fix already applied, skipping..."
    else
        # Insert memset after the ah_attr declaration in ibgda_setup_cq_and_qp()
        sed -i '/^[[:space:]]*struct ibv_ah_attr ah_attr;$/a\    memset(\&ah_attr, 0, sizeof(ah_attr));' \
            "$IBGDA_FILE"
        log_info "RoCE fix applied successfully"
    fi

    CUDA_HOME="$CUDA_HOME" \
    NVSHMEM_SHMEM_SUPPORT=0 NVSHMEM_UCX_SUPPORT=0 NVSHMEM_USE_NCCL=0 \
    NVSHMEM_MPI_SUPPORT=0 NVSHMEM_PMIX_SUPPORT=0 NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
    NVSHMEM_USE_GDRCOPY=0 \
    NVSHMEM_IBGDA_SUPPORT="$NVSHMEM_IBGDA_SUPPORT" \
    cmake -GNinja -S . -B build/ \
        -DCMAKE_INSTALL_PREFIX="$NVSHMEM_HOME" \
        -DCMAKE_CUDA_ARCHITECTURES=100 \
        -DNVSHMEM_BUILD_EXAMPLES=OFF \
        -DNVSHMEM_BUILD_PERFTEST=OFF

    cmake --build build/ --target install

    # Cleanup
    rm -rf "$BUILD_DIR"

    log_info "NVSHMEM installation complete"
}

# ============================================================================
# Step 8: Verify NVSHMEM Configuration
# ============================================================================
verify_nvshmem() {
    local NVSHMEM_INFO_CMD="${NVSHMEM_HOME}/bin/nvshmem-info"

    log_info "Verifying NVSHMEM configuration..."

    if [ ! -x "$NVSHMEM_INFO_CMD" ]; then
        log_error "nvshmem-info command not found after installation"
    fi

    if "$NVSHMEM_INFO_CMD" -a | grep -q "NVSHMEM_IBGDA_SUPPORT=ON"; then
        log_info "NVSHMEM_IBGDA_SUPPORT is enabled correctly"
    else
        log_error "NVSHMEM_IBGDA_SUPPORT is not enabled"
    fi
}

# ============================================================================
# Step 9: Build and Install DeepEP
# ============================================================================
install_deepep() {
    log_info "Checking DeepEP installation..."

    if python3 -c "import deep_ep" &> /dev/null; then
        log_info "DeepEP is already installed"
        return 0
    fi

    log_info "Installing DeepEP..."

    # Install PyTorch
    pip3 install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu130 \
        --break-system-packages

    # Build DeepEP
    local BUILD_DIR="/tmp/deepep_build"
    rm -rf "$BUILD_DIR"
    git clone https://github.com/deepseek-ai/DeepEP.git "$BUILD_DIR"
    cd "$BUILD_DIR"

    # Fix: Add CCCL include path for libcudacxx headers (cuda/std/tuple etc.)
    # NVSHMEM 3.4.5 requires libcudacxx headers which are in CUDA's cccl directory
    log_info "Patching DeepEP setup.py to add CCCL include path..."
    sed -i "s|include_dirs = \['csrc/'\]|cuda_home = os.getenv('CUDA_HOME', '/usr/local/cuda')\n    include_dirs = ['csrc/', f'{cuda_home}/include/cccl']  # cccl for libcudacxx headers|" setup.py

    export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${NVSHMEM_HOME}/lib:${LD_LIBRARY_PATH}

    TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" NVSHMEM_DIR="$NVSHMEM_HOME" \
        python3 setup.py build

    ln -sf build/lib.linux-x86_64-cpython-312/deep_ep_cpp.cpython-312-x86_64-linux-gnu.so

    TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" NVSHMEM_DIR="$NVSHMEM_HOME" \
        python3 setup.py install

    # Verify installation
    if ! python3 -c "import deep_ep" &> /dev/null; then
        log_error "Failed to install DeepEP"
    fi

    log_info "DeepEP installation complete"
}

# ============================================================================
# Main Installation Flow
# ============================================================================
main() {
    log_info "============================================"
    log_info "DeepEP Installation Script"
    log_info "============================================"
    log_info "Configuration:"
    log_info "  CUDA_HOME=$CUDA_HOME"
    log_info "  NVSHMEM_HOME=$NVSHMEM_HOME"
    log_info "  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
    log_info "  NVSHMEM_IBGDA_SUPPORT=$NVSHMEM_IBGDA_SUPPORT"
    log_info "  NVSHMEM_USE_GDRCOPY=$NVSHMEM_USE_GDRCOPY (disabled, using PeerMappingOverride)"
    log_info "  USE_NVPEERMEM=$USE_NVPEERMEM"
    log_info "============================================"

    check_root

    apt-get update -y -qq

    install_doca_ofed
    install_gpu_driver_cuda
    # GDRCopy skipped - using traditional IBGDA with PeerMappingOverride
    configure_nvidia_options
    load_nvpeermem
    check_reboot_needed
    install_nvshmem
    verify_nvshmem
    install_deepep

    log_info "============================================"
    log_info "DeepEP installation completed successfully!"
    log_info "============================================"
    log_info ""
    log_info "Environment variables to set in your shell:"
    log_info "  export CUDA_HOME=$CUDA_HOME"
    log_info "  export NVSHMEM_HOME=$NVSHMEM_HOME"
    log_info "  export LD_LIBRARY_PATH=\$NVSHMEM_HOME/lib:\$LD_LIBRARY_PATH"
    log_info ""
    log_info "To verify: python3 -c 'import deep_ep; print(deep_ep)'"
}

main "$@"
