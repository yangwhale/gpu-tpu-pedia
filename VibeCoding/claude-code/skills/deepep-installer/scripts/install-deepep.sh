#!/bin/bash
# DeepEP Installation Script for SGLang
# Adapted from deepep-installer.yaml for bare-metal/VM environments
# Assumes NVIDIA driver and IB driver are already installed

set -e

# ============================================================================
# Configuration
# ============================================================================
export NVSHMEM_IBGDA_SUPPORT=${NVSHMEM_IBGDA_SUPPORT:-1}
export NVSHMEM_USE_GDRCOPY=${NVSHMEM_USE_GDRCOPY:-1}
export GDRCOPY_HOME=${GDRCOPY_HOME:-/opt/deepep/gdrcopy}
export NVSHMEM_HOME=${NVSHMEM_HOME:-/opt/deepep/nvshmem}
export USE_NVPEERMEM=${USE_NVPEERMEM:-1}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"10.0"}  # B200

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_section() { echo -e "\n${GREEN}========== $1 ==========${NC}"; }

# ============================================================================
# Pre-flight Checks
# ============================================================================
log_section "Pre-flight Checks"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    log_error "Please run as root (sudo)"
    exit 1
fi

# Check NVIDIA driver
check_nvidia_driver() {
    log_info "Checking NVIDIA driver..."
    if lsmod | grep -q nvidia; then
        log_ok "NVIDIA driver module loaded"
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
        return 0
    else
        log_error "NVIDIA driver module NOT loaded"
        log_error "Please install NVIDIA driver first"
        exit 1
    fi
}

# Check CUDA
check_cuda() {
    log_info "Checking CUDA installation..."

    # Try common CUDA paths
    for cuda_path in "$CUDA_HOME" /usr/local/cuda /usr/local/cuda-12.9 /usr/local/cuda-12; do
        if [ -d "$cuda_path" ] && [ -f "$cuda_path/bin/nvcc" ]; then
            export CUDA_HOME="$cuda_path"
            CUDA_VERSION=$($CUDA_HOME/bin/nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
            log_ok "CUDA found at $CUDA_HOME (version $CUDA_VERSION)"
            return 0
        fi
    done

    log_warn "CUDA toolkit not found"
    return 1
}

# Install CUDA toolkit
install_cuda() {
    log_section "CUDA Toolkit Installation"

    if check_cuda; then
        log_ok "CUDA toolkit already installed, skipping"
        return 0
    fi

    log_info "Installing CUDA toolkit 12.9..."

    # Add NVIDIA CUDA repository
    if [ ! -f /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list ]; then
        log_info "Adding NVIDIA CUDA repository..."
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
        dpkg -i /tmp/cuda-keyring.deb
        apt-get update -y -qq
        rm -f /tmp/cuda-keyring.deb
    fi

    # Install CUDA toolkit (without driver - we already have it)
    apt-get install -y -qq cuda-toolkit-12-9

    # Set CUDA_HOME
    export CUDA_HOME=/usr/local/cuda-12.9

    # Create symlink if not exists
    if [ ! -e /usr/local/cuda ]; then
        ln -s /usr/local/cuda-12.9 /usr/local/cuda
    fi

    # Verify
    if [ -f "$CUDA_HOME/bin/nvcc" ]; then
        log_ok "CUDA toolkit installed successfully"
        $CUDA_HOME/bin/nvcc --version | grep release
    else
        log_error "CUDA toolkit installation failed"
        return 1
    fi
}

# Check IB driver
check_ib_driver() {
    log_info "Checking InfiniBand/RoCE driver..."

    # Check kernel module
    if lsmod | grep -q mlx5_core; then
        log_ok "mlx5_core module loaded"
    else
        log_warn "mlx5_core module NOT loaded"
        log_warn "IB/RoCE cards may not be present or driver not installed"
    fi

    # Try rdma link first (more reliable on modern systems)
    if command -v rdma &> /dev/null; then
        RDMA_DEVICES=$(rdma link 2>/dev/null | grep -c "state ACTIVE" || echo 0)
        if [ "$RDMA_DEVICES" -gt 0 ]; then
            log_ok "Found $RDMA_DEVICES active RDMA device(s)"
            rdma link 2>/dev/null | head -8
            return 0
        fi
    fi

    # Fallback to ibv_devinfo
    if command -v ibv_devinfo &> /dev/null; then
        IB_DEVICES=$(ibv_devinfo 2>/dev/null | grep "hca_id" | wc -l)
        if [ "$IB_DEVICES" -gt 0 ]; then
            log_ok "Found $IB_DEVICES IB device(s)"
            ibv_devinfo 2>/dev/null | grep -E "hca_id|port:|state:" | head -10
        else
            log_warn "No IB devices found via ibv_devinfo"
        fi
    else
        # Check sysfs as last resort
        if [ -d /sys/class/infiniband ] && [ "$(ls -A /sys/class/infiniband 2>/dev/null)" ]; then
            log_ok "RDMA devices found in /sys/class/infiniband"
            ls /sys/class/infiniband/
        else
            log_warn "No RDMA devices detected"
        fi
    fi
}

# Check OFED version
check_ofed() {
    log_info "Checking OFED/DOCA installation..."

    if command -v ofed_info &> /dev/null; then
        OFED_VERSION=$(ofed_info -s 2>/dev/null || echo "unknown")
        log_ok "OFED/DOCA installed: $OFED_VERSION"
        return 0
    else
        log_warn "OFED/DOCA not found (ofed_info command missing)"
        log_warn "System IB driver may work, but DOCA provides better IBGDA support"
        return 1
    fi
}

# Run pre-flight checks
check_nvidia_driver
CUDA_INSTALLED=true
check_cuda || CUDA_INSTALLED=false
check_ib_driver
DOCA_INSTALLED=true
check_ofed || DOCA_INSTALLED=false

# ============================================================================
# DOCA OFED Installation (Optional)
# ============================================================================
install_doca_ofed() {
    log_section "DOCA OFED Installation"

    if [ "$DOCA_INSTALLED" = true ]; then
        log_ok "DOCA OFED already installed, skipping"
        return 0
    fi

    # Skip DOCA installation in non-interactive mode or if SKIP_DOCA is set
    if [ "${SKIP_DOCA:-0}" = "1" ] || [ ! -t 0 ]; then
        log_warn "Skipping DOCA installation (non-interactive mode or SKIP_DOCA=1)"
        log_info "Built-in mlx5 driver is usually sufficient for IBGDA"
        return 1
    fi

    read -p "DOCA OFED not found. Install it? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_warn "Skipping DOCA installation"
        log_info "Built-in mlx5 driver is usually sufficient for IBGDA"
        return 1
    fi

    log_info "Installing DOCA OFED 3.0.0..."

    # Detect Ubuntu version
    UBUNTU_VERSION=$(lsb_release -rs 2>/dev/null || echo "24.04")
    UBUNTU_CODENAME=$(echo $UBUNTU_VERSION | tr -d '.')

    DOCA_URL="https://www.mellanox.com/downloads/DOCA/DOCA_v3.0.0/host/doca-host_3.0.0-058000-25.04-ubuntu${UBUNTU_CODENAME}_amd64.deb"

    cd /tmp
    wget -q "$DOCA_URL" -O doca-host.deb || {
        log_error "Failed to download DOCA package"
        log_info "Try manual download from: https://developer.nvidia.com/networking/doca"
        return 1
    }

    dpkg -i doca-host.deb
    apt-get update -y -qq
    apt-get -y install doca-ofed

    log_ok "DOCA OFED installed successfully"
    ofed_info -s

    rm -f /tmp/doca-host.deb
    DOCA_INSTALLED=true
}

# ============================================================================
# gdrcopy Installation
# ============================================================================
install_gdrcopy() {
    log_section "gdrcopy Installation"

    if [ "$NVSHMEM_USE_GDRCOPY" != "1" ]; then
        log_info "NVSHMEM_USE_GDRCOPY=0, skipping gdrcopy"
        return 0
    fi

    # Check if already installed
    if ls /usr/lib/x86_64-linux-gnu/libgdrapi* >/dev/null 2>&1; then
        log_ok "gdrcopy library already installed"

        # Check kernel module
        if lsmod | grep -q gdrdrv; then
            log_ok "gdrdrv kernel module loaded"
        else
            log_warn "gdrdrv kernel module NOT loaded, attempting to load..."
            modprobe gdrdrv || {
                log_error "Failed to load gdrdrv module"
                log_info "You may need to rebuild gdrcopy for your kernel version"
                log_info "Or reboot the system after DKMS builds the module"
            }
        fi
        return 0
    fi

    log_info "Installing gdrcopy v2.5.1..."

    # Install dependencies
    apt-get update -y -qq
    apt-get install -y -qq --no-install-recommends \
        build-essential devscripts debhelper fakeroot pkg-config dkms git

    # Check for NVIDIA kernel source (needed for nv-p2p.h)
    NVIDIA_SRC_DIR=""
    for nvidia_src in /usr/src/nvidia-*/nvidia; do
        if [ -f "$nvidia_src/nv-p2p.h" ]; then
            NVIDIA_SRC_DIR="$nvidia_src"
            log_ok "Found NVIDIA kernel source at $NVIDIA_SRC_DIR"
            break
        fi
    done

    if [ -z "$NVIDIA_SRC_DIR" ]; then
        log_info "NVIDIA kernel source not found, installing nvidia-kernel-source-580-open..."
        apt-get install -y -qq nvidia-kernel-source-580-open || {
            log_error "Failed to install nvidia-kernel-source-580-open"
            log_info "You may need to manually install the NVIDIA kernel source"
            return 1
        }
        # Find the installed source
        for nvidia_src in /usr/src/nvidia-*/nvidia; do
            if [ -f "$nvidia_src/nv-p2p.h" ]; then
                NVIDIA_SRC_DIR="$nvidia_src"
                log_ok "Found NVIDIA kernel source at $NVIDIA_SRC_DIR"
                break
            fi
        done
    fi

    if [ -z "$NVIDIA_SRC_DIR" ]; then
        log_error "Could not find nv-p2p.h - cannot build gdrdrv module"
        log_info "Continuing with library-only installation..."
    fi

    # Build gdrcopy
    rm -rf /tmp/gdrcopy
    git clone --depth 1 --branch v2.5.1 https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy
    cd /tmp/gdrcopy

    make -j$(nproc)
    make prefix="$GDRCOPY_HOME" CUDA="$CUDA_HOME" install

    # Build and install deb packages (includes DKMS module)
    cd packages
    NVIDIA_SRC_DIR="$NVIDIA_SRC_DIR" CUDA="$CUDA_HOME" ./build-deb-packages.sh
    dpkg -i *.deb

    log_ok "gdrcopy installed successfully"

    # Cleanup
    rm -rf /tmp/gdrcopy

    # Load module
    modprobe gdrdrv || log_warn "gdrdrv module load failed, reboot may be required"
}

# ============================================================================
# NVIDIA PeerMem Configuration
# ============================================================================
configure_nvidia_peermem() {
    log_section "NVIDIA PeerMem Configuration"

    # Configure nvidia module options for SGLang
    NVIDIA_CONF="/etc/modprobe.d/nvidia-graphics-drivers-kms.conf"

    if ! grep -q "PeerMappingOverride=1" "$NVIDIA_CONF" 2>/dev/null; then
        log_info "Configuring PeerMappingOverride for SGLang..."
        echo 'options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"' | \
            tee -a "$NVIDIA_CONF"
        log_ok "NVIDIA module options configured"
        log_warn "Reboot required for changes to take effect"
    else
        log_ok "PeerMappingOverride already configured"
        cat "$NVIDIA_CONF"
    fi

    # Load nvidia_peermem
    if [ "$USE_NVPEERMEM" == "1" ]; then
        log_info "Loading nvidia_peermem module..."
        if lsmod | grep -q nvidia_peermem; then
            log_ok "nvidia_peermem already loaded"
        else
            # Check if module exists
            if modinfo nvidia_peermem &>/dev/null; then
                modprobe nvidia_peermem || {
                    log_warn "Failed to load nvidia_peermem, reboot may be required"
                }
            else
                log_warn "nvidia_peermem module not found in kernel"
                log_info "This is optional for DeepEP - NVSHMEM IBGDA is the key requirement"
                log_info "To build nvidia_peermem, you need nvidia-dkms package (may conflict with preinstalled driver)"
            fi
        fi
    fi
}

# ============================================================================
# NVSHMEM Installation
# ============================================================================
install_nvshmem() {
    log_section "NVSHMEM Installation"

    NVSHMEM_INFO_CMD="${NVSHMEM_HOME}/bin/nvshmem-info"

    # Check if already installed
    if [ -x "$NVSHMEM_INFO_CMD" ]; then
        log_ok "NVSHMEM already installed at $NVSHMEM_HOME"

        # Verify IBGDA support
        if "$NVSHMEM_INFO_CMD" -a | grep -q "NVSHMEM_IBGDA_SUPPORT=ON"; then
            log_ok "NVSHMEM has IBGDA support enabled"
        else
            log_warn "NVSHMEM installed but IBGDA support is OFF"
            log_info "Consider reinstalling with NVSHMEM_IBGDA_SUPPORT=1"
        fi
        return 0
    fi

    log_info "Installing NVSHMEM 3.2.5-1..."

    # Install dependencies (including libibverbs-dev for IBGDA)
    apt-get update -y -qq
    apt-get install -y -qq --no-install-recommends \
        python3-venv python3-pip ninja-build cmake \
        python3-dev build-essential git \
        libibverbs-dev librdmacm-dev

    # Download and extract NVSHMEM
    BUILD_DIR="/tmp/nvshmem_build_src"
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"

    log_info "Downloading NVSHMEM source..."
    wget -q "https://developer.nvidia.com/downloads/assets/secure/nvshmem/nvshmem_src_3.2.5-1.txz" \
        -O "${BUILD_DIR}/nvshmem_src.txz" || {
        log_error "Failed to download NVSHMEM"
        log_info "You may need to download manually from NVIDIA Developer"
        return 1
    }

    tar -xf "${BUILD_DIR}/nvshmem_src.txz" -C "$BUILD_DIR"
    cd "${BUILD_DIR}/nvshmem_src"

    log_info "Building NVSHMEM (this may take a while)..."

    # Configure build
    CUDA_HOME="$CUDA_HOME" GDRCOPY_HOME="$GDRCOPY_HOME" \
    NVSHMEM_SHMEM_SUPPORT=0 NVSHMEM_UCX_SUPPORT=0 NVSHMEM_USE_NCCL=0 \
    NVSHMEM_MPI_SUPPORT=0 NVSHMEM_PMIX_SUPPORT=0 NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
    NVSHMEM_USE_GDRCOPY="$NVSHMEM_USE_GDRCOPY" \
    NVSHMEM_IBGDA_SUPPORT="$NVSHMEM_IBGDA_SUPPORT" \
    cmake -GNinja -S . -B build/ \
        -DCMAKE_INSTALL_PREFIX="$NVSHMEM_HOME" \
        -DCMAKE_CUDA_ARCHITECTURES=100 \
        -DNVSHMEM_BUILD_EXAMPLES=OFF \
        -DNVSHMEM_BUILD_PERFTEST=OFF

    cmake --build build/ --target install

    log_ok "NVSHMEM installed to $NVSHMEM_HOME"

    # Cleanup
    rm -rf "$BUILD_DIR"

    # Verify
    if [ -x "$NVSHMEM_INFO_CMD" ]; then
        log_info "NVSHMEM configuration:"
        "$NVSHMEM_INFO_CMD" -a | grep -E "IBGDA|GDRCOPY"
    fi
}

# ============================================================================
# DeepEP Installation
# ============================================================================
install_deepep() {
    log_section "DeepEP Installation"

    # Check if already installed
    if python3 -c "import deep_ep" &> /dev/null; then
        log_ok "DeepEP already installed"
        python3 -c "import deep_ep; print(f'DeepEP version: {deep_ep.__version__ if hasattr(deep_ep, \"__version__\") else \"unknown\"}')" 2>/dev/null || true
        return 0
    fi

    log_info "Installing DeepEP..."

    # Ensure libcuda.so exists for linking
    if [ ! -f /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
        log_warn "libcuda.so.1 not found, reinstalling libnvidia-compute package..."
        apt-get install -y --reinstall libnvidia-compute-580-server 2>/dev/null || \
        apt-get install -y --reinstall libnvidia-compute-580 2>/dev/null || \
        log_warn "Could not reinstall libnvidia-compute, linking may fail"
    fi

    # Create libcuda.so symlink if missing (needed for compile-time linking)
    if [ ! -f /usr/lib/x86_64-linux-gnu/libcuda.so ]; then
        log_info "Creating libcuda.so symlink..."
        ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so
    fi

    # Install PyTorch if not present
    if ! python3 -c "import torch" &> /dev/null; then
        log_info "Installing PyTorch..."
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129 --break-system-packages
    else
        TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
        log_ok "PyTorch already installed: $TORCH_VERSION"
    fi

    # Clone and build DeepEP
    BUILD_DIR="/tmp/deepep_build"
    rm -rf "$BUILD_DIR"
    git clone https://github.com/deepseek-ai/DeepEP.git "$BUILD_DIR"
    cd "$BUILD_DIR"

    # Set up environment for linking
    export LD_LIBRARY_PATH="${NVSHMEM_HOME}/lib:${GDRCOPY_HOME}/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
    export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${CUDA_HOME}/lib64/stubs:$LIBRARY_PATH"

    log_info "Building DeepEP (CUDA arch: $TORCH_CUDA_ARCH_LIST)..."
    TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" NVSHMEM_DIR="$NVSHMEM_HOME" python3 setup.py build
    TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" NVSHMEM_DIR="$NVSHMEM_HOME" python3 setup.py install

    # Verify installation
    if python3 -c "import deep_ep" &> /dev/null; then
        log_ok "DeepEP installed successfully"
    else
        log_error "DeepEP installation failed"
        return 1
    fi

    # Keep build dir for reference (contains the .so file)
    log_info "Build directory preserved at: $BUILD_DIR"
}

# ============================================================================
# Kernel Module Verification
# ============================================================================
verify_kernel_modules() {
    log_section "Kernel Module Verification"

    REBOOT_NEEDED=false

    # Check nvidia
    if ! lsmod | grep -q nvidia; then
        log_error "nvidia module not loaded"
        REBOOT_NEEDED=true
    else
        log_ok "nvidia module loaded"
    fi

    # Check nvidia_peermem
    if [ "$USE_NVPEERMEM" == "1" ]; then
        if lsmod | grep -q nvidia_peermem; then
            log_ok "nvidia_peermem module loaded"
        else
            log_warn "nvidia_peermem module NOT loaded"
            REBOOT_NEEDED=true
        fi
    fi

    # Check gdrdrv
    if [ "$NVSHMEM_USE_GDRCOPY" == "1" ]; then
        if lsmod | grep -q gdrdrv; then
            log_ok "gdrdrv module loaded"
        else
            log_warn "gdrdrv module NOT loaded"
            REBOOT_NEEDED=true
        fi
    fi

    # Check PeerMappingOverride
    if [ -f /proc/driver/nvidia/params ]; then
        if grep -q "PeerMappingOverride=1" /proc/driver/nvidia/params 2>/dev/null; then
            log_ok "PeerMappingOverride is set"
        else
            log_warn "PeerMappingOverride not set in running kernel"
            REBOOT_NEEDED=true
        fi
    fi

    if [ "$REBOOT_NEEDED" = true ]; then
        echo
        log_warn "================================"
        log_warn "REBOOT REQUIRED"
        log_warn "Some kernel modules are not loaded or configured."
        log_warn "Please reboot and run this script again to verify."
        log_warn "================================"
        return 1
    else
        log_ok "All kernel modules loaded correctly"
        return 0
    fi
}

# ============================================================================
# Main
# ============================================================================
main() {
    log_section "DeepEP Installation Script"
    echo "This script will install DeepEP and its dependencies for SGLang."
    echo "Target: NVIDIA B200 GPUs with IB networking"
    echo
    echo "Configuration:"
    echo "  CUDA_HOME:          $CUDA_HOME"
    echo "  NVSHMEM_HOME:       $NVSHMEM_HOME"
    echo "  GDRCOPY_HOME:       $GDRCOPY_HOME"
    echo "  NVSHMEM_IBGDA:      $NVSHMEM_IBGDA_SUPPORT"
    echo "  NVSHMEM_USE_GDRCOPY: $NVSHMEM_USE_GDRCOPY"
    echo "  TORCH_CUDA_ARCH:    $TORCH_CUDA_ARCH_LIST"
    echo

    # Installation steps
    install_cuda
    install_doca_ofed
    install_gdrcopy
    configure_nvidia_peermem
    install_nvshmem
    install_deepep

    echo
    verify_kernel_modules

    log_section "Installation Complete"

    echo
    echo "To use DeepEP, add to your environment:"
    echo "  export LD_LIBRARY_PATH=${NVSHMEM_HOME}/lib:${GDRCOPY_HOME}/lib:\$LD_LIBRARY_PATH"
    echo
    echo "Test with:"
    echo "  python3 -c 'import deep_ep; print(\"DeepEP OK\")'"
}

# Run main if not sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
