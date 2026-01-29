#!/bin/bash
# DeepEP Installation Script for SGLang
# Adapted from deepep-installer.yaml for bare-metal/VM environments
# Now includes automatic driver installation if not present
# Reference: /home/chrisya/gpu-tpu-pedia/gpu/deepep/deepep-installer.yaml

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

# Update apt
apt-get update -y -qq

# ============================================================================
# DOCA OFED Installation (FIRST - before GPU driver per YAML order)
# ============================================================================
install_doca_ofed() {
    log_section "DOCA OFED Installation"

    # Check if already installed
    if command -v ofed_info &> /dev/null; then
        log_ok "DOCA OFED already installed"
        ofed_info -s
        return 0
    fi

    # Skip if explicitly requested
    if [ "${SKIP_DOCA:-0}" = "1" ]; then
        log_warn "Skipping DOCA installation (SKIP_DOCA=1)"
        log_info "Built-in mlx5 driver is usually sufficient for IBGDA"
        return 1
    fi

    log_info "DOCA OFED not found, installing now..."

    # Detect Ubuntu version (default to 24.04)
    UBUNTU_VERSION=$(lsb_release -rs 2>/dev/null || echo "24.04")
    UBUNTU_CODENAME=$(echo $UBUNTU_VERSION | tr -d '.')

    cd /tmp
    log_info "Downloading DOCA package..."
    wget -q "https://www.mellanox.com/downloads/DOCA/DOCA_v3.0.0/host/doca-host_3.0.0-058000-25.04-ubuntu${UBUNTU_CODENAME}_amd64.deb" -O doca-host.deb || {
        log_error "Failed to download DOCA package"
        log_info "Continuing with built-in mlx5 driver..."
        return 1
    }

    log_info "Installing DOCA package..."
    dpkg -i doca-host.deb || true
    apt-get update -y -qq
    apt-get -y install doca-ofed || {
        log_warn "DOCA OFED installation had errors (common with newer kernels)"
        log_info "Built-in mlx5 driver should still work for IBGDA"
    }

    if command -v ofed_info &> /dev/null; then
        log_ok "DOCA OFED installed successfully"
        ofed_info -s
    fi

    rm -f /tmp/doca-host.deb
}

# ============================================================================
# GPU Driver Installation (AFTER DOCA per YAML order)
# ============================================================================
install_nvidia_driver() {
    log_section "NVIDIA GPU Driver Installation"

    if lsmod | grep -q nvidia; then
        log_ok "NVIDIA driver already installed"
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
        return 0
    fi

    log_info "GPU driver module not found, installing now..."

    # Add NVIDIA CUDA repository
    if [ ! -f /etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list ]; then
        log_info "Adding NVIDIA repository..."
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
        dpkg -i /tmp/cuda-keyring.deb
        rm -f /tmp/cuda-keyring.deb
        apt-get update -y -qq
    fi

    # Install nvidia-open driver (575 is stable for B200/H100)
    log_info "Installing nvidia-open-575 driver..."
    apt-get install -y nvidia-open-575

    # Install CUDA toolkit
    log_info "Installing CUDA toolkit 12.9..."
    apt-get install -y cuda-toolkit-12-9

    # Set CUDA_HOME
    export CUDA_HOME=/usr/local/cuda-12.9

    # Create symlink if not exists
    if [ ! -e /usr/local/cuda ]; then
        ln -sf /usr/local/cuda-12.9 /usr/local/cuda
    fi

    # Verify
    if lsmod | grep -q nvidia; then
        log_ok "NVIDIA driver installed successfully"
        nvidia-smi
    else
        log_warn "NVIDIA driver installed but module not loaded"
        log_info "A reboot will be required"
    fi
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

    # Check if already installed (check for libgdr* per YAML)
    if ls /usr/lib/x86_64-linux-gnu/libgdr* >/dev/null 2>&1; then
        log_ok "gdrcopy library already installed"

        # Check kernel module
        if lsmod | grep -q gdrdrv; then
            log_ok "gdrdrv kernel module loaded"
        else
            log_warn "gdrdrv kernel module NOT loaded"
            modprobe gdrdrv 2>/dev/null || log_info "Will load after reboot"
        fi
        return 0
    fi

    log_info "gdrcopy library not found, installing now..."

    # Install dependencies (per YAML)
    apt-get install -y -qq --no-install-recommends \
        build-essential devscripts debhelper fakeroot pkg-config dkms git

    # Build gdrcopy (per YAML flow)
    rm -rf /tmp/gdrcopy
    git clone https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy
    cd /tmp/gdrcopy
    git checkout tags/v2.5.1

    make -j$(nproc)
    make prefix="$GDRCOPY_HOME" CUDA=/usr/local/cuda install

    # Build and install deb packages (per YAML)
    cd packages
    CUDA=/usr/local/cuda ./build-deb-packages.sh || log_warn "deb package build had issues"
    dpkg -i *.deb || log_warn "deb package install had issues"

    log_ok "gdrcopy installation complete"

    # Cleanup
    cd /tmp
    rm -rf /tmp/gdrcopy
}

# ============================================================================
# PeerMappingOverride Configuration (per YAML)
# ============================================================================
configure_peermapping() {
    log_section "PeerMappingOverride Configuration"

    NVIDIA_CONF="/etc/modprobe.d/nvidia-graphics-drivers-kms.conf"

    # SGLang needs this (per YAML comment)
    if ! grep -q "PeerMappingOverride=1" "$NVIDIA_CONF" 2>/dev/null; then
        log_info "Configuring PeerMappingOverride for SGLang..."
        echo 'options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"' | \
            tee -a "$NVIDIA_CONF"
        log_ok "NVIDIA module options configured"
    else
        log_ok "PeerMappingOverride already configured"
        cat "$NVIDIA_CONF"
    fi
}

# ============================================================================
# Load nvidia_peermem (per YAML)
# ============================================================================
load_nvidia_peermem() {
    log_section "NVIDIA PeerMem Loading"

    if [ "$USE_NVPEERMEM" != "1" ]; then
        log_info "USE_NVPEERMEM=0, skipping nvidia_peermem"
        return 0
    fi

    log_info "USE_NVPEERMEM=1, loading nvidia_peermem..."

    if lsmod | grep -q nvidia_peermem; then
        log_ok "nvidia_peermem loaded successfully"
    else
        modprobe nvidia_peermem || log_warn "Failed to load nvidia_peermem module, continuing anyway..."

        if lsmod | grep -q nvidia_peermem; then
            log_ok "nvidia_peermem loaded successfully"
        else
            log_warn "nvidia_peermem module not loaded, but continuing installation..."
        fi
    fi
}

# ============================================================================
# Check if Reboot is Needed (per YAML logic lines 156-178)
# ============================================================================
check_reboot_needed() {
    log_section "Reboot Check"

    REBOOT_NEEDED=false

    # Check 1: nvidia module not loaded
    if ! lsmod | grep -q nvidia; then
        REBOOT_NEEDED=true
        log_warn "nvidia module not loaded - reboot required"
    else
        log_ok "nvidia module loaded"
    fi

    # Check 2: nvidia_peermem not loaded (when USE_NVPEERMEM=1)
    if [ "$USE_NVPEERMEM" == "1" ] && ! lsmod | grep -q nvidia_peermem; then
        REBOOT_NEEDED=true
        log_warn "nvidia_peermem module not loaded - reboot required"
    elif [ "$USE_NVPEERMEM" == "1" ]; then
        log_ok "nvidia_peermem module loaded"
    fi

    # Check 3: PeerMappingOverride not set (only when GDRCOPY not used)
    if [ "$NVSHMEM_USE_GDRCOPY" != "1" ] && ! grep -q "PeerMappingOverride=1" /proc/driver/nvidia/params 2>/dev/null; then
        REBOOT_NEEDED=true
        log_warn "PeerMappingOverride not set - reboot required"
    fi

    # Check 4: gdrdrv not loaded (when GDRCOPY used)
    if [ "$NVSHMEM_USE_GDRCOPY" == "1" ] && ! lsmod | grep -q gdrdrv; then
        REBOOT_NEEDED=true
        log_warn "gdrdrv module not loaded - reboot required"
    elif [ "$NVSHMEM_USE_GDRCOPY" == "1" ]; then
        log_ok "gdrdrv module loaded"
    fi

    if [ "$REBOOT_NEEDED" == "true" ]; then
        log_warn "================================"
        log_warn "REBOOT REQUIRED"
        log_warn "Some kernel modules are not loaded."
        log_warn "Please reboot and run this script again."
        log_warn "================================"

        # Update initramfs before suggesting reboot
        update-initramfs -u 2>/dev/null || true

        if [ "${AUTO_REBOOT:-0}" == "1" ]; then
            log_info "AUTO_REBOOT=1, rebooting now..."
            reboot
        fi
        return 1
    else
        log_ok "All required modules are loaded correctly"
        return 0
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

    log_info "NVSHMEM not found, installing now..."

    # Install dependencies (per YAML)
    apt-get install -y -qq --no-install-recommends \
        python3-venv python3-pip ninja-build cmake \
        python3.12-dev python3.12 \
        build-essential devscripts debhelper dkms git \
        libibverbs-dev librdmacm-dev

    # Build nvshmem (per YAML)
    BUILD_DIR="/tmp/nvshmem_build_src"
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"

    log_info "Downloading NVSHMEM source..."
    wget -q "https://developer.nvidia.com/downloads/assets/secure/nvshmem/nvshmem_src_3.2.5-1.txz" \
        -O "${BUILD_DIR}/nvshmem_src_cuda12-all.tar.gz" || {
        log_error "Failed to download NVSHMEM"
        return 1
    }

    tar -xf "${BUILD_DIR}/nvshmem_src_cuda12-all.tar.gz" -C "$BUILD_DIR"
    cd "${BUILD_DIR}/nvshmem_src"

    log_info "Building NVSHMEM (this may take a while)..."

    # Configure and build (per YAML)
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

    log_ok "NVSHMEM installation complete"

    # Cleanup
    cd /tmp
    rm -rf "$BUILD_DIR"

    # Verify IBGDA support (per YAML)
    log_info "Verifying NVSHMEM configuration..."
    if [ -x "$NVSHMEM_INFO_CMD" ]; then
        if "$NVSHMEM_INFO_CMD" -a | grep -q "NVSHMEM_IBGDA_SUPPORT=ON"; then
            log_ok "NVSHMEM_IBGDA_SUPPORT is enabled correctly"
        else
            log_error "NVSHMEM_IBGDA_SUPPORT is not enabled, exiting..."
            exit 1
        fi
    else
        log_error "nvshmem-info command not found after installation, exiting..."
        exit 1
    fi
}

# ============================================================================
# DeepEP Installation
# ============================================================================
install_deepep() {
    log_section "DeepEP Installation"

    # Check if already installed (per YAML)
    if python3 -c "import deep_ep" &> /dev/null; then
        log_ok "DeepEP already installed"
        return 0
    fi

    log_info "DeepEP not found, installing now..."

    # Install PyTorch (per YAML)
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129 --break-system-packages

    # Clone and build DeepEP (per YAML)
    BUILD_DIR="/tmp/deepep_build"
    rm -rf "$BUILD_DIR"
    git clone https://github.com/deepseek-ai/DeepEP.git "$BUILD_DIR"
    cd "$BUILD_DIR"

    # Set up environment (per YAML)
    export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/opt/deepep/nvshmem/lib:/opt/deepep/gdrcopy/lib:$LD_LIBRARY_PATH

    log_info "Building DeepEP (CUDA arch: $TORCH_CUDA_ARCH_LIST)..."
    TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" NVSHMEM_DIR=/opt/deepep/nvshmem python3 setup.py build

    # Create symlink (per YAML)
    ln -sf build/lib.linux-x86_64-cpython-312/deep_ep_cpp.cpython-312-x86_64-linux-gnu.so . 2>/dev/null || true

    TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" NVSHMEM_DIR=/opt/deepep/nvshmem python3 setup.py install

    # Verify installation (per YAML)
    if python3 -c "import deep_ep" &> /dev/null; then
        log_ok "DeepEP installation complete"
    else
        log_error "Failed to install DeepEP, exiting..."
        exit 1
    fi
}

# ============================================================================
# Create Environment Script
# ============================================================================
create_env_script() {
    log_section "Creating Environment Script"

    mkdir -p /opt/deepep

    cat > /opt/deepep/env.sh << 'EOF'
#!/bin/bash
# DeepEP Environment Configuration
# Usage: source /opt/deepep/env.sh

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export GDRCOPY_HOME=/opt/deepep/gdrcopy
export NVSHMEM_HOME=/opt/deepep/nvshmem
export NVSHMEM_DIR=/opt/deepep/nvshmem

# Collect nvidia pip package lib paths
NVIDIA_LIB_PATHS=""
for d in /usr/local/lib/python3.*/dist-packages/nvidia/*/lib; do
    [ -d "$d" ] && NVIDIA_LIB_PATHS="${d}:${NVIDIA_LIB_PATHS}"
done
for d in $HOME/.local/lib/python3.*/site-packages/nvidia/*/lib; do
    [ -d "$d" ] && NVIDIA_LIB_PATHS="${d}:${NVIDIA_LIB_PATHS}"
done

export LD_LIBRARY_PATH=${NVSHMEM_HOME}/lib:${GDRCOPY_HOME}/lib:${CUDA_HOME}/lib64:${NVIDIA_LIB_PATHS}${LD_LIBRARY_PATH}

# HuggingFace cache on LSSD (if available)
[ -d /lssd/huggingface ] && export HF_HOME=/lssd/huggingface

echo "DeepEP environment loaded"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  NVSHMEM_HOME: $NVSHMEM_HOME"
echo "  GDRCOPY_HOME: $GDRCOPY_HOME"
EOF

    chmod +x /opt/deepep/env.sh
    log_ok "Environment script created at /opt/deepep/env.sh"

    # Add to bashrc if not already there
    if ! grep -q "source /opt/deepep/env.sh" ~/.bashrc 2>/dev/null; then
        echo 'source /opt/deepep/env.sh' >> ~/.bashrc
        log_info "Added to ~/.bashrc"
    fi
}

# ============================================================================
# Main (per YAML order)
# ============================================================================
main() {
    log_section "DeepEP Installation Script"
    echo "This script will install DeepEP and its dependencies for SGLang."
    echo "Target: NVIDIA B200 GPUs with IB networking"
    echo "Reference: deepep-installer.yaml"
    echo
    echo "Configuration:"
    echo "  CUDA_HOME:           $CUDA_HOME"
    echo "  NVSHMEM_HOME:        $NVSHMEM_HOME"
    echo "  GDRCOPY_HOME:        $GDRCOPY_HOME"
    echo "  NVSHMEM_IBGDA:       $NVSHMEM_IBGDA_SUPPORT"
    echo "  NVSHMEM_USE_GDRCOPY: $NVSHMEM_USE_GDRCOPY"
    echo "  USE_NVPEERMEM:       $USE_NVPEERMEM"
    echo "  TORCH_CUDA_ARCH:     $TORCH_CUDA_ARCH_LIST"
    echo

    # Installation steps (per YAML order)
    # Step 1: Install DOCA OFED
    install_doca_ofed

    # Step 2: Install GPU driver and CUDA
    install_nvidia_driver

    # Step 3: Build gdrcopy
    install_gdrcopy

    # Step 4: Configure PeerMappingOverride (SGLang needs this)
    configure_peermapping

    # Step 5: Load nvidia_peermem
    load_nvidia_peermem

    # Step 6: Check if reboot is needed (per YAML logic)
    if ! check_reboot_needed; then
        log_warn "Please reboot and run this script again to continue with NVSHMEM and DeepEP installation"
        exit 0
    fi

    # Step 7: Build NVSHMEM
    install_nvshmem

    # Step 8: Build DeepEP
    install_deepep

    # Step 9: Create environment script
    create_env_script

    log_section "Installation Complete"

    echo
    echo "To use DeepEP, run:"
    echo "  source /opt/deepep/env.sh"
    echo
    echo "Test with:"
    echo "  python3 -c 'import deep_ep; print(\"DeepEP OK\")'"
    echo
    echo "Run intranode test:"
    echo "  cd /tmp/deepep_build"
    echo "  python3 tests/test_intranode.py --num-processes 2"
}

# Run main if not sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
