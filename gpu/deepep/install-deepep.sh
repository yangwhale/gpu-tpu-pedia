#!/bin/bash
# DeepEP Installation Script (No GDRCopy - GPU NIC Handler Mode)
# Optimized for NVSHMEM_IBGDA_NIC_HANDLER=gpu which doesn't require GDRCopy
#
# Usage: sudo bash install-deepep-no-gdrcopy.sh
#
# Phase 1: CUDA + DOCA + PeerMappingOverride → Reboot
# Phase 2: NVSHMEM (no GDRCopy) + PyTorch + DeepEP

set -ex

export DEBIAN_FRONTEND=noninteractive

# =============================================================================
# Version Configuration
# =============================================================================
CUDA_VERSION="12.9"
DOCA_OFED_VERSION="3.2.1"
NVSHMEM_VERSION="v3.5.19-1"
CUDA_ARCH="100"  # sm_100 for B200/Blackwell

# =============================================================================
# Path Configuration
# =============================================================================
INSTALL_PREFIX="/opt/deepep"
NVSHMEM_HOME="${INSTALL_PREFIX}/nvshmem"
DEEPEP_HOME="${INSTALL_PREFIX}/source"
CUDA_HOME="/usr/local/cuda"
PHASE1_MARKER="${INSTALL_PREFIX}/.phase1_complete"

# =============================================================================
# Phase Detection
# =============================================================================
mkdir -p ${INSTALL_PREFIX}

if [ -f "${PHASE1_MARKER}" ]; then
    echo "=========================================="
    echo "DeepEP Installation - Phase 2 (Post-Reboot)"
    echo "=========================================="
    CURRENT_PHASE=2
else
    echo "=========================================="
    echo "DeepEP Installation - Phase 1 (Pre-Reboot)"
    echo "=========================================="
    echo "Versions: CUDA ${CUDA_VERSION}, DOCA ${DOCA_OFED_VERSION}, NVSHMEM ${NVSHMEM_VERSION}"
    echo "Mode: GPU NIC Handler (no GDRCopy)"
    echo "=========================================="
    CURRENT_PHASE=1
fi

# =============================================================================
# Phase 1: CUDA + DOCA + PeerMappingOverride (requires reboot)
# =============================================================================
if [ "$CURRENT_PHASE" -eq 1 ]; then

    # === 1. CUDA Toolkit ===
    echo "[1/3] Installing CUDA Toolkit ${CUDA_VERSION}..."
    CUDA_PKG_VERSION=$(echo ${CUDA_VERSION} | tr '.' '-')
    if ! command -v nvcc &>/dev/null; then
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
        dpkg -i cuda-keyring_1.1-1_all.deb && rm -f cuda-keyring_1.1-1_all.deb
        apt-get update -qq
        apt-get install -y cuda-toolkit-${CUDA_PKG_VERSION}
        ln -sf /usr/local/cuda-${CUDA_VERSION} /usr/local/cuda
    else
        echo "CUDA already installed: $(nvcc --version | grep release)"
    fi
    export PATH=${CUDA_HOME}/bin:$PATH

    # === 2. DOCA-OFED ===
    echo "[2/3] Installing DOCA-OFED ${DOCA_OFED_VERSION}..."
    if ! dpkg -l doca-ofed 2>/dev/null | grep -q '^ii'; then
        curl -fsSL https://linux.mellanox.com/public/repo/doca/GPG-KEY-Mellanox.pub | \
            gpg --dearmor -o /usr/share/keyrings/doca.gpg
        echo "deb [signed-by=/usr/share/keyrings/doca.gpg] https://linux.mellanox.com/public/repo/doca/${DOCA_OFED_VERSION}/ubuntu24.04/x86_64/ ./" | \
            tee /etc/apt/sources.list.d/doca.list
        apt-get update -qq
        apt-get install -y doca-ofed
    else
        echo "DOCA-OFED already installed"
    fi

    # === 3. PeerMappingOverride ===
    echo "[3/3] Configuring PeerMappingOverride..."
    if ! grep -q "PeerMappingOverride=1" /etc/modprobe.d/nvidia-graphics-drivers-kms.conf 2>/dev/null; then
        echo 'options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"' | \
            tee -a /etc/modprobe.d/nvidia-graphics-drivers-kms.conf
    else
        echo "PeerMappingOverride already configured"
    fi

    touch "${PHASE1_MARKER}"

    echo ""
    echo "=========================================="
    echo "Phase 1 Complete - Reboot Required!"
    echo "=========================================="
    echo "After reboot, run this script again:"
    echo "  sudo bash $0"
    echo "=========================================="
    exit 0
fi

# =============================================================================
# Phase 2: NVSHMEM + PyTorch + DeepEP (post-reboot, NO GDRCopy)
# =============================================================================
export PATH=${CUDA_HOME}/bin:$PATH

# === 1. Load nvidia_peermem ===
echo "[1/5] Loading nvidia_peermem module..."
modprobe nvidia_peermem || echo "Warning: nvidia_peermem module not loaded"

# === 2. NVSHMEM (WITHOUT GDRCopy) ===
echo "[2/5] Installing NVSHMEM ${NVSHMEM_VERSION} (IBGDA only, no GDRCopy)..."
if [ ! -x "${NVSHMEM_HOME}/bin/nvshmem-info" ]; then
    apt-get install -y -qq cmake ninja-build python3-venv python3-pip \
        python3.12-dev python3.12 build-essential git

    BUILD_DIR="/tmp/nvshmem_build_src"
    rm -rf "$BUILD_DIR"

    git clone --depth 1 --branch ${NVSHMEM_VERSION} https://github.com/NVIDIA/nvshmem.git "$BUILD_DIR"
    cd "$BUILD_DIR"

    # Build NVSHMEM with IBGDA but WITHOUT GDRCopy
    # GPU NIC handler mode doesn't need GDRCopy
    CUDA_HOME=${CUDA_HOME} \
    NVSHMEM_IBGDA_SUPPORT=1 \
    NVSHMEM_USE_GDRCOPY=0 \
    NVSHMEM_MPI_SUPPORT=0 \
    NVSHMEM_SHMEM_SUPPORT=0 \
    NVSHMEM_UCX_SUPPORT=0 \
    NVSHMEM_USE_NCCL=0 \
    NVSHMEM_PMIX_SUPPORT=0 \
    NVSHMEM_TIMEOUT_DEVICE_POLLING=0 \
    cmake -GNinja -S . -B build \
        -DCMAKE_INSTALL_PREFIX=${NVSHMEM_HOME} \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DNVSHMEM_BUILD_EXAMPLES=OFF \
        -DNVSHMEM_BUILD_TESTS=OFF \
        -DNVSHMEM_BUILD_PERFTEST=ON

    cmake --build build --target install

    echo "${NVSHMEM_HOME}/lib" > /etc/ld.so.conf.d/nvshmem.conf
    ldconfig

    # Verify IBGDA support
    if ${NVSHMEM_HOME}/bin/nvshmem-info -a | grep -q "NVSHMEM_IBGDA_SUPPORT=ON"; then
        echo "NVSHMEM IBGDA support enabled"
    else
        echo "ERROR: NVSHMEM_IBGDA_SUPPORT not enabled"
        exit 1
    fi

    rm -rf "$BUILD_DIR"
else
    echo "NVSHMEM already installed"
fi

cd ~

# === 3. PyTorch ===
echo "[3/5] Installing PyTorch..."
PYTORCH_CUDA_SUFFIX="cu$(echo ${CUDA_VERSION} | tr -d '.')"
if ! python3 -c "import torch" 2>/dev/null; then
    apt-get install -y -qq python3-pip
    pip3 install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/${PYTORCH_CUDA_SUFFIX} \
        --break-system-packages
else
    echo "PyTorch already installed: $(python3 -c 'import torch; print(torch.__version__)')"
fi

# === 4. DeepEP ===
echo "[4/5] Installing DeepEP (HEAD + PR #466 GPU-NIC mapping)..."
if ! python3 -c "import deep_ep" 2>/dev/null; then
    rm -rf ${DEEPEP_HOME}
    git clone https://github.com/deepseek-ai/DeepEP.git ${DEEPEP_HOME}
    cd ${DEEPEP_HOME}

    # Apply PR #466: GPU-to-NIC mapping for GCP A3 Ultra/A4/B200
    # This fixes PCIe topology issues where 2 GPUs share 2 NICs on same PCIe switch
    echo "Applying PR #466 (GPU-to-NIC mapping)..."
    python3 << 'PATCH_EOF'
import re

buffer_py = "deep_ep/buffer.py"
with open(buffer_py, "r") as f:
    content = f.read()

# Check if already patched
if "_setup_device_hca_mapping" in content:
    print("PR #466 already applied, skipping...")
else:
    # Add the new method before the last class method or at end of class
    new_method = '''
    def _setup_device_hca_mapping(self):
        """
        Set up device to NIC mapping using DEEP_EP_DEVICE_TO_HCA_MAPPING environment variable.
        The mapping format is: "0:mlx5_0,1:mlx5_1,..." where each entry maps a CUDA device ID
        to an HCA name separated by colon.
        """
        if "DEEP_EP_DEVICE_TO_HCA_MAPPING" in os.environ:
            device_mapping = {}
            mapping_str = os.environ["DEEP_EP_DEVICE_TO_HCA_MAPPING"]
            for mapping in mapping_str.split(","):
                assert ":" in mapping, f"Invalid mapping format: {mapping}, expected format: 'device_id:hca_name'"
                parts = mapping.split(":", 1)
                device_id = int(parts[0])
                hca_name = parts[1]
                device_mapping[device_id] = hca_name

            current_device = torch.cuda.current_device()
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
                current_device = int(visible_devices[current_device])

            os.environ["NVSHMEM_ENABLE_NIC_PE_MAPPING"] = "1"
            os.environ["NVSHMEM_HCA_LIST"] = device_mapping[current_device]

'''

    # Find the __init__ method and add call to _setup_device_hca_mapping
    # Look for the pattern where RDMA is being initialized
    init_pattern = r'(if self\.low_latency_mode:.*?nvshmem_init_by_uniqueid\([^)]+\))'

    def add_mapping_call(match):
        original = match.group(1)
        return f"self._setup_device_hca_mapping()\\n        {original}"

    content_with_call = re.sub(init_pattern, add_mapping_call, content, flags=re.DOTALL)

    # Add the new method before "def get_" methods
    insert_pattern = r'(\n    def get_dispatch_config)'
    content_final = re.sub(insert_pattern, new_method + r'\1', content_with_call)

    with open(buffer_py, "w") as f:
        f.write(content_final)

    print("PR #466 applied successfully")
PATCH_EOF

    export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${NVSHMEM_HOME}/lib:${LD_LIBRARY_PATH:-}
    TORCH_CUDA_ARCH_LIST="10.0" NVSHMEM_DIR=${NVSHMEM_HOME} python3 setup.py build
    ln -sf build/lib.linux-x86_64-cpython-312/deep_ep_cpp.cpython-312-x86_64-linux-gnu.so .
    TORCH_CUDA_ARCH_LIST="10.0" NVSHMEM_DIR=${NVSHMEM_HOME} python3 setup.py install

    if ! python3 -c "import deep_ep" 2>/dev/null; then
        echo "ERROR: DeepEP installation failed"
        exit 1
    fi

    # Copy module for non-root users
    mkdir -p ${INSTALL_PREFIX}/python
    cp -r build/lib.linux-x86_64-cpython-312/* ${INSTALL_PREFIX}/python/

    # Also copy deep_ep Python source (includes PR #466 patch)
    cp -r deep_ep ${INSTALL_PREFIX}/python/

    # Verify PR #466 patch is in installed location
    if grep -q "_setup_device_hca_mapping" ${INSTALL_PREFIX}/python/deep_ep/buffer.py; then
        echo "PR #466 verified in installed module"
    else
        echo "ERROR: PR #466 not found in installed module"
        exit 1
    fi
else
    echo "DeepEP already installed"
fi

# === 5. Environment Script (No GDRCopy) ===
echo "[5/5] Creating environment script..."

# Auto-detect HCA prefix
HCA_DEVICE=$(ls /sys/class/infiniband/ 2>/dev/null | head -1)
if [[ $HCA_DEVICE == mlx5* ]]; then
    HCA_PREFIX="mlx5"
elif [[ $HCA_DEVICE == rocep* ]]; then
    HCA_PREFIX="rocep"
else
    HCA_PREFIX="mlx5"
fi
echo "Detected HCA: $HCA_DEVICE -> $HCA_PREFIX"

cat > ${INSTALL_PREFIX}/unified-env.sh << EOF
#!/bin/bash
# DeepEP Environment (GPU NIC Handler Mode - No GDRCopy)

export NVSHMEM_HOME=/opt/deepep/nvshmem
export CUDA_HOME=/usr/local/cuda

# Library paths (no GDRCopy)
export LD_LIBRARY_PATH=\${NVSHMEM_HOME}/lib:\${CUDA_HOME}/lib64:\${LD_LIBRARY_PATH:-}

# Force our NVSHMEM over PyTorch bundled version
export LD_PRELOAD="\${NVSHMEM_HOME}/lib/libnvshmem_host.so.3"

# NVSHMEM IBGDA Configuration
export NVSHMEM_REMOTE_TRANSPORT=ibgda
export NVSHMEM_IB_ENABLE_IBGDA=1

# GPU NIC Handler (no GDRCopy needed)
export NVSHMEM_IBGDA_NIC_HANDLER=gpu

export NVSHMEM_HCA_PREFIX=${HCA_PREFIX}
export NVSHMEM_IB_GID_INDEX=3
export NVSHMEM_DISABLE_CUDA_VMM=1
export NVSHMEM_ENABLE_NIC_PE_MAPPING=1

# Performance settings
export NVSHMEM_IBGDA_NUM_RC_PER_PE=8
export NVSHMEM_IBGDA_NUM_DCI=4

# PR #466: GPU to NIC static mapping for GCP B200 (8 GPU : 8 NIC)
# This fixes PCIe topology issues where 2 GPUs may share 2 NICs on same PCIe switch
# Format: <CUDA_DEVICE_ID>:<HCA_NAME>,...
export DEEP_EP_DEVICE_TO_HCA_MAPPING=0:mlx5_0,1:mlx5_1,2:mlx5_2,3:mlx5_3,4:mlx5_4,5:mlx5_5,6:mlx5_6,7:mlx5_7

export PYTHONPATH=/opt/deepep/python:\${PYTHONPATH:-}

echo "DeepEP ready (HCA=${HCA_PREFIX}, NIC_HANDLER=gpu, no GDRCopy)"
EOF
chmod +x ${INSTALL_PREFIX}/unified-env.sh

rm -f "${PHASE1_MARKER}"

echo ""
echo "=========================================="
echo "Installation Complete! (No GDRCopy)"
echo "=========================================="
echo "Verify: source /opt/deepep/unified-env.sh && python3 -c \"import deep_ep\""
echo "=========================================="
