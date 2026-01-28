---
name: deepep-installer
description: This skill should be used when users need to install, configure, or troubleshoot DeepEP (DeepSeek Expert Parallelism) on NVIDIA GPU systems. It covers the complete installation workflow including CUDA, gdrcopy, NVSHMEM with IBGDA support, and DeepEP itself. The skill is particularly useful for B200/H100/A100 GPUs with RoCE/InfiniBand networking, and includes comprehensive debugging capabilities for common installation failures.
---

# DeepEP Installer

## Overview

DeepEP is DeepSeek's Expert Parallelism library for Mixture-of-Experts (MoE) models. It enables high-performance all-to-all communication between GPUs using NVSHMEM with IBGDA (InfiniBand GPUDirect Async) support.

This skill provides:
1. Complete installation workflow for DeepEP and its dependencies
2. Diagnostic capabilities for identifying installation issues
3. Solutions for common problems encountered on different systems
4. Verification and testing procedures

## Prerequisites Assessment

Before installation, assess the system environment:

### Required Components
- **NVIDIA GPU**: B200, H100, A100, or compatible GPU
- **NVIDIA Driver**: Version 550+ (580+ recommended for B200)
- **CUDA Toolkit**: 12.x (12.9 recommended)
- **RDMA Network**: RoCE or InfiniBand with mlx5 driver
- **Python**: 3.10+ with PyTorch

### Pre-flight Diagnostic Commands

```bash
# Check GPU and driver
nvidia-smi
lsmod | grep nvidia

# Check CUDA
nvcc --version
ls -la /usr/local/cuda

# Check RDMA/IB
rdma link                          # Modern systems
ibv_devinfo                        # Fallback
ls /sys/class/infiniband/          # Last resort

# Check kernel modules
lsmod | grep -E "gdrdrv|nvidia_peermem|mlx5"
```

## Installation Workflow

The installation follows this order (dependencies must be installed sequentially):

```
1. CUDA Toolkit → 2. gdrcopy → 3. NVSHMEM → 4. DeepEP
```

### Step 1: CUDA Toolkit

To install CUDA toolkit without driver (when driver is pre-installed):

```bash
# Add NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update

# Install toolkit only (no driver)
apt-get install -y cuda-toolkit-12-9

# Set environment
export CUDA_HOME=/usr/local/cuda-12.9
ln -s /usr/local/cuda-12.9 /usr/local/cuda  # if not exists
```

### Step 2: gdrcopy

gdrcopy provides GPU Direct copy functionality with the gdrdrv kernel module.

```bash
# Install dependencies
apt-get install -y build-essential devscripts debhelper fakeroot pkg-config dkms

# Get NVIDIA kernel source for nv-p2p.h
apt-get install -y nvidia-kernel-source-580-open  # Match driver version

# Find NVIDIA source directory
NVIDIA_SRC_DIR=$(find /usr/src/nvidia-*/nvidia -name "nv-p2p.h" -printf "%h" -quit)

# Build gdrcopy
git clone --depth 1 --branch v2.5.1 https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy
cd /tmp/gdrcopy
make -j$(nproc)
make prefix=/opt/deepep/gdrcopy CUDA=$CUDA_HOME install

# Build and install kernel module
cd packages
NVIDIA_SRC_DIR="$NVIDIA_SRC_DIR" CUDA=$CUDA_HOME ./build-deb-packages.sh
dpkg -i *.deb

# Load module
modprobe gdrdrv
```

### Step 3: NVSHMEM with IBGDA

NVSHMEM requires IBGDA support for optimal performance with DeepEP.

```bash
# Install IB development libraries (critical for IBGDA)
apt-get install -y libibverbs-dev librdmacm-dev

# Download NVSHMEM
wget https://developer.nvidia.com/downloads/assets/secure/nvshmem/nvshmem_src_3.2.5-1.txz
tar -xf nvshmem_src_3.2.5-1.txz

# Configure and build
cd nvshmem_src
CUDA_HOME=$CUDA_HOME \
GDRCOPY_HOME=/opt/deepep/gdrcopy \
NVSHMEM_IBGDA_SUPPORT=1 \
NVSHMEM_USE_GDRCOPY=1 \
NVSHMEM_SHMEM_SUPPORT=0 \
NVSHMEM_UCX_SUPPORT=0 \
NVSHMEM_USE_NCCL=0 \
NVSHMEM_MPI_SUPPORT=0 \
cmake -GNinja -S . -B build/ \
    -DCMAKE_INSTALL_PREFIX=/opt/deepep/nvshmem \
    -DCMAKE_CUDA_ARCHITECTURES=100 \  # 100 for B200, 90 for H100, 80 for A100
    -DNVSHMEM_BUILD_EXAMPLES=OFF

cmake --build build/ --target install

# Verify IBGDA support
/opt/deepep/nvshmem/bin/nvshmem-info -a | grep IBGDA
```

### Step 4: DeepEP

```bash
# Ensure libcuda.so exists
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so

# Install PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# Clone and build DeepEP
git clone https://github.com/deepseek-ai/DeepEP.git /tmp/deepep_build
cd /tmp/deepep_build

export LD_LIBRARY_PATH=/opt/deepep/nvshmem/lib:/opt/deepep/gdrcopy/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$CUDA_HOME/lib64/stubs:$LIBRARY_PATH

TORCH_CUDA_ARCH_LIST="10.0" NVSHMEM_DIR=/opt/deepep/nvshmem python3 setup.py install
# CUDA arch: 10.0 for B200, 9.0 for H100, 8.0 for A100

# Verify
python3 -c "import deep_ep; print('DeepEP imported successfully')"
```

## Troubleshooting Guide

### Problem: CUDA not found

**Symptoms:**
```
nvcc: command not found
CUDA_HOME is not set
```

**Diagnosis:**
```bash
ls /usr/local/cuda*
which nvcc
echo $CUDA_HOME
```

**Solutions:**
1. Install CUDA toolkit: `apt-get install cuda-toolkit-12-9`
2. Set environment: `export CUDA_HOME=/usr/local/cuda-12.9`
3. Create symlink: `ln -s /usr/local/cuda-12.9 /usr/local/cuda`

---

### Problem: gdrcopy compilation fails - nv-p2p.h not found

**Symptoms:**
```
fatal error: nv-p2p.h: No such file or directory
```

**Diagnosis:**
```bash
find /usr/src -name "nv-p2p.h" 2>/dev/null
dpkg -l | grep nvidia-kernel-source
```

**Solutions:**
1. Install NVIDIA kernel source matching driver version:
   ```bash
   apt-get install nvidia-kernel-source-580-open  # For driver 580.x
   ```
2. Set NVIDIA_SRC_DIR before building:
   ```bash
   export NVIDIA_SRC_DIR=/usr/src/nvidia-580.126.09/nvidia
   ```

---

### Problem: NVSHMEM IBGDA build fails - MLX5_lib NOTFOUND

**Symptoms:**
```
CMake Error: Could not find MLX5_lib
NVSHMEM_IBGDA_SUPPORT will be OFF
```

**Diagnosis:**
```bash
dpkg -l | grep -E "libibverbs|librdmacm"
ls /usr/lib/x86_64-linux-gnu/libmlx5*
```

**Solutions:**
1. Install IB development libraries:
   ```bash
   apt-get install libibverbs-dev librdmacm-dev
   ```
2. Verify MLX5 libraries exist:
   ```bash
   ls /usr/lib/x86_64-linux-gnu/libmlx5*
   ```

---

### Problem: DeepEP build fails - libcuda.so not found

**Symptoms:**
```
cannot find -lcuda
ld: cannot find libcuda.so
```

**Diagnosis:**
```bash
ls -la /usr/lib/x86_64-linux-gnu/libcuda*
ldconfig -p | grep cuda
```

**Solutions:**
1. Create symlink:
   ```bash
   ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so
   ```
2. If libcuda.so.1 is missing, reinstall compute library:
   ```bash
   apt-get install --reinstall libnvidia-compute-580-server
   ```
3. Add stubs to LIBRARY_PATH:
   ```bash
   export LIBRARY_PATH=$CUDA_HOME/lib64/stubs:$LIBRARY_PATH
   ```

---

### Problem: nvidia-smi not found after installation

**Symptoms:**
```
nvidia-smi: command not found
```

**Solutions:**
```bash
apt-get install nvidia-utils-580-server  # Match driver version
```

---

### Problem: gdrdrv module not loaded

**Symptoms:**
```
lsmod | grep gdrdrv  # Empty output
```

**Diagnosis:**
```bash
modinfo gdrdrv
dmesg | grep -i gdr
```

**Solutions:**
1. Load module: `modprobe gdrdrv`
2. If module not found, rebuild gdrcopy packages with correct NVIDIA_SRC_DIR
3. May require reboot after DKMS installation

---

### Problem: RDMA devices not detected

**Symptoms:**
```
rdma link  # No output or no ACTIVE devices
No IB devices found
```

**Diagnosis:**
```bash
lsmod | grep mlx5_core
ls /sys/class/infiniband/
ip link | grep -i rdma
```

**Solutions:**
1. Load mlx5 modules:
   ```bash
   modprobe mlx5_core
   modprobe mlx5_ib
   ```
2. Check hardware presence:
   ```bash
   lspci | grep -i mellanox
   ```

---

### Problem: nvidia_peermem module not available

**Symptoms:**
```
modinfo nvidia_peermem  # Module not found
```

**Note:** nvidia_peermem is OPTIONAL for DeepEP. The critical requirement is NVSHMEM with IBGDA support.

**If needed:**
1. Requires nvidia-dkms package (may conflict with pre-installed drivers)
2. Often unavailable on cloud VMs with pre-installed drivers
3. DeepEP works without it using NVSHMEM IBGDA

## Verification and Testing

### Module Verification

```bash
# Required modules
lsmod | grep gdrdrv         # Must be loaded
lsmod | grep mlx5_core      # For RDMA

# NVSHMEM IBGDA check
/opt/deepep/nvshmem/bin/nvshmem-info -a | grep "NVSHMEM_IBGDA_SUPPORT=ON"
```

### DeepEP Intranode Test

```bash
export LD_LIBRARY_PATH=/opt/deepep/nvshmem/lib:/opt/deepep/gdrcopy/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

cd /tmp/deepep_build

# Quick test with 2 GPUs
python3 tests/test_intranode.py --num-processes 2

# Full test with all GPUs
python3 tests/test_intranode.py --num-processes 8
```

### Expected Test Output

```
[testing] Running with BF16, without top-k (async=False, previous=False) ... passed
[testing] Running with BF16, without top-k (async=True, previous=False) ... passed
...
[tuning] Best dispatch (BF16): SMs 24, NVL chunk 24, 35.55 GB/s (NVL)
[tuning] Best combine: SMs 24, NVL chunk 16: 36.64 GB/s (NVL)
```

## Environment Variables Reference

```bash
# Required
export CUDA_HOME=/usr/local/cuda
export NVSHMEM_HOME=/opt/deepep/nvshmem
export GDRCOPY_HOME=/opt/deepep/gdrcopy

# Runtime
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$GDRCOPY_HOME/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Build-time
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$CUDA_HOME/lib64/stubs:$LIBRARY_PATH

# NVSHMEM options
export NVSHMEM_IBGDA_SUPPORT=1
export NVSHMEM_USE_GDRCOPY=1

# CUDA architecture (set during build)
# B200: TORCH_CUDA_ARCH_LIST="10.0"
# H100: TORCH_CUDA_ARCH_LIST="9.0"
# A100: TORCH_CUDA_ARCH_LIST="8.0"
```

## Resources

### scripts/

- `install-deepep.sh` - Complete automated installation script

### references/

- `troubleshooting.md` - Extended troubleshooting guide with more edge cases
