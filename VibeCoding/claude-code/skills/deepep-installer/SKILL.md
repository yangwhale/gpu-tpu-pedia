---
name: deepep-installer
description: This skill should be used when users need to install, configure, or troubleshoot DeepEP (DeepSeek Expert Parallelism) on NVIDIA GPU systems. It covers the complete installation workflow including CUDA, NVSHMEM with IBGDA support, and DeepEP itself. The skill is particularly useful for B200/H100/A100 GPUs with RoCE/InfiniBand networking, and includes comprehensive debugging capabilities for common installation failures.
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
- **CUDA Toolkit**: 13.0 (recommended for B200)
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

# Check Python and pip (IMPORTANT: Ubuntu 24.04 doesn't include pip by default)
python3 --version
python3 -m pip --version || echo "pip not installed - run: sudo apt install python3-pip"

# Check RDMA/IB
rdma link                          # Modern systems
ibv_devinfo                        # Fallback
ls /sys/class/infiniband/          # Last resort

# Check kernel modules
lsmod | grep -E "nvidia_peermem|mlx5"
```

### Important Notes

1. **Ubuntu 24.04 pip**: Ubuntu 24.04 doesn't include pip by default. Install it first:
   ```bash
   sudo apt-get install -y python3-pip
   ```

2. **DOCA OFED Kernel Compatibility**: DOCA OFED 3.0.0 may not compile with newer kernels (e.g., kernel 6.14.0 on GCP). The built-in mlx5 driver is usually sufficient - DOCA is optional.

## Installation Workflow

The installation follows this order (dependencies must be installed sequentially):

```
1. CUDA Toolkit → 2. PeerMappingOverride → 3. NVSHMEM → 4. DeepEP
```

**Note:** gdrcopy is **optional** and disabled by default. We use `PeerMappingOverride=1` instead, which is simpler and sufficient for most use cases.

### Step 1: CUDA Toolkit

To install CUDA toolkit without driver (when driver is pre-installed):

```bash
# Add NVIDIA repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update

# Install toolkit only (no driver)
apt-get install -y cuda-toolkit-13-0

# Set environment
export CUDA_HOME=/usr/local/cuda-13.0
ln -s /usr/local/cuda-13.0 /usr/local/cuda  # if not exists
```

### Step 2: Configure PeerMappingOverride

PeerMappingOverride enables GPU-to-GPU P2P memory access, required for NVSHMEM IBGDA.

```bash
# Create nvidia module options file
cat > /etc/modprobe.d/nvidia-peermem.conf << 'EOF'
options nvidia NVreg_PeerMappingOverride=1
EOF

# Reload nvidia module (or reboot)
# Note: This may disconnect GPU processes, so do it when GPU is idle
rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia 2>/dev/null || true
modprobe nvidia
modprobe nvidia_uvm

# Verify PeerMappingOverride is enabled
grep PeerMappingOverride /proc/driver/nvidia/params
# Should show: PeerMappingOverride: 1
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

# IMPORTANT: Must set CUDACXX explicitly, otherwise cmake cannot find nvcc
export CUDACXX=/usr/local/cuda-13.0/bin/nvcc

CUDA_HOME=$CUDA_HOME \
CUDACXX=$CUDACXX \
NVSHMEM_IBGDA_SUPPORT=1 \
NVSHMEM_USE_GDRCOPY=0 \
NVSHMEM_SHMEM_SUPPORT=0 \
NVSHMEM_UCX_SUPPORT=0 \
NVSHMEM_USE_NCCL=0 \
NVSHMEM_MPI_SUPPORT=0 \
cmake -GNinja -S . -B build/ \
    -DCMAKE_INSTALL_PREFIX=/opt/deepep/nvshmem \
    -DCMAKE_CUDA_COMPILER=$CUDACXX \
    -DCMAKE_CUDA_ARCHITECTURES=100 \
    -DNVSHMEM_BUILD_EXAMPLES=OFF
# CUDA arch: 100 for B200, 90 for H100, 80 for A100
# Note: NVSHMEM_USE_GDRCOPY=0 uses PeerMappingOverride instead (simpler setup)

# Use sudo for installation to /opt
sudo cmake --build build/ --target install

# Verify IBGDA support
/opt/deepep/nvshmem/bin/nvshmem-info -a | grep IBGDA
```

### Step 4: DeepEP

```bash
# Ensure libcuda.so exists
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so

# Install PyTorch with CUDA 13.0 support (nightly)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

# Clone and build DeepEP
git clone https://github.com/deepseek-ai/DeepEP.git /tmp/deepep_build
cd /tmp/deepep_build

export LD_LIBRARY_PATH=/opt/deepep/nvshmem/lib:$LD_LIBRARY_PATH
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
1. Install CUDA toolkit: `apt-get install cuda-toolkit-13-0`
2. Set environment: `export CUDA_HOME=/usr/local/cuda-13.0`
3. Create symlink: `ln -s /usr/local/cuda-13.0 /usr/local/cuda`

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
1. Load module: `sudo insmod /tmp/gdrcopy/src/gdrdrv/gdrdrv.ko`
2. If module not found (after reboot), rebuild:
   ```bash
   cd /tmp
   git clone --depth 1 https://github.com/NVIDIA/gdrcopy.git gdrcopy
   cd gdrcopy
   export NVIDIA_SRC_DIR=$(find /usr/src -name "nv-p2p.h" -printf "%h" -quit)
   make driver
   sudo insmod src/gdrdrv/gdrdrv.ko
   ```
3. Note: gdrdrv is optional for single-node DeepEP (NVLink works without it)

---

### Problem: gdrdrv compilation fails with vm_flags_set redefinition

**Symptoms:**
```
error: redefinition of 'vm_flags_set'
error: passing argument 4 of 'proc_create' from incompatible pointer type
```

**Root Cause:**
Building from wrong directory. The detection scripts (`scripts/test_gdrdrv_HAVE_VM_FLAGS_SET.sh`) don't run, so the Makefile doesn't set `-DGDRDRV_HAVE_VM_FLAGS_SET`.

**Diagnosis:**
```bash
# Run detection scripts manually
/tmp/gdrcopy/scripts/test_gdrdrv_HAVE_VM_FLAGS_SET.sh -k $(uname -r)
# Should output "y" for kernel 6.3+
```

**Solution:**
```bash
# WRONG - don't build from src/gdrdrv directly:
# cd /tmp/gdrcopy/src/gdrdrv && make  ❌

# CORRECT - build from project root:
cd /tmp/gdrcopy
export NVIDIA_SRC_DIR=$(find /usr/src -name "nv-p2p.h" -printf "%h" -quit)
make driver  ✅
```

**Technical Details:**
- Linux 6.3+ already defines `vm_flags_set()` in `<linux/mm.h>`
- gdrcopy has a fallback definition for older kernels
- Detection script tests if kernel has the function and sets `HAVE_VM_FLAGS_SET=y`
- This tells the compiler to skip the fallback definition via `#ifndef GDRDRV_HAVE_VM_FLAGS_SET`

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

---

### Problem: NVSHMEM "Unable to create ah" on RoCE (Internode)

**Symptoms:**
```
ibgda.cpp:2234 NVSHMEM ERROR: ibv_create_ah_from_wc: Unable to create ah
nvshmem_transport_init() failed
```

**Root Cause:**
NVSHMEM IBGDA transport has uninitialized `struct ibv_ah_attr` containing garbage values. On RoCE (unlike InfiniBand), the `static_rate` field must be properly initialized or it causes EINVAL.

**Reference:** [GitHub Issue #21](https://github.com/NVIDIA/nvshmem/issues/21)

**Solution - Apply memset fix:**

```bash
# 1. Download NVSHMEM source
wget https://developer.nvidia.com/downloads/assets/secure/nvshmem/nvshmem_src_3.4.5-0.txz
tar -xf nvshmem_src_3.4.5-0.txz
cd nvshmem-3.4.5-0

# 2. Find and patch the IBGDA transport
# Location: src/modules/transport/ibgda/ibgda.cpp
# Find: struct ibv_ah_attr ah_attr;
# Add after: memset(&ah_attr, 0, sizeof(ah_attr));

# There are multiple locations (lines ~248, ~1369, ~2157)
# The critical one is around line 2157 in create_dc_target()

# 3. Rebuild NVSHMEM with the fix
export CUDACXX=/usr/local/cuda/bin/nvcc
CUDA_HOME=/usr/local/cuda \
NVSHMEM_IBGDA_SUPPORT=1 \
NVSHMEM_USE_GDRCOPY=0 \
cmake -GNinja -S . -B build/ \
    -DCMAKE_INSTALL_PREFIX=/opt/deepep/nvshmem \
    -DCMAKE_CUDA_COMPILER=$CUDACXX \
    -DCMAKE_CUDA_ARCHITECTURES=100

sudo cmake --build build/ --target install
```

---

### Problem: PyTorch bundled NVSHMEM overrides custom build

**Symptoms:**
- Applied memset fix to `/opt/deepep/nvshmem/lib/` but still getting "Unable to create ah" error
- Custom NVSHMEM build not being used despite LD_LIBRARY_PATH

**Root Cause:**
PyTorch nightly includes its own NVSHMEM libraries in:
```
~/.local/lib/python3.12/site-packages/nvidia/nvshmem/lib/
```

Python loads these bundled libraries **before** your custom `/opt/deepep/nvshmem/lib/` version, regardless of LD_LIBRARY_PATH.

**Diagnosis:**
```bash
# Check if PyTorch has bundled NVSHMEM
ls ~/.local/lib/python3.*/site-packages/nvidia/nvshmem/lib/

# You'll see:
# libnvshmem_host.so.3
# nvshmem_transport_ibgda.so.3  # <-- This is the problem
# nvshmem_transport_ibrc.so.3
```

**Solution - Replace PyTorch's bundled IBGDA module:**

```bash
# Get PyTorch's NVSHMEM location
PYTORCH_NVSHMEM_DIR=$(python3 -c "import site; print(site.getusersitepackages())" | sed 's|site-packages|site-packages/nvidia/nvshmem/lib|')

# Replace bundled IBGDA with patched version
cp /opt/deepep/nvshmem/lib/nvshmem_transport_ibgda.so.3.0.0 \
   ${PYTORCH_NVSHMEM_DIR}/nvshmem_transport_ibgda.so.3

# Verify
ls -la ${PYTORCH_NVSHMEM_DIR}/nvshmem_transport_ibgda.so.3
```

**Important:** This must be done on ALL nodes in the cluster.

---

### Problem: GID Index misconfiguration for RoCE

**Symptoms:**
- NVSHMEM initialization fails on RoCE networks
- Works on InfiniBand but not on RoCE

**Root Cause:**
RoCE uses GID (Global ID) index to select the correct network address format:
- GID[0-1]: Link-local addresses
- GID[2]: RoCE v1
- GID[3]: RoCE v2 with IPv4-mapped address (recommended)

**Solution:**

```bash
# Set GID index for RoCE v2
export NVSHMEM_IB_GID_INDEX=3

# Also recommended for RoCE:
export NVSHMEM_DISABLE_CUDA_VMM=1
export NVSHMEM_DEBUG=WARN  # For debugging
```

Add these to your launch script before running DeepEP tests.

## Reloading gdrdrv After Reboot

The gdrdrv module built via `insmod` is NOT persistent across reboots. After reboot:

```bash
# Option 1: If /tmp/gdrcopy still exists (unlikely after reboot)
sudo insmod /tmp/gdrcopy/src/gdrdrv/gdrdrv.ko

# Option 2: Rebuild the module (recommended)
cd /tmp
git clone --depth 1 https://github.com/NVIDIA/gdrcopy.git gdrcopy
cd gdrcopy
export NVIDIA_SRC_DIR=$(find /usr/src -name "nv-p2p.h" -printf "%h" -quit)
make driver
sudo insmod src/gdrdrv/gdrdrv.ko
lsmod | grep gdrdrv  # Verify loaded
```

**Note:** gdrdrv is optional for single-node NVLink communication. DeepEP works without it:
- **With gdrdrv**: Faster CPU↔GPU small data copies
- **Without gdrdrv**: NVLink and IBGDA still work at full speed

## Verification and Testing

### Module Verification

```bash
# Check PeerMappingOverride
grep PeerMappingOverride /proc/driver/nvidia/params
# Should show: PeerMappingOverride: 1

# Check RDMA
lsmod | grep mlx5_core

# NVSHMEM IBGDA check
/opt/deepep/nvshmem/bin/nvshmem-info -a | grep "NVSHMEM_IBGDA_SUPPORT=ON"
```

### DeepEP Intranode Test

```bash
export LD_LIBRARY_PATH=/opt/deepep/nvshmem/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

cd /tmp/deepep_build

# Quick test with 2 GPUs
python3 tests/test_intranode.py --num-processes 2

# Full test with all GPUs
python3 tests/test_intranode.py --num-processes 8
```

### DeepEP Internode Test (Multi-Node)

For testing DeepEP across multiple nodes with RDMA/RoCE:

**1. Create the test script (`/tmp/test_deepep_internode.py`):**

```python
#!/usr/bin/env python3
"""Simple DeepEP internode connectivity test."""

import os
import sys
import torch
import torch.distributed as dist
import deep_ep

def main():
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    global_rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    local_world_size = int(os.getenv('LOCAL_WORLD_SIZE', 8))

    num_nodes = world_size // local_world_size
    node_rank = global_rank // local_world_size

    print(f"[Rank {global_rank}] Starting: node={node_rank}/{num_nodes}, "
          f"local_rank={local_rank}", flush=True)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    try:
        group = dist.new_group(list(range(world_size)))
        buffer = deep_ep.Buffer(
            group,
            int(256e6),  # 256MB for rdma
            int(128e6),  # 128MB for nvl
            low_latency_mode=False,
            num_qps_per_rank=24,
            explicitly_destroy=True  # Required for explicit destroy()
        )
        print(f"[Rank {global_rank}] DeepEP buffer created!", flush=True)

        # Test dispatch layout
        num_experts = 256
        num_tokens = 512
        num_topk = 8
        torch.manual_seed(42 + global_rank)
        topk_idx = torch.randint(0, num_experts, (num_tokens, num_topk), device='cuda')

        buffer.get_dispatch_layout(topk_idx, num_experts)
        print(f"[Rank {global_rank}] Dispatch layout test PASSED!", flush=True)

        buffer.destroy()
    except Exception as e:
        print(f"[Rank {global_rank}] ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        dist.destroy_process_group()
        sys.exit(1)

    dist.barrier()
    dist.destroy_process_group()
    if global_rank == 0:
        print("\n=== All internode tests PASSED! ===", flush=True)

if __name__ == '__main__':
    main()
```

**2. Create the launcher script (`/tmp/run_internode_test.sh`):**

```bash
#!/bin/bash
set -e

MASTER_ADDR="${1:-10.8.0.2}"
MASTER_PORT="${2:-29500}"
NODE_RANK="${3:-0}"
NUM_GPUS=8

unset WORLD_SIZE RANK

export LD_LIBRARY_PATH=/opt/deepep/nvshmem/lib:${LD_LIBRARY_PATH}
export CUDA_HOME=/usr/local/cuda
export NVSHMEM_HOME=/opt/deepep/nvshmem

# RoCE configuration
export NVSHMEM_IB_GID_INDEX=3
export NVSHMEM_DISABLE_CUDA_VMM=1
export NVSHMEM_DEBUG=WARN

echo "Starting node $NODE_RANK with MASTER_ADDR=$MASTER_ADDR"

exec torchrun \
    --nnodes=2 \
    --nproc_per_node=$NUM_GPUS \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    /tmp/test_deepep_internode.py
```

**3. Run on both nodes:**

```bash
# Node 0 (master, e.g., 10.8.0.2)
bash /tmp/run_internode_test.sh 10.8.0.2 29500 0

# Node 1 (worker, e.g., 10.8.0.108)
bash /tmp/run_internode_test.sh 10.8.0.2 29500 1
```

**Expected output:**
```
[Rank 0-15] DeepEP buffer created!
[Rank 0-15] Dispatch layout test PASSED!
=== All internode tests PASSED! ===
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

# Runtime
export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Build-time
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$CUDA_HOME/lib64/stubs:$LIBRARY_PATH

# NVSHMEM options
export NVSHMEM_IBGDA_SUPPORT=1

# CUDA architecture (set during build)
# B200: TORCH_CUDA_ARCH_LIST="10.0"
# H100: TORCH_CUDA_ARCH_LIST="9.0"
# A100: TORCH_CUDA_ARCH_LIST="8.0"
```

## Environment Setup Script

After installation, create an environment script for easy activation:

```bash
cat > /opt/deepep/env.sh << 'EOF'
#!/bin/bash
# DeepEP Environment Configuration
# Usage: source /opt/deepep/env.sh

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
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

export LD_LIBRARY_PATH=${NVSHMEM_HOME}/lib:${CUDA_HOME}/lib64:${NVIDIA_LIB_PATHS}${LD_LIBRARY_PATH}

# HuggingFace cache on LSSD (if available)
[ -d /lssd/huggingface ] && export HF_HOME=/lssd/huggingface

echo "DeepEP environment loaded"
echo "  NVSHMEM_HOME: $NVSHMEM_HOME"
EOF

chmod +x /opt/deepep/env.sh
echo 'source /opt/deepep/env.sh' >> ~/.bashrc
```

## Terminal Configuration

### Enable tmux Mouse Support

When working in tmux sessions (recommended for long-running installations), enable mouse support for easier scrolling and pane selection:

```bash
# Add mouse support to tmux config
if ! grep -q "set -g mouse on" ~/.tmux.conf 2>/dev/null; then
    echo "set -g mouse on" >> ~/.tmux.conf
    echo "✓ tmux mouse support enabled"
fi

# Reload tmux config if inside tmux
if [ -n "$TMUX" ]; then
    tmux source-file ~/.tmux.conf
    echo "✓ tmux config reloaded"
fi
```

**Benefits:**
- Scroll through long build outputs with mouse wheel
- Click to select panes in split view
- Click to select windows in status bar
- Resize panes by dragging borders

## Resources

### scripts/

- `install-deepep.sh` - Complete automated installation script

### references/

- `troubleshooting.md` - Extended troubleshooting guide with more edge cases

## PyTorch Version Compatibility

### CRITICAL: PyTorch ABI Compatibility

DeepEP is compiled against a specific PyTorch version. If PyTorch is upgraded or downgraded after DeepEP installation, you will see ABI errors:

**Symptom:**
```
ImportError: .../deep_ep_cpp.cpython-312-x86_64-linux-gnu.so: undefined symbol: _ZNK3c1010TensorImpl15incref_pyobjectEv
```

**Common Causes:**
1. Installing vLLM after DeepEP (vLLM pins PyTorch 2.9.1)
2. Installing FlashInfer (may upgrade PyTorch to 2.10)
3. Reinstalling PyTorch with a different version

**Solution: Recompile DeepEP**
```bash
cd /tmp/deepep_build  # or wherever DeepEP was cloned
rm -rf build/ dist/ *.egg-info
rm -rf ~/.local/lib/python3.12/site-packages/deep_ep-*.egg

export CUDA_HOME=/usr/local/cuda-13.0
export LD_LIBRARY_PATH=/opt/deepep/nvshmem/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$CUDA_HOME/lib64/stubs:$LIBRARY_PATH

TORCH_CUDA_ARCH_LIST="10.0" NVSHMEM_DIR=/opt/deepep/nvshmem python3 setup.py install --user
```

### Recommended Installation Order

When installing DeepEP with SGLang and vLLM, follow this order:

```
1. LSSD Mount (/lssd-mounter)
2. DeepEP (this skill)
3. SGLang (/sglang-installer)
4. vLLM (/vllm-installer)
5. *** Recompile DeepEP *** (after vLLM changes PyTorch)
```

**Important:** After installing vLLM, always verify DeepEP works:
```bash
python3 -c "import deep_ep; print('DeepEP OK')"
```

If it fails with ABI errors, recompile DeepEP as shown above.

### PyTorch Version Matrix

| Framework | PyTorch Version | Notes |
|-----------|-----------------|-------|
| DeepEP (initial) | nightly+cu130 | From pip install torch (nightly for CUDA 13.0) |
| SGLang 0.5.8 | 2.9.1+cu126 | May be different |
| vLLM 0.14.1 | 2.9.1+cu126 | Pins this version |
| **Final (after vLLM)** | **2.9.1+cu126** | DeepEP needs recompile |

## Unified Environment Script

After installing DeepEP + SGLang + vLLM, use the unified environment script:

```bash
source /opt/deepep/unified-env.sh
```

This script is automatically created and includes all necessary paths for DeepEP, SGLang, and vLLM.

## Critical IBGDA Runtime Configuration

When running DeepEP with IBGDA transport (required for internode communication), several environment variables are **critical**:

### LD_PRELOAD - Force Custom NVSHMEM

**Problem:** PyTorch ships `nvidia-nvshmem-cu13` v3.4.5, which loads before your custom NVSHMEM 3.5.19.

**Symptom:**
```
NVSHMEM device library version does not match with NVSHMEM host library version
```

**Solution:**
```bash
export LD_PRELOAD=/opt/deepep/nvshmem/lib/libnvshmem_host.so.3
```

### NVSHMEM_REMOTE_TRANSPORT - Force IBGDA

**Problem:** NVSHMEM defaults to IBRC transport, which doesn't support GPU-initiated APIs that DeepEP requires.

**Symptom:**
```
NVSHMEM INFO Successfully initialized the transport: IBRC
# (but DeepEP dispatch operations fail or timeout)
```

**Solution:**
```bash
export NVSHMEM_REMOTE_TRANSPORT=ibgda
```

### NVSHMEM_IBGDA_NIC_HANDLER - GDRCopy CPU Handler

**Problem:** Without GDRCopy, IBGDA falls back to host memory mode which is slower.

**Solution:**
```bash
export NVSHMEM_IBGDA_NIC_HANDLER=cpu
```

### NVSHMEM_HCA_PREFIX - RoCE Device Detection

**Problem:** GCP B200 uses RoCE with `rocep*` device names, not InfiniBand `mlx5*`.

**Solution:**
```bash
export NVSHMEM_HCA_PREFIX=rocep
```

### Complete Runtime Environment

Create and source `/opt/deepep/unified-env.sh`:

```bash
#!/bin/bash
# DeepEP Unified Environment for IBGDA

export NVSHMEM_HOME=/opt/deepep/nvshmem
export GDRCOPY_HOME=/opt/deepep/gdrcopy
export CUDA_HOME=/usr/local/cuda

# Library paths
export LD_LIBRARY_PATH=${NVSHMEM_HOME}/lib:${GDRCOPY_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# CRITICAL: Force our NVSHMEM over PyTorch's bundled version
export LD_PRELOAD=${NVSHMEM_HOME}/lib/libnvshmem_host.so.3

# NVSHMEM IBGDA Configuration
export NVSHMEM_REMOTE_TRANSPORT=ibgda
export NVSHMEM_IBGDA_NIC_HANDLER=cpu
export NVSHMEM_HCA_PREFIX=rocep

# DeepEP source path (for tests)
export PYTHONPATH=/tmp/deepep_build:${PYTHONPATH}
```

---

### Problem: nvidia_peermem fails to load

**Symptom:**
```
modprobe nvidia_peermem
modprobe: ERROR: could not insert 'nvidia_peermem': Invalid parameters
```

**Root Cause:**
nvidia_peermem may have compatibility issues with certain driver/kernel combinations. However, **this is NOT a blocker** - NVSHMEM IBGDA uses dma-buf for GPU memory registration instead.

**Verification that dma-buf works:**
```
NVSHMEM INFO ibv_reg_dmabuf_mr handle 0x... mr 0x...
```

**Solution:** Ignore the nvidia_peermem error if you see successful `ibv_reg_dmabuf_mr` in NVSHMEM logs.

---

### Problem: Old DeepEP in site-packages conflicts

**Symptom:**
- DeepEP imports but uses old cached version
- Module not reflecting new build

**Solution:**
```bash
# Remove old installation
rm -rf ~/.local/lib/python3.*/site-packages/deep_ep*

# Reinstall from build
cd /tmp/deepep_build
python3 setup.py install --user
```

---

### Problem: Ubuntu 24.04 NVSHMEM IBGDA compilation fails - MLX5DV not found

**Symptoms:**
```
'MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT' was not declared in this scope
'MLX5DV_VERSION' was not declared in this scope
```

**Root Cause:**
Ubuntu 24.04's built-in rdma-core 50.0 is too old and lacks the MLX5DV extensions required for NVSHMEM IBGDA support.

**Diagnosis:**
```bash
# Check rdma-core version
dpkg -l rdma-core | grep rdma-core

# Check if MLX5DV extensions exist
grep "MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT" /usr/include/infiniband/mlx5_api.h
# No output = missing extensions
```

**Solution - Install DOCA-OFED 2.9.0:**
```bash
# Add DOCA repository
curl -fsSL https://linux.mellanox.com/public/repo/doca/GPG-KEY-Mellanox.pub | \
    gpg --dearmor -o /usr/share/keyrings/doca.gpg
echo "deb [signed-by=/usr/share/keyrings/doca.gpg] \
    https://linux.mellanox.com/public/repo/doca/2.9.0/ubuntu24.04/x86_64/ ./" \
    > /etc/apt/sources.list.d/doca.list
apt-get update

# IMPORTANT: Install specific mft version first to avoid conflicts
apt-get install -y mft=4.30.0-139

# Install DOCA-OFED userspace
apt-get install -y doca-ofed-userspace

# Verify
grep "MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT" /usr/include/infiniband/mlx5_api.h
```

**Note:** CUDA repository provides mft 4.34.x which conflicts with DOCA-OFED's requirement for 4.30.0-139.

---

### Problem: Internode communication fails after installation

**Symptoms:**
- Intranode tests pass (24/24)
- Internode tests fail with -1 counters or no data
- RDMA connectivity verified working via `ib_write_bw`

**Root Cause:**
NVIDIA driver's `PeerMappingOverride=1` option is not active. This setting enables dma-buf GPU memory registration required for IBGDA transport.

**Diagnosis:**
```bash
# Check if PeerMappingOverride is enabled
grep PeerMappingOverride /proc/driver/nvidia/params
# Should show: PeerMappingOverride: 1

# If shows 0 or missing, the config wasn't applied
cat /etc/modprobe.d/nvidia-peermem.conf
```

**Solution:**
```bash
# Create modprobe config
echo 'options nvidia NVreg_EnableStreamMemOPs=1 NVreg_RegistryDwords="PeerMappingOverride=1;"' \
    > /etc/modprobe.d/nvidia-peermem.conf

# REBOOT REQUIRED - cannot hot-reload this setting
sudo reboot
```

**Verification after reboot:**
```bash
# Check PeerMappingOverride is now 1
grep PeerMappingOverride /proc/driver/nvidia/params
# Expected: PeerMappingOverride: 1

# Check dma-buf registration in NVSHMEM logs
export NVSHMEM_DEBUG=INFO
# Look for: ibv_reg_dmabuf_mr handle 0x... mr 0x...
```

---

## Version History

- **2026-02-01**: Added Ubuntu 24.04 specific requirements
  - **CRITICAL**: Ubuntu 24.04 rdma-core 50.0 lacks `MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT` required for IBGDA
  - **SOLUTION**: Install DOCA-OFED 2.9.0 userspace: `apt-get install doca-ofed-userspace`
  - **NEW**: mft package version conflict - CUDA repo has 4.34.x, DOCA needs 4.30.0-139
  - **FIX**: Install specific mft version first: `apt-get install -y mft=4.30.0-139`
  - **CRITICAL**: PeerMappingOverride driver option requires **reboot** to take effect
  - **NEW**: GDRCopy library also needs LD_PRELOAD for runtime loading
  - **UPDATE**: LD_PRELOAD now includes both libnvshmem_host.so.3 and libgdrapi.so.2

- **2026-01-31**: Added IBGDA runtime configuration section
  - **CRITICAL**: Documented LD_PRELOAD requirement for NVSHMEM version conflict
  - **CRITICAL**: Documented NVSHMEM_REMOTE_TRANSPORT=ibgda requirement
  - **NEW**: NVSHMEM_IBGDA_NIC_HANDLER=cpu for GDRCopy support
  - **NEW**: NVSHMEM_HCA_PREFIX=rocep for GCP RoCE devices
  - **NEW**: nvidia_peermem failure is non-blocking if dma-buf works
  - **NEW**: Complete unified-env.sh script with all required variables

- **2026-01-31**: Added internode RoCE troubleshooting and testing
  - **NEW**: Troubleshooting for "Unable to create ah" NVSHMEM error on RoCE
  - **ROOT CAUSE**: Uninitialized `struct ibv_ah_attr` causes EINVAL on RoCE (GitHub Issue #21)
  - **FIX**: Apply memset fix to NVSHMEM IBGDA transport module
  - **NEW**: Troubleshooting for PyTorch bundled NVSHMEM overriding custom build
  - **CRITICAL**: PyTorch nightly bundles NVSHMEM in `~/.local/lib/python3.*/site-packages/nvidia/nvshmem/lib/`
  - **FIX**: Replace PyTorch's bundled IBGDA module with patched version
  - **NEW**: GID index configuration for RoCE v2 (`NVSHMEM_IB_GID_INDEX=3`)
  - **NEW**: Complete internode test script and launcher for multi-node DeepEP verification
  - **UPDATED**: setup-deepep.sh now automatically replaces PyTorch's bundled NVSHMEM

- **2026-01-30**: Simplified installation - removed gdrcopy, updated to CUDA 13.0
  - **CHANGE**: Removed gdrcopy from default installation, using PeerMappingOverride instead
  - **CHANGE**: CUDA Toolkit version changed from 12.9 to 13.0
  - **CHANGE**: PyTorch wheel changed to nightly/cu130 for native CUDA 13.0 support
  - **NOTE**: Installation is now simpler: CUDA → PeerMappingOverride → NVSHMEM → DeepEP

- **2026-01-29**: Fixed gdrdrv kernel 6.14+ compatibility
  - **ROOT CAUSE**: gdrdrv supports 6.14+ kernels, but must be built from project root directory
  - **FIX**: Updated Step 2 to emphasize correct build procedure (`make driver` from `/tmp/gdrcopy`)
  - **NEW**: Added troubleshooting for `vm_flags_set redefinition` error
  - **NEW**: Added "Reloading gdrdrv After Reboot" section
  - **CLARIFIED**: gdrdrv is optional for single-node NVLink - IBGDA handles cross-node RDMA
  - **UPDATED**: Use latest `main` branch instead of v2.5.1 for better kernel compatibility

- **2026-01-29**: Synchronized with deepep-installer.yaml
  - **CHANGE**: Installation order now matches YAML: DOCA OFED → GPU Driver → gdrcopy → PeerMappingOverride → nvidia_peermem → Reboot Check → NVSHMEM → DeepEP
  - **NEW**: Added `check_reboot_needed()` function with YAML logic (lines 156-178)
  - **NEW**: Added `configure_peermapping()` for SGLang PeerMappingOverride
  - **NEW**: Added `load_nvidia_peermem()` to attempt loading nvidia_peermem
  - **NEW**: Added `AUTO_REBOOT=1` environment variable for automatic reboot
  - **FIX**: gdrcopy now checks for `libgdr*` per YAML instead of `libgdrapi*`
  - **FIX**: NVSHMEM now installs `libibverbs-dev librdmacm-dev` dependencies
  - **FIX**: DeepEP build creates symlink per YAML

- **2026-01-29**: Added automatic driver installation
  - **NEW**: `install_nvidia_driver()` - Auto-installs nvidia-open-575 if GPU driver not loaded
  - **NEW**: `install_doca_ofed()` now auto-installs DOCA 3.0.0 if not present (no more interactive prompt)
  - **CHANGE**: Installation order updated to: GPU Driver → DOCA OFED → CUDA → gdrcopy → NVSHMEM → DeepEP
  - **NOTE**: If driver install requires reboot, script will prompt user to reboot and re-run

- **2026-01-29**: Improved install-deepep.sh for non-interactive mode
  - **FIX**: Script now auto-skips DOCA installation in non-interactive mode (no stdin)
  - **NEW**: Added SKIP_DOCA=1 environment variable to skip DOCA prompt
  - **NOTE**: Built-in mlx5 driver is usually sufficient for IBGDA

- **2026-01-29**: Added PyTorch ABI compatibility section
  - **CRITICAL**: Documented PyTorch version mismatch causing ABI errors
  - **NEW**: Added recompilation instructions after vLLM install
  - **NEW**: Added recommended installation order for multi-framework setup
  - **NEW**: Added PyTorch version matrix for different frameworks
  - **NEW**: Referenced unified environment script

- **2026-01-29**: Added Terminal Configuration section
  - **NEW**: tmux mouse support setup for easier scrolling and pane selection

- **2026-01-29**: Updated based on full installation workflow
  - **CRITICAL**: Added CUDACXX and CMAKE_CUDA_COMPILER requirement for NVSHMEM cmake
  - Added sudo requirement for NVSHMEM installation to /opt
  - Fixed cmake command format (removed trailing backslash issues)

- **2026-01-28**: Updated based on installation experience on GCP with kernel 6.14.0
  - Added pip installation check for Ubuntu 24.04
  - Clarified NVIDIA_SRC_DIR requirement for gdrcopy
  - Added DOCA OFED kernel compatibility warning
  - Added environment setup script generation
