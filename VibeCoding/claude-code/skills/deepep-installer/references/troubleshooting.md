# DeepEP Troubleshooting Reference

This document provides extended troubleshooting guidance for DeepEP installation issues across different environments.

## Ubuntu 24.04 Specific Issues

### MLX5DV Extensions Missing

Ubuntu 24.04's built-in rdma-core 50.0 lacks the MLX5DV extensions required for NVSHMEM IBGDA.

**Symptom:**
```
'MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT' was not declared in this scope
```

**Solution:**
```bash
# Install DOCA-OFED 2.9.0 userspace
curl -fsSL https://linux.mellanox.com/public/repo/doca/GPG-KEY-Mellanox.pub | \
    gpg --dearmor -o /usr/share/keyrings/doca.gpg
echo "deb [signed-by=/usr/share/keyrings/doca.gpg] \
    https://linux.mellanox.com/public/repo/doca/2.9.0/ubuntu24.04/x86_64/ ./" \
    > /etc/apt/sources.list.d/doca.list
apt-get update

# Fix mft version conflict (CUDA repo has 4.34.x, DOCA needs 4.30.0-139)
apt-get install -y mft=4.30.0-139
apt-get install -y doca-ofed-userspace
```

### PeerMappingOverride Not Active After Config

**Symptom:**
- Config file exists at `/etc/modprobe.d/nvidia-peermem.conf`
- But `grep PeerMappingOverride /proc/driver/nvidia/params` shows 0

**Solution:**
This setting requires a **full reboot**. It cannot be hot-reloaded by unloading/loading the nvidia module.

```bash
sudo reboot
# After reboot, verify:
grep PeerMappingOverride /proc/driver/nvidia/params
# Expected: PeerMappingOverride: 1
```

### GDRCopy Runtime Loading

**Symptom:**
```
NVSHMEM_IBGDA_NIC_HANDLER=cpu requires GDRCopy
```

Even with GDRCopy installed to `/opt/deepep/gdrcopy`.

**Solution:**
Add libgdrapi to LD_PRELOAD:
```bash
export LD_PRELOAD="/opt/deepep/nvshmem/lib/libnvshmem_host.so.3:/opt/deepep/gdrcopy/lib/libgdrapi.so.2"
```

## Environment Detection

### Identifying the System Type

```bash
# Check if running in container
cat /proc/1/cgroup 2>/dev/null | grep -q docker && echo "Docker container"
cat /proc/1/cgroup 2>/dev/null | grep -q kubepods && echo "Kubernetes pod"

# Check cloud provider
curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone 2>/dev/null && echo "GCP"
curl -s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null && echo "AWS"

# Check OS version
cat /etc/os-release | grep -E "^(NAME|VERSION)="

# Check kernel version
uname -r
```

### GPU Detection

```bash
# List GPUs
nvidia-smi -L

# Check GPU architecture
nvidia-smi --query-gpu=compute_cap --format=csv

# Mapping:
# 10.0 = B200/B100 (Blackwell)
# 9.0  = H100/H200 (Hopper)
# 8.0  = A100 (Ampere)
# 8.6  = A40/A10 (Ampere)
# 7.5  = T4 (Turing)
```

## Driver Version Mismatches

### Problem: NVIDIA kernel source version mismatch

When gdrcopy fails with version mismatch errors.

**Diagnosis:**
```bash
# Check installed driver version
cat /proc/driver/nvidia/version

# Check available kernel sources
ls /usr/src/ | grep nvidia

# Check currently loaded modules version
modinfo nvidia | grep version
```

**Solutions:**

1. Match kernel source to driver version:
   ```bash
   # Extract version from driver
   DRIVER_VERSION=$(cat /proc/driver/nvidia/version | head -1 | awk '{print $8}')
   echo "Driver version: $DRIVER_VERSION"

   # Install matching source
   apt-get install nvidia-kernel-source-${DRIVER_VERSION%%.*}-open
   ```

2. For custom drivers (e.g., GKE, cloud VMs):
   ```bash
   # Find where nv-p2p.h might be
   find /usr -name "nv-p2p.h" 2>/dev/null
   find /opt -name "nv-p2p.h" 2>/dev/null

   # May be in non-standard locations on cloud VMs
   ```

## DOCA vs System RDMA

### When is DOCA needed?

DOCA (NVIDIA/Mellanox OFED) provides:
- Enhanced RDMA performance
- GPUDirect RDMA optimizations
- Better IBGDA support

However, for DeepEP on modern systems:
- **Not strictly required** if system rdma-core works
- `libibverbs-dev` and `librdmacm-dev` are sufficient for NVSHMEM IBGDA

### Checking RDMA Stack

```bash
# System RDMA (rdma-core)
dpkg -l | grep rdma-core

# DOCA/OFED
ofed_info -s 2>/dev/null || echo "DOCA/OFED not installed"

# RDMA library providers
ls /usr/lib/x86_64-linux-gnu/libibverbs/
```

### DOCA Installation (if needed)

```bash
# For Ubuntu 24.04
wget https://www.mellanox.com/downloads/DOCA/DOCA_v3.0.0/host/doca-host_3.0.0-058000-25.04-ubuntu2404_amd64.deb
dpkg -i doca-host_*.deb
apt-get update && apt-get install doca-ofed
ofed_info -s
```

## GKE-Specific Issues

### Pre-installed Driver Conflicts

GKE nodes often have pre-installed NVIDIA drivers that conflict with package installations.

**Symptoms:**
- `apt-get install nvidia-*` fails with conflicts
- `modprobe nvidia_peermem` fails because nvidia-dkms conflicts

**Solutions:**

1. Use existing driver, only install toolkit:
   ```bash
   apt-get install cuda-toolkit-12-9  # No driver
   ```

2. For gdrcopy, install only kernel source:
   ```bash
   apt-get install nvidia-kernel-source-580-open
   # NOT nvidia-dkms-580-open (conflicts)
   ```

3. Accept nvidia_peermem won't be available:
   - DeepEP works without it
   - NVSHMEM IBGDA is the critical path

### GKE GPUDirect RDMA Setup

On GKE with GPUDirect RDMA:
```bash
# Check if gpuXrdma devices exist
ls /sys/class/infiniband/ | grep gpu

# Expected output for 8-GPU system:
# gpu0rdma0 gpu1rdma0 ... gpu7rdma0

# These are the RoCE devices paired with GPUs
```

## VM vs Bare Metal Differences

### VM Considerations

1. **Driver installation**: Often pre-installed, don't overwrite
2. **Kernel modules**: May require reboot after DKMS install
3. **RDMA**: RoCE typically available, IB less common
4. **nvidia_peermem**: Often not available

### Bare Metal Considerations

1. **Full control**: Can install any driver version
2. **DOCA**: Recommended for best performance
3. **nvidia_peermem**: Should be available with nvidia-dkms

## Build Failures

### CMake Cache Issues

If build fails after configuration changes:
```bash
rm -rf build/
cmake -GNinja -S . -B build/ [options]
```

### Ninja vs Make

NVSHMEM prefers Ninja:
```bash
apt-get install ninja-build
cmake -GNinja ...
```

Fallback to Make:
```bash
cmake -G "Unix Makefiles" ...
make -j$(nproc)
```

### Python Version Issues

DeepEP requires Python 3.10+:
```bash
python3 --version

# If too old:
apt-get install python3.12 python3.12-dev
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
```

## Runtime Issues

### Import Errors

**Symptom:** `import deep_ep` fails

```python
import deep_ep
# ImportError: libxxx.so not found
```

**Solution:** Set LD_LIBRARY_PATH:
```bash
export LD_LIBRARY_PATH=/opt/deepep/nvshmem/lib:/opt/deepep/gdrcopy/lib:$LD_LIBRARY_PATH
```

### NCCL Conflicts

DeepEP and NCCL can coexist but may have initialization conflicts.

**Symptom:** Hang or crash when using both

**Solution:** Initialize DeepEP first, then NCCL:
```python
import deep_ep  # First
import torch.distributed as dist  # Then NCCL
```

### Multi-GPU Synchronization

**Symptom:** Tests hang with multiple GPUs

**Diagnosis:**
```bash
# Check if all GPUs are accessible
nvidia-smi -L

# Check NVLink topology
nvidia-smi topo -m
```

**Solutions:**
1. Ensure all GPUs have NVLink connectivity
2. Check CUDA_VISIBLE_DEVICES is not limiting GPUs
3. Verify gdrdrv is loaded

## Performance Tuning

### Optimal NVLink Bandwidth

Expected performance on B200 (8-GPU NVLink):
- Dispatch: ~35 GB/s per direction
- Combine: ~36 GB/s per direction

If performance is lower:
1. Check NVLink topology: `nvidia-smi topo -m`
2. Verify IBGDA is enabled: `nvshmem-info -a | grep IBGDA`
3. Check for thermal throttling: `nvidia-smi -q -d PERFORMANCE`

### SM Count Tuning

The test automatically tunes SM count, but manual override:
```python
# In DeepEP code
config.num_sms = 24  # Typical optimal value
```

## Recovery Procedures

### Clean Reinstall

If installation is corrupted:
```bash
# Remove installed packages
pip3 uninstall deep_ep -y
rm -rf /opt/deepep/nvshmem
rm -rf /opt/deepep/gdrcopy
rm -rf /tmp/deepep_build
rm -rf /tmp/nvshmem_build_src
rm -rf /tmp/gdrcopy

# Remove kernel modules
rmmod gdrdrv 2>/dev/null || true
apt-get remove --purge gdrdrv-dkms libgdrapi 2>/dev/null || true

# Start fresh
# ... follow installation steps ...
```

### Module Recovery

If kernel modules fail to load after update:
```bash
# Rebuild DKMS modules
dkms autoinstall

# Or manually rebuild gdrcopy
cd /tmp/gdrcopy/packages
./build-deb-packages.sh
dpkg -i *.deb
modprobe gdrdrv
```
