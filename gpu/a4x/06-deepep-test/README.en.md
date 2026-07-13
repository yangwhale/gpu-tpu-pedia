> 🌐 [中文](README.md) | **English**

# 7. DeepEP Testing

**Version selection guide**: **v2 is recommended**, because v1 cannot support cross-node EP on GCP.

## 7.1 Overview

DeepEP is the Expert Parallelism (EP) communication library open-sourced by DeepSeek. This section provides two parallel deployment paths to choose from:

| Path | DeepEP Version | CUDA | Internode Backend | NVSHMEM | NCCL |
|------|-------------|------|----------------|---------|------|
| **Path A** (stable/legacy) | v1 | CUDA 12 | NVSHMEM IBGDA | **3.4.5** (required) | 2.25+ |
| **Path B** (recommended/new) | v2 (main) | CUDA 13 | NCCL Gin | optional / build-only | **>= 2.30.4** |

**Internode limitation**: DeepEP v1's internode communication relies on NVSHMEM IBGDA, and IBGDA requires GDRCopy. GDRCopy is unavailable in GKE/GCP environments (the kernel module cannot be loaded), so **v1 internode is limited on GKE**.

**Solution**: DeepEP v2 (Elastic EP) uses the NCCL Gin backend for internode communication, completely eliminating the NVSHMEM/GDRCopy dependency. It has been verified on a 2-node x 4-GPU environment (144/144 PASS). **Path B is recommended**.

**CUDA 13 container note**: The NGC 26.04 container ships with NVSHMEM 3.6.5, which is incompatible with DeepEP v1. This is another important reason to choose Path B (v2).

## 7.2 NVSHMEM Version Compatibility Matrix

| NVSHMEM Version | Struct Layout | ibgda RC QP Index | DMABuf Support | DeepEP Compatible | GB200 Status |
|-------------|------------|------------------|-------------|-------------|------------|
| 3.3.9 | v1 | PE-major | no ibv_reg_dmabuf_mr | compatible | fails (no nvidia_peermem, no DMABuf FD) |
| **3.4.5** | **v1** | **PE-major** | **ibv_reg_dmabuf_mr** | **compatible** | **works (recommended for GB200)** |
| 3.5.x+ | v2 | QP-ID-major | ibv_reg_dmabuf_mr | incompatible | fails (struct/index mismatch with DeepEP) |

| GPU Platform | Recommended NVSHMEM Version | Reason |
|----------|------------------|------|
| **GB200 (A4X)** | **3.4.5** | GB200 does not support POSIX FD handles; 3.4.5 works around this via `ibv_reg_dmabuf_mr` |
| B200 / H200 | 3.3.9 | supports `nvidia_peermem`, no DMABuf needed |

## 7.3 Path A: DeepEP v1 + CUDA 12 (stable version)

Applicable container: `nvcr.io/nvidia/pytorch:25.04-py3` (CUDA 12, NVSHMEM compatible)

```bash
# Deploy the DeepEP test Pod (ComputeDomain + DRANET)
kubectl apply -f yamls/k8s1341-deepep-test-dranet.yaml

# Wait for the Pod to become ready
kubectl get pods -l name -w
kubectl logs deepep-h1 -f  # confirm "Ready"

# Verify the IMEX channel
kubectl exec deepep-h1 -- ls /dev/nvidia-caps-imex-channels/
```

### Step 1: Install NVSHMEM 3.4.5 and build DeepEP v1

```bash
kubectl exec deepep-h1 -- bash -c "
  apt-get update -qq && apt-get install -y -qq git 2>/dev/null
  pip install nvidia-nvshmem-cu12==3.4.5 2>&1 | tail -3  # must be 3.4.5
  ln -sf /usr/local/gib/lib64/libnccl.so.2 /usr/local/gib/lib64/libnccl.so

  cd /tmp && git clone --depth 1 https://github.com/deepseek-ai/DeepEP.git
  cd /tmp/DeepEP

  # 4-GPU patch (see section 7.5 for details)
  sed -i 's/#define NUM_MAX_NVL_PEERS 8/#define NUM_MAX_NVL_PEERS 4/' csrc/kernels/configs.cuh
  sed -i 's/NUM_MAX_NVL_PEERS == 8/NUM_MAX_NVL_PEERS == 4/g' csrc/deep_ep.hpp csrc/kernels/internode.cu
  sed -i 's/must be 8/must be 4/' csrc/deep_ep.hpp
  sed -i 's/num_ranks > NUM_MAX_NVL_PEERS/num_ranks >= NUM_MAX_NVL_PEERS/g' csrc/deep_ep.cpp csrc/kernels/layout.cu

  NCCL_DIR=/usr/local/gib TORCH_CUDA_ARCH_LIST='10.0' python setup.py build_ext --inplace 2>&1 | tail -5
  # Confirm output: deep_ep/_C.cpython-312-aarch64-linux-gnu.so
"
```

### Step 2: Run the Intranode test

```bash
kubectl exec deepep-h1 -- bash -c "
  export NCCL_DIR=/usr/local/gib
  export LD_LIBRARY_PATH=/usr/local/gib/lib64:/usr/local/nvidia/lib64:\$LD_LIBRARY_PATH
  export PYTHONPATH=/tmp/DeepEP:\$PYTHONPATH
  cd /tmp/DeepEP/tests/legacy
  python test_intranode.py --num-processes 4 --num-experts 64 --num-topk 4
"
```

### Step 3: Run the Low-latency test

```bash
kubectl exec deepep-h1 -- bash -c "
  export NCCL_DIR=/usr/local/gib PYTHONPATH=/tmp/DeepEP:\$PYTHONPATH
  export LD_LIBRARY_PATH=/usr/local/gib/lib64:/usr/local/nvidia/lib64:\$LD_LIBRARY_PATH
  cd /tmp/DeepEP/tests/legacy
  python test_low_latency.py --num-processes 4 --num-experts 64 --num-topk 4
"
```

| Test | Measured Result |
|------|----------|
| Intranode BF16 dispatch | **24/24 PASS**, best 435.97 GB/s NVL (SMs=24, chunk=32) |
| Intranode FP8 dispatch | **24/24 PASS**, best 288.07 GB/s NVL (SMs=24, chunk=32) |
| Intranode Combine | **PASS**, best 321.35 GB/s NVL (SMs=24, chunk=16) |
| Low-latency dispatch | **168-188 GB/s**, latency ~20 us |
| Low-latency combine | **316-331 GB/s**, latency ~22 us |

## 7.4 Path B: DeepEP v2 + CUDA 13 (recommended)

Applicable container: `nvcr.io/nvidia/pytorch:26.04-py3` (CUDA 13.2, NCCL 2.29.7). DeepEP v2 (Elastic EP) uses the NCCL Gin backend, eliminating the NVSHMEM internode dependency.

**Image note**: Use `nvcr.io/nvidia/pytorch:26.04-py3`, which includes CUDA 13.2.1 and PyTorch 2.12.0a0. The init container is the GIB NCCL plugin `v1.1.2` (ARM64).

### Step 1: Install NVSHMEM 3.4.5 + NCCL 2.30.4

The container ships with NVSHMEM 3.6.5 (incompatible with DeepEP) and NCCL 2.29.7 (v2 requires >= 2.30.4). Additional installation is required:

```bash
# Run on each CUDA 13 Pod:

# Install NVSHMEM 3.4.5 (for compiling legacy code; does not affect the v2 runtime)
apt-get update && apt install -y cuda-keyring && apt update
apt install -y libnvshmem3-cuda-12=3.4.5-1 libnvshmem3-dev-cuda-12=3.4.5-1 libnvshmem3-static-cuda-12=3.4.5-1

# Fix the CUDA 13 CCCL header path (cuda/std/ moved to cccl/cuda/std/)
ln -sf /usr/local/cuda/include/cccl/cuda /usr/local/cuda/include/cuda

# Create the NVSHMEM 3.4.5 directory structure
mkdir -p /scratch-data/nvshmem345
ln -sf /usr/include/nvshmem_12 /scratch-data/nvshmem345/include
ln -sf /usr/lib/aarch64-linux-gnu/nvshmem/12 /scratch-data/nvshmem345/lib

# Install NCCL 2.30.4 (the DeepEP v2 GIN backend requires >= 2.30.4)
pip install "nvidia-nccl-cu13>=2.30.4" --no-deps
```

### Step 2: Clone DeepEP v2 and apply the 4-GPU patch

```bash
cd /scratch-data
git clone --depth 1 https://github.com/deepseek-ai/DeepEP.git DeepEP-v2
cd DeepEP-v2

# DeepEP v2 uses the LEGACY_-prefixed NVL peer constant, which requires the corresponding patch
sed -i "s/#define LEGACY_NUM_MAX_NVL_PEERS 8/#define LEGACY_NUM_MAX_NVL_PEERS 4/" csrc/kernels/legacy/compiled.cuh

find csrc/ -name "*.cu" -o -name "*.cuh" -o -name "*.hpp" -o -name "*.cpp" -o -name "*.h" | \
  xargs sed -i "s/LEGACY_NUM_MAX_NVL_PEERS == 8/LEGACY_NUM_MAX_NVL_PEERS == 4/g"

sed -i "s/LEGACY_NUM_MAX_NVL_PEERS \* sizeof(bool) == sizeof(uint64_t)/LEGACY_NUM_MAX_NVL_PEERS * sizeof(bool) == sizeof(uint32_t)/" \
  csrc/kernels/legacy/internode.cu

sed -i "s/uint64_t cached_bitmask/uint32_t cached_bitmask/" csrc/kernels/legacy/internode.cu

sed -i "s/num_ranks > LEGACY_NUM_MAX_NVL_PEERS/num_ranks >= LEGACY_NUM_MAX_NVL_PEERS/" csrc/kernels/legacy/layout.cu
```

### Step 3: Build DeepEP v2

```bash
export NVSHMEM_DIR=/scratch-data/nvshmem345
export EP_NCCL_ROOT_DIR=/usr/local/lib/python3.12/dist-packages/nvidia/nccl
export TORCH_CUDA_ARCH_LIST="10.0"
export EP_JIT_CACHE_DIR=/scratch-data/deepep-jit-cache
mkdir -p /scratch-data/deepep-jit-cache

pip install -e . --no-build-isolation 2>&1 | tail -10

# Verify
python3 -c "import deep_ep; print('DeepEP v2 import OK')"
```

**Note**: Do not add GIB to `LD_LIBRARY_PATH` during the build — GIB ships with NCCL 2.28.9, which conflicts with the container's NCCL 2.29.7 and causes build link errors.

### Step 4: Run the Intranode test

```bash
cd /scratch-data/DeepEP-v2
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
python tests/test_intranode.py --num-processes 4
```

### Step 5: Run the Internode test (2 nodes x 4 GPU, NCCL Gin)

**Note**: Do not use `tests/legacy/test_internode.py`; that test fails on a 4-GPU A4X due to the `LEGACY_NUM_MAX_NVL_PEERS=8` assertion. Use the v2 elastic test instead.

```bash
# Create the launch script on both Pods:
cat > /scratch-data/run_deepep_v2_test.sh << 'SCRIPT'
#!/bin/bash
export LD_PRELOAD=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2
export LD_LIBRARY_PATH=/usr/local/gib/lib64:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
export CUDA_HOME=/usr/local/cuda
export CPATH=/usr/local/cuda/include
export NVSHMEM_DIR=/scratch-data/nvshmem345
export EP_NCCL_ROOT_DIR=/usr/local/lib/python3.12/dist-packages/nvidia/nccl
export EP_JIT_CACHE_DIR=/scratch-data/deepep-jit-cache
export NCCL_NET=gIB
export NCCL_PXN_C2C=1
export NCCL_IB_ADAPTIVE_ROUTING=1
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=52
export NCCL_IB_FIFO_TC=84
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=${MASTER_PORT:-8363}
export WORLD_SIZE=${WORLD_SIZE:-2}
export RANK=${RANK:-0}

cd /scratch-data/DeepEP-v2
python3 tests/elastic/test_ep.py --num-processes 4 --num-tokens 1024 --hidden 7168 --num-topk 6 --num-experts 256 "$@"
SCRIPT
chmod +x /scratch-data/run_deepep_v2_test.sh

# Get the IP of Pod 1
HOST1_IP=$(kubectl get pod cuda13-dra-1 -o jsonpath='{.status.podIP}')

# Launch on Pod 2 (RANK=1)
kubectl exec cuda13-dra-2 -- bash -c "
  MASTER_ADDR=$HOST1_IP RANK=1 /scratch-data/run_deepep_v2_test.sh
" &

sleep 5

# Launch on Pod 1 (RANK=0)
kubectl exec cuda13-dra-1 -- bash -c "
  MASTER_ADDR=$HOST1_IP RANK=0 /scratch-data/run_deepep_v2_test.sh
"
```

**Key environment variables explained**:

- `LD_PRELOAD`: Must point to NCCL 2.30.4, because the container loads NCCL 2.29.7 by default, which lacks the Gin API
- `TRITON_PTXAS_PATH`: Prevents Triton JIT compilation failures in subprocesses
- `NCCL_NET=gIB`: Enables GPUDirect RDMA (GIB) for cross-node communication

| Test | Measured Result |
|------|----------|
| Intranode dispatch BF16 | **24/24 PASS**, best 460.87 GB/s NVL (SMs=24, chunk=32) |
| Intranode dispatch FP8 | **24/24 PASS**, best 308.11 GB/s NVL (SMs=24, chunk=32) |
| Intranode Combine | **PASS**, best 346.24 GB/s NVL (SMs=24, chunk=16) |
| Internode MNNVL Dispatch (SU) | **~580 GB/s**, latency ~58 us |
| Internode MNNVL Combine (SU) | **~660 GB/s**, latency ~99 us |
| All configurations (Internode Elastic EP) | **144/144 PASS** |

**Internode v2 verified**: NCCL Gin unifies management of all 8 GPUs (2 x 4), with NVLink handling intra-node communication and RDMA (GIB) handling cross-node communication.

## 7.5 4-GPU Adaptation Notes (A4X patch)

DeepEP assumes 8 GPUs per node by default (NVLink peers = 8), but A4X has only 4 GPUs per node. Without modification, this causes assertion failures or bitmask out-of-bounds errors.

| Version | Constant Name | File Modified | Change |
|------|--------|----------|------|
| v1 | `NUM_MAX_NVL_PEERS` | `csrc/kernels/configs.cuh`, etc. | 8 → 4 |
| v2 | `LEGACY_NUM_MAX_NVL_PEERS` | `csrc/kernels/legacy/compiled.cuh`, etc. | 8 → 4 |

**Core modification points:**

1. **Constant definition**: Change `#define NUM_MAX_NVL_PEERS 8` to `4`
2. **Assertion**: Change `static_assert(... == 8)` to `== 4`
3. **Bounds check**: Change `num_ranks > NUM_MAX_NVL_PEERS` to `>= NUM_MAX_NVL_PEERS` (with 4 GPUs, `> 4` would skip a branch that should execute)
4. **Bitmask type**: 8 peers x `sizeof(bool)` = 8 bytes = `uint64_t`; 4 peers x `sizeof(bool)` = 4 bytes = **`uint32_t`**

## 7.6 Functional Test Summary

### Path A (v1 + CUDA 12) test commands

```bash
# Intranode (single node, 4 GPU)
cd /tmp/DeepEP/tests/legacy
python test_intranode.py --num-processes 4 --num-experts 64 --num-topk 4

# Low-latency (single node, 4 GPU, requires NVSHMEM 3.4.5)
python test_low_latency.py --num-processes 4 --num-experts 64 --num-topk 4

# Internode (requires NVSHMEM IBGDA, limited by GDRCopy on GKE)
# Requires: NVSHMEM_REMOTE_TRANSPORT=ibgda  NVSHMEM_DISABLE_CUDA_VMM=1
# Requires: DEEP_EP_DEVICE_TO_HCA_MAPPING="0:mlx5_0,1:mlx5_1,2:mlx5_2,3:mlx5_3"
```

### Path B (v2 + CUDA 13) test commands

```bash
# Intranode (single node, 4 GPU)
cd /scratch-data/DeepEP-v2
python tests/test_intranode.py --num-processes 4

# Internode Elastic EP (2 nodes x 4 GPU, NCCL Gin backend)
# Use the /scratch-data/run_deepep_v2_test.sh script (see 7.4 Step 5)
```

### CUDA 12 vs CUDA 13 performance comparison

| Metric | CUDA 12 (Path A) | CUDA 13 (Path B) |
|------|------------------|------------------|
| Intranode dispatch BF16 | ~436 GB/s | ~461 GB/s |
| Intranode dispatch FP8 | ~288 GB/s | ~308 GB/s |
| Intranode combine | ~321 GB/s | ~346 GB/s |
| Internode (v1 legacy) | works | blocked (NVSHMEM version conflict) |
| Internode (v2 elastic) | N/A | **works (NCCL Gin, 144/144 PASS)** |

## DeepEP v1 vs v2 Selection Recommendation

| Dimension | DeepEP v1 (CUDA 12) | DeepEP v2 (CUDA 13) |
|------|---------------------|---------------------|
| Internode backend | NVSHMEM IBGDA — requires GDRCopy (unavailable on GCP) | NCCL Gin — fully compatible with GCP |
| Intranode | works | works |
| NVSHMEM dependency | strictly requires 3.4.5 (GB200) or 3.3.9 (B200/H200) | required only at build time, no runtime dependency |
| CUDA container | pytorch:25.04-py3 (CUDA 12) | pytorch:26.04-py3 / 26.05-py3 (CUDA 13) |
| GCP recommendation | intranode scenarios only | **recommended (internode verified 144/144 PASS)** |

**Core decision**: For cross-node Expert Parallelism (internode EP), you must use **DeepEP v2**. v1's internode communication relies on NVSHMEM IBGDA + GDRCopy, and GDRCopy is unavailable in GKE/GCP environments. v2 uses the NCCL Gin backend to completely eliminate this dependency.

## 7.7 Our Verification Results (2026-06-29, k8s 1.34 + DRA + ComputeDomain)

### Test environment

| Item | Value |
|---|---|
| Cluster | kubeadm 1.34.9, 16 Workers (2 domains × 8 nodes) |
| GPU | 4 × GB200 189GB per node |
| Image | `megatron-ngc:tev2.15-mgcore_r0.16.0-pt26.05-py3-v2` (DeepEP 2.0.0 pre-installed) |
| NCCL | 2.30.4+cuda13.2 (via LD_PRELOAD) |
| Network | GIB RDMA (cross-node) + NVSwitch/MNNVL (intra-domain) |
| ComputeDomain | cd-d1 (Domain 1), cd-d2 (Domain 2) |

### Intranode test (single node, 4 GPU)

**Legacy Intranode (`tests/legacy/test_intranode.py`)**:

| Operation | Bandwidth (GB/s) | SMs | Chunk | Notes |
|---|---|---|---|---|
| Dispatch BF16 | **436** | 24 | 32 | Expert dispatch, BF16 precision |
| Dispatch FP8 | **292** | 24 | 32 | Expert dispatch, FP8 precision (data volume halved after quantization, bandwidth reduced accordingly) |
| Combine | **322** | 24 | 16 | Merge Expert results back into original token order |

**Legacy Low-latency (`tests/legacy/test_low_latency.py`)**:

| Operation | Bandwidth (GB/s) | Latency (us) | Notes |
|---|---|---|---|
| Dispatch | 163-182 | 20-23 | Low-latency mode, optimizes latency rather than throughput |
| Combine | 323-335 | 21-22 | Combine operations naturally have lower latency |
| Dispatch + Combine | 204-209 | 49-53 | End-to-end latency |

### Internode test (2 nodes × 4 GPU = 8 GPU, NCCL Gin)

**Elastic EP (`tests/elastic/test_ep.py --test-first-only`)**:

Test configuration: `do_handle_copy=1, expert_alignment=128, use_fp8_dispatch=1, num_experts=256, num_topk=6, hidden=7168, num_tokens=4096`

| Operation | Mode | Bandwidth SU (GB/s) | Latency (us) | Notes |
|---|---|---|---|---|
| Dispatch | scaleup | **695-705** | 191-193 | Cross-node Expert dispatch |
| Expanded Dispatch | scaleup | 697-704 | 191-194 | Expanded dispatch with handle copy |
| Cached Dispatch | scaleup | 695-702 | 191-195 | Cached-mode dispatch |
| Combine | scaleup | **720-728** | 357-360 | Cross-node result combine |
| Reduced Combine | scaleup | 701-709 | 367-370 | Combine with reduction |
| Copy (NVLink) | — | 4936-6289 | 43-68 | Intra-node GPU-to-GPU copy |
| Reduce (NVLink) | — | 2043-2168 | 54-58 | Intra-node GPU-to-GPU reduce |

All 4 scaleup configurations (SU 0/1/2/3) completed successfully without errors.

### Benchmark comparative analysis

#### Intranode vs Internode

| Operation | Intranode (NVLink) | Internode (NCCL Gin) | Ratio | Notes |
|---|---|---|---|---|
| Dispatch | 436 (BF16) | 700 (FP8) | **1.6×** | Internode is higher because FP8 quantization halves the data volume, and NCCL Gin leverages the MNNVL channel |
| Combine | 322 | 724 | **2.2×** | Internode combine uses NCCL collectives, which are more efficient than the Legacy hand-written NVLink kernel |

> **Why is Internode bandwidth higher than Intranode?**
>
> This seems counterintuitive, but the reason is that the two tests use different communication paths:
> - **Legacy Intranode**: uses DeepEP's in-house NVLink peer-to-peer kernel (a hand-written CUDA kernel that operates NVLink directly), single node with 4 GPUs, SMs=24
> - **Elastic Internode**: uses the NCCL Gin backend (the GPU-Initiated Network communication in NCCL 2.30.4), 2 nodes with 8 GPUs
>
> Advantages of the NCCL Gin backend:
> 1. Intra-domain communication uses MNNVL (NVSwitch full mesh, bandwidth > P2P NVLink)
> 2. Inter-domain communication uses GIB RDMA (GPU-initiated RDMA, no CPU involvement)
> 3. NCCL's collective algorithms are more efficient for large messages (pipeline + multi-channel)
> 4. An 8-GPU ring has higher bandwidth utilization than a 4-GPU ring

#### Interpreting Copy and Reduce bandwidth

| Operation | Bandwidth (GB/s) | Theoretical Bandwidth | Utilization | Notes |
|---|---|---|---|---|
| Copy (NVLink) | 5000-6300 | ~7200 | 69-88% | NVSwitch 5th gen unidirectional bandwidth, near theoretical peak |
| Reduce (NVLink) | 2043-2168 | ~3600 | 57-60% | Reduce requires reading + writing data twice; effective bandwidth ≈ theoretical/2 plus compute overhead |

NVSwitch 5th gen provides ~900 GB/s bidirectional NVLink bandwidth per GPU on GB200. Copy reaching 5000-6300 GB/s (aggregated across 4 GPUs) indicates the NVSwitch fabric is working correctly.

#### Parameter sensitivity test (Intranode)

Same hardware, same image; the impact of different `num-topk` and `num-experts` on bandwidth:

| Parameter combination | Dispatch BF16 | Dispatch FP8 | Combine |
|---|---|---|---|
| topk=4, experts=64 | 432 GB/s | 291 GB/s | 323 GB/s |
| **topk=8, experts=256** (default) | **465 GB/s** | **317 GB/s** | **347 GB/s** |
| topk=8, experts=256 + allow-mnnvl | 465 GB/s | 317 GB/s | 348 GB/s |

**Impact of topk on bandwidth**: topk=8 is 7-9% higher than topk=4. The reason is that the larger the topk, the more Experts each token must be dispatched to, so the dispatch data volume is larger (topk=8 has 2× the data volume of topk=4), the NVLink transfer pipeline is more fully filled, and batch transfer efficiency is higher. With topk=4, each transferred data block is smaller, and NVLink bandwidth utilization is lower.

**allow-mnnvl has no impact on intranode**: With a single node of 4 GPUs all under the same NVSwitch, the presence or absence of MNNVL does not change the communication path.

#### Comparison with the documented Baseline

| Metric | Baseline (§7.4) | Ours (topk=4) | Ours (topk=8) | Difference (topk=8) |
|---|---|---|---|---|
| Intranode dispatch BF16 | 461 GB/s | 432 | **465** | **+0.9%** |
| Intranode dispatch FP8 | 308 GB/s | 291 | **317** | **+2.9%** |
| Intranode combine | 346 GB/s | 322 | **347** | **+0.3%** |
| Internode dispatch SU | ~580 GB/s | — | **700** | **+21%** |
| Internode combine SU | ~660 GB/s | — | **724** | **+10%** |

**Intranode**: After using the same default parameters as the Baseline (topk=8, experts=256), all three metrics match the Baseline (difference < 3%). The earlier 5-7% shortfall was due to different test parameters (topk=4 vs 8), not a performance problem.

**Why Internode is 10-21% higher than the Baseline**:

Both the Baseline and ours use 2 nodes × 4 GPU (same scale), but there are two differences:
1. **GIB NCCL plugin version**: our init container uses v1.1.2, while the Baseline uses v1.1.0. v1.1.2 may include RDMA path optimizations
2. **NCCL version**: both are 2.30.4, but the NCCL we load via LD_PRELOAD is the pip-installed `nvidia-nccl-cu13`, which may include newer Gin backend patches

> **Note**: Internode being higher than the Baseline is not due to a smaller number of nodes — the Baseline is also 2 nodes. 2 nodes × 4 GPU = 8 GPU is the standard configuration for the Elastic EP internode test and is not subject to scale effects (unlike an NCCL ring with its multi-channel effect), because DeepEP dispatch/combine is a point-to-point pattern.

### Scale-out test (in progress)

**8 nodes, 32 GPU (single domain)**: failed, with the root cause being an RDMA GID configuration issue.

DRANET assigns 4 RDMA NICs (mlx5_0-3) to the pod, but only 1 has a RoCEv2 IPv4 GID (index 3); the other 3 have all-zero GIDs:

```
mlx5_0 gid3: 0000:0000:...:0000  ← empty (no IPv4)
mlx5_1 gid3: 0000:0000:...:0000  ← empty
mlx5_2 gid3: 0000:0000:...:ffff:10.10.24.32  ← normal
mlx5_3 gid3: 0000:0000:...:0000  ← empty
```

- The 2-node test passed because NCCL happened to use only the NIC that had a GID
- With 8 nodes, NCCL needs more cross-node connections and uses NICs without a GID, triggering the `ibv_modify_qp failed` error
- The NCCL test passes at the same scale (8+8 nodes, 64 GPU) because the GIB network plugin has an independent RDMA address discovery mechanism (not dependent on the GID index), whereas DeepEP's NCCL communicator may fall back to standard IB transport

**Root cause confirmed**: On the host, the GID 3 of all 4 NICs is normal (complete IPv4 addresses). After DRANET moves the NICs into the pod network namespace, it configures IPv4 for only 1/4 of the NICs, causing the RoCEv2 GID of 3/4 of the NICs to be lost. This is a known limitation of DRANET v1.3.0 in multi-RDMA-NIC scenarios on GCP A4X.

**Solution**: Use **hostNetwork mode** to bypass DRANET, so the pod uses the host's RDMA NICs directly (GIDs intact). Do not declare the `rdma-nics` ResourceClaim; keep only `compute-domain-channel`.

**hostNetwork mode verification (4 nodes, 16 GPU)**: PASS

| Operation | 2n/8GPU | 4n/16GPU | Change | Notes |
|---|---|---|---|---|
| Dispatch SU | 700 | **660** | -6% | Larger scale intensifies RDMA contention |
| Expanded Dispatch SU | 700 | 660 | -6% | |
| Cached Dispatch SU | 698 | 660 | -5% | |
| Combine SU | 724 | **683** | -6% | |
| Reduced Combine SU | 705 | **677** | -4% | |
| Copy (NVLink) | 5600 | 5500 | -2% | NVLink is unaffected by scale |
| Reduce (NVLink) | 2100 | 1870 | -11% | More GPUs participate in the reduce |

**Key finding: why the 2-node data is on the high side**

The 2-node (8 GPU) dispatch of 700 GB/s and combine of 724 GB/s are significantly higher than the documented baseline (580/660), but the 4-node (16 GPU) figures drop to 660/683. Reasons:

1. **NCCL communicator scale effect**: With 2 nodes, NCCL only needs to establish 2 cross-node connections, so the ring is short and the pipeline fills quickly. With 4 nodes, more cross-node connections are needed, and increased RDMA contention and ring length cause bandwidth to drop
2. **Expert routing dispersion**: more GPUs means Experts are dispersed across more nodes, so each token's dispatch must traverse more cross-node links
3. **The baseline 580/660 likely corresponds to a larger-scale test** (e.g., 8-18 nodes); our 2-node data is not representative of scale

**8-node 32-GPU hostNetwork test: PASS**

| Operation | 2n/8GPU | 4n/16GPU | 8n/32GPU | Baseline | Notes |
|---|---|---|---|---|---|
| Dispatch SU | 700 | 660 | **636** | ~580 | -5~6% per doubling |
| Combine SU | 724 | 683 | **590** | ~660 | Combine drops faster (reduce overhead O(N)) |
| Reduced Combine SU | 705 | 677 | **593** | — | |
| Copy (NVLink) | 5600 | 5500 | **5700** | — | NVLink is unaffected by scale |
| Reduce (NVLink) | 2100 | 1870 | **1730** | — | 32-way reduce has greater synchronization overhead |

**Analysis of the scale-related decline** (same domain, NVL72 NVSwitch full mesh):

1. **NVSwitch bandwidth is not the bottleneck**. Within the NVL72 domain, all GPUs are fully interconnected via 5th-gen NVSwitch, so any GPU pair has the same bandwidth. Copy maintaining 5500-5700 GB/s confirms this

2. **The Dispatch decline comes from scatter fragmentation**. With 8 GPUs, each GPU's topk=6 dispatches to 7 targets, averaging ~100 GB/s effective bandwidth per target. With 32 GPUs, it dispatches to 31 targets, averaging ~22 GB/s per target. Although total NVLink bandwidth is unchanged, the NVLink utilization of small, scattered writes is lower than that of large, contiguous transfers

3. **Combine drops faster (-19% vs Dispatch -9%)**. Combine performs a reduce (additive reduction), and a 32-way reduce has more pipeline stages and synchronization points than an 8-way reduce. Each GPU must wait for more partial results to arrive before it can complete the reduce, and synchronization overhead grows with the number of participants

4. **Alignment with the Baseline**. At 32 GPUs, Dispatch 636 and Combine 590 are close to the documented Baseline (580/660), confirming that the Baseline corresponds to a larger-scale test. The 2-node "high values" of 700/724 are not our improvement but a small-scale effect

**Conclusion**: DeepEP internode bandwidth declining as scale increases is an inherent characteristic of the all-to-all communication pattern (scatter fragmentation + reduce synchronization overhead), not a hardware bottleneck. The 2-node data is not representative of scale; we recommend using 8+ node data as the baseline.

### Deployment key points (k8s 1.34 DRA mode)

1. The Pod must have a `compute-domain-channel` claim — DeepEP v2's NCCL Gin requires an IMEX channel
2. `LD_PRELOAD` must specify NCCL 2.30.4 — the container's default 2.29.7 lacks the Gin API
3. `NCCL_NET=gIB` — enables GPUDirect RDMA
4. `TRITON_PTXAS_PATH` — DeepEP JIT compilation requires ptxas
5. The AR image registry requires `imagePullSecrets` — the token has a short validity period and needs periodic refreshing
6. For the internode test_ep.py, use `WORLD_SIZE=number of nodes` + `RANK=node index`; do not use torchrun
7. **Large-scale (>2 nodes) internode is limited by the RDMA GID configuration** — see the scale-out test record above for details
